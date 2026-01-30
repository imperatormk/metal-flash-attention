//
//  AttentionKernel+OuterProduct.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// M x K x N
// parallelization x head x traversal

struct AttentionOuterProductDescriptor {
  var A: AttentionOperand?
  var B: AttentionOperand?
  var C: AttentionOperand?
}

extension AttentionKernel {
  func outerProduct(
    descriptor outerProductDesc: AttentionOuterProductDescriptor
  ) -> String {
    guard let A = outerProductDesc.A,
          let B = outerProductDesc.B,
          let C = outerProductDesc.C else {
      fatalError("Descriptor was incomplete.")
    }
    
    // MARK: - Initialize
    
    func allocateAccumulator() -> String {
      """
      
      simdgroup_matrix_storage<\(registerName(C))> \
      \(C)_sram[\(blockDimensions.traversal) / 8];
      
      """
    }
    
    func initializeAccumulator() -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
        auto \(C) = \(C)_sram + c / 8;
        *\(C) = simdgroup_matrix_storage<\(registerName(C))>(0);
      }
      
      """
    }
    
    func allocateLHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard !cached(A) else {
        return ""
      }
      return """
      
      simdgroup_matrix_storage<\(registerName(A))> \
      \(A)_sram[\(descriptor.registerSize) / 8];
      
      """
    }
    
    // MARK: - Load LHS
    
    func declareLHSLocation(
      descriptor: LoopIterationDescriptor
    ) -> String {
      switch descriptor.addressSpaceLHS! {
      case .device:
        return """
        
        uint2 \(A)_src_offset(
          morton_offset.x + d_outer,
          \(clampedParallelizationThreadOffset));
        auto \(A)_src = simdgroup_matrix_storage<\(memoryName(A))>
        ::apply_offset(
          \(A), \(leadingDimension(A)),
          \(A)_src_offset, \(transposed(A)));
        
        """
      case .threadgroup:
        return """
        
        ushort2 \(A)_block_offset(
          morton_offset.x, 
          morton_offset.y + sidx * 8);
        auto \(A)_src = (threadgroup \(memoryName(A))*)(threadgroup_block);
        \(A)_src = simdgroup_matrix_storage<\(memoryName(A))>
        ::apply_offset(
          \(A)_src, \(leadingBlockDimension(A)),
          \(A)_block_offset, \(transposed(A)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        """
      }
    }
    
    func asyncLoadLHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(A)_offset(d_outer, \(parallelizationGroupOffset));
        auto src = simdgroup_matrix_storage<\(memoryName(A))>
        ::apply_offset(
          \(A), \(leadingDimension(A)),
          \(A)_offset, \(transposed(A)));
        auto dst = (threadgroup \(memoryName(A))*)(threadgroup_block);
        
        ushort D_src_dimension = min(
          ushort(\(blockDimensions.head)),
          ushort(\(headDimension) - d_outer));
        ushort D_dst_dimension = \(descriptor.registerSize);
        ushort R_dimension = min(
          uint(\(blockDimensions.parallelization)),
          uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
        ushort2 tile_src(D_src_dimension, R_dimension);
        ushort2 tile_dst(D_dst_dimension, R_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingBlockDimension(A)), tile_dst,
          src, \(leadingDimension(A)), tile_src, \(transposed(A)));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    func loadLHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard !cached(A) else {
        return ""
      }
      
      switch descriptor.addressSpaceLHS! {
      case .device:
        return """
        
        \(declareLHSLocation(descriptor: descriptor))
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          ushort2 \(A)_origin(d, 0);
          \(A)_sram[d / 8].\(loadFunction(A))(
            \(A)_src, \(leadingDimension(A)),
            \(A)_origin, \(transposed(A)));
        }
        
        """
      case .threadgroup:
        return """
        
        \(asyncLoadLHS(descriptor: descriptor))
        \(declareLHSLocation(descriptor: descriptor))
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          ushort2 \(A)_origin(d, 0);
          \(A)_sram[d / 8].\(loadFunction(A))(
            \(A)_src, \(leadingBlockDimension(A)),
            \(A)_origin, \(transposed(A)));
        }
        
        """
      }
    }
    
    // MARK: - Load RHS
    
    func leadingDimensionRHS(
      _ descriptor: LoopIterationDescriptor
    ) -> String {
      switch descriptor.addressSpaceRHS! {
      case .device:
        return leadingDimension(B)
      case .threadgroup:
        return "\(leadingBlockDimension(B))"
      }
    }
    
    func declareRHSLocation(
      descriptor: LoopIterationDescriptor
    ) -> String {
      // For quantized types, we can't use simdgroup_matrix_storage::apply_offset
      // because it's only defined for half/float/bfloat. Compute offset manually.
      let isQuant = isQuantizedLoad(B)

      switch descriptor.addressSpaceRHS! {
      case .device:
        if isQuant {
          // For quantized types, save the outer loop offset before inner loop shadows it
          // The inner loop uses local variable 'c' which shadows the outer loop's 'c'
          return """

          // Save outer loop offset for quantized loading (inner loop will shadow 'c')
          uint \(B)_seq_base = \(traversalOffset);

          """
        } else {
          return """

          uint2 \(B)_src_offset(
            morton_offset.y + d_outer,
            morton_offset.x + \(traversalOffset));
          auto \(B)_src = simdgroup_matrix_storage<\(memoryName(B))>
          ::apply_offset(
            \(B), \(leadingDimension(B)),
            \(B)_src_offset, \(transposed(B)));

          """
        }
      case .threadgroup:
        if isQuant {
          // For quantized in threadgroup path, set up pointer to threadgroup memory
          // Data was already copied in loadRHS, now just set up pointer for inner loop
          return """

          threadgroup uchar* \(B)_tg_base = (threadgroup uchar*)(threadgroup_block);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          """
        } else {
          return """

          ushort2 \(B)_block_offset(
            morton_offset.x,
            morton_offset.y);
          auto \(B)_src = (threadgroup \(memoryName(B))*)(threadgroup_block);
          \(B)_src = simdgroup_matrix_storage<\(memoryName(B))>
          ::apply_offset(
            \(B)_src, \(leadingBlockDimension(B)),
            \(B)_block_offset, \(!transposed(B)));
          threadgroup_barrier(mem_flags::mem_threadgroup);

          """
        }
      }
    }
    
    func loadRHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      switch descriptor.addressSpaceRHS! {
      case .device:
        return declareRHSLocation(descriptor: descriptor)
      case .threadgroup:
        let isQuant = isQuantizedLoad(B)
        if isQuant {
          // For quantized types, manually copy to threadgroup memory
          // Can't use async_copy/apply_offset since they're not specialized for uchar
          //
          // We use ALL threads in simdgroup 0 to parallelize the copy
          // Each thread copies a subset of the data
          return """

          threadgroup_barrier(mem_flags::mem_threadgroup);
          // Manual copy of quantized K to threadgroup memory using all threads in sidx==0
          if (sidx == 0) {
            device const uchar* src = \(B);
            threadgroup uchar* dst = (threadgroup uchar*)(threadgroup_block);

            ushort D_dimension = min(
              ushort(\(blockDimensions.head)),
              ushort(\(headDimension) - d_outer));
            ushort C_dimension = min(
              uint(\(blockDimensions.traversal)),
              uint(\(traversalDimension) - \(traversalOffset)));

            // Copy tile: K[traversalOffset..+C_dimension, d_outer..+D_dimension]
            // Storage is K[seq, head], so address = seq * headDim + head
            // Parallelize across all 32 threads in the simdgroup
            ushort total_elements = C_dimension * D_dimension;
            for (ushort i = lane_id; i < total_elements; i += 32) {
              ushort seq = i / D_dimension;
              ushort head = i % D_dimension;
              uint src_addr = (\(traversalOffset) + seq) * \(leadingDimension(B)) + (d_outer + head);
              uint dst_addr = seq * \(leadingBlockDimension(B)) + head;
              dst[dst_addr] = src[src_addr];
            }
          }

          \(declareRHSLocation(descriptor: descriptor))

          """
        }
        return """

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sidx == 0) {
          uint2 \(B)_offset(d_outer, \(traversalOffset));
          auto src = simdgroup_matrix_storage<\(memoryName(B))>
          ::apply_offset(
            \(B), \(leadingDimension(B)),
            \(B)_offset, \(transposed(B)));
          auto dst = (threadgroup \(memoryName(B))*)(threadgroup_block);

          ushort D_src_dimension = min(
            ushort(\(blockDimensions.head)),
            ushort(\(headDimension) - d_outer));
          ushort D_dst_dimension = \(descriptor.registerSize);
          ushort C_src_dimension = min(
            uint(\(blockDimensions.traversal)),
            uint(\(traversalDimension) - \(traversalOffset)));
          ushort C_dst_dimension = max(
            ushort(\(paddedTraversalEdge)),
            ushort(C_src_dimension));
          ushort2 tile_src(D_src_dimension, C_src_dimension);
          ushort2 tile_dst(D_dst_dimension, C_dst_dimension);

          simdgroup_event event;
          event.async_copy(
            dst, \(leadingBlockDimension(B)), tile_dst,
            src, \(leadingDimension(B)), tile_src, \(transposed(B)));
          simdgroup_event::wait(1, &event);
        }

        \(declareRHSLocation(descriptor: descriptor))

        """
      }
    }
    
    // MARK: - Inner Loop

    func innerLoopTraversal(
      traversalStart: String,
      traversalEnd: String,
      descriptor: LoopIterationDescriptor
    ) -> String {
      // Check if B (RHS) is quantized and needs special loading
      if isQuantizedLoad(B) {
        return innerLoopTraversalQuantized(
          traversalStart: traversalStart,
          traversalEnd: traversalEnd,
          descriptor: descriptor
        )
      }

      // Standard non-quantized path
      return """

      #pragma clang loop unroll(full)
      for (ushort c = \(traversalStart); c < \(traversalEnd); c += 8) {
        // Load the RHS from memory.
        ushort2 \(B)_origin(c, d);
        simdgroup_matrix_storage<\(registerName(B))> \(B);
        \(B).\(loadFunction(B))(
          \(B)_src, \(leadingDimensionRHS(descriptor)),
          \(B)_origin, \(!transposed(B)));

        // Issue one SIMD matmul instruction.
        \(C)_sram[c / 8].multiply(
          \(A)_sram[(\(descriptor.registerOffset) + d) / 8],
          \(B), \(descriptor.accumulateConditional));
      }

      """
    }

    /// Quantized inner loop with on-the-fly dequantization
    func innerLoopTraversalQuantized(
      traversalStart: String,
      traversalEnd: String,
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard let quantizedKV = self.quantizedKV else {
        fatalError("Called quantized path without quantizedKV set")
      }

      // Generate the dequantization function name
      let dequantFunc: String
      switch quantizedKV {
      case .FP8_E4M3:
        dequantFunc = "fp8_e4m3_to_half"
      case .FP8_E5M2:
        dequantFunc = "fp8_e5m2_to_half"
      case .INT8:
        dequantFunc = "uint8_to_half_signed"
      default:
        // NF4 needs special handling (2 values per byte)
        dequantFunc = "fp8_e4m3_to_half"  // Fallback
      }

      // Scale buffer name
      let scaleBuffer = "\(B)_scale"

      // IMPORTANT: The standard load uses !transposed(B), not transposed(B)
      // If K is stored non-transposed (transposed(K) = false), we load with transpose=true
      // This means the memory layout interpretation is INVERTED from the transpose flag
      let loadTranspose = !transposed(B)

      // Determine whether we're loading from device or threadgroup memory
      let useThreadgroup = descriptor.addressSpaceRHS == .threadgroup

      // For threadgroup: data is laid out as [seq, head] in threadgroup memory
      // with leading dimension = leadingBlockDimension(B)
      let leadingDim = useThreadgroup ? "\(leadingBlockDimension(B))" : "\(leadingDimension(B))"

      // Base pointer depends on address space
      let quantBaseDecl: String
      if useThreadgroup {
        quantBaseDecl = "threadgroup uchar* quant_base = \(B)_tg_base;"
      } else {
        quantBaseDecl = "device const uchar* quant_base = \(B);  // K is device const uchar*"
      }

      // For threadgroup: seq offset is relative to tile (starts at 0)
      // For device: seq offset is absolute (needs K_seq_base)
      let seqBaseExpr = useThreadgroup ? "0" : "\(B)_seq_base"
      // For threadgroup: head offset is relative to tile (starts at 0)
      // For device: head offset is absolute (needs d_outer)
      let headBaseExpr = useThreadgroup ? "d" : "(d_outer + d)"

      return """

      #pragma clang loop unroll(full)
      for (ushort c = \(traversalStart); c < \(traversalEnd); c += 8) {
        simdgroup_matrix_storage<half> \(B)_tile;

        // Load and dequantize 2 elements per thread
        {
          \(quantBaseDecl)
          float scale_val = \(scaleBuffer)[0];

          // For simdgroup_matrix_storage 8x8 tile:
          // - morton_offset.x = column position in tile (0-7, step 2)
          // - morton_offset.y = row position in tile (0-7)
          // - Each thread loads 2 adjacent columns in the same row
          //
          // With loadTranspose=true (K not transposed in memory):
          // - Tile columns map to memory rows (seq dimension)
          // - Tile rows map to memory columns (head dimension)
          // - So 2 adjacent elements are at adjacent seq positions, same head position
          //
          // K[seq, head] storage: address = seq * leadingDim + head

          uint addr0, addr1;
          if (\(loadTranspose)) {
            // loadTranspose=true: tile columns = memory rows (seq), tile rows = memory cols (head)
            // seq_pos = base + inner_c + morton_offset.x
            // head_pos = head_base + morton_offset.y
            uint seq_pos0 = \(seqBaseExpr) + c + morton_offset.x;
            uint seq_pos1 = seq_pos0 + 1;
            uint head_pos = \(headBaseExpr) + morton_offset.y;
            addr0 = seq_pos0 * \(leadingDim) + head_pos;
            addr1 = seq_pos1 * \(leadingDim) + head_pos;
          } else {
            // loadTranspose=false: tile columns = memory cols (head), tile rows = memory rows (seq)
            // head_pos = base + inner_c + morton_offset.x
            // seq_pos = head_base + morton_offset.y
            uint head_pos0 = \(seqBaseExpr) + c + morton_offset.x;
            uint head_pos1 = head_pos0 + 1;
            uint seq_pos = \(headBaseExpr) + morton_offset.y;
            addr0 = seq_pos * \(leadingDim) + head_pos0;
            addr1 = seq_pos * \(leadingDim) + head_pos1;
          }

          uchar quant_val0 = quant_base[addr0];
          uchar quant_val1 = quant_base[addr1];

          half dequant0 = \(dequantFunc)(quant_val0, scale_val);
          half dequant1 = \(dequantFunc)(quant_val1, scale_val);

          // Store into simdgroup matrix storage
          ((thread half*)\(B)_tile.thread_elements())[0] = dequant0;
          ((thread half*)\(B)_tile.thread_elements())[1] = dequant1;
        }

        // Issue one SIMD matmul instruction.
        \(C)_sram[c / 8].multiply(
          \(A)_sram[(\(descriptor.registerOffset) + d) / 8],
          \(B)_tile, \(descriptor.accumulateConditional));
      }

      """
    }
    
    func innerLoopHead(
      descriptor: LoopIterationDescriptor
    ) -> String {
      if descriptor.addressSpaceLHS! == .device ||
          descriptor.addressSpaceRHS! == .device {
        return """
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          \(innerLoopTraversal(
              traversalStart: "0",
              traversalEnd: "\(blockDimensions.traversal)",
              descriptor: descriptor))
        }
        
        """
      } else {
        return """
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          \(innerLoopTraversal(
              traversalStart: "0",
              traversalEnd: paddedTraversalEdge,
              descriptor: descriptor))
          if (\(traversalOffset) + \(blockDimensions.traversal)
              < \(traversalDimension)) {
            \(innerLoopTraversal(
                traversalStart: paddedTraversalEdge,
                traversalEnd: "\(blockDimensions.traversal)",
                descriptor: descriptor))
          }
        }
        
        """
      }
    }
    
    // MARK: - Outer Loop
    
    struct LoopIterationDescriptor {
      // Whether to accumulate in the SIMD matmul.
      var accumulateConditional: String = ""
      var addressSpaceLHS: MTLAddressSpace?
      var addressSpaceRHS: MTLAddressSpace?
      var registerOffset: String = ""
      var registerSize: UInt16 = .zero
    }
    
    func loopIteration(
      descriptor: LoopIterationDescriptor
    ) -> String {
      return """
      
      \(allocateLHS(descriptor: descriptor))
      \(loadLHS(descriptor: descriptor))
      \(loadRHS(descriptor: descriptor))
      \(innerLoopHead(descriptor: descriptor))
      
      """
    }
    
    func gatedLoopIteration(
      descriptor: LoopIterationDescriptor
    ) -> String {
      var descriptorThreadgroup = descriptor
      descriptorThreadgroup.addressSpaceLHS = .threadgroup
      descriptorThreadgroup.addressSpaceRHS = .threadgroup
      if preferAsyncCache && preferAsyncLoad {
        return loopIteration(descriptor: descriptorThreadgroup)
      }
      
      var descriptorDevice = descriptor
      if preferAsyncCache {
        descriptorDevice.addressSpaceLHS = .threadgroup
      } else {
        descriptorDevice.addressSpaceLHS = .device
      }
      if preferAsyncLoad {
        descriptorDevice.addressSpaceRHS = .threadgroup
      } else {
        descriptorDevice.addressSpaceRHS = .device
      }
      
      let blockDim = blockDimensions.traversal
      let condition = """
      (
        (\(traversalDimension) % \(blockDim) == 0) ||
        (\(traversalOffset) + \(blockDim) <= \(traversalDimension))
      ) && (
        (\(headDimension) % 8 == 0) ||
        (d_outer + \(descriptor.registerSize) <= \(headDimension))
      )
      """
      
      return """
      
      if (\(condition)) {
        \(loopIteration(descriptor: descriptorDevice))
      } else {
        \(loopIteration(descriptor: descriptorThreadgroup))
      }
      
      """
    }
    
    // MARK: - Top Level Specification
    
    func loopEnd() -> UInt16 {
      paddedHeadDimension
    }
    
    func loopEndFloor() -> UInt16 {
      loopEnd() - loopEnd() % blockDimensions.head
    }
    
    func unrollStatement() -> String {
      if cached(A) {
        return "#pragma clang loop unroll(full)"
      } else {
        return "#pragma clang loop unroll(disable)"
      }
    }
    
    func initializeStatement() -> String {
      if cached(A) {
        // Zero-initialize during the multiply-accumulate loop.
        return ""
      } else {
        // Zero-initialize beforehand.
        return initializeAccumulator()
      }
    }
    
    func accumulateConditional() -> String {
      if cached(A) {
        return "((d_outer > 0) || (d > 0))"
      } else {
        // The accumulator is already initialized.
        return "true"
      }
    }
    
    func registerOffset() -> String {
      if cached(A) {
        return "d_outer"
      } else {
        return "0"
      }
    }
    
    func firstIterations() -> String {
      var descriptor = LoopIterationDescriptor()
      descriptor.accumulateConditional = accumulateConditional()
      descriptor.registerOffset = registerOffset()
      descriptor.registerSize = blockDimensions.head
      
      return """
      
      \(unrollStatement())
      for (
        ushort d_outer = 0;
        d_outer < \(loopEndFloor());
        d_outer += \(blockDimensions.head)
      ) {
        \(gatedLoopIteration(descriptor: descriptor))
      }
      
      """
    }
    
    func lastIteration() -> String {
      var descriptor = LoopIterationDescriptor()
      descriptor.accumulateConditional = accumulateConditional()
      descriptor.registerOffset = registerOffset()
      descriptor.registerSize = paddedHeadEdge
      
      return """
      
      if (\(loopEndFloor() < loopEnd())) {
        ushort d_outer = \(loopEndFloor());
        \(gatedLoopIteration(descriptor: descriptor))
      }
      
      """
    }
    
    // Collect all of the statements into one string.
    return """
    
    \(allocateAccumulator())
    \(initializeStatement())
    
    \(firstIterations())
    \(lastIteration())
    
    """
  }
}
