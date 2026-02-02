//
//  AttentionKernel+Accumulate.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// M x K x N
// parallelization x traversal x head

struct AttentionAccumulateDescriptor {
  var A: AttentionOperand?
  var B: AttentionOperand?
  var C: AttentionOperand?
  
  /// Optional. Factor to multiply every time the accumulator is loaded.
  var everyIterationScale: String?
  
  /// Optional. Factor to multiply, on the last iteration of the K dimension.
  var lastIterationScale: String?
}

extension AttentionKernel {
  func accumulate(
    descriptor accumulateDesc: AttentionAccumulateDescriptor
  ) -> String {
    guard let A = accumulateDesc.A,
          let B = accumulateDesc.B,
          let C = accumulateDesc.C else {
      fatalError("Descriptor was incomplete.")
    }
    
    // MARK: - Initialize
    
    func allocateAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard !cached(C) else {
        return ""
      }
      return """
      
      simdgroup_matrix_storage<\(registerName(C))> \
      \(C)_sram[\(descriptor.registerSize) / 8];
      
      """
    }
    
    func initializeAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        auto \(C) = \(C)_sram + (\(descriptor.registerOffset) + d) / 8;
        *\(C) = simdgroup_matrix_storage<\(registerName(C))>(0);
      }
      
      """
    }
    
    func scaleAccumulator(
      by scale: String?,
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard let scale else {
        return ""
      }
      return """
      
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        auto \(C) = \(C)_sram + (\(descriptor.registerOffset) + d) / 8;
        *(\(C)->thread_elements()) *= \(scale);
      }
      
      """
    }
    
    // MARK: - Load/Store Accumulator
    
    func declareAccumulatorLocation(
      descriptor: LoopIterationDescriptor
    ) -> String {
      switch descriptor.addressSpaceLHS! {
      case .device:
        return """
        
        uint2 \(C)_src_offset(
          morton_offset.x + d_outer,
          \(clampedParallelizationThreadOffset));
        auto \(C)_src = simdgroup_matrix_storage<\(memoryName(C))>
        ::apply_offset(
          \(C), \(leadingDimension(C)),
          \(C)_src_offset, \(transposed(C)));
        
        """
      case .threadgroup:
        return """
        
        ushort2 \(C)_block_offset(
          morton_offset.x,
          morton_offset.y + sidx * 8);
        auto \(C)_src = (threadgroup \(memoryName(C))*)(threadgroup_block);
        \(C)_src = simdgroup_matrix_storage<\(memoryName(C))>
        ::apply_offset(
          \(C)_src, \(leadingBlockDimension(C)),
          \(C)_block_offset, \(transposed(C)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        """
      }
    }
    
    func asyncLoadAccumulator() -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(C)_offset(d_outer, \(parallelizationGroupOffset));
        auto src = simdgroup_matrix_storage<\(memoryName(C))>
        ::apply_offset(
          \(C), \(leadingDimension(C)),
          \(C)_offset, \(transposed(C)));
        auto dst = (threadgroup \(memoryName(C))*)(threadgroup_block);
        
        ushort D_dimension = min(
          ushort(\(blockDimensions.head)),
          ushort(\(headDimension) - d_outer));
        ushort R_dimension = min(
          uint(\(blockDimensions.parallelization)),
          uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
        ushort2 tile(D_dimension, R_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingBlockDimension(C)), tile,
          src, \(leadingDimension(C)), tile, \(transposed(C)));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    func asyncStoreAccumulator() -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(C)_offset(d_outer, \(parallelizationGroupOffset));
        auto src = (threadgroup \(memoryName(C))*)(threadgroup_block);
        auto dst = simdgroup_matrix_storage<\(memoryName(C))>
        ::apply_offset(
          \(C), \(leadingDimension(C)),
          \(C)_offset, \(transposed(C)));
        
        ushort D_dimension = min(
          ushort(\(blockDimensions.head)),
          ushort(\(headDimension) - d_outer));
        ushort R_dimension = min(
          uint(\(blockDimensions.parallelization)),
          uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
        ushort2 tile(D_dimension, R_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingDimension(C)), tile,
          src, \(leadingBlockDimension(C)), tile, \(transposed(C)));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    func loadAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
      switch descriptor.addressSpaceLHS! {
      case .device:
        return """
        
        \(declareAccumulatorLocation(descriptor: descriptor))
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          ushort2 \(C)_origin(d, 0);
          \(C)_sram[d / 8].\(loadFunction(C))(
            \(C)_src, \(leadingDimension(C)),
            \(C)_origin, \(transposed(C)));
        }
        
        """
      case .threadgroup:
        return """
        
        \(asyncLoadAccumulator())
        \(declareAccumulatorLocation(descriptor: descriptor))
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          ushort2 \(C)_origin(d, 0);
          \(C)_sram[d / 8].\(loadFunction(C))(
            \(C)_src, \(leadingBlockDimension(C)), 
            \(C)_origin, \(transposed(C)));
        }
        
        """
      }
    }
    
    func storeAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
      switch descriptor.addressSpaceLHS! {
      case .device:
        return """
        
        \(declareAccumulatorLocation(descriptor: descriptor))
        
        if (\(unsafeParallelizationThreadOffset) < \(parallelizationDimension)) {
          #pragma clang loop unroll(full)
          for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
            ushort2 \(C)_origin(d, 0);
            \(C)_sram[d / 8].\(storeFunction(C))(
              \(C)_src, \(leadingDimension(C)),
              \(C)_origin, \(transposed(C)));
          }
        }
        
        """
      case .threadgroup:
        return """
        
        \(declareAccumulatorLocation(descriptor: descriptor))
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          ushort2 \(C)_origin(d, 0);
          \(C)_sram[d / 8].\(storeFunction(C))(
            \(C)_src, \(leadingBlockDimension(C)),
            \(C)_origin, \(transposed(C)));
        }
        
        \(asyncStoreAccumulator())
        
        """
      }
    }
    
    func cacheAccumulator(
      descriptor: LoopIterationDescriptor,
      type: CachingOperationType
    ) -> String {
      guard !cached(C) else {
        return ""
      }
      
      if type == .load {
        return loadAccumulator(descriptor: descriptor)
      } else {
        return storeAccumulator(descriptor: descriptor)
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
        }
        return """

        uint2 \(B)_src_offset(
          morton_offset.x + d_outer,
          morton_offset.y + \(traversalOffset));
        auto \(B)_src = simdgroup_matrix_storage<\(memoryName(B))>
        ::apply_offset(
          \(B), \(leadingDimension(B)),
          \(B)_src_offset, \(transposed(B)));

        """
      case .threadgroup:
        if isQuant {
          // For quantized in threadgroup path, set up pointer to threadgroup memory
          // Data was already copied in loadRHS, now just set up pointer for inner loop
          return """

          threadgroup uchar* \(B)_tg_base = (threadgroup uchar*)(threadgroup_block);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          """
        }
        return """

        ushort2 \(B)_block_offset(
          morton_offset.x,
          morton_offset.y);
        auto \(B)_src = (threadgroup \(memoryName(B))*)(threadgroup_block);
        \(B)_src = simdgroup_matrix_storage<\(memoryName(B))>
        ::apply_offset(
          \(B)_src, \(leadingBlockDimension(B)),
          \(B)_block_offset, \(transposed(B)));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        """
      }
    }
    
    func loadRHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      let isQuant = isQuantizedLoad(B)

      switch descriptor.addressSpaceRHS! {
      case .device:
        return declareRHSLocation(descriptor: descriptor)
      case .threadgroup:
        if isQuant {
          // For quantized types, manually copy to threadgroup memory
          // Can't use async_copy/apply_offset since they're not specialized for uchar
          //
          // NF4 is special: 2 values packed per byte along head dim
          // So memory stride is D/2, not D
          let isNF4 = (quantizedKV == .NF4)
          let packedHeadDim = isNF4 ? "(\(headDimension) / 2)" : "\(headDimension)"
          let packedBlockHead = isNF4 ? "(\(blockDimensions.head) / 2)" : "\(blockDimensions.head)"

          // We use ALL threads in simdgroup 0 to parallelize the copy
          return """

          threadgroup_barrier(mem_flags::mem_threadgroup);
          // Manual copy of quantized V to threadgroup memory using all threads in sidx==0
          if (sidx == 0) {
            device const uchar* src = \(B);
            threadgroup uchar* dst = (threadgroup uchar*)(threadgroup_block);

            // For NF4: D_dimension is in packed bytes (D/2), not logical elements
            ushort D_dimension = min(
              ushort(\(packedBlockHead)),
              ushort(\(packedHeadDim) - d_outer / 2));
            ushort C_dimension = min(
              uint(\(blockDimensions.traversal)),
              uint(\(traversalDimension) - \(traversalOffset)));

            // Copy tile: V[traversalOffset..+C_dimension, d_outer..+D_dimension]
            // For NF4: memory stride is D/2 (packed bytes)
            // For FP8/INT8: memory stride is D (1 byte per element)
            uint src_stride = \(packedHeadDim);
            ushort dst_stride = \(packedBlockHead);

            // Parallelize across all 32 threads in the simdgroup
            ushort total_bytes = C_dimension * D_dimension;
            for (ushort i = lane_id; i < total_bytes; i += 32) {
              ushort seq = i / D_dimension;
              ushort byte_col = i % D_dimension;
              // For NF4: d_outer is in logical head units, convert to packed byte offset
              uint src_byte_col = \(isNF4 ? "(d_outer / 2) + byte_col" : "d_outer + byte_col");
              uint src_addr = (\(traversalOffset) + seq) * src_stride + src_byte_col;
              uint dst_addr = seq * dst_stride + byte_col;
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

          ushort D_dimension = min(
            ushort(\(blockDimensions.head)),
            ushort(\(headDimension) - d_outer));
          ushort C_src_dimension = min(
            uint(\(blockDimensions.traversal)),
            uint(\(traversalDimension) - \(traversalOffset)));
          ushort C_dst_dimension = max(
            ushort(\(paddedTraversalEdge)),
            ushort(C_src_dimension));
          ushort2 tile_src(D_dimension, C_src_dimension);
          ushort2 tile_dst(D_dimension, C_dst_dimension);

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

    func innerLoopHead(
      descriptor: LoopIterationDescriptor
    ) -> String {
      // Check if B (RHS) is quantized and needs special loading
      if isQuantizedLoad(B) {
        return innerLoopHeadQuantized(descriptor: descriptor)
      }

      return """

      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        // Load the RHS from memory.
        ushort2 \(B)_origin(d, c);
        simdgroup_matrix_storage<\(registerName(B))> \(B);
        \(B).\(loadFunction(B))(
          \(B)_src, \(leadingDimensionRHS(descriptor)),
          \(B)_origin, \(transposed(B)));

        // Issue one SIMD matmul instruction.
        \(C)_sram[(\(descriptor.registerOffset) + d) / 8].multiply(
          \(A)_sram[c / 8], \(B), /*accumulate=*/true);
      }

      """
    }

    /// Quantized inner loop head with on-the-fly dequantization
    func innerLoopHeadQuantized(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard let quantizedKV = self.quantizedKV else {
        fatalError("Called quantized path without quantizedKV set")
      }

      // NF4 needs special handling (2 values per byte)
      if quantizedKV == .NF4 {
        return innerLoopHeadQuantizedNF4(descriptor: descriptor)
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
        dequantFunc = "fp8_e4m3_to_half"
      }

      // Scale buffer name
      let scaleBuffer = "\(B)_scale"

      // For V in attention: V[seq, head], loaded with transposed(B) flag.
      let loadTranspose = transposed(B)

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
        quantBaseDecl = "device const uchar* quant_base = \(B);  // V is device const uchar*"
      }

      // For threadgroup: seq offset is relative to tile (starts at 0)
      // For device: seq offset is absolute (needs V_seq_base)
      let seqBaseExpr = useThreadgroup ? "0" : "\(B)_seq_base"
      // For threadgroup: head offset is relative to tile (starts at 0)
      // For device: head offset is absolute (needs d_outer)
      let headBaseExpr = useThreadgroup ? "d" : "(d_outer + d)"

      return """

      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        simdgroup_matrix_storage<half> \(B)_tile;

        // Load and dequantize 2 elements per thread
        {
          \(quantBaseDecl)
          float scale_val = \(scaleBuffer)[batch_head_idx];

          // For simdgroup_matrix_storage 8x8 tile in accumulate (O += P * V):
          // - B_origin would be (d, c) where d is head offset, c is traversal offset
          // - morton_offset.x = column position in tile (0-7, step 2)
          // - morton_offset.y = row position in tile (0-7)
          // - Each thread loads 2 adjacent columns in the same row
          //
          // Full offsets:
          // - head_pos = head_base + (tile column contribution)
          // - seq_pos = seq_base + c + (tile row contribution)
          //
          // With transposed(B)=false (V not transposed in memory):
          // - V[seq, head] storage: address = seq * headDim + head
          // - Tile columns map to head dimension, tile rows map to seq dimension
          // - 2 adjacent elements are at adjacent head positions, same seq position

          uint addr0, addr1;
          if (\(loadTranspose)) {
            // loadTranspose=true: tile columns = memory rows, tile rows = memory cols
            // seq_pos = base + inner_c + morton_offset.x
            // head_pos = head_base + morton_offset.y
            uint seq_pos0 = \(seqBaseExpr) + c + morton_offset.x;
            uint seq_pos1 = seq_pos0 + 1;
            uint head_pos = \(headBaseExpr) + morton_offset.y;
            addr0 = seq_pos0 * \(leadingDim) + head_pos;
            addr1 = seq_pos1 * \(leadingDim) + head_pos;
          } else {
            // loadTranspose=false: tile columns = memory cols (head), tile rows = memory rows (seq)
            // head_pos = head_base + morton_offset.x
            // seq_pos = seq_base + inner_c + morton_offset.y
            uint head_pos0 = \(headBaseExpr) + morton_offset.x;
            uint head_pos1 = head_pos0 + 1;
            uint seq_pos = \(seqBaseExpr) + c + morton_offset.y;
            addr0 = seq_pos * \(leadingDim) + head_pos0;
            addr1 = seq_pos * \(leadingDim) + head_pos1;
          }

          uchar quant_val0 = quant_base[addr0];
          uchar quant_val1 = quant_base[addr1];

          half dequant0 = \(dequantFunc)(quant_val0, scale_val);
          half dequant1 = \(dequantFunc)(quant_val1, scale_val);

          ((thread half*)\(B)_tile.thread_elements())[0] = dequant0;
          ((thread half*)\(B)_tile.thread_elements())[1] = dequant1;
        }

        // Issue one SIMD matmul instruction.
        \(C)_sram[(\(descriptor.registerOffset) + d) / 8].multiply(
          \(A)_sram[c / 8], \(B)_tile, /*accumulate=*/true);
      }

      """
    }

    /// NF4 quantized inner loop head - special handling for 2 values per byte
    ///
    /// NF4 packs 2 values per byte along the HEAD DIMENSION (D):
    /// - Python stores V as (B, H, N, D//2) uint8
    /// - Low nibble = even D index, high nibble = odd D index
    /// - Memory layout: V[seq, packed_head] where packed_head = D//2
    /// - byte_addr = seq * (D//2) + (head // 2)
    /// - nibble = (head % 2 == 0) ? low : high
    func innerLoopHeadQuantizedNF4(
      descriptor: LoopIterationDescriptor
    ) -> String {
      let scaleBuffer = "\(B)_scale"
      let loadTranspose = transposed(B)
      let useThreadgroup = descriptor.addressSpaceRHS == .threadgroup

      // NF4 packed stride = D/2 (NOT the regular leadingDimension which is D)
      // For threadgroup: we only copied the tile, so stride is blockDimensions.head/2
      // For device: stride is full headDimension/2
      let packedStride = useThreadgroup ? "\(blockDimensions.head / 2)" : "(\(headDimension) / 2)"

      let quantBaseDecl: String
      if useThreadgroup {
        quantBaseDecl = "threadgroup uchar* quant_base = \(B)_tg_base;"
      } else {
        quantBaseDecl = "device const uchar* quant_base = \(B);"
      }

      // Seq base: for device memory, use saved outer loop value; for threadgroup, start at 0
      let seqBaseExpr = useThreadgroup ? "0" : "\(B)_seq_base"
      // Head base: for device memory, use d_outer + d; for threadgroup, just d (relative to tile)
      let headBaseExpr = useThreadgroup ? "d" : "(d_outer + d)"

      return """

      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        simdgroup_matrix_storage<half> \(B)_tile;

        // NF4: 2 values packed per byte along HEAD dimension
        // Memory: V[seq, packed_head] with packed_head = D//2
        // byte_addr = seq * (D/2) + (head / 2)
        // nibble = low if head%2==0, high if head%2==1
        {
          \(quantBaseDecl)
          float scale_val = \(scaleBuffer)[batch_head_idx];
          uint packed_stride = \(packedStride);

          // simdgroup_matrix_storage 8x8 tile mapping:
          // - morton_offset.x = column within tile (0,2,4,6)
          // - morton_offset.y = row within tile (0-7)
          // For accumulate (O += P * V), we load V[d, c] tile

          if (\(loadTranspose)) {
            // loadTranspose=true:
            // tile columns -> seq positions (c dimension)
            // tile rows -> head positions (d dimension)
            // 2 adjacent tile columns = 2 adjacent seq, same head
            uint seq0 = \(seqBaseExpr) + c + morton_offset.x;
            uint seq1 = seq0 + 1;
            uint head = \(headBaseExpr) + morton_offset.y;

            uint byte0 = seq0 * packed_stride + (head / 2);
            uint byte1 = seq1 * packed_stride + (head / 2);

            uchar packed0 = quant_base[byte0];
            uchar packed1 = quant_base[byte1];

            bool high = (head & 1) != 0;
            half val0 = nf4_to_half(packed0, high, scale_val);
            half val1 = nf4_to_half(packed1, high, scale_val);

            ((thread half*)\(B)_tile.thread_elements())[0] = val0;
            ((thread half*)\(B)_tile.thread_elements())[1] = val1;
          } else {
            // loadTranspose=false:
            // tile columns -> head positions (d dimension)
            // tile rows -> seq positions (c dimension)
            // 2 adjacent tile columns = 2 adjacent heads, same seq
            uint head0 = \(headBaseExpr) + morton_offset.x;
            uint head1 = head0 + 1;
            uint seq = \(seqBaseExpr) + c + morton_offset.y;

            uint byte0 = seq * packed_stride + (head0 / 2);
            uint byte1 = seq * packed_stride + (head1 / 2);

            uchar packed0 = quant_base[byte0];
            uchar packed1 = quant_base[byte1];

            half val0 = nf4_to_half(packed0, (head0 & 1) != 0, scale_val);
            half val1 = nf4_to_half(packed1, (head1 & 1) != 0, scale_val);

            ((thread half*)\(B)_tile.thread_elements())[0] = val0;
            ((thread half*)\(B)_tile.thread_elements())[1] = val1;
          }
        }

        \(C)_sram[(\(descriptor.registerOffset) + d) / 8].multiply(
          \(A)_sram[c / 8], \(B)_tile, /*accumulate=*/true);
      }

      """
    }

    func innerLoopTraversal(
      traversalStart: String,
      traversalEnd: String,
      descriptor: LoopIterationDescriptor
    ) -> String {
      return """

      #pragma clang loop unroll(full)
      for (ushort c = \(traversalStart); c < \(traversalEnd); c += 8) {
        \(innerLoopHead(descriptor: descriptor))
      }
      
      """
    }
    
    // MARK: - Outer Loop
    
    struct LoopIterationDescriptor {
      var addressSpaceLHS: MTLAddressSpace?
      var addressSpaceRHS: MTLAddressSpace?
      var registerOffset: String = ""
      var registerSize: UInt16 = .zero
    }
    
    func loopIteration(
      descriptor: LoopIterationDescriptor
    ) -> String {
      func multiplyAB() -> String {
        if descriptor.addressSpaceLHS! == .device ||
            descriptor.addressSpaceRHS! == .device {
          let blockDim = blockDimensions.traversal
          return """
          
          \(innerLoopTraversal(
              traversalStart: "0",
              traversalEnd: "\(blockDim)",
              descriptor: descriptor))
          if (
            (\(traversalDimension) % \(blockDim) == 0) &&
            (\(traversalOffset) + \(blockDim) == \(traversalDimension))
          ) {
             \(scaleAccumulator(
                 by: accumulateDesc.lastIterationScale,
                 descriptor: descriptor))
          }
          
          """
        } else {
          return """
          
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
          } else {
            \(scaleAccumulator(
                by: accumulateDesc.lastIterationScale,
                descriptor: descriptor))
          }
          
          """
        }
      }
      
      return """
      
      \(allocateAccumulator(descriptor: descriptor))
      if (\(traversalOffset) == 0) {
        \(initializeAccumulator(descriptor: descriptor))
      } else {
        \(cacheAccumulator(
            descriptor: descriptor,
            type: .load))
        \(scaleAccumulator(
            by: accumulateDesc.everyIterationScale,
            descriptor: descriptor))
      }
      \(loadRHS(descriptor: descriptor))
      \(multiplyAB())
      \(cacheAccumulator(
          descriptor: descriptor,
          type: .store))
      
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
      if cached(C) {
        return "#pragma clang loop unroll(full)"
      } else {
        return "#pragma clang loop unroll(disable)"
      }
    }
    
    func registerOffset() -> String {
      if cached(C) {
        return "d_outer"
      } else {
        return "0"
      }
    }
    
    func firstIterations() -> String {
      var descriptor = LoopIterationDescriptor()
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
    
    \(firstIterations())
    \(lastIteration())
    
    """
  }
}
