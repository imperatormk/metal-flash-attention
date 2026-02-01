//
//  AttentionKernel+Softmax.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// Elementwise operations on the attention matrix.

// MARK: - Scale Factor

extension AttentionKernel {
  // The scale factor in scaled dot product attention.
  //
  // Parameters:
  // - derivative: Whether this is the derivative softmax.
  func dotProductScale(derivative: Bool) -> Float {
    let logBase2E: Float = 1.442695041
    let rsqrtD = 1 / Float(headDimension).squareRoot()
    
    if !derivative {
      return logBase2E * rsqrtD
    } else {
      return rsqrtD
    }
  }
}

// MARK: - Compute D (dO * O)

extension AttentionKernel {
  func computeD() -> String {
    // Parts of the dO * O reduction that fall within block bounds.
    func bulkContributions(truncatedHeadDimension: UInt16) -> String {
      // Recycle most of the cached values for dO.
      func declareDerivativeOLocation() -> String {
        if cached(.dO) {
          return ""
        } else {
          return """
          
          // Where the dO data will be read from.
          auto dO_src = simdgroup_matrix_storage<\(memoryName(.dO))>
          ::apply_offset(
            dO, \(leadingDimension(.dO)), 
            offset_src, \(transposed(.dO)));
          
          """
        }
      }
      func loadDerivativeO() -> String {
        if cached(.dO) {
          return """
          
          auto dO = dO_sram[d / 8];
          
          """
        } else {
          return """
          
          simdgroup_matrix_storage<\(registerName(.dO))> dO;
          dO.\(loadFunction(.dO))(
            dO_src, \(leadingDimension(.dO)),
            ushort2(d, 0), \(transposed(.dO)));
          
          """
        }
      }
      
      return """
      
      // Threads outside of the matrix along the row dimension,
      // have their origin shifted in-bounds.
      uint D_offset = morton_offset.x;
      uint R_offset = \(clampedParallelizationThreadOffset);
      uint2 offset_src(D_offset, R_offset);
      
      \(declareDerivativeOLocation())
      
      // Where the O data will be read from.
      auto O_src = simdgroup_matrix_storage<\(memoryName(.O))>
      ::apply_offset(
        O, \(leadingDimension(.O)),
        offset_src, \(transposed(.O)));
      
      // Going to use async copy to handle the matrix edge.
      #pragma clang loop unroll(disable)
      for (ushort d = 0; d < \(truncatedHeadDimension); d += 8) {
        \(loadDerivativeO())
        
        simdgroup_matrix_storage<\(registerName(.O))> O;
        O.\(loadFunction(.O))(
          O_src, \(leadingDimension(.O)),
          ushort2(d, 0), \(transposed(.O)));
        
        // Perform the pointwise multiplication.
        auto dO_value = *(dO.thread_elements());
        auto O_value = *(O.thread_elements());
        D_accumulator += float2(dO_value) * float2(O_value);
      }

      """
    }
    
    // Parts of the dO * O reduction that fall on an indivisible edge.
    func edgeContributions(truncatedHeadDimension: UInt16) -> String {
      guard headDimension % 8 != 0 else {
        return ""
      }
      
      // Abbreviated block, only covers the last 8 elements.
      func leadingBlockDimension(_ operand: AttentionOperand) -> UInt16 {
        if transposed(operand) {
          return blockSequenceLength(operand)
        } else {
          return 8
        }
      }
      
      // Distinct from the block bytes that would be used to calculate
      // the threadgroup memory allocation.
      func blockBytesDerivativeO() -> UInt16 {
        let memoryPrecision = memoryPrecisions[.dO]!
        let size = UInt16(memoryPrecision.size)
        return blockDimensions.parallelization * 8 * size
      }
      
      return """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint D_offset = \(truncatedHeadDimension);
        uint R_offset = \(parallelizationGroupOffset);
        uint2 offset_src(D_offset, R_offset);
        
        auto dO_src = simdgroup_matrix_storage<\(memoryName(.dO))>
        ::apply_offset(
          dO, \(leadingDimension(.dO)), 
          offset_src, \(transposed(.dO)));
        auto O_src = simdgroup_matrix_storage<\(memoryName(.O))>
        ::apply_offset(
          O, \(leadingDimension(.O)), 
          offset_src, \(transposed(.O)));
        
        auto dO_dst = (threadgroup \(memoryName(.dO))*)(threadgroup_block);
        auto O_dst = (threadgroup \(memoryName(.O))*)(
          threadgroup_block + \(blockBytesDerivativeO()));
        
        ushort D_src_dimension = \(headDimension) % 8;
        ushort D_dst_dimension = 8;
        ushort R_dimension = min(
          uint(\(blockDimensions.parallelization)),
          uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
        ushort2 tile_src(D_src_dimension, R_dimension);
        ushort2 tile_dst(D_dst_dimension, R_dimension);
        
        // Issue two async copies.
        simdgroup_event events[2];
        events[0].async_copy(
          dO_dst, \(leadingBlockDimension(.dO)), tile_dst,
          dO_src, \(leadingDimension(.dO)), tile_src, \(transposed(.dO)));
        events[1].async_copy(
          O_dst, \(leadingBlockDimension(.O)), tile_dst,
          O_src, \(leadingDimension(.O)), tile_src, \(transposed(.O)));
        simdgroup_event::wait(2, events);
      }
      
      // Where the dO and O data will be read from.
      ushort2 offset_src(morton_offset.x, morton_offset.y + sidx * 8);
      auto dO_block = (threadgroup \(memoryName(.dO))*)(threadgroup_block);
      auto O_block = (threadgroup \(memoryName(.O))*)(
        threadgroup_block + \(blockBytesDerivativeO()));
      
      dO_block = simdgroup_matrix_storage<\(memoryName(.dO))>
      ::apply_offset(
        dO_block, \(leadingBlockDimension(.dO)),
        offset_src, \(transposed(.dO)));
      O_block = simdgroup_matrix_storage<\(memoryName(.O))>
      ::apply_offset(
        O_block, \(leadingBlockDimension(.O)),
        offset_src, \(transposed(.O)));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      // Load the zero-padded edge data.
      ushort2 origin(0, 0);
      simdgroup_matrix_storage<\(registerName(.dO))> dO;
      simdgroup_matrix_storage<\(registerName(.O))> O;
      dO.\(loadFunction(.dO))(
        dO_block, \(leadingBlockDimension(.dO)),
        origin, \(transposed(.dO)));
      O.\(loadFunction(.O))(
        O_block, \(leadingBlockDimension(.O)),
        origin, \(transposed(.O)));
      
      // Perform the pointwise multiplication.
      auto dO_value = *(dO.thread_elements());
      auto O_value = *(O.thread_elements());
      D_accumulator += float2(dO_value) * float2(O_value);
      
      """
    }
    
    // Outer loop over the head dimension.
    let loopEndFloor = headDimension - headDimension % 8
    return """
    
    float2 D_accumulator(0);
    {
      \(bulkContributions(truncatedHeadDimension: loopEndFloor))
    }
    {
      \(edgeContributions(truncatedHeadDimension: loopEndFloor))
    }
    
    float D_sram = D_accumulator[0] + D_accumulator[1];
    D_sram += simd_shuffle_xor(D_sram, 1);
    D_sram += simd_shuffle_xor(D_sram, 8);
    D_sram *= \(dotProductScale(derivative: true));
    
    """
  }
}

// MARK: - Mask

extension AttentionKernel {
  // Prevent the zero padding from changing the values of 'm' and 'l'.
  func maskAttentionMatrixEdge() -> String {
    let blockDim = blockDimensions.traversal
    let remainder = "(\(traversalDimension) % \(blockDim))"
    let remainderFloor = "(\(remainder) - (\(remainder) % 8))";
    let logBase2E: Float = 1.442695041

    return """

    if ((\(remainder) != 0) &&
        (\(traversalOffset) + \(blockDim) > \(traversalDimension))) {
      // Prevent the value from becoming -INF during the FMA before the
      // exponentiation. If the multiplication during FMA returns -INF,
      // subtracting a positive 'm' value will turn it into zero. We don't want
      // that. exp(0) evaluates to 1.00 and corrupts the value of 'l'.
      const \(registerName(.S)) mask_value =
      (0.875 / \(logBase2E)) * -numeric_limits<\(registerName(.S))>::max();

      #pragma clang loop unroll(full)
      for (ushort index = 0; index < 2; ++index) {
        if (morton_offset.x + index >= \(remainder) - \(remainderFloor)) {
          auto S_elements = S_sram[\(remainderFloor) / 8].thread_elements();
          (*S_elements)[index] = mask_value;
        }
      }
      #pragma clang loop unroll(full)
      for (ushort c = \(remainderFloor) + 8; c < \(blockDim); c += 8) {
        auto S_elements = S_sram[c / 8].thread_elements();
        *S_elements = mask_value;
      }
    }

    """
  }

  // Apply external attention mask from buffer
  // Mask buffer is [seq_q, seq_k] where non-zero = masked (-inf), 0 = attend
  // This matches PyTorch's boolean mask convention (True = masked)
  func maskWithExternalMask() -> String {
    guard hasMask else { return "" }

    let logBase2E: Float = 1.442695041
    let blockDim = blockDimensions.traversal

    switch type {
    case .forward, .backwardQuery:
      // row = parallelization position (Q position), col = traversal position (KV position)
      return """

      // Apply external attention mask
      {
        const \(registerName(.S)) mask_value =
        (0.875 / \(logBase2E)) * -numeric_limits<\(registerName(.S))>::max();

        // Current row position (Q position within the sequence)
        uint row_idx = \(parallelizationGroupOffset) + sidx * 8 + morton_offset.y;

        #pragma clang loop unroll(full)
        for (ushort c_block = 0; c_block < \(blockDim); c_block += 8) {
          // Column position (KV position within the sequence)
          uint col_base = \(traversalOffset) + c_block;

          auto S_elements = S_sram[c_block / 8].thread_elements();

          #pragma clang loop unroll(full)
          for (ushort index = 0; index < 2; ++index) {
            uint col_idx = col_base + morton_offset.x + index;
            // Check bounds and load mask value
            if (row_idx < R && col_idx < C) {
              uint mask_idx = row_idx * C + col_idx;
              uchar mask_val = mask[mask_idx];
              // mask_val != 0 means masked out (-inf), mask_val == 0 means attend
              if (mask_val != 0) {
                (*S_elements)[index] = mask_value;
              }
            }
          }
        }
      }

      """

    case .backwardKeyValue:
      // In backwardKeyValue, we compute S^T = K * Q^T
      // parallelization = KV position, traversal = Q position
      return """

      // Apply external attention mask (transposed access for backward KV)
      {
        const \(registerName(.S)) mask_value =
        (0.875 / \(logBase2E)) * -numeric_limits<\(registerName(.S))>::max();

        // Current KV position (parallelization dimension)
        uint kv_idx = \(parallelizationGroupOffset) + sidx * 8 + morton_offset.y;

        #pragma clang loop unroll(full)
        for (ushort r_block = 0; r_block < \(blockDim); r_block += 8) {
          // Q position (traversal dimension)
          uint q_base = \(traversalOffset) + r_block;

          auto S_elements = S_sram[r_block / 8].thread_elements();

          #pragma clang loop unroll(full)
          for (ushort index = 0; index < 2; ++index) {
            uint q_idx = q_base + morton_offset.x + index;
            // Check bounds and load mask value (mask is [Q, KV])
            if (q_idx < R && kv_idx < C) {
              uint mask_idx = q_idx * C + kv_idx;
              uchar mask_val = mask[mask_idx];
              if (mask_val != 0) {
                (*S_elements)[index] = mask_value;
              }
            }
          }
        }
      }

      """
    }
  }

  // Apply sliding window masking: each token only attends to windowSize previous tokens
  // Mask positions where col_idx < row_idx - windowSize (tokens too far in the past)
  // Combined with causal, this creates: row_idx - windowSize <= col_idx <= row_idx
  func maskSlidingWindow() -> String {
    guard let windowSize = windowSize, windowSize > 0 else { return "" }

    let logBase2E: Float = 1.442695041
    let blockDim = blockDimensions.traversal

    switch type {
    case .forward, .backwardQuery:
      // row = parallelization position (Q position), col = traversal position (KV position)
      // Mask where col_idx < row_idx - windowSize (too far in the past)
      return """

      // Sliding window masking: mask positions where col_idx < row_idx - windowSize
      {
        const \(registerName(.S)) mask_value =
        (0.875 / \(logBase2E)) * -numeric_limits<\(registerName(.S))>::max();

        // Current row position (Q position within the sequence)
        uint row_idx = \(parallelizationGroupOffset) + sidx * 8 + morton_offset.y;
        // Calculate window start: positions before this are masked
        int window_start = max(0, int(row_idx) - int(\(windowSize)));

        #pragma clang loop unroll(full)
        for (ushort c_block = 0; c_block < \(blockDim); c_block += 8) {
          // Column position (KV position within the sequence)
          uint col_base = \(traversalOffset) + c_block;

          auto S_elements = S_sram[c_block / 8].thread_elements();

          #pragma clang loop unroll(full)
          for (ushort index = 0; index < 2; ++index) {
            int col_idx = int(col_base + morton_offset.x + index);
            // Mask if col_idx < window_start (too far in the past)
            if (col_idx < window_start) {
              (*S_elements)[index] = mask_value;
            }
          }
        }
      }

      """

    case .backwardKeyValue:
      // In backwardKeyValue, we compute S^T = K * Q^T
      // parallelization = KV position, traversal = Q position
      // Sliding window: Q position must be within [KV position, KV position + windowSize]
      // Equivalently: mask where traversal_idx < parallelization_idx - windowSize
      return """

      // Sliding window masking for backward KV
      {
        const \(registerName(.S)) mask_value =
        (0.875 / \(logBase2E)) * -numeric_limits<\(registerName(.S))>::max();

        // Current KV position (parallelization dimension)
        uint kv_idx = \(parallelizationGroupOffset) + sidx * 8 + morton_offset.y;

        #pragma clang loop unroll(full)
        for (ushort r_block = 0; r_block < \(blockDim); r_block += 8) {
          // Q position (traversal dimension)
          uint q_base = \(traversalOffset) + r_block;

          auto S_elements = S_sram[r_block / 8].thread_elements();

          #pragma clang loop unroll(full)
          for (ushort index = 0; index < 2; ++index) {
            int q_idx = int(q_base + morton_offset.x + index);
            // For P^T[kv, q], the original P[q, kv] has sliding window
            // P[q, kv] is masked when kv < q - windowSize
            // So mask when kv_idx < q_idx - windowSize, i.e., q_idx > kv_idx + windowSize
            if (q_idx > int(kv_idx) + int(\(windowSize))) {
              (*S_elements)[index] = mask_value;
            }
          }
        }
      }

      """
    }
  }

  // Apply causal masking: mask out positions where row < column (future tokens)
  // For forward/backwardQuery: row is the parallelization dim, column is traversal
  // For backwardKeyValue: we compute S^T, so row and column are swapped
  func maskCausal() -> String {
    guard causal else { return "" }

    let logBase2E: Float = 1.442695041
    let blockDim = blockDimensions.traversal

    // In forward/backwardQuery: S[row, col], mask where row < col
    // In backwardKeyValue: we compute S^T = K * Q^T, so S^T[kv_pos, q_pos]
    //   This is P^T where P = softmax(S). For causal: S[q, kv] masked where q < kv
    //   So S^T[kv, q] is masked where q < kv, i.e., col < row in S^T space
    //   But we're iterating q (traversal), kv (parallelization)
    //   Mask where: traversal_idx < parallelization_idx (the Q position < KV position)

    switch type {
    case .forward, .backwardQuery:
      // row = parallelization position (Q position), col = traversal position (KV position)
      // Mask where row < col (can't attend to future)
      return """

      // Causal masking: mask positions where row_idx < col_idx
      {
        const \(registerName(.S)) mask_value =
        (0.875 / \(logBase2E)) * -numeric_limits<\(registerName(.S))>::max();

        // Current row position (Q position within the sequence)
        uint row_idx = \(parallelizationGroupOffset) + sidx * 8 + morton_offset.y;

        #pragma clang loop unroll(full)
        for (ushort c_block = 0; c_block < \(blockDim); c_block += 8) {
          // Column position (KV position within the sequence)
          uint col_base = \(traversalOffset) + c_block;

          auto S_elements = S_sram[c_block / 8].thread_elements();

          #pragma clang loop unroll(full)
          for (ushort index = 0; index < 2; ++index) {
            uint col_idx = col_base + morton_offset.x + index;
            // Mask if row < col (future position)
            if (row_idx < col_idx) {
              (*S_elements)[index] = mask_value;
            }
          }
        }
      }

      """

    case .backwardKeyValue:
      // In backwardKeyValue, we compute S^T = K * Q^T
      // parallelization = KV position, traversal = Q position
      // We need P^T where P[q, kv] has causal mask (q < kv is masked)
      // So P^T[kv, q] should mask where q < kv, i.e., traversal_idx < parallelization_idx
      return """

      // Causal masking for backward KV: mask positions where q_idx < kv_idx
      {
        const \(registerName(.S)) mask_value =
        (0.875 / \(logBase2E)) * -numeric_limits<\(registerName(.S))>::max();

        // Current KV position (parallelization dimension)
        uint kv_idx = \(parallelizationGroupOffset) + sidx * 8 + morton_offset.y;

        #pragma clang loop unroll(full)
        for (ushort r_block = 0; r_block < \(blockDim); r_block += 8) {
          // Q position (traversal dimension)
          uint q_base = \(traversalOffset) + r_block;

          auto S_elements = S_sram[r_block / 8].thread_elements();

          #pragma clang loop unroll(full)
          for (ushort index = 0; index < 2; ++index) {
            uint q_idx = q_base + morton_offset.x + index;
            // Mask if q < kv (this position was masked in forward pass)
            if (q_idx < kv_idx) {
              (*S_elements)[index] = mask_value;
            }
          }
        }
      }

      """
    }
  }
}

// MARK: - Reduce

extension AttentionKernel {
  // Reduce maximum during the online softmax.
  func onlineReduceMaximum() -> String {
    """
    
    // update 'm'
    vec<\(registerName(.S)), 2> m_new_accumulator;
    #pragma clang loop unroll(full)
    for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
      auto S_elements = S_sram[c / 8].thread_elements();
      if (c == 0) {
        m_new_accumulator = *S_elements;
      } else {
        m_new_accumulator = max(m_new_accumulator, *S_elements);
      }
    }
    float m_new = max(m_new_accumulator[0], m_new_accumulator[1]);
    m_new = max(m_new, simd_shuffle_xor(m_new, 1));
    m_new = max(m_new, simd_shuffle_xor(m_new, 8));
    m_new *= \(dotProductScale(derivative: false));
    
    """
  }
  
  // Rescale 'O' to reflect the new maximum.
  func onlineCorrectO() -> String {
    """

    // update 'O'
    float correction = 1;
    if (m_new > m) {
      correction = fast::exp2(m - m_new);
      m = m_new;
    }

    """
  }
  
  // Reduce sum during the online softmax.
  func onlineReduceSum() -> String {
    """
    
    // update 'l'
    float2 l_new_accumulator;
    #pragma clang loop unroll(full)
    for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
      auto P_elements = P_sram[c / 8].thread_elements();
      if (c == 0) {
        l_new_accumulator = float2(*P_elements);
      } else {
        l_new_accumulator += float2(*P_elements);
      }
    }
    float l_new = l_new_accumulator[0] + l_new_accumulator[1];
    l_new += simd_shuffle_xor(l_new, 1);
    l_new += simd_shuffle_xor(l_new, 8);
    l = l * correction + l_new;
    
    """
  }
}

// MARK: - Softmax

extension AttentionKernel {
  // A softmax where the per-row statistics have been reduced beforehand.
  //
  // Parameters:
  // - derivative: Whether this is the derivative softmax.
  func softmax(derivative: Bool) -> String {
    let operand: AttentionOperand = derivative ? .D : .L
    
    func allocateOutput() -> String {
      let blockDim = blockDimensions.traversal
      if !derivative {
        return """
        
        simdgroup_matrix_storage<\(registerName(.P))> \
        P_sram[\(blockDim) / 8];
        
        """
      } else {
        return """
        
        simdgroup_matrix_storage<\(registerName(.dS))> \
        dS_sram[\(blockDim) / 8];
        
        """
      }
    }
    
    func loadOperand() -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        auto \(operand)_src = \(operand) + \(traversalOffset);
        auto \(operand)_dst =
        (threadgroup \(memoryName(operand))*)(threadgroup_block);
        
        ushort R_src_dimension = min(
          uint(\(blockDimensions.traversal)),
          uint(\(traversalDimension) - \(traversalOffset)));
        ushort R_dst_dimension = max(
          ushort(\(paddedTraversalEdge)),
          ushort(R_src_dimension));
        
        // Issue an async copy.
        simdgroup_event event;
        event.async_copy(
          \(operand)_dst, 1, ushort2(R_dst_dimension, 1),
          \(operand)_src, 1, ushort2(R_src_dimension, 1));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    // Declares the source of L or D.
    //
    // Also guards against unsafe accesses to the declared pointer (barrier).
    func declareOperandLocation(addressSpace: MTLAddressSpace) -> String {
      if addressSpace == .device {
        return """
        
        auto \(operand)_src = \(operand);
        \(operand)_src += \(traversalOffset) + morton_offset.x;
        
        """
      } else {
        return """
        
        auto \(operand)_src =
        (threadgroup \(memoryName(operand))*)(threadgroup_block);
        \(operand)_src += morton_offset.x;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        """
      }
    }
    
    func overwriteAttentionMatrixElements() -> String {
      let scale = dotProductScale(derivative: derivative)

      if !derivative {
        // Softmax: P = exp2(S * scale - m)
        // When all positions in a row are masked, m = mask_value * scale.
        // mask_value ≈ -39744 (unscaled), so m ≈ -10000 after scaling.
        // Normal attention scores after scaling are in [-100, 100].
        // If m < -1000, the row is fully masked and P should be 0.
        return """

        auto S = *(S_sram[c / 8].thread_elements());
        auto P_computed = vec<\(registerName(.P)), 2>(
          fast::exp2(float2(S) * \(scale) - float2(L_elements)));
        // Fully masked rows have m < -1000; set P = 0 to avoid accumulating garbage
        // Use vec<bool, 2> for proper SIMD select behavior
        vec<bool, 2> row_masked(L_elements < -1000.0f);
        auto P = select(P_computed, vec<\(registerName(.P)), 2>(0), row_masked);
        *(P_sram[c / 8].thread_elements()) = P;

        """
      } else {
        return """
        
        auto P = *(P_sram[c / 8].thread_elements());
        auto dP = *(dP_sram[c / 8].thread_elements());
        auto dS = vec<\(registerName(.dS)), 2>(
          float2(P) * (float2(dP) * \(scale) - float2(D_elements)));
        *(dS_sram[c / 8].thread_elements()) = dS;
        
        """
      }
    }
    
    func innerLoop() -> String {
      switch type {
      case .forward:
        return """
        
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
          auto L_elements = m;
          \(overwriteAttentionMatrixElements())
        }
        
        """
      case .backwardQuery:
        return """
        
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
          auto \(operand)_elements = \(operand)_sram;
          \(overwriteAttentionMatrixElements())
        }
        
        """
      case .backwardKeyValue:
        return """
        
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
          ushort2 \(operand)_origin(c, 0);
          simdgroup_matrix_storage<\(registerName(operand))> \(operand);
          \(operand).\(loadFunction(operand))(
            \(operand)_src, 1,
            \(operand)_origin, false);
          auto \(operand)_elements = *(\(operand).thread_elements());
          
          \(overwriteAttentionMatrixElements())
        }
        
        """
      }
    }
    
    switch type {
    case .forward, .backwardQuery:
      return """
      
      \(allocateOutput())
      {
        \(innerLoop())
      }
      
      """
    case .backwardKeyValue:
      let blockDim = blockDimensions.traversal
      let condition = """
      \(!preferAsyncLoad) && (
        (\(traversalDimension) % \(blockDim) == 0) ||
        (\(traversalOffset) + \(blockDim) <= \(traversalDimension))
      )
      """
      
      return """
      
      \(allocateOutput())
      if (\(condition)) {
        \(declareOperandLocation(addressSpace: .device))
        \(innerLoop())
      } else {
        \(loadOperand())
        \(declareOperandLocation(addressSpace: .threadgroup))
        \(innerLoop())
      }

      """
    }
  }

  // Add attention bias to attention scores
  // Bias buffer is pre-offset by the encoder to point to the correct [seq_q, seq_k] slice
  // for the current batch/head. Kernel just adds bias[row_idx, col_idx] to S.
  func addAttnBias() -> String {
    guard hasAttnBias else { return "" }

    let blockDim = blockDimensions.traversal

    switch type {
    case .forward, .backwardQuery:
      // row = parallelization position (Q position), col = traversal position (KV position)
      return """

      // Add attention bias
      {
        // Current row position (Q position within the sequence)
        uint row_idx = \(parallelizationGroupOffset) + sidx * 8 + morton_offset.y;

        #pragma clang loop unroll(full)
        for (ushort c_block = 0; c_block < \(blockDim); c_block += 8) {
          // Column position (KV position within the sequence)
          uint col_base = \(traversalOffset) + c_block;

          auto S_elements = S_sram[c_block / 8].thread_elements();

          #pragma clang loop unroll(full)
          for (ushort index = 0; index < 2; ++index) {
            uint col_idx = col_base + morton_offset.x + index;
            // Check bounds
            if (row_idx < R && col_idx < C) {
              // Bias buffer is pre-offset by encoder - just index [row, col]
              uint bias_idx = row_idx * C + col_idx;
              \(registerName(.S)) bias_val = attn_bias[bias_idx];
              (*S_elements)[index] += bias_val;
            }
          }
        }
      }

      """

    case .backwardKeyValue:
      // In backwardKeyValue, we compute S^T = K * Q^T
      // parallelization = KV position, traversal = Q position
      return """

      // Add attention bias (transposed access for backward KV)
      {
        // Current KV position (parallelization dimension)
        uint kv_idx = \(parallelizationGroupOffset) + sidx * 8 + morton_offset.y;

        #pragma clang loop unroll(full)
        for (ushort r_block = 0; r_block < \(blockDim); r_block += 8) {
          // Q position (traversal dimension)
          uint q_base = \(traversalOffset) + r_block;

          auto S_elements = S_sram[r_block / 8].thread_elements();

          #pragma clang loop unroll(full)
          for (ushort index = 0; index < 2; ++index) {
            uint q_idx = q_base + morton_offset.x + index;
            // Check bounds
            if (kv_idx < C && q_idx < R) {
              // For S^T[kv, q], we need bias[q, kv]
              // Bias buffer is pre-offset by encoder
              uint bias_idx = q_idx * C + kv_idx;
              \(registerName(.S)) bias_val = attn_bias[bias_idx];
              (*S_elements)[index] += bias_val;
            }
          }
        }
      }

      """
    }
  }
}
