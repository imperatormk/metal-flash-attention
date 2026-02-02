//
//  AttentionKernel+Source.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/2/24.
//

// Top level specification of the code structure.

extension AttentionKernel {
  public func createSource() -> String {
    func createLoop() -> String {
      switch type {
      case .forward:
        return loopForward()
      case .backwardQuery:
        return loopBackwardQuery()
      case .backwardKeyValue:
        return loopBackwardKeyValue()
      }
    }
    
    return """
    #include <metal_stdlib>

    \(createMetalSimdgroupEvent())
    \(createMetalSimdgroupMatrixStorage())
    using namespace metal;

    \(createQuantizationUtilities())
    \(createConstants())

    // Batched dispatch uniform buffer (passed at runtime, not baked into pipeline)
    // This struct holds per-head strides for buffer indexing
    struct BatchedParams {
      uint num_heads;
      uint Q_head_stride;
      uint K_head_stride;
      uint V_head_stride;
      uint O_head_stride;
      uint L_head_stride;
      uint mask_head_stride;
    };

    // Declare the function.
    kernel void attention(
      \(createBufferBindings())
      constant BatchedParams& batched_params [[buffer(30)]],
      threadgroup uchar *threadgroup_block [[threadgroup(0)]],

      uint3 gid [[threadgroup_position_in_grid]],
      ushort sidx [[simdgroup_index_in_threadgroup]],
      ushort lane_id [[thread_index_in_simdgroup]]
    ) {
      ushort2 morton_offset = morton_order(lane_id);

      // gid.x = block within sequence, gid.y = head index, gid.z = batch index
      uint parallelization_group_offset = gid.x;
      parallelization_group_offset *= \(blockDimensions.parallelization);

      // Compute buffer offsets for this batch/head (in elements)
      uint batch_head_idx = gid.z * batched_params.num_heads + gid.y;

      // Offset all buffer pointers for batched dispatch
      \(createBatchedPointerOffsets())

      // Return early if the entire SIMD is out of bounds.
      if (\(parallelizationGroupOffset) >= \(parallelizationDimension)) {
        return;
      }

      \(createSetup())
      \(createLoop())
      \(createCleanup(type: type))
    }

    """
  }
}

// MARK: - Batched Dispatch

extension AttentionKernel {
  /// Creates pointer offset code for batched dispatch
  /// This offsets all buffer pointers by (batch * num_heads + head) * stride
  func createBatchedPointerOffsets() -> String {
    // Determine which operands need offsetting based on kernel type
    var offsets: [String] = []

    switch type {
    case .forward:
      offsets.append("Q += batch_head_idx * batched_params.Q_head_stride;")
      offsets.append("K += batch_head_idx * batched_params.K_head_stride;")
      offsets.append("V += batch_head_idx * batched_params.V_head_stride;")
      offsets.append("O += batch_head_idx * batched_params.O_head_stride;")
      offsets.append("L += batch_head_idx * batched_params.L_head_stride;")
      if hasMask {
        offsets.append("mask += batch_head_idx * batched_params.mask_head_stride;")
      }
      // Note: attn_bias handled separately with its own stride logic

    case .backwardQuery:
      offsets.append("Q += batch_head_idx * batched_params.Q_head_stride;")
      offsets.append("K += batch_head_idx * batched_params.K_head_stride;")
      offsets.append("V += batch_head_idx * batched_params.V_head_stride;")
      offsets.append("O += batch_head_idx * batched_params.O_head_stride;")
      offsets.append("L += batch_head_idx * batched_params.L_head_stride;")
      offsets.append("D += batch_head_idx * batched_params.L_head_stride;")  // D has same stride as L
      offsets.append("dO += batch_head_idx * batched_params.O_head_stride;")
      offsets.append("dQ += batch_head_idx * batched_params.Q_head_stride;")
      if hasMask {
        offsets.append("mask += batch_head_idx * batched_params.mask_head_stride;")
      }

    case .backwardKeyValue:
      offsets.append("Q += batch_head_idx * batched_params.Q_head_stride;")
      offsets.append("K += batch_head_idx * batched_params.K_head_stride;")
      offsets.append("V += batch_head_idx * batched_params.V_head_stride;")
      offsets.append("L += batch_head_idx * batched_params.L_head_stride;")
      offsets.append("D += batch_head_idx * batched_params.L_head_stride;")
      offsets.append("dO += batch_head_idx * batched_params.O_head_stride;")
      offsets.append("dK += batch_head_idx * batched_params.K_head_stride;")
      offsets.append("dV += batch_head_idx * batched_params.V_head_stride;")
      if hasMask {
        offsets.append("mask += batch_head_idx * batched_params.mask_head_stride;")
      }
    }

    return offsets.joined(separator: "\n      ")
  }
}

// MARK: - Function Signature

extension AttentionKernel {
  func createQuantizationUtilities() -> String {
    // Only include quantization utilities if we have quantized K/V
    guard let quantizedKV = self.quantizedKV, quantizedKV.isQuantized else {
      return ""
    }

    // Include the appropriate dequantization functions based on the quantization type
    switch quantizedKV {
    case .FP8_E4M3:
      return QuantizationUtilities.fp8E4M3ToFloat
    case .FP8_E5M2:
      return QuantizationUtilities.fp8E5M2ToFloat
    case .INT8:
      return QuantizationUtilities.int8ToHalf
    case .NF4:
      return QuantizationUtilities.nf4ToHalf
    default:
      return ""
    }
  }

  func createConstants() -> String {
    """

    // R = row dimension (output sequence)
    // C = column dimension (input sequence)
    constant uint R [[function_constant(0)]];
    constant uint C [[function_constant(1)]];

    """
  }
  
  func createBufferBindings() -> String {
    // What operands does the kernel use?
    var operands: [AttentionOperand] = []
    switch type {
    case .forward:
      // To simplify the implementation, we always compute log-sum-exp in the
      // forward pass. Even when it will never be used (model inference).
      // If this is an issue, clients can change the code to selectively
      // omit the 'L' operand.
      operands += [.Q, .K, .V, .O]
      operands += [.L]
      if hasMask {
        operands += [.mask]
      }
      if hasAttnBias {
        operands += [.attnBias]
      }
    case .backwardQuery:
      operands += [.Q, .K, .V, .O]
      operands += [.dO, .dQ]
      operands += [.L, .D]
      if hasMask {
        operands += [.mask]
      }
      if hasAttnBias {
        operands += [.attnBias]
      }
    case .backwardKeyValue:
      operands += [.Q, .K, .V]
      operands += [.dO, .dV, .dK]
      operands += [.L, .D]
      if hasMask {
        operands += [.mask]
      }
      if hasAttnBias {
        operands += [.attnBias]
      }
    }
    operands.sort {
      $0.bufferBinding! < $1.bufferBinding!
    }

    var output: String = ""
    for operand in operands {
      if operand == .mask {
        // Mask is boolean (stored as uchar for Metal compatibility)
        var line = "device const uchar* \(operand) "
        line += "[[buffer(\(operand.bufferBinding!))]],"
        output += "  " + line + "\n"
      } else if operand == .attnBias {
        // Attention bias is float (same precision as S)
        var line = "device const \(registerName(.S))* \(operand) "
        line += "[[buffer(\(operand.bufferBinding!))]],"
        output += "  " + line + "\n"
      } else if (operand == .K || operand == .V), let quantizedKV = self.quantizedKV, quantizedKV.isQuantized {
        // Quantized K/V use uchar* for the raw quantized data
        var line = "device const uchar* \(operand) "
        line += "[[buffer(\(operand.bufferBinding!))]],"
        output += "  " + line + "\n"
      } else {
        var line = "device \(memoryName(operand))* \(operand) "
        line += "[[buffer(\(operand.bufferBinding!))]],"
        output += "  " + line + "\n"
      }
    }

    // Add scale buffers for quantized K/V
    if let quantizedKV = self.quantizedKV, quantizedKV.isQuantized {
      // K_scale: per-head scale factors for K, shape [num_heads] or [batch, num_heads]
      output += "  device const float* K_scale [[buffer(20)]],\n"
      // V_scale: per-head scale factors for V, shape [num_heads] or [batch, num_heads]
      output += "  device const float* V_scale [[buffer(21)]],\n"
    }

    return output
  }
}

// MARK: - Outer Loop

// Forward
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     (m, l, P) = softmax(m, l, S * scaleFactor)
//
//     O *= correction
//     load V[c]
//     O += P * V
//   }
//   O /= l
//
//   L = m + logBaseE(l)
//
// Backward Query
//   D = dO * O
//
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     P = exp(S - L)
//
//     load V[c]
//     dP = dO * V^T
//     dS = P * (dP - D) * scaleFactor
//
//     load K[c]
//     dQ += dS * K
//   }
//
// Backward Key-Value
//   for r in 0..<R {
//     load Q[r]
//     load L[r]
//     S^T = K * Q^T
//     P^T = exp(S^T - L)
//
//     load dO[r]
//     dV += P^T * dO
//
//     load dO[r]
//     load D[r]
//     dP^T = V * dO^T
//     dS^T = P^T * (dP^T - D) * scaleFactor
//
//     load Q[r]
//     dK += dS^T * Q
//   }

extension AttentionKernel {
  func loopForward() -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .Q
    outerProductDesc.B = .K
    outerProductDesc.C = .S
    let QKT = outerProduct(descriptor: outerProductDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = .P
    accumulateDesc.B = .V
    accumulateDesc.C = .O
    accumulateDesc.everyIterationScale = "correction"
    // Clamp l to avoid division by zero when all positions are masked
    accumulateDesc.lastIterationScale = "fast::divide(1, max(l, 1e-9f))"
    let PV = accumulate(descriptor: accumulateDesc)

    // Generate early exit condition for sliding window
    // Skip KV blocks that are entirely outside the attention window for ALL rows in the Q block
    // IMPORTANT: Never skip c=0 because the accumulator is initialized there
    let slidingWindowEarlyExit: String
    if let windowSize = windowSize, windowSize > 0 {
      // The Q block covers rows [parallelization_group_offset, parallelization_group_offset + blockDimensions.parallelization)
      // We can skip a KV block [c, c + traversal) if its END is before the window start of the MINIMUM row
      slidingWindowEarlyExit = """

      // Early exit: skip KV blocks entirely before the sliding window for all Q rows
      // Never skip c=0 because that's where O accumulator is initialized
      if (c > 0 && \(parallelizationGroupOffset) >= \(windowSize)) {
        uint window_start_min_row = \(parallelizationGroupOffset) - \(windowSize);
        if (c + \(blockDimensions.traversal) <= window_start_min_row) {
          continue;
        }
      }

"""
    } else {
      slidingWindowEarlyExit = ""
    }

    return """

    // Outer loop over the traversal dimension.
    for (uint c = 0; c < C; c += \(blockDimensions.traversal)) {
      \(slidingWindowEarlyExit)
      // S = Q * K^T
      \(QKT)
      \(maskAttentionMatrixEdge())
      \(maskCausal())
      \(maskSlidingWindow())
      \(maskWithExternalMask())
      \(addAttnBias())

      // m = reduce(m)
      \(onlineReduceMaximum())

      // correction = exp(m_old) / exp(m_new)
      \(onlineCorrectO())

      // P = softmax(S * scaleFactor)
      \(softmax(derivative: false))

      // l = reduce(l)
      \(onlineReduceSum())

      // O *= correction
      // O += P * V
      // O /= l
      \(PV)
    }

    """
  }
  
  func loopBackwardQuery() -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .Q
    outerProductDesc.B = .K
    outerProductDesc.C = .S
    let QKT = outerProduct(descriptor: outerProductDesc)
    
    outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .dO
    outerProductDesc.B = .V
    outerProductDesc.C = .dP
    let dOVT = outerProduct(descriptor: outerProductDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = .dS
    accumulateDesc.B = .K
    accumulateDesc.C = .dQ
    let dSK = accumulate(descriptor: accumulateDesc)
    
    return """

    // Outer loop over the traversal dimension.
    for (uint c = 0; c < C; c += \(blockDimensions.traversal)) {
      // S = Q * K^T
      \(QKT)
      \(maskCausal())
      \(maskSlidingWindow())
      \(maskWithExternalMask())
      \(addAttnBias())

      // P = softmax(S * scaleFactor)
      \(softmax(derivative: false))

      // dP = dO * V^T
      \(dOVT)

      // dS = P * (dP - D) * scaleFactor
      \(softmax(derivative: true))

      // dQ += dS * K
      \(dSK)
    }

    """
  }
  
  func loopBackwardKeyValue() -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .K
    outerProductDesc.B = .Q
    outerProductDesc.C = .S // S^T
    let KQT = outerProduct(descriptor: outerProductDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = .P // P^T
    accumulateDesc.B = .dO
    accumulateDesc.C = .dV
    let PTdO = accumulate(descriptor: accumulateDesc)
    
    outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .V
    outerProductDesc.B = .dO
    outerProductDesc.C = .dP // dP^T
    let VdOT = outerProduct(descriptor: outerProductDesc)
    
    accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = .dS // dS^T
    accumulateDesc.B = .Q
    accumulateDesc.C = .dK
    let dSTQ = accumulate(descriptor: accumulateDesc)
    
    return """

    // Outer loop over the traversal dimension.
    for (uint r = 0; r < R; r += \(blockDimensions.traversal)) {
      // S^T = K * Q^T
      \(KQT)
      \(maskCausal())
      \(maskSlidingWindow())
      \(maskWithExternalMask())
      \(addAttnBias())

      // P^T = exp(S^T - L)
      \(softmax(derivative: false))

      // dV += P^T * dO
      \(PTdO)

      // dP^T = V * dO^T
      \(VdOT)

      // dS^T = P^T * (dP^T - D) * scaleFactor
      \(softmax(derivative: true))

      // dK += dS^T * Q
      \(dSTQ)
    }

    """
  }
}
