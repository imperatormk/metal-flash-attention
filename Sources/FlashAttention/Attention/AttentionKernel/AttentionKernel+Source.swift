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

    let loadRPs = cachingLoadResumePoints()
    let storeRPs = cachingStoreResumePoints()
    let hasAsyncCaching = loadRPs > 0 || storeRPs > 0

    return """

    \(createReverseLinkingProtocol())
    \(createCooperativeCopy())
    \(createMetalSimdgroupMatrixStorage())
    using namespace metal;

    \(createConstants())

    // Reverse-linking visible function for attention.
    // Called by the pre-compiled IR kernel shell with incrementing resume_point.
    // Async copy is handled by the shell; this function does all compute.
    //
    // TG layout:
    //   [0..127]             = command area (32 x uint32)
    //   [128..128+TGA]       = data area for operand tiles
    //   [128+TGA..]          = save area for register state between resume points

    [[visible]]
    void attention_body(
      threadgroup uchar *tg,
      \(createRawBufferBindings())
      uint resume_point,
      uint gid_x,
      uint lane_id_u,
      uint sidx_u
    ) {
      auto cmd = (threadgroup uint*)(tg);
      auto threadgroup_block = tg + 128;  // data area starts at byte 128

      \(createTypedPointerCasts())

      ushort sidx = ushort(sidx_u);
      ushort lane_id = ushort(lane_id_u);
      ushort2 morton_offset = morton_order(lane_id);
      uint tid = uint(sidx) * 32 + uint(lane_id);
      uint tg_size = \(threadgroupSize);
      uint gid = gid_x;
      uint parallelization_group_offset = gid;
      parallelization_group_offset *= \(blockDimensions.parallelization);

      // Return early if the entire SIMD is out of bounds.
      if (\(parallelizationGroupOffset) >= \(parallelizationDimension)) {
        cmd[0] = CMD_DONE;
        return;
      }

    \(hasAsyncCaching ? createStateMachineDispatch(createLoop: createLoop) : createSinglePassBody(createLoop: createLoop))
    }

    """
  }

  /// Single-pass body when no operands are cached via async copy.
  /// Falls back to the original sequential approach.
  private func createSinglePassBody(createLoop: () -> String) -> String {
    """
      \(createSetup())
      \(createLoop())
      \(createCleanup(type: type))

      cmd[0] = CMD_DONE;
    """
  }

  /// State machine dispatch for async caching.
  /// Each resume_point corresponds to one step in the load/loop/store pipeline.
  private func createStateMachineDispatch(createLoop: () -> String) -> String {
    let loadRPs = cachingLoadResumePoints()
    let storeRPs = cachingStoreResumePoints()
    let chunks = headDimensionChunks()

    // setup_end = loadRPs (all load resume points)
    // loop_point = loadRPs (reads last load chunk + runs loop + first store)
    // cleanup stores = loop_point + 1 .. loop_point + storeRPs
    // scalar cleanup = loop_point + storeRPs + 1
    let loopPoint = loadRPs
    let scalarPoint = loopPoint + 1 + storeRPs

    if storeRPs > 0 {
      return """
        // State machine: resume_point-based dispatch for async caching.
        // Load phase: resume_point 0..\(loadRPs - 1)
        // Loop phase: resume_point \(loopPoint)
        // Store phase: resume_point \(loopPoint + 1)..\(loopPoint + storeRPs)
        // Scalar+done: resume_point \(scalarPoint)

        if (resume_point < \(loopPoint)) {
          // --- Async caching load phase ---
          \(createAsyncCachingLoadDispatch(chunks: chunks))
        } else if (resume_point == \(loopPoint)) {
          // --- Loop phase: read last load chunk, run traversal, begin stores ---
          \(createLoopPhase(createLoop: createLoop, chunks: chunks))
        } else if (resume_point <= \(loopPoint + storeRPs)) {
          // --- Async caching store phase ---
          \(createAsyncCachingStoreDispatch(chunks: chunks, loopPoint: loopPoint))
        } else {
          // --- Scalar cleanup + done ---
          \(createScalarCleanup())
          cmd[0] = CMD_DONE;
        }
      """
    } else {
      // No store operands — loop phase handles scalar cleanup and CMD_DONE
      // directly. No store/scalar-cleanup resume points needed.
      return """
        // State machine: resume_point-based dispatch for async caching.
        // Load phase: resume_point 0..\(loadRPs - 1)
        // Loop+cleanup: resume_point \(loopPoint)

        if (resume_point < \(loopPoint)) {
          // --- Async caching load phase ---
          \(createAsyncCachingLoadDispatch(chunks: chunks))
        } else {
          // --- Loop phase: read last load chunk, run traversal, cleanup, done ---
          \(createLoopPhase(createLoop: createLoop, chunks: chunks))
        }
      """
    }
  }
}

// MARK: - Function Signature

extension AttentionKernel {
  func createConstants() -> String {
    """

    // R = row dimension (output sequence)
    // C = column dimension (input sequence)
    constant uint R [[function_constant(0)]];
    constant uint C [[function_constant(1)]];

    """
  }

  /// Generate raw `device uchar*` buffer arguments for the visible function.
  func createRawBufferBindings() -> String {
    // All 10 possible buffers, as raw uchar pointers.
    // The shell passes them positionally matching buffer indices 0-9.
    var output = ""
    let bufferNames = ["buf0", "buf1", "buf2", "buf3", "buf4",
                       "buf5", "buf6", "buf7", "buf8", "buf9"]
    for name in bufferNames {
      output += "  device uchar *\(name),\n"
    }
    return output
  }

  /// Generate typed pointer casts from raw buffers to typed operand pointers.
  func createTypedPointerCasts() -> String {
    var operands: [AttentionOperand] = []
    switch type {
    case .forward:
      operands += [.Q, .K, .V, .O, .L]
    case .backwardQuery:
      operands += [.Q, .K, .V, .O, .dO, .dQ, .L, .D]
    case .backwardKeyValue:
      operands += [.Q, .K, .V, .dO, .dV, .dK, .L, .D]
    }

    var output = ""
    for operand in operands {
      guard let binding = operand.bufferBinding else { continue }
      output += "auto \(operand) = (device \(memoryName(operand))*)(buf\(binding));\n"
    }
    return output
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
    case .backwardQuery:
      operands += [.Q, .K, .V, .O]
      operands += [.dO, .dQ]
      operands += [.L, .D]
    case .backwardKeyValue:
      operands += [.Q, .K, .V]
      operands += [.dO, .dV, .dK]
      operands += [.L, .D]
    }
    operands.sort {
      $0.bufferBinding! < $1.bufferBinding!
    }

    var output: String = ""
    for operand in operands {
      var line = "device \(memoryName(operand))* \(operand) "
      line += "[[buffer(\(operand.bufferBinding!))]],"
      output += "  " + line + "\n"
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
    accumulateDesc.lastIterationScale = "fast::divide(1, l)"
    let PV = accumulate(descriptor: accumulateDesc)

    return """

    // Outer loop over the traversal dimension.
    for (uint c = 0; c < C; c += \(blockDimensions.traversal)) {
      // S = Q * K^T
      \(QKT)
      \(maskAttentionMatrixEdge())

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
