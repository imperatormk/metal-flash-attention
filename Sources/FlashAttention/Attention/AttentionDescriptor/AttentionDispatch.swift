//
//  AttentionDispatch.swift
//  FlashAttention
//
//  High-level dispatch utilities for multi-head attention.
//  Handles pipeline lookup, BatchedParams buffer creation, and dispatch.
//

import Metal

/// Dispatch flash attention forward pass for multi-head (optionally causal) attention.
///
/// Buffer layout: each operand is packed as [numHeads, seqLen, D] contiguously.
/// For single-head: pass numHeads=1, buffers are just [seqLen, D].
///
/// - Parameters:
///   - encoder: Active compute command encoder with buffers 0-9 already bound.
///   - Q: Buffer at index 0, shape [numHeads * R * D] elements.
///   - K: Buffer at index 1, shape [numHeads * C * D] elements.
///   - V: Buffer at index 2, shape [numHeads * C * D] elements.
///   - O: Buffer at index 3, shape [numHeads * R * D] elements (output).
///   - L: Buffer at index 4, shape [numHeads * R] elements (logsumexp output).
///   - R: Row sequence length (query sequence length).
///   - C: Column sequence length (key/value sequence length).
///   - D: Head dimension.
///   - numHeads: Number of attention heads.
///   - batchSize: Batch dimension (default 1).
///   - causal: Whether to apply causal masking.
///   - lowPrecision: Whether to use FP16 inputs/intermediates.
public struct FlashAttentionDispatch {

  public static func forward(
    encoder: MTLComputeCommandEncoder,
    R: Int, C: Int, D: Int,
    numHeads: Int = 1,
    numKVHeads: Int? = nil,
    batchSize: Int = 1,
    causal: Bool = false,
    causalOffset: Int = 0,
    lowPrecision: Bool = false
  ) {
    var attDesc = AttentionDescriptor()
    attDesc.lowPrecisionInputs = lowPrecision
    attDesc.lowPrecisionIntermediates = lowPrecision
    attDesc.matrixDimensions = (
      row: UInt32(R), column: UInt32(C), head: UInt16(D))
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)
    attDesc.causal = causal

    let (kernel, pipeline) = AttentionKernel.pipeline(for: attDesc, type: .forward)

    let batchParams = AttentionKernel.createBatchedParamsBuffer(
      numHeads: UInt32(numHeads),
      numKVHeads: numKVHeads.map { UInt32($0) },
      R: UInt32(R), C: UInt32(C), D: UInt32(D),
      causalOffset: UInt32(causalOffset))

    AttentionKernel.dispatch(
      encoder: encoder,
      kernel: kernel,
      pipeline: pipeline,
      batchedParams: batchParams,
      parallelizationDimension: R,
      numHeads: numHeads,
      batchSize: batchSize)
  }

  /// Convenience: create dummy buffers for unused backward operands.
  /// Returns buffers for indices 5 (D_buf), 6 (dO), 7 (dV), 8 (dK), 9 (dQ).
  public static func createDummyBackwardBuffers() -> [MTLBuffer] {
    let device = MTLContext.global.device
    let dummy = device.makeBuffer(length: 4, options: .storageModeShared)!
    return [dummy, dummy, dummy, dummy, dummy]
  }

  /// Bind all buffers for a forward-only dispatch.
  /// Q at 0, K at 1, V at 2, O at 3, L at 4, dummies at 5-9.
  public static func bindForwardBuffers(
    encoder: MTLComputeCommandEncoder,
    Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer,
    O: MTLBuffer, L: MTLBuffer
  ) {
    encoder.setBuffer(Q, offset: 0, index: 0)
    encoder.setBuffer(K, offset: 0, index: 1)
    encoder.setBuffer(V, offset: 0, index: 2)
    encoder.setBuffer(O, offset: 0, index: 3)
    encoder.setBuffer(L, offset: 0, index: 4)
    let dummies = createDummyBackwardBuffers()
    for (i, buf) in dummies.enumerated() {
      encoder.setBuffer(buf, offset: 0, index: 5 + i)
    }
  }
}
