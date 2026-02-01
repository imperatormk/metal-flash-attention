//
//  AttentionKernelDescriptor.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/28/24.
//

public struct AttentionKernelDescriptor {
  public var blockDimensions: (
    parallelization: UInt16, traversal: UInt16, head: UInt16)?

  /// Whether each operand is cached in registers.
  public var cacheState: [AttentionOperand: Bool] = [:]

  /// Required. The problem size along the head dimension.
  public var headDimension: UInt16?

  public var memoryPrecisions: [AttentionOperand: GEMMOperandPrecision] = [:]

  /// Reads with a one-to-one mapping to threads (like GEMM store) and writes.
  public var preferAsyncCache: Bool?

  /// Reads that are shared among threads (like GEMM load).
  public var preferAsyncLoad: Bool?

  public var registerPrecisions: [AttentionOperand: GEMMOperandPrecision] = [:]

  /// Whether each operand is transposed in RAM.
  ///
  /// If the layout is row-major, where a row spans D contiguous elements in
  /// memory, enter `false`. If the layout is column-major, where a row spans
  /// D widely separated elements in memory, enter `true`.
  ///
  /// The transpose state of a derivative (e.g. dQ for Q) must match the
  /// corresponding input from the forward pass.
  ///
  /// > NOTE: To implement multi-head attention, clients may need to modify
  /// the stride of matrix elements in memory. If and only if the transpose
  /// state is `false`, change the stride from `D` to `D * H`. Ensure the
  /// value of H is known at compile time, so the product `D * H` can be
  /// embedded into the GPU assembly code.
  public var transposeState: [AttentionOperand: Bool] = [:]

  public var type: AttentionKernelType?

  /// Causal masking - applies lower triangular mask to prevent attending to future tokens
  public var causal: Bool = false

  /// External attention mask - when true, expects a boolean mask buffer
  public var hasMask: Bool = false

  /// Additive attention bias - when true, expects a float bias buffer
  /// Used for relative position bias in Swin Transformer, ALiBi, etc.
  public var hasAttnBias: Bool = false

  /// Strides for attention bias broadcasting
  /// biasBatchStride: 0 = broadcast across batch, else stride for batch dim
  /// biasHeadStride: 0 = broadcast across heads, else stride for head dim
  public var biasBatchStride: UInt32 = 0
  public var biasHeadStride: UInt32 = 0

  /// Number of unique bias patterns that repeat across batch
  /// 0 = no repeat, >0 = pattern repeats (batch_idx % biasRepeatCount gives pattern index)
  public var biasRepeatCount: UInt32 = 0

  /// Sliding window attention size - if set, each token only attends to windowSize previous tokens
  /// This enables efficient attention for models like Mistral and Llama 3.2
  /// nil = full attention (default), 0 = full attention, >0 = sliding window of that size
  public var windowSize: UInt32? = nil

  /// Quantized K/V precision for memory-efficient inference
  /// When set to a quantized format (FP8_E4M3, FP8_E5M2, INT8, NF4),
  /// K and V are loaded from quantized buffers with per-head scales
  /// nil = standard precision (default)
  public var quantizedKV: GEMMOperandPrecision? = nil

  public init() {

  }
}
