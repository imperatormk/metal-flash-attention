//
//  AttentionDescriptor.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/8/24.
//

import Metal

public struct AttentionDescriptor {
  // Q, K, V, dO - when true, uses FP16 for inputs
  public var lowPrecisionInputs: Bool = false

  // Q, K, V, dO - when true, uses BF16 for inputs (overrides lowPrecisionInputs)
  // BF16 has same exponent range as FP32 so no overflow, but same memory as FP16
  public var useBF16Inputs: Bool = false

  // S, P, L, D, dP, dS
  public var lowPrecisionIntermediates: Bool = false

  // O, dV, dK, dQ - when true, outputs are FP16 (forward) or BF16 (backward)
  // Default is false for compatibility - outputs are always FP32
  // Set to true for memory efficiency when input precision matches output needs
  public var lowPrecisionOutputs: Bool = false

  // O - when true, output O uses BF16 instead of FP16 (only applies when lowPrecisionOutputs=true)
  public var useBF16Outputs: Bool = false

  // Causal masking - when true, applies lower triangular mask to attention
  // This prevents attending to future tokens (row index < column index is masked)
  public var causal: Bool = false

  // External attention mask - when true, expects a boolean mask buffer
  // Mask shape: [seq_q, seq_k] where true = attend, false = mask out (-inf)
  public var hasMask: Bool = false

  // Additive attention bias - when true, expects a float bias buffer
  // Shape: [batch, num_heads, seq_q, seq_k] - added to attention scores before softmax
  // This is used for relative position bias in Swin Transformer, ALiBi, etc.
  public var hasAttnBias: Bool = false

  // Stride for attention bias batch dimension (0 = broadcast across batch)
  // If > 0, bias is indexed as: bias[batch_idx * biasBatchStride + head * seq_q * seq_k + ...]
  public var biasBatchStride: UInt32 = 0

  // Stride for attention bias head dimension (0 = broadcast across heads)
  public var biasHeadStride: UInt32 = 0

  // Number of unique bias patterns that repeat across batch dimension
  // 0 = no repeat (use biasBatchStride), >0 = bias pattern repeats every biasRepeatCount batches
  // Used for window attention where nW window patterns repeat for batch_per_window images
  public var biasRepeatCount: UInt32 = 0

  // Sliding window attention - each token only attends to windowSize previous tokens
  // nil or 0 = full attention (default), >0 = sliding window of that size
  // Used by Mistral, Llama 3.2, and other efficient attention models
  public var windowSize: UInt32? = nil

  // Quantized K/V cache support for memory-efficient inference
  // When set, K and V are stored in the specified quantized format with per-head scales
  // Q remains in standard precision for computation accuracy
  // Supported formats: .FP8_E4M3, .FP8_E5M2, .INT8, .NF4
  // Set to nil (default) for standard precision K/V
  public var quantizedKV: GEMMOperandPrecision? = nil

  // row:    Output sequence length; rows of the attention matrix.
  // column: Input sequence length; columns of the attention matrix.
  // head:   Head dimension, typically 32 - 256.
  public var matrixDimensions: (row: UInt32, column: UInt32, head: UInt16)?

  public var transposeState: (Q: Bool, K: Bool, V: Bool, O: Bool)?

  public init() {

  }
}

extension AttentionDescriptor {
  /// Initialize the kernel descriptor using another descriptor, which just
  /// specifies the problem size. Then, forget the information about problem
  /// size.
  public func kernelDescriptor(
    type: AttentionKernelType
  ) -> AttentionKernelDescriptor {
    // Fetch the kernel-specific parameters.
    let file = parameterFile(type: type)
    let table = AttentionParameterRow.parseTable(file)
    let row = row(table: table)
    
    func createBlockDimensions() -> (UInt16, UInt16, UInt16) {
      guard let parallelization = UInt16(row.parallelization),
            let traversal = UInt16(row.traversal),
            let originalHead = UInt16(row.head) else {
        fatalError("Could not decode block dimensions.")
      }
      
      // Enforce the rule that head block dimension <= head dimension.
      let headDimension = createHeadDimension()
      let paddedHeadDimension = (headDimension + 7) / 8 * 8
      let revisedHead = min(originalHead, paddedHeadDimension)
      
      return (parallelization, traversal, revisedHead)
    }
    
    func createCacheState() -> [AttentionOperand: Bool] {
      var expectedOperands: Set<AttentionOperand>
      switch type {
      case .forward:
        expectedOperands = [.Q, .O]
      case .backwardQuery:
        expectedOperands = [.Q, .dO, .dQ]
      case .backwardKeyValue:
        expectedOperands = [.K, .V, .dV, .dK]
      }
      
      // Check for unexpected operands.
      let cachedOperands = AttentionParameterRow
        .parseOperands(row.cachedOperands)
      for operand in cachedOperands {
        guard expectedOperands.contains(operand) else {
          fatalError("Unexpected operand: \(operand)")
        }
      }
      
      // Convert the list into a dictionary.
      var output: [AttentionOperand: Bool] = [:]
      for operand in expectedOperands {
        output[operand] = false
      }
      for operand in cachedOperands {
        output[operand] = true
      }
      
      return output
    }
    
    func createHeadDimension() -> UInt16 {
      guard let matrixDimensions = self.matrixDimensions else {
        fatalError("Descriptor was incomplete.")
      }
      return matrixDimensions.head
    }
    
    func createTransposeState() -> [AttentionOperand: Bool] {
      guard let transposeState = self.transposeState else {
        fatalError("Descriptor was incomplete.")
      }
      
      var output: [AttentionOperand: Bool] = [:]
      output[.Q] = transposeState.Q
      output[.K] = transposeState.K
      output[.V] = transposeState.V
      output[.O] = transposeState.O
      
      output[.dO] = transposeState.O
      output[.dV] = transposeState.V
      output[.dK] = transposeState.K
      output[.dQ] = transposeState.Q
      return output
    }
    
    var output = AttentionKernelDescriptor()
    output.blockDimensions = createBlockDimensions()
    output.cacheState = createCacheState()
    output.headDimension = createHeadDimension()
    output.memoryPrecisions = memoryPrecisions
    if MTLContext.global.device.supportsFamily(.apple9) {
      output.preferAsyncCache = true
      output.preferAsyncLoad = false
    } else {
      output.preferAsyncCache = false
      output.preferAsyncLoad = true
    }
    output.registerPrecisions = registerPrecisions
    output.transposeState = createTransposeState()
    output.type = type
    output.causal = causal
    output.hasMask = hasMask
    output.hasAttnBias = hasAttnBias
    output.biasBatchStride = biasBatchStride
    output.biasHeadStride = biasHeadStride
    output.biasRepeatCount = biasRepeatCount
    output.windowSize = windowSize
    output.quantizedKV = quantizedKV

    return output
  }
}

extension AttentionDescriptor {
  // Specialize the Metal function with this attention descriptor.
  //
  // You can initialize a MTLFunctionConstantValues object once, then recycle
  // it for all three kernels when gradient is requested. This may simplify
  // the code or incrementally reduce the compilation latency.
  public func setFunctionConstants(_ constants: MTLFunctionConstantValues) {
    guard let matrixDimensions = self.matrixDimensions else {
      fatalError("Descriptor was incomplete.")
    }

    var rowDimension = matrixDimensions.row
    var columnDimension = matrixDimensions.column
    constants.setConstantValue(&rowDimension, type: .uint, index: 0)
    constants.setConstantValue(&columnDimension, type: .uint, index: 1)
  }
}
