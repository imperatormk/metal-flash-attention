//
//  AttentionKernel+BackwardIR.swift
//  FlashAttention
//
//  Backward attention kernel IR generation stubs.
//  TODO: Implement backward_query and backward_keyvalue.
//

extension AttentionKernel {

  func generateBackwardQueryKernel(
    desc: MonolithicDescriptor,
    R: UInt32, C: UInt32, D: UInt32,
    blockP: UInt16, blockT: UInt16, blockH: UInt16,
    parallelDim: UInt32, traversalDim: UInt32,
    paddedD: UInt32, headEdge: UInt32,
    headLoopFloor: UInt32,
    sSramCount: Int,
    scaleFactor: Float, tgFloats: Int,
    leadingDim: (AttentionOperand) -> UInt32,
    leadingBlockDim: (AttentionOperand) -> UInt32,
    memPrec: (AttentionOperand) -> GEMMOperandPrecision,
    regPrec: (AttentionOperand) -> GEMMOperandPrecision,
    memSize: (AttentionOperand) -> UInt32
  ) -> String {
    // Stub: just jump to exit
    return """

      br label %exit

    """
  }

  func generateBackwardKeyValueKernel(
    desc: MonolithicDescriptor,
    R: UInt32, C: UInt32, D: UInt32,
    blockP: UInt16, blockT: UInt16, blockH: UInt16,
    parallelDim: UInt32, traversalDim: UInt32,
    paddedD: UInt32, headEdge: UInt32,
    headLoopFloor: UInt32,
    sSramCount: Int,
    scaleFactor: Float, tgFloats: Int,
    leadingDim: (AttentionOperand) -> UInt32,
    leadingBlockDim: (AttentionOperand) -> UInt32,
    memPrec: (AttentionOperand) -> GEMMOperandPrecision,
    regPrec: (AttentionOperand) -> GEMMOperandPrecision,
    memSize: (AttentionOperand) -> UInt32
  ) -> String {
    // Stub: just jump to exit
    return """

      br label %exit

    """
  }
}
