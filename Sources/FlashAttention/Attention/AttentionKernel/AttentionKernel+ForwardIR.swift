//
//  AttentionKernel+ForwardIR.swift
//  FlashAttention
//
//  Forward attention kernel IR generation.
//  S = Q * K^T, softmax, O += P * V, O /= l, store L.
//

extension AttentionKernel {

  func generateForwardKernel(
    desc: MonolithicDescriptor,
    R: UInt32, C: UInt32, D: UInt32,
    blockP: UInt16, blockT: UInt16, blockH: UInt16,
    parallelDim: UInt32, traversalDim: UInt32,
    paddedD: UInt32, headEdge: UInt32,
    headLoopFloor: UInt32,
    oCachedCount: Int, sSramCount: Int,
    scaleFactor: Float, tgFloats: Int,
    leadingDim: (AttentionOperand) -> UInt32,
    leadingBlockDim: (AttentionOperand) -> UInt32,
    memPrec: (AttentionOperand) -> GEMMOperandPrecision,
    regPrec: (AttentionOperand) -> GEMMOperandPrecision,
    memSize: (AttentionOperand) -> UInt32
  ) -> String {
    var ir = ""

    let regS = regPrec(.S)
    let regP = regPrec(.P)
    let regO = regPrec(.O)
    let regQ = regPrec(.Q)
    let regK = regPrec(.K)
    let regV = regPrec(.V)
    let memQ = memPrec(.Q)
    let memK = memPrec(.K)
    let memV = memPrec(.V)
    let memO = memPrec(.O)
    let memL = memPrec(.L)

    let isCachedQ = cached(.Q)
    let isCachedO = cached(.O)

    // Number of O accumulators
    let oCount: Int
    if isCachedO {
      oCount = oCachedCount  // paddedD / 8
    } else {
      oCount = Int(blockH / 8)
    }

    // MARK: - Setup

    // Initialize O accumulators to zero
    ir += "  ; === Forward setup ===\n"
    for i in 0..<oCachedCount {
      ir += "  \(irZeroVec64(result: "%o_init_\(i)", precision: regO))\n"
    }
    ir += "  ; m = -MAX, l = denorm_min\n"
    ir += "  %m_init = bitcast float 0xFFF0000000000000 to float ; -inf\n"
    // denorm_min ≈ 1.4e-45
    ir += "  %l_init = bitcast float 0x36A0000000000000 to float ; denorm_min\n"

    // Cache load Q if cached
    if isCachedQ {
      ir += generateCacheLoad(
        operand: .Q, prefix: "cq_",
        parallelDim: parallelDim, D: D, paddedD: paddedD,
        blockP: blockP, blockH: blockH,
        leadingDim: leadingDim(.Q), leadingBlockDim: leadingBlockDim(.Q),
        memPrec: memQ, regPrec: regQ,
        transposed: transposed(.Q)
      )
    }

    // MARK: - Traversal Loop

    // Use a dedicated label so phi predecessors are correct even when
    // cache-load inserts extra basic blocks between valid_group and here.
    ir += """

      ; === Traversal loop ===
      br label %before_loop
    before_loop:
      br label %loop_header

    loop_header:
      %c = phi i32 [0, %before_loop], [%c_next, %loop_latch]
      %m_phi = phi float [%m_init, %before_loop], [%m_updated, %loop_latch]
      %l_phi = phi float [%l_init, %before_loop], [%l_updated, %loop_latch]

    """

    // Phi nodes for O accumulators
    for i in 0..<oCachedCount {
      ir += "  %o_phi_\(i) = phi \(irVecType(regO)) [%o_init_\(i), %before_loop], [%o_acc_\(i), %loop_latch]\n"
    }

    ir += """

      %loop_done = icmp uge i32 %c, \(traversalDim)
      br i1 %loop_done, label %cleanup, label %loop_body

    loop_body:

    """

    // MARK: - Outer Product (S = Q * K^T)

    // Initialize S accumulators to zero
    for i in 0..<sSramCount {
      ir += "  \(irZeroVec64(result: "%s_init_\(i)", precision: regS))\n"
    }

    // Loop over head dimension (d_outer)
    // For each d_outer block, we:
    //   1. Load K from device → TG (async copy)
    //   2. Barrier
    //   3. Load Q (from cached registers or device/TG)
    //   4. Multiply: S += Q_block * K_block^T
    ir += generateOuterProduct(
      prefix: "op_",
      A: .Q, B: .K, C_name: "s",
      sSramCount: sSramCount,
      blockP: blockP, blockT: blockT, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim, traversalDim: traversalDim,
      traversalOffset: "%c",
      regA: regQ, regB: regK, regC: regS,
      memA: memQ, memB: memK,
      leadingDimA: leadingDim(.Q), leadingDimB: leadingDim(.K),
      leadingBlockDimA: leadingBlockDim(.Q), leadingBlockDimB: leadingBlockDim(.K),
      cachedA: isCachedQ, transposedA: transposed(.Q), transposedB: transposed(.K)
    )

    // MARK: - Mask Attention Matrix Edge

    ir += generateMaskEdge(
      prefix: "mask_",
      sSramCount: sSramCount,
      blockT: blockT, traversalDim: traversalDim,
      traversalOffset: "%c",
      regS: regS, scaleFactor: scaleFactor
    )

    // MARK: - Online Softmax: Reduce Maximum

    ir += generateReduceMax(
      prefix: "rmax_",
      sSramCount: sSramCount,
      regS: regS, scaleFactor: scaleFactor
    )

    // MARK: - Online Softmax: Correct O

    ir += generateCorrectO(
      prefix: "corr_",
      oCachedCount: oCachedCount,
      regO: regO
    )

    // MARK: - Online Softmax: Compute P = exp2(S * scale - m)

    ir += generateComputeP(
      prefix: "sp_",
      sSramCount: sSramCount,
      regS: regS, regP: regP,
      scaleFactor: scaleFactor
    )

    // MARK: - Online Softmax: Reduce Sum

    ir += generateReduceSum(
      prefix: "rsum_",
      sSramCount: sSramCount,
      regP: regP
    )

    // MARK: - Accumulate O += P * V

    ir += generateAccumulate(
      prefix: "acc_",
      A: .P, B: .V, C_name: "o",
      accCount: oCachedCount,
      blockP: blockP, blockT: blockT, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim, traversalDim: traversalDim,
      traversalOffset: "%c",
      regA: regP, regB: regV, regC: regO,
      memB: memPrec(.V),
      leadingDimB: leadingDim(.V),
      leadingBlockDimB: leadingBlockDim(.V),
      transposedB: transposed(.V),
      cachedC: isCachedO,
      isFinalScale: false,
      scaleCorrection: "corr_correction"
    )

    // MARK: - Loop Latch

    // Terminate acc_after_head block before loop_latch
    ir += "  br label %loop_latch\n"
    ir += """

    loop_latch:
      %m_updated = phi float [%corr_m_upd, %\(isCachedO ? "acc_after_head" : "acc_after_head")]
      %l_updated = phi float [%rsum_l_new, %\(isCachedO ? "acc_after_head" : "acc_after_head")]

    """

    for i in 0..<oCachedCount {
      ir += "  %o_acc_\(i) = phi \(irVecType(regO)) [%acc_o_final_\(i), %\(isCachedO ? "acc_after_head" : "acc_after_head")]\n"
    }

    ir += """

      %c_next = add i32 %c, \(blockT)
      br label %loop_header

    """

    // MARK: - Cleanup: O /= l, store O, store L

    ir += generateForwardCleanup(
      prefix: "cl_",
      oCachedCount: oCachedCount,
      blockP: blockP, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim,
      regO: regO, memO: memO, memL: memL,
      leadingDimO: leadingDim(.O),
      leadingBlockDimO: leadingBlockDim(.O),
      transposedO: transposed(.O),
      cachedO: isCachedO
    )

    return ir
  }
}
