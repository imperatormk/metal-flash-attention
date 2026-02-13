import XCTest
import FlashAttention
import MetalASM

/// Tests for external attention mask and additive bias features.
final class MaskBiasAttentionTest: XCTestCase {

  /// Test that external mask correctly zeros out masked positions.
  func testExternalMask() throws {
    // Small problem: 16x16, D=32
    let seq = 16
    let D = 32
    let headDim = D

    var networkDesc = NetworkDescriptor()
    networkDesc.rowDimension = seq
    networkDesc.columnDimension = seq
    networkDesc.headDimension = headDim
    let network = Network(descriptor: networkDesc)

    // Mask upper-right triangle (like causal but via external mask)
    // mask[r, c] = 1.0 (masked) when c > r
    var maskF = [Float](repeating: 0, count: seq * seq)
    var maskU = [UInt8](repeating: 0, count: seq * seq)
    for r in 0..<seq {
      for c in 0..<seq {
        if c > r {
          maskF[r * seq + c] = 1.0
          maskU[r * seq + c] = 1
        }
      }
    }

    let expectedO = referenceAttentionWithMask(network: network, mask: maskU, seq: seq, D: headDim)

    // GPU: compile with hasMask = true
    var attDesc = AttentionDescriptor()
    attDesc.lowPrecisionInputs = false
    attDesc.lowPrecisionIntermediates = false
    attDesc.hasMask = true
    attDesc.matrixDimensions = (row: UInt32(seq), column: UInt32(seq), head: UInt16(D))
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)

    let kernelDesc = attDesc.kernelDescriptor(type: .forward)
    let kernel = AttentionKernel(descriptor: kernelDesc)

    var monoDesc = AttentionKernel.MonolithicDescriptor()
    monoDesc.R = UInt32(seq)
    monoDesc.C = UInt32(seq)
    let d32 = UInt32(D)
    monoDesc.leadingDimensions[.Q] = d32
    monoDesc.leadingDimensions[.K] = d32
    monoDesc.leadingDimensions[.V] = d32
    monoDesc.leadingDimensions[.O] = d32

    let ir = kernel.createSource(descriptor: monoDesc)
    try ir.write(toFile: "/tmp/mask_ir_debug.ll", atomically: true, encoding: .utf8)
    #if os(macOS)
    let metallibData = try MetalASM.assemble(ir: ir, platform: .macOS(version: 26))
    #elseif os(iOS)
    let metallibData = try MetalASM.assemble(ir: ir, platform: .iOS(version: 26))
    #endif

    let device = MTLContext.global.device
    let dispatchData = metallibData.withUnsafeBytes { DispatchData(bytes: $0) }
    let library = try device.makeLibrary(data: dispatchData)
    let function = library.makeFunction(name: "attention")!
    let pipeline = try device.makeComputePipelineState(function: function)

    let memPrec = attDesc.memoryPrecisions
    let bufQ = MTLContext.global.createBuffer(network.Q, memPrec[.Q]!)
    let bufK = MTLContext.global.createBuffer(network.K, memPrec[.K]!)
    let bufV = MTLContext.global.createBuffer(network.V, memPrec[.V]!)
    var resultO = [Float](repeating: .nan, count: seq * headDim)
    let bufO = MTLContext.global.createBuffer(resultO, memPrec[.O]!)
    let resultL = [Float](repeating: 0, count: seq)
    let bufL = MTLContext.global.createBuffer(resultL, memPrec[.L]!)
    let dummy = device.makeBuffer(length: 4, options: .storageModeShared)!

    // Mask buffer (uint8)
    let bufMask = device.makeBuffer(bytes: maskU, length: maskU.count, options: .storageModeShared)!

    // BatchedParams
    var bp: [UInt32] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, UInt32(seq), UInt32(seq)]
    let bufBP = device.makeBuffer(bytes: &bp, length: bp.count * 4, options: .storageModeShared)!

    let cmdBuf = MTLContext.global.commandQueue.makeCommandBuffer()!
    let enc = cmdBuf.makeComputeCommandEncoder()!

    // Bind all 13 device buffers: Q(0) K(1) V(2) O(3) L(4) D(5) dO(6) dV(7) dK(8) dQ(9) mask(10) bias(11) bp(30)
    for i in 0..<10 { enc.setBuffer(dummy, offset: 0, index: i) }
    enc.setBuffer(bufQ, offset: 0, index: 0)
    enc.setBuffer(bufK, offset: 0, index: 1)
    enc.setBuffer(bufV, offset: 0, index: 2)
    enc.setBuffer(bufO, offset: 0, index: 3)
    enc.setBuffer(bufL, offset: 0, index: 4)
    enc.setBuffer(bufMask, offset: 0, index: 10)
    enc.setBuffer(dummy, offset: 0, index: 11)  // bias (unused)
    enc.setBuffer(bufBP, offset: 0, index: 30)

    enc.setComputePipelineState(pipeline)
    enc.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)

    let blockCount = (seq + Int(kernel.blockDimensions.parallelization) - 1)
      / Int(kernel.blockDimensions.parallelization)
    enc.dispatchThreadgroups(
      MTLSize(width: blockCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1))

    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    MTLContext.copy(bufO, into: &resultO, precision: memPrec[.O]!)

    var errorCount = 0
    for i in 0..<(seq * headDim) {
      let err = abs(expectedO[i] - resultO[i])
      if err > 2e-5 || err.isNaN {
        if (expectedO[i].isNaN || expectedO[i].isInfinite),
           (resultO[i].isNaN || resultO[i].isInfinite) { continue }
        errorCount += 1
        if errorCount <= 5 {
          print("  mask: error[\(i)] = \(err), expected=\(expectedO[i]), got=\(resultO[i])")
        }
      }
    }
    XCTAssertEqual(errorCount, 0, "External mask test failed: \(errorCount) errors")
  }

  /// Test that additive bias is correctly applied before scaling.
  func testAdditiveBias() throws {
    let seq = 16
    let D = 32
    let headDim = D

    var networkDesc = NetworkDescriptor()
    networkDesc.rowDimension = seq
    networkDesc.columnDimension = seq
    networkDesc.headDimension = headDim
    let network = Network(descriptor: networkDesc)

    // Create bias: small random-ish values
    var bias = [Float](repeating: 0, count: seq * seq)
    for r in 0..<seq {
      for c in 0..<seq {
        // Deterministic pattern: bias[r,c] = sin(r*7 + c*13) * 0.5
        let val = sin(Float(r * 7 + c * 13)) * 0.5
        bias[r * seq + c] = val
      }
    }

    let expectedO = referenceAttentionWithBias(network: network, bias: bias, seq: seq, D: headDim)

    var attDesc = AttentionDescriptor()
    attDesc.lowPrecisionInputs = false
    attDesc.lowPrecisionIntermediates = false
    attDesc.hasAttnBias = true
    attDesc.matrixDimensions = (row: UInt32(seq), column: UInt32(seq), head: UInt16(D))
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)

    let kernelDesc = attDesc.kernelDescriptor(type: .forward)
    let kernel = AttentionKernel(descriptor: kernelDesc)

    var monoDesc = AttentionKernel.MonolithicDescriptor()
    monoDesc.R = UInt32(seq)
    monoDesc.C = UInt32(seq)
    let d32 = UInt32(D)
    monoDesc.leadingDimensions[.Q] = d32
    monoDesc.leadingDimensions[.K] = d32
    monoDesc.leadingDimensions[.V] = d32
    monoDesc.leadingDimensions[.O] = d32

    let ir = kernel.createSource(descriptor: monoDesc)
    #if os(macOS)
    let metallibData = try MetalASM.assemble(ir: ir, platform: .macOS(version: 26))
    #elseif os(iOS)
    let metallibData = try MetalASM.assemble(ir: ir, platform: .iOS(version: 26))
    #endif

    let device = MTLContext.global.device
    let dispatchData = metallibData.withUnsafeBytes { DispatchData(bytes: $0) }
    let library = try device.makeLibrary(data: dispatchData)
    let function = library.makeFunction(name: "attention")!
    let pipeline = try device.makeComputePipelineState(function: function)

    let memPrec = attDesc.memoryPrecisions
    let bufQ = MTLContext.global.createBuffer(network.Q, memPrec[.Q]!)
    let bufK = MTLContext.global.createBuffer(network.K, memPrec[.K]!)
    let bufV = MTLContext.global.createBuffer(network.V, memPrec[.V]!)
    var resultO = [Float](repeating: .nan, count: seq * headDim)
    let bufO = MTLContext.global.createBuffer(resultO, memPrec[.O]!)
    let resultL = [Float](repeating: 0, count: seq)
    let bufL = MTLContext.global.createBuffer(resultL, memPrec[.L]!)
    let dummy = device.makeBuffer(length: 4, options: .storageModeShared)!

    let bufBias = device.makeBuffer(bytes: bias, length: bias.count * 4, options: .storageModeShared)!

    var bp: [UInt32] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, UInt32(seq), UInt32(seq)]
    let bufBP = device.makeBuffer(bytes: &bp, length: bp.count * 4, options: .storageModeShared)!

    let cmdBuf = MTLContext.global.commandQueue.makeCommandBuffer()!
    let enc = cmdBuf.makeComputeCommandEncoder()!

    for i in 0..<10 { enc.setBuffer(dummy, offset: 0, index: i) }
    enc.setBuffer(bufQ, offset: 0, index: 0)
    enc.setBuffer(bufK, offset: 0, index: 1)
    enc.setBuffer(bufV, offset: 0, index: 2)
    enc.setBuffer(bufO, offset: 0, index: 3)
    enc.setBuffer(bufL, offset: 0, index: 4)
    enc.setBuffer(dummy, offset: 0, index: 10)  // mask (unused)
    enc.setBuffer(bufBias, offset: 0, index: 11)
    enc.setBuffer(bufBP, offset: 0, index: 30)

    enc.setComputePipelineState(pipeline)
    enc.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)

    let blockCount = (seq + Int(kernel.blockDimensions.parallelization) - 1)
      / Int(kernel.blockDimensions.parallelization)
    enc.dispatchThreadgroups(
      MTLSize(width: blockCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1))

    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    MTLContext.copy(bufO, into: &resultO, precision: memPrec[.O]!)

    var errorCount = 0
    for i in 0..<(seq * headDim) {
      let err = abs(expectedO[i] - resultO[i])
      if err > 2e-5 || err.isNaN {
        if (expectedO[i].isNaN || expectedO[i].isInfinite),
           (resultO[i].isNaN || resultO[i].isInfinite) { continue }
        errorCount += 1
        if errorCount <= 5 {
          print("  bias: error[\(i)] = \(err), expected=\(expectedO[i]), got=\(resultO[i])")
        }
      }
    }
    XCTAssertEqual(errorCount, 0, "Additive bias test failed: \(errorCount) errors")
  }

  /// Test sliding window attention (causal + window).
  func testSlidingWindow() throws {
    let seq = 32
    let D = 32
    let windowSize: UInt32 = 8

    var networkDesc = NetworkDescriptor()
    networkDesc.rowDimension = seq
    networkDesc.columnDimension = seq
    networkDesc.headDimension = D
    let network = Network(descriptor: networkDesc)

    let expectedO = referenceAttentionSlidingWindow(
      network: network, seq: seq, D: D, windowSize: Int(windowSize))

    var attDesc = AttentionDescriptor()
    attDesc.lowPrecisionInputs = false
    attDesc.lowPrecisionIntermediates = false
    attDesc.causal = true
    attDesc.windowSize = windowSize
    attDesc.matrixDimensions = (row: UInt32(seq), column: UInt32(seq), head: UInt16(D))
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)

    let kernelDesc = attDesc.kernelDescriptor(type: .forward)
    let kernel = AttentionKernel(descriptor: kernelDesc)

    var monoDesc = AttentionKernel.MonolithicDescriptor()
    monoDesc.R = UInt32(seq)
    monoDesc.C = UInt32(seq)
    let d32 = UInt32(D)
    monoDesc.leadingDimensions[.Q] = d32
    monoDesc.leadingDimensions[.K] = d32
    monoDesc.leadingDimensions[.V] = d32
    monoDesc.leadingDimensions[.O] = d32

    let ir = kernel.createSource(descriptor: monoDesc)
    #if os(macOS)
    let metallibData = try MetalASM.assemble(ir: ir, platform: .macOS(version: 26))
    #elseif os(iOS)
    let metallibData = try MetalASM.assemble(ir: ir, platform: .iOS(version: 26))
    #endif

    let device = MTLContext.global.device
    let dispatchData = metallibData.withUnsafeBytes { DispatchData(bytes: $0) }
    let library = try device.makeLibrary(data: dispatchData)
    let function = library.makeFunction(name: "attention")!
    let pipeline = try device.makeComputePipelineState(function: function)

    let memPrec = attDesc.memoryPrecisions
    let bufQ = MTLContext.global.createBuffer(network.Q, memPrec[.Q]!)
    let bufK = MTLContext.global.createBuffer(network.K, memPrec[.K]!)
    let bufV = MTLContext.global.createBuffer(network.V, memPrec[.V]!)
    var resultO = [Float](repeating: .nan, count: seq * D)
    let bufO = MTLContext.global.createBuffer(resultO, memPrec[.O]!)
    let resultL = [Float](repeating: 0, count: seq)
    let bufL = MTLContext.global.createBuffer(resultL, memPrec[.L]!)
    let dummy = device.makeBuffer(length: 4, options: .storageModeShared)!

    var bp: [UInt32] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, UInt32(seq), UInt32(seq)]
    let bufBP = device.makeBuffer(bytes: &bp, length: bp.count * 4, options: .storageModeShared)!

    let cmdBuf = MTLContext.global.commandQueue.makeCommandBuffer()!
    let enc = cmdBuf.makeComputeCommandEncoder()!

    for i in 0..<10 { enc.setBuffer(dummy, offset: 0, index: i) }
    enc.setBuffer(bufQ, offset: 0, index: 0)
    enc.setBuffer(bufK, offset: 0, index: 1)
    enc.setBuffer(bufV, offset: 0, index: 2)
    enc.setBuffer(bufO, offset: 0, index: 3)
    enc.setBuffer(bufL, offset: 0, index: 4)
    enc.setBuffer(dummy, offset: 0, index: 10)
    enc.setBuffer(dummy, offset: 0, index: 11)
    enc.setBuffer(bufBP, offset: 0, index: 30)

    enc.setComputePipelineState(pipeline)
    enc.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)

    let blockCount = (seq + Int(kernel.blockDimensions.parallelization) - 1)
      / Int(kernel.blockDimensions.parallelization)
    enc.dispatchThreadgroups(
      MTLSize(width: blockCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1))

    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    MTLContext.copy(bufO, into: &resultO, precision: memPrec[.O]!)

    var errorCount = 0
    for i in 0..<(seq * D) {
      let err = abs(expectedO[i] - resultO[i])
      if err > 2e-5 || err.isNaN {
        if (expectedO[i].isNaN || expectedO[i].isInfinite),
           (resultO[i].isNaN || resultO[i].isInfinite) { continue }
        errorCount += 1
        if errorCount <= 5 {
          print("  window: error[\(i)] = \(err), expected=\(expectedO[i]), got=\(resultO[i])")
        }
      }
    }
    XCTAssertEqual(errorCount, 0, "Sliding window test failed: \(errorCount) errors")
  }
}

// MARK: - CPU Reference Implementations

/// CPU reference: attention with external mask (True = masked → -inf).
private func referenceAttentionWithMask(
  network: Network, mask: [UInt8], seq: Int, D: Int
) -> [Float] {
  let scale = 1.0 / Float(D).squareRoot()

  // S = Q * K^T
  var S = [Float](repeating: 0, count: seq * seq)
  for r in 0..<seq {
    for c in 0..<seq {
      var dot: Float = 0
      for d in 0..<D {
        dot += network.Q[r * D + d] * network.K[c * D + d]
      }
      S[r * seq + c] = dot
    }
  }

  // Apply mask: where mask != 0, set to -inf
  for r in 0..<seq {
    for c in 0..<seq {
      if mask[r * seq + c] != 0 {
        S[r * seq + c] = -.infinity
      }
    }
  }

  // Scale + softmax + O = P * V
  return applySoftmaxAndProject(S: S, V: network.V, seq: seq, D: D, scale: scale)
}

/// CPU reference: attention with additive bias (added to UNSCALED scores, then scaled).
/// Formula: softmax((Q @ K^T + bias) * scale) @ V
private func referenceAttentionWithBias(
  network: Network, bias: [Float], seq: Int, D: Int
) -> [Float] {
  let scale = 1.0 / Float(D).squareRoot()

  // S = Q * K^T
  var S = [Float](repeating: 0, count: seq * seq)
  for r in 0..<seq {
    for c in 0..<seq {
      var dot: Float = 0
      for d in 0..<D {
        dot += network.Q[r * D + d] * network.K[c * D + d]
      }
      S[r * seq + c] = dot
    }
  }

  // Add bias (before scaling)
  for i in 0..<(seq * seq) {
    S[i] += bias[i]
  }

  return applySoftmaxAndProject(S: S, V: network.V, seq: seq, D: D, scale: scale)
}

/// CPU reference: causal + sliding window attention.
/// Attend only to positions c where c <= r AND c >= r - windowSize.
private func referenceAttentionSlidingWindow(
  network: Network, seq: Int, D: Int, windowSize: Int
) -> [Float] {
  let scale = 1.0 / Float(D).squareRoot()

  var S = [Float](repeating: 0, count: seq * seq)
  for r in 0..<seq {
    for c in 0..<seq {
      var dot: Float = 0
      for d in 0..<D {
        dot += network.Q[r * D + d] * network.K[c * D + d]
      }
      S[r * seq + c] = dot
    }
  }

  // Apply causal + sliding window mask
  for r in 0..<seq {
    for c in 0..<seq {
      if c > r || c < r - windowSize {
        S[r * seq + c] = -.infinity
      }
    }
  }

  return applySoftmaxAndProject(S: S, V: network.V, seq: seq, D: D, scale: scale)
}

/// Shared: scale S, softmax per row, project with V.
private func applySoftmaxAndProject(
  S: [Float], V: [Float], seq: Int, D: Int, scale: Float
) -> [Float] {
  var P = S

  // Scale and softmax per row
  for r in 0..<seq {
    // Find max
    var maxVal: Float = -.infinity
    for c in 0..<seq {
      let val = P[r * seq + c] * scale
      P[r * seq + c] = val
      if val > maxVal { maxVal = val }
    }
    // exp and sum
    var sumExp: Float = 0
    for c in 0..<seq {
      let e = exp(P[r * seq + c] - maxVal)
      P[r * seq + c] = e
      sumExp += e
    }
    // normalize
    for c in 0..<seq {
      P[r * seq + c] /= sumExp
    }
  }

  // O = P * V
  var O = [Float](repeating: 0, count: seq * D)
  for r in 0..<seq {
    for d in 0..<D {
      var acc: Float = 0
      for c in 0..<seq {
        acc += P[r * seq + c] * V[c * D + d]
      }
      O[r * D + d] = acc
    }
  }
  return O
}
