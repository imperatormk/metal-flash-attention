import XCTest
import FlashAttention
import MetalASM

/// Tests for BF16 input and low-precision output flags.
final class PrecisionAttentionTest: XCTestCase {

  /// Test BF16 inputs (useBF16Inputs = true).
  func testBF16Inputs() throws {
    let seq = 16
    let D = 32

    var networkDesc = NetworkDescriptor()
    networkDesc.rowDimension = seq
    networkDesc.columnDimension = seq
    networkDesc.headDimension = D
    let network = Network(descriptor: networkDesc)

    let expectedO = referenceAttention(network: network, seq: seq, D: D)

    var attDesc = AttentionDescriptor()
    attDesc.useBF16Inputs = true
    attDesc.matrixDimensions = (row: UInt32(seq), column: UInt32(seq), head: UInt16(D))
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)

    let resultO = try runForwardKernel(attDesc: attDesc, network: network, seq: seq, D: D)

    // BF16 inputs lose some precision — allow 5e-2 tolerance
    let errorCount = countErrors(expected: expectedO, got: resultO, tolerance: 5e-2, label: "bf16_in")
    XCTAssertEqual(errorCount, 0, "BF16 inputs test failed: \(errorCount) errors")
  }

  /// Test FP16 outputs (lowPrecisionOutputs = true, useBF16Outputs = false).
  func testFP16Outputs() throws {
    let seq = 16
    let D = 32

    var networkDesc = NetworkDescriptor()
    networkDesc.rowDimension = seq
    networkDesc.columnDimension = seq
    networkDesc.headDimension = D
    let network = Network(descriptor: networkDesc)

    let expectedO = referenceAttention(network: network, seq: seq, D: D)

    var attDesc = AttentionDescriptor()
    attDesc.lowPrecisionOutputs = true
    attDesc.useBF16Outputs = false
    attDesc.matrixDimensions = (row: UInt32(seq), column: UInt32(seq), head: UInt16(D))
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)

    let resultO = try runForwardKernel(attDesc: attDesc, network: network, seq: seq, D: D)

    // FP16 output: ~5e-4 tolerance
    let errorCount = countErrors(expected: expectedO, got: resultO, tolerance: 5e-3, label: "fp16_out")
    XCTAssertEqual(errorCount, 0, "FP16 outputs test failed: \(errorCount) errors")
  }

  /// Test BF16 outputs (lowPrecisionOutputs = true, useBF16Outputs = true).
  func testBF16Outputs() throws {
    let seq = 16
    let D = 32

    var networkDesc = NetworkDescriptor()
    networkDesc.rowDimension = seq
    networkDesc.columnDimension = seq
    networkDesc.headDimension = D
    let network = Network(descriptor: networkDesc)

    let expectedO = referenceAttention(network: network, seq: seq, D: D)

    var attDesc = AttentionDescriptor()
    attDesc.lowPrecisionOutputs = true
    attDesc.useBF16Outputs = true
    attDesc.matrixDimensions = (row: UInt32(seq), column: UInt32(seq), head: UInt16(D))
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)

    let resultO = try runForwardKernel(attDesc: attDesc, network: network, seq: seq, D: D)

    // BF16 output: ~1e-2 tolerance
    let errorCount = countErrors(expected: expectedO, got: resultO, tolerance: 5e-2, label: "bf16_out")
    XCTAssertEqual(errorCount, 0, "BF16 outputs test failed: \(errorCount) errors")
  }

  /// Test combined: BF16 inputs + BF16 outputs.
  func testBF16InputsAndOutputs() throws {
    let seq = 16
    let D = 32

    var networkDesc = NetworkDescriptor()
    networkDesc.rowDimension = seq
    networkDesc.columnDimension = seq
    networkDesc.headDimension = D
    let network = Network(descriptor: networkDesc)

    let expectedO = referenceAttention(network: network, seq: seq, D: D)

    var attDesc = AttentionDescriptor()
    attDesc.useBF16Inputs = true
    attDesc.lowPrecisionOutputs = true
    attDesc.useBF16Outputs = true
    attDesc.matrixDimensions = (row: UInt32(seq), column: UInt32(seq), head: UInt16(D))
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)

    let resultO = try runForwardKernel(attDesc: attDesc, network: network, seq: seq, D: D)

    // Combined BF16: ~5e-2 tolerance
    let errorCount = countErrors(expected: expectedO, got: resultO, tolerance: 5e-2, label: "bf16_both")
    XCTAssertEqual(errorCount, 0, "BF16 inputs+outputs test failed: \(errorCount) errors")
  }
}

// MARK: - Helpers

private func runForwardKernel(
  attDesc: AttentionDescriptor, network: Network, seq: Int, D: Int
) throws -> [Float] {
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
  return resultO
}

private func referenceAttention(network: Network, seq: Int, D: Int) -> [Float] {
  let scale = 1.0 / Float(D).squareRoot()
  var S = [Float](repeating: 0, count: seq * seq)
  for r in 0..<seq {
    for c in 0..<seq {
      var dot: Float = 0
      for d in 0..<D { dot += network.Q[r * D + d] * network.K[c * D + d] }
      S[r * seq + c] = dot
    }
  }

  var P = S
  for r in 0..<seq {
    var maxVal: Float = -.infinity
    for c in 0..<seq {
      let val = P[r * seq + c] * scale
      P[r * seq + c] = val
      if val > maxVal { maxVal = val }
    }
    var sumExp: Float = 0
    for c in 0..<seq {
      let e = exp(P[r * seq + c] - maxVal)
      P[r * seq + c] = e
      sumExp += e
    }
    for c in 0..<seq { P[r * seq + c] /= sumExp }
  }

  var O = [Float](repeating: 0, count: seq * D)
  for r in 0..<seq {
    for d in 0..<D {
      var acc: Float = 0
      for c in 0..<seq { acc += P[r * seq + c] * network.V[c * D + d] }
      O[r * D + d] = acc
    }
  }
  return O
}

private func countErrors(
  expected: [Float], got: [Float], tolerance: Float, label: String
) -> Int {
  var errorCount = 0
  for i in 0..<expected.count {
    let err = abs(expected[i] - got[i])
    if err > tolerance || err.isNaN {
      if (expected[i].isNaN || expected[i].isInfinite),
         (got[i].isNaN || got[i].isInfinite) { continue }
      errorCount += 1
      if errorCount <= 5 {
        print("  \(label): error[\(i)] = \(err), expected=\(expected[i]), got=\(got[i])")
      }
    }
  }
  return errorCount
}
