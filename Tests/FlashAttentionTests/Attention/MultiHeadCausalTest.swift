import XCTest
import FlashAttention
import Metal
import func Foundation.exp
import func Foundation.log

// MARK: - Multi-Head Batched Dispatch Test

final class MultiHeadCausalTest: XCTestCase {

  func testMultiHead() throws {
    // 4 heads, seq=32, D=16
    let numHeads = 4
    let seq = 32
    let D = 16
    let device = MTLContext.global.device

    // Create per-head networks and compute CPU reference
    var networks: [Network] = []
    var cpuO: [[Float]] = []
    for _ in 0..<numHeads {
      var desc = NetworkDescriptor()
      desc.rowDimension = seq
      desc.columnDimension = seq
      desc.headDimension = D
      let net = Network(descriptor: desc)
      cpuO.append(net.inferenceAttention())
      networks.append(net)
    }

    // Pack multi-head Q, K, V into contiguous buffers
    let headElements = seq * D
    var packedQ = [Float](repeating: 0, count: numHeads * headElements)
    var packedK = [Float](repeating: 0, count: numHeads * headElements)
    var packedV = [Float](repeating: 0, count: numHeads * headElements)
    var packedO = [Float](repeating: 0, count: numHeads * headElements)
    for h in 0..<numHeads {
      let off = h * headElements
      for i in 0..<headElements {
        packedQ[off + i] = networks[h].Q[i]
        packedK[off + i] = networks[h].K[i]
        packedV[off + i] = networks[h].V[i]
      }
    }

    // Create GPU buffers
    let bufQ = device.makeBuffer(bytes: packedQ, length: packedQ.count * 4, options: .storageModeShared)!
    let bufK = device.makeBuffer(bytes: packedK, length: packedK.count * 4, options: .storageModeShared)!
    let bufV = device.makeBuffer(bytes: packedV, length: packedV.count * 4, options: .storageModeShared)!
    let bufO = device.makeBuffer(bytes: packedO, length: packedO.count * 4, options: .storageModeShared)!
    let bufL = device.makeBuffer(length: numHeads * seq * 4, options: .storageModeShared)!
    let bufD = device.makeBuffer(length: numHeads * seq * 4, options: .storageModeShared)!
    let dummy = device.makeBuffer(length: 4, options: .storageModeShared)!

    // Compile kernel
    var attDesc = AttentionDescriptor()
    attDesc.lowPrecisionInputs = false
    attDesc.lowPrecisionIntermediates = false
    attDesc.matrixDimensions = (row: UInt32(seq), column: UInt32(seq), head: UInt16(D))
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)

    let (kernel, pipeline) = AttentionKernel.pipeline(for: attDesc, type: .forward)

    // Create batched params
    let batchParams = AttentionKernel.createBatchedParamsBuffer(
      numHeads: UInt32(numHeads), R: UInt32(seq), C: UInt32(seq), D: UInt32(D))

    // Dispatch
    let cmdQueue = MTLContext.global.commandQueue
    let cmdBuf = cmdQueue.makeCommandBuffer()!
    let enc = cmdBuf.makeComputeCommandEncoder()!

    enc.setBuffer(bufQ, offset: 0, index: 0)
    enc.setBuffer(bufK, offset: 0, index: 1)
    enc.setBuffer(bufV, offset: 0, index: 2)
    enc.setBuffer(bufO, offset: 0, index: 3)
    enc.setBuffer(bufL, offset: 0, index: 4)
    enc.setBuffer(bufD, offset: 0, index: 5)
    enc.setBuffer(dummy, offset: 0, index: 6)
    enc.setBuffer(dummy, offset: 0, index: 7)
    enc.setBuffer(dummy, offset: 0, index: 8)
    enc.setBuffer(dummy, offset: 0, index: 9)

    AttentionKernel.dispatch(
      encoder: enc,
      kernel: kernel,
      pipeline: pipeline,
      batchedParams: batchParams,
      parallelizationDimension: seq,
      numHeads: numHeads,
      batchSize: 1)

    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    // Read back and validate
    let resultPtr = bufO.contents().bindMemory(to: Float.self, capacity: numHeads * headElements)
    var errorCount = 0
    for h in 0..<numHeads {
      for i in 0..<headElements {
        let expected = cpuO[h][i]
        let actual = resultPtr[h * headElements + i]
        let error = abs(expected - actual)
        if error > 2e-5 && !expected.isNaN && !expected.isInfinite {
          if errorCount < 10 {
            print("head=\(h) i=\(i): expected=\(expected) actual=\(actual) error=\(error)")
          }
          errorCount += 1
        }
      }
    }
    XCTAssertEqual(errorCount, 0, "Multi-head attention had \(errorCount) errors")
  }

  // MARK: - Causal Masking Test

  func testCausal() throws {
    let seq = 32
    let D = 16
    let device = MTLContext.global.device

    // Create network
    var netDesc = NetworkDescriptor()
    netDesc.rowDimension = seq
    netDesc.columnDimension = seq
    netDesc.headDimension = D
    let net = Network(descriptor: netDesc)

    // CPU reference: causal attention (mask where col > row)
    let cpuO = causalAttentionCPU(
      Q: net.Q, K: net.K, V: net.V,
      seq: seq, D: D)

    // GPU
    let bufQ = device.makeBuffer(bytes: net.Q, length: net.Q.count * 4, options: .storageModeShared)!
    let bufK = device.makeBuffer(bytes: net.K, length: net.K.count * 4, options: .storageModeShared)!
    let bufV = device.makeBuffer(bytes: net.V, length: net.V.count * 4, options: .storageModeShared)!
    var zeros = [Float](repeating: 0, count: seq * D)
    let bufO = device.makeBuffer(bytes: &zeros, length: zeros.count * 4, options: .storageModeShared)!
    let bufL = device.makeBuffer(length: seq * 4, options: .storageModeShared)!
    let bufD = device.makeBuffer(length: seq * 4, options: .storageModeShared)!
    let dummy = device.makeBuffer(length: 4, options: .storageModeShared)!

    var attDesc = AttentionDescriptor()
    attDesc.lowPrecisionInputs = false
    attDesc.lowPrecisionIntermediates = false
    attDesc.matrixDimensions = (row: UInt32(seq), column: UInt32(seq), head: UInt16(D))
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)
    attDesc.causal = true

    let (kernel, pipeline) = AttentionKernel.pipeline(for: attDesc, type: .forward)

    var batchParams: [UInt32] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    let bufBatch = device.makeBuffer(bytes: &batchParams, length: 52, options: .storageModeShared)!

    let cmdQueue = MTLContext.global.commandQueue
    let cmdBuf = cmdQueue.makeCommandBuffer()!
    let enc = cmdBuf.makeComputeCommandEncoder()!

    enc.setBuffer(bufQ, offset: 0, index: 0)
    enc.setBuffer(bufK, offset: 0, index: 1)
    enc.setBuffer(bufV, offset: 0, index: 2)
    enc.setBuffer(bufO, offset: 0, index: 3)
    enc.setBuffer(bufL, offset: 0, index: 4)
    enc.setBuffer(bufD, offset: 0, index: 5)
    enc.setBuffer(dummy, offset: 0, index: 6)
    enc.setBuffer(dummy, offset: 0, index: 7)
    enc.setBuffer(dummy, offset: 0, index: 8)
    enc.setBuffer(dummy, offset: 0, index: 9)

    AttentionKernel.dispatch(
      encoder: enc,
      kernel: kernel,
      pipeline: pipeline,
      batchedParams: bufBatch,
      parallelizationDimension: seq)

    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    let resultPtr = bufO.contents().bindMemory(to: Float.self, capacity: seq * D)
    var errorCount = 0
    for i in 0..<(seq * D) {
      let expected = cpuO[i]
      let actual = resultPtr[i]
      let error = abs(expected - actual)
      if error > 2e-5 && !expected.isNaN && !expected.isInfinite {
        if errorCount < 10 {
          print("i=\(i) row=\(i/D) col=\(i%D): expected=\(expected) actual=\(actual) error=\(error)")
        }
        errorCount += 1
      }
    }
    XCTAssertEqual(errorCount, 0, "Causal attention had \(errorCount) errors")
  }
}

// MARK: - CPU Causal Attention Reference

private func causalAttentionCPU(
  Q: [Float], K: [Float], V: [Float],
  seq: Int, D: Int
) -> [Float] {
  var O = [Float](repeating: 0, count: seq * D)
  let scale = 1.0 / Float(D).squareRoot()

  for row in 0..<seq {
    // S = Q[row] * K^T, with causal mask
    var S = [Float](repeating: 0, count: seq)
    for col in 0..<seq {
      if col > row {
        S[col] = -.greatestFiniteMagnitude
      } else {
        var dot: Float = 0
        for d in 0..<D {
          dot += Q[row * D + d] * K[col * D + d]
        }
        S[col] = dot * scale
      }
    }

    // Softmax
    var maxS: Float = -.greatestFiniteMagnitude
    for col in 0..<seq { maxS = max(maxS, S[col]) }
    var sumExp: Float = 0
    for col in 0..<seq {
      S[col] = exp(S[col] - maxS)
      sumExp += S[col]
    }
    for col in 0..<seq { S[col] /= sumExp }

    // O = P * V
    for d in 0..<D {
      var acc: Float = 0
      for col in 0..<seq {
        acc += S[col] * V[col * D + d]
      }
      O[row * D + d] = acc
    }
  }
  return O
}
