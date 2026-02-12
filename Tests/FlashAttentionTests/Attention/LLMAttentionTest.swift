import XCTest
import FlashAttention
import Metal
import func Foundation.exp
import func Foundation.log
import func Foundation.sqrt

/// Tests for LLM-shaped attention: multi-head causal prefill and decode.
final class LLMAttentionTest: XCTestCase {

  // MARK: - Prefill: Multi-head causal attention (SmolLM-like)

  func testPrefillSmolLM() throws {
    // SmolLM 135M: 9 heads, D=64, typical prefill seq=128
    try validatePrefill(numHeads: 9, seqLen: 128, D: 64)
  }

  func testPrefillSmall() throws {
    // Small case for debugging
    try validatePrefill(numHeads: 4, seqLen: 32, D: 16)
  }

  func testPrefillOddDimensions() throws {
    // Non-power-of-2 dimensions
    try validatePrefill(numHeads: 3, seqLen: 37, D: 48)
  }

  // MARK: - Decode: R=1, multi-head, non-causal (attending to full cache)

  func testDecodeSmolLM() throws {
    // SmolLM decode: 1 new token attending to 128 cached tokens
    try validateDecode(numHeads: 9, cacheLen: 128, D: 64)
  }

  func testDecodeSmall() throws {
    try validateDecode(numHeads: 4, cacheLen: 32, D: 16)
  }

  func testDecodeLongContext() throws {
    // Longer context
    try validateDecode(numHeads: 4, cacheLen: 512, D: 32)
  }

  // MARK: - GQA (Grouped Query Attention)

  func testGQASmall() throws {
    // 8 Q heads, 2 KV heads (repeat factor 4), seq=32, D=16
    try validateGQA(numHeads: 8, numKVHeads: 2, seqLen: 32, D: 16)
  }

  func testGQALlama() throws {
    // Llama 3.2-like: 32 Q heads, 8 KV heads (repeat factor 4)
    try validateGQA(numHeads: 32, numKVHeads: 8, seqLen: 64, D: 32)
  }

  func testGQADecode() throws {
    // GQA decode: R=1
    try validateGQADecode(numHeads: 8, numKVHeads: 2, cacheLen: 64, D: 16)
  }

  func testGQADecodeLlamaD128() throws {
    // Matches real Llama 3.2 3B: 24 Q heads, 8 KV heads, D=128
    try validateGQADecode(numHeads: 24, numKVHeads: 8, cacheLen: 512, D: 128)
  }

  func testGQAPrefillLlamaD128() throws {
    // Llama 3.2 3B prefill: 24 Q heads, 8 KV heads, D=128, causal
    try validateGQAPrefillCausal(numHeads: 24, numKVHeads: 8, seqLen: 64, D: 128)
  }

  func testChunkedPrefillCausal() throws {
    // Chunked prefill: second chunk (R=32, C=64, offset=32)
    try validateChunkedPrefill(numHeads: 4, numKVHeads: 2, chunkSize: 32, totalSeq: 64, D: 32)
  }

  func testChunkedPrefillLlama() throws {
    // Llama-like chunked prefill: R=64, C=128, offset=64
    try validateChunkedPrefill(numHeads: 24, numKVHeads: 8, chunkSize: 64, totalSeq: 128, D: 128)
  }

  // MARK: - Validation Helpers

  private func validatePrefill(numHeads: Int, seqLen: Int, D: Int) throws {
    let device = MTLContext.global.device
    print("[LLM] prefill: numHeads=\(numHeads) seqLen=\(seqLen) D=\(D)")

    // Generate random data per head
    let headElements = seqLen * D
    var allQ = [Float](repeating: 0, count: numHeads * headElements)
    var allK = [Float](repeating: 0, count: numHeads * headElements)
    var allV = [Float](repeating: 0, count: numHeads * headElements)

    for i in 0..<allQ.count {
      allQ[i] = Float.random(in: -1...1)
      allK[i] = Float.random(in: -1...1)
      allV[i] = Float.random(in: -1...1)
    }

    // CPU reference: per-head causal attention
    var cpuO = [Float](repeating: 0, count: numHeads * headElements)
    for h in 0..<numHeads {
      let off = h * headElements
      let headO = causalAttentionCPU(
        Q: Array(allQ[off..<off+headElements]),
        K: Array(allK[off..<off+headElements]),
        V: Array(allV[off..<off+headElements]),
        R: seqLen, C: seqLen, D: D)
      for i in 0..<headElements {
        cpuO[off + i] = headO[i]
      }
    }

    // GPU
    let bufQ = device.makeBuffer(bytes: allQ, length: allQ.count * 4, options: .storageModeShared)!
    let bufK = device.makeBuffer(bytes: allK, length: allK.count * 4, options: .storageModeShared)!
    let bufV = device.makeBuffer(bytes: allV, length: allV.count * 4, options: .storageModeShared)!
    let bufO = device.makeBuffer(length: numHeads * headElements * 4, options: .storageModeShared)!
    let bufL = device.makeBuffer(length: numHeads * seqLen * 4, options: .storageModeShared)!

    let cmdQueue = MTLContext.global.commandQueue
    let cmdBuf = cmdQueue.makeCommandBuffer()!
    let enc = cmdBuf.makeComputeCommandEncoder()!

    FlashAttentionDispatch.bindForwardBuffers(
      encoder: enc, Q: bufQ, K: bufK, V: bufV, O: bufO, L: bufL)

    FlashAttentionDispatch.forward(
      encoder: enc,
      R: seqLen, C: seqLen, D: D,
      numHeads: numHeads,
      causal: true)

    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    // Validate
    let resultPtr = bufO.contents().bindMemory(to: Float.self, capacity: numHeads * headElements)
    var errorCount = 0
    for i in 0..<(numHeads * headElements) {
      let expected = cpuO[i]
      let actual = resultPtr[i]
      let error = abs(expected - actual)
      if error > 2e-5 && !expected.isNaN && !expected.isInfinite {
        if errorCount < 10 {
          let h = i / headElements
          let pos = (i % headElements) / D
          let d = i % D
          print("  head=\(h) pos=\(pos) d=\(d): expected=\(expected) actual=\(actual) err=\(error)")
        }
        errorCount += 1
      }
    }
    XCTAssertEqual(errorCount, 0, "Prefill had \(errorCount) errors (H=\(numHeads) S=\(seqLen) D=\(D))")
  }

  private func validateDecode(numHeads: Int, cacheLen: Int, D: Int) throws {
    let device = MTLContext.global.device
    let R = 1  // single new token
    let C = cacheLen
    print("[LLM] decode: numHeads=\(numHeads) cacheLen=\(cacheLen) D=\(D)")

    // Generate random data
    let qElements = numHeads * R * D
    let kvElements = numHeads * C * D
    var allQ = [Float](repeating: 0, count: qElements)
    var allK = [Float](repeating: 0, count: kvElements)
    var allV = [Float](repeating: 0, count: kvElements)

    for i in 0..<qElements { allQ[i] = Float.random(in: -1...1) }
    for i in 0..<kvElements {
      allK[i] = Float.random(in: -1...1)
      allV[i] = Float.random(in: -1...1)
    }

    // CPU reference: per-head non-causal attention (R=1 attending to full cache)
    var cpuO = [Float](repeating: 0, count: qElements)
    for h in 0..<numHeads {
      let qOff = h * R * D
      let kvOff = h * C * D
      let headO = standardAttentionCPU(
        Q: Array(allQ[qOff..<qOff + R*D]),
        K: Array(allK[kvOff..<kvOff + C*D]),
        V: Array(allV[kvOff..<kvOff + C*D]),
        R: R, C: C, D: D)
      for i in 0..<(R * D) {
        cpuO[qOff + i] = headO[i]
      }
    }

    // GPU
    let bufQ = device.makeBuffer(bytes: allQ, length: allQ.count * 4, options: .storageModeShared)!
    let bufK = device.makeBuffer(bytes: allK, length: allK.count * 4, options: .storageModeShared)!
    let bufV = device.makeBuffer(bytes: allV, length: allV.count * 4, options: .storageModeShared)!
    let bufO = device.makeBuffer(length: qElements * 4, options: .storageModeShared)!
    let bufL = device.makeBuffer(length: numHeads * R * 4, options: .storageModeShared)!

    let cmdQueue = MTLContext.global.commandQueue
    let cmdBuf = cmdQueue.makeCommandBuffer()!
    let enc = cmdBuf.makeComputeCommandEncoder()!

    FlashAttentionDispatch.bindForwardBuffers(
      encoder: enc, Q: bufQ, K: bufK, V: bufV, O: bufO, L: bufL)

    // Decode: R=1, C=cacheLen, NOT causal (single token sees everything)
    FlashAttentionDispatch.forward(
      encoder: enc,
      R: R, C: C, D: D,
      numHeads: numHeads,
      causal: false)

    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    // Validate
    let resultPtr = bufO.contents().bindMemory(to: Float.self, capacity: qElements)
    var errorCount = 0
    for i in 0..<qElements {
      let expected = cpuO[i]
      let actual = resultPtr[i]
      let error = abs(expected - actual)
      if error > 2e-5 && !expected.isNaN && !expected.isInfinite {
        if errorCount < 10 {
          let h = i / (R * D)
          let d = i % D
          print("  head=\(h) d=\(d): expected=\(expected) actual=\(actual) err=\(error)")
        }
        errorCount += 1
      }
    }
    XCTAssertEqual(errorCount, 0, "Decode had \(errorCount) errors (H=\(numHeads) C=\(cacheLen) D=\(D))")
  }

  private func validateGQA(numHeads: Int, numKVHeads: Int, seqLen: Int, D: Int) throws {
    let device = MTLContext.global.device
    let kvRepeatFactor = numHeads / numKVHeads
    print("[LLM] GQA prefill: numHeads=\(numHeads) numKVHeads=\(numKVHeads) seqLen=\(seqLen) D=\(D)")

    let qHeadElements = seqLen * D
    let kvHeadElements = seqLen * D
    var allQ = [Float](repeating: 0, count: numHeads * qHeadElements)
    var allK = [Float](repeating: 0, count: numKVHeads * kvHeadElements)
    var allV = [Float](repeating: 0, count: numKVHeads * kvHeadElements)

    for i in 0..<allQ.count { allQ[i] = Float.random(in: -1...1) }
    for i in 0..<allK.count { allK[i] = Float.random(in: -1...1) }
    for i in 0..<allV.count { allV[i] = Float.random(in: -1...1) }

    // CPU reference: each Q head uses KV head = q_head / kvRepeatFactor
    var cpuO = [Float](repeating: 0, count: numHeads * qHeadElements)
    for qh in 0..<numHeads {
      let kvh = qh / kvRepeatFactor
      let qOff = qh * qHeadElements
      let kvOff = kvh * kvHeadElements
      let headO = standardAttentionCPU(
        Q: Array(allQ[qOff..<qOff + qHeadElements]),
        K: Array(allK[kvOff..<kvOff + kvHeadElements]),
        V: Array(allV[kvOff..<kvOff + kvHeadElements]),
        R: seqLen, C: seqLen, D: D)
      for i in 0..<qHeadElements {
        cpuO[qOff + i] = headO[i]
      }
    }

    // GPU
    let bufQ = device.makeBuffer(bytes: allQ, length: allQ.count * 4, options: .storageModeShared)!
    let bufK = device.makeBuffer(bytes: allK, length: allK.count * 4, options: .storageModeShared)!
    let bufV = device.makeBuffer(bytes: allV, length: allV.count * 4, options: .storageModeShared)!
    let bufO = device.makeBuffer(length: numHeads * qHeadElements * 4, options: .storageModeShared)!
    let bufL = device.makeBuffer(length: numHeads * seqLen * 4, options: .storageModeShared)!

    let cmdQueue = MTLContext.global.commandQueue
    let cmdBuf = cmdQueue.makeCommandBuffer()!
    let enc = cmdBuf.makeComputeCommandEncoder()!

    FlashAttentionDispatch.bindForwardBuffers(
      encoder: enc, Q: bufQ, K: bufK, V: bufV, O: bufO, L: bufL)

    FlashAttentionDispatch.forward(
      encoder: enc,
      R: seqLen, C: seqLen, D: D,
      numHeads: numHeads,
      numKVHeads: numKVHeads,
      causal: false)

    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    let resultPtr = bufO.contents().bindMemory(to: Float.self, capacity: numHeads * qHeadElements)
    var errorCount = 0
    for i in 0..<(numHeads * qHeadElements) {
      let expected = cpuO[i]
      let actual = resultPtr[i]
      let error = abs(expected - actual)
      if error > 2e-5 && !expected.isNaN && !expected.isInfinite {
        if errorCount < 10 {
          let h = i / qHeadElements
          let pos = (i % qHeadElements) / D
          let d = i % D
          print("  head=\(h) pos=\(pos) d=\(d): expected=\(expected) actual=\(actual) err=\(error)")
        }
        errorCount += 1
      }
    }
    XCTAssertEqual(errorCount, 0, "GQA prefill had \(errorCount) errors (QH=\(numHeads) KVH=\(numKVHeads) S=\(seqLen) D=\(D))")
  }

  private func validateGQAPrefillCausal(numHeads: Int, numKVHeads: Int, seqLen: Int, D: Int) throws {
    let device = MTLContext.global.device
    let kvRepeatFactor = numHeads / numKVHeads
    print("[LLM] GQA causal prefill: numHeads=\(numHeads) numKVHeads=\(numKVHeads) seqLen=\(seqLen) D=\(D)")

    let qHeadElements = seqLen * D
    let kvHeadElements = seqLen * D
    var allQ = [Float](repeating: 0, count: numHeads * qHeadElements)
    var allK = [Float](repeating: 0, count: numKVHeads * kvHeadElements)
    var allV = [Float](repeating: 0, count: numKVHeads * kvHeadElements)

    for i in 0..<allQ.count { allQ[i] = Float.random(in: -1...1) }
    for i in 0..<allK.count { allK[i] = Float.random(in: -1...1) }
    for i in 0..<allV.count { allV[i] = Float.random(in: -1...1) }

    var cpuO = [Float](repeating: 0, count: numHeads * qHeadElements)
    for qh in 0..<numHeads {
      let kvh = qh / kvRepeatFactor
      let qOff = qh * qHeadElements
      let kvOff = kvh * kvHeadElements
      let headO = causalAttentionCPU(
        Q: Array(allQ[qOff..<qOff + qHeadElements]),
        K: Array(allK[kvOff..<kvOff + kvHeadElements]),
        V: Array(allV[kvOff..<kvOff + kvHeadElements]),
        R: seqLen, C: seqLen, D: D)
      for i in 0..<qHeadElements { cpuO[qOff + i] = headO[i] }
    }

    let bufQ = device.makeBuffer(bytes: allQ, length: allQ.count * 4, options: .storageModeShared)!
    let bufK = device.makeBuffer(bytes: allK, length: allK.count * 4, options: .storageModeShared)!
    let bufV = device.makeBuffer(bytes: allV, length: allV.count * 4, options: .storageModeShared)!
    let bufO = device.makeBuffer(length: numHeads * qHeadElements * 4, options: .storageModeShared)!
    let bufL = device.makeBuffer(length: numHeads * seqLen * 4, options: .storageModeShared)!

    let cmdQueue = MTLContext.global.commandQueue
    let cmdBuf = cmdQueue.makeCommandBuffer()!
    let enc = cmdBuf.makeComputeCommandEncoder()!

    FlashAttentionDispatch.bindForwardBuffers(
      encoder: enc, Q: bufQ, K: bufK, V: bufV, O: bufO, L: bufL)

    FlashAttentionDispatch.forward(
      encoder: enc,
      R: seqLen, C: seqLen, D: D,
      numHeads: numHeads,
      numKVHeads: numKVHeads,
      causal: true)

    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    let resultPtr = bufO.contents().bindMemory(to: Float.self, capacity: numHeads * qHeadElements)
    var errorCount = 0
    for i in 0..<(numHeads * qHeadElements) {
      let expected = cpuO[i]
      let actual = resultPtr[i]
      let error = abs(expected - actual)
      if error > 2e-5 && !expected.isNaN && !expected.isInfinite {
        if errorCount < 10 {
          let h = i / qHeadElements
          let pos = (i % qHeadElements) / D
          let d = i % D
          print("  head=\(h) pos=\(pos) d=\(d): expected=\(expected) actual=\(actual) err=\(error)")
        }
        errorCount += 1
      }
    }
    XCTAssertEqual(errorCount, 0, "GQA causal prefill had \(errorCount) errors (QH=\(numHeads) KVH=\(numKVHeads) S=\(seqLen) D=\(D))")
  }

  /// Validate chunked causal prefill: Q is the second chunk (rows offset..offset+R-1),
  /// K/V is the full sequence (0..C-1). Causal mask: mask where col > (local_row + offset).
  private func validateChunkedPrefill(
    numHeads: Int, numKVHeads: Int, chunkSize: Int, totalSeq: Int, D: Int
  ) throws {
    let device = MTLContext.global.device
    let kvRepeatFactor = numHeads / numKVHeads
    let R = chunkSize
    let C = totalSeq
    let offset = C - R
    print("[LLM] chunked prefill: numHeads=\(numHeads) numKVHeads=\(numKVHeads) R=\(R) C=\(C) offset=\(offset) D=\(D)")

    // Generate random data — Q is for the second chunk, K/V is full sequence
    var allQ = [Float](repeating: 0, count: numHeads * R * D)
    var allK = [Float](repeating: 0, count: numKVHeads * C * D)
    var allV = [Float](repeating: 0, count: numKVHeads * C * D)

    for i in 0..<allQ.count { allQ[i] = Float.random(in: -1...1) }
    for i in 0..<allK.count { allK[i] = Float.random(in: -1...1) }
    for i in 0..<allV.count { allV[i] = Float.random(in: -1...1) }

    // CPU reference: causal with offset (mask where col > row + offset)
    var cpuO = [Float](repeating: 0, count: numHeads * R * D)
    for qh in 0..<numHeads {
      let kvh = qh / kvRepeatFactor
      let qOff = qh * R * D
      let kvOff = kvh * C * D
      let headO = causalWithOffsetCPU(
        Q: Array(allQ[qOff..<qOff + R * D]),
        K: Array(allK[kvOff..<kvOff + C * D]),
        V: Array(allV[kvOff..<kvOff + C * D]),
        R: R, C: C, D: D, offset: offset)
      for i in 0..<(R * D) { cpuO[qOff + i] = headO[i] }
    }

    let bufQ = device.makeBuffer(bytes: allQ, length: allQ.count * 4, options: .storageModeShared)!
    let bufK = device.makeBuffer(bytes: allK, length: allK.count * 4, options: .storageModeShared)!
    let bufV = device.makeBuffer(bytes: allV, length: allV.count * 4, options: .storageModeShared)!
    let bufO = device.makeBuffer(length: numHeads * R * D * 4, options: .storageModeShared)!
    let bufL = device.makeBuffer(length: numHeads * R * 4, options: .storageModeShared)!

    let cmdQueue = MTLContext.global.commandQueue
    let cmdBuf = cmdQueue.makeCommandBuffer()!
    let enc = cmdBuf.makeComputeCommandEncoder()!

    FlashAttentionDispatch.bindForwardBuffers(
      encoder: enc, Q: bufQ, K: bufK, V: bufV, O: bufO, L: bufL)

    FlashAttentionDispatch.forward(
      encoder: enc,
      R: R, C: C, D: D,
      numHeads: numHeads,
      numKVHeads: numKVHeads,
      causal: true,
      causalOffset: offset)

    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    let resultPtr = bufO.contents().bindMemory(to: Float.self, capacity: numHeads * R * D)
    var errorCount = 0
    for i in 0..<(numHeads * R * D) {
      let expected = cpuO[i]
      let actual = resultPtr[i]
      let error = abs(expected - actual)
      if error > 2e-5 && !expected.isNaN && !expected.isInfinite {
        if errorCount < 10 {
          let h = i / (R * D)
          let pos = (i % (R * D)) / D
          let d = i % D
          print("  head=\(h) pos=\(pos) d=\(d): expected=\(expected) actual=\(actual) err=\(error)")
        }
        errorCount += 1
      }
    }
    XCTAssertEqual(errorCount, 0, "Chunked prefill had \(errorCount) errors (QH=\(numHeads) KVH=\(numKVHeads) R=\(R) C=\(C) off=\(offset) D=\(D))")
  }

  private func validateGQADecode(numHeads: Int, numKVHeads: Int, cacheLen: Int, D: Int) throws {
    let device = MTLContext.global.device
    let R = 1
    let C = cacheLen
    let kvRepeatFactor = numHeads / numKVHeads
    print("[LLM] GQA decode: numHeads=\(numHeads) numKVHeads=\(numKVHeads) cacheLen=\(cacheLen) D=\(D)")

    let qElements = numHeads * R * D
    let kvElements = numKVHeads * C * D
    var allQ = [Float](repeating: 0, count: qElements)
    var allK = [Float](repeating: 0, count: kvElements)
    var allV = [Float](repeating: 0, count: kvElements)

    for i in 0..<qElements { allQ[i] = Float.random(in: -1...1) }
    for i in 0..<kvElements { allK[i] = Float.random(in: -1...1); allV[i] = Float.random(in: -1...1) }

    var cpuO = [Float](repeating: 0, count: qElements)
    for qh in 0..<numHeads {
      let kvh = qh / kvRepeatFactor
      let qOff = qh * R * D
      let kvOff = kvh * C * D
      let headO = standardAttentionCPU(
        Q: Array(allQ[qOff..<qOff + R*D]),
        K: Array(allK[kvOff..<kvOff + C*D]),
        V: Array(allV[kvOff..<kvOff + C*D]),
        R: R, C: C, D: D)
      for i in 0..<(R * D) { cpuO[qOff + i] = headO[i] }
    }

    let bufQ = device.makeBuffer(bytes: allQ, length: allQ.count * 4, options: .storageModeShared)!
    let bufK = device.makeBuffer(bytes: allK, length: allK.count * 4, options: .storageModeShared)!
    let bufV = device.makeBuffer(bytes: allV, length: allV.count * 4, options: .storageModeShared)!
    let bufO = device.makeBuffer(length: qElements * 4, options: .storageModeShared)!
    let bufL = device.makeBuffer(length: numHeads * R * 4, options: .storageModeShared)!

    let cmdQueue = MTLContext.global.commandQueue
    let cmdBuf = cmdQueue.makeCommandBuffer()!
    let enc = cmdBuf.makeComputeCommandEncoder()!

    FlashAttentionDispatch.bindForwardBuffers(
      encoder: enc, Q: bufQ, K: bufK, V: bufV, O: bufO, L: bufL)

    FlashAttentionDispatch.forward(
      encoder: enc,
      R: R, C: C, D: D,
      numHeads: numHeads,
      numKVHeads: numKVHeads,
      causal: false)

    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    let resultPtr = bufO.contents().bindMemory(to: Float.self, capacity: qElements)
    var errorCount = 0
    for i in 0..<qElements {
      let expected = cpuO[i]
      let actual = resultPtr[i]
      let error = abs(expected - actual)
      if error > 2e-5 && !expected.isNaN && !expected.isInfinite {
        if errorCount < 10 {
          let h = i / (R * D)
          let d = i % D
          print("  head=\(h) d=\(d): expected=\(expected) actual=\(actual) err=\(error)")
        }
        errorCount += 1
      }
    }
    XCTAssertEqual(errorCount, 0, "GQA decode had \(errorCount) errors (QH=\(numHeads) KVH=\(numKVHeads) C=\(cacheLen) D=\(D))")
  }
}

// MARK: - CPU Reference Implementations

private func causalWithOffsetCPU(
  Q: [Float], K: [Float], V: [Float],
  R: Int, C: Int, D: Int, offset: Int
) -> [Float] {
  var O = [Float](repeating: 0, count: R * D)
  let scale = 1.0 / sqrt(Float(D))

  for row in 0..<R {
    let globalRow = row + offset
    var S = [Float](repeating: 0, count: C)
    for col in 0..<C {
      if col > globalRow {
        S[col] = -.greatestFiniteMagnitude
      } else {
        var dot: Float = 0
        for d in 0..<D { dot += Q[row * D + d] * K[col * D + d] }
        S[col] = dot * scale
      }
    }

    var maxS: Float = -.greatestFiniteMagnitude
    for col in 0..<C { maxS = max(maxS, S[col]) }
    var sumExp: Float = 0
    for col in 0..<C { S[col] = exp(S[col] - maxS); sumExp += S[col] }
    for col in 0..<C { S[col] /= sumExp }

    for d in 0..<D {
      var acc: Float = 0
      for col in 0..<C { acc += S[col] * V[col * D + d] }
      O[row * D + d] = acc
    }
  }
  return O
}

private func causalAttentionCPU(
  Q: [Float], K: [Float], V: [Float],
  R: Int, C: Int, D: Int
) -> [Float] {
  var O = [Float](repeating: 0, count: R * D)
  let scale = 1.0 / sqrt(Float(D))

  for row in 0..<R {
    var S = [Float](repeating: 0, count: C)
    for col in 0..<C {
      if col > row {
        S[col] = -.greatestFiniteMagnitude
      } else {
        var dot: Float = 0
        for d in 0..<D { dot += Q[row * D + d] * K[col * D + d] }
        S[col] = dot * scale
      }
    }

    var maxS: Float = -.greatestFiniteMagnitude
    for col in 0..<C { maxS = max(maxS, S[col]) }
    var sumExp: Float = 0
    for col in 0..<C { S[col] = exp(S[col] - maxS); sumExp += S[col] }
    for col in 0..<C { S[col] /= sumExp }

    for d in 0..<D {
      var acc: Float = 0
      for col in 0..<C { acc += S[col] * V[col * D + d] }
      O[row * D + d] = acc
    }
  }
  return O
}

private func standardAttentionCPU(
  Q: [Float], K: [Float], V: [Float],
  R: Int, C: Int, D: Int
) -> [Float] {
  var O = [Float](repeating: 0, count: R * D)
  let scale = 1.0 / sqrt(Float(D))

  for row in 0..<R {
    var S = [Float](repeating: 0, count: C)
    for col in 0..<C {
      var dot: Float = 0
      for d in 0..<D { dot += Q[row * D + d] * K[col * D + d] }
      S[col] = dot * scale
    }

    var maxS: Float = -.greatestFiniteMagnitude
    for col in 0..<C { maxS = max(maxS, S[col]) }
    var sumExp: Float = 0
    for col in 0..<C { S[col] = exp(S[col] - maxS); sumExp += S[col] }
    for col in 0..<C { S[col] /= sumExp }

    for d in 0..<D {
      var acc: Float = 0
      for col in 0..<C { acc += S[col] * V[col * D + d] }
      O[row * D + d] = acc
    }
  }
  return O
}
