import XCTest
import FlashAttention
import MetalASM

/// Tests for quantized KV cache (FP8_E4M3, FP8_E5M2, INT8, NF4).
final class QuantizedKVTest: XCTestCase {

  /// INT8 quantized K/V: each byte is signed int8, dequant = int8(val) * scale
  func testINT8QuantizedKV() throws {
    try runQuantizedTest(quantPrec: .INT8, tolerance: 5e-2)
  }

  /// FP8 E4M3 quantized K/V: linear scheme, dequant = (val-128)/127 * 448 * scale
  func testFP8E4M3QuantizedKV() throws {
    try runQuantizedTest(quantPrec: .FP8_E4M3, tolerance: 5e-2)
  }

  /// FP8 E5M2 quantized K/V: linear scheme, dequant = (val-128)/127 * 57344 * scale
  func testFP8E5M2QuantizedKV() throws {
    // E5M2 has fp8_max=57344, producing large intermediate values (~500)
    // where half-precision rounding gives ~0.25 absolute error
    try runQuantizedTest(quantPrec: .FP8_E5M2, tolerance: 5e-1)
  }

  /// NF4 quantized K/V: 4-bit NormalFloat, 2 values per byte, codebook lookup
  func testNF4QuantizedKV() throws {
    // NF4 is 4-bit quantization — reference uses max_diff<0.5, mean_diff<0.1
    try runNF4Test(tolerance: 5e-1)
  }
}

// MARK: - Shared Test Logic

private func runQuantizedTest(
  quantPrec: GEMMOperandPrecision, tolerance: Float
) throws {
  let seq = 16
  let D = 32
  let scale: Float = 0.01  // small scale to keep values reasonable

  // Generate random Q (float32)
  var Q = [Float](repeating: 0, count: seq * D)
  for i in 0..<Q.count { Q[i] = Float.random(in: -1...1) }

  // Generate random quantized K/V (uint8)
  var K_quant = [UInt8](repeating: 0, count: seq * D)
  var V_quant = [UInt8](repeating: 0, count: seq * D)
  for i in 0..<K_quant.count {
    K_quant[i] = UInt8.random(in: 0...255)
    V_quant[i] = UInt8.random(in: 0...255)
  }

  // Dequantize on CPU for reference
  let K_float = dequantize(K_quant, scale: scale, prec: quantPrec, seq: seq, D: D)
  let V_float = dequantize(V_quant, scale: scale, prec: quantPrec, seq: seq, D: D)

  // CPU reference attention
  let expectedO = referenceAttention(Q: Q, K: K_float, V: V_float, seq: seq, D: D)

  // Set up attention descriptor with quantized KV
  var attDesc = AttentionDescriptor()
  attDesc.lowPrecisionInputs = false
  attDesc.lowPrecisionIntermediates = false
  attDesc.quantizedKV = quantPrec
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
  try ir.write(toFile: "/tmp/quant_ir_\(quantPrec).ll", atomically: true, encoding: .utf8)
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

  // Create GPU buffers
  // Q is FP32 (non-quantized)
  let bufQ = device.makeBuffer(bytes: Q, length: Q.count * 4, options: .storageModeShared)!
  // K/V are uint8 (quantized)
  let bufK = device.makeBuffer(bytes: K_quant, length: K_quant.count, options: .storageModeShared)!
  let bufV = device.makeBuffer(bytes: V_quant, length: V_quant.count, options: .storageModeShared)!
  // O output (FP32)
  var resultO = [Float](repeating: .nan, count: seq * D)
  let bufO = device.makeBuffer(bytes: resultO, length: resultO.count * 4, options: .storageModeShared)!
  // L (logsumexp, FP32)
  let resultL = [Float](repeating: 0, count: seq)
  let bufL = device.makeBuffer(bytes: resultL, length: resultL.count * 4, options: .storageModeShared)!
  let dummy = device.makeBuffer(length: 4, options: .storageModeShared)!

  // Scale buffers: K_scale and V_scale (float, 1 per head → just 1 value for single-head)
  var kScale = [scale]
  var vScale = [scale]
  let bufKScale = device.makeBuffer(bytes: &kScale, length: 4, options: .storageModeShared)!
  let bufVScale = device.makeBuffer(bytes: &vScale, length: 4, options: .storageModeShared)!

  // Batch params
  var bp: [UInt32] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, UInt32(seq), UInt32(seq)]
  let bufBP = device.makeBuffer(bytes: &bp, length: bp.count * 4, options: .storageModeShared)!

  let cmdBuf = MTLContext.global.commandQueue.makeCommandBuffer()!
  let enc = cmdBuf.makeComputeCommandEncoder()!

  // Buffer bindings matching air.location_index:
  // 0=Q, 1=K, 2=V, 3=O, 4=L, 5=D_buf, 6=dO, 7=dV, 8=dK, 9=dQ, 10=mask, 11=bias
  // 20=K_scale, 21=V_scale, 30=bp
  enc.setBuffer(bufQ, offset: 0, index: 0)
  enc.setBuffer(bufK, offset: 0, index: 1)
  enc.setBuffer(bufV, offset: 0, index: 2)
  enc.setBuffer(bufO, offset: 0, index: 3)
  enc.setBuffer(bufL, offset: 0, index: 4)
  enc.setBuffer(dummy, offset: 0, index: 5)   // D_buf
  enc.setBuffer(dummy, offset: 0, index: 6)   // dO
  enc.setBuffer(dummy, offset: 0, index: 7)   // dV
  enc.setBuffer(dummy, offset: 0, index: 8)   // dK
  enc.setBuffer(dummy, offset: 0, index: 9)   // dQ
  enc.setBuffer(dummy, offset: 0, index: 10)  // mask
  enc.setBuffer(dummy, offset: 0, index: 11)  // bias
  enc.setBuffer(bufKScale, offset: 0, index: 20)  // K_scale
  enc.setBuffer(bufVScale, offset: 0, index: 21)  // V_scale
  enc.setBuffer(bufBP, offset: 0, index: 30)  // batch_params

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

  // Read back
  let oPtr = bufO.contents().bindMemory(to: Float.self, capacity: seq * D)
  for i in 0..<(seq * D) { resultO[i] = oPtr[i] }

  // Compare
  var errorCount = 0
  for i in 0..<(seq * D) {
    let err = abs(expectedO[i] - resultO[i])
    if err > tolerance || err.isNaN {
      if (expectedO[i].isNaN || expectedO[i].isInfinite),
         (resultO[i].isNaN || resultO[i].isInfinite) { continue }
      errorCount += 1
      if errorCount <= 5 {
        print("  \(quantPrec): error[\(i)] = \(err), expected=\(expectedO[i]), got=\(resultO[i])")
      }
    }
  }
  XCTAssertEqual(errorCount, 0, "\(quantPrec) quantized KV failed: \(errorCount) errors")
}

// MARK: - NF4 Test

private let NF4_CODEBOOK: [Float] = [
  -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
  -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
  0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
  0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
]

private func runNF4Test(tolerance: Float) throws {
  let seq = 16
  let D = 32  // must be even for NF4
  let scale: Float = 1.0  // NF4 codebook values are in [-1,1], scale=1 keeps them small

  // Generate random Q (float32)
  var Q = [Float](repeating: 0, count: seq * D)
  for i in 0..<Q.count { Q[i] = Float.random(in: -1...1) }

  // Generate random packed NF4 K/V (uint8, each byte = 2 nibbles)
  let packedCount = seq * D / 2
  var K_packed = [UInt8](repeating: 0, count: packedCount)
  var V_packed = [UInt8](repeating: 0, count: packedCount)
  for i in 0..<packedCount {
    K_packed[i] = UInt8.random(in: 0...255)
    V_packed[i] = UInt8.random(in: 0...255)
  }

  // Dequantize on CPU: low nibble = even head index, high nibble = odd
  let K_float = dequantizeNF4(K_packed, scale: scale, seq: seq, D: D)
  let V_float = dequantizeNF4(V_packed, scale: scale, seq: seq, D: D)

  let expectedO = referenceAttention(Q: Q, K: K_float, V: V_float, seq: seq, D: D)

  var attDesc = AttentionDescriptor()
  attDesc.lowPrecisionInputs = false
  attDesc.lowPrecisionIntermediates = false
  attDesc.quantizedKV = .NF4
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
  try ir.write(toFile: "/tmp/quant_ir_NF4.ll", atomically: true, encoding: .utf8)
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

  let bufQ = device.makeBuffer(bytes: Q, length: Q.count * 4, options: .storageModeShared)!
  let bufK = device.makeBuffer(bytes: K_packed, length: K_packed.count, options: .storageModeShared)!
  let bufV = device.makeBuffer(bytes: V_packed, length: V_packed.count, options: .storageModeShared)!
  var resultO = [Float](repeating: .nan, count: seq * D)
  let bufO = device.makeBuffer(bytes: resultO, length: resultO.count * 4, options: .storageModeShared)!
  let resultL = [Float](repeating: 0, count: seq)
  let bufL = device.makeBuffer(bytes: resultL, length: resultL.count * 4, options: .storageModeShared)!
  let dummy = device.makeBuffer(length: 4, options: .storageModeShared)!

  var kScale: [Float] = [scale]
  var vScale: [Float] = [scale]
  let bufKScale = device.makeBuffer(bytes: &kScale, length: 4, options: .storageModeShared)!
  let bufVScale = device.makeBuffer(bytes: &vScale, length: 4, options: .storageModeShared)!

  var bp: [UInt32] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, UInt32(seq), UInt32(seq)]
  let bufBP = device.makeBuffer(bytes: &bp, length: bp.count * 4, options: .storageModeShared)!

  let cmdBuf = MTLContext.global.commandQueue.makeCommandBuffer()!
  let enc = cmdBuf.makeComputeCommandEncoder()!

  enc.setBuffer(bufQ, offset: 0, index: 0)
  enc.setBuffer(bufK, offset: 0, index: 1)
  enc.setBuffer(bufV, offset: 0, index: 2)
  enc.setBuffer(bufO, offset: 0, index: 3)
  enc.setBuffer(bufL, offset: 0, index: 4)
  enc.setBuffer(dummy, offset: 0, index: 5)
  enc.setBuffer(dummy, offset: 0, index: 6)
  enc.setBuffer(dummy, offset: 0, index: 7)
  enc.setBuffer(dummy, offset: 0, index: 8)
  enc.setBuffer(dummy, offset: 0, index: 9)
  enc.setBuffer(dummy, offset: 0, index: 10)
  enc.setBuffer(dummy, offset: 0, index: 11)
  enc.setBuffer(bufKScale, offset: 0, index: 20)
  enc.setBuffer(bufVScale, offset: 0, index: 21)
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

  let oPtr = bufO.contents().bindMemory(to: Float.self, capacity: seq * D)
  for i in 0..<(seq * D) { resultO[i] = oPtr[i] }

  var errorCount = 0
  for i in 0..<(seq * D) {
    let err = abs(expectedO[i] - resultO[i])
    if err > tolerance || err.isNaN {
      if (expectedO[i].isNaN || expectedO[i].isInfinite),
         (resultO[i].isNaN || resultO[i].isInfinite) { continue }
      errorCount += 1
      if errorCount <= 5 {
        print("  NF4: error[\(i)] = \(err), expected=\(expectedO[i]), got=\(resultO[i])")
      }
    }
  }
  XCTAssertEqual(errorCount, 0, "NF4 quantized KV failed: \(errorCount) errors")
}

private func dequantizeNF4(
  _ packed: [UInt8], scale: Float, seq: Int, D: Int
) -> [Float] {
  var result = [Float](repeating: 0, count: seq * D)
  for row in 0..<seq {
    for col in stride(from: 0, to: D, by: 2) {
      let byteIdx = row * (D / 2) + col / 2
      let byte = packed[byteIdx]
      let lowNibble = Int(byte & 0x0F)
      let highNibble = Int(byte >> 4)
      result[row * D + col] = NF4_CODEBOOK[lowNibble] * scale
      result[row * D + col + 1] = NF4_CODEBOOK[highNibble] * scale
    }
  }
  return result
}

// MARK: - Helpers

private func dequantize(
  _ data: [UInt8], scale: Float, prec: GEMMOperandPrecision, seq: Int, D: Int
) -> [Float] {
  var result = [Float](repeating: 0, count: seq * D)
  for i in 0..<result.count {
    let val = data[i]
    switch prec {
    case .FP8_E4M3:
      result[i] = (Float(val) - 128.0) / 127.0 * 448.0 * scale
    case .FP8_E5M2:
      result[i] = (Float(val) - 128.0) / 127.0 * 57344.0 * scale
    case .INT8:
      result[i] = Float(Int8(bitPattern: val)) * scale
    default:
      fatalError("Unsupported quant precision: \(prec)")
    }
  }
  return result
}

private func referenceAttention(
  Q: [Float], K: [Float], V: [Float], seq: Int, D: Int
) -> [Float] {
  let scale = 1.0 / Float(D).squareRoot()

  // S = Q * K^T
  var S = [Float](repeating: 0, count: seq * seq)
  for r in 0..<seq {
    for c in 0..<seq {
      var dot: Float = 0
      for d in 0..<D { dot += Q[r * D + d] * K[c * D + d] }
      S[r * seq + c] = dot * scale
    }
  }

  // Softmax
  var P = S
  for r in 0..<seq {
    var maxVal: Float = -.infinity
    for c in 0..<seq { if P[r * seq + c] > maxVal { maxVal = P[r * seq + c] } }
    var sumExp: Float = 0
    for c in 0..<seq {
      let e = exp(P[r * seq + c] - maxVal)
      P[r * seq + c] = e
      sumExp += e
    }
    for c in 0..<seq { P[r * seq + c] /= sumExp }
  }

  // O = P * V
  var O = [Float](repeating: 0, count: seq * D)
  for r in 0..<seq {
    for d in 0..<D {
      var acc: Float = 0
      for c in 0..<seq { acc += P[r * seq + c] * V[c * D + d] }
      O[r * D + d] = acc
    }
  }
  return O
}
