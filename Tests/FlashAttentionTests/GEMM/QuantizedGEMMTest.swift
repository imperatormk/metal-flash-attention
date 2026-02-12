import XCTest
import FlashAttention
import MetalASM

/// Tests for quantized (4-bit) GEMM: C[M,N] = A[M,K] @ W[N,K]^T
/// where W is MLX-style 4-bit packed (uint32), with f16 scales and biases.
final class QuantizedGEMMTest: XCTestCase {

  /// Basic correctness test: small matrix, known dimensions.
  func testQuantizedGEMMCorrectness() throws {
    // Test several (M, N, K) combos with groupSize=64
    let testCases: [(M: UInt32, N: UInt32, K: UInt32)] = [
      (1, 64, 64),       // decode-like: single token
      (4, 64, 64),       // small batch
      (8, 128, 128),     // medium
      (16, 64, 256),     // K > N
      (32, 256, 128),    // prefill-like
    ]

    for (M, N, K) in testCases {
      let groupSize: UInt32 = 64
      try runQuantizedTest(M: M, N: N, K: K, groupSize: groupSize)
    }
  }

  /// Test with non-aligned K (K not multiple of K_group).
  func testQuantizedGEMMEdgeK() throws {
    // K=128 with groupSize=64, should work with edge tiles
    try runQuantizedTest(M: 4, N: 64, K: 128, groupSize: 64)
  }

  private func runQuantizedTest(
    M: UInt32, N: UInt32, K: UInt32, groupSize: UInt32
  ) throws {
    // K must be divisible by 8 (packing) and groupSize
    precondition(K % 8 == 0)
    precondition(K % groupSize == 0)

    let packedK = K / 8
    let groupsPerRow = K / groupSize

    // Generate random input A [M, K] as float
    var operandA = [Float](repeating: 0, count: Int(M * K))
    let norm = 1.0 / Float(K).squareRoot()
    for i in operandA.indices {
      operandA[i] = Float.random(in: -1..<1) * norm
    }

    // Generate random quantized weights
    // packed [N, K/8] uint32 — each uint32 has 8 nibbles
    var packedWeights = [UInt32](repeating: 0, count: Int(N * packedK))
    for i in packedWeights.indices {
      // Random 8 nibbles packed into uint32
      var val: UInt32 = 0
      for s in 0..<8 {
        let nibble = UInt32.random(in: 0..<16)
        val |= nibble << (s * 4)
      }
      packedWeights[i] = val
    }

    // scales [N, K/groupSize] f16, biases [N, K/groupSize] f16
    var scales = [Float16](repeating: 0, count: Int(N * groupsPerRow))
    var biases = [Float16](repeating: 0, count: Int(N * groupsPerRow))
    for i in scales.indices {
      scales[i] = Float16(Float.random(in: 0.01..<0.5))
      biases[i] = Float16(Float.random(in: -1..<1))
    }

    // CPU reference: dequantize weights and compute matmul
    // W[n, k] = float(nibble) * scale + bias
    var cpuC = [Float](repeating: 0, count: Int(M * N))
    for m in 0..<Int(M) {
      for n in 0..<Int(N) {
        var dot: Float = 0
        for k in 0..<Int(K) {
          let a = operandA[m * Int(K) + k]

          let packIdx = k / 8
          let subIdx = k % 8
          let packed = packedWeights[n * Int(packedK) + packIdx]
          let nibble = (packed >> (subIdx * 4)) & 0xF

          let gIdx = k / Int(groupSize)
          let scale = Float(scales[n * Int(groupsPerRow) + gIdx])
          let bias = Float(biases[n * Int(groupsPerRow) + gIdx])

          let w = Float(nibble) * scale + bias
          dot += a * w
        }
        cpuC[m * Int(N) + n] = dot
      }
    }

    // Create GPU buffers
    let device = MTLContext.global.device

    let bufA = MTLContext.global.createBuffer(operandA, .FP16)
    let bufW = device.makeBuffer(
      bytes: packedWeights,
      length: packedWeights.count * 4,
      options: .storageModeShared)!
    let bufScales = device.makeBuffer(
      bytes: scales,
      length: scales.count * 2,
      options: .storageModeShared)!
    let bufBiases = device.makeBuffer(
      bytes: biases,
      length: biases.count * 2,
      options: .storageModeShared)!
    var resultC = [Float](repeating: .nan, count: Int(M * N))
    let bufC = MTLContext.global.createBuffer(resultC, .FP32)

    // Compile quantized GEMM kernel
    var gemmDesc = GEMMDescriptor()
    gemmDesc.matrixDimensions = (M: M, N: N, K: K)
    gemmDesc.memoryPrecisions = (A: .FP16, B: .FP16, C: .FP32)
    gemmDesc.transposeState = (A: false, B: true)

    var kernelDesc = GEMMKernelDescriptor(descriptor: gemmDesc)
    // On M1 (pre-Apple9), preferAsyncStore may be nil for 48x48 heuristic.
    // Force a value for direct kernel construction.
    if kernelDesc.preferAsyncStore == nil {
      kernelDesc.preferAsyncStore = false
    }
    let kernel = GEMMKernel(descriptor: kernelDesc)

    var monoDesc = GEMMKernel.MonolithicDescriptor()
    monoDesc.M = M
    monoDesc.N = N
    monoDesc.K = K
    monoDesc.leadingDimensionA = K   // A is [M, K], non-transposed, ld = K
    monoDesc.leadingDimensionB = K   // W is [N, K], transposed B, ld = K
    monoDesc.leadingDimensionC = N   // C is [M, N], non-transposed, ld = N
    monoDesc.quantizedB = true
    monoDesc.groupSize = groupSize

    let ir = kernel.createSource(descriptor: monoDesc)
    #if os(macOS)
    let metallibData = try MetalASM.assemble(ir: ir, platform: .macOS(version: 26))
    #elseif os(iOS)
    let metallibData = try MetalASM.assemble(ir: ir, platform: .iOS(version: 26))
    #endif

    let dispatchData = metallibData.withUnsafeBytes { DispatchData(bytes: $0) }
    let library = try device.makeLibrary(data: dispatchData)
    let function = library.makeFunction(name: "gemm")!
    let pipeline = try device.makeComputePipelineState(function: function)

    // Dispatch
    let cmdBuf = MTLContext.global.commandQueue.makeCommandBuffer()!
    let enc = cmdBuf.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)

    // Buffer bindings: A(0), W(1), scales(2), biases(3), C(4)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.setBuffer(bufW, offset: 0, index: 1)
    enc.setBuffer(bufScales, offset: 0, index: 2)
    enc.setBuffer(bufBiases, offset: 0, index: 3)
    enc.setBuffer(bufC, offset: 0, index: 4)

    func ceilDiv(_ a: UInt32, _ b: UInt16) -> Int {
      (Int(a) + Int(b) - 1) / Int(b)
    }
    let gridSize = MTLSize(
      width: ceilDiv(N, kernel.blockDimensions.N),
      height: ceilDiv(M, kernel.blockDimensions.M),
      depth: 1)
    let groupSize2 = MTLSize(
      width: Int(kernel.threadgroupSize), height: 1, depth: 1)
    enc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize2)

    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    // Read back results
    let cPtr = bufC.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<Int(M * N) {
      resultC[i] = cPtr[i]
    }

    // Compare
    var errorCount = 0
    var maxErr: Float = 0
    for i in 0..<Int(M * N) {
      let err = abs(cpuC[i] - resultC[i])
      maxErr = max(maxErr, err)
      // Tolerance: f16 dequant + f16 A → expect ~1e-2 relative error
      let threshold: Float = max(0.05, abs(cpuC[i]) * 0.05)
      if err > threshold || err.isNaN {
        errorCount += 1
        if errorCount <= 5 {
          let m = i / Int(N)
          let n = i % Int(N)
          print("  M=\(M) N=\(N) K=\(K): error[\(m),\(n)] = \(err), expected=\(cpuC[i]), got=\(resultC[i])")
        }
      }
    }
    XCTAssertEqual(errorCount, 0,
      "quantized GEMM failed for M=\(M) N=\(N) K=\(K): \(errorCount)/\(M*N) errors, maxErr=\(maxErr)")
    if errorCount == 0 {
      print("  ✓ M=\(M) N=\(N) K=\(K): maxErr=\(String(format: "%.6f", maxErr))")
    }
  }
}
