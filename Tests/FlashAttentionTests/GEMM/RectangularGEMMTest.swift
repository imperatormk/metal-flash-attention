import XCTest
import FlashAttention

/// Test rectangular GEMM with dimensions matching DiT workloads.
/// Verifies correctness by spot-checking rows rather than full O(MNK) reference.
final class RectangularGEMMTest: XCTestCase {

  /// Test FP32×FP32→FP32 GEMM with DiT-like rectangular dimensions.
  func testRectangularF32() throws {
    let configs: [(M: Int, K: Int, N: Int, String)] = [
      (128, 4096, 2048, "captionProj1"),
      (512, 128, 2048, "patchifyProj"),
      (512, 2048, 8192, "FFN_proj"),
    ]

    for (M, K, N, label) in configs {
      let normFactor = 1.0 / Float(K).squareRoot()
      var A = [Float](repeating: 0, count: M * K)
      var B = [Float](repeating: 0, count: N * K)

      for i in A.indices { A[i] = Float.random(in: -2..<2) * normFactor }
      for i in B.indices { B[i] = Float.random(in: -1..<1) * normFactor }

      // GPU GEMM
      var gemmDesc = GEMMDescriptor()
      gemmDesc.matrixDimensions = (M: UInt32(M), N: UInt32(N), K: UInt32(K))
      gemmDesc.memoryPrecisions = (A: .FP32, B: .FP32, C: .FP32)
      gemmDesc.transposeState = (A: false, B: true)
      gemmDesc.quantizedB = false

      let bufA = MTLContext.global.createBuffer(A, .FP32)
      let bufB = MTLContext.global.createBuffer(B, .FP32)
      let bufC = MTLContext.global.device.makeBuffer(
        length: M * N * 4, options: .storageModeShared)!

      GEMMKernel.register(descriptor: gemmDesc)
      let (kernel, pipeline) = GEMMKernel.pipelineCache[gemmDesc]!

      let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
      let encoder = commandBuffer.makeComputeCommandEncoder()!
      encoder.setComputePipelineState(pipeline)
      encoder.setThreadgroupMemoryLength(
        Int(kernel.threadgroupMemoryAllocation), index: 0)
      encoder.setBuffer(bufA, offset: 0, index: 0)
      encoder.setBuffer(bufB, offset: 0, index: 1)
      encoder.setBuffer(bufC, offset: 0, index: 2)
      let gridSize = MTLSize(
        width: (N + Int(kernel.blockDimensions.N) - 1) / Int(kernel.blockDimensions.N),
        height: (M + Int(kernel.blockDimensions.M) - 1) / Int(kernel.blockDimensions.M),
        depth: 1)
      encoder.dispatchThreadgroups(
        gridSize,
        threadsPerThreadgroup: MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1))
      encoder.endEncoding()
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()

      // Spot-check: compute CPU reference for a few rows
      let gpuPtr = bufC.contents().assumingMemoryBound(to: Float.self)
      let checkRows = [0, 1, M/2, M-1]
      let checkCols = [0, 1, N/2, N-1]
      var maxError: Float = 0
      var errorCount = 0

      for m in checkRows {
        for n in checkCols {
          var dot: Float = 0
          for k in 0..<K {
            dot += A[m * K + k] * B[n * K + k]
          }
          let gpuVal = gpuPtr[m * N + n]
          let err = abs(gpuVal - dot)
          maxError = max(maxError, err)
          if err > 1e-3 {
            print("  \(label) ERROR [\(m),\(n)]: cpu=\(dot) gpu=\(gpuVal) err=\(err)")
            errorCount += 1
          }
        }
      }

      print("\(label) M=\(M) K=\(K) N=\(N): maxErr=\(String(format: "%.6f", maxError)) errors=\(errorCount)/\(checkRows.count * checkCols.count)")
      XCTAssertEqual(errorCount, 0, "\(label): \(errorCount) spot-check errors, maxErr=\(maxError)")
    }
  }
}
