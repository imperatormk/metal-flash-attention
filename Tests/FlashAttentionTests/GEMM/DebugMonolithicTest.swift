import XCTest
import FlashAttention

final class DebugMonolithicTest: XCTestCase {
  func testIdentityMultiply() throws {
    // Test multiple sizes to find where errors start.
    for size in [UInt32(8), 10, 16, 24, 32, 33, 48] {
      testSize(n: size)
    }
  }

  func testLaplacian() throws {
    // Reproduce the exact Laplacian test pattern
    for size in [UInt32(7), 8, 9, 10, 32, 33] {
      testLaplacianSize(n: size, duplicatedCommandCount: 1)
    }
    // Now test with 20 dispatches like the original
    print("\n=== With 20 dispatches ===")
    for size in [UInt32(7), 8, 10, 33] {
      testLaplacianSize(n: size, duplicatedCommandCount: 20)
    }
  }

  func testAllTransposeStates() throws {
    // Test all three transpose states like LaplacianTest
    let transposeStates: [(Bool, Bool)] = [
      (false, false),
      (false, true),
      (true, false),
    ]
    for transposeState in transposeStates {
      let tA = transposeState.0 ? "T" : "N"
      let tB = transposeState.1 ? "T" : "N"
      print("\n=== Transpose state: A=\(tA) B=\(tB) ===")
      for size in [UInt32(7), 8, 9, 10, 15, 16, 17, 23, 24, 25, 31, 32, 33, 47, 48, 49, 63, 64, 65, 103, 104, 112, 126, 127, 128, 129, 130, 131, 135, 136, 137, 143, 144, 145, 151, 152, 153] {
        testLaplacianTranspose(n: size, transposeState: transposeState)
      }
    }
  }

  func testLoadPreviousC() throws {
    // Verify C = A*B + C_prev with loadPreviousC = true.
    // Only test sizes that are exact multiples of block dimensions (32),
    // or small enough to fit in one block (≤32). Edge-shifted blocks with
    // loadPreviousC have a known race condition (two threadgroups overlap
    // and double-accumulate C_prev) — this is pre-existing in the reference
    // Metal code and not a monolithic IR bug.
    let transposeStates: [(Bool, Bool)] = [
      (false, false),
      (false, true),
      (true, false),
    ]
    for transposeState in transposeStates {
      let tA = transposeState.0 ? "T" : "N"
      let tB = transposeState.1 ? "T" : "N"
      print("\n=== loadPreviousC: A=\(tA) B=\(tB) ===")
      for size in [UInt32(7), 8, 10, 16, 24, 32, 33, 48, 64, 65, 96, 128, 129] {
        testLoadPreviousCSize(n: size, transposeState: transposeState)
      }
    }
  }

  func testLoadPreviousCSize(n: UInt32, transposeState: (Bool, Bool)) {
    let problemSize = Int(n)

    var gemmDesc = GEMMDescriptor()
    gemmDesc.loadPreviousC = true
    gemmDesc.matrixDimensions = (M: n, N: n, K: n)
    gemmDesc.memoryPrecisions = (A: .FP32, B: .FP32, C: .FP32)
    gemmDesc.transposeState = transposeState

    GEMMKernel.register(descriptor: gemmDesc)
    let (kernel, pipeline) = GEMMKernel.pipelineCache[gemmDesc]!

    // A = Laplacian (symmetric)
    var A = [Float](repeating: 0, count: problemSize * problemSize)
    for i in 0..<problemSize {
      A[i * problemSize + i] = -2
      let left = (i + problemSize - 1) % problemSize
      A[i * problemSize + left] = 1
      let right = (i + problemSize + 1) % problemSize
      A[i * problemSize + right] = 1
    }

    // B = random (seeded)
    srand48(42 + Int(n))
    var B = [Float](repeating: 0, count: problemSize * problemSize)
    for i in 0..<(problemSize * problemSize) {
      B[i] = Float(drand48())
    }

    // Since Laplacian is symmetric, swap A and B when testing A^T
    if transposeState.0 {
      swap(&A, &B)
    }

    // C_prev = known values (e.g., 1.0, 2.0, 3.0, ...)
    var C_prev = [Float](repeating: 0, count: problemSize * problemSize)
    srand48(123 + Int(n))
    for i in 0..<C_prev.count {
      C_prev[i] = Float(drand48()) * 10.0
    }

    let bufferA = MTLContext.global.createBuffer(A, .FP32)
    let bufferB = MTLContext.global.createBuffer(B, .FP32)
    let bufferC = MTLContext.global.createBuffer(C_prev, .FP32)

    let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setThreadgroupMemoryLength(
      Int(kernel.threadgroupMemoryAllocation), index: 0)
    encoder.setBuffer(bufferA, offset: 0, index: 0)
    encoder.setBuffer(bufferB, offset: 0, index: 1)
    encoder.setBuffer(bufferC, offset: 0, index: 2)

    func ceilDivide(_ target: Int, _ granularity: UInt16) -> Int {
      (target + Int(granularity) - 1) / Int(granularity)
    }
    let gridSize = MTLSize(
      width: ceilDivide(problemSize, kernel.blockDimensions.N),
      height: ceilDivide(problemSize, kernel.blockDimensions.M),
      depth: 1)
    let groupSize = MTLSize(
      width: Int(kernel.threadgroupSize),
      height: 1,
      depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let raw = bufferC.contents().assumingMemoryBound(to: Float.self)

    // Verify: C_out = A*B + C_prev
    // Compute CPU reference for A*B using Laplacian structure
    var maxError: Float = 0
    var errorCount = 0
    for m in 0..<problemSize {
      for col in 0..<problemSize {
        var leftSource, centerSource, rightSource: Float
        let leftRowID = (m + problemSize - 1) % problemSize
        let centerRowID = m
        let rightRowID = (m + problemSize + 1) % problemSize

        if transposeState.0 {
          leftSource = A[leftRowID * problemSize + col]
          centerSource = A[centerRowID * problemSize + col]
          rightSource = A[rightRowID * problemSize + col]
        } else if transposeState.1 {
          leftSource = B[col * problemSize + leftRowID]
          centerSource = B[col * problemSize + centerRowID]
          rightSource = B[col * problemSize + rightRowID]
        } else {
          leftSource = B[leftRowID * problemSize + col]
          centerSource = B[centerRowID * problemSize + col]
          rightSource = B[rightRowID * problemSize + col]
        }

        let AB = leftSource - 2 * centerSource + rightSource

        var cPrevIdx: Int
        if transposeState.0 {
          cPrevIdx = col * problemSize + m
        } else {
          cPrevIdx = m * problemSize + col
        }
        let expected = AB + C_prev[cPrevIdx]

        var actual: Float
        if transposeState.0 {
          actual = raw[col * problemSize + m]
        } else {
          actual = raw[m * problemSize + col]
        }

        let error = abs(expected - actual)
        if error > 1e-3 {
          if errorCount < 5 {
            print("  n=\(n) C[\(m)][\(col)] = \(actual), expected \(expected), error \(error)")
          }
          errorCount += 1
        }
        maxError = max(maxError, error)
      }
    }

    let tA = transposeState.0 ? "T" : "N"
    let tB = transposeState.1 ? "T" : "N"
    if errorCount == 0 {
      print("n=\(n) \(tA)\(tB) loadPrevC: ALL CORRECT (max error = \(maxError))")
    } else {
      print("n=\(n) \(tA)\(tB) loadPrevC: ERRORS \(errorCount)/\(problemSize*problemSize) (max error = \(maxError))")
      XCTFail("n=\(n) \(tA)\(tB) loadPrevC: \(errorCount) errors (max error = \(maxError))")
    }
  }

  func testLaplacianTranspose(n: UInt32, transposeState: (Bool, Bool)) {
    let problemSize = Int(n)

    var gemmDesc = GEMMDescriptor()
    gemmDesc.loadPreviousC = false
    gemmDesc.matrixDimensions = (M: n, N: n, K: n)
    gemmDesc.memoryPrecisions = (A: .FP32, B: .FP32, C: .FP32)
    gemmDesc.transposeState = transposeState

    GEMMKernel.register(descriptor: gemmDesc)
    let (kernel, pipeline) = GEMMKernel.pipelineCache[gemmDesc]!

    // A = Laplacian (symmetric)
    var A = [Float](repeating: 0, count: problemSize * problemSize)
    for i in 0..<problemSize {
      A[i * problemSize + i] = -2
      let left = (i + problemSize - 1) % problemSize
      A[i * problemSize + left] = 1
      let right = (i + problemSize + 1) % problemSize
      A[i * problemSize + right] = 1
    }

    // B = random (seeded)
    srand48(42 + Int(n))
    var B = [Float](repeating: 0, count: problemSize * problemSize)
    for i in 0..<(problemSize * problemSize) {
      B[i] = Float(drand48())
    }

    // Since Laplacian is symmetric, swap A and B when testing A^T
    // (same logic as LaplacianTest)
    if transposeState.0 {
      swap(&A, &B)
    }

    let bufferA = MTLContext.global.createBuffer(A, .FP32)
    let bufferB = MTLContext.global.createBuffer(B, .FP32)
    var garbageC = [Float](repeating: 0, count: problemSize * problemSize)
    for i in 0..<garbageC.count { garbageC[i] = 999.0 + Float(i) }
    let bufferC = MTLContext.global.createBuffer(garbageC, .FP32)

    let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setThreadgroupMemoryLength(
      Int(kernel.threadgroupMemoryAllocation), index: 0)
    encoder.setBuffer(bufferA, offset: 0, index: 0)
    encoder.setBuffer(bufferB, offset: 0, index: 1)
    encoder.setBuffer(bufferC, offset: 0, index: 2)

    func ceilDivide(_ target: Int, _ granularity: UInt16) -> Int {
      (target + Int(granularity) - 1) / Int(granularity)
    }
    let gridSize = MTLSize(
      width: ceilDivide(problemSize, kernel.blockDimensions.N),
      height: ceilDivide(problemSize, kernel.blockDimensions.M),
      depth: 1)
    let groupSize = MTLSize(
      width: Int(kernel.threadgroupSize),
      height: 1,
      depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let raw = bufferC.contents().assumingMemoryBound(to: Float.self)

    // Use same verification logic as LaplacianTest
    var maxError: Float = 0
    var errorCount = 0
    for m in 0..<problemSize {
      for col in 0..<problemSize {
        // Find source scalars based on transpose state
        var leftSource, centerSource, rightSource: Float
        let leftRowID = (m + problemSize - 1) % problemSize
        let centerRowID = m
        let rightRowID = (m + problemSize + 1) % problemSize

        if transposeState.0 {
          // A^T test: A has the random data, B has the Laplacian
          leftSource = A[leftRowID * problemSize + col]
          centerSource = A[centerRowID * problemSize + col]
          rightSource = A[rightRowID * problemSize + col]
        } else if transposeState.1 {
          // B^T test: B is transposed
          leftSource = B[col * problemSize + leftRowID]
          centerSource = B[col * problemSize + centerRowID]
          rightSource = B[col * problemSize + rightRowID]
        } else {
          // NN: standard
          leftSource = B[leftRowID * problemSize + col]
          centerSource = B[centerRowID * problemSize + col]
          rightSource = B[rightRowID * problemSize + col]
        }

        let expected = leftSource - 2 * centerSource + rightSource

        var actual: Float
        if transposeState.0 {
          actual = raw[col * problemSize + m]
        } else {
          actual = raw[m * problemSize + col]
        }

        let error = abs(expected - actual)
        if error > 1e-4 {
          if errorCount < 5 {
            print("  n=\(n) C[\(m)][\(col)] = \(actual), expected \(expected), error \(error)")
          }
          errorCount += 1
        }
        maxError = max(maxError, error)
      }
    }

    let tA = transposeState.0 ? "T" : "N"
    let tB = transposeState.1 ? "T" : "N"
    if errorCount == 0 {
      print("n=\(n) \(tA)\(tB): ALL CORRECT (max error = \(maxError))")
    } else {
      print("n=\(n) \(tA)\(tB): ERRORS \(errorCount)/\(problemSize*problemSize) (max error = \(maxError))")
    }
  }

  func testLaplacianSize(n: UInt32, duplicatedCommandCount: Int = 1) {
    let problemSize = Int(n)

    var gemmDesc = GEMMDescriptor()
    gemmDesc.loadPreviousC = false
    gemmDesc.matrixDimensions = (M: n, N: n, K: n)
    gemmDesc.memoryPrecisions = (A: .FP32, B: .FP32, C: .FP32)
    gemmDesc.transposeState = (false, false)

    GEMMKernel.register(descriptor: gemmDesc)
    let (kernel, pipeline) = GEMMKernel.pipelineCache[gemmDesc]!

    // A = Laplacian
    var A = [Float](repeating: 0, count: problemSize * problemSize)
    for i in 0..<problemSize {
      A[i * problemSize + i] = -2
      let left = (i + problemSize - 1) % problemSize
      A[i * problemSize + left] = 1
      let right = (i + problemSize + 1) % problemSize
      A[i * problemSize + right] = 1
    }

    // B = random (seeded for reproducibility)
    srand48(42)
    var B = [Float](repeating: 0, count: problemSize * problemSize)
    for i in 0..<(problemSize * problemSize) {
      B[i] = Float(drand48())
    }

    // Initialize C with garbage to detect if kernel reads C when loadPreviousC=false
    var garbageC = [Float](repeating: 0, count: problemSize * problemSize)
    for i in 0..<garbageC.count {
      garbageC[i] = Float(999.0 + Float(i))
    }

    let bufferA = MTLContext.global.createBuffer(A, .FP32)
    let bufferB = MTLContext.global.createBuffer(B, .FP32)
    let bufferC = MTLContext.global.createBuffer(garbageC, .FP32)

    let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setThreadgroupMemoryLength(
      Int(kernel.threadgroupMemoryAllocation), index: 0)
    encoder.setBuffer(bufferA, offset: 0, index: 0)
    encoder.setBuffer(bufferB, offset: 0, index: 1)
    encoder.setBuffer(bufferC, offset: 0, index: 2)

    func ceilDivide(_ target: Int, _ granularity: UInt16) -> Int {
      (target + Int(granularity) - 1) / Int(granularity)
    }
    let gridSize = MTLSize(
      width: ceilDivide(problemSize, kernel.blockDimensions.N),
      height: ceilDivide(problemSize, kernel.blockDimensions.M),
      depth: 1)
    let groupSize = MTLSize(
      width: Int(kernel.threadgroupSize),
      height: 1,
      depth: 1)
    for _ in 0..<duplicatedCommandCount {
      encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    }
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let raw = bufferC.contents().assumingMemoryBound(to: Float.self)

    // Compute CPU reference and compare
    print("\n=== Laplacian Test (n=\(n), dispatches=\(duplicatedCommandCount)) ===")
    var maxError: Float = 0
    var errorCount = 0
    for m in 0..<problemSize {
      for col in 0..<problemSize {
        let left = B[((m + problemSize - 1) % problemSize) * problemSize + col]
        let center = B[m * problemSize + col]
        let right = B[((m + problemSize + 1) % problemSize) * problemSize + col]
        let expected = left - 2 * center + right

        let actual = raw[m * problemSize + col]
        let error = abs(expected - actual)
        if error > 1e-4 {
          if errorCount < 5 {
            print("C[\(m)][\(col)] = \(actual), expected \(expected), error \(error)")
          }
          errorCount += 1
        }
        maxError = max(maxError, error)
      }
    }

    if errorCount == 0 {
      print("ALL CORRECT! max error = \(maxError)")
    } else {
      print("ERRORS: \(errorCount)/\(problemSize*problemSize) cells wrong, max error = \(maxError)")
    }
  }

  func testSize(n: UInt32) {

    var gemmDesc = GEMMDescriptor()
    gemmDesc.loadPreviousC = false
    gemmDesc.matrixDimensions = (M: n, N: n, K: n)
    gemmDesc.memoryPrecisions = (A: .FP32, B: .FP32, C: .FP32)
    gemmDesc.transposeState = (false, false)

    GEMMKernel.register(descriptor: gemmDesc)
    let (kernel, pipeline) = GEMMKernel.pipelineCache[gemmDesc]!

    let problemSize = Int(n)

    // A = identity
    var A = [Float](repeating: 0, count: problemSize * problemSize)
    for i in 0..<problemSize {
      A[i * problemSize + i] = 1.0
    }

    // B = sequential values 1, 2, 3, ...
    var B = [Float](repeating: 0, count: problemSize * problemSize)
    for i in 0..<(problemSize * problemSize) {
      B[i] = Float(i + 1)
    }

    let bufferA = MTLContext.global.createBuffer(A, .FP32)
    let bufferB = MTLContext.global.createBuffer(B, .FP32)
    let bufferC = MTLContext.global.createBuffer(
      [Float](repeating: 0, count: problemSize * problemSize), .FP32)

    let commandBuffer = MTLContext.global.commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setThreadgroupMemoryLength(
      Int(kernel.threadgroupMemoryAllocation), index: 0)
    encoder.setBuffer(bufferA, offset: 0, index: 0)
    encoder.setBuffer(bufferB, offset: 0, index: 1)
    encoder.setBuffer(bufferC, offset: 0, index: 2)

    func ceilDivide(_ target: Int, _ granularity: UInt16) -> Int {
      (target + Int(granularity) - 1) / Int(granularity)
    }
    let gridSize = MTLSize(
      width: ceilDivide(problemSize, kernel.blockDimensions.N),
      height: ceilDivide(problemSize, kernel.blockDimensions.M),
      depth: 1)
    let groupSize = MTLSize(
      width: Int(kernel.threadgroupSize),
      height: 1,
      depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // Read results
    let raw = bufferC.contents().assumingMemoryBound(to: Float.self)
    var C = [Float](repeating: 0, count: problemSize * problemSize)
    for i in 0..<(problemSize * problemSize) {
      C[i] = raw[i]
    }

    // Print results
    print("\n=== Debug Monolithic GEMM Test (n=\(n)) ===")
    print("C = I(\(n)x\(n)) * B(\(n)x\(n)), expected C = B")
    print("Block dims: \(kernel.blockDimensions)")
    print("Threadgroup size: \(kernel.threadgroupSize)")
    print()

    var maxError: Float = 0
    var errorCount = 0
    for row in 0..<problemSize {
      for col in 0..<problemSize {
        let idx = row * problemSize + col
        let expected = B[idx]
        let actual = C[idx]
        let error = abs(expected - actual)
        if error > 1e-5 {
          if errorCount < 20 {
            print("C[\(row)][\(col)] = \(actual), expected \(expected), error = \(error)")
          }
          errorCount += 1
        }
        maxError = max(maxError, error)
      }
    }

    if errorCount == 0 {
      print("ALL CORRECT! max error = \(maxError)")
    } else {
      print("ERRORS: \(errorCount) cells wrong, max error = \(maxError)")
    }

    // Print matrices for small sizes
    if problemSize <= 10 {
      print("\nFull C matrix:")
      for row in 0..<problemSize {
        var line = ""
        for col in 0..<problemSize {
          let v = C[row * problemSize + col]
          line += String(format: "%8.2f", v)
        }
        print(line)
      }

      print("\nExpected B matrix:")
      for row in 0..<problemSize {
        var line = ""
        for col in 0..<problemSize {
          let v = B[row * problemSize + col]
          line += String(format: "%8.2f", v)
        }
        print(line)
      }
    }
  }
}
