import XCTest
import FlashAttention
import MetalASM

/// Tests that dynamicRC kernels (R/C loaded from batch_params at runtime)
/// produce identical results to static kernels (R/C baked as literals).
final class DynamicRCTest: XCTestCase {

  /// Forward-only correctness: compile ONE dynamic kernel, run with multiple R/C.
  func testDynamicForwardCorrectness() throws {
    // Compile a dynamic kernel with large R/C (used only for block dimension selection).
    // The actual R/C come from batch_params at runtime.
    let maxSeq: UInt32 = 128
    let D: UInt16 = 32

    var attDesc = AttentionDescriptor()
    attDesc.lowPrecisionInputs = false
    attDesc.lowPrecisionIntermediates = false
    attDesc.matrixDimensions = (row: maxSeq, column: maxSeq, head: D)
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)

    // Build dynamic kernel
    let kernelDesc = attDesc.kernelDescriptor(type: .forward)
    let kernel = AttentionKernel(descriptor: kernelDesc)

    var monoDesc = AttentionKernel.MonolithicDescriptor()
    monoDesc.R = maxSeq
    monoDesc.C = maxSeq
    monoDesc.dynamicRC = true
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

    // Test multiple sequence lengths with the SAME compiled kernel
    let testCases: [(R: Int, C: Int)] = [
      (8, 8), (10, 10), (23, 23), (32, 32), (64, 64),
      (10, 32), (32, 10), (25, 64), (64, 25),
    ]

    for (R, C) in testCases {
      let headDim = Int(D)
      var networkDesc = NetworkDescriptor()
      networkDesc.rowDimension = R
      networkDesc.columnDimension = C
      networkDesc.headDimension = headDim
      let network = Network(descriptor: networkDesc)

      // Expected (CPU reference)
      let expectedO = network.inferenceAttention()

      // Create GPU buffers
      let memPrec = attDesc.memoryPrecisions
      let bufQ = MTLContext.global.createBuffer(network.Q, memPrec[.Q]!)
      let bufK = MTLContext.global.createBuffer(network.K, memPrec[.K]!)
      let bufV = MTLContext.global.createBuffer(network.V, memPrec[.V]!)
      var resultO = [Float](repeating: .nan, count: R * headDim)
      let bufO = MTLContext.global.createBuffer(resultO, memPrec[.O]!)
      let resultL = [Float](repeating: 0, count: R)
      let bufL = MTLContext.global.createBuffer(resultL, memPrec[.L]!)
      let dummy = device.makeBuffer(length: 4, options: .storageModeShared)!

      // Batch params with dynamic R, C at indices 13, 14
      var bp: [UInt32] = [
        1, 1,  // numHeads, kvRepeatFactor
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // strides (all 0 for single-head)
        0,  // causalOffset
        UInt32(R), UInt32(C)  // dynamic R, C
      ]
      let bufBP = device.makeBuffer(bytes: &bp, length: bp.count * 4, options: .storageModeShared)!

      let cmdQueue = MTLContext.global.commandQueue
      let cmdBuf = cmdQueue.makeCommandBuffer()!
      let enc = cmdBuf.makeComputeCommandEncoder()!

      enc.setBuffer(bufQ, offset: 0, index: 0)
      enc.setBuffer(bufK, offset: 0, index: 1)
      enc.setBuffer(bufV, offset: 0, index: 2)
      enc.setBuffer(bufO, offset: 0, index: 3)
      enc.setBuffer(bufL, offset: 0, index: 4)
      enc.setBuffer(dummy, offset: 0, index: 5)  // D_buf (unused in fwd)
      enc.setBuffer(dummy, offset: 0, index: 6)  // dO
      enc.setBuffer(dummy, offset: 0, index: 7)  // dV
      enc.setBuffer(dummy, offset: 0, index: 8)  // dK
      enc.setBuffer(dummy, offset: 0, index: 9)  // dQ
      enc.setBuffer(bufBP, offset: 0, index: 10)

      enc.setComputePipelineState(pipeline)
      enc.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)

      let blockCount = (R + Int(kernel.blockDimensions.parallelization) - 1)
        / Int(kernel.blockDimensions.parallelization)
      enc.dispatchThreadgroups(
        MTLSize(width: blockCount, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1))

      enc.endEncoding()
      cmdBuf.commit()
      cmdBuf.waitUntilCompleted()

      // Read back and compare
      MTLContext.copy(bufO, into: &resultO, precision: memPrec[.O]!)

      var errorCount = 0
      for i in 0..<(R * headDim) {
        let err = abs(expectedO[i] - resultO[i])
        if err > 2e-5 || err.isNaN {
          if (expectedO[i].isNaN || expectedO[i].isInfinite),
             (resultO[i].isNaN || resultO[i].isInfinite) { continue }
          errorCount += 1
          if errorCount <= 3 {
            print("  R=\(R) C=\(C): error[\(i)] = \(err), expected=\(expectedO[i]), got=\(resultO[i])")
          }
        }
      }
      XCTAssertEqual(errorCount, 0, "dynamicRC forward failed for R=\(R) C=\(C): \(errorCount) errors")
    }
  }

  /// Verify that a single dynamic kernel can be reused across different R/C
  /// without recompilation (the whole point of this feature).
  func testSingleKernelMultipleSizes() throws {
    let D: UInt16 = 64

    var attDesc = AttentionDescriptor()
    attDesc.lowPrecisionInputs = false
    attDesc.lowPrecisionIntermediates = false
    attDesc.matrixDimensions = (row: 256, column: 256, head: D)
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)

    let kernelDesc = attDesc.kernelDescriptor(type: .forward)
    let kernel = AttentionKernel(descriptor: kernelDesc)

    var monoDesc = AttentionKernel.MonolithicDescriptor()
    monoDesc.R = 256
    monoDesc.C = 256
    monoDesc.dynamicRC = true
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

    // Run with 3 different sizes — same pipeline, just different batch params
    for seq in [32, 93, 128] {
      let headDim = Int(D)
      var networkDesc = NetworkDescriptor()
      networkDesc.rowDimension = seq
      networkDesc.columnDimension = seq
      networkDesc.headDimension = headDim
      let network = Network(descriptor: networkDesc)
      let expectedO = network.inferenceAttention()

      let memPrec = attDesc.memoryPrecisions
      let bufQ = MTLContext.global.createBuffer(network.Q, memPrec[.Q]!)
      let bufK = MTLContext.global.createBuffer(network.K, memPrec[.K]!)
      let bufV = MTLContext.global.createBuffer(network.V, memPrec[.V]!)
      var resultO = [Float](repeating: .nan, count: seq * headDim)
      let bufO = MTLContext.global.createBuffer(resultO, memPrec[.O]!)
      let resultL = [Float](repeating: 0, count: seq)
      let bufL = MTLContext.global.createBuffer(resultL, memPrec[.L]!)
      let dummy = device.makeBuffer(length: 4, options: .storageModeShared)!

      var bp: [UInt32] = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          UInt32(seq), UInt32(seq)]
      let bufBP = device.makeBuffer(bytes: &bp, length: bp.count * 4, options: .storageModeShared)!

      let cmdBuf = MTLContext.global.commandQueue.makeCommandBuffer()!
      let enc = cmdBuf.makeComputeCommandEncoder()!
      for i in 0..<10 { enc.setBuffer(dummy, offset: 0, index: i) }
      enc.setBuffer(bufQ, offset: 0, index: 0)
      enc.setBuffer(bufK, offset: 0, index: 1)
      enc.setBuffer(bufV, offset: 0, index: 2)
      enc.setBuffer(bufO, offset: 0, index: 3)
      enc.setBuffer(bufL, offset: 0, index: 4)
      enc.setBuffer(bufBP, offset: 0, index: 10)

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
          if errorCount <= 3 {
            print("  seq=\(seq): error[\(i)] = \(err), expected=\(expectedO[i]), got=\(resultO[i])")
          }
        }
      }
      XCTAssertEqual(errorCount, 0, "dynamicRC reuse failed for seq=\(seq): \(errorCount) errors")
    }
  }
}
