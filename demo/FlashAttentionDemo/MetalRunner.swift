import Metal
import FlashAttention

struct BenchmarkResult: Identifiable {
    let id = UUID()
    let name: String
    let status: Status
    let timeMs: Double
    let detail: String

    enum Status {
        case pass, fail, running
    }
}

@MainActor
final class MetalRunner: ObservableObject {
    @Published var deviceName: String = ""
    @Published var results: [BenchmarkResult] = []
    @Published var isRunning = false

    init() {
        deviceName = MTLContext.global.device.name
    }

    func runAll() {
        guard !isRunning else { return }
        isRunning = true
        results = []

        Task.detached {
            let gemmResult = runGEMMBenchmark()
            await MainActor.run { self.results.append(gemmResult) }

            let attnResult = runAttentionBenchmark()
            await MainActor.run {
                self.results.append(attnResult)
                self.isRunning = false
            }
        }
    }
}

// MARK: - Helpers

private func makeF32Buffer(_ data: [Float]) -> MTLBuffer {
    MTLContext.global.device.makeBuffer(bytes: data, length: data.count * 4)!
}

private func ceilDiv(_ a: Int, _ b: UInt16) -> Int {
    (a + Int(b) - 1) / Int(b)
}

// MARK: - GEMM Benchmark (reverse-linking pipeline)

private func runGEMMBenchmark() -> BenchmarkResult {
    let n: UInt32 = 512
    let problemSize = Int(n)
    let precision: GEMMOperandPrecision = .FP32

    var gemmDesc = GEMMDescriptor()
    gemmDesc.loadPreviousC = false
    gemmDesc.matrixDimensions = (M: n, N: n, K: n)
    gemmDesc.memoryPrecisions = (A: precision, B: precision, C: precision)
    gemmDesc.transposeState = (A: false, B: false)

    // Register uses reverse-linking: shell .metallib + JIT visible function
    GEMMKernel.register(descriptor: gemmDesc)
    guard let cached = GEMMKernel.pipelineCache[gemmDesc] else {
        return BenchmarkResult(name: "GEMM 512x512", status: .fail, timeMs: 0,
                               detail: "Pipeline creation failed")
    }
    let kernel = cached.kernel
    let pipeline = cached.pipeline

    // Laplacian matrix A
    var A = [Float](repeating: 0, count: problemSize * problemSize)
    for d in 0..<problemSize {
        A[d * problemSize + d] = -2
        A[d * problemSize + (d + problemSize - 1) % problemSize] = 1
        A[d * problemSize + (d + problemSize + 1) % problemSize] = 1
    }

    // Random matrix B
    var B = [Float](repeating: 0, count: problemSize * problemSize)
    for i in B.indices { B[i] = Float.random(in: -1...1) }

    let bufA = makeF32Buffer(A)
    let bufB = makeF32Buffer(B)
    let bufC = makeF32Buffer([Float](repeating: 0, count: problemSize * problemSize))

    let cmdBuf = MTLContext.global.commandQueue.makeCommandBuffer()!
    let encoder = cmdBuf.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)
    encoder.setBuffer(bufA, offset: 0, index: 0)
    encoder.setBuffer(bufB, offset: 0, index: 1)
    encoder.setBuffer(bufC, offset: 0, index: 2)

    let gridSize = MTLSize(
        width: ceilDiv(problemSize, kernel.blockDimensions.N),
        height: ceilDiv(problemSize, kernel.blockDimensions.M),
        depth: 1)
    let groupSize = MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    encoder.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    let gpuTime = (cmdBuf.gpuEndTime - cmdBuf.gpuStartTime) * 1000.0

    // Validate: Laplacian(B)
    let ptr = bufC.contents().assumingMemoryBound(to: Float.self)
    var maxError: Float = 0
    for m in 0..<min(64, problemSize) {
        for col in 0..<min(64, problemSize) {
            let left = (m + problemSize - 1) % problemSize
            let right = (m + problemSize + 1) % problemSize
            let expected = B[left * problemSize + col] - 2 * B[m * problemSize + col] + B[right * problemSize + col]
            maxError = max(maxError, abs(expected - ptr[m * problemSize + col]))
        }
    }

    let passed = maxError < 1e-3
    let ops = 2 * problemSize * problemSize * problemSize
    let gflops = Int(Double(ops) / (gpuTime / 1000.0) / 1e9)

    return BenchmarkResult(
        name: "GEMM \(problemSize)x\(problemSize)",
        status: passed ? .pass : .fail,
        timeMs: gpuTime,
        detail: passed ? "\(gflops) GFLOPS, max err \(String(format: "%.1e", maxError))"
                       : "FAILED: max err \(String(format: "%.1e", maxError))")
}

// MARK: - Attention Forward Benchmark (reverse-linking pipeline)

private func runAttentionBenchmark() -> BenchmarkResult {
    let device = MTLContext.global.device
    let seqLen = 64
    let headDim = 32

    var attentionDesc = AttentionDescriptor()
    attentionDesc.lowPrecisionInputs = false
    attentionDesc.lowPrecisionIntermediates = false
    attentionDesc.matrixDimensions = (
        row: UInt32(seqLen),
        column: UInt32(seqLen),
        head: UInt16(headDim))
    attentionDesc.transposeState = (Q: false, K: false, V: false, O: false)

    let kernelDesc = attentionDesc.kernelDescriptor(type: .forward)
    let kernel = AttentionKernel(descriptor: kernelDesc)
    let source = kernel.createSource()

    // JIT compile the visible function (pure Metal, no __asm)
    let jitLibrary: MTLLibrary
    do {
        jitLibrary = try device.makeLibrary(source: source, options: nil)
    } catch {
        return BenchmarkResult(name: "Attention Forward", status: .fail, timeMs: 0,
                               detail: "JIT compile failed: \(error.localizedDescription)")
    }

    let constants = MTLFunctionConstantValues()
    attentionDesc.setFunctionConstants(constants)

    // Load shell (pre-compiled .metallib with async copy intrinsics)
    let shellLib = AttentionKernel.loadShellLibrary(device: device)
    let kernelFunction: MTLFunction
    do {
        kernelFunction = try shellLib.makeFunction(
            name: "attention", constantValues: MTLFunctionConstantValues())
    } catch {
        return BenchmarkResult(name: "Attention Forward", status: .fail, timeMs: 0,
                               detail: "Shell function failed: \(error.localizedDescription)")
    }

    // Get JIT visible function
    let visibleFunction: MTLFunction
    do {
        visibleFunction = try jitLibrary.makeFunction(
            name: "attention_body", constantValues: constants)
    } catch {
        return BenchmarkResult(name: "Attention Forward", status: .fail, timeMs: 0,
                               detail: "Visible function failed: \(error.localizedDescription)")
    }

    // Reverse-linking pipeline: shell + privateFunctions
    let pipelineDesc = MTLComputePipelineDescriptor()
    pipelineDesc.computeFunction = kernelFunction
    pipelineDesc.maxTotalThreadsPerThreadgroup = 1024

    let linkedFunctions = MTLLinkedFunctions()
    linkedFunctions.privateFunctions = [visibleFunction]
    pipelineDesc.linkedFunctions = linkedFunctions

    let pipeline: MTLComputePipelineState
    do {
        pipeline = try device.makeComputePipelineState(
            descriptor: pipelineDesc, options: [], reflection: nil)
    } catch {
        return BenchmarkResult(name: "Attention Forward", status: .fail, timeMs: 0,
                               detail: "Pipeline failed: \(error.localizedDescription)")
    }

    // Create random input data
    let operandSize = seqLen * headDim

    var Q = [Float](repeating: 0, count: operandSize)
    var K = [Float](repeating: 0, count: operandSize)
    var V = [Float](repeating: 0, count: operandSize)
    for i in 0..<operandSize {
        Q[i] = Float.random(in: -0.5...0.5)
        K[i] = Float.random(in: -0.5...0.5)
        V[i] = Float.random(in: -0.5...0.5)
    }

    let bufQ = makeF32Buffer(Q)
    let bufK = makeF32Buffer(K)
    let bufV = makeF32Buffer(V)
    let bufO = makeF32Buffer([Float](repeating: 0, count: operandSize))
    let bufL = makeF32Buffer([Float](repeating: 0, count: seqLen))
    let bufD = makeF32Buffer([Float](repeating: 0, count: seqLen))
    let bufDO = makeF32Buffer([Float](repeating: 0, count: operandSize))
    let bufDV = makeF32Buffer([Float](repeating: 0, count: operandSize))
    let bufDK = makeF32Buffer([Float](repeating: 0, count: operandSize))
    let bufDQ = makeF32Buffer([Float](repeating: 0, count: operandSize))

    let cmdBuf = MTLContext.global.commandQueue.makeCommandBuffer()!
    let encoder = cmdBuf.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)

    encoder.setBuffer(bufQ, offset: 0, index: 0)
    encoder.setBuffer(bufK, offset: 0, index: 1)
    encoder.setBuffer(bufV, offset: 0, index: 2)
    encoder.setBuffer(bufO, offset: 0, index: 3)
    encoder.setBuffer(bufL, offset: 0, index: 4)
    encoder.setBuffer(bufD, offset: 0, index: 5)
    encoder.setBuffer(bufDO, offset: 0, index: 6)
    encoder.setBuffer(bufDV, offset: 0, index: 7)
    encoder.setBuffer(bufDK, offset: 0, index: 8)
    encoder.setBuffer(bufDQ, offset: 0, index: 9)

    let blockCount = ceilDiv(seqLen, kernel.blockDimensions.parallelization)
    let gridSize = MTLSize(width: blockCount, height: 1, depth: 1)
    let groupSize = MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1)
    encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: groupSize)
    encoder.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    let gpuTime = (cmdBuf.gpuEndTime - cmdBuf.gpuStartTime) * 1000.0

    // Validate: O should not contain NaN or all-zero
    let oPtr = bufO.contents().assumingMemoryBound(to: Float.self)
    var hasNaN = false
    var allZero = true
    var sum: Float = 0
    for i in 0..<operandSize {
        let val = oPtr[i]
        if val.isNaN { hasNaN = true }
        if val != 0 { allZero = false }
        sum += val
    }

    let passed = !hasNaN && !allZero
    let detail: String
    if passed {
        let avgO = sum / Float(operandSize)
        detail = String(format: "seq=%d head=%d, avg(O)=%.4f", seqLen, headDim, avgO)
    } else if hasNaN {
        detail = "FAILED: output contains NaN"
    } else {
        detail = "FAILED: output is all zeros"
    }

    return BenchmarkResult(
        name: "Attention Forward",
        status: passed ? .pass : .fail,
        timeMs: gpuTime,
        detail: detail)
}
