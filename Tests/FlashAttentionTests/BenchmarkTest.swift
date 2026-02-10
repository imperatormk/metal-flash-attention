// Drop into Tests/FlashAttentionTests/ on the reverse-linking branch
// Run: swift test -Xswiftc -Ounchecked --filter BenchmarkTest
// Results written to /tmp/bench_monolithic_results.txt
import XCTest
import FlashAttention

final class BenchmarkTest: XCTestCase {

  func testFullBenchmark() throws {
    var report = ""
    func log(_ s: String) { print(s); report += s + "\n" }

    let chip = {
      let p = Process()
      p.executableURL = URL(fileURLWithPath: "/usr/sbin/system_profiler")
      p.arguments = ["SPHardwareDataType"]
      let pipe = Pipe(); p.standardOutput = pipe; try? p.run(); p.waitUntilExit()
      let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
      return out.components(separatedBy: "\n").first(where: { $0.contains("Chip") })?
        .components(separatedBy: ": ").last?.trimmingCharacters(in: .whitespaces) ?? "Unknown"
    }()

    log("╔══════════════════════════════════════════════════════════════╗")
    log("║      metal-flash-attention benchmark (monolithic IR)       ║")
    log("╠══════════════════════════════════════════════════════════════╣")
    log("║ Chip: \(chip.padding(toLength: 53, withPad: " ", startingAt: 0)) ║")
    log("║ macOS: \((ProcessInfo.processInfo.operatingSystemVersionString).padding(toLength: 52, withPad: " ", startingAt: 0)) ║")
    log("║ Date: \(ISO8601DateFormatter().string(from: Date()).padding(toLength: 53, withPad: " ", startingAt: 0)) ║")
    log("╚══════════════════════════════════════════════════════════════╝")
    log("")

    // --- GEMM ---
    log("┌──────────────────────────────────────────┐")
    log("│            GEMM Benchmark                │")
    log("├──────┬───────────┬────────┬──────────────┤")
    log("│ Size │ Precision │ Trans  │       GFLOPS │")
    log("├──────┼───────────┼────────┼──────────────┤")

    let sizes: [Int] = [32, 64, 128, 256, 512, 1024, 2048]
    let precisions: [(GEMMOperandPrecision, String)] = [(.FP32, "FP32"), (.BF16, "BF16")]
    let transposes: [(Bool, Bool, String)] = [
      (false, false, "A·B"),
      (false, true,  "A·Bᵀ"),
      (true,  false, "Aᵀ·B"),
    ]

    for (prec, pName) in precisions {
      for size in sizes {
        for (tA, tB, tName) in transposes {
          let n = UInt32(size)
          var desc = GEMMDescriptor()
          desc.loadPreviousC = false
          desc.matrixDimensions = (M: n, N: n, K: n)
          desc.memoryPrecisions = (A: prec, B: prec, C: prec)
          desc.transposeState = (A: tA, B: tB)
          let gflops = profileGEMM(descriptor: desc)
          let sizeS = String(size).padding(toLength: 4, withPad: " ", startingAt: 0)
          let precS = pName.padding(toLength: 9, withPad: " ", startingAt: 0)
          let transS = tName.padding(toLength: 6, withPad: " ", startingAt: 0)
          let gfS = String(gflops).leftPad(12)
          log("│ \(sizeS) │ \(precS) │ \(transS) │ \(gfS) │")
        }
      }
    }
    log("└──────┴───────────┴────────┴──────────────┘")
    log("")

    // --- Attention ---
    log("┌──────────┬────────┬──────────────┐")
    log("│          │        │              │")
    log("│  Config  │ Kernel │      GINSTRS │")
    log("├──────────┼────────┼──────────────┤")

    let configs: [(Int, Int)] = [
      (32,64), (64,64), (128,64), (256,64), (512,64), (1024,64), (2048,64), (4096,64),
      (1024,16), (1024,32), (1024,128), (1024,256),
    ]
    let kernels: [(AttentionKernelType, String)] = [
      (.forward, "FWD"), (.backwardQuery, "BWD_Q"), (.backwardKeyValue, "BWD_KV"),
    ]

    for (N, D) in configs {
      for (kt, kn) in kernels {
        let g = profileAttention(N: N, D: D, kernel: kt)
        let cfgS = "\(N)x\(D)".padding(toLength: 8, withPad: " ", startingAt: 0)
        let knS = kn.padding(toLength: 6, withPad: " ", startingAt: 0)
        let gS = String(g).leftPad(12)
        log("│ \(cfgS) │ \(knS) │ \(gS) │")
      }
    }
    log("└──────────┴────────┴──────────────┘")

    // Write to file
    let path = "/tmp/bench_monolithic_results.txt"
    try report.write(toFile: path, atomically: true, encoding: .utf8)
    print("\n>>> Results written to \(path)")
  }
}

private extension String {
  func leftPad(_ width: Int) -> String {
    if count >= width { return self }
    return String(repeating: " ", count: width - count) + self
  }
}

private func profileGEMM(descriptor: GEMMDescriptor) -> Int {
  let size = Int(descriptor.matrixDimensions!.M)
  var A = [Float](repeating: 0, count: size * size)
  var B = [Float](repeating: 0, count: size * size)
  let C = [Float](repeating: 0, count: size * size)
  for d in 0..<size {
    A[d * size + d] = -2
    A[d * size + (d + size - 1) % size] = 1
    A[d * size + (d + size + 1) % size] = 1
  }
  for i in B.indices { B[i] = Float.random(in: -1...1) }
  if descriptor.transposeState!.A { swap(&A, &B) }

  GEMMKernel.register(descriptor: descriptor)
  let (kernel, pipeline) = GEMMKernel.pipelineCache[descriptor]!
  let bufA = MTLContext.global.createBuffer(A, descriptor.memoryPrecisions!.A)
  let bufB = MTLContext.global.createBuffer(B, descriptor.memoryPrecisions!.B)
  let bufC = MTLContext.global.createBuffer(C, descriptor.memoryPrecisions!.C)

  func ceilDiv(_ a: Int, _ b: UInt16) -> Int { (a + Int(b) - 1) / Int(b) }
  let grid = MTLSize(width: ceilDiv(size, kernel.blockDimensions.N),
                     height: ceilDiv(size, kernel.blockDimensions.M), depth: 1)
  let group = MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1)

  // Warmup (3 passes)
  for _ in 0..<3 {
    let cmd = MTLContext.global.commandQueue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.setBuffer(bufB, offset: 0, index: 1)
    enc.setBuffer(bufC, offset: 0, index: 2)
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
    enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
  }

  var maxGFLOPS = 0
  for _ in 0..<15 {
    let batch = 30
    let cmd = MTLContext.global.commandQueue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)
    enc.setBuffer(bufA, offset: 0, index: 0)
    enc.setBuffer(bufB, offset: 0, index: 1)
    enc.setBuffer(bufC, offset: 0, index: 2)
    for _ in 0..<batch { enc.dispatchThreadgroups(grid, threadsPerThreadgroup: group) }
    enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
    let ops = 2 * size * size * size * batch
    let gflops = Int(Double(ops) / (cmd.gpuEndTime - cmd.gpuStartTime) / 1e9)
    maxGFLOPS = max(maxGFLOPS, gflops)
  }
  return maxGFLOPS
}

private func profileAttention(N: Int, D: Int, kernel kt: AttentionKernelType) -> Int {
  var networkDesc = NetworkDescriptor()
  networkDesc.rowDimension = N
  networkDesc.columnDimension = N
  networkDesc.headDimension = D
  let network = Network(descriptor: networkDesc)

  var desc = AttentionDescriptor()
  desc.lowPrecisionInputs = false
  desc.lowPrecisionIntermediates = false
  desc.matrixDimensions = (row: UInt32(N), column: UInt32(N), head: UInt16(D))
  desc.transposeState = (Q: false, K: false, V: false, O: false)

  let kernel = AttentionKernel(descriptor: desc.kernelDescriptor(type: kt))
  let source = kernel.createSource()
  let device = MTLContext.global.device
  let library = try! device.makeLibrary(source: source, options: nil)
  let constants = MTLFunctionConstantValues()
  desc.setFunctionConstants(constants)

  // Reverse-linking pipeline: shell .metallib + visible fn
  let shellLib = AttentionKernel.loadShellLibrary(device: device)
  let kernelFn = try! shellLib.makeFunction(
    name: "attention", constantValues: MTLFunctionConstantValues())
  let visibleFn = try! library.makeFunction(
    name: "attention_body", constantValues: constants)
  let pipelineDesc = MTLComputePipelineDescriptor()
  pipelineDesc.computeFunction = kernelFn
  pipelineDesc.maxTotalThreadsPerThreadgroup = 1024
  let linked = MTLLinkedFunctions()
  linked.privateFunctions = [visibleFn]
  pipelineDesc.linkedFunctions = linked
  let pipeline = try! device.makeComputePipelineState(
    descriptor: pipelineDesc, options: [], reflection: nil)

  let opSize = N * D
  let bufQ = MTLContext.global.createBuffer(network.Q, .FP32)
  let bufK = MTLContext.global.createBuffer(network.K, .FP32)
  let bufV = MTLContext.global.createBuffer(network.V, .FP32)
  let bufO = MTLContext.global.createBuffer([Float](repeating: 0, count: opSize), .FP32)
  let bufL = MTLContext.global.createBuffer([Float](repeating: 0, count: N), .FP32)
  let bufD = MTLContext.global.createBuffer([Float](repeating: 0, count: N), .FP32)
  let bufDO = MTLContext.global.createBuffer(network.dO, .FP32)
  let bufDV = MTLContext.global.createBuffer([Float](repeating: 0, count: opSize), .FP32)
  let bufDK = MTLContext.global.createBuffer([Float](repeating: 0, count: opSize), .FP32)
  let bufDQ = MTLContext.global.createBuffer([Float](repeating: 0, count: opSize), .FP32)

  func ceilDiv(_ a: Int, _ b: UInt16) -> Int { (a + Int(b) - 1) / Int(b) }
  let blocks = ceilDiv(N, kernel.blockDimensions.parallelization)
  let grid = MTLSize(width: blocks, height: 1, depth: 1)
  let group = MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1)

  func encode(_ enc: MTLComputeCommandEncoder) {
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(bufQ, offset: 0, index: 0)
    enc.setBuffer(bufK, offset: 0, index: 1)
    enc.setBuffer(bufV, offset: 0, index: 2)
    enc.setBuffer(bufO, offset: 0, index: 3)
    enc.setBuffer(bufL, offset: 0, index: 4)
    enc.setBuffer(bufD, offset: 0, index: 5)
    enc.setBuffer(bufDO, offset: 0, index: 6)
    enc.setBuffer(bufDV, offset: 0, index: 7)
    enc.setBuffer(bufDK, offset: 0, index: 8)
    enc.setBuffer(bufDQ, offset: 0, index: 9)
    enc.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
  }

  // Warmup (3 passes)
  for _ in 0..<3 {
    let cmd = MTLContext.global.commandQueue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    encode(enc)
    enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
  }

  var maxG = 0
  for _ in 0..<15 {
    let batch = 10
    let cmd = MTLContext.global.commandQueue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    for _ in 0..<batch { encode(enc) }
    enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
    var ops: Int
    switch kt {
    case .forward: ops = 2 * D + 5
    case .backwardQuery: ops = 3 * D + 5
    case .backwardKeyValue: ops = 4 * D + 5
    }
    ops *= N * N * batch
    let g = Int(Double(ops) / (cmd.gpuEndTime - cmd.gpuStartTime) / 1e9)
    maxG = max(maxG, g)
  }
  return maxG
}
