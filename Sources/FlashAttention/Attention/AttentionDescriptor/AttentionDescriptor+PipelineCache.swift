//
//  AttentionDescriptor+PipelineCache.swift
//  FlashAttention
//

import Metal
import MetalASM

/// Observable log for pipeline compilation events.
public class FlashAttentionLog: @unchecked Sendable {
  public static let shared = FlashAttentionLog()
  private let lock = NSLock()
  private var _entries: [String] = []
  public var onChange: (() -> Void)?

  public var entries: [String] {
    lock.lock()
    defer { lock.unlock() }
    return _entries
  }

  public func append(_ message: String) {
    lock.lock()
    _entries.append(message)
    lock.unlock()
    print(message)
    DispatchQueue.main.async { self.onChange?() }
  }

  public func clear() {
    lock.lock()
    _entries.removeAll()
    lock.unlock()
    DispatchQueue.main.async { self.onChange?() }
  }
}

extension AttentionKernel {
  public typealias PipelineValue = (
    kernel: AttentionKernel, pipeline: MTLComputePipelineState)

  public static var pipelineCache: [
    AttentionDescriptor: [AttentionKernelType: PipelineValue]] = [:]
}

extension AttentionKernel {
  /// Lazily get a pipeline for a specific kernel type.
  /// Compiles only the requested kernel on first access; cached thereafter.
  public static func pipeline(
    for descriptor: AttentionDescriptor, type: AttentionKernelType
  ) -> PipelineValue {
    if let cached = pipelineCache[descriptor]?[type] {
      return cached
    }
    compileSingle(descriptor: descriptor, type: type)
    return pipelineCache[descriptor]![type]!
  }

  /// Eagerly register all kernel types (or just forward with inferenceOnly).
  /// Use this for warmup screens where you want to pay compile cost upfront.
  public static func register(
    descriptor: AttentionDescriptor, inferenceOnly: Bool = false
  ) {
    let types: [AttentionKernelType] = inferenceOnly
      ? [.forward]
      : [.forward, .backwardQuery, .backwardKeyValue]

    for type in types {
      if pipelineCache[descriptor]?[type] != nil { continue }
      compileSingle(descriptor: descriptor, type: type)
    }
  }

  /// Compile a single kernel type for the given descriptor.
  private static func compileSingle(
    descriptor: AttentionDescriptor, type: AttentionKernelType
  ) {
    guard let matrixDimensions = descriptor.matrixDimensions,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete.")
    }

    let device = MTLContext.global.device
    let R = matrixDimensions.row
    let C = matrixDimensions.column
    let D = UInt32(matrixDimensions.head)

    var monoDesc = AttentionKernel.MonolithicDescriptor()
    monoDesc.R = R
    monoDesc.C = C
    monoDesc.leadingDimensions[.Q] = transposeState.Q ? R : D
    monoDesc.leadingDimensions[.K] = transposeState.K ? C : D
    monoDesc.leadingDimensions[.V] = transposeState.V ? C : D
    monoDesc.leadingDimensions[.O] = transposeState.O ? R : D
    monoDesc.leadingDimensions[.dO] = transposeState.O ? R : D
    monoDesc.leadingDimensions[.dV] = transposeState.V ? C : D
    monoDesc.leadingDimensions[.dK] = transposeState.K ? C : D
    monoDesc.leadingDimensions[.dQ] = transposeState.Q ? R : D

    let t1 = CFAbsoluteTimeGetCurrent()
    let kernelDesc = descriptor.kernelDescriptor(type: type)
    let kernel = AttentionKernel(descriptor: kernelDesc)

    let ir = kernel.createSource(descriptor: monoDesc)
    let t2 = CFAbsoluteTimeGetCurrent()

    #if os(macOS)
    let metallibData = try! MetalASM.assemble(
      ir: ir, platform: .macOS(version: 26))
    #elseif os(iOS)
    let metallibData = try! MetalASM.assemble(
      ir: ir, platform: .iOS(version: 26))
    #endif
    let t3 = CFAbsoluteTimeGetCurrent()

    let dispatchData = metallibData.withUnsafeBytes {
      DispatchData(bytes: $0)
    }
    let library = try! device.makeLibrary(data: dispatchData)
    let function = library.makeFunction(name: "attention")!
    let pipeline = try! device.makeComputePipelineState(function: function)
    let t4 = CFAbsoluteTimeGetCurrent()

    FlashAttentionLog.shared.append(
      String(format: "  %@: IR=%.0fms (%dKB) asm=%.0fms gpu=%.0fms",
             "\(type)", (t2-t1)*1000, ir.utf8.count / 1024, (t3-t2)*1000, (t4-t3)*1000))
    FlashAttentionLog.shared.append("    \(MetalASM._lastTiming)")

    if pipelineCache[descriptor] == nil {
      pipelineCache[descriptor] = [:]
    }
    pipelineCache[descriptor]![type] = (kernel, pipeline)
  }
}

// MARK: - Dispatch Helpers

/// BatchedParams buffer layout (15 x UInt32 = 60 bytes):
/// [numHeads, kvRepeatFactor, Q_stride, K_stride, V_stride, O_stride, L_stride, D_stride,
///  dO_stride, dV_stride, dK_stride, dQ_stride, causalOffset, R, C]
/// Strides are in elements (not bytes). For single-head: numHeads=1, all strides=0.
/// causalOffset: added to row index for causal masking (for chunked prefill where R < C).
/// R, C at bp[13..14]: used by dynamicRC kernels to avoid recompilation when dimensions change.
/// For GQA: kvRepeatFactor = numHeads / numKVHeads. K/V use kv_head_idx = q_head / kvRepeatFactor.
extension AttentionKernel {

  /// Create a BatchedParams MTLBuffer for multi-head batched attention.
  /// - Parameters:
  ///   - numHeads: Number of query attention heads.
  ///   - numKVHeads: Number of key/value heads (defaults to numHeads for MHA).
  ///   - R: Row sequence length.
  ///   - C: Column sequence length.
  ///   - D: Head dimension.
  /// - Returns: MTLBuffer containing the BatchedParams struct.
  public static func createBatchedParamsBuffer(
    numHeads: UInt32, numKVHeads: UInt32? = nil, R: UInt32, C: UInt32, D: UInt32,
    causalOffset: UInt32 = 0, quantizedKV: GEMMOperandPrecision? = nil
  ) -> MTLBuffer {
    let kvHeads = numKVHeads ?? numHeads
    let kvRepeatFactor = numHeads / kvHeads

    // For NF4, K/V pack 2 values per byte so stride is halved
    let kvD = (quantizedKV == .NF4) ? D / 2 : D

    // Strides = per-head element count for each operand
    let qStride = R * D
    let kStride = C * kvD
    let vStride = C * kvD
    let oStride = R * D
    let lStride = R
    let dStride = R
    let doStride = R * D
    let dvStride = C * D
    let dkStride = C * D
    let dqStride = R * D

    var params: [UInt32] = [
      numHeads, kvRepeatFactor,
      qStride, kStride, vStride, oStride,
      lStride, dStride,
      doStride, dvStride, dkStride, dqStride,
      causalOffset,
      R, C  // bp[13], bp[14]: used by dynamicRC kernels
    ]
    return MTLContext.global.device.makeBuffer(
      bytes: &params, length: params.count * 4,
      options: .storageModeShared)!
  }

  /// Dispatch an attention kernel with multi-head batched support.
  public static func dispatch(
    encoder: MTLComputeCommandEncoder,
    kernel: AttentionKernel,
    pipeline: MTLComputePipelineState,
    batchedParams: MTLBuffer,
    parallelizationDimension: Int,
    numHeads: Int = 1,
    batchSize: Int = 1
  ) {
    encoder.setComputePipelineState(pipeline)
    encoder.setThreadgroupMemoryLength(
      Int(kernel.threadgroupMemoryAllocation), index: 0)
    encoder.setBuffer(batchedParams, offset: 0, index: 30)

    let blockCount = (parallelizationDimension + Int(kernel.blockDimensions.parallelization) - 1)
      / Int(kernel.blockDimensions.parallelization)
    let gridSize = MTLSize(
      width: blockCount,
      height: numHeads,
      depth: batchSize)
    let groupSize = MTLSize(
      width: Int(kernel.threadgroupSize),
      height: 1,
      depth: 1)
    encoder.dispatchThreadgroups(
      gridSize, threadsPerThreadgroup: groupSize)
  }
}
