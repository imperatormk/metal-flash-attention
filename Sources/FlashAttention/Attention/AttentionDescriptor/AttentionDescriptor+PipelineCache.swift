//
//  AttentionDescriptor+PipelineCache.swift
//  FlashAttention
//

import Metal
import MetalASM

extension AttentionKernel {
  public typealias PipelineValue = (
    kernel: AttentionKernel, pipeline: MTLComputePipelineState)

  public static var pipelineCache: [
    AttentionDescriptor: [AttentionKernelType: PipelineValue]] = [:]
}

extension AttentionKernel {
  /// Register all three kernel types (forward, backwardQuery, backwardKeyValue)
  /// for the given attention descriptor.
  public static func register(descriptor: AttentionDescriptor) {
    if pipelineCache[descriptor] != nil {
      return
    }

    guard let matrixDimensions = descriptor.matrixDimensions,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete.")
    }

    let device = MTLContext.global.device
    let R = matrixDimensions.row
    let C = matrixDimensions.column
    let D = UInt32(matrixDimensions.head)

    // Build the MonolithicDescriptor (shared across all kernel types).
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

    var entries: [AttentionKernelType: PipelineValue] = [:]

    for type in [AttentionKernelType.forward, .backwardQuery, .backwardKeyValue] {
      let kernelDesc = descriptor.kernelDescriptor(type: type)
      let kernel = AttentionKernel(descriptor: kernelDesc)

      let ir = kernel.createMonolithicIR(descriptor: monoDesc)

      #if os(macOS)
      let metallibData = try! MetalASM.assemble(
        ir: ir, platform: .macOS(version: 26))
      #elseif os(iOS)
      let metallibData = try! MetalASM.assemble(
        ir: ir, platform: .iOS(version: 26))
      #endif

      let dispatchData = metallibData.withUnsafeBytes {
        DispatchData(bytes: $0)
      }
      let library = try! device.makeLibrary(data: dispatchData)
      let function = library.makeFunction(name: "attention")!
      let pipeline = try! device.makeComputePipelineState(function: function)

      entries[type] = (kernel, pipeline)
    }

    pipelineCache[descriptor] = entries
  }
}
