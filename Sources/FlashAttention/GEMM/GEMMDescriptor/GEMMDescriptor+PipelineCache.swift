//
//  PipelineCache.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

import Metal

extension GEMMKernel {
  public typealias LibraryValue = (
    kernel: GEMMKernel, library: MTLLibrary)
  public typealias PipelineValue = (
    kernel: GEMMKernel, pipeline: MTLComputePipelineState)

  public static var libraryCache: [
    GEMMKernelDescriptor: LibraryValue] = [:]
  public static var pipelineCache: [
    GEMMDescriptor: PipelineValue] = [:]

  /// Pre-compiled IR kernel shell library (loaded once).
  public static var shellLibrary: MTLLibrary?

  /// Load the pre-compiled GEMM shell metallib.
  public static func loadShellLibrary(device: MTLDevice) -> MTLLibrary {
    if let lib = shellLibrary { return lib }

    // Look for gemm_shell_v2.metallib in the ReverseLink directory
    // relative to the package sources.
    let bundle = Bundle.module
    guard let url = bundle.url(
      forResource: "gemm_shell_v2",
      withExtension: "metallib"
    ) else {
      fatalError("""
        Could not find gemm_shell_v2.metallib. \
        Ensure it is included in the package resources.
        """)
    }
    let lib = try! device.makeLibrary(URL: url)
    shellLibrary = lib
    return lib
  }
}

extension GEMMKernel {
  // Register this problem configuration in the cache.
  public static func register(descriptor: GEMMDescriptor) {
    guard pipelineCache[descriptor] == nil else {
      return
    }

    var kernelDescriptor = GEMMKernelDescriptor(descriptor: descriptor)

    let device = MTLContext.global.device
    if device.supportsFamily(.apple9) {
      kernelDescriptor.preferAsyncStore = false
    } else {
      guard let blockDimensions = kernelDescriptor.blockDimensions else {
        fatalError("Block dimensions were not set.")
      }
      if blockDimensions == (48, 48, 32) {
        kernelDescriptor.preferAsyncStore = nil
      } else {
        kernelDescriptor.preferAsyncStore = true
      }
    }

    func createLibrary(
      _ kernelDescriptor: GEMMKernelDescriptor
    ) -> LibraryValue {
      if let output = GEMMKernel.libraryCache[kernelDescriptor] {
        return output
      } else {
        let kernel = GEMMKernel(descriptor: kernelDescriptor)
        let source = kernel.createSource()
        let library = try! device.makeLibrary(source: source, options: nil)

        let output = (kernel, library)
        GEMMKernel.libraryCache[kernelDescriptor] = output
        return output
      }
    }

    func createPipeline(
      _ libraryValue: LibraryValue
    ) -> PipelineValue {
      let constants = MTLFunctionConstantValues()
      descriptor.setFunctionConstants(constants)

      // Load the pre-compiled shell library.
      let shellLib = GEMMKernel.loadShellLibrary(device: device)

      // Get the kernel function from the shell.
      let kernelFunction = try! shellLib.makeFunction(
        name: "gemm", constantValues: MTLFunctionConstantValues())

      // Get the visible function from the JIT-compiled library.
      let visibleFunction = try! libraryValue.library.makeFunction(
        name: "gemm_body", constantValues: constants)

      // Create pipeline with reverse linking.
      let pipelineDescriptor = MTLComputePipelineDescriptor()
      pipelineDescriptor.computeFunction = kernelFunction

      let linkedFunctions = MTLLinkedFunctions()
      linkedFunctions.privateFunctions = [visibleFunction]
      pipelineDescriptor.linkedFunctions = linkedFunctions

      let pipeline = try! device.makeComputePipelineState(
        descriptor: pipelineDescriptor, options: [], reflection: nil)
      return (libraryValue.kernel, pipeline)
    }

    if kernelDescriptor.preferAsyncStore == nil {
      var candidates: [PipelineValue] = []
      for candidateID in 0..<4 {
        var blockDimensions: (M: UInt16, N: UInt16, K: UInt16)
        var preferAsyncStore: Bool
        switch candidateID {
        case 0:
          blockDimensions = (48, 48, 32)
          preferAsyncStore = false
        case 1:
          blockDimensions = (48, 48, 40)
          preferAsyncStore = false
        case 2:
          blockDimensions = (48, 48, 32)
          preferAsyncStore = true
        case 3:
          blockDimensions = (48, 48, 40)
          preferAsyncStore = true
        default:
          fatalError("This should never happen.")
        }

        // Set the attributes unique to this variant.
        var modifiedKernelDescriptor = kernelDescriptor
        modifiedKernelDescriptor.blockDimensions = blockDimensions
        modifiedKernelDescriptor.preferAsyncStore = preferAsyncStore

        let libraryValue = createLibrary(modifiedKernelDescriptor)
        let pipelineValue = createPipeline(libraryValue)
        candidates.append(pipelineValue)
      }

      // Find the maximum occupancy.
      var maximumOccupancy: Int = -1
      for candidate in candidates {
        let pipeline = candidate.pipeline
        let occupancy = pipeline.maxTotalThreadsPerThreadgroup
        maximumOccupancy = max(maximumOccupancy, occupancy)
      }
      candidates.removeAll(where: {
        $0.pipeline.maxTotalThreadsPerThreadgroup != maximumOccupancy
      })

      // Choose the highest-performing candidate.
      GEMMKernel.pipelineCache[descriptor] = candidates.last!
    } else {
      let libraryValue = createLibrary(kernelDescriptor)
      let pipelineValue = createPipeline(libraryValue)
      GEMMKernel.pipelineCache[descriptor] = pipelineValue
    }
  }
}
