//
//  AttentionKernel+Caching.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/22/24.
//

// N x D
// parallelization x head

extension AttentionKernel {
  // Enumeration that encapsulates both loading and storing.
  enum CachingOperationType {
    case load
    case store
  }

  func cache(
    operand: AttentionOperand,
    type: CachingOperationType
  ) -> String {
    // MARK: - Operand

    func allocateOperand() -> String {
      if type == .load {
        return """

        simdgroup_matrix_storage<\(registerName(operand))> \
        \(operand)_sram[\(paddedHeadDimension / 8)];

        """
      } else {
        return ""
      }
    }

    func asyncAccessOperand() -> String {
      if type == .load {
        return """

        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
          uint2 \(operand)_offset(d_outer, \(parallelizationGroupOffset));
          auto src = simdgroup_matrix_storage<\(memoryName(operand))>
          ::apply_offset(
            \(operand), \(leadingDimension(operand)),
            \(operand)_offset, \(transposed(operand)));
          auto dst = (threadgroup \(memoryName(operand))*)(threadgroup_block);

          ushort D_src_dimension = min(
            ushort(\(blockDimensions.head)),
            ushort(\(headDimension) - d_outer));
          ushort D_dst_dimension = min(
            ushort(\(blockDimensions.head)),
            ushort(\(paddedHeadDimension) - d_outer));
          ushort R_dimension = min(
            uint(\(blockDimensions.parallelization)),
            uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
          ushort2 tile_src(D_src_dimension, R_dimension);
          ushort2 tile_dst(D_dst_dimension, R_dimension);

          cooperative_copy_2d(
            dst, \(leadingBlockDimension(operand)), tile_dst,
            src, \(leadingDimension(operand)), tile_src,
            \(transposed(operand)), tid, tg_size);
        }

        """
      } else {
        return """

        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
          uint2 \(operand)_offset(d_outer, \(parallelizationGroupOffset));
          auto src = (threadgroup \(memoryName(operand))*)(threadgroup_block);
          auto dst = simdgroup_matrix_storage<\(memoryName(operand))>
          ::apply_offset(
            \(operand), \(leadingDimension(operand)),
            \(operand)_offset, \(transposed(operand)));

          ushort D_dimension = min(
            ushort(\(blockDimensions.head)),
            ushort(\(headDimension) - d_outer));
          ushort R_dimension = min(
            uint(\(blockDimensions.parallelization)),
            uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
          ushort2 tile(D_dimension, R_dimension);

          cooperative_store_2d(
            dst, \(leadingDimension(operand)), tile,
            src, \(leadingBlockDimension(operand)),
            \(transposed(operand)), tid, tg_size);
        }

        """
      }
    }

    func leadingDimensionOperand(
      _ descriptor: LoopIterationDescriptor
    ) -> String {
      if descriptor.addressSpace == .device {
        return leadingDimension(operand)
      } else {
        return "\(leadingBlockDimension(operand))"
      }
    }

    func declareOperandLocation(
      descriptor: LoopIterationDescriptor
    ) -> String {
      if descriptor.addressSpace == .device {
        return """

        uint2 \(operand)_src_offset(
          morton_offset.x + d_outer,
          \(clampedParallelizationThreadOffset));
        auto \(operand)_src = simdgroup_matrix_storage<\(memoryName(operand))>
        ::apply_offset(
          \(operand), \(leadingDimension(operand)),
          \(operand)_src_offset, \(transposed(operand)));

        """
      } else {
        return """

        ushort2 \(operand)_block_offset(
          morton_offset.x,
          morton_offset.y + sidx * 8);
        auto \(operand)_src =
        (threadgroup \(memoryName(operand))*)(threadgroup_block);

        \(operand)_src = simdgroup_matrix_storage<\(memoryName(operand))>
        ::apply_offset(
          \(operand)_src, \(leadingBlockDimension(operand)),
          \(operand)_block_offset, \(transposed(operand)));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        """
      }
    }

    // MARK: - Inner Loop

    func innerLoopHead(
      headStart: UInt16,
      headEnd: UInt16,
      descriptor: LoopIterationDescriptor
    ) -> String {
      if type == .load {
        return """

        #pragma clang loop unroll(full)
        for (ushort d = \(headStart); d < \(headEnd); d += 8) {
          ushort2 \(operand)_origin(d, 0);
          \(operand)_sram[(d_outer + d) / 8].\(loadFunction(operand))(
            \(operand)_src, \(leadingDimensionOperand(descriptor)),
            \(operand)_origin, \(transposed(operand)));
        }

        """
      } else {
        return """

        #pragma clang loop unroll(full)
        for (ushort d = \(headStart); d < \(headEnd); d += 8) {
          ushort2 \(operand)_origin(d, 0);
          \(operand)_sram[(d_outer + d) / 8].\(storeFunction(operand))(
            \(operand)_src, \(leadingDimensionOperand(descriptor)),
            \(operand)_origin, \(transposed(operand)));
        }

        """
      }
    }

    // MARK: - Outer Loop

    struct LoopIterationDescriptor {
      var addressSpace: MTLAddressSpace = .threadgroup
    }

    func loopIteration(
      descriptor: LoopIterationDescriptor
    ) -> String {
      func loadOperand() -> String {
        if type == .load {
          return asyncAccessOperand()
        } else {
          return ""
        }
      }

      func storeOperand() -> String {
        if type == .load {
          return ""
        } else {
          return asyncAccessOperand()
        }
      }

      if descriptor.addressSpace == .device {
        return """

        \(declareOperandLocation(descriptor: descriptor))
        if (
          \(type == .load) ||
          (\(unsafeParallelizationThreadOffset) < \(parallelizationDimension))
        ) {
        \(innerLoopHead(
            headStart: 0,
            headEnd: blockDimensions.head,
            descriptor: descriptor))
        }

        """
      } else {
        return """

        \(loadOperand())
        \(declareOperandLocation(descriptor: descriptor))
        if (d_outer + \(blockDimensions.head) <= \(headDimension)) {
          \(innerLoopHead(
              headStart: 0,
              headEnd: blockDimensions.head,
              descriptor: descriptor))
        } else {
          \(innerLoopHead(
              headStart: 0,
              headEnd: headDimension % blockDimensions.head,
              descriptor: descriptor))
        }
        \(storeOperand())

        """
      }
    }

    func gatedLoopIteration() -> String {
      var descriptorDevice = LoopIterationDescriptor()
      var descriptorThreadgroup = LoopIterationDescriptor()
      descriptorDevice.addressSpace = .device
      descriptorThreadgroup.addressSpace = .threadgroup

      let condition = """
      \(!preferAsyncCache) && (
        (\(headDimension) % \(blockDimensions.head) == 0) ||
        (d_outer + \(blockDimensions.head) <= \(headDimension))
      )
      """

      return """

      if (\(condition)) {
        \(loopIteration(descriptor: descriptorDevice))
      } else {
        \(loopIteration(descriptor: descriptorThreadgroup))
      }

      """
    }

    return """

    \(allocateOperand())

    #pragma clang loop unroll(full)
    for (
      ushort d_outer = 0;
      d_outer < \(headDimension);
      d_outer += \(blockDimensions.head)
    ) {
      \(gatedLoopIteration())
    }

    """
  }
}

// MARK: - Setup and Cleanup (non-async path, still used when no operands are cached)

extension AttentionKernel {
  // Prepare the addresses and registers for the attention loop.
  func createSetup() -> String {
    // Allocate registers for the specified operand.
    func allocate(operand: AttentionOperand) -> String {
      """

      simdgroup_matrix_storage<\(registerName(operand))> \
      \(operand)_sram[\(paddedHeadDimension / 8)];

      """
    }

    // Initialize the output string.
    var output: String = ""

    switch type {
    case .forward:
      if cached(.Q) {
        output += cache(operand: .Q, type: .load)
      }
      if cached(.O) {
        output += allocate(operand: .O)
      }
      output += """

      float m = -numeric_limits<float>::max();
      float l = numeric_limits<float>::denorm_min();

      """

    case .backwardQuery:
      if cached(.Q){
        output += cache(operand: .Q, type: .load)
      }
      if cached(.dO) {
        output += cache(operand: .dO, type: .load)
      }
      if cached(.dQ) {
        output += allocate(operand: .dQ)
      }

      guard let memoryPrecisionL = memoryPrecisions[.L],
            memoryPrecisionL != .BF16 else {
        fatalError("Invalid memory precision for L.")
      }

      // L is always either FP16 or FP32, so we don't need custom type
      // conversion code here.
      output += """

      float L_sram = L[\(clampedParallelizationThreadOffset)];
      \(computeD())

      """

    case .backwardKeyValue:
      if cached(.K) {
        output += cache(operand: .K, type: .load)
      }
      if cached(.V) {
        output += cache(operand: .V, type: .load)
      }
      if cached(.dK) {
        output += allocate(operand: .dK)
      }
      if cached(.dV) {
        output += allocate(operand: .dV)
      }
    }

    return output
  }

  // Store any cached outputs to memory.
  func createCleanup(type: AttentionKernelType) -> String {
    // Initialize the output string.
    var output: String = ""

    switch type {
    case .forward:
      if cached(.O) {
        output += cache(operand: .O, type: .store)
      }

      // L is always either FP16 or FP32, so we don't need custom type
      // conversion code here.
      output += """

      if (\(unsafeParallelizationThreadOffset) < \(parallelizationDimension)) {
        // Premultiplied by log_base_2(e).
        float L_sram = m + fast::log2(l);
        L[\(clampedParallelizationThreadOffset)] = L_sram;
      }

      """

    case .backwardQuery:
      if cached(.dQ) {
        output += cache(operand: .dQ, type: .store)
      }

      // Cast D from FP32 to potentially BF16.
      func storeD() -> String {
        switch memoryPrecisions[.D] {
        case .FP32:
          return """

          D[\(clampedParallelizationThreadOffset)] = D_sram;

          """
        case .BF16:
          return """

          bfloat2 registerForm = *(thread bfloat2*)(&D_sram);
          bfloat memoryForm = registerForm[1];
          D[\(clampedParallelizationThreadOffset)] = memoryForm;

          """
        default:
          fatalError("Invalid memory precision for D.")
        }
      }
      output += """

      if (\(unsafeParallelizationThreadOffset) < \(parallelizationDimension)) {
        \(storeD())
      }

      """

    case .backwardKeyValue:
      if cached(.dK) {
        output += cache(operand: .dK, type: .store)
      }
      if cached(.dV) {
        output += cache(operand: .dV, type: .store)
      }
    }

    return output
  }
}

// MARK: - Async Caching State Machine

extension AttentionKernel {
  /// Generate the save area offset for a given operand.
  /// Each operand gets a contiguous region in the TG save area.
  private func saveAreaOffset(
    for operand: AttentionOperand,
    in operandList: [AttentionOperand]
  ) -> UInt16 {
    var offset = saveTGOffset()
    for op in operandList {
      if op == operand { break }
      offset += saveAreaSize(operand: op)
    }
    return offset
  }

  /// Generate code to save an operand's registers to TG save area.
  /// Each thread saves its own vec<T,2> elements.
  func createSaveRegisters(operand: AttentionOperand, saveOffset: UInt16) -> String {
    let precision = registerPrecisions[operand]!
    let regType = precision.name
    let elemCount = paddedHeadDimension / 8
    // Each thread saves elemCount * 2 values of type regType.
    // threadSaveOffset = saveOffset + (sidx * 32 + lane_id) * elemCount * 2 * sizeof(regType)
    let elemsPerThread = elemCount * 2

    return """

    // Save \(operand)_sram to TG save area.
    {
      auto save_ptr = (threadgroup \(regType)*)(tg + \(saveOffset));
      uint thread_offset = (uint(sidx) * 32 + uint(lane_id)) * \(elemsPerThread);
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < \(elemCount); i++) {
        auto elems = *(\(operand)_sram[i].thread_elements());
        save_ptr[thread_offset + i * 2 + 0] = elems[0];
        save_ptr[thread_offset + i * 2 + 1] = elems[1];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    """
  }

  /// Generate code to restore an operand's registers from TG save area.
  func createRestoreRegisters(operand: AttentionOperand, saveOffset: UInt16) -> String {
    let precision = registerPrecisions[operand]!
    let regType = precision.name
    let elemCount = paddedHeadDimension / 8
    let elemsPerThread = elemCount * 2

    return """

    // Restore \(operand)_sram from TG save area.
    {
      auto save_ptr = (threadgroup \(regType)*)(tg + \(saveOffset));
      uint thread_offset = (uint(sidx) * 32 + uint(lane_id)) * \(elemsPerThread);
      #pragma clang loop unroll(full)
      for (ushort i = 0; i < \(elemCount); i++) {
        vec<\(regType), 2> elems;
        elems[0] = save_ptr[thread_offset + i * 2 + 0];
        elems[1] = save_ptr[thread_offset + i * 2 + 1];
        *(\(operand)_sram[i].thread_elements()) = elems;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    """
  }

  /// Generate code to save scalar values (m, l, L_sram, D_sram) to TG save area.
  /// These go after all operand save areas.
  private func scalarSaveOffset() -> UInt16 {
    var offset = saveTGOffset()
    // All load operands + all store operands contribute to save area
    let allOps = cachedLoadOperands() + cachedStoreOperands()
    for op in allOps {
      offset += saveAreaSize(operand: op)
    }
    return offset
  }

  /// Request an async copy for one chunk of one operand (load: device→TG).
  private func requestAsyncLoad(
    operand: AttentionOperand,
    chunkIndex: Int
  ) -> String {
    let dOuter = UInt16(chunkIndex) * blockDimensions.head
    let bufferIndex = operand.bufferBinding!

    return """

    // Request async load: \(operand) chunk \(chunkIndex) (d_outer=\(dOuter))
    if (sidx == 0 && lane_id == 0) {
      ushort d_outer_val = \(dOuter);
      uint2 offset(d_outer_val, \(parallelizationGroupOffset));
      auto src = simdgroup_matrix_storage<\(memoryName(operand))>
      ::apply_offset(
        \(operand), \(leadingDimension(operand)),
        offset, \(transposed(operand)));

      ushort D_src_dimension = min(
        ushort(\(blockDimensions.head)),
        ushort(\(headDimension) - d_outer_val));
      ushort D_dst_dimension = min(
        ushort(\(blockDimensions.head)),
        ushort(\(paddedHeadDimension) - d_outer_val));
      ushort R_dimension = min(
        uint(\(blockDimensions.parallelization)),
        uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));

      request_single_async_copy_indexed(
        cmd,
        CMD_SINGLE_LOAD,
        src,
        (const device \(memoryName(operand))*)(buf\(bufferIndex)),
        \(leadingBlockDimension(operand)),
        uint(\(leadingDimension(operand))),
        ushort2(D_src_dimension, R_dimension),
        ushort2(D_dst_dimension, R_dimension),
        uint(128),  // tg_byte_offset = data area
        \(transposed(operand)),
        uint(\(bufferIndex)));
    }

    """
  }

  /// Read one chunk from TG data area into operand sram array.
  private func readChunkFromTG(
    operand: AttentionOperand,
    chunkIndex: Int
  ) -> String {
    let dOuter = UInt16(chunkIndex) * blockDimensions.head

    return """

    // Read \(operand) chunk \(chunkIndex) from TG data area into sram.
    {
      ushort2 block_offset(
        morton_offset.x,
        morton_offset.y + sidx * 8);
      auto src = (threadgroup \(memoryName(operand))*)(threadgroup_block);
      src = simdgroup_matrix_storage<\(memoryName(operand))>
      ::apply_offset(
        src, \(leadingBlockDimension(operand)),
        block_offset, \(transposed(operand)));
      threadgroup_barrier(mem_flags::mem_threadgroup);

      ushort head_chunk = min(
        ushort(\(blockDimensions.head)),
        ushort(\(paddedHeadDimension) - \(dOuter)));
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < head_chunk; d += 8) {
        ushort2 origin(d, 0);
        \(operand)_sram[(\(dOuter) + d) / 8].\(loadFunction(operand))(
          src, \(leadingBlockDimension(operand)),
          origin, \(transposed(operand)));
      }
    }

    """
  }

  /// Write one chunk from operand sram to TG data area (for store path).
  private func writeChunkToTG(
    operand: AttentionOperand,
    chunkIndex: Int
  ) -> String {
    let dOuter = UInt16(chunkIndex) * blockDimensions.head

    return """

    // Write \(operand) chunk \(chunkIndex) from sram to TG data area.
    {
      ushort2 block_offset(
        morton_offset.x,
        morton_offset.y + sidx * 8);
      auto dst = (threadgroup \(memoryName(operand))*)(threadgroup_block);
      dst = simdgroup_matrix_storage<\(memoryName(operand))>
      ::apply_offset(
        dst, \(leadingBlockDimension(operand)),
        block_offset, \(transposed(operand)));

      ushort head_chunk = min(
        ushort(\(blockDimensions.head)),
        ushort(\(headDimension) - \(dOuter)));
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < head_chunk; d += 8) {
        ushort2 origin(d, 0);
        \(operand)_sram[(\(dOuter) + d) / 8].\(storeFunction(operand))(
          dst, \(leadingBlockDimension(operand)),
          origin, \(transposed(operand)));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    """
  }

  /// Request async store for one chunk of one operand (TG→device).
  private func requestAsyncStore(
    operand: AttentionOperand,
    chunkIndex: Int
  ) -> String {
    let dOuter = UInt16(chunkIndex) * blockDimensions.head
    let bufferIndex = operand.bufferBinding!

    return """

    // Request async store: \(operand) chunk \(chunkIndex) (d_outer=\(dOuter))
    if (sidx == 0 && lane_id == 0) {
      ushort d_outer_val = \(dOuter);
      uint2 offset(d_outer_val, \(parallelizationGroupOffset));
      auto dst = simdgroup_matrix_storage<\(memoryName(operand))>
      ::apply_offset(
        \(operand), \(leadingDimension(operand)),
        offset, \(transposed(operand)));

      ushort D_dimension = min(
        ushort(\(blockDimensions.head)),
        ushort(\(headDimension) - d_outer_val));
      ushort R_dimension = min(
        uint(\(blockDimensions.parallelization)),
        uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));

      request_single_async_copy_indexed(
        cmd,
        CMD_SINGLE_STORE,
        dst,
        (const device \(memoryName(operand))*)(buf\(bufferIndex)),
        \(leadingBlockDimension(operand)),
        uint(\(leadingDimension(operand))),
        ushort2(D_dimension, R_dimension),
        ushort2(D_dimension, R_dimension),
        uint(128),  // tg_byte_offset = data area
        \(transposed(operand)),
        uint(\(bufferIndex)));
    }

    """
  }

  // MARK: - State Machine Phase Generators

  /// Generate the load phase dispatch.
  /// Resume points are indexed linearly across all load operands × chunks.
  func createAsyncCachingLoadDispatch(chunks: Int) -> String {
    let loadOps = cachedLoadOperands()
    guard !loadOps.isEmpty else { return "cmd[0] = CMD_DONE;\n" }

    // Compute all load operand + store operand list for save offsets
    let allSaveOps = cachedLoadOperands() + cachedStoreOperands()

    var output = ""

    // Allocate sram arrays for all cached load operands.
    for op in loadOps {
      output += """
      simdgroup_matrix_storage<\(registerName(op))> \
      \(op)_sram[\(paddedHeadDimension / 8)];

      """
    }

    // Build a flat list of (operand, chunkIndex) pairs.
    // rp 0: request first chunk (no read)
    // rp 1: read chunk 0, request chunk 1
    // rp N-1: read chunk N-2, request chunk N-1 (last chunk read by loop point)
    var cases: [(op: AttentionOperand, chunkIndex: Int, rpIndex: Int)] = []
    var rpIndex = 0
    for op in loadOps {
      for c in 0..<chunks {
        cases.append((op: op, chunkIndex: c, rpIndex: rpIndex))
        rpIndex += 1
      }
    }

    // Generate switch statement.
    output += "switch (resume_point) {\n"
    for (i, entry) in cases.enumerated() {
      let op = entry.op
      let c = entry.chunkIndex
      let saveOff = saveAreaOffset(for: op, in: allSaveOps)

      output += "case \(i): {\n"

      if i == 0 {
        // First resume point: just request the first load.
        output += requestAsyncLoad(operand: op, chunkIndex: c)
        output += "break;\n}\n"
      } else {
        // Determine what the previous resume point loaded.
        let prevEntry = cases[i - 1]
        let prevOp = prevEntry.op
        let prevChunk = prevEntry.chunkIndex

        // If we crossed an operand boundary, the previous operand's
        // save includes all chunks. Restore it from save area.
        // Otherwise, restore the same operand (which has chunks 0..prevChunk-1).

        // Restore all previously-saved load operands.
        if i > 1 {
          // We need to restore the current operand's partial state.
          // If it's a new operand (prevOp != op), restore the previous operand
          // fully and the new one isn't started yet (no restore needed).
          // If same operand, restore it.
          if prevOp == op {
            output += createRestoreRegisters(operand: op, saveOffset: saveOff)
          }
        }

        // Read the chunk that was async-loaded into TG data area.
        output += readChunkFromTG(operand: prevOp, chunkIndex: prevChunk)

        // Save all current register state.
        // If prevOp == op: we're building up the same operand.
        // If prevOp != op: prevOp is now fully loaded, save it.
        //   And op hasn't started yet (nothing to save for it).
        if prevOp == op {
          output += createSaveRegisters(operand: op, saveOffset: saveOff)
        } else {
          let prevSaveOff = saveAreaOffset(for: prevOp, in: allSaveOps)
          output += createSaveRegisters(operand: prevOp, saveOffset: prevSaveOff)
        }

        // Request next load.
        output += requestAsyncLoad(operand: op, chunkIndex: c)
        output += "break;\n}\n"
      }
    }
    output += "default: break;\n}\n"

    return output
  }

  /// Generate the loop phase.
  /// Reads the last load chunk, runs the full traversal loop, starts stores.
  func createLoopPhase(
    createLoop: () -> String,
    chunks: Int
  ) -> String {
    let loadOps = cachedLoadOperands()
    let storeOps = cachedStoreOperands()
    let allSaveOps = cachedLoadOperands() + cachedStoreOperands()

    var output = ""

    // Allocate sram arrays for all cached load operands.
    for op in loadOps {
      output += """
      simdgroup_matrix_storage<\(registerName(op))> \
      \(op)_sram[\(paddedHeadDimension / 8)];

      """
    }

    // Restore all load operands from save area (except the last chunk).
    if cachingLoadResumePoints() > 1 {
      for op in loadOps {
        let saveOff = saveAreaOffset(for: op, in: allSaveOps)
        output += createRestoreRegisters(operand: op, saveOffset: saveOff)
      }
    }

    // Read the last load chunk from TG data area.
    if !loadOps.isEmpty {
      let lastOp = loadOps.last!
      let lastChunk = chunks - 1
      output += readChunkFromTG(operand: lastOp, chunkIndex: lastChunk)
    }

    // Now all cached load operands are fully in sram.
    // Run the setup code that doesn't involve caching (allocate output, init scalars).
    output += createSetupNonCaching()

    // Run the full traversal loop.
    output += createLoop()

    // Now handle stores.
    if !storeOps.isEmpty {
      // Save all store operands to TG save area.
      for op in storeOps {
        let saveOff = saveAreaOffset(for: op, in: allSaveOps)
        output += createSaveRegisters(operand: op, saveOffset: saveOff)
      }

      // Also save scalar state if needed.
      output += createSaveScalars()

      // Write first store chunk to TG data area and request async store.
      let firstStoreOp = storeOps[0]
      output += writeChunkToTG(operand: firstStoreOp, chunkIndex: 0)
      output += requestAsyncStore(operand: firstStoreOp, chunkIndex: 0)
    } else {
      // No stores — do scalar cleanup and done.
      output += createScalarCleanup()
      output += "cmd[0] = CMD_DONE;\n"
    }

    return output
  }

  /// Generate the store phase dispatch.
  func createAsyncCachingStoreDispatch(chunks: Int, loopPoint: Int) -> String {
    let storeOps = cachedStoreOperands()
    guard !storeOps.isEmpty else { return "cmd[0] = CMD_DONE;\n" }

    let allSaveOps = cachedLoadOperands() + cachedStoreOperands()

    var output = ""

    // Allocate sram arrays for all cached store operands.
    for op in storeOps {
      output += """
      simdgroup_matrix_storage<\(registerName(op))> \
      \(op)_sram[\(paddedHeadDimension / 8)];

      """
    }

    // Build flat list: (operand, chunkIndex) pairs for stores.
    // The first store (op[0], chunk 0) was already done at loop_point.
    // So store resume points start at (op[0], chunk 1) or (op[1], chunk 0).
    var cases: [(op: AttentionOperand, chunkIndex: Int)] = []
    for (opIdx, op) in storeOps.enumerated() {
      for c in 0..<chunks {
        if opIdx == 0 && c == 0 { continue } // already done at loop point
        cases.append((op: op, chunkIndex: c))
      }
    }

    // Restore store operands from save area.
    output += "{\n"
    for op in storeOps {
      let saveOff = saveAreaOffset(for: op, in: allSaveOps)
      output += createRestoreRegisters(operand: op, saveOffset: saveOff)
    }

    output += "uint store_rp = resume_point - \(loopPoint + 1);\n"
    output += "switch (store_rp) {\n"

    for (i, entry) in cases.enumerated() {
      let op = entry.op
      let c = entry.chunkIndex

      output += "case \(i): {\n"
      output += writeChunkToTG(operand: op, chunkIndex: c)
      output += requestAsyncStore(operand: op, chunkIndex: c)
      output += "break;\n}\n"
    }
    output += "default: break;\n}\n"
    output += "}\n"

    return output
  }

  /// Generate code for the non-caching parts of setup (allocating output
  /// accumulators, initializing scalars, computing D, etc.).
  func createSetupNonCaching() -> String {
    func allocate(operand: AttentionOperand) -> String {
      """

      simdgroup_matrix_storage<\(registerName(operand))> \
      \(operand)_sram[\(paddedHeadDimension / 8)];

      """
    }

    var output = ""

    switch type {
    case .forward:
      if cached(.O) {
        output += allocate(operand: .O)
      }
      output += """

      float m = -numeric_limits<float>::max();
      float l = numeric_limits<float>::denorm_min();

      """

    case .backwardQuery:
      if cached(.dQ) {
        output += allocate(operand: .dQ)
      }

      guard let memoryPrecisionL = memoryPrecisions[.L],
            memoryPrecisionL != .BF16 else {
        fatalError("Invalid memory precision for L.")
      }

      output += """

      float L_sram = L[\(clampedParallelizationThreadOffset)];
      \(computeD())

      """

    case .backwardKeyValue:
      if cached(.dK) {
        output += allocate(operand: .dK)
      }
      if cached(.dV) {
        output += allocate(operand: .dV)
      }
    }

    return output
  }

  /// Generate code to save scalar values (m, l for forward; L_sram, D_sram for backward).
  func createSaveScalars() -> String {
    let scalarOff = scalarSaveOffset()

    switch type {
    case .forward:
      // Save m and l per thread.
      return """

      // Save scalars m, l to TG save area.
      {
        auto scalar_ptr = (threadgroup float*)(tg + \(scalarOff));
        uint scalar_offset = (uint(sidx) * 32 + uint(lane_id)) * 2;
        scalar_ptr[scalar_offset + 0] = m;
        scalar_ptr[scalar_offset + 1] = l;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      """

    case .backwardQuery:
      // D_sram needs to be saved for scalar cleanup.
      return """

      // Save scalar D_sram to TG save area.
      {
        auto scalar_ptr = (threadgroup float*)(tg + \(scalarOff));
        uint scalar_offset = uint(sidx) * 32 + uint(lane_id);
        scalar_ptr[scalar_offset] = D_sram;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      """

    case .backwardKeyValue:
      return ""
    }
  }

  /// Generate code to restore scalar values.
  func createRestoreScalars() -> String {
    let scalarOff = scalarSaveOffset()

    switch type {
    case .forward:
      return """

      // Restore scalars m, l from TG save area.
      float m, l;
      {
        auto scalar_ptr = (threadgroup float*)(tg + \(scalarOff));
        uint scalar_offset = (uint(sidx) * 32 + uint(lane_id)) * 2;
        m = scalar_ptr[scalar_offset + 0];
        l = scalar_ptr[scalar_offset + 1];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      """

    case .backwardQuery:
      return """

      // Restore scalar D_sram from TG save area.
      float D_sram;
      {
        auto scalar_ptr = (threadgroup float*)(tg + \(scalarOff));
        uint scalar_offset = uint(sidx) * 32 + uint(lane_id);
        D_sram = scalar_ptr[scalar_offset];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      """

    case .backwardKeyValue:
      return ""
    }
  }

  /// Generate the scalar cleanup (store L, D, etc.) and CMD_DONE.
  func createScalarCleanup() -> String {
    let hasStores = !cachedStoreOperands().isEmpty
    var output = ""

    if hasStores {
      output += createRestoreScalars()
    }

    switch type {
    case .forward:
      output += """

      if (\(unsafeParallelizationThreadOffset) < \(parallelizationDimension)) {
        // Premultiplied by log_base_2(e).
        float L_sram = m + fast::log2(l);
        L[\(clampedParallelizationThreadOffset)] = L_sram;
      }

      """

    case .backwardQuery:
      func storeD() -> String {
        switch memoryPrecisions[.D] {
        case .FP32:
          return """

          D[\(clampedParallelizationThreadOffset)] = D_sram;

          """
        case .BF16:
          return """

          bfloat2 registerForm = *(thread bfloat2*)(&D_sram);
          bfloat memoryForm = registerForm[1];
          D[\(clampedParallelizationThreadOffset)] = memoryForm;

          """
        default:
          fatalError("Invalid memory precision for D.")
        }
      }
      output += """

      if (\(unsafeParallelizationThreadOffset) < \(parallelizationDimension)) {
        \(storeD())
      }

      """

    case .backwardKeyValue:
      // No scalar cleanup needed.
      break
    }

    return output
  }
}
