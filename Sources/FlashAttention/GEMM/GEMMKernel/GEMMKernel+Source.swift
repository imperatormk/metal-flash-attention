//
//  GEMMKernel+Source.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/3/24.
//

extension GEMMKernel {
  public func createSource() -> String {
    return """

\(createMetalSimdgroupEvent())
\(createMetalSimdgroupMatrixStorage())
using namespace metal;

\(createConstants())
\(createUtilities())

// Reverse-linking visible function for GEMM.
// Called by the pre-compiled IR kernel shell with incrementing resume_point.
// Async copy is handled by the shell; this function does all compute.
//
// TG layout:
//   [0..127]   = command area (32 x uint32)
//   [128..]    = data area for A, B, C tiles + C_sram save area

// Size of C accumulator save area per thread (2 floats per 8x8 tile).
constant constexpr uint C_SRAM_COUNT = \((registerM / 8) * (registerN / 8));

// Byte offset where C_sram save area starts in TG.
// After the A+B data area used by async copy.
constant constexpr uint C_SAVE_TG_OFFSET = 128 + \(blockBytes("A")) + \(blockBytes("B"));

[[visible]]
void gemm_body(
    threadgroup uchar *tg,
    device uchar *A_raw,
    device uchar *B_raw,
    device uchar *C_raw,
    uint resume_point,
    uint gid_x,
    uint lane_id_u,
    uint sidx_u
) {
  auto cmd = (threadgroup uint*)(tg);
  auto threadgroup_block = tg + 128;  // data area starts at byte 128

  auto A = (device \(memoryName("A"))*)(A_raw);
  auto B = (device \(memoryName("B"))*)(B_raw);
  auto C = (device \(memoryName("C"))*)(C_raw);

  ushort sidx = ushort(sidx_u);
  ushort lane_id = ushort(lane_id_u);
  ushort2 sid(sidx % \(splits.N), sidx / \(splits.N));
  ushort2 morton_offset = morton_order(lane_id);

  // Read gid.y from TG (written by shell before first call).
  uint gid_y = cmd[25];

  uint M_offset = gid_y * M_group;
  uint N_offset = gid_x * N_group;
  bool out_of_bounds = (M_offset + sid.y * \(registerM) >= M ||
                        N_offset + sid.x * \(registerN) >= N);
  ushort2 offset_in_group(sid.x * \(registerN) + morton_offset.x,
                          sid.y * \(registerM) + morton_offset.y);

  if ((M_shift != 0) && (gid_y * M_group >= M_edge)) {
    M_offset -= M_shift;
  }
  if ((N_shift != 0) && (gid_x * N_group >= N_edge)) {
    N_offset -= N_shift;
  }

  \(createVisibleFunctionBody())
}

"""
  }
}

extension GEMMKernel {
  func createConstants() -> String {
    """
    
// Dimensions of each matrix.
// - Limitations to matrix size:
//   - 2^32 in each dimension (M/N/K).
//   - Extending to 2^64 may require changing 'uint' to 'ulong'. There is a
//     good chance this will significantly degrade performance, and require
//     changing the data type of several variables that process addresses. The
//     client is responsible for ensuring correctness and performance with
//     matrices spanning several billion elements in one direction.
//   - The matrix dimensions must be known at compile time, via function
//     constants. Dynamic matrix shapes are beyond the scope of this reference
//     implementation. Dynamic shapes cause a non-negligible regression to
//     shader execution speed. However, they could minimize a compilation
//     latency bottleneck in some use cases.
// - Limitations to batch size:
//   - Dictated by how the client modifies the code to implement batching.
//   - Dynamic batch shapes would likely not harm performance much. For example,
//     someone could enter an array of pointers/memory offsets to different
//     matrices in the batch. Each slice of a 3D thread grid could read a
//     different pointer from memory, and use that pointer as the A/B/C matrix.
//     Another approach is to restrict the input format, so all matrices are
//     stored contiguously in memory. Then, the memory offset could be computed
//     analytically from matrix size and the Z dimension in a 3D thread grid.
//
// Another note:
// - The rows of the matrix must be contiguous in memory. Supporting strides
//   that differ from the actual matrix dimensions should not be difficult, but
//   it is out of scope for this reference kernel.
constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];

// Specify the leading dimensions at PSO creation time.
constant uint A_leading_dimension [[function_constant(5)]];
constant uint B_leading_dimension [[function_constant(6)]];
constant uint C_leading_dimension [[function_constant(7)]];

// Whether to load the previous value of C, and add it to the accumulator.
constant bool load_previous_C [[function_constant(10)]];

// Whether each matrix is transposed.
constant bool A_trans = \(transposeState.A);
constant bool B_trans = \(transposeState.B);

// Define the memory layout of the matrix block.
constant ushort M_group = \(blockDimensions.M);
constant ushort N_group = \(blockDimensions.N);
constant ushort K_group = \(blockDimensions.K);

// Thresholds that mark the matrix edge.
constant uint M_edge = M - (M % M_group);
constant uint N_edge = N - (N % N_group);

// Find the number of elements in the final block. If the matrix
// dimensions are perfectly divisibly by block dimensions, we don't want
// this value to be zero. The final block is a full block.
constant ushort M_remainder = (M % \(registerM) == 0)
  ? \(registerM) : M % \(registerM);
constant ushort N_remainder = (N % \(registerN) == 0)
  ? \(registerN) : N % \(registerN);
constant ushort K_remainder = (K % K_group == 0)
  ? K_group : K % K_group;
constant ushort K_remainder_padded = (K_remainder + 7) / 8 * 8;

// Shift the final block, so it doesn't access out-of-bounds memory.
constant ushort M_shift = (M < M_group) ? 0 : \(registerM) - M_remainder;
constant ushort N_shift = (N < N_group) ? 0 : \(registerN) - N_remainder;

// Number of async K iterations.
constant uint async_iterations_start = \(preferAsyncLoad ? "0" : "(K - (K % K_group))");
constant uint num_async_iterations = (K - async_iterations_start + K_group - 1) / K_group;

"""
  }
}

extension GEMMKernel {
  func createVisibleFunctionBody() -> String {
    let cSramCount = (registerM / 8) * (registerN / 8)
    let paddedCeilingK = "(K + K_remainder_padded - K_remainder)"

    var loadFunctionC: String
    if memoryPrecisions.C == .BF16,
       registerPrecisions.C == .FP32 {
      loadFunctionC = "load_bfloat"
    } else {
      loadFunctionC = "load"
    }

    return """

  // C accumulator in registers.
  simdgroup_matrix_storage<\(registerName("C"))> C_sram[\(cSramCount)];

  // --- RESUME POINT 0: Initialize + device-direct iterations + request first async copy ---
  if (resume_point == 0) {
    // Initialize C (always, even for out_of_bounds — keeps zeros for TG store).
#pragma clang loop unroll(full)
    for (ushort m = 0; m < \(registerM); m += 8) {
#pragma clang loop unroll(full)
      for (ushort n = 0; n < \(registerN); n += 8) {
        ushort2 origin(n, m);
        auto C_tile = get_sram(C_sram, \(registerN), origin);
        *C_tile = simdgroup_matrix_storage<\(registerName("C"))>(0);
      }
    }

    if (!out_of_bounds) {
      if (load_previous_C) {
        uint2 C_offset(N_offset + offset_in_group.x,
                       M_offset + offset_in_group.y);
        auto C_dst = simdgroup_matrix_storage<\(memoryName("C"))>::apply_offset(
          C, \(leadingDimension("C")), C_offset);
#pragma clang loop unroll(full)
        for (ushort m = 0; m < \(registerM); m += 8) {
#pragma clang loop unroll(full)
          for (ushort n = 0; n < \(registerN); n += 8) {
            ushort2 origin(n, m);
            auto C_tile = get_sram(C_sram, \(registerN), origin);
            C_tile->\(loadFunctionC)(C_dst, \(leadingDimension("C")), origin);
          }
        }
      }

      // Perform device-direct multiply iterations (non-async part).
      for (uint k = 0; k < async_iterations_start; k += 8) {
        uint2 A_offset(k, M_offset);
        uint2 B_offset(N_offset, k);
        A_offset += uint2(morton_offset.x, offset_in_group.y);
        B_offset += uint2(offset_in_group.x, morton_offset.y);

        auto A_src = simdgroup_matrix_storage<\(memoryName("A"))>::apply_offset(
          A, \(leadingDimension("A")), A_offset, A_trans);
        auto B_src = simdgroup_matrix_storage<\(memoryName("B"))>::apply_offset(
          B, \(leadingDimension("B")), B_offset, B_trans);

        simdgroup_matrix_storage<\(registerName("A"))> A_sram[\(registerM / 8) * (8 / 8)];
        simdgroup_matrix_storage<\(registerName("B"))> B_sram[(8 / 8) * \(registerN / 8)];
        multiply_accumulate(A_src, B_src, A_sram, B_sram, C_sram, 0);
      }
    }

    if (num_async_iterations == 0) {
      // No async iterations needed — store C via TG-buffered async copy.
      \(createDirectStoreC())
      return;
    }

    // Save C_sram to TG before returning for async copy.
    {
      auto C_save = (threadgroup \(registerName("C"))*)(tg + C_SAVE_TG_OFFSET);
      uint tid = sidx_u * 32 + lane_id_u;
#pragma clang loop unroll(full)
      for (uint i = 0; i < C_SRAM_COUNT; i++) {
        auto elems = C_sram[i].thread_elements();
        C_save[tid * C_SRAM_COUNT * 2 + i * 2 + 0] = (*elems)[0];
        C_save[tid * C_SRAM_COUNT * 2 + i * 2 + 1] = (*elems)[1];
      }
    }

    // Request first A+B async copy.
    // Use block-level offsets (no per-thread morton/offset_in_group) since
    // async copy copies the entire block tile from device to threadgroup.
    {
      uint k = async_iterations_start;
      uint2 A_block_origin(k, M_offset);
      uint2 B_block_origin(N_offset, k);
      auto A_src = simdgroup_matrix_storage<\(memoryName("A"))>::apply_offset(
        A, \(leadingDimension("A")), A_block_origin, A_trans);
      auto B_src = simdgroup_matrix_storage<\(memoryName("B"))>::apply_offset(
        B, \(leadingDimension("B")), B_block_origin, B_trans);

      ushort M_tile_dimension = min(uint(M_group), M - M_offset);
      ushort N_tile_dimension = min(uint(N_group), N - N_offset);
      ushort K_tile_dimension = min(uint(K_group), K - k);
      ushort K_tile_padded = min(uint(K_group), \(paddedCeilingK) - k);

      ushort2 A_tile_src(K_tile_dimension, M_tile_dimension);
      ushort2 B_tile_src(N_tile_dimension, K_tile_dimension);
      // Use full block dims for dst_tile so hardware zero-fills unused positions.
      // This ensures TG rows/columns beyond the actual matrix are zeroed.
      ushort2 A_tile_dst(K_group, M_group);
      ushort2 B_tile_dst(N_group, K_group);

      request_dual_async_copy(
        cmd,
        A_src, A, \(leadingBlockDimensions.A), \(leadingDimension("A")),
        A_tile_src, A_tile_dst, 128, A_trans,
        B_src, B, \(leadingBlockDimensions.B), \(leadingDimension("B")),
        B_tile_src, B_tile_dst, 128 + \(blockBytes("A")), B_trans
      );
    }
    return;
  }

  // --- RESUME POINTS 1..N: Multiply from TG data + request next copy or finish ---
  // resume_point > num_async_iterations means the C store copy just finished.
  // Also handle num_async_iterations == 0 (store-only path from resume_point 0).
  if (resume_point > num_async_iterations) {
    cmd[0] = CMD_DONE;
    return;
  }

  {
    // Restore C_sram from TG.
    auto C_save = (threadgroup \(registerName("C"))*)(tg + C_SAVE_TG_OFFSET);
    uint tid = sidx_u * 32 + lane_id_u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < C_SRAM_COUNT; i++) {
      \(registerName("C")) v0 = C_save[tid * C_SRAM_COUNT * 2 + i * 2 + 0];
      \(registerName("C")) v1 = C_save[tid * C_SRAM_COUNT * 2 + i * 2 + 1];
      *(C_sram[i].thread_elements()) = vec<\(registerName("C")), 2>(v0, v1);
    }

    // Determine which K iteration this is.
    uint async_iter = resume_point;  // 1-based
    uint k_iter = async_iterations_start + (async_iter - 1) * K_group;
    bool is_last_iter = (k_iter + K_group >= K);

    // Multiply-accumulate from TG (data placed by shell's async copy).
    if (!out_of_bounds) {
      auto A_block = (threadgroup \(memoryName("A"))*)(threadgroup_block);
      auto B_block = (threadgroup \(memoryName("B"))*)(threadgroup_block + \(blockBytes("A")));

      ushort2 A_block_offset(morton_offset.x, offset_in_group.y);
      ushort2 B_block_offset(offset_in_group.x, morton_offset.y);
      auto A_block_src = simdgroup_matrix_storage<\(memoryName("A"))>::apply_offset(
        A_block, \(leadingBlockDimensions.A), A_block_offset, A_trans);
      auto B_block_src = simdgroup_matrix_storage<\(memoryName("B"))>::apply_offset(
        B_block, \(leadingBlockDimensions.B), B_block_offset, B_trans);

      simdgroup_matrix_storage<\(registerName("A"))> A_sram[\(registerM / 8) * (K_group / 8)];
      simdgroup_matrix_storage<\(registerName("B"))> B_sram[(K_group / 8) * \(registerN / 8)];

      if (is_last_iter) {
#pragma clang loop unroll(full)
        for (ushort k = 0; k < K_remainder_padded; k += 8) {
          multiply_accumulate(A_block_src, B_block_src,
                              A_sram, B_sram, C_sram, k);
        }
      } else {
#pragma clang loop unroll(full)
        for (ushort k = 0; k < K_group; k += 8) {
          multiply_accumulate(A_block_src, B_block_src,
                              A_sram, B_sram, C_sram, k);
        }
      }
    }

    if (async_iter < num_async_iterations) {
      // More iterations to come. Save C_sram, request next copy.
#pragma clang loop unroll(full)
      for (uint i = 0; i < C_SRAM_COUNT; i++) {
        auto elems = C_sram[i].thread_elements();
        C_save[tid * C_SRAM_COUNT * 2 + i * 2 + 0] = (*elems)[0];
        C_save[tid * C_SRAM_COUNT * 2 + i * 2 + 1] = (*elems)[1];
      }

      // Use block-level offsets for async copy (no per-thread components).
      uint next_k = k_iter + K_group;
      uint2 A_block_origin(next_k, M_offset);
      uint2 B_block_origin(N_offset, next_k);
      auto A_src = simdgroup_matrix_storage<\(memoryName("A"))>::apply_offset(
        A, \(leadingDimension("A")), A_block_origin, A_trans);
      auto B_src = simdgroup_matrix_storage<\(memoryName("B"))>::apply_offset(
        B, \(leadingDimension("B")), B_block_origin, B_trans);

      ushort M_tile_dimension = min(uint(M_group), M - M_offset);
      ushort N_tile_dimension = min(uint(N_group), N - N_offset);
      ushort K_tile_dimension = min(uint(K_group), K - next_k);
      ushort K_tile_padded = min(uint(K_group), \(paddedCeilingK) - next_k);

      ushort2 A_tile_src(K_tile_dimension, M_tile_dimension);
      ushort2 B_tile_src(N_tile_dimension, K_tile_dimension);
      // Full block dst_tile for zero-fill of unused positions.
      ushort2 A_tile_dst(K_group, M_group);
      ushort2 B_tile_dst(N_group, K_group);

      request_dual_async_copy(
        cmd,
        A_src, A, \(leadingBlockDimensions.A), \(leadingDimension("A")),
        A_tile_src, A_tile_dst, 128, A_trans,
        B_src, B, \(leadingBlockDimensions.B), \(leadingDimension("B")),
        B_tile_src, B_tile_dst, 128 + \(blockBytes("A")), B_trans
      );
      return;
    }

    // All iterations done. Store C via TG-buffered async copy.
    \(createDirectStoreC())
    return;
  }

"""
  }

  /// Store C to device memory via threadgroup buffer + async copy.
  /// The TG-buffered path ensures only the valid portion of C is written,
  /// avoiding out-of-bounds writes when the matrix is smaller than the block.
  func createDirectStoreC() -> String {
    var storeFunctionC: String
    if memoryPrecisions.C == .BF16,
       registerPrecisions.C == .FP32 {
      storeFunctionC = "store_bfloat"
    } else {
      storeFunctionC = "store"
    }

    return """
    {
      auto C_block = (threadgroup \(memoryName("C"))*)(threadgroup_block);

      // Only in-bounds threads store their C registers to TG.
      if (!out_of_bounds) {
        auto C_block_dst = simdgroup_matrix_storage<\(memoryName("C"))>::apply_offset(
          C_block, \(leadingBlockDimensions.C), offset_in_group);

#pragma clang loop unroll(full)
        for (ushort m = 0; m < \(registerM); m += 8) {
#pragma clang loop unroll(full)
          for (ushort n = 0; n < \(registerN); n += 8) {
            ushort2 origin(n, m);
            auto C_tile = get_sram(C_sram, \(registerN), origin);
            C_tile->\(storeFunctionC)(
              C_block_dst, \(leadingBlockDimensions.C), origin);
          }
        }
      }

      // Request async copy from TG to device with clamped tile dimensions.
      // Use M_offset/N_offset (already shifted for edge blocks).
      ushort2 C_tile(min(uint(N_group), N - N_offset),
                     min(uint(M_group), M - M_offset));

      uint2 C_dev_offset(N_offset, M_offset);
      auto C_dst = simdgroup_matrix_storage<\(memoryName("C"))>::apply_offset(
        C, \(leadingDimension("C")), C_dev_offset);

      request_single_async_copy(
        cmd,
        CMD_SINGLE_STORE,
        C_dst, C,
        \(leadingBlockDimensions.C), \(leadingDimension("C")),
        C_tile,
        uint(reinterpret_cast<threadgroup uchar*>(C_block) - tg),
        false
      );
    }
"""
  }
}
