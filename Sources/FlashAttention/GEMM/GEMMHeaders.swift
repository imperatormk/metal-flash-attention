//
//  GEMMHeaders.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

/// Create the source code for the reverse-linking async copy protocol.
///
/// Instead of using __asm("air.simdgroup_async_copy...") intrinsics (which
/// are broken on Xcode 26.2+), the visible function writes async copy
/// parameters to a command area in threadgroup memory. The pre-compiled IR
/// kernel shell reads these params and executes the async copy intrinsics.
///
/// TG Command Area Layout (first 128 bytes = 32 x uint32):
///   cmd[0]  = command code: 0=done, 1=dual A+B copy, 2=single dev→TG, 3=single TG→dev
///   cmd[1]  = A dst_elements_per_row
///   cmd[2]  = A src_elements_per_row
///   cmd[3]  = A src_tile_w
///   cmd[4]  = A src_tile_h
///   cmd[5]  = A src_byte_offset_lo
///   cmd[6]  = A src_byte_offset_hi
///   cmd[7]  = A tg_byte_offset
///   cmd[8..10] = reserved
///   cmd[11] = B dst_elements_per_row
///   cmd[12] = B src_elements_per_row
///   cmd[13] = B src_tile_w
///   cmd[14] = B src_tile_h
///   cmd[15] = B src_byte_offset_lo
///   cmd[16] = B src_byte_offset_hi
///   cmd[17] = B tg_byte_offset
///   cmd[18..20] = reserved
///   cmd[21] = A dst_tile_w
///   cmd[22] = A dst_tile_h
///   cmd[23] = B dst_tile_w
///   cmd[24] = B dst_tile_h
///   cmd[25] = gid_y (written by shell)
///   cmd[26] = gid_z (written by shell)
func createReverseLinkingProtocol() -> String {
  return """
// -*- Metal -*-
//===-- reverse_linking_protocol ------------------------------------------===//
// Async copy via pre-compiled IR shell + JIT visible functions.
// The visible function writes copy params to TG cmd area; the shell executes.
//===----------------------------------------------------------------------===//

#ifndef __REVERSE_LINKING_PROTOCOL
#define __REVERSE_LINKING_PROTOCOL

// Command codes for the dispatch loop.
constant constexpr uint CMD_DONE = 0;
constant constexpr uint CMD_DUAL_COPY = 1;
constant constexpr uint CMD_SINGLE_LOAD = 2;
constant constexpr uint CMD_SINGLE_STORE = 3;

// Write dual async copy params (A+B device→TG) to the command area.
// The IR shell will read these and execute 2D async copy.
template <typename T_A, typename T_B>
METAL_FUNC void request_dual_async_copy(
  threadgroup uint* cmd,
  // A params
  const device T_A* A_src,
  const device T_A* A_base,
  ushort A_dst_elements_per_row,
  uint A_src_elements_per_row,
  ushort2 A_tile_src,
  ushort2 A_tile_dst,
  uint A_tg_byte_offset,
  bool A_transpose,
  // B params
  const device T_B* B_src,
  const device T_B* B_base,
  ushort B_dst_elements_per_row,
  uint B_src_elements_per_row,
  ushort2 B_tile_src,
  ushort2 B_tile_dst,
  uint B_tg_byte_offset,
  bool B_transpose
) {
  // Compute byte offsets from buffer base
  ulong A_byte_offset = ulong(reinterpret_cast<const device uchar*>(A_src) -
                               reinterpret_cast<const device uchar*>(A_base));
  ulong B_byte_offset = ulong(reinterpret_cast<const device uchar*>(B_src) -
                               reinterpret_cast<const device uchar*>(B_base));

  // Handle transpose by swapping tile dimensions
  if (A_transpose) {
    A_tile_src = A_tile_src.yx;
    A_tile_dst = A_tile_dst.yx;
  }
  if (B_transpose) {
    B_tile_src = B_tile_src.yx;
    B_tile_dst = B_tile_dst.yx;
  }

  // All strides and tile dims are converted to BYTES for the shell
  // (shell uses element_size=1, alignment=1).
  constexpr uint A_sz = sizeof(T_A);
  constexpr uint B_sz = sizeof(T_B);

  // A copy params (in bytes)
  cmd[1]  = A_dst_elements_per_row * A_sz;
  cmd[2]  = A_src_elements_per_row * A_sz;
  cmd[3]  = A_tile_src.x * A_sz;
  cmd[4]  = A_tile_src.y;
  cmd[5]  = uint(A_byte_offset & 0xFFFFFFFF);
  cmd[6]  = uint(A_byte_offset >> 32);
  cmd[7]  = A_tg_byte_offset;

  // A dst tile (in bytes, height unchanged)
  cmd[21] = A_tile_dst.x * A_sz;
  cmd[22] = A_tile_dst.y;

  // B copy params (in bytes)
  cmd[11] = B_dst_elements_per_row * B_sz;
  cmd[12] = B_src_elements_per_row * B_sz;
  cmd[13] = B_tile_src.x * B_sz;
  cmd[14] = B_tile_src.y;
  cmd[15] = uint(B_byte_offset & 0xFFFFFFFF);
  cmd[16] = uint(B_byte_offset >> 32);
  cmd[17] = B_tg_byte_offset;

  // B dst tile (in bytes, height unchanged)
  cmd[23] = B_tile_dst.x * B_sz;
  cmd[24] = B_tile_dst.y;

  // Issue command
  cmd[0] = CMD_DUAL_COPY;
}

// Write single async copy params (device→TG or TG→device) to the command area.
template <typename T>
METAL_FUNC void request_single_async_copy(
  threadgroup uint* cmd,
  uint command_code,
  const device T* src_or_dst,
  const device T* base,
  ushort tg_elements_per_row,
  uint dev_elements_per_row,
  ushort2 tile_dimensions,
  uint tg_byte_offset,
  bool transpose
) {
  ulong byte_offset = ulong(reinterpret_cast<const device uchar*>(src_or_dst) -
                              reinterpret_cast<const device uchar*>(base));
  if (transpose) {
    tile_dimensions = tile_dimensions.yx;
  }

  // Convert to bytes for shell (element_size=1).
  constexpr uint sz = sizeof(T);
  uint dst_stride = (command_code == CMD_SINGLE_STORE) ? dev_elements_per_row : tg_elements_per_row;
  uint src_stride = (command_code == CMD_SINGLE_STORE) ? tg_elements_per_row : dev_elements_per_row;
  cmd[1]  = dst_stride * sz;
  cmd[2]  = src_stride * sz;
  cmd[3]  = tile_dimensions.x * sz;
  cmd[4]  = tile_dimensions.y;
  cmd[5]  = uint(byte_offset & 0xFFFFFFFF);
  cmd[6]  = uint(byte_offset >> 32);
  cmd[7]  = tg_byte_offset;

  cmd[0] = command_code;
}

// Buffer-indexed variants for attention shell (which has 10 buffers).
// cmd[8] = buffer index for single copy, cmd[8]/cmd[18] = A/B buffer indices for dual copy.

template <typename T>
METAL_FUNC void request_single_async_copy_indexed(
  threadgroup uint* cmd,
  uint command_code,
  const device T* src_or_dst,
  const device T* base,
  ushort tg_elements_per_row,
  uint dev_elements_per_row,
  ushort2 tile_src,
  ushort2 tile_dst,
  uint tg_byte_offset,
  bool transpose,
  uint buffer_index
) {
  ulong byte_offset = ulong(reinterpret_cast<const device uchar*>(src_or_dst) -
                              reinterpret_cast<const device uchar*>(base));
  if (transpose) {
    tile_src = tile_src.yx;
    tile_dst = tile_dst.yx;
  }

  constexpr uint sz = sizeof(T);
  uint dst_stride = (command_code == CMD_SINGLE_STORE) ? dev_elements_per_row : tg_elements_per_row;
  uint src_stride = (command_code == CMD_SINGLE_STORE) ? tg_elements_per_row : dev_elements_per_row;
  cmd[1]  = dst_stride * sz;
  cmd[2]  = src_stride * sz;
  cmd[3]  = tile_src.x * sz;
  cmd[4]  = tile_src.y;
  cmd[5]  = uint(byte_offset & 0xFFFFFFFF);
  cmd[6]  = uint(byte_offset >> 32);
  cmd[7]  = tg_byte_offset;
  cmd[8]  = buffer_index;

  // dst tile (for zero-padding)
  cmd[21] = tile_dst.x * sz;
  cmd[22] = tile_dst.y;

  cmd[0] = command_code;
}

template <typename T_A, typename T_B>
METAL_FUNC void request_dual_async_copy_indexed(
  threadgroup uint* cmd,
  // A params
  const device T_A* A_src,
  const device T_A* A_base,
  ushort A_dst_elements_per_row,
  uint A_src_elements_per_row,
  ushort2 A_tile_src,
  ushort2 A_tile_dst,
  uint A_tg_byte_offset,
  bool A_transpose,
  uint A_buffer_index,
  // B params
  const device T_B* B_src,
  const device T_B* B_base,
  ushort B_dst_elements_per_row,
  uint B_src_elements_per_row,
  ushort2 B_tile_src,
  ushort2 B_tile_dst,
  uint B_tg_byte_offset,
  bool B_transpose,
  uint B_buffer_index
) {
  ulong A_byte_offset = ulong(reinterpret_cast<const device uchar*>(A_src) -
                               reinterpret_cast<const device uchar*>(A_base));
  ulong B_byte_offset = ulong(reinterpret_cast<const device uchar*>(B_src) -
                               reinterpret_cast<const device uchar*>(B_base));

  if (A_transpose) {
    A_tile_src = A_tile_src.yx;
    A_tile_dst = A_tile_dst.yx;
  }
  if (B_transpose) {
    B_tile_src = B_tile_src.yx;
    B_tile_dst = B_tile_dst.yx;
  }

  constexpr uint A_sz = sizeof(T_A);
  constexpr uint B_sz = sizeof(T_B);

  cmd[1]  = A_dst_elements_per_row * A_sz;
  cmd[2]  = A_src_elements_per_row * A_sz;
  cmd[3]  = A_tile_src.x * A_sz;
  cmd[4]  = A_tile_src.y;
  cmd[5]  = uint(A_byte_offset & 0xFFFFFFFF);
  cmd[6]  = uint(A_byte_offset >> 32);
  cmd[7]  = A_tg_byte_offset;
  cmd[8]  = A_buffer_index;

  cmd[21] = A_tile_dst.x * A_sz;
  cmd[22] = A_tile_dst.y;

  cmd[11] = B_dst_elements_per_row * B_sz;
  cmd[12] = B_src_elements_per_row * B_sz;
  cmd[13] = B_tile_src.x * B_sz;
  cmd[14] = B_tile_src.y;
  cmd[15] = uint(B_byte_offset & 0xFFFFFFFF);
  cmd[16] = uint(B_byte_offset >> 32);
  cmd[17] = B_tg_byte_offset;
  cmd[18] = B_buffer_index;

  cmd[23] = B_tile_dst.x * B_sz;
  cmd[24] = B_tile_dst.y;

  cmd[0] = CMD_DUAL_COPY;
}

#endif // __REVERSE_LINKING_PROTOCOL
"""
}

/// Create cooperative threadgroup copy helpers that replace simdgroup_event
/// async copies. All threads in the threadgroup participate in the copy,
/// with automatic zero-padding for dst tiles larger than src tiles.
///
/// This eliminates all __asm("air.simdgroup_async_copy...") usage.
func createCooperativeCopy() -> String {
  return """
// -*- Metal -*-
//===-- cooperative_copy --------------------------------------------------===//
// Threadgroup-cooperative copy replacing simdgroup_event async copies.
// All threads participate; zero-pads dst region beyond src tile.
//===----------------------------------------------------------------------===//

#ifndef __COOPERATIVE_COPY
#define __COOPERATIVE_COPY

// 2D device→threadgroup copy with zero-padding.
// dst_tile may be larger than src_tile; extra elements are zeroed.
template <typename T>
METAL_FUNC void cooperative_copy_2d(
  threadgroup T* dst,
  ushort dst_elements_per_row,
  ushort2 dst_tile,          // (width, height)
  const device T* src,
  uint src_elements_per_row,
  ushort2 src_tile,          // (width, height)
  bool transpose,
  uint tid,                  // thread index in threadgroup
  uint threadgroup_size      // total threads in threadgroup
) {
  if (transpose) {
    src_tile = src_tile.yx;
    dst_tile = dst_tile.yx;
  }

  uint total_dst = uint(dst_tile.x) * uint(dst_tile.y);
  for (uint i = tid; i < total_dst; i += threadgroup_size) {
    ushort row = ushort(i / dst_tile.x);
    ushort col = ushort(i % dst_tile.x);
    T value = T(0);
    if (row < src_tile.y && col < src_tile.x) {
      value = src[uint(row) * src_elements_per_row + col];
    }
    dst[uint(row) * dst_elements_per_row + col] = value;
  }
}

// 2D threadgroup→device copy.
template <typename T>
METAL_FUNC void cooperative_store_2d(
  device T* dst,
  uint dst_elements_per_row,
  ushort2 tile,              // (width, height)
  const threadgroup T* src,
  ushort src_elements_per_row,
  bool transpose,
  uint tid,
  uint threadgroup_size
) {
  if (transpose) {
    tile = tile.yx;
  }

  uint total = uint(tile.x) * uint(tile.y);
  for (uint i = tid; i < total; i += threadgroup_size) {
    ushort row = ushort(i / tile.x);
    ushort col = ushort(i % tile.x);
    dst[uint(row) * dst_elements_per_row + col] =
      src[uint(row) * src_elements_per_row + col];
  }
}

// 1D device→threadgroup copy with zero-padding.
template <typename T>
METAL_FUNC void cooperative_copy_1d(
  threadgroup T* dst,
  ushort dst_count,
  const device T* src,
  ushort src_count,
  uint tid,
  uint threadgroup_size
) {
  for (uint i = tid; i < dst_count; i += threadgroup_size) {
    dst[i] = (i < src_count) ? src[i] : T(0);
  }
}

#endif // __COOPERATIVE_COPY
"""
}

/// Create the source code for the 'metal\_simdgroup\_matrix\_storage' header.
func createMetalSimdgroupMatrixStorage() -> String {
  // How this header spawning code was designed.
  //
  // Find the patterns between the load/store functions:
  // - device has 'uint' elements_per_row
  // - threadgroup has 'ushort' elements_per_row
  // - both have 'ushort2' matrix_origin
  //
  // The origin is 'ushort2' because the 32-bit part of the address should have
  // been applied previously during 'apply_offset'. The 16-bit part should be
  // hard-coded into the assembly when the GEMM loop is unrolled.
  //
  // Transpose path:
  // - load: reads two values; should split each one onto a separate line.
  //   - overwrites the value of *thread_elements() with a new vec<T, 2>
  // - store: the two instructions are on two separate lines.
  //   - fetches from lane 0 or 1 of thread_elements()[0]
  // - adds 0 or 1 to the hard-coded matrix_origin.x
  //
  // Address generation:
  // - casts some intermediate address fragments to 'ulong' for 'device'
  // - keeps all address fragments in 'ushort' for 'threadgroup'
  
  
  
  enum Action {
    case load
    case store
  }
  
  struct MemoryAccessDescriptor {
    var action: Action?
    var addressSpace: MTLAddressSpace?
    var decodingBF16: Bool?
    var indentationSpaceCount: Int = .zero
  }
  
  func createMemoryAccess(
    descriptor: MemoryAccessDescriptor
  ) -> String {
    guard let action = descriptor.action,
          let addressSpace = descriptor.addressSpace,
          let decodingBF16 = descriptor.decodingBF16 else {
      fatalError("Descriptor was incomplete.")
    }
    let indentation = String(
      repeating: " ", count: descriptor.indentationSpaceCount)
    
    // Determine the arguments.
    var arguments: [String] = []
    func addPointerArgument(dataType: String) {
      if action == .load {
        arguments.append("const \(addressSpace.keyword) \(dataType) *src")
      } else {
        arguments.append("\(addressSpace.keyword) \(dataType) *dst")
      }
    }
    if decodingBF16 {
      addPointerArgument(dataType: "bfloat")
    } else {
      addPointerArgument(dataType: "U")
    }
    arguments.append("\(addressSpace.offsetType) elements_per_row")
    arguments.append("ushort2 matrix_origin")
    arguments.append("bool transpose_matrix = false")
    
    // Create the warning comment.
    var output: String = ""
    if decodingBF16 {
      output += "\(indentation)// WARNING: 'T' must be 'float'.\n"
    } else {
      output += "\(indentation)template <typename U>\n"
    }
    
    // Create the function signature.
    output += "\(indentation)METAL_FUNC void"
    if action == .load {
      output += " load"
    } else {
      output += " store"
    }
    if decodingBF16 {
      output += "_bfloat"
    }
    output += "("
    for argumentID in arguments.indices {
      let argument = arguments[argumentID]
      output += argument
      if argumentID < arguments.count - 1 {
        output += ", "
      }
    }
    output += ") {\n"
    
    func createAddress(transposed: Bool, offset: Int) -> String {
      let lineY = "\(addressSpace.offsetType)(matrix_origin.y)"
      var lineX = "matrix_origin.x + \(offset)"
      lineX = "\(addressSpace.offsetType)(\(lineX))"
      
      if transposed {
        return "\(lineX) * elements_per_row + \(lineY)"
      } else {
        return "\(lineY) * elements_per_row + \(lineX)"
      }
    }
    
    func createTwoPartAccess(transposed: Bool) -> [String] {
      // Generate the addresses.
      var lines: [String] = []
      for laneID in 0..<2 {
        lines.append(
          "\(addressSpace.offsetType) address\(laneID) = " +
          createAddress(transposed: transposed, offset: laneID))
      }
      
      if action == .load {
        if decodingBF16 {
          lines.append("bfloat memoryForm0 = src[address0]")
          lines.append("bfloat memoryForm1 = src[address1]")
        } else {
          lines.append("U memoryForm0 = src[address0]")
          lines.append("U memoryForm1 = src[address1]")
        }
      }
      
      if action == .load {
        if decodingBF16 {
          // Separate the loading logic from the decoding logic for clarity.
          lines.append(
            "")
          
          // BF16 decoding logic.
          lines.append(
            "bfloat4 registerForm = *(thread bfloat4*)(thread_elements())")
          lines.append(
            "registerForm[1] = memoryForm0")
          lines.append(
            "registerForm[3] = memoryForm1")
          lines.append(
            "((thread bfloat4*)thread_elements())[0] = registerForm")
        } else {
          // Perform a type cast natively supported by the hardware.
          lines.append("((thread T*)thread_elements())[0] = T(memoryForm0)")
          lines.append("((thread T*)thread_elements())[1] = T(memoryForm1)")
        }
      } else {
        if decodingBF16 {
          // BF16 encoding logic.
          lines.append(
            "bfloat4 registerForm = *(thread bfloat4*)(thread_elements())")
          lines.append(
            "registerForm[2] = registerForm[1]")
        } else {
          // Type casts supported natively by the hardware.
          lines.append("T registerForm0 = ((thread T*)thread_elements())[0]")
          lines.append("T registerForm1 = ((thread T*)thread_elements())[1]")
        }
      }
      
      if action == .store {
        if decodingBF16 {
          lines.append("dst[address0] = registerForm[2]")
          lines.append("dst[address1] = registerForm[3]")
        } else {
          lines.append("dst[address0] = U(registerForm0)")
          lines.append("dst[address1] = U(registerForm1)")
        }
      }
      return lines
    }
    
    func createOnePartAccess() -> [String] {
      var lines: [String] = []
      do {
        let address = createAddress(transposed: false, offset: 0)
        lines.append("auto combinedAddress = \(address)")
      }
      if action == .load {
        if decodingBF16 {
          lines.append(
            "bfloat2 memoryForm = " +
            "*(const \(addressSpace.keyword) packed_bfloat2*)(src + combinedAddress)")
          
          // Separate the loading logic from the decoding logic for clarity.
          lines.append(
            "")
          
          // BF16 decoding logic.
          lines.append(
            "bfloat4 registerForm = *(thread bfloat4*)(thread_elements())")
          lines.append(
            "((thread float*)&registerForm)[1] = *(thread float*)(&memoryForm)")
          lines.append(
            "((thread bfloat*)&registerForm)[1] = memoryForm[0]")
          lines.append(
            "((thread bfloat4*)thread_elements())[0] = registerForm")
        } else {
          lines.append(
            "vec<U, 2> memoryForm = " +
            "*(const \(addressSpace.keyword) vec<U, 2>*)(src + combinedAddress)")
          lines.append(
            "*(thread_elements()) = vec<T, 2>(memoryForm)")
        }
      } else {
        if decodingBF16 {
          // BF16 encoding logic.
          lines.append(
            "bfloat4 registerForm = *(thread bfloat4*)(thread_elements())")
          lines.append(
            "registerForm[2] = registerForm[1]")
          lines.append(
            "float memoryForm = ((thread float*)&registerForm)[1]")
          lines.append(
            "*(\(addressSpace.keyword) float*)(dst + combinedAddress) = " +
            "memoryForm")
        } else {
          lines.append(
            "vec<T, 2> registerForm = *(thread_elements())")
          lines.append(
            "*(\(addressSpace.keyword) vec<U, 2>*)(dst + combinedAddress) = " +
            "vec<U, 2>(registerForm)")
        }
      }
      return lines
    }
    
    func addBlockContents(_ block: [String]) -> [String] {
      block.map {
        if $0.allSatisfy(\.isWhitespace) {
          return "  "
        } else {
          return "  \($0);"
        }
      }
    }
    
    // Determine the lines of the 'if' block.
    var body: [String] = []
    body.append("if (transpose_matrix) {")
    body += addBlockContents(createTwoPartAccess(transposed: true))
    
    // Determine the lines of the 'else' block.
    if decodingBF16 {
      var blockContents: [String]
      if action == .load {
        blockContents = createOnePartAccess()
      } else {
        blockContents = createTwoPartAccess(transposed: false)
      }
      
      body.append("} else {")
      body += addBlockContents(blockContents)
      body.append("}")
    } else {
      body.append("} else if (elements_per_row % 2 != 0) {")
      body += addBlockContents(createTwoPartAccess(transposed: false))
      body.append("} else {")
      body += addBlockContents(createOnePartAccess())
      body.append("}")
    }
    
    // Create the function body.
    for line in body {
      output += "\(indentation)  \(line)\n"
    }
    output += "\(indentation)}\n"
    return output
  }
  
  // Add the first section of the shader.
  var output: String = ""
  output += """
// -*- Metal -*-
//===-- metal_simdgroup_matrix_storage ------------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_MATRIX_STORAGE
#define __METAL_SIMDGROUP_MATRIX_STORAGE

// The layout of threads within a SIMD matrix.
//
//  0  0  1  1  8  8  9  9
//  2  2  3  3 10 10 11 11
//  4  4  5  5 12 12 13 13
//  6  6  7  7 14 14 15 15
// 16 16 17 17 24 24 25 25
// 18 18 19 19 26 26 27 27
// 20 20 21 21 28 28 29 29
// 22 22 23 23 30 30 31 31
//
// This is Morton order, a method for coalescing data accesses. It is used
// in a variety of contexts, from ray tracing acceleration structures, to
// nodal-point Laplacians, to sorting large lattices of atoms.
//
// Source: https://patents.google.com/patent/US11256518B2
METAL_FUNC static ushort2 morton_order(ushort thread_index_in_simdgroup) {
  ushort lane_id = thread_index_in_simdgroup;
  ushort quad_id = lane_id / 4;
  
  constexpr ushort QUADRANT_SPAN_M = 4;
  constexpr ushort THREADS_PER_QUADRANT = 8;
  ushort M_floor_of_quadrant = (quad_id / 4) * QUADRANT_SPAN_M;
  ushort M_in_quadrant = (lane_id / 2) % (THREADS_PER_QUADRANT / 2);
  ushort M_in_simd = M_floor_of_quadrant + M_in_quadrant;
  
  ushort N_floor_of_quadrant = (quad_id & 2) * 2; // 0 or 4
  ushort N_in_quadrant = (lane_id % 2) * 2; // 0 or 2
  ushort N_in_simd = N_floor_of_quadrant + N_in_quadrant;
  
  return ushort2(N_in_simd, M_in_simd);
}

#pragma METAL internals : enable
namespace metal
{
  template <typename T>
  struct simdgroup_matrix_storage {
    typedef vec<T, 64> storage_type;
    
    storage_type t;
    
    METAL_FUNC thread vec<T, 2>* thread_elements() thread {
      return reinterpret_cast<thread vec<T, 2>*>(&t);
    }
    
    METAL_FUNC simdgroup_matrix_storage() thread = default;
    
    METAL_FUNC simdgroup_matrix_storage(vec<T, 2> thread_elements) thread {
      *(this->thread_elements()) = thread_elements;
    }

    METAL_FUNC static device T* apply_offset(device T *src, uint elements_per_row, uint2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        return src + ulong(matrix_origin.x * elements_per_row) + matrix_origin.y;
      } else {
        return src + ulong(matrix_origin.y * elements_per_row) + matrix_origin.x;
      }
    }
    
    METAL_FUNC static threadgroup T* apply_offset(threadgroup T *src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        return src + matrix_origin.x * elements_per_row + matrix_origin.y;
      } else {
        return src + matrix_origin.y * elements_per_row + matrix_origin.x;
      }
    }

"""
  
  var desc = MemoryAccessDescriptor()
  desc.indentationSpaceCount = 4
  
  for action in [Action.load, .store] {
    for addressSpace in [MTLAddressSpace.device, .threadgroup] {
      for decodingBF16 in [false, true] {
        desc.action = action
        desc.addressSpace = addressSpace
        
        desc.decodingBF16 = decodingBF16
        output += createMemoryAccess(descriptor: desc)
        output += "\n"
      }
    }
  }
  
  // Add the last section of the header.
  output += """
    template <typename U, typename V>
    METAL_FUNC void multiply(simdgroup_matrix_storage<U> a, simdgroup_matrix_storage<V> b, bool accumulate = true) {
      if (!accumulate) {
        *(thread_elements()) = vec<T, 2>(0);
      }
      t = __metal_simdgroup_matrix_8x8_multiply_accumulate(a.t, b.t, t, typename simdgroup_matrix_storage<T>::storage_type());
    }
  };
} // namespace metal
#pragma METAL internals : disable

#endif // __METAL_SIMDGROUP_MATRIX_STORAGE

"""
  return output
}

