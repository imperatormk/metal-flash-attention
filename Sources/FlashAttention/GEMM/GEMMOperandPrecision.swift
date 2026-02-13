//
//  GEMMOperandPrecision.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

/// An enumeration of the precisions supported by the kernel.
///
/// If you wish to support quantized precisions, copy/translate the source code
/// and integrate a modified version into your app. Something similar to a Swift
/// `enum` (e.g. C++ `enum class`) could enumerate the quantization formats
/// used by application code. An exemplary set could be:
/// - FP32
/// - FP16
/// - BF16
/// - signed 8-bit integer
/// - s1ezm7
/// - FP8
/// - palletized
///
/// If you support non-floating-point formats, you have the responsibility of
/// authoring correct and performant GPU code for them. A general rule of thumb,
/// is keep the data compressed in `device` or `threadgroup` memory. Transform
/// into a floating point type while loading into the registers. Keep the
/// accumulator in floating point until the output needs to be written.
/// If the output is quantized, it will be compressed when writing back to
/// `device` memory (or `threadgroup` before the async copy in edge cases).
///
/// For example, the reference implementation treats BF16 like a quantized
/// integer type on Apple7 and Apple8 GPUs. It is decompressed to FP32 in
/// registers.
public enum GEMMOperandPrecision: UInt16 {
  case FP32 = 0
  case FP16 = 1
  case BF16 = 2
  case FP8_E4M3 = 3  // Linear uint8 quantization, max ±448
  case FP8_E5M2 = 4  // Linear uint8 quantization, max ±57344
  case INT8 = 5       // Signed int8 with per-head scale
  case NF4 = 6        // 4-bit NormalFloat, 2 values packed per byte

  // The MSL keyword corresponding to the precision.
  public var name: String {
    switch self {
    case .FP32: return "float"
    case .FP16: return "half"
    case .BF16: return "bfloat"
    case .FP8_E4M3, .FP8_E5M2, .INT8: return "uchar"
    case .NF4: return "uchar"  // packed
    }
  }

  // The size of a scalar, in bytes.
  public var size: Int {
    switch self {
    case .FP32: return 4
    case .FP16: return 2
    case .BF16: return 2
    case .FP8_E4M3, .FP8_E5M2, .INT8: return 1
    case .NF4: return 1  // 1 byte per 2 elements (packed); head dim halved in TG layout
    }
  }

  // Whether this is a quantized integer type (not native float).
  public var isQuantized: Bool {
    switch self {
    case .FP32, .FP16, .BF16: return false
    case .FP8_E4M3, .FP8_E5M2, .INT8, .NF4: return true
    }
  }
}
