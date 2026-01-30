//
//  GEMMOperandPrecision.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/21/24.
//

/// An enumeration of the precisions supported by the kernel.
///
/// The kernel supports both standard floating-point formats (FP32, FP16, BF16)
/// and quantized formats (FP8, INT8, NF4) for memory-efficient inference.
///
/// Quantized formats keep data compressed in `device` or `threadgroup` memory.
/// They are decompressed to a floating-point type when loading into registers.
/// The accumulator stays in floating point until written back to memory.
///
/// For example, the reference implementation treats BF16 like a quantized
/// integer type on Apple7 and Apple8 GPUs. It is decompressed to FP32 in
/// registers.
public enum GEMMOperandPrecision: UInt16 {
  case FP32 = 0
  case FP16 = 1
  case BF16 = 2

  // Quantized formats for memory-efficient inference
  case FP8_E4M3 = 3  // 8-bit: 1 sign, 4 exponent, 3 mantissa. Range: ±448
  case FP8_E5M2 = 4  // 8-bit: 1 sign, 5 exponent, 2 mantissa. Range: ±57344
  case INT8 = 5      // 8-bit signed integer with per-row/per-head scale
  case NF4 = 6       // 4-bit NormalFloat with 16-value codebook

  // The MSL keyword corresponding to the precision.
  public var name: String {
    switch self {
    case .FP32:
      return "float"
    case .FP16:
      return "half"
    case .BF16:
      return "bfloat"
    case .FP8_E4M3, .FP8_E5M2, .INT8:
      return "uchar"  // Stored as unsigned char in memory
    case .NF4:
      return "uchar"  // 2 values packed per byte
    }
  }

  // The MSL keyword for the register type after dequantization.
  public var registerName: String {
    switch self {
    case .FP32:
      return "float"
    case .FP16:
      return "half"
    case .BF16:
      return "bfloat"
    case .FP8_E4M3, .FP8_E5M2, .INT8, .NF4:
      return "half"  // Dequantize to half for computation
    }
  }

  // The size of a scalar, in bytes.
  public var size: Int {
    switch self {
    case .FP32:
      return 4
    case .FP16:
      return 2
    case .BF16:
      return 2
    case .FP8_E4M3, .FP8_E5M2, .INT8:
      return 1
    case .NF4:
      return 1  // Note: 2 values per byte, but we report 1 for indexing
    }
  }

  // Whether this precision requires a scale buffer for dequantization.
  public var requiresScale: Bool {
    switch self {
    case .FP32, .FP16, .BF16:
      return false
    case .FP8_E4M3, .FP8_E5M2, .INT8, .NF4:
      return true
    }
  }

  // Whether this is a quantized (non-native) format.
  public var isQuantized: Bool {
    switch self {
    case .FP32, .FP16, .BF16:
      return false
    case .FP8_E4M3, .FP8_E5M2, .INT8, .NF4:
      return true
    }
  }
}
