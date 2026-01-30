//
//  QuantizationUtilities.swift
//  FlashAttention
//
//  FP8 and INT8 quantization/dequantization utilities for Metal shaders.
//

/// Generates Metal shader code for FP8 E4M3 and E5M2 conversion utilities.
/// These functions are used for on-the-fly dequantization during attention computation.
///
/// NOTE: The quantization uses a LINEAR scheme, not actual FP8 bit representation:
/// - Encode: quantized = (value / scale / fp8_max) * 127 + 128
/// - Decode: value = (quantized - 128) / 127 * fp8_max * scale
public struct QuantizationUtilities {

  /// Metal shader code for FP8 E4M3 dequantization (LINEAR scheme).
  /// Uses linear quantization with zero-point=128, not actual FP8 bit layout.
  /// FP8 E4M3 max representable: 448.0
  public static let fp8E4M3ToFloat = """

  // Dequantize from linear FP8 E4M3 encoding
  // Encoding: quantized = (value / scale / 448.0) * 127 + 128
  // Decoding: value = (quantized - 128) / 127 * 448.0 * scale
  inline half fp8_e4m3_to_half(uchar quant, float scale) {
      const float fp8_max = 448.0f;
      float normalized = (float(quant) - 128.0f) / 127.0f;
      return half(normalized * fp8_max * scale);
  }

  """

  /// Metal shader code for FP8 E5M2 dequantization (LINEAR scheme).
  /// Uses linear quantization with zero-point=128, not actual FP8 bit layout.
  /// FP8 E5M2 max representable: 57344.0
  public static let fp8E5M2ToFloat = """

  // Dequantize from linear FP8 E5M2 encoding
  // Encoding: quantized = (value / scale / 57344.0) * 127 + 128
  // Decoding: value = (quantized - 128) / 127 * 57344.0 * scale
  inline half fp8_e5m2_to_half(uchar quant, float scale) {
      const float fp8_max = 57344.0f;
      float normalized = (float(quant) - 128.0f) / 127.0f;
      return half(normalized * fp8_max * scale);
  }

  """

  /// Metal shader code for INT8 to half conversion with scale.
  public static let int8ToHalf = """

  // Convert INT8 to half with scale
  inline half int8_to_half(char val, float scale) {
      return half(float(val) * scale);
  }

  // Convert uchar (as signed) to half with scale
  inline half uint8_to_half_signed(uchar val, float scale) {
      return half(float(char(val)) * scale);
  }

  """

  /// Metal shader code for NF4 (NormalFloat 4-bit) dequantization.
  /// NF4 uses a 16-value codebook optimized for normally distributed weights.
  public static let nf4ToHalf = """

  // NF4 codebook: 16 values optimized for normal distribution
  constant float NF4_CODEBOOK[16] = {
      -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
      -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
      0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
      0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
  };

  // Decode NF4: 2 values packed per byte
  inline half2 nf4_to_half2(uchar packed, float scale) {
      uint low_idx = packed & 0xF;
      uint high_idx = (packed >> 4) & 0xF;
      return half2(
          half(NF4_CODEBOOK[low_idx] * scale),
          half(NF4_CODEBOOK[high_idx] * scale)
      );
  }

  // Decode single NF4 value (for odd-indexed elements)
  inline half nf4_to_half(uchar packed, bool high, float scale) {
      uint idx = high ? ((packed >> 4) & 0xF) : (packed & 0xF);
      return half(NF4_CODEBOOK[idx] * scale);
  }

  """

  /// Combined Metal shader header with all quantization utilities.
  public static var allUtilities: String {
    return """

    // ============================================================================
    // Quantization Utilities for Flash Attention
    // ============================================================================

    \(fp8E4M3ToFloat)
    \(fp8E5M2ToFloat)
    \(int8ToHalf)
    \(nf4ToHalf)

    """
  }

  /// Returns the dequantization function call for a given precision.
  public static func dequantizeCall(
    precision: GEMMOperandPrecision,
    valueExpr: String,
    scaleExpr: String
  ) -> String {
    switch precision {
    case .FP32, .FP16, .BF16:
      return valueExpr  // No dequantization needed
    case .FP8_E4M3:
      return "fp8_e4m3_to_half(\(valueExpr), \(scaleExpr))"
    case .FP8_E5M2:
      return "fp8_e5m2_to_half(\(valueExpr), \(scaleExpr))"
    case .INT8:
      return "uint8_to_half_signed(\(valueExpr), \(scaleExpr))"
    case .NF4:
      return "nf4_to_half(\(valueExpr), false, \(scaleExpr))"  // Default to low nibble
    }
  }
}
