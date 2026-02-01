//
//  AttentionOperand.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/3/24.
//

/// The memory allocations used in attention kernels.
public enum AttentionOperand: Hashable, Equatable, CustomStringConvertible {
  case Q
  case K
  case S
  case P
  case V
  case O
  
  case L
  case D
  
  case dO
  case dV
  case dP
  case dS
  case dK
  case dQ

  // External attention mask (boolean, true = attend, false = mask out)
  case mask

  // Additive attention bias (float, added to attention scores before softmax)
  // Shape: [batch, num_heads, seq_q, seq_k] or broadcastable
  case attnBias

  // Second additive mask (float, for window masks etc.)
  // Shape: [nW, seq_q, seq_k] or similar - encoder handles offsets
  case attnMask2

  /// The name in the shader source.
  ///
  /// Since the `AttentionOperand` type conforms to `CustomStringConvertible`,
  /// the name can be injected through string interpolation.
  public var description: String {
    switch self {
    case .Q: return "Q"
    case .K: return "K"
    case .S: return "S"
    case .P: return "P"
    case .V: return "V"
    case .O: return "O"
      
    case .L: return "L"
    case .D: return "D"
      
    case .dO: return "dO"
    case .dV: return "dV"
    case .dP: return "dP"
    case .dS: return "dS"
    case .dK: return "dK"
    case .dQ: return "dQ"
    case .mask: return "mask"
    case .attnBias: return "attn_bias"
    case .attnMask2: return "attn_mask2"
    }
  }
  
  public var bufferBinding: UInt8? {
    switch self {
    case .Q: return 0
    case .K: return 1
    case .S: return nil
    case .P: return nil
    case .V: return 2
    case .O: return 3
      
    case .L: return 4
    case .D: return 5
      
    case .dO: return 6
    case .dV: return 7
    case .dP: return nil
    case .dS: return nil
    case .dK: return 8
    case .dQ: return 9
    case .mask: return 10
    case .attnBias: return 11
    case .attnMask2: return 12
    }
  }
}
