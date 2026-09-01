//
//  AsyncCopySupport.swift
//  FlashAttention
//
//  Some GPU generations (M5 / AGXMetalG17X) crash the shader backend on
//  `air.wait_simdgroup_events` ("cannot select: llvm.agx2.syncbuf"), and the
//  async copy never delivers data there. They still report
//  MTLGPUFamily.apple9, so support is discovered by compiling the real kernel:
//  when pipeline creation for an async-copy kernel fails, the process falls
//  back to cooperative copies (see AttentionKernel.compileSingle).
//

import Foundation

extension MTLContext {
  /// Whether attention kernels are generated with simdgroup async copies.
  /// Starts true (or per `MFA_ASYNC_COPY=0|1`) and is cleared for the rest of
  /// the process the first time such a kernel fails to build a pipeline.
  public static var asyncCopyEnabled: Bool = {
    guard let forced = ProcessInfo.processInfo.environment["MFA_ASYNC_COPY"] else {
      return true
    }
    let on = !(forced == "0" || forced.lowercased() == "false")
    FlashAttentionLog.shared.append(
      "async copy: forced \(on ? "on" : "off") via MFA_ASYNC_COPY")
    return on
  }()

  /// True when the setting came from the environment, so a failed compile
  /// should not silently override what the user asked for.
  static let asyncCopyForced: Bool =
    ProcessInfo.processInfo.environment["MFA_ASYNC_COPY"] != nil
}
