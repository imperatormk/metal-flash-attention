//
//  AttentionKernel+CooperativeCopy.swift
//  FlashAttention
//
//  Software replacement for the simdgroup async copy used when the device's
//  shader backend cannot compile `air.wait_simdgroup_events` (see
//  MTLContext.supportsSimdgroupAsyncCopy). Every thread in the threadgroup
//  strides over the destination byte tile; positions outside the source
//  tile are zero-filled, matching async_copy_2d clamp mode 0.
//

extension AttentionKernel {

  /// Same tile geometry as generateAsyncCopyDeviceToTG, copied by all threads.
  /// Ends after a threadgroup barrier.
  func generateCooperativeCopyDeviceToTG(
    prefix p: String,
    buffer: String,
    operand: AttentionOperand,
    dOuter: String,
    seqOffset: String,
    seqDim: String,
    blockSeq: UInt16,
    blockHead: UInt16,
    D: UInt32,
    leadingDim: UInt32,
    leadingBlockDim: UInt32,
    memPrec: GEMMOperandPrecision,
    transposed: Bool,
    tgOffset: String = "0"
  ) -> String {
    let elemSize = UInt32(memPrec.size)
    let isNF4 = (memPrec == .NF4)
    var ir = ""

    ir += "  ; Cooperative copy \(operand) device→TG (\(p))\n"

    let dOuterPacked: String
    let DPacked: UInt32
    let blockHeadPacked: UInt16
    if isNF4 && !transposed {
      ir += "  %\(p)dOuter_packed = lshr i32 \(dOuter), 1\n"
      dOuterPacked = "%\(p)dOuter_packed"
      DPacked = D / 2
      blockHeadPacked = blockHead / 2
    } else {
      dOuterPacked = dOuter
      DPacked = D
      blockHeadPacked = blockHead
    }

    if transposed {
      ir += "  %\(p)src_row = mul i32 \(dOuterPacked), \(leadingDim)\n"
      ir += "  %\(p)src_off32 = add i32 %\(p)src_row, \(seqOffset)\n"
    } else {
      ir += "  %\(p)src_row = mul i32 \(seqOffset), \(leadingDim)\n"
      ir += "  %\(p)src_off32 = add i32 %\(p)src_row, \(dOuterPacked)\n"
    }
    ir += "  %\(p)src_off = zext i32 %\(p)src_off32 to i64\n"
    ir += "  %\(p)src_byte = mul i64 %\(p)src_off, \(elemSize)\n"
    ir += "  %\(p)src_p = getelementptr i8, i8 addrspace(1)* \(buffer), i64 %\(p)src_byte\n"

    ir += "  %\(p)d_rem_32 = sub i32 \(DPacked), \(dOuterPacked)\n"
    ir += "  %\(p)d_cmp = icmp ult i32 %\(p)d_rem_32, \(blockHeadPacked)\n"
    ir += "  %\(p)d_src = select i1 %\(p)d_cmp, i32 %\(p)d_rem_32, i32 \(blockHeadPacked)\n"
    ir += "  %\(p)seq_oob = icmp uge i32 \(seqOffset), \(seqDim)\n"
    ir += "  %\(p)seq_rem_raw = sub i32 \(seqDim), \(seqOffset)\n"
    ir += "  %\(p)seq_rem_32 = select i1 %\(p)seq_oob, i32 0, i32 %\(p)seq_rem_raw\n"
    ir += "  %\(p)seq_cmp = icmp ult i32 %\(p)seq_rem_32, \(blockSeq)\n"
    ir += "  %\(p)seq_src = select i1 %\(p)seq_cmp, i32 %\(p)seq_rem_32, i32 \(blockSeq)\n"

    let dstStride = leadingBlockDim * elemSize
    let srcStride = leadingDim * elemSize

    // Source tile in (bytes wide, rows); destination tile is constant.
    let dstW: UInt32
    let dstH: UInt32
    let srcH: String
    if transposed {
      ir += "  %\(p)srcW = mul i32 %\(p)seq_src, \(elemSize)\n"
      srcH = "%\(p)d_src"
      dstW = UInt32(blockSeq) * elemSize
      dstH = UInt32(blockHeadPacked)
    } else {
      ir += "  %\(p)srcW = mul i32 %\(p)d_src, \(elemSize)\n"
      srcH = "%\(p)seq_src"
      dstW = UInt32(blockHeadPacked) * elemSize
      dstH = UInt32(blockSeq)
    }

    ir += "  %\(p)dst_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 \(tgOffset)\n"

    // Copy unit: one element (source offsets are element-aligned).
    let unit = max(elemSize, 1)
    let ty = unit == 4 ? "i32" : (unit == 2 ? "i16" : "i8")
    let cols = dstW / unit
    let total = cols * dstH
    let nthreads = UInt32(threadgroupSize)

    ir += "  %\(p)lid_s = mul i32 %sidx, 32\n"
    ir += "  %\(p)lid = add i32 %\(p)lid_s, %lane_id\n"
    ir += "  br label %\(p)cc_pre\n\n"
    ir += "\(p)cc_pre:\n"
    ir += "  br label %\(p)cc_head\n\n"

    ir += "\(p)cc_head:\n"
    ir += "  %\(p)i = phi i32 [ %\(p)lid, %\(p)cc_pre ], [ %\(p)i_next, %\(p)cc_latch ]\n"
    ir += "  %\(p)in = icmp ult i32 %\(p)i, \(total)\n"
    ir += "  br i1 %\(p)in, label %\(p)cc_body, label %\(p)cc_done\n\n"

    ir += "\(p)cc_body:\n"
    ir += "  %\(p)row = udiv i32 %\(p)i, \(cols)\n"
    ir += "  %\(p)col = urem i32 %\(p)i, \(cols)\n"
    ir += "  %\(p)colb = mul i32 %\(p)col, \(unit)\n"
    ir += "  %\(p)row_ok = icmp ult i32 %\(p)row, \(srcH)\n"
    ir += "  %\(p)col_ok = icmp ult i32 %\(p)colb, %\(p)srcW\n"
    ir += "  %\(p)ok = and i1 %\(p)row_ok, %\(p)col_ok\n"
    ir += "  br i1 %\(p)ok, label %\(p)cc_load, label %\(p)cc_store\n\n"

    ir += "\(p)cc_load:\n"
    ir += "  %\(p)soff_r = mul i32 %\(p)row, \(srcStride)\n"
    ir += "  %\(p)soff = add i32 %\(p)soff_r, %\(p)colb\n"
    ir += "  %\(p)soff64 = zext i32 %\(p)soff to i64\n"
    ir += "  %\(p)sp8 = getelementptr i8, i8 addrspace(1)* %\(p)src_p, i64 %\(p)soff64\n"
    ir += "  %\(p)sp = bitcast i8 addrspace(1)* %\(p)sp8 to \(ty) addrspace(1)*\n"
    ir += "  %\(p)ld = load \(ty), \(ty) addrspace(1)* %\(p)sp, align \(unit)\n"
    ir += "  br label %\(p)cc_store\n\n"

    ir += "\(p)cc_store:\n"
    ir += "  %\(p)val = phi \(ty) [ %\(p)ld, %\(p)cc_load ], [ 0, %\(p)cc_body ]\n"
    ir += "  %\(p)doff_r = mul i32 %\(p)row, \(dstStride)\n"
    ir += "  %\(p)doff = add i32 %\(p)doff_r, %\(p)colb\n"
    ir += "  %\(p)doff64 = zext i32 %\(p)doff to i64\n"
    ir += "  %\(p)dp8 = getelementptr i8, i8 addrspace(3)* %\(p)dst_p, i64 %\(p)doff64\n"
    ir += "  %\(p)dp = bitcast i8 addrspace(3)* %\(p)dp8 to \(ty) addrspace(3)*\n"
    ir += "  store \(ty) %\(p)val, \(ty) addrspace(3)* %\(p)dp, align \(unit)\n"
    ir += "  br label %\(p)cc_latch\n\n"

    ir += "\(p)cc_latch:\n"
    ir += "  %\(p)i_next = add i32 %\(p)i, \(nthreads)\n"
    ir += "  br label %\(p)cc_head\n\n"

    ir += "\(p)cc_done:\n"
    ir += "  call void @air.wg.barrier(i32 2, i32 1)\n\n"

    return ir
  }

  /// Cooperative replacement for the TG → device async store: copies an exact
  /// `tileH` rows × `tileWBytes` tile from `srcPtr` (threadgroup) to `dstPtr`
  /// (device). `tileWBytes` / `tileH` are i32 SSA names or literals. Does not
  /// emit a barrier; callers already synchronize before and after the store.
  func generateCooperativeStoreTGToDevice(
    prefix p: String,
    dstPtr: String,
    srcPtr: String,
    tileWBytes: String,
    tileH: String,
    dstStride: UInt32,
    srcStride: UInt32,
    unit: UInt32
  ) -> String {
    let ty = unit == 4 ? "i32" : (unit == 2 ? "i16" : "i8")
    let nthreads = UInt32(threadgroupSize)
    var ir = ""

    ir += "  ; Cooperative store TG→device (\(p))\n"
    ir += "  %\(p)cols = udiv i32 \(tileWBytes), \(unit)\n"
    ir += "  %\(p)total = mul i32 %\(p)cols, \(tileH)\n"
    ir += "  %\(p)lid_s = mul i32 %sidx, 32\n"
    ir += "  %\(p)lid = add i32 %\(p)lid_s, %lane_id\n"
    ir += "  br label %\(p)cs_pre\n\n"
    ir += "\(p)cs_pre:\n"
    ir += "  br label %\(p)cs_head\n\n"

    ir += "\(p)cs_head:\n"
    ir += "  %\(p)i = phi i32 [ %\(p)lid, %\(p)cs_pre ], [ %\(p)i_next, %\(p)cs_body ]\n"
    ir += "  %\(p)in = icmp ult i32 %\(p)i, %\(p)total\n"
    ir += "  br i1 %\(p)in, label %\(p)cs_body, label %\(p)cs_done\n\n"

    ir += "\(p)cs_body:\n"
    ir += "  %\(p)row = udiv i32 %\(p)i, %\(p)cols\n"
    ir += "  %\(p)col = urem i32 %\(p)i, %\(p)cols\n"
    ir += "  %\(p)colb = mul i32 %\(p)col, \(unit)\n"
    ir += "  %\(p)soff_r = mul i32 %\(p)row, \(srcStride)\n"
    ir += "  %\(p)soff = add i32 %\(p)soff_r, %\(p)colb\n"
    ir += "  %\(p)soff64 = zext i32 %\(p)soff to i64\n"
    ir += "  %\(p)sp8 = getelementptr i8, i8 addrspace(3)* \(srcPtr), i64 %\(p)soff64\n"
    ir += "  %\(p)sp = bitcast i8 addrspace(3)* %\(p)sp8 to \(ty) addrspace(3)*\n"
    ir += "  %\(p)val = load \(ty), \(ty) addrspace(3)* %\(p)sp, align \(unit)\n"
    ir += "  %\(p)doff_r = mul i32 %\(p)row, \(dstStride)\n"
    ir += "  %\(p)doff = add i32 %\(p)doff_r, %\(p)colb\n"
    ir += "  %\(p)doff64 = zext i32 %\(p)doff to i64\n"
    ir += "  %\(p)dp8 = getelementptr i8, i8 addrspace(1)* \(dstPtr), i64 %\(p)doff64\n"
    ir += "  %\(p)dp = bitcast i8 addrspace(1)* %\(p)dp8 to \(ty) addrspace(1)*\n"
    ir += "  store \(ty) %\(p)val, \(ty) addrspace(1)* %\(p)dp, align \(unit)\n"
    ir += "  %\(p)i_next = add i32 %\(p)i, \(nthreads)\n"
    ir += "  br label %\(p)cs_head\n\n"

    ir += "\(p)cs_done:\n"
    return ir
  }
}
