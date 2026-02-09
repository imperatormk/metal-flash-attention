; ModuleID = 'attention_shell'
; Attention kernel shell: dispatch loop with async copy commands.
; Same architecture as gemm_shell_v2 but with 10 device buffer bindings.
; Visible fn: attention_body(tg, Q,K,V,O,L,D,dO,dV,dK,dQ, resume_point, gid_x, lane_id, sidx)
; cmd[8] = buffer index for single load/store (0=Q,1=K,...,9=dQ)

source_filename = "attention_shell"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%event_t = type opaque

; 32KB = 8192 floats = max threadgroup memory
@tg_buf = internal addrspace(3) global [8192 x float] undef, align 4

declare void @attention_body.MTL_VISIBLE_FN_REF(
    i8 addrspace(3)*,
    i8 addrspace(1)*,
    i8 addrspace(1)*,
    i8 addrspace(1)*,
    i8 addrspace(1)*,
    i8 addrspace(1)*,
    i8 addrspace(1)*,
    i8 addrspace(1)*,
    i8 addrspace(1)*,
    i8 addrspace(1)*,
    i8 addrspace(1)*,
    i32,
    i32,
    i32,
    i32
) local_unnamed_addr section "air.externally_defined"

declare %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(i64, i64, i8 addrspace(3)*, i64, i64, <2 x i64>, i8 addrspace(1)*, i64, i64, <2 x i64>, <2 x i64>, i32) #1
declare %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p1i8.p3i8(i64, i64, i8 addrspace(1)*, i64, i64, <2 x i64>, i8 addrspace(3)*, i64, i64, <2 x i64>, <2 x i64>, i32) #1
declare void @air.wait_simdgroup_events(i32, %event_t addrspace(3)**) #1
declare void @air.wg.barrier(i32, i32) #1
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

define void @attention(
    i8 addrspace(1)* noundef "air-buffer-no-alias" %buf0,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %buf1,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %buf2,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %buf3,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %buf4,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %buf5,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %buf6,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %buf7,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %buf8,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %buf9,
    <3 x i32> noundef %gid,
    i16 noundef %sidx_i16,
    i16 noundef %lane_id_i16
) local_unnamed_addr #0 {
entry:
  %sidx = zext i16 %sidx_i16 to i32
  %lane_id = zext i16 %lane_id_i16 to i32
  %gid_x = extractelement <3 x i32> %gid, i64 0

  %ev = alloca [2 x %event_t addrspace(3)*], align 8
  %ev_i8 = bitcast [2 x %event_t addrspace(3)*]* %ev to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %ev_i8) #3

  %tg_float = getelementptr [8192 x float], [8192 x float] addrspace(3)* @tg_buf, i64 0, i64 0
  %tg_base = bitcast float addrspace(3)* %tg_float to i8 addrspace(3)*
  %tg_cmd = bitcast float addrspace(3)* %tg_float to i32 addrspace(3)*

  br label %dispatch_loop

dispatch_loop:
  %resume_point = phi i32 [0, %entry], [%next_resume, %after_async]

  call void @air.wg.barrier(i32 2, i32 1)

  call void @attention_body.MTL_VISIBLE_FN_REF(
    i8 addrspace(3)* %tg_base,
    i8 addrspace(1)* %buf0,
    i8 addrspace(1)* %buf1,
    i8 addrspace(1)* %buf2,
    i8 addrspace(1)* %buf3,
    i8 addrspace(1)* %buf4,
    i8 addrspace(1)* %buf5,
    i8 addrspace(1)* %buf6,
    i8 addrspace(1)* %buf7,
    i8 addrspace(1)* %buf8,
    i8 addrspace(1)* %buf9,
    i32 %resume_point,
    i32 %gid_x,
    i32 %lane_id,
    i32 %sidx
  ) #4

  call void @air.wg.barrier(i32 2, i32 1)

  %cmd_ptr = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 0
  %cmd = load i32, i32 addrspace(3)* %cmd_ptr

  %is_done = icmp eq i32 %cmd, 0
  br i1 %is_done, label %exit, label %check_cmd1

check_cmd1:
  %is_dual = icmp eq i32 %cmd, 1
  br i1 %is_dual, label %do_dual_copy, label %check_cmd2

check_cmd2:
  %is_load = icmp eq i32 %cmd, 2
  br i1 %is_load, label %do_single_load, label %check_cmd3

check_cmd3:
  %is_store = icmp eq i32 %cmd, 3
  br i1 %is_store, label %do_single_store, label %exit

; ═══════════════════════════════════════════════════════
; DUAL COPY: 2D async copy A+B device→TG (all threads)
; ═══════════════════════════════════════════════════════
do_dual_copy:
  ; ── Read buffer index for A from cmd[8] ──
  %da8 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 8
  %da_buf_idx = load i32, i32 addrspace(3)* %da8
  ; Select A base pointer based on buffer index
  %da_is0 = icmp eq i32 %da_buf_idx, 0
  %da_p0 = select i1 %da_is0, i8 addrspace(1)* %buf0, i8 addrspace(1)* %buf1
  %da_is1 = icmp eq i32 %da_buf_idx, 1
  %da_p1 = select i1 %da_is1, i8 addrspace(1)* %buf1, i8 addrspace(1)* %da_p0
  %da_is2 = icmp eq i32 %da_buf_idx, 2
  %da_p2 = select i1 %da_is2, i8 addrspace(1)* %buf2, i8 addrspace(1)* %da_p1
  %da_is3 = icmp eq i32 %da_buf_idx, 3
  %da_p3 = select i1 %da_is3, i8 addrspace(1)* %buf3, i8 addrspace(1)* %da_p2
  %da_is4 = icmp eq i32 %da_buf_idx, 4
  %da_p4 = select i1 %da_is4, i8 addrspace(1)* %buf4, i8 addrspace(1)* %da_p3
  %da_is5 = icmp eq i32 %da_buf_idx, 5
  %da_p5 = select i1 %da_is5, i8 addrspace(1)* %buf5, i8 addrspace(1)* %da_p4
  %da_is6 = icmp eq i32 %da_buf_idx, 6
  %da_p6 = select i1 %da_is6, i8 addrspace(1)* %buf6, i8 addrspace(1)* %da_p5
  %da_is7 = icmp eq i32 %da_buf_idx, 7
  %da_p7 = select i1 %da_is7, i8 addrspace(1)* %buf7, i8 addrspace(1)* %da_p6
  %da_is8 = icmp eq i32 %da_buf_idx, 8
  %da_p8 = select i1 %da_is8, i8 addrspace(1)* %buf8, i8 addrspace(1)* %da_p7
  %da_is9 = icmp eq i32 %da_buf_idx, 9
  %da_base = select i1 %da_is9, i8 addrspace(1)* %buf9, i8 addrspace(1)* %da_p8

  ; ── A copy params ──
  %a1 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 1
  %a_dst_s32 = load i32, i32 addrspace(3)* %a1
  %a_dst_s = zext i32 %a_dst_s32 to i64

  %a2 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 2
  %a_src_s32 = load i32, i32 addrspace(3)* %a2
  %a_src_s = zext i32 %a_src_s32 to i64

  %a3 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 3
  %a_stw32 = load i32, i32 addrspace(3)* %a3
  %a_stw = zext i32 %a_stw32 to i64
  %a4 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 4
  %a_sth32 = load i32, i32 addrspace(3)* %a4
  %a_sth = zext i32 %a_sth32 to i64

  %a5 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 5
  %a_off_lo = load i32, i32 addrspace(3)* %a5
  %a6 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 6
  %a_off_hi = load i32, i32 addrspace(3)* %a6
  %a_off_lo64 = zext i32 %a_off_lo to i64
  %a_off_hi64 = zext i32 %a_off_hi to i64
  %a_off_hi_s = shl i64 %a_off_hi64, 32
  %a_off = or i64 %a_off_lo64, %a_off_hi_s

  %a7 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 7
  %a_tg_off32 = load i32, i32 addrspace(3)* %a7
  %a_tg_off = zext i32 %a_tg_off32 to i64

  ; dst tile dims
  %a21 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 21
  %a_dtw32 = load i32, i32 addrspace(3)* %a21
  %a_dtw = zext i32 %a_dtw32 to i64
  %a22 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 22
  %a_dth32 = load i32, i32 addrspace(3)* %a22
  %a_dth = zext i32 %a_dth32 to i64

  %a_sv0 = insertelement <2 x i64> zeroinitializer, i64 %a_stw, i32 0
  %a_stile = insertelement <2 x i64> %a_sv0, i64 %a_sth, i32 1
  %a_dv0 = insertelement <2 x i64> zeroinitializer, i64 %a_dtw, i32 0
  %a_dtile = insertelement <2 x i64> %a_dv0, i64 %a_dth, i32 1

  %a_src_p = getelementptr i8, i8 addrspace(1)* %da_base, i64 %a_off
  %a_dst_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %a_tg_off

  %ev0p = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 0
  %a_ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
    i64 1, i64 1,
    i8 addrspace(3)* %a_dst_p, i64 %a_dst_s, i64 1, <2 x i64> %a_dtile,
    i8 addrspace(1)* %a_src_p, i64 %a_src_s, i64 1, <2 x i64> %a_stile,
    <2 x i64> zeroinitializer, i32 0
  )
  store %event_t addrspace(3)* %a_ev, %event_t addrspace(3)** %ev0p

  ; ── Read buffer index for B from cmd[18] ──
  %db18 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 18
  %db_buf_idx = load i32, i32 addrspace(3)* %db18
  ; Select B base pointer
  %db_is0 = icmp eq i32 %db_buf_idx, 0
  %db_p0 = select i1 %db_is0, i8 addrspace(1)* %buf0, i8 addrspace(1)* %buf1
  %db_is1 = icmp eq i32 %db_buf_idx, 1
  %db_p1 = select i1 %db_is1, i8 addrspace(1)* %buf1, i8 addrspace(1)* %db_p0
  %db_is2 = icmp eq i32 %db_buf_idx, 2
  %db_p2 = select i1 %db_is2, i8 addrspace(1)* %buf2, i8 addrspace(1)* %db_p1
  %db_is3 = icmp eq i32 %db_buf_idx, 3
  %db_p3 = select i1 %db_is3, i8 addrspace(1)* %buf3, i8 addrspace(1)* %db_p2
  %db_is4 = icmp eq i32 %db_buf_idx, 4
  %db_p4 = select i1 %db_is4, i8 addrspace(1)* %buf4, i8 addrspace(1)* %db_p3
  %db_is5 = icmp eq i32 %db_buf_idx, 5
  %db_p5 = select i1 %db_is5, i8 addrspace(1)* %buf5, i8 addrspace(1)* %db_p4
  %db_is6 = icmp eq i32 %db_buf_idx, 6
  %db_p6 = select i1 %db_is6, i8 addrspace(1)* %buf6, i8 addrspace(1)* %db_p5
  %db_is7 = icmp eq i32 %db_buf_idx, 7
  %db_p7 = select i1 %db_is7, i8 addrspace(1)* %buf7, i8 addrspace(1)* %db_p6
  %db_is8 = icmp eq i32 %db_buf_idx, 8
  %db_p8 = select i1 %db_is8, i8 addrspace(1)* %buf8, i8 addrspace(1)* %db_p7
  %db_is9 = icmp eq i32 %db_buf_idx, 9
  %db_base = select i1 %db_is9, i8 addrspace(1)* %buf9, i8 addrspace(1)* %db_p8

  ; ── B copy params ──
  %b11 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 11
  %b_dst_s32 = load i32, i32 addrspace(3)* %b11
  %b_dst_s = zext i32 %b_dst_s32 to i64

  %b12 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 12
  %b_src_s32 = load i32, i32 addrspace(3)* %b12
  %b_src_s = zext i32 %b_src_s32 to i64

  %b13 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 13
  %b_stw32 = load i32, i32 addrspace(3)* %b13
  %b_stw = zext i32 %b_stw32 to i64
  %b14 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 14
  %b_sth32 = load i32, i32 addrspace(3)* %b14
  %b_sth = zext i32 %b_sth32 to i64

  %b15 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 15
  %b_off_lo = load i32, i32 addrspace(3)* %b15
  %b16 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 16
  %b_off_hi = load i32, i32 addrspace(3)* %b16
  %b_off_lo64 = zext i32 %b_off_lo to i64
  %b_off_hi64 = zext i32 %b_off_hi to i64
  %b_off_hi_s = shl i64 %b_off_hi64, 32
  %b_off = or i64 %b_off_lo64, %b_off_hi_s

  %b17 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 17
  %b_tg_off32 = load i32, i32 addrspace(3)* %b17
  %b_tg_off = zext i32 %b_tg_off32 to i64

  %b23 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 23
  %b_dtw32 = load i32, i32 addrspace(3)* %b23
  %b_dtw = zext i32 %b_dtw32 to i64
  %b24 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 24
  %b_dth32 = load i32, i32 addrspace(3)* %b24
  %b_dth = zext i32 %b_dth32 to i64

  %b_sv0 = insertelement <2 x i64> zeroinitializer, i64 %b_stw, i32 0
  %b_stile = insertelement <2 x i64> %b_sv0, i64 %b_sth, i32 1
  %b_dv0 = insertelement <2 x i64> zeroinitializer, i64 %b_dtw, i32 0
  %b_dtile = insertelement <2 x i64> %b_dv0, i64 %b_dth, i32 1

  %b_src_p = getelementptr i8, i8 addrspace(1)* %db_base, i64 %b_off
  %b_dst_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %b_tg_off

  %ev1p = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 1
  %b_ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
    i64 1, i64 1,
    i8 addrspace(3)* %b_dst_p, i64 %b_dst_s, i64 1, <2 x i64> %b_dtile,
    i8 addrspace(1)* %b_src_p, i64 %b_src_s, i64 1, <2 x i64> %b_stile,
    <2 x i64> zeroinitializer, i32 0
  )
  store %event_t addrspace(3)* %b_ev, %event_t addrspace(3)** %ev1p

  ; Wait for both
  call void @air.wait_simdgroup_events(i32 2, %event_t addrspace(3)** %ev0p)
  call void @air.wg.barrier(i32 2, i32 1)
  %nr_d = add i32 %resume_point, 1
  br label %after_async

; ═══════════════════════════════════════════════════════
; SINGLE LOAD: 2D async copy device→TG
; cmd[8] = buffer index for the device pointer
; ═══════════════════════════════════════════════════════
do_single_load:
  ; Read buffer index from cmd[8]
  %sl8 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 8
  %sl_buf_idx = load i32, i32 addrspace(3)* %sl8
  ; Select base pointer
  %sl_is0 = icmp eq i32 %sl_buf_idx, 0
  %sl_p0 = select i1 %sl_is0, i8 addrspace(1)* %buf0, i8 addrspace(1)* %buf1
  %sl_is1 = icmp eq i32 %sl_buf_idx, 1
  %sl_p1 = select i1 %sl_is1, i8 addrspace(1)* %buf1, i8 addrspace(1)* %sl_p0
  %sl_is2 = icmp eq i32 %sl_buf_idx, 2
  %sl_p2 = select i1 %sl_is2, i8 addrspace(1)* %buf2, i8 addrspace(1)* %sl_p1
  %sl_is3 = icmp eq i32 %sl_buf_idx, 3
  %sl_p3 = select i1 %sl_is3, i8 addrspace(1)* %buf3, i8 addrspace(1)* %sl_p2
  %sl_is4 = icmp eq i32 %sl_buf_idx, 4
  %sl_p4 = select i1 %sl_is4, i8 addrspace(1)* %buf4, i8 addrspace(1)* %sl_p3
  %sl_is5 = icmp eq i32 %sl_buf_idx, 5
  %sl_p5 = select i1 %sl_is5, i8 addrspace(1)* %buf5, i8 addrspace(1)* %sl_p4
  %sl_is6 = icmp eq i32 %sl_buf_idx, 6
  %sl_p6 = select i1 %sl_is6, i8 addrspace(1)* %buf6, i8 addrspace(1)* %sl_p5
  %sl_is7 = icmp eq i32 %sl_buf_idx, 7
  %sl_p7 = select i1 %sl_is7, i8 addrspace(1)* %buf7, i8 addrspace(1)* %sl_p6
  %sl_is8 = icmp eq i32 %sl_buf_idx, 8
  %sl_p8 = select i1 %sl_is8, i8 addrspace(1)* %buf8, i8 addrspace(1)* %sl_p7
  %sl_is9 = icmp eq i32 %sl_buf_idx, 9
  %sl_base = select i1 %sl_is9, i8 addrspace(1)* %buf9, i8 addrspace(1)* %sl_p8

  %l1 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 1
  %l_dst_s32 = load i32, i32 addrspace(3)* %l1
  %l_dst_s = zext i32 %l_dst_s32 to i64

  %l2 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 2
  %l_src_s32 = load i32, i32 addrspace(3)* %l2
  %l_src_s = zext i32 %l_src_s32 to i64

  %l3 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 3
  %l_tw32 = load i32, i32 addrspace(3)* %l3
  %l_tw = zext i32 %l_tw32 to i64
  %l4 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 4
  %l_th32 = load i32, i32 addrspace(3)* %l4
  %l_th = zext i32 %l_th32 to i64

  %l5 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 5
  %l_off_lo = load i32, i32 addrspace(3)* %l5
  %l6 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 6
  %l_off_hi = load i32, i32 addrspace(3)* %l6
  %l_off_lo64 = zext i32 %l_off_lo to i64
  %l_off_hi64 = zext i32 %l_off_hi to i64
  %l_off_hi_s = shl i64 %l_off_hi64, 32
  %l_off = or i64 %l_off_lo64, %l_off_hi_s

  %l7 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 7
  %l_tg_off32 = load i32, i32 addrspace(3)* %l7
  %l_tg_off = zext i32 %l_tg_off32 to i64

  %l_src_p = getelementptr i8, i8 addrspace(1)* %sl_base, i64 %l_off
  %l_dst_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %l_tg_off

  ; Read dst tile dims from cmd[21]/cmd[22] (if set) or reuse src tile
  %l21 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 21
  %l_dtw32 = load i32, i32 addrspace(3)* %l21
  %l_dtw_raw = zext i32 %l_dtw32 to i64
  ; If dst tile width is 0, use src tile width
  %l_dtw_zero = icmp eq i64 %l_dtw_raw, 0
  %l_dtw = select i1 %l_dtw_zero, i64 %l_tw, i64 %l_dtw_raw

  %l22 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 22
  %l_dth32 = load i32, i32 addrspace(3)* %l22
  %l_dth_raw = zext i32 %l_dth32 to i64
  %l_dth_zero = icmp eq i64 %l_dth_raw, 0
  %l_dth = select i1 %l_dth_zero, i64 %l_th, i64 %l_dth_raw

  %l_sv0 = insertelement <2 x i64> zeroinitializer, i64 %l_tw, i32 0
  %l_stile = insertelement <2 x i64> %l_sv0, i64 %l_th, i32 1
  %l_dv0 = insertelement <2 x i64> zeroinitializer, i64 %l_dtw, i32 0
  %l_dtile = insertelement <2 x i64> %l_dv0, i64 %l_dth, i32 1

  %l_evp = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 0
  %l_ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
    i64 1, i64 1,
    i8 addrspace(3)* %l_dst_p, i64 %l_dst_s, i64 1, <2 x i64> %l_dtile,
    i8 addrspace(1)* %l_src_p, i64 %l_src_s, i64 1, <2 x i64> %l_stile,
    <2 x i64> zeroinitializer, i32 0
  )
  store %event_t addrspace(3)* %l_ev, %event_t addrspace(3)** %l_evp
  call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %l_evp)
  call void @air.wg.barrier(i32 2, i32 1)
  %nr_l = add i32 %resume_point, 1
  br label %after_async

; ═══════════════════════════════════════════════════════
; SINGLE STORE: 2D async copy TG→device
; cmd[8] = buffer index for the device pointer
; ═══════════════════════════════════════════════════════
do_single_store:
  ; Read buffer index from cmd[8]
  %ss8 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 8
  %ss_buf_idx = load i32, i32 addrspace(3)* %ss8
  ; Select base pointer
  %ss_is0 = icmp eq i32 %ss_buf_idx, 0
  %ss_p0 = select i1 %ss_is0, i8 addrspace(1)* %buf0, i8 addrspace(1)* %buf1
  %ss_is1 = icmp eq i32 %ss_buf_idx, 1
  %ss_p1 = select i1 %ss_is1, i8 addrspace(1)* %buf1, i8 addrspace(1)* %ss_p0
  %ss_is2 = icmp eq i32 %ss_buf_idx, 2
  %ss_p2 = select i1 %ss_is2, i8 addrspace(1)* %buf2, i8 addrspace(1)* %ss_p1
  %ss_is3 = icmp eq i32 %ss_buf_idx, 3
  %ss_p3 = select i1 %ss_is3, i8 addrspace(1)* %buf3, i8 addrspace(1)* %ss_p2
  %ss_is4 = icmp eq i32 %ss_buf_idx, 4
  %ss_p4 = select i1 %ss_is4, i8 addrspace(1)* %buf4, i8 addrspace(1)* %ss_p3
  %ss_is5 = icmp eq i32 %ss_buf_idx, 5
  %ss_p5 = select i1 %ss_is5, i8 addrspace(1)* %buf5, i8 addrspace(1)* %ss_p4
  %ss_is6 = icmp eq i32 %ss_buf_idx, 6
  %ss_p6 = select i1 %ss_is6, i8 addrspace(1)* %buf6, i8 addrspace(1)* %ss_p5
  %ss_is7 = icmp eq i32 %ss_buf_idx, 7
  %ss_p7 = select i1 %ss_is7, i8 addrspace(1)* %buf7, i8 addrspace(1)* %ss_p6
  %ss_is8 = icmp eq i32 %ss_buf_idx, 8
  %ss_p8 = select i1 %ss_is8, i8 addrspace(1)* %buf8, i8 addrspace(1)* %ss_p7
  %ss_is9 = icmp eq i32 %ss_buf_idx, 9
  %ss_base = select i1 %ss_is9, i8 addrspace(1)* %buf9, i8 addrspace(1)* %ss_p8

  %s1 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 1
  %s_dst_s32 = load i32, i32 addrspace(3)* %s1
  %s_dst_s = zext i32 %s_dst_s32 to i64

  %s2 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 2
  %s_src_s32 = load i32, i32 addrspace(3)* %s2
  %s_src_s = zext i32 %s_src_s32 to i64

  %s3 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 3
  %s_tw32 = load i32, i32 addrspace(3)* %s3
  %s_tw = zext i32 %s_tw32 to i64
  %s4 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 4
  %s_th32 = load i32, i32 addrspace(3)* %s4
  %s_th = zext i32 %s_th32 to i64

  %s5 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 5
  %s_off_lo = load i32, i32 addrspace(3)* %s5
  %s6 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 6
  %s_off_hi = load i32, i32 addrspace(3)* %s6
  %s_off_lo64 = zext i32 %s_off_lo to i64
  %s_off_hi64 = zext i32 %s_off_hi to i64
  %s_off_hi_s = shl i64 %s_off_hi64, 32
  %s_off = or i64 %s_off_lo64, %s_off_hi_s

  %s7 = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 7
  %s_tg_off32 = load i32, i32 addrspace(3)* %s7
  %s_tg_off = zext i32 %s_tg_off32 to i64

  ; dst = device, src = TG
  %s_dst_p = getelementptr i8, i8 addrspace(1)* %ss_base, i64 %s_off
  %s_src_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %s_tg_off

  %s_v0 = insertelement <2 x i64> zeroinitializer, i64 %s_tw, i32 0
  %s_tile = insertelement <2 x i64> %s_v0, i64 %s_th, i32 1

  %s_evp = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 0
  %s_ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p1i8.p3i8(
    i64 1, i64 1,
    i8 addrspace(1)* %s_dst_p, i64 %s_dst_s, i64 1, <2 x i64> %s_tile,
    i8 addrspace(3)* %s_src_p, i64 %s_src_s, i64 1, <2 x i64> %s_tile,
    <2 x i64> zeroinitializer, i32 0
  )
  store %event_t addrspace(3)* %s_ev, %event_t addrspace(3)** %s_evp
  call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %s_evp)
  call void @air.wg.barrier(i32 2, i32 1)
  %nr_s = add i32 %resume_point, 1
  br label %after_async

after_async:
  %next_resume = phi i32 [%nr_d, %do_dual_copy], [%nr_l, %do_single_load], [%nr_s, %do_single_store]
  br label %dispatch_loop

exit:
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %ev_i8) #3
  ret void
}

attributes #0 = { convergent mustprogress nounwind willreturn "frame-pointer"="none" "min-legal-vector-width"="96" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { argmemonly mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { nounwind }
attributes #4 = { nobuiltin nounwind "no-builtins" }

!air.kernel = !{!0}
!llvm.module.flags = !{!8, !9, !10, !11, !12, !13, !14}
!air.compile_options = !{!15, !16, !17}
!air.visible_function_references = !{!30}
!llvm.ident = !{!19}
!air.version = !{!20}
!air.language_version = !{!21}
!air.source_file_name = !{!22}

; Visible function metadata
!30 = !{!"air.visible_function_reference", void (i8 addrspace(3)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i32, i32, i32, i32)* @attention_body.MTL_VISIBLE_FN_REF, !"attention_body"}

; Kernel metadata: 10 buffers + gid + sidx + lane_id
!0 = !{void (i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, <3 x i32>, i16, i16)* @attention, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7, !23, !24, !25, !26, !27, !28, !29, !31}

; Buffer bindings 0-9
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"buf0"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"buf1"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"buf2"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"buf3"}
!7 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"buf4"}
!23 = !{i32 5, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"buf5"}
!24 = !{i32 6, !"air.buffer", !"air.location_index", i32 6, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"buf6"}
!25 = !{i32 7, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"buf7"}
!26 = !{i32 8, !"air.buffer", !"air.location_index", i32 8, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"buf8"}
!27 = !{i32 9, !"air.buffer", !"air.location_index", i32 9, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"buf9"}

; System values
!28 = !{i32 10, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
!29 = !{i32 11, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"sidx"}
!31 = !{i32 12, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane_id"}

!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"air.max_device_buffers", i32 31}
!10 = !{i32 7, !"air.max_constant_buffers", i32 31}
!11 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!12 = !{i32 7, !"air.max_textures", i32 128}
!13 = !{i32 7, !"air.max_read_write_textures", i32 8}
!14 = !{i32 7, !"air.max_samplers", i32 16}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.850 (metalfe-32023.850)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"attention_shell.ll"}
