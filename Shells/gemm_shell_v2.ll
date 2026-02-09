; ModuleID = 'gemm_shell_v2'
; Simplified GEMM shell: No sidx gate on async copy, element_size=1 (byte-level).
; Single visible fn with resume_point protocol.

source_filename = "gemm_shell_v2"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%event_t = type opaque

; 32KB = 8192 floats = max threadgroup memory
@tg_buf = internal addrspace(3) global [8192 x float] undef, align 4

declare void @gemm_body.MTL_VISIBLE_FN_REF(
    i8 addrspace(3)*,
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

define void @gemm(
    i8 addrspace(1)* noundef "air-buffer-no-alias" %A,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %B,
    i8 addrspace(1)* noundef "air-buffer-no-alias" %C,
    <3 x i32> noundef %gid,
    i16 noundef %sidx_i16,
    i16 noundef %lane_id_i16
) local_unnamed_addr #0 {
entry:
  %sidx = zext i16 %sidx_i16 to i32
  %lane_id = zext i16 %lane_id_i16 to i32
  %gid_x = extractelement <3 x i32> %gid, i64 0
  %gid_y = extractelement <3 x i32> %gid, i64 1
  %gid_z = extractelement <3 x i32> %gid, i64 2

  %ev = alloca [2 x %event_t addrspace(3)*], align 8
  %ev_i8 = bitcast [2 x %event_t addrspace(3)*]* %ev to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %ev_i8) #3

  %tg_float = getelementptr [8192 x float], [8192 x float] addrspace(3)* @tg_buf, i64 0, i64 0
  %tg_base = bitcast float addrspace(3)* %tg_float to i8 addrspace(3)*
  %tg_cmd = bitcast float addrspace(3)* %tg_float to i32 addrspace(3)*

  ; Write gid.y, gid.z to TG
  %gid_y_ptr = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 25
  store i32 %gid_y, i32 addrspace(3)* %gid_y_ptr
  %gid_z_ptr = getelementptr i32, i32 addrspace(3)* %tg_cmd, i64 26
  store i32 %gid_z, i32 addrspace(3)* %gid_z_ptr

  br label %dispatch_loop

dispatch_loop:
  %resume_point = phi i32 [0, %entry], [%next_resume, %after_async]

  call void @air.wg.barrier(i32 2, i32 1)

  call void @gemm_body.MTL_VISIBLE_FN_REF(
    i8 addrspace(3)* %tg_base,
    i8 addrspace(1)* %A,
    i8 addrspace(1)* %B,
    i8 addrspace(1)* %C,
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
; DUAL COPY: No sidx gate — all threads participate
; Element size hardcoded to 4 (float)
; ═══════════════════════════════════════════════════════
do_dual_copy:
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

  %a_src_p = getelementptr i8, i8 addrspace(1)* %A, i64 %a_off
  %a_dst_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %a_tg_off

  %ev0p = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 0
  %a_ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
    i64 1, i64 1,
    i8 addrspace(3)* %a_dst_p, i64 %a_dst_s, i64 1, <2 x i64> %a_dtile,
    i8 addrspace(1)* %a_src_p, i64 %a_src_s, i64 1, <2 x i64> %a_stile,
    <2 x i64> zeroinitializer, i32 0
  )
  store %event_t addrspace(3)* %a_ev, %event_t addrspace(3)** %ev0p

  ; ── B copy ──
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

  %b_src_p = getelementptr i8, i8 addrspace(1)* %B, i64 %b_off
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
; SINGLE LOAD: 2D async copy device→TG (no sidx gate)
; ═══════════════════════════════════════════════════════
do_single_load:
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

  %l_src_p = getelementptr i8, i8 addrspace(1)* %C, i64 %l_off
  %l_dst_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %l_tg_off

  %l_v0 = insertelement <2 x i64> zeroinitializer, i64 %l_tw, i32 0
  %l_tile = insertelement <2 x i64> %l_v0, i64 %l_th, i32 1

  %l_evp = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 0
  %l_ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
    i64 1, i64 1,
    i8 addrspace(3)* %l_dst_p, i64 %l_dst_s, i64 1, <2 x i64> %l_tile,
    i8 addrspace(1)* %l_src_p, i64 %l_src_s, i64 1, <2 x i64> %l_tile,
    <2 x i64> zeroinitializer, i32 0
  )
  store %event_t addrspace(3)* %l_ev, %event_t addrspace(3)** %l_evp
  call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %l_evp)
  call void @air.wg.barrier(i32 2, i32 1)
  %nr_l = add i32 %resume_point, 1
  br label %after_async

; ═══════════════════════════════════════════════════════
; SINGLE STORE: 2D async copy TG→device (no sidx gate)
; ═══════════════════════════════════════════════════════
do_single_store:
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

  ; dst = device (C + offset), src = TG
  %s_dst_p = getelementptr i8, i8 addrspace(1)* %C, i64 %s_off
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
!air.visible_function_references = !{!18}
!llvm.ident = !{!19}
!air.version = !{!20}
!air.language_version = !{!21}
!air.source_file_name = !{!22}

!0 = !{void (i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, <3 x i32>, i16, i16)* @gemm, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !30, !6, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"A"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"B"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"C"}
!30 = !{i32 3, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
!6 = !{i32 4, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"sidx"}
!7 = !{i32 5, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane_id"}

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

!18 = !{!"air.visible_function_reference", void (i8 addrspace(3)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i32, i32, i32, i32)* @gemm_body.MTL_VISIBLE_FN_REF, !"gemm_body"}

!19 = !{!"Apple metal version 32023.850 (metalfe-32023.850)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"gemm_shell_v2.ll"}
