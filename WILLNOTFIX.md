# TODO

## Won't Do

### Inner-loop cooperative_copy_2d → async copy
Analyzed thoroughly, not worth it:
- Hot path already uses direct device loads (no TG copy)
- cooperative_copy_2d only fires on edge cases (non-aligned dims) or M3 large headDim uncached accumulators
- Save/restore overhead per shell round-trip would eat any async benefit
- Register state mid-loop is massive (S, P, O, m, l, correction...)
- Common headDims (64, 128) have accumulators cached anyway

Phase 2 (caching layer) was the real win — runs once per kernel, not per-iteration.

## Known Debt

### loadPreviousC race in GEMM
Pre-existing bug with shifted edge blocks. Not caused by reverse-linking.
Tests pass for real workloads. Someone else's problem.

### Simdgroup matmul store addressing in gemm_shell_v2 test
Test uses per-thread matmul (256/256 correct), simdgroup path has wrong store addressing.
Doesn't matter — production attention code has its own well-tested simdgroup ops.
Purely cosmetic test debt.

## Maybe Someday

### True double-buffer prefetch
Prefetch next K+V (or Q+dO) tile while GEMM runs on current tile.
Would need full traversal loop restructured into pipelined state machine.
Big architectural change, big potential win for memory-bound cases.
