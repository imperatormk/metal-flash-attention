# Shell IR Sources

LLVM IR for the pre-compiled Metal kernel shells. These contain the `__asm` async copy intrinsics that can't be JIT-compiled on macOS 15+.

The shells are compiled to `.metallib` and bundled in `Sources/FlashAttention/Resources/`. JIT-compiled visible functions (pure Metal, no `__asm`) are linked at runtime via `MTLLinkedFunctions.privateFunctions`.

## Build

```
./build.sh
```

Requires Xcode with Metal toolchain (`metal-as`, `metallib`).
