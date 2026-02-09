#!/bin/bash
# Compiles the IR shell sources (.ll) into .metallib files.
# Output goes to Sources/FlashAttention/Resources/.
#
# Requirements: Xcode with Metal toolchain (xcrun metal-as, xcrun metallib)

set -euo pipefail
cd "$(dirname "$0")"

OUT="../Sources/FlashAttention/Resources"

for name in gemm_shell_v2 attention_shell; do
  echo "Building ${name}..."
  xcrun metal-as "${name}.ll" -o "${name}.air"
  xcrun metallib "${name}.air" -o "${OUT}/${name}.metallib"
  rm "${name}.air"
  echo "  -> ${OUT}/${name}.metallib"
done

echo "Done."
