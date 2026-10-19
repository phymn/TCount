#!/bin/sh
# Downloads the two large files that are not stored in git into crowd/.
# Run from the repo root (works in iSH: apk add curl). Verifies sizes.
set -e
REL=${REL:-https://github.com/phymn/TCount/releases/download/v1.0}
ORT=https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort-wasm-simd-threaded.asyncify.wasm
dl() { echo "-> $2"; curl -fL --retry 3 -o "$2" "$1"; s=$(wc -c < "$2" | tr -d ' '); [ "$s" = "$3" ] || { echo "size mismatch for $2: $s != $3"; exit 1; }; }
dl "$ORT" crowd/vendor/ort-wasm-simd-threaded.asyncify.wasm 26781914
dl "$REL/dmcount_qnrf_w8.onnx" crowd/models/dmcount_qnrf_w8.onnx 21560628 || echo "dense model optional: app falls back to tiled detection estimate"
echo done
