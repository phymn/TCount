#!/bin/sh
# RTMPose-t Body7 ONNX fetch — DISABLED by default.
# Weight redistribution is blocked until commercial/dataset licensing is cleared
# (see crowd/LICENSES.txt). This script refuses unless ALLOW_UNCLEARED_POSE_WEIGHTS=1
# is set for local engineering tests. Never commit the resulting .onnx.
set -e
cd "$(dirname "$0")/.."
OUT=crowd/models/rtmpose_t_body7_256x192.onnx
URL=https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip
EXPECT=13350364

if [ "${ALLOW_UNCLEARED_POSE_WEIGHTS:-}" != "1" ]; then
  cat <<EOF
Refusing to download RTMPose weights.

Reason: Body7 dataset commercial eligibility is unresolved (AI Challenger and related
terms). COCO-WholeBody is explicitly non-commercial. See crowd/LICENSES.txt.

For local tests only:
  ALLOW_UNCLEARED_POSE_WEIGHTS=1 sh tools/fetch_pose_model.sh

Do not commit $OUT. Do not deploy uncleared weights to production.
EOF
  exit 2
fi

echo "WARNING: downloading uncleared RTMPose weights for LOCAL TEST only → $OUT"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
curl -fL --retry 3 -o "$tmpdir/rtmpose-t.zip" "$URL"
unzip -q -o "$tmpdir/rtmpose-t.zip" -d "$tmpdir"
src=$(find "$tmpdir" -name 'end2end.onnx' | head -n 1)
[ -n "$src" ] || { echo "end2end.onnx not found in zip"; exit 1; }
cp "$src" "$OUT"
s=$(wc -c < "$OUT" | tr -d ' ')
[ "$s" = "$EXPECT" ] || { echo "size mismatch: $s != $EXPECT"; exit 1; }
echo "wrote $OUT ($s bytes) — gitignored; not for redistribution"
