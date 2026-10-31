#!/usr/bin/env bash
# Download the local detector model (YOLOX-tiny, Apache-2.0, ~19 MB) into models/ and verify it.
set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx"
SHA256="427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"
OUT="models/yolox_tiny.onnx"

sha() { python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }

mkdir -p models
if [ -f "$OUT" ] && [ "$(sha "$OUT")" = "$SHA256" ]; then
  echo "$OUT is already here and verified."
  exit 0
fi
curl -fL --progress-bar -o "$OUT.part" "$URL"
if [ "$(sha "$OUT.part")" != "$SHA256" ]; then
  rm -f "$OUT.part"
  echo "Checksum mismatch: the download is not the expected model. Nothing was installed." >&2
  exit 1
fi
mv "$OUT.part" "$OUT"
echo "Downloaded and verified $OUT."
