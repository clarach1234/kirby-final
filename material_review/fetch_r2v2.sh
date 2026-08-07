#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 WEIGHTS_DIR" >&2
  exit 2
fi

weights_dir="$(realpath -m "$1")"
destination="${weights_dir}/external/r2v2"
if [[ -e "${destination}" ]]; then
  echo "Refusing to overwrite ${destination}" >&2
  exit 2
fi

temporary="$(mktemp -d)"
cleanup() { rm -rf "${temporary}"; }
trap cleanup EXIT

commit="7f6a8ea7a51782b1e0f89723a9ec137ba0a29913"
git clone --filter=blob:none https://github.com/j-morano/R2-V2.git "${temporary}/R2-V2"
git -C "${temporary}/R2-V2" checkout --detach "${commit}"

mkdir -p "${destination}/source"
for name in infer.py model.py preprocessing.py transformations.py; do
  cp "${temporary}/R2-V2/${name}" "${destination}/source/${name}"
done
curl -fL https://github.com/j-morano/R2-V2/releases/download/v1/av.pth \
  -o "${destination}/av.pth"
curl -fL https://github.com/j-morano/R2-V2/releases/download/v1/av_config.json \
  -o "${destination}/av_config.json"

printf '%s  %s\n' \
  74d425afb714384cb3f4d5db9cc852c1ea6d7552e46c866e29a3777db12b9d80 \
  "${destination}/av.pth" | sha256sum -c -
printf '%s  %s\n' \
  8c4bb170f0f4df5cc21ce6929ac1e6e738c82404fe420310181974f572beff54 \
  "${destination}/av_config.json" | sha256sum -c -

echo "Prepared pinned R2-V2 files in ${destination}"
