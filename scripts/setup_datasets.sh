#!/usr/bin/env bash
# Clone the public dataset repos used by this project under
# data_exploration/datasets/. Idempotent — skips repos that already exist.
#
# Total footprint after a clean run: ~240 MB of text annotations, no images.
# For full training data (images), see Documentation/datasets_overview.md.
set -euo pipefail

cd "$(dirname "$0")/.."  # repo root
mkdir -p data_exploration/datasets
cd data_exploration/datasets

clone_if_missing() {
    local dest="$1" url="$2"
    if [[ -d "$dest" ]]; then
        echo "[skip]  $dest already exists"
    else
        echo "[clone] $url -> $dest"
        git clone --depth 1 "$url" "$dest"
    fi
}

clone_if_missing fashion-iq           https://github.com/XiaoxiaoGuo/fashion-iq.git
clone_if_missing fashion-iq-metadata  https://github.com/hongwang600/fashion-iq-metadata.git
clone_if_missing facap-repo           https://github.com/fgxaos/facap-sigir25-gennext.git
clone_if_missing fashion-200k         https://github.com/xthan/fashion-200k.git

echo
echo "Repos ready under data_exploration/datasets/."
echo "Image samples are NOT downloaded by this script:"
echo "  - FashionIQ thumbnails (Amazon CDN per ASIN):"
echo "      run cells in data_exploration/dataset_inspection.ipynb"
echo "  - FACap dress sample via Marqo/fashion200k HuggingFace mirror:"
echo "      data_exploration/venv/bin/python data_exploration/fetch_facap_sample.py"
echo
echo "For full training-time datasets, see Documentation/datasets_overview.md."
