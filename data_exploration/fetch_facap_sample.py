"""Stream a small slice of Marqo/fashion200k and save images that appear in FACap dress triplets.

Run:
    data_exploration/venv/bin/python data_exploration/fetch_facap_sample.py
"""
import json
from pathlib import Path

from datasets import load_dataset

BASE = Path(__file__).parent
TRIPLETS = BASE / "datasets/facap-repo/data/facap/cir_triplets/dress_train_triplets.json"
OUT_DIR = BASE / "datasets/facap-images"
OUT_DIR.mkdir(exist_ok=True, parents=True)

STREAM_N = 300           # how many items to pull from HF
MAX_MATCHES = 4          # stop once we have this many complete triplets

def path_to_id(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".jpeg")

def main():
    with open(TRIPLETS) as f:
        triplets = json.load(f)

    print(f"Streaming up to {STREAM_N} items from Marqo/fashion200k …")
    ds = load_dataset("Marqo/fashion200k", split="data", streaming=True)
    collected = {}
    for i, ex in enumerate(ds):
        if i >= STREAM_N:
            break
        collected[ex["item_ID"]] = ex["image"]
    print(f"  collected {len(collected)} item_IDs")

    matches = []
    for t in triplets:
        c_id, tgt_id = path_to_id(t["candidate"]), path_to_id(t["target"])
        if c_id in collected and tgt_id in collected:
            matches.append({"triplet": t, "cand_id": c_id, "tgt_id": tgt_id})
            if len(matches) >= MAX_MATCHES:
                break
    print(f"  found {len(matches)} triplets with both images in sample")

    for m in matches:
        for key in ("cand_id", "tgt_id"):
            img = collected[m[key]]
            img.save(OUT_DIR / f"{m[key]}.jpeg")
    print(f"  saved {len(matches)*2} images to {OUT_DIR}")

    # Save matches manifest for the notebook to pick up
    manifest = OUT_DIR / "dress_sample_manifest.json"
    with open(manifest, "w") as f:
        json.dump([m["triplet"] for m in matches], f, indent=2)
    print(f"  wrote manifest to {manifest}")

if __name__ == "__main__":
    main()
