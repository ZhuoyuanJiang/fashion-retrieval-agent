# Storage Layout — `runs/` lives outside this repo

The `runs/` directory in this project is a **symlink** to a path on
**vllab6's local SSD**, not real data inside the home folder.

## Why

Training artifacts (checkpoints, caption embedding DBs, eval outputs) grew
the project to ~15 GB and were eating the 100 GB NAS home quota. Per lab
policy, large data belongs on local server drives.

## Where it actually lives

- **Server**: vllab6 (hostname: `vision-s006`)
- **Real path**: `/ssd1/zhuoyuan/fashion-retrieval-agent_runs/`
- **Symlink in repo**: `runs/` → `/ssd1/zhuoyuan/fashion-retrieval-agent_runs/`
- **Moved on**: 2026-05-05

## Important caveats

1. **The symlink only resolves on vllab6.** From any other server (vllab10,
   vllab11, etc.) the symlink target does not exist — `runs/` will appear
   empty / broken. Plan training and evaluation that needs these artifacts
   on vllab6.

2. **`/ssd1` is the same physical drive** that holds `facap-images/`,
   `wandb_cache/`, `hf_cache/`, `pip_cache/` for this project. All related
   project data is co-located on vllab6's `/ssd1`.

3. **Future training runs** should also write to `/ssd1/zhuoyuan/fashion-retrieval-agent_runs/`
   (or any path under `runs/`) — because of the symlink, code that writes
   to `./runs/...` automatically lands on the SSD, not on the NAS.

## What was moved

The full prior `runs/` tree (~15 GB):
- `runs/plan5/` — 4 full training runs (each 18 LoRA epoch checkpoints,
  ~2.2 GB per run) plus smaller smoke runs
- `runs/caption_db/` — pre-computed caption embeddings for 9 baseline
  models (Qwen3-Embedding 0.6B/4B/8B, BGE, e5, mpnet, CLIP-L/14,
  marqo-fashionCLIP, marqo-fashionSigLIP, gte-modernbert)
- `runs/baseline_v1_speechqwen2vl*/` — baseline eval outputs

## Recovery / re-creating the symlink

If the symlink is ever lost (e.g., someone runs `rm runs` thinking it's a
real dir, or the project is freshly cloned on vllab6):

```bash
cd /home/zhuoyuan/projects/fashion-retrieval-agent
ln -s /ssd1/zhuoyuan/fashion-retrieval-agent_runs runs
```

To verify the symlink points to existing data:
```bash
ls -la runs                                      # shows '-> /ssd1/...'
ls /ssd1/zhuoyuan/fashion-retrieval-agent_runs/  # shows plan5/, caption_db/, etc.
```

## Moving to a different server

If you ever need to access these runs from a different server, options:
- `rsync` the whole tree across servers
- Re-run training on the target server's local SSD
- Mount vllab6 over NFS (lab does not normally do this)
