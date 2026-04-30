# tests/

Reproducibility checks for the Plan_2 baseline. Each file is
dual-mode — runnable as a plain Python script (no pytest required) or
discoverable by pytest:

```bash
# Script mode — prints ✓/✗ per case
python -m tests.test_m1_facap_dataset
python -m tests.test_m2_caption_db
python -m tests.test_m3_pipeline

# Or all 13 cases at once via pytest (optional install)
pytest tests/
```

| File | Covers | Cases |
|---|---|---|
| `test_m1_facap_dataset.py` | M1 — `FacapDataset` schema + `load_image()` | 5 |
| `test_m2_caption_db.py` | M2 — caption DB shape, L2-norm, source split, provenance | 5 |
| `test_m3_pipeline.py` | M3 — oracle R@1=1.0, mock clean exit, real-backend VRAM gate | 3 |
| `_runner.py` | Shared script-mode runner (no test cases) | — |

Tests build into fresh `runs/_test_*/` directories so they don't
collide with the user's smoke runs (all under the gitignored `runs/`).

For run commands and a summary, see the root
[`README.md`](../README.md) → *Tests*.
For what each individual case verifies, see
[`Documentation/Progress_2_20260420.md`](../Documentation/Progress_2_20260420.md)
→ *Test suite*.
