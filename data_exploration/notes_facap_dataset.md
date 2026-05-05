# Notes: FACap dataset shape and design assumptions

Quick-reference notes on what FACap actually is, what it covers, and —
most importantly — what it *doesn't* cover by design. Useful for
framing scope in reports and for understanding why some "edge cases"
(e.g. cross-category queries) cannot exist in this benchmark.

> Surfaced from a Plan_4 reading-pass discussion. Full execution log
> for that pass is in `Documentation/Progress_4_20260501.md`.

---

## 1. Six categories, fully partitioned

FACap ships per-category triplet and caption files under
`data_exploration/datasets/facap-repo/data/facap/`:

```
cir_triplets/{cat}_train_triplets.json
image_captions/{cat}_train_captions.json
```

Counts (train split):

| Category | Triplets | Captioned images |
|---|---:|---:|
| dress | 59,082 | 59,082 |
| jacket | 27,122 | 27,122 |
| outfit | 42,544 | 42,544 |
| pants | 31,644 | 31,644 |
| skirt | 35,120 | 35,120 |
| top | 32,168 | 32,168 |
| **total** | **227,680** | **227,680** |

Triplets = number of (candidate, modification, target) annotations in
that category. Captions = number of unique images that have a
pre-computed long-form caption. Per category, **the two counts are
identical** — every triplet's candidate and target image has a
caption. No data-side mismatch.

## 2. Within-category modification — the key design assumption

Every FACap triplet's candidate and target belong to the **same
category**. The modification text (`captions[0]` in each triplet)
varies attributes like color, length, neckline, sleeves, fabric, and
pattern, but never changes the garment type. Sampling from
`dress_train_triplets.json`:

> *"The dress is longer and made of linen, featuring a V-neckline and
> long sleeves..."*
>
> *"The dress is brighter in color and has a relaxed fit with
> three-quarter length sleeves..."*
>
> *"The dress is shorter and features a high neckline, has a uniform
> gray color..."*

Every modification reads as *"The dress is ..."* — the implicit
contract is "same garment type, modified attributes."

**FACap does not contain cross-category triplets.** There is no "turn
this dress into a skirt" or "make this top a jacket" example anywhere
in the dataset.

## 3. What this means for the baseline

### Phase A: dress only

We're running the full pipeline on the `dress` category exclusively
(Plan 3 baseline + Plan 4 caption analysis + Plan 5 contrastive). Two
reasons:

1. Largest single-category gallery (59 k > all others), so retrieval
   is the most discriminating.
2. Single-category lets Phase A focus on quality of caption + encoder,
   not on category routing.

### Gallery is complete by construction

Because every triplet target has a caption, building the DB from
`dress_train_captions.json` (59,082 rows) and evaluating on triplets
sampled from `dress_train_triplets.json` produces `n_unranked = 0` for
all 1000 evaluation queries in `runs/baseline_v1_speechqwen2vl/`.
There are no orphaned targets.

### Why the notebook still has a "missing-from-DB" bucket

`notebooks/caption_analysis.py` Section 6 stratifies queries by rank
into 5 ranked buckets + a `missing-from-DB` bucket. That last bucket
is **always empty in the current run** (n_unranked = 0). It exists as
defensive scaffolding for configurations that *would* produce orphans:

- **DB down-sampling for ablations** — `scripts/run_baseline_v1.sh`
  supports `DB_SIZE < 59082` (smaller gallery for studying R@K vs
  gallery size, or for fast plumbing iteration on tiny galleries). A
  random subset of captions can leave some triplet targets outside the
  DB.
- **Cross-split eval** — running query triplets from `dress_val` (if
  it existed) against a DB built from `dress_train` would produce
  orphans.
- **Cross-category eval** — see §4.

Code that silently treats `rank=None` as a successful retrieval would
hide real bugs in any of these configurations; the explicit bucket
forces the failure mode to be visible in metrics.

## 4. Cross-category queries are out of scope by dataset choice

A natural-sounding query like *"give me the same look but as a
skirt"* is **structurally unrepresentable in FACap**. There is no
training signal for cross-category modifications and no evaluation
triplet that tests them. Building a system that handles such queries
would require:

- a dataset with cross-category modification triplets (e.g.
  Fashion200k has limited cross-category coverage; newer benchmarks
  like CIReVL and LasCo target freer queries),
- a DB indexed across categories rather than per-category,
- a captioner / training objective that can output a target-category
  description even when the input image is a different category.

This belongs in the **Limitations** section of the final report, not
in the system itself. Phase A and Phase B both stay within-category.

## 5. Artifacts and where to look

- Triplets / captions JSON: `data_exploration/datasets/facap-repo/data/facap/`
- Image cache (PIL files, keyed by image_id): `data_exploration/datasets/facap-images/`
  → symlink to `/ssd1/zhuoyuan/facap-images/` (3.7 GB, 201,550 images
  from the Marqo/fashion200k mirror — covers all categories).
- `src/data/facap_dataset.py` — `FacapDataset(category, split)` reads
  the per-category JSONs; `ds.captions` is the image-path → caption
  dict; `_path_to_image_id()` converts FACap-relative paths to image
  IDs for indexing.

## See also

- `Documentation/Progress_4_20260501.md` — caption analysis findings
  (length gap, Spearman, per-bucket overlap, where this assumption was
  surfaced).
- `Documentation/notes_design_considerations.md` — what R@K targets
  are "good" for which UX, and why R@10 is the headline metric for
  Phase A.
