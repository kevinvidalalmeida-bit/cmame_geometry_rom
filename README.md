# Geometry-Compiled Sobol+POD ROM

This repository builds reduced elasticity operators for fixed voxel geometries
and varying constituent properties.

The active workflow is intentionally small:

1. generate the interpretable geometry design;
2. generate the ten fixed voxel geometries;
3. solve a fixed 14-material Sobol training set with FFTHomPy;
4. retain the full numerical snapshot span;
5. compile the affine reduced operators `Kq`, `Bq`, and `Dq` in mixed precision;
6. evaluate the resulting ROM over large material sets.

The single entry point and its configuration are:

```text
scripts/cmame_interpretable_pipeline/main.py
scripts/cmame_interpretable_pipeline/campaign_config.json
```

Run the existing geometries with:

```bash
python scripts/cmame_interpretable_pipeline/main.py --stage pipeline
```

The principal offline backend stores the complete raw snapshot span
contiguously in `float32` and never forms its Euclidean Gram matrix during
enrichment. It evaluates the exact constitutive-rank affine contractions on
the GPU in `float32`, keeps the reduced `float64` accumulators resident, and
overlaps pinned snapshot transfers with the next contraction. Once stopping
is reached, a reference-energy Cholesky QR orthonormalizes the full span from
the already assembled raw Ritz blocks, without another voxel pass. No
snapshot direction, affine coefficient, or POD mode is truncated. Online
material batches use vectorized CUDA algebra. The adaptive run solves five
monitor materials once and uses them only for stopping. After the ROM is
frozen, an independent 20-material maximin design selected from a 4096-point
Sobol pool validates the ROM and never controls its construction. Validation
reports the observed counts below `1e-4` and `1e-3`, equivalent to `0.01%` and
`0.1%` relative Frobenius error. These percentages are empirical and are not
presented as coverage of the entire continuous material domain.

The active configuration points to the completed `60`, `150`, and `240` voxel
geometries. The final validation set provides finite empirical error evidence;
the workflow makes no global certification claim over the continuous material
domain.

For full-rank operation, raw-snapshot memory grows as `O(Nvox*p)`,
offline K/B/D contractions as `O(Nvox*p^2)`, and dense online solves as
`O(r^3)`, with `r=p` in the reported campaign. The active pipeline solves and
loads one material at a time. Its two asynchronous buffers are voxel chunks,
not material batches, and their total pinned-memory cap is `1 GiB`. The
historical Gram compiler remains available through the three `--no-*` route
switches documented in the pipeline README.
