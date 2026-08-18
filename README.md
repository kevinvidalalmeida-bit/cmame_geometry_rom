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

The optimized offline backend stores the voxel basis contiguously in `float32`,
reveals the numerically resolved rank through the small snapshot Gram matrix,
orders voxels for orientation-grouped affine kernels, and accumulates reduced
blocks in `float64`. Ill-conditioned FP32 Ritz coordinates are discarded with
a fixed tolerance; an SPD failure triggers one unregularized FP64 rebuild.
Online material batches use vectorized maps and CUDA when available. In the default
fixed mode no preliminary FFT monitor pool is solved. After the 14 training
materials are frozen, a new independent five-material FFT set validates the
ROM and never controls its construction. An explicit `--adaptive` run instead
solves five preliminary monitor materials by default and reuses them only for
stopping. Validation reports the observed count and percentage below
`1e-4`, equivalent to `0.01%` relative Frobenius error. This percentage is
empirical and is not presented as coverage of the entire continuous material
domain.

The active configuration points to the completed `60`, `150`, and `240` voxel
geometries. The final validation set provides finite empirical error evidence;
the workflow makes no global certification claim over the continuous material
domain.

For exact full-rank operation, basis memory grows as `O(Nvox*r)`, offline POD
and K/B/D work as `O(Nvox*r^2)`, and dense online solves as `O(r^3)`. The active
pipeline shrinks POD, K/B/D, and ROM batches as rank grows. It fails before
creating output only when the exact basis plus minimum workspaces cannot fit.
