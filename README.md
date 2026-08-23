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

The optimized offline backend stores the complete raw snapshot span
contiguously in `float32`, evaluates the snapshot Gram products on the CPU in
`float64`, applies the orientation-grouped affine voxel kernels on the GPU in
`float32`, and performs the small transformations in `float64`. A
`1e-15` machine-zero guard removed no direction in the reported campaign; no
regularization or automatic rebuild is used. Online material batches use
vectorized CUDA algebra. The adaptive run solves five monitor materials once
and uses them only for stopping. After the ROM is frozen, an independent
20-material maximin design selected from a 4096-point Sobol pool validates the
ROM and never controls its construction. Validation reports the observed
counts below `1e-4` and `1e-3`, equivalent to `0.01%` and `0.1%` relative
Frobenius error. These percentages are empirical and are not presented as
coverage of the entire continuous material domain.

The active configuration points to the completed `60`, `150`, and `240` voxel
geometries. The final validation set provides finite empirical error evidence;
the workflow makes no global certification claim over the continuous material
domain.

For full-rank operation, raw-snapshot memory grows as `O(Nvox*p)`,
offline POD and K/B/D contractions as `O(Nvox*p^2)`, and dense online solves
as `O(r^3)`, with `r=p` in the reported campaign. The active pipeline shrinks
POD, K/B/D, and ROM batches as rank grows. It fails before creating output
only when the snapshots plus minimum workspaces cannot fit.
