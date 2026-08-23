# Local-frame and gathered-factor Ritz study

## Purpose

This study tested two exact alternatives to the grouped global-frame
factorized Ritz compiler. Neither route changes the snapshot span, applies a
low-rank truncation, or changes `load_batch_size=1`. The gathered-factor
alternative was subsequently promoted to the nominal campaign route; the
local-frame route remains experimental.

1. `--experimental-local-frame-ritz` rotates the fiber-supported snapshot
   fields once into each voxel's local Mandel frame. All fiber orientations
   then share the same constitutive factors during the Ritz contraction.
2. `--gathered-factor-ritz` keeps the snapshots in the global
   frame and gathers the rotated constitutive factors once per voxel chunk.
   This removes the repeated orientation-group loop without requiring a
   snapshot conversion.

The two routes are mutually exclusive and require the factorized Ritz path.
`gathered_factor_ritz` is enabled by default in `campaign_config.json`.

## Exact transformation

Let `M_g` be the energy-preserving Mandel rotation from local to global
coordinates for orientation group `g`. For a global snapshot strain `e`,

```text
e_local = M_g.T @ e
C_global = M_g @ C_local @ M_g.T
```

therefore

```text
e.T @ C_global @ f = e_local.T @ C_local @ f_local.
```

The local-frame route applies this identity to the fiber-supported snapshot
coordinates. The gathered-factor route instead evaluates the algebraically
equivalent global factors `M_g @ F_local` in one gathered tensor operation.

## Commands

```bash
python scripts/cmame_interpretable_pipeline/04_sobol_pod_pipeline.py \
  --geometry-id 3 --adaptive --final-validation-count 1 \
  --run-name experimental_gathered_factor_g03_active_20260823 \
  --gathered-factor-ritz --overwrite

python scripts/cmame_interpretable_pipeline/04_sobol_pod_pipeline.py \
  --geometry-id 9 --adaptive --final-validation-count 1 \
  --run-name experimental_gathered_factor_g09_active_20260823 \
  --gathered-factor-ritz --overwrite
```

## Results

All measurements used one training material at a time. The principal
comparison is the existing factorized asynchronous route.

| Geometry | Route | Assembly [s] | Local transform [s] | Compilation [s] | Stress workspace [GiB] | Pinned host [GiB] |
|---|---|---:|---:|---:|---:|---:|
| G03 | principal | 1.357 | 0 | 3.831 | 0.340 | 0.637 |
| G03 | local frame | 1.395 | 0.043 | 3.792 | 0.340 | 0.637 |
| G03 | gathered factors | 1.258 | 0 | 3.570 | 0.340 | 0.637 |
| G09 | principal | 35.236 | 0 | 188.362 | 0.979 | 1.000 |
| G09 | local frame | 30.685 | 3.366 | 187.364 | 1.170 | 1.000 |
| G09 | gathered factors | 31.362 | 0 | 181.720 | 1.337 | 1.000 |

For G09, gathered factors reduce assembly by 3.874 s, or 11.0%. The observed
compilation reduction is 6.642 s, or 3.5%, although 2.396 s of that difference
comes from normal variation in the FOM solve time. Holding the solve time
fixed, the directly attributable total reduction is about 2.1%.

The gathered route reached the same adaptive stops as the principal route:
G03 used 11 materials and rank 66, while G09 used 16 materials and rank 96.
Against the principal G09 archive, the relative differences in the raw
affine blocks were `1.76e-7` for `Kq`, `1.17e-7` for `Bq`, and zero for `Dq`.

## Decision

The gathered-factor route was promoted to the nominal compiler. It delivers
a meaningful reduction on G09 and avoids the one-time PCIe round trip
required by local-frame snapshot storage. The local-frame implementation is
retained only as an exact reference experiment.

The next promising optimization is a material-block contraction layout.
The current G09 gathered run spends 23.35 s packing rank-major snapshots into
pinned spatial chunks. Processing contiguous six-load material blocks could
remove that host copy, but it requires caching the new-material features so
they are not recomputed for every old/new cross block. This should remain a
separate experiment because it changes the streaming schedule and GPU memory
lifetime substantially.
