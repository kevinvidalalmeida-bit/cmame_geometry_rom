# Interpretable 10-Geometry Pipeline

Edit the campaign in one place:

```text
scripts/cmame_interpretable_pipeline/campaign_config.json
```

That file contains the design space, the ten manual cases, geometry seeds,
SAM tolerances, Sobol material seeds, adaptive stopping, and validation settings.

The scripts automatically relaunch inside:

```text
~/Documentos/ANDRES/COMPUTATIONAL_WORKSPACEV4/.venv
```

For an interactive terminal, activate it with:

```bash
source scripts/cmame_interpretable_pipeline/activate_env.sh
```

Default full campaign:

```bash
python scripts/cmame_interpretable_pipeline/main.py --stage all
```

Useful staged runs:

```bash
python scripts/cmame_interpretable_pipeline/main.py --stage design
python scripts/cmame_interpretable_pipeline/main.py --stage geometries
python scripts/cmame_interpretable_pipeline/main.py --stage pipeline
```

Default choices:

- Geometry space: `Vf_target` from `0.05` to `0.25`, aspect ratio from `5` to `20`.
- Ten manual interpretable cases: baseline, low/high `Vf`, short/long `AR`, planar xy, aligned x, biased triaxial, dilute-short, dense-long.
- Geometry resolution: at least `6` voxels per fiber diameter on odd grids,
  giving `61`, `151`, and `241` voxels per side for the current cases. Binary
  voxelization is retained.
- ROM campaign: sequential Sobol + numerical full-rank POD, adding one material at a time.
- FFT precision: training, monitoring, and final validation use `float64` with
  `rtol=1e-8`. Snapshot/POD storage and contractions remain contiguous
  `float32`; reduced operators and online solves remain `float64`.
- Adaptive stop: solve `16` independent FFT monitor materials once, evaluate
  from training material 2 onward, and stop after the maximum monitor error is
  below `1e-5` twice consecutively. This stricter internal target provides one
  order of magnitude of margin for the reported `1e-4` accuracy threshold.
- Final validation: after freezing the ROM, solve a new independent set of
  `16` FFT materials exactly once with `truth_profile=snapshot` (`float64`,
  `rtol=1e-8`). These final points never control training. The output reports
  the count and percentage with relative Frobenius error at or below `1e-4`.
  Since `100 * 1e-4 = 0.01`, that threshold means `0.01%` relative tensor-norm
  error on each observed geometry-material pair; it is not a global-domain
  guarantee or a componentwise error bound.
- Training limit: `0` means continue until convergence, bounded only by the 1024-candidate pool and the exact full-rank memory preflight.
- Online timing: evaluate the final ROM at `10000` independent material points.
- Memory controls: POD and affine-stress workspaces are capped at `8 GiB`, ROM
  batches at `1 GiB`, and exact full-rank runs must fit within 80% of available
  host memory. POD projection uses an in-place BLAS GEMM, so it does not create
  a second snapshot-sized array.
- Online backend: persistent `float64` CUDA operators and batched solves by
  default. Use `--rom-backend auto` only when a CPU fallback is desired.

Scaling with `D = 6*Nvox` and reduced rank `r`:

- Stored basis: `O(D*r)` memory. It is never written to the final run directory.
- Exact incremental POD and K/B/D compilation: `O(D*r^2)` total work.
- Reduced operators: `O(r^2)` memory.
- Dense online solve: `O(r^3)` work per material and `O(chunk*r^2)` workspace.

The adaptive pipeline consumes one material per exact Sobol prefix, bounds the
affine-coefficient and online-query workspaces as rank grows, and retains only
the final reduced operators. Intermediate `Kq/Bq/Dq`, voxel solutions, and the
POD basis are not written to disk. Lower asymptotic growth requires truncated
POD; it is not enabled implicitly because it changes the model space.

Smoke test:

```bash
python scripts/cmame_interpretable_pipeline/main.py --stage all --smoke
```

Primal-dual CRE pilot:

```bash
python scripts/cmame_interpretable_pipeline/05_primal_dual_cre_pilot.py
```

The `cre_pilot` block in `campaign_config.json` controls this experiment. It
builds float64 compatible-strain and equilibrated-stress POD spaces, evaluates
a two-sided GaNi discrete bracket on a finite configured Sobol pool, and runs
the independent FFT validation only after the reduced model is frozen. This is
an experimental validation path, not a global certification of the continuous
material domain. See `CRE_VALIDATION_STATUS.md` before using its results in the
paper. Odd grids avoid the extra Nyquist modes present on even grids, although
they do not by themselves establish a continuous-problem guarantee.

After a smoke test, run the full geometry stage again before the real pipeline:

```bash
python scripts/cmame_interpretable_pipeline/main.py --stage geometries
python scripts/cmame_interpretable_pipeline/main.py --stage pipeline
```

Outputs:

- `results/cmame_method/interpretable_vf05_25_ar5_20_odd/design/`
- `results/cmame_method/interpretable_vf05_25_ar5_20_odd/geometries/`
- `results/cmame_method/interpretable_vf05_25_ar5_20_odd/runs/run_YYYYMMDD_HHMMSS/`

Named run:

```bash
python scripts/cmame_interpretable_pipeline/main.py --stage pipeline --run-name run_sobol_pod_v1
```

Move the campaign results to the desktop trash for a fresh restart:

```bash
gio trash results/cmame_method/interpretable_vf05_25_ar5_20_odd
```
