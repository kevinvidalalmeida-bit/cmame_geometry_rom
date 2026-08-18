# Interpretable 10-Geometry Pipeline

Edit the campaign in one place:

```text
scripts/cmame_interpretable_pipeline/campaign_config.json
```

That file contains the design space, the ten manual cases, geometry seeds,
SAM tolerances, Sobol material seeds, fixed training, and validation settings.

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
- Geometry resolution: `6` voxels per fiber diameter, giving `60`, `150`, and
  `240` voxels per side for the current cases. Binary voxelization is retained.
- ROM campaign: a fixed 14-material Sobol set followed by numerical full-rank
  POD/Ritz compilation. The solve order may change for warm starts, but the
  selected Sobol set does not.
- FFT precision: training and final validation use the configured `snapshot`
  profile (`float32`, `rtol=1e-5`). Snapshot storage remains contiguous
  `float32`; reduced operators and online solves remain `float64`.
- The default fixed mode solves no preliminary monitor or calibration FOMs.
  Reduced operators are compiled once after the 14 training materials.
- Optional adaptive mode is enabled explicitly with `--adaptive`. It solves an
  independent pool of `5` monitor materials once before training and reuses
  their truth solutions at each stopping check. `--monitor-count` can override
  that default without changing the final held-out validation set.

For example, this activates those five preliminary monitor solves:

```bash
python scripts/cmame_interpretable_pipeline/04_sobol_pod_pipeline.py \
  --geometry-id 9 --adaptive
```
- Final validation: after freezing the ROM, solve a new independent set of
  `5` FFT materials exactly once with `truth_profile=snapshot`. These final
  points never control training. The output reports
  the count and percentage with relative Frobenius error at or below `1e-4`.
  Since `100 * 1e-4 = 0.01`, that threshold means `0.01%` relative tensor-norm
  error on each observed geometry-material pair; it is not a global-domain
  guarantee or a componentwise error bound.
- Monotonicity: in exact arithmetic, retaining the previous Ritz basis and
  only appending new directions makes the tensor error nonincreasing. Mixed
  precision and finite solver tolerances can introduce small numerical
  perturbations. The ten completed curves contained no observed increases.
- Training limit: fixed mode uses `14` materials. Adaptive mode defaults to
  `adaptive_training_limit=0`, meaning it continues until all five monitor
  cases meet the tolerance or the memory-safe candidate limit is reached. An
  explicit `--training-limit` overrides that behavior. Reaching such a limit
  freezes and validates the best available ROM with a warning; it is not a
  fatal pipeline error.
- Online timing: evaluate the final ROM at `10000` independent material points.
- Memory controls: POD and affine-stress workspaces are capped at `8 GiB`, ROM
  batches at `1 GiB`, and exact full-rank runs must fit within 80% of available
  host memory. The default raw-Ritz path reveals numerical rank through the
  small snapshot Gram matrix and avoids a second snapshot-sized array.
- Online backend: persistent `float64` CUDA operators and batched solves by
  default. Use `--rom-backend auto` only when a CPU fallback is desired.

Scaling with `D = 6*Nvox` and reduced rank `r`:

- Stored basis: `O(D*r)` memory. It is never written to the final run directory.
- Exact incremental POD and K/B/D compilation: `O(D*r^2)` total work.
- Reduced operators: `O(r^2)` memory.
- Dense online solve: `O(r^3)` work per material and `O(chunk*r^2)` workspace.

The fixed pipeline consumes the prescribed Sobol training set, compiles at the
final rank, bounds affine-coefficient and online-query workspaces, and retains
only the final reduced operators. Intermediate voxel solutions and the POD
basis are not written to disk. Lower asymptotic growth requires truncated POD;
it is not enabled implicitly because it changes the model space.

Smoke test:

```bash
python scripts/cmame_interpretable_pipeline/main.py --stage all --smoke
```

After a smoke test, run the full geometry stage again before the real pipeline:

```bash
python scripts/cmame_interpretable_pipeline/main.py --stage geometries
python scripts/cmame_interpretable_pipeline/main.py --stage pipeline
```

Outputs:

- `results/cmame_method/interpretable_vf05_25_ar5_20/design/`
- `results/cmame_method/interpretable_vf05_25_ar5_20/geometries/`
- `results/cmame_method/interpretable_vf05_25_ar5_20/runs/run_YYYYMMDD_HHMMSS/`

Named run:

```bash
python scripts/cmame_interpretable_pipeline/main.py --stage pipeline --run-name run_sobol_pod_v1
```

Move the campaign results to the desktop trash for a fresh restart:

```bash
gio trash results/cmame_method/interpretable_vf05_25_ar5_20
```
