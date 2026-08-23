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
- ROM campaign: adaptive Sobol-prefix training starts from two materials and
  stops at the first maximum monitor error below `1e-4`. Five independent
  monitor materials are solved once and never enter the snapshot matrix.
- FFT precision: training, monitoring, and FOM timing use `float32` with
  `rtol=1e-5`; held-out references use `float64` with `rtol=1e-6` only to
  measure error. Snapshots and exact constitutive-rank contractions use
  `float32`; reduced accumulation, reference-energy QR, and online dense
  solves use `float64`. No structural audit is part of the nominal campaign.
- Full-span coordinates: enrichment retains raw snapshots and assembles exact
  affine Ritz blocks without a Euclidean snapshot Gram. At freeze, the mean
  candidate coefficient vector defines a coercive reference operator whose
  Cholesky factor supplies the QR coordinates. This introduces no truncation
  and requires no additional voxel-scale pass.
- Ritz streaming: two CUDA streams and two pinned voxel-chunk buffers overlap
  host-to-device transfer with factorized contractions. The total pinned cap
  is `1 GiB`; snapshot generation and loading remain one material at a time.
- The ROM archive and SHA-256 hash are written before the held-out design is
  created. Twenty common validation materials are selected deterministically
  by affine-coefficient maximin traversal of an independent 4096-point Sobol
  pool with seed `20260901`.

For example, this activates those five preliminary monitor solves:

```bash
python scripts/cmame_interpretable_pipeline/04_sobol_pod_pipeline.py \
  --geometry-id 9 --adaptive
```
- Final validation: after freezing the ROM, solve the 20-point maximin design
  once with the `reference` profile. These final points never control
  training. The output reports
  the count and percentage with relative Frobenius error at or below `1e-4`
  and `1e-3`.
  Since `100 * 1e-4 = 0.01`, that threshold means `0.01%` relative tensor-norm
  error on each observed geometry-material pair; it is not a global-domain
  guarantee or a componentwise error bound.
- Monotonicity: the complete snapshot spans are nested as response blocks are
  appended. In exact arithmetic this gives monotone Loewner convergence of
  the Ritz--Schur operators and nonincreasing monitor errors. The first
  crossing remains a finite-monitor heuristic rather than a domain-wide
  guarantee.
- Training limit: adaptive mode defaults to `adaptive_training_limit=0`,
  meaning it continues until all five monitor
  cases meet the tolerance or the memory-safe candidate limit is reached. An
  explicit `--training-limit` overrides that behavior. Reaching such a limit
  freezes and validates the best available ROM with a warning; it is not a
  fatal pipeline error.
- Online timing: use one NumPy/OpenBLAS thread for 1,000 hot isolated queries
  and CuPy/CUDA with resident operators for ten synchronized repetitions of
  one 10,000-query batch. Both paths include affine-coefficient evaluation and
  result return. Ten fresh-process CUDA cold starts are recorded separately;
  speed-up and break-even use the lower isolated-query CPU latency.
- Memory controls: POD and affine-stress workspaces are capped at `8 GiB`, ROM
  batches at `4 GiB`, pinned Ritz buffers at `1 GiB`, and full-rank runs must
  fit within 80% of available host memory. The raw-Ritz path avoids a second
  snapshot-sized array.
- Online backend: one-thread CPU for isolated latency and persistent `float64`
  CUDA operators for batched throughput.

Scaling with `D = 6*Nvox`, raw snapshot count `p`, and reduced rank `r`:

- Stored raw snapshots: `O(D*p)` memory. They are discarded after compilation.
- Incremental K/B/D compilation: `O(D*p^2)` total work.
- Reduced operators: `O(r^2)` memory.
- Dense online solve: `O(r^3)` work per material and `O(chunk*r^2)` workspace.

The adaptive pipeline consumes a fixed Sobol prefix, compiles the complete
snapshot span, bounds affine-coefficient and online-query workspaces, and
retains only the final reduced operators. Intermediate voxel solutions and
the Ritz basis are not written to disk. Lower asymptotic growth would require
truncating or otherwise approximating the snapshot space; that defines a
different model and is not enabled.

The principal route is enabled by `reference_energy_qr`, `factorized_ritz`,
`async_ritz`, and `gathered_factor_ritz` in `campaign_config.json`. For each
voxel chunk, the implementation uploads the raw snapshots once, gathers the
orientation-dependent exact constitutive factors once, and reuses both while
assembling all seven affine blocks. Pinned double buffering overlaps packing
and host-to-device transfer of the next chunk with the current GPU
contractions. Materials are still solved and appended one at a time, so this
route does not increase the material batch size or its VRAM footprint.

To audit the historical CPU-Gram/component-action compiler, disable the four
dependent switches:

```bash
python scripts/cmame_interpretable_pipeline/04_sobol_pod_pipeline.py \
  --geometry-id 9 --adaptive \
  --no-gathered-factor-ritz --no-async-ritz \
  --no-factorized-ritz --no-reference-energy-qr
```

Additional reproducible checks are available as:

```bash
python scripts/cmame_interpretable_pipeline/07_fft_adaptation_verification.py --require-gpu
python scripts/cmame_interpretable_pipeline/08_voxel_resolution_convergence.py
```

The second command revoxelizes the unchanged continuous masters for G08, G00,
and G09 at 3, 4, 5, and 6 voxels per micrometre using the declared Carbon Fiber
(290 GPa)/Resin Epoxy material and the nominal `snapshot32` profile.

Rebenchmark the frozen ROMs without repeating any FFT solve:

```bash
python scripts/cmame_interpretable_pipeline/09_rom_backend_benchmark.py \
  --summary-dir results/cmame_method/interpretable_vf05_25_ar5_20/runs/full_span_gathered_20260823_summary
```

This writes CPU isolated-query and CuPy/CUDA batch measurements to
`rom_backend_benchmark.csv` and records the full protocol in the companion
JSON file.

## Data-only surrogate baselines

After the ten ROMs and validation references have been frozen, compare the
same Sobol prefixes against RBF and Kriging without running any additional FFT
solve:

```bash
python scripts/cmame_interpretable_pipeline/06_surrogate_baselines.py \
  --runs-root results/cmame_method/interpretable_vf05_25_ar5_20/runs \
  --base-run-name full_span_gathered_20260823 \
  --output-dir results/cmame_method/interpretable_vf05_25_ar5_20/runs/full_span_gathered_20260823_summary \
  --paper-figure-dir paper/figures \
  --jobs 4
```

The benchmark uses the seven affine coefficients as inputs and the 21
independent Mandel stiffness entries as outputs. Input ranges come only from
the declared candidate pool. At every geometry and prefix, the RBF kernel,
shape parameter, smoothing, and admissible polynomial degree are selected by
minimum leave-one-out relative tensor error on that training prefix. The
search includes linear, thin-plate, cubic, quintic, multiquadric,
inverse-multiquadric, inverse-quadratic, and Gaussian kernels. Kriging compares
isotropic and ARD Matern, squared-exponential, rational-quadratic, and mixed
linear covariance families by optimized training log marginal likelihood;
the selected family is then refitted with ten deterministic optimizer
restarts. Neither method is selected or tuned with the 20 held-out responses.
The detailed, learning-curve, per-geometry, global, and complete
hyperparameter-selection records are written to the aggregated campaign
directory. The flattened selections are available in
`surrogate_baseline_prefix_hyperparameters.csv` and
`surrogate_baseline_final_hyperparameters.csv`; the JSON protocol also keeps
the scores of every Kriging covariance candidate.

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
