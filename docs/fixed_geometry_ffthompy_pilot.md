# Fixed-Geometry FFTHomPy Pilot

Date: 2026-08-11

## Purpose

Validate the immediate paper workflow with one fixed short-fiber geometry:

- one fixed geometry `G`, no geometry realizations;
- `AR = 15`;
- `V_f = 0.20`;
- approximately 100 short fibers;
- isotropic matrix + transversely isotropic fiber;
- many material points solved with the same `phase.npy` and `ori.npy`;
- FFTHomPy/CuPy full-order solves on GPU.

This is the production full-order baseline needed before compiling the
geometry into a reduced constitutive transfer operator.

## Entry Point

`FFT/` is now kept clean as the SAM + FFTHomPy runtime core. The fixed-geometry
paper workflow lives outside it:

```bash
./scripts/fixed_geometry_ffthompy_sweep.py --material-points 8 \
  --solution-field-material-ids 0 \
  --out-name fixed_geometry_ar15_vf20_sobol8_center_fields
```

The script:

1. generates one SAM geometry using `FFT/pipeline/rve_generator.py`;
2. stores `_fixed_geometry/phase.npy` and `_fixed_geometry/ori.npy`;
3. hashes those arrays;
4. samples the seven-dimensional constituent domain
   `(Em, nu_m, Ef_L, Ef_T, G_LT, nu_LT, nu_TT)`;
5. solves each material with `FFT/pipeline/fft_solver.py` and FFTHomPy/CuPy;
6. writes `Ceff.npy`, solver timings, material CSV/XLSX files and a manifest.

`--solution-field-material-ids 0` saves the six zero-mean fluctuation fields for
the center material. Those are the first tangential enrichment directions for
the future reduced operator.

## Current Run

Authoritative output:

```text
results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields
```

Geometry:

| Quantity | Value |
|---|---:|
| Target AR | 15.0 |
| Target Vf | 0.20 |
| Nominal fiber count | 100 |
| Actual fibers | 96 |
| Grid | 91^3 |
| Domain length | 18.059971 um |
| Lf / Ldom | 0.830566 |
| Actual voxel Vf | 0.199284474 |
| Vf error | -0.07155 percentage points |
| Continuous A2 relative error | 0.002410 |
| Voxelized A2 relative error | 0.003536 |
| Final overlap | 0.04609 voxel |
| Vf / overlap / A2 checks | passed |
| Geometry hash | `0c167523f0a5_23b849c7644a` |
| SAM backend | numba |
| SAM strategy | collective_fire |

Material sweep:

| Quantity | Value |
|---|---:|
| Material design | center + Sobol |
| Material points | 8 |
| Successful GPU solves | 8 |
| Failed solves | 0 |
| Unique `phase.npy` hashes | 1 |
| Unique `ori.npy` hashes | 1 |
| Mean material solve wall time | 0.460 s |
| Max Ceff symmetry relative error | 8.498e-08 |
| Minimum eigenvalue over sym(Ceff) | 3.374 |

The mean wall time includes the center-material solve that also wrote the six
full fluctuation fields. The non-field solves are around 0.1-0.3 s each on the
current GPU run.

## Material Points

| ID | Em | nu_m | Ef_L | Ef_T | G_LT | nu_LT | nu_TT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.750000 | 0.385000 | 233.500000 | 14.500000 | 19.000000 | 0.230000 | 0.375000 |
| 1 | 4.136018 | 0.403813 | 290.843164 | 6.320013 | 26.154376 | 0.240749 | 0.360239 |
| 2 | 1.933353 | 0.372175 | 152.422051 | 16.226382 | 10.723105 | 0.219325 | 0.395428 |
| 3 | 1.452036 | 0.389688 | 346.479402 | 10.260819 | 16.078560 | 0.208937 | 0.365947 |
| 4 | 3.374373 | 0.351352 | 166.968956 | 20.677680 | 20.626966 | 0.250985 | 0.377611 |
| 5 | 3.119848 | 0.400516 | 81.979914 | 20.881198 | 23.745899 | 0.233591 | 0.368948 |
| 6 | 1.816257 | 0.362418 | 260.190144 | 14.315568 | 18.642245 | 0.226480 | 0.387103 |
| 7 | 2.498368 | 0.414897 | 217.183527 | 16.952877 | 13.024971 | 0.201099 | 0.350936 |

Selected effective engineering constants:

| ID | E1 | E2 | E3 | G12 | G13 | G23 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.735135 | 6.723655 | 7.347859 | 2.819249 | 2.548552 | 2.584210 |
| 1 | 8.816814 | 8.877620 | 9.631022 | 3.651188 | 3.308114 | 3.373853 |
| 2 | 4.723408 | 4.694723 | 5.127304 | 1.982351 | 1.793706 | 1.817491 |
| 3 | 4.641872 | 4.690855 | 5.203509 | 2.002483 | 1.793947 | 1.802773 |
| 4 | 7.239935 | 7.189774 | 7.734471 | 3.017202 | 2.767113 | 2.804160 |
| 5 | 5.956528 | 5.900378 | 6.236938 | 2.341141 | 2.198773 | 2.218250 |
| 6 | 5.195039 | 5.206611 | 5.736617 | 2.241847 | 2.013367 | 2.031576 |
| 7 | 6.267424 | 6.240021 | 6.851046 | 2.572448 | 2.321928 | 2.353193 |

## Tangential Snapshot

The center material saved:

```text
material_0000/solution_fields.npz
```

Contents:

- `fluctuation_load0` through `fluctuation_load5`;
- each field has shape `6 x 91 x 91 x 91`;
- dtype is `float32`;
- macro field is subtracted;
- component means are near zero, with absolute means below `3e-9` in this run.

These six fields are the first basis block for the intended tangential ROM:

```text
G -> {six FFTHomPy fluctuation responses at xi_center} -> V_6
```

Additional tangential fields were then collected on the same fixed geometry
for the current worst material after each ROM stage:

```bash
./scripts/fixed_geometry_collect_tangential.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --material-ids 1
```

The base-set enrichment material IDs are `1, 3, 5, 7, 2`. Each recomputed
FFTHomPy tensor matched the original full-order CSV with relative differences
between `4.0e-08` and `2.3e-07`, confirming that the same fixed geometry was
reused.

Independent validation then exposed two additional material points that were
added adaptively from the validation CSV while still reusing the same fixed
`phase.npy` and `ori.npy`:

| Source validation ID | Stored snapshot ID | Label | Recompute error |
|---:|---:|---|---:|
| 13 | 1013 | `independent_sobol_0013` | 1.05e-07 |
| 3 | 1003 | `independent_sobol_0003` | 2.02e-08 |
| 15 | 1015 | `independent_sobol_0015` | 6.70e-08 |

## Reduced Operator Check

The reduced operators are compiled outside `FFT/`:

```bash
./scripts/fixed_geometry_reduced_operator.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --snapshot-material-ids 0 1 2 3 5 7 \
  --out-name rom_tangential_r36_center_m1_m2_m3_m5_m7
```

The script assembles seven affine reduced blocks:

```text
matrix_lambda, matrix_mu,
fiber_C_TT, fiber_C_TT_cross, fiber_C_LT, fiber_C_LL, fiber_G_LT
```

This is an affine stiffness-coefficient representation of the physical input
space `(Em, nu_m, Ef_L, Ef_T, G_LT, nu_LT, nu_TT)`.

Greedy enrichment on the eight full-order material points:

| Snapshot material IDs | r | Mean rel. error | P95 rel. error | Max rel. error | Worst material | Median online time |
|---|---:|---:|---:|---:|---:|---:|
| `[0]` | 6 | 3.247e-02 | 8.002e-02 | 9.467e-02 | 1 | 2.28e-05 s |
| `[0, 1]` | 12 | 1.430e-02 | 4.695e-02 | 5.267e-02 | 3 | 2.86e-05 s |
| `[0, 1, 3]` | 18 | 2.123e-03 | 7.359e-03 | 1.002e-02 | 5 | 5.22e-05 s |
| `[0, 1, 3, 5]` | 24 | 6.873e-04 | 1.952e-03 | 2.112e-03 | 7 | 6.47e-05 s |
| `[0, 1, 3, 5, 7]` | 30 | 2.910e-04 | 9.745e-04 | 1.115e-03 | 2 | 7.54e-05 s |
| `[0, 1, 2, 3, 5, 7]` | 36 | 5.451e-05 | 2.550e-04 | 3.420e-04 | 6 | 1.01e-04 s |
| `[0, 1, 2, 3, 5, 7, 1013]` | 42 | 2.251e-05 | 9.331e-05 | 1.017e-04 | 6 | 4.26e-03 s |
| `[0, 1, 2, 3, 5, 7, 1003, 1013]` | 48 | 1.766e-05 | 7.148e-05 | 7.423e-05 | 4 | 1.49e-04 s |
| `[0, 1, 2, 3, 5, 7, 1003, 1013, 1015]` | 54 | 1.437e-05 | 5.760e-05 | 5.866e-05 | 6 | 1.73e-04 s |
| `[0, 1, 2, 3, 5, 7, 1003, 1013, 1015, 3001]` | 60 | 1.263e-05 | 5.267e-05 | 5.840e-05 | 6 | 2.04e-04 s |
| `[0, 1, 2, 3, 5, 7, 1003, 1013, 1015, 3001, 4030]` | 66 | 1.243e-05 | 5.199e-05 | 5.803e-05 | 6 | 2.28e-04 s |

The center material is reproduced by the initial `r=6` operator with relative
tensor error `8.6e-08`, which verifies the sign convention and Mandel/TI-axis
bookkeeping. The base-set `r=36` operator is below `0.1%` maximum relative
tensor error on the initial eight-point material set, and the current enriched
pilot operator at that stage was `r=66` before the later 32-point derivative
enrichment loop.

The reduced tensors stayed positive definite. The minimum value of
`eig(Crom - Cfom)` over the nested runs was about `-2.4e-06`, consistent with
single-precision FFTHomPy tolerance in the enriched training points.

## Independent Validation

The `r=36` ROM was then tested on 16 independent Sobol material points over the
same material domain and the same fixed geometry:

```bash
./scripts/fixed_geometry_validate_rom.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --rom-dir results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/rom_tangential_r36_center_m1_m2_m3_m5_m7 \
  --material-points 16 \
  --material-seed 20260923 \
  --out-name independent_validation_r36_sobol16_seed20260923
```

This first independent gate showed that the base-set `r=36` ROM was not yet
domain-certified: the maximum error was `9.724e-03` at validation material `13`.
Two adaptive enrichments were then performed by adding validation materials
`13` and `3` as stored snapshots `1013` and `1003`.

| ROM | Snapshot IDs added beyond base r36 | Independent mean error | Independent P95 error | Independent max error | Worst validation ID |
|---|---|---:|---:|---:|---:|
| `r36` | none | 1.513e-03 | 6.209e-03 | 9.724e-03 | 13 |
| `r42` | `1013` | 3.726e-04 | 1.058e-03 | 1.247e-03 | 3 |
| `r48` | `1013, 1003` | 2.261e-04 | 7.226e-04 | 9.803e-04 | 15 |
| `r54` | `1013, 1003, 1015` | 1.065e-04 | 2.321e-04 | 3.168e-04 | 9 |
| `r60` | `1013, 1003, 1015, 3001` | 7.577e-05 | 1.959e-04 | 2.532e-04 | 9 |

The `r48` operator was the first model below `0.1%` maximum relative tensor
error on this 16-point independent validation set. The `r54` and `r60`
operators reduced that sampled maximum further to `3.168e-04` and `2.532e-04`.
This is a strong pilot result for the fixed-geometry workflow, but it is still
an empirical sample check rather than a rigorous certificate over the continuous
material domain.

## QoI/Energy Indicator Check

The first production-style residual indicator over the fixed FFTHomPy geometry
is implemented in:

```bash
./scripts/fixed_geometry_qoi_indicator.py
```

It requires a ROM compiled with `--save-basis-fields`, because residual
evaluation reconstructs the six reduced strain fields, applies the affine local
stiffness on the fixed voxel geometry, projects the resulting stress residual
onto compatible strain modes and scores

```text
eta_C = || R^T M^{-1} R ||_F / || C_r ||_F
```

where `M^{-1}` is a homogeneous isotropic reference inverse. On the 16
independent validation materials, using the `r48` basis, the indicator ranked
material `15` first:

| Quantity | Value |
|---|---:|
| Spearman energy indicator vs true error | 0.971 |
| Spearman field residual vs true error | 0.968 |
| Top indicator material | 15 |
| True worst material | 15 |
| Median indicator time per candidate, Fourier path | 2.62 s |
| Legacy physical-space median time per candidate | 5.40 s |

Adding that indicator-selected material as snapshot `1015` produced the `r54`
operator above and reduced the independent validation maximum from `9.803e-04`
to `3.168e-04`.

The same indicator was then scaled from the 16 validation materials to a fresh
64-point Sobol candidate pool without full-order tensors. Using the `r54` basis,
the top candidates were:

| Energy rank | Candidate ID | Stored snapshot ID | eta_C | Em | Ef_L | Ef_T | G_LT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 3001 | 5.733e-03 | 3.106662 | 348.844956 | 7.709963 | 10.481410 |
| 2 | 60 | - | 5.353e-03 | 1.828477 | 325.871625 | 8.457922 | 8.195496 |
| 3 | 4 | - | 4.713e-03 | 1.272152 | 365.964830 | 11.341434 | 25.556453 |

Only candidate `1` was solved with FFTHomPy/CuPy on the same fixed geometry and
stored as `material_3001`. The `r54` ROM had relative tensor error `1.526e-03`
at this indicator-selected point; after adding its six tangential fields, the
`r60` ROM reproduced it with error `9.54e-08`.

## Larger Validation And Second Indicator Step

The `r60` ROM was then validated on a new independent 32-point Sobol material
set. This larger gate found one remaining gap:

| ROM | Validation points | Mean error | P95 error | Max error | Worst ID |
|---|---:|---:|---:|---:|---:|
| `r60` | 32 | 1.324e-04 | 3.469e-04 | 1.110e-03 | 30 |

The QoI/energy indicator was evaluated on those 32 points using the saved `r60`
basis. It ranked the true worst point first:

| Quantity | Value |
|---|---:|
| Spearman energy indicator vs true error | 0.920 |
| Spearman field residual vs true error | 0.944 |
| Top indicator material | 30 |
| True worst material | 30 |
| Energy indicator at material 30 | 3.260e-03 |

Solving only material `30` with FFTHomPy/CuPy and storing it as snapshot `4030`
produced the `r66` operator. Reusing the same 32 full-order validation tensors:

| ROM | Validation points | Mean error | P95 error | Max error | Worst ID |
|---|---:|---:|---:|---:|---:|
| `r66` | 32 | 8.579e-05 | 2.333e-04 | 3.650e-04 | 7 |

This second adaptive step is important evidence that the indicator can find
previously unseen material-space gaps and close them with one additional
six-load FFTHomPy solve.

## POD/RB Comparator

A first conventional POD/RB baseline is generated by:

```bash
./scripts/fixed_geometry_pod_baseline.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --out-name pod_baseline_final_snapshots_validation32 \
  --overwrite
```

This is a strong POD comparator: all final tangential snapshot fields from
materials `[0, 1, 2, 3, 5, 7, 1003, 1013, 1015, 3001, 4030]` are visible before
the POD truncation is chosen. Therefore its low-rank points are not
equal-offline-budget adaptive runs; they are an optimistic conventional
snapshot/POD-RB baseline using the final 66-field library.

On the same 32 independent validation materials:

| Rank | POD max error | Tangential max error | POD snapshot energy |
|---:|---:|---:|---:|
| 6 | 1.117e-01 | 1.004e-01 | 0.950296 |
| 12 | 3.013e-02 | 8.787e-02 | 0.980750 |
| 18 | 1.978e-02 | 2.945e-02 | 0.995258 |
| 24 | 4.746e-03 | 1.312e-02 | 0.998638 |
| 36 | 1.774e-03 | 6.858e-03 | 0.999622 |
| 48 | 1.021e-03 | 3.028e-03 | 0.999900 |
| 54 | 7.659e-04 | 1.722e-03 | 0.999954 |
| 60 | 6.813e-04 | 1.110e-03 | 0.999991 |
| 66 | 3.650e-04 | 3.650e-04 | 1.000000 |

The full `r=66` POD and tangential spaces give the same validation errors
because they span the same final snapshot space. At intermediate ranks the POD
baseline is often better in maximum error because it has seen all final
snapshots before truncation.

The most important methodological lesson is different: POD snapshot energy is
not a reliable certificate for the homogenized tensor. The first six POD modes
already capture `95.03%` of snapshot energy but still give order-10% maximum
tensor error. Even at `r=18`, with `99.53%` snapshot energy, the max tensor
error remains `1.98e-02`. This supports the paper's goal-oriented position:
adaptive control should be based on effective-stiffness error indicators, not
only on global field energy.

## Sensitivity Check And r72 Derivative Enrichment

Analytic ROM sensitivities are checked with:

```bash
./scripts/fixed_geometry_sensitivity_check.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --rom-dir results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/rom_tangential_r66_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v30 \
  --material-ids 0 \
  --out-name sensitivity_r66_center_fom_fd \
  --overwrite --run-full-order-fd --fom-rel-step 5e-3
```

The same script can read the independent validation full-order CSV and compare
the ROM sensitivities against FFTHomPy/CuPy finite differences at a selected
validation material:

```bash
./scripts/fixed_geometry_sensitivity_check.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --rom-dir results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/rom_tangential_r66_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v30 \
  --material-csv results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/independent_validation_r66_sobol32_seed20261111_reuse/validation_full_order_results.csv \
  --material-ids 7 \
  --out-name sensitivity_r66_validation7_fom_fd \
  --overwrite --run-full-order-fd --fom-rel-step 5e-3
```

This exposed a useful paper-level nuance. The `r66` ROM already had maximum
tensor error `3.650e-04` on the 32-point independent validation set, but the
worst tensor point, validation material `7`, was also poor for sensitivities.
The seven physical-parameter sensitivity errors against full-order finite
differences were:

| Parameter | r66 validation 7 | r72 validation 7 |
|---|---:|---:|
| `Em` | 6.393e-05 | 2.453e-05 |
| `nu_m` | 7.614e-04 | 2.227e-04 |
| `Ef_L` | 5.467e-03 | 7.510e-05 |
| `Ef_T` | 8.477e-03 | 3.155e-04 |
| `G_LT` | 2.907e-02 | 2.822e-04 |
| `nu_LT` | 1.097e-02 | 4.569e-04 |
| `nu_TT` | 1.353e-02 | 2.096e-04 |

Validation material `7` was therefore collected as one more tangential snapshot
block, stored as `material_4007`:

```bash
./scripts/fixed_geometry_collect_tangential.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --results-csv results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/independent_validation_r66_sobol32_seed20261111_reuse/validation_full_order_results.csv \
  --material-ids 7 --material-id-offset 4000
```

Then the `r72` operator was compiled and validated on the same 32 full-order
tensors:

```bash
./scripts/fixed_geometry_reduced_operator.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --snapshot-material-ids 0 1 2 3 5 7 1003 1013 1015 3001 4007 4030 \
  --out-name rom_tangential_r72_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v7_v30

./scripts/fixed_geometry_validate_rom.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --rom-dir results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/rom_tangential_r72_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v7_v30 \
  --full-order-results-csv results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/independent_validation_r66_sobol32_seed20261111_reuse/validation_full_order_results.csv \
  --out-name independent_validation_r72_sobol32_seed20261111_reuse
```

Summary:

| Check | Mean error | Median error | Max error | Worst ID |
|---|---:|---:|---:|---:|
| `r66` center sensitivity vs FOM FD | 4.353e-04 | 1.550e-04 | 1.651e-03 | - |
| `r66` validation-7 sensitivity vs FOM FD | 9.762e-03 | 8.477e-03 | 2.907e-02 | 7 |
| `r72` validation-7 sensitivity vs FOM FD | 2.267e-04 | 2.227e-04 | 4.569e-04 | 7 |
| `r72` 32-point tensor validation | 6.365e-05 | 5.266e-05 | 2.122e-04 | 9 |

The interpretation is important: tensor-error greedy can miss derivative QoIs.
Adding one derivative-problematic material point improved both the tensor
validation set and the sensitivity agreement at that point.

## Mixed Tensor/Sensitivity Indicator And Adaptive Loop

`fixed_geometry_qoi_indicator.py` now supports a mixed score:

```bash
./scripts/fixed_geometry_qoi_indicator.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --rom-dir results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/rom_tangential_r66_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v30_basis \
  --candidate-results-csv results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/independent_validation_r66_sobol32_seed20261111_reuse/validation_full_order_results.csv \
  --score-mode mixed \
  --out-name mixed_indicator_r66_validation32 \
  --overwrite --top-k 10
```

The tensor part is the existing Fourier-domain residual correction
`eta_C = ||R^T M^{-1}R||_F / ||C_r||_F`. The sensitivity part perturbs each of
the seven physical material parameters inside the admissible domain and scores
the finite-difference derivative of the same residual-correction matrix. The
default mixed score is

```text
sqrt(eta_C^2 + eta_S^2)
```

with `eta_S` aggregated as the maximum domain-scaled derivative over
`(Em, nu_m, Ef_L, Ef_T, G_LT, nu_LT, nu_TT)`.

On the same 32 validation materials with `r66`, the mixed indicator selected
the derivative-problematic point directly:

| Rank | Material | Mixed score | eta_C | eta_S | True tensor error |
|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 1.135e-02 | 1.262e-03 | 1.128e-02 | 3.650e-04 |
| 2 | 9 | 6.992e-03 | 1.098e-03 | 6.905e-03 | 2.642e-04 |
| 3 | 1 | 5.898e-03 | 5.962e-04 | 5.868e-03 | 1.667e-04 |
| 4 | 6 | 3.040e-03 | 5.622e-04 | 2.988e-03 | 2.077e-04 |

For tensor error alone, Spearman correlation was `0.829` for the mixed score,
`0.880` for `eta_C`, and `0.909` for the field residual on this 32-point set.
That is acceptable but not the main point: `eta_S` is a derivative-risk signal,
not a replacement for the tensor-error estimator.

After adding material `7`, the `r72` mixed subset check showed the signal was
closed at that point:

| ROM | Material | eta_C | eta_S | Mixed score | True tensor error |
|---|---:|---:|---:|---:|---:|
| `r66` | 7 | 1.262e-03 | 1.128e-02 | 1.135e-02 | 3.650e-04 |
| `r72` | 7 | 9.476e-09 | 4.811e-07 | 4.811e-07 | 8.537e-08 |
| `r72` | 9 | 8.241e-04 | 5.673e-03 | 5.733e-03 | 2.122e-04 |

The next mixed-indicator point after `r72` was therefore validation material
`9`. Full-order finite-difference sensitivities confirmed that this was a real
derivative gap:

| Check | Mean sensitivity error | Median sensitivity error | Max sensitivity error |
|---|---:|---:|---:|
| `r72` validation 9 vs FOM FD | 1.163e-02 | 5.610e-03 | 4.723e-02 |
| `r78` validation 9 vs FOM FD | 8.077e-04 | 7.773e-04 | 1.406e-03 |

Material `9` was collected as `material_4009` and compiled into:

```bash
./scripts/fixed_geometry_reduced_operator.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --snapshot-material-ids 0 1 2 3 5 7 1003 1013 1015 3001 4007 4009 4030 \
  --out-name rom_tangential_r78_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v7_v9_v30_basis \
  --save-basis-fields
```

On the same 32 full-order validation tensors:

| ROM | Validation points | Mean error | P95 error | Max error | Worst ID |
|---|---:|---:|---:|---:|---:|
| `r66` | 32 | 8.579e-05 | 2.333e-04 | 3.650e-04 | 7 |
| `r72` | 32 | 6.365e-05 | 1.762e-04 | 2.122e-04 | 9 |
| `r78` | 32 | 4.763e-05 | 1.239e-04 | 1.790e-04 | 11 |
| `r84` | 32 | 3.726e-05 | 9.334e-05 | 1.228e-04 | 1 |
| `r90` | 32 | 2.821e-05 | 7.081e-05 | 9.671e-05 | 31 |
| `r96` | 32 | 2.219e-05 | 5.374e-05 | 7.181e-05 | 6 |
| `r102` | 32 | 1.749e-05 | 4.772e-05 | 5.258e-05 | 10 |
| `r108` | 32 | 1.175e-05 | 3.157e-05 | 3.548e-05 | 17 |
| `r114` | 32 | 9.305e-06 | 2.318e-05 | 2.758e-05 | 12 |
| `r120` | 32 | 7.943e-06 | 2.037e-05 | 2.314e-05 | 23 |
| `r126` | 32 | 5.662e-06 | 1.561e-05 | 1.742e-05 | 8 |
| `r132` | 32 | 4.199e-06 | 1.238e-05 | 1.516e-05 | 25 |
| `r138` | 32 | 2.839e-06 | 9.346e-06 | 1.246e-05 | 13 |
| `r144` | 32 | 2.135e-06 | 7.557e-06 | 8.992e-06 | 16 |
| `r150` | 32 | 1.816e-06 | 6.517e-06 | 8.452e-06 | 22 |
| `r156` | 32 | 1.378e-06 | 6.081e-06 | 6.270e-06 | 24 |
| `r162` | 32 | 9.692e-07 | 4.018e-06 | 5.607e-06 | 14 |
| `r168` | 32 | 6.502e-07 | 1.730e-06 | 5.203e-06 | 20 |

The r78 subset check confirms that material `9` is now also closed by the
indicator:

| ROM | Material | eta_C | eta_S | Mixed score |
|---|---:|---:|---:|---:|
| `r72` | 9 | 8.241e-04 | 5.673e-03 | 5.733e-03 |
| `r78` | 9 | 2.832e-08 | 2.394e-06 | 2.394e-06 |
| `r78` | 11 | 2.343e-04 | 1.219e-03 | 1.242e-03 |

This is strong evidence for a multi-QoI adaptive loop: the indicator selected
material `7`, enrichment fixed material `7`; then it selected material `9`,
enrichment fixed material `9`.

The same loop then continued:

| Step | Selected material | Pre-enrichment sensitivity mean | Pre max | Post-enrichment sensitivity mean | Post max |
|---|---:|---:|---:|---:|---:|
| `r78 -> r84` | 11 | 6.472e-03 | 2.264e-02 | 4.487e-04 | 1.317e-03 |
| `r84 -> r90` | 1 | 3.529e-03 | 1.110e-02 | 6.020e-04 | 1.220e-03 |
| `r90 -> r96` | 31 | 1.952e-03 | 3.855e-03 | 2.285e-04 | 8.025e-04 |
| `r96 -> r102` | 6 | 4.083e-03 | 2.030e-02 | 5.524e-04 | 2.368e-03 |
| `r102 -> r108` | 10 | 2.716e-03 | 1.393e-02 | 1.945e-04 | 5.141e-04 |
| `r108 -> r114` | 17 | 2.406e-03 | 1.033e-02 | 6.776e-04 | 1.692e-03 |
| `r114 -> r120` | 12 | 1.078e-03 | 2.039e-03 | 8.720e-04 | 2.221e-03 |
| `r120 -> r126` | 23 | 9.948e-04 | 3.701e-03 | 2.664e-04 | 1.176e-03 |
| `r126 -> r132` | 8 | 1.003e-03 | 2.850e-03 | 3.362e-04 | 1.617e-03 |
| `r132 -> r138` | 25 | 1.654e-03 | 8.806e-03 | 3.531e-04 | 5.404e-04 |
| `r138 -> r144` | 13 | 6.395e-04 | 1.625e-03 | 1.840e-04 | 6.445e-04 |
| `r144 -> r150` | 16 | boundary-limited central FD | 3.368e-02 | 4.465e-04 | 1.676e-03 |
| `r150 -> r156` | 22 | 9.717e-04 | 2.206e-03 | 6.594e-04 | 1.868e-03 |
| `r156 -> r162` | 24 | 6.851e-04 | 2.314e-03 | 4.621e-04 | 2.354e-03 |
| `r162 -> r168` | 14 | 6.909e-04 | 2.251e-03 | 6.560e-04 | 2.282e-03 |

The material-16 row is deliberately marked as boundary-limited. That point has
`nu_LT` very close to the upper admissible bound, so the historical central
full-order finite-difference stencil produced a misleading spike. The
sensitivity checker now supports `--fom-fd-scheme edge-auto`, using central
differences where possible and second-order one-sided inward differences near
domain bounds. With that scheme, the r150 material-16 comparison is mean
`4.465e-04`, median `1.138e-04`, maximum `1.676e-03`.

The indicator subset checks also closed the selected points and exposed the
next hotspot:

| ROM | Material | eta_C | eta_S | Mixed score |
|---|---:|---:|---:|---:|
| `r78` | 11 | 2.343e-04 | 1.219e-03 | 1.242e-03 |
| `r84` | 11 | 8.640e-09 | 6.070e-07 | 6.070e-07 |
| `r84` | 1 | 3.864e-04 | 4.429e-03 | 4.446e-03 |
| `r90` | 1 | 3.336e-08 | 5.611e-07 | 5.621e-07 |
| `r90` | 31 | 9.619e-05 | 1.155e-03 | 1.159e-03 |
| `r96` | 31 | 5.470e-09 | 1.751e-07 | 1.752e-07 |
| `r96` | 6 | 1.477e-04 | 8.778e-04 | 8.902e-04 |
| `r102` | 6 | 2.081e-08 | 1.148e-06 | 1.148e-06 |
| `r102` | 10 | 1.072e-04 | 5.697e-04 | 5.797e-04 |
| `r108` | 10 | 1.655e-08 | 3.022e-07 | 3.027e-07 |
| `r108` | 17 | 7.413e-05 | 5.000e-04 | 5.055e-04 |
| `r114` | 17 | 2.612e-08 | 1.146e-06 | 1.146e-06 |
| `r114` | 12 | 4.645e-05 | 1.520e-04 | 1.589e-04 |
| `r120` | 12 | 2.087e-08 | 6.477e-07 | 6.481e-07 |
| `r120` | 23 | 4.642e-05 | 1.224e-04 | 1.309e-04 |
| `r126` | 23 | 1.427e-08 | 3.132e-07 | 3.136e-07 |
| `r126` | 8 | 3.683e-05 | 1.193e-04 | 1.249e-04 |
| `r132` | 8 | 9.491e-09 | 2.177e-07 | 2.179e-07 |
| `r132` | 25 | 4.228e-05 | 2.545e-04 | 2.580e-04 |
| `r138` | 25 | 4.318e-08 | 5.323e-07 | 5.340e-07 |
| `r138` | 13 | 2.442e-05 | 7.291e-05 | 7.689e-05 |
| `r144` | 13 | 2.203e-08 | 1.601e-07 | 1.616e-07 |
| `r144` | 16 | 1.957e-05 | 1.351e-04 | 1.365e-04 |
| `r150` | 16 | 7.014e-08 | 7.410e-07 | 7.443e-07 |
| `r150` | 22 | 2.207e-05 | 1.232e-04 | 1.252e-04 |
| `r156` | 22 | 8.260e-08 | 7.197e-07 | 7.245e-07 |
| `r156` | 24 | 9.066e-06 | 2.987e-05 | 3.121e-05 |
| `r162` | 24 | 3.343e-08 | 2.358e-07 | 2.382e-07 |
| `r162` | 14 | 2.230e-05 | 7.125e-05 | 7.466e-05 |
| `r168` | 14 | 1.938e-07 | 1.185e-06 | 1.201e-06 |
| `r168` | 20 | 7.560e-06 | 4.224e-05 | 4.291e-05 |

The current empirical milestone is `r168`: the independent 32-point tensor
maximum is `5.203e-06`, and the last seventeen mixed-indicator selected points
(`7`, `9`, `11`, `1`, `31`, `6`, `10`, `17`, `12`, `23`, `8`, `25`, `13`,
`16`, `22`, `24`, `14`) all dropped to near-zero indicator values after their
six-load enrichment. Material `20` is the next visible hotspot from the r168
subset check; its r168 full-order sensitivity check gives mean `9.407e-04`,
median `1.999e-04`, maximum `5.406e-03`, with boundary-limited `nu_m`.
The implementation is still expensive
because the sensitivity score performs 14 extra residual-indicator evaluations
per candidate. It should be optimized or analytically differentiated before
using large candidate pools.

A later 128-point independent validation exposed a stricter tensor hotspot for
the same `r168` ROM. The 128-point run gives mean tensor error `3.866e-06`,
P95 `1.477e-05`, maximum `4.147e-05`, and worst material `94`; eight of the 128
points exceed `1e-05`. Material `94` is an extreme-contrast point
(`Ef_L/Em=342.67`, `G_LT/Gm=43.93`) and should be the next targeted enrichment
if the fixed-geometry loop continues.

## Stop Policy

The adaptive loop should not keep adding material points after the configured
scientific tolerance is already met. A reproducible stop/go gate is implemented
in:

```bash
./scripts/fixed_geometry_stop_policy.py
```

The default pilot policy uses:

| Quantity | Stop target |
|---|---:|
| Mean validation tensor error | `3e-06` |
| P95 validation tensor error | `1e-05` |
| Max validation tensor error | `1e-05` |
| Max energy indicator | `1e-04` |
| Max sensitivity indicator | `1e-04` |
| Max mixed indicator | `1e-04` |
| Mean full-order sensitivity FD error | `1e-03` |
| Max full-order sensitivity FD error | `1e-02` |

For the current r168 state:

```bash
python scripts/fixed_geometry_stop_policy.py \
  --validation results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/independent_validation_r168_sobol32_seed20261111_reuse \
  --indicator results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/mixed_indicator_r168_validation_subset_m14_m20 \
  --sensitivity results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/sensitivity_r168_validation20_fom_fd_edge_auto \
  --require-indicator --require-sensitivity
```

returns:

```text
[STOP-POLICY] decision=STOP | r=168 | worst=20 | failures=0
```

The generated summary is:

```text
independent_validation_r168_sobol32_seed20261111_reuse/stop_policy_summary.md
```

This means material `20` should not be collected automatically under the
default paper-pilot tolerance. It should be solved only if a new validation set
fails, if a larger candidate set exposes a higher indicator, or if the
manuscript intentionally adopts a stricter derivative target.

The larger 128-point validation is such a new validation-set failure:

```bash
python scripts/fixed_geometry_validate_rom.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --rom-dir results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/rom_tangential_r168_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v6_v7_v8_v9_v10_v11_v12_v13_v14_v16_v17_v22_v23_v24_v25_v30_v31_basis \
  --material-points 128 \
  --material-seed 20261217 \
  --out-name independent_validation_r168_sobol128_seed20261217_gpu \
  --geometry-backend numba \
  --solver-tol 1e-3 \
  --no-runtime-reexec
```

The first sandboxed attempt created
`independent_validation_r168_sobol128_seed20261217` with only the material-point
design files. The authoritative result is
`independent_validation_r168_sobol128_seed20261217_gpu`.

| Check | 128-point value | Target | Status |
|---|---:|---:|---|
| Mean validation tensor error | 3.866e-06 | 3.000e-06 | fail |
| P95 validation tensor error | 1.477e-05 | 1.000e-05 | fail |
| Max validation tensor error | 4.147e-05 | 1.000e-05 | fail |

Top tensor-error points:

| Rank | Material | Relative tensor error |
|---:|---:|---:|
| 1 | 94 | 4.147e-05 |
| 2 | 29 | 4.032e-05 |
| 3 | 30 | 3.169e-05 |
| 4 | 46 | 2.980e-05 |
| 5 | 71 | 1.988e-05 |

If the fixed-geometry accuracy loop is resumed, the next targeted enrichment
would be validation material `94` stored as `material_4094`, followed by
compiling `r174` and revalidating against the already computed 128 full-order
tensors with `--full-order-results-csv`. Per the current project decision, this
enrichment is paused. Material `94` remains the recorded optional hotspot,
while the active work moves to non-enrichment paper stages.

## Material Screening Demo

The first many-query application step is implemented in:

```bash
python scripts/fixed_geometry_material_screening.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --rom-dir results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/rom_tangential_r168_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v6_v7_v8_v9_v10_v11_v12_v13_v14_v16_v17_v22_v23_v24_v25_v30_v31_basis \
  --material-points 4096 \
  --material-seed 20270117 \
  --objective balanced_gain
```

Authoritative output:

```text
material_screening_r168_sobol4096_seed20270117
```

This stage queries only `reduced_operators.npz`; it runs zero FFTHomPy/CuPy
full-order solves and adds no tangential snapshots.

| Quantity | Value |
|---|---:|
| Candidate material queries | 4096 |
| Eligible candidates | 4096 |
| Pareto-front candidates | 64 |
| Wall time | 7.791 s |
| Median ROM query time | 1.024e-03 s |
| Best balanced candidate | 413 |
| Best balanced gain | 1.553 |

The best balanced candidate is:

| Parameter | Value |
|---|---:|
| `Em` | 4.287314 |
| `nu_m` | 0.351007 |
| `Ef_L` | 383.581319 |
| `Ef_T` | 22.051695 |
| `G_LT` | 25.860555 |
| `nu_LT` | 0.254861 |
| `nu_TT` | 0.390380 |

Its ROM-predicted engineering constants include `E1=10.4596`, `E2=10.4555`,
`E3=11.4120`, and `G_min=4.0286`. This gives the paper a concrete application
line:

```text
fixed G -> compiled ROM -> thousands of material candidates -> ranked designs
```

## Equal-Budget POD/RB Comparator

The strict fair-budget POD/RB report is generated by:

```bash
python scripts/fixed_geometry_pod_budget_report.py
```

Authoritative output:

```text
pod_equal_offline_budget_r66_validation32
```

The report reuses the existing strong POD baseline and runs zero new
FFTHomPy/CuPy solves. The key distinction is budget accounting: the low-rank
POD curve sees all final 66 snapshot fields before truncation, so those points
are oracle diagnostics rather than equal-offline-budget adaptive runs.

At the strict equal-offline and equal-online point:

| Quantity | Value |
|---|---:|
| Snapshot material IDs | `[0, 1, 2, 3, 5, 7, 1003, 1013, 1015, 3001, 4030]` |
| Offline snapshot fields | 66 |
| Equal online rank | 66 |
| POD mean relative tensor error | 8.579e-05 |
| Tangential mean relative tensor error | 8.579e-05 |
| POD maximum relative tensor error | 3.650e-04 |
| Tangential maximum relative tensor error | 3.650e-04 |
| Max-error ratio POD/tangential | 1.000000 |

The fair conclusion is that full-rank POD and tangential RB tie when they have
the same final snapshot space. The lower-rank POD rows remain useful for a
different point: snapshot energy does not certify \(C^{\mathrm{eff}}\). The
first six POD modes capture `95.03%` of snapshot energy but still have max
tensor error `1.117e-01`; at `r=18`, `99.53%` energy still leaves max tensor
error `1.978e-02`.

## Inverse Identification Demo

The first inverse-identification application is generated by:

```bash
python scripts/fixed_geometry_inverse_identification.py
```

Authoritative output:

```text
inverse_identification_r168_validation128_m94
```

This stage uses the `r168` reduced operators and analytic affine ROM
sensitivities, chained with the physical-to-affine coefficient Jacobian. It
fits the FFTHomPy/CuPy tensor of validation material `94` using zero new
full-order solves.

| Quantity | Value |
|---|---:|
| Target material | 94 |
| Multistarts | 8 |
| Best start | `sobol_02` |
| Initial residual norm | 8.787e-01 |
| Final tensor fit relative error | 1.256e-06 |
| ROM mismatch at true target | 4.147e-05 |
| Function evaluations | 43 |
| Best-start wall time | 0.218 s |
| Maximum parameter relative error | 1.247e-03 |

Recovered parameters:

| Parameter | Target | Recovered | Relative error |
|---|---:|---:|---:|
| `Em` | 1.141788 | 1.141794 | 5.48e-06 |
| `nu_m` | 0.350962 | 0.350957 | 1.54e-05 |
| `Ef_L` | 391.256075 | 391.222874 | 8.49e-05 |
| `Ef_T` | 7.105549 | 7.104378 | 1.65e-04 |
| `G_LT` | 18.562782 | 18.555679 | 3.83e-04 |
| `nu_LT` | 0.216863 | 0.217133 | 1.25e-03 |
| `nu_TT` | 0.351779 | 0.351882 | 2.92e-04 |

The tensor fit is below the ROM-vs-FOM mismatch at the true target, so this is a
clean sensitivity-enabled ROM application for the current fixed geometry.

## Indicator Effectivity

The empirical effectivity study is generated by:

```bash
python scripts/fixed_geometry_indicator_effectivity.py
```

Authoritative output:

```text
indicator_effectivity_fixed_geometry
```

It computes

```text
effectivity = indicator_rel / true_relative_tensor_error
```

on saved indicator rankings with finite full-order error. It runs zero new
FFTHomPy/CuPy solves.

| Indicator | Samples | Runs | Min | Median | P95 | Max | Underestimates | Conservative fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| energy | 86 | 6 | 6.90e-02 | 1.96e+00 | 3.59e+00 | 4.16e+00 | 9 | 0.895 |
| field residual | 86 | 6 | 1.66e+01 | 6.55e+01 | 3.24e+02 | 7.57e+02 | 0 | 1.000 |
| mixed | 38 | 4 | 2.40e+00 | 1.06e+01 | 3.15e+01 | 3.53e+01 | 0 | 1.000 |
| sensitivity | 38 | 4 | 2.40e+00 | 1.05e+01 | 3.13e+01 | 3.51e+01 | 0 | 1.000 |

The energy indicator is a useful ranking signal, but it is not a certified
upper bound on the saved samples because it underestimates the true tensor error
in 9 of 86 comparisons. The field residual is conservative but loose. Mixed and
sensitivity indicators are conservative on the saved comparisons, but should
still be reported as empirical effectivity until the exact Ritz/Schur
complement error bound is derived.

## Ritz/Schur Complement Bound Derivation

The paper-ready derivation is drafted in:

```text
docs/ritz_schur_error_bound.md
```

For the full-order Schur complement

```text
Ceff = D - B^T K^{-1} B
```

and the Ritz ROM

```text
Cr = D - Br^T Kr^{-1} Br,
```

the residual matrix \(R=B-KX_r\) satisfies the exact identity

\[
C_r^{\mathrm{eff}}-C_{\mathrm{FOM}}^{\mathrm{eff}}
=
R^T K^{-1}R
\succeq0.
\]

If the homogeneous reference operator \(M\) satisfies

\[
K(\xi)\succeq \beta(\xi)M(\xi),
\]

then

\[
0\preceq
C_r^{\mathrm{eff}}-C_{\mathrm{FOM}}^{\mathrm{eff}}
\preceq
\beta(\xi)^{-1}R^T M^{-1}R.
\]

Thus the current energy indicator becomes certified only after multiplying by
the pointwise spectral-equivalence factor \(\beta^{-1}\). The current
`indicator_effectivity_fixed_geometry` table should therefore be read as
empirical effectivity; the next code step is to compute \(\beta(\xi)\) from
local generalized eigenvalues and regenerate a certified-bound table.

## Certified Bound Runner

The beta-scaled bound table is generated by:

```bash
python scripts/fixed_geometry_certified_bound.py --overwrite
```

Authoritative output:

```text
certified_bound_fixed_geometry
```

This runner reads saved `qoi_indicator_ranking.csv` files, computes
\(\beta(\xi)\) from local generalized eigenvalues against the same isotropic
reference stiffness used by the indicator, and writes both empirical and
certified effectivity. It runs zero FFTHomPy/CuPy full-order solves.

| Quantity | Value |
|---|---:|
| Indicator runs | 6 |
| Rows | 86 |
| Rows with absolute effectivity | 86 |
| beta min | 1.405e-01 |
| beta median | 2.558e-01 |
| beta max | 4.109e-01 |
| Min certified absolute effectivity | 3.173e-01 |
| Median certified absolute effectivity | 1.076e+01 |
| Strict certified absolute violations | 2 |
| Certified absolute violations above `1e-4` true relative error | 0 |

The two strict violations are below the numerical floor of this pilot:

| Run | Material | True relative error | Certified effectivity |
|---|---:|---:|---:|
| `qoi_indicator_r48_independent16_fourier` | 3 | 9.24e-08 | 3.17e-01 |
| `mixed_indicator_r66_validation32` | 30 | 3.64e-07 | 9.79e-01 |

This is the first certified-bound table for the fixed-geometry workflow. For
the manuscript, use `1e-4` as the scientific certification floor, report the
beta-scaled result as reliable above that scale, and keep the unscaled energy
indicator labelled as a ranking indicator.

## Cost And Break-Even

A reproducible timing report is generated by:

```bash
./scripts/fixed_geometry_cost_report.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --out-name cost_break_even_r66_fourier_indicator \
  --overwrite
```

For the current `r66` operator, validated on the 32-point independent set:

| Quantity | Median / value |
|---|---:|
| Full FFTHomPy query, no stored fields | 1.966e-01 s |
| ROM online query | 2.259e-04 s |
| Full/ROM median query speedup | 870x |
| Tangential FFTHomPy solve storing six fields | 2.287 s |
| QoI indicator per candidate, Fourier CPU path | 2.659 s |
| Final r66 operator assembly | 12.673 s |
| Final build excluding indicator scans | 38.050 s |
| Executed indicator scans | 255.197 s |
| Break-even excluding indicator scans | 194 material queries |
| Break-even including executed indicator scans | 1494 material queries |

This timing split is important. The reduced online solve is already far cheaper
than full FFTHomPy for repeated material queries. The Fourier-domain indicator
path removes one major CPU bottleneck while preserving the selected candidates
and indicator values to roundoff: on the executed r54/64 and r60/32 scans, the
total indicator time dropped from `534.105 s` to `255.197 s`. It is still a
NumPy/CPU implementation and remains the main adaptive-scan cost, so these
numbers should be treated as pilot implementation timings, not final manuscript
speed-up claims.

## Output Contract

Important files:

```text
run_manifest.json
run_summary.md
material_points.csv
fixed_geometry_ffthompy_results.csv
_fixed_geometry/geometry_manifest.json
_fixed_geometry/phase.npy
_fixed_geometry/ori.npy
_fixed_geometry/continuous_fibers_master.csv
material_*/Ceff.npy
material_*/solver_timing.json
material_0000/solution_fields.npz
material_4001/solution_fields.npz
material_4006/solution_fields.npz
material_4007/solution_fields.npz
material_4008/solution_fields.npz
material_4009/solution_fields.npz
material_4010/solution_fields.npz
material_4011/solution_fields.npz
material_4012/solution_fields.npz
material_4013/solution_fields.npz
material_4014/solution_fields.npz
material_4016/solution_fields.npz
material_4017/solution_fields.npz
material_4022/solution_fields.npz
material_4023/solution_fields.npz
material_4024/solution_fields.npz
material_4025/solution_fields.npz
material_4031/solution_fields.npz
rom_tangential_r*/reduced_operators.npz
rom_tangential_r*/rom_validation_results.csv
rom_tangential_r*/rom_manifest.json
rom_tangential_r72_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v7_v30/rom_manifest.json
rom_tangential_r78_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v7_v9_v30_basis/rom_manifest.json
rom_tangential_r84_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v7_v9_v11_v30_basis/rom_manifest.json
rom_tangential_r90_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v7_v9_v11_v30_basis/rom_manifest.json
rom_tangential_r96_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v7_v9_v11_v30_v31_basis/rom_manifest.json
rom_tangential_r102_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v6_v7_v9_v11_v30_v31_basis/rom_manifest.json
rom_tangential_r108_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v6_v7_v9_v10_v11_v30_v31_basis/rom_manifest.json
rom_tangential_r114_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v6_v7_v9_v10_v11_v17_v30_v31_basis/rom_manifest.json
rom_tangential_r120_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v6_v7_v9_v10_v11_v12_v17_v30_v31_basis/rom_manifest.json
rom_tangential_r126_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v6_v7_v9_v10_v11_v12_v17_v23_v30_v31_basis/rom_manifest.json
rom_tangential_r132_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v6_v7_v8_v9_v10_v11_v12_v17_v23_v30_v31_basis/rom_manifest.json
rom_tangential_r138_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v6_v7_v8_v9_v10_v11_v12_v17_v23_v25_v30_v31_basis/rom_manifest.json
rom_tangential_r144_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v6_v7_v8_v9_v10_v11_v12_v13_v17_v23_v25_v30_v31_basis/rom_manifest.json
rom_tangential_r150_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v6_v7_v8_v9_v10_v11_v12_v13_v16_v17_v23_v25_v30_v31_basis/rom_manifest.json
rom_tangential_r156_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v6_v7_v8_v9_v10_v11_v12_v13_v16_v17_v22_v23_v25_v30_v31_basis/rom_manifest.json
rom_tangential_r162_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v6_v7_v8_v9_v10_v11_v12_v13_v16_v17_v22_v23_v24_v25_v30_v31_basis/rom_manifest.json
rom_tangential_r168_center_m1_m2_m3_m5_m7_v13_v3_v15_qoi64_m1_v1_v6_v7_v8_v9_v10_v11_v12_v13_v14_v16_v17_v22_v23_v24_v25_v30_v31_basis/rom_manifest.json
independent_validation_r72_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r78_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r84_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r90_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r96_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r102_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r108_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r114_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r120_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r126_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r132_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r138_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r144_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r150_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r156_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r162_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r168_sobol32_seed20261111_reuse/validation_manifest.json
independent_validation_r168_sobol128_seed20261217_gpu/validation_manifest.json
mixed_indicator_r66_validation32/qoi_indicator_summary.json
mixed_indicator_r72_validation_subset_m7_m9/qoi_indicator_summary.json
mixed_indicator_r78_validation_subset_m9_m11/qoi_indicator_summary.json
mixed_indicator_r84_validation_subset_m11_m1/qoi_indicator_summary.json
mixed_indicator_r90_validation_subset_m1_m31/qoi_indicator_summary.json
mixed_indicator_r96_validation_subset_m31_m6/qoi_indicator_summary.json
mixed_indicator_r102_validation_subset_m6_m10/qoi_indicator_summary.json
mixed_indicator_r108_validation_subset_m10_m17/qoi_indicator_summary.json
mixed_indicator_r114_validation_subset_m17_m12/qoi_indicator_summary.json
mixed_indicator_r120_validation_subset_m12_m23/qoi_indicator_summary.json
mixed_indicator_r126_validation_subset_m23_m8/qoi_indicator_summary.json
mixed_indicator_r132_validation_subset_m8_m25/qoi_indicator_summary.json
mixed_indicator_r138_validation_subset_m25_m13/qoi_indicator_summary.json
mixed_indicator_r144_validation_subset_m13_m16/qoi_indicator_summary.json
mixed_indicator_r150_validation_subset_m16_m22/qoi_indicator_summary.json
mixed_indicator_r156_validation_subset_m22_m24/qoi_indicator_summary.json
mixed_indicator_r162_validation_subset_m24_m14/qoi_indicator_summary.json
mixed_indicator_r168_validation_subset_m14_m20/qoi_indicator_summary.json
sensitivity_r66_center_fom_fd/sensitivity_summary.json
sensitivity_r66_validation7_fom_fd/sensitivity_summary.json
sensitivity_r72_validation7_fom_fd/sensitivity_summary.json
sensitivity_r72_validation9_fom_fd/sensitivity_summary.json
sensitivity_r78_validation9_fom_fd/sensitivity_summary.json
sensitivity_r78_validation11_fom_fd/sensitivity_summary.json
sensitivity_r84_validation11_fom_fd/sensitivity_summary.json
sensitivity_r84_validation1_fom_fd/sensitivity_summary.json
sensitivity_r90_validation1_fom_fd/sensitivity_summary.json
sensitivity_r90_validation31_fom_fd/sensitivity_summary.json
sensitivity_r96_validation31_fom_fd/sensitivity_summary.json
sensitivity_r96_validation6_fom_fd/sensitivity_summary.json
sensitivity_r102_validation6_fom_fd/sensitivity_summary.json
sensitivity_r102_validation10_fom_fd/sensitivity_summary.json
sensitivity_r108_validation10_fom_fd/sensitivity_summary.json
sensitivity_r108_validation17_fom_fd/sensitivity_summary.json
sensitivity_r114_validation17_fom_fd/sensitivity_summary.json
sensitivity_r114_validation12_fom_fd/sensitivity_summary.json
sensitivity_r120_validation12_fom_fd/sensitivity_summary.json
sensitivity_r120_validation23_fom_fd/sensitivity_summary.json
sensitivity_r126_validation23_fom_fd/sensitivity_summary.json
sensitivity_r126_validation8_fom_fd/sensitivity_summary.json
sensitivity_r132_validation8_fom_fd/sensitivity_summary.json
sensitivity_r132_validation25_fom_fd/sensitivity_summary.json
sensitivity_r138_validation25_fom_fd/sensitivity_summary.json
sensitivity_r138_validation13_fom_fd/sensitivity_summary.json
sensitivity_r144_validation13_fom_fd/sensitivity_summary.json
sensitivity_r144_validation16_fom_fd/sensitivity_summary.json
sensitivity_r150_validation16_fom_fd/sensitivity_summary.json
sensitivity_r150_validation16_fom_fd_step2e2/sensitivity_summary.json
sensitivity_r150_validation16_fom_fd_edge_auto/sensitivity_summary.json
sensitivity_r150_validation22_fom_fd_edge_auto/sensitivity_summary.json
sensitivity_r156_validation22_fom_fd_edge_auto/sensitivity_summary.json
sensitivity_r156_validation24_fom_fd_edge_auto/sensitivity_summary.json
sensitivity_r162_validation24_fom_fd_edge_auto/sensitivity_summary.json
sensitivity_r162_validation14_fom_fd_edge_auto/sensitivity_summary.json
sensitivity_r168_validation14_fom_fd_edge_auto/sensitivity_summary.json
sensitivity_r168_validation20_fom_fd_edge_auto/sensitivity_summary.json
independent_validation_r168_sobol32_seed20261111_reuse/stop_policy_summary.json
independent_validation_r168_sobol32_seed20261111_reuse/stop_policy_summary.md
independent_validation_r168_sobol128_seed20261217_gpu/stop_policy_summary.json
independent_validation_r168_sobol128_seed20261217_gpu/stop_policy_summary.md
material_screening_r168_sobol4096_seed20270117/screening_summary.json
material_screening_r168_sobol4096_seed20270117/screening_summary.md
material_screening_r168_sobol4096_seed20270117/screening_candidates.csv
material_screening_r168_sobol4096_seed20270117/screening_top_candidates.csv
material_screening_r168_sobol4096_seed20270117/screening_pareto_front.csv
pod_equal_offline_budget_r66_validation32/pod_equal_budget_summary.json
pod_equal_offline_budget_r66_validation32/pod_equal_budget_summary.md
pod_equal_offline_budget_r66_validation32/pod_equal_budget_curve.csv
inverse_identification_r168_validation128_m94/inverse_identification_summary.json
inverse_identification_r168_validation128_m94/inverse_identification_summary.md
inverse_identification_r168_validation128_m94/inverse_multistart_results.csv
inverse_identification_r168_validation128_m94/inverse_parameter_recovery.csv
indicator_effectivity_fixed_geometry/indicator_effectivity_summary.json
indicator_effectivity_fixed_geometry/indicator_effectivity_summary.md
indicator_effectivity_fixed_geometry/indicator_effectivity_rows.csv
indicator_effectivity_fixed_geometry/indicator_effectivity_by_run.csv
certified_bound_fixed_geometry/certified_bound_summary.json
certified_bound_fixed_geometry/certified_bound_summary.md
certified_bound_fixed_geometry/certified_bound_rows.csv
certified_bound_fixed_geometry/certified_bound_by_run.csv
cost_break_even_r66/cost_break_even_summary.md
cost_break_even_r66_fourier_indicator/cost_break_even_summary.md
pod_baseline_final_snapshots_validation32/pod_baseline_summary.md
pod_baseline_final_snapshots_validation32/pod_vs_tangential_validation_curve.csv
```

The manifest explicitly records:

- `no_geometry_realizations = true`;
- `same_phase_ori_for_all_materials = true`;
- SHA256 hashes for the fixed `phase.npy` and `ori.npy`;
- the material domain bounds;
- FFTHomPy/CuPy runtime settings.

## Interpretation

This run establishes the first clean production-style full-order baseline for
the new paper:

```text
one fixed short-fiber G -> many FFTHomPy/CuPy Ceff(xi) solves
```

It deliberately separates geometric variability from constituent variability.
The next ROM work should therefore treat `_fixed_geometry/phase.npy` and
`_fixed_geometry/ori.npy` as immutable geometry data and spend additional
full-order solves only at selected material points. After the `r72` sensitivity
through `r168` sensitivity checks, those selected points should be allowed to
target either tensor error or derivative error, depending on the intended
application. Full-order finite differences near material-domain bounds should
use `--fom-fd-scheme edge-auto`; otherwise boundary-clipped central stencils can
overstate derivative disagreement. The 32-point gate marked `r168` as complete,
but the larger 128-point gate failed on tensor error. For now, enrichment is
paused and material `94` is only a recorded optional hotspot; the active path is
to build paper evidence that does not require new full-order solves.

## Next Technical Step

1. Do not collect more material snapshots for now; material `94` is recorded as
   an optional future hotspot, not the active task.
2. Use the material-screening, equal-budget POD/RB, inverse-identification and
   effectivity outputs as the fixed-geometry paper evidence bundle.
3. Use `docs/ritz_schur_error_bound.md` as the written mathematical section for
   the Ritz/Schur complement identity and reference-operator bound.
4. Use `certified_bound_fixed_geometry` as the beta-scaled bound table with
   `1e-4` as the paper-level certification floor; revisit sub-`1e-6` strict
   violations only if the manuscript later claims ultra-tight certification.
5. Optimize the mixed indicator beyond finite differences,
   especially candidate batching, precomputed residual actions and/or GPU
   evaluation; the current `eta_S` path costs roughly 15 tensor-indicator
   evaluations per candidate.
6. Keep `FFT/` as the clean runtime core; all ROM and paper workflows stay in
   `scripts/`, `src/`, `tests/` and `docs/`.
