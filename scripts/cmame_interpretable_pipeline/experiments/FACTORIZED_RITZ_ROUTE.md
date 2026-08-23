# Exact factorized Ritz route

Date: 2026-08-23

Status: principal compiler route. Promoted after exactness tests and the G03/G09
timing studies. The gathered-factor schedule described in the companion
study is now part of this route.

## Route

The principal compiler combines four changes:

1. Exact algebraic-rank factorization of each symmetric 6x6 affine Mandel
   basis, `A_q = U_q diag(d_q) U_q^T`.
2. Float64 reduced `K_q` and `B_q` accumulators remain resident on the GPU;
   only the completed small blocks return to the CPU.
3. An optional two-stream CUDA pipeline overlaps factorized contractions with
   pinned host-to-device snapshot transfers. The total pinned-buffer cap is
   1 GiB. Training still loads and solves one material at a time.
4. Orientation-dependent fiber factors are gathered once per voxel chunk and
   reused across the five supported affine coefficients.

The exact nonzero constitutive ranks are:

- matrix: `[1, 6]` instead of 12 component passes;
- fiber: `[3, 3, 2, 1, 2]`, totaling 11 instead of 30 component passes.

These are exact ranks of the fixed 6x6 constitutive bases. No snapshot
direction, POD mode, or affine coefficient is truncated.

The route is enabled by default in `campaign_config.json`. An explicit run is:

```bash
python scripts/cmame_interpretable_pipeline/04_sobol_pod_pipeline.py \
  --geometry-id 9 --adaptive \
  --reference-energy-qr \
  --factorized-ritz \
  --async-ritz \
  --gathered-factor-ritz
```

## Measurements

All runs used the same Sobol/monitor seeds, float32 FFT and Ritz contractions,
one material per load batch, and the RTX 5070 Ti. Final validation was reduced
to one point because this experiment measures compilation, not campaign
generalization.

| Geometry | Route | Assembly [s] | Compilation [s] | Assembly change |
|---|---|---:|---:|---:|
| G03, 60^3 | energy QR baseline | 1.049 | 3.089 | reference |
| G03, 60^3 | factorized resident | 1.030 | 3.359 | -1.7% |
| G03, 60^3 | factorized + async | 1.357 | 3.831 | +29.4% |
| G09, 240^3 | energy QR baseline | 75.676 | 219.970 | reference |
| G09, 240^3 | factorized resident | 51.887 | 205.589 | -31.4% |
| G09, 240^3 | factorized + async | 35.236 | 188.362 | -53.4% |

The observed G09 compilation reduction is 14.4%. Its snapshot solves were
6.76 s slower and its monitor solves were 1.85 s slower than the baseline
run. Replacing only those two noisy contributions by the baseline values gives
an adjusted compilation time of about 179.75 s, or an 18.3% reduction.

## Memory

At the largest G09 prefix, the asynchronous route used at most 1.00 GiB of
pinned host buffers and 0.98 GiB of factor workspace. The two GPU snapshot
buffers use approximately another 1.00 GiB, so the measured GPU peak is
about 1.98 GiB plus small CuPy caches and reduced accumulators. This is viable
on the 16 GiB device and does not change `load_batch_size=1`.

## Numerical comparison

Small-array tests compare the standard and factorized raw affine blocks. In
float32, relative differences were 3.75e-7 for `K_q`, 2.26e-8 for `B_q`, and
zero for `D_q`. Factorized synchronous and asynchronous full assembly were
bitwise identical; asynchronous incremental extension matched within the
2e-6 test tolerance.

Across the five final monitor tensors, the maximum relative difference from
the prior energy-QR route was 3.15e-6 for G03 and 5.64e-6 for G09. Both routes
selected the same training counts and ranks: 11/66 for G03 and 16/96 for G09.

## Decision

Use the factorized asynchronous route for all campaign geometries. Its small
G03 overhead is negligible in absolute compilation time, while the G09
assembly reduction is substantial. Keeping one deterministic route also
avoids a grid-size policy becoming another campaign variable. The historical
CPU-Gram route remains available through explicit `--no-*` switches for
audits.

## Full campaign confirmation

The promoted route was rerun for G00--G09 as
`full_span_gathered_20260823`, with 20 independent float64 held-out
references per geometry. All ten geometries reached the monitor target, no
reduced evaluation failed, and the selected training sizes/ranks ranged from
6/36 to 16/96. The campaign-wide held-out mean and maximum errors were
`2.138e-4` and `1.306e-3`; 89/200 cases met `1e-4` and 195/200 met `1e-3`.

The measured compilation times ranged from 1.79 s to 174.93 s. For the two
largest grids, G04 compiled in 128.80 s and G09 in 174.93 s, reductions of
37.5% and 33.3% from the preceding component-action/Euclidean-Gram campaign.
The comparison includes both the gathered factor schedule and removal of the
repeated Euclidean-Gram transformations, so it is an end-to-end route result.

The local-frame alternative was also implemented and benchmarked; the
companion `LOCAL_FRAME_RITZ_EXPERIMENT.md` records why it remains an audit
route rather than the campaign default.
