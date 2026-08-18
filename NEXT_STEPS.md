# Next Steps

## Current Priority

Optimize the fixed-geometry Sobol+POD offline stage.

Measured `geometry_00` at 32 materials and rank 192:

| Stage | Float64 baseline [s] | Final optimized [s] | Speedup |
|---|---:|---:|---:|
| FFTHomPy snapshots | 43.03 | 43.54 | 0.99x |
| Full-rank POD updates | 257.96 | 9.77 | 26.41x |
| Reduced `Kq/Bq/Dq` compilation | 172.09 | 31.34 | 5.49x |
| Total training stage | 477.01 | 90.42 | 5.28x |
| Complete pipeline | 502.29 | 116.90 | 4.30x |

The final held-out maximum error changed from `1.249e-5` to `1.241e-5`.
Against the previous optimized implementation, 1024 independent ROM tensors
changed by at most `4.34e-6` relative. The final run uses contiguous `float32`
basis storage, an in-place BLAS projection per checkpoint, orientation-grouped
affine kernels, eight BLAS threads, and ROM chunks of 640 queries. The measured
CPU 10,000-query solve took `3.645 s`. Keeping the operators resident on CUDA
reduced the same rank-192 kernel to `0.307 s`, with a relative difference of
`4.6e-16` against CPU. Peak RSS before immediately releasing the consumed POD
input block was `24.28 GiB`; the active implementation now releases that block
before K/B/D and performs projection without a snapshot-sized temporary.
POD input blocks are capped at `8 GiB`: this allows 16 materials per block on
the `150^3` geometries and automatically limits `240^3` cases to 4.

A complete rank-24 smoke run reduced the 10,000-query stage from `0.410 s` to
`0.223 s`; its ROM kernel was `0.044 s`. Vectorized and previous scalar outputs
agreed to `6.8e-15` relative. A synthetic POD projection benchmark reduced wall
time from `0.0356 s` to `0.0263 s` and peak RSS from `710.9 MiB` to `642.3 MiB`.

## Rank Scaling

With `D = 6*Nvox`, exact full-rank storage is `O(D*r)`, cumulative POD and
K/B/D compilation are `O(D*r^2)`, and each dense online solve is `O(r^3)`.
Over the measured ranks 24--192, cumulative POD and K/B/D times grew with
empirical exponents about 1.22 and 1.15; these favorable finite-range slopes do
not replace the quadratic asymptotic bound.

For `240^3` and `float32`, basis memory is `29.66 GiB` at rank 96,
`59.33 GiB` at rank 192, and `118.65 GiB` at rank 384. The active preflight
accepts rank 288 by reducing POD to one material and K/B/D to two coefficients;
it rejects rank 384 before creating output. ROM batches shrink from 640 queries
at rank 192 to 225 at rank 384 and 56 at rank 768. With a fixed 56-query GPU
batch, the measured exponent over ranks 96--768 was `3.10`, consistent with the
dense cubic solve becoming dominant.

Measured alternatives that should not be reintroduced:

- 12 BLAS threads increased the eight-material K/B/D time by 4.7%; keep 8.
- phase-compressed K/B/D used 2.36x less stress workspace but was 27% slower
  and introduced hidden strided basis copies; keep contiguous GEMMs.
- batched Cholesky was slower than the current dense batched ROM solve.
- writing `reduced_operators_current.npz` after every material had no useful
  runtime benefit and is unnecessary because the active route does not resume
  from it.

Next implementation work:

1. repeat the optimized backend on the largest voxel geometry and record peak RAM;
2. run the ten-geometry campaign only after the largest case passes;
3. generate the final timing and held-out validation tables from those runs.

Do not add stopping or material-selection machinery until the ten-geometry
Sobol+POD baseline is complete.
