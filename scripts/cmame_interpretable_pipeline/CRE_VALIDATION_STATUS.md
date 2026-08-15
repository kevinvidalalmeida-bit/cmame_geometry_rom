# CRE Validation Status

The primal-dual CRE path is experimental. It must not yet be described in the
paper as a global certificate for the continuous material domain.

## What Has Been Checked

- The seven-term affine compliance decomposition reconstructs the matrix and
  transversely isotropic fiber compliances to floating-point precision.
- The discrete compatible and static projectors pass idempotence and zero-mean
  tests on even and odd grids.
- On the even-grid `geometry_03` pilot, a 1024-point Sobol pool reached the
  `1e-4` target after 31 training materials with primal and dual ranks of 186.
- Sixteen independent FOM evaluations had no observed bracket violation or
  bound underestimate. The largest observed true energy error was
  `2.3621e-6`; the largest CRE indicator was `4.4984e-6`.

These checks support the implementation of a bracket for the current discrete
GaNi model. They are not evidence for every material in the continuous box.

## Known Counterexample To The 1024-Point Claim

An independent 65,536-point Sobol screen of the same frozen even-grid ROM found
21 points above `1e-4`. Its worst reduced indicator was `1.8848e-4`. A strict
FOM solve at that material produced a true energy error of `1.2584e-4`, with
effectivity `1.4977` and no observed bracket violation.

Therefore, passing the original 1024 candidates did not validate the full
material domain and did not even validate the larger finite screen. The pilot
status field consequently says `finite_candidate_pool_passed`, never
`certified`.

## Why The Next Campaign Uses Odd Grids

The current even sizes `60`, `150`, and `240` contain Nyquist modes that need a
special definition of the static complement. The next campaign uses the first
odd sizes above them: `61`, `151`, and `241`. This increases voxel counts by
about 5.1%, 2.0%, and 1.3%, respectively, while preserving at least six voxels
per fiber diameter.

Odd grids simplify the discrete primal-dual decomposition. They do not prove a
continuous-problem bound, and all odd-grid geometries and ROMs must be generated
as a separate campaign.

## Required Before A Certification Claim

1. Reproduce the discrete bracket on all ten odd-grid geometries.
2. Validate bracket orientation and effectivity on independent high-precision
   FOM points, including adversarial material contrasts.
3. Implement and test conservative interval enclosures for the reduced primal
   and dual systems over parameter boxes.
4. Complete branch-and-bound over the seven-dimensional material box, with
   outward rounding and an auditable unresolved-box report.
5. Establish precisely whether the claimed scope is the discrete GaNi model or
   the continuous homogenization problem. A continuous claim needs a separate
   variational and integration argument.

Until all five items pass, the paper should report the ordinary independent FFT
validation of Sobol+POD and describe CRE only as ongoing work.
