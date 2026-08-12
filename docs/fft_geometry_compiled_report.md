# Geometry-Compiled Tangential FFT Homogenization — Feasibility Round 2

Date: 2026-08-10

## 1. Objective

Test whether the previously developed geometry-compiled constitutive transfer operator remains effective when the full-order solver is changed from a small periodic FEM prototype to a matrix-free 3D Fourier-Galerkin / FFT homogenization formulation.

The reduced model targets the effective stiffness directly,

\[
C_r^{\mathrm{eff}}(\xi)=D(\xi)-B_r(\xi)^T K_r(\xi)^{-1}B_r(\xi),
\]

with affine constituent decomposition and a nested tangential reduced space generated from the six macroscopic strain directions.

## 2. FFT formulation tested

The full-order cell problem is solved in the compatible zero-mean strain space using the Fourier compatibility projector. No global stiffness matrix is assembled. For isotropic two-phase elasticity,

\[
\mathcal A(\xi)=\sum_{q=1}^4 \xi_q\mathcal A_q(G),
\]

where the four affine coefficients are the two Lame parameters of matrix and reinforcement. The same construction was generalized to seven affine coefficients for an isotropic matrix plus a transversely isotropic reinforcement.

Odd voxel counts were used in this proof-of-concept to avoid the classical Nyquist-mode ambiguity of the continuous Fourier projector used here. A production implementation should use a discrete/staggered/rotated Green operator or an established Fourier-Galerkin implementation.

## 3. Isotropic 3D FFT test

Analytic microstructure: four irregular ellipsoidal inclusions, approximately 19–20% reinforcement volume fraction.

Parameter domain:

- fiber/matrix Young-modulus contrast: 1 to 1000;
- matrix Poisson ratio: 0.30 to 0.45;
- reinforcement Poisson ratio: 0.18 to 0.32.

For the same analytic geometry at increasing FFT resolutions, the reduced dimension required for maximum tensor error below 0.1% was approximately:

| Grid | Voxels | Required r |
|---|---:|---:|
| 7^3 | 343 | 42 |
| 9^3 | 729 | 42 |
| 11^3 | 1331 | 48 |

Thus the reduced constitutive dimension increased only slightly while the voxel count increased by almost four times.

With r=66, independent-test maximum relative Frobenius errors were approximately 0.027% (7^3), 0.068% (9^3), and 0.020% (11^3) in the executed runs.

## 4. 3D transversely isotropic reinforcement

The FFT operator was generalized to:

- isotropic matrix;
- transversely isotropic reinforcement;
- five independent reinforcement stiffness coefficients;
- a different local fiber orientation for each inclusion;
- seven affine constitutive coefficients in total.

Physical sampling used the ranges already adopted in the composite project (after removing the irrelevant global stiffness scale):

- Em: 1.1–4.4 GPa;
- nu_m: 0.35–0.42;
- E_L: 72–395 GPa;
- E_T: 6–23 GPa;
- G_LT: 8–30 GPa;
- nu_LT: 0.20–0.26;
- nu_TT: 0.35–0.40.

Results:

- 7^3 grid, r=60: mean tensor error 0.00115%, maximum 0.00580%; median full FFT query 0.270 s; median ROM query 1.41e-4 s.
- 9^3 grid, r=48 (smaller training campaign): mean tensor error 0.00505%, maximum 0.0154%; median full FFT query 0.382 s; median ROM query 1.23e-4 s.

The apparent online speedups (about 1900x and 3100x) are Python proof-of-concept numbers and must NOT be used as publication claims until compared against an optimized FFT code.

## 5. Break-even

In the executed TI proof-of-concept, the cost of the offline full-order enrichment was amortized after roughly 8–10 new material queries. This is already relevant for material screening, optimization, UQ, and inverse identification, where hundreds to millions of constitutive queries are common.

## 6. The important estimator result

A field-residual greedy strategy was tested and was not consistently superior to random sampling at equal reduced dimension. This was a useful negative result.

An output/energy-oriented estimator was then tested. For residual columns R associated with the six macroscopic strain states, the estimator uses an inexpensive homogeneous-reference inverse M^{-1} and approximates the constitutive error through

\[
\widehat E = R^T M^{-1}R.
\]

At r=30 in the TI problem:

- Spearman correlation between ordinary field residual and actual Ceff error: about 0.703 in the comparison test;
- Spearman correlation between energy/QoI estimator and actual Ceff error: about 0.969.

At equal r=36:

- field-residual greedy maximum error: 0.0442%;
- energy/QoI greedy maximum error: 0.0193%;
- median maximum error across random-selection trials: 0.0328%.

This makes the energy/QoI estimator a substantially better candidate for the adaptive algorithm.

## 7. Structure preservation and a classical Ritz consequence

Because the reduced model is a Ritz/Galerkin restriction of the same discrete energy minimization problem, for a nested reduced space one expects

\[
C_r^{\mathrm{eff}}-C_{\mathrm{FOM}}^{\mathrm{eff}} \succeq 0,
\]

and

\[
C_{r_k}^{\mathrm{eff}}-C_{r_{k+1}}^{\mathrm{eff}}\succeq0.
\]

This was verified numerically in the TI FFT tests. The smallest eigenvalue of C_r-C_FOM was positive in every tested case; the nested reduced stiffness also decreased monotonically in Loewner order to numerical precision.

This is stronger than merely preserving positive definiteness: it gives an energy upper-bound hierarchy.

## 8. A posteriori certification

Let R contain the six equilibrium residual fields. If alpha is a coercivity lower bound of the local stiffness, then

\[
\|C_r^{\mathrm{eff}}-C^{\mathrm{eff}}\|_F
\leq
\operatorname{tr}(R^T K^{-1}R)
\leq
\frac{1}{\alpha}\sum_{m=1}^{6}\|r_m\|^2.
\]

The simple coercivity bound was verified without violation but was conservative (roughly 15–43 times the observed error in one campaign).

A second certified form used a homogeneous reference operator M and a pointwise spectral-equivalence constant beta such that K >= beta M:

\[
\|C_r^{\mathrm{eff}}-C^{\mathrm{eff}}\|_F
\leq
\frac{1}{\beta}\sum_{m=1}^{6}\langle r_m,M^{-1}r_m\rangle.
\]

This also showed no violations in the executed tests. It remains conservative but provides a practical path toward domain-wide certification.

A fuller Ritz/Schur complement derivation for the current fixed-geometry paper
workflow is now documented in `docs/ritz_schur_error_bound.md`. The important
distinction is that the unscaled quantity `R^T M^{-1} R` is an empirical
ranking indicator, while the certified bound requires the factor
`beta^{-1}` from the pointwise generalized eigenvalue comparison between the
true local stiffness and the homogeneous reference stiffness.

## 9. Sensitivities

For affine coordinates xi_q,

\[
K=\sum_q\xi_qK_q,\quad B=\sum_q\xi_qB_q,\quad D=\sum_q\xi_qD_q,
\]

and X=K^{-1}B, the reduced constitutive derivative is

\[
\frac{\partial C^{\mathrm{eff}}}{\partial\xi_q}
=
D_q-B_q^TX-X^TB_q+X^TK_qX.
\]

At r=54, comparison with finite-difference derivatives of the full TI FFT solver gave approximately:

- mean derivative error: 0.054%;
- maximum derivative error: 0.100%.

This supports optimization and inverse identification without finite-difference FFT sweeps.

## 10. Current interpretation

The promising object is no longer merely a geometry/material separation or a conventional snapshot ROM. The strongest formulation emerging from the tests is:

**Structure-Preserving Goal-Oriented Tangential Constitutive Transfer Operator for FFT Homogenization.**

Its distinguishing ingredients are:

1. compile a fixed voxel microstructure into affine constitutive operators;
2. use only the six physically realized macroscopic directions at each interpolation point;
3. enrich adaptively using an output/energy estimator targeted at Ceff;
4. retain a Ritz structure that gives symmetry, positivity and an upper-bound hierarchy;
5. obtain analytic material sensitivities from the same reduced operator;
6. provide a route to certified error bounds over a material domain.

## 11. Most important next tests

1. Replace the proof-of-concept continuous projector with FFTHomPy/Fourier-Galerkin exact integration or a modern discrete Green operator.
2. Run realistic short-fiber voxel RVEs (50–100 fibers) with the same orientation tensor held fixed.
3. Test r against grid resolution, volume fraction, aspect ratio, orientation tensor, clustering and interface density.
4. Compare offline cost, online cost, break-even and accuracy against Pham–Lee RBHM/POD-style reduced homogenization.
5. Tighten the certified bound using improved spectral-equivalence estimates or residual correction spaces.
6. Perform inverse constituent identification and Monte Carlo UQ as application demonstrations.
