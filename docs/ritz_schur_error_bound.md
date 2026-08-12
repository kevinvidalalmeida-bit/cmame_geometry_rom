# Ritz/Schur Complement Error Bound

Date: 2026-08-11

## Purpose

This note turns the current empirical indicator

\[
\widehat E_M(\xi)=R(\xi)^T M(\xi)^{-1}R(\xi)
\]

into the mathematical section needed for the paper. The key point is that the
effective-stiffness error of the Ritz/Galerkin ROM is exactly the missing Schur
complement associated with the unresolved residual. A computable upper bound
then follows from spectral equivalence between the true stiffness operator and a
reference operator.

## Attribution And Proof Scope

The Ritz energy identity, compliant-output residual bound, Loewner consequence,
and coercivity scaling used below are classical reduced-basis/variational
results; reduced Schur systems and their error control also predate this work.
The closest references are
[Prud'homme et al. (2002)](https://doi.org/10.1051/m2an:2002035),
[Boyaval (2008)](https://doi.org/10.1137/070688791),
[Huynh et al. (2013)](https://doi.org/10.1051/m2an/2012022),
[Eftang and Patera (2013)](https://doi.org/10.1002/nme.4543), and
[Smetana (2015)](https://doi.org/10.1016/j.cma.2014.09.020).
The manuscript cites these results and does not reproduce their proofs. The
short calculations retained in this internal note are only sign and notation
checks for the six coupled Mandel outputs; they are not paper contributions.

The paper-specific derivation starts with the pairwise transverse/longitudinal
Fourier compilation of the six-output affine residual atoms into `BB`, `BK`,
and `KK`, followed by online evaluation without voxel fields or FFTs. No direct
antecedent for that exact realization was identified in the audited sources.

The result also explains the current effectivity table:

- the unscaled energy indicator is a good ranking signal;
- it is not a certified bound unless the spectral-equivalence factor is applied;
- a certified version is available once the factor \(\beta(\xi)\) is computed.

## Discrete Setting

Fix one microstructure \(G\) and one admissible material point \(\xi\). After
the periodic FFT/Fourier-Galerkin discretization, write the energy blocks as

\[
K(\xi)=\sum_q \xi_q K_q(G),\qquad
B(\xi)=\sum_q \xi_q B_q(G),\qquad
D(\xi)=\sum_q \xi_q D_q(G).
\]

Here \(K(\xi)\) is symmetric positive definite on the compatible, zero-mean
fluctuation space. The six columns of \(B(\xi)\) correspond to the six
macroscopic Mandel strain load cases. The full-order effective stiffness is the
Schur complement

\[
C^{\mathrm{eff}}(\xi)
=
D(\xi)-B(\xi)^T K(\xi)^{-1}B(\xi).
\]

Let

\[
X(\xi)=K(\xi)^{-1}B(\xi),
\]

so that

\[
C^{\mathrm{eff}}=D-B^TX.
\]

The code stores the fluctuation amplitudes with the opposite sign,
`amplitudes = -X_r`; this note uses \(X\) because it makes the Schur complement
identity cleaner.

## Ritz ROM

Let \(V_r\in\mathbb R^{n\times r}\) be the reduced basis, with columns spanning
the tangential snapshot space. Define

\[
K_r=V_r^T K V_r,\qquad B_r=V_r^T B.
\]

The reduced solution matrix is

\[
X_r=V_r K_r^{-1}B_r,
\]

and the ROM effective stiffness is

\[
C_r^{\mathrm{eff}}
=
D-B_r^T K_r^{-1}B_r
=
D-B^T X_r.
\]

Equivalently, \(X_r\) is the Ritz projection of \(X\) in the
\(K\)-inner product:

\[
V_r^T K(X-X_r)=0.
\]

## Classical Result 1: Ritz Upper Bound And Loewner Monotonicity

For every admissible \(\xi\),

\[
C_r^{\mathrm{eff}}(\xi)-C^{\mathrm{eff}}(\xi)\succeq 0.
\]

If \(V_r\subset V_s\), then

\[
C_r^{\mathrm{eff}}(\xi)-C_s^{\mathrm{eff}}(\xi)\succeq 0.
\]

### Internal Sign Check (Not A Manuscript Proof)

For a macroscopic strain vector \(g\in\mathbb R^6\), the full cell problem is
the minimum of a quadratic energy over the full fluctuation space. The reduced
cell problem is the same minimization restricted to \(V_r\). Restricting the
admissible space can only increase the minimum energy, so

\[
g^T C_r^{\mathrm{eff}}g \ge g^T C^{\mathrm{eff}}g
\]

for all \(g\). This is exactly the Loewner inequality. If \(V_r\subset V_s\),
then \(V_s\) has a larger admissible set than \(V_r\), giving the nested
monotonicity.

This is the mathematical reason the observed reduced hierarchy decreases
toward the full-order effective stiffness.

## Classical Result 2: Exact Schur/Ritz Error Identity

Define the unresolved solution error and the residual matrix

\[
E_X=X-X_r,\qquad R=B-KX_r.
\]

Then

\[
R=K E_X,
\]

and the exact effective-stiffness error is

\[
\Delta C_r
:=
C_r^{\mathrm{eff}}-C^{\mathrm{eff}}
=
E_X^T K E_X
=
R^T K^{-1}R.
\]

Columnwise, for macroscopic load directions \(i,j=1,\dots,6\),

\[
(\Delta C_r)_{ij}
=
\langle e_i,e_j\rangle_K
=
\langle r_i,K^{-1}r_j\rangle.
\]

### Internal Algebra Check (Not A Manuscript Proof)

Using \(C_r^{\mathrm{eff}}=D-B^TX_r\) and
\(C^{\mathrm{eff}}=D-B^TX\),

\[
\Delta C_r
=
B^T(X-X_r).
\]

Since \(B=KX\),

\[
\Delta C_r
=
X^T K E_X.
\]

The Ritz orthogonality \(X_r^T K E_X=0\) gives

\[
X^T K E_X=(X-X_r)^TK(X-X_r)=E_X^T K E_X.
\]

Finally \(R=B-KX_r=K(X-X_r)=K E_X\), so

\[
E_X^T K E_X=R^T K^{-1}R.
\]

Thus the residual-energy matrix is not just a heuristic object: with the true
inverse \(K^{-1}\), it is the exact Schur complement error.

## Certified Bound With A Coercivity Constant

If

\[
K(\xi)\succeq \alpha(\xi) I
\]

on the compatible fluctuation space, then

\[
K(\xi)^{-1}\preceq \alpha(\xi)^{-1}I,
\]

and therefore

\[
\Delta C_r
\preceq
\alpha(\xi)^{-1}R^T R.
\]

Consequently,

\[
\|\Delta C_r\|_F
\le
\alpha(\xi)^{-1}\|R^T R\|_F
\le
\alpha(\xi)^{-1}\operatorname{tr}(R^TR).
\]

This is rigorous but usually loose, because the identity preconditioner ignores
the elastic structure of the residual problem.

For voxel elasticity in Mandel notation, a valid \(\alpha(\xi)\) can be taken
as the minimum local stiffness eigenvalue over all material phases and
orientations:

\[
\alpha(\xi)
=
\min_{x\in G}
\lambda_{\min}\big(C(x;\xi)\big).
\]

## Certified Bound With A Reference Operator

Let \(M(\xi)\) be a symmetric positive definite reference operator on the same
compatible fluctuation space. In the current code, \(M\) is the homogeneous
isotropic Fourier reference operator generated from `reference_lambda0` and
`reference_mu0`.

Assume the spectral-equivalence condition

\[
K(\xi)\succeq \beta(\xi)M(\xi),
\qquad \beta(\xi)>0.
\]

Then

\[
K(\xi)^{-1}\preceq \beta(\xi)^{-1}M(\xi)^{-1}.
\]

Combining this with Classical Result 2 gives the certified matrix inequality

\[
0\preceq
\Delta C_r(\xi)
\preceq
\beta(\xi)^{-1}
R(\xi)^T M(\xi)^{-1}R(\xi).
\]

Therefore a certified absolute Frobenius bound is

\[
\boxed{
\|\Delta C_r(\xi)\|_F
\le
\eta_{\mathrm{cert}}(\xi)
:=
\beta(\xi)^{-1}
\left\|R(\xi)^T M(\xi)^{-1}R(\xi)\right\|_F.
}
\]

This is the rigorous version of the current energy indicator.

## Computing The Spectral-Equivalence Factor

For an operator assembled by voxelwise elastic stiffnesses, it is enough to
enforce the pointwise inequality

\[
C(x;\xi)\succeq \beta(\xi) C_0(\xi)
\quad\text{for every voxel }x,
\]

where \(C_0(\xi)\) is the homogeneous reference stiffness used by \(M(\xi)\).
Then the integrated/projected operator also satisfies

\[
K(\xi)\succeq \beta(\xi)M(\xi).
\]

Thus a computable choice is

\[
\boxed{
\beta(\xi)
=
\min_{x\in G}
\lambda_{\min}
\left(C(x;\xi),C_0(\xi)\right),
}
\]

where \(\lambda_{\min}(A,B)\) denotes the smallest generalized eigenvalue
\(Av=\lambda Bv\).

Implementation detail for the current isotropic-matrix/TI-fiber case:

1. build the matrix Mandel stiffness \(C_m(\xi)\);
2. build the local TI fiber Mandel stiffness \(C_f(\xi)\);
3. build the isotropic reference \(C_0(\xi)\) from the same
   `reference_lambda0`, `reference_mu0` used by the indicator;
4. compute

\[
\beta(\xi)
=
\min\left\{
\lambda_{\min}(C_m,C_0),
\lambda_{\min}(C_f,C_0)
\right\}.
\]

If the reference \(C_0\) is isotropic, rotating the TI fiber does not change the
generalized eigenvalues. If a non-isotropic reference is used later, the minimum
must be taken over all unique fiber orientations.

All eigenvalue calculations must use the same Mandel convention as the solver,
including the shear factors.

## Relationship To The Current Indicator

The current fixed-geometry indicator computes

\[
\eta_C(\xi)
=
\frac{
\left\|R^T M^{-1}R\right\|_F
}{
\left\|C_r^{\mathrm{eff}}\right\|_F
}.
\]

This omits \(\beta(\xi)^{-1}\). Therefore it is a ranking indicator, not a
certified upper bound. This matches the empirical effectivity study:

- energy indicator: useful correlation, but 9 underestimates in 86 comparisons;
- field residual: conservative on the saved samples, but very loose;
- mixed/sensitivity indicators: conservative on saved samples, but not yet a
  theorem for derivative QoIs.

The certified absolute bound should be reported as

\[
\eta_{\mathrm{cert}}(\xi)
=
\beta(\xi)^{-1}
\left\|R^T M^{-1}R\right\|_F.
\]

For validation studies, the effectivity index should be computed with the
absolute tensor error:

\[
\mathrm{eff}_{\mathrm{cert}}(\xi)
=
\frac{\eta_{\mathrm{cert}}(\xi)}
{\left\|C_r^{\mathrm{eff}}(\xi)-C_{\mathrm{FOM}}^{\mathrm{eff}}(\xi)\right\|_F}.
\]

The relative indicator

\[
\frac{\eta_{\mathrm{cert}}(\xi)}{\|C_r^{\mathrm{eff}}(\xi)\|_F}
\]

is useful for online stopping and ranking, but the strict theorem is an
absolute error theorem unless a separate lower bound for
\(\|C_{\mathrm{FOM}}^{\mathrm{eff}}\|_F\) is introduced.

## Hierarchical Tail Correction

Let (V_r\subset V_s) and assume that (U_s) is a valid matrix upper bound
for the enriched error (C_s^{\mathrm{eff}}-C^{\mathrm{eff}}). The exact
decomposition

\[
C_r^{\mathrm{eff}}-C^{\mathrm{eff}}
=
\left(C_r^{\mathrm{eff}}-C_s^{\mathrm{eff}}\right)
+
\left(C_s^{\mathrm{eff}}-C^{\mathrm{eff}}\right)
\]

and the nested Ritz hierarchy give

\[
0\preceq C_r^{\mathrm{eff}}-C_s^{\mathrm{eff}}
\preceq C_r^{\mathrm{eff}}-C^{\mathrm{eff}}
\preceq C_r^{\mathrm{eff}}-C_s^{\mathrm{eff}}+U_s.
\]

Unlike a hierarchical difference indicator that relies on a saturation
assumption, the last matrix is rigorous because the unresolved (s)-level tail
is retained and bounded. Hierarchical RB estimation itself is classical; see
[Hain et al. (2019)](https://doi.org/10.1007/s10444-019-09675-z).

The implemented (48\rightarrow168) bridge reduces median effectivity from
8.939 for the direct (r=48) bound to 1.098, with P95 1.388, maximum 1.411,
and zero violations on 16 independent high-precision materials. The enriched
tail is only 0.855% of the coarse error in median. This result removes the need
for a new SCM or anisotropic-reference implementation for the present quality
target. SCM remains the classical alternative for improving online coercivity
lower bounds; see [Huynh et al. (2007)](https://doi.org/10.1016/j.crma.2007.09.019).
The complete 4096-query hierarchical benchmark takes 6.64 s with one BLAS
thread (median 1.53 ms, P95 1.95 ms per query), including both reduced solves,
129-reference selection, and bound assembly.

The sharp bridge certifies the (r=48) output using (r=168) as the auxiliary
space. It does not, by itself, sharpen the bound on the (r=168) output; doing
that requires a still richer nested auxiliary space or a sharper direct tail
bound.

## Paper-Ready Specialized Statement

Under the assumptions:

1. \(K(\xi)\) is symmetric positive definite on the compatible zero-mean
   fluctuation space;
2. \(V_r\) is a Ritz/Galerkin reduced space;
3. \(M(\xi)\) is symmetric positive definite on the same space;
4. \(K(\xi)\succeq \beta(\xi)M(\xi)\) with \(\beta(\xi)>0\);

the reduced effective stiffness satisfies

\[
C_r^{\mathrm{eff}}(\xi)-C^{\mathrm{eff}}(\xi)\succeq0,
\]

and

\[
0\preceq
C_r^{\mathrm{eff}}(\xi)-C^{\mathrm{eff}}(\xi)
\preceq
\beta(\xi)^{-1}
R(\xi)^T M(\xi)^{-1}R(\xi).
\]

Consequently,

\[
\|C_r^{\mathrm{eff}}(\xi)-C^{\mathrm{eff}}(\xi)\|_F
\le
\beta(\xi)^{-1}
\|R(\xi)^T M(\xi)^{-1}R(\xi)\|_F.
\]

This gives the mathematically certified counterpart of the empirical QoI
indicator used in the adaptive loop.

## Implemented Bound Runners

The high-precision direct/hierarchical study is executed with:

```bash
python scripts/cmame_bound_study.py
```

Its frozen outputs are under `results/cmame_method/bound_study/`.

The earlier broad diagnostic table can still be regenerated with:

The current empirical table has been extended by:

```bash
python scripts/fixed_geometry_certified_bound.py --overwrite
```

The runner:

1. loads saved `qoi_indicator_ranking.csv` files;
2. computes \(\beta(\xi)\) from local generalized eigenvalues against the same
   isotropic reference stiffness;
3. outputs both:
   - empirical indicator: \(\|R^T M^{-1}R\|_F/\|C_r\|_F\);
   - certified absolute bound:
     \(\beta^{-1}\|R^T M^{-1}R\|_F\);
4. when full-order tensors are available, computes certified effectivity:

\[
\frac{\beta^{-1}\|R^T M^{-1}R\|_F}
{\|C_r-C_{\mathrm{FOM}}\|_F}.
\]

The earlier 86-row result below is retained as a historical ranking diagnostic,
not as the final strict-bound evidence:

- 86 saved indicator rows processed;
- \(\beta_{\min}=1.405\times10^{-1}\);
- median certified absolute effectivity: \(1.076\times10^1\);
- two strict absolute violations, both far below true relative error \(10^{-4}\);
- zero absolute violations above the configured \(10^{-4}\) scientific floor.

The final strict evidence is the converged `float64/rtol=1e-10` campaign and
its hierarchical bound described above. Neither study certifies voxel
discretization or continuum error.

## Caveats

- The proof is for the final discrete problem actually solved by the code.
  Therefore the FFT projection convention, Mandel factors and inner-product
  normalization must match between \(K\), \(R\), \(M^{-1}\) and \(C^{eff}\).
- The current energy indicator is deliberately unscaled by \(\beta^{-1}\) for
  ranking. Its underestimates do not contradict the theorem.
- The mixed sensitivity indicator is not yet a certified derivative bound. A
  derivative certificate would require differentiating the residual identity or
  bounding the derivative of the residual-correction operator.
- The Loewner monotonicity relies on nested Ritz spaces and exact reuse of the
  same full-order discrete problem. If geometry, quadrature, projector or
  solver tolerance changes between comparisons, the monotonicity should be
  treated as a numerical diagnostic rather than a theorem.
