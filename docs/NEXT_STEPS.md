# NEXT STEPS — Geometry-Compiled Constitutive Transfer Operator

**Status date:** 2026-08-12

## 0. CMAME implementation checkpoint

The following items are now executed and must not trigger more indiscriminate
material enrichment:

- FFT profiles and hard failure gates: truth `float64/rtol=1e-10`, snapshots
  `float64/rtol=1e-8`, timing `float32/rtol=1e-5`; an additional
  `rom_floor` pilot uses `float32/rtol=1e-4` for rapid feasibility checks.
- End-to-end FFTHomPy pilot on `geometry_02` (`76^3`, binary, res6) completed
  with one basis material and one independent truth material. All six load
  cases converged; the basis solve took `0.219 s`, the compiled estimator
  append `6.146 s`, and the ROM query `0.047 ms`. The training tensor was
  reproduced with relative error `4.29e-8`; the independent one-material
  check was `1.998e-2`, so the flow is functional but one snapshot is not a
  global seven-parameter basis. Evidence:
  `results/cmame_method/ffthompy_one_material_geometry02_rtol1e4/`.
- Timing audit after the 1024-candidate refactor: the batched dense Schur
  scan is `1.76 s` for 1024 candidates at `r=120` versus `2.86 s` for the
  scalar loop, with maximum score difference below `1.5e-13` on the audit
  subset. The fast basis smoke completed three snapshots in `0.60 s`; its
  actual compiler cost, not candidate screening, remains the dominant offline
  term.
- Homogeneous, analytical laminate, Mandel, compatibility, equilibrium and
  Hill--Mandel tests: `tests/test_fft_physics_verification.py`.
- Voxel-independent Schur estimator with `BB/BK/KK` contractions and 129
  isotropic references: `src/constitutive_transfer/schur_estimator.py`.
- Dense/direct diagnostic tolerance is `1e-8`; the final `r=168` discrepancy is
  `2.17e-9` relative and `9.39e-14` absolute. This is separate from the
  scientific ROM floor `1e-4`.
- High-precision 16-material bound campaign: zero strict violations. The
  direct `r=48` bound has median effectivity `8.94`; the exact `48->168` drop
  plus bounded tail lowers it to `1.098` (P95 `1.388`, max `1.411`), meeting
  the quality target `<5`.
- Complete hierarchical online benchmark: 4096 certified queries in `6.638 s`
  with median `1.527 ms` and P95 `1.955 ms`, using one BLAS thread.
- Same continuous geometry at `48^3`, `64^3`, `91^3`, `128^3`:
  `results/cmame_method/voxel_scaling/`.
- Production discretization decision for new runs: binary SAM/FFT, no composite
  voxel in the main method, `6` voxels per fiber diameter, and
  `L_domain = 2 L_fiber`. The fast first-pass multi-geometry campaign uses
  moderate physical short-fiber geometries (`AR in [6,12]`,
  `Vf in [0.10,0.24]`) because the required fiber count scales like
  `Vf*AR^2` under this box rule. Composite voxels remain only an ablation for
  aggressive coarsening.
- Spherical-inclusion refinement passes Voigt/Reuss matrix bounds and the
  Hashin--Shtrikman bulk interval: `results/cmame_method/inclusion_benchmark/`.
- Ten accepted, descriptor-separated RVEs with periodic Gaussian-mixture
  clustering: `results/cmame_method/geometries/`.
- Final-operator screening, rank-revealing conditional identification, and UQ
  with `100000 + 100000` samples: `results/cmame_method/`.
- Noisy inverse audit: 120 converged optimizations over three noise levels,
  ten replicates, and four Tikhonov weights. It supports tensor recovery but
  rejects a robust seven-parameter identification claim.
- Conditional inverse audit: fixing five independently characterized inputs
  and estimating `(Ef_L, Ef_T)` lowers the Jacobian condition from `4.58e4`
  to about `15`; median normalized RMSE is `0.0065`, `0.0237`, and `0.0508`
  at `0.1%`, `0.5%`, and `1%` noise.
- CMAME figures/tables regenerated from manifests:
  `scripts/cmame_generate_paper_assets.py`.

The next work is narrowly ordered:

1. Integrate the completed causal and black-box comparisons. The frozen
   4096-candidate/256-truth campaign rejects universal QoI dominance: QoI has
   the best intermediate-budget worst error, while energy wins the 20-material
   maximum. The same-budget log-SPD control is thousands of times less accurate.
2. Compute `r_1e-2`, `r_1e-3`, `r_1e-4` at each voxel grid and then across the
   ten geometries. The current voxel campaign measures discretization error,
   not these ROM ranks.
   The immediate implementation path is the economical loop:
   10 initial truth materials, one adaptive Schur batch of 10 selected from
   1024 ROM-scored Sobol candidates, and 30 independent FFT truths per accepted
   geometry. A second adaptive batch is launched only after validation shows
   that the one-batch operator is insufficient; the raw Schur score alone is
   not allowed to keep consuming materials indefinitely.
   Implementation defaults now reflect this: `--max-materials 20`,
   `--adaptive-batch-size 10`, `--max-adaptive-rounds 1`, and
   `--candidate-count 1024`.
   The economical basis pass uses `--basis-profile timing` (`float32`,
   `rtol=1e-5`) by default; its measured effective-tensor discrepancy is
   below `1e-7` on the pilot and therefore remains below the scientific ROM
   floor. Use `--basis-profile snapshot` for a strict `float64`, `rtol=1e-8`
   offline audit. Held-out validation remains independent truth (`float64`,
   `rtol=1e-10`) in either mode.
   Use `results/cmame_method/multigeometry_validation_binary_res6_optimized/`
   for the clean post-refactor campaign; the earlier
   `multigeometry_validation_binary_res6/geometry_02` run is an interrupted
   pilot and should not be used as final evidence.
3. Build the predeclared causal auxiliary space above `r=168` with
   `scripts/cmame_final_bound_auxiliary.py`, then test the `168->r_aux`
   hierarchical certificate. SCM/anisotropic references remain fallback
   ablations, not prerequisites for the completed `48->168` target.
4. Execute the full log-SPD black-box campaign.
5. FEM remains explicitly deferred by project decision.

The 36-point status and exact evidence paths are maintained in
`paper/cmame_point_by_point_audit.md`.

## 1. Where we stopped

The project is no longer framed as merely "geometry–material separation" or as a generic ROM.
The current paper concept is:

> **Compile a fixed composite microstructure once into a small, structure-preserving constitutive transfer operator that can be queried for many constituent-property combinations without rerunning the full homogenization problem.**

For a fixed geometry `G`, the operator is written in affine constitutive form

\[
K(\boldsymbol\xi)=\sum_q \xi_q K_q(G),\qquad
B(\boldsymbol\xi)=\sum_q \xi_q B_q(G),\qquad
D(\boldsymbol\xi)=\sum_q \xi_q D_q(G),
\]

with

\[
C^{\mathrm{eff}}(\boldsymbol\xi)
=D(\boldsymbol\xi)-B(\boldsymbol\xi)^T K(\boldsymbol\xi)^{-1} B(\boldsymbol\xi).
\]

The ROM is therefore **goal-oriented toward the homogenized constitutive tensor**, rather than being designed primarily to reconstruct the entire microscopic field.

---

## 2. What has already been demonstrated numerically

### 2.1 2D feasibility

- Fixed irregular two-phase RVE.
- Constituent contrast tested up to approximately `E_f/E_m = 1000`.
- Multi-parameter variation of modulus contrast and Poisson ratios.
- Microscopic systems with thousands of DOF were represented by reduced spaces of only tens of states at sub-percent, often sub-0.1%, tensor error.

### 2.2 Reduced dimension vs discretization

The required reduced dimension did **not** scale proportionally with the number of microscopic DOF in the tested refinements.
This supports the working hypothesis that effective constitutive complexity is governed more strongly by input-output/material-space complexity than by voxel/mesh count.

### 2.3 Spectral test

A direct truncation of microscopic eigenmodes required many more modes than the rational reduced model.
Therefore the useful interpretation is **not** "only a few physical eigenmodes are active".
The better interpretation is:

> A dense microscopic spectrum can be condensed into a small number of effective rational/input-output states for the map \(\boldsymbol\xi\mapsto C^{\mathrm{eff}}\).

### 2.4 3D feasibility

- Periodic 3D elasticity was tested.
- The macroscopic excitation rank is six, corresponding to the six independent 3D strain states.
- The reduction remained effective after the 2D-to-3D transition.

### 2.5 Transversely isotropic fibers

Tested:

- isotropic matrix;
- transversely isotropic fibers;
- local 3D fiber orientations;
- five independent TI fiber constants plus matrix constants.

A naive global constitutive-input construction increased the relevant input rank, but a **tangential enrichment** strategy using only the six physical macroscopic excitations at each selected material point reduced the ROM size substantially.

### 2.6 FFT formulation

A matrix-free Fourier/FFT research prototype was implemented.
The reduction remained effective for both isotropic and TI constituent tests.
This is important because the final method should be benchmarked in an FFT-based computational-homogenization setting rather than only with small FEM matrices.

### 2.7 Goal-oriented enrichment

A raw microscopic residual was found to be a mediocre criterion for choosing the next constituent combination.
A QoI/energy-oriented estimator of the form

\[
\widehat{\Delta C}\sim R^T M^{-1}R
\]

correlated substantially better with the true error in \(C^{\mathrm{eff}}\) and produced a better ROM at equal reduced dimension in the tested case.

This is an important methodological direction:

> **Spend the next full-order solve where it improves the homogenized constitutive tensor most, not where the microscopic field residual is simply largest.**

### 2.8 Structure preservation

The Ritz/Galerkin reduced hierarchy numerically showed

\[
C_{r_1}^{\mathrm{eff}}\succeq
C_{r_2}^{\mathrm{eff}}\succeq\cdots\succeq
C_{\mathrm{FOM}}^{\mathrm{eff}}
\]

in the tested problems.
This suggests a useful energy upper-bound / Loewner-monotonicity result that should be derived rigorously for the exact final formulation.

### 2.9 Constituent sensitivities

Analytic reduced sensitivities can be computed from the affine decomposition without rerunning the full FFT problem:

\[
\frac{\partial C^{\mathrm{eff}}}{\partial\xi_q}
=D_q-B_q^T X-X^T B_q+X^T K_q X,
\qquad X=K^{-1}B.
\]

Prototype comparisons with full-order finite differences were accurate enough to justify continuing toward inverse identification and gradient-based design applications.

The fixed-geometry FFTHomPy pilot added a sharper lesson: controlling
\(C^{\mathrm{eff}}\) does not automatically certify all sensitivities. At the
center material, the `r66` ROM matched full-order finite-difference
sensitivities with mean error `4.35e-04` and max error `1.65e-03`. At the
worst 32-point validation material (`7`), the same `r66` tensor error was only
`3.65e-04`, but the physical sensitivity error rose to mean `9.76e-03` and max
`2.91e-02`. Adding that derivative-problematic point as one more tangential
snapshot block produced `r72`, reducing the validation-material-7 sensitivity
error to mean `2.27e-04` and max `4.57e-04`.

---

## 3. What the paper contribution should be

Do **not** claim novelty for any of the following in isolation:

- geometry/material separation;
- reduced basis homogenization;
- rational Krylov;
- tangential interpolation;
- adaptive reduced spaces;
- Schur complements, reduced Schur systems, or their residual certification;
- Ritz identities, Loewner monotonicity, or coercivity bounds;
- homogenized-property sensitivities or their use in inverse/design problems;
- spectral representations of effective properties;
- FFT homogenization.

Each has important prior literature.

The contribution should instead be assembled around the following **specific combination**:

### Contribution A — Geometry-compiled constitutive transfer formulation

Compile a fixed microstructure into constitutive operator blocks and reuse them across a broad constituent-property domain.

### Contribution B — Goal-oriented tangential reduced homogenization

Build the reduced space from the six physical macroscopic excitations at adaptively selected material points, targeting \(C^{\mathrm{eff}}\) directly rather than global microscopic-field accuracy.

### Contribution C — Structure-preserving and controlled constitutive prediction

Derive and validate:

- symmetry/positive definiteness;
- the classical Ritz energy interpretation specialized to the constitutive map;
- the classical Loewner-monotone consequence under the final Ritz assumptions;
- a standard coercivity bound evaluated by the paper-specific six-output compiler;
- analytic constituent sensitivities and their classical projection-interpolation property, used as ingredients rather than isolated novelty claims.

The proposition-level attribution and proof policy are frozen in
`paper/proposition_literature_audit.md`. Huynh--Knezevic--Patera, Eftang--Patera,
and Smetana are the nearest Schur/static-condensation antecedents. Kochmann et
al. (2019), Vondrejc et al. (2020), and Hauck et al. (2026) delimit prior FFT
reduction through sparse frequencies, low-rank tensors, and tensor-train Fourier
acceleration. They must be cited even though they do not contain the present
six-output, two-kernel `BB/BK/KK` constitutive compiler.

The paper should therefore be positioned as a **computational homogenization / model-order-reduction method**, not as ML and not as a new micromechanical closed-form model.

---

## 4. Decisive work still required before writing the final paper

### Priority 1 — Production-quality FFT baseline

Replace/validate the research FFT projector against an established Fourier–Galerkin implementation, preferably FFTHomPy or an independently verified equivalent.

Required checks:

1. periodicity and projection conventions;
2. Kelvin/Voigt factors;
3. six independent 3D load cases;
4. energy consistency/Hill–Mandel;
5. comparison with analytical laminate/inclusion benchmarks;
6. convergence under voxel refinement;
7. robust handling of high material contrast and nearly incompressible matrices.

**Do not use prototype Python timings as final speed-up claims.**

### Priority 2 — Real short-fiber RVEs

Move from a few idealized inclusions to statistically meaningful short-fiber microstructures.

Recommended first campaign:

- fixed number of fibers: ~100;
- periodic RVE;
- isotropic matrix;
- TI fibers;
- prescribed orientation tensor;
- controlled aspect ratio and volume fraction;
- avoid changing geometry while testing constituent transfer.

Then separately vary geometric complexity to study ROM dimension.

### Priority 3 — Scaling study

Measure reduced dimension and cost against:

\[
r(N),\quad r(V_f),\quad r(AR),\quad r(A_{ij}),\quad
r(\text{clustering}),\quad r(\text{interface density}).
\]

The strongest possible result would be evidence that `r` grows slowly with voxel count but responds meaningfully to microstructural/topological complexity.

### Priority 4 — Proper comparator set

At minimum compare:

1. full FFT/Fourier–Galerkin homogenization;
2. conventional snapshot/POD or reduced-basis homogenization;
3. proposed goal-oriented tangential constitutive ROM.

Report:

- offline cost;
- online cost;
- memory;
- reduced dimension;
- max/mean/P95 tensor error;
- break-even number of material queries;
- sensitivity error;
- robustness outside/near edges of the material domain.

### Priority 5 — Error-control derivation

The current residual-energy indicator is promising, but the final paper should distinguish clearly between:

- a heuristic/empirical error indicator;
- a rigorous upper bound.

State and cite the classical error identity associated with the Ritz projection
and Schur complement. Derive only its six-output/two-kernel implementation and
the tightest practical bound obtainable from a reference/preconditioning operator.
Quantify its effectivity index:

\[
\mathrm{eff}(\xi)=\frac{\eta(\xi)}{\|C_r-C_{\mathrm{FOM}}\|}.
\]

A useful bound should be reliable and not excessively conservative.

### Priority 6 — Parameter-domain certification

Define a physically admissible material domain \(\Xi\) for matrix + TI fibers.
The offline greedy should seek control over that domain, not merely over a handful of test points.

Use independent validation samples and, if feasible, a deterministic covering/adaptive partition strategy.
Avoid claiming a mathematically rigorous supremum over a continuous domain until such certification is actually established.

### Priority 7 — Demonstration application

Use at least one many-query application so the ROM has a reason to exist.
Recommended order:

1. **material screening** — easiest and clearest;
2. **inverse constituent identification** — strongest use of analytic sensitivities;
3. **uncertainty quantification** — strongest use of very large query counts.

A compact final paper could use screening + inverse identification and leave full UQ for supplementary material or a follow-up.

---

## 5. Recommended next numerical experiment

### Current narrowed pilot

Start with the single fixed geometry requested for validation before expanding
the geometric campaign:

- one 3D periodic RVE only;
- `AR = 15`;
- `V_f = 0.20`;
- no geometry realizations;
- center + Sobol material points over the existing matrix/TI-fiber domain;
- FFTHomPy/CuPy full-order solves on GPU using the same `phase.npy` and
  `ori.npy`;
- first tangential snapshot block from the six center-material load cases.

Current runner:

```bash
./scripts/fixed_geometry_ffthompy_sweep.py --material-points 8 \
  --solution-field-material-ids 0
```

Current baseline output:

```text
results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields
```

Current reduced-operator status:

- `r=6` from the center material reproduces the center full-order tensor with
  relative error `8.6e-08`;
- greedy tangential enrichment with material IDs `[0, 1, 2, 3, 5, 7]` reaches
  `r=36`;
- on the initial eight material points, `r=36` gives mean tensor error
  `5.45e-05` and maximum tensor error `3.42e-04`;
- independent Sobol validation with 16 new material points showed that `r=36`
  was not yet sufficient over the sampled domain: mean error `1.51e-03`,
  maximum error `9.72e-03`, worst validation material `13`;
- adding validation material `13` as snapshot `1013` produced `r=42` and reduced
  the independent maximum error to `1.25e-03`;
- adding validation material `3` as snapshot `1003` produced `r=48` and reduced
  the independent errors to mean `2.26e-04`, P95 `7.23e-04`, maximum
  `9.80e-04` on the 16-point validation set;
- a first QoI/energy residual indicator was implemented over the saved `r48`
  basis; on those 16 independent points it selected material `15`, which was
  also the true worst point, with Spearman correlation `0.971` against true
  tensor error;
- adding indicator-selected material `15` as snapshot `1015` produced `r=54`
  and reduced the independent errors to mean `1.06e-04`, P95 `2.32e-04`,
  maximum `3.17e-04`, worst material `9`;
- scaling the QoI indicator to a fresh 64-point Sobol candidate pool selected
  candidate `1` without using full-order errors; one FFTHomPy/CuPy solve stored
  it as snapshot `3001`;
- the selected candidate had `r54` error `1.53e-03` and `r60` error
  `9.54e-08` after enrichment;
- on the original 16 independent validation points, `r60` gives mean
  `7.58e-05`, P95 `1.96e-04`, maximum `2.53e-04`, worst material `9`;
- a larger 32-point independent validation exposed a remaining r60 gap:
  mean `1.32e-04`, P95 `3.47e-04`, maximum `1.11e-03`, worst material `30`;
- the r60 QoI indicator ranked material `30` first on that 32-point set
  with Spearman correlation `0.920` against true tensor error;
- solving only material `30` and adding it as snapshot `4030` produced `r=66`;
  on the same 32-point validation set, r66 gives mean `8.58e-05`, P95
  `2.33e-04`, maximum `3.65e-04`, worst material `7`;
- sensitivity checks were implemented for the seven affine coefficients and
  the seven physical material parameters
  `(Em, nu_m, Ef_L, Ef_T, G_LT, nu_LT, nu_TT)`;
- at the center material, `r66` sensitivity agreement against full-order
  FFTHomPy/CuPy finite differences is strong: mean `4.35e-04`, median
  `1.55e-04`, maximum `1.65e-03`;
- at validation material `7`, however, `r66` exposes a derivative gap despite
  good tensor error: sensitivity mean `9.76e-03`, median `8.48e-03`, maximum
  `2.91e-02`;
- collecting validation material `7` as snapshot `4007` and compiling
  snapshots `[0, 1, 2, 3, 5, 7, 1003, 1013, 1015, 3001, 4007, 4030]`
  produced the `r72` operator;
- on the same 32-point validation set, `r72` improves tensor errors to mean
  `6.36e-05`, P95 `1.76e-04`, maximum `2.12e-04`, worst material `9`;
- at validation material `7`, `r72` reduces sensitivity error to mean
  `2.27e-04`, median `2.23e-04`, maximum `4.57e-04`;
- the mixed tensor/sensitivity indicator is now implemented in
  `fixed_geometry_qoi_indicator.py` via `--score-mode mixed`; it differentiates
  the residual-correction estimator by finite differences over the seven
  physical material parameters;
- on the r66 32-point validation set, the mixed indicator selected material
  `7` first: score `1.135e-02`, `eta_C=1.262e-03`, `eta_S=1.128e-02`;
  this is exactly the material where full-order finite differences exposed the
  sensitivity gap;
- after adding material `7`, the r72 mixed subset check shows material `7`
  closes to `eta_S=4.81e-07`, while material `9` remains high with
  `eta_S=5.67e-03`;
- r72 full-order sensitivity finite differences at validation material `9`
  confirm a second derivative gap: mean `1.16e-02`, median `5.61e-03`,
  maximum `4.72e-02`;
- collecting validation material `9` as snapshot `4009` and compiling
  snapshots `[0, 1, 2, 3, 5, 7, 1003, 1013, 1015, 3001, 4007, 4009, 4030]`
  produced the `r78` operator;
- on the same 32-point validation set, `r78` improves tensor errors to mean
  `4.76e-05`, P95 `1.24e-04`, maximum `1.79e-04`, worst material `11`;
- at validation material `9`, `r78` reduces sensitivity error from r72 mean
  `1.16e-02`, maximum `4.72e-02` to mean `8.08e-04`, maximum `1.41e-03`;
- the r78 mixed subset check closes material `9` to `eta_S=2.39e-06` and
  leaves material `11` as the next visible hotspot with `eta_S=1.22e-03`;
- r78 full-order sensitivity finite differences at validation material `11`
  confirm the third derivative gap: mean `6.47e-03`, median `3.45e-03`,
  maximum `2.26e-02`;
- collecting validation material `11` as snapshot `4011` and compiling
  snapshots `[0, 1, 2, 3, 5, 7, 1003, 1013, 1015, 3001, 4007, 4009, 4011, 4030]`
  produced the `r84` operator;
- on the same 32-point validation set, `r84` improves tensor errors to mean
  `3.73e-05`, P95 `9.33e-05`, maximum `1.23e-04`, worst material `1`;
- at validation material `11`, `r84` reduces sensitivity error from r78 mean
  `6.47e-03`, maximum `2.26e-02` to mean `4.49e-04`, maximum `1.32e-03`;
- the r84 mixed subset check closes material `11` to `eta_S=6.07e-07` and
  leaves material `1` as the next visible hotspot with `eta_S=4.43e-03`;
- r84 full-order sensitivity finite differences at validation material `1`
  confirm the fourth derivative gap: mean `3.53e-03`, median `1.77e-03`,
  maximum `1.11e-02`;
- collecting validation material `1` as snapshot `4001` and compiling
  snapshots `[0, 1, 2, 3, 5, 7, 1003, 1013, 1015, 3001, 4001, 4007, 4009, 4011, 4030]`
  produced the `r90` operator;
- on the same 32-point validation set, `r90` improves tensor errors to mean
  `2.82e-05`, P95 `7.08e-05`, maximum `9.67e-05`, worst material `31`;
- at validation material `1`, `r90` reduces sensitivity error from r84 mean
  `3.53e-03`, maximum `1.11e-02` to mean `6.02e-04`, maximum `1.22e-03`;
- the r90 mixed subset check closes material `1` to `eta_S=5.61e-07` and
  leaves material `31` as the next visible hotspot with `eta_S=1.15e-03`;
- r90 full-order sensitivity finite differences at validation material `31`
  confirm another derivative gap: mean `1.95e-03`, median `1.53e-03`,
  maximum `3.85e-03`, dominated by `nu_LT`;
- collecting validation material `31` as snapshot `4031` and compiling it into
  the nested tangential space produced `r96`;
- on the same 32-point validation set, `r96` improves tensor errors to mean
  `2.22e-05`, P95 `5.37e-05`, maximum `7.18e-05`, worst material `6`;
- at validation material `31`, `r96` reduces sensitivity error from r90 mean
  `1.95e-03`, maximum `3.85e-03` to mean `2.28e-04`, maximum `8.03e-04`;
- the r96 mixed subset check closes material `31` to `eta_S=1.75e-07` and
  leaves material `6` as the next visible hotspot with `eta_S=8.78e-04`;
- r96 full-order sensitivity finite differences at validation material `6`
  confirm the next derivative gap: mean `4.08e-03`, median `1.00e-03`,
  maximum `2.03e-02`, dominated by `G_LT`;
- collecting validation material `6` as snapshot `4006` produced `r102`;
- on the same 32-point validation set, `r102` improves tensor errors to mean
  `1.75e-05`, P95 `4.77e-05`, maximum `5.26e-05`, worst material `10`;
- at validation material `6`, `r102` reduces sensitivity error from r96 mean
  `4.08e-03`, maximum `2.03e-02` to mean `5.52e-04`, maximum `2.37e-03`;
- the r102 mixed subset check closes material `6` to `eta_S=1.15e-06` and
  leaves material `10` as the next visible hotspot with `eta_S=5.70e-04`;
- r102 full-order sensitivity finite differences at validation material `10`
  confirm another derivative gap: mean `2.72e-03`, median `5.97e-04`,
  maximum `1.39e-02`, dominated by `nu_LT`;
- collecting validation material `10` as snapshot `4010` produced `r108`;
- on the same 32-point validation set, `r108` improves tensor errors to mean
  `1.17e-05`, P95 `3.16e-05`, maximum `3.55e-05`, worst material `17`;
- at validation material `10`, `r108` reduces sensitivity error from r102 mean
  `2.72e-03`, maximum `1.39e-02` to mean `1.94e-04`, maximum `5.14e-04`;
- the r108 mixed subset check closes material `10` to `eta_S=3.02e-07` and
  leaves material `17` as the next visible hotspot with `eta_S=5.00e-04`;
- r108 full-order sensitivity finite differences at validation material `17`
  confirm the next derivative gap: mean `2.41e-03`, median `1.50e-03`,
  maximum `1.03e-02`, dominated by `nu_TT`;
- collecting validation material `17` as snapshot `4017` produced `r114`;
- on the same 32-point validation set, `r114` improves tensor errors to mean
  `9.31e-06`, P95 `2.32e-05`, maximum `2.76e-05`, worst material `12`;
- at validation material `17`, `r114` reduces sensitivity error from r108 mean
  `2.41e-03`, maximum `1.03e-02` to mean `6.78e-04`, maximum `1.69e-03`;
- the r114 mixed subset check closes material `17` to `eta_S=1.15e-06` and
  leaves material `12` as the next visible hotspot with `eta_S=1.52e-04`;
- r114 full-order sensitivity finite differences at validation material `12`
  confirm a smaller derivative gap: mean `1.08e-03`, median `1.07e-03`,
  maximum `2.04e-03`, dominated by `nu_m`;
- collecting validation material `12` as snapshot `4012` produced `r120`;
- on the same 32-point validation set, `r120` improves tensor errors to mean
  `7.94e-06`, P95 `2.04e-05`, maximum `2.31e-05`, worst material `23`;
- at validation material `12`, `r120` reduces the mean sensitivity error to
  `8.72e-04`; the maximum remains `2.22e-03`, still dominated by `nu_m`;
- the r120 mixed subset check closes material `12` to `eta_S=6.48e-07` and
  leaves material `23` as the next visible hotspot with `eta_S=1.22e-04`;
- r120 full-order sensitivity finite differences at validation material `23`
  confirm the next derivative gap: mean `9.95e-04`, median `3.31e-04`,
  maximum `3.70e-03`, dominated by `nu_TT`;
- collecting validation material `23` as snapshot `4023` produced `r126`;
- on the same 32-point validation set, `r126` improves tensor errors to mean
  `5.66e-06`, P95 `1.56e-05`, maximum `1.74e-05`, worst material `8`;
- at validation material `23`, `r126` reduces sensitivity error from r120 mean
  `9.95e-04`, maximum `3.70e-03` to mean `2.66e-04`, maximum `1.18e-03`;
- the r126 mixed subset check closes material `23` to `eta_S=3.13e-07` and
  leaves material `8` as the next visible hotspot with `eta_S=1.19e-04`;
- r126 full-order sensitivity finite differences at validation material `8`
  confirm another derivative gap: mean `1.00e-03`, median `7.70e-04`,
  maximum `2.85e-03`, dominated by `nu_LT`;
- collecting validation material `8` as snapshot `4008` produced `r132`;
- on the same 32-point validation set, `r132` improves tensor errors to mean
  `4.20e-06`, P95 `1.24e-05`, maximum `1.52e-05`, worst material `25`;
- at validation material `8`, `r132` reduces sensitivity error from r126 mean
  `1.00e-03`, maximum `2.85e-03` to mean `3.36e-04`, maximum `1.62e-03`;
- the r132 mixed subset check closes material `8` to `eta_S=2.18e-07` and
  leaves material `25` as the next visible hotspot with `eta_S=2.54e-04`;
- r132 full-order sensitivity finite differences at validation material `25`
  confirm the next derivative gap: mean `1.65e-03`, median `5.88e-04`,
  maximum `8.81e-03`, dominated by `nu_TT`;
- collecting validation material `25` as snapshot `4025` produced `r138`;
- on the same 32-point validation set, `r138` improves tensor errors to mean
  `2.84e-06`, P95 `9.35e-06`, maximum `1.25e-05`, worst material `13`;
- at validation material `25`, `r138` reduces sensitivity error from r132 mean
  `1.65e-03`, maximum `8.81e-03` to mean `3.53e-04`, maximum `5.40e-04`;
- the r138 mixed subset check closes material `25` to `eta_S=5.32e-07` and
  leaves material `13` as the next visible hotspot with `eta_S=7.29e-05`;
- r138 full-order sensitivity finite differences at validation material `13`
  confirm the next derivative gap: mean `6.40e-04`, median `2.81e-04`,
  maximum `1.62e-03`;
- collecting validation material `13` as snapshot `4013` produced `r144`;
- on the same 32-point validation set, `r144` improves tensor errors to mean
  `2.13e-06`, P95 `7.56e-06`, maximum `8.99e-06`, worst material `16`;
- at validation material `13`, `r144` reduces sensitivity error from r138 mean
  `6.40e-04`, maximum `1.62e-03` to mean `1.84e-04`, maximum `6.45e-04`;
- the r144 mixed subset check closes material `13` to `eta_S=1.60e-07` and
  leaves material `16` as the next visible hotspot with `eta_S=1.35e-04`;
- validation material `16` exposed an important finite-difference methodology
  issue: its `nu_LT` coordinate is very close to the upper material-domain
  bound, so symmetric full-order finite differences can become dominated by
  a boundary-limited stencil rather than ROM derivative error;
- `fixed_geometry_sensitivity_check.py` now supports
  `--fom-fd-scheme edge-auto`, using central full-order differences where
  possible and second-order one-sided inward differences near bounds; with
  that scheme, the r150 material-16 sensitivity check is mean `4.47e-04`,
  median `1.14e-04`, maximum `1.68e-03`;
- collecting validation material `16` as snapshot `4016` produced `r150`;
- on the same 32-point validation set, `r150` improves tensor errors to mean
  `1.82e-06`, P95 `6.52e-06`, maximum `8.45e-06`, worst material `22`;
- the r150 mixed subset check closes material `16` to `eta_S=7.41e-07` and
  leaves material `22` as the next visible hotspot with `eta_S=1.23e-04`;
- r150 full-order sensitivity finite differences at validation material `22`
  with `edge-auto` confirm a moderate derivative gap: mean `9.72e-04`,
  median `6.31e-04`, maximum `2.21e-03`;
- collecting validation material `22` as snapshot `4022` produced `r156`;
- on the same 32-point validation set, `r156` improves tensor errors to mean
  `1.38e-06`, P95 `6.08e-06`, maximum `6.27e-06`, worst material `24`;
- at validation material `22`, `r156` reduces sensitivity error from r150 mean
  `9.72e-04`, maximum `2.21e-03` to mean `6.59e-04`, maximum `1.87e-03`;
- the r156 mixed subset check closes material `22` to `eta_S=7.20e-07` and
  leaves material `24` as the next visible hotspot with `eta_S=2.99e-05`;
- r156 full-order sensitivity finite differences at validation material `24`
  confirm the next derivative gap: mean `6.85e-04`, median `3.98e-04`,
  maximum `2.31e-03`;
- collecting validation material `24` as snapshot `4024` produced `r162`;
- on the same 32-point validation set, `r162` improves tensor errors to mean
  `9.69e-07`, P95 `4.02e-06`, maximum `5.61e-06`, worst material `14`;
- at validation material `24`, `r162` reduces sensitivity mean from
  `6.85e-04` to `4.62e-04`; the maximum remains about `2.35e-03`, so
  full-order finite-difference noise and per-parameter diagnostics should
  continue to be tracked;
- the r162 mixed subset check closes material `24` to `eta_S=2.36e-07` and
  leaves material `14` as the next visible hotspot with `eta_S=7.13e-05`;
- r162 full-order sensitivity finite differences at validation material `14`
  confirm the next derivative gap: mean `6.91e-04`, median `4.90e-04`,
  maximum `2.25e-03`;
- collecting validation material `14` as snapshot `4014` produced `r168`;
- on the same 32-point validation set, `r168` improves tensor errors to mean
  `6.50e-07`, P95 `1.73e-06`, maximum `5.20e-06`, worst material `20`;
- at validation material `14`, `r168` gives sensitivity mean `6.56e-04`,
  median `3.92e-04`, maximum `2.28e-03`; however the mixed indicator closes
  material `14` to `eta_S=1.19e-06`;
- the r168 mixed subset check leaves material `20` as the next visible hotspot
  with `eta_S=4.22e-05`;
- r168 full-order sensitivity finite differences at validation material `20`
  confirm the next derivative gap: mean `9.41e-04`, median `2.00e-04`,
  maximum `5.41e-03`; `nu_m` is boundary-limited and uses the `edge-auto`
  one-sided stencil;
- a formal stop gate was added in `scripts/fixed_geometry_stop_policy.py` to
  prevent the greedy loop from adding material points after the scientific
  tolerance is already met. With the default pilot policy
  `max(Ceff error) <= 1e-05`, `P95(Ceff error) <= 1e-05`,
  `max mixed indicator <= 1e-04`, `mean sensitivity FD error <= 1e-03` and
  `max sensitivity FD error <= 1e-02`, the 32-point `r168` state returns
  `STOP`; therefore validation material `20` should not be collected
  automatically under the default paper-pilot tolerance;
- a larger 128-point independent Sobol validation was then run as a stricter
  robustness check for `r168`. It gives mean tensor error `3.87e-06`, P95
  `1.48e-05`, maximum `4.15e-05`, and worst material `94`; eight of the 128
  points exceed `1e-05`. The stop policy returns `CONTINUE` on tensor criteria
  only. This is a new validation-set failure, not a reason to add material `20`;
  the next full-order enrichment, if continuing the fixed-geometry loop, should
  be only validation material `94` stored as snapshot `4094`;
- the methodological conclusion is that sensitivities are their own QoI for
  inverse identification and gradient-based use cases; the mixed indicator has
  now successfully selected seventeen derivative-problematic points
  (`7`, `9`, `11`, `1`, `31`, `6`, `10`, `17`, `12`, `23`, `8`, `25`, `13`,
  `16`, `22`, `24`, then `14`),
  and each six-load enrichment closed the selected point by the mixed
  indicator; the latest enrichments improve mean sensitivity error but leave
  small max-error caveats to track against full-order finite differences;
- measured r66 timing on the 32-point validation set gives median full-order
  FFTHomPy query time `1.97e-01 s`, median ROM online time `2.26e-04 s`
  and a median full/ROM query speed-up of about `870x`;
- the QoI indicator now has a Fourier-domain CPU path that avoids residual
  FFT round-trips while preserving the same ranking and indicator values to
  roundoff; on the r48/16 validation check it reduced wall time from
  `86.33 s` to `41.98 s` and median candidate time from `5.40 s` to
  `2.62 s`, with the same top material `15`;
- rerunning the executed r54/64 and r60/32 indicator scans with this Fourier
  path preserved the selected top points (`1` and `30`) and reduced total scan
  time from `534.11 s` to `255.20 s`; break-even remains about `194` material
  queries excluding indicator scans and improves from about `2914` to about
  `1494` including the two executed scans;
- a first conventional POD/RB comparator was built from all final 66 tangential
  snapshot fields and validated on the same 32 independent materials. This is a
  deliberately strong POD baseline because it sees every final snapshot before
  truncation: POD reaches max error `7.66e-04` at `r=54`, while the adaptive
  tangential hierarchy reaches the final max error `3.65e-04` at `r=66`;
  the full `r=66` POD and tangential spaces give identical errors because they
  span the same final snapshot space;
- the POD comparator revealed an important diagnostic for the paper: snapshot
  energy is not enough to certify the homogenized tensor. The first six POD
  modes capture `95.03%` of snapshot energy but still give max tensor error
  `1.12e-01` on the 32-point validation set; even `r=18` captures `99.53%`
  energy while max tensor error remains `1.98e-02`;
- `r48` was the first fixed-geometry pilot below `0.1%` maximum relative tensor
  error on a 16-point independent material sample; `r66`, `r72`, `r78`, `r84`,
  `r90`, `r96`, `r102`, `r108`, `r114`, `r120`, `r126`, `r132`, `r138`,
  `r144`, `r150`, `r156`, `r162` and `r168` now support the same conclusion on
  a 32-point sample, with `r168` at maximum sampled tensor error `5.20e-06`,
  but the 128-point follow-up shows that this remains an empirical sample check
  rather than a rigorous certificate over the continuous domain.

### Stop rule for the fixed-geometry pilot

Do not keep enriching the ROM just because the greedy loop can still identify
the next-worst material. The fixed-geometry pilot now uses an explicit
stop/go gate:

```bash
python scripts/fixed_geometry_stop_policy.py \
  --validation results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/independent_validation_r168_sobol32_seed20261111_reuse \
  --indicator results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/mixed_indicator_r168_validation_subset_m14_m20 \
  --sensitivity results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/sensitivity_r168_validation20_fom_fd_edge_auto \
  --require-indicator --require-sensitivity
```

Default pilot targets:

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

Under this policy, `r168` is a `STOP` state on the 32-point validation gate:

| Check | Current value | Target |
|---|---:|---:|
| Mean validation tensor error | `6.50e-07` | `3e-06` |
| P95 validation tensor error | `1.73e-06` | `1e-05` |
| Max validation tensor error | `5.20e-06` | `1e-05` |
| Max mixed indicator | `4.29e-05` | `1e-04` |
| Mean sensitivity FD error at material 20 | `9.41e-04` | `1e-03` |
| Max sensitivity FD error at material 20 | `5.41e-03` | `1e-02` |

For the paper, report `r144` as the first tensor-only stop point below
`1e-05` maximum sampled tensor error, and `r168` as the current conservative
tensor-plus-sensitivity pilot stop. Continue to `material_4020` only if a new
validation set fails this policy or if the manuscript intentionally adopts a
stricter derivative target.

A later 128-point independent Sobol validation with seed `20261217` found a
new tensor hotspot:

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
design files. The authoritative GPU result is the `_gpu` directory.

| Check | 128-point value | Target | Status |
|---|---:|---:|---|
| Mean validation tensor error | `3.87e-06` | `3e-06` | fail |
| P95 validation tensor error | `1.48e-05` | `1e-05` | fail |
| Max validation tensor error | `4.15e-05` | `1e-05` | fail |

The worst point is material `94`, with
`Em=1.141788`, `nu_m=0.350962`, `Ef_L=391.256075`, `Ef_T=7.105549`,
`G_LT=18.562782`, `nu_LT=0.216863`, `nu_TT=0.351779`; this is an extreme
contrast case with `Ef_L/Em=342.67` and `G_LT/Gm=43.93`. The top tensor-error
points are `94`, `29`, `30`, `46`, `71`, `103`, `97`, and `9`.

Because this new validation set fails the stop policy, material `94` is the
recorded hotspot if the fixed-geometry accuracy loop is resumed. Per the
current decision, do **not** keep focusing on adding more materials now; pause
that loop and move to the next paper stages with the `r168` operator while
reporting the 128-point validation caveat.

Optional future enrichment command, only if the accuracy loop is resumed:

```bash
python scripts/fixed_geometry_collect_tangential.py \
  results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields \
  --results-csv results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/independent_validation_r168_sobol128_seed20261217_gpu/validation_full_order_results.csv \
  --material-ids 94 \
  --material-id-offset 4000 \
  --geometry-backend numba \
  --solver-tol 1e-3 \
  --no-runtime-reexec
```

Then compile `r174` with the previous snapshots plus `4094`, revalidate against
the already computed 128 full-order tensors using `--full-order-results-csv`,
and run the stop policy again. Continue one hotspot at a time only if that
revalidation still fails.

### Many-query material screening demo

The next non-enrichment paper stage is now implemented in:

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
results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/material_screening_r168_sobol4096_seed20270117
```

This uses only the compiled ROM and runs zero FFTHomPy full-order solves.

| Quantity | Value |
|---|---:|
| Candidate material queries | `4096` |
| Eligible candidates | `4096` |
| Pareto-front candidates | `64` |
| Wall time | `7.79 s` |
| Median ROM query time | `1.02e-03 s` |
| Best balanced candidate | `413` |
| Best balanced gain | `1.553` |

The best balanced candidate has
`Em=4.287314`, `nu_m=0.351007`, `Ef_L=383.581319`, `Ef_T=22.051695`,
`G_LT=25.860555`, `nu_LT=0.254861`, `nu_TT=0.390380`, with ROM-predicted
`E1=10.4596`, `E2=10.4555`, `E3=11.4120`, and minimum shear modulus
`G_min=4.0286`. This gives the paper a first concrete many-query application:

```text
fixed G -> compiled ROM -> thousands of material candidates -> ranked designs
```

Screening artifacts:

```text
screening_candidates.csv
screening_top_candidates.csv
screening_top_candidates.xlsx
screening_pareto_front.csv
screening_pareto_front.xlsx
screening_summary.json
screening_summary.md
```

### Equal-offline-budget POD/RB comparator

The fair-budget POD/RB interpretation is now generated by:

```bash
python scripts/fixed_geometry_pod_budget_report.py
```

Authoritative output:

```text
results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/pod_equal_offline_budget_r66_validation32
```

This script runs zero FFTHomPy solves. It reuses the existing
`pod_baseline_final_snapshots_validation32` curve and separates the strict
equal-budget result from lower-rank oracle diagnostics.

Strict equal offline and online budget:

| Quantity | Value |
|---|---:|
| Snapshot material IDs | `[0, 1, 2, 3, 5, 7, 1003, 1013, 1015, 3001, 4030]` |
| Offline snapshot fields | `66` |
| Equal online rank | `66` |
| POD mean relative tensor error | `8.58e-05` |
| Tangential mean relative tensor error | `8.58e-05` |
| POD maximum relative tensor error | `3.65e-04` |
| Tangential maximum relative tensor error | `3.65e-04` |
| Max-error ratio POD/tangential | `1.000000` |

Conclusion: at the only strict equal-offline and equal-online point available
in this saved baseline, full-rank POD and tangential RB tie to numerical
precision because both span the same final snapshot space.

The lower-rank POD rows remain useful, but only as oracle diagnostics because
POD saw all 66 final snapshot fields before truncation. They support a different
paper claim: POD snapshot energy is not a tensor-QoI certificate. For example,
the first six POD modes capture `95.03%` of snapshot energy but still give max
tensor error `1.12e-01`; at `r=18`, `99.53%` energy still leaves max tensor
error `1.98e-02`.

### Inverse-identification demo with ROM sensitivities

The inverse-identification demo is now implemented in:

```bash
python scripts/fixed_geometry_inverse_identification.py
```

Authoritative output:

```text
results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/inverse_identification_r168_validation128_m94
```

It uses the `r168` reduced operators and analytic affine ROM sensitivities,
chained with the physical-to-affine coefficient Jacobian. The noiseless
seven-parameter solve below is retained as a local interpolation diagnostic,
not as the robust inverse application claim. It runs zero new full-order solves.

| Quantity | Value |
|---|---:|
| Target material | `94` |
| Multistarts | `8` |
| Best start | `sobol_02` |
| Initial residual norm | `8.79e-01` |
| Final tensor fit relative error | `1.26e-06` |
| ROM mismatch at true target | `4.15e-05` |
| Function evaluations | `43` |
| Best-start wall time | `0.218 s` |
| Maximum parameter relative error | `1.25e-03` |

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

The normalized seven-parameter Jacobian has condition `4.58e4`, and the
120-run noise audit gives parameter RMSE `0.183--0.239`. The manuscript now
uses the rank-revealing alternative implemented in:

```bash
python scripts/cmame_conditional_inverse.py
```

The main application fixes `(Em, nu_m, G_LT, nu_LT, nu_TT)` as independently
characterized inputs and identifies only `(Ef_L, Ef_T)`. Its Jacobian condition
is about `15`; at `0.1%`, `0.5%`, and `1%` tensor noise, median normalized
parameter RMSE is `0.0065`, `0.0237`, and `0.0508`, respectively. The
unrestricted seven-parameter result remains in the appendix as a negative
identifiability control.

### Indicator effectivity study

The empirical effectivity study is now implemented in:

```bash
python scripts/fixed_geometry_indicator_effectivity.py
```

Authoritative output:

```text
results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/indicator_effectivity_fixed_geometry
```

The study computes

\[
\eta_{\mathrm{eff}}(\xi)=
\frac{\mathrm{indicator}_{\mathrm{rel}}(\xi)}
{\|C_r(\xi)-C_{\mathrm{FOM}}(\xi)\|_{\mathrm{rel}}}
\]

on saved indicator rankings with finite `true_relative_error`. It runs zero new
full-order solves and separates empirical indicators from certified bounds.

Overall effectivity:

| Indicator | Samples | Runs | Min | Median | P95 | Max | Underestimates | Conservative fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| energy | 86 | 6 | 6.90e-02 | 1.96e+00 | 3.59e+00 | 4.16e+00 | 9 | 0.895 |
| field residual | 86 | 6 | 1.66e+01 | 6.55e+01 | 3.24e+02 | 7.57e+02 | 0 | 1.000 |
| mixed | 38 | 4 | 2.40e+00 | 1.06e+01 | 3.15e+01 | 3.53e+01 | 0 | 1.000 |
| sensitivity | 38 | 4 | 2.40e+00 | 1.05e+01 | 3.13e+01 | 3.51e+01 | 0 | 1.000 |

Interpretation:

- the energy indicator is useful for ranking but not reliable as a strict
  bound, because it underestimates true tensor error on 9 of 86 sampled
  comparisons;
- the field residual is conservative on these samples but very loose, with
  median effectivity about `65x`;
- mixed and sensitivity indicators are conservative on the saved samples, but
  should still be reported as empirical effectivity until the Ritz/Schur
  complement bound is derived rigorously.

### Classical Ritz/Schur identity and specialized bound

The supporting specialization is documented in:

```text
docs/ritz_schur_error_bound.md
```

The central identity is a classical Galerkin/compliant-output result, cited in
the paper rather than claimed as a new proposition:

\[
\Delta C_r
:=
C_r^{\mathrm{eff}}-C_{\mathrm{FOM}}^{\mathrm{eff}}
=
R^T K^{-1}R
=
E_X^T K E_X
\succeq 0,
\]

where \(R=B-KX_r\) is the residual matrix for the six macroscopic load cases and
\(E_X=X-X_r\) is the unresolved Ritz error. Under the stated SPD and nested-space
assumptions, it implies both:

- \(C_r^{\mathrm{eff}}\succeq C_{\mathrm{FOM}}^{\mathrm{eff}}\);
- nested Ritz spaces give a Loewner-monotone decreasing stiffness sequence.

If a reference operator \(M\) satisfies

\[
K(\xi)\succeq \beta(\xi)M(\xi),
\]

then the certified matrix bound is

\[
0\preceq
C_r^{\mathrm{eff}}-C_{\mathrm{FOM}}^{\mathrm{eff}}
\preceq
\beta(\xi)^{-1}R^T M^{-1}R.
\]

Therefore

\[
\|C_r^{\mathrm{eff}}-C_{\mathrm{FOM}}^{\mathrm{eff}}\|_F
\le
\beta(\xi)^{-1}\|R^T M^{-1}R\|_F.
\]

The original energy indicator omits \(\beta^{-1}\), so it remains an empirical
ranking indicator. The implemented certified-bound runner computes
\(\beta(\xi)\) from local generalized eigenvalues against the same homogeneous
reference used by the indicator.

### Legacy beta-scaled diagnostic table (superseded)

The earlier broad diagnostic runner is retained for traceability:

```bash
python scripts/fixed_geometry_certified_bound.py --overwrite
```

Authoritative output:

```text
results/fixed_geometry_ffthompy/fixed_geometry_ar15_vf20_sobol8_center_fields/certified_bound_fixed_geometry
```

It reads the saved indicator rankings, computes the pointwise
spectral-equivalence factor \(\beta(\xi)\), and reports

\[
\eta_{\mathrm{cert}}(\xi)
=
\beta(\xi)^{-1}\|R^TM^{-1}R\|_F.
\]

It runs zero new FFTHomPy/CuPy solves. For rankings that store the full
energy-correction matrix, the scalar table uses the larger of the Frobenius and
trace bounds; legacy rankings without the matrix use Frobenius only.

| Quantity | Value |
|---|---:|
| Indicator runs | `6` |
| Rows | `86` |
| Rows with absolute effectivity | `86` |
| \(\beta_{\min}\) | `1.405e-01` |
| \(\beta_{\mathrm{median}}\) | `2.558e-01` |
| \(\beta_{\max}\) | `4.109e-01` |
| Minimum certified absolute effectivity | `3.17e-01` |
| Median certified absolute effectivity | `1.08e+01` |
| Strict certified absolute violations | `2` |
| Certified absolute violations above `1e-4` true relative error | `0` |

The two strict violations occur only in the tiny-error regime:

| Run | Material | True relative error | Certified effectivity |
|---|---:|---:|---:|
| `qoi_indicator_r48_independent16_fourier` | 3 | `9.24e-08` | `3.17e-01` |
| `mixed_indicator_r66_validation32` | 30 | `3.64e-07` | `9.79e-01` |

This 86-row table is not the final strict-bound evidence. It motivated the
separate converged `float64/rtol=1e-10` campaign in
`results/cmame_method/bound_study/`. The latter has zero strict violations and
uses the rigorous `48->168` hierarchical tail correction to reach median
effectivity `1.098` without discarding small-error cases.

### Current remaining gates

- Finish the causal 256-truth comparator and paired method statistics.
- Execute rank-versus-grid and rank-versus-geometry campaigns, including the
  exploratory interface-density relation.
- Execute full noisy-inverse and black-box baselines.
- Keep `1e-4` as the scientific ROM floor and keep continuum/voxel error outside
  the discrete ROM certificate.
- Either build an auxiliary space above `r=168` or state explicitly that the
  sharp hierarchical certificate applies to the `r=48` output.

After the fixed-geometry ROM path is verified, broaden the study below.

### Full geometric campaign

The immediate next experiment should be:

### Dataset / geometry

- 3D periodic RVE;
- 100 short fibers;
- `V_f = 0.15, 0.20, 0.25`;
- `AR = 10, 20, 30`;
- three orientation states: nearly aligned, planar, quasi-isotropic;
- at least two realizations per geometric state initially.

### Constituent domain

Use the already considered broad ranges for an isotropic matrix and TI fibers, but parameterize them in a numerically stable affine basis.

### For each fixed geometry

1. build/verify full FFT operator;
2. build tangential ROM adaptively;
3. stop at target estimated constitutive error;
4. validate on an independent Sobol/random material set;
5. record `r`, offline cost, online cost, tensor error, sensitivity error and bound effectivity.

### Primary plots

1. `r` vs voxel count;
2. `r` vs interface/topological complexity;
3. max constitutive error vs `r`;
4. offline/online cost and break-even;
5. proposed estimator vs true error;
6. POD-RB vs proposed method at equal offline budget;
7. inverse-identification convergence with full FFT vs ROM.

---

## 6. Provisional paper positioning

### Working title option

**Certified Goal-Oriented Constitutive Transfer Operators for FFT-Based Composite Homogenization**

Alternative:

**Structure-Preserving Tangential Reduced Homogenization for Rapid Constituent-Space Exploration**

### One-sentence contribution

> A fixed composite microstructure is compiled into a small, goal-oriented constitutive transfer operator that reuses the FFT physics across many constituent-property combinations, enriches only through the six physical macroscopic strain directions, preserves the variational structure, and provides controlled effective-stiffness predictions and analytic material sensitivities without snapshot-POD training or machine learning.

---

## 7. Claims that are currently supported only as prototype evidence

Treat these as **preliminary until reproduced with the production FFT solver and realistic short-fiber RVEs**:

- very large online speed-ups;
- break-even near ten material queries;
- exact numerical values of the reduced dimensions obtained in toy/medium RVEs;
- universal relationship between interface density and reduced dimension;
- tightness of the current rigorous bound;
- general Loewner monotonicity beyond the assumptions of the final Ritz formulation.

The qualitative trends are promising, but final manuscript claims must be regenerated from the production benchmark campaign.

---

## 8. Resume point

When work resumes, start from the fixed-geometry adaptive ROM loop and the
baseline comparator work:

1. do not collect more material snapshots for now; material `94` is recorded as
   an optional future hotspot, but the active path has moved to non-enrichment
   paper stages;
2. use `material_screening_r168_sobol4096_seed20270117` as the first
   many-query application evidence for the fixed-geometry ROM;
3. use `pod_equal_offline_budget_r66_validation32` as the fair POD/RB
   comparator result and keep lower-rank POD rows labelled as oracle
   diagnostics;
4. use `inverse_identification_r168_validation128_m94` as the first
   sensitivity-enabled inverse-identification demo;
5. use `docs/ritz_schur_error_bound.md` as the paper-ready derivation of the
   Ritz/Schur complement identity and reference-operator bound;
6. use `certified_bound_fixed_geometry` as the beta-scaled bound table, with
   certification decisions judged above the `1e-4` paper floor;
7. optimize the mixed indicator beyond finite differences, especially
   candidate batching, precomputed residual actions and/or GPU evaluation; the
   current `eta_S` path costs roughly 15 tensor-indicator evaluations per
   candidate;
8. keep broad geometry campaigns as the next phase after this fixed-geometry
   operator is benchmarked.

That experiment will decide whether the current proof-of-concept becomes a strong computational-mechanics paper or needs another methodological refinement.
