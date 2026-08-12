# CMAME Point-By-Point Audit

Status date: 2026-08-11. A status is `complete` only when executable evidence
and a result path exist. `Partial` means that the implementation or one part of
the numerical protocol exists but the requested campaign is not complete.

## Decisions

| Decision | Status | Evidence |
|---|---|---|
| Keep the seven physical material inputs | Complete, project exception | `paper/main.tex`, Section 2.2 and Appendix E |
| Use `1e-4` as the scientific ROM floor | Complete | `paper/main.tex`, numerical protocol |
| Do not add FEM now | Deferred, project exception | Analytical/energy checks are in `tests/test_fft_physics_verification.py` |
| Keep “Goal-Oriented” in the title | Complete | `paper/main.tex` |
| Consider “Certified” only with zero strict violations | Complete criterion; title unchanged | `results/cmame_method/bound_study/bound_summary.json` |
| Preserve old results and isolate the new campaign | Complete | `results/cmame_method/` |

## Audit Of The 36 Methodology Points

| No. | Status | Evidence or remaining gate |
|---:|---|---|
| 1 | Complete | Scientific question is constitutive transfer compilation in `paper/main.tex`. |
| 2 | Complete | The reduced object is the Schur map, not full-field reconstruction. |
| 3 | Complete | General SPD affine cell problem precedes FFT specialization. |
| 4 | Complete | `H_G: xi -> Ceff` is the central operator. |
| 5 | Complete | Geometry blocks `Kq`, `Bq`, `Dq` are compiled offline. |
| 6 | Exception | Seven-dimensional physical interface retained by decision; scale homogeneity is derived in the appendix. |
| 7 | Complete | Six-load physical tangential enrichment is formalized. |
| 8 | Complete, classical specialization | Tensor and sensitivity interpolation follows established projection/interpolatory MOR; the static constitutive specialization is stated and tested. |
| 9 | Complete method | Candidate selection uses estimator scores; no candidate FOM errors enter the rule. Full five-method comparator remains point 19. |
| 10 | Complete | Error control targets the 6x6 effective tensor. |
| 11 | Complete | `src/constitutive_transfer/schur_estimator.py`; `online_estimator.npz` stores `BB/BK/KK`. |
| 12 | Complete, classical specialization | Ritz/Schur identity and Loewner hierarchy are classical variational consequences specialized to the six-output map; `results/cmame_method/bound_study/loewner_hierarchy.csv`. |
| 13 | Complete decision | Title remains “Goal-Oriented”. |
| 14 | Complete validation | 129 references, zero strict violations on 16 float64 truths; direct median effectivity 8.94 and hierarchical median 1.10. |
| 15 | Complete quality | Exact `48->168` reduced drop plus bounded enriched tail gives median/P95/max effectivity 1.098/1.388/1.411, below the target 5. |
| 16 | Complete, established ingredient | Analytic constitutive sensitivities drive inverse identification; homogenized-property sensitivities and derivative-enabled constitutive surrogates are explicitly cited as prior work. |
| 17 | Complete | Tensor-oriented and sensitivity-aware selections both reached budget 20 and were evaluated on 256 untouched truths. Sensitivity-aware gives max error `1.54e-4` at 20 materials but costs `2129 s`, so no cost-free advantage is claimed. |
| 18 | Complete, narrow result | Equal-span POD/RB tie is retained without superiority claim. |
| 19 | Complete | Frozen 4096 candidates, 256 truths, eight budgets, five methods, and ten random replicates. QoI wins intermediate-budget worst error but not the endpoint; no universal dominance claim. |
| 20 | Complete | `causal_comparator_curve.csv` reports materials, 6x loads, rank, memory, wall cost and mean/P95/max error for every method and budget. |
| 21 | Complete for stated claim | Same geometry rasterized at 48/64/91/128 in `results/cmame_method/voxel_scaling`; the paper claims voxel-error separation, not a universal rank law. |
| 22 | Complete for stated claim | Physical descriptors vary across ten accepted geometries, each treated as a separate compilation; no rank--descriptor law is claimed. |
| 23 | Complete generation | Ten genuinely distinct RVEs pass local and descriptor-separation gates under `results/cmame_method/geometries`. |
| 24 | Complete except declared FEM deferral | Homogeneous, laminate, Mandel, compatibility, equilibrium, Hill-Mandel, voxel and spherical-inclusion refinement pass; Voigt/Reuss and HS bulk bounds pass. |
| 25 | Complete | Screening and conditional identification are applications, not novelty claims. The seven-unknown stress test is a negative control; the main inverse demo restricts the unknowns to `(Ef_L, Ef_T)` after the Jacobian audit. |
| 26 | Complete | 100000 global plus 100000 local samples in `results/cmame_method/uq`. |
| 27 | Complete | Methodological title; short-fiber system is the demonstration. |
| 28 | Complete | The title avoids Certified, Rational, Novel and Short-fiber composites. |
| 29 | Complete | CMAME-style mathematical sequence with key equations, figures, tables, sections, and appendices explicitly cross-referenced throughout. |
| 30 | Complete | Introduction narrows novelty relative to direct RB/CRE/FFT work. |
| 31 | Complete | Verified literature table in `paper/tables/literature_positioning.tex` and proposition-level proof policy in `paper/proposition_literature_audit.md`, including direct static-condensation Schur antecedents. |
| 32 | Complete | Claim is narrowed to the six-output constitutive residual compiler, its two-kernel `BB/BK/KK` realization, and its coupling to physical causal enrichment; classical identities are not claimed. |
| 33 | Complete | Decisive causal and same-budget black-box figures are generated from frozen manifests. |
| 34 | Removed from primary claim | Online complexity is stated for fixed compiled rank; rank dependence on geometry/domain is explicit and is not fitted from a small morphology sample. |
| 35 | Complete | Six-eigenvalue Loewner figure in `paper/figures/cmame_loewner_eigenvalues.pdf`. |
| 36 | Complete framing | General computational-homogenization method demonstrated on 3D short-fiber RVEs. |

## Executed Evidence

| Campaign | Key result | Path |
|---|---|---|
| Solver physics | 15 integrated tests pass; homogeneous error `4.63e-16` | `tests/` |
| FFT truth bound set | 16/16 converged, max residual `9.96e-11` | `results/cmame_method/bound_truth_r48_v3/` |
| Dense/direct estimator | r48 discrepancy `9.12e-12`; r168 relative `2.17e-9`, absolute `9.39e-14`, accepted at diagnostic `1e-8` | ROM directories under `results/fixed_geometry_ffthompy/` |
| Bound | 0 violations; direct/hierarchical median effectivity `8.94`/`1.10`; 4096 complete certified queries in `6.64 s` | `results/cmame_method/bound_study/` |
| Loewner hierarchy | Minimum error eigenvalue `3.17e-4`; minimum nested eigenvalue `1.74e-9` | `results/cmame_method/bound_study/loewner_hierarchy.csv` |
| Voxel refinement | Error vs `128^3`: 4.58e-2, 3.19e-2, 1.25e-2 | `results/cmame_method/voxel_scaling/` |
| Spherical inclusion | Four grids; Voigt/Reuss and HS bulk pass; max residual `9.03e-11` | `results/cmame_method/inclusion_benchmark/` |
| Ten RVEs | 10/10 accepted and descriptor-separated | `results/cmame_method/geometries/` |
| Screening | 4096 queries, 64 Pareto, 0 FOM solves | `results/cmame_method/screening/` |
| Identification | fit `1.26e-6`, rank 7, condition `4.58e4` | `results/cmame_method/inverse_identification/` |
| Noisy identification | 120/120 converged; best median normalized parameter RMSE `0.183--0.239` over `0.1--1%` noise | `results/cmame_method/noisy_inverse/` |
| Conditional identification | Jacobian condition about `15`; median normalized `(Ef_L, Ef_T)` RMSE `0.0065--0.0508` over `0.1--1%` noise | `results/cmame_method/conditional_inverse/` |
| UQ | 200000 tensors, 0 SPD failures, 46.2 s | `results/cmame_method/uq/` |
| Causal design | 4096 candidates, 256 held-out points, random 10x20 frozen before FOM errors | `results/cmame_method/causal_comparator/` |
| Causal evaluation | No universal winner; at 20 materials Sobol mean `1.93e-5`, energy max `1.50e-4`, QoI max `1.65e-4` | `results/cmame_method/causal_comparator/causal_comparator_curve.csv` |
| Black-box control | Training-only-LOOCV log-SPD RBF mean/max error `0.041--0.062`/`0.137--0.335`; mean error remains `1115--2437` times the intrusive ROM | `results/cmame_method/causal_comparator/blackbox_log_spd_rbf_cv/` |
| Paper assets | 29 currently hashed outputs; causal, black-box, and rank assets are implemented and await completed campaigns | `paper/cmame_assets_manifest.json`, `scripts/cmame_generate_paper_assets.py` |

## Honest Submission Gate

The article is now a substantially stronger methodological draft. The causal
result explicitly rejects a universal method-ranking claim. The hierarchical bound meets
the effectivity target for the `r=48` output; a similarly sharp certificate for
the final `r=168` output still requires a richer auxiliary space. The title
therefore remains “Goal-Oriented”.
