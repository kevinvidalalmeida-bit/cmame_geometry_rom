# Reviewer-Gap Audit

Status date: 2026-08-11. This ledger is intentionally stricter than the
manuscript. A gap is closed only when the response, executable evidence, and
claim boundary are all present.

| Likely reviewer question | Risk | Current answer and evidence | Status / required action |
|---|---:|---|---|
| Which mathematical statements are actually new? | Critical | `paper/proposition_literature_audit.md` maps every statement to direct antecedents. Only the six-output two-kernel compiler remains a formal paper proposition. | Closed; repeat Scopus/WoS citation-chain audit before submission. |
| Is Schur/Ritz/Loewner or Hermite interpolation being repackaged as novelty? | Critical | The manuscript cites Prud'homme, Quarteroni, Baur, Huynh, Eftang, Smetana, and Hain; classical proofs were removed from `paper/main.tex`. | Closed. |
| Does the online estimator secretly use voxel fields or FFTs? | Critical | `online_estimator.npz` contains `BB/BK/KK`; the proof in Appendix D and six unit tests verify dense/direct equivalence and hierarchical composition. | Closed. |
| Is the residual bound reliable but too loose to be useful? | Critical | Direct `r=48` median effectivity is 8.939. The exact `48->168` Schur drop plus bounded tail gives median 1.098, P95 1.388, max 1.411, and zero violations. | Closed for the `r=48` output. |
| Why not use SCM or an anisotropic reference? | High | SCM targets the coercivity lower bound and an anisotropic reference targets the residual norm. The hierarchical ablation improves median effectivity by 8.15x without either extra certification problem. | Closed for the present quality target; retain both as alternatives, not missing prerequisites. |
| Is the sharp hierarchical result also a certificate for the final `r=168` output? | Critical | The first causal auxiliary level `168->228` has zero upper-bound violations; its 4096-candidate relative certificate has median/P95 `9.50e-6/6.87e-5`, but 112 candidates exceed the `1e-4` target. A second frozen ten-material block is triggered only by this computable width, never by validation FOM error. The current `rtol=1e-10` FOM comparison also has two small lower-Loewner sign failures and therefore cannot yet authorize a Certified label. | In progress: finish the width-driven `r=288` level and repeat the 16 truths at tighter tolerance before final wording. |
| Does the claimed voxel-independent online complexity survive a 4096-query sweep? | Critical | Complete two-ROM/129-reference benchmark: 4096 queries in 6.638 s, median 1.527 ms, P95 1.955 ms, one BLAS thread. | Closed. |
| Is method ranking causal and fair at equal offline budgets? | Critical | The frozen 4096-candidate/256-truth campaign is complete for five methods and ten random replicates. QoI is best in intermediate-budget worst error but does not dominate at budget 20; all claims are narrowed accordingly. | Closed; disclose both FOM-material budget and unequal total scan times. |
| Would a generic black-box SPD surrogate perform similarly, or is the baseline under-tuned? | High | The same-selected-material log-SPD RBF now selects among three kernels, five shape scales, and four smoothing levels by training-only leave-one-out error. At 20 materials its mean/max errors are `0.041--0.062`/`0.137--0.335`; mean errors remain `1115--2437` times those of the intrusive ROMs, and every prediction is SPD. | Closed; the stronger tuned baseline replaces the median-distance pilot, log-Euclidean geometry is cited as classical, and only the observed value of microscopic structure is claimed. |
| Does voxel-independent online complexity imply a geometry-independent rank? | Critical | No. The complexity statement is now explicitly conditional on fixed compiled rank, and every geometry is compiled independently. The ten-RVE evidence remains a generation/diversity check; no universal rank--morphology law is claimed. | Closed by precise claim boundary. |
| Are ten accepted RVEs enough to claim geometry transfer? | High | They establish generation validity and descriptor diversity only. The manuscript does not claim that one ROM or one rank transfers between geometries. | Closed by scope statement. |
| Is the inverse problem robust to observation noise and weak Jacobian directions? | High | The 120-run seven-unknown stress test gives normalized parameter RMSE `0.183--0.239` and is retained only as a negative control. A rank-revealing reformulation fixes five independently characterized inputs and identifies `(Ef_L, Ef_T)`; its Jacobian condition drops from `4.58e4` to about `15`, with median normalized RMSE `0.0065`, `0.0237`, and `0.0508` at `0.1%`, `0.5%`, and `1%` noise. | Closed by replacing the main application claim with conditional identification; no unrestricted seven-parameter recovery claim. |
| Are sensitivities validated broadly enough? | High | Analytic ROM derivatives match internal differences; selected FOM finite-difference checks show why sensitivity-aware enrichment is needed. The sensitivity-aware causal rule is complete but costs `2129 s` at budget 20 and does not improve tensor error enough to justify that cost alone. | Closed for the stated local interpolation and selection-cost claims; no global derivative-accuracy claim. |
| Is `100000` an arbitrary UQ sample count? | Medium | At 50000 samples, maximum mean change is `7.53e-4`; maximum q05/q95 changes are `5.54e-4` and `2.71e-3`. `uq_convergence.csv` contains the full sequence. | Closed. |
| Does `1e-4` ROM accuracy imply continuum accuracy? | Critical | No. The manuscript separates ROM error from the `1.25e-2` `91^3`-to-`128^3` voxel discrepancy. | Closed by explicit claim boundary; continuum extrapolation remains outside scope. |
| Is the FFT truth solver independently validated? | High | Homogeneous, laminate, Mandel, compatibility, equilibrium, Hill-Mandel, voxel refinement, inclusion refinement, and classical bounds pass. | Closed under the declared no-FEM exception; a cross-code FEM comparison remains deferred. |
| Are floating-point cancellation and truth iteration uncertainty hidden? | High | Dense/direct relative and absolute discrepancies are both reported; a `1e-8` diagnostic tolerance and iterative-truth guard are explicit. | Closed for the discrete study. |
| Can the fixed-geometry linear result be generalized to nonlinear or evolving microstructures? | Medium | No such claim is made. Geometry recompilation and linear SPD affinity are explicit assumptions. | Closed by scope statement. |

## Submission-Critical Order

1. Execute the frozen certificate-driven auxiliary construction above `r=168` and
   report its certificate without changing that budget after seeing validation.
2. Rebuild every asset and compile the manuscript with no undefined citations
   or references.

## Claim Rule

An open row is not converted into positive language. It is either completed,
reported as a limitation, or removed from the claim surface. No post-hoc
selection of geometries, materials, noise realizations, or comparator budgets
is permitted.
