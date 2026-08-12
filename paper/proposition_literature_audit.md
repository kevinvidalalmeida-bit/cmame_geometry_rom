# Proposition-Level Literature Audit

This audit separates classical results, problem-specific specializations, and
the methodological derivation claimed by the paper. It is intentionally more
conservative than a keyword-level novelty search.

| Manuscript statement | Closest direct antecedent | Classification in the paper | Proof policy |
|---|---|---|---|
| Galerkin/Ritz best approximation and quadratic compliant-output error | Prud'homme et al. (2002), Quarteroni et al. (2011) | Classical | Cite and state the matrix-output specialization; do not retain a separate proof. |
| Pairwise Gram precomputation for affine residual dual norms | Prud'homme et al. (2002), Quarteroni et al. (2011), and standard certified RB implementations | Classical offline-online mechanism | Cite; do not claim pairwise residual contractions or mesh-independent affine residual evaluation in isolation. |
| RB homogenization with offline-online separation and residual control | Boyaval (2008) | Classical | Cite; do not claim RB homogenization or affine separation as new. |
| POD/RB and variational reduction of constitutive computational homogenization | Yvonnet and He (2007); Fritzen and Leuschner (2013); Hernandez et al. (2014); Michel and Suquet (2016); Kunc and Fritzen (2019) | Established neighboring field | Cite; do not claim reduced evaluation of effective stress/stiffness, variational-structure preservation, or constitutive response as new. |
| Reduced Schur systems, Schur-complement perturbation error, and static-condensation certification | Huynh et al. (2013); Eftang and Patera (2013); Smetana (2015) | Classical adjacent methodology | Cite; explicitly separate a reduced Schur system or Schur error estimator from the present six-output constitutive/Fourier compiler. |
| CRE upper/lower control in computational homogenization | Kerfriden et al. (2014) | Classical, distinct estimator family | Cite and contrast the stress-equilibrated CRE construction with the present reference-energy bound. |
| Goal-oriented/two-field certified elasticity RB | Hoang et al. (2016) | Classical, distinct estimator family | Cite; do not claim goal-oriented RB or certification in isolation. |
| Parameter value, gradient, and Hessian interpolation under projection conditions | Baur et al. (2011) | Classical | Cite and state the static Schur consequence; do not present it as a paper proposition. |
| Sensitivity analysis of homogenized elasticity, reduced constitutive models with derivatives, and sensitivity-guided homogenization reduction | Kaminski (2003); Guo et al. (2021); Lukes and Rohan (2026 preprint) | Established/active neighboring methodology | Cite; do not claim homogenized-property sensitivities, optimization, inverse use, or sensitivity-guided reduction in isolation. |
| Sensitivity-matrix conditioning and identifiable parameter-subset selection | Brun et al. (2001) and the practical-identifiability literature | Classical inverse-problem diagnostic | Cite; present the `(Ef_L, Ef_T)` restriction as a problem reformulation supported by this campaign, not as a new subset-selection theorem. |
| Fourier Green operator for an isotropic comparison medium | Moulinec and Suquet (1998); Vondrejc et al. (2014) | Classical | Cite; do not claim the longitudinal/transverse inverse itself. |
| Variational upper/lower bounds for FFT homogenization | Vondrejc et al. (2015) | Classical | Cite; distinguish discretization bounds from the present ROM-error bound. |
| FFT model reduction and acceleration by sparse frequencies or low-rank/tensor-train Fourier representations | Kochmann et al. (2019); Vondrejc et al. (2020); Hauck et al. (2026) | Prior FFT-MOR/acceleration routes | Cite and contrast; no novelty claim for FFT reduction, low-rank field representation, or tensor-train Fourier acceleration alone. |
| Reduced-basis thermal/elastic periodic homogenization and reduced corrector gradients | Pham and Lee (2025) | Closest recent RB homogenization baseline | Cite and compare numerically/methodologically; no novelty claim for RB corrector evaluation. |
| Schur complement representation of the effective tensor | Standard block elimination / static condensation | Classical algebra | State without novelty language or a separate proof. |
| Loewner monotonicity for nested Ritz spaces | Classical variational consequence | Classical | Cite and state under the nested Ritz assumptions; no separate proof. |
| Coercivity-scaled residual bound | Classical certified RB result | Classical specialization | Cite and state the six-column Loewner form; derive only the project-specific local beta evaluation and compiler. |
| Hierarchical estimation from two nested reduced approximations | Hain et al. (2019) | Classical estimator strategy | Cite; the paper uses the exact reduced Schur drop plus a rigorous residual bound on the enriched tail, without claiming hierarchical estimation itself. |
| Successive Constraint Method for online coercivity lower bounds | Huynh et al. (2007) | Classical alternative not required by the final implementation | Cite in the ablation discussion; explain that SCM would target the coercivity factor, whereas the adopted nested bridge removes most of the error before the same valid two-kernel tail bound is applied. |
| Log-Euclidean representation of SPD matrices | Arsigny et al. (2007) | Classical geometry used only by the black-box control | Cite; no novelty claim for the matrix logarithm, exponential reconstruction, RBF kernels, or training-only hyperparameter selection. |
| Exact elastic-scale homogeneity | Elementary consequence of linearity | Observation | State and explicitly disclaim novelty. |
| Conversion from a computable absolute upper bound to a relative tolerance through the reverse triangle inequality | Elementary norm inequality | Classical stopping corollary | State without novelty language; use it to replace unstable effectivity-based stopping below the scientific error floor. |
| Pairwise two-kernel compilation of all six-output affine residual atoms into `BB`, `BK`, and `KK`, followed by voxel-free dense online evaluation | Generic residual Gram precomputation is classical; no direct antecedent for this six-output, two-physical-kernel realization was identified; static-condensation RB is the nearest Schur-based family | Paper-specific methodological proposition | Provide the paper's formal proposition, complete proof, algorithm, complexity, direct-Fourier equivalence test, and numerical evidence. Do not present pairwise Gram assembly, the Schur complement, or the Fourier split separately as new. |
| Coupling that compiler to six-load physical enrichment and tensor/sensitivity-aware causal selection | No direct antecedent identified in the audited sources as this exact combination | Paper-specific construction and numerical claim | Describe fully and support only with the frozen equal-budget comparator. |

## Claim Boundary

The paper must not claim novelty for POD, RB, affine decomposition, Galerkin
orthogonality, Schur complements or their static-condensation reduction,
parameter-derivative interpolation, CRE,
isotropic Fourier Green operators, or FFT homogenization separately.

The defensible novelty claim is the specific combination and realization:

1. compile a fixed microstructure into a six-output constitutive Schur map;
2. enrich it with the six physical macroscopic responses;
3. preserve exact value and first-sensitivity interpolation at selected points;
4. compile every affine residual pairing through two Fourier kernels into
   `BB`, `BK`, and `KK` arrays; and
5. evaluate the matrix error bound and adaptive scores online without voxel
   fields or FFTs.

The phrase "no direct antecedent identified" is a literature-audit result, not
proof that no antecedent exists. It should be stated as "to the authors'
knowledge" in the manuscript and revisited before submission using Scopus/Web
of Science forward and backward citation searches.

## Search Scope

The audit was updated on 2026-08-11 using publisher records, DOI metadata,
open-access manuscripts, arXiv, and formula-focused searches for the following
families: reduced-basis homogenization, static-condensation Schur error control,
matrix/compliant-output residual identities, homogenized-property
sensitivities, constitutive surrogate derivatives, and FFT/Fourier offline-online
estimators. Both backward references and recent adjacent work through 2026 were
checked.

This is a defensible working audit, not a proof of bibliographic absence. Before
submission, the exact `BB/BK/KK`, six-output, and two-kernel query set must be
repeated in Scopus and Web of Science, followed by forward citation searches
from Huynh et al. (2013), Smetana (2015), Kochmann et al. (2019), Vondrejc et
al. (2020), Guo et al. (2021), Pham and Lee (2025), and Hauck et al. (2026).
