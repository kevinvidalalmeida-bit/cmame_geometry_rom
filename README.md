# Map of Repository Architecture and Codebase Inventory

This document provides a comprehensive inventory of all source files, execution scripts, applications, and test suites in this repository.

---

## 1. Core Model & ROM Engine (`src/` and `scripts/`)

These modules provide the reusable mathematical foundation for periodic FFT computational homogenization, Reduced Order Modeling (ROM), POD basis extraction, and online error estimation.

* **`src/constitutive_transfer/schur_estimator.py`**  
  Core 2-kernel online-offline estimator compiler. Compiles $BB$, $BK$, and $KK$ contraction matrices, computes Loewner energy bounds, performs reference searching, and handles online Cholesky factorizations.
* **`src/constitutive_transfer/__init__.py`**  
  Package initialization for the `constitutive_transfer` module.
* **`scripts/rom_reduced_operator.py`**  
  Assembles reduced stiffness operators ($K_q, B_q, D_q$) and provides vectorized batch ROM evaluation (`_rom_ceff_batch`) to compute effective constitutive tensors and relative Frobenius errors.
* **`scripts/rom_pod_basis.py`**  
  Constructs orthonormal POD bases from snapshot fluctuation field matrices using accelerated eigendecomposition (`scipy.linalg.eigh`).
* **`scripts/fft_homogenization_solver.py`**  
  Interface to the underlying FFTHomPy solver engine. Manages GPU/CPU runtime settings, material domain bounds, and parameter sweeps.
* **`scripts/schur_estimator_compiler.py`**  
  Compiles 2-kernel online estimator contraction matrices directly from snapshot basis fields for a fixed geometry.
* **`scripts/schur_energy_indicators.py`**  
  Calculates residual energy indicators and Loewner energy bounds.
* **`scripts/rom_validation_utils.py`**  
  Utilities for evaluating ROM effective stiffness tensors against exact full-order FFT homogenization solves across material sets.
* **`scripts/inverse_identification_utils.py`**  
  Helper functions for sensitivity-guided inverse identification, Mandel 21-vector conversions, and Gaussian noise matrix perturbation.
* **`scripts/sensitivity_verification.py`**  
  Analytical ROM sensitivity verification against full-order finite differences.
* **`scripts/cmame_campaign_common.py`**  
  Shared execution infrastructure: CUDA environment setup, BLAS thread limiting, snapshot cache disk I/O, candidate material table loading, and error statistics.

---

## 2. Main Execution Pipelines (`scripts/`)

These scripts execute the primary adaptive Sobol+POD homogenization and model reduction workflows:

* **`scripts/cmame_sobol_pod_adaptive.py`** *(Single-Geometry Pipeline)*  
  Main pipeline script for a single microstructure. Sequentially solves Sobol snapshots on GPU, adaptively builds full-rank POD bases until target relative error tolerance is reached, and validates against held-out FFT solves.
* **`scripts/cmame_multigeometry_adaptive.py`** *(Multi-Geometry Pipeline)*  
  Main pipeline script for multi-geometry campaigns. Runs the adaptive ROM compilation across 10 (or $N$) microstructures and validates against 32 independent FFT solves per geometry.

---

## 3. Paper Applications & Post-Processing (`scripts/applications/`)

Specific study applications and figure generation scripts isolated from the main compilation pipeline:

* **`scripts/applications/cmame_generate_paper_assets.py`**  
  Generates all vector PDF figures and LaTeX tables for the manuscript.
* **`scripts/applications/cmame_uq.py`**  
  Uncertainty Quantification: Monte Carlo and Quasi-Monte Carlo sampling over material parameter space using the compiled ROM.
* **`scripts/applications/cmame_conditional_inverse.py`**  
  Sensitivity-guided inverse identification of constituent elastic properties from noisy macroscopic stiffness tensor queries.
* **`scripts/applications/cmame_bound_study.py`**  
  Reliability study of certified offline-online error bounds and Loewner hierarchy verification.
* **`scripts/applications/cmame_final_bound_auxiliary.py`**  
  Hierarchical tail-enrichment bound study using auxiliary spaces.
* **`scripts/applications/cmame_inclusion_benchmark.py`**  
  Spherical inclusion benchmark validating predictions against analytical Hashin-Shtrikman bounds.
* **`scripts/applications/cmame_rank_scalability.py`**  
  Rank vs error target ($10^{-2}, 10^{-3}, 10^{-4}$) scalability study across voxel resolutions ($32^3, 64^3, 128^3$).
* **`scripts/applications/cmame_voxel_scaling.py`**  
  Discretization grid resolution scaling verification.
* **`scripts/applications/cmame_composite_voxel_validation.py`**  
  Voxel grid validation metrics across short-fiber composite geometries.
* **`scripts/applications/cmame_geometry_campaign.py`**  
  Stochastic 3D short-fiber microstructure generation and voxel rasterization.

---

## 4. Automated Test Suite (`tests/`)

Automated unit tests run via `pytest tests/` to verify physical laws and numerical equivalence:

* **`tests/conftest.py`**  
  Pytest configuration for automatic `sys.path` import resolution.
* **`tests/test_fft_physics_verification.py`**  
  Verifies strain compatibility, stress equilibrium, and Hill-Mandel macroscopic energy conservation.
* **`tests/test_fft_projection_cache.py`**  
  Verifies GPU projection operator memory caching across fixed-grid solves.
* **`tests/test_ffthompy_relative_solver.py`**  
  Verifies relative conjugate gradient convergence tolerances and scale invariance.
* **`tests/test_schur_offline_online.py`**  
  Verifies 2-kernel estimator identities, contraction matrix assembly, and Loewner bounds.
* **`tests/test_schur_incremental.py`**  
  Verifies incremental contraction matrix updates against batch compilation.
* **`tests/test_schur_reference_envelope.py`**  
  Verifies convex reference envelope search for optimal beta bounding.
* **`tests/test_cmame_incremental_compilation.py`**  
  Verifies numerical equivalence between block Gram-Schmidt orthonormalization and full batch assembly.
* **`tests/test_cmame_geometry_protocol.py`**  
  Verifies stochastic 3D fiber generation, wrapping, and rasterization.
* **`tests/test_cmame_inclusion_bounds.py`**  
  Verifies predictions lie strictly between analytical Hashin-Shtrikman upper and lower bounds.
* **`tests/test_cmame_noisy_inverse.py`**  
  Verifies Gaussian noise matrix perturbation scaling and Mandel vector norm preservation.
* **`tests/test_cmame_rank_scalability.py`**  
  Verifies rank thresholding functions for error targets.
* **`tests/test_cmame_uq.py`**  
  Verifies truncated normal sampling bounds and UQ statistics convergence.

---

## 5. Paper Source (`paper/`)

* **`paper/main.tex`**: Primary Elsevier LaTeX manuscript file.
* **`paper/figures/`**: Generated vector PDF figures.
* **`paper/tables/`**: Generated LaTeX input tables.
