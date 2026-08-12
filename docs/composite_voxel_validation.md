# Composite Voxel Validation

Status date: 2026-08-12.

## Literature position

Composite voxels are an established FFT/coarsening technique, not a new
contribution of this paper.  The relevant thread is:

- Kabel, Merkert and Schneider introduced laminate composite voxels for
  FFT-based homogenization of voxel grids whose interfaces are not resolved
  exactly.
- Ma, Shakoor, Vasiukov, Lomov and Park identify irregular voxelized
  interfaces as a dominant source of FFT oscillations and review composite
  voxels and neighbor averaging as reduction strategies.  The user-supplied
  ResearchGate figure illustrates the standard Kabel-style phase-fraction
  estimate: a boundary voxel is subdivided into subvoxels, each subvoxel is
  assigned by its centroid, and the resulting fractions define the local
  composite-voxel law.
- Gelebart and Ouaki proposed material-property filtering for FFT
  homogenization; their two-layer filter is reported as more robust than pure
  Voigt/Reuss filters.
- Keshav, Fritzen and Kabel generalized the idea to composite boxels and
  emphasized that the method is most useful when several fine voxels are
  condensed into one coarse voxel.
- Lendvai and Schneider recast laminate composite voxels as an assumed-strain
  discretization and clarified volume-fraction/normal-estimation issues.
- Lendvai, Kabel and Schneider later compared digital-image normal estimators,
  including lightweight local estimates and more accurate regression-based
  alternatives.

The method is therefore best treated as a discretization/coarsening option.  It
should not be presented as the constitutive-transfer novelty.

## Local implementation

The validation adds an experimental precomputed-`Cfield` path to
`FFT/pipeline/fft_solver.py`.  This lets FFTHomPy solve a voxelwise stiffness
field directly, while keeping the CG solver, projection, Mandel conventions and
postprocessing unchanged.

This branch is now outside the production campaign.  Production multi-geometry
runs use binary SAM/FFT at six voxels per fiber diameter; composite voxels are
retained only as evidence that aggressive coarsening was considered.

Validation script:

```bash
python scripts/cmame_composite_voxel_validation.py --coarse-grid 64
python scripts/cmame_composite_voxel_validation.py --coarse-grid 32
```

Implemented coarse rules:

- `binary`: majority downsampling of the `128^3` reference image.
- `voigt`: subvoxel fiber fraction with affine Voigt mixing.
- `laminate`: subvoxel fiber fraction plus a centroid-to-centroid interface
  normal and a rank-one laminate stiffness.

The Voigt rule is affine in the phase stiffnesses and is compatible with the
current reduced-operator structure.  The laminate rule is closer to the
composite-voxel literature, but is locally nonlinear in the material
parameters; integrating it into the offline-online affine ROM would require a
separate approximation or a changed operator decomposition.

## Results

Reference: existing `128^3` truth tensor in
`results/cmame_method/voxel_scaling/N128/Ceff_truth.npy`, center material.

| Coarsening | Rule | Relative error vs `128^3` | Solve time | Comment |
|---|---:|---:|---:|---|
| `128^3 -> 64^3` | binary | `6.91e-03` | `2.11 s` | best at this grid |
| `128^3 -> 64^3` | laminate | `2.47e-02` | `1.71 s` | worse than binary |
| `128^3 -> 64^3` | voigt | `9.34e-02` | `1.72 s` | too stiff |
| `128^3 -> 32^3` | laminate | `7.70e-03` | `0.52 s` | best; 25x better than binary |
| `128^3 -> 32^3` | binary | `1.93e-01` | `0.91 s` | severe coarsening error |
| `128^3 -> 32^3` | voigt | `1.98e-01` | `0.53 s` | no improvement |

Evidence paths:

- `results/cmame_method/composite_voxel_validation_N064/`
- `results/cmame_method/composite_voxel_validation_N032/`
- `results/cmame_method/composite_voxel_validation_summary.csv`
- `results/cmame_method/composite_voxel_validation_summary.pdf`

## Decision

Composite voxels are worth keeping only as a discretization ablation for
aggressive coarsening.  They do not justify replacing the production binary
SAM/FFT truth path.  The active production decision is binary voxelization with
`6` voxels per fiber diameter and a domain side length equal to twice the fiber
length when generating new RVEs.

Recommended manuscript wording:

> As an implementation check, we tested laminate composite voxels as an
> aggressive coarsening device.  They reduce the `32^3` coarse-grid error by a
> factor of 25 relative to majority binarization, but do not improve the already
> moderate `64^3` grid.  We therefore keep the production truth and ROM
> campaigns on binary SAM/FFT discretizations with six voxels per fiber
> diameter and report composite voxels only as a possible coarse-grid
> acceleration, not as part of the proposed transfer operator.
