# Geometry-Compiled Rational Homogenization — feasibility tests

## Main numerical findings

- 2D one-parameter contrast, fixed irregular microstructure: full DOF increased from 510 to 8190, while the reduced dimension stayed between 15 and 18 for a <0.1% maximum tensor-error target.
- 2D multiparametric material domain: reduced dimensions stayed between 27 and 45 across 510–8190 DOF.
- Topology study at 4606 DOF: laminate r=9, single circular inclusion r=15, random fibers r=39.
- Direct eigenmode truncation was not sparse: for the 16x16 case, about 265 original modes were needed to get below 0.1% maximum error, whereas the rational transfer model required r=15.
- 3D periodic elasticity: rank(B)=6, as expected from the six independent macroscopic strain states. For a 10^3 voxel RVE with Vf=0.209, r=60 gave mean error 0.0378% and maximum error 0.1501% over 60 unseen material combinations.
- Measured 3D online speed-up in this prototype: 5748x. This is a prototype comparison against sparse direct FEM solves, not yet against optimized FFT.
- Analytic Lamé-parameter sensitivities: the stacked-gradient error was separately checked and remained much smaller than 0.1% in the follow-up diagnostic; individual near-zero derivative components can show larger relative errors.

## Interpretation

The strongest result is not that the microscopic spectrum contains only a few active eigenmodes. It does not. Instead, a dense microscopic spectrum can be represented, for the effective constitutive input-output map, by a very small rational transfer model. The required reduced order appears to depend much more strongly on microstructural/interface complexity than on voxel count.

## 3D transversely isotropic fiber test

A 7-coefficient affine constitutive decomposition was tested: two isotropic matrix coefficients plus five independent transversely-isotropic fiber stiffness coefficients, with a local fiber axis attached to each inclusion. The union of all parameter-independent coupling blocks had rank 22, so a fixed-block strategy grew to r=176 for max error 0.108%. A parameter-dependent tangential strategy, enriching only with the six actual macroscopic right-hand sides G(xi) at each selected material point, reduced this to r=66 while reaching max error 0.132% over 60 unseen material combinations. The measured online speed-up against the sparse direct 3D FE prototype was 796x.
