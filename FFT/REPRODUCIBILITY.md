# Reproducibility Guide

Esta guia cubre solo el flujo conservado en `FFT/`: SAM para generar RVEs
periodicos de fibras cortas y FFTHomPy/CuPy para resolver homogenizacion FFT
3D. Los resultados del paper nuevo deben construirse a partir de este baseline
full-order y de los prototipos geometry-compiled ubicados fuera de `FFT/`.

## Controles Principales

| Proceso | Valor por defecto |
|---|---|
| Secuencia Sobol | `SOBOL_SCRAMBLE=True`, `SOBOL_SEED=20260621` |
| Semillas Monte Carlo | `MONTE_CARLO_BASE_SEED=20260621` |
| Minimo de seeds | `MIN_SEEDS_BEFORE_STOP=10` |
| Maximo de seeds | `MAX_SEEDS_PER_DESIGN=400` |
| Criterio estadistico | Student-t al 95%, error relativo objetivo `0.02` |
| Resolucion | `VOXELS_PER_FIBER_DIAMETER=5` |
| Backend SAM oficial | `numba` |
| Backend FFT oficial | `cupy` |

Los valores editables estan al inicio de `main.py`. `workflow.py` traduce esa
configuracion a variables de entorno para `generate_sobol_designs.py`,
`sobol_gpu.py`, `rve_generator.py` y `fft_solver.py`.

## Aceptacion De Geometria

Cada RVE se acepta antes del solve FFT solo si satisface los criterios
configurados por el perfil de entorno:

| Chequeo | Valor por defecto |
|---|---|
| Fraccion de fibra voxelizada | `SAM_VF_TOLERANCE=0.005` |
| Error relativo de `A2` continuo | `SAM_A2_TOLERANCE=0.01` |
| Error relativo de `A2` voxelizado | `SAM_VOXEL_A2_TOLERANCE=0.01` |
| Penetracion maxima | `SAM_OVERLAP_TOLERANCE=0.05` voxel |

Con cinco voxeles por diametro, `SAM_OVERLAP_TOLERANCE=0.05` equivale a 1% de
un diametro. En casos densos el generador usa insercion por lotes, compactacion
y relajacion colectiva antes de voxelizar.

## Perfil CPU/GPU

El perfil recomendado mantiene `SAM_GEOMETRY_BACKEND=numba`: varios procesos
CPU generan geometrias mientras la GPU queda libre para FFTHomPy. Para
diagnosticos dedicados se puede usar:

```bash
./main.py --points 10 --seeds 1 --geometry-backend cupy --no-overlap
```

La ejecucion escribe resultados en `FFT/results/`, que esta ignorado por git y
puede regenerarse desde la configuracion y las semillas.

## Comandos Reproducibles

Smoke test de un diseno y una seed:

```bash
./main.py --smoke
```

Piloto corto:

```bash
./main.py --points 10 --seeds 1
```

Validacion corta con multiples seeds:

```bash
./main.py --points 10 --seeds 10
```

Campana base:

```bash
./main.py --points 1024
```

Reanudar una campana interrumpida:

```bash
VALID_EXCEL_PATH=results/Estudio_Sobol_Continuo_sobol_designs_TIMESTAMP/sobol_points_valid.xlsx \
RUN_DIR=results/Estudio_Sobol_Continuo_sobol_gpu_convergence_TIMESTAMP \
  ./main.py --points 1024
```

Los disenos completados solo deben reutilizarse si coinciden configuracion,
criterio Student-t, limites de seeds, derivacion de semillas, paralelismo y
criterios de aceptacion de geometria.

## Salidas

Las campanas generan hojas Excel/CSV de disenos y resumen, manifiestos JSON,
geometrias temporales `phase.npy`/`ori.npy`, tensores efectivos `Ceff.npy` y
timings del solver. Por defecto, las geometrias temporales se eliminan despues
del solve para controlar espacio en disco.

Estas salidas son el baseline full-order para validar despues, fuera de
`FFT/`, el flujo del paper:

```text
G fijo -> operador constitutivo reducido -> Ceff(xi) para muchos materiales
```
