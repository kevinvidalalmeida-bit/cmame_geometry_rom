# SAM + FFTHomPy FFT Core

`FFT/` queda como el nucleo operativo para las prioridades inmediatas de
`NEXT_STEPS.md`: generar geometria SAM en CPU, resolver homogenizacion FFT con
FFTHomPy/CuPy en GPU y producir campanas full-order que sirvan como baseline
para el paper nuevo.

Los prototipos ROM, pruebas geometry-compiled, reportes y comparadores viven
fuera de esta carpeta, principalmente en `src/`, `tests/` y `docs/`.

## Flujo

1. `main.py` define los rangos de geometria/material y prepara el entorno CUDA
   configurado en `VENV_PATH`.
2. `pipeline/generate_sobol_designs.py` genera los puntos Sobol y parametros
   derivados.
3. `pipeline/sam_generator.py` y `pipeline/rve_generator.py` construyen RVEs
   periodicos de fibras cortas y voxelizan `phase`/`ori`.
4. `pipeline/fft_solver.py` llama el nucleo FFTHomPy/CuPy para las seis cargas
   macroscopicas 3D.
5. `pipeline/sobol_gpu.py` orquesta semillas, convergencia y escritura de
   resultados en `FFT/results/`, carpeta ignorada por git y recreada al correr.

El perfil recomendado es `--geometry-backend numba`: la CPU genera geometrias
SAM mientras la GPU queda reservada para el solve FFT.

## Comandos

Desde la raiz del repositorio:

```bash
cd FFT
```

Smoke test de un diseno y una seed:

```bash
./main.py --smoke
```

Campana piloto pequena:

```bash
./main.py --points 10 --seeds 1
```

Validacion corta con varias seeds por diseno:

```bash
./main.py --points 10 --seeds 10
```

Campana base con el numero de puntos configurado desde la linea de comandos:

```bash
./main.py --points 1024
```

Diagnostico de geometria en GPU, sin solapar con FFTHomPy:

```bash
./main.py --points 10 --seeds 1 --geometry-backend cupy --no-overlap
```

`--geometry-backend` acepta `numba`, `cupy` o `auto`. Para produccion, usar
`numba` salvo que se este diagnosticando exclusivamente la generacion SAM.

## Estructura Conservada

- `main.py`: configuracion editable, parser CLI y arranque del entorno.
- `requirements.txt`: dependencias runtime SAM/FFT.
- `REPRODUCIBILITY.md`: controles reproducibles del flujo actual.
- `pipeline/workflow.py`: perfil de entorno y ejecucion de campana.
- `pipeline/generate_sobol_designs.py`: diseno Sobol de geometria/material.
- `pipeline/sam_generator.py`: generador SAM continuo.
- `pipeline/rve_generator.py`: voxelizacion y auditoria de geometria.
- `pipeline/fft_solver.py`: homogenizacion FFTHomPy/CuPy.
- `pipeline/sobol_gpu.py`: orquestacion CPU/GPU y convergencia.
- `ffthompy_core/`: copia local del nucleo FFTHomPy usado por el solver.

## Relacion Con El Paper

Esta carpeta no intenta implementar el operador reducido final. Su papel es
producir datos full-order confiables para:

- validar periodicidad, convenciones Voigt/Kelvin y seis cargas 3D;
- generar RVEs 3D realistas de fibras cortas;
- estudiar estabilidad frente a contraste y casi incomprensibilidad;
- alimentar, desde fuera de `FFT/`, la construccion y validacion del operador
  constitutivo geometry-compiled.
