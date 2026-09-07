# Enfoque y pendientes para *Computational Mechanics*

El artículo activo es `paper/main.tex`. Su contribución es un compilador de
operadores afines reutilizables para homogeneización elástica FFT en geometrías
voxelizadas fijas. Sobol adaptativo decide cuántos materiales FOM se necesitan;
la compilación factorizada reduce el trabajo de ensamblar los operadores
reducidos. El mismo mapa de Schur sirve para RB de espacio completo y POD, por
lo que la aportación no es un nuevo principio de proyección.

## Evidencia incorporada

La campaña reproducible `sobol_ritz_factorized_all_run01` cubre G00--G09 con
1.024 candidatos Sobol, cinco monitores independientes, objetivo `1e-4`, QR de
energía de referencia y compilación factorizada. Requirió 6--16 materiales de
entrenamiento, 11--21 FOM de construcción por geometría y rangos 36--96. En
sus 200 validaciones reservadas obtuvo media `0.02169 %`, percentil 95
`0.07765 %` y máximo `0.13216 %`. La tabla, la figura de validación y las
métricas mecánicas de `main.tex` usan esta campaña. Los valores sin redondear
están en [TABLA_CAMPANA_SOBOL_RITZ_RUN01.md](TABLA_CAMPANA_SOBOL_RITZ_RUN01.md).

La comparación POD/RB queda separada porque mide el compilador factorizado y
la ruta convencional bajo la misma frontera temporal, con dos repeticiones en
G08, G00 y G09. Sustituirla con una sola repetición de producción eliminaría
el control del comparador.

## Pendientes de mayor valor científico

1. Medir la recompilación final en `float64` para G04 y G09, cuantificando
   coste y efecto sobre los pequeños `schur_eta` negativos de `float32`.
2. Añadir una ablación controlada que separe factorización constitutiva,
   actualización incremental y reutilización de métrica, con precisión y FOM
   idénticos para cada variante.
3. Reforzar la separación entre error ROM y error espacial: G09 cambia hasta
   `0.910 %` entre cinco y seis vóxeles por diámetro.
4. Incorporar un caso mecánico de muchas consultas, como cribado de materiales
   con verificación FOM de las decisiones finales.
5. Cerrar disponibilidad de código/datos, bibliografía y presentación antes
   del envío.

La validación es sobre materiales reservados y el FOM voxelizado; no se afirma
certificación sobre todo el dominio constitutivo.
