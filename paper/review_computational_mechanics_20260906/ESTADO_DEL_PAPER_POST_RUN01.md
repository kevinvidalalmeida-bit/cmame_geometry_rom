# Estado del manuscrito tras la campaña Sobol--Ritz G00--G09 `run01`

Actualizado el 6 de septiembre de 2026. El único manuscrito activo es
`paper/main.tex` con su PDF compilado. La copia editorial anterior y los pilotos descartados fueron eliminados por
decisión del autor. Esta carpeta conserva solo el registro de la campaña de
producción y el estado de envío.

## Estado científico actual

La ruta principal está reproducida en las diez geometrías: Sobol adaptativo,
cinco monitores independientes, QR de energía de referencia y compilación afín
factorizada. La repetición `sobol_ritz_factorized_all_run01` terminó las diez
geometrías sin fallo ROM. Repite exactamente los intervalos del manuscrito:
6--16 materiales de entrenamiento, 11--21 FOM de construcción y rangos 36--96.

Los 200 pares de validación de esta repetición dan error Frobenius medio
`0.02169 %`, percentil 95 `0.07765 %` y máximo `0.13216 %`; 196/200 quedan
bajo `0.1 %`. Los cinco monitores cumplen `0.01 %` en cada geometría. La
tabla y los datos sin redondear están en
[TABLA_CAMPANA_SOBOL_RITZ_RUN01.md](TABLA_CAMPANA_SOBOL_RITZ_RUN01.md).

La repetición confirma la evidencia central del artículo. No sustituye por sí
sola la tabla de rendimiento de `main.tex`: esa tabla compara el compilador
factorizado contra POD L2 convencional en G08, G00 y G09, con dos repeticiones
y una frontera temporal común. Para no mezclar una repetición de producción con
una comparación controlada, se mantienen allí los promedios publicados.

## Qué ya está listo

- Problema mecánico: homogeneización elástica de diez microestructuras
  voxelizadas, hasta `240^3` voxeles.
- Método: espacio Sobol--Ritz adaptativo y operador afín reutilizable.
- Economía de datos: 111 materiales de entrenamiento y 50 monitores para toda
  la campaña; 161 FOM de construcción, cada uno con seis cargas.
- Precisión: monitores separados de 20 validaciones externas por geometría;
  error tensorial, energético, compliance y módulos direccionales.
- Rendimiento: comparación controlada POD/RB y presupuesto temporal G09; el
  ensamblaje afín pasa de 104.919 a 30.714 s en esa comparación.
- Reproducibilidad: configuración, semillas, comandos, resúmenes y tabla de
  la nueva repetición quedan archivados.

## Pendientes antes de enviar

1. **Estabilidad numérica de los bloques.** La repetición no tuvo fallos ROM y
   todos los `K_r` evaluados fueron definidos positivos, pero algunos
   `schur_eta` son ligeramente negativos con snapshots/contracciones
   `float32` (hasta aproximadamente `-2.19e-6` en G09). No se debe presentar
   una cota numérica estricta. Falta medir en G04 y G09 la recompilación final
   `float64` ya implementada, cuantificar su coste y decidir si se adopta o se
   limita el lenguaje a diagnósticos energéticos discretos.

2. **Aislar el mecanismo del ahorro.** La comparación actual prueba una mejora
   total de compilación en G00 y G09, pero agrupa factorización constitutiva,
   contracciones de bloques y reutilización de métrica. Una tabla de ablación
   pequeña y con la misma precisión reforzaría que el aporte es un compilador
   y no una diferencia incidental de implementación.

3. **Separar ROM de error espacial.** El cambio entre cinco y seis voxeles por
   diámetro llega a `0.910 %` en G09, mayor que el error ROM. Falta un nivel
   espacial adicional o un estudio más focalizado para explicar el límite de
   la referencia FOM discretizada.

4. **Demostración mecánica de muchas consultas.** Es el refuerzo editorial más
   útil, aunque no una condición formal: cribado de materiales o selección
   con una respuesta de rigidez definida, midiendo construcción y consultas,
   con comprobación FOM de los candidatos finales.

5. **Cierre de envío.** Congelar una versión de código y datos, completar la
   declaración de disponibilidad real, revisar referencias/figuras y reducir
   explicación teórica repetida. El PDF tiene 41 páginas en una columna; no
   es un impedimento formal, pero conviene concentrarlo.
