# Campaña Sobol--Ritz factorizada G00--G09: run01

Ejecución finalizada el 6 de septiembre de 2026 con `snapshot32` en entrenamiento y monitores, FOM `reference` para la validación, 1.024 candidatos Sobol, cinco monitores, objetivo `1e-4`, QR de energía de referencia y ensamblaje afín factorizado.

## Control de integridad

- Geometrías completas sin fallo ROM: 10/10.
- Monitores bajo 0.01 % en las diez geometrías: sí.
- Validación: 200 pares; media 0.02169 %, p95 0.07765 %, máximo 0.13216 %.
- Pares de validación bajo 0.01 %: 88/200; bajo 0.1 %: 196/200.
- Materiales de entrenamiento: 6--16; FOM de construcción (entrenamiento + monitores): 11--21; rangos: 36--96.
- Total de la campaña: 111 materiales de entrenamiento + 50 monitores = 161 FOM de construcción (966 casos de carga).
- Suma secuencial de construcción hasta operador congelado: 465.823 s. La corrida completa sumó 2126.064 s porque incluye validación FOM y benchmarks ROM; no se debe usar como tiempo de construcción.

## Tabla actualizada de construcción y validación

| Geometría | Malla | Entrenamiento | Monitores | FOM construcción | Rango | Monitor máx. [%] | Validación media [%] | Validación máx. [%] | ≤0.01 % [de 20] | Construcción [s] | Ensamblaje [s] | FOM mediano [s] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G00 | 150×150×150 | 11 | 5 | 16 | 66 | 0.00962 | 0.02373 | 0.09871 | 7 | 30.103 | 5.916 | 1.552 |
| G01 | 150×150×150 | 6 | 5 | 11 | 36 | 0.00906 | 0.01829 | 0.05248 | 7 | 18.064 | 2.382 | 1.501 |
| G02 | 150×150×150 | 15 | 5 | 20 | 90 | 0.00952 | 0.02437 | 0.10317 | 10 | 40.555 | 9.574 | 1.544 |
| G03 | 60×60×60 | 11 | 5 | 16 | 66 | 0.00808 | 0.01648 | 0.06047 | 10 | 3.872 | 1.337 | 0.123 |
| G04 | 240×240×240 | 12 | 5 | 17 | 72 | 0.00909 | 0.02095 | 0.07929 | 9 | 119.326 | 18.342 | 6.233 |
| G05 | 150×150×150 | 11 | 5 | 16 | 66 | 0.00977 | 0.02434 | 0.10458 | 9 | 29.965 | 5.895 | 1.533 |
| G06 | 150×150×150 | 11 | 5 | 16 | 66 | 0.00921 | 0.02597 | 0.10833 | 9 | 30.033 | 5.893 | 1.531 |
| G07 | 150×150×150 | 12 | 5 | 17 | 72 | 0.00801 | 0.01873 | 0.07006 | 10 | 32.692 | 6.796 | 1.540 |
| G08 | 60×60×60 | 6 | 5 | 11 | 36 | 0.00921 | 0.01786 | 0.07948 | 8 | 2.076 | 0.433 | 0.120 |
| G09 | 240×240×240 | 16 | 5 | 21 | 96 | 0.00975 | 0.02615 | 0.13216 | 9 | 159.137 | 30.749 | 6.091 |

Construcción es `offline_ready_wall_s`: desde la preparación hasta el operador Sobol--Ritz congelado, con entrenamiento, monitores, ensamblaje y congelamiento. No incluye los 20 FOM de validación ni la medición de consultas. Los tiempos son una nueva campaña de una repetición y no sustituyen la tabla controlada de dos repeticiones frente a POD convencional en `main.tex`.

La tabla CSV con valores sin redondear está en `TABLA_CAMPANA_SOBOL_RITZ_RUN01.csv`.
