# Evidencia técnica reproducible: segmentación de clientes con minería de datos e IA

Guía 10 · Minería de datos con IA. Esta carpeta reúne lo que pide la guía como *evidencia técnica*: dataset, código fuente,
evidencias de ejecución, resultados del modelo y visualizaciones, con los pasos para reproducir el proceso.

> **Los datos son sintéticos** (semilla 42), con errores sembrados a propósito para practicar la preparación de datos.

## Contenido

| Carpeta o archivo | Qué contiene | Qué pide la guía |
|---|---|---|
| `datos/clientes.csv` | Dataset de entrada: 1.230 registros × 11 variables | Dataset utilizado |
| `codigo/` | `generar_datos.py` (crea el dataset), `motor.py` (limpieza, K-Means, modelos), `reproducir.py` (todo el análisis sin interfaz), `verificar.py` (comprueba la reproducibilidad), `pruebas.py` (22 pruebas automáticas) y `app.py` (aplicación de escritorio) | Notebook o código fuente |
| `resultados/` | CSV y JSON con los datos limpios, los clientes segmentados, la trazabilidad y las métricas de los modelos | Resultados del modelo |
| `visualizaciones/` | 10 gráficos en PNG, uno por archivo | Visualizaciones |
| `evidencias/` | Salida real de las ejecuciones, entorno, checksums y capturas de la aplicación | Evidencias de ejecución |
| `pseint/` | Versión académica en pseudocódigo (`mineria_clientes.psc`, para abrir en PSeInt) y su diagrama de flujo exportado (SVG y PNG) | Diagramación |
| `requirements.txt` | Dependencias con versión exacta | Documentación para reproducir |

## Requisitos

- Probado con **Python 3.14.6** en Windows. No se probaron otras versiones de Python.
- Librerías de `requirements.txt` (numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, matplotlib 3.11.2). La interfaz usa `tkinter`, que viene con Python.

## Pasos para reproducir

Ejecute los comandos en una terminal, en esta carpeta.

1. **Crear un entorno e instalar dependencias**
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   En Linux o macOS, la activación es `source .venv/bin/activate`.

2. **(Opcional) Regenerar el dataset.** Ya viene incluido; con la semilla fija se obtiene un archivo idéntico (ver `evidencias/verificacion_dataset.txt`).
   ```
   cd datos
   python ../codigo/generar_datos.py
   cd ..
   ```

3. **Reproducir todo el análisis**
   ```
   cd codigo
   python reproducir.py
   ```
   Lee `datos/clientes.csv`, limpia, segmenta con K-Means, evalúa la clasificación y la regresión, y escribe `resultados/` y `visualizaciones/`. Imprime cada paso con sus cifras (ver `evidencias/ejecucion_reproducir.txt`).

4. **Verificar que los resultados coinciden con los entregados**
   ```
   python verificar.py
   ```
   Vuelve a ejecutar el análisis en una carpeta temporal y compara, archivo por archivo, los CSV y JSON con `resultados/`. Debe terminar con "11 de 11 archivos". Los hashes de referencia están en `evidencias/checksums_sha256.txt`.

5. **Ejecutar las pruebas automáticas**
   ```
   python pruebas.py
   ```
   Verifican la limpieza, la trazabilidad, K-Means, los modelos, la validación cruzada, la estabilidad de los grupos, el CRUD y la interfaz. Resultado esperado: `Ran 22 tests ... OK`.

6. **Abrir la aplicación de escritorio**
   ```
   python app.py
   ```
   Botón *Cargar CSV* y elegir `datos/clientes.csv`; después *Limpiar datos* y *Ejecutar K-Means*.

7. **(Opcional) Reconstruir el .exe portable** (requiere `requirements-extra.txt`)
   ```
   pip install -r ../requirements-extra.txt
   python -m PyInstaller --onefile --windowed --name MineriaDatosIA app.py
   ```

## Resultados esperados

| Medida | Valor |
|---|---|
| Registros original → limpio | 1.230 → 1.200 (30 duplicados eliminados) |
| Celdas modificadas (todas con motivo) | 963 |
| K-Means: k elegido / silhouette | 3 / 0,457 |
| Segmentos (clientes) | Frecuentes 413, Intermedios 485, Ocasionales 302 |
| Clasificación (Random Forest) | exactitud 0,777, F1 0,773 |
| Regresión (Random Forest) | R² 0,530, RMSE $518.295 |
| Validación cruzada (Random Forest) | exactitud 0,797 ± 0,020; R² 0,550 ± 0,027 |
| Estabilidad de K-Means (Rand ajustado) | mínimo 0,997 con 20 semillas; medio 0,990 con submuestras del 80 % |

## Qué hay en `resultados/`

- `clientes_limpio.csv`: datos tras la preparación. `clientes_segmentados.csv`: lo mismo más las columnas `cluster` y `segmento`.
- `cambios_trazabilidad.csv`: una fila por celda modificada (`id_cliente`, campo, valor original, valor limpio y motivo).
- `comparacion_original_vs_limpio.csv`: nulos y estadísticos por variable, antes y después.
- `perfil_segmentos.csv`: promedios por segmento. `metricas_clasificacion.csv`, `metricas_regresion.csv` e `importancia_variables.csv`: evaluación de los modelos.
- `validacion_cruzada.csv` (5 particiones) y `estabilidad_kmeans.csv` (20 semillas y 20 submuestras del 80 %, con el índice de Rand ajustado): comprueban que los resultados no dependen de una sola partición ni del azar.
- `resumen.json`: todas las cifras clave en un solo archivo.

## Visualizaciones

- `calidad_cambios_por_motivo.png`
- `clasificacion_importancia_variables.png`
- `clasificacion_matriz_confusion.png`
- `clustering_eleccion_k.png`
- `clustering_segmentos.png`
- `eda_distribucion_edad.png`
- `eda_frecuencia_por_ciudad.png`
- `eda_frecuencia_vs_valor.png`
- `eda_ventas_por_producto.png`
- `perfil_segmentos.png`

## Garantías y límites de la reproducibilidad

- Todo es determinista: el dataset usa la semilla 42 y los modelos `random_state = 0`.
- Los checksums de `resultados/` valen para las versiones de `evidencias/entorno.txt`. Con otras versiones de pandas o scikit-learn las cifras pueden variar en decimales o, en K-Means, en la numeración de los grupos (los segmentos se nombran por su frecuencia, no por su número).
- Los PNG pueden diferir byte a byte entre equipos; lo que se garantiza es el contenido de los CSV y JSON.
- Los datos son sintéticos: las conclusiones sirven para practicar el método, no describen una empresa real.
