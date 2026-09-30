# Pronóstico de precios mayoristas de alimentos en Colombia (SIPSA–DANE)

> **Precios MAYORISTAS (SIPSA–DANE), no precios al consumidor final.**
> Los pronósticos son estimaciones con incertidumbre, no garantías.

Pipeline reproducible que pronostica precios mayoristas semanales por producto y plaza,
con intervalos de predicción, validación walk-forward, registro de pronósticos en vivo
y una app de Streamlit.

Proyecto personal con fines de aprendizaje. El objetivo no es "ganarle al mercado",
sino ilustrar buenas prácticas de series de tiempo y **comunicar con honestidad dónde
el modelo sirve y dónde falla**.

---

## Fuente de datos

### Lo que NO funciona (y por qué)

El dataset de datos.gov.co **`gkqq-k3k5`** ("SIPSA – Precios Mayoristas 2013–2024")
**no es consultable por la API de Socrata**. Es un `assetType: "federated_href"`:
una ficha de catálogo que apunta a otro sitio, sin filas propias.

```console
$ curl "https://www.datos.gov.co/resource/gkqq-k3k5.csv?\$limit=3"
{"error" : true, "message" : "no row or column access to non-tabular tables"}
```

Las **25** entradas de SIPSA en datos.gov.co son `federated_href`. No hay `$limit`,
`$offset` ni `sodapy` que sirvan. Esa vía está cerrada.

### Lo que sí funciona: el servicio web oficial del DANE

El DANE publica un **servicio SOAP público**, actualizado **a las 2:00 p.m. todos los días**:

```
https://appweb.dane.gov.co/sipsaWS/SrvSipsaUpraBeanService?WSDL
```

No requiere autenticación y **no es scraping**: es una API documentada por el DANE en su
página *"Servicio web para consulta de la base de datos de SIPSA"*.

Expone cuatro operaciones de precios. Hay un intercambio claro entre **amplitud e historia**:

| Operación | Registros | Rango | Frecuencia | Productos | Plazas |
|---|---:|---|---|---:|---:|
| `promediosSipsaParcial` | 697.816 | 2020-02-01 → hoy | diaria (6 días/sem) | 36 | 26 |
| `promediosSipsaCiudad` | 387.063 | 2020-02-03 → hoy | diaria | 33 | 21 |
| `promediosSipsaSemanaMadr` | 225.640 | 2025-10-04 → hoy | semanal | 351 | 80 |
| `promediosSipsaMesMadr` | 189.312 | 2023-02-01 → 2026-07 | mensual | 360 | 147 |

**Este proyecto usa `promediosSipsaParcial`**: es la única con historia suficiente
(349 semanas) para estimar estacionalidad anual y hacer walk-forward con varios orígenes.
Devuelve el precio **ya en pesos por kilogramo** (`promedioKg`, `minimoKg`, `maximoKg`),
así que no hay conversión de unidades ni el riesgo asociado.

### Limitaciones que debes conocer

1. **Sin filtros del lado del servidor.** Ninguna operación acepta parámetros: cada
   llamada descarga el histórico completo (~310 MB de XML, ~55 s). De ahí el caché
   en `data/raw/` y la consolidación incremental.
2. **Cobertura acotada.** `promediosSipsaParcial` solo cubre frutas, verduras y
   tubérculos. **No hay arroz, fríjol, huevo, panela ni maíz**: esos productos solo
   existen en el endpoint semanal, con apenas ~50 semanas de historia.
3. **La historia empieza en 2020-02.** No hay datos previos por esta vía. Los
   microdatos del DANE (catálogo 776) cubren 2013–2024 pero son **mensuales** y con un
   esquema que cambia entre años, así que no alimentan un modelo semanal directamente.
4. **La semana en curso está incompleta.** El pipeline descarta semanas parciales;
   si no, el último punto queda sesgado.
5. **Precios mayoristas.** No son lo que paga un hogar en la tienda. El margen
   minorista varía por ciudad, canal y producto.

Licencia de los datos: DANE, Creative Commons BY-SA 4.0.

---

## Alcance de la v1

**8 productos** con comportamiento deliberadamente distinto (CV semanal entre 33% y 57%)
y **4 plazas** de mejor cobertura. Todo configurable en `config/products.yaml`.

Productos: Papa negra\*, Papa criolla, Tomate\*, Cebolla cabezona blanca,
Plátano hartón verde, Yuca\*, Zanahoria, Mango tommy.

Plazas: Bogotá (Corabastos), Medellín (Central Mayorista de Antioquia),
Bucaramanga (Centroabastos), Cúcuta (Cenabastos).

> El asterisco forma parte del nombre en SIPSA y denota un agregado de variedades.

---

## Cómo ejecutar

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
make pipeline
```

Si no tienes `make` (habitual en Windows), cada etapa se invoca directamente:

```bash
.venv/Scripts/python -m precios.pipeline.ingest
.venv/Scripts/python -m precios.pipeline.clean
.venv/Scripts/python -m precios.pipeline.evaluate
.venv/Scripts/python -m pytest
```

Los datos **no se versionan**: `data/` está en `.gitignore`. `make ingest` los
regenera desde el servicio del DANE en poco más de un minuto.

### Etapas

| Etapa | Entrada | Salida |
|---|---|---|
| `ingest` | SOAP del DANE | `data/raw/<operacion>_<fecha>.parquet`, `data/processed/diario.parquet`, `ingest_status.json` |
| `validate` | `diario.parquet` | detiene el pipeline si el esquema o los valores no cuadran |
| `clean` | `diario.parquet` | `data/processed/semanal.parquet`, `reporte_series.csv` |
| `tune` | `semanal.parquet` | `config/lgbm_params.yaml`, `busqueda_hiperparametros.csv` |
| `evaluate` | `semanal.parquet` | `backtest.parquet`, `metricas_por_horizonte.csv`, `metricas_por_serie.csv`, `cobertura_intervalos.csv`, `degradacion_por_regimen.csv`, `semanas_choque.csv`, `importancia_features.csv` |
| `train` | `semanal.parquet` | `models/<fecha>_<hash>/` + `models/latest.json` |
| `forecast` | `models/latest.json` | `forecasts_log.csv`, `alertas_log.csv` (append-only) |
| `reconcile` | los logs + `semanal.parquet` | `reconciliacion.csv`, `desempeno_en_vivo.csv`, `alertas_reconciliadas.csv`, `desempeno_alertas.csv` |

`ingest` es idempotente: si el origen no trae fechas posteriores a lo consolidado,
no reescribe nada y lo deja anotado en `ingest_status.json` (`hay_datos_nuevos: false`),
para que el reentrenamiento pueda saltarse.

`make pipeline` ejecuta el **ciclo semanal**: `ingest → clean → train → forecast
→ reconcile`. No incluye `evaluate`, que son ~30 minutos de backtest y cuyas
métricas no cambian de forma útil semana a semana; se corre a mano cuando se
toca el modelado.

### Esquema de `semanal.parquet`

`semana` (lunes ISO), `producto_id`, `plaza_id`, `producto`, `plaza`, `ciudad`,
`grupo`, `precio_kg`, `precio_kg_min`, `precio_kg_max`, `n_dias`, y los flags de
calidad `flag_outlier`, `flag_pocos_dias`, `flag_faltante`.

---

## Decisiones de limpieza

Todas viven en `config/cleaning.yaml` y se pueden cambiar sin tocar código.

- **Mediana y no promedio** para pasar de diario a semanal. El histórico tiene
  errores de digitación de un solo día (Tomate\* en Popayán a 250 COP/kg frente a
  una mediana de ~2.400). La mediana los absorbe; el promedio los propaga.
- **Se descarta la semana en curso.** SIPSA cotiza ~6 días por semana; una
  corrida a mitad de semana vería 2 días y produciría un último punto sesgado.
- **Los outliers se marcan, nunca se borran.** Un salto grande puede ser un error
  de captura *o* un choque real (un paro, una helada). Distinguirlos es el
  objetivo del análisis, así que se conservan con un flag.
- **Las semanas faltantes se marcan, no se imputan.** Decidir si interpolar
  pertenece a la construcción de features, donde se sabe cuál es el origen del
  pronóstico y por tanto qué información es legítimo usar.
- **Detección de outliers con MAD sobre log-retornos**, más un piso de relevancia
  del 15%. Sin ese piso, una serie casi plana tiene MAD ≈ 0 y el z-score marca
  variaciones del 1% como anomalías.

---

## Modelos

| Modelo | Tipo | Intervalos |
|---|---|---|
| `naive` | último valor | no |
| `naive_estacional` | valor de la semana del año anterior (lag 52) | no |
| `promedio_movil_4` / `_8` | media de las últimas k semanas | no |
| `naive_deriva` | naive más tendencia media | no |
| `ets` | suavizado exponencial (statsforecast) | no |
| `auto_arima` | ARIMA con órdenes automáticos (statsforecast) | no |
| `lgbm_cuantil` | LightGBM global, regresión cuantílica | q10–q90 |
| `lgbm_conformal` | LightGBM global, conformal split | calibrados |

### Por qué ETS y ARIMA van sin estacionalidad

Con datos semanales el periodo anual es 52. ETS y ARIMA estacionales con m=52
tienen que estimar decenas de parámetros estacionales, son numéricamente
frágiles y muy lentos. Y sobre todo: el naive estacional ya fracasó con
MASE ~5,2, o sea que **la evidencia dice que no hay ciclo anual estable que
explotar**. Forzar un componente estacional añadiría varianza sin señal.

### Por qué el objetivo es el log-cambio

El modelo global ve aguacate a 7.000 COP/kg y zanahoria a 1.000 en las mismas
filas. Si predijera niveles, gastaría su capacidad en distinguir productos.
Prediciendo `log(y[t+h] / y[t])`, todas las series hablan el mismo idioma y
**"empatar con el naive" equivale exactamente a predecir cambio cero**, que es
la comparación que interesa. El nivel se reconstruye como `y[t] · exp(pred)`.

Se usa estrategia **directa** (un modelo por horizonte) en vez de iterar un
modelo de un paso: evita acumular error de forma opaca y deja que cada horizonte
aprenda su dinámica (a 1 semana manda la inercia; a 4, la reversión a la media).

### Dos vías de incertidumbre, y por qué se comparan

- **Regresión cuantílica**: LightGBM aprende directamente q10 y q90. Es barata,
  pero nada garantiza que su cobertura empírica sea la nominal.
- **Conformal split**: el modelo se entrena en un tramo y el ancho del intervalo
  se calibra con **residuos fuera de muestra** de un tramo posterior que el
  modelo nunca vio. Bajo intercambiabilidad garantiza la cobertura marginal.

Con series de tiempo la intercambiabilidad no se cumple del todo (los residuos
están correlacionados y el régimen cambia), así que la garantía es aproximada.
Por eso el proyecto **mide** la cobertura empírica en lugar de darla por buena.

### Búsqueda de hiperparámetros

Rejilla fija de seis configuraciones, definida de antemano en
`src/precios/models/busqueda.py`. Los orígenes se parten en dos: la **mitad
antigua** elige la configuración, la **mitad reciente** se reserva para
reportar. Buscar y reportar sobre los mismos orígenes produce una mejora que no
existe fuera de la muestra. Se registran **todas** las configuraciones probadas
en `data/processed/busqueda_hiperparametros.csv`, no solo la ganadora, y la
elegida queda versionada en `config/lgbm_params.yaml` con su procedencia.

**Resultado de la búsqueda, con su decepción incluida:** ganó la configuración
más pequeña y regularizada (`num_leaves=15`, `n_estimators=300`,
`min_child_samples=60`), coherente con tener solo ~9.000 filas de
entrenamiento. Pero **la rejilla salió casi plana**: entre la mejor y la peor
configuración hay un 2,8% de MASE (1,8355 frente a 1,8871). Trasladado a la
evaluación final, afinar mejoró el MASE de 1,156 a 1,154 en h=1 y de 2,324 a
2,273 en h=4. Es decir: **ajustar hiperparámetros aquí no es donde está el
problema**, y seguir buscando más allá de esta rejilla sería perseguir ruido.

---

## Resultados: los baselines

Validación rolling-origin con ventana expansiva: 61 orígenes por serie, primer
origen tras 104 semanas, un origen nuevo cada 4 semanas, horizontes 1 a 4.
**36.495 pronósticos** sobre 30 series. Nunca hay split aleatorio.

MASE por modelo y horizonte (menor es mejor; la escala es el naive de un paso
dentro del train, así que el MASE crece con el horizonte por construcción):

| Modelo | h=1 | h=2 | h=3 | h=4 |
|---|---:|---:|---:|---:|
| **naive** | **1,179** | **1,656** | **2,081** | **2,348** |
| naive con deriva | 1,182 | 1,663 | 2,097 | 2,371 |
| promedio móvil 4 | 1,617 | 1,912 | 2,326 | 2,581 |
| promedio móvil 8 | 2,169 | 2,395 | 2,766 | 2,999 |
| naive estacional (52) | 5,359 | 5,193 | 5,195 | 5,277 |

**Ningún baseline le gana al naive simple, en ningún horizonte.** Conviene
decirlo sin adornos:

- El **naive estacional es pésimo** (MASE ~5,2). Estos precios no tienen un
  ciclo anual estable que se repita en la misma semana: el nivel se mueve por
  oferta de corto plazo, y el periodo 2020-2026 incluye dos rupturas de régimen
  (COVID y el paro de 2021) que rompen cualquier analogía con el año anterior.
- Los **promedios móviles pierden** porque suavizan justo la información que
  importa: el último precio. Cuanto más larga la ventana, peor (el de 8 semanas
  es casi el doble de malo que el naive en h=1).
- La **deriva no aporta nada**: empata con el naive y lo empeora un pelo. No hay
  tendencia lineal explotable.

Dificultad por producto, medida como sMAPE del naive:

| Producto | sMAPE h=1 | sMAPE h=4 |
|---|---:|---:|
| Yuca | 5,1% | 9,3% |
| Plátano hartón verde | 5,5% | 11,7% |
| Papa negra | 7,9% | 15,0% |
| Cebolla cabezona blanca | 10,0% | 23,9% |
| Papa criolla | 11,5% | 19,3% |
| Zanahoria | 11,7% | 20,6% |
| Mango tommy | 11,8% | 32,6% |
| **Tomate** | **17,2%** | **28,1%** |

El tomate es el más difícil a una semana; el mango tommy es el que peor se
degrada al alejarse el horizonte (32,6% a cuatro semanas), consistente con su
estacionalidad de cosecha. Yuca y plátano son los más predecibles.

Esta tabla es la vara contra la que se miden ETS, ARIMA y LightGBM.

---

## Resultados: ¿alguien le gana al naive?

61 orígenes, 30 series, horizontes 1–4. **65.691 pronósticos** de 9 modelos,
todos medidos sobre exactamente los mismos orígenes.

| Modelo | h=1 | h=2 | h=3 | h=4 |
|---|---:|---:|---:|---:|
| `lgbm_cuantil` | **1,154** | **1,562** | **1,992** | **2,273** |
| `auto_arima` | 1,177 | 1,603 | 2,032 | 2,321 |
| `naive` | 1,179 | 1,656 | 2,081 | 2,348 |
| `naive_deriva` | 1,182 | 1,663 | 2,097 | 2,371 |
| `ets` | 1,190 | 1,654 | 2,085 | 2,382 |
| `lgbm_conformal` | 1,227 | 1,674 | 2,146 | 2,433 |
| `promedio_movil_4` | 1,617 | 1,912 | 2,326 | 2,581 |
| `promedio_movil_8` | 2,169 | 2,395 | 2,766 | 2,999 |
| `naive_estacional` | 5,359 | 5,193 | 5,195 | 5,277 |

Sí: LightGBM y AutoARIMA le ganan al naive en los cuatro horizontes. Pero la
mejora es **pequeña (1–5%)** y, sobre todo, el promedio esconde lo importante.

### La mejora la produce casi un solo producto

| Producto | h=1 naive | h=1 lgbm | mejora h=1 | mejora h=4 |
|---|---:|---:|---:|---:|
| **Mango tommy** | 1,170 | **0,915** | **+21,8%** | **+35,6%** |
| Tomate | 1,245 | 1,158 | +7,0% | +8,2% |
| Yuca | 1,140 | 1,130 | +0,9% | −11,2% |
| Zanahoria | 1,238 | 1,245 | −0,6% | +1,0% |
| Papa criolla | 1,194 | 1,210 | −1,3% | +3,4% |
| Cebolla cabezona blanca | 1,009 | 1,024 | −1,5% | −7,4% |
| Papa negra | 1,202 | 1,222 | −1,7% | −6,9% |
| Plátano hartón verde | 1,238 | 1,277 | −3,2% | −0,8% |

Quitando mango tommy y tomate, **LightGBM empata o pierde contra el naive**.
Y contando combinaciones individuales de (serie × horizonte), solo gana en
**70 de 120 (58,3%)**. A h=1 gana en el **46,7%**: a un horizonte de una semana
**pierde más veces de las que gana**.

Las features más importantes explican por qué: `ret_52`, `ret_26` y `rel_ma52`
dominan. Es decir, el modelo sí encuentra estructura anual, pero **relativa**
(dónde está el precio frente a su propio nivel de hace un año), no de nivel
absoluto, que es lo que copia el naive estacional y por lo que fracasa. Y solo
el mango tommy —fruta de cosecha marcada— tiene esa estructura con fuerza.

### El régimen reciente es algo más favorable

Sobre la mitad reservada de orígenes (224–344, los más recientes), la ventaja
se concentra en horizontes largos en vez de cortos:

| Modelo | h=1 | h=2 | h=3 | h=4 |
|---|---:|---:|---:|---:|
| `lgbm_cuantil` | +0,6% | +4,6% | +7,0% | +7,1% |
| `lgbm_conformal` | −1,1% | +4,4% | +6,5% | +5,7% |
| `auto_arima` | +0,2% | +5,0% | +4,3% | +3,0% |
| `ets` | −0,1% | +0,4% | −0,2% | −1,8% |

(diferencia porcentual de MASE frente al naive; positivo = mejor)

**A una semana no hay nada que hacer: el último precio es la mejor predicción.**
A 2–4 semanas hay un margen real pero modesto, del 3 al 7%.

---

## Resultados: los intervalos de predicción

Esta es la parte donde la diferencia entre métodos es grande y clara.
Nivel nominal: **80%**.

| Modelo | h=1 | h=2 | h=3 | h=4 |
|---|---:|---:|---:|---:|
| `lgbm_conformal` | **81,2%** | **81,2%** | **79,3%** | **78,6%** |
| `lgbm_cuantil` | 74,2% | 71,0% | 67,7% | 64,7% |

**La regresión cuantílica miente, y cada vez más.** Su intervalo "del 80%" cubre
el 74,2% a una semana y solo el **64,7%** a cuatro. La calibración conformal, en
cambio, se queda entre el 78,6% y el 81,2% en todos los horizontes.

El precio de esa honestidad es doble:

1. **Intervalos más anchos**: ±33,3% del precio a h=1 frente a ±26,6% de la
   cuantílica; a h=4, ±63,8% frente a ±46,0%.
2. **Peor pronóstico puntual**: `lgbm_conformal` tiene MASE 1,227 frente a 1,154
   de `lgbm_cuantil` a h=1, porque reserva 26 semanas para calibrar y entrena con
   menos datos. Calibrar bien cuesta precisión.

---

## Análisis de fallos: dónde se rompe

`config/episodios.yaml` nombra los episodios, pero la detección es data-driven:
una semana es "de choque" si el movimiento **mediano** de todo el panel supera
el percentil 90 de su propia distribución. Salen **35 semanas de 348**, y las
más violentas se explican solas:

| Semana | Movimiento mediano | Episodio |
|---|---:|---|
| 2021-05-03 | **63,7%** | Paro nacional y bloqueos de vías |
| 2024-09-02 | 45,3% | Alta volatilidad 2024-Q3 |
| 2020-03-16 | 41,5% | Inicio de la cuarentena |
| 2024-09-09 | 34,0% | Alta volatilidad 2024-Q3 |
| 2021-05-10 | 33,2% | Paro nacional |

Las siete semanas más volátiles del histórico caen todas dentro de episodios
conocidos. La detección no sabía nada de ellos.

### Todos los modelos se rompen igual

MASE a h=1, semanas normales frente a semanas de choque:

| Modelo | Normal | Choque | Factor |
|---|---:|---:|---:|
| `naive` | 1,10 | 3,56 | ×3,24 |
| `auto_arima` | 1,10 | 3,43 | ×3,12 |
| `lgbm_cuantil` | 1,07 | 3,51 | ×3,27 |
| `lgbm_conformal` | 1,15 | 3,48 | ×3,02 |

(1.764 pronósticos en semanas normales frente a 60 en semanas de choque)

**Ningún modelo aguanta un choque.** Todos triplican su error, y las diferencias
entre ellos son ruido. Si un paro bloquea las vías, el precio de la semana
siguiente no está en la historia.

### Y los intervalos también se rompen — este es el hallazgo incómodo

Cobertura del intervalo conformal del 80%, por régimen:

| h | Semanas normales | Semanas de choque |
|---|---:|---:|
| 1 | 82,4% | **45,0%** |
| 2 | 82,4% | 68,5% |
| 3 | 79,5% | 75,0% |
| 4 | 80,2% | 66,8% |

La cobertura global del 81% es un **promedio que esconde el fallo justo cuando
más importaría**. En una semana de choque el intervalo del 80% falla más de la
mitad de las veces a un horizonte de una semana.

No es un error de implementación: la predicción conformal garantiza cobertura
**marginal**, no condicional. Promete acertar el 80% de las veces en promedio,
no el 80% en cada régimen. Con series de tiempo, donde los choques rompen la
intercambiabilidad, esa distinción deja de ser teórica.

**La lectura práctica: estos intervalos son útiles en condiciones normales y no
son de fiar durante un choque, que es exactamente cuando alguien querría
consultarlos.** La app lo dice donde se ve.

---

## Registro de modelos y de pronósticos en vivo

### Registro de modelos

Cada `make train` produce `models/<fecha>_<hash>/` con el modelo, un
`metadata.json` y los diagnósticos. El hash resume **datos + hiperparámetros +
semilla + commit**, así que reentrenar con las mismas entradas cae en el mismo
directorio en vez de llenar el registro de copias. `models/latest.json` apunta
a la versión vigente (un archivo, no un symlink: en Windows los enlaces
simbólicos necesitan permisos especiales y romperían el pipeline en silencio).

La `metadata.json` responde la única pregunta que importa meses después, cuando
un pronóstico sale raro: *¿con qué datos, qué código y qué semilla se produjo
esto?* Incluye rango de datos, series, productos, plazas, horizontes, nivel de
intervalo, semilla, hiperparámetros, commit, radios conformales y las métricas
del último backtest.

Se versiona la metadata pero **no** los pesos (`.joblib`, 1,8 MB): se regeneran
con `make train`.

### Se registra el modelo conformal, no el cuantílico

Es una decisión deliberada con un coste explícito. `lgbm_conformal` tiene peor
pronóstico puntual que `lgbm_cuantil` (MASE 1,227 frente a 1,154 en h=1), porque
reserva 26 semanas para calibrar y entrena con menos datos. Pero sus intervalos
son los únicos que cubren lo que prometen (79–81% frente a 65–74%). Para algo
que publica bandas de incertidumbre, **un intervalo honesto vale más que un 6%
de MASE**.

### El log en vivo

`forecasts_log.csv` es **append-only**. Se escribe *antes* de que exista el dato
real y nunca se reescribe una fila pasada. Si se vuelve a correr con el mismo
origen y modelo no se añade nada; si cambia el modelo o avanza el origen, se
añade una fila nueva y la anterior queda intacta. Si el esquema del archivo
cambia, el pipeline **falla con instrucciones** en vez de migrar en silencio.

Se registra también el pronóstico del **naive** para los mismos puntos: sin la
referencia, el error acumulado no dice nada.

Esta es la parte del proyecto que no se puede falsear. Un backtest se puede
repetir hasta que salga bien; un pronóstico escrito antes de los hechos, no.

### `vivo` frente a `backfill`

El log en vivo empieza vacío por definición: su primera fila no se resuelve
hasta que pasa la semana. Para que la app tenga algo que mostrar desde el día
uno, se siembran los 14.598 pronósticos del walk-forward en
`forecasts_backfill.csv`, marcados `procedencia="backfill"`.

Son igual de out-of-sample —ningún modelo vio datos posteriores a su origen—
pero **no se escribieron antes de los hechos**, así que tienen valor descriptivo
y no probatorio. Las tablas de desempeño los reportan **siempre por separado** y
nunca los mezclan. Viven en archivos distintos por higiene de git: el sembrado
son 2 MB estáticos, el log en vivo crece ~35 KB por semana.

Estado actual: **240 pronósticos en vivo, todos pendientes.** El primero se
resuelve el 2026-09-28. Es lo honesto que se puede decir hoy.

---

## Alertas de alza

`config/alertas.yaml`. Probabilidad de que el precio suba más de un umbral en
las próximas 2 semanas.

### El umbral es por producto, y esa es toda la cuestión

Con un umbral plano del 10%, la alerta se dispararía el **13,7% de las semanas
en yuca y el 36,6% en tomate**. La misma etiqueta significando cosas distintas,
y una alerta que suena un tercio del tiempo no informa de nada.

Cada umbral es el percentil 75 de las alzas históricas a 2 semanas de ese
producto, redondeado al 5% con piso del 10%:

| Producto | Umbral | Tasa base |
|---|---:|---:|
| Yuca | 10% | 13,7% |
| Plátano hartón verde | 15% | 10,3% |
| Papa negra | 15% | 16,6% |
| Papa criolla | 25% | 12,5% |
| Zanahoria | 25% | 13,2% |
| Cebolla cabezona blanca | 30% | 11,8% |
| Mango tommy | 35% | 12,6% |
| Tomate | 40% | 12,4% |

La tasa base queda entre el 10% y el 17% en todos los productos, frente al
rango 14–38% del umbral plano. Ahora "alerta activa" significa lo mismo en
todas partes.

### De dónde sale la probabilidad

De los **residuos de calibración conformal con signo**: la distribución
predictiva empírica alrededor del pronóstico, ya validada fuera de muestra. No
se asume normalidad.

Los dos horizontes se acoplan de forma **comonotónica** (se evalúan en el mismo
nivel de cuantil). Suponer independencia inflaría la probabilidad —los errores
de h=1 y h=2 están muy correlacionados: si el precio se dispara, se dispara
para ambos— y tomar solo el máximo por horizonte ignoraría que dos
oportunidades son más que una.

**Advertencia que hereda de los intervalos:** estas probabilidades se apoyan en
la calibración conformal, cuya cobertura se desploma en semanas de choque. La
alerta es informativa en régimen normal y **no es de fiar durante un choque**.


---

## Estado

**Fase 1 (datos y limpieza): completa.** 37 tests en verde.

Panel actual: **31 series** (8 productos × 4 plazas − 1 excluida), **348 semanas**
(2020-01-27 → 2026-09-21), 10.682 observaciones y 103 semanas faltantes marcadas.
**30 de 31 series son aptas** para modelar; `mango_tommy @ cucuta_cenabastos`
queda fuera por 16,8% de faltantes.

**Fase 2 (baselines y validación walk-forward): completa.** 74 tests en verde,
incluidos los de ausencia de leakage y construcción de folds.

**Fase 3 (modelos estadísticos, LightGBM global e intervalos): completa.**
115 tests en verde. 65.691 pronósticos de 9 modelos sobre los mismos orígenes.
Conclusión corta: a una semana el naive es imbatible; a 2–4 semanas hay un
3–7% de margen; los intervalos conformales están bien calibrados en promedio y
fallan en los choques.

**Fase 4 (registro de modelos, log en vivo y alertas): completa.** 166 tests
en verde. Modelo registrado con trazabilidad completa, log de pronósticos
append-only y alertas con umbral por producto.

Pendiente: app de Streamlit, CI y variables exógenas.
