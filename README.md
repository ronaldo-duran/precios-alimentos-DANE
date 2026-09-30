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

`ingest` es idempotente: si el origen no trae fechas posteriores a lo consolidado,
no reescribe nada y lo deja anotado en `ingest_status.json` (`hay_datos_nuevos: false`),
para que el reentrenamiento pueda saltarse.

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

## Estado

**Fase 1 (datos y limpieza): completa.** 37 tests en verde.

Panel actual: **31 series** (8 productos × 4 plazas − 1 excluida), **348 semanas**
(2020-01-27 → 2026-09-21), 10.682 observaciones y 103 semanas faltantes marcadas.
**30 de 31 series son aptas** para modelar; `mango_tommy @ cucuta_cenabastos`
queda fuera por 16,8% de faltantes.

Pendiente: baselines y validación walk-forward, modelos e intervalos, registro,
app, CI y variables exógenas.
