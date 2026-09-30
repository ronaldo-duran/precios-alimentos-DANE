# Pipeline de precios mayoristas SIPSA-DANE.
# En Windows se ejecuta con Git Bash o WSL; PY apunta al venv creado por uv.

PY := .venv/Scripts/python
ifeq ($(OS),)
PY := .venv/bin/python
endif

.PHONY: help setup ingest validate clean tune evaluate train forecast reconcile pipeline test lint app

help:
	@echo "make setup     - crea el entorno e instala dependencias"
	@echo "make ingest    - descarga y consolida los datos del DANE"
	@echo "make validate  - valida el esquema diario"
	@echo "make clean     - normaliza, agrega a semanal y escribe el panel"
	@echo "make tune      - busqueda limitada de hiperparametros de LightGBM"
	@echo "make evaluate  - backtest walk-forward de todos los modelos"
	@echo "make train     - entrena el modelo de produccion y lo registra"
	@echo "make forecast  - pronostica y anexa al log en vivo (append-only)"
	@echo "make reconcile - compara el log con los precios reales publicados"
	@echo "make pipeline  - ciclo semanal: ingest + clean + train + forecast + reconcile"
	@echo "                 (make evaluate es el backtest completo, ~30 min; no va semanal)"
	@echo "make test      - corre los tests"
	@echo "make lint      - ruff"
	@echo "make app       - lanza la app de Streamlit"

setup:
	uv venv --python 3.12
	uv pip install -e ".[dev,models,app]"

ingest:
	$(PY) -m precios.pipeline.ingest

validate:
	$(PY) -m precios.pipeline.clean --solo-validar

clean:
	$(PY) -m precios.pipeline.clean

tune:
	$(PY) -m precios.pipeline.tune

evaluate:
	$(PY) -m precios.pipeline.evaluate

train:
	$(PY) -m precios.pipeline.train

forecast:
	$(PY) -m precios.pipeline.forecast

reconcile:
	$(PY) -m precios.pipeline.reconcile

# El ciclo semanal NO reejecuta el backtest completo: 61 origenes por 9 modelos
# son ~30 minutos y las metricas no cambian de forma util semana a semana.
# `make evaluate` se corre cuando se toca el modelado.
pipeline: ingest clean train forecast reconcile

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

app:
	$(PY) -m streamlit run app/Inicio.py
