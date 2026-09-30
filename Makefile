# Pipeline de precios mayoristas SIPSA-DANE.
# En Windows se ejecuta con Git Bash o WSL; PY apunta al venv creado por uv.

PY := .venv/Scripts/python
ifeq ($(OS),)
PY := .venv/bin/python
endif

.PHONY: help setup ingest validate clean evaluate pipeline test lint app

help:
	@echo "make setup     - crea el entorno e instala dependencias"
	@echo "make ingest    - descarga y consolida los datos del DANE"
	@echo "make validate  - valida el esquema diario"
	@echo "make clean     - normaliza, agrega a semanal y escribe el panel"
	@echo "make evaluate  - backtest walk-forward de los baselines"
	@echo "make pipeline  - ingest + validate + clean + evaluate"
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

evaluate:
	$(PY) -m precios.pipeline.evaluate

pipeline: ingest clean evaluate

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

app:
	$(PY) -m streamlit run app/Inicio.py
