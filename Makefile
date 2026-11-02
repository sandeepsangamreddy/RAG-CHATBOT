PYTHON ?= python
ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
else
VENV_PYTHON := .venv/bin/python
endif

.PHONY: install ingest chat ask eval test

install:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

ingest:
	$(VENV_PYTHON) -m src.ingest --pdf data/Active_One_Max_policy_wording-160925.pdf --rebuild

chat:
	$(VENV_PYTHON) -m streamlit run app/streamlit_app.py

ask:
	$(VENV_PYTHON) -m src.rag --ask "$(Q)"

eval:
	$(VENV_PYTHON) -m eval.run_eval

test:
	$(VENV_PYTHON) -m pytest -q
