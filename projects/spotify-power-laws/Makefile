.PHONY: all download schema forensics estimand validate models distributional uncertainty figures genre

PYTHON ?= python3
SRC := src

export PYTHONPATH := $(SRC):$(PYTHONPATH)

all: download schema forensics estimand validate models distributional uncertainty figures genre

download:
	$(PYTHON) scripts/00_download.py

schema:
	$(PYTHON) scripts/01_schema.py

forensics:
	$(PYTHON) scripts/02_forensics.py

estimand:
	$(PYTHON) scripts/03_estimand.py

validate:
	$(PYTHON) scripts/04_validate.py

models:
	$(PYTHON) scripts/05_models.py

distributional:
	$(PYTHON) scripts/06_distributional.py

uncertainty:
	$(PYTHON) scripts/07_uncertainty.py

figures:
	$(PYTHON) scripts/08_figures.py

genre:
	$(PYTHON) scripts/09_genre.py
