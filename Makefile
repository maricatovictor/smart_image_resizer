.PHONY: configure
configure:
	pip install pip-tools
	pip-compile requirements.in

.PHONY: install
install: configure
	pip install -r requirements.txt

.PHONY: clean-notebooks
clean-notebooks:
	analyses/*.ipynb

.PHONY: repro
repro: clean-notebooks
	PYTHONPATH=. dvc repro -c analyses/ ${TARGET}


.PHONY: build
build:
	streamlit/bin/build.sh


.PHONY: run
run: build
	streamlit/bin/run.sh
