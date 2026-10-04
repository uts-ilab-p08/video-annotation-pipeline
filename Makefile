#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = annotation_pipeline
PYTHON_VERSION = 3.14
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv lock --upgrade-package videometa
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) annotation_pipeline/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)


## Rebuild reports/coverage_explorer_data/ from the bronze schema (run after notebook 08; ~1-2 min)
.PHONY: coverage-explorer-data
coverage-explorer-data:
	.venv/bin/python reports/build_coverage_explorer.py


## Serve the coverage explorer at http://localhost:8000/reports/coverage_explorer.html
## (builds the data first only when it is missing; the server then refreshes a video's data from
## the database whenever the page opens it, so the page always reflects Supabase)
.PHONY: coverage-explorer
coverage-explorer:
	@test -f reports/coverage_explorer_data/index.json || $(MAKE) coverage-explorer-data
	.venv/bin/python reports/serve_coverage_explorer.py
