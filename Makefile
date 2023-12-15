# Makefile for Python Flask Application

PYTHON=python

SCRIPT_NAME=server.py

# Default target
.PHONY: all
all: run

# Target to run the main function and start the server
.PHONY: run
run:
	@$(PYTHON) $(SCRIPT_NAME)

# Target to run the demo function, showing the performance of the ai model
.PHONY: stats
stats:
	@$(PYTHON) -c 'import $(basename $(SCRIPT_NAME) .py); $(basename $(SCRIPT_NAME) .py).demo()'

# Target to clean the uploads directory
.PHONY: clean
clean:
	@if exist uploads\* del /Q uploads\*
	@echo Cleaned uploads directory.

# Target to install dependencies from requirements.txt
.PHONY: build
build:
	@$(PYTHON) -m pip install -r requirements.txt
	@echo Dependencies installed.
