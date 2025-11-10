.PHONY: install lint test clean

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .

lint:
	@echo "Running code linting..."
	ruff check src tests

test:
	@echo "Running pytest..."
	pytest -v --maxfail=1 --disable-warnings

clean:
	@echo "Cleaning up cache and temporary files..."
	rm -rf .pytest_cache __pycache__ */__pycache__