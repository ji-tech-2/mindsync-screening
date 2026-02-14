.PHONY: help install test coverage quality quality-fix format lint complexity security duplication clean

help:
	@echo "MindSync Model Flask - Development Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install all dependencies including dev tools"
	@echo ""
	@echo "Testing & Coverage:"
	@echo "  make test             Run all tests"
	@echo "  make coverage         Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make quality          Run all quality checks (coverage, complexity, security)"
	@echo "  make quality-fix      Run quality checks and auto-fix formatting issues"
	@echo "  make format           Format code with Black"
	@echo "  make lint             Run flake8 and pylint"
	@echo "  make complexity       Check cyclomatic complexity with Radon"
	@echo "  make security         Run security analysis with Bandit"
	@echo "  make duplication      Check for code duplication"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove generated files and caches"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest --disable-warnings

coverage:
	pytest --cov=flaskr --cov-report=term-missing --cov-report=html --cov-report=xml

quality:
	python check_quality.py

quality-fix:
	python check_quality.py --fix

format:
	black flaskr tests *.py

lint:
	@echo "Running Flake8..."
	flake8 flaskr tests --config=.flake8
	@echo ""
	@echo "Running Pylint..."
	pylint flaskr --rcfile=.pylintrc

complexity:
	@echo "Cyclomatic Complexity:"
	radon cc flaskr -a -nb
	@echo ""
	@echo "Maintainability Index:"
	radon mi flaskr -nb

security:
	bandit -r flaskr -ll -f screen

duplication:
	pylint flaskr --disable=all --enable=duplicate-code --rcfile=.pylintrc

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf .coverage
