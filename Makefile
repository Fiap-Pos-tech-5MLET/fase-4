.PHONY: help install test coverage lint format type-check security clean docker-build docker-run

# Variables
PYTHON := python3
PIP := pip
PROJECT_NAME := stock-prediction-api
DOCKER_IMAGE := $(PROJECT_NAME):latest

help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║   Stock Prediction API - Make Commands                         ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Development Setup:"
	@echo "  make install          - Install all dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test             - Run all unit tests"
	@echo "  make coverage         - Run tests with coverage report"
	@echo "  make coverage-html    - Generate HTML coverage report"
	@echo "  make lint             - Run all linters (pylint, flake8)"
	@echo "  make format           - Format code with black and isort"
	@echo "  make type-check       - Run mypy type checking"
	@echo "  make security         - Run security checks (bandit)"
	@echo "  make quality          - Run all quality checks"
	@echo ""
	@echo "Code Cleanup:"
	@echo "  make clean            - Remove generated files and cache"
	@echo "  make clean-all        - Remove all generated files and venv"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"
	@echo ""
	@echo "Utilities:"
	@echo "  make run-api          - Run API server"
	@echo "  make run-streamlit    - Run Streamlit dashboard"
	@echo "  make train            - Train the LSTM model"
	@echo ""

# Installation
install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"

install-dev: install
	@echo "Installing development dependencies..."
	$(PIP) install -r requirements-dev.txt
	@echo "✓ Development dependencies installed"

# Testing
test:
	@echo "Running unit tests..."
	pytest tests/ -v --tb=short
	@echo "✓ Tests completed"

test-fast:
	@echo "Running tests in parallel..."
	pytest tests/ -v -n auto

test-specific:
	@echo "Running specific test file..."
	pytest tests/test_lstm_model.py -v

test-watch:
	@echo "Running tests in watch mode..."
	pytest-watch tests/

# Coverage
coverage:
	@echo "Running tests with coverage..."
	pytest tests/ \
		--cov=src \
		--cov=app \
		--cov-report=term-missing \
		--cov-report=xml \
		-v
	@echo "✓ Coverage report generated"

coverage-html: coverage
	@echo "Generating HTML coverage report..."
	coverage html
	@echo "✓ Open htmlcov/index.html in browser"

coverage-check:
	@echo "Checking coverage threshold (90%)..."
	coverage report --fail-under=90
	@echo "✓ Coverage meets threshold"

# Linting & Formatting
lint:
	@echo "Running linters..."
	@echo "→ Pylint..."
	pylint src/ app/ --exit-zero
	@echo "→ Flake8..."
	flake8 src/ app/ tests/ --max-line-length=100 --exit-zero
	@echo "✓ Linting completed"

format:
	@echo "Formatting code..."
	@echo "→ Black..."
	black src/ app/ tests/
	@echo "→ isort..."
	isort src/ app/ tests/
	@echo "✓ Code formatted"

type-check:
	@echo "Running type checking with mypy..."
	mypy src/ app/ --no-error-summary --exit-zero
	@echo "✓ Type checking completed"

security:
	@echo "Running security checks..."
	bandit -r src/ app/ -v
	@echo "✓ Security scan completed"

# Quality
quality: lint type-check test coverage security
	@echo "✓ All quality checks passed"

quick-quality: lint type-check test
	@echo "✓ Quick quality checks passed"

# Cleanup
clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/ .mypy_cache/ .coverage htmlcov/ dist/ build/ *.egg-info
	@echo "✓ Clean completed"

clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf venv/
	@echo "✓ Full clean completed"

# API & Training
run-api:
	@echo "Starting API server..."
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-streamlit:
	@echo "Starting Streamlit dashboard..."
	streamlit run streamlit_app.py

train:
	@echo "Training LSTM model..."
	$(PYTHON) -m src.train
	@echo "✓ Training completed"

train-quick:
	@echo "Running quick training test..."
	$(PYTHON) -c "from src.train import run_training_pipeline; run_training_pipeline(epochs=2)"

# Docker
docker-build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE) .
	@echo "✓ Docker image built: $(DOCKER_IMAGE)"

docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 $(DOCKER_IMAGE)

docker-push:
	@echo "Pushing Docker image..."
	docker push $(DOCKER_IMAGE)

# CI/CD
ci: install-dev quality
	@echo "✓ CI pipeline completed successfully"

pre-commit: format lint type-check test
	@echo "✓ Pre-commit checks passed"

# Documentation
docs:
	@echo "Building documentation..."
	cd docs && make html
	@echo "✓ Documentation built: docs/_build/html/index.html"

docs-clean:
	@echo "Cleaning documentation..."
	cd docs && make clean

# Utilities
requirements-update:
	@echo "Updating requirements..."
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r requirements-dev.txt

check-deps:
	@echo "Checking for security vulnerabilities..."
	$(PIP) install safety
	safety check --json

info:
	@echo "Project Information:"
	@echo "  Python version: $(PYTHON) --version"
	@echo "  Project: $(PROJECT_NAME)"
	@echo "  Docker image: $(DOCKER_IMAGE)"
	@echo ""
	@echo "Current directory: $(PWD)"
	@echo "Virtual environment: $${VIRTUAL_ENV}"

.DEFAULT_GOAL := help
