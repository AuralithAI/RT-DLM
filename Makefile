# RT-DLM Makefile
# Common development and deployment tasks

.PHONY: help install install-dev test test-fast test-cov lint format type-check \
        security-check clean build docker-build docker-push \
        helm-install helm-upgrade helm-uninstall helm-template helm-lint helm-dry-run \
        k8s-status k8s-logs train-tiny train-small docs serve-docs pre-commit

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# Variables
# =============================================================================
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
ISORT := isort
MYPY := mypy
FLAKE8 := flake8
DOCKER := docker
DOCKER_COMPOSE := docker-compose
KUBECTL := kubectl
HELM := helm

# Docker settings
DOCKER_REGISTRY ?= ghcr.io
DOCKER_IMAGE ?= auralithai/rt-dlm
DOCKER_TAG ?= latest

# Helm settings
HELM_RELEASE ?= rtdlm
HELM_NAMESPACE ?= rtdlm
HELM_CHART ?= ./helm/rtdlm

# Training settings
CHECKPOINT_DIR ?= ./checkpoints
DATA_DIR ?= ./data
LOG_DIR ?= ./logs

# =============================================================================
# Help
# =============================================================================
help: ## Show this help message
	@echo "RT-DLM Development Commands"
	@echo "==========================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================
install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev,docs,monitoring]"
	pre-commit install

install-all: ## Install all dependencies including optional
	$(PIP) install -e ".[all]"

# =============================================================================
# Testing
# =============================================================================
test: ## Run all tests
	cd src/tests && $(PYTEST) . -v --tb=short

test-fast: ## Run tests excluding slow ones
	cd src/tests && $(PYTEST) . -v --tb=short -m "not slow"

test-cov: ## Run tests with coverage report
	cd src/tests && $(PYTEST) . -v --cov=src --cov-report=html --cov-report=term-missing

test-parallel: ## Run tests in parallel
	cd src/tests && $(PYTEST) . -v -n auto

test-specific: ## Run specific test file (usage: make test-specific FILE=test_model.py)
	cd src/tests && $(PYTEST) $(FILE) -v --tb=long

# =============================================================================
# Code Quality
# =============================================================================
lint: ## Run all linting checks
	$(BLACK) --check .
	$(ISORT) --check-only .
	$(FLAKE8) . --max-line-length=120 --extend-ignore=E203,W503

format: ## Format code with black and isort
	$(BLACK) .
	$(ISORT) .

type-check: ## Run mypy type checking
	$(MYPY) src/core/ src/config/ --ignore-missing-imports

security-check: ## Run security scanning
	bandit -r src/core/ src/modules/ -ll -ii
	safety check

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

# =============================================================================
# Cleaning
# =============================================================================
clean: ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean ## Clean everything including checkpoints and logs
	rm -rf $(CHECKPOINT_DIR)/*
	rm -rf $(LOG_DIR)/*

# =============================================================================
# Building
# =============================================================================
build: ## Build Python package
	$(PYTHON) -m build

build-wheel: ## Build wheel only
	$(PYTHON) -m build --wheel

# =============================================================================
# Docker
# =============================================================================
docker-build: ## Build Docker training image
	$(DOCKER) build -t $(DOCKER_IMAGE):$(DOCKER_TAG) -f Dockerfile.train .

docker-build-cpu: ## Build CPU-only Docker image
	$(DOCKER) build -t $(DOCKER_IMAGE):$(DOCKER_TAG)-cpu -f Dockerfile.train --target cpu .

docker-push: ## Push Docker image to registry
	$(DOCKER) tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	$(DOCKER) push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-compose-build: ## Build all Docker Compose services
	$(DOCKER_COMPOSE) build

docker-compose-up: ## Start training with Docker Compose
	$(DOCKER_COMPOSE) up training

docker-compose-up-monitoring: ## Start training with monitoring stack
	$(DOCKER_COMPOSE) --profile monitoring up

docker-compose-down: ## Stop all Docker Compose services
	$(DOCKER_COMPOSE) down

# =============================================================================
# Training
# =============================================================================
train-tiny: ## Run training with tiny preset (for testing)
	$(PYTHON) src/train.py \
		--preset tiny \
		--batch_size 4 \
		--epochs 10 \
		--checkpoint_dir $(CHECKPOINT_DIR) \
		--log_dir $(LOG_DIR)

train-small: ## Run training with small preset
	$(PYTHON) src/train.py \
		--preset small \
		--batch_size 16 \
		--epochs 50 \
		--checkpoint_dir $(CHECKPOINT_DIR) \
		--log_dir $(LOG_DIR)

train-medium: ## Run training with medium preset
	$(PYTHON) src/train.py \
		--preset medium \
		--batch_size 32 \
		--epochs 100 \
		--checkpoint_dir $(CHECKPOINT_DIR) \
		--log_dir $(LOG_DIR) \
		--enable_wandb

train-resume: ## Resume training from latest checkpoint
	$(PYTHON) src/train.py \
		--resume_from $(CHECKPOINT_DIR)/latest \
		--checkpoint_dir $(CHECKPOINT_DIR) \
		--log_dir $(LOG_DIR)

# =============================================================================
# Kubernetes (Helm)
# =============================================================================
helm-install: ## Install Helm chart to Kubernetes
	$(HELM) install $(HELM_RELEASE) $(HELM_CHART) -n $(HELM_NAMESPACE) --create-namespace

helm-upgrade: ## Upgrade Helm release
	$(HELM) upgrade $(HELM_RELEASE) $(HELM_CHART) -n $(HELM_NAMESPACE) --reuse-values

helm-uninstall: ## Uninstall Helm release
	$(HELM) uninstall $(HELM_RELEASE) -n $(HELM_NAMESPACE)

helm-template: ## Render Helm templates locally
	$(HELM) template $(HELM_RELEASE) $(HELM_CHART) -n $(HELM_NAMESPACE)

helm-lint: ## Lint Helm chart
	$(HELM) lint $(HELM_CHART)

helm-dry-run: ## Dry-run Helm install
	$(HELM) install $(HELM_RELEASE) $(HELM_CHART) -n $(HELM_NAMESPACE) --dry-run --debug

k8s-status: ## Check Kubernetes deployment status
	$(KUBECTL) get all -n $(HELM_NAMESPACE)

k8s-logs: ## Get logs from Kubernetes training pod
	$(KUBECTL) logs -l app.kubernetes.io/name=rtdlm -n $(HELM_NAMESPACE) --tail=100 -f

# =============================================================================
# Documentation
# =============================================================================
docs: ## Build documentation
	cd docs && sphinx-build -b html . _build/html

serve-docs: ## Serve documentation locally
	cd docs/_build/html && $(PYTHON) -m http.server 8080

# =============================================================================
# Utilities
# =============================================================================
check-env: ## Check environment setup
	@echo "Python: $$($(PYTHON) --version)"
	@echo "JAX: $$($(PYTHON) -c 'import jax; print(jax.__version__)')"
	@echo "JAX devices: $$($(PYTHON) -c 'import jax; print(jax.devices())')"
	@echo "CUDA available: $$($(PYTHON) -c 'import jax; print(any(\"cuda\" in str(d).lower() or \"gpu\" in str(d).lower() for d in jax.devices()))')"

gpu-info: ## Show GPU information
	nvidia-smi

profile-memory: ## Profile memory usage during training
	$(PYTHON) -c "from src.core.memory_profiler import MemoryProfiler; MemoryProfiler.print_memory_stats()"

validate-model: ## Validate model initialization
	$(PYTHON) -c "\
from src.rtdlm import create_rtdlm_agi; \
from src.config.agi_config import AGIConfig; \
config = AGIConfig.from_preset('tiny'); \
print('Config:', config); \
print('Model configuration valid!')"
