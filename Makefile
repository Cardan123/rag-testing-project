# RAG Testing Project Makefile

.PHONY: help install setup test evaluate clean lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  setup        - Set up the project and database"
	@echo "  test         - Run unit tests"
	@echo "  evaluate     - Run evaluation with both frameworks"
	@echo "  evaluate-ragas - Run only Ragas evaluation"
	@echo "  evaluate-deepeval - Run only DeepEval evaluation"
	@echo "  clean        - Clean up generated files"
	@echo "  lint         - Run code linting"
	@echo "  format       - Format code"
	@echo "  check-env    - Check environment setup"

# Install dependencies
install:
	pip install -r requirements.txt

# Set up the project
setup: check-env
	python scripts/setup_data.py

# Run unit tests
test:
	python -m pytest tests/test_rag_system.py -v

# Run evaluations
evaluate: check-env
	python scripts/run_evaluations.py --framework both

evaluate-ragas: check-env
	python scripts/run_evaluations.py --framework ragas

evaluate-deepeval: check-env
	python scripts/run_evaluations.py --framework deepeval

# Clean up
clean:
	rm -rf evaluation_results/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Code quality
lint:
	isort --check-only src/ tests/
	black --check src/ tests/

format:
	isort src/ tests/
	black src/ tests/

# Environment check
check-env:
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found. Please copy .env.example to .env and configure it."; \
		exit 1; \
	fi
	@python -c "from src.config import settings; print('Environment check passed')"

# Quick demo
demo: setup evaluate
	@echo "Demo completed! Check evaluation_results/ for results."

# Development setup
dev-setup: install
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from template. Please edit it with your API keys."; \
	fi