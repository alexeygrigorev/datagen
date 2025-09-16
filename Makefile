# Makefile for datagen project

.PHONY: data tests help

# Default target
help:
	@echo "Available targets:"
	@echo "  data   - Generate synthetic dataset"
	@echo "  tests  - Run tests with coverage"
	@echo "  help   - Show this help message"

# Generate synthetic dataset
data:
	uv run python datagen.py

# Run tests with coverage
tests:
	uv run pytest --cov=datagen --cov-report=html --cov-report=term tests/