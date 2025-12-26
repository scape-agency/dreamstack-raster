#!/usr/bin/env bash

# =============================================================================
# Script: run_tests.sh
# =============================================================================
# This script runs the test suite using pytest and generates a coverage report
# if specified.
# =============================================================================

# Function to display a message and exit on error
function exit_on_error {
    echo "Error: $1"
    exit 1
}

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || exit_on_error "Failed to activate virtual environment."

# Run tests with coverage
echo "Running tests with coverage..."
pytest --cov=swing_hello --cov-report=term-missing || exit_on_error "Tests failed."

echo "All tests passed successfully."
