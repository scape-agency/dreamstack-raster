#!/usr/bin/env bash

# =============================================================================
# Script: lint_code.sh
# =============================================================================
# This script runs flake8 to check for coding standard violations in the codebase.
# =============================================================================

# Function to display a message and exit on error
function exit_on_error {
    echo "Error: $1"
    exit 1
}

# Check if flake8 is installed
if ! command -v flake8 &> /dev/null; then
    exit_on_error "flake8 is not installed. Please install it and try again."
fi

# Run flake8 to lint the codebase
echo "Linting the codebase with flake8..."
flake8 swing_hello/ || exit_on_error "Linting failed."

echo "Codebase linted successfully."
