#!/usr/bin/env python3
"""
Coverage runner script for datagen project.
Provides easy commands to check test coverage.
"""

import subprocess
import sys
from pathlib import Path


def run_coverage():
    """Run tests with coverage reporting."""
    print("🧪 Running tests with coverage...")
    result = subprocess.run([
        "uv", "run", "pytest", 
        "--cov=datagen",
        "--cov-report=html",
        "--cov-report=term-missing",
        "tests/"
    ], capture_output=False)
    
    if result.returncode == 0:
        print("\n✅ Coverage report generated!")
        print("📊 View detailed HTML report: htmlcov/index.html")
        print("📋 Coverage summary saved to COVERAGE_REPORT.md")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)


def coverage_summary():
    """Show coverage summary."""
    print("📊 Coverage Summary:")
    subprocess.run([
        "uv", "run", "coverage", "report", "--show-missing"
    ], capture_output=False)


def coverage_html():
    """Generate HTML coverage report."""
    print("🌐 Generating HTML coverage report...")
    subprocess.run([
        "uv", "run", "coverage", "html"
    ], capture_output=False)
    print("📊 HTML report: htmlcov/index.html")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python coverage_runner.py [run|summary|html]")
        print("  run     - Run tests with coverage")
        print("  summary - Show coverage summary") 
        print("  html    - Generate HTML report")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "run":
        run_coverage()
    elif command == "summary":
        coverage_summary()
    elif command == "html":
        coverage_html()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()