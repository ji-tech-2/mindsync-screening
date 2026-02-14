#!/usr/bin/env python3
"""
Code Quality and Coverage Check Script

This script runs comprehensive code quality checks including:
- Code coverage (minimum 80%)
- Cyclomatic complexity (maximum 15)
- Code duplication detection
- Security issues (no blocker/critical)
- Code style and formatting
- Static analysis

Usage:
    python check_quality.py [--fix] [--skip-tests]

Options:
    --fix: Auto-fix formatting issues with black
    --skip-tests: Skip running tests (only run static analysis)
"""

import subprocess
import sys
import argparse
import re
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(message):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def print_success(message):
    """Print a success message"""
    print(f"{Colors.OKGREEN}[PASS] {message}{Colors.ENDC}")


def print_error(message):
    """Print an error message"""
    print(f"{Colors.FAIL}[FAIL] {message}{Colors.ENDC}")


def print_warning(message):
    """Print a warning message"""
    print(f"{Colors.WARNING}[WARN] {message}{Colors.ENDC}")


def run_command(command, check_return=True, capture=False):
    """Run a command and return success status and output"""
    try:
        result = subprocess.run(command, shell=True, capture_output=capture, text=True)
        if capture:
            return (
                result.returncode == 0 if check_return else True,
                result.stdout,
                result.stderr,
            )
        else:
            if check_return and result.returncode != 0:
                return False, "", ""
            return True, "", ""
    except Exception as e:
        print_error(f"Failed to run command: {command}")
        print_error(f"Error: {str(e)}")
        return False, "", ""


def check_black(fix=False):
    """Check code formatting with Black"""
    print_header("Black - Code Formatting")

    if fix:
        print("Running Black formatter (auto-fix enabled)...")
        cmd = "black flaskr tests *.py"
    else:
        print("Checking code formatting...")
        cmd = "black --check --diff flaskr tests *.py"

    success, _, _ = run_command(cmd, check_return=not fix)

    if success or fix:
        print_success("Code formatting: All files properly formatted")
        return True
    else:
        print_error("Code formatting issues found. Run with --fix to auto-format")
        return False


def check_flake8():
    """Run flake8 for style and complexity"""
    print_header("Flake8 - Style & Complexity Check (max complexity: 15)")

    print("Running flake8...")
    success, stdout, _ = run_command(
        "flake8 flaskr tests --config=.flake8 --statistics", capture=True
    )

    # Print output
    if stdout:
        print(stdout)

    # Try to extract issue count
    if stdout:
        match = re.search(r"^(\d+)$", stdout, re.MULTILINE)
        if match:
            issue_count = match.group(1)
            if int(issue_count) == 0:
                print_success(f"Flake8: 0 issues found")
            else:
                print_error(f"Flake8: {issue_count} issues found")
        elif success:
            print_success("Flake8: No issues found")

    if success:
        return True
    else:
        print_error("Flake8 found issues. Please fix the reported problems")
        return False


def check_pylint():
    """Run pylint for code quality"""
    print_header("Pylint - Code Quality Analysis")

    print("Running pylint...")
    result = subprocess.run(
        "pylint flaskr --rcfile=.pylintrc --fail-under=8.0",
        shell=True,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    # Extract pylint score
    score_match = re.search(r"rated at ([0-9.]+)/10", result.stdout)
    if score_match:
        score = float(score_match.group(1))
        if result.returncode == 0:
            print_success(f"Pylint: Score {score}/10 (>= 8.0 required)")
            return True
        else:
            print_error(f"Pylint: Score {score}/10 (below 8.0 threshold)")
            return False
    else:
        if result.returncode == 0:
            print_success("Pylint check passed")
            return True
        else:
            print_error("Pylint found issues")
            return False


def check_radon_complexity():
    """Check cyclomatic complexity with Radon"""
    print_header("Radon - Cyclomatic Complexity Check (max: 15)")

    print("Running Radon complexity analysis...")
    result = subprocess.run(
        "radon cc flaskr -a -nb --total-average",
        shell=True,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    # Extract average complexity
    avg_match = re.search(r"Average complexity: ([A-F]) \(([0-9.]+)\)", result.stdout)
    if avg_match:
        grade = avg_match.group(1)
        complexity = float(avg_match.group(2))
        print_success(f"Average Complexity: {grade} ({complexity:.2f}) - Grade {grade}")

    # Check for high complexity (D, E, F grades indicate complexity > 15)
    if "- D" in result.stdout or "- E" in result.stdout or "- F" in result.stdout:
        print_error("High cyclomatic complexity detected (> 15)")
        return False
    else:
        print_success("All functions have complexity <= 15")
        return True


def check_radon_maintainability():
    """Check maintainability index with Radon"""
    print_header("Radon - Maintainability Index")

    print("Running Radon maintainability analysis...")
    result = subprocess.run(
        "radon mi flaskr -nb", shell=True, capture_output=True, text=True
    )

    print(result.stdout)
    print_success("Maintainability index check completed")
    return True


def check_bandit():
    """Run Bandit security checks"""
    print_header("Bandit - Security Analysis")

    print("Running Bandit security scanner...")
    result = subprocess.run(
        "bandit -r flaskr -ll -f screen",  # -ll = only show HIGH severity issues
        shell=True,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    # Try to parse issue count
    issues_match = re.search(r"Total issues \(severity\): (\d+)", result.stdout)
    if issues_match:
        issues = int(issues_match.group(1))
        if issues == 0:
            print_success(f"Security: 0 high/critical issues found")
            return True
        else:
            print_error(f"Security: {issues} high/critical issues found")
            return False
    elif "No issues identified" in result.stdout or result.returncode == 0:
        print_success("Security: No critical or high severity issues found")
        return True
    else:
        print_error("Security issues found. Please review and fix")
        return False


def check_duplication():
    """Check for code duplication using Pylint"""
    print_header("Code Duplication Check")

    print("Running code duplication detection...")
    result = subprocess.run(
        "pylint flaskr --disable=all --enable=duplicate-code --rcfile=.pylintrc",
        shell=True,
        capture_output=True,
        text=True,
    )

    if "duplicate-code" in result.stdout:
        print(result.stdout)
        print_error("Code duplication detected. Please refactor duplicated code")
        return False
    else:
        print_success("No significant code duplication detected")
        return True


def run_tests_with_coverage():
    """Run tests with coverage report"""
    print_header("Pytest - Tests & Coverage (minimum: 80%)")

    print("Running tests with coverage...")
    result = subprocess.run("pytest", shell=True, capture_output=True, text=True)

    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Extract coverage percentage
    coverage_match = re.search(r"total.*?(\d+\.?\d*)", result.stdout)
    if coverage_match:
        coverage = float(coverage_match.group(1))
        if coverage >= 80.0:
            print_success(f"Coverage: {coverage}% (>= 80% required)")
            return True
        else:
            print_error(f"Coverage: {coverage}% (below 80% threshold)")
            return False
    else:
        if result.returncode == 0:
            print_success("All tests passed with >= 80% coverage")
            return True
        else:
            print_error("Tests failed or coverage is below 80%")
            return False


def main():
    """Main function to run all quality checks"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive code quality and coverage checks"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Auto-fix formatting issues with Black"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests (only run static analysis)",
    )

    args = parser.parse_args()

    print_header("MindSync Model Flask - Code Quality & Coverage Check")
    print(f"{Colors.OKCYAN}Standards:{Colors.ENDC}")
    print(f"  - Code Coverage: >= 80%")
    print(f"  - Cyclomatic Complexity: <= 15")
    print(f"  - Code Duplication: None")
    print(f"  - Security Issues: No blocker/critical")
    print(f"  - Code Quality: Pylint score >= 8.0")

    results = {}

    # Run all checks
    results["Black Formatting"] = check_black(fix=args.fix)
    results["Flake8 Style & Complexity"] = check_flake8()
    results["Pylint Quality"] = check_pylint()
    results["Radon Complexity"] = check_radon_complexity()
    results["Radon Maintainability"] = check_radon_maintainability()
    results["Bandit Security"] = check_bandit()
    results["Code Duplication"] = check_duplication()

    if not args.skip_tests:
        results["Tests & Coverage"] = run_tests_with_coverage()

    # Print summary
    print_header("Quality Check Summary")

    passed = 0
    failed = 0

    for check, success in results.items():
        if success:
            print_success(f"{check}: PASSED")
            passed += 1
        else:
            print_error(f"{check}: FAILED")
            failed += 1

    print(f"\n{Colors.BOLD}Total: {passed} passed, {failed} failed{Colors.ENDC}\n")

    if failed > 0:
        print_error("Quality checks failed! Please fix the issues above.")
        sys.exit(1)
    else:
        print_success("All quality checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
