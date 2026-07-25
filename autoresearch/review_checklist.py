"""
Autoresearch Reviewer — automated code review before running experiments.

Catches common quant research mistakes:
  1. Data leakage (future information in features)
  2. Overfitting patterns (too many features, target peeking)
  3. Code correctness (NaN handling, import safety)
  4. Constraint violations (balanced training, forbidden packages)

Usage:
  from review_checklist import review_experiment
  issues = review_experiment()  # returns list of (severity, message)

Severity levels:
  BLOCK  — experiment must not run, fix first
  WARN   — suspicious, log but allow
  INFO   — observation, no action needed
"""

import ast
import importlib
import inspect
import re
import sys
from pathlib import Path
from typing import Callable

EXPERIMENT_FILE = Path(__file__).parent / 'experiment_auto.py'

# Packages available in requirements.txt
ALLOWED_IMPORTS = {
    'numpy', 'np', 'pandas', 'pd', 'sklearn', 'scipy', 'statsmodels',
    'json', 'math', 'time', 'warnings', 'pathlib', 'collections',
    'itertools', 'functools', 'typing', 'dataclasses', 'os', 'sys',
}

# Patterns that indicate future data leakage
LEAKAGE_PATTERNS = [
    (r'\.shift\(\s*-', 'Negative shift() looks into the future'),
    (r'\.rolling\(.*\)\.apply.*future', 'Rolling window referencing future'),
    (r'target.*feature|feature.*target', 'Target variable used in feature construction (check context)'),
    (r'y_test.*fit|y_test.*train', 'Test labels used during training'),
    (r'X_test.*fit_transform', 'Test data used in fit_transform (data leakage)'),
]

# Patterns that indicate overfitting risk
OVERFIT_PATTERNS = [
    (r'class_weight\s*=\s*[\'"]balanced[\'"]', 'class_weight=balanced inflates probabilities (forbidden)'),
    (r'sample_weight.*pos.*neg|sample_weight.*neg.*pos', 'Manual sample reweighting mimics balanced training'),
    (r'GridSearchCV|RandomizedSearchCV', 'Hyperparameter search inside experiment can overfit to validation set'),
    (r'PolynomialFeatures\(degree=[3-9]', 'High-degree polynomial features are overfitting magnets'),
]

# Patterns that indicate correctness issues
CORRECTNESS_PATTERNS = [
    (r'\.fillna\(0\).*pct_change|pct_change.*\.fillna\(0\)', 'fillna(0) on pct_change hides missing data'),
    (r'import\s+torch|import\s+tensorflow|import\s+keras', 'Deep learning frameworks not needed for tabular macro data'),
]


def _read_source() -> str:
    """Read experiment_auto.py source code."""
    return EXPERIMENT_FILE.read_text()


def _strip_comments_and_docstrings(source: str) -> str:
    """Remove comments and docstrings so regex checks only scan real code."""
    tree = ast.parse(source)
    docstring_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            ds = ast.get_docstring(node, clean=False)
            if ds and hasattr(node, 'body') and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    for ln in range(first.lineno, first.end_lineno + 1):
                        docstring_lines.add(ln)

    lines = source.splitlines()
    cleaned = []
    for i, line in enumerate(lines, 1):
        if i in docstring_lines:
            continue
        code = line.split('#')[0]
        cleaned.append(code)
    return '\n'.join(cleaned)


def _parse_ast(source: str) -> ast.Module:
    """Parse source into AST."""
    return ast.parse(source)


def check_leakage(source: str) -> list[tuple[str, str]]:
    """Check for data leakage patterns."""
    issues = []
    for pattern, msg in LEAKAGE_PATTERNS:
        matches = re.findall(pattern, source, re.IGNORECASE)
        if matches:
            issues.append(('BLOCK', f'LEAKAGE: {msg}'))
    return issues


def check_overfitting(source: str) -> list[tuple[str, str]]:
    """Check for overfitting-prone patterns."""
    issues = []
    for pattern, msg in OVERFIT_PATTERNS:
        if re.search(pattern, source, re.IGNORECASE):
            issues.append(('BLOCK', f'OVERFIT: {msg}'))
    return issues


def check_correctness(source: str) -> list[tuple[str, str]]:
    """Check for common correctness issues."""
    issues = []
    for pattern, msg in CORRECTNESS_PATTERNS:
        if re.search(pattern, source, re.IGNORECASE):
            issues.append(('WARN', f'CORRECTNESS: {msg}'))
    return issues


def check_imports(tree: ast.Module) -> list[tuple[str, str]]:
    """Check that only allowed packages are imported."""
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split('.')[0]
                if top not in ALLOWED_IMPORTS:
                    issues.append(('BLOCK', f'IMPORT: {alias.name} is not in allowed packages'))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split('.')[0]
                if top not in ALLOWED_IMPORTS:
                    issues.append(('BLOCK', f'IMPORT: {node.module} is not in allowed packages'))
    return issues


def check_interface(source: str, tree: ast.Module) -> list[tuple[str, str]]:
    """Check that required interface functions exist with correct signatures."""
    issues = []

    func_names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    if 'build_features' not in func_names:
        issues.append(('BLOCK', 'INTERFACE: build_features() function is missing'))
    if 'train_model' not in func_names:
        issues.append(('BLOCK', 'INTERFACE: train_model() function is missing'))

    if 'DESCRIPTION' not in source:
        issues.append(('BLOCK', 'INTERFACE: DESCRIPTION variable is missing'))

    return issues


def check_feature_count(source: str) -> list[tuple[str, str]]:
    """Heuristic: count feature assignments in build_features."""
    issues = []
    assignments = re.findall(r"features\[(?:'|\")(\w+)(?:'|\")\]", source)
    n = len(set(assignments))
    if n > 30:
        issues.append(('WARN', f'COMPLEXITY: {n} features detected (max recommended: 30)'))
    elif n > 20:
        issues.append(('INFO', f'COMPLEXITY: {n} features detected — consider pruning'))
    return issues


def check_train_model_safety(source: str) -> list[tuple[str, str]]:
    """Check train_model doesn't access global data or evaluation harness."""
    issues = []
    if 'evaluate_harness' in source or 'evaluate(' in source:
        issues.append(('BLOCK', 'SAFETY: experiment_auto.py must not import evaluate_harness'))
    if 'results.tsv' in source or 'RESULTS_TSV' in source:
        issues.append(('BLOCK', 'SAFETY: experiment_auto.py must not access results.tsv'))
    if 'run.log' in source:
        issues.append(('BLOCK', 'SAFETY: experiment_auto.py must not read run.log'))
    return issues


def review_experiment() -> list[tuple[str, str]]:
    """Run all review checks. Returns list of (severity, message) tuples.

    Returns empty list if all clear.
    Severity: BLOCK (must fix), WARN (suspicious), INFO (observation).
    """
    source = _read_source()
    tree = _parse_ast(source)
    code_only = _strip_comments_and_docstrings(source)

    all_issues = []
    all_issues.extend(check_interface(source, tree))
    all_issues.extend(check_imports(tree))
    all_issues.extend(check_leakage(code_only))
    all_issues.extend(check_overfitting(code_only))
    all_issues.extend(check_correctness(code_only))
    all_issues.extend(check_feature_count(code_only))
    all_issues.extend(check_train_model_safety(code_only))

    return all_issues


def has_blockers(issues: list[tuple[str, str]]) -> bool:
    """Return True if any issue is a BLOCK."""
    return any(sev == 'BLOCK' for sev, _ in issues)


def format_review(issues: list[tuple[str, str]]) -> str:
    """Format issues for terminal output."""
    if not issues:
        return '✓ REVIEW PASSED: no issues found'

    lines = ['=== CODE REVIEW ===']
    for sev, msg in sorted(issues, key=lambda x: {'BLOCK': 0, 'WARN': 1, 'INFO': 2}[x[0]]):
        icon = {'BLOCK': '✗', 'WARN': '⚠', 'INFO': '·'}[sev]
        lines.append(f'  {icon} [{sev}] {msg}')

    blocks = sum(1 for s, _ in issues if s == 'BLOCK')
    warns = sum(1 for s, _ in issues if s == 'WARN')
    if blocks:
        lines.append(f'\n✗ REVIEW FAILED: {blocks} blocker(s) — fix before running')
    elif warns:
        lines.append(f'\n⚠ REVIEW PASSED with {warns} warning(s)')
    else:
        lines.append('\n✓ REVIEW PASSED')

    return '\n'.join(lines)


if __name__ == '__main__':
    issues = review_experiment()
    print(format_review(issues))
    sys.exit(1 if has_blockers(issues) else 0)
