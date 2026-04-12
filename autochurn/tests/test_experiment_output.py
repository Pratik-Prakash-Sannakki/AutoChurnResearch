"""
Tests that experiment.py produces output the agent can reliably grep.
These tests import and run the experiment output logic directly.
"""
import re
import subprocess
import sys


def test_output_contains_greppable_f1_line():
    """Output must contain a line starting with 'f1:' followed by a float."""
    result = subprocess.run(
        [sys.executable, "experiment.py"],
        capture_output=True, text=True, cwd="."
    )
    output = result.stdout + result.stderr
    match = re.search(r"^f1:\s+\d+\.\d+", output, re.MULTILINE)
    assert match is not None, f"No 'f1:' line found in output:\n{output}"


def test_output_contains_greppable_precision_line():
    """Output must contain a line starting with 'precision:' followed by a float."""
    result = subprocess.run(
        [sys.executable, "experiment.py"],
        capture_output=True, text=True, cwd="."
    )
    output = result.stdout + result.stderr
    match = re.search(r"^precision:\s+\d+\.\d+", output, re.MULTILINE)
    assert match is not None, f"No 'precision:' line found in output:\n{output}"


def test_output_contains_greppable_recall_line():
    """Output must contain a line starting with 'recall:' followed by a float."""
    result = subprocess.run(
        [sys.executable, "experiment.py"],
        capture_output=True, text=True, cwd="."
    )
    output = result.stdout + result.stderr
    match = re.search(r"^recall:\s+\d+\.\d+", output, re.MULTILINE)
    assert match is not None, f"No 'recall:' line found in output:\n{output}"


def test_output_contains_balance_ok_line():
    """Output must contain a line starting with 'balance_ok:' with value 'yes' or 'no'."""
    result = subprocess.run(
        [sys.executable, "experiment.py"],
        capture_output=True, text=True, cwd="."
    )
    output = result.stdout + result.stderr
    match = re.search(r"^balance_ok:\s+(yes|no)", output, re.MULTILINE)
    assert match is not None, f"No 'balance_ok:' line found in output:\n{output}"


def test_output_separator_line():
    """Output must contain the '---' separator line before metrics."""
    result = subprocess.run(
        [sys.executable, "experiment.py"],
        capture_output=True, text=True, cwd="."
    )
    output = result.stdout + result.stderr
    assert "---" in output, f"No '---' separator found in output:\n{output}"
