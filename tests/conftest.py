"""Pytest configuration for the Quant test suite.

Adds the project root to sys.path so tests can `from create_factors import ...`
without needing a pyproject.toml or installed package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
