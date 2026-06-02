"""Pytest configuration for the Quant test suite.

Adds the project root to sys.path so tests can import top-level modules
(config, utils, optimize_portfolio) and pipeline-package modules
(pipeline.create_factors, pipeline.update_constituents) without a
pyproject.toml or installed package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
