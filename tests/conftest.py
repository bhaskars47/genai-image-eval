# tests/conftest.py
#
# pytest configuration: ensures the repository root is on sys.path so the
# tests can do `from evaluators.* import ...` and `from pipeline.* import ...`
# without requiring an editable install or PYTHONPATH manipulation by the user.
#
# Usage:
#   pip install -r requirements-dev.txt
#   pytest -v
#
# Notes:
#   - Tests deliberately stick to pure-logic surfaces (helpers, static
#     methods, dataclasses) so they run on a clean checkout without
#     loading any ML weights.
#   - Heavy modules (torch, transformers, insightface) are NOT required to
#     run the harness; tests that touch them use pytest.importorskip or
#     stub-via-sys.modules to keep CI fast and self-contained.

import sys
from pathlib import Path

# Repo root = parent of this `tests/` directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
