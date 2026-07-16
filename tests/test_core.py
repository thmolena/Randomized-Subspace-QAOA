"""Deprecated top-level test module. The maintained suite lives in code/tests/.

Run:  cd code && pip install -e '.[dev]' && pytest
"""
import pytest
pytest.skip("Tests moved to code/tests/ -- run pytest from the code/ directory.",
            allow_module_level=True)
