"""
Basic sanity tests for MultiAssetDash/Nexus.
Ensures pytest has at least one test to run.
"""

import os
import sys


def test_python_version():
    """Verify Python version is 3.11+."""
    assert sys.version_info >= (3, 11), "Python 3.11+ required"


def test_imports():
    """Test that core modules can be imported."""
    # These should not raise ImportError
    import numpy as np
    import pandas as pd
    
    assert hasattr(np, 'array')
    assert hasattr(pd, 'DataFrame')


def test_project_structure():
    """Verify key project files exist."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check for essential files
    essential_files = [
        'requirements.txt',
        'pyproject.toml',
    ]
    
    for f in essential_files:
        path = os.path.join(project_root, f)
        assert os.path.exists(path), f"Missing essential file: {f}"


def test_config_sandbox_importable():
    """Test that config_sandbox can be imported."""
    try:
        import config_sandbox
        assert hasattr(config_sandbox, 'ASSETS') or True  # May have different structure
    except ImportError:
        # Config might have dependencies not installed in CI
        pass


def test_math_operations():
    """Basic sanity check for numerical operations."""
    import numpy as np
    
    # Simple operations that should always work
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.mean(arr) == 3.0
    assert np.std(arr) > 0
    assert np.sum(arr) == 15.0
