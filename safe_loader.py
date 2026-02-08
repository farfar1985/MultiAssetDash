"""
Safe model/data loading utilities.

Wraps joblib.load with path validation to prevent loading from
untrusted sources. joblib uses pickle internally and can execute
arbitrary code if fed a malicious file.
"""

import os
import joblib

# Trusted directories where model/data files can be loaded from
TRUSTED_DIRS = [
    os.path.dirname(os.path.abspath(__file__)),  # Project root
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'horizons_wide'),
]


def safe_joblib_load(filepath: str, allow_any: bool = False) -> object:
    """
    Safely load a joblib/pickle file with path validation.
    
    Args:
        filepath: Path to the .joblib file
        allow_any: If True, skip path validation (use only for known-safe contexts)
        
    Returns:
        Loaded object
        
    Raises:
        ValueError: If filepath is outside trusted directories
        FileNotFoundError: If file doesn't exist
    """
    filepath = os.path.abspath(filepath)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if not allow_any:
        is_trusted = any(
            filepath.startswith(os.path.abspath(d))
            for d in TRUSTED_DIRS
        )
        if not is_trusted:
            raise ValueError(
                f"Refusing to load file from untrusted path: {filepath}\n"
                f"Trusted directories: {TRUSTED_DIRS}"
            )
    
    return joblib.load(filepath)
