"""
Pytest fixtures for QDT Nexus tests.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def valid_asset_names():
    """Return list of valid asset names."""
    return [
        "Crude_Oil", "Bitcoin", "Gold", "Silver", "Natural_Gas",
        "Copper", "Platinum", "Palladium", "Wheat", "Corn",
        "Soybeans", "Coffee", "Sugar", "Cotton", "Nikkei_225"
    ]


@pytest.fixture
def valid_asset_ids():
    """Return mapping of valid asset IDs to names."""
    return {
        "1866": "Crude_Oil",
        "1860": "Bitcoin",
        "1861": "Gold",
        "1862": "Silver",
        "1863": "Natural_Gas",
    }


@pytest.fixture
def invalid_inputs():
    """Return list of invalid/malicious inputs for testing."""
    return [
        "",
        "   ",
        None,
        "../../../etc/passwd",
        "../../secret.txt",
        "Gold; rm -rf /",
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "Fake_Asset",
        "GOLD",  # Wrong case
        "gold",  # Wrong case
    ]


@pytest.fixture
def valid_dates():
    """Return list of valid date strings."""
    return [
        "2024-01-01",
        "2025-12-31",
        "2026-02-03",
        "2000-01-01",
        "2099-12-31",
    ]


@pytest.fixture
def invalid_dates():
    """Return list of invalid date strings."""
    return [
        "01-01-2024",
        "2024/01/01",
        "2024-1-1",
        "not-a-date",
        "2024-13-01",  # Invalid month
        "2024-01-32",  # Invalid day
        "1999-01-01",  # Before 2000
        "2101-01-01",  # After 2100
    ]


@pytest.fixture
def sample_api_key():
    """Return a sample API key for testing."""
    return "qdt_test_key_abc123xyz789"
