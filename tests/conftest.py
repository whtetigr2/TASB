"""pytest configuration for thermobridge tests."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that require a GPU and downloaded model",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: requires GPU and downloaded model (~10 min); skipped by default",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Pass --run-slow to run GPU tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
