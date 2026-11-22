import pytest

pytestmark = [
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]

def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings", "ignore::pytest.PytestUnhandledThreadExceptionWarning")
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests with live broker")
    config.addinivalue_line("markers", "slow: Slow tests")
