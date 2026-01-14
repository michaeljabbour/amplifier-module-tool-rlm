"""Pytest configuration and fixtures for tool-rlm tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_shadow_tool():
    """Create a mock shadow tool for testing."""
    tool = MagicMock()
    tool.execute = AsyncMock()
    return tool


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider for testing."""
    provider = MagicMock()
    provider.complete = AsyncMock()
    return provider


@pytest.fixture
def mock_hooks():
    """Create a mock hooks registry for testing."""
    hooks = MagicMock()
    hooks.emit = AsyncMock()
    return hooks


@pytest.fixture
def mock_coordinator(mock_shadow_tool, mock_provider, mock_hooks):
    """Create a mock module coordinator with all dependencies."""
    coordinator = MagicMock()

    # Set up tools registry
    tools = {"shadow": mock_shadow_tool}

    # Set up providers registry
    providers = {"anthropic": mock_provider}

    def get_side_effect(key: str):
        if key == "tools":
            return tools
        if key == "providers":
            return providers
        if key == "hooks":
            return mock_hooks
        return None

    coordinator.get = MagicMock(side_effect=get_side_effect)
    coordinator.get_capability = MagicMock(return_value=None)
    coordinator.register_capability = MagicMock()
    coordinator.mount = AsyncMock()

    return coordinator


@pytest.fixture
def sample_content():
    """Sample content for testing RLM processing."""
    return """
def calculate_sum(numbers):
    '''Calculate the sum of a list of numbers.'''
    return sum(numbers)

def calculate_average(numbers):
    '''Calculate the average of a list of numbers.'''
    if not numbers:
        return 0
    return calculate_sum(numbers) / len(numbers)

def find_max(numbers):
    '''Find the maximum value in a list.'''
    if not numbers:
        return None
    return max(numbers)

def find_min(numbers):
    '''Find the minimum value in a list.'''
    if not numbers:
        return None
    return min(numbers)
"""


@pytest.fixture
def large_content():
    """Large content for testing chunking behavior."""
    base = "This is line {} of the test content.\n"
    return "".join(base.format(i) for i in range(10000))


@pytest.fixture
def mock_chat_response():
    """Create a mock ChatResponse with code block."""
    response = MagicMock()
    response.content = [
        MagicMock(
            text="""```python
# Check the content length
print(f"Content length: {len(context)}")
FINAL(f"The content has {len(context)} characters")
```"""
        )
    ]
    response.usage = MagicMock(input_tokens=100, output_tokens=50)
    return response


@pytest.fixture
def mock_shadow_create_response():
    """Mock response for shadow create operation."""
    return MagicMock(
        success=True,
        output={
            "shadow_id": "test-shadow-123",
            "mode": "container",
            "local_sources": [],
            "status": "running",
            "snapshot_commits": {},
            "env_vars_passed": [],
        },
    )


@pytest.fixture
def mock_shadow_exec_success():
    """Mock response for successful shadow exec operation."""
    return MagicMock(
        success=True,
        output={
            "exit_code": 0,
            "stdout": "[RLM_INIT_SUCCESS]\nContext loaded: 1,000 characters",
            "stderr": "",
        },
    )


@pytest.fixture
def mock_shadow_exec_failure():
    """Mock response for failed shadow exec operation."""
    return MagicMock(
        success=False,
        output={
            "exit_code": 1,
            "stdout": "",
            "stderr": "NameError: name 'undefined_var' is not defined",
        },
        error={"message": "Command failed with exit code 1"},
    )
