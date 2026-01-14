"""Integration tests for RLM tool with actual shadow environment.

These tests require:
1. Docker daemon running
2. Shadow bundle installed
3. A valid LLM provider configured

Run with: pytest tests/test_integration.py -v -m integration
Skip with: pytest tests/ -v -m "not integration"
"""

from __future__ import annotations

import os

import pytest

# Check if Docker is available
DOCKER_AVAILABLE = os.system("docker ps > /dev/null 2>&1") == 0

# Skip all tests in this module if Docker is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker daemon not running"),
]


class TestRLMIntegration:
    """Integration tests with real shadow environment."""

    @pytest.mark.asyncio
    async def test_shadow_creation_and_cleanup(self):
        """Test that shadow environment can be created and destroyed."""
        # This test validates Phase 0: Shadow integration works
        # TODO: Implement when Docker is available
        pytest.skip("Requires Docker and shadow bundle - run manually")

    @pytest.mark.asyncio
    async def test_simple_content_processing(self):
        """Test processing simple content through RLM."""
        # This test validates the basic RLM flow works end-to-end
        pytest.skip("Requires Docker and LLM provider - run manually")

    @pytest.mark.asyncio
    async def test_recursive_subcall(self):
        """Test that recursive LLM sub-calls work correctly."""
        # This test validates the recursive capability
        pytest.skip("Requires Docker and LLM provider - run manually")

    @pytest.mark.asyncio
    async def test_large_content_chunking(self):
        """Test processing content larger than context window."""
        # This test validates the chunking strategy
        pytest.skip("Requires Docker and LLM provider - run manually")


class TestRLMBenchmarks:
    """Benchmark tests against RLM paper datasets.

    These tests calibrate our implementation against the MIT paper's findings.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_oolong_benchmark(self):
        """Test against OOLONG dataset (single-hop reasoning)."""
        pytest.skip("Benchmark test - run with benchmark scripts")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sniah_benchmark(self):
        """Test against S-NIAH dataset (needle in haystack)."""
        pytest.skip("Benchmark test - run with benchmark scripts")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_browsecomp_benchmark(self):
        """Test against BrowseComp-Plus dataset."""
        pytest.skip("Benchmark test - run with benchmark scripts")
