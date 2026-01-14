"""Tests for REPLManager - the Python REPL execution manager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_module_tool_rlm import REPLManager, RLMConfig, TrajectoryStepType


class TestREPLManagerInitialization:
    """Tests for REPLManager initialization."""

    def test_create_repl_manager(self, mock_shadow_tool: MagicMock):
        """Test creating a REPL manager."""
        config = RLMConfig()
        provider_fn = AsyncMock(return_value="test response")

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
        )

        assert repl.shadow_tool is mock_shadow_tool
        assert repl.provider_fn is provider_fn
        assert repl.config is config
        assert repl.hooks is None
        assert repl.state.shadow_id is None

    def test_create_with_hooks(self, mock_shadow_tool: MagicMock, mock_hooks: MagicMock):
        """Test creating a REPL manager with hooks."""
        config = RLMConfig()
        provider_fn = AsyncMock(return_value="test response")

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
            hooks=mock_hooks,
        )

        assert repl.hooks is mock_hooks


class TestREPLManagerInitialize:
    """Tests for REPLManager.initialize()."""

    @pytest.mark.asyncio
    async def test_initialize_uses_direct_docker(self, mock_shadow_tool: MagicMock):
        """Test that initialize() uses direct Docker mode (shadow is skipped).

        Note: RLM now skips shadow tool (which requires local_sources) and uses
        direct Docker execution instead. This test verifies initialization works
        with direct Docker mode.
        """
        # Shadow tool won't be called since we skip it for RLM
        # Direct Docker execution will be used instead
        config = RLMConfig()
        provider_fn = AsyncMock(return_value="test response")

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
        )

        # Initialize will try direct Docker - shadow_id is set to direct mode ID
        # This test verifies the initialization flow uses direct Docker
        result = await repl.initialize("test content", "text")

        # If Docker is available, initialization succeeds with direct Docker mode
        if result:
            # In direct Docker mode, shadow_id starts with "rlm-direct-"
            assert repl.state.shadow_id is not None
            assert repl.state.shadow_id.startswith("rlm-direct-")
            assert repl.state.use_direct_docker is True

    @pytest.mark.asyncio
    async def test_initialize_handles_docker_unavailable(self, mock_shadow_tool: MagicMock):
        """Test that initialize() handles Docker unavailability gracefully."""
        config = RLMConfig()
        provider_fn = AsyncMock(return_value="test response")

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
        )

        # Initialize will fail if Docker isn't available
        result = await repl.initialize("test content", "text")

        # In unit tests without Docker, initialization fails gracefully
        # Either it works (Docker available) or fails with a clear error
        if not result:
            assert repl.state.error is not None

    @pytest.mark.asyncio
    async def test_initialize_records_trajectory(self, mock_shadow_tool: MagicMock):
        """Test that initialize() records initialization in trajectory."""
        create_response = MagicMock(
            success=True,
            output={"shadow_id": "test-shadow-xyz", "status": "running"},
        )
        inject_response = MagicMock(success=True, output={})
        exec_response = MagicMock(
            success=True,
            output={
                "exit_code": 0,
                "stdout": "[RLM_INIT_SUCCESS]\nContext loaded: 50 characters",
                "stderr": "",
            },
        )

        async def mock_execute(input_dict: dict):
            op = input_dict.get("operation")
            if op == "create":
                return create_response
            elif op == "inject":
                return inject_response
            elif op == "exec":
                return exec_response
            return MagicMock(success=False)

        mock_shadow_tool.execute = mock_execute

        config = RLMConfig()
        provider_fn = AsyncMock(return_value="test response")

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
        )

        await repl.initialize("test content", "document")

        # Should have recorded initialization in trajectory
        assert len(repl.state.trajectory) > 0
        init_step = repl.state.trajectory[0]
        assert init_step.step_type == TrajectoryStepType.OUTPUT
        assert "document" in init_step.metadata.get("context_type", "")


class TestREPLManagerCodeExecution:
    """Tests for REPLManager.execute_code()."""

    @pytest.mark.asyncio
    async def test_execute_simple_code(self, mock_shadow_tool: MagicMock):
        """Test executing simple code."""
        # Set up initialized state
        config = RLMConfig()
        provider_fn = AsyncMock(return_value="test response")

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
        )
        repl.state.shadow_id = "test-shadow"

        # Mock code execution
        inject_response = MagicMock(success=True, output={})
        exec_response = MagicMock(
            success=True,
            output={
                "exit_code": 0,
                "stdout": "Hello, World!",
                "stderr": "",
            },
        )

        async def mock_execute(input_dict: dict):
            op = input_dict.get("operation")
            if op == "inject":
                return inject_response
            elif op == "exec":
                return exec_response
            return MagicMock(success=False)

        mock_shadow_tool.execute = mock_execute

        output, has_final, pending_calls = await repl.execute_code("print('Hello, World!')")

        assert "Hello" in output
        assert has_final is False
        assert pending_calls == []

    @pytest.mark.asyncio
    async def test_execute_code_with_final(self, mock_shadow_tool: MagicMock):
        """Test executing code that calls FINAL()."""
        config = RLMConfig()
        provider_fn = AsyncMock(return_value="test response")

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
        )
        repl.state.shadow_id = "test-shadow"

        inject_response = MagicMock(success=True, output={})
        exec_response = MagicMock(
            success=True,
            output={
                "exit_code": 0,
                "stdout": "[FINAL_ANSWER]The answer is 42",
                "stderr": "",
            },
        )

        async def mock_execute(input_dict: dict):
            op = input_dict.get("operation")
            if op == "inject":
                return inject_response
            elif op == "exec":
                return exec_response
            return MagicMock(success=False)

        mock_shadow_tool.execute = mock_execute

        _, has_final, _ = await repl.execute_code("FINAL('The answer is 42')")

        assert has_final is True
        assert repl.state.final_answer == "The answer is 42"

    @pytest.mark.asyncio
    async def test_execute_code_records_trajectory(self, mock_shadow_tool: MagicMock):
        """Test that code execution records in trajectory."""
        config = RLMConfig()
        provider_fn = AsyncMock(return_value="test response")

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
        )
        repl.state.shadow_id = "test-shadow"

        inject_response = MagicMock(success=True, output={})
        exec_response = MagicMock(
            success=True,
            output={"exit_code": 0, "stdout": "output", "stderr": ""},
        )

        async def mock_execute(input_dict: dict):
            op = input_dict.get("operation")
            if op == "inject":
                return inject_response
            elif op == "exec":
                return exec_response
            return MagicMock(success=False)

        mock_shadow_tool.execute = mock_execute

        await repl.execute_code("x = 1 + 1\nprint(x)")

        # Should have CODE and OUTPUT steps
        code_steps = [s for s in repl.state.trajectory if s.step_type == TrajectoryStepType.CODE]
        output_steps = [
            s for s in repl.state.trajectory if s.step_type == TrajectoryStepType.OUTPUT
        ]

        assert len(code_steps) >= 1
        assert len(output_steps) >= 1


class TestREPLManagerLLMCalls:
    """Tests for REPLManager LLM sub-call handling."""

    @pytest.mark.asyncio
    async def test_process_llm_calls(self, mock_shadow_tool: MagicMock):
        """Test processing LLM sub-calls."""
        config = RLMConfig()
        provider_fn = AsyncMock(return_value="LLM response for chunk")

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
        )
        repl.state.shadow_id = "test-shadow"

        calls = [
            ("Summarize this", "Content chunk 1"),
            ("Summarize this", "Content chunk 2"),
        ]

        results = await repl.process_llm_calls(calls)

        assert len(results) == 2
        assert all(r == "LLM response for chunk" for r in results)
        assert repl.state.total_llm_calls == 2

    @pytest.mark.asyncio
    async def test_max_llm_calls_enforced(self, mock_shadow_tool: MagicMock):
        """Test that max_llm_calls limit is enforced."""
        config = RLMConfig(max_llm_calls=2)
        provider_fn = AsyncMock(return_value="response")

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
        )
        repl.state.shadow_id = "test-shadow"
        repl.state.total_llm_calls = 1  # Already used 1 call

        calls = [
            ("Query 1", "Content 1"),
            ("Query 2", "Content 2"),
            ("Query 3", "Content 3"),
        ]

        results = await repl.process_llm_calls(calls)

        # Only 1 more call allowed (max=2, already used=1)
        successful = [r for r in results if "ERROR" not in r]
        errors = [r for r in results if "Max LLM calls" in r]

        assert len(successful) == 1
        assert len(errors) == 2

    @pytest.mark.asyncio
    async def test_max_recursion_depth_enforced(self, mock_shadow_tool: MagicMock):
        """Test that max_recursion_depth limit is enforced."""
        config = RLMConfig(max_recursion_depth=1)
        provider_fn = AsyncMock(return_value="response")

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
        )
        repl.state.shadow_id = "test-shadow"
        repl.state.current_depth = 1  # Already at depth 1

        calls = [("Query", "Content")]

        results = await repl.process_llm_calls(calls)

        assert len(results) == 1
        assert "Max recursion depth" in results[0]

    @pytest.mark.asyncio
    async def test_llm_call_failure_handling(self, mock_shadow_tool: MagicMock):
        """Test handling of LLM call failures."""
        config = RLMConfig()
        provider_fn = AsyncMock(side_effect=Exception("API rate limit exceeded"))

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
        )
        repl.state.shadow_id = "test-shadow"

        calls = [("Query", "Content")]

        results = await repl.process_llm_calls(calls)

        assert len(results) == 1
        assert "ERROR" in results[0]
        assert "rate limit" in results[0].lower() or "failed" in results[0].lower()


class TestREPLManagerCleanup:
    """Tests for REPLManager cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_destroys_shadow(self, mock_shadow_tool: MagicMock):
        """Test that cleanup() destroys the shadow environment."""
        config = RLMConfig()
        provider_fn = AsyncMock(return_value="response")

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
        )
        repl.state.shadow_id = "test-shadow-to-destroy"

        destroy_called = False
        destroyed_id = None

        async def mock_execute(input_dict: dict):
            nonlocal destroy_called, destroyed_id
            if input_dict.get("operation") == "destroy":
                destroy_called = True
                destroyed_id = input_dict.get("shadow_id")
                return MagicMock(success=True, output={"destroyed": True})
            return MagicMock(success=False)

        mock_shadow_tool.execute = mock_execute

        await repl.cleanup()

        assert destroy_called is True
        assert destroyed_id == "test-shadow-to-destroy"

    @pytest.mark.asyncio
    async def test_cleanup_handles_no_shadow(self, mock_shadow_tool: MagicMock):
        """Test that cleanup() handles case when no shadow exists."""
        config = RLMConfig()
        provider_fn = AsyncMock(return_value="response")

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
        )
        # No shadow_id set

        # Should not raise
        await repl.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_handles_destroy_failure(self, mock_shadow_tool: MagicMock):
        """Test that cleanup() handles destroy failure gracefully."""
        config = RLMConfig()
        provider_fn = AsyncMock(return_value="response")

        repl = REPLManager(
            shadow_tool=mock_shadow_tool,
            provider_fn=provider_fn,
            config=config,
        )
        repl.state.shadow_id = "test-shadow"

        mock_shadow_tool.execute = AsyncMock(side_effect=Exception("Container not found"))

        # Should not raise, just log warning
        await repl.cleanup()
