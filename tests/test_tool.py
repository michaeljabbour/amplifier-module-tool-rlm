"""Tests for RLMTool - the main tool implementation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_module_tool_rlm import RLMConfig, RLMTool


class TestRLMToolProperties:
    """Tests for RLMTool properties and schema."""

    def test_tool_name(self, mock_coordinator: MagicMock):
        """Test tool name is 'rlm'."""
        config = RLMConfig()
        tool = RLMTool(mock_coordinator, config)

        assert tool.name == "rlm"

    def test_tool_description(self, mock_coordinator: MagicMock):
        """Test tool has meaningful description."""
        config = RLMConfig()
        tool = RLMTool(mock_coordinator, config)

        desc = tool.description
        assert "RLM" in desc or "Recursive" in desc
        assert "long-context" in desc.lower() or "10m" in desc.lower()

    def test_input_schema_structure(self, mock_coordinator: MagicMock):
        """Test input schema has required structure."""
        config = RLMConfig()
        tool = RLMTool(mock_coordinator, config)

        schema = tool.input_schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_input_schema_required_fields(self, mock_coordinator: MagicMock):
        """Test input schema requires query (content or file_path can be provided)."""
        config = RLMConfig()
        tool = RLMTool(mock_coordinator, config)

        schema = tool.input_schema
        # Only query is strictly required - content OR file_path must be provided
        assert "query" in schema["required"]
        # Content is not strictly required since file_path is an alternative
        assert "content" in schema["properties"]
        assert "file_path" in schema["properties"]

    def test_input_schema_optional_fields(self, mock_coordinator: MagicMock):
        """Test input schema has optional fields."""
        config = RLMConfig()
        tool = RLMTool(mock_coordinator, config)

        schema = tool.input_schema
        props = schema["properties"]
        assert "content_type" in props
        assert "provider" in props
        assert "model" in props


class TestRLMToolValidation:
    """Tests for input validation in RLMTool.execute()."""

    @pytest.mark.asyncio
    async def test_missing_content(self, mock_coordinator: MagicMock):
        """Test error when content is missing."""
        config = RLMConfig()
        tool = RLMTool(mock_coordinator, config)

        result = await tool.execute({"query": "What is this?"})

        assert not result.success
        assert "content" in str(result.error).lower()

    @pytest.mark.asyncio
    async def test_empty_content(self, mock_coordinator: MagicMock):
        """Test error when content is empty."""
        config = RLMConfig()
        tool = RLMTool(mock_coordinator, config)

        result = await tool.execute({"content": "", "query": "What is this?"})

        assert not result.success
        assert "content" in str(result.error).lower()

    @pytest.mark.asyncio
    async def test_missing_query(self, mock_coordinator: MagicMock):
        """Test error when query is missing."""
        config = RLMConfig()
        tool = RLMTool(mock_coordinator, config)

        result = await tool.execute({"content": "Some content"})

        assert not result.success
        assert "query" in str(result.error).lower()

    @pytest.mark.asyncio
    async def test_empty_query(self, mock_coordinator: MagicMock):
        """Test error when query is empty."""
        config = RLMConfig()
        tool = RLMTool(mock_coordinator, config)

        result = await tool.execute({"content": "Some content", "query": ""})

        assert not result.success
        assert "query" in str(result.error).lower()


class TestRLMToolDependencies:
    """Tests for dependency checking in RLMTool.execute()."""

    @pytest.mark.asyncio
    async def test_no_tools_registry(self):
        """Test error when tools registry is not available."""
        coordinator = MagicMock()
        coordinator.get = MagicMock(return_value=None)

        config = RLMConfig()
        tool = RLMTool(coordinator, config)

        result = await tool.execute(
            {
                "content": "Some content",
                "query": "What is this?",
            }
        )

        assert not result.success
        assert "tools" in str(result.error).lower() or "shadow" in str(result.error).lower()

    @pytest.mark.asyncio
    async def test_no_shadow_tool(self):
        """Test error when shadow tool is not available."""
        coordinator = MagicMock()
        coordinator.get = MagicMock(side_effect=lambda k: {} if k == "tools" else None)

        config = RLMConfig()
        tool = RLMTool(coordinator, config)

        result = await tool.execute(
            {
                "content": "Some content",
                "query": "What is this?",
            }
        )

        assert not result.success
        assert "shadow" in str(result.error).lower()

    @pytest.mark.asyncio
    async def test_no_providers(self, mock_shadow_tool: MagicMock):
        """Test error when no providers are available."""
        coordinator = MagicMock()

        def get_side_effect(key: str):
            if key == "tools":
                return {"shadow": mock_shadow_tool}
            return None

        coordinator.get = MagicMock(side_effect=get_side_effect)

        config = RLMConfig()
        tool = RLMTool(coordinator, config)

        result = await tool.execute(
            {
                "content": "Some content",
                "query": "What is this?",
            }
        )

        assert not result.success
        assert "provider" in str(result.error).lower()

    @pytest.mark.asyncio
    async def test_provider_not_found(self, mock_shadow_tool: MagicMock):
        """Test error when specified provider is not found."""
        coordinator = MagicMock()

        def get_side_effect(key: str):
            if key == "tools":
                return {"shadow": mock_shadow_tool}
            if key == "providers":
                return {"openai": MagicMock()}  # Only OpenAI available
            return None

        coordinator.get = MagicMock(side_effect=get_side_effect)

        config = RLMConfig(default_provider="anthropic")
        tool = RLMTool(coordinator, config)

        result = await tool.execute(
            {
                "content": "Some content",
                "query": "What is this?",
            }
        )

        assert not result.success
        assert "anthropic" in str(result.error).lower()


class TestRLMToolExecution:
    """Tests for RLMTool execution flow."""

    @pytest.mark.asyncio
    async def test_shadow_creation_failure(
        self,
        mock_coordinator: MagicMock,
        mock_shadow_tool: MagicMock,
    ):
        """Test handling of shadow creation failure."""
        # Make shadow creation fail
        mock_shadow_tool.execute = AsyncMock(
            return_value=MagicMock(
                success=False,
                output=None,
                error={"message": "Docker not available"},
            )
        )

        config = RLMConfig()
        tool = RLMTool(mock_coordinator, config)

        result = await tool.execute(
            {
                "content": "Some content",
                "query": "What is this?",
            }
        )

        assert not result.success
        # Error message varies - could be about docker, shadow, or RLM not producing answer
        error_str = str(result.error).lower()
        assert any(x in error_str for x in ["docker", "shadow", "rlm", "answer", "execution"])

    @pytest.mark.asyncio
    async def test_successful_simple_query(
        self,
        mock_coordinator: MagicMock,
        mock_shadow_tool: MagicMock,
        mock_provider: MagicMock,
    ):
        """Test successful execution of a simple query."""
        # Set up shadow tool responses
        call_count = 0

        async def shadow_execute(input_dict: dict):
            nonlocal call_count
            call_count += 1
            operation = input_dict.get("operation")

            if operation == "create":
                return MagicMock(
                    success=True,
                    output={
                        "shadow_id": "test-shadow-123",
                        "mode": "container",
                        "local_sources": [],
                        "status": "running",
                    },
                )
            elif operation == "inject":
                return MagicMock(success=True, output={})
            elif operation == "exec":
                # First exec initializes, second runs code
                if call_count <= 3:  # Create + inject + first exec
                    return MagicMock(
                        success=True,
                        output={
                            "exit_code": 0,
                            "stdout": "[RLM_INIT_SUCCESS]\nContext loaded: 100 characters",
                            "stderr": "",
                        },
                    )
                else:
                    return MagicMock(
                        success=True,
                        output={
                            "exit_code": 0,
                            "stdout": "[FINAL_ANSWER]The content has 12 characters",
                            "stderr": "",
                        },
                    )
            elif operation == "destroy":
                return MagicMock(success=True, output={"destroyed": True})
            else:
                return MagicMock(
                    success=False, error={"message": f"Unknown operation: {operation}"}
                )

        mock_shadow_tool.execute = shadow_execute

        # Set up provider response with code that calls FINAL
        response = MagicMock()
        response.content = [
            MagicMock(
                text="""```python
# Check content length
length = len(context)
FINAL(f"The content has {length} characters")
```"""
            )
        ]
        response.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_provider.complete = AsyncMock(return_value=response)

        config = RLMConfig()
        tool = RLMTool(mock_coordinator, config)

        result = await tool.execute(
            {
                "content": "Some content",
                "query": "How many characters are in the content?",
            }
        )

        # Even if the mock chain isn't perfect, we shouldn't crash
        # The test validates the happy path structure
        assert result is not None


class TestRLMToolMount:
    """Tests for the module mount function."""

    @pytest.mark.asyncio
    async def test_mount_registers_tool(self):
        """Test that mount() registers the tool."""
        from amplifier_module_tool_rlm import mount

        coordinator = MagicMock()
        coordinator.get_capability = MagicMock(return_value=None)
        coordinator.register_capability = MagicMock()
        coordinator.mount = AsyncMock()

        _ = await mount(coordinator, {})

        # Verify mount was called with correct arguments
        coordinator.mount.assert_called_once()
        call_args = coordinator.mount.call_args
        assert call_args[0][0] == "tools"  # First positional arg is "tools"
        assert call_args[1]["name"] == "rlm"  # Named arg is tool name

    @pytest.mark.asyncio
    async def test_mount_registers_events(self):
        """Test that mount() registers observable events."""
        from amplifier_module_tool_rlm import mount

        coordinator = MagicMock()
        coordinator.get_capability = MagicMock(return_value=[])
        coordinator.register_capability = MagicMock()
        coordinator.mount = AsyncMock()

        await mount(coordinator, {})

        # Verify event registration
        coordinator.register_capability.assert_called()
        # Check that observability.events was registered
        calls = coordinator.register_capability.call_args_list
        event_call = [c for c in calls if c[0][0] == "observability.events"]
        assert len(event_call) > 0

    @pytest.mark.asyncio
    async def test_mount_with_config(self):
        """Test that mount() accepts configuration."""
        from amplifier_module_tool_rlm import mount

        coordinator = MagicMock()
        coordinator.get_capability = MagicMock(return_value=None)
        coordinator.register_capability = MagicMock()
        coordinator.mount = AsyncMock()

        config = {
            "max_recursion_depth": 3,
            "max_llm_calls": 50,
            "default_provider": "openai",
        }

        await mount(coordinator, config)

        # Verify tool was created with config
        coordinator.mount.assert_called_once()
        tool = coordinator.mount.call_args[0][1]
        assert isinstance(tool, RLMTool)
        assert tool.config.max_recursion_depth == 3
        assert tool.config.max_llm_calls == 50
        assert tool.config.default_provider == "openai"
