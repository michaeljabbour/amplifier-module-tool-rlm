"""Tests for RLM configuration models."""

from __future__ import annotations

from amplifier_module_tool_rlm import RLMConfig


class TestRLMConfig:
    """Tests for RLMConfig validation and defaults."""

    def test_default_values(self):
        """Test that default configuration values are sensible."""
        config = RLMConfig()

        assert config.max_recursion_depth == 5
        assert config.max_llm_calls == 100
        assert config.max_trajectory_steps == 50
        assert config.exec_timeout == 60
        assert config.default_provider == "anthropic"
        assert config.default_model is None
        assert config.chars_per_token == 4.0

    def test_custom_values(self):
        """Test that custom values are accepted."""
        config = RLMConfig(
            max_recursion_depth=10,
            max_llm_calls=200,
            max_trajectory_steps=100,
            exec_timeout=120,
            default_provider="openai",
            default_model="gpt-4",
            chars_per_token=3.5,
        )

        assert config.max_recursion_depth == 10
        assert config.max_llm_calls == 200
        assert config.max_trajectory_steps == 100
        assert config.exec_timeout == 120
        assert config.default_provider == "openai"
        assert config.default_model == "gpt-4"
        assert config.chars_per_token == 3.5

    def test_from_dict(self):
        """Test creating config from dictionary (as mount() receives)."""
        config_dict = {
            "max_recursion_depth": 3,
            "max_llm_calls": 50,
        }
        # Use model_validate for dict input (Pydantic v2 idiom)
        config = RLMConfig.model_validate(config_dict)

        assert config.max_recursion_depth == 3
        assert config.max_llm_calls == 50
        # Defaults for unspecified fields
        assert config.max_trajectory_steps == 50
        assert config.default_provider == "anthropic"

    def test_validation_positive_integers(self):
        """Test that configuration validates positive values."""
        # These should work (positive values)
        config = RLMConfig(max_recursion_depth=1, max_llm_calls=1)
        assert config.max_recursion_depth == 1
        assert config.max_llm_calls == 1

    def test_zero_values_allowed(self):
        """Test that zero values are allowed (edge case)."""
        # Zero might be valid for some use cases (disable recursion, etc.)
        config = RLMConfig(max_recursion_depth=0, max_llm_calls=0)
        assert config.max_recursion_depth == 0
        assert config.max_llm_calls == 0
