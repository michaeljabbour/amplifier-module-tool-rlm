"""Tests for RLM data models (TrajectoryStep, RLMState)."""

from __future__ import annotations

import time

from amplifier_module_tool_rlm import RLMState, TrajectoryStep, TrajectoryStepType


class TestTrajectoryStepType:
    """Tests for TrajectoryStepType enum."""

    def test_enum_values(self):
        """Test that all expected step types exist."""
        assert TrajectoryStepType.CODE.value == "code"
        assert TrajectoryStepType.OUTPUT.value == "output"
        assert TrajectoryStepType.LLM_CALL.value == "llm_call"
        assert TrajectoryStepType.ERROR.value == "error"
        assert TrajectoryStepType.FINAL.value == "final"

    def test_enum_membership(self):
        """Test enum membership checking."""
        assert "code" in [e.value for e in TrajectoryStepType]
        assert "invalid" not in [e.value for e in TrajectoryStepType]


class TestTrajectoryStep:
    """Tests for TrajectoryStep dataclass."""

    def test_create_code_step(self):
        """Test creating a code step."""
        step = TrajectoryStep(
            step_type=TrajectoryStepType.CODE,
            content="print('hello')",
        )

        assert step.step_type == TrajectoryStepType.CODE
        assert step.content == "print('hello')"
        assert step.timestamp > 0
        assert step.metadata == {}

    def test_create_step_with_metadata(self):
        """Test creating a step with metadata."""
        step = TrajectoryStep(
            step_type=TrajectoryStepType.OUTPUT,
            content="hello",
            metadata={"exit_code": 0, "duration": 0.5},
        )

        assert step.step_type == TrajectoryStepType.OUTPUT
        assert step.metadata["exit_code"] == 0
        assert step.metadata["duration"] == 0.5

    def test_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        before = time.time()
        step = TrajectoryStep(
            step_type=TrajectoryStepType.CODE,
            content="test",
        )
        after = time.time()

        assert before <= step.timestamp <= after

    def test_custom_timestamp(self):
        """Test setting a custom timestamp."""
        custom_time = 1234567890.0
        step = TrajectoryStep(
            step_type=TrajectoryStepType.CODE,
            content="test",
            timestamp=custom_time,
        )

        assert step.timestamp == custom_time


class TestRLMState:
    """Tests for RLMState dataclass."""

    def test_initial_state(self):
        """Test initial state values."""
        state = RLMState()

        assert state.shadow_id is None
        assert state.trajectory == []
        assert state.current_depth == 0
        assert state.total_llm_calls == 0
        assert state.total_tokens_in == 0
        assert state.total_tokens_out == 0
        assert state.final_answer is None
        assert state.error is None

    def test_state_with_shadow_id(self):
        """Test state with shadow_id set."""
        state = RLMState(shadow_id="test-123")

        assert state.shadow_id == "test-123"

    def test_trajectory_append(self):
        """Test appending to trajectory."""
        state = RLMState()
        step = TrajectoryStep(
            step_type=TrajectoryStepType.CODE,
            content="print('test')",
        )
        state.trajectory.append(step)

        assert len(state.trajectory) == 1
        assert state.trajectory[0].content == "print('test')"

    def test_depth_tracking(self):
        """Test recursion depth tracking."""
        state = RLMState()

        # Simulate entering recursion
        state.current_depth += 1
        assert state.current_depth == 1

        state.current_depth += 1
        assert state.current_depth == 2

        # Simulate exiting recursion
        state.current_depth -= 1
        assert state.current_depth == 1

    def test_token_counting(self):
        """Test token counting accumulation."""
        state = RLMState()

        state.total_tokens_in += 100
        state.total_tokens_out += 50
        assert state.total_tokens_in == 100
        assert state.total_tokens_out == 50

        state.total_tokens_in += 200
        state.total_tokens_out += 100
        assert state.total_tokens_in == 300
        assert state.total_tokens_out == 150

    def test_final_answer_set(self):
        """Test setting final answer."""
        state = RLMState()

        state.final_answer = "The answer is 42"
        assert state.final_answer == "The answer is 42"

    def test_error_state(self):
        """Test error state handling."""
        state = RLMState()

        state.error = "Container failed to start"
        assert state.error == "Container failed to start"
        assert state.final_answer is None  # Error doesn't set final answer
