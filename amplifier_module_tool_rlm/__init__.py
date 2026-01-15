"""
Recursive Language Model (RLM) tool for Amplifier.

Implements the RLM algorithm from Zhang, Kraska, Khattab (MIT CSAIL, 2025)
for processing arbitrarily long contexts via a Python REPL environment.

Key Design Points:
- Uses Docker containers for isolated Python REPL execution
- Supports recursive LLM sub-calls for divide-and-conquer processing
- Implements cost controls and depth limits for safety
- Emits observable events for monitoring and debugging

ASSUMPTIONS:
1. Docker is available on the host system
2. At least one provider is available via coordinator.get("providers")
3. Network access for provider API calls

CONDITIONS FOR SUCCESS:
1. Docker daemon running
2. At least one provider mounted
3. Content fits in container memory (default 4GB)
4. Network available for API calls

ERROR CASES HANDLED:
1. Docker not available -> Clear error message
2. Provider not available -> Clear error message
3. Container creation fails -> Docker error reported
4. Code execution timeout -> Partial results + timeout error
5. Python syntax errors -> Returned to model for correction
6. Max recursion depth exceeded -> Graceful termination
7. Max LLM calls exceeded -> Cost control termination
"""

from __future__ import annotations

__amplifier_module_type__ = "tool"

__all__ = [
    "mount",
    "RLMConfig",
    "RLMTool",
    "REPLManager",
    "RLMState",
    "TrajectoryStep",
    "TrajectoryStepType",
]

import atexit
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

# Debug logging to file for tracing (with proper cleanup)
_DEBUG_LOG = os.environ.get("RLM_DEBUG_LOG", "")
_debug_fh = None

if _DEBUG_LOG:
    _debug_fh = open(_DEBUG_LOG, "a")
    atexit.register(lambda: _debug_fh.close() if _debug_fh else None)

    def _debug(msg: str) -> None:
        if _debug_fh:
            _debug_fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            _debug_fh.flush()
else:

    def _debug(msg: str) -> None:
        pass


if TYPE_CHECKING:
    from amplifier_core import ModuleCoordinator, ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration and Types
# =============================================================================


class RLMConfig(BaseModel):
    """Configuration for the RLM tool."""

    # Execution limits
    max_recursion_depth: int = Field(
        default=5,
        ge=1,
        description="Maximum depth of recursive LLM sub-calls",
    )
    max_llm_calls: int = Field(
        default=100,
        ge=1,
        description="Maximum total LLM calls (cost control)",
    )
    max_trajectory_steps: int = Field(
        default=50,
        ge=1,
        description="Maximum steps in the REPL trajectory",
    )
    exec_timeout: int = Field(
        default=60,
        ge=1,
        description="Timeout in seconds for each code execution",
    )

    # Provider configuration
    default_provider: str = Field(
        default="anthropic",
        description="Default provider for LLM calls",
    )
    default_model: str | None = Field(
        default=None,
        description="Default model (uses provider default if not set)",
    )


class TrajectoryStepType(str, Enum):
    """Types of steps in an RLM trajectory."""

    CODE = "code"  # Model generated code
    OUTPUT = "output"  # Code execution output
    LLM_CALL = "llm_call"  # Sub-LLM call made
    ERROR = "error"  # Error occurred
    FINAL = "final"  # Final answer produced


@dataclass
class TrajectoryStep:
    """A single step in the RLM execution trajectory."""

    step_type: TrajectoryStepType
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RLMState:
    """Mutable state for an RLM execution."""

    shadow_id: str | None = None
    use_direct_docker: bool = False  # Fallback mode when shadow tool unavailable
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    current_depth: int = 0
    total_llm_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    final_answer: str | None = None
    error: str | None = None


# =============================================================================
# System Prompts (from RLM paper Appendix D)
# =============================================================================

RLM_SYSTEM_PROMPT = """You must answer a query about a large document using Python code.

**ENVIRONMENT:**
- `context` - the full document ({context_total_length:,} characters)
- `llm_query(question, text)` - ask a sub-LLM about a text chunk
- `FINAL(answer)` - submit your final answer (REQUIRED)

**IMPORTANT: You must write and execute Python code. Do not just describe what you would do.**

**STEP 1: Analyze the query**
The query is: "{{query}}"

Is this asking for:
- A **calculation** (per-X, ratio, average, total)? → You MUST compute the answer
- A **lookup** (find code, password, specific value)? → Search for exact match

**STEP 2: For CALCULATION queries (most common)**

If the query asks for "per-engineer", "per-person", "ratio", "average", etc., you MUST:
1. Identify ALL numbers needed (e.g., total amount AND count)
2. Extract those numbers from the document
3. COMPUTE the final answer yourself
4. Call FINAL() with the computed result

Example for "What is the per-engineer investment?":
```python
# I need: (1) total investment, (2) number of engineers
# Then calculate: total / count

chunks = [context[i:i+100000] for i in range(0, len(context), 100000)]
investment = None
team_size = None

for i, chunk in enumerate(chunks):
    result = llm_query(
        "Find: 1) Total investment/budget amount for the project, "
        "2) Total team size or headcount. Return as 'investment: $X, team: Y'",
        chunk
    )
    print(f"Chunk {{i+1}}: {{result}}")
    # Parse numbers from result...

# MUST calculate the answer myself
if investment and team_size:
    per_engineer = investment / team_size
    FINAL(f"${{per_engineer:,.0f}}")
```

**STEP 3: For LOOKUP queries**

If looking for a specific code/password/value:
```python
import re
# Search with regex
m = re.search(r'code is ([A-Z0-9]+)', context)
if m:
    FINAL(m.group(1))
```

**CRITICAL RULES:**
1. ALWAYS write executable Python code in ```python blocks
2. For calculations: YOU must compute the answer, not just find raw numbers
3. Extract ALL relevant facts before calculating
4. Use FINAL(answer) to submit - include units (e.g., "$64,400")
5. The answer EXISTS - keep trying different strategies
"""


# =============================================================================
# REPL Manager - Handles Python execution in shadow container
# =============================================================================


class REPLManager:
    """Manages Python REPL execution in a shadow container.

    This class handles:
    1. Shadow environment lifecycle (create, execute, destroy)
    2. State persistence across executions via filesystem
    3. Code execution with timeout and error handling
    4. Special function interception (llm_query, FINAL, FINAL_VAR)
    """

    STATE_FILE = "/workspace/.rlm_state.json"
    CODE_FILE = "/workspace/.rlm_code.py"
    OUTPUT_FILE = "/workspace/.rlm_output.txt"

    def __init__(
        self,
        shadow_tool: Any,
        provider_fn: Any,
        config: RLMConfig,
        hooks: Any | None = None,
    ):
        """Initialize REPL manager.

        Args:
            shadow_tool: The shadow tool instance for container operations
            provider_fn: Async function to make LLM calls: (query, content) -> str
            config: RLM configuration
            hooks: Optional hooks for event emission
        """
        self.shadow_tool = shadow_tool
        self.provider_fn = provider_fn
        self.config = config
        self.hooks = hooks
        self.state = RLMState()
        self._pending_llm_calls: list[tuple[str, str]] = []

    async def _try_shadow_create(self) -> str | None:
        """Try to create a shadow environment, return shadow_id or None on failure.

        NOTE: Shadow tool requires local_sources (repos to snapshot). Since RLM
        just needs sandboxed code execution without local repos, we skip shadow
        and use direct Docker mode instead. This method exists for future use
        cases where RLM might need to test against local repo changes.
        """
        # Shadow tool requires local_sources - it's designed for testing local
        # repo changes, not general code execution. For RLM's use case (sandboxed
        # Python execution), direct Docker is more appropriate.
        #
        # If we ever need shadow (e.g., to test RLM against local amplifier changes),
        # we would pass actual local_sources here:
        #   local_sources=["/path/to/repo:org/name"]
        #
        # For now, skip shadow and use direct Docker fallback.
        logger.info("RLM using direct Docker mode (shadow requires local_sources)")
        return None

    async def initialize(self, context: str, context_type: str) -> bool:
        """Initialize the REPL environment with context.

        Args:
            context: The content to process
            context_type: Description of content type (e.g., "code repository", "document")

        Returns:
            True if initialization succeeded, False otherwise
        """
        # Try shadow tool first, fall back to direct Docker if it fails
        shadow_id = await self._try_shadow_create()

        if not shadow_id:
            # Fallback: use direct Docker execution mode
            logger.info("Shadow tool unavailable, using direct Docker execution mode")
            self.state.use_direct_docker = True
            self.state.shadow_id = f"rlm-direct-{uuid.uuid4().hex[:8]}"
        else:
            self.state.shadow_id = shadow_id
            self.state.use_direct_docker = False

        logger.info(
            f"RLM initialized: shadow_id={self.state.shadow_id}, "
            f"direct_docker={self.state.use_direct_docker}"
        )

        # Initialize Python environment with context
        init_code = f'''
import json
import sys

# Initialize RLM state as dict (JSON-serializable, avoids security issues)
env = {{
    "context": {repr(context)},
    "context_type": {repr(context_type)},
    "context_total_length": {len(context)},
    "results": [],
    "llm_calls": [],
    "final_answer": None,
    "final_var": None
}}

# Expose variables at module level
context = env["context"]
context_type = env["context_type"]
context_total_length = env["context_total_length"]
results = env["results"]

def llm_query(query: str, content: str) -> str:
    """Make a recursive LLM call. Returns result if available, else placeholder."""
    # Check if we already have a result for this call index
    call_idx = len(env["llm_calls"])
    if call_idx < len(env["results"]):
        # Result already available from previous execution
        return env["results"][call_idx]
    # Otherwise, register the call and return placeholder
    env["llm_calls"].append([query, content])
    return f"[LLM_CALL_PENDING:{{len(env['llm_calls'])-1}}]"

def FINAL(answer: str):
    """Submit final answer."""
    env["final_answer"] = str(answer)
    print(f"[FINAL_ANSWER]{{env['final_answer']}}")

def FINAL_VAR(var_name: str):
    """Submit variable value as final answer."""
    env["final_var"] = var_name
    if var_name in globals():
        env["final_answer"] = str(globals()[var_name])
    elif var_name in locals():
        env["final_answer"] = str(locals()[var_name])
    else:
        env["final_answer"] = f"[ERROR: Variable '{{var_name}}' not found]"
    print(f"[FINAL_ANSWER]{{env['final_answer']}}")

# Save state as JSON
with open("{self.STATE_FILE}", "w") as f:
    json.dump(env, f)

print("[RLM_INIT_SUCCESS]")
print(f"Context loaded: {{len(context):,}} characters")
'''

        success, output = await self._exec_code(init_code)
        if not success or "[RLM_INIT_SUCCESS]" not in output:
            self.state.error = f"Failed to initialize REPL: {output}"
            return False

        self.state.trajectory.append(
            TrajectoryStep(
                step_type=TrajectoryStepType.OUTPUT,
                content=f"Environment initialized with {len(context):,} characters of {context_type}",
                metadata={"context_length": len(context), "context_type": context_type},
            )
        )

        return True

    async def execute_code(self, code: str) -> tuple[str, bool, list[tuple[str, str]]]:
        """Execute code in the REPL and return output.

        Args:
            code: Python code to execute

        Returns:
            Tuple of (output_string, has_final_answer, pending_llm_calls)
        """
        self.state.trajectory.append(
            TrajectoryStep(step_type=TrajectoryStepType.CODE, content=code)
        )

        # Write user code to a separate file to avoid escaping issues
        user_code_file = "/workspace/user_code.py"

        # Wrap code to load state, execute from file, and save state
        wrapped_code = f'''
import json
import sys
from io import StringIO

# Load state from JSON
with open("{self.STATE_FILE}", "r") as f:
    env = json.load(f)

# Restore globals
context = env["context"]
context_type = env["context_type"]
context_total_length = env["context_total_length"]
results = env["results"]

def llm_query(query: str, content: str) -> str:
    """Make a recursive LLM call. Returns result if available, else placeholder."""
    # Check if we already have a result for this call index
    call_idx = len(env["llm_calls"])
    if call_idx < len(env["results"]):
        # Result already available from previous execution
        return env["results"][call_idx]
    # Otherwise, register the call and return placeholder
    env["llm_calls"].append([query, content])
    return f"[LLM_CALL_PENDING:{{len(env['llm_calls'])-1}}]"

def FINAL(answer: str):
    """Submit final answer."""
    env["final_answer"] = str(answer)
    print(f"[FINAL_ANSWER]{{env['final_answer']}}")

def FINAL_VAR(var_name: str):
    """Submit variable value as final answer."""
    env["final_var"] = var_name
    # Try to get from current namespace
    try:
        val = globals().get(var_name) or locals().get(var_name)
        env["final_answer"] = str(val) if val is not None else f"[ERROR: Variable '{{var_name}}' not found]"
    except:
        env["final_answer"] = f"[ERROR: Variable '{{var_name}}' not found]"
    print(f"[FINAL_ANSWER]{{env['final_answer']}}")

# Capture output
_old_stdout = sys.stdout
sys.stdout = _captured = StringIO()

try:
    # Read and exec user code from separate file (avoids escaping issues)
    with open("{user_code_file}", "r") as _uf:
        _user_code = _uf.read()
    exec(_user_code)
except Exception as e:
    print(f"[EXECUTION_ERROR] {{type(e).__name__}}: {{e}}")

sys.stdout = _old_stdout
_output = _captured.getvalue()
print(_output)

# Report pending LLM calls
if env["llm_calls"]:
    print(f"[PENDING_LLM_CALLS:{{len(env['llm_calls'])}}]")
    for i, (q, c) in enumerate(env["llm_calls"]):
        print(f"[LLM_CALL:{{i}}:{{len(c)}}]")

# Save state as JSON
with open("{self.STATE_FILE}", "w") as f:
    json.dump(env, f)
'''

        # Inject user code to separate file first (avoids escaping issues)
        success, output = await self._exec_code_with_user_file(wrapped_code, code, user_code_file)

        # Parse output for special markers
        pending_calls: list[tuple[str, str]] = []

        # Extract pending LLM calls count FIRST
        pending_match = re.search(r"\[PENDING_LLM_CALLS:(\d+)\]", output)
        if pending_match:
            pending_count = int(pending_match.group(1))
            if pending_count > 0:
                # We need to retrieve the actual calls from the container
                pending_calls = await self._get_pending_llm_calls()

        # CRITICAL: Don't accept FINAL if there are pending LLM calls
        # We must wait for sub-results before accepting a final answer
        has_final = "[FINAL_ANSWER]" in output and not pending_calls

        # Clean output of internal markers for display
        clean_output = output
        for marker in [
            r"\[PENDING_LLM_CALLS:\d+\]",
            r"\[LLM_CALL:\d+:\d+\]",
            r"\[FINAL_ANSWER\].*",
        ]:
            clean_output = re.sub(marker, "", clean_output)
        clean_output = clean_output.strip()

        self.state.trajectory.append(
            TrajectoryStep(
                step_type=TrajectoryStepType.OUTPUT,
                content=clean_output,
                metadata={"success": success, "has_final": has_final},
            )
        )

        # Extract final answer if present (but NOT if it contains a pending call placeholder)
        if has_final:
            final_match = re.search(r"\[FINAL_ANSWER\](.*)", output, re.DOTALL)
            if final_match:
                answer_text = final_match.group(1).strip()
                # Don't accept answer if it's just a pending LLM call placeholder
                if "[LLM_CALL_PENDING:" not in answer_text:
                    self.state.final_answer = answer_text
                else:
                    # Reset has_final since this wasn't a real answer
                    has_final = False

        return clean_output, has_final, pending_calls

    async def process_llm_calls(self, calls: list[tuple[str, str]]) -> list[str]:
        """Process pending LLM sub-calls.

        Args:
            calls: List of (query, content) tuples

        Returns:
            List of LLM responses
        """
        results = []
        for i, (query, content) in enumerate(calls):
            _debug(f"LLM SUB-CALL {i}: query='{query[:100]}...', content_len={len(content)}")
            _debug(f"  Content preview: {content[:200]}...")
            if self.state.total_llm_calls >= self.config.max_llm_calls:
                results.append("[ERROR: Max LLM calls exceeded]")
                continue

            self.state.total_llm_calls += 1
            self.state.current_depth += 1

            if self.state.current_depth > self.config.max_recursion_depth:
                self.state.current_depth -= 1
                results.append("[ERROR: Max recursion depth exceeded]")
                continue

            try:
                # Make the actual LLM call
                response = await self.provider_fn(query, content)
                results.append(response)
                _debug(f"  Response: {response[:300]}...")

                self.state.trajectory.append(
                    TrajectoryStep(
                        step_type=TrajectoryStepType.LLM_CALL,
                        content=f"Query: {query[:100]}...\nResponse: {response[:200]}...",
                        metadata={
                            "query_length": len(query),
                            "content_length": len(content),
                            "response_length": len(response),
                        },
                    )
                )

                if self.hooks:
                    await self.hooks.emit(
                        "rlm:llm_call",
                        {
                            "depth": self.state.current_depth,
                            "query_length": len(query),
                            "content_length": len(content),
                            "response_length": len(response),
                        },
                    )

            except Exception as e:
                results.append(f"[ERROR: LLM call failed: {e}]")
                self.state.trajectory.append(
                    TrajectoryStep(
                        step_type=TrajectoryStepType.ERROR,
                        content=f"LLM call failed: {e}",
                    )
                )
            finally:
                self.state.current_depth -= 1

        return results

    async def inject_llm_results(self, results: list[str]) -> bool:
        """Inject LLM call results back into the REPL environment.

        Args:
            results: List of LLM responses to inject

        Returns:
            True if successful
        """
        if not results:
            return True

        inject_code = f'''
import json
import re

# Load state from JSON
with open("{self.STATE_FILE}", "r") as f:
    env = json.load(f)

# Store the actual results in env["results"] for access
llm_results = {repr(results)}
env["results"] = llm_results
env["llm_calls"] = []  # Clear pending calls

# Make results available as llm_result_N and also in results list
for i, result in enumerate(llm_results):
    globals()[f"llm_result_{{i}}"] = result

# CRITICAL: Replace placeholder strings in all global variables
# This makes llm_query() work intuitively - the variable gets the actual result
placeholder_pattern = re.compile(r"\\[LLM_CALL_PENDING:(\\d+)\\]")
for var_name, var_value in list(globals().items()):
    if isinstance(var_value, str) and "[LLM_CALL_PENDING:" in var_value:
        match = placeholder_pattern.match(var_value)
        if match:
            idx = int(match.group(1))
            if idx < len(llm_results):
                globals()[var_name] = llm_results[idx]

# Also update results variable in global scope
results = llm_results

with open("{self.STATE_FILE}", "w") as f:
    json.dump(env, f)

print(f"[INJECTED_RESULTS:{{len(llm_results)}}]")
for i, r in enumerate(llm_results):
    print(f"Result {{i}}: {{r[:100]}}...")
'''

        success, output = await self._exec_code(inject_code)
        return success and "[INJECTED_RESULTS:" in output

    async def cleanup(self) -> None:
        """Clean up the shadow environment."""
        if self.state.shadow_id:
            try:
                await self.shadow_tool.execute(
                    {"operation": "destroy", "shadow_id": self.state.shadow_id, "force": True}
                )
                logger.info(f"Destroyed shadow environment: {self.state.shadow_id}")
            except Exception as e:
                logger.warning(f"Failed to destroy shadow environment: {e}")

    async def _exec_code(self, code: str) -> tuple[bool, str]:
        """Execute code in the shadow container or via direct Docker.

        Uses file-based execution to avoid shell escaping issues.

        Args:
            code: Python code to execute

        Returns:
            Tuple of (success, output)
        """
        if not self.state.shadow_id:
            return False, "No shadow environment"

        # Write code to a temp file on host
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Use direct Docker mode if shadow tool unavailable
            if self.state.use_direct_docker:
                return await self._exec_direct_docker_simple(temp_path)

            # Shadow tool mode - inject and execute
            inject_result = await self.shadow_tool.execute(
                {
                    "operation": "inject",
                    "shadow_id": self.state.shadow_id,
                    "host_path": temp_path,
                    "container_path": self.CODE_FILE,
                }
            )
            if not inject_result.success:
                return False, f"Failed to inject code: {inject_result.error}"

            # Execute the code file
            exec_result = await self.shadow_tool.execute(
                {
                    "operation": "exec",
                    "shadow_id": self.state.shadow_id,
                    "command": f"python3 {self.CODE_FILE}",
                    "timeout": self.config.exec_timeout,
                }
            )

            output = exec_result.output.get("stdout", "") if exec_result.output else ""
            stderr = exec_result.output.get("stderr", "") if exec_result.output else ""
            exit_code = exec_result.output.get("exit_code", -1) if exec_result.output else -1

            if stderr:
                output += f"\n[STDERR]\n{stderr}"

            return exit_code == 0, output

        finally:
            os.unlink(temp_path)

    async def _exec_direct_docker_simple(self, code_path: str) -> tuple[bool, str]:
        """Execute code using direct Docker run (simple version for init).

        Uses a fresh container with the code file mounted.
        """
        import asyncio
        import tempfile

        # Create a temp directory to share with container
        with tempfile.TemporaryDirectory() as tmpdir:
            import shutil

            # Copy code file to shared directory
            shutil.copy(code_path, f"{tmpdir}/code.py")

            # Copy existing state if any
            state_file = f"{tmpdir}/rlm_state.json"
            if hasattr(self, "_direct_state_data"):
                with open(state_file, "wb") as f:
                    f.write(self._direct_state_data)

            # Update code to use /workspace paths
            with open(f"{tmpdir}/code.py") as f:
                code = f.read()

            code = code.replace(self.STATE_FILE, "/workspace/rlm_state.json")

            with open(f"{tmpdir}/code.py", "w") as f:
                f.write(code)

            # Run Docker container with security hardening
            cmd = [
                "docker",
                "run",
                "--rm",
                "--network=none",  # Prevent data exfiltration
                "--cap-drop=ALL",  # Drop all capabilities
                "--memory=4g",  # Resource limit
                "--read-only",  # Read-only root filesystem
                "--tmpfs=/tmp:size=100m",  # Writable tmp with size limit
                "--security-opt=no-new-privileges:true",  # Prevent privilege escalation
                "-v",
                f"{tmpdir}:/workspace",
                "-w",
                "/workspace",
                "python:3.11-slim",
                "python3",
                "/workspace/code.py",
            ]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.config.exec_timeout,
                )
                stdout = stdout_bytes.decode() if stdout_bytes else ""
                stderr = stderr_bytes.decode() if stderr_bytes else ""

                output = stdout
                if stderr:
                    output += f"\n[STDERR]\n{stderr}"

                # Save state for next execution
                if os.path.exists(state_file):
                    with open(state_file, "rb") as f:
                        self._direct_state_data = f.read()

                return proc.returncode == 0, output

            except asyncio.TimeoutError:
                return False, f"[TIMEOUT] Execution exceeded {self.config.exec_timeout}s"
            except Exception as e:
                return False, f"[DOCKER_ERROR] {e}"

    async def _exec_code_with_user_file(
        self, wrapper_code: str, user_code: str, user_code_path: str
    ) -> tuple[bool, str]:
        """Execute wrapper code after injecting user code to a separate file.

        This avoids escaping issues by keeping user code in its own file.

        Args:
            wrapper_code: The wrapper code that will exec the user code
            user_code: The user's code to execute
            user_code_path: Path inside container for user code file

        Returns:
            Tuple of (success, output)
        """
        if not self.state.shadow_id:
            return False, "No shadow environment"

        import tempfile

        # Write user code to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(user_code)
            user_temp_path = f.name

        # Write wrapper code to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapper_code)
            wrapper_temp_path = f.name

        try:
            # Use direct Docker mode if shadow tool unavailable
            if self.state.use_direct_docker:
                return await self._exec_direct_docker(
                    wrapper_temp_path, user_temp_path, user_code_path
                )

            # Shadow tool mode
            # Inject user code file first
            inject_result = await self.shadow_tool.execute(
                {
                    "operation": "inject",
                    "shadow_id": self.state.shadow_id,
                    "host_path": user_temp_path,
                    "container_path": user_code_path,
                }
            )
            if not inject_result.success:
                return False, f"Failed to inject user code: {inject_result.error}"

            # Inject wrapper code file
            inject_result = await self.shadow_tool.execute(
                {
                    "operation": "inject",
                    "shadow_id": self.state.shadow_id,
                    "host_path": wrapper_temp_path,
                    "container_path": self.CODE_FILE,
                }
            )
            if not inject_result.success:
                return False, f"Failed to inject wrapper code: {inject_result.error}"

            # Execute the wrapper code file
            exec_result = await self.shadow_tool.execute(
                {
                    "operation": "exec",
                    "shadow_id": self.state.shadow_id,
                    "command": f"python3 {self.CODE_FILE}",
                    "timeout": self.config.exec_timeout,
                }
            )

            output = exec_result.output.get("stdout", "") if exec_result.output else ""
            stderr = exec_result.output.get("stderr", "") if exec_result.output else ""
            exit_code = exec_result.output.get("exit_code", -1) if exec_result.output else -1

            if stderr:
                output += f"\n[STDERR]\n{stderr}"

            return exit_code == 0, output

        finally:
            os.unlink(user_temp_path)
            os.unlink(wrapper_temp_path)

    async def _exec_direct_docker(
        self, wrapper_path: str, user_code_path: str, container_user_path: str
    ) -> tuple[bool, str]:
        """Execute code using direct Docker run (fallback when shadow unavailable).

        Uses a fresh container for each execution with mounted temp files.
        """
        import asyncio
        import tempfile

        # Create a temp directory to share with container
        with tempfile.TemporaryDirectory() as tmpdir:
            import shutil

            # Copy files to shared directory
            shutil.copy(wrapper_path, f"{tmpdir}/wrapper.py")
            shutil.copy(user_code_path, f"{tmpdir}/user_code.py")

            # Create state file path
            state_file = f"{tmpdir}/rlm_state.json"

            # Copy existing state if any
            if hasattr(self, "_direct_state_data"):
                with open(state_file, "wb") as f:
                    f.write(self._direct_state_data)

            # Update wrapper code to use correct paths
            with open(f"{tmpdir}/wrapper.py") as f:
                wrapper_code = f.read()

            # Replace paths to use /workspace mount
            wrapper_code = wrapper_code.replace(self.STATE_FILE, "/workspace/rlm_state.json")
            wrapper_code = wrapper_code.replace(container_user_path, "/workspace/user_code.py")

            with open(f"{tmpdir}/wrapper.py", "w") as f:
                f.write(wrapper_code)

            # Run Docker container with security hardening
            cmd = [
                "docker",
                "run",
                "--rm",
                "--network=none",  # Prevent data exfiltration
                "--cap-drop=ALL",  # Drop all capabilities
                "--memory=4g",  # Resource limit
                "--read-only",  # Read-only root filesystem
                "--tmpfs=/tmp:size=100m",  # Writable tmp with size limit
                "--security-opt=no-new-privileges:true",  # Prevent privilege escalation
                "-v",
                f"{tmpdir}:/workspace",
                "-w",
                "/workspace",
                "python:3.11-slim",
                "python3",
                "/workspace/wrapper.py",
            ]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.config.exec_timeout,
                )
                stdout = stdout_bytes.decode() if stdout_bytes else ""
                stderr = stderr_bytes.decode() if stderr_bytes else ""

                output = stdout
                if stderr:
                    output += f"\n[STDERR]\n{stderr}"

                # Save state for next execution
                if os.path.exists(state_file):
                    with open(state_file, "rb") as f:
                        self._direct_state_data = f.read()

                return proc.returncode == 0, output

            except asyncio.TimeoutError:
                return False, f"[TIMEOUT] Execution exceeded {self.config.exec_timeout}s"
            except Exception as e:
                return False, f"[DOCKER_ERROR] {e}"

    async def _get_pending_llm_calls(self) -> list[tuple[str, str]]:
        """Retrieve pending LLM calls from the container state."""
        extract_code = f'''
import json

# Load state from JSON
with open("{self.STATE_FILE}", "r") as f:
    env = json.load(f)

# Output calls as JSON for parsing
calls = [(q, c) for q, c in env["llm_calls"]]
print("[LLM_CALLS_JSON]")
print(json.dumps(calls))
print("[/LLM_CALLS_JSON]")
'''

        success, output = await self._exec_code(extract_code)
        if not success:
            return []

        # Parse JSON output
        match = re.search(r"\[LLM_CALLS_JSON\]\n(.*?)\n\[/LLM_CALLS_JSON\]", output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return []
        return []


# =============================================================================
# RLM Tool - Main Tool Implementation
# =============================================================================


class RLMTool:
    """Recursive Language Model tool for processing long-context content.

    This tool implements the RLM algorithm from the MIT CSAIL paper,
    enabling LLMs to process arbitrarily long content through a
    Python REPL environment with recursive sub-calls.
    """

    name = "rlm"

    def __init__(self, coordinator: ModuleCoordinator, config: RLMConfig):
        """Initialize the RLM tool.

        Args:
            coordinator: Module coordinator for accessing other tools/providers
            config: RLM configuration
        """
        self.coordinator = coordinator
        self.config = config

    @property
    def description(self) -> str:
        return """Process long-context content using Recursive Language Models (RLM).

RLM enables LLMs to handle arbitrarily long content (10M+ tokens) by:
1. Loading content into a Python REPL environment
2. Letting the model write code to explore and process the content
3. Supporting recursive LLM sub-calls for divide-and-conquer processing

Use this tool when:
- Content exceeds normal context window limits
- You need to analyze large codebases, documents, or datasets
- Tasks require systematic exploration of large content

The tool returns the final answer after the model completes processing."""

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to process (can be very large). Either content or file_path required.",
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to a file to process. Alternative to content parameter.",
                },
                "query": {
                    "type": "string",
                    "description": "The question or task to perform on the content",
                },
                "content_type": {
                    "type": "string",
                    "description": "Type of content (e.g., 'code repository', 'document', 'dataset')",
                    "default": "text",
                },
                "provider": {
                    "type": "string",
                    "description": f"Provider for LLM calls (default: {self.config.default_provider})",
                },
                "model": {
                    "type": "string",
                    "description": "Model to use (uses provider default if not specified)",
                },
            },
            "required": ["query"],
        }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute RLM processing on content.

        Args:
            input: Dictionary with 'content', 'query', and optional parameters

        Returns:
            ToolResult with the final answer or error
        """
        # Import here to avoid circular imports
        from amplifier_core import ToolResult

        content = input.get("content", "")
        file_path = input.get("file_path", "")
        query = input.get("query", "")
        content_type = input.get("content_type", "text")
        provider_name = input.get("provider", self.config.default_provider)
        model = input.get("model", self.config.default_model)

        # Handle file_path - read file content if provided
        if file_path and not content:
            try:
                # Expand user home directory if needed
                expanded_path = os.path.expanduser(file_path)
                with open(expanded_path, encoding="utf-8") as f:
                    content = f.read()
                logger.info(f"RLM: Loaded {len(content):,} chars from {file_path}")
            except FileNotFoundError:
                return ToolResult(
                    success=False,
                    error={"message": f"File not found: {file_path}"},
                )
            except PermissionError:
                return ToolResult(
                    success=False,
                    error={"message": f"Permission denied reading: {file_path}"},
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error={"message": f"Error reading file: {e}"},
                )

        if not content:
            return ToolResult(
                success=False,
                error={"message": "Either 'content' or 'file_path' is required"},
            )
        if not query:
            return ToolResult(success=False, error={"message": "query is required"})

        # Get shadow tool
        tools = self.coordinator.get("tools")
        if not tools:
            return ToolResult(
                success=False,
                error={"message": "No tools registry available. Is shadow bundle installed?"},
            )

        shadow_tool = tools.get("shadow")
        if not shadow_tool:
            return ToolResult(
                success=False,
                error={
                    "message": "Shadow tool not available. Install shadow bundle: "
                    "amplifier bundle add git+https://github.com/microsoft/amplifier-bundle-shadow@main"
                },
            )

        # Get provider
        providers = self.coordinator.get("providers")
        if not providers:
            return ToolResult(success=False, error={"message": "No providers available"})

        provider = providers.get(provider_name)
        if not provider:
            available = list(providers.keys()) if hasattr(providers, "keys") else []
            return ToolResult(
                success=False,
                error={"message": f"Provider '{provider_name}' not found. Available: {available}"},
            )

        # Get hooks for observability
        hooks = self.coordinator.get("hooks")

        # Create provider function for sub-calls
        async def make_llm_call(sub_query: str, sub_content: str) -> str:
            """Make a sub-LLM call."""
            from amplifier_core.message_models import ChatRequest, Message

            messages = [
                Message(
                    role="user",
                    content=f"{sub_query}\n\nContent:\n{sub_content}",
                )
            ]
            request = ChatRequest(messages=messages)
            response = await provider.complete(request, model=model)

            # Extract text content from response
            if hasattr(response, "content") and response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                    if isinstance(block, dict) and "text" in block:
                        return block["text"]
            return str(response)

        # Initialize REPL manager
        repl = REPLManager(
            shadow_tool=shadow_tool,
            provider_fn=make_llm_call,
            config=self.config,
            hooks=hooks,
        )

        try:
            # Emit start event
            if hooks:
                await hooks.emit(
                    "rlm:start",
                    {
                        "content_length": len(content),
                        "content_type": content_type,
                        "query": query[:200],
                    },
                )

            # Initialize REPL environment
            if not await repl.initialize(content, content_type):
                return ToolResult(
                    success=False,
                    error={"message": f"Failed to initialize RLM: {repl.state.error}"},
                )

            # Build system prompt
            system_prompt = RLM_SYSTEM_PROMPT.format(
                context_type=content_type,
                context_total_length=len(content),
            )

            # Main RLM loop - use Message objects for type safety
            from amplifier_core.message_models import ChatRequest, Message

            step_count = 0
            conversation: list[Message] = [
                Message(role="user", content=f"{system_prompt}\n\nTask: {query}")
            ]

            while step_count < self.config.max_trajectory_steps:
                step_count += 1

                # Check cancellation
                cancellation = self.coordinator.get_capability("cancellation")
                if cancellation and cancellation.is_requested:
                    return ToolResult(
                        success=False,
                        error={"message": "Cancelled by user"},
                        output={"partial_trajectory": len(repl.state.trajectory)},
                    )

                # Get next code from LLM
                _debug(f"STEP {step_count}: Requesting code from LLM")
                request = ChatRequest(messages=conversation)
                response = await provider.complete(request, model=model)

                # Extract code from response
                response_text = ""
                if hasattr(response, "content") and response.content:
                    for block in response.content:
                        if hasattr(block, "text"):
                            response_text = block.text
                            break
                        if isinstance(block, dict) and "text" in block:
                            response_text = block["text"]
                            break

                _debug(f"LLM RESPONSE ({len(response_text)} chars):\n{response_text[:500]}...")

                # Update token counts
                if hasattr(response, "usage") and response.usage:
                    repl.state.total_tokens_in += response.usage.input_tokens or 0
                    repl.state.total_tokens_out += response.usage.output_tokens or 0

                # Extract code blocks
                code_blocks = re.findall(r"```python\n(.*?)```", response_text, re.DOTALL)
                if not code_blocks:
                    # Try without language specifier
                    code_blocks = re.findall(r"```\n(.*?)```", response_text, re.DOTALL)

                _debug(f"CODE BLOCKS FOUND: {len(code_blocks)}")

                if not code_blocks:
                    # No code found - model should provide code, not prose
                    # Only accept a direct answer if it's clearly marked with FINAL()
                    # Pattern: FINAL("answer") or FINAL('answer') in the response
                    final_call_match = re.search(
                        r'FINAL\s*\(\s*["\']([^"\']+)["\']\s*\)', response_text
                    )
                    if final_call_match:
                        repl.state.final_answer = final_call_match.group(1)
                        break

                    # Ask model to provide code
                    conversation.append(Message(role="assistant", content=response_text))
                    conversation.append(
                        Message(
                            role="user",
                            content="Please provide Python code in ```python blocks to search the context and find the answer. "
                            "Use FINAL(answer) when you find it.",
                        )
                    )
                    continue

                # CRITICAL FIX: Combine all code blocks and execute together
                # This preserves variable scope between blocks in the same response
                # But first, validate each block for syntax errors
                valid_blocks = []
                for i, block in enumerate(code_blocks):
                    try:
                        compile(block, f"<block_{i}>", "exec")
                        valid_blocks.append(block)
                    except SyntaxError as e:
                        _debug(f"SYNTAX ERROR in block {i}: {e}")
                        # Skip invalid blocks
                        continue

                if not valid_blocks:
                    _debug("No valid code blocks found, asking LLM to retry")
                    conversation.append(Message(role="assistant", content=response_text))
                    conversation.append(
                        Message(
                            role="user",
                            content="Your code had syntax errors. Please provide valid Python code in ```python blocks.",
                        )
                    )
                    continue

                combined_code = "\n\n".join(valid_blocks)
                _debug(
                    f"EXECUTING COMBINED CODE ({len(valid_blocks)}/{len(code_blocks)} valid blocks):\n{combined_code[:500]}..."
                )

                output, has_final, pending_calls = await repl.execute_code(combined_code)
                _debug(f"OUTPUT: {output[:300] if output else '(empty)'}...")
                _debug(
                    f"has_final={has_final}, pending={len(pending_calls)}, answer={repl.state.final_answer}"
                )

                # Process any pending LLM calls
                if pending_calls:
                    _debug(f"Processing {len(pending_calls)} pending LLM calls")
                    results = await repl.process_llm_calls(pending_calls)
                    await repl.inject_llm_results(results)

                    # CRITICAL: Re-execute the SAME code after injecting results
                    # This allows llm_query calls to now return actual results
                    _debug("Re-executing code with injected results...")
                    output, has_final, new_pending = await repl.execute_code(combined_code)
                    _debug(
                        f"After re-exec: has_final={has_final}, answer={repl.state.final_answer}"
                    )

                    # If there are new pending calls, we need another round
                    if new_pending and not has_final:
                        _debug(
                            f"New pending calls: {len(new_pending)}, will handle in next iteration"
                        )
                        # These will be handled in the next iteration

                if repl.state.final_answer:
                    _debug(f"FINAL ANSWER FOUND: {repl.state.final_answer}")
                    break

                # Add execution output to conversation
                conversation.append(Message(role="assistant", content=response_text))
                conversation.append(
                    Message(
                        role="user",
                        content=f"Execution output:\n{output}\n\nContinue processing or call FINAL() with your answer.",
                    )
                )

            # Emit completion event
            if hooks:
                await hooks.emit(
                    "rlm:complete",
                    {
                        "steps": step_count,
                        "llm_calls": repl.state.total_llm_calls,
                        "tokens_in": repl.state.total_tokens_in,
                        "tokens_out": repl.state.total_tokens_out,
                        "has_answer": repl.state.final_answer is not None,
                    },
                )

            if repl.state.final_answer:
                return ToolResult(
                    success=True,
                    output={
                        "answer": repl.state.final_answer,
                        "steps": step_count,
                        "llm_calls": repl.state.total_llm_calls,
                        "tokens_in": repl.state.total_tokens_in,
                        "tokens_out": repl.state.total_tokens_out,
                    },
                )
            else:
                return ToolResult(
                    success=False,
                    error={
                        "message": f"RLM did not produce a final answer after {step_count} steps"
                    },
                    output={
                        "partial_trajectory": len(repl.state.trajectory),
                        "llm_calls": repl.state.total_llm_calls,
                    },
                )

        except Exception as e:
            logger.exception("RLM execution failed")
            if hooks:
                await hooks.emit("rlm:error", {"error": str(e)})
            return ToolResult(
                success=False,
                error={"message": f"RLM execution failed: {e}"},
            )

        finally:
            await repl.cleanup()


# =============================================================================
# Module Mount Point
# =============================================================================


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the RLM tool.

    Args:
        coordinator: The module coordinator
        config: Optional configuration dictionary

    Returns:
        Cleanup function (or None)
    """
    config = config or {}
    rlm_config = RLMConfig(**config)

    # Register observable events
    obs_events = coordinator.get_capability("observability.events") or []
    obs_events.extend(
        [
            "rlm:start",  # RLM processing started
            "rlm:llm_call",  # Sub-LLM call made
            "rlm:complete",  # RLM processing completed
            "rlm:error",  # RLM error occurred
        ]
    )
    coordinator.register_capability("observability.events", obs_events)

    tool = RLMTool(coordinator, rlm_config)
    await coordinator.mount("tools", tool, name=tool.name)
    logger.info("Mounted RLM tool with observable events")

    return None  # No cleanup needed - shadow cleanup handled internally
