# amplifier-module-tool-rlm

Recursive Language Model (RLM) tool for [Amplifier](https://github.com/microsoft/amplifier) - enables processing of arbitrarily long contexts through recursive decomposition.

## Overview

This module implements RLM (Recursive Language Models) as described in the MIT paper by Zhang, Kraska, and Khattab (2025). RLM is an inference-time strategy that allows LLMs to handle arbitrarily long prompts without truncation or summarization.

### Key Features

- **Unbounded Context**: Process documents of any size through recursive chunking
- **REPL-Based Processing**: Sandboxed Docker environment for programmatic context manipulation
- **Recursive Sub-calls**: LLM spawns sub-calls on smaller chunks automatically
- **Multi-hop Reasoning**: Combines facts from different sections to compute answers

## Installation

### Via Bundle Configuration

Add to your bundle's `behaviors/` YAML:

```yaml
tools:
  - module: tool-rlm
    source: git+https://github.com/michaeljabbour/amplifier-module-tool-rlm@main
```

### For Development

```bash
git clone https://github.com/michaeljabbour/amplifier-module-tool-rlm
cd amplifier-module-tool-rlm
uv sync
```

## Requirements

- **Docker**: Required for sandboxed Python REPL execution
- **Amplifier**: This is a tool module for the Amplifier AI agent framework
- **Python 3.11+**

## Usage

In an Amplifier session:

```
Use the rlm tool with file_path="/path/to/large/file.txt" and query="Your question here"
```

Or with inline content:

```
Use the rlm tool with content="<your content>" and query="Your question"
```

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `query` | Yes | - | Your question or task |
| `file_path` | One of | - | Path to file to process |
| `content` | these | - | Inline content to process |
| `content_type` | No | "text" | Hint: "code", "document", or "data" |

### Output

```python
{
    "answer": "The answer to your query",
    "trajectory_steps": 15,
    "llm_calls": 3,
    "tokens_in": 125000,
    "tokens_out": 2500,
    "trajectory": [...]  # Full execution history
}
```

## How It Works

1. **Context Loading**: Content is loaded into a sandboxed Docker container as the `context` variable

2. **Query Analysis**: The LLM determines if this is a lookup or analytical query

3. **Chunked Processing**: For large content, splits into chunks and queries each

4. **Fact Synthesis**: Combines facts from all chunks to compute the answer

## Configuration

```yaml
tools:
  - module: tool-rlm
    config:
      max_recursion_depth: 5      # Max nesting of LLM sub-calls
      max_llm_calls: 100          # Budget for total LLM calls
      max_trajectory_steps: 50    # Max REPL execution steps
      exec_timeout: 60            # Timeout per execution (seconds)
      default_provider: anthropic # Provider for sub-calls
```

## Testing

```bash
# Run unit tests
uv run pytest tests/ -v --ignore=tests/test_integration.py

# Run with coverage
uv run pytest tests/ --cov=amplifier_module_tool_rlm --ignore=tests/test_integration.py

# Type checking
uv run pyright

# Lint
uv run ruff check .
```

## Module Structure

```
amplifier-module-tool-rlm/
├── amplifier_module_tool_rlm/
│   └── __init__.py          # RLMTool, REPLManager, mount()
├── tests/
│   ├── test_tool.py         # Tool tests
│   ├── test_repl_manager.py # REPL manager tests
│   ├── test_config.py       # Config tests
│   └── test_models.py       # Model tests
├── pyproject.toml
├── README.md
└── LICENSE
```

## References

- **RLM Paper**: Zhang, Kraska, Khattab. "Recursive Language Models." MIT CSAIL, Dec 2025.
- **Amplifier**: [github.com/microsoft/amplifier](https://github.com/microsoft/amplifier)
- **Module Development Guide**: [MODULE_DEVELOPMENT.md](https://github.com/microsoft/amplifier/blob/main/docs/MODULE_DEVELOPMENT.md)

## License

MIT License - See LICENSE file for details.
