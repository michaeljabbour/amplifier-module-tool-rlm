# amplifier-module-tool-rlm

Recursive Language Model (RLM) tool for [Amplifier](https://github.com/microsoft/amplifier) - enables processing of arbitrarily long contexts through recursive decomposition.

## The Problem: Context Window Limits

Modern LLMs have context window limits:

| Model | Max Context | Practical Limit |
|-------|-------------|-----------------|
| Claude Sonnet | ~200K tokens | ~150K usable |
| GPT-4 | ~128K tokens | ~100K usable |
| Gemini | ~1M tokens | ~800K usable |

**What happens when your document exceeds these limits?**
- Truncation loses critical information
- Summarization loses detail
- RAG may miss relevant chunks

**RLM solves this** by letting the LLM recursively process chunks while maintaining the ability to synthesize information across the entire document.

## How RLM Works

RLM treats the input as an **external environment** that the LLM can programmatically explore:

```
┌─────────────────────────────────────────────────────────────┐
│  Your 5M Token Document                                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│  │ Chunk 1 │ │ Chunk 2 │ │ Chunk 3 │ │ Chunk N │ ...        │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘            │
│       │           │           │           │                  │
│       ▼           ▼           ▼           ▼                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Python REPL (sandboxed Docker)                     │    │
│  │  - Search with regex                                │    │
│  │  - Extract relevant sections                        │    │
│  │  - Make recursive LLM calls on chunks               │    │
│  │  - Combine facts to compute answers                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│                    Final Answer                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Mechanism

1. **Context as Variable**: Content loaded into sandboxed Python REPL as `context`
2. **Programmatic Exploration**: LLM writes Python code to search, filter, chunk
3. **Recursive Sub-calls**: LLM can call itself on smaller pieces
4. **Fact Synthesis**: Combine information from multiple chunks into final answer

## When to Use RLM

| Scenario | Use RLM? | Why |
|----------|----------|-----|
| Document > 200K tokens | ✅ Yes | Exceeds model context window |
| Multi-hop reasoning | ✅ Yes | Need to combine facts from different sections |
| Analytical queries | ✅ Yes | "Calculate X from data scattered across doc" |
| Simple lookup in small file | ❌ No | Regular read is faster |
| Real-time streaming | ❌ No | RLM processes in batches |
| Vague exploratory queries | ❌ No | Works best with specific questions |

### Ideal Use Cases

- **Financial Analysis**: "What's the per-engineer investment?" (requires finding budget AND headcount)
- **Legal Document Review**: "Find all clauses related to liability across this 500-page contract"
- **Codebase Understanding**: "How does the authentication flow work?" (across multiple files)
- **Research Synthesis**: "What are the key findings across these 50 papers?"

## Validated Results

We tested RLM with multi-hop analytical queries requiring fact synthesis:

### Test Setup
- **Query**: "What is the per-engineer quarterly investment in Project Titan?"
- **Answer requires**: Finding budget ($12.3M) AND team size (191) from different sections
- **Expected**: $64,400 per engineer per quarter

### Results

| File Size | Tokens (approx) | RLM Result | Status |
|-----------|-----------------|------------|--------|
| 256KB | ~64K | $64,400 ✅ | Correct |
| 1MB | ~250K | $64,400 ✅ | Correct |
| 5MB | ~1.25M | $64,400 ✅ | Correct |

**Key Finding**: RLM correctly performs multi-hop reasoning even when the answer requires combining information from different document sections.

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
- **LLM Provider**: Anthropic, OpenAI, or other configured provider

## Usage

### Basic Usage

In an Amplifier session:

```
Use the rlm tool with file_path="/path/to/large/document.txt" and query="What is the total revenue for Q3?"
```

### With Inline Content

```
Use the rlm tool with content="<paste your content here>" and query="Summarize the key findings"
```

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `query` | Yes | - | Your question or task (be specific!) |
| `file_path` | One of | - | Path to file to process |
| `content` | these | - | Inline content to process |
| `content_type` | No | "text" | Hint: "code", "document", or "data" |

### Output

```python
{
    "answer": "The per-engineer quarterly investment is $64,400",
    "trajectory_steps": 15,    # REPL execution steps
    "llm_calls": 3,            # Total LLM invocations
    "tokens_in": 125000,       # Input tokens consumed
    "tokens_out": 2500,        # Output tokens generated
    "trajectory": [...]        # Full execution history for debugging
}
```

## Best Practices

### Write Specific Queries

```
# Good - specific and answerable
"What is the per-engineer quarterly investment in Project Titan?"
"Find all references to 'liability' in sections 4-7"
"What are the three main risk factors mentioned?"

# Bad - too vague
"Tell me about this document"
"What's interesting here?"
```

### Provide Context Type Hints

```
# For code files
content_type="code"

# For business documents
content_type="document"

# For structured data (JSON, CSV)
content_type="data"
```

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
# Run unit tests (51 tests)
uv run pytest tests/ -v --ignore=tests/test_integration.py

# Run with coverage
uv run pytest tests/ --cov=amplifier_module_tool_rlm

# Type checking
uv run pyright

# Lint
uv run ruff check .
```

## Architecture

```
amplifier-module-tool-rlm/
├── amplifier_module_tool_rlm/
│   └── __init__.py          # RLMTool, REPLManager, mount()
├── tests/
│   ├── test_tool.py         # Tool interface tests
│   ├── test_repl_manager.py # REPL execution tests
│   ├── test_config.py       # Configuration tests
│   └── test_models.py       # Data model tests
├── pyproject.toml           # Entry point: tool-rlm
├── README.md
└── LICENSE
```

## References

- **RLM Paper**: Zhang, Kraska, Khattab. "Recursive Language Models." MIT CSAIL, Dec 2025.
  - arXiv: https://arxiv.org/html/2512.24601v1
  - Key findings: 28-58% accuracy improvement on information-dense tasks, handles 10M+ tokens
- **Amplifier**: https://github.com/microsoft/amplifier
- **Module Development Guide**: https://github.com/microsoft/amplifier/blob/main/docs/MODULE_DEVELOPMENT.md

## License

MIT License - See LICENSE file for details.
