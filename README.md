# LLM Prompt Semantic Diff

A CLI tool for managing and comparing LLM prompts using semantic diffing instead of traditional text-based comparison.

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

## Overview

LLM Prompt Semantic Diff delivers a lightweight command‑line workflow for managing, packaging, and *semantic* diffing of Large Language Model prompts. It addresses the blind spot where ordinary text‑based `git diff` fails to reveal meaning‑level changes that materially affect model behaviour.

Read More: https://medium.com/@aatakansalar/catching-prompt-regressions-before-they-ship-semantic-diffing-for-llm-workflows-feb3014ccac3

## Key Features

- **F-1**: `prompt init` - Generates skeletal prompt files and default manifests
- **F-2**: `prompt pack` - Embeds prompts into `.pp.json` with semantic versioning
- **F-3**: `prompt diff` - Semantic comparison with percentage scores and exit codes
- **F-4**: Dual embedding providers (OpenAI cloud + SentenceTransformers local)
- **F-5**: JSON output for CI/CD integration
- **F-6**: Schema validation for all manifests
- **F-7**: Comprehensive test suite with >75% coverage

## Installation

Install from source:

```bash
git clone https://github.com/aatakansalar/llm-prompt-semantic-diff
cd llm-prompt-semantic-diff
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Quick Start

### 1. Initialize a New Prompt

```bash
prompt init my-greeting
```

Creates `my-greeting.prompt` and `my-greeting.pp.json` with default structure.

### 2. Package an Existing Prompt

```bash
prompt pack my-prompt.prompt
```

Generates embeddings and creates a versioned manifest.

### 3. Compare Prompt Versions

```bash
# Human-readable output
prompt diff v1.pp.json v2.pp.json --threshold 0.8

# JSON output for CI/CD
prompt diff v1.pp.json v2.pp.json --json --threshold 0.8
```

Returns exit code 1 if similarity below threshold.

### 4. Validate Manifest Schema

```bash
prompt validate my-prompt.pp.json
```

## Embedding Providers

### Local (Default)
Uses SentenceTransformers with `all-MiniLM-L6-v2` model:
```bash
prompt pack my-prompt.prompt --provider sentence-transformers
```

### Cloud (OpenAI)
Requires `OPENAI_API_KEY` environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
prompt pack my-prompt.prompt --provider openai
```

## CI/CD Integration

Use `--json` flag for machine-readable output:

```yaml
- name: Check prompt changes
  run: |
    prompt diff main.pp.json feature.pp.json --json --threshold 0.8
    if [ $? -eq 1 ]; then
      echo "Prompt changes exceed threshold - review required"
      exit 1
    fi
```

## Manifest Format

Prompts are packaged into `.pp.json` files:

```json
{
  "content": "Your prompt text here...",
  "version": "0.1.0",
  "embeddings": [0.1, -0.2, 0.3, ...],
  "description": "Optional description",
  "tags": ["category", "type"],
  "model": "gpt-4"
}
```

## Example Workflow

```bash
# Create new prompt
prompt init greeting

# Edit greeting.prompt file
# ... make changes ...

# Package with embeddings
prompt pack greeting.prompt

# Create modified version
cp greeting.prompt greeting-v2.prompt
# ... make more changes ...
prompt pack greeting-v2.prompt

# Compare versions
prompt diff greeting.pp.json greeting-v2.pp.json

# Output:
# Semantic similarity: 85.2%
# Threshold: 80.0%
# Above threshold: Yes
# Version A: 0.1.0
# Version B: 0.1.0
```

## Security & Privacy

- **Local-first**: No data leaves your machine unless OpenAI provider is explicitly selected
- **API keys**: Only read from environment variables (`OPENAI_API_KEY`)
- **No telemetry**: No analytics, tracking, or hidden network calls

## Development

```bash
git clone https://github.com/aatakansalar/llm-prompt-semantic-diff
cd llm-prompt-semantic-diff
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details. 
