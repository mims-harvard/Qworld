# Qworld

Generate a set of unique evaluation criteria from one question or a list of questions using LLMs. The pipeline expands scenarios, perspectives, and criteria through multiple iterations, then deduplicates and assigns scores.

## Installation

**From PyPI**:

```bash
pip install qworld
```

**From PyPI with local embedding fallback (optional extra):**

```bash
pip install "qworld[local-embeddings]"
```

**From GitHub**:

```bash
pip install "qworld @ git+https://github.com/mims-harvard/Qworld.git"
```

**From local clone (for development):**

```bash
git clone https://github.com/mims-harvard/Qworld.git
cd Qworld
pip install -e .
```

### Embedding Dependency Behavior

- If `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY` is available, qworld uses OpenAI/Azure embeddings.
- If `GOOGLE_API_KEY` is available, qworld uses Gemini embeddings.
- If non of these keys are available, qworld falls back to local `sentence-transformers` embeddings.
- To enable that fallback, install the optional extra: `pip install "qworld[local-embeddings]"`.

## API Keys (.env)

Create a `.env` file in the project root with the keys for your chosen provider:

```bash
# OpenAI (default)
OPENAI_API_KEY=sk-...

# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com

# Claude
ANTHROPIC_API_KEY=sk-ant-...

# Gemini
GOOGLE_API_KEY=...   # or GEMINI_API_KEY

# Grok
XAI_API_KEY=...

# DeepSeek
DEEPSEEK_API_KEY=...

# vLLM
VLLM_SERVER_URL=http://localhost:8000/v1
```

Load API keys before running or use `python-dotenv` to load `.env` in your script.

## Quick Start

```python
from qworld import CriteriaGenerator

gen = CriteriaGenerator(model="gpt-4.1")

# Single question (string)
result = gen.generate("What is machine learning?")
print(result["final_criteria"])

# Single question (dict)
result = gen.generate({"id": "q1", "question": "What is AI?"})

# Batch
results = gen.generate([
    {"id": "q1", "question": "What is AI?"},
    {"id": "q2", "question": "How does deep learning work?"},
])
```

### Input Arguments

#### 1. CriteriaGenerator Init Parameters

| Parameter                 | Type  | Default                      | Description                                         |
| ------------------------- | ----- | ---------------------------- | --------------------------------------------------- |
| `model`                 | str   | `"gpt-4.1"`                | Model name (auto-detects provider)                  |
| `base_url`              | str   | None                         | API base URL (required for vLLM)                    |
| `api_key`               | str   | None                         | API key (uses env vars if not provided)             |
| `temperature`           | float | 0.4                          | Generation temperature                              |
| `embedding_model`       | str   | `"text-embedding-3-small"` | Model for embeddings (deduplication)                |
| `n_scenario_expands`    | int   | 3                            | Scenario expansion iterations                       |
| `n_perspective_expands` | int   | 4                            | Perspective expansion iterations                    |
| `n_criteria_expands`    | int   | 3                            | Criteria expansion iterations                       |
| `dedup_threshold`       | float | 0.7                          | Cosine similarity threshold for deduplication (0-1) |
| `max_workers`           | int   | 8                            | Parallel workers for batch processing               |
| `max_retries`           | int   | 5                            | Max retries on rate limit errors                    |
| `debug`                 | bool  | False                        | Print raw LLM outputs for debugging                 |

#### 2. generate() Input Arguments

| Input Type | Format                               | Required                           | Optional                            |
| ---------- | ------------------------------------ | ---------------------------------- | ----------------------------------- |
| `str`    | `"question text"`                  | —                                 | —                                  |
| `dict`   | `{"id": "...", "question": "..."}` | `id`, `question`               | `image`, `web_content`          |
| `list`   | `[{...}, {...}]`                   | each item has `id`, `question` | `image`, `web_content` per item |

- **`id`**: Identifier for the question (auto-assigned if omitted).
- **`question`**: The question text.
- **`image`**: Base64 string (with or without `data:image/...;base64,` prefix) for vision-capable models.
- **`web_content`**: Retrieved web context; appended to the question as `[Retrieved Web Context] ... [End of Web Context]`.

### Output Format

```python
{
    "id": "q1",
    "question": "What is AI?",
    "scenarios": [...],
    "raw_perspectives": [...],
    "reviewed_perspectives": [...],
    "raw_criteria": [...],
    "reviewed_criteria": [...],
    "final_criteria": [
        {"criterion": "Explains core concepts", "points": 3},
        {"criterion": "Provides examples", "points": 2},
    ],
}
```

## Supported Models

| Provider     | Models                                  | Env Variable                                        |
| ------------ | --------------------------------------- | --------------------------------------------------- |
| OpenAI       | gpt-5, gpt-4.1, ...                     | `OPENAI_API_KEY`                                  |
| Azure OpenAI | gpt-5, gpt-4.1, ...                     | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| Claude       | claude-4-5-opus, claude-4-5-sonnet, ... | `ANTHROPIC_API_KEY`                               |
| Gemini       | gemini-3-pro, gemini-3-flash, ...       | `GOOGLE_API_KEY`                                  |
| Grok         | grok-4.1-fast, ...                      | `XAI_API_KEY`                                     |
| DeepSeek     | deepseek-chat, deepseek-reasoner        | `DEEPSEEK_API_KEY`                                |
| vLLM         | Qwen3-30B, ...                          | `VLLM_SERVER_URL` or `base_url` param           |

**vLLM**: Uses a separate server for hosting. You need to install the vLLM package suitable for your server environment or use the official vLLM Docker image. Once the server is running, provide the model name and the server URL (e.g. `VLLM_SERVER_URL=http://localhost:8000/v1` or `base_url` when initializing `CriteriaGenerator`).

## Data

Raw data and generated criteria (gpt-4.1 with scenario expand x3, perspective expand x4, criteria expand x3) are available at [https://huggingface.co/datasets/suyc21/qworld](https://huggingface.co/datasets/suyc21/qworld).
