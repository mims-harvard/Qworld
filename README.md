<div align="center">

<h1>🌍 Qworld: Question-Specific Evaluation Criteria for LLMs</h1>

**Generate rich, multi-dimensional evaluation criteria from any question.**

[![Paper](https://img.shields.io/badge/Paper-arXiv%202603.23522-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2603.23522)
[![Website](https://img.shields.io/badge/Website-qworld-blue?style=for-the-badge&logo=googlechrome&logoColor=white)](https://qworld.openscientist.ai/)
[![Video](https://img.shields.io/badge/Video-Explainer-red?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/ie3IWz49b9U)

[![Demo](https://img.shields.io/badge/Demo-Try%20Qworld-orange?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/suyc21/qworld-demo)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow?style=for-the-badge)](https://huggingface.co/datasets/suyc21/qworld)
[![PyPI](https://img.shields.io/pypi/v/qworld?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/qworld/)


[![Qworld Explainer Video](https://img.youtube.com/vi/ie3IWz49b9U/maxresdefault.jpg)](https://youtu.be/ie3IWz49b9U)

</div>

## About Qworld

Evaluating large language models (LLMs) on open-ended questions is difficult because response quality depends on the question's context. Binary scores and static rubrics fail to capture these context-dependent requirements. Existing methods define criteria at the dataset level or generate them in a single pass, which limits their ability to explore the evaluation space implied by each question.

We introduce **One-Question-One-World (Qworld)**, a method that generates question-specific evaluation criteria using a recursive expansion tree. Given a question, Qworld decomposes it into scenarios, perspectives, and fine-grained binary criteria through structured hierarchical and horizontal expansion. The resulting criteria specify what a high-quality answer must address for that question.

On HealthBench, Qworld covers 89% of expert-authored criteria and generates 79% novel criteria validated by human experts. Experts rate Qworld criteria higher in insight and granularity than those produced by prior methods. When applied to 11 frontier LLMs on HealthBench and Humanity's Last Exam, Qworld reveals capability differences in dimensions such as long-term impact, equity, error handling, and interdisciplinary reasoning that coarse rubrics do not distinguish.

By formulating criteria generation as structured coverage of question-implied evaluation axes, Qworld enables evaluation that adapts to each question rather than relying on fixed task-level criteria.

---

## Getting Started

Qworld supports three usage scenarios:

1. **🌐 Try it online** — Want to experience Qworld quickly? Try our interactive demo at [qworld.openscientist.ai](https://qworld.openscientist.ai/#try-it).
2. **🤖 Agentic systems** — Want to integrate Qworld into your agentic workflow? Check out our [skills](https://github.com/mims-harvard/Qworld/tree/main/skills) for plug-and-play usage.
3. **📊 Batch generation** — Need to generate criteria for an entire dataset? Install the package and use the Python API below — pass a list of questions and set `max_workers` to parallelize.

---

## Installation

```bash
# From PyPI
pip install qworld

# With local embedding fallback (optional)
pip install "qworld[local-embeddings]"

# From GitHub
pip install "qworld @ git+https://github.com/mims-harvard/Qworld.git"
```

<details>
<summary><b>Install from local clone (for development)</b></summary>

```bash
git clone https://github.com/mims-harvard/Qworld.git
cd Qworld
pip install -e .
```

</details>

### Embedding Dependency Behavior

| Priority | Condition                                          | Backend Used                                                                          |
| :------: | -------------------------------------------------- | ------------------------------------------------------------------------------------- |
|    1    | `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY` set | OpenAI / Azure embeddings                                                             |
|    2    | `GOOGLE_API_KEY` set                             | Gemini embeddings                                                                     |
|    3    | No API key available                               | Local `sentence-transformers` (requires `pip install "qworld[local-embeddings]"`) |

---

## API Keys

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

---

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

### `CriteriaGenerator` Parameters

| Parameter                 | Type |           Default           | Description                                  |
| ------------------------- | :---: | :--------------------------: | -------------------------------------------- |
| `model`                 |  str  |        `"gpt-4.1"`        | Model name (auto-detects provider)           |
| `base_url`              |  str  |             None             | API base URL (required for vLLM)             |
| `api_key`               |  str  |             None             | API key (uses env vars if not provided)      |
| `temperature`           | float |             0.4             | Generation temperature                       |
| `embedding_model`       |  str  | `"text-embedding-3-small"` | Model for embeddings (deduplication)         |
| `n_scenario_expands`    |  int  |              3              | Scenario expansion iterations                |
| `n_perspective_expands` |  int  |              4              | Perspective expansion iterations             |
| `n_criteria_expands`    |  int  |              3              | Criteria expansion iterations                |
| `dedup_threshold`       | float |             0.7             | Cosine similarity threshold for dedup (0–1) |
| `max_workers`           |  int  |              8              | Parallel workers for batch processing        |
| `max_retries`           |  int  |              5              | Max retries on rate-limit errors             |
| `debug`                 | bool |            False            | Print raw LLM outputs for debugging          |

### `generate()` Input

| Input Type | Format                               | Required                           | Optional                            |
| :--------: | ------------------------------------ | ---------------------------------- | ----------------------------------- |
|  `str`  | `"question text"`                  | —                                 | —                                  |
|  `dict`  | `{"id": "...", "question": "..."}` | `id`, `question`               | `image`, `web_content`          |
|  `list`  | `[{...}, {...}]`                   | each item has `id`, `question` | `image`, `web_content` per item |

- **`image`** — Base64 string (with or without `data:image/...;base64,` prefix) for vision-capable models.
- **`web_content`** — Retrieved web context; appended as `[Retrieved Web Context] ... [End of Web Context]`.

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

---

## Supported Models

| Provider               | Example Models                         | Env Variable                                         |
| ---------------------- | -------------------------------------- | ---------------------------------------------------- |
| **OpenAI**       | gpt-5, gpt-4.1, …                     | `OPENAI_API_KEY`                                   |
| **Azure OpenAI** | gpt-5, gpt-4.1, …                     | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |
| **Claude**       | claude-4-5-opus, claude-4-5-sonnet, … | `ANTHROPIC_API_KEY`                                |
| **Gemini**       | gemini-3-pro, gemini-3-flash, …       | `GOOGLE_API_KEY`                                   |
| **Grok**         | grok-4.1-fast, …                      | `XAI_API_KEY`                                      |
| **DeepSeek**     | deepseek-chat, deepseek-reasoner       | `DEEPSEEK_API_KEY`                                 |
| **vLLM**         | Qwen3-30B, …                          | `VLLM_SERVER_URL` or `base_url` param            |

> **vLLM** requires a separate server. Install the vLLM package for your environment or use the official Docker image. Once the server is running, provide the model name and server URL (e.g. `VLLM_SERVER_URL=http://localhost:8000/v1` or `base_url` in `CriteriaGenerator`).

---

## Data

Raw data and generated criteria (gpt-4.1 with scenario expand ×3, perspective expand ×4, criteria expand ×3) are available on [🤗 HuggingFace](https://huggingface.co/datasets/suyc21/qworld).

### Reproduction Defaults

The released generated criteria use the same expansion depths as the package
defaults:

| Setting | Value |
| --- | --- |
| Criteria generation model | `gpt-4.1` |
| Temperature | `0.4` |
| Scenario expansion rounds | `3` |
| Perspective expansion rounds | `4` |
| Criteria expansion rounds | `3` |
| Embedding model | `text-embedding-3-small` |
| Deduplication threshold | `0.7` |
| Max retries | `5` |

`max_workers` controls local parallelism for batch processing and does not
change the generated criteria for an individual question.

---

## Citation

```bibtex
@misc{gao2026qworldquestionspecificevaluationcriteria,
      title={Qworld: Question-Specific Evaluation Criteria for LLMs},
      author={Shanghua Gao and Yuchang Su and Pengwei Sui and Curtis Ginder and Marinka Zitnik},
      year={2026},
      eprint={2603.23522},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.23522},
}
```
