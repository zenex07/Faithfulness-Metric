# RAG‑Scorecard: Faithfulness • Citation • Dialect Fidelity

[![PyPI version](https://img.shields.io/pypi/v/rag-scorecard.svg)](https://pypi.org/project/rag-scorecard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/your‑username/rag-scorecard?style=social)](https://github.com/your-username/rag-scorecard/stargazers)
[![Last Updated](https://img.shields.io/github/last-modified/github/your-username/rag-scorecard)](https://github.com/your-username/rag-scorecard)

> **A tiny, standalone toolkit that scores any Retrieval‑Augmented Generation (RAG) pipeline** for **faithfulness**, **citation accuracy**, and **dialect fidelity**.  
> Turn your research prototype into a reusable, community‑wide utility – perfect for portfolios, internships, or a startup side‑project.

---

## 🎯 Why This Exists

- **RAG developers** spend hours debugging “why is my model hallucinating?”  
- Existing evaluation suites are **research‑centric** (paper‑only) and **hard to reuse**.  
- **This project** extracts the evaluation protocol from the **IJIRT 2023** paper, packages it as a **pip‑installable library**, and makes it **drop‑in** for any pipeline.

> **Result:** You (or anyone) can point the tool at a LangChain, LlamaIndex, or plain‑Python RAG system and instantly get a **3‑metric report** that tells you *exactly* where it fails.

---

## 🛠️ Features

| Metric | What it measures | How it’s computed |
|--------|------------------|-------------------|
| **Faithfulness** | Is the answer *relevant* and *grounded* in the retrieved context? | BERTScore (semantic similarity) between `(answer, question)` and `(question, context)` |
| **Citation Accuracy** | Does the model quote the retrieved docs verbatim? | Token‑level alignment (Levenshtein distance) between model n‑grams and doc n‑grams |
| **Dialect Fidelity** | Does the output use a consistent dialect (e.g., US vs. UK English)? | Regex‑based lexical consistency check (e.g., “colour” vs “color”) |

---

## 🚀 Quick‑Start (5‑minute install)

### 1️⃣ Install (Colab / local)

```bash
# In a notebook or terminal:
pip install rag-scorecard   # <-- once you publish to PyPI

2️⃣Minimal usage example
Python

Copy
from rag_scorecard.faithfulness import score_faithfulness

question = "What is quantum computing?"
context  = "Quantum computing uses qubits instead of bits."
answer   = "Quantum computing is a type of computing."

print(score_faithfulness(question, context, answer))  # → 0.85 (high faithfulness)
3️⃣ Run the full CLI (once you have all three metrics)
Bash

Copy
rag-scorecard --pipeline ./my_rag_pipeline.py \
    --question "What is quantum computing?" \
    --context ./retrieved_context.txt \
    --answer ./model_output.txt
Output (JSON):

JSON

Copy
{
  "faithfulness_score": 0.87,
  "citation_accuracy": 0.62,
  "dialect_fidelity": 0.95,
  "overall_score": 0.81
}
📂 Repository Structure
Bash

Copy
rag-scorecard/
├── README.md                # you are reading it!
├── requirements.txt         # pin dependencies (bert-score, rich, etc.)
├── faithfulness.py          # core metric (see code above)
├── citation.py              # token‑level citation checker
├── dialect.py               # dialect consistency (US/UK/Indian English)
├── main.py                  # CLI entrypoint (argparse)
├── examples/                # HotpotQA / LangChain demos
└── tests/                   # pytest suite (run `pytest`)

# OR, for a quick test without publishing:
pip install bert-score
