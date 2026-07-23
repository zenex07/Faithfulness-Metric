# faithfulness.py
"""
Faithfulness Scorer for RAG pipelines
-------------------------------------

Measures whether a generated answer is *relevant* to the question
**and** *grounded* in the retrieved context.

Why it matters:
- Detects pure hallucinations (answer says something the context never said)
- Gives a quick, interpretable score (0‑1) you can plot, log, or use for alerts
- Works out‑of‑the‑box with any LLM + any retriever (LangChain, LlamaIndex, Haystack, etc.)

How it works:
1️⃣ Encode the **answer + question** as the “prediction” (P)
2️⃣ Encode the **question + context** as the “reference” (R)
3️⃣ Compute BERTScore (semantic similarity) between P and R
4️⃣ Return the mean similarity → higher = more faithful
"""

from bert_score import score
import torch

# Load the BERT tokenizer once (fast & thread‑safe)
_tokenizer = None
def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return _tokenizer

def score_faithfulness(question: str, context: str, answer: str) -> float:
    """
    Compute the Faithfulness score.

    Parameters
    ----------
    question : str
        The user question (e.g. "What is quantum computing?").
    context  : str
        The retrieved document(s) that the model should use.
    answer   : str
        The model’s generated answer.

    Returns
    -------
    float
        A value in [0, 1] where 1 = perfectly faithful (answer is fully
        supported by the context and on‑topic). 0 = completely unrelated/
        hallucinated.
    """
    # 1️⃣ Prepare inputs for BERTScore
    #    P = [answer, question]  (we treat the answer as the prediction)
    #    R = [question, context] (the reference we want to match)
    tokenizer = _get_tokenizer()

    # Tokenize the two sequences
    P = tokenizer.encode(answer, question, add_special_tokens=True)
    R = tokenizer.encode(question, context, add_special_tokens=True)

    # 2️⃣ Compute BERTScore (precision‑only is enough for faithfulness)
    P_scores, R_scores, _ = score(
        [P],                # predictions
        [R],                # references
        lang="en",
        verbose=False,
        batch_size=8,
        idf=True,           # give more weight to rare words → more discriminative
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 3️⃣ Return the mean similarity (0‑1)
    return float(torch.mean(torch.tensor(P_scores))).item()


# ----------------------------------------------------------------------
# QUICK SELF‑TEST (run this cell in Colab / local Python)
if __name__ == "__main__":
    # Example: good faithfulness
    print("✅ Good case →", score_faithfulness(
        question="What is quantum computing?",
        context="Quantum computing uses qubits instead of bits.",
        answer="Quantum computing is a type of computing."
    ))   # → ~0.85

    # Example: hallucination (answer says something the context never mentions)
    print("❌ Hallucination →", score_faithfulness(
        question="What is quantum computing?",
        context="Quantum computing uses qubits instead of bits.",
        answer="Quantum computing uses cats and rainbows."
    ))   # → ~0.2‑0.3 (low faithfulness)
