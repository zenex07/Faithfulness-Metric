from faithfulness import score_faithfulness

# 1️⃣ Get the pieces from your RAG pipeline
question = "What is the capital of France?"
context  = "Paris is the capital city of France."
answer   = "The capital of France is Paris."   # <-- model output

# 2️⃣ Score it
faithful_score = score_faithfulness(question, context, answer)
print(f"Faithfulness: {faithful_score:.2f}")   # e.g. 0.92
