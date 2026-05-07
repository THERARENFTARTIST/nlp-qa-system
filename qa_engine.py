# ============================================================
# MODULE 2 — qa_engine.py
# Owner: Person 2
# Responsibility: Load the model and run QA inference
# ============================================================

from transformers import pipeline

# We use DistilBERT fine-tuned on SQuAD — lightweight and CPU-friendly
MODEL_NAME = "deepset/roberta-base-squad2"

# Load the pipeline once (cached globally to avoid reloading on every call)
_qa_pipeline = None


def load_model():
    """
    Loads the HuggingFace QA pipeline.
    Called once at startup. Model is cached after first load.
    """
    global _qa_pipeline
    if _qa_pipeline is None:
        print(f"Loading model: {MODEL_NAME} ...")
        _qa_pipeline = pipeline("question-answering", model=MODEL_NAME)
        print("Model loaded successfully.")
    return _qa_pipeline


def get_answer(context: str, question: str) -> dict:
    """
    Runs the QA model on the given context and question.

    Args:
        context  : The passage to search for the answer in.
        question : The question to answer.

    Returns a dict with:
        {
            "answer"  : str   — the extracted answer text,
            "score"   : float — confidence score (0 to 1),
            "start"   : int   — start character index in context,
            "end"     : int   — end character index in context
        }
    """
    qa = load_model()
    result = qa(question=question, context=context)
    return result