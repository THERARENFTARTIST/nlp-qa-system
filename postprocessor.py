# ============================================================
# MODULE 3 — postprocessor.py
# Owner: Person 3
# Responsibility: Format and evaluate the model's raw output
# ============================================================


def format_confidence(score: float) -> tuple[str, str]:
    """
    Converts a raw confidence score (0–1) into a
    human-readable label and an emoji indicator.

    Returns:
        (label: str, emoji: str)
    """
    if score >= 0.80:
        return "High", "🟢"
    elif score >= 0.50:
        return "Medium", "🟡"
    else:
        return "Low", "🔴"


def highlight_answer_in_context(context: str, start: int, end: int) -> str:
    """
    Wraps the answer span inside the context with markdown bold
    so it can be visually highlighted in the Streamlit UI.

    Example:
        "The sky is blue." → "The sky is **blue**."
    """
    before  = context[:start]
    answer  = context[start:end]
    after   = context[end:]
    return f"{before}**{answer}**{after}"


def build_response(raw_result: dict, context: str) -> dict:
    """
    Takes the raw output from qa_engine and enriches it
    with confidence labels and highlighted context.

    Args:
        raw_result : dict returned by qa_engine.get_answer()
        context    : the original cleaned context string

    Returns a response dict with:
        {
            "answer"             : str,
            "confidence_score"   : float,
            "confidence_label"   : str,
            "confidence_emoji"   : str,
            "highlighted_context": str
        }
    """
    answer  = raw_result.get("answer", "No answer found.")
    score   = raw_result.get("score", 0.0)
    start   = raw_result.get("start", 0)
    end     = raw_result.get("end", 0)

    label, emoji = format_confidence(score)
    highlighted  = highlight_answer_in_context(context, start, end)

    return {
        "answer"             : answer,
        "confidence_score"   : round(score * 100, 2),  # as percentage
        "confidence_label"   : label,
        "confidence_emoji"   : emoji,
        "highlighted_context": highlighted
    }