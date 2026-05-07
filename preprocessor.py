# ============================================================
# MODULE 1 — preprocessor.py
# Owner: Person 1
# Responsibility: Clean and prepare raw text input
# ============================================================

import re

def clean_text(text: str) -> str:
    """
    Cleans raw input text by removing extra whitespace,
    special characters, and normalizing the content.
    """
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing spaces
    text = text.strip()

    return text


def split_into_sentences(text: str) -> list[str]:
    """
    Splits a cleaned paragraph into individual sentences.
    Useful for displaying context alongside answers.
    """
    # Simple sentence splitter using punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def validate_inputs(context: str, question: str) -> tuple[bool, str]:
    """
    Validates that both context and question are non-empty
    and meet minimum length requirements.

    Returns:
        (is_valid: bool, error_message: str)
    """
    if not context or not context.strip():
        return False, "⚠️ Please provide a context passage."

    if not question or not question.strip():
        return False, "⚠️ Please enter a question."

    if len(context.strip()) < 20:
        return False, "⚠️ Context is too short. Please provide more text."

    if len(question.strip()) < 5:
        return False, "⚠️ Question is too short. Please be more specific."

    return True, ""


def preprocess(context: str, question: str) -> tuple[str, str]:
    """
    Main preprocessing function.
    Cleans both context and question before passing to QA engine.

    Returns:
        (cleaned_context, cleaned_question)
    """
    cleaned_context = clean_text(context)
    cleaned_question = clean_text(question)
    return cleaned_context, cleaned_question