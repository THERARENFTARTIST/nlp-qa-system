# ============================================================
# MODULE 5 — file_handler.py
# Owner: Person 1 (extension of preprocessor role)
# Responsibility: Read uploaded PDF/TXT files and chunk them
# ============================================================

import pdfplumber
import re
from collections import Counter


def read_txt_file(file) -> str:
    """
    Reads a plain .txt file uploaded via Streamlit.
    Returns the full text as a string.
    """
    return file.read().decode("utf-8")


def remove_repeated_lines(pages: list[str], threshold: int = 3) -> list[str]:
    """
    Detects and removes lines that repeat across many pages
    (i.e. headers and footers like college name, roll number, subject, etc.)

    How it works:
        - Counts how many pages each line appears in
        - If a line appears on `threshold` or more pages, it's a header/footer → remove it

    Args:
        pages     : list of per-page text strings
        threshold : min number of pages a line must appear on to be considered a header/footer

    Returns:
        Cleaned list of per-page text strings.
    """
    # Collect all lines across all pages
    all_lines = []
    for page in pages:
        lines = [line.strip() for line in page.split("\n") if line.strip()]
        all_lines.extend(lines)

    # Count how many times each line appears
    line_counts = Counter(all_lines)

    # Lines appearing on `threshold` or more pages are headers/footers
    repeated = {line for line, count in line_counts.items() if count >= threshold}

    # Remove repeated lines from each page
    cleaned_pages = []
    for page in pages:
        lines = page.split("\n")
        cleaned = [l for l in lines if l.strip() not in repeated]
        cleaned_pages.append("\n".join(cleaned))

    return cleaned_pages


def read_pdf_file(file) -> str:
    """
    Reads a .pdf file uploaded via Streamlit using pdfplumber.
    Extracts text from all pages, then removes repeated header/footer lines.
    Returns the full cleaned text as a string.
    """
    pages = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)

    # Remove repeated headers/footers across pages
    cleaned_pages = remove_repeated_lines(pages, threshold=3)
    return "\n".join(cleaned_pages)


def extract_text_from_file(file, file_type: str) -> str:
    """
    Routes to the correct reader based on file type.

    Args:
        file      : uploaded file object from Streamlit
        file_type : "pdf" or "txt"

    Returns:
        Extracted text as a string.
    """
    if file_type == "pdf":
        return read_pdf_file(file)
    elif file_type == "txt":
        return read_txt_file(file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def split_into_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Splits a long document into overlapping word-level chunks.

    Why overlap? So that answers near chunk boundaries are not missed.

    Args:
        text       : full document text
        chunk_size : number of words per chunk (default 200)
        overlap    : number of words shared between consecutive chunks (default 50)

    Returns:
        A list of text chunks (strings).
    """
    # Clean and split into words
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # slide window with overlap

    return chunks