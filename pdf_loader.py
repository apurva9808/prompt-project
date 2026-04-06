"""
pdf_loader.py
-------------
Utility to extract plain text from resume.pdf located in the same
directory as this file.  Requires the `pypdf` package.

Usage:
    from pdf_loader import load_resume
    text = load_resume()
"""

import os

# Path to the resume (same folder as this script)
RESUME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resume.pdf")


def load_resume(path: str = RESUME_PATH) -> str:
    """
    Extract and return all text from a PDF file.

    Parameters
    ----------
    path : str
        Absolute path to the PDF.  Defaults to resume.pdf in the project folder.

    Returns
    -------
    str
        Concatenated text of every page, or a helpful error string if the file
        is missing or cannot be parsed.
    """
    if not os.path.exists(path):
        return (
            "[ERROR] resume.pdf not found.\n"
            f"Please copy your resume PDF to:\n  {path}\n"
            "Then re-run the demo."
        )

    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        return (
            "[ERROR] pypdf is not installed.\n"
            "Run:  pip install pypdf\n"
            "Then re-run the demo."
        )

    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        if not text:
            return "[ERROR] Could not extract any text from resume.pdf. The PDF may be image-based."
        return text
    except Exception as exc:
        return f"[ERROR] Failed to read PDF: {exc}"
