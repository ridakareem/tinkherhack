"""
services/pdf_service.py
========================
Handles PDF upload, storage, and text extraction.

LIBRARY DEPENDENCY:
-------------------
- pypdf: Used for extracting text from PDF files.
  Install: pip install pypdf

STORAGE:
--------
- PDFs are stored as BLOB data in SQLite.
- Extracted text is also stored to avoid re-parsing on every quiz generation.
- PDFs are scoped to users — a user can only access their own PDFs.

LIMITATIONS:
------------
- Scanned PDFs (image-only) will produce empty or poor text extraction.
- The system assumes text-based PDFs (notes, articles, textbooks).
"""

import io
from typing import Optional, Tuple


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract plain text from PDF binary data using pypdf.

    Args:
        pdf_bytes: Raw PDF file content as bytes

    Returns:
        Extracted text as a single string (pages separated by newlines)

    Raises:
        ImportError: If pypdf is not installed
        RuntimeError: If the PDF cannot be read
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required. Install it with: pip install pypdf")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")


def save_pdf(db, user_id: int, topic: str, filename: str, pdf_bytes: bytes) -> Tuple[int, str]:
    """
    Store a PDF in the database and extract its text.

    Args:
        db: SQLite database connection
        user_id: Current user's ID
        topic: Topic tag entered by the user
        filename: Original filename of the upload
        pdf_bytes: Raw PDF bytes

    Returns:
        Tuple of (pdf_id: int, extracted_text: str)
    """
    extracted_text = extract_text_from_pdf(pdf_bytes)

    cursor = db.execute("""
        INSERT INTO pdfs (user_id, topic, filename, content, extracted_text)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, topic.strip(), filename, pdf_bytes, extracted_text))
    db.commit()

    return cursor.lastrowid, extracted_text


def get_user_pdfs(db, user_id: int):
    """
    Retrieve all PDFs belonging to a user (metadata only, no BLOB).

    Args:
        db: SQLite database connection
        user_id: Current user's ID

    Returns:
        List of dicts with keys: id, topic, filename, uploaded_at
    """
    rows = db.execute("""
        SELECT id, topic, filename, uploaded_at
        FROM pdfs
        WHERE user_id = ?
        ORDER BY uploaded_at DESC
    """, (user_id,)).fetchall()

    return [dict(row) for row in rows]


def get_pdf_text(db, pdf_id: int, user_id: int) -> Optional[str]:
    """
    Retrieve extracted text for a specific PDF (access-controlled).

    Args:
        db: SQLite database connection
        pdf_id: PDF record ID
        user_id: Requesting user's ID (ownership check)

    Returns:
        Extracted text string, or None if not found / not authorized
    """
    row = db.execute("""
        SELECT extracted_text FROM pdfs
        WHERE id = ? AND user_id = ?
    """, (pdf_id, user_id)).fetchone()

    return row["extracted_text"] if row else None
