"""
PDF reading utilities for KARMA.

This module provides functionality to extract text from PDF files
with error handling and optimization for academic papers.

Supports PyMuPDF (fitz) as primary extractor with PyPDF2 fallback.
"""

import logging
import re
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF — preferred, handles complex font encodings
except ImportError:
    fitz = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

logger = logging.getLogger(__name__)


class PDFReader:
    """
    PDF text extraction utility with error handling.

    This class provides robust PDF text extraction with fallback
    mechanisms and special handling for academic papers.

    Extraction priority:
      1. PyMuPDF (fitz) — best quality, handles custom font encodings
      2. PyPDF2 — fallback
    """

    def __init__(self):
        """Initialize PDF reader."""
        if fitz is not None:
            logger.info("Using PyMuPDF (fitz) for PDF extraction.")
        elif PyPDF2 is not None:
            logger.warning(
                "PyMuPDF not installed; falling back to PyPDF2. "
                "Some PDFs with custom font encodings may produce garbled text. "
                "Install PyMuPDF for better results: pip install PyMuPDF"
            )
        else:
            logger.warning("No PDF library installed. PDF reading will not work.")

    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ImportError: If no PDF library is available
            Exception: If PDF reading fails
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if fitz is not None:
            return self._extract_with_pymupdf(pdf_path)
        elif PyPDF2 is not None:
            return self._extract_with_pypdf2(pdf_path)
        else:
            raise ImportError(
                "No PDF library available. Install PyMuPDF (recommended) or PyPDF2:\n"
                "  pip install PyMuPDF   # recommended\n"
                "  pip install PyPDF2"
            )

    # ------------------------------------------------------------------
    # Extraction backends
    # ------------------------------------------------------------------

    def _extract_with_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF (fitz)."""
        try:
            doc = fitz.open(str(pdf_path))

            if doc.is_encrypted:
                logger.warning(f"PDF {pdf_path} is encrypted. Attempting to decrypt...")
                if not doc.authenticate(""):
                    raise RuntimeError(f"Failed to decrypt PDF: {pdf_path}")

            total_pages = len(doc)
            logger.info(f"Extracting text from {total_pages} pages (PyMuPDF)...")

            parts = []
            for page_num in range(total_pages):
                try:
                    page_text = doc[page_num].get_text()
                    if page_text:
                        parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")

            doc.close()

            text = "\n\n".join(parts)
            if not text.strip():
                logger.warning(f"No text extracted from PDF: {pdf_path}")
                return ""

            text = self._post_process_text(text)
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text

        except Exception as e:
            logger.error(f"Failed to read PDF {pdf_path} with PyMuPDF: {e}")
            raise

    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2 (fallback)."""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                if pdf_reader.is_encrypted:
                    logger.warning(f"PDF {pdf_path} is encrypted. Attempting to decrypt...")
                    try:
                        pdf_reader.decrypt('')
                    except Exception as e:
                        logger.error(f"Failed to decrypt PDF: {e}")
                        raise

                total_pages = len(pdf_reader.pages)
                logger.info(f"Extracting text from {total_pages} pages (PyPDF2)...")

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue

                if not text.strip():
                    logger.warning(f"No text extracted from PDF: {pdf_path}")
                    return ""

                text = self._post_process_text(text)
                logger.info(f"Successfully extracted {len(text)} characters from PDF")
                return text

        except Exception as e:
            logger.error(f"Failed to read PDF {pdf_path} with PyPDF2: {e}")
            raise

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _post_process_text(self, text: str) -> str:
        """
        Post-process extracted text to improve quality.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Normalize paragraph breaks (collapse 3+ newlines to 2)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Normalize spaces / tabs within lines
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove leading spaces on lines
        text = re.sub(r'\n ', '\n', text)

        # Fix common PDF extraction issues
        # Remove hyphenation at line breaks
        text = re.sub(r'-\n([a-z])', r'\1', text)
        # Fix broken words across lines
        text = re.sub(r'([a-z])\n([a-z])', r'\1\2', text)

        return text.strip()

    def is_pdf_readable(self, pdf_path: Path) -> bool:
        """
        Check if a PDF file can be read.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            True if PDF is readable
        """
        try:
            self.extract_text(pdf_path)
            return True
        except Exception:
            return False

    def get_pdf_info(self, pdf_path: Path) -> Optional[dict]:
        """
        Get metadata information from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with PDF metadata or None if failed
        """
        if PyPDF2 is None:
            return None

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                info = {
                    'pages': len(pdf_reader.pages),
                    'encrypted': pdf_reader.is_encrypted,
                    'metadata': {}
                }

                # Extract metadata if available
                if pdf_reader.metadata:
                    metadata = pdf_reader.metadata
                    info['metadata'] = {
                        'title': metadata.get('/Title', ''),
                        'author': metadata.get('/Author', ''),
                        'subject': metadata.get('/Subject', ''),
                        'creator': metadata.get('/Creator', ''),
                        'producer': metadata.get('/Producer', ''),
                        'creation_date': str(metadata.get('/CreationDate', '')),
                        'modification_date': str(metadata.get('/ModDate', ''))
                    }

                return info

        except Exception as e:
            logger.error(f"Failed to get PDF info: {e}")
            return None