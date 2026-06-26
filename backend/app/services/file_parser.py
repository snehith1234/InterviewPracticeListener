from pathlib import Path
import fitz
from docx import Document


def extract_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(file_path)
    if suffix == ".docx":
        return extract_docx(file_path)
    if suffix == ".txt":
        return path.read_text(errors="ignore")
    raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")


def extract_pdf(file_path: str) -> str:
    text_parts = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts).strip()


def extract_docx(file_path: str) -> str:
    doc = Document(file_path)
    lines = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(lines).strip()
