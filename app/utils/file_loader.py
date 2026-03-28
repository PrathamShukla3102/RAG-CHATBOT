import os
import pdfplumber
from docx import Document


class FileLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.extension = self._get_extension()

    def _get_extension(self):
        return os.path.splitext(self.file_path)[1].lower()

    def load(self) -> str:
        if self.extension == ".pdf":
            return self._load_pdf()
        elif self.extension == ".docx":
            return self._load_docx()
        elif self.extension == ".txt":
            return self._load_txt()
        else:
            raise ValueError(f"Unsupported file type: {self.extension}")

    #  PDF 
    def _load_pdf(self) -> str:
        text = []
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
        except Exception as e:
            raise RuntimeError(f"Error reading PDF: {e}")

        return "\n".join(text)

    # DOCX 
    def _load_docx(self) -> str:
        try:
            doc = Document(self.file_path)
            text = [para.text for para in doc.paragraphs if para.text.strip()]
        except Exception as e:
            raise RuntimeError(f"Error reading DOCX: {e}")

        return "\n".join(text)

    # TXT 
    def _load_txt(self) -> str:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Error reading TXT: {e}")