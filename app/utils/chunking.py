from typing import List, Dict


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str) -> List[Dict]:
        chunks = []

        start = 0
        text_length = len(text)
        chunk_id = 0

        while start < text_length:
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text.strip(),
                "source": source,
                "chunk_id": chunk_id
            })

            chunk_id += 1

            # Move start forward with overlap
            start += self.chunk_size - self.chunk_overlap

        return chunks