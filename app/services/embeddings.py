from dotenv import load_dotenv
import os
from typing import List
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EmbeddingService:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        try:
            response = client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")

    
    def embed_in_batches(self, texts: List[str], batch_size: int = 100):
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embed_text(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings