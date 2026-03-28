from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv

from app.services.embeddings import EmbeddingService
from app.db.faiss_store import FAISSStore

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class RAGService:
    def __init__(self, faiss_store: FAISSStore):
        self.embedder = EmbeddingService()
        self.store = faiss_store

    # ---------- Retrieve ----------
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.embedder.embed_text([query])[0]
        results = self.store.search(query_embedding, top_k=top_k)

        # DEBUG (important)
        print("\nETRIEVED CHUNKS:")
        for r in results:
            print(f"\nSource: {r['source']} | Chunk: {r['chunk_id']}")
            print(r["text"][:200])

        return results

    # ---------- Build Prompt ----------
    def build_prompt(self, query: str, contexts: List[Dict]) -> str:
        context_text = "\n\n".join(
            [
                f"Document: {c['source']} (chunk {c['chunk_id']})\nContent:\n{c['text']}"
                for c in contexts
            ]
        )

        prompt = f"""
You are a highly intelligent AI assistant.

Your task is to answer the user's question using ONLY the provided context.

Instructions:
- Carefully read ALL the context before answering.
- Combine information from multiple chunks if needed.
- Give a clear, structured, and complete answer.
- If the answer is partially available, provide the best possible answer.
- Only say "I don't know" if the answer is completely absent.
- Always include source references in this format: (source, chunk_id)

Context:
{context_text}

Question:
{query}

Answer:
"""
        return prompt

    # ---------- Generate ----------
    def generate(self, query: str, top_k: int = 5):
        contexts = self.retrieve(query, top_k=top_k)

        # Handle empty retrieval
        if not contexts:
            return {
                "answer": "No relevant information found in documents.",
                "sources": []
            }

        prompt = self.build_prompt(query, contexts)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise question-answering assistant that strictly uses provided context."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            answer = response.choices[0].message.content

            return {
                "answer": answer,
                "sources": contexts
            }

        except Exception as e:
            print("LLM ERROR:", e)
            return {
                "answer": f"Error occurred: {str(e)}",
                "sources": []
            }