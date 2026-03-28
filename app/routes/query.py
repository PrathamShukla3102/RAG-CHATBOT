from fastapi import APIRouter
from pydantic import BaseModel

from app.db.faiss_store import FAISSStore
from app.services.retrieval import RAGService

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


@router.post("/query")
def query_rag(request: QueryRequest):
    # Load FAISS
    store = FAISSStore(dim=1536)

    rag = RAGService(store)

    response = rag.generate(request.query)

    return {
        "answer": response["answer"],
        "sources": response["sources"]
    }