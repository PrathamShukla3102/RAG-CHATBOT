import os
from fastapi import APIRouter, UploadFile, File

from app.utils.file_loader import FileLoader
from app.utils.chunking import TextChunker
from app.services.embeddings import EmbeddingService
from app.db.faiss_store import FAISSStore

router = APIRouter()

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Step 1: Load
    loader = FileLoader(file_path)
    text = loader.load()

    # Step 2: Chunk
    chunker = TextChunker()
    chunks = chunker.chunk_text(text, source=file.filename)

    texts = [c["text"] for c in chunks]

    # Step 3: Embed
    embedder = EmbeddingService()
    embeddings = embedder.embed_in_batches(texts)

    # Step 4: Store
    dim = len(embeddings[0])
    store = FAISSStore(dim=dim)
    # CLEAR previous data
    store.index.reset()
    store.metadata = []
    store.add(embeddings, chunks)
    store.save()

    return {
        "message": "File processed successfully",
        "chunks": len(chunks)
    }