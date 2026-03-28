from fastapi import FastAPI
from app.routes import upload, query
from app.routes import reset
app = FastAPI(title="RAG Chatbot API")

app.include_router(upload.router)
app.include_router(query.router)
app.include_router(reset.router)

@app.get("/")
def root():
    return {"message": "RAG API is running 🚀"}