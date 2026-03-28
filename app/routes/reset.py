from fastapi import APIRouter
import os

router = APIRouter()

@router.post("/reset")
def reset_data():
    try:
        if os.path.exists("vector_store/faiss.index"):
            os.remove("vector_store/faiss.index")

        if os.path.exists("vector_store/metadata.pkl"):
            os.remove("vector_store/metadata.pkl")

        return {"message": "Data cleared successfully "}

    except Exception as e:
        return {"error": str(e)}