import uuid
from fastapi import APIRouter, UploadFile, File
from app.utils.pdf import extract_text
from app.utils.chunking import naive_chunk
from app.utils.embedding import embed_and_store

router = APIRouter()

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    contents = await file.read()
    text = extract_text(contents)
    chunks = naive_chunk(text)
    embed_and_store(chunks, doc_id)
    return {"message": "Document uploaded successfully", "doc_id": doc_id}


from app.store import index, metadata

@router.get("/debug")
def debug():
    return {
        "total_vectors": index.ntotal,
        "total_chunks": len(metadata),
        "sample": metadata[:3]
    }