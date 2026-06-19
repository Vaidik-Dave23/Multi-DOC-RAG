import uuid
from fastapi import APIRouter, UploadFile, File
from app.utils.pdf import extract_text
from app.utils.chunking import naive_chunk
from app.utils.embedding import embed_and_store
from app.store import save_store

router = APIRouter()

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    contents = await file.read()
    text = extract_text(contents)
    chunks = naive_chunk(text)
    embed_and_store(chunks, doc_id, file.filename)
    save_store()
    return {"message": "Document uploaded successfully", "doc_id": doc_id}


import app.store as store

@router.get("/debug")
def debug():
    return {
        "total_vectors": store.index.ntotal,
        "total_chunks": len(store.metadata),
        "sample": store.metadata[:3]
    }

@router.get("/documents")
def get_documents():
    docs = {}
    for chunk in store.metadata:
        doc_id = chunk["doc_id"]
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "chunk_count": 0,
                "filename": chunk.get("filename", "Untitled")
            }
        docs[doc_id]["chunk_count"] += 1
    return {"documents": list(docs.values())}
@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    original_count = len(store.metadata)
    store.metadata = [c for c in store.metadata if c["doc_id"] != doc_id]
    removed = original_count - len(store.metadata)
    if removed == 0:
        return {"error": "Document not found"}
    save_store()
    return {"message": f"Removed {removed} chunks for {doc_id}"}