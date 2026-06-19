from fastapi import APIRouter
from pydantic import BaseModel
from google import genai
import numpy as np
import os
from dotenv import load_dotenv
from app.utils.embedding import embedding_model ,embed_query
from typing import List
import uuid
from app.store import responses_log
from app.store import save_store
import app.store as store
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
router = APIRouter()

class CompareInput(BaseModel):
    doc_ids: List[str]
    query: str

@router.post("/compare")
async def compare(body: CompareInput):
    query_embedding = embed_query(body.query)
    
    doc_contexts = []
    for i, doc_id in enumerate(body.doc_ids):
        doc_chunks = [c for c in store.metadata if c["doc_id"] == doc_id]
        
        if not doc_chunks:
            continue
            
        chunk_texts = [c["text"] for c in doc_chunks]
        chunk_embeddings = np.array(
            [c["embedding"] for c in doc_chunks], dtype="float32"
        )  # ← was: model.encode(chunk_texts) — now reads cached vectors instead
        
        scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:3]
        top_chunks = [chunk_texts[j] for j in top_indices]
        
        doc_contexts.append(f"Document {i+1} (ID: {doc_id}):\n" + "\n".join(top_chunks))
    
    combined_context = "\n\n---\n\n".join(doc_contexts)
    
    prompt = f"""
    You are a professional document comparison analyst working at a large legal and financial firm.

    Below are excerpts from multiple documents, each labeled separately:

    {combined_context}

    Compare these documents on the following:
    {body.query}

    Structure your response as:
    - A comparison for each document
    - A final summary of which document is most favorable/unfavorable and why
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    response_id = str(uuid.uuid4())
    responses_log[response_id] = {
        "query": body.query,
        "answer": response.text,
        "doc_ids": body.doc_ids
    }
    save_store()
    
    return {
        "response_id": response_id,
        "comparison": response.text,
        "contexts": doc_contexts,
        "doc_ids": body.doc_ids,
        "query": body.query
    }