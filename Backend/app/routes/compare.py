from fastapi import APIRouter
from pydantic import BaseModel
from google import genai
from sklearn.metrics.pairwise import cosine_similarity
from app.store import metadata
from app.utils.embedding import embed_query
import numpy as np
import os
from dotenv import load_dotenv
from app.utils.embedding import embedding_model
from typing import List

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
        doc_chunks = [c for c in metadata if c["doc_id"] == doc_id]
        
        if not doc_chunks:
            continue
            
        chunk_texts = [c["text"] for c in doc_chunks]
        chunk_embeddings = chunk_embeddings = embedding_model.encode(chunk_texts).astype('float32')
        
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
    
    return {
        "comparison": response.text,
        "doc_ids": body.doc_ids,
        "query": body.query
    }