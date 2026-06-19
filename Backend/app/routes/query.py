from fastapi import APIRouter
from pydantic import BaseModel
from app.utils.embedding import embed_query
import app.store as store
from google import genai
import os
import hashlib
import json
from dotenv import load_dotenv
from app.store import responses_log
import uuid
from app.store import save_store


load_dotenv()


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

router = APIRouter()


class QueryInput(BaseModel):
    question: str

@router.post("/query")
async def query(body: QueryInput):
    embedding = embed_query(body.question)
    
    retrieved_chunks = []
    if len(store.metadata) > 0 and store.index.ntotal > 0:
        distances, indices = store.index.search(embedding, k=5)
        if indices is not None and len(indices) > 0:
            for i in indices[0]:
                if i != -1 and 0 <= i < len(store.metadata):
                    retrieved_chunks.append(store.metadata[i])
                    
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    

    prompt = f"""
    You are a professional document reviewer working in the document review department of a large firm.

    The user has uploaded documents and the relevant information extracted from them is below:

    {context}

    Based on the above information, answer the following question clearly and professionally:

    {body.question}

    Always mention which part of the document your answer is based on.
    """
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    answer = response.text
    response_id = str(uuid.uuid4())

    
    responses_log[response_id] = {
        "query": body.question,
        "answer": answer,
        "doc_ids": [chunk["doc_id"] for chunk in retrieved_chunks]
    }
    response_data = {
        "response_id": response_id,
        "answer": answer,
        "contexts": [chunk["text"] for chunk in retrieved_chunks],
        "sources": retrieved_chunks
    }

    save_store()
    
    return response_data
