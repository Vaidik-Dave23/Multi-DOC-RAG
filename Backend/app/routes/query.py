from fastapi import APIRouter
from pydantic import BaseModel
from app.utils.embedding import embed_query
from app.store import index, metadata
from google import genai
import os
from dotenv import load_dotenv


load_dotenv()


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

router = APIRouter()

class QueryInput(BaseModel):
    question: str

@router.post("/query")
async def query(body: QueryInput):
    embedding = embed_query(body.question)
    distances, indices = index.search(embedding, k=5)
    retrieved_chunks = []
    for i in indices[0]:
        retrieved_chunks.append(metadata[i])
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
        model="gemini-2.0-flash",
        contents=prompt
    )
    answer = response.text
    return {
        "answer": answer ,
        "sources": retrieved_chunks
    }