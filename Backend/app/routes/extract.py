from fastapi import APIRouter
from google import genai
import os
from dotenv import load_dotenv
import json
from app.store import metadata
from pydantic import BaseModel

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

router = APIRouter()

class ExtractInput(BaseModel):
    doc_id: str
    fields: list[str]

@router.post("/extract")
def extract_text(body: ExtractInput):
    doc_chunks = [chunk for chunk in metadata if chunk["doc_id"] == body.doc_id]
    context = "\n\n".join([chunk["text"] for chunk in doc_chunks])

    prompt = f"""
    You are a professional data extraction specialist.

    Below is the content extracted from a document:

    {context}

    Extract the following fields from the document and return ONLY a valid JSON object with no extra text, no markdown, no explanation:

    Fields to extract: {body.fields}

    Example format:
    {{
        "field_name": "extracted value or null if not found",
    }}
    """
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    raw = response.text.strip().replace("```json", "").replace("```", "")
    extracted = json.loads(raw)
    return {"extracted_fields": extracted, "doc_id": body.doc_id}