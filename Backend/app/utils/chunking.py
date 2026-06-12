import re
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def naive_chunk(text: str, chunk_size=500, overlap=50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def sentence_chunk(text: str, chunk_size=500, overlap=50) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) > chunk_size:
            chunks.append(current.strip())
            current = current[-overlap:] + sentence
        else:
            current += " " + sentence
    if current:
        chunks.append(current.strip())
    return chunks

def semantic_chunk(text: str, threshold=0.85) -> List[str]:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = text.split('. ')
    embeddings = model.encode(sentences)
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        if sim < threshold:
            chunks.append('. '.join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append('. '.join(current))
    return chunks

def hierarchical_chunk(text: str) -> List[str]:
    sections = re.split(r'\n(?=[A-Z][^\n]{0,50}\n)', text)
    chunks = []
    for section in sections:
        paragraphs = section.split('\n\n')
        for para in paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            chunks.append('. '.join(sentences))
    return chunks