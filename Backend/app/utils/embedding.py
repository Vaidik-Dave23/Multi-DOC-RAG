import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.store import index, metadata

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_and_store(chunks: list[str], doc_id: str):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    for i, chunk in enumerate(chunks):
        metadata.append({
            "doc_id": doc_id,
            "chunk_index": i,
            "text": chunk
        })

def embed_query(text: str):
    embedding = model.encode([text])
    embedding = np.array(embedding).astype('float32')
    faiss.normalize_L2(embedding)
    return embedding

embedding_model = model