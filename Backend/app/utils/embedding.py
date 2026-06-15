import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import app.store as store

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_and_store(chunks: list[str], doc_id: str , filename: str= "Unknown"):
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    store.index.add(embeddings)
    for i, chunk in enumerate(chunks):
        store.metadata.append({
            "doc_id": doc_id,
            "chunk_index": i,
            "text": chunk,
            "filename": filename   
        })

def embed_query(text: str):
    embedding = model.encode([text])
    embedding = np.array(embedding).astype('float32')
    faiss.normalize_L2(embedding)
    return embedding

embedding_model = model