import faiss
import numpy as np
import os
import json

dimension = 384
index = faiss.IndexFlatL2(dimension)
metadata = []
feedback_log = []
responses_log = {}


def save_store():
    os.makedirs("store", exist_ok=True)
    faiss.write_index(index, "store/indexes.bin")
    with open("store/metadata.json", "w") as f: 
        json.dump(metadata, f)
    with open("store/responses_log.json", "w") as f: 
        json.dump(responses_log, f)
    with open("store/feedback_log.json", "w") as f: 
        json.dump(feedback_log, f)


def load_store():
    global index, metadata, responses_log, feedback_log
    
    if os.path.exists("store/indexes.bin"):
        index = faiss.read_index("store/indexes.bin")
    
    if os.path.exists("store/metadata.json"):
        with open("store/metadata.json", "r") as f:
            metadata = json.load(f)
    
    if os.path.exists("store/responses_log.json"):
        with open("store/responses_log.json", "r") as f:
            responses_log = json.load(f)
    
    if os.path.exists("store/feedback_log.json"):
        with open("store/feedback_log.json", "r") as f:
            feedback_log = json.load(f)