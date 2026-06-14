import faiss
import numpy as np

dimension = 384
index = faiss.IndexFlatL2(dimension)
metadata = []
feedback_log = []
responses_log = {}