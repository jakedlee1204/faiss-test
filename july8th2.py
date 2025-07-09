import faiss
import numpy as np
from sklearn.preprocessing import normalize
import time
import pandas as pd

# Parameters
n_runs = 100
n_vectors = 1000000
n_dimensions = 128
k = 5
n_regions = 1000
nprobe = 10

# Generate normalized vectors
def generate_vectors(n, d):
    vectors = np.random.randn(n, d)
    return normalize(vectors, norm='l2').astype('float32')

# Brute-force cosine similarity using NumPy
def brute_force_cosine_search(database, query):
    sims = np.dot(database, query)
    return np.argsort(sims)[-k:][::-1]

# FAISS flat L2
def faiss_flat_l2_search(index, query):
    _, indices = index.search(query.reshape(1, -1), k)
    return indices[0]

# FAISS IVF with custom quantizer and metric
def faiss_ivf_search(index, query):
    _, indices = index.search(query.reshape(1,-1),k)
    return indices[0]

# Time a function across n runs
def average_time(search_fn):
    times = []
    for _ in range(n_runs):
        query = generate_vectors(1, n_dimensions)[0]
        start = time.time()
        search_fn(query)
        times.append(time.time() - start)
    return np.mean(times)

# Generate a fixed database once
database_vectors = generate_vectors(n_vectors, n_dimensions)

# -------------------------------
# Build FAISS Flat index (L2)
flat_index = faiss.IndexFlatL2(n_dimensions)
flat_index.add(database_vectors)

# -------------------------------
# Build IVF indexes (trained once, reused)

def build_ivf_index(quantizer_type, metric):
    if quantizer_type == "L2":
        quantizer = faiss.IndexFlatL2(n_dimensions)
    elif quantizer_type == "IP":
        quantizer = faiss.IndexFlatIP(n_dimensions)
    elif quantizer_type == "HNSW":
        quantizer = faiss.IndexHNSWFlat(n_dimensions, 32)
    else:
        raise ValueError("Unknown quantizer type")

    index = faiss.IndexIVFFlat(quantizer, n_dimensions, n_regions, metric)
    index.train(database_vectors)
    index.add(database_vectors)
    index.nprobe = nprobe
    return index

# set up index'; 
# 1st param is for quantizer (the coarse-level index for vectors' closest region assignment for train and adding); 
# 2nd param is for search metric FAISS uses when comparing the query vector to vectors inside the selected regions)
ivf_l2_index = build_ivf_index("L2", faiss.METRIC_L2) 
ivf_ip_index = build_ivf_index("IP", faiss.METRIC_INNER_PRODUCT)
ivf_hnsw_index = build_ivf_index("HNSW", faiss.METRIC_L2)

# Prepare search functions
results = {}

results["Brute Force Cosine"] = average_time(
    lambda q: brute_force_cosine_search(database_vectors, q)
)

results["FAISS Flat L2"] = average_time(
    lambda q: faiss_flat_l2_search(flat_index, q)
)

# For IVF, we batch the query vector
results["FAISS IVF (L2 Quantizer)"] = average_time(
    lambda q: faiss_ivf_search(ivf_l2_index, q)
)

results["FAISS IVF (IP Quantizer)"] = average_time(
    lambda q: faiss_ivf_search(ivf_ip_index, q)
)

results["FAISS IVF (HNSW Quantizer)"] = average_time(
    lambda q: faiss_ivf_search(ivf_hnsw_index, q)
)

# Display results as a table
df = pd.DataFrame.from_dict(results, orient='index', columns=['Avg Search Time (s)'])
df.index.name = "Method"

print(df.to_string(index=True))  # nicely formatted output

