import faiss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import time
from mpl_toolkits.mplot3d import Axes3D

def generate_sample_vectors(n_vectors=100, n_dimensions=3, seed=30):
    np.random.seed(seed) # set the seed for NumPy's random number generator (preventing different random #s)
    vectors = np.random.randn(n_vectors, n_dimensions)
    normalized_vectors = normalize(vectors, norm='l2') #appy L2 normalization (for cosine similarity)
    return normalized_vectors

def plot_vectors_3d(vectors, query_vector=None, matches=None, title="Vector Space"):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection = '3d')

    ax.scatter(vectors[:,0], vectors[:, 1], vectors[:, 2], c='blue', alpha=0.5, label='')

    if query_vector is not None:
        ax.scatter(query_vector[0], query_vector[1], query_vector[2], c='red', s=100, label='Query vector')

    if matches is not None:
        match_vectors = vectors[matches]
        ax.scatter(match_vectors[:,0], match_vectors[:,1], match_vectors[:,2], c='green', s=100, label='Matches')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()

def plot_vectors_with_regions(vectors, centroids, query_vector=None, matches=None, searched_regions=None,
                              title="Vector Space with FAISS Regions"):
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection = '3d')

    distances, assignments = compute_vector_assignments(vectors, centroids)

    colors = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
    for i in range(len(centroids)):
        cluster_vectors = vectors[assignments == i]
        if len(cluster_vectors) > 0:
            alpha = 1.0 if searched_regions is None or i in searched_regions else 0.5
            ax.scatter(cluster_vectors[:,0], cluster_vectors[:,1], cluster_vectors[:,2],
                       c=[colors[i]], alpha=alpha, label=f'Region {i}')
    
    ax.scatter(centroids[:,0], centroids[:, 1], centroids[:, 2], c='black', s=100, marker='*',label='Region Centers')

    if query_vector is not None:
        ax.scatter(query_vector[0], query_vector[1], query_vector[2], c='red', s=200, marker='x',label='Query vector')

    if matches is not None:
        match_vectors = vectors[matches]
        ax.scatter(match_vectors[:,0], match_vectors[:,1], match_vectors[:,2], c='green', s=100, label='Matches')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()

def compute_vector_assignments(vectors, centroids): # assign each vector to its closest centroid (region)
    """Compute which vectors belong to which centroids"""
    index = faiss.IndexFlatL2(vectors.shape[1]) # create a flat FAISS index using L2 distance w/ one dimension per vector
    index.add(centroids) # add centroids to the index 
    distances, assignments = index.search(vectors, 1) # for each vector, find the nearest centroids then return L2 distance to the nearest centroid & index of the closest centroid for each vector
    return distances, assignments.ravel() # distances (2D array of shape [n,1]); flatten the array to shape [n] for processing

def train_kmeans_get_centroids(vectors, n_clusters): # train KMeans to get region centroids
    """Train k-means (divide vectors into clusters)and get centroids"""
    kmeans = faiss.Kmeans(d=vectors.shape[1], k=n_clusters, niter=20, verbose=False) #d = dimension of the vectors; k = number of clusters; niter = # of iterations; verbose = logging output
    kmeans.train(vectors) # run the k-means clustering algorithm
    return kmeans.centroids # return the learned centroid coordinates

def brute_force_cosine_search(database_vectors, query_vector, k=5): # compute dot product and full-scan linear search
    """Brute force cosine similarity search"""
    start_time = time.time()
    similarities = np.dot(database_vectors, query_vector) # calculate cosine similarity b/w the query vector and each vector in the database; since vectors normalized, dot product equals cosine similarity
    top_k_indices = np.argsort(similarities)[-k:][::-1] # sort similarities in ascending order, then take the last k values, and reverse them to get top-k
    end_time = time.time()
    return top_k_indices, end_time - start_time # return indices of the top-k most similar vectors & time taken to compute

def faiss_flat_l2_search(database_vectors, query_vector, k=5): # use FAISS's exact IndexFlatL2 for brute-force L2 search
    """basic FAISS L2 search without regions"""
    dimension = database_vectors.shape[1] # grabs the dimensionality of the vectors
    index = faiss.IndexFlatL2(dimension) # create an exact search FAISS index using L2 distance

    start_time = time.time()
    index.add(database_vectors) # add full database to the FAISS index; all vectors stored in memory for brute-force L2 search
    distances, indices = index.search(query_vector.reshape(1,-1), k) # queries the index with query_vector, reshaped to shape [1,d]; returns squared L2 distances to nearest vectors & indices of top-k matches
    end_time = time.time() 

    return indices[0], end_time - start_time # returns top-k matches (since batch size is 1) & time taken 

def faiss_ivf_search(database_vectors, query_vectors, k=5, n_regions=10, nprobe=3): # use IVF by training n_regions clusters 
                                                                                    # and storing vectors in regions then searching 
                                                                                    # only nprobe closest regions to query
    dimension = database_vectors.shape[1] # get vector dimensionality

    print("Training index")
    quantizer = faiss.IndexFlatL2(dimension) # flat L2 (coarse-level) index used only as a quantizer to train and assign vectors to closest regions
    index = faiss.IndexIVFFlat(quantizer, dimension, n_regions, faiss.METRIC_L2) # creates IVF index with n_regions clusters, a FAISS index used only to assign vectors to regions, and stores all vectors within each region as-is

    train_start = time.time()
    index.train(database_vectors) # trains k-means on the database vectors and generate n_regions of centroids
    train_time = time.time() - train_start

    add_start = time.time()
    index.add(database_vectors) # assign each vector to its closest region centroid (stores inverted lists)
    add_time = time.time() - add_start

    index.nprobe = nprobe # controls how many regions (centroids) FAISS searches during a query

    search_start = time.time() 
    distances, indices = index.search(query_vectors, k) # actual IVF search only inside nprobe regions using L2 distance
    search_time = time.time() - search_start

    return indices, search_time, train_time, add_time # returning match indices, search time, & time to train and add vec

# 1. Generate sample data
n_vectors = 1000 # number of vec in database
n_dimensions = 3 # dimensionality
k = 5 # top-k matches to retrieve per search
print(f"Generating {n_vectors} vectors with {n_dimensions} dimensions...")
database_vectors = generate_sample_vectors(n_vectors, n_dimensions) # generate a database of 1000 L2-normalized 3D vectors for search

# 2. Generate a random query vector
query_vector = generate_sample_vectors(1, n_dimensions)[0] # create one random L2-normalized query vector
query_vector_batch = query_vector.reshape(1, -1) # Reshape vector into 2D array for FAISSS batch processing

# 3. Visualize initial vector space
print("\nVisualizing initial vector space...")
plot_vectors_3d(database_vectors, query_vector, title="Initial Vector Space") # display a 3D scatter plot of all database vectors

# 4. Perform brute force cosine similarity search
print("\nPerforming brute force cosine similarity search...")
cosine_matches, cosine_time = brute_force_cosine_search(database_vectors, query_vector) # run a manual cosne similarity search using NumPy dot product; get indices of the k=5 most similar vector to the query
print(f"Brute force seach time: {cosine_time: .6f} seconds")
print(f"Top {k} cosine similarity matches (indices): {cosine_matches}")

# 5. Visualize cosine results
plot_vectors_3d(database_vectors, query_vector, cosine_matches, "Brute Force Cosine Similarity Results")

#6. Peform Basic FAISS L2 Search
print("\nPerforming basic FAISS L2 Search...")
faiss_matches, faiss_time = faiss_flat_l2_search(database_vectors, query_vector, k) #use FAISS's IndexFlatL2 to perform exact L2-based NN-search
print(f"FAISS L2 search time: {faiss_time:.6f} seconds")
print(f"Top {k} L2 distance matches (indices): {faiss_matches}")

# 7. Visualize basic FAISS results
plot_vectors_3d(database_vectors, query_vector, faiss_matches, "FAISS L2 search results")

# 8. Perform  FAISS IVF search with regions
print("\nPerforming FAISS IVF search with regions...")
n_regions = 10 # number of clusters (centroids) to divide the database into
nprobe = 3 # how many regions to actually search

# Create and train index
dimension = database_vectors.shape[1] # number of features (3)
quantizer = faiss.IndexFlatL2(dimension) # a flat L2 index used to assign vectors to regions
index = faiss.IndexIVFFlat(quantizer, dimension, n_regions, faiss.METRIC_L2) #IVF index that stores vectors grouped into regions, each region being assigned by the quantizer

print("Training index...")
train_start = time.time()
index.train(database_vectors) # run k-means under the hood to define n_regions clusters; training must be done before adding vectors
train_time = time.time() - train_start
print(f"Training time: {train_time:.6f} seconds")

print("Adding vectors...")
add_start = time.time()
index.add(database_vectors) #assign each vector to its closest region based on the trained quantizer
add_time = time.time() - add_start
print(f"Adding time: {add_time:.6f} seconds")

# Set # of regions to search
index.nprobe = nprobe

# Search
print("Searching...")
search_start = time.time() 
distances, ivf_matches = index.search(query_vector_batch, k) #perform IVF search
search_time = time.time() - search_start
print(f"Search time: {search_time:.6f} seconds")
ivf_matches = ivf_matches[0] #first batch results of top-k L2-nearest neighbors

# centroids for visuals
centroids = train_kmeans_get_centroids(database_vectors, n_regions)

# searched regions (approximated using nearest centroids to query)
_, searched_regions = quantizer.search(query_vector_batch, nprobe) 
searched_regions = searched_regions[0]

print(f"Total time (train + add + search): {train_time + add_time + search_time:.6f} seconds")
print(f"Search-only time: {search_time:.6f} seconds")
print(f"Searched {nprobe} out of {n_regions} regions: {searched_regions}")

# 9. Visualize IVF results with regions
plot_vectors_with_regions(
    database_vectors,
    centroids,
    query_vector,
    ivf_matches,
    searched_regions,
    "FAISS IVF Search Results (Highlighted Searched Regions)"
)

# 10. Compare results
print("\n Comparing results between methods:")
common_matches_basic = set(cosine_matches).intersection(set(faiss_matches))
common_matches_ivf = set(cosine_matches).intersection(set(ivf_matches))
print(f"Common matches (Cosine vs Basic FAISS): {len(common_matches_basic)}")
print(f"Common match indices: {common_matches_basic}")
print(f"Common matches (Cosine vs IVF FAISS): {len(common_matches_ivf)}")
print(f"Common match indices: {common_matches_ivf}")















