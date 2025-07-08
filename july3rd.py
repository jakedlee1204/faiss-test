import faiss
import numpy as np

d = 128
nb = 1000000
nlist = 100
m = 16
k = 10

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

xb = np.random.random((50000, d)).astype('float32')
index.train(xb)

index.add(xb)

index.nprobe = 10

xq = np.random.random((5,d)).astype('float32')
D, I = index.search(xq, k)
print(I)

print("Query Vector[0]:")
print(xq[0])

top_index = I[0][0]

print(f"\nClosest vector in database (xb[{top_index}]):")
print(xb[top_index])


from numpy.linalg import norm
# Compute L2 distance
dist = norm(xq[0] - xb[top_index])
print(f"\nL2 distance between query[0] and closest xb vector: {dist}")