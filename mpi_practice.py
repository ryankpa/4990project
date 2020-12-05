from mpi4py import MPI
import networkx as nx
import time
import numpy as np
from fibheap import *


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

start_time = time.time()
if rank == 0:   # broadcaster
    # reading in graph data here
    # G = nx.read_edgelist("twitter_combined.txt", create_using=nx.DiGraph)
    G = nx.path_graph(5)
    # convert G to numpy matrix
    M = nx.to_numpy_matrix(G)
    all_nodes = list(G.nodes)
    # divide into chunks
    nodes_subset = [[] for _ in range(size)]
    for i, chunk in enumerate(all_nodes):
        nodes_subset[i % size].append(chunk)
else:
    M = None
    nodes_subset = None

# MPI calls
M = comm.bcast(M, root=0)
nodes_subset = comm.scatter(nodes_subset, root=0)

print("I am processor", rank)
print("These are the nodes I have:")
print(nodes_subset)

#
dist = []
heap1 = makefheap()

# lists for results
lengths = []
sums = []
ccs = []

# closeness centrality calculations here / dijkstra's
for source in nodes_subset:
    for v in all_nodes:
        dist[v] = float("inf")
        fheappush(heap1, (dist[v], v))
    dist[source] = 0
    fheappush(heap1, dist[source], source)
    while heap1 is not None:
        u = fheappop(heap1)
        for v in M:
            if v in heap1:
                alt = dist[u] + M[u, v]
                if alt < dist[v]:
                    dist[v] = alt
    print(dist)




# gathering all closeness centrality values
cc_vals = comm.gather(ccs, root=0)
end_time = time.time()

if rank == 0:
    print(cc_vals, '\n')
    print("Time taken:", (end_time - start_time))
