from mpi4py import MPI
import networkx as nx
import time


comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()  # 5

start_time = time.time()
if rank == 0:   # broadcaster
    # reading in graph data here
    # G = nx.read_edgelist("twitter_combined.txt", create_using=nx.DiGraph)
    G = nx.path_graph(20)
    all_nodes = list(G.nodes)
    # divide into chunks
    nodes_subset = [[] for _ in range(size)]
    for i, chunk in enumerate(all_nodes):
        nodes_subset[i % size].append(chunk)
else:
    G = None
    nodes_subset = None

# MPI calls
G = comm.bcast(G, root=0)
nodes_subset = comm.scatter(nodes_subset, root=0)

print("I am processor", rank)
print("These are the nodes I have:")
print(nodes_subset)

# lists for results
lengths = []
sums = []
ccs = []

# doing dijkstra's
#for i in nodes_subset:
    # print("Source:", i)
#    for j in all_nodes:
#        if nx.has_path(G, i, j):
#            length, path = (nx.bidirectional_dijkstra(G, i, j))
#            lengths.append(length)
#    sums.append(sum(lengths))
#    for j in sums:
#        if not j == 0:
#            j = (N-1)/j
#    ccs.append(j)
#    lengths.clear()

# gathering all closeness centrality values
#cc_vals = comm.gather(ccs, root=0)
end_time = time.time()

if rank == 0:
#    print(cc_vals, '\n')
    print("Time taken:", (end_time - start_time))
