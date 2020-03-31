# importing networkx
import networkx as nx
# importing matplotlib.pyplot
import matplotlib.pyplot as plt

# Dataset to use - options FB15K, FB15K237, WN18, WN18RR
dataset = "FB15K237"

g = nx.Graph()

test2id = list()
with open("../myTests/" + dataset + "_PROCESSED/whole_edgelist.txt", 'r') as f1:
    for line in f1:
        values = line.split(" ")
        test2id.append(values)


for row in test2id:
    g.add_edge(row[0].rstrip(), row[1].rstrip())

print("Total number of nodes: ", int(g.number_of_nodes()))
print("Total number of edges: ", int(g.number_of_edges()))

zero_counter = 0
for node, degree in dict(g.degree()).items():
    if degree == 0:
        zero_counter += 1

print("Nodes with zero degree: " + str(zero_counter))
print("Graph is connected: " + str(nx.is_connected(g)))
print("Total number of self-loops: ", int(g.number_of_selfloops()))
