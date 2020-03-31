import numpy as np
import time
from numpy import dot
from numpy.linalg import norm

# Dataset to use - options FB15K, FB15K237, WN18, WN18RR
dataset = "FB15K237"
# Method to use - options node2vec, deepwalk
method = "node2vec"


def cosine_calc(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def get_embedding(x):
    return embeddings_dict[x]


def caching(a, b):
    if (a, b) in cache_keys:
        cos_sim = cache[(a, b)]
    elif (b, a) in cache_keys:
        cos_sim = cache[(b, a)]
    else:
        cos_sim = -1
    return cos_sim


embeddings_dict = {}
entities2id = list()
test2id = list()

with open("../myTests/" + dataset + "_PROCESSED/test_edgelist.txt", 'r') as f1:
    for line in f1:
        values = line.split(" ")
        test2id.append(values)

with open("../myTests/" + dataset + "_PROCESSED/entity2id.txt", 'r') as f0:
    for line in f0:
        values = line.split("\t")
        entities2id.append(values)

with open("../experiment/embeddings/" + method + "/" + dataset + "_EMBEDDINGS/whole.emd", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

emb_dict_keys = list(embeddings_dict.keys())

cache = {}
cache_keys = cache.keys()

MR = 0
HITS = 0
length = 0
for row in test2id:

    print(row[0])

    start_time = time.time()

    sim_list = list()
    temp = caching(row[0], row[1].rstrip())
    if temp == -1:
        correct_sim = cosine_calc(get_embedding(row[0]),
                                  get_embedding(row[1].rstrip()))
        cache[(row[0], row[1].rstrip())] = correct_sim
        cache_keys = cache.keys()
    else:
        correct_sim = temp

    for row1 in entities2id:

        if row1[1].rstrip() in emb_dict_keys:

            temp = caching(row[0], row1[1].rstrip())
            if temp != -1:
                sim_list.append(temp)
            else:
                similarity = cosine_calc(get_embedding(row[0]),
                                         get_embedding(row1[1].rstrip()))
                sim_list.append(similarity)
                cache[(row[0], row1[1].rstrip())] = similarity
                cache_keys = cache.keys()

    sim_list.sort(reverse=True)
    length = len(sim_list)
    corr_sim_index = sim_list.index(correct_sim)
    MR += corr_sim_index
    if corr_sim_index < 9:
        HITS += 1
    print("--- %s seconds ---" % (time.time() - start_time))

print("MR before division and then after division: ")
print(MR)
MR = MR / length
print(MR)

print("HITS before division and then after division: ")
print(HITS)
HITS = HITS / length
print(HITS)

# TODO filter setting
