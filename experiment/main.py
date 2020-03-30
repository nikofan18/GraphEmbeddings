import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time
from numpy import dot
from numpy.linalg import norm

# Dataset to use - options FB15K, FB15K237, WN18, WN18RR
dataset = "FB15K237"


def cosine_calc(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def get_embedding(x):
    return embeddings_dict[str(x)]


def caching(a, b):
    if (a, b) in cache_keys:
        cos_sim = cache[(a, b)]
    elif (b, a) in cache_keys:
        cos_sim = cache[(b, a)]
    else:
        cos_sim = -1
    return cos_sim


entities2id = pd.read_csv("../myTests/"+dataset+"_PROCESSED/entity2id.txt", sep="\t", header=None)
test2id = pd.read_csv("../myTests/"+dataset+"_PROCESSED/test_edgelist.txt", sep=" ", header=None)
embeddings_dict = {}

with open("../experiment/embeddings/"+dataset+"_EMBEDDINGS/whole.emd", 'r') as f:
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
for index0, row in test2id.iterrows():
    start_time = time.time()
    print(row[0])
    sim_list = list()
    temp = caching(row[0], row[1])
    if temp == -1:
        correct_sim = cosine_calc(get_embedding(row[0]),
                                  get_embedding(row[1]))
        cache[(row[0], row[1])] = correct_sim
        cache_keys = cache.keys()
    else:
        correct_sim = temp

    for index1, row1 in entities2id.iterrows():

        if str(row1[1]) in emb_dict_keys:

            temp = caching(row[0], row1[1])
            if temp != -1:
                sim_list.append(temp)
            else:
                similarity = cosine_calc(get_embedding(row[0]),
                                         get_embedding(row1[1]))
                sim_list.append(similarity)
                cache[(row[0], row1[1])] = similarity
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
