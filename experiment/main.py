import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time
from numpy import dot
from numpy.linalg import norm

start_time = time.time()


def cosine_calc(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def get_embedding(x, keys, values):
    result = list()
    ind = keys[values.index(x)]
    for i in range(128):
        if i > 0:
            result.append(embeddings.at[ind, i])
    return np.asarray(result)


def caching(a, b):
    cache_keys = cache.keys()

    if (a, b) in cache_keys:
        cos_sim = cache[(a, b)]
    elif (b, a) in cache_keys:
        cos_sim = cache[(b, a)]
    else:
        cos_sim = -1
    return cos_sim


# produce the whole dataset as edgelist
# test_edgelist = pd.read_csv("../myTests/FB15K_PROCESSED/test_edgelist.txt", sep=" ", header=None)
# train_edgelist = pd.read_csv("../myTests/FB15K_PROCESSED/train_edgelist.txt", sep=" ", header=None)
# valid_edgelist = pd.read_csv("../myTests/FB15K_PROCESSED/valid_edgelist.txt", sep=" ", header=None)
# whole_edgelist = pd.concat([test_edgelist, train_edgelist, valid_edgelist]).drop_duplicates().reset_index(drop=True)
# whole_edgelist.to_csv("../myTests/FB15K_PROCESSED/whole_edgelist.txt", index=False, header=False, sep=" ")

embeddings = pd.read_csv("../experiment/embeddings/FB15K_EMBEDDINGS/whole.emd", sep=" ", header=None)
entities2id = pd.read_csv("../myTests/FB15K_PROCESSED/entity2id.txt", sep="\t", header=None)
test2id = pd.read_csv("../myTests/FB15K_PROCESSED/test_edgelist.txt", sep=" ", header=None)

emb_dict = embeddings[0].to_dict()
emb_dict_keys = list(emb_dict.keys())
emb_dict_values = list(emb_dict.values())

cache = {}

# print("--- %s seconds ---" % (time.time() - start_time))
# exit(0)

MR = 0
HITS = 0
MRR = 0
for index0, row in test2id.iterrows():

    sim_list = list()
    temp = caching(row[0], row[1])
    if temp == -1:
        correct_sim = cosine_calc(get_embedding(row[0], emb_dict_keys, emb_dict_values),
                                  get_embedding(row[1], emb_dict_keys, emb_dict_values))
        cache[(row[0], row[1])] = correct_sim
    else:
        correct_sim = temp

    for index1, row1 in entities2id.iterrows():

        if row1[1] in emb_dict_values:

            temp = caching(row[0], row1[1])
            if temp != -1:
                sim_list.append(temp)
            else:
                similarity = cosine_calc(get_embedding(row[0], emb_dict_keys, emb_dict_values),
                                         get_embedding(row1[1], emb_dict_keys, emb_dict_values))
                sim_list.append(similarity)
                cache[(row[0], row1[1])] = similarity

    sim_list.sort(reverse=True)
    corr_sim_index = sim_list.index(correct_sim)
    MR += corr_sim_index
    # MRR += 1 / corr_sim_index + 1
    if corr_sim_index <= 10:
        HITS += 1
# TODO complete the calculation of metrics
