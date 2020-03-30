import pandas as pd

data = pd.read_csv("../data/FB15K/valid2id.txt", sep=" ", header=None)
data.columns = ["e1", "r", "e2"]
data = data.drop("r", axis=1)
data = data.sort_values(by=['e1'])
data = data.drop_duplicates(subset=['e1', 'e2'], keep="first")
data.to_csv("../myTests/FB15K_PROCESSED/valid_edgelist.txt", index=False, header=False, sep=" ")
