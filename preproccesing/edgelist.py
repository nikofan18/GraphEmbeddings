import pandas as pd

# produce edgelists in order to run node2vec for them

source = "FB15K237"
dest = "FB15K237_PROCESSED"

data = pd.read_csv("../data/"+source+"/valid2id.txt", sep=" ", header=None)
data.columns = ["e1", "r", "e2"]
data = data.drop("r", axis=1)
data = data.sort_values(by=['e1'])
data = data.drop_duplicates(subset=['e1', 'e2'], keep="first")
data.to_csv("../myTests/"+dest+"/valid_edgelist.txt", index=False, header=False, sep=" ")

data = pd.read_csv("../data/"+source+"/test2id.txt", sep=" ", header=None)
data.columns = ["e1", "r", "e2"]
data = data.drop("r", axis=1)
data = data.sort_values(by=['e1'])
data = data.drop_duplicates(subset=['e1', 'e2'], keep="first")
data.to_csv("../myTests/"+dest+"/test_edgelist.txt", index=False, header=False, sep=" ")

data = pd.read_csv("../data/"+source+"/train2id.txt", sep=" ", header=None)
data.columns = ["e1", "r", "e2"]
data = data.drop("r", axis=1)
data = data.sort_values(by=['e1'])
data = data.drop_duplicates(subset=['e1', 'e2'], keep="first")
data.to_csv("../myTests/"+dest+"/train_edgelist.txt", index=False, header=False, sep=" ")

# produce the whole dataset as edgelist
test_edgelist = pd.read_csv("../myTests/"+dest+"/test_edgelist.txt", sep=" ", header=None)
train_edgelist = pd.read_csv("../myTests/"+dest+"/train_edgelist.txt", sep=" ", header=None)
valid_edgelist = pd.read_csv("../myTests/"+dest+"/valid_edgelist.txt", sep=" ", header=None)
whole_edgelist = pd.concat([test_edgelist, train_edgelist, valid_edgelist]).drop_duplicates().reset_index(drop=True)
whole_edgelist.to_csv("../myTests/"+dest+"/whole_edgelist.txt", index=False, header=False, sep=" ")
