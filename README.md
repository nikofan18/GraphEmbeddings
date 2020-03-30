 # GraphEmbeddings

It's an experiment for graph embedding models based on these 2 papers:
https://arxiv.org/pdf/1703.08098.pdf
https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf

##  Quick Start

1) git clone this repo https://github.com/nikofan18/GraphEmbeddings
2) Simple run GraphEmbeddings/experiment/main.py

This way you are going to run the experiment for the default dataset FB15K.
You could choose an other dataset changing the value in line 9 of GraphEmbeddings/experiment/main.py.


### Pipeline followed

1) git clone the node2vec for generating the embeddings https://github.com/aditya-grover/node2vec
2) The node2vec requires edgelist as input, but the datasets I downloaded are not in acceptable format.
   By running the GraphEmbeddings/preproccesing/edgelist.py in our repo, we have the default dataset (FB15K) in acceptable format.
   You can change te line 5 and line 6 of GraphEmbeddings/preproccesing/edgelist.py for another dataset
3) You can find in GraphEmbeddings/myTests the corect formated datasets and move them to node2vec tool in order to produce the embeddings.
   In order to produce the embeddings you could run the following command in node2vec repo:

   python src/main.py --input graph/FB15K_PROCESSED/whole.edgelist --output emb/FB15K_EMBEDDINGS/whole.emd
4) Move the embeddings from emb/FB15K_EMBEDDINGS/ to GraphEmbeddings/experiment/embeddings
5) Finally, run the GraphEmbeddings/experiment/main.py for the experiment