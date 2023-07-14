# source-code-dsaa-2023

DSAA 2023 Competition

Link prediction is an important task applied in networks. Given a pair of nodes ( u, v ) we need to predict if the edge between nodes u and v will be present or not. Link prediction is strongly related to recommendations, network reconstruction and network evolution. In this competition, we focus on the link prediction task applied to Wikipedia articles. In particular, given a sparsified subgraph of the Wikipedia network, we need to predict if a link exists between two Wikipedia pages u and v.

More specifically, you are given a ground-truth file which contains pairs of nodes corresponding to positive of negative samples. For example, if an edge exists between two nodes then the corresponding label is set to 1. Otherwise, the label is 0. From the original file, 20% of the information has been removed. This includes positive pairs (the edge exists) as well as negative pairs (the edge does not exist). Your mission is to correctly identify the positive and negative pairs.

abtract
The Data Science and Advanced Analytics (DSAA) 2023 competition \cite{dsaa-2023-competition} focuses on proposing link prediction methods to solve challenges about network-like data structure such as network reconstruction, network development, etc from articles on Wikipedia. In this challenge, our "UIT Dark Cow" team propose the Mutual Attention Transformer (MAT) method to predict if there is a link between two Wikipedia pages. Our method achieved the 5th and 4th position on the leaderboard for the public test and private test, respectively.
