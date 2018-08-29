# UnderstandingNetworkEmbedding

This project implements the proposed framework for running the experiments in the paper: <br>

**On Interpretation of Network Embedding via Taxonomy Induction**<br>
Ninghao Liu, Xiao Huang, Jundong Li, Xia Hu<br>
Proceedings of KDD'18, London, UK <br>

**Please cite our paper if you use the codes. Thanks!**

## Example to run the codes in MATLAB:
```
[Beta_est, G, tree, splits, is_leaf, clusters, priorities, Y] = UnderstandingNetworkEmbedding(Emb, X, C, name_net, name_embd)
```
Some input variable and parameters are introduced as below:
- Emb: The embedding matrix (D dimension * N data points)
- X: The attribute matrix (N data points * M attributes)
- C: The number of leaf nodes to be generated in the taxonomy
- name_net: A string about the name of the dataset
- name_embd: A string about the name of the embedding model
