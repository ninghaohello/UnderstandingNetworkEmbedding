function [Beta_est, G, tree, splits, is_leaf, clusters, priorities, Y] = UnderstandingNetworkEmbedding(Emb, X, C, name_net, name_embd)
% The input data sources include:
% 1. An embedding matrix (D dimension * N data points)
% 2. An attribute matrix (N data points * M attributes)
%
% The meaning of variables and parameters are introduced in each function.
%
% Ninghao Liu, Xiao Huang, Jundong Li, Xia Hu
% Aug 2018
%

[G, tree, splits, is_leaf, clusters, timings, Ws, priorities, outliers] = GlobalView(Emb, C, name_net, name_embd);

[Beta_est, Y] = LocalView(X, C, name_net, name_embd);