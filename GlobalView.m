function [G, tree, splits, is_leaf, clusters, timings, Ws, priorities, outliers] = GlobalView(Emb, C, name_net, name_embd)
%% Global-view interpretation
% Input parameters -------------------------------------------------------
% Emb: Embedding matrix (D dimension * N data points)
% C: The number of leaf nodes to be generated
%
% Output parameters ------------------------------------------------------
% Details of output parameters are included in the function hier_sym_NMF()
%
% Tuning parameters ------------------------------------------------------
% sigma: Edge weight decay
% num_nbr_ratio: The number of edges for each node in the affinity graph is log2(N)*num_nbr_ratio
%
% Ninghao Liu, Xiao Huang, Jundong Li, Xia Hu
% Aug 2018
%

sigma = 0.5;
num_nbr_ratio = 2;

tic

embs = Emb(2:end, :);
size(Emb)

% Affinity graph
disp('Build Graph')
G = Emb2Graph(embs, sigma, num_nbr_ratio);
save(strcat(name_net, name_embd, '_G.mat'), 'G')

% Hierarchical clustering
disp('Clustering')
[tree, splits, is_leaf, clusters, timings, Ws, priorities, outliers] = hier_sym_NMF(G, C);

% Save the tree structure to be used for local-view interpretation
save(strcat(name_net, name_embd, '_clusters.mat'), 'clusters')
save(strcat(name_net, name_embd, '_isleaf.mat'), 'is_leaf')
save(strcat(name_net, name_embd, '_tree.mat'), 'tree')

toc

function G = Emb2Graph(embs, sigma, num_nbr_ratio)
%% Construct affinity graph
% KNN graph
N = size(embs, 2);
num_nbr = ceil(log2(N)) * num_nbr_ratio;
[ids, ds] = knnsearch(embs', embs', 'K', num_nbr+1);
ids = ids(:, 2:end);
ds = ds(:, 2:end);
mean(mean(ds, 2))

% Graph edge weights
Gijs = exp(-ds.^2/sigma);
Gijs = reshape(Gijs', size(Gijs,1)*size(Gijs,2), 1);
j_s = reshape(ids', size(ids,1)*size(ids,2), 1);
i_s = ones(N*num_nbr, 1);
for i = 1:N
    i_s((i-1)*num_nbr+1:i*num_nbr) = i;
end
i_s = double(i_s); j_s = double(j_s); Gijs = double(Gijs);
G = sparse([i_s;j_s], [j_s;i_s], [Gijs; Gijs]);
%G = G + max(max(G))*eye(size(G, 1));%%
%G = diag(sum(G).^(-0.5)) * G * diag(sum(G).^(-0.5));%%
%%