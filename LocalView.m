function [Beta_est, Y] = LocalView(X, C, name_net, name_embd)
%% Local-view interpretation
% Input parameters -------------------------------------------------------
% C: Number of leaf clusters to be considered.
%    A small value means pruning the tree.
%
% Output parameters ------------------------------------------------------
% Beta_est: The weight matrix (U in the paper) indicating the importance 
%           of attributes within each cluster in the tree.
% X: Attribute matrix (N nodes * M attributes)
% Y: Label matrix, each column of which corresponds to one node in the tree
%    (N nodes * k classes)
%
% Tuning parameters ------------------------------------------------------
% lambda: regularization coefficient
% iters: number of iterations
% ratio_pair_per_iter: controls batch size of each iteration
% delta: step size for weight update
% ratio_shrink: decease delta as iteration proceed
%
% Ninghao Liu, Xiao Huang, Jundong Li, Xia Hu
% Aug 2018
%

lambda = 0.4;
iters = 1500;
ratio_pair_per_iter = 1;
delta = 0.001;
ratio_shrink = 0.9998;

% Load data
A = load(strcat(name_net, name_embd, '_G.mat'), 'G');
A = A.G;
num_nodes = size(A, 1);

% Load global-view results
clusters = load(strcat(name_net, name_embd, '_clusters.mat'), 'clusters');
is_leaf = load(strcat(name_net, name_embd, '_isleaf.mat'), 'is_leaf');
tree = load(strcat(name_net, name_embd, '_tree.mat'), 'tree');
clusters = clusters.clusters;
is_leaf = is_leaf.is_leaf;
tree = tree.tree;
size_layer1 = 2;

Y = Clusters2Y(clusters, num_nodes, C);    % labels

wvs = ComputeWV(tree, is_leaf, size_layer1);
wvs = wvs(1:(2*C - 2));

% Get the weight matrix of local-view interpretation
Beta_est = Multitask(X, Y, A, wvs, tree, clusters, is_leaf, lambda, iters, ratio_pair_per_iter, delta, ratio_shrink);


function depths = DepthOfTreeNodes(tree)
%% Find the depth of each tree node
depths = zeros(1, size(tree, 2));
depths([1,2]) = 1;
cursor = 1;
tree_traverse = [1,2];
while cursor <= length(tree_traverse)
    children_new = tree(:, tree_traverse(cursor))';
    children_new = children_new(children_new ~= 0);
    tree_traverse = [tree_traverse, children_new];
    if length(children_new) > 0
        depths(children_new(1)) = depths(tree_traverse(cursor)) + 1;
        depths(children_new(2)) = depths(tree_traverse(cursor)) + 1;
    end
    cursor = cursor + 1;
end

function Y_sub = Clusters2Y(clusters, num_nodes, C)
%% Obtain multitask labels Y from clustering results

num_clusters = size(clusters, 2);
Y = zeros(num_nodes, num_clusters);     % initialize Y
for c = 1:num_clusters
    Y(clusters{c}, c) = 1;      % full matrix
end
Y_sub = Y(:, 1:(2*C-2));

function Beta = Multitask(X, Y, A, wvs, tree, clusters, is_leaf, lambda, iters, ratio_pair_per_iter, delta, ratio_shrink)
%% Local interpretation as multitask classification

num_nodes = size(X, 1);
num_attrs = size(X, 2);
num_tasks = size(Y, 2);

num_nz = nnz(A)
size_batch = 0;
for c = 1:num_tasks
    size_batch = size_batch + length(clusters{c});
end
size_batch = ceil(size_batch * ratio_pair_per_iter);

% The times a regularizer appears depend on the depth of the tree node
depths = DepthOfTreeNodes(tree);

% Initialize parameters
Beta = rand(num_attrs, num_tasks);      % matrix U in the paper
Djv = zeros(num_attrs, num_tasks);      % matrix g in the paper

% Optimization
for t = 1:iters
    % update djv
    sum_w_beta = realmin;
    for j = 1:num_attrs
        for v = 1:num_tasks
            beta_jv = GetBetaJV(Beta, j, v, tree, num_tasks);
            Djv(j, v) = wvs(v) * norm(beta_jv);
            sum_w_beta = sum_w_beta + Djv(j, v);
        end
    end
    Djv = Djv/sum_w_beta;
    Djv = max(0.05/num_attrs/num_tasks, Djv);

    % Compute Beta gradient
    Beta_grad = zeros(size(Beta));
    % iterate over each task (i.e., tree node)
    for c = 1:num_tasks
        % random sampling edges within the task
        size_batch_c = min(ceil(length(clusters{c})*ratio_pair_per_iter), 2000);
        Ac = sparse(size(A,1), size(A,2));
        Ac(clusters{c}, clusters{c}) = A(clusters{c}, clusters{c});
        [rcand, ccand, vcand] = find(Ac);
        ids = randsample(1:length(rcand), size_batch_c, true);
        rowc = rcand(ids);
        colc = ccand(ids);
        valc = vcand(ids);
        
        % SGD w.r.t. edge samples
        for n = 1:size_batch_c
            rn = rowc(n); cn = colc(n); vn = valc(n);
            exp_const = exp(-vn * X(rn,:) .* Beta(:, c)' * X(cn,:)');
            Beta_grad(:, c) = Beta_grad(:, c) + exp_const/(1+exp_const) * (-vn) * X(rn,:)' .* X(cn,:)';     % gradient
            
            % 3. contrast from other clusters
            for cc = 1:num_tasks
                if cc ~= c && is_leaf(cc)
                    exp_const = exp(vn * X(rn,:) .* Beta(:, cc)' * X(cn,:)');
                    Beta_grad(:, cc) = Beta_grad(:, cc) + exp_const/(1+exp_const) * vn * X(rn,:)' .* X(cn,:)'/num_tasks;
                end
            end
        end
        
        Beta_grad_c_regu = depths(c) * 2*lambda*size_batch/num_nz * wvs(c).^2* Beta(:, c)./Djv(:, c);       % regularization
        Beta_grad(:, c) = Beta_grad(:, c) + Beta_grad_c_regu;
    end
    
    % Update Beta
    Beta = Beta - delta * Beta_grad;
    
    delta = delta * ratio_shrink;
    if mod(t, 100) == 0 || t < 15
        t
        error = ComputeError(X, Y, A, Beta, clusters);
        sprintf('%f', error)
        save(strcat('test', '_Beta.mat'), 'Beta');
    end
end

function beta_jv = GetBetaJV(Beta, j, v, tree, num_tasks)
%% get output tree nodes in the subtree rooted at v
children_v = [v];
cursor = 1;
while cursor <= length(children_v)
    children_new = tree(:, children_v(cursor))';
    children_new = children_new(children_new ~= 0);
    children_v = [children_v, children_new];
    cursor = cursor + 1;
end
children_v = children_v(children_v <= num_tasks);
beta_jv = Beta(j, children_v);

function error = ComputeError(X, Y, A, Beta, clusters)
num_tasks = size(Y, 2);
error = 0;
for c = 1:num_tasks
    Ac = A(clusters{c}, clusters{c});
    XXc = X(clusters{c}, :) * diag(Beta(:, c)) * X(clusters{c}, :)';
    error = error + sum(sum(log(1 + exp(-Ac .* XXc))));
end

function wvs = ComputeWV(tree, is_leaf, size_layer1)
%% Compute wv for each tree node v (no root)
% wvs: an array of wv, has the same length as 'ls_leaf', and cols of 'tree'

num_treenode = length(is_leaf);
wvs = ones(1, num_treenode);
height = zeros(1, num_treenode);
for curnode = 1:num_treenode
    leftchild = tree(1, curnode);
    rightchild = tree(2, curnode);
    height(curnode) = max(HeightSubtree(leftchild, tree), HeightSubtree(rightchild, tree)) + 1;
end
height_root = max(height) + 1;
hv_p = height/height_root;      % relative subtree-height of cluster v

gv = 1 - hv_p;
sv = hv_p;

for curnode = 1:num_treenode
    if is_leaf(curnode) ~= 1
            wvs(curnode) = wvs(curnode) * gv(curnode);
    end

    v = curnode;
    u = 1;
    while ~ismember(v, 1:size_layer1)
        if ismember(v, tree(:, u))
            wvs(curnode) = wvs(curnode) * sv(u);
            v = u;
            u = 0;
        end
        u = u + 1;
    end
end

function height = HeightSubtree(v, tree)
%% Compute the height of subtree rooted at v, used in ComputeWV()
if v == 0
    height = 0;
else
    leftchild = tree(1, v);
    rightchild = tree(2, v);
    height = max(HeightSubtree(leftchild, tree), HeightSubtree(rightchild, tree)) + 1;
end

%%