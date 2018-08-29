function [tree, splits, is_leaf, clusters, timings, Ws, priorities, outliers] = hier_sym_NMF(X, C)
%%Hierarchical symmetric nonnegative matrix factorization
% Input parameters -------------------------------------------------------
% X: N*N affinity matrix
% C: The number of leaf nodes to be generated
%
% Output parameters ------------------------------------------------------
% The output parameters provide the details for reconstructing the tree of
% hierarchical clustering.
%
% For a binary tree with k leaf nodes, the total number of nodes (including leaf and non-leaf nodes)
% is 2*(k-1) plus the root node, because k-1 splits are performed and each split generates two new nodes.
%
% tree: A 2-by-(k-1) matrix that encodes the tree structure. The two entries in the i-th column are the numberings
%       of the two children of the node with numbering i.
%       The root node has numbering 0, with its two children always having numbering 1 and numbering 2.
%       Thus the root node is NOT included in the 'tree' variable.
% splits: An array of length k-1. It keeps track of the numberings of the nodes being split
%         from the 1st split to the (k-1)-th split. (The first entry is always 0.)
% is_leaf: An array of length 2*(k-1). A "1" at index i means that the node with numbering i is a leaf node
%          in the final tree generated, and "0" indicates non-leaf nodes in the final tree.
% clusters: A cell array of length 2*(k-1). The i-th element contains the subset of items
%           at the node with numbering i.
% timings: An array of length k-1.
% Ws: A cell array of length 2*(k-1).
%     Its i-th element is the cluster vector at the node with numbering i.
% priorities: An array of length 2*(k-1).
%             Its i-th element is the ncut at the node with numbering i.
%
% Tuning parameters ------------------------------------------------------
% trial_max: Number of trials allowed for removing outliers and splitting a node again
% unbalanced: A threshold to determine if one of the two clusters is an outlier set.
%             A smaller value means more tolerance for unbalance between two clusters.
%
% Ninghao Liu, Xiao Huang, Jundong Li, Xia Hu
% Aug 2018
% 
% The code is modified based on the HierNMF2 project https://github.com/dakuang/hiernmf2
%

% 1. Some parameters
trial_max = 3;
unbalanced = 0.05;

% 2. Hierarchical symmetric NMF
t0 = tic;

n = size(X, 1);

% initialize output terms
timings = zeros(1, C-1);
clusters = cell(1, 2*(C-1));
Ws = cell(1, 2*(C-1));
W_buffer = cell(1, 2*(C-1));
H_buffer = cell(1, 2*(C-1));
priorities = zeros(1, 2*(C-1));
is_leaf = -1 * ones(1, 2*(C-1));
tree = zeros(2, 2*(C-1));
splits = -1 * ones(1, C-1);
outliers = {};

% first split
term_subset = find(sum(X, 2) ~= 0);
W = rand(length(term_subset), 2);
H = rand(2, length(term_subset));
if length(term_subset) == n
	[W, H] = twinne_nmf_rank2(X, W, H);
else
	[W_tmp, H_tmp] = twinne_nmf_rank2(X(term_subset, term_subset), W, H);
	W = zeros(n, 2);
	W(term_subset, :) = W_tmp;  % deal with all-0 rows more efficiently
	clear W_tmp; clear H_tmp;
end

% main hierarchical iterations
index_cnt_current = 0;
for i = 1 : C-1
    i
	timings(i) = toc(t0);
    
	if i == 1                   % start from root
		split_node = 0;
		new_nodes = [1 2];      % two children, default
		min_priority = 1e308;
		split_subset = 1:n;
	else
		leaves = find(is_leaf == 1);
		temp_priority = priorities(leaves);
		min_priority = min(temp_priority(temp_priority > 0));
		[max_priority, split_node] = max(temp_priority);    % max priority
		if max_priority < 0
			fprintf('Cannot generate all %d leaf clusters\n', C);
			return;
		end
		split_node = leaves(split_node);
		is_leaf(split_node) = 0;
		W = W_buffer{split_node};
		H = H_buffer{split_node};
		split_subset = clusters{split_node};
		new_nodes = [index_cnt_current+1 index_cnt_current+2];
		tree(1, split_node) = new_nodes(1);
		tree(2, split_node) = new_nodes(2);
	end

	index_cnt_current = index_cnt_current + 2;
	[~, cluster_subset] = max(H);       % divide to two clusters
	clusters{new_nodes(1)} = split_subset(find(cluster_subset == 1));
	clusters{new_nodes(2)} = split_subset(find(cluster_subset == 2));
	Ws{new_nodes(1)} = W(:, 1);         % record term items for each new cluster, based on W
	Ws{new_nodes(2)} = W(:, 2);
	splits(i) = split_node;
	is_leaf(new_nodes) = 1;
    
    % trial split for the two new nodes, compute split priorities
	subset = clusters{new_nodes(1)};    % current set of nodes to split
	[subset, W_buffer_one, H_buffer_one, priority_one, outliers] = split_judgement(trial_max, unbalanced, min_priority, X, subset, W(:, 1), outliers);
	clusters{new_nodes(1)} = subset;
	W_buffer{new_nodes(1)} = W_buffer_one;
	H_buffer{new_nodes(1)} = H_buffer_one;
	priorities(new_nodes(1)) = priority_one;

	subset = clusters{new_nodes(2)};    % current set of nodes to split
	[subset, W_buffer_one, H_buffer_one, priority_one, outliers] = split_judgement(trial_max, unbalanced, min_priority, X, subset, W(:, 2), outliers);
	clusters{new_nodes(2)} = subset;
	W_buffer{new_nodes(2)} = W_buffer_one;
	H_buffer{new_nodes(2)} = H_buffer_one;
	priorities(new_nodes(2)) = priority_one;
    
    if i == 1
        priorities(new_nodes(1)) = 1;
        priorities(new_nodes(2)) = 1;
    end
end

%--------------------------------------------------------------------------------------------------------------------

function [subset, W_buffer_one, H_buffer_one, priority_one, outliers] = split_judgement(trial_max, unbalanced, min_priority, X, subset, W_parent, outliers)

trial = 0;
subset_backup = subset;
while trial < trial_max       % trial_allowance -> T in Alg3
	[cluster_subset, W_buffer_one, H_buffer_one, priority_one] = actual_split(X, subset, W_parent);
	if priority_one < 0
		break;
	end
	unique_cluster_subset = unique(cluster_subset);
	if length(unique_cluster_subset) ~= 2
		error('Invalid number of unique sub-clusters!');
	end
	length_cluster1 = length(find(cluster_subset == unique_cluster_subset(1)));
	length_cluster2 = length(find(cluster_subset == unique_cluster_subset(2)));
	if min(length_cluster1, length_cluster2) < unbalanced * length(cluster_subset)  % too imbalanced
		[~, idx_small] = min([length_cluster1, length_cluster2]);
		subset_small = find(cluster_subset == unique_cluster_subset(idx_small));
		subset_small = subset(subset_small);
        % try to split the small cluster to see if doable
		[cluster_subset_small, W_buffer_one_small, H_buffer_one_small, priority_one_small] = actual_split(X, subset_small, W_buffer_one(:, idx_small));
		if priority_one_small < min_priority    % shoud not split, just outliers
			trial = trial + 1;
			if trial < trial_max
				disp(['Drop ', num2str(length(subset_small)), ' documents ...']);
				subset = setdiff(subset, subset_small);     % remove outliers, then try split the major cluster
                outliers{end+1} = subset_small;
			end
		else
			break;
		end
	else
		break;
	end
end

if trial == trial_max
	disp(['Recycle ', num2str(length(subset_backup) - length(subset)), ' documents ...']);
	subset = subset_backup;
	W_buffer_one = zeros(length(subset), 2);%%
	H_buffer_one = zeros(2, length(subset));
	priority_one = -2;
end

%--------------------------------------------------------------------------------------------------------------------

function [cluster_subset, W_buffer_one, H_buffer_one, priority_one] = actual_split(X, subset, W_parent)
[m, n] = size(X);
if length(subset) <= 3
	cluster_subset = ones(1, length(subset));
	W_buffer_one = zeros(m, 2);
	H_buffer_one = zeros(2, length(subset));
	priority_one = -1;
else
    % exclude all zero lines
	term_subset = find(sum(X(:, subset), 2) ~= 0);
    term_subset = intersect(term_subset, subset);
    % NMF with the non-all-zero patch
	X_subset = X(term_subset, term_subset);
	W = rand(length(term_subset), 2);
	H = rand(2, length(term_subset));
	[W, H] = twinne_nmf_rank2(X_subset, W, H);
	[~, cluster_subset] = max(H);
    % recover, W records full size n, H only records current cluster size
	W_buffer_one = zeros(m, 2);
	W_buffer_one(term_subset, :) = W;
	H_buffer_one = W_buffer_one(subset, :)';
	if length(unique(cluster_subset)) > 1
        priority_one = compute_ngd(X, subset, W_buffer_one);
	else
		priority_one = -1;
	end
end


function priority = compute_ngd(X, subset, W_child)
%%Partition priority, the smaller the better.
[~, cluster_subset] = max(W_child');
unique_cluster_subset = unique(cluster_subset);
if length(unique_cluster_subset) ~= 2
    error('Invalid number of unique sub-clusters!');
end
cluster_subset_1 = find(cluster_subset == unique_cluster_subset(1));
cluster_subset_2 = find(cluster_subset == unique_cluster_subset(2));
cluster_subset_1 = intersect(cluster_subset_1, subset);
cluster_subset_2 = intersect(cluster_subset_2, subset);

% compute in/out degrees, A -> (B1, B2)
degree_all_B1 = sum(sum(X(cluster_subset_1, :)));
degree_in_B1 = sum(sum(X(cluster_subset_1, cluster_subset_1)));
degree_all_B2 = sum(sum(X(cluster_subset_2, :)));
degree_in_B2 = sum(sum(X(cluster_subset_2, cluster_subset_2)));
ngd = (degree_all_B1 - degree_in_B1)/degree_all_B1 + ...
    (degree_all_B2 - degree_in_B2)/degree_all_B2;

priority = 1 - ngd;