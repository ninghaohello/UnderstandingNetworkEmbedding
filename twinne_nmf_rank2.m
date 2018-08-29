function [W, H] = twinne_nmf_rank2(A, Winit, Hinit)
% Input parameters -------------------------------------------------------
% A: n*n submatrix of the affinity matrix
% Winit: n*2 initialization of W
% Hinit: 2*n initialization of H
%
% Output parameters ------------------------------------------------------
% W, H: Factor matrices after factorizing A
%
% Tuning parameters ------------------------------------------------------
% tol_WH: Tolerance parameter for stopping criterion
% maxiter: Maximum number of iterations
% alpha: Forces W and H to be close, increases as iterations proceed
%
% Ninghao Liu, Xiao Huang, Jundong Li, Xia Hu
% Aug 2018
%

tol_WH = 4e-4;
maxiter = 100;
alpha = 0.1;
vec_norm = 2.0;
normW = true;

W = Winit;
H = Hinit;
if size(W, 2) ~= 2
	error('Wrong size of W!');
end
if size(H, 1) ~= 2
	error('Wrong size of H!')
end

for iter = 1 : maxiter
    iter
    % reformulate
	W = H';
    W_comb = [W; sqrt(alpha)*eye(2)];
    A_comb = [A; sqrt(alpha)*W'];
    
    % nonneg lsq column by column
    H_new = H;
    for j = 1:size(A_comb,2)
        H_new(:, j) = lsqnonneg(W_comb, A_comb(:, j));
    end
    H = H_new;
    
    % break if W and H are close enough
    err_wh = norm(W-H', 'fro')
    err_h = norm(H, 'fro')
    if err_wh/err_h < tol_WH
        W = H';
        break
    end
    alpha = alpha * 1.1;
    
    if mod(iter, 7) == 0
        err_wha = norm(W*H - A, 'fro')
    end
end
W = H';

if vec_norm ~= 0
	if normW
        	norms = sum(W.^vec_norm) .^ (1/vec_norm);
	        W = bsxfun(@rdivide, W, norms);
        	H = bsxfun(@times, H, norms');
	else    
        	norms = sum(H.^vec_norm, 2) .^ (1/vec_norm);
	        W = bsxfun(@times, W, norms');
        	H = bsxfun(@rdivide, H, norms);
	end
end
