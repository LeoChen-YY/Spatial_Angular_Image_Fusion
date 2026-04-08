%% CT Sinogram Fusion using ADMM with TGV Regularization & Grid Search
% 
% Optimization Problem:
% min_s  1/2 * ||s_S - P_S * s||^2_F + alpha/2 * ||s_A - P_A * s||^2_F + TGV_{gamma}(s)
%
% Where:
% s: Target sinogram to be reconstructed (vectorized)
% P_S: Spatial blurring/subsampling operator
% P_A: Angular masking operator
% TGV: Total Generalized Variation, defined as:
%      min_v  gamma1 * ||Ds - v||_1 + gamma0 * ||Ev||_1
%      (D is the first-order derivative, E is the symmetrized derivative)
%
% This formulation reduces "staircase" artifacts compared to standard TV.

clear; clc; close all;

%% 1. Parameter Settings and Data Generation
img_size = 256;     
theta = 0:1:179;    
filter = 'Ram-Lak'; 

% --- Grid Search Range Settings ---
% (Tested with 2 values per parameter to maintain reasonable computation time)
gamma0_list = [0.01, 0.1]; 
gamma1_list = [0.01, 0.1];
mu0_list    = [0.1, 0.5];     
mu1_list    = [0.1, 0.5];     
alpha_list  = [0.1, 1];      
max_iter    = 50;           % Iterations during Grid Search
pcg_tol = 1e-6;
pcg_maxIt = 50;

% Generate Ground Truth Data
X_true = phantom(img_size);
sinogram_true = radon(X_true, theta);
[M, N] = size(sinogram_true);
s_true = sinogram_true(:); 
mn = length(s_true);

% 2.1 Construct P_S_sub (Spatial Blurring Operator)
K = 2; 
P_S_sub = sparse(M, M);
for i = 1:K:M
    idx = i : min(i + K - 1, M);
    P_S_sub(idx, idx) = 1/length(idx);
end

% 2.2 Construct P_A (Angular Mask)
mask_A = zeros(M, N);
mask_A(:, 1:2:end) = 1; % Sample every other angle
P_A_vec = sparse(mask_A(:)); 

% 2.3 Generate Observations 
% s_S: Spatially blurred / s_A: Angularly sparse but sharp
s_S_mat = P_S_sub * sinogram_true; 
s_S = s_S_mat(:);
s_A = P_A_vec .* s_true;

% 2.4 Initialize Operators
[Dx, Dy] = get_diff_ops(M, N);
D = [Dx; Dy]; 

% 2.5.0 Prepare coordinates for interpolation ---
% N is total angles, idx_sampled is the indices of available angles
idx_all = 1:N;
idx_sampled = find(any(reshape(full(P_A_vec), [M, N]), 1)); 

% 2.5.1 Linear Interpolation for Initialization ---
s_A_mat = reshape(full(s_A), [M, N]);
s_interp_mat = zeros(M, N);
for i = 1:M
    % Interpolate missing angles per detector row
    s_interp_mat(i, :) = interp1(idx_sampled, s_A_mat(i, idx_sampled), idx_all, 'linear', 'extrap');
end

% 2.5.2 Set Interpolated Result as Initial Guess for s_k ---
s_init = s_interp_mat(:);

%% 2. Grid Search Execution
results = [];
total_configs = length(gamma0_list) * length(gamma1_list) * length(mu0_list) * length(mu1_list) * length(alpha_list);
count = 0;
best_psnr = -inf;
best_params = [];

fprintf('Starting Grid Search (TGV) with %d configurations...\n', total_configs);

for g0 = gamma0_list
    for g1 = gamma1_list
        for m0 = mu0_list
            for m1 = mu1_list
                for al = alpha_list
                    count = count + 1;
                    
                    % Variables Initialization
                    s_k = s_init;
                    v_k = zeros(2*mn, 1);
                    z1_k = zeros(2*mn, 1);
                    z2_k = zeros(3*mn, 1);
                    u1_k = zeros(2*mn, 1);
                    u2_k = zeros(3*mn, 1);

                    % Operators for current parameter set
                    A_s_op = @(s) apply_LHS_s(s, P_S_sub, P_A_vec, D, m0, al, M, N);
                    A_v_op = @(v) apply_LHS_v(v, Dx, Dy, m0, m1, M, N);

                    % ADMM Iterations
                    for k = 1:max_iter
                        % z-update (Shrinkage / Soft-thresholding)
                        tmp1 = D*s_k - v_k + u1_k;
                        z1_k = sign(tmp1) .* max(abs(tmp1) - g0/m0, 0);

                        Ev_k = apply_E(v_k, Dx, Dy, M, N);
                        tmp2 = Ev_k + u2_k;
                        z2_k = sign(tmp2) .* max(abs(tmp2) - g1/m1, 0);

                        % s-update (Least Squares via PCG)
                        s_S_mat_in = reshape(s_S, [M, N]);
                        term1 = P_S_sub' * s_S_mat_in; % Adjoint of spatial blur
                        RHS_s = term1(:) + al * s_A + m0 * D' * (z1_k + v_k - u1_k);
                        [s_k, ~] = pcg(A_s_op, RHS_s, pcg_tol, pcg_maxIt, [], [], s_k);

                        % v-update (Least Squares via PCG)
                        Et_term = apply_Et(z2_k - u2_k, Dx, Dy, M, N);
                        RHS_v = m0 * (D*s_k - z1_k + u1_k) + m1 * Et_term;
                        [v_k, ~] = pcg(A_v_op, RHS_v, pcg_tol, pcg_maxIt, [], [], v_k);

                        % Dual variables u-update
                        u1_k = u1_k + (D*s_k - v_k - z1_k);
                        u2_k = u2_k + (apply_E(v_k, Dx, Dy, M, N) - z2_k);
                    end

                    % Use Remixing PSNR as performance metric
                    s_remix = s_k; 
                    s_remix(logical(P_A_vec)) = s_true(logical(P_A_vec));
                    X_remix = iradon(full(reshape(s_remix, [M, N])), theta, 'linear', filter, 1, img_size);
                    curr_psnr = psnr(X_remix, X_true);
                    
                    results = [results; g0, g1, m0, m1, al, curr_psnr];
                    
                    if curr_psnr > best_psnr
                        best_psnr = curr_psnr;
                        best_params = [g0, g1, m0, m1, al];
                    end
                    
                    fprintf('[%d/%d] g0:%.2f, g1:%.2f, m0:%.1f, m1:%.1f, al:%.1f | Remix Image PSNR: %.4f\n', ...
                        count, total_configs, g0, g1, m0, m1, al, curr_psnr);
                end
            end
        end
    end
end

%% 3. Final Reconstruction with Best Parameters
bg0 = best_params(1); bg1 = best_params(2); 
bm0 = best_params(3); bm1 = best_params(4); bal = best_params(5);

fprintf('\nFinal Reconstruction using Best Parameters (g0=%.2f, g1=%.2f, m0=%.1f, m1=%.1f, al=%.1f)...\n', bg0, bg1, bm0, bm1, bal);

s_k = s_init; v_k = zeros(2*mn, 1);
z1_k = zeros(2*mn, 1); z2_k = zeros(3*mn, 1);
u1_k = zeros(2*mn, 1); u2_k = zeros(3*mn, 1);

A_s_best = @(s) apply_LHS_s(s, P_S_sub, P_A_vec, D, bm0, bal, M, N);
A_v_best = @(v) apply_LHS_v(v, Dx, Dy, bm0, bm1, M, N);

for k = 1:max_iter * 2 % Double iterations for final result
    tmp1 = D*s_k - v_k + u1_k;
    z1_k = sign(tmp1) .* max(abs(tmp1) - bg0/bm0, 0);
    tmp2 = apply_E(v_k, Dx, Dy, M, N) + u2_k;
    z2_k = sign(tmp2) .* max(abs(tmp2) - bg1/bm1, 0);
    
    s_S_mat_in = reshape(s_S, [M, N]);
    term1 = P_S_sub' * s_S_mat_in;
    RHS_s = term1(:) + bal * s_A + bm0 * D' * (z1_k + v_k - u1_k);
    [s_k, ~] = pcg(A_s_best, RHS_s, pcg_tol, pcg_maxIt, [], [], s_k);
    
    RHS_v = bm0 * (D*s_k - z1_k + u1_k) + bm1 * apply_Et(z2_k - u2_k, Dx, Dy, M, N);
    [v_k, ~] = pcg(A_v_best, RHS_v, pcg_tol, pcg_maxIt, [], [], v_k);
    
    u1_k = u1_k + (D*s_k - v_k - z1_k);
    u2_k = u2_k + (apply_E(v_k, Dx, Dy, M, N) - z2_k);
end

% --- Generate Remix Result ---
S_final = reshape(s_k, [M, N]);
S_remix = S_final;
S_remix(mask_A == 1) = sinogram_true(mask_A == 1); % Replace with true samples

% --- Transform to Image Domain ---
img_final = iradon(S_final, theta, 'linear', filter, 1, img_size);
img_remix = iradon(S_remix, theta, 'linear', filter, 1, img_size);

% --- Calculate Metrics ---
m_final = eval_metrics(X_true, img_final);
m_remix = eval_metrics(X_true, img_remix);

fprintf('\nFinal Performance Comparison:\n');
fprintf('Final -> RMSE: %.4f, PSNR: %.2f dB, SSIM: %.4f\n', m_final.RMSE, m_final.PSNR, m_final.SSIM);
fprintf('Remix -> RMSE: %.4f, PSNR: %.2f dB, SSIM: %.4f\n', m_remix.RMSE, m_remix.PSNR, m_remix.SSIM);

%% 4. Plotting Results
% Figure 1: Sinogram Domain
figure('Name', 'Best Parameters: Sinogram Domain');
subplot(1,3,1); imshow(sinogram_true, []); title('Original Sinogram');
subplot(1,3,2); imshow(S_final, []); title('Final (Reconstructed)');
subplot(1,3,3); imshow(S_remix, []); title('Remix (Fused)');

% Figure 2: Image Domain
figure('Name', 'Best Parameters: Image Domain');
subplot(1,3,1); imshow(X_true, []); title('Ground Truth');
subplot(1,3,2); imshow(img_final, []); 
title(sprintf('Final (PSNR: %.2f dB)', m_final.PSNR));
subplot(1,3,3); imshow(img_remix, []); 
title(sprintf('Remix (PSNR: %.2f dB)', m_remix.PSNR));

%% --- Helper Functions ---

function m = eval_metrics(gt, recon)
    m.RMSE = norm(gt(:) - recon(:)) / norm(gt(:));
    m.PSNR = psnr(recon, gt);
    m.SSIM = ssim(recon, gt);
end

% LHS Operator for s-update: (P_S'P_S + alpha*P_A'P_A + mu0*D'D)
function y = apply_LHS_s(s, P_S_sub, P_A_vec, D, mu0, alpha, M, N)
    S_mat = reshape(s, [M, N]);
    PsPsS = P_S_sub' * (P_S_sub * S_mat);
    y = PsPsS(:) + alpha * (P_A_vec.^2 .* s) + mu0 * (D' * (D * s));
end

% LHS Operator for v-update: (mu0 * I + mu1 * E'E)
function y = apply_LHS_v(v, Dx, Dy, mu0, mu1, M, N)
    Ev = apply_E(v, Dx, Dy, M, N);
    EtEv = apply_Et(Ev, Dx, Dy, M, N);
    y = mu0 * v + mu1 * EtEv;
end

% Symmetrized derivative operator E
function Ev = apply_E(v, Dx, Dy, M, N)
    MN = M*N;
    vx = v(1:MN); vy = v(MN+1:end);
    Ev = [Dx*vx; Dy*vy; 0.5*(Dy*vx + Dx*vy)];
end

% Adjoint symmetrized derivative operator E'
function Etz = apply_Et(z, Dx, Dy, M, N)
    MN = M*N;
    z11 = z(1:MN); z22 = z(MN+1:2*MN); z12 = z(2*MN+1:end);
    vx = Dx'*z11 + 0.5*Dy'*z12;
    vy = Dy'*z22 + 0.5*Dx'*z12;
    Etz = [vx; vy];
end

% Generate forward difference operators
function [Dx, Dy] = get_diff_ops(M, N)
    e = ones(M*N, 1);
    Dx = spdiags([-e e], [0 1], M*N, M*N); 
    Dy = spdiags([-e e], [0 M], M*N, M*N);
end