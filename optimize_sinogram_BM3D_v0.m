%% CT Sinogram Fusion using PnP-ADMM (BM3D) with Grid Search
% 
% Optimization Problem (Plug-and-Play ADMM):
% This code performs Sinogram-domain fusion by solving:
%
% min_{s}  1/2 * ||P_S * s - s_S||_2^2 + alpha/2 * ||P_A * s - s_A||_2^2 + eta * Phi(s)
%
% Where:
% s: The fused target sinogram (vectorized)
% P_S: Spatial subsampling/blurring operator
% P_A: Angular masking operator
% Phi(s): Implicit regularizer replaced by the BM3D denoiser (Plug-and-Play)
% alpha: Weighting factor between the spatial and angular data fidelity
% eta: Regularization parameter (linked to denoiser strength)
% mu: ADMM penalty parameter (augmented Lagrangian parameter)

clear; clc; close all;
addpath('bm3d_matlab_package_4.0.3\bm3d');

%% 1. Parameter Settings and Data Generation
img_size = 256;     
theta = 0:1:179;    

% --- Radon Transform Settings ---
filter = 'Ram-Lak'; % Reconstruction filter

% --- Grid Search Range Settings ---
eta_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]; % Regularization weight
mu_list  = [0.1, 0.5, 1, 5];                     % ADMM step size / penalty
alpha_list = [0.01, 0.1, 1];                     % Weight for angular observation
max_iter = 30;                                   % Iterations (kept low for search speed)

% --- Linear Solver (PCG) Settings ---
pcg_tol = 1e-6;
pcg_maxIt = 50;

% Generate Ground Truth Sinogram
X_true = phantom(img_size);
sinogram_true = radon(X_true, theta);
[M, N] = size(sinogram_true);
s_true = sinogram_true(:); 
mn = length(s_true);

% 2.1 Construct P_S_sub (Spatial Blurring/Averaging Operator)
K = 2; 
P_S_sub = sparse(M, M);
for i = 1:K:M
    idx = i : min(i + K - 1, M);
    P_S_sub(idx, idx) = 1/length(idx);
end

% 2.2 Construct P_A (Angular Mask)
mask_A = zeros(M, N);
mask_A(:, 1:2:end) = 1; % Subsample angles by half
P_A_vec = sparse(mask_A(:)); 

% 2.3 Generate Observations s_S and s_A
s_S_mat = P_S_sub * sinogram_true; 
s_S = s_S_mat(:);
s_A = P_A_vec .* s_true;

% 2.4 Operator Definition
[D, ~] = forward_diff_operators(M, N); 

%% 2. Grid Search Execution
results = []; % Table to store performance metrics

fprintf('Starting Grid Search...\n');
fprintf('%-10s %-10s %-10s %-10s %-10s %-10s\n', 'eta', 'mu', 'alpha', 'RMSE', 'PSNR', 'SSIM');

for e_idx = 1:length(eta_list)
    for m_idx = 1:length(mu_list)
        for a_idx = 1:length(alpha_list)
            eta = eta_list(e_idx);
            mu = mu_list(m_idx);
            alpha = alpha_list(a_idx);

            % --- ADMM Initialization ---
            s_k = zeros(mn, 1);  
            z_k = zeros(mn, 1);  
            u_k = zeros(mn, 1);  
            A_op = @(x) apply_LHS(x, P_S_sub, P_A_vec, mu, alpha, M, N);

            for k = 1:max_iter
                % --- Step 1: s-update (Least Squares Data Fidelity) ---
                % Solves: (P_S'P_S + alpha*P_A'P_A + mu*I)s = P_S's_S + alpha*P_A's_A + mu(z-u)
                s_S_mat_in = reshape(s_S, [M, N]);
                term1 = P_S_sub' * s_S_mat_in; % P_S' * s_S
                RHS = term1(:) + alpha * s_A + mu * (z_k - u_k);
                [s_k, ~] = pcg(A_op, RHS, pcg_tol, pcg_maxIt, [], [], s_k);

                % --- Step 2: z-update (Plug-and-Play BM3D Denoiser) ---
                sigma_denoise = sqrt(eta / mu);
                v_k = reshape(s_k + u_k, [M, N]);
                % Apply BM3D as the proximal operator
                z_mat = BM3D(v_k, sigma_denoise);
                z_k = z_mat(:);

                % --- Step 3: u-update (Dual Variable Update) ---
                u_k = u_k + (s_k - z_k);
            end

            % --- Sinogram Remixing and Final Reconstruction ---
            S_final = reshape(s_k, [M, N]);
            S_remix = S_final;
            % Replace reconstructed values with known measurements where available
            S_remix(mask_A == 1) = sinogram_true(mask_A == 1);
            img_remix = iradon(S_remix, theta, 'linear', filter, 1, img_size);

            % --- Calculate Metrics ---
            curr_rmse = sqrt(mean((X_true(:) - img_remix(:)).^2));
            curr_psnr = psnr(img_remix, X_true);
            curr_ssim = ssim(img_remix, X_true);

            % Record results
            results = [results; eta, mu, alpha, curr_rmse, curr_psnr, curr_ssim];
            fprintf('%-10.3f %-10.3f %-10.3f %-10.4f %-10.2f %-10.4f\n', eta, mu, alpha, curr_rmse, curr_psnr, curr_ssim);
        end
    end
end

%% 3. Identify Best Parameters
[max_psnr, best_idx] = max(results(:, 5));
best_eta    = results(best_idx, 1);
best_mu     = results(best_idx, 2);
best_alpha  = results(best_idx, 3);

fprintf('\nGrid Search Finished!\n');
fprintf('Best Parameters: eta = %.3f, mu = %.3f, alpha = %.3f\n', best_eta, best_mu, best_alpha);
fprintf('Best PSNR: %.2f dB\n', max_psnr);

%% 4. Final Reconstruction with Optimal Parameters
fprintf('\nRunning final reconstruction with optimal parameters...\n');

s_k = zeros(mn, 1);
z_k = zeros(mn, 1);
u_k = zeros(mn, 1);
A_op_best = @(x) apply_LHS(x, P_S_sub, P_A_vec, best_mu, best_alpha, M, N);

for k = 1:max_iter
    s_S_mat_in = reshape(s_S, [M, N]);
    term1 = P_S_sub' * s_S_mat_in;
    RHS = term1(:) + best_alpha * s_A + best_mu * (z_k - u_k);
    [s_k, ~] = pcg(A_op_best, RHS, pcg_tol, pcg_maxIt, [], [], s_k);
    
    sigma_denoise = sqrt(best_eta / best_mu);
    v_k = reshape(s_k + u_k, [M, N]);
    z_mat = BM3D(v_k, sigma_denoise);
    z_k = z_mat(:);
    
    u_k = u_k + (s_k - z_k);
end

% Result Generation
S_final = reshape(s_k, [M, N]);
S_remix = S_final;
S_remix(mask_A == 1) = sinogram_true(mask_A == 1);

% Back-projection to Image Domain
img_final = iradon(S_final, theta, 'linear', filter, 1, img_size);
img_remix = iradon(S_remix, theta, 'linear', filter, 1, img_size);

m_final = struct('PSNR', psnr(img_final, X_true), 'SSIM', ssim(img_final, X_true));
m_remix = struct('PSNR', psnr(img_remix, X_true), 'SSIM', ssim(img_remix, X_true));

% --- Plotting Results ---
figure('Name', 'Best Parameters: Sinogram Domain');
subplot(1,3,1); imshow(sinogram_true, []); title('Original Sinogram');
subplot(1,3,2); imshow(S_final, []); title('Final Sinogram');
subplot(1,3,3); imshow(S_remix, []); title('Remix Sinogram');

figure('Name', 'Best Parameters: Image Domain');
subplot(1,3,1); imshow(X_true, []); title('Original Image');
subplot(1,3,2); imshow(img_final, []); 
title(sprintf('Final (PSNR: %.2f dB)', m_final.PSNR));
subplot(1,3,3); imshow(img_remix, []); 
title(sprintf('Remix (PSNR: %.2f dB)', m_remix.PSNR));

%% --- Helper Functions ---

% Function to apply the Left-Hand Side (LHS) operator
function y = apply_LHS(x_vec, P_S_sub, P_A_vec, mu, alpha, M, N)
    X = reshape(x_vec, [M, N]);
    % Mathematically: (P_S' * P_S + alpha * P_A' * P_A + mu * I) * x
    PsPsX = P_S_sub' * (P_S_sub * X); 
    alphaPaPaX = alpha * (P_A_vec.^2) .* x_vec; % P_A is a diagonal mask, so P_A'P_A is P_A.^2
    y = PsPsX(:) + alphaPaPaX + mu * x_vec;
end

% Function to create forward difference operators (Standard TV)
function [D, D_t] = forward_diff_operators(M, N)
    e = ones(M*N, 1);
    Dx = spdiags([-e e], [0 M], M*N, M*N);
    Dy = spdiags([-e e], [0 1], M*N, M*N);
    D = [Dx; Dy];
    D_t = D';
end