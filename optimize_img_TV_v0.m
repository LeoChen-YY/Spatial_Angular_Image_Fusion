%% CT Image Fusion using ADMM with Grid Search
% 
% Optimization Problem:
% The code solves a Total Variation (TV) regularized image reconstruction problem:
% 
% min_{x}  1/2 * ||x - x_S||_2^2 + alpha/2 * ||x - x_A||_2^2 + eta * ||Dx||_1
%
% Where:
% x: The target image to reconstruct (vectorized)
% x_S: Reconstruction from spatially blurred sinogram
% x_A: Reconstruction from angularly masked sinogram
% D: Forward difference operator (Total Variation)
% alpha: Weighting factor between the two data fidelity terms
% eta: Regularization parameter for sparsity/smoothness
%
% ADMM formulation uses Mu (µ) as the augmented Lagrangian penalty parameter.

clear; clc; close all;

%% 1. Parameter Settings and Data Generation
img_size = 256;     
theta = 0:1:179;    

% --- Radon Transform Settings ---
filter = 'Ram-Lak'; % Options: "Ram-Lak", "Shepp-Logan", "Cosine", "Hamming", "Hann", "None"

% --- Grid Search Range Settings ---
eta_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]; % Testing different eta (regularization strength)
mu_list  = [0.1, 0.5, 1, 5];                      % Testing different mu (ADMM step size/penalty)
alpha_list = [0.01, 0.1, 1, 10, 100];             % Testing different alpha (data fidelity weight)
max_iter = 200;                                   % Iterations limited for search efficiency

% --- Linear Solver (PCG) Settings ---
pcg_tol = 1e-6;
pcg_maxIt = 100;

% Generate s_true (Vectorized Sinogram)
X_true = phantom(img_size);
sinogram_true = radon(X_true, theta);
[M, N] = size(sinogram_true);
s_true = sinogram_true(:); 
x_true = X_true(:);
mn = length(x_true);

% 2.1 Construct P_S_sub (Spatial Blurring Operator)
K = 3; 
P_S_sub = sparse(M, M);
for i = 1:K:M
    idx = i : min(i + K - 1, M);
    P_S_sub(idx, idx) = 1/length(idx);
end

% 2.2 Construct P_A (Angular Mask)
theta_A = 1:5:180;
mask_A = zeros(M, N);
mask_A(:, theta_A) = 1;
P_A_vec = sparse(mask_A(:)); 

% 2.3 Generate Observed Sinograms: s_S (blurred), s_A (masked)
s_S_mat = P_S_sub * sinogram_true; 
s_S = s_S_mat(:);
s_A = P_A_vec .* s_true;
s_A_mat = reshape(s_A, [M, N]);

% 2.4 Generate Preliminary Reconstructions: x_S, x_A
X_S = iradon(s_S_mat, theta, 'linear', filter, 1, img_size);
X_A = iradon(full(s_A_mat(:, theta_A)), theta_A, 'linear', filter, 1, img_size);
x_S = X_S(:);
x_A = X_A(:);

% 2.5 Define Difference Operators
[D, ~] = forward_diff_operators(img_size, img_size); 

%% 2. Grid Search Execution
results = []; % Table to store results

fprintf('Starting Grid Search...\n');
fprintf('%-10s %-10s %-10s %-10s %-10s %-10s\n', 'eta', 'mu', 'alpha', 'RMSE', 'PSNR', 'SSIM');

for e_idx = 1:length(eta_list)
    for m_idx = 1:length(mu_list)
        for a_idx = 1:length(alpha_list)
            eta = eta_list(e_idx);
            mu = mu_list(m_idx);
            alpha = alpha_list(a_idx);

            % --- ADMM Core Logic ---
            x_k = zeros(mn, 1);
            u_k = zeros(2*mn, 1);
            A_op = @(x) apply_LHS(x, D, mu, alpha);

            for k = 1:max_iter
                % Soft-thresholding (z-update)
                v = D * x_k + u_k;
                z_k = sign(v) .* max(abs(v) - eta/mu, 0);

                % Linear System Solve (x-update) using Conjugate Gradient
                RHS = x_S + alpha * x_A + mu * D' * (z_k - u_k);
                [x_k, ~] = pcg(A_op, RHS, pcg_tol, pcg_maxIt, [], [], x_k);

                % Dual variable update (u-update)
                u_k = u_k + (D * x_k - z_k);
            end

            % --- Reshape Final Image ---
            X_final = reshape(x_k, [img_size, img_size]);

            % --- Calculate Metrics ---
            curr_rmse = sqrt(mean((X_true(:) - X_final(:)).^2));
            curr_psnr = psnr(X_final, X_true);
            curr_ssim = ssim(X_final, X_true);

            % Record Results
            results = [results; eta, mu, alpha, curr_rmse, curr_psnr, curr_ssim];
            fprintf('%-10.3f %-10.3f %-10.3f %-10.4f %-10.2f %-10.4f\n', eta, mu, alpha, curr_rmse, curr_psnr, curr_ssim);
        end
    end
end

%% 3. Identify Best Results
[max_psnr, best_idx] = max(results(:, 5));
best_eta    = results(best_idx, 1);
best_mu     = results(best_idx, 2);
best_alpha  = results(best_idx, 3);

fprintf('\nGrid Search Finished!\n');
fprintf('Best Parameters: eta = %.3f, mu = %.3f, alpha = %.3f\n', best_eta, best_mu, best_alpha);
fprintf('Best PSNR: %.2f dB\n', max_psnr);

%% 4. Final Reconstruction with Best Parameters
fprintf('\nPerforming final reconstruction with optimal parameters...\n');

% Re-initialize variables with optimal parameters
x_k = zeros(mn, 1);
u_k = zeros(2*mn, 1); 
A_op_best = @(x) apply_LHS(x, D, best_mu, best_alpha);

% Final iteration loop
for k = 1:max_iter
    v = D * x_k + u_k;
    z_k = sign(v) .* max(abs(v) - best_eta/best_mu, 0);
    
    RHS = x_S + best_alpha * x_A + best_mu * D' * (z_k - u_k);
    [x_k, ~] = pcg(A_op_best, RHS, pcg_tol, pcg_maxIt, [], [], x_k);
    
    u_k = u_k + (D * x_k - z_k);
end

% Generate final results
X_final = reshape(x_k, [img_size, img_size]);
m_final = struct('PSNR', psnr(X_final, X_true), 'SSIM', ssim(X_final, X_true));

% --- Visualization ---
% Figure 2: Image Domain Comparison
figure('Name', 'Best Parameters: Image Domain');
subplot(1,2,1); imshow(X_true, []); title('Original Image');
subplot(1,2,2); imshow(X_final, []); 
title(sprintf('Final (PSNR: %.2f dB)', m_final.PSNR));

%% --- Helper Functions ---

% Function to apply the Left-Hand Side (LHS) operator for the linear system
function y = apply_LHS(x_vec, D, mu, alpha)
    data_fitting_term = (1+alpha) * x_vec;
    reg_term = mu * (D' * (D * x_vec));
    y = data_fitting_term + reg_term;
end

% Function to create forward difference operators for Total Variation
function [D, D_t] = forward_diff_operators(M, N)
    e = ones(M*N, 1);
    Dx = spdiags([-e e], [0 M], M*N, M*N);
    Dy = spdiags([-e e], [0 1], M*N, M*N);
    D = [Dx; Dy];
    D_t = D';
end