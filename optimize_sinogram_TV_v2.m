%% CT Sinogram Fusion using ADMM v3 (Hard Constraint)
%
% Optimization Problem:
% min_{s}  1/2 * || P_Ac * s_S - P_Ac * P_S * s ||^2_F + eta * || D * s ||_1
% subject to: P_A * s = s_A
%
% Where:
% s:      The target sinogram to be reconstructed (vectorized).
% s_S:    Observed spatially blurred sinogram.
% s_A:    Measured sharp sinogram samples at specific angles (Hard Constraint).
% P_S:    Spatial blurring/averaging operator.
% P_A:    Angular sampling mask for sampled angles.
% P_Ac:   Angular sampling mask for unsampled (complementary) angles.
% D:      Forward difference operator (Total Variation).
% eta:    Regularization parameter controlling smoothness.
% mu:     ADMM augmented Lagrangian penalty parameter.
%
% This version applies the spatial data fidelity term only to the unsampled angles
% while strictly enforcing measured values at the sampled angles.

clear; clc; close all;

%% 1. Parameter Settings and Data Generation
img_size = 256;     
theta = 0:1:179;    
filter = 'Ram-Lak'; 

% --- Grid Search Range Settings ---
eta_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]; 
mu_list  = [0.1, 0.5, 1, 5];       
max_iter = 50; 
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

% 2.2 Construct P_A and P_Ac (Angular Masks)
mask_A = zeros(M, N);
mask_A(:, 1:2:end) = 1;      % Sampled angles
mask_Ac = 1 - mask_A;        % Unsampled angles
P_A_vec = sparse(mask_A(:)); 

% 2.3 Generate Observations
s_S_mat = P_S_sub * sinogram_true; 
s_S = s_S_mat(:);
s_A = P_A_vec .* s_true; % Target for the Hard Constraint

% 2.4 Initialize Operators
[D, ~] = forward_diff_operators(M, N); 

%% 2. Grid Search Execution
results = []; 
fprintf('Starting Grid Search (v3 - Hard Constraint)...\n');
fprintf('%-10s %-10s %-10s %-10s %-10s\n', 'eta', 'mu', 'RMSE', 'PSNR', 'SSIM');

for e_idx = 1:length(eta_list)
    for m_idx = 1:length(mu_list)
        eta = eta_list(e_idx);
        mu = mu_list(m_idx);
        
        % --- ADMM Core Logic ---
        s_k = zeros(mn, 1);
        u_k = zeros(2*mn, 1);
        A_op = @(x) apply_LHS_v2(x, P_S_sub, mask_Ac, D, mu, M, N);
        
        for k = 1:max_iter
            % Step 1: z-update (Shrinkage / Soft Thresholding)
            v = D * s_k + u_k;
            z_k = sign(v) .* max(abs(v) - eta/mu, 0);
            
            % Step 2: s-update (Solve Linear System via PCG)
            % RHS logic: P_S' * P_Ac' * P_Ac * s_S + mu * D' * (z_k - u_k)
            s_S_mat_in = reshape(s_S, [M, N]);
            term1 = P_S_sub' * (mask_Ac .* s_S_mat_in); 
            RHS = term1(:) + mu * D' * (z_k - u_k);
            [s_k, ~] = pcg(A_op, RHS, pcg_tol, pcg_maxIt, [], [], s_k);
            
            % Step 2.5: Hard Constraint Projection (Sa = Pa * s)
            s_k(P_A_vec == 1) = s_A(P_A_vec == 1); 
            
            % Step 3: u-update (Dual variable)
            u_k = u_k + (D * s_k - z_k);
        end
        
        % --- Image Reconstruction ---
        img_recon = iradon(reshape(s_k, [M, N]), theta, 'linear', filter, 1, img_size);
        
        % --- Metrics Calculation ---
        curr_rmse = sqrt(mean((X_true(:) - img_recon(:)).^2));
        curr_psnr = psnr(img_recon, X_true);
        curr_ssim = ssim(img_recon, X_true);
        results = [results; eta, mu, curr_rmse, curr_psnr, curr_ssim];
        fprintf('%-10.3f %-10.3f %-10.4f %-10.2f %-10.4f\n', eta, mu, curr_rmse, curr_psnr, curr_ssim);
    end
end

%% 3. Identify Best Results
[max_psnr, best_idx] = max(results(:, 4));
best_eta = results(best_idx, 1);
best_mu  = results(best_idx, 2);
fprintf('\nGrid Search Finished! Best Params: eta = %.3f, mu = %.3f, PSNR: %.2f dB\n', best_eta, best_mu, max_psnr);

%% 4. Final Reconstruction with Optimal Parameters
s_k = zeros(mn, 1);
u_k = zeros(2*mn, 1); 
A_op_best = @(x) apply_LHS_v2(x, P_S_sub, mask_Ac, D, best_mu, M, N);

for k = 1:max_iter
    v = D * s_k + u_k;
    z_k = sign(v) .* max(abs(v) - best_eta/best_mu, 0);
    
    s_S_mat_in = reshape(s_S, [M, N]);
    term1 = P_S_sub' * (mask_Ac .* s_S_mat_in);
    RHS = term1(:) + best_mu * D' * (z_k - u_k);
    [s_k, ~] = pcg(A_op_best, RHS, pcg_tol, pcg_maxIt, [], [], s_k);
    
    % Hard constraint projection
    s_k(P_A_vec == 1) = s_A(P_A_vec == 1); 
    u_k = u_k + (D * s_k - z_k);
end

S_final = reshape(s_k, [M, N]);
img_final = iradon(S_final, theta, 'linear', filter, 1, img_size);

% --- Plotting Results ---
figure('Name', 'V3 Hard Constraint Results');
subplot(1,4,1); imshow(sinogram_true, []); title('Origin Sinogram');
subplot(1,4,2); imshow(S_final, []); title('Final Sinogram (Hard Cstr)');
subplot(1,4,3); imshow(X_true, []); title('Ground Truth Image');
subplot(1,4,4); imshow(img_final, []); 
title(sprintf('Recon Image (PSNR: %.2f dB)', psnr(img_final, X_true)));

%% --- Helper Functions ---

% Function to apply the Left-Hand Side (LHS) operator
function y = apply_LHS_v2(x_vec, P_S_sub, mask_Ac, D, mu, M, N)
    X = reshape(x_vec, [M, N]);
    
    % Data fidelity term derivation: Ps' * (Pa_c)' * (Pa_c) * Ps * s
    % 1. Spatial Blurring Ps * s
    PsX = P_S_sub * X;
    % 2. Apply Complementary Mask Pa_c * (Ps * s)
    PaCPsX = mask_Ac .* PsX;
    % 3. Adjoint Spatial Blur Ps' * (PaCPsX)
    PsPaCPsX = P_S_sub' * PaCPsX;
    
    % Regularization term: mu * D' * D * s
    reg_term = mu * (D' * (D * x_vec));
    
    y = PsPaCPsX(:) + reg_term;
end

% Function to generate forward difference operators (Total Variation)
function [D, D_t] = forward_diff_operators(M, N)
    e = ones(M*N, 1);
    Dx = spdiags([-e e], [0 M], M*N, M*N);
    Dy = spdiags([-e e], [0 1], M*N, M*N);
    D = [Dx; Dy];
    D_t = D';
end