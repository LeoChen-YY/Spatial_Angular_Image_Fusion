% CT Sinogram Fusion using ADMM v3 (Hard Constraint)
% min_s 1/2 * || PaC*s_S - PaC*Ps*s ||^2_F + eta * TV(s)  s.t. Sa = Pa*s
clear; clc; close all;

%% 1. 參數設定與資料生成
img_size = 256;     
theta = 0:1:179;    
filter = 'Ram-Lak'; 

% --- Grid Search 範圍設定 ---
eta_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]; 
mu_list  = [0.1, 0.5, 1, 5];       
max_iter = 50; 

pcg_tol = 1e-6;
pcg_maxIt = 50;

% 生成基礎資料
X_true = phantom(img_size);
sinogram_true = radon(X_true, theta);
[M, N] = size(sinogram_true);
s_true = sinogram_true(:); 
mn = length(s_true);

% 2.1 構造 P_S_sub
K = 2; 
P_S_sub = sparse(M, M);
for i = 1:K:M
    idx = i : min(i + K - 1, M);
    P_S_sub(idx, idx) = 1/length(idx);
end

% 2.2 構造 P_A (用於遮罩與投影)
mask_A = zeros(M, N);
mask_A(:, 1:2:end) = 1;
mask_Ac = 1 - mask_A;
P_A_vec = sparse(mask_A(:)); 

% 2.3 生成觀測值
s_S_mat = P_S_sub * sinogram_true; 
s_S = s_S_mat(:);
s_A = P_A_vec .* s_true; % 硬約束目標

% 2.4 算子定義
[D, ~] = forward_diff_operators(M, N); 

%% 2. Grid Search 執行
results = []; 

fprintf('開始 Grid Search (v2 - Hard Constraint)...\n');
fprintf('%-10s %-10s %-10s %-10s %-10s\n', 'eta', 'mu', 'RMSE', 'PSNR', 'SSIM');

for e_idx = 1:length(eta_list)
    for m_idx = 1:length(mu_list)
        eta = eta_list(e_idx);
        mu = mu_list(m_idx);

        % --- ADMM 核心邏輯 ---
        s_k = zeros(mn, 1);
        u_k = zeros(2*mn, 1);
        A_op = @(x) apply_LHS_v2(x, P_S_sub, mask_Ac, D, mu, M, N);

        for k = 1:max_iter
            % Step 1: z-update (Shrinkage)
            v = D * s_k + u_k;
            z_k = sign(v) .* max(abs(v) - eta/mu, 0);

            % Step 2: s-update (PCG)
            % RHS: Ps' * Pa' * Pa * s_S + mu * D' * (z_k - u_k)
            % 因為 Pa 是對角遮罩且 Pa*s_S 已在觀察時隱含，簡化為下式：
            s_S_mat_in = reshape(s_S, [M, N]);
            term1 = P_S_sub * (mask_Ac .* s_S_mat_in); 
            RHS = term1(:) + mu * D' * (z_k - u_k);
            [s_k, ~] = pcg(A_op, RHS, pcg_tol, pcg_maxIt, [], [], s_k);

            % Step 2.5: Let Sa = Pa * s (Hard Constraint Projection)
            s_k(P_A_vec == 1) = s_A(P_A_vec == 1); 

            % Step 3: u-update
            u_k = u_k + (D * s_k - z_k);
        end

        % --- 影像重建 ---
        img_recon = iradon(reshape(s_k, [M, N]), theta, 'linear', filter, 1, img_size);

        % --- 計算指標 ---
        curr_rmse = sqrt(mean((X_true(:) - img_recon(:)).^2));
        curr_psnr = psnr(img_recon, X_true);
        curr_ssim = ssim(img_recon, X_true);

        results = [results; eta, mu, curr_rmse, curr_psnr, curr_ssim];
        fprintf('%-10.3f %-10.3f %-10.4f %-10.2f %-10.4f\n', eta, mu, curr_rmse, curr_psnr, curr_ssim);
    end
end

%% 3. 找出最佳結果
[max_psnr, best_idx] = max(results(:, 4));
best_eta = results(best_idx, 1);
best_mu  = results(best_idx, 2);

fprintf('\nGrid Search 結束！最佳參數: eta = %.3f, mu = %.3f, PSNR: %.2f dB\n', best_eta, best_mu, max_psnr);

%% 4. 以最佳參數重新跑一次並顯示圖片
s_k = zeros(mn, 1);
u_k = zeros(2*mn, 1); 
A_op_best = @(x) apply_LHS_v2(x, P_S_sub, mask_Ac, D, best_mu, M, N);

for k = 1:max_iter
    v = D * s_k + u_k;
    z_k = sign(v) .* max(abs(v) - best_eta/best_mu, 0);
    
    s_S_mat_in = reshape(s_S, [M, N]);
    term1 = P_S_sub * (mask_Ac .* s_S_mat_in);
    RHS = term1(:) + best_mu * D' * (z_k - u_k);
    [s_k, ~] = pcg(A_op_best, RHS, pcg_tol, pcg_maxIt, [], [], s_k);
    
    s_k(P_A_vec == 1) = s_A(P_A_vec == 1); 
    u_k = u_k + (D * s_k - z_k);
end

S_final = reshape(s_k, [M, N]);
img_final = iradon(S_final, theta, 'linear', filter, 1, img_size);

figure('Name', 'V2 Hard Constraint Results');
subplot(1,4,1); imshow(sinogram_true, []); title('Origin Sinogram');
subplot(1,4,2); imshow(S_final, []); title('Final Sinogram (Hard Cstr)');
subplot(1,4,3); imshow(X_true, []); title(sprintf('Origin Image'));
subplot(1,4,4); imshow(img_final, []); 
title(sprintf('Recon Image (PSNR: %.2f dB)', psnr(img_final, X_true)));

%% --- 輔助函數 ---
function y = apply_LHS_v2(x_vec, P_S_sub, mask_Ac, D, mu, M, N)
    X = reshape(x_vec, [M, N]);
    
    % 手寫稿推導：Ps' * (Pa_c)' * (Pa_c) * Ps * s
    % 1. 空間模糊 Ps * s
    PsX = P_S_sub * X;
    % 2. 未採樣角度遮罩 Pa_c * (Ps * s)
    PaCPsX = mask_Ac .* PsX;
    % 3. 轉置空間模糊 Ps' * (PaCPsX)
    PsPaCPsX = P_S_sub * PaCPsX;
    
    % 正則項: mu * D'D * s
    reg_term = mu * (D' * (D * x_vec));
    
    y = PsPaCPsX(:) + reg_term;
end

function [D, D_t] = forward_diff_operators(M, N)
    e = ones(M*N, 1);
    Dx = spdiags([-e e], [0 M], M*N, M*N);
    Dy = spdiags([-e e], [0 1], M*N, M*N);
    D = [Dx; Dy];
    D_t = D';
end