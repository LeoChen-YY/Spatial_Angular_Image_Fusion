% CT Sinogram Fusion using ADMM with TGV Regularization & Grid Search (Hard Constraint)
% min_s 1/2 * || s_S - P_S * s||^2_F + TGV(s)   s.t. Sa = Pa*s
clear; clc; close all;

%% 1. 參數設定與資料生成
img_size = 256;     
theta = 0:1:179;    
filter = 'Ram-Lak'; 

% --- Grid Search 範圍設定 (建議先從各 2 組開始測試，避免計算過久) ---
gamma0_list = [0.01, 0.1]; 
gamma1_list = [0.01, 0.1];
mu0_list    = [0.1, 0.5];     
mu1_list    = [0.1, 0.5];     
max_iter    = 50;           % Grid Search 時的迭代次數

pcg_tol = 1e-6;
pcg_maxIt = 50;

% 生成基礎資料 (True Image & Sinogram)
X_true = phantom(img_size);
sinogram_true = radon(X_true, theta);
[M, N] = size(sinogram_true);
s_true = sinogram_true(:); 
mn = length(s_true);

% 2.1 構造 P_S_sub (空間模糊)
K = 2; 
P_S_sub = sparse(M, M);
for i = 1:K:M
    idx = i : min(i + K - 1, M);
    P_S_sub(idx, idx) = 1/length(idx);
end

% 2.2 構造 P_A (角度遮罩)
mask_A = zeros(M, N);
mask_A(:, 1:2:end) = 1;
P_A_vec = sparse(mask_A(:)); 

% 2.3 生成觀測值 (s_S 是模糊的, s_A 是稀疏採樣但清晰的)
s_S_mat = P_S_sub * sinogram_true; 
s_S = s_S_mat(:);
s_A = P_A_vec .* s_true;

% 2.4 算子初始化
[Dx, Dy] = get_diff_ops(M, N);
D = [Dx; Dy]; 

%% 2. Grid Search 開始
results = [];
total_configs = length(gamma0_list) * length(gamma1_list) * length(mu0_list) * length(mu1_list);
count = 0;

best_psnr = -inf;
best_params = [];

fprintf('開始 Grid Search (TGV)，共 %d 組配置...\n', total_configs);

for g0 = gamma0_list
    for g1 = gamma1_list
        for m0 = mu0_list
            for m1 = mu1_list
                    count = count + 1;
                    
                    % 變數初始化
                    s_k = zeros(mn, 1);
                    v_k = zeros(2*mn, 1);
                    z1_k = zeros(2*mn, 1);
                    z2_k = zeros(3*mn, 1);
                    u1_k = zeros(2*mn, 1);
                    u2_k = zeros(3*mn, 1);

                    % 定義當前參數下的算子
                    A_s_op = @(s) apply_LHS_s(s, P_S_sub, D, m0, M, N);
                    A_v_op = @(v) apply_LHS_v(v, Dx, Dy, m0, m1, M, N);

                    % ADMM 迭代
                    for k = 1:max_iter
                        % z 更新 (Shrinkage)
                        tmp1 = D*s_k - v_k + u1_k;
                        z1_k = sign(tmp1) .* max(abs(tmp1) - g0/m0, 0);
                        Ev_k = apply_E(v_k, Dx, Dy, M, N);
                        tmp2 = Ev_k + u2_k;
                        z2_k = sign(tmp2) .* max(abs(tmp2) - g1/m1, 0);

                        % s 更新 (PCG)
                        s_S_mat_in = reshape(s_S, [M, N]);
                        term1 = P_S_sub * s_S_mat_in;
                        RHS_s = term1(:) + m0 * D' * (z1_k + v_k - u1_k);
                        [s_k, ~] = pcg(A_s_op, RHS_s, pcg_tol, pcg_maxIt, [], [], s_k);
                        s_k(P_A_vec == 1) = s_A(P_A_vec == 1); 

                        % v 更新 (PCG)
                        Et_term = apply_Et(z2_k - u2_k, Dx, Dy, M, N);
                        RHS_v = m0 * (D*s_k - z1_k + u1_k) + m1 * Et_term;
                        [v_k, ~] = pcg(A_v_op, RHS_v, pcg_tol, pcg_maxIt, [], [], v_k);

                        % u 更新
                        u1_k = u1_k + (D*s_k - v_k - z1_k);
                        u2_k = u2_k + (apply_E(v_k, Dx, Dy, M, N) - z2_k);
                    end

                    % 使用 Sinogram PSNR 作為篩選指標
                    img_recon = iradon(full(reshape(s_k, [M, N])), theta, 'linear', filter, 1, img_size);
                    curr_psnr = psnr(img_recon, X_true);
                    results = [results; g0, g1, m0, m1, curr_psnr];
                    
                    if curr_psnr > best_psnr
                        best_psnr = curr_psnr;
                        best_params = [g0, g1, m0, m1];
                    end
                    
                    fprintf('[%d/%d] g0:%.2f, g1:%.2f, m0:%.1f, m1:%.1f | Remix Image PSNR: %.4f\n', ...
                        count, total_configs, g0, g1, m0, m1, curr_psnr);
            end
        end
    end
end

%% 3. 以最佳參數進行最終重建與 Remix
bg0 = best_params(1); bg1 = best_params(2); 
bm0 = best_params(3); bm1 = best_params(4);

fprintf('\n使用最佳參數 (g0=%.2f, g1=%.2f, m0=%.1f, m1=%.1f) 重建...\n', bg0, bg1, bm0, bm1);

s_k = zeros(mn, 1); v_k = zeros(2*mn, 1);
z1_k = zeros(2*mn, 1); z2_k = zeros(3*mn, 1);
u1_k = zeros(2*mn, 1); u2_k = zeros(3*mn, 1);

A_s_best = @(s) apply_LHS_s(s, P_S_sub, D, bm0, M, N);
A_v_best = @(v) apply_LHS_v(v, Dx, Dy, bm0, bm1, M, N);

for k = 1:max_iter * 2 % 最終重建增加迭代次數
    % ADMM 核心步驟 (同上)
    tmp1 = D*s_k - v_k + u1_k;
    z1_k = sign(tmp1) .* max(abs(tmp1) - bg0/bm0, 0);
    tmp2 = apply_E(v_k, Dx, Dy, M, N) + u2_k;
    z2_k = sign(tmp2) .* max(abs(tmp2) - bg1/bm1, 0);
    
    s_S_mat_in = reshape(s_S, [M, N]);
    term1 = P_S_sub * s_S_mat_in;
    RHS_s = term1(:) + bm0 * D' * (z1_k + v_k - u1_k);
    [s_k, ~] = pcg(A_s_best, RHS_s, pcg_tol, pcg_maxIt, [], [], s_k);
    s_k(P_A_vec == 1) = s_A(P_A_vec == 1); 
    
    RHS_v = bm0 * (D*s_k - z1_k + u1_k) + bm1 * apply_Et(z2_k - u2_k, Dx, Dy, M, N);
    [v_k, ~] = pcg(A_v_best, RHS_v, pcg_tol, pcg_maxIt, [], [], v_k);
    
    u1_k = u1_k + (D*s_k - v_k - z1_k);
    u2_k = u2_k + (apply_E(v_k, Dx, Dy, M, N) - z2_k);
end

% --- 轉回影像空間 ---
S_final = reshape(s_k, [M, N]);
img_final = iradon(S_final, theta, 'linear', filter, 1, img_size);

% --- 指標計算 ---
m_final = eval_metrics(X_true, img_final);

fprintf('\n最終結果指標比較:\n');
fprintf('Final -> RMSE: %.4f, PSNR: %.2f dB, SSIM: %.4f\n', m_final.RMSE, m_final.PSNR, m_final.SSIM);

%% 4. 繪圖
figure('Name', 'TGV V2 Hard Constraint Results');
subplot(1,4,1); imshow(sinogram_true, []); title('Origin Sinogram');
subplot(1,4,2); imshow(S_final, []); title('Final Sinogram (Hard Cstr)');
subplot(1,4,3); imshow(X_true, []); title(sprintf('Origin Image'));
subplot(1,4,4); imshow(img_final, []); 
title(sprintf('Recon Image (PSNR: %.2f dB)', psnr(img_final, X_true)));

%% --- 輔助函數 ---
function m = eval_metrics(gt, recon)
    m.RMSE = norm(gt(:) - recon(:)) / norm(gt(:));
    m.PSNR = psnr(recon, gt);
    m.SSIM = ssim(recon, gt);
end

function y = apply_LHS_s(s, P_S_sub, D, mu0, M, N)
    S_mat = reshape(s, [M, N]);
    PsPsS = P_S_sub * (P_S_sub * S_mat);
    y = PsPsS(:) + mu0 * (D' * (D * s));
end

function y = apply_LHS_v(v, Dx, Dy, mu0, mu1, M, N)
    Ev = apply_E(v, Dx, Dy, M, N);
    EtEv = apply_Et(Ev, Dx, Dy, M, N);
    y = mu0 * v + mu1 * EtEv;
end

function Ev = apply_E(v, Dx, Dy, M, N)
    MN = M*N;
    vx = v(1:MN); vy = v(MN+1:end);
    Ev = [Dx*vx; Dy*vy; 0.5*(Dy*vx + Dx*vy)];
end

function Etz = apply_Et(z, Dx, Dy, M, N)
    MN = M*N;
    z11 = z(1:MN); z22 = z(MN+1:2*MN); z12 = z(2*MN+1:end);
    vx = Dx'*z11 + 0.5*Dy'*z12;
    vy = Dy'*z22 + 0.5*Dx'*z12;
    Etz = [vx; vy];
end

function [Dx, Dy] = get_diff_ops(M, N)
    e = ones(M*N, 1);
    Dx = spdiags([-e e], [0 1], M*N, M*N); 
    Dy = spdiags([-e e], [0 M], M*N, M*N);
end