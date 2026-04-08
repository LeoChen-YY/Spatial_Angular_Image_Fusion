# Spatial Angular Image Fusion 程式說明

這個資料夾是一組以 MATLAB 撰寫的 CT 影像/正弦圖 (sinogram) 融合實驗。核心概念是：先用同一個 phantom 產生真實影像與真實 sinogram，再用真實 sinogram 產生兩種退化觀測(低空間解析度、低角度解析度)，最後透過不同的目標函數與 ADMM 把資訊融合回來。此流程即 Wald's protocol 框架。

## 專案在做什麼

專案主要比較「不同正則化或先驗」對 sinogram 融合品質的影響。整體流程大致如下：

1. 產生 `phantom(256)` 當作 ground truth。
2. 對 ground truth 做 `radon` 轉換得到完整 sinogram。
3. 建立兩種觀測：
   - 空間模糊版本：對 detector 維度做分組平均，模擬 `P_S`。
   - 角度遮罩版本：只保留部分角度，模擬 `P_A`。
4. 用 ADMM 架構求解 fused sinogram。
5. 再用 `iradon` 轉回影像空間。
6. 用 `RMSE`、`PSNR`、`SSIM` 評估重建品質。

## 各腳本的角色

### `optimize_sinogram_TV_v0.m`

基本的 sinogram-domain TV 融合版本。問題可寫成：

$$
\min_s \frac{1}{2}\lVert s_S - P_S s \rVert_2^2 + \frac{\alpha}{2}\lVert s_A - P_A s \rVert_2^2 + \eta\lVert Ds\rVert_1
$$

特點是把空間資訊與角度資訊都當成 soft data fidelity，搭配 TV 正則化做平滑重建。程式內使用 grid search，找出最佳 `eta`、`mu`、`alpha`。

### `optimize_sinogram_TV_v1.m`

這個版本改成 hard constraint。角度遮罩部分不再只是權重項，而是直接強制已量測到的角度滿足 `P_A s = s_A`。因此它更強調保留已知角度的正確值，避免被正則化過度修正。

### `optimize_sinogram_TV_v2.m`

這是 TV 的進一步版本。把空間資料保真項套用在未取樣角度上，也就是用 complementary mask `P_Ac` 來避免已知角度被重複約束。這個版本的重點是減少對已量測角度的干擾。

### `optimize_sinogram_TGV_v0.m`

基本款 TGV 版本。它把 TV 換成 TGV，並引入輔助變數 `v`，目標是降低 TV 常見的 staircase artifact。標頭裡的形式可概括為：

$$
\min_{s,v} \frac{1}{2}\lVert s_S - P_S s \rVert_2^2 + \frac{\alpha}{2}\lVert s_A - P_A s \rVert_2^2 + \gamma_1\lVert Ds - v\rVert_1 + \gamma_0\lVert Ev\rVert_1
$$

其中 `E` 是對稱梯度，讓二階結構也能被正則化。

### `optimize_sinogram_TGV_v1.m`

TGV 修改版，但把角度觀測改成 hard constraint，也就是直接把 sampled angles 投影回已知值。它比 v0 更強烈地保留量測資訊。

### `optimize_sinogram_TGV_v2.m`

最進階的 TGV 版本。除了 TGV 外，還加入 image-space BM3D refinement，也就是每次 ADMM 更新後，會把 sinogram 先 `iradon` 回影像，再用 BM3D 去雜訊，最後再 `radon` 回 sinogram，形成混合式先驗。這個版本在標頭裡也清楚寫出 hybrid TGV + BM3D 的設計。

### `optimize_sinogram_BM3D_v0.m`

這是 Plug-and-Play ADMM 版本。它不再顯式寫出 TV 或 TGV 正則化，而是把 BM3D 當作 proximal operator 使用。也就是說，`z-update` 由 BM3D 完成，藉由 denoiser 充當隱式先驗。

### `optimize_img_TV_v0.m`

這個版本把融合搬到 image domain。不直接重建 sinogram，而是先從兩個退化 sinogram 做 preliminary reconstructions，得到 `x_S` 與 `x_A`，再在影像空間做 TV 融合：

$$
\min_x \frac{1}{2}\lVert x - x_S \rVert_2^2 + \frac{\alpha}{2}\lVert x - x_A \rVert_2^2 + \eta\lVert Dx\rVert_1
$$

## 共用的實驗設定

- 影像尺寸：`256 x 256`
- 角度：`0:1:179`
- 重建濾波器：多數使用 `Ram-Lak`
- 空間退化：對 sinogram 的 detector 維度分組平均
- 角度退化：每隔一個角度或依固定間隔做遮罩
- 求解器：`ADMM` 搭配 `pcg`
- 評估指標：`RMSE`、`PSNR`、`SSIM`

## BM3D 套件

`bm3d_matlab_package_4.0.3/` 下載自 https://webpages.tuni.fi/foi/GCF-BM3D/index.html#ref_software。此專案使用資料夾中的 `BM3D.m` (`optimize_sinogram_BM3D_v0.m` 與 `optimize_sinogram_TGV_v2.m` 使用)。
