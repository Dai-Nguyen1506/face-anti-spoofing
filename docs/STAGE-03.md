# STAGE 03: Xây dựng Kiến trúc Đa nhiệm Đa Không gian Màu (Dual-Space Multi-Task Learning - Model 3) cho Nhận diện Gian lận Quét mặt (Face Anti-Spoofing)

## 1. Tổng quan & Mục đích (Overview & Purpose)

Tài liệu này mô tả chi tiết quy trình thiết kế kiến trúc đa nhiệm, cơ chế sinh nhãn giám sát phụ trực tiếp trên RAM, tối ưu hóa siêu tham số và đánh giá **Giai đoạn 3 (`03_Advanced_Model.ipynb`)** thuộc dự án *Hệ thống CNN Học sâu cho Chống giả mạo Khuôn mặt Đa môi trường tích hợp Học đa nhiệm (Multi-task Learning)*.

* **Mục đích cốt lõi:** Xây dựng kiến trúc mạng học đa nhiệm tự thiết kế (**Model 3 - `MultiTaskFASNet**`), tích hợp đồng thời **không gian màu kép (`RGB` // `YCrCb`)** và **2 nhánh giám sát phụ (`3D Depth Regression` & `FFT/LBP Noise Regression`)** trên nền tảng khung xương thắng cuộc từ Giai đoạn 2 (`mobilenet_v3_large`) nhằm ép mô hình học bản chất hình học 3D và đặc trưng vật lý bề mặt thay vì học vẹt ngữ cảnh ánh sáng cục bộ.
* **Vai trò trong toàn bộ dự án:**
1. Giải quyết bài toán suy giảm hiệu năng do chênh lệch miền dữ liệu (**Domain Shift**) và khắc phục các ca lỗi khó của Model 2 (như tấn công phát lại màn hình độ phân giải cao cắt sát mặt hoặc người thật đeo kính râm phản quang ngoài nắng).
2. Đóng gói toàn bộ kiến trúc thành **1 tệp trọng số tự chứa duy nhất (`model3_multitask_best.pt`)** — không phụ thuộc vào mô hình sinh nhãn phụ lúc suy luận thực tế (Inference) — để sẵn sàng triển khai cho **Ứng dụng Camera Thời gian thực (Live Deployment)** và làm điểm khởi tạo cho **Giai đoạn 4 (`CelebA-Spoof`)**.



---

## 2. Mục tiêu Kỹ thuật (Technical Objectives)

1. **Tối ưu hóa Phần cứng Local (`Intel iGPU/XPU` & `16GB RAM`) cho Mạng Đa nhiệm Phức hợp:** Vận hành kiến trúc đa nhánh trên `PyTorch 2.14.0+xpu` (`xpu:0` với 13.24 GB bộ nhớ hợp nhất) sử dụng cơ chế tính toán hỗn hợp `bfloat16` (`AMP`), định dạng bộ nhớ `channels_last` và vô hiệu hóa toàn bộ cờ `inplace=False` trong các tầng kích hoạt để đảm bảo an toàn tuyệt đối cho bộ nhớ đồ họa tích hợp.
2. **Sinh Nhãn Giám sát Phụ (`3D Depth` & `FFT/LBP Noise`) Trực tiếp trên `Shared Memory RAM`:**
* Sử dụng mô hình ước lượng độ sâu tiền huấn luyện **`MiDaS_small`** chạy trực tiếp trên `xpu:0` làm Teacher sinh bản đồ độ sâu 3D (`24x24`) cho toàn bộ ảnh người thật (`Real = 1`), sau đó tự động giải phóng `MiDaS_small` khỏi bộ nhớ ngay khi nạp xong Cache.
* Sử dụng biến đổi **Fourier nhanh 2 chiều lọc thông cao (`2D-FFT High-Pass`)** kết hợp **Mẫu nhị phân cục bộ vector hóa (`Vectorized LBP`)** chạy đa luồng (`12 CPU threads`) để sinh bản đồ nhiệt nhiễu vân sóng màn hình Moiré và vi kết cấu (`24x24`) cho ảnh giả mạo (`Spoof = 0`).
* Lưu trữ toàn bộ ảnh `RGB (192x192)` cùng 2 bản đồ `Depth (24x24)` và `Noise (24x24)` dưới dạng `torch.uint8` trong `Shared Memory Tensor` mà không cần ghi tệp trung gian ra ổ cứng.


3. **Đồng bộ hóa Phép Biến đổi Hình học Đa mục tiêu (`Multi-Target Augmentation`):** Cấu hình `Albumentations` với `additional_targets={"depth": "mask", "noise": "mask"}` kết hợp nội suy `F.interpolate(mode="area")` trên Tensor nhằm đảm bảo khi ảnh RGB bị lật, xoay, dịch chuyển hoặc cắt lỗ (`CoarseDropout`), hai bản đồ nhãn phụ `Depth` và `Noise` cũng biến đổi khớp chính xác từng tọa độ pixel.
4. **Kế thừa Toàn bộ Kỹ thuật Chống Lệch Nhãn & Hội tụ từ Stage 1 & 2:** Duy trì bộ lấy mẫu cân bằng lớp `WeightedRandomSampler` (tỷ lệ 50/50 trong mỗi mini-batch), hàm mất mát phân loại `FocalLossWithSmoothing` ($\gamma = 2.0, \alpha_{\text{real}} = 0.55, \varepsilon = 0.08$), tốc độ học phân tầng (`Differential Learning Rate`) và bộ giảm tốc độ học tự động `ReduceLROnPlateau (factor=0.5, patience=3)`.

---

## 3. Pipeline Dữ liệu Đa nhiệm & Cơ chế Sinh Nhãn Giám sát Phụ trên RAM

### 3.1. Hiệu năng Tiền nạp `Shared Memory Cache` Đa nhiệm

Trên cùng phân phối dữ liệu **LCC-FASD** gồm **8,299 ảnh Train**, **2,948 ảnh Val** và **7,580 ảnh Test**, hệ thống tiền xử lý đa nhiệm tại Cell 5 đạt hiệu năng lưu trữ như sau:

| Tập dữ liệu (Split) | Số lượng mẫu | Độ phân giải Tensor trong RAM | Thời gian Nạp & Sinh nhãn (`MiDaS` + `FFT/LBP`) | Dung lượng Shared RAM Cố định |
| --- | --- | --- | --- | --- |
| **Training Split** | `8,299` mẫu | RGB `192x192x3` + Depth `24x24` + Noise `24x24` (`uint8`) | **98.62 giây** | **884.4 MB** |
| **Validation Split** | `2,948` mẫu | RGB `192x192x3` + Depth `24x24` + Noise `24x24` (`uint8`) | **11.97 giây** | **314.2 MB** |
| **Evaluation (Test)** | `7,580` mẫu | Đọc trực tiếp khi đánh giá cuối | — | `0 MB` *(Tiết kiệm RAM)* |
| **Tổng cộng Cache** | **`11,247` mẫu** | **3 luồng dữ liệu đồng bộ** | **110.59 giây** | **`1,198.6 MB` (~1.17 GB)** |

* **Nhận xét kỹ thuật:** Việc bổ sung thêm 2 bản đồ mục tiêu `24x24` cho 11,247 ảnh chỉ làm tăng dung lượng RAM thêm đúng **12.4 MB** so với mức `1,186.2 MB` của Stage 1 và Stage 2, giữ bộ nhớ hệ thống hoàn toàn an toàn dưới ngưỡng `3,200 MB` của máy 16GB RAM.

### 3.2. Quy ước Đối xứng Vật lý cho Hai Nhánh Giám sát Phụ

Để giúp mô hình phân tách rạch ròi giữa đặc trưng sinh trắc học của người thật và nhiễu vật liệu của tác nhân giả mạo, nhãn Ground-Truth của hai nhánh phụ được thiết kế đối xứng nghịch đảo:

* **Nhánh Độ sâu 3D (`Head 2 - 3D Depth Map GT` kích thước `1x24x24` $\in [0, 1]$):**
* **Ảnh `Real (1)`:** Được suy luận từ mô hình `MiDaS_small` (`midas_v21_small_256.pt`), chuẩn hóa Min-Max về $[0.0, 1.0]$, phản ánh cấu trúc lồi lõm tự nhiên của khuôn mặt (sống mũi, trán, gò má nhô cao đạt giá trị gần `1.0`, hốc mắt và nền xung quanh thấp dần về `0.0`).
* **Ảnh `Spoof (0)`:** Gán ma trận phẳng tuyệt đối $\mathbf{0}_{24\times24}$ (`min = 0.00, max = 0.00`) do ảnh in trên giấy hoặc màn hình phát lại chỉ là mặt phẳng 2D.


* **Nhánh Nhiễu & Phản xạ (`Head 3 - FFT/LBP Noise Map GT` kích thước `1x24x24` $\in [0, 1]$):**
* **Ảnh `Spoof (0)`:** Tổ hợp tuyến tính `50% Năng lượng Phổ tần số cao 2D-FFT` (với mặt nạ lọc thông cao Gaussian bán kính cắt `cutoff_ratio = 0.12`) và `50% Phương sai Vi kết cấu LBP 8 hướng lân cận`, chuẩn hóa bền vững theo phân vị $[P_5, P_{98}]$ về $[0.0, 1.0]$ nhằm làm nổi bật vân giao thoa màn hình Moiré và hạt mực in.
* **Ảnh `Real (1)`:** Gán ma trận $\mathbf{0}_{24\times24}$ để dạy mạng rằng các vùng sáng tự nhiên trên da người thật không phải là nhiễu giả mạo.



---

## 4. Phương pháp luận & Kiến trúc Mô hình (`DualSpaceMultiTaskFASNet`)

### 4.1. Sơ đồ Khối Song song & Nối tiếp bên trong `MultiTaskFASNet`

Mô hình `MultiTaskFASNet` (~6.8 triệu tham số) nhận đầu vào duy nhất là Tensor ảnh RGB chuẩn hóa `(B, 3, 192, 192)` và xử lý qua 5 khối chức năng kết hợp song song và nối tiếp:

1. **Nhánh Song song 1 — Trích xuất Đa tỷ lệ trên Không gian `RGB` (`Shared Backbone` + `Cascaded FPN Neck`):**
* Chia chuỗi `self.encoder` (`mobilenet_v3_large`) thành 3 chặng nối tiếp để trích xuất 3 bản đồ đặc trưng ở 3 độ phân giải không gian khác nhau:
* `Stage 2` (`encoder[:7]`): Xuất Tensor `(B, 40, 24, 24)` giữ chi tiết cạnh cục bộ.
* `Stage 4` (`encoder[7:13]`): Xuất Tensor `(B, 112, 12, 12)` nắm bắt cấu trúc ngũ quan.
* `Stage 6` (`encoder[13:]`): Xuất Tensor `(B, 960, 6, 6)` nắm bắt ngữ cảnh toàn cục.


* **Tháp Đặc trưng Nối tiếp (`Cascaded Multi-Scale FPN Neck`):** Chiếu cả 3 tầng về `256` kênh bằng `Conv1x1`, sau đó nội suy song tuyến tính (`Bilinear Upsample x2`) và cộng nối tiếp từ `6x6` $\rightarrow$ `12x12` $\rightarrow$ `24x24` qua các khối tích chập tách chiều sâu (`Depthwise Separable Conv 3x3`), thu được bản đồ đặc trưng đa tỷ lệ `P2` kích thước `(B, 256, 24, 24)`.


2. **Nhánh Song song 2 — Không gian Sắc độ `YCrCb` & Tích chập Sai phân Trung tâm (`Conv2d_CD Stem`):**
* Lớp `DifferentiableRGBToYCrCb` chuyển đổi trực tiếp Tensor RGB sang không gian màu `YCrCb` (chuẩn ITU-R BT.601) ngay trên GPU XPU mà không tốn thêm RAM lưu trữ.
* Dẫn qua chuỗi 3 tầng `Conv2d_CD` và `ResidualCDBlock` (với hệ số đạo hàm sai phân trung tâm $\theta = 0.40$ kế thừa từ Stage 1 và chú ý kênh `SEBlock`) hạ độ phân giải từ `192x192` $\rightarrow$ `96x96` $\rightarrow$ `48x48` $\rightarrow$ `(B, 64, 24, 24)`.


3. **Khối Hợp nhất Đa Không gian Màu (`Cross-Space SE Fusion`):**
* Ghép nối (`Concat`) bản đồ `RGB FPN (256 kênh)` và bản đồ `YCrCb CD (64 kênh)` thành `320` kênh, nén qua `Conv1x1` về `256` kênh và lọc qua khối chú ý kênh `SEBlock(256)` để tạo **Bản đồ Đặc trưng Hợp nhất `F_fused (B, 256, 24, 24)**`.


4. **Hai Nhánh Giải mã Giám sát Phụ Song song (`Head 2` // `Head 3`):**
* **`Head 2 (3D Depth Decoder)`:** Đưa `F_fused` qua chuỗi 3 khối **Tích chập Giãn nở `DilatedResidualBlock` (`dilation = 1, 2, 4`)** với độ rộng `128` kênh nhằm mở rộng vùng cảm thụ ra toàn khuôn mặt trên lưới `24x24`, xuất ra đặc trưng ẩn `d_feat (B, 128, 24, 24)` và bản đồ dự đoán `pred_depth (B, 1, 24, 24)` qua hàm `Sigmoid`.
* **`Head 3 (Noise/Moiré Decoder)`:** Đưa `F_fused` qua 2 khối `ResidualCDBlock (128 kênh)` kết hợp **Cổng Chú ý Không gian `SpatialAttentionGate (kernel_size=7)**` để khoanh vùng các điểm lóa phản xạ và sọc màn hình, xuất ra đặc trưng ẩn `n_feat (B, 128, 24, 24)` và bản đồ dự đoán `pred_noise (B, 1, 24, 24)`.


5. **Cầu nối Tương tác Vật lý (`Physics-Guided Gating`) & Nhánh Phân loại Chính (`Head 1`):**
* Sử dụng trực tiếp `pred_depth` và `pred_noise` làm mặt nạ chú ý không gian nhân trọng số vào đặc trưng ẩn: $\text{Gated\_D} = d_{\text{feat}} \odot (1 + \text{pred\_depth})$ và $\text{Gated\_N} = n_{\text{feat}} \odot (1 + \text{pred\_noise})$.
* Áp dụng `Dual Pooling (AvgPool + MaxPool)` trên tầng cuối Backbone (`1920` chiều), nhánh Depth (`256` chiều) và nhánh Noise (`256` chiều), hợp nhất thành **Siêu Vector Đặc trưng `2432` chiều** đi vào mạng phân loại 3 tầng (`2432 -> 512 -> 256 -> 2`) để xuất Logits `Real / Spoof`.



### 4.2. Hàm Mục tiêu Đa nhiệm Tổng hợp (`CompositeMultiTaskLoss`)

Hàm mất mát tổng hợp kết hợp 3 thành phần có trọng số điều chỉnh $\lambda_{\text{depth}}$ và $\lambda_{\text{noise}}$:

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Focal\_Cls}} + \lambda_{\text{depth}} \cdot \mathcal{L}_{\text{Depth}} + \lambda_{\text{noise}} \cdot \mathcal{L}_{\text{Noise}}$$

Trong đó:

* $\mathcal{L}_{\text{Focal\_Cls}}$: Hàm `FocalLossWithSmoothing` ($\gamma = 2.0, \alpha_{\text{real}} = 0.55, \varepsilon = 0.08$).
* $\mathcal{L}_{\text{Depth}} = \text{SmoothL1}(\hat{D}, D) + 0.5 \cdot \left( \Vert{}\nabla_x \hat{D} - \nabla_x D\Vert{}_1 + \Vert{}\nabla_y \hat{D} - \nabla_y D\Vert{}_1 \right)$: Kết hợp sai số cường độ Huber với **Sai số Đạo hàm Không gian (`Spatial Gradient Loss`)** theo trục $x, y$ để buộc mạng tái tạo đúng độ dốc 3D của sống mũi và hốc mắt.
* $\mathcal{L}_{\text{Noise}} = 0.7 \cdot \text{SmoothL1}(\hat{N}, N) + 0.3 \cdot \text{MSE}(\hat{N}, N)$: Đo sai số tái tạo bản đồ nhiệt nhiễu tần số cao.

---

## 5. Kết quả Pha 1 & Pha 2: Đọ sức Khởi tạo & Tối ưu Siêu tham số Đa nhiệm (Optuna)

### 5.1. Pha 1 — Lựa chọn Chế độ Khởi tạo Backbone (`stage2_warmstart` vs. `imagenet_freshstart`)

Hệ thống tiến hành đọ sức trực tiếp giữa hai chiến lược khởi tạo phần thân `mobilenet_v3_large`:

* **`stage2_warmstart`:** Kế thừa trọng số `encoder` đã tinh chỉnh chuyên biệt cho dữ liệu FAS từ `model2_transfer_best.pt` (Stage 2).
* **`imagenet_freshstart`:** Khởi tạo lại `encoder` từ trọng số `ImageNet` gốc của `torchvision`.
* 👉 **Kết quả Pha 1:** Chế độ **`stage2_warmstart`** giành chiến thắng thuyết phục và được chọn làm phương thức khởi tạo chuẩn cho Pha 2 và Pha 3. Việc tận dụng nền tảng đặc trưng đã hội tụ từ Stage 2 giúp các nhánh phụ mới (`YCrCb`, `FPN`, `Depth/Noise Decoders`) bắt nhịp ngay lập tức ở những epoch đầu tiên.

### 5.2. Pha 2 — Kết quả Tối ưu Siêu tham số Đa nhiệm với Optuna (`16 Trials x 8 Epochs`)

Quá trình tìm kiếm Optuna chạy 16 trials (mỗi trial 8 epochs) cho chế độ `stage2_warmstart` ghi nhận độ ổn định rất cao: 15/16 trials hoàn tất (`DONE`) với `Best Val ACER` đều đạt từ `2.45%` đến `3.70%` và chỉ có 1 trial bị cắt tỉa sớm (`Trial 09` bị `PRUNED` sau 460.1s).

* **Kỷ lục Optuna:** **Trial #14** đạt kết quả tốt nhất với **`Val ACER = 2.45%`** và **`Val ROC-AUC = 99.78%`** (vượt qua kỷ lục Optuna `2.59%` của Stage 2):

| Siêu tham số (Hyperparameter) | Khoảng tìm kiếm (Search Space) | Giá trị Tối ưu (Trial #14) | Ý nghĩa Vật lý & Kỹ thuật |
| --- | --- | --- | --- |
| **`init_mode`** | Sàng lọc từ Pha 1 | **`stage2_warmstart`** | Kế thừa trọng số Backbone từ `model2_transfer_best.pt`. |
| **`lambda_depth` ($\lambda_d$)** | $[0.15, 1.20]$ (step `0.05`) | **`0.55`** | Trọng số điều chuẩn nhánh hình học 3D Depth vừa phải, giữ ổn định đạo hàm. |
| **`lambda_noise` ($\lambda_n$)** | $[0.15, 1.20]$ (step `0.05`) | **`0.85`** | Ưu tiên phạt mạnh nhánh nhiễu FFT/LBP để bắt nhạy các vân sóng màn hình Moiré. |
| **`lr_head`** | $[3 \times 10^{-4}, 1.8 \times 10^{-3}]$ | **`0.001162`** | Tốc độ học khởi điểm giúp các khối FPN, YCrCb và Decoder hội tụ nhanh. |
| **`backbone_lr_ratio`** | $[0.10, 0.35]$ | **`0.18`** | Tốc độ học của Backbone chỉ bằng 18% của Head, bảo toàn đặc trưng Stage 2. |
| **`unfreeze_epoch`** | $\{2, 3\}$ | **`3`** | Đóng băng Backbone trong 2 epochs đầu (Ep 1 & 2) để làm nóng các nhánh phụ mới. |
| **`weight_decay` / `dropout_rate**` | $[10^{-4}, 1.5 \times 10^{-2}]$ / $[0.25, 0.50]$ | **`0.00254` / `0.25**` | Mức chuẩn hóa vừa đủ vì bản thân 2 nhánh phụ đã đóng vai trò Regularizer mạnh. |
| **`theta_cd` / `gamma` / `alpha_real` / `smoothing**` | Kế thừa từ Stage 1 & 2 | **`0.40` / `2.0` / `0.55` / `0.08**` | Giữ nguyên cấu hình tích chập sai phân trung tâm và hàm Focal Loss. |

---

## 6. Kết quả Huấn luyện Toàn phần & Đánh giá Đối kháng 3 Mô hình theo Chuẩn ISO/IEC 30107-3

### 6.1. Diễn biến Huấn luyện Toàn phần (Full Multi-Task Training)

Mô hình **Model 3 (`MultiTaskFASNet`)** được huấn luyện với bộ siêu tham số tối ưu từ Trial #14 cùng cơ chế `ReduceLROnPlateau (patience=3, factor=0.5)` và `EarlyStopping (patience=15)`:

* **Tổng thời gian thực thi:** **69.4 phút** (dừng sớm tại **Epoch 26** sau 15 epochs chờ cải thiện tính từ đỉnh **Epoch 11**; mỗi epoch đóng băng mất ~127 giây và mỗi epoch mở khóa toàn mạng mất ~155–166 giây trên `xpu:0`).
* **Sự hội tụ đồng thời của cả 3 hàm mất mát (Không xảy ra xung đột đạo hàm):**
* **Giai đoạn Làm nóng (`Epoch 1 -> 2`, `FROZEN-WARMUP`, `LR_head = 0.001162`):** Ngay ở Epoch 1, mô hình đạt `Val ACER = 3.03%` (`AUC = 99.73%`). Sang Epoch 2, sai số của cả 3 nhánh giảm mạnh: `Cls Loss` giảm từ `0.0048` $\rightarrow$ `0.0020`, `Depth Loss (D)` giảm từ `0.1028` $\rightarrow$ `0.0397`, và `Noise Loss (N)` giảm từ `0.0725` $\rightarrow$ `0.0145`.
* **Giai đoạn Mở khóa Bậc 1 & Bậc 2 (`Epoch 3 -> 9`, `LR_head = 0.001162 -> 0.000581`):** Khi mở khóa toàn mạng ở Epoch 3, các nhánh bắt đầu điều chỉnh trọng số chung với Backbone. Sau lần giảm LR tại Epoch 6, `Train Loss` tổng tiếp tục hạ xuống `0.0207` (`Cls = 0.0009, D = 0.0226, N = 0.0087`) tại Epoch 9.
* **Giai đoạn Mở khóa Bậc 3 — Đạt Đỉnh Hiệu Năng (`Epoch 10 -> 15`, `LR_head = 0.000291`):** Ngay khi tốc độ học giảm xuống bậc thứ 3 ở Epoch 10, `Val ACER` giảm vọt xuống `3.22%` và lập **Kỷ lục Toàn cục tại Epoch 11 với `Val ACER = 2.92%` (`APCER = 2.87%`, `BPCER = 2.96%`, `ROC-AUC = 99.59%`, `Accuracy = 97.12%`)** ở ngưỡng quyết định tối ưu **$\tau^* = 0.1109$** (`TrLoss = 0.0174`, `Cls = 0.0005`, `D = 0.0186`, `N = 0.0078`).
* **Giai đoạn Hội tụ Sâu (`Epoch 16 -> 26`, `LR_head = 0.000145 -> 0.000036`):** Hai nhánh phụ tiếp tục hội tụ mịn về `D = 0.0147` và `N = 0.0069` trong khi `Val ROC-AUC` duy trì ổn định ở mức rất cao (`99.65% – 99.72%` từ Epoch 21 đến Epoch 26) trước khi dừng sớm an toàn.



### 6.2. Bảng Kết quả Đánh giá Chính thức của Model 3 trên Tập Validation & Test (7,580 ảnh)

Trọng số tốt nhất tại **Epoch 11** (`model3_multitask_best.pt`) được nạp độc lập cùng ngưỡng quyết định **$\tau^* = 0.1109$** để đánh giá trên **7,580 ảnh của tập Test (`LCC_FASD_evaluation`)** (thời gian đánh giá cả Val + Test chỉ mất **58.45 giây**):

| Tập Đánh Giá | Ngưỡng Quyết định ($\tau$) | APCER (%) ↓ *(Lọt giả mạo)* | BPCER (%) ↓ *(Từ chối thật)* | ACER (%) ↓ *(Lỗi TB)* | EER (%) ↓ *(Điểm cân bằng)* | ROC-AUC (%) ↑ | Accuracy (%) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Validation (Development)** | `0.1109` *(Val $\tau^*$)* | **2.87%** | **2.96%** | **2.92%** | **2.98%** | **99.59%** | **97.12%** |
| **Test (Áp ngưỡng $\tau^*$ từ Val)** | `0.1109` *(Cố định)* | **7.39%** | **6.37%** | **6.88%** | **6.94%** | **98.05%** | **92.65%** |
| **Test (Tại ngưỡng Test EER)** | `0.1234` *(Test EER)* | **6.87%** | **7.01%** | **6.94%** | **6.94%** | **98.05%** | **93.13%** |

### 6.3. Bảng Đối sánh Trực diện 3 Giai đoạn: Model 1 vs. Model 2 vs. Model 3

| Mô hình | Phương pháp & Kiến trúc | Val ACER (%) ↓ | Test APCER (%) ↓ | Test BPCER (%) ↓ | Test ACER (%) ↓ | Test EER (%) ↓ | Test Accuracy (%) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Model 1: `CustomFASNet**` | Train from Scratch (`RGB Only` - Stage 1) | `9.93%` | `9.51%` *(691 lỗi)* | `13.38%` *(42 lỗi)* | `11.44%` | `11.46%` | `90.33%` |
| **Model 2: `TransferFASNet**` | Pretrained Fine-tuned (`RGB Only` - Stage 2) | **`2.70%`** | `8.73%` *(634 lỗi)* | `6.69%` *(21 lỗi)* | `7.71%` | `7.93%` | `91.36%` |
| **Model 3: `MultiTaskFASNet**` | **Dual-Space (`RGB+YCrCb`) + `3D Depth` + `FFT/LBP Noise` (Stage 3)** | `2.92%` | **`7.39%` *(537 lỗi)*** | **`6.37%` *(20 lỗi)*** | **`6.88%`** | **`6.94%`** | **`92.65%`** |

#### Phân tích Chi tiết Ma trận Nhầm lẫn của Model 3 & Chứng minh Khả năng Kháng Domain Shift:

* **Ma trận Nhầm lẫn trên Tập Test (7,580 ảnh tại $\tau^* = 0.1109$):**
* **True Negative ($\text{TN} = 6,729$):** Chặn đứng chính xác **6,729 / 7,266** cuộc tấn công giả mạo (tỷ lệ chặn đúng đạt **92.61%**, chặn thành công nhiều hơn **+97 ca giả mạo** so với Model 2 và nhiều hơn **+154 ca** so với Model 1).
* **False Positive ($\text{FP} = 537$):** Số mẫu Spoof lọt lưới giảm mạnh từ `691` (Model 1) $\rightarrow$ `634` (Model 2) $\rightarrow$ **`537` (Model 3)** ($\text{APCER} = 7.39\%$).
* **True Positive ($\text{TP} = 294$):** Nhận diện đúng **294 / 314** mẫu người thật (tỷ lệ nhận đúng người thật đạt **93.63%**).
* **False Negative ($\text{FN} = 20$):** Chỉ còn đúng **20 mẫu người thật bị từ chối nhầm** trên tổng số 314 mẫu ($\text{BPCER} = 6.37\%$).


* **Minh chứng Thực nghiệm cho Lý thuyết Học Đa nhiệm (Multi-Task Regularization):**
* Ở Stage 2, Model 2 đạt `Val ACER = 2.70%` nhưng khi sang tập `Test` thì tăng lên `7.71%` (độ lệch tổng quát hóa $\Delta_{\text{Val}\rightarrow\text{Test}} = +5.01\%$).
* Ở Stage 3, mặc dù Model 3 có `Val ACER = 2.92%`, nhưng trên tập `Test` độc lập, Model 3 lại bứt phá lên vị trí dẫn đầu với **`Test ACER = 6.88%`** và **`Test EER = 6.94%`** (thu hẹp độ lệch $\Delta_{\text{Val}\rightarrow\text{Test}}$ xuống chỉ còn **`+3.96%`**). Điều này chứng minh rằng việc buộc mạng dự đoán thêm bản đồ `3D Depth` và `FFT/LBP Noise` trên không gian `RGB + YCrCb` đã ngăn chặn hiệu quả hiện tượng học vẹt vào tập Validation, giúp mô hình tổng quát hóa vượt trội khi gặp điều kiện thu nhận mới.



---

## 7. Giải thích Vật lý (Explainable AI) & Phân tích Lỗi Thực tế trên Tập Test

### 7.1. Khả năng Tự Tái tạo Bản đồ 3D Depth & Noise lúc Suy luận (`multitask_predictions_stage3.png`)

Khi chạy trên tập Test (không còn Teacher `MiDaS_small` hay hàm `FFT/LBP`), bản thân tệp `model3_multitask_best.pt` tự động xuất ra các bằng chứng vật lý cực kỳ trực quan:

1. **Tác dụng vạch trần vân sóng màn hình của kênh `YCrCb (Cr)`:** Ở hai mẫu giả mạo `spoof_2201.png` và `spoof_2206.png`, trong khi ảnh RGB gốc trông rất tự nhiên thì trên kênh sắc độ `YCrCb (Cr)`, các **dải sọc giao thoa thẳng đứng (Vertical Moiré Stripes)** của màn hình phát lại hiện lên rõ rệt dọc toàn bộ khuôn mặt.
2. **Sự phân cực rõ nét của 2 nhánh giải mã `Depth` và `Noise`:**
* **Trên mẫu người thật (`real_90.png`, `real_107.png`):** Model 3 tự tái tạo bản đồ `3D Depth` nổi khối sáng rõ (`Mean = 0.74` và `0.80`, `Max = 1.00`) và dự đoán bản đồ `Noise/Moiré` tối đen tuyệt đối (`Mean = 0.00, Max = 0.00 – 0.01`), đưa ra xác suất `P(Real) = 1.0000`.
* **Trên mẫu giả mạo (`spoof_2201.png`, `spoof_2206.png`):** Model 3 dự đoán bản đồ `3D Depth` phẳng bằng `0` (`Mean = 0.00, Max = 0.00`) và kích hoạt bản đồ nhiệt `Noise/Moiré` sáng rực khắp khuôn mặt (`Mean = 0.22 – 0.24`, `Max = 0.86 – 0.90`), đưa ra xác suất `P(Real) = 0.0000`.



### 7.2. Phân tích 10 Ca Lỗi Nặng Nhất trên Tập Test (`error_analysis_stage3.png`)

1. **Khắc phục triệt để lỗi kính râm ngoài nắng của Stage 2:** Ở Stage 2, mẫu người thật đeo kính râm đen ngoài nắng (`real_69.png`) nằm trong nhóm bị từ chối nhầm nặng nhất. Sang Stage 3, nhờ nhánh 3D Depth bảo toàn thông tin hình khối khuôn mặt thật quanh kính, trường hợp này đã được phân loại đúng và biến mất khỏi nhóm lỗi nặng nhất.
2. **Nguyên nhân của 5 ca lỗi `False Negative (Real -> Spoof)` nặng nhất còn lại (`FN = 20`):**
* Có tới 4/5 mẫu (`real_48.png`, `real_51.png`, `real_47.png`, `real_49.png` với $P(\text{Real}) \in [0.001, 0.003]$) là các khung hình liên tiếp của **cùng một chủ thể (1 video duy nhất)** bị phủ lớp mờ sương trắng (low-contrast haze) kèm bức tranh in nét đen trắng sắc cạnh ở góc trên bên phải gây kích hoạt nhầm bộ dò tần số cao.
* Mẫu thứ 5 (`real_152.png`, $P(\text{Real}) = 0.003$) là khuôn mặt bị đèn flash đánh trực diện làm bẹt khối sáng kết hợp phông nền kẻ ô vuông bàn cờ (checkerboard) phía sau lưng có tần số không gian trùng với lưới pixel màn hình.


3. **Nguyên nhân của 5 ca lỗi `False Positive (Spoof -> Real)` nặng nhất (`FP = 537`):**
* Các mẫu `spoof_5222.png` ($P(\text{Real}) = 0.996$), `spoof_7274.png` ($0.994$), `spoof_6818.png` ($0.994$), `spoof_695.png` ($0.978$) và `spoof_7275.png` ($0.977$) là các cuộc tấn công phát lại trên màn hình độ phân giải cao cắt sát khuôn mặt (trong đó `spoof_7274` và `spoof_7275` thuộc cùng 1 chuỗi video có dính con trỏ chuột trắng rất nhỏ ở góc dưới cằm). Đây chính là cơ sở thực tiễn để tiếp tục mở rộng miền dữ liệu sang **CelebA-Spoof** ở Giai đoạn 4.



---

## 8. Danh mục Tệp Đầu ra (Saved Artifacts) & Định hướng Giai đoạn 4 & 5

### 8.1. Các tệp đã lưu trữ sau khi hoàn tất Giai đoạn 3

* `../models/model3_multitask_best.pt`: **Tệp trọng số tự chứa duy nhất để Deploy** (Epoch 11, `Val ACER = 2.92%`, `Test ACER = 6.88%`, `Test EER = 6.94%`, `Test AUC = 98.05%`), đóng gói đầy đủ `model_state_dict`, `model_config`, `best_params`, và `val_threshold = 0.1109`.
* `../models/model3_multitask_latest.pt`: Checkpoint trạng thái cuối cùng tại thời điểm dừng sớm (Epoch 26).
* `../checkpoint/stage3/init_screening_state_stage3.json` & `init_screening_results_stage3.csv`: Kết quả Pha 1 đọ sức chế độ khởi tạo Backbone.
* `../checkpoint/stage3/optuna_state_stage3.json`, `optuna_trials_stage3.csv` & `best_params_stage3.json`: Lịch sử 16 trials và bộ siêu tham số đa nhiệm tối ưu (Trial #14).
* `../checkpoint/stage3/training_state_stage3.json` & `train_history_stage3.csv`: Nhật ký huấn luyện chi tiết của cả 3 hàm Loss và các chỉ số ISO/IEC 30107-3 qua từng epoch.
* `../checkpoint/stage3/test_evaluation_stage3.json`: Báo cáo đánh giá chuẩn ISO/IEC 30107-3 trên tập Validation và Test.
* `../checkpoint/stage3/training_curves_stage3.png`, `test_diagnostics_stage3.png`, `multitask_predictions_stage3.png`, `error_analysis_stage3.png`: Bộ biểu đồ hội tụ đa nhiệm, chẩn đoán ROC/EER, giải thích vật lý 3D/Noise và phân tích lỗi.

### 8.2. Định hướng Triển khai Tiếp theo (Live Camera App & Giai đoạn 4/5)

1. **Triển khai Ứng dụng Camera Thời gian thực (Live Deployment):** Nạp trực tiếp tệp `../models/model3_multitask_best.pt` vào giao diện Webcam real-time. Hệ thống hỗ trợ cả 2 chế độ: **Chế độ Tốc độ cao** (`return_maps=False`, chỉ chạy nhánh phân loại) và **Chế độ Trực quan hóa Vật lý** (`return_maps=True`, hiển thị trực tiếp bản đồ 3D Depth và bản đồ nhiệt Noise/Moiré theo thời gian thực trên màn hình).
2. **Giai đoạn 4 & 5 (Huấn luyện Đa miền trên `CelebA-Spoof` & Đánh giá Cross-Dataset):** Sử dụng `model3_multitask_best.pt` làm điểm khởi tạo để huấn luyện tiếp trên tập dữ liệu lớn **CelebA-Spoof** tạo ra **Model 4**, đồng thời đưa cả 4 mô hình vào khung đánh giá chéo (Cross-dataset evaluation) trên các miền dữ liệu chưa từng gặp.