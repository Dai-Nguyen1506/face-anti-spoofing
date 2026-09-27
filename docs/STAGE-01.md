# STAGE 01: Xây dựng & Huấn luyện Mô hình Custom CNN (Model 1) cho Nhận diện Gian lận Quét mặt (Face Anti-Spoofing)

## 1. Tổng quan & Mục đích (Overview & Purpose)

Tài liệu này mô tả chi tiết quy trình thiết kế, tối ưu hóa phần cứng, huấn luyện và đánh giá **Giai đoạn 1 (`01_Basic_CNN.ipynb`)** thuộc dự án *Hệ thống CNN Học sâu cho Chống giả mạo Khuôn mặt Đa môi trường*.

* **Mục đích cốt lõi:** Xây dựng một mạng tích chập sâu tự thiết kế từ đầu (**Custom CNN from scratch** — không sử dụng trọng số tiền huấn luyện/pretrained backbone) nhằm giải quyết bài toán phân loại nhị phân giữa người thật (**Real / Bona Fide = `1**`) và hành vi giả mạo (**Spoof / Presentation Attack = `0**`).


* **Vai trò trong toàn bộ dự án:** Thiết lập mốc hiệu năng cơ sở (Baseline) vững chắc trên bộ dữ liệu **LCC-FASD** và đóng gói trọng số tốt nhất (**Model 1**) làm ứng viên bộ mã hóa đặc trưng (**Candidate Encoder**) cho **Giai đoạn 2** và **Giai đoạn 3**.

---

## 2. Mục tiêu Kỹ thuật (Technical Objectives)

1. **Tối ưu hóa phần cứng Local (Intel iGPU/XPU & 16GB RAM):** Khai thác tối đa sức mạnh của backend `PyTorch 2.14.0+xpu` trên GPU tích hợp Intel Graphics thông qua cơ chế tính toán hỗn hợp `bfloat16` (Mixed Precision), định dạng bộ nhớ `channels_last` và quản lý bộ nhớ chia sẻ (`Shared Memory`).

2. **Triệt tiêu nút thắt đọc dữ liệu (I/O Bottleneck):** Loại bỏ hoàn toàn độ trễ đọc đĩa cứng trong quá trình tìm kiếm siêu tham số và huấn luyện dài ngày mà không gây tràn bộ nhớ (Out-Of-Memory) trên hệ thống 16GB RAM.

3. **Giải quyết triệt để hiện tượng mất cân bằng lớp (Extreme Class Imbalance):** Xử lý tỷ lệ lệch nhãn nghiêm trọng giữa ảnh Spoof và ảnh Real trong tập LCC-FASD ở cả tầng nạp dữ liệu lẫn tầng hàm mục tiêu.

4. **Tối ưu siêu tham số tự động (Automated Hyperparameter Tuning):** Sử dụng **Optuna** với thuật toán **TPE (Tree-structured Parzen Estimator)** và cơ chế cắt tỉa sớm **Median Pruner** để tìm ra cấu hình mạng và tham số học tốt nhất.

5. **Đánh giá chuẩn quốc tế ISO/IEC 30107-3:** Đánh giá mô hình dựa trên bộ chỉ số chuyên ngành chống giả mạo sinh trắc học gồm **APCER**, **BPCER**, **ACER**, **EER** và **ROC-AUC**.

---

## 3. Phân tích Dữ liệu & Chiến lược Tiền xử lý (Data Pipeline)

### 3.1. Cấu trúc và Phân phối Tập dữ liệu LCC-FASD

Bộ dữ liệu **LCC-FASD** tại đường dẫn cục bộ `../dataset/LCC_FASD/` được chia cố định thành 3 tập với sự mất cân bằng lớn nghiêng về lớp giả mạo (`Spoof`):

| Tập dữ liệu (Split) | Thư mục nguồn | Tổng số ảnh | Số ảnh Real (`1`) | Số ảnh Spoof (`0`) | Tỷ lệ mất cân bằng (Spoof : Real) |
| --- | --- | --- | --- | --- | --- |
| **Training** | `LCC_FASD_training`<br> | **8,299**<br> | 1,223 (14.7%)

 | 7,076 (85.3%)

 | **5.79 : 1** |
| **Validation (Dev)** | `LCC_FASD_development`<br> | **2,948**<br> | 405 (13.7%)

 | 2,543 (86.3%)

 | **6.28 : 1** |
| **Evaluation (Test)** | `LCC_FASD_evaluation`<br> | **7,580**<br> | 314 (4.1%)

 | 7,266 (95.9%)

 | **23.14 : 1** |
| **Tổng cộng** | — | **18,827**<br> | **1,942 (10.3%)** | **16,885 (89.7%)** | **8.69 : 1** |

### 3.2. Các Hành động Cải tiến trên Pipeline Dữ liệu

* **Kích thước ảnh chuẩn (`IMG_SIZE = 192x192`):** Giữ lại đầy đủ các chi tiết vi kết cấu bề mặt (vân sóng màn hình Moiré, rìa giấy in, phản xạ ánh sáng) trong khi giảm ~30% chi phí tính toán so với kích thước `224x224`.

* **Tiền nạp đa luồng vào Bộ nhớ Chia sẻ (`Shared Memory Tensor Cache`):**
* Thay vì sử dụng `dict` Python thông thường (dễ bị nhân bản bộ nhớ theo cơ chế *Copy-on-Write* khi chạy đa tiến trình `num_workers = 4` trên Linux), hệ thống sử dụng `ThreadPoolExecutor` (12 luồng CPU) để giải mã bằng `cv2`, chuyển hệ màu RGB và resize về `192x192` ngay từ đầu.

* Toàn bộ tập **Train (8,299 ảnh)** được nạp vào khối `torch.uint8` chia sẻ trong **9.32 giây** (chiếm cố định **875.3 MB RAM**); tập **Val (2,948 ảnh)** được nạp trong **3.43 giây** (chiếm cố định **310.9 MB RAM**).
* Riêng tập **Test (7,580 ảnh)** chỉ sử dụng 1 lần ở bước đánh giá cuối cùng nên được đọc trực tiếp từ ổ cứng, giúp tiết kiệm ~800 MB RAM cho GPU tích hợp `xpu:0` (có 13.24 GB bộ nhớ hợp nhất khả dụng).

* **Cân bằng Mini-Batch bằng `WeightedRandomSampler`:** Gán trọng số lấy mẫu tỷ lệ nghịch với tần suất xuất hiện của từng lớp ($w_c = \frac{1}{N_c}$) trên tập Train, đảm bảo mỗi mini-batch đưa vào mạng luôn có phân phối cân bằng xấp xỉ **50% Real – 50% Spoof**.
* **Tăng cường dữ liệu chuyên biệt (FAS-Specific Augmentation):** Sử dụng thư viện `Albumentations` với các phép biến đổi:
* Hình học: `HorizontalFlip (p=0.5)`, `ShiftScaleRotate (p=0.4)`.

* Quang học & Nhiễu: `ColorJitter (p=0.4)` chống học vẹt màu sắc môi trường; `OneOf([GaussianBlur, ImageCompression, GaussNoise], p=0.35)` mô phỏng camera mờ, nén video phát lại và nhiễu cảm biến.

* Cắt bỏ cục bộ: `CoarseDropout (p=0.25)` che ngẫu nhiên 1–4 vùng nhỏ ($12\times12$ đến $24\times24$ pixel), buộc mạng học kết cấu da mặt trên toàn bộ khung hình.

---

## 4. Phương pháp luận & Kiến trúc Mô hình

### 4.1. Kiến trúc `CustomFASNet` (CDCN-Lite + Residual + SE-Attention)

Mô hình được thiết kế chuyên biệt cho bài toán chống giả mạo khuôn mặt với cấu trúc tách biệt rõ phần **Encoder (`extract_features`)** và phần **Head phân loại (`classifier`)**:

1. **Tích chập Sai phân Trung tâm (`Conv2d_CD` - Central Difference Convolution):**
* Kết hợp giữa tích chập cường độ điểm ảnh truyền thống và tích chập đạo hàm bậc nhất vùng lân cận với hệ số điều chỉnh $\theta \in [0, 1]$:

$$y(p_0) = \underbrace{\sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0 + p_n)}_{\text{Vanilla Convolution}} - \theta \cdot \underbrace{x(p_0) \sum_{p_n \in \mathcal{R}} w(p_n)}_{\text{Central Difference}}$$

* Giúp mạng cực kỳ nhạy bén với các đường gợn sóng màn hình điện thoại và hạt mực in trên giấy.

2. **Khối `ResidualCDBlock` tích hợp `SEBlock` (Squeeze-and-Excitation):**
* Gồm 4 tầng (Stages) giảm chiều không gian từ $192\times192 \rightarrow 96\times96 \rightarrow 48\times48 \rightarrow 24\times24 \rightarrow 12\times12 \rightarrow 6\times6$ với độ rộng kênh tăng dần $C \rightarrow 2C \rightarrow 4C \rightarrow 8C$.
* Mỗi khối sử dụng hàm kích hoạt `SiLU (Swish)`, chuẩn hóa `BatchNorm2d`, kết nối tắt `Residual Shortcut` và khối chú ý kênh `SEBlock` (tỷ lệ nén `reduction = 8`).

3. **Dual Pooling Feature Extractor (`extract_features`):**
* Kết hợp song song `AdaptiveAvgPool2d(1)` (nắm bắt đặc trưng ngữ cảnh toàn cục) và `AdaptiveMaxPool2d(1)` (bắt các tín hiệu cực trị như đốm lóa sáng phản quang).
* Với `base_channels = 48` (tầng cuối $8C = 384$ kênh), vector đặc trưng đầu ra của Encoder có kích thước **`768` chiều** ($384 \times 2$).

### 4.2. Hàm Mục tiêu `FocalLossWithSmoothing`

Để phối hợp nhịp nhàng với `WeightedRandomSampler` mà không gây hiện tượng phạt kép (double-counting), hàm mất mát kết hợp **Focal Loss** và **Label Smoothing**:

$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \sum_{c=0}^{1} \tilde{y}_c \log(p_c)$$

Trong đó:

* $\tilde{y}_c = (1 - \varepsilon)y_c + \frac{\varepsilon}{2}$ là nhãn đã được làm mượt với hệ số `label_smoothing` $\varepsilon$, ngăn chặn mô hình trở nên quá tự tin (overconfident) trên tập Train.
* $(1 - p_t)^\gamma$ là trọng số tập trung phạt nặng các mẫu khó phân loại (hard examples).
* $\alpha_t$ là trọng số cân bằng nhẹ giữa lớp Real ($\alpha_{\text{real}}$) và lớp Spoof ($1 - \alpha_{\text{real}}$).

---

## 5. Tối ưu Siêu tham số với Optuna (Hyperparameter Tuning)

* **Thiết lập:** Chạy 20 trials $\times$ 6 epochs/trial trực tiếp trên bộ nhớ (`In-Memory`, không phụ thuộc SQLite), sử dụng thuật toán **TPE** và bộ cắt tỉa **Median Pruner** theo mục tiêu **tối thiểu hóa `Val ACER**`.

* **Thời gian thực thi:** 208.2 phút.
* **Kết quả hội tụ:** Kể từ Trial 14 đến Trial 19, thuật toán TPE hội tụ chặt chẽ về cấu hình `batch_size = 32` và `base_channels = 48`. **Trial 15** đạt kỷ lục tốt nhất với **`Val ACER = 15.06%`** (`AUC = 0.8984`) chỉ sau 6 epochs và được xuất ra tệp `best_params_stage1.json`:

| Siêu tham số (Hyperparameter) | Khoảng tìm kiếm (Search Space) | Giá trị Tối ưu (Trial 15) | Ý nghĩa Kỹ thuật |
| --- | --- | --- | --- |
| **`lr` (Learning Rate)** | $[10^{-4}, 3 \times 10^{-3}]$ (log) | **`0.001454`** | Tốc độ học khởi điểm lý tưởng trước khi giảm theo bậc thang. |
| **`weight_decay`** | $[10^{-5}, 10^{-2}]$ (log) | **`0.009257`** | Phạt chuẩn hóa $L_2$ mạnh giúp chống học vẹt các mẫu Real lặp lại. |
| **`batch_size`** | $\{32, 64\}$ | **`32`** | Cập nhật trọng số 259 lần/epoch, thoát cực trị địa phương tốt hơn. |
| **`base_channels`** | $\{32, 48\}$ | **`48`** | Mở rộng mạng lên ~3.1 triệu tham số, xuất vector đặc trưng 768 chiều. |
| **`dropout_rate`** | $[0.15, 0.45]$ | **`0.35`** | Tỷ lệ ngắt kết nối ở tầng Classifier giúp tăng tính tổng quát hóa. |
| **`theta` (CDCN ratio)** | $[0.4, 0.8]$ | **`0.40`** | Pha trộn 40% đạo hàm vi bề mặt + 60% tích chập chuẩn, tránh khuếch đại nhiễu. |
| **`focal_gamma` ($\gamma$)** | $[1.0, 2.5]$ | **`2.00`** | Mức tập trung chuẩn mực vào các mẫu biên khó phân biệt. |
| **`alpha_real` ($\alpha_1$)** | $[0.50, 0.65]$ | **`0.55`** | Ưu tiên nhẹ (55% : 45%) cho lớp Real nhằm kéo giảm chỉ số BPCER. |
| **`label_smoothing` ($\varepsilon$)** | $[0.02, 0.10]$ | **`0.08`** | Làm mượt nhãn ở mức 8%, giúp phân phối xác suất đầu ra êm hơn. |

---

## 6. Kết quả Huấn luyện Toàn phần & Đánh giá Chuẩn ISO/IEC 30107-3

### 6.1. Quá trình Huấn luyện Toàn phần (Full Training)

Mô hình `CustomFASNet` được khởi tạo với bộ tham số tối ưu từ Trial 15 và huấn luyện với cấu hình `FULL_EPOCHS = 50`, `EARLY_STOP_PATIENCE = 12`, kết hợp bộ điều chỉnh tốc độ học `ReduceLROnPlateau (factor=0.5, patience=3)` và cơ chế lưu **2 Checkpoint song song** (`model1_custom_cnn_latest.pt` cập nhật mỗi epoch và `model1_custom_cnn_best.pt` chỉ cập nhật khi lập kỷ lục `Val ACER` mới):

* **Thời gian hoàn tất:** 102.0 phút (trung bình ~125 giây/epoch cho cả Train + Val trên `xpu:0`).
* **Diễn biến hội tụ theo bậc thang Learning Rate**:

* **Giai đoạn 1 (`Epoch 1 -> 8`, `LR = 0.00145`):** `Val ACER` giảm từ `20.81%` (Epoch 1) xuống **`15.98%`** (`AUC = 0.9097` tại Epoch 4).


* **Giai đoạn 2 (`Epoch 9 -> 17`, `LR = 0.00073`):** Sau lần giảm LR đầu tiên ở cuối Epoch 8, mô hình liên tiếp lập kỷ lục mới tại Epoch 10 (`14.58%`), Epoch 11 (`14.04%`) và Epoch 13 (**`13.69%`**, `AUC = 0.9359`).

* **Giai đoạn 3 (`Epoch 18 -> 34`, `LR = 0.00036`):** Tiếp tục bứt phá mạnh mẽ tại Epoch 21 (`12.44%`), Epoch 25 (`12.11%`), Epoch 28 (`11.86%`) và Epoch 30 (**`11.32%`**, `AUC = 0.9643`).

* **Giai đoạn 4 (`Epoch 35 -> 49`, `LR = 0.00018 -> 0.000023`):** Dao động răng cưa của `Val Loss` được triệt tiêu hoàn toàn. Mô hình phá mốc 10% tại Epoch 36 (`10.88%`) và đạt **đỉnh hiệu năng tại Epoch 37 với `Val ACER = 9.93%` (`APCER = 9.99%`, `BPCER = 9.88%`, `AUC = 0.9627`)** trước khi dừng sớm (Early Stopping) tại Epoch 49.

### 6.2. Bảng Kết quả Đánh giá Chính thức (Validation & Test)

Trọng số tốt nhất từ **Epoch 37** (`model1_custom_cnn_best.pt`) được nạp lại cùng ngưỡng quyết định tối ưu xác lập trên tập Validation là **$\tau^* = 0.3877$** để đánh giá độc lập trên **7,580 ảnh của tập Test (`LCC_FASD_evaluation`)**:

| Tập Đánh Giá | Ngưỡng Quyết định ($\tau$) | APCER (%) ↓ *(Lọt giả mạo)* | BPCER (%) ↓ *(Từ chối thật)* | ACER (%) ↓ *(Lỗi TB)* | EER (%) ↓ *(Điểm cân bằng)* | ROC-AUC (%) ↑ | Accuracy (%) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Validation (Development)** | `0.3877` *(Val EER)* | **9.99%** | **9.88%** | **9.93%** | **9.93%** | **96.27%**<br> | **90.03%** |
| **Test (Áp ngưỡng $\tau^*$ từ Val)** | `0.3877` *(Cố định)*<br> | **9.51%** | **13.38%** | **11.44%** | **11.46%** | **95.76%**<br> | **90.33%** |
| **Test (Tại ngưỡng Test EER)** | `0.3174` *(Test EER)* | **11.46%** | **11.46%** | **11.46%** | **11.46%** | **95.76%**<br> | **88.54%** |

### 6.3. Phân tích Chi tiết Ma trận Nhầm lẫn & Đường cong Chẩn đoán trên Tập Test

* **Ma trận nhầm lẫn (Confusion Matrix tại $\tau^* = 0.388$)**:

* **True Negative ($\text{TN} = 6,575$):** Chặn đứng chính xác **6,575 / 7,266** cuộc tấn công giả mạo (đạt tỷ lệ chặn đúng **90.49%**).

* **False Positive ($\text{FP} = 691$):** Có 691 mẫu Spoof bị lọt lưới ($\text{APCER} = 9.51\%$).

* **True Positive ($\text{TP} = 272$):** Nhận diện đúng **272 / 314** mẫu người thật (đạt tỷ lệ nhận đúng **86.62%**).

* **False Negative ($\text{FN} = 42$):** Có 42 mẫu người thật bị từ chối nhầm ($\text{BPCER} = 13.38\%$).




* **Sự ổn định giữa Validation và Test:** Chỉ số `ROC-AUC` trên tập Validation (`96.27%`) và tập Test (`95.76%`) chỉ chênh lệch **0.51%**, đồng thời `ACER` khi áp ngưỡng cố định từ Val (`11.44%`) gần như trùng khớp với mức `EER` nội tại của tập Test (`11.46%`), khẳng định mô hình có khả năng tổng quát hóa rất cao và không bị quá khớp (Overfitting).

---

## 7. Phân tích Lỗi Thực tế trên Tập Test (Visual Error Analysis)

Qua trực quan hóa các mẫu bị phân loại nhầm trong tệp `error_analysis_stage1.png`, các nguyên nhân vật lý gây lỗi được xác định rõ ràng như sau:

1. **Nguyên nhân gây lỗi `False Positive (Spoof -> Real)` (Nhận nhầm giả mạo là thật)**:

* **Hiện tượng phơi sáng quá mức / Cháy sáng trắng (Overexposure):** Ở các mẫu có $P(\text{Real}) = 0.793, 0.837, 0.492$, ánh sáng gắt làm vùng da mặt trên ảnh giả bị bão hòa thành mảng trắng mịn, xóa nhòa các vân sóng màn hình (Moiré) và rìa hạt mực, khiến tầng tích chập `Conv2d_CD` không thu được tín hiệu tần số cao của vật liệu giả.

* **Mờ chuyển động (Motion Blur) và độ phân giải cực thấp:** Mẫu có $P(\text{Real}) = 0.633$ bị nhòe nặng làm mất đặc trưng cạnh vi bề mặt.

2. **Nguyên nhân gây lỗi `False Negative (Real -> Spoof)` (Từ chối nhầm người thật)**:

* **Phản xạ gương trên mắt kính râm (Specular Reflection on Sunglasses):** Ở các mẫu có $P(\text{Real}) = 0.030, 0.025$, người thật đeo kính râm đen bóng loáng ngoài trời nắng gắt. Đặc tính phản quang trên bề mặt tròng kính râm tương tự như hiện tượng lóa sáng trên mặt kính màn hình điện thoại/iPad trong tấn công phát lại (Replay Attack), làm mô hình đánh giá xác suất giả mạo rất cao.

* **Nhiễu hậu cảnh có họa tiết in ấn (Printed Background Artifacts):** Ở mẫu có $P(\text{Real}) = 0.244$, người thật đứng trước bức tranh vẽ nhân vật hoạt hình có nét in sắc cạnh, kích hoạt nhầm các bộ lọc dò tìm tấn công ảnh in (Print Attack).

* **Sự xuất hiện của ngón tay ở rìa ảnh:** Ở mẫu có $P(\text{Real}) = 0.068$, bàn tay người thật đưa lên sát vùng tóc tạo hình thái giống hành vi cầm giữ tấm ảnh giả trước ống kính.

---

## 8. Danh mục Tệp Đầu ra (Saved Artifacts) & Định hướng Giai đoạn 2 & 3

### 8.1. Các tệp đã lưu trữ sau khi hoàn tất Giai đoạn 1

* `../models/model1_custom_cnn_best.pt`: Checkpoint tốt nhất (Epoch 37, `Val ACER = 9.93%`, `Test ACER = 11.44%`), lưu đầy đủ `model_state_dict`, `model_config`, `best_params`, và `val_threshold = 0.3877`.
* `../models/model1_custom_cnn_latest.pt`: Checkpoint trạng thái cuối cùng tại thời điểm dừng huấn luyện (Epoch 49).
* `../checkpoint/stage1/best_params_stage1.json`: Bộ siêu tham số tối ưu tìm được từ Optuna Trial 15.
* `../checkpoint/stage1/train_history_stage1.csv`: Bảng nhật ký chỉ số qua toàn bộ 49 epochs.
* `../checkpoint/stage1/test_evaluation_stage1.json`: Báo cáo số liệu đánh giá ISO/IEC 30107-3 trên tập Validation và Test.
* `../checkpoint/stage1/training_curves_stage1.png`, `test_diagnostics_stage1.png`, `error_analysis_stage1.png`: Bộ biểu đồ phục vụ báo cáo.

### 8.2. Định hướng cho Giai đoạn 2 & 3

* Tái sử dụng phương thức `model1.extract_features(x)` (trích xuất vector đặc trưng `768` chiều từ `model1_custom_cnn_best.pt`) làm ứng viên Encoder cơ sở.

* Kết hợp thêm các kiến trúc Pretrained Backbone có khả năng nhận thức ngữ cảnh bậc cao (High-level semantic context) và đánh giá chéo đa tập dữ liệu (Cross-dataset evaluation) để xử lý triệt để các trường hợp ngoại lệ như đeo kính râm phản quang hay phơi sáng gắt ngoài trời.
