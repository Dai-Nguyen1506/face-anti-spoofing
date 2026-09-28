# STAGE 02: Huấn luyện Chuyển giao (Transfer Learning - Model 2) & Lựa chọn Backbone Tối ưu cho Nhận diện Gian lận Quét mặt (Face Anti-Spoofing)

## 1. Tổng quan & Mục đích (Overview & Purpose)

Tài liệu này mô tả chi tiết quy trình sàng lọc ứng viên, tối ưu hóa siêu tham số, huấn luyện tinh chỉnh hai giai đoạn (Two-Stage Fine-Tuning) và đánh giá **Giai đoạn 2 (`02_Transfer_Learning.ipynb`)** thuộc dự án *Hệ thống CNN Học sâu cho Chống giả mạo Khuôn mặt Đa môi trường tích hợp Học đa nhiệm (Multi-task Learning)*.

* **Mục đích cốt lõi:** Khảo sát và tinh chỉnh các mạng khung xương tiền huấn luyện trên ImageNet (**Pretrained Backbones**) cho bài toán phân loại nhị phân giữa người thật (**Real / Bona Fide = `1`**) và hành vi giả mạo (**Spoof / Presentation Attack = `0`**) nhằm khắc phục hạn chế về nhận thức ngữ cảnh bậc cao (High-level semantic context) của mô hình tự thiết kế từ đầu ở Giai đoạn 1.
* **Vai trò trong toàn bộ dự án:**
  1. Sàng lọc thực nghiệm 3 họ kiến trúc Pretrained kinh điển (**MobileNetV3**, **EfficientNet**, **ResNet**) trên cùng tập dữ liệu **LCC-FASD** để chọn ra mô hình Học chuyển giao tốt nhất (**Model 2**).
  2. Đối chiếu trực diện hiệu năng chuẩn quốc tế **ISO/IEC 30107-3** giữa **Model 1 (`CustomFASNet`)** và **Model 2 (`TransferFASNet`)** trên tập Test độc lập (7,580 ảnh) để quyết định **Bộ mã hóa đặc trưng dùng chung (Shared Encoder)** cho kiến trúc Đa nhiệm (**Model 3 - Multi-task Learning**) ở Giai đoạn 3.

---

## 2. Mục tiêu Kỹ thuật (Technical Objectives)

1. **Tương thích & Tối ưu hóa Phần cứng Local (`Intel iGPU/XPU` & `16GB RAM`):** Đảm bảo các kiến trúc Pretrained của `torchvision` vận hành ổn định tuyệt đối với cơ chế tính toán hỗn hợp `bfloat16` (`AMP`) và định dạng bộ nhớ `channels_last` trên `xpu:0` thông qua việc vô hiệu hóa các phép biến đổi tại chỗ (`inplace=False`) trong các tầng kích hoạt (`Hardswish`, `Hardsigmoid`, `SiLU`, `ReLU`).
2. **Kế thừa Pipeline Dữ liệu & Hàm Mục tiêu từ Stage 1:** Giữ nguyên cơ chế nạp đa luồng `Shared Memory Tensor` trên RAM (~1.18 GB cho Train + Val ở kích thước `192x192`), bộ lấy mẫu cân bằng lớp `WeightedRandomSampler` (tỷ lệ 50/50) và hàm mất mát `FocalLossWithSmoothing` ($\gamma = 2.0, \alpha_{\text{real}} = 0.55, \varepsilon = 0.08$) để đảm bảo tính nhất quán khi so sánh với Model 1.
3. **Chiến lược Tinh chỉnh Hai Giai đoạn (Two-Stage Fine-Tuning & Differential Learning Rate):**
   * **Giai đoạn Làm nóng (`FROZEN-WARMUP`):** Đóng băng toàn bộ trọng số Backbone và khóa thống kê `BatchNorm2d` ở chế độ `.eval()` trong các epoch đầu nhằm huấn luyện hội tụ nhanh tầng `Classifier Head` mới, ngăn ngừa hiện tượng xung đột đạo hàm lớn làm hỏng trọng số tiền huấn luyện ImageNet.
   * **Giai đoạn Mở khóa Toàn mạng (`UNFROZEN-FINETUNE`):** Mở khóa toàn bộ Backbone và áp dụng **Tốc độ học phân tầng (Differential Learning Rate)**: $\text{LR}_{\text{backbone}} = \eta_{\text{ratio}} \times \text{LR}_{\text{head}}$ kết hợp bộ giảm tốc độ học `ReduceLROnPlateau`.
4. **Quy trình 3 Pha tích hợp Cơ chế Checkpoint Thông minh (Smart Resumable Checkpointing):** Tự động đối chiếu chữ ký cấu hình (`config_signature`), lưu trữ trạng thái qua từng ứng viên sàng lọc, từng Trial của Optuna và từng Epoch huấn luyện (`latest.pt` & `best.pt`).

---

## 3. Phương pháp luận & Kiến trúc Mô hình (`TransferFASNet`)

### 3.1. Ba Ứng viên Pretrained Backbones Phù hợp cho Phần cứng Tích hợp

Để đảm bảo tốc độ thực thi thời gian thực (Real-time eKYC) và tối ưu hóa bộ nhớ trên Intel XPU, ba kiến trúc đại diện cho 3 trường phái thiết kế mạng tích chập hiện đại được lựa chọn khảo sát:

1. **`mobilenet_v3_large` (~4.2M tham số):** Sử dụng khối tích chập tách chiều sâu đảo ngược (*Inverted Residual with Depthwise Separable Convolutions*), tích hợp cơ chế chú ý kênh *Squeeze-and-Excitation (SE)* và hàm kích hoạt phi tuyến `Hardswish`. Xuất bản đồ đặc trưng không gian $6\times6$ với $C = 960$ kênh.
2. **`efficientnet_b0` (~4.0M tham số):** Đại diện họ kiến trúc co giãn phức hợp (*Compound Scaling*) sử dụng khối `MBConv` tích hợp `SE-Attention` và hàm kích hoạt `SiLU`. Xuất bản đồ đặc trưng không gian $6\times6$ với $C = 1280$ kênh.
3. **`resnet34` (~21.3M tham số):** Đại diện họ kiến trúc kết nối tắt kinh điển (*Deep Residual Learning*) sử dụng các khối tích chập chuẩn $3\times3$. Xuất bản đồ đặc trưng không gian $6\times6$ với $C = 512$ kênh.

### 3.2. Thiết kế Khối Bọc Thống nhất (`TransferFASNet`) Chuẩn bị cho Giai đoạn 3

Cả 3 ứng viên được đóng gói trong cấu trúc lớp `TransferFASNet` thống nhất với 3 tầng chức năng tách biệt rõ ràng:

* **`forward_feature_map(x)`:** Trích xuất trực tiếp Tensor bản đồ đặc trưng không gian 2D kích thước $(B, C, 6, 6)$ từ tầng cuối của Backbone (sẵn sàng để gắn thêm **Depth Regression Head** và **Noise/Reflection Head** ở Giai đoạn 3).
* **`extract_features(x)` (Dual Pooling Feature Extractor):** Kết hợp song song `AdaptiveAvgPool2d(1)` và `AdaptiveMaxPool2d(1)` để tạo vector đặc trưng 1D có số chiều $2C$ (với `mobilenet_v3_large`, vector đặc trưng đầu ra có kích thước **`1920` chiều** = $960 \times 2$).
* **`classifier` (Classification Head):** Gồm `Linear(2C -> 256, bias=False)` $\rightarrow$ `BatchNorm1d(256)` $\rightarrow$ `SiLU()` $\rightarrow$ `Dropout(p)` $\rightarrow$ `Linear(256 -> 2)`.

---

## 4. Kết quả Pha 1 & Pha 2: Sàng lọc Backbone & Tối ưu Siêu tham số (Optuna)

### 4.1. Pha 1 — Sàng lọc Nhanh 3 Ứng viên Pretrained Backbones (Quick Screening)

Trong Pha 1, các ứng viên được huấn luyện sàng lọc qua **6 epochs** (`Epoch 1 -> 2`: đóng băng Backbone `FROZEN-WARMUP`; `Epoch 3 -> 6`: mở khóa toàn mạng `UNFROZEN-FINETUNE` với $\text{LR}_{\text{head}} = 10^{-3}, \eta_{\text{ratio}} = 0.15$):

* **`mobilenet_v3_large`:** Thể hiện tốc độ hội tụ và khả năng thích nghi vượt trội trên dữ liệu FAS. Ở 2 epochs đóng băng đầu tiên (chỉ mất ~21s/epoch ở Epoch 2), `Val ACER` giảm từ `19.00%` xuống `14.30%` (`AUC = 93.10%`). Ngay khi mở khóa Backbone ở Epoch 3, `Val ACER` giảm sâu xuống **`5.71%`** (`AUC = 98.12%`), tiếp tục hạ xuống `5.51%` (Epoch 4) và lập kỷ lục sàng lọc tại **Epoch 6 với `Val ACER = 3.92%` (`APCER = 3.89%`, `BPCER = 3.95%`, `AUC = 99.50%`)** với thời gian trung bình chỉ ~60 giây/epoch.
* **`efficientnet_b0`:** Trong giai đoạn đóng băng đạt `Val ACER = 19.30%` (Epoch 1) và khi mở khóa ở Epoch 3 giảm xuống `11.38%` (`AUC = 96.22%`), tuy nhiên thời gian thực thi khi mở khóa nặng hơn đáng kể (~110 giây/epoch) và tốc độ hội tụ chậm hơn so với `mobilenet_v3_large`.
* 👉 **Quyết định lựa chọn Backbone:** **`mobilenet_v3_large`** chính thức được chọn làm kiến trúc khung xương vô địch (**Winner Backbone**) cho **Model 2** nhờ đạt độ chính xác cao nhất và tốc độ tính toán trên Intel XPU nhanh gần gấp đôi so với đối thủ.

### 4.2. Pha 2 — Tối ưu Siêu tham số với Optuna cho `mobilenet_v3_large`

* **Thiết lập:** Chạy **12 trials $\times$ 6 epochs/trial** sử dụng thuật toán **TPE** và cơ chế cắt tỉa sớm **Median Pruner** theo mục tiêu tối thiểu hóa `Val ACER`.
* **Diễn biến tìm kiếm:** Bộ cắt tỉa `MedianPruner` đã phát hiện và dừng sớm chính xác 4 cấu hình kém hiệu quả (Trial 6, 7, 8, 9 — chủ yếu rơi vào nhóm mở khóa muộn ở `unfreeze_epoch = 3` chỉ sau 37 giây). Các cấu hình mở khóa sớm ở **`unfreeze_epoch = 2`** đều đạt `Val ACER < 4.2%` và `AUC > 99.0%`.
* **Cấu hình tối ưu nhất (Trial #0 — Lưu tại `best_params_stage2.json`):** Đạt **`Val ACER = 2.59%`** (`AUC = 99.67%`):

| Siêu tham số (Hyperparameter) | Khoảng tìm kiếm (Search Space) | Giá trị Tối ưu (Trial #0) | Ý nghĩa Kỹ thuật trong Transfer Learning |
| --- | --- | --- | --- |
| **`backbone_name`** | Sàng lọc từ Pha 1 | **`mobilenet_v3_large`** | Khung xương nhẹ (~4.5M tham số tính cả Head), xuất vector 1920 chiều. |
| **`lr_head`** | $[3 \times 10^{-4}, 2.5 \times 10^{-3}]$ (log) | **`0.000664`** | Tốc độ học khởi điểm vừa phải giúp tầng Classifier ổn định mượt mà. |
| **`backbone_lr_ratio` ($\eta_{\text{ratio}}$)** | $[0.05, 0.30]$ | **`0.2877`** | Tốc độ học của Backbone bằng ~28.8% của Head ($\text{LR}_{\text{backbone}} \approx 1.91 \times 10^{-4}$). |
| **`unfreeze_epoch`** | $\{2, 3\}$ | **`2`** | Chỉ cần làm nóng Head đúng 1 epoch (18.7s) rồi mở khóa toàn mạng ngay từ Epoch 2. |
| **`weight_decay`** | $[10^{-4}, 2 \times 10^{-2}]$ (log) | **`0.004834`** | Chuẩn hóa $L_2$ trên `AdamW` giúp ngăn ngừa quá khớp khi tinh chỉnh sâu. |
| **`dropout_rate`** | $[0.20, 0.50]$ (step `0.05`) | **`0.40`** | Tăng cường ngắt kết nối ngẫu nhiên trên vector đặc trưng 1920 chiều. |
| **`batch_size` / `focal_gamma` / `alpha_real` / `smoothing`** | Kế thừa từ Stage 1 | **`32` / `2.0` / `0.55` / `0.08`** | Duy trì sự đồng nhất về phân phối mini-batch và hàm mục tiêu với Model 1. |

---

## 5. Kết quả Huấn luyện Toàn phần & Đánh giá Chuẩn ISO/IEC 30107-3

### 5.1. Quá trình Huấn luyện Toàn phần (Full Fine-Tuning)

Mô hình `TransferFASNet (mobilenet_v3_large)` được khởi tạo với bộ siêu tham số tối ưu từ Trial #0 và huấn luyện với cấu hình `FULL_EPOCHS = 50`, `EARLY_STOP_PATIENCE = 12`, kết hợp bộ điều chỉnh tốc độ học `ReduceLROnPlateau (factor=0.5, patience=3)`:

* **Thời gian hoàn tất:** **22.0 phút** cho 24 epochs (trung bình chỉ **18.7 giây** cho Epoch Warm-up và **~56.5 giây/epoch** cho giai đoạn Full Fine-tuning trên `xpu:0` — nhanh gấp **4.6 lần** so với 102.0 phút của Stage 1).
* **Diễn biến hội tụ theo 4 bậc thang Learning Rate:**
  * **Epoch 1 (`FROZEN-WARMUP`, `LR_head = 0.000664`, `LR_backbone = 0.0`):** Chỉ sau 18.7 giây làm nóng Head, mô hình đạt ngay `Val ACER = 18.56%` (`AUC = 89.64%`).
  * **Giai đoạn Mở khóa Bậc 1 (`Epoch 2 -> 10`, `LR_head = 0.000664`, `LR_backbone = 0.000191`):** Ngay khi mở khóa toàn mạng ở Epoch 2, `Val ACER` giảm hơn một nửa xuống `8.69%` (`AUC = 97.49%` — vượt qua kỷ lục 49 epochs của Stage 1 chỉ ở Epoch thứ 2!). Mô hình liên tiếp phá kỷ lục ở Epoch 3 (`5.45%`), Epoch 4 (`5.39%`), Epoch 5 (`4.93%`) và chạm mốc **`2.94%`** (`AUC = 99.58%`) tại **Epoch 6**.
  * **Giai đoạn Mở khóa Bậc 2 (`Epoch 11 -> 16`, `LR_head = 0.000332`, `LR_backbone = 0.000095`):** Sau 4 epochs (Epoch 7 $\rightarrow$ 10) không phá mốc `2.94%`, `ReduceLROnPlateau` cắt giảm 50% tốc độ học. Nhờ bước giảm LR này, tại **Epoch 12**, mô hình xác lập **đỉnh hiệu năng toàn cục mới với `Val ACER = 2.70%` (`APCER = 2.67%`, `BPCER = 2.72%`, `ROC-AUC = 99.76%`, `Accuracy = 97.32%`)** ở ngưỡng quyết định tối ưu **$\tau^* = 0.1987$**.
  * **Giai đoạn Mở khóa Bậc 3 & 4 (`Epoch 17 -> 24`, `LR_head = 0.000166 -> 0.000083`):** Tốc độ học tiếp tục được giảm thêm 2 bậc tại Epoch 17 và Epoch 21. Lúc này `TrainLoss` đã hội tụ sát về `0.0004` và `Val AUC` duy trì ổn định ở mức `99.5% - 99.7%`. Cơ chế `EarlyStopping` kích hoạt dừng an toàn tại **Epoch 24**, bảo toàn trọng số tốt nhất tại **Epoch 12** trong tệp `model2_transfer_best.pt`.

### 5.2. Bảng Kết quả Đánh giá Chính thức của Model 2 (Validation & Test)

Trọng số tốt nhất từ **Epoch 12** (`model2_transfer_best.pt`) được nạp lại cùng ngưỡng quyết định xác lập trên tập Validation là **$\tau^* = 0.1987$** để đánh giá độc lập trên **7,580 ảnh của tập Test (`LCC_FASD_evaluation`)**:

| Tập Đánh Giá | Ngưỡng Quyết định ($\tau$) | APCER (%) ↓ *(Lọt giả mạo)* | BPCER (%) ↓ *(Từ chối thật)* | ACER (%) ↓ *(Lỗi TB)* | EER (%) ↓ *(Điểm cân bằng)* | ROC-AUC (%) ↑ | Accuracy (%) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Validation (Development)** | `0.1987` *(Val EER)* | **2.67%** | **2.72%** | **2.70%** | **2.70%** | **99.76%** | **97.32%** |
| **Test (Áp ngưỡng $\tau^*$ từ Val)** | `0.1987` *(Cố định)* | **8.73%** | **6.69%** | **7.71%** | **7.93%** | **98.34%** | **91.36%** |
| **Test (Tại ngưỡng Test EER)** | `0.2220` *(Test EER)* | **7.90%** | **7.96%** | **7.93%** | **7.93%** | **98.34%** | **92.10%** |

### 5.3. Phân tích Ma trận Nhầm lẫn trên Tập Test & So sánh Đối kháng với Model 1

| Mô hình | Loại Kiến Trúc | Val ACER (%) ↓ | Test APCER (%) ↓ | Test BPCER (%) ↓ | Test ACER (%) ↓ | Test EER (%) ↓ | Test ROC-AUC (%) ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Model 1: `CustomFASNet` (CDCN + SE)** | Train from Scratch (Stage 1) | `9.93%` | `9.51%` *(691 lỗi)* | `13.38%` *(42 lỗi)* | `11.44%` | `11.46%` | `95.76%` |
| **Model 2: `TransferFASNet` (`mobilenet_v3_large`)** | Pretrained Fine-tuned (Stage 2) | **`2.70%`** | **`8.73%`** *(634 lỗi)* | **`6.69%`** *(21 lỗi)* | **`7.71%`** | **`7.93%`** | **`98.34%`** |

* **Phân tích Ma trận Nhầm lẫn của Model 2 (tại $\tau^* = 0.1987$):**
  * **True Negative ($\text{TN} = 6,632$):** Chặn đứng chính xác **6,632 / 7,266** cuộc tấn công giả mạo (tỷ lệ chặn đúng đạt **91.27%**, tăng thêm 57 mẫu chặn thành công so với Model 1).
  * **False Positive ($\text{FP} = 634$):** Có 634 mẫu Spoof bị lọt lưới ($\text{APCER} = 8.73\%$).
  * **True Positive ($\text{TP} = 293$):** Nhận diện đúng **293 / 314** mẫu người thật (tỷ lệ nhận đúng người thật đạt **93.31%**, tăng vọt **+6.69%** so với mức 86.62% của Model 1).
  * **False Negative ($\text{FN} = 21$):** Chỉ còn **21 mẫu người thật bị từ chối nhầm** ($\text{BPCER} = 6.69\%$, giảm đúng **50% số ca lỗi từ chối người thật** so với 42 ca của Model 1).

---

## 6. Phân tích Lỗi Thực tế trên Tập Test & Động lực Thiết kế Giai đoạn 3 (Multi-task Learning)

Qua trực quan hóa 10 mẫu phân loại nhầm nghiêm trọng nhất trong tệp `error_analysis_stage2.png`, nguyên nhân vật lý của các ca lỗi còn sót lại ở Model 2 được lý giải rõ ràng:

1. **Nguyên nhân gây lỗi `False Positive (Spoof -> Real)` (Các mẫu `spoof_5222.png`, `spoof_5227.png`, `spoof_7274.png`, `spoof_6406.png`, `spoof_5239.png` có $P(\text{Real}) \in [0.986, 1.000]$):**
   * **Tấn công phát lại màn hình độ phân giải cao (High-Resolution Replay Attack) được cắt sát mặt:** Ở các mẫu `spoof_5222`, `spoof_5227` và `spoof_5239`, ảnh khuôn mặt hiển thị trên màn hình sắc nét, ánh sáng tự nhiên và không lộ viền thiết bị điện tử. Đặc biệt, ở mẫu `spoof_7274.png`, có sự xuất hiện của **con trỏ chuột máy tính màu trắng nhỏ ở góc dưới bên phải cằm** — một dấu hiệu vật lý của màn hình phát lại nhưng kích thước quá nhỏ khiến mạng phân loại nhị phân thuần ảnh RGB bị đánh lừa bởi ngữ cảnh khuôn mặt hoàn hảo.
2. **Nguyên nhân gây lỗi `False Negative (Real -> Spoof)` (Các mẫu `real_152.png`, `real_192.png`, `real_48.png`, `real_23.png`, `real_69.png` có $P(\text{Real}) \in [0.024, 0.066]$):**
   * **Lóa sáng đèn Flash trực diện & Kính râm đen phản quang:** Ở mẫu `real_152`, `real_192` và `real_23`, nguồn sáng mạnh chiếu thẳng vào trán và gò má tạo vùng phản xạ trắng giống mặt kính màn hình. Ở mẫu `real_69.png`, người thật đeo kính râm đen ngoài nắng gắt kèm nền trời cháy sáng trắng hoàn toàn.

---

## 7. Danh mục Tệp Đầu ra (Saved Artifacts) & Quyết định Kiến trúc cho Giai đoạn 3

### 7.1. Các tệp đã lưu trữ sau khi hoàn tất Giai đoạn 2

* `../models/model2_transfer_best.pt`: Trọng số tốt nhất của Model 2 (`mobilenet_v3_large` tại Epoch 12, `Val ACER = 2.70%`, `Test ACER = 7.71%`, `Test AUC = 98.34%`), lưu đầy đủ `model_state_dict`, `model_config`, `best_params`, và `val_threshold = 0.1987`.
* `../models/model2_transfer_latest.pt`: Trọng thái checkpoint tại thời điểm dừng huấn luyện (Epoch 24).
* `../checkpoint/stage2/screening_state_stage2.json` & `screening_results_stage2.csv`: Kết quả sàng lọc các ứng viên Pretrained Backbones.
* `../checkpoint/stage2/optuna_state_stage2.json`, `optuna_trials_stage2.csv` & `best_params_stage2.json`: Lịch sử tìm kiếm và bộ siêu tham số tối ưu từ Optuna.
* `../checkpoint/stage2/training_state_stage2.json` & `train_history_stage2.csv`: Nhật ký huấn luyện chi tiết qua 24 epochs.
* `../checkpoint/stage2/test_evaluation_stage2.json`: Báo cáo số liệu đánh giá chuẩn ISO/IEC 30107-3 trên tập Validation và Test.
* `../checkpoint/stage2/training_curves_stage2.png`, `test_diagnostics_stage2.png`, `error_analysis_stage2.png`: Bộ biểu đồ chẩn đoán và phân tích lỗi trực quan.

### 7.2. Quyết định Lựa chọn Encoder cho Giai đoạn 3 (`03_Multitask_Architecture.ipynb`)

* **Kết luận đối kháng (Model 1 vs. Model 2):** **Model 2 (`TransferFASNet - mobilenet_v3_large`)** giành chiến thắng thuyết phục trước Model 1 trên toàn bộ các chỉ số (`Test ACER` đạt **`7.71%`** so với `11.44%`, `Test ROC-AUC` đạt **`98.34%`** so với `95.76%`, và giảm 50% lỗi `BPCER`).
* **Định hướng Giai đoạn 3:** Sử dụng phần thân mạng **`encoder` của `model2_transfer_best.pt` (`mobilenet_v3_large`)** làm **Backbone dùng chung (Shared Encoder)**. Từ bản đồ đặc trưng không gian $6\times6\times960$ (`forward_feature_map`), hệ thống sẽ gắn thêm 2 nhánh giám sát phụ (**Head 2: Depth Map Regression** và **Head 3: FFT/LBP Noise & Moiré Map Regression**) nhằm buộc mạng học cấu trúc hình học 3D thực sự và tín hiệu phản xạ vật lý, giải quyết triệt để các ca tấn công màn hình độ phân giải cao (`spoof_5222`, `spoof_7274`) cũng như hiện tượng lóa sáng trên người thật.