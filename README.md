# **Hệ thống CNN Học sâu cho Chống giả mạo Khuôn mặt Đa môi trường tích hợp Học đa nhiệm (Multi-task Learning)**

*(A Deep Learning-Based CNN System for Cross-Environment Face Anti-Spoofing with Multi-task Auxiliary Supervision)*

## 1. Tóm tắt dự án

Dự án phát triển hệ thống Phát hiện gian lận quét mặt (Face Anti-Spoofing – FAS / Liveness Detection) ứng dụng Mạng nơ-ron tích chập (CNN), đóng vai trò lớp phòng thủ sinh trắc học cốt lõi cho các ứng dụng định danh điện tử (eKYC).

Điểm cốt lõi của dự án là giải quyết bài toán **Domain Shift** (chênh lệch miền dữ liệu giữa các bộ dataset/thiết bị/điều kiện thu nhận khác nhau) thông qua ba hướng tiếp cận được thực nghiệm và so sánh trực tiếp với nhau, tạo ra tổng cộng **4 model** được đánh giá cùng lúc ở giai đoạn cuối:

- **Mô hình Custom (Model 1)** — CNN tự thiết kế từ đầu (from scratch), không phụ thuộc backbone pretrained.
- **Mô hình Transfer Learning (Model 2)** — so sánh nhiều backbone pretrained (ResNet/MobileNetV3/EfficientNet) fine-tune lại cho bài toán FAS, chốt lại backbone tốt nhất thành một model độc lập.
- **Mô hình Kiến trúc (Model 3)** — kiến trúc Multi-task Learning tự thiết kế, tích hợp giám sát phụ (Auxiliary Depth Map, Auxiliary Noise/Reflection Map) để ép mô hình học đặc trưng vật lý thực chất thay vì học vẹt nhiễu camera hay đặc điểm riêng của từng dataset.
- **Model 4** — Model 3 được huấn luyện tiếp trên một bộ dữ liệu domain khác (CelebA-Spoof) để kiểm tra khả năng tổng quát hóa.

Toàn bộ 4 model trên được đánh giá và so sánh cùng lúc bằng chiến lược **cross-dataset** nghiêm ngặt, dùng các bộ dữ liệu hoàn toàn không xuất hiện trong huấn luyện để đo khả năng khái quát hóa thực sự.

## 2. Vấn đề đặt ra & Các véc-tơ tấn công (Problem & Attack Vectors)

Hệ thống được thiết kế để chống lại 3 nhóm tấn công (Presentation Attacks) phổ biến:

1. **Print Attack (2D Static):** Sử dụng ảnh chân dung in trên giấy, ảnh bóng, hoặc mặt nạ giấy khoét lỗ ngũ quan.
2. **Replay Attack (2D Dynamic):** Phát lại video khuôn mặt nạn nhân trên màn hình thiết bị điện tử (smartphone, tablet, laptop).
3. **3D Mask Attack (Complex):** (Định hướng mở rộng) Tấn công bằng mặt nạ silicon/resin mô phỏng cấu trúc 3D.

## 3. Cơ sở lý thuyết & Đặc trưng khai thác

Thay vì nhận diện danh tính (Identity), kiến trúc CNN trong dự án được thiết kế chuyên biệt để khai thác sự khác biệt về **quang học và tính chất vật lý** giữa cá thể sống và vật liệu giả mạo:

| **Yếu tố vật lý** | **Khuôn mặt thật (Live)** | **Khuôn mặt giả mạo (Spoof)** |
|---|---|---|
| **Độ sâu 3D (Depth)** | Có cấu trúc lồi lõm tự nhiên (mũi cao, hốc mắt sâu). | Phẳng (giấy/màn hình) hoặc sai lệch trường sâu. |
| **Hiện tượng Moiré** | Tuyệt đối không có. | Xuất hiện vân sọc/lượn sóng tần số cao do lưới pixel màn hình chồng lên lưới sensor camera. |
| **Vân bề mặt (Micro-texture)** | Lỗ chân lông, khuếch tán ánh sáng tự nhiên. | Bề mặt nhẵn của kính, hạt mực in phân tán, phản xạ lóa đèn. |

## 4. Kiến trúc hệ thống (System Pipeline)

### 4.1. Luồng xử lý real-time (Inference)

```
[ Camera Feed / Input Image ]
      │
      ▼
[ Bước 1: Face Detection & Alignment ]  ──► MTCNN / RetinaFace, crop chuẩn khung mặt.
      │
      ▼
[ Bước 2: Backbone / Encoder duy nhất ] ──► Trích xuất feature map chung.
      │
      ▼
[ Bước 3: Classification Head ]         ──► Global Average Pooling → FC → Real/Fake.
```

Ở lúc chạy thật (inference, kể cả app real-time camera ở Giai đoạn 5), hệ thống **chỉ cần 1 forward pass duy nhất** qua backbone + classification head — các nhánh phụ (depth, noise) của kiến trúc Multi-task chỉ tồn tại **trong lúc huấn luyện**, không chạy lúc inference.

### 4.2. Kiến trúc lúc huấn luyện Model 3 (Multi-task Head — Giai đoạn 3)

```
Ảnh RGB đầu vào
      │
      ▼
  Backbone / Encoder dùng chung
  (backbone thắng cuộc từ Giai đoạn 1 vs Giai đoạn 2)
      │
      ▼
  Feature map chung
      │
      ├──► Head 1 — Classification:       GAP → FC → Real/Fake
      ├──► Head 2 — Depth Regression:      Conv nhỏ → Depth map dự đoán
      └──► Head 3 — Noise/Reflection Regr: Conv nhỏ → Noise map dự đoán

  Loss tổng = L_classification + λ1·L_depth + λ2·L_noise
```

**Nguồn ground-truth cho 2 nhánh phụ (chỉ dùng để tạo label lúc chuẩn bị dữ liệu, không cần lúc train/infer runtime):**
- **Depth map:** sinh bằng một model depth-estimation pretrained có sẵn (áp cho ảnh live; ảnh spoof gán depth map phẳng = 0).
- **Noise/reflection map:** sinh bằng phương pháp xử lý tín hiệu cổ điển — FFT/high-pass filter để bắt hiện tượng Moiré (tín hiệu tần số cao có tính chu kỳ), kết hợp LBP (Local Binary Pattern) để bắt micro-texture bất thường.

## 5. Chiến lược Huấn luyện & Thực nghiệm (5 Giai đoạn)

### Giai đoạn 1 — Mô hình Custom → **Model 1**

Xây dựng một CNN **tự thiết kế từ đầu (train from scratch, không dùng backbone pretrained)** cho bài toán phân loại nhị phân Real/Fake.

- **Dataset:** [LCC-FASD](https://www.kaggle.com/datasets/faber24/lcc-fasd/data) — đa dạng thiết bị thu nhận và điều kiện ánh sáng tự nhiên.
- **Huấn luyện:** Hyperparameter tuning (learning rate, batch size, optimizer, weight decay, số epoch).
- **Hạ tầng:** Local (Intel Ultra 7 155U, 16GB RAM, Intel iGPU/XPU – không CUDA). Benchmark thực tế cho thấy data loading, không phải compute, là nút thắt chính; với cấu hình tối ưu, 50 epoch trên toàn bộ LCC-FASD ước tính chỉ mất khoảng 25–40 phút — đủ khả thi để **huấn luyện full ngay tại local**.
- **Kết quả:** **Model 1 (Custom Model)**.

### Giai đoạn 2 — Mô hình Transfer Learning → **Model 2**

Huấn luyện và so sánh **3 backbone pretrained** (ví dụ: ResNet50, EfficientNet-B3, MobileNetV3-Large) trên **cùng kiến trúc phân loại đơn giản** như Giai đoạn 1 và **cùng dataset LCC-FASD**, để đảm bảo so sánh công bằng.

- **Dataset:** LCC-FASD (giống Giai đoạn 1).
- **Huấn luyện:** Fine-tune từng backbone (đã có trọng số ImageNet) với head phân loại mới gắn thêm.
- **Kết quả:**
  - Backbone tốt nhất trong 3 backbone được giữ lại nguyên vẹn thành 1 model độc lập — **Model 2 (Transfer Learning)**.
  - Đồng thời, Model 2 được so sánh với Model 1 (Giai đoạn 1) để quyết định: backbone nào (custom hay pretrained thắng cuộc) sẽ được dùng làm encoder dùng chung cho kiến trúc Multi-task ở Giai đoạn 3.

*Câu hỏi cần trả lời: Trong 3 backbone pretrained, backbone nào phù hợp nhất với bài toán FAS? Và giữa mô hình tự thiết kế (Model 1) với backbone pretrained tốt nhất, hướng nào cho nền tảng tốt hơn để xây kiến trúc phức tạp hơn ở bước tiếp theo?*

### Giai đoạn 3 — Mô hình Kiến trúc (Architecture) → **Model 3**

Xây dựng kiến trúc **Multi-task Head** (xem mục 4.2) trên backbone thắng cuộc từ Giai đoạn 1 vs Giai đoạn 2.

- Thêm 2 nhánh giám sát phụ: Depth Regression (label từ model depth-estimation pretrained) và Noise/Reflection Regression (label từ FFT/high-pass + LBP).
- Backbone dùng chung được khởi tạo từ trọng số của model thắng cuộc (Model 1 hoặc Model 2) — không train lại hoàn toàn từ đầu, mà tận dụng làm điểm khởi tạo.
- **Dataset:** LCC-FASD (giữ nguyên, để so sánh công bằng với Model 1 / Model 2).
- **Kết quả:** **Model 3 (Architecture Model / Optimized)**.

*Câu hỏi cần trả lời: Việc bổ sung giám sát phụ (depth, noise) có giúp mô hình học được đặc trưng FAS tổng quát hơn so với backbone gốc (dù là custom hay pretrained) hay không?*

### Giai đoạn 4 — Huấn luyện trên domain mới (chạy song song Giai đoạn 5) → **Model 4**

Kiến trúc Model 3 được huấn luyện tiếp trên một bộ dữ liệu lớn hơn, thuộc domain khác.

- **Dataset:** [CelebA-Spoof](https://www.kaggle.com/datasets/attentionlayer241/celeba-spoof-for-antispoofing) — subset cân bằng theo stratified sampling theo subject.
- **Mô hình khởi tạo:** Checkpoint Model 3, không train lại từ đầu.
- **Hạ tầng:** Kaggle (T4 GPU, phiên tối đa 12 tiếng, hỗ trợ chạy ngầm — ưu tiên chính); Google Colab chỉ dùng làm phương án dự phòng. Checkpoint lưu định kỳ theo epoch (kèm optimizer/scheduler state), đè lên checkpoint cũ trên Kaggle Dataset/Input để resume qua các phiên bị ngắt hoặc đổi tài khoản.
- **Kết quả:** **Model 4**.

*Giai đoạn này chạy song song với Giai đoạn 5 — khi Model 4 hoàn tất, checkpoint được bổ sung trực tiếp vào bộ đánh giá của Giai đoạn 5.*

### Giai đoạn 5 — Đánh giá Cross-Dataset & Ứng dụng Real-time (chạy song song Giai đoạn 4)

Trong lúc Giai đoạn 4 huấn luyện Model 4, Giai đoạn 5 được chuẩn bị song song: xây dựng bộ khung đánh giá cross-dataset và chương trình test real-time camera, đánh giá trước với Model 1, Model 2, và Model 3; khi Model 4 sẵn sàng thì thêm vào chạy cùng.

- **Datasets đánh giá (unseen domains, không dùng trong huấn luyện):**
  - [SiW (Spoof in the Wild)](https://www.kaggle.com/datasets/nagatoyuki1218/anti-spoofing-siw-dataset)
  - [CASIA-FASD](https://www.kaggle.com/datasets/immada/casia-fasd)
  - [OULU-NPU — Test Partition](https://www.kaggle.com/datasets/mizaku/oulu-npu-test)
  - [MSU-MFSD — Processed into Frames](https://www.kaggle.com/datasets/minhtranv/msu-mfsd-processed-into-frames)
- **Quy trình:** Cả 4 model (**Model 1**, **Model 2**, **Model 3**, **Model 4**) được đánh giá trên cùng 4 bộ dataset theo cùng quy trình, so sánh theo APCER/BPCER/ACER.
- **Ứng dụng real-time:** Chương trình chạy camera trực tiếp, cho phép chọn giữa các model để so sánh trực quan độ chính xác và độ trễ thực tế.

Thiết kế này cho phép trả lời:
- **Model 1 vs Model 2:** Tự thiết kế từ đầu hay dùng backbone pretrained cho kết quả tốt hơn trên bài toán FAS?
- **→ Model 3:** Bổ sung kiến trúc Multi-task (depth + noise) có cải thiện khả năng tổng quát hóa so với backbone gốc hay không?
- **Model 3 → Model 4:** Huấn luyện thêm trên domain mới (CelebA-Spoof) có giúp mô hình generalize tốt hơn sang các domain hoàn toàn xa lạ hay không?
- **Cross-dataset tổng thể:** Trong cả 4 model, model nào giữ ACER thấp nhất và ổn định nhất trên 4 bộ dataset chưa từng thấy.

> **Lưu ý về giới hạn dữ liệu (cần ghi rõ khi báo cáo):**
> - Bản OULU-NPU sử dụng chỉ là **test partition**, cần xác minh file protocol đi kèm khớp đủ số video theo đúng Protocol 1 gốc trước khi báo cáo kết quả.
> - MSU-MFSD chỉ bao phủ Print Attack và Replay Attack (không có 3D mask), trong khi CelebA-Spoof và LCC-FASD dùng ở các giai đoạn trước có phổ tấn công đa dạng hơn — nên phân tích APCER theo từng loại tấn công riêng biệt thay vì gộp chung.

## 6. Tiêu chuẩn Đánh giá (Evaluation Metrics)

Dự án tuân thủ chuẩn đo lường bảo mật sinh trắc học quốc tế **ISO/IEC 30107-3**, không dùng Accuracy thông thường làm chỉ số chính:

- **APCER (Attack Presentation Classification Error Rate):** Tỷ lệ tấn công thành công (đo rủi ro hệ thống bị qua mặt).
- **BPCER (Bona Fide Presentation Classification Error Rate):** Tỷ lệ từ chối người dùng thật (đo rủi ro ảnh hưởng trải nghiệm người dùng).
- **ACER (Average Classification Error Rate):** Trung bình cân bằng, thước đo tổng thể chính.

$$ACER = \frac{APCER + BPCER}{2}$$

## 7. Hạ tầng & Công cụ

| Giai đoạn | Nơi chạy | Ghi chú |
|---|---|---|
| 1 (Custom Model) | Local (Intel Ultra 7 155U, 16GB RAM, Intel iGPU) | Data loading là bottleneck chính; khả thi full training tại local theo benchmark thực tế |
| 2 (Transfer Learning, 3 backbone) | Local hoặc Kaggle | Tùy tốc độ fine-tune thực tế của từng backbone trên máy local |
| 3 (Kiến trúc Multi-task) | Local hoặc Kaggle | Tùy độ phức tạp khi thêm 2 nhánh phụ |
| 4 (Model 4, CelebA-Spoof) | Kaggle (chính), Google Colab (dự phòng) | T4 GPU, phiên 12h; Kaggle hỗ trợ chạy ngầm nên được ưu tiên; checkpoint định kỳ resume qua Kaggle Dataset/Input |
| 5 (Đánh giá + Real-time) | Local / Kaggle | Đánh giá cross-dataset + app real-time camera |

## 8. Tài liệu tham khảo (References)

**Kiến trúc & phương pháp:**
- Liu, Y., Jourabloo, A., & Liu, X. (2018). *Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary Supervision*. CVPR 2018. — nguồn gốc ý tưởng giám sát phụ bằng Depth Map cho FAS.
- Yu, Z., et al. (2020). *Searching Central Difference Convolutional Networks for Face Anti-Spoofing*. CVPR 2020. — kiến trúc CDCN, tham khảo cho thiết kế Multi-task Head.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016. — ResNet, backbone pretrained thử nghiệm ở Giai đoạn 2.
- Howard, A., et al. (2019). *Searching for MobileNetV3*. ICCV 2019. — backbone pretrained thử nghiệm ở Giai đoạn 2.
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019. — backbone pretrained thử nghiệm ở Giai đoạn 2.
- Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). *Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks*. IEEE Signal Processing Letters. — MTCNN, dùng cho Face Detection & Alignment.
- Deng, J., et al. (2020). *RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild*. CVPR 2020. — lựa chọn thay thế cho Face Detection & Alignment.
- Ranftl, R., et al. (2020). *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer*. IEEE TPAMI. — mô hình depth-estimation pretrained tham khảo, dùng để sinh ground-truth depth map.
- ISO/IEC 30107-3:2017. *Information technology — Biometric presentation attack detection — Part 3: Testing and reporting*. — chuẩn định nghĩa APCER/BPCER/ACER.

**Datasets:**
- Zhang, Y., et al. (2020). *CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations*. ECCV 2020. — [Kaggle link](https://www.kaggle.com/datasets/attentionlayer241/celeba-spoof-for-antispoofing)
- LCC-FASD (Large Crowdcollected Facial Anti-Spoofing Dataset) — [Kaggle link](https://www.kaggle.com/datasets/faber24/lcc-fasd/data)
- Liu, Y., Jourabloo, A., & Liu, X. (2018). *SiW Dataset — Spoof in the Wild*. CVPR 2018. — [Kaggle link](https://www.kaggle.com/datasets/nagatoyuki1218/anti-spoofing-siw-dataset)
- Zhang, Z., et al. (2012). *A Face Antispoofing Database with Diverse Attacks*. ICB 2012. — CASIA-FASD, [Kaggle link](https://www.kaggle.com/datasets/immada/casia-fasd)
- Boulkenafet, Z., Komulainen, J., Li, L., Feng, X., & Hadid, A. (2017). *OULU-NPU: A Mobile Face Presentation Attack Database with Real-World Variations*. FG 2017. — [Kaggle link](https://www.kaggle.com/datasets/mizaku/oulu-npu-test)
- Wen, D., Han, H., & Jain, A. K. (2015). *Face Spoof Detection with Image Distortion Analysis*. IEEE Transactions on Information Forensics and Security. — MSU-MFSD, [Kaggle link](https://www.kaggle.com/datasets/minhtranv/msu-mfsd-processed-into-frames)