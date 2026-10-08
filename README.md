# Face Anti-Spoofing (FAS) - Phát hiện gian lận quét khuôn mặt trên điện thoại

## Giới thiệu dự án
Bài toán của dự án này là xây dựng và huấn luyện mô hình phát hiện gian lận quét khuôn mặt (Face Anti-Spoofing) nhằm ứng dụng trên các thiết bị điện thoại di động.

Quy trình thực hiện dự án bao gồm 3 giai đoạn chính:
1. **Huấn luyện ban đầu:** Xây dựng và huấn luyện 3 loại mô hình (kiến trúc) khác nhau trên bộ dữ liệu **LCC-FASD**.
2. **Fine-Tuning:** Tinh chỉnh (fine-tune) các mô hình đã được huấn luyện với bộ dữ liệu **CelebA-Spoof** nhằm tăng cường khả năng chống giả mạo.
3. **Đánh giá (Cross-dataset Evaluation):** Đánh giá hiệu năng và khả năng tổng quát hóa của mô hình trên 4 bộ dữ liệu độc lập khác.

## Bộ dữ liệu
Dự án sử dụng tổng cộng 6 bộ dữ liệu phân bổ cho các giai đoạn khác nhau:

### 1. Tập huấn luyện và thử nghiệm
* **LCC-FASD**: [Large Crowdcollected Face Anti-Spoofing Dataset](https://www.kaggle.com/datasets/faber24/lcc-fasd/data)

### 2. Tập dữ liệu tăng cường
* **CelebA-Spoof**: [CelebA Spoof For Face AntiSpoofing](https://www.kaggle.com/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing/code)

### 3. Tập đánh giá và so sánh
* **SiW**: [Anti-spoofing SiW Dataset](https://www.kaggle.com/datasets/nagatoyuki1218/anti-spoofing-siw-dataset/data)
* **CASIA-FASD**: [CASIA-FASD](https://www.kaggle.com/datasets/immada/casia-fasd)
* **MSU-MFSD**: [MSU-MFSD Processed into frames](https://www.kaggle.com/datasets/minhtranv/msu-mfsd-processed-into-frames)
* **OULU-NPU**: [Oulu-NPU](https://www.kaggle.com/datasets/mizaku/oulu-npu-test/code?datasetId=8890414&sortBy=dateRun&tab=profile&excludeNonAccessedDatasources=false)

## Quy trình thực hiện (Workflow)
- **Tiền xử lý (Preprocessing):** Trích xuất khung hình từ video, nhận diện khuôn mặt (face detection), và chuẩn bị dữ liệu.
- **Baseline Models:** Xây dựng 3 kiến trúc mô hình học sâu để phân loại khuôn mặt thật (real) và giả mạo (spoof).
- **Fine-Tuning:** Áp dụng Transfer Learning từ LCC-FASD sang CelebA-Spoof.
- **Benchmark:** Đánh giá độ chính xác (Accuracy), tỉ lệ lỗi HTER, APCER, BPCER trên 4 bộ dữ liệu SiW, CASIA-FASD, MSU-MFSD và OULU-NPU.