# STAGE-02 — Transfer Learning: Pretrained Backbone cho Face Anti-Spoofing

## 1. Mục tiêu

Giai đoạn 2 xây dựng **Model 2 — Transfer Learning Model** cho bài toán Face Anti-Spoofing (FAS).

Mục tiêu là huấn luyện và so sánh 3 backbone CNN pretrained trên ImageNet:

- ResNet50
- EfficientNet-B3
- MobileNetV3-Large

Cả 3 backbone phải được huấn luyện trên **cùng dataset LCC-FASD**, cùng cách tiền xử lý, cùng classification head và cùng protocol thực nghiệm để bảo đảm so sánh công bằng.

Sau khi hoàn thành, backbone tốt nhất được giữ lại thành **Model 2**. Model 2 sau đó được so sánh với Model 1 của Stage 01 để quyết định encoder dùng làm nền tảng cho Stage 03 — Multi-task Learning.

> Stage 02 chạy **độc lập và song song với Stage 01**. Không được phụ thuộc checkpoint của Stage 01 để bắt đầu training.

---

## 2. Bài toán

Đầu vào là ảnh khuôn mặt đã được chuẩn hóa.

Đầu ra là phân loại nhị phân:

- `Real / Live`
- `Fake / Spoof`

Pipeline:

```text
LCC-FASD
   │
   ▼
Face image
   │
   ▼
Preprocessing
   │
   ▼
Pretrained CNN Backbone
   │
   ▼
Global Average Pooling
   │
   ▼
Classification Head
   │
   ▼
Real / Fake
```

Stage 02 **không có Depth Head hoặc Noise/Reflection Head**. Hai nhánh auxiliary chỉ xuất hiện ở Stage 03.

---

## 3. Dataset

### 3.1 Dataset chính

Sử dụng:

**LCC-FASD (Large Crowdcollected Facial Anti-Spoofing Dataset)**

Dataset phải giống Stage 01 để phép so sánh Model 1 ↔ Model 2 có ý nghĩa.

### 3.2 Split

Phải cố định train/validation/test split.

Không được tạo một split khác cho từng backbone.

Ví dụ:

```text
LCC-FASD
├── train
│   ├── real
│   └── fake
├── val
│   ├── real
│   └── fake
└── test
    ├── real
    └── fake
```

Nếu dataset thực tế không có đúng cấu trúc này, code phải xây dựng Dataset/DataLoader tương ứng mà không làm thay đổi protocol.

### 3.3 Fair comparison

Ba experiment phải dùng chung:

- dataset
- split
- random seed
- image preprocessing
- augmentation
- image size
- batch size
- classification head
- loss
- validation protocol
- checkpoint selection rule
- evaluation protocol

Chỉ thay đổi backbone.

---

## 4. Môi trường Kaggle

Stage 02 chạy trên Kaggle GPU.

Khuyến nghị cấu trúc notebook:

```text
01_environment
02_config
03_dataset
04_dataloader
05_model
06_checkpoint
07_train
08_validate
09_evaluate
10_experiment
11_compare
12_export_model2
```

Không nên gom toàn bộ chương trình vào một cell duy nhất.

### 4.1 Kiểm tra GPU

Notebook phải kiểm tra:

```text
CUDA available?
GPU name
GPU memory
PyTorch version
Torchvision version
```

Nếu có GPU thì sử dụng GPU.

Nếu Kaggle environment thay đổi, notebook phải báo rõ device đang sử dụng.

---

## 5. Ba experiment độc lập

Nên coi mỗi backbone là một experiment riêng:

```text
Stage02_ResNet50
Stage02_EfficientNetB3
Stage02_MobileNetV3
```

Có thể dùng 3 notebook/job riêng hoặc một notebook có tham số `BACKBONE`.

Không được để checkpoint của backbone này ghi đè checkpoint của backbone khác.

Cấu trúc:

```text
stage2_checkpoints/
├── resnet50/
├── efficientnet_b3/
└── mobilenet_v3_large/
```

---

## 6. Model

Mỗi model có cấu trúc:

```text
Pretrained Backbone
        │
        ▼
Feature Map
        │
        ▼
Global Average Pooling
        │
        ▼
Classification Head
        │
        ▼
2 classes
```

### 6.1 Backbone

Khởi tạo từ pretrained ImageNet weights:

```text
ResNet50
EfficientNet-B3
MobileNetV3-Large
```

### 6.2 Classification head

Classification head phải giống nhau về mặt logic giữa 3 experiment.

Đầu ra:

```text
logits = [real_score, fake_score]
```

Không được thay đổi classification head để làm một backbone có lợi thế riêng.

---

## 7. Fine-tuning strategy

Stage 02 sử dụng Transfer Learning.

Quy trình đề xuất:

### Phase A — Warm-up head

- Load pretrained backbone.
- Freeze backbone.
- Thay classification head bằng head mới.
- Chỉ train classification head trong giai đoạn warm-up.

### Phase B — Fine-tune

- Unfreeze backbone.
- Train backbone + classification head.
- Dùng learning rate nhỏ hơn Phase A.

Warm-up là chiến lược triển khai đề xuất; số epoch cụ thể phải được ghi trong config và giữ nhất quán giữa các backbone.

---

## 8. Loss

Đây là binary classification.

Có thể sử dụng Cross Entropy:

```text
L = CrossEntropyLoss(y, logits)
```

Không sử dụng các auxiliary loss ở Stage 02.

---

## 9. Optimizer và Scheduler

Optimizer, learning rate, weight decay, scheduler và số epoch phải được đặt trong config.

Ví dụ:

```yaml
training:
  epochs: ...
  batch_size: ...
  learning_rate: ...
  weight_decay: ...
  optimizer: AdamW
  scheduler: ...
```

Không hard-code các giá trị quan trọng ở nhiều nơi trong notebook.

Nếu thay đổi hyperparameter, phải lưu lại trong `config.json` cùng checkpoint.

---

# 10. CHECKPOINT — BẮT BUỘC

Stage 02 phải được thiết kế để Kaggle có thể bị ngắt phiên mà không làm mất toàn bộ training.

## 10.1 Không chỉ lưu model weights

Mỗi checkpoint phải chứa ít nhất:

```text
model_state_dict
optimizer_state_dict
scheduler_state_dict
epoch
global_step
best_metric
scaler_state_dict
config
random_state
```

Nếu không sử dụng scheduler/scaler thì có thể lưu `null` hoặc bỏ phần tương ứng, nhưng code resume phải xử lý được.

## 10.2 Hai loại checkpoint

Mỗi experiment phải có:

```text
latest.pt
best.pt
```

### latest.pt

Được cập nhật sau mỗi epoch.

Dùng để resume nếu Kaggle bị disconnect/timeout.

### best.pt

Chỉ cập nhật khi validation metric đạt kết quả tốt hơn checkpoint tốt nhất trước đó.

Dùng làm model cuối cùng của experiment.

---

## 11. Checkpoint sau MỖI EPOCH

Không chỉ checkpoint mỗi 10 hoặc 20 epoch.

Flow bắt buộc:

```text
Train epoch N
     │
     ▼
Validation
     │
     ├── metric tốt hơn?
     │       │
     │       └── Yes → save best.pt
     │
     ▼
save latest.pt
     │
     ▼
save history/log
```

Nếu Kaggle dừng sau epoch 17:

```text
latest.pt = epoch 17
```

Session tiếp theo phải resume từ epoch 18.

---

## 12. Resume tự động

Khi notebook khởi động:

```text
if latest checkpoint exists:
    load checkpoint
    restore model
    restore optimizer
    restore scheduler
    restore scaler
    restore epoch
    restore global_step
    restore best_metric
    restore random state

    start_epoch = saved_epoch + 1

else:
    initialize pretrained model
    start_epoch = 0
```

Không được tự động khởi tạo lại model nếu checkpoint hợp lệ đã tồn tại.

Notebook phải in rõ:

```text
[RESUME]
Backbone: resnet50
Checkpoint: latest.pt
Last epoch: 17
Best metric: ...
Resume from epoch: 18
```

hoặc:

```text
[NEW RUN]
Backbone: resnet50
No checkpoint found.
Starting from epoch 0.
```

---

# 13. Persistent storage trên Kaggle

Không được giả định `/kaggle/working` là nơi lưu trữ lâu dài qua mọi session.

Checkpoint phải được đưa vào một cơ chế persistent storage của workflow Kaggle, ví dụ một Kaggle Dataset dành riêng cho checkpoint.

Khuyến nghị:

```text
face-antispoofing-stage2-checkpoints
```

Cấu trúc:

```text
stage2_checkpoints/
├── resnet50/
│   ├── latest.pt
│   └── best.pt
├── efficientnet_b3/
│   ├── latest.pt
│   └── best.pt
└── mobilenet_v3_large/
    ├── latest.pt
    └── best.pt
```

Lưu ý:

- Kaggle Input thường là read-only.
- Không code theo kiểu ghi trực tiếp vào `/kaggle/input/...`.
- Checkpoint mới cần được xuất/cập nhật vào persistent storage theo workflow Kaggle đang sử dụng.
- Khi mở session mới, attach/load checkpoint dataset rồi copy checkpoint cần resume vào `/kaggle/working`.

Nếu workflow không cho phép cập nhật persistent Dataset tự động trong cùng session, phải có bước publish/update checkpoint rõ ràng thay vì giả định dữ liệu sẽ tự tồn tại.

---

# 14. Có nên checkpoint mỗi batch?

Mặc định:

```text
checkpoint mỗi epoch = BẮT BUỘC
```

Không bắt buộc checkpoint mỗi batch.

Nếu một epoch cực kỳ dài, có thể bổ sung:

```text
checkpoint mỗi N batches
```

nhưng phải cân nhắc I/O overhead.

Thiết kế mặc định:

```text
Epoch checkpoint
+ best checkpoint
```

là đủ.

---

# 15. Mixed Precision

Nếu GPU Kaggle hỗ trợ, có thể sử dụng AMP/mixed precision để tăng tốc và giảm VRAM.

Nếu sử dụng GradScaler, phải lưu:

```text
scaler_state_dict
```

vào checkpoint.

Khi resume phải restore scaler.

---

# 16. DataLoader

DataLoader phải được cấu hình phù hợp với GPU.

Các tham số như:

```text
batch_size
num_workers
pin_memory
persistent_workers
```

phải nằm trong config hoặc được ghi log.

Cần benchmark `num_workers` thay vì giả định giá trị tối ưu.

Nếu DataLoader trở thành bottleneck, điều chỉnh DataLoader trước khi tăng độ phức tạp model.

---

# 17. Training loop

Một epoch:

```text
model.train()

for batch:
    images, labels = batch

    move images/labels to device

    optimizer.zero_grad()

    forward

    calculate loss

    backward

    optimizer.step()

    update global_step
```

Sau đó:

```text
model.eval()

with no_grad:
    run validation
    collect predictions
    calculate validation metrics
```

Cuối epoch:

```text
update scheduler

save latest.pt

if validation metric improved:
    save best.pt

append history.csv
```

---

# 18. Metric và model selection

Dự án sử dụng chuẩn đánh giá FAS theo ISO/IEC 30107-3:

- APCER
- BPCER
- ACER

Trong đó:

```text
ACER = (APCER + BPCER) / 2
```

Accuracy không phải chỉ số chính của dự án.

Do đó, protocol validation phải xác định rõ metric dùng để chọn `best.pt`.

Ví dụ:

```text
best_metric = lowest validation ACER
```

Nếu validation protocol chưa đủ dữ liệu để tính APCER/BPCER đúng nghĩa, phải ghi rõ limitation và không được âm thầm thay bằng Accuracy trong báo cáo.

---

# 19. History và logging

Mỗi epoch phải ghi ít nhất:

```text
epoch
train_loss
val_loss
validation_metric
learning_rate
epoch_time
global_step
```

Khuyến nghị:

```text
history.csv
```

Ví dụ:

```text
epoch,train_loss,val_loss,val_acer,lr,time
1,...
2,...
3,...
```

Ngoài ra có thể lưu:

```text
training.log
config.json
final_metrics.json
```

---

# 20. Cấu trúc output của một experiment

Ví dụ ResNet50:

```text
outputs/
└── resnet50/
    ├── checkpoints/
    │   ├── latest.pt
    │   └── best.pt
    ├── history.csv
    ├── config.json
    ├── final_metrics.json
    └── training.log
```

EfficientNet:

```text
outputs/
└── efficientnet_b3/
    ├── checkpoints/
    │   ├── latest.pt
    │   └── best.pt
    ├── history.csv
    ├── config.json
    ├── final_metrics.json
    └── training.log
```

MobileNet:

```text
outputs/
└── mobilenet_v3_large/
    ├── checkpoints/
    │   ├── latest.pt
    │   └── best.pt
    ├── history.csv
    ├── config.json
    ├── final_metrics.json
    └── training.log
```

---

# 21. Experiment tracking

Mỗi experiment phải có ID/name rõ ràng:

```text
stage02_resnet50
stage02_efficientnet_b3
stage02_mobilenet_v3_large
```

Config phải ghi:

```text
stage
experiment_name
backbone
pretrained
dataset
seed
image_size
batch_size
optimizer
learning_rate
weight_decay
scheduler
epochs
augmentation
```

Nhờ vậy có thể truy nguyên chính xác checkpoint được tạo bằng cấu hình nào.

---

# 22. Reproducibility

Cố định seed cho:

```text
Python random
NumPy
PyTorch
DataLoader
```

Đồng thời ghi seed vào config/checkpoint.

Ba backbone phải dùng cùng seed và split.

Không tuyên bố kết quả GPU hoàn toàn deterministic nếu chưa bật đầy đủ deterministic settings.

---

# 23. Chạy ba backbone

Có hai cách.

## Cách A — 3 notebook

```text
Stage02_ResNet50.ipynb
Stage02_EfficientNetB3.ipynb
Stage02_MobileNetV3.ipynb
```

Ưu điểm:

- checkpoint độc lập
- dễ resume
- một experiment lỗi không ảnh hưởng experiment khác
- dễ chạy song song

## Cách B — một notebook với tham số

```text
BACKBONE = "resnet50"
```

Sau đó đổi thành:

```text
BACKBONE = "efficientnet_b3"
```

và:

```text
BACKBONE = "mobilenet_v3_large"
```

Trong trường hợp này, checkpoint path bắt buộc phụ thuộc `BACKBONE`.

Khuyến nghị ưu tiên **Cách A nếu cần chạy song song trên nhiều Kaggle session**.

---

# 24. Không được làm các việc sau

### Không được:

```text
Train 3 backbone với 3 split khác nhau
```

### Không được:

```text
Chỉ lưu model weights
```

### Không được:

```text
Checkpoint mỗi 20 epoch rồi coi như đủ
```

### Không được:

```text
Resume model nhưng reset optimizer
```

### Không được:

```text
Resume model nhưng reset scheduler
```

### Không được:

```text
Chọn best model bằng test set
```

### Không được:

```text
Dùng Depth/Noise auxiliary loss ở Stage 02
```

### Không được:

```text
Đổi augmentation hoặc classification head riêng cho một backbone
```

### Không được:

```text
Ghi checkpoint trực tiếp vào Kaggle Input nếu Input là read-only
```

---

# 25. Quy trình đầy đủ

```text
START
  │
  ▼
Load Kaggle environment
  │
  ▼
Load LCC-FASD
  │
  ▼
Load fixed split
  │
  ▼
Select BACKBONE
  │
  ├── ResNet50
  ├── EfficientNet-B3
  └── MobileNetV3-Large
  │
  ▼
Load ImageNet pretrained weights
  │
  ▼
Replace classification head
  │
  ▼
Check persistent checkpoint
  │
  ├── Found
  │     │
  │     ▼
  │   Restore full training state
  │     │
  │     ▼
  │   Resume
  │
  └── Not found
        │
        ▼
      Start new run
        │
        ▼
  Warm-up classification head
        │
        ▼
  Fine-tune backbone + head
        │
        ▼
  Validation
        │
        ▼
  Calculate FAS metrics
        │
        ├── Better?
        │     └── save best.pt
        │
        ▼
  save latest.pt
        │
        ▼
  save history/log
        │
        ▼
  Next epoch
        │
        ▼
      DONE
```

---

# 26. Tổng hợp ba model

Sau khi ba experiment hoàn thành:

```text
ResNet50
    ↓
best.pt
    ↓
metrics

EfficientNet-B3
    ↓
best.pt
    ↓
metrics

MobileNetV3-Large
    ↓
best.pt
    ↓
metrics
```

Tạo bảng:

| Backbone | Validation metric | Test metric | Parameters | Inference latency |
|---|---:|---:|---:|---:|
| ResNet50 | ... | ... | ... | ... |
| EfficientNet-B3 | ... | ... | ... | ... |
| MobileNetV3-Large | ... | ... | ... | ... |

Các phép đo phải dùng cùng protocol.

---

# 27. Tạo Model 2

Sau khi hoàn thành 3 experiment:

```text
3 pretrained backbones
        │
        ▼
3 best checkpoints
        │
        ▼
Same evaluation protocol
        │
        ▼
Select backbone theo tiêu chí
đã định trước
        │
        ▼
MODEL 2
```

Model 2 là checkpoint của backbone được chọn, không phải một model được train lại từ đầu.

README của dự án quy định backbone tốt nhất trong ba backbone được giữ lại thành Model 2. Model 2 sau đó được so sánh với Model 1 để quyết định encoder dùng cho Stage 03.

---

# 28. Handoff sang Stage 03

Stage 02 phải bàn giao tối thiểu:

```text
Model 2 checkpoint
Model 2 config
Model 2 architecture definition
Model 2 preprocessing
Model 2 classification head
Model 2 metrics
Model 2 training history
```

Stage 03 sẽ sử dụng backbone thắng cuộc làm encoder chung cho Multi-task Architecture.

Không được thay đổi backbone sau khi đã công bố kết quả Stage 02 mà không ghi lại một experiment mới.

---

# 29. Tiêu chí hoàn thành Stage 02

Stage 02 chỉ được coi là hoàn thành khi:

- [ ] LCC-FASD được load thành công.
- [ ] Train/val/test split được cố định.
- [ ] ResNet50 được fine-tune.
- [ ] EfficientNet-B3 được fine-tune.
- [ ] MobileNetV3-Large được fine-tune.
- [ ] Ba experiment dùng cùng protocol.
- [ ] Checkpoint `latest.pt` được lưu sau mỗi epoch.
- [ ] `best.pt` được lưu theo validation metric.
- [ ] Checkpoint chứa đủ model/optimizer/scheduler/scaler/epoch/config state cần thiết.
- [ ] Có thể resume sau khi restart Kaggle.
- [ ] Checkpoint được bảo toàn qua session bằng persistent storage workflow.
- [ ] Có `history.csv`.
- [ ] Có `config.json`.
- [ ] Có final metrics.
- [ ] Ba backbone được đánh giá trên cùng protocol.
- [ ] Model 2 được xác định.
- [ ] Model 2 có checkpoint có thể load độc lập.
- [ ] Có đầy đủ artifact để Stage 03 sử dụng Model 2 làm encoder.

---

# 30. Quan hệ với các Stage khác

Stage 02 chạy song song với Stage 01:

```text
             PROJECT
                │
       ┌────────┴────────┐
       ▼                 ▼
   STAGE 01           STAGE 02
   Custom CNN        Transfer Learning
       │                 │
       ▼                 ▼
    Model 1            Model 2
       │                 │
       └────────┬────────┘
                ▼
             STAGE 03
          Multi-task Model
```

Stage 02 **không cần chờ Stage 01 hoàn thành để bắt đầu**.

Chỉ đến cuối Stage 02/Stage 01 mới cần hai kết quả để thực hiện phép so sánh:

```text
Model 1 vs Model 2
```

Kết quả so sánh này được dùng để xác định encoder khởi tạo cho Stage 03.

---

## 31. Kết quả mong đợi

Stage 02 phải tạo ra một bộ artifact có thể chuyển giao:

```text
STAGE-02/
├── resnet50/
│   ├── best.pt
│   ├── latest.pt
│   ├── history.csv
│   ├── config.json
│   └── final_metrics.json
│
├── efficientnet_b3/
│   ├── best.pt
│   ├── latest.pt
│   ├── history.csv
│   ├── config.json
│   └── final_metrics.json
│
├── mobilenet_v3_large/
│   ├── best.pt
│   ├── latest.pt
│   ├── history.csv
│   ├── config.json
│   └── final_metrics.json
│
└── MODEL_2/
    ├── model.pt
    ├── config.json
    └── metrics.json
```

**Nguyên tắc cuối cùng:** ưu tiên tính tái lập và khả năng resume hơn việc tối giản notebook. Một Kaggle session bị ngắt không được khiến phải huấn luyện lại toàn bộ Stage 02.
