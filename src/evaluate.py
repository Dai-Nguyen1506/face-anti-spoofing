import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay


def find_eer_threshold(y_true, y_prob):
    """Tìm ngưỡng mà APCER = BPCER (điểm EER)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    index = np.nanargmin(np.abs((1 - tpr) - fpr))
    return min(thresholds[index], 1.0)


def calculate_metrics(y_true, y_prob, threshold):
    """Tính EER, APCER, BPCER, ACER, AUC, ACC (đơn vị %). Nhãn: real = 0, spoof = 1."""
    y_pred = (y_prob >= threshold).astype(int)

    apcer = np.mean(y_pred[y_true == 1] == 0)
    bpcer = np.mean(y_pred[y_true == 0] == 1)
    acer = (apcer + bpcer) / 2
    acc = np.mean(y_pred == y_true)
    auc = roc_auc_score(y_true, y_prob)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    eer = fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]

    return {
        "EER (%)": eer * 100,
        "APCER (%)": apcer * 100,
        "BPCER (%)": bpcer * 100,
        "ACER (%)": acer * 100,
        "AUC (%)": auc * 100,
        "ACC (%)": acc * 100,
    }


class MetricsCallback(Callback):
    """Cuối mỗi epoch: tính APCER, BPCER, ACER trên tập Val và lưu mô hình có ACER thấp nhất."""
    def __init__(self, val_dataset, save_dir="saved_models"):
        super().__init__()
        self.val_dataset = val_dataset
        self.save_dir = save_dir
        self.y_true = np.concatenate([y for x, y in val_dataset]).flatten()
        self.best_acer = float('inf')
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        y_prob = self.model.predict(self.val_dataset, verbose=0).flatten()
        threshold = find_eer_threshold(self.y_true, y_prob)
        m = calculate_metrics(self.y_true, y_prob, threshold)

        apcer = m["APCER (%)"] / 100
        bpcer = m["BPCER (%)"] / 100
        acer = m["ACER (%)"] / 100
        logs['val_apcer'] = apcer
        logs['val_bpcer'] = bpcer
        logs['val_acer'] = acer

        is_best = acer < self.best_acer
        if is_best:
            self.best_acer = acer
            self.model.save(os.path.join(self.save_dir, "best_model.keras"))
            with open(os.path.join(self.save_dir, "best_threshold.txt"), "w") as f:
                f.write(str(threshold))

        print(f" - APCER: {apcer*100:.2f}% - BPCER: {bpcer*100:.2f}% - ACER: {acer*100:.2f}% "
              f"(t*: {threshold:.4f})" + (" ⭐ NEW BEST" if is_best else ""))


def save_and_show(output):
    """Lưu hình (nếu có đường dẫn) rồi hiển thị."""
    plt.tight_layout()
    if output is not None:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ tại: {output}")
    plt.show()


def plot_training_history(history, output=None):
    """Vẽ lịch sử huấn luyện: Loss, Accuracy và các tỷ lệ lỗi trên tập Val."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    groups = {
        "1. Loss": ["loss", "val_loss"],
        "2. Accuracy": ["accuracy", "val_accuracy"],
        "3. Validation Error Rates": ["val_acer", "val_apcer", "val_bpcer"],
    }
    for ax, (title, keys) in zip(axes, groups.items()):
        for key in keys:
            ax.plot(history[key], label=key, marker='o')
        ax.set_title(title)
        ax.set_xlabel('Epochs')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    save_and_show(output)


def evaluate_model(y_test, y_val, threshold, output=None):
    """Vẽ Confusion Matrix, ROC và đồ thị đánh đổi APCER/BPCER. Ngưỡng được chọn từ tập Val."""
    y_true, y_prob = y_test
    y_val_true, y_val_prob = y_val
    y_pred = (y_prob >= threshold).astype(int)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax_cm, ax_roc, ax_tradeoff) = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['REAL (0)', 'SPOOF (1)'], cmap='Blues', values_format='d', ax=ax_cm)

    ax_cm.set_title(f"Confusion Matrix - Tập Test (t* = {threshold:.3f})", fontweight='bold', fontsize=14)
    ax_cm.set_xlabel("Dự đoán", fontsize=12)
    ax_cm.set_ylabel("Thực tế", fontsize=12)
    ax_cm.grid(False)

    # 2. ROC Curve
    RocCurveDisplay.from_predictions(y_val_true, y_val_prob, name='Val ROC', ax=ax_roc)
    RocCurveDisplay.from_predictions(y_true, y_prob, name='Test ROC', ax=ax_roc)

    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Ngẫu nhiên (AUC = 0.5)')
    ax_roc.set_title("Đường cong ROC", fontweight='bold', fontsize=14)
    ax_roc.set_xlabel("False Positive Rate (BPCER)", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate (1 - APCER)", fontsize=12)
    ax_roc.legend(fontsize=11)

    # 3. Đánh đổi APCER và BPCER theo ngưỡng
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    keep = thresholds <= 1.0
    ax_tradeoff.plot(thresholds[keep], (1 - tpr[keep]) * 100, label='APCER (%) - Nhận nhầm giả', color='#e63946', linewidth=2.5)
    ax_tradeoff.plot(thresholds[keep], fpr[keep] * 100, label='BPCER (%) - Từ chối nhầm thật', color='#4361ee', linewidth=2.5)
    ax_tradeoff.axvline(x=threshold, color='#2a9d8f', linestyle='--', linewidth=2, label=f'Ngưỡng Val τ* ({threshold:.3f})')
    ax_tradeoff.set_title("Đánh đổi APCER vs. BPCER", fontweight='bold', fontsize=14)
    ax_tradeoff.set_xlabel("Ngưỡng xác suất (Threshold)", fontsize=12)
    ax_tradeoff.set_ylabel("Tỷ lệ lỗi (%)", fontsize=12)
    ax_tradeoff.legend(fontsize=11)

    save_and_show(output)