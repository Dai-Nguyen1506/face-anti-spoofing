import os
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    roc_auc_score, 
    ConfusionMatrixDisplay, 
    RocCurveDisplay
)
class MetricsCallback(Callback):
    def __init__(self, val_dataset, save_dir="saved_models", output_name="final_output"):
        super().__init__()
        self.val_dataset = val_dataset
        self.output_name = output_name
        self.save_dir = save_dir
        self.y_true = np.concatenate([y for x, y in val_dataset], axis=0).flatten()
        self.best_acer = float('inf')
        
        # Đảm bảo thư mục lưu tồn tại
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        preds = self.model.predict(self.val_dataset, verbose=0)
        if isinstance(preds, dict):
            preds = preds[self.output_name]
        y_pred_probs = preds.flatten()
        
        # Tính EER và tìm ngưỡng tối ưu (Threshold)
        fpr, tpr, thresholds = roc_curve(self.y_true, y_pred_probs)
        fnr = 1 - tpr
        eer_index = np.nanargmin(np.absolute((fnr - fpr)))
        eer_threshold = thresholds[eer_index]
        
        if np.isinf(eer_threshold) or eer_threshold > 1.0:
            eer_threshold = 1.0
        
        # Tính ACER
        y_pred_bin = (y_pred_probs >= eer_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred_bin).ravel()
        
        bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        acer = (apcer + bpcer) / 2

        logs['val_apcer'] = apcer
        logs['val_bpcer'] = bpcer
        logs['val_acer'] = acer

        if acer < self.best_acer:
            self.best_acer = acer

            model_path = os.path.join(self.save_dir, "best_model.keras")
            self.model.save(model_path)

            threshold_path = os.path.join(self.save_dir, "best_threshold.txt")
            with open(threshold_path, "w") as f:
                f.write(str(eer_threshold))
                
            print(f" - APCER: {apcer*100:.2f}% - BPCER: {bpcer*100:.2f}% - ACER: {acer*100:.2f}% (t*: {eer_threshold:.4f}) ⭐ NEW BEST")
        else:
            print(f" - APCER: {apcer*100:.2f}% - BPCER: {bpcer*100:.2f}% - ACER: {acer*100:.2f}% (t*: {eer_threshold:.4f}) ")

def plot_training_history(history_dict, output=None):
    """Vẽ 3 biểu đồ lịch sử huấn luyện (Loss, Accuracy, Error Rates)"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    ax1.plot(history_dict['loss'], label='Train Loss', color='blue', marker='o')
    ax1.plot(history_dict['val_loss'], label='Val Loss', color='red', marker='o')
    ax1.set_title('1. Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Accuracy (mô hình nhiều đầu ra thì lấy đầu ra cuối cùng)
    acc_key = 'final_output_accuracy' if 'final_output_accuracy' in history_dict else 'accuracy'
    ax2.plot(history_dict[acc_key], label='Train Accuracy', color='blue', marker='o')
    ax2.plot(history_dict['val_' + acc_key], label='Val Accuracy', color='red', marker='o')
    ax2.set_title('2. Training & Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
        
    # Error Rates
    ax3.plot(history_dict['val_acer'], label='Val ACER', color='red', linewidth=2)
    ax3.plot(history_dict['val_apcer'], label='Val APCER', color='blue', linestyle='--')
    ax3.plot(history_dict['val_bpcer'], label='Val BPCER', color='yellow', linestyle='--')
    ax3.set_title('3. Validation Error Rates')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Error Rate')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()

    if output is not None:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ lịch sử tại: {output}")

    plt.show()

def evaluate_model(y_test, y_val, threshold, output=None):
    """Đánh giá toàn diện trên Test, dùng threshold từ Val"""
    y_true, y_pred_probs = y_test
    y_val_true, y_val_pred_probs = y_val
    
    y_pred = (y_pred_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    ax_cm = axes[0]
    ax_roc = axes[1]
    ax_tradeoff = axes[2]
    
    # 1. Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['REAL (0)', 'SPOOF (1)'])
    disp.plot(cmap='Blues', ax=ax_cm, values_format='d', text_kw={'fontsize': 12})
    ax_cm.set_title(f"Confusion Matrix - Tập Test (τ* = {threshold:.3f})", fontweight='bold', fontsize=14)
    ax_cm.set_xlabel("Dự đoán", fontsize=12)
    ax_cm.set_ylabel("Thực tế", fontsize=12)
    ax_cm.grid(False)
    
    # 2. ROC Curve
    fpr_val, tpr_val, _ = roc_curve(y_val_true, y_val_pred_probs)
    auc_val = roc_auc_score(y_val_true, y_val_pred_probs)
    RocCurveDisplay(fpr=fpr_val, tpr=tpr_val, roc_auc=auc_val).plot(ax=ax_roc, name='Val ROC')
    
    fpr_test, tpr_test, _ = roc_curve(y_true, y_pred_probs)
    auc_test = roc_auc_score(y_true, y_pred_probs)
    RocCurveDisplay(fpr=fpr_test, tpr=tpr_test, roc_auc=auc_test).plot(ax=ax_roc, name='Test ROC')
    
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Ngẫu nhiên (AUC = 0.5)')
    ax_roc.set_title("Đường cong ROC", fontweight='bold', fontsize=14)
    ax_roc.set_xlabel("False Positive Rate (BPCER)", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate (1 - APCER)", fontsize=12)
    ax_roc.legend(fontsize=11)
    
    # 3. Đồ thị đánh đổi APCER vs BPCER theo Ngưỡng
    fpr_all, tpr_all, thresholds_all = roc_curve(y_true, y_pred_probs)
    fnr_all = 1 - tpr_all
    
    valid_idx = thresholds_all <= 1.0
    fpr_all = fpr_all[valid_idx]
    fnr_all = fnr_all[valid_idx]
    thresholds_all = thresholds_all[valid_idx]
    
    ax_tradeoff.plot(thresholds_all, fnr_all * 100, label='APCER (%) - Nhận nhầm giả', color='#e63946', linewidth=2.5)
    ax_tradeoff.plot(thresholds_all, fpr_all * 100, label='BPCER (%) - Từ chối nhầm thật', color='#4361ee', linewidth=2.5)
    ax_tradeoff.axvline(x=threshold, color='#2a9d8f', linestyle='--', linewidth=2, label=f'Ngưỡng Val τ* ({threshold:.3f})')
    ax_tradeoff.set_title("Đánh đổi APCER vs. BPCER", fontweight='bold', fontsize=14)
    ax_tradeoff.set_xlabel("Ngưỡng xác suất (Threshold)", fontsize=12)
    ax_tradeoff.set_ylabel("Tỷ lệ lỗi (%)", fontsize=12)
    ax_tradeoff.legend(fontsize=11)
    
    plt.tight_layout()
    
    if output is not None:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ tại: {output}")
        
    plt.show()


def calculate_metrics(y_true, y_prob, threshold):
    """
    Tính toán các chỉ số EER, APCER, BPCER, ACER, AUC, ACC dựa trên nhãn thực tế và xác suất dự đoán.
    """
    y_pred = (y_prob >= threshold).astype(int)
    attack_samples = y_true == 1
    apcer = np.mean(y_pred[attack_samples] == 0) if np.any(attack_samples) else 0.0

    real_samples = y_true == 0
    bpcer = np.mean(y_pred[real_samples] == 1) if np.any(real_samples) else 0.0

    acer = (apcer + bpcer) / 2
    acc = accuracy_score(y_true, y_pred)


    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fnr = 1 - tpr
        eer = fpr[np.argmin(np.absolute(fnr - fpr))]
    else:
        auc = 0.0
        eer = 0.0


    return {
        "EER (%)": eer * 100,
        "APCER (%)": apcer * 100,
        "BPCER (%)": bpcer * 100,
        "ACER (%)": acer * 100,
        "AUC (%)": auc * 100,
        "ACC (%)": acc * 100
    }