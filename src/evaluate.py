import os
import pandas as pd
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
    def __init__(self, val_dataset, threshold_path=None):
        super().__init__()
        self.val_dataset = val_dataset
        self.threshold_path = threshold_path
        self.y_true = np.concatenate([y for x, y in val_dataset], axis=0).flatten()
        self.best_acer = float('inf')
        self.best_threshold = 0.5

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Dự đoán toàn bộ tập Validation (tắt verbose để đỡ rác terminal)
        y_pred_probs = self.model.predict(self.val_dataset, verbose=0).flatten()
        
        # Tính toán EER và tìm ngưỡng tối ưu (Threshold)
        fpr, tpr, thresholds = roc_curve(self.y_true, y_pred_probs)
        fnr = 1 - tpr
        eer_index = np.nanargmin(np.absolute((fnr - fpr)))
        eer_threshold = thresholds[eer_index]
        if np.isinf(eer_threshold) or eer_threshold > 1.0:
            eer_threshold = 1.0
        eer = (fpr[eer_index] + fnr[eer_index]) / 2
        
        # Phân loại lại dựa trên ngưỡng EER và tính ACER
        y_pred_bin = (y_pred_probs >= eer_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred_bin).ravel()
        
        bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        acer = (apcer + bpcer) / 2

        logs['val_apcer'] = apcer
        logs['val_bpcer'] = bpcer
        logs['val_eer'] = eer
        logs['val_acer'] = acer

        if acer < self.best_acer:
            self.best_acer = acer
            self.best_threshold = eer_threshold
    
            if self.threshold_path:
                os.makedirs(os.path.dirname(self.threshold_path), exist_ok=True)
                with open(self.threshold_path, "w") as f:
                    f.write(str(eer_threshold))
        
            print(f" - APCER: {apcer*100:.2f}% - BPCER: {bpcer*100:.2f}% - EER: {eer*100:.2f}% - ACER: {acer*100:.2f}% (t*: {eer_threshold:.4f}) ⭐ NEW BEST")
        else:
            print(f" - APCER: {apcer*100:.2f}% - BPCER: {bpcer*100:.2f}% - EER: {eer*100:.2f}% - ACER: {acer*100:.2f}% (t*: {eer_threshold:.4f}) ")

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

    # Accurancy
    ax2.plot(history_dict['accuracy'], label='Train Accuracy', color='blue', marker='o')
    ax2.plot(history_dict['val_accuracy'], label='Val Accuracy', color='red', marker='o')
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

def compare_models(model_names, y_probs, y_trues):
    """
    So sánh nhiều mô hình và in ra bảng DataFrame đẹp mắt.
    Tính toán threshold EER cục bộ cho từng model để report.
    """
    results = []
    for name, y_prob, y_true in zip(model_names, y_probs, y_trues):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.absolute(fnr - fpr))
        
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        best_thresh = thresholds[eer_idx]
        if np.isinf(best_thresh) or best_thresh > 1.0:
            best_thresh = 1.0
            
        y_pred = (y_prob >= best_thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
        apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        acer = (apcer + bpcer) / 2
        
        auc = roc_auc_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)
        
        results.append({
            "Model": name,
            "EER (%)": eer * 100,
            "APCER (%)": apcer * 100,
            "BPCER (%)": bpcer * 100,
            "ACER (%)": acer * 100,
            "AUC": auc,
            "ACC (%)": acc * 100
        })
        
    df = pd.DataFrame(results)
    
    try:
        from IPython.display import display
        # Tô đậm 3 cột cuối (ACER, AUC, ACC)
        styled_df = df.style.apply(lambda col: ['font-weight: bold; color: #d62828' if col.name in ['ACER (%)', 'AUC', 'ACC (%)'] else '' for _ in col], axis=0)
        styled_df = styled_df.format({
            "EER (%)": "{:.2f}",
            "APCER (%)": "{:.2f}",
            "BPCER (%)": "{:.2f}",
            "ACER (%)": "{:.2f}",
            "AUC": "{:.4f}",
            "ACC (%)": "{:.2f}"
        })
        display(styled_df)
    except ImportError:
        print(df)
        
    return df
