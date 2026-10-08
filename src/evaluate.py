import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import roc_curve, confusion_matrix

class MetricsCallback(Callback):
    def __init__(self, val_dataset):
        super().__init__()
        self.val_dataset = val_dataset
        # Trích xuất nhãn thực tế (y_true) 1 lần duy nhất ở hàm init để tiết kiệm thời gian tính toán
        self.y_true = np.concatenate([y for x, y in val_dataset], axis=0).flatten()
        # Lưu lại điểm ACER tốt nhất
        self.best_acer = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Dự đoán toàn bộ tập Validation (tắt verbose để đỡ rác terminal)
        y_pred_probs = self.model.predict(self.val_dataset, verbose=0).flatten()
        
        # Tính toán EER và tìm ngưỡng tối ưu (Threshold)
        fpr, tpr, thresholds = roc_curve(self.y_true, y_pred_probs)
        fnr = 1 - tpr
        eer_index = np.nanargmin(np.absolute((fnr - fpr)))
        eer_threshold = thresholds[eer_index]
        eer = (fpr[eer_index] + fnr[eer_index]) / 2
        
        # Phân loại lại dựa trên ngưỡng EER và tính ACER
        y_pred_bin = (y_pred_probs >= eer_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred_bin).ravel()
        
        bpcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        apcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        acer = (apcer + bpcer) / 2
        
        logs['val_eer'] = eer
        logs['val_acer'] = acer
        
        print(f"APCER: {apcer*100:.2f}% - BPCER: {bpcer*100:.2f}% - EER: {eer*100:.2f}% - ACER: {acer*100:.2f}% (t*: {eer_threshold:.4f})", end="")
        
        # Thông báo nổi bật nếu ACER tốt hơn
        if acer < self.best_acer:
            if self.best_acer == float('inf'):
                prev_text = "lần đầu tiên"
            else:
                prev_text = f"từ {self.best_acer*100:.2f}%"
            
            print(f"\n\033[1;92m KẾT QUẢ TỐT NHẤT MỚI: ACER giảm {prev_text} xuống còn {acer*100:.2f}%!\033[0m")
            self.best_acer = acer
        else:
            print()
