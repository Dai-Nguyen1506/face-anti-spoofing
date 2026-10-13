from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from keras.applications import EfficientNetB0

def backbone_model(input_shape=(224, 224, 3), num_classes=1, freeze_backbone=True):
    """Khởi tạo mô hình sử dụng Transfer Learning với Backbone EfficientNet-B0."""
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Thêm EfficientNet-B0 làm backbone
    backbone = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,  
        weights='imagenet'
    )

    # Thiết lập trạng thái trainable của backbone dựa trên tham số freeze_backbone
    backbone.trainable = not freeze_backbone
    model.add(backbone)

    # Gộp đặt trưng bằng Global Average Pooling để giảm số lượng tham số
    model.add(GlobalAveragePooling2D())

    # Thêm các lớp Dense và Dropout để giảm overfitting
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='sigmoid', name='output'))
    
    return model