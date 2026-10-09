from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Rescaling
from keras.applications import EfficientNetB0

def backbone_model(input_shape=(224, 224, 3), num_classes=1, freeze_backbone=True):
    """
    Khởi tạo mô hình sử dụng Transfer Learning với Backbone EfficientNetB0.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Rescaling(255.0))

    backbone = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,  
        weights='imagenet'
    )
    
    backbone.trainable = not freeze_backbone
    model.add(backbone)

    model.add(GlobalAveragePooling2D())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation='sigmoid', name='output'))
    
    return model