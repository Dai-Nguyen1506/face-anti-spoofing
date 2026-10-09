from keras.models import Sequential
from keras.layers import Input, Conv2D, SeparableConv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense

def custom_model(input_shape=(224, 224, 3), num_classes=1):
    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(Conv2D(16, (3, 3), strides=2, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('silu'))

    model.add(SeparableConv2D(32, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('silu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(SeparableConv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('silu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(SeparableConv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('silu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(SeparableConv2D(256, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('silu'))

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='silu'))
    model.add(Dense(num_classes, activation='sigmoid'))

    return model