# Reference: CBAM.
# https://arxiv.org/pdf/1807.06521

# Reference: Multi-Stream CNN
# https://link.springer.com/chapter/10.1007/978-3-032-29909-3_43

import os
import sys
sys.path.append(os.path.abspath('..'))

import tensorflow as tf
from keras import Model
from keras.models import Sequential
from keras.layers import Concatenate, Layer, Input, Lambda
from keras.layers import Dense, Conv2D, SeparableConv2D, BatchNormalization, ReLU, Dropout, Normalization
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.utils import register_keras_serializable

from src.transforms import texture_channels, color_channels, fft_channels, srm_channels, rgb_downsample
from src.transforms import LOW_RES, fft_width
from src.transforms import TEXTURE_CHANNELS, COLOR_CHANNELS, FFT_CHANNELS, SRM_CHANNELS


# GAP + GMP
def pool(f):
    """Ghép Global Average Pooling và Global Max Pooling: (B, h, w, C) -> (B, 2C)."""
    avg = GlobalAveragePooling2D()(f)
    maximum = GlobalMaxPooling2D()(f)
    return Concatenate()([avg, maximum])


@register_keras_serializable()
class SAM(Layer):
    def __init__(self, bias=False, **kwargs):
        super(SAM, self).__init__(**kwargs)
        self.bias = bias
        self.conv = Conv2D(filters=1, kernel_size=7, strides=1, padding="same",
                           dilation_rate=1, use_bias=self.bias)

    def call(self, x):
        max = tf.reduce_max(x, axis=-1, keepdims=True)
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        concat = tf.concat([max, avg], axis=-1)
        output = self.conv(concat)
        output = tf.sigmoid(output) * x
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"bias": self.bias})
        return config


# Channel Attention Module
@register_keras_serializable()
class CAM(Layer):
    def __init__(self, channels, r, **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.channels = channels
        self.r = r
        self.linear = Sequential([
            Dense(self.channels // self.r, activation="relu", use_bias=True),
            Dense(self.channels, use_bias=True),
        ])

    def call(self, x):
        max = tf.reduce_max(x, axis=[1, 2])
        avg = tf.reduce_mean(x, axis=[1, 2])
        linear_max = tf.reshape(self.linear(max), (-1, 1, 1, self.channels))
        linear_avg = tf.reshape(self.linear(avg), (-1, 1, 1, self.channels))
        output = linear_max + linear_avg
        output = tf.sigmoid(output) * x
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels, "r": self.r})
        return config


# Convolutional Block Attention Module
@register_keras_serializable()
class CBAM(Layer):
    def __init__(self, channels, r=16, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def call(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels, "r": self.r})
        return config


def create_branch(input_shape, name, first_kernel=3, dilation=1):
    """Nhánh CNN nhỏ với 4 khối Conv2D + BN + ReLU. Khối cuối có thể giãn nở (dilation). Cuối cùng là CBAM."""
    cnn = Sequential(name=name)

    cnn.add(Input(shape=input_shape))
    cnn.add(BatchNormalization())

    # Khối 1
    cnn.add(Conv2D(32, first_kernel, strides=2, padding="same", use_bias=False))
    cnn.add(BatchNormalization())
    cnn.add(ReLU())

    # Khối 2
    cnn.add(SeparableConv2D(64, 3, strides=2, padding="same", use_bias=False))
    cnn.add(BatchNormalization())
    cnn.add(ReLU())

    # Khối 3
    cnn.add(SeparableConv2D(128, 3, strides=2, padding="same", use_bias=False))
    cnn.add(BatchNormalization())
    cnn.add(ReLU())

    # Khối 4
    if dilation > 1:
        cnn.add(SeparableConv2D(128, 3, strides=1, dilation_rate=dilation, padding="same", use_bias=False))
    else:
        cnn.add(SeparableConv2D(128, 3, strides=2, padding="same", use_bias=False))
    cnn.add(BatchNormalization())
    cnn.add(ReLU())

    # CBAM
    cnn.add(CBAM(128))

    return cnn


def multibranch_model(input_shape=(224, 224, 3), num_classes=1):
    h, w, _ = input_shape

    inputs = Input(shape=input_shape, name="rgb_input")

    # Nhánh 1: Texture (LBP one-hot + DoG)
    texture = Lambda(texture_channels, output_shape=(h, w, TEXTURE_CHANNELS), name="texture_channels")(inputs)
    texture = Normalization(name="texture_norm")(texture)
    texture_branch = create_branch((h, w, TEXTURE_CHANNELS), name="texture_branch")
    texture_vector = pool(texture_branch(texture))

    # Nhánh 2: Color (Cr, Cb, S, V, A, B)
    color = Lambda(color_channels, output_shape=(h, w, COLOR_CHANNELS), name="color_channels")(inputs)
    color = Normalization(name="color_norm")(color)
    color_branch = create_branch((h, w, COLOR_CHANNELS), name="color_branch")
    color_vector = pool(color_branch(color))

    # Nhánh 3: RGB (conv đầu 7x7 và khối cuối giãn nở -> nhìn toàn cục)
    rgb_low = Lambda(rgb_downsample, output_shape=(h, w, 3), name="rgb_low")(inputs)
    rgb_branch = create_branch((h, w, 3), name="rgb_branch", first_kernel=7, dilation=4)
    rgb_vector = pool(rgb_branch(rgb_low))

    # Nhánh 4: FFT (nửa phổ + kênh khoảng cách hướng tâm; đã chuẩn hóa theo từng ảnh trong transform)
    fft = Lambda(fft_channels, output_shape=(h, fft_width(w), FFT_CHANNELS), name="fft_channels")(inputs)
    fft_branch = create_branch((h, fft_width(w), FFT_CHANNELS), name="fft_branch")
    fft_vector = pool(fft_branch(fft))

    # Nhánh 5: SRM
    srm = Lambda(srm_channels, output_shape=(h, w, SRM_CHANNELS), name="srm_channels")(inputs)
    srm = Normalization(name="srm_norm")(srm)
    srm_branch = create_branch((h, w, SRM_CHANNELS), name="srm_branch")
    srm_vector = pool(srm_branch(srm))

    # Đầu ra từng nhánh (để so sánh)
    texture_output = Dense(num_classes, activation="sigmoid", name="texture_output")(texture_vector)
    color_output = Dense(num_classes, activation="sigmoid", name="color_output")(color_vector)
    rgb_output = Dense(num_classes, activation="sigmoid", name="rgb_output")(rgb_vector)
    fft_output = Dense(num_classes, activation="sigmoid", name="fft_output")(fft_vector)
    srm_output = Dense(num_classes, activation="sigmoid", name="srm_output")(srm_vector)

    # Ghép các vector của 5 nhánh
    combined_vector = Concatenate(name="combined_vector")([
        texture_vector,
        color_vector,
        rgb_vector,
        fft_vector,
        srm_vector,
    ])

    x = Dropout(0.3)(combined_vector)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    final_output = Dense(num_classes, activation="sigmoid", name="final_output")(x)

    model = Model(
        inputs=inputs,
        outputs={
            "texture_output": texture_output,
            "color_output": color_output,
            "rgb_output": rgb_output,
            "fft_output": fft_output,
            "srm_output": srm_output,
            "final_output": final_output,
        },
        name="multibranch_model",
    )
    return model