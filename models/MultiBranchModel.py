import os
import sys
sys.path.append(os.path.abspath('..'))

from keras import Model
from keras.layers import Concatenate, Input, Lambda, Rescaling, Add
from keras.layers import Dense, Conv2D, SeparableConv2D, BatchNormalization, ReLU, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D

from src.transforms import color_channels, noise_channels, edge_channels, fft_channels

COLOR_CHANNELS = 9
NOISE_CHANNELS = 3
EDGE_CHANNELS = 3
FFT_CHANNELS = 1
TRUNK = [(64, 3), (128, 3), (256, 3)]

def pool(f):
    """Ghép Global Average Pooling và Global Max Pooling: (B, h, w, C) -> (B, 2C)"""
    avg = GlobalAveragePooling2D()(f)
    maximum = GlobalMaxPooling2D()(f)
    return Concatenate()([avg, maximum])


def residual_block(x, filters, strides=1):
    """Khối residual: 2 lớp SeparableConv + đường tắt (shortcut)."""
    shortcut = x
    if strides != 1 or x.shape[-1] != filters:
        if strides != 1:
            shortcut = AveragePooling2D(strides, padding="same")(shortcut)
        shortcut = Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    out = SeparableConv2D(filters, 3, strides=strides, padding="same", use_bias=False)(x)
    out = BatchNormalization()(out)
    out = ReLU()(out)

    out = SeparableConv2D(filters, 3, strides=1, padding="same", use_bias=False)(out)
    out = BatchNormalization()(out)

    out = Add()([out, shortcut])
    return ReLU()(out)


def create_blocks(x, stages):
    """Tạo một chuỗi các khối residual, mỗi stage có số kênh và số khối khác nhau."""
    for filters, num_blocks in stages:
        for i in range(num_blocks):
            strides = 2 if i == 0 else 1
            x = residual_block(x, filters, strides)
    return x


def create_trunk(x):
    """Tạo một thân sâu từ các khối residual, rồi ghép GAP + GMP để ra vector đặc trưng."""
    vectors = []
    for stage in TRUNK:
        x = create_blocks(x, [stage])
        vectors.append(pool(x))
    return Concatenate()(vectors)


def create_branch(x):
    """Nhánh riêng cho từng loại đặc trưng"""
    x = BatchNormalization(epsilon=1e-5)(x)

    # Lớp conv đầu cho mỗi nhánh, giảm kích thước không gian và tăng số kênh
    x = Conv2D(16, 3, strides=2, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Tạo khối residual
    return create_blocks(x, [(32, 1)])


def multibranch_model(input_shape=(224, 224, 3), num_classes=1):
    """Mạng đa nhánh với 4 nhánh đặc trưng (Màu, Nhiễu, Cạnh, FFT)"""
    h, w, _ = input_shape

    inputs = Input(shape=input_shape, name="rgb_input")
    x = Rescaling(1.0 / 255, name="rescale")(inputs)

    # Chuyển đổi ảnh gốc sang 4 loại đặc trưng: Màu, Nhiễu, Cạnh, FFT
    color = Lambda(color_channels, output_shape=(h, w, COLOR_CHANNELS), name="color_channels")(x)
    noise = Lambda(noise_channels, output_shape=(h, w, NOISE_CHANNELS), name="noise_channels")(x)
    edge = Lambda(edge_channels, output_shape=(h, w, EDGE_CHANNELS), name="edge_channels")(x)
    fft = Lambda(fft_channels, output_shape=(h, w, FFT_CHANNELS), name="fft_channels")(x)

    # Chuyền từng loại đặc trưng qua từng nhánh
    color_map = create_branch(color)
    noise_map = create_branch(noise)
    edge_map = create_branch(edge)
    fft_map = create_branch(fft)

    # Ghép 3 nhánh đầu (Màu, Nhiễu, Cạnh) thành một bản đồ không gian, rồi đi qua một thân sâu
    spatial_map = Concatenate(name="spatial_map")([color_map, noise_map, edge_map])
    spatial_map = Conv2D(64, 1, padding="same", use_bias=False)(spatial_map)
    spatial_map = BatchNormalization()(spatial_map)
    spatial_map = ReLU()(spatial_map)
    spatial_vector = create_trunk(spatial_map)

    # Chỉ nhánh FFT đi qua một thân sâu riêng, để giữ lại thông tin tần số
    fft_vector = create_trunk(fft_map)

    # Gộp hai vector đặc trưng rồi đi qua một vài lớp fully-connected để ra dự đoán cuối cùng
    combined_vector = Concatenate(name="combined_vector")([
        spatial_vector,
        fft_vector,
    ])

    # Lớp fully-connected cuối cùng
    x = Dropout(0.3)(combined_vector)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    final_output = Dense(num_classes, activation="sigmoid", name="final_output")(x)

    model = Model(inputs=inputs, outputs=final_output, name="multibranch_model")
    return model