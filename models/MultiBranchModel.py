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


# GAP + GMP
def pool(f):
    """Ghép Global Average Pooling và Global Max Pooling: (B, h, w, C) -> (B, 2C).

    GAP cho cái nhìn chung của cả ảnh, GMP bắt dấu hiệu mạnh nhất ở một vùng nhỏ
    (dấu vết giả mạo thường chỉ xuất hiện cục bộ).
    """
    avg = GlobalAveragePooling2D()(f)
    maximum = GlobalMaxPooling2D()(f)
    return Concatenate()([avg, maximum])


def residual_block(x, filters, strides=1):
    """Khối residual: 2 lớp SeparableConv + đường tắt (shortcut)."""
    # Đường tắt: nếu đổi kích thước hoặc số kênh thì dùng conv 1x1 để khớp lại
    shortcut = x
    if strides != 1 or x.shape[-1] != filters:
        if strides != 1:
            shortcut = AveragePooling2D(strides, padding="same")(shortcut)
        shortcut = Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Đường chính
    out = SeparableConv2D(filters, 3, strides=strides, padding="same", use_bias=False)(x)
    out = BatchNormalization()(out)
    out = ReLU()(out)

    out = SeparableConv2D(filters, 3, strides=1, padding="same", use_bias=False)(out)
    out = BatchNormalization()(out)

    # Cộng hai đường lại
    out = Add()([out, shortcut])
    return ReLU()(out)


def create_blocks(x, stages):
    """Xếp tuần tự nhiều khối residual theo danh sách (số kênh, số khối).

    Khối đầu của mỗi giai đoạn giảm kích thước ảnh một nửa, các khối sau giữ nguyên.
    """
    for filters, num_blocks in stages:
        for i in range(num_blocks):
            strides = 2 if i == 0 else 1
            x = residual_block(x, filters, strides)
    return x


def create_trunk(x):
    """Thân sâu: sau mỗi giai đoạn lấy vector pool rồi ghép lại, để có cả đặc trưng mức thấp lẫn mức cao."""
    vectors = []
    for stage in TRUNK:
        x = create_blocks(x, [stage])
        vectors.append(pool(x))
    return Concatenate()(vectors)


def create_branch(x):
    """Nhánh riêng cho từng loại đặc trưng: (h, w) -> (h/4, w/4), 32 kênh."""
    x = BatchNormalization(epsilon=1e-5)(x)

    # Lớp conv đầu (h/2)
    x = Conv2D(16, 3, strides=2, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 1 khối residual (h/4)
    return create_blocks(x, [(32, 1)])


def multibranch_model(input_shape=(224, 224, 3), num_classes=1):
    h, w, _ = input_shape

    inputs = Input(shape=input_shape, name="rgb_input")
    x = Rescaling(1.0 / 255, name="rescale")(inputs)

    # Các kênh đặc trưng cho 4 nhánh
    color = Lambda(color_channels, output_shape=(h, w, COLOR_CHANNELS), name="color_channels")(x)  # RGB + HSV + YUV
    noise = Lambda(noise_channels, output_shape=(h, w, NOISE_CHANNELS), name="noise_channels")(x)  # ảnh trừ bản làm mờ
    edge = Lambda(edge_channels, output_shape=(h, w, EDGE_CHANNELS), name="edge_channels")(x)      # Sobel + độ nhám cục bộ
    fft = Lambda(fft_channels, output_shape=(h, w, FFT_CHANNELS), name="fft_channels")(x)          # FFT

    # Giai đoạn 1: 4 nhánh riêng, mỗi nhánh học đặc trưng mức thấp của loại thông tin đó
    color_map = create_branch(color)
    noise_map = create_branch(noise)
    edge_map = create_branch(edge)
    fft_map = create_branch(fft)

    # Giai đoạn 2a: gộp Màu + Nhiễu + Cạnh (cùng vị trí trên ảnh gốc) rồi đi qua thân sâu
    spatial_map = Concatenate(name="spatial_map")([color_map, noise_map, edge_map])
    spatial_map = Conv2D(64, 1, padding="same", use_bias=False)(spatial_map)   # conv 1x1: trộn 3 nhánh, giảm số kênh
    spatial_map = BatchNormalization()(spatial_map)
    spatial_map = ReLU()(spatial_map)
    spatial_vector = create_trunk(spatial_map)

    # Giai đoạn 2b: FFT đi riêng qua một thân sâu khác
    # (ảnh phổ có trục là tần số chứ không phải vị trí, nên không ghép chồng với ảnh gốc)
    fft_vector = create_trunk(fft_map)

    # Giai đoạn 3: ghép 2 vector ở cuối
    combined_vector = Concatenate(name="combined_vector")([
        spatial_vector,
        fft_vector,
    ])

    x = Dropout(0.3)(combined_vector)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    final_output = Dense(num_classes, activation="sigmoid", name="final_output")(x)

    model = Model(inputs=inputs, outputs=final_output, name="multibranch_model")
    return model