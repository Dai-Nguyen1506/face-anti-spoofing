import tensorflow as tf
from keras.utils import register_keras_serializable

# Nhánh Màu: RGB + HSV + YUV (ảnh vào đã nằm trong [0, 1])
@register_keras_serializable()
def color_channels(x):
    """(B, H, W, 9): [R, G, B, H, S, V, Y, U, V]."""
    hsv = tf.image.rgb_to_hsv(x)
    yuv = tf.image.rgb_to_yuv(x)
    return tf.concat([x, hsv, yuv], axis=-1)


# Nhánh Nhiễu: ảnh trừ bản làm mờ, chỉ còn lại chi tiết nhỏ và nhiễu
@register_keras_serializable()
def noise_channels(x):
    """(B, H, W, 3): phần dư nhiễu của từng kênh R, G, B."""
    blur = tf.nn.avg_pool2d(x, ksize=3, strides=1, padding="SAME")
    return x - blur


# Nhánh Cạnh: kết cấu bề mặt bằng Sobel và độ nhám cục bộ
@register_keras_serializable()
def edge_channels(x):
    """(B, H, W, 3): [Sobel dọc, Sobel ngang, độ lệch chuẩn cục bộ] trên ảnh xám."""
    gray = tf.image.rgb_to_grayscale(x)       # (B, H, W, 1)

    # Cạnh Sobel
    edges = tf.image.sobel_edges(gray)        # (B, H, W, 1, 2)
    edges = edges[:, :, :, 0, :]              # (B, H, W, 2)

    # Độ lệch chuẩn trong cửa sổ 5x5: vùng nhẵn thì nhỏ, vùng nhiều chi tiết thì lớn
    mean = tf.nn.avg_pool2d(gray, ksize=5, strides=1, padding="SAME")
    mean_sq = tf.nn.avg_pool2d(gray ** 2, ksize=5, strides=1, padding="SAME")
    local_std = tf.sqrt(tf.maximum(mean_sq - mean ** 2, 1e-6))

    return tf.concat([edges, local_std], axis=-1)


# Nhánh FFT: tần số không gian
@register_keras_serializable()
def fft_channels(x):
    """(B, H, W, 1): log-magnitude của FFT trên ảnh xám, tần số thấp được dịch vào giữa."""
    gray = tf.image.rgb_to_grayscale(x)[..., 0]               # (B, H, W)
    f = tf.signal.fft2d(tf.cast(gray, tf.complex64))
    magnitude = tf.math.log1p(tf.abs(f))
    magnitude = tf.signal.fftshift(magnitude, axes=[1, 2])    # chỉ dịch theo H và W, không dịch batch
    return magnitude[..., tf.newaxis]