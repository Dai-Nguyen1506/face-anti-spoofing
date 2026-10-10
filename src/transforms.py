import functools

import numpy as np
import tensorflow as tf
from keras.utils import register_keras_serializable

# Hệ số giảm độ phân giải cho nhánh màu và nhánh RGB toàn cục
LOW_RES = 2

# Số kênh đầu ra của từng hàm (model dùng để khai báo output_shape)
TEXTURE_CHANNELS = 11   # 10 (LBP one-hot) + 1 (DoG)
COLOR_CHANNELS = 6
FFT_CHANNELS = 2        # log-magnitude + khoảng cách hướng tâm
SRM_CHANNELS = 3


def fft_width(w):
    """Chiều rộng ảnh phổ khi dùng nửa phổ (rfft2d)."""
    return w // 2 + 1


# Chuyển ảnh RGB sang ảnh xám
def rgb_to_gray(x):
    w = tf.constant([0.299, 0.587, 0.114], dtype=x.dtype)
    return tf.reduce_sum(x * w, axis=-1, keepdims=True)


# Giảm độ phân giải (trung bình 2x2) cho nhánh màu và nhánh RGB
@register_keras_serializable()
def rgb_downsample(x):
    return tf.nn.avg_pool2d(x, LOW_RES, LOW_RES, padding="VALID")


# Nhánh 1: DoG + LBP - Kết cấu bề mặt
def _gauss_1d(sigma, radius):
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    return k / k.sum()


_DOG_SIGMA1, _DOG_SIGMA2 = 1.0, 2.0
_DOG_R1 = max(1, int(3 * _DOG_SIGMA1))
_DOG_R2 = max(1, int(3 * _DOG_SIGMA2))
_k1 = np.pad(_gauss_1d(_DOG_SIGMA1, _DOG_R1), (_DOG_R2 - _DOG_R1, _DOG_R2 - _DOG_R1))
_k2 = _gauss_1d(_DOG_SIGMA2, _DOG_R2)
# Lượt 1 (theo chiều rộng): 1 kênh vào -> 2 kênh (blur sigma1, blur sigma2)
_DOG_KX = np.stack([_k1, _k2], axis=-1).reshape(1, -1, 1, 2).astype(np.float32)
# Lượt 2 (theo chiều cao): 2 kênh vào -> 1 kênh ra, trọng số [+k1, -k2] => blur1 - blur2
_DOG_KY = np.stack([_k1, -_k2], axis=-1).reshape(-1, 1, 2, 1).astype(np.float32)


def dog(gray):
    """Difference of Gaussians (sigma 1 và 2). gray: (B, H, W, 1) -> (B, H, W, 1)."""
    r = _DOG_R2
    x = tf.pad(gray, [[0, 0], [r, r], [r, r], [0, 0]], mode="REFLECT")
    x = tf.nn.conv2d(x, tf.constant(_DOG_KX, dtype=gray.dtype), strides=1, padding="VALID")
    x = tf.nn.conv2d(x, tf.constant(_DOG_KY, dtype=gray.dtype), strides=1, padding="VALID")
    return x


def lbp_onehot(gray):
    """LBP uniform rotation-invariant (P=8, R=1) dạng one-hot: (B, H, W, 10).

    Mã 0..8 = số bit 1 của mẫu uniform, mã 9 = mẫu không uniform.
    Bản đơn giản: lấy 8 điểm lân cận trên lưới nguyên (không nội suy).
    """
    shape = tf.shape(gray)
    h, w = shape[1], shape[2]
    center = gray[..., 0]  # (B, H, W)
    # SYMMETRIC với biên 1 pixel = lặp lại pixel biên (giống replicate)
    p = tf.pad(gray, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")
    # 8 lân cận theo chiều kim đồng hồ, bắt đầu từ góc trên-trái
    offsets = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    # Mỗi bit là một mặt phẳng (B, H, W) float: cộng/trừ từng mặt phẳng nhanh hơn ~4 lần
    # so với gom thành tensor (B, H, W, 8) rồi roll/reduce trên trục cuối.
    bits = [tf.cast(p[:, dy:dy + h, dx:dx + w, 0] >= center, tf.float32) for dy, dx in offsets]
    ones = tf.add_n(bits)
    transitions = tf.add_n([tf.abs(bits[i] - bits[(i + 1) % 8]) for i in range(8)])
    code = tf.where(transitions <= 2.0, ones, 9.0)  # (B, H, W), giá trị nguyên 0..9
    return tf.cast(code[..., tf.newaxis] == tf.range(10, dtype=tf.float32), gray.dtype)


@register_keras_serializable()
def texture_channels(x):
    """(B, H, W, 11): [LBP one-hot (10 kênh), DoG]."""
    gray = rgb_to_gray(x)
    return tf.concat([lbp_onehot(gray), dog(gray)], axis=-1)


# Nhánh 2: Đặc điểm màu sắc từ YCrBr, HSV và LaB - Cr, Br, S, V, A, B
def _rgb_to_lab_ab(x):
    """Trả về kênh a, b của CIELAB (chia 128 để về khoảng ~[-1, 1])."""
    x = tf.clip_by_value(x, 0.0, 1.0)
    lin = tf.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    m = tf.constant([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
        ],
        dtype=x.dtype,
    )
    xyz = tf.einsum("bhwc,dc->bhwd", lin, m)
    white = tf.constant([0.95047, 1.0, 1.08883], dtype=x.dtype)
    t = xyz / white
    f = tf.where(t > 0.008856, tf.maximum(t, 1e-6) ** (1.0 / 3.0), 7.787 * t + 16.0 / 116.0)
    fx, fy, fz = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    a = 500.0 * (fx - fy) / 128.0
    b = 200.0 * (fy - fz) / 128.0
    return a, b


@register_keras_serializable()
def color_channels(x):
    """(B, H/2, W/2, 6): [Cr, Cb, S, V, A, B], tính trên ảnh đã giảm một nửa độ phân giải."""
    x = rgb_downsample(x)
    y = rgb_to_gray(x)
    cr = (x[..., 0:1] - y) * 0.713 + 0.5
    cb = (x[..., 2:3] - y) * 0.564 + 0.5
    v = tf.reduce_max(x, axis=-1, keepdims=True)
    s = (v - tf.reduce_min(x, axis=-1, keepdims=True)) / (v + 1e-6)
    a, b = _rgb_to_lab_ab(x)
    return tf.concat([cr, cb, s, v, a, b], axis=-1)


# Nhánh 4: FFT - Tần số không gian
@functools.lru_cache(maxsize=None)
def _fft_constants(h, w):
    """Cửa sổ Hann (h, w) và bản đồ khoảng cách hướng tâm (1, h, w//2+1, 1) trong [0, 1]."""
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    ky = np.fft.fftshift(np.fft.fftfreq(h))   # [-0.5, 0.5), đã dịch tâm theo hàng
    kx = np.fft.rfftfreq(w)                   # [0, 0.5]
    radius = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2) / np.sqrt(0.5)
    return window, radius[None, :, :, None].astype(np.float32)


@register_keras_serializable()
def fft_channels(x):
    """(B, H, W//2+1, 2): [log-magnitude FFT chuẩn hóa theo từng ảnh, khoảng cách hướng tâm].

    Ảnh xám được trừ trung bình rồi nhân cửa sổ Hann trước FFT. Dùng rfft2d nên chỉ giữ nửa
    phổ (cột tần số >= 0); hàng được dịch tâm (fftshift) nên tần số 0 nằm ở giữa mép trái.
    """
    h, w = x.shape[1], x.shape[2]
    if h is None or w is None:
        raise ValueError("fft_channels cần kích thước ảnh cố định (H, W).")
    window, radius = _fft_constants(h, w)
    gray = rgb_to_gray(x)[..., 0]  # (B, H, W)
    gray = (gray - tf.reduce_mean(gray, axis=[1, 2], keepdims=True)) * window
    f = tf.signal.rfft2d(tf.cast(gray, tf.float32))
    f = tf.roll(f, shift=h // 2, axis=1)  # fftshift theo hàng
    m = tf.math.log1p(tf.abs(f))[..., tf.newaxis]
    mean, var = tf.nn.moments(m, axes=[1, 2], keepdims=True)
    m = (m - mean) / (tf.sqrt(var) + 1e-6)
    radius = tf.broadcast_to(tf.constant(radius, dtype=m.dtype), tf.shape(m))
    return tf.concat([m, radius], axis=-1)


_SRM = [
    [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
]
_SRM_SCALE = [4.0, 12.0, 2.0]
# (5, 5, 1, 3): 3 bộ lọc cố định cho 1 kênh vào
_SRM_KERNEL = np.stack(
    [np.array(f, dtype=np.float32) / s for f, s in zip(_SRM, _SRM_SCALE)], axis=-1
)[:, :, None, :]


@register_keras_serializable()
def srm_channels(x):
    """(B, H, W, 3): phần dư nhiễu qua 3 bộ lọc SRM cố định, tính trên ảnh xám."""
    gray = rgb_to_gray(x)
    gray = tf.pad(gray, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="REFLECT")
    return tf.nn.conv2d(gray, tf.constant(_SRM_KERNEL, dtype=x.dtype), strides=1, padding="VALID")