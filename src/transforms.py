import tensorflow as tf

def conv2d_fixed(x, kernels):
    """x: (B,H,W,1); kernels: (kh,kw,1,out) cố định."""
    return tf.nn.conv2d(x, kernels, strides=1, padding="SAME")

def gray(x):                                   # (B,H,W,1)
    return tf.image.rgb_to_grayscale(x)

# ---- Nhánh màu: Y, Cr, Cb, S, V ----
def make_color(x):
    ycbcr = tf.image.rgb_to_yuv(x)             # Y,U,V (xấp xỉ YCbCr); cần YCrCb chuẩn thì dùng ma trận bên dưới
    hsv = tf.image.rgb_to_hsv(x)               # H,S,V
    y, u, v = tf.split(ycbcr, 3, axis=-1)
    s, val = hsv[..., 1:2], hsv[..., 2:3]      # bỏ H
    return tf.concat([y, u, v, s, val], axis=-1)   # 5 kênh

# ---- Nhánh nhiễu: SRM + FFT ----
SRM = tf.constant([
    [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],
    [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],
    [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],
], dtype=tf.float32)
SRM = tf.transpose(SRM / tf.constant([4., 12., 2.])[:, None, None], [1, 2, 0])[:, :, None, :]  # (5,5,1,3)

def make_noise(x):
    g = gray(x)
    srm = tf.clip_by_value(conv2d_fixed(g, SRM), -2, 2) / 2          # (B,H,W,3)
    spec = tf.signal.fftshift(tf.abs(tf.signal.fft2d(
        tf.cast(tf.squeeze(g, -1), tf.complex64))), axes=(-2, -1))
    spec = tf.math.log1p(spec)[..., None]
    spec = (spec - tf.reduce_mean(spec, [1, 2], keepdims=True)) / (tf.math.reduce_std(spec, [1, 2], keepdims=True) + 1e-6)
    return tf.concat([srm, spec], axis=-1)                           # 4 kênh

# ---- Nhánh kết cấu: LBP, DoG, Sobel (+ HOG sau) ----
def make_lbp(g):
    pad = tf.pad(g, [[0,0],[1,1],[1,1],[0,0]], mode="REFLECT")
    H, W = tf.shape(g)[1], tf.shape(g)[2]
    offs = [(0,0),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0)]
    code = tf.zeros_like(g)
    for i, (dy, dx) in enumerate(offs):
        nb = pad[:, dy:dy+H, dx:dx+W, :]
        code += tf.cast(nb >= g, tf.float32) * (2 ** i)
    return code / 255.0

def gaussian_kernel(sigma, size=9):
    ax = tf.range(-(size//2), size//2 + 1, dtype=tf.float32)
    k = tf.exp(-(ax**2) / (2 * sigma**2)); k /= tf.reduce_sum(k)
    return (k[:, None] * k[None, :])[:, :, None, None]

def make_texture(x):
    g = gray(x)
    lbp = make_lbp(g)
    dog1 = conv2d_fixed(g, gaussian_kernel(1.0)) - conv2d_fixed(g, gaussian_kernel(2.0))
    dog2 = conv2d_fixed(g, gaussian_kernel(2.0)) - conv2d_fixed(g, gaussian_kernel(4.0))
    sob = tf.image.sobel_edges(g)                                    # (B,H,W,1,2)
    mag = tf.sqrt(tf.reduce_sum(sob**2, -1) + 1e-6)
    return tf.concat([lbp, dog1 * 4, dog2 * 4, mag], axis=-1)        # 4 kênh