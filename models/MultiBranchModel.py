import os
import sys
sys.path.append(os.path.abspath('..'))

import tensorflow as tf
from keras import layers, Model
from src.transforms import make_color, make_noise, make_texture

def small_branch(inp, name, emb=128):
    x = inp
    for f in (32, 64, 96, emb):
        x = layers.Conv2D(f, 3, 2, "same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(f, 3, 1, "same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    return layers.GlobalAveragePooling2D(name=name)(x)

class BranchDropout(layers.Layer):
    """Zero cả embedding của một nhánh với xác suất p (chỉ khi train)."""
    def __init__(self, p=0.25, **kw): super().__init__(**kw); self.p = p
    def call(self, x, training=None):
        if not training: return x
        keep = tf.cast(tf.random.uniform([tf.shape(x)[0], 1]) > self.p, x.dtype)
        return x * keep

def build_model(size=224, num_classes=2):
    inp = layers.Input((size, size, 3))                     # RGB [0,255]
    x01 = layers.Rescaling(1/255.)(inp)

    # Nhánh A: MobileNetV3 (bản Keras đã tích hợp tiền xử lý, nhận [0,255])
    backbone = tf.keras.applications.MobileNetV3Large(
        include_top=False, 
        weights="imagenet", 
        input_shape=(size, size, 3), 
        include_preprocessing=True
    )
    backbone.trainable = True
    f_rgb = layers.Dense(256, activation="relu")(layers.GlobalAveragePooling2D()(backbone(inp)))

    # Nhánh phụ
    f_col = small_branch(layers.Lambda(make_color)(x01), "f_color")
    f_noi = small_branch(layers.Lambda(make_noise)(x01), "f_noise")
    f_tex = small_branch(layers.Lambda(make_texture)(x01), "f_tex")

    # Head phụ (aux)
    logit_rgb = layers.Dense(num_classes, name="aux_rgb")(f_rgb)
    aux_col = layers.Dense(num_classes, name="aux_color")(f_col)
    aux_noi = layers.Dense(num_classes, name="aux_noise")(f_noi)
    aux_tex = layers.Dense(num_classes, name="aux_tex")(f_tex)

    # Fusion: logit = logit_rgb + delta, delta khởi tạo 0
    z = layers.Concatenate()([BranchDropout()(f_col), BranchDropout()(f_noi), BranchDropout()(f_tex)])
    z = layers.Dense(128, activation="relu")(z); z = layers.Dropout(0.3)(z)
    delta = layers.Dense(num_classes, kernel_initializer="zeros", bias_initializer="zeros")(z)
    logit = layers.Add(name="main")([logit_rgb, delta])

    return Model(inp, [logit, logit_rgb, aux_col, aux_noi, aux_tex])