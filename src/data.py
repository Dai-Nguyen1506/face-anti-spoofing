import os
from pathlib import Path

import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Rescaling, RandomFlip, RandomRotation, RandomZoom,
    RandomBrightness, RandomContrast, RandomTranslation,
)

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {'.png'}
CLASS_NAMES = ['real', 'spoof']
SHUFFLE_BUFFER = 2048

def get_data_stats(base_dir):
    """
    Hàm đọc dữ liệu từ thư mục gốc, trả về đường dẫn, danh sách nhãn và phân bổ số lượng.
    """
    splits = ['train', 'val', 'test']
    classes = ['real', 'spoof']
    stats = {}
    paths = {}

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        paths[split] = split_dir
        stats[split] = {}
        
        if os.path.exists(split_dir):
            for cls in classes:
                cls_dir = os.path.join(split_dir, cls)
                if os.path.exists(cls_dir):
                    stats[split][cls] = len(os.listdir(cls_dir))
                else:
                    stats[split][cls] = 0
                    
    return paths, classes, stats


def list_images(folder):
    """Danh sách đường dẫn ảnh (đã sắp xếp) trong một thư mục lớp."""
    return sorted(
        str(p) for p in Path(folder).rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_cache_file(cache_dir, name, img_size):
    """Đường dẫn file cache. Trả '' (= cache trong RAM) nếu cache_dir là None.

    Tên có kèm img_size để đổi kích thước ảnh thì không dùng nhầm cache cũ.
    """
    if cache_dir is None:
        return ''
    os.makedirs(cache_dir, exist_ok=True)
    return str(Path(cache_dir) / f'{name}_{img_size}')


def clear_cache(cache_dir):
    """Xóa toàn bộ file cache trong thư mục (gọi khi dữ liệu thay đổi)."""
    if cache_dir is not None and os.path.isdir(cache_dir):
        for f in Path(cache_dir).glob('*'):
            if f.is_file():
                f.unlink()


def load_image(path, label, img_size):
    """Đọc ảnh, resize, lưu uint8 (0-255) để cache nhẹ. Nhãn shape (1,)."""
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(tf.round(img), tf.uint8)
    label = tf.reshape(tf.cast(label, tf.float32), [1])
    return img, label


def build_augmentation():
    """Augmentation chạy trên ảnh 0-255; Rescaling đặt CUỐI CÙNG.

    RandomBrightness/RandomContrast mặc định value_range=(0, 255),
    nên không được chia 255 trước khi augment.
    """
    return Sequential([
        RandomFlip('horizontal'),
        RandomRotation(0.05),
        RandomZoom(0.2),
        RandomBrightness(0.2),
        RandomContrast(0.2),
        RandomTranslation(0.01, 0.01),
        Rescaling(1. / 255),
    ])


def make_class_ds(paths, label, img_size, seed, cache_file):
    """Dataset vô hạn của MỘT lớp: đọc -> cache -> shuffle lại mỗi vòng -> repeat."""
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: load_image(p, label, img_size), num_parallel_calls=AUTOTUNE)
    ds = ds.cache(cache_file)
    ds = ds.shuffle(SHUFFLE_BUFFER, seed=seed, reshuffle_each_iteration=True)
    ds = ds.repeat()
    return ds


def build_balanced_train_ds(train_dir, img_size, batch_size, cache_dir=None, seed=42):
    """Dataset train: mỗi batch = batch_size//2 real + batch_size//2 spoof.
    """
    assert batch_size % 2 == 0, 'BATCH_SIZE phải chẵn để chia đều 2 lớp'
    half = batch_size // 2

    real_paths = list_images(Path(train_dir) / 'real')
    spoof_paths = list_images(Path(train_dir) / 'spoof')
    print(f'Train: {len(real_paths)} real | {len(spoof_paths)} spoof | 'f'mỗi batch {half} real + {half} spoof')

    real_ds = make_class_ds(
        real_paths, 0, img_size, seed,
        get_cache_file(cache_dir, 'train_real', img_size),
    ).batch(half)
    spoof_ds = make_class_ds(
        spoof_paths, 1, img_size, seed + 1,
        get_cache_file(cache_dir, 'train_spoof', img_size),
    ).batch(half)

    ds = tf.data.Dataset.zip((real_ds, spoof_ds))
    ds = ds.map(
        lambda r, s: (tf.concat([r[0], s[0]], axis=0), tf.concat([r[1], s[1]], axis=0)),
        num_parallel_calls=AUTOTUNE,
    )

    augment = build_augmentation()
    ds = ds.map(
        lambda x, y: (augment(tf.cast(x, tf.float32), training=True), y),
        num_parallel_calls=AUTOTUNE,
    )
    return ds.prefetch(AUTOTUNE)


def build_eval_ds(folder, img_size, batch_size, name, cache_dir=None):
    """Dataset val/test: không shuffle, không augment, chỉ chia 255.
    """
    paths, labels = [], []
    for label, cls in enumerate(CLASS_NAMES):
        cls_paths = list_images(Path(folder) / cls)
        paths += cls_paths
        labels += [label] * len(cls_paths)
    print(f'{name}: {len(paths)} ảnh')

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: load_image(p, l, img_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.cache(get_cache_file(cache_dir, name, img_size))
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)