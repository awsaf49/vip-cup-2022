# SEEDING
import numpy as np
import random
import os
import tensorflow as tf
from dataset.augment import apply_augment


# SEEDING


def seeding(CFG):
    SEED = CFG.seed
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    tf.random.set_seed(SEED)


# DATA PIPELINE

def build_decoder(with_labels, img_size, CFG, ext='jpg'):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == 'png':
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")
        img = tf.cast(img, tf.float32)
        # some images from test have different dimensions
        if tf.math.reduce_any(img_size!=(200, 200)):
                img = tf.image.resize(img, img_size, method=CFG.resize_method)
        img = tf.cast(img, tf.float32)
        # normalization
        img = img / 255.0
        img = tf.reshape(img, [*img_size, 3])
        return img
    
    def decode_with_labels(path, label):
        if CFG.num_classes>1:
            label = tf.one_hot(label,CFG.num_classes)
            label = tf.squeeze(label)
        label = tf.cast(label, tf.float32)
        return decode(path), label
    
    return decode_with_labels if with_labels else decode

def build_augmenter(with_labels=True, img_size=[200,200], CFG=None):
    def augment(img, img_size=img_size):
        img = apply_augment(img, CFG=CFG)        
        img = tf.reshape(img, [*img_size, 3])
        return img
    
    def augment_with_labels(img, label):    
        return augment(img), label
    
    return augment_with_labels if with_labels else augment




def build_dataset(paths, labels=None,
                  batch_size=32, cache=True,
                  decode_fn=None, augment_fn=None, dim=[200, 200],
                  augment=True, repeat=True, shuffle=1024,
                  cache_dir="", drop_remainder=False, CFG=None):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)

    CFG.is_train = labels is not None

    if decode_fn is None:
        decode_fn = build_decoder(
            labels is not None, img_size=CFG.img_size, CFG=CFG)

    if augment_fn is None:
        augment_fn = build_augmenter(
            labels is not None, img_size=CFG.img_size, CFG=CFG)

    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    ds = ds.cache(cache_dir) if cache else ds
    ds = ds.repeat() if repeat else ds
    opt = tf.data.Options()
    if shuffle:
        ds = ds.shuffle(shuffle, seed=CFG.seed)
        opt.experimental_deterministic = False
    if CFG.device == 'GPU':
        opt.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    ds = ds.with_options(opt)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTO)
    return ds



def load_image(image_path, dsize=[224, 224]):
    file_bytes = tf.io.read_file(image_path)
    img        = tf.image.decode_jpeg(file_bytes, 3)
    img        = tf.image.resize(img, dsize)
    return img.numpy()/255.0

    
