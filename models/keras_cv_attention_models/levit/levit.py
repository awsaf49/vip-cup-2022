import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.attention_layers import batchnorm_with_activation, conv2d_no_bias, activation_by_name, add_pre_post_process


PRETRAINED_DICT = {
    "levit128s": {"imagenet": "5e35073bb6079491fb0a1adff833da23"},
    "levit128": {"imagenet": "730c100fa4d5a10cf48fb923bb7da5c3"},
    "levit192": {"imagenet": "b078d2fe27857d0bdb26e101703210e2"},
    "levit256": {"imagenet": "9ada767ba2798c94aa1c894a00ae40fd"},
    "levit384": {"imagenet": "520f207f7f4c626b83e21564dd0c92a3"},
}


@tf.keras.utils.register_keras_serializable(package="levit")
class MultiHeadPositionalEmbedding(keras.layers.Layer):
    def __init__(self, query_height=-1, key_height=-1, **kwargs):
        super(MultiHeadPositionalEmbedding, self).__init__(**kwargs)
        self.query_height, self.key_height = query_height, key_height

    def build(self, input_shape, **kwargs):
        _, num_heads, qq_blocks, kk_blocks = input_shape
        self.bb = self.add_weight(name="positional_embedding", shape=(kk_blocks, num_heads), initializer="zeros", trainable=True)

        if self.query_height == -1:
            q_blocks_h = q_blocks_w = int(tf.math.sqrt(float(qq_blocks)))  # hh == ww
        else:
            q_blocks_h, q_blocks_w = self.query_height, int(qq_blocks / self.query_height)

        strides = int(tf.math.ceil(tf.math.sqrt(float(kk_blocks / qq_blocks))))
        if self.key_height == -1:
            k_blocks_h = q_blocks_h * strides
            while kk_blocks % k_blocks_h != 0:
                k_blocks_h -= 1
            k_blocks_w = int(kk_blocks / k_blocks_h)
        else:
            k_blocks_h, k_blocks_w = self.key_height, int(kk_blocks / self.key_height)
        self.k_blocks_h, self.k_blocks_w = k_blocks_h, k_blocks_w
        # print(f"{q_blocks_h = }, {q_blocks_w = }, {k_blocks_h = }, {k_blocks_w = }, {strides = }")

        x1, y1 = tf.meshgrid(range(q_blocks_h), range(q_blocks_w))
        x2, y2 = tf.meshgrid(range(k_blocks_h), range(k_blocks_w))
        aa = tf.concat([tf.reshape(x1, (-1, 1)), tf.reshape(y1, (-1, 1))], axis=-1)
        bb = tf.concat([tf.reshape(x2, (-1, 1)), tf.reshape(y2, (-1, 1))], axis=-1)
        # print(f">>>> {aa.shape = }, {bb.shape = }") # aa.shape = (16, 2), bb.shape = (49, 2)
        cc = [tf.math.abs(bb - ii * strides) for ii in aa]
        self.bb_pos = tf.stack([ii[:, 0] + ii[:, 1] * k_blocks_h for ii in cc])
        # print(f">>>> {self.bb_pos.shape = }")    # self.bb_pos.shape = (16, 49)

        super(MultiHeadPositionalEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        pos_bias = tf.gather(self.bb, self.bb_pos)
        pos_bias = tf.transpose(pos_bias, [2, 0, 1])
        return inputs + pos_bias

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"query_height": self.query_height, "key_height": self.key_height})
        return base_config

    def load_resized_pos_emb(self, source_layer, method="nearest"):
        if isinstance(source_layer, dict):
            source_bb = source_layer["positional_embedding:0"]  # weights
        else:
            source_bb = source_layer.bb  # layer
        hh = ww = int(tf.math.sqrt(float(source_bb.shape[0])))
        ss = tf.reshape(source_bb, (hh, ww, source_bb.shape[-1]))  # [hh, ww, num_heads]
        # target_hh = target_ww = int(tf.math.sqrt(float(self.bb.shape[0])))
        tt = tf.image.resize(ss, [self.k_blocks_h, self.k_blocks_w], method=method)  # [target_hh, target_ww, num_heads]
        tt = tf.reshape(tt, (self.bb.shape))  # [target_hh * target_ww, num_heads]
        self.bb.assign(tt)

    def show_pos_emb(self, rows=1, base_size=2):
        import matplotlib.pyplot as plt

        hh = ww = int(tf.math.sqrt(float(self.bb.shape[0])))
        ss = tf.reshape(self.bb, (hh, ww, -1)).numpy()
        cols = int(tf.math.ceil(ss.shape[-1] / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(base_size * cols, base_size * rows))
        for id, ax in enumerate(axes.flatten()):
            ax.imshow(ss[:, :, id])
            ax.set_axis_off()
        fig.tight_layout()
        return fig


def scaled_dot_product_attention(qq, kk, vv, key_dim, attn_ratio, output_shape, use_bn=True, out_bias=False, activation=None, name=None):
    height, width, output_dim = output_shape
    # qq, kk, vv: [batch, num_heads, blocks, key_dim]
    qk_scale = float(1.0 / tf.math.sqrt(tf.cast(key_dim, "float32")))
    # print(f"{qq.shape = }, {kk.shape = }")
    # attn = tf.matmul(qq, kk, transpose_b=True) * qk_scale   # [batch, num_heads, q_blocks, k_blocks]
    attn = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1], transpose_b=True))([qq, kk]) * qk_scale
    # print(f"{attn.shape = }")
    attn = MultiHeadPositionalEmbedding(query_height=height, name=name and name + "attn_pos")(attn)
    # attn = tf.nn.softmax(attn, axis=-1)
    attn = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attn)

    # output = tf.matmul(attn, vv)    # [batch, num_heads, q_blocks, key_dim * attn_ratio]
    output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attn, vv])
    output = tf.transpose(output, perm=[0, 2, 1, 3])  # [batch, q_blocks, num_heads, key_dim * attn_ratio]
    output = tf.reshape(output, [-1, height, width, output.shape[2] * output.shape[3]])  # [batch, q_blocks, channel * attn_ratio]
    if activation:
        output = activation_by_name(output, activation=activation, name=name)
    output = keras.layers.Dense(output_dim, use_bias=out_bias, name=name and name + "out")(output)
    if use_bn:
        output = batchnorm_with_activation(output, activation=None, zero_gamma=True, name=name and name + "out_")
    return output


def mhsa_with_multi_head_position(
    inputs, num_heads, key_dim=-1, output_dim=-1, attn_ratio=1, use_bn=True, qkv_bias=False, out_bias=False, activation=None, name=None
):
    _, height, width, input_channels = inputs.shape
    key_dim = key_dim if key_dim > 0 else input_channels // num_heads
    output_dim = output_dim if output_dim > 0 else input_channels
    embed_dim = key_dim * num_heads

    qkv_dim = (attn_ratio + 1 + 1) * embed_dim
    qkv = keras.layers.Dense(qkv_dim, use_bias=qkv_bias, name=name and name + "qkv")(inputs)
    qkv = batchnorm_with_activation(qkv, activation=None, name=name and name + "qkv_") if use_bn else qkv
    qkv = tf.reshape(qkv, (-1, qkv.shape[1] * qkv.shape[2], num_heads, qkv_dim // num_heads))
    qkv = tf.transpose(qkv, perm=[0, 2, 1, 3])
    qq, kk, vv = tf.split(qkv, [key_dim, key_dim, key_dim * attn_ratio], axis=-1)
    output_shape = (height, width, output_dim)
    return scaled_dot_product_attention(qq, kk, vv, key_dim, attn_ratio, output_shape, use_bn, out_bias, activation=activation, name=name)


def mhsa_with_multi_head_position_and_strides(
    inputs, num_heads, key_dim=-1, output_dim=-1, attn_ratio=1, strides=1, use_bn=True, qkv_bias=False, out_bias=False, activation=None, name=None
):
    _, _, _, input_channels = inputs.shape
    key_dim = key_dim if key_dim > 0 else input_channels // num_heads
    output_dim = output_dim if output_dim > 0 else input_channels
    emded_dim = num_heads * key_dim

    embed_dim = key_dim * num_heads

    qq = inputs[:, ::strides, ::strides, :] if strides > 1 else inputs
    height, width = qq.shape[1], qq.shape[2]
    # print(f"{height = }, {width = }, {strides = }, {inputs.shape = }")
    qq = keras.layers.Dense(embed_dim, use_bias=qkv_bias, name=name and name + "q")(qq)
    qq = batchnorm_with_activation(qq, activation=None, name=name and name + "q_") if use_bn else qq
    qq = tf.reshape(qq, [-1, qq.shape[1] * qq.shape[2], num_heads, key_dim])
    qq = tf.transpose(qq, [0, 2, 1, 3])

    kv_dim = (attn_ratio + 1) * embed_dim
    kv = keras.layers.Dense(kv_dim, use_bias=qkv_bias, name=name and name + "kv")(inputs)
    kv = batchnorm_with_activation(kv, activation=None, name=name and name + "kv_") if use_bn else kv
    kv = tf.reshape(kv, (-1, kv.shape[1] * kv.shape[2], num_heads, kv_dim // num_heads))
    kv = tf.transpose(kv, perm=[0, 2, 1, 3])
    kk, vv = tf.split(kv, [key_dim, key_dim * attn_ratio], axis=-1)
    output_shape = (height, width, output_dim)
    # print(f"{qq.shape = }, {kk.shape = }, {vv.shape = }, {output_shape = }")
    return scaled_dot_product_attention(qq, kk, vv, key_dim, attn_ratio, output_shape, use_bn, out_bias, activation=activation, name=name)


def res_mhsa_with_multi_head_position(inputs, embed_dim, num_heads, key_dim, attn_ratio, drop_rate=0, activation="hard_swish", name=""):
    nn = mhsa_with_multi_head_position(inputs, num_heads, key_dim, embed_dim, attn_ratio, activation=activation, name=name)
    if drop_rate > 0:
        nn = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop")(nn)
    return keras.layers.Add(name=name + "add")([inputs, nn])


def res_mlp_block(inputs, mlp_ratio, drop_rate=0, use_bias=False, activation="hard_swish", name=""):
    in_channels = inputs.shape[-1]
    nn = keras.layers.Dense(in_channels * mlp_ratio, use_bias=use_bias, name=name + "1_dense")(inputs)
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
    nn = keras.layers.Dense(in_channels, use_bias=use_bias, name=name + "2_dense")(nn)
    nn = batchnorm_with_activation(nn, activation=None, name=name + "2_")
    if drop_rate > 0:
        nn = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop")(nn)
    return keras.layers.Add(name=name + "add")([inputs, nn])


def attention_mlp_stack(inputs, out_channel, num_heads, depth, key_dim, attn_ratio, mlp_ratio, strides, stack_drop=0, activation="hard_swish", name=""):
    nn = inputs
    embed_dim = nn.shape[-1]
    stack_drop_s, stack_drop_e = stack_drop if isinstance(stack_drop, (list, tuple)) else [stack_drop, stack_drop]
    for ii in range(depth):
        block_name = name + "block{}_".format(ii + 1)
        drop_rate = stack_drop_s + (stack_drop_e - stack_drop_s) * ii / depth
        nn = res_mhsa_with_multi_head_position(nn, embed_dim, num_heads, key_dim, attn_ratio, drop_rate, activation=activation, name=block_name)
        if mlp_ratio > 0:
            nn = res_mlp_block(nn, mlp_ratio, drop_rate, activation=activation, name=block_name + "mlp_")
    if embed_dim != out_channel:
        block_name = name + "downsample_"
        ds_num_heads = embed_dim // key_dim
        ds_attn_ratio = attn_ratio * strides
        nn = mhsa_with_multi_head_position_and_strides(nn, ds_num_heads, key_dim, out_channel, ds_attn_ratio, strides, activation=activation, name=block_name)
        if mlp_ratio > 0:
            nn = res_mlp_block(nn, mlp_ratio, drop_rate, activation=activation, name=block_name + "mlp_")
    return keras.layers.Activation("linear", name=name + "output")(nn)  # Identity, Just need a name here


def patch_stem(inputs, stem_width, activation="hard_swish", name=""):
    nn = conv2d_no_bias(inputs, stem_width // 8, 3, strides=2, padding="same", name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
    nn = conv2d_no_bias(nn, stem_width // 4, 3, strides=2, padding="same", name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "2_")
    nn = conv2d_no_bias(nn, stem_width // 2, 3, strides=2, padding="same", name=name + "3_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "3_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=2, padding="same", name=name + "4_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "4_")
    return nn


def LeViT(
    patch_channel=128,
    out_channels=[256, 384, 384],  # C
    num_heads=[4, 6, 8],  # N
    depthes=[2, 3, 4],  # X
    key_dims=[16, 16, 16],  # D
    attn_ratios=[2, 2, 2],  # attn_ratio
    mlp_ratios=[2, 2, 2],  # mlp_ratio
    strides=[2, 2, 0],  # down_ops, strides
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="hard_swish",
    drop_connect_rate=0,
    dropout=0,
    classifier_activation=None,
    use_distillation=True,
    pretrained="imagenet",
    model_name="levit",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)
    nn = patch_stem(inputs, patch_channel, activation=activation, name="stem_")
    # nn = tf.reshape(nn, [-1, nn.shape[1] * nn.shape[2], patch_channel])

    global_block_id = 0
    total_blocks = sum(depthes)
    drop_connect_s, drop_connect_e = drop_connect_rate if isinstance(drop_connect_rate, (list, tuple)) else (drop_connect_rate, drop_connect_rate)
    for id, (out_channel, num_head, depth, key_dim, attn_ratio, mlp_ratio, stride) in enumerate(
        zip(out_channels, num_heads, depthes, key_dims, attn_ratios, mlp_ratios, strides)
    ):
        name = "stack{}_".format(id + 1)
        stack_drop_s = drop_connect_s + (drop_connect_e - drop_connect_s) * global_block_id / total_blocks
        stack_drop_e = drop_connect_s + (drop_connect_e - drop_connect_s) * (global_block_id + depth) / total_blocks
        stack_drop = (stack_drop_s, stack_drop_e)
        nn = attention_mlp_stack(nn, out_channel, num_head, depth, key_dim, attn_ratio, mlp_ratio, stride, stack_drop, activation, name=name)
        global_block_id += depth

    if num_classes == 0:
        out = nn
    else:
        nn = keras.layers.GlobalAveragePooling2D()(nn)  # tf.reduce_mean(nn, axis=1)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        out = batchnorm_with_activation(nn, activation=None, name="head_")
        out = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="head")(out)

        if use_distillation:
            distill = batchnorm_with_activation(nn, activation=None, name="distill_head_")
            distill = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="distill_head")(distill)
            out = [out, distill]

    model = keras.models.Model(inputs, out, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "levit", pretrained, MultiHeadPositionalEmbedding)
    return model


def LeViT128S(input_shape=(224, 224, 3), num_classes=1000, use_distillation=True, classifier_activation=None, pretrained="imagenet", **kwargs):
    return LeViT(**locals(), model_name="levit128s", **kwargs)


def LeViT128(input_shape=(224, 224, 3), num_classes=1000, use_distillation=True, classifier_activation=None, pretrained="imagenet", **kwargs):
    num_heads = [4, 8, 12]
    depthes = [4, 4, 4]
    return LeViT(**locals(), model_name="levit128", **kwargs)


def LeViT192(input_shape=(224, 224, 3), num_classes=1000, use_distillation=True, classifier_activation=None, pretrained="imagenet", **kwargs):
    patch_channel = 192
    out_channels = [288, 384, 384]
    num_heads = [3, 5, 6]
    depthes = [4, 4, 4]
    key_dims = [32, 32, 32]
    return LeViT(**locals(), model_name="levit192", **kwargs)


def LeViT256(input_shape=(224, 224, 3), num_classes=1000, use_distillation=True, classifier_activation=None, pretrained="imagenet", **kwargs):
    patch_channel = 256
    out_channels = [384, 512, 512]
    num_heads = [4, 6, 8]
    depthes = [4, 4, 4]
    key_dims = [32, 32, 32]
    return LeViT(**locals(), model_name="levit256", **kwargs)


def LeViT384(input_shape=(224, 224, 3), num_classes=1000, use_distillation=True, classifier_activation=None, pretrained="imagenet", **kwargs):
    patch_channel = 384
    out_channels = [512, 768, 768]
    num_heads = [6, 9, 12]
    depthes = [4, 4, 4]
    key_dims = [32, 32, 32]
    return LeViT(**locals(), model_name="levit384", **kwargs)
