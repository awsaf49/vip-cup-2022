import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.attention_layers import (
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    layer_norm,
    MultiHeadRelativePositionalEmbedding,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "cmt_tiny": {"imagenet": {160: "cb269248643e9d50d8bea051563e20a6", 224: "d0f2f0cf649a7aea48a1a4e3a476606c"}},
    "cmt_tiny_torch": {"imagenet": {160: "d800c892ef5581d73cbdef5ba61bc443"}},
    "cmt_xs_torch": {"imagenet": {192: "cb8250ce61d3bcd0d24ace6c4a803f8b"}},
    "cmt_small_torch": {"imagenet": {224: "2efa6fafb040dc617f6eb8f3cbfd051a"}},
    "cmt_base_torch": {"imagenet": {256: "7cb17018b0bc73d33892e1fb3c57f82b"}},
}


@tf.keras.utils.register_keras_serializable(package="kecam/cmt")
class BiasPositionalEmbedding(keras.layers.Layer):
    def __init__(self, axis=[1, 2, 3], attn_height=-1, initializer="zeros", **kwargs):
        super().__init__(**kwargs)
        self.axis, self.initializer, self.attn_height = axis, initializer, attn_height

    def build(self, input_shape):
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            bb_shape = (input_shape[-1],)
        else:
            bb_shape = [1] * len(input_shape)
            axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
            for ii in axis:
                bb_shape[ii] = input_shape[ii]
            bb_shape = bb_shape[1:]  # exclude batch dimension
        self.bb = self.add_weight(name="positional_embedding", shape=bb_shape, initializer=self.initializer, trainable=True)

        self.query_hh = int(tf.math.sqrt(float(input_shape[2]))) if self.attn_height == -1 else self.attn_height
        self.query_ww = int(float(input_shape[2]) / self.query_hh)
        sr_ratio = int(tf.math.sqrt(float(input_shape[2]) / float(input_shape[3])))
        self.kv_hh = self.query_hh // sr_ratio
        self.kv_ww = int(float(input_shape[3]) / self.kv_hh)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.bb

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "attn_height": self.attn_height})  # Not saving initializer in config
        return config

    def load_resized_pos_emb(self, source_layer, method="nearest"):
        if isinstance(source_layer, dict):
            source_tt = source_layer["positional_embedding:0"]  # weights
            # source_tt = source_layer["pos_emb:0"]  # weights
        else:
            source_tt = source_layer.bb  # layer
        num_heads = source_tt.shape[0]
        source_query_hh = source_query_ww = int(tf.math.sqrt(float(source_tt.shape[1])))  # assume source weights are all square shape
        source_kv_hh = source_kv_ww = int(tf.math.sqrt(float(source_tt.shape[2])))  # assume source weights are all square shape

        tt = tf.reshape(source_tt, [num_heads, source_query_hh, source_query_ww, source_kv_hh * source_kv_ww])  # resize on query dimension first
        tt = tf.image.resize(tt, [self.query_hh, self.query_ww], method=method)  # [num_heads, self.query_hh, self.query_ww, source_kv_hh * source_kv_ww]
        tt = tf.reshape(tt, [num_heads, self.query_hh * self.query_ww, source_kv_hh, source_kv_ww])  # resize on key_value dimension
        tt = tf.transpose(tt, [0, 2, 3, 1])  # [num_heads, source_kv_hh, source_kv_ww, self.query_hh * self.query_ww]
        tt = tf.image.resize(tt, [self.kv_hh, self.kv_ww], method=method)  # [num_heads, self.kv_hh, self.kv_ww, self.query_hh * self.query_ww]
        tt = tf.reshape(tt, [num_heads, self.kv_hh * self.kv_ww, self.query_hh * self.query_ww])
        tt = tf.transpose(tt, [0, 2, 1])  # [num_heads, self.query_hh * self.query_ww, self.kv_hh * self.kv_ww]
        self.bb.assign(tt)


def light_mhsa_with_multi_head_relative_position_embedding(
    inputs, num_heads=4, key_dim=0, sr_ratio=1, qkv_bias=False, pos_emb=None, use_bn=False, out_shape=None, out_weight=True, out_bias=False, dropout=0, name=""
):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = float(1.0 / tf.math.sqrt(tf.cast(key_dim, "float32")))
    out_shape = cc if out_shape is None or not out_weight else out_shape
    emb_dim = num_heads * key_dim

    query = keras.layers.Dense(emb_dim, use_bias=qkv_bias, name=name and name + "query")(inputs) * qk_scale
    # print(f">>>> {inputs.shape = }, {query.shape = }, {sr_ratio = }")
    # query = [batch, num_heads, hh * ww, key_dim]
    query = tf.transpose(tf.reshape(query, [-1, inputs.shape[1] * inputs.shape[2], num_heads, key_dim]), [0, 2, 1, 3])

    if sr_ratio > 1:
        key_value = depthwise_conv2d_no_bias(inputs, kernel_size=sr_ratio, strides=sr_ratio, use_bias=qkv_bias, name=name + "kv_sr_")
        key_value = batchnorm_with_activation(key_value, activation=None, name=name + "kv_sr_") if use_bn else layer_norm(key_value, name=name + "kv_sr_")
        # key_value = keras.layers.AvgPool2D(sr_ratio, strides=sr_ratio, name=name + "kv_sr_")(inputs)
    else:
        key_value = inputs
    _, kv_hh, kv_ww, _ = key_value.shape
    # key_value = [batch, num_heads, hh, ww, kv_kernel * kv_kernel, key_dim * 2]
    key_value = keras.layers.Dense(emb_dim * 2, use_bias=qkv_bias, name=name and name + "key_value")(key_value)
    # key = keras.layers.Dense(emb_dim, use_bias=qkv_bias, name=name and name + "key")(key_value)
    # value = keras.layers.Dense(emb_dim, use_bias=qkv_bias, name=name and name + "value")(key_value)
    # key_value = conv2d_no_bias(inputs, emb_dim * 2, kernel_size=sr_ratio, strides=sr_ratio, use_bias=False, name=name + "key_value")
    # print(f">>>> {key_value.shape = }")

    # dim, head, kv
    key_value = tf.reshape(key_value, [-1, kv_hh * kv_ww, key_dim, num_heads, 2])
    key = tf.transpose(key_value[:, :, :, :, 0], [0, 3, 2, 1])  # [batch, num_heads, key_dim, hh * ww]
    value = tf.transpose(key_value[:, :, :, :, 1], [0, 3, 1, 2])  # [batch, num_heads, hh * ww, key_dim]
    # kv, head, dim
    # key, value = tf.split(key_value, 2, axis=-1)
    # key = tf.transpose(tf.reshape(key, [-1, kv_hh * kv_ww, num_heads, key_dim]), [0, 2, 3, 1]) # [batch, num_heads, key_dim, hh * ww]
    # value = tf.transpose(tf.reshape(value, [-1, kv_hh * kv_ww, num_heads, key_dim]), [0, 2, 1, 3]) # [batch, num_heads, hh * ww, key_dim]

    # print(f">>>> {attn_query.shape = }, {key.shape = }, {value.shape = }, {kv_inp.shape = }, {pos_query.shape = }")
    # attention_scores = [batch, num_heads, hh * ww, kv_hh * kv_ww]
    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key])
    if pos_emb is None:
        attention_scores = MultiHeadRelativePositionalEmbedding(with_cls_token=False, attn_height=hh, name=name and name + "pos_emb")(attention_scores)
    else:
        attention_scores = pos_emb(attention_scores)
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)

    if dropout > 0:
        attention_scores = keras.layers.Dropout(dropout, name=name and name + "attn_drop")(attention_scores)
    # value = [batch, num_heads, key_hh * key_ww, key_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * key_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    return attention_output


def inverted_residual_feed_forward(inputs, expansion=4, activation="gelu", name=""):
    """IRFFN(X) = Conv(F(Conv(X))), F(X) = DWConv(X) + X"""
    in_channel = inputs.shape[-1]
    expanded = conv2d_no_bias(inputs, int(in_channel * expansion), kernel_size=1, use_bias=True, name=name + "1_")
    expanded = batchnorm_with_activation(expanded, activation=activation, act_first=True, name=name + "1_")

    dw = depthwise_conv2d_no_bias(expanded, kernel_size=3, padding="SAME", use_bias=True, name=name)
    dw = keras.layers.Add(name=name + "dw_add")([expanded, dw])
    dw = batchnorm_with_activation(dw, activation=activation, act_first=True, name=name + "2_")

    pw = conv2d_no_bias(dw, in_channel, kernel_size=1, use_bias=True, name=name + "3_")
    pw = batchnorm_with_activation(pw, activation=None, name=name + "3_")
    return pw


def cmt_block(
    inputs, num_heads=4, sr_ratio=1, expansion=4, qkv_bias=False, pos_emb=None, attn_use_bn=False, attn_out_bias=False, activation="gelu", drop_rate=0, name=""
):
    """X0 = LPU(Xi), X1 = LMHSA(LN(X0)) + X0, X2 = IRFFN(LN(X1)) + X1"""
    """ Local Perception Unit, LPU(X) = DWConv(X) + X """
    lpu = depthwise_conv2d_no_bias(inputs, kernel_size=3, padding="SAME", use_bias=True, name=name)
    # lpu = batchnorm_with_activation(lpu, activation=activation, name=name + "lpu_", act_first=True)
    lpu_out = keras.layers.Add(name=name + "lpu_out")([inputs, lpu])

    """ light multi head self attention """
    attn = layer_norm(lpu_out, name=name + "attn_")
    attn = light_mhsa_with_multi_head_relative_position_embedding(
        attn, num_heads=num_heads, sr_ratio=sr_ratio, qkv_bias=qkv_bias, pos_emb=pos_emb, use_bn=attn_use_bn, out_bias=attn_out_bias, name=name + "light_mhsa_"
    )
    attn = drop_block(attn, drop_rate=drop_rate)
    attn_out = keras.layers.Add(name=name + "attn_out")([lpu_out, attn])

    """ inverted residual feed forward """
    ffn = layer_norm(attn_out, name=name + "ffn_")
    ffn = inverted_residual_feed_forward(ffn, expansion=expansion, activation=activation, name=name + "ffn_")
    ffn = drop_block(ffn, drop_rate=drop_rate)
    ffn_out = keras.layers.Add(name=name + "ffn_output")([attn_out, ffn])

    return ffn_out


def cmt_stem(inputs, stem_width, activation="gelu", name="", **kwargs):
    nn = conv2d_no_bias(inputs, stem_width, kernel_size=3, strides=2, padding="same", use_bias=True, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=True, name=name + "1_")
    nn = conv2d_no_bias(nn, stem_width, kernel_size=3, strides=1, padding="same", use_bias=True, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=True, name=name + "2_")
    nn = conv2d_no_bias(nn, stem_width, kernel_size=3, strides=1, padding="same", use_bias=True, name=name + "3_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=True, name=name + "3_")
    return nn


def CMT(
    num_blocks,
    out_channels,
    stem_width=16,
    num_heads=[1, 2, 4, 8],
    sr_ratios=[8, 4, 2, 1],
    ffn_expansion=4,
    qkv_bias=False,  # for CMT_torch
    attn_out_bias=False,  # for CMT_torch
    attn_use_bn=False,  # for CMT_torch
    use_block_pos_emb=False,  # for CMT_torch
    feature_activation=None,  # for CMT_torch
    feature_act_first=True,  # for CMT_torch
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    output_num_features=1280,
    dropout=0,
    pretrained=None,
    model_name="cmt",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)
    nn = cmt_stem(inputs, stem_width=stem_width, activation=activation, name="stem_")

    """ stage [1, 2, 3, 4] """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, num_head, sr_ratio) in enumerate(zip(num_blocks, out_channels, num_heads, sr_ratios)):
        stage_name = "stack{}_".format(stack_id + 1)
        nn = conv2d_no_bias(nn, out_channel, kernel_size=2, strides=2, use_bias=True, name=stage_name + "down_sample")
        nn = layer_norm(nn, name=stage_name)

        if use_block_pos_emb:
            block_pos_emb = BiasPositionalEmbedding(axis=[1, 2, 3], attn_height=nn.shape[1], name=stage_name + "pos_emb")
            block_pos_emb.build([None, num_head, nn.shape[1] * nn.shape[2], (nn.shape[1] // sr_ratio) * (nn.shape[2] // sr_ratio)])
        else:
            block_pos_emb = None

        for block_id in range(num_block):
            name = stage_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            nn = cmt_block(
                nn, num_head, sr_ratio, ffn_expansion, qkv_bias, block_pos_emb, attn_use_bn, attn_out_bias, activation, drop_rate=block_drop_rate, name=name
            )

    if output_num_features > 0:
        nn = conv2d_no_bias(nn, output_num_features, 1, strides=1, use_bias=True, name="features_")
        feature_activation = activation if feature_activation is None else feature_activation
        nn = batchnorm_with_activation(nn, activation=feature_activation, act_first=feature_act_first, name="features_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if dropout > 0:
            nn = keras.layers.Dropout(dropout, name="head_drop")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    mismatch_class = BiasPositionalEmbedding if use_block_pos_emb else MultiHeadRelativePositionalEmbedding
    reload_model_weights(model, PRETRAINED_DICT, "cmt", pretrained, mismatch_class)
    return model


def CMT_torch(qkv_bias=True, attn_out_bias=True, attn_use_bn=True, use_block_pos_emb=True, feature_activation="swish", feature_act_first=False, **kwargs):
    kwargs.pop("kwargs", None)
    return CMT(**locals(), **kwargs)


def CMTTiny(input_shape=(160, 160, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 10, 2]
    out_channels = [46, 92, 184, 368]
    stem_width = 16
    ffn_expansion = 3.6
    return CMT(**locals(), model_name="cmt_tiny", **kwargs)


def CMTTiny_torch(input_shape=(160, 160, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 10, 2]
    out_channels = [46, 92, 184, 368]
    stem_width = 16
    ffn_expansion = 3.6
    return CMT_torch(**locals(), model_name="cmt_tiny_torch", **kwargs)


def CMTXS_torch(input_shape=(192, 192, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 12, 3]
    out_channels = [52, 104, 208, 416]
    stem_width = 16
    ffn_expansion = 3.77
    return CMT_torch(**locals(), model_name="cmt_xs_torch", **kwargs)


def CMTSmall_torch(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 16, 3]
    out_channels = [64, 128, 256, 512]
    stem_width = 32
    ffn_expansion = 4
    return CMT_torch(**locals(), model_name="cmt_small_torch", **kwargs)


def CMTBase_torch(input_shape=(256, 256, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [4, 4, 20, 4]
    out_channels = [76, 152, 304, 608]
    stem_width = 38
    ffn_expansion = 4
    return CMT_torch(**locals(), model_name="cmt_base_torch", **kwargs)
