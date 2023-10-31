import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.attention_layers import (
    BiasLayer,
    # ChannelAffine,
    drop_block,
    layer_norm,
    mlp_block,
    output_block,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "swin_transformer_v2_base_window12": {"imagenet21k": {192: "b68add0f06aeede5d4dd10cd26855f36"}},
    "swin_transformer_v2_base_window16": {"imagenet22k": {256: "1caba96ed702d467e650465503033c85"}},
    "swin_transformer_v2_base_window16": {"imagenet": {256: "a7bfb9ae0733807baa702d05eb8feab8"}},
    "swin_transformer_v2_base_window24": {"imagenet22k": {384: "acd467e1d8555e15542c8254bdae1b72"}},
    "swin_transformer_v2_base_window8": {"imagenet": {256: "15454d9f6ba2ccca940f9c45b6935af6"}},
    "swin_transformer_v2_large_window12": {"imagenet21k": {192: "ace20b0d634eb92989ece52e300440d5"}},
    "swin_transformer_v2_large_window16": {"imagenet22k": {256: "151bb82a138f956613ce4b9885bfdd18"}},
    "swin_transformer_v2_large_window24": {"imagenet22k": {384: "04fa3b195e5201c2d1068d1e19c8a0c5"}},
    "swin_transformer_v2_small_window16": {"imagenet": {256: "3b2ca43d1927cca1b414b60e1044a84d"}},
    "swin_transformer_v2_small_window8": {"imagenet": {256: "0a8468bd9acdf2056fc401e9f5067f97"}},
    "swin_transformer_v2_tiny_window16": {"imagenet": {256: "37ce8c5f514c2249ef10d9c3acc37d29"}},
    "swin_transformer_v2_tiny_window8": {"imagenet": {256: "9317f155e37e4081a09d290ca99bf7cd"}},
}


@tf.keras.utils.register_keras_serializable(package="kecam")
class ExpLogitScale(keras.layers.Layer):
    def __init__(self, axis=-1, init_value=10.0, max_value=100.0, **kwargs):
        super().__init__(**kwargs)
        self.axis, self.init_value, self.max_value = axis, init_value, max_value

    def build(self, input_shape):
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            weight_shape = (input_shape[-1],)
        else:
            weight_shape = [1] * len(input_shape)
            axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
            for ii in axis:
                weight_shape[ii] = input_shape[ii]

        initializer = tf.initializers.constant(tf.math.log(self.init_value))
        self.scale = self.add_weight(name="weight", shape=weight_shape, initializer=initializer, trainable=True)
        self.__max_value__ = tf.math.log(self.max_value)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs * tf.math.exp(tf.minimum(self.scale, self.__max_value__))

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "max_value": self.max_value})
        return config


@tf.keras.utils.register_keras_serializable(package="kecam")
class PairWiseRelativePositionalEmbedding(keras.layers.Layer):
    def __init__(self, pos_scale=-1, **kwargs):
        # No weight, just need to wrapper a layer, or will not in model structure
        self.pos_scale = pos_scale
        super().__init__(**kwargs)

    def __build_pairwise_relative_position_index__(self, input_shape):
        # input_shape: [batch * window_patch, window_height, window_width, channel]
        height, width = input_shape[1], input_shape[2]  # [12, 15]
        hh, ww = tf.meshgrid(range(height), range(width))
        coords = tf.stack([hh, ww], axis=-1)  # [15, 12, 2]
        coords_flatten = tf.reshape(coords, [-1, 2])  # [180, 2]
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]  # [180, 180, 2]
        # relative_coords = tf.reshape(relative_coords, [-1, 2])  # [196 * 196, 2]

        relative_coords_hh = relative_coords[:, :, 0] + height - 1
        relative_coords_ww = (relative_coords[:, :, 1] + width - 1) * (2 * height - 1)
        relative_coords_hhww = tf.stack([relative_coords_hh, relative_coords_ww], axis=-1)
        self.relative_position_index = tf.reduce_sum(relative_coords_hhww, axis=-1)  # [180, 180]

    def __build_relative_coords_table__(self, input_shape):
        # input_shape: [batch * window_patch, window_height, window_width, channel]
        height, width = input_shape[1], input_shape[2]  # [12, 15]
        hh, ww = tf.meshgrid(range(-height + 1, height), range(-width + 1, width), indexing="ij")
        coords = tf.cast(tf.stack([hh, ww], axis=-1), self.dtype)
        if self.pos_scale == -1:
            pos_scale = [height, width]
        else:
            # If pretrined weights are from different input_shape or window_size, pos_scale is previous actually using window_size
            pos_scale = self.pos_scale if isinstance(self.pos_scale, (list, tuple)) else [self.pos_scale, self.pos_scale]
        coords = coords * 8 / [float(pos_scale[0] - 1), float(pos_scale[1] - 1)]  # [23, 29, 2], normalize to -8, 8
        # torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        relative_log_coords = tf.sign(coords) * tf.math.log(1.0 + tf.abs(coords)) / (tf.math.log(2.0) * 3.0)
        self.relative_log_coords = tf.reshape(relative_log_coords, [-1, 2])  # [23 * 29, 2]
        self.height, self.width = height, width  # For reload with shape mismatched

    def build(self, input_shape):
        # input_shape: [batch * window_patch, window_height, window_width, channel]
        self.__build_relative_coords_table__(input_shape)
        self.__build_pairwise_relative_position_index__(input_shape)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.relative_log_coords, self.relative_position_index

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"pos_scale": self.pos_scale})
        return base_config


@tf.keras.utils.register_keras_serializable(package="kecam")
class WindowAttentionMask(keras.layers.Layer):
    def __init__(self, height, width, window_height, window_width, shift_height=0, shift_width=0, **kwargs):
        # No weight, just need to wrapper a layer, or will meet some error in model saving or loading...
        # float_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        self.height, self.width, self.window_height, self.window_width = height, width, window_height, window_width
        self.shift_height, self.shift_width = shift_height, shift_width
        self.blocks = (self.height // self.window_height) * (self.width // self.window_width)
        super().__init__(**kwargs)

    def build(self, input_shape):
        hh_split = [0, self.height - self.window_height, self.height - self.shift_height, self.height]
        ww_split = [0, self.width - self.window_width, self.width - self.shift_width, self.width]
        mask_value, total_ww, mask = 0, len(ww_split) - 1, []
        for hh_id in range(len(hh_split) - 1):
            hh = hh_split[hh_id + 1] - hh_split[hh_id]
            rr = [tf.zeros([hh, ww_split[id + 1] - ww_split[id]]) + (id + mask_value) for id in range(total_ww)]
            mask.append(tf.concat(rr, axis=-1))
            mask_value += total_ww
        mask = tf.concat(mask, axis=0)
        # return mask

        mask = tf.reshape(mask, [self.height // self.window_height, self.window_height, self.width // self.window_width, self.window_width])
        mask = tf.transpose(mask, [0, 2, 1, 3])
        mask = tf.reshape(mask, [-1, self.window_height * self.window_width])
        attn_mask = tf.expand_dims(mask, 1) - tf.expand_dims(mask, 2)
        attn_mask = tf.cast(tf.where(attn_mask != 0, -100, 0), self._compute_dtype)
        self.attn_mask = tf.expand_dims(tf.expand_dims(attn_mask, 1), 0)  # expand dims on batch and num_heads

        self.num_heads, self.query_blocks = input_shape[1], input_shape[2]

    def call(self, inputs, **kwargs):
        # inputs: [batch_size * blocks, num_heads, query_blocks, query_blocks]
        # where query_blocks = `window_height * window_width`, blocks = `(height // window_height) * (width // window_width)`
        nn = tf.reshape(inputs, [-1, self.blocks, self.num_heads, self.query_blocks, self.query_blocks])
        nn = nn + self.attn_mask
        return tf.reshape(nn, [-1, self.num_heads, self.query_blocks, self.query_blocks])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "window_height": self.window_height,
                "window_width": self.window_width,
                "shift_height": self.shift_height,
                "shift_width": self.shift_width,
            }
        )
        return config


def window_mhsa_with_pair_wise_positional_embedding(
    inputs, num_heads=4, key_dim=0, meta_hidden_dim=512, mask=None, pos_scale=-1, out_bias=True, qv_bias=True, attn_dropout=0, out_dropout=0, name=None
):
    input_channel = inputs.shape[-1]
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    qk_out = key_dim * num_heads

    qkv = keras.layers.Dense(qk_out * 3, use_bias=False, name=name and name + "qkv")(inputs)
    qkv = tf.reshape(qkv, [-1, qkv.shape[1] * qkv.shape[2], qkv.shape[-1]])
    query, key, value = tf.split(qkv, 3, axis=-1)
    if qv_bias:
        query = BiasLayer(name=name and name + "query_bias")(query)
        value = BiasLayer(name=name and name + "value_bias")(value)
    query = tf.transpose(tf.reshape(query, [-1, query.shape[1], num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, key_dim]
    key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 3, 1])  # [batch, num_heads, key_dim, hh * ww]
    value = tf.transpose(tf.reshape(value, [-1, value.shape[1], num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, vv_dim]

    # cosine attention
    norm_query, norm_key = tf.nn.l2_normalize(query, axis=-1, epsilon=1e-6), tf.nn.l2_normalize(key, axis=-2, epsilon=1e-6)
    attn = tf.matmul(norm_query, norm_key)  # [batch, num_heads, hh * ww, hh * ww]
    attn = ExpLogitScale(axis=1, name=name and name + "scale")(attn)  # axis=1 is head dimension

    # PairWiseRelativePositionalEmbedding -> mlp -> add with attn
    pos_coords, pos_index = PairWiseRelativePositionalEmbedding(pos_scale=pos_scale, name=name and name + "pos_emb")(inputs)
    relative_position_bias = keras.layers.Dense(meta_hidden_dim, use_bias=True, name=name and name + "meta_dense_1")(pos_coords)
    relative_position_bias = keras.layers.Activation("relu")(relative_position_bias)
    relative_position_bias = keras.layers.Dense(num_heads, use_bias=False, name=name and name + "meta_dense_2")(relative_position_bias)

    relative_position_bias = tf.gather(relative_position_bias, pos_index)  # [hh * ww, hh * ww, num_heads]
    relative_position_bias = tf.nn.sigmoid(relative_position_bias) * 16.0
    relative_position_bias = tf.expand_dims(tf.transpose(relative_position_bias, [2, 0, 1]), 0)  # [1, num_heads, hh * ww, hh * ww]
    attn = attn + relative_position_bias

    if mask is not None:
        attn = mask(attn)
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attn)

    if attn_dropout > 0:
        attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)
    attention_output = tf.matmul(attention_scores, value)
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * key_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    # [batch, hh, ww, num_heads * vv_dim] * [num_heads * vv_dim, out] --> [batch, hh, ww, out]
    attention_output = keras.layers.Dense(qk_out, use_bias=out_bias, name=name and name + "output")(attention_output)
    attention_output = keras.layers.Dropout(out_dropout, name=name and name + "out_drop")(attention_output) if out_dropout > 0 else attention_output
    return attention_output


def shifted_window_attention(inputs, window_size, num_heads=4, shift_size=0, pos_scale=-1, name=None):
    input_channel = inputs.shape[-1]
    window_size = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
    window_height = window_size[0] if window_size[0] < inputs.shape[1] else inputs.shape[1]
    window_width = window_size[1] if window_size[1] < inputs.shape[2] else inputs.shape[2]
    shift_size = 0 if (window_height == inputs.shape[1] and window_width == inputs.shape[2]) else shift_size
    should_shift = shift_size > 0

    # window_partition, partition windows, ceil mode padding if not divisible by window_size
    # patch_height, patch_width = inputs.shape[1] // window_height, inputs.shape[2] // window_width
    patch_height, patch_width = int(tf.math.ceil(inputs.shape[1] / window_height)), int(tf.math.ceil(inputs.shape[2] / window_width))
    should_pad_hh, should_pad_ww = patch_height * window_height - inputs.shape[1], patch_width * window_width - inputs.shape[2]
    # print(f">>>> shifted_window_attention {inputs.shape = }, {should_pad_hh = }, {should_pad_ww = }")
    if should_pad_hh or should_pad_ww:
        inputs = tf.pad(inputs, [[0, 0], [0, should_pad_hh], [0, should_pad_ww], [0, 0]])

    if should_shift:
        shift_height, shift_width = int(window_height * shift_size), int(window_width * shift_size)
        # tf.roll is not supported by tflite
        # inputs = tf.roll(inputs, shift=(shift_height * -1, shift_width * -1), axis=[1, 2])
        inputs = tf.concat([inputs[:, shift_height:], inputs[:, :shift_height]], axis=1)
        inputs = tf.concat([inputs[:, :, shift_width:], inputs[:, :, :shift_width]], axis=2)

    # print(f">>>> shifted_window_attention {inputs.shape = }, {patch_height = }, {patch_width = }, {window_height = }, {window_width = }")
    # [batch * patch_height, window_height, patch_width, window_width * input_channel], limit transpose perm <= 4
    nn = tf.reshape(inputs, [-1, window_height, patch_width, window_width * input_channel])
    nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch * patch_height, patch_width, window_height, window_width * input_channel]
    nn = tf.reshape(nn, [-1, window_height, window_width, input_channel])  # [batch * patch_height * patch_width, window_height, window_width, input_channel]

    mask = WindowAttentionMask(inputs.shape[1], inputs.shape[2], window_height, window_width, shift_height, shift_width) if should_shift else None
    nn = window_mhsa_with_pair_wise_positional_embedding(nn, num_heads=num_heads, mask=mask, pos_scale=pos_scale, name=name)

    # window_reverse, merge windows
    # [batch * patch_height, patch_width, window_height, window_width * input_channel], limit transpose perm <= 4
    nn = tf.reshape(nn, [-1, patch_width, window_height, window_width * input_channel])
    nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch * patch_height, window_height, patch_width, window_width * input_channel]
    nn = tf.reshape(nn, [-1, patch_height * window_height, patch_width * window_width, input_channel])

    if should_shift:
        # nn = tf.roll(nn, shift=(shift_height, shift_width), axis=[1, 2])
        nn = tf.concat([nn[:, -shift_height:], nn[:, :-shift_height]], axis=1)
        nn = tf.concat([nn[:, :, -shift_width:], nn[:, :, :-shift_width]], axis=2)

    # print(f">>>> shifted_window_attention before: {nn.shape = }, {should_pad_hh = }, {should_pad_ww = }")
    if should_pad_hh or should_pad_ww:
        nn = nn[:, : nn.shape[1] - should_pad_hh, : nn.shape[2] - should_pad_ww, :]  # In case should_pad_hh or should_pad_ww is 0
    # print(f">>>> shifted_window_attention after: {nn.shape = }")

    return nn


def swin_transformer_block(
    inputs, window_size, num_heads=4, shift_size=0, pos_scale=-1, mlp_ratio=4, mlp_drop_rate=0, attn_drop_rate=0, drop_rate=0, name=None
):
    input_channel = inputs.shape[-1]
    attn = shifted_window_attention(inputs, window_size, num_heads, shift_size, pos_scale=pos_scale, name=name + "attn_")
    attn = layer_norm(attn, zero_gamma=True, name=name + "attn_")
    # attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "1_gamma")(attn) if layer_scale >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name + "attn_")
    attn_out = keras.layers.Add(name=name + "attn_out")([inputs, attn])

    mlp = mlp_block(attn_out, int(input_channel * mlp_ratio), drop_rate=mlp_drop_rate, use_conv=False, activation="gelu", name=name + "mlp_")
    mlp = layer_norm(mlp, zero_gamma=True, name=name + "mlp_")
    # mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "2_gamma")(mlp) if layer_scale >= 0 else mlp
    mlp = drop_block(mlp, drop_rate=drop_rate, name=name + "mlp_")
    return keras.layers.Add(name=name + "output")([attn_out, mlp])


def patch_merging(inputs, name=""):
    input_channel = inputs.shape[-1]
    should_pad_hh, should_pad_ww = inputs.shape[1] % 2, inputs.shape[2] % 2
    # print(f">>>> patch_merging {inputs.shape = }, {should_pad_hh = }, {should_pad_ww = }")
    if should_pad_hh or should_pad_ww:
        inputs = tf.pad(inputs, [[0, 0], [0, should_pad_hh], [0, should_pad_ww], [0, 0]])

    # limit transpose perm <= 4
    nn = tf.reshape(inputs, [-1, 2, inputs.shape[2], input_channel])  # [batch * inputs.shape[1] // 2, height 2, inputs.shape[2], input_channel]
    nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch * inputs.shape[1] // 2, inputs.shape[2], height 2, input_channel]
    nn = tf.reshape(nn, [-1, inputs.shape[1] // 2, inputs.shape[2] // 2, 2 * 2 * input_channel])
    nn = keras.layers.Dense(2 * input_channel, use_bias=False, name=name + "dense")(nn)
    nn = layer_norm(nn, name=name)
    return nn


def SwinTransformerV2(
    num_blocks=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    embed_dim=96,
    window_size=7,
    pos_scale=-1,  # If pretrained weights are from different input_shape or window_size, pos_scale is previous actually using window_size
    stem_patch_size=4,
    use_stack_norm=False,  # True for extra layer_norm on each stack end
    extra_norm_period=0,  # > 0 for extra layer_norm frequency in each stack. May combine with use_stack_norm=True
    input_shape=(224, 224, 3),
    num_classes=1000,
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="swin_transformer_v2",
    kwargs=None,
):
    """Patch stem"""
    inputs = keras.layers.Input(input_shape)
    nn = keras.layers.Conv2D(embed_dim, kernel_size=stem_patch_size, strides=stem_patch_size, use_bias=True, name="stem_conv")(inputs)
    nn = layer_norm(nn, name="stem_")
    window_size = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, num_head) in enumerate(zip(num_blocks, num_heads)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            # height, width downsample * 0.5, channel upsample * 2
            nn = patch_merging(nn, name=stack_name + "downsample_")
        cur_pos_scale = pos_scale[stack_id] if isinstance(pos_scale, (list, tuple)) else pos_scale
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            shift_size = 0 if block_id % 2 == 0 else 0.5
            nn = swin_transformer_block(nn, window_size, num_head, shift_size, cur_pos_scale, drop_rate=block_drop_rate, name=block_name)
            global_block_id += 1
            if extra_norm_period > 0 and (block_id + 1) % extra_norm_period == 0 and not (use_stack_norm and block_id == num_block - 1):
                nn = layer_norm(nn, name=block_name + "output_")
        if use_stack_norm and stack_id != len(num_blocks) - 1:  # Exclude last stack
            nn = layer_norm(nn, name=stack_name + "output_")
    nn = layer_norm(nn, name="pre_output_")

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "swin_transformer_v2", pretrained)
    return model


def SwinTransformerV2Tiny_window8(input_shape=(256, 256, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    window_size = kwargs.pop("window_size", 8)
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_tiny_window8", **kwargs)


def SwinTransformerV2Tiny_window16(input_shape=(256, 256, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    window_size = kwargs.pop("window_size", 16)
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_tiny_window16", **kwargs)


def SwinTransformerV2Small_window8(input_shape=(256, 256, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 18, 2]
    window_size = kwargs.pop("window_size", 8)
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_small_window8", **kwargs)


def SwinTransformerV2Small_window16(input_shape=(256, 256, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 18, 2]
    window_size = kwargs.pop("window_size", 16)
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_small_window16", **kwargs)


def SwinTransformerV2Base_window8(input_shape=(256, 256, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    embed_dim = 128
    window_size = kwargs.pop("window_size", 8)
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_base_window8", **kwargs)


def SwinTransformerV2Base_window12(input_shape=(192, 192, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet21k", **kwargs):
    num_blocks = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    embed_dim = 128
    window_size = kwargs.pop("window_size", 12)
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_base_window12", **kwargs)


def SwinTransformerV2Base_window16(input_shape=(256, 256, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    embed_dim = 128
    window_size = kwargs.pop("window_size", 16)
    pos_scale = kwargs.pop("pos_scale", [12, 12, 12, 6] if pretrained == "imagenet22k" else -1)  # 22k model is fine-tuned from window12
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_base_window16", **kwargs)


def SwinTransformerV2Base_window24(input_shape=(384, 384, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet22k", **kwargs):
    num_blocks = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    embed_dim = 128
    window_size = kwargs.pop("window_size", 24)
    pos_scale = kwargs.pop("pos_scale", [12, 12, 12, 6] if pretrained == "imagenet22k" else -1)  # 22k model is fine-tuned from window12
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_base_window24", **kwargs)


def SwinTransformerV2Large_window12(input_shape=(192, 192, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet21k", **kwargs):
    num_blocks = [2, 2, 18, 2]
    num_heads = [6, 12, 24, 48]
    embed_dim = 192
    window_size = kwargs.pop("window_size", 12)
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_large_window12", **kwargs)


def SwinTransformerV2Large_window16(input_shape=(256, 256, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet22k", **kwargs):
    num_blocks = [2, 2, 18, 2]
    num_heads = [6, 12, 24, 48]
    embed_dim = 192
    window_size = kwargs.pop("window_size", 16)
    pos_scale = kwargs.pop("pos_scale", [12, 12, 12, 6] if pretrained == "imagenet22k" else -1)  # 22k model is fine-tuned from window12
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_large_window16", **kwargs)


def SwinTransformerV2Large_window24(input_shape=(384, 384, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet22k", **kwargs):
    num_blocks = [2, 2, 18, 2]
    num_heads = [6, 12, 24, 48]
    embed_dim = 192
    window_size = kwargs.pop("window_size", 24)
    pos_scale = kwargs.pop("pos_scale", [12, 12, 12, 6] if pretrained == "imagenet22k" else -1)  # 22k model is fine-tuned from window12
    return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_large_window24", **kwargs)


# def SwinTransformerV2Giant(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained=None, **kwargs):
#     num_blocks = [2, 2, 42, 2]
#     num_heads = [16, 32, 64, 128]
#     embed_dim = 512
#     extra_norm_period = 6
#     return SwinTransformerV2(**locals(), model_name="swin_transformer_v2_giant", **kwargs)
