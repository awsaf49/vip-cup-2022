import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    ChannelAffine,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    MixupToken,
    mlp_block,
    output_block,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
LAYER_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {
    "uniformer_base_32": {"token_label": {224: "992e2534c5741b2fb4f35a99b40b9c98"}},
    "uniformer_base_64": {"imagenet": {224: "e852a51824f01ef9a87792395d0d8820"}, "token_label": {224: "f72f7683bfe19854c79f9395f31bdb23"}},
    "uniformer_small_32": {"token_label": {224: "efb9e6531fcc2b560abdddc16b7e8297"}},
    "uniformer_small_64": {"imagenet": {224: "867648a1d96af15ef553337e27b53ede"}, "token_label": {224: "4d8d4f874b5bcf47594587800ff61fbd"}},
    "uniformer_small_plus_32": {"imagenet": {224: "7796cce29b5ea6572330547ba7eb5e0d"}, "token_label": {224: "b1d32f5e5714b66d76ef2fecce636dfb"}},
    "uniformer_small_plus_64": {"imagenet": {224: "7d10381f4527496adb2d39c4a665c808"}, "token_label": {224: "15d6af207a0f09957a5534ae1ad540ed"}},
    "uniformer_large_64": {"token_label": {224: "b1020b4e8029209a326e8fe7183d7d28", 384: "809ba104d43e905d5b24a8ec6ee02bdd"}},
}


def multi_head_self_attention(
    inputs, num_heads=4, key_dim=0, out_shape=None, out_weight=True, qkv_bias=False, out_bias=False, attn_dropout=0, output_dropout=0, name=None
):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = float(1.0 / tf.math.sqrt(tf.cast(key_dim, "float32")))
    out_shape = cc if out_shape is None or not out_weight else out_shape
    qk_out = num_heads * key_dim
    vv_dim = out_shape // num_heads

    qkv = keras.layers.Dense(qk_out * 2 + out_shape, use_bias=qkv_bias, name=name and name + "qkv")(inputs)
    qkv = tf.reshape(qkv, [-1, qkv.shape[1] * qkv.shape[2], qkv.shape[-1]])
    query, key, value = tf.split(qkv, [qk_out, qk_out, out_shape], axis=-1)
    query = tf.transpose(tf.reshape(query, [-1, query.shape[1], num_heads, key_dim]), [0, 2, 1, 3])  #  [batch, num_heads, hh * ww, key_dim]
    key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 3, 1])  # [batch, num_heads, key_dim, hh * ww]
    value = tf.transpose(tf.reshape(value, [-1, value.shape[1], num_heads, vv_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, vv_dim]

    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key]) * qk_scale  # [batch, num_heads, hh * ww, hh * ww]
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)
    attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores) if attn_dropout > 0 else attention_scores

    # value = [batch, num_heads, hh * ww, vv_dim], attention_output = [batch, num_heads, hh * ww, vv_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * vv_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, hh, ww, num_heads * vv_dim] * [num_heads * vv_dim, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    attention_output = keras.layers.Dropout(output_dropout, name=name and name + "out_drop")(attention_output) if output_dropout > 0 else attention_output
    return attention_output


def block(inputs, out_channel, num_heads=0, qkv_bias=True, mlp_ratio=4, mlp_drop_rate=0, attn_drop_rate=0, drop_rate=0, gamma=-1, activation="gelu", name=""):
    is_conv = False if num_heads > 0 else True  # decide by if num_heads > 0
    input_channel = inputs.shape[-1]  # Same with out_channel
    pos_emb = depthwise_conv2d_no_bias(inputs, kernel_size=3, padding="SAME", use_bias=True, name=name + "pos_emb_")
    pos_out = keras.layers.Add()([inputs, pos_emb])

    # print(f">>>> {is_conv = }, {num_heads = }")
    if is_conv:
        attn = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, name=name + "attn_bn")(pos_out)
        attn = conv2d_no_bias(attn, out_channel, 1, use_bias=True, name=name + "attn_1_")
        attn = depthwise_conv2d_no_bias(attn, kernel_size=5, padding="SAME", use_bias=True, name=name + "attn_")
        attn = conv2d_no_bias(attn, out_channel, 1, use_bias=True, name=name + "attn_2_")
    else:
        attn = keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name=name + "attn_ln")(pos_out)
        # attn = multi_head_self_attention(
        #     attn, num_heads, qkv_bias=qkv_bias, out_bias=True, attn_dropout=attn_drop_rate, output_dropout=mlp_drop_rate, name=name + "attn_mhsa_"
        # )
        attn = multi_head_self_attention(attn, num_heads, qkv_bias=qkv_bias, out_bias=True, attn_dropout=attn_drop_rate, name=name + "attn_mhsa_")
    attn = ChannelAffine(use_bias=False, weight_init_value=gamma, name=name + "1_gamma")(attn) if gamma >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name + "attn_")
    attn_out = keras.layers.Add(name=name + "attn_out")([pos_out, attn])

    if is_conv:
        mlp = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, name=name + "mlp_bn")(attn_out)
    else:
        mlp = keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name=name + "mlp_ln")(attn_out)
    mlp = mlp_block(mlp, int(out_channel * mlp_ratio), drop_rate=mlp_drop_rate, use_conv=is_conv, activation=activation, name=name + "mlp_")
    mlp = ChannelAffine(use_bias=False, weight_init_value=gamma, name=name + "2_gamma")(mlp) if gamma >= 0 else mlp
    mlp = drop_block(mlp, drop_rate=drop_rate, name=name + "mlp_")
    return keras.layers.Add(name=name + "output")([attn_out, mlp])


def stem(inputs, stem_width, use_conv_stem=False, drop_rate=0, activation="gelu", name="stem_"):
    if use_conv_stem:
        nn = conv2d_no_bias(inputs, stem_width // 2, kernel_size=3, strides=2, padding="same", use_bias=True, name=name + "1_")
        nn = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, name=name + "1_bn")(nn)
        nn = activation_by_name(nn, activation, name=name)
        nn = conv2d_no_bias(nn, stem_width, kernel_size=3, strides=2, padding="same", use_bias=True, name=name + "2_")
        nn = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, name=name + "2_bn")(nn)
    else:
        nn = conv2d_no_bias(inputs, stem_width, 4, strides=4, padding="valid", use_bias=True, name=name)
        nn = keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name=name + "ln")(nn)
    nn = keras.layers.Dropout(drop_rate) if drop_rate > 0 else nn
    return nn


def Uniformer(
    num_blocks=[3, 4, 8, 3],
    out_channels=[64, 128, 320, 512],
    head_dimension=64,
    use_conv_stem=False,
    block_types=["conv", "conv", "transform", "transform"],
    stem_width=-1,
    qkv_bias=True,
    mlp_ratio=4,
    layer_scale=-1,
    mix_token=False,
    token_label_top=False,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    mlp_drop_rate=0,
    attn_drop_rate=0,
    drop_connect_rate=0,
    dropout=0,
    classifier_activation="softmax",
    pretrained=None,
    model_name="uniformer",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)

    """ stem """
    stem_width = stem_width if stem_width > 0 else out_channels[0]
    nn = stem(inputs, stem_width, use_conv_stem, drop_rate=mlp_drop_rate, activation=activation, name="stem_")  # It's using mlp_drop_rate for stem

    if mix_token and token_label_top:
        scale = 8  # downsample 3 times
        mixup_token = MixupToken(scale=scale)
        bbox = mixup_token(nn)
        nn = mixup_token.do_mixup_token(nn, bbox * scale)

    """ stage [1, 2, 3, 4] """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, block_type) in enumerate(zip(num_blocks, out_channels, block_types)):
        stack_name = "stack{}_".format(stack_id + 1)
        is_conv_block = True if block_type[0].lower() == "c" else False
        num_heads = 0 if is_conv_block else out_channel // head_dimension
        if stack_id > 0:
            if use_conv_stem:
                nn = conv2d_no_bias(nn, out_channel, kernel_size=3, strides=2, padding="same", use_bias=True, name=stack_name + "downsample_")
                nn = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, name=stack_name + "downsample_bn")(nn)
            else:
                nn = conv2d_no_bias(nn, out_channel, kernel_size=2, strides=2, use_bias=True, name=stack_name + "downsample_")
                nn = keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name=stack_name + "downsample_ln")(nn)

        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = block(nn, out_channel, num_heads, qkv_bias, mlp_ratio, mlp_drop_rate, attn_drop_rate, block_drop_rate, layer_scale, activation, block_name)
            global_block_id += 1
    nn = keras.layers.BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, name="post_bn")(nn)

    """ output """
    if token_label_top and num_classes > 0:
        # Training with label token
        nn_cls = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=None)  # Don't use softmax here
        nn_aux = keras.layers.Dense(num_classes, name="aux_head")(nn)

        if mix_token:
            nn_aux = mixup_token.do_mixup_token(nn_aux, bbox)
            nn_aux = keras.layers.Reshape((-1, nn_aux.shape[-1]), dtype="float32", name="aux")(nn_aux)

            left, top, right, bottom = bbox
            lam = 1 - ((right - left) * (bottom - top) / (nn_aux.shape[1] * nn_aux.shape[2]))
            lam_repeat = tf.expand_dims(tf.repeat(lam, tf.shape(nn_cls)[0], axis=0), 1)
            nn_cls = keras.layers.Concatenate(axis=-1, dtype="float32", name="class")([nn_cls, lam_repeat])
        else:
            nn_aux = keras.layers.Reshape((-1, nn_aux.shape[-1]), dtype="float32", name="aux")(nn_aux)
        out = [nn_cls, nn_aux]
    else:
        out = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)

    model = keras.models.Model(inputs, out, name=model_name)
    post_process = token_label_imagenet_decode_predictions if token_label_top else None
    add_pre_post_process(model, rescale_mode="torch", post_process=post_process)
    pretrained = "token_label" if pretrained is not None and "token" in pretrained.lower() else pretrained
    reload_model_weights(model, PRETRAINED_DICT, "uniformer", pretrained)
    return model


def token_label_imagenet_decode_predictions(preds, top=5, classifier_activation="softmax", do_decode=True):
    preds = preds[0] + 0.5 * tf.reduce_max(preds[1], axis=1)
    preds = getattr(keras.activations, classifier_activation)(preds) if classifier_activation is not None else preds
    return tf.keras.applications.imagenet_utils.decode_predictions(preds.numpy(), top=top) if do_decode else preds


def UniformerSmall32(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", token_label_top=False, pretrained="token_label", **kwargs):
    num_blocks = [3, 4, 8, 3]
    head_dimension = 32
    return Uniformer(**locals(), model_name="uniformer_small_32", **kwargs)


def UniformerSmall64(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", token_label_top=False, pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 8, 3]
    head_dimension = 64
    return Uniformer(**locals(), model_name="uniformer_small_64", **kwargs)


def UniformerSmallPlus32(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", token_label_top=False, pretrained="imagenet", **kwargs):
    num_blocks = [3, 5, 9, 3]
    head_dimension = 32
    use_conv_stem = True
    return Uniformer(**locals(), model_name="uniformer_small_plus_32", **kwargs)


def UniformerSmallPlus64(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", token_label_top=False, pretrained="imagenet", **kwargs):
    num_blocks = [3, 5, 9, 3]
    head_dimension = 64
    use_conv_stem = True
    return Uniformer(**locals(), model_name="uniformer_small_plus_64", **kwargs)


def UniformerBase32(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", token_label_top=False, pretrained="token_label", **kwargs):
    num_blocks = [5, 8, 20, 7]
    head_dimension = 32
    return Uniformer(**locals(), model_name="uniformer_base_32", **kwargs)


def UniformerBase64(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", token_label_top=False, pretrained="imagenet", **kwargs):
    num_blocks = [5, 8, 20, 7]
    head_dimension = 64
    return Uniformer(**locals(), model_name="uniformer_base_64", **kwargs)


def UniformerLarge64(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", token_label_top=False, pretrained="token_label", **kwargs):
    num_blocks = [5, 10, 24, 7]
    out_channels = [128, 192, 448, 640]
    head_dimension = 64
    layer_scale = 1e-6
    return Uniformer(**locals(), model_name="uniformer_large_64", **kwargs)
