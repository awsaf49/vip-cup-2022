import tensorflow as tf

from .feature import ReduceSize


@tf.keras.utils.register_keras_serializable(package="gcvit")
class Stem(tf.keras.layers.Layer):
    def __init__(self, dim, first_strides=2, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.first_strides = first_strides

    def build(self, input_shape):
        self.pad = tf.keras.layers.ZeroPadding2D(1, name='pad')
        self.proj = tf.keras.layers.Conv2D(self.dim, kernel_size=3, strides=2, name='proj')  # first down-sample
        self.conv_down = ReduceSize(keep_dim=True, first_strides=self.first_strides, name='conv_down')
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.pad(inputs)
        x = self.proj(x)
        x = self.conv_down(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'dim': self.dim,
                       'first_strides': self.first_strides})
        return config