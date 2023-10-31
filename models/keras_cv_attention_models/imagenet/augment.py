"""
Copied from: https://github.com/tensorflow/models/blob/master/official/vision/image_classification/augment.py
Midified according to: https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py
"""
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AutoAugment and RandAugment policies for enhanced image preprocessing.

AutoAugment Reference: https://arxiv.org/abs/1805.09501
RandAugment Reference: https://arxiv.org/abs/1909.13719
"""

import math
import tensorflow as tf
from typing import Any, Dict, List, Optional, Text, Tuple, Union

# from tensorflow.python.keras.layers.preprocessing import image_preprocessing as image_ops
from keras.layers.preprocessing import image_preprocessing as image_ops

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.0


def to_4d(image: tf.Tensor) -> tf.Tensor:
    """Converts an input Tensor to 4 dimensions.

    4D image => [N, H, W, C] or [N, C, H, W]
    3D image => [1, H, W, C] or [1, C, H, W]
    2D image => [1, H, W, 1]

    Args:
      image: The 2/3/4D input tensor.

    Returns:
      A 4D image tensor.

    Raises:
      `TypeError` if `image` is not a 2/3/4D tensor.

    """
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = tf.concat(
        [
            tf.ones(shape=left_pad, dtype=tf.int32),
            shape,
            tf.ones(shape=right_pad, dtype=tf.int32),
        ],
        axis=0,
    )
    return tf.reshape(image, new_shape)


def from_4d(image: tf.Tensor, ndims: tf.Tensor) -> tf.Tensor:
    """Converts a 4D image back to `ndims` rank."""
    shape = tf.shape(image)
    begin = tf.cast(tf.less_equal(ndims, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(ndims, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)


def _convert_translation_to_transform(translations: tf.Tensor) -> tf.Tensor:
    """Converts translations to a projective transform.

    The translation matrix looks like this:
      [[1 0 -dx]
       [0 1 -dy]
       [0 0 1]]

    Args:
      translations: The 2-element list representing [dx, dy], or a matrix of
        2-element lists representing [dx dy] to translate for each image. The
        shape must be static.

    Returns:
      The transformation matrix of shape (num_images, 8).

    Raises:
      `TypeError` if
        - the shape of `translations` is not known or
        - the shape of `translations` is not rank 1 or 2.

    """
    translations = tf.convert_to_tensor(translations, dtype=tf.float32)
    if translations.get_shape().ndims is None:
        raise TypeError("translations rank must be statically known")
    elif len(translations.get_shape()) == 1:
        translations = translations[None]
    elif len(translations.get_shape()) != 2:
        raise TypeError("translations should have rank 1 or 2.")
    num_translations = tf.shape(translations)[0]

    return tf.concat(
        values=[
            tf.ones((num_translations, 1), tf.dtypes.float32),
            tf.zeros((num_translations, 1), tf.dtypes.float32),
            -translations[:, 0, None],
            tf.zeros((num_translations, 1), tf.dtypes.float32),
            tf.ones((num_translations, 1), tf.dtypes.float32),
            -translations[:, 1, None],
            tf.zeros((num_translations, 2), tf.dtypes.float32),
        ],
        axis=1,
    )


def _convert_angles_to_transform(angles: tf.Tensor, image_width: tf.Tensor, image_height: tf.Tensor) -> tf.Tensor:
    """Converts an angle or angles to a projective transform.

    Args:
      angles: A scalar to rotate all images, or a vector to rotate a batch of
        images. This must be a scalar.
      image_width: The width of the image(s) to be transformed.
      image_height: The height of the image(s) to be transformed.

    Returns:
      A tensor of shape (num_images, 8).

    Raises:
      `TypeError` if `angles` is not rank 0 or 1.

    """
    angles = tf.convert_to_tensor(angles, dtype=tf.float32)
    if len(angles.get_shape()) == 0:  # pylint:disable=g-explicit-length-test
        angles = angles[None]
    elif len(angles.get_shape()) != 1:
        raise TypeError("Angles should have a rank 0 or 1.")
    x_offset = ((image_width - 1) - (tf.math.cos(angles) * (image_width - 1) - tf.math.sin(angles) * (image_height - 1))) / 2.0
    y_offset = ((image_height - 1) - (tf.math.sin(angles) * (image_width - 1) + tf.math.cos(angles) * (image_height - 1))) / 2.0
    num_angles = tf.shape(angles)[0]
    return tf.concat(
        values=[
            tf.math.cos(angles)[:, None],
            -tf.math.sin(angles)[:, None],
            x_offset[:, None],
            tf.math.sin(angles)[:, None],
            tf.math.cos(angles)[:, None],
            y_offset[:, None],
            tf.zeros((num_angles, 2), tf.dtypes.float32),
        ],
        axis=1,
    )


def marge_two_transforms(aa: tf.Tensor, bb: tf.Tensor) -> tf.Tensor:
    cc = tf.reshape(tf.concat([aa, [[1.0]]], axis=1), [3, 3])
    dd = tf.reshape(tf.concat([bb, [[1.0]]], axis=1), [3, 3])
    return tf.reshape(tf.matmul(cc, dd), [1, -1])[:, :8]


def transform(image: tf.Tensor, transforms) -> tf.Tensor:
    """Prepares input data for `image_ops.transform`."""
    original_ndims = tf.rank(image)
    transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
    if transforms.shape.rank == 1:
        transforms = transforms[None]
    image = to_4d(image)
    # image = image_ops.transform(images=image, transforms=transforms, interpolation="nearest", fill_mode="constant")
    image = image_ops.transform(images=image, transforms=transforms, interpolation="bilinear", fill_mode="constant")
    return from_4d(image, original_ndims)


def blend(image1: tf.Tensor, image2: tf.Tensor, factor: float) -> tf.Tensor:
    """Blend image1 and image2 using 'factor'.

    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.

    Args:
      image1: An image Tensor of type uint8.
      image2: An image Tensor of type uint8.
      factor: A floating point value above 0.0.

    Returns:
      A blended image Tensor of type uint8.
    """
    if factor == 0.0:
        return tf.convert_to_tensor(image1)
    if factor == 1.0:
        return tf.convert_to_tensor(image2)

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    difference = image2 - image1
    scaled = factor * difference

    # Do addition in float.
    temp = tf.cast(image1, tf.float32) + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        return tf.cast(temp, tf.uint8)

    # Extrapolate:
    #
    # We need to clip and then cast.
    return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


def cutout(image: tf.Tensor, pad_size: int, replace: int = 0) -> tf.Tensor:
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.

    Args:
      image: An image Tensor of type uint8.
      pad_size: Specifies how big the zero mask that will be generated is that is
        applied to the image. The mask will be of size (2*pad_size x 2*pad_size).
      replace: What pixel value to fill in the image in the area that has the
        cutout mask applied to it.

    Returns:
      An image Tensor that is of type uint8.
    """
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = tf.random.uniform(shape=[], minval=0, maxval=image_height, dtype=tf.int32)

    cutout_center_width = tf.random.uniform(shape=[], minval=0, maxval=image_width, dtype=tf.int32)

    lower_pad = tf.maximum(0, cutout_center_height - pad_size)
    upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = tf.maximum(0, cutout_center_width - pad_size)
    right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [image_height - (lower_pad + upper_pad), image_width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])
    image = tf.where(tf.equal(mask, 0), tf.ones_like(image, dtype=image.dtype) * replace, image)
    return image


def solarize(image: tf.Tensor, threshold: int = 128) -> tf.Tensor:
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    threshold = tf.cast(threshold, image.dtype)
    return tf.where(image < threshold, image, 255 - image)


def solarize_add(image: tf.Tensor, addition: int = 0, threshold: int = 128) -> tf.Tensor:
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    added_image = tf.cast(image, tf.int32) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
    threshold = tf.cast(threshold, image.dtype)
    return tf.where(image < threshold, added_image, image)


def color(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Equivalent of PIL Color."""
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
    return blend(degenerate, image, factor)


def contrast(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Equivalent of PIL Contrast."""
    degenerate = tf.image.rgb_to_grayscale(image)
    # Cast before calling tf.histogram.
    degenerate = tf.cast(degenerate, tf.int32)

    # Compute the grayscale histogram, then compute the mean pixel value,
    # and create a constant image size of that value.  Use that as the
    # blending degenerate target of the original image.
    hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
    mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
    degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
    return blend(degenerate, image, factor)


def brightness(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Equivalent of PIL Brightness."""
    degenerate = tf.zeros_like(image)
    return blend(degenerate, image, factor)


def posterize(image: tf.Tensor, bits: int) -> tf.Tensor:
    """Equivalent of PIL Posterize."""
    shift = tf.cast(8 - bits, image.dtype)
    return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def rotate(image: tf.Tensor, degrees: float, replace: int, return_affine_matrix: bool = False) -> tf.Tensor:
    """Rotates the image by degrees either clockwise or counterclockwise.

    Args:
      image: An image Tensor of type uint8.
      degrees: Float, a scalar angle in degrees to rotate all images by. If
        degrees is positive the image will be rotated clockwise otherwise it will
        be rotated counterclockwise.

    Returns:
      The rotated version of image.

    """
    # Convert from degrees to radians.
    degrees_to_radians = math.pi / 180.0
    radians = tf.cast(degrees * degrees_to_radians, tf.float32)

    image_height = tf.cast(tf.shape(image)[0], tf.float32)
    image_width = tf.cast(tf.shape(image)[1], tf.float32)
    transforms = _convert_angles_to_transform(angles=radians, image_width=image_width, image_height=image_height)
    if return_affine_matrix:
        return transforms

    image = transform(image=wrap(image), transforms=transforms)
    image = unwrap(image, replace)
    return image


def translate_x_relative(image: tf.Tensor, level: float, replace: int, return_affine_matrix: bool = False) -> tf.Tensor:
    """Equivalent of PIL Translate in X dimension."""
    pixels = level * image.shape[0]
    transforms = _convert_translation_to_transform([-pixels, 0])
    if return_affine_matrix:
        return transforms

    image = transform(image=wrap(image), transforms=transforms)
    image = unwrap(image, replace)
    return image


def translate_y_relative(image: tf.Tensor, level: float, replace: int, return_affine_matrix: bool = False) -> tf.Tensor:
    """Equivalent of PIL Translate in Y dimension."""
    pixels = level * image.shape[1]
    transforms = _convert_translation_to_transform([0, -pixels])
    if return_affine_matrix:
        return transforms

    image = transform(image=wrap(image), transforms=transforms)
    image = unwrap(image, replace)
    return image


def translate_x(image: tf.Tensor, pixels: int, replace: int, return_affine_matrix: bool = False) -> tf.Tensor:
    """Equivalent of PIL Translate in X dimension."""
    transforms = _convert_translation_to_transform([-pixels, 0])
    if return_affine_matrix:
        return transforms

    image = transform(image=wrap(image), transforms=transforms)
    image = unwrap(image, replace)
    return image


def translate_y(image: tf.Tensor, pixels: int, replace: int, return_affine_matrix: bool = False) -> tf.Tensor:
    """Equivalent of PIL Translate in Y dimension."""
    transforms = _convert_translation_to_transform([0, -pixels])
    if return_affine_matrix:
        return transforms

    image = transform(image=wrap(image), transforms=transforms)
    image = unwrap(image, replace)
    return image


def shear_x(image: tf.Tensor, level: float, replace: int, return_affine_matrix: bool = False) -> tf.Tensor:
    """Equivalent of PIL Shearing in X dimension."""
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1  level
    #  0  1].
    transforms = tf.convert_to_tensor([[1.0, level, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    if return_affine_matrix:
        return transforms

    image = transform(image=wrap(image), transforms=transforms)
    image = unwrap(image, replace)
    return image


def shear_y(image: tf.Tensor, level: float, replace: int, return_affine_matrix: bool = False) -> tf.Tensor:
    """Equivalent of PIL Shearing in Y dimension."""
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1  0
    #  level  1].
    transforms = tf.convert_to_tensor([[1.0, 0.0, 0.0, level, 1.0, 0.0, 0.0, 0.0]])
    if return_affine_matrix:
        return transforms

    image = transform(image=wrap(image), transforms=transforms)
    image = unwrap(image, replace)
    return image


def scale_x(image: tf.Tensor, level: float, replace: int, return_affine_matrix: bool = False) -> tf.Tensor:
    """Equivalent of PIL scaling in X dimension."""
    transforms = tf.convert_to_tensor([[level, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    if return_affine_matrix:
        return transforms

    image = transform(image=wrap(image), transforms=transforms)
    image = unwrap(image, replace)
    return image


def scale_y(image: tf.Tensor, level: float, replace: int, return_affine_matrix: bool = False) -> tf.Tensor:
    """Equivalent of PIL scaling in Y dimension."""
    transforms = tf.convert_to_tensor([[1.0, 0.0, 0.0, 0.0, level, 0.0, 0.0, 0.0]])
    if return_affine_matrix:
        return transforms

    image = transform(image=wrap(image), transforms=transforms)
    image = unwrap(image, replace)
    return image


def scale(image: tf.Tensor, level: float, replace: int, return_affine_matrix: bool = False) -> tf.Tensor:
    """Equivalent of PIL scaling in X and Y dimension."""
    transforms = tf.convert_to_tensor([[level, 0.0, 0.0, 0.0, level, 0.0, 0.0, 0.0]])
    if return_affine_matrix:
        return transforms

    image = transform(image=wrap(image), transforms=transforms)
    image = unwrap(image, replace)
    return image


def autocontrast(image: tf.Tensor) -> tf.Tensor:
    """Implements Autocontrast function from PIL using TF ops.

    Args:
      image: A 3D uint8 tensor.

    Returns:
      The image after it has had autocontrast applied to it and will be of type
      uint8.
    """

    def scale_channel(image: tf.Tensor) -> tf.Tensor:
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(image), tf.float32)
        hi = tf.cast(tf.reduce_max(image), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = tf.stack([s1, s2, s3], 2)
    return image


def sharpness(image: tf.Tensor, factor: float) -> tf.Tensor:
    """Implements Sharpness function from PIL using TF ops."""
    orig_image = image
    image = tf.cast(image, tf.float32)
    # Make image 4D for conv operation.
    image = tf.expand_dims(image, 0)
    # SMOOTH PIL Kernel.
    kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]) / 13.0
    # Tile across channel dimension.
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]
    degenerate = tf.nn.depthwise_conv2d(image, kernel, strides, padding="VALID", dilations=[1, 1])
    degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
    degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

    # For the borders of the resulting image, fill in the values of the
    # original image.
    mask = tf.ones_like(degenerate)
    padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

    # Blend the final result.
    return blend(result, orig_image, factor)


def equalize(image: tf.Tensor) -> tf.Tensor:
    """Implements Equalize function from PIL using TF ops."""

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(tf.equal(step, 0), lambda: im, lambda: tf.gather(build_lut(histo, step), im))

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)
    return image


def invert(image: tf.Tensor) -> tf.Tensor:
    """Inverts the image pixels."""
    image = tf.convert_to_tensor(image)
    return 255 - image


def wrap(image: tf.Tensor) -> tf.Tensor:
    """Returns 'image' with an extra channel set to all 1s."""
    shape = tf.shape(image)
    extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
    extended = tf.concat([image, extended_channel], axis=2)
    return extended


def unwrap(image: tf.Tensor, replace: int) -> tf.Tensor:
    """Unwraps an image produced by wrap.

    Where there is a 0 in the last channel for every spatial position,
    the rest of the three channels in that spatial dimension are grayed
    (set to 128).  Operations like translate and shear on a wrapped
    Tensor will leave 0s in empty locations.  Some transformations look
    at the intensity of values to do preprocessing, and we want these
    empty pixels to assume the 'average' value, rather than pure black.


    Args:
      image: A 3D Image Tensor with 4 channels.
      replace: A one or three value 1D tensor to fill empty pixels.

    Returns:
      image: A 3D image Tensor with 3 channels.
    """
    image_shape = tf.shape(image)
    # Flatten the spatial dimensions.
    flattened_image = tf.reshape(image, [-1, image_shape[2]])

    # Find all pixels where the last channel is zero.
    alpha_channel = tf.expand_dims(flattened_image[:, 3], axis=-1)

    replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

    # Where they are zero, fill them in with 'replace'.
    flattened_image = tf.where(tf.equal(alpha_channel, 0), tf.ones_like(flattened_image, dtype=image.dtype) * replace, flattened_image)

    image = tf.reshape(flattened_image, image_shape)
    image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
    return image


def _randomly_negate_tensor(tensor):
    """With 50% prob turn the tensor negative."""
    should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
    final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
    return final_tensor


def _rotate_level_to_arg(level: float):
    level = (level / _MAX_LEVEL) * 30.0
    level = _randomly_negate_tensor(level)
    return (level,)


def _shrink_level_to_arg(level: float):
    """Converts level to ratio by which we shrink the image content."""
    if level == 0:
        return (1.0,)  # if level is zero, do not shrink the image
    # Maximum shrinking ratio is 2.9.
    level = 2.0 / (_MAX_LEVEL / level) + 0.9
    return (level,)


def _enhance_level_to_arg(level: float):
    return ((level / _MAX_LEVEL) * 1.8 + 0.1,)  # range [0.1, 1.9]


def _enhance_increasing_level_to_arg(level: float):
    # the 'no change' level is 1.0, moving away from that towards 0. or 2.0 increases the enhancement blend
    # range [0.1, 1.9] if level <= _LEVEL_DENOM
    level = (level / _MAX_LEVEL) * 0.9
    level = 1.0 + _randomly_negate_tensor(level)
    level = tf.reduce_max([level, 0.1])  # keep it >= 0.1
    return (level,)


def _shear_level_to_arg(level: float):
    level = (level / _MAX_LEVEL) * 0.3
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return (level,)


def _scale_level_to_arg(level: float):
    level = (level / _MAX_LEVEL) * 0.5
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level) + 1.0
    return (level,)


def _translate_level_to_arg(level: float, translate_const: float):
    level = (level / _MAX_LEVEL) * float(translate_const)
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level)
    return (level,)


def _cutout_level_to_arg(level: float, cutout_const: float):
    return (int((level / _MAX_LEVEL) * cutout_const),)


def _posterize_level_to_arg(level: float):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 4),)


def _posterize_increasing_level_to_arg(level: float):
    # As per Tensorflow models research and UDA impl
    # range [4, 0], 'keep 4 down to 0 MSB of original image',
    # intensity/severity of augmentation increases with level
    return (4 - int((level / _MAX_LEVEL) * 4),)


def _posterize_original_level_to_arg(level: float):
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 4) + 4,)


def _solarize_level_to_arg(level: float):
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_LEVEL) * 256),)


def _solarize_increasing_level_to_arg(level: float):
    # range [0, 256]
    # intensity/severity of augmentation increases with level
    return (256 - int((level / _MAX_LEVEL) * 256),)


def _solarize_add_level_to_arg(level: float):
    # range [0, 110]
    return (int((level / _MAX_LEVEL) * 110),)


def _apply_func_with_prob(func: Any, image: tf.Tensor, args: Any, prob: float):
    """Apply `func` to image w/ `args` as input with probability `prob`."""
    assert isinstance(args, tuple)

    # Apply the function with probability `prob`.
    should_apply_op = tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
    augmented_image = tf.cond(should_apply_op, lambda: func(image, *args), lambda: image)
    return augmented_image


def select_and_apply_random_policy(policies: Any, image: tf.Tensor):
    """Select a random policy from `policies` and apply it to `image`."""
    policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
    # Note that using tf.case instead of tf.conds would result in significantly
    # larger graphs and would even break export for some larger policies.
    for (i, policy) in enumerate(policies):
        image = tf.cond(tf.equal(i, policy_to_select), lambda selected_policy=policy: selected_policy(image), lambda: image)
    return image


NAME_TO_FUNC = {
    "AutoContrast": autocontrast,
    "Equalize": equalize,
    "Invert": invert,
    "Rotate": rotate,
    "Posterize": posterize,
    "PosterizeIncreasing": posterize,
    "PosterizeOriginal": posterize,
    "Solarize": solarize,
    "SolarizeIncreasing": solarize,
    "SolarizeAdd": solarize_add,
    "Color": color,
    "ColorIncreasing": color,
    "Contrast": contrast,
    "ContrastIncreasing": contrast,
    "Brightness": brightness,
    "BrightnessIncreasing": brightness,
    "Sharpness": sharpness,
    "SharpnessIncreasing": sharpness,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "ScaleX": scale_x,
    "ScaleY": scale_y,
    "Scale": scale,
    "TranslateX": translate_x,
    "TranslateY": translate_y,
    "TranslateXRel": translate_x_relative,
    "TranslateYRel": translate_y_relative,
    "Cutout": cutout,
}

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset(
    {
        "Rotate",
        "TranslateX",
        "TranslateXRel",
        "ShearX",
        "ShearY",
        "ScaleX",
        "ScaleY",
        "Scale",
        "TranslateY",
        "TranslateYRel",
        "Cutout",
    }
)

LEVEL_TO_ARG = {
    "AutoContrast": lambda level: (),
    "Equalize": lambda level: (),
    "Invert": lambda level: (),
    "Rotate": _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various Tensorflow/Google repositories/papers
    "Posterize": _posterize_level_to_arg,
    "PosterizeIncreasing": _posterize_increasing_level_to_arg,
    "PosterizeOriginal": _posterize_original_level_to_arg,
    "Solarize": _solarize_level_to_arg,
    "SolarizeIncreasing": _solarize_increasing_level_to_arg,
    "SolarizeAdd": _solarize_add_level_to_arg,
    "Color": _enhance_level_to_arg,
    "ColorIncreasing": _enhance_increasing_level_to_arg,
    "Contrast": _enhance_level_to_arg,
    "ContrastIncreasing": _enhance_increasing_level_to_arg,
    "Brightness": _enhance_level_to_arg,
    "BrightnessIncreasing": _enhance_increasing_level_to_arg,
    "Sharpness": _enhance_level_to_arg,
    "SharpnessIncreasing": _enhance_increasing_level_to_arg,
    "ShearX": _shear_level_to_arg,
    "ShearY": _shear_level_to_arg,
    "ScaleX": _scale_level_to_arg,
    "ScaleY": _scale_level_to_arg,
    "Scale": _scale_level_to_arg,
    "TranslateX": _translate_level_to_arg,
    "TranslateY": _translate_level_to_arg,
    "TranslateXRel": _translate_level_to_arg,
    "TranslateYRel": _translate_level_to_arg,
    "Cutout": _cutout_level_to_arg,
}


def _parse_policy_info(name: Text, prob: float, level: float, replace_value: List[int], cutout_const: float, translate_const: float) -> Tuple[Any, float, Any]:
    """Return the function that corresponds to `name` and update `level` param."""
    func = NAME_TO_FUNC[name]
    level_func = LEVEL_TO_ARG[name]
    if name == "Cutout":
        args = level_func(level, cutout_const)
    elif name in ["TranslateX", "TranslateY", "TranslateXRel", "TranslateYRel"]:
        args = level_func(level, translate_const)
    else:
        args = level_func(level)

    if name in REPLACE_FUNCS:
        # Add in replace arg if it is required for the function that is called.
        args = tuple(list(args) + [replace_value])

    return func, prob, args


class ImageAugment(object):
    """Image augmentation class for applying image distortions."""

    def distort(self, image: tf.Tensor) -> tf.Tensor:
        """Given an image tensor, returns a distorted image with the same shape.

        Args:
          image: `Tensor` of shape [height, width, 3] representing an image.

        Returns:
          The augmented version of `image`.
        """
        return self.__call__(image)


class AutoAugment(ImageAugment):
    """Applies the AutoAugment policy to images.

    AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.
    """

    def __init__(
        self,
        augmentation_name: Text = "v0",
        policies: Optional[Dict[Text, Any]] = None,
        cutout_const: float = 100,
        translate_const: float = 250,
    ):
        """Applies the AutoAugment policy to images.

        Args:
          augmentation_name: The name of the AutoAugment policy to use. The
            available options are `v0` and `test`. `v0` is the policy used for all
            of the results in the paper and was found to achieve the best results on
            the COCO dataset. `v1`, `v2` and `v3` are additional good policies found
            on the COCO dataset that have slight variation in what operations were
            used during the search procedure along with how many operations are
            applied in parallel to a single image (2 vs 3).
          policies: list of lists of tuples in the form `(func, prob, level)`,
            `func` is a string name of the augmentation function, `prob` is the
            probability of applying the `func` operation, `level` is the input
            argument for `func`.
          cutout_const: multiplier for applying cutout.
          translate_const: multiplier for applying translation.
        """
        super(AutoAugment, self).__init__()

        if policies is None:
            self.available_policies = {
                "v0": self.policy_v0(),
                "test": self.policy_test(),
                "simple": self.policy_simple(),
            }

        if augmentation_name not in self.available_policies:
            raise ValueError("Invalid augmentation_name: {}".format(augmentation_name))

        self.augmentation_name = augmentation_name
        self.policies = self.available_policies[augmentation_name]
        self.cutout_const = float(cutout_const)
        self.translate_const = float(translate_const)

    def __call__(self, image: tf.Tensor) -> tf.Tensor:
        """Applies the AutoAugment policy to `image`.

        AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.

        Args:
          image: `Tensor` of shape [height, width, 3] representing an image.

        Returns:
          A version of image that now has data augmentation applied to it based on
          the `policies` pass into the function.
        """
        input_image_type = image.dtype

        if input_image_type != tf.uint8:
            image = tf.clip_by_value(image, 0.0, 255.0)
            image = tf.cast(image, dtype=tf.uint8)

        replace_value = [128] * 3

        # func is the string name of the augmentation function, prob is the
        # probability of applying the operation and level is the parameter
        # associated with the tf op.

        # tf_policies are functions that take in an image and return an augmented
        # image.
        tf_policies = []
        for policy in self.policies:
            tf_policy = []
            # Link string name to the correct python function and make sure the
            # correct argument is passed into that function.
            for policy_info in policy:
                policy_info = list(policy_info) + [replace_value, self.cutout_const, self.translate_const]
                tf_policy.append(_parse_policy_info(*policy_info))
            # Now build the tf policy that will apply the augmentation procedue
            # on image.
            def make_final_policy(tf_policy_):
                def final_policy(image_):
                    for func, prob, args in tf_policy_:
                        image_ = _apply_func_with_prob(func, image_, args, prob)
                    return image_

                return final_policy

            tf_policies.append(make_final_policy(tf_policy))

        image = select_and_apply_random_policy(tf_policies, image)
        image = tf.cast(image, dtype=input_image_type)
        return image

    @staticmethod
    def policy_v0():
        """Autoaugment policy that was used in AutoAugment Paper.

        Each tuple is an augmentation operation of the form
        (operation, probability, magnitude). Each element in policy is a
        sub-policy that will be applied sequentially on the image.

        Returns:
          the policy.
        """

        # TODO(dankondratyuk): tensorflow_addons defines custom ops, which
        # for some reason are not included when building/linking
        # This results in the error, "Op type not registered
        # 'Addons>ImageProjectiveTransformV2' in binary" when running on borg TPUs
        policy = [
            [("Color", 0.4, 9), ("Equalize", 0.6, 3)],
            [("Solarize", 0.8, 3), ("Equalize", 0.4, 7)],
            [("Solarize", 0.4, 2), ("Solarize", 0.6, 2)],
            [("Color", 0.2, 0), ("Equalize", 0.8, 8)],
            [("Equalize", 0.4, 8), ("SolarizeAdd", 0.8, 3)],
            [("Color", 0.6, 1), ("Equalize", 1.0, 2)],
            [("Color", 0.4, 7), ("Equalize", 0.6, 0)],
            [("Posterize", 0.4, 6), ("AutoContrast", 0.4, 7)],
            [("Solarize", 0.6, 8), ("Color", 0.6, 9)],
            [("Equalize", 0.8, 4), ("Equalize", 0.0, 8)],
            [("Equalize", 1.0, 4), ("AutoContrast", 0.6, 2)],
            [("Posterize", 0.8, 2), ("Solarize", 0.6, 10)],
            [("Solarize", 0.6, 8), ("Equalize", 0.6, 1)],
            [("Equalize", 0.8, 1), ("ShearY", 0.8, 4)],
            [("Color", 0.4, 1), ("Rotate", 0.6, 8)],
            [("ShearX", 0.2, 9), ("Rotate", 0.6, 8)],
            [("Invert", 0.4, 9), ("Rotate", 0.6, 0)],
            [("Equalize", 1.0, 9), ("ShearY", 0.6, 3)],
            [("Solarize", 0.2, 4), ("Rotate", 0.8, 9)],
            [("Rotate", 1.0, 7), ("TranslateY", 0.8, 9)],
            [("ShearX", 0.0, 0), ("Solarize", 0.8, 4)],
            [("ShearY", 0.8, 0), ("Color", 0.6, 4)],
            [("Color", 1.0, 0), ("Rotate", 0.6, 2)],
            [("ShearY", 0.4, 7), ("SolarizeAdd", 0.6, 7)],
            [("Color", 0.8, 6), ("Rotate", 0.4, 5)],
        ]
        return policy

    @staticmethod
    def policy_simple():
        """Same as `policy_v0`, except with custom ops removed."""

        policy = [
            [("Color", 0.4, 9), ("Equalize", 0.6, 3)],
            [("Solarize", 0.8, 3), ("Equalize", 0.4, 7)],
            [("Solarize", 0.4, 2), ("Solarize", 0.6, 2)],
            [("Color", 0.2, 0), ("Equalize", 0.8, 8)],
            [("Equalize", 0.4, 8), ("SolarizeAdd", 0.8, 3)],
            [("Color", 0.6, 1), ("Equalize", 1.0, 2)],
            [("Color", 0.4, 7), ("Equalize", 0.6, 0)],
            [("Posterize", 0.4, 6), ("AutoContrast", 0.4, 7)],
            [("Solarize", 0.6, 8), ("Color", 0.6, 9)],
            [("Equalize", 0.8, 4), ("Equalize", 0.0, 8)],
            [("Equalize", 1.0, 4), ("AutoContrast", 0.6, 2)],
            [("Posterize", 0.8, 2), ("Solarize", 0.6, 10)],
            [("Solarize", 0.6, 8), ("Equalize", 0.6, 1)],
        ]
        return policy

    @staticmethod
    def policy_positional():
        policy = [
            [("Equalize", 0.8, 1), ("ShearY", 0.8, 4)],
            [("Color", 0.4, 1), ("Rotate", 0.6, 8)],
            [("ShearX", 0.2, 9), ("Rotate", 0.6, 8)],
            [("Invert", 0.4, 9), ("Rotate", 0.6, 0)],
            [("Equalize", 1.0, 9), ("ShearY", 0.6, 3)],
            [("Solarize", 0.2, 4), ("Rotate", 0.8, 9)],
            [("Rotate", 1.0, 7), ("TranslateY", 0.8, 9)],
            [("ShearX", 0.0, 0), ("Solarize", 0.8, 4)],
            [("ShearY", 0.8, 0), ("Color", 0.6, 4)],
            [("Color", 1.0, 0), ("Rotate", 0.6, 2)],
            [("ShearY", 0.4, 7), ("SolarizeAdd", 0.6, 7)],
            [("Color", 0.8, 6), ("Rotate", 0.4, 5)],
        ]
        return policy

    @staticmethod
    def policy_test():
        """Autoaugment test policy for debugging."""
        policy = [
            [("TranslateX", 1.0, 4), ("Equalize", 1.0, 10)],
        ]
        return policy


class RandAugment(ImageAugment):
    """Applies the RandAugment policy to images.

    RandAugment is from the paper https://arxiv.org/abs/1909.13719,
    """

    def __init__(
        self,
        num_layers: int = 2,
        magnitude: float = 10.0,
        magnitude_max: float = _MAX_LEVEL,
        magnitude_std: float = 0.5,
        cutout_const: float = 40.0,
        translate_const: float = 0.45,  # (0, 1) for relative, > 1 for absolute.
        use_cutout: bool = False,
        use_relative_translate: bool = True,
        use_color_increasing: bool = True,
        use_positional_related_ops: bool = True,  # Set False to exlude [shear, rotate, translate]
        apply_probability: float = 0.5,
        image_mean: Union[list, tuple] = [124, 117, 104],
    ):
        """Applies the RandAugment policy to images.

        Args:
          num_layers: Integer, the number of augmentation transformations to apply
            sequentially to an image. Represented as (N) in the paper. Usually best
            values will be in the range [1, 3].
          magnitude: Integer, shared magnitude across all augmentation operations.
            Represented as (M) in the paper. Usually best values are in the range
            [5, 10].
          cutout_const: multiplier for applying cutout.
          translate_const: multiplier for applying translation.
        """
        super(RandAugment, self).__init__()

        self.num_layers, self.apply_probability, self.image_mean = num_layers, apply_probability, image_mean
        self.magnitude, self.magnitude_max, self.magnitude_std = float(magnitude), float(magnitude_max), float(magnitude_std)
        self.cutout_const, self.translate_const = float(cutout_const), float(translate_const)
        self.basic_ops = [
            "AutoContrast",
            "Equalize",
            "Invert",
            "SolarizeAdd",
        ]
        self.color_ops = ["Posterize", "Solarize", "Color", "Contrast", "Brightness", "Sharpness"]
        self.color_increasing_ops = [
            "PosterizeIncreasing",
            "SolarizeIncreasing",
            "ColorIncreasing",
            "ContrastIncreasing",
            "BrightnessIncreasing",
            "SharpnessIncreasing",
        ]
        self.positional_related_ops = ["Rotate", "ShearX", "ShearY"]
        self.positional_related_ops += ["TranslateXRel", "TranslateYRel"] if use_relative_translate else ["TranslateX", "TranslateY"]

        self.available_ops = self.basic_ops
        self.available_ops += self.color_increasing_ops if use_color_increasing else self.color_ops
        self.available_ops += ["Cutout"] if use_cutout else []
        if use_positional_related_ops:
            self.available_ops += self.positional_related_ops

    def __magnitude_with_noise__(self):
        if self.magnitude_std > 0:
            magnitude = tf.random.normal((), mean=self.magnitude, stddev=self.magnitude_std)
        else:
            magnitude = self.magnitude
        return tf.clip_by_value(magnitude, 0, self.magnitude_max)

    def apply_policy(self, policy, image):
        # print(f">>>> {policy = }")
        magnitude = self.__magnitude_with_noise__()
        # policy = random.choice(self.available_ops)
        func, _, args = _parse_policy_info(policy, 0.0, magnitude, self.image_mean, self.cutout_const, self.translate_const)
        return func(image, *args)

    def select_and_apply_random_policy(self, image):
        """Select a random policy from `policies` and apply it to `image`."""
        policy_to_select = tf.random.uniform([], maxval=len(self.available_ops), dtype=tf.int32)
        branch_fns = [(id, lambda selected=ii: self.apply_policy(selected, image)) for id, ii in enumerate(self.available_ops)]
        image = tf.switch_case(branch_index=policy_to_select, branch_fns=branch_fns, default=lambda: tf.identity(image))
        # Note that using tf.case instead of tf.conds would result in significantly
        # larger graphs and would even break export for some larger policies.
        # for (i, policy) in enumerate(self.available_ops):
        #     image = tf.cond(tf.equal(i, policy_to_select), lambda policy=policy: self.apply_policy(policy, image), lambda: image)
        return image

    def __call__(self, image: tf.Tensor) -> tf.Tensor:
        """Applies the RandAugment policy to `image`.

        Args:
          image: `Tensor` of shape [height, width, 3] representing an image.

        Returns:
          The augmented version of `image`.
        """
        input_image_type = image.dtype

        if input_image_type != tf.uint8:
            image = tf.clip_by_value(image, 0.0, 255.0)
            image = tf.cast(image, dtype=tf.uint8)

        for _ in range(self.num_layers):
            should_apply_op = tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + self.apply_probability), tf.bool)
            image = tf.cond(should_apply_op, lambda: self.select_and_apply_random_policy(image), lambda: image)
        image = tf.cast(image, dtype=input_image_type)
        return image


class PositionalRandAugment(RandAugment):
    """Applies the RandAugment positional related policy to images. Including [shear, rotate, translate], Also returns affine transform matrix"""

    def __init__(
        self,
        num_layers: int = 2,
        magnitude: float = 10.0,
        magnitude_max: float = _MAX_LEVEL,
        magnitude_std: float = 0.5,
        translate_const: float = 0.45,  # (0, 1) for relative, > 1 for absolute.
        use_relative_translate: bool = True,
        apply_probability: float = 0.5,
        image_mean: Union[list, tuple] = [124, 117, 104],
        positional_augment_methods: str = "rts",
        **kwargs,  # Not using, just in case any additional params.
    ):
        self.num_layers, self.apply_probability, self.image_mean = num_layers, apply_probability, image_mean
        self.magnitude, self.magnitude_max, self.magnitude_std = float(magnitude), float(magnitude_max), float(magnitude_std)
        self.translate_const = float(translate_const)

        positional_augment_methods = "" if positional_augment_methods is None else positional_augment_methods.lower()
        self.available_ops = ["ScaleX", "ScaleY"] if "x" in positional_augment_methods else ["Scale"]
        if "r" in positional_augment_methods:
            self.available_ops += ["Rotate"]
        if "s" in positional_augment_methods:
            self.available_ops += ["ShearX", "ShearY"]
        if "t" in positional_augment_methods:
            self.available_ops += ["TranslateXRel", "TranslateYRel"] if use_relative_translate else ["TranslateX", "TranslateY"]
        self.DEFAULT_AFFINE = tf.constant([[1.0, 0, 0, 0, 1, 0, 0, 0]])

    def apply_policy(self, policy, image):
        # print(f">>>> {policy = }")
        magnitude = self.__magnitude_with_noise__()
        func, _, args = _parse_policy_info(policy, 0.0, magnitude, self.image_mean, 0, self.translate_const)
        return func(image, *args, return_affine_matrix=True)

    def select_and_apply_random_policy(self, image):
        """Select a random policy from `policies` and apply it to `image`."""
        policy_to_select = tf.random.uniform([], maxval=len(self.available_ops), dtype=tf.int32)
        branch_fns = [(id, lambda selected=ii: self.apply_policy(selected, image)) for id, ii in enumerate(self.available_ops)]
        return tf.switch_case(branch_index=policy_to_select, branch_fns=branch_fns, default=lambda: self.DEFAULT_AFFINE)

    def __call__(self, image: tf.Tensor) -> tf.Tensor:
        input_image_type = image.dtype

        if input_image_type != tf.uint8:
            image = tf.clip_by_value(image, 0.0, 255.0)
            image = tf.cast(image, dtype=tf.uint8)

        result_affine_matrix = self.DEFAULT_AFFINE
        for _ in range(self.num_layers):
            should_apply_op = tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + self.apply_probability), tf.bool)
            affine_matrix = tf.cond(should_apply_op, lambda: self.select_and_apply_random_policy(image), lambda: self.DEFAULT_AFFINE)
            result_affine_matrix = marge_two_transforms(result_affine_matrix, affine_matrix)

        image = transform(image=wrap(image), transforms=result_affine_matrix)
        image = unwrap(image, self.image_mean)
        image = tf.cast(image, dtype=input_image_type)
        return image, result_affine_matrix
