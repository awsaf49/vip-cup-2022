import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import math
import numpy as np
import os

# HELPER


def random_int(shape=[], minval=0, maxval=1):
    return tf.random.uniform(
        shape=shape, minval=minval, maxval=maxval, dtype=tf.int32)


def random_float(shape=[], minval=0.0, maxval=1.0):
    rnd = tf.random.uniform(
        shape=shape, minval=minval, maxval=maxval, dtype=tf.float32)
    return rnd


def debuglogger(filename, txt):
    if os.path.exists(filename):
        f = open(filename, "a")
    else:
        f = open(filename, "w")
    f.write(txt + '\n')
    f.close()
    
def get_mat(shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    #rotation = math.pi * rotation / 180.
    shear    = math.pi * shear    / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    
    # ROTATION MATRIX
#     c1   = tf.math.cos(rotation)
#     s1   = tf.math.sin(rotation)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    
#     rotation_matrix = get_3x3_mat([c1,   s1,   zero, 
#                                    -s1,  c1,   zero, 
#                                    zero, zero, one])    
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)    
    
    shear_matrix = get_3x3_mat([one,  s2,   zero, 
                               zero, c2,   zero, 
                                zero, zero, one])        
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 
                               zero,            one/width_zoom, zero, 
                               zero,            zero,           one])    
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one,  zero, height_shift, 
                                zero, one,  width_shift, 
                                zero, zero, one])
    

    return  K.dot(shear_matrix,K.dot(zoom_matrix, shift_matrix))   

def ShiftScaleShearRotate(image, DIM, ROT, SHR, H_ZOOM, V_ZOOM, H_SHIFT, V_SHIFT, FILL_MODE, prob):
    if random_float() > prob:
        return image
    
    if DIM[0]>DIM[1]:
        diff  = (DIM[0]-DIM[1])
        pad   = [diff//2, diff//2 + diff%2]
        image = tf.pad(image, [[0, 0], [pad[0], pad[1]],[0, 0]])
        NEW_DIM = DIM[0]
    elif DIM[0]<DIM[1]:
        diff  = (DIM[1]-DIM[0])
        pad   = [diff//2, diff//2 + diff%2]
        image = tf.pad(image, [[pad[0], pad[1]], [0, 0],[0, 0]])
        NEW_DIM = DIM[1]
    
    rot = ROT * tf.random.normal([1], dtype='float32')
    shr = SHR * tf.random.normal([1], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / H_ZOOM
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / V_ZOOM
    h_shift = H_SHIFT * tf.random.normal([1], dtype='float32') 
    w_shift = V_SHIFT * tf.random.normal([1], dtype='float32') 
    
    transformation_matrix=tf.linalg.inv(get_mat(shr,h_zoom,w_zoom,h_shift,w_shift))
    
    flat_tensor=tfa.image.transform_ops.matrices_to_flat_transforms(transformation_matrix)
    
    image=tfa.image.transform(image,flat_tensor, fill_mode=FILL_MODE)
    
    rotation = math.pi * rot / 180.
    
    image=tfa.image.rotate(image,-rotation, fill_mode=FILL_MODE)
    
    if DIM[0]>DIM[1]:
        image=tf.reshape(image, [NEW_DIM, NEW_DIM,3])
        image = image[:, pad[0]:-pad[1],:]
    elif DIM[1]>DIM[0]:
        image=tf.reshape(image, [NEW_DIM, NEW_DIM,3])
        image = image[pad[0]:-pad[1],:,:]
    image = tf.reshape(image, [*DIM, 3])    
    return image
 

def JpegCompress(img, quality=[85, 95], prob=0.5):
    if random_float() < prob:
        img = tf.image.random_jpeg_quality(img, quality[0], quality[1])
    return img

def RandomFlip(img, prob_hflip=0.5, prob_vflip=0.0):
    if random_float() < prob_hflip:
        img = tf.image.flip_left_right(img)
    if random_float() < prob_vflip:
        img = tf.image.flip_up_down(img)
    return img

def RandomJitter(img, hue, sat, cont, bri, prob=0.25):
    if random_float() > prob:
        return img
    img = tf.image.random_hue(img, hue)
    img = tf.image.random_saturation(img, sat[0], sat[1])
    img = tf.image.random_contrast(img, cont[0], cont[1])
    img = tf.image.random_brightness(img, bri)
    return img

def Blur(img):
    if random_float()<0.5: # do median blur
        filter_size  = np.random.randint(3, 4)
        filter_shape = (filter_size, filter_size)
        aug_img = tfa.image.median_filter2d(img, filter_shape=filter_shape)
    else:
        filter_size  = np.random.randint(3, 4)
        filter_shape = (filter_size, filter_size)
        aug_img = tfa.image.gaussian_filter2d(img, filter_shape=filter_shape)
    return aug_img

def RandomGray(img, prob=0.5):
    if random_float() < prob:
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.grayscale_to_rgb(img)
    return img

def RandomBGR(img, prob=0.5):
    if random_float() < prob: # rgb to bgr
        img = img[...,::-1]
    return img

def apply_augment(image, CFG=None):
    CFG.augment_prob = 0.80
    CFG.hflip = 0.5 # horizontal flip
    CFG.vflip = 0.5 # vertical flip
    CFG.gray_prob =  0.3
    CFG.sssr_prob = 0.65
    CFG.fill_mode = 'constant'
    CFG.rot = 5.0
    CFG.shr = 5.0
    CFG.h_zoom = 50.0
    CFG.v_zoom = 50.0
    CFG.h_shift = 30.0
    CFG.v_shift = 30.0
    
    if random_float() > CFG.augment_prob:
        return image
    
    image = RandomFlip(image, prob_hflip=CFG.hflip, prob_vflip=CFG.vflip)
    image = RandomGray(image, prob=CFG.gray_prob)
    # image = ShiftScaleShearRotate(image, 
    #                             CFG.img_size, 
    #                             CFG.rot, CFG.shr, 
    #                             CFG.h_zoom,
    #                             CFG.v_zoom, 
    #                             CFG.h_shift, 
    #                             CFG.v_shift, 
    #                             CFG.fill_mode,
    #                             prob=CFG.sssr_prob)
        
    return image





