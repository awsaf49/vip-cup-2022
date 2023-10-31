import tensorflow as tf

def get_device():
    gpus = tf.config.list_logical_devices('GPU')
    ngpu = len(gpus)
    if ngpu:  # if number of GPUs are 0 then CPU
        strategy = tf.distribute.MirroredStrategy(gpus)  # single-GPU or multi-GPU
        device = 'GPU'
    else:
        strategy = tf.distribute.get_strategy()
        device = 'CPU'
    print(f'\n> DEVICE: {device}')
    return strategy, device
