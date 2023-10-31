import json
import numpy as np

class Config:
    def __init__(self, data):
        self.__dict__.update(**data)
        
def dict2cfg(cfg_dict):
    """Create config from dictionary.

    Args:
        cfg_dict (dict): dictionary with configs to be converted to config.

    Returns:
        cfg: python class object as config
    """

    cfg = Config(cfg_dict)  # dict to cfg
    cfg.label2name = dict(zip(cfg.class_labels, cfg.class_names))
    return cfg


def cfg2dict(cfg):
    """Create dictionary from config.

    Args:
        cfg (config): python class object as config.

        Returns:
            cfg_dict (dict): dictionary with configs.
    """
    return {k: v for k, v in dict(vars(cfg)).items() if '__' not in k}


class NumpyEncoder(json.JSONEncoder):
    """
    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    Special json encoder for numpy types
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
