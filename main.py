# Module Path
import sys
from sys import argv
paths = sys.argv[0].rsplit('/', 1)
if len(paths)>1:
    cwd = paths[0]
    d = f'{cwd}/models'
else:
    d = './models'
    cwd = '.'
sys.path.append(d)

# Import
import json
import time, os
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf

# Dependencies
from utils.config import Config
from dataset.dataset import build_dataset,seeding
from utils.device import get_device

# Models
from keras_cv_attention_models import (
    nfnets,
    resnet_family,
    volo, resnest,
    mlp_family,
    efficientnet,
    swin_transformer_v2)
import resnet_rs
import tfimm
import gcvit

# Logging
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

NAME2BS = {
    "convnext_large_384_in22ft1k-200x200": 16,
    "convnext_large_in22ft1k-200x200": 16,
    "convnext_base_384_in22ft1k-200x200": 32,
    "HorNetBase-200x200": 32,
    "EfficientNetV2M-200x200": 64,
    "convnext_base_in22k-200x200": 32,
    "ECA_NFNetL2-200x200": 32,
    "GCViTBase-224x224": 48,
    "ResNest200-200x200": 64,
    "EfficientNetV2L-200x200": 32,
    "ResNetRS200-200x200": 32,
    "ResNet200D-200x200": 32,
}

def predict_soln(CFG,ensemble=False):
    if CFG.verbose == 1:
        print('='*35)
        print('### INFERENCE ###')
        print('='*35)
        
    # PREDICTION FOR ALL MODELS
    id_keys={}
    pred_dfs = []
    for model_idx, (model_paths, dim,idx) in enumerate(CFG.ckpt_cfg):
        preds=[]
        
        source_path = os.path.dirname(os.path.dirname(model_paths[0]))
        model_name = os.path.basename(source_path)
        pred_save_path = CFG.temp_save_dir
        if CFG.verbose:
            print(f'> MODEL({model_idx+1}/{len(CFG.ckpt_cfg)}): {model_name} | DIM: {dim}')

        # META DATA
        test_csv = pd.read_csv(CFG.test_csv)
        test_names = test_csv.filename.values
        test_paths =  [os.path.join(CFG.infer_path,name) for name in test_names]

        # CONFIGURE BATCHSIZE
        if CFG.debug:
            test_paths = test_paths[:100]
        # BATCH SIZE
        CFG.batch_size = 8 * NAME2BS.get(model_name, 16) # 8 * 16
        print("> BATCH SIZE : ",CFG.batch_size)
        CFG.img_size = dim
        # BUILD DATASET
        dtest = build_dataset(
                    test_paths,
                    labels=None,
                    augment=CFG.tta > 1,
                    repeat=True,
                    cache=False,
                    shuffle=False,
                    batch_size=CFG.batch_size,
                    drop_remainder=False,
                    CFG=CFG)

        # PREDICTION FOR ONE MODEL -> N FOLDS
        for model_path in sorted(model_paths):
            # SavedModel
            if '.pb' in model_path:
                model_path = model_path.rsplit('/',1)[0]
            # Load Model
            with strategy.scope():
                model = tf.keras.models.load_model(model_path, compile=False)
            # Predict
            pred = model.predict(dtest, steps = max(CFG.tta*len(test_paths)/CFG.batch_size,1), verbose=1)
            pred = pred[:CFG.tta*len(test_paths),:]
            pred = getattr(np, CFG.agg)(pred.reshape((CFG.tta, len(test_paths), -1)),axis=0)
            # MultiClass to Binary
            if pred.shape[1] > 1:
                pred = 1 - pred[:, 0:1] # multi to binary
            preds.append(pred) 
        
        if CFG.verbose:
            print('> PROCESSING SUBMISSION')

        # PROCESSS PREDICTION
        preds = getattr(np, CFG.agg)(preds, axis=0)
        test_names = np.array(test_names)   
        columns = ['filename', 'logit']
        pred_df = pd.DataFrame(np.concatenate([test_names[:,None], preds], axis=1), columns=columns)
        pred_path = os.path.join(pred_save_path, model_name+'_pred.csv') 
        
        pred_df = test_csv.merge(pred_df, on=['filename'], how='right').reset_index(drop=True)
        pred_dfs.append(pred_df)
        #pred_df.to_csv(pred_path,index=False)
        
        if CFG.verbose == 1:
            #print(F'\n> SUBMISSION SAVED TO: {pred_path}')
            print(pred_df.head(2))
        id_keys[idx]=pred_path #----------------------------------------------------------------------------
        
        print('\n\n')
        
    #all_sub_paths = [id_keys[x] for x in sorted(id_keys.keys())]
    
    if ensemble:
        #dfs = pd.concat([pd.read_csv(i) for i in all_sub_paths])
        dfs = pd.concat(pred_dfs)
        pred_df = dfs.groupby('filename')[['logit']].mean().reset_index()
        pred_df['logit'] = (pred_df.logit>CFG.thr)*1.0  # modified
        pred_df.to_csv(CFG.output_csv_path, index = False)
        
        if CFG.verbose:
            print('\n> FINAL PREDICTION SAVED TO ',CFG.output_csv_path)
        print(pred_df.head(2))
        
if __name__ == '__main__':
    
    #================================================================================
    # Input Argument
    input_csv_path = argv[1]  # input csv
    output_csv_path = argv[2] # output csv
    
    model_dir = os.path.join(cwd,'ckpts') ## CHANGE THIS
    ckpt_cfg = os.path.join(cwd,'ckpts', 'ckpts.json')
    
    debug = 0
    verbose = 1
    output =  os.path.dirname(output_csv_path) 
    infer_path = os.path.dirname(input_csv_path) # directory of testset
    temp_save_dir = os.path.join(output, 'temp')
    os.makedirs(temp_save_dir, exist_ok= True)
    tta = 1 # number of tta

    #================================================================================
    # LOADING CONFIG
    CFG = Config({})

    # LOADING CKPT CFG

    CFG.test_csv = input_csv_path
    CFG.output_csv_path = output_csv_path
    CFG.verbose = verbose
    CFG.model_dir = model_dir
    CFG.temp_save_dir = temp_save_dir
    os.system(f'mkdir -p {CFG.temp_save_dir}')

    CKPT_CFG = []
    CKPT_CFG_PATH = ckpt_cfg
    

    for base_dir, dim, idx in json.load(open(CKPT_CFG_PATH, 'r')):
        h5_paths = sorted(glob(os.path.join(model_dir,base_dir,'ckpt','*h5')))
        sm_path = os.path.join(model_dir,base_dir,'ckpt','saved_model.pb')
        if len(h5_paths):  # h5py format
            CKPT_CFG.append([h5_paths, dim, idx])
        elif os.path.isfile(sm_path):  # SavedModel format
            CKPT_CFG.append([[sm_path], dim, idx])
        else:
            raise ValueError('no model found for :', base_dir)
    print('\n> CHECKPOINTS: ')
    [print(i) for i in CKPT_CFG]
    CFG.ckpt_cfg = CKPT_CFG
    CFG.infer_path = infer_path

    # OVERWRITE
    if debug is not None:
        CFG.debug = debug

    if CFG.verbose:
        print('> DEBUG MODE:', bool(CFG.debug))

    if tta is not None:
        CFG.tta = tta

    # CREATE SUBMISSION DIRECTORY
    CFG.output_dir = output

    # CONFIGURE DEVICE
    strategy, device = get_device()
    CFG.device   = device
    AUTO         = tf.data.experimental.AUTOTUNE
    CFG.replicas = strategy.num_replicas_in_sync
    print(f'> REPLICAS: {CFG.replicas}')   

    # SETTING UP CFG VARIABLES
    CFG.agg = 'mean'
    CFG.resize_method = 'bicubic'
    CFG.num_classes = 1
    CFG.seed = 42
    CFG.thr = 0.487
    
    # SEEDING
    seeding(CFG)

    # Prediction
    start = time.time()
    predict_soln(CFG,ensemble=True)
    end = time.time()
    eta = (end-start)/60  # time for prediction
    print(f'\n> TIME TO INFER: {eta:0.2f} min')