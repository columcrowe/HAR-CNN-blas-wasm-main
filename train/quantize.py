#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import torch
torch.backends.nnpack.enabled = False
from model import Conv1D_MLP
from utils import load_A, resample_2fast
from skdh.io import ReadCwa
import traceback
import sys

def to_fixed_point(x, scale=127.0, np_type=np.int8):
    maxVal = 5.0 #np.max(np.abs(x))
    x_scaled = np.clip(x * (scale/maxVal), ((scale+1)*(-1)), scale)
    #print(scale / np.max(np.abs(x)))
    return x_scaled.astype(np_type)
def convert_cwa_to_bin(input_file):
    reader = ReadCwa()
    data = reader.predict(file=input_file)

    # 'accel' key contains acceleration data as (N x 3) ndarray in units of g
    accel_array = np.asarray(data.get('accel'), dtype=np.float32)
    time_array = np.asarray(data.get('time'))
    
    #TODO: crop to n hrs for now
    n=36;
    accel_array = accel_array[0:50*60*60*n,:] 

    #Resample to resampling_frequency (Hz)
    fs = 50.0;        
    accel_array, time_array = resample_2fast(accel_array, time_array, resampling_frequency=fs)
    ws=128
    
    # quantization
    # scale = 127.0 # INT8 max
    # accel_array = to_fixed_point(accel_array, scale=scale, np_type=np.int8)
    scale = 32767.0 # INT16 max
    accel_array = to_fixed_point(accel_array, scale=scale, np_type=np.int16)

    # save to binary instead
    filename = f"accsamp_q.dat"
    filepath = os.path.join(os.getcwd(), filename)   
    accel_array.tofile(filepath)
    # # Save metadata
    # metadata = {
        # "n_channels": accel_array.shape[1],
        # "dtype": "int8",
        # "n_samples": accel_array.shape[0]
    # }
    # with open("meta_data.json", "w") as f:
        # json.dump(metadata, f)
    
    return


if __name__ == '__main__':
    try:
        convert_cwa_to_bin(sys.argv[1])
    except:
        traceback.print_exc()



