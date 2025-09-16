import argparse
from skdh.io import ReadCwa
import pandas as pd
import numpy as np
from scipy.signal import butter, besselap, zpk2ss, ss2tf, lp2lp, lfilter, filtfilt, resample, resample_poly
import os

def resample_2(accel_array, data_time, resampling_frequency=50.0):
    #Resample to resampling_frequency (Hz)
    N = len(data_time);
    #Get sampling_frequency
    dT = np.median(np.diff(data_time,axis=0));
    fs = 1 / dT; #raw data sampling rate (Hz)
    dfs = resampling_frequency;
    N = round((dfs/fs)*N);
    accel_array = resample(accel_array, N);
    fs = dfs;
    dT = 1.0 / fs;
    time_array = np.linspace(0.0, N*dT, N);
    return accel_array, time_array
def resample_2fast(accel_array, data_time, resampling_frequency=50.0):
    #Resample to resampling_frequency (Hz)
    N = len(data_time);
    #Get sampling_frequency
    dT = np.median(np.diff(data_time,axis=0));
    fs = 1 / dT; #raw data sampling rate (Hz)
    dfs = resampling_frequency;
    N = round((dfs/fs)*N);
    accel_array = resample_poly(accel_array, int(resampling_frequency), int(round(fs)), axis=0)
    fs = dfs;
    dT = 1.0 / fs;
    time_array = np.linspace(0.0, N*dT, N);
    return accel_array, time_array

def standardize_sensor_orientation(acc):
    """
    Check Sensor Orientation
    Align so that the Z-axis points along the gravity direction
    """
    # find the direction of gravity
    g = np.mean(acc, axis=0)
    g /= np.linalg.norm(g) #unit vector
    #alignment_axis = np.array([1, 0, 0]) #align +x -> gravity direction
    #alignment_axis = np.array([0, 1, 0]) #align +y -> gravity direction
    alignment_axis = np.array([0, 0, 1]) #align +z -> gravity direction
    
    # rotate
    v = np.cross(g, alignment_axis)
    c = np.dot(g, alignment_axis)
    s = np.linalg.norm(v)
    if s != 0:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
        acc = acc @ R.T
        
    ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
    # Perform flipping x and y axes to ensure standard orientation
    # For correct orientation, x-angle should be mostly negative
    # So, if median x-angle is positive, flip both x and y axes
    # Ref: https://github.com/wadpac/hsmm4acc/blob/524743744068e83f468a4e217dde745048a625fd/UKMovementSensing/prepacc.py
    angle_x = np.arctan2(ax, np.sqrt(ay*ay + az*az)) * 180.0/np.pi
    if np.median(angle_x) > 0:
      ax *= -1
      ay *= -1
      print("non-standard sensor orientation detected")

    return np.stack([ax, ay, az], axis=1)

def BPfilterButter(x,Fs,order=4,cutoff_low=3,cutoff_high=8):
    """
    Band pass Butterworth filter data.
    Inputs
    ------
    x : numpy.ndarray
        The input data.
    Fs : float
         The sampling frequency.
    order : int
            The filter order.
    cutoff_low : float
                 The desired lower bound cutoff frequency in Hertz.
    cutoff_high : float
                  The desired upper bound cutoff frequency in Hertz.
    Outputs
    -------
    Notes
    -----


    """
    #filter design
    cutoff_low_norm = cutoff_low/(0.5*Fs)
    cutoff_high_norm = cutoff_high/(0.5*Fs)
    b,a = butter(order,[cutoff_low_norm,cutoff_high_norm],btype='bandpass',analog=False)
    #filter data
    xfilt = np.zeros_like(x)
    for i in range(x.shape[1]):  # filter each channel separately
        xfilt[:, i] = filtfilt(b, a, x[:, i], method='pad')
    return xfilt

def convert_cwa_to_csv(input_file):
    reader = ReadCwa()
    data = reader.predict(file=input_file)

    # 'accel' key contains acceleration data as (N x 3) ndarray in units of g
    accel_array = np.asarray(data.get('accel'))
    time_array = np.asarray(data.get('time'))
    
    #TODO: crop to n hrs for now
    #n=24;
    #accel_array = accel_array[0:50*60*60*n,:] 

    #Resample to resampling_frequency (Hz)
    fs = 50.0;        
    accel_array, time_array = resample_2fast(accel_array, time_array, resampling_frequency=fs)
    ws=128
       
    df = pd.DataFrame(accel_array, columns=['acc_x', 'acc_y', 'acc_z'])
    filename = f"accsamp.csv"
    filepath = os.path.join(os.getcwd(), filename)   
    #Save to csv in chunks so js doesn't choke on header
    chunk_size = ws*100 #TODO: hardcoded for now
    for i in range(0, len(df), chunk_size):
        df.iloc[i:i+chunk_size].to_csv(
            filepath,
            mode='a',          # append after first chunk
            index=False,
            header=(i==0),     # write header only for first chunk
            #float_format='%.2f'
        )
    print(f"Saved to {filepath}")
    #sed -i '1s/.*/acc_x,acc_y,acc_z/' accsamp.csv
    
    return

def preprocess(acc):
    # Pass
    return acc

def save_A(filename, numpy_array):
    if numpy_array.ndim == 1:
        numpy_array = numpy_array[:, None, None]
    elif numpy_array.ndim == 2:
        numpy_array = numpy_array[:, :, None]
    elif numpy_array.ndim == 3:
        numpy_array = numpy_array[:, :, :]
    else:
        print("check arr shape")
        return
    print(numpy_array.shape)        
    A = np.zeros((numpy_array.shape[0], numpy_array.shape[1], numpy_array.shape[2]), dtype = np.float32, order = 'F')
    A[0:numpy_array.shape[0],0:numpy_array.shape[1],0:numpy_array.shape[2]] = numpy_array[:, :, :]
    with open(filename, "wb") as binary_file:
        binary_file.write(A.tobytes('F'))
    
# Load weights and biases to model
def load_A(filename, shape):
    with open(filename, "rb") as f:
        numpy_array = np.frombuffer(f.read(), dtype=np.float32, count=np.prod(shape))
        numpy_array = numpy_array.reshape(shape, order='F')
    return numpy_array