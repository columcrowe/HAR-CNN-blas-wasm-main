import sys
import numpy as np
from utils import convert_cwa_to_csv
from model import Conv1D_MLP
from utils import load_A



def main():
  convert_cwa_to_csv(sys.argv[1])
  #Create model
  ws = 2.56 #seconds
  fs = 50 #Hz
  window_size = int(ws*fs); #128
  step_size = window_size; #64;
  n_channels = 3
  n_input = window_size #128*3 #28*28
  n_output = 4 #6 #10
  batch_size = 32
  model = Conv1D_MLP(n_channels, n_input, n_output)#.double() #np.float64 and np.double are same
  floats = load_A("conv1_weight.data", model.conv1.weight.shape)
  print(floats.shape)
  print(floats[:10])
  print(floats.dtype)
	
if __name__ == '__main__':
  main()