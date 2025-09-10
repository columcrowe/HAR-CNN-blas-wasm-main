import os
import pandas as pd
import numpy as np
import torch
torch.backends.nnpack.enabled = False
from sklearn.metrics import accuracy_score
from tqdm import tqdm
#from keras.datasets import mnist
#from keras.utils import to_categorical
from load import load_dataset
from utils import preprocess




ws = 30 #seconds
fs = 50
window_size = ws*fs; #128
step_size = window_size;#64;
# Load data
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train, x_test, y_test = load_dataset()
n_input = window_size #128*3 #28*28
n_output = 4 #6 #10

for i, accel_array in enumerate(x_train):
    # Preprocessing
    accel_array = preprocess(accel_array)
    # Overwrite original
    x_train[i] = accel_array
    
for i, accel_array in enumerate(x_test):
    # Preprocessing
    accel_array = preprocess(accel_array)
    # Overwrite original
    x_test[i] = accel_array

x_train_row = np.transpose(x_train, (0, 2, 1)) #np.reshape(x_train, (-1, n_input))
x_test_row = np.transpose(x_test, (0, 2, 1)) #np.reshape(x_test, (-1, n_input))
y_train = y_train.ravel() 
y_test = y_test.ravel() 
batch_size = 32
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(x_train_row, dtype=torch.float64), torch.tensor(y_train, dtype=torch.long)), batch_size=batch_size, shuffle=True)

# Create some examples
output_dir = "class_samples"
os.makedirs(output_dir, exist_ok=True)
for class_label in range(n_output):
    class_indices = np.where(y_train == class_label)[0]
    if len(class_indices) == 0:
        print(f"No samples found for class {class_label}")
        continue
    sample_idx = np.random.choice(class_indices)
    sample = x_train[sample_idx]
    df_sample = pd.DataFrame(sample, columns=['total_acc_x', 'total_acc_y', 'total_acc_z'])
    filename = f"accsamp_class{class_label}.csv"
    filepath = os.path.join(output_dir, filename)
    df_sample.to_csv(filepath, index=False)
    print(f"Saved sample for class {class_label} to {filepath}")

#Model defintion
class Conv1D_MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_channels=3
        self.conv1 = torch.nn.Conv1d(in_channels=self.n_channels, out_channels=64, kernel_size=11, stride=5)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=5)
        self.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=11, stride=5)
        self.dropout = torch.nn.Dropout(0.5)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate size after conv+pool to set fc1 input size:
        # Input length = n_input = window_size
        # After conv1 (kernel=n, stride=1, padding=0): length = n_input - kernel_size_n + 1 = x1
        # After conv2: length = x1 - kernel_size_n + 1 = x2
        # After maxpool (kernel=m, stride=kernel_size_m by default, padding=0):
        # length = ((x2 +2p - kernel_size_m) // s) + 1 = x3
        conv_kernel_size=11 #receptive field = 1 + sum[for n in n_conv_layers]((kernel_size_n-1)*cum[for n](conv_stride)) + (kernel_size_m-1) * cum[for n_conv_layers](conv_stride)
        n_conv_layers=3
        #self.fc1 = torch.nn.Linear(64 * ((n_input-(conv_kernel_size-1)*n_conv_layers)//2), 64)
        
        self.gap = torch.nn.AdaptiveAvgPool1d(1) #GAP instead - lose fine temporal patterns
        self.fc1 = torch.nn.Linear(64, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, n_output)
    def forward(self, x):
        # x shape: (batch_size, n_input)
        # x = x.unsqueeze(1)                              # (batch_size, 1, n_input)
        x = x.view(x.size(0), self.n_channels, n_input)   # (batch, channels=n_channels, n_input=window_size)
        x = torch.nn.functional.relu(self.conv1(x))       # (batch_size, 64, x1)
        x = torch.nn.functional.relu(self.conv2(x))       # (batch_size, 64, x2)
        x = torch.nn.functional.relu(self.conv3(x))       # (batch_size, 64, x3)
        x = self.dropout(x)
        x = self.pool(x)                                  # (batch_size, 64, x3)
        x = self.gap(x)
        x = x.view(x.size(0), -1)                         # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        
#Create model
model = Conv1D_MLP().double()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

# Main training loop
epochs=5
for epoch in tqdm(range(epochs), desc="Epochs"):
    model.train()
    for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(x_test_row, dtype=torch.float64))
        _, predicted = torch.max(output, 1)
    print("PyTorch MLP Accuracy:", accuracy_score(y_test, predicted.numpy()))

# Save weights and biases to data file
with torch.no_grad():
    conv1_w = model.conv1.weight.detach().cpu().numpy()
    conv1_b = model.conv1.bias.detach().cpu().numpy()
    conv2_w = model.conv2.weight.detach().cpu().numpy()
    conv2_b = model.conv2.bias.detach().cpu().numpy()
    conv3_w = model.conv3.weight.detach().cpu().numpy()
    conv3_b = model.conv3.bias.detach().cpu().numpy()
    fc1_w = model.fc1.weight.t().detach().numpy()
    fc1_b = model.fc1.bias.detach().numpy()
    fc2_w = model.fc2.weight.t().detach().numpy()
    fc2_b = model.fc2.bias.detach().numpy()
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
    A = np.zeros((numpy_array.shape[0], numpy_array.shape[1], numpy_array.shape[2]), dtype = np.double, order = 'F')
    A[0:numpy_array.shape[0],0:numpy_array.shape[1],0:numpy_array.shape[2]] = numpy_array[:, :, :]
    with open(filename, "wb") as binary_file:
        binary_file.write(A.tobytes('F'))
save_A("conv1_weight.data",conv1_w)
save_A("conv1_bias.data",conv1_b)
save_A("conv2_weight.data",conv2_w)
save_A("conv2_bias.data",conv2_b)
save_A("conv3_weight.data",conv3_w)
save_A("conv3_bias.data",conv3_b)
save_A("fc1_weight.data",fc1_w)
save_A("fc1_bias.data",fc1_b)
save_A("fc2_weight.data",fc2_w)
save_A("fc2_bias.data",fc2_b)

# # Save weights and biases to model
# def load_A(filename, shape):
    # with open(filename, "rb") as f:
        # numpy_array = np.frombuffer(f.read(), dtype=np.float64, count=np.prod(shape))
        # numpy_array = numpy_array.reshape(shape, order='F')
    # return numpy_array
# with torch.no_grad():
    # model.conv1.weight.data.copy_(torch.tensor(load_A("conv1_weight.data", model.conv1.weight.shape)))
    # model.conv1.bias.data.copy_(torch.tensor(load_A("conv1_bias.data", model.conv1.bias.shape)))
    # model.conv2.weight.data.copy_(torch.tensor(load_A("conv2_weight.data", model.conv2.weight.shape)))
    # model.conv2.bias.data.copy_(torch.tensor(load_A("conv2_bias.data", model.conv2.bias.shape)))
    # model.conv3.weight.data.copy_(torch.tensor(load_A("conv3_weight.data", model.conv3.weight.shape)))
    # model.conv3.bias.data.copy_(torch.tensor(load_A("conv3_bias.data", model.conv3.bias.shape)))
    # model.fc1.weight.data.copy_(torch.tensor(load_A("fc1_weight.data", model.fc1.weight.t().shape)).t())
    # model.fc1.bias.data.copy_(torch.tensor(load_A("fc1_bias.data", model.fc1.bias.shape)))
    # model.fc2.weight.data.copy_(torch.tensor(load_A("fc2_weight.data", model.fc2.weight.t().shape)).t())
    # model.fc2.bias.data.copy_(torch.tensor(load_A("fc2_bias.data", model.fc2.bias.shape)))