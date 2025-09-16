#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import torch
torch.backends.nnpack.enabled = False
from sklearn.metrics import accuracy_score
from tqdm import tqdm
#from keras.datasets import mnist
from load import load_dataset, load_data_loaders
from model import Conv1D_MLP
from utils import load_A, save_A


#config
ws = 2.56 #seconds
fs = 50 #Hz
window_size = int(ws*fs); #128
step_size = window_size; #64;
n_channels = 3
n_input = window_size #128*3 #28*28
n_output = 4#6 #10
batch_size = 1024#32

# Load data
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train, y_train, x_test, y_test = load_dataset()
# x_train_row = np.transpose(x_train, (0, 2, 1)) #np.reshape(x_train, (-1, n_input))
# x_test_row = np.transpose(x_test, (0, 2, 1)) #np.reshape(x_test, (-1, n_input))
# y_train = y_train.ravel() 
# y_test = y_test.ravel() 
# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(x_train_row, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(x_test_row, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=batch_size, shuffle=True)

train_loader, test_loader = load_data_loaders()
x_train = []
y_train = []
for xb, yb in train_loader:
    x_train.append(xb.numpy())
    y_train.append(yb.numpy())
    if len(x_train) >= batch_size:
        break
x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
x_train = np.transpose(x_train, (0, 2, 1))

# Create some examples from the training data
output_dir = "class_samples"
os.makedirs(output_dir, exist_ok=True)
for class_label in range(n_output):
    class_indices = np.where(y_train == class_label)[0]
    if len(class_indices) == 0:
        print(f"No samples found for class {class_label}")
        continue
    sample_idx = np.random.choice(class_indices)
    sample = x_train[sample_idx]
    df_sample = pd.DataFrame(sample, columns=['acc_x', 'acc_y', 'acc_z'])
    filename = f"accsamp_class{class_label}.csv"
    filepath = os.path.join(output_dir, filename)
    df_sample.to_csv(filepath, index=False)
    print(f"Saved sample for class {class_label} to {filepath}")


#Create model
model = Conv1D_MLP(n_channels, n_input, n_output)#.double() #np.float64 and np.double are same
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
# # Load weights and biases to model
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


# Main training loop
epochs=20
try:
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        total, correct = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = model(batch_x)
                _, predicted = torch.max(output, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        test_accuracy = correct / total
        print(f"PyTorch MLP Accuracy: {test_accuracy:.4f}")
except KeyboardInterrupt:
    print("\nTraining interrupted!")


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

