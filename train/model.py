import torch


#Model defintion
class Conv1D_MLP(torch.nn.Module):
    def __init__(self, n_channels, n_input, n_output):
        super().__init__()
        self.n_channels=n_channels
        self.n_input=n_input
        self.n_output=n_output
        self.conv1 = torch.nn.Conv1d(in_channels=self.n_channels, out_channels=64, kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.dropout = torch.nn.Dropout(0.5)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate size after conv+pool to set fc1 input size:
        # Input length = n_input = window_size
        # After conv1 (kernel=n, stride=1, padding=0): length = n_input - kernel_size_n + 1 = x1
        # After conv2: length = x1 - kernel_size_n + 1 = x2
        # After maxpool (kernel=m, stride=kernel_size_m by default, padding=0):
        # length = ((x2 +2p - kernel_size_m) // s) + 1 = x3
        conv_kernel_size=3 #receptive field = 1 + sum[for n in n_conv_layers]((kernel_size_n-1)*cum[for n](conv_stride)) + (kernel_size_m-1) * cum[for n_conv_layers](conv_stride)
        n_conv_layers=3
        #self.fc1 = torch.nn.Linear(64 * ((n_input-(conv_kernel_size-1)*n_conv_layers)//2), 64)
        
        self.gap = torch.nn.AdaptiveAvgPool1d(1) #GAP instead - lose fine temporal patterns
        self.fc1 = torch.nn.Linear(64, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, self.n_output)
    def forward(self, x):
        # x shape: (batch_size, n_input)
        # x = x.unsqueeze(1)                                    # (batch_size, 1, n_input)
        x = x.view(x.size(0), self.n_channels, self.n_input)    # (batch, channels=n_channels, n_input=window_size)
        x = torch.nn.functional.relu(self.conv1(x))             # (batch_size, 64, x1)
        x = torch.nn.functional.relu(self.conv2(x))             # (batch_size, 64, x2)
        x = torch.nn.functional.relu(self.conv3(x))             # (batch_size, 64, x3)
        x = self.dropout(x)
        x = self.pool(x)                                        # (batch_size, 64, x3)
        x = self.gap(x)
        x = x.view(x.size(0), -1)                               # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x