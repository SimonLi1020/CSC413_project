from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime  # Correct import for datetime

# apply a white background with grid lines
sns.set_style('whitegrid')


plt.style.use("fivethirtyeight")
# %matplotlib inline

yf.pdr_override()

# The tech stocks we'll use for this analysis
#tech_list = ['AAPL', 'GOOG', "GOOGL", 'AMZN', 'TSLA', 'MSFT']
tech_list = ['AAPL']

# Set up Start and End times for data grab
start = datetime(2019, 1, 1)  # Start date
end = datetime(2020, 12, 31)  # End date

# Downloading the stock data
for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)

#company_name = ["APPLE", "GOOGLE", "Google Inc", "AMAZON", "Tesla", "MICROSOFT"]
#company_list = [AAPL, GOOG, GOOGL, AMZN, TSLA, MSFT]
company_list = [AAPL]

company_name = ["APPLE"]

# Adding company names
for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name

# Concatenating the dataframes
df = pd.concat(company_list, axis=0)
df.tail(10)


ma_day = [10, 20, 50]

# Assuming company_list is a list of DataFrames for each company
for company in company_list:
    for ma in ma_day:
        column_name = f"MA for {ma} days"
        company[column_name] = company['Adj Close'].rolling(ma).mean()

# Creating a 2x3 subplot grid
fig, axes = plt.subplots(nrows=2, ncols=3)
fig.set_figheight(10)
fig.set_figwidth(15)

# Plotting each company on its respective subplot
AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,0])
axes[0,0].set_title('APPLE')

# GOOG[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,1])
# axes[0,1].set_title('GOOGLE')

# GOOGL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,2])
# axes[0,2].set_title('GOOGLE Inc')

# AMZN[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,0])
# axes[1,0].set_title('AMAZON')

# TSLA[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,1])
# axes[1,1].set_title('TESLA')

# MSFT[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,2])
# axes[1,2].set_title('MICROSOFT')

fig.tight_layout()

# Assuming AAPL (or any other stock in the list) has the date as index
close_prices = pd.DataFrame({stock: globals()[stock]['Close'] for stock in tech_list}, index=globals()[tech_list[0]].index)

# Normalize or scale your data - Example using Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
series = scaler.fit_transform(close_prices)

print(close_prices[:5])  # Print the first 5 rows
print("Shape:", close_prices.shape)
print("data type:", close_prices.dtypes)

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, series, window_size):
        """
        series: A 2D array-like object where each row is a time step and columns are features.
        window_size: Number of time steps in each input sequence.
        """
        self.series = series
        self.window_size = window_size

    def __len__(self):
        # Adjusting length to avoid out-of-bounds indexing
        return len(self.series) - self.window_size

    def __getitem__(self, index):
        # Ensure that we don't go out of bounds
        if index + self.window_size >= len(self.series):
            raise IndexError("Index out of bounds")
        """
        Returns a tuple (input_sequence, target_value) for the given index.
        """
        x = self.series[index:index + self.window_size]  # Windowed input sequence
        y = self.series[index + self.window_size]        # Target value (next time step)
        return torch.Tensor(x), torch.Tensor(y)



from torch.utils.data import TensorDataset

train_size = int(len(series) * 0.7)
val_size = int(len(series) * 0.15)

train_series = series[:train_size]
val_series = series[train_size:train_size + val_size]
test_series = series[train_size + val_size:]

print(train_series[:5])


window_size = 10

# Create instances of TimeSeriesDataset
train_dataset = TimeSeriesDataset(train_series, window_size)
val_dataset = TimeSeriesDataset(val_series, window_size)
test_dataset = TimeSeriesDataset(test_series, window_size)


from torch.utils.data import DataLoader

# DataLoader for batching
batch_size = 3
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


import torch.nn.functional as F
import matplotlib.pyplot as plt

class TemporalPatternAttention(nn.Module):
    def __init__(self, num_units, window_size, num_filters):
        super(TemporalPatternAttention, self).__init__()
        self.num_filters = num_filters
        self.conv_filters = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=window_size) for _ in range(window_size)])
        self.Wa = nn.Linear(num_filters, num_units)
        self.Wh = nn.Linear(num_units, num_units)
        self.Wv = nn.Linear(num_filters, num_units)

    def forward(self, H, ht):
        # H: [batch_size, window_size, num_units], ht: [batch_size, num_units]
        batch_size, window_size, num_units = H.shape

        # Apply CNN filters
        HC = torch.cat([self.conv_filters[i](H[:, i:i+1, :]).permute(0, 2, 1) for i in range(window_size)], dim=1)  # [batch_size, window_size, num_filters]

        # Attention weights
        alpha = torch.sigmoid(self.Wa(HC) + self.Wh(ht).unsqueeze(1))  # [batch_size, window_size, num_units]
        vt = torch.sum(alpha * HC, dim=1)  # [batch_size, num_filters]

        # Final prediction
        ht_new = self.Wv(vt) + self.Wh(ht)  # [batch_size, num_units]
        return ht_new

class RNNWithTemporalPatternAttention(nn.Module):
    def __init__(self, input_size, hidden_size, window_size, num_filters, output_size):
        super(RNNWithTemporalPatternAttention, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.attention = TemporalPatternAttention(hidden_size, window_size, num_filters)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.shape

        # RNN
        rnn_out, (ht, _) = self.rnn(x)  # [batch_size, seq_len, hidden_size], ht: [1, batch_size, hidden_size]
        ht = ht.squeeze(0)  # [batch_size, hidden_size]

        # Temporal Pattern Attention
        attn_out = self.attention(rnn_out[:, -window_size:], ht)  # [batch_size, hidden_size]

        # Final Output
        out = self.fc(attn_out)  # [batch_size, output_size]
        return out

# Model initialization
num_filters = 50  # Example number of filters
model = RNNWithTemporalPatternAttention(input_size=10, hidden_size=50, window_size=window_size, num_filters=num_filters, output_size=1)



class Length(nn.Module):
    """Computes length of vectors in PyTorch."""
    def forward(self, inputs):
        return torch.sqrt(torch.sum(torch.square(inputs), dim=-1))

class Mask(nn.Module):
    """Implements the mask layer in PyTorch."""
    def forward(self, inputs):
        if isinstance(inputs, list) and len(inputs) == 2:
            inputs, mask = inputs
        else:
            x = torch.sqrt(torch.sum(torch.square(inputs), dim=-1))
            mask = F.one_hot(torch.argmax(x, dim=1), num_classes=x.size(1))

        masked = inputs * mask.unsqueeze(-1)
        return masked.view(masked.size(0), -1)

def squash(vectors, axis=-1):
    """Squashing function in PyTorch."""
    s_squared_norm = torch.sum(torch.square(vectors), axis, keepdim=True)
    scale = s_squared_norm / (1 + s_squared_norm) / torch.sqrt(s_squared_norm + 1e-7)
    return scale * vectors

class CapsuleLayer(nn.Module):
    """Capsule Layer implemented in PyTorch."""
    def __init__(self, num_capsule, dim_capsule, num_routing=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing
        self.W = nn.Parameter(torch.randn(num_capsule, dim_capsule, dim_capsule))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        inputs = inputs.unsqueeze(1).repeat(1, self.num_capsule, 1, 1)
        inputs_hat = torch.einsum('bij,ijk->bik', [inputs, self.W])

        b = torch.zeros(batch_size, self.num_capsule, inputs.size(2)).to(inputs.device)

        for i in range(self.num_routing):
            c = F.softmax(b, dim=1)
            outputs = squash(torch.einsum('bij,bjk->bik', [c, inputs_hat]))
            if i < self.num_routing - 1:
                b = torch.einsum('bij,bjk->bik', [outputs, inputs_hat])

        return outputs

class PrimaryCap(nn.Module):
    """Primary capsule implemented in PyTorch."""
    def __init__(self, dim_capsule, n_channels, kernel_size, stride, padding):
        super(PrimaryCap, self).__init__()
        self.conv = nn.Conv1d(in_channels=dim_capsule, out_channels=dim_capsule * n_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        output = self.conv(x)
        output = output.view(x.size(0), -1, self.dim_capsule)
        return squash(output)


import torch.optim as optim

def train_model(model, train_data, val_data, learning_rate=0.001, batch_size=100, num_epochs=10, plot_every=50, plot=True):
    # Check for device availability (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # DataLoader setup for train and validation data
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists for tracking progress
    iters, train_loss, val_loss = [], [], []
    iter_count = 0
    best_val_loss = float("inf")

    try:
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_count += 1
                if iter_count % plot_every == 0:
                    # Track training loss
                    iters.append(iter_count)
                    train_loss.append(loss.item())
                    print(f"Iteration {iter_count}, Training Loss: {loss.item()}")

                    # Evaluate on validation data
                    model.eval()  # Set model to evaluation mode
                    with torch.no_grad():
                        total_val_loss = 0
                        for val_inputs, val_targets in val_loader:
                            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                            val_outputs = model(val_inputs)
                            val_loss_value = criterion(val_outputs, val_targets)
                            total_val_loss += val_loss_value.item()
                        avg_val_loss = total_val_loss / len(val_loader)
                        val_loss.append(avg_val_loss)
                        print(f"Iteration {iter_count}, Validation Loss: {avg_val_loss}")

                        # Save the model if it has the best validation loss so far
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            torch.save(model.state_dict(), "best_model.pth")
                            print("Saved Best Model")

    finally:
        if plot:
            # Plot training and validation loss
            plt.figure()
            plt.plot(iters, train_loss, label='Training Loss')
            plt.plot(iters, val_loss, label='Validation Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Loss over Iterations')
            plt.legend()
            plt.show()

train_model(model, train_dataset, val_dataset, learning_rate=0.002, batch_size=3, num_epochs=5, plot_every=100, plot=True)
