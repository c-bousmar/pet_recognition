import pandas as pd
import matplotlib.pyplot as plt

# Read the txt file into a pandas DataFrame
# Assuming the data in the text file is formatted with commas or spaces
file_path = 'temp_res.txt'  # Replace with the actual path to your text file

# Read the file (assuming each line follows the format 'Epoch x/200, Train Loss: value, Validation Loss: value')
# You can adjust the separator based on how your data is structured (e.g., ',' or whitespace)

data = pd.read_csv(file_path, sep=',', header=None, names=["Epoch", "Train Loss", "Validation Loss"])

# Clean the data by extracting numbers from the strings in "Train Loss" and "Validation Loss"
data["Train Loss"] = data["Train Loss"].str.split(": ").str[1].astype(float)
data["Validation Loss"] = data["Validation Loss"].str.split(": ").str[1].astype(float)

# Apply rolling mean to smooth the curves
window_size = 3  # Adjust window size to change the level of smoothing
data["Train Loss Smooth"] = data["Train Loss"].rolling(window=window_size, min_periods=1).mean()
data["Validation Loss Smooth"] = data["Validation Loss"].rolling(window=window_size, min_periods=1).mean()

# Calculate rolling standard deviation for error approximation
data["Train Loss Std"] = data["Train Loss"].rolling(window=window_size, min_periods=1).std()
data["Validation Loss Std"] = data["Validation Loss"].rolling(window=window_size, min_periods=1).std()

# Plotting the smoothed data with error bounds
plt.figure(figsize=(10, 6))

# Train Loss with Error Bounds
plt.plot(data.index, data["Train Loss Smooth"], label='Train Loss')
plt.fill_between(data.index, data["Train Loss Smooth"] - data["Train Loss Std"], data["Train Loss Smooth"] + data["Train Loss Std"],
                 color='blue', alpha=0.2)

# Validation Loss with Error Bounds
plt.plot(data.index, data["Validation Loss Smooth"], label='Validation Loss')
plt.fill_between(data.index, data["Validation Loss Smooth"] - data["Validation Loss Std"], data["Validation Loss Smooth"] + data["Validation Loss Std"],
                 color='orange', alpha=0.2)

# Adding a vertical line at a specific x-value (e.g., x=50)
specific_x_value = 47  # Replace this value with your desired x-value
plt.axvline(x=specific_x_value, color='red', linestyle='--', label=f'Selected epoch: {specific_x_value}')

# Labeling the axes and the plot
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss Curves (w/ Error Bounds)')

# Show the legend
plt.legend()

# Set fewer ticks on the x-axis (e.g., every 10th epoch, based on row index)
xticks = data.index[::5]  # data.index starts from 0

# Ensure the last epoch (100 or 200) is included
last_epoch = data.index[-1] + 1
xticks = sorted(set(xticks).union({last_epoch}))  # Ensure the last epoch is included
plt.xticks(xticks)  # Set the x-ticks to show only these values

# Show the plot
plt.show()
