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

# Plotting the data
plt.plot(data.index, data["Train Loss"], label='Train Loss')
plt.plot(data.index, data["Validation Loss"], label='Validation Loss')

# Labeling the axes and the plot
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss Curves')

# Show the legend
plt.legend()

# Set fewer ticks on the x-axis (e.g., every 10th epoch, based on row index)
xticks = data.index[::10] # data.index starts from 0

# Ensure the last epoch (100 or 200) is included
last_epoch = data.index[-1] + 1
xticks = sorted(set(xticks).union({last_epoch}))  # Ensure the last epoch is included
plt.xticks(xticks)  # Set the x-ticks to show only these values

# Show the plot
plt.show()
