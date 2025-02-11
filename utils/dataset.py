import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class VehicleDataset(Dataset):
    def __init__(self, data_dir, grid_size=(32, 32), max_seq_length=100):
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.max_seq_length = max_seq_length
        self.data_files = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        data = pd.read_csv(file_path)

        # Grid representation
        grid = np.zeros((*self.grid_size, 4))  # 4 channels: x, y, velocity, acceleration
        for _, row in data.iterrows():
            x_idx = int(row['Local_X'] / self.grid_size[0])
            y_idx = int(row['Local_Y'] / self.grid_size[1])
            grid[x_idx, y_idx, :] = [row['Local_X'], row['Local_Y'], row['v_Vel'], row['v_Acc']]

        # Normalize grid
        grid = torch.tensor(grid, dtype=torch.float32).permute(2, 0, 1)

        # Load R*-tree sequence label
        label_path = file_path.replace('.csv', '_label.txt')  # Assuming labels are stored in a similar format
        with open(label_path, 'r') as f:
            label = f.read().strip().split(',')
        label_tensor = torch.tensor([float(x) for x in label], dtype=torch.float32)
        label_tensor = label_tensor[:self.max_seq_length]  # Truncate to max length

        return grid, label_tensor

# Example Usage
if __name__ == "__main__":
    dataset = VehicleDataset("../dataprocess/data/processed/splitByGlobalTime")
    grid, label = dataset[0]
    print("Grid shape:", grid.shape)
    print("Label shape:", label.shape)
