import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np



def split_dataset(data, labels, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Train, validation, and test ratios must sum to 1."

    total_samples = len(data)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    train_data, train_labels = data[:train_end], labels[:train_end]
    val_data, val_labels = data[train_end:val_end], labels[train_end:val_end]
    test_data, test_labels = data[val_end:], labels[val_end:]

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def create_dataloaders(data, labels, batch_size=32):
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = split_dataset(data, labels)

    train_dataset = TimeSeriesDataset(train_data, train_labels)
    val_dataset = TimeSeriesDataset(val_data, val_labels)
    test_dataset = TimeSeriesDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# 生成输入序列和标签（针对 CNN + LSTM）
def generate_sequences_sin(processed_data, T, step=2):
    total_time_slices = len(processed_data)
    input_sequences = []
    labels = []
    for i in range(0, total_time_slices - T, step):  # 使用步长来控制时间间隔，减少采样频率
        sequence = processed_data[i:i + T:step]  # 每隔 `step` 个时间片取样
        label = processed_data[i + T]  # 第 T + 1 个时间片的数据作为标签
        input_sequences.append(sequence)
        labels.append(label)
    input_sequences = np.array(input_sequences)  # 形状为 (num_sequences, T, M, N, k)
    labels = np.array(labels)  # 形状为 (num_sequences, M, N, k)
    return input_sequences, labels


# Function to generate input sequences and labels
def generate_sequences(processed_data, T, F_future=None, step=1):
    total_time_slices = len(processed_data)
    input_sequences = []
    labels = []

    if F_future is None:
        # Generate sequences for single future prediction (e.g., for LSTM)
        for i in range(0, total_time_slices - T, step):
            sequence = processed_data[i : i + T:step]  # 每隔 `step` 个时间片取样
            label = processed_data[i + T]  # 第 T + 1 个时间片的数据作为标签
            input_sequences.append(sequence)
            labels.append(label)
    else:
        # Generate sequences for multiple future predictions (e.g., for Transformer)
        for i in range(0, total_time_slices - T - F_future + 1, step):
            sequence = processed_data[i : i + T:step]  # Take every `step` element to reduce frequency
            label = processed_data[i + T : i + T + F_future:step]  # Take every `step` element for labels as well
            input_sequences.append(sequence)
            labels.append(label)

    input_sequences = np.array(input_sequences)  # Shape: (num_sequences, T, M, N, k)
    labels = np.array(labels)  # Shape: (num_sequences, M, N, k) or (num_sequences, F_future, M, N, k)
    return input_sequences, labels


# data loading, generating sequences, and splitting dataset
def preprocess_and_split_data(processed_data, T, F_future=None, step=2, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
                              batch_size=32):
    input_sequences, labels = generate_sequences(processed_data, T, F_future, step)

    ((train_data, train_labels),
     (val_data, val_labels),
     (test_data, test_labels)) = split_dataset(input_sequences, labels, train_ratio, val_ratio, test_ratio)

    return create_dataloaders(train_data, train_labels, val_data, val_labels, test_data, test_labels, batch_size)


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]