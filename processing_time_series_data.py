import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import os


time_series_files = os.listdir("/content/drive/MyDrive/Github_projects/Double_pendulum/dataset/timeseries")

# Load all 'coords' tensors into a list
coords_tensors = []
for file_name in time_series_files:
    file_path = os.path.join("/content/drive/MyDrive/Github_projects/Double_pendulum/dataset/timeseries", file_name)
    # Load the 'coords' array and convert it to a PyTorch tensor
    tensor = torch.from_numpy(np.load(file_path)['coords'])
    coords_tensors.append(tensor)
coords_tensors = torch.stack(coords_tensors)


number_of_validation_data, number_of_testing_data = int(0.1 * coords_tensors.shape[0]), int(0.1 * coords_tensors.shape[0])
number_of_training_data = int(coords_tensors.shape[0] - number_of_validation_data - number_of_testing_data)

testing_dataset, validation_dataset, training_dataset = random_split(coords_tensors, [number_of_testing_data, number_of_validation_data, number_of_training_data],torch.Generator().manual_seed(42))