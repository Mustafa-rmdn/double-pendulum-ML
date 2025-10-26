import os
import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image


def load_data_coordinates():
    # List all files in the 'dataset/timeseries' directory
    time_series_files = os.listdir("dataset/timeseries")

    # Load all 'coords' tensors into a list
    coords_tensors = []
    for file_name in time_series_files:
        file_path = os.path.join("dataset/timeseries", file_name)
        # Load the 'coords' array and convert it to a PyTorch tensor
        tensor = torch.from_numpy(np.load(file_path)['coords']).float()
        coords_tensors.append(tensor)
    coords_tensors = torch.stack(coords_tensors)

    gt_parameters = pd.read_csv("dataset/indexes/timeseries_index.csv")
    gt_parameters_tensor = torch.tensor(gt_parameters.iloc[:, 2:-1].values, dtype=torch.float32)
    return coords_tensors, gt_parameters_tensor

def load_data_images():

    image_directory = "dataset/images"

    image_files = os.listdir(image_directory)
    image_files = np.sort(image_files)
    image_paths = [os.path.join(image_directory, f) for f in image_files]

    image_tensor = torch.stack([read_image(im).float() for im in image_paths])  # [N, C, H, W]


    gt_parameters = pd.read_csv("dataset/indexes/image_index.csv")
    gt_parameters_tensor = torch.tensor(gt_parameters.iloc[:,3:].values, dtype=torch.float32)

    return image_tensor, gt_parameters_tensor