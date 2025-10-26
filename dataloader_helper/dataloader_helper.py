
class Double_pendulum_dataset_coordinates:
  def __init__(self, coords_tensors,ground_truth_parameters):

    self.coords_tensors = coords_tensors
    self.ground_truth_parameters = ground_truth_parameters

  def __len__(self):
    return self.coords_tensors.shape[0]

  def __getitem__(self, idx):

    coords = self.coords_tensors[idx]
    ground_truth = self.ground_truth_parameters[idx]

    return coords, ground_truth
  
  
  
class Double_pendulum_dataset_images:
  def __init__(self, image_tensor,ground_truth_parameters):

    self.image_tensor = image_tensor
    self.ground_truth_parameters = ground_truth_parameters

  def __len__(self):
    return self.image_tensor.shape[0]

  def __getitem__(self, idx):

    image = self.image_tensor[idx]
    ground_truth = self.ground_truth_parameters[idx]

    return image, ground_truth