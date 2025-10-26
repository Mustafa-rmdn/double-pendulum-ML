import torch
import torch.nn as nn   
import torch.nn.functional as F

import lightning as L
from torchmetrics.regression import MeanSquaredError, R2Score


class CNN_architecture(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv_layer_1 = nn.Conv2d(in_channels= 4, out_channels= 32, kernel_size=3, stride=1, padding=1)
    self.conv_layer_2 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size=3, stride=1, padding=1)
    self.max_pool = nn.MaxPool2d(kernel_size= 2, stride=2)
    self.relu = nn.ReLU()

    self.adapt_pool = nn.AdaptiveAvgPool2d((64, 64))
    self.flatten = nn.Flatten()


    self.fc = nn.Sequential(
        nn.Linear(in_features= 64*64*64, out_features= 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(in_features= 512, out_features= 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(in_features=256, out_features= 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(in_features= 128, out_features= 5)
    )

  def forward(self,x):
    x = self.conv_layer_1(x)          # (B,C,H,W) = (B,32,256,256)
    x = self.max_pool(x)            # (B,32,128,128)
    x = self.relu(x)
    x = self.conv_layer_2(x)          # (B, 64, 128,128)
    x = self.max_pool(x)            # (B,64,64,64)
    x = self.relu(x)
    x = self.adapt_pool(x)          # (B, 64,64,64)
    x = self.flatten(x)
    x = self.fc(x)

    return x



class CNN_lighting(L.LightningModule):
  def __init__(self,learning_rate):
    super().__init__()

    self.save_hyperparameters()
    self.learning_rate = learning_rate

    self.model = CNN_architecture()

    self.train_mse = MeanSquaredError()
    self.train_r2 = R2Score()

    self.valid_mse = MeanSquaredError()
    self.valid_r2 = R2Score()

    self.test_mse = MeanSquaredError()
    self.test_r2 = R2Score()

    self.counts = {"train": 0, "validation": 0, "test": 0}



  def forward(self, image_tensor):

    output = self.model(image_tensor)
    return output

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)

    return optimizer


  def common_step(self, batch, batch_idx,stage):
    image_tensor, ground_truth = batch

    output = self.model(image_tensor)

    loss = F.mse_loss(output, ground_truth)

    self.log(f"{stage}_loss",loss,on_epoch=True, on_step=False,logger=True)

    if stage == "train":
        self.train_mse.update(output, ground_truth)
        self.train_r2.update(output, ground_truth)
    elif stage == "validation":
        self.valid_mse.update(output, ground_truth)
        self.valid_r2.update(output, ground_truth)
    else:
        self.test_mse.update(output, ground_truth)
        self.test_r2.update(output, ground_truth)

    # increment by batch size
    self.counts[stage] += 1



    return loss



  def training_step(self, batch, batch_idx):
    return self.common_step(batch, batch_idx, "train")

  def validation_step(self, batch, batch_idx):
    return self.common_step(batch, batch_idx, "validation")

  def test_step(self, batch, batch_idx):
    return self.common_step(batch, batch_idx, "test")

  def on_validation_epoch_end(self):

    valid_mse_result = float(self.valid_mse.compute())
    if self.counts["validation"] >= 2:
      valid_r2_result = float(self.valid_r2.compute())
    else:
      valid_r2_result = 0.0

    self.log("validation_mse", valid_mse_result,logger=True, prog_bar=True)
    self.log("validation_r2",  valid_r2_result,logger=True, prog_bar=True)

    self.valid_mse.reset()
    self.valid_r2.reset()


  def on_test_epoch_end(self):

    test_mse_result = float(self.test_mse.compute())
    if self.counts["test"] >= 2:
      test_r2_result = float(self.test_r2.compute())
    else:
      test_r2_result = 0.0


    self.log("test_mse", test_mse_result,logger=True, prog_bar=True)
    self.log("test_r2",  test_r2_result,logger=True, prog_bar=True)

    self.test_mse.reset()
    self.test_r2.reset()

  def on_train_epoch_end(self):

    train_mse_result = float(self.train_mse.compute())
    if self.counts["train"] >= 2:
      train_r2_result = float(self.train_r2.compute())
    else:
      train_r2_result = 0.0

    self.log("train_mse", train_mse_result,logger=True, prog_bar=True)
    self.log("train_r2",  train_r2_result,logger=True, prog_bar=True)


    self.train_mse.reset()
    self.train_r2.reset()




