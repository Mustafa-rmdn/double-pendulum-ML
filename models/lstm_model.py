import torch
import torch.nn
import torch.nn.functional as F

import lightning as L
from torchmetrics.regression import MeanSquaredError, R2Score



class LSTM_fc_architecture(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers_for_lstm, num_output):
      super().__init__()


      self.input_size = input_size
      self.hidden_size = hidden_size
      self.num_layers = num_layers_for_lstm
      self.num_output = num_output

      self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout= 0.5, batch_first= True)  # lstm expect input as (batch, row, column= Batch, sequence_length(time), features)
      self.fc1 = torch.nn.Linear(in_features = hidden_size, out_features = 128)
      self.fc2 = torch.nn.Linear(in_features= 128, out_features= 64)
      self.fc3 = torch.nn.Linear(in_features= 64, out_features= self.num_output)
      self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self,x):
      output_lstm, (h_n, c_n) = self.lstm(x)      # output_lstm (B, T, Hidden)
      x = self.fc1(output_lstm[:, -1, :])         # last time_step (B,Hidden)
      x = F.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)
      x = F.relu(x)
      x = self.dropout(x)
      x = self.fc3(x)
      return x


# Lighting Module 


class LSTM_fc_lighting(L.LightningModule):
  def __init__(self,learning_rate):
    super().__init__()

    self.save_hyperparameters()
    self.learning_rate = learning_rate

    self.model = LSTM_fc_architecture(input_size= 4, hidden_size= 256, num_layers_for_lstm= 2, num_output= 5)

    self.train_mse = MeanSquaredError()
    self.train_r2 = R2Score()

    self.valid_mse = MeanSquaredError()
    self.valid_r2 = R2Score()

    self.test_mse = MeanSquaredError()
    self.test_r2 = R2Score()

    self.counts = {"train": 0, "validation": 0, "test": 0}




  def forward(self, coords_tensors):

    output = self.model(coords_tensors)
    return output

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)

    return optimizer


  def common_step(self, batch, batch_idx,stage):
    coords_tensors, ground_truth = batch
    output = self.model(coords_tensors)

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

    loss = self.common_step(batch, batch_idx, "train")


    return loss

  def validation_step(self, batch, batch_idx):

    loss = self.common_step(batch, batch_idx, "validation")

    return loss

  def test_step(self, batch, batch_idx):

    loss = self.common_step(batch, batch_idx, "test")

    return loss


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





