import torch
import torch.nn as nn   
import torch.nn.functional as F
import math
import lightning as L
from torchmetrics.regression import MeanSquaredError, R2Score




class PostionalEncoding(nn.Module):
  def __init__(self,d_model, max_seq_length):
    super().__init__()
    self.d_model = d_model
    self.max_seq_length = max_seq_length

    pe = torch.zeros((self.max_seq_length, self.d_model), dtype= torch.float32)
    position = torch.arange(0,self.max_seq_length, dtype= torch.float32).unsqueeze(1)
    divide_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(self.max_seq_length) / d_model)
        )
    pe[:,0::2]= torch.sin(position * divide_term)
    pe[:,1::2]= torch.cos(position * divide_term)

    pe = pe.unsqueeze(0)
    self.register_buffer("pe", pe)

  def forward(self, x):
    B, T, d = x.shape
    return x + self.pe[:,:T,:]



class Transformer_architecture(nn.Module):
  def __init__(self,d_model, n_head, num_layers):
    super().__init__()
    self.d_model = d_model
    self.n_head = n_head
    self.num_layers = num_layers

    self.projection_512 = nn.Linear(in_features = 4, out_features= self.d_model)
    self.encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model , nhead = self.n_head, batch_first= True)
    self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers= self.num_layers)
    self.positional_encoder = PostionalEncoding(d_model= self.d_model, max_seq_length= 10000)

    self.learnable_layer = nn.Parameter(torch.zeros(1,1,d_model), requires_grad=True)

    self.fc1 = nn.Linear(in_features = self.d_model, out_features= int(self.d_model/2))
    self.fc2 = nn.Linear(in_features = int(self.d_model/2), out_features= int(self.d_model/4))
    self.fc3 = nn.Linear(in_features = int(self.d_model/4), out_features= 5)
    self.dropout = nn.Dropout(p=0.2)

  def forward(self,x):
    x = self.projection_512(x)


    B = x.shape[0]

    learnable_layer = self.learnable_layer.expand(B,-1,-1)

    full_encoder = torch.cat((learnable_layer,x),dim = 1)
    full_encoder = self.positional_encoder(full_encoder)


    encoder_output = self.encoder(full_encoder)

    final_feature = encoder_output[:,0,:]

    x = self.fc1(final_feature)

    x = F.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.dropout(x)
    x = self.fc3(x)
    return x




class Transformer_lighting(L.LightningModule):
  def __init__(self,learning_rate):
    super().__init__()

    self.save_hyperparameters()
    self.learning_rate = learning_rate

    self.model = Transformer_architecture(d_model= 512, n_head= 8, num_layers= 5)

    self.train_mse = MeanSquaredError()
    self.train_r2 = R2Score()

    self.valid_mse = MeanSquaredError()
    self.valid_r2 = R2Score()

    self.test_mse = MeanSquaredError()
    self.test_r2 = R2Score()

    self.counts = {"train": 0, "validation": 0, "test": 0}



  def forward(self, coords_tensors):
    # coords_tensors = coords_tensors.to_device(self.device)
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



