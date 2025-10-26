import torch
import pandas as pd
from models.lstm_model import LSTM_fc_lighting
from models.Transformer_model import Transformer_lighting
from models.CNN_model import CNN_lighting

def show_prediction_and_ground_truth(backbone:str, dataloader, n_limit=5):
    """
    Collect predictions and ground truths for all samples in a dataloader.

    model: trained Lightning or torch model
    dataloader: DataLoader yielding (x, y)
    param_names: list of parameter names ["L1","L2","m1","m2","k"]
    n_limit: optional cap on number of samples to include
    """
    if backbone == "Lstm":

        model = LSTM_fc_lighting.load_from_checkpoint(
            "trained_models_weights/LSTM_model.ckpt",
            learning_rate= 1e-5
        )
    elif backbone == "Transformer":

        model = Transformer_lighting.load_from_checkpoint(
            "trained_models_weights/Attention_model.ckpt",
            learning_rate= 3e-5
        )
    elif backbone == "CNN":


        model = CNN_lighting.load_from_checkpoint(
            "trained_models_weights/CNN_model.ckpt",
            learning_rate= 3e-5
        )

    model.eval()
    param_names = ["L1","L2","m1","m2","k"]

    preds, gts = [], []
    total = 0

    for x, y in dataloader:
        x = x.to(model.device)
        with torch.no_grad():
            y_hat = model(x)
        preds.append(y_hat.cpu())
        gts.append(y.cpu())

        total += len(y)
        if n_limit and total >= n_limit:
        # if total >= n_limit:
            break

    preds = torch.cat(preds, dim=0).numpy()
    gts = torch.cat(gts, dim=0).numpy()

    df_pred = pd.DataFrame(preds, columns=[f"pred_{n}" for n in param_names])
    df_gt = pd.DataFrame(gts, columns=[f"gt_{n}" for n in param_names])

    df = pd.concat([df_gt, df_pred], axis=1)
    df.to_csv(f"outputs/predictions_{backbone}.csv", index=False)
    return df


    
