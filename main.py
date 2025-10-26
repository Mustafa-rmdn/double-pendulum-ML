import torch
from dataloader_helper.dataloader_helper import Double_pendulum_dataset_coordinates, Double_pendulum_dataset_images
from torch.utils.data import random_split, DataLoader
from load_data import load_data_coordinates, load_data_images
from models.lstm_model import LSTM_fc_lighting
from models.Transformer_model import Transformer_lighting
from models.CNN_model import CNN_lighting
from prediction import show_prediction_and_ground_truth
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import argparse



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["Lstm","Transformer","CNN"], default="Transformer")
    parser.add_argument("--mode", choices=["train","test"], default="test")
    args = parser.parse_args()

    if args.backbone in ["Lstm","Transformer"]:

        coords_tensors, gt_parameters_tensor = load_data_coordinates()
        full_trajectories = Double_pendulum_dataset_coordinates(coords_tensors = coords_tensors, ground_truth_parameters= gt_parameters_tensor)
    
    elif args.backbone == "CNN":

        image_tensor, gt_parameters_tensor = load_data_images()
        full_trajectories = Double_pendulum_dataset_images(image_tensor = image_tensor, ground_truth_parameters= gt_parameters_tensor)


    number_of_validation_data, number_of_testing_data = int(0.1 * len(full_trajectories)), int(0.1 * len(full_trajectories))
    number_of_training_data = int(len(full_trajectories) - number_of_validation_data - number_of_testing_data)

    testing_dataset, validation_dataset, training_dataset = random_split(full_trajectories, [number_of_testing_data, number_of_validation_data, number_of_training_data],torch.Generator().manual_seed(42))

    training_dataloader = DataLoader(training_dataset,
                                    batch_size= 10,
                                    shuffle= True,
                                    )

    validation_dataloader = DataLoader(validation_dataset,
                                    batch_size= 10,
                                    shuffle= False,
                                    )

    testing_dataloader = DataLoader(testing_dataset,
                                    batch_size = 10,
                                    shuffle= False,               
                                    )

    if args.backbone == "Lstm":
    # Instantiating the model
        lstm_fc_model = LSTM_fc_lighting(learning_rate= 1e-5)

        csv_logger = CSVLogger(save_dir="outputs/Lstm")
        tensorboard_logger = TensorBoardLogger(save_dir = "outputs/Lstm")
        

        ckpt = ModelCheckpoint(
            monitor="validation_mse",
            mode="min",
            save_top_k=1,
            filename="LSTM-{epoch:02d}-{validation_mse:.4f}"
        )

        trainer_for_lstm_fc = L.Trainer(max_epochs= 3,  #200
                            # fast_dev_run=True,
                            log_every_n_steps = 1,
                            gradient_clip_val=1.0,
                            logger = [csv_logger,tensorboard_logger],
                            callbacks=[ckpt, RichProgressBar()],
                            default_root_dir="outputs/Lstm")

        if args.mode == "train":

            trainer_for_lstm_fc.fit(model = lstm_fc_model, train_dataloaders= training_dataloader, val_dataloaders= validation_dataloader)
        elif args.mode == "test":

            prediction = show_prediction_and_ground_truth(backbone="Lstm", dataloader = testing_dataloader)
            
            print(prediction)

    elif args.backbone == "Transformer":


        transformer_model = Transformer_lighting(learning_rate= 3e-5)

        tensorboard_logger = TensorBoardLogger(save_dir = "outputs/Transformer")
        csv_logger = CSVLogger(save_dir="outputs/Transformer")


        ckpt = ModelCheckpoint(
            monitor="validation_mse",
            mode="min",
            save_top_k=1,
            filename="Attention-{epoch:02d}-{validation_mse:.4f}"
        )


        trainer_for_transformer = L.Trainer(max_epochs= 3,
                            # fast_dev_run=True,
                            log_every_n_steps = 1,
                            logger = [csv_logger, tensorboard_logger],
                            accelerator="auto",
                            gradient_clip_val=1.0,
                            callbacks=[ckpt, RichProgressBar()],
                            default_root_dir="outputs/Transformer")
        
        if args.mode == "train":

            trainer_for_transformer.fit(model = transformer_model, train_dataloaders= training_dataloader, val_dataloaders= validation_dataloader)

        elif args.mode == "test":

            prediction = show_prediction_and_ground_truth(backbone="Transformer", dataloader = testing_dataloader)
            print(prediction)

    elif args.backbone == "CNN":

        CNN_model = CNN_lighting(learning_rate= 3e-5)

        tensorboard_logger = TensorBoardLogger(save_dir = "outputs/CNN")
        csv_logger = CSVLogger(save_dir="outputs/CNN")

        ckpt = ModelCheckpoint(
            monitor="validation_mse",
            mode="min",
            save_top_k=1,
            filename="CNN-{epoch:02d}-{validation_mse:.4f}"
        )

        trainer_for_CNN = L.Trainer(max_epochs= 3,
                            # fast_dev_run=True,
                            log_every_n_steps = 1,
                            logger = [csv_logger,tensorboard_logger],
                            accelerator="auto",
                            gradient_clip_val=1.0,
                            callbacks=[ckpt, RichProgressBar()],
                            default_root_dir="outputs/CNN")
        
        if args.mode == "train":

            trainer_for_CNN.fit(model = CNN_model, train_dataloaders= training_dataloader, val_dataloaders= validation_dataloader)

        elif args.mode == "test":

            prediction = show_prediction_and_ground_truth(backbone="CNN", dataloader = testing_dataloader)

            print(prediction)
    

if __name__ == "__main__":
    main()