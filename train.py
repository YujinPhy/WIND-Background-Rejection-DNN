import torch
import argparse
import re
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve, auc

from datasets.DataModule import WIND_DataModule
from models.ResNet18 import resnet18
from models.cnn_reference import cnn_reference
from models.Sparse_ResNet18 import sparse_resnet18
from models.HitMapCNN import HitMapLightningModel

from analysis.check_training import *

def parse_args():
    parser = argparse.ArgumentParser(description="WIND Background Rejection Training Script")

    # Data paths 
    parser.add_argument("--es-path", type=str, required=True, help="Path to ES signal dataset")
    parser.add_argument("--n16-path", type=str, required=True, help="Path to 16N background dataset")

    # Hardware setup
    parser.add_argument("--gpu", action="store_true", help="Use NVIDIA GPU (CUDA) if available")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of workers for data loading")
    
    # Log & Checkpoint & Resume
    parser.add_argument("--log-path", type=str, default=None, 
                        help="Path to the log files")
    parser.add_argument("--log-name", type=str, default=None, 
                        help="Name to the sub log files")

    # Training hyperparamters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Ratio of test data (0.0 ~ 1.0)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Ratio of validation data (0.0 ~ 1.0)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Whether to shuffle the training data")
    return parser.parse_args()

def get_model(model_name, args):   
    if model_name == "HitMap":
        model_kwargs = {
            "args": args,
            "image_h": 91,
            "image_w": 142,
            "n_classes": 2,
            "lr": args.lr
        }
        return HitMapLightningModel(**model_kwargs)
    elif model_name == "resnet18":
        model_kwargs = {
            "args": args,
            "lr": args.lr,
        }
        return resnet18(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

if __name__ == "__main__":
    # ==== Setup ====
    args = parse_args()

    print(f" #### [ Seed ] #### ")
    pl.seed_everything(args.seed, workers=True)

    print(" #### [ Hardware Detection ] #### ")
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        accelerator = "gpu"
        num_devices = 1
        strategy = "auto"
        pin_memory = True  
        print(f" - NVIDIA GPU Detected (CUDA): {torch.cuda.get_device_name(0)}")
        print(f" - Device Count: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
        accelerator = "cpu"
        num_devices = 1
        strategy = "auto"
        pin_memory = True
        if args.gpu:
            print(" - GPU was requested, but CUDA is not available. Falling back to CPU.")
        else:
            print(" - Device: CPU")


    print(" #### [ Loading Data ] #### ")
    dm = WIND_DataModule(es_h5=args.es_path,
                         n16_h5=args.n16_path,
                         test_ratio=args.test_ratio,
                         val_ratio=args.val_ratio,
                         seed=args.seed,
                         batch_size=args.batch_size,
                         shuffle=args.shuffle,
                         pin_memory=pin_memory,
                         num_workers=args.num_workers)

    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()

    model = get_model("HitMap", args)
    # model = get_model("resnet18", args)
    model.to(device) 

    # ==== Checkpoint ====
    csv_logger = CSVLogger(save_dir=args.log_path,
                           name=args.log_name)
    checkpoint_callback = ModelCheckpoint(filename="best_model",
                                          monitor="val_loss",
                                          mode="min",
                                          save_top_k=1)
    
    # ==== Start Training ====
    trainer = pl.Trainer(max_epochs=args.epochs,
                         accelerator=accelerator,
                         strategy=strategy,
                         devices=num_devices,
                         logger=csv_logger,
                         callbacks=[checkpoint_callback],
                         log_every_n_steps=10,
                         enable_progress_bar=True)
    
    trainer.fit(model, datamodule=dm)

    # ==== Test ====
    print("\n #### [ Evaluation ] #### ")
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        print(" [!] No best model found, using current weights for testing.")
        best_model_path = None
    else:
        print(f" >>> Best model found at: {best_model_path}")

    test_results = trainer.test(model, datamodule=dm, ckpt_path=best_model_path)

    # ==== Analysis ====
    # Learning Curve
    # loss_acc_analysis(metrics_path=csv_logger.log_dir,
    #                   output_path=csv_logger.log_dir,
    #                   png_title=f"loss_acc_curve.png")


    # ROC Curve
    model.eval() # Set to evaluation mode
    all_probs = []
    all_targets = []

    test_loader = dm.test_dataloader()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            logits = model(x.to(device))
            # Get the probability for the 'Signal' class (index 1)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.numpy())

            plot_name, _ = os.path.splitext(args.log_name) if args.log_name else ("Result", "")
    
    save_dir = os.path.join(args.log_path, args.log_name, f"version_{csv_logger.version}")
    os.makedirs(save_dir, exist_ok=True)
    roc_save_path = os.path.join(save_dir, f"roc_curve_{plot_name}.png")

    # plot_roc_curve(all_targets, all_probs, roc_save_path, plot_name)