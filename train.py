import torch
import argparse
import re
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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
    parser.add_argument("--ckpt-path", type=str, default=None, 
                        help="Path to the checkpoint file (.ckpt)")
    parser.add_argument("--resume", action="store_true", 
                        help="If true, resume training including optimizer states from ckpt-path")

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
            print(" - GPU was requested (IS_GPU=True) but CUDA is not available. Falling back to CPU.")
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

    # model = get_model("HitMap", args)
    model = get_model("resnet18", args)
    model.to(device) 

    # ==== Checkpoint ====
    existing_version = None
    if args.resume and args.ckpt_path:
        match = re.search(r"version_(\d+)", args.ckpt_path)
        if match:
            existing_version = int(match.group(1))
            print(f" >>> Resuming log in existing version: {existing_version}")
            
            log_file = os.path.join(args.log_path, args.log_name, f"version_{existing_version}", "metrics.csv")
            
            if os.path.exists(log_file):
                print(f" >>> Found existing metrics.csv. Backup created at {log_file}.bak")
                os.system(f"cp {log_file} {log_file}.bak")
            else:
                print(f" [!] Warning: {log_file} not found even though resume is active.")

    csv_logger = CSVLogger(save_dir=args.log_path,
                           name=args.log_name,
                           version=existing_version)
    checkpoint_callback = ModelCheckpoint(filename="best_model-{epoch:02d}-{val_loss:.3f}",
                                          monitor="val_loss",
                                          mode="min",
                                          save_top_k=1,       
                                          save_last=True)
    
    # ==== Start Training ====
    trainer = pl.Trainer(max_epochs=args.epochs,
                         accelerator=accelerator,
                         strategy=strategy,
                         devices=num_devices,
                         logger=csv_logger,
                         callbacks=[checkpoint_callback],
                         log_every_n_steps=10,
                         enable_progress_bar=True)
    
    if args.resume and args.ckpt_path:
        print(f" >>> Resuming training from checkpoint: {args.ckpt_path}")
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
    elif args.ckpt_path:
        print(f" >>> Loading weights only from: {args.ckpt_path} and starting fresh")
        model = HitMapLightningModel.load_from_checkpoint(args.ckpt_path)
        trainer.fit(model, datamodule=dm)
    else:
        print(" >>> Starting training from scratch")
        trainer.fit(model, datamodule=dm)    
