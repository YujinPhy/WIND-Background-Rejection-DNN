import argparse
import os
import glob
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, classification_report, roc_auc_score

from datasets.DataModule import WIND_DataModule
from models.HitMapCNN import HitMapCNN
from models.ResNet18 import ResNet18
from models.DSNB_CNN import DSNB_CNN

from analysis.check_training import *
from analysis.physical_evaluation import *

def parse_args():
    parser = argparse.ArgumentParser(description="WIND Background Rejection Training Script")

    # Data paths 
    parser.add_argument("--es-path", type=str, required=True, help="Path to ES signal dataset")
    parser.add_argument("--n16-path", type=str, required=True, help="Path to 16N background dataset")

    # Input data
    parser.add_argument("--in-ch", type=int, default=2, help="Number of input channels")
    parser.add_argument("--image-h", type=int, default=91, help="Height of input images")
    parser.add_argument("--image-w", type=int, default=142, help="Width of input images")

    # Hardware setup
    parser.add_argument("--gpu", action="store_true", help="Use NVIDIA GPU (CUDA) if available")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of workers for data loading")
    
    # Log & Checkpoint & Resume
    parser.add_argument("--log-path", type=str, default=None, help="Path to the log files")
    parser.add_argument("--log-name", type=str, default=None, help="Name to the sub log files")

    # Training hyperparamters
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Ratio of test data (0.0 ~ 1.0)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Ratio of validation data (0.0 ~ 1.0)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Whether to shuffle the training data")

    # Physcical analysis
    parser.add_argument("--target-bkg-residual", type=float, default=0.03, help="Target background residual for physical evaluation (e.g., 0.03 for 3%)")
                        
    return parser.parse_args()

def get_model(model_name, args):   
    if model_name == "HitMapCNN":
        model_kwargs = {
            "args": args,
            "in_ch": args.in_ch,
            "image_h": args.image_h,
            "image_w": args.image_w,
            "n_classes": 2,
            "lr": args.lr
        }
        return HitMapCNN(**model_kwargs)
    
    elif model_name == "ResNet18":
        model_kwargs = {
            "args": args,
            "in_ch": args.in_ch,
            "n_classes": 2,
            "lr": args.lr,
        }
        return ResNet18(**model_kwargs)
    
    elif model_name == "DSNB_CNN":
        model_kwargs = {
            "args": args,
            "in_ch": args.in_ch,
            "image_h": args.image_h,
            "image_w": args.image_w,
            "n_classes": 2,
            "lr": args.lr
        }
        return DSNB_CNN(**model_kwargs)

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

    dm.setup(stage="test")
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()


    # ==== Get Best model ====
    model = get_model(args.model_name, args).to(device)

    ckpt_search_path = os.path.join(args.log_path, args.log_name, "best_model*.ckpt")
    ckpt = torch.load(glob.glob(ckpt_search_path)[0], map_location=device)
    model.load_state_dict(ckpt["state_dict"])


    # ==== Start Evaluation (No Training and Logger) ====
    # Learning Curve
    logger_dir  = os.path.join(args.log_path, args.log_name)
    loss_acc_analysis(metrics_path=logger_dir,
                      output_path=logger_dir,
                      png_title=f"loss_acc_curve.png")

    # ROC Curve
    criterion = nn.CrossEntropyLoss()
    target_bkg_residual = 0.03
    
    performance_summary(model, val_loader, test_loader, criterion, device, target_bkg_residual, logger_dir )