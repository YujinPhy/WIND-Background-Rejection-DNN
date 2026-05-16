import argparse
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import pytorch_lightning as pl

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

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

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_probs = []
    all_y = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)
        probs = torch.softmax(logits, dim=1)[:, 1]

        total_loss += float(loss.item()) * len(yb)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == yb).sum().item())
        total += len(yb)

        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())

    y_true = np.concatenate(all_y)
    prob_sig = np.concatenate(all_probs)
    return total_loss / total, correct / total, y_true, prob_sig

def working_point_at_bkg_residual(y_true, prob_internal, target_bkg_residual=0.03):
    """
    Choose a threshold using the background score distribution.

    The threshold is selected so that approximately target_bkg_residual
    of true 16N events survive as internal-like.

    Returns:
      threshold, internal_efficiency, actual_16N_residual
    """
    y_true = np.asarray(y_true)
    prob_internal = np.asarray(prob_internal)

    bkg_scores = prob_internal[y_true == 0]
    sig_scores = prob_internal[y_true == 1]

    if len(bkg_scores) == 0 or len(sig_scores) == 0:
        return np.nan, np.nan, np.nan

    target_bkg_residual = float(target_bkg_residual)
    target_bkg_residual = min(max(target_bkg_residual, 0.0), 1.0)

    # High threshold for high purity.
    # Example: target_bkg_residual=0.03 -> 97th percentile of 16N scores.
    threshold = float(np.quantile(bkg_scores, 1.0 - target_bkg_residual))

    pred_internal = prob_internal >= threshold
    internal_eff = float(np.mean(pred_internal[y_true == 1]))
    bkg_residual = float(np.mean(pred_internal[y_true == 0]))

    return threshold, internal_eff, bkg_residual

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
    model = get_model("HitMap", args).to(device)
    # model = get_model("resnet18", args).to(device)

    ckpt_search_path = os.path.join(args.log_path, args.log_name, "checkpoints", "best_model*.ckpt")
    ckpt = torch.load(glob.glob(ckpt_search_path)[0], map_location=device)
    model.load_state_dict(ckpt["state_dict"])


    # ==== Start Evaluation (No Training and Logger) ====
    # Learning Curve
    metrics_path = os.path.join(args.log_path, args.log_name)
    # loss_acc_analysis(metrics_path=metrics_path,
    #                   output_path=metrics_path,
    #                   png_title=f"loss_acc_curve.png")

    # ROC Curve
    criterion = nn.CrossEntropyLoss()
    target_bkg_residual = 0.03

    val_loss, val_acc, y_val, p_val = evaluate(model, val_loader, criterion, device)
    wp_threshold, val_eff_at_target, val_bkg_residual = working_point_at_bkg_residual(
        y_val, p_val, target_bkg_residual=target_bkg_residual
    )

    test_loss, test_acc, y_test, p_test = evaluate(model, test_loader, criterion, device)
    test_auc = roc_auc_score(y_test, p_test)

    y_pred_05 = (p_test >= 0.5).astype(np.int64)
    y_pred_wp = (p_test >= wp_threshold).astype(np.int64)

    test_internal_eff_wp = float(np.mean(y_pred_wp[y_test == 1]))
    test_bkg_residual_wp = float(np.mean(y_pred_wp[y_test == 0]))
    test_improvement_wp = np.inf if test_bkg_residual_wp == 0 else test_internal_eff_wp / test_bkg_residual_wp

    print("\n[TEST RESULT: threshold 0.5]")
    print(f"loss = {test_loss:.4f}")
    print(f"accuracy = {accuracy_score(y_test, y_pred_05):.4f}")
    print(f"AUC = {test_auc:.4f}")
    print("Confusion matrix, rows=true, cols=pred:")
    print(confusion_matrix(y_test, y_pred_05))

    print(f"\n[HIGH-PURITY WORKING POINT]")
    print(f"Threshold chosen on validation set for 16N residual ~= {target_bkg_residual:.3f}")
    print(f"threshold = {wp_threshold:.6f}")
    print(f"validation internal efficiency = {val_eff_at_target:.4f}")
    print(f"validation 16N residual       = {val_bkg_residual:.4f}")
    print(f"test internal efficiency       = {test_internal_eff_wp:.4f}")
    print(f"test 16N residual              = {test_bkg_residual_wp:.4f}")
    print(f"test S/B improvement factor    = {test_improvement_wp:.4f}")

    print("\n[TEST classification report at high-purity working point]")
    print(
        classification_report(
            y_test,
            y_pred_wp,
            target_names=["16N background", "internal ES signal"],
            digits=4,
        )
    )

    fpr, tpr, thresholds = roc_curve(y_test, p_test)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter([test_bkg_residual_wp], [test_internal_eff_wp], label="working point")
    plt.xlabel("False positive rate: 16N misidentified as internal")
    plt.ylabel("True positive rate: internal efficiency")
    plt.title("ES internal vs 16N CNN ROC")
    plt.legend()
    plt.tight_layout()
    outdir = os.path.join(metrics_path,"roc_curve.png")
    plt.savefig(outdir, dpi=160)
    plt.close()

