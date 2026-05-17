import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

def loss_acc_analysis(metrics_path, output_path, png_title):
    # 1. Filename Processing (Extract pure name without extension)
    base_name, _ = os.path.splitext(png_title)
    
    csv_path = os.path.join(metrics_path, "metrics.csv")
    bak_path = os.path.join(metrics_path, "metrics.csv.bak")
    
    # Load data
    dfs = []
    if os.path.exists(bak_path):
        dfs.append(pd.read_csv(bak_path))
        print(f" [INFO] Loaded backup log: {bak_path}")
    if os.path.exists(csv_path):
        dfs.append(pd.read_csv(csv_path))
        print(f" [INFO] Loaded current log: {csv_path}")
    
    if not dfs:
        print(" [ERROR] No log files found for analysis.")
        return

    # Merge and Clean Data
    df_full = pd.concat(dfs, axis=0, ignore_index=True)
    train_df = df_full.dropna(subset=['train_loss']).groupby('epoch').mean().reset_index()
    val_df = df_full.dropna(subset=['val_loss']).groupby('epoch').last().reset_index()

    # 2. Key Metrics Calculation
    # (1) Best Loss Point (Based on Val Loss)
    best_loss_idx = val_df['val_loss'].idxmin()
    best_loss_epoch = val_df.loc[best_loss_idx, 'epoch']
    best_loss_val = val_df.loc[best_loss_idx, 'val_loss']
    # Extract Accuracy at the point of Minimum Loss
    acc_at_best_loss = val_df.loc[best_loss_idx, 'val_acc']

    # (2) Best Accuracy Point (Based on Val Acc)
    best_acc_idx = val_df['val_acc'].idxmax()
    best_acc_epoch = val_df.loc[best_acc_idx, 'epoch']
    best_acc_val = val_df.loc[best_acc_idx, 'val_acc']

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'Learning Curve', fontsize=16, fontweight='bold')

    # --- Left Plot: Loss Analysis ---
    ax1.plot(train_df['epoch'], train_df['train_loss'], label='Train Loss', color='royalblue', alpha=0.5)
    ax1.plot(val_df['epoch'], val_df['val_loss'], label='Val Loss', color='darkorange', marker='o', markersize=4)
    
    label_loss = f'Best Loss: {best_loss_val:.4f} (Ep {int(best_loss_epoch)})'
    ax1.axvline(x=best_loss_epoch, color='red', linestyle='--', alpha=0.8, label=label_loss)
    ax1.scatter(best_loss_epoch, best_loss_val, color='red', s=60, zorder=5)
    
    ax1.set_title('Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Right Plot: Accuracy Analysis ---
    ax2.plot(train_df['epoch'], train_df['train_acc'], label='Train Acc', color='seagreen', alpha=0.5)
    ax2.plot(val_df['epoch'], val_df['val_acc'], label='Val Acc', color='crimson', marker='s', markersize=4)
    
    # (A) Best Accuracy Point (Blue dashed line)
    label_best_acc = f'Best Acc: {best_acc_val:.4f} (Ep {int(best_acc_epoch)})'
    ax2.axvline(x=best_acc_epoch, color='blue', linestyle='--', alpha=0.6, label=label_best_acc)
    ax2.scatter(best_acc_epoch, best_acc_val, color='blue', s=60, zorder=5)

    # (B) Accuracy at Best Loss (Purple dotted line)
    label_at_loss = f'Acc at Best Loss: {acc_at_best_loss:.4f} (Ep {int(best_loss_epoch)})'
    ax2.axvline(x=best_loss_epoch, color='purple', linestyle=':', alpha=0.7, label=label_at_loss)
    ax2.scatter(best_loss_epoch, acc_at_best_loss, color='purple', marker='X', s=70, zorder=6)

    ax2.set_title('Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 4. Save Results
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    save_path = os.path.join(output_path, f"{base_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" [SUCCESS] Analysis complete. Title: '{base_name}', Saved to: {save_path}")



if __name__ == "__main__":
    log_path = "/home/yujin/projects/wind/WIND_bkg_rejection/logs"
    sub_path = "test/version_0"

    logger_dir  = os.path.join(log_path, sub_path)
    output_path = os.path.join(log_path, sub_path)
    loss_acc_analysis(metrics_path=logger_dir , output_path=output_path, png_title="loss_acc_curve.png" )