import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, classification_report, roc_auc_score

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

def roc_curve_analysis(y_test, p_test, test_auc, test_internal_eff_wp, test_bkg_residual_wp, output_path):
    fpr, tpr, thresholds = roc_curve(y_test, p_test)
    roc_data = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    })
    roc_csv_path = os.path.join(output_path, "roc_curve_values.csv")
    roc_data.to_csv(roc_csv_path, index=False)
    print(f"[DATA] ROC curve points saved to: {roc_csv_path}")

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter([test_bkg_residual_wp], [test_internal_eff_wp], label="working point")
    plt.xlabel("False positive rate: 16N misidentified as internal")
    plt.ylabel("True positive rate: internal efficiency")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    outdir = os.path.join(output_path, "roc_curve.png")
    plt.savefig(outdir, dpi=160)
    plt.close()

def performance_summary(model, val_loader, test_loader, criterion, device, target_bkg_residual, output_path): 
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

    summary_text = ""

    summary_text += f"\n[TEST RESULT: threshold 0.5]\n"
    summary_text += f"loss = {test_loss:.4f}\n"
    summary_text += f"accuracy = {accuracy_score(y_test, y_pred_05):.4f}\n"
    summary_text += f"AUC = {test_auc:.4f}\n"
    summary_text += "Confusion matrix, rows=true, cols=pred:\n"
    summary_text += f"{confusion_matrix(y_test, y_pred_05)}\n"

    summary_text += f"\n[HIGH-PURITY WORKING POINT]\n"
    summary_text += f"Target background residual set at: {target_bkg_residual:.3f}\n"
    summary_text += f"Calculated threshold = {wp_threshold:.6f}\n"
    summary_text += f"Validation internal efficiency = {val_eff_at_target:.4f}\n"
    summary_text += f"Validation 16N residual       = {val_bkg_residual:.4f}\n"
    summary_text += f"Test internal efficiency       = {test_internal_eff_wp:.4f}\n"
    summary_text += f"Test 16N residual               = {test_bkg_residual_wp:.4f}\n"
    summary_text += f"Test S/B improvement factor    = {test_improvement_wp:.4f}\n"

    summary_text += "\n[TEST classification report at high-purity working point]\n"
    summary_text += classification_report(
        y_test,
        y_pred_wp,
        target_names=["16N background", "internal ES signal"],
        digits=4,
    )

    # 6. 화면에 출력
    print(summary_text)

    # 7. 파일로 저장
    # output_path가 폴더라면 파일명을 붙여주고, 파일이라면 그대로 사용합니다.
    if os.path.isdir(output_path):
        log_file_path = os.path.join(output_path, f"test_summary.txt")
    else:
        log_file_path = output_path

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    print(f"\n[INFO] Summary saved to: {log_file_path}")


    roc_curve_analysis(y_test, p_test, test_auc, test_internal_eff_wp, test_bkg_residual_wp, output_path)