import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
import timm

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessing import CXRPreprocessor
from src.data.dataset import NIHChestXrayDataset

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_ece(y_true, y_prob, n_bins=10):
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)

@torch.no_grad()
def get_logits_and_targets(model, loader, device):
    model.eval()
    all_targets = []
    all_logits = []
    
    for idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        with autocast('cuda'):
            outputs = model(images)
        all_targets.append(labels.cpu().numpy())
        all_logits.append(outputs.cpu().numpy())
        
    return np.concatenate(all_targets, axis=0), np.concatenate(all_logits, axis=0)

def fit_temperature(val_logits, val_targets):
    # Enforce NIH validation-only input criteria
    assert val_logits.shape == (val_targets.shape[0], 14), f"Fit error: input shapes mismatch {val_logits.shape}"
    assert len(val_targets) == 8936, f"Fit error: input size {len(val_targets)} does not match NIH validation size 8936"
    
    print("Optimizing temperature parameter on NIH validation split...")
    
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    targets_t = torch.tensor(val_targets, dtype=torch.float32)
    
    # Initialize temperature
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    def eval_loss():
        optimizer.zero_grad()
        loss = loss_fn(logits_t / temperature, targets_t)
        loss.backward()
        return loss
        
    optimizer.step(eval_loss)
    optimal_t = float(temperature.item())
    print(f"Optimal Temperature: T = {optimal_t:.4f}")
    return optimal_t

def get_youden_threshold(y_true, y_prob):
    if len(np.unique(y_true)) <= 1:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.argmax(tpr - fpr)
    return float(np.clip(thresholds[idx], 0.01, 0.99))

def get_sensitivity_targeted_threshold(y_true, y_prob, target_sens=0.80):
    if len(np.unique(y_true)) <= 1:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid_idx = np.where(tpr >= target_sens)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmin(fpr[valid_idx])]
        return float(np.clip(thresholds[best_idx], 0.01, 0.99))
    else:
        best_idx = np.argmax(tpr)
        return float(np.clip(thresholds[best_idx], 0.01, 0.99))

def compute_metrics_for_thresholds(y_true, y_prob, thresholds, classes):
    """
    Computes threshold-dependent and independent metrics.
    """
    n_classes = len(classes)
    y_pred_bin = np.zeros_like(y_prob, dtype=int)
    for i in range(n_classes):
        y_pred_bin[:, i] = (y_prob[:, i] >= thresholds[i]).astype(int)
        
    # Micro/Macro stats
    tp = np.sum((y_true == 1) & (y_pred_bin == 1), axis=0)
    fp = np.sum((y_true == 0) & (y_pred_bin == 1), axis=0)
    tn = np.sum((y_true == 0) & (y_pred_bin == 0), axis=0)
    fn = np.sum((y_true == 1) & (y_pred_bin == 0), axis=0)
    
    sens = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    spec = np.where((tn + fp) > 0, tn / (tn + fp), 1.0)
    prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    f1 = np.where((prec + sens) > 0, 2 * prec * sens / (prec + sens), 0.0)
    
    # AUROC/AUPRC
    auc_scores = np.zeros(n_classes)
    auprc_scores = np.zeros(n_classes)
    ece_scores = np.zeros(n_classes)
    brier_scores = np.zeros(n_classes)
    
    for i in range(n_classes):
        # AUROC
        try:
            if len(np.unique(y_true[:, i])) > 1:
                auc_scores[i] = roc_auc_score(y_true[:, i], y_prob[:, i])
            else:
                auc_scores[i] = 0.5
        except Exception:
            auc_scores[i] = 0.5
            
        # AUPRC
        try:
            auprc_scores[i] = average_precision_score(y_true[:, i], y_prob[:, i])
        except Exception:
            auprc_scores[i] = 0.0
            
        # ECE
        ece_scores[i] = compute_ece(y_true[:, i], y_prob[:, i])
        # Brier
        brier_scores[i] = np.mean((y_prob[:, i] - y_true[:, i])**2)
        
    metrics = {
        'macro_auroc': float(np.mean(auc_scores)),
        'macro_auprc': float(np.mean(auprc_scores)),
        'macro_f1': float(np.mean(f1)),
        'macro_sensitivity': float(np.mean(sens)),
        'macro_specificity': float(np.mean(spec)),
        'macro_ece': float(np.mean(ece_scores)),
        'macro_brier': float(np.mean(brier_scores)),
        'per_class': {}
    }
    
    for i, d in enumerate(classes):
        metrics['per_class'][d] = {
            'auroc': float(auc_scores[i]),
            'auprc': float(auprc_scores[i]),
            'f1': float(f1[i]),
            'sensitivity': float(sens[i]),
            'specificity': float(spec[i]),
            'ece': float(ece_scores[i]),
            'brier': float(brier_scores[i]),
            'threshold_selected': float(thresholds[i]),
            'confusion_statistics': {
                'tn': int(tn[i]),
                'fp': int(fp[i]),
                'fn': int(fn[i]),
                'tp': int(tp[i])
            }
        }
        
    return metrics

def plot_calibration_curves_comparison(y_true, y_prob_uncal, y_prob_cal, classes, save_path, title):
    plt.figure(figsize=(10, 8), dpi=300)
    
    # We plot the average reliability curve across all classes
    mean_prob_pred_uncal = np.zeros(10)
    mean_prob_true_uncal = np.zeros(10)
    mean_prob_pred_cal = np.zeros(10)
    mean_prob_true_cal = np.zeros(10)
    
    count_uncal = 0
    count_cal = 0
    
    for idx in range(len(classes)):
        if len(np.unique(y_true[:, idx])) > 1:
            # Uncalibrated
            true_uncal, pred_uncal = calibration_curve(y_true[:, idx], y_prob_uncal[:, idx], n_bins=10, strategy='uniform')
            if len(pred_uncal) == 10:
                mean_prob_pred_uncal += pred_uncal
                mean_prob_true_uncal += true_uncal
                count_uncal += 1
            # Calibrated
            true_cal, pred_cal = calibration_curve(y_true[:, idx], y_prob_cal[:, idx], n_bins=10, strategy='uniform')
            if len(pred_cal) == 10:
                mean_prob_pred_cal += pred_cal
                mean_prob_true_cal += true_cal
                count_cal += 1
                
    if count_uncal > 0:
        mean_prob_pred_uncal /= count_uncal
        mean_prob_true_uncal /= count_uncal
        plt.plot(mean_prob_pred_uncal, mean_prob_true_uncal, marker='o', ls='-', color='red', label='Uncalibrated (Baseline)')
        
    if count_cal > 0:
        mean_prob_pred_cal /= count_cal
        mean_prob_true_cal /= count_cal
        plt.plot(mean_prob_pred_cal, mean_prob_true_cal, marker='s', ls='-', color='blue', label='Calibrated (Temperature Scaled)')
        
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Perfect Calibration')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean Predicted Probability (Confidence)', fontsize=12)
    plt.ylabel('Fraction of Positives (Accuracy)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved calibration comparison to {save_path}")

def plot_tradeoff_curves(y_true, y_prob, classes, save_path):
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Plot trade-off curves for select classes (e.g. Effusion, Cardiomegaly, Atelectasis, No Finding)
    target_classes = ['Cardiomegaly', 'Effusion', 'Atelectasis', 'No Finding']
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, d in enumerate(target_classes):
        if d in classes:
            class_idx = classes.index(d)
            if len(np.unique(y_true[:, class_idx])) > 1:
                fpr, tpr, thresholds = roc_curve(y_true[:, class_idx], y_prob[:, class_idx])
                # Precision, recall, thresholds_pr = precision_recall_curve(y_true[:, class_idx], y_prob[:, class_idx])
                
                # Plot sensitivity vs. specificity
                specificities = 1.0 - fpr
                plt.plot(thresholds, tpr, ls='-', color=colors[idx], label=f'{d} Sensitivity (TPR)')
                plt.plot(thresholds, specificities, ls='--', color=colors[idx], label=f'{d} Specificity (TNR)')
                
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('Metric Score', fontsize=12)
    plt.title('Sensitivity-Specificity Trade-off vs. Threshold (NIH Validation)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower center", fontsize=8, ncol=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved tradeoff curves to {save_path}")

def main():
    print("==================================================")
    print("      CALIBRATION AND THRESHOLD ANALYSIS          ")
    print("==================================================")
    
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Paths configuration
    checkpoint_path = "/home/23adr188/chest_xray_project/experiments/checkpoints/densenet121/best.pt"
    nih_val_csv = "/home/23adr188/chest_xray_project/data/splits/val.csv"
    nih_test_csv = "/home/23adr188/chest_xray_project/data/splits/test.csv"
    vindr_metadata_csv = "/home/23adr188/chest_xray_project/data/external/vinbigdata_metadata.csv"
    
    output_dir = "/home/23adr188/chest_xray_project/experiments/results/calibration_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. DataLoaders and model loading
    print("Loading data splits and DenseNet-121 model...")
    preprocessor = CXRPreprocessor(img_size=(224, 224), is_training=False)
    
    val_dataset = NIHChestXrayDataset(csv_path=nih_val_csv, transform=preprocessor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    test_dataset = NIHChestXrayDataset(csv_path=nih_test_csv, transform=preprocessor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    vindr_dataset = NIHChestXrayDataset(csv_path=vindr_metadata_csv, transform=preprocessor)
    vindr_loader = DataLoader(vindr_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    model = timm.create_model('densenet121', pretrained=False, num_classes=14)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 3. Extract validation split predictions (logits and targets)
    print("Extracting validation predictions...")
    val_targets, val_logits = get_logits_and_targets(model, val_loader, device)
    
    # 4. Fit Temperature Scaling
    optimal_t = fit_temperature(val_logits, val_targets)
    
    # Save temperature scaling parameter
    torch.save({'temperature': optimal_t}, "/home/23adr188/chest_xray_project/experiments/checkpoints/densenet121/temperature_scaling.pt")
    print("Calibration parameter saved to temperature_scaling.pt")
    
    # Calibrated validation probabilities
    val_probs_uncal = 1.0 / (1.0 + np.exp(-val_logits))
    val_probs_cal = 1.0 / (1.0 + np.exp(-val_logits / optimal_t))
    
    # 5. Extract test predictions (NIH test and VinDr)
    print("Extracting test predictions (NIH Test)...")
    test_targets, test_logits = get_logits_and_targets(model, test_loader, device)
    test_probs_uncal = 1.0 / (1.0 + np.exp(-test_logits))
    test_probs_cal = 1.0 / (1.0 + np.exp(-test_logits / optimal_t))
    
    print("Extracting test predictions (VinDr External)...")
    vindr_targets_all, vindr_logits_all = get_logits_and_targets(model, vindr_loader, device)
    vindr_probs_all_uncal = 1.0 / (1.0 + np.exp(-vindr_logits_all))
    vindr_probs_all_cal = 1.0 / (1.0 + np.exp(-vindr_logits_all / optimal_t))
    
    # 6. Format VinDr datasets to 9 compatible classes
    # compatible index indices: Atelectasis: 0, Cardiomegaly: 1, Effusion: 2, Infiltration: 3, Consolidation: 8, Pneumothorax: 7, Fibrosis: 11, Pleural_Thickening: 12
    vindr_compatible_idx = [0, 1, 2, 3, 8, 7, 11, 12]
    vindr_classes = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Consolidation',
        'Pneumothorax', 'Fibrosis', 'Pleural_Thickening', 'No Finding'
    ]
    
    vindr_true_comp = vindr_targets_all[:, vindr_compatible_idx]
    vindr_true_no_finding = vindr_dataset.df['No Finding'].values
    vindr_targets_9 = np.hstack([vindr_true_comp, vindr_true_no_finding.reshape(-1, 1)])
    
    # Uncalibrated VinDr
    vindr_probs_comp_uncal = vindr_probs_all_uncal[:, vindr_compatible_idx]
    vindr_probs_no_finding_uncal = 1.0 - np.max(vindr_probs_comp_uncal, axis=1)
    vindr_probs_9_uncal = np.hstack([vindr_probs_comp_uncal, vindr_probs_no_finding_uncal.reshape(-1, 1)])
    
    # Calibrated VinDr
    vindr_probs_comp_cal = vindr_probs_all_cal[:, vindr_compatible_idx]
    vindr_probs_no_finding_cal = 1.0 - np.max(vindr_probs_comp_cal, axis=1)
    vindr_probs_9_cal = np.hstack([vindr_probs_comp_cal, vindr_probs_no_finding_cal.reshape(-1, 1)])
    
    # 7. Threshold Selection on NIH Validation Split ONLY
    nih_classes = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    
    # Format No Finding for NIH Validation
    nih_val_no_finding_true = (np.sum(val_targets, axis=1) == 0).astype(int)
    nih_val_no_finding_prob_cal = 1.0 - np.max(val_probs_cal[:, [0,1,2,3,8,7,11,12]], axis=1)
    
    # A. Youden's J Thresholds
    print("Selecting Youden J thresholds on validation data...")
    val_youden_thresholds_14 = [get_youden_threshold(val_targets[:, i], val_probs_cal[:, i]) for i in range(14)]
    val_youden_no_finding_thresh = get_youden_threshold(nih_val_no_finding_true, nih_val_no_finding_prob_cal)
    
    # B. Sensitivity Targeted (80%) Thresholds
    print("Selecting sensitivity-targeted (80%) thresholds on validation data...")
    val_sens80_thresholds_14 = [get_sensitivity_targeted_threshold(val_targets[:, i], val_probs_cal[:, i], target_sens=0.80) for i in range(14)]
    val_sens80_no_finding_thresh = get_sensitivity_targeted_threshold(nih_val_no_finding_true, nih_val_no_finding_prob_cal, target_sens=0.80)
    
    # Format Thresholds lists
    default_thresholds_14 = [0.5] * 14
    default_thresholds_9 = [0.5] * 9
    
    # Youden J thresholds mapping for VinDr compatible classes
    youden_thresholds_9 = [val_youden_thresholds_14[idx] for idx in vindr_compatible_idx] + [val_youden_no_finding_thresh]
    
    # Sensitivity-targeted thresholds mapping for VinDr compatible classes
    sens80_thresholds_9 = [val_sens80_thresholds_14[idx] for idx in vindr_compatible_idx] + [val_sens80_no_finding_thresh]
    
    # 8. Run Comparative Evaluations
    print("Evaluating Model A vs Model B across thresholds...")
    
    # A. Validation metrics
    val_metrics_uncal = compute_metrics_for_thresholds(val_targets, val_probs_uncal, default_thresholds_14, nih_classes)
    val_metrics_cal_default = compute_metrics_for_thresholds(val_targets, val_probs_cal, default_thresholds_14, nih_classes)
    val_metrics_cal_youden = compute_metrics_for_thresholds(val_targets, val_probs_cal, val_youden_thresholds_14, nih_classes)
    val_metrics_cal_sens80 = compute_metrics_for_thresholds(val_targets, val_probs_cal, val_sens80_thresholds_14, nih_classes)
    
    # B. NIH Test metrics
    test_metrics_uncal_default = compute_metrics_for_thresholds(test_targets, test_probs_uncal, default_thresholds_14, nih_classes)
    test_metrics_cal_default = compute_metrics_for_thresholds(test_targets, test_probs_cal, default_thresholds_14, nih_classes)
    test_metrics_cal_youden = compute_metrics_for_thresholds(test_targets, test_probs_cal, val_youden_thresholds_14, nih_classes)
    test_metrics_cal_sens80 = compute_metrics_for_thresholds(test_targets, test_probs_cal, val_sens80_thresholds_14, nih_classes)
    
    # C. VinDr metrics
    vindr_metrics_uncal_default = compute_metrics_for_thresholds(vindr_targets_9, vindr_probs_9_uncal, default_thresholds_9, vindr_classes)
    vindr_metrics_cal_default = compute_metrics_for_thresholds(vindr_targets_9, vindr_probs_9_cal, default_thresholds_9, vindr_classes)
    vindr_metrics_cal_youden = compute_metrics_for_thresholds(vindr_targets_9, vindr_probs_9_cal, youden_thresholds_9, vindr_classes)
    vindr_metrics_cal_sens80 = compute_metrics_for_thresholds(vindr_targets_9, vindr_probs_9_cal, sens80_thresholds_9, vindr_classes)
    
    # Programmatic Verification of locked baseline results
    nih_test_auroc = test_metrics_uncal_default['macro_auroc']
    vindr_test_auroc = vindr_metrics_uncal_default['macro_auroc']
    
    print(f"Reproduced Baseline - NIH Test Macro AUROC: {nih_test_auroc:.6f} (Expected: 0.799723)")
    print(f"Reproduced Baseline - VinDr Test Macro AUROC: {vindr_test_auroc:.6f} (Expected: 0.825733)")
    
    # Assert baseline reproducibility within tolerance
    assert abs(nih_test_auroc - 0.799723) < 1e-4, f"LOCKED EXCEPTION: NIH baseline AUROC mismatch: {nih_test_auroc}"
    assert abs(vindr_test_auroc - 0.825733) < 1e-4, f"LOCKED EXCEPTION: VinDr baseline AUROC mismatch: {vindr_test_auroc}"
    print("Baseline reference metrics reproduced successfully! Continuing...")
    
    # 9. Save Machine-Readable JSON results
    output_json = {
        'calibration_parameter': {
            'optimal_temperature': optimal_t,
            'fit_dataset': 'NIH ChestX-ray14 validation split'
        },
        'thresholds': {
            'youden_j_nih_14': {nih_classes[i]: val_youden_thresholds_14[i] for i in range(14)},
            'youden_j_vindr_9': {vindr_classes[i]: youden_thresholds_9[i] for i in range(9)},
            'sens80_nih_14': {nih_classes[i]: val_sens80_thresholds_14[i] for i in range(14)},
            'sens80_vindr_9': {vindr_classes[i]: sens80_thresholds_9[i] for i in range(9)}
        },
        'validation_metrics': {
            'uncalibrated_default': val_metrics_uncal,
            'calibrated_default': val_metrics_cal_default,
            'calibrated_youden': val_metrics_cal_youden,
            'calibrated_sens80': val_metrics_cal_sens80
        },
        'nih_test_metrics': {
            'uncalibrated_default_baseline_modelA': test_metrics_uncal_default,
            'calibrated_default': test_metrics_cal_default,
            'calibrated_youden_modelB': test_metrics_cal_youden,
            'calibrated_sens80': test_metrics_cal_sens80
        },
        'vindr_metrics': {
            'uncalibrated_default_baseline_modelA': vindr_metrics_uncal_default,
            'calibrated_default': vindr_metrics_cal_default,
            'calibrated_youden_modelB': vindr_metrics_cal_youden,
            'calibrated_sens80': vindr_metrics_cal_sens80
        },
        'meta': {
            'random_seed': 42,
            'checkpoint_path': checkpoint_path,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    json_path = os.path.join(output_dir, "calibration_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"Calibration metrics saved to: {json_path}")
    
    # 10. Generate figures
    plot_calibration_curves_comparison(test_targets, test_probs_uncal, test_probs_cal, nih_classes, 
                                       os.path.join(output_dir, "nih_calibration_comparison.png"),
                                       "NIH Test Cohort Average Reliability Diagram")
                                       
    plot_calibration_curves_comparison(vindr_targets_9, vindr_probs_9_uncal, vindr_probs_9_cal, vindr_classes, 
                                       os.path.join(output_dir, "vindr_calibration_comparison.png"),
                                       "VinDr-CXR Cohort Average Reliability Diagram")
                                       
    plot_tradeoff_curves(val_targets, val_probs_cal, nih_classes, 
                         os.path.join(output_dir, "tradeoff_curves.png"))
                         
    print("==================================================")
    print("         CALIBRATION RUN COMPLETED SUCCESSFULLY   ")
    print("==================================================")

if __name__ == "__main__":
    main()
