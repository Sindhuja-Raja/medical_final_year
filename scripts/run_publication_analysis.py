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
def get_predictions(model, loader, device):
    model.eval()
    all_targets = []
    all_outputs = []
    
    for idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        with autocast('cuda'):
            outputs = model(images)
        probs = torch.sigmoid(outputs)
        all_targets.append(labels.cpu().numpy())
        all_outputs.append(probs.cpu().numpy())
        
    return np.concatenate(all_targets, axis=0), np.concatenate(all_outputs, axis=0)

def evaluate_metrics_subset(y_true, y_pred, classes):
    """
    Computes AUROC, AUPRC, F1, sensitivity, specificity point estimates in a highly vectorized manner.
    """
    metrics = {
        'macro_auroc': 0.0,
        'micro_auroc': 0.0,
        'macro_auprc': 0.0,
        'micro_auprc': 0.0,
        'macro_f1': 0.0,
        'macro_sensitivity': 0.0,
        'macro_specificity': 0.0,
        'per_class': {}
    }
    
    n_classes = len(classes)
    
    # 1. Overall and Per-Class AUROC/AUPRC using vectorized scikit-learn calls
    try:
        # Check class variance to prevent roc_auc_score crash on bootstrapping degenerate folds
        valid_classes = [i for i in range(n_classes) if len(np.unique(y_true[:, i])) > 1]
        
        auc_scores = np.zeros(n_classes)
        if len(valid_classes) > 0:
            auc_scores[valid_classes] = roc_auc_score(y_true[:, valid_classes], y_pred[:, valid_classes], average=None)
            invalid_classes = list(set(range(n_classes)) - set(valid_classes))
            if invalid_classes:
                auc_scores[invalid_classes] = 0.5
        else:
            auc_scores.fill(0.5)
            
        metrics['macro_auroc'] = float(np.mean(auc_scores))
        
        if len(valid_classes) > 0:
            metrics['micro_auroc'] = float(roc_auc_score(y_true[:, valid_classes], y_pred[:, valid_classes], average='micro'))
        else:
            metrics['micro_auroc'] = 0.5
    except Exception:
        auc_scores = np.zeros(n_classes)
        auc_scores.fill(0.5)
        
    try:
        auprc_scores = average_precision_score(y_true, y_pred, average=None)
        if isinstance(auprc_scores, float):
            auprc_scores = np.array([auprc_scores])
        metrics['macro_auprc'] = float(np.mean(auprc_scores))
        metrics['micro_auprc'] = float(average_precision_score(y_true, y_pred, average='micro'))
    except Exception:
        auprc_scores = np.zeros(n_classes)
        
    # 2. Vectorized Binary Metrics (F1, Sensitivity, Specificity, Confusion Stats)
    y_pred_bin = (y_pred >= 0.5).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred_bin == 1), axis=0)
    fp = np.sum((y_true == 0) & (y_pred_bin == 1), axis=0)
    tn = np.sum((y_true == 0) & (y_pred_bin == 0), axis=0)
    fn = np.sum((y_true == 1) & (y_pred_bin == 0), axis=0)
    
    sens = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    spec = np.where((tn + fp) > 0, tn / (tn + fp), 1.0)
    prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    f1 = np.where((prec + sens) > 0, 2 * prec * sens / (prec + sens), 0.0)
    
    metrics['macro_f1'] = float(np.mean(f1))
    metrics['macro_sensitivity'] = float(np.mean(sens))
    metrics['macro_specificity'] = float(np.mean(spec))
    
    for idx, d in enumerate(classes):
        metrics['per_class'][d] = {
            'auroc': float(auc_scores[idx]) if idx < len(auc_scores) else 0.5,
            'auprc': float(auprc_scores[idx]) if idx < len(auprc_scores) else 0.0,
            'f1': float(f1[idx]),
            'sensitivity': float(sens[idx]),
            'specificity': float(spec[idx]),
            'confusion_statistics': {
                'tn': int(tn[idx]),
                'fp': int(fp[idx]),
                'fn': int(fn[idx]),
                'tp': int(tp[idx])
            }
        }
        
    return metrics

def run_bootstrapping(y_true, y_pred, classes, n_replicates=1000):
    print(f"Running bootstrapping with {n_replicates} replicates...")
    n_samples = len(y_true)
    
    # Store metrics for each replicate
    rep_results = []
    
    start_time = time.time()
    for b in range(n_replicates):
        if (b + 1) % 200 == 0:
            print(f"  Processed {b + 1}/{n_replicates} replicates...")
            
        # Resample indices with replacement
        boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_b = y_true[boot_indices]
        y_pred_b = y_pred[boot_indices]
        
        # Calculate metrics
        metrics_b = evaluate_metrics_subset(y_true_b, y_pred_b, classes)
        rep_results.append(metrics_b)
        
    print(f"Bootstrapping complete in {time.time() - start_time:.2f} seconds.")
    
    # Calculate 95% Confidence Intervals (2.5th and 97.5th percentiles)
    ci_results = {}
    
    # Keys for overall metrics
    overall_keys = ['macro_auroc', 'micro_auroc', 'macro_auprc', 'micro_auprc', 'macro_f1', 'macro_sensitivity', 'macro_specificity']
    for key in overall_keys:
        vals = sorted([r[key] for r in rep_results])
        ci_results[key] = {
            'lower': float(np.percentile(vals, 2.5)),
            'upper': float(np.percentile(vals, 97.5))
        }
        
    # Keys for per class metrics
    ci_results['per_class'] = {}
    for d in classes:
        ci_results['per_class'][d] = {}
        for metric in ['auroc', 'auprc', 'f1', 'sensitivity', 'specificity']:
            vals = sorted([r['per_class'][d][metric] for r in rep_results])
            ci_results['per_class'][d][metric] = {
                'lower': float(np.percentile(vals, 2.5)),
                'upper': float(np.percentile(vals, 97.5))
            }
            
    return ci_results

def plot_roc_curves(y_true, y_pred, classes, save_path):
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Compute overall macro ROC curve
    all_fpr = []
    all_tpr = []
    
    # Plot individual class curves
    for idx, d in enumerate(classes):
        if len(np.unique(y_true[:, idx])) > 1:
            fpr, tpr, _ = roc_curve(y_true[:, idx], y_pred[:, idx])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, alpha=0.7, label=f'{d} (AUC = {roc_auc:.3f})')
            
            # Interpolate for macro average
            mean_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            all_fpr.append(mean_fpr)
            all_tpr.append(mean_tpr)
            
    if all_tpr:
        macro_tpr = np.mean(all_tpr, axis=0)
        macro_auc = auc(np.linspace(0, 1, 100), macro_tpr)
        plt.plot(np.linspace(0, 1, 100), macro_tpr, color='black', linestyle='--', lw=2.5, label=f'Macro Average (AUC = {macro_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=8, ncol=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curves saved to {save_path}")

def plot_pr_curves(y_true, y_pred, classes, save_path):
    plt.figure(figsize=(10, 8), dpi=300)
    
    all_precision = []
    all_recall = []
    
    # Plot individual class curves
    for idx, d in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true[:, idx], y_pred[:, idx])
        auprc = average_precision_score(y_true[:, idx], y_pred[:, idx])
        plt.plot(recall, precision, lw=1.5, alpha=0.7, label=f'{d} (AUPRC = {auprc:.3f})')
        
        # Interpolate for macro average
        mean_recall = np.linspace(0, 1, 100)
        mean_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        all_precision.append(mean_precision)
        all_recall.append(mean_recall)
        
    if all_precision:
        macro_precision = np.mean(all_precision, axis=0)
        macro_auprc = average_precision_score(y_true, y_pred, average='macro')
        plt.plot(np.linspace(0, 1, 100), macro_precision, color='black', linestyle='--', lw=2.5, label=f'Macro Average (AUPRC = {macro_auprc:.3f})')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.title('Precision-Recall (PR) Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=8, ncol=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"PR curves saved to {save_path}")

def plot_calibration_curves(y_true, y_pred, classes, save_path):
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Plot individual class curves
    for idx, d in enumerate(classes):
        if len(np.unique(y_true[:, idx])) > 1:
            prob_true, prob_pred = calibration_curve(y_true[:, idx], y_pred[:, idx], n_bins=10, strategy='uniform')
            ece = compute_ece(y_true[:, idx], y_pred[:, idx])
            plt.plot(prob_pred, prob_true, marker='o', lw=1.5, alpha=0.7, label=f'{d} (ECE = {ece:.3f})')
            
    plt.plot([0, 1], [0, 1], color='black', linestyle=':', lw=1.5, label='Perfect Calibration')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean Predicted Probability (Confidence)', fontsize=12)
    plt.ylabel('Fraction of Positives (Accuracy)', fontsize=12)
    plt.title('Reliability Calibration Diagrams', fontsize=14, fontweight='bold')
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Calibration curves saved to {save_path}")

def main():
    print("==================================================")
    print("        RUNNING PUBLICATION ANALYSIS PIPELINE     ")
    print("==================================================")
    
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = "/home/23adr188/chest_xray_project/experiments/checkpoints/densenet121/best.pt"
    nih_test_csv = "/home/23adr188/chest_xray_project/data/splits/test.csv"
    vindr_metadata_csv = "/home/23adr188/chest_xray_project/data/external/vinbigdata_metadata.csv"
    
    output_dir = "/home/23adr188/chest_xray_project/experiments/results/publication_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize DataLoader & Model
    print("Loading locked model and loaders...")
    preprocessor = CXRPreprocessor(img_size=(224, 224), is_training=False)
    
    nih_dataset = NIHChestXrayDataset(csv_path=nih_test_csv, transform=preprocessor)
    nih_loader = DataLoader(nih_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    vindr_dataset = NIHChestXrayDataset(csv_path=vindr_metadata_csv, transform=preprocessor)
    vindr_loader = DataLoader(vindr_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    model = timm.create_model('densenet121', pretrained=False, num_classes=14)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 2. Get predictions
    print("Extracting predictions for NIH test split...")
    nih_true_all, nih_pred_all = get_predictions(model, nih_loader, device)
    
    print("Extracting predictions for VinDr external cohort...")
    vindr_true_all, vindr_pred_all = get_predictions(model, vindr_loader, device)
    
    # 3. Define classes
    nih_classes = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    
    # For VinDr: 9 compatible classes
    vindr_compatible_indices = [0, 1, 2, 3, 8, 7, 11, 12]
    vindr_classes = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Consolidation',
        'Pneumothorax', 'Fibrosis', 'Pleural_Thickening', 'No Finding'
    ]
    
    # Format VinDr predictions and ground truth
    vindr_true_comp = vindr_true_all[:, vindr_compatible_indices]
    vindr_pred_comp = vindr_pred_all[:, vindr_compatible_indices]
    vindr_true_no_finding = vindr_dataset.df['No Finding'].values
    vindr_pred_no_finding = 1.0 - np.max(vindr_pred_comp, axis=1)
    
    vindr_true_9 = np.hstack([vindr_comp_true for vindr_comp_true in (vindr_true_comp, vindr_true_no_finding.reshape(-1, 1))])
    vindr_pred_9 = np.hstack([vindr_comp_pred for vindr_comp_pred in (vindr_pred_comp, vindr_pred_no_finding.reshape(-1, 1))])
    
    # 4. Compute Point Estimates & ECE
    print("Computing baseline point estimates & ECE values...")
    nih_metrics = evaluate_metrics_subset(nih_true_all, nih_pred_all, nih_classes)
    for idx, d in enumerate(nih_classes):
        nih_metrics['per_class'][d]['ece'] = compute_ece(nih_true_all[:, idx], nih_pred_all[:, idx])
    nih_metrics['macro_ece'] = float(np.mean([nih_metrics['per_class'][d]['ece'] for d in nih_classes]))
    
    vindr_metrics = evaluate_metrics_subset(vindr_true_9, vindr_pred_9, vindr_classes)
    for idx, d in enumerate(vindr_classes):
        vindr_metrics['per_class'][d]['ece'] = compute_ece(vindr_true_9[:, idx], vindr_pred_9[:, idx])
    vindr_metrics['macro_ece'] = float(np.mean([vindr_metrics['per_class'][d]['ece'] for d in vindr_classes]))
    
    # 5. Run Bootstrapping
    print("\n--- Running NIH Bootstrapping ---")
    nih_ci = run_bootstrapping(nih_true_all, nih_pred_all, nih_classes, n_replicates=1000)
    
    print("\n--- Running VinDr Bootstrapping ---")
    vindr_ci = run_bootstrapping(vindr_true_9, vindr_pred_9, vindr_classes, n_replicates=1000)
    
    # Save full stats dictionaries (Point estimates + CIs)
    nih_full_results = {
        'point_estimates': nih_metrics,
        'confidence_intervals': nih_ci
    }
    with open(os.path.join(output_dir, "nih_results.json"), "w") as f:
        json.dump(nih_full_results, f, indent=2)
        
    vindr_full_results = {
        'point_estimates': vindr_metrics,
        'confidence_intervals': vindr_ci
    }
    with open(os.path.join(output_dir, "vindr_results.json"), "w") as f:
        json.dump(vindr_full_results, f, indent=2)
        
    print(f"Metrics JSON results saved in {output_dir}")
    
    # 6. Generate Figures
    print("\nGenerating figures...")
    plot_roc_curves(nih_true_all, nih_pred_all, nih_classes, os.path.join(output_dir, "nih_roc_curves.png"))
    plot_pr_curves(nih_true_all, nih_pred_all, nih_classes, os.path.join(output_dir, "nih_pr_curves.png"))
    plot_calibration_curves(nih_true_all, nih_pred_all, nih_classes, os.path.join(output_dir, "nih_calibration.png"))
    
    plot_roc_curves(vindr_true_9, vindr_pred_9, vindr_classes, os.path.join(output_dir, "vindr_roc_curves.png"))
    plot_pr_curves(vindr_true_9, vindr_pred_9, vindr_classes, os.path.join(output_dir, "vindr_pr_curves.png"))
    plot_calibration_curves(vindr_true_9, vindr_pred_9, vindr_classes, os.path.join(output_dir, "vindr_calibration.png"))
    
    print("==================================================")
    print("            PUBLICATION ANALYSIS COMPLETED        ")
    print("==================================================")

if __name__ == "__main__":
    main()
