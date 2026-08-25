import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
import timm

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, confusion_matrix

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

def evaluate_subgroup_metrics(y_true, y_prob, thresholds, classes):
    """
    Computes macro AUROC, AUPRC, F1, sensitivity, specificity for a subset of predictions.
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return {
            'macro_auroc': 0.0, 'macro_auprc': 0.0, 'macro_f1': 0.0,
            'macro_sensitivity': 0.0, 'macro_specificity': 0.0, 'sample_size': 0
        }
        
    n_classes = len(classes)
    y_pred_bin = np.zeros_like(y_prob, dtype=int)
    for i in range(n_classes):
        y_pred_bin[:, i] = (y_prob[:, i] >= thresholds[i]).astype(int)
        
    tp = np.sum((y_true == 1) & (y_pred_bin == 1), axis=0)
    fp = np.sum((y_true == 0) & (y_pred_bin == 1), axis=0)
    tn = np.sum((y_true == 0) & (y_pred_bin == 0), axis=0)
    fn = np.sum((y_true == 1) & (y_pred_bin == 0), axis=0)
    
    sens = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    spec = np.where((tn + fp) > 0, tn / (tn + fp), 1.0)
    prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    f1 = np.where((prec + sens) > 0, 2 * prec * sens / (prec + sens), 0.0)
    
    auc_scores = np.zeros(n_classes)
    auprc_scores = np.zeros(n_classes)
    
    for i in range(n_classes):
        try:
            if len(np.unique(y_true[:, i])) > 1:
                auc_scores[i] = roc_auc_score(y_true[:, i], y_prob[:, i])
            else:
                auc_scores[i] = 0.5
        except Exception:
            auc_scores[i] = 0.5
            
        try:
            auprc_scores[i] = average_precision_score(y_true[:, i], y_prob[:, i])
        except Exception:
            auprc_scores[i] = 0.0
            
    return {
        'macro_auroc': float(np.mean(auc_scores)),
        'macro_auprc': float(np.mean(auprc_scores)),
        'macro_f1': float(np.mean(f1)),
        'macro_sensitivity': float(np.mean(sens)),
        'macro_specificity': float(np.mean(spec)),
        'sample_size': int(n_samples)
    }

def main():
    print("==================================================")
    print("            NIH SUBGROUP ANALYSIS                ")
    print("==================================================")
    
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    checkpoint_path = "/home/23adr188/chest_xray_project/experiments/checkpoints/densenet121/best.pt"
    test_csv_path = "/home/23adr188/chest_xray_project/data/splits/test.csv"
    calibration_metrics_json = "/home/23adr188/chest_xray_project/experiments/results/calibration_analysis/calibration_metrics.json"
    
    output_dir = "/home/23adr188/chest_xray_project/experiments/results/final_analysis/subgroup_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Calibration JSON
    if not os.path.exists(calibration_metrics_json):
        print(f"ERROR: Calibration file {calibration_metrics_json} not found.")
        sys.exit(1)
        
    with open(calibration_metrics_json, 'r') as f:
        calib_data = json.load(f)
        
    optimal_t = calib_data["calibration_parameter"]["optimal_temperature"]
    nih_classes = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    
    youden_thresholds = [calib_data["thresholds"]["youden_j_nih_14"][c] for c in nih_classes]
    sens80_thresholds = [calib_data["thresholds"]["sens80_nih_14"][c] for c in nih_classes]
    default_thresholds = [0.5] * 14
    
    # 2. Model & Loader Setup
    print("Loading test dataset and model...")
    preprocessor = CXRPreprocessor(img_size=(224, 224), is_training=False)
    test_dataset = NIHChestXrayDataset(csv_path=test_csv_path, transform=preprocessor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    model = timm.create_model('densenet121', pretrained=False, num_classes=14)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 3. Inference
    print("Running inference on NIH test split...")
    test_targets, test_logits = get_logits_and_targets(model, test_loader, device)
    
    # Probs
    test_probs_uncal = 1.0 / (1.0 + np.exp(-test_logits))
    test_probs_cal = 1.0 / (1.0 + np.exp(-test_logits / optimal_t))
    
    # 4. Load Metadata DF
    df = pd.read_csv(test_csv_path)
    
    # Define Subgroups masks
    subgroups = {
        'Gender': {
            'Male': df['Patient Gender'] == 'M',
            'Female': df['Patient Gender'] == 'F'
        },
        'View Position': {
            'AP': df['View Position'] == 'AP',
            'PA': df['View Position'] == 'PA'
        },
        'Age': {
            'Pediatric (<20)': (df['Patient Age'] >= 0) & (df['Patient Age'] < 20),
            'Young Adult (20-40)': (df['Patient Age'] >= 20) & (df['Patient Age'] < 40),
            'Middle Aged (40-60)': (df['Patient Age'] >= 40) & (df['Patient Age'] < 60),
            'Older Adult (60-80)': (df['Patient Age'] >= 60) & (df['Patient Age'] < 80),
            'Geriatric (>=80)': (df['Patient Age'] >= 80) & (df['Patient Age'] <= 100),
            'Unknown/Outlier': (df['Patient Age'] < 0) | (df['Patient Age'] > 100) | df['Patient Age'].isna()
        }
    }
    
    subgroup_results = {}
    
    # 5. Evaluate Subgroups
    print("Evaluating metrics across subgroups...")
    for group_cat, group_dict in subgroups.items():
        subgroup_results[group_cat] = {}
        for subgroup_name, mask in group_dict.items():
            mask_np = mask.values
            y_true_sub = test_targets[mask_np]
            y_prob_uncal_sub = test_probs_uncal[mask_np]
            y_prob_cal_sub = test_probs_cal[mask_np]
            
            subgroup_results[group_cat][subgroup_name] = {
                'sample_size': int(np.sum(mask_np)),
                'ModelA_uncal_default': evaluate_subgroup_metrics(y_true_sub, y_prob_uncal_sub, default_thresholds, nih_classes),
                'ModelB_cal_youden': evaluate_subgroup_metrics(y_true_sub, y_prob_cal_sub, youden_thresholds, nih_classes),
                'ModelB_cal_sens80': evaluate_subgroup_metrics(y_true_sub, y_prob_cal_sub, sens80_thresholds, nih_classes)
            }
            
    # Save JSON results
    json_path = os.path.join(output_dir, "subgroup_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(subgroup_results, f, indent=2)
    print(f"Subgroup metrics JSON saved to: {json_path}")
    
    # Save Markdown report
    md_path = os.path.join(output_dir, "subgroup_report.md")
    with open(md_path, 'w') as f:
        f.write("# Subgroup Analysis Report — Model A vs. Model B\n\n")
        f.write("This report presents subgroup performance evaluation across Patient Gender, View Position, and Patient Age brackets on the NIH test split.\n\n")
        
        f.write("## 1. Safety Lock Confirmations\n")
        f.write("- **Model weights locked:** `best.pt` unchanged.\n")
        f.write("- **No demographics tuning:** All decision thresholds were frozen from NIH validation data and kept identical across subgroups.\n\n")
        
        for category, sub_dict in subgroup_results.items():
            f.write(f"## 2. Evaluation by {category}\n\n")
            f.write("| Subgroup | Sample Size | Configuration | AUROC | AUPRC | F1 | Sensitivity | Specificity |\n")
            f.write("|---|---:|---|:---:|:---:|:---:|:---:|:---:|\n")
            
            for sub_name, results in sub_dict.items():
                size = results['sample_size']
                
                # Model A
                mA = results['ModelA_uncal_default']
                f.write(f"| **{sub_name}** | {size:,} | Model A (Baseline, 0.5) | {mA['macro_auroc']:.4f} | {mA['macro_auprc']:.4f} | {mA['macro_f1']:.4f} | {mA['macro_sensitivity']:.4f} | {mA['macro_specificity']:.4f} |\n")
                # Model B Youden
                mBy = results['ModelB_cal_youden']
                f.write(f"| | | Model B (Calibrated, Youden) | {mBy['macro_auroc']:.4f} | {mBy['macro_auprc']:.4f} | {mBy['macro_f1']:.4f} | {mBy['macro_sensitivity']:.4f} | {mBy['macro_specificity']:.4f} |\n")
                # Model B Sens80
                mBs = results['ModelB_cal_sens80']
                f.write(f"| | | Model B (Calibrated, Sens80) | {mBs['macro_auroc']:.4f} | {mBs['macro_auprc']:.4f} | {mBs['macro_f1']:.4f} | {mBs['macro_sensitivity']:.4f} | {mBs['macro_specificity']:.4f} |\n")
                
            f.write("\n")
            
    print(f"Subgroup report markdown saved to: {md_path}")
    print("==================================================")
    print("            SUBGROUP ANALYSIS COMPLETE            ")
    print("==================================================")

if __name__ == "__main__":
    main()
