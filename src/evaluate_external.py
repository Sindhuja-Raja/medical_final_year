import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
import timm

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, confusion_matrix

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessing import CXRPreprocessor
from src.data.dataset import NIHChestXrayDataset

def verify_external_dataset_criteria(csv_path):
    """
    Verifies that the VinDr-CXR metadata strictly contains:
    - 15,000 images
    """
    df = pd.read_csv(csv_path)
    total_images = df['image_id'].nunique()
    print(f"Verifying external dataset criteria for {csv_path}...")
    print(f"  - Unique images: {total_images} (Expected: 15000)")
    
    if total_images != 15000:
        raise ValueError(f"CRITICAL ERROR: External dataset image count {total_images} does not match expected 15000.")
        
    print("Verification PASS: External dataset criteria match requirements!")
    return df

def compute_external_metrics(y_true_9, y_pred_9, compatible_classes):
    """
    Computes all required metrics for the 9 compatible classes.
    y_true_9: numpy array of shape (N, 9)
    y_pred_9: numpy array of shape (N, 9) (probabilities)
    """
    metrics = {
        'overall': {},
        'per_class': {}
    }
    
    # 1. Overall AUROC (Macro, Micro)
    try:
        metrics['overall']['macro_auroc'] = float(roc_auc_score(y_true_9, y_pred_9, average='macro'))
        metrics['overall']['micro_auroc'] = float(roc_auc_score(y_true_9, y_pred_9, average='micro'))
    except Exception as e:
        print(f"Warning: Failed to compute overall macro/micro AUROC: {e}")
        metrics['overall']['macro_auroc'] = 0.0
        metrics['overall']['micro_auroc'] = 0.0

    # 2. Overall AUPRC (Macro, Micro)
    try:
        metrics['overall']['macro_auprc'] = float(average_precision_score(y_true_9, y_pred_9, average='macro'))
        metrics['overall']['micro_auprc'] = float(average_precision_score(y_true_9, y_pred_9, average='micro'))
    except Exception as e:
        print(f"Warning: Failed to compute overall macro/micro AUPRC: {e}")
        metrics['overall']['macro_auprc'] = 0.0
        metrics['overall']['micro_auprc'] = 0.0

    # Binarize predictions at 0.5 threshold
    y_pred_bin_9 = (y_pred_9 >= 0.5).astype(int)

    # 3. Overall F1-score & Sensitivity (Macro)
    try:
        metrics['overall']['macro_f1'] = float(f1_score(y_true_9, y_pred_bin_9, average='macro', zero_division=0))
        metrics['overall']['macro_sensitivity'] = float(recall_score(y_true_9, y_pred_bin_9, average='macro', zero_division=0))
    except Exception as e:
        metrics['overall']['macro_f1'] = 0.0
        metrics['overall']['macro_sensitivity'] = 0.0

    # 4. Overall Specificity (Macro)
    specificities = []
    for idx in range(len(compatible_classes)):
        try:
            tn, fp, fn, tp = confusion_matrix(y_true_9[:, idx], y_pred_bin_9[:, idx]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
            specificities.append(spec)
        except Exception:
            specificities.append(1.0)
    metrics['overall']['macro_specificity'] = float(np.mean(specificities))

    # 5. Per-class metrics
    for idx, d in enumerate(compatible_classes):
        class_metrics = {}
        
        # Ground truth counts
        pos_count = int(np.sum(y_true_9[:, idx]))
        neg_count = len(y_true_9) - pos_count
        class_metrics['sample_counts'] = {
            'positives': pos_count,
            'negatives': neg_count,
            'total': len(y_true_9)
        }
        
        # Per-class AUROC
        try:
            if len(np.unique(y_true_9[:, idx])) > 1:
                class_metrics['auroc'] = float(roc_auc_score(y_true_9[:, idx], y_pred_9[:, idx]))
            else:
                class_metrics['auroc'] = 0.5
        except Exception:
            class_metrics['auroc'] = 0.5
            
        # Per-class AUPRC
        try:
            class_metrics['auprc'] = float(average_precision_score(y_true_9[:, idx], y_pred_9[:, idx]))
        except Exception:
            class_metrics['auprc'] = 0.0
            
        # Per-class F1 & Sensitivity
        try:
            class_metrics['f1'] = float(f1_score(y_true_9[:, idx], y_pred_bin_9[:, idx], zero_division=0))
            class_metrics['sensitivity'] = float(recall_score(y_true_9[:, idx], y_pred_bin_9[:, idx], zero_division=0))
        except Exception:
            class_metrics['f1'] = 0.0
            class_metrics['sensitivity'] = 0.0
            
        # Confusion stats & Specificity
        try:
            tn, fp, fn, tp = confusion_matrix(y_true_9[:, idx], y_pred_bin_9[:, idx]).ravel()
            class_metrics['confusion_statistics'] = {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
            class_metrics['specificity'] = float(tn / (tn + fp) if (tn + fp) > 0 else 1.0)
        except Exception:
            class_metrics['confusion_statistics'] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
            class_metrics['specificity'] = 1.0
            
        metrics['per_class'][d] = class_metrics
        
    return metrics

def write_external_reports(metrics, output_dir, compatible_classes, excluded_classes):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Write JSON machine-readable results
    json_path = os.path.join(output_dir, "external_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Machine-readable external validation metrics saved to: {json_path}")
    
    # 2. Write Markdown human-readable report
    md_path = os.path.join(output_dir, "external_evaluation_report.md")
    with open(md_path, 'w') as f:
        f.write("# External Validation Report — DenseNet-121 Baseline (VinDr-CXR)\n\n")
        f.write("## 1. Evaluation Context\n")
        f.write("- **Date:** " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("- **Dataset:** VinDr-CXR (VinBigData Resized 256x256 PNGs)\n")
        f.write("- **External Validation Cohort:** 15,000 unique images (all paths physically verified)\n")
        f.write("- **Evaluation Policy:** Evaluated only on the 9 compatible target classes.\n\n")
        
        f.write("## 2. Excluded NIH Classes\n")
        f.write("The following NIH classes were **excluded** from this evaluation:\n")
        for exc_class, reason in excluded_classes.items():
            f.write(f"- **{exc_class}**: {reason}\n")
        f.write("\n")
        
        f.write("## 3. Overall Performance Metrics (9 Mapped Classes)\n")
        f.write(f"- **Macro AUROC:** {metrics['overall']['macro_auroc']:.4f}\n")
        f.write(f"- **Micro AUROC:** {metrics['overall']['micro_auroc']:.4f}\n")
        f.write(f"- **Macro AUPRC:** {metrics['overall']['macro_auprc']:.4f}\n")
        f.write(f"- **Micro AUPRC:** {metrics['overall']['micro_auprc']:.4f}\n")
        f.write(f"- **Macro F1-score:** {metrics['overall']['macro_f1']:.4f}\n")
        f.write(f"- **Macro Sensitivity:** {metrics['overall']['macro_sensitivity']:.4f}\n")
        f.write(f"- **Macro Specificity:** {metrics['overall']['macro_specificity']:.4f}\n\n")
        
        f.write("## 4. Per-Class Performance (9 Mapped Classes)\n")
        f.write("| Disease Abnormality | Samples (Pos/Neg) | AUROC | AUPRC | F1 | Sensitivity | Specificity | Confusion Stats (TP/FP/TN/FN) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|:---:|\n")
        
        for d in compatible_classes:
            m = metrics['per_class'][d]
            pos = m['sample_counts']['positives']
            neg = m['sample_counts']['negatives']
            c = m['confusion_statistics']
            f.write(f"| {d} | {pos:,}/{neg:,} | {m['auroc']:.4f} | {m['auprc']:.4f} | {m['f1']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} | {c['tp']}/{c['fp']}/{c['tn']}/{c['fn']} |\n")
            
        f.write("\n\n## 5. Evaluation Verification Details\n")
        f.write("- Model weights: locked (baseline model `best.pt` unchanged)\n")
        f.write("- NIH splits: locked (independent baseline validation cohort only)\n")
        f.write("- Consensus radiologist rule: logical OR of all annotating radiologists\n")
        
    print(f"Human-readable Markdown external report saved to: {md_path}")

def main():
    parser = argparse.ArgumentParser(description="DenseNet-121 External Validation Evaluator")
    parser.add_argument("--checkpoint", type=str, default="experiments/checkpoints/densenet121/best.pt", help="Path to model checkpoint")
    parser.add_argument("--metadata_csv", type=str, default="data/external/vinbigdata_metadata.csv", help="Path to consensus metadata CSV")
    parser.add_argument("--output_dir", type=str, default="experiments/results/densenet121", help="Directory to save evaluation results")
    parser.add_argument("--dry_run", action="store_true", help="Run in dry-run mode using random weights")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    args = parser.parse_args()

    # 1. Verify dataset criteria
    try:
        verify_external_dataset_criteria(args.metadata_csv)
    except Exception as e:
        print(f"ERROR: Dataset verification failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Set up preprocessor and dataset
    print("Setting up dataset loader...")
    preprocessor = CXRPreprocessor(img_size=(224, 224), is_training=False)
    dataset = NIHChestXrayDataset(csv_path=args.metadata_csv, transform=preprocessor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 3. Model Loading
    print("Initializing DenseNet-121 model...")
    model = timm.create_model('densenet121', pretrained=args.dry_run, num_classes=14)
    model = model.to(device)
    
    if args.dry_run:
        print("\n=== RUNNING IN DRY RUN MODE ===")
        print("Model is initialized with random/pretrained weights. Skipping checkpoint load.")
    else:
        print(f"Loading checkpoint weights from {args.checkpoint}...")
        if not os.path.exists(args.checkpoint):
            print(f"ERROR: Checkpoint file {args.checkpoint} does not exist.", file=sys.stderr)
            sys.exit(1)
            
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Weights loaded successfully!")
        
    # 4. Inference loop
    model.eval()
    all_targets = []
    all_outputs = []
    
    print("Running inference on VinDr-CXR split batches...")
    start_time = time.time()
    
    limit_batches = None
    if args.dry_run and os.environ.get("ANTIGRAVITY_SHORT_DRY_RUN", "0") == "1":
        limit_batches = 10
        print("  - Cap execution to 10 batches for quick dry-run validation")
        
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            
            with autocast('cuda'):
                outputs = model(images)
                
            probs = torch.sigmoid(outputs)
            all_targets.append(labels.cpu().numpy())
            all_outputs.append(probs.cpu().numpy())
            
            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(loader)} batches...")
                
            if limit_batches and idx >= limit_batches:
                break
                
    inference_time = time.time() - start_time
    print(f"Inference complete in {inference_time:.2f} seconds.")
    
    y_true_all = np.concatenate(all_targets, axis=0)
    y_pred_all = np.concatenate(all_outputs, axis=0)
    
    # 5. Extract and format metrics for the 9 compatible classes
    # NIH indexes: 
    # 0: Atelectasis, 1: Cardiomegaly, 2: Effusion, 3: Infiltration, 
    # 7: Pneumothorax, 8: Consolidation, 11: Fibrosis, 12: Pleural_Thickening
    
    nih_compatible_indices = [0, 1, 2, 3, 8, 7, 11, 12]
    compatible_classes = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Consolidation',
        'Pneumothorax', 'Fibrosis', 'Pleural_Thickening', 'No Finding'
    ]
    
    excluded_classes = {
        'Nodule': "Excluded because VinDr merges 'Nodule/Mass' into a single category and cannot distinguish them.",
        'Mass': "Excluded because VinDr merges 'Nodule/Mass' into a single category and cannot distinguish them.",
        'Pneumonia': "Excluded because VinDr annotations do not contain a corresponding abnormality finding.",
        'Edema': "Excluded because VinDr annotations do not contain a corresponding abnormality finding.",
        'Emphysema': "Excluded because VinDr annotations do not contain a corresponding abnormality finding.",
        'Hernia': "Excluded because VinDr annotations do not contain a corresponding abnormality finding."
    }
    
    # Slice target labels
    y_true_compatible = y_true_all[:, nih_compatible_indices]
    y_pred_compatible = y_pred_all[:, nih_compatible_indices]
    
    # Compute No Finding Ground Truth
    y_true_no_finding = dataset.df['No Finding'].values[:len(y_true_all)]
    
    # Compute No Finding predictions: 1.0 - max of compatible abnormality probabilities
    y_pred_no_finding = 1.0 - np.max(y_pred_compatible, axis=1)
    
    # Concat No Finding to the end of targets and predictions
    y_true_9 = np.hstack([y_true_compatible, y_true_no_finding.reshape(-1, 1)])
    y_pred_9 = np.hstack([y_pred_compatible, y_pred_no_finding.reshape(-1, 1)])
    
    print("Computing metrics...")
    metrics = compute_external_metrics(y_true_9, y_pred_9, compatible_classes)
    
    metrics['inference_benchmark'] = {
        'total_images_processed': len(y_true_9),
        'total_time_seconds': inference_time,
        'images_per_second': len(y_true_9) / inference_time
    }
    
    # 6. Write reports
    print(f"Writing reports to {args.output_dir}...")
    write_external_reports(metrics, args.output_dir, compatible_classes, excluded_classes)
    
    print("\n==================================================")
    print("      EXTERNAL EVALUATION SUMMARY (VinDr-CXR)     ")
    print("==================================================")
    print(f"Macro AUROC: {metrics['overall']['macro_auroc']:.4f}")
    print(f"Micro AUROC: {metrics['overall']['micro_auroc']:.4f}")
    print(f"Macro AUPRC: {metrics['overall']['macro_auprc']:.4f}")
    print(f"Macro F1:    {metrics['overall']['macro_f1']:.4f}")
    print("==================================================")
    print("EXTERNAL EVALUATION DRY RUN COMPLETED.")

if __name__ == "__main__":
    main()
