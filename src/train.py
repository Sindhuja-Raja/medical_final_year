import os
import sys
import argparse
import random
import yaml
import time
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(y_true, y_pred):
    """
    Computes multiple classification metrics for multi-label chest X-ray targets.
    y_true: numpy array of shape (N, 14)
    y_pred: numpy array of shape (N, 14) (probabilities)
    """
    diseases = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    
    metrics = {}
    
    # 1. AUROC (Macro, Micro, Per-class)
    try:
        metrics['macro_auroc'] = float(roc_auc_score(y_true, y_pred, average='macro'))
        metrics['micro_auroc'] = float(roc_auc_score(y_true, y_pred, average='micro'))
    except Exception as e:
        print(f"Warning: Failed to compute macro/micro AUROC: {e}")
        metrics['macro_auroc'] = 0.0
        metrics['micro_auroc'] = 0.0
        
    per_class_auroc = {}
    for idx, d in enumerate(diseases):
        try:
            # Check if both classes are present in y_true for this disease
            if len(np.unique(y_true[:, idx])) > 1:
                per_class_auroc[d] = float(roc_auc_score(y_true[:, idx], y_pred[:, idx]))
            else:
                per_class_auroc[d] = 0.5
        except Exception:
            per_class_auroc[d] = 0.5
    metrics['per_class_auroc'] = per_class_auroc

    # 2. AUPRC (Macro, Micro, Per-class)
    try:
        metrics['macro_auprc'] = float(average_precision_score(y_true, y_pred, average='macro'))
        metrics['micro_auprc'] = float(average_precision_score(y_true, y_pred, average='micro'))
    except Exception as e:
        print(f"Warning: Failed to compute macro/micro AUPRC: {e}")
        metrics['macro_auprc'] = 0.0
        metrics['micro_auprc'] = 0.0
        
    per_class_auprc = {}
    for idx, d in enumerate(diseases):
        try:
            per_class_auprc[d] = float(average_precision_score(y_true[:, idx], y_pred[:, idx]))
        except Exception:
            per_class_auprc[d] = 0.0
    metrics['per_class_auprc'] = per_class_auprc

    # Binarize predictions at 0.5 threshold
    y_pred_bin = (y_pred >= 0.5).astype(int)

    # 3. Multilabel F1 (Macro)
    try:
        metrics['macro_f1'] = float(f1_score(y_true, y_pred_bin, average='macro', zero_division=0))
    except Exception:
        metrics['macro_f1'] = 0.0

    # 4. Sensitivity/Recall (Macro)
    try:
        metrics['macro_sensitivity'] = float(recall_score(y_true, y_pred_bin, average='macro', zero_division=0))
    except Exception:
        metrics['macro_sensitivity'] = 0.0

    # 5. Specificity (Macro)
    specificities = []
    for idx in range(len(diseases)):
        try:
            tn, fp, fn, tp = confusion_matrix(y_true[:, idx], y_pred_bin[:, idx]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
            specificities.append(spec)
        except Exception:
            specificities.append(1.0)
    metrics['macro_specificity'] = float(np.mean(specificities))
    metrics['per_class_specificity'] = {d: float(specificities[idx]) for idx, d in enumerate(diseases)}
    
    return metrics

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Mixed precision context
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        
    return running_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        running_loss += loss.item() * images.size(0)
        
        # Convert sigmoid probabilities
        probs = torch.sigmoid(outputs)
        all_targets.append(labels.cpu().numpy())
        all_outputs.append(probs.cpu().numpy())
        
    val_loss = running_loss / len(loader.dataset)
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_outputs, axis=0)
    
    metrics = compute_metrics(y_true, y_pred)
    metrics['loss'] = float(val_loss)
    
    return metrics

def save_checkpoint(state, checkpoint_dir, filename="last.pt"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="DenseNet-121 Baseline Trainer")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override maximum epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Overrides
    if args.epochs is not None:
        config['training']['max_epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
        
    # Seed everything
    seed = config['training'].get('random_seed', 42)
    seed_everything(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths config
    checkpoint_dir = config['paths']['checkpoint_dir']
    log_dir = config['paths']['log_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. Dataset & DataLoader setup
    print("Setting up datasets and data loaders...")
    train_preprocessor = CXRPreprocessor(img_size=tuple(config['training']['img_size']), is_training=True)
    val_preprocessor = CXRPreprocessor(img_size=tuple(config['training']['img_size']), is_training=False)
    
    train_dataset = NIHChestXrayDataset(csv_path=config['paths']['train_csv'], transform=train_preprocessor)
    val_dataset = NIHChestXrayDataset(csv_path=config['paths']['val_csv'], transform=val_preprocessor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    # 2. Model Initialization
    print(f"Initializing {config['model']['name']} model...")
    model = timm.create_model(
        config['model']['name'], 
        pretrained=config['model']['pretrained'], 
        num_classes=config['model']['num_classes']
    )
    model = model.to(device)
    
    # Use BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    scaler = GradScaler(enabled=config['training'].get('amp_enabled', True))
    
    start_epoch = 0
    best_val_macro_auroc = -1.0
    early_stopping_counter = 0
    history = []
    
    # Resuming logic
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_macro_auroc = checkpoint['best_val_macro_auroc']
            early_stopping_counter = checkpoint['early_stopping_counter']
            history = checkpoint.get('history', [])
            
            # Restore RNG states
            random.setstate(checkpoint['rng_states']['python'])
            np.random.set_state(checkpoint['rng_states']['numpy'])
            torch.set_rng_state(checkpoint['rng_states']['torch'].cpu())
            if torch.cuda.is_available() and checkpoint['rng_states']['torch_cuda'] is not None:
                torch.cuda.set_rng_state_all([t.cpu() for t in checkpoint['rng_states']['torch_cuda']])
                
            print(f"Resumed from epoch {start_epoch} (Best Val Macro AUROC: {best_val_macro_auroc:.4f})")
        else:
            print(f"ERROR: Checkpoint file {args.resume} not found. Starting from scratch.")
            
    # Config parameters for early stopping
    max_epochs = config['training']['max_epochs']
    patience = config['training'].get('early_stopping_patience', 7)
    min_delta = config['training'].get('min_delta', 0.0001)
    
    # Dry-run check for user
    if max_epochs == 0:
        print("Dry run validation check...")
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Dry run metrics: Loss: {val_metrics['loss']:.4f}, Macro AUROC: {val_metrics['macro_auroc']:.4f}")
        return
        
    print("Starting training loop...")
    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{max_epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Step LR scheduler using validation Macro AUROC
        scheduler.step(val_metrics['macro_auroc'])
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Macro AUROC: {val_metrics['macro_auroc']:.4f} | Val Micro AUROC: {val_metrics['micro_auroc']:.4f}")
        print(f"  Val Macro AUPRC: {val_metrics['macro_auprc']:.4f} | Val Micro AUPRC: {val_metrics['micro_auprc']:.4f}")
        print(f"  Val Macro F1:    {val_metrics['macro_f1']:.4f} | Val Macro Sensitivity: {val_metrics['macro_sensitivity']:.4f}")
        
        # Record history
        epoch_record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'time_taken_seconds': epoch_time
        }
        history.append(epoch_record)
        
        # Log to file
        log_file = os.path.join(log_dir, "metrics.json")
        with open(log_file, "w") as f:
            json.dump(history, f, indent=2)
            
        # Get RNG states
        rng_states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        
        # Checkpoint state dictionary
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_macro_auroc': best_val_macro_auroc,
            'early_stopping_counter': early_stopping_counter,
            'history': history,
            'rng_states': rng_states,
            'config': config
        }
        
        # Save last.pt
        save_checkpoint(state, checkpoint_dir, filename="last.pt")
        
        # Check validation improvement for early stopping
        val_macro_auroc = val_metrics['macro_auroc']
        improvement = val_macro_auroc - best_val_macro_auroc
        
        if improvement >= min_delta:
            print(f"Val Macro AUROC improved from {best_val_macro_auroc:.4f} to {val_macro_auroc:.4f}.")
            best_val_macro_auroc = val_macro_auroc
            early_stopping_counter = 0
            
            # Update best_val_macro_auroc in the saved checkpoint state dictionary and save best.pt
            state['best_val_macro_auroc'] = best_val_macro_auroc
            save_checkpoint(state, checkpoint_dir, filename="best.pt")
        else:
            early_stopping_counter += 1
            print(f"No significant improvement. Early stopping counter: {early_stopping_counter}/{patience}")
            
        if early_stopping_counter >= patience:
            print(f"\nEarly stopping triggered! Training stopped after {epoch+1} epochs.")
            print(f"Best Validation Macro AUROC: {best_val_macro_auroc:.4f}")
            break
            
    print("\nTraining completed.")

if __name__ == "__main__":
    main()
