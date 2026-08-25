import os
import sys
import torch
from torch.utils.data import DataLoader

# Add project root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessing import CXRPreprocessor
from src.data.dataset import NIHChestXrayDataset

def main():
    print("==================================================")
    print("            DATALOADER SMOKE TEST                 ")
    print("==================================================")
    
    project_dir = "/home/23adr188/chest_xray_project"
    train_csv = os.path.join(project_dir, "data/splits/train.csv")
    
    if not os.path.exists(train_csv):
        print(f"ERROR: train.csv not found at {train_csv}")
        sys.exit(1)
        
    print("Initializing Preprocessor (Training Mode)...")
    preprocessor = CXRPreprocessor(img_size=(224, 224), is_training=True)
    
    print("Initializing NIHChestXrayDataset...")
    dataset = NIHChestXrayDataset(csv_path=train_csv, transform=preprocessor)
    print(f"Dataset initialized successfully with {len(dataset)} images.")
    
    print("Creating DataLoader...")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Try fetching one batch
    print("Fetching first batch from DataLoader...")
    try:
        iterator = iter(loader)
        images, labels = next(iterator)
        print("Success! Batch loaded.")
        print(f"  - Image tensor shape: {images.shape} (Expected: [4, 3, 224, 224])")
        print(f"  - Label tensor shape: {labels.shape} (Expected: [4, 14])")
        print(f"  - Image dtype:        {images.dtype}")
        print(f"  - Label dtype:        {labels.dtype}")
        
        # Validation checks
        assert images.shape == (4, 3, 224, 224), f"Unexpected image shape: {images.shape}"
        assert labels.shape == (4, 14), f"Unexpected label shape: {labels.shape}"
        assert images.dtype == torch.float32, f"Unexpected image dtype: {images.dtype}"
        assert labels.dtype == torch.float32, f"Unexpected label dtype: {labels.dtype}"
        
        print("\nVerification RESULT: SUCCESS")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: Failed to load batch or verification failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
