import os
import glob
import pandas as pd
import numpy as np
import yaml

def main():
    print("==================================================")
    print("          NIH CHESTX-RAY14 SPLIT CREATOR          ")
    print("==================================================")
    
    project_dir = "/home/23adr188/chest_xray_project"
    csv_path = os.path.join(project_dir, "data/raw/datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv")
    dataset_root = os.path.join(project_dir, "data/raw/datasets/nih-chest-xrays/data/versions/3")
    train_val_list_path = os.path.join(dataset_root, "train_val_list.txt")
    test_list_path = os.path.join(dataset_root, "test_list.txt")
    
    # 1. Load Data
    print("Loading metadata CSV...")
    df = pd.read_csv(csv_path)
    
    # 2. Map physical images recursively to get absolute paths
    print("Mapping physical image paths on disk...")
    physical_images = {}
    for root, dirs, files in os.walk(dataset_root):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                physical_images[f] = os.path.abspath(os.path.join(root, f))
                
    # Add Image Path column to dataframe
    df['Image Path'] = df['Image Index'].map(physical_images)
    
    # Check if there are any missing paths
    missing_paths = df['Image Path'].isnull().sum()
    if missing_paths > 0:
        print(f"WARNING: {missing_paths} images do not have physical file paths mapped.")
        # Try finding in raw/images or other folders
        
    # 3. Load official split lists
    with open(train_val_list_path, 'r') as f:
        train_val_filenames = set(line.strip() for line in f if line.strip())
    with open(test_list_path, 'r') as f:
        test_filenames = set(line.strip() for line in f if line.strip())
        
    # 4. Filter train/val and test dataframes
    train_val_df = df[df['Image Index'].isin(train_val_filenames)].copy()
    test_df = df[df['Image Index'].isin(test_filenames)].copy()
    
    print(f"Official splits count:\n  - Train/Val images: {len(train_val_df)}\n  - Test images: {len(test_df)}")
    
    # 5. Split train/val set into 90% train and 10% val at patient level
    print("Performing patient-safe 90/10 split on Train/Val set...")
    train_val_patients = sorted(list(train_val_df['Patient ID'].unique()))
    
    # Seed configuration
    seed = 42
    np.random.seed(seed)
    np.random.shuffle(train_val_patients)
    
    split_idx = int(0.90 * len(train_val_patients))
    train_patients = set(train_val_patients[:split_idx])
    val_patients = set(train_val_patients[split_idx:])
    
    # Filter datasets
    train_df = train_val_df[train_val_df['Patient ID'].isin(train_patients)].copy()
    val_df = train_val_df[train_val_df['Patient ID'].isin(val_patients)].copy()
    
    print(f"Created splits:")
    print(f"  - Train: {len(train_df)} images, {len(train_patients)} patients")
    print(f"  - Val:   {len(val_df)} images, {len(val_patients)} patients")
    print(f"  - Test:  {len(test_df)} images, {test_df['Patient ID'].nunique()} patients")
    
    # 6. Verify patient-safe disjointness
    test_patients = set(test_df['Patient ID'].unique())
    
    overlap_train_val = train_patients.intersection(val_patients)
    overlap_train_test = train_patients.intersection(test_patients)
    overlap_val_test = val_patients.intersection(test_patients)
    
    print(f"Patient ID Overlaps:")
    print(f"  - Train ∩ Val:  {len(overlap_train_val)}")
    print(f"  - Train ∩ Test: {len(overlap_train_test)}")
    print(f"  - Val ∩ Test:   {len(overlap_val_test)}")
    
    if len(overlap_train_val) > 0 or len(overlap_train_test) > 0 or len(overlap_val_test) > 0:
        print("ERROR: Patient leak detected!")
        sys.exit(1)
    else:
        print("Success: All splits are patient-disjoint!")
        
    # 7. Binarize labels in split datasets
    diseases = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    
    for split_df in (train_df, val_df, test_df):
        for disease in diseases:
            split_df[disease] = split_df['Finding Labels'].apply(lambda x: 1 if disease in str(x).split('|') else 0)
        split_df['No Finding'] = split_df['Finding Labels'].apply(lambda x: 1 if str(x) == 'No Finding' else 0)
        
    # 8. Save CSV files
    splits_dir = os.path.join(project_dir, "data/splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"CSVs written successfully to {splits_dir}")
    
    # 9. Update configs/dataset.yaml
    yaml_path = os.path.join(project_dir, "configs/dataset.yaml")
    
    # Load existing config
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Append splitting configs
    config['dataset']['splitting'] = {
        'random_seed': seed,
        'val_ratio': 0.10,
        'paths': {
            'train_csv': train_csv,
            'val_csv': val_csv,
            'test_csv': test_csv
        },
        'statistics': {
            'train': {
                'images': int(len(train_df)),
                'patients': int(len(train_patients))
            },
            'val': {
                'images': int(len(val_df)),
                'patients': int(len(val_patients))
            },
            'test': {
                'images': int(len(test_df)),
                'patients': int(test_df['Patient ID'].nunique())
            }
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"Updated dataset configuration in {yaml_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
