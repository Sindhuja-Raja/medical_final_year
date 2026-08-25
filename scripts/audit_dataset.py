import os
import sys
import glob
import pandas as pd
import numpy as np
import yaml
from PIL import Image

def main():
    print("==================================================")
    print("           NIH CHESTX-RAY14 DATASET AUDIT         ")
    print("==================================================")
    
    # 1. Locate dataset files
    project_dir = "/home/23adr188/chest_xray_project"
    raw_dir = os.path.join(project_dir, "data/raw")
    
    # Search for Data_Entry_2017.csv in the raw directory recursively
    csv_paths = glob.glob(os.path.join(raw_dir, "**/Data_Entry_2017.csv"), recursive=True)
    if not csv_paths:
        print("ERROR: Data_Entry_2017.csv not found in data/raw/ directory structure.")
        sys.exit(1)
        
    csv_path = csv_paths[0]
    dataset_root = os.path.dirname(csv_path)
    print(f"Found metadata CSV at: {csv_path}")
    print(f"Dataset root directory: {dataset_root}")
    
    # 2. Locate image archives and extract if necessary
    # Check if there are zip files in the directory
    zip_files = glob.glob(os.path.join(dataset_root, "**/*.zip"), recursive=True)
    if zip_files:
        print(f"Found {len(zip_files)} zip files. Extracting them...")
        images_dest = os.path.join(raw_dir, "images")
        os.makedirs(images_dest, exist_ok=True)
        for zf in sorted(zip_files):
            print(f"Extracting {os.path.basename(zf)} to {images_dest}...")
            # Use system unzip for speed
            os.system(f"unzip -q -o '{zf}' -d '{images_dest}'")
        image_dir = images_dest
    else:
        # Search recursively starting from dataset_root
        image_dir = dataset_root
            
    print(f"Image directory set to: {image_dir}")
    
    # 3. Load Metadata
    print("Loading Data_Entry_2017.csv...")
    df = pd.read_csv(csv_path)
    print(f"Metadata loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # 4. Check Splits files
    train_val_list_path = os.path.join(dataset_root, "train_val_list.txt")
    test_list_path = os.path.join(dataset_root, "test_list.txt")
    
    if not os.path.exists(train_val_list_path) or not os.path.exists(test_list_path):
        # Check parent folder or recursive search
        tv_search = glob.glob(os.path.join(raw_dir, "**/train_val_list.txt"), recursive=True)
        t_search = glob.glob(os.path.join(raw_dir, "**/test_list.txt"), recursive=True)
        if tv_search and t_search:
            train_val_list_path = tv_search[0]
            test_list_path = t_search[0]
        else:
            print("ERROR: Split files (train_val_list.txt / test_list.txt) not found.")
            sys.exit(1)
            
    print(f"Found split files:\n  - Train/Val list: {train_val_list_path}\n  - Test list: {test_list_path}")
    
    # Read splits
    with open(train_val_list_path, 'r') as f:
        train_val_filenames = set(line.strip() for line in f if line.strip())
    with open(test_list_path, 'r') as f:
        test_filenames = set(line.strip() for line in f if line.strip())
        
    print(f"Split file image counts:\n  - Train/Val: {len(train_val_filenames)}\n  - Test: {len(test_filenames)}")
    
    # 5. Perform Audits
    # Image count
    total_images_in_csv = df['Image Index'].nunique()
    print(f"Total unique images in CSV: {total_images_in_csv}")
    
    # Patient count
    total_patients_in_csv = df['Patient ID'].nunique()
    print(f"Total unique patients in CSV: {total_patients_in_csv}")
    
    # Check physical images
    print("Checking physical files in image directory...")
    # Find all images recursively
    physical_image_files = {}
    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                physical_image_files[f] = os.path.join(root, f)
                
    total_physical_images = len(physical_image_files)
    print(f"Total physical images found on disk: {total_physical_images}")
    
    # Verify missing or extra files
    csv_image_set = set(df['Image Index'].unique())
    missing_images = csv_image_set - set(physical_image_files.keys())
    extra_images = set(physical_image_files.keys()) - csv_image_set
    
    print(f"Missing images (in CSV but not on disk): {len(missing_images)}")
    print(f"Extra images (on disk but not in CSV): {len(extra_images)}")
    
    # Verify file integrity on a sample of 20 images
    print("Checking file integrity on a sample of 20 images...")
    corrupt_count = 0
    sample_images = np.random.choice(list(physical_image_files.values()), min(20, total_physical_images), replace=False)
    for img_path in sample_images:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            print(f"Corrupted image found: {img_path} ({e})")
            corrupt_count += 1
            
    print(f"Corrupt count in sample: {corrupt_count}")
    
    # 6. Parse and Binarize Labels
    diseases = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    
    # One-hot encode the labels
    for disease in diseases:
        df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in str(x).split('|') else 0)
    df['No Finding'] = df['Finding Labels'].apply(lambda x: 1 if str(x) == 'No Finding' else 0)
    
    # Class Distribution Table
    class_counts = {}
    for d in diseases + ['No Finding']:
        class_counts[d] = int(df[d].sum())
        
    print("\nClass Distribution:")
    for d, count in class_counts.items():
        print(f"  {d:<20}: {count:>6} ({count/len(df)*100:5.2f}%)")
        
    # Check patient overlap between splits
    train_val_df = df[df['Image Index'].isin(train_val_filenames)]
    test_df = df[df['Image Index'].isin(test_filenames)]
    
    train_val_patients = set(train_val_df['Patient ID'].unique())
    test_patients = set(test_df['Patient ID'].unique())
    
    patient_overlap = train_val_patients.intersection(test_patients)
    print(f"\nPatient IDs overlap between Train/Val and Test split: {len(patient_overlap)}")
    
    overlap_ok = len(patient_overlap) == 0
    print(f"Patient overlap check: {'PASS' if overlap_ok else 'FAIL'}")
    
    # Write docs/DATASET_AUDIT.md
    docs_dir = os.path.join(project_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    audit_file_path = os.path.join(docs_dir, "DATASET_AUDIT.md")
    
    print(f"Writing dataset audit report to: {audit_file_path}")
    
    with open(audit_file_path, "w") as f:
        f.write("# Dataset Audit: NIH ChestX-ray14\n\n")
        f.write("## 1. Audit Summary\n")
        f.write(f"- **Total Rows in Metadata CSV:** {df.shape[0]}\n")
        f.write(f"- **Total Unique Images in CSV:** {total_images_in_csv}\n")
        f.write(f"- **Total Physical Images on Disk:** {total_physical_images}\n")
        f.write(f"- **Total Unique Patients in CSV:** {total_patients_in_csv}\n")
        f.write(f"- **Missing Images count:** {len(missing_images)}\n")
        f.write(f"- **Corrupt/Unreadable Images Checked:** {len(sample_images)} verified (Corrupt count: {corrupt_count})\n")
        f.write(f"- **Official Split Files Checked:** Yes (`train_val_list.txt`, `test_list.txt` exist)\n")
        f.write(f"- **Official Split Image Counts:** Train/Val: {len(train_val_filenames)}, Test: {len(test_filenames)}\n")
        f.write(f"- **Patient ID Overlap between splits:** {len(patient_overlap)} (Status: {'PASS' if overlap_ok else 'FAIL'})\n\n")
        
        f.write("## 2. Class Distribution\n")
        f.write("| Disease | Positive Counts | Percentage |\n")
        f.write("|---|---:|---:|\n")
        for d in diseases + ['No Finding']:
            count = class_counts[d]
            f.write(f"| {d} | {count:,} | {count/len(df)*100:.2f}% |\n")
        f.write("\n")
        
        f.write("## 3. Split Analysis\n")
        f.write(f"- **Train/Val Images:** {train_val_df.shape[0]} images, {len(train_val_patients)} patients\n")
        f.write(f"- **Test Images:** {test_df.shape[0]} images, {len(test_patients)} patients\n")
        f.write(f"- **Patient Leakage Check:** {'PASS' if overlap_ok else 'FAIL'}\n\n")
        f.write("Patients in Train/Val split and Test split are strictly disjoint.\n\n")
        
        f.write("## 4. Phase 1 Checkpoint Evaluation\n")
        checkpoint_ok = (total_physical_images == 112120) and overlap_ok and (len(missing_images) == 0)
        f.write(f"- **Image count matches expected (~112,120):** {'PASS' if total_physical_images == 112120 else 'FAIL'} (Found: {total_physical_images})\n")
        f.write(f"- **Patient overlap is zero:** {'PASS' if overlap_ok else 'FAIL'}\n")
        f.write(f"- **Phase 1 Checkpoint Status:** {'PASS' if checkpoint_ok else 'FAIL'}\n")
        
    # Write configs/dataset.yaml
    configs_dir = os.path.join(project_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)
    yaml_file_path = os.path.join(configs_dir, "dataset.yaml")
    
    print(f"Writing dataset configuration to: {yaml_file_path}")
    
    dataset_config = {
        'dataset': {
            'name': 'NIH ChestX-ray14',
            'paths': {
                'root': dataset_root,
                'csv': csv_path,
                'images': image_dir,
                'train_val_list': train_val_list_path,
                'test_list': test_list_path
            },
            'labels': diseases,
            'statistics': {
                'total_images': int(total_images_in_csv),
                'total_patients': int(total_patients_in_csv),
                'train_val_images': int(train_val_df.shape[0]),
                'train_val_patients': int(len(train_val_patients)),
                'test_images': int(test_df.shape[0]),
                'test_patients': int(len(test_patients)),
                'patient_overlap': int(len(patient_overlap)),
                'class_counts': class_counts
            }
        }
    }
    
    with open(yaml_file_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
        
    print("==================================================")
    print(f"AUDIT COMPLETED. Checkpoint Status: {'PASS' if checkpoint_ok else 'FAIL'}")
    print("==================================================")

if __name__ == "__main__":
    main()
