import os
import pandas as pd
import numpy as np

def main():
    print("==================================================")
    print("         VINDR-CXR DATASET AUDITING               ")
    print("==================================================")
    
    project_dir = "/home/23adr188/chest_xray_project"
    dataset_root = os.path.join(project_dir, "data/external/vinbigdata")
    
    train_meta_path = os.path.join(dataset_root, "train_meta.csv")
    train_csv_path = os.path.join(dataset_root, "train.csv")
    train_images_dir = os.path.join(dataset_root, "train")
    test_images_dir = os.path.join(dataset_root, "test")
    
    # 1. Print discovered information (columns, directories, image counts)
    print("Discovered Dataset Structure:")
    print(f"  - Dataset Root: {dataset_root}")
    print(f"  - Train Image Directory: {train_images_dir}")
    print(f"  - Test Image Directory:  {test_images_dir}")
    
    # Image counts
    train_images_count = 0
    test_images_count = 0
    if os.path.exists(train_images_dir):
        train_images_count = len([f for f in os.listdir(train_images_dir) if f.lower().endswith('.png')])
    if os.path.exists(test_images_dir):
        test_images_count = len([f for f in os.listdir(test_images_dir) if f.lower().endswith('.png')])
        
    print(f"  - Discovered Train Images count: {train_images_count}")
    print(f"  - Discovered Test Images count:  {test_images_count}")
    
    # CSV file columns
    if os.path.exists(train_meta_path):
        meta_df = pd.read_csv(train_meta_path, nrows=5)
        print(f"  - Found train_meta.csv columns: {list(meta_df.columns)}")
    else:
        print("  - train_meta.csv not found!")
        
    if os.path.exists(train_csv_path):
        train_df = pd.read_csv(train_csv_path, nrows=5)
        print(f"  - Found train.csv columns: {list(train_df.columns)}")
    else:
        print("  - train.csv not found!")
        
    print("--------------------------------------------------")
    
    # 2. Check critical folder structures for audit
    if not os.path.exists(train_csv_path):
        print(f"ERROR: Annotation file train.csv not found at {train_csv_path}")
        return
    if not os.path.exists(train_images_dir):
        print(f"ERROR: Train image directory not found at {train_images_dir}")
        return
        
    print("Running full audit...")
    print("Verifying image files on disk...")
    physical_images = {f for f in os.listdir(train_images_dir) if f.lower().endswith('.png')}
    
    # 3. Parse annotations
    print("Parsing annotations from train.csv...")
    df = pd.read_csv(train_csv_path)
    print(f"Total annotation rows: {len(df)}")
    
    # Define targets
    diseases = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia', 'No Finding'
    ]
    
    disease_map = {
        'Atelectasis': ['Atelectasis'],
        'Cardiomegaly': ['Cardiomegaly'],
        'Pleural effusion': ['Effusion'],
        'Infiltration': ['Infiltration'],
        'Consolidation': ['Consolidation'],
        'Pneumothorax': ['Pneumothorax'],
        'Pulmonary fibrosis': ['Fibrosis'],
        'Pleural thickening': ['Pleural_Thickening'],
        'No finding': ['No Finding']
    }
    
    # Group annotations by image_id to build multi-label matrix (logical OR consensus)
    print("Grouping annotations to build multi-label consensus matrix...")
    grouped = df.groupby('image_id')
    
    processed_rows = []
    for image_id, group in grouped:
        row = {d: 0 for d in diseases}
        row['image_id'] = image_id
        row['Image Index'] = f"{image_id}.png"
        row['Image Path'] = os.path.abspath(os.path.join(train_images_dir, f"{image_id}.png"))
        
        # Get unique findings annotated by any radiologist for this image
        findings = group['class_name'].dropna().unique()
        
        for finding in findings:
            if finding in disease_map:
                for nih_class in disease_map[finding]:
                    row[nih_class] = 1
                    
        # Check if the physical file exists
        if f"{image_id}.png" in physical_images:
            processed_rows.append(row)
            
    processed_df = pd.DataFrame(processed_rows)
    print(f"Processed consensus metadata for {len(processed_df)} images.")
    
    # Write metadata
    out_csv = os.path.join(project_dir, "data/external/vinbigdata_metadata.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    processed_df.to_csv(out_csv, index=False)
    print(f"Consensus metadata CSV written to: {out_csv}")
    
    print("\nAuditing Stats:")
    print(f"  - Total unique images mapped: {len(processed_df)}")
    print(f"  - Excluded/Missing physical images: {len(grouped) - len(processed_df)}")
    for d in diseases:
        pos_count = processed_df[d].sum() if d in processed_df.columns else 0
        print(f"  - {d}: {pos_count} positive cases ({pos_count/len(processed_df)*100:.2f}%)")
        
    print("==================================================")
    print("AUDIT COMPLETE.")

if __name__ == "__main__":
    main()
