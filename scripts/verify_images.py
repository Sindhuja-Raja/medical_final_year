import os
import glob
import pandas as pd
from PIL import Image
import time
import json

def main():
    print("==================================================")
    print("         NIH CHESTX-RAY14 IMAGE INTEGRITY SCAN    ")
    print("==================================================")
    
    csv_path = "/home/23adr188/chest_xray_project/data/raw/datasets/nih-chest-xrays/data/versions/3/Data_Entry_2017.csv"
    dataset_root = "/home/23adr188/chest_xray_project/data/raw/datasets/nih-chest-xrays/data/versions/3"
    
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV path {csv_path} does not exist.")
        return
        
    # 1. Map physical images recursively
    print("Mapping physical image paths recursively...")
    start_map = time.time()
    physical_images = {}
    for root, dirs, files in os.walk(dataset_root):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                physical_images[f] = os.path.join(root, f)
    print(f"Mapped {len(physical_images)} images on disk in {time.time() - start_map:.2f} seconds.")
    
    # 2. Check CSV matching
    df = pd.read_csv(csv_path)
    csv_filenames = set(df['Image Index'].unique())
    print(f"Total unique filenames in CSV: {len(csv_filenames)}")
    
    missing_on_disk = csv_filenames - set(physical_images.keys())
    if missing_on_disk:
        print(f"ERROR: {len(missing_on_disk)} images in CSV are missing on disk.")
    else:
        print("All CSV filenames exist on disk.")
        
    # 3. Full integrity scan
    print("Scanning images for readability, size, and mode...")
    start_scan = time.time()
    corrupt_images = []
    unexpected_dims = []
    unexpected_modes = []
    
    count = 0
    total = len(physical_images)
    for filename, path in physical_images.items():
        count += 1
        if count % 20000 == 0:
            print(f"  Scanned {count}/{total} images ({count/total*100:.1f}%)...")
        try:
            with Image.open(path) as img:
                # Get dimensions and mode (read headers only)
                width, height = img.size
                mode = img.mode
                
                # Check for standard NIH size 1024x1024
                if width != 1024 or height != 1024:
                    unexpected_dims.append({
                        'filename': filename,
                        'path': path,
                        'size': [width, height]
                    })
                # Check for standard L or RGB modes
                if mode not in ('L', 'RGB'):
                    unexpected_modes.append({
                        'filename': filename,
                        'path': path,
                        'mode': mode
                    })
                # Verify file structural integrity (fast header check)
                img.verify()
        except Exception as e:
            corrupt_images.append({
                'filename': filename,
                'path': path,
                'error': str(e)
            })
            
    scan_time = time.time() - start_scan
    print(f"Scan complete in {scan_time:.2f} seconds.")
    
    summary = {
        'total_scanned': len(physical_images),
        'missing_on_disk_count': len(missing_on_disk),
        'corrupt_count': len(corrupt_images),
        'unexpected_dims_count': len(unexpected_dims),
        'unexpected_modes_count': len(unexpected_modes),
        'scan_time_seconds': scan_time,
        'corrupt_list': corrupt_images,
        'unexpected_dims_list': unexpected_dims[:50],  # cap dimensions log to prevent huge file
        'unexpected_modes_list': unexpected_modes
    }
    
    out_json = "/home/23adr188/chest_xray_project/docs/image_integrity_results.json"
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Integrity check summary written to {out_json}")

if __name__ == "__main__":
    main()
