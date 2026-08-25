import os
import sys
import time

def main():
    print("==================================================")
    print("   DOWNLOADING NIH CHESTX-RAY14 VIA KAGGLEHUB    ")
    print("==================================================")
    
    raw_dir = "/home/23adr188/chest_xray_project/data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    # Configure kagglehub to download directly into data/raw/
    os.environ["KAGGLEHUB_CACHE"] = raw_dir
    print(f"KAGGLEHUB_CACHE set to: {raw_dir}")
    
    start_time = time.time()
    try:
        import kagglehub
        print("Initiating dataset download (this might take several minutes)...")
        path = kagglehub.dataset_download("nih-chest-xrays/data")
        print("\nDownload and extraction completed successfully!")
        print(f"Dataset path: {path}")
        print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: Failed to download dataset: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
