import os
import shutil
import kagglehub

def get_folder_stats(folder_path):
    total_size = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
                    file_count += 1
            except Exception:
                pass
    return file_count, total_size

def main():
    print("==================================================")
    print("      DOWNLOADING VINDR-CXR (VINBIGDATA 256x256)   ")
    print("==================================================")
    
    # 1. Download dataset via kagglehub
    print("Downloading dataset from Kaggle...")
    cache_path = kagglehub.dataset_download("xhlulu/vinbigdata-chest-xray-resized-png-256x256")
    cache_path = os.path.abspath(cache_path)
    print(f"Dataset downloaded to cache at: {cache_path}")
    
    # Get stats of cache
    cache_count, cache_size = get_folder_stats(cache_path)
    print(f"Cache Stats: {cache_count} files, {cache_size / (1024**2):.2f} MB")
    
    # 2. Define target path
    target_path = "/home/23adr188/chest_xray_project/data/external/vinbigdata"
    os.makedirs(target_path, exist_ok=True)
    
    # 3. Copy files to project folder
    print(f"Copying files to project directory: {target_path}...")
    if os.path.exists(cache_path):
        for item in os.listdir(cache_path):
            s = os.path.join(cache_path, item)
            d = os.path.join(target_path, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        print("Copy completed.")
    else:
        print("ERROR: Cache download path not found.")
        return
        
    # 4. Verify copy
    target_count, target_size = get_folder_stats(target_path)
    print(f"Target Stats: {target_count} files, {target_size / (1024**2):.2f} MB")
    
    if cache_count == target_count and cache_size == target_size:
        print("Verification SUCCESS: File counts and sizes match perfectly!")
        
        # 5. Remove the Kaggle cache directory
        print("Cleaning up temporary Kaggle cache folder...")
        dataset_cache_root = os.path.abspath(os.path.join(cache_path, "../.."))
        if os.path.exists(dataset_cache_root):
            shutil.rmtree(dataset_cache_root)
            print("Temporary Kaggle cache folder successfully removed.")
        else:
            print("Warning: Could not locate dataset cache root for deletion.")
    else:
        print("ERROR: Verification FAILED! Copy is incomplete or corrupted.")
        print(f"  - File counts: cache={cache_count}, target={target_count}")
        print(f"  - Sizes (bytes): cache={cache_size}, target={target_size}")
        return
        
    # 6. Print final validation info
    print("\nFinal Dataset Validation:")
    print(f"  - Final Path:    {target_path}")
    print(f"  - File Count:    {target_count}")
    print(f"  - Dataset Size:  {target_size / (1024**2):.2f} MB")
    
    # Get remaining /home disk space
    usage = shutil.disk_usage("/home")
    print(f"  - Remaining /home disk space: {usage.free / (1024**3):.2f} GB")
    print("==================================================")

if __name__ == "__main__":
    main()
