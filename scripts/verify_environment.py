#!/usr/bin/env python3
import sys
import importlib

def main():
    print("==================================================")
    print("          ENVIRONMENT VERIFICATION SCRIPT         ")
    print("==================================================")
    
    # 1. Check Python Version
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    # 2. Verify PyTorch & CUDA
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_avail}")
        if cuda_avail:
            device_count = torch.cuda.device_count()
            print(f"GPU Count: {device_count}")
            for i in range(device_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Simple computation check on all GPUs
            for i in range(device_count):
                device = torch.device(f"cuda:{i}")
                x = torch.randn(10, 10, device=device)
                y = torch.matmul(x, x)
                print(f"  GPU {i} Computation Test: SUCCESS (shape {y.shape})")
        else:
            print("ERROR: CUDA is not available in PyTorch build.")
            sys.exit(1)
    except ImportError as e:
        print(f"ERROR: PyTorch could not be imported: {e}")
        sys.exit(1)

    # 3. Check Required Imports
    imports_to_test = [
        ("torchvision", "torchvision"),
        ("timm", "timm"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("OpenCV", "cv2"),
        ("Pillow (PIL)", "PIL"),
        ("matplotlib", "matplotlib"),
        ("pytest", "pytest"),
        ("scikit-learn", "sklearn")
    ]
    
    print("\nVerifying package imports:")
    failed = False
    for name, module_name in imports_to_test:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  [PASS] {name:<15} : Version {version}")
        except ImportError as e:
            print(f"  [FAIL] {name:<15} : {e}")
            failed = True
            
    print("==================================================")
    if failed:
        print("Verification RESULT: FAILED (missing packages)")
        sys.exit(1)
    else:
        print("Verification RESULT: SUCCESS")
        sys.exit(0)

if __name__ == '__main__':
    main()
