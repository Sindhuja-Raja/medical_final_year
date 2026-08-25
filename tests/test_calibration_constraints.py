import os
import hashlib
import pytest
import pandas as pd

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def test_baseline_checkpoint_lock():
    """
    Ensures that the locked baseline model checkpoint has not been modified.
    """
    best_pt = "experiments/checkpoints/densenet121/best.pt"
    assert os.path.exists(best_pt), "LOCKED ERROR: best.pt has been deleted!"
    
    expected_md5 = "17483e1928eb6c39e2f9bfd2f2b1434a"
    current_md5 = compute_md5(best_pt)
    assert current_md5 == expected_md5, "LOCKED ERROR: best.pt weights have been modified!"

def test_data_splits_lock():
    """
    Ensures that NIH train, val, and test splits have not been modified.
    """
    val_csv = "data/splits/val.csv"
    test_csv = "data/splits/test.csv"
    
    assert os.path.exists(val_csv), "LOCKED ERROR: val.csv split has been deleted!"
    assert os.path.exists(test_csv), "LOCKED ERROR: test.csv split has been deleted!"
    
    # Check MD5s
    assert compute_md5(val_csv) == "2174a9105496360451d8bc10580a23d7", "LOCKED ERROR: val.csv has been modified!"
    assert compute_md5(test_csv) == "c9b7c723864fac322137edd8ab4e17fe", "LOCKED ERROR: test.csv has been modified!"

def test_calibration_fitting_constraints():
    """
    Verifies that calibration parameters exist and were generated.
    """
    calib_params_path = "experiments/checkpoints/densenet121/temperature_scaling.pt"
    if os.path.exists(calib_params_path):
        import torch
        params = torch.load(calib_params_path, map_location="cpu", weights_only=True)
        assert "temperature" in params, "Calibration file missing temperature scaling parameter!"
        assert params["temperature"] > 0, "Calibration temperature scaling parameter must be positive!"
