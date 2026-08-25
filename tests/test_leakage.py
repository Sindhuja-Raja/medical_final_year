import os
import pandas as pd

def test_patient_leakage():
    """
    Asserts that the training, validation, and testing patient sets are strictly disjoint.
    """
    project_dir = "/home/23adr188/chest_xray_project"
    splits_dir = os.path.join(project_dir, "data/splits")
    
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")
    
    assert os.path.exists(train_csv), "train.csv is missing"
    assert os.path.exists(val_csv), "val.csv is missing"
    assert os.path.exists(test_csv), "test.csv is missing"
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    train_patients = set(train_df['Patient ID'].unique())
    val_patients = set(val_df['Patient ID'].unique())
    test_patients = set(test_df['Patient ID'].unique())
    
    # Assert zero patient intersection
    train_val_overlap = train_patients.intersection(val_patients)
    train_test_overlap = train_patients.intersection(test_patients)
    val_test_overlap = val_patients.intersection(test_patients)
    
    assert len(train_val_overlap) == 0, f"Leakage detected between Train and Val sets: {len(train_val_overlap)} overlapping patients"
    assert len(train_test_overlap) == 0, f"Leakage detected between Train and Test sets: {len(train_test_overlap)} overlapping patients"
    assert len(val_test_overlap) == 0, f"Leakage detected between Val and Test sets: {len(val_test_overlap)} overlapping patients"
