# Phase 2 Report: Preprocessing & Patient-Safe Split

This report details the execution and results of Phase 2, which includes the full image integrity scan, patient-safe dataset partitioning, preprocessing specifications, and testing verification.

## 1. Full Image Integrity Scan Results
We performed a full read/write validation on all 112,120 chest X-ray images on disk:
- **Total Images Scanned:** 112,120
- **Missing Images on Disk:** 0
- **Corrupt / Unreadable Images:** 0
- **Unexpected Dimensions (non-1024x1024):** 0
- **Unexpected Image Modes (non-L/RGB):** 519 images
  - **Details:** 519 images are stored in `RGBA` format instead of grayscale (`L`) or standard `RGB`.
  - **Handling:** These 519 images are handled safely by the model-agnostic `CXRPreprocessor` in `src/data/preprocessing.py`, which explicitly converts any non-RGB image to RGB mode prior to tensor transformations.

## 2. Dataset Split Statistics
Using a fixed random seed of `42`, we partitioned the training subset (`train_val_list.txt` containing 28,008 patients) into a **90% training / 10% validation** patient-safe split. The test split is mapped directly from the official `test_list.txt` subset.

| Split | Image Count | Patient Count | Patient Percentage (of total) |
|---|---:|---:|---:|
| **Train** | 77,588 | 25,207 | 81.8% |
| **Validation** | 8,936 | 2,801 | 9.1% |
| **Test** | 25,596 | 2,797 | 9.1% |
| **Total** | **112,120** | **30,805** | **100.0%** |

## 3. Label Distributions Across Splits
The following table lists the positive counts and class percentages for each of the 14 abnormalities plus "No Finding" across all three splits:

| Disease | Train Count | Train % | Val Count | Val % | Test Count | Test % |
|---|---:|---:|---:|---:|---:|---:|
| **Atelectasis** | 7,467 | 9.62% | 813 | 9.10% | 3,279 | 12.81% |
| **Cardiomegaly** | 1,554 | 2.00% | 153 | 1.71% | 1,069 | 4.18% |
| **Effusion** | 7,749 | 9.99% | 910 | 10.18% | 4,658 | 18.20% |
| **Infiltration** | 12,312 | 15.87% | 1,470 | 16.45% | 6,112 | 23.88% |
| **Mass** | 3,611 | 4.65% | 423 | 4.73% | 1,748 | 6.83% |
| **Nodule** | 4,218 | 5.44% | 490 | 5.48% | 1,623 | 6.34% |
| **Pneumonia** | 789 | 1.02% | 87 | 0.97% | 555 | 2.17% |
| **Pneumothorax** | 2,369 | 3.05% | 268 | 3.00% | 2,665 | 10.41% |
| **Consolidation** | 2,564 | 3.30% | 288 | 3.22% | 1,815 | 7.09% |
| **Edema** | 1,236 | 1.59% | 142 | 1.59% | 925 | 3.61% |
| **Emphysema** | 1,274 | 1.64% | 149 | 1.67% | 1,093 | 4.27% |
| **Fibrosis** | 1,139 | 1.47% | 112 | 1.25% | 435 | 1.70% |
| **Pleural_Thickening** | 2,025 | 2.61% | 217 | 2.43% | 1,143 | 4.47% |
| **Hernia** | 121 | 0.16% | 20 | 0.22% | 86 | 0.34% |
| **No Finding** | 45,306 | 58.39% | 5,194 | 58.12% | 9,861 | 38.53% |

## 4. Leakage Test Results
The automated leakage test `tests/test_leakage.py` was executed using `pytest` and successfully validated that there is zero patient ID overlap across any splits:
- **Train ∩ Val Overlap:** 0 patients
- **Train ∩ Test Overlap:** 0 patients
- **Val ∩ Test Overlap:** 0 patients
- **Status:** **PASS**

## 5. Splitting and Preprocessing Configuration
- **Random Seed:** `42` (recorded in `configs/dataset.yaml`)
- **Preprocessing Pipelines:**
  - **Training Mode:**
    - Resized to `(224, 224)` (Default, configurable at model runtime).
    - Random Horizontal Flip.
    - Random Rotation (up to 15 degrees).
    - Tensor normalization using ImageNet stats: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
  - **Inference/Validation Mode:**
    - Resized to `(224, 224)`.
    - Tensor normalization using ImageNet stats.
  - **Image Mode Normalization:** All images converted to 3-channel RGB.

## 6. DataLoader Smoke Test
The DataLoader test (`scripts/smoke_test_loader.py`) fetched batches successfully.
- **Batch Size:** 4
- **Image Tensor Shape:** `[4, 3, 224, 224]`
- **Label Tensor Shape:** `[4, 14]`
- **Data Types:** Both tensors are represented in `torch.float32`.
- **Status:** **PASS**

## 7. Remaining Warnings / Issues
- **GPU 0 Occupied:** GPU 0 continues to be heavily utilized by other system users' tasks. GPU 1 is idle and will be targeted for model training in Phase 3.
- **NFS Storage Quota:** NFS partitions do not have user-level quota limits queryable by local command-line tools. Standard filesystem stats report **136 GB** available, which is sufficient.
