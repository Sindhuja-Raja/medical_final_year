# Dataset Audit: NIH ChestX-ray14

## 1. Audit Summary
- **Total Rows in Metadata CSV:** 112120
- **Total Unique Images in CSV:** 112120
- **Total Physical Images on Disk:** 112120
- **Total Unique Patients in CSV:** 30805
- **Missing Images count:** 0
- **Corrupt/Unreadable Images Checked:** 20 verified (Corrupt count: 0)
- **Official Split Files Checked:** Yes (`train_val_list.txt`, `test_list.txt` exist)
- **Official Split Image Counts:** Train/Val: 86524, Test: 25596
- **Patient ID Overlap between splits:** 0 (Status: PASS)

## 2. Class Distribution
| Disease | Positive Counts | Percentage |
|---|---:|---:|
| Atelectasis | 11,559 | 10.31% |
| Cardiomegaly | 2,776 | 2.48% |
| Effusion | 13,317 | 11.88% |
| Infiltration | 19,894 | 17.74% |
| Mass | 5,782 | 5.16% |
| Nodule | 6,331 | 5.65% |
| Pneumonia | 1,431 | 1.28% |
| Pneumothorax | 5,302 | 4.73% |
| Consolidation | 4,667 | 4.16% |
| Edema | 2,303 | 2.05% |
| Emphysema | 2,516 | 2.24% |
| Fibrosis | 1,686 | 1.50% |
| Pleural_Thickening | 3,385 | 3.02% |
| Hernia | 227 | 0.20% |
| No Finding | 60,361 | 53.84% |

## 3. Split Analysis
- **Train/Val Images:** 86524 images, 28008 patients
- **Test Images:** 25596 images, 2797 patients
- **Patient Leakage Check:** PASS

Patients in Train/Val split and Test split are strictly disjoint.

## 4. Phase 1 Checkpoint Evaluation
- **Image count matches expected (~112,120):** PASS (Found: 112120)
- **Patient overlap is zero:** PASS
- **Phase 1 Checkpoint Status:** PASS
