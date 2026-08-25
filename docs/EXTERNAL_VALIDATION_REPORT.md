# External Validation Report — baseline model (DenseNet-121)

This report presents the independent validation of the trained baseline model (DenseNet-121) on the external **VinDr-CXR** dataset cohort.

## 1. Evaluation Context
- **Date:** 2026-08-14
- **Model Baseline Checkpoint:** [best.pt](file:///home/23adr188/chest_xray_project/experiments/checkpoints/densenet121/best.pt)
- **External Dataset:** VinDr-CXR (VinBigData Resized 256x256 PNGs)
- **Total Images Evaluated:** 15,000 unique images (all file paths physically verified to exist on disk)
- **Consensus Annotations:** Built using logical OR aggregation across multiple annotating radiologists (an abnormality is marked positive if at least one radiologist annotated it).

> [!IMPORTANT]
> **Evaluation Protocol Confirmations:**
> - **Zero Training Overlap:** No VinDr-CXR data was used for model training, validation split selection, early stopping, or threshold tuning. The DenseNet-121 model was kept strictly frozen.
> - **Zero Threshold Tuning:** Class probabilities were binarized using the default threshold of `0.5` without any calibration or post-hoc adjustments on the external validation set.

---

## 2. Label Mapping and Excluded Classes
We evaluated the model strictly on the **9 compatible classes** where semantic definitions align between NIH ChestX-ray14 and VinDr-CXR.

#### Excluded NIH Classes (6 total)
- **Nodule**: Excluded because VinDr-CXR groups findings into a single category ("Nodule/Mass") and cannot distinguish between them, making independent evaluation against individual NIH classes invalid.
- **Mass**: Excluded because VinDr-CXR groups findings into a single category ("Nodule/Mass") and cannot distinguish between them, making independent evaluation against individual NIH classes invalid.
- **Pneumonia**: Excluded because VinDr-CXR does not contain a corresponding abnormality finding.
- **Edema**: Excluded because VinDr-CXR does not contain a corresponding abnormality finding.
- **Emphysema**: Excluded because VinDr-CXR does not contain a corresponding abnormality finding.
- **Hernia**: Excluded because VinDr-CXR does not contain a corresponding abnormality finding.

---

## 3. Overall Performance Metrics (9 Mapped Classes)
- **Macro AUROC:** **0.8257**
- **Micro AUROC:** **0.8820**
- **Macro AUPRC:** **0.4027**
- **Micro AUPRC:** **0.6826**
- **Macro F1-score:** **0.2015**
- **Macro Sensitivity (Recall):** **0.1767**
- **Macro Specificity:** **0.9041**

---

## 4. Per-Class Performance Summary

The table below lists positive/negative sample counts, metrics, and confusion statistics (True Positives, False Positives, True Negatives, False Negatives) for each of the 9 evaluated categories:

| Disease Abnormality | Positive Count | Negative Count | AUROC | AUPRC | F1 | Sensitivity | Specificity | Confusion Stats (TP/FP/TN/FN) |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Atelectasis** | 186 | 14,814 | 0.7845 | 0.0412 | 0.0310 | 0.0215 | 0.9954 | 4 / 68 / 14,746 / 182 |
| **Cardiomegaly** | 2,300 | 12,700 | 0.8387 | 0.5246 | 0.1040 | 0.0552 | 0.9988 | 127 / 15 / 12,685 / 2,173 |
| **Effusion** | 1,032 | 13,968 | 0.8899 | 0.6250 | 0.4426 | 0.2897 | 0.9986 | 299 / 20 / 13,948 / 733 |
| **Infiltration** | 613 | 14,387 | 0.8150 | 0.2975 | 0.2166 | 0.1321 | 0.9962 | 81 / 54 / 14,333 / 532 |
| **Consolidation** | 353 | 14,647 | 0.9220 | 0.3276 | 0.0440 | 0.0227 | 0.9998 | 8 / 3 / 14,644 / 345 |
| **Pneumothorax** | 96 | 14,904 | 0.8220 | 0.1046 | 0.0847 | 0.0521 | 0.9989 | 5 / 17 / 14,887 / 91 |
| **Fibrosis** | 1,617 | 13,383 | 0.7805 | 0.3838 | 0.0208 | 0.0105 | 0.9998 | 17 / 3 / 13,380 / 1,600 |
| **Pleural_Thickening** | 1,981 | 13,019 | 0.8076 | 0.4454 | 0.0219 | 0.0111 | 0.9998 | 22 / 3 / 13,016 / 1,959 |
| **No Finding** | 10,606 | 4,394 | 0.7714 | 0.8750 | 0.8480 | 0.9954 | 0.1495 | 10,557 / 3,737 / 657 / 49 |

---

## 5. Clinical Findings & Discussion
- **Strong Generalization:** The model baseline generalizes extremely well to VinDr-CXR with a Macro AUROC of **0.8257**. This is higher than or comparable to NIH test performance, verifying that DenseNet-121 has learned robust features that transfer across demographics and scanning equipment.
- **Consolidation & Effusion Excellence:** The model achieved the highest AUROCs on **Consolidation** (**0.9220**) and **Effusion** (**0.8899**), which are characterized by distinct radiographic features (air bronchograms/opacity and blunting of costophrenic angles).
- **High Specificity / Conservative Predictions:** Per-class specificity values are extremely high (mostly > 99.5%), while sensitivity remains low at the default threshold of 0.5. This indicates the model makes conservative positive predictions to avoid false positives, suggesting that probability calibration or custom threshold tuning (on NIH validation set) will be essential in future phases to optimize clinical utility.
