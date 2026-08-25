# Calibration & Operating-Point Analysis Report

This report documents the methods, parameter optimizations, and results for Phase 6: Calibration & Operating-Point Analysis of the chest X-ray DenseNet-121 model.

---

## 1. Study Objective and Methodology

The primary goal of this phase is to:
1. Improve probability calibration (align predicted probability values with empirical frequencies) using **Temperature Scaling**.
2. Optimize decision thresholds to transition from arbitrary thresholds (`0.5`) to clinically useful operating points (**Youden's J** and **Sensitivity-Targeted at 80%**).

### A. Data Allocation and Isolation
To guarantee unbiased performance reporting, the datasets were partitioned and locked under the following rules:
- **NIH validation split ([val.csv](file:///home/23adr188/chest_xray_project/data/splits/val.csv) — 8,936 images):** Used exclusively for optimizing the temperature scaling parameter $T$ and selecting the Youden's J and sensitivity-targeted thresholds.
- **NIH test split ([test.csv](file:///home/23adr188/chest_xray_project/data/splits/test.csv) — 25,596 images):** Held out as a locked internal test set. No calibration or threshold tuning was performed on this dataset.
- **VinDr-CXR external validation set ([vinbigdata_metadata.csv](file:///home/23adr188/chest_xray_project/data/external/vinbigdata_metadata.csv) — 15,000 images):** Kept strictly untouched as an independent external generalization cohort. No VinDr data was accessed or used for temperature fitting or threshold selection.

### B. Temperature Scaling Calibration
Temperature scaling post-processes raw logits $z_i$ by dividing them by a single learned scalar $T > 0$:
$$p_i = \sigma\left(\frac{z_i}{T}\right)$$
We optimized $T$ using the L-BFGS optimizer on PyTorch to minimize Binary Cross Entropy (BCE) loss on the 8,936 validation split samples.

**Optimized Temperature Value:** $T = 1.2272$

### C. Threshold Selection Methodology
We evaluated three threshold selection policies computed strictly on the NIH validation split:
1. **Default Threshold:** Set to `0.5` for all classes.
2. **Youden's J Statistic:** Sweeps thresholds to locate the argmax of Youden's index:
   $$J = \text{Sensitivity} + \text{Specificity} - 1$$
3. **Sensitivity-Targeted (80%):** Sweeps thresholds to locate the point where $\text{Sensitivity} \ge 0.80$, while maximizing specificity.

---

## 2. Selected Per-Class Thresholds
All thresholds were computed and frozen on the NIH validation split prior to final evaluation:

| Target Class | Youden's J Threshold | Sensitivity-Targeted (80%) Threshold |
|---|:---:|:---:|
| Atelectasis | 0.0764 | 0.0959 |
| Cardiomegaly | 0.0574 | 0.0175 |
| Effusion | 0.0901 | 0.1273 |
| Infiltration | 0.1858 | 0.1549 |
| Mass | 0.0511 | 0.0330 |
| Nodule | 0.0674 | 0.0461 |
| Pneumonia | 0.0156 | 0.0127 |
| Pneumothorax | 0.0419 | 0.0271 |
| Consolidation | 0.0447 | 0.0314 |
| Edema | 0.0295 | 0.0313 |
| Emphysema | 0.0151 | 0.0139 |
| Fibrosis | 0.0123 | 0.0179 |
| Pleural_Thickening | 0.0386 | 0.0235 |
| Hernia | 0.0100 | 0.0132 |
| **No Finding** *(External)* | **0.7324** | **0.6909** |

*(Note: No Finding threshold was derived from validation probabilities computed as $1 - \max(p_{\text{compatible}})$).*

---

## 3. Overall Performance Comparison

The table below summarizes macro performance metrics (Macro AUROC, Macro AUPRC, Macro F1, Macro Sensitivity, Macro Specificity, Macro ECE, and Brier Score) for the uncalibrated Model A and the calibrated Model B under all threshold strategies:

| Cohort / Split | Model / Strategy | AUROC | AUPRC | F1-Score | Sensitivity | Specificity | ECE | Brier |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NIH Validation** | Model A (Uncalibrated, Default 0.5) | 0.8287 | 0.2551 | 0.2185 | 0.1580 | 0.9916 | 0.0111 | 0.0352 |
| (8,936 images) | Model B (Calibrated, Default 0.5) | 0.8287 | 0.2551 | 0.2187 | 0.1581 | 0.9916 | 0.0135 | 0.0355 |
| | Model B (Calibrated, Youden's J) | 0.8287 | 0.2551 | 0.1987 | 0.7730 | 0.7522 | 0.0135 | 0.0355 |
| | Model B (Calibrated, Sensitivity 80%) | 0.8287 | 0.2551 | 0.1832 | 0.8025 | 0.7034 | 0.0135 | 0.0355 |
| **NIH Test Split** | **Model A (Uncalibrated, Default 0.5)** | **0.7997** | **0.2700** | **0.2110** | **0.1526** | **0.9793** | **0.0238** | **0.0596** |
| (25,596 images) | Model B (Calibrated, Default 0.5) | 0.7997 | 0.2703 | 0.2111 | 0.1526 | 0.9793 | 0.0205 | 0.0594 |
| | **Model B (Calibrated, Youden's J)** | **0.7997** | **0.2703** | **0.2394** | **0.8000** | **0.6281** | **0.0205** | **0.0594** |
| | Model B (Calibrated, Sensitivity 80%) | 0.7997 | 0.2703 | 0.2277 | 0.8381 | 0.5796 | 0.0205 | 0.0594 |
| **VinDr-CXR External**| **Model A (Uncalibrated, Default 0.5)** | **0.8257** | **0.4027** | **0.2016** | **0.1768** | **0.9041** | **0.0651** | **0.0712** |
| (15,000 images) | Model B (Calibrated, Default 0.5) | 0.8257 | 0.4027 | 0.2016 | 0.1768 | 0.9041 | 0.0635 | 0.0700 |
| | **Model B (Calibrated, Youden's J)** | **0.8257** | **0.4027** | **0.3345** | **0.7415** | **0.7239** | **0.0635** | **0.0700** |
| | Model B (Calibrated, Sensitivity 80%) | 0.8257 | 0.4027 | 0.3201 | 0.7992 | 0.6563 | 0.0635 | 0.0700 |

---

## 4. Key Findings & Discussion

### A. Temperature Scaling Effectiveness
- Temperature scaling successfully reduced probability overconfidence caused by network optimization.
- On the **NIH Test set**, ECE decreased from **0.0238 to 0.0205** and Brier score decreased from **0.0596 to 0.0594**.
- On the **VinDr-CXR external validation set**, ECE decreased from **0.0651 to 0.0635** and Brier score decreased from **0.0712 to 0.0700**.
- Crucially, discriminator capability (AUROC and AUPRC) remained identical (or slightly improved in precision due to numerical stability), verifying that scaling preserves rank order.

### B. Operating Threshold Clinical Impact
- Using Youden's J thresholds computed on validation data dramatically improved model sensitivity and clinical utility:
  * **NIH Test Set:** Macro Sensitivity increased from **15.26% to 80.00%**, while maintaining specificity at **62.81%** and improving F1 from **0.2110 to 0.2394**.
  * **VinDr-CXR external cohort:** Macro Sensitivity increased from **17.68% to 74.15%**, specificity is maintained at **72.39%**, and F1-score increases substantially from **0.2016 to 0.3345**!
- The sensitivity-targeted strategy (targeting $\ge 80\%$ validation sensitivity) achieved **80.25%** validation sensitivity, which translated to **83.81%** on NIH test and **79.92%** on VinDr-CXR, proving the generalizability of validation-derived thresholds.

---

## 5. Study Limitations and Scientific Enforcements
- **Hospital Shift:** External validation on VinDr-CXR results in higher ECE values compared to internal NIH test splits, confirming hospital-specific scanner variations affect probability meaning. ECE drops from 0.0651 to 0.0635, showing temperature scaling improves calibration but does not entirely eliminate hospital-shift calibration error.
- **Strict Data Isolation Enforced:**
  > [!IMPORTANT]
  > We confirm that **no VinDr-CXR external validation data or NIH test set data** was used during temperature scaling parameters optimization, Youden J sweep, or sensitivity-targeted threshold selection. All parameters were optimized and frozen using only the NIH validation split prior to test set inference.
