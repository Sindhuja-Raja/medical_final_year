# Results

This section presents the performance evaluation, probability calibration, clinical decision threshold optimization, and subgroup generalizability for our deep learning chest X-ray multi-label diagnostic model.

---

## 1. Baseline Performance and Probability Calibration

The baseline model (Model A) represents the frozen DenseNet-121 network using a default decision threshold of `0.5` across all classes. To correct probability overconfidence, a single scalar temperature parameter ($T = 1.2272$) was optimized and frozen exclusively on the NIH validation split (8,936 images).

### A. Discrimination Performance
Model discrimination remains unchanged before and after temperature scaling, as the scaling function is strictly monotonic ($T > 0$) and preserves rank order exactly. On the locked internal NIH Test split (25,596 images), the model achieved a macro-average Area Under the Receiver Operating Characteristic (AUROC) curve of **0.7997 [95% CI: 0.7956, 0.8040]** and a macro-average Area Under the Precision-Recall Curve (AUPRC) of **0.2700 [95% CI: 0.2620, 0.2808]** (1,000 bootstrap replicates).

### B. Probability Calibration
Temperature scaling calibration successfully reduced expected calibration error (ECE) and Brier scores across both test cohorts:
- **NIH Test Split:** ECE decreased from **0.0238 to 0.0205**, and Brier score decreased from **0.0596 to 0.0594**.
- **VinDr-CXR External Split:** ECE decreased from **0.0651 to 0.0635**, and Brier score decreased from **0.0712 to 0.0700**.

This improvement indicates that calibrated probabilities (Model B) represent clinical diagnostic confidence more reliably than the raw model probabilities (Model A), although hospital/scanner variations still introduce residual calibration error on the external split.

---

## 2. Clinical Operating Points and Threshold Optimization

Decision thresholds were optimized exclusively on the NIH validation split under two strategies (Youden's J statistic and an 80% sensitivity-targeted constraint) and frozen prior to final evaluations. This optimization reveals a stark sensitivity-specificity trade-off compared to the default `0.5` threshold:

1. **Default Threshold (0.5):** Model A (Baseline) exhibits high specificity (NIH Test: **97.94% [95% CI: 0.9789, 0.9799]**; VinDr-CXR: **90.41% [95% CI: 0.9030, 0.9051]**) but extremely low clinical sensitivity (NIH Test: **15.25% [95% CI: 0.1451, 0.1604]**; VinDr-CXR: **17.67% [95% CI: 0.1701, 0.1845]**). This operating point is clinically highly conservative and prone to high false-negative rates.
2. **Youden's J Thresholds:** Operating at Youden-optimal thresholds (Model B) shifts the balance, increasing macro sensitivity on NIH Test to **80.00%** and on VinDr-CXR to **74.15%**. This sensitivity gain is accompanied by a specificity trade-off, with specificity dropping to **62.81%** on NIH Test and **72.39%** on VinDr-CXR.
3. **Sensitivity-Targeted (80%) Thresholds:** Selecting thresholds to target at least 80% sensitivity on validation achieved **83.81%** sensitivity on NIH Test and **79.92%** sensitivity on VinDr-CXR, proving the generalizability of validation-derived operating points. Specificities dropped further to **57.96%** (NIH Test) and **65.63%** (VinDr-CXR).

---

## 3. Subgroup and Generalizability Analysis

To evaluate model generalizability across clinical populations, subgroup analyses were performed on the locked NIH test split using the frozen thresholds.

### A. Patient Gender
Point estimates for macro AUROC are identical across male and female subgroups (AUROC = **0.7995** for both). Macro sensitivity under Youden's J is **79.38%** for male and **80.32%** for female subgroups. However, in the absence of formal statistical significance testing for difference margins, an absence of demographic bias cannot be definitively concluded.

### B. View Position
Classification performance varies between View Positions, with Posteroanterior (PA) views yielding higher discrimination (AUROC = **0.8029**) than Anteroposterior (AP) views (AUROC = **0.7684**). This difference is clinically expected, as AP view radiographs are typically acquired from bed-bound or severely ill patients, which are technically more challenging and contain overlapping pathologies.

### C. Patient Age
The model maintains stable discrimination across pediatric and adult age brackets:
- **Pediatric (< 20 years, n = 1,382):** AUROC = **0.7687**, Youden J Sensitivity = **73.66%**, Specificity = **65.66%**.
- **Young Adult (20 - 40 years, n = 6,908):** AUROC = **0.7484**, Youden J Sensitivity = **74.37%**, Specificity = **66.03%**.
- **Middle Aged (40 - 60 years, n = 11,224):** AUROC = **0.7972**, Youden J Sensitivity = **79.97%**, Specificity = **62.15%**.
- **Older Adult (60 - 80 years, n = 5,826):** AUROC = **0.7912**, Youden J Sensitivity = **80.64%**, Specificity = **59.84%**.
- **Geriatric (>= 80 years, n = 252):** AUROC = **0.7534**, Youden J Sensitivity = **76.64%**, Specificity = **53.73%**.
- **Unknown/Outlier (n = 4):** AUROC = **0.7143**, Youden J Sensitivity = **42.86%**, Specificity = **54.17%**.

---

## 4. Evaluation Performance Tables

### Table 1: Baseline (Model A) vs. Calibrated Operating-Points (Model B)
Summary of macro-average metrics. Baseline 95% CIs are derived from 1,000 bootstrap replicates. Validation-derived decision thresholds were selected exclusively on NIH validation data and frozen.

| Cohort | Model / Threshold Strategy | Macro AUROC (95% CI) | Macro AUPRC (95% CI) | Macro F1-Score (95% CI) | Macro Sensitivity (95% CI) | Macro Specificity (95% CI) | Expected Calibration Error (ECE) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **NIH Test** | **Model A** (Baseline, Default 0.5) | 0.7997 [0.7956, 0.8040] | 0.2700 [0.2620, 0.2808] | 0.2110 [0.2013, 0.2207] | 0.1525 [0.1451, 0.1604] | 0.9794 [0.9789, 0.9799] | 0.0238 |
| (25,596 images) | **Model B** (Calibrated, Youden's J) | 0.7997 | 0.2703 | 0.2394 | 80.00% | 62.81% | 0.0205 |
| | **Model B** (Calibrated, Sens80) | 0.7997 | 0.2703 | 0.2277 | 83.81% | 57.96% | 0.0205 |
| **VinDr-CXR** | **Model A** (Baseline, Default 0.5) | 0.8257 [0.8182, 0.8326] | 0.4027 [0.3929, 0.4156] | 0.2015 [0.1911, 0.2128] | 0.1767 [0.1701, 0.1845] | 0.9041 [0.9030, 0.9051] | 0.0651 |
| (15,000 images) | **Model B** (Calibrated, Youden's J) | 0.8257 | 0.4027 | 0.3345 | 74.15% | 72.39% | 0.0635 |
| | **Model B** (Calibrated, Sens80) | 0.8257 | 0.4027 | 0.3201 | 79.92% | 65.63% | 0.0635 |

### Table 2: Per-Class Performance on NIH Test Split (Model A vs. Model B Youden)
Individual pathology detection performance on the 14 NIH classes:

| Disease Pathology | Model A AUROC | Model A Sensitivity | Model A Specificity | Youden Threshold | Model B Sensitivity | Model B Specificity |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Atelectasis | 0.7665 | 0.1650 | 0.9717 | 0.0764 | 0.8829 | 0.4597 |
| Cardiomegaly | 0.8772 | 0.2095 | 0.9907 | 0.0574 | 0.6773 | 0.8827 |
| Effusion | 0.8199 | 0.4210 | 0.9192 | 0.0901 | 0.9064 | 0.5145 |
| Infiltration | 0.6884 | 0.2587 | 0.8999 | 0.1858 | 0.8181 | 0.4120 |
| Mass | 0.8108 | 0.2162 | 0.9836 | 0.0511 | 0.7706 | 0.6818 |
| Nodule | 0.7393 | 0.0844 | 0.9922 | 0.0674 | 0.7277 | 0.6080 |
| Pneumonia | 0.6986 | 0.0018 | 1.0000 | 0.0156 | 0.8180 | 0.4519 |
| Pneumothorax | 0.8442 | 0.1790 | 0.9806 | 0.0419 | 0.7887 | 0.7519 |
| Consolidation | 0.7306 | 0.0132 | 0.9966 | 0.0447 | 0.8623 | 0.4585 |
| Edema | 0.8418 | 0.0757 | 0.9926 | 0.0295 | 0.8649 | 0.6511 |
| Emphysema | 0.8912 | 0.2177 | 0.9905 | 0.0151 | 0.8710 | 0.7349 |
| Fibrosis | 0.8200 | 0.0253 | 0.9988 | 0.0123 | 0.8483 | 0.5835 |
| Pleural_Thickening | 0.7737 | 0.0472 | 0.9946 | 0.0386 | 0.7935 | 0.6277 |
| Hernia | 0.8937 | 0.2209 | 0.9996 | 0.0100 | 0.5698 | 0.9757 |

### Table 3: Per-Class Performance on VinDr-CXR External Split (Model A vs. Model B Youden)
Individual pathology detection performance on the 9 compatible categories:

| Disease Abnormality | Model A AUROC | Model A Sensitivity | Model A Specificity | Youden Threshold | Model B Sensitivity | Model B Specificity |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Atelectasis | 0.7845 | 0.0215 | 0.9954 | 0.0764 | 0.7043 | 0.7386 |
| Cardiomegaly | 0.8387 | 0.0552 | 0.9988 | 0.0574 | 0.5665 | 0.8830 |
| Effusion | 0.8899 | 0.2907 | 0.9986 | 0.0901 | 0.6948 | 0.9041 |
| Infiltration | 0.8150 | 0.1321 | 0.9962 | 0.1858 | 0.7439 | 0.7122 |
| Consolidation | 0.9220 | 0.0227 | 0.9998 | 0.0447 | 0.8385 | 0.8524 |
| Pneumothorax | 0.8220 | 0.0521 | 0.9989 | 0.0419 | 0.5625 | 0.8984 |
| Fibrosis | 0.7805 | 0.0105 | 0.9998 | 0.0123 | 0.9307 | 0.3325 |
| Pleural_Thickening| 0.8076 | 0.0111 | 0.9998 | 0.0386 | 0.7269 | 0.7372 |
| No Finding | 0.7714 | 0.9954 | 0.1495 | 0.7324 | 0.9057 | 0.4570 |
