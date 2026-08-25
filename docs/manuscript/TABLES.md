# Publication-Ready Tables

This document contains consolidated, publication-ready performance tables for the chest X-ray DenseNet-121 model.

---

## Table 1: Baseline (Model A) vs. Calibrated Operating-Points (Model B)

Summary of macro-average metrics across the internal NIH Test and external VinDr-CXR validation splits. Baseline 95% Confidence Intervals (CIs) are derived from 1,000 bootstrap replicates. Validation-derived decision thresholds were selected exclusively on NIH validation data and frozen.

| Cohort | Model / Threshold Strategy | Macro AUROC (95% CI) | Macro AUPRC (95% CI) | Macro F1-Score (95% CI) | Macro Sensitivity (95% CI) | Macro Specificity (95% CI) | Expected Calibration Error (ECE) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **NIH Test** | **Model A** (Baseline, Default 0.5) | 0.7997 [0.7956, 0.8040] | 0.2700 [0.2620, 0.2808] | 0.2110 [0.2013, 0.2207] | 0.1525 [0.1451, 0.1604] | 0.9794 [0.9789, 0.9799] | 0.0238 |
| (25,596 images) | **Model B** (Calibrated, Youden's J) | 0.7997 | 0.2703 | 0.2394 | 80.00% | 62.81% | 0.0205 |
| | **Model B** (Calibrated, Sens80) | 0.7997 | 0.2703 | 0.2277 | 83.81% | 57.96% | 0.0205 |
| **VinDr-CXR** | **Model A** (Baseline, Default 0.5) | 0.8257 [0.8182, 0.8326] | 0.4027 [0.3929, 0.4156] | 0.2015 [0.1911, 0.2128] | 0.1767 [0.1701, 0.1845] | 0.9041 [0.9030, 0.9051] | 0.0651 |
| (15,000 images) | **Model B** (Calibrated, Youden's J) | 0.8257 | 0.4027 | 0.3345 | 74.15% | 72.39% | 0.0635 |
| | **Model B** (Calibrated, Sens80) | 0.8257 | 0.4027 | 0.3201 | 79.92% | 65.63% | 0.0635 |

---

## Table 2: Per-Class Performance on NIH Test Split (Model A vs. Model B Youden)

Evaluation of individual pathology detection performance on the 14 NIH classes:

| Disease Pathology Abnormality | Model A AUROC | Model A Sensitivity | Model A Specificity | Youden Threshold | Model B Sensitivity | Model B Specificity |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Atelectasis** | 0.7665 | 0.1650 | 0.9717 | 0.0764 | 0.8829 | 0.4597 |
| **Cardiomegaly** | 0.8772 | 0.2095 | 0.9907 | 0.0574 | 0.6773 | 0.8827 |
| **Effusion** | 0.8199 | 0.4210 | 0.9192 | 0.0901 | 0.9064 | 0.5145 |
| **Infiltration** | 0.6884 | 0.2587 | 0.8999 | 0.1858 | 0.8181 | 0.4120 |
| **Mass** | 0.8108 | 0.2162 | 0.9836 | 0.0511 | 0.7706 | 0.6818 |
| **Nodule** | 0.7393 | 0.0844 | 0.9922 | 0.0674 | 0.7277 | 0.6080 |
| **Pneumonia** | 0.6986 | 0.0018 | 1.0000 | 0.0156 | 0.8180 | 0.4519 |
| **Pneumothorax** | 0.8442 | 0.1790 | 0.9806 | 0.0419 | 0.7887 | 0.7519 |
| **Consolidation** | 0.7306 | 0.0132 | 0.9966 | 0.0447 | 0.8623 | 0.4585 |
| **Edema** | 0.8418 | 0.0757 | 0.9926 | 0.0295 | 0.8649 | 0.6511 |
| **Emphysema** | 0.8912 | 0.2177 | 0.9905 | 0.0151 | 0.8710 | 0.7349 |
| **Fibrosis** | 0.8200 | 0.0253 | 0.9988 | 0.0123 | 0.8483 | 0.5835 |
| **Pleural_Thickening** | 0.7737 | 0.0472 | 0.9946 | 0.0386 | 0.7935 | 0.6277 |
| **Hernia** | 0.8937 | 0.2209 | 0.9996 | 0.0100 | 0.5698 | 0.9757 |

---

## Table 3: Per-Class Performance on VinDr-CXR External Split (Model A vs. Model B Youden)

Evaluation of individual pathology detection performance on the 9 compatible categories:

| Disease Abnormality | Model A AUROC | Model A Sensitivity | Model A Specificity | Youden Threshold | Model B Sensitivity | Model B Specificity |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Atelectasis** | 0.7845 | 0.0215 | 0.9954 | 0.0764 | 0.7043 | 0.7386 |
| **Cardiomegaly** | 0.8387 | 0.0552 | 0.9988 | 0.0574 | 0.5665 | 0.8830 |
| **Effusion** | 0.8899 | 0.2907 | 0.9986 | 0.0901 | 0.6948 | 0.9041 |
| **Infiltration** | 0.8150 | 0.1321 | 0.9962 | 0.1858 | 0.7439 | 0.7122 |
| **Consolidation** | 0.9220 | 0.0227 | 0.9998 | 0.0447 | 0.8385 | 0.8524 |
| **Pneumothorax** | 0.8220 | 0.0521 | 0.9989 | 0.0419 | 0.5625 | 0.8984 |
| **Fibrosis** | 0.7805 | 0.0105 | 0.9998 | 0.0123 | 0.9307 | 0.3325 |
| **Pleural_Thickening** | 0.8076 | 0.0111 | 0.9998 | 0.0386 | 0.7269 | 0.7372 |
| **No Finding** | 0.7714 | 0.9954 | 0.1495 | 0.7324 | 0.9057 | 0.4570 |

---

## Table 4: Subgroup Analysis on NIH Test Split

Performance breakdown by Patient Gender, View Position, and Patient Age brackets:

| Subgroup Category | Subgroup | Sample Size | Configuration | Macro AUROC | Macro F1 | Macro Sensitivity | Macro Specificity |
|---|---|---:|---|:---:|:---:|:---:|:---:|
| **Gender** | **Male** | 14,882 | Model A (Baseline) | 0.7994 | 0.2047 | 14.78% | 97.98% |
| | | | Model B (Youden) | 0.7995 | 0.2361 | 79.38% | 62.42% |
| | **Female** | 10,714 | Model A (Baseline) | 0.7995 | 0.2139 | 15.59% | 97.87% |
| | | | Model B (Youden) | 0.7995 | 0.2423 | 80.32% | 63.35% |
| **View Position** | **AP** | 14,500 | Model A (Baseline) | 0.7684 | 0.1693 | 12.41% | 97.39% |
| | | | Model B (Youden) | 0.7684 | 0.2240 | 74.03% | 61.34% |
| | **PA** | 11,096 | Model A (Baseline) | 0.8029 | 0.2272 | 16.40% | 98.49% |
| | | | Model B (Youden) | 0.8029 | 0.2359 | 78.77% | 64.20% |
| **Age** | **Pediatric** (<20) | 1,382 | Model A (Baseline) | 0.7687 | 0.1601 | 11.41% | 97.87% |
| | | | Model B (Youden) | 0.7687 | 0.2395 | 73.66% | 65.66% |
| | **Young Adult** (20-40) | 6,908 | Model A (Baseline) | 0.7470 | 0.1804 | 12.93% | 98.21% |
| | | | Model B (Youden) | 0.7484 | 0.2279 | 74.37% | 66.03% |
| | **Middle Aged** (40-60)| 11,224 | Model A (Baseline) | 0.7972 | 0.2168 | 15.87% | 97.81% |
| | | | Model B (Youden) | 0.7972 | 0.2399 | 79.97% | 62.15% |
| | **Older Adult** (60-80) | 5,826 | Model A (Baseline) | 0.7912 | 0.2058 | 15.19% | 97.84% |
| | | | Model B (Youden) | 0.7912 | 0.2336 | 80.64% | 59.84% |
| | **Geriatric** (>=80) | 252 | Model A (Baseline) | 0.7534 | 0.1570 | 11.78% | 97.73% |
| | | | Model B (Youden) | 0.7534 | 0.2468 | 76.64% | 53.73% |
| | **Unknown/Outlier** | 4 | Model A (Baseline) | 0.7143 | 0.0714 | 7.14% | 100.00% |
| | | | Model B (Youden) | 0.7143 | 0.2786 | 42.86% | 54.17% |
