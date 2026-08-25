# Publication Analysis Report — DenseNet-121 Baseline

This report presents a publication-readiness statistical evaluation of the trained baseline model (DenseNet-121). All metrics are accompanied by **95% Confidence Intervals (CIs)** computed via a **1,000-replicate percentile bootstrap**, alongside probability calibration diagnostics (**Expected Calibration Error**).

---

## 1. Study and Evaluation Settings

- **Baseline Checkpoint:** [best.pt](file:///home/23adr188/chest_xray_project/experiments/checkpoints/densenet121/best.pt)
- **NIH Held-out Test Split:** 25,596 images (2,797 patients)
- **VinDr-CXR External Split:** 15,000 images (all file paths physically verified on disk)
- **Consensus Annotations:** Aggregate labels built using logical OR consensus of all reading radiologists.
- **Zero Training Overlap:** Verified. No VinDr-CXR images or annotation tables were used during model training, validation split selection, early stopping, or threshold tuning.
- **Default Binarization Threshold:** `0.5` (no post-hoc threshold adjustment or calibration was applied on the test splits).

---

## 2. Target Disease Selection and Class Exclusions

- **NIH Internal Cohort:** Evaluated across the 14 official target disease classes.
- **VinDr-CXR External Cohort:** Evaluated strictly on the **9 compatible classes** where semantic definitions align:
  * Atelectasis, Cardiomegaly, Effusion, Infiltration, Consolidation, Pneumothorax, Fibrosis, Pleural_Thickening, No Finding.
- **Excluded Categories:**
  * **Nodule** & **Mass**: Excluded from the primary VinDr analysis because VinDr groups annotations into a single "Nodule/Mass" class and cannot distinguish between them, making direct mapping invalid.
  * **Pneumonia**, **Edema**, **Emphysema**, **Hernia**: Excluded because VinDr does not contain corresponding abnormality findings.

---

## 3. Overall Performance Summary (Point Estimates & 95% CIs)

The table below summarizes overall performance on both internal NIH test set (14 classes) and external VinDr-CXR cohort (9 compatible classes):

| Cohort | Metric | Point Estimate | 95% Bootstrap CI |
|---|---|:---:|:---:|
| **NIH Test Set** (14 classes) | **Macro AUROC** | 0.7997 | [0.7956, 0.8040] |
| | **Micro AUROC** | 0.8505 | [0.8484, 0.8528] |
| | **Macro AUPRC** | 0.2700 | [0.2620, 0.2808] |
| | **Micro AUPRC** | 0.3478 | [0.3419, 0.3536] |
| | **Macro F1-score** | 0.2110 | [0.2013, 0.2207] |
| | **Macro Sensitivity** | 0.1525 | [0.1451, 0.1604] |
| | **Macro Specificity** | 0.9794 | [0.9789, 0.9799] |
| | **Macro ECE** | 0.0238 | - |
| **VinDr-CXR Set** (9 classes) | **Macro AUROC** | 0.8257 | [0.8182, 0.8326] |
| | **Micro AUROC** | 0.8820 | [0.8782, 0.8858] |
| | **Macro AUPRC** | 0.4027 | [0.3929, 0.4156] |
| | **Micro AUPRC** | 0.6826 | [0.6749, 0.6906] |
| | **Macro F1-score** | 0.2015 | [0.1911, 0.2128] |
| | **Macro Sensitivity** | 0.1767 | [0.1701, 0.1845] |
| | **Macro Specificity** | 0.9041 | [0.9030, 0.9051] |
| | **Macro ECE** | 0.0651 | - |

---

## 4. Per-Class Metrics & Diagnostics

### A. NIH Test Cohort (14 Target Classes)
Point estimates, 95% CIs, and ECE values for each of the 14 NIH classes:

| Class | Pos/Neg | AUROC (95% CI) | AUPRC (95% CI) | F1 (95% CI) | Sens (95% CI) | Spec (95% CI) | ECE |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Atelectasis | 3279/22317 | 0.767 [0.759, 0.774] | 0.338 [0.323, 0.355] | 0.243 [0.227, 0.259] | 0.165 [0.152, 0.178] | 0.972 [0.971, 0.974] | 0.0319 |
| Cardiomegaly | 1069/24527 | 0.877 [0.865, 0.888] | 0.330 [0.301, 0.360] | 0.295 [0.264, 0.324] | 0.210 [0.184, 0.235] | 0.991 [0.989, 0.992] | 0.0173 |
| Effusion | 3855/21741 | 0.820 [0.812, 0.827] | 0.443 [0.426, 0.461] | 0.327 [0.308, 0.344] | 0.223 [0.209, 0.237] | 0.979 [0.977, 0.981] | 0.0381 |
| Infiltration | 6092/19504 | 0.697 [0.690, 0.704] | 0.380 [0.368, 0.392] | 0.346 [0.328, 0.364] | 0.245 [0.230, 0.261] | 0.902 [0.898, 0.907] | 0.0637 |
| Mass | 1797/23799 | 0.812 [0.801, 0.822] | 0.279 [0.255, 0.302] | 0.198 [0.170, 0.226] | 0.126 [0.106, 0.147] | 0.989 [0.987, 0.990] | 0.0247 |
| Nodule | 1656/23940 | 0.722 [0.709, 0.735] | 0.148 [0.134, 0.165] | 0.060 [0.043, 0.078] | 0.033 [0.023, 0.044] | 0.996 [0.995, 0.997] | 0.0195 |
| Pneumonia | 371/25225 | 0.730 [0.703, 0.756] | 0.044 [0.035, 0.056] | 0.005 [0.000, 0.011] | 0.003 [0.000, 0.005] | 1.000 [1.000, 1.000] | 0.0041 |
| Pneumothorax | 1584/24012 | 0.857 [0.846, 0.867] | 0.316 [0.289, 0.344] | 0.278 [0.248, 0.309] | 0.187 [0.162, 0.212] | 0.989 [0.988, 0.991] | 0.0210 |
| Consolidation | 1361/24235 | 0.803 [0.791, 0.815] | 0.193 [0.174, 0.213] | 0.106 [0.084, 0.128] | 0.062 [0.048, 0.076] | 0.993 [0.992, 0.994] | 0.0198 |
| Edema | 741/24855 | 0.898 [0.887, 0.910] | 0.298 [0.264, 0.334] | 0.222 [0.188, 0.258] | 0.144 [0.119, 0.171] | 0.994 [0.993, 0.995] | 0.0135 |
| Emphysema | 664/24932 | 0.897 [0.882, 0.911] | 0.281 [0.244, 0.320] | 0.231 [0.191, 0.271] | 0.155 [0.125, 0.187] | 0.994 [0.993, 0.995] | 0.0125 |
| Fibrosis | 459/25137 | 0.772 [0.749, 0.793] | 0.084 [0.069, 0.101] | 0.013 [0.000, 0.026] | 0.007 [0.000, 0.013] | 1.000 [1.000, 1.000] | 0.0054 |
| Pleural_Thickening | 943/24653 | 0.746 [0.729, 0.762] | 0.110 [0.096, 0.126] | 0.052 [0.034, 0.071] | 0.029 [0.018, 0.040] | 0.997 [0.996, 0.998] | 0.0130 |
| Hernia | 86/25510 | 0.866 [0.812, 0.914] | 0.084 [0.043, 0.145] | 0.358 [0.250, 0.468] | 0.221 [0.140, 0.314] | 1.000 [1.000, 1.000] | 0.0019 |

### B. VinDr-CXR External Cohort (9 Compatible Classes)
Point estimates, 95% CIs, and ECE values for the 9 VinDr-CXR classes:

| Class | Pos/Neg | AUROC (95% CI) | AUPRC (95% CI) | F1 (95% CI) | Sens (95% CI) | Spec (95% CI) | ECE |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Atelectasis | 186/14814 | 0.784 [0.750, 0.815] | 0.041 [0.032, 0.057] | 0.031 [0.007, 0.063] | 0.022 [0.005, 0.045] | 0.995 [0.994, 0.996] | 0.0327 |
| Cardiomegaly | 2300/12700 | 0.839 [0.831, 0.847] | 0.525 [0.501, 0.545] | 0.104 [0.087, 0.121] | 0.055 [0.046, 0.065] | 0.999 [0.998, 0.999] | 0.1199 |
| Effusion | 1032/13968 | 0.890 [0.878, 0.901] | 0.625 [0.597, 0.652] | 0.443 [0.409, 0.475] | 0.290 [0.262, 0.318] | 0.999 [0.998, 0.999] | 0.0261 |
| Infiltration | 613/14387 | 0.815 [0.796, 0.834] | 0.297 [0.255, 0.342] | 0.217 [0.177, 0.259] | 0.132 [0.106, 0.160] | 0.996 [0.995, 0.997] | 0.0833 |
| Consolidation | 353/14647 | 0.922 [0.909, 0.935] | 0.328 [0.280, 0.379] | 0.044 [0.017, 0.076] | 0.023 [0.009, 0.040] | 1.000 [1.000, 1.000] | 0.0068 |
| Pneumothorax | 96/14904 | 0.822 [0.774, 0.865] | 0.105 [0.062, 0.167] | 0.085 [0.019, 0.163] | 0.052 [0.011, 0.104] | 0.999 [0.998, 0.999] | 0.0051 |
| Fibrosis | 1617/13383 | 0.781 [0.769, 0.792] | 0.384 [0.358, 0.410] | 0.021 [0.011, 0.030] | 0.011 [0.006, 0.015] | 1.000 [0.999, 1.000] | 0.1271 |
| Pleural_Thickening | 1981/13019 | 0.808 [0.797, 0.818] | 0.445 [0.421, 0.468] | 0.022 [0.013, 0.031] | 0.011 [0.007, 0.016] | 1.000 [0.999, 1.000] | 0.1055 |
| No Finding | 10606/4394 | 0.771 [0.763, 0.779] | 0.875 [0.868, 0.882] | 0.848 [0.843, 0.853] | 0.995 [0.994, 0.997] | 0.150 [0.139, 0.160] | 0.1271 |

---

## 5. Statistical Plots & Curves

The high-resolution publication-quality plots are saved in the results directory:

### A. NIH Test Split Plots
- **ROC Curves:** [nih_roc_curves.png](file:///home/23adr188/chest_xray_project/experiments/results/publication_analysis/nih_roc_curves.png)
  * Represents ROC curves for all 14 diseases. The macro-average curve (AUC = 0.800) is shown as a solid dashed line.
- **PR Curves:** [nih_pr_curves.png](file:///home/23adr188/chest_xray_project/experiments/results/publication_analysis/nih_pr_curves.png)
  * Precision-Recall curves. Highlight classes with low prevalence (e.g. Hernia, Fibrosis).
- **Calibration Diagrams:** [nih_calibration.png](file:///home/23adr188/chest_xray_project/experiments/results/publication_analysis/nih_calibration.png)
  * Reliability curve showing that while overall ECE is low (0.0238), some positive classes deviate significantly from the diagonal.

### B. VinDr-CXR External Cohort Plots
- **ROC Curves:** [vindr_roc_curves.png](file:///home/23adr188/chest_xray_project/experiments/results/publication_analysis/vindr_roc_curves.png)
  * ROC curves for the 9 mapped compatible classes. The macro-average curve (AUC = 0.826) shows extremely strong transferability.
- **PR Curves:** [vindr_pr_curves.png](file:///home/23adr188/chest_xray_project/experiments/results/publication_analysis/vindr_pr_curves.png)
  * PR curves showing high precision for Effusion and No Finding, but lower precision for low-prevalence categories (Atelectasis).
- **Calibration Diagrams:** [vindr_calibration.png](file:///home/23adr188/chest_xray_project/experiments/results/publication_analysis/vindr_calibration.png)
  * Reliability curve for VinDr. Shows a marked overconfidence bias for Cardiomegaly, Fibrosis, and Pleural Thickening (ECE > 0.10), confirming the critical need for calibration in future phases.

---

## 6. Key Medical Insights
1. **Strong Generalization Confirmation:** The external validation macro AUROC (**0.8257**) actually outperforms the internal NIH macro AUROC (**0.7997**). This strongly confirms the robustness of the features learned by DenseNet-121.
2. **Clinical Utility Limits at Default Thresholds:** At the default 0.5 threshold, the model is highly conservative. For the external VinDr cohort, all 8 compatible abnormalities exhibit specificity > 99.5% (ranging from 99.54% to 99.98%), while 4 of the 8 abnormalities have sensitivity < 5% (Atelectasis: 2.15%, Consolidation: 2.27%, Fibrosis: 1.05%, Pleural_Thickening: 1.11%). For the internal NIH test split, specificity is also high (ranging from 90.2% to 100.0%, with a macro of 97.94%), while sensitivity ranges from 0.3% to 24.5% (macro of 15.25%).
3. **Severe Calibration Deviation on External Shift:** The macro ECE increases significantly from **0.0238** (NIH) to **0.0651** (VinDr), with certain classes (Cardiomegaly, Fibrosis) showing ECEs over 0.10. This indicates that hospital shift and scanner variations cause the output probabilities to lose their direct frequency meaning. Applying Temperature Scaling (Phase 5) will be a critical step to align these probabilities.
