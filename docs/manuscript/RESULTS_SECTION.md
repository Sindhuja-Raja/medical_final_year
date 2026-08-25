# Manuscript Results Section

This section presents the performance evaluation, probability calibration, clinical decision threshold optimization, and subgroup generalizability for our deep learning chest X-ray multi-label diagnostic model.

---

## 1. Internal Baseline Performance and Probability Calibration

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
