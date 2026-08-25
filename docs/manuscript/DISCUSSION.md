# Discussion

## 1. Generalization on External Data
The deep learning classifier maintained strong diagnostic discrimination when tested on the external VinDr-CXR validation split (Model A Macro AUROC = **0.8257 [95% CI: 0.8182, 0.8326]**). This high performance across independent institutions suggests that the DenseNet-121 features capture robust pathophysiological markers that generalize beyond the training center.

---

## 2. Impact of Probability Calibration
The optimization of temperature scaling parameter ($T = 1.2272$) strictly on validation data successfully corrected probability overconfidence. On the locked test sets, Expected Calibration Error (ECE) decreased consistently (NIH Test: **0.0238 to 0.0205**; VinDr External: **0.0651 to 0.0635**). While calibration does not alter classification discrimination (AUROC), it is essential for clinical decision-making. Calibrated probabilities reflect true disease frequencies more accurately, allowing clinicians to interpret model outputs as true predictive confidence. However, a calibration shift remains visible, with higher ECE values on the external cohort indicating scanner and scanner-setting differences across sites.

---

## 3. Threshold Sweep and Diagnostic Sensitivity Gains
Using a default threshold of `0.5` yields high specificity but unacceptably low sensitivity (~15-17%) for screening. Validation-derived Youden's J thresholds shift the operating balance, increasing macro sensitivity to **80.00%** on the NIH Test set and **74.15%** on the external cohort. This sensitivity improvement occurs at the expense of specificity (which drops to **62.81%** on NIH Test and **72.39%** on VinDr). For screening contexts where missing abnormalities carries severe consequences, this clinical operating point is far more appropriate.

---

## 4. Subgroup Analysis Implications
The model demonstrates balanced classification performance (AUROC) across genders (M: `0.7995`, F: `0.7995`). In view positions, posteroanterior (PA) view films yield higher discrimination than anteroposterior (AP) view films, which is clinically expected because AP films are typically performed on bed-bound, severely ill patients. While subgroup point estimates show consistent classification scores across patient groups, **no formal significance tests for group difference margins were conducted, so we do not claim the absence of demographic bias.**
