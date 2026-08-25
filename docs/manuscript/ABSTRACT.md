# Abstract

## Background
Deep learning classifiers for multi-label chest X-ray interpretation typically exhibit probability overconfidence (poor calibration) and rely on default decision thresholds (0.5) that yield low diagnostic sensitivity. We present a systematic framework to calibrate classification probabilities and optimize clinical decision thresholds using validation data, followed by validation on an independent external cohort.

## Methods
A DenseNet-121 model was trained on the NIH ChestX-ray14 dataset. A patient-safe splitting protocol was enforced (77,588 train, 8,936 validation, and 25,596 locked test images) with zero patient overlap. Temperature scaling calibration parameter ($T = 1.2272$) and class-specific decision thresholds (Youden's J and 80% sensitivity-targeted) were optimized and frozen exclusively on the NIH validation split. The model was evaluated on the locked NIH test split and an independent external cohort from VinDr-CXR (15,000 images, evaluated on 9 compatible abnormality classes). Statistical confidence intervals (CIs) were computed using a 1,000-replicate bootstrap.

## Results
In baseline evaluation (Model A, threshold 0.5), discrimination was high (NIH Test Macro AUROC = **0.7997 [95% CI: 0.7956, 0.8040]**; VinDr External Macro AUROC = **0.8257 [95% CI: 0.8182, 0.8326]**), but clinical sensitivity was low (NIH Test: **15.25% [95% CI: 0.1451, 0.1604]**; VinDr External: **17.67% [95% CI: 0.1701, 0.1845]**). Temperature scaling reduced calibration error (NIH Test ECE: **0.0238 to 0.0205**; VinDr External ECE: **0.0651 to 0.0635**), while preserving AUROC. Evaluating Model B with validation-derived Youden's J thresholds significantly increased macro sensitivity on the NIH Test split to **80.00%** (specificity: **62.81%**) and on the external VinDr cohort to **74.15%** (specificity: **72.39%**).

## Conclusions
Validation-derived calibration scaling and decision thresholds generalize successfully to external cohorts, enabling substantial sensitivity improvements for clinical screening. This gain, however, occurs at the expense of specificity, and domain shifts introduce residual calibration errors on external datasets. These trade-offs must be factored into diagnostic triage designs.
