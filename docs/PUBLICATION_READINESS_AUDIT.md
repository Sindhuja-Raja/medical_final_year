# Publication-Readiness & Reproducibility Audit

This report presents a thorough review of the chest X-ray baseline classification pipeline (DenseNet-121) to assess its reproducibility and identify gaps that must be resolved prior to peer-reviewed publication.

---

## 1. Reproducibility & Pipeline Integrity Verification

We verified the reproducibility status of the model checkpoints and data splits:

- **Data Splitting Lock: VERIFIED**
  - NIH ChestX-ray14 training, validation, and test splits are strictly locked under `data/splits/`.
  - VinDr-CXR external validation consensus metadata is strictly locked under `data/external/vinbigdata_metadata.csv`.
  - Zero patient leakage is programmatically enforced and verified by unit tests (passing `pytest tests/test_leakage.py`).
- **Configuration & Seed Records: VERIFIED**
  - Training configuration `configs/train.yaml` and dataset settings `configs/dataset.yaml` record the fixed random seed (`42`).
  - Saved model checkpoints (`last.pt`, `best.pt`) contain the complete configuration dictionary and Python/NumPy/PyTorch random number generator states at the time of checkpointing.
- **Evaluation Scripts: VERIFIED**
  - Evaluators `src/evaluate.py` and `src/evaluate_external.py` use locked configurations and programmatically verify image/patient counts (NIH test: 25,596 images, 2,797 patients; VinDr-CXR: 15,000 images) prior to run start.
  - Verification test runs confirm that all metrics can be fully replicated from checkpoints.

---

## 2. Identified Publication Gaps (Missing Components)

For a peer-reviewed medical imaging or AI publication, the following components are currently missing or underspecified and must be generated:

### A. Statistical Analysis & Confidence Intervals (CIs)
- **Gap:** Overall and per-class metrics (AUROC, AUPRC, Sensitivity, Specificity, F1) are currently reported only as point estimates.
- **Requirement:** Report **95% Confidence Intervals (CIs)** for all primary metrics.
- **Resolution Plan:** Implement a bootstrapping verification script (e.g., 1,000 bootstrap replicates with percentile method) to calculate and append 95% CIs to all results tables.

### B. Visualization and Plotting
- **Gap:** No graphical plots are currently generated or saved in the results directories.
- **Requirement:** A medical publication requires the following figures:
  1. **Receiver Operating Characteristic (ROC) Curves:** Showing curves for overall macro-average and individual abnormalities.
  2. **Precision-Recall (PR) Curves:** Essential for imbalanced datasets (e.g., Pneumothorax with 0.64% prevalence in VinDr).
  3. **Reliability Diagrams (Calibration Curves):** Showing model probability calibration compared to true fraction of positives.
- **Resolution Plan:** Create a plotting module (`src/visualization/plots.py`) to generate and save publication-quality vector graphics (PDF/PNG format).

### C. Probability Calibration
- **Gap:** Model specificity is extremely high while sensitivity is low, indicating that the raw sigmoid probabilities are uncalibrated for clinical thresholding at 0.5.
- **Requirement:** Compute and report **Expected Calibration Error (ECE)**, and apply temperature scaling or Platt scaling to calibrate predictions.
- **Resolution Plan:** Execute calibration (Phase 5/6 of the project plan) using the validation split only to fit temperature scaling.

### D. Patient Demographics & Subgroup Analysis
- **Gap:** No demographics distributions (Age, Sex, View Position) are reported for the splits.
- **Requirement:** Report a "Table 1" detailing cohort demographics and evaluate performance across subgroups (e.g., male vs. female, AP vs. PA view) to check for systemic bias.
- **Resolution Plan:** Extract demographics from `Data_Entry_2017.csv` and generate a demographics breakdown script.

### E. Methodological Documentation
- **Gap:** The exact consensus rules and mapping exclusions are documented in code and phase reports, but lack a formal publication-style appendix.
- **Requirement:** Write a structured appendix detailing:
  1. **VinDr-CXR Consensus Logic:** Rationale behind using logical OR consensus for annotations.
  2. **NIH Excluded Categories:** Explanation of why Nodule, Mass, Pneumonia, Edema, Emphysema, and Hernia were excluded from external validation (e.g., label pollution, category grouping).

---

## 3. Publication Readiness Matrix

| Feature / Metric | Status | Location / Artifact | Action Required |
|---|:---:|---|---|
| **Reproducibility** | **PASS** | `configs/`, `src/train.py`, `src/evaluate.py` | None |
| **NIH Baseline Metrics** | **PASS** | `experiments/results/densenet121/` | None |
| **External Baselines** | **PASS** | `experiments/results/external_validation.json` | None |
| **95% Confidence Intervals** | **MISSING** | - | Implement bootstrapping |
| **ROC/PR Curves** | **MISSING** | - | Create plotting script |
| **ECE & Calibration** | **MISSING** | - | Complete Phase 5 & 6 |
| **Demographics Table** | **MISSING** | - | Compile demographics script |
| **Label Map Appendix** | **PASS** | `docs/EXTERNAL_VALIDATION_REPORT.md` | Formalize in appendix |
