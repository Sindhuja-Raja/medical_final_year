# Methods

## 1. Datasets and Splitting Protocol

### A. NIH ChestX-ray14
The internal cohort was derived from the NIH ChestX-ray14 dataset containing 112,120 chest radiographs. We implemented a patient-safe splitting protocol where patients were strictly partitioned at the Patient ID level to prevent data leakage. The dataset splits are allocated as:
- **Training split:** 77,588 images (25,207 unique patients)
- **Validation split:** 8,936 images (2,801 unique patients)
- **Locked Test split:** 25,596 images (2,797 unique patients)

No patient overlap exists between the training, validation, or test sets:
$$\text{Patients(train)} \cap \text{Patients(validation)} = \emptyset$$
$$\text{Patients(train)} \cap \text{Patients(test)} = \emptyset$$
$$\text{Patients(validation)} \cap \text{Patients(test)} = \emptyset$$

### B. External VinDr-CXR validation split
For external validation, we used the VinDr-CXR dataset (15,000 images, binarized ground-truth annotations from a consensus of 17 radiologists). Since VinDr-CXR groups nodules and masses into a single category, those classes were excluded. Evaluation was performed on the **9 compatible classes**: Atelectasis, Cardiomegaly, Effusion, Infiltration, Consolidation, Pneumothorax, Fibrosis, Pleural Thickening, and No Finding.

---

## 2. Model Architecture and Training

We utilized a DenseNet-121 backbone initialized with pre-trained ImageNet weights. The network classifier head was modified to output a 14-dimensional vector corresponding to the NIH pathology classes, trained using Binary Cross Entropy (BCE) loss:
$$\mathcal{L} = -\sum_{i=1}^{14} \left[ y_i \log \sigma(z_i) + (1 - y_i) \log (1 - \sigma(z_i)) \right]$$
The model was optimized using Adam with a learning rate of $1\times10^{-4}$, weight decay of $1\times10^{-4}$, batch size of 32, mixed-precision training (AMP), early stopping (patience of 5 epochs on validation AUROC), and a random seed of 42.

---

## 3. Probability Calibration and Temperature Scaling

To correct model overconfidence, we implemented Temperature Scaling on validation logits. A single scalar parameter $T > 0$ is optimized to minimize Binary Cross Entropy on the 8,936 validation split samples, keeping the network weights frozen:
$$\min_{T} -\frac{1}{N \cdot 14} \sum_{n=1}^{N} \sum_{i=1}^{14} \left[ y_{n,i} \log \sigma\left(\frac{z_{n,i}}{T}\right) + (1 - y_{n,i}) \log \left(1 - \sigma\left(\frac{z_{n,i}}{T}\right)\right) \right]$$
This L-BFGS optimization yielded $T = 1.2272$, which was frozen. Probability calibration was evaluated using Expected Calibration Error (ECE) and Brier scores.

---

## 4. Operating Threshold Selection

All classification decision thresholds were optimized and frozen exclusively on the NIH validation split prior to test split evaluations:
1. **Youden's J Statistic:** For each class $i$, we swiped thresholds to locate the argmax of Youden's index:
   $$J_i = \text{Sensitivity}_i + \text{Specificity}_i - 1 = \text{TPR}_i - \text{FPR}_i$$
2. **Sensitivity-Targeted Constraint:** Selected thresholds to satisfy a clinical target sensitivity of $\ge 80\%$ while maximizing specificity.

---

## 5. Statistical Evaluation and Subgroup Slicing

### A. Vectorized Percentile Bootstrapping
To compute 95% Confidence Intervals (CIs) for point estimates (AUROC, AUPRC, F1, sensitivity, specificity), we performed a 1,000-replicate percentile bootstrap.

### B. Subgroup Analysis
We evaluated the generalizability of Model A and Model B configurations across subgroups defined by:
- Patient Gender (Male vs. Female)
- View Position (Anteroposterior [AP] vs. Posteroanterior [PA])
- Patient Age Brackets (Pediatric `< 20`, Young Adult `20-40`, Middle Aged `40-60`, Older Adult `60-80`, Geriatric `\ge 80`, Unknown/Outliers)

### C. Integrity Verification Checksums
Safety tests enforce that the md5 checksums of the locked model `best.pt` (`17483e1928eb6c39e2f9bfd2f2b1434a`), validation split `val.csv` (`2174a9105496360451d8bc10580a23d7`), and test split `test.csv` (`c9b7c723864fac322137edd8ab4e17fe`) are verified before reporting.
