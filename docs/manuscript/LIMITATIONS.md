# Limitations

The clinical and statistical findings of this study are subject to the following limitations:

1. **Single External Dataset:** External validation was limited to one Vietnamese dataset (VinDr-CXR). While this demonstrates cross-site generalizability, validation across multiple geographically and ethnically diverse sites is necessary to fully confirm generalizability.
2. **Domain Shift and ECE Degradation:** The calibration error (ECE) degrades from **0.0205** (NIH Test) to **0.0635** (VinDr External) due to differences in scanner manufacturers and scanner settings across sites. Although temperature scaling improves calibration, it does not completely eliminate this cross-site calibration shift.
3. **No Prospective Clinical Validation:** The study evaluates model performance retrospectively on locked cohorts. Prospective clinical trials in active screening workflows are required to assess the model's impact on clinical outcomes and radiologist workflows.
4. **Validation-Based Threshold Selection:** Optimal thresholds were chosen based on a single validation split. While these thresholds generalized successfully to the test split, thresholds may need to be adjusted dynamically to match local clinical prevalence or clinical sensitivity targets.
5. **Lack of Demographic Significance Testing:** While subgroup point estimates show consistent classification scores across age brackets and genders, no formal statistical significance tests were performed on group differences. Therefore, we do not claim a complete absence of demographic bias.
6. **Prevalence and Class Imbalance:** Some disease categories (such as Hernia, Pneumonia, and Fibrosis) have very low positive sample sizes, which leads to higher variance in classification point estimates.
7. **Label Mapping Discrepancies:** VinDr-CXR groups nodules and masses into a single category, which required excluding those categories from the external validation split. Differences in annotations and labeling guidelines between datasets introduce mapping limits that affect comparative evaluations.
8. **OR-Based Radiologist Consensus:** VinDr-CXR ground-truth consensus is defined using a logical OR (positive if at least one radiologist identifies the abnormality). This consensus logic may lead to a higher sensitivity and false alarm rate in ground-truth labels compared to NIH labels.
9. **Calibration Readiness:** While temperature scaling improves probability calibration, this does not imply that the model is ready for autonomous clinical deployment without active radiologist oversight.
