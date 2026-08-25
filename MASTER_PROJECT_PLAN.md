# MASTER PROJECT PLAN

## Evidence-Aware Stateful Agentic Framework for Chest X-Ray Disease Classification via Heterogeneous Evidence Reconciliation

> **Status:** Working master document. Update in place as phases complete — do not fork new plan files.
> **Last foundation freeze:** NIH ChestX-ray14 primary / META-CXR base paper / Cross-source evidence reconciliation novelty.

---

# 1. Project Definition

### 1.1 Final Title

**Primary:** Stateful Heterogeneous-Evidence Reconciliation for Uncertainty-Aware Chest X-Ray Disease Classification

**Short:** Evidence-Aware Stateful Agent for Chest X-Ray Diagnosis Support

Do not finalize the title until Phase 15 (critical experiment) produces results — the title should reflect what was *demonstrated*, not just what was *attempted*.

### 1.2 Research Objective

Build and evaluate a chest X-ray decision-support system in which a small, interpretable, rule-based agent accumulates heterogeneous evidence (visual classifiers, XAI, VLM, RAG) in a persistent per-case state, and uses disagreement/uncertainty in that state to decide what additional evidence to acquire, when to stop, and when to escalate — and to test whether **statefulness itself** (not just adaptive tool selection) is responsible for any observed benefit.

### 1.3 Research Question

> **Primary:** Can a stateful evidence-reconciliation policy improve uncertainty handling and reduce unnecessary evidence-tool invocation compared with (a) a fixed sequential pipeline and (b) an adaptive-but-stateless policy, on chest X-ray multi-label classification?

### 1.4 Research Hypotheses

- **H-Adapt:** An adaptive policy that selects tools based on current evidence outperforms a fixed pipeline on tool-efficiency metrics without degrading classification/verification quality.
- **H-State (primary, first-class hypothesis):** Persistent accumulation and reconciliation of heterogeneous evidence enables better conflict resolution and evidence-aware decisions than an otherwise-equivalent **stateless** adaptive policy.
- Expected ordering on conflict-resolution/evidence-quality metrics: **Stateful > Stateless > Fixed**. If this ordering does not hold, report it honestly — a null result on H-State is still a valid, publishable finding if the experiment was designed correctly.

### 1.5 Base Paper

**META-CXR** — *Chest X-Ray Report Generation Using Abnormality Guided Vision Language Model*, IEEE Access, 2025, DOI 10.1109/ACCESS.2025.3606961.

Provides: multi-encoder (CNN + ViT + Swin) visual representation → MHCAC multi-label abnormality classification with uncertainty modeling → META-Former → report generation.

### 1.6 Explicit MIMIC → NIH Dataset Transfer Statement

State this plainly in the thesis/paper, not implicitly:

> META-CXR's own reported evaluation uses the MIMIC-CXR test set (CheXpert-style 14-pathology labels including "No Finding") for both its classification and report-generation components. This project transfers META-CXR's multi-encoder classification *framing* — not its trained weights or its benchmark — to a different dataset, NIH ChestX-ray14, and does not claim to reproduce META-CXR's reported numbers. The base paper is the methodological foundation; the dataset and orchestration layer are this project's experimental environment and contribution respectively.

Do not imply, in any figure or table, that META-CXR was evaluated on NIH ChestX-ray14. It was not.

---

# 2. Dataset

### 2.1 Primary Dataset

**NIH ChestX-ray14**
- ~112,120 frontal-view images, ~30,805 patients, 14 disease labels (NLP-derived from radiology reports; no "No Finding" as a 15th disease category the way CheXpert/MIMIC define it — verify this against the actual label file in Phase 1, do not assume).
- Publicly downloadable, no DUA/credentialing required.

### 2.2 External Validation

**VinDr-CXR** — used only for labels/findings that are semantically compatible with the NIH label set. Build an explicit label-compatibility map before using it; do not force-map incompatible categories.

### 2.3 Explicitly Excluded

**MIMIC-CXR Kaggle re-upload** (`kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset`) — excluded. MIMIC-CXR is PhysioNet credentialed-access data; the DUA prohibits redistribution. If genuine MIMIC-CXR access is later needed (e.g., for report-text grounding in RAG), it must come from an individually credentialed PhysioNet account, never from this or any other third-party mirror.

### 2.4 Label Mapping

Freeze in `configs/dataset.yaml` after Phase 1's audit. Do not assume NIH's 14 labels match any other paper's label set without checking the actual column names and value encodings first.

### 2.5 Patient-Level Split (mandatory)

```text
Patients(train) ∩ Patients(validation) = ∅
Patients(train) ∩ Patients(test) = ∅
Patients(validation) ∩ Patients(test) = ∅
```

Automated leakage test required before any model training (see Phase 2 checkpoint).

### 2.6 Leakage Rules

- Classification models never see ground-truth report text or labels of the case being predicted.
- RAG corpus is a frozen, external, "approved medical knowledge" source — never the dataset's own reports, and never anything derived from the test split.
- Calibration parameters fit only on the validation split.

### 2.7 Dataset Size Strategy

| Stage | Size | Purpose |
|---|---|---|
| A — Smoke test | 1k–5k | pipeline correctness |
| B — Development | 10k–20k | early comparison, agent prototyping |
| C — Serious comparison | ~50k | meaningful model benchmark |
| D — Final experiment | 50k–120k (patient-safe, balanced) | final reported results |

---

# 3. Five-Model Visual Benchmark

### 3.1 Candidates

1. ResNet50 — classical CNN baseline, Grad-CAM reference
2. DenseNet121 — strong CXR-specific CNN comparison
3. EfficientNet-B0/B2 — efficiency/performance comparison
4. ConvNeXt-Tiny — modern CNN comparison
5. Swin-Tiny — Transformer candidate

### 3.2 Selection Criteria

Same dataset, same patient-safe split, same evaluation protocol (macro AUROC primary, per-class AUROC, macro F1). Best CNN = highest macro AUROC among candidates 1–4, transfer-learned (ImageNet-pretrained), not trained from scratch.

### 3.3 Pipeline

```text
4 CNN candidates → benchmark → best CNN
                                    +
                                  Swin-Tiny
                                    ↓
                          compare / optional fusion
```

Note: CheX-DS (2025) already reports 83.76% macro AUC with a DenseNet+Swin ensemble on this exact dataset. **CNN+Swin fusion is not this project's novelty** — it is visual-foundation infrastructure. Report fusion results honestly (Section 8 covers the fusion decision gate).

---

# 4. VLM Specification

Fill this in during Phase 8, before writing any agent code that calls a VLM — do not leave it generic.

- **Exact model:** _(record chosen pretrained VLM, e.g., a CXR-tuned open VLM or a general medical VLM — decide and record here once selected)_
- **Why this VLM:** _(license, inference cost, structured-output reliability, availability on the NVIDIA server)_
- **Pretraining-data check:** Record whether the VLM's pretraining/fine-tuning data overlaps NIH ChestX-ray14, CheXpert, or MIMIC-CXR. If it does, the VLM is **not statistically independent** of the visual classifiers or of the benchmark itself — this must be disclosed in the evidence-taxonomy and limitations sections.
- **What it does:** produces structured findings + confidence + observations (see JSON schema, Section 5.3).
- **What it does NOT do:** it is not the final classifier, not trained/fine-tuned by default (LoRA fine-tuning is Optional only), and its output is validated before being written into Evidence State.

---

# 5. Evidence Taxonomy

Evidence sources are **heterogeneous/complementary**, not statistically independent. State this explicitly in the methods section — do not claim independence.

### 5.1 Visual Classifier Evidence
- CNN prediction (probability + confidence)
- Transformer (Swin) prediction (probability + confidence)
- XAI localization/evidence strength — **derived from** the CNN/Transformer, not independent of them

### 5.2 Multimodal Evidence
- VLM structured findings — correlated with visual classifiers to the extent its pretraining data overlaps the same distributions (see Section 4)

### 5.3 External Contextual Evidence
- RAG retrieval over a frozen, approved external medical-knowledge corpus (not the dataset's own reports)

### 5.4 VLM Output Schema

```json
{
  "findings": [{"name": "cardiomegaly", "confidence": 0.78}],
  "observations": ["enlarged cardiac silhouette"],
  "uncertainty": "moderate"
}
```

---

# 6. Three-Layer Architecture

```text
LAYER 1 — VISUAL INTELLIGENCE
ResNet50 / DenseNet121 / EfficientNet / ConvNeXt-Tiny / Swin-Tiny
        ↓
Model comparison → Best CNN + Swin → (optional fusion)
        ↓
Disease classification

LAYER 2 — EVIDENCE INTELLIGENCE
Classification → Calibration → XAI → VLM → RAG
(heterogeneous evidence production, not yet reconciled)

LAYER 3 — AGENTIC INTELLIGENCE
Evidence State → Rule-based Agent → Select evidence → Verify → Continue / Result / Escalate
(THIS IS WHERE THE RESEARCH CONTRIBUTION LIVES)
```

---

# 7. Stateful Agent

### 7.1 Evidence State Schema

```json
{
  "prediction": {"cardiomegaly": 0.92},
  "confidence": "high",
  "cnn": {"prediction": "positive", "confidence": 0.92},
  "swin": {"prediction": "negative", "confidence": 0.43},
  "xai": {"evidence": "strong"},
  "vlm": {"prediction": "positive", "confidence": 0.76},
  "rag": null,
  "conflicts": ["CNN_vs_Swin"],
  "tools_used": ["XAI", "VLM"],
  "next_action": "VERIFY",
  "step": 3,
  "status": "UNDER_REVIEW"
}
```

### 7.2 Evidence-Gap Types

```text
EVIDENCE_GAP_VISUAL      → XAI
EVIDENCE_GAP_SEMANTIC    → VLM
EVIDENCE_GAP_CONTEXTUAL  → RAG
EVIDENCE_CONFLICT        → VERIFICATION
```

### 7.3 Tool-Selection Rules (decision table)

| Situation | Agent Action | Tool |
|---|---|---|
| High confidence + CNN/Swin agree | Stop | none → RESULT |
| Low confidence, no disagreement | Gather visual evidence | XAI |
| CNN/Swin disagreement | Diagnose visual gap, then seek independent view | XAI → VLM |
| VLM disagrees with visual models | Seek external context | RAG |
| Evidence still insufficient after all tools | Reconcile | VERIFICATION |
| Persistent conflict after verification | Stop, defer to human | ESCALATE |

### 7.4 Agent Loop

```text
Evidence State → Agent Policy → Select Tool → Tool Result → Update State
        → Re-evaluate State
              ├── Evidence sufficient → RESULT
              └── Evidence insufficient → Select next tool → Update State
                        → max_steps reached? ──NO──→ continue
                                              └─YES──→ ESCALATE
```

### 7.5 Maximum Steps

Fix `max_steps` (recommend 4–6) before experiments begin; record it in `configs/agent.yaml`. Do not tune it against the test set.

### 7.6 Verification

Cross-source agreement check across whichever evidence has been gathered so far (majority support / persistent single-source conflict). Verification never uses a model to verify its own output (no circular verification).

### 7.7 Escalation

Triggered by: max_steps reached with unresolved conflict, or verification explicitly fails to reconcile. Escalated cases are logged with full evidence trace for error analysis (Phase 18).

---

# 8. Critical 3-Arm Experiment

This is the central experiment of the entire project. Do not treat it as "just another ablation."

```text
ARM A — FIXED PIPELINE
X-ray → CNN+Swin → XAI → VLM → RAG → Verification → Result
(every tool always called, in fixed order)

ARM B — STATELESS ADAPTIVE AGENT
X-ray → CNN+Swin → Evidence State (LATEST RESULT ONLY, no history) → Agent → adaptive tool choice → Result/Escalate
(adaptive routing, but the agent only ever sees the most recent tool output — no accumulated conflict history)

ARM C — STATEFUL ADAPTIVE AGENT (proposed system)
X-ray → CNN+Swin → Evidence State (PERSISTENT, accumulates all tool outputs + conflict history) → Agent → adaptive tool choice → Verification → Result/Escalate
```

### 8.1 What Each Arm Isolates

- **A vs B:** tests H-Adapt (does adaptive routing alone help, regardless of memory?)
- **B vs C:** tests H-State (does *persistent* evidence — not just adaptive routing — improve outcomes?)
- **A vs C:** overall system benefit

### 8.2 Metrics (identical across all three arms)

Classification quality, calibration (ECE), conflict-detection rate, conflict-resolution rate, escalation precision/recall, tool calls (total, VLM-specific, RAG-specific), latency.

### 8.3 Expected Result Pattern

`Stateful (C) > Stateless (B) > Fixed (A)` on conflict-resolution and evidence-quality metrics, with B ≥ A on tool-efficiency metrics. If this ordering does not hold, report the actual ordering — do not adjust the experiment post hoc to force it.

### 8.4 Implementation Note

Arms A, B, and C should share the exact same tool implementations (XAI, VLM, RAG, verification) and differ *only* in the policy/state layer. If they differ in anything else, the comparison is confounded.

---

# 9. Related Work — Explicit Distinctions

| Existing work | Main mechanism | Our distinction |
|---|---|---|
| **META-CXR** (2025) | Multi-encoder VLM: CNN+ViT+Swin → MHCAC classification → META-Former report generation, evaluated on MIMIC-CXR | We transfer the multi-encoder classification framing to NIH ChestX-ray14 and add adaptive evidence orchestration; we do not reproduce its report-generation benchmark |
| **CheX-DS** (2025) | DenseNet + Swin ensemble on NIH ChestX-ray14, 83.76% macro AUC | We use CNN/Transformer comparison as visual *foundation*, not novelty; fusion is not our claimed contribution |
| **AT-CXR** (Aug 2025) | Single-model uncertainty/distributional-fit → rule-based or LLM router → accept/abstain, on NIH ChestX-ray14 subset | We use multi-source disagreement across heterogeneous evidence + persistent state, not single-model confidence routing |
| **MedRAX** (Feb 2025, ICML) | LLM/ReAct loop dynamically selects among many CXR tools, open-source | We use a small, interpretable, rule-based evidence policy — explicitly not an LLM planner — and isolate the effect of statefulness itself via the 3-arm experiment |

Cite all four explicitly in the related-work section. Do not omit AT-CXR or MedRAX — a reviewer familiar with 2025 agentic-CXR literature will ask about them if they're missing.

---

# 10. Novelty Claim

> We investigate an interpretable, stateful evidence-reconciliation agent that uses disagreement among heterogeneous visual and multimodal evidence sources to determine which additional evidence tool should be invoked, when sufficient evidence has been obtained, and when a case should be escalated — and we experimentally isolate the contribution of statefulness itself from adaptive routing alone.

**Explicitly not claimed:**
- CNN+Swin fusion is not novel (CheX-DS precedent).
- "Agentic AI for chest X-rays" is not novel by itself (AT-CXR, MedRAX precedent).
- The VLM and RAG components are not novel individually — they are evidence-producing tools.

---

# 11. Full Implementation Phases

Each phase uses the fixed template below. `Antigravity responsibilities` = what can reasonably be delegated to AI-assisted coding; `Manual verification` = what you personally must check before moving on.

---

## Phase 0 — Server & Environment Setup

1. **Objective:** Working NVIDIA GPU environment with reproducible dependencies.
2. **Why this phase exists:** Nothing downstream is trustworthy without a stable, versioned environment.
3. **Prerequisites:** none.
4. **Hardware/software:** NVIDIA server access; Python 3.10+; CUDA matching GPU driver; git.
5. **Exact tasks:** create virtualenv/conda env; install PyTorch (CUDA build), timm, torchvision; set up `requirements.txt`; init git repo.
6. **Files:** `requirements.txt`, `.gitignore`, `README.md`, `AGENTS.md` (Section placeholder).
7. **Step-by-step:** (a) provision env, (b) verify `torch.cuda.is_available()`, (c) pin versions, (d) commit initial repo skeleton.
8. **Commands:** `python -m venv .venv`, `pip install -r requirements.txt`, `python -c "import torch;print(torch.cuda.is_available())"`.
9. **Expected output:** GPU visible, all imports succeed.
10. **Metrics:** n/a.
11. **Checkpoint:** `torch.cuda.is_available() == True`.
12. **Failure conditions:** CUDA/driver mismatch; blocked network for package installs.
13. **Antigravity responsibilities:** generate `requirements.txt`, scaffold repo structure.
14. **Manual verification:** confirm actual GPU model/VRAM available (needed for later batch-size decisions).
15. **What must be recorded for the paper:** hardware spec (GPU model, VRAM, CUDA version).
16. **Go/No-Go:** GO only if GPU check passes.

---

## Phase 1 — Dataset Acquisition & Audit

1. **Objective:** Verified NIH ChestX-ray14 data with a complete `DATASET_AUDIT.md`.
2. **Why this phase exists:** No training should begin on an unverified dataset (per Section 2's dataset lesson).
3. **Prerequisites:** Phase 0 complete.
4. **Hardware/software:** ~50GB+ storage for full dataset; `kagglehub` or NIH's official box link.
5. **Exact tasks:** download NIH ChestX-ray14; inspect `Data_Entry_2017.csv`; count images/patients/labels; check for corrupted files; check class imbalance; confirm official train/test split file exists (`train_val_list.txt`, `test_list.txt`).
6. **Files:** `docs/DATASET_AUDIT.md`, `configs/dataset.yaml`.
7. **Step-by-step:** (a) download, (b) verify file counts match documentation (~112,120 images), (c) parse label column, (d) compute per-class frequency, (e) check patient ID field, (f) document findings.
8. **Commands:** dataset download script; `pandas` audit script (row counts, `nunique()` on patient ID, label value counts).
9. **Expected output:** complete `DATASET_AUDIT.md`.
10. **Metrics:** total images, unique patients, per-class positive counts, missing/corrupted file count.
11. **Checkpoint:** audit file complete and reviewed by you personally.
12. **Failure conditions:** image count mismatch, corrupted archive, missing official split file.
13. **Antigravity responsibilities:** write the audit script, generate the report skeleton.
14. **Manual verification:** you personally check the audit numbers against the NIH paper's published stats (~112,120 / ~30,805) before trusting them.
15. **What must be recorded for paper:** final dataset statistics table.
16. **Go/No-Go:** GO only if audit passes and matches known dataset stats.

---

## Phase 2 — Preprocessing & Patient-Safe Split

1. **Objective:** Reproducible, leakage-free train/val/test splits and image preprocessing pipeline.
2. **Why this phase exists:** Patient leakage silently invalidates every downstream result.
3. **Prerequisites:** Phase 1 audit passed.
4. **Hardware/software:** PIL/OpenCV, pandas.
5. **Exact tasks:** implement patient-level split (use official split as base, verify no patient overlap); build image preprocessing (resize, normalize) matching chosen model input sizes; write automated leakage test.
6. **Files:** `data/splits/{train,val,test}.csv`, `src/data/preprocessing.py`, `tests/test_leakage.py`.
7. **Step-by-step:** (a) load official split, (b) verify patient-set disjointness programmatically, (c) build `Dataset`/`DataLoader` classes, (d) run leakage test.
8. **Commands:** `pytest tests/test_leakage.py`.
9. **Expected output:** leakage test passes; DataLoader yields correctly shaped batches.
10. **Metrics:** patient overlap count (must be 0).
11. **Checkpoint:** `Patients(train) ∩ Patients(val) ∩ Patients(test) = ∅` verified.
12. **Failure conditions:** any patient overlap; DataLoader shape mismatches.
13. **Antigravity responsibilities:** write Dataset class and leakage test.
14. **Manual verification:** personally re-run the leakage test after any dataset code change, not just once.
15. **What must be recorded:** split sizes, leakage-test pass confirmation.
16. **Go/No-Go:** GO only if `Dataset audit PASS`, `Patient split PASS`, `Leakage test PASS` (Section 2.5).

---

## Phase 3 — Visual Model Benchmark (5 Models)

1. **Objective:** Trained, comparably-evaluated ResNet50, DenseNet121, EfficientNet, ConvNeXt-Tiny, Swin-Tiny.
2. **Why this phase exists:** Establishes the defensible visual foundation (Section 3).
3. **Prerequisites:** Phase 2 complete.
4. **Hardware/software:** `timm` model zoo; ImageNet-pretrained weights.
5. **Exact tasks:** implement shared training loop; train each of the 5 models identically (same augmentation, loss — weighted BCE for imbalance, optimizer, schedule); log per-model metrics.
6. **Files:** `src/models/{resnet,densenet,efficientnet,convnext,swin}.py`, `src/train.py`, `experiments/benchmark_results.csv`.
7. **Step-by-step:** (a) build shared trainer, (b) train on Stage B (10k-20k) subset first as smoke test, (c) scale to Stage C (~50k) for real comparison, (d) log AUROC/F1 per model.
8. **Commands:** `python src/train.py --model resnet50 --config configs/train.yaml` (repeat per model).
9. **Expected output:** 5 trained checkpoints + comparison table.
10. **Metrics:** macro AUROC (primary), per-class AUROC, macro F1.
11. **Checkpoint:** all 5 models train to convergence without divergence/NaN loss.
12. **Failure conditions:** any model fails to converge; GPU OOM (reduce batch size, not model count).
13. **Antigravity responsibilities:** boilerplate training loop, logging, checkpoint saving.
14. **Manual verification:** personally inspect loss curves for at least the best and worst performing model.
15. **What must be recorded:** full benchmark table for the paper's Table 1.
16. **Go/No-Go:** GO only if `All baseline models train successfully` (Section 2.5/checkpoints).

---

## Phase 4 — Best CNN Selection + Swin Comparison

1. **Objective:** Identify best CNN by macro AUROC; formally compare vs Swin-Tiny.
2. **Why this phase exists:** Feeds Layer 1 output into Layer 2/3; needed before fusion decision.
3. **Prerequisites:** Phase 3 complete.
4. **Hardware/software:** same as Phase 3.
5. **Exact tasks:** rank Phase 3 results; select best CNN; run best-CNN-vs-Swin statistical comparison (bootstrap CI on AUROC difference).
6. **Files:** `experiments/best_cnn_vs_swin.csv`.
7. **Step-by-step:** (a) rank table, (b) bootstrap CI, (c) document decision.
8. **Commands:** analysis notebook/script.
9. **Expected output:** documented best-CNN choice with justification.
10. **Metrics:** AUROC delta + confidence interval.
11. **Checkpoint:** best CNN formally recorded in `configs/model.yaml`.
12. **Failure conditions:** near-identical performance across all CNNs (still fine — pick most efficient one and say so).
13. **Antigravity responsibilities:** bootstrap CI script.
14. **Manual verification:** sanity-check the winning model against known literature ranges (macro AUROC roughly 0.80–0.86 is the expected band for this dataset).
15. **What must be recorded:** justification paragraph for best-CNN choice.
16. **Go/No-Go:** GO only if `Visual model is stable` (per Section 2.5).

---

## Phase 5 — Optional CNN–Swin Fusion

1. **Objective:** Test whether simple feature fusion improves over best-CNN and Swin alone.
2. **Why this phase exists:** Section 3.3 — fusion is optional infrastructure, not novelty; must be tested honestly, not assumed to help.
3. **Prerequisites:** Phase 4 complete.
4. **Hardware/software:** same as Phase 3.
5. **Exact tasks:** implement simple fusion (projection → concatenation → classifier head, no cross-attention); train; compare A) best CNN, B) Swin, C) fusion.
6. **Files:** `src/models/fusion.py`, `experiments/fusion_results.csv`.
7. **Step-by-step:** (a) freeze best-CNN and Swin backbones or fine-tune jointly (record which), (b) train fusion head, (c) compare against A/B.
8. **Commands:** `python src/train.py --model fusion`.
9. **Expected output:** three-way comparison table.
10. **Metrics:** macro AUROC for A vs B vs C.
11. **Checkpoint:** fusion decision documented (mandatory or dropped) regardless of outcome.
12. **Failure conditions:** fusion does not improve — this is an acceptable, reportable result (Risk 7, Section 13).
13. **Antigravity responsibilities:** fusion module implementation.
14. **Manual verification:** confirm the comparison used identical eval protocol across A/B/C.
15. **What must be recorded:** honest fusion result, whether kept in final pipeline.
16. **Go/No-Go:** proceed regardless of outcome; do not let fusion block Layer 2/3 work.

---

## Phase 6 — Calibration

1. **Objective:** Calibrated confidence scores feeding the Evidence State.
2. **Why this phase exists:** Agent decisions in Layer 3 depend directly on confidence being meaningful.
3. **Prerequisites:** Phase 4/5 (final visual model chosen).
4. **Hardware/software:** standard.
5. **Exact tasks:** compute ECE/reliability diagrams on raw model; apply temperature scaling fit on validation only; recompute ECE.
6. **Files:** `src/calibration.py`, `experiments/calibration_report.md`.
7. **Step-by-step:** (a) compute pre-calibration ECE, (b) fit temperature on val split, (c) recompute post-calibration ECE, (d) freeze temperature parameter.
8. **Commands:** calibration script.
9. **Expected output:** reduced ECE post-calibration.
10. **Metrics:** ECE, Brier score, reliability diagram.
11. **Checkpoint:** temperature parameter frozen and saved to config.
12. **Failure conditions:** calibration fit on test data (leakage) — must not happen.
13. **Antigravity responsibilities:** temperature scaling implementation.
14. **Manual verification:** personally confirm the calibration script only touches the validation split.
15. **What must be recorded:** pre/post ECE numbers.
16. **Go/No-Go:** GO once calibrated confidence is available for Evidence State.

---

## Phase 7 — XAI

1. **Objective:** Grad-CAM (CNN) and an attention/evidence visualization method (Swin) as a queryable evidence tool.
2. **Why this phase exists:** First evidence-producing tool in Layer 2; feeds `EVIDENCE_GAP_VISUAL` in the agent policy.
3. **Prerequisites:** Phase 4/5 model frozen.
4. **Hardware/software:** `pytorch-grad-cam` or equivalent.
5. **Exact tasks:** implement Grad-CAM for CNN; implement Swin-appropriate attention visualization; define "evidence strength" scoring (e.g., activation concentration in plausible anatomical region) that the agent can consume as a discrete signal (weak/moderate/strong).
6. **Files:** `src/xai/gradcam.py`, `src/xai/swin_attention.py`, `src/xai/evidence_strength.py`.
7. **Step-by-step:** (a) implement CAM generation, (b) define and validate evidence-strength scoring on a hand-checked sample, (c) wrap as a callable tool matching the agent's tool interface.
8. **Commands:** XAI generation script + sample visualization notebook.
9. **Expected output:** callable `xai_tool(image, model) -> {"evidence": "strong"/"moderate"/"weak"}`.
10. **Metrics:** qualitative spot-check against known abnormality regions (not a formal metric, but document a sample of N cases reviewed).
11. **Checkpoint:** tool interface matches Section 7.1 schema exactly.
12. **Failure conditions:** evidence-strength scoring produces near-constant output (uninformative signal).
13. **Antigravity responsibilities:** CAM generation code, visualization scripts.
14. **Manual verification:** personally review a sample of Grad-CAM overlays for plausibility (this is where "treat XAI as evidence, not clinical truth" gets tested).
15. **What must be recorded:** example figures for the paper, evidence-strength scoring definition.
16. **Go/No-Go:** GO once tool interface is stable and validated on a hand-checked sample.

---

## Phase 8 — VLM Integration

1. **Objective:** Working pretrained-VLM evidence tool producing the Section 5.3 JSON schema.
2. **Why this phase exists:** Second evidence-producing tool; feeds `EVIDENCE_GAP_SEMANTIC`.
3. **Prerequisites:** Phase 7 tool interface pattern established.
4. **Hardware/software:** chosen pretrained VLM (record exact model per Section 4), GPU memory budget for inference.
5. **Exact tasks:** select and record exact VLM (Section 4); implement structured-output prompting/parsing; implement output validation (reject malformed responses) before writing to Evidence State; check and document pretraining-data overlap.
6. **Files:** `src/tools/vlm.py`, `docs/VLM_SPEC.md` (fills Section 4 of this document).
7. **Step-by-step:** (a) select VLM, (b) test structured-output reliability on a sample batch, (c) implement JSON schema validation with retry/fallback, (d) wrap as callable tool.
8. **Commands:** VLM inference script + validation test.
9. **Expected output:** callable `vlm_tool(image) -> schema-conformant JSON`, with a measured malformed-output rate.
10. **Metrics:** structured-output success rate (target: reliably high before proceeding — this is a go/no-go criterion), inference latency.
11. **Checkpoint:** `Structured output is reliable` (Section 2.5).
12. **Failure conditions:** frequent malformed/unparseable output; VLM effectively becoming the de facto final classifier if the agent starts routing to it too often (watch for this in Phase 15).
13. **Antigravity responsibilities:** prompt engineering, JSON schema validation/retry logic.
14. **Manual verification:** personally review a sample of VLM outputs against the actual images for plausibility.
15. **What must be recorded:** VLM spec (Section 4 filled in), structured-output success rate.
16. **Go/No-Go:** do not route the agent through the VLM until this checkpoint passes.

---

## Phase 9 — RAG Pipeline

1. **Objective:** Working RAG tool over a frozen, approved external medical-knowledge corpus.
2. **Why this phase exists:** Third evidence-producing tool; feeds `EVIDENCE_GAP_CONTEXTUAL`; must not be decorative.
3. **Prerequisites:** Phase 8 tool pattern established.
4. **Hardware/software:** embedding model, FAISS (or equivalent vector index).
5. **Exact tasks:** select and document the exact corpus source (e.g., public disease-definition references — record the actual source, not "medical knowledge" generically); clean/chunk/embed; build vector index; implement retriever; implement and enforce the leakage rule (Section 2.6, Section 10 of prior audit).
6. **Files:** `src/tools/rag.py`, `docs/RAG_CORPUS.md` (source, license, freeze date).
7. **Step-by-step:** (a) select corpus, (b) clean/chunk, (c) embed + index, (d) freeze corpus (record checksum/version), (e) implement retrieval + leakage check.
8. **Commands:** corpus build script, retrieval test script.
9. **Expected output:** callable `rag_tool(query) -> retrieved context passages`.
10. **Metrics:** retrieval relevance spot-check (qualitative), corpus size/coverage.
11. **Checkpoint:** `Corpus frozen`, `No evaluation leakage` (Section 2.5).
12. **Failure conditions:** corpus overlaps test-set content in any way; retrieval returns irrelevant passages consistently.
13. **Antigravity responsibilities:** chunking/embedding/indexing pipeline.
14. **Manual verification:** personally confirm the corpus source and that it contains zero content derived from the NIH test split or VinDr-CXR.
15. **What must be recorded:** corpus provenance, freeze checksum.
16. **Go/No-Go:** do not include RAG in final evaluation until this checkpoint passes.

---

## Phase 10 — Evidence State Implementation

1. **Objective:** Working persistent Evidence State object matching Section 7.1 schema.
2. **Why this phase exists:** The data structure the entire Layer 3 depends on.
3. **Prerequisites:** Phases 7–9 tools return schema-conformant output.
4. **Hardware/software:** none beyond standard Python.
5. **Exact tasks:** implement `EvidenceState` class (init, update-from-tool-result, conflict detection, serialization); implement conflict detection logic (e.g., CNN vs Swin disagreement threshold).
6. **Files:** `src/agent/evidence_state.py`, `tests/test_evidence_state.py`.
7. **Step-by-step:** (a) implement class, (b) implement conflict detection, (c) unit test against the worked examples in Section 8.1 of the design discussion (agree case, disagreement case).
8. **Commands:** `pytest tests/test_evidence_state.py`.
9. **Expected output:** state object correctly reproduces the worked examples (Section 7.1 JSON).
10. **Metrics:** unit test pass rate.
11. **Checkpoint:** all unit tests pass.
12. **Failure conditions:** conflict detection fails to flag known disagreement cases.
13. **Antigravity responsibilities:** class implementation, unit tests.
14. **Manual verification:** personally trace through one real case by hand and compare to the code's output.
15. **What must be recorded:** schema definition for the paper's methods section.
16. **Go/No-Go:** GO once unit tests pass.

---

## Phase 11 — Fixed Pipeline (Arm A)

1. **Objective:** Working fixed sequential pipeline as the experimental baseline.
2. **Why this phase exists:** Arm A of the critical 3-arm experiment (Section 8).
3. **Prerequisites:** Phases 6–10 complete.
4. **Hardware/software:** standard.
5. **Exact tasks:** implement fixed sequence: CNN+Swin → XAI → VLM → RAG → Verification → Result, always calling every tool regardless of confidence.
6. **Files:** `src/pipelines/fixed_pipeline.py`.
7. **Step-by-step:** wire tools in fixed order; no conditional logic.
8. **Commands:** `python src/pipelines/fixed_pipeline.py --input <case>`.
9. **Expected output:** end-to-end result for a sample case, with full tool-call log.
10. **Metrics:** tool-call count (should equal total tools every time, by construction).
11. **Checkpoint:** runs end-to-end on a sample batch without error.
12. **Failure conditions:** any tool call failure breaks the whole pipeline (add basic error handling).
13. **Antigravity responsibilities:** pipeline wiring.
14. **Manual verification:** confirm this arm literally always calls all tools (this is the control condition — it must not accidentally be adaptive).
15. **What must be recorded:** baseline tool-call counts and latency.
16. **Go/No-Go:** GO once stable on sample batch.

---

## Phase 12 — Stateless Adaptive Agent (Arm B)

1. **Objective:** Adaptive agent that selects tools per the decision table (Section 7.3) but retains **no** history beyond the latest tool result.
2. **Why this phase exists:** Isolates H-Adapt from H-State (Section 8.1).
3. **Prerequisites:** Phase 11 complete, Phase 10 Evidence State available (but deliberately restricted here).
4. **Hardware/software:** standard.
5. **Exact tasks:** implement policy using Section 7.3's rules, but the state passed to the policy each step contains only the most recent tool output, not accumulated conflicts/history.
6. **Files:** `src/agent/stateless_policy.py`, `src/pipelines/stateless_agent.py`.
7. **Step-by-step:** (a) implement policy function operating on latest-result-only input, (b) wire into agent loop (Section 7.4) with the history mechanism explicitly disabled/stripped, (c) test on sample cases.
8. **Commands:** `python src/pipelines/stateless_agent.py --input <case>`.
9. **Expected output:** adaptive tool-call trace with no evidence history retained between steps.
10. **Metrics:** tool-call count (expected lower than Arm A on easy cases).
11. **Checkpoint:** confirmed via code review that no state persists across steps beyond the latest result.
12. **Failure conditions:** history accidentally leaks through shared mutable state — check this carefully, it is the easiest bug that would invalidate the whole B-vs-C comparison.
13. **Antigravity responsibilities:** policy implementation.
14. **Manual verification:** personally verify (by inspecting the state object at each step) that Arm B genuinely has no memory — this is the most important manual check in the whole project, since the B-vs-C contrast is your primary result.
15. **What must be recorded:** confirmation that statelessness is real, not just nominal.
16. **Go/No-Go:** do not proceed to Phase 15 comparison until this is verified.

---

## Phase 13 — Stateful Adaptive Agent (Arm C — Proposed System)

1. **Objective:** Full proposed system: adaptive routing + persistent Evidence State + conflict history.
2. **Why this phase exists:** This is the actual research contribution (Layer 3, Section 6).
3. **Prerequisites:** Phase 12 complete and verified.
4. **Hardware/software:** standard.
5. **Exact tasks:** implement full agent loop (Section 7.4) with persistent `EvidenceState` (Phase 10) accumulating across steps; implement verification and escalation (Section 7.6–7.7); wire max_steps.
6. **Files:** `src/agent/policy.py`, `src/agent/loop.py`, `src/pipelines/stateful_agent.py`.
7. **Step-by-step:** (a) implement full policy using Section 7.3 table, (b) implement loop with persistent state accumulation, (c) implement verification, (d) implement escalation on max_steps, (e) test against the worked examples from the design discussion (agree case, disagreement→XAI→VLM→verification case).
8. **Commands:** `python src/pipelines/stateful_agent.py --input <case>`.
9. **Expected output:** correctly reproduces the two worked examples from Section 7 exactly (same tool sequence, same final status).
10. **Metrics:** tool-call count, resolution rate, escalation rate on a dev batch.
11. **Checkpoint:** `Different evidence states produce different tool choices` (Section 2.5) — verify this is actually true, not assumed.
12. **Failure conditions:** agent behaves identically regardless of evidence state (would mean it's not genuinely stateful/adaptive — see Risk 4).
13. **Antigravity responsibilities:** loop/policy implementation.
14. **Manual verification:** personally trace 5–10 real cases end-to-end and confirm the agent's decisions make sense given the evidence at each step.
15. **What must be recorded:** full agent trace examples for the paper's qualitative results section (Section 47 concept from earlier plan).
16. **Go/No-Go:** do not claim agentic novelty until this checkpoint passes (Section 2.5).

---

## Phase 14 — Verification & Escalation Hardening

1. **Objective:** Robust, non-circular verification logic and well-calibrated escalation.
2. **Why this phase exists:** Weak verification would make Arm C's advantage illusory.
3. **Prerequisites:** Phase 13 complete.
4. **Hardware/software:** standard.
5. **Exact tasks:** stress-test verification against synthetic conflict cases; confirm no model verifies its own output; tune escalation criteria on validation data only.
6. **Files:** `src/agent/verification.py` (hardened), `tests/test_verification.py`.
7. **Step-by-step:** (a) construct synthetic conflict scenarios, (b) run through verification, (c) check escalation triggers correctly.
8. **Commands:** `pytest tests/test_verification.py`.
9. **Expected output:** verification correctly resolves resolvable conflicts and correctly escalates unresolvable ones in synthetic tests.
10. **Metrics:** synthetic-test pass rate.
11. **Checkpoint:** no circular verification (a model is never used to verify its own prediction).
12. **Failure conditions:** verification logic implicitly re-uses the same model's own confidence as its "independent" check.
13. **Antigravity responsibilities:** synthetic test-case generation.
14. **Manual verification:** personally audit the verification code path for circularity.
15. **What must be recorded:** verification logic description for methods section.
16. **Go/No-Go:** GO once synthetic tests pass and circularity is ruled out.

---

## Phase 15 — Critical 3-Arm Experiment

1. **Objective:** Run Arms A, B, C on an identical evaluation set and test H-Adapt and H-State.
2. **Why this phase exists:** This is the paper's central result (Section 8).
3. **Prerequisites:** Phases 11–14 complete and individually verified.
4. **Hardware/software:** full evaluation compute budget.
5. **Exact tasks:** run all three arms on the same frozen test set; compute all metrics (Section 8.2) per arm; run statistical significance tests on key deltas (B vs A, C vs B).
6. **Files:** `experiments/three_arm_results.csv`, `experiments/three_arm_analysis.ipynb`.
7. **Step-by-step:** (a) freeze eval set, (b) run Arm A, (c) run Arm B, (d) run Arm C, (e) compute metrics, (f) statistical comparison, (g) write up whichever ordering actually resulted.
8. **Commands:** run scripts per arm + analysis notebook.
9. **Expected output:** full comparison table + significance tests.
10. **Metrics:** all metrics from Section 8.2, per arm.
11. **Checkpoint:** `Fixed vs adaptive comparison completed` (Section 2.5).
12. **Failure conditions:** confounded comparison (arms differ in more than policy/state — re-check Phase 8.4's implementation note).
13. **Antigravity responsibilities:** batch-running scripts, result aggregation.
14. **Manual verification:** personally confirm the three arms used byte-identical tool implementations and the same eval set.
15. **What must be recorded:** the full results table — this is the paper's primary results table, report the actual ordering even if it isn't Stateful > Stateless > Fixed.
16. **Go/No-Go:** do not claim Agent improvement in the paper until this phase's results exist (Section 2.5).

---

## Phase 16 — Additional Ablations

1. **Objective:** 4–6 supporting ablations beyond the 3-arm experiment.
2. **Why this phase exists:** Strengthens the causal story around H-State/H-Adapt.
3. **Prerequisites:** Phase 15 complete.
4. **Hardware/software:** standard.
5. **Exact tasks:** run: (a) with/without VLM, (b) with/without verification, (c) different max_steps values, (d) with/without XAI evidence input to policy, (e) fusion vs no-fusion visual backbone feeding the agent.
6. **Files:** `experiments/ablations.csv`.
7. **Step-by-step:** toggle one component at a time, holding everything else at the Arm C configuration.
8. **Commands:** ablation runner script with config flags.
9. **Expected output:** ablation table.
10. **Metrics:** same metric set as Phase 15, per ablation.
11. **Checkpoint:** each ablation isolates exactly one variable.
12. **Failure conditions:** ablation accidentally changes more than one thing.
13. **Antigravity responsibilities:** config-flag-driven ablation runner.
14. **Manual verification:** personally check each ablation's diff against the Arm C baseline config.
15. **What must be recorded:** ablation table for the paper.
16. **Go/No-Go:** proceed to external validation once ablations are documented.

---

## Phase 17 — External Validation (VinDr-CXR)

1. **Objective:** Test generalization on VinDr-CXR where labels are compatible.
2. **Why this phase exists:** Demonstrates the system isn't overfit to NIH ChestX-ray14's specific label noise/distribution.
3. **Prerequisites:** Phase 15/16 complete on NIH.
4. **Hardware/software:** VinDr-CXR download.
5. **Exact tasks:** build label-compatibility map (Section 2.2); run best system config (Arm C) on VinDr-CXR compatible subset; report results separately (do not merge with NIH numbers).
6. **Files:** `docs/VINDR_LABEL_MAP.md`, `experiments/external_validation.csv`.
7. **Step-by-step:** (a) map labels, (b) run inference (no retraining unless explicitly testing domain adaptation), (c) report metrics.
8. **Commands:** external validation script.
9. **Expected output:** external validation table, expected to show some performance drop (normal and reportable).
10. **Metrics:** same classification metrics, on compatible label subset only.
11. **Checkpoint:** label map reviewed for correctness before running.
12. **Failure conditions:** forcing incompatible label mappings to inflate coverage.
13. **Antigravity responsibilities:** inference script reuse.
14. **Manual verification:** personally review the label-compatibility map for semantic correctness.
15. **What must be recorded:** external validation table + honest discussion of performance gap.
16. **Go/No-Go:** proceed to error analysis regardless of magnitude of drop, as long as it's honestly reported.

---

## Phase 18 — Error Analysis

1. **Objective:** Structured analysis of failure cases, especially escalated ones.
2. **Why this phase exists:** Required for a credible thesis discussion section; also validates escalation is meaningful.
3. **Prerequisites:** Phase 15–17 complete.
4. **Hardware/software:** standard.
5. **Exact tasks:** sample escalated cases and misclassified cases; manually review with agent traces; categorize failure types.
6. **Files:** `docs/ERROR_ANALYSIS.md`.
7. **Step-by-step:** (a) pull escalated/misclassified cases, (b) review traces, (c) categorize (e.g., genuinely ambiguous, VLM hallucination, XAI misleading, label noise).
8. **Commands:** case-review notebook.
9. **Expected output:** categorized failure taxonomy with examples.
10. **Metrics:** failure category frequencies.
11. **Checkpoint:** at least N cases reviewed (set N, e.g., 30–50) with documented categorization.
12. **Failure conditions:** skipping this in favor of only aggregate metrics.
13. **Antigravity responsibilities:** trace extraction/formatting for review.
14. **Manual verification:** this entire phase is manual review by you — do not delegate the judgment calls.
15. **What must be recorded:** failure taxonomy + illustrative examples for the paper.
16. **Go/No-Go:** proceed to API/UI once documented.

---

## Phase 19 — API (FastAPI)

1. **Objective:** A FastAPI service exposing the Arm C pipeline as a demo endpoint.
2. **Why this phase exists:** Needed for demo/defense, not for the research claim itself.
3. **Prerequisites:** Phase 13 stable.
4. **Hardware/software:** FastAPI, uvicorn.
5. **Exact tasks:** wrap stateful agent pipeline in an endpoint; return prediction + evidence trace + status.
6. **Files:** `api/main.py`, `api/schemas.py`.
7. **Step-by-step:** (a) define request/response schema, (b) wire pipeline call, (c) add basic error handling.
8. **Commands:** `uvicorn api.main:app --reload`.
9. **Expected output:** working `/predict` endpoint.
10. **Metrics:** response latency.
11. **Checkpoint:** endpoint returns valid response for sample images.
12. **Failure conditions:** endpoint exposes internal file paths/PHI-adjacent info in responses.
13. **Antigravity responsibilities:** FastAPI boilerplate.
14. **Manual verification:** confirm no dataset paths or sensitive info leak into API responses.
15. **What must be recorded:** API spec for the appendix.
16. **Go/No-Go:** optional gate — this does not block research completion.

---

## Phase 20 — UI (React)

1. **Objective:** Minimal React frontend demonstrating a case walkthrough (upload → prediction → evidence trace → status).
2. **Why this phase exists:** Demo/defense value; explicitly optional/lower priority than the research phases.
3. **Prerequisites:** Phase 19 API working.
4. **Hardware/software:** React, standard frontend tooling.
5. **Exact tasks:** build upload form, results display, evidence-trace visualization (show which tools were called and why).
6. **Files:** `ui/` React app.
7. **Step-by-step:** (a) scaffold app, (b) build upload/result flow, (c) visualize agent trace.
8. **Commands:** `npm run dev`.
9. **Expected output:** working demo UI against the local API.
10. **Metrics:** n/a (demo quality, not research metric).
11. **Checkpoint:** end-to-end demo works for at least 3 sample cases.
12. **Failure conditions:** none blocking — this is the most cuttable phase if time runs short.
13. **Antigravity responsibilities:** most of the frontend scaffolding.
14. **Manual verification:** personally demo it once before defense.
15. **What must be recorded:** screenshots for appendix/demo section.
16. **Go/No-Go:** cut first if timeline is at risk — never at the expense of Phase 15.

---

## Phase 21 — Paper/Thesis Writing & Publication Artifacts

1. **Objective:** Complete thesis chapters / paper draft with all required artifacts.
2. **Why this phase exists:** The actual deliverable.
3. **Prerequisites:** Phases 15–18 complete (research phases; 19–20 optional).
4. **Hardware/software:** LaTeX/Overleaf (per existing preference).
5. **Exact tasks:** write methods (Sections 1–10 of this document map directly to thesis sections), results (Phase 15–17 tables/figures), related work (Section 9 table), discussion (Phase 18 error analysis), limitations (VLM independence caveat from Section 4, dataset-transfer caveat from Section 1.6).
6. **Files:** `paper/` LaTeX project.
7. **Step-by-step:** draft chapter by chapter, feeding in artifacts generated by each research phase as they complete (do not wait until the end to start writing).
8. **Commands:** n/a.
9. **Expected output:** complete draft.
10. **Metrics:** n/a.
11. **Checkpoint:** every claim in the paper traces to a specific phase's recorded artifact (Section 15 of each phase template).
12. **Failure conditions:** any claim (e.g., "the agent is stateful," "AT-CXR is different because...") not backed by a phase artifact.
13. **Antigravity responsibilities:** LaTeX formatting, citation management, draft polishing.
14. **Manual verification:** you write and own the core arguments — this is the one phase that should not be primarily AI-drafted.
15. **What must be recorded:** the paper itself.
16. **Go/No-Go:** final submission.

---

# 12. Experiments & Ablations Summary

| # | Experiment | Tests | Phase |
|---|---|---|---|
| 1 | 5-model visual benchmark | which CNN backbone | 3 |
| 2 | Best CNN vs Swin vs fusion | transformer value, fusion value | 4–5 |
| 3 | Fixed vs Stateless vs Stateful (3-arm) | H-Adapt, H-State | 15 |
| 4 | With/without VLM | VLM contribution | 16 |
| 5 | With/without verification | verification contribution | 16 |
| 6 | Max-steps sweep | agent efficiency/quality tradeoff | 16 |
| 7 | External validation | generalization | 17 |

---

# 13. Evaluation Metrics

**Mandatory:** macro AUROC, per-class AUROC, macro F1, ECE, tool-call counts (total/VLM/RAG), conflict-detection rate, conflict-resolution rate, escalation precision/recall, latency.

**Optional:** PR-AUC, micro F1, Brier score.

---

# 14. Error Analysis

Covered in Phase 18. Output: `docs/ERROR_ANALYSIS.md` with categorized failure taxonomy.

---

# 15. External Validation

Covered in Phase 17. VinDr-CXR, label-compatible subset only.

---

# 16. API

Covered in Phase 19. Optional relative to research completion.

---

# 17. UI

Covered in Phase 20. Most cuttable phase under time pressure.

---

# 18. Storage / Hardware

| Item | Estimate |
|---|---|
| NIH ChestX-ray14 (full) | ~45GB |
| VinDr-CXR (compatible subset) | varies, budget ~20GB |
| 5 model checkpoints + fusion + agent artifacts | ~5–10GB |
| VLM (pretrained, inference only) | model-dependent, budget 10–30GB |
| RAG index (small curated corpus) | <1GB |
| **Recommended safe total** | **~150GB** |

Do not train VLM from scratch or fully fine-tune it by default (Section 24 of the prior detailed plan — LoRA only, Optional). Use transfer learning for all 5 visual backbones.

---

# 19. Timeline

| Milestone | Estimate |
|---|---|
| Prototype (Phases 0–13, smoke-scale) | 4–6 weeks |
| Research-complete (through Phase 18) | 8–12 weeks |
| Publication-quality (through Phase 21) | 12–16 weeks |

Biggest schedule risk: Phase 15 (3-arm experiment) — do not let VLM/RAG integration debugging (Phases 8–9) eat into this budget; those phases have hard go/no-go gates for exactly this reason.

---

# 20. Publication Artifacts Checklist

- [ ] `DATASET_AUDIT.md`
- [ ] `REPOSITORY_AUDIT.md`
- [ ] Benchmark table (Phase 3)
- [ ] Best-CNN-vs-Swin table (Phase 4)
- [ ] Fusion result (Phase 5, honest either way)
- [ ] Calibration report (Phase 6)
- [ ] 3-arm results table + significance tests (Phase 15) — **primary result**
- [ ] Ablation table (Phase 16)
- [ ] External validation table (Phase 17)
- [ ] Error analysis / failure taxonomy (Phase 18)
- [ ] Agent trace examples (Phase 13)
- [ ] Related-work distinction table (Section 9)

---

# 21. Risk Register

| Risk | Severity | Fix |
|---|---|---|
| VLM/visual-model distributional overlap undermines "heterogeneous evidence" framing | High | disclose in Section 4, discuss in limitations |
| Arm B accidentally retains state (confound) | Critical | Phase 12's manual verification step |
| Circular verification | Critical | Phase 14 audit |
| Novelty overlap with AT-CXR/MedRAX | Critical | Section 9 explicit distinction, cite both |
| Fusion doesn't improve | Medium | report honestly (Phase 5) |
| VLM becomes de facto final classifier | High | monitor VLM call frequency in Phase 15 results |
| Scope creep into UI/API at expense of Phase 15 | Medium | Phase 19–20 explicitly cuttable |
| Patient/report leakage | Critical | Phase 2 leakage test, Phase 9 corpus freeze |

---

# 22. Final Thesis/Paper Structure

1. Introduction (research question, Section 1.3)
2. Related Work (Section 9 table, expanded)
3. Dataset (Section 2)
4. Methods — Visual Foundation (Sections 3–6, Layer 1)
5. Methods — Evidence Intelligence (Section 5, Layer 2)
6. Methods — Agentic Intelligence (Section 7, Layer 3)
7. Experimental Design (Section 8, 3-arm setup)
8. Results (Phase 15–17 tables)
9. Ablations (Phase 16)
10. Error Analysis & Discussion (Phase 18)
11. Limitations (VLM independence caveat, dataset-transfer caveat, single-dataset primary training)
12. Conclusion
13. Appendix (API/UI, agent trace examples, VLM/RAG specs)

---

*End of master document. Update phases in place as they complete — record actual results directly into each phase's "What must be recorded" field rather than maintaining a separate results log.*
