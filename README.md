# Stateful Heterogeneous-Evidence Reconciliation for Uncertainty-Aware Chest X-Ray Disease Classification

This repository implements an evidence-aware stateful agentic framework for multi-label chest X-ray abnormality classification, based on the **META-CXR** methodology and evaluated on the **NIH ChestX-ray14** dataset.

## Research Objective
Build and evaluate a chest X-ray decision-support system in which a small, interpretable, rule-based agent accumulates heterogeneous evidence (visual classifiers, XAI, VLM, RAG) in a persistent per-case state, and uses disagreement/uncertainty in that state to decide what additional evidence to acquire, when to stop, and when to escalate — and to test whether **statefulness itself** (not just adaptive tool selection) is responsible for any observed benefit.

## Project Structure
- `configs/`: YAML configuration files for models and datasets.
- `data/`: Dataset storage folder (local dataset files excluded from git).
- `docs/`: Project documentation and reports (e.g., environment audit, dataset audits).
- `experiments/`: Experiment runs, metrics, and logs.
- `src/`: Core Python source code (data loading, preprocessing, model architectures, agent logic).
- `tests/`: Programmatic verification tests and split leakage testing.
- `api/`: API endpoint layer.
- `ui/`: User interface components.

## Environment Audit
The environment audit report is located at [docs/ENVIRONMENT_AUDIT.md](docs/ENVIRONMENT_AUDIT.md).
The Phase 0 completion report is located at [docs/PHASE_0_COMPLETION.md](docs/PHASE_0_COMPLETION.md).

## Verification
To run the environment verification script:
```bash
/opt/intel/oneapi/intelpython/envs/cuda-env/bin/python scripts/verify_environment.py
```
