# Agent Rules and Development Constraints

This file details the development policies, project constraints, and guidelines that must be adhered to at all times during the execution of this project.

## General Development Policies
- **Read Plan first:** Always read `MASTER_PROJECT_PLAN.md` before making any architectural changes.
- **Phase-by-Phase Execution:** Implement exactly one project phase at a time. Do not proceed to the next phase until the current phase checkpoint passes.
- **No Unilateral Research Changes:** Do not change the research design, model options, or evaluation protocol without explicit user approval.
- **Do Not Interfere:** Do not kill, stop, modify, or interfere with other users' processes running on the server (especially those on GPU 0).

## Data and Splitting Constraints
- **Leakage Prevention:** Never mix test split data into the training, validation, calibration, or RAG reference corpus.
- **Patient-Level Splits:** Ensure that patient subsets are strictly disjoint:
  $$Patients(\text{train}) \cap Patients(\text{validation}) = \emptyset$$
  $$Patients(\text{train}) \cap Patients(\text{test}) = \emptyset$$
  $$Patients(\text{validation}) \cap Patients(\text{test}) = \emptyset$$
  All patients must be split cleanly at the patient ID level, never at the image level.

## Orchestration and Agent Arm Constraints
- **Keep Implementations Identical:** Keep Arm A (Fixed Sequential), Arm B (Adaptive Stateless), and Arm C (Adaptive Stateful) tool implementations identical except for the intended policy/state difference.
- **No Stateless History:** Never give the stateless agent (Arm B) any persistent evidence history across tool calls.
- **No Circular Verification:** Never allow a tool to use its own output to verify itself.

## Experiment Logging
- **Reproducibility:** Record all important experiment configurations, hyper-parameters, hardware specs, and metric tables.
