# Environment Audit

## 1. Audit Date
- **Audit Date:** August 13, 2026, 11:26 AM IST (Local Server Time)

## 2. Server Hardware
- **OS Version:** Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- **Kernel Version:** Linux 6.8.0-124-generic
- **CPU Model:** Intel(R) Xeon(R) Gold 6430
- **CPU Cores/Threads:** 64 logical cores (2 sockets, 32 cores per socket, 1 thread per core)
- **Total System RAM:** 251 GiB
- **Available System RAM:** 223 GiB

## 3. GPU Configuration

| GPU | Model | VRAM | Driver | CUDA | Status |
|---|---|---|---|---|---|
| GPU 0 | NVIDIA L4 | 23034 MiB (~22.5 GB) | 550.144.03 | 12.4 | Active (95% Util, 3 python processes running) |
| GPU 1 | NVIDIA L4 | 23034 MiB (~22.5 GB) | 550.144.03 | 12.4 | Idle (0% Util, 0 python processes running) |

- **GPU 0 Running Processes:**
  - `ipykernel_launcher` (PID 2147085) - 4044 MiB
  - `ipykernel_launcher` (PID 2162747) - 3406 MiB
  - `stylegan2-ada-pytorch/train.py` (PID 2163707) - 2540 MiB
  - `Xorg` (PID 2356) - 4 MiB
- **GPU 1 Running Processes:**
  - `Xorg` (PID 2356) - 4 MiB

## 4. GPU PyTorch Verification
PyTorch is verified to successfully run on both GPUs. The verification was conducted using:
`/opt/intel/oneapi/intelpython/envs/cuda-env/bin/python`

### Verification Command Execution Results
```python
import torch
print('torch version:', torch.__version__)
print('torch cuda version:', torch.version.cuda)
print('cuda is available:', torch.cuda.is_available())
print('cuda device count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'Device {i}: {torch.cuda.get_device_name(i)}')
# GPU 0 check
try:
    x = torch.randn(100, 100, device='cuda:0')
    y = torch.matmul(x, x)
    print('GPU 0 computation success:', y.shape)
except Exception as e:
    print('GPU 0 computation failed:', e)
# GPU 1 check
try:
    a = torch.randn(100, 100, device='cuda:1')
    b = torch.matmul(a, a)
    print('GPU 1 computation success:', b.shape)
except Exception as e:
    print('GPU 1 computation failed:', e)
```

**Output:**
```text
torch version: 2.7.0+cu126
torch cuda version: 12.6
cuda is available: True
cuda device count: 2
Device 0: NVIDIA L4
Device 1: NVIDIA L4
GPU 0 computation success: torch.Size([100, 100])
GPU 1 computation success: torch.Size([100, 100])
```

- **PyTorch GPU Accessibility:** **PASS**. Both GPU 0 and GPU 1 are successfully accessible and performing tensor computations under PyTorch.

## 5. CPU / RAM
- **CPU Model:** Intel(R) Xeon(R) Gold 6430 (Sapphire Rapids architecture)
- **Cores/Threads:** 64 logical processors
- **System Memory (RAM):** 251 GiB total (94 GiB free, 131 GiB buff/cache, 223 GiB available)
- **Swap Space:** 119 GiB total (118 GiB free)

## 6. Storage

| Mount | Total | Used | Available | Usage | Filesystem Type |
|---|---:|---:|---:|---:|---|
| `/` | 319G | 202G | 102G | 67% | `ext4` (on `/dev/sdb4`) |
| `/boot` | 943M | 210M | 669M | 24% | `ext4` (on `/dev/sdb1`) |
| `/boot/efi` | 976M | 6.1M | 969M | 1% | `vfat` (on `/dev/sdb2`) |
| `/home` | 880G | 654G | 181G | 79% | `nfs4` (NFS mapped on `/dev/sda1` layout) |

- **Project Workspace Mount:** `/home` (specifically path `/home/23adr188/chest_xray_project`)
- **Storage Suitability Analysis:**
  - The project plan estimates a safe storage total of ~150 GB (NIH ChestX-ray14: ~45 GB, VinDr-CXR: ~20 GB, checkpoints: ~5-10 GB, VLM: ~10-30 GB, RAG: <1 GB).
  - The available storage on `/home` is **181 GB**.
  - **Verdict:** **SUFFICIENT**. The server has enough storage space to accommodate all dataset and model requirements.

## 7. Python Environment
- **Active System Python:** Python 3.11.9 (`/opt/intel/oneapi/intelpython/bin/python3`)
- **Target Project Python Environment:** Conda environment `cuda-env` (`/opt/intel/oneapi/intelpython/envs/cuda-env/bin/python`)
- **pip version:** 26.1.2 (for `base` and pointing to user `.local` site-packages)
- **Virtualenv / Conda availability:** Conda (24.11.3) is available. `virtualenv` is not installed on the system.

## 8. ML Dependencies
The status of dependencies checked inside the `cuda-env` target python environment:

- **PyTorch (torch):** Installed (`2.7.0+cu126`)
- **torchvision:** Installed (`0.22.0+cu126`)
- **timm:** **NOT Installed**
- **numpy:** Installed (`2.4.6`)
- **pandas:** Installed (`2.2.3`)
- **scikit-learn:** **NOT Installed**
- **OpenCV (cv2):** Installed (`5.0.0`)
- **PIL (Pillow):** Installed (`11.1.0`)
- **matplotlib:** Installed (`3.10.1`)
- **Jupyter:** Installed (IPython 9.1.0, notebook 7.4.0, jupyterlab 4.4.0)
- **pytest:** **NOT Installed**

## 9. Development Tools
- **git version:** 2.34.1
- **git configuration:** No user-level config (e.g., `user.name` / `user.email`) initialized.
- **gcc/g++:** 11.4.0
- **make:** 4.3
- **curl:** **NOT Installed**
- **wget:** 1.21.2
- **rsync:** 3.2.7
- **unzip:** 6.00
- **tar:** 1.34
- **available terminal/shell:** `/bin/bash`

## 10. Network Check
Network connectivity was tested using `wget --spider`:
- **google.com:** **PASS** (Successfully resolved and established connection)
- **github.com:** **PASS** (Successfully resolved and established connection)
- **kaggle.com:** **PASS** (Successfully resolved and connected; returned HTTP 404 in spider/HEAD mode which is standard Kaggle firewall behavior, confirming open routing)
- **Conclusion:** The server has full internet access and is capable of downloading external datasets and packages.

## 11. Current Project Workspace
- **Workspace Location:** `/home/23adr188/chest_xray_project`
- **Git Initialized:** No (fatal: not a git repository)
- **Existing files/folders:**
  - `MASTER_PROJECT_PLAN.md` (Size: 53901 bytes)
  - `.ipynb_checkpoints/`
- **Existing python environment:** None (local to the directory; we are utilizing the global `/opt/intel/oneapi/intelpython/envs/cuda-env` environment).
- **Existing configuration/metadata files:**
  - `requirements.txt`: **Missing**
  - `README.md`: **Missing**
  - `AGENTS.md`: **Missing**

## 12. Problems / Missing Dependencies

### PASS
- Dual NVIDIA L4 GPUs are present and fully function inside PyTorch.
- RAM (251 GiB total, 223 GiB available) and CPU (64-core Gold 6430) are excellent.
- Storage is sufficient (181 GiB available vs 150 GiB required).
- Network is online.
- PyTorch, torchvision, numpy, pandas, OpenCV, Pillow, matplotlib, and Jupyter are installed and working.

### WARNING
- `curl` is missing, but `wget` is available and covers downloading needs.
- No Git user config is initialized.
- System base environment does not have PyTorch; the `cuda-env` conda environment must be target-called explicitly.

### FAIL
- **timm**, **scikit-learn**, and **pytest** are missing in the `cuda-env` environment.
- Git repository is not initialized in the project directory.
- `requirements.txt`, `.gitignore`, `README.md`, and `AGENTS.md` are missing from the workspace.

## 13. Phase 0 Checkpoint
Evaluating: *"GPU visible, all imports succeed."*
- **Status:** **FAIL** (or **PARTIAL PASS**)
- **Rationale:** While the GPU is visible and accessible via the `cuda-env` conda environment, and PyTorch/torchvision/numpy/pandas/OpenCV/PIL import successfully, the checkpoint cannot be marked as fully passed because key imports (`timm` and `sklearn`) and the testing package (`pytest`) are missing. They must be installed before we can consider Phase 0 complete and proceed.

## 14. Recommended Next Step
- **Status:** **NOT READY TO PROCEED TO PHASE 1**
- **Action Plan to complete Phase 0:**
  1. Install the missing python libraries (`timm`, `scikit-learn`, and `pytest`) into the `cuda-env` environment.
  2. Initialize the Git repository in `/home/23adr188/chest_xray_project`.
  3. Create `requirements.txt` containing all dependencies.
  4. Scaffold the skeleton repository files: `.gitignore`, `README.md`, and `AGENTS.md`.
  5. Once these steps are performed, the project will be:
     **"READY FOR PHASE 1 — DATASET ACQUISITION & AUDIT"**
