# Evidence-Fused Neutrosophic Framework

This work presents a neutrosophic healthcare treatment selection approach using an MCDM framework that integrates Neutrosophic Logic, IVFS, DST, and an extended TOPSIS method. The proposed approach provides a structured mechanism to address uncertainty, indeterminacy, and conflict in expert evaluation, thereby enabling a flexible and robust framework for evidence-based decision-making in complex clinical environments.

## Brief Overview

The framework performs uncertainty-aware treatment ranking for healthcare decision support by combining:
- Neutrosophic representation of expert judgments
- Dempster-Shafer evidence fusion for multi-expert aggregation
- Interval-valued fuzzy modeling for uncertainty bounds
- TOPSIS-based relative closeness scoring for final ranking

## Repository Contents

- Code_Evidence-Fused Neutrosophic Framework.pyb`: Main implementation and analysis workflow.
- `results/`: Generated plots and figures (PNG/PDF).
- `outputs/`: Text outputs from execution, including Monte Carlo summaries and tabulated results.



## Setup for new user
### 1) Clone the repository

```bash
git clones https://github.com/mukesh311a/Evidence-Fused-Neutrosophic-Framework.git
cd Evidence-Fused-Neutrosophic-Framework
```

### 2) Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -U pip
pip install numpy pandas matplotlib seaborn jupyter ipython
```

### 4) Run the script

```bash
python "Downloads/final_code_paper.py"
```

### 5) Check outputs

- Figures: `Downloads/results/`
- Text log: `Downloads/outputs/run_output.txt`

##  Reproducible Setup
.

### A) Pin repository state

```bash
git fetch --all --tags
git checkout main
git pull --ff-only
```

Optional (exact run reproducibility):

```bash
git checkout <commit_sha>
```

### B) Create isolated environment

Option 1 (venv):

```bash
python -m venv .venv
```

Option 2 (conda):

```bash
conda create -n neutro-fw python=3.11 -y
conda activate neutro-fw
```

Then upgrade packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

### C) Install locked dependencies

Create `requirements.lock.txt` with:

```txt
numpy==2.1.1
pandas==2.2.2
matplotlib==3.9.2
seaborn==0.13.2
jupyter==1.1.1
ipython==8.27.0
```

Install:

```bash
pip install -r requirements.lock.txt
```

Verify:

```bash
python -c "import numpy,pandas,matplotlib,seaborn,IPython; print('numpy',numpy.__version__); print('pandas',pandas.__version__); print('matplotlib',matplotlib.__version__); print('seaborn',seaborn.__version__); print('ipython',IPython.__version__)"
```

### D) Optional deterministic runtime settings

Windows (PowerShell):

```powershell
$env:PYTHONHASHSEED="0"
$env:MPLCONFIGDIR="$PWD\.mplconfig"
```

Linux/macOS:

```bash
export PYTHONHASHSEED=0
export MPLCONFIGDIR="$PWD/.mplconfig"
```

### E) Run complete pipeline

```bash
python "Downloads/final_code_paper.py"
```

Expected artifacts:

- `Downloads/results/` (PNG/PDF figures)
- `Downloads/outputs/run_output.txt` (text outputs)





