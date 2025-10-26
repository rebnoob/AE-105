# AE105 Conda Environment Setup

This project uses a reproducible Conda environment defined in `environment.yml`.  
Follow the steps below to set up and run the environment on your own system.

---

## Setup Instructions

### 1. Install Conda
If you don’t already have Conda, install either:
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lightweight)
- [Anaconda](https://www.anaconda.com/download)

Check that Conda is available:
```bash
conda --version

---

### 2. Create the Environment

From the project root (where this file and `environment.yml` are located), run:

```bash
conda env create -f environment.yml
```

This will create a Conda environment named **`ae105`**.

---

### 3. Activate the Environment

Once created, activate it with:

```bash
conda activate ae105
```

You should see `(ae105)` in your terminal prompt.

---

### 4. Verify Installation

Run the following to confirm all key packages are installed:

```bash
python -c "import numpy, pandas, matplotlib, spiceypy; print('AE105 environment ready ✅')"
```

---

### 5. Updating the Environment

If new dependencies are added later, update your setup with:

```bash
conda env update -f environment.yml --prune
```

---

### 6. Exporting a New Version

After installing new packages, you can update the environment file with:

```bash
conda env export | grep -v 'prefix' > environment.yml
```

Commit the updated `environment.yml` to Git so others can stay in sync.

---

**Environment name:** `ae105`
**Python version:** 3.13
**Main tools:** NumPy, Pandas, Matplotlib, SpiceyPy
**Channels:** conda-forge, defaults


