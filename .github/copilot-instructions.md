# Copilot Instructions for AE-105

## Project Overview
- This is an academic codebase for AE-105 assignments, organized by homework in `src/HW_1/`, `src/HW_2/`, and `src/HW_3/`.
- Each homework folder contains Python scripts and, for HW3, Jupyter notebooks and relevant CSV data files.
- The environment is managed via Conda (`environment.yml`).

## Environment Setup
- Always use the `ae105` Conda environment. See `README.md` for setup:
  - `conda env create -f environment.yml`
  - `conda activate ae105`
- Update dependencies with `conda env update -f environment.yml --prune`.
- Export new environment versions with `conda env export | grep -v 'prefix' > environment.yml`.

## Key Patterns & Conventions
- Scripts are organized by homework and problem number (e.g., `HW1_P2.py`, `HW2_P3_1a.py`).
- Data files (CSV) are placed alongside scripts that use them.
- Jupyter notebooks are used for HW3; code and analysis are often split across multiple notebooks.
- Use NumPy, Pandas, Matplotlib, and SpiceyPy for computation and plotting.
- No custom build system; run scripts directly with Python or execute notebooks interactively.

## Developer Workflows
- **Testing:** Some folders have `test.py` or `test.ipynb` for validation. Run these to check code correctness.
- **Debugging:** Use print statements or notebook cells for stepwise inspection.
- **Data:** CSV files are loaded with Pandas; ensure file paths are correct relative to script location.
- **Version Control:** Commit changes to `environment.yml` when updating dependencies.

## Integration Points
- No external APIs or services; all dependencies are Python packages listed in `environment.yml`.
- SpiceyPy is used for space science calculations (see scripts in HW2/HW3).

## Examples
- To run HW1 problem 2: `python src/HW_1/HW1_P2.py`
- To test HW2: `python src/HW_2/test.py`
- To open HW3 notebook: use Jupyter or VS Code's notebook editor on `src/HW_3/HW3_P3.ipynb`

## References
- See `README.md` for full environment setup and update instructions.
- Key files: `environment.yml`, `src/HW_1/`, `src/HW_2/`, `src/HW_3/`

---

**For AI agents:**
- Always activate the correct Conda environment before running code.
- Respect the folder structure and naming conventions for new files.
- Prefer Pandas for data loading, Matplotlib for plotting, and SpiceyPy for space-related calculations.
- When adding dependencies, update `environment.yml` and document changes in `README.md`.
