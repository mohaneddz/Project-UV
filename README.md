<h1 style="font-family: Arial, sans-serif; font-size: 36px; color: #3C8D6B; border-bottom: 3px solid #3C8D6B; padding-bottom: 8px;">
  Project UV - UV Index Data Mining and Health Impact Study
</h1>

Project UV is a data-mining project focused on UV index behavior and its effects on health and environment.
The repository combines exploratory notebooks, modeling scripts, and a lightweight web layer for prediction workflows.

---

## Core Scope

- UV trend analysis across Algeria and multiple countries
- Health correlation studies (melanoma, sunburn, risk patterns)
- Race/country/environment comparisons and temporal analyses
- ML-based risk classification and assessment experiments

---

## Main Assets

- Notebooks: `*.ipynb` (EDA, risk, cancer, sunburn, ocean/environment)
- Data folders: `data/` (country, race, health, environmental sources)
- Python utilities: `utils/`, `main.py`, `update_notebook.py`
- Outputs: `output/` charts and generated analysis artifacts
- Web module: `web/frontend` + `web/backend`

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

For notebook-driven analysis, run the relevant notebook files directly in Jupyter.

---

## Notes

- Some folders include large experimental artifacts and backups (`backup/`, `tmp/`).
- Prefer using curated notebooks (`eda.ipynb`, `risk_classification.ipynb`, `cancer_assessment.ipynb`) as entry points.