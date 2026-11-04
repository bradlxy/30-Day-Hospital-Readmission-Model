# 30-Day Hospital Readmission Prediction

A pipeline for predicting 30-day hospital readmission risk using ICU/hospital admissions data, a Charlson Comorbidity Index feature set, and a comparison of logistic regression, random forest, and XGBoost models.

## Project Overview

This project extracts a patient cohort, builds a comorbidity-based feature matrix, and trains/compares multiple classification models to predict whether a patient will be readmitted within 30 days of discharge.

**Current data source:** MIMIC-IV Clinical Database Demo (~100 patients, publicly available, no credentialing required). This is used to validate the pipeline end-to-end; results on the demo data are **not clinically meaningful** given the small sample size, and should not be interpreted as a real assessment of predictive performance.

Full MIMIC-IV access (via PhysioNet credentialing) or the Diabetes 130-US Hospitals dataset are potential next steps for training on a larger, more representative sample.

## Pipeline

| Step | Script | Description | Output |
| --- | --- | --- | --- |
| 1 | `01_extract_data.py` | Builds a local SQLite DB from the MIMIC-IV demo CSVs and runs the cohort extraction query (admissions, patients, ICU stays, diagnoses; labels 30-day readmission via `LEAD()` window function) | `mimic_raw_icd10_cohort.csv` |
| 2 | `02_build_cci_matrix.py` | Maps ICD-10 codes to the 17 Charlson comorbidity flags + aggregate score using `comorbidipy` (Quan mapping) | `mimiciv_readmission_ready.csv` |
| 3 | `03_train_models.py` | Trains Logistic Regression, Random Forest, and XGBoost on the comorbidity feature matrix; exports metrics, feature importance, and test-set predictions for downstream analysis | `model_comparison_summary.csv`, `feature_importance.csv`, `logistic_regression_coefficients.csv`, `test_predictions.csv` |

Downstream plotting and visual analysis (ROC curves, feature importance charts, confusion matrices) are done in R using the exported CSVs from step 3.

## Setup

```bash
# clone and enter the repo
git clone <repo-url>
cd 30day_hospital

# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

### Data

Download the MIMIC-IV demo dataset and place it in a `mimic_data/` folder in the project root:

```bash
mkdir mimic_data
# download from https://physionet.org/content/mimic-iv-demo/2.2/
# unzip into mimic_data/, so the structure looks like:
#   mimic_data/mimic-iv-clinical-database-demo-2.2/hosp/...
#   mimic_data/mimic-iv-clinical-database-demo-2.2/icu/...
```

## Running the pipeline

```bash
python 01_extract_data.py
python 02_build_cci_matrix.py
python 03_train_models.py
```

Each script reads the previous step's output CSV, so they must be run in order on first use. Re-running `01_extract_data.py` will reuse the existing local SQLite database (`mimic_demo.db`) unless it's deleted first.

## Notes / Caveats

- The demo dataset's small size (~100 patients) means very few positive readmission cases survive the cohort filters — model metrics here validate that the pipeline runs correctly, not real-world predictive performance.

## Repo structure

```
.
├── 01_extract_data.py
├── 02_build_cci_matrix.py
├── 03_train_models.py
├── requirements.txt
├── README.md
└── mimic_data/          # not tracked in git - see .gitignore
```
