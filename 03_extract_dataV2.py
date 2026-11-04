import os
import glob
import sqlite3
import pandas as pd
 
# ============================================================
# 1. CONFIG - adjust this path once you confirm your folder layout
# ============================================================
MIMIC_DATA_DIR = "mimic_data"   # <-- UPDATE to the folder containing hosp/ and icu/ subfolders
DB_PATH = "mimic_demo.db"
 
# ============================================================
# 2. Build local SQLite DB from the demo CSV.gz files
# ============================================================
# Map: table_name -> relative path within MIMIC_DATA_DIR
TABLES_TO_LOAD = {
    "admissions":     "hosp/admissions.csv.gz",
    "patients":       "hosp/patients.csv.gz",
    "diagnoses_icd":  "hosp/diagnoses_icd.csv.gz",
    "icustays":       "icu/icustays.csv.gz",
}       
 
def build_database():
    conn = sqlite3.connect(DB_PATH)
    for table_name, rel_path in TABLES_TO_LOAD.items():
        full_path = os.path.join(MIMIC_DATA_DIR, rel_path)
        if not os.path.exists(full_path):
            # try to find it anywhere under MIMIC_DATA_DIR in case folder nesting differs
            matches = glob.glob(os.path.join(MIMIC_DATA_DIR, "**", os.path.basename(rel_path)), recursive=True)
            if matches:
                full_path = matches[0]
            else:
                raise FileNotFoundError(f"Could not find {rel_path} under {MIMIC_DATA_DIR}. "
                                         f"Run `find {MIMIC_DATA_DIR}` to check the real folder layout.")
        print(f"Loading {table_name} from {full_path} ...")
        df = pd.read_csv(full_path, compression="gzip", low_memory=False)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"  -> loaded {len(df)} rows into '{table_name}'")
    conn.commit()
    conn.close()
    print(f"Local SQLite database built at: {DB_PATH}")
 
# ============================================================
# 3. SQL adjusted for SQLite syntax
#    - DATETIME_DIFF(...,DAY)      -> julianday(a) - julianday(b)
#    - STRING_AGG(DISTINCT x, sep) -> GROUP_CONCAT(DISTINCT x)  (SQLite only supports default ',' separator with DISTINCT)
#    - LEAD() OVER (...)           -> same syntax, supported natively in SQLite 3.25+
# ============================================================
cohort_sql = """
WITH PrimaryCohort AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        p.anchor_age AS age,
        p.gender,
        CAST(julianday(a.dischtime) - julianday(a.admittime) AS INTEGER) AS length_of_stay_days,
        CASE WHEN icu.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS is_icu_patient,
        COALESCE(icu.first_careunit, 'Emergency Department') AS care_unit_name,
        LEAD(a.admittime) OVER (
            PARTITION BY a.subject_id
            ORDER BY a.admittime
        ) AS next_admittime
    FROM admissions a
    JOIN patients p
        ON a.subject_id = p.subject_id
    LEFT JOIN (
        SELECT hadm_id, GROUP_CONCAT(DISTINCT first_careunit) AS first_careunit
        FROM icustays
        GROUP BY hadm_id
    ) icu ON a.hadm_id = icu.hadm_id
    WHERE a.hospital_expire_flag = 0
      AND p.anchor_age >= 18
      AND (a.admission_type IN ('EMERGENCY', 'URGENT', 'EW EMER.') OR icu.hadm_id IS NOT NULL)
),
LabeledCohort AS (
    SELECT
        subject_id,
        hadm_id,
        age,
        gender,
        care_unit_name,
        is_icu_patient,
        length_of_stay_days,
        CASE
            WHEN CAST(julianday(next_admittime) - julianday(dischtime) AS INTEGER) BETWEEN 0 AND 30 THEN 1
            ELSE 0
        END AS target_30day_readmit
    FROM PrimaryCohort
)
SELECT
    c.*,
    d.icd_code,
    d.icd_version
FROM LabeledCohort c
JOIN diagnoses_icd d
  ON c.hadm_id = d.hadm_id
WHERE d.icd_version = 10;
"""
 
# ============================================================
# 4. Run the query and save results
# ============================================================
def run_extraction():
    conn = sqlite3.connect(DB_PATH)
    print("Running extraction query against local SQLite database...")
    raw_df = pd.read_sql_query(cohort_sql, conn)
    conn.close()
    print(f"Extracted {len(raw_df)} total diagnosis rows.")
    raw_df.to_csv("mimic_raw_icd10_cohort.csv", index=False)
    print("Saved extracted dataset to: mimic_raw_icd10_cohort.csv")
 
if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        build_database()
    else:
        print(f"{DB_PATH} already exists, skipping rebuild. Delete it to reload from CSVs.")
    run_extraction()