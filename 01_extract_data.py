import os 
import time
import pandas as pd 
import numpy as np
from google.cloud import bigquery

# 1. Setting up the connection to Google BigQuery using the service account key file
    # Pointing to the service account key file

print(os.path.exists("gcp_key.json"), os.path.getsize("gcp_key.json") if os.path.exists("gcp_key.json") else "missing")
    
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"
GCP_project_id = "marine-tractor-507200-d1"

client = bigquery.Client(project=GCP_project_id)
print("Connection to Google BigQuery established successfully.")

# 2. Defining the SQL query to retrieve 30-day hospital readmission data
cohort_sql = """
WITH PrimaryCohort AS (
    SELECT 
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        p.anchor_age AS age,
        p.gender,
        DATETIME_DIFF(a.dischtime, a.admittime, DAY) AS length_of_stay_days,
        CASE WHEN icu.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS is_icu_patient,
        COALESCE(icu.first_careunit, 'Emergency Department') AS care_unit_name,
        LEAD(a.admittime) OVER (
            PARTITION BY a.subject_id 
            ORDER BY a.admittime
        ) AS next_admittime
    FROM `physionet-data.mimiciv_demo.admissions` a
    JOIN `physionet-data.mimiciv_demo.patients` p 
        ON a.subject_id = p.subject_id
    LEFT JOIN (
        SELECT hadm_id, STRING_AGG(DISTINCT careunit, ', ') AS first_careunit 
        FROM `physionet-data.mimiciv_demo.icustays` 
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
            WHEN DATETIME_DIFF(next_admittime, dischtime, DAY) BETWEEN 0 AND 30 THEN 1 
            ELSE 0 
        END AS target_30day_readmit
    FROM PrimaryCohort
)
SELECT 
    c.*,
    d.icd_code,
    d.icd_version
FROM LabeledCohort c
JOIN `physionet-data.mimiciv_demo.diagnoses_icd` d
  ON c.hadm_id = d.hadm_id
WHERE d.icd_version = 10;
"""

# 3. Submit and monitor query job
start_time = time.time()
print("Submitting MIMIC-IV extraction query to Google Cloud...")

query_job = client.query(cohort_sql)

# Poll while query runs
while query_job.state == "RUNNING":
    print("  --> BigQuery is processing query... (waiting 3s)")
    time.sleep(3)
    query_job.reload()

print(f"Query completed in {round(time.time() - start_time, 2)} seconds.")
print("Downloading raw cohort rows to local DataFrame...")

# 4. Convert query results to Pandas and save locally
raw_df = query_job.to_dataframe()
print(f"Extracted {len(raw_df)} total diagnosis rows.")

raw_df.to_csv("mimic_raw_icd10_cohort.csv", index=False)
print("Saved extracted dataset to: mimic_raw_icd10_cohort.csv")