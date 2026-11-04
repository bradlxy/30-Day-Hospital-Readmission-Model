import pandas as pd
import polars as pl
from comorbidipy import comorbidity

print("Loading raw extracted cohort from disk...")
raw_df = pd.read_csv("mimic_raw_icd10_cohort.csv")

print("Processing ICD-10 codes into 17 Charlson binary flags...")
pl_df = pl.from_pandas(raw_df[['hadm_id', 'icd_code', 'age']])

# Run Quan ICD-10 mapping with hierarchy overrides (assign0=True)
cci_results = comorbidity(
    pl_df,
    id_col="hadm_id",
    code_col="icd_code",
    age_col="age",
    assign0=True
)
cci_pandas = cci_results.to_pandas()
if "score" in cci_pandas.columns:
    cci_pandas.rename(columns={"score": "cci_score"}, inplace=True)

# Merge metadata back with binary disease matrix
metadata_df = raw_df[[
    'subject_id', 'hadm_id', 'age', 'gender', 
    'care_unit_name', 'is_icu_patient', 'length_of_stay_days', 
    'target_30day_readmit'
]].drop_duplicates(subset=['hadm_id'])

final_df = pd.merge(metadata_df, cci_pandas, on='hadm_id', how='inner')

output_file = "mimiciv_readmission_ready.csv"
final_df.to_csv(output_file, index=False)

print("\n--- MATRIX GENERATION COMPLETE ---")
print(f"Final model dataset shape: {final_df.shape}")
print(f"File saved to: {output_file}")