# MIMIC-IV Data Access Guide

## Overview

MIMIC-IV v3.1 is the latest version, containing de-identified health records from ~300,000 patients at Beth Israel Deaconess Medical Center (2008-2019).

## Requirements

1. **CITI Training**: Complete "Data or Specimens Only Research" training (~2-4 hours)
   - https://physionet.org/about/citi-course/
   - Must be renewed every 3 years

2. **PhysioNet Account**: Create at https://physionet.org/settings/profile/

3. **Credentialing**: Submit credentialing application via PhysioNet
   - Upload CITI completion certificate
   - Wait for approval (typically 2-5 business days)

4. **Data Use Agreement**: Sign PhysioNet Credentialed Health Data License 1.5.0
   - Cannot share access with others
   - Cannot attempt to re-identify patients
   - Must report any potential PHI
   - Code from publications must be open-sourced

## Access Steps

### Step 1: CITI Training
- Go to https://physionet.org/about/citi-course/
- Complete "Data or Specimens Only Research" module
- Download completion certificate (PDF)

### Step 2: PhysioNet Credentialing
- Go to https://physionet.org/settings/profile/
- Complete credentialing form
- Upload CITI certificate
- Wait for approval email

### Step 3: MIMIC-IV Access
- Go to https://physionet.org/content/mimiciv/
- Click "Request access"
- Sign DUA electronically
- Choose access method:
  - **BigQuery** (recommended): Query data directly in Google Cloud
  - **Google Cloud Storage**: Download CSV files
  - **Local download**: Download all files (requires ~15 GB storage)

### Step 4: Set Up Cloud Access
- For BigQuery: Use `physionet.org/content/mimiciv` → "Request access using Google BigQuery"
- For GCS: Use "Request access to files using Google Cloud Storage Browser"
- Free tier available for BigQuery ($1TB/month free queries)

## Timeline

| Step | Time Required |
|------|--------------|
| CITI Training | 2-4 hours |
| PhysioNet Credentialing | 2-5 business days |
| MIMIC-IV Access | Instant after credentialing |
| **Total** | **~1 week** |

## What You Get

- ~300,000 patients
- ~500,000 hospital admissions
- Clinical data: demographics, diagnoses, lab results, medications, procedures
- ICU data: vitals, ventilator settings, fluid balance
- Chart notes: clinical text (requires separate access)
- ~50 GB total

## Integration with BioDigitalTwin

Once you have access:

```python
# Example: Load MIMIC-IV data
from app.personalization.mimic_loader import MIMICLoader

loader = MIMICLoader(project_id="your-gcp-project")
patients = loader.load_cohort(
    n_patients=100,
    cohort_type="mixed",
    include_icu=True
)

# Run validation pipeline
from app.personalization.mimic_validation import MIMICValidationPipeline

pipeline = MIMICValidationPipeline(seed=42)
results = pipeline.run_and_save(
    patients=patients,  # Use real patients instead of synthetic
    output_dir="mimic_results"
)
```

## Important Notes

- **No AWS support yet**: MIMIC-IV only available on GCP (BigQuery/GCS)
- **BigQuery is cheapest**: Pay-per-query, first 1TB/month free
- **Each user needs their own account**: Cannot share credentials
- **Data is de-identified**: Safe for research, but still requires DUA
- **Code from publications must be open-sourced**: Per DUA requirement
