# Transforming the Healthcare Revenue Cycle with AI
## Claim Denial Prediction System — Complete Project Documentation

---

## Table of Contents

1. [Project Scope](#1-project-scope)
2. [Problem Statement](#2-problem-statement)
3. [Solution Overview](#3-solution-overview)
4. [What Was Built](#4-what-was-built)
5. [How It Works — End to End](#5-how-it-works--end-to-end)
6. [Data Sources & About the Data](#6-data-sources--about-the-data)
7. [Data Engineering](#7-data-engineering)
8. [Data Architecture](#8-data-architecture)
9. [Feature Engineering](#9-feature-engineering)
10. [NLP & Clinical Language Processing](#10-nlp--clinical-language-processing)
11. [Machine Learning & Model Training](#11-machine-learning--model-training)
12. [AI Components & Intelligent Automation](#12-ai-components--intelligent-automation)
13. [Batch Scoring Pipeline](#13-batch-scoring-pipeline)
14. [Post-Deployment Monitoring](#14-post-deployment-monitoring)
15. [Analytics Dashboard](#15-analytics-dashboard)
16. [Project Structure](#16-project-structure)
17. [Key Technical Decisions](#17-key-technical-decisions)
18. [Testing Strategy](#18-testing-strategy)
19. [Infrastructure & Scale](#19-infrastructure--scale)
20. [Compliance & Security](#20-compliance--security)

---

## 1. Project Scope

This project builds an **AI-powered Claim Denial Prediction System** as part of a Healthcare Revenue Cycle Management (RCM) platform. The system is designed for hospitals, health systems, and medical billing organizations that submit thousands of insurance claims daily.

**In scope:**

- Ingestion and parsing of X12 837 EDI claim files from insurance payers
- Linking claims to Electronic Health Record (EHR) data via FHIR R4 APIs
- Engineering a rich, 50+ column feature set across five clinical and administrative dimensions
- Extracting signals from free-text clinical notes using NLP and clinical BERT embeddings
- Training gradient-boosted ensemble ML models (XGBoost, LightGBM, CatBoost)
- Deploying a nightly batch scoring pipeline over all active claims
- Monitoring model performance in production and triggering automated retraining
- An interactive 8-page Streamlit analytics dashboard backed by DuckDB SQL

**Out of scope:**

- Real-time (sub-second) claim scoring at point of care
- Claim submission or clearinghouse connectivity
- Direct EHR write-back or clinical workflow integration

---

## 2. Problem Statement

Healthcare providers in the United States submit billions of insurance claims annually. A significant percentage — industry estimates range from **5% to 25%** — are denied by payers on first submission. Each denial triggers a costly appeal process: billing staff must identify the denial reason, gather supporting documentation, correct the claim, and resubmit. Many denials are never appealed at all, resulting in permanent revenue loss.

The core challenges are:

| Challenge | Impact |
|---|---|
| Denials discovered after submission | No time to correct before submission |
| High denial rate variability by payer | Billing staff cannot anticipate payer-specific behavior |
| Documentation gaps for medical necessity | Clinical notes don't adequately support the procedure |
| Prior authorization gaps | Procedures performed without required pre-authorization |
| Coding mismatches | Diagnosis codes inconsistent with procedure codes |
| Volume at scale | Thousands of claims per day make manual review impossible |

The business cost is severe: a single large health system can lose tens of millions of dollars annually to preventable claim denials.

---

## 3. Solution Overview

The solution predicts, **before submission**, the probability that any given claim will be denied by its payer. This gives billing staff a prioritized worklist of high-risk claims to review and correct before they are sent — converting reactive denial management into proactive prevention.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NIGHTLY PIPELINE (02:00 local)                  │
│                                                                         │
│  X12 837 EDI  ──► Claim_Processor ──►  Data Lake                       │
│  EHR (FHIR)   ──► EHR_Ingestion   ──►  (S3 / Azure Blob)               │
│                              │                                          │
│                              ▼                                          │
│                    Feature_Engineer  ◄──  Clinical BERT (NLP)           │
│                              │                                          │
│                              ▼                                          │
│                    Feature_Store  (HBase / Cassandra)                   │
│                              │                                          │
│                              ▼                                          │
│                    Batch_Scorer  ──►  predicted_denial_score [0,1]      │
│                              │                                          │
│                              ▼                                          │
│                    Billing Worklist: "Review these 47 high-risk claims" │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key promises of the system:**

- Score **all active claims** within a 4-hour nightly window
- Achieve **Precision ≥ 0.90** (when the model predicts denial, it is right ≥ 90% of the time)
- Maintain **Expected Calibration Error ≤ 0.05** (scores are well-calibrated probabilities)
- **Automatically retrain** when real-world precision drops below 0.80
- Retain a full **7-year audit trail** of raw EDI files per regulatory requirements

---

## 4. What Was Built

### Python Package: `claim_denial`

A fully structured, production-grade Python package with 20+ modules organized into sub-packages:

| Sub-package | What it contains |
|---|---|
| `ingestion/` | X12 837 EDI parser, FHIR EHR client, Data Lake writer |
| `features/` | Four feature engineering modules (claim, provider, clinical, payer) |
| `nlp/` | Clinical BERT NLP processor with embedding and keyword detection |
| `training/` | Dataset builder, model trainer (XGBoost/LightGBM/CatBoost + Optuna) |
| `scoring/` | Nightly batch scorer with Spark + local fallback |
| `monitoring/` | Rolling precision monitoring, alert dispatch, retraining trigger |
| `registry/` | MLflow-backed model registry with stage machine |
| `schemas/` | Avro schema definitions and version migration registry |
| `utils/` | Feature Store client, Avro serializer, synthetic data generator |
| `dags/` | Apache Airflow DAG for nightly orchestration |
| `pipeline/` | PySpark distributed pipeline wiring for all phases |

### Streamlit Dashboard: `dashboard/`

An 8-page interactive analytics dashboard that makes the system's outputs visible to billing operations teams, clinical analysts, and ML engineers.

### Specifications: `.kiro/specs/claim-denial-prediction/`

Formal requirements, design document, and implementation task checklist — 15 requirements with 90+ acceptance criteria.

---

## 5. How It Works — End to End

### Step 1: Claim Ingestion (02:00 — 03:00)

The Airflow DAG triggers at 02:00 local facility time. `ClaimProcessor` scans the landing zone for new X12 837 EDI files.

For each file:
1. **Structural validation** — checks ASC X12 005010 envelope (ISA/GS/ST/SE/GE/IEA segments, element delimiters, 16-element ISA)
2. **Parse** — extracts header (NPI, Payer ID, Patient Account), patient demographics (DOB, gender, member ID), service lines (CPT/HCPCS codes, charges, units), and diagnosis codes (principal ICD-10, up to N secondary ICD-10 codes)
3. **Normalize** — NPI trimmed to 10-digit numeric, ICD-10 codes upper-cased and dot-stripped, dates converted to ISO-8601, charges rounded to 2 decimal places
4. **Write to Data Lake** — raw EDI + 7-year retention metadata sidecar, parsed claim JSON
5. **Error handling** — structural failures quarantine the entire file; missing required fields skip only that claim record; Data Lake write failures retry 3× with 1s/2s/4s exponential backoff before quarantining

### Step 2: EHR Ingestion (02:30 — 03:00)

`EHRIngestionService` joins each claim to EHR data using the Patient Account Number as the join key.

1. Queries FHIR R4 API (injectable `FHIRClient`) for all encounters for the patient
2. **Encounter selection** — picks the encounter whose date most closely precedes the claim statement period start date (avoids matching future admissions)
3. Extracts structured fields: age, sex, race, ethnicity, vitals (BP, HR, temp, O2 sat), LOINC lab results, SNOMED-coded problem list, active medications, admitting department, physician ID, length of stay
4. Extracts clinical notes scoped to that encounter: History & Physical, Discharge Summary, Progress Notes
5. Sets `ehr_null_match_flag=1` if no EHR found; sets `partial_ehr_flag` with absent field names if some fields missing
6. Writes EHR JSON to Data Lake partitioned by `PatientAccountNumber/encounter_date`

### Step 3: Feature Engineering (03:00 — 04:30)

Four independent feature engineers compute their feature groups and upsert results into the Feature Store, which is a denormalized wide-column NoSQL table (HBase or Cassandra) keyed by Claim ID.

See [Section 9](#9-feature-engineering) for details on all 50+ features.

### Step 4: NLP Processing (03:00 — 06:00, 3-hour SLA)

`NLPProcessor` runs on all clinical notes extracted in Step 2:

1. Concatenates all note types in chronological order, truncates to 10,000 tokens
2. Encodes through a clinical BERT model (stub in development; real model in production) → 768-dim float32 embedding
3. Derives a scalar `medical_necessity` score from the embedding
4. Applies a regex/keyword classifier for prior authorization gaps
5. Writes all three NLP features to the Feature Store

### Step 5: Batch Scoring (04:30 — 06:00)

`BatchScorer` reads all claims with `claim_status = "Active"` from the Feature Store, loads the Production model from the Model Registry, and runs distributed inference via Spark (or a local fallback).

Each claim gets a `predicted_denial_score` ∈ [0.0, 1.0] written back to the Feature Store along with the model version ID and UTC scoring timestamp.

### Step 6: Monitoring (06:00 — 06:30)

`MonitoringService` computes rolling 30-day Precision against real payer adjudication outcomes (ERA/835 files) and writes metrics to the monitoring dashboard. Alerts are dispatched if precision degrades.

---

## 6. Data Sources & About the Data

### Primary Data Sources

#### X12 837 EDI Files (Claims)
The standard electronic format for healthcare claim submission in the United States. Files conform to ASC X12 005010X222A1 (professional) and 005010X223A2 (institutional) specifications.

Key data elements extracted:

| Field | Format | Notes |
|---|---|---|
| Provider NPI | 10-digit numeric | National Provider Identifier |
| Payer ID | Alphanumeric | Insurance company identifier |
| Patient Account Number | Alphanumeric | Join key to EHR |
| Date of Birth | ISO-8601 YYYY-MM-DD | For patient age computation |
| Gender | M / F / U | Single character, upper-case |
| Principal ICD-10 | Upper-case, dot-stripped | Primary diagnosis code |
| Secondary ICD-10 | List, same format | Comorbidity codes |
| CPT/HCPCS Codes | 5-character | Procedure codes |
| Charge Amount | Decimal (2 places) | Billed amount |
| Service Dates | ISO-8601 | Date services were rendered |
| Revenue Codes | 4-digit zero-padded | Institutional billing |

#### Electronic Health Records (EHR via FHIR R4)
Structured clinical data from the hospital's EHR system, retrieved through a FHIR R4 RESTful API:

- **Demographics**: age, sex, race, ethnicity
- **Vital signs**: BP, HR, temperature, O2 saturation
- **Lab results**: LOINC-coded (e.g., HbA1c, creatinine, CBC)
- **Problem list**: SNOMED-CT coded active/resolved diagnoses
- **Medications**: RxNorm-coded active prescriptions with dose and frequency
- **Clinical notes**: free-text H&P, discharge summaries, progress notes
- **Administrative**: admitting department, physician ID, length of stay

#### ERA/835 Files (Adjudication Outcomes)
Electronic Remittance Advice — the payer's response indicating whether each claim was paid, denied, or adjusted. These form the ground truth labels for model training and production monitoring.

### Synthetic Data

All development and testing uses **100% synthetic data** generated by `SyntheticDataGenerator` in `claim_denial/utils/synthetic_data.py`. The generator produces:

- Structurally valid X12 837 EDI files with randomized but spec-conformant data
- Realistic EHR encounter records with LOINC labs, SNOMED problems, and medication lists
- Clinical note text using structured templates (H&P, Discharge Summary, Progress Notes)
- ERA/835 adjudication records with configurable denial rates
- Labeled claim datasets for training with configurable class balance

No real PHI (patient names, dates of birth, member IDs, NPIs) appears anywhere in the codebase.

### Reference Tables

| Table | Used for |
|---|---|
| HCUP CCS single-level | Maps ~70,000 ICD-10 codes → ~285 clinical categories |
| HCUP Procedure categories | Maps CPT/HCPCS codes → 25 procedure group indices |
| Charlson Comorbidity mapping | Maps ICD-10 prefixes → integer comorbidity weights |
| Diagnosis–procedure mismatch | Maps CCS categories → expected procedure categories |
| NUCC taxonomy | Maps provider NPI → specialty code |
| Facility master | Maps provider NPI → bed size |
| Payer contract reference | Maps payer ID + date range → stop-loss clause flag |

---

## 7. Data Engineering

### Ingestion Architecture

Data flows from source systems into a **Data Lake** (AWS S3 or Azure Blob Storage) before entering the Feature Store. The Data Lake serves as the immutable source of record.

```
Landing Zone                Data Lake
─────────────               ─────────────────────────────────────────
X12 837 EDI  ───►  ClaimProcessor  ───►  claims/YYYY/MM/DD/raw/
                                         claims/YYYY/MM/DD/parsed/
                                         quarantine/YYYY/MM/DD/

EHR (FHIR)   ───►  EHRIngestionService  ───►  ehr/{pan}/{date}/

ERA/835      ───►  MonitoringService  ───►  era/YYYY/MM/DD/
```

### Data Lake Partition Layout

```
data_lake/
├── claims/
│   └── YYYY/MM/DD/
│       ├── raw/            ← original X12 837 EDI bytes
│       │   ├── claim.edi
│       │   └── claim.edi.meta.json   ← 7-year retention tag
│       ├── parsed/
│       │   └── claim.edi.json        ← normalized claim records
│       └── errors.jsonl              ← structured error log
├── ehr/
│   └── {patient_account_number}/
│       └── {encounter_date}/
│           └── ehr_record.json
├── era/
│   └── YYYY/MM/DD/
│       └── era_835.json
└── quarantine/
    └── YYYY/MM/DD/
        └── {file_name}               ← invalid or failed files
```

### Retention Policy

Every raw EDI file written to the Data Lake receives a `.meta.json` sidecar with a 7-year retention policy tag, satisfying HIPAA/CMS regulatory requirements:

```json
{
  "retention_policy": "regulatory",
  "retention_years": 7,
  "retention_rule": "HIPAA / CMS 7-year raw EDI retention",
  "file_name": "claim_20240115.edi",
  "ingestion_timestamp_utc": "2024-01-15T02:03:47.123456",
  "ingestion_date": "2024-01-15"
}
```

### Error Handling & Quarantine

The data engineering layer is designed to be fault-tolerant and never silently discard data:

| Error Type | Action |
|---|---|
| X12 structural validation failure | Quarantine entire file, log `{file_name, segment_id, violation_description}` |
| Missing required claim fields | Skip that claim, log `{claim_id, field_name}`, continue rest of file |
| Data Lake write failure | Retry ×3 with 1s/2s/4s backoff; quarantine + structured error log on exhaustion |
| Unsupported file format | Quarantine, log unsupported-format error |
| EHR record not found | Set `ehr_null_match_flag=1`; allow claim-only features |
| Partial EHR fields | Set `partial_ehr_flag="age,vitals"` (comma-separated absent fields) |
| Feature Store write failure | Retry ×3 with exponential backoff; log failure, continue next claim |

### Feature Store

The Feature Store is a **wide-column NoSQL store** (HBase or Cassandra) holding a single denormalized "Claim Point of View" table. Each row is keyed by `claim_id` and contains all features computed by every feature engineering module, plus scoring outputs and ground truth labels as they become available.

This design allows the batch scorer to read a single row per claim and immediately score it — no joins required.

The schema is versioned using **Avro** (`fastavro`) with embedded `schema_version` fields. The `SchemaRegistry` class handles chained migration rules when reading records with older schema versions.

### Apache Spark Integration

All heavy data transformation runs on Apache Spark for horizontal scalability:

```python
# SparkPipelineConfig — default production settings
SparkPipelineConfig(
    app_name="ClaimDenialPrediction",
    master="spark://host:7077",
    executor_cores=4,
    executor_memory="8g",
    num_executors=10,    # sustains ≥1M claims/year in 4-hour window
)
```

Each pipeline phase (ingestion, feature engineering, NLP, scoring) has a Spark path using `mapPartitions` for efficiency, with a transparent local sequential fallback when Spark is unavailable.

---

## 8. Data Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION LAYER                                │
│                   Apache Airflow DAG — 02:00 nightly                         │
│    log_container >> ingest_claims >> ingest_ehr >> engineer_features          │
│                  >> run_nlp >> score_claims >> run_monitoring                 │
└──────────────────────────────────────────────────────────────────────────────┘
         │               │               │               │
         ▼               ▼               ▼               ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐
│  Landing    │  │   EHR System │  │   ERA/835     │  │   Model Registry      │
│  Zone (EDI) │  │   (FHIR R4)  │  │  (Payer ACK)  │  │   (MLflow)            │
└──────┬──────┘  └──────┬───────┘  └──────┬────────┘  └──────────┬────────────┘
       │                │                 │                       │
       ▼                ▼                 ▼                       │
┌────────────────────────────────────────────────────────────┐   │
│                       DATA LAKE                            │   │
│              (AWS S3 / Azure Blob Storage)                 │   │
│   claims/raw  claims/parsed  ehr/  era/  quarantine/       │   │
└────────────────────────┬───────────────────────────────────┘   │
                         │                                        │
                         ▼                                        │
┌────────────────────────────────────────────────────────────┐   │
│                FEATURE ENGINEERING (Spark)                 │   │
│   ClaimLevel + Provider/Facility + Clinical + Payer +      │   │
│   NLP (clinical BERT) → 50+ feature columns               │   │
└────────────────────────┬───────────────────────────────────┘   │
                         │                                        │
                         ▼                                        │
┌────────────────────────────────────────────────────────────┐   │
│                    FEATURE STORE                           │   │
│              (HBase / Cassandra) — Avro v1.0.0             │   │
│   claim_id (row key) → all 50+ feature columns             │   │
└────────────────────────┬───────────────────────────────────┘   │
                         │                                        │
                         ▼                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          BATCH SCORER (Spark)                              │
│   Load Production model ──► score Active claims ──► write scores back     │
│   predicted_denial_score ∈ [0.0, 1.0]  +  model_version_id  +  timestamp  │
└────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        MONITORING SERVICE                                  │
│   ERA/835 linkage → rolling 30-day Precision → alerts → retraining trigger │
└────────────────────────────────────────────────────────────────────────────┘
```

### Feature Store Row Schema (Avro v1.0.0)

Every row in the Feature Store follows this versioned schema. Fields are grouped by the module that writes them:

```
ClaimFeatureRecord {
  // Core
  schema_version: string              // "1.0.0"
  claim_id: string                    // row key
  claim_status: string                // "Active" | "Submitted" | etc.

  // Claim-Level (ClaimLevelFeatureEngineer)
  claim_submission_date_doy: int      // 1–366
  claim_submission_date_month: int    // 1–12
  patient_age: int                    // whole years, -1 if unknown
  gender: int                         // 0=F, 1=M, -1=unknown
  ins_type_medicare: int              // one-hot binary
  ins_type_medicaid: int
  ins_type_commercial: int
  ins_type_tricare: int
  ins_type_champva: int
  ins_type_other: int
  claim_type_professional: int
  claim_type_institutional: int
  adm_type_emergency: int
  adm_type_elective: int
  adm_type_urgent: int
  adm_type_trauma: int
  adm_src_physician_referral: int
  adm_src_transfer_hospital: int
  adm_src_transfer_snf: int
  adm_src_er: int
  adm_src_court_law: int
  adm_src_not_available: int
  partial_ehr_flag: string|null
  ehr_null_match_flag: int

  // Provider/Facility (ProviderFacilityFeatureEngineer)
  billing_provider_id_encoded: float  // LOO target encoding or pop-mean
  performing_physician_specialty: string  // NUCC taxonomy, "UNKNOWN" sentinel
  provider_historical_denial_rate: float
  provider_claim_volume: int
  facility_bed_size: int

  // Clinical (ClinicalFeatureEngineer)
  principal_dx_ccs_category: int      // 0–285
  secondary_dx_count: int
  procedure_category_vector: [int]    // 25-element binary array
  length_of_stay: int                 // ≥0, -1 discharge absent, -2 both absent
  comorbidity_risk_score: int         // Charlson Comorbidity Index
  diagnosis_procedure_mismatch: int   // binary 0/1

  // Payer Behavior (PayerBehaviorFeatureEngineer)
  payer_historical_denial_rate: float
  payer_historical_denial_rate_by_type: float
  days_since_last_payment: int        // -1 for cold-start
  payer_contract_stop_loss: int

  // NLP (NLPProcessor)
  documents_medical_necessity: float  // [0.0, 1.0], 4 decimal places
  mentions_lack_of_pre_auth: int      // 0 or 1
  tx_plan_complexity_embedding: [float]  // 768-dim float32
  nlp_notes_absent_flag: int          // 0=ok, 1=absent, 2=error
  nlp_model_name: string|null
  nlp_model_version: string|null

  // Scoring output (BatchScorer)
  predicted_denial_score: float|null  // [0.0, 1.0]
  score_quality_flag: string|null     // "degraded" or null
  model_version_id: string|null
  scoring_timestamp_utc: string|null

  // Ground truth (MonitoringService — post adjudication)
  adjudication_outcome: string|null   // "paid" | "denied"
  adjudication_timestamp_utc: string|null
}
```

### Model Registry Schema

Each model version in MLflow carries:

| Field | Type | Description |
|---|---|---|
| `version_id` | string | Unique identifier (e.g., `v-a1b2c3d4`) |
| `stage` | string | Staging / Production / Archived / Failed |
| `training_date` | ISO-8601 | When training completed |
| `algorithm` | string | XGBoost / CatBoost / LightGBM |
| `algorithm_version` | string | Library version |
| `hyperparameters` | JSON | Full Optuna best-trial dict |
| `precision` | float | Held-out test set precision |
| `recall` | float | Held-out test set recall |
| `f1` | float | Held-out test set F1 |
| `auc_roc` | float | AUC-ROC on test set |
| `auc_pr` | float | AUC-PR on test set |
| `ece` | float | Expected Calibration Error (10-bin) |
| `shap_top20` | JSON array | Top-20 features by mean \|SHAP\| |
| `train_record_count` | int | Records in training split |
| `denied_class_pct` | float | Denied class % in training |
| `reviewer_id` | string | Human reviewer (gate actions) |
| `review_timestamp` | ISO-8601 | When reviewed |
| `rejection_reason` | string | Populated on rejection |
| `trigger_reason` | string | e.g., `precision_threshold_breach` |

---

## 9. Feature Engineering

Feature engineering is the core intellectual work of this project. Five independent feature groups are computed and written to the Feature Store. All windows are **trailing 90 calendar days** anchored to the claim submission date.

### 9.1 Claim-Level Features

**Module**: `claim_denial/features/claim_level.py`  
**Class**: `ClaimLevelFeatureEngineer`

These features capture the administrative characteristics of the claim itself.

| Feature | Type | Logic |
|---|---|---|
| `claim_submission_date_doy` | int (1–366) | Day of year from statement period start |
| `claim_submission_date_month` | int (1–12) | Month from statement period start |
| `patient_age` | int (≥0) | Whole years from DOB to statement start; -1 if DOB absent |
| `gender` | int | 0=female, 1=male, -1=unknown/null/absent |
| `ins_type_*` | 6 binary ints | One-hot: Medicare, Medicaid, Commercial, TriCare, ChampVA, Other |
| `claim_type_*` | 2 binary ints | One-hot: Professional, Institutional |
| `adm_type_*` | 4 binary ints | One-hot: Emergency, Elective, Urgent, Trauma |
| `adm_src_*` | 6 binary ints | One-hot: Physician Referral, Transfer, SNF, ER, Court/Law, N/A |

All one-hot vectors sum to exactly 0 (unrecognized) or 1 (valid). Sentinel -1 for unknown demographics doesn't block downstream computation.

### 9.2 Provider & Facility Features

**Module**: `claim_denial/features/provider_facility.py`  
**Class**: `ProviderFacilityFeatureEngineer`

These features capture historical denial behavior of the billing provider and facility characteristics.

| Feature | Type | Logic |
|---|---|---|
| `billing_provider_id_encoded` | float [0,1] | **Leave-one-out target encoding** when provider has >30 known-outcome claims in trailing 90 days; **population-mean denial rate** fallback when ≤30 |
| `performing_physician_specialty` | string | NUCC Health Care Provider Taxonomy code from NPI registry; "UNKNOWN" sentinel |
| `provider_historical_denial_rate` | float [0,1] | Fraction denied over trailing 90 days (known-outcome claims only); population-mean fallback when <30 claims |
| `provider_claim_volume` | int | All claims submitted in trailing 90 days (regardless of adjudication status) |
| `facility_bed_size` | int | From facility master by NPI; integer mean of all facilities when NPI absent |

**Leave-one-out encoding** avoids target leakage — the current claim is excluded from the computation, so the model learns true historical rates rather than the claim's own label.

### 9.3 Clinical Features

**Module**: `claim_denial/features/clinical.py`  
**Class**: `ClinicalFeatureEngineer`

These features capture the clinical complexity and appropriateness of the claim.

| Feature | Type | Logic |
|---|---|---|
| `principal_dx_ccs_category` | int (0–285) | HCUP CCS single-level category mapped from principal ICD-10; 0 = unknown |
| `secondary_dx_count` | int (≥0) | Count of secondary diagnosis codes on the claim |
| `procedure_category_vector` | int[25] | Multi-hot binary vector over 25 HCUP procedure categories; unrecognized CPT codes silently ignored |
| `length_of_stay` | int | Whole days (discharge − admission); -1 if discharge absent; -2 if both dates absent |
| `comorbidity_risk_score` | int (≥0) | **Charlson Comorbidity Index** computed from secondary ICD-10 codes using hierarchical prefix matching (Quan et al. 2005 weights) |
| `diagnosis_procedure_mismatch` | int (0/1) | 1 if principal CCS category has no expected procedure category present in the vector; 0 otherwise |

**Charlson Comorbidity Index** assigns integer weights to comorbid conditions:
- Weight 1: MI, CHF, PVD, cerebrovascular disease, dementia, COPD, peptic ulcer, mild liver disease, DM
- Weight 2: hemiplegia, renal disease, DM with end-organ damage, malignancy
- Weight 3: moderate/severe liver disease
- Weight 6: metastatic cancer, AIDS/HIV

### 9.4 Payer Behavior Features

**Module**: `claim_denial/features/payer_behavior.py`  
**Class**: `PayerBehaviorFeatureEngineer`

These features capture whether a specific payer tends to deny claims, and any contract-level risk factors.

| Feature | Type | Logic |
|---|---|---|
| `payer_historical_denial_rate` | float [0,1] | Fraction of known-outcome claims denied by this payer in trailing 90 days; population-mean fallback when <50 claims |
| `payer_historical_denial_rate_by_type` | float [0,1] | Same but segmented by Professional vs Institutional claim type; population-mean fallback |
| `days_since_last_payment` | int | Calendar days since most recent paid ERA/835 for this payer; -1 for cold-start (no prior paid ERA) |
| `payer_contract_stop_loss` | int (0/1) | Binary: stop-loss clause is active on the contract effective date; sourced from payer contract reference table |

A high `days_since_last_payment` can signal a payer in dispute or experiencing processing issues, which correlates with denial risk.

### 9.5 Lookback Windows & Population-Mean Fallback

All historical rate features use a **trailing 90-day window**. When a provider or payer has insufficient claim history (below the minimum threshold), the system falls back to the **population-mean denial rate** — the mean rate across all entities with at least one known-outcome claim in the same window.

This prevents cold-start problems for new providers/payers while keeping feature values in a valid [0.0, 1.0] range. The thresholds are:
- Providers: < 30 known-outcome claims → population-mean fallback
- Payers: < 50 known-outcome claims → population-mean fallback

---

## 10. NLP & Clinical Language Processing

**Module**: `claim_denial/nlp/nlp_processor.py`  
**Class**: `NLPProcessor`

Clinical notes are a gold mine of denial-risk signals that structured data misses. A patient's history might show a procedure was medically necessary, or a note might reveal that prior authorization was never obtained. The NLP pipeline surfaces these signals automatically.

### Pipeline

```
clinical_notes (H&P + Discharge Summary + Progress Notes)
        │
        ▼
1. prepare_notes()
   - Sort notes chronologically by note_datetime
   - Concatenate: note_1_text + " " + note_2_text + ...
   - Tokenize on whitespace
   - Truncate to 10,000 tokens (avoids BERT context limit issues)
        │
        ▼
2. compute_bert_embedding()
   - Encode truncated text through clinical BERT model
   - Returns float32 numpy array of shape (768,)
   - Logs model_name and model_version alongside each output
        │
        ├──► 3. compute_medical_necessity_score()
        │       - Derives scalar from embedding: mean(embedding), clamped to [0,1]
        │       - Rounded to 4 decimal places
        │       - Higher = stronger documentation of medical necessity
        │
        └──► 4. detect_prior_auth_gap()
                - Applies regex/keyword classifier to note text (case-insensitive)
                - Returns 1 if any of these patterns found:
                  "prior authorization", "pre-authorization", "pre authorization",
                  "auth gap", "prior auth", "preauth",
                  "authorization required", "not authorized"
                - Returns 0 otherwise
```

### Output Features

| Feature | Description |
|---|---|
| `documents_medical_necessity` | float ∈ [0.0, 1.0], 4 decimal places. Higher = stronger medical necessity documentation |
| `mentions_lack_of_pre_auth` | binary 0/1. 1 = prior authorization gap detected in notes |
| `tx_plan_complexity_embedding` | float32[768]. Full BERT embedding for downstream use |

### Default Handling

When notes are unavailable or the model fails:

| Condition | `medical_necessity` | `pre_auth` | `embedding` | `flag` |
|---|---|---|---|---|
| No clinical notes | 0.5 (neutral) | 0 | zeros(768) | 1 |
| NLP inference error | 0.5 (neutral) | 0 | zeros(768) | 2 |

The inference error path logs the claim ID and failure reason, then continues processing remaining claims without interruption.

### Clinical BERT Model

In development, a `MockClinicalBERT` stub is used. It produces **deterministic, realistic-looking** 768-dim float32 embeddings by seeding a NumPy `RandomState` with an MD5 hash of the input text. This means:
- Tests are reproducible
- The stub behaves like a real model (same interface, same shapes)
- Swapping in the real `transformers` clinical BERT requires only replacing the `_model` attribute

Production models would use models such as `emilyalsentzer/Bio_ClinicalBERT` or `allenai/biomed_roberta_base` from HuggingFace Transformers.

### SLA Compliance

NLP must complete all claims within **3 hours** of pipeline start, leaving 1 hour of headroom within the 4-hour batch window. For scale, PySpark `mapPartitions` is used to parallelize note processing across the cluster.

---

## 11. Machine Learning & Model Training

**Modules**: `claim_denial/training/dataset_builder.py`, `claim_denial/training/model_trainer.py`

### Training Dataset Construction

**Class**: `TrainingDatasetBuilder`

```python
dataset = builder.build(
    construction_date=date.today(),
    feature_store_records=records,
)
```

1. **90-day window**: pulls historical claims from the Feature Store within 90 days
2. **Window expansion**: if fewer than 1,000 labeled records available in 90 days, expands to 180 days
3. **Labeling**: `adjudication_outcome == "denied"` → label 1; `"paid"` → label 0; all other statuses excluded and count logged to MLflow
4. **Halt condition**: if labeled records after exclusions < 1,000, halts and raises `InsufficientDataError` — no partial model is trained

**Temporal train/test split (80/20)**:
- Test set = most recent 20% sorted by submission date
- All claims on the boundary date go to the test set (prevents data leakage)
- Cutoff date, total count, denied count, paid count, denied% logged to MLflow

**Class imbalance handling**: if denied class < 5% of training records, applies class-weight balancing (`scale_pos_weight` for XGBoost, `class_weight` for LightGBM/CatBoost) — never under/over-samples, preserving training data volume.

### Model Training

**Class**: `ModelTrainer`

```python
result = trainer.run_training(
    training_dataset=dataset,
    trigger_reason="precision_threshold_breach",  # optional
)
```

**Step-by-step**:

1. **Temporal 70/15/15 split** on submission date (train/validation/test)
2. **Convert to feature matrix**: scalar numeric fields from ClaimFeatureRecord extracted into float32 numpy arrays
3. **Bayesian hyperparameter optimization** via Optuna (minimum 50 trials), maximizing Precision on the 15% validation split
4. **Final training** with best hyperparameters on the 70% train split
5. **Evaluation** on held-out 15% test set: Precision, Recall, F1, AUC-ROC, AUC-PR, ECE (10-bin)
6. **SHAP feature importance**: top-20 features by mean |SHAP| logged to MLflow
7. **Registration**: Staging if Precision ≥ 0.90 AND ECE ≤ 0.05; Failed otherwise

### Supported Algorithms

| Algorithm | Library | Distributed via |
|---|---|---|
| XGBoost | `xgboost==2.0.3` | `SparkXGBClassifier` / sklearn API fallback |
| LightGBM | `lightgbm==4.3.0` | sklearn API (multi-threaded) |
| CatBoost | `catboost==1.2.5` | sklearn API |

XGBoost hyperparameter search space:
- `n_estimators`: 100–1000
- `max_depth`: 3–10
- `learning_rate`: 0.001–0.3 (log scale)
- `subsample`: 0.5–1.0
- `colsample_bytree`: 0.5–1.0
- `min_child_weight`: 1–10
- `reg_alpha`, `reg_lambda`: 1e-8 to 10.0 (log scale)

### Evaluation Metrics

| Metric | Target | Description |
|---|---|---|
| Precision | ≥ 0.90 | Of predicted denials, fraction that are true denials. **Primary metric.** |
| Recall | tracked | Of true denials, fraction correctly predicted |
| F1 Score | tracked | Harmonic mean of precision and recall |
| AUC-ROC | tracked | Area under the ROC curve |
| AUC-PR | tracked | Area under the precision-recall curve |
| ECE | ≤ 0.05 | Expected Calibration Error (10-bin): how well scores reflect true probabilities |

**Precision** is the primary metric because false positives (flagging a claim that would have been paid) create unnecessary work for billing staff and erode trust in the system.

**ECE** is required to be ≤ 0.05 because the denial score is used as a probability — users need to trust that a score of 0.8 means the claim is ~80% likely to be denied, not 95%.

### Model Stage Machine

```
Training outcome
      │
      ├── Precision ≥ 0.90 AND ECE ≤ 0.05  ──►  [Staging]
      │                                              │
      │                                   Human review
      │                                              │
      │                              ┌───────────────┴───────────┐
      │                         Approve                        Reject
      │                              │                            │
      │                        [Production]  ──►  prior Prod ──►  [Archived]
      │                                                           [Archived]
      │
      └── Otherwise  ──►  [Failed]  ──►  (terminal)
```

---

## 12. AI Components & Intelligent Automation

Beyond the core ML model, several AI-driven automation components make the system self-maintaining.

### 12.1 Automated Retraining Trigger

**Module**: `claim_denial/monitoring/retraining_trigger.py`  
**Class**: `RetrainingTrigger`

The system monitors its own production precision continuously. When precision degrades, it automatically initiates retraining without human intervention:

```
MonitoringService detects:
  rolling_30d_precision < 0.80
        │
        ▼
  retraining_state["retraining_trigger"] = "active"
        │
        ▼
  RetrainingTrigger.check_and_trigger()
  (called after each nightly pipeline)
        │
        ▼
  ModelTrainer.run_training(
    trigger_reason="precision_threshold_breach"
  )
        │
        ├── Staging promoted → mark_retrain_succeeded() → trigger = "inactive"
        └── Failed → consecutive_failures += 1
                         │
                         ├── < 3 failures: failure alert, trigger stays "active"
                         └── ≥ 3 failures: escalation alert (human required)
```

The system retains the existing Production model throughout retraining — billing operations are never disrupted.

### 12.2 Intelligent Alert Dispatch

**Module**: `claim_denial/monitoring/monitoring_service.py`

The monitoring system uses tiered alerting with retry logic:

| Precision Level | Action | Alert Type |
|---|---|---|
| ≥ 0.85 | No action | — |
| 0.80–0.85 | Degradation alert | Email / PagerDuty webhook |
| < 0.80 | Urgent alert + retraining trigger | Immediate escalation |

Alert delivery failures are retried 3× at 5-minute intervals before logging a final DELIVERY_FAILURE record — the system never silently drops alerts.

### 12.3 Smart Feature Fallbacks

The system incorporates intelligence into feature computation to handle real-world data quality issues:

- **Cold-start handling**: population-mean denial rates for new providers/payers with insufficient history
- **Missing date sentinels**: distinguishable -1 (discharge absent) vs -2 (both dates absent) in length-of-stay
- **NLP degradation**: neutral defaults (0.5, 0, zeros) when clinical notes are absent or inference fails
- **Degraded scoring**: claims score even when some features are null; `score_quality_flag = "degraded"` alerts downstream users

### 12.4 Property-Based Correctness Verification

The system uses Hypothesis-based property-based testing to verify 14 mathematical properties that must hold across all possible inputs — not just the specific test cases a developer thought to write. See [Section 18](#18-testing-strategy).

---

## 13. Batch Scoring Pipeline

**Module**: `claim_denial/scoring/batch_scorer.py`  
**Class**: `BatchScorer`

### Pipeline Flow

```
02:00 local  ──► Airflow triggers DAG
                        │
                        ▼
               BatchScorer.load_production_model()
               ← ModelRegistry.get_production_model()
               ! PipelineBlockedError if no Production model
                        │
                        ▼
               BatchScorer.fetch_active_claims()
               ← FeatureStore.scan(claim_status="Active")
                        │
                        ▼
               BatchScorer.score_claims(model, claims)
               - Spark mapPartitions (or local fallback)
               - _detect_quality_flag() → "degraded" if null features
               - _invoke_model() → supports predict_proba / predict / callable
               - Clamp score to [0.0, 1.0], cast to float32
                        │
                        ▼
               BatchScorer.write_scores(results)
               - Upsert: predicted_denial_score, model_version_id,
                         scoring_timestamp_utc, score_quality_flag
               - Retry ×3 with 1s/2s/4s backoff per claim
                        │
                        ▼
               BatchScorer.emit_completion_event(stats)
               {total_active, total_scored, degraded, failed, wall_clock_secs}
                        │
                        ▼
               Airflow SLA check: wall_clock > 4 hours?
               → SLA breach alert emitted (pipeline continues)
```

### SLA Guarantees

- All active claims scored within **4 hours** of pipeline start
- NLP phase completes within **3 hours** (leaves 1-hour buffer)
- SLA breach triggers alert but **never halts** the pipeline (partial scoring is better than no scoring)
- Container image tag and digest logged at pipeline start for reproducibility and audit

### Dockerized Execution

The Batch_Scorer runs inside a Dockerized container with:
- Model artifact embedded at build time
- All Python dependencies at pinned versions
- Container image tag and digest logged by Airflow at each run start

---

## 14. Post-Deployment Monitoring

**Module**: `claim_denial/monitoring/monitoring_service.py`  
**Class**: `MonitoringService`

### ERA/835 Outcome Linkage

When a payer returns an ERA/835 adjudication response, `MonitoringService.link_adjudication_outcome()` matches the claim to its Feature Store row and writes the final outcome:

```python
{
  "claim_id": "CLM-000123",
  "adjudication_status": "denied",
  "processed_timestamp_utc": "2024-01-16T14:30:00Z"
}
```

This must complete within **1 hour** of the ERA/835 being written to the Data Lake.

### Rolling 30-Day Precision

After each nightly pipeline, the system computes rolling precision:

```
For all claims where:
  - predicted_denial_score is not None
  - adjudication_outcome ∈ {"paid", "denied"}
  - scoring_timestamp_utc within last 30 days

precision = true_positives / (true_positives + false_positives)
where predicted_denial = score ≥ 0.5

Returns None if fewer than 30 qualifying claims (insufficient data)
```

### Monitoring Dashboard Record

Written nightly to the dashboard store:

```python
MonitoringRecord(
    computation_date=date.today(),
    rolling_precision=0.9234,       # or None
    window_days=30,
    total_scored_claims=847,
    predicted_denials=187,
    true_positive_denials=173,
    false_positive_denials=14,
    computation_timestamp_utc=datetime.now(utc),
)
```

---

## 15. Analytics Dashboard

**Module**: `dashboard/app.py`

The Streamlit dashboard provides visibility into the system for billing operations managers, clinical analysts, and ML engineers. It is backed by DuckDB (in-memory SQL) for analytical queries over synthetic claims data.

### 8 Pages

| Page | Audience | Key Content |
|---|---|---|
| 🏠 Executive Summary | Operations manager | KPI cards (total claims, denial rate, revenue at risk, model precision), rolling 7-day trend, top denial reasons |
| 📈 Denial Trend Analysis | Operations analyst | Rolling 7-day + 30-day denial rate, month-over-month cohort, MoM change table |
| 🏦 Payer Intelligence | Billing director | Payer ranking table (RANK/NTILE), scatter of revenue risk vs denials, pre-auth gap impact |
| 🩺 Clinical Analysis | Clinical analyst | ICD-10 hotspot treemap, specialty denial ranking, avg LOS by specialty |
| 🤖 Model Performance | ML engineer | Confusion matrix breakdown, score histogram, calibration chart, SHAP top-20, rolling precision history |
| 📦 Model Registry | ML engineer | Version history, metrics comparison across versions, stage visualization |
| ⚙️ Pipeline Health | DevOps/MLOps | Nightly run history, wall-clock vs 4-hour SLA, scored vs failed counts |
| 🔍 SQL Workbench | Any analyst | Live DuckDB SQL editor over the claims table, 8 preset analytical queries |

### SQL Analytics Engine

**Module**: `dashboard/analytics.py` — 18 DuckDB SQL queries

The dashboard makes extensive use of SQL window functions:

```sql
-- Rolling 7-day denial rate
AVG(denied * 1.0 / NULLIF(denied + paid, 0))
OVER (ORDER BY dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)

-- Payer ranking
RANK() OVER (ORDER BY denied DESC)
NTILE(4) OVER (ORDER BY denied DESC)       -- quartile
PERCENT_RANK() OVER (ORDER BY denied DESC) -- percentile

-- Month-over-month change
LAG(denial_rate) OVER (ORDER BY month)

-- Cumulative revenue at risk
SUM(daily_denied_charges) OVER (ORDER BY dt)

-- High-risk claims (top decile)
NTILE(10) OVER (ORDER BY predicted_denial_score DESC)

-- Provider performance percentile
PERCENT_RANK() OVER (ORDER BY denial_rate)
```

Notable analytical queries:
1. Rolling 7-day denial rate with revenue at risk (window functions)
2. Payer ranking with RANK, PERCENT_RANK, NTILE quartiles
3. Monthly MoM cohort with LAG for delta computation
4. High-risk claim detection (top 10% by denial score)
5. Specialty denial analysis with cumulative running totals
6. Confusion-matrix breakdown (TP/FP/TN/FN) with % of total
7. ICD-10 hotspot analysis with FIRST_VALUE
8. Score bucket calibration (actual denial rate per 0.1-wide score bucket)
9. Provider summary with FIRST_VALUE / dept-level risk rollup
10. Pre-auth gap impact comparison

### Global Filters

The dashboard supports cross-page filtering by:
- Payer (multiselect)
- Claim type (multiselect)
- Denial score range (0.0–1.0 slider)
- Date range (date picker)

---

## 16. Project Structure

```
Transforming_the_Healthcare_Revenue_Cycle_with_AI/
│
├── claim_denial/                          ← Main Python package
│   ├── __init__.py
│   ├── constants.py                       ← HCUP CCS/procedure lookups, sentinels, categories
│   ├── models.py                          ← Domain dataclasses (ClaimRecord, EHRRecord,
│   │                                         ClaimFeatureRecord, MonitoringRecord, etc.)
│   │
│   ├── dags/
│   │   ├── __init__.py
│   │   └── claim_denial_dag.py            ← Airflow DAG (02:00 nightly, 4-hour SLA)
│   │
│   ├── features/
│   │   ├── claim_level.py                 ← Temporal, demographics, one-hot encoding
│   │   ├── clinical.py                    ← CCS, Charlson, procedure vector, LOS, mismatch
│   │   ├── payer_behavior.py              ← Payer denial rates, days-since-payment, stop-loss
│   │   └── provider_facility.py          ← LOO encoding, specialty, volume, bed size
│   │
│   ├── ingestion/
│   │   ├── claim_processor.py             ← X12 837 parser, Data Lake writer, retry/quarantine
│   │   └── ehr_ingestion_service.py       ← FHIR R4 client, encounter selection, note extraction
│   │
│   ├── monitoring/
│   │   ├── monitoring_service.py          ← Rolling precision, ERA linkage, alert dispatch
│   │   └── retraining_trigger.py          ← Automated retraining on precision degradation
│   │
│   ├── nlp/
│   │   └── nlp_processor.py               ← Clinical BERT (stub), pre-auth gap classifier
│   │
│   ├── pipeline/
│   │   └── spark_pipeline.py              ← PySpark wiring for all 5 pipeline phases
│   │
│   ├── registry/
│   │   └── model_registry.py              ← MLflow-backed registry, stage machine, human gates
│   │
│   ├── schemas/
│   │   └── registry.py                    ← Avro schema migration registry (version hop chains)
│   │
│   ├── scoring/
│   │   └── batch_scorer.py                ← Nightly batch scorer, Spark/local inference
│   │
│   ├── training/
│   │   ├── dataset_builder.py             ← 90/180-day window, labeling, temporal split
│   │   └── model_trainer.py               ← Optuna HPO, XGBoost/LGB/CatBoost, ECE, SHAP
│   │
│   └── utils/
│       ├── feature_store_client.py        ← HBase/Cassandra abstraction with retry
│       ├── serializer.py                  ← Avro serialize/deserialize, migration, pretty_print
│       └── synthetic_data.py              ← PHI-free X12 837, EHR, ERA/835, clinical note generators
│
├── dashboard/
│   ├── app.py                             ← 8-page Streamlit dashboard
│   ├── analytics.py                       ← 18 DuckDB SQL analytical queries
│   ├── data_generator.py                  ← Synthetic dashboard data (claims, monitoring, etc.)
│   └── sql_analytics.py                   ← SQL layer for app.py pages
│
├── .kiro/
│   └── specs/claim-denial-prediction/
│       ├── requirements.md                ← 15 requirements, 90+ acceptance criteria
│       ├── design.md                      ← Architecture, component interfaces, data models
│       └── tasks.md                       ← Implementation task checklist (19 task groups)
│
├── pyproject.toml                         ← Build config, pinned dependencies
└── PROJECT_OVERVIEW.md                    ← This document
```

---

## 17. Key Technical Decisions

### Why gradient-boosted trees (not deep learning)?

- Claims data is **tabular** — GBTs consistently outperform deep learning on tabular data
- GBTs provide **SHAP feature importances**, which are required for regulatory explainability
- Training time is fast enough for nightly retraining cycles
- XGBoost/LightGBM have mature Spark ML integration for distributed training

### Why BERT only for NLP (not a larger LLM)?

- Clinical notes are long (discharge summaries can be thousands of words); BERT handles this with the 10,000-token truncation strategy
- Clinical BERT models (Bio_ClinicalBERT, etc.) are pre-trained on clinical text, making them well-suited for this domain
- A 768-dim embedding captures the full semantic complexity of the note as a single feature vector
- LLMs would be overkill and prohibitively expensive to run in a 3-hour nightly window for thousands of claims

### Why Avro for the Feature Store schema?

- Avro is the standard serialization format for Hadoop/HBase ecosystems (which includes many healthcare IT environments)
- Schema evolution is built-in via the `schema_version` field and migration registry
- Binary format is compact and fast for the wide (50+ column) rows
- `fastavro` is a pure-Python implementation that works without JVM

### Why DuckDB for the dashboard?

- Runs **in-process** — no server required, no network calls, no infrastructure
- Speaks standard ANSI SQL including window functions
- Registers pandas DataFrames directly as tables
- Executes complex analytical queries on 2,000+ row synthetic datasets in milliseconds

### Why Optuna for hyperparameter optimization?

- Optuna's TPE (Tree-structured Parzen Estimator) sampler is state-of-the-art for hyperparameter search
- Native integration with XGBoost, LightGBM, and CatBoost
- Supports pruning of unpromising trials for efficiency
- MLflow integration for tracking each trial

### Why population-mean fallback instead of zero/null?

- Zero denial rate would signal "never denied" which is misleading for new providers/payers
- 0.5 (neutral) could be used but the population mean is more informative — it reflects the true base rate
- The population mean is computed from the same trailing window, so it is contextually relevant

---

## 18. Testing Strategy

The project uses a **dual testing approach** combining example-based unit tests with property-based tests using Hypothesis.

### Property-Based Tests (Hypothesis)

14 mathematical correctness properties are verified across 100–200 randomized inputs each:

| Property | What It Verifies |
|---|---|
| **P1**: Temporal split boundary | All claims on boundary date → test set; test = most recent 20% |
| **P2**: Class weight volume | Applying class weights never changes record count |
| **P3**: LOO encoding range | Leave-one-out target encoding ∈ [0.0, 1.0] for all providers with >30 claims |
| **P4**: Population-mean range | Fallback denial rate ∈ [0.0, 1.0] from entities with ≥1 known outcome |
| **P5**: Charlson non-negative | CCI score ≥ 0 for any secondary ICD-10 list; = 0 for empty list |
| **P6**: CCS category range | HCUP CCS category ∈ [0, 285] for any ICD-10 string |
| **P7**: Procedure vector binary | Vector length = 25 (fixed), all elements ∈ {0, 1} for any CPT set |
| **P8**: One-hot consistency | Each one-hot vector: all values ∈ {0,1}, sum ∈ {0, 1} |
| **P9**: Avro round-trip fidelity | serialize → deserialize → serialize produces identical bytes (200 examples) |
| **P10**: Gender sentinel | Gender = -1 for any null/unknown/absent input, no downstream blocking |
| **P11**: LOS sentinel distinguishable | -1 (discharge absent), -2 (both absent), ≥0 (normal) are distinguishable |
| **P12**: NLP defaults consistent | For absent/failed notes: necessity=0.5, pre_auth=0, embedding=zeros(768), flag∈{1,2} |
| **P13**: Score is probability | predicted_denial_score ∈ [0.0, 1.0] for any feature vector |
| **P14**: Payer rate range | payer_historical_denial_rate and by_type both ∈ [0.0, 1.0] |

### Unit Tests

Covering every module with targeted example-based tests:

- `test_claim_level.py` — DOY/month edge cases (Dec 31, Feb 29), age boundary (same-day birthday, future DOB, null DOB), all one-hot categories including "all-zeros" fallback
- `test_clinical.py` — CCS mapping (known/unknown codes), procedure vector (empty, unrecognized, recognized), Charlson (from mapping, not in mapping, empty list)
- `test_payer_behavior.py` — denial rate threshold boundary (49 vs 50), days_since_last_payment cold-start (-1), stop-loss lookup missing payer
- `test_provider_facility.py` — LOO boundary (exactly 30 vs 31 claims), NUCC sentinel, bed-size mean fallback
- `test_nlp_processor.py` — chronological note ordering, 10,000-token truncation, keyword classifier (known phrases → 1, no phrase → 0), model name/version logging
- `test_monitoring_service.py` — rolling Precision computation, alert thresholds (0.84 → degradation, 0.79 → urgent), alert retry exhaustion, ERA linkage
- `test_retraining_trigger.py` — state machine transitions, escalation at 3 consecutive failures, mark_retrain_succeeded sets trigger inactive
- `test_serializer.py` — round-trip per field type, schema migration v1.0.0 → v1.1.0, QuarantineError on missing migration rule, pretty_print valid JSON
- `test_feature_store_client.py` — upsert retry (2 failures + success), retention window alert
- `test_training_dataset_builder.py` — 90-day / 180-day expansion, exclusion logging, halt condition

### HIPAA / PHI Testing Requirements

- All PHI fields (patient name, DOB, member ID, NPI) must use synthetic data in every test
- Integration test environments use de-identified or synthetic claim datasets
- No real patient data in test fixtures, logs, or CI artifacts

---

## 19. Infrastructure & Scale

### Throughput Target

The system is designed to handle **≥ 1 million claims per year**, which translates to approximately **2,740 claims per day**. This must complete within the **4-hour nightly batch window**.

Spark cluster configuration for the throughput target:
```
executors: 10
executor_cores: 4  
executor_memory: 8g
shuffle_partitions: 80 (num_executors × executor_cores × 2)
```

### Feature Store Retention

The Feature Store must retain **at least 90 days** of historical claim records at all times, because:
- Payer and provider denial rates use a trailing 90-day window
- Removing records from the window would change feature values for active claims

A daily retention check (`FeatureStoreClient.check_retention_window()`) verifies the oldest record timestamp and emits an alert if the window falls below 90 days.

### Key Infrastructure Components

| Component | Technology | Purpose |
|---|---|---|
| Orchestration | Apache Airflow 2.9.1 | Nightly 02:00 DAG, SLA monitoring, task dependency chain |
| Distributed compute | Apache Spark 3.5.1 (PySpark) | Feature engineering, batch scoring, NLP at scale |
| Feature Store | HBase / Cassandra | Wide-column NoSQL, Claim ID as row key |
| Data Lake | AWS S3 / Azure Blob | Raw EDI, parsed claims, EHR records, ERA files |
| ML models | XGBoost 2.0.3, LightGBM 4.3.0 | Gradient-boosted trees for denial prediction |
| Experiment tracking | MLflow 2.13.0 | Model versioning, metrics, artifacts, promotion |
| Hyperparameter search | Optuna 3.6.1 | Bayesian optimization (≥50 trials) |
| NLP model | Transformers 4.41.0 (HuggingFace) | Clinical BERT embeddings |
| Feature explainability | SHAP 0.45.0 | Top-20 feature importance per model |
| Serialization | fastavro 1.9.4, pyarrow 16.0.0 | Avro schema with versioning |
| Testing | Hypothesis 6.103.1 | Property-based testing |
| Dashboard | Streamlit + Plotly | 8-page analytics UI |
| SQL analytics | DuckDB | In-process analytical SQL with window functions |

### Retry Policies

All external write operations use **exponential backoff** (1s → 2s → 4s):

| Operation | Max Retries | Backoff |
|---|---|---|
| Data Lake write | 3 | 1s, 2s, 4s |
| Feature Store upsert | 3 | 1s, 2s, 4s |
| Alert delivery | 3 | 5 min, 5 min, 5 min |

After retry exhaustion:
- Data Lake write failures → quarantine + structured error log
- Feature Store failures → log write-failure record, continue next claim
- Alert failures → log DELIVERY_FAILURE record

---

## 20. Compliance & Security

### HIPAA Compliance

- **PHI handling**: All patient identifiers are treated as PHI. Synthetic data is used in all tests and development environments.
- **7-year retention**: Every raw EDI file receives a `.meta.json` sidecar tagging it for 7-year retention per HIPAA/CMS regulatory requirements.
- **Audit trail**: Every scoring event records `model_version_id`, `scoring_timestamp_utc`, and `score_quality_flag` in the Feature Store — a complete, queryable audit log.
- **Data partitioning**: Data Lake is partitioned by date and data source type, limiting blast radius of any accidental exposure.

### Access Controls

- Production deployment requires human reviewer approval to promote any model from Staging to Production
- `reviewer_id`, `review_timestamp`, and optionally `rejection_reason` are recorded on every gate action
- No model can skip the Staging → Production gate programmatically

### Data Integrity

- Avro schema with embedded `schema_version` in every record prevents silent schema drift
- Records with unknown schema versions are **quarantined** (not silently ignored) and logged with claim ID, embedded version, and current version
- `ValidationResult` from structural validation is explicit — a file is either valid or it returns the specific violation segment identifier and description

### No Real PHI in Codebase

The `SyntheticDataGenerator` was built specifically to enable comprehensive testing without any risk of PHI exposure:
- Generates structurally valid X12 837 EDI files with random (non-real) NPIs, patient account numbers, and dates
- Uses template-based clinical note generation with synthetic patient encounters
- All provider identifiers, payer IDs, and member IDs are randomly generated alphanumeric strings

---

## Appendix: Dependency Summary

```toml
[project.dependencies]
pyspark = "3.5.1"              # Distributed compute
xgboost = "2.0.3"              # ML model (primary)
lightgbm = "4.3.0"             # ML model (alternative)
mlflow = "2.13.0"              # Experiment tracking + model registry
hypothesis = "6.103.1"         # Property-based testing
fastavro = "1.9.4"             # Avro serialization
pyarrow = "16.0.0"             # Columnar data
apache-airflow = "2.9.1"       # Pipeline orchestration
optuna = "3.6.1"               # Bayesian hyperparameter optimization
shap = "0.45.0"                # SHAP feature importance
transformers[torch] = "4.41.0" # Clinical BERT NLP
pydantic = "2.7.1"             # Data validation
numpy = "1.26.4"               # Numerical computing
```

Additional development dependencies (dashboard):
```
streamlit, plotly, duckdb, pandas, scikit-learn
```

---

*This document covers the complete Transforming the Healthcare Revenue Cycle with AI project — from raw X12 837 EDI files through NLP, ML model training, nightly batch scoring, automated monitoring, and the analytics dashboard that makes it all visible.*
