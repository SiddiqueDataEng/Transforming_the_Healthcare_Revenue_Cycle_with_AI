# Transforming the Healthcare Revenue Cycle with AI

> **Predict claim denials before they happen — and stop revenue from walking out the door.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PySpark 3.5](https://img.shields.io/badge/spark-3.5.1-orange)](https://spark.apache.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-2.0.3-green)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/mlflow-2.13.0-blue)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/dashboard-streamlit-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

---

## What This Project Does

US hospitals lose billions of dollars annually to insurance claim denials. Most denials are preventable — wrong diagnosis codes, missing prior authorizations, payer-specific quirks that billing staff can't memorize for every payer.

This system uses machine learning to **score every claim for denial risk the night before submission**, giving billing teams a prioritized worklist of high-risk claims to fix while there is still time.

```
X12 837 EDI + EHR Data
        ↓
  Feature Engineering (50+ features)
  Clinical NLP (BERT embeddings)
        ↓
  Gradient-Boosted Ensemble Model
        ↓
  predicted_denial_score ∈ [0.0, 1.0]
        ↓
  "Fix these 47 claims before 8am"
```

**Key targets:** Precision ≥ 0.90 · ECE ≤ 0.05 · All claims scored within 4 hours

---

## Quick Start

### Run the Analytics Dashboard

```bash
# Clone and set up
git clone https://github.com/your-org/healthcare-rcm-ai.git
cd healthcare-rcm-ai

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -e ".[dev]"

# Launch the dashboard
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

### Run the Test Suite

```bash
# All tests
pytest claim_denial/ -v

# Property-based tests only
pytest claim_denial/tests/ -k "pbt" -v

# With coverage
pytest claim_denial/ --cov=claim_denial --cov-report=html
```

---

## Dashboard

The project ships with an **8-page interactive Streamlit dashboard** powered by DuckDB in-memory SQL analytics. No server required — it runs entirely in your browser.

| Page | What you'll see |
|---|---|
| 🏠 **Executive Summary** | KPI cards, rolling 7-day denial trend, revenue at risk by payer |
| 📈 **Denial Trend Analysis** | Rolling averages, month-over-month cohort analysis |
| 🏦 **Payer Intelligence** | Payer ranking (RANK/NTILE), risk scatter, pre-auth gap impact |
| 🩺 **Clinical Analysis** | ICD-10 hotspot treemap, specialty denial breakdown, avg LOS |
| 🤖 **Model Performance** | Confusion matrix, score calibration, SHAP feature importance |
| 📦 **Model Registry** | Version history, metric comparison across model versions |
| ⚙️ **Pipeline Health** | Nightly run stats, SLA tracking, scored vs failed counts |
| 🔍 **SQL Workbench** | Live DuckDB SQL editor — run your own queries over claims data |

> All dashboard data is **100% synthetic**. No real PHI is used anywhere in this project.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Apache Airflow DAG  ·  02:00 nightly               │
│  ingest_claims → ingest_ehr → engineer_features → run_nlp       │
│               → score_claims → run_monitoring                   │
└──────────┬──────────────┬───────────────────────────────────────┘
           │              │
    ┌──────▼──────┐  ┌────▼──────────┐
    │  X12 837    │  │  EHR (FHIR)   │
    │  EDI Files  │  │  Clinical Notes│
    └──────┬──────┘  └────┬──────────┘
           │              │
           └──────┬───────┘
                  ▼
          ┌───────────────┐        ┌──────────────────┐
          │   Data Lake   │        │  Clinical BERT   │
          │ (S3 / Blob)   │        │  NLP Processor   │
          └──────┬────────┘        └────────┬─────────┘
                 │                          │
                 └──────────┬───────────────┘
                            ▼
                  ┌─────────────────────┐
                  │    Feature Store    │
                  │ (HBase / Cassandra) │
                  │  50+ columns/claim  │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │    Batch Scorer     │
                  │  (Apache Spark)     │
                  │  score ∈ [0, 1]     │
                  └──────────┬──────────┘
                             │
              ┌──────────────▼──────────────┐
              │     Monitoring Service      │
              │  Rolling 30-day Precision   │
              │  Auto-retraining trigger    │
              └─────────────────────────────┘
```

---

## Feature Engineering

The model is trained on **50+ engineered features** across five groups:

### Claim-Level (22 features)
Temporal signals, patient demographics, and one-hot encoded administrative codes.

```
claim_submission_date_doy, claim_submission_date_month
patient_age, gender
ins_type_medicare/medicaid/commercial/tricare/champva/other  (6)
claim_type_professional, claim_type_institutional
adm_type_emergency/elective/urgent/trauma  (4)
adm_src_physician_referral/transfer_hospital/snf/er/court/na  (6)
```

### Provider & Facility (5 features)
Historical denial behavior of the billing provider, encoded without data leakage.

```
billing_provider_id_encoded    ← Leave-one-out target encoding
                                 (population-mean fallback if <30 claims)
performing_physician_specialty ← NUCC taxonomy code
provider_historical_denial_rate
provider_claim_volume
facility_bed_size
```

### Clinical (6 features)
ICD-10 mapped to HCUP CCS categories, Charlson Comorbidity Index, procedure matching.

```
principal_dx_ccs_category      ← HCUP CCS (1–285), 0 = unknown
secondary_dx_count
procedure_category_vector      ← 25-element multi-hot binary vector
length_of_stay                 ← days; −1 discharge absent; −2 both absent
comorbidity_risk_score         ← Charlson Comorbidity Index (ICD-10 prefix matching)
diagnosis_procedure_mismatch   ← binary: diagnosis ↔ procedure inconsistency
```

### Payer Behavior (4 features)
Payer-specific denial tendencies from trailing 90-day ERA/835 history.

```
payer_historical_denial_rate
payer_historical_denial_rate_by_type  ← Professional vs Institutional
days_since_last_payment               ← −1 for cold-start
payer_contract_stop_loss              ← binary from contract reference table
```

### NLP — Clinical Notes (3 features)
Free-text clinical notes processed through a clinical BERT model.

```
documents_medical_necessity    ← scalar [0,1] from 768-dim BERT embedding
mentions_lack_of_pre_auth      ← binary keyword/regex classifier
tx_plan_complexity_embedding   ← float32[768] full BERT embedding
```

---

## Machine Learning

### Model

Gradient-boosted ensemble (XGBoost / LightGBM / CatBoost) trained with Bayesian hyperparameter optimization via Optuna (≥ 50 trials), maximizing **Precision** on a temporal validation split.

| Metric | Threshold | Why |
|---|---|---|
| **Precision** | ≥ 0.90 | Primary — false positives waste billing staff time and erode trust |
| **ECE** | ≤ 0.05 | Score must be a calibrated probability, not just a ranking |
| Recall | tracked | Secondary — want to catch as many true denials as possible |
| AUC-ROC | tracked | Overall discriminative ability |
| AUC-PR | tracked | Performance on imbalanced classes |

### Training Pipeline

```
Historical claims (90-day window, expanded to 180 if <1,000 records)
        ↓
Temporal 70 / 15 / 15 split  (train / validation / test)
        ↓
Optuna Bayesian HPO  (≥50 trials, maximize Precision on validation)
        ↓
Final model training  (best hyperparameters)
        ↓
Evaluation on held-out test set
        ↓
SHAP feature importance  (top-20 logged to MLflow)
        ↓
Register: Staging (Precision≥0.90 AND ECE≤0.05)  or  Failed
```

### Model Registry Stage Machine

```
Training
   ├── Precision≥0.90 AND ECE≤0.05  ──►  Staging
   │                                        │
   │                              Human review gate
   │                          ┌──────────────┴──────────┐
   │                       Approve                    Reject
   │                          │                          │
   │                     Production  ──►  prior Prod ──► Archived
   │                                                     Archived
   └── Otherwise  ──►  Failed
```

---

## NLP Pipeline

```
Clinical Notes (H&P + Discharge Summary + Progress Notes)
        ↓
Chronological sort by note_datetime
        ↓
Concatenate + truncate to 10,000 tokens
        ↓
Clinical BERT encode  →  float32[768]
        ├──►  medical_necessity score  =  mean(embedding), clamped [0,1]
        └──►  pre_auth_gap  =  regex scan for:
                               "prior authorization", "pre-authorization",
                               "auth gap", "preauth", "not authorized", …
```

**Defaults when notes are absent or inference fails:**

| Scenario | medical_necessity | pre_auth | embedding | flag |
|---|---|---|---|---|
| No clinical notes | 0.5 | 0 | zeros(768) | 1 |
| Inference error | 0.5 | 0 | zeros(768) | 2 |

---

## Automated Retraining

The system monitors its own production precision and self-corrects:

```
Nightly: rolling 30-day Precision computed
   │
   ├── ≥ 0.85  ──►  No action
   ├── 0.80–0.85  ──►  Degradation alert  (email / PagerDuty)
   └── < 0.80  ──►  Urgent alert  +  retraining_trigger = "active"
                            │
                   RetrainingTrigger fires
                            │
                   ModelTrainer.run_training(
                       trigger_reason="precision_threshold_breach"
                   )
                            │
              ┌─────────────┴──────────────┐
           Success                      Failed
              │                            │
     Staging → Production        consecutive_failures++
     trigger = "inactive"        trigger stays "active"
                                 failure alert dispatched
                                      │
                                 ≥3 consecutive
                                      │
                                 escalation alert
                                 (human required)
```

The existing Production model **always stays in service** during retraining.

---

## Data Engineering

### Data Lake Layout

```
data_lake/
├── claims/YYYY/MM/DD/
│   ├── raw/                 ← original X12 837 EDI + .meta.json (7-year retention)
│   └── parsed/              ← normalized claim JSON
├── ehr/{account}/{date}/    ← structured EHR + clinical notes
├── era/YYYY/MM/DD/          ← ERA/835 adjudication outcomes
└── quarantine/YYYY/MM/DD/   ← invalid or failed files
```

### Error Handling

Every failure is handled without silently dropping data:

| Error | Action |
|---|---|
| X12 structural validation failure | Quarantine file + log `{file, segment_id, violation}` |
| Missing required claim fields | Skip claim + log `{claim_id, field_name}`, continue file |
| Data Lake write failure | Retry ×3 (1s→2s→4s) → quarantine + error log |
| EHR not found | `ehr_null_match_flag=1`, continue with claim-only features |
| Feature Store write failure | Retry ×3 (1s→2s→4s) → log failure, continue |
| Alert delivery failure | Retry ×3 at 5-min intervals → log DELIVERY_FAILURE |

### Feature Store Schema (Avro v1.0.0)

The Feature Store holds a single denormalized row per claim with **50+ columns** covering all feature groups, scoring outputs, and ground-truth labels. Schema is versioned with embedded `schema_version` and a `SchemaRegistry` handles chained migrations.

---

## Testing

### Property-Based Tests (Hypothesis)

14 mathematical correctness properties verified across **100–200 randomized inputs** each:

| # | Property | Validates |
|---|---|---|
| P1 | Temporal split boundary date assignment | Req 8.3 |
| P2 | Class weight balancing preserves record count | Req 8.5 |
| P3 | LOO encoding ∈ [0.0, 1.0] | Req 4.1 |
| P4 | Population-mean fallback ∈ [0.0, 1.0] | Req 4.2, 6.5 |
| P5 | Charlson score ≥ 0; = 0 for empty list | Req 5.5 |
| P6 | HCUP CCS category ∈ [0, 285] | Req 5.1 |
| P7 | Procedure vector: fixed length, all ∈ {0,1} | Req 5.3 |
| P8 | One-hot vectors: all ∈ {0,1}, sum ∈ {0,1} | Req 3.5–3.8 |
| P9 | Avro round-trip: byte-identical (200 examples) | Req 15.4 |
| P10 | Gender sentinel = −1 for null/unknown | Req 3.4 |
| P11 | LOS sentinels −1/−2/≥0 are distinguishable | Req 5.4 |
| P12 | NLP defaults consistent for absent/failed notes | Req 7.4, 7.5 |
| P13 | Denial score ∈ [0.0, 1.0] | Req 11.3 |
| P14 | Payer denial rates ∈ [0.0, 1.0] | Req 6.1, 6.2 |

### Unit Tests

Full unit test coverage across all modules — ingestion, feature engineering, NLP, training, scoring, monitoring, registry, and serialization.

```bash
pytest claim_denial/ -v --tb=short
```

---

## Project Structure

```
├── claim_denial/
│   ├── constants.py              HCUP lookups, sentinels, one-hot categories
│   ├── models.py                 Domain dataclasses (Claim, EHR, Features, etc.)
│   ├── dags/
│   │   └── claim_denial_dag.py   Airflow DAG — 02:00 nightly, 4-hour SLA
│   ├── features/
│   │   ├── claim_level.py        Temporal, demographics, one-hot
│   │   ├── clinical.py           CCS, Charlson, procedure vector, LOS, mismatch
│   │   ├── payer_behavior.py     Denial rates, days-since-payment, stop-loss
│   │   └── provider_facility.py  LOO encoding, specialty, volume, bed size
│   ├── ingestion/
│   │   ├── claim_processor.py    X12 837 parser, Data Lake writer
│   │   └── ehr_ingestion_service.py  FHIR client, encounter selection, notes
│   ├── monitoring/
│   │   ├── monitoring_service.py Rolling precision, ERA linkage, alerts
│   │   └── retraining_trigger.py Automated retraining on precision degradation
│   ├── nlp/
│   │   └── nlp_processor.py      Clinical BERT (stub), pre-auth classifier
│   ├── pipeline/
│   │   └── spark_pipeline.py     PySpark wiring for all 5 pipeline phases
│   ├── registry/
│   │   └── model_registry.py     Stage machine, human review gates, MLflow
│   ├── schemas/
│   │   └── registry.py           Avro schema migration registry
│   ├── scoring/
│   │   └── batch_scorer.py       Nightly batch scorer, Spark + local fallback
│   ├── training/
│   │   ├── dataset_builder.py    90/180-day window, labeling, temporal split
│   │   └── model_trainer.py      Optuna HPO, XGBoost/LGB/CatBoost, SHAP
│   └── utils/
│       ├── feature_store_client.py  HBase/Cassandra abstraction with retry
│       ├── serializer.py            Avro serialize/deserialize, migrations
│       └── synthetic_data.py        PHI-free X12 837, EHR, ERA generators
│
├── dashboard/
│   ├── app.py                    8-page Streamlit dashboard
│   ├── analytics.py              18 DuckDB SQL queries (window functions)
│   ├── data_generator.py         Synthetic dashboard data
│   └── sql_analytics.py          SQL analytics layer
│
├── .kiro/specs/claim-denial-prediction/
│   ├── requirements.md           15 requirements, 90+ acceptance criteria
│   ├── design.md                 Architecture, component interfaces, schemas
│   └── tasks.md                  Implementation task checklist
│
├── PROJECT_OVERVIEW.md           Comprehensive technical documentation (20 sections)
├── pyproject.toml                Pinned dependencies
└── README.md                     This file
```

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pyspark` | 3.5.1 | Distributed feature engineering and scoring |
| `xgboost` | 2.0.3 | Primary ML algorithm |
| `lightgbm` | 4.3.0 | Alternative ML algorithm |
| `mlflow` | 2.13.0 | Experiment tracking and model registry |
| `apache-airflow` | 2.9.1 | Nightly pipeline orchestration |
| `optuna` | 3.6.1 | Bayesian hyperparameter optimization |
| `shap` | 0.45.0 | SHAP feature importance |
| `transformers[torch]` | 4.41.0 | Clinical BERT NLP |
| `fastavro` | 1.9.4 | Avro serialization with schema versioning |
| `hypothesis` | 6.103.1 | Property-based testing |
| `streamlit` | 1.63.0 | Analytics dashboard |
| `duckdb` | 1.5.5 | In-process SQL analytics |
| `numpy` | 1.26.4 | Numerical computing |
| `pydantic` | 2.7.1 | Data validation |

---

## Compliance

- **HIPAA**: All PHI fields treated as protected. Synthetic data only in tests and development.
- **7-year retention**: Every raw EDI file receives a `.meta.json` retention tag per HIPAA/CMS requirements.
- **Human review gate**: No model reaches Production without a reviewer approval (recorded with `reviewer_id` + timestamp).
- **Audit trail**: Every scoring event writes `model_version_id`, `scoring_timestamp_utc`, and `score_quality_flag` to the Feature Store.
- **No real PHI**: The `SyntheticDataGenerator` enables full integration testing without any real patient data.

---

## Documentation

- **[`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md)** — deep-dive technical documentation covering all 20 aspects of the system
---

## Disclaimer

> This project is an engineering and machine-learning reference implementation using **synthetic healthcare data**. It is not medical advice, a clinical decision-support system, a billing guarantee, or a substitute for professional revenue-cycle, coding, compliance, legal, or clinical review.

Production deployment would require appropriate validation, security controls, privacy assessments, organizational approval, payer-specific validation, and applicable regulatory/compliance review.
---

*Built with Python 3.10+ · Apache Spark · XGBoost · Clinical BERT · MLflow · Apache Airflow · Streamlit*
