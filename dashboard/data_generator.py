"""
Synthetic data generator for the Streamlit dashboard.

Produces realistic-looking (but entirely synthetic, no real PHI) DataFrames
that power every page of the dashboard.  All random state is seeded so the
UI is reproducible across refreshes unless the user changes the seed.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd

# ── Seed ─────────────────────────────────────────────────────────────────────
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)
random.seed(RNG_SEED)

# ── Lookup tables ─────────────────────────────────────────────────────────────
PAYERS = ["Medicare", "Medicaid", "BlueCross", "Aetna", "Cigna", "UnitedHealth", "Humana"]
CLAIM_TYPES = ["Professional", "Institutional"]
DENIAL_REASONS = [
    "Missing Prior Auth",
    "Medical Necessity",
    "Duplicate Claim",
    "Incorrect Coding",
    "Eligibility Issue",
    "Timely Filing",
    "Out of Network",
    "Coverage Lapsed",
]
SPECIALTIES = [
    "Internal Medicine", "Cardiology", "Orthopedics", "Oncology",
    "Neurology", "Radiology", "Emergency Medicine", "Family Practice",
]
ICD10_DESCRIPTIONS = {
    "I21": "Acute MI", "J18": "Pneumonia", "M17": "Knee Osteoarthritis",
    "K80": "Cholelithiasis", "N18": "CKD", "G35": "Multiple Sclerosis",
    "C34": "Lung Cancer", "E11": "Type 2 Diabetes",
}
DEPARTMENTS = ["Med/Surg", "ICU", "Cardiology", "Oncology", "Orthopedics", "ED", "Neurology"]
MODEL_VERSIONS = ["v-a1b2c3d4", "v-e5f6g7h8", "v-i9j0k1l2"]
TODAY = date.today()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_dates(n: int, days_back: int = 90) -> List[date]:
    deltas = rng.integers(0, days_back, size=n)
    return [TODAY - timedelta(days=int(d)) for d in deltas]


def _denial_score(denied: bool) -> float:
    if denied:
        return float(np.clip(rng.normal(0.75, 0.12), 0.50, 0.99))
    return float(np.clip(rng.normal(0.22, 0.12), 0.01, 0.49))


# ── Public generators ─────────────────────────────────────────────────────────

def generate_claims(n: int = 2000) -> pd.DataFrame:
    """Return a DataFrame of n synthetic scored claims."""
    denied_mask = rng.random(n) < 0.22          # ~22 % denial rate
    submission_dates = _random_dates(n, 90)
    adjudication_dates = [d + timedelta(days=int(rng.integers(3, 21))) for d in submission_dates]

    records = []
    for i in range(n):
        denied = bool(denied_mask[i])
        score = _denial_score(denied)
        predicted_denial = score >= 0.5
        payer = random.choice(PAYERS)
        specialty = random.choice(SPECIALTIES)
        claim_type = random.choice(CLAIM_TYPES)
        icd_code = random.choice(list(ICD10_DESCRIPTIONS.keys()))
        model_ver = random.choice(MODEL_VERSIONS)
        charge = float(rng.uniform(300, 45000))
        los = int(rng.integers(0, 14))
        comorbidity = int(rng.integers(0, 8))

        records.append({
            "claim_id": f"CLM-{i+1:06d}",
            "submission_date": submission_dates[i],
            "adjudication_date": adjudication_dates[i],
            "payer": payer,
            "claim_type": claim_type,
            "specialty": specialty,
            "principal_icd10": icd_code,
            "diagnosis_description": ICD10_DESCRIPTIONS[icd_code],
            "charge_amount": round(charge, 2),
            "length_of_stay": los,
            "comorbidity_score": comorbidity,
            "predicted_denial_score": round(score, 4),
            "predicted_denial": predicted_denial,
            "actual_outcome": "denied" if denied else "paid",
            "denial_reason": random.choice(DENIAL_REASONS) if denied else None,
            "model_version": model_ver,
            "score_quality": "degraded" if rng.random() < 0.03 else "normal",
            "provider_denial_rate": round(float(rng.uniform(0.05, 0.45)), 3),
            "payer_denial_rate": round(float(rng.uniform(0.08, 0.40)), 3),
            "medical_necessity_score": round(float(rng.uniform(0.2, 1.0)), 4),
            "pre_auth_gap": int(rng.random() < 0.18),
            "department": random.choice(DEPARTMENTS),
        })

    df = pd.DataFrame(records)
    df["submission_date"] = pd.to_datetime(df["submission_date"])
    df["adjudication_date"] = pd.to_datetime(df["adjudication_date"])
    df["revenue_at_risk"] = df.apply(
        lambda r: r["charge_amount"] if r["predicted_denial"] else 0.0, axis=1
    )
    return df


def generate_monitoring_history(days: int = 90) -> pd.DataFrame:
    """Return rolling 30-day precision metrics for the last `days` days."""
    rows = []
    base_precision = 0.91
    for i in range(days):
        d = TODAY - timedelta(days=days - i)
        drift = rng.normal(0, 0.008)
        precision = float(np.clip(base_precision + drift, 0.72, 0.99))
        base_precision = precision  # random walk
        total = int(rng.integers(400, 900))
        pred_denials = int(total * rng.uniform(0.18, 0.28))
        tp = int(pred_denials * precision)
        fp = pred_denials - tp
        rows.append({
            "date": d,
            "rolling_precision": round(precision, 4),
            "total_scored": total,
            "predicted_denials": pred_denials,
            "true_positives": tp,
            "false_positives": fp,
            "alert": (
                "urgent" if precision < 0.80
                else "degradation" if precision < 0.85
                else None
            ),
        })
    return pd.DataFrame(rows)


def generate_model_versions() -> pd.DataFrame:
    """Return a registry of synthetic model versions."""
    rows = []
    for idx, vid in enumerate(["v-a1b2c3d4", "v-e5f6g7h8", "v-i9j0k1l2"]):
        precision = round(float(rng.uniform(0.90, 0.96)), 4)
        rows.append({
            "version_id": vid,
            "stage": "Production" if idx == 2 else "Archived",
            "algorithm": random.choice(["XGBoost", "LightGBM"]),
            "training_date": str(TODAY - timedelta(days=60 - idx * 20)),
            "precision": precision,
            "recall": round(float(rng.uniform(0.78, 0.88)), 4),
            "f1": round(float(rng.uniform(0.84, 0.92)), 4),
            "auc_roc": round(float(rng.uniform(0.93, 0.98)), 4),
            "auc_pr": round(float(rng.uniform(0.88, 0.95)), 4),
            "ece": round(float(rng.uniform(0.01, 0.05)), 4),
            "train_records": int(rng.integers(8000, 25000)),
            "denied_class_pct": round(float(rng.uniform(0.18, 0.26)), 4),
        })
    return pd.DataFrame(rows)


def generate_pipeline_runs(n: int = 30) -> pd.DataFrame:
    """Return nightly pipeline run summaries."""
    rows = []
    for i in range(n):
        run_date = TODAY - timedelta(days=n - i)
        total = int(rng.integers(600, 1200))
        scored = total - int(rng.integers(0, 5))
        degraded = int(rng.integers(0, 20))
        failed = total - scored
        wall_clock = float(rng.uniform(45, 210))
        rows.append({
            "run_date": run_date,
            "total_active_claims": total,
            "total_scored": scored,
            "degraded_count": degraded,
            "failed_count": failed,
            "wall_clock_minutes": round(wall_clock, 1),
            "sla_breach": wall_clock > 240,
            "model_version": MODEL_VERSIONS[-1],
        })
    return pd.DataFrame(rows)


def generate_shap_importance() -> pd.DataFrame:
    """Return top-20 feature importance values (synthetic SHAP)."""
    features = [
        "payer_historical_denial_rate", "provider_historical_denial_rate",
        "medical_necessity_score", "pre_auth_gap", "comorbidity_score",
        "length_of_stay", "principal_dx_ccs_category", "billing_provider_id_encoded",
        "payer_contract_stop_loss", "days_since_last_payment",
        "secondary_dx_count", "diagnosis_procedure_mismatch",
        "patient_age", "claim_submission_date_doy", "payer_denial_rate_by_type",
        "facility_bed_size", "provider_claim_volume", "ins_type_medicare",
        "adm_type_emergency", "documents_medical_necessity",
    ]
    shap_vals = sorted(
        [float(np.abs(rng.normal(0, 0.05))) for _ in features],
        reverse=True,
    )
    return pd.DataFrame({"feature": features, "mean_abs_shap": shap_vals})


def generate_payer_stats(claims_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-payer KPIs from the claims DataFrame."""
    return (
        claims_df.groupby("payer")
        .agg(
            total_claims=("claim_id", "count"),
            total_denied=("actual_outcome", lambda x: (x == "denied").sum()),
            total_predicted_denied=("predicted_denial", "sum"),
            avg_charge=("charge_amount", "mean"),
            total_revenue_at_risk=("revenue_at_risk", "sum"),
            avg_denial_score=("predicted_denial_score", "mean"),
        )
        .reset_index()
        .assign(
            actual_denial_rate=lambda df: (df["total_denied"] / df["total_claims"]).round(4),
            predicted_denial_rate=lambda df: (df["total_predicted_denied"] / df["total_claims"]).round(4),
            avg_charge=lambda df: df["avg_charge"].round(2),
            avg_denial_score=lambda df: df["avg_denial_score"].round(4),
        )
        .sort_values("total_revenue_at_risk", ascending=False)
    )


def generate_denial_reason_stats(claims_df: pd.DataFrame) -> pd.DataFrame:
    denied = claims_df[claims_df["actual_outcome"] == "denied"]
    return (
        denied.groupby("denial_reason")
        .agg(count=("claim_id", "count"), avg_charge=("charge_amount", "mean"))
        .reset_index()
        .assign(avg_charge=lambda df: df["avg_charge"].round(2))
        .sort_values("count", ascending=False)
    )
