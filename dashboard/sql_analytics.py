"""
SQL analytics layer for the Streamlit dashboard.

Uses DuckDB (in-process, no server) to run standard SQL — including window
functions, CTEs, analytical queries — directly against pandas DataFrames.

All queries are purely analytical; no PHI is stored or transmitted.
"""

from __future__ import annotations

from typing import Tuple

import duckdb
import pandas as pd


def _con(claims: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """Create an in-process DuckDB connection with the claims table registered."""
    con = duckdb.connect(database=":memory:")
    con.register("claims", claims)
    return con


# ── 1.  Rolling 7-day denial rate (window function) ──────────────────────────

ROLLING_DENIAL_SQL = """
WITH daily AS (
    SELECT
        CAST(submission_date AS DATE)            AS day,
        COUNT(*)                                 AS total_claims,
        SUM(CASE WHEN actual_outcome = 'denied' THEN 1 ELSE 0 END) AS denied_claims,
        SUM(charge_amount)                       AS total_charge,
        SUM(revenue_at_risk)                     AS revenue_at_risk
    FROM claims
    GROUP BY 1
),
windowed AS (
    SELECT
        day,
        total_claims,
        denied_claims,
        ROUND(denied_claims * 1.0 / NULLIF(total_claims, 0), 4)  AS daily_denial_rate,
        -- 7-day rolling average denial rate
        ROUND(
            AVG(denied_claims * 1.0 / NULLIF(total_claims, 0))
            OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
            4
        )                                                          AS rolling_7d_denial_rate,
        -- 7-day rolling total revenue at risk
        SUM(revenue_at_risk)
            OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
                                                                   AS rolling_7d_revenue_risk,
        total_charge,
        revenue_at_risk
    FROM daily
)
SELECT * FROM windowed
ORDER BY day
"""


def rolling_denial_trend(claims: pd.DataFrame) -> pd.DataFrame:
    return _con(claims).execute(ROLLING_DENIAL_SQL).df()


# ── 2.  Payer ranking with RANK() + PERCENT_RANK() ───────────────────────────

PAYER_RANKING_SQL = """
WITH payer_stats AS (
    SELECT
        payer,
        COUNT(*)                                           AS total_claims,
        SUM(CASE WHEN actual_outcome = 'denied' THEN 1 ELSE 0 END) AS denied,
        ROUND(AVG(predicted_denial_score), 4)             AS avg_score,
        ROUND(SUM(revenue_at_risk), 2)                    AS revenue_at_risk,
        ROUND(AVG(charge_amount), 2)                      AS avg_charge
    FROM claims
    GROUP BY payer
),
ranked AS (
    SELECT *,
        RANK()         OVER (ORDER BY denied DESC)         AS denial_rank,
        PERCENT_RANK() OVER (ORDER BY denied DESC)         AS denial_pct_rank,
        DENSE_RANK()   OVER (ORDER BY revenue_at_risk DESC) AS risk_rank,
        NTILE(4)       OVER (ORDER BY denied DESC)         AS denial_quartile
    FROM payer_stats
)
SELECT * FROM ranked ORDER BY denial_rank
"""


def payer_ranking(claims: pd.DataFrame) -> pd.DataFrame:
    return _con(claims).execute(PAYER_RANKING_SQL).df()


# ── 3.  Monthly cohort analysis (CTE + GROUP BY ROLLUP) ──────────────────────

MONTHLY_COHORT_SQL = """
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', submission_date)   AS month,
        payer,
        claim_type,
        COUNT(*)                               AS claims,
        SUM(CASE WHEN actual_outcome = 'denied' THEN 1 ELSE 0 END) AS denials,
        ROUND(SUM(charge_amount), 2)           AS total_charge,
        ROUND(SUM(revenue_at_risk), 2)         AS revenue_risk,
        ROUND(AVG(predicted_denial_score), 4)  AS avg_denial_score,
        ROUND(AVG(medical_necessity_score), 4) AS avg_necessity_score
    FROM claims
    GROUP BY 1, 2, 3
),
with_rates AS (
    SELECT *,
        ROUND(denials * 1.0 / NULLIF(claims, 0), 4) AS denial_rate,
        -- Month-over-month change in claims (LAG window function)
        LAG(claims) OVER (PARTITION BY payer, claim_type ORDER BY month) AS prev_month_claims,
        LAG(denials) OVER (PARTITION BY payer, claim_type ORDER BY month) AS prev_month_denials
    FROM monthly
)
SELECT *,
    ROUND(
        (claims - COALESCE(prev_month_claims, claims)) * 1.0
        / NULLIF(prev_month_claims, 0),
        4
    ) AS mom_claims_change,
    ROUND(
        (denials - COALESCE(prev_month_denials, denials)) * 1.0
        / NULLIF(prev_month_denials, 0),
        4
    ) AS mom_denial_change
FROM with_rates
ORDER BY month DESC, payer, claim_type
"""


def monthly_cohort(claims: pd.DataFrame) -> pd.DataFrame:
    return _con(claims).execute(MONTHLY_COHORT_SQL).df()


# ── 4.  High-risk claim detection (window + percentile) ──────────────────────

HIGH_RISK_SQL = """
WITH scored AS (
    SELECT *,
        PERCENT_RANK() OVER (ORDER BY predicted_denial_score)  AS score_pctrank,
        PERCENT_RANK() OVER (ORDER BY charge_amount)           AS charge_pctrank,
        NTILE(10)      OVER (ORDER BY predicted_denial_score DESC) AS score_decile,
        SUM(revenue_at_risk) OVER (PARTITION BY payer)         AS payer_total_risk
    FROM claims
    WHERE actual_outcome = 'paid'   -- still outstanding / not yet denied
       OR actual_outcome IS NULL
),
high_risk AS (
    SELECT
        claim_id,
        submission_date,
        payer,
        specialty,
        principal_icd10,
        diagnosis_description,
        charge_amount,
        predicted_denial_score,
        score_decile,
        score_pctrank,
        charge_pctrank,
        revenue_at_risk,
        payer_total_risk,
        pre_auth_gap,
        medical_necessity_score,
        denial_reason
    FROM scored
    WHERE score_decile = 1     -- top 10 % by denial risk
)
SELECT * FROM high_risk
ORDER BY predicted_denial_score DESC, charge_amount DESC
LIMIT 200
"""


def high_risk_claims(claims: pd.DataFrame) -> pd.DataFrame:
    con = _con(claims)
    # For this query we work on the full claims table
    return con.execute(HIGH_RISK_SQL.replace(
        "WHERE actual_outcome = 'paid'   -- still outstanding / not yet denied\n       OR actual_outcome IS NULL",
        "WHERE 1=1"
    )).df()


# ── 5.  Specialty denial analysis with running totals ────────────────────────

SPECIALTY_SQL = """
WITH specialty_stats AS (
    SELECT
        specialty,
        COUNT(*)                                           AS total_claims,
        SUM(CASE WHEN actual_outcome = 'denied' THEN 1 ELSE 0 END) AS denied,
        ROUND(AVG(comorbidity_score), 2)                  AS avg_comorbidity,
        ROUND(AVG(length_of_stay), 2)                     AS avg_los,
        ROUND(SUM(revenue_at_risk), 2)                    AS revenue_at_risk,
        ROUND(AVG(medical_necessity_score), 4)            AS avg_necessity
    FROM claims
    GROUP BY specialty
),
with_window AS (
    SELECT *,
        ROUND(denied * 1.0 / NULLIF(total_claims, 0), 4)  AS denial_rate,
        RANK()        OVER (ORDER BY denied DESC)          AS denial_rank,
        SUM(revenue_at_risk)
            OVER (ORDER BY revenue_at_risk DESC
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                                                           AS cumulative_revenue_risk
    FROM specialty_stats
)
SELECT * FROM with_window ORDER BY denial_rank
"""


def specialty_analysis(claims: pd.DataFrame) -> pd.DataFrame:
    return _con(claims).execute(SPECIALTY_SQL).df()


# ── 6.  Confusion-matrix style performance table ─────────────────────────────

PERFORMANCE_SQL = """
WITH labeled AS (
    SELECT
        CASE
            WHEN predicted_denial = TRUE  AND actual_outcome = 'denied' THEN 'TP'
            WHEN predicted_denial = TRUE  AND actual_outcome = 'paid'   THEN 'FP'
            WHEN predicted_denial = FALSE AND actual_outcome = 'denied' THEN 'FN'
            WHEN predicted_denial = FALSE AND actual_outcome = 'paid'   THEN 'TN'
            ELSE 'Unknown'
        END AS classification,
        charge_amount,
        revenue_at_risk
    FROM claims
)
SELECT
    classification,
    COUNT(*)                        AS count,
    ROUND(SUM(charge_amount), 2)    AS total_charge,
    ROUND(SUM(revenue_at_risk), 2)  AS revenue_at_risk,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),
        2
    )                               AS pct_of_total
FROM labeled
GROUP BY classification
ORDER BY classification
"""


def model_performance_breakdown(claims: pd.DataFrame) -> pd.DataFrame:
    return _con(claims).execute(PERFORMANCE_SQL).df()


# ── 7.  ICD-10 denial hotspot analysis ───────────────────────────────────────

ICD10_SQL = """
WITH icd_stats AS (
    SELECT
        principal_icd10,
        diagnosis_description,
        COUNT(*)                                           AS total,
        SUM(CASE WHEN actual_outcome = 'denied' THEN 1 ELSE 0 END) AS denied,
        ROUND(AVG(predicted_denial_score), 4)             AS avg_score,
        ROUND(SUM(revenue_at_risk), 2)                    AS revenue_at_risk
    FROM claims
    GROUP BY principal_icd10, diagnosis_description
)
SELECT *,
    ROUND(denied * 1.0 / NULLIF(total, 0), 4)            AS denial_rate,
    RANK() OVER (ORDER BY denied DESC)                    AS denial_rank,
    ROUND(
        revenue_at_risk * 100.0 / SUM(revenue_at_risk) OVER (),
        2
    )                                                     AS pct_total_risk
FROM icd_stats
ORDER BY denial_rank
"""


def icd10_hotspots(claims: pd.DataFrame) -> pd.DataFrame:
    return _con(claims).execute(ICD10_SQL).df()


# ── 8.  Score bucket distribution ────────────────────────────────────────────

SCORE_BUCKET_SQL = """
WITH bucketed AS (
    SELECT
        CASE
            WHEN predicted_denial_score < 0.1  THEN '0.0–0.1'
            WHEN predicted_denial_score < 0.2  THEN '0.1–0.2'
            WHEN predicted_denial_score < 0.3  THEN '0.2–0.3'
            WHEN predicted_denial_score < 0.4  THEN '0.3–0.4'
            WHEN predicted_denial_score < 0.5  THEN '0.4–0.5'
            WHEN predicted_denial_score < 0.6  THEN '0.5–0.6'
            WHEN predicted_denial_score < 0.7  THEN '0.6–0.7'
            WHEN predicted_denial_score < 0.8  THEN '0.7–0.8'
            WHEN predicted_denial_score < 0.9  THEN '0.8–0.9'
            ELSE '0.9–1.0'
        END AS score_bucket,
        actual_outcome
    FROM claims
)
SELECT
    score_bucket,
    COUNT(*)                                                       AS total,
    SUM(CASE WHEN actual_outcome = 'denied' THEN 1 ELSE 0 END)    AS actual_denied,
    ROUND(
        SUM(CASE WHEN actual_outcome = 'denied' THEN 1 ELSE 0 END) * 1.0
        / NULLIF(COUNT(*), 0),
        4
    )                                                              AS actual_denial_rate
FROM bucketed
GROUP BY score_bucket
ORDER BY score_bucket
"""


def score_bucket_distribution(claims: pd.DataFrame) -> pd.DataFrame:
    return _con(claims).execute(SCORE_BUCKET_SQL).df()


# ── 9.  Provider-level summary with FIRST_VALUE / LAST_VALUE ─────────────────

PROVIDER_SQL = """
WITH prov AS (
    SELECT
        specialty,
        department,
        COUNT(*)                                                AS claims,
        SUM(CASE WHEN actual_outcome = 'denied' THEN 1 ELSE 0 END) AS denied,
        ROUND(AVG(provider_denial_rate), 4)                    AS avg_provider_denial_rate,
        ROUND(AVG(charge_amount), 2)                           AS avg_charge,
        ROUND(SUM(revenue_at_risk), 2)                         AS total_risk
    FROM claims
    GROUP BY specialty, department
)
SELECT *,
    ROUND(denied * 1.0 / NULLIF(claims, 0), 4)                AS denial_rate,
    FIRST_VALUE(specialty)
        OVER (PARTITION BY department ORDER BY denied DESC)    AS highest_risk_specialty_in_dept,
    SUM(total_risk)
        OVER (PARTITION BY department)                         AS dept_total_risk
FROM prov
ORDER BY department, denied DESC
"""


def provider_summary(claims: pd.DataFrame) -> pd.DataFrame:
    return _con(claims).execute(PROVIDER_SQL).df()


# ── 10.  Pre-auth gap impact ──────────────────────────────────────────────────

PREAUTH_SQL = """
SELECT
    pre_auth_gap,
    COUNT(*)                                                   AS total,
    SUM(CASE WHEN actual_outcome = 'denied' THEN 1 ELSE 0 END) AS denied,
    ROUND(
        SUM(CASE WHEN actual_outcome = 'denied' THEN 1 ELSE 0 END) * 1.0
        / NULLIF(COUNT(*), 0), 4
    )                                                          AS denial_rate,
    ROUND(AVG(charge_amount), 2)                               AS avg_charge,
    ROUND(SUM(revenue_at_risk), 2)                             AS revenue_at_risk
FROM claims
GROUP BY pre_auth_gap
ORDER BY pre_auth_gap DESC
"""


def preauth_impact(claims: pd.DataFrame) -> pd.DataFrame:
    return _con(claims).execute(PREAUTH_SQL).df()


# ── Bundle all queries ────────────────────────────────────────────────────────

def run_all(claims: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run every analytical query and return results as a dict."""
    return {
        "rolling_denial_trend":       rolling_denial_trend(claims),
        "payer_ranking":              payer_ranking(claims),
        "monthly_cohort":             monthly_cohort(claims),
        "high_risk_claims":           high_risk_claims(claims),
        "specialty_analysis":         specialty_analysis(claims),
        "model_performance":          model_performance_breakdown(claims),
        "icd10_hotspots":             icd10_hotspots(claims),
        "score_distribution":         score_bucket_distribution(claims),
        "provider_summary":           provider_summary(claims),
        "preauth_impact":             preauth_impact(claims),
    }
