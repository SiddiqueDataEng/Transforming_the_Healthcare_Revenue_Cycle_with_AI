"""
SQL-based analytics engine for the Claim Denial Prediction dashboard.

Uses DuckDB (in-memory) to execute ANSI SQL with window functions,
CTEs, and analytical queries against the synthetic claim DataFrame.
All queries are exposed as named functions for direct use in Streamlit.
"""

from __future__ import annotations

from typing import Any, Optional

import duckdb
import pandas as pd


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def _con(df: pd.DataFrame, table: str = "claims") -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection with *df* registered as *table*."""
    con = duckdb.connect(database=":memory:")
    con.register(table, df)
    return con


# ===========================================================================
# 1. KPI SUMMARY
# ===========================================================================

def kpi_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return a flat dict of top-level KPIs computed via SQL.

    Metrics
    -------
    total_claims, total_denied, total_paid, total_pending,
    overall_denial_rate, avg_denial_score, high_risk_count (score≥0.5),
    avg_charge_amount, total_charge_amount, degraded_scores_pct
    """
    con = _con(df)
    row = con.execute("""
        SELECT
            COUNT(*)                                                   AS total_claims,
            SUM(CASE WHEN adjudication_outcome = 'denied' THEN 1 ELSE 0 END) AS total_denied,
            SUM(CASE WHEN adjudication_outcome = 'paid'   THEN 1 ELSE 0 END) AS total_paid,
            SUM(CASE WHEN adjudication_outcome = 'pending' THEN 1 ELSE 0 END) AS total_pending,
            ROUND(
                SUM(CASE WHEN adjudication_outcome = 'denied' THEN 1 ELSE 0 END) * 1.0
                / NULLIF(SUM(CASE WHEN adjudication_outcome IN ('denied','paid') THEN 1 ELSE 0 END),0)
            , 4)                                                       AS overall_denial_rate,
            ROUND(AVG(predicted_denial_score), 4)                     AS avg_denial_score,
            SUM(CASE WHEN predicted_denial_score >= 0.5 THEN 1 ELSE 0 END) AS high_risk_count,
            ROUND(AVG(charge_amount), 2)                              AS avg_charge_amount,
            ROUND(SUM(charge_amount), 2)                              AS total_charge_amount,
            ROUND(
                SUM(CASE WHEN score_quality_flag = 'degraded' THEN 1 ELSE 0 END) * 100.0
                / COUNT(*), 2)                                         AS degraded_scores_pct
        FROM claims
    """).fetchone()

    keys = [
        "total_claims", "total_denied", "total_paid", "total_pending",
        "overall_denial_rate", "avg_denial_score", "high_risk_count",
        "avg_charge_amount", "total_charge_amount", "degraded_scores_pct",
    ]
    return dict(zip(keys, row))


# ===========================================================================
# 2. DAILY TREND WITH ROLLING AVERAGES (window functions)
# ===========================================================================

def daily_denial_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily claim volume, denial count, denial rate, and 7-day / 30-day
    rolling averages using SQL window functions.
    """
    con = _con(df)
    return con.execute("""
        WITH daily AS (
            SELECT
                CAST(submission_date AS DATE)   AS dt,
                COUNT(*)                         AS total_claims,
                SUM(CASE WHEN adjudication_outcome = 'denied' THEN 1 ELSE 0 END) AS denied,
                SUM(CASE WHEN adjudication_outcome = 'paid'   THEN 1 ELSE 0 END) AS paid,
                SUM(charge_amount)               AS total_charges
            FROM claims
            GROUP BY 1
        )
        SELECT
            dt,
            total_claims,
            denied,
            paid,
            ROUND(denied * 1.0 / NULLIF(denied + paid, 0), 4)         AS denial_rate,
            ROUND(AVG(denied * 1.0 / NULLIF(denied + paid, 0))
                  OVER (ORDER BY dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 4)
                                                                        AS rolling_7d_denial_rate,
            ROUND(AVG(denied * 1.0 / NULLIF(denied + paid, 0))
                  OVER (ORDER BY dt ROWS BETWEEN 29 PRECEDING AND CURRENT ROW), 4)
                                                                        AS rolling_30d_denial_rate,
            ROUND(total_charges, 2)                                     AS total_charges,
            ROUND(SUM(total_charges) OVER (ORDER BY dt), 2)            AS cumulative_charges
        FROM daily
        ORDER BY dt
    """).df()


# ===========================================================================
# 3. DENIAL RATE BY PAYER (with rank)
# ===========================================================================

def denial_by_payer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Denial rate, claim volume, and revenue at risk per payer,
    ranked by denial rate descending.
    """
    con = _con(df)
    return con.execute("""
        WITH payer_stats AS (
            SELECT
                payer_name,
                COUNT(*)                                                      AS total_claims,
                SUM(CASE WHEN adjudication_outcome = 'denied' THEN 1 ELSE 0 END) AS denied,
                SUM(CASE WHEN adjudication_outcome = 'paid'   THEN 1 ELSE 0 END) AS paid,
                ROUND(AVG(predicted_denial_score), 4)                         AS avg_risk_score,
                ROUND(SUM(CASE WHEN adjudication_outcome = 'denied'
                               THEN charge_amount ELSE 0 END), 2)             AS denied_charges,
                ROUND(SUM(charge_amount), 2)                                  AS total_charges
            FROM claims
            GROUP BY payer_name
        )
        SELECT
            payer_name,
            total_claims,
            denied,
            paid,
            ROUND(denied * 1.0 / NULLIF(denied + paid, 0), 4)        AS denial_rate,
            avg_risk_score,
            denied_charges,
            total_charges,
            ROUND(denied_charges * 100.0 / NULLIF(total_charges, 0), 2) AS denied_charges_pct,
            RANK() OVER (ORDER BY denied * 1.0 / NULLIF(denied + paid, 0) DESC) AS denial_rank
        FROM payer_stats
        ORDER BY denial_rate DESC
    """).df()


# ===========================================================================
# 4. DENIAL RATE BY PROVIDER SPECIALTY
# ===========================================================================

def denial_by_specialty(df: pd.DataFrame) -> pd.DataFrame:
    """Denial rate and revenue at risk by provider specialty."""
    con = _con(df)
    return con.execute("""
        SELECT
            provider_specialty,
            COUNT(*)                                                      AS total_claims,
            SUM(CASE WHEN adjudication_outcome = 'denied' THEN 1 ELSE 0 END) AS denied,
            ROUND(SUM(CASE WHEN adjudication_outcome = 'denied' THEN 1 ELSE 0 END)
                  * 1.0 / NULLIF(COUNT(*), 0), 4)                        AS denial_rate,
            ROUND(AVG(predicted_denial_score), 4)                        AS avg_risk_score,
            ROUND(SUM(CASE WHEN adjudication_outcome = 'denied'
                           THEN charge_amount ELSE 0 END), 2)            AS denied_charges,
            RANK() OVER (ORDER BY
                SUM(CASE WHEN adjudication_outcome='denied' THEN 1 ELSE 0 END)
                * 1.0 / NULLIF(COUNT(*), 0) DESC)                        AS rank
        FROM claims
        GROUP BY provider_specialty
        ORDER BY denial_rate DESC
    """).df()


# ===========================================================================
# 5. DENIAL REASONS BREAKDOWN
# ===========================================================================

def denial_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency and charge impact of each denial reason."""
    con = _con(df)
    return con.execute("""
        SELECT
            COALESCE(denial_reason, 'N/A (Paid/Pending)')  AS denial_reason,
            COUNT(*)                                         AS count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total,
            ROUND(SUM(charge_amount), 2)                     AS total_charges,
            ROUND(AVG(charge_amount), 2)                     AS avg_charge,
            ROUND(AVG(predicted_denial_score), 4)            AS avg_risk_score
        FROM claims
        WHERE denial_reason IS NOT NULL
        GROUP BY denial_reason
        ORDER BY count DESC
    """).df()


# ===========================================================================
# 6. RISK SCORE DISTRIBUTION (histogram buckets)
# ===========================================================================

def risk_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """10 equal-width buckets of predicted_denial_score with outcome breakdown."""
    con = _con(df)
    return con.execute("""
        SELECT
            ROUND(FLOOR(predicted_denial_score * 10) / 10, 1)  AS score_bucket,
            COUNT(*)                                             AS total,
            SUM(CASE WHEN adjudication_outcome = 'denied' THEN 1 ELSE 0 END) AS actual_denied,
            SUM(CASE WHEN adjudication_outcome = 'paid'   THEN 1 ELSE 0 END) AS actual_paid,
            SUM(CASE WHEN adjudication_outcome = 'pending' THEN 1 ELSE 0 END) AS pending,
            ROUND(AVG(charge_amount), 2)                        AS avg_charge
        FROM claims
        GROUP BY score_bucket
        ORDER BY score_bucket
    """).df()


# ===========================================================================
# 7. MONTH-OVER-MONTH ANALYSIS (window functions)
# ===========================================================================

def monthly_mom_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly denial rate, volume, charges, and month-over-month deltas
    using LAG window function.
    """
    con = _con(df)
    return con.execute("""
        WITH monthly AS (
            SELECT
                DATE_TRUNC('month', submission_date)::DATE AS month,
                COUNT(*)                                    AS total_claims,
                SUM(CASE WHEN adjudication_outcome='denied' THEN 1 ELSE 0 END) AS denied,
                SUM(CASE WHEN adjudication_outcome='paid'   THEN 1 ELSE 0 END) AS paid,
                ROUND(SUM(charge_amount), 2)                AS total_charges
            FROM claims
            GROUP BY 1
        )
        SELECT
            month,
            total_claims,
            denied,
            paid,
            ROUND(denied * 1.0 / NULLIF(denied + paid, 0), 4) AS denial_rate,
            total_charges,
            LAG(denied * 1.0 / NULLIF(denied + paid, 0))
                OVER (ORDER BY month)                          AS prev_denial_rate,
            ROUND(
                (denied * 1.0 / NULLIF(denied + paid, 0))
                - LAG(denied * 1.0 / NULLIF(denied + paid, 0)) OVER (ORDER BY month)
            , 4)                                               AS denial_rate_mom_delta,
            LAG(total_claims) OVER (ORDER BY month)            AS prev_total_claims,
            ROUND(
                (total_claims - LAG(total_claims) OVER (ORDER BY month)) * 100.0
                / NULLIF(LAG(total_claims) OVER (ORDER BY month), 0), 2
            )                                                   AS claims_volume_pct_change
        FROM monthly
        ORDER BY month
    """).df()


# ===========================================================================
# 8. PROVIDER-LEVEL PERFORMANCE (window + percentile)
# ===========================================================================

def provider_performance(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Per-provider NPI denial rate, volume, and percentile rank.
    Returns the top *top_n* highest-risk providers.
    """
    con = _con(df)
    return con.execute(f"""
        WITH prov AS (
            SELECT
                provider_npi,
                provider_specialty,
                provider_state,
                COUNT(*)                                               AS total_claims,
                SUM(CASE WHEN adjudication_outcome='denied' THEN 1 ELSE 0 END) AS denied,
                ROUND(AVG(predicted_denial_score), 4)                  AS avg_risk_score,
                ROUND(SUM(charge_amount), 2)                           AS total_charges,
                ROUND(SUM(CASE WHEN adjudication_outcome='denied'
                               THEN charge_amount ELSE 0 END), 2)      AS denied_charges
            FROM claims
            GROUP BY provider_npi, provider_specialty, provider_state
            HAVING COUNT(*) >= 3
        )
        SELECT
            provider_npi,
            provider_specialty,
            provider_state,
            total_claims,
            denied,
            ROUND(denied * 1.0 / NULLIF(total_claims, 0), 4)          AS denial_rate,
            avg_risk_score,
            total_charges,
            denied_charges,
            ROUND(
                PERCENT_RANK() OVER (ORDER BY denied * 1.0 / NULLIF(total_claims, 0))
            , 4)                                                        AS denial_rate_percentile
        FROM prov
        ORDER BY denial_rate DESC
        LIMIT {top_n}
    """).df()


# ===========================================================================
# 9. PAYER × CLAIM-TYPE CROSS-TAB (pivot-style)
# ===========================================================================

def payer_claimtype_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """
    Denial rate cross-tabulation: payer (rows) × claim type (columns).
    """
    con = _con(df)
    return con.execute("""
        SELECT
            payer_name,
            ROUND(AVG(CASE WHEN claim_type='professional' AND adjudication_outcome IN ('denied','paid')
                           THEN CASE WHEN adjudication_outcome='denied' THEN 1.0 ELSE 0.0 END
                      END), 4)  AS professional_denial_rate,
            ROUND(AVG(CASE WHEN claim_type='institutional' AND adjudication_outcome IN ('denied','paid')
                           THEN CASE WHEN adjudication_outcome='denied' THEN 1.0 ELSE 0.0 END
                      END), 4)  AS institutional_denial_rate,
            COUNT(*)             AS total_claims,
            SUM(CASE WHEN adjudication_outcome='denied' THEN 1 ELSE 0 END) AS total_denied
        FROM claims
        GROUP BY payer_name
        ORDER BY total_denied DESC
    """).df()


# ===========================================================================
# 10. HIGH-RISK CLAIMS TABLE (actionable worklist)
# ===========================================================================

def high_risk_claims(df: pd.DataFrame, threshold: float = 0.70, limit: int = 100) -> pd.DataFrame:
    """
    Claims with predicted_denial_score >= *threshold* that are still Active/Pending,
    ordered by score descending — the billing staff worklist.
    """
    con = _con(df)
    return con.execute(f"""
        SELECT
            claim_id,
            submission_date,
            payer_name,
            provider_specialty,
            provider_state,
            claim_type,
            admission_type,
            patient_age,
            comorbidity_risk_score,
            length_of_stay,
            ROUND(charge_amount, 2)             AS charge_amount,
            ROUND(predicted_denial_score, 4)    AS denial_score,
            documents_medical_necessity,
            mentions_lack_of_pre_auth,
            score_quality_flag,
            adjudication_outcome
        FROM claims
        WHERE predicted_denial_score >= {threshold}
          AND adjudication_outcome = 'pending'
        ORDER BY predicted_denial_score DESC
        LIMIT {limit}
    """).df()


# ===========================================================================
# 11. CALIBRATION TABLE (ECE bins)
# ===========================================================================

def calibration_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    10-bin calibration table: mean predicted score vs actual denial fraction.
    Used to compute Expected Calibration Error (ECE).
    """
    con = _con(df)
    return con.execute("""
        WITH binned AS (
            SELECT
                FLOOR(predicted_denial_score * 10) / 10.0  AS bin_lower,
                predicted_denial_score,
                CASE WHEN adjudication_outcome = 'denied' THEN 1.0 ELSE 0.0 END AS actual
            FROM claims
            WHERE adjudication_outcome IN ('denied', 'paid')
        )
        SELECT
            bin_lower,
            ROUND(bin_lower + 0.1, 1)               AS bin_upper,
            COUNT(*)                                  AS n_samples,
            ROUND(AVG(predicted_denial_score), 4)    AS mean_predicted,
            ROUND(AVG(actual), 4)                    AS actual_denial_fraction,
            ROUND(ABS(AVG(predicted_denial_score) - AVG(actual)), 4) AS calibration_gap,
            ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (), 4)         AS bin_weight
        FROM binned
        GROUP BY bin_lower
        ORDER BY bin_lower
    """).df()


# ===========================================================================
# 12. ADMISSION TYPE ANALYSIS
# ===========================================================================

def admission_type_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Denial metrics grouped by admission type."""
    con = _con(df)
    return con.execute("""
        SELECT
            admission_type,
            COUNT(*)                                                       AS total_claims,
            SUM(CASE WHEN adjudication_outcome='denied' THEN 1 ELSE 0 END) AS denied,
            ROUND(SUM(CASE WHEN adjudication_outcome='denied' THEN 1 ELSE 0 END)
                  * 1.0 / NULLIF(COUNT(*),0), 4)                           AS denial_rate,
            ROUND(AVG(charge_amount), 2)                                   AS avg_charge,
            ROUND(AVG(predicted_denial_score), 4)                          AS avg_risk_score,
            ROUND(AVG(length_of_stay), 2)                                  AS avg_los,
            ROUND(AVG(comorbidity_risk_score), 2)                          AS avg_comorbidity
        FROM claims
        GROUP BY admission_type
        ORDER BY denial_rate DESC
    """).df()


# ===========================================================================
# 13. CUMULATIVE REVENUE AT RISK  (running total window)
# ===========================================================================

def cumulative_revenue_at_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily cumulative denied charges (revenue at risk) using a running SUM window.
    """
    con = _con(df)
    return con.execute("""
        WITH daily AS (
            SELECT
                CAST(submission_date AS DATE)     AS dt,
                SUM(CASE WHEN adjudication_outcome='denied'
                         THEN charge_amount ELSE 0 END) AS daily_denied_charges,
                SUM(charge_amount)                AS daily_total_charges
            FROM claims
            GROUP BY 1
        )
        SELECT
            dt,
            ROUND(daily_denied_charges, 2)         AS daily_denied_charges,
            ROUND(daily_total_charges, 2)           AS daily_total_charges,
            ROUND(SUM(daily_denied_charges)
                  OVER (ORDER BY dt), 2)            AS cumulative_denied_charges,
            ROUND(SUM(daily_total_charges)
                  OVER (ORDER BY dt), 2)            AS cumulative_total_charges
        FROM daily
        ORDER BY dt
    """).df()


# ===========================================================================
# 14. STATE-LEVEL HEATMAP DATA
# ===========================================================================

def state_denial_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Denial rate and volume by provider state, for geographic analysis."""
    con = _con(df)
    return con.execute("""
        SELECT
            provider_state                                                 AS state,
            COUNT(*)                                                       AS total_claims,
            SUM(CASE WHEN adjudication_outcome='denied' THEN 1 ELSE 0 END) AS denied,
            ROUND(SUM(CASE WHEN adjudication_outcome='denied' THEN 1 ELSE 0 END)
                  * 1.0 / NULLIF(COUNT(*), 0), 4)                         AS denial_rate,
            ROUND(SUM(charge_amount), 2)                                   AS total_charges,
            ROUND(SUM(CASE WHEN adjudication_outcome='denied'
                           THEN charge_amount ELSE 0 END), 2)              AS denied_charges
        FROM claims
        GROUP BY provider_state
        ORDER BY denial_rate DESC
    """).df()


# ===========================================================================
# 15. NLP FEATURE IMPACT
# ===========================================================================

def nlp_feature_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Denial rate split by pre-auth gap flag and medical necessity score quartile.
    """
    con = _con(df)
    return con.execute("""
        WITH scored AS (
            SELECT *,
                NTILE(4) OVER (ORDER BY documents_medical_necessity) AS necessity_quartile
            FROM claims
        )
        SELECT
            necessity_quartile,
            mentions_lack_of_pre_auth,
            COUNT(*)                                                       AS total,
            SUM(CASE WHEN adjudication_outcome='denied' THEN 1 ELSE 0 END) AS denied,
            ROUND(SUM(CASE WHEN adjudication_outcome='denied' THEN 1 ELSE 0 END)
                  * 1.0 / NULLIF(COUNT(*), 0), 4)                         AS denial_rate,
            ROUND(AVG(documents_medical_necessity), 4)                    AS avg_necessity_score,
            ROUND(AVG(predicted_denial_score), 4)                         AS avg_risk_score
        FROM scored
        WHERE adjudication_outcome IN ('denied','paid')
        GROUP BY necessity_quartile, mentions_lack_of_pre_auth
        ORDER BY necessity_quartile, mentions_lack_of_pre_auth
    """).df()


# ===========================================================================
# 16. CONFUSION MATRIX (at 0.5 threshold)
# ===========================================================================

def confusion_matrix_stats(df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """
    Return TP, FP, TN, FN, precision, recall, F1, and accuracy.
    """
    con = _con(df)
    row = con.execute(f"""
        WITH scored AS (
            SELECT
                CASE WHEN predicted_denial_score >= {threshold} THEN 1 ELSE 0 END AS pred_denial,
                CASE WHEN adjudication_outcome = 'denied' THEN 1 ELSE 0 END        AS actual_denial
            FROM claims
            WHERE adjudication_outcome IN ('denied', 'paid')
        )
        SELECT
            SUM(CASE WHEN pred_denial=1 AND actual_denial=1 THEN 1 ELSE 0 END) AS tp,
            SUM(CASE WHEN pred_denial=1 AND actual_denial=0 THEN 1 ELSE 0 END) AS fp,
            SUM(CASE WHEN pred_denial=0 AND actual_denial=0 THEN 1 ELSE 0 END) AS tn,
            SUM(CASE WHEN pred_denial=0 AND actual_denial=1 THEN 1 ELSE 0 END) AS fn,
            COUNT(*) AS total
        FROM scored
    """).fetchone()
    tp, fp, tn, fn, total = row
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy  = (tp + tn) / max(total, 1)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn,
                precision=round(precision, 4), recall=round(recall, 4),
                f1=round(f1, 4), accuracy=round(accuracy, 4), total=total)


# ===========================================================================
# 17. COHORT ANALYSIS — first vs repeat payers
# ===========================================================================

def payer_cohort_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split claims into first-appearance vs repeat payer interactions
    using ROW_NUMBER window function, and compare denial rates.
    """
    con = _con(df)
    return con.execute("""
        WITH ordered AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY payer_name ORDER BY submission_date) AS claim_seq
            FROM claims
        ),
        cohort AS (
            SELECT *,
                CASE WHEN claim_seq = 1 THEN 'First Claim' ELSE 'Repeat Claim' END AS cohort
            FROM ordered
        )
        SELECT
            cohort,
            COUNT(*)                                                       AS total,
            SUM(CASE WHEN adjudication_outcome='denied' THEN 1 ELSE 0 END) AS denied,
            ROUND(SUM(CASE WHEN adjudication_outcome='denied' THEN 1 ELSE 0 END)
                  * 1.0 / NULLIF(COUNT(*), 0), 4)                         AS denial_rate,
            ROUND(AVG(charge_amount), 2)                                   AS avg_charge
        FROM cohort
        WHERE adjudication_outcome IN ('denied','paid')
        GROUP BY cohort
    """).df()


# ===========================================================================
# 18. COMORBIDITY vs DENIAL RATE (binned)
# ===========================================================================

def comorbidity_denial_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Denial rate grouped by Charlson Comorbidity Index score bucket.
    """
    con = _con(df)
    return con.execute("""
        SELECT
            CASE
                WHEN comorbidity_risk_score = 0         THEN '0 (None)'
                WHEN comorbidity_risk_score BETWEEN 1 AND 2 THEN '1-2 (Low)'
                WHEN comorbidity_risk_score BETWEEN 3 AND 5 THEN '3-5 (Moderate)'
                ELSE '6+ (High)'
            END                                                            AS comorbidity_group,
            COUNT(*)                                                       AS total_claims,
            SUM(CASE WHEN adjudication_outcome='denied' THEN 1 ELSE 0 END) AS denied,
            ROUND(SUM(CASE WHEN adjudication_outcome='denied' THEN 1 ELSE 0 END)
                  * 1.0 / NULLIF(COUNT(*),0), 4)                          AS denial_rate,
            ROUND(AVG(charge_amount), 2)                                   AS avg_charge,
            ROUND(AVG(predicted_denial_score), 4)                          AS avg_risk_score
        FROM claims
        WHERE adjudication_outcome IN ('denied','paid')
        GROUP BY comorbidity_group
        ORDER BY MIN(comorbidity_risk_score)
    """).df()
