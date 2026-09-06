"""
Claim Denial Prediction — Streamlit Analytics Dashboard
========================================================

Pages
-----
1.  Executive Summary          – top-level KPIs & revenue-at-risk
2.  Denial Trend Analysis      – rolling window, daily trend, MoM cohort
3.  Payer Intelligence         – per-payer rankings, risk matrix, heat-map
4.  Clinical Analysis          – ICD-10 hotspots, specialty breakdown, LOS
5.  Model Performance          – confusion matrix, score distribution, SHAP
6.  Model Registry             – version history, metrics comparison
7.  Pipeline Health            – nightly run stats, SLA tracking
8.  SQL Workbench              – live DuckDB SQL editor over the claims data

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import sys
import os
# Ensure the dashboard directory is on sys.path regardless of how the app is invoked
sys.path.insert(0, os.path.dirname(__file__))

from data_generator import (
    generate_claims,
    generate_denial_reason_stats,
    generate_model_versions,
    generate_monitoring_history,
    generate_payer_stats,
    generate_pipeline_runs,
    generate_shap_importance,
)
from sql_analytics import (
    high_risk_claims,
    icd10_hotspots,
    model_performance_breakdown,
    monthly_cohort,
    payer_ranking,
    preauth_impact,
    provider_summary,
    rolling_denial_trend,
    score_bucket_distribution,
    specialty_analysis,
    run_all,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Claim Denial Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme colours ─────────────────────────────────────────────────────────────
DANGER  = "#e74c3c"
WARNING = "#f39c12"
SUCCESS = "#27ae60"
INFO    = "#2980b9"
PURPLE  = "#8e44ad"

# ── Cached data ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    claims      = generate_claims(2000)
    monitoring  = generate_monitoring_history(90)
    models      = generate_model_versions()
    pipeline    = generate_pipeline_runs(30)
    shap        = generate_shap_importance()
    payer_stats = generate_payer_stats(claims)
    denial_rsns = generate_denial_reason_stats(claims)
    sql_results = run_all(claims)
    return claims, monitoring, models, pipeline, shap, payer_stats, denial_rsns, sql_results


claims, monitoring, models_df, pipeline, shap_df, payer_stats, denial_rsns, sql = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=60)
    st.title("Claim Denial AI")
    st.caption("Healthcare Revenue Cycle Analytics")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Executive Summary",
            "📈 Denial Trend Analysis",
            "🏦 Payer Intelligence",
            "🩺 Clinical Analysis",
            "🤖 Model Performance",
            "📦 Model Registry",
            "⚙️ Pipeline Health",
            "🔍 SQL Workbench",
        ],
    )

    st.divider()
    st.markdown("**Filters**")
    payer_filter = st.multiselect(
        "Payer", options=sorted(claims["payer"].unique()), default=[]
    )
    type_filter = st.multiselect(
        "Claim Type", options=sorted(claims["claim_type"].unique()), default=[]
    )
    score_range = st.slider("Denial Score Range", 0.0, 1.0, (0.0, 1.0), 0.01)
    date_range = st.date_input(
        "Date Range",
        value=(claims["submission_date"].min().date(), claims["submission_date"].max().date()),
    )

# ── Apply global filters ──────────────────────────────────────────────────────
filtered = claims.copy()
if payer_filter:
    filtered = filtered[filtered["payer"].isin(payer_filter)]
if type_filter:
    filtered = filtered[filtered["claim_type"].isin(type_filter)]
filtered = filtered[
    (filtered["predicted_denial_score"] >= score_range[0])
    & (filtered["predicted_denial_score"] <= score_range[1])
]
if len(date_range) == 2:
    filtered = filtered[
        (filtered["submission_date"].dt.date >= date_range[0])
        & (filtered["submission_date"].dt.date <= date_range[1])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: KPI metric card
# ─────────────────────────────────────────────────────────────────────────────
def kpi(col, label: str, value: str, delta: str | None = None, color: str = INFO):
    delta_html = f"<div style='font-size:0.75rem;color:#aaa'>{delta}</div>" if delta else ""
    card_html = (
        f'<div style="background:{color}22;border-left:4px solid {color};'
        f'padding:12px 16px;border-radius:6px;">'
        f'<div style="font-size:0.78rem;color:#888;margin-bottom:2px">{label}</div>'
        f'<div style="font-size:1.6rem;font-weight:700;color:{color}">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )
    with col:
        st.markdown(card_html, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Executive Summary
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Summary":
    st.title("🏥 Executive Summary")
    st.caption("Real-time KPIs for the Claim Denial Prediction system")

    total       = len(filtered)
    denied      = (filtered["actual_outcome"] == "denied").sum()
    pred_denied = filtered["predicted_denial"].sum()
    revenue_risk = filtered["revenue_at_risk"].sum()
    avg_score   = filtered["predicted_denial_score"].mean()
    tp = ((filtered["predicted_denial"]) & (filtered["actual_outcome"] == "denied")).sum()
    fp = ((filtered["predicted_denial"]) & (filtered["actual_outcome"] == "paid")).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Row 1 — primary KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi(c1, "Total Claims",        f"{total:,}",                    color=INFO)
    kpi(c2, "Actual Denials",      f"{denied:,}",                   f"{denied/total*100:.1f}% rate", DANGER)
    kpi(c3, "Predicted Denials",   f"{int(pred_denied):,}",         f"{pred_denied/total*100:.1f}% rate", WARNING)
    kpi(c4, "Revenue at Risk",     f"${revenue_risk:,.0f}",         "predicted denial $", DANGER)
    kpi(c5, "Avg Denial Score",    f"{avg_score:.3f}",              color=PURPLE)
    kpi(c6, "Model Precision",     f"{precision:.1%}",              "30-day rolling", SUCCESS if precision >= 0.90 else WARNING)

    st.divider()

    # Row 2 — charts
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("Daily Claims & Denial Rate (Rolling 7-day avg)")
        trend_df = rolling_denial_trend(filtered)
        if not trend_df.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=trend_df["day"], y=trend_df["total_claims"],
                                 name="Total Claims", marker_color=INFO+"88"), secondary_y=False)
            fig.add_trace(go.Scatter(x=trend_df["day"], y=trend_df["rolling_7d_denial_rate"],
                                     name="7-day Denial Rate", line=dict(color=DANGER, width=2)),
                          secondary_y=True)
            fig.update_layout(height=320, legend=dict(x=0, y=1.1, orientation="h"),
                               margin=dict(l=0, r=0, t=20, b=0))
            fig.update_yaxes(title_text="Claims", secondary_y=False)
            fig.update_yaxes(title_text="Denial Rate", tickformat=".0%", secondary_y=True)
            st.plotly_chart(fig)

    with col_r:
        st.subheader("Revenue at Risk by Payer")
        ps = generate_payer_stats(filtered).head(7)
        fig2 = px.bar(ps, x="total_revenue_at_risk", y="payer", orientation="h",
                      color="actual_denial_rate", color_continuous_scale="Reds",
                      labels={"total_revenue_at_risk": "Revenue at Risk ($)", "payer": ""},
                      height=320)
        fig2.update_layout(margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig2)

    st.divider()

    # Row 3 — denial reasons + score gauge
    col_a, col_b, col_c = st.columns([2, 2, 1])

    with col_a:
        st.subheader("Top Denial Reasons")
        dr = generate_denial_reason_stats(filtered)
        fig3 = px.bar(dr, x="count", y="denial_reason", orientation="h",
                      color="count", color_continuous_scale="Oranges",
                      labels={"count": "# Denied Claims", "denial_reason": ""},
                      height=280)
        fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig3)

    with col_b:
        st.subheader("Actual vs Predicted Denial Rate by Payer")
        ps2 = generate_payer_stats(filtered)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=ps2["payer"], y=ps2["actual_denial_rate"],
                              name="Actual", marker_color=DANGER))
        fig4.add_trace(go.Bar(x=ps2["payer"], y=ps2["predicted_denial_rate"],
                              name="Predicted", marker_color=WARNING))
        fig4.update_layout(barmode="group", height=280,
                           yaxis_tickformat=".0%",
                           margin=dict(l=0, r=0, t=10, b=0),
                           legend=dict(x=0, y=1.1, orientation="h"))
        st.plotly_chart(fig4)

    with col_c:
        st.subheader("Model Health")
        latest_prec = monitoring["rolling_precision"].iloc[-1]
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_prec * 100,
            title={"text": "Precision %"},
            gauge={
                "axis": {"range": [70, 100]},
                "bar": {"color": SUCCESS if latest_prec >= 0.90 else WARNING if latest_prec >= 0.85 else DANGER},
                "steps": [
                    {"range": [70, 80],  "color": DANGER+"33"},
                    {"range": [80, 85],  "color": WARNING+"33"},
                    {"range": [85, 100], "color": SUCCESS+"33"},
                ],
                "threshold": {"line": {"color": DANGER, "width": 2}, "thickness": 0.75, "value": 90},
            },
        ))
        gauge.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(gauge)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Denial Trend Analysis
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Denial Trend Analysis":
    st.title("📈 Denial Trend Analysis")

    # Rolling trend
    st.subheader("Rolling 7-day Denial Rate & Revenue at Risk")
    trend = rolling_denial_trend(filtered)
    if not trend.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=trend["day"], y=trend["rolling_7d_denial_rate"],
                                 name="7-day Denial Rate", fill="tozeroy",
                                 line=dict(color=DANGER)), secondary_y=False)
        fig.add_trace(go.Scatter(x=trend["day"], y=trend["rolling_7d_revenue_risk"],
                                 name="7-day Revenue Risk ($)",
                                 line=dict(color=WARNING, dash="dot")), secondary_y=True)
        fig.update_yaxes(tickformat=".0%", title_text="Denial Rate", secondary_y=False)
        fig.update_yaxes(title_text="Revenue Risk ($)", tickprefix="$", secondary_y=True)
        fig.update_layout(height=350, legend=dict(orientation="h"))
        st.plotly_chart(fig)

    st.divider()

    # Monthly cohort
    st.subheader("Monthly Cohort — Claims, Denials & MoM Change")
    cohort = monthly_cohort(filtered)
    if not cohort.empty:
        cohort["month_str"] = cohort["month"].astype(str).str[:7]
        monthly_agg = (
            cohort.groupby("month_str")
            .agg(total_claims=("claims", "sum"), total_denials=("denials", "sum"),
                 total_charge=("total_charge", "sum"), revenue_risk=("revenue_risk", "sum"))
            .reset_index()
        )
        monthly_agg["denial_rate"] = monthly_agg["total_denials"] / monthly_agg["total_claims"].replace(0, 1)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(monthly_agg, x="month_str", y=["total_claims", "total_denials"],
                         barmode="group", labels={"month_str": "Month", "value": "Count"},
                         color_discrete_map={"total_claims": INFO, "total_denials": DANGER},
                         height=300)
            fig.update_layout(legend=dict(orientation="h"), margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig)
        with c2:
            fig2 = px.line(monthly_agg, x="month_str", y="denial_rate",
                           labels={"month_str": "Month", "denial_rate": "Denial Rate"},
                           markers=True, height=300)
            fig2.update_traces(line_color=DANGER, line_width=2)
            fig2.update_yaxes(tickformat=".0%")
            fig2.update_layout(margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig2)

    st.divider()
    st.subheader("Monthly Cohort Detail Table (with MoM Change)")
    if not cohort.empty:
        show_cols = ["month_str", "payer", "claim_type", "claims", "denials",
                     "denial_rate", "total_charge", "revenue_risk",
                     "mom_claims_change", "mom_denial_change"]
        display = cohort[[c for c in show_cols if c in cohort.columns]].copy()
        st.dataframe(
            display.style.format({
                "denial_rate": "{:.1%}", "total_charge": "${:,.0f}",
                "revenue_risk": "${:,.0f}",
                "mom_claims_change": "{:+.1%}", "mom_denial_change": "{:+.1%}",
            }).background_gradient(subset=["denial_rate"], cmap="Reds"), height=400,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Payer Intelligence
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏦 Payer Intelligence":
    st.title("🏦 Payer Intelligence")

    ranked = payer_ranking(filtered)

    st.subheader("Payer Ranking — Denials, Risk & Quartile (Window Functions)")
    if not ranked.empty:
        st.dataframe(
            ranked.style.format({
                "avg_score": "{:.4f}", "revenue_at_risk": "${:,.0f}",
                "avg_charge": "${:,.2f}", "denial_pct_rank": "{:.1%}",
            })
            .background_gradient(subset=["denied"], cmap="Reds")
            .background_gradient(subset=["revenue_at_risk"], cmap="Oranges"),
        )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue at Risk vs Denial Count")
        if not ranked.empty:
            fig = px.scatter(
                ranked, x="denied", y="revenue_at_risk", size="total_claims",
                color="denial_quartile", text="payer",
                labels={"denied": "# Denied Claims", "revenue_at_risk": "Revenue at Risk ($)"},
                color_continuous_scale="RdYlGn_r", height=380,
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=True)
            st.plotly_chart(fig)

    with col2:
        st.subheader("Avg Denial Score by Payer")
        if not ranked.empty:
            fig2 = px.bar(
                ranked.sort_values("avg_score", ascending=True),
                x="avg_score", y="payer", orientation="h",
                color="avg_score", color_continuous_scale="RdYlGn_r",
                labels={"avg_score": "Avg Predicted Denial Score", "payer": ""},
                height=380,
            )
            fig2.add_vline(x=0.5, line_dash="dash", line_color=DANGER, annotation_text="Threshold 0.5")
            fig2.update_layout(margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig2)

    st.divider()
    st.subheader("Pre-Auth Gap Impact by Payer")
    preauth = preauth_impact(filtered)
    if not preauth.empty:
        preauth["pre_auth_label"] = preauth["pre_auth_gap"].map({0: "No Pre-Auth Gap", 1: "Pre-Auth Gap Detected"})
        fig3 = px.bar(preauth, x="pre_auth_label",
                      y=["denial_rate", "avg_charge"],
                      barmode="group",
                      labels={"value": "Value", "pre_auth_label": ""},
                      color_discrete_map={"denial_rate": DANGER, "avg_charge": WARNING},
                      height=300)
        fig3.update_layout(legend=dict(orientation="h"), margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig3)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Clinical Analysis
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🩺 Clinical Analysis":
    st.title("🩺 Clinical Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ICD-10 Denial Hotspots (Ranked)")
        icd = icd10_hotspots(filtered)
        if not icd.empty:
            fig = px.treemap(
                icd, path=["principal_icd10", "diagnosis_description"],
                values="denied", color="denial_rate",
                color_continuous_scale="Reds",
                labels={"denied": "# Denied", "denial_rate": "Denial Rate"},
                height=380,
            )
            fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig)

    with col2:
        st.subheader("ICD-10 Revenue at Risk (% of Total)")
        if not icd.empty:
            fig2 = px.bar(icd, x="principal_icd10", y="pct_total_risk",
                          color="denial_rate", color_continuous_scale="Oranges",
                          labels={"pct_total_risk": "% of Total Risk", "principal_icd10": "ICD-10"},
                          height=380)
            fig2.update_layout(margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig2)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Specialty Denial Analysis (Running Total)")
        spec = specialty_analysis(filtered)
        if not spec.empty:
            fig3 = px.bar(spec, x="specialty", y=["total_claims", "denied"],
                          barmode="group",
                          color_discrete_map={"total_claims": INFO, "denied": DANGER},
                          labels={"value": "Claims", "specialty": ""},
                          height=320)
            fig3.update_layout(xaxis_tickangle=-30, legend=dict(orientation="h"),
                               margin=dict(l=0, r=0, t=20, b=40))
            st.plotly_chart(fig3)

    with col4:
        st.subheader("Avg Length of Stay & Comorbidity by Specialty")
        if not spec.empty:
            fig4 = make_subplots(specs=[[{"secondary_y": True}]])
            fig4.add_trace(go.Bar(x=spec["specialty"], y=spec["avg_los"],
                                  name="Avg LOS (days)", marker_color=INFO), secondary_y=False)
            fig4.add_trace(go.Scatter(x=spec["specialty"], y=spec["avg_comorbidity"],
                                      name="Avg Comorbidity Score",
                                      line=dict(color=PURPLE, width=2), mode="lines+markers"),
                           secondary_y=True)
            fig4.update_layout(height=320, xaxis_tickangle=-30,
                               legend=dict(orientation="h"),
                               margin=dict(l=0, r=0, t=20, b=40))
            st.plotly_chart(fig4)

    st.divider()
    st.subheader("ICD-10 Full Detail Table")
    if not icd.empty:
        st.dataframe(
            icd.style.format({
                "denial_rate": "{:.1%}", "avg_score": "{:.4f}",
                "revenue_at_risk": "${:,.0f}", "pct_total_risk": "{:.1f}%",
            }).background_gradient(subset=["denial_rate"], cmap="Reds"),
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Performance
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    # ── Confusion matrix breakdown ────────────────────────────────────────
    st.subheader("Prediction Classification Breakdown")
    perf = model_performance_breakdown(filtered)
    label_map = {"TP": "True Positive", "FP": "False Positive",
                 "FN": "False Negative", "TN": "True Negative"}
    color_map = {"TP": SUCCESS, "FP": WARNING, "FN": DANGER, "TN": INFO}

    if not perf.empty:
        cols = st.columns(len(perf))
        for i, row in perf.iterrows():
            cl = str(row["classification"])
            kpi(cols[i], label_map.get(cl, cl),
                f"{int(row['count']):,}",
                f"{row['pct_of_total']:.1f}% of total",
                color_map.get(cl, INFO))

        st.markdown("")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(perf, values="count", names="classification",
                         color="classification",
                         color_discrete_map={"TP": SUCCESS, "FP": WARNING, "FN": DANGER, "TN": INFO},
                         height=320, title="Prediction Distribution")
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig)
        with c2:
            fig2 = px.bar(perf, x="classification", y="revenue_at_risk",
                          color="classification",
                          color_discrete_map={"TP": SUCCESS, "FP": WARNING, "FN": DANGER, "TN": INFO},
                          labels={"revenue_at_risk": "Revenue at Risk ($)", "classification": ""},
                          height=320, title="Revenue at Risk by Classification")
            fig2.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2)

    st.divider()

    # ── Score distribution ────────────────────────────────────────────────
    st.subheader("Denial Score Distribution vs Actual Outcome")
    score_dist = score_bucket_distribution(filtered)
    c3, c4 = st.columns(2)

    with c3:
        fig3 = px.histogram(filtered, x="predicted_denial_score", color="actual_outcome",
                            nbins=40, barmode="overlay", opacity=0.7,
                            color_discrete_map={"denied": DANGER, "paid": SUCCESS},
                            labels={"predicted_denial_score": "Denial Score", "actual_outcome": "Outcome"},
                            height=320, title="Score Histogram by Actual Outcome")
        fig3.add_vline(x=0.5, line_dash="dash", line_color="black", annotation_text="Threshold")
        fig3.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig3)

    with c4:
        if not score_dist.empty:
            fig4 = px.bar(score_dist, x="score_bucket", y="actual_denial_rate",
                          color="actual_denial_rate", color_continuous_scale="Reds",
                          labels={"score_bucket": "Score Bucket", "actual_denial_rate": "Actual Denial Rate"},
                          height=320, title="Actual Denial Rate per Score Bucket (Calibration)")
            fig4.update_yaxes(tickformat=".0%")
            fig4.update_layout(margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig4)

    st.divider()

    # ── SHAP importance ───────────────────────────────────────────────────
    st.subheader("Top-20 Feature Importances (Mean |SHAP|)")
    fig5 = px.bar(shap_df.head(20), x="mean_abs_shap", y="feature",
                  orientation="h", color="mean_abs_shap",
                  color_continuous_scale="Blues",
                  labels={"mean_abs_shap": "Mean |SHAP|", "feature": ""},
                  height=500)
    fig5.update_layout(margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig5)

    st.divider()

    # ── Rolling precision over time ───────────────────────────────────────
    st.subheader("30-Day Rolling Precision History")
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=monitoring["date"], y=monitoring["rolling_precision"],
                              mode="lines+markers", name="Rolling Precision",
                              line=dict(color=INFO, width=2),
                              marker=dict(color=monitoring["alert"].apply(
                                  lambda a: DANGER if a == "urgent" else WARNING if a == "degradation" else SUCCESS
                              ))))
    fig6.add_hline(y=0.90, line_dash="dash", line_color=SUCCESS,  annotation_text="Target 0.90")
    fig6.add_hline(y=0.85, line_dash="dot",  line_color=WARNING, annotation_text="Degradation 0.85")
    fig6.add_hline(y=0.80, line_dash="dot",  line_color=DANGER,  annotation_text="Urgent 0.80")
    fig6.update_yaxes(tickformat=".1%", range=[0.7, 1.0])
    fig6.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig6)

    # ── Monitoring stats table ────────────────────────────────────────────
    st.subheader("Monitoring History")
    st.dataframe(
        monitoring.style.format({
            "rolling_precision": "{:.1%}",
        }).map(
            lambda v: f"background-color: {DANGER}33" if isinstance(v, str) and v == "urgent"
                      else f"background-color: {WARNING}33" if isinstance(v, str) and v == "degradation"
                      else "",
            subset=["alert"],
        ),
        height=300,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Model Registry
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📦 Model Registry":
    st.title("📦 Model Registry")

    stage_colors = {"Production": SUCCESS, "Archived": "#888", "Staging": WARNING, "Failed": DANGER}

    st.subheader("Registered Model Versions")
    for _, row in models_df.iterrows():
        color = stage_colors.get(str(row["stage"]), INFO)
        with st.expander(f"**{row['version_id']}** — {row['stage']}  |  {row['algorithm']}  |  trained {row['training_date']}"):
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            kpi(c1, "Precision",  f"{row['precision']:.4f}",  color=SUCCESS if row['precision'] >= 0.90 else DANGER)
            kpi(c2, "Recall",     f"{row['recall']:.4f}",     color=INFO)
            kpi(c3, "F1 Score",   f"{row['f1']:.4f}",         color=INFO)
            kpi(c4, "AUC-ROC",    f"{row['auc_roc']:.4f}",    color=INFO)
            kpi(c5, "AUC-PR",     f"{row['auc_pr']:.4f}",     color=INFO)
            kpi(c6, "ECE",        f"{row['ece']:.4f}",        color=SUCCESS if row['ece'] <= 0.05 else DANGER)
            st.caption(f"Train records: {row['train_records']:,}  |  Denied class: {row['denied_class_pct']:.1%}")

    st.divider()
    st.subheader("Metrics Comparison Across Versions")
    metrics_to_plot = ["precision", "recall", "f1", "auc_roc", "auc_pr", "ece"]
    melt = models_df.melt(id_vars="version_id", value_vars=metrics_to_plot,
                          var_name="metric", value_name="value")
    fig = px.bar(melt, x="metric", y="value", color="version_id", barmode="group",
                 height=380, labels={"value": "Score", "metric": "Metric", "version_id": "Version"})
    fig.add_hline(y=0.90, line_dash="dash", line_color=SUCCESS, annotation_text="Precision target 0.90")
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h"))
    st.plotly_chart(fig)

    st.divider()
    st.subheader("Full Registry Table")
    st.dataframe(
        models_df.style.format({
            "precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}",
            "auc_roc": "{:.4f}", "auc_pr": "{:.4f}", "ece": "{:.4f}",
            "train_records": "{:,}", "denied_class_pct": "{:.1%}",
        }).map(
            lambda v: f"color: {SUCCESS}; font-weight: bold" if v == "Production" else "",
            subset=["stage"],
        ),
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Pipeline Health
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Pipeline Health":
    st.title("⚙️ Pipeline Health")

    last = pipeline.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Last Run Claims",   f"{int(last['total_active_claims']):,}", color=INFO)
    kpi(c2, "Successfully Scored", f"{int(last['total_scored']):,}", color=SUCCESS)
    kpi(c3, "Degraded Quality",  f"{int(last['degraded_count']):,}", color=WARNING)
    kpi(c4, "Failed Scoring",    f"{int(last['failed_count']):,}", color=DANGER)
    kpi(c5, "Wall Clock",        f"{last['wall_clock_minutes']:.1f} min",
        "SLA: 240 min", DANGER if last["sla_breach"] else SUCCESS)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Nightly Pipeline Duration (minutes)")
        fig = px.area(pipeline, x="run_date", y="wall_clock_minutes",
                      color_discrete_sequence=[INFO],
                      labels={"run_date": "Date", "wall_clock_minutes": "Minutes"},
                      height=320)
        fig.add_hline(y=240, line_dash="dash", line_color=DANGER,
                      annotation_text="4-hour SLA (240 min)")
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig)

    with col2:
        st.subheader("Claims Scored vs Failed per Run")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=pipeline["run_date"], y=pipeline["total_scored"],
                              name="Scored", marker_color=SUCCESS))
        fig2.add_trace(go.Bar(x=pipeline["run_date"], y=pipeline["failed_count"],
                              name="Failed", marker_color=DANGER))
        fig2.update_layout(barmode="stack", height=320,
                           legend=dict(orientation="h"),
                           margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig2)

    st.divider()
    st.subheader("Pipeline Run History")
    # Format sla_breach as string before styling to avoid bool/format conflicts
    pipeline_display = pipeline.copy()
    pipeline_display["sla_breach"] = pipeline_display["sla_breach"].map({True: "⚠️ YES", False: "✅ NO"})
    st.dataframe(
        pipeline_display.style.format({
            "wall_clock_minutes": "{:.1f}",
        }).map(
            lambda v: f"background-color: {DANGER}33" if v == "⚠️ YES" else "",
            subset=["sla_breach"],
        ),
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 8 — SQL Workbench
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 SQL Workbench":
    st.title("🔍 SQL Workbench")
    st.caption("Run DuckDB SQL directly against the claims table. All data is synthetic (no real PHI).")

    import duckdb as _duckdb

    st.info("**Available table:** `claims`  — columns: " +
            ", ".join(f"`{c}`" for c in claims.columns[:12]) + " … and more")

    preset_queries = {
        "— select a preset —": "",
        "Rolling 7-day denial rate (window fn)": """
SELECT
    CAST(submission_date AS DATE) AS day,
    COUNT(*) AS total,
    SUM(CASE WHEN actual_outcome='denied' THEN 1 ELSE 0 END) AS denied,
    ROUND(
        AVG(SUM(CASE WHEN actual_outcome='denied' THEN 1 ELSE 0 END)*1.0/COUNT(*))
        OVER (ORDER BY CAST(submission_date AS DATE) ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
    4) AS rolling_7d_rate
FROM claims
GROUP BY 1
ORDER BY 1
""",
        "Top 5 payers by revenue at risk": """
SELECT payer,
       COUNT(*) AS claims,
       SUM(revenue_at_risk) AS revenue_at_risk,
       ROUND(SUM(CASE WHEN actual_outcome='denied' THEN 1 ELSE 0 END)*1.0/COUNT(*), 3) AS denial_rate
FROM claims
GROUP BY payer
ORDER BY revenue_at_risk DESC
LIMIT 5
""",
        "Payer RANK + NTILE quartile": """
SELECT payer,
       SUM(CASE WHEN actual_outcome='denied' THEN 1 ELSE 0 END) AS denied,
       RANK()   OVER (ORDER BY SUM(CASE WHEN actual_outcome='denied' THEN 1 ELSE 0 END) DESC) AS denial_rank,
       NTILE(4) OVER (ORDER BY SUM(CASE WHEN actual_outcome='denied' THEN 1 ELSE 0 END) DESC) AS quartile
FROM claims
GROUP BY payer
ORDER BY denial_rank
""",
        "Monthly MoM denial change (LAG)": """
WITH m AS (
    SELECT DATE_TRUNC('month', submission_date) AS month,
           COUNT(*) AS claims,
           SUM(CASE WHEN actual_outcome='denied' THEN 1 ELSE 0 END) AS denied
    FROM claims GROUP BY 1
)
SELECT month,
       claims,
       denied,
       ROUND(denied*1.0/claims, 3) AS denial_rate,
       LAG(denied) OVER (ORDER BY month) AS prev_denied,
       ROUND((denied - LAG(denied) OVER (ORDER BY month))*1.0 / NULLIF(LAG(denied) OVER (ORDER BY month),0), 3) AS mom_change
FROM m ORDER BY month
""",
        "Score bucket calibration": """
SELECT
    FLOOR(predicted_denial_score * 10)/10 AS score_bin,
    COUNT(*) AS claims,
    SUM(CASE WHEN actual_outcome='denied' THEN 1 ELSE 0 END) AS actual_denied,
    ROUND(SUM(CASE WHEN actual_outcome='denied' THEN 1 ELSE 0 END)*1.0/COUNT(*), 4) AS actual_denial_rate
FROM claims
GROUP BY 1
ORDER BY 1
""",
        "High-risk claims (PERCENT_RANK + NTILE)": """
SELECT claim_id, payer, specialty, charge_amount, predicted_denial_score,
       ROUND(PERCENT_RANK() OVER (ORDER BY predicted_denial_score), 4) AS pct_rank,
       NTILE(10) OVER (ORDER BY predicted_denial_score DESC) AS risk_decile
FROM claims
WHERE NTILE(10) OVER (ORDER BY predicted_denial_score DESC) = 1
ORDER BY predicted_denial_score DESC
LIMIT 50
""",
        "Cumulative revenue at risk by specialty": """
SELECT specialty,
       ROUND(SUM(revenue_at_risk), 2) AS specialty_risk,
       ROUND(SUM(SUM(revenue_at_risk)) OVER (ORDER BY SUM(revenue_at_risk) DESC
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS cumulative_risk
FROM claims GROUP BY specialty ORDER BY specialty_risk DESC
""",
        "ICD-10 hotspots with FIRST_VALUE": """
SELECT
    principal_icd10,
    diagnosis_description,
    COUNT(*) AS total,
    SUM(CASE WHEN actual_outcome='denied' THEN 1 ELSE 0 END) AS denied,
    ROUND(SUM(CASE WHEN actual_outcome='denied' THEN 1 ELSE 0 END)*1.0/COUNT(*), 3) AS denial_rate,
    FIRST_VALUE(payer) OVER (PARTITION BY principal_icd10
                             ORDER BY SUM(CASE WHEN actual_outcome='denied' THEN 1 ELSE 0 END) DESC) AS top_payer
FROM claims
GROUP BY principal_icd10, diagnosis_description
ORDER BY denied DESC
""",
    }

    preset = st.selectbox("Preset Queries", list(preset_queries.keys()))
    default_sql = preset_queries[preset] if preset != "— select a preset —" else (
        "SELECT payer, COUNT(*) AS claims, "
        "ROUND(SUM(CASE WHEN actual_outcome='denied' THEN 1 ELSE 0 END)*1.0/COUNT(*),3) AS denial_rate "
        "FROM claims GROUP BY payer ORDER BY denial_rate DESC"
    )

    user_sql = st.text_area("SQL Query", value=default_sql.strip(), height=180)

    if st.button("▶ Run Query", type="primary"):
        try:
            con = _duckdb.connect(":memory:")
            con.register("claims", filtered)
            result = con.execute(user_sql).df()
            st.success(f"✅  {len(result):,} rows returned")
            st.dataframe(result, height=400)
            csv = result.to_csv(index=False).encode()
            st.download_button("⬇ Download CSV", csv, "query_result.csv", "text/csv")
        except Exception as e:
            st.error(f"❌ Query error: {e}")

    st.divider()
    with st.expander("📖 Schema Reference"):
        schema_df = pd.DataFrame({
            "Column": claims.columns,
            "DType": [str(claims[c].dtype) for c in claims.columns],
            "Example": [str(claims[c].iloc[0]) for c in claims.columns],
        })
        st.dataframe(schema_df)



