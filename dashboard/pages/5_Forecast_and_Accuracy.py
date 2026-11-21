import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from src.config import BACKTEST_RESULTS_FILE, MODEL_COMPARISON_FILE, OUTPUTS_DIR
from src.dashboard_shared import render_sidebar_filters

st.set_page_config(page_title="Forecast & Accuracy", page_icon="🎯", layout="wide")
st.title("🎯 Forecast Accuracy & Model Evaluation")
st.markdown("Rolling-origin backtest validation metrics (WAPE, Bias, MAE, RMSE) and benchmark comparisons.")

# Shared Sidebar Filters
render_sidebar_filters()

if not os.path.exists(MODEL_COMPARISON_FILE) or not os.path.exists(BACKTEST_RESULTS_FILE):
    st.info("Backtest results are calculating. Please wait...")
    st.stop()

@st.cache_data
def load_eval():
    summary = pd.read_csv(MODEL_COMPARISON_FILE)
    backtest = pd.read_csv(BACKTEST_RESULTS_FILE)
    return summary, backtest

summary_df, backtest_df = load_eval()

# Top KPIs for Best Model (Clean display: NO ^ or v symbols)
best_model = summary_df.loc[summary_df['WAPE_Pct'].idxmin()] if not summary_df.empty else None
if best_model is not None:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Top Performing Model", str(best_model['Model']))
        st.caption("Selected Production Architecture")
    with m2:
        st.metric("Validation WAPE", f"{best_model['WAPE_Pct']:.2f}%")
        st.caption("Weighted Absolute Percentage Error")
    with m3:
        st.metric("Forecast Bias", f"{best_model['Bias_Pct']:+.2f}%")
        st.caption("Unbiased Horizon Mean")
    with m4:
        st.metric("Validation MAE", f"{best_model['MAE']:,.1f}")
        st.caption("Mean Absolute Error (Units)")

st.markdown("---")

st.subheader("🏆 Model Comparison Leaderboard")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)
with c1:
    fig_wape = px.bar(
        summary_df, x='Model', y='WAPE_Pct', color='Model',
        title="WAPE (%) Comparison across Models (Lower is Better)",
        text_auto='.2f'
    )
    st.plotly_chart(fig_wape, use_container_width=True)

with c2:
    fig_bias = px.bar(
        summary_df, x='Model', y='Bias_Pct', color='Model',
        title="Forecast Bias (%) Comparison (Closer to 0 is Unbiased)",
        text_auto='.2f'
    )
    st.plotly_chart(fig_bias, use_container_width=True)

st.subheader("📈 Detailed Backtesting Folds by Horizon")
st.dataframe(backtest_df, use_container_width=True, hide_index=True)
