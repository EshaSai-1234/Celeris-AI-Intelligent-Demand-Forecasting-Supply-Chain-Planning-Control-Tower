import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from src.config import OUTPUTS_DIR, SHAP_SUMMARY_FILE
from src.dashboard_shared import render_sidebar_filters

st.set_page_config(page_title="Macro Analysis", page_icon="🌐", layout="wide")
st.title("🌐 Macroeconomic Drivers & Lead/Lag Heatmaps")
st.markdown("Quantify how industrial PMI, fuel prices, auto production, and EV adoption shift demand across product families.")

# Shared Sidebar Filters
render_sidebar_filters()

macro_clean_path = os.path.join(OUTPUTS_DIR, "cleaned_macro_indicators.csv")
if not os.path.exists(macro_clean_path):
    st.info("Macro data loading...")
    st.stop()

@st.cache_data
def load_macro():
    macro = pd.read_csv(macro_clean_path)
    shap_data = None
    if os.path.exists(SHAP_SUMMARY_FILE):
        with open(SHAP_SUMMARY_FILE, 'r') as f:
            shap_data = json.load(f)
    return macro, shap_data

df_macro, shap_data = load_macro()

# Time series of key macro indicators
st.subheader("📈 Global Macroeconomic Indicators (2019-2028)")
sel_indicators = st.multiselect(
    "Select Indicators to Plot",
    [c for c in df_macro.columns if c not in ['Date', 'Year', 'Month']],
    default=['ISM_Manufacturing_PMI', 'EV_Adoption_Index', 'Auto_Production_YoY_Pct']
)

if sel_indicators:
    fig_macro = px.line(df_macro, x='Date', y=sel_indicators, title="Historical & Projected Macro Trends")
    # Safe vertical line (using add_shape to avoid Plotly string axis crash)
    fig_macro.add_shape(
        type="line",
        x0="2026-10", x1="2026-10",
        y0=0, y1=1, yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    fig_macro.add_annotation(
        x="2026-10", y=1, yref="paper",
        text="Forecast Cutover (Oct 2026)",
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(color="red", size=12)
    )
    st.plotly_chart(fig_macro, use_container_width=True)

# Heatmap
if shap_data and 'macro_family_correlations' in shap_data:
    st.subheader("🔥 Macroeconomic Correlation Heatmap by Product Family")
    corr_df = pd.DataFrame(shap_data['macro_family_correlations']).T
    fig_heat = px.imshow(
        corr_df, text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r", title="Correlation Matrix: Macro Indicators vs Family Unconstrained Demand"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# Global SHAP Feature Importance
if shap_data and 'global_feature_importance' in shap_data:
    st.subheader("💡 Global Feature Importance (SHAP)")
    shap_df = pd.DataFrame(shap_data['global_feature_importance'])
    fig_shap = px.bar(
        shap_df, x='Mean_Abs_SHAP', y='Feature', orientation='h',
        title="Top Drivers of Demand Forecasts", color='Mean_Abs_SHAP',
        color_continuous_scale='Blues'
    )
    fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_shap, use_container_width=True)
