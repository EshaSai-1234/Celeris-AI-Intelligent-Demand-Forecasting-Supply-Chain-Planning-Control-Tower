import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.config import ANOMALY_REGISTER_FILE, CLEANED_TRAIN_FILE
from src.dashboard_shared import render_sidebar_filters, apply_filters

st.set_page_config(page_title="Anomaly Monitor", page_icon="🚨", layout="wide")
st.title("🚨 Anomaly Monitor & Outlier Register")
st.markdown("Examine business disruptions, supply constraints, price pre-buying surges, and return adjustments.")

# Shared Sidebar Filters
filters = render_sidebar_filters()

if not os.path.exists(ANOMALY_REGISTER_FILE):
    st.info("Anomaly Register is generating...")
    st.stop()

@st.cache_data
def load_anomalies():
    return pd.read_csv(ANOMALY_REGISTER_FILE)

df_anom = load_anomalies()
filtered_anom = apply_filters(df_anom, filters)

# KPI Header (Clean metric display: NO ^ or v symbols)
k1, k2, k3, k4 = st.columns(4)
total_anom = len(filtered_anom)
supply_const = (filtered_anom['Anomaly_Type'] == 'Supply Constraint / Stockout').sum() if 'Anomaly_Type' in filtered_anom.columns else 0
stat_spikes = (filtered_anom['Anomaly_Type'] == 'Statistical Demand Surge').sum() if 'Anomaly_Type' in filtered_anom.columns else 0
high_sev = (filtered_anom['Severity'] == 'High').sum() if 'Severity' in filtered_anom.columns else 0

with k1:
    st.metric("Total Anomalies Detected", f"{total_anom:,}")
    st.caption("Registered Outliers")
with k2:
    st.metric("Supply Constraints", f"{supply_const:,}")
    st.caption("Stockout Disrupted")
with k3:
    st.metric("Statistical Spikes", f"{stat_spikes:,}")
    st.caption("Demand Outliers")
with k4:
    st.metric("High Severity Events", f"{high_sev:,}")
    st.caption("Critical Priority")

st.markdown("---")

c1, c2 = st.columns(2)
with c1:
    st.subheader("📊 Breakdown by Anomaly Type")
    if not filtered_anom.empty and 'Anomaly_Type' in filtered_anom.columns:
        fig_type = px.pie(filtered_anom, names='Anomaly_Type', title="Distribution of Anomaly Types", hole=0.35)
        st.plotly_chart(fig_type, use_container_width=True)

with c2:
    st.subheader("⚠️ Severity Distribution")
    if not filtered_anom.empty and 'Severity' in filtered_anom.columns:
        fig_sev = px.bar(
            filtered_anom['Severity'].value_counts().reset_index(),
            x='Severity', y='count', color='Severity',
            title="Anomaly Count by Severity Level", text_auto=True,
            color_discrete_map={'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}
        )
        st.plotly_chart(fig_sev, use_container_width=True)

st.subheader("🔍 Interactive Anomaly Register")
col_f1, col_f2 = st.columns(2)
with col_f1:
    types = ['All'] + sorted(list(filtered_anom['Anomaly_Type'].dropna().unique())) if 'Anomaly_Type' in filtered_anom.columns else ['All']
    sel_t = st.selectbox("Filter Anomaly Type", types)
with col_f2:
    sevs = ['All'] + sorted(list(filtered_anom['Severity'].dropna().unique())) if 'Severity' in filtered_anom.columns else ['All']
    sel_s = st.selectbox("Filter Severity", sevs)

display_anom = filtered_anom.copy()
if sel_t != 'All':
    display_anom = display_anom[display_anom['Anomaly_Type'] == sel_t]
if sel_s != 'All':
    display_anom = display_anom[display_anom['Severity'] == sel_s]

st.dataframe(display_anom, use_container_width=True, hide_index=True)

csv = display_anom.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Filtered Anomaly Register (CSV)",
    data=csv,
    file_name="celeris_anomaly_register.csv",
    mime="text/csv"
)
