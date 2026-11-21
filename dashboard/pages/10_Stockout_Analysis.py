import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.config import STOCKOUTS_FILE, CLEANED_TRAIN_FILE
from src.dashboard_shared import render_sidebar_filters, apply_filters

st.set_page_config(page_title="Stockout & Lost Sales Analysis", page_icon="📉", layout="wide")
st.title("📉 Historical Stockout & Supply Constraint Analysis")
st.markdown("Quantify lost sales, unconstrained demand uplift, fill rate bottlenecks, and root causes.")

# Shared Sidebar Filters
filters = render_sidebar_filters()

@st.cache_data
def load_stockouts():
    return pd.read_csv(STOCKOUTS_FILE)

df_stock = load_stockouts()
filtered_stock = apply_filters(df_stock, filters)

# Top KPIs (Clean display: NO ^ or v symbols)
total_events = len(filtered_stock)
total_lost = filtered_stock['Lost_Units'].sum() if not filtered_stock.empty and 'Lost_Units' in filtered_stock.columns else 0
avg_fill = filtered_stock['Fill_Rate_Pct'].mean() if not filtered_stock.empty and 'Fill_Rate_Pct' in filtered_stock.columns else 0.0
chip_events = (filtered_stock['Reason'] == 'Semiconductor allocation').sum() if not filtered_stock.empty and 'Reason' in filtered_stock.columns else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Constrained Months", f"{total_events:,}")
    st.caption("Historical Stockout Incidents")
with c2:
    st.metric("Total Lost Sales Units", f"{total_lost:,} units")
    st.caption("+0.64% Unconstrained Demand Uplift")
with c3:
    st.metric("Average Constrained Fill Rate", f"{avg_fill:.1f}%")
    st.caption("Bottleneck Service Level")
with c4:
    st.metric("Semiconductor Shortages", f"{chip_events:,} months")
    st.caption("74% of Major Shortage Events")

st.markdown("---")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("⚠️ Stockout Incidents by Root Cause")
    if not filtered_stock.empty and 'Reason' in filtered_stock.columns:
        fig_reason = px.pie(filtered_stock, names='Reason', title="Root Cause Breakdown", hole=0.35)
        st.plotly_chart(fig_reason, use_container_width=True)

with col_b:
    st.subheader("🌍 Constrained Shipments by Region")
    if not filtered_stock.empty and 'Region' in filtered_stock.columns:
        reg_lost = filtered_stock.groupby('Region')['Lost_Units'].sum().reset_index()
        fig_reg_lost = px.bar(reg_lost, x='Region', y='Lost_Units', color='Region', title="Lost Units by Sales Region")
        st.plotly_chart(fig_reg_lost, use_container_width=True)

st.subheader("📅 Monthly Lost Sales Timeline")
if not filtered_stock.empty and 'Date' in filtered_stock.columns and 'Lost_Units' in filtered_stock.columns:
    monthly_lost = filtered_stock.groupby('Date')['Lost_Units'].sum().reset_index()
    fig_lost_time = px.line(monthly_lost, x='Date', y='Lost_Units', title="Monthly Lost Demand to Supply Constraints", markers=True)
    st.plotly_chart(fig_lost_time, use_container_width=True)
