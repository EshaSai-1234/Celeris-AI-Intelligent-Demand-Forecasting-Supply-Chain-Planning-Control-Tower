import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from src.config import CLEANED_TRAIN_FILE, OUTPUTS_DIR
from src.dashboard_shared import render_sidebar_filters, apply_filters

st.set_page_config(page_title="Executive Overview", page_icon="📊", layout="wide")
st.title("📊 Executive Overview: Continuous Demand & Revenue Control Tower")
st.markdown("Continuous historical sales transitioning into 19-month unconstrained forecasts with portfolio mix.")

reconciled_path = os.path.join(OUTPUTS_DIR, "reconciled_forecasts_detailed.csv")
if not os.path.exists(reconciled_path) or not os.path.exists(CLEANED_TRAIN_FILE):
    st.info("Pipeline outputs are loading. Please wait a moment.")
    st.stop()

@st.cache_data
def load_data():
    hist = pd.read_csv(CLEANED_TRAIN_FILE)
    rec = pd.read_csv(reconciled_path)
    return hist, rec

df_hist, df_rec = load_data()

# Shared Sidebar Filters (All defaulting to 'All')
filters = render_sidebar_filters(df_hist)

# Filter Datasets
h_filtered = apply_filters(df_hist, filters)
r_filtered = apply_filters(df_rec, filters)

# Top KPI Cards (Clean metric displays: NO ^ or v symbols)
c1, c2, c3, c4 = st.columns(4)
total_hist_units = h_filtered['Units_Sold'].sum() if not h_filtered.empty else 0
total_fcst_units = r_filtered['Units_Sold_Forecast'].sum() if not r_filtered.empty else 0
p10_sum = r_filtered['P10'].sum() if not r_filtered.empty and 'P10' in r_filtered.columns else total_fcst_units * 0.85
p90_sum = r_filtered['P90'].sum() if not r_filtered.empty and 'P90' in r_filtered.columns else total_fcst_units * 1.15

with c1:
    st.metric("Historical Units Sold", f"{total_hist_units:,.0f}")
    st.caption("2019-01 through 2026-10")
with c2:
    st.metric("19-Month Forecast Demand", f"{total_fcst_units:,.0f}")
    st.caption("2026-11 through 2028-05")
with c3:
    st.metric("P10 Lower Bound", f"{p10_sum:,.0f}")
    st.caption("90% Confidence Floor")
with c4:
    st.metric("P90 Upper Bound", f"{p90_sum:,.0f}")
    st.caption("Stockout Buffer Ceiling")

st.markdown("---")

# Monthly Timeline: History into Forecast
if not h_filtered.empty and not r_filtered.empty:
    hist_monthly = h_filtered.groupby('Date').agg({'Units_Sold': 'sum', 'Unconstrained_Demand': 'sum'}).reset_index()
    fcst_monthly = r_filtered.groupby('Date').agg({'Units_Sold_Forecast': 'sum', 'P10': 'sum', 'P90': 'sum'}).reset_index()
    fcst_monthly = fcst_monthly.rename(columns={'Units_Sold_Forecast': 'Demand'})

    fig = go.Figure()
    # Actual Shipments
    fig.add_trace(go.Scatter(
        x=hist_monthly['Date'], y=hist_monthly['Units_Sold'],
        mode='lines+markers', name='Observed Shipments',
        line=dict(color='#1f77b4', width=2)
    ))
    # Unconstrained History
    fig.add_trace(go.Scatter(
        x=hist_monthly['Date'], y=hist_monthly['Unconstrained_Demand'],
        mode='lines', name='Unconstrained Demand (History)',
        line=dict(color='#ff7f0e', width=2, dash='dot')
    ))
    # 19-Month Forecast
    fig.add_trace(go.Scatter(
        x=fcst_monthly['Date'], y=fcst_monthly['Demand'],
        mode='lines+markers', name='19-Month Forecast',
        line=dict(color='#2ca02c', width=3)
    ))
    # P10/P90 Ribbon
    if 'P10' in fcst_monthly.columns and 'P90' in fcst_monthly.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([fcst_monthly['Date'], fcst_monthly['Date'][::-1]]),
            y=pd.concat([fcst_monthly['P90'], fcst_monthly['P10'][::-1]]),
            fill='toself', fillcolor='rgba(44, 160, 44, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=True, name='P10-P90 Uncertainty Band'
        ))

    # Add vertical divider line safely (no string math crash)
    fig.add_shape(
        type="line",
        x0="2026-10", x1="2026-10",
        y0=0, y1=1, yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    fig.add_annotation(
        x="2026-10", y=1, yref="paper",
        text="Forecast Cutover (Oct 2026)",
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(color="red", size=12)
    )

    fig.update_layout(
        title="Continuous Demand Horizon: History (Jan 2019 - Oct 2026) → 19-Month Forecast (Nov 2026 - May 2028)",
        xaxis_title="Date (YYYY-MM)", yaxis_title="Units Sold / Forecast Demand",
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# Breakdown Views
c_left, c_right = st.columns(2)
with c_left:
    st.subheader("🌐 Demand by Sales Region")
    if not r_filtered.empty and 'Region' in r_filtered.columns:
        reg_df = r_filtered.groupby('Region')['Units_Sold_Forecast'].sum().reset_index()
        fig_reg = px.bar(reg_df, x='Region', y='Units_Sold_Forecast', color='Region', title="19-Month Forecast by Region")
        st.plotly_chart(fig_reg, use_container_width=True)

with c_right:
    st.subheader("🚗 Demand by Product Family")
    if not r_filtered.empty and 'Product_Family' in r_filtered.columns:
        fam_df = r_filtered.groupby('Product_Family')['Units_Sold_Forecast'].sum().reset_index()
        fig_fam = px.pie(fam_df, names='Product_Family', values='Units_Sold_Forecast', title="19-Month Portfolio Mix", hole=0.4)
        st.plotly_chart(fig_fam, use_container_width=True)
