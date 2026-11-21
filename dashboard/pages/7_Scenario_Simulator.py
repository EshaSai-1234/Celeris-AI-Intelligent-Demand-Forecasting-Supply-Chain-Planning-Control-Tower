import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from src.config import OUTPUTS_DIR
from src.dashboard_shared import render_sidebar_filters, apply_filters

st.set_page_config(page_title="Scenario Simulator", page_icon="🎛️", layout="wide")
st.title("🎛️ Interactive What-If Scenario Simulator")
st.markdown("Simulate macroeconomic shocks (e.g. PMI drops, EV adoption surge, material cost spikes) on 19-month demand.")

rec_path = os.path.join(OUTPUTS_DIR, "reconciled_forecasts_detailed.csv")
if not os.path.exists(rec_path):
    st.info("Generating forecasts...")
    st.stop()

@st.cache_data
def load_base():
    return pd.read_csv(rec_path)

df_base_raw = load_base()

# Shared Global Filters in Sidebar
filters = render_sidebar_filters(df_base_raw)
df_base = apply_filters(df_base_raw, filters)

# Scenario Controls in Main Body
st.subheader("🎛️ Macroeconomic Shock Controls")
c_s1, c_s2, c_s3 = st.columns(3)
with c_s1:
    pmi_shock = st.slider("ISM PMI Shift (points)", -10.0, 10.0, -3.0, 0.5)
with c_s2:
    mat_cost_shock = st.slider("Raw Material Cost Shift (%)", -30.0, 30.0, 10.0, 2.0)
with c_s3:
    ev_shock = st.slider("EV Adoption Shift (%)", -20.0, 20.0, 5.0, 1.0)

# Apply Scenario Elasticities
# Industrial elasticity: ~1.8% demand change per PMI point
# Price/Material elasticity: ~-0.4% demand change per % material cost
# EV adoption: +0.6% on Electrical/Thermal, -0.5% on Powertrain ICE
pmi_factor = 1.0 + (pmi_shock * 0.018)
mat_factor = 1.0 - (mat_cost_shock * 0.004)

df_sim = df_base.copy()
sim_demands = []

for _, row in df_sim.iterrows():
    base_d = row['Units_Sold_Forecast']
    fam = str(row.get('Product_Family', ''))
    
    ev_fam_factor = 1.0
    if 'Electrical' in fam or 'Thermal' in fam:
        ev_fam_factor += (ev_shock * 0.006)
    elif 'Powertrain' in fam:
        ev_fam_factor -= (ev_shock * 0.005)
        
    sim_d = base_d * pmi_factor * mat_factor * ev_fam_factor
    sim_demands.append(max(0, round(sim_d)))

df_sim['Simulated_Demand'] = sim_demands

base_total = df_base['Units_Sold_Forecast'].sum() if not df_base.empty else 0
sim_total = df_sim['Simulated_Demand'].sum() if not df_sim.empty else 0
delta_total = sim_total - base_total
pct_total = (delta_total / base_total) * 100.0 if base_total > 0 else 0.0

# KPI Summary (Clean display: NO ^ or v symbols)
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Baseline 19M Forecast", f"{base_total:,.0f} units")
    st.caption("Standard Production Plan")
with k2:
    st.metric("Simulated 19M Demand", f"{sim_total:,.0f} units")
    st.caption(f"Net Shift: {pct_total:+.2f}%")
with k3:
    st.metric("Absolute Volume Impact", f"{delta_total:+,.0f} units")
    st.caption("Change in Total Production")

st.markdown("---")

# Monthly Timeline Comparison
if not df_sim.empty:
    sim_monthly = df_sim.groupby('Date').agg({
        'Units_Sold_Forecast': 'sum',
        'Simulated_Demand': 'sum'
    }).reset_index()

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        x=sim_monthly['Date'], y=sim_monthly['Units_Sold_Forecast'],
        mode='lines+markers', name='Baseline Forecast', line=dict(color='#1f77b4', width=2)
    ))
    fig_sim.add_trace(go.Scatter(
        x=sim_monthly['Date'], y=sim_monthly['Simulated_Demand'],
        mode='lines+markers', name='Simulated Scenario Demand', line=dict(color='#d62728', width=3, dash='dash')
    ))
    fig_sim.update_layout(
        title=f"What-If Demand Impact: PMI ({pmi_shock:+0.1f} pts), Materials ({mat_cost_shock:+0.1f}%), EV ({ev_shock:+0.1f}%)",
        xaxis_title="Date", yaxis_title="Units Forecast",
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_sim, use_container_width=True)
