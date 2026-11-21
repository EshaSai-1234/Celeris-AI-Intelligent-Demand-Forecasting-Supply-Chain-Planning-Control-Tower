import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from src.config import INVENTORY_PLAN_FILE
from src.dashboard_shared import render_sidebar_filters, apply_filters

st.set_page_config(page_title="Procurement Planner", page_icon="📦", layout="wide")
st.title("📦 Procurement Planner & Inventory Optimization")
st.markdown("Automated Safety Stock, Reorder Points (ROP), and P90 Procurement targets to hedge against supply chain stockouts.")

# Shared Sidebar Filters
filters = render_sidebar_filters()

if not os.path.exists(INVENTORY_PLAN_FILE):
    st.info("Inventory plan is generating...")
    st.stop()

@st.cache_data
def load_inv():
    return pd.read_csv(INVENTORY_PLAN_FILE)

df_inv = load_inv()
filtered_inv = apply_filters(df_inv, filters)

# Header KPIs (Clean display: NO ^ or v symbols)
total_ss = filtered_inv['Safety_Stock_Units'].sum() if not filtered_inv.empty and 'Safety_Stock_Units' in filtered_inv.columns else 0
total_p90_proc = filtered_inv['Monthly_Procurement_P90'].sum() if not filtered_inv.empty and 'Monthly_Procurement_P90' in filtered_inv.columns else 0
high_risk_count = (filtered_inv['Inventory_Risk_Class'] == 'High Volatility (Stockout Risk)').sum() if not filtered_inv.empty and 'Inventory_Risk_Class' in filtered_inv.columns else 0

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Recommended Safety Stock", f"{total_ss:,} units")
    st.caption("Z=1.645 (95% Service Level)")
with c2:
    st.metric("Monthly P90 Procurement Rate", f"{total_p90_proc:,} units/mo")
    st.caption("Stockout-Hedged Procurement Target")
with c3:
    st.metric("High Volatility SKUs", f"{high_risk_count:,} SKUs")
    st.caption("Requires Dynamic Lead-Time Buffering")

st.markdown("---")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("🛡️ Inventory Risk Classification")
    if not filtered_inv.empty and 'Inventory_Risk_Class' in filtered_inv.columns:
        fig_risk = px.pie(filtered_inv, names='Inventory_Risk_Class', title="SKU Portfolio Risk Distribution", hole=0.35)
        st.plotly_chart(fig_risk, use_container_width=True)

with col_b:
    st.subheader("📊 Safety Stock by Product Family")
    if not filtered_inv.empty and 'Product_Family' in filtered_inv.columns:
        fam_ss = filtered_inv.groupby('Product_Family')['Safety_Stock_Units'].sum().reset_index()
        fig_ss = px.bar(fam_ss, x='Product_Family', y='Safety_Stock_Units', color='Product_Family', title="Total Safety Stock Buffer by Family")
        st.plotly_chart(fig_ss, use_container_width=True)

st.subheader("📋 SKU-Level Inventory & Procurement Plan")
st.dataframe(filtered_inv, use_container_width=True, hide_index=True)

csv_inv = filtered_inv.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Procurement Plan (CSV)",
    data=csv_inv,
    file_name="celeris_procurement_plan.csv",
    mime="text/csv"
)
