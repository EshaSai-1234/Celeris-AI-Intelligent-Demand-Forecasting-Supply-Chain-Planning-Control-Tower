import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from src.config import CLEANED_TRAIN_FILE
from src.dashboard_shared import render_sidebar_filters, apply_filters

st.set_page_config(page_title="Demand Explorer", page_icon="🔍", layout="wide")
st.title("🔍 Historical Demand & Intermittency Explorer")
st.markdown("Drill down across hierarchies and diagnose demand intermittency via Syntetos-Boylan ADI & CV².")

if not os.path.exists(CLEANED_TRAIN_FILE):
    st.info("Training data is loading...")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv(CLEANED_TRAIN_FILE)

df = load_data()

# Shared Sidebar Filters (All defaulting to 'All')
filters = render_sidebar_filters(df)
filtered = apply_filters(df, filters)

# Demand Time Series Plot
st.subheader("📈 Observed Shipments vs Reconstructed Demand")
if not filtered.empty:
    monthly_trend = filtered.groupby('Date').agg({
        'Units_Sold': 'sum',
        'Unconstrained_Demand': 'sum',
        'Is_Stockout': 'sum'
    }).reset_index()

    fig_trend = px.line(
        monthly_trend, x='Date', y=['Units_Sold', 'Unconstrained_Demand'],
        title="Demand Trajectory (Observed vs Unconstrained)",
        labels={'value': 'Units', 'variable': 'Series'},
        color_discrete_map={'Units_Sold': '#1f77b4', 'Unconstrained_Demand': '#ff7f0e'}
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# Seasonality & Intermittency
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("📅 Monthly Seasonality Index")
    if not filtered.empty:
        seasonality = filtered.groupby('Month')['Unconstrained_Demand'].mean().reset_index()
        overall_mean = filtered['Unconstrained_Demand'].mean()
        seasonality['Seasonality_Index'] = seasonality['Unconstrained_Demand'] / (overall_mean if overall_mean > 0 else 1.0)
        fig_seas = px.bar(
            seasonality, x='Month', y='Seasonality_Index',
            title="Monthly Seasonality Factor (>1.0 = Peak Season)",
            color='Seasonality_Index', color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_seas, use_container_width=True)

with col_right:
    st.subheader("🎯 Demand Intermittency Classification (ADI vs CV²)")
    if not filtered.empty:
        sku_intermittent = filtered.groupby(['SKU_ID', 'Region']).agg(
            total_p=('Unconstrained_Demand', 'count'),
            non_zeros=('Unconstrained_Demand', lambda x: (x > 0).sum()),
            mean_nz=('Unconstrained_Demand', lambda x: x[x > 0].mean() if (x > 0).sum() > 0 else 0),
            std_nz=('Unconstrained_Demand', lambda x: x[x > 0].std() if (x > 0).sum() > 1 else 0)
        ).reset_index()
        
        sku_intermittent['ADI'] = sku_intermittent['total_p'] / np.maximum(1, sku_intermittent['non_zeros'])
        sku_intermittent['CV2'] = (sku_intermittent['std_nz'] / np.maximum(0.001, sku_intermittent['mean_nz'])) ** 2
        
        fig_quad = px.scatter(
            sku_intermittent.head(300), x='ADI', y='CV2', color='Region',
            hover_data=['SKU_ID'], title="Syntetos-Boylan Demand Classification Grid"
        )
        fig_quad.add_hline(y=0.49, line_dash='dash', line_color='red')
        fig_quad.add_shape(
            type="line", x0=1.32, x1=1.32, y0=0, y1=1, yref="paper",
            line=dict(color="red", dash="dash")
        )
        st.plotly_chart(fig_quad, use_container_width=True)
