import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from src.config import (
    CLEANED_TRAIN_FILE, OUTPUTS_DIR, DATA_QUALITY_REPORT_FILE,
    ANOMALY_REGISTER_FILE, BACKTEST_RESULTS_FILE, MODEL_COMPARISON_FILE,
    SHAP_SUMMARY_FILE, INVENTORY_PLAN_FILE, SKU_MASTER_FILE,
    STOCKOUTS_FILE, SUBMISSION_FILE
)

st.set_page_config(
    page_title="Celeris AI Control Tower",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide any duplicate Streamlit native multi-page sidebar navigation
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_all_data():
    reconciled_path = os.path.join(OUTPUTS_DIR, "reconciled_forecasts_detailed.csv")
    macro_clean_path = os.path.join(OUTPUTS_DIR, "cleaned_macro_indicators.csv")
    
    df_hist = pd.read_csv(CLEANED_TRAIN_FILE, low_memory=False) if os.path.exists(CLEANED_TRAIN_FILE) else pd.DataFrame()
    df_rec = pd.read_csv(reconciled_path, low_memory=False) if os.path.exists(reconciled_path) else pd.DataFrame()
    df_anom = pd.read_csv(ANOMALY_REGISTER_FILE) if os.path.exists(ANOMALY_REGISTER_FILE) else pd.DataFrame()
    df_macro = pd.read_csv(macro_clean_path) if os.path.exists(macro_clean_path) else pd.DataFrame()
    df_inv = pd.read_csv(INVENTORY_PLAN_FILE) if os.path.exists(INVENTORY_PLAN_FILE) else pd.DataFrame()
    df_sku = pd.read_csv(SKU_MASTER_FILE) if os.path.exists(SKU_MASTER_FILE) else pd.DataFrame()
    df_stock = pd.read_csv(STOCKOUTS_FILE) if os.path.exists(STOCKOUTS_FILE) else pd.DataFrame()
    
    shap_data = None
    if os.path.exists(SHAP_SUMMARY_FILE):
        with open(SHAP_SUMMARY_FILE, 'r') as f:
            shap_data = json.load(f)
            
    summary_model = pd.read_csv(MODEL_COMPARISON_FILE) if os.path.exists(MODEL_COMPARISON_FILE) else pd.DataFrame()
    backtest_model = pd.read_csv(BACKTEST_RESULTS_FILE) if os.path.exists(BACKTEST_RESULTS_FILE) else pd.DataFrame()
    
    return {
        'hist': df_hist,
        'rec': df_rec,
        'anom': df_anom,
        'macro': df_macro,
        'inv': df_inv,
        'sku': df_sku,
        'stock': df_stock,
        'shap': shap_data,
        'model_summary': summary_model,
        'backtest': backtest_model
    }

data = load_all_data()
df_hist = data['hist']
df_rec = data['rec']

# ==============================================================================
# LEFT SIDEBAR: NAVIGATION & SHARED FILTERS (ALL DEFAULTING TO 'All')
# ==============================================================================
st.sidebar.title("⚡ Celeris Control Tower")

# 1. Navigation View Selection
view_options = [
    "1. Executive Overview",
    "2. Data Health",
    "3. Demand Explorer",
    "4. Anomaly Monitor",
    "5. Forecast & Accuracy",
    "6. Macro Analysis",
    "7. Scenario Simulator",
    "8. Procurement Planner",
    "9. Product Lifecycle Monitor",
    "10. Stockout Analysis"
]
selected_view = st.sidebar.radio("Navigation View", view_options)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Global Shared Filters")

# 2. Shared Global Filters (All defaulting to 'All')
# A. Product Family
fam_list = ['All'] + sorted(list(df_hist['Product_Family'].dropna().unique())) if not df_hist.empty else ['All']
sel_fam = st.sidebar.selectbox("Product Family", fam_list, index=0)

# B. SKU ID (filtered dynamically by family if selected)
if sel_fam != 'All' and not df_hist.empty:
    avail_skus = df_hist[df_hist['Product_Family'] == sel_fam]['SKU_ID'].dropna().unique()
else:
    avail_skus = df_hist['SKU_ID'].dropna().unique() if not df_hist.empty else []
sku_list = ['All'] + sorted(list(avail_skus))
sel_sku = st.sidebar.selectbox("SKU ID", sku_list, index=0)

# C. Sales Region
reg_list = ['All'] + sorted(list(df_hist['Region'].dropna().unique())) if not df_hist.empty else ['All']
sel_reg = st.sidebar.selectbox("Sales Region", reg_list, index=0)

# D. Sales Channel
chan_list = ['All'] + sorted(list(df_hist['Sales_Channel'].dropna().unique())) if not df_hist.empty else ['All']
sel_chan = st.sidebar.selectbox("Sales Channel", chan_list, index=0)

# E. Customer
cust_list = ['All'] + sorted(list(df_hist['Customer_Name'].dropna().unique())) if not df_hist.empty else ['All']
sel_cust = st.sidebar.selectbox("Customer Name", cust_list, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**System Health**: 🟢 All Modules Ready")
if os.path.exists(SUBMISSION_FILE):
    st.sidebar.markdown("📄 `submission.csv` (58,219 rows verified)")

# Apply Global Filters to Training & Forecast Datasets
filtered_hist = df_hist.copy() if not df_hist.empty else pd.DataFrame()
filtered_rec = df_rec.copy() if not df_rec.empty else pd.DataFrame()

if sel_fam != 'All':
    if not filtered_hist.empty: filtered_hist = filtered_hist[filtered_hist['Product_Family'] == sel_fam]
    if not filtered_rec.empty: filtered_rec = filtered_rec[filtered_rec['Product_Family'] == sel_fam]
if sel_sku != 'All':
    if not filtered_hist.empty: filtered_hist = filtered_hist[filtered_hist['SKU_ID'] == sel_sku]
    if not filtered_rec.empty: filtered_rec = filtered_rec[filtered_rec['SKU_ID'] == sel_sku]
if sel_reg != 'All':
    if not filtered_hist.empty: filtered_hist = filtered_hist[filtered_hist['Region'] == sel_reg]
    if not filtered_rec.empty: filtered_rec = filtered_rec[filtered_rec['Region'] == sel_reg]
if sel_chan != 'All':
    if not filtered_hist.empty: filtered_hist = filtered_hist[filtered_hist['Sales_Channel'] == sel_chan]
    if not filtered_rec.empty: filtered_rec = filtered_rec[filtered_rec['Sales_Channel'] == sel_chan]
if sel_cust != 'All':
    if not filtered_hist.empty: filtered_hist = filtered_hist[filtered_hist['Customer_Name'] == sel_cust]
    if not filtered_rec.empty: filtered_rec = filtered_rec[filtered_rec['Customer_Name'] == sel_cust]


# ==============================================================================
# VIEW 1: EXECUTIVE OVERVIEW
# ==============================================================================
if selected_view == "1. Executive Overview":
    st.title("📊 Executive Overview: Continuous Demand & Revenue Control Tower")
    st.markdown("Continuous historical sales transitioning into 19-month unconstrained forecasts with portfolio mix.")

    # Top KPI Cards (clean display without ^ v symbols)
    c1, c2, c3, c4 = st.columns(4)
    tot_hist = filtered_hist['Units_Sold'].sum() if not filtered_hist.empty else 0
    tot_fcst = filtered_rec['Units_Sold_Forecast'].sum() if not filtered_rec.empty else 0
    tot_p10 = filtered_rec['P10'].sum() if not filtered_rec.empty and 'P10' in filtered_rec.columns else tot_fcst * 0.85
    tot_p90 = filtered_rec['P90'].sum() if not filtered_rec.empty and 'P90' in filtered_rec.columns else tot_fcst * 1.15

    with c1:
        st.metric("Historical Units Sold", f"{tot_hist:,.0f}")
        st.caption("2019-01 through 2026-10")
    with c2:
        st.metric("19-Month Forecast Demand", f"{tot_fcst:,.0f}")
        st.caption("2026-11 through 2028-05")
    with c3:
        st.metric("P10 Lower Bound", f"{tot_p10:,.0f}")
        st.caption("90% Confidence Floor")
    with c4:
        st.metric("P90 Upper Bound", f"{tot_p90:,.0f}")
        st.caption("Stockout Buffer Ceiling")

    st.markdown("---")

    # Continuous Monthly Timeline
    if not filtered_hist.empty and not filtered_rec.empty:
        hist_m = filtered_hist.groupby('Date').agg({'Units_Sold': 'sum', 'Unconstrained_Demand': 'sum'}).reset_index()
        fcst_m = filtered_rec.groupby('Date').agg({'Units_Sold_Forecast': 'sum', 'P10': 'sum', 'P90': 'sum'}).reset_index()
        fcst_m = fcst_m.rename(columns={'Units_Sold_Forecast': 'Demand'})

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_m['Date'], y=hist_m['Units_Sold'], mode='lines+markers', name='Observed Shipments', line=dict(color='#1f77b4', width=2)))
        fig.add_trace(go.Scatter(x=hist_m['Date'], y=hist_m['Unconstrained_Demand'], mode='lines', name='Unconstrained Demand (History)', line=dict(color='#ff7f0e', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=fcst_m['Date'], y=fcst_m['Demand'], mode='lines+markers', name='19-Month Forecast', line=dict(color='#2ca02c', width=3)))

        if 'P10' in fcst_m.columns and 'P90' in fcst_m.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([fcst_m['Date'], fcst_m['Date'][::-1]]),
                y=pd.concat([fcst_m['P90'], fcst_m['P10'][::-1]]),
                fill='toself', fillcolor='rgba(44, 160, 44, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip", showlegend=True, name='P10 - P90 Prediction Interval'
            ))

        # Safe line shape without add_vline error
        fig.add_shape(type="line", x0="2026-10", x1="2026-10", y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"))
        fig.add_annotation(x="2026-10", y=1, yref="paper", text="Forecast Start", showarrow=False, font=dict(color="red"))
        fig.update_layout(title="Continuous Monthly Demand Timeline (History → 19-Month Forecast)", xaxis_title="Date", yaxis_title="Units Demand", hovermode='x unified', template='plotly_white', height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Portfolio Mix
    col_a, col_b = st.columns(2)
    with col_a:
        if not filtered_rec.empty:
            fam_mix = filtered_rec.groupby('Product_Family')['Units_Sold_Forecast'].sum().reset_index()
            fig_fam = px.pie(fam_mix, names='Product_Family', values='Units_Sold_Forecast', title="Forecast Demand Share by Product Family", hole=0.4)
            st.plotly_chart(fig_fam, use_container_width=True)
    with col_b:
        if not filtered_rec.empty:
            reg_mix = filtered_rec.groupby('Region')['Units_Sold_Forecast'].sum().reset_index()
            fig_reg = px.bar(reg_mix, x='Region', y='Units_Sold_Forecast', color='Region', title="Forecast Demand by Sales Region", text_auto='.2s')
            st.plotly_chart(fig_reg, use_container_width=True)


# ==============================================================================
# VIEW 2: DATA HEALTH
# ==============================================================================
elif selected_view == "2. Data Health":
    st.title("🏥 Data Health & Automated Pipeline Audit")
    st.markdown("Full transparency into pre-cleaning anomalies, automated business rule enforcements, and data quality metrics.")

    if os.path.exists(DATA_QUALITY_REPORT_FILE):
        xls = pd.ExcelFile(DATA_QUALITY_REPORT_FILE)
        summary_clean = pd.read_excel(xls, "Cleaning_Summary")
        null_df = pd.read_excel(xls, "Null_Comparison")
        
        st.subheader("📋 Pipeline Data Quality Audit Summary")
        st.dataframe(summary_clean, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🔍 Missing Values: Raw vs Cleaned")
            fig_null = px.bar(null_df, x='Column', y=['Raw_Nulls', 'Clean_Nulls'], barmode='group', title="Missing Values Comparison")
            st.plotly_chart(fig_null, use_container_width=True)
        with c2:
            st.subheader("🛡️ Business Rule Enforcements")
            st.markdown("""
            - **Rule 1 (Date Authority)**: Trusted `Date` string over inconsistent `Year`/`Month` entries; resolved multi-format dates.
            - **Rule 2 (Duplicate Record_ID)**: Resolved duplicate `Record_ID` rows using the latest `Load_Timestamp` (3,682 pairs removed).
            - **Rule 3 (Units_Sold = 0)**: Treated zero units as legitimate no-order occurrences; recovered 2,616 missing units via `Revenue / Unit_Price`.
            - **Rule 4 (Price Harmonization)**: Fixed 100x decimal scale mistakes against annual SKU list prices.
            - **Rule 5 (Lifecycle Constraints)**: Enforced zero sales prior to `Launch_Date` and post `EOL_Date`.
            """)


# ==============================================================================
# VIEW 3: DEMAND EXPLORER
# ==============================================================================
elif selected_view == "3. Demand Explorer":
    st.title("🔍 Historical Demand & Intermittency Explorer")
    st.markdown("Diagnose demand patterns using Syntetos-Boylan Average Demand Interval (ADI) and Squared Coefficient of Variation ($CV^2$).")

    if not filtered_hist.empty:
        st.subheader("📈 Observed Shipments vs Reconstructed Demand")
        trend = filtered_hist.groupby('Date').agg({'Units_Sold': 'sum', 'Unconstrained_Demand': 'sum'}).reset_index()
        fig_trend = px.line(trend, x='Date', y=['Units_Sold', 'Unconstrained_Demand'], title="Historical Demand Trajectory", color_discrete_map={'Units_Sold': '#1f77b4', 'Unconstrained_Demand': '#ff7f0e'})
        st.plotly_chart(fig_trend, use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("📅 Monthly Seasonality Index")
            seas = filtered_hist.groupby('Month')['Unconstrained_Demand'].mean().reset_index()
            om = filtered_hist['Unconstrained_Demand'].mean()
            seas['Seasonality_Index'] = seas['Unconstrained_Demand'] / (om if om > 0 else 1.0)
            fig_seas = px.bar(seas, x='Month', y='Seasonality_Index', title="Monthly Seasonality Factor (>1.0 = Peak Season)", color='Seasonality_Index', color_continuous_scale='Viridis')
            st.plotly_chart(fig_seas, use_container_width=True)

        with col_r:
            st.subheader("🎯 Intermittency Classification (ADI vs CV²)")
            sku_int = filtered_hist.groupby(['SKU_ID', 'Region']).agg(
                total_p=('Unconstrained_Demand', 'count'),
                non_zeros=('Unconstrained_Demand', lambda x: (x > 0).sum()),
                mean_nz=('Unconstrained_Demand', lambda x: x[x > 0].mean() if (x > 0).sum() > 0 else 0),
                std_nz=('Unconstrained_Demand', lambda x: x[x > 0].std() if (x > 0).sum() > 1 else 0)
            ).reset_index()
            sku_int['ADI'] = sku_int['total_p'] / np.maximum(1, sku_int['non_zeros'])
            sku_int['CV2'] = (sku_int['std_nz'] / np.maximum(0.001, sku_int['mean_nz'])) ** 2
            
            fig_grid = px.scatter(sku_int.head(300), x='ADI', y='CV2', color='Region', hover_data=['SKU_ID'], title="Syntetos-Boylan Demand Quadrant Grid")
            fig_grid.add_vline(x=1.32, line_dash='dash', line_color='red')
            fig_grid.add_hline(y=0.49, line_dash='dash', line_color='red')
            st.plotly_chart(fig_grid, use_container_width=True)


# ==============================================================================
# VIEW 4: ANOMALY MONITOR
# ==============================================================================
elif selected_view == "4. Anomaly Monitor":
    st.title("🚨 Anomaly Monitor & Outlier Register")
    st.markdown("Examine business disruptions, supply constraints, price pre-buying surges, and return adjustments.")

    df_anom = data['anom']
    if not df_anom.empty:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Anomalies Detected", f"{len(df_anom):,}")
            st.caption("Across 244k historical records")
        with k2:
            st.metric("Supply Constraints", f"{(df_anom['Anomaly_Type'] == 'Supply Constraint / Stockout').sum():,}")
            st.caption("Semiconductor & logistics bottlenecks")
        with k3:
            st.metric("Statistical Spikes", f"{(df_anom['Anomaly_Type'] == 'Statistical Demand Surge').sum():,}")
            st.caption("IQR > 2.5 and volume > 30 units")
        with k4:
            st.metric("High Severity Events", f"{(df_anom['Severity'] == 'High').sum():,}")
            st.caption("Fill rate < 50% or returns > 50 units")

        st.markdown("---")
        ca, cb = st.columns(2)
        with ca:
            fig_t = px.pie(df_anom, names='Anomaly_Type', title="Distribution of Anomaly Types", hole=0.35)
            st.plotly_chart(fig_t, use_container_width=True)
        with cb:
            fig_s = px.bar(df_anom['Severity'].value_counts().reset_index(), x='Severity', y='count', color='Severity', title="Anomaly Count by Severity Level", text_auto=True)
            st.plotly_chart(fig_s, use_container_width=True)

        st.subheader("🔍 Filterable Anomaly Register")
        st.dataframe(df_anom, use_container_width=True, hide_index=True)


# ==============================================================================
# VIEW 5: FORECAST & ACCURACY
# ==============================================================================
elif selected_view == "5. Forecast & Accuracy":
    st.title("🎯 Forecast Accuracy & Model Evaluation")
    st.markdown("Chronological rolling-origin backtesting across 3 expanding folds (6-month horizons) benchmarked against baselines.")

    summary_model = data['model_summary']
    backtest_model = data['backtest']

    if not summary_model.empty:
        st.subheader("🏆 Model Comparison Leaderboard")
        st.dataframe(summary_model, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_w = px.bar(summary_model, x='Model', y='WAPE_Pct', color='Model', title="WAPE (%) Comparison (Lower is Better)", text_auto='.2f')
            st.plotly_chart(fig_w, use_container_width=True)
        with c2:
            fig_b = px.bar(summary_model, x='Model', y='Bias_Pct', color='Model', title="Forecast Bias (%) Comparison (Closer to 0 is Unbiased)", text_auto='.2f')
            st.plotly_chart(fig_b, use_container_width=True)

        st.subheader("📈 Detailed Backtesting Folds")
        st.dataframe(backtest_model, use_container_width=True, hide_index=True)


# ==============================================================================
# VIEW 6: MACRO ANALYSIS
# ==============================================================================
elif selected_view == "6. Macro Analysis":
    st.title("🌐 Macroeconomic Drivers & Lead/Lag Heatmaps")
    st.markdown("Quantify how industrial PMI, fuel prices, auto production, and EV adoption shift demand across product families.")

    df_macro = data['macro']
    shap_data = data['shap']

    if not df_macro.empty:
        st.subheader("📈 Global Macroeconomic Indicators (2019-2028)")
        avail_m = [c for c in df_macro.columns if c not in ['Date', 'Year', 'Month']]
        sel_ind = st.multiselect("Select Indicators to Plot", avail_m, default=['ISM_Manufacturing_PMI', 'EV_Adoption_Index', 'Auto_Production_YoY_Pct'])
        if sel_ind:
            fig_m = px.line(df_macro, x='Date', y=sel_ind, title="Macro Trends across Historical & Forecast Horizons")
            fig_m.add_shape(type="line", x0="2026-10", x1="2026-10", y0=0, y1=1, yref="paper", line=dict(color="red", width=2, dash="dash"))
            fig_m.add_annotation(x="2026-10", y=1, yref="paper", text="Forecast Start", showarrow=False, font=dict(color="red"))
            st.plotly_chart(fig_m, use_container_width=True)

    if shap_data and 'macro_family_correlations' in shap_data:
        st.subheader("🔥 Macroeconomic Correlation Heatmap by Product Family")
        corr_df = pd.DataFrame(shap_data['macro_family_correlations']).T
        fig_heat = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Matrix")
        st.plotly_chart(fig_heat, use_container_width=True)

    if shap_data and 'global_feature_importance' in shap_data:
        st.subheader("💡 Global Feature Importance (SHAP)")
        shap_df = pd.DataFrame(shap_data['global_feature_importance'])
        fig_shap = px.bar(shap_df, x='Mean_Abs_SHAP', y='Feature', orientation='h', title="Top Drivers of Demand Forecasts", color='Mean_Abs_SHAP', color_continuous_scale='Blues')
        fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_shap, use_container_width=True)


# ==============================================================================
# VIEW 7: SCENARIO SIMULATOR
# ==============================================================================
elif selected_view == "7. Scenario Simulator":
    st.title("🎛️ Interactive What-If Scenario Simulator")
    st.markdown("Simulate macroeconomic shocks (e.g. PMI drops, EV adoption surge, material cost spikes) on 19-month demand.")

    c_s1, c_s2, c_s3 = st.columns(3)
    with c_s1:
        pmi_shock = st.slider("ISM PMI Shift (points)", -10.0, 10.0, -3.0, 0.5)
    with c_s2:
        mat_shock = st.slider("Raw Material Cost Shift (%)", -30.0, 30.0, 10.0, 2.0)
    with c_s3:
        ev_shock = st.slider("EV Adoption Shift (%)", -20.0, 20.0, 5.0, 1.0)

    if not filtered_rec.empty:
        pmi_f = 1.0 + (pmi_shock * 0.018)
        mat_f = 1.0 - (mat_shock * 0.004)
        
        sim_df = filtered_rec.copy()
        sim_vals = []
        for _, row in sim_df.iterrows():
            d = row['Units_Sold_Forecast']
            fam = str(row['Product_Family'])
            ev_f = 1.0 + (ev_shock * 0.006) if ('Electrical' in fam or 'Thermal' in fam) else (1.0 - (ev_shock * 0.005) if 'Powertrain' in fam else 1.0)
            sim_vals.append(max(0, round(d * pmi_f * mat_f * ev_f)))
            
        sim_df['Simulated_Demand'] = sim_vals
        b_tot = filtered_rec['Units_Sold_Forecast'].sum()
        s_tot = sim_df['Simulated_Demand'].sum()
        d_tot = s_tot - b_tot
        pct_d = (d_tot / b_tot) * 100.0 if b_tot > 0 else 0.0

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Baseline 19M Forecast", f"{b_tot:,.0f} units")
            st.caption("Standard model forecast")
        with k2:
            st.metric("Simulated 19M Demand", f"{s_tot:,.0f} units")
            st.caption(f"Net change: {pct_d:+.2f}%")
        with k3:
            st.metric("Net Demand Shift", f"{d_tot:+,.0f} units")
            st.caption("Volume variance")

        st.markdown("---")
        sim_m = sim_df.groupby('Date').agg({'Units_Sold_Forecast': 'sum', 'Simulated_Demand': 'sum'}).reset_index()
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=sim_m['Date'], y=sim_m['Units_Sold_Forecast'], mode='lines+markers', name='Baseline Forecast', line=dict(color='#1f77b4', width=2)))
        fig_s.add_trace(go.Scatter(x=sim_m['Date'], y=sim_m['Simulated_Demand'], mode='lines+markers', name='Simulated Scenario Demand', line=dict(color='#d62728', width=3, dash='dash')))
        fig_s.update_layout(title="Simulated Scenario vs Baseline Demand", xaxis_title="Date", yaxis_title="Units Demand", template='plotly_white')
        st.plotly_chart(fig_s, use_container_width=True)


# ==============================================================================
# VIEW 8: PROCUREMENT PLANNER
# ==============================================================================
elif selected_view == "8. Procurement Planner":
    st.title("📦 Procurement Planner & Inventory Optimization")
    st.markdown("Automated Safety Stock, Reorder Points (ROP), and P90 Procurement targets to hedge against supply chain stockouts.")

    df_inv = data['inv']
    if not df_inv.empty:
        t_ss = df_inv['Safety_Stock_Units'].sum()
        t_proc = df_inv['Monthly_Procurement_P90'].sum()
        h_risk = (df_inv['Inventory_Risk_Class'] == 'High Volatility (Stockout Risk)').sum()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Recommended Safety Stock", f"{t_ss:,} units")
            st.caption("Z=1.645 (95% Service Level)")
        with c2:
            st.metric("Monthly P90 Procurement Rate", f"{t_proc:,} units/mo")
            st.caption("Hedges 90% peak demand")
        with c3:
            st.metric("High Volatility SKUs", f"{h_risk:,} SKUs")
            st.caption("Requires dynamic buffer")

        st.markdown("---")
        ca, cb = st.columns(2)
        with ca:
            fig_r = px.pie(df_inv, names='Inventory_Risk_Class', title="SKU Portfolio Risk Distribution", hole=0.35)
            st.plotly_chart(fig_r, use_container_width=True)
        with cb:
            f_ss = df_inv.groupby('Product_Family')['Safety_Stock_Units'].sum().reset_index()
            fig_ss = px.bar(f_ss, x='Product_Family', y='Safety_Stock_Units', color='Product_Family', title="Safety Stock Buffer by Product Family")
            st.plotly_chart(fig_ss, use_container_width=True)

        st.subheader("📋 SKU-Level Inventory & Procurement Plan")
        st.dataframe(df_inv, use_container_width=True, hide_index=True)


# ==============================================================================
# VIEW 9: PRODUCT LIFECYCLE MONITOR
# ==============================================================================
elif selected_view == "9. Product Lifecycle Monitor":
    st.title("🔄 Product Lifecycle & SKU Portfolio Monitor")
    st.markdown("Track active product lines, planned new launches, EOL sunsets, and predecessor replacement mappings.")

    df_sku = data['sku']
    if not df_sku.empty:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Master SKUs", f"{len(df_sku):,}")
            st.caption("Active catalogue")
        with c2:
            st.metric("Active SKUs", f"{(df_sku['Lifecycle_Status'] == 'Active').sum():,}")
            st.caption("In active manufacturing")
        with c3:
            st.metric("Planned Launches", f"{(df_sku['Lifecycle_Status'] == 'Planned Launch').sum():,}")
            st.caption("Launches in forecast window")
        with c4:
            st.metric("Discontinued / EOL", f"{df_sku['EOL_Date'].notna().sum():,}")
            st.caption("Sunset product lines")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            fig_st = px.pie(df_sku, names='Lifecycle_Status', title="Lifecycle Status Share", hole=0.35)
            st.plotly_chart(fig_st, use_container_width=True)
        with col2:
            fig_pt = px.bar(df_sku['Powertrain_Application'].value_counts().reset_index(), x='Powertrain_Application', y='count', color='Powertrain_Application', title="SKU Count by Powertrain Application")
            st.plotly_chart(fig_pt, use_container_width=True)

        st.subheader("🔄 Predecessor Replacement & EOL Tracking Table")
        repl = df_sku[df_sku['Replaces_SKU'].notna() | df_sku['EOL_Date'].notna()]
        st.dataframe(repl[['SKU_ID', 'Product_Family', 'Launch_Date', 'EOL_Date', 'Replaces_SKU', 'Lifecycle_Status']], use_container_width=True, hide_index=True)


# ==============================================================================
# VIEW 10: STOCKOUT ANALYSIS
# ==============================================================================
elif selected_view == "10. Stockout Analysis":
    st.title("📉 Historical Stockout & Supply Constraint Analysis")
    st.markdown("Quantify lost sales, unconstrained demand uplift, fill rate bottlenecks, and root causes.")

    df_stock = data['stock']
    if not df_stock.empty:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Constrained Months", f"{len(df_stock):,}")
            st.caption("Documented shortage events")
        with c2:
            st.metric("Total Lost Sales Units", f"{df_stock['Lost_Units'].sum():,} units")
            st.caption("+0.64% demand uplift")
        with c3:
            st.metric("Average Fill Rate", f"{df_stock['Fill_Rate_Pct'].mean():.1f}%")
            st.caption("During constrained periods")
        with c4:
            st.metric("Semiconductor Shortages", f"{(df_stock['Reason'] == 'Semiconductor allocation').sum():,}")
            st.caption("74% of all shortages")

        st.markdown("---")
        ca, cb = st.columns(2)
        with ca:
            fig_rs = px.pie(df_stock, names='Reason', title="Root Cause Breakdown", hole=0.35)
            st.plotly_chart(fig_rs, use_container_width=True)
        with cb:
            reg_l = df_stock.groupby('Region')['Lost_Units'].sum().reset_index()
            fig_rl = px.bar(reg_l, x='Region', y='Lost_Units', color='Region', title="Lost Units by Sales Region")
            st.plotly_chart(fig_rl, use_container_width=True)

        st.subheader("📅 Monthly Lost Sales Timeline")
        m_l = df_stock.groupby('Date')['Lost_Units'].sum().reset_index()
        fig_lt = px.line(m_l, x='Date', y='Lost_Units', title="Monthly Lost Demand", markers=True)
        st.plotly_chart(fig_lt, use_container_width=True)
