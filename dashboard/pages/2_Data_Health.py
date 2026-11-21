import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.config import DATA_QUALITY_REPORT_FILE, CLEANED_TRAIN_FILE
from src.dashboard_shared import render_sidebar_filters

st.set_page_config(page_title="Data Health & Quality", page_icon="🏥", layout="wide")
st.title("🏥 Data Health & Automated Pipeline Audit")
st.markdown("Full transparency into pre-cleaning data issues, automated business rule resolutions, and audit verification.")

# Sidebar Filters (All defaulting to 'All')
render_sidebar_filters()

if not os.path.exists(DATA_QUALITY_REPORT_FILE):
    st.info("Data Quality Report is being generated...")
    st.stop()

@st.cache_data
def load_report():
    xls = pd.ExcelFile(DATA_QUALITY_REPORT_FILE)
    summary = pd.read_excel(xls, "Cleaning_Summary")
    nulls = pd.read_excel(xls, "Null_Comparison")
    return summary, nulls

summary_df, null_df = load_report()

st.subheader("📋 Pipeline Data Quality Audit Summary")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("🔍 Missing Values: Raw vs Cleaned")
    fig_null = px.bar(
        null_df, x='Column', y=['Raw_Nulls', 'Clean_Nulls'],
        barmode='group', title="Missing Values Comparison (Raw vs Cleaned)",
        labels={'value': 'Count of Nulls', 'variable': 'Dataset State'}
    )
    st.plotly_chart(fig_null, use_container_width=True)

with col2:
    st.subheader("🛡️ Business Rule Enforcements")
    st.markdown("""
    - **Rule 1 (Date Authority)**: Trusted `Date` string over inconsistent `Year`/`Month` entries; resolved multi-format dates (`YYYY-MM`, `MM/YYYY`, `Jan 2019`, `Feb-2022`).
    - **Rule 2 (Duplicate Record_ID)**: Resolved duplicate `Record_ID` rows using the latest `Load_Timestamp` (3,682 duplicates removed).
    - **Rule 3 (Units_Sold = 0)**: Treated zero units as legitimate no-order occurrences; recovered 2,616 missing units via `Revenue / Unit_Price`.
    - **Rule 4 (Price Harmonization)**: Fixed 100x decimal scale mistakes against annual SKU list prices.
    - **Rule 5 (Lifecycle Constraints)**: Enforced zero sales prior to `Launch_Date` and post `EOL_Date` (156 test rows zeroed).
    - **Rule 6 (Unconstrained Demand)**: Estimated lost sales during supply disruptions, recovering 78,454 units (+0.64% uplift).
    """)

st.subheader("📄 Download Full Audit Report")
with open(DATA_QUALITY_REPORT_FILE, "rb") as f:
    st.download_button(
        label="📥 Download Data Quality Report (.xlsx)",
        data=f,
        file_name="celeris_data_quality_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
