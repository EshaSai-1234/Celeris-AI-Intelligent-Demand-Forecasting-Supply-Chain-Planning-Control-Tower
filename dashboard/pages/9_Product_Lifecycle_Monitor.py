import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.config import SKU_MASTER_FILE
from src.dashboard_shared import render_sidebar_filters, apply_filters

st.set_page_config(page_title="Product Lifecycle Monitor", page_icon="🔄", layout="wide")
st.title("🔄 Product Lifecycle & SKU Portfolio Monitor")
st.markdown("Track active product lines, planned new launches, EOL sunsets, and predecessor replacement mappings.")

# Shared Sidebar Filters
filters = render_sidebar_filters()

@st.cache_data
def load_sku_master():
    return pd.read_csv(SKU_MASTER_FILE)

df_sku = load_sku_master()
filtered_sku = apply_filters(df_sku, filters)

# Lifecycle KPIs (Clean display: NO ^ or v symbols)
c1, c2, c3, c4 = st.columns(4)
total_skus = len(filtered_sku)
active_skus = (filtered_sku['Lifecycle_Status'] == 'Active').sum() if 'Lifecycle_Status' in filtered_sku.columns else 0
planned_launches = (filtered_sku['Lifecycle_Status'] == 'Planned Launch').sum() if 'Lifecycle_Status' in filtered_sku.columns else 0
eol_count = filtered_sku['EOL_Date'].notna().sum() if 'EOL_Date' in filtered_sku.columns else 0

with c1:
    st.metric("Total Master SKUs", f"{total_skus:,}")
    st.caption("Scoped Portfolio Count")
with c2:
    st.metric("Active SKUs", f"{active_skus:,}")
    st.caption("Active Commercial Models")
with c3:
    st.metric("Planned Launches", f"{planned_launches:,}")
    st.caption("Pipeline Introductions")
with c4:
    st.metric("Discontinued / EOL", f"{eol_count:,}")
    st.caption("Zeroed Post-Sunset")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Lifecycle Status Breakdown")
    if not filtered_sku.empty and 'Lifecycle_Status' in filtered_sku.columns:
        fig_status = px.pie(filtered_sku, names='Lifecycle_Status', title="Lifecycle Status Share", hole=0.35)
        st.plotly_chart(fig_status, use_container_width=True)

with col2:
    st.subheader("⚡ Powertrain Application Mix")
    if not filtered_sku.empty and 'Powertrain_Application' in filtered_sku.columns:
        fig_pwt = px.bar(
            filtered_sku['Powertrain_Application'].value_counts().reset_index(),
            x='Powertrain_Application', y='count', color='Powertrain_Application',
            title="SKU Count by Powertrain Application"
        )
        st.plotly_chart(fig_pwt, use_container_width=True)

st.subheader("🔄 Predecessor Replacement & EOL Tracking Table")
replaces_df = filtered_sku[filtered_sku['Replaces_SKU'].notna() | filtered_sku['EOL_Date'].notna()] if not filtered_sku.empty else pd.DataFrame()
if not replaces_df.empty:
    cols = [c for c in ['SKU_ID', 'Product_Family', 'Launch_Date', 'EOL_Date', 'Replaces_SKU', 'Lifecycle_Status'] if c in replaces_df.columns]
    st.dataframe(replaces_df[cols], use_container_width=True, hide_index=True)
else:
    st.info("No predecessor replacements or EOL constraints match the selected filters.")
