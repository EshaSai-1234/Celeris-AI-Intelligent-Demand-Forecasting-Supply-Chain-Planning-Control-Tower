import streamlit as st
import pandas as pd
import numpy as np
import os
from src.config import CLEANED_TRAIN_FILE, SKU_MASTER_FILE

def render_sidebar_filters(df_hist=None):
    """
    Renders standardized shared global filters in the left sidebar:
    1. Product Family (default 'All')
    2. SKU ID (default 'All', dynamically filtered by Family)
    3. Sales Region (default 'All')
    4. Sales Channel (default 'All')
    5. Customer Name (default 'All')
    
    All selectboxes default to 'All'.
    """
    st.sidebar.title("⚡ Celeris Control Tower")
    st.sidebar.subheader("🎯 Global Shared Filters")
    
    if df_hist is None or df_hist.empty:
        if os.path.exists(CLEANED_TRAIN_FILE):
            df_hist = pd.read_csv(CLEANED_TRAIN_FILE)
        else:
            df_hist = pd.DataFrame()
            
    # 1. Product Family
    fam_opts = ['All']
    if not df_hist.empty and 'Product_Family' in df_hist.columns:
        fam_opts += sorted([str(x) for x in df_hist['Product_Family'].dropna().unique()])
    sel_family = st.sidebar.selectbox("Product Family", fam_opts, index=0)
    
    # 2. SKU ID (dynamically scoped)
    sku_opts = ['All']
    if not df_hist.empty and 'SKU_ID' in df_hist.columns:
        if sel_family != 'All':
            scoped = df_hist[df_hist['Product_Family'] == sel_family]
            sku_opts += sorted([str(x) for x in scoped['SKU_ID'].dropna().unique()])
        else:
            sku_opts += sorted([str(x) for x in df_hist['SKU_ID'].dropna().unique()])
    sel_sku = st.sidebar.selectbox("SKU ID", sku_opts, index=0)
    
    # 3. Sales Region
    reg_opts = ['All']
    if not df_hist.empty and 'Region' in df_hist.columns:
        reg_opts += sorted([str(x) for x in df_hist['Region'].dropna().unique()])
    sel_region = st.sidebar.selectbox("Sales Region", reg_opts, index=0)
    
    # 4. Sales Channel
    chan_opts = ['All']
    if not df_hist.empty and 'Sales_Channel' in df_hist.columns:
        chan_opts += sorted([str(x) for x in df_hist['Sales_Channel'].dropna().unique()])
    sel_channel = st.sidebar.selectbox("Sales Channel", chan_opts, index=0)
    
    # 5. Customer Name
    cust_opts = ['All']
    if not df_hist.empty and 'Customer_Name' in df_hist.columns:
        cust_opts += sorted([str(x) for x in df_hist['Customer_Name'].dropna().unique()])
    sel_customer = st.sidebar.selectbox("Customer Name", cust_opts, index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Health**: 🟢 Pipeline Operational")
    st.sidebar.caption("All filters active & defaulting to 'All'")
    
    return {
        'family': sel_family,
        'sku': sel_sku,
        'region': sel_region,
        'channel': sel_channel,
        'customer': sel_customer
    }

def apply_filters(df, filters):
    """
    Applies the 5 shared global filters to a given DataFrame if the columns exist.
    """
    if df is None or df.empty:
        return df
    
    res = df.copy()
    if filters.get('family') and filters['family'] != 'All' and 'Product_Family' in res.columns:
        res = res[res['Product_Family'] == filters['family']]
    if filters.get('sku') and filters['sku'] != 'All' and 'SKU_ID' in res.columns:
        res = res[res['SKU_ID'] == filters['sku']]
    if filters.get('region') and filters['region'] != 'All' and 'Region' in res.columns:
        res = res[res['Region'] == filters['region']]
    if filters.get('channel') and filters['channel'] != 'All' and 'Sales_Channel' in res.columns:
        res = res[res['Sales_Channel'] == filters['channel']]
    if filters.get('customer') and filters['customer'] != 'All' and 'Customer_Name' in res.columns:
        res = res[res['Customer_Name'] == filters['customer']]
        
    return res
