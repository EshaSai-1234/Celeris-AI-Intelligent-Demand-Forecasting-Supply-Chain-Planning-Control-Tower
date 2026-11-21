import os
import json
import numpy as np
import pandas as pd
import shap
from src.config import SHAP_SUMMARY_FILE, OUTPUTS_DIR

class ModelExplainer:
    """
    Computes TreeExplainer SHAP values and macroeconomic driver attributions.
    """
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def compute_explanations(self, X_sample, df_context=None):
        print(">>> Computing SHAP Feature Importance & Driver Explanations...")
        
        # Calculate SHAP values
        sample_size = min(500, len(X_sample))
        X_sub = X_sample.sample(sample_size, random_state=42)
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_exp = explainer(X_sub)
            shap_vals = shap_exp.values if hasattr(shap_exp, 'values') else explainer.shap_values(X_sub)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        except Exception as e:
            print(f"-> SHAP fallback to model feature_importances_: {e}")
            mean_abs_shap = self.model.feature_importances_ / np.sum(self.model.feature_importances_)
        
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values(by='Mean_Abs_SHAP', ascending=False)
        
        top_10 = feature_importance.head(10).to_dict(orient='records')
        print("-> Top 10 Global Features by SHAP Importance:")
        for r in top_10:
            print(f"   {r['Feature']:<30}: {r['Mean_Abs_SHAP']:.4f}")

        # Compute Macro Lead/Lag Correlations with Product Families
        macro_leads = {}
        if df_context is not None:
            macro_cols = ['ISM_Manufacturing_PMI', 'Auto_Production_YoY_Pct', 'Vehicle_Inventory_Days',
                          'Raw_Material_Cost_Index', 'New_Vehicle_Orders_Index', 'Fuel_Price_Index', 'EV_Adoption_Index']
            avail_macro = [c for c in macro_cols if c in df_context.columns]
            
            for fam, grp in df_context.groupby('Product_Family', observed=True):
                fam_corrs = {}
                for m in avail_macro:
                    corr = grp['Unconstrained_Demand'].corr(grp[m])
                    fam_corrs[m] = round(float(corr), 3) if not np.isnan(corr) else 0.0
                macro_leads[str(fam)] = fam_corrs

        # Compile Summary
        summary = {
            'global_feature_importance': top_10,
            'macro_family_correlations': macro_leads,
            'sample_size_evaluated': sample_size
        }
        
        with open(SHAP_SUMMARY_FILE, 'w') as f:
            json.dump(summary, f, indent=4)
            
        print(f"-> SHAP and Driver Summary saved to {SHAP_SUMMARY_FILE}")
        print(">>> Explainability Analysis Complete!\n")
        return summary
