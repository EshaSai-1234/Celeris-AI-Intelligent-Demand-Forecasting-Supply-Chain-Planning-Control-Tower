import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.config import RANDOM_STATE, MODELS_DIR

class QuantileForecaster:
    """
    Trains quantile LightGBM models for P10, P50, and P90 uncertainty estimation.
    Enforces non-negativity and P10 <= P50 <= P90 quantile monotonicity.
    """
    def __init__(self, quantiles=[0.10, 0.50, 0.90], n_estimators=180, learning_rate=0.05):
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = {}
        self.use_residual = False

    def fit(self, X_train, y_train, categorical_features=None):
        self.use_residual = 'feat_seasonal_baseline' in X_train.columns
        if self.use_residual:
            base = X_train['feat_seasonal_baseline'].values
            y_target = y_train - base
        else:
            y_target = y_train

        for q in self.quantiles:
            print(f"-> Fitting Quantile LightGBM for alpha={q:.2f} (P{int(q*100)})...")
            q_model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=5,
                num_leaves=31,
                objective='quantile',
                alpha=q,
                random_state=RANDOM_STATE,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                verbosity=-1
            )
            q_model.fit(
                X_train, y_target,
                categorical_feature=categorical_features if categorical_features else 'auto'
            )
            self.models[f'P{int(q*100)}'] = q_model
        return self

    def predict(self, X_test):
        """
        Returns a DataFrame with columns ['P10', 'P50', 'P90'].
        Enforces P10 <= P50 <= P90 and >= 0.
        """
        results = {}
        base = X_test['feat_seasonal_baseline'].values if (self.use_residual and 'feat_seasonal_baseline' in X_test.columns) else 0

        for q_name, model in self.models.items():
            raw_p = model.predict(X_test)
            results[q_name] = np.maximum(0, np.round(base + raw_p)).astype(int)
            
        pred_df = pd.DataFrame(results, index=X_test.index)
        
        # Enforce quantile monotonicity: P10 <= P50 <= P90
        pred_df['P50'] = np.maximum(pred_df['P10'], pred_df['P50'])
        pred_df['P90'] = np.maximum(pred_df['P50'], pred_df['P90'])
        
        return pred_df

    def save(self):
        for q_name, model in self.models.items():
            path = os.path.join(MODELS_DIR, f"lgbm_quantile_{q_name}.pkl")
            joblib.dump({'model': model, 'use_residual': self.use_residual}, path)
            print(f"Quantile model {q_name} saved to {path}")

    def load(self):
        for q in self.quantiles:
            q_name = f'P{int(q*100)}'
            path = os.path.join(MODELS_DIR, f"lgbm_quantile_{q_name}.pkl")
            if os.path.exists(path):
                saved = joblib.load(path)
                if isinstance(saved, dict):
                    self.models[q_name] = saved['model']
                    self.use_residual = saved.get('use_residual', False)
                else:
                    self.models[q_name] = saved
        return self
