import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from src.config import RANDOM_STATE, MODELS_DIR

class GlobalLightGBMForecaster:
    """
    Global LightGBM Hybrid Regressor: Combines granular seasonal structural baseline
    with gradient boosted tree learning on dynamic residuals (macro, lifecycle, shocks).
    """
    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=45):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.model = None
        self.features = None
        self.use_residual = False

    def fit(self, X_train, y_train, categorical_features=None):
        self.features = list(X_train.columns)
        self.use_residual = 'feat_seasonal_baseline' in X_train.columns
        
        if self.use_residual:
            base = X_train['feat_seasonal_baseline'].values
            y_target = y_train - base
        else:
            y_target = y_train

        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            random_state=RANDOM_STATE,
            objective='regression',
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbosity=-1
        )
        self.model.fit(
            X_train, y_target,
            categorical_feature=categorical_features if categorical_features else 'auto'
        )
        return self

    def predict(self, X_test):
        raw_pred = self.model.predict(X_test)
        if self.use_residual and 'feat_seasonal_baseline' in X_test.columns:
            base = X_test['feat_seasonal_baseline'].values
            final_pred = base + raw_pred
        else:
            final_pred = raw_pred
            
        return np.maximum(0, np.round(final_pred)).astype(int)

    def save(self, filename="production_lgbm_model.pkl"):
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump({'model': self.model, 'use_residual': self.use_residual, 'features': self.features}, path)
        print(f"Model saved to {path}")

    def load(self, filename="production_lgbm_model.pkl"):
        path = os.path.join(MODELS_DIR, filename)
        saved = joblib.load(path)
        if isinstance(saved, dict):
            self.model = saved['model']
            self.use_residual = saved.get('use_residual', False)
            self.features = saved.get('features', None)
        else:
            self.model = saved
        return self


class GlobalXGBoostForecaster:
    """
    Global XGBoost Hybrid Regressor for multi-model benchmarking.
    """
    def __init__(self, n_estimators=180, learning_rate=0.05, max_depth=6):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None
        self.use_residual = False

    def fit(self, X_train, y_train, categorical_features=None):
        self.use_residual = 'feat_seasonal_baseline' in X_train.columns
        if self.use_residual:
            base = X_train['feat_seasonal_baseline'].values
            y_target = y_train - base
        else:
            y_target = y_train

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=RANDOM_STATE,
            subsample=0.8,
            colsample_bytree=0.8,
            enable_categorical=True,
            tree_method='hist',
            n_jobs=-1
        )
        self.model.fit(X_train, y_target)
        return self

    def predict(self, X_test):
        raw_pred = self.model.predict(X_test)
        if self.use_residual and 'feat_seasonal_baseline' in X_test.columns:
            base = X_test['feat_seasonal_baseline'].values
            final_pred = base + raw_pred
        else:
            final_pred = raw_pred
            
        return np.maximum(0, np.round(final_pred)).astype(int)

    def save(self, filename="production_xgb_model.pkl"):
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump({'model': self.model, 'use_residual': self.use_residual}, path)
        print(f"XGBoost Model saved to {path}")

    def load(self, filename="production_xgb_model.pkl"):
        path = os.path.join(MODELS_DIR, filename)
        saved = joblib.load(path)
        if isinstance(saved, dict):
            self.model = saved['model']
            self.use_residual = saved.get('use_residual', False)
        else:
            self.model = saved
        return self
