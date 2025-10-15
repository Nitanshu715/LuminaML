# src/models.py
"""
LightGBM training, saving, loading, prediction utilities and SMAPE metric.
"""
import os
import numpy as np
import lightgbm as lgb

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
if not os.path.exists(ARTIFACT_DIR):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
def predict_ensemble(X_test, models_dir="student_resource/src/artifacts/models"):
    """Loads all models from the folder and averages their predictions."""
    preds = []
    for f in os.listdir(models_dir):
        if f.endswith(".pkl"):
            model = joblib.load(os.path.join(models_dir, f))
            preds.append(model.predict(X_test))
    preds = np.array(preds)
    final_preds = preds.mean(axis=0)
    return final_preds

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (in percent).
    y_true and y_pred are arrays in the original price scale.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-6, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0


def train_lgb_cv(X, y, n_splits=5, random_state=42, params=None, verbose_eval=100):
    """
    Train LightGBM with K-Fold CV on X, y (y is expected to be log1p(price)).
    Returns: list of lgb.Booster objects (one per fold) and oof predictions (log-space).
    """
    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 128,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "seed": random_state,
            "verbosity": -1,
        }

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    models = []
    oof = np.zeros(X.shape[0], dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"=== Fold {fold+1}/{n_splits} ===")
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val)

        bst = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        num_boost_round=1000,
        callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(100),
    ],

        )

        preds_val = bst.predict(X_val, num_iteration=bst.best_iteration)
        oof[val_idx] = preds_val
        models.append(bst)

        fold_smape = smape(np.expm1(y_val), np.expm1(preds_val))
        print(f"Fold {fold+1} SMAPE: {fold_smape:.4f}%")

    overall = smape(np.expm1(y), np.expm1(oof))
    print(f"OOF SMAPE (overall): {overall:.4f}%")
    return models, oof


def save_models(models, prefix="lgbm"):
    """
    Save LightGBM Booster models using booster.save_model method.
    Files will be saved to src/artifacts/{prefix}_fold{fold}.txt
    """
    for i, m in enumerate(models):
        path = os.path.join(ARTIFACT_DIR, f"{prefix}_fold{i}.txt")
        m.save_model(path)
        print(f"Saved model to: {path}")


def load_models(prefix="lgbm"):
    """
    Load saved LightGBM models from artifacts folder.
    Returns list of lgb.Booster
    """
    models = []
    i = 0
    while True:
        p = os.path.join(ARTIFACT_DIR, f"{prefix}_fold{i}.txt")
        if not os.path.exists(p):
            break
        bst = lgb.Booster(model_file=p)
        models.append(bst)
        i += 1
    print(f"Loaded {len(models)} model(s) from artifacts.")
    return models


def predict_with_models(models, X):
    """
    Predict using a list of booster models. Returns mean prediction (log space).
    """
    if len(models) == 0:
        raise ValueError("No models provided for prediction.")
    preds = np.zeros((len(models), X.shape[0]), dtype=float)
    for i, m in enumerate(models):
        preds[i] = m.predict(X, num_iteration=m.best_iteration)
    return preds.mean(axis=0)
