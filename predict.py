# src/predict.py
# Backwards-compatible predict() for sample_code.py
# Safe, robust, uses absolute paths, loads saved TF-IDF (no refit), optional SVD.

import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import traceback

ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "artifacts")

def _get_project_paths():
    """Returns tuple: (project_dir, dataset_dir, artifacts_dir)"""
    src_dir = os.path.dirname(os.path.abspath(__file__))        # .../student_resource/src
    project_dir = os.path.dirname(src_dir)                      # .../student_resource
    dataset_dir = os.path.join(project_dir, "dataset")
    artifacts_dir = os.path.join(src_dir, "artifacts")
    return project_dir, dataset_dir, artifacts_dir

def _safe_load_joblib(path, required=False, name=None):
    """Safely loads a joblib artifact"""
    name = name or os.path.basename(path)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Required artifact not found: {path}")
        else:
            print(f"⚠️ Optional artifact not found: {path} — continuing without it.")
            return None
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load {name} from {path}: {e}")

def _make_predictions(models, X):
    """Make predictions using a list of LightGBM or sklearn-like models"""
    preds = []
    for m in models:
        try:
            if isinstance(m, lgb.basic.Booster):
                p = m.predict(X, num_iteration=m.best_iteration or None)
            else:
                p = m.predict(X)
            preds.append(p)
        except Exception as e:
            # Fallback: convert to dense array if sparse
            try:
                X_dense = X.toarray() if hasattr(X, "toarray") else np.array(X)
                if isinstance(m, lgb.basic.Booster):
                    p = m.predict(X_dense, num_iteration=m.best_iteration or None)
                else:
                    p = m.predict(X_dense)
                preds.append(p)
            except Exception as e2:
                raise RuntimeError(f"Model prediction failed: {e} / {e2}")
    preds = np.vstack(preds)
    avg = preds.mean(axis=0)
    return avg

def predict():
    """
    Backwards-compatible function for sample_code.py
    Runs prediction on dataset/test.csv using artifacts in src/artifacts and writes dataset/test_out.csv
    """
    try:
        project_dir, dataset_dir, artifacts_dir = _get_project_paths()
        test_csv = os.path.join(dataset_dir, "test.csv")
        out_csv = os.path.join(dataset_dir, "test_out.csv")

        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"Test file not found: {test_csv}")

        print("🔮 Starting prediction phase...")
        df_test = pd.read_csv(test_csv)
        print(f"✅ Test data loaded: {len(df_test)} rows")

        # Check catalog_content column
        if "catalog_content" not in df_test.columns:
            raise KeyError("Test CSV missing required column: 'catalog_content'")

        # Load TF-IDF (required)
        tfidf_path = os.path.join(artifacts_dir, "tfidf.pkl")
        tfidf = _safe_load_joblib(tfidf_path, required=True, name="TF-IDF vectorizer")

        # Load optional SVD
        svd_path = os.path.join(artifacts_dir, "svd.pkl")
        svd = _safe_load_joblib(svd_path, required=False, name="SVD")

        # Transform text ONLY using the saved vectorizer
        print("Transforming catalog text with saved TF-IDF...")
        texts = df_test["catalog_content"].fillna("").astype(str).tolist()
        X_text = tfidf.transform(texts)
        print(f"✅ TF-IDF shape (test): {getattr(X_text, 'shape', 'unknown')}")

        # Apply SVD if available
        if svd is not None:
            print("Applying saved SVD to reduce dimensionality...")
            X_text = svd.transform(X_text)
            print(f"✅ After SVD shape: {getattr(X_text, 'shape', 'unknown')}")

        # Load only LightGBM booster models
        models = []
        for f in os.listdir(artifacts_dir):
            if f.startswith("lgbm_fold") and f.endswith(".txt"):
                full_path = os.path.join(artifacts_dir, f)
                try:
                    model = lgb.Booster(model_file=full_path)
                    models.append(model)
                except Exception as e:
                    print(f"⚠️ Skipped {f}: not a valid LightGBM model ({e})")

        if not models:
            raise RuntimeError("No LightGBM model files found for prediction.")

        print(f"✅ Loaded {len(models)} LightGBM model(s) for prediction.")

        # Predict
        print("🧮 Generating predictions...")
        preds = _make_predictions(models, X_text)

        # Ensure positive floats
        preds = np.array(preds, dtype=float)
        preds = np.clip(preds, 0.01, None)

        out_df = pd.DataFrame({
            "sample_id": df_test["sample_id"],
            "price": preds
        })

        if len(out_df) != len(df_test):
            raise RuntimeError("Output row count mismatch with test input.")

        out_df.to_csv(out_csv, index=False)
        print(f"✅ Predictions saved to: {out_csv}")
        print("🎉 Prediction complete.")
        return out_csv

    except Exception as e:
        print("❗ Prediction failed with error:")
        traceback.print_exc()
        raise

# --- End of predict.py ---
