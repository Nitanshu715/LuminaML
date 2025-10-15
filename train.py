# src/train.py
import os
import sys
import numpy as np
import pandas as pd
import joblib

# Ensure src/ is in path for relative imports
sys.path.append(os.path.dirname(__file__))

from src.features import build_basic_features, TextFeatureBuilder, ARTIFACT_DIR
from models import train_lgb_cv, save_models

"""
Train script to run baseline:
- loads dataset/train.csv
- builds features (text TF-IDF + SVD + simple numeric)
- trains LightGBM with 5-fold CV on log1p(price)
- saves artifacts (tfidf, svd, models) under src/artifacts/
"""

# Paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# Ensure artifacts folder exists
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def main():
    print("🚀 Starting training pipeline...")
    print(f"Loading training data from: {TRAIN_PATH}")

    train = pd.read_csv(TRAIN_PATH)
    assert "price" in train.columns, "❌ train.csv must contain 'price' column"
    assert "catalog_content" in train.columns, "❌ train.csv must contain 'catalog_content' text column"

    # Clean and prepare target
    train = train[train["price"].notna()].copy()
    train["price"] = train["price"].astype(float).clip(lower=0.01)
    train["log_price"] = np.log1p(train["price"])

    print("Building features (this may take a while depending on TF-IDF size)...")
    X, train_df, text_builder = build_basic_features(train, text_col="catalog_content", fit_builder=None)

    # ✅ Save TF-IDF vectorizer for prediction phase
    tfidf_obj = None
    if hasattr(text_builder, "tfidf") and text_builder.tfidf is not None:
        tfidf_obj = text_builder.tfidf
    elif hasattr(text_builder, "vectorizer") and text_builder.vectorizer is not None:
        tfidf_obj = text_builder.vectorizer

    if tfidf_obj is not None:
        tfidf_path = os.path.join(ARTIFACT_DIR, "tfidf.pkl")
        joblib.dump(tfidf_obj, tfidf_path)
        print(f"✅ TF-IDF vectorizer saved to: {tfidf_path}")
    else:
        print("⚠️ Could not find TF-IDF vectorizer inside text_builder.")


    print("Saving text artifacts...")
    text_builder.save(prefix="text")
    joblib.dump(text_builder, os.path.join(ARTIFACT_DIR, "text_builder.joblib"))
    print("📦 Text artifacts saved to:", ARTIFACT_DIR)

    y = train_df["log_price"].values

    print("Training LightGBM with 5-fold CV...")
    models, oof = train_lgb_cv(X, y, n_splits=5, random_state=42)

    print("Saving models...")
    save_models(models, prefix="lgbm")
    joblib.dump({"oof": oof}, os.path.join(ARTIFACT_DIR, "oof.joblib"))
    print("✅ All artifacts saved to:", ARTIFACT_DIR)
    print("🎯 Training complete.")


def train_model():
    """Entry point for sample_code.py"""
    return main()


if __name__ == "__main__":
    main()
