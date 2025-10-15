# src/features.py
"""
Text feature builder for the Amazon ML hackathon baseline.
- Cleans text
- Extracts simple engineered numeric features (IPQ, lengths)
- Builds TF-IDF + TruncatedSVD embeddings
- Saves/loads vectorizer + SVD artifacts under src/artifacts/
"""
import re
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Artifact directory (inside src)
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
if not os.path.exists(ARTIFACT_DIR):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def clean_text(s):
    """Lowercase, remove urls, non-alphanumeric characters, collapse whitespace."""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    # allow alphanum and spaces only
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_ipq(s):
    """
    Extract Item Pack Quantity (IPQ) or numeric counts from text.
    Return integer if found else 1.
    """
    if pd.isna(s):
        return 1
    s = str(s).lower()
    # common patterns
    patterns = [
        r"pack of (\d+)",
        r"(\d+) pack",
        r"(\d+) count",
        r"set of (\d+)",
        r"contains (\d+)",
        r"qty (\d+)",
        r"\b(\d+)x\b",
        r"(\d+)\s*pcs?\b",
        r"(\d+)\s*pill\b",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            try:
                return int(m.group(1))
            except:
                continue
    # fallback: any number present
    m = re.search(r"(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    return 1


class TextFeatureBuilder:
    """
    TF-IDF + TruncatedSVD builder.
    Use fit_transform on training data, transform on test data.
    """

    def __init__(self, tfidf_max_features=50000, svd_n_components=128):
        self.tfidf_max = tfidf_max_features
        self.svd_comp = svd_n_components
        self.vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max, ngram_range=(1, 2), analyzer="word"
        )
        self.svd = TruncatedSVD(n_components=self.svd_comp, random_state=42)

    def fit_transform(self, texts):
        """
        Fit TF-IDF on texts and then SVD; returns dense array (n_samples, svd_comp).
        texts: iterable of raw strings
        """
        X_tfidf = self.vectorizer.fit_transform(texts)
        X_svd = self.svd.fit_transform(X_tfidf)
        return X_svd

    def transform(self, texts):
        """
        Transform new texts using already fit vectorizer + svd.
        """
        X_tfidf = self.vectorizer.transform(texts)
        X_svd = self.svd.transform(X_tfidf)
        return X_svd

    def save(self, prefix="text"):
        joblib.dump(self.vectorizer, os.path.join(ARTIFACT_DIR, f"{prefix}_tfidf.joblib"))
        joblib.dump(self.svd, os.path.join(ARTIFACT_DIR, f"{prefix}_svd.joblib"))

    def load(self, prefix="text"):
        self.vectorizer = joblib.load(os.path.join(ARTIFACT_DIR, f"{prefix}_tfidf.joblib"))
        self.svd = joblib.load(os.path.join(ARTIFACT_DIR, f"{prefix}_svd.joblib"))


def build_basic_features(df, text_col="catalog_content", fit_builder=None):
    """
    Build features for a dataframe.

    Returns:
      X: np.ndarray (n_samples, feature_dim)
      df_out: copy of df with engineered columns (ipq, counts)
      text_builder: fitted TextFeatureBuilder (if fit_builder was None, this is the fitted one)
    """
    df = df.copy()
    # engineered columns
    df["catalog_clean"] = df[text_col].map(clean_text)
    df["ipq"] = df[text_col].map(extract_ipq)
    df["char_count"] = df["catalog_clean"].map(lambda x: len(x) if pd.notna(x) else 0)
    df["word_count"] = df["catalog_clean"].map(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    df["num_digits"] = df["catalog_clean"].map(lambda x: len(re.findall(r"\d", str(x))) if pd.notna(x) else 0)

    # text features
    if fit_builder is None:
        text_builder = TextFeatureBuilder()
        text_feats = text_builder.fit_transform(df["catalog_clean"].fillna(""))
    else:
        text_builder = fit_builder
        text_feats = text_builder.transform(df["catalog_clean"].fillna(""))

    # numeric features
    num_feats = df[["ipq", "char_count", "word_count", "num_digits"]].fillna(0).values.astype(float)

    # concat text + numeric
    X = np.hstack([text_feats, num_feats])
    return X, df, text_builder
