import os
import numpy as np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred)
    return np.mean(200 * diff / denominator)

def print_banner(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60 + "\n")
