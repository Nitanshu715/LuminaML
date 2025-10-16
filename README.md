# 🌟 LuminaML – Intelligent Product Price Prediction System

> **Amazon ML Challenge 2025 Submission**  
> An end-to-end machine learning pipeline built to predict product prices using Amazon catalog data, integrating TF-IDF, SVD, and LightGBM models. Designed with robustness, modularity, and production readiness in mind.

---

## 🚀 Overview

**LuminaML** is a refined and production-grade ML inference pipeline developed as part of the **Amazon ML Challenge 2025**.  
It processes raw product catalog content, transforms it into numerical embeddings, and predicts price ranges with optimized LightGBM models.

This project focuses on *stability, compatibility, and interpretability*, ensuring the prediction process runs flawlessly even across changing data structures or missing artifacts. Here is the  [Test Cases and Artifacts](https://drive.google.com/drive/folders/1ohb0MNvgbovcwIZCNzCaQI3ChWPDN0Gu?usp=sharing).

---

## 🧠 Core Features

- 🧩 **Modular Architecture** — Seamlessly structured for dataset, artifact, and model management.  
- 🔍 **TF-IDF + SVD Pipeline** — Extracts semantic meaning from catalog descriptions efficiently.  
- ⚙️ **LightGBM Ensemble** — Combines multiple trained models for improved accuracy and stability.  
- 💾 **Robust Artifact Handling** — Safely loads joblib artifacts with version fallback mechanisms.  
- 🧱 **Error-Resilient Design** — Protects against missing files, shape mismatches, and broken models.  
- 🧮 **Automatic Output Generation** — Generates a complete `test_out.csv` with zero missing predictions.

---

## 🧩 Tech Stack

| Layer | Tools / Frameworks |
|--------|--------------------|
| **Language** | Python 3.13 |
| **ML Models** | LightGBM, scikit-learn |
| **Feature Extraction** | TF-IDF, TruncatedSVD |
| **Libraries** | Pandas, NumPy, Joblib |
| **Versioning** | Git & GitHub |
| **Runtime** | Command Line / CLI Execution |

---

## 🧬 Project Structure

```
student_resource/
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── test_out.csv        ← auto-generated predictions
│
├── src/
│   ├── artifacts/
│   │   ├── tfidf.pkl
│   │   ├── svd.pkl
│   │   ├── lgbm_fold_0.txt
│   │   └── lgbm_fold_1.txt
│   ├── predict.py          ← main inference logic
│
├── sample_code.py          ← entry script (runs predict())
└── README.md               ← project documentation
```

---

## ⚡ How It Works

1. **Load Dataset** → Reads test data from `dataset/test.csv`  
2. **Load Artifacts** → Imports pre-trained TF-IDF, SVD (if available), and LightGBM models  
3. **Transform Text Data** → Converts catalog descriptions into vectorized embeddings  
4. **Generate Predictions** → LightGBM ensemble predicts final prices  
5. **Save Output** → Results are saved in `dataset/test_out.csv` automatically  

---

## 🧾 Example Output

| sample_id | price |
|------------|--------|
| 100179     | 59.32  |
| 245611     | 12.48  |
| 146263     | 89.14  |
| ...        | ...    |

✅ **File saved at:** `dataset/test_out.csv`

---

## 🛠️ How to Run

```bash
# Step 1: Navigate to project folder
cd AmazonML/student_resource

# Step 2: Run the sample code
python sample_code.py

# Step 3: Check results
cat dataset/test_out.csv
```

💡 **Tip:** Make sure your `src/artifacts/` folder contains the trained `.pkl` and `.txt` model files before running.

---

## 🧠 Challenges Solved

- 🩹 Fixed broken path references and inconsistent artifact loading logic.  
- 🧾 Enhanced TF-IDF and SVD usage to ensure backward compatibility with existing training setups.  
- 🔄 Unified dataset paths for consistency between submission and local runs.  
- 🧰 Built a fail-safe prediction loop that works across varied data structures.

---

## 🌈 Vision

**LuminaML** was built with a singular goal — to illuminate the dark corners of machine learning inference pipelines.  
It showcases how structured engineering, debugging, and architecture design can convert a failing model system into a production-grade, scalable pipeline.

---

## 👨‍💻 Author

**Nitanshu Tak**  
B.Tech CSE (CCVT) @ UPES, Dehradun  
- 📧 [nitanshutak070105@gmail.com](mailto:nitanshutak070105@gmail.com)  
- 🌐 [LinkedIn](https://linkedin.com/in/nitanshu) | [GitHub](https://github.com/Nitanshu)  

---

> “Precision through clarity — that’s the spirit of Lumina.” 🌟
