# Network Anomaly Detection System

This project implements a **network intrusion detection system** using **unsupervised machine learning models** on the **KDD Cup 1999 dataset**. The goal is to detect anomalies or intrusions in network traffic data with minimal labeled data.

---

## Features

-  **Dataset**: KDD Cup 1999 (10% subset)
- ⚙️ **Models Used**:
  - Isolation Forest (unsupervised anomaly detection)
  - Autoencoder-style PCA (dimensionality reduction + reconstruction error)
- 📊 **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- 📈 **Visualizations**:
  - Radar chart of performance
  - Prediction breakdowns

---

##  Model Performance

### 🔬 Autoencoder-Style PCA:
- **Accuracy**: `97%`
- **Precision** (attack): `99%`
- **Recall** (attack): `98%`
- **F1 Score** (attack): `0.98`

### 🧪 Isolation Forest:
- **Accuracy**: `27%`
- **Precision** (attack): `74%`
- **Recall** (attack): `14%`
- **F1 Score** (attack): `0.23`

---

### Setup locally

```bash
git clone https://github.com/priyanshhu04/Network-Anomaly-Detection-System.git
cd Network-Anomaly-Detection-System
pip install -r requirements.txt
streamlit run app.py
