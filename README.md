# 🌾 Wheat Yield Prediction using Machine Learning

## 📌 Overview

This project focuses on predicting annual wheat yield in Punjab (India) using historical weather data from **1998–2024**.
It compares two approaches:

* **LassoCV Regression** (manual feature engineering)
* **TSFresh + Support Vector Regression (SVR)** (automated feature extraction)

---

## 📂 Project Structure

```
├── featureGen_featureSelection_yield_prediction.py   # TSFresh + SVR pipeline
├── LassoCV_wheat_yield_prediction.py                # LassoCV model
├── final report_report.docx                         # Detailed project report
├── Wheat_Punjab_Rabi.xlsx                          # Dataset (not included / optional)
└── README.md
```

---

## 📊 Dataset

* Source: Punjab Rabi Wheat dataset
* Time range: **1998–2024**
* Rows: **27**
* Features: Weather-based (rainfall, temperature)
* Target: `yield` (tonnes per hectare)

---

## ⚙️ Approach 1 — LassoCV

* Feature Engineering:

  * Combined rainfall features
  * Yeo-Johnson transformation
* Model: **LassoCV**
* Scaling: **RobustScaler**



---

## ⚙️ Approach 2 — TSFresh + SVR

* Converted dataset to **time-series format**
* Extracted **hundreds of features using TSFresh**
* Selected top **10 features (SelectKBest)**
* Model: **SVR (RBF Kernel)**



---

## 📊 Key Insights

* Both models perform similarly due to:

  * Small dataset size (27 samples)
  * Weak correlation between weather & yield
* TSFresh + SVR gives **slightly better RMSE**
* LassoCV is more **interpretable**

---

## ⚠️ Limitations

* Very small dataset
* Weak feature relationships
* No soil or satellite data
* Limited time-series depth (only few months per year)

---

## 🚀 Future Improvements

* Add satellite data (NDVI)
* Include soil features
* Expand dataset (multiple districts)
* Use deep learning (LSTM)

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* TSFresh
* Plotly / Matplotlib

---

## 📌 Author

**MD Abu Sufyan**
Computer Science Engineering (AI & ML)

---

## 📜 License

This project is for academic and learning purposes.
