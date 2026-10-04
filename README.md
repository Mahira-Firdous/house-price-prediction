# 🏠 House Price Prediction

A beginner-friendly Machine Learning web application that predicts median house prices using the **California Housing dataset**.

Built with **Python · scikit-learn · Flask · Streamlit**.

---

## 📁 Project Structure

```
house_price_prediction/
│
├── data/                          # (empty — dataset loads automatically)
├── model/                         # Generated after training
│   ├── house_price_model.pkl      # Saved Random Forest model
│   └── scaler.pkl                 # Saved StandardScaler
│
├── notebooks/
│   └── eda_and_training.ipynb     # Full EDA + training walkthrough
│
├── src/
│   ├── preprocess.py              # Data loading & preprocessing
│   ├── train.py                   # Model training, evaluation & saving
│   └── predict.py                 # Prediction helper
│
├── app.py                         # Flask REST API (backend)
├── streamlit_app.py               # Streamlit UI (frontend)
├── requirements.txt
└── README.md
```

---

## 🗃️ Dataset

| Property | Detail |
|---|---|
| Name | California Housing |
| Source | `sklearn.datasets.fetch_california_housing` |
| Samples | 20,640 |
| Features | 8 numeric |
| Target | Median house value (×$100,000) |

**Features used:**

| Feature | Description |
|---|---|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age (years) |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |

---

## 🤖 Machine Learning Model

| | Linear Regression | **Random Forest** ✅ |
|---|---|---|
| Type | Baseline | Main model |
| MAE | ~$53,000 | ~$33,000 |
| R² | ~0.60 | ~0.81 |

**Random Forest Regressor** was chosen as the final model because it:
- Handles non-linear relationships naturally
- Requires minimal tuning
- Provides built-in feature importances
- Significantly outperforms the linear baseline

---

## ⚙️ Prerequisites

- Python 3.9+
- pip

---

## 🚀 Quick Start

### 1. Clone / download the project

```bash
git clone <repository-url>
cd house_price_prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train.py
```

This will:
- Download the California Housing dataset (automatic, ~1 MB)
- Train a Linear Regression baseline and a Random Forest model
- Print a comparison table of metrics
- Save `model/house_price_model.pkl` and `model/scaler.pkl`

Expected output:
```
Loading and preprocessing data …
Training Linear Regression (baseline) …
Training Random Forest Regressor …

==============================================
Model                      MAE     RMSE       R²
==============================================
Linear Regression       0.5332   0.7456   0.5958
Random Forest           0.3296   0.5024   0.8050
==============================================

✓ Model saved  → model/house_price_model.pkl
✓ Scaler saved → model/scaler.pkl
```

> **Note:** `model/*.pkl` files are listed in `.gitignore` and are **not committed to the repository**.
> Run `python src/train.py` after cloning to regenerate them locally.

### 4. Start the Flask API

Open a **new terminal** and run:

```bash
python app.py
```

The API will start at `http://127.0.0.1:5000`.

### 5. Start the Streamlit UI

Open another **new terminal** and run:

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 🌐 API Reference

### `GET /health`

Check if the API is running.

```bash
curl http://127.0.0.1:5000/health
```

Response:
```json
{"status": "ok"}
```

### `POST /predict`

Predict the median house value.

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984,
    "AveBedrms": 1.024,
    "Population": 322.0,
    "AveOccup": 2.555,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

Response:
```json
{
  "predicted_price": "$452,600",
  "predicted_price_raw": 452600.0
}
```

---

## 📓 Jupyter Notebook

The notebook at `notebooks/eda_and_training.ipynb` covers:

1. Dataset overview (`.describe()`, `.info()`)
2. Missing value check
3. Feature distributions (histograms)
4. Correlation heatmap
5. Scatter plots: features vs price
6. Model training & evaluation
7. Actual vs Predicted plots
8. Feature importances bar chart
9. Saving the model

To run it:

```bash
pip install jupyter
jupyter notebook notebooks/eda_and_training.ipynb
```

---

## 🏗️ How It Works

```
User (Browser)
    │
    │  enters house features (sliders/inputs)
    ▼
Streamlit UI  (streamlit_app.py)
    │
    │  POST /predict  (JSON)
    ▼
Flask API  (app.py)
    │
    │  loads model + scaler, scales input, predicts
    ▼
Random Forest Model  (model/house_price_model.pkl)
    │
    │  returns price in $
    ▼
Streamlit UI  →  displays "$XXX,XXX"
```

---

## 📂 File Descriptions

| File | Purpose |
|---|---|
| `src/preprocess.py` | Loads dataset, applies StandardScaler, splits data |
| `src/train.py` | Trains models, prints metrics, saves artefacts |
| `src/predict.py` | Loads saved model, accepts feature dict, returns price |
| `app.py` | Flask API with `/health` and `/predict` endpoints |
| `streamlit_app.py` | Interactive UI — sliders, prediction button, feature chart |
| `notebooks/eda_and_training.ipynb` | Full analysis notebook for learning |
| `requirements.txt` | All Python dependencies |

---

## 🧑‍💻 Tech Stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.9+ | Language |
| pandas | 2.2.x | Data manipulation |
| NumPy | 1.26.x | Numerical computing |
| scikit-learn | 1.5.x | ML model + preprocessing |
| joblib | 1.4.x | Model serialisation |
| Flask | 3.0.x | REST API backend |
| Streamlit | 1.35.x | Web UI frontend |
| Matplotlib/Seaborn | 3.9.x / 0.13.x | Data visualisation |

---

## 📝 License

This project is intended for educational purposes.
