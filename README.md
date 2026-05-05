# Freight Demand Forecasting System

> A complete end-to-end machine learning project that predicts freight demand (order volume) using historical logistics data. Includes data preprocessing, model training, comparison, visualization, and a web-based prediction interface.

---

## Project Overview

This system uses a real-world logistics dataset covering daily operations from 2020 to 2024 (1,461 records) to forecast daily freight order volumes. Two regression models are trained and compared:

- **Linear Regression** — baseline model for interpretability
- **Random Forest Regressor** — ensemble model for higher accuracy

The best-performing model is deployed behind a Flask web application with a modern, responsive user interface.

---

## Project Structure

```
Freight_Demand_Forecasting/
├── data/
│   ├── archive.zip                          # Original compressed dataset
│   └── logistics_dataset_2020_2024.csv      # Raw dataset (1,461 rows x 21 cols)
├── src/
│   ├── data_preprocessing.py                # Data cleaning, feature engineering, scaling
│   ├── train_model.py                       # Model training, evaluation, visualization
│   └── predict.py                           # Single-sample prediction utility
├── model/
│   ├── best_model_random_forest.pkl         # Saved best model (pickle)
│   ├── random_forest_model.pkl              # Saved Random Forest model
│   ├── scaler.pkl                           # Fitted StandardScaler
│   └── feature_names.pkl                    # Feature name list for predictions
├── app/
│   ├── app.py                               # Flask web application
│   └── templates/
│       └── index.html                       # Modern UI with prediction form
├── outputs/
│   ├── actual_vs_predicted_best_model.png   # Best model: actual vs predicted scatter
│   ├── actual_vs_predicted_random_forest.png
│   ├── actual_vs_predicted_linear_regression.png
│   ├── prediction_timeline_best_model.png   # Best model: timeline comparison
│   ├── feature_importance.png               # Random Forest feature importance
│   └── model_comparison.png                 # MAE/RMSE comparison bar chart
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## Dataset Features

| Feature | Type | Description |
|---|---|---|
| `date` | datetime | Calendar date (2020-01-01 to 2024-12-31) |
| `holiday_flag` | binary | 1 if public holiday, 0 otherwise |
| `promo_flag` | binary | 1 if promotional campaign active |
| `delivery_count` | numeric | Number of deliveries made |
| `inventory_level` | numeric | Current warehouse inventory |
| `lead_time_avg` | float | Average order lead time (days) |
| `warehouse_utilization` | float | Warehouse capacity usage (0-1) |
| `vehicle_availability` | numeric | Available vehicles |
| `fuel_price_index` | float | Fuel cost index |
| `economic_index` | float | Economic activity index |
| `temperature` | float | Ambient temperature |
| `rainfall_mm` | float | Daily rainfall in mm |
| `lag_1_order` | numeric | Order volume from previous day |
| `lag_7_order` | numeric | Order volume from 7 days ago |
| `lag_30_order` | numeric | Order volume from 30 days ago |
| **`order_volume`** | **numeric** | **TARGET: daily order volume** |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

```bash
# 1. Navigate to the project directory
cd Freight_Demand_Forecasting

# 2. Create and activate a virtual environment (recommended)
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1: Train the Model

Run the full training pipeline (preprocessing + model training + visualization):

```bash
cd src
python train_model.py
```

This will:
- Preprocess the raw dataset
- Train Linear Regression and Random Forest models
- Compare performance using MAE, RMSE, and R²
- Save the best model to `model/`
- Generate visualization plots in `outputs/`

Expected output:
```
MODEL COMPARISON
Model                           MAE          RMSE         R²
-----------------------------------------------------------------
Linear Regression              XX.XXXX      XX.XXXX      0.XXXX
Random Forest                  XX.XXXX      XX.XXXX      0.XXXX
-----------------------------------------------------------------
  >>> Best Model: Random Forest (RMSE = XX.XXXX)
```

### Step 2: Run the Web Application

```bash
cd app
python app.py
```

Then open your browser and navigate to:

```
http://localhost:5000
```

### Step 3: Make Predictions

1. Fill in the input fields with feature values
2. Click the **Predict Demand** button
3. View the predicted freight demand with estimated range

### Step 4: Test Single Prediction (CLI)

```bash
cd src
python predict.py
```

## tests
```bash
venv\Scripts\activate
pip install pytest
python -m pytest
```
---

## Model Performance

Models are evaluated on a held-out test set (20% of data) using:

- **MAE** (Mean Absolute Error): Average absolute prediction error
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **R² Score**: Proportion of variance explained by the model

The system automatically selects the model with the lowest RMSE as the best model for deployment.

---

## API Endpoints

| Method | URL | Description |
|---|---|---|
| GET | `/` | Render prediction form with visualizations |
| POST | `/predict` | Accept feature inputs, return prediction JSON |
| GET | `/health` | Health check endpoint |

### Prediction API Response

```json
{
    "success": true,
    "prediction": 425,
    "lower_bound": 391,
    "upper_bound": 459,
    "raw_prediction": 424.87,
    "model_used": "Random Forest Regressor",
    "input_summary": {
        "delivery_count": 400,
        "vehicle_availability": 30,
        "fuel_price_index": 85.0,
        "temperature": 22.0,
        "lag_1_order": 420
    }
}
```

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computing |
| Scikit-learn | Machine learning models and metrics |
| Matplotlib | Data visualization |
| Flask | Web application framework |
| HTML5 + CSS3 | Frontend user interface |

---

## License

This project is for educational and demonstration purposes.
