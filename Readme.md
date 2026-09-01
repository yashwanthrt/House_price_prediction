# House Price Prediction - MLOps Pipeline

A production-ready MLOps pipeline for house price prediction using **Prefect** for orchestration, **MLflow** for model tracking, **DVC** for data versioning, and **FastAPI** for serving predictions.

---

## 🎯 Overview

This project predicts house prices based on features like area, bedrooms, city, furnishing, and locality tier. It implements **continuous retraining** with automatic model promotion:

- **Automated Training**: Pipeline runs every 30 seconds, detects new data, retrains model
- **Auto Model Promotion**: Only promotes if new model's MAE is better than production
- **Model Versioning**: All versions tracked in MLflow with rollback capability
- **Data Versioning**: DVC tracks historical datasets
- **API Serving**: FastAPI serves predictions with automatic model reloading every 10 seconds
- **Containerized**: Docker support for deployment

---

## 🏗️ Architecture

```
data/new/ → Pipeline (Prefect) → Train (sklearn) → MLflow (versioning)
                ↓
           Merge Data
                ↓
           Compare MAE
                ↓
         Production Model
                ↓
         API (FastAPI)
                ↓
         Predictions
```

### Components:
- **`pipeline.py`**: Orchestrates data merging, training, model promotion (Prefect flows/tasks)
- **`train.py`**: Trains LinearRegression model with sklearn preprocessing
- **`deploy.py`**: Registers Prefect deployment (30-second schedule)
- **`app.py`**: FastAPI server for predictions, auto-reloads model every 10 seconds
- **`rollback.py`**: Manually revert to previous model versions
- **`dvc.yaml`**: DVC pipeline definition for reproducibility
- **`mlflow.db`**: MLflow metadata store (run info, metrics, version aliases)

---

## 📦 Prerequisites

- **Python 3.9+**
- **Git** (for version control)
- **Initial data file**: `data/house_prices.csv` (required for training)
- **DVC** (optional, for data versioning)

### ⚠️ Important
MLflow **must be running before** the pipeline starts training. Otherwise, models won't be logged.

---

## 🚀 Installation

### 1. Clone Repository
```bash
git clone <repo-url>
cd House_price_pred
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔧 Running the Project

### Quick Start (Manual Mode - No Automation)
If you just want to test without automatic 30-second scheduling:

**Terminal 1:**
```bash
mlflow server --host 127.0.0.1 --port 5000
```

**Terminal 2:**
```bash
uvicorn src.app:app --reload --port 8000
```

**Terminal 3 (whenever you want to retrain):**
```bash
python src/pipeline.py
```

---

### Full Setup (With Automation - 30 Second Scheduling)

### Step 1: Start MLflow Server (Terminal 1)
```bash
mlflow server --host 127.0.0.1 --port 5000
```
Access at: `http://127.0.0.1:5000`

### Step 2: Start Prefect Server (Terminal 2)
```bash
prefect server start
```
Access at: `http://127.0.0.1:4200`

### Step 3: Start Prefect Worker (Terminal 3)
**Mandatory** — executes scheduled pipeline runs automatically.
```bash
prefect worker start --pool 'default'
```

### Step 4: Register Prefect Deployment (Terminal 4)
```bash
python src/deploy.py
```

### Step 5: Start FastAPI Server (Terminal 5)
```bash
uvicorn src.app:app --reload --port 8000
```
Access API docs at: `http://127.0.0.1:8000/docs`

### Step 6: Verify Initial Data Exists
Make sure `data/house_prices.csv` exists (initial training data). If not, create it from a sample dataset.

### Step 7: Add New Data to Trigger Pipeline
Place new CSV files in `data/new/`:
```bash
cp house_prices_v4_dataset.csv data/new/
```

Pipeline auto-triggers in 30 seconds → trains → promotes if MAE improves → updates production model.

---

## 📁 Project Structure

```
House_price_pred/
├── src/
│   ├── train.py           # Model training script
│   ├── pipeline.py        # Prefect orchestration
│   ├── deploy.py          # Deployment scheduling
│   ├── app.py             # FastAPI server
│   └── rollback.py        # Manual rollback utility
├── data/
│   ├── house_prices.csv   # Main training data
│   └── new/               # New data for retraining (pipeline watches this)
│       └── processed/     # Archived after processing
├── models/
│   └── house_price_model.joblib  # Trained model (joblib)
├── mlruns/                # MLflow experiment artifacts
├── requirements.txt       # Python dependencies
├── dvc.yaml              # DVC pipeline config
├── dvc.lock              # DVC lock file
├── .gitignore            # Git ignore
├── .dvcignore            # DVC ignore
├── .dockerignore         # Docker ignore
├── Dockerfile            # Container definition
└── README.md             # This file
```

---

## 🔌 API Endpoints

### `/health` (GET)
Health check endpoint.
```bash
curl http://127.0.0.1:8000/health
```
Response:
```json
{"status": "ok"}
```

### `/model-info` (GET)
Get current production model info.
```bash
curl http://127.0.0.1:8000/model-info
```
Response:
```json
{
  "model_name": "house_price_model",
  "version": 3,
  "alias": "production",
  "run_id": "abc123def456"
}
```

### `/predict` (POST)
Predict house price.
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area": 1500,
    "bedrooms": 3,
    "city": "Bangalore",
    "furnishing": "Semi-Furnished",
    "locality_tier": "2"
  }'
```
Response:
```json
{"predicted_price_lakhs": 156.42}
```

---

## 🔄 Workflow

### 1. **Data Ingestion**
   - Place new CSV files in `data/new/`
   - Files must have columns: `area`, `bedrooms`, `city`, `furnishing`, `locality_tier`, `price_lakhs`

### 2. **Automatic Pipeline Trigger (Every 30 seconds)**
   - Prefect worker detects scheduled run
   - `find_new_files()` scans `data/new/`
   - `merge_new_data()` concatenates with main dataset, archives source
   - `run_training()` trains LinearRegression model on merged data
   - Logs metrics (MAE, RMSE, R²) to MLflow

### 3. **Model Promotion**
   - `promote_if_better()` compares new model's MAE against production
   - If `new_mae < current_production_mae`: promotes new version to production alias
   - If not: keeps current production model

### 4. **API Updates**
   - Every 10 seconds, API checks MLflow for new production version
   - If version changed, reloads model automatically
   - Serves predictions using latest production model

### 5. **Manual Rollback (Optional)**
   ```bash
   python src/rollback.py 2  # Rollback to version 2
   ```
   API reloads version 2 on next refresh cycle (10 seconds).

---

## 📊 Monitoring

### MLflow UI
- View all training runs, metrics, model versions
- Compare model performance across versions
- Check production model alias

### Prefect UI
- Monitor pipeline execution history
- View task logs and flow runs
- Reschedule or trigger manual runs

### API Logs
- Check terminal where `uvicorn` is running for request logs

---

## 🐳 Docker Deployment (Optional)

### Build Image
```bash
docker build -t house-price-api .
```

### Run Container
```bash
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI="http://host.docker.internal:5000" \
  house-price-api
```

---

## ❌ Troubleshooting

### Issue: "Deployment not found"
**Solution**: Run `python src/deploy.py` to register deployment.

### Issue: Pipeline not triggering automatically
**Solution**: 
- Confirm Prefect server is running: `prefect server start`
- Confirm worker is running with correct API URL
- Check worker logs for errors

### Issue: Model not being promoted
**Solution**: 
- Check MLflow for new model's MAE
- If MAE is higher than production, it won't promote (this is correct behavior)
- Generate better training data or adjust model parameters

### Issue: API returns 503 "Model not loaded"
**Solution**: 
- Ensure MLflow server is running
- Check MLflow has a model with "production" alias
- Check API logs for detailed error

### Issue: Import error in `pipeline.py`
**Solution**: 
- Ensure you're running from project root
- Check `PYTHONPATH`: `set PYTHONPATH=%cd%` (Windows) or `export PYTHONPATH=.` (Linux)

### Issue: DVC not tracking data
**Solution**: 
```bash
dvc add data/house_prices.csv
dvc push  # Push to remote storage
git add data/house_prices.csv.dvc
git commit -m "Add data"
```
