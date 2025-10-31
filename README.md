# 🌸 Iris Classification with MLOps Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.0+-green.svg)
![Feast](https://img.shields.io/badge/Feast-0.53-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

*A production-ready MLOps pipeline for Iris species classification using Feature Store (Feast) and Experiment Tracking (MLflow)*

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Pipeline Workflow](#-pipeline-workflow)
- [Results](#-results)
- [MLflow Integration](#-mlflow-integration)
- [Feast Feature Store](#-feast-feature-store)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project demonstrates a complete **MLOps pipeline** for building, training, and deploying a machine learning model to classify Iris flower species. It integrates modern MLOps tools and best practices:

- ✨ **Feature Store Management** with Feast for consistent feature engineering
- 📊 **Experiment Tracking** with MLflow for reproducible model development
- 🔄 **Hyperparameter Tuning** with automated model selection
- 🚀 **Model Registry** for version control and deployment
- 📈 **Production Deployment** with stage transitions

The pipeline classifies Iris flowers into three species (Setosa, Versicolor, Virginica) based on sepal and petal measurements using a Decision Tree Classifier.

---

## ✨ Features

### 🎨 Core Capabilities

- **Feature Store Integration**: Centralized feature management using Feast
  - Offline store for training data retrieval
  - Online store for real-time inference
  - Time-travel capabilities for point-in-time correct features

- **Automated Hyperparameter Tuning**: 
  - Grid search over multiple hyperparameters
  - Automatic tracking of all experiments
  - Best model selection based on accuracy metrics

- **MLflow Experiment Tracking**:
  - Comprehensive logging of parameters, metrics, and models
  - Experiment comparison and visualization
  - Model versioning and registry

- **Production-Ready Deployment**:
  - Model staging (None → Staging → Production)
  - Version control with automatic archiving
  - Easy model loading for inference

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                          │
│  CSV → Parquet Conversion + Timestamp Processing            │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              FEAST FEATURE STORE                             │
│  ┌──────────────────┐        ┌──────────────────┐          │
│  │  Offline Store   │        │   Online Store   │          │
│  │  (Historical)    │        │   (Real-time)    │          │
│  └──────────────────┘        └──────────────────┘          │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│            HYPERPARAMETER TUNING                             │
│  ┌──────────────────────────────────────────────┐           │
│  │  Grid Search:                                │           │
│  │  • Criterion: gini, entropy                  │           │
│  │  • Max Depth: 2, 3, 5, 10                   │           │
│  │  • Total Experiments: 8                      │           │
│  └──────────────────────────────────────────────┘           │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                MLFLOW TRACKING                               │
│  ┌──────────────────────────────────────────────┐           │
│  │  • Parameter Logging                         │           │
│  │  • Metric Tracking (Accuracy)                │           │
│  │  • Model Artifacts                           │           │
│  │  • Experiment Comparison                     │           │
│  └──────────────────────────────────────────────┘           │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              MODEL REGISTRY & DEPLOYMENT                     │
│  Best Model Selection → Registration → Production Stage     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠 Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Programming Language | 3.10+ |
| **Feast** | Feature Store | 0.53.0 |
| **MLflow** | Experiment Tracking & Model Registry | 3.5.1 |
| **scikit-learn** | Machine Learning Framework | 1.7.2 |
| **pandas** | Data Manipulation | 2.3.2 |
| **PyArrow** | Parquet File Handling | 16.0.0 |
| **NumPy** | Numerical Computing | 2.0.0 |

---

## 📁 Project Structure

```
21F1000690_Assignment_5_MAY_2025_MLOps/
│
├── 📓 week5.ipynb                          # Main Jupyter notebook
├── 📄 README.md                            # Project documentation
│
├── 📂 data/                                # Data directory
│   ├── iris_data_adapted_for_feast.csv    # Original CSV data
│   └── iris_data_adapted_for_feast.parquet # Converted Parquet data
│
└── 📂 iris_feast_repo/                    # Feast repository
    ├── feature_store.yaml                  # Feast configuration
    ├── features.py                         # Feature definitions
    └── data/                               # Feast registry data
```

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/21F1000690_Assignment_5_MAY_2025_MLOps.git
cd 21F1000690_Assignment_5_MAY_2025_MLOps
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install feast scikit-learn 'feast[gcp]' mlflow pandas pyarrow jupyter
```

### Step 4: Verify Installation

```bash
python -c "import feast, mlflow, sklearn; print('✅ All packages installed successfully!')"
```

---

## 🚀 Usage

### Quick Start

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook week5.ipynb
   ```

2. **Run All Cells**: Execute the notebook cells sequentially to:
   - Install dependencies
   - Convert data to Parquet format
   - Deploy Feast feature store
   - Run hyperparameter tuning
   - Register the best model
   - Deploy to production

### Step-by-Step Execution

#### 1️⃣ Data Preparation

```python
# Convert CSV to Parquet for Feast
csv_path = "data/iris_data_adapted_for_feast.csv"
df = pd.read_csv(csv_path)
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
df.to_parquet("data/iris_data_adapted_for_feast.parquet", index=False)
```

#### 2️⃣ Deploy Feast Feature Store

```python
# Apply Feast configuration
os.chdir(FEAST_REPO_PATH)
!feast apply
os.chdir(current_dir)
```

#### 3️⃣ Retrieve Features for Training

```python
# Get historical features from Feast offline store
fs = feast.FeatureStore(repo_path=FEAST_REPO_PATH)
training_df = fs.get_historical_features(
    entity_df=entity_df,
    features=["iris_stats:sepal_length", "iris_stats:sepal_width", 
              "iris_stats:petal_length", "iris_stats:petal_width"]
).to_df()
```

#### 4️⃣ Run Hyperparameter Tuning

```python
# Grid search with MLflow tracking
hyperparameters = {
    "max_depth": [2, 3, 5, 10],
    "criterion": ["gini", "entropy"]
}

for criterion in hyperparameters["criterion"]:
    for depth in hyperparameters["max_depth"]:
        with mlflow.start_run() as run:
            # Train model and log to MLflow
            mlflow.log_param("max_depth", depth)
            mlflow.log_param("criterion", criterion)
            # ... training code ...
            mlflow.sklearn.log_model(mod_dt, artifact_path="model")
```

#### 5️⃣ Register Best Model

```python
# Find and register the best performing model
best_run = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
).iloc[0]

model_version = mlflow.register_model(
    model_uri=best_model_uri,
    name="iris_classifier"
)
```

#### 6️⃣ Deploy to Production

```python
# Transition model to production stage
client.transition_model_version_stage(
    name="iris_classifier",
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True
)
```

---

## 🔄 Pipeline Workflow

### Phase 1: Data Pipeline
1. Load Iris dataset with temporal information
2. Convert to Parquet format for efficient storage
3. Configure Feast feature store
4. Deploy feature definitions

### Phase 2: Training Pipeline
1. Retrieve historical features from Feast offline store
2. Split data into training and testing sets (60/40 split)
3. Perform stratified sampling to maintain class distribution

### Phase 3: Experimentation Pipeline
1. Set up MLflow experiment tracking
2. Execute grid search over hyperparameter space:
   - **Criterion**: `gini`, `entropy`
   - **Max Depth**: `2`, `3`, `5`, `10`
3. Log parameters, metrics, and models for each run
4. Total experiments: **8 runs**

### Phase 4: Model Selection & Deployment
1. Query MLflow for best performing model (highest accuracy)
2. Register model in MLflow Model Registry
3. Transition to Production stage
4. Archive previous production models

### Phase 5: Inference Pipeline
1. Load production model from registry
2. Retrieve online features from Feast
3. Generate predictions
4. Monitor model performance

---

