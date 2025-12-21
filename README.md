# UV Prediction System

Comprehensive time series prediction system for UV radiation data in Tunisia, featuring 20+ machine learning, deep learning, and statistical models with an interactive CLI interface.

## 📁 Project Structure

```
Project UV/
├── data/
│   └── UV-Tunisia.csv          # Main dataset (8524 samples, 16 UV variables)
├── models/                     # Saved trained models (.pkl, .h5)
├── gen/                        # Generated plots and visualizations
├── preprocess.py               # Data preprocessing utilities
├── plot.py                     # Visualization functions
├── ts.py                       # Time series models
├── ml.py                       # Machine learning models
├── dl.py                       # Deep learning models
├── boosters.py                 # Boosting models
├── main.py                     # Interactive CLI interface
├── requirements.txt            # Python dependencies
└── MODELS_LIST.txt            # Complete list of available models
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core dependencies (required)
pip install pandas numpy matplotlib scikit-learn statsmodels scipy

# Optional: Machine Learning models
pip install xgboost lightgbm

# Optional: Deep Learning models
pip install tensorflow

# Optional: Boosting models
pip install catboost

# Or install all at once:
pip install -r requirements.txt
```

### 2. Run the Interactive System

```bash
python main.py
```

## 📊 Available Models

### Time Series Models (5)
- **SARIMA** - Seasonal ARIMA with weekly patterns
- **ARIMA** - Classic autoregressive model
- **ETS** - Exponential Smoothing (Holt-Winters)
- **NAIVE** - Simple persistence baseline
- **SEASONAL_NAIVE** - Seasonal persistence

### Machine Learning Models (7)
- **RANDOM_FOREST** - Ensemble of decision trees
- **XGBOOST** - Extreme Gradient Boosting
- **LIGHTGBM** - Fast gradient boosting framework
- **LINEAR_REGRESSION** - Simple linear model
- **RIDGE** - L2 regularized regression
- **LASSO** - L1 regularized regression
- **SVR** - Support Vector Regression

### Deep Learning Models (5)
- **LSTM** - Long Short-Term Memory network
- **GRU** - Gated Recurrent Unit
- **BILSTM** - Bidirectional LSTM
- **DEEP_LSTM** - Multi-layer LSTM
- **DEEP_GRU** - Multi-layer GRU

### Boosting Models (3)
- **CATBOOST** - Categorical boosting
- **ADABOOST** - Adaptive boosting
- **GRADIENT_BOOSTING** - Gradient boosting trees

## 💡 Usage Examples

### Train a Single Model
1. Run `python main.py`
2. Select category (e.g., "1" for Time Series)
3. Select model (e.g., "1" for SARIMA)
4. View results and plots

### Train All Models in a Category
1. Run `python main.py`
2. Select category
3. Choose "All models" option
4. Compare results

### Train Multiple Custom Models
1. Run `python main.py`
2. Select "5. Train Multiple Models"
3. Add models from different categories
4. Compare all results

## 📈 Features

### Data Preprocessing
- Automatic date conversion and numeric casting
- Missing value interpolation
- Feature engineering (lags, rolling statistics, time features)
- Train/test splitting (last 30 days for testing)

### Visualization
- 4x4 grid plots showing forecasts vs actuals
- Performance metrics bar charts (RMSE, MAE)
- Feature importance plots (for tree-based models)
- Training history plots (for deep learning)
- Residual analysis plots
- All plots auto-saved to `gen/` folder

### Model Management
- Automatic model saving to `models/` folder
- Clear naming convention for models and plots
- Consistent evaluation metrics across all models
- Support for multi-target prediction (all 16 UV variables)

## 🔧 Module Details

### `preprocess.py`
```python
# Load data
train_df, test_df, full_df = preprocess.full_pipeline('data/UV-Tunisia.csv')

# Create features for ML
X_train, X_test, y_train, y_test, features = preprocess.prepare_for_ml(
    full_df, target_cols, test_days=30
)
```

### `ts.py` - Time Series
```python
# Train SARIMA for all variables
metrics_df, predictions_dict = ts.train_and_evaluate(
    train_df, test_df, model_type='SARIMA'
)
```

### `ml.py` - Machine Learning
```python
# Train Random Forest with features
metrics_df, predictions_dict, model = ml.train_and_evaluate(
    X_train, X_test, y_train, y_test, model_type='RANDOM_FOREST'
)
```

### `dl.py` - Deep Learning
```python
# Train LSTM
metrics_df, predictions_dict, model, history = dl.train_and_evaluate(
    train_df, test_df, model_type='LSTM', lookback=30, epochs=50
)
```

### `plot.py` - Visualization
```python
# Save all standard plots
plot.save_all_plots('SARIMA', predictions_dict, metrics_df)

# Plot metrics summary
plot.print_metrics_summary(metrics_df)
```

## 📊 Dataset

**UV-Tunisia.csv** contains:
- **8,524 samples** from July 2002 onwards
- **16 variables**: UV indices, errors, ozone, CMF
- **Daily frequency** with 7-day seasonality
- Train/test split: Last 30 days for testing

## 🎯 Performance Metrics

All models evaluated using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

Metrics calculated per variable and averaged across all 16 UV indicators.

## 📝 Output Files

### Models Folder (`models/`)
- `SARIMA_model.pkl`
- `RANDOM_FOREST_model.pkl`
- `LSTM_model.h5`
- `LSTM_scaler.pkl`
- etc.

### Generated Plots (`gen/`)
- `SARIMA_forecast_grid_20251207_143022.png`
- `SARIMA_metrics_20251207_143022.png`
- `RANDOM_FOREST_feature_importance.png`
- `LSTM_training_history.png`
- etc.

## 🛠️ Customization

### Adjust Test Period
Edit in `preprocess.py`:
```python
def split_train_test(df, test_days=30):  # Change to your preference
```

### Modify Model Parameters
Each model module has `get_default_params()` - edit defaults or pass custom params.

### Add New Models
1. Add model function in appropriate module
2. Update `MODELS` list
3. Add to `train_and_evaluate()` function

## 📚 Dependencies

**Core (Required):**
- pandas, numpy, matplotlib, scikit-learn, statsmodels, scipy

**Optional:**
- xgboost, lightgbm (ML models)
- tensorflow (DL models)
- catboost (boosting)

Missing optional dependencies will disable specific models but won't break the system.

## 🤝 Contributing

This is an academic project. Feel free to extend with:
- New model types
- Advanced feature engineering
- Hyperparameter optimization
- Cross-validation
- Model ensembling

## 📄 License

Academic/Educational use

## 👤 Author

Data Mining Course Project - UV Radiation Prediction in Tunisia

---

**Note:** First run will take longer as models are trained and saved. Subsequent runs can load saved models for faster predictions.
