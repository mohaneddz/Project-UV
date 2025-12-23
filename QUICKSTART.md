# Quick Start Guide - UV Prediction System

## Installation (5 minutes)

### Step 1: Install Core Dependencies
```bash
pip install pandas numpy matplotlib scikit-learn statsmodels scipy
```

### Step 2: Install Optional Packages (for more models)
```bash
# For XGBoost and LightGBM
pip install xgboost lightgbm

# For Deep Learning models
pip install tensorflow

# For CatBoost
pip install catboost
```

**OR** install everything at once:
```bash
pip install -r requirements.txt
```

## Running the System

### Interactive Mode (Recommended)
```bash
python main.py
```

This launches an interactive menu where you can:
- Select individual models to train
- Train all models in a category
- Compare multiple models
- View results and plots automatically

### Example Session
```
1. Select "1" for Time Series Models
2. Select "1" for SARIMA model
3. Wait for training (30 seconds - 2 minutes)
4. View metrics printed to console
5. Check gen/ folder for plots
6. Check models/ folder for saved models
```

## Quick Examples

### Example 1: Train SARIMA Only
```python
import preprocess
import ts
import plot

# Load data
train_df, test_df, full_df = preprocess.full_pipeline('data/UV-Algeria.csv')

# Train SARIMA
metrics_df, predictions_dict = ts.train_and_evaluate(train_df, test_df, 'SARIMA')

# Visualize
plot.save_all_plots('SARIMA', predictions_dict, metrics_df)
```

### Example 2: Train Random Forest
```python
import preprocess
import ml
import plot

# Load data
train_df, test_df, full_df = preprocess.full_pipeline('data/UV-Algeria.csv')

# Prepare features
target_cols = train_df.columns.tolist()
X_train, X_test, y_train, y_test, features = preprocess.prepare_for_ml(
    full_df, target_cols, test_days=30
)

# Train
metrics_df, predictions_dict, model = ml.train_and_evaluate(
    X_train, X_test, y_train, y_test, 'RANDOM_FOREST'
)

# Visualize
plot.save_all_plots('RANDOM_FOREST', predictions_dict, metrics_df)
```

### Example 3: Train LSTM
```python
import preprocess
import dl
import plot

# Load data
train_df, test_df, full_df = preprocess.full_pipeline('data/UV-Algeria.csv')

# Train
metrics_df, predictions_dict, model, history = dl.train_and_evaluate(
    train_df, test_df, 'LSTM', lookback=30, epochs=50
)

# Visualize
plot.save_all_plots('LSTM', predictions_dict, metrics_df)
```

## What You Get

After training any model, you'll find:

### In Console
- Training progress
- Performance metrics (RMSE, MAE per variable)
- Summary statistics

### In `gen/` Folder
- Forecast grid plots (4x4 showing all variables)
- Metrics comparison bar charts
- Feature importance plots (for tree models)
- Training history plots (for DL models)

### In `models/` Folder
- Saved model files (.pkl for sklearn/stats, .h5 for Keras)
- Scaler files (for DL models)

## File Structure After First Run

```
Project UV/
├── data/
│   └── UV-Tunisia.csv
├── models/
│   ├── SARIMA_UVIEF.pkl
│   ├── RANDOM_FOREST_model.pkl
│   ├── LSTM_model.h5
│   └── ...
├── gen/
│   ├── SARIMA_forecast_grid_20251207_143022.png
│   ├── SARIMA_metrics_20251207_143022.png
│   └── ...
└── [Python modules...]
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'xgboost'"
- XGBoost is optional. Either install it (`pip install xgboost`) or just skip XGBoost models.

### "TensorFlow not found"
- Deep Learning models require TensorFlow. Install with `pip install tensorflow` or skip DL models.

### "Data file not found"
- Make sure `data/UV-Algeria.csv` exists
- Check the file was moved correctly from root

### Out of Memory
- Reduce batch size for DL models
- Train fewer models at once
- Use simpler models (ARIMA, Linear Regression)

## Recommended First Run

1. **Start Simple**: Train SARIMA first (fast, reliable)
   ```bash
   python main.py
   # Select: 1 → 1
   ```

2. **Try ML**: Train Random Forest
   ```bash
   python main.py
   # Select: 2 → 1
   ```

3. **Compare All Time Series**: 
   ```bash
   python main.py
   # Select: 1 → 6 (All models)
   ```

## Performance Tips

- **Time Series models**: Fast (30 sec - 2 min)
- **ML models**: Medium (1-5 min)
- **DL models**: Slow (5-20 min depending on epochs)
- **Boosting models**: Medium-Fast (1-3 min)

## Next Steps

1. Run `python main.py` and explore the interactive menu
2. Compare different model categories
3. Check generated plots in `gen/`
4. Review metrics to find best performing models
5. Use saved models for future predictions

## Getting Help

- Check `MODELS_LIST.txt` for all available models
- Read `README.md` for detailed documentation
- Review module docstrings for function details

---

**Happy Modeling! 🎯📊**
