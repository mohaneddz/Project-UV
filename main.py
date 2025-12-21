"""Interactive CLI for UV prediction model training and evaluation."""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Import modules
import preprocess
import plot
import ts
import ml
import dl
import boosters


# Model categories and available models
MODEL_CATEGORIES = {
    '1': {
        'name': 'Time Series Models',
        'module': ts,
        'models': ts.MODELS,
        'requires_features': False
    },
    '2': {
        'name': 'Machine Learning Models',
        'module': ml,
        'models': ml.MODELS,
        'requires_features': True
    },
    '3': {
        'name': 'Deep Learning Models',
        'module': dl,
        'models': dl.MODELS,
        'requires_features': False
    },
    '4': {
        'name': 'Boosting Models',
        'module': boosters,
        'models': boosters.MODELS,
        'requires_features': True
    }
}


def print_header():
    """Print application header."""
    print("\n" + "="*70)
    print(" UV PREDICTION MODEL TRAINING SYSTEM")
    print("="*70)


def print_main_menu():
    """Print main menu options."""
    print("\n--- MAIN MENU ---")
    print("1. Train Time Series Models")
    print("2. Train Machine Learning Models")
    print("3. Train Deep Learning Models")
    print("4. Train Boosting Models")
    print("5. Train Multiple Models")
    print("6. Load Saved Model")
    print("7. Exit")
    print("-" * 30)


def print_model_menu(category_key):
    """Print models in a category."""
    category = MODEL_CATEGORIES[category_key]
    print(f"\n--- {category['name']} ---")
    
    models = category['models']
    if not models:
        print("No models available (missing dependencies)")
        return []
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    print(f"{len(models) + 1}. All models")
    print(f"{len(models) + 2}. Back to main menu")
    print("-" * 30)
    
    return models


def get_user_choice(prompt, max_value):
    """Get and validate user choice."""
    while True:
        try:
            choice = input(prompt)
            choice_int = int(choice)
            if 1 <= choice_int <= max_value:
                return choice_int
            else:
                print(f"Please enter a number between 1 and {max_value}")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def get_yes_no(prompt):
    """Get yes/no response."""
    while True:
        response = input(f"{prompt} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")


def load_data(filepath):
    """Load and preprocess data."""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        sys.exit(1)
    
    train_df, test_df, full_df = preprocess.full_pipeline(filepath)
    
    print(f"\nData loaded successfully!")
    print(f"Full dataset: {len(full_df)} samples")
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Variables: {len(full_df.columns)}")
    print(f"Columns: {', '.join(full_df.columns.tolist()[:5])}...")
    
    return train_df, test_df, full_df


def train_ts_model(train_df, test_df, model_type):
    """Train time series model."""
    print(f"\n{'='*70}")
    print(f"TRAINING {model_type} MODEL")
    print(f"{'='*70}")
    
    # Get default parameters
    params = ts.get_default_params(model_type)
    
    # Train and evaluate
    metrics_df, predictions_dict = ts.train_and_evaluate(
        train_df, test_df, model_type, params
    )
    
    return metrics_df, predictions_dict


def train_ml_model(train_df, test_df, full_df, model_type):
    """Train machine learning model."""
    print(f"\n{'='*70}")
    print(f"TRAINING {model_type} MODEL")
    print(f"{'='*70}")
    
    # Prepare features
    print("\nPreparing features...")
    target_cols = train_df.columns.tolist()
    
    X_train, X_test, y_train, y_test, feature_names = preprocess.prepare_for_ml(
        full_df, target_cols, feature_cols=None, test_days=len(test_df)
    )
    
    print(f"Features created: {len(feature_names)}")
    
    # Get default parameters
    params = ml.get_default_params(model_type)
    
    # Train and evaluate
    metrics_df, predictions_dict, model = ml.train_and_evaluate(
        X_train, X_test, y_train, y_test, model_type, params
    )
    
    # Plot feature importance if available
    importances = ml.get_feature_importance(model, feature_names, model_type)
    if importances is not None:
        model_folder = os.path.join('gen', model_type.lower())
        os.makedirs(model_folder, exist_ok=True)
        plot.plot_feature_importance(importances, feature_names, model_type, 
                                     save_path=f'{model_folder}/{model_type}_feature_importance.png')
    
    return metrics_df, predictions_dict


def train_dl_model(train_df, test_df, model_type):
    """Train deep learning model."""
    print(f"\n{'='*70}")
    print(f"TRAINING {model_type} MODEL")
    print(f"{'='*70}")
    
    # Get default parameters
    params = dl.get_default_params(model_type)
    
    # Train and evaluate
    metrics_df, predictions_dict, model, history = dl.train_and_evaluate(
        train_df, test_df, model_type, params, 
        lookback=30, epochs=50, batch_size=32
    )
    
    # Plot training history
    model_folder = os.path.join('gen', model_type.lower())
    os.makedirs(model_folder, exist_ok=True)
    plot.plot_training_history(history, save_path=f'{model_folder}/{model_type}_training_history.png', model_name=model_type)
    
    return metrics_df, predictions_dict


def train_booster_model(train_df, test_df, full_df, model_type):
    """Train boosting model."""
    print(f"\n{'='*70}")
    print(f"TRAINING {model_type} MODEL")
    print(f"{'='*70}")
    
    # Prepare features
    print("\nPreparing features...")
    target_cols = train_df.columns.tolist()
    
    X_train, X_test, y_train, y_test, feature_names = preprocess.prepare_for_ml(
        full_df, target_cols, feature_cols=None, test_days=len(test_df)
    )
    
    print(f"Features created: {len(feature_names)}")
    
    # Get default parameters
    params = boosters.get_default_params(model_type)
    
    # Train and evaluate
    metrics_df, predictions_dict, model = boosters.train_and_evaluate(
        X_train, X_test, y_train, y_test, model_type, params
    )
    
    # Plot feature importance if available
    importances = boosters.get_feature_importance(model, feature_names, model_type)
    if importances is not None:
        model_folder = os.path.join('gen', model_type.lower())
        os.makedirs(model_folder, exist_ok=True)
        plot.plot_feature_importance(importances, feature_names, model_type,
                                     save_path=f'{model_folder}/{model_type}_feature_importance.png')
    
    return metrics_df, predictions_dict


def visualize_results(model_type, metrics_df, predictions_dict):
    """Visualize and save results."""
    print(f"\n{'='*70}")
    print(f"RESULTS FOR {model_type}")
    print(f"{'='*70}")
    
    # Print metrics
    plot.print_metrics_summary(metrics_df)
    
    # Save all plots
    plot.save_all_plots(model_type, predictions_dict, metrics_df)


def train_single_model(category_key, model_choice, train_df, test_df, full_df):
    """Train a single model."""
    category = MODEL_CATEGORIES[category_key]
    models = category['models']
    
    if model_choice > len(models):
        return None
    
    model_type = models[model_choice - 1]
    
    # Train based on category
    if category_key == '1':  # Time Series
        metrics_df, predictions_dict = train_ts_model(train_df, test_df, model_type)
    elif category_key == '2':  # Machine Learning
        metrics_df, predictions_dict = train_ml_model(train_df, test_df, full_df, model_type)
    elif category_key == '3':  # Deep Learning
        metrics_df, predictions_dict = train_dl_model(train_df, test_df, model_type)
    elif category_key == '4':  # Boosting
        metrics_df, predictions_dict = train_booster_model(train_df, test_df, full_df, model_type)
    
    # Visualize
    visualize_results(model_type, metrics_df, predictions_dict)
    
    return model_type


def train_all_in_category(category_key, train_df, test_df, full_df):
    """Train all models in a category."""
    category = MODEL_CATEGORIES[category_key]
    models = category['models']
    
    if not models:
        print("No models available in this category.")
        return
    
    print(f"\n{'='*70}")
    print(f"TRAINING ALL {category['name'].upper()}")
    print(f"{'='*70}")
    
    all_metrics = []
    
    for model_type in models:
        try:
            # Train based on category
            if category_key == '1':  # Time Series
                metrics_df, predictions_dict = train_ts_model(train_df, test_df, model_type)
            elif category_key == '2':  # Machine Learning
                metrics_df, predictions_dict = train_ml_model(train_df, test_df, full_df, model_type)
            elif category_key == '3':  # Deep Learning
                metrics_df, predictions_dict = train_dl_model(train_df, test_df, model_type)
            elif category_key == '4':  # Boosting
                metrics_df, predictions_dict = train_booster_model(train_df, test_df, full_df, model_type)
            
            # Visualize
            visualize_results(model_type, metrics_df, predictions_dict)
            
            # Collect metrics
            metrics_df['Model'] = model_type
            all_metrics.append(metrics_df)
            
        except Exception as e:
            print(f"\nError training {model_type}: {e}")
            continue
    
    # Print comparison
    if all_metrics:
        print(f"\n{'='*70}")
        print("COMPARISON OF ALL MODELS")
        print(f"{'='*70}")
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        print(combined_metrics.to_string(index=False))


def train_multiple_models(train_df, test_df, full_df):
    """Train multiple selected models."""
    print("\n--- TRAIN MULTIPLE MODELS ---")
    print("Select models from different categories")
    
    selected_models = []
    
    while True:
        print("\nSelect a category to add models from:")
        for key, cat in MODEL_CATEGORIES.items():
            print(f"{key}. {cat['name']}")
        print("5. Done selecting")
        
        choice = get_user_choice("Enter choice: ", 5)
        
        if choice == 5:
            break
        
        category_key = str(choice)
        category = MODEL_CATEGORIES[category_key]
        models = print_model_menu(category_key)
        
        if not models:
            continue
        
        model_choice = get_user_choice(f"Select model (1-{len(models) + 2}): ", len(models) + 2)
        
        if model_choice == len(models) + 2:  # Back
            continue
        elif model_choice == len(models) + 1:  # All
            for model in models:
                selected_models.append((category_key, model))
        else:
            selected_models.append((category_key, models[model_choice - 1]))
    
    if not selected_models:
        print("No models selected.")
        return
    
    print(f"\nSelected {len(selected_models)} model(s):")
    for cat_key, model in selected_models:
        print(f"  - {model} ({MODEL_CATEGORIES[cat_key]['name']})")
    
    if not get_yes_no("\nProceed with training?"):
        return
    
    # Train all selected models
    all_metrics = []
    
    for category_key, model_type in selected_models:
        try:
            if category_key == '1':  # Time Series
                metrics_df, predictions_dict = train_ts_model(train_df, test_df, model_type)
            elif category_key == '2':  # Machine Learning
                metrics_df, predictions_dict = train_ml_model(train_df, test_df, full_df, model_type)
            elif category_key == '3':  # Deep Learning
                metrics_df, predictions_dict = train_dl_model(train_df, test_df, model_type)
            elif category_key == '4':  # Boosting
                metrics_df, predictions_dict = train_booster_model(train_df, test_df, full_df, model_type)
            
            visualize_results(model_type, metrics_df, predictions_dict)
            
            metrics_df['Model'] = model_type
            all_metrics.append(metrics_df)
            
        except Exception as e:
            print(f"\nError training {model_type}: {e}")
            continue
    
    # Print final comparison
    if all_metrics:
        print(f"\n{'='*70}")
        print("FINAL COMPARISON OF ALL MODELS")
        print(f"{'='*70}")
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        print(combined_metrics.to_string(index=False))


def load_saved_model():
    """Load and display information about a saved model."""
    print(f"\n{'='*70}")
    print("LOAD SAVED MODEL")
    print(f"{'='*70}")
    
    # List available model directories
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print(f"\nNo models directory found. Train a model first.")
        return
    
    # Find all subdirectories in models/
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    if not subdirs:
        print(f"\nNo saved models found. Train a model first.")
        return
    
    print("\nAvailable model types:")
    for i, subdir in enumerate(subdirs, 1):
        # Count model files in subdirectory
        model_dir = os.path.join(models_dir, subdir)
        files = os.listdir(model_dir)
        model_count = len([f for f in files if f.endswith('.pkl') or f.endswith('.h5')])
        print(f"{i}. {subdir.upper()} ({model_count} file(s))")
    
    print(f"{len(subdirs) + 1}. Back to main menu")
    
    choice = get_user_choice(f"\nSelect model type (1-{len(subdirs) + 1}): ", len(subdirs) + 1)
    
    if choice == len(subdirs) + 1:
        return
    
    selected_model = subdirs[choice - 1].upper()
    
    # Determine model category
    if selected_model in ts.MODELS:
        category = 'ts'
    elif selected_model in ml.MODELS:
        category = 'ml'
    elif selected_model in dl.MODELS:
        category = 'dl'
    elif selected_model in boosters.MODELS:
        category = 'boosters'
    else:
        print(f"\nUnknown model type: {selected_model}")
        return
    
    # Load the model
    try:
        print(f"\nLoading {selected_model} model...")
        
        if category == 'ts':
            # For time series, list all variable models
            model_dir = os.path.join(models_dir, selected_model.lower())
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            print(f"\nFound {len(model_files)} trained variable models:")
            for f in model_files:
                print(f"  - {f}")
            print("\nUse ts.load_model(filepath) to load individual models")
        
        elif category == 'ml':
            model = ml.load_model(selected_model)
            print(f"\n✓ Model loaded successfully!")
            print(f"Model type: {type(model).__name__}")
            
        elif category == 'dl':
            model, scaler = dl.load_model(selected_model)
            print(f"\n✓ Model and scaler loaded successfully!")
            print(f"Model type: {type(model).__name__}")
            print(f"Scaler type: {type(scaler).__name__}")
        
        elif category == 'boosters':
            model = boosters.load_model(selected_model)
            print(f"\n✓ Model loaded successfully!")
            print(f"Model type: {type(model).__name__}")
        
        print("\nModel is now loaded and ready for predictions.")
        print("Note: To use the model for predictions, call the appropriate predict function.")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")


def main():
    """Main application loop."""
    print_header()
    
    # Load data
    data_path = 'data/UV-Tunisia.csv'
    train_df, test_df, full_df = load_data(data_path)
    
    # Ensure output directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('gen', exist_ok=True)
    
    # Main loop
    while True:
        print_main_menu()
        main_choice = get_user_choice("Enter your choice: ", 7)
        
        if main_choice == 7:  # Exit
            print("\nThank you for using UV Prediction System!")
            print("="*70)
            break
        
        elif main_choice == 6:  # Load saved model
            load_saved_model()
        
        elif main_choice == 5:  # Multiple models
            train_multiple_models(train_df, test_df, full_df)
        
        else:  # Single category
            category_key = str(main_choice)
            models = print_model_menu(category_key)
            
            if not models:
                input("\nPress Enter to continue...")
                continue
            
            model_choice = get_user_choice(f"Enter your choice (1-{len(models) + 2}): ", len(models) + 2)
            
            if model_choice == len(models) + 2:  # Back
                continue
            elif model_choice == len(models) + 1:  # All models
                train_all_in_category(category_key, train_df, test_df, full_df)
            else:  # Single model
                train_single_model(category_key, model_choice, train_df, test_df, full_df)
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
