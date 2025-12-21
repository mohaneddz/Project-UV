"""
LEGACY SCRIPT - KEPT FOR REFERENCE
===================================
This script has been superseded by the modular structure.

Use the new system:
    python main.py

For manual SARIMA training, use:
    import preprocess
    import ts
    import plot
    
    train_df, test_df, full_df = preprocess.full_pipeline('data/UV-Tunisia.csv')
    metrics_df, predictions_dict = ts.train_and_evaluate(train_df, test_df, 'SARIMA')
    plot.save_all_plots('SARIMA', predictions_dict, metrics_df)
"""

# Quick example using new modules
if __name__ == "__main__":
    import preprocess
    import ts
    import plot
    
    print("Training SARIMA model using new modular structure...\n")
    
    # Load data
    train_df, test_df, full_df = preprocess.full_pipeline('data/UV-Tunisia.csv')
    
    # Train SARIMA
    metrics_df, predictions_dict = ts.train_and_evaluate(
        train_df, test_df, 
        model_type='SARIMA'
    )
    
    # Visualize results
    plot.save_all_plots('SARIMA', predictions_dict, metrics_df)
    plot.print_metrics_summary(metrics_df)
    
    print("\n" + "="*70)
    print("For the interactive system with all models, run: python main.py")
    print("="*70)