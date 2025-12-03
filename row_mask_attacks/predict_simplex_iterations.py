"""Build a predictive model for solver simplex iterations."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

def build_prediction_model():
    """Build a model to predict solver_metrics_simplex_iterations."""
    
    # Load gathered results
    results_file = Path('./results/row_mask_attacks/result.parquet')
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run gather.py first to create the parquet file.")
        return
    
    df = pd.read_parquet(results_file)
    
    # Filter for Gurobi solver data with simplex iterations
    if 'solver_metrics_solver' in df.columns:
        df = df[df['solver_metrics_solver'] == 'gurobi'].copy()
    
    # Define features and target
    feature_cols = ['solve_type', 'nrows', 'mask_size', 'nunique', 'noise', 'nqi', 'vals_per_qi']
    target_col = 'solver_metrics_simplex_iterations'
    
    # Check if all columns exist
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Remove rows with missing target values
    df_clean = df[feature_cols + [target_col]].dropna()
    
    print(f"Total rows: {len(df)}")
    print(f"Rows with complete data: {len(df_clean)}")
    
    if len(df_clean) < 10:
        print("Not enough data to build a model.")
        return
    
    # Prepare features
    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].values
    
    # Handle categorical variables (solve_type)
    if X['solve_type'].dtype == 'object':
        X = pd.get_dummies(X, columns=['solve_type'], drop_first=True)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns after encoding:")
    print(list(X.columns))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create output directory
    plots_dir = Path('./results/row_mask_attacks/prediction_model')
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Train multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    
    results = {}
    
    print("\n=== MODEL PERFORMANCE ===\n")
    
    for name, model in models.items():
        # Train
        if 'Linear' in name or 'Ridge' in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"{name}:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.3f}")
        print()
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model_info = results[best_model_name]
    
    print(f"Best model: {best_model_name} (R² = {best_model_info['r2']:.3f})")
    
    # Plot: Predicted vs Actual for best model
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, best_model_info['predictions'], alpha=0.6, s=50, 
                color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line
    min_val = min(y_test.min(), best_model_info['predictions'].min())
    max_val = max(y_test.max(), best_model_info['predictions'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Actual Simplex Iterations')
    plt.ylabel('Predicted Simplex Iterations')
    plt.title(f'{best_model_name}: Predicted vs Actual\nR² = {best_model_info["r2"]:.3f}, RMSE = {best_model_info["rmse"]:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_file = plots_dir / 'predicted_vs_actual.png'
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"\nSaved plot: {plot_file}")
    
    # Plot: Residuals
    residuals = y_test - best_model_info['predictions']
    plt.figure(figsize=(10, 6))
    plt.scatter(best_model_info['predictions'], residuals, alpha=0.6, s=50,
                color='steelblue', edgecolors='black', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Simplex Iterations')
    plt.ylabel('Residuals')
    plt.title(f'{best_model_name}: Residual Plot')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_file = plots_dir / 'residuals.png'
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_file}")
    
    # Feature importance (for tree-based models)
    if 'Random Forest' in best_model_name or 'Gradient Boosting' in best_model_name:
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model_info['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n=== FEATURE IMPORTANCE ===\n")
        print(feature_importance.to_string(index=False))
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'{best_model_name}: Feature Importance')
        plt.tight_layout()
        plot_file = plots_dir / 'feature_importance.png'
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"\nSaved plot: {plot_file}")
    
    # Model comparison plot
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    rmse_scores = [results[name]['rmse'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, r2_scores, width, label='R²', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE', color='coral', alpha=0.8)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('R² Score', color='steelblue')
    ax2.set_ylabel('RMSE', color='coral')
    ax1.set_title('Model Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plot_file = plots_dir / 'model_comparison.png'
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_file}")
    
    print(f"\n\nAll plots saved to: {plots_dir}")

if __name__ == '__main__':
    build_prediction_model()
