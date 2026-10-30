import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_cleaning import load_and_clean_data
from src.feature_engineering import engineer_features
from src.visualization import save_eda_plots, save_model_plots

def run_pipeline(data_path: str, output_dir: str = "artifacts", figures_dir: str = "reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("[1/5] Loading and cleaning raw dataset...")
    df = load_and_clean_data(data_path)
    processed_csv = "data/processed/cleaned_zomato.csv"
    os.makedirs(os.path.dirname(processed_csv), exist_ok=True)
    df.to_csv(processed_csv, index=False)
    print(f"      Saved cleaned dataset ({len(df):,} rows) to {processed_csv}")
    
    print("[2/5] Generating and saving EDA figures...")
    save_eda_plots(df, figures_dir)
    print(f"      Saved figures to {figures_dir}")
    
    print("[3/5] Engineering feature matrix...")
    X, y, feature_names = engineer_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"      Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    print("[4/5] Training and benchmarking regression models...")
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"      Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results.append({
            "Model": name,
            "R2_Score": round(float(r2), 4),
            "RMSE": round(float(rmse), 4),
            "MAE": round(float(mae), 4)
        })
        trained_models[name] = model
        print(f"        -> R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        
    results_df = pd.DataFrame(results).sort_values("R2_Score", ascending=False).reset_index(drop=True)
    comparison_path = os.path.join(output_dir, "model_comparison.csv")
    results_df.to_csv(comparison_path, index=False)
    print(f"      Saved model benchmarks to {comparison_path}")
    
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"      Best Model selected: {best_model_name}")
    
    # Feature importance calculation
    fi_df = pd.DataFrame()
    if hasattr(best_model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        fi_path = os.path.join(output_dir, "feature_importance.csv")
        fi_df.to_csv(fi_path, index=False)
        print(f"      Saved feature importance to {fi_path}")
        
    print("[5/5] Exporting model bundle & comparison charts...")
    save_model_plots(results_df, fi_df, figures_dir)
    
    bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_names": feature_names,
        "metrics": results_df.to_dict(orient="records")
    }
    bundle_path = os.path.join(output_dir, "models", "zomato_rating_model.joblib")
    joblib.dump(bundle, bundle_path)
    print(f"      Model bundle successfully saved to {bundle_path}!")
    return results_df, bundle_path

if __name__ == "__main__":
    raw_data = "data/raw/zomato.csv"
    run_pipeline(raw_data)
