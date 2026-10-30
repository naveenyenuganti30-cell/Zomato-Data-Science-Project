import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List

def group_low_frequency(series: pd.Series, threshold: int = 300, replacement: str = "Others") -> pd.Series:
    counts = series.value_counts()
    frequent = counts[counts >= threshold].index
    return series.apply(lambda x: x if x in frequent else replacement)

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Transforms cleaned data into ML-ready feature matrix X and target y.
    """
    data = df.copy()
    
    # 1. Number of cuisines offered
    data["num_cuisines"] = data["cuisines"].apply(lambda x: len([c.strip() for c in x.split(",")]) if pd.notna(x) else 1)
    
    # 2. Check if popular cuisines are present
    popular_cuisines = ["North Indian", "South Indian", "Chinese", "Continental", "Italian", "Cafe", "Fast Food", "Biryani", "Desserts", "Beverages"]
    for cuisine in popular_cuisines:
        col_name = f"is_{cuisine.lower().replace(' ', '_')}"
        data[col_name] = data["cuisines"].apply(lambda x: 1 if cuisine.lower() in str(x).lower() else 0)
        
    # 3. Simplify locations and rest_type to prevent extreme dimensionality
    data["location_grouped"] = group_low_frequency(data["location"], threshold=250)
    data["rest_type_grouped"] = group_low_frequency(data["rest_type"], threshold=250)
    
    # 4. Log-transform skewed numerical features
    data["log_votes"] = np.log1p(data["votes"])
    data["log_cost"] = np.log1p(data["cost_for_two"])
    
    # 5. Cost-per-person tier
    data["is_fine_dining"] = (data["cost_for_two"] >= 1500).astype(int)
    
    # 6. One-hot encode categorical features
    cat_cols = ["location_grouped", "rest_type_grouped", "type"]
    encoded = pd.get_dummies(data[cat_cols], drop_first=True, dtype=int)
    
    # Features to select
    numeric_features = [
        "online_order", "book_table", "cost_for_two", "log_cost", 
        "votes", "log_votes", "num_cuisines", "is_fine_dining"
    ]
    cuisine_features = [f"is_{c.lower().replace(' ', '_')}" for c in popular_cuisines]
    
    X = pd.concat([data[numeric_features + cuisine_features], encoded], axis=1)
    y = data["rate"]
    
    feature_names = X.columns.tolist()
    return X, y, feature_names
