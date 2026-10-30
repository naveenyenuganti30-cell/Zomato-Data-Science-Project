import os
import joblib
import pandas as pd
import numpy as np

def load_predictor(bundle_path: str = "artifacts/models/zomato_rating_model.joblib"):
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Model bundle not found at {bundle_path}. Please train the model first.")
    bundle = joblib.load(bundle_path)
    return bundle

def predict_rating(restaurant_info: dict, bundle_path: str = "artifacts/models/zomato_rating_model.joblib") -> float:
    """
    Takes a dict of restaurant features and predicts rating out of 5.0.
    """
    bundle = load_predictor(bundle_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    
    # Create single-row dataframe initialized to 0
    input_row = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Map provided numeric features
    cost = restaurant_info.get("cost_for_two", 600)
    votes = restaurant_info.get("votes", 100)
    
    input_row["cost_for_two"] = cost
    input_row["log_cost"] = np.log1p(cost)
    input_row["votes"] = votes
    input_row["log_votes"] = np.log1p(votes)
    input_row["online_order"] = 1 if restaurant_info.get("online_order", True) else 0
    input_row["book_table"] = 1 if restaurant_info.get("book_table", False) else 0
    input_row["is_fine_dining"] = 1 if cost >= 1500 else 0
    input_row["num_cuisines"] = len(restaurant_info.get("cuisines", ["North Indian"]))
    
    # Map cuisines
    for cuisine in restaurant_info.get("cuisines", []):
        col = f"is_{cuisine.lower().replace(' ', '_')}"
        if col in input_row.columns:
            input_row[col] = 1
            
    # Map location
    loc_col = f"location_grouped_{restaurant_info.get('location', 'Others')}"
    if loc_col in input_row.columns:
        input_row[loc_col] = 1
        
    # Map rest type
    rest_col = f"rest_type_grouped_{restaurant_info.get('rest_type', 'Others')}"
    if rest_col in input_row.columns:
        input_row[rest_col] = 1
        
    pred = model.predict(input_row)[0]
    return round(float(np.clip(pred, 1.0, 5.0)), 2)

if __name__ == "__main__":
    sample_restaurant = {
        "location": "Koramangala 5th Block",
        "rest_type": "Casual Dining",
        "cost_for_two": 800,
        "votes": 350,
        "online_order": True,
        "book_table": True,
        "cuisines": ["Continental", "Italian", "North Indian"]
    }
    try:
        predicted = predict_rating(sample_restaurant)
        print(f"Predicted Zomato Rating: {predicted} / 5.0")
    except Exception as e:
        print(f"Prediction info: {e}")
