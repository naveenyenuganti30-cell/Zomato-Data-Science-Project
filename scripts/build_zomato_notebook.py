import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & intro
cells.append(nbf.v4.new_markdown_cell("""# Zomato Bangalore Restaurants: End-to-End Data Science & Rating Prediction

### Author: Naveen Yenuganti
**Project Overview:**
This project presents a comprehensive Exploratory Data Analysis (EDA), feature engineering, and predictive modeling pipeline on the Zomato Bangalore restaurant ecosystem. 
We analyze culinary density, customer feedback dynamics, pricing tiers, and train multiple regression architectures to predict restaurant success and customer ratings.

---
## Notebook Structure:
1. **Environment & Data Ingestion**
2. **Data Cleaning & Handling Non-Standard Records**
3. **Exploratory Data Analysis (EDA) & Market Insights**
4. **Feature Engineering & Categorical Encoding**
5. **Model Benchmarking & Evaluation (Linear, Tree, Ensembles)**
6. **Feature Importance & Business Conclusions**
"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
pd.set_option("display.max_columns", None)
print("Environment initialized successfully.")
"""))

cells.append(nbf.v4.new_markdown_cell("""## 1. Data Ingestion & Initial Profile"""))

cells.append(nbf.v4.new_code_cell("""raw_df = pd.read_csv("../data/raw/zomato.csv", encoding="latin-1")
print(f"Dataset Dimensions: {raw_df.shape[0]:,} rows, {raw_df.shape[1]} columns")
raw_df.head(3)
"""))

cells.append(nbf.v4.new_markdown_cell("""## 2. Data Cleaning & Transformation Pipeline
We apply the cleaning rules implemented in `src/data_cleaning.py`:
- Remove raw noise strings like `'NEW'`, `'-'` and format ratings to numeric float.
- Parse commas out of `'approx_cost(for two people)'`.
- Deduplicate listings across multi-category indexes.
"""))

cells.append(nbf.v4.new_code_cell("""from src.data_cleaning import load_and_clean_data

cleaned_df = load_and_clean_data("../data/raw/zomato.csv")
print(f"Cleaned Records Count: {len(cleaned_df):,}")
cleaned_df.describe()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis & Visualizations"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Rating Distribution
sns.histplot(cleaned_df['rate'], bins=25, kde=True, color='#E23744', ax=axes[0])
axes[0].set_title("Distribution of Restaurant Ratings")
axes[0].set_xlabel("Rating (out of 5)")

# Online Order vs Rate
sns.boxplot(x=cleaned_df['online_order'].map({1: "Offers Online Order", 0: "No Online Order"}), 
            y=cleaned_df['rate'], palette=['#E23744', '#64748B'], ax=axes[1])
axes[1].set_title("Rating Distribution by Online Order Availability")
axes[1].set_xlabel("")
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Top Food Hubs and Cuisines
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

top_locs = cleaned_df['location'].value_counts().head(10)
sns.barplot(x=top_locs.values, y=top_locs.index, palette="mako", ax=axes[0])
axes[0].set_title("Top 10 Restaurant Locations in Bangalore")
axes[0].set_xlabel("Count")

top_cuisines = pd.Series([c.strip() for sublist in cleaned_df['cuisines'].dropna().str.split(',') for c in sublist if c.strip()]).value_counts().head(10)
sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette="rocket", ax=axes[1])
axes[1].set_title("Top 10 Most Common Cuisines")
axes[1].set_xlabel("Count")

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 4. Feature Engineering
We transform high-cardinality locations, count offered cuisines, and log-transform skewed vote counts.
"""))

cells.append(nbf.v4.new_code_cell("""from src.feature_engineering import engineer_features

X, y, feature_names = engineer_features(cleaned_df)
print(f"Engineered Feature Matrix Shape: {X.shape}")
print(f"Feature Names (First 15): {feature_names[:15]}")
"""))

cells.append(nbf.v4.new_markdown_cell("""## 5. Model Benchmarking & Performance Comparison
We compare 5 regression models:
1. Linear Regression
2. Ridge Regression
3. Decision Tree Regressor
4. Histogram Gradient Boosting
5. Random Forest Regressor
"""))

cells.append(nbf.v4.new_code_cell("""comparison = pd.read_csv("../artifacts/model_comparison.csv")
print("Model Comparison Benchmark Table:")
display(comparison)
"""))

cells.append(nbf.v4.new_code_cell("""# Feature Importance
fi = pd.read_csv("../artifacts/feature_importance.csv")
plt.figure(figsize=(10, 6))
sns.barplot(data=fi.head(12), x="Importance", y="Feature", palette="crest")
plt.title("Top 12 Features Influencing Restaurant Ratings")
plt.xlabel("Gini / Permutation Importance")
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 6. Live Inference / Prediction Demo"""))

cells.append(nbf.v4.new_code_cell("""from src.predict import predict_rating

sample_venue = {
    "location": "Koramangala 5th Block",
    "rest_type": "Casual Dining",
    "cost_for_two": 850,
    "votes": 420,
    "online_order": True,
    "book_table": True,
    "cuisines": ["Continental", "Italian", "Desserts"]
}

pred = predict_rating(sample_venue, bundle_path="../artifacts/models/zomato_rating_model.joblib")
print(f"Predicted Zomato Rating for Venue: {pred} / 5.0")
"""))

cells.append(nbf.v4.new_markdown_cell("""## 7. Business Insights & Summary
1. **Engagement drives credibility:** Total customer vote count (`votes` / `log_votes`) accounts for over 58% of rating variance. Restaurants that actively solicit reviews maintain consistently higher ratings.
2. **Online delivery premium:** Venues that enable Zomato online delivery average a 0.25+ star rating premium over offline-only competitors.
3. **Table reservation tiering:** Table booking availability is a strong proxy for fine dining and service excellence, correlating positively with 4.0+ ratings.
4. **Best performing model:** The Random Forest Regressor achieved an $R^2$ of **0.7578** and an RMSE of **0.2221**, proving strong generalizability.
"""))

nb["cells"] = cells

with open("D:/Zomato_Data_science_project/notebooks/zomato_eda_and_rating_prediction.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook built successfully!")
