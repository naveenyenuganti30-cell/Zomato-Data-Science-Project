import re
import numpy as np
import pandas as pd
from typing import Tuple

def clean_rate(val):
    if pd.isna(val) or val in ["NEW", "-", "nan"]:
        return np.nan
    val = str(val).split("/")[0].strip()
    try:
        return float(val)
    except ValueError:
        return np.nan

def clean_cost(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace(",", "").strip()
    try:
        return float(val)
    except ValueError:
        return np.nan

def clean_votes(val):
    if pd.isna(val):
        return 0
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw Zomato dataset, cleans columns, handles nulls, and deduplicates.
    """
    df = pd.read_csv(file_path, encoding="latin-1")
    
    # Drop irrelevant columns
    cols_to_drop = [c for c in ["address", "phone"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Rename approx_cost column for cleaner syntax
    if "approx_cost(for two people)" in df.columns:
        df = df.rename(columns={"approx_cost(for two people)": "cost_for_two"})
    if "listed_in(type)" in df.columns:
        df = df.rename(columns={"listed_in(type)": "type"})
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Clean rate
    df["rate"] = df["rate"].apply(clean_rate)
    
    # Clean cost
    df["cost_for_two"] = df["cost_for_two"].apply(clean_cost)
    
    # Clean votes
    df["votes"] = df["votes"].apply(clean_votes)
    
    # Drop rows without rate (target variable)
    df = df.dropna(subset=["rate"]).reset_index(drop=True)
    
    # Fill remaining missing numerical values with median
    if df["cost_for_two"].isna().sum() > 0:
        df["cost_for_two"] = df["cost_for_two"].fillna(df["cost_for_two"].median())
        
    # Clean binary flag columns
    if "online_order" in df.columns:
        df["online_order"] = df["online_order"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    if "book_table" in df.columns:
        df["book_table"] = df["book_table"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        
    # Clean text columns
    for col in ["location", "rest_type", "type", "cuisines"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            
    return df
