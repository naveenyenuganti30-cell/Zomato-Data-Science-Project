# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"

def save_eda_plots(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Rating Distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["rate"], bins=25, kde=True, color="#E23744", ax=ax, edgecolor="white", alpha=0.7)
    mean_rate = df["rate"].mean()
    median_rate = df["rate"].median()
    ax.axvline(mean_rate, color="#1D1D1D", linestyle="--", linewidth=2, label=f"Mean: {mean_rate:.2f}")
    ax.axvline(median_rate, color="#008080", linestyle="-", linewidth=2, label=f"Median: {median_rate:.2f}")
    ax.set_title("Distribution of Restaurant Ratings in Bangalore", pad=12)
    ax.set_xlabel("Rating (out of 5)")
    ax.set_ylabel("Number of Restaurants")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "rating_distribution.png"), dpi=180)
    plt.close(fig)
    
    # 2. Online Ordering vs Rating
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df["online_order"].map({1: "Offers Online Order", 0: "No Online Order"}), 
                y=df["rate"], palette=["#E23744", "#808080"], ax=ax, width=0.45)
    ax.set_title("Impact of Online Ordering Availability on Restaurant Rating", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Rating (out of 5)")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "online_order_vs_rating.png"), dpi=180)
    plt.close(fig)

    # 3. Cost for Two vs Rating
    fig, ax = plt.subplots(figsize=(10, 6))
    sample_df = df.sample(min(3000, len(df)), random_state=42)
    sns.scatterplot(data=sample_df, x="cost_for_two", y="rate", hue="book_table", 
                    palette={1: "#E23744", 0: "#3B82F6"}, alpha=0.6, s=40, ax=ax)
    sns.regplot(data=sample_df, x="cost_for_two", y="rate", scatter=False, ax=ax, color="#1E293B", line_kws={"linewidth": 2})
    ax.set_title("Approximate Cost for Two vs. Restaurant Rating", pad=12)
    ax.set_xlabel("Approx Cost for Two People (INR)")
    ax.set_ylabel("Rating (out of 5)")
    ax.legend(title="Table Booking", labels=["No Table Booking", "Table Booking Available"])
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "cost_vs_rating.png"), dpi=180)
    plt.close(fig)

    # 4. Top 12 Locations by Restaurant Density
    fig, ax = plt.subplots(figsize=(10, 6))
    top_locs = df["location"].value_counts().head(12)
    sns.barplot(x=top_locs.values, y=top_locs.index, palette="mako", ax=ax)
    ax.set_title("Top 12 Food Hubs in Bangalore by Restaurant Count", pad=12)
    ax.set_xlabel("Number of Registered Restaurants")
    ax.set_ylabel("Locality")
    for i, v in enumerate(top_locs.values):
        ax.text(v + 15, i, f"{v:,}", va="center", fontweight="semibold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "top_locations.png"), dpi=180)
    plt.close(fig)

    # 5. Top Popular Cuisines
    fig, ax = plt.subplots(figsize=(10, 6))
    all_cuisines = [c.strip() for sublist in df["cuisines"].dropna().str.split(",") for c in sublist if c.strip() != ""]
    top_c = pd.Series(all_cuisines).value_counts().head(12)
    sns.barplot(x=top_c.values, y=top_c.index, palette="rocket", ax=ax)
    ax.set_title("Top 12 Most Frequently Offered Cuisines", pad=12)
    ax.set_xlabel("Frequency Count")
    ax.set_ylabel("Cuisine")
    for i, v in enumerate(top_c.values):
        ax.text(v + 20, i, f"{v:,}", va="center", fontweight="semibold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "top_cuisines.png"), dpi=180)
    plt.close(fig)

def save_model_plots(comparison_df: pd.DataFrame, feature_importance_df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # Model Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=comparison_df, x="Model", y="R2_Score", palette="viridis", ax=axes[0])
    axes[0].set_title("Model Comparison: R-Squared Score (Higher is Better)")
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(comparison_df["R2_Score"]):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    sns.barplot(data=comparison_df, x="Model", y="RMSE", palette="magma", ax=axes[1])
    axes[1].set_title("Model Comparison: RMSE (Lower is Better)")
    axes[1].tick_params(axis="x", rotation=25)
    for i, v in enumerate(comparison_df["RMSE"]):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=180)
    plt.close(fig)

    # Feature Importance
    if feature_importance_df is not None and not feature_importance_df.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        top_features = feature_importance_df.head(15)
        sns.barplot(data=top_features, x="Importance", y="Feature", palette="crest", ax=ax)
        ax.set_title("Top 15 Most Influential Features for Rating Prediction", pad=12)
        ax.set_xlabel("Feature Importance Score")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=180)
        plt.close(fig)
