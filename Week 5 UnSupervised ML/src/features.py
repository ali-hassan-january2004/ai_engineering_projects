import numpy as np
import pandas as pd

def create_customer_features(df):
    """
    Engineers RFM (Recency, Frequency, Monetary) and behavioral features 
    for e-commerce customer segmentation with built-in safety checks.
    """
    df = df.copy()
    
    # 1. Ratio / Intensity Feature: Average Order Value (AOV)
    # Safe division to prevent division by zero errors
    df["avg_order_value"] = df["total_spend"] / df["total_orders"].replace(0, 1)
    
    # 2. Numerical Transformation: Stabilizing skewed recency distribution using log1p
    df["log_recency"] = np.log1p(df["days_since_last_purchase"])
    
    # 3. Interaction / Spending Intensity Feature
    df["spend_frequency_interaction"] = df["total_spend"] * df["purchase_frequency"]
    
    # 4. Binary Indicator Flag: High-Value Customer Top Quartile
    spending_threshold = df["total_spend"].quantile(0.75)
    df["high_value_flag"] = (df["total_spend"] >= spending_threshold).astype(int)
    
    return df