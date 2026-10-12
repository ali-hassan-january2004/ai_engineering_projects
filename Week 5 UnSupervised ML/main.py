import pandas as pd
from src.features import create_customer_features
from src.clustering import build_clustering_pipeline, evaluate_kmeans_range, get_pca_projection

def main():
    print("1. Loading raw customer dataset...")
    df = pd.read_csv("data/customer_data.csv")
    print(f"Dataset shape: {df.shape}")

    print("\n2. Applying feature engineering...")
    df_engineered = create_customer_features(df)
    print("Engineered columns added:", [col for col in df_engineered.columns if col not in df.columns])

    # Select numerical features for machine learning
    feature_cols = [
        "total_spend", "total_orders", "purchase_frequency", 
        "days_since_last_purchase", "avg_order_value", 
        "log_recency", "spend_frequency_interaction"
    ]
    X = df_engineered[feature_cols]

    print("\n3. Running K-Means Clustering Pipeline (k=3)...")
    pipeline = build_clustering_pipeline(n_clusters=3)
    labels = pipeline.fit_predict(X)
    df_engineered["cluster_label"] = labels
    print("Cluster assignment sample:")
    print(df_engineered[["customer_id", "total_spend", "cluster_label"]].head())

    print("\n4. Computing PCA Projections...")
    scaler = pipeline.named_steps["scaler"]
    X_scaled = scaler.fit_transform(X)
    X_pca, explained_variance = get_pca_projection(X_scaled, n_components=2)
    print(f"PCA Explained Variance Ratio: {explained_variance}")
    print(f"Total Variance Captured by 2D PCA: {explained_variance.sum() * 100:.2f}%")

    print("\nPipeline execution complete successfully!")

if __name__ == "__main__":
    main()