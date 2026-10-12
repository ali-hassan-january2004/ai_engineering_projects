from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def build_clustering_pipeline(n_clusters=4):
    """
    Builds a robust scikit-learn pipeline combining standard scaling 
    and K-Means clustering.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init="auto"))
    ])
    return pipeline

def evaluate_kmeans_range(X_scaled, min_k=2, max_k=8):
    """
    Evaluates candidate k values using inertia (elbow method) and silhouette scores.
    """
    results = []
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": sil
        })
    return results

def get_pca_projection(X_scaled, n_components=2):
    """
    Projects scaled feature matrices into principal component space 
    and returns explained variance ratios.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    return X_pca, explained_variance