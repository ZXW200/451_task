"""Simplified Clustering Analysis - For Intermediate Students"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
# [新增] 导入轮廓系数计算，用于定量评估
from sklearn.metrics import silhouette_score

def find_best_k(data, max_k=10):
    """Find optimal number of clusters using Elbow Method and Silhouette Score"""
    print("\nFinding optimal number of clusters...")

    sse_values = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    # Try different k values
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)

        # Calculate metrics
        sse_values.append(kmeans.inertia_)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)

        print(f"  k={k}: SSE={kmeans.inertia_:.2f}, Silhouette={score:.3f}")

    # Plot Elbow and Silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Elbow Curve
    ax1.plot(k_range, sse_values, 'bo-', markersize=8)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Sum of Squared Errors (SSE)')
    ax1.set_title('Elbow Method (Lower is Better)')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Silhouette Score
    ax2.plot(k_range, silhouette_scores, 'ro-', markersize=8)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis (Higher is Better)')
    ax2.grid(True, alpha=0.3)

    # Mark the best k based on Silhouette Score
    best_k_silhouette = k_range[np.argmax(silhouette_scores)]
    ax2.axvline(x=best_k_silhouette, color='green', linestyle='--', label=f'Best k={best_k_silhouette}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('elbow_curve.png')
    print("Optimization plot saved: elbow_curve.png")
    plt.close()

    # Use the k with the highest silhouette score as the suggested k
    print(f"\nSuggested optimal k: {best_k_silhouette} (based on Silhouette Score)")

    return best_k_silhouette


def perform_kmeans(data, n_clusters):
    """Perform K-Means clustering"""
    print(f"\nPerforming K-Means clustering (k={n_clusters})...")

    # Create K-Means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    # Train model
    labels = kmeans.fit_predict(data)

    print(f"Clustering complete!")
    print(f"Number of cluster centers: {len(kmeans.cluster_centers_)}")

    # Count samples in each cluster
    for i in range(n_clusters):
        cluster_size = sum(labels == i)
        percentage = (cluster_size / len(labels)) * 100
        print(f"  Cluster {i}: {cluster_size} samples ({percentage:.1f}%)")

    return labels, kmeans


def perform_dbscan(data, eps=0.5, min_samples=5):
    """Perform DBSCAN clustering"""
    print(f"\nPerforming DBSCAN clustering (eps={eps}, min_samples={min_samples})...")

    # 初始化 DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"DBSCAN complete!")
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Noise points detected: {n_noise} ({n_noise/len(labels)*100:.2f}%)")

    return labels


def visualize_clusters_2d(data, labels, title='Clustering Results', save_path=None):
    """Visualize clusters using PCA dimension reduction to 2D"""
    print(f"\nGenerating visualization: {title}...")

    # Use PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)

    # Create scatter plot
    plt.figure(figsize=(10, 8))

    # Use different colors for each cluster
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        if label == -1:
            # Noise points (black, smaller, x marker)
            plt.scatter(data_2d[mask, 0], data_2d[mask, 1],
                       c='black', label='Noise',
                       alpha=0.3, s=20, marker='x')
        else:
            plt.scatter(data_2d[mask, 0], data_2d[mask, 1],
                       c=[color], label=f'Cluster {label}',
                       alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.title(f'{title} (PCA 2D Projection)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved: {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_clusters(data, labels, method_name="K-Means"):
    """Analyze characteristics of each cluster"""
    print(f"\nAnalyzing {method_name} cluster characteristics...")

    # Add labels to data
    data_with_labels = data.copy()
    data_with_labels['cluster'] = labels

    # Analyze each cluster
    unique_labels = sorted(set(labels))

    cluster_stats = {}

    for label in unique_labels:
        if label == -1:
            continue # Skip detailed stats for noise

        print(f"\nCluster {label} statistics:")

        # Extract cluster data
        cluster_data = data_with_labels[data_with_labels['cluster'] == label]
        cluster_data = cluster_data.drop('cluster', axis=1)

        # Calculate mean values
        mean_values = cluster_data.mean()

        # Show top 5 most significant features
        sorted_features = mean_values.abs().sort_values(ascending=False)
        print("  Top 5 significant features:")
        for i, (feature, value) in enumerate(sorted_features.head(5).items(), 1):
            print(f"    {i}. {feature}: {mean_values[feature]:.3f}")

        # Save statistics
        cluster_stats[f'cluster_{label}'] = {
            'size': len(cluster_data),
            'mean_values': mean_values.to_dict()
        }

    return cluster_stats


def clustering_pipeline(data_path, n_clusters=None):
    """Complete clustering analysis pipeline"""
    print("=" * 50)
    print("Starting Clustering Analysis Pipeline")
    print("=" * 50)

    # 1. Load data
    print(f"\nLoading data: {data_path}")
    data = pd.read_csv(data_path)
    print(f"Data shape: {data.shape[0]} rows x {data.shape[1]} columns")

    if data.isnull().any().any():
        print("Warning: NaN values detected! Dropping...")
        data = data.dropna()

    # --- K-Means Section ---
    print("\n--- Method 1: K-Means Clustering ---")

    # Find optimal k if not specified
    if n_clusters is None:
        # This will now plot both Elbow and Silhouette
        n_clusters = find_best_k(data, max_k=10)

    # Perform K-Means
    kmeans_labels, _ = perform_kmeans(data, n_clusters)

    # Calculate K-Means Score for comparison
    kmeans_score = silhouette_score(data, kmeans_labels)
    print(f"[Quantitative Metric] K-Means Silhouette Score: {kmeans_score:.3f}")

    visualize_clusters_2d(data, kmeans_labels, 'K-Means Clusters', 'clusters_kmeans.png')
    kmeans_stats = analyze_clusters(data, kmeans_labels, "K-Means")

    # --- DBSCAN Section ---
    print("\n--- Method 2: DBSCAN Clustering ---")
    try:
        dbscan_labels = perform_dbscan(data, eps=2.0, min_samples=5)
        visualize_clusters_2d(data, dbscan_labels, 'DBSCAN Clusters', 'clusters_dbscan.png')
        dbscan_stats = analyze_clusters(data, dbscan_labels, "DBSCAN")

        # === [新增] 计算并打印 DBSCAN 的轮廓系数 ===
        # 逻辑：去除噪点(-1)后，至少需要有2个簇才能计算轮廓系数
        labels_no_noise = dbscan_labels[dbscan_labels != -1]
        data_no_noise = data.iloc[dbscan_labels != -1]

        if len(set(labels_no_noise)) >= 2:
             db_score = silhouette_score(data_no_noise, labels_no_noise)
             print(f"\n[Quantitative Metric] DBSCAN Silhouette Score (excluding noise): {db_score:.3f}")
             print("  (Use this score to compare with K-Means in your report)")
        else:
             print(f"\n[Quantitative Metric] DBSCAN Silhouette Score: Not defined.")
             print(f"  Reason: Found {len(set(labels_no_noise))} valid clusters. Need at least 2.")
        # ================================================

    except Exception as e:
        print(f"DBSCAN failed: {e}")
        import traceback
        traceback.print_exc()
        dbscan_labels = None

    # Save data with cluster labels
    data_with_clusters = data.copy()
    data_with_clusters['kmeans_cluster'] = kmeans_labels
    if dbscan_labels is not None:
        data_with_clusters['dbscan_cluster'] = dbscan_labels

    output_path = 'data_with_clusters.csv'
    data_with_clusters.to_csv(output_path, index=False)
    print(f"\nData with cluster labels saved: {output_path}")

    print("\n" + "=" * 50)
    print("Clustering Analysis Complete!")
    print("=" * 50)

    return kmeans_labels, kmeans_stats

if __name__ == "__main__":
    data_file = "preprocessed_data.csv"
    clustering_pipeline(data_file, n_clusters=None)