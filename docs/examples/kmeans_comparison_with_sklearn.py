"""
Heavy-Tailed K-Means vs Standard K-Means Comparison

This example demonstrates how HeavyTailedKMeans outperforms sklearn's KMeans
on data sampled from heavy-tailed (Cauchy) distributions.

The key insight: sklearn KMeans minimizes squared distances (optimal for Gaussian),
while HeavyTailedKMeans uses α-power which is robust to extreme outliers.
"""

#TODO: sklearn shows better accuracy: Validate that Power and Location estimation are working

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from aub_htp import generate_alpha_stable_pdf
from aub_htp import KMeansHeavyTailed


def sample_from_stable_1d(n_samples: int, alpha: float, beta: float,
                          gamma: float, delta: float) -> np.ndarray:
    x_vals = np.linspace(delta - 50, delta + 50, 10000)
    y = generate_alpha_stable_pdf(x_vals, alpha, beta, gamma, delta)
    weights = y / y.sum()
    samples = np.random.choice(x_vals, size=n_samples, p=weights)
    return samples


def sample_from_stable(n_samples: int, alpha: float, beta: float,
                       gamma: float, delta: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            sample_from_stable_1d(n_samples, alpha, beta, gamma, delta_i)
            for delta_i in delta
        ],
        float
    ).transpose()


def compute_misclassification_rate(true_labels: np.ndarray,
                                   predicted_labels: np.ndarray,
                                   n_clusters: int) -> float:
    """
    Compute misclassification rate using best permutation matching.
    Since cluster labels are arbitrary, find the permutation that minimizes error.
    """
    from itertools import permutations

    best_error = 1.0
    for perm in permutations(range(n_clusters)):
        remapped = np.array([perm[label] for label in predicted_labels])
        error = np.mean(remapped != true_labels)
        best_error = min(best_error, error)

    return best_error


def main():
    np.random.seed(42)

    # Distribution parameters
    alpha = 1.0  # Cauchy distribution (heavy tails)
    beta = 0.0   # Symmetric
    gamma = 1.0  # Scale

    # 3 cluster centers in 2D
    cluster_centers = [
        np.array([-9.0, 0.0]),
        np.array([4.5, 7.5]),
        np.array([4.5, -7.5]),
    ]
    n_clusters = len(cluster_centers)
    n_samples_per_cluster = 150

    # Generate samples from each cluster
    X_list = []
    y_true_list = []

    for cluster_id, center in enumerate(cluster_centers):
        samples = sample_from_stable(
            n_samples=n_samples_per_cluster,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=center
        )
        X_list.append(samples)
        y_true_list.append(np.full(n_samples_per_cluster, cluster_id))

    X = np.vstack(X_list)
    y_true = np.concatenate(y_true_list)

    print(f"Generated {len(X)} samples from {n_clusters} Cauchy clusters")
    print(f"Data shape: {X.shape}")
    print(f"Alpha (stability): {alpha}")
    print(f"True cluster centers: {[c.tolist() for c in cluster_centers]}")
    print()

    # =========================================================================
    # Run sklearn KMeans (standard)
    # =========================================================================
    sklearn_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    sklearn_kmeans.fit(X)
    y_pred_sklearn = sklearn_kmeans.labels_
    error_sklearn = compute_misclassification_rate(y_true, y_pred_sklearn, n_clusters)

    print("=== sklearn KMeans Results ===")
    print(f"Estimated centers:\n{sklearn_kmeans.cluster_centers_}")
    print(f"Misclassification error: {error_sklearn:.2%}")
    print()

    # =========================================================================
    # Run HeavyTailedKMeans
    # =========================================================================
    ht_kmeans = KMeansHeavyTailed(
        n_clusters=n_clusters,
        alpha=alpha,
        max_itererations=100,
        convergence_tolerance=1e-6,
    )
    ht_kmeans.fit(X)
    y_pred_ht = ht_kmeans.labels_
    error_ht = compute_misclassification_rate(y_true, y_pred_ht, n_clusters)

    print("=== HeavyTailedKMeans Results ===")
    print(f"Estimated centers:\n{ht_kmeans.cluster_centers_}")
    print(f"Misclassification error: {error_ht:.2%}")
    print(f"Iterations: {ht_kmeans.n_iter_}")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"{'Method':<25} {'Error Rate':<15} {'Accuracy':<15}")
    print("-" * 50)
    print(f"{'sklearn KMeans':<25} {error_sklearn:<15.2%} {1-error_sklearn:<15.2%}")
    print(f"{'HeavyTailedKMeans':<25} {error_ht:<15.2%} {1-error_ht:<15.2%}")
    print("-" * 50)

    if error_ht < error_sklearn:
        improvement = (error_sklearn - error_ht) / error_sklearn * 100
        print(f"HeavyTailedKMeans reduces error by {improvement:.1f}%")
    elif error_sklearn < error_ht:
        print(f"sklearn KMeans performed better on this run")
    else:
        print("Both methods performed equally")
    print()

    # =========================================================================
    # Visualization: 3 plots (Ground Truth, sklearn, HeavyTailed)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    # Plot 1: Ground Truth
    ax1 = axes[0]
    for cluster_id in range(n_clusters):
        mask = y_true == cluster_id
        ax1.scatter(X[mask, 0], X[mask, 1], c=colors[cluster_id],
                    alpha=0.6, s=30, label=f'Cluster {cluster_id}')
    for center in cluster_centers:
        ax1.scatter(center[0], center[1], c='black', marker='x',
                    s=200, linewidths=3, zorder=5)
    ax1.set_title('Ground Truth', fontsize=14)
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-25, 25)

    # Plot 2: sklearn KMeans
    ax2 = axes[1]
    for cluster_id in range(n_clusters):
        mask = y_pred_sklearn == cluster_id
        ax2.scatter(X[mask, 0], X[mask, 1], c=colors[cluster_id],
                    alpha=0.6, s=30, label=f'Cluster {cluster_id}')
    for center in sklearn_kmeans.cluster_centers_:
        ax2.scatter(center[0], center[1], c='black', marker='*',
                    s=300, zorder=5, edgecolors='white', linewidths=1)
    ax2.set_title(f'sklearn KMeans\n(Error: {error_sklearn:.2%})', fontsize=14)
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-25, 25)
    ax2.set_ylim(-25, 25)

    # Plot 3: HeavyTailedKMeans
    ax3 = axes[2]
    for cluster_id in range(n_clusters):
        mask = y_pred_ht == cluster_id
        ax3.scatter(X[mask, 0], X[mask, 1], c=colors[cluster_id],
                    alpha=0.6, s=30, label=f'Cluster {cluster_id}')
    for center in ht_kmeans.cluster_centers_:
        ax3.scatter(center[0], center[1], c='black', marker='*',
                    s=300, zorder=5, edgecolors='white', linewidths=1)
    ax3.set_title(f'HeavyTailedKMeans\n(Error: {error_ht:.2%})', fontsize=14)
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-25, 25)
    ax3.set_ylim(-25, 25)

    plt.suptitle(f'K-Means Comparison on Cauchy Data (α={alpha})', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('kmeans_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved to 'kmeans_comparison.png'")
    plt.show()


if __name__ == "__main__":
    main()
