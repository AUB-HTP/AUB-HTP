"""
Heavy-Tailed K-Means Clustering Example

This example demonstrates:
1. Sampling from 3 separate heavy-tailed distributions with different locations
2. Running HeavyTailedKMeans clustering
3. Computing the misclassification error rate
"""

import numpy as np
import matplotlib.pyplot as plt

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

    print(f"Generated {len(X)} samples from {n_clusters} clusters")
    print(f"Data shape: {X.shape}")
    print(f"Alpha (stability): {alpha}")
    print(f"Cluster centers: {cluster_centers}")
    print()

    # Run Heavy-Tailed K-Means
    kmeans = KMeansHeavyTailed(
        n_clusters=n_clusters,
        alpha=alpha,
        max_itererations=100,
        convergence_tolerance=1e-6,
    )
    kmeans.fit(X)

    print(f"Predicted Cluster centers:\n {kmeans.cluster_centers_}")
    
if __name__ == "__main__":
    main()
