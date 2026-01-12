# kmeans.py
"""
Heavy-tailed k-means clustering.

Uses α-location and α-power instead of mean and variance,
providing robust clustering for heavy-tailed data.
"""
from __future__ import annotations

import numpy as np

from .power import P_alpha
from .location import location_L


def init_locations_percentiles(x: np.ndarray, K: int) -> np.ndarray:
    """
    Initialize centroids using percentiles.
    
    Parameters
    ----------
    x : ndarray of shape (n,) or (n, d)
        Data points.
    K : int
        Number of clusters.
    
    Returns
    -------
    ndarray of shape (K,) or (K, d)
        Initial centroid locations.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return np.array([np.percentile(x, 100.0 * (2 * i + 1) / (2 * K)) for i in range(K)], dtype=float)
    else:
        n, d = x.shape
        centroids = np.zeros((K, d), dtype=float)
        for dim in range(d):
            for i in range(K):
                centroids[i, dim] = np.percentile(x[:, dim], 100.0 * (2 * i + 1) / (2 * K))
        return centroids


def assign_by_nearest_mu(x: np.ndarray, mus: np.ndarray) -> np.ndarray:
    """
    Assign each point to the nearest centroid.
    
    Parameters
    ----------
    x : ndarray of shape (n, d)
        Data points.
    mus : ndarray of shape (K, d)
        Centroid locations.
    
    Returns
    -------
    ndarray of shape (n,)
        Cluster assignments (0 to K-1).
    """
    x = np.atleast_2d(x)
    mus = np.atleast_2d(mus)
    dists = np.linalg.norm(x[:, None, :] - mus[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def update_locations_and_cluster_powers(
    x: np.ndarray,
    assignments: np.ndarray,
    K: int,
    alpha: float,
    h_lookup: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute new cluster locations μ_j and cluster α-power P_j.
    
    Parameters
    ----------
    x : ndarray of shape (n, d)
        Data points.
    assignments : ndarray of shape (n,)
        Current cluster assignments.
    K : int
        Number of clusters.
    alpha : float
        Stability parameter.
    h_lookup : dict, optional
        Entropy lookup for α ≠ 1.
    
    Returns
    -------
    mus : ndarray of shape (K, d)
        Updated centroid locations.
    P_cluster : ndarray of shape (K,)
        α-power for each cluster.
    """
    x = np.atleast_2d(x)
    n, d = x.shape
    mus = np.zeros((K, d), dtype=float)
    P_cluster = np.zeros(K, dtype=float)

    for j in range(K):
        xj = x[assignments == j]
        if xj.size == 0:
            continue
        mu_j, P_j = location_L(xj, alpha=alpha, h_lookup=h_lookup)
        if np.isscalar(mu_j):
            mus[j] = mu_j
        else:
            mus[j] = mu_j
        P_cluster[j] = P_j
    return mus, P_cluster


def global_power_I(
    x: np.ndarray,
    assignments: np.ndarray,
    mus: np.ndarray,
    alpha: float,
    h_lookup: dict | None,
) -> float:
    """
    Compute global inertia: P_alpha over all cluster residuals.
    
    Parameters
    ----------
    x : ndarray of shape (n, d)
        Data points.
    assignments : ndarray of shape (n,)
        Cluster assignments.
    mus : ndarray of shape (K, d)
        Centroid locations.
    alpha : float
        Stability parameter.
    h_lookup : dict, optional
        Entropy lookup for α ≠ 1.
    
    Returns
    -------
    float
        Global α-power (inertia).
    """
    x = np.atleast_2d(x)
    mus = np.atleast_2d(mus)
    residuals = x - mus[assignments]
    return P_alpha(residuals, alpha=alpha, h_lookup=h_lookup)


def kmeans_heavy_tailed(
    x: np.ndarray,
    K: int,
    alpha: float = 1.0,
    h_lookup: dict | None = None,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> dict:
    """
    Heavy-tailed k-means clustering.
    
    Uses α-location for centroids and α-power for inertia,
    providing robust clustering for heavy-tailed data.
    
    Parameters
    ----------
    x : ndarray of shape (n,) or (n, d)
        Data points to cluster.
    K : int
        Number of clusters.
    alpha : float, default=1.0
        Stability parameter. α=1 is Cauchy.
    h_lookup : dict, optional
        Pre-computed entropies. Required for α ≠ 1.
    tol : float, default=1e-6
        Convergence tolerance (relative change in inertia).
    max_iter : int, default=100
        Maximum iterations.
    
    Returns
    -------
    dict
        {
            "assignments": cluster labels (n,),
            "locations_mu": centroid locations (K, d),
            "cluster_power_P": α-power per cluster (K,),
            "global_power_I": final global inertia,
            "I_history": inertia at each iteration,
            "iterations": number of iterations run
        }
    
    Examples
    --------
    >>> result = kmeans_heavy_tailed(X, K=3, alpha=1.0)
    >>> labels = result["assignments"]
    >>> centroids = result["locations_mu"]
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    
    # Initialize centroids
    mus = init_locations_percentiles(x, K)
    if mus.ndim == 1:
        mus = mus[:, None]
    assignments = assign_by_nearest_mu(x, mus)
    P_hist = []
    I = 0.0

    for it in range(max_iter):
        # Step 1: update μ_j and cluster P_j
        mus, P_cluster = update_locations_and_cluster_powers(x, assignments, K, alpha, h_lookup)

        # Step 2: compute global inertia
        I = global_power_I(x, assignments, mus, alpha, h_lookup)
        P_hist.append(I)

        # Step 3: assign points to nearest centroid
        new_assignments = assign_by_nearest_mu(x, mus)

        # Step 4: check convergence
        if it >= 1:
            rel_change = abs(P_hist[-1] - P_hist[-2]) / max(P_hist[-2], 1e-12)
            if rel_change <= tol:
                assignments = new_assignments
                break

        assignments = new_assignments

    return {
        "assignments": assignments,
        "locations_mu": mus,
        "cluster_power_P": P_cluster,
        "global_power_I": I,
        "I_history": np.array(P_hist),
        "iterations": it + 1,
    }
