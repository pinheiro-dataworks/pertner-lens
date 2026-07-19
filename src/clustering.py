"""K-Means segmentation, k-selection diagnostics, stability check, DBSCAN
stress test, and a PCA(2) projection used strictly for visualization.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def build_pipeline(k: int, seed: int = config.SEED, n_init: int = config.KMEANS_N_INIT) -> Pipeline:
    """Scaling + clustering as a single fit/predict unit.

    This is not an aesthetic choice: it guarantees the scaler used at
    inference time is the exact one fit at training time, eliminating the
    "scaled in notebook 2, clustered in notebook 3 with a different scaler"
    class of bug, and it serializes as one artifact.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, n_init=n_init, random_state=seed)),
        ]
    )


def elbow_silhouette(X: np.ndarray, k_range=config.K_CANDIDATES, seed: int = config.SEED) -> pd.DataFrame:
    """Inertia and mean silhouette across candidate k, on scaled features."""
    Xs = StandardScaler().fit_transform(X)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=seed).fit(Xs)
        sil = silhouette_score(Xs, km.labels_)
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
    return pd.DataFrame(rows)


def stability_ari(X: np.ndarray, k: int = config.N_CLUSTERS, seeds=config.STABILITY_SEEDS) -> pd.DataFrame:
    """Refit K-Means across several seeds; clusters that survive re-initialization
    are real structure, not an artifact of a lucky starting point."""
    Xs = StandardScaler().fit_transform(X)
    labelings = {
        seed: KMeans(n_clusters=k, n_init=config.KMEANS_N_INIT, random_state=seed).fit(Xs).labels_
        for seed in seeds
    }
    rows = [
        {"seed_a": s1, "seed_b": s2, "ari": float(adjusted_rand_score(labelings[s1], labelings[s2]))}
        for s1, s2 in combinations(seeds, 2)
    ]
    return pd.DataFrame(rows)


def pca_projection(X: np.ndarray, n_components: int = 2, seed: int = config.SEED):
    """2D projection for the scatter plot only -- never used as the clustering space itself."""
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=seed).fit(Xs)
    coords = pca.transform(Xs)
    return coords, pca.explained_variance_ratio_


def k_distance_data(X: np.ndarray, k: int = config.DBSCAN_MIN_SAMPLES) -> np.ndarray:
    """Sorted distance to each point's k-th nearest neighbor, the standard
    diagnostic for picking DBSCAN's eps from its "knee"."""
    Xs = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k).fit(Xs)
    distances, _ = nn.kneighbors(Xs)
    return np.sort(distances[:, -1])


def estimate_eps_from_knee(kth_distances: np.ndarray) -> float:
    """Knee of the sorted k-distance curve: point of maximum perpendicular
    distance from the straight line connecting its two endpoints."""
    n = len(kth_distances)
    line = np.linspace(kth_distances[0], kth_distances[-1], n)
    knee_idx = int(np.argmax(np.abs(kth_distances - line)))
    return float(kth_distances[knee_idx])


def run_dbscan(X: np.ndarray, eps: float, min_samples: int = config.DBSCAN_MIN_SAMPLES) -> dict:
    Xs = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Xs)
    labels = db.labels_
    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    core_mask = labels != -1
    sil = silhouette_score(Xs[core_mask], labels[core_mask]) if n_clusters > 1 else None
    return {
        "eps": round(float(eps), 3),
        "min_samples": int(min_samples),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": round(100 * n_noise / len(labels), 1),
        "silhouette_core_points": None if sil is None else round(float(sil), 3),
        "labels": labels,
    }


def fit_final_model(X: np.ndarray, k: int = config.N_CLUSTERS, seed: int = config.SEED) -> Pipeline:
    pipe = build_pipeline(k, seed)
    pipe.fit(X)
    return pipe
