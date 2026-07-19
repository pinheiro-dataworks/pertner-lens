"""End-to-end pipeline: raw CSVs -> processed parquet/JSON artifacts consumed by the Streamlit app.

Run with: python scripts/build_pipeline.py
This script trains nothing at app runtime -- it is the offline half of the
train/serve split described in the README.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import clustering, config, data_loader, features, profiling


def log(msg: str) -> None:
    print(f"[build_pipeline] {msg}")


def main() -> None:
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log("Loading raw tables and validating referential integrity...")
    tables, quality_report = data_loader.load_and_validate()
    log(f"data quality report: {quality_report}")

    log("Deduplicating reviews and aggregating geolocation...")
    reviews_dedup = data_loader.dedupe_reviews(tables["order_reviews"])
    _ = data_loader.aggregate_geolocation(tables["geolocation"])  # validated, kept for completeness

    log("Building seller-order pairs...")
    pairs = features.build_seller_order_pairs(tables["order_items"], tables["orders"])
    pairs = features.attach_review_scores(pairs, reviews_dedup)
    pairs.to_parquet(config.PROCESSED_FILES["seller_order_pairs"], index=False)
    log(f"seller_order_pairs: {len(pairs):,} rows")

    multi_seller_stats = features.multi_seller_order_stats(tables["order_items"])
    log(f"multi-seller order stats: {multi_seller_stats}")

    reference_date = tables["orders"]["order_purchase_timestamp"].max()
    log(f"reference date (max order_purchase_timestamp): {reference_date}")

    log("Building seller-level features...")
    seller_features = features.build_seller_features(pairs, reference_date)
    seller_features = seller_features.merge(
        tables["sellers"][["seller_id", "seller_state", "seller_city"]], on="seller_id", how="left"
    )
    seller_features = features.add_log_features(seller_features)

    sensitivity = features.threshold_sensitivity(seller_features)
    log(f"threshold sensitivity:\n{sensitivity}")

    eligible, excluded = features.apply_min_order_threshold(seller_features)
    log(f"eligible for clustering: {len(eligible):,} / excluded (low data): {len(excluded):,}")

    # ---- correlation diagnostic (drives the feature-set decision below) ----
    corr_matrix = eligible[config.CANDIDATE_FEATURES].corr(method="pearson").round(3)
    log(f"feature correlation matrix, candidate set (eligible population):\n{corr_matrix}")

    # ---- k-selection diagnostics ----
    X_eligible = eligible[config.CLUSTER_FEATURES].to_numpy()
    log("Running elbow/silhouette across k=2..10...")
    elbow_df = clustering.elbow_silhouette(X_eligible)
    log(f"elbow/silhouette:\n{elbow_df}")

    log(f"Running stability check (ARI across seeds) at k={config.N_CLUSTERS}...")
    stability_df = clustering.stability_ari(X_eligible, k=config.N_CLUSTERS)
    log(f"stability ARI:\n{stability_df}")

    # ---- DBSCAN critical comparison ----
    log("Computing k-distance plot for DBSCAN eps estimation...")
    kth_distances = clustering.k_distance_data(X_eligible)
    eps_estimate = clustering.estimate_eps_from_knee(kth_distances)
    log(f"estimated eps from knee: {eps_estimate:.3f}")
    dbscan_result = clustering.run_dbscan(X_eligible, eps=eps_estimate)
    dbscan_labels = dbscan_result.pop("labels")
    log(f"DBSCAN result (knee eps): {dbscan_result}")

    log("Sweeping eps around the knee to show the merge/fragment transition...")
    dbscan_sweep = []
    for mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
        r = clustering.run_dbscan(X_eligible, eps=eps_estimate * mult)
        r.pop("labels")
        r["eps_multiplier"] = mult
        dbscan_sweep.append(r)
    log(f"DBSCAN eps sweep:\n{pd.DataFrame(dbscan_sweep)}")

    # ---- PCA (visualization only) ----
    log("Computing PCA(2) projection for visualization...")
    pca_coords, explained_var = clustering.pca_projection(X_eligible)
    log(f"PCA explained variance ratio: {explained_var}, sum={explained_var.sum():.3f}")

    # ---- final K-Means model ----
    log(f"Fitting final K-Means (k={config.N_CLUSTERS})...")
    final_pipe = clustering.fit_final_model(X_eligible, k=config.N_CLUSTERS)
    eligible = eligible.copy()
    eligible["cluster"] = final_pipe.named_steps["kmeans"].labels_
    eligible["pc1"] = pca_coords[:, 0]
    eligible["pc2"] = pca_coords[:, 1]
    eligible["dbscan_label"] = dbscan_labels
    eligible["dbscan_is_noise"] = dbscan_labels == -1

    # ---- health score (bounds fit on eligible, applied to everyone) ----
    bounds = profiling.fit_health_score_bounds(eligible)
    eligible["health_score"] = profiling.compute_health_score(eligible, bounds)
    excluded = excluded.copy()
    excluded["health_score"] = profiling.compute_health_score(excluded, bounds)
    excluded["cluster"] = -1
    excluded["pc1"] = np.nan
    excluded["pc2"] = np.nan
    excluded["dbscan_label"] = np.nan
    excluded["dbscan_is_noise"] = False

    # ---- cluster profiling & naming ----
    # Shares are computed against the FULL seller base (eligible + excluded),
    # not just the eligible population, so the "New / Low Data" bucket's
    # share is on the same footing and every segment's revenue_share_pct
    # sums to 100% together in the app's charts.
    grand_total_revenue = seller_features["total_revenue"].sum()
    grand_total_sellers = len(seller_features)
    profile_table = profiling.build_cluster_profile_table(
        eligible, total_revenue=grand_total_revenue, total_sellers=grand_total_sellers
    )
    log(f"cluster profile table:\n{profile_table}")

    cluster_to_key = profiling.name_segments(profile_table, eligible)
    log(f"cluster -> segment key mapping: {cluster_to_key}")
    impacts = profiling.quantify_recommendations(profile_table, cluster_to_key)

    eligible["segment_key"] = eligible["cluster"].map(cluster_to_key)
    eligible["segment_name"] = eligible["segment_key"].map(lambda k: profiling.SEGMENT_META[k]["name"])
    excluded["segment_key"] = "new_low_data"
    excluded["segment_name"] = "New / Low Data"

    key_to_color = {key: color for key, color in zip(
        ["premium", "emerging", "declining", "underperforming", "atrisk"],
        [config.PALETTE["gold"], config.PALETTE["teal"], config.PALETTE["violet"], config.PALETTE["rust"], config.PALETTE["red"]],
    )}

    all_sellers = pd.concat([eligible, excluded], ignore_index=True, sort=False)
    all_sellers["segment_color"] = all_sellers["segment_key"].map(
        lambda k: key_to_color.get(k, config.EXCLUDED_SEGMENT_COLOR)
    )
    all_sellers["health_tier"] = all_sellers["health_score"].apply(profiling.health_tier)

    all_sellers.to_parquet(config.PROCESSED_FILES["sellers_segmented"], index=False)
    seller_features.to_parquet(config.PROCESSED_FILES["seller_features"], index=False)
    log(f"sellers_segmented.parquet written: {len(all_sellers):,} rows")

    # ---- assemble cluster_profiles.json (business layer) ----
    profile_records = []
    for _, row in profile_table.iterrows():
        cid = int(row["cluster"])
        key = cluster_to_key[cid]
        meta = profiling.SEGMENT_META[key]
        profile_records.append(
            {
                "cluster": cid,
                "segment_key": key,
                "name": meta["name"],
                "action_tag": meta["action_tag"],
                "description": meta["description"],
                "action": meta["action"],
                "impact": impacts.get(cid, ""),
                "color": key_to_color[key],
                "n_sellers": int(row["n_sellers"]),
                "seller_share_pct": round(float(row["seller_share_pct"]), 2),
                "revenue_share_pct": round(float(row["revenue_share_pct"]), 2),
                "median_revenue": round(float(row["median_revenue"]), 2),
                "median_frequency": round(float(row["median_frequency"]), 2),
                "median_recency_days": round(float(row["median_recency_days"]), 1),
                "median_avg_delay_days": round(float(row["median_avg_delay_days"]), 2),
                "median_neg_review_rate": round(float(row["median_neg_review_rate"]), 4),
                "median_cancel_rate": round(float(row["median_cancel_rate"]), 4),
                "median_health": round(float(row["median_health"]), 1),
            }
        )

    excluded_total_revenue = excluded["total_revenue"].sum()
    total_revenue_all = all_sellers["total_revenue"].sum()
    profile_records.append(
        {
            "cluster": -1,
            "segment_key": "new_low_data",
            "name": "New / Low Data",
            "action_tag": "MONITOR",
            "description": (
                f"Sellers with fewer than {config.MIN_ORDERS_THRESHOLD} distinct orders. Rate-based features "
                "(review, cancellation) are degenerate at this volume, so they are excluded from the K-Means "
                "population rather than distorting cluster centroids -- but they are never dropped from the "
                "business view."
            ),
            "action": (
                "No formal segmentation yet. Monitor early quality signals (first-order review, first delivery "
                f"delay) and re-evaluate once a seller crosses {config.MIN_ORDERS_THRESHOLD} orders."
            ),
            "impact": (
                f"{len(excluded):,} sellers ({100*len(excluded)/len(all_sellers):.1f}% of the base) contribute "
                f"{100*excluded_total_revenue/total_revenue_all:.1f}% of total GMV -- excluding them from the "
                "clustering model costs negligible revenue coverage."
            ),
            "color": config.EXCLUDED_SEGMENT_COLOR,
            "n_sellers": int(len(excluded)),
            "seller_share_pct": round(100 * len(excluded) / len(all_sellers), 2),
            "revenue_share_pct": round(100 * excluded_total_revenue / total_revenue_all, 2),
            "median_revenue": round(float(excluded["total_revenue"].median()), 2) if len(excluded) else 0,
            "median_frequency": round(float(excluded["frequency"].median()), 2) if len(excluded) else 0,
            "median_recency_days": round(float(excluded["recency_days"].median()), 1) if len(excluded) else 0,
            "median_avg_delay_days": round(float(excluded["avg_delay_days"].median()), 2) if len(excluded) else 0,
            "median_neg_review_rate": round(float(excluded["neg_review_rate"].median()), 4) if len(excluded) else 0,
            "median_cancel_rate": round(float(excluded["cancel_rate"].median()), 4) if len(excluded) else 0,
            "median_health": round(float(excluded["health_score"].median()), 1) if len(excluded) else 0,
        }
    )

    with open(config.PROCESSED_FILES["cluster_profiles"], "w") as f:
        json.dump(profile_records, f, indent=2)
    log("cluster_profiles.json written")

    # ---- model_diagnostics.json (Segmentation / Methodology tabs) ----
    diagnostics = {
        "reference_date": str(reference_date.date()),
        "min_orders_threshold": config.MIN_ORDERS_THRESHOLD,
        "n_clusters": config.N_CLUSTERS,
        "seed": config.SEED,
        "cluster_features": config.CLUSTER_FEATURES,
        "candidate_features": config.CANDIDATE_FEATURES,
        "n_sellers_total": int(len(all_sellers)),
        "n_sellers_eligible": int(len(eligible)),
        "n_sellers_excluded": int(len(excluded)),
        "multi_seller_order_stats": multi_seller_stats,
        "threshold_sensitivity": sensitivity.to_dict(orient="records"),
        "correlation_matrix": corr_matrix.to_dict(),
        "elbow_silhouette": elbow_df.to_dict(orient="records"),
        "chosen_k_silhouette": float(elbow_df.loc[elbow_df["k"] == config.N_CLUSTERS, "silhouette"].iloc[0]),
        "stability_ari": stability_df.to_dict(orient="records"),
        "stability_ari_mean": float(stability_df["ari"].mean()),
        "n_stability_seeds": len(config.STABILITY_SEEDS),
        "dbscan": dbscan_result,
        "dbscan_eps_sweep": dbscan_sweep,
        "dbscan_k_distance": kth_distances.tolist(),
        "pca_explained_variance_ratio": explained_var.tolist(),
        "pca_explained_variance_sum": float(explained_var.sum()),
        "health_score_weights": config.HEALTH_SCORE_WEIGHTS,
        "health_score_bounds": bounds,
    }
    with open(config.PROCESSED_FILES["model_diagnostics"], "w") as f:
        json.dump(diagnostics, f, indent=2)
    log("model_diagnostics.json written")

    log("Pipeline complete.")


if __name__ == "__main__":
    main()
