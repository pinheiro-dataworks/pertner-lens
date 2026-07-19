"""Seller-level feature engineering, built up from the seller-order pair.

Why seller-order pairs and not raw order_items or raw orders: an order_id in
Olist can contain items from more than one seller (it's a marketplace). Sum
straight from order_items and a 3-item order from one seller inflates
frequency to 3 "events". Aggregate straight from orders and a seller who sold
one item in a 5-item order gets credited the order's full revenue. Both
distort RFM. Materializing the seller x order_id grain first makes revenue
"the seller's share of the order" and frequency "distinct orders" correct by
construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_seller_order_pairs(items: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """One row = one seller's participation in one order."""
    pairs = items.groupby(["seller_id", "order_id"], as_index=False).agg(
        seller_revenue=("price", "sum"),
        seller_freight=("freight_value", "sum"),
        n_items=("order_item_id", "count"),
    )
    order_cols = [
        "order_id",
        "order_status",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    return pairs.merge(orders[order_cols], on="order_id", how="left")


def attach_review_scores(pairs: pd.DataFrame, reviews_dedup: pd.DataFrame) -> pd.DataFrame:
    """Reviews are attached at the order level, not the line-item level.

    Known limitation, kept honest rather than hidden: in a multi-seller order,
    every participating seller inherits the same review score, even if only
    one of them caused it. See multi_seller_order_stats() for the measured
    blast radius of this assumption (~1.3% of orders, ~2% of revenue -- see
    the Methodology tab / notebooks/01_eda.ipynb).
    """
    return pairs.merge(reviews_dedup, on="order_id", how="left")


def multi_seller_order_stats(items: pd.DataFrame) -> dict:
    """Quantify, not assume, how many orders/how much revenue the multi-seller review-attribution limitation touches."""
    sellers_per_order = items.groupby("order_id")["seller_id"].nunique()
    multi_order_ids = sellers_per_order[sellers_per_order > 1].index
    total_orders = len(sellers_per_order)
    total_revenue = items["price"].sum()
    multi_revenue = items.loc[items["order_id"].isin(multi_order_ids), "price"].sum()
    return {
        "total_orders_with_items": int(total_orders),
        "multi_seller_orders": int(len(multi_order_ids)),
        "multi_seller_orders_pct": round(100 * len(multi_order_ids) / total_orders, 2),
        "multi_seller_revenue_pct": round(100 * multi_revenue / total_revenue, 2),
    }


def build_seller_features(pairs: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Collapse seller-order pairs (with review scores attached) to one row per seller."""
    g = pairs.groupby("seller_id")

    frequency = g["order_id"].nunique().rename("frequency")
    total_revenue = g["seller_revenue"].sum().rename("total_revenue")
    last_order_date = g["order_purchase_timestamp"].max().rename("last_order_date")
    # Cancel rate uses ALL seller-order pairs, before any delivered-only
    # filter -- computing it after that filter would divide by the wrong
    # denominator (only delivered orders), silently hiding cancellations.
    cancel_rate = g["order_status"].apply(lambda s: (s == "canceled").mean()).rename("cancel_rate")
    neg_review_rate = (
        g["review_score"]
        .apply(lambda s: (s <= 2).mean() if s.notna().any() else np.nan)
        .rename("neg_review_rate")
    )

    delivered = pairs[pairs["order_status"] == "delivered"].dropna(subset=["order_delivered_customer_date"]).copy()
    delivered["delivery_days"] = (
        delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
    ).dt.days
    # Delay vs. the promised date, not just absolute delivery time: this is
    # what actually breaks buyer trust (a promise missed), so it is the
    # feature used downstream, not raw delivery_days.
    delivered["delay_days"] = (
        delivered["order_delivered_customer_date"] - delivered["order_estimated_delivery_date"]
    ).dt.days
    dg = delivered.groupby("seller_id")
    avg_delivery_days = dg["delivery_days"].mean().rename("avg_delivery_days")
    avg_delay_days = dg["delay_days"].mean().rename("avg_delay_days")
    n_delivered_orders = dg.size().rename("n_delivered_orders")

    features = pd.concat(
        [frequency, total_revenue, last_order_date, cancel_rate, neg_review_rate, avg_delivery_days, avg_delay_days, n_delivered_orders],
        axis=1,
    ).reset_index()

    features["avg_ticket"] = features["total_revenue"] / features["frequency"]
    features["recency_days"] = (reference_date - features["last_order_date"]).dt.days
    features["n_delivered_orders"] = features["n_delivered_orders"].fillna(0).astype(int)

    # Sellers with zero delivered orders (everything pending/canceled/
    # unavailable) have no delivery signal. Impute with the population
    # median rather than dropping the seller, and flag it explicitly so the
    # app never presents an imputed number as measured.
    features["delivery_data_imputed"] = features["avg_delivery_days"].isna()
    features["avg_delivery_days"] = features["avg_delivery_days"].fillna(features["avg_delivery_days"].median())
    features["avg_delay_days"] = features["avg_delay_days"].fillna(features["avg_delay_days"].median())

    features["review_data_imputed"] = features["neg_review_rate"].isna()
    features["neg_review_rate"] = features["neg_review_rate"].fillna(features["neg_review_rate"].median())

    return features


def apply_min_order_threshold(features: pd.DataFrame, min_orders: int = config.MIN_ORDERS_THRESHOLD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the seller base into the clustering-eligible population and the excluded low-data tail.

    The excluded tail is never dropped from the business layer: it becomes
    the "New / Low Data" bucket in the app so every seller still gets a
    profile.
    """
    eligible = features[features["frequency"] >= min_orders].copy()
    excluded = features[features["frequency"] < min_orders].copy()
    return eligible, excluded


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """log1p on monetary/count features: both are heavily right-skewed by a
    handful of high-GMV sellers, which would otherwise dominate Euclidean
    distance and dedicate whole clusters to a handful of outliers.
    """
    df = df.copy()
    df["log_revenue"] = np.log1p(df["total_revenue"])
    df["log_frequency"] = np.log1p(df["frequency"])
    return df


def threshold_sensitivity(features: pd.DataFrame, thresholds=(1, 2, 3, 5, 10, 15, 20)) -> pd.DataFrame:
    """How many sellers / how much revenue survive each candidate minimum-order threshold."""
    total_sellers = len(features)
    total_revenue = features["total_revenue"].sum()
    rows = []
    for t in thresholds:
        kept = features[features["frequency"] >= t]
        rows.append(
            {
                "threshold": t,
                "sellers_kept": int(len(kept)),
                "sellers_kept_pct": round(100 * len(kept) / total_sellers, 1),
                "revenue_kept_pct": round(100 * kept["total_revenue"].sum() / total_revenue, 1),
            }
        )
    return pd.DataFrame(rows)
