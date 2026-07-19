"""Load the raw Olist CSVs and validate referential integrity before any joins.

The Olist schema is a near-star schema centered on `orders`. `order_items` is
the only table linking an order to a seller, which is why it is treated as
the central table of this project (see features.build_seller_order_pairs).
"""
from __future__ import annotations

import json

import pandas as pd

from . import config

TIMESTAMP_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
    "order_items": ["shipping_limit_date"],
}


def load_raw_tables() -> dict[str, pd.DataFrame]:
    """Read every raw CSV with the right dtypes, parsing dates once at load time."""
    tables: dict[str, pd.DataFrame] = {}
    for name, path in config.RAW_FILES.items():
        parse_dates = TIMESTAMP_COLUMNS.get(name)
        tables[name] = pd.read_csv(path, parse_dates=parse_dates)
    return tables


def validate_relational_integrity(orders: pd.DataFrame, items: pd.DataFrame, reviews: pd.DataFrame) -> dict:
    """Referential-integrity checks run before any join, to catch silent row loss early."""
    order_ids = set(orders["order_id"])
    item_order_ids = set(items["order_id"])
    review_counts = reviews.groupby("order_id").size()

    return {
        "n_orders": int(len(orders)),
        "n_order_items": int(len(items)),
        "orphan_items": int((~items["order_id"].isin(order_ids)).sum()),
        "orders_without_items": int((~orders["order_id"].isin(item_order_ids)).sum()),
        "duplicate_reviews_per_order": int((review_counts > 1).sum()),
        "orders_with_review": int(reviews["order_id"].nunique()),
        "delivered_orders_missing_delivered_date": int(
            orders.loc[orders["order_status"] == "delivered", "order_delivered_customer_date"].isna().sum()
        ),
    }


def dedupe_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Olist has orders with more than one review row; keep the most recent per order_id."""
    return (
        reviews.sort_values("review_answer_timestamp")
        .drop_duplicates(subset="order_id", keep="last")
        .loc[:, ["order_id", "review_score"]]
    )


def aggregate_geolocation(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Collapse geolocation's many rows per zip prefix to one row (mean lat/lng).

    Joining on zip prefix without this step explodes cardinality: a single
    zip prefix can have dozens of raw lat/lng samples in the source table.
    """
    return (
        geolocation.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean"))
        .rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"})
    )


def load_and_validate() -> tuple[dict[str, pd.DataFrame], dict]:
    """Convenience entry point used by the pipeline script and notebooks."""
    tables = load_raw_tables()
    report = validate_relational_integrity(tables["orders"], tables["order_items"], tables["order_reviews"])

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PROCESSED_FILES["data_quality_report"], "w") as f:
        json.dump(report, f, indent=2)

    return tables, report
