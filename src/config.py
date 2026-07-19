"""Central configuration for the PartnerLens pipeline.

Every path, seed, threshold and weight used across notebooks and src/ modules
is defined here so there is a single place to answer "where is X decided?".
"""
from pathlib import Path

# ---------------------------------------------------------------- paths ----
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
APP_ASSETS_DIR = ROOT_DIR / "app" / "assets"

RAW_FILES = {
    "orders": DATA_RAW_DIR / "olist_orders_dataset.csv",
    "order_items": DATA_RAW_DIR / "olist_order_items_dataset.csv",
    "order_payments": DATA_RAW_DIR / "olist_order_payments_dataset.csv",
    "order_reviews": DATA_RAW_DIR / "olist_order_reviews_dataset.csv",
    "products": DATA_RAW_DIR / "olist_products_dataset.csv",
    "sellers": DATA_RAW_DIR / "olist_sellers_dataset.csv",
    "customers": DATA_RAW_DIR / "olist_customers_dataset.csv",
    "geolocation": DATA_RAW_DIR / "olist_geolocation_dataset.csv",
    "category_translation": DATA_RAW_DIR / "product_category_name_translation.csv",
}

PROCESSED_FILES = {
    "seller_order_pairs": DATA_PROCESSED_DIR / "seller_order_pairs.parquet",
    "seller_features": DATA_PROCESSED_DIR / "seller_features_all.parquet",
    "sellers_segmented": DATA_PROCESSED_DIR / "sellers_segmented.parquet",
    "cluster_profiles": DATA_PROCESSED_DIR / "cluster_profiles.json",
    "model_diagnostics": DATA_PROCESSED_DIR / "model_diagnostics.json",
    "data_quality_report": DATA_PROCESSED_DIR / "data_quality_report.json",
}

# ---------------------------------------------------------- reproducibility --
SEED = 42

# ------------------------------------------------------------- thresholds --
# Sellers with fewer distinct orders than this have degenerate rate-based
# features (e.g. a negative-review rate that can only be 0%, 50% or 100% on
# 1-2 orders), which distorts cluster centroids. They are excluded from the
# clustering population but never dropped from the business layer: they are
# reported as a separate "New / Low Data" bucket. See
# notebooks/02_features.ipynb for the sensitivity analysis behind this value.
MIN_ORDERS_THRESHOLD = 5

# Chosen k for the delivered segmentation. Selected via elbow + silhouette +
# stability (ARI across seeds) + business interpretability -- the full
# reasoning is in notebooks/03_clustering.ipynb and the app's Methodology tab.
N_CLUSTERS = 5
K_CANDIDATES = list(range(2, 11))

KMEANS_N_INIT = 20
STABILITY_SEEDS = [1, 7, 21, 42, 99]

# DBSCAN is run as a critical stress test, never as the delivered segmentation
# (see Methodology tab for why density-based clustering doesn't fit this
# feature space). min_samples follows the common eps ~ 2 * n_features heuristic
# capped to a business-meaningful minimum cluster size.
DBSCAN_MIN_SAMPLES = 15

# Features feeding the clustering model, in original business units before
# scaling. Monetary/count features are log1p-transformed (see features.py)
# because revenue and order volume are heavily right-skewed by a small number
# of high-GMV sellers, which would otherwise dominate Euclidean distance.
RAW_FEATURES = ["total_revenue", "frequency", "recency_days", "avg_delay_days", "neg_review_rate", "cancel_rate"]

# Candidate set shown on the Feature Correlation chart, BEFORE the drop
# decision below -- this is what the diagnostic is for.
CANDIDATE_FEATURES = ["log_revenue", "log_frequency", "recency_days", "avg_delay_days", "neg_review_rate", "cancel_rate"]

# log_revenue and log_frequency correlate at 0.80 on the eligible population
# (see notebooks/03_clustering.ipynb / Methodology tab) -- comfortably past
# the ~0.7 flag threshold. Keeping both would double-weight one underlying
# "scale" dimension in Euclidean distance. Decision: drop frequency from the
# clustering feature set and keep revenue, since revenue already captures
# both order count and ticket size and is the more business-critical axis
# for GMV-based prioritization. Frequency is not discarded from the project
# -- it still feeds the health score and every profile card.
CLUSTER_FEATURES = ["log_revenue", "recency_days", "avg_delay_days", "neg_review_rate", "cancel_rate"]

FEATURE_LABELS = {
    "log_revenue": "Revenue (log)",
    "log_frequency": "Frequency (log)",
    "recency_days": "Recency",
    "avg_delay_days": "Delivery delay",
    "neg_review_rate": "Neg. review rate",
    "cancel_rate": "Cancel rate",
    "total_revenue": "Revenue",
    "frequency": "Frequency",
}

# Direction correction so that +1 always means "healthier" once z-scored:
# recency, delay, negative reviews and cancellations are "bad when high", so
# they enter the health score inverted.
INVERTED_FEATURES = {"recency_days", "avg_delay_days", "neg_review_rate", "cancel_rate"}

# Health score weights (sum to 1.0). Quality (delay + negative reviews +
# cancellations = 0.55) is weighted above raw volume (revenue + frequency +
# recency = 0.45) on purpose: in a marketplace, the cost of one bad partner is
# buyer-trust erosion that bleeds into the whole platform, not just their own
# GMV. This is a business decision, not a statistical one -- documented here
# so it is never "numerology" (see Methodology tab).
HEALTH_SCORE_WEIGHTS = {
    "revenue": 0.20,
    "frequency": 0.10,
    "recency": 0.15,
    "delay": 0.20,
    "neg_review": 0.25,
    "cancel": 0.10,
}

# ------------------------------------------------------------------ design --
PALETTE = {
    "bg": "#14171C",
    "bg_elevated": "#1B1F27",
    "surface": "#1F242E",
    "surface_hover": "#262C38",
    "border": "#2B303C",
    "border_strong": "#3A4152",
    "text_primary": "#ECE9E2",
    "text_secondary": "#9CA1AD",
    "text_tertiary": "#666C79",
    "gold": "#E8A33D",
    "rust": "#C1694A",
    "violet": "#8B85C7",
    "teal": "#4FB3A9",
    "red": "#D6685F",
    "green": "#6FBF7D",
}

# Assigned to clusters in rank order (highest health score first) at profiling
# time -- see profiling.name_segments(). Kept separate from cluster_id because
# raw KMeans labels are arbitrary and not stable in meaning across re-runs.
SEGMENT_COLOR_ORDER = [PALETTE["gold"], PALETTE["teal"], PALETTE["violet"], PALETTE["rust"], PALETTE["red"]]
EXCLUDED_SEGMENT_COLOR = PALETTE["text_tertiary"]

PROJECT_NAME = "PartnerLens"
PROJECT_TAGLINE = "Partner Segmentation & Performance Intelligence"
AUTHOR_NAME = "Renan Pinheiro"
AUTHOR_GITHUB = "https://github.com/pinheiro-dataworks"
