"""Generate notebooks/01_eda.ipynb, 02_features.ipynb, 03_clustering.ipynb from
cell definitions below, so the storytelling notebooks and src/ modules can
never silently drift apart (the notebooks import the same functions the
pipeline and app use).

Run with: python scripts/build_notebooks.py
Then execute each with:
  jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"

SETUP_CELL = '''\
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent))
from src import clustering, config, data_loader, features, profiling

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)
'''

# ---------------------------------------------------------------- 01_eda ----
EDA_CELLS = [
    ("md", """# 01 — Exploratory Data Analysis

PartnerLens segments Olist **sellers**, not customers, so this notebook explores the
raw relational schema through that lens: what links an order to a seller, where the
schema has traps that would silently distort seller-level aggregates, and how skewed
the underlying distributions are before any transformation.

Findings here justify every decision in `src/features.py` and `src/data_loader.py`."""),
    ("code", SETUP_CELL),
    ("md", "## 1. Load raw tables and check referential integrity\n\nEvery join risks silent row loss. Check before joining, not after."),
    ("code", '''\
tables, quality_report = data_loader.load_and_validate()
quality_report'''),
    ("md", """`orphan_items = 0` confirms every `order_items` row has a matching order — safe to
inner/left join without losing item rows. `orders_without_items` (775) are orders that
never got a line item (unavailable/cancelled before fulfillment); they contribute to
cancellation-rate features but must not be summed into revenue. `duplicate_reviews_per_order`
(547) and `delivered_orders_missing_delivered_date` (8) are handled explicitly in
`data_loader.dedupe_reviews` and `features.build_seller_features`."""),
    ("md", "## 2. The central limitation: orders with more than one seller\n\nOlist is a marketplace — `order_items` is the only table linking an order to a seller, and a single `order_id` can contain items from multiple sellers."),
    ("code", '''\
mss = features.multi_seller_order_stats(tables["order_items"])
mss'''),
    ("code", '''\
sellers_per_order = tables["order_items"].groupby("order_id")["seller_id"].nunique()

fig, ax = plt.subplots(figsize=(6, 3.5))
sellers_per_order.value_counts().sort_index().plot(kind="bar", ax=ax, color="#4FB3A9")
ax.set_xlabel("distinct sellers in the order")
ax.set_ylabel("number of orders (log scale)")
ax.set_yscale("log")
ax.set_title(f"{mss['multi_seller_orders_pct']}% of orders span >1 seller "
             f"({mss['multi_seller_revenue_pct']}% of revenue)")
plt.tight_layout()
plt.show()'''),
    ("md", """Only **1.3% of orders** (1,278 of 98,666) involve more than one seller, carrying
**~2% of total revenue**. This is the exact blast radius of the multi-seller
review-attribution limitation described in `features.attach_review_scores`: an order's
review score is assigned to every participating seller because Olist attaches reviews
at the order level, not the line-item level. Measuring it — rather than assuming it away
— shows the distortion is marginal, not a reason to discard the feature."""),
    ("md", "## 3. Known Olist traps, quantified"),
    ("code", '''\
geo_raw = tables["geolocation"]
geo_agg = data_loader.aggregate_geolocation(geo_raw)
print(f"geolocation: {len(geo_raw):,} raw rows -> {len(geo_agg):,} rows after "
      f"aggregating to one row per zip prefix ({len(geo_raw)/len(geo_agg):.1f}x cardinality)")

review_counts = tables["order_reviews"].groupby("order_id").size()
print(f"reviews: {(review_counts > 1).sum():,} orders have more than one review row "
      f"({100*(review_counts > 1).mean():.2f}%) -- deduplicated by keeping the most recent")

delivered = tables["orders"][tables["orders"]["order_status"] == "delivered"]
print(f"delivered orders missing delivered_customer_date: "
      f"{delivered['order_delivered_customer_date'].isna().sum()} of {len(delivered):,}")'''),
    ("md", "## 4. Revenue and order-value skew\n\nMonetary features are the reason `log1p` is applied before scaling in `features.add_log_features`."),
    ("code", '''\
item_revenue = tables["order_items"]["price"]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].hist(item_revenue, bins=80, color="#C1694A")
axes[0].set_title("order_items.price -- raw")
axes[0].set_xlabel("R$")
axes[1].hist(np.log1p(item_revenue), bins=80, color="#4FB3A9")
axes[1].set_title("order_items.price -- log1p")
axes[1].set_xlabel("log1p(R$)")
plt.tight_layout()
plt.show()

print(item_revenue.describe(percentiles=[.5, .9, .99]))'''),
    ("md", """A right-skewed distribution like this, aggregated to seller level, is exactly what would
let 5-10 outlier high-GMV sellers dominate Euclidean distance in K-Means if left untransformed."""),
    ("md", "## 5. Review score is bimodal, not normal\n\nThis is why `features.build_seller_features` uses a *rate* (share of orders scoring ≤2) rather than the mean review score."),
    ("code", '''\
fig, ax = plt.subplots(figsize=(6, 3.5))
tables["order_reviews"]["review_score"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#8B85C7")
ax.set_xlabel("review_score")
ax.set_ylabel("count")
ax.set_title("Review scores cluster at the extremes, not the center")
plt.tight_layout()
plt.show()'''),
    ("md", "## Next\n\n`02_features.ipynb` builds the seller-order pair table and the full seller-level feature matrix from these raw tables."),
]

# ----------------------------------------------------------- 02_features ----
FEATURES_CELLS = [
    ("md", """# 02 — Feature Engineering

Builds the seller-order pair table (the atomic unit for every seller-level metric) and
aggregates it into one row per seller. See `src/features.py` for the implementation;
this notebook is the narrative and sensitivity analysis behind those choices."""),
    ("code", SETUP_CELL),
    ("code", '''\
tables, _ = data_loader.load_and_validate()
reviews_dedup = data_loader.dedupe_reviews(tables["order_reviews"])
reference_date = tables["orders"]["order_purchase_timestamp"].max()
print("reference date:", reference_date)'''),
    ("md", """## 1. Seller-order pairs

One row = one seller's participation in one order. Aggregating straight from
`order_items` (sum per seller) would inflate frequency when a seller sells >1 item in
the same order; aggregating straight from `orders` would credit a seller with revenue
from items they didn't sell. The pair table makes both correct by construction."""),
    ("code", '''\
pairs = features.build_seller_order_pairs(tables["order_items"], tables["orders"])
pairs = features.attach_review_scores(pairs, reviews_dedup)
print(f"{len(pairs):,} seller-order pairs from {tables['order_items']['order_id'].nunique():,} orders "
      f"and {pairs['seller_id'].nunique():,} sellers")
pairs.head(3)'''),
    ("md", "## 2. Seller-level feature matrix"),
    ("code", '''\
seller_features = features.build_seller_features(pairs, reference_date)
seller_features = features.add_log_features(seller_features)
print(f"{len(seller_features):,} sellers, {seller_features['delivery_data_imputed'].sum()} with "
      f"imputed delivery metrics, {seller_features['review_data_imputed'].sum()} with imputed review rate")
seller_features[config.RAW_FEATURES + ["frequency"]].describe()'''),
    ("md", """## 3. Minimum-order threshold: sensitivity analysis

Sellers with 1-2 orders have degenerate rate features -- a negative-review rate can only
land on 0%, 50% or 100%. The question is where to draw the line, and what it costs in
revenue coverage."""),
    ("code", '''\
sens = features.threshold_sensitivity(seller_features)
sens'''),
    ("code", '''\
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(sens["threshold"], sens["sellers_kept_pct"], "o-", color="#4FB3A9", label="% sellers kept")
ax1.plot(sens["threshold"], sens["revenue_kept_pct"], "o-", color="#E8A33D", label="% revenue kept")
ax1.axvline(config.MIN_ORDERS_THRESHOLD, color="#666C79", linestyle=":", label=f"chosen threshold = {config.MIN_ORDERS_THRESHOLD}")
ax1.set_xlabel("minimum distinct orders")
ax1.set_ylabel("%")
ax1.legend()
ax1.set_title("Raising the order threshold sheds sellers far faster than it sheds revenue")
plt.tight_layout()
plt.show()'''),
    ("md", """At the chosen threshold (**≥5 orders**, `config.MIN_ORDERS_THRESHOLD`), the eligible
population keeps **58.0% of sellers** but **95.8% of revenue** -- the model covers the
overwhelming majority of GMV while dropping the half of the base whose rate-based
features would otherwise be noise. The excluded ~42% are not discarded from the project:
they become the "New / Low Data" bucket in the business layer (`profiling.py`,
`cluster_profiles.json`), so every seller still gets a profile."""),
    ("md", "## 4. Threshold split"),
    ("code", '''\
eligible, excluded = features.apply_min_order_threshold(seller_features)
print(f"eligible: {len(eligible):,} sellers -- excluded (new/low data): {len(excluded):,} sellers")'''),
    ("md", "## 5. Cancel rate: denominator matters\n\nComputed over **every** seller-order pair, before any delivered-only filter -- filtering to delivered orders first would divide by the wrong denominator and hide cancellations entirely."),
    ("code", '''\
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(eligible["cancel_rate"], bins=40, color="#D6685F")
ax.set_xlabel("cancel_rate")
ax.set_title(f"median cancel_rate = {eligible['cancel_rate'].median():.3f}, "
             f"{100*(eligible['cancel_rate'] > 0).mean():.1f}% of sellers have any cancellations")
plt.tight_layout()
plt.show()'''),
    ("md", "## Next\n\n`03_clustering.ipynb` takes `eligible` from here, checks feature correlation, selects k, stress-tests with DBSCAN, and fits the final segmentation."),
]

# --------------------------------------------------------- 03_clustering ----
CLUSTERING_CELLS = [
    ("md", """# 03 — Clustering & Segmentation

K-Means on `StandardScaler`-transformed, `log1p`-treated seller features, with an
honest DBSCAN stress test and a PCA(2) projection used strictly for visualization.
Produces the artifacts consumed by the Streamlit app's Segmentation and Profiles tabs."""),
    ("code", SETUP_CELL),
    ("code", '''\
tables, _ = data_loader.load_and_validate()
reviews_dedup = data_loader.dedupe_reviews(tables["order_reviews"])
reference_date = tables["orders"]["order_purchase_timestamp"].max()

pairs = features.build_seller_order_pairs(tables["order_items"], tables["orders"])
pairs = features.attach_review_scores(pairs, reviews_dedup)
seller_features = features.build_seller_features(pairs, reference_date)
seller_features = features.add_log_features(seller_features)
eligible, excluded = features.apply_min_order_threshold(seller_features)
print(f"eligible: {len(eligible):,}")'''),
    ("md", "## 1. Feature correlation -- does the candidate set need trimming?"),
    ("code", '''\
corr = eligible[config.CANDIDATE_FEATURES].corr(method="pearson")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, shrink=0.8)
plt.tight_layout()
plt.show()'''),
    ("md", """`log_revenue` and `log_frequency` correlate at **0.80** -- comfortably past the ~0.7 flag
threshold. Keeping both would double-weight one underlying "scale" dimension in Euclidean
distance. **Decision: drop `log_frequency` from the clustering feature set, keep
`log_revenue`.** Revenue already captures both order count and ticket size and is the
more business-critical axis for GMV-based prioritization; frequency is not discarded from
the project, it still feeds the health score and every profile card. See
`config.CLUSTER_FEATURES` vs `config.CANDIDATE_FEATURES`."""),
    ("code", "X = eligible[config.CLUSTER_FEATURES].to_numpy()\nconfig.CLUSTER_FEATURES"),
    ("md", "## 2. Scaler choice\n\nK-Means uses Euclidean distance. `MinMaxScaler` is defined by the extremes of each feature, so a single outlier compresses the other 99% of points into a narrow slice of that dimension, effectively erasing it from the distance calculation. `StandardScaler` (z-score) preserves relative structure and, combined with the `log1p` already applied to revenue, leaves every feature reasonably well-behaved. `RobustScaler` (median/IQR-based) was considered as a third option and would be worth an A/B check if outliers dominated post-log1p -- they don't here, so the simpler, more common choice (`StandardScaler`) is used."),
    ("md", "## 3. k selection -- elbow & silhouette"),
    ("code", '''\
elbow_df = clustering.elbow_silhouette(X)
elbow_df'''),
    ("code", '''\
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(elbow_df["k"], elbow_df["inertia"], "o-", color="#4FB3A9")
ax1.set_xlabel("k"); ax1.set_ylabel("inertia", color="#4FB3A9")
ax2 = ax1.twinx()
ax2.plot(elbow_df["k"], elbow_df["silhouette"], "o-", color="#E8A33D")
ax2.set_ylabel("silhouette", color="#E8A33D")
ax1.axvline(config.N_CLUSTERS, color="#666C79", linestyle=":")
ax1.set_title(f"chosen k={config.N_CLUSTERS}")
plt.tight_layout()
plt.show()'''),
    ("md", """The elbow flattens from k=4 onward -- it narrows the candidate range, it doesn't pick a
winner. Silhouette's **global** maximum is the coarse k=2 solution (0.41); k=5 is a
**local** peak (0.25, higher than both k=4's 0.23 and k=6's 0.23) but not the statistical
optimum. This is expected for continuous behavioral data with no natural density valleys
between clusters -- a modest silhouette is not a red flag here, it's the shape of the
problem (see the DBSCAN section below, which hits the same wall from a different angle).

**k=5 is chosen on business interpretability, not silhouette alone**: k=2/3 collapse
segments that need opposite actions (e.g. a healthy low-volume seller and a
quality-risk seller both fall into one "small seller" bucket at k=2), which defeats the
purpose of a segmentation meant to drive differentiated partner actions."""),
    ("md", "## 4. Stability check -- Adjusted Rand Index across seeds\n\nClusters that survive re-initialization are real structure, not an artifact of a lucky starting point."),
    ("code", '''\
stability_df = clustering.stability_ari(X, k=config.N_CLUSTERS)
print(f"mean pairwise ARI across {len(config.STABILITY_SEEDS)} seeds: {stability_df['ari'].mean():.3f}")
stability_df'''),
    ("md", "An ARI this close to 1.0 across independent seeds means the k=5 partition is essentially deterministic given this feature space -- not sensitive to K-Means' random initialization."),
    ("md", "## 5. DBSCAN -- a critical stress test, not a competing segmenter\n\nDBSCAN finds clusters as dense regions separated by density valleys. Seller performance features form a continuous gradient with no such valleys, so the honest expectation is one giant cluster plus a noise fraction -- and that's what should show up below."),
    ("code", '''\
kth_dist = clustering.k_distance_data(X, k=config.DBSCAN_MIN_SAMPLES)
eps_knee = clustering.estimate_eps_from_knee(kth_dist)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(kth_dist, color="#8B85C7")
ax.axhline(eps_knee, color="#E8A33D", linestyle=":", label=f"knee eps ≈ {eps_knee:.2f}")
ax.set_xlabel(f"points, sorted by distance to {config.DBSCAN_MIN_SAMPLES}th nearest neighbor")
ax.set_ylabel("distance")
ax.legend()
ax.set_title("k-distance plot")
plt.tight_layout()
plt.show()'''),
    ("code", '''\
sweep = []
for mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
    r = clustering.run_dbscan(X, eps=eps_knee * mult)
    r.pop("labels")
    r["eps_multiplier"] = mult
    sweep.append(r)
pd.DataFrame(sweep)'''),
    ("md", """As eps grows from half the knee to 1.5x the knee, the noise fraction falls monotonically
(from ~30% toward single digits) while the cluster count never rises above 1 -- there is no
eps that produces several well-separated, business-meaningful clusters. DBSCAN either
strands a large minority of sellers as noise or collapses everyone into one blob,
confirming the density-based assumption doesn't hold on this feature space.

**Reframing the result as insight rather than a dead end:** the points DBSCAN flags as
noise are, by definition, sellers sitting in sparse regions of the feature space --
behavioral outliers. Cross-referencing that noise set against the K-Means segments (next
section) turns a "failed" second algorithm into an outlier-detection cross-check on the
first one."""),
    ("md", "## 6. PCA -- for the scatter plot only, never the clustering space"),
    ("code", '''\
pca_coords, explained_var = clustering.pca_projection(X)
print(f"explained variance: PC1={explained_var[0]:.1%}, PC2={explained_var[1]:.1%}, "
      f"sum={explained_var.sum():.1%}")'''),
    ("md", """PC1+PC2 explain **~54%** of variance -- below the ~60% mark that would let the 2D
scatter be read as a faithful map of inter-cluster distance. The projection below is
**illustrative**: useful for seeing that clusters occupy distinguishable regions, not for
reading exact distances between them. Clustering itself was run on the original 5
scaled features, each with direct business meaning, precisely so centroids translate
back into "this segment has high revenue / high recency" rather than an uninterpretable
PCA axis."""),
    ("md", "## 7. Final model, cluster profiles, and segment naming"),
    ("code", '''\
final_pipe = clustering.fit_final_model(X, k=config.N_CLUSTERS)
eligible = eligible.copy()
eligible["cluster"] = final_pipe.named_steps["kmeans"].labels_
eligible["pc1"], eligible["pc2"] = pca_coords[:, 0], pca_coords[:, 1]

fig, ax = plt.subplots(figsize=(6.5, 5))
colors = ["#E8A33D", "#4FB3A9", "#8B85C7", "#C1694A", "#D6685F"]
for c, color in zip(sorted(eligible["cluster"].unique()), colors):
    sub = eligible[eligible["cluster"] == c]
    ax.scatter(sub["pc1"], sub["pc2"], s=8, alpha=0.5, color=color, label=f"cluster {c}")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(markerscale=2)
ax.set_title("K-Means (k=5) clusters projected onto PC1/PC2")
plt.tight_layout()
plt.show()'''),
    ("code", '''\
bounds = profiling.fit_health_score_bounds(eligible)
eligible["health_score"] = profiling.compute_health_score(eligible, bounds)

grand_total_revenue = seller_features["total_revenue"].sum()
grand_total_sellers = len(seller_features)
profile_table = profiling.build_cluster_profile_table(
    eligible, total_revenue=grand_total_revenue, total_sellers=grand_total_sellers
)
cluster_to_key = profiling.name_segments(profile_table, eligible)
profile_table["segment_name"] = profile_table["cluster"].map(
    lambda c: profiling.SEGMENT_META[cluster_to_key[c]]["name"]
)
profile_table[["cluster", "segment_name", "n_sellers", "seller_share_pct", "revenue_share_pct", "median_health"]]'''),
    ("md", """Names are assigned by rank (healthiest -> Premium, most inactive of the rest ->
Declining, least healthy remainder -> At Risk, and the final two split by health into
Emerging / Underperforming), never by raw cluster id, since K-Means labels are arbitrary
integers that carry no meaning on their own.

Worth flagging honestly: the original design sketch anticipated a "high volume, low
quality" archetype. It doesn't appear in the real k=5 solution -- above-median order
frequency only shows up in the healthiest cluster. Quality problems (elevated negative
reviews, worse delivery delay) concentrate in **low/mid-volume** sellers instead. The
segment name and description reflect what the data actually shows."""),
    ("md", "## 8. Health score distribution within each segment\n\nA named segment can still hide an internal tail worth acting on."),
    ("code", '''\
fig, ax = plt.subplots(figsize=(7, 4))
order = profile_table.sort_values("median_health", ascending=False)["cluster"]
data = [eligible.loc[eligible["cluster"] == c, "health_score"] for c in order]
labels = [profiling.SEGMENT_META[cluster_to_key[c]]["name"] for c in order]
ax.boxplot(data, tick_labels=labels, patch_artist=True,
           boxprops=dict(facecolor="#1F242E", color="#9CA1AD"),
           medianprops=dict(color="#E8A33D"))
plt.xticks(rotation=20, ha="right")
ax.set_ylabel("health score")
ax.set_title("Even 'healthy' segments carry a lower tail worth watching")
plt.tight_layout()
plt.show()'''),
    ("md", "## Next\n\nRun `python scripts/build_pipeline.py` to regenerate the processed artifacts (`data/processed/*.parquet`, `cluster_profiles.json`, `model_diagnostics.json`) that the Streamlit app in `app/streamlit_app.py` reads at boot -- the app itself trains nothing."),
]


def build_notebook(cells_spec, path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(content) if kind == "md" else nbf.v4.new_code_cell(content)
        for kind, content in cells_spec
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {path}")


if __name__ == "__main__":
    build_notebook(EDA_CELLS, NOTEBOOKS_DIR / "01_eda.ipynb")
    build_notebook(FEATURES_CELLS, NOTEBOOKS_DIR / "02_features.ipynb")
    build_notebook(CLUSTERING_CELLS, NOTEBOOKS_DIR / "03_clustering.ipynb")
