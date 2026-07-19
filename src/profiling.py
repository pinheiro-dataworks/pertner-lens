"""Business layer: health score, cluster -> named-profile translation, and
quantified recommendations. This is where the clustering output turns into a
document a partnerships team can act on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _minmax_norm(series: pd.Series, lo: float, hi: float) -> pd.Series:
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return ((series - lo) / (hi - lo)).clip(0, 1)


def fit_health_score_bounds(eligible: pd.DataFrame) -> dict:
    """1st/99th percentile bounds fit on the clustering-eligible population
    only, so a handful of extreme low-data sellers can't compress the scale."""
    bounds = {}
    for feat in ["total_revenue", "frequency", "recency_days", "avg_delay_days"]:
        base = np.log1p(eligible[feat]) if feat in ("total_revenue", "frequency") else eligible[feat]
        bounds[feat] = (float(base.quantile(0.01)), float(base.quantile(0.99)))
    return bounds


def compute_health_score(df: pd.DataFrame, bounds: dict, weights: dict = config.HEALTH_SCORE_WEIGHTS) -> pd.Series:
    """Weighted 0-100 composite. Recency, delay, negative reviews and
    cancellations are 'bad when high', so they enter inverted -- +1 always
    means healthier once combined. Weights favor quality (0.55) over raw
    volume (0.45); see config.HEALTH_SCORE_WEIGHTS for the business rationale.
    """
    rev = _minmax_norm(np.log1p(df["total_revenue"]), *bounds["total_revenue"])
    freq = _minmax_norm(np.log1p(df["frequency"]), *bounds["frequency"])
    recency = 1 - _minmax_norm(df["recency_days"], *bounds["recency_days"])
    delay = 1 - _minmax_norm(df["avg_delay_days"], *bounds["avg_delay_days"])
    neg_review = 1 - df["neg_review_rate"].clip(0, 1)
    cancel = 1 - df["cancel_rate"].clip(0, 1)

    score = (
        weights["revenue"] * rev
        + weights["frequency"] * freq
        + weights["recency"] * recency
        + weights["delay"] * delay
        + weights["neg_review"] * neg_review
        + weights["cancel"] * cancel
    )
    return (100 * score).clip(0, 100)


def health_tier(score: float) -> str:
    if score >= 75:
        return "Strong"
    if score >= 50:
        return "Stable"
    if score >= 30:
        return "Fragile"
    return "Critical"


def health_tier_color(score: float) -> str:
    if score >= 75:
        return config.PALETTE["green"]
    if score >= 50:
        return config.PALETTE["gold"]
    if score >= 30:
        return config.PALETTE["rust"]
    return config.PALETTE["red"]


def build_cluster_profile_table(
    df: pd.DataFrame, cluster_col: str = "cluster", total_revenue: float | None = None, total_sellers: int | None = None
) -> pd.DataFrame:
    """Per-cluster median of every raw (unscaled) feature, plus size and GMV
    share. Segment names are defended by this table, not by feeling.

    total_revenue/total_sellers default to sums over df itself, but should be
    passed explicitly as the FULL seller base (clustered + excluded) whenever
    this table will be shown alongside the excluded "New / Low Data" bucket --
    otherwise the two population's shares are computed against different
    denominators and won't sum to 100% together.
    """
    total_revenue = df["total_revenue"].sum() if total_revenue is None else total_revenue
    total_sellers = len(df) if total_sellers is None else total_sellers
    rows = []
    for cid, g in df.groupby(cluster_col):
        rows.append(
            {
                "cluster": int(cid),
                "n_sellers": int(len(g)),
                "seller_share_pct": 100 * len(g) / total_sellers,
                "revenue_share_pct": 100 * g["total_revenue"].sum() / total_revenue,
                "median_revenue": float(g["total_revenue"].median()),
                "median_frequency": float(g["frequency"].median()),
                "median_recency_days": float(g["recency_days"].median()),
                "median_avg_delay_days": float(g["avg_delay_days"].median()),
                "median_neg_review_rate": float(g["neg_review_rate"].median()),
                "median_cancel_rate": float(g["cancel_rate"].median()),
                "median_health": float(g["health_score"].median()),
                "mean_health": float(g["health_score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("median_health", ascending=False).reset_index(drop=True)


def name_segments(profile_table: pd.DataFrame, population: pd.DataFrame) -> dict[int, str]:
    """Deterministic, general cluster -> business-profile mapping (a valid
    bijection over exactly 5 clusters by construction: each rule below removes
    one cluster from the remaining pool, so nothing can be assigned twice).

    KMeans labels are arbitrary integers with no inherent meaning and are not
    stable across re-runs, so clusters are named from where they RANK against
    each other on health, recency and quality -- not from fixed population
    thresholds, which would misfire if the underlying cluster structure shifts.

    On the real Olist population, this resolves to (see
    notebooks/03_clustering.ipynb): the "high order volume + bad quality"
    archetype anticipated from the original design sketch does not actually
    appear -- above-median frequency only shows up in the healthiest cluster.
    Quality problems concentrate in low/mid-volume sellers instead, so the
    5th archetype is named for what the data actually shows (elevated
    complaint/delay rate at modest scale), not forced into a "high volume"
    label it doesn't earn.
    """
    pt = profile_table.copy()
    remaining = set(pt["cluster"])

    def row(cid):
        return pt.loc[pt["cluster"] == cid].iloc[0]

    # 1. Premium & Consistent: the single healthiest cluster.
    premium = max(remaining, key=lambda c: row(c)["median_health"])
    remaining.discard(premium)

    # 2. Declining: of what's left, the one that has gone quiet the longest.
    declining = max(remaining, key=lambda c: row(c)["median_recency_days"])
    remaining.discard(declining)

    # 3. At Risk / Low Engagement: of what's left, the least healthy.
    atrisk = min(remaining, key=lambda c: row(c)["median_health"])
    remaining.discard(atrisk)

    # 4. Of the final two, the healthier one is graduating (Emerging); the
    #    other carries the segment's quality risk (Underperforming).
    emerging, underperforming = sorted(remaining, key=lambda c: row(c)["median_health"], reverse=True)

    return {
        premium: "premium",
        declining: "declining",
        atrisk: "atrisk",
        emerging: "emerging",
        underperforming: "underperforming",
    }


SEGMENT_META = {
    # Description text is written directly against the current pipeline run's
    # cluster_profiles.json medians (see notebooks/03_clustering.ipynb) rather
    # than generic archetype copy -- if the pipeline is re-run with a
    # different threshold, feature set or k, re-check these numbers still
    # hold before shipping.
    "premium": {
        "name": "Premium & Consistent",
        "action_tag": "RETAIN",
        "description": (
            "The 22% of sellers who generate roughly 81% of marketplace GMV: median R$8,100 revenue, ~59 "
            "orders, and the most recent activity of any segment. Delivery is the fastest in the base, but the "
            "negative-review rate tracks the marketplace median rather than beating it -- scale alone doesn't "
            "buy better reviews, which is why quality investment still matters even here."
        ),
        "action": (
            "Priority support tier and early access to new marketplace placements. Negotiate loyalty-linked "
            "commission tiers before a competing marketplace poaches them -- this segment is the one most "
            "worth over-investing in."
        ),
    },
    "underperforming": {
        "name": "Underperforming - Quality Risk",
        "action_tag": "DEVELOP",
        "description": (
            "Modest, unremarkable order volume carrying a negative-review rate more than double the "
            "marketplace median and the weakest on-time delivery margin of any segment. Not a high-volume "
            "segment -- the risk here is reputational, not revenue concentration: this is where quality "
            "complaints cluster relative to scale."
        ),
        "action": (
            "Structured 90-day quality plan: delivery SLA, a review-remediation checklist, and milestone "
            "tracking. Escalate to enforcement (see At Risk) only if metrics don't move after two review cycles."
        ),
    },
    "declining": {
        "name": "Declining",
        "action_tag": "REACTIVATE",
        "description": (
            "Real revenue while active (median R$1,200+), but a median of roughly 13 months since the last "
            "order -- more than five times the marketplace's typical recency. Quality was never the issue here "
            "(review and delivery metrics in line with the marketplace); disengagement is."
        ),
        "action": (
            "Targeted win-back outreach from a named account contact, paired with a temporary incentive "
            "(fee waiver or featured placement) to resume listing activity."
        ),
    },
    "emerging": {
        "name": "Emerging",
        "action_tag": "GROW",
        "description": (
            "Lower-volume sellers (median ~10 orders) with the strongest quality signal in the marketplace: a "
            "negative-review rate roughly half the population median. Recency is in line with the typical "
            "seller -- these are healthy, not-yet-scaled partners, not a risk segment."
        ),
        "action": (
            "Onboarding acceleration and seller-education content aimed at order velocity, plus limited "
            "marketing placement support to help them find early traction."
        ),
    },
    "atrisk": {
        "name": "At Risk / Low Engagement",
        "action_tag": "DEPRIORITIZE",
        "description": (
            "The smallest and least healthy segment: lowest revenue and order volume, long gaps between "
            "orders, the worst negative-review rate in the base, and the only segment carrying a meaningful "
            "cancellation rate. The compounding-risk tail every open marketplace accumulates."
        ),
        "action": (
            "Low-cost automated reactivation nudge first. If there's no response within one review window, "
            "deprioritize placement or move toward delisting to protect marketplace quality and cut support overhead."
        ),
    },
}


def quantify_recommendations(profile_table: pd.DataFrame, cluster_to_key: dict[int, str]) -> dict[int, str]:
    """Impact statements computed from the actual profile table -- no placeholders."""
    key_to_cluster = {v: k for k, v in cluster_to_key.items()}
    impacts: dict[int, str] = {}

    if "premium" in key_to_cluster:
        row = profile_table.set_index("cluster").loc[key_to_cluster["premium"]]
        ratio = row["revenue_share_pct"] / row["seller_share_pct"] if row["seller_share_pct"] else float("nan")
        impacts[key_to_cluster["premium"]] = (
            f"{row['seller_share_pct']:.1f}% of sellers generate {row['revenue_share_pct']:.1f}% of GMV -- "
            f"a {ratio:.1f}x revenue concentration relative to headcount, the strongest ratio of any segment."
        )

    if "underperforming" in key_to_cluster:
        row = profile_table.set_index("cluster").loc[key_to_cluster["underperforming"]]
        pop_neg = profile_table["median_neg_review_rate"].median()
        impacts[key_to_cluster["underperforming"]] = (
            f"{row['n_sellers']:.0f} sellers ({row['seller_share_pct']:.1f}% of the seller base) run a "
            f"{row['median_neg_review_rate']*100:.1f}% negative-review rate against a {pop_neg*100:.1f}% segment "
            f"median elsewhere -- this is where quality complaints concentrate, not where GMV is concentrated, "
            f"which is exactly why a remediation plan is cheaper here than a revenue write-off."
        )

    if "declining" in key_to_cluster:
        row = profile_table.set_index("cluster").loc[key_to_cluster["declining"]]
        opportunity = row["n_sellers"] * row["median_revenue"]
        impacts[key_to_cluster["declining"]] = (
            f"{row['n_sellers']:.0f} sellers with a median historical revenue of "
            f"R$ {row['median_revenue']:,.0f} each (R$ {opportunity:,.0f} combined) generated while active -- "
            f"the size of the win-back opportunity if reactivation succeeds."
        )

    if "emerging" in key_to_cluster:
        row = profile_table.set_index("cluster").loc[key_to_cluster["emerging"]]
        impacts[key_to_cluster["emerging"]] = (
            f"{row['n_sellers']:.0f} sellers ({row['seller_share_pct']:.1f}% of the base) already median health "
            f"score {row['median_health']:.0f}/100 on low volume -- track this cohort's health-score trajectory "
            f"over the following quarters to validate that 'emerging' predicts graduation to 'premium'."
        )

    if "atrisk" in key_to_cluster:
        row = profile_table.set_index("cluster").loc[key_to_cluster["atrisk"]]
        impacts[key_to_cluster["atrisk"]] = (
            f"{row['seller_share_pct']:.1f}% of the seller base but only {row['revenue_share_pct']:.1f}% of GMV -- "
            f"deprioritizing the unresponsive tail of this segment is a net positive for support cost, not a GMV risk."
        )

    return impacts
