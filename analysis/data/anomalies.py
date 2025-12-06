# Anomaly analysis for EquiPro session-level features (trot only)
# - Computes per-horse baselines via z-scores
# - Runs three anomaly-detection methods:
#     1) Z-score thresholding
#     2) K-means clustering distance
#     3) Isolation Forest
# - Compares which sessions are flagged as abnormal by each method
#
# You can run this as a standalone Python script or import pieces into a notebook.

# -----------------------------------------
# 1. Load packages and data
# -----------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd

from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Path to the session-level feature matrix created by the trot-only script
DATA_PATH = Path(__file__).resolve().parent / "session_features_trot.csv"

session_features = pd.read_csv(DATA_PATH)

print("Session-level feature matrix (trot only):")
print(f"  - shape: {session_features.shape}")
print("\nFirst few rows:")
display(session_features.head())

meta_cols = ["horse", "session_id", "n_rows"]
feature_cols = [c for c in session_features.columns if c not in meta_cols]

print("\nNumber of feature columns:", len(feature_cols))


# -----------------------------------------
# 2. Helper: prepare per-horse features
# -----------------------------------------

def prepare_features(
    df_horse: pd.DataFrame,
    feature_cols,
    max_missing_frac: float = 0.4,
) -> pd.DataFrame:
    """Clean and prepare feature matrix for one horse.

    Steps
    -----
    - Keep only the numeric feature columns (drop meta columns).
    - Drop columns that are entirely NaN.
    - Drop columns where more than `max_missing_frac` of entries are NaN.
    - Impute any remaining NaN values with the column median.

    Returns
    -------
    X : pd.DataFrame
        Cleaned feature matrix for this horse (sessions x features).
    """
    X = df_horse[feature_cols].copy()

    # Remove columns that are entirely NaN
    X = X.dropna(axis=1, how="all")

    # Drop columns with too much missing data
    missing_frac = X.isna().mean()
    keep_cols = missing_frac[missing_frac <= max_missing_frac].index
    X = X[keep_cols]

    # Median imputation for any remaining NaNs
    X = X.apply(lambda col: col.fillna(col.median()), axis=0)

    return X


# -----------------------------------------
# 3. Helper: per-horse anomaly analysis
# -----------------------------------------

def analyze_horse_anomalies(
    df_horse: pd.DataFrame,
    feature_cols,
    z_thresh: float = 2.5,
    k_default: int = 2,
    iforest_contamination: float = 0.25,
    random_state: int = 42,
):
    """Run all three anomaly-detection methods for a single horse.

    Methods
    -------
    1) Z-score thresholding:
         - Standardize features
         - Compute max absolute z-score per session
         - Count how many features exceed |z| > z_thresh
    2) K-means distance:
         - Fit K-means in z-scored space
         - Compute distance of each session to its assigned cluster center
         - Label top ~20% (or ~1/3 for very small N) as outliers
    3) Isolation Forest:
         - Run IsolationForest in z-scored space
         - Use -score_samples as anomaly score (higher = more isolated)
         - Use model's predicted label (-1 = anomaly)

    Returns
    -------
    horse_results : pd.DataFrame
        One row per session with meta info + anomaly scores + outlier labels.
    X_z : pd.DataFrame
        Z-scored features for this horse (sessions x features).
    """
    # Reset index so everything is clean
    df_horse = df_horse.reset_index(drop=True)

    # 1) Prepare clean feature matrix
    X = prepare_features(df_horse, feature_cols)
    n_samples = X.shape[0]

    # If too few sessions, return minimal output
    if n_samples < 2:
        print(f"Not enough sessions for horse {df_horse['horse'].iloc[0]} to run anomalies.")
        X_z = pd.DataFrame(index=df_horse.index)
        out = df_horse[["horse", "session_id", "n_rows"]].copy()
        out["z_max_abs"] = np.nan
        out["z_mean_abs"] = np.nan
        out["z_n_features_big"] = 0
        out["z_is_outlier"] = 0
        out["kmeans_dist"] = np.nan
        out["kmeans_is_outlier"] = 0
        out["iforest_score"] = np.nan
        out["iforest_is_outlier"] = 0
        out["n_methods_flagged"] = 0
        return out, X_z

    # 2) Standardize features (z-scores)
    scaler = StandardScaler()
    X_z_values = scaler.fit_transform(X)
    X_z = pd.DataFrame(X_z_values, columns=X.columns, index=df_horse.index)

    # ---------- Method 1: Z-score thresholding ----------
    z_abs = X_z.abs()
    z_max = z_abs.max(axis=1)
    z_mean = z_abs.mean(axis=1)          # NEW: average |z| across features
    z_nbig = (z_abs > z_thresh).sum(axis=1)
    z_label = (z_max > z_thresh).astype(int)

    # ---------- Method 2: K-means distance ----------
    # Choose a reasonable number of clusters given how few sessions we have
    if n_samples >= 3:
        n_clusters = min(k_default, n_samples - 1)
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=random_state,
        )
        cluster_labels = kmeans.fit_predict(X_z)
        centers = kmeans.cluster_centers_

        # Euclidean distance to the assigned cluster center
        dists = []
        for i, x in enumerate(X_z.to_numpy()):
            center = centers[cluster_labels[i]]
            dists.append(np.linalg.norm(x - center))
        dists = pd.Series(dists, index=X_z.index, name="kmeans_dist")

        # Flag the largest distances as outliers
        if n_samples >= 5:
            cutoff = dists.quantile(0.80)  # top 20% as candidate outliers
        else:
            cutoff = dists.quantile(2 / 3)  # for very small N, top ~1/3
        kmeans_label = (dists >= cutoff).astype(int)
    else:
        # If there are only 2 sessions we skip K-means
        dists = pd.Series(np.nan, index=X_z.index, name="kmeans_dist")
        kmeans_label = pd.Series(0, index=X_z.index, name="kmeans_is_outlier")

    # ---------- Method 3: Isolation Forest ----------
    if n_samples >= 5:
        iforest = IsolationForest(
            n_estimators=200,
            contamination=iforest_contamination,
            random_state=random_state,
        )
        if_labels = iforest.fit_predict(X_z)         # -1 = anomaly
        if_scores = -iforest.score_samples(X_z)      # higher = more isolated

        if_label = (pd.Series(if_labels, index=X_z.index) == -1).astype(int)
        if_score = pd.Series(if_scores, index=X_z.index, name="iforest_score")
    else:
        if_label = pd.Series(0, index=X_z.index, name="iforest_is_outlier")
        if_score = pd.Series(np.nan, index=X_z.index, name="iforest_score")

    # Assemble output for this horse
    out = df_horse[["horse", "session_id", "n_rows"]].copy()
    out["z_max_abs"] = z_max
    out["z_mean_abs"] = z_mean        # NEW column
    out["z_n_features_big"] = z_nbig
    out["z_is_outlier"] = z_label
    out["kmeans_dist"] = dists
    out["kmeans_is_outlier"] = kmeans_label
    out["iforest_score"] = if_score
    out["iforest_is_outlier"] = if_label

    # How many methods flag each session?
    method_cols = ["z_is_outlier", "kmeans_is_outlier", "iforest_is_outlier"]
    out["n_methods_flagged"] = out[method_cols].sum(axis=1)

    return out, X_z


# -----------------------------------------
# 4. Run analysis for all horses
# -----------------------------------------

all_results = []
z_spaces = {}  # store z-scored features if you want to look later

for horse in session_features["horse"].unique():
    df_horse = session_features[session_features["horse"] == horse]
    print(f"\n=== Analyzing horse: {horse} ===")
    print(f"Number of sessions: {df_horse.shape[0]}")

    horse_results, X_z = analyze_horse_anomalies(
        df_horse=df_horse,
        feature_cols=feature_cols,
        z_thresh=2.5,
        k_default=2,
        iforest_contamination=0.25,
        random_state=42,
    )

    all_results.append(horse_results)
    z_spaces[horse] = X_z

def top_contributing_features(anomaly_summary, z_spaces, session_id, horse, top_n=5):
    """Return the biomechanical features that deviate most at that session."""
    
    # Grab the correct z-score matrix for that horse
    X_z = z_spaces[horse]

    # Ensure the session ordering in X_z still matches the horse data
    df_horse = anomaly_summary[anomaly_summary["horse"] == horse].reset_index(drop=True)

    # Look up position within this horse’s data
    pos = df_horse[df_horse["session_id"] == session_id].index[0]

    # Extract that row from standardized feature matrix
    z_vals = X_z.loc[pos].abs().sort_values(ascending=False)

    return z_vals.head(top_n)


# Combine results for all horses
anomaly_summary = pd.concat(all_results, ignore_index=True)

print("\nCombined anomaly summary for all horses:")
display(
    anomaly_summary.sort_values(
        ["horse", "n_methods_flagged", "z_max_abs"],
        ascending=[True, False, False],
    )
)

# -----------------------------------------
# 5. Quick sanity checks
# -----------------------------------------

print("\nHow many sessions are flagged by each method?")
for col in ["z_is_outlier", "kmeans_is_outlier", "iforest_is_outlier"]:
    counts = anomaly_summary.groupby("horse")[col].sum()
    print(f"\nMethod: {col}")
    print(counts)

print("\nHow many sessions are flagged by \u22652 methods?")
flagged_two_plus = anomaly_summary[anomaly_summary["n_methods_flagged"] >= 2]
display(
    flagged_two_plus.sort_values(
        ["horse", "n_methods_flagged", "z_max_abs"],
        ascending=[True, False, False],
    )
)


# -----------------------------------------
# 6. Simple visualizations
# -----------------------------------------

import matplotlib.pyplot as plt

# Convert session_id to a true datetime and sort by it
anomaly_summary["session_dt"] = pd.to_datetime(
    anomaly_summary["session_id"].str[:8],  # yyyymmdd
    format="%Y%m%d"
)

# Optional: pretty label for ticks
anomaly_summary["session_label"] = anomaly_summary["session_dt"].dt.strftime("%m/%d/%y")

# Sort once by date so plots are in chronological order
anomaly_summary = anomaly_summary.sort_values("session_dt")

# --- Plot 1: max |z|-score vs. session across horses ---
plt.figure()
for horse, df_h in anomaly_summary.groupby("horse"):
    plt.scatter(df_h["session_dt"], df_h["z_max_abs"], label=horse)

plt.xticks(rotation=90)
plt.ylabel("max |z-score| across features")
plt.xlabel("session date")
plt.title("Session-level deviation from baseline (max |z|)")
plt.legend()
plt.tight_layout()
plt.savefig("fig_max_zscore.png", dpi=300, bbox_inches="tight")

# --- Plot 2: mean |z|-score vs. session across horses ---
plt.figure()
for horse, df_h in anomaly_summary.groupby("horse"):
    plt.scatter(df_h["session_dt"], df_h["z_mean_abs"], label=horse)

plt.xticks(rotation=90)
plt.ylabel("mean |z-score| across features")
plt.xlabel("session date")
plt.title("Session-level deviation from baseline (mean |z|)")
plt.legend()
plt.tight_layout()
plt.savefig("fig_mean_zscore.png", dpi=300, bbox_inches="tight")

# --- Plot 3: number of methods that flag each session ---
plt.figure()
for horse, df_h in anomaly_summary.groupby("horse"):
    plt.scatter(df_h["session_dt"], df_h["n_methods_flagged"], label=horse)

plt.xticks(rotation=90)
plt.ylabel("number of methods flagging session")
plt.xlabel("session date")
plt.title("Agreement between anomaly-detection methods")
plt.legend()
plt.tight_layout()
plt.savefig("fig_method_agreement.png", dpi=300, bbox_inches="tight")



# heat map plot
import matplotlib.pyplot as plt
import seaborn as sns
# Copy relevant columns
heat_df = anomaly_summary[
    ["horse", "session_id", "session_dt", 
     "z_is_outlier", "kmeans_is_outlier", "iforest_is_outlier"]
].copy()

# Create readable date label
heat_df["date_label"] = heat_df["session_dt"].dt.strftime("%m/%d/%y")

# Combine horse + date (e.g., "Duque 08/24/24")
heat_df["session_label"] = heat_df["horse"] + " " + heat_df["date_label"]

# Sort by number of methods flagged (descending)
heat_df["n_flag"] = (
    heat_df[["z_is_outlier", "kmeans_is_outlier", "iforest_is_outlier"]]
    .sum(axis=1)
)
heat_df = heat_df.sort_values(["horse", "n_flag"], ascending=[True, False])

# Rename columns for nicer axis titles
heat_df = heat_df.rename(columns={
    "z_is_outlier": "Z-score",
    "kmeans_is_outlier": "K-means",
    "iforest_is_outlier": "Isolation Forest"
})

# Build heatmap matrix
matrix = heat_df.set_index("session_label")[["Z-score", "K-means", "Isolation Forest"]]

# Plot Heatmap
plt.figure(figsize=(7, max(5, matrix.shape[0] * 0.35)))
sns.heatmap(
    matrix,
    cmap=["white", "tab:red"],   # white = 0 (not flagged), red = 1 (flagged)
    linewidths=0.5,
    linecolor="gray",
    cbar=False,
    annot=True,
    fmt=""
)

plt.xlabel("Anomaly Detection Method")
plt.ylabel("Session (Horse + Date)")
plt.title("Method Agreement Heatmap: Flagged Sessions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("fig_method_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Plot 5: top contributing features for high-confidence anomalies ---

# Re-use the helper you already defined:
# def top_contributing_features(anomaly_summary, z_spaces, session_id, horse, top_n=5):

# Take only sessions flagged by >= 2 methods
flagged_two_plus = anomaly_summary[anomaly_summary["n_methods_flagged"] >= 2]

for _, row in flagged_two_plus.iterrows():
    horse = row["horse"]
    session_id = row["session_id"]

    # Get top |z|-score features for this session
    top_feats = top_contributing_features(
        anomaly_summary=anomaly_summary,
        z_spaces=z_spaces,
        session_id=session_id,
        horse=horse,
        top_n=8,   # change if you want more/less bars
    )

    # Make a horizontal bar plot
    plt.figure(figsize=(6, 4))
    top_feats.sort_values().plot(
        kind="barh",
        title=f"Top Feature Deviations\n{horse} {session_id}",
    )
    plt.xlabel("|z-score| (deviation from horse baseline)")
    plt.ylabel("Biomechanical feature")
    plt.tight_layout()

    # Save one figure per session (safe filename)
    safe_id = session_id.replace(":", "").replace("T", "_")
    out_name = f"fig_top_features_{horse}_{safe_id}.png"
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close()



