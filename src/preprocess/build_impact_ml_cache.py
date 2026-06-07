"""Build the Tab 3 "The Anatomy of Impact" ML cache files.

Reads the paper-level merged files ONCE and writes small, dashboard-ready CSVs
into Dataset/dashboard_cache/ (all prefixed ``tab3_``). The Shiny app only reads
these small files, so every chart renders instantly at demo time.

Impact label: a paper is "high impact" if its citation_velocity (citations per
year) is in the top 10% OF ITS OWN PUBLICATION YEAR. Ranking within each year
removes the age bias that would otherwise punish recent papers.

Run:
  python src/preprocess/build_impact_ml_cache.py \
    --input Dataset/ai_works_merge_2000_2009.csv Dataset/ai_works_merge_2010_2019.csv Dataset/ai_works_merge_2020_2025.csv \
    --output-dir Dataset/dashboard_cache

Outputs:
  tab3_lorenz.csv          Lorenz curve points (paper_frac, citation_frac)  [beat 1]
  tab3_funnel.csv          citation-threshold funnel counts                 [beat 1]
  tab3_concentration.csv   median / mean / never-cited / top1% / gini       [beat 1]
  tab3_correlations.csv    high-impact rate by group, per dimension         [beat 2]
  tab3_drivers.csv         standardized logistic-regression coefficients    [beat 3]
  tab3_model_metrics.csv   baseline / logreg / gradient-boosting scores     [beat 4]
  tab3_roc_curve.csv       ROC curve points for the gradient model          [beat 4]
  tab3_calibration.csv     reliability curve (predicted vs observed)        [beat 4]
  tab3_forecast.csv        LSTM forecast of high-impact share by family     [beat 5]
"""
from __future__ import annotations

import os

# macOS fix: numpy ships GNU OpenMP (libgomp) while LightGBM ships LLVM OpenMP
# (libomp). Loading both OpenMP runtimes in one process deadlocks gbm.fit().
# These must be set BEFORE numpy / lightgbm import, so they sit at the very top.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Same 7-family taxonomy the rest of the dashboard uses (mirrors
# build_core_dashboard_cache.classify_bucket) so Tab 3 stays consistent.
TOPIC_RULES = [
    ("NLP", ["natural language", "language", "nlp", "text", "speech", "semantic",
             "sentiment", "dialogue", "topic modeling", "translation"]),
    ("Core ML / Deep Learning", ["machine learning", "deep learning", "neural", "classification",
                                 "clustering", "graph neural", "adversarial", "representation learning"]),
    ("ML Theory & Optimization", ["optimization", "bayesian", "probabilistic", "algorithm", "theory",
                                  "causal", "quantum", "data compression"]),
    ("Robotics", ["robot", "robotics", "control", "tracking", "sensor", "autonomous",
                  "planning", "navigation", "motion planning"]),
    ("Healthcare AI", ["health", "healthcare", "medical", "clinical", "cancer", "disease",
                       "diagnosis", "radiology", "patient", "neuroscience"]),
    ("AI Ethics & Fairness", ["privacy", "fairness", "ethics", "bias", "explainable", "xai",
                              "trust", "law", "safety", "intellectual property"]),
    ("Reinforcement Learning", ["reinforcement", "agent", "policy", "reward", "multi-agent", "negotiation"]),
]
FALLBACK_FAMILY = "Applied / Interdisciplinary AI"
USECOLS = ["publication_year", "citation_count", "citation_velocity", "referenced_works_count",
           "primary_topic", "venue_type", "is_oa", "author_count", "institution_count", "country_count"]
IMPACT_QUANTILE = 0.90
RANDOM_STATE = 42
HORIZON, WINDOW, EPOCHS, HIDDEN = 3, 6, 400, 16


def classify_family(topic: object) -> str:
    text = str(topic or "").lower()
    for label, keywords in TOPIC_RULES:
        if any(k in text for k in keywords):
            return label
    return FALLBACK_FAMILY


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output-dir", default="Dataset/dashboard_cache")
    return p.parse_args()


def load_frame(inputs: list[str]) -> pd.DataFrame:
    frames = []
    for f in inputs:
        print(f"Reading {Path(f).name} ...")
        frames.append(pd.read_csv(f, usecols=USECOLS, encoding="utf-8-sig", low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    print(f"  total rows: {len(df):,}")

    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    for c in ["citation_count", "citation_velocity", "referenced_works_count",
              "author_count", "institution_count", "country_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["publication_year"].between(2000, 2025)].copy()
    df["publication_year"] = df["publication_year"].astype(int)
    df["citation_count"] = df["citation_count"].fillna(0)
    df["citation_velocity"] = df["citation_velocity"].fillna(0.0)
    df["is_oa"] = df["is_oa"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["venue_type"] = df["venue_type"].fillna("unknown").astype(str).str.lower()
    df["primary_topic"] = df["primary_topic"].fillna("").astype(str)
    df["family"] = df["primary_topic"].map({t: classify_family(t) for t in df["primary_topic"].unique()})

    cut = df.groupby("publication_year")["citation_velocity"].transform(lambda s: s.quantile(IMPACT_QUANTILE))
    df["high_impact"] = (df["citation_velocity"] >= cut) & (df["citation_velocity"] > 0)
    print(f"  high-impact papers: {int(df['high_impact'].sum()):,} ({100 * df['high_impact'].mean():.1f}%)")
    return df


# ---- Beat 1: concentration ------------------------------------------------- #
def build_concentration(df: pd.DataFrame, out: Path) -> None:
    c = np.sort(df["citation_count"].to_numpy(dtype=float))
    n = len(c)
    total = c.sum()

    # Lorenz curve, downsampled to 101 points
    cum = np.cumsum(c) / max(total, 1)
    idx = np.linspace(0, n - 1, 101).astype(int)
    lorenz = pd.DataFrame({"paper_frac": (idx + 1) / n, "citation_frac": cum[idx]})
    lorenz = pd.concat([pd.DataFrame({"paper_frac": [0.0], "citation_frac": [0.0]}), lorenz], ignore_index=True)
    lorenz.to_csv(out / "tab3_lorenz.csv", index=False)

    # Gini coefficient
    rank = np.arange(1, n + 1)
    gini = (2 * np.sum(rank * c) / (n * total)) - (n + 1) / n if total > 0 else 0.0

    funnel = pd.DataFrame({
        "stage": ["All AI papers", "Cited ≥ 1", "Cited ≥ 10", "Cited ≥ 100", "Cited ≥ 1000"],
        "count": [n, int((c >= 1).sum()), int((c >= 10).sum()),
                  int((c >= 100).sum()), int((c >= 1000).sum())],
    })
    funnel.to_csv(out / "tab3_funnel.csv", index=False)

    top1 = 100.0 * c[-max(1, int(n * 0.01)):].sum() / max(total, 1)
    pd.DataFrame({
        "metric": ["median_citations", "mean_citations", "never_cited_pct",
                   "top1pct_citation_share", "ge1000_pct", "gini"],
        "value": [float(np.median(c)), float(c.mean()), float(100 * (c == 0).mean()),
                  float(top1), float(100 * (c >= 1000).mean()), float(gini)],
    }).to_csv(out / "tab3_concentration.csv", index=False)
    print(f"  beat1: gini={gini:.2f}, top1% hold {top1:.0f}% of citations")


# ---- Beat 2: descriptive correlations -------------------------------------- #
def build_correlations(df: pd.DataFrame, out: Path) -> None:
    rows = []

    def add(dimension, series_labels):
        for order, (label, mask) in enumerate(series_labels):
            sub = df[mask]
            if len(sub) == 0:
                continue
            rows.append({"dimension": dimension, "group": label, "order": order,
                         "n_papers": int(len(sub)),
                         "high_impact_rate": float(100 * sub["high_impact"].mean())})

    add("International collaboration", [
        ("Single country", df["country_count"] <= 1),
        ("Multi-country", df["country_count"] >= 2)])
    add("Open access", [("Closed", ~df["is_oa"]), ("Open access", df["is_oa"])])
    add("Venue type", [
        ("Journal", df["venue_type"] == "journal"),
        ("Conference", df["venue_type"].isin(["proceedings", "proceedings-article", "conference"])),
        ("Other", ~df["venue_type"].isin(["journal", "proceedings", "proceedings-article", "conference"]))])
    team = df["author_count"].fillna(0)
    add("Team size", [
        ("Solo (1)", team == 1), ("Small (2-3)", team.between(2, 3)),
        ("Medium (4-6)", team.between(4, 6)), ("Large (7+)", team >= 7)])
    refs = df["referenced_works_count"].fillna(0)
    add("References", [
        ("0", refs == 0), ("1-9", refs.between(1, 9)),
        ("10-29", refs.between(10, 29)), ("30+", refs >= 30)])

    pd.DataFrame(rows).to_csv(out / "tab3_correlations.csv", index=False)
    print(f"  beat2: {len(rows)} group rows across 5 dimensions")


# ---- Beats 3 + 4: model -------------------------------------------------- #
def build_model(df: pd.DataFrame, out: Path) -> None:
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    feats = pd.DataFrame({
        "References (log)": np.log1p(df["referenced_works_count"].fillna(0)),
        "Team size": df["author_count"].fillna(0),
        "Institutions": df["institution_count"].fillna(0),
        "International reach": df["country_count"].fillna(0),
        "Open access": df["is_oa"].astype(float),
        "Journal venue": (df["venue_type"] == "journal").astype(float),
        "Conference venue": df["venue_type"].isin(["proceedings", "proceedings-article", "conference"]).astype(float),
    })
    y = df["high_impact"].astype(int).to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(feats, y, test_size=0.2,
                                              random_state=RANDOM_STATE, stratify=y)

    # Beat 3: standardized logistic-regression drivers
    scaler = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(scaler.transform(X_tr), y_tr)
    (pd.DataFrame({"feature": feats.columns, "coef": lr.coef_[0]})
     .assign(direction=lambda d: np.where(d["coef"] >= 0, "raises", "lowers"))
     .sort_values("coef")
     .to_csv(out / "tab3_drivers.csv", index=False))

    # Beat 4: gradient boosting performance
    gbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=48,
                         subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=4)
    gbm.fit(X_tr, y_tr)
    p_gbm = gbm.predict_proba(X_te)[:, 1]
    p_lr = lr.predict_proba(scaler.transform(X_te))[:, 1]

    def lift10(yt, sc):
        k = max(1, int(len(sc) * 0.10))
        return float(yt[np.argsort(sc)[::-1][:k]].mean() / max(yt.mean(), 1e-9))

    pd.DataFrame([
        {"model": "Baseline (rate)", "roc_auc": 0.5, "pr_auc": float(y_tr.mean()), "lift_at_10": 1.0},
        {"model": "Logistic regression", "roc_auc": roc_auc_score(y_te, p_lr),
         "pr_auc": average_precision_score(y_te, p_lr), "lift_at_10": lift10(y_te, p_lr)},
        {"model": "Gradient boosting", "roc_auc": roc_auc_score(y_te, p_gbm),
         "pr_auc": average_precision_score(y_te, p_gbm), "lift_at_10": lift10(y_te, p_gbm)},
    ]).to_csv(out / "tab3_model_metrics.csv", index=False)

    fpr, tpr, _ = roc_curve(y_te, p_gbm)
    keep = np.linspace(0, len(fpr) - 1, num=min(len(fpr), 200)).astype(int)
    pd.DataFrame({"fpr": fpr[keep], "tpr": tpr[keep]}).to_csv(out / "tab3_roc_curve.csv", index=False)

    bins = np.linspace(0, 1, 11)
    (pd.DataFrame({"p": p_gbm, "y": y_te})
     .assign(bin=lambda d: pd.cut(d["p"], bins, include_lowest=True))
     .groupby("bin", observed=True)
     .agg(predicted=("p", "mean"), observed=("y", "mean"), count=("y", "size"))
     .reset_index(drop=True)
     .to_csv(out / "tab3_calibration.csv", index=False))
    print(f"  beats3-4: GBM ROC-AUC={roc_auc_score(y_te, p_gbm):.3f}, lift@10={lift10(y_te, p_gbm):.2f}")


# ---- Beat 5: LSTM forecast of high-impact share by family ------------------ #
def build_forecast(df: pd.DataFrame, out: Path) -> None:
    import torch
    from torch import nn

    # macOS fix: torch ships its own libomp; combined with numpy's libgomp and
    # sklearn's libomp this deadlocks torch's parallel ops. The LSTM here is tiny
    # (a few hundred samples), so forcing single-threaded torch costs nothing and
    # sidesteps the OpenMP deadlock entirely.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    hi = df[df["high_impact"]]
    counts = hi.groupby(["family", "publication_year"]).size().reset_index(name="count")
    years = list(range(2000, 2026))
    pivot = counts.pivot(index="family", columns="publication_year", values="count").reindex(columns=years).fillna(0)
    share = pivot.div(pivot.sum(axis=0), axis=1) * 100.0   # each year sums to 100% of high-impact papers

    values = share.to_numpy(dtype=np.float32)
    scale = float(values.max()) or 1.0
    norm = values / scale
    X, Y = [], []
    for row in norm:
        for t in range(WINDOW, row.shape[0]):
            X.append(row[t - WINDOW:t]); Y.append(row[t])
    X = torch.tensor(np.array(X)).unsqueeze(-1)
    Y = torch.tensor(np.array(Y)).unsqueeze(-1)

    class LSTMF(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, HIDDEN, batch_first=True)
            self.head = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1, :])

    model = LSTMF()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(EPOCHS):
        opt.zero_grad(); loss = loss_fn(model(X), Y); loss.backward(); opt.step()
    with torch.no_grad():
        resid = (model(X).squeeze(-1).numpy() - Y.squeeze(-1).numpy()) * scale
    resid_std = float(np.std(resid))

    records = []
    for fam, row in zip(share.index, norm):
        hist = list(row)
        for h in range(1, HORIZON + 1):
            win = torch.tensor(np.array(hist[-WINDOW:], dtype=np.float32)).reshape(1, WINDOW, 1)
            with torch.no_grad():
                nxt = max(float(model(win).item()), 0.0)
            hist.append(nxt)
            pt = nxt * scale
            band = 1.28 * resid_std * np.sqrt(h)
            records.append({"family": fam, "year": years[-1] + h, "share": pt,
                            "kind": "forecast", "lo": max(pt - band, 0.0), "hi": pt + band})
    fc = pd.DataFrame(records)
    for yr in fc["year"].unique():                 # renormalize so families sum to 100%
        m = fc["year"] == yr
        tot = fc.loc[m, "share"].sum()
        if tot > 0:
            fc.loc[m, ["share", "lo", "hi"]] *= 100.0 / tot
    hist_df = (share.reset_index().melt(id_vars="family", var_name="year", value_name="share")
               .assign(kind="history", lo=np.nan, hi=np.nan))
    out_df = pd.concat([hist_df, fc], ignore_index=True).sort_values(["family", "year"])
    out_df.to_csv(out / "tab3_forecast.csv", index=False)
    print(f"  beat5: forecast {len(share)} families, 2026-2028")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_frame(args.input)
    build_concentration(df, out)
    build_correlations(df, out)
    build_model(df, out)
    build_forecast(df, out)
    print("\nAll Tab 3 ML cache files written to", out)


if __name__ == "__main__":
    main()
