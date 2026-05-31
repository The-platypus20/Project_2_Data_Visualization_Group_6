"""Machine-learning & NLP analytics powering the "ML & NLP" tab.

Three independent, cached components (all computed on the full corpus so they
are stable and fast to re-render; the cost is paid once):

1. cluster_titles(k)  - unsupervised NLP. TF-IDF on paper titles -> KMeans into
   k clusters -> 2-D TruncatedSVD projection for plotting. Returns the per-paper
   cluster labels, 2-D coordinates, the top terms per cluster, and a silhouette
   quality score (on a subsample).

2. forecast_publications(cutoff, horizon) - time-series ML. Fits Holt's linear
   trend (exponential smoothing) on annual publication counts up to ``cutoff``
   and projects ``horizon`` years ahead with an approximate confidence band.

3. citation_model() - supervised ML. A Random Forest regresses log-citations on
   paper metadata (age, references, team size, openness, topic, sector, venue).
   Returns held-out R², predicted-vs-actual points, and feature importances —
   i.e. *which factors are associated with higher citation impact*.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from .data import load_data


# ---------------------------------------------------------------------------
# 1. NLP: title clustering
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=8)
def cluster_titles(k: int = 6, max_features: int = 2000, sample: int = 4000):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import Normalizer

    df = load_data()
    titles = df["title"].fillna("").astype(str)
    mask = titles.str.len() > 0

    # sublinear_tf + dropping ubiquitous terms reduces the "everything is about
    # learning/data" effect that otherwise collapses titles into one cluster.
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          max_features=max_features, min_df=5, max_df=0.4,
                          sublinear_tf=True)
    X = vec.fit_transform(titles[mask])
    terms = np.array(vec.get_feature_names_out())

    # Standard LSA pipeline: SVD then L2-normalise so KMeans uses cosine-style
    # distance, which yields far more balanced, interpretable clusters.
    svd = TruncatedSVD(n_components=min(100, X.shape[1] - 1), random_state=0)
    Xr = Normalizer(copy=False).fit_transform(svd.fit_transform(X))

    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(Xr)

    # Top terms per cluster from the mean TF-IDF vector of its members.
    top_terms = {}
    Xarr = X
    for c in range(k):
        rows = np.where(labels == c)[0]
        if rows.size == 0:
            top_terms[c] = []
            continue
        mean_tfidf = np.asarray(Xarr[rows].mean(axis=0)).ravel()
        top_terms[c] = list(terms[mean_tfidf.argsort()[::-1][:10]])

    # Silhouette on a subsample (full pairwise is too costly at 15k).
    rng = np.random.default_rng(0)
    idx = rng.choice(Xr.shape[0], size=min(2000, Xr.shape[0]), replace=False)
    sil = float(silhouette_score(Xr[idx], labels[idx])) if k > 1 else float("nan")

    coords = pd.DataFrame({
        "x": Xr[:, 0], "y": Xr[:, 1], "cluster": labels,
        "title": titles[mask].values,
        "year": df.loc[mask, "year"].values,
        "citations": df.loc[mask, "citation_count"].values,
    })
    if len(coords) > sample:
        coords = coords.sample(sample, random_state=0)

    sizes = pd.Series(labels).value_counts().sort_index()
    return {"coords": coords, "top_terms": top_terms, "sizes": sizes,
            "silhouette": sil, "labels": labels, "n": int(mask.sum())}


# ---------------------------------------------------------------------------
# 2. Time-series: publication forecast
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=16)
def forecast_publications(cutoff: int, horizon: int = 4):
    from statsmodels.tsa.holtwinters import Holt

    df = load_data()
    counts = df.groupby("year").size().sort_index()
    train = counts[counts.index <= cutoff]
    if len(train) < 3:
        return {"history": counts, "forecast": pd.Series(dtype=float),
                "lower": pd.Series(dtype=float), "upper": pd.Series(dtype=float),
                "cutoff": cutoff}

    model = Holt(train.values.astype(float), initialization_method="estimated").fit()
    fc = model.forecast(horizon)
    resid_std = float(np.std(train.values - model.fittedvalues))
    years = np.arange(cutoff + 1, cutoff + 1 + horizon)
    fc = pd.Series(np.clip(fc, 0, None), index=years)
    band = 1.96 * resid_std
    return {"history": counts, "forecast": fc,
            "lower": (fc - band).clip(lower=0), "upper": fc + band,
            "cutoff": cutoff, "fitted": pd.Series(model.fittedvalues, index=train.index)}


# ---------------------------------------------------------------------------
# 3. Supervised: predicting high citation velocity
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def citation_model():
    """Classify whether a paper reaches the TOP QUARTILE of citation velocity
    (citations per year). Using velocity rather than raw citations controls for
    paper age, so the model speaks to *intrinsic* impact drivers.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split

    df = load_data().copy()
    num = ["paper_age", "referenced_works_count", "author_count",
           "institution_count", "country_count"]
    for c in num:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["is_oa_int"] = df["is_oa"].astype(int)

    cat = pd.get_dummies(df[["topic_bucket", "sector", "venue_group"]],
                         prefix=["topic", "sector", "venue"])
    X = pd.concat([df[num + ["is_oa_int"]], cat], axis=1).fillna(0)
    threshold = df["citations_per_year"].quantile(0.75)
    y = (df["citations_per_year"] >= threshold).astype(int).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0,
                                          stratify=y)
    clf = RandomForestClassifier(n_estimators=250, max_depth=16, n_jobs=-1,
                                 class_weight="balanced", random_state=0)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(yte, proba))
    acc = float(accuracy_score(yte, pred))
    fpr, tpr, _ = roc_curve(yte, proba)
    cm = confusion_matrix(yte, pred)

    imp = (pd.Series(clf.feature_importances_, index=X.columns)
           .sort_values(ascending=False).head(12))
    imp.index = (imp.index.str.replace("topic_", "Topic: ")
                 .str.replace("sector_", "Sector: ")
                 .str.replace("venue_", "Venue: ")
                 .str.replace("referenced_works_count", "Reference count")
                 .str.replace("paper_age", "Paper age")
                 .str.replace("author_count", "Author count")
                 .str.replace("institution_count", "Institution count")
                 .str.replace("country_count", "Country count")
                 .str.replace("is_oa_int", "Open access"))
    return {"auc": auc, "accuracy": acc, "importance": imp,
            "roc": (fpr, tpr), "confusion": cm, "threshold": float(threshold)}
