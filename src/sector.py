"""Academia vs Industry sector classification for institutions.

WHAT THIS DOES & HOW (read me — this is a heuristic, not ground truth)
----------------------------------------------------------------------
OpenAlex gives us raw institution *names* but no machine-readable sector label.
We infer a sector for each institution name with a transparent, rule-based
keyword classifier (no training data required), in this priority order:

1. INDUSTRY  - the name matches a curated list of well-known technology /
               pharma companies (Google, Microsoft, Huawei, Pfizer, ...) OR
               ends in a corporate suffix (Inc, Ltd, LLC, GmbH, Corp, AG, ...).
2. GOVERNMENT/OTHER - national labs, ministries, academies of sciences, NASA,
               CNRS, Max Planck, etc. (public research that is neither a
               company nor a degree-granting university).
3. ACADEMIA  - names containing university / college / institute of technology /
               polytechnic / école / hochschule / school of ... etc.
4. OTHER     - anything we cannot confidently place (incl. hospitals).

A *paper* usually lists several institutions, so we also derive a paper-level
collaboration category from the set of sectors involved:

    "Academia"           - only academic affiliations
    "Industry"           - only company affiliations
    "Academia–Industry"  - at least one of each (the interesting collaboration)
    "Other / Mixed"      - everything else (government-only, unknown, ...)

LIMITATIONS: keyword matching misclassifies edge cases (e.g. a company named
"… Institute", or a university press). It is meant to reveal broad structural
trends (e.g. the rise of academia–industry collaboration in AI), not to label
any single paper authoritatively.
"""
from __future__ import annotations

# Curated company names frequently seen as AI-research affiliations.
_COMPANIES = {
    "google", "deepmind", "alphabet", "microsoft", "meta", "facebook", "openai",
    "anthropic", "amazon", "apple", "ibm", "nvidia", "intel", "huawei", "tencent",
    "baidu", "alibaba", "samsung", "sony", "bytedance", "bosch", "siemens",
    "qualcomm", "adobe", "yahoo", "netflix", "uber", "bell labs", "toyota",
    "honda", "nokia", "ericsson", "sap ", "salesforce", "oracle", "tesla",
    "cohere", "hitachi", "fujitsu", "mitsubishi", "philips", "general electric",
    "pfizer", "roche", "novartis", "astrazeneca", "genentech", "merck",
    "johnson & johnson", "sanofi", "glaxosmithkline", "boeing", "raytheon",
    "lockheed", "xerox", "yandex", "naver", "kakao", "didi", "megvii",
    "sensetime", "nec ", "panasonic", "schlumberger", "shell", "exxon",
}
_CORP_SUFFIXES = (" inc", " inc.", " ltd", " ltd.", " llc", " corp", " corp.",
                  " gmbh", " co.", " co ", " ag", " s.a", " plc", " pte",
                  " pty", " kk", " srl", " bv", " nv")
_CORP_WORDS = ("technologies", "laboratories inc", "research labs", "motors",
               "pharmaceutical", "pharmaceuticals", "semiconductor")

_GOV_KEYWORDS = ("national laboratory", "national lab", "ministry", "academy of sciences",
                 "national institutes of health", "nasa", "cnrs", "max planck",
                 "chinese academy", "national research council", "national center",
                 "national centre", "government", "veterans affairs", "los alamos",
                 "oak ridge", "argonne", "national physical laboratory")

_ACADEMIA_KEYWORDS = ("universit", "college", "institute of technology", "polytechnic",
                      "école", "ecole", "hochschule", "politecnico", "universidade",
                      "school of", "faculty", "institut ", "institute", "academy",
                      "epfl", "eth ", "mit", "caltech")

INDUSTRY = "Industry"
ACADEMIA = "Academia"
GOVERNMENT = "Government/Other"
OTHER = "Other"

PAPER_CATEGORIES = ["Academia", "Industry", "Academia–Industry", "Other / Mixed"]


def classify_institution(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return OTHER
    n = name.lower()
    if any(c in n for c in _COMPANIES) or n.endswith(_CORP_SUFFIXES) \
            or any(w in n for w in _CORP_WORDS) or any(s in n for s in _CORP_SUFFIXES):
        return INDUSTRY
    if any(k in n for k in _GOV_KEYWORDS):
        return GOVERNMENT
    if any(k in n for k in _ACADEMIA_KEYWORDS):
        return ACADEMIA
    return OTHER


def paper_sector(institutions: list[str]) -> str:
    """Collaboration category for a paper given its institution list."""
    sectors = {classify_institution(i) for i in institutions if i}
    has_a = ACADEMIA in sectors
    has_i = INDUSTRY in sectors
    if has_a and has_i:
        return "Academia–Industry"
    if has_i:
        return "Industry"
    if has_a:
        return "Academia"
    return "Other / Mixed"
