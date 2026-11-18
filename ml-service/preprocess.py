"""
Text preprocessing for the Review Intelligence System.

Shared by training and prediction so both paths apply identical transforms.
Goal: normalize noise while preserving sentiment signal — crucially,
negation words (not, no, never…) and negative contractions (don't, isn't,
can't, won't…) are never removed because "not good" and "good" are opposite
sentiments.

NLTK resources (stopwords, wordnet, omw-1.4) are NEVER downloaded at module
import time or during request handling. When NLTK corpora are not installed
locally, preprocessing falls back cleanly to built-in English stopwords and
safe rule-based lemmatization without crashing.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Iterable

logger = logging.getLogger("ml-service.preprocess")

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:  # pragma: no cover - defensive fallback if nltk package is absent
    nltk = None  # type: ignore[assignment]
    nltk_stopwords = None  # type: ignore[assignment]
    WordNetLemmatizer = None  # type: ignore[assignment]

# Built-in fallback English stopwords (standard NLTK English list) used when
# corpora/stopwords is not installed locally.
_FALLBACK_ENGLISH_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an",
    "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can", "couldn",
    "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't",
    "doing", "don", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven",
    "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's",
    "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more",
    "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor",
    "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't",
    "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some",
    "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't",
    "we", "were", "weren", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't",
    "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves",
})

# Negations and negative contractions excluded from stopword removal.
NEGATION_WORDS: frozenset[str] = frozenset({
    # Core negations
    "not", "no", "never", "nor", "neither", "none", "nobody",
    "nothing", "nowhere", "without", "cannot", "can't", "n't",
    # Negative contractions with apostrophe
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't",
    "weren't", "haven't", "hasn't", "hadn't", "won't", "wouldn't",
    "shouldn't", "couldn't", "mustn't", "mightn't", "needn't",
    "shan't", "ain't", "daren't",
    # Negative contractions without apostrophe
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt",
    "werent", "havent", "hasnt", "hadnt", "wont", "wouldnt",
    "shouldnt", "couldnt", "mustnt", "mightnt", "neednt",
    "shant", "aint", "darent", "cant",
    # Contrastive / sentiment-shifting words
    "but", "however", "although", "though", "yet", "against",
    "unless", "except",
})

SENTIMENT_CRITICAL: frozenset[str] = frozenset({
    "very", "too", "so", "really", "extremely", "barely",
    "hardly", "scarcely", "seldom", "rarely", "less", "least",
    "more", "most",
})

_PROTECTED_WORDS: frozenset[str] = NEGATION_WORDS | SENTIMENT_CRITICAL

_STOPWORDS: set[str] | None = None
_STOPWORDS_FALLBACK_LOGGED = False
_LEMMATIZER_FALLBACK_LOGGED = False
_WORDNET_AVAILABLE: bool | None = None
_LEMMATIZER = WordNetLemmatizer() if WordNetLemmatizer is not None else None


def check_nltk_resources() -> dict[str, bool]:
    """Check which optional NLTK resources are locally available without downloading."""
    status = {"stopwords": False, "wordnet": False, "omw-1.4": False}
    if nltk is None:
        return status
    for pkg in status:
        try:
            nltk.data.find(f"corpora/{pkg}")
            status[pkg] = True
        except (LookupError, OSError):
            try:
                nltk.data.find(f"corpora/{pkg}.zip")
                status[pkg] = True
            except (LookupError, OSError):
                status[pkg] = False
    return status


def setup_nltk_resources(quiet: bool = True) -> dict[str, bool]:
    """Explicit setup helper to download NLTK corpora (for CLI/build-time use only).

    Never invoked automatically at import time or during API requests.
    """
    if nltk is None:
        logger.warning("NLTK is not installed; skipping resource download.")
        return {"stopwords": False, "wordnet": False, "omw-1.4": False}
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=quiet)
        except Exception as exc:  # pragma: no cover - network failure safe
            logger.warning("Could not download NLTK package '%s': %s", pkg, exc)
    reset_nltk_caches()
    return check_nltk_resources()


def reset_nltk_caches() -> None:
    """Reset cached stopwords, WordNet status, and warning flags (useful in tests)."""
    global _STOPWORDS, _STOPWORDS_FALLBACK_LOGGED, _LEMMATIZER_FALLBACK_LOGGED, _WORDNET_AVAILABLE
    _STOPWORDS = None
    _STOPWORDS_FALLBACK_LOGGED = False
    _LEMMATIZER_FALLBACK_LOGGED = False
    _WORDNET_AVAILABLE = None


def is_negation_or_critical(token: str) -> bool:
    """Return True if a token is a negation, negative contraction, or sentiment modifier."""
    lower = token.lower()
    return lower in _PROTECTED_WORDS or lower.endswith("n't")


def get_stopwords() -> set[str]:
    """English stopwords minus negation, contraction, and sentiment-critical words.

    Gracefully falls back to the built-in English stopword list if NLTK's
    `stopwords` corpus is missing or unreadable.
    """
    global _STOPWORDS, _STOPWORDS_FALLBACK_LOGGED
    if _STOPWORDS is None:
        base: set[str]
        try:
            if nltk_stopwords is None:
                raise LookupError("nltk.corpus.stopwords unavailable")
            base = set(nltk_stopwords.words("english"))
        except (LookupError, OSError, AttributeError):
            if not _STOPWORDS_FALLBACK_LOGGED:
                logger.warning(
                    "NLTK 'stopwords' corpus not found locally; using built-in fallback stopword list."
                )
                _STOPWORDS_FALLBACK_LOGGED = True
            base = set(_FALLBACK_ENGLISH_STOPWORDS)
        _STOPWORDS = {
            w for w in base
            if w not in _PROTECTED_WORDS and not w.endswith("n't")
        }
    return _STOPWORDS


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_APOSTROPHE_VARIANTS_RE = re.compile(r"[\u2018\u2019\u201b\u02bc\u0060\u00b4]")
_NON_ALPHANUM_RE = re.compile(r"[^a-z\s']")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, normalize apostrophes, strip noise, and collapse whitespace.

    Keeps ASCII apostrophes so contractions like "don't", "isn't", "can't",
    and "won't" survive tokenization intact.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _APOSTROPHE_VARIANTS_RE.sub("'", text)
    text = text.lower()
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _NON_ALPHANUM_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_review_for_dedup(text: Any) -> str:
    """Canonical normalization used for duplicate detection across both
    dataset inspection and dataframe cleaning.
    """
    if text is None or not isinstance(text, str):
        return ""
    stripped = text.strip()
    if not stripped:
        return ""
    normalized = normalize_text(stripped)
    return normalized if normalized else _WHITESPACE_RE.sub(" ", stripped.lower())


def tokenize(text: str) -> list[str]:
    """Split normalized text on whitespace and drop pure apostrophe tokens."""
    if not isinstance(text, str) or not text.strip():
        return []
    tokens: list[str] = []
    for raw in normalize_text(text).split():
        cleaned = raw.strip("'")
        if cleaned:
            tokens.append(cleaned)
    return tokens


def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    """Remove stopwords while strictly preserving negations and contractions."""
    stop = get_stopwords()
    result: list[str] = []
    for t in tokens:
        if is_negation_or_critical(t):
            result.append(t)
        elif t not in stop and len(t) > 1:
            result.append(t)
    return result


_FALLBACK_INVARIANT_WORDS: frozenset[str] = frozenset({
    "as", "is", "was", "has", "his", "this", "yes", "us", "bus", "plus",
    "less", "ness", "class", "glass", "pass", "mass", "boss", "loss", "cross",
    "status", "analysis", "basis", "crisis", "series", "species", "news",
    "always", "sometimes", "besides", "towards", "goods", "clothes", "means",
})


def _fallback_lemmatize_token(token: str) -> str:
    """Safe, deterministic fallback lemmatizer when WordNet is not installed."""
    if len(token) <= 3 or "'" in token or is_negation_or_critical(token):
        return token
    if token in _FALLBACK_INVARIANT_WORDS or token.endswith(("ss", "us", "is")):
        return token
    if token.endswith("ies") and len(token) > 4 and token[-4] not in "aeiou":
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4 and token[:-2].endswith(("sh", "ch", "x", "z", "s")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is", "ies")):
        return token[:-1]
    return token


def lemmatize(tokens: Iterable[str]) -> list[str]:
    """Lemmatize tokens with WordNet, falling back safely if WordNet is missing."""
    global _LEMMATIZER_FALLBACK_LOGGED, _WORDNET_AVAILABLE
    token_list = list(tokens)
    if not token_list:
        return []

    output: list[str] = []
    for t in token_list:
        if "'" in t or is_negation_or_critical(t):
            output.append(t)
            continue
        if _LEMMATIZER is not None and _WORDNET_AVAILABLE is not False:
            try:
                output.append(_LEMMATIZER.lemmatize(t))
                _WORDNET_AVAILABLE = True
                continue
            except (LookupError, OSError, AttributeError):
                _WORDNET_AVAILABLE = False
                if not _LEMMATIZER_FALLBACK_LOGGED:
                    logger.warning(
                        "NLTK 'wordnet'/'omw-1.4' corpus not found locally; using safe fallback lemmatizer."
                    )
                    _LEMMATIZER_FALLBACK_LOGGED = True
        output.append(_fallback_lemmatize_token(t))
    return output


def preprocess_text(text: str, *, remove_stops: bool = True, lemmatize_tokens: bool = True) -> str:
    """Full pipeline for one review. Used identically by train.py and predict.py."""
    if not isinstance(text, str) or not text.strip():
        return ""
    tokens = tokenize(text)
    if remove_stops:
        tokens = remove_stopwords(tokens)
    if lemmatize_tokens:
        tokens = lemmatize(tokens)
    return " ".join(tokens)


def clean_dataframe(
    df,
    text_column: str,
    label_column: str | None = None,
    *,
    apply_text_pipeline: bool = True,
    drop_conflicting_duplicates: bool = False,
    config=None,
):
    """Clean a raw dataframe for training: drop unusable rows and duplicates.

    - Treats None, NaN, empty strings (""), and whitespace-only strings ("   ")
      as missing for both review text and labels.
    - Normalizes review text consistently (`normalize_review_for_dedup`) when
      detecting duplicate reviews (so case/whitespace variants like
      "Great Product" and "great product" are recognized as duplicates).
    - Does NOT silently drop reviews that share normalized text but have
      different sentiment labels; instead reports the conflict in
      `stats["conflicting_label_duplicates"]` and `stats["conflicting_label_rows"]`
      (and only drops them if `drop_conflicting_duplicates=True`).

    Returns (clean_df, stats).
    """
    import pandas as pd

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in dataframe columns: {list(df.columns)}")
    if label_column is not None and label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataframe columns: {list(df.columns)}")

    stats: dict[str, Any] = {}
    original = len(df)
    df = df.copy()

    # 1. Drop missing, empty, or whitespace-only review text
    raw_text = df[text_column]
    stripped_text = raw_text.astype("string").str.strip()
    valid_text_mask = raw_text.notna() & stripped_text.notna() & (stripped_text.str.len() > 0)
    df = df[valid_text_mask].copy()
    df[text_column] = stripped_text[valid_text_mask]
    stats["dropped_missing_or_empty_text"] = int(original - len(df))

    # 2. Drop missing, empty, or whitespace-only labels (when label_column provided)
    if label_column:
        before = len(df)
        raw_label = df[label_column]
        stripped_label = raw_label.astype("string").str.strip().str.lower()
        valid_label_mask = raw_label.notna() & stripped_label.notna() & (stripped_label.str.len() > 0)
        df = df[valid_label_mask].copy()
        df[label_column] = stripped_label[valid_label_mask]
        stats["dropped_missing_labels"] = int(before - len(df))

    # 3. Canonical duplicate detection & conflicting sentiment label handling
    df["_dedup_key"] = df[text_column].map(normalize_review_for_dedup)
    before_dedup_empty = len(df)
    df = df[df["_dedup_key"].str.len() > 0].copy()
    stats["dropped_missing_or_empty_text"] += int(before_dedup_empty - len(df))

    if label_column:
        # First detect duplicate reviews that have conflicting sentiment labels
        if len(df) > 0:
            label_nunique = df.groupby("_dedup_key")[label_column].nunique(dropna=True)
            conflicting_keys = set(label_nunique[label_nunique > 1].index)
        else:
            conflicting_keys = set()

        stats["conflicting_label_duplicates"] = int(len(conflicting_keys))

        # Remove exact duplicates (same normalized review text AND same sentiment label)
        before_exact_dedup = len(df)
        df = df.drop_duplicates(subset=["_dedup_key", label_column], keep="first").copy()
        stats["dropped_duplicate_reviews"] = int(before_exact_dedup - len(df))

        conflicting_mask = df["_dedup_key"].isin(conflicting_keys)
        stats["conflicting_label_rows"] = int(conflicting_mask.sum())

        if conflicting_keys:
            logger.warning(
                "Detected %d normalized review text(s) (%d row(s)) with conflicting sentiment labels.",
                len(conflicting_keys),
                stats["conflicting_label_rows"],
            )
            if drop_conflicting_duplicates:
                before_conflict_drop = len(df)
                df = df[~conflicting_mask].copy()
                stats["dropped_conflicting_label_rows"] = int(before_conflict_drop - len(df))
            else:
                stats["dropped_conflicting_label_rows"] = 0
        else:
            stats["dropped_conflicting_label_rows"] = 0
    else:
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["_dedup_key"], keep="first").copy()
        stats["dropped_duplicate_reviews"] = int(before_dedup - len(df))
        stats["conflicting_label_duplicates"] = 0
        stats["conflicting_label_rows"] = 0
        stats["dropped_conflicting_label_rows"] = 0

    df = df.drop(columns=["_dedup_key"])
    stats["remaining_rows"] = int(len(df))

    if apply_text_pipeline:
        df = df.assign(**{
            f"{text_column}_clean": df[text_column].map(preprocess_text)
        })
        before = len(df)
        df = df[df[f"{text_column}_clean"].str.len() > 0].copy()
        stats["dropped_empty_after_preprocessing"] = int(before - len(df))
        stats["remaining_rows"] = int(len(df))

    return df, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess utilities and optional NLTK setup.")
    parser.add_argument(
        "--setup-nltk",
        action="store_true",
        help="Download optional NLTK corpora (stopwords, wordnet, omw-1.4).",
    )
    args = parser.parse_args()
    if args.setup_nltk:
        result = setup_nltk_resources(quiet=False)
        print("NLTK resource status:", result)
    else:
        print("NLTK resource status:", check_nltk_resources())
