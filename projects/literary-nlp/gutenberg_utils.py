"""Shared corpus helpers for the literary NLP project."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

WORDS_PER_PAGE = 250

BOOKS: list[dict] = [
    {
        "book_id": "dracula",
        "title": "Dracula",
        "author": "Bram Stoker",
        "year": 1897,
        "gutenberg_id": 345,
        "arc": "Gothic horror that tightens around the hunt",
    },
    {
        "book_id": "wuthering_heights",
        "title": "Wuthering Heights",
        "author": "Emily Brontë",
        "year": 1847,
        "gutenberg_id": 768,
        "arc": "Stormy passion across two generations on the moors",
    },
    {
        "book_id": "time_machine",
        "title": "The Time Machine",
        "author": "H. G. Wells",
        "year": 1895,
        "gutenberg_id": 35,
        "arc": "A short scientific romance that darkens in the far future",
    },
    {
        "book_id": "pride_and_prejudice",
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "year": 1813,
        "gutenberg_id": 1342,
        "arc": "Social comedy of manners with a marriage-plot resolution",
    },
    {
        "book_id": "frankenstein",
        "title": "Frankenstein",
        "author": "Mary Shelley",
        "year": 1818,
        "gutenberg_id": 84,
        "arc": "Creator and creature spiral into isolation and revenge",
    },
    {
        "book_id": "alice",
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "year": 1865,
        "gutenberg_id": 11,
        "arc": "Whimsical, anarchic, and comparatively cheerful",
    },
    {
        "book_id": "christmas_carol",
        "title": "A Christmas Carol",
        "author": "Charles Dickens",
        "year": 1843,
        "gutenberg_id": 46,
        "arc": "A textbook dark-to-light redemption in five staves",
    },
    {
        "book_id": "dorian_gray",
        "title": "The Picture of Dorian Gray",
        "author": "Oscar Wilde",
        "year": 1891,
        "gutenberg_id": 174,
        "arc": "Aestheticism curdles into moral decay",
    },
]

BOOK_COLORS = {
    "dracula": "#4a0e0e",
    "wuthering_heights": "#6d4c41",
    "time_machine": "#1565c0",
    "pride_and_prejudice": "#ad1457",
    "frankenstein": "#2e7d32",
    "alice": "#6a1b9a",
    "christmas_carol": "#c62828",
    "dorian_gray": "#37474f",
}

THEMATIC_KEYWORDS: dict[str, list[str]] = {
    "dracula": ["blood", "night", "fear", "sleep", "death", "castle", "vampire"],
    "wuthering_heights": ["love", "hate", "moor", "storm", "revenge", "grave", "passion"],
    "time_machine": ["time", "machine", "morlock", "eloi", "future", "dark", "fear"],
    "pride_and_prejudice": ["love", "marriage", "pride", "dance", "letter", "fortune", "sister"],
    "frankenstein": ["creature", "science", "death", "nature", "revenge", "ice", "miserable"],
    "alice": ["queen", "rabbit", "hatter", "wonder", "king", "garden", "tea"],
    "christmas_carol": ["christmas", "ghost", "spirit", "money", "poor", "heart", "joy"],
    "dorian_gray": ["beauty", "youth", "portrait", "sin", "pleasure", "soul", "secret"],
}

CHARACTER_ALIASES: dict[str, dict[str, list[str]]] = {
    "dracula": {
        "Dracula": ["dracula", "count dracula"],
        "Jonathan": ["jonathan harker", "jonathan"],
        "Mina": ["mina harker", "mina murray", "mina"],
        "Lucy": ["lucy westenra", "lucy"],
        "Van Helsing": ["van helsing", "helsing"],
        "Seward": ["seward"],
        "Renfield": ["renfield"],
    },
    "wuthering_heights": {
        "Heathcliff": ["heathcliff"],
        "Catherine": ["catherine earnshaw", "catherine linton", "cathy", "catherine"],
        "Edgar": ["edgar linton", "edgar"],
        "Nelly": ["nelly dean", "ellen dean", "nelly"],
        "Hindley": ["hindley"],
        "Hareton": ["hareton"],
        "Isabella": ["isabella"],
    },
    "time_machine": {
        "Time Traveller": ["time traveller", "time traveler"],
        "Weena": ["weena"],
        "Eloi": ["eloi"],
        "Morlocks": ["morlocks", "morlock"],
    },
    "pride_and_prejudice": {
        "Elizabeth": ["elizabeth bennet", "miss elizabeth", "lizzy", "eliza", "elizabeth"],
        "Darcy": ["mr. darcy", "darcy"],
        "Jane": ["jane bennet", "jane"],
        "Bingley": ["bingley"],
        "Wickham": ["wickham"],
        "Lydia": ["lydia"],
        "Collins": ["mr. collins", "collins"],
        "Lady Catherine": ["lady catherine"],
    },
    "frankenstein": {
        "Victor": ["victor frankenstein", "victor"],
        "Creature": ["creature", "daemon", "wretch"],
        "Elizabeth": ["elizabeth"],
        "Clerval": ["clerval", "henry"],
        "Walton": ["walton"],
        "Justine": ["justine"],
    },
    "alice": {
        "Alice": ["alice"],
        "White Rabbit": ["white rabbit"],
        "Queen": ["queen of hearts", "the queen"],
        "Hatter": ["hatter"],
        "Duchess": ["duchess"],
        "Caterpillar": ["caterpillar"],
        "Cheshire Cat": ["cheshire"],
    },
    "christmas_carol": {
        "Scrooge": ["scrooge"],
        "Marley": ["marley"],
        "Christmas Past": ["christmas past"],
        "Christmas Present": ["christmas present"],
        "Yet to Come": ["yet to come", "christmas yet"],
        "Tiny Tim": ["tiny tim"],
        "Cratchit": ["cratchit", "bob"],
        "Fred": ["fred"],
    },
    "dorian_gray": {
        "Dorian": ["dorian gray", "dorian"],
        "Lord Henry": ["lord henry", "henry wotton", "harry"],
        "Basil": ["basil hallward", "basil"],
        "Sibyl": ["sibyl vane", "sibyl"],
        "James Vane": ["james vane"],
    },
}

EXTRA_STOPWORDS = {
    "gutenberg", "project", "ebook", "chapter", "illustration", "illustrated",
    "said", "mr", "mrs", "miss", "one", "would", "could", "upon", "shall",
    "may", "must", "much", "well", "know", "see", "come", "came", "went",
    "go", "get", "got", "like", "even", "still", "though", "yet", "also",
    "every", "never", "nothing", "something", "anything", "himself", "herself",
    "towards", "toward", "among", "without", "within", "another", "whose",
    "through", "before", "after", "again", "once", "here", "there", "very",
    "into", "from", "them", "they", "their", "this", "that", "these", "those",
    "with", "have", "been", "were", "was", "are", "had", "has", "not", "but",
    "and", "the", "for", "his", "her", "she", "him", "you", "your", "our",
    "stave", "letter", "us", "thing", "things", "might", "looked", "seemed",
    "made", "now", "old", "two", "first", "last", "back", "away", "long",
}

CHAPTER_HEADING_RE = re.compile(
    r"^(?:CHAPTER|Chapter|STAVE|Stave|LETTER|Letter)\s+"
    r"(?:[IVXLCDM]+\b|\d+\b).*$",
    re.MULTILINE,
)
ROMAN_DOT_RE = re.compile(r"^\s*([IVXLCDM]{1,8})\.\s*$", re.MULTILINE)
ROMAN_CHAPTER_RE = re.compile(r"^([IVXLCDM]{1,8})\s*$", re.MULTILINE)
EPILOGUE_RE = re.compile(r"^(Epilogue|EPILOGUE)\s*$", re.MULTILINE)
START_RE = re.compile(r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG", re.I)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG", re.I)


def project_dir() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "gutenberg_utils.py").exists() or (here / "data").exists():
        return here
    alt = Path("projects/literary-nlp").resolve()
    return alt if alt.exists() else here


def project_data_dir() -> Path:
    return project_dir() / "data"


def book_catalog() -> pd.DataFrame:
    return pd.DataFrame(BOOKS)


def title_of(book_id: str) -> str:
    for book in BOOKS:
        if book["book_id"] == book_id:
            return book["title"]
    return book_id


def strip_gutenberg_boilerplate(text: str) -> str:
    start = START_RE.search(text)
    if start:
        text = text[start.end():]
        nl = text.find("\n")
        if 0 <= nl < 200:
            text = text[nl + 1:]
    end = END_RE.search(text)
    if end:
        text = text[: end.start()]
    text = re.sub(r"\[Illustration:?[^\]]*\]", " ", text, flags=re.I)
    return text.strip()


def _split_by_matches(text: str, matches: list[re.Match]) -> list[tuple[str, str]]:
    if not matches:
        return [("Body", text.strip())]
    chapters: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble and len(preamble.split()) > 80:
        chapters.append(("Preamble", preamble))
    for i, match in enumerate(matches):
        start = match.start()
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = " ".join(match.group(0).split())
        body = text[start:stop]
        body = body[len(match.group(0)):].strip()
        first_line = body.split("\n", 1)[0].strip()
        if first_line and len(first_line.split()) <= 8 and not first_line.endswith((".", "!", "?")):
            heading = f"{heading} {first_line}".strip()
            body = body[len(first_line):].strip()
        if len(body.split()) < 20:
            continue
        chapters.append((heading[:80], body))
    return chapters or [("Body", text.strip())]


def split_chapters(text: str) -> list[tuple[str, str]]:
    matches = list(CHAPTER_HEADING_RE.finditer(text))
    if len(matches) >= 3:
        return _split_by_matches(text, matches)
    roman_dot = list(ROMAN_DOT_RE.finditer(text))
    if len(roman_dot) >= 5:
        extra = [m for m in EPILOGUE_RE.finditer(text) if m.start() > roman_dot[-1].start()]
        return _split_by_matches(text, sorted(roman_dot + extra, key=lambda m: m.start()))
    roman = list(ROMAN_CHAPTER_RE.finditer(text))
    if len(roman) >= 5:
        return _split_by_matches(text, roman)
    return [("Body", text.strip())]


def paginate_words(text: str, words_per_page: int = WORDS_PER_PAGE) -> list[str]:
    words = text.split()
    if not words:
        return []
    pages = [" ".join(words[i: i + words_per_page]) for i in range(0, len(words), words_per_page)]
    if len(pages) > 1 and len(pages[-1].split()) < 40:
        pages[-2] = pages[-2] + " " + pages[-1]
        pages.pop()
    return pages


def pages_from_text(book: dict, cleaned: str, words_per_page: int = WORDS_PER_PAGE) -> pd.DataFrame:
    rows: list[dict] = []
    page_idx = 0
    for chapter_idx, (heading, body) in enumerate(split_chapters(cleaned)):
        for page_text in paginate_words(body, words_per_page):
            rows.append(
                {
                    "book_id": book["book_id"],
                    "title": book["title"],
                    "author": book["author"],
                    "year": book["year"],
                    "chapter_idx": chapter_idx,
                    "chapter_title": heading,
                    "page_idx": page_idx,
                    "n_words": len(page_text.split()),
                    "text": page_text,
                }
            )
            page_idx += 1
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    n = df["page_idx"].max()
    df["progress"] = 0.0 if n == 0 else df["page_idx"] / n
    return df


def load_pages() -> pd.DataFrame:
    path = project_data_dir() / "pages.csv"
    df = pd.read_csv(path)
    if "progress" not in df.columns:
        df["progress"] = df.groupby("book_id")["page_idx"].transform(
            lambda s: 0.0 if s.max() == 0 else s / s.max()
        )
    return df


def extra_stopwords() -> set[str]:
    return set(EXTRA_STOPWORDS)


def english_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except LookupError:
        return set()


def all_stopwords() -> set[str]:
    return {w.lower() for w in english_stopwords() | extra_stopwords()}


def ensure_nltk_data() -> None:
    import nltk

    for pkg in ("vader_lexicon", "punkt", "punkt_tab", "stopwords"):
        nltk.download(pkg, quiet=True)


def rolling_by_book(df: pd.DataFrame, column: str, window: int = 8) -> pd.Series:
    return (
        df.sort_values(["book_id", "page_idx"])
        .groupby("book_id", sort=False)[column]
        .transform(lambda s: s.rolling(window, min_periods=1, center=True).mean())
    )


def mention_count(text: str, aliases: Iterable[str]) -> int:
    total = 0
    lowered = text.lower()
    for alias in aliases:
        pattern = r"\b" + re.escape(alias.lower()) + r"\b"
        total += len(re.findall(pattern, lowered))
    return total


def character_mentions(pages: pd.DataFrame, book_id: str) -> pd.DataFrame:
    aliases = CHARACTER_ALIASES.get(book_id, {})
    subset = pages.loc[pages["book_id"] == book_id].copy()
    for name, names in aliases.items():
        subset[name] = subset["text"].map(lambda t, a=names: mention_count(str(t), a))
    return subset


def keyword_counts(pages: pd.DataFrame, book_id: str) -> pd.DataFrame:
    words = THEMATIC_KEYWORDS.get(book_id, [])
    subset = pages.loc[pages["book_id"] == book_id].copy()
    lowered = subset["text"].str.lower()
    for word in words:
        pattern = r"\b" + re.escape(word) + r"\b"
        subset[word] = lowered.str.count(pattern)
    return subset


def add_vader_sentiment(pages: pd.DataFrame) -> pd.DataFrame:
    ensure_nltk_data()
    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    df = pages.copy()
    scores = [sia.polarity_scores(str(t)) for t in df["text"]]
    df["compound"] = [s["compound"] for s in scores]
    df["pos"] = [s["pos"] for s in scores]
    df["neg"] = [s["neg"] for s in scores]
    df["neu"] = [s["neu"] for s in scores]
    df["compound_smooth"] = rolling_by_book(df, "compound")
    return df


NRC_EMOTIONS = ("joy", "fear", "sadness", "anger", "trust", "anticipation")
_TOKEN_RE = re.compile(r"[A-Za-z']+")


@lru_cache(maxsize=1)
def _nrc_lexicon() -> dict:
    """Load NRC word→emotion map from the nrclex package (API-stable JSON)."""
    import json
    from importlib import resources

    path = resources.files("nrclex.data").joinpath("nrc_en.json")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def nrc_frequencies(text: str) -> dict[str, float]:
    lexicon = _nrc_lexicon()
    counts = {emo: 0 for emo in NRC_EMOTIONS}
    total = 0
    for token in _TOKEN_RE.findall(str(text).lower()):
        emotions = lexicon.get(token)
        if not emotions:
            continue
        for emo in emotions:
            if emo in counts:
                counts[emo] += 1
                total += 1
    if total == 0:
        return {emo: 0.0 for emo in NRC_EMOTIONS}
    return {emo: counts[emo] / total for emo in NRC_EMOTIONS}


def add_nrc_emotions(pages: pd.DataFrame) -> pd.DataFrame:
    df = pages.copy()
    records = [nrc_frequencies(t) for t in df["text"]]
    emo_df = pd.DataFrame(records)
    out = pd.concat([df.reset_index(drop=True), emo_df], axis=1)
    for emo in NRC_EMOTIONS:
        out[f"{emo}_smooth"] = rolling_by_book(out, emo)
    return out


def third_label(progress: pd.Series) -> pd.Series:
    return pd.cut(progress, bins=[-0.01, 1 / 3, 2 / 3, 1.01], labels=["beginning", "middle", "end"])
