# build_mood_dataset.py
# Purpose: Build an Azure Language Studio–importable dataset for
# Custom Multi-Label Text Classification using GoEmotions → your 8 moods.

from datasets import load_dataset
from pathlib import Path
import json, re
from tqdm import tqdm

# ---------- CONFIG (edit if needed) ----------
PROJECT_NAME   = "moodtunesv2-moodclass-v1"   # keep your chosen name
CONTAINER_NAME = "moodtunesv2-data"        # keep your container name
LANG           = "en-us"

OUT_ROOT     = Path("artifacts") / PROJECT_NAME
DOCS_LOCAL   = OUT_ROOT / "docs"
BLOB_PREFIX  = PROJECT_NAME  # we upload this whole folder into the container

MOODS = ["happy","calm","sad","melancholy","energetic","angry","anxious","romantic"]

# For a tiny dry run:
CAP_TRAIN = None
CAP_VAL   = None
CAP_TEST  = None
# ---------- END CONFIG -----------------------

# Base mapping GoEmotions → your moods
GO_TO_MOODS = {
    "admiration": ["happy"], "amusement": ["happy"], "approval": ["happy"],
    "gratitude": ["happy"], "joy": ["happy"], "optimism": ["happy"], "pride": ["happy"],
    "caring": ["romantic"],
    "relief": ["calm","happy"],  # low arousal + pleasant
    "love": ["romantic"], "desire": ["romantic"],
    "excitement": ["energetic"],
    "anger": ["angry"], "annoyance": ["angry"], "disapproval": ["angry"], "disgust": ["angry"],
    "fear": ["anxious"], "nervousness": ["anxious"], "embarrassment": ["anxious"],
    "sadness": ["sad"], "disappointment": ["sad"], "grief": ["melancholy"], "remorse": ["melancholy"],
    "curiosity": [], "realization": [], "confusion": [], "surprise": [], "neutral": []
}

# Gentle heuristics
nostal_rx = re.compile(r"\b(nostalgia|nostalgic|miss|remember( when)?|years ago|back in the day)\b", re.I)
calm_rx   = re.compile(r"\b(chill|calm|cozy|coffee|tea|rain(y)?|lofi|sunset|quiet|peaceful|serene)\b", re.I)
pos_surprise_rx = re.compile(r"\b(amazing|awesome|so happy|so excited)\b|!{2,}", re.I)

# Will be filled after we load the dataset (maps index -> label name)
LABEL_NAMES: list[str] | None = None

def to_label_names(emos) -> list[str]:
    """Convert a row's labels to names, whether they're ints or strings."""
    global LABEL_NAMES
    if not emos:
        return []
    # If already strings, return as-is
    if isinstance(emos[0], str):
        return list(emos)
    # If ints, map via LABEL_NAMES
    if isinstance(emos[0], int):
        if not LABEL_NAMES:
            raise RuntimeError("LABEL_NAMES not initialized; call after loading dataset.")
        return [LABEL_NAMES[i] for i in emos]
    # Fallback: cast to str
    return [str(x) for x in emos]

def map_to_moods(text: str, emos: list[str]) -> list[str]:
    """Turn GoEmotions labels (as names) + text into your moods (multi-label)."""
    moods = set()
    for e in emos:
        moods.update(GO_TO_MOODS.get(e, []))

    # surprise → happy only if clearly positive
    if "surprise" in emos and pos_surprise_rx.search(text):
        moods.add("happy")
    # nostalgia strengthens melancholy
    if (("sad" in moods) or ("melancholy" in moods) or ("sadness" in emos) or ("grief" in emos)) and nostal_rx.search(text):
        moods.add("melancholy")
    # calm keywords add calm
    if calm_rx.search(text):
        moods.add("calm")

    return sorted([m for m in moods if m in MOODS])

def write_split(split_ds, split_name: str, cap=None):
    split_dir = DOCS_LOCAL / split_name.capitalize()  # Train/Validation/Test
    split_dir.mkdir(parents=True, exist_ok=True)

    items = []
    iterator = split_ds if cap is None else split_ds.select(range(min(len(split_ds), cap)))

    for i, row in tqdm(enumerate(iterator), total=len(iterator), desc=f"{split_name}"):
        text = (row["text"] or "").strip()
        if not text:
            continue

        raw = row["labels"]
        emos = raw if isinstance(raw, list) else [raw]
        emos = to_label_names(emos)  # <-- convert ints to names if needed

        moods = map_to_moods(text, emos)
        if not moods:
            continue  # skip examples that don't map to any mood

        fname_local = f"doc_{i:06d}.txt"
        (split_dir / fname_local).write_text(text, encoding="utf-8")

        blob_location = f"{BLOB_PREFIX}/docs/{split_name.Capitalize() if hasattr(split_name,'Capitalize') else split_name.capitalize()}/{fname_local}"

        items.append({
            "location": blob_location,
            "language": LANG,
            "dataset": "Train" if split_name=="train" else ("Validation" if split_name=="validation" else "Test"),
            "classes": [{"category": m} for m in moods]
        })
    return items

def main():
    global LABEL_NAMES
    print("Loading GoEmotions…")
    ds = load_dataset("go_emotions", "simplified")  # returns train/validation/test

    # Extract label names from the features (works even if rows hold ints)
    feats = ds["train"].features["labels"]
    # In go_emotions, labels is a Sequence(ClassLabel)
    LABEL_NAMES = getattr(getattr(feats, "feature", None), "names", None) or getattr(feats, "names", None)
    if not LABEL_NAMES:
        raise RuntimeError("Could not obtain label names from dataset features.")

    print("Writing docs + labels.json …")
    train_items = write_split(ds["train"], "train", CAP_TRAIN)
    val_items   = write_split(ds["validation"], "validation", CAP_VAL)
    test_items  = write_split(ds["test"], "test", CAP_TEST)

    labels = {
        "projectFileVersion": "2022-05-01",
        "stringIndexType": "Utf16CodeUnit",
        "metadata": {
            "projectKind": "CustomMultiLabelClassification",
            "storageInputContainerName": CONTAINER_NAME,
            "projectName": PROJECT_NAME,
            "multilingual": False,
            "description": "MoodTunes: 8 emotion labels mapped from GoEmotions",
            "language": LANG
        },
        "assets": {
            "projectKind": "CustomMultiLabelClassification",
            "classes": [{"category": c} for c in MOODS],
            "documents": train_items + val_items + test_items
        }
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    # Stats
    from collections import Counter
    counts = Counter()
    for it in (train_items + val_items + test_items):
        for c in it["classes"]:
            counts[c["category"]] += 1

    print("\nWrote: ", OUT_ROOT / "labels.json")
    print("Local docs root: ", DOCS_LOCAL.resolve())
    print("Blob prefix used in 'location':", BLOB_PREFIX)
    print("Label counts:", dict(counts))

if __name__ == "__main__":
    main()
