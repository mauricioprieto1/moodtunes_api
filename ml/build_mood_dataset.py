# build_mood_dataset.py
# Purpose:
#   - Use the GoEmotions dataset as ground truth (we use its labels!)
#   - Map GoEmotions -> your 8 moods
#   - For Train/Validation ONLY, add negation-aware keyword/slang/emoji/emoticon boosts
#   - Keep Test "pure" (no boosts) so evaluation is apples-to-apples
#
# Output (Azure Language Studio import format):
#   artifacts/<PROJECT_NAME>/labels.json
#   artifacts/<PROJECT_NAME>/docs/{Train|Validation|Test}/doc_*.txt
#
# How to use:
#   1) Edit PROJECT_NAME/CONTAINER_NAME if needed
#   2) python build_mood_dataset.py
#   3) Upload ONLY labels.json (overwrite) to <container>/<PROJECT_NAME>/labels.json
#   4) In Language Studio -> Data labeling -> Reimport labels -> Train

from datasets import load_dataset
from pathlib import Path
from collections import Counter
import json
import re
from tqdm import tqdm

# ================= USER CONFIG =================
PROJECT_NAME   = "moodtunesv2-moodclass-v1"  # blob folder + project name shown in Studio
CONTAINER_NAME = "moodtunesv2-data"          # your blob container
LANG           = "en-us"

OUT_ROOT   = Path("artifacts/") / PROJECT_NAME
DOCS_LOCAL = OUT_ROOT / "docs"
BLOB_PREFIX = PROJECT_NAME

# Full run => None; put small integers to do quick caps
CAP_TRAIN = None
CAP_VAL   = None
CAP_TEST  = None

# Minimum text length to keep (tiny strings like "lol" alone are noisy)
MIN_CHARS = 2

MOODS = ["happy","calm","sad","melancholy","energetic","angry","anxious","romantic"]
# ===============================================


# ===== 1) GoEmotions -> Mood (you ARE using the dataset’s labels) =====
GO_TO_MOODS = {
    # pleasant/positive
    "admiration": ["happy"], "amusement": ["happy"], "approval": ["happy"],
    "gratitude": ["happy"], "joy": ["happy"], "optimism": ["happy"], "pride": ["happy"],
    "caring": ["romantic"],
    "relief": ["calm","happy"],
    "love": ["romantic"], "desire": ["romantic"],
    "excitement": ["energetic"],
    # negative
    "anger": ["angry"], "annoyance": ["angry"], "disapproval": ["angry"], "disgust": ["angry"],
    "fear": ["anxious"], "nervousness": ["anxious"], "embarrassment": ["anxious"],
    "sadness": ["sad"], "disappointment": ["sad"], "grief": ["melancholy"], "remorse": ["melancholy"],
    # ignored on their own
    "curiosity": [], "realization": [], "confusion": [], "surprise": [], "neutral": []
}


# ===== 2) Heuristic boosts (rich lexicons + slang + emojis + emoticons) =====
# Unicode emoji “charsets”: any char from the set indicates that mood
HAPPY_EMO   = "🙂😊😀😄😁😸😻✨🎉🥳👍👌🙌😍🥰❤️"
CALM_EMO    = "😌🧘🌅🌄🌙🍵☕🌧️🌿"
SAD_EMO     = "😢😭☹️🙁💔🥀"
ANGRY_EMO   = "😠😡🤬"
ANX_EMO     = "😰😱😥😓😩😫🫨"
ENERG_EMO   = "💪🔥⚡🏃🏋️‍♂️🏋️‍♀️🎧🎶"
ROM_EMO     = "💖💘💞💓💗❤️‍🔥💍💑"
MEL_EMO     = "🥺💔😔"

# Classic ASCII emoticons & slang
HAPPY_EMOTICONS = re.compile(r"(?:\:\-?\)|\:D|=D|=\)|xD|XD|\^\_\^|\:-?P|:p)", re.I)
SAD_EMOTICONS   = re.compile(r"(?:\:\-?\(|:\'\(|;\(|T_T|;-;|</3|:\[|D:|\:\()", re.I)
ANX_EMOTICONS   = re.compile(r"(?:\:\-?\/|\:-?S|:s)", re.I)     # “:/”, “:S”
ROM_EMOTICONS   = re.compile(r"(?:<3|❤)", re.I)

HAPPY_SLANG  = re.compile(r"\b(lol|lmao|lmfao|rofl|haha+|hehe+|yay+|woo+hoo+|yass+|yas+|yess+|pog(?:gers)?)\b", re.I)
ENERG_SLANG  = re.compile(r"\b(lets?\s?go+|goooo+|grind|turnt|turn\s?up|amped|hyped?|cranked|beast\s?mode)\b", re.I)

# Large keyword bundles per mood (hashtags & music subgenres included)
CALM_WORDS = re.compile(
    r"(?:calm|chill|chilled|chillin'?|relax(?:ed|ing)?|cozy|peaceful|serene|soothing|"
    r"ambient|instrumental|acoustic|piano|study|focus|concentrat(?:e|ing)|sleep|nap|"
    r"meditat(?:e|ion)|breath(?:e|ing)|yoga|mindful|zen|lofi|chillhop|rain(?:y)?|storm|thunder|"
    r"fireplace|campfire|ocean|waves|sunset|coffee|tea|café|cafe|quiet|soft|slow|#studywithme|#chill|#lofi|#cozy|#focus|#asmr)"
, re.I)

ENERG_WORDS = re.compile(
    r"(?:hype(?:d)?|pumped|let'?s\s?go+|go{2,}|workout|gym|run(?:ning)?|cardio|sprint|deadlift|pr|pb|"
    r"dance|party|mosh|club|rave|edm|house|techno|trance|dubstep|dnb|drum\s?&?\s?bass|"
    r"rock|metal|hardstyle|bass|drop|banger|fire|heat|turn\s?up|beast\s?mode|crank(?:ed)?|amp(?:ed)?)"
, re.I)

MELAN_WORDS = re.compile(
    r"(?:nostalgia|nostalgic|long(?:ing)?|wistful|yearn(?:ing)?|homesick|bittersweet|"
    r"miss(?:\s+you)?|remember(?:\s+when)?|throwback|back\s+in\s+the\s+day|old\s+times|"
    r"tears?|lonely|alone|empt(?:y|iness)|hollow|melancholy)"
, re.I)

ANX_WORDS = re.compile(
    r"(?:anxious|anxiety|panic|attack|overwhelm(?:ed|ing)?|stress(?:ed|ful)?|"
    r"worried|uneasy|on\s+edge|can'?t\s+sleep|insomnia|overthink(?:ing)?|jitters?|nervous)"
, re.I)

ROM_WORDS = re.compile(
    r"(?:love(?:\s+song)?|in\s+love|crush|date\s+night|romantic|valentine|bae|boo|"
    r"boyfriend|girlfriend|wedding|honeymoon|heart(?:s)?|cuddle|kiss|slow\s+dance|couple\s+goals|ships?)"
, re.I)

SAD_WORDS = re.compile(
    r"(?:sad|down|blue|depress(?:ed|ing)?|heartbrok(?:e|en)|grief|tears?|cry(?:ing)?|alone|lonely|hurts?)"
, re.I)

POS_SURPRISE = re.compile(r"\b(amazing|awesome|so\s+happy|so\s+excited)\b|!{2,}", re.I)


# ===== 3) Guardrails: negation awareness =====
NEG_WORDS = {"no","not","never","without","dont","don't","cant","can't","wont","won't","stop","avoid","hardly","rarely"}
WORD_RX = re.compile(r"[a-z']+", re.I)

def has_emoji(s: str, charset: str) -> bool:
    return any(ch in s for ch in charset)

def not_negated(match_span: tuple[int,int], text: str, window_tokens: int = 5) -> bool:
    """
    True if there is no negation token within ~window_tokens before the match.
    We scan ~80 chars left of match, tokenize, and inspect last N tokens.
    """
    start = match_span[0]
    left = text[max(0, start-80):start].lower()
    tokens = WORD_RX.findall(left)
    return not any(tok in NEG_WORDS for tok in tokens[-window_tokens:])

def guarded_search(rx: re.Pattern, text: str) -> bool:
    """
    Find the FIRST match of rx that is NOT negated; return True if found.
    (We do not need all matches; one good hit is enough to add the label.)
    """
    for m in rx.finditer(text):
        if not_negated(m.span(), text):
            return True
    return False


# ===== 4) Label name resolution (IDs -> names) =====
LABEL_NAMES: list[str] | None = None

def to_label_names(emos) -> list[str]:
    """Convert a row's labels to names, whether ints or strings."""
    global LABEL_NAMES
    if not emos:
        return []
    if isinstance(emos[0], str):
        return list(emos)
    if isinstance(emos[0], int):
        if not LABEL_NAMES:
            raise RuntimeError("LABEL_NAMES not initialized.")
        return [LABEL_NAMES[i] for i in emos]
    return [str(x) for x in emos]


# ===== 5) Core mapping fn =====
def map_to_moods(text: str, emos: list[str], *, use_keywords: bool) -> list[str]:
    """
    Build final multi-label mood set:
      - Always: base GoEmotions->Mood mapping (uses the dataset’s labels)
      - If use_keywords=True (Train/Validation): add negation-aware boosts
      - If use_keywords=False (Test): no boosts (keeps eval fair)
    """
    t = (text or "").strip()
    moods = set()

    # Base mapping from dataset labels
    for e in emos:
        moods.update(GO_TO_MOODS.get(e, []))

    # surprise + obviously positive phrasing => happy
    if "surprise" in emos and POS_SURPRISE.search(t):
        moods.add("happy")

    if not use_keywords:
        return sorted([m for m in moods if m in MOODS])

    # ---- Negation-aware boosts ----
    # calm
    if guarded_search(CALM_WORDS, t) or has_emoji(t, CALM_EMO):
        moods.add("calm")

    # energetic
    if guarded_search(ENERG_WORDS, t) or has_emoji(t, ENERG_EMO) or ENERG_SLANG.search(t):
        # ensure the slang isn't negated either (e.g., "not hyped")
        slang_m = ENERG_SLANG.search(t)
        if (slang_m is None) or not_negated(slang_m.span(), t):
            moods.add("energetic")

    # melancholy
    if guarded_search(MELAN_WORDS, t) or has_emoji(t, MEL_EMO):
        moods.add("melancholy")

    # anxious
    if guarded_search(ANX_WORDS, t) or has_emoji(t, ANX_EMO) or ANX_EMOTICONS.search(t):
        moods.add("anxious")

    # romantic
    if guarded_search(ROM_WORDS, t) or has_emoji(t, ROM_EMO) or ROM_EMOTICONS.search(t):
        moods.add("romantic")

    # sad
    if guarded_search(SAD_WORDS, t) or has_emoji(t, SAD_EMO) or SAD_EMOTICONS.search(t):
        moods.add("sad")

    # happy: slang/emoticons/emojis (conservative; lots of casual "lol")
    hsl = HAPPY_SLANG.search(t)
    if (hsl and not_negated(hsl.span(), t)) or HAPPY_EMOTICONS.search(t) or has_emoji(t, HAPPY_EMO):
        moods.add("happy")

    # tiny nudge: neutral + calm cues
    if "neutral" in emos and ("lofi" in t.lower() or guarded_search(CALM_WORDS, t)):
        moods.add("calm")

    return sorted([m for m in moods if m in MOODS])


# ===== 6) Write splits and labels.json =====
def write_split(split_ds, split_name: str, cap=None):
    """
    Writes docs/<Split>/doc_*.txt locally and returns 'documents' entries for labels.json.
    'dataset' must be exactly "Train" or "Test" for Language Studio.
    """
    split_dir = DOCS_LOCAL / split_name.capitalize()  # Train / Validation / Test
    split_dir.mkdir(parents=True, exist_ok=True)

    items = []
    iterator = split_ds if cap is None else split_ds.select(range(min(len(split_ds), cap)))

    for i, row in tqdm(enumerate(iterator), total=len(iterator), desc=f"{split_name}"):
        text = (row["text"] or "").strip()
        if len(text) < MIN_CHARS:
            continue

        raw = row["labels"]
        emos = raw if isinstance(raw, list) else [raw]
        emos = to_label_names(emos)

        use_kw = (split_name in ("train", "validation"))  # boosts only on Train/Validation
        moods = map_to_moods(text, emos, use_keywords=use_kw)
        if not moods:
            continue

        fname_local = f"doc_{i:06d}.txt"
        (split_dir / fname_local).write_text(text, encoding="utf-8")

        blob_location = f"{BLOB_PREFIX}/docs/{split_name.capitalize()}/{fname_local}"
        items.append({
            "location": blob_location,
            "language": LANG,
            "dataset": "Train" if split_name in ("train", "validation") else "Test",
            "classes": [{"category": m} for m in moods]
        })
    return items


def main():
    global LABEL_NAMES

    print("Loading GoEmotions …")
    ds = load_dataset("go_emotions", "simplified")

    # Resolve label names from dataset feature metadata
    feats = ds["train"].features["labels"]
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
            "description": "MoodTunes: 8 moods from GoEmotions + negation-aware heuristics",
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
    counts_all = Counter()
    counts_train = Counter()
    counts_test = Counter()
    for it in (train_items + val_items + test_items):
        for c in it["classes"]:
            counts_all[c["category"]] += 1
            if it["dataset"] == "Train":
                counts_train[c["category"]] += 1
            else:
                counts_test[c["category"]] += 1

    print("\nWrote:", OUT_ROOT / "labels.json")
    print("Local docs root:", DOCS_LOCAL.resolve())
    print("Blob prefix used in 'location':", BLOB_PREFIX)
    print("Label counts (ALL):", dict(counts_all))
    print("Label counts (TRAIN):", dict(counts_train))
    print("Label counts (TEST):", dict(counts_test))


if __name__ == "__main__":
    main()
