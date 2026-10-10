"""Deterministic lexicon extraction with clause-level context. No trained NER claims."""
import re

TERMS = {
    "FINDING": ["tension pneumothorax", "pleural effusion", "pulmonary edema", "pneumothorax", "consolidation", "atelectasis", "cardiomegaly", "opacity", "opacities", "emphysema", "fracture", "pneumonia", "hemothorax"],
    "ANATOMY": ["left lung", "right lung", "left lower lobe", "right lower lobe", "lungs", "lung", "pleura", "heart", "mediastinum", "aorta", "rib", "spine"],
    "LATERALITY": ["left", "right", "bilateral"],
    "SEVERITY": ["large", "small", "moderate", "mild", "severe", "massive"],
    "TEMPORAL": ["acute", "new", "worsening", "stable", "chronic", "resolved"]
}
PATTERNS = {label: re.compile(r"\b(?:" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b", re.I) for label, terms in TERMS.items()}

def extract_entities(text: str) -> list[dict]:
    entities = []
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            # OCR inserts visual line breaks inside a sentence. Preserve cues
            # across those wraps; punctuation and section headings reset scope.
            left = max(text.rfind(".", 0, match.start()), text.rfind(";", 0, match.start())) + 1
            prefix = text[left:match.start()].lower()
            # A contrast or new section resets the preceding cue.
            prefix = re.split(r"\b(?:but|however|findings|impression)\b:?", prefix)[-1]
            status = "affirmed"
            if re.search(r"\b(no|without|absent|negative for|no evidence of)\b", prefix):
                status = "negated"
            elif re.search(r"\b(possible|may|cannot exclude|suspected|questionable)\b", prefix):
                status = "uncertain"
            elif re.search(r"\b(history of|previous|resolved|prior)\b", prefix):
                status = "historical"
            entities.append({"text": match.group(), "label": label, "start": match.start(), "end": match.end(), "assertion": status})
    return sorted(entities, key=lambda entity: (entity["start"], entity["end"], entity["label"]))
