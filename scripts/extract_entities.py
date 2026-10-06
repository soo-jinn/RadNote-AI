"""Small CPU rule-based spaCy entity prototype for report-text experiments.

This lexicon is a Sprint 1 scaffold, not a clinical NLP model. Review matches
and extend the patterns with Sonnelo before describing it as evaluated NER.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import spacy


PATTERNS = {
    "FINDING": (
        "pleural effusion", "pneumothorax", "pulmonary edema", "consolidation",
        "atelectasis", "cardiomegaly", "opacity", "emphysema", "fracture",
        "pulmonary embolism", "pneumonia",
    ),
    "ANATOMY": (
        "left lung", "right lung", "lungs", "lung", "pleura", "heart",
        "mediastinum", "aorta", "rib", "spine",
    ),
    "LATERALITY": ("left", "right", "bilateral", "left-sided", "right-sided"),
}


def build_pipeline():
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER", "overwrite_ents": False})
    ruler.add_patterns(
        {"label": label, "pattern": phrase}
        for label, phrases in PATTERNS.items()
        for phrase in phrases
    )
    return nlp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Report text; omit to read from stdin")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.text is None:
        import sys
        text = sys.stdin.read()
    else:
        text = args.text
    entities = [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in build_pipeline()(text).ents
    ]
    result = json.dumps({"entities": entities, "method": "spaCy EntityRuler lexicon prototype"}, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
