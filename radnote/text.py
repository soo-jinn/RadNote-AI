"""Normalize report text and preserve Findings/Impression for classification."""
import re
import unicodedata

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    return re.sub(r"[ \t]+", " ", text).strip()

def parse_report(text: str) -> tuple[str, str, list[str]]:
    text = clean_text(text)
    # OCR can merge headings into a single line. Require a heading delimiter.
    headings = list(re.finditer(r"\b(FINDINGS|IMPRESSION)\s*[:\-]", text, re.I))
    sections = {"findings": "", "impression": ""}
    for index, match in enumerate(headings):
        stop = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        key = match.group(1).lower()
        sections[key] += " " + text[match.end():stop].strip()
    if not headings:
        return text, "", ["Report headings were not detected; all extracted text was used as Findings."]
    return clean_text(sections["findings"]), clean_text(sections["impression"]), []

def combine(findings: str, impression: str) -> str:
    return f"{clean_text(findings)}\n{clean_text(impression)}".strip()
