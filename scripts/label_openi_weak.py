"""Apply Group 4's provisional, text-first weak-label screen to a private CSV.

This is an educational rule-based label proposal, not clinical triage. Keep the
source-derived CSV private. Unclear or conflicting examples remain REVIEW.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path


CRITICAL = {
    "tension pneumothorax": r"\btension pneumothorax\b",
    "large pneumothorax": r"\blarge pneumothorax\b",
    "large hemothorax": r"\blarge hemothorax\b",
    "massive hemoptysis": r"\bmassive hemoptysis\b",
}
URGENT = {
    "pneumothorax": r"\bpneumothorax\b",
    "acute pulmonary edema": r"\bacute pulmonary edema\b|\bpulmonary edema\b.{0,35}\b(acute|worsening|increased)\b",
    "new consolidation": r"\b(new|acute|worsening)\b.{0,35}\b(consolidation|airspace opacity)\b",
    "acute pleural effusion": r"\b(new|acute|increasing|increased|worsening)\b.{0,35}\bpleural effusion\b",
    "pulmonary embolism": r"\bpulmonary embol(us|ism)\b",
    "acute fracture": r"\b(acute|new)\b.{0,30}\bfracture\b",
}
ROUTINE = (
    r"\bno acute (cardiopulmonary|pulmonary|chest|osseous) (process|finding|findings|disease)\b",
    r"\bno acute disease\b",
    r"\bnormal (chest|cardiopulmonary|study|examination)\b",
    r"\b(stable|unchanged) (chronic|mild|known)\b",
    r"\bchronic (change|changes|disease|scarring|atelectasis)\b",
)
NEGATION = re.compile(r"\b(no|not|without|negative for|free of|absence of|rule out|cannot exclude)\b", re.I)
HISTORICAL = re.compile(r"\b(history of|previously|resolved|prior|remote|old)\b", re.I)
UNCERTAIN = re.compile(r"\b(possible|possibly|probable|question of|may represent|cannot exclude|versus|\?)\b", re.I)


def _active(phrase: str, pattern: str, text: str) -> bool:
    for match in re.finditer(pattern, text, re.I):
        context = text[max(0, match.start() - 55) : match.start()]
        local = text[match.start() : min(len(text), match.end() + 30)]
        if NEGATION.search(context) or HISTORICAL.search(context) or UNCERTAIN.search(local):
            continue
        if re.search(r"\b(resolved|no longer|previously seen)\b", local, re.I):
            continue
        return True
    return False


def label_row(row: dict[str, str]) -> tuple[str, str]:
    impression = " ".join((row.get("impression") or "").split())
    if not impression:
        return "REVIEW", "missing Impression"

    # MeSH concepts corroborate report-text cues; they never create acuity alone.
    mesh = " ".join((row.get("mesh_major") or "", row.get("mesh_minor") or "")).casefold()
    candidates: list[tuple[str, str]] = []
    for reason, pattern in CRITICAL.items():
        if _active(reason, pattern, impression):
            candidates.append(("Critical", reason))
    for reason, pattern in URGENT.items():
        if _active(reason, pattern, impression):
            candidates.append(("Urgent", reason))
    if candidates:
        labels = {label for label, _ in candidates}
        # Critical takes precedence when a severe phrase also matches its
        # broader urgent concept (for example, "tension pneumothorax").
        if "Critical" in labels:
            label = "Critical"
        elif len(labels) > 1:
            return "REVIEW", "conflicting acuity cues: " + ", ".join(reason for _, reason in candidates)
        else:
            label = next(iter(labels))
        reasons = [reason for candidate_label, reason in candidates if candidate_label == label]
        concept_words = ("pneumothorax", "hemothorax", "hemoptysis", "pulmonary edema", "consolidation", "pleural effusion", "pulmonary embol")
        corroborated = any(word in mesh for word in concept_words if any(word in reason for reason in reasons))
        suffix = " (MeSH corroborated)" if corroborated else " (report text only; MeSH mismatch/absent)"
        return label, "; ".join(reasons) + suffix

    if UNCERTAIN.search(impression):
        return "REVIEW", "uncertain Impression wording"
    if any(re.search(pattern, impression, re.I) for pattern in ROUTINE):
        return "Routine", "explicit normal/stable/no-acute Impression"
    # A descriptor such as normal can support an explicit routine statement, but
    # descriptor-only inference is intentionally rejected.
    return "REVIEW", "no supported urgency phrase in Impression"


def _split_key(row: dict[str, str]) -> str:
    text = " ".join(f"{row.get('findings', '')} {row.get('impression', '')}".casefold().split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_splits(rows: list[dict[str, str]]) -> dict[str, str]:
    # Group duplicate normalized text before assigning a deterministic split.
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        label = row["weak_urgency_label"]
        if label != "REVIEW":
            groups[(label, _split_key(row))].append(row["report_id"])
    split_by_id: dict[str, str] = {}
    by_label: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for (label, digest), ids in groups.items():
        by_label[label].append((digest, ids))
    for label, label_groups in by_label.items():
        label_groups.sort(key=lambda group: hashlib.sha256(f"{label}:{group[0]}".encode()).hexdigest())
        total = sum(len(ids) for _, ids in label_groups)
        train_cut = round(total * 0.70)
        validation_cut = round(total * 0.85)
        seen = 0
        for _, ids in label_groups:
            midpoint = seen + len(ids) / 2
            split = "train" if midpoint <= train_cut else "validation" if midpoint <= validation_cut else "test"
            for report_id in ids:
                split_by_id[report_id] = split
            seen += len(ids)
    return split_by_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Input CSV is empty")
    for row in rows:
        label, reason = label_row(row)
        row["weak_urgency_label"] = label
        row["urgency_label"] = "" if label == "REVIEW" else label
        row["label_reason"] = reason
    splits = make_splits(rows)
    for row in rows:
        row["split"] = splits.get(row["report_id"], "review")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    model_columns = [name for name in columns if name != "weak_urgency_label"]
    for split_name in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split_name and row["urgency_label"]]
        split_path = args.output.with_name(f"{args.output.stem}_{split_name}{args.output.suffix}")
        with split_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=model_columns)
            writer.writeheader()
            writer.writerows({key: row[key] for key in model_columns} for row in split_rows)
    counts = Counter(row["weak_urgency_label"] for row in rows)
    partition_counts = Counter((row["split"], row["weak_urgency_label"]) for row in rows)
    print(f"Wrote provisional labels (private data): {args.output}")
    print("Label counts:", dict(sorted(counts.items())))
    print("Partition counts:", dict(sorted(partition_counts.items())))
    print("Model-ready partition files were written beside the weak-label CSV (REVIEW rows excluded).")
    print("Review all Critical examples and audit random samples before fitting a model.")


if __name__ == "__main__":
    main()
