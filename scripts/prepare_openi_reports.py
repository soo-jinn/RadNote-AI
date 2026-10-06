from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _sections(root: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for element in root.findall("./MedlineCitation/Article/Abstract/AbstractText"):
        label = (element.get("Label") or "").strip().upper()
        if label in {"FINDINGS", "IMPRESSION"}:
            result[label.lower()] = _clean(" ".join(element.itertext()))
    return result


def _mesh_terms(root: ET.Element, kind: str) -> list[str]:
    mesh = root.find("MeSH")
    if mesh is None:
        return []
    return sorted(
        {
            cleaned
            for element in mesh.iter(kind)
            if (cleaned := _clean(" ".join(element.itertext())))
        }
    )


def convert(input_dir: Path, output_csv: Path) -> tuple[int, int, int]:
    xml_files = sorted(input_dir.glob("*.xml"), key=lambda path: int(path.stem))
    if not xml_files:
        raise FileNotFoundError(f"No XML report files found in {input_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    findings_count = 0
    impression_count = 0
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "report_id",
                "findings",
                "impression",
                "mesh_major",
                "mesh_minor",
                "urgency_label",
            ),
        )
        writer.writeheader()
        for xml_path in xml_files:
            root = ET.parse(xml_path).getroot()
            report = _sections(root)
            findings = report.get("findings", "")
            impression = report.get("impression", "")
            findings_count += bool(findings)
            impression_count += bool(impression)
            writer.writerow(
                {
                    "report_id": xml_path.stem,
                    "findings": findings,
                    "impression": impression,
                    "mesh_major": " | ".join(_mesh_terms(root, "major")),
                    "mesh_minor": " | ".join(_mesh_terms(root, "minor")),
                    # Intentionally blank until Group 4 agrees on and documents the mapping.
                    "urgency_label": "",
                }
            )
    return len(xml_files), findings_count, impression_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    total, with_findings, with_impression = convert(args.input_dir, args.output_csv)
    print(f"Reports converted: {total}")
    print(f"Reports with Findings: {with_findings}")
    print(f"Reports with Impression: {with_impression}")
    print(f"Private working CSV: {args.output_csv}")
    print("Urgency labels are blank pending the group's documented weak-label policy.")


if __name__ == "__main__":
    main()
