"""Download official EasyOCR English weights once before offline demonstration."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radnote.ocr import ReportOCR
if __name__ == "__main__":
    folder = Path(__file__).resolve().parents[1]/"models/easyocr"
    folder.mkdir(parents=True, exist_ok=True)
    ReportOCR(folder, allow_download=True, threads=2)
    print(f"OCR cache ready: {folder}")
