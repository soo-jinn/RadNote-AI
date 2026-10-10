"""60 original synthetic report-page test images. No Open-I text is reused."""
import argparse
import json
import random
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

CASES = [
    ("Routine", "The lungs are clear. Heart size is normal. No pleural effusion or pneumothorax.", "No acute cardiopulmonary process."),
    ("Routine", "Heart size is within normal limits. The lungs are well expanded.", "No acute disease."),
    ("Routine", "Mild chronic cardiomegaly is unchanged. There is no focal consolidation.", "Stable chronic findings. No acute cardiopulmonary process."),
    ("Routine", "The mediastinum is normal. No focal opacity is present.", "Normal chest report."),
    ("Urgent", "A new focal consolidation is present in the right lower lobe.", "New focal consolidation in the right lower lobe."),
    ("Urgent", "Increasing bilateral opacities are present with small pleural effusions.", "Acute pulmonary edema."),
    ("Urgent", "An increasing left pleural effusion is present.", "Increasing pleural effusion on the left."),
    ("Urgent", "A right pneumothorax is present without tension features.", "New right pneumothorax."),
    ("Critical", "Large right pneumothorax with mediastinal shift is present.", "Tension pneumothorax. Emergent review is requested."),
    ("Critical", "A large hemothorax is present in the left pleural space.", "Large hemothorax. Immediate review is requested."),
    ("Critical", "A large left pneumothorax is present.", "Large pneumothorax. Critical finding for immediate review."),
    ("Critical", "The right lung is compressed with shift of the mediastinum.", "Tension pneumothorax requires emergent review.")
]

def get_font(path, size):
    choices = [path] if path else ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
    for candidate in choices:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)

def generate(output: Path, font_path=None):
    images = output/"images"; images.mkdir(parents=True, exist_ok=True)
    rows = []; rng = random.Random(179)
    for class_index in range(3):
        for number in range(20):
            label, findings, impression = CASES[class_index*4 + number%4]
            variant = number//4
            case_id = f"case_{class_index*20+number+1:03d}"
            text = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
            image = Image.new("RGB", (1000, 700), "white"); draw = ImageDraw.Draw(image)
            font = get_font(font_path, [25,26,24,27,25][variant])
            y = 45
            for paragraph in text.split("\n"):
                for line in textwrap.wrap(paragraph, 65) or [""]:
                    draw.text((45, y), line, font=font, fill=(20,20,20)); y += 43
            angle = [0,1,-1,2,-2][variant]
            image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor="white")
            if variant == 3:
                image = image.filter(ImageFilter.GaussianBlur(.45))
            if variant == 4:
                pixels = image.load()
                for _ in range(1000):
                    x, y = rng.randrange(image.width), rng.randrange(image.height)
                    pixels[x,y] = (rng.randrange(160,245),)*3
            image.save(images/f"{case_id}.png")
            rows.append({"id":case_id, "image":f"images/{case_id}.png", "findings":findings, "impression":impression, "reference_text":text, "scenario_label":label, "variant":variant, "skew_degrees":angle, "font_size":font.size, "font_file":Path(getattr(font,"path","") or "Pillow-default").name, "seed":179})
    (output/"manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} original synthetic images. Scenario labels are not clinical ground truth.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("samples/ocr"))
    parser.add_argument("--font", type=Path)
    args = parser.parse_args(); generate(args.output, args.font)
