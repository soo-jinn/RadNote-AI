"""Minimal report-upload client for Raspberry Pi and laptop simulation (stdlib only)."""
import argparse
import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

def upload(server: str, image: Path, timeout: int = 180) -> dict:
    mime = mimetypes.guess_type(image.name)[0]
    if mime not in {"image/png", "image/jpeg"}:
        raise ValueError("Choose a PNG or JPEG synthetic report page.")
    if image.stat().st_size > 8_000_000:
        raise ValueError("Image exceeds 8 MB.")
    boundary = uuid.uuid4().hex
    # Avoid embedding a user-supplied filename into multipart headers.
    head = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="report-page"\r\nContent-Type: {mime}\r\n\r\n'.encode()
    body = head + image.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(server.rstrip("/") + "/predict-image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Server returned {error.code}: {error.read().decode('utf-8', errors='replace')}") from error
    result["client_round_trip_ms"] = round((time.perf_counter()-start)*1000, 2)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="e.g. http://192.168.1.20:8000")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = upload(args.server, args.image)
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")

if __name__ == "__main__":
    main()
