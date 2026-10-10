"""Local browser client/proxy. All inference stays on the Ubuntu server."""
import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BODY = 8_100_000

def make_handler(server):
    parsed = urllib.parse.urlsplit(server)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("--server must be an HTTP(S) origin without credentials, path, or query.")
    base = server.rstrip("/")
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, data, mime):
            self.send_response(status); self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def error_json(self, status, detail):
            self.respond(status, json.dumps({"detail": detail}).encode(), "application/json")
        def forward(self, body=None):
            path = urllib.parse.urlsplit(self.path).path
            request = urllib.request.Request(base + path, data=body, method="POST" if body is not None else "GET")
            if body is not None:
                request.add_header("Content-Type", self.headers.get("Content-Type", "application/octet-stream"))
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    self.respond(response.status, response.read(2_000_000), response.headers.get("Content-Type", "application/json"))
            except urllib.error.HTTPError as error:
                self.respond(error.code, error.read(2_000_000), "application/json")
            except (OSError, urllib.error.URLError):
                self.error_json(502, "Cannot reach the Ubuntu server. Check its address and VM networking.")
        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            files = {"/": ROOT/"web/index.html", "/static/style.css": ROOT/"web/style.css", "/static/app.js": ROOT/"web/app.js"}
            if path in files:
                mime = "text/html" if path == "/" else "text/css" if path.endswith(".css") else "text/javascript"
                self.respond(200, files[path].read_bytes(), mime)
            elif path in {"/health", "/ready"} or path in {"/samples/case_001.png", "/samples/case_021.png", "/samples/case_041.png"}:
                self.forward()
            else:
                self.error_json(404, "Not found")
        def do_POST(self):
            if self.path not in {"/predict", "/predict-image"}:
                self.error_json(404, "Not found"); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.error_json(400, "Invalid Content-Length"); return
            if not 0 < length <= MAX_BODY:
                self.error_json(413, "Request body is empty or exceeds the upload limit"); return
            self.connection.settimeout(30)
            try:
                body = self.rfile.read(length)
                if len(body) != length:
                    self.error_json(400, "Incomplete request body"); return
            except OSError:
                self.error_json(408, "Upload timed out"); return
            self.forward(body)
        def log_message(self, fmt, *args):
            pass
    return Handler

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.server))
    print(f"Open http://127.0.0.1:{args.port} in this device's browser. Inference server: {args.server}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

if __name__ == "__main__":
    main()
