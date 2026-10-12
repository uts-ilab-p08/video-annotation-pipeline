"""Serve the coverage explorer and rebuild its data from the bronze schema on request.

Serves the project root like `python -m http.server`, with two extra endpoints the page uses:

* GET /api/status                      -> {"live": true}
* GET /api/rebuild?video=<id>[&force=1] -> rebuilds that video's explorer file when the database
                                           has changed since it was built (always with force=1)

Static files are served with Cache-Control: no-store so a rebuilt JSON file is never stale in
the browser. Run with `make coverage-explorer`.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "reports"))
from build_coverage_explorer import build  # noqa: E402

PORT = 8000
_build_lock = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/status":
            return self._json({"live": True})
        if url.path == "/api/rebuild":
            query = parse_qs(url.query)
            video = query.get("video", [None])[0]
            if not video:
                return self._json({"ok": False, "error": "video is required"}, 400)
            force = query.get("force", ["0"])[0] == "1"
            try:
                with _build_lock:
                    entries = build(video, only_if_stale=not force)
            except SystemExit as error:  # build() reports a bad filter this way
                return self._json({"ok": False, "error": str(error)}, 404)
            except Exception as error:  # noqa: BLE001 - report any failure to the page
                return self._json({"ok": False, "error": f"{type(error).__name__}: {error}"}, 500)
            return self._json({"ok": True, "videos": entries})
        return super().do_GET()

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 - signature fixed by the base class
        if self.path.startswith("/api/"):
            super().log_message(format, *args)


def main() -> None:
    server = ThreadingHTTPServer(("", PORT), Handler)
    print(f">>> Open http://localhost:{PORT}/reports/coverage_explorer.html  (Ctrl-C stops the server)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
