from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_OUTPUT = Path(r"H:\Desktop\visual_ssl_paper_reports\preferences\visual_ssl_preferences.json")
ALLOWED_ORIGINS = {
    "https://jaywalk18.github.io",
    "http://localhost",
    "http://127.0.0.1",
    "null",
}


def normalize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def sanitize(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "visual-ssl-preferences/v1",
        "receivedAt": datetime.now(timezone.utc).isoformat(),
        "exportedAt": str(payload.get("exportedAt") or ""),
        "source": str(payload.get("source") or ""),
        "readIds": normalize_list(payload.get("readIds")),
        "likedIds": normalize_list(payload.get("likedIds")),
        "dismissedIds": normalize_list(payload.get("dismissedIds")),
        "visitDays": normalize_list(payload.get("visitDays")),
    }


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        tmp_name = fh.name
    Path(tmp_name).replace(path)
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = backup_dir / f"{path.stem}-{stamp}{path.suffix}"
    backup.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    backups = sorted(backup_dir.glob(f"{path.stem}-*{path.suffix}"), key=lambda p: p.name, reverse=True)
    for old in backups[30:]:
        old.unlink(missing_ok=True)


def make_handler(output: Path):
    class PreferenceHandler(BaseHTTPRequestHandler):
        server_version = "VisualSSLPreferenceReceiver/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def _origin(self) -> str:
            origin = self.headers.get("Origin", "")
            if origin in ALLOWED_ORIGINS:
                return origin
            return "https://jaywalk18.github.io"

        def _headers(self, status: int = 200, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", self._origin())
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def do_OPTIONS(self) -> None:
            self._headers(204)

        def do_GET(self) -> None:
            if self.path.rstrip("/") not in {"", "/visual-ssl/preferences"}:
                self._headers(404)
                self.wfile.write(b'{"ok":false,"error":"not found"}')
                return
            payload = {
                "ok": True,
                "output": str(output),
                "exists": output.exists(),
                "updatedAt": datetime.fromtimestamp(output.stat().st_mtime, timezone.utc).isoformat() if output.exists() else None,
                "preferences": json.loads(output.read_text(encoding="utf-8")) if output.exists() else None,
            }
            self._headers(200)
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/visual-ssl/preferences":
                self._headers(404)
                self.wfile.write(b'{"ok":false,"error":"not found"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > 1_000_000:
                self._headers(400)
                self.wfile.write(b'{"ok":false,"error":"invalid content length"}')
                return
            try:
                raw = self.rfile.read(length).decode("utf-8")
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("payload must be an object")
                clean = sanitize(data)
                atomic_write_json(output, clean)
            except Exception as exc:
                self._headers(400)
                self.wfile.write(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False).encode("utf-8"))
                return
            self._headers(200)
            self.wfile.write(json.dumps({"ok": True, "output": str(output)}, ensure_ascii=False).encode("utf-8"))

    return PreferenceHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive Visual SSL paper preferences from the static GitHub Pages site.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.output))
    print(f"Visual SSL preference receiver listening on http://{args.host}:{args.port}/visual-ssl/preferences")
    print(f"Writing preferences to {args.output}")
    server.serve_forever()


if __name__ == "__main__":
    main()
