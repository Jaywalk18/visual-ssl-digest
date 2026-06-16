from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path


REPORT_ROOT = Path(r"H:\Desktop\visual_ssl_paper_reports")
ARCHIVE_ROOT = REPORT_ROOT / "archive"
REPORT_ARCHIVE = ARCHIVE_ROOT / "reports"
SCRATCH_ARCHIVE = ARCHIVE_ROOT / "scratch"

DATED_REPORT_RE = re.compile(r"^20\d\d-\d\d-\d\d\.md$")
DATE_RE = re.compile(r"20\d\d-\d\d-\d\d")
KEEP_ROOT_FILES = {"latest.md", "package.json"}
KEEP_ROOT_DIRS = {"archive", "preferences"}


def archive_target(path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    if not target.exists():
        return target
    if path.is_file() and target.is_file() and path.read_bytes() == target.read_bytes():
        return target
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return target_dir / f"{path.stem}-{stamp}{path.suffix}"


def move_path(path: Path, target_dir: Path, dry_run: bool) -> tuple[str, Path, Path]:
    target = archive_target(path, target_dir)
    if dry_run:
        return ("would_move", path, target)
    if target.exists() and path.is_file() and target.is_file() and path.read_bytes() == target.read_bytes():
        path.unlink()
        return ("deduped", path, target)
    shutil.move(str(path), str(target))
    return ("moved", path, target)


def scratch_bucket(path: Path) -> Path:
    match = DATE_RE.search(path.name)
    if match:
        return SCRATCH_ARCHIVE / match.group(0)
    return SCRATCH_ARCHIVE / "legacy"


def cleanup(keep_reports: int, dry_run: bool) -> list[tuple[str, Path, Path]]:
    root = REPORT_ROOT.resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    actions: list[tuple[str, Path, Path]] = []

    reports = sorted(
        (p for p in REPORT_ROOT.glob("20??-??-??.md") if p.is_file()),
        key=lambda p: p.name,
        reverse=True,
    )
    keep = {p.name for p in reports[:keep_reports]}
    for report in reports[keep_reports:]:
        actions.append(move_path(report, REPORT_ARCHIVE, dry_run))

    for path in sorted(REPORT_ROOT.iterdir(), key=lambda p: p.name.lower()):
        resolved = path.resolve()
        if path.name in KEEP_ROOT_DIRS:
            continue
        if not str(resolved).lower().startswith(str(root).lower()):
            raise RuntimeError(f"Refusing to archive outside report root: {resolved}")
        if path.is_file():
            if path.name in KEEP_ROOT_FILES:
                continue
            if DATED_REPORT_RE.match(path.name) and path.name in keep:
                continue
            if DATED_REPORT_RE.match(path.name):
                continue
            actions.append(move_path(path, scratch_bucket(path), dry_run))
        elif path.is_dir():
            actions.append(move_path(path, SCRATCH_ARCHIVE / "dirs", dry_run))

    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive old Visual SSL reports and scratch files.")
    parser.add_argument("--keep-reports", type=int, default=7, help="Number of newest dated Markdown reports to keep in the root.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without changing files.")
    args = parser.parse_args()

    actions = cleanup(max(args.keep_reports, 1), args.dry_run)
    counts: dict[str, int] = {}
    for action, src, dst in actions:
        counts[action] = counts.get(action, 0) + 1
        print(f"{action}: {src} -> {dst}")
    print("summary:", ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "nothing to archive")


if __name__ == "__main__":
    main()
