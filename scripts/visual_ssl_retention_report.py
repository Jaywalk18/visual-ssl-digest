from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


SITE_ROOT = Path(r"H:\Desktop\visual_ssl_digest_site")
REPORT_ROOT = Path(r"H:\Desktop\visual_ssl_paper_reports")
DEFAULT_PREFS = REPORT_ROOT / "preferences" / "visual_ssl_preferences.json"


KEEP_PRIORITIES = {"P0", "P1", "高", "中高"}
LOW_PRIORITIES = {"P3", "扫读", "状态补录", "中"}


@dataclass
class SizeInfo:
    mineru_bytes: int
    html_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.mineru_bytes + self.html_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create retention candidates from Visual SSL paper preferences.")
    parser.add_argument("--site-root", type=Path, default=SITE_ROOT)
    parser.add_argument("--preferences", type=Path, default=DEFAULT_PREFS)
    parser.add_argument("--out-dir", type=Path, default=REPORT_ROOT / "archive" / "retention")
    parser.add_argument("--today", default=date.today().isoformat())
    parser.add_argument("--recent-days", type=int, default=30)
    parser.add_argument("--old-days", type=int, default=45)
    parser.add_argument("--dismiss-days", type=int, default=7)
    return parser.parse_args()


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def preference_path(path: Path) -> Path | None:
    candidates = [
        path,
        REPORT_ROOT / "visual_ssl_preferences.json",
        REPORT_ROOT / "preferences.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def as_set(value) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def size_info(site_root: Path, paper_id: str) -> SizeInfo:
    mineru = site_root / "assets" / "mineru" / paper_id
    html = site_root / "papers" / f"{paper_id}.html"
    return SizeInfo(mineru_bytes=dir_size(mineru), html_bytes=dir_size(html))


def parse_day(value: str) -> date | None:
    try:
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return None


def classify(paper: dict, liked: set[str], dismissed: set[str], read: set[str], today: date, args: argparse.Namespace) -> tuple[str, str]:
    pid = str(paper.get("id", ""))
    priority = str(paper.get("priority", ""))
    paper_day = parse_day(str(paper.get("date", "")))
    age = (today - paper_day).days if paper_day else 9999

    if pid in liked:
        return "keep", "已关注，保留全文和图像资产"
    if age <= args.recent_days:
        return "keep", f"{age} 天内新增，暂不清理"
    if priority in KEEP_PRIORITIES and age <= 180:
        return "keep", f"{priority} 且未超过 180 天，保留观察"
    if pid in dismissed and age >= args.dismiss_days:
        return "prune", "已标记略过且超过冷却期"
    if priority in LOW_PRIORITIES and age >= args.old_days:
        return "prune", f"{priority} 且超过 {args.old_days} 天，优先清理候选"
    if pid in read and priority not in KEEP_PRIORITIES and age >= args.old_days:
        return "slim", "已读但未关注，可转轻量保留"
    return "watch", "未命中强保留或强清理规则，继续观察"


def mb(value: int) -> str:
    return f"{value / 1024 / 1024:.2f} MB"


def write_outputs(rows: list[dict], prefs_path: Path | None, args: argparse.Namespace) -> tuple[Path, Path]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    today = str(args.today)
    json_path = args.out_dir / f"{today}_retention_candidates.json"
    md_path = args.out_dir / f"{today}_retention_candidates.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    bytes_by_action: dict[str, int] = {}
    for row in rows:
        counts[row["action"]] = counts.get(row["action"], 0) + 1
        bytes_by_action[row["action"]] = bytes_by_action.get(row["action"], 0) + row["bytes_total"]

    lines = [
        f"# Visual SSL 论文保留/清理候选 {today}",
        "",
        f"- 偏好文件：{prefs_path if prefs_path else '未找到，按空偏好评估'}",
        f"- 总论文数：{len(rows)}",
        f"- keep/watch/slim/prune：{counts.get('keep', 0)} / {counts.get('watch', 0)} / {counts.get('slim', 0)} / {counts.get('prune', 0)}",
        f"- prune 关联资产估算：{mb(bytes_by_action.get('prune', 0))}",
        "",
        "## 优先清理候选",
        "",
        "| 动作 | 论文 | 日期 | 优先级 | 估算体积 | 理由 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in [x for x in rows if x["action"] == "prune"][:80]:
        lines.append(
            f"| {row['action']} | {row['title']} (`{row['id']}`) | {row['date']} | {row['priority']} | {mb(row['bytes_total'])} | {row['reason']} |"
        )
    lines.extend([
        "",
        "## 轻量保留候选",
        "",
        "| 动作 | 论文 | 日期 | 优先级 | 估算体积 | 理由 |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in [x for x in rows if x["action"] == "slim"][:80]:
        lines.append(
            f"| {row['action']} | {row['title']} (`{row['id']}`) | {row['date']} | {row['priority']} | {mb(row['bytes_total'])} | {row['reason']} |"
        )
    lines.extend([
        "",
        "## 说明",
        "",
        "- `keep`：保留目录、详情页、MinerU 图像。",
        "- `watch`：继续观察，不清理。",
        "- `slim`：后续可只保留目录元数据和轻量摘要页，清理大图。",
        "- `prune`：后续可清理详情页和 MinerU 图像，但应先确认没有被关注。",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    args = parse_args()
    papers_path = args.site_root / "data" / "papers.json"
    papers = load_json(papers_path)
    if not isinstance(papers, list):
        raise ValueError(f"Expected list in {papers_path}")

    prefs_path = preference_path(args.preferences)
    prefs = load_json(prefs_path) if prefs_path else {}
    liked = as_set(prefs.get("likedIds") if isinstance(prefs, dict) else [])
    dismissed = as_set(prefs.get("dismissedIds") if isinstance(prefs, dict) else [])
    read = as_set(prefs.get("readIds") if isinstance(prefs, dict) else [])
    today = date.fromisoformat(str(args.today)[:10])

    rows = []
    for paper in papers:
        pid = str(paper.get("id", ""))
        action, reason = classify(paper, liked, dismissed, read, today, args)
        size = size_info(args.site_root, pid)
        rows.append({
            "id": pid,
            "title": str(paper.get("short") or paper.get("title") or pid),
            "full_title": str(paper.get("title") or ""),
            "date": str(paper.get("date") or ""),
            "priority": str(paper.get("priority") or ""),
            "category": str(paper.get("category") or ""),
            "action": action,
            "reason": reason,
            "liked": pid in liked,
            "dismissed": pid in dismissed,
            "read": pid in read,
            "bytes_mineru": size.mineru_bytes,
            "bytes_html": size.html_bytes,
            "bytes_total": size.total_bytes,
        })
    rows.sort(key=lambda row: ({"prune": 0, "slim": 1, "watch": 2, "keep": 3}[row["action"]], -row["bytes_total"], row["date"]))
    md_path, json_path = write_outputs(rows, prefs_path, args)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
