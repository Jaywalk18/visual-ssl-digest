from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


WORKSPACE = Path(r"C:\Users\Administrator\Documents\New project")
SITE_ROOT = Path(r"H:\Desktop\visual_ssl_digest_site")
REPORT_ROOT = Path(r"H:\Desktop\visual_ssl_paper_reports")
GENERATOR = WORKSPACE / "scripts" / "build_visual_ssl_site.py"
MINERU_BATCH = Path(r"C:\Users\Administrator\.claude\skills\paper-digest-site\scripts\submit_mineru.py")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def strip_md(text: str) -> str:
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)]\(([^)]+)\)", r"\1", text)
    return text.replace("**", "").replace("`", "").strip()


def report_path_for(date: str | None) -> Path:
    if date:
        path = REPORT_ROOT / f"{date}.md"
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    latest = REPORT_ROOT / "latest.md"
    if latest.exists():
        return latest
    reports = sorted(REPORT_ROOT.glob("20??-??-??.md"), key=lambda p: p.name)
    if not reports:
        raise FileNotFoundError(f"No reports found under {REPORT_ROOT}")
    return reports[-1]


def report_date(path: Path, text: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", text[:200]) or re.search(r"\d{4}-\d{2}-\d{2}", path.name)
    if not match:
        raise ValueError("Cannot infer report date")
    return match.group(0)


def safe_name(text: str, max_len: int = 72) -> str:
    clean = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", " ", strip_md(text))
    clean = re.sub(r"\s+", " ", clean).strip(" .")
    return (clean[:max_len].rstrip(" .") or "paper")


def parse_arxiv_papers(markdown: str) -> list[tuple[str, str, str]]:
    start = markdown.find("## 论文索引")
    if start < 0:
        return []
    rows: list[list[str]] = []
    in_table = False
    for line in markdown[start:].splitlines():
        if line.strip().startswith("|"):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", p or "") for p in parts):
                in_table = True
                continue
            rows.append(parts)
            in_table = True
        elif in_table:
            break
    papers: list[tuple[str, str, str]] = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        priority, paper_cell = row[0], row[1]
        if "过滤" in priority:
            continue
        link = re.search(r"\[([^\]]+)]\((https://arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?)\)", paper_cell)
        if not link:
            continue
        title, _, arxiv_id = link.groups()
        papers.append((arxiv_id, strip_md(title), strip_md(priority)))
    return papers


def download_pdfs(papers: list[tuple[str, str, str]], date: str) -> Path:
    staging = SITE_ROOT / "staging_pdfs" / date
    staging.mkdir(parents=True, exist_ok=True)
    for arxiv_id, title, priority in papers:
        if priority not in {"P1", "P2", "P3", "扫读"}:
            continue
        path = staging / f"{arxiv_id} - {safe_name(title)}.pdf"
        if path.exists() and path.stat().st_size > 100_000:
            print(f"skip existing PDF: {path.name}")
            continue
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"download {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "visual-ssl-digest/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            path.write_bytes(response.read())
    return staging


def cleanup_mineru_json(papers: list[tuple[str, str, str]]) -> None:
    for arxiv_id, _, _ in papers:
        root = SITE_ROOT / "assets" / "mineru" / arxiv_id
        if not root.exists():
            continue
        for pattern in ("layout.json", "*_content_list_v2.json"):
            for path in root.rglob(pattern):
                path.unlink(missing_ok=True)


def git_commit_and_push(date: str, push: bool) -> None:
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(SITE_ROOT), text=True)
    if not status.strip():
        print("no site changes to commit")
        return
    run(["git", "add", "-A"], cwd=SITE_ROOT)
    run(["git", "commit", "-m", f"daily: {date} web digest"], cwd=SITE_ROOT)
    if push:
        run(["git", "push"], cwd=SITE_ROOT)


def validate_site_quality() -> None:
    bad_patterns = [
        r"MINERU EXTRACTED",
        r"MinerU extracted figure without structured caption",
        r"MinerU full\.md is now part of the source bundle",
        r"当前页面是站点原型",
        r"下一步怎么接进日报",
        r"正式流程会把每篇论文",
        r"GitHub Pages MVP",
        r"当前 MVP",
        r"试运行",
        r"飞书文字版",
        r"原始日报",
        r"latest\.md",
        r"本地归档路径",
        r"Generated \d{4}",
        r"用于图文归档",
        r"每日 Markdown",
        r"自动生成",
    ]
    problems: list[str] = []
    paper_paths = list((SITE_ROOT / "papers").glob("*.html"))
    for path in paper_paths + list((SITE_ROOT / "issues").glob("*.html")) + [SITE_ROOT / "index.html"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in bad_patterns:
            if re.search(pattern, text, flags=re.I):
                problems.append(f"{path.relative_to(SITE_ROOT)} contains forbidden text: {pattern}")
        if path in paper_paths and 'class="full-fig"' in text and 'class="fig-zh"' not in text and "TC-JEPA" not in text:
            # Warn as a hard failure for generated paper pages: figure captions must be readable, not raw MinerU fallbacks.
            problems.append(f"{path.relative_to(SITE_ROOT)} has figures without bilingual caption spans")
    if problems:
        raise SystemExit("Site quality check failed:\n" + "\n".join(f"- {p}" for p in problems[:20]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the daily Visual SSL GitHub Pages digest.")
    parser.add_argument("--date", help="Report date, e.g. 2026-05-25. Defaults to latest.md/header date.")
    parser.add_argument("--no-mineru", action="store_true", help="Skip MinerU PDF extraction.")
    parser.add_argument("--no-push", action="store_true", help="Build and commit locally without pushing.")
    parser.add_argument("--dry-run", action="store_true", help="Parse the report and print selected papers without changing files.")
    args = parser.parse_args()

    report = report_path_for(args.date)
    markdown = report.read_text(encoding="utf-8")
    date = report_date(report, markdown)
    papers = parse_arxiv_papers(markdown)
    if not papers:
        raise SystemExit("No arXiv paper links found in the paper index.")
    if args.dry_run:
        for arxiv_id, title, priority in papers:
            print(f"{date}\t{priority}\t{arxiv_id}\t{title}")
        return

    if not args.no_mineru:
        staging = download_pdfs(papers, date)
        if MINERU_BATCH.exists():
            run([sys.executable, str(MINERU_BATCH), "--in", str(staging), "--out", str(SITE_ROOT / "assets" / "mineru")])
            cleanup_mineru_json(papers)
        else:
            print(f"MinerU batch script missing: {MINERU_BATCH}", file=sys.stderr)

    run([sys.executable, str(GENERATOR)])
    scripts_dir = SITE_ROOT / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    shutil.copy2(GENERATOR, scripts_dir / "build_visual_ssl_site.py")
    shutil.copy2(Path(__file__), scripts_dir / "publish_visual_ssl_site.py")
    validate_site_quality()
    git_commit_and_push(date, push=not args.no_push)
    print(f"published {date}: {len(papers)} indexed arXiv papers")


if __name__ == "__main__":
    main()
