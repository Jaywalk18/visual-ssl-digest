from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path


ROOT = Path(r"H:\Desktop\visual_ssl_digest_site")
REPORT_ROOT = Path(r"H:\Desktop\visual_ssl_paper_reports")
CSS_VERSION = "20260524"


PAPERS = [
    {
        "id": "2605.03245",
        "title": "Text-Conditional JEPA for Learning Semantically Rich Visual Representations",
        "short": "TC-JEPA",
        "category": "JEPA / VLM pretraining",
        "priority": "P1",
        "date": "2026-05-24",
        "venue": "ICML 2026 · CCF A",
        "url": "https://arxiv.org/abs/2605.03245",
        "thesis": "文本条件不是给 I-JEPA 加一个 caption 头，而是在降低 masked feature prediction 的多解性。",
        "takeaway": "视觉自监督和视觉语言预训练之间的边界被进一步打薄：它仍然预测 feature，不直接重建像素，也不完全依赖对比学习。",
        "method": "caption-conditioned masked feature prediction with sparse cross-attention text conditioner",
    },
    {
        "id": "2605.17537",
        "title": "Self-supervised Hierarchical Visual Reasoning with World Model",
        "short": "ResDreamer",
        "category": "World model / video SSL",
        "priority": "P1",
        "date": "2026-05-24",
        "venue": "ICML 2026 · CCF A",
        "url": "https://arxiv.org/abs/2605.17537",
        "thesis": "世界模型的价值不在 photorealistic rollouts，而在让残差层级逼出更抽象的动态 latent。",
        "takeaway": "它把视频 SSL 的目标从重建下一帧推向重建“低层没有解释完的部分”。",
        "method": "hierarchical residual world model for self-supervised visual foresight",
    },
    {
        "id": "2605.22819",
        "title": "Cambrian-P: Pose-Grounded Video Understanding",
        "short": "Cambrian-P",
        "category": "Video MLLM / geometry",
        "priority": "P1",
        "date": "2026-05-24",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2605.22819",
        "thesis": "相机位姿不只是下游几何任务标签，而可以成为视频理解模型的基础坐标系监督。",
        "takeaway": "用伪位姿把跨帧 token 固定到共享空间，是视频 VFM/MLLM 很值得跟的结构化监督。",
        "method": "learnable camera tokens + pose regression head for frame-level grounding",
    },
    {
        "id": "2605.22823",
        "title": "Which Way Did It Move? Directional Motion Blindness in Video-LLMs",
        "short": "DeltaDirect",
        "category": "Video diagnostics",
        "priority": "P2",
        "date": "2026-05-24",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2605.22823",
        "thesis": "Video-LLM 不是完全看不到运动，而是运动方向信号在 projector/LLM 接口处被错绑。",
        "takeaway": "这类诊断比单纯刷视频 QA 更有价值，因为它定位了视觉表征到语言读出的断点。",
        "method": "feature-delta motion vector objective at projector level",
    },
    {
        "id": "2605.22558",
        "title": "GeoWeaver: Grounding Visual Tokens with Geometric Evidence before Scene Reasoning",
        "short": "GeoWeaver",
        "category": "Geometry-grounded VLM",
        "priority": "P2",
        "date": "2026-05-24",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2605.22558",
        "thesis": "几何不应该只在 reasoning 阶段后融合，它应该先改造视觉 token 的坐标感。",
        "takeaway": "这篇不是 JEPA，而是 VLM 视觉 token grounding：冻结几何编码器的证据被分配给多层视觉 token。",
        "method": "multi-level geometry bank + residual grounding before LLM reasoning",
    },
    {
        "id": "2605.21059",
        "title": "Multimodal LLMs under Pairwise Modalities",
        "short": "Pairwise Modalities",
        "category": "Multimodal representation",
        "priority": "P1",
        "date": "2026-05-22",
        "venue": "arXiv · MinerU figure demo",
        "url": "https://arxiv.org/abs/2605.21059",
        "thesis": "多模态模型不一定需要完整 joint tuples；只要模态图连通，pairwise supervision 可以成为可扩展替代品。",
        "takeaway": "这是本 MVP 的 MinerU 图文样例页：Fig.1 解释动机，Fig.2 解释共享/私有 latent 生成过程。",
        "method": "self-modal reconstruction + pairwise contrastive latent alignment",
    },
]


def e(text: str | None) -> str:
    return html.escape(text or "", quote=True)


def strip_md(text: str) -> str:
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)]\(([^)]+)\)", r"\1", text)
    return text.replace("**", "").replace("`", "")


def flatten_caption(value) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def content_list_path(pid: str) -> Path | None:
    root = ROOT / "assets" / "mineru" / pid
    paths = sorted(root.glob("*content_list.json"))
    return paths[0] if paths else None


def figures_for(pid: str, depth: int = 1, max_count: int = 2) -> list[dict]:
    prefix = "../" * depth
    fallback = {
        "2605.21059": [
            {
                "src": f"{prefix}assets/mineru/2605.21059/images/fig1.jpg",
                "label": "Fig. 1 · Pairwise supervision is the scalable object",
                "caption": "Joint tuples are expensive; pairwise text-image, image-tactile, and text-3D data can form a connected modality graph.",
            },
            {
                "src": f"{prefix}assets/mineru/2605.21059/images/fig2.jpg",
                "label": "Fig. 2 · Shared and modality-specific latent factors",
                "caption": "The paper separates common causal factors from modality-private variables before alignment.",
            },
        ]
    }
    path = content_list_path(pid)
    if not path:
        return fallback.get(pid, [])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback.get(pid, [])

    selected = []
    keywords = [
        "figure 1", "figure 2", "overview", "architecture", "framework",
        "pipeline", "illustration", "comparison", "model", "deltadirect",
        "grounding", "conditioning",
    ]
    for item in data:
        if item.get("type") != "image" or not item.get("img_path"):
            continue
        img_rel = str(item["img_path"]).replace("\\", "/")
        img_path = ROOT / "assets" / "mineru" / pid / img_rel
        if not img_path.exists():
            continue
        cap = flatten_caption(item.get("image_caption"))
        blob = f"{cap} {img_rel}".lower()
        score = img_path.stat().st_size / 1000
        for idx, kw in enumerate(keywords):
            if kw in blob:
                score += 500 - idx * 20
        # Avoid tiny fragments when a real method figure exists.
        if img_path.stat().st_size < 6000:
            score -= 120
        selected.append({
            "score": score,
            "src": f"{prefix}assets/mineru/{pid}/{img_rel}",
            "label": cap.split(". ")[0][:90] if cap else "MinerU extracted figure",
            "caption": cap or "MinerU extracted figure without structured caption.",
        })
    selected.sort(key=lambda x: x["score"], reverse=True)
    return [{k: v for k, v in fig.items() if k != "score"} for fig in selected[:max_count]] or fallback.get(pid, [])


def md_table_rows(md: str, heading: str) -> list[list[str]]:
    marker = f"## {heading}"
    start = md.find(marker)
    if start < 0:
        return []
    lines = md[start:].splitlines()
    rows: list[list[str]] = []
    in_table = False
    for line in lines:
        if line.strip().startswith("|"):
            parts = [strip_md(p.strip()) for p in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", p or "") for p in parts):
                in_table = True
                continue
            rows.append(parts)
            in_table = True
        elif in_table:
            break
    return rows


def section_text(md: str, heading: str, limit: int = 900) -> str:
    marker = f"## {heading}"
    start = md.find(marker)
    if start < 0:
        return ""
    tail = md[start + len(marker):]
    next_h = tail.find("\n## ")
    if next_h >= 0:
        tail = tail[:next_h]
    clean = "\n".join(
        strip_md(line).strip()
        for line in tail.splitlines()
        if line.strip() and not line.strip().startswith("|") and not line.startswith("### ")
    )
    clean = re.sub(r"\n{2,}", "\n", clean).strip()
    return clean[:limit] + ("…" if len(clean) > limit else "")


def nav(depth: int = 0, active: str = "") -> str:
    prefix = "../" * depth
    items = [
        ("index.html", "头版"),
        ("issues/2026-05-24.html", "今日速递"),
        ("pages/catalog.html", "论文目录"),
        ("pages/timeline.html", "时间线"),
    ]
    return "\n".join(
        f'<a class="nav-link{" active" if active == href else ""}" href="{prefix}{href}">{label}</a>'
        for href, label in items
    )


def shell(title: str, body: str, depth: int = 0, active: str = "") -> str:
    prefix = "../" * depth
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{e(title)} · 通用视觉自监督研究报</title>
  <link rel="stylesheet" href="{prefix}assets/styles.css?v={CSS_VERSION}">
  <link rel="stylesheet" href="{prefix}assets/visual-ssl.css?v={CSS_VERSION}">
</head>
<body>
  <header class="masthead">
    <div class="masthead-kicker">Visual SSL Daily · GitHub Pages MVP · 2026-05-24</div>
    <a class="masthead-title" href="{prefix}index.html">通用视觉自监督研究报</a>
    <div class="masthead-meta">
      <span>图像表征 · VFM · JEPA · 视频预训练</span>
      <span>CCF A/B 会议优先</span>
      <span>MinerU 图文版试运行</span>
    </div>
  </header>
  <nav class="site-nav" aria-label="主导航">
    {nav(depth, active)}
  </nav>
  <main>{body}</main>
  <footer class="footer">
    <div>原始日报继续同步飞书；本站用于图文归档和长读。</div>
    <div class="muted">Generated 2026-05-24 · MinerU figures where available</div>
  </footer>
</body>
</html>"""


def paper_card(p: dict, depth: int = 0) -> str:
    prefix = "../" * depth
    figs = figures_for(p["id"], depth, 1)
    if figs:
        thumb = f"""<img src="{e(figs[0]['src'])}" alt="{e(p['short'])} figure"><span>{e(p['priority'])}</span>"""
    else:
        thumb = f"""<span>{e(p['priority'])}</span><strong>{e(p['short'])}</strong><em>{e(p['venue'])}</em>"""
    return f"""<article class="paper-card" data-category="{e(p['category'])}" data-title="{e(p['title'].lower())}">
  <a class="paper-thumb paper-thumb-text" href="{prefix}papers/{p['id']}.html">
    {thumb}
  </a>
  <div>
    <div class="paper-meta">{e(p['date'])} · {e(p['category'])} · {e(p['venue'])}</div>
    <h3><a href="{prefix}papers/{p['id']}.html">{e(p['title'])}</a></h3>
    <p>{e(p['takeaway'])}</p>
    <p class="card-tech"><b>方法：</b>{e(p['method'])}</p>
    <div class="paper-actions"><a href="{prefix}papers/{p['id']}.html">读图文页</a><a href="{e(p['url'])}">原文</a></div>
  </div>
</article>"""


def write_index(report_md: str) -> None:
    summary = md_table_rows(report_md, "快速摘要")
    summary_html = "".join(
        f"<li><b>{e(row[0])}</b><span>{e(row[1])}</span></li>"
        for row in summary[1:] if len(row) >= 2
    )
    hero = PAPERS[0]
    hero_figs = figures_for(hero["id"], 0, 1)
    hero_img = f'<figure class="hero-figure"><img src="{e(hero_figs[0]["src"])}" alt="{e(hero["short"])} figure"><figcaption>{e(hero_figs[0]["caption"][:180])}</figcaption></figure>' if hero_figs else ""
    cards = "\n".join(paper_card(p, 0) for p in PAPERS[:5])
    body = f"""<section class="hero-grid">
  <article class="lead-story">
    <div class="kicker">今日主线 · {e(hero['venue'])}</div>
    <h1>{e(hero['short'])}：JEPA 的不确定性开始交给语言处理</h1>
    <p class="dek">{e(hero['thesis'])}</p>
    {hero_img}
    <div class="paper-actions"><a href="issues/2026-05-24.html">阅读 5 月 24 日速递</a><a href="papers/{hero['id']}.html">打开主文页</a></div>
  </article>
  <aside class="issue-brief">
    <h2>快速摘要</h2>
    <ul>{summary_html}</ul>
  </aside>
</section>
<section class="section-block">
  <div class="section-heading"><h2>今日推荐论文</h2><a href="pages/catalog.html">全目录 →</a></div>
  <div class="paper-list">{cards}</div>
</section>"""
    (ROOT / "index.html").write_text(shell("头版", body, 0, "index.html"), encoding="utf-8")


def write_issue(report_md: str) -> None:
    route = section_text(report_md, "阅读路线", 1200)
    index_rows = md_table_rows(report_md, "论文索引")
    table = ""
    if index_rows:
        head, *body_rows = index_rows
        ths = "".join(f"<th>{e(h)}</th>" for h in head)
        trs = "".join(
            "<tr>" + "".join(f"<td>{e(c)}</td>" for c in row) + "</tr>"
            for row in body_rows
        )
        table = f'<table class="compact-table"><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>'
    cards = "\n".join(paper_card(p, 1) for p in PAPERS[:5])
    body = f"""<article class="paper-detail issue-detail">
  <div class="paper-main">
    <div class="kicker">Daily issue · 2026-05-24 · CCF A/B 会议优先</div>
    <h1 class="paper-headline">5 月 24 日：JEPA、世界模型与视频几何信号同时补强视觉表征</h1>
    <p class="dek">今天不是纯图像 SSL 单点爆发，而是文本条件 JEPA、层级世界模型、位姿 grounding 和运动方向诊断共同推高通用视觉表征的边界。</p>
    <div class="feature-body">
      <p class="lead dropcap">{e(route)}</p>
      <h2>论文索引</h2>
      {table}
      <h2>主线阅读</h2>
      <div class="paper-list issue-list">{cards}</div>
    </div>
  </div>
  <aside class="paper-side">
    <div class="side-box"><h4>今日重点</h4><p>TC-JEPA、ResDreamer、Cambrian-P、DeltaDirect。</p></div>
    <div class="side-box"><h4>会议口径</h4><p>只把 CCF A/B 放进主提醒；临近截止或 CCF C 仅记录。</p></div>
    <div class="side-box"><h4>飞书文字版</h4><p><a href="https://www.feishu.cn/file/T684bdqYloXgmVxCV4ocxCm1nMc">latest.md →</a></p></div>
  </aside>
</article>"""
    (ROOT / "issues" / "2026-05-24.html").write_text(shell("2026-05-24 速递", body, 1, "issues/2026-05-24.html"), encoding="utf-8")


def write_paper(p: dict) -> None:
    figs = ""
    paper_figs = figures_for(p["id"], 1, 2)
    if paper_figs:
        figs = "\n".join(
            f"""<figure class="full-fig"><img src="{e(fig['src'])}" alt="{e(fig['label'])}"><figcaption><b>{e(fig['label'])}</b>{e(fig['caption'])}</figcaption></figure>"""
            for fig in paper_figs
        )
    else:
        figs = f"""<aside class="marginalia"><b>图像状态</b> 这篇 MVP 先使用文字解读；接入每日 MinerU 批处理后，这里会自动替换为 pipeline / motivation 图。</aside>"""
    body = f"""<article class="paper-detail">
  <div class="paper-main">
    <div class="kicker">{e(p['category'])} · {e(p['priority'])} · {e(p['date'])}</div>
    <h1 class="paper-headline">{e(p['short'])}：{e(p['thesis'])}</h1>
    <p class="dek">{e(p['takeaway'])}</p>
    <p class="byline"><strong>{e(p['title'])}</strong>　{e(p['venue'])}　<a href="{e(p['url'])}">原文链接</a></p>
    <div class="paper-facts">
      <span><b>编号</b>{e(p['id'])}</span>
      <span><b>优先级</b>{e(p['priority'])}</span>
      <span><b>类别</b>{e(p['category'])}</span>
      <span><b>会议</b>{e(p['venue'])}</span>
      <span><b>方法</b>{e(p['method'])}</span>
      <span><b>来源</b>arXiv / OpenReview</span>
    </div>
    <div class="feature-body">
      <p class="lead dropcap">先说结论。{e(p['thesis'])} {e(p['takeaway'])}</p>
      {figs}
      <h2>它真正改变的是训练信号，而不是榜单数字</h2>
      <p>{e(p['method'])}。这类工作值得被放在通用视觉自监督日报里，不是因为它一定立刻刷新所有 benchmark，而是因为它改变了模型“从哪里拿监督”的方式。</p>
      <blockquote class="inline-quote">MinerU full.md is now part of the source bundle for this page; the next pass will use it for paper-native English quotes and tighter argument writing.</blockquote>
      <h2>读这篇时要盯住的风险</h2>
      <p>当前页面是站点原型，细节还没有逐段引用 MinerU 全文。正式流程会把每篇论文的 abstract、method、caption 和关键表格放进页面生成上下文，并保留至少两段原文引文。</p>
      <h2>下一步怎么接进日报</h2>
      <p>每日自动化先筛出 P1/P2，再对 P1 论文跑 MinerU；有 pipeline 或 motivation 图的页面进入 GitHub Pages，飞书只发摘要与入口。</p>
    </div>
  </div>
  <aside class="paper-side">
    <div class="side-box"><h4>关键判断</h4><dl><dt>相关性</dt><dd>{e(p['priority'])}</dd><dt>类别</dt><dd>{e(p['category'])}</dd><dt>会议</dt><dd>{e(p['venue'])}</dd></dl></div>
    <div class="side-box"><h4>原文</h4><p><a href="{e(p['url'])}">{e(p['url'])}</a></p></div>
  </aside>
</article>"""
    (ROOT / "papers" / f"{p['id']}.html").write_text(shell(p["short"], body, 1), encoding="utf-8")


def write_catalog() -> None:
    cards = "\n".join(paper_card(p, 1) for p in PAPERS)
    body = f"""<section class="page-header"><h1>论文目录</h1><p>当前 MVP 收录 {len(PAPERS)} 篇样例；正式版会读取跨天去重 JSON 和每日筛选结果。</p></section><div class="paper-list">{cards}</div>"""
    (ROOT / "pages" / "catalog.html").write_text(shell("论文目录", body, 1, "pages/catalog.html"), encoding="utf-8")


def write_timeline() -> None:
    grouped: dict[str, list[dict]] = {}
    for p in PAPERS:
        grouped.setdefault(p["date"], []).append(p)
    parts = []
    for date in sorted(grouped, reverse=True):
        parts.append(f'<section class="timeline-day"><h2>{e(date)}</h2>')
        for p in grouped[date]:
            parts.append(f'<article><span>{e(p["priority"])}</span><a href="../papers/{p["id"]}.html">{e(p["short"])}</a><em>{e(p["category"])}</em></article>')
        parts.append("</section>")
    body = '<section class="page-header"><h1>时间线</h1><p>新增、更新与状态补录会在这里按天归档。</p></section>' + "\n".join(parts)
    (ROOT / "pages" / "timeline.html").write_text(shell("时间线", body, 1, "pages/timeline.html"), encoding="utf-8")


def write_css() -> None:
    css = """
.hero-grid{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(320px,.55fr);gap:36px;align-items:stretch;margin:34px auto 42px;max-width:1500px;padding:0 24px}
.lead-story{border-top:5px solid #111;padding-top:22px}
.lead-story h1{font-size:clamp(42px,6vw,86px);line-height:.94;letter-spacing:0;margin:10px 0 18px;font-family:Georgia,'Times New Roman',serif}
.lead-story .dek{font-size:22px;line-height:1.45;max-width:880px;color:#30302d}
.issue-brief{border:1px solid #d7d1c6;background:#f7f3ea;padding:22px}
.issue-brief h2{margin:0 0 12px;font-size:22px}
.issue-brief ul{list-style:none;margin:0;padding:0;display:grid;gap:12px}
.issue-brief li{border-bottom:1px solid #ddd5c6;padding-bottom:10px}
.issue-brief li b{display:block;font-size:13px;color:#7d2f24;text-transform:uppercase}
.issue-brief li span{display:block;font-size:15px;line-height:1.5}
.section-block{max-width:1500px;margin:0 auto;padding:0 24px 44px}
.section-heading{display:flex;align-items:end;justify-content:space-between;border-bottom:2px solid #111;margin-bottom:18px}
.section-heading h2{margin:0 0 8px;font-size:28px}
.paper-list{display:grid;gap:18px}
.paper-thumb-text{display:flex;flex-direction:column;justify-content:center;min-height:172px;background:#171717;color:#f7f3ea;text-decoration:none;padding:18px;border-radius:2px}
.paper-thumb-text img{width:100%;height:172px;object-fit:cover;display:block;margin:-18px -18px 12px;width:calc(100% + 36px);max-width:none;border-bottom:1px solid rgba(255,255,255,.18)}
.paper-thumb-text span{font-size:13px;color:#f2c14e}
.paper-thumb-text strong{font-size:25px;line-height:1.05;margin:10px 0;font-family:Georgia,'Times New Roman',serif}
.paper-thumb-text em{font-size:12px;color:#cfc7b7;font-style:normal}
.hero-figure{margin:24px 0;border-top:1px solid #d8d0c3;border-bottom:1px solid #d8d0c3;padding:16px 0}
.hero-figure img{width:100%;max-height:430px;object-fit:contain;background:#fff}
.hero-figure figcaption{font-size:13px;line-height:1.45;color:#5f5b55;margin-top:8px}
.issue-detail .paper-list{break-inside:avoid;column-span:all}
.page-header{max-width:1500px;margin:32px auto;padding:0 24px;border-bottom:2px solid #111}
.page-header h1{font-size:54px;line-height:1;margin:0 0 10px;font-family:Georgia,'Times New Roman',serif}
.timeline-day{max-width:1100px;margin:28px auto;padding:0 24px}
.timeline-day h2{border-bottom:2px solid #111;padding-bottom:8px}
.timeline-day article{display:grid;grid-template-columns:70px 1fr 240px;gap:16px;border-bottom:1px solid #ddd;padding:12px 0}
.timeline-day span{font-weight:700;color:#8a2f21}
.timeline-day em{font-style:normal;color:#65615a}
@media (max-width:900px){.hero-grid{grid-template-columns:1fr}.lead-story h1{font-size:44px}.timeline-day article{grid-template-columns:60px 1fr}.timeline-day em{grid-column:2}}
"""
    (ROOT / "assets" / "visual-ssl.css").write_text(css, encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for sub in ["assets", "issues", "papers", "pages", "data"]:
        (ROOT / sub).mkdir(exist_ok=True)

    source_css = Path(r"C:\Users\Administrator\.claude\skills\paper-digest-site\templates\styles.css")
    if source_css.exists():
        shutil.copy2(source_css, ROOT / "assets" / "styles.css")

    src_img_dir = REPORT_ROOT / "mineru_v4" / "2605.21059" / "unzipped" / "images"
    dst_img_dir = ROOT / "assets" / "mineru" / "2605.21059" / "images"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    img_pairs = {
        "75a9ed294f67085fc5768f2ac1eb0c9152d555c7052fb8f302b250608e2e6bc3.jpg": "fig1.jpg",
        "ea0f86648f1fa8df761bf52d5fed3751ffe384aa86c71a8c0ef5727ea8d6b209.jpg": "fig2.jpg",
    }
    for src, dst in img_pairs.items():
        if (src_img_dir / src).exists():
            shutil.copy2(src_img_dir / src, dst_img_dir / dst)

    report_md = (REPORT_ROOT / "2026-05-24.md").read_text(encoding="utf-8")
    (ROOT / "data" / "papers.json").write_text(json.dumps(PAPERS, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / ".nojekyll").write_text("", encoding="utf-8")
    write_css()
    write_index(report_md)
    write_issue(report_md)
    write_catalog()
    write_timeline()
    for p in PAPERS:
        write_paper(p)
    print(f"Built site at {ROOT}")


if __name__ == "__main__":
    main()
