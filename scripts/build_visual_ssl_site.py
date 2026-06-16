from __future__ import annotations

import html
import json
import re
import shutil
from datetime import date, datetime
from pathlib import Path


ROOT = Path(r"H:\Desktop\visual_ssl_digest_site")
REPORT_ROOT = Path(r"H:\Desktop\visual_ssl_paper_reports")
CSS_VERSION = "20260615ssl6"
CURRENT_DATE = "2026-05-25"


CONFERENCE_REMINDERS = [
    {
        "name": "ACM MM 2026",
        "ccf": "CCF A",
        "track": "Main regular paper",
        "deadline": "2026-04-01",
        "url": "https://2026.acmmm.org/site/cfp-guidelines.html",
        "note": "主会 regular paper 已截稿；不再用 workshop/demo/open-source 充当投稿提醒。",
    },
    {
        "name": "NeurIPS 2026",
        "ccf": "CCF A",
        "track": "Main paper",
        "deadline": "2026-05-07",
        "url": "https://ccf-cycle.vercel.app/conference/neurips",
        "note": "主会 submission 已截稿；仅保留状态，不作为未来投稿窗口。",
    },
    {
        "name": "AAAI 2027",
        "ccf": "CCF A",
        "track": "Main paper submission",
        "deadline": "2026-07-27",
        "url": "https://github.com/ccfddl/ccf-deadlines/blob/main/conference/AI/aaai.yml",
        "note": "CCF DDL: abstract 2026-07-20；进入 1 个月窗口后重点提醒。",
    },
]


PAPERS = [
    {
        "id": "2605.22098",
        "title": "TextTeacher: What Can Language Teach About Images?",
        "short": "TextTeacher",
        "category": "Language-guided visual representation",
        "priority": "P1",
        "date": "2026-05-25",
        "venue": "TMLR 2026",
        "url": "https://arxiv.org/abs/2605.22098",
        "thesis": "冻结文本编码器可以作为训练期语义锚点，推理时仍保留纯视觉模型。",
        "takeaway": "今天最值得先读：它不是完整 CLIP 式预训练，而是把语言语义变成低成本的视觉表征塑形目标。",
        "method": "frozen text encoder semantic anchors for image representation training",
    },
    {
        "id": "2605.23482",
        "title": "Multimodal Distribution Matching for Vision-Language Dataset Distillation",
        "short": "MDM",
        "category": "VLM dataset distillation",
        "priority": "P1",
        "date": "2026-05-25",
        "venue": "CVPR 2026",
        "url": "https://arxiv.org/abs/2605.23482",
        "thesis": "图文数据蒸馏不能只压缩样本，还要保留联合嵌入空间里的跨模态几何关系。",
        "takeaway": "适合关注多模态预训练数据效率的人精读：它把合成图文对放进分布匹配和对齐保持问题里。",
        "method": "geometric multimodal distribution matching for image-text dataset distillation",
    },
    {
        "id": "2605.21931",
        "title": "EvoVid: Temporal-Centric Self-Evolution for Video Large Language Models",
        "short": "EvoVid",
        "category": "Video SSL / self-evolution",
        "priority": "P2",
        "date": "2026-05-25",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2605.21931",
        "thesis": "未标注视频可以通过时间敏感问题生成和片段定位奖励构造自演化训练信号。",
        "takeaway": "它把视频自监督和 Video-LLM 后训练接起来，值得和 V-JEPA、视频掩码建模放在一起看。",
        "method": "temporal-aware questioner reward and temporal-grounded solver reward",
    },
    {
        "id": "2605.21642",
        "title": "Ablate-to-Validate: Are Vision-Language Models Really Using Continuous Thought Tokens?",
        "short": "Ablate-to-Validate",
        "category": "VLM diagnostics",
        "priority": "P2",
        "date": "2026-05-25",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2605.21642",
        "thesis": "很多连续视觉 thought token 的收益可能来自长度、锚点或正则，而不一定来自 token 内容本身。",
        "takeaway": "这是一个诊断工具型论文：以后看 latent visual token 方法时，应该要求内容替换消融。",
        "method": "token replacement test for continuous and discrete visual thought tokens",
    },
    {
        "id": "2605.20302",
        "title": "Neural Collapse by Design: Learning Class Prototypes on the Hypersphere",
        "short": "Neural Collapse by Design",
        "category": "Representation geometry",
        "priority": "P3",
        "date": "2026-05-25",
        "venue": "ICML 2026 · status update",
        "url": "https://arxiv.org/abs/2605.20302",
        "thesis": "监督对比学习和分类原型几何可以统一到单位超球面上的 prototype contrast。",
        "takeaway": "偏理论和监督表征，但对理解 linear probe、原型、均匀性和表征几何有参考价值。",
        "method": "prototype contrast with NTCE / NONL objectives on the hypersphere",
    },
    {
        "id": "2605.23790",
        "title": "Exploring Deep Learning for Event-Based Saliency Prediction with a Transformer-Based Model",
        "short": "SEST",
        "category": "Event-camera SSL transfer",
        "priority": "扫读",
        "date": "2026-05-25",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2605.23790",
        "thesis": "自监督 event Swin backbone 可以迁移到标注稀缺的事件相机 saliency prediction。",
        "takeaway": "模态较专，适合作为自监督 backbone 迁移到异构视觉模态的扫读案例。",
        "method": "self-supervised event-based Swin Transformer plus lightweight saliency decoder",
    },
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
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/2605.21059",
        "thesis": "多模态模型不一定需要完整 joint tuples；只要模态图连通，pairwise supervision 可以成为可扩展替代品。",
        "takeaway": "它把多模态表征学习的问题改写成模态图连通性问题，重点不在堆更多模态，而在证明 pairwise supervision 也能学习共享/私有 latent 结构。",
        "method": "self-modal reconstruction + pairwise contrastive latent alignment",
    },
]


def e(text: str | None) -> str:
    return html.escape(text or "", quote=True)


def strip_md(text: str) -> str:
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)]\(([^)]+)\)", r"\1", text)
    return text.replace("**", "").replace("`", "")


def arxiv_id_from_url(url: str) -> str | None:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", url)
    return match.group(1) if match else None


def first_md_link(text: str) -> tuple[str, str] | None:
    match = re.search(r"\[([^\]]+)]\(([^)]+)\)", text)
    if not match:
        return None
    return strip_md(match.group(1)).strip(), match.group(2).strip()


def split_md_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def report_date_from_path(path: Path) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", path.name)
    if match:
        return match.group(0)
    return date.today().isoformat()


def zh_date(date_str: str) -> str:
    try:
        parsed = date.fromisoformat(date_str)
    except ValueError:
        return date_str
    return f"{parsed.month} 月 {parsed.day} 日"


def latest_report_path() -> Path:
    latest = REPORT_ROOT / "latest.md"
    if latest.exists():
        return latest
    reports = sorted(REPORT_ROOT.glob("20??-??-??.md"), key=lambda p: p.name)
    if not reports:
        raise FileNotFoundError(f"No Markdown reports found under {REPORT_ROOT}")
    return reports[-1]


def flatten_caption(value) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def zh_caption_for(en_caption: str, title: str) -> str:
    clean = re.sub(r"\s+", " ", en_caption).strip()
    lower = clean.lower()
    if "first-page visual preview" in lower or "full figure extraction is pending" in lower:
        return f"这是 {title} 的 PDF 首页预览兜底图，不是 MinerU 结构化抽取出的论文图表。正式抽图完成后应优先展示方法图、实验图或表格。"
    if any(word in lower for word in ["overview", "framework", "pipeline", "architecture"]):
        return f"这张图概括 {title} 的整体方法流程。阅读时先看模块之间传递的训练信号，再看作者如何把目标拆成可优化的子问题。"
    if any(word in lower for word in ["ablation", "comparison", "results", "accuracy", "performance", "table"]):
        return f"这张图/表用于判断 {title} 的实验收益来自哪里。重点看替换、消融或跨模型设置下趋势是否一致，而不是只看单个最高分。"
    if any(word in lower for word in ["attention", "similarity", "grounding", "visualization"]):
        return f"这张可视化用来解释 {title} 学到的中间表征或对齐关系。重点看它是否支持正文里的机制判断。"
    return f"这张图来自论文 PDF 的结构化抽取。当前用于辅助理解 {title} 的方法或实验，请结合正文精读段落一起看。"


def figure_label_from_caption(caption: str, fallback: str) -> str:
    clean = re.sub(r"\s+", " ", caption).strip()
    if not clean:
        return fallback
    match = re.match(r"((?:Figure|Fig\.?|Table)\s*\d+[A-Za-z]?)\.?\s*(.*)", clean, flags=re.I)
    if match:
        prefix = match.group(1).replace("Fig.", "Figure").replace("Fig", "Figure")
        rest = match.group(2).strip()
        if rest:
            title = rest.split(". ")[0].strip()
            return f"{prefix} · {title[:70]}"
        return prefix
    return clean.split(". ")[0][:90]


_IMG_SIZE_CACHE: dict[str, tuple[int, int] | None] = {}


def image_size(path: Path) -> tuple[int, int] | None:
    """Read (width, height) from a JPEG/PNG/GIF without external deps.
    Returns None if it can't be determined."""
    key = str(path)
    if key in _IMG_SIZE_CACHE:
        return _IMG_SIZE_CACHE[key]
    size: tuple[int, int] | None = None
    try:
        with open(path, "rb") as fh:
            head = fh.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
                w = int.from_bytes(head[16:20], "big")
                h = int.from_bytes(head[20:24], "big")
                size = (w, h)
            elif head[:6] in (b"GIF87a", b"GIF89a"):
                w = int.from_bytes(head[6:8], "little")
                h = int.from_bytes(head[8:10], "little")
                size = (w, h)
            elif head[:2] == b"\xff\xd8":  # JPEG
                fh.seek(2)
                b = fh.read(1)
                while b and b != b"":
                    while b != b"\xff":
                        b = fh.read(1)
                        if not b:
                            break
                    while b == b"\xff":
                        b = fh.read(1)
                    if not b:
                        break
                    marker = b[0]
                    if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                                  0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        fh.read(3)  # length(2) + precision(1)
                        hh = fh.read(2)
                        ww = fh.read(2)
                        size = (int.from_bytes(ww, "big"), int.from_bytes(hh, "big"))
                        break
                    else:
                        seg_len = int.from_bytes(fh.read(2), "big")
                        if seg_len < 2:
                            break
                        fh.seek(seg_len - 2, 1)
                        b = fh.read(1)
    except Exception:
        size = None
    _IMG_SIZE_CACHE[key] = size
    return size


def aspect_ratio_str(path: Path, default: str = "16/9") -> str:
    dims = image_size(path)
    if not dims or dims[1] == 0:
        return default
    return f"{dims[0]}/{dims[1]}"


def is_bad_thumbnail(path: Path) -> bool:
    """True for thin banner strips / equation rows / tiny fragments that make
    ugly thumbnails (the MinerU 'first image is a header bar' problem)."""
    dims = image_size(path)
    if not dims:
        return False
    w, h = dims
    if w == 0 or h == 0:
        return True
    ar = w / h
    if ar > 3.6 or ar < 0.32:   # long thin strip (header/equation) or sliver
        return True
    if w < 120 or h < 90:        # too small to read
        return True
    return False


def content_list_path(pid: str) -> Path | None:
    root = ROOT / "assets" / "mineru" / pid
    paths = sorted(root.glob("*content_list.json"))
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "First-page visual preview" not in text and "full figure extraction is pending" not in text:
            return path
    return paths[0] if paths else None


def figures_for(pid: str, depth: int = 1, max_count: int = 2) -> list[dict]:
    prefix = "../" * depth
    fallback = {
        "2605.21059": [
            {
                "src": f"{prefix}assets/mineru/2605.21059/images/fig1.jpg",
                "label": "Fig. 1 · Pairwise supervision is the scalable object",
                "caption_en": "Joint tuples are expensive; pairwise text-image, image-tactile, and text-3D data can form a connected modality graph.",
                "caption_zh": "这张图说明论文的核心动机：不一定要收集完整多模态 tuple，只要 pairwise supervision 形成连通的模态图，也能支撑共享表示学习。",
            },
            {
                "src": f"{prefix}assets/mineru/2605.21059/images/fig2.jpg",
                "label": "Fig. 2 · Shared and modality-specific latent factors",
                "caption_en": "The paper separates common causal factors from modality-private variables before alignment.",
                "caption_zh": "这张图展示共享 latent 与模态私有变量的生成假设：先分离跨模态共有因素，再做 pairwise alignment，避免把私有噪声也强行对齐。",
            },
        ]
    }
    if pid == "2605.21642":
        return [
            {
                "src": f"{prefix}assets/mineru/2605.21642/images/1be6853025daf7f1c0ddea23f8e0f0352225fb287accf6be05baeb6301a09169.jpg",
                "label": "Figure 1 · Token Replacement Test",
                "caption_en": "Overview of TRT. The test fixes the prompt, image, token budget, and decoding procedure, then replaces the intermediate thought-token span with identity, random, first-repeat, or oracle tokens to separate real content utilization from span-position and token-budget effects.",
                "caption_zh": "TRT 的核心不是再训练一个模型，而是在推理时只替换中间 thought-token span。若模型真正读取 token 内容，random/zero 应明显掉点，oracle 应带来可解释收益；若几乎不变，收益更可能来自固定位置、额外 token budget 或训练正则。",
            },
            {
                "src": f"{prefix}assets/mineru/2605.21642/images/c166965b7e312881d8f4ba3670107db69dd78ffc0cade63fc95b2c6591e56594.jpg",
                "label": "Table/Figure · TRT results snapshot",
                "caption_en": "A high-signal extracted result panel from the paper. Use it together with Tables 1-4: continuous depth spans often keep much of their utility under content replacement, while discrete depth tokens show larger drops under random replacement.",
                "caption_zh": "这张结果图用于辅助读实验部分：连续 depth span 在替换内容后常常仍保留收益，说明模型可能没有充分消费 token 内容；离散 depth token 在 random replacement 下更容易掉点，更像真正的信息瓶颈。",
            },
        ][:max_count]
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
    title = next((p["short"] for p in PAPERS if p.get("id") == pid), "this paper")
    for item in data:
        if item.get("type") not in {"image", "chart"} or not item.get("img_path"):
            continue
        img_rel = str(item["img_path"]).replace("\\", "/")
        img_path = ROOT / "assets" / "mineru" / pid / img_rel
        if not img_path.exists():
            continue
        cap = (
            flatten_caption(item.get("image_caption"))
            or flatten_caption(item.get("chart_caption"))
            or flatten_caption(item.get("table_caption"))
        )
        content = str(item.get("content") or "")
        blob = f"{cap} {content} {img_rel}".lower()
        score = img_path.stat().st_size / 1000
        if cap:
            score += 220
        if item.get("type") == "chart":
            score += 180
        for idx, kw in enumerate(keywords):
            if kw in blob:
                score += 500 - idx * 20
        # Avoid tiny fragments when a real method figure exists.
        if img_path.stat().st_size < 6000:
            score -= 120
        if not cap and selected:
            score -= 250
        # Reject thin banner strips / equation rows / slivers as thumbnails:
        # this is the MinerU "first extracted image is a header bar" problem.
        if is_bad_thumbnail(img_path):
            score -= 900
        dims = image_size(img_path)
        if dims and dims[1]:
            ar = dims[0] / dims[1]
            if 0.6 <= ar <= 2.2:   # nicely-proportioned figure
                score += 60
        fallback_label = "Figure · Extracted visual evidence"
        fallback_en = f"This visual block was extracted from the paper PDF without a structured caption. It is included only as supporting visual evidence for {title}; prefer figures with explicit captions when available."
        selected.append({
            "score": score,
            "src": f"{prefix}assets/mineru/{pid}/{img_rel}",
            "label": figure_label_from_caption(cap, fallback_label),
            "caption_en": cap or fallback_en,
            "caption_zh": zh_caption_for(cap or fallback_en, title),
        })
    selected.sort(key=lambda x: x["score"], reverse=True)
    return [{k: v for k, v in fig.items() if k != "score"} for fig in selected[:max_count]] or fallback.get(pid, [])


def figure_by_caption(pid: str, depth: int, needle: str) -> dict | None:
    prefix = "../" * depth
    path = content_list_path(pid)
    if not path:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for item in data:
        if item.get("type") != "image" or not item.get("img_path"):
            continue
        cap = flatten_caption(item.get("image_caption"))
        if needle.lower() not in cap.lower():
            continue
        img_rel = str(item["img_path"]).replace("\\", "/")
        img_path = ROOT / "assets" / "mineru" / pid / img_rel
        if not img_path.exists():
            continue
        return {
            "src": f"{prefix}assets/mineru/{pid}/{img_rel}",
            "label": cap.split(". ")[0][:90] if cap else needle,
            "caption": cap,
        }
    return None


def render_fig(fig: dict, cls: str = "full-fig") -> str:
    if fig.get("caption_en") or fig.get("caption_zh"):
        caption = f"""<span class="fig-en">{e(fig.get('caption_en') or '')}</span><span class="fig-zh">{e(fig.get('caption_zh') or '')}</span>"""
    else:
        caption = e(fig["caption"])
    return f"""<figure class="{cls}"><img src="{e(fig['src'])}" alt="{e(fig['label'])}"><figcaption><b>{e(fig['label'])}</b>{caption}</figcaption></figure>"""


def tc_jepa_figure(pid: str, needle: str, label: str, caption_en: str, caption_zh: str) -> dict | None:
    fig = figure_by_caption(pid, 1, needle)
    if not fig:
        return None
    fig["label"] = label
    fig["caption_en"] = caption_en
    fig["caption_zh"] = caption_zh
    return fig


def tc_jepa_body(p: dict) -> str:
    figs = [
        tc_jepa_figure(
            p["id"],
            "Figure 2",
            "Figure 2 · Text-conditioned JEPA predictor",
            "TC-JEPA conditions the I-JEPA predictor g_phi on multiple captions. Patch features cross-attend to T5 word embeddings at several predictor layers, then max-pool across captions; L_sparse and L_consistency regularize patch-word similarities.",
            "TC-JEPA 不是把 caption 当作额外分类头，而是在 JEPA predictor 的多层里用词级 cross-attention 条件化 patch feature。多句 caption 分别提供条件信号，再通过 max-pooling 选择最有帮助的语义线索；稀疏和跨层一致性约束让 patch-word 对应更集中、更稳定。",
        ),
        tc_jepa_figure(
            p["id"],
            "Figure 5",
            "Figure 5 · Patch-word grounding and prediction error",
            "The visualization links sparse patch-word similarities with feature prediction quality. TC-JEPA's conditioned predictor produces lower target-patch error than plain I-JEPA on the shown masked regions.",
            "这张图要看两件事：上半部分说明模型学到了稀疏的 patch-word 对应，下半部分比较 I-JEPA 与 TC-JEPA 的目标 patch 预测误差。作者想证明文字条件确实在降低被遮挡区域的预测不确定性，而不是只让 attention 图更好看。",
        ),
        tc_jepa_figure(
            p["id"],
            "Figure 4",
            "Figure 4 · Ablation of text-conditioning components",
            "The ablation separates the gains from text conditioning, sparsity regularization, consistency regularization, and multi-layer conditioning.",
            "消融图用来判断收益来自哪里：只有 caption 不够，词级对应的稀疏约束、多层一致性以及多层 predictor 条件化都在贡献最终的分类和分割性能。",
        ),
        tc_jepa_figure(
            p["id"],
            "Figure 8",
            "Figure 8 · Efficiency and scaling behavior",
            "The efficiency plot compares downstream gains against pretraining compute. Because text conditioning is attached to the lightweight predictor, the added cost is smaller than modifying the visual encoder.",
            "效率图的重点是成本位置：TC-JEPA 把文字条件加在较轻的 predictor 上，测试时还会丢弃 predictor 和 text conditioner，因此额外训练成本相对可控，主要收益体现在分割和细粒度任务。",
        ),
    ]
    fig_html = "\n".join(render_fig(fig) for fig in figs if fig)
    body = f"""<article class="paper-detail deep-read">
  <div class="paper-main">
    <div class="kicker">{e(p['category'])} · {e(p['priority'])} · {e(p['date'])}</div>
    <h1 class="paper-headline">TC-JEPA：把 masked feature prediction 的不确定性，交给 caption 来收束</h1>
    <p class="dek">这篇的重点不是“又做了一个图文预训练模型”，而是把 JEPA 的预测目标从纯视觉上下文变成 text-conditional feature prediction：测试时仍只用视觉 encoder，训练时用 caption 帮 predictor 学到更语义化的 patch 表征。</p>
    <p class="byline"><strong>{e(p['title'])}</strong>　{e(p['venue'])}　<a href="{e(p['url'])}">原文链接</a></p>
    <div class="paper-facts">
      <span><b>编号</b>{e(p['id'])}</span>
      <span><b>优先级</b>{e(p['priority'])}</span>
      <span><b>类别</b>JEPA / vision-language SSL</span>
      <span><b>训练信号</b>synthetic captions, feature prediction only</span>
      <span><b>测试开销</b>丢弃 predictor 与 text conditioner</span>
      <span><b>图表</b>方法、预测误差、消融与效率图</span>
    </div>
    <div class="feature-body">
      <p class="lead dropcap">先读这篇要从“为什么 I-JEPA 会不确定”开始，而不是从分数开始。I-JEPA 只拿可见 context patch 去预测 target patch feature，遮挡区域可能对应很多合理语义；TC-JEPA 的假设是 caption 能提供场景组成、属性和空间关系，把 target feature 的可行解空间变窄。</p>

      <section class="reading-note">
        <h2>读前定位</h2>
        <p><b>先抓训练信号。</b>TC-JEPA 的核心不是给 I-JEPA 加一个语言标签头，而是把“预测被遮挡 patch feature”改成“在文字条件下预测被遮挡 patch feature”。测试时文字分支和 predictor 都会丢弃，留下的仍是纯视觉 encoder。</p>
        <p><b>再抓边界。</b>它不是 CLIP 替代品，也不是像素重建路线；它更像一条条件化 JEPA 路线：借 caption 降低目标 feature 的歧义，同时保留 JEPA 对 dense representation 的偏好。</p>
      </section>

      {fig_html}

      <h2>方法拆解：它到底在哪里用文字</h2>
      <ol class="method-steps">
        <li><b>视觉骨架仍是 JEPA。</b>context encoder 处理可见 patch，target encoder 给出目标 patch feature，predictor 负责从 context 与 mask token 预测 target feature。</li>
        <li><b>caption 不进最终 encoder。</b>每张图先用 ShareGPT4V 生成多句 caption，训练时随机采样默认 8 句；这些句子只服务于 predictor 的条件化。</li>
        <li><b>T5 word embedding 替代 CLIP text encoder。</b>作者强调这里要保留自然语言组合与顺序信息，目标是 patch-word 细粒度对应，而不是全局图文 embedding 对齐。</li>
        <li><b>多层 sparse cross-attention。</b>在 predictor 的多个层里，patch query 对 caption word sequence 做 cross-attention；每个 caption 分开条件化，再用 max-pooling 选出最有用的条件信号。</li>
        <li><b>两个正则约束。</b>sparsity 让 patch-word 对应更集中，cross-layer consistency 让不同层的对应关系更稳定；Figure 4 的消融说明这两项不是装饰。</li>
      </ol>

      <h2>图表导读：先看哪几张</h2>
      <ul class="method-steps">
        <li><b>Figure 2</b> 定义模型接口：caption 只进 predictor，多层 cross-attention 负责把词级语义写入 masked feature prediction。</li>
        <li><b>Figure 5</b> 是最关键证据：它把 patch-word grounding 和 prediction error 放在一起，避免只展示漂亮 attention。</li>
        <li><b>Figure 4</b> 用来判断收益来源：文字条件、稀疏约束、一致性约束、多层注入都要分开看。</li>
        <li><b>Figure 8</b> 负责回答成本问题：收益是否值得额外 caption 与 text conditioner 训练成本。</li>
      </ul>

      <h2>关键结果：更像“细粒度视觉表征”而不是 CLIP 替代品</h2>
      <div class="stat-grid">
        <div><b>IN-1k linear</b><span>TC-JEPA ViT-B/L/H: 75.8 / 79.6 / 80.4</span><em>I-JEPA 同尺度为 72.9 / 77.5 / 79.3。</em></div>
        <div><b>ADE20k linear seg</b><span>CC27M ViT-L/16: 42.1 mIoU</span><em>超过 DINOv2-L/14 distilled on LVD-142M 的 41.8。</em></div>
        <div><b>Dense transfer</b><span>COCO APb 55.2, ADE20k FT 55.7</span><em>在检测、分割上比 MIM baseline 更稳。</em></div>
        <div><b>VLM downstream</b><span>COCO CIDEr 111.6, GQA 46.3, VQAv2 57.8</span><em>冻结视觉 encoder 后接 LiT-Decoder，优于 CLIP/SPARC baseline。</em></div>
      </div>

      <h2>为什么这对通用视觉自监督重要</h2>
      <p>它给 JEPA 开了一条很实用的路：训练阶段可以借弱文本监督降低 feature prediction 的歧义，但最终部署仍保留纯视觉 encoder。这比“把所有东西都塞进图文对比学习”更接近自监督表征学习的核心问题，也更适合关注 dense prediction、patch-level semantics 和 video/image foundation model 的读者。</p>
      <p>Figure 5 最值得细看：它不是只展示 attention 好看，而是把 patch-word similarity 和 prediction error 放在一起。若这个现象在更大数据和视频上成立，后续很可能出现 text-conditioned JEPA、motion-conditioned JEPA、depth/pose-conditioned JEPA 等一系列“条件化预测表征”路线。</p>

      <h2>我会保留的疑问</h2>
      <p>第一，caption 是 ShareGPT4V 合成的，强 captioner 本身带来的先验有多大还需要更干净的 ablation。第二，多 caption 的默认 N=8 已经接近饱和，但这会让预处理和数据资产成本上升。第三，它在 classification 上仍未完全替代强 augmentation/invariance SSL；真正的优势更集中在 segmentation、fine-grained grounding 和 VLM downstream。</p>

      <h2>精读顺序</h2>
      <p>建议先读 Figure 2 和方法 3.2/3.3，理解 word-level conditioner；再读 Table 1/3 与 Figure 4/5，判断收益来自哪里；最后扫 Appendix 的 compute、caption robustness 和 N sensitivity，确认它是不是能放进自己的预训练 pipeline。</p>
    </div>
  </div>
  <aside class="paper-side">
    <div class="side-box"><h4>图表导读</h4><p>先看 Figure 2/5，再看 Figure 4/8；分别对应方法接口、预测证据、组件消融和训练成本。</p></div>
    <div class="side-box"><h4>核心判断</h4><dl><dt>相关性</dt><dd>P1，通用视觉自监督强相关</dd><dt>会议等级</dt><dd>ICML 2026，CCF A</dd><dt>读法</dt><dd>方法图优先，表格次之</dd></dl></div>
    <div class="side-box"><h4>原文</h4><p><a href="{e(p['url'])}">{e(p['url'])}</a></p></div>
  </aside>
</article>"""
    return shell(p["short"], body, 1, display_date=p["date"])


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


def parse_paper_index(md: str, report_date: str) -> list[dict]:
    marker = "## 论文索引"
    start = md.find(marker)
    if start < 0:
        return []
    lines = md[start:].splitlines()
    rows: list[list[str]] = []
    in_table = False
    for line in lines:
        if line.strip().startswith("|"):
            parts = split_md_row(line)
            if all(re.fullmatch(r":?-{3,}:?", p or "") for p in parts):
                in_table = True
                continue
            rows.append(parts)
            in_table = True
        elif in_table:
            break

    if len(rows) < 2:
        return []

    parsed: list[dict] = []
    for row in rows[1:]:
        if len(row) < 5:
            continue
        priority, paper_cell, kind, relevance, reason = row[:5]
        if "过滤" in priority:
            continue
        if strip_md(priority).strip() == "-" and "无新增" in strip_md(paper_cell):
            continue
        link = first_md_link(paper_cell)
        if link:
            title, url = link
        else:
            title = strip_md(paper_cell).strip()
            url = ""
        pid = arxiv_id_from_url(url) or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48]
        short = title.split(":")[0].strip()
        if len(short) > 42:
            short = " ".join(short.split()[:5]) or title[:24]
        category = kind.replace("arXiv +", "").replace("arXiv 更新 +", "").strip()
        if not category or category == "arXiv":
            category = "Visual SSL / representation"
        venue = kind.strip() or "arXiv"
        if re.search(r"ICML|CVPR|NeurIPS|ICLR|ECCV|AAAI|ACL|TMLR|MM", kind, re.I):
            venue = kind.strip()
        method = strip_md(reason).strip().rstrip("。")
        parsed.append(
            {
                "id": pid,
                "title": title,
                "short": short,
                "category": category,
                "priority": strip_md(priority).strip(),
                "date": report_date,
                "venue": venue,
                "url": url or "#",
                "thesis": method + "。" if method else "本期日报自动纳入的候选论文。",
                "takeaway": f"{strip_md(relevance).strip()}相关；详见方法、贡献和实验边界。" if relevance else f"{title} 是本期日报自动纳入的候选论文。",
                "method": method or f"{relevance} relevance; see original report for details",
            }
        )
    return parsed


def merge_report_papers(report_papers: list[dict]) -> list[dict]:
    if not report_papers:
        return PAPERS
    merged: list[dict] = []
    seen: set[str] = set()
    static_by_id = {p["id"]: p for p in PAPERS}
    for p in report_papers:
        richer = static_by_id.get(p["id"])
        if richer:
            item = dict(richer)
            item["date"] = p["date"]
            merged.append(item)
        else:
            merged.append(p)
        seen.add(p["id"])
    for p in PAPERS:
        if p["id"] not in seen:
            merged.append(p)
            seen.add(p["id"])
    return merged


def load_catalog_papers(current_report_path: Path, current_report_md: str, current_date: str) -> list[dict]:
    report_papers = parse_paper_index(current_report_md, current_date)
    seen_dates = {current_date}
    for report_path in sorted(REPORT_ROOT.glob("20??-??-??.md"), key=lambda p: p.name, reverse=True):
        report_date = report_date_from_path(report_path)
        if report_date in seen_dates:
            continue
        try:
            md = report_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            md = report_path.read_text(encoding="utf-8-sig")
        parsed = parse_paper_index(md, report_date)
        if parsed:
            report_papers.extend(parsed)
            seen_dates.add(report_date)
    return merge_report_papers(report_papers)


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


def norm_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", strip_md(text).lower())


def detail_section(md: str, p: dict) -> str:
    target = norm_for_match(p["title"])
    short = norm_for_match(p["short"])
    sections = list(re.finditer(r"^###\s+(.+?)\s*$", md, flags=re.M))
    for idx, match in enumerate(sections):
        heading = match.group(1)
        normalized = norm_for_match(heading)
        if not normalized:
            continue
        if target and (target in normalized or normalized in target):
            start = match.end()
        elif short and len(short) > 5 and short in normalized:
            start = match.end()
        else:
            continue
        end = sections[idx + 1].start() if idx + 1 < len(sections) else len(md)
        next_major = md.find("\n## ", start, end)
        if next_major >= 0:
            end = next_major
        return md[start:end].strip()
    return ""


def detail_fields(md: str, p: dict) -> dict[str, str]:
    section = detail_section(md, p)
    fields: dict[str, str] = {}
    if not section:
        return fields
    current_key = ""
    for raw in section.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = re.match(r"-\s+\*\*([^*]+)\*\*[：:]\s*(.*)$", line)
        if not match:
            match = re.match(r"-\s+([^：:*][^：:]{1,30})[：:]\s*(.*)$", line)
        if match:
            current_key = strip_md(match.group(1)).strip()
            fields[current_key] = strip_md(match.group(2)).strip()
        elif current_key and (line.startswith("  ") or not line.startswith("- ")):
            fields[current_key] = (fields[current_key] + " " + strip_md(line)).strip()
    return fields


def choose_field(fields: dict[str, str], *names: str) -> str:
    for name in names:
        if fields.get(name):
            return fields[name]
    return ""


def paragraph(text: str) -> str:
    text = strip_md(text).strip()
    if not text:
        return ""
    return f"<p>{e(text)}</p>"


def first_sentence(text: str, limit: int = 110) -> str:
    text = strip_md(text).strip()
    if not text:
        return ""
    match = re.match(r"(.+?[。.!?])(?:\s|$)", text)
    sentence = match.group(1) if match else text
    if len(sentence) > limit:
        sentence = sentence[:limit].rstrip() + "..."
    return sentence


def unique_sentence_join(*parts: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        clean = strip_md(part).strip()
        if not clean:
            continue
        key = norm_for_match(clean)
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return " ".join(out)


def time_machine_reading_html(fields: dict[str, str], p: dict) -> str:
    method = choose_field(fields, "核心方法") or p["method"]
    contribution = choose_field(fields, "主要贡献") or p["takeaway"]
    experiment = choose_field(fields, "实验亮点")
    limitation = choose_field(fields, "局限性")
    relevance = choose_field(
        fields,
        "与通用视觉自监督的关系",
        "为什么相关",
        "为什么与通用视觉自监督方向相关",
        "与通用视觉 SSL 的相关性",
    )
    return f"""
      <section class="reading-note">
        <h2>读前定位</h2>
        <p><b>先按“视频表征预训练目标”读。</b>这篇的核心不是再做一个视频生成或视频问答系统，而是把点轨迹当作可遮蔽、可重建的运动模态，让模型从运动连续性中学习可迁移表示。</p>
        <p><b>再检查它是否真的通用。</b>重点看 zero-shot 任务覆盖、训练数据规模对照，以及收益是否来自 point-track 监督本身，而不是更强数据、更长训练或更大的 backbone。</p>
      </section>

      <h2>核心问题</h2>
      <p>视频自监督长期夹在两类目标之间：重建像素容易学到低层纹理，图文对齐又容易被 caption 覆盖范围限制。TIME Machine 选择第三条路：直接让模型预测物体和区域随时间移动的轨迹，把 motion 当成比字幕更原生的视频监督信号。</p>
      <p>因此它要回答的问题是：如果视觉系统只从点轨迹的时序结构中学习，是否能形成足够通用的 motion-centric representation，并迁移到不止一个视频理解任务。</p>

      <h2>方法拆解</h2>
      <p>{e(method)}</p>
      <p>读方法时可以拆成三层：第一层是 point-track 抽取和表示格式，决定监督信号的噪声与覆盖；第二层是 MAE 式遮蔽比例、重建目标和时序上下文，决定模型是否学习运动规律而不是记忆局部插值；第三层是 TIME embedding 如何进入下游 zero-shot 或迁移评估，决定它是不是一个可复用表征。</p>

      <h2>主要贡献</h2>
      <p>{e(contribution)}</p>
      <p>对通用视觉自监督更关键的是，它把“运动”从视频模型的辅助线索提升为主训练目标：不依赖人工标签，不依赖 caption，也不要求先把视频转成语义问答。这个设定适合和 V-JEPA、VideoMAE、masked feature prediction、world-model 预训练放在同一条谱系里比较。</p>

      <h2>实验看点</h2>
      {paragraph(experiment) or "<p>实验部分先看作者怎样做 zero-shot transfer，再看是否控制训练数据量、backbone 规模、训练步数和 point-track 质量。只要这些对照不完整，就不能把性能差异全部归因给 motion MAE 目标。</p>"}
      <p>最值得抠的是两类消融：一是 mask/reconstruction 目标和普通视频 MAE、光流/轨迹预测之间的差异；二是跨数据、跨任务迁移是否仍保持趋势。如果只在运动显著的数据或任务上强，结论应收窄为 motion-biased representation，而不是完整通用视频表征。</p>

      <h2>局限与读法</h2>
      <p>{e(limitation or "point-track 质量、合成到真实视频的迁移、静态外观语义不足，是这条路线最需要警惕的三类边界。")}</p>
      <p>{e(relevance or "它与通用视觉自监督的关系在于：把视频中的运动连续性变成可扩展预训练信号。")} 精读时建议先看方法图和数据构造，再看 zero-shot 表格，最后看消融；不要只拿一句“少 4 个数量级数据接近 SOTA”当结论。</p>
    """


def ablate_to_validate_reading_html() -> str:
    return """
      <section class="reading-note">
        <h2>读前定位</h2>
        <p><b>先把它当成评测协议。</b>这篇不是要发明更强的视觉 thought token，而是给所有 latent visual token 方法加一把尺：把中间 token 替换掉之后，性能是否还成立。</p>
        <p><b>再看它的反直觉结论。</b>连续 token 的收益不一定来自 token 内容本身；在一些设置里，位置、长度、分布匹配或训练过程可能解释掉相当一部分收益。</p>
      </section>

      <h2>核心问题</h2>
      <p>这篇论文不是在提出一个更强的视觉 token 模块，而是在追问一个更基础的问题：VLM 加了连续/latent thought token 之后，准确率上涨到底是因为模型读懂了 token 内容，还是因为多了一段固定位置、额外 token budget 或训练正则？这个问题对视觉自监督和多模态预训练很关键，因为很多方法会把“中间视觉表征”包装成推理能力，但只报告最终 accuracy 很难证明这些中间表征真的被消费。</p>
      <p>作者把问题收窄到可干预的形式：固定 prompt、图像、token 数量和解码过程，只替换中间 thought-token span。如果模型真的依赖其中的视觉信息，随机化或清零应该明显伤害性能，注入 oracle/ground-truth token 应该带来可解释收益；如果性能几乎不变，说明 token 更可能是位置锚点、预算扩容或训练脚手架。</p>

      <h2>图表导读：先看哪几处</h2>
      <ul class="method-steps">
        <li><b>Figure 1</b> 先看 TRT 的操作边界：固定 prompt、图像和解码，只替换 thought-token span。</li>
        <li><b>Table 1-2</b> 看连续 token 与离散 token 在 controlled depth setting 下是否真的携带内容。</li>
        <li><b>Table 3-4</b> 看 random、distribution-matched random、first-repeat、oracle 之间的差异，判断收益来自内容还是形状。</li>
        <li><b>系统测试</b> 再扫 Mirage、Mull-Tokens、CoVT：它们分别暴露出不同的内容依赖、分布依赖和 token 预算依赖。</li>
      </ul>

      <h2>方法拆解</h2>
      <p>Token Replacement Test（TRT）是一组推理时替换实验，而不是新的训练目标。它把视觉 thought span 看成一个可替换对象，分别测试 identity re-injection、zero replacement、random replacement、distribution-matched random、first-repeat、count-matched variant 和 oracle/ground-truth injection。</p>
      <ul>
        <li><b>Zero / random</b>：移除或破坏 token 内容，用来检验模型是否对内容敏感。</li>
        <li><b>Distribution-matched random</b>：匹配预测 token 的边际分布，避免“随机向量尺度不对”导致误判。</li>
        <li><b>First-repeat / count-matched</b>：保留 span 位置和 token 数量，但去掉 token 多样性，拆开 token budget 与 token content 的贡献。</li>
        <li><b>Oracle / GT injection</b>：在有监督视觉目标时注入更好的 token，检验模型是否有能力利用高质量视觉信号。</li>
      </ul>
      <p>为了让替换实验有清楚语义，作者构造了 relative depth reasoning testbed：给图像上的 3/4/5 个点，问哪一个离相机更近。depth span 有连续版本，也有离散 codebook token 版本；连续 token 可来自 SigLIP2、CLIP、DINOv2 等冻结特征空间，离散版本用 VQ-VAE codebook。这样可以把“预测 depth token”和“使用 depth token”分开测试。</p>

      <h2>主要贡献</h2>
      <p>第一，论文把 latent visual token 的有效性从“有没有涨点”改成“内容替换后是否还有效”的诊断问题，这比单纯追加 benchmark 更有解释力。第二，它给出一个可复用的 TRT 协议，后续任何声称引入视觉 thought token、perception token 或 continuous visual reasoning token 的方法，都可以按同一套替换实验报告结果。第三，作者把 TRT 同时放到自建 depth-span testbed 和现成 visual-thinking 系统上，包括 Mirage、Mull-Tokens 和 CoVT，说明这个诊断不是只为某个模型量身定制。</p>
      <p>对日报方向来说，它最有价值的地方是提供了“反证式评估”模板：如果一种自监督/多模态预训练方法声称学到了中间视觉表征，就应该证明下游模型确实会读取这些表征，而不是只从 token 位置、长度或训练辅助损失中获益。</p>

      <h2>实验看点</h2>
      <p>HardBLINK depth reasoning 上，连续 token 并没有稳定压过离散 token。LLaVA-13B 的 no-aux baseline 约 76.68，离散 depth token 到 77.69，而 best continuous 只有 74.46；Qwen2.5-VL-3B 上 no-aux 为 58.87，离散 token 为 71.24，best continuous 为 68.55。这个结果提示：连续表征不天然等于更可用的推理信道。</p>
      <p>TRT 的关键结果是“内容-效用缺口”。在连续 depth span 中，identity、random、oracle、first-repeat 有时差距很小，说明模型可能保留了辅助 span 的收益，却没有真正读取每个 token 的细粒度内容。离散 token 更容易在 random replacement 下掉点，例如 Qwen2.5-VL-3B discrete 从 identity 71.24 掉到 random 51.34，说明离散 codebook 在这个设定里更像可消费的信息瓶颈。</p>
      <p>对现成系统的测试也不完全一致：Mirage 在 Spatial Planning 上 zero latent 会从 76.25 掉到 51.50，但 HardBLINK depth 上更严重，从 26.08 掉到 8.06；CoVT 在 CV-Bench 里 random 会明显崩，但 distribution-matched random 仍接近 identity。这说明不能简单说“所有 latent token 都没用”，更准确的读法是：不同系统对内容、分布、位置和预算的依赖不同，必须拆开测。</p>

      <h2>局限与读法</h2>
      <p>这是一篇诊断论文，不是新的通用视觉预训练框架。它的控制实验集中在 relative depth reasoning，虽然 depth 是很好的几何信号，但不能直接覆盖所有视觉推理类型。另一个局限是 TRT 需要能拦截和替换中间 token；对闭源模型或没有显式 token span 的方法，协议迁移会更麻烦。</p>
      <p>读这篇时建议先看 Figure 1 的 TRT 设计，再看 Table 1 到 Table 4：前者定义“怎么替换”，后者说明 continuous/discrete token 在 controlled setting 下的差异。最后再扫 Mirage、Mull-Tokens、CoVT 的替换实验，重点看 random、distribution-matched random、oracle 三类结果是否符合“模型真的读 token 内容”的预期。</p>
    """


def nav(depth: int = 0, active: str = "") -> str:
    prefix = "../" * depth
    items = [
        ("index.html", "头版"),
        (f"issues/{CURRENT_DATE}.html", "今日速递"),
        ("pages/catalog.html", "论文目录"),
        ("pages/timeline.html", "时间线"),
    ]
    links = "\n".join(
        f'<a class="nav-link{" active" if active == href else ""}" href="{prefix}{href}">{label}</a>'
        for href, label in items
    )
    flame = '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0c.7 2.5-1.2 3.7-2 5-1 1.6-1 4 .8 4.8C6.4 8.5 7.4 7.8 8 7c.3 1.3-.4 2 .9 3 .9.7 1.3 1.6 1.1 2.6C9.7 14.6 8.9 16 6.8 16 4.6 16 3 14.3 3 12c0-3.3 2.6-4.8 2.4-8C6.9 4.6 8.4 5 8 0z"/></svg>'
    streak = f'<span class="nav-streak" title="连续追踪天数">{flame} 连续 <b data-streak>0</b> 天</span>'
    return links + "\n" + streak


def conference_widget(depth: int = 0) -> str:
    try:
        today = date.today()
    except Exception:
        today = datetime.now().date()
    visible = []
    later = []
    for item in CONFERENCE_REMINDERS:
        try:
            deadline = date.fromisoformat(item["deadline"])
        except ValueError:
            continue
        days = (deadline - today).days
        if days < -7:
            continue
        if days < 0:
            status = f"已过 {abs(days)} 天"
            cls = "past"
        elif days == 0:
            status = "今天截止"
            cls = "urgent"
        elif days <= 7:
            status = f"{days} 天内"
            cls = "urgent"
        elif days <= 31:
            status = f"{days} 天内"
            cls = "soon"
        else:
            status = f"{days} 天"
            cls = ""
        entry = (days, cls, status, item)
        if days <= 31:
            visible.append(entry)
        else:
            later.append(entry)
    if not visible and later:
        visible = later[:1]
    visible.sort(key=lambda entry: entry[0])
    if not visible:
        return ""
    rows = []
    for _, cls, status, item in visible[:5]:
        rows.append(
            f"""<a class="conf-item {cls}" href="{e(item['url'])}">
  <span><b>{e(item['name'])}</b><em>{e(item['ccf'])} · {e(item['track'])}</em></span>
  <strong>{e(status)}</strong>
  <small>{e(item['deadline'])} · {e(item['note'])}</small>
</a>"""
        )
    return f"""<aside class="conf-float" data-conf-reminder="{e(CURRENT_DATE)}-main-ddl" aria-label="CCF A/B 主投稿提醒">
  <button class="conf-close" type="button" aria-label="关闭会议提醒">×</button>
  <details open>
    <summary>CCF A/B 主投稿提醒</summary>
    <div class="conf-list">{''.join(rows)}</div>
  </details>
</aside>"""


def shell(title: str, body: str, depth: int = 0, active: str = "", display_date: str | None = None) -> str:
    prefix = "../" * depth
    page_date = display_date or CURRENT_DATE
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{e(title)} · 通用视觉自监督研究报</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="{prefix}assets/visual-ssl.css?v={CSS_VERSION}">
</head>
<body>
  <header class="masthead">
    <a class="masthead-title" href="{prefix}index.html">通用视觉自监督研究报</a>
    <div class="masthead-kicker">A Daily Digest of Visual Self-Supervised Learning</div>
    <div class="masthead-meta">
      <span>{e(page_date)}</span>
      <span>图像表征 · VFM · JEPA · 视频预训练</span>
    </div>
  </header>
  <nav class="site-nav" aria-label="主导航">
    {nav(depth, active)}
  </nav>
  <main>{body}</main>
  {conference_widget(depth)}
  <footer class="footer">
    <div>面向通用视觉自监督、视觉基础模型与多模态表征学习的每日研究报。</div>
    <div class="muted">Issue {e(page_date)} · Curated readings and figure notes</div>
  </footer>
  <script>
  (() => {{
    const box = document.querySelector('[data-conf-reminder]');
    if (!box) return;
    const key = `visualSslConfReminderClosed:${{box.dataset.confReminder || '{e(CURRENT_DATE)}-main-ddl'}}`;
    try {{
      if (localStorage.getItem(key) === '1') box.hidden = true;
    }} catch (err) {{}}
    const close = box.querySelector('.conf-close');
    if (close) {{
      close.addEventListener('click', () => {{
        box.hidden = true;
        try {{ localStorage.setItem(key, '1'); }} catch (err) {{}}
      }});
    }}
  }})();
  </script>
  <script src="{prefix}assets/ssl-game.js"></script>
</body>
</html>"""


_READ_CHECK = ('<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.4" '
               'stroke-linecap="round" stroke-linejoin="round"><path d="M4 10.5l4 4 8-9"/></svg>')


def topic_key(category: str) -> str:
    c = (category or "").lower()
    if "jepa" in c:
        return "JEPA"
    if "token" in c:
        return "tokenizer"
    if "video" in c or "motion" in c or "temporal" in c:
        return "video"
    if "3d" in c or "point" in c or "geometr" in c or "pose" in c:
        return "3D"
    if "diffus" in c or "generat" in c:
        return "diffusion"
    if any(k in c for k in ("interpret", "circuit", "diagnos", "probing", "analysis")):
        return "interpretability"
    if "contrast" in c:
        return "contrastive"
    if any(k in c for k in ("vlm", "vision-language", "vision language", "multimodal", "language", "caption")):
        return "VLM"
    return "default"


def read_btn(pid: str) -> str:
    return (f'<button class="readbtn" type="button" data-read-toggle="{e(pid)}" '
            f'title="标记已读" aria-label="标记已读">{_READ_CHECK}</button>')


def paper_card(p: dict, depth: int = 0) -> str:
    prefix = "../" * depth
    figs = figures_for(p["id"], depth, 1)
    if figs:
        thumb = (f'<a class="paper-thumb" href="{prefix}papers/{p["id"]}.html">'
                 f'<img src="{e(figs[0]["src"])}" alt="{e(p["short"])} figure"></a>')
    else:
        thumb = (f'<a class="paper-thumb paper-thumb-text" data-topic="{topic_key(p["category"])}" '
                 f'href="{prefix}papers/{p["id"]}.html">'
                 f'<span>{e(p["priority"])}</span><strong>{e(p["short"])}</strong><em>{e(p["venue"])}</em></a>')
    keywords = " ".join([p["title"], p["short"], p["category"], p["priority"], p["date"], p["venue"], p["method"], p["takeaway"]]).lower()
    return f"""<article class="paper-card" data-pid="{e(p['id'])}" data-category="{e(p['category'])}" data-priority="{e(p['priority'])}" data-date="{e(p['date'])}" data-title="{e(p['title'].lower())}" data-keywords="{e(keywords)}">
  {thumb}
  <div>
    <div class="paper-meta">{e(p['date'])} · {e(p['category'])} · {e(p['venue'])}</div>
    <h3><a href="{prefix}papers/{p['id']}.html">{e(p['title'])}</a></h3>
    <p>{e(p['takeaway'])}</p>
    <p class="card-tech"><b>方法：</b>{e(p['method'])}</p>
    <div class="paper-actions"><a href="{prefix}papers/{p['id']}.html">读图文页</a><a href="{e(p['url'])}">原文</a></div>
  </div>
  {read_btn(p['id'])}
</article>"""


def write_index(report_md: str) -> None:
    summary = md_table_rows(report_md, "快速摘要")
    summary_html = "".join(
        f"<li><b>{e(row[0])}</b><span>{e(row[1])}</span></li>"
        for row in summary[1:]
        if len(row) >= 2 and not any(term in row[0] for term in ("本地归档", "飞书", "同步"))
    )
    hero = PAPERS[0]
    hero_figs = figures_for(hero["id"], 0, 1)
    hero_caption = ""
    if hero_figs:
        hero_caption = hero_figs[0].get("caption") or hero_figs[0].get("caption_zh") or hero_figs[0].get("caption_en") or hero_figs[0].get("label") or ""
    hero_img = f'<figure class="hero-figure"><img src="{e(hero_figs[0]["src"])}" alt="{e(hero["short"])} figure"><figcaption>{e(hero_caption[:180])}</figcaption></figure>' if hero_figs else ""
    cards = "\n".join(paper_card(p, 0) for p in PAPERS[:5])
    today_ids = ",".join(p["id"] for p in PAPERS if p["date"] == CURRENT_DATE) or ",".join(p["id"] for p in PAPERS[:5])
    flame_svg = ('<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0c.7 2.5-1.2 3.7-2 5'
                 '-1 1.6-1 4 .8 4.8C6.4 8.5 7.4 7.8 8 7c.3 1.3-.4 2 .9 3 .9.7 1.3 1.6 1.1 2.6C9.7 14.6 8.9 16 6.8 16'
                 ' 4.6 16 3 14.3 3 12c0-3.3 2.6-4.8 2.4-8C6.9 4.6 8.4 5 8 0z"/></svg>')
    body = f"""<section class="hero-grid">
  <article class="lead-story">
    <div class="kicker">今日主线 · {e(hero['venue'])}</div>
    <h1>{e(hero['short'])}：{e(hero['thesis'].rstrip('。'))}</h1>
    <p class="dek">{e(hero['thesis'])}</p>
    {hero_img}
    <div class="paper-actions"><a href="issues/{CURRENT_DATE}.html">阅读 {zh_date(CURRENT_DATE)}速递</a><a href="papers/{hero['id']}.html">打开主文页</a></div>
  </article>
  <aside class="issue-brief">
    <div class="prog-widget">
      <div class="pw-t">我的进度</div>
      <div class="prog-top">
        <div class="ringwrap" data-ring data-ids="{today_ids}"></div>
        <div class="prog-meta">
          <div class="pm-t">今日精读进度</div>
          <div class="pm-s">累计已读 <b data-readcount>0</b> 篇</div>
          <div class="prog-streak">{flame_svg} 连续追踪 <b data-streak>0</b> 天</div>
        </div>
      </div>
      <div data-heat data-weeks="16"></div>
      <div class="heat-cap"><span>最近 16 周签到</span><span>今天 →</span></div>
      <div class="badges" data-badges></div>
    </div>
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
    issue_papers = [p for p in PAPERS if p["date"] == CURRENT_DATE]
    if issue_papers:
        cards = "\n".join(paper_card(p, 1) for p in issue_papers)
        issue_title = f"{zh_date(CURRENT_DATE)}：{issue_papers[0]['short']} 等视觉表征主线更新"
    else:
        cards = '<p class="empty-state">今天没有通过去重与方向过滤的新论文；目录和首页继续保留最近一期可精读论文，便于回看。</p>'
        issue_title = f"{zh_date(CURRENT_DATE)}：无新增通用视觉自监督论文"
    body = f"""<article class="paper-detail issue-detail">
  <div class="paper-main">
    <div class="kicker">Daily issue · {e(CURRENT_DATE)} · CCF A/B 会议优先</div>
    <h1 class="paper-headline">{e(issue_title)}</h1>
    <p class="dek">本期聚焦通用视觉表征、视觉语言预训练与视频自监督中最值得跟进的论文，并保留 CCF A/B、OpenReview 与 arXiv 状态线索。</p>
    <div class="feature-body">
      <p class="lead dropcap">{e(route)}</p>
      <h2>论文索引</h2>
      {table}
      <h2>主线阅读</h2>
      <div class="paper-list issue-list">{cards}</div>
    </div>
  </div>
  <aside class="paper-side">
    <div class="side-box"><h4>今日重点</h4><p>TextTeacher、MDM、EvoVid、Ablate-to-Validate。</p></div>
    <div class="side-box"><h4>会议口径</h4><p>只把 CCF A/B 放进主提醒；临近截止或 CCF C 仅记录。</p></div>
    <div class="side-box"><h4>阅读重点</h4><p>先看 P1/P2 与方法图，再扫状态变化和边界案例。</p></div>
  </aside>
</article>"""
    (ROOT / "issues" / f"{CURRENT_DATE}.html").write_text(shell(f"{CURRENT_DATE} 速递", body, 1, f"issues/{CURRENT_DATE}.html"), encoding="utf-8")


def write_archive_issues() -> None:
    dates = sorted({p["date"] for p in PAPERS if p["date"] != CURRENT_DATE}, reverse=True)
    for day in dates:
        issue_papers = [p for p in PAPERS if p["date"] == day]
        if not issue_papers:
            continue
        lead = issue_papers[0]
        cards = "\n".join(paper_card(p, 1) for p in issue_papers)
        body = f"""<article class="paper-detail issue-detail">
  <div class="paper-main">
    <div class="kicker">Daily issue · {e(day)} · CCF A/B 会议优先</div>
    <h1 class="paper-headline">{zh_date(day)}：{e(lead['short'])} 等视觉表征论文归档</h1>
    <p class="dek">本页按日期归档近期论文，保留优先级、主题、来源与方法线索，方便回看同一批工作的研究脉络。</p>
    <div class="feature-body">
      <p class="lead dropcap">先读 {e(lead['short'])}。{e(lead['thesis'])} 其余论文按 P1/P2 与主题顺序扫读，重点看方法图、消融设置和是否能迁移到通用视觉表征。</p>
      <h2>论文索引</h2>
      <div class="paper-list issue-list">{cards}</div>
    </div>
  </div>
  <aside class="paper-side">
    <div class="side-box"><h4>阅读重点</h4><p>先看 P1/P2 与方法图，再扫状态变化和边界案例。</p></div>
    <div class="side-box"><h4>会议口径</h4><p>主提醒只保留 CCF A/B；CCF C 与过期截止仅作状态记录。</p></div>
  </aside>
</article>"""
        (ROOT / "issues" / f"{day}.html").write_text(shell(f"{day} 速递", body, 1, f"issues/{day}.html", display_date=day), encoding="utf-8")


def write_paper(p: dict, report_md: str) -> None:
    if p["id"] == "2605.03245":
        (ROOT / "papers" / f"{p['id']}.html").write_text(tc_jepa_body(p), encoding="utf-8")
        return
    fields = detail_fields(report_md, p)
    authors = choose_field(fields, "作者/机构", "作者")
    date_info = choose_field(fields, "发布日期/会议信息", "会议信息")
    method = choose_field(fields, "核心方法") or p["method"]
    contribution = choose_field(fields, "主要贡献") or p["takeaway"]
    experiment = choose_field(fields, "实验亮点")
    limitation = choose_field(fields, "局限性") or "这篇论文的结论需要结合任务设置、训练数据规模和消融实验一起看；不要只凭单个指标判断它对通用视觉表征的价值。"
    relevance = choose_field(
        fields,
        "为什么相关",
        "与通用视觉自监督的关系",
        "为什么与通用视觉自监督方向相关",
        "与通用视觉 SSL 的相关性",
        "与通用视觉 SSL 的关系",
    ) or f"它和通用视觉自监督的关系在于：{p['thesis']}"
    figs = ""
    paper_figs = figures_for(p["id"], 1, 2)
    if paper_figs:
        figs = "\n".join(
            render_fig(fig)
            for fig in paper_figs
        )
    else:
        figs = f"""<aside class="marginalia"><b>图像状态</b> 这篇暂未抽到足够稳定的 pipeline / motivation 图，因此正文以方法和实验解读为主。</aside>"""
    author_line = f"　{e(authors)}" if authors else ""
    date_line = f"　{e(date_info)}" if date_info else f"　{e(p['venue'])}"
    experiment_html = paragraph(experiment) if experiment else "<p>实验部分建议重点看两类证据：一是作者是否把方法收益和更强数据、更长训练、更大模型区分开；二是跨模型、跨数据或跨任务迁移是否还能保留同样趋势。</p>"
    if p["id"] == "2605.21642":
        reading_html = ablate_to_validate_reading_html()
    elif p["id"] == "2605.23045":
        reading_html = time_machine_reading_html(fields, p)
    else:
        reading_html = f"""
      <h2>核心问题</h2>
      <p>{e(relevance)}</p>
      <h2>方法拆解</h2>
      <p>{e(method)}</p>
      <h2>主要贡献</h2>
      <p>{e(contribution)}</p>
      <h2>实验看点</h2>
      {experiment_html}
      <h2>局限与读法</h2>
      <p>{e(limitation)}</p>
    """
    headline_thesis = first_sentence(relevance) or p["thesis"]
    dek = first_sentence(contribution, 160) or p["takeaway"]
    lead = unique_sentence_join(headline_thesis, dek)
    article_class = "paper-detail deep-read" if p["id"] == "2605.21642" else "paper-detail"
    side_intro = ""
    if p["id"] == "2605.21642":
        side_intro = """    <div class="side-box"><h4>图表导读</h4><p>先看 Figure 1 的替换协议，再看 Table 1-4 和 Mirage / CoVT 的替换实验。</p></div>
"""
    body = f"""<article class="{article_class}">
  <div class="paper-main">
    <div class="kicker">{e(p['category'])} · {e(p['priority'])} · {e(p['date'])}</div>
    <h1 class="paper-headline">{e(p['short'])}：{e(headline_thesis.rstrip('。'))}</h1>
    <p class="dek">{e(dek)}</p>
    <p class="byline"><strong>{e(p['title'])}</strong>{author_line}{date_line}　<a href="{e(p['url'])}">原文链接</a></p>
    <div class="paper-facts">
      <span><b>编号</b>{e(p['id'])}</span>
      <span><b>优先级</b>{e(p['priority'])}</span>
      <span><b>类别</b>{e(p['category'])}</span>
      <span><b>会议</b>{e(p['venue'])}</span>
      <span><b>方法</b>{e(method)}</span>
      <span><b>来源</b>arXiv / OpenReview</span>
    </div>
    <div class="feature-body">
      <p class="lead dropcap">先说结论。{e(lead)}</p>
      {figs}
      {reading_html}
    </div>
  </div>
  <aside class="paper-side">
{side_intro}
    <div class="side-box"><h4>关键判断</h4><dl><dt>相关性</dt><dd>{e(p['priority'])}</dd><dt>类别</dt><dd>{e(p['category'])}</dd><dt>会议</dt><dd>{e(p['venue'])}</dd><dt>读法</dt><dd>方法图优先，表格次之</dd></dl></div>
    <div class="side-box"><h4>原文</h4><p><a href="{e(p['url'])}">{e(p['url'])}</a></p></div>
  </aside>
</article>"""
    (ROOT / "papers" / f"{p['id']}.html").write_text(shell(p["short"], body, 1, display_date=p["date"]), encoding="utf-8")


def write_catalog() -> None:
    cards = "\n".join(paper_card(p, 1) for p in PAPERS)
    categories = sorted({p["category"] for p in PAPERS})
    dates = sorted({p["date"] for p in PAPERS}, reverse=True)
    priority_order = ["P0", "P1", "P2", "P3", "扫读", "状态补录"]
    priorities = [value for value in priority_order if any(p["priority"] == value for p in PAPERS)]

    def filter_buttons(values: list[str], group: str, all_label: str) -> str:
        buttons = [f'<button class="filter-pill is-active" type="button" data-filter-value="all" aria-pressed="true">{e(all_label)}</button>']
        buttons.extend(
            f'<button class="filter-pill" type="button" data-filter-value="{e(value)}" aria-pressed="false">{e(value)}</button>'
            for value in values
        )
        return f'<div class="filter-row" data-filter-group="{e(group)}">' + "".join(buttons) + "</div>"

    catalog_script = """
<script>
(() => {
  const root = document.querySelector('[data-catalog]');
  if (!root) return;
  const cards = Array.from(root.querySelectorAll('.paper-card'));
  const search = root.querySelector('[data-catalog-search]');
  const status = root.querySelector('[data-catalog-status]');
  const pager = root.querySelector('[data-catalog-pager]');
  const pageInfo = root.querySelector('[data-page-info]');
  const prev = root.querySelector('[data-page-prev]');
  const next = root.querySelector('[data-page-next]');
  const state = { category: 'all', priority: 'all', date: 'all', query: '' };
  const pageSize = 24;
  let page = 1;

  const setCardVisible = (card, visible) => {
    card.hidden = !visible;
    card.style.display = visible ? '' : 'none';
  };

  const matchCard = (card) => {
    if (state.category !== 'all' && card.dataset.category !== state.category) return false;
    if (state.priority !== 'all' && card.dataset.priority !== state.priority) return false;
    if (state.date !== 'all' && card.dataset.date !== state.date) return false;
    if (state.query && !(card.dataset.keywords || '').includes(state.query)) return false;
    return true;
  };

  const render = () => {
    const matched = cards.filter(matchCard);
    const totalPages = Math.max(1, Math.ceil(matched.length / pageSize));
    if (page > totalPages) page = totalPages;
    const visible = new Set(matched.slice((page - 1) * pageSize, page * pageSize));
    cards.forEach((card) => { setCardVisible(card, visible.has(card)); });
    status.textContent = `${matched.length} 篇 · 第 ${page}/${totalPages} 页`;
    pageInfo.textContent = `${page} / ${totalPages}`;
    pager.hidden = matched.length <= pageSize;
    prev.disabled = page <= 1;
    next.disabled = page >= totalPages;
  };

  root.addEventListener('click', (event) => {
    const button = event.target.closest('[data-filter-value]');
    if (!button) return;
    const group = button.closest('[data-filter-group]');
    if (!group) return;
    state[group.dataset.filterGroup] = button.dataset.filterValue;
    group.querySelectorAll('.filter-pill').forEach((item) => {
      const active = item === button;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    page = 1;
    render();
  });

  search.addEventListener('input', () => {
    state.query = search.value.trim().toLowerCase();
    page = 1;
    render();
  });
  prev.addEventListener('click', () => { page -= 1; render(); });
  next.addEventListener('click', () => { page += 1; render(); });
  render();
})();
</script>"""
    body = f"""<section class="page-header"><h1>论文目录</h1><p>收录近期通用视觉自监督、视觉基础模型与多模态表征学习论文，按日期、优先级与主题归档。</p></section>
<section class="catalog-shell" data-catalog>
  <div class="catalog-tools">
    <input class="catalog-search" type="search" data-catalog-search placeholder="搜索标题、主题、方法" aria-label="搜索论文">
    {filter_buttons(categories, "category", "全部主题")}
    {filter_buttons(priorities, "priority", "全部优先级")}
    {filter_buttons(dates, "date", "全部日期")}
    <div class="catalog-status" data-catalog-status></div>
  </div>
  <div class="paper-list catalog-list">{cards}</div>
  <nav class="catalog-pager" data-catalog-pager aria-label="目录分页">
    <button type="button" data-page-prev>上一页</button>
    <span data-page-info></span>
    <button type="button" data-page-next>下一页</button>
  </nav>
</section>
{catalog_script}"""
    (ROOT / "pages" / "catalog.html").write_text(shell("论文目录", body, 1, "pages/catalog.html"), encoding="utf-8")


def write_timeline() -> None:
    grouped: dict[str, list[dict]] = {}
    for p in PAPERS:
        grouped.setdefault(p["date"], []).append(p)
    parts = []
    for day in sorted(grouped, reverse=True):
        items = grouped[day]
        day_ids = ",".join(x["id"] for x in items)
        rows = "\n".join(
            f'<article data-pid="{e(x["id"])}"><span>{e(x["priority"])}</span>'
            f'<a href="../papers/{x["id"]}.html">{e(x["short"])}</a>'
            f'<em>{e(x["category"])}</em></article>'
            for x in items
        )
        parts.append(
            f'<section class="timeline-day" data-dayids="{e(day_ids)}">'
            f'<div class="tl-daycol"><h2>{e(zh_date(day))}</h2>'
            f'<span class="tl-done" data-day-done hidden></span></div>'
            f'<div class="tl-items">{rows}</div></section>'
        )
    body = ('<section class="page-header"><h1>时间线</h1>'
            '<p>每日刊物按时间倒序归档；已读的论文会在这里同步标记。</p></section>'
            + "\n".join(parts))
    (ROOT / "pages" / "timeline.html").write_text(shell("时间线", body, 1, "pages/timeline.html"), encoding="utf-8")


def write_css() -> None:
    css = r"""
/* ============================================================
   通用视觉自监督研究报 · 设计 v6 · 极简学术博客 + 养成系
   自包含样式（shell 不再加载旧 styles.css）。
   暖纸白 + 一种低饱和钢蓝 accent；养成元素用暖黄铜。
   英文固有名词加重、中文做解释；图按原始比例不裁切。
   ============================================================ */
:root{
  --paper:#fcfbf8;--paper2:#f5f3ed;--card:#ffffff;
  --ink:#1b1a17;--ink-soft:#3c3933;--muted:#6f6a61;--muted-2:#9a9488;
  --hair:#e7e3da;--hair2:#efece4;--rule:#d3cdc1;
  --accent:#355d86;--accent-deep:#274964;--accent-soft:#e6edf4;--accent-line:rgba(53,93,134,.28);
  --gold:#b07d3a;--gold-deep:#8a5d24;--gold-soft:#f4ead7;--alert:#b0492f;
  --serif:"Source Serif 4","Songti SC","Noto Serif SC",Georgia,"Times New Roman",serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif;
  --mono:"JetBrains Mono","SF Mono",ui-monospace,Consolas,monospace;
  --page:min(94vw,1480px);--read:46rem;
}
*{box-sizing:border-box;}
html{-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);font-size:18px;line-height:1.72;font-feature-settings:"kern","liga";}
img{max-width:100%;height:auto;display:block;}
::selection{background:var(--accent-soft);color:var(--ink);}
a{color:var(--accent-deep);text-decoration:none;}
h1,h2,h3,h4{font-family:var(--serif);font-weight:600;color:var(--ink);text-wrap:balance;word-break:keep-all;}
p{text-wrap:pretty;}

/* ---------- masthead 报名牌 ---------- */
.masthead{max-width:var(--page);margin:0 auto;padding:30px 28px 0;display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;}
.masthead-title{order:1;font-family:var(--serif);font-weight:600;font-size:clamp(22px,2.3vw,28px);letter-spacing:-.015em;color:var(--ink);text-decoration:none;display:block;}
.masthead-title:hover{color:var(--accent-deep);}
.masthead-kicker{order:0;flex-basis:100%;font-family:var(--sans);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted-2);margin-bottom:5px;}
.masthead-meta{order:2;text-align:right;display:flex;flex-direction:column;gap:2px;font-family:var(--sans);font-size:11.5px;color:var(--muted-2);}
.masthead-meta span{white-space:nowrap;}
.masthead-meta span:first-child{font-family:var(--mono);letter-spacing:.04em;}

/* ---------- nav ---------- */
.site-nav{max-width:var(--page);margin:16px auto 0;padding:0 28px;display:flex;gap:24px;align-items:center;border-bottom:1px solid var(--hair);font-family:var(--sans);font-size:14px;flex-wrap:wrap;}
.nav-link{color:var(--muted);text-decoration:none;padding:11px 0;border-bottom:2px solid transparent;margin-bottom:-1px;transition:color .15s;}
.nav-link:hover{color:var(--ink);}
.nav-link.active{color:var(--ink);border-bottom-color:var(--accent);}
.nav-streak{margin-left:auto;display:inline-flex;align-items:center;gap:6px;font-family:var(--sans);font-size:12.5px;font-weight:600;color:var(--gold-deep);background:var(--gold-soft);border:1px solid #e7d3ad;border-radius:30px;padding:4px 11px 4px 9px;white-space:nowrap;}
.nav-streak svg{width:13px;height:13px;}
.nav-streak b{font-variant-numeric:tabular-nums;}

main{display:block;}

/* ---------- kicker / headline / dek ---------- */
.kicker{font-family:var(--sans);font-weight:600;font-size:11.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--accent);margin:0 0 12px;display:block;}
.dek{font-family:var(--serif);font-size:20px;line-height:1.55;color:var(--ink-soft);margin:0 0 22px;max-width:60ch;}

/* ============================================================
   HOME · hero + brief（两栏）
   ============================================================ */
.hero-grid{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:clamp(40px,5vw,72px);align-items:start;max-width:var(--page);margin:34px auto 8px;padding:0 28px;}
@media(max-width:900px){.hero-grid{grid-template-columns:1fr;gap:24px;}}
.lead-story{min-width:0;}
.lead-story h1{font-size:clamp(32px,3.4vw,46px);line-height:1.12;letter-spacing:-.02em;margin:0 0 16px;font-family:var(--serif);}
.lead-story .dek{font-size:20px;}
.hero-figure{margin:24px 0 18px;border:1px solid var(--hair);border-radius:8px;overflow:hidden;background:var(--card);}
.hero-figure img{width:100%;max-height:460px;object-fit:contain;background:#fff;}
.hero-figure figcaption{font-family:var(--sans);font-size:13px;color:var(--muted);line-height:1.6;padding:10px 14px;border-top:1px solid var(--hair);}
.paper-actions{display:flex;gap:10px;flex-wrap:wrap;align-items:center;font-family:var(--sans);margin-top:16px;}
.paper-actions a{font-size:13.5px;font-weight:500;color:var(--ink);text-decoration:none;border:1px solid var(--rule);border-radius:8px;padding:8px 15px;white-space:nowrap;transition:background .15s,border-color .15s,transform .15s;}
.paper-actions a:hover{background:var(--paper2);border-color:var(--ink-soft);}
.paper-actions a:first-child{background:var(--ink);color:var(--paper);border-color:var(--ink);}
.paper-actions a:first-child:hover{background:var(--accent-deep);border-color:var(--accent-deep);transform:translateY(-1px);}

/* 右栏 brief + 进度 */
.issue-brief{font-family:var(--sans);position:sticky;top:22px;}
.issue-brief h2{font-family:var(--sans);font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:600;margin:0 0 12px;}
.issue-brief ul{list-style:none;margin:0 0 4px;padding:0;}
.issue-brief li{padding:0 0 12px;margin-bottom:12px;border-bottom:1px solid var(--hair);}
.issue-brief li b{display:block;font-size:12px;color:var(--ink);font-weight:600;margin-bottom:3px;}
.issue-brief li span{display:block;font-size:13px;line-height:1.55;color:var(--ink-soft);}

/* 我的进度 widget */
.prog-widget{font-family:var(--sans);border-top:2px solid var(--ink);padding-top:16px;margin-top:4px;}
.prog-widget .pw-t{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:600;margin-bottom:14px;}
.prog-top{display:flex;align-items:center;gap:14px;margin-bottom:16px;}
.ringwrap{position:relative;width:64px;height:64px;flex:none;}
.ringsvg{width:64px;height:64px;transform:rotate(-90deg);}
.ring-bg{fill:none;stroke:var(--hair);stroke-width:6;}
.ring-fg{fill:none;stroke:var(--gold);stroke-width:6;stroke-linecap:round;transition:stroke-dashoffset .7s cubic-bezier(.4,0,.2,1);}
.ring-num{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;line-height:1;}
.ring-num b{font-family:var(--serif);font-size:20px;font-weight:600;color:var(--ink);}
.ring-num span{font-size:10px;color:var(--muted-2);margin-top:1px;}
.prog-meta .pm-t{font-size:13px;font-weight:600;color:var(--ink);}
.prog-meta .pm-s{font-size:12px;color:var(--muted);line-height:1.5;margin-top:3px;}
.prog-streak{display:inline-flex;align-items:center;gap:5px;font-size:12.5px;color:var(--gold-deep);font-weight:600;margin-top:6px;}
.prog-streak svg{width:12px;height:12px;}
.heat-grid{display:flex;gap:3px;}
.heat-col{display:flex;flex-direction:column;gap:3px;}
.heat-cell{width:11px;height:11px;border-radius:2.5px;background:var(--hair2);border:1px solid rgba(0,0,0,.03);}
.heat-cell.on{background:var(--gold);}
.heat-cell.fut{opacity:.35;}
.heat-cap{font-size:11px;color:var(--muted-2);margin-top:8px;display:flex;justify-content:space-between;}
.heat-cap span{white-space:nowrap;}
.badges{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;}
.badge{width:38px;height:38px;border-radius:50%;display:grid;place-items:center;font-family:var(--sans);font-weight:700;font-size:12px;color:var(--muted-2);background:var(--paper2);border:1.5px dashed var(--rule);}
.badge.on{color:#fff;background:radial-gradient(circle at 32% 28%,#cd9a55,var(--gold-deep));border:1.5px solid var(--gold-deep);box-shadow:0 2px 8px rgba(176,125,58,.28);}

/* ---------- section ---------- */
.section-block{max-width:var(--page);margin:0 auto;padding:18px 28px 44px;}
.section-heading{display:flex;align-items:baseline;justify-content:space-between;border-top:2px solid var(--ink);padding-top:14px;margin-bottom:6px;}
.section-heading h2{margin:0;font-family:var(--sans);font-weight:600;font-size:15px;letter-spacing:.02em;}
.section-heading a{font-family:var(--sans);font-size:13px;color:var(--muted);}
.paper-list{display:flex;flex-direction:column;}

/* ---------- paper-card（横向条目 + 缩略图 + 已读勾） ---------- */
.paper-card{position:relative;display:grid;grid-template-columns:148px minmax(0,1fr) auto;gap:18px;align-items:start;padding:18px 0;border-top:1px solid var(--hair);}
.paper-card .paper-thumb{grid-column:1;width:148px;height:96px;border-radius:8px;overflow:hidden;border:1px solid var(--hair);background:var(--paper2);text-decoration:none;display:block;}
.paper-card .paper-thumb img{width:100%;height:100%;object-fit:cover;background:#fff;}
.paper-card > div{grid-column:2;min-width:0;}
.paper-meta{font-family:var(--sans);font-size:12px;color:var(--muted);margin-bottom:5px;}
.paper-card h3{margin:0 0 6px;font-family:var(--serif);font-size:20px;line-height:1.28;font-weight:600;}
.paper-card h3 a{color:var(--ink);text-decoration:none;}
.paper-card:hover h3 a{text-decoration:underline;text-decoration-color:var(--accent-line);text-underline-offset:3px;}
.paper-card p{margin:0 0 6px;font-size:16px;line-height:1.6;color:var(--ink-soft);}
.paper-card .card-tech{font-family:var(--sans);font-size:12.5px;color:var(--muted);}
.paper-card .card-tech b{color:var(--ink-soft);}
.paper-card .paper-actions{margin-top:10px;}
.paper-card .paper-actions a{font-size:12.5px;padding:6px 12px;}
.paper-card .paper-actions a:first-child{background:var(--card);color:var(--ink);border-color:var(--rule);}
.paper-card .paper-actions a:first-child:hover{background:var(--paper2);}
.paper-card .readbtn{grid-column:3;}

/* 主题封面（无图时） */
.paper-thumb-text{position:relative;display:flex!important;flex-direction:column;justify-content:space-between;padding:11px 12px!important;color:#fff!important;background:linear-gradient(135deg,var(--c1,#3a4a63),var(--c2,#26303f))!important;border:0!important;}
.paper-thumb-text::after{content:"";position:absolute;inset:0;opacity:.14;mix-blend-mode:overlay;background-image:repeating-linear-gradient(45deg,#fff 0 1px,transparent 1px 9px);}
.paper-thumb-text strong{position:relative;font-family:var(--sans);font-weight:700;font-size:15px;line-height:1.1;letter-spacing:.04em;}
.paper-thumb-text span{position:relative;align-self:flex-start;order:-1;font-family:var(--sans);font-size:10.5px;font-weight:700;background:rgba(255,255,255,.22);border-radius:20px;padding:2px 8px;}
.paper-thumb-text em{position:relative;font-family:var(--sans);font-size:10px;font-style:normal;opacity:.85;}
[data-topic="JEPA"]{--c1:#3f4d8f;--c2:#272f5c;}
[data-topic="tokenizer"]{--c1:#1f7a73;--c2:#124b46;}
[data-topic="VLM"]{--c1:#6d4a86;--c2:#412a54;}
[data-topic="video"]{--c1:#b1565a;--c2:#6f2f34;}
[data-topic="3D"]{--c1:#9a6b2f;--c2:#5e3f18;}
[data-topic="diffusion"]{--c1:#3273a3;--c2:#1d4868;}
[data-topic="interpretability"]{--c1:#3f7d54;--c2:#244a31;}
[data-topic="contrastive"]{--c1:#a35f33;--c2:#643619;}
[data-topic="default"]{--c1:#4a5568;--c2:#2c333f;}

/* 已读态 */
.paper-card.is-read{opacity:.6;}
.paper-card.is-read h3 a{text-decoration:line-through;text-decoration-color:var(--rule);}

/* readbtn 通用 */
.readbtn{flex:none;width:30px;height:30px;border-radius:50%;cursor:pointer;border:1.5px solid var(--rule);background:var(--card);color:var(--muted-2);display:grid;place-items:center;transition:all .15s;padding:0;}
.readbtn svg{width:15px;height:15px;}
.readbtn:hover{border-color:var(--gold);color:var(--gold-deep);}
.readbtn.on{background:var(--gold);border-color:var(--gold);color:#fff;}
.readbtn.pop{animation:pop .32s ease;}
@keyframes pop{0%{transform:scale(.7);}55%{transform:scale(1.18);}100%{transform:scale(1);}}

/* ============================================================
   PAPER DETAIL · 单栏阅读 + 破栏图
   ============================================================ */
.paper-detail{max-width:var(--page);margin:0 auto;padding:30px 28px 50px;display:grid;grid-template-columns:minmax(0,1fr) 232px;gap:clamp(28px,4vw,56px);align-items:start;}
@media(max-width:1040px){.paper-detail{grid-template-columns:1fr;}}
.paper-main{min-width:0;max-width:var(--read);margin:0 auto;}
.paper-headline{font-size:clamp(28px,3vw,40px);line-height:1.16;letter-spacing:-.015em;margin:0 0 16px;}
.byline{font-family:var(--sans);font-size:13.5px;color:var(--muted);line-height:1.6;margin:0 0 18px;}
.byline strong{color:var(--ink-soft);font-weight:600;}
.paper-facts{display:flex;flex-wrap:wrap;gap:8px 22px;border-top:1px solid var(--hair);border-bottom:1px solid var(--hair);padding:12px 0;margin:0 0 30px;font-family:var(--sans);font-size:12.5px;color:var(--muted);}
.paper-facts span b{color:var(--ink-soft);font-weight:600;margin-right:6px;}
.feature-body{font-size:18px;}
.feature-body h2{font-size:25px;line-height:1.25;margin:42px 0 14px;letter-spacing:-.01em;}
.feature-body h2::before{content:"";display:block;width:26px;height:2px;background:var(--accent);margin-bottom:14px;}
.feature-body h3{font-size:19px;margin:26px 0 8px;}
.feature-body p{margin:0 0 18px;}
.feature-body strong{font-weight:640;color:var(--ink);}
.lead{font-size:19.5px;line-height:1.7;color:var(--ink);}
.lead.dropcap::first-letter{font-family:var(--serif);font-weight:600;float:left;font-size:3.5em;line-height:.86;padding:6px 10px 0 0;margin:4px 2px 0 0;color:var(--accent);}

/* 破栏图：图按原始比例，绝不裁切 */
.full-fig{margin:30px 0;max-width:none;}
.full-fig img{width:100%;max-height:760px;object-fit:contain;background:#fff;border:1px solid var(--hair);border-radius:6px;}
.full-fig figcaption{font-family:var(--sans);margin-top:12px;}
.full-fig figcaption b{display:block;font-weight:600;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent-deep);margin-bottom:7px;}
.full-fig .fig-en,.full-fig .fig-zh{display:block;line-height:1.6;}
.full-fig .fig-zh{font-size:14.5px;color:var(--ink-soft);}
.full-fig .fig-en{font-family:var(--serif);font-style:italic;font-size:13.5px;color:var(--muted);margin-top:7px;padding-top:7px;border-top:1px dashed var(--hair);}

.reading-note{background:var(--paper2);border:1px solid var(--hair);border-radius:10px;padding:18px 22px;margin:26px 0;}
.reading-note h2{font-family:var(--sans);font-size:16px;font-weight:600;margin:0 0 8px;letter-spacing:.02em;}
.reading-note h2::before{display:none;}
.reading-note p{font-size:15.5px;line-height:1.66;margin:0 0 10px;color:var(--ink-soft);}
.reading-note p:last-child{margin:0;}
.method-steps{margin:18px 0;padding-left:1.4rem;}
.method-steps li{margin:10px 0;line-height:1.62;}
.method-steps li::marker{color:var(--accent);font-weight:600;}
.method-steps b{font-weight:640;}
.stat-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:1px;background:var(--hair);border:1px solid var(--hair);border-radius:10px;overflow:hidden;margin:22px 0;}
.stat-grid div{background:var(--card);padding:16px 18px;}
.stat-grid b{display:block;font-family:var(--sans);font-size:11.5px;letter-spacing:.04em;color:var(--accent-deep);text-transform:uppercase;font-weight:600;margin-bottom:6px;}
.stat-grid span{display:block;font-family:var(--mono);font-size:15px;color:var(--ink);line-height:1.4;}
.stat-grid em{display:block;font-style:normal;font-size:13px;color:var(--muted);margin-top:5px;line-height:1.5;}
@media(max-width:640px){.stat-grid{grid-template-columns:1fr;}}
.marginalia{font-family:var(--sans);font-size:13px;color:var(--ink-soft);background:var(--gold-soft);border:1px solid #e7d3ad;border-radius:10px;padding:14px 16px;margin:24px 0;}
.marginalia b{color:var(--gold-deep);}
.compact-table{width:100%;border-collapse:collapse;font-family:var(--sans);font-size:13px;margin:14px 0 8px;}
.compact-table th,.compact-table td{border:1px solid var(--hair);padding:8px 10px;text-align:left;vertical-align:top;}
.compact-table th{background:var(--paper2);font-weight:600;color:var(--ink);}
.compact-table td{color:var(--ink-soft);}
.issue-list{display:flex;flex-direction:column;}
.empty-state{font-family:var(--sans);color:var(--muted);font-size:14px;border:1px dashed var(--rule);border-radius:10px;padding:24px;text-align:center;}

/* 右侧 side-box */
.paper-side{font-family:var(--sans);position:sticky;top:22px;display:flex;flex-direction:column;gap:16px;}
@media(max-width:1040px){.paper-side{position:static;}}
.side-box{border:1px solid var(--hair);border-radius:10px;background:var(--card);padding:14px 16px;}
.side-box h4{font-family:var(--sans);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted-2);font-weight:600;margin:0 0 8px;}
.side-box p{font-size:13px;line-height:1.6;color:var(--ink-soft);margin:0;}
.side-box a{color:var(--accent-deep);word-break:break-all;}
.side-box dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:6px 12px;font-size:12.5px;}
.side-box dt{color:var(--muted);}
.side-box dd{margin:0;color:var(--ink);font-weight:500;}

/* ============================================================
   CATALOG
   ============================================================ */
.page-header{max-width:var(--page);margin:30px auto 0;padding:0 28px;}
.page-header h1{font-size:34px;margin:0 0 6px;}
.page-header p{font-family:var(--sans);font-size:14.5px;color:var(--muted);margin:0;}
.catalog-shell{max-width:var(--page);margin:18px auto 0;padding:0 28px 60px;display:grid;grid-template-columns:248px minmax(0,1fr);gap:clamp(28px,3.5vw,52px);align-items:start;}
@media(max-width:880px){.catalog-shell{grid-template-columns:1fr;}}
.catalog-tools{font-family:var(--sans);position:sticky;top:22px;display:flex;flex-direction:column;gap:18px;}
@media(max-width:880px){.catalog-tools{position:static;}}
.catalog-search{width:100%;font-family:var(--sans);font-size:14px;color:var(--ink);border:1px solid var(--rule);border-radius:9px;padding:10px 12px;background:var(--card);}
.catalog-search:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft);}
.filter-row{display:flex;flex-wrap:wrap;gap:6px;}
.filter-pill{font-family:var(--sans);font-size:12.5px;color:var(--ink-soft);cursor:pointer;background:var(--card);border:1px solid var(--hair);border-radius:30px;padding:5px 11px;transition:all .14s;white-space:nowrap;}
.filter-pill:hover{border-color:var(--rule);}
.filter-pill.is-active{background:var(--ink);color:var(--paper);border-color:var(--ink);}
.catalog-status{font-family:var(--sans);font-size:13px;color:var(--muted);}
.catalog-list{margin-top:0;}
.catalog-pager{display:flex;align-items:center;justify-content:center;gap:14px;margin-top:24px;font-family:var(--sans);grid-column:2;}
@media(max-width:880px){.catalog-pager{grid-column:1;}}
.catalog-pager button{border:1px solid var(--rule);border-radius:8px;background:var(--card);color:var(--ink);padding:8px 14px;cursor:pointer;font-size:13px;}
.catalog-pager button:disabled{opacity:.4;cursor:not-allowed;}
.catalog-pager span{font-family:var(--mono);color:var(--muted);font-size:13px;}

/* ============================================================
   TIMELINE
   ============================================================ */
.timeline-day{max-width:var(--page);margin:0 auto;padding:24px 28px;border-top:1px solid var(--hair);display:grid;grid-template-columns:132px minmax(0,1fr);gap:clamp(16px,3vw,40px);}
.timeline-day:first-of-type{border-top:0;}
.timeline-day h2{position:sticky;top:18px;align-self:start;font-family:var(--serif);font-size:20px;font-weight:600;margin:0;color:var(--ink);white-space:nowrap;}
.tl-daycol{position:sticky;top:18px;align-self:start;}
.tl-daycol h2{position:static;}
.tl-done{display:inline-flex;margin-top:8px;font-family:var(--sans);font-size:11.5px;font-weight:600;color:var(--gold-deep);background:var(--gold-soft);border:1px solid #e7d3ad;border-radius:20px;padding:2px 9px;}
.timeline-day > div,.timeline-day .tl-items{display:flex;flex-direction:column;}
.timeline-day article{display:grid;grid-template-columns:auto 1fr auto;gap:6px 14px;align-items:center;padding:12px 0;border-top:1px solid var(--hair2);text-decoration:none;}
.timeline-day article:first-child{border-top:0;}
.timeline-day span{font-family:var(--sans);font-size:11.5px;font-weight:600;color:var(--accent-deep);background:var(--accent-soft);border-radius:20px;padding:3px 9px;white-space:nowrap;}
.timeline-day a{font-family:var(--serif);font-size:17px;font-weight:600;color:var(--ink);text-decoration:none;}
.timeline-day a:hover{text-decoration:underline;text-decoration-color:var(--accent-line);text-underline-offset:3px;}
.timeline-day em{font-family:var(--sans);font-size:12px;font-style:normal;color:var(--muted);}
@media(max-width:620px){.timeline-day{grid-template-columns:1fr;gap:12px;}.timeline-day h2{position:static;}.timeline-day article{grid-template-columns:auto 1fr;}.timeline-day em{grid-column:2;}}

/* ============================================================
   CCF DDL 浮窗（极简化）
   ============================================================ */
.conf-float{position:fixed;right:20px;bottom:20px;z-index:40;width:min(300px,calc(100vw - 36px));background:var(--card);border:1px solid var(--hair);border-left:3px solid var(--alert);border-radius:10px;box-shadow:0 8px 28px rgba(27,26,23,.12);padding:13px 38px 13px 15px;font-family:var(--sans);}
.conf-float details{margin:0;}
.conf-float summary{cursor:pointer;font-weight:600;font-size:12.5px;color:var(--ink);list-style:none;}
.conf-float summary::-webkit-details-marker{display:none;}
.conf-close{position:absolute;right:9px;top:9px;width:24px;height:24px;border:0;background:none;color:var(--muted-2);font-size:17px;line-height:1;cursor:pointer;}
.conf-close:hover{color:var(--ink);}
.conf-list{display:grid;gap:8px;margin-top:10px;}
.conf-item{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:3px 10px;text-decoration:none;color:var(--ink);}
.conf-item b{font-size:12.5px;color:var(--ink);font-weight:600;}
.conf-item em{font-style:normal;font-size:11px;color:var(--muted-2);}
.conf-item strong{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--alert);white-space:nowrap;}
.conf-item small{grid-column:1/-1;color:var(--muted);font-size:11px;line-height:1.4;}
.conf-item.urgent strong{color:var(--alert);}
.conf-item.past{opacity:.6;}

/* ---------- footer ---------- */
.footer{max-width:var(--page);margin:60px auto 0;padding:22px 28px 50px;border-top:1px solid var(--hair);font-family:var(--sans);font-size:12.5px;color:var(--muted);line-height:1.7;}
.footer .muted{color:var(--muted-2);}

@media(prefers-reduced-motion:reduce){.ring-fg{transition:none;}.readbtn.pop{animation:none;}}
"""
    (ROOT / "assets" / "visual-ssl.css").write_text(css, encoding="utf-8")


def write_js() -> None:
    js = r"""
/* ssl-game.js · 养成系：连续追踪 streak / 进度环 / 标记已读 / 里程碑 / 签到热力图
   仅使用自己的 localStorage 键：sslReadIds, sslVisitDays。 */
(function(){
  var RKEY='sslReadIds', VKEY='sslVisitDays';
  function load(k){ try{ return JSON.parse(localStorage.getItem(k)||'[]'); }catch(e){ return []; } }
  function save(k,v){ try{ localStorage.setItem(k,JSON.stringify(v)); }catch(e){} }
  function todayStr(d){ d=d||new Date(); return d.toISOString().slice(0,10); }

  var readSet=load(RKEY), visits=load(VKEY);
  (function visit(){ var t=todayStr(); if(visits.indexOf(t)===-1){ visits.push(t); save(VKEY,visits); } })();

  function streak(){
    var set={}; visits.forEach(function(d){ set[d]=1; });
    var n=0, d=new Date();
    while(set[todayStr(d)]){ n++; d.setDate(d.getDate()-1); }
    return n;
  }
  var listeners=[];
  function emit(){ listeners.forEach(function(fn){ try{ fn(); }catch(e){} }); }
  var Game={
    isRead:function(id){ return readSet.indexOf(id)!==-1; },
    toggleRead:function(id){ var i=readSet.indexOf(id); if(i===-1) readSet.push(id); else readSet.splice(i,1); save(RKEY,readSet); emit(); return this.isRead(id); },
    readCount:function(){ return readSet.length; },
    streak:streak,
    onChange:function(fn){ listeners.push(fn); }
  };
  window.SSLGame=Game;

  function ring(el){
    var total=+el.dataset.total||10, done=0;
    if(el.dataset.ids){ var ids=el.dataset.ids.split(',').filter(Boolean); done=ids.filter(function(id){return readSet.indexOf(id)!==-1;}).length; total=ids.length||1; }
    else { done=Math.min(readSet.length,total); }
    var pct=total?done/total:0, r=26, C=2*Math.PI*r;
    el.innerHTML='<svg viewBox="0 0 64 64" class="ringsvg"><circle cx="32" cy="32" r="'+r+'" class="ring-bg"></circle><circle cx="32" cy="32" r="'+r+'" class="ring-fg" stroke-dasharray="'+C+'" stroke-dashoffset="'+(C*(1-pct))+'"></circle></svg><div class="ring-num"><b>'+done+'</b><span>/'+total+'</span></div>';
  }
  var MILE=[10,50,100,250,500];
  function badges(el){ var c=readSet.length; el.innerHTML=MILE.map(function(m){ return '<span class="badge'+(c>=m?' on':'')+'" title="累计精读 '+m+' 篇">'+m+'</span>'; }).join(''); }
  function heat(el){
    var set={}; visits.forEach(function(d){ set[d]=1; });
    var weeks=+el.dataset.weeks||16, today=new Date();
    var start=new Date(today); start.setDate(start.getDate()-(weeks*7-1));
    var dow=(start.getDay()+6)%7; start.setDate(start.getDate()-dow);
    var html='<div class="heat-grid">';
    for(var w=0;w<weeks;w++){ html+='<div class="heat-col">';
      for(var dd=0;dd<7;dd++){ var d=new Date(start); d.setDate(d.getDate()+w*7+dd); var fut=d>today; html+='<i class="heat-cell'+(set[todayStr(d)]?' on':'')+(fut?' fut':'')+'" title="'+todayStr(d)+'"></i>'; }
      html+='</div>'; }
    html+='</div>'; el.innerHTML=html;
  }
  function renderAll(){
    document.querySelectorAll('[data-ring]').forEach(ring);
    document.querySelectorAll('[data-streak]').forEach(function(el){ el.textContent=streak(); });
    document.querySelectorAll('[data-readcount]').forEach(function(el){ el.textContent=readSet.length; });
    document.querySelectorAll('[data-badges]').forEach(badges);
    document.querySelectorAll('[data-heat]').forEach(heat);
    document.querySelectorAll('[data-pid]').forEach(function(node){
      var r=Game.isRead(node.dataset.pid);
      node.classList.toggle('is-read', r);
      var b=node.querySelector('.readbtn'); if(b) b.classList.toggle('on', r);
    });
    document.querySelectorAll('[data-dayids]').forEach(function(el){
      var ids=(el.dataset.dayids||'').split(',').filter(Boolean);
      var read=ids.filter(function(id){return Game.isRead(id);}).length;
      var badge=el.querySelector('[data-day-done]');
      if(badge){ if(read){ badge.hidden=false; badge.textContent='已读 '+read+'/'+ids.length; } else { badge.hidden=true; } }
    });
  }
  document.addEventListener('click', function(e){
    var btn=e.target.closest && e.target.closest('[data-read-toggle]');
    if(!btn) return;
    e.preventDefault(); e.stopPropagation();
    var now=Game.toggleRead(btn.dataset.readToggle);
    btn.classList.toggle('on', now);
    if(now){ btn.classList.remove('pop'); void btn.offsetWidth; btn.classList.add('pop'); }
  });
  Game.onChange(renderAll);
  if(document.readyState!=='loading') renderAll();
  else document.addEventListener('DOMContentLoaded', renderAll);
})();
"""
    (ROOT / "assets" / "ssl-game.js").write_text(js, encoding="utf-8")


def main() -> None:
    global CURRENT_DATE, PAPERS

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

    report_path = latest_report_path()
    report_md = report_path.read_text(encoding="utf-8")
    header_date = re.search(r"\d{4}-\d{2}-\d{2}", report_md[:200])
    CURRENT_DATE = header_date.group(0) if header_date else report_date_from_path(report_path)
    PAPERS = load_catalog_papers(report_path, report_md, CURRENT_DATE)
    (ROOT / "data" / "papers.json").write_text(json.dumps(PAPERS, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / ".nojekyll").write_text("", encoding="utf-8")
    write_css()
    write_js()
    write_index(report_md)
    write_issue(report_md)
    write_archive_issues()
    write_catalog()
    write_timeline()
    for p in PAPERS:
        write_paper(p, report_md)
    print(f"Built site at {ROOT}")


if __name__ == "__main__":
    main()
