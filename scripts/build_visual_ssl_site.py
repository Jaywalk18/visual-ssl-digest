from __future__ import annotations

import html
import json
import re
import shutil
from datetime import date, datetime
from pathlib import Path


ROOT = Path(r"H:\Desktop\visual_ssl_digest_site")
REPORT_ROOT = Path(r"H:\Desktop\visual_ssl_paper_reports")
CSS_VERSION = "20260525c"
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
      <span><b>MinerU</b>正文、表格、Figure 2/4/5/8 已接入</span>
    </div>
    <div class="feature-body">
      <p class="lead dropcap">先读这篇要从“为什么 I-JEPA 会不确定”开始，而不是从分数开始。I-JEPA 只拿可见 context patch 去预测 target patch feature，遮挡区域可能对应很多合理语义；TC-JEPA 的假设是 caption 能提供场景组成、属性和空间关系，把 target feature 的可行解空间变窄。</p>

      <section class="reading-note">
        <h2>一句话动机</h2>
        <p>论文自己的问题定义很直接：masked positions have “large uncertainties”。TC-JEPA 不重建像素，也不做 CLIP 式全局对比，而是在 JEPA predictor 的多层里加入 word-token 级 cross-attention，让 patch feature 在训练时能被文字条件化。</p>
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
    <div class="side-box"><h4>本页图源</h4><p>图像来自 MinerU 对 PDF 的结构化抽取；当前优先展示 pipeline、prediction visualization、ablation 和 efficiency。</p></div>
    <div class="side-box"><h4>核心判断</h4><dl><dt>相关性</dt><dd>P1，通用视觉自监督强相关</dd><dt>会议等级</dt><dd>ICML 2026，CCF A</dd><dt>读法</dt><dd>方法图优先，表格次之</dd></dl></div>
    <div class="side-box"><h4>原文</h4><p><a href="{e(p['url'])}">{e(p['url'])}</a></p></div>
  </aside>
</article>"""
    return shell(p["short"], body, 1)


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
                "takeaway": method + "。" if method else f"{title} 是本期日报自动纳入的候选论文。",
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


def nav(depth: int = 0, active: str = "") -> str:
    prefix = "../" * depth
    items = [
        ("index.html", "头版"),
        (f"issues/{CURRENT_DATE}.html", "今日速递"),
        ("pages/catalog.html", "论文目录"),
        ("pages/timeline.html", "时间线"),
    ]
    return "\n".join(
        f'<a class="nav-link{" active" if active == href else ""}" href="{prefix}{href}">{label}</a>'
        for href, label in items
    )


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
    <div class="masthead-kicker">Visual SSL Daily · GitHub Pages MVP · {e(CURRENT_DATE)}</div>
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
  {conference_widget(depth)}
  <footer class="footer">
    <div>原始日报继续同步飞书；本站用于图文归档和长读。</div>
    <div class="muted">Generated {e(CURRENT_DATE)} · MinerU figures where available</div>
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
    <h1>{e(hero['short'])}：{e(hero['thesis'].rstrip('。'))}</h1>
    <p class="dek">{e(hero['thesis'])}</p>
    {hero_img}
    <div class="paper-actions"><a href="issues/{CURRENT_DATE}.html">阅读 {zh_date(CURRENT_DATE)}速递</a><a href="papers/{hero['id']}.html">打开主文页</a></div>
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
    issue_papers = [p for p in PAPERS if p["date"] == CURRENT_DATE]
    cards = "\n".join(paper_card(p, 1) for p in issue_papers)
    issue_title = f"{zh_date(CURRENT_DATE)}：{PAPERS[0]['short']} 等视觉表征主线更新"
    body = f"""<article class="paper-detail issue-detail">
  <div class="paper-main">
    <div class="kicker">Daily issue · {e(CURRENT_DATE)} · CCF A/B 会议优先</div>
    <h1 class="paper-headline">{e(issue_title)}</h1>
    <p class="dek">本页由每日 Markdown 速递和 MinerU 图文抽取自动生成，优先保留 CCF A/B、OpenReview 和 arXiv 中对通用视觉表征有帮助的工作。</p>
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
    <div class="side-box"><h4>飞书文字版</h4><p><a href="https://www.feishu.cn/file/T684bdqYloXgmVxCV4ocxCm1nMc">latest.md →</a></p></div>
  </aside>
</article>"""
    (ROOT / "issues" / f"{CURRENT_DATE}.html").write_text(shell(f"{CURRENT_DATE} 速递", body, 1, f"issues/{CURRENT_DATE}.html"), encoding="utf-8")


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
    relevance = choose_field(fields, "为什么相关") or f"它和通用视觉自监督的关系在于：{p['thesis']}"
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
    body = f"""<article class="paper-detail">
  <div class="paper-main">
    <div class="kicker">{e(p['category'])} · {e(p['priority'])} · {e(p['date'])}</div>
    <h1 class="paper-headline">{e(p['short'])}：{e(p['thesis'])}</h1>
    <p class="dek">{e(p['takeaway'])}</p>
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
      <p class="lead dropcap">先说结论。{e(p['thesis'])} {e(p['takeaway'])}</p>
      {figs}
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
    </div>
  </div>
  <aside class="paper-side">
    <div class="side-box"><h4>关键判断</h4><dl><dt>相关性</dt><dd>{e(p['priority'])}</dd><dt>类别</dt><dd>{e(p['category'])}</dd><dt>会议</dt><dd>{e(p['venue'])}</dd><dt>读法</dt><dd>方法图优先，表格次之</dd></dl></div>
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
.masthead-meta{display:flex;flex-wrap:wrap;gap:8px 14px}
.masthead-meta span:before{content:none!important}
.lead-story{border-top:5px solid #111;padding-top:22px}
.lead-story h1{font-size:clamp(36px,4.5vw,62px);line-height:1.02;letter-spacing:0;margin:10px 0 18px;font-family:Georgia,'Times New Roman',serif;overflow-wrap:anywhere;word-break:break-word}
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
.paper-card{align-items:start;grid-template-columns:minmax(380px,520px) minmax(0,1fr);gap:20px;padding:16px}
.paper-card h3{margin-top:0}
.paper-card .paper-actions{position:relative;z-index:2;margin-top:14px;display:flex;flex-wrap:wrap;gap:10px}
.paper-card .paper-actions a{white-space:nowrap}
.paper-thumb.paper-thumb-text{position:relative;display:block;min-height:0;aspect-ratio:16/9;background:#fff;color:#1f1f1d;text-decoration:none;padding:10px;border:1px solid #d8d0c3;border-radius:2px;overflow:hidden}
.paper-thumb-text img{width:100%;height:100%;max-height:none;object-fit:contain;display:block;margin:0;background:#fff;border:0}
.paper-thumb-text span{position:absolute;left:10px;top:10px;display:inline-block;margin:0;padding:4px 7px;background:#8a2f21;color:#fff;font-size:12px;line-height:1;font-weight:700;letter-spacing:.04em}
.paper-thumb-text strong{display:block;font-size:25px;line-height:1.05;margin:10px 0;font-family:Georgia,'Times New Roman',serif}
.paper-thumb-text em{font-size:12px;color:#6b665f;font-style:normal}
.hero-figure{margin:24px 0;border-top:1px solid #d8d0c3;border-bottom:1px solid #d8d0c3;padding:16px 0}
.hero-figure img{width:100%;max-height:430px;object-fit:contain;background:#fff}
.hero-figure figcaption{font-size:13px;line-height:1.45;color:#5f5b55;margin-top:8px}
.deep-read .full-fig{margin:28px 0;padding:14px;background:#fbfaf6;border:1px solid #d8d0c3}
.deep-read .full-fig img{width:100%;max-height:720px;object-fit:contain;background:#fff}
.deep-read .full-fig figcaption b{display:block;margin-bottom:8px;color:#8a2f21;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase}
.deep-read .full-fig .fig-en,.deep-read .full-fig .fig-zh{display:block;font-size:14px;line-height:1.62}
.deep-read .full-fig .fig-en{color:#4f4b45}
.deep-read .full-fig .fig-zh{margin-top:6px;color:#1f1f1d}
.deep-read .feature-body{column-count:1;column-width:auto}
.reading-note{border-left:5px solid #8a2f21;background:#f7f3ea;padding:16px 18px;margin:24px 0}
.reading-note h2{margin-top:0}
.method-steps{padding-left:22px;display:grid;gap:10px}
.method-steps li{line-height:1.62}
.stat-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:20px 0}
.stat-grid div{border:1px solid #d8d0c3;background:#fff;padding:14px}
.stat-grid b,.stat-grid span,.stat-grid em{display:block}
.stat-grid b{color:#8a2f21;font-size:13px;text-transform:uppercase}
.stat-grid span{font-size:20px;line-height:1.25;margin:6px 0}
.stat-grid em{font-style:normal;color:#64605a;font-size:13px;line-height:1.45}
.issue-detail .paper-list{break-inside:avoid;column-span:all}
.issue-detail .paper-headline{max-width:980px;font-size:clamp(31px,3.35vw,42px);line-height:1.16;text-wrap:pretty;word-break:normal;overflow-wrap:normal;line-break:loose}
.page-header{max-width:1500px;margin:32px auto;padding:0 24px;border-bottom:2px solid #111}
.page-header h1{font-size:54px;line-height:1;margin:0 0 10px;font-family:Georgia,'Times New Roman',serif}
.timeline-day{max-width:1100px;margin:28px auto;padding:0 24px}
.timeline-day h2{border-bottom:2px solid #111;padding-bottom:8px}
.timeline-day article{display:grid;grid-template-columns:70px 1fr 240px;gap:16px;border-bottom:1px solid #ddd;padding:12px 0}
.timeline-day span{font-weight:700;color:#8a2f21}
.timeline-day em{font-style:normal;color:#65615a}
.conf-float{position:fixed;right:18px;bottom:18px;z-index:45;width:min(360px,calc(100vw - 32px));background:#fbfaf6;border:1px solid #d8d0c3;border-radius:8px;box-shadow:0 16px 36px rgba(31,31,29,.18);padding:14px 42px 14px 14px}
.conf-float details{margin:0}
.conf-float summary{cursor:pointer;font-weight:700;color:#1f1f1d;list-style:none}
.conf-float summary::-webkit-details-marker{display:none}
.conf-close{position:absolute;right:10px;top:9px;width:24px;height:24px;border:1px solid #d8d0c3;border-radius:999px;background:#fff;color:#5f5b55;font-size:17px;line-height:20px;cursor:pointer;padding:0}
.conf-close:hover{border-color:#8a2f21;color:#8a2f21}
.conf-list{display:grid;gap:8px;margin-top:10px}
.conf-item{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 10px;text-decoration:none;color:#1f1f1d;border-left:4px solid #b8afa1;background:#fff;padding:9px 10px;border-radius:6px}
.conf-item b,.conf-item em,.conf-item small{display:block}
.conf-item em{font-style:normal;font-size:12px;color:#6b665f;margin-top:2px}
.conf-item strong{color:#5b5a56;font-size:13px;white-space:nowrap}
.conf-item small{grid-column:1/-1;color:#6b665f;line-height:1.45}
.conf-item.urgent{border-left-color:#8a2f21}
.conf-item.urgent strong,.conf-item.soon strong{color:#8a2f21}
.conf-item.past{opacity:.72}
@media (max-width:900px){.hero-grid{grid-template-columns:1fr;padding:0 12px;box-sizing:border-box;max-width:100%;width:100%;margin-left:0;margin-right:0}.lead-story h1{font-size:40px}.paper-card{grid-template-columns:1fr}.timeline-day article{grid-template-columns:60px 1fr}.timeline-day em{grid-column:2}.stat-grid{grid-template-columns:1fr}.paper-thumb.paper-thumb-text{aspect-ratio:16/10}.conf-float{position:static;width:auto;margin:16px 12px;box-sizing:border-box}}
"""
    (ROOT / "assets" / "visual-ssl.css").write_text(css, encoding="utf-8")


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
    PAPERS = merge_report_papers(parse_paper_index(report_md, CURRENT_DATE))
    (ROOT / "data" / "papers.json").write_text(json.dumps(PAPERS, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / ".nojekyll").write_text("", encoding="utf-8")
    write_css()
    write_index(report_md)
    write_issue(report_md)
    write_catalog()
    write_timeline()
    for p in PAPERS:
        write_paper(p, report_md)
    print(f"Built site at {ROOT}")


if __name__ == "__main__":
    main()
