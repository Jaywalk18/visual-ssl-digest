# Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating

Zhe Cheng \* 1 Wenyu Chen \* 1 Fode Zhang 1 Dehuan Shen 2

# Abstract

Large vision-language models (LVLMs) often hallucinate content that is fluent yet unsupported by the image, limiting their reliability in real-world deployment. We show that a key failure mode arises from route competition: even when visual tokens receive attention, the final token decision can be dominated by the textual pathway, causing the decoder to follow linguistic priors over visual evidence. To mitigate this, we propose a trainingfree, decision-aligned intervention that decomposes each attention head into a visual route and a text route, and estimates their token-level effects using an efficient one-forward/one-gradient approximation. These estimates reveal route conflict within heads and identify prior-dominant ones, enabling selective suppression of only the text route while keeping the visual route intact. Across five benchmarks spanning discriminative and generative settings, our method consistently reduces hallucination-related errors across models with limited impact on overall multimodal performance, while incurring a modest inference-time overhead.

# 1. Introduction

LVLMs have become a core interface for visual understanding, enabling systems that can answer questions about images and generate grounded descriptions (Yin et al., 2024a; Caffagni et al., 2024). However, a central obstacle to reliable deployment is hallucination: the model produces fluent, plausible content that is not supported by the image (Liu et al., 2024c). Importantly, this is not merely “poor writing”

\*Equal contribution 1Center of Statistical Research, School of Statistics and Data Science, Southwestern University of Finance and Economics, Chengdu, China. 2Department of Biomedical Engineering, College of Design and Engineering, National University of Singapore, Singapore. Correspondence to: Fode Zhang <fredzh@swufe.edu.cn>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

![](images/7b252723accfac980791e28b7db2326739a1a237991447fd4b4117af7e9989a9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input: Banana Image"] --> B["Question: What color is the banana in this image?"]
    B --> C["Baseline: Text-Prior Hallucination"]
    C --> D["MHA"]
    D --> E["Text Route (Ovis)"]
    E --> F["Priors: Yellow"]
    F --> G["Visual Route (Ovis)"]
    G --> H["Answer: The banana... is yellow, Conflict-B: Δtct > 0 Promote Δvis < 0 Opposes"]
    H --> I["Conflict-Aware Text Gating (gText)"]
    I --> J["Suppress Text Prior"]
    J --> K["Ours: Correcting via Route Intervention"]
    K --> L["MHA"]
    L --> M["Text Route"]
    M --> N["Priors: Yellow"]
    N --> O["Visual Route (Ovis)"]
    O --> P["Evidence: Black"]
    P --> Q["Answer: The banana... is black, Visual Evidence Restored"]
    Q --> R["Conflict-Aware Text Gating (gText)"]
    R --> S["1.0"]
    S --> T["0.3"]
    T --> U["Intervention"]
```
</details>

Figure 1. When language priors override visual inputs, the baseline model tends to produce hallucinated predictions (left). To address this, a conflict-aware intervention applies text-route gating to suppress language priors (middle), enabling the model to correctly rely on visual evidence after intervention (right).

or generic inaccuracy; rather, it often reflects a failure of grounding, where the model fills in missing visual details using language priors (Favero et al., 2024; Wu et al., 2025; Zhou et al., 2025). Such prior-driven fabrications directly undermine trustworthiness and can be harmful in safetycritical or decision-support settings (Wang et al., 2023b). At a high level, hallucination often reflects a mismatch between visual evidence and language priors: the model may attend to the image, yet the final token choice is still dominated by the textual pathway, favoring what is plausible over what is visible.

Training-free mitigation has become attractive for deployment. Among approaches that act at inference time without additional training, two representative paradigms emerge: (i) output-level decoding control, and (ii) proxy-guided internal intervention. The first paradigm modifies generation behavior through decoding-time heuristics or constraints (e.g., Visual Contrastive Decoding (VCD) (Leng et al., 2024), OPERA (Huang et al., 2024), MaskCD (Deng & Yang, 2025)). While effective, these methods primarily act as external policies and provide limited insight into which internal components drive prior-dominant errors. The second paradigm performs internal intervention by selecting modules using attention-based proxies (Chefer et al., 2021; Yuksekg¨ on¨ ul et al.¨ , 2024), most notably VAR-style measures (Jiang et al., 2025; Zheng & Zhang, 2025; Jung et al., 2025) that quantify the attention share on visual tokens. Yet such proxies are inherently correlational: higher visual attention does not necessarily translate to larger causal contribution to the output (Jain & Wallace, 2019; Serrano & Smith, 2019), and softmax normalization can spuriously change the visual share by weakening textual competition (Vaswani et al., 2017). Moreover, proxy-based selection is typically paired with coarse head rescaling (Jiang et al., 2025; Yu et al., 2026), which inevitably suppresses both routes and may harm genuinely grounded reasoning.

To move beyond attention-based proxies, we expose a headinternal visual route and text route, so that we can intervene on either route at inference time (Fig. 1). For the current token $y ^ { * }$ , we measure decision-aligned route effects by do interventions: $\Delta _ { l , h } ^ { \mathrm { v i s } } = \ell _ { l , h } ( 1 , 1 ) - \ell _ { l , h } ( 0 , 1 )$ and $\Delta _ { l , h } ^ { \mathrm { t x t } } = \ell _ { l , h } ( 1 , 1 ) - \ell _ { l , h } ( 1 , 0 )$ . These effects directly reveal route conflict: a head is conflicting when sign $( \Delta _ { l , h } ^ { \mathrm { v i s } } ) \neq$ $\mathrm { s i g n } ( \Delta _ { l , h } ^ { \mathrm { t x t } } )$ , with Conflict-A (+, −) and Conflict-B $( - , + )$ . Within conflicting heads, we identify prior-dominant ones by $\begin{array} { r } { V R I _ { l , h } ~ = ~ \frac { \left| \Delta _ { l , h } ^ { \mathrm { v i s } } \right| } { \left| \Delta _ { l , h } ^ { \mathrm { v i s } } \right| + \left| \Delta _ { l , h } ^ { \mathrm { t x t } } \right| + \epsilon } } \end{array}$ (smaller $V R I _ { l , h }$ indicates more text-driven behavior). At decoding time, we estimate $\Delta _ { l , h } ^ { \mathrm { v i s } }$ and $\Delta _ { l , h } ^ { \mathrm { t x t } }$ with a one-forward/one-gradient approximation, select Top-k smallest $V R I _ { l , h }$ within the conflict sets, and suppress only their text routes using a smooth rankbased schedule, with stronger suppression for Conflict-B than Conflict-A (visual routes remain intact). We evaluate our method on five benchmarks spanning discriminative and generative settings. Results indicate that the proposed method can reduce hallucination-related errors across models, with limited impact on overall multimodal performance.

Contributions. (1) We introduce route-level, decisionaligned causal effects for attention heads and a corresponding Visual Reliance Index. (2) We derive an efficient firstorder estimator that enables per-step head selection during decoding. (3) We propose conflict-aware text-route gating that targets prior-dominant, decision-conflicting heads without suppressing visual routes. (4) We theoretically analyze VAR-style proxies, showing attention mass can be misaligned with decision-relevant visual grounding.

# 2. Related Work

Hallucination in LVLMs. Hallucination in LVLMs has received growing attention because it undermines visual grounding (Li et al., 2023; Wang et al., 2023a; Fu et al., 2025; Ye et al., 2023). Unlike textual LLMs, LVLMs exhibit prominent object hallucination (Rohrbach et al., 2018), where the model mentions objects or attributes not supported by the image. Mitigation methods span trainingbased alignment/fine-tuning (Liu et al., 2024a; Gunjal et al., 2024; Hu et al., 2023) and training-free inference-time control (Yin et al., 2024b; Tong et al., 2024; Chen et al., 2024; Zhang et al., 2025; Tang et al., 2025). On the training-free side, methods broadly fall into output-level decoding control and intervention-style approaches that act on internal computation at inference time (Sarkar et al., 2025). For example, PAI (Liu et al., 2024d) encourages stronger reliance on visual evidence, and latent-space steering methods such as VTI (Liu et al., 2025) modify hidden representations to suppress hallucination tendencies. Our work is most closely related to these training-free intervention paradigms, but differs in where and how we act: we compute token-level within-head route effects and apply route-specific gating at decoding time to reduce language-prior influence.

Causal Interventions for Mechanistic Attribution. Mechanistic interpretability (Elhage et al., 2021; Sharkey et al., 2025) increasingly emphasizes interventional attribution, where one perturbs internal states and measures the induced behavioral change (Geiger et al., 2021; 2025). Prior work on pruning and ablations suggests substantial redundancy in multi-head attention, but indiscriminate removals are often a blunt instrument and do not directly localize causal pathways (Michel et al., 2019; Voita et al., 2019). To pinpoint causal structure, Causal Mediation Analysis (CMA) (Vig et al., 2020a; Yang et al., 2025) and activation patching (a.k.a. causal tracing) (Zhang & Nanda, 2024) perform targeted interventions under controlled inputs to estimate component effects (Vig et al., 2020b; Geiger et al., 2021; Meng et al., 2022; Yang et al., 2025), and path-level methods (Qian et al., 2025) further attribute behaviors to specific information-flow paths (Goldowsky-Dill et al., 2023). Since head contributions can be highly interactive and nonmodular, Causal Head Gating (CHG) learns differentiable head gates and searches for sufficient head sets directly from data, reducing reliance on hand-crafted prompt pairs (Nam et al., 2025). Building on this interventional line, we extend head-level interventions to within-head routes: unlike CHG’s learned head gates, we estimate route-level effects to drive decoding-time route-specific gating that reduces language-prior dominance and maintains visual grounding.

# 3. Causal Route Effects

Figure 2 provides an overview; we now detail Step 1 (Calculate CRE) and later introduce conflict-aware gating.

# 3.1. Exact, Head-Internal Decomposition

We consider a pretrained LVLM parameterized by fθ. At a given decoding step, the model receives a prefix of length $L = M + T$ , consisting of M visual tokens indexed by $I _ { \mathrm { v i s } } = \{ 1 , \dots , M \}$ and $\bar { T }$ textual tokens indexed by $I _ { \mathrm { t x t } } =$ $\{ M + 1 , \ldots , L \}$ .

Many modern LVLMs employ Transformer-based language decoders with multi-head self-attention (MHA). At layer l, for attention head $h \in \{ 1 , \ldots , H \}$ , we use learned projections $W _ { l , h } ^ { Q } , W _ { l , h } ^ { K } , W _ { l , h } ^ { V } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { h } }$ , forming $Q _ { l , h } , K _ { l , h } , V _ { l , h }$ in the standard way. The causal attention weights are then computed by a masked softmax:

![](images/dc9eb512dc6d9bae38fe5a3226a6620ad87ea8a14161e50bb6f36f2e7c571b08.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Step1: Calculate Causal Route Effects"] --> B["MHA"]
    B --> C["Multi-Head Attention"]
    C --> D["MHA"]
    E["Visual Tokens"] --> F["Apply Gate"]
    F --> G["Head Output O^vis_{l,h}"]
    G --> H["g^vis_{l,h}"]
    H --> I["+ g^txt_{l,h}"]
    I --> J["Textual Effects Δ^vis_{l,h}"]
    J --> K["Textual Effect Δ^txt_{l,h}"]
    
    L["Step2: Conflict Identification"] --> M["Δ^vis_{l,h} > 0"]
    M --> N["Agreement (++) No Interv"]
    M --> O["Conflict-B (-) Strong Interv"]
    M --> P["Conflict-A (+) Mild Interv"]
    M --> Q["Agreement (-) No. Interv"]
    N --> R["Select Top-k smallest VRI in Conflicts"]
    O --> R
    P --> R
    Q --> R
    
    S["Step3: Intervention"] --> T["Mild Gating (A)"]
    T --> U["Strong Gating (B)"]
    U --> V["Ranks"]
    U --> W["Ranks"]
    V --> X["δ̃_l,h = 1 * O^vis_{l,h} + g^txt_{l,h} * O^txt_{l,h}"]
    W --> Y["δ̃_l,h"]
```
</details>

Figure 2. Overview of our CRG framework. Step 1 computes modal causal effects for each attention head by separating visual and textual components and measuring their contributions. Step 2 identifies conflicts by comparing the signs of visual and textual effects, distinguishing Agreement vs. Conflict (A/B), and selects the top-k conflicting heads for intervention. Step 3 applies head gating, where Conflict-A heads receive mild gating and Conflict-B heads receive strong gating, reducing language-prior dominance while keeping image-grounded evidence.

$$
\alpha_ {l, h} = \mathrm{softmax} \left(\frac {Q _ {l , h} K _ {l , h} ^ {\top}}{\sqrt {d _ {h}}} + M _ {\mathrm{causal}}\right),
$$

where $M _ { \mathrm { c a u s a l } } \in \mathbb { R } ^ { L \times L }$ assigns −∞ to future positions to enforce autoregressive attention. Finally, the head output is

$$
O _ {l, h} = \alpha_ {l, h} V _ {l, h}.
$$

To isolate modality-specific contributions, we leverage the visual/textual index sets $I _ { \mathrm { v i s } }$ and $I _ { \mathrm { t x t } }$ introduced above and define diagonal selection matrices $S _ { \mathrm { v i s } } , S _ { \mathrm { t x t } } \in \{ 0 , 1 \} ^ { L \times L }$ as

$$
S _ {\mathrm{vis}} = \mathrm{diag} \big (\mathbf {1} [ 1 \in I _ {\mathrm{vis}} ], \ldots , \mathbf {1} [ L \in I _ {\mathrm{vis}} ] \big),
$$

$$
S _ {\mathrm{txt}} = \mathrm{diag} \big (\mathbf {1} [ 1 \in I _ {\mathrm{txt}} ], \ldots , \mathbf {1} [ L \in I _ {\mathrm{txt}} ] \big).
$$

and they satisfy

$$
S _ {\mathrm{vis}} + S _ {\mathrm{txt}} = I _ {L}, \qquad S _ {\mathrm{vis}} S _ {\mathrm{txt}} = 0,
$$

so the two masks are complementary and non-overlapping.

Applying these masks to the value matrix yields two routespecific value blocks, and consequently two route-specific head outputs:

$$
O _ {l, h} ^ {\mathrm{vis}} = \alpha_ {l, h} \big (S _ {\mathrm{vis}} V _ {l, h} \big), \qquad O _ {l, h} ^ {\mathrm{txt}} = \alpha_ {l, h} \big (S _ {\mathrm{txt}} V _ {l, h} \big).
$$

Since $V _ { l , h } = ( S _ { \mathrm { v i s } } + S _ { \mathrm { t x t } } ) V _ { l , h }$ , the head output splits exactly into visual and textual routes,

$$
O _ {l, h} = \alpha_ {l, h} V _ {l, h} = O _ {l, h} ^ {\mathrm{vis}} + O _ {l, h} ^ {\mathrm{txt}}.
$$

Propagation through the MHA output. After computing all head outputs, the multi-head attention module concatenates them and applies the output projection:

$$
Y _ {l} ^ {\mathrm{MHA}} = \mathrm{Concat} _ {h = 1} ^ {H} (O _ {l, h}) W _ {l} ^ {O},
$$

$$
W _ {l} ^ {O} \in \mathbb {R} ^ {H d _ {h} \times d _ {\mathrm{model}}}.
$$

Because concatenation and the linear map $W _ { l } ^ { O }$ are both linear operations, the same decomposition carries through the full MHA block:

$$
Y _ {l} ^ {\mathrm{MHA}} = \operatorname{Concat} _ {h} \left(O _ {l, h} ^ {\text {vis}}\right) W _ {l} ^ {O} + \operatorname{Concat} _ {h} \left(O _ {l, h} ^ {\text {txt}}\right) W _ {l} ^ {O}
$$

$$
\triangleq Y _ {l} ^ {\mathrm{vis}} + Y _ {l} ^ {\mathrm{txt}}.
$$

This gives an exact, layer-wise split of the MHA output into a visual-route component and a text-route component, which we will later exploit for route-specific interventions.

# 3.2. Interventional Gates and Causal Estimands

For each layer l and head h, after computing the jointsoftmax attention and forming the exact split $O _ { l , h } ^ { \mathrm { v i s } }$ and $O _ { l , h } ^ { \mathrm { t x t } }$ , we attach head-internal, route-specific scalar gates $g _ { l , h } ^ { \mathrm { v i s } } , g _ { l , h } ^ { \mathrm { t x t } } \ \in \ \mathbb { R } _ { \ge 0 }$ before the output projection $W _ { l } ^ { O }$ (Qiu et al., 2025). The gated head output is

$$
\tilde {O} _ {l, h} (g _ {l, h} ^ {\text { vis }}, g _ {l, h} ^ {\text { txt }}) = g _ {l, h} ^ {\text { vis }} O _ {l, h} ^ {\text { vis }} + g _ {l, h} ^ {\text { txt }} O _ {l, h} ^ {\text { txt }}.
$$

To quantify the effect of each route, we perform single-head interventions: we only modify the two route gates of one head $( l , h )$ and keep the rest of the network unchanged. As a reference, the baseline configuration sets all gates to one, i.e., gvisl′,h′ = $g _ { l ^ { \prime } , h ^ { \prime } } ^ { \mathrm { v i s } } = g _ { l ^ { \prime } , h ^ { \prime } } ^ { \mathrm { t x t } } = 1$ for every head $( l ^ { \prime } , h ^ { \prime } )$ .

For the selected head $( l , h )$ , we then consider binary gate values $g _ { \mathrm { v i s } } , g _ { \mathrm { t x t } } \in \{ 0 , 1 \}$ : setting $g _ { \mathrm { v i s } } ~ = ~ 0$ turns off the visual route of this head, and setting $g _ { \mathrm { t x t } } = 0$ turns off its text route. All other heads remain at the baseline (1, 1).

Let f (l,h)gvis,gtxt $f _ { g _ { \mathrm { v i s } } , g _ { \mathrm { t x t } } } ^ { ( l , h ) }$ denote the intervened network in which head $( l , h )$ uses gates $( g _ { \mathrm { v i s } } , g _ { \mathrm { t x t } } )$ . Given a task-specific scalar decision score $\ell ( \cdot )$ , we define

$$
\ell_ {l, h} (g _ {\text { vis }}, g _ {\text { txt }}) := \ell \left(f _ {g _ {\text { vis }}, g _ {\text { txt }}} ^ {(l, h)}\right).
$$

Here $\ell _ { l , h } ( g _ { \mathrm { v i s } } , g _ { \mathrm { t x t } } )$ is the decision score obtained under the single-head intervention on $( l , h )$ with gates $( g _ { \mathrm { v i s } } , g _ { \mathrm { t x t } } )$ . We define the visual and text Causal Route Effects (CRE) for head (l, h) by the following do-effects:

$$
\Delta_ {l, h} ^ {\text {vis}} := \ell_ {l, h} (1, 1) - \ell_ {l, h} (0, 1), \tag {1}
$$

$$
\Delta_ {l, h} ^ {\mathrm{txt}} := \ell_ {l, h} (1, 1) - \ell_ {l, h} (1, 0).
$$

Intuitively, $\Delta _ { l , h } ^ { \mathrm { v i s } }$ measures how much the decision score changes when we remove only the visual route of this head (while keeping its text route on), and $\Delta _ { l , h } ^ { \mathrm { t x t } }$ is defined analogously. A positive $\Delta$ indicates that the corresponding route supports the decision (increases ℓ), whereas a negative $\Delta$ indicates an inhibiting effect.

The choice of ℓ is task-dependent: it is a scalar score that summarizes the model’s current decision, so that $\Delta _ { l , h } ^ { \mathrm { v i s } }$ and $\Delta _ { l , h } ^ { \mathrm { t x t } }$ quantify how the visual/text routes of head $( l , h )$ support or oppose that decision. For generative benchmarks, we set $\ell = \log p ( y ^ { * } )$ , the log-probability of the current token $y ^ { * }$ , to measure how each route contributes to producing the desired output. For binary QA, we use the Yes/No margin $\ell = \log p ( \mathrm { Y e s } ) - \log p ( \mathrm { N o } )$ , so that the sign of ∆ directly indicates whether a route pushes the prediction toward Yes or toward No. All interventions are local: we only toggle the selected gates while keeping the rest of the computation fixed. Details of ℓ are provided in Appendix D.3.

To summarize whether a head’s influence on the decision score ℓ is dominated by visual evidence or by textual context, we further define a normalized Visual Reliance Index (VRI):

$$
\mathrm{VRI} _ {l, h} = \frac {\left| \Delta_ {l , h} ^ {\text { vis }} \right|}{\left| \Delta_ {l , h} ^ {\text { vis }} \right| + \left| \Delta_ {l , h} ^ {\text { txt }} \right| + \varepsilon}. \tag {2}
$$

where $\varepsilon > 0$ is a small stability constant for numerical stability (we use $\varepsilon = 1 0 ^ { - 8 }$ in experiments). We use $\mathrm { V R I } _ { l , h }$ to rank heads for inference-time gating, and analogously define layer- or model-level indices by aggregation.

# 3.3. One-Forward Estimation

We estimate the binary do-effects using a first-order approximation that requires only one forward pass and a single autograd gradient query per decoded token. Specifically, we run a standard decode forward with all gates set to 1, caching the route-specific head outputs $O _ { l , h } ^ { \mathrm { v i s } }$ and $O _ { l , h } ^ { \mathrm { t x t } }$ (with the KV-cache intact). We then obtain the sensitivity $G _ { l , h } = \partial \ell / \partial \tilde { O } _ { l , h }$ by directly querying reverse-mode autodiff (e.g., via torch.autograd.grad) at the $\mathrm { p r e } { \cdot } W _ { l } ^ { O }$ tensor. This yields directional-derivative estimators along the two gate axes:

Proposition 3.1 (Directional-derivative estimator). At $( g _ { \mathrm { v i s } } , g _ { \mathrm { t x t } } ) = ( 1 , 1 )$ ,

$$
\widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} = \left\langle G _ {l, h}, O _ {l, h} ^ {\mathrm{vis}} \right\rangle , \quad \widehat {\Delta} _ {l, h} ^ {\mathrm{txt}} = \left\langle G _ {l, h}, O _ {l, h} ^ {\mathrm{txt}} \right\rangle , \tag {3}
$$

which coincide with the first-order terms of the exact dodifferences $\Delta _ { l , h } ^ { \mathrm { v i s } }$ s and ∆txt. $\Delta _ { l , h } ^ { \mathrm { t x t } }$ (Proof in Appendix C.)

Remark 3.2 (Coarse estimates for downstream intervention). Since $\widehat { \Delta } _ { l , h }$ is first-order and local, we treat it as a coarse signal for the intervention in Sec. 4, which mainly depends on sign/ranking rather than precise effect magnitudes. This aligns with the broader insight that allocation-style objectives can often be optimized with coarse effect estimates (Casacuberta & Hardt, 2026).

To justify our first-order estimator, we bound its deviation from the exact do-effects under a Lipschitz condition on $G _ { l , h }$ (Appendix B.1). We also validate it empirically by comparing $\hat { \Delta }$ with exact hard-intervention effects, with an emphasis on sign consistency that underlies our conflict taxonomy (Appendix B.2, B.3). We then compute the plugin VRI as

$$
\widehat {\mathrm{VRI}} _ {l, h} = \frac {| \widehat {\Delta} _ {l , h} ^ {\mathrm{vis}} |}{| \widehat {\Delta} _ {l , h} ^ {\mathrm{vis}} | + | \widehat {\Delta} _ {l , h} ^ {\mathrm{txt}} | + \varepsilon}.
$$

# 4. Conflict-Aware Text-Route Intervention

Building on $( \hat { \Delta } _ { l , h } ^ { \mathrm { v i s } } , \hat { \Delta } _ { l , h } ^ { \mathrm { t x t } } )$ , we derive a lightweight, conflictaware rule that selectively suppresses unreliable text contributions while preserving visual evidence (Fig. 2, Steps 2–3).

We categorize heads by the sign pattern of $( \hat { \Delta } _ { l , h } ^ { \mathrm { v i s } } , \hat { \Delta } _ { l , h } ^ { \mathrm { t x t } } )$ into four regimes (Table 1). When the signs agree $( + + \thinspace \mathrm { o r } - )$ , both routes move ℓ in the same direction, so we leave these heads unchanged to avoid unnecessary perturbations. We intervene only under modality conflicts $( + \mathrm { ~ o r ~ } \to )$ , and only through the text route $g _ { l , h } ^ { \mathrm { t x t } }$ (keeping $g _ { l , h } ^ { \mathrm { v i s } } = 1 )$ .

Table 1. Sign taxonomy of route gains, where $\hat { \Delta } > 0$ increases the decision score ℓ. We intervene only under conflicts (A/B) and leave agreement cases unchanged. 

<table><tr><td></td><td> $\hat{\Delta}^{\text{vis}} > 0$ </td><td> $\hat{\Delta}^{\text{vis}} < 0$ </td></tr><tr><td> $\hat{\Delta}^{\text{txt}} > 0$ </td><td>Agreement (+,+)</td><td>Conflict-B (-,+)</td></tr><tr><td> $\hat{\Delta}^{\text{txt}} < 0$ </td><td>Conflict-A (+,-)</td><td>Agreement (-,-)</td></tr></table>

# Algorithm 1 Conflict-Aware Text-Route Intervention

Require: Per-head estimates ∆ˆ visl,h, $\hat { \Delta } _ { l , h } ^ { \mathrm { t x t } }$ ; VRI scores $v _ { l , h } ;$ top-k values $k ;$ gate ranges $( \dot { g } _ { \mathrm { m i n } } ^ { A } , \dot { g } _ { \mathrm { m a x } } ^ { A } ) , ( g _ { \mathrm { m i n } } ^ { B } , g _ { \mathrm { m a x } } ^ { B } ) \colon$ $\gamma , \epsilon .$

Ensure: Text gates $\{ g _ { l , h } ^ { \mathrm { t x t } } \}$ (default 1).

1: Build conflict sets $\mathcal { H } _ { A } , \mathcal { H } _ { B }$ in Eq. (4).   
2: for $\mathcal { H } \in \{ \mathcal { H } _ { A } , \mathcal { H } _ { B } \}$ do   
3: Select S = TopKSmallest ${ \mathrm { \Omega } } ( \left\{ v _ { l , h } \right\} , k )$ within $\mathcal { H } .$   
4: Sort S by $v _ { l , h }$ ascending; assign rank i.   
5: Set $s _ { i } = i / ( | S | - 1 ) \ \mathrm { ( i f \ } | S | > 1 ) ;$ ; else set $g ^ { \mathrm { t x t } } =$ gmin.   
6: Assign $\begin{array} { r l r } { g _ { ( i ) } ^ { \mathrm { t x t } } } & { { } = } & { g _ { \mathrm { m i n } } + ( g _ { \mathrm { m a x } } - g _ { \mathrm { m i n } } ) . } \end{array}$ clip $( s _ { i } ^ { \gamma } , \epsilon , \stackrel { . } { 1 } - \epsilon ) .$   
7: Update the corresponding heads’ $g _ { l , h } ^ { \mathrm { t x t } }  g _ { ( i ) } ^ { \mathrm { t x t } }$ g (i) .   
8: end for   
9: return $\{ g _ { l , h } ^ { \mathrm { t x t } } \}$

Conflict-A (+,−): visual increases ℓ while text decreases ℓ. Here the visual route promotes the current decision score, but the text route acts as an obstruction. This pattern is consistent with text-side noise or local linguistic mismatch rather than a reliable correction to visual evidence. Our goal is stabilization: we mildly suppress the text route on a small subset of heads so that visual evidence can dominate without distorting generation.

Conflict-B (−,+): visual decreases ℓ while text increases ℓ (hallucination hallmark). Here visual evidence discourages the current decision, yet the text route actively promotes it. This corresponds to the regime where language priors override vision. Our goal is prior cutting: we strongly suppress the text route (possibly near-zero) on the most prior-dominant heads, so that visual counter-evidence can be reflected in the logits/decision score.

# 4.1. VRI-Based Intervention

Algorithm 1 summarizes the procedure.

Selecting the most relevant heads using VRI. Recall the two modality-conflict regimes: Conflict-A (+,−) and Conflict-B (−,+). We denote their head sets by

$$
\mathcal {H} _ {A} = \{(l, h): \hat {\Delta} _ {l, h} ^ {\text {vis}} > 0, \hat {\Delta} _ {l, h} ^ {\text {txt}} <   0 \}, \tag {4}
$$

$$
\mathcal {H} _ {B} = \{(l, h): \hat {\Delta} _ {l, h} ^ {\mathrm{vis}} <   0, \hat {\Delta} _ {l, h} ^ {\mathrm{txt}} > 0 \}.
$$

![](images/92bb7eaa3ad7748c85112d5645c6cd9b7dab0e173453529b5a70fe0336a9c7ea.jpg)

<details>
<summary>heatmap</summary>

| Layer | VAR Value | VRI Value |
|-------|-----------|-----------|
| 0     | 0.0       | 0.0       |
| 4     | 0.1       | 0.2       |
| 8     | 0.3       | 0.4       |
| 12    | 0.5       | 0.6       |
| 16    | 0.7       | 0.8       |
| 20    | 0.9       | 0.6       |
| 24    | 0.7       | 0.4       |
| 28    | 0.5       | 0.2       |
</details>

Figure 3. Per-layer, per-head VAR (left) and VRI (right) for the generated token $\mathrm { \cdots { Y e s } ^ { , , } }$ from LLaVA-1.5-7B. VAR reflects visual attention allocation, while VRI reflects relative visual reliance computed from decision-aligned route effects.

Within each conflict set $\mathcal { H } \in \{ \mathcal { H } _ { A } , \mathcal { H } _ { B } \}$ , we assign each head a scalar score $v _ { l , h } = \mathrm { V R I } _ { l , h }$ . We then select a small subset $\cal S \subseteq { \mathcal { H } }$ of size k by taking the heads with the smallest VRI values: $\begin{array} { r } { S = \mathrm { T o p K S m a l l e s } } \end{array}$ t $( \{ v _ { l , h } : ( l , h ) \in \mathcal { H } \} , k )$ . This concentrates the intervention on a limited number of heads, improving controllability and reducing unintended side effects. More broadly, this top-k selection can be viewed as a budgeted allocation step, where coarse effectderived scores may suffice for near-optimal allocation value (Casacuberta & Hardt, 2026).

Rank-range gate schedule. Given the selected heads S with $| S | = n ,$ we sort them by VRI in ascending order and assign rank $i \in \{ 0 , \ldots , n - 1 \}$ . For $n > 1$ , we define $s _ { i } = i / ( n - 1 )$ and map it to a gate value: $g _ { ( i ) } ^ { \mathrm { t x t } } = g _ { \mathrm { m i n } } +$ $( g _ { \operatorname* { m a x } } - g _ { \operatorname* { m i n } } ) \cdot \mathrm { c l i p } ( s _ { i } ^ { \gamma } , \epsilon , 1 - \epsilon )$ , where $\gamma > 0$ controls the nonlinearity of the rank-to-gate mapping and $\epsilon > 0$ avoids extreme values. Only heads in $s$ receive scheduled gates, while all other heads keep $g _ { l , h } ^ { \mathrm { t x t } } = 1$ . This schedule ensures a smooth and monotonic suppression strength across the selected heads instead of a hard threshold.

Conflict-specific gate ranges. We use different gate ranges for the two conflicts: (i) Conflict-A applies mild text suppression with $( g _ { \mathrm { m i n } } ^ { A } , g _ { \mathrm { m a x } } ^ { A } ) = ( 0 . 5 , 1 . 0 )$ , treating the text route as potentially noisy but not adversarial. (ii) Conflict-B applies strong text suppression with $( g _ { \mathrm { m i n } } ^ { B } , g _ { \mathrm { m a x } } ^ { B } ) = ( 0 , 0 . 5 )$ , aiming to cut the prior-supporting pathway when language priors promote $y ^ { * }$ despite visual opposition. In both cases, $g _ { l , h } ^ { \mathrm { v i s } }$ remains 1. We keep these ranges fixed across models as a simple mild/strong semantic split, while model- and task-level adaptation comes from VRI-based head selection and within-range rank ordering.

Table 2. Main results on POPE tasks. The best performances are bolded. 

<table><tr><td rowspan="2">Setting</td><td rowspan="2">Method</td><td colspan="2">LLaVA-1.5-7B</td><td colspan="2">Qwen-VL-Chat</td><td colspan="2">Qwen2.5-VL-7B-Instruct</td></tr><tr><td>Accuracy↑</td><td>F1-Score↑</td><td>Accuracy↑</td><td>F1-Score↑</td><td>Accuracy↑</td><td>F1-Score↑</td></tr><tr><td rowspan="6">Random</td><td>Regular</td><td>83.29</td><td>81.33</td><td>84.63</td><td>82.61</td><td>84.52</td><td>84.62</td></tr><tr><td>VCD</td><td>87.73</td><td>87.16</td><td>86.93</td><td>85.46</td><td>87.82</td><td>88.23</td></tr><tr><td>OPERA</td><td>89.20</td><td>88.81</td><td>85.71</td><td>84.64</td><td>89.72</td><td>90.02</td></tr><tr><td>PAI</td><td>86.33</td><td>84.56</td><td>85.38</td><td>85.54</td><td>89.54</td><td>88.72</td></tr><tr><td>VTI</td><td>89.50</td><td>88.89</td><td>86.73</td><td>85.59</td><td>90.43</td><td>89.44</td></tr><tr><td>CRG(ours)</td><td>90.30 (+7.01)</td><td>89.51 (+8.18)</td><td>89.46 (+4.83)</td><td>88.33 (+5.72)</td><td>91.45 (+6.93)</td><td>89.23 (+4.61)</td></tr><tr><td rowspan="6">Popular</td><td>Regular</td><td>81.88</td><td>80.06</td><td>83.63</td><td>81.53</td><td>84.67</td><td>85.13</td></tr><tr><td>VCD</td><td>85.38</td><td>85.06</td><td>85.17</td><td>83.68</td><td>86.89</td><td>87.35</td></tr><tr><td>OPERA</td><td>86.64</td><td>86.62</td><td>84.82</td><td>83.99</td><td>87.57</td><td>88.12</td></tr><tr><td>PAI</td><td>85.33</td><td>83.62</td><td>84.20</td><td>83.10</td><td>88.14</td><td>87.32</td></tr><tr><td>VTI</td><td>87.36</td><td>86.69</td><td>85.67</td><td>84.48</td><td>89.23</td><td>86.23</td></tr><tr><td>CRG(ours)</td><td>88.40 (+6.52)</td><td>86.54 (+6.48)</td><td>87.63 (+4.00)</td><td>86.55 (+5.02)</td><td>90.73 (+6.06)</td><td>89.34 (+4.21)</td></tr><tr><td rowspan="6">Adversarial</td><td>Regular</td><td>78.96</td><td>77.57</td><td>81.03</td><td>79.30</td><td>82.79</td><td>83.15</td></tr><tr><td>VCD</td><td>80.88</td><td>81.33</td><td>83.10</td><td>82.04</td><td>84.78</td><td>84.63</td></tr><tr><td>OPERA</td><td>81.24</td><td>81.38</td><td>82.67</td><td>79.89</td><td>84.93</td><td>85.17</td></tr><tr><td>PAI</td><td>83.17</td><td>81.67</td><td>82.19</td><td>82.06</td><td>85.12</td><td>84.77</td></tr><tr><td>VTI</td><td>82.57</td><td>82.11</td><td>83.13</td><td>82.16</td><td>85.78</td><td>85.14</td></tr><tr><td>CRG(ours)</td><td>84.43 (+5.47)</td><td>83.77 (+6.20)</td><td>84.70 (+3.67)</td><td>83.78 (+4.48)</td><td>86.98 (+4.19)</td><td>87.07 (+3.92)</td></tr></table>

# 4.2. Why a High Visual Attention Ratio Does Not Guarantee Grounded Outputs

A popular proxy for “visual reliance” is the Visual Attention Ratio (VAR) (Jiang et al., 2025), which measures how much attention mass a head assigns to visual keys. Using the same visual index set $I _ { \mathrm { v i s } }$ as Sec. 3, for a decoding query position i we adopt VARl,h := Pj∈Ivis $\begin{array} { r } { \mathrm { V A R } _ { l , h } : = \sum _ { j \in I _ { \mathrm { v i s } } } \alpha _ { i j } ^ { ( l , h ) } } \end{array}$ αij , optionally averaged over query positions. However, VAR is not decision-aligned. We use our decision-aligned visual route effect $\widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } }$ to expose this misalignment, and use VRI to summarize normalized modality reliance. In Fig. 3, VAR and VRI share a similar pattern in the first few layers, where many heads allocate substantial attention to visual keys. However, in subsequent middle layers VAR becomes nearly flat even for the correct “Yes” token, while VRI still exhibits nontrivial structure. This indicates that, for this QA pattern, attention mass alone is not a reliable proxy for decision-relevant visual grounding.

The reason is that VAR depends only on attention weights, while the visual route effect depends on both weights and value content. Recall our first-order estimator in Eq. (3), $\widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } \ \approx \ \langle G _ { l , h } , O _ { l , h } ^ { \mathrm { v i s } } \rangle$ , where $G _ { l , h } : =$ $\partial \ell / \partial { \tilde { O } } _ { l , h }$ for a task-dependent scalar score ℓ. Writing $\begin{array} { r l r } { O _ { l , h } ^ { \mathrm { v i s } } } & { { } = } & { \sum _ { j \in I _ { \mathrm { v i s } } } \stackrel { - } { \alpha } _ { i j } ^ { ( l , h ) } v _ { j } ^ { ( l , h ) } } \end{array}$ Pj∈Ivis αij v j yields $\begin{array} { r l } { { 1 } } & { { } \widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } ~ \approx } \end{array}$ Pj∈Ivis $\begin{array} { r l } { \sum _ { j \in I _ { \mathrm { v i s } } } \alpha _ { i j } ^ { ( l , h ) } \langle G _ { l , h } , v _ { j } ^ { ( l , h ) } \rangle } \end{array}$ αij ⟨Gl,h, v(l,h)j ⟩. Thus, even when VARl,h is $\mathrm { V A R } _ { l , h }$ large, the signed projections $\langle G _ { l , h } , v _ { j } ^ { ( l , h ) } \rangle$ (l,h)⟩ can be small, can-1For vectors $a , b _ { j }$ and scalars $\begin{array} { r l } { c _ { j } , \left. a , \sum _ { j } c _ { j } b _ { j } \right. } & { { } = } \end{array}$ $\begin{array} { r } { \sum _ { j } \left. a , c _ { j } b _ { j } \right. = \sum _ { j } c _ { j } \left. a , b _ { j } \right. } \end{array}$ .

cel out, or be negative, making the net visual influence on the decision weak or even harmful—information that VAR discards.

Moreover, VAR is confounded by softmax competition. Let $\alpha _ { j } = \exp ( s _ { j } ) / \sum _ { k } \exp ( s _ { k } )$ , where $s _ { j }$ is the pre-softmax attention logit, and $\begin{array} { r } { R = \sum _ { j \in I _ { \mathrm { v i s } } } \alpha _ { j } } \end{array}$ (the per-query VAR). For a textual key k ∈ Itxt, ∂ $\begin{array} { r } { k \in I _ { \mathrm { t x t } } , \frac { \partial R } { \partial s _ { \iota } } = - R \alpha _ { k } < 0 , } \end{array}$ , so decreasing textual logits increases R even if all visual logits stay unchanged. Therefore, a higher VAR may reflect weakened textual competition rather than stronger visual grounding.

In summary, VAR is neither sufficient nor necessary for identifying decision-critical (or hallucination-prone) components, motivating our use of decision-aligned CRE in Eq. (1). For further analysis and experiment comparison, readers can refer to Appendix A.

# 5. Experiments

Models. We adopt LLaVA-1.5-7B (Liu et al., 2024b), Qwen-VL-Chat-7B (Bai et al., 2023), and Qwen2.5-VL-7B-Instruct (Bai et al., 2025). Appendix E.1 further reports results on more advanced LVLMs.

Benchmarks and Metrics. We evaluate our method on five benchmarks spanning discriminative and generative settings. We use POPE (Li et al., 2023) for binary hallucination QA (Accuracy/F1), CHAIR (Rohrbach et al., 2018) (caption-level object hallucination), MME (Fu et al., 2025) for comprehensive multimodal evaluation (overall score), MMHal-Bench (Sun et al., 2024) for open-ended hallucination assessment (official hallucination metrics), and AM-BER (Wang et al., 2023a), a unified benchmark covering both generative and discriminative hallucination behaviors. For detailed Benchmarks and Metrics, readers can refer to Appendix D.1.

![](images/a48ac6b34fe595ae86bffddb2bfc96bcf4d037e6837ec00eead70600fc3c31d0.jpg)

<details>
<summary>bar</summary>

| Category | Regular | VCD | OPERA | Ours |
| :--- | :--- | :--- | :--- | :--- |
| Existence | 160 | 180 | 185 | 195 |
| Count | 115 | 135 | 128 | 148 |
| Position | 108 | 120 | 125 | 135 |
| Color | 140 | 158 | 162 | 167 |
| Posters | 132 | 148 | 145 | 152 |
| Celebrity | 142 | 145 | 150 | 160 |
| Scene | 140 | 158 | 168 | 168 |
| Landmark | 148 | 165 | 170 | 172 |
| Artwork | 125 | 135 | 138 | 142 |
| OCR | 110 | 125 | 135 | 128 |
| Commonsense Reasoning | 90 | 102 | 112 | 115 |
| Numerical Calculation | 48 | 47 | 47 | 55 |
| Text Translation | 112 | 108 | 113 | 115 |
| Code Reasoning | 55 | 55 | 55 | 60 |
</details>

Figure 4. Category-wise MME scores (higher is better) comparing Regular decoding with VCD, OPERA, and our method.

Baselines and Implementation Details. We use greedy decoding unless otherwise specified. We denote our inferencetime method as Causal Route Gating (CRG), and compare against representative training-free baselines, including VCD (Leng et al., 2024), OPERA (Huang et al., 2024), PAI (Liu et al., 2024d), and VTI (Liu et al., 2025). Detailed experimental settings, model configurations, and decoding hyperparameters are provided in Appendix D.2. Additional analyses and extensions are reported in Appendix E.1 (stronger LVLMs), Appendix E.2 (combining with VCD), and Appendix A.4 (comparison with VAR-style proxies).

# 5.1. Main Results

Results on POPE As shown in Table 2, our method achieves the best accuracy and is competitive on F1 across settings. Relative to Regular, our method improves performance by 6.3% accuracy and 7.0% F1 on LLaVA-1.5-7B, 4.2% accuracy and 5.1% F1 on Qwen-VL-Chat, and 5.7% accuracy and 4.2% F1 on Qwen2.5-VL-7B-Instruct. Compared with the strongest baseline VTI, our method still yields consistent gains, improving accuracy/F1 by 1.2%/0.7% on LLaVA-1.5-7B, 2.1%/2.1% on Qwen-VL-Chat, and 1.2%/1.6% on Qwen2.5-VL-7B-Instruct.

Results on MME. As shown in Fig. 4, our method consistently improves over Regular decoding across almost all categories, with clear gains on grounding-sensitive skills such as Existence, Count, Position, and Color, while also improving higher-level categories (e.g., Commonsense Reasoning and Numerical Calculation). These results indicate that our conflict-aware suppression reduces hallucination without sacrificing general multi-modal performance.

Results on CHAIR. As shown in Table 3, our method achieves the lowest hallucination rates on both backbones: on LLaVA-1.5-7B, $C _ { S } / C _ { I }$ drop from 52.8/15.9 to 34.2/11.2 while keeping recall close to Regular (77.8 vs. 77.3); on Qwen-VL-Chat, $C _ { S } / C _ { I }$ decrease from 50.2/14.2 to 32.4/10.4 with recall improving relative to Regular (81.6 vs. 76.4). These improvements come with only a modest increase in length, indicating reduced hallucination without aggressively shortening the generation.

Table 3. Results on CHAIR benchmark. Max new tokens are set to be 512. 

<table><tr><td rowspan="2">Method</td><td colspan="4">LLaVA-1.5-7B</td><td colspan="4">Qwen-VL-Chat</td></tr><tr><td> $C_S \downarrow$ </td><td> $C_I \downarrow$ </td><td>Recall↑</td><td>Len</td><td> $C_S \downarrow$ </td><td> $C_I \downarrow$ </td><td>Recall↑</td><td>Len</td></tr><tr><td>Regular</td><td>52.8</td><td>15.9</td><td>77.3</td><td>93.4</td><td>50.2</td><td>14.2</td><td>76.4</td><td>92.1</td></tr><tr><td>VCD</td><td>51.6</td><td>14.9</td><td>77.2</td><td>101.9</td><td>47.3</td><td>13.7</td><td>75.3</td><td>98.1</td></tr><tr><td>OPERA</td><td>44.6</td><td>12.8</td><td>78.5</td><td>95.3</td><td>42.3</td><td>12.3</td><td>79.4</td><td>97.5</td></tr><tr><td>PAI</td><td>39.1</td><td>12.4</td><td>76.9</td><td>94.4</td><td>37.2</td><td>11.9</td><td>82.3</td><td>96.3</td></tr><tr><td>VTI</td><td>37.6</td><td>12.9</td><td>79.3</td><td>93.8</td><td>35.6</td><td>11.5</td><td>81.2</td><td>95.3</td></tr><tr><td>CRG</td><td>34.2</td><td>11.2</td><td>77.8</td><td>98.1</td><td>32.4</td><td>10.4</td><td>81.6</td><td>96.6</td></tr></table>

Table 4. Results on AMBER under the official evaluation pipeline (LLaVA-1.5-7B). AMBER Score is computed as (100 − CHAIR + F 1)/2. 

<table><tr><td>Method</td><td>CHAIR↓</td><td>Accuracy↑</td><td>F1↑</td><td>AMBER Score↑</td></tr><tr><td>Regular</td><td>8.3</td><td>72.7</td><td>73.7</td><td>82.70</td></tr><tr><td>CRG</td><td>4.6 (-3.7)</td><td>77.4 (+4.7)</td><td>77.5 (+3.8)</td><td>86.45 (+3.75)</td></tr></table>

Results on MMHal-Bench. Figure 5 provides a categorywise breakdown, where our method consistently dominates the radar plots across most MMHal categories for both LLaVA-1.5-7B and Qwen-VL-Chat. For detailed results of MMHal-Bench, please refer to Table 16 in Appendix E.3.

Results on AMBER. Table 4 shows that our method reduces hallucinations (CHAIR: 8.3 → 4.6) and improves discriminative performance (F1: 73.7 → 77.5) on LLaVA-1.5-7B. As a result, the AMBER Score increases by +3.75. Detailed AMBER results are provided in Table 17 (Appendix E.3).

![](images/c5e14e0d8f8257497cd054f89534c6d6c50c917cd0ed9c944db13f42ce786c24.jpg)

<details>
<summary>radar</summary>

| Metric   | Value |
| -------- | ----- |
| ATTR     | 3.5   |
| ADV      | 2.7   |
| COMP     | 2.8   |
| COUNT    | 2.6   |
| REL      | 2.9   |
| ENV      | 3.0   |
| HOL      | 3.2   |
| OTHER    | 2.8   |
| Average  | 3.1   |
</details>

![](images/a372bbf2fb6b6b633a096157d38ee9621ca8fe6a66f633af23dcad8c633d84e9.jpg)

<details>
<summary>radar</summary>

| Metric | Regular | VCD  | OPERA | Ours |
|--------|---------|------|-------|------|
| ATTR   | 1.8     | 1.9  | 1.7   | 1.8  |
| ADV    | 1.7     | 1.8  | 1.6   | 1.7  |
| COMP   | 1.6     | 1.7  | 1.5   | 1.6  |
| COUNT  | 1.5     | 1.6  | 1.4   | 1.5  |
| REL    | 1.4     | 1.5  | 1.3   | 1.4  |
| ENV    | 1.3     | 1.4  | 1.2   | 1.3  |
| HOL    | 1.2     | 1.3  | 1.1   | 1.2  |
| OTHER  | 1.1     | 1.2  | 1.0   | 1.1  |
| Average| 1.0     | 1.1  | 0.9   | 1.0  |
</details>

Figure 5. Per-category radar comparison on MMHal-Bench for two VLMs (LLaVA-1.5-7B and Qwen-VL-Chat).

Table 5. Ablation study on LLaVA-1.5-7B across four benchmarks. 

<table><tr><td>Method</td><td>POPE-Avg↑</td><td>CHAIRC $_s$ ↓</td><td>MMHal↑</td><td>MME↑</td></tr><tr><td>Regular</td><td>81.37</td><td>52.8</td><td>2.23</td><td>1640</td></tr><tr><td>CRG (A+B)</td><td>87.71</td><td>34.2</td><td>2.69</td><td>1897.42</td></tr><tr><td>w/o A</td><td>87.47</td><td>34.7</td><td>2.54</td><td>1892.18</td></tr><tr><td>w/o B</td><td>87.31</td><td>35.2</td><td>2.62</td><td>1884.32</td></tr></table>

# 5.2. Ablation Study and Hyperparameters Analysis

Conflict-Aware Strategy Selection (A/B). To identify which conflict intervention contributes most, we ablate our strategy by removing one component at a time while keeping the rest unchanged. On LLaVA-1.5-7B, we evaluate POPE, CHAIR, MMHal, and MME, comparing our full strategy (A+B) with w/o A (only Conflict B) and w/o B (only Conflict A). Table 5 shows that A+B consistently outperforms the LLaVA-1.5-7B baseline on all four benchmarks. Both ablations also improve over the baseline, but fall short of the full method, indicating that Conflict A and Conflict B provide complementary signals. Moreover, w/o A performs slightly better than w/o B in most cases, suggesting Conflict B tends to be the stronger single component in this setting.

Intervention Layer Range. We ablate the intervention layer range $\left( L _ { \mathrm { s t a r t } } , L _ { \mathrm { e n d } } \right)$ on POPE-popular for LLaVA-1.5-7B and Qwen-VL-Chat-7B by sweeping $L _ { \mathrm { s t a r t } } \in [ 5 , 2 4 ]$ and $L _ { \mathrm { e n d } } \in [ L _ { \mathrm { s t a r t } } + 7 , 3 2 ]$ . Fig. 6 shows accuracy heatmaps (star: best). LLaVA-1.5-7B peaks at (8, 19) with 0.8840, which is consistent with the trend in Fig. 3, while Qwen-VL-Chat-7B peaks at (10, 24) with 0.8763. Both models favor intervening on a contiguous mid-to-upper layer block; very shallow or very late ranges are consistently less effective.

Hyperparameter Sensitivity of k and γ. Figure 7 reports CHAIR results when varying the head budget k and the rankto-gate nonlinearity γ (lower $C _ { s } / C _ { i }$ is better; higher F1 is better). Varying k (left), increasing k generally reduces $C _ { s }$ and $C _ { i }$ and improves F1, with the best performance at a moderate budget (around k ≈ 11); larger k brings diminishing returns and slightly lowers F1. Varying $\gamma \ ( \mathrm { { r i g h t } } )$ , moderate values yield the best trade-off, with peak F1 and lowest hallucination rates around $\gamma \approx 0 . 3 – 0 . 7$ (best near 0.5), while extreme γ values are less effective.

![](images/6b8ee33953f9a0a264036da213abb181fc0e1e961947da24200b99686359e352.jpg)

<details>
<summary>heatmap</summary>

| L_end | 5    | 7    | 9    | 11   | 13   | 15   | 17   | 19   | 21   | 23   |
|-------|------|------|------|------|------|------|------|------|------|------|
| 12    |      |      |      |      |      |      |      |      |      |      |
| 14    |      |      |      |      |      |      |      |      |      |      |
| 16    |      |      |      |      |      |      |      |      |      |      |
| 18    |      |      |      |      |      |      |      |      |      |      |
| 20    |      |      |      |      |      |      |      |      |      |      |
| 22    |      |      |      |      |      |      |      |      |      |      |
| 24    |      |      |      |      |      |      |      |      |      |      |
| 26    |      |      |      |      |      |      |      |      |      |      |
| 28    |      |      |      |      |      |      |      |      |      |      |
| 30    |      |      |      |      |      |      |      |      |      |      |
| 32    |      |      |      |      |      |      |      |      |      |      |
Best (8, 19) = 0.8840
</details>

![](images/5dbfe6f40239362183b36ce28b658735da2f65c8b24c350244ad745617bd8ea7.jpg)

<details>
<summary>heatmap</summary>

| L_end | 5    | 7    | 9    | 11   | 13   | 15   | 17   | 19   | 21   | 23   |
|-------|------|------|------|------|------|------|------|------|------|------|
| 12    | 0.844|      |      |      |      |      |      |      |      |      |
| 14    |      |      |      |      |      |      |      |      |      |      |
| 16    |      |      |      |      |      |      |      |      |      |      |
| 18    |      |      |      |      |      |      |      |      |      |      |
| 20    |      |      |      |      |      |      |      |      |      |      |
| 22    |      |      |      |      |      |      |      |      |      |      |
| 24    |      |      |      |      |      |      |      |      |      | 0.8763|
| 26    |      |      |      |      |      |      |      |      |      |      |
| 28    |      |      |      |      |      |      |      |      |      |      |
| 30    |      |      |      |      |      |      |      |      |      |      |
| 32    |      |      |      |      |      |      |      |      |      |      |
The data is a heatmap of accuracy values based on the L_end column. The 'best' value for L_end = 24 is explicitly labeled as 0.8763. The 'Accuracy' values are calculated using the formula `np.exp(-x)`. The heatmap is colored based on the 'Accuracy' values.
</details>

Figure 6. Layer-range ablation for intervention (POPE Setting Popular). Heatmap shows accuracy for intervening on layers $[ L _ { \mathrm { s t a r t } } , L _ { \mathrm { e n d } } ] _ { \mathrm { : } }$ the star marks the best range.

Table 6. Runtime and peak GPU memory overhead under the same decoding setup. 

<table><tr><td>Method</td><td>Avg. Latency↓</td><td>GPU Memory↓</td></tr><tr><td>Regular</td><td>61.3 ms (×1.00)</td><td>14945 MB (×1.00)</td></tr><tr><td>VCD</td><td>123.2 ms (×2.01)</td><td>15749 MB (×1.05)</td></tr><tr><td>OPERA</td><td>435.7 ms (×7.12)</td><td>22706 MB (×1.52)</td></tr><tr><td>CRG</td><td>139.2 ms (×2.27)</td><td>14950 MB (×1.00)</td></tr></table>

# 5.3. Complexity Analysis

Runtime and memory overhead. Table 6 reports average latency and peak GPU memory under the same decoding setup. CRG incurs essentially no additional memory cost (×1.00), since we backpropagate only through lightweight gate scalars with frozen model parameters. Notably, this overhead is close to VCD (2.01×; 123.2 ms), while OPERA is substantially slower (7.12×; 435.7 ms). Latency is reported per decoding step under the same setup using Py-Torch eager attention. Additional cost analysis is provided in Appendix G.

# 6. Conclusion and Limitations

Conclusion. Hallucination in LVLMs is often driven by a mismatch between visual evidence and linguistic priors, where plausible continuations override what is supported by the image. We move beyond attention-based proxies by exposing two head-internal routes—a visual route and a text route—and measuring their decision-aligned effects on the current token via do-style interventions. This routelevel view reveals route conflict and enables a conflict-aware inference-time intervention that selectively suppresses only the prior-dominant text route while keeping the visual route intact, avoiding the indiscriminate suppression induced by coarse head rescaling. Across five benchmarks spanning discriminative and generative settings, our method consistently reduces hallucination-related errors across models with limited impact on overall multimodal performance, with a modest and controllable inference-time overhead.

![](images/edab323f5bcb95581d5fa5ecae22bd54ccee28b1d17f1f1b276664fce1fb127e.jpg)

<details>
<summary>bar_line</summary>

| k | Regular Cx | Regular Ci | Cs ↓ | Ci ↓ | F1 ↑ |
|---|---|---|---|---|---|
| 3 | 35 | 42 | 14 | 40 | 70 |
| 5 | 40 | 40 | 13 | 39 | 76 |
| 7 | 40 | 38 | 12 | 38 | 77 |
| 9 | 40 | 36 | 11 | 36 | 77 |
| 11 | 40 | 35 | 11 | 35 | 78 |
| 13 | 40 | 35 | 12 | 36 | 76 |
| 15 | 40 | 37 | 12 | 38 | 73 |
</details>

Figure 7. Hyperparameter sensitivity of k and γ.

Limitations. First, route effects are estimated using a lightweight one-forward/one-gradient approximation; the estimate is local to the current decoding step and may be noisy under ambiguous evidence or strong cross-layer interactions. More broadly, CRG is targeted at conflict-mediated hallucinations, and is not expected to correct cases where both visual and textual routes share the same bias, or errors dominated by perception, OCR, or multi-step reasoning failures. Second, the overhead grows with generation length since effect estimation and gating are applied during decoding; while the per-token cost is modest in our experiments, long-form generation remains more expensive. Third, the method requires white-box access to internal activations/gradients and inference-time patching, which may not apply to closed-source APIs or restricted deployment settings. Finally, selectively suppressing the text route can make the model more conservative, potentially reducing descriptive richness when visual evidence is weak or when the task legitimately benefits from linguistic priors; balancing faithfulness and helpfulness remains an important direction for future work. We discuss these scope and deployment boundaries in more detail in Appendix F.

# Acknowledgements

This work was supported in part by the Sichuan Science and Technology Program under Grant 2024ZYD0135, in part by the National Natural Science Foundation of China under Grants 12071372 and 12201395, and in part by the Fundamental Research Funds for the Central Universities (JBK2507005).

# Impact Statement

This paper proposes CRG, an inference-time intervention that estimates token-level, decision-aligned effects of withinhead visual and text routes and selectively suppresses priordominant text routes to improve visual grounding and reduce hallucinated claims in vision–language models. We use VRI as a lightweight ranking signal to focus interventions on a small subset of heads. The primary societal benefit is improved reliability and interpretability of model behavior in downstream applications. Potential risks include misuse to produce more convincing misinformation, uneven effects across demographic or cultural contexts, and overreliance on automated mitigation despite residual failure modes. We therefore recommend careful evaluation across diverse settings, transparent reporting of limitations, and human oversight for high-stakes deployment.

# References

Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond. CoRR, abs/2308.12966, 2023. doi: 10.48550/arXiv.2308.12966. URL https://doi. org/10.48550/arXiv.2308.12966.   
Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. Qwen2.5-VL technical report. CoRR, abs/2502.13923, 2025. doi: 10.48550/ARXIV.2 502.13923. URL https://doi.org/10.48550 /arXiv.2502.13923.   
Caffagni, D., Cocchi, F., Barsellotti, L., Moratelli, N., Sarto, S., Baraldi, L., Cornia, M., and Cucchiara, R. The revolution of multimodal large language models: A survey. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pp. 13590–13618. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.FINDINGS- ACL.807. URL https://doi.org/10.18653/v1/2024.fin dings-acl.807.   
Casacuberta, S. and Hardt, M. Good allocations from bad estimates. In The Fourteenth International Conference on Learning Representations, 2026. URL https://op enreview.net/forum?id=rxZdaKhu2I.   
Chefer, H., Gur, S., and Wolf, L. Generic attention-model explainability for interpreting bi-modal and encoderdecoder transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 397–406. IEEE, October 2021. doi: 10.1109/ICCV4892 2.2021.00045. URL https://doi.org/10.1109/ ICCV48922.2021.00045.   
Chen, J., Zhang, T., Huang, S., Niu, Y., Zhang, L., Wen, L., and Hu, X. ICT: Image-object cross-level trusted intervention for mitigating object hallucination in large vision-language models. In Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4209–4221, 2025.   
Chen, Z., Zhao, Z., Luo, H., Yao, H., Li, B., and Zhou, J. HALC: object hallucination reduction via adaptive focal-contrast decoding. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 7824–7846. PMLR, 2024. URL https://proceedi ngs.mlr.press/v235/chen24bi.html.   
Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I., and Xing, E. P. Vicuna: An open-source chatbot impressing GPT-4 with 90%\* ChatGPT quality, March 2023. URL https://lmsys.org/blog/2023-0 3-30-vicuna/.   
Deng, J. and Yang, Y. MaskCD: Mitigating LVLM hallucinations by image head masked contrastive decoding. In Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 18854–18866. Association for Computational Linguistics, November 2025. doi: 10.18653/v1/2025.findings-emnlp.1025. URL https://aclanthology.org/2025.findin gs-emnlp.1025/.   
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=Yicb FdNTTy.   
Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., and Olah, C. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021. URL https://transformer-circuits. pub/2021/framework/index.html.   
Favero, A., Zancato, L., Trager, M., Choudhary, S., Perera, P., Achille, A., Swaminathan, A., and Soatto, S. Multi-modal hallucination control by visual information grounding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 14303–14312. IEEE, 2024. doi: 10.1109/CVPR52733.2024.01356. URL https://do i.org/10.1109/CVPR52733.2024.01356.

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., Wu, Y., Ji, R., Shan, C., and He, R. MME: A comprehensive evaluation benchmark for multimodal large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. URL https://openreview.net/forum?id=DgH9 YCsqWm. Spotlight.   
Gao, Y., Chen, K., Peng, Z., Lu, H., and Xu, S. Knowledge transfer from interaction learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3585–3595, 2025.   
Geiger, A., Lu, H., Icard, T., and Potts, C. Causal abstractions of neural networks. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pp. 9574–9586, 2021.   
Geiger, A., Ibeling, D., Zur, A., Chaudhary, M., Chauhan, S., Huang, J., Arora, A., Wu, Z., Goodman, N., Potts, C., and Icard, T. Causal abstraction: A theoretical foundation for mechanistic interpretability. Journal of Machine Learning Research, 26(83):1–64, 2025. URL https://www.jm lr.org/papers/v26/23-0058.html.   
Goldowsky-Dill, N., MacLeod, C., Sato, L., and Arora, A. Localizing model behavior with path patching, 2023. URL https://doi.org/10.48550/arXiv.2 304.05969.   
Google DeepMind. Gemini 3 pro image: External model card. Model card (External Model Card), November 2025. URL https://deepmind.google/mo dels/model-cards/gemini-3-pro-image/. Accessed: 2025-12-30.   
Gunjal, A., Yin, J., and Bas, E. Detecting and preventing hallucinations in large vision language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 18135–18143, 2024. doi: 10.1609/ aaai.v38i16.29771. URL https://ojs.aaai.org /index.php/AAAI/article/view/29771.   
Hu, H., Zhang, J., Zhao, M., and Sun, Z. CIEM: Contrastive instruction evaluation method for better instruction tuning. In Workshop on Instruction Tuning and Instruction Following at NeurIPS 2023, 2023. URL https: //openreview.net/forum?id=HVduJbHSSO.   
Huang, Q., Dong, X., Zhang, P., Wang, B., He, C., Wang, J., Lin, D., Zhang, W., and Yu, N. OPERA: alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22,

2024, pp. 13418–13427. IEEE, 2024. doi: 10.1109/CV PR52733.2024.01274. URL https://doi.org/10 .1109/CVPR52733.2024.01274.   
Ilharco, G., Wortsman, M., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., and Schmidt, L. OpenCLIP. Zenodo, July 2021. URL https://doi.org/10.5281/zeno do.5143773. Version 0.1.   
Jain, S. and Wallace, B. C. Attention is not explanation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pp. 3543–3556. Association for Computational Linguistics, 2019. doi: 10.18653/V1/N19-1357. URL https://doi.org/ 10.18653/v1/n19-1357.   
Jiang, Z., Chen, J., Zhu, B., Luo, T., Shen, Y., and Yang, X. Devils in middle layers of large vision-language models: Interpreting, detecting and mitigating object hallucinations via attention lens. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pp. 25004– 25014. Computer Vision Foundation / IEEE, 2025. doi: 10.1109/CVPR52734.2025.02328.   
Jung, M., Lee, S., Kim, E., and Yoon, S. Visual attention never fades: Selective progressive attention ReCalibration for detailed image captioning in multimodal large language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 28527– 28551. PMLR, 2025. URL https://proceedings. mlr.press/v267/jung25c.html.   
Leng, S., Zhang, H., Chen, G., Li, X., Lu, S., Miao, C., and Bing, L. Mitigating object hallucinations in large visionlanguage models through visual contrastive decoding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 13872–13882. IEEE, 2024. doi: 10.1109/CV PR52733.2024.01316. URL https://doi.org/10 .1109/CVPR52733.2024.01316.   
Li, B., Zhang, K., Zhang, H., Guo, D., Zhang, R., Li, F., Zhang, Y., Liu, Z., and Li, C. LLaVA-NeXT: Stronger LLMs supercharge multimodal capabilities in the wild. LLaVA Blog, May 2024. URL https://llava-v l.github.io/blog/2024-05-10-llava-nex t-stronger-llms/. Blog post. Accessed: 2025-12- 25.   
Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, X., and Wen, J. Evaluating object hallucination in large vision-language

models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pp. 292–305. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.20. URL https:// aclanthology.org/2023.emnlp-main.20/.   
Liu, F., Lin, K., Li, L., Wang, J., Yacoob, Y., and Wang, L. Mitigating hallucination in large multi-modal models via robust instruction tuning. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024a. URL https://openreview.net/forum?id= J44HfH4JCg.   
Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 26296–26306. IEEE, 2024b. doi: 10.1109/CVPR52733.2024.02484. URL https://doi.org/10.1109/CVPR52733. 2024.02484.   
Liu, H., Xue, W., Chen, Y., Chen, D., Zhao, X., Wang, K., Hou, L., Li, R., and Peng, W. A survey on hallucination in large vision-language models. CoRR, abs/2402.00253, 2024c. doi: 10.48550/ARXIV.2402.00253. URL http s://doi.org/10.48550/arXiv.2402.00253.   
Liu, S., Zheng, K., and Chen, W. Paying more attention to image: A training-free method for alleviating hallucination in LVLMs. In Computer Vision – ECCV 2024, pp. 125–140. Springer, 2024d. doi: 10.1007/978-3-031-7 3010-8 8. URL https://doi.org/10.1007/97 8-3-031-73010-8\_8.   
Liu, S., Ye, H., and Zou, J. Reducing hallucinations in large vision-language models via latent space steering. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https: //openreview.net/forum?id=LBl7Hez0fF.   
Meng, K., Bau, D., Andonian, A., and Belinkov, Y. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, pp. 17359–17372, 2022. URL https://papers.nips.cc/paper\_files/p aper/2022/hash/6f1d43d5a82a37e89b066 5b33bf3a182-Abstract-Conference.html.   
Michel, P., Levy, O., and Neubig, G. Are sixteen heads really better than one? In Advances in Neural Information Processing Systems 32: Annual Conference on Neural

Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 14014– 14024, 2019. URL https://proceedings.neur ips.cc/paper/2019/hash/2c601ad9d2ff9 bc8b282670cdd54f69f-Abstract.html.   
Nam, A., Conklin, H., Yang, Y., Griffiths, T., Cohen, J. D., and Leslie, S.-J. Causal head gating: A framework for interpreting roles of attention heads in transformers. In Advances in Neural Information Processing Systems 38, 2025. URL https://papers.nips.cc/paper \_files/paper/2025/hash/a7f530e11fa19 e9551b7a51dbd0f336f-Abstract-Confere nce.html.   
Qian, J., Zheng, G., Zhu, Y., and Yang, S. Intervene-All-Paths: Unified mitigation of LVLM hallucinations across alignment formats. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id= HRBhNqkG03.   
Qiu, Z., Wang, Z., Zheng, B., Huang, Z., Wen, K., Yang, S., Men, R., Yu, L., Huang, F., Huang, S., Liu, D., Zhou, J., and Lin, J. Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free. In Advances in Neural Information Processing Systems 38, 2025. URL https://papers.nips.cc/paper files/paper/2025/hash/904e89bb4e632 e75fb47f093b620b257-Abstract-Confere nce.html.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pp. 8748–8763. PMLR, 2021. URL http://proceedings.mlr.press/v139/rad ford21a.html.   
Rohrbach, A., Hendricks, L. A., Burns, K., Darrell, T., and Saenko, K. Object hallucination in image captioning. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pp. 4035–4045. Association for Computational Linguistics, 2018. doi: 10.18653/V1/D18-1437. URL https: //doi.org/10.18653/v1/d18-1437.   
Sarkar, S., Che, Y., Gavin, A., Beerel, P. A., and Kundu, S. Mitigating hallucinations in vision-language models through image-guided head suppression. In Proceedings of the 2025 Conference on Empirical Methods in Natural

Language Processing, pp. 12481–12500, Suzhou, China, November 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.emnlp-main.631. URL https://aclanthology.org/2025.emnlp-m ain.631/.   
Serrano, S. and Smith, N. A. Is attention interpretable? In Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pp. 2931–2951. Association for Computational Linguistics, 2019. doi: 10.18653/V1/P19-1282. URL https: //doi.org/10.18653/v1/p19-1282.   
Sharkey, L., Chughtai, B., Batson, J., Lindsey, J., Wu, J., Bushnaq, L., Goldowsky-Dill, N., Heimersheim, S., Ortega, A., Bloom, J. I., Biderman, S., Garriga-Alonso, A., Conmy, A., Nanda, N., Rumbelow, J. M., Wattenberg, M., Schoots, N., Miller, J., Saunders, W., Michaud, E. J., Casper, S., Tegmark, M., Bau, D., Todd, E., Geiger, A., Geva, M., Hoogland, J., Murfet, D., and Mc-Grath, T. Open problems in mechanistic interpretability. Trans. Mach. Learn. Res., 2025, 2025. URL https: //openreview.net/forum?id=91H76m9Z94.   
Sun, Z., Shen, S., Cao, S., Liu, H., Li, C., Shen, Y., Gan, C., Gui, L., Wang, Y., Yang, Y., Keutzer, K., and Darrell, T. Aligning large multimodal models with factually augmented RLHF. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pp. 13088– 13110. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.FINDINGS-ACL.775. URL https://doi.org/10.18653/v1/2024.fin dings-acl.775.   
Sundararajan, M., Taly, A., and Yan, Q. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pp. 3319– 3328. PMLR, 2017. URL http://proceedings. mlr.press/v70/sundararajan17a.html.   
Tang, F., Liu, C., Xu, Z., Hu, M., Huang, Z., Xue, H., Chen, Z., Peng, Z., Yang, Z., Zhou, S., Li, W., Li, Y., Song, W., Su, S., Feng, W., Su, J., Lin, M., Peng, Y., Cheng, X., Razzak, I., and Ge, Z. Seeing far and clearly: Mitigating hallucinations in MLLMs with attention causal decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26147–26159, June 2025. doi: 10.1109/CVPR52734.20 25.02435. URL https://openaccess.thecvf. com/content/CVPR2025/html/Tang\_Seein g\_Far\_and\_Clearly\_Mitigating\_Halluci

nations\_in\_MLLMs\_with\_Attention\_CVPR \_2025\_paper.html.   
Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., and Xie, S. Eyes wide shut? exploring the visual shortcomings of multimodal LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9568–9578, June 2024. URL https:// openaccess.thecvf.com/content/CVPR20 24/html/Tong\_Eyes\_Wide\_Shut\_Explorin g\_the\_Visual\_Shortcomings\_of\_Multimo dal\_LLMs\_CVPR\_2024\_paper.html.   
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pp. 5998–6008, 2017. URL https://proceedings.neurips.cc/paper /2017/hash/3f5ee243547dee91fbd053c1c 4a845aa-Abstract.html.   
Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Sakenis, S., Huang, J., Singer, Y., and Shieber, S. Causal mediation analysis for interpreting neural NLP: The case of gender bias. CoRR, abs/2004.12265, 2020a. doi: 10.4 8550/arXiv.2004.12265. URL https://doi.org/ 10.48550/arXiv.2004.12265.   
Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., and Shieber, S. M. Investigating gender bias in language models using causal mediation analysis. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, pp. 12388–12401, 2020b. URL https://proceedings.neurips.cc/paper /2020/hash/92650b2e92217715fe312e6fa 7b90d82-Abstract.html.   
Voita, E., Talbot, D., Moiseev, F., Sennrich, R., and Titov, I. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 5797–5808, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/V1/P19- 1580. URL https://aclanthology.org/P19-1580/.   
Wan, Z., Zhang, C., Yong, S., Ma, M. Q., Stepputtis, S., Morency, L.-P., Ramanan, D., Sycara, K., and Xie, Y. ONLY: One-layer intervention sufficiently mitigates hallucinations in large vision-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3225–3234, 2025.

Wang, J., Wang, Y., Xu, G., Zhang, J., Gu, Y., Jia, H., Wang, J., Xu, H., Yan, M., Zhang, J., and Sang, J. AMBER: An LLM-free multi-dimensional benchmark for MLLMs hallucination evaluation. CoRR, abs/2311.07397, 2023a. doi: 10.48550/arXiv.2311.07397. URL https://doi. org/10.48550/arXiv.2311.07397.   
Wang, J., Zhou, Y., Xu, G., Shi, P., Zhao, C., Xu, H., Ye, Q., Yan, M., Zhang, J., Zhu, J., Sang, J., and Tang, H. Evaluation and analysis of hallucination in large visionlanguage models. CoRR, abs/2308.15126, 2023b. doi: 10.48550/ARXIV.2308.15126. URL https://doi. org/10.48550/arXiv.2308.15126.   
Wu, Z., Niu, Y., Gao, H., Lin, M., Zhang, Z., Zhang, Z., Shi, Q., Wang, Y., Fu, S., Xu, J., Ao, J., Dai, E., Feng, L., Zhang, X., and Wang, S. LanP: Rethinking the impact of language priors in large vision-language models. CoRR, abs/2502.12359, 2025. doi: 10.48550/ARXIV.2502.12 359. URL https://doi.org/10.48550/arXiv .2502.12359.   
Yang, T., Li, Z., Cao, J., and Xu, C. Understanding and mitigating hallucination in large vision-language models via modular attribution and intervention. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum ?id=Bjq4W7P2Us.   
Ye, Q., Xu, H., Xu, G., Ye, J., Yan, M., Zhou, Y., Wang, J., Hu, A., Shi, P., Shi, Y., Li, C., Xu, Y., Chen, H., Tian, J., Qi, Q., Zhang, J., Huang, F., and Zhou, J. mPLUG-Owl: Modularization empowers large language models with multimodality. CoRR, abs/2304.14178, 2023. doi: 10.48550/arXiv.2304.14178. URL https://doi.or g/10.48550/arXiv.2304.14178.   
Yin, J., Chen, Q., Chen, K., Zhou, J., Wu, X., and He, L. Dynamic multimodal activation steering for hallucination mitigation in large vision-language models. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net /forum?id=YtWZdwEG5K.   
Yin, S., Fu, C., Zhao, S., Li, K., Sun, X., Xu, T., and Chen, E. A survey on multimodal large language models. National Science Review, 11(12):nwae403, 2024a. doi: 10.1093/nsr/nwae403. URL https://doi.org/10 .1093/nsr/nwae403.   
Yin, S., Fu, C., Zhao, S., Xu, T., Wang, H., Sui, D., Shen, Y., Li, K., Sun, X., and Chen, E. Woodpecker: Hallucination correction for multimodal large language models. Science China Information Sciences, 67(12):220105, 2024b. doi: 10.1007/s11432-024-4251-x. URL https://doi.or g/10.1007/s11432-024-4251-x.

Yu, L., Chen, Z., Kuang, P., Feng, Z., Zhou, F., Wang, L., and Dobbie, G. Causally-grounded dual-path attention intervention for object hallucination mitigation in LVLMs. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 36021–36029, 2026. doi: 10.1609/aaai.v40i42.40918. URL https://ojs.aaai.org/index.php/AAAI/ article/view/40918.   
Yuksekg ¨ on¨ ul, M., Chandrasekaran, V., Jones, E., Gunasekar, ¨ S., Naik, R., Palangi, H., Kamar, E., and Nushi, B. Attention satisfies: A constraint-satisfaction lens on factual errors of language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id= gfFVATffPd.   
Zhang, C., Wan, Z., Kan, Z., Ma, M. Q., Stepputtis, S., Ramanan, D., Salakhutdinov, R., Morency, L.-P., Sycara, K. P., and Xie, Y. Self-correcting decoding with generative feedback for mitigating hallucinations in large visionlanguage models. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=tTBXePRKSx.   
Zhang, F. and Nanda, N. Towards best practices of activation patching in language models: Metrics and methods. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https: //openreview.net/forum?id=Hf17y6u9BC.   
Zheng, H. and Zhang, Z. Modality bias in LVLMs: Analyzing and mitigating object hallucination via attention lens. CoRR, abs/2508.02419, 2025. doi: 10.48550/ARX IV.2508.02419. URL https://doi.org/10.485 50/arXiv.2508.02419.   
Zhong, L., He, Z., Zheng, J., Li, J., Wang, Z. J., and Kang, X. AdaIAT: Adaptively increasing attention to generated text to alleviate hallucinations in LVLM. CoRR, abs/2603.04908, 2026. doi: 10.48550/arXiv.2603.04908. URL https://doi.org/10.48550/arXiv.2 603.04908.   
Zhou, G., Yan, Y., Zou, X., Wang, K., Liu, A., and Hu, X. Mitigating modality prior-induced hallucinations in multimodal large language models via deciphering attention causality. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=AV7OXVlAyi.

# Appendix

# Contents

# A Analysis of VAR-Style Attention Proxies 16

A.1 CRE vs. VAR-Style Proxies 16   
A.2 Theoretical Analysis: What VAR Can (and Cannot) Guarantee 16   
A.3 Three Canonical Failure Modes of VAR 19   
A.4 Empirical Alignment of VAR with Decision-Aligned Effects . . 19

# B Validity of the First-Order Do-Effect Approximation . . 20

B.1 First-Order Error Bound 20   
B.2 Sign Reliability . 21   
B.3 Exact Validation on a Small Subset . 22

# C Proofs of Main Results 23

# D Implementation Details . 24

D.1 Benchmark Details 24   
D.2 Models and Experiments Setup 25   
D.3 Choice of The Scalar Objective ℓ . 26

# E Additional Experiments .27

E.1 Results on More Advanced LVLMs . 27   
E.2 Implementation with VCD 27   
E.3 Detailed Results of AMBER & MMHal-Bench 28   
E.4 Supplementary POPE Comparisons with Recent Methods . 29

# F Limitations and More Discussion . 29

F.1 Scope Boundary of CRG 29   
F.2 Cross-Gate Interactions 30   
F.3 Per-Token CRG in Open-Ended Generation 31

# G Computational Cost Analysis 31

# H Case Studies 32

# A. Analysis of VAR-Style Attention Proxies

# A.1. CRE vs. VAR-Style Proxies

The main text argues that attention-mass proxies, such as VAR (Jiang et al., 2025), are not decision-aligned for diagnosing and intervening on hallucination behaviors in LVLMs. In our framework, the target of interest is the decision score ℓ (defined in the main text for discriminative and generative settings), and we quantify how each head-route causally changes ℓ via dointerventions. Concretely, for head $( l , h )$ , we define the visual-route and textual-route CRE as $\Delta _ { l , h } ^ { \mathrm { v i s } } : = \ell _ { l , h } ( 1 , 1 ) - \ell _ { l , h } ( 0 , 1 )$ and $\Delta _ { l , h } ^ { \mathrm { t x t } } : = \ell _ { l , h } ( 1 , 1 ) - \ell _ { l , h } ( 1 , 0 )$ , where $\ell _ { l , h } ( g _ { \mathrm { v i s } } , g _ { \mathrm { t x t } } )$ denotes the score under route gates $( g _ { \mathrm { v i s } } , g _ { \mathrm { t x t } } ) \in \{ 0 , 1 \} ^ { 2 }$ . We further normalize the relative reliance via the VRIl,h : $\begin{array} { r } { \mathrm { V R I } _ { l , h } : = \frac { \lvert \Delta _ { l , h } ^ { \mathrm { v i s } } \rvert } { \lvert \Delta _ { l , h } ^ { \mathrm { v i s } } \rvert + \lvert \Delta _ { l , h } ^ { \mathrm { t x t } } \rvert + \varepsilon } } \end{array}$ = |∆visl,h|+|∆txtl,h|+ε , where ε > 0 is a small constant for numerical |∆visl,h| $\varepsilon > 0$ stability.

In contrast, VAR summarizes only the amount of attention mass assigned to visual tokens. Let $I _ { \mathrm { v i s } }$ be the index set of visual tokens, and let $\alpha _ { i j } ^ { ( l , h ) }$ denote the attention weight from query position i to key position $j$ at head $( l , h )$ . A typical VAR definition takes the form $\begin{array} { r } { \mathrm { V A R } _ { l , h } ( i ) : = \sum _ { j \in I _ { \mathrm { v i s } } } \alpha _ { i j } ^ { ( l , h ) } } \end{array}$ , optionally averaged over query positions i (e.g., over decoded positions or over all tokens in the prefix). Crucially, $\mathrm { V A R } _ { l , h }$ does not condition on (i) the content carried by value vectors, nor (ii) how such content aligns with the local sensitivity of the score ℓ. Therefore, VAR is inherently a correlational proxy: it measures where the head attends, but not whether the attended information supports (or opposes) the current decision.

# A.2. Theoretical Analysis: What VAR Can (and Cannot) Guarantee

We formalize the limitation of VAR by connecting it to a first-order, decision-aligned approximation of route effects. Let $v _ { i } ^ { ( l , h ) }$ be the value vector at key position j for head (l, h), and define the visual-route output as Ovisl,h(i) := Pj∈Ivis $j$ $( l , h )$ $\begin{array} { r } { O _ { l , h } ^ { \mathrm { v i s } } ( i ) : = \sum _ { j \in I _ { \mathrm { v i s } } } \alpha _ { i j } ^ { ( l , h ) } v _ { j } ^ { \top { l } , { h } ) } } \end{array}$ αij . Let $G _ { l , h } ( i )$ denote the local sensitivity (gradient-like) vector used in the main text, so that the first-order approximation of the visual-route effect can be written as $\widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } ( i ) : = \langle G _ { l , h } ( i ) , O _ { l , h } ^ { \mathrm { v i s } } ( i ) \rangle$ . Expanding the definition yields $\widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } ( i ) =$ Pj∈Ivis α(l,h)ij $\begin{array} { r l } { \sum _ { j \in I _ { \mathrm { v i s } } } \alpha _ { i j } ^ { ( l , h ) } \langle G _ { l , h } ( i ) , v _ { j } ^ { ( l , h ) } \rangle } & { { } } \end{array}$ , which makes expned alignment terms $\langle G , v \rangle$ at the decision-aligned quantity depends not only on attention.

Lemma A.1 (VAR provides only a coarse upper bound). Define $\begin{array} { r l r } { m _ { l , h } ( i ) } & { : = } & { \operatorname* { m a x } _ { j \in I _ { \mathrm { v i s } } } \left| \langle G _ { l , h } ( i ) , v _ { j } ^ { ( l , h ) } \rangle \right| } \end{array}$ . Then the first-order visual-route effect satisfies

$$
\left| \widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} (i) \right| \leq m _ {l, h} (i) \cdot \operatorname{VAR} _ {l, h} (i).
$$

Proof. Recall that the first-order visual-route effect can be written as a weighted sum over visual keys:

$$
\widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} (i) = \sum_ {j \in I _ {\mathrm{vis}}} \alpha_ {i j} ^ {(l, h)} \langle G _ {l, h} (i), v _ {j} ^ {(l, h)} \rangle .
$$

Taking the absolute value and applying the triangle inequality yields

$$
\left| \widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} (i) \right| = \Big | \sum_ {j \in I _ {\mathrm{vis}}} \alpha_ {i j} ^ {(l, h)} \left\langle G _ {l, h} (i), v _ {j} ^ {(l, h)} \right\rangle \Big | \leq \sum_ {j \in I _ {\mathrm{vis}}} \alpha_ {i j} ^ {(l, h)} \left| \left\langle G _ {l, h} (i), v _ {j} ^ {(l, h)} \right\rangle \right|.
$$

By the definition of $m _ { l , h } ( i )$ , for every $j \in I _ { \mathrm { v i s } }$ we have

$$
\left| \langle G _ {l, h} (i), v _ {j} ^ {(l, h)} \rangle \right| \leq m _ {l, h} (i).
$$

Substituting this bound into the previous inequality gives

$$
\sum_ {j \in I _ {\mathrm{vis}}} \alpha_ {i j} ^ {(l, h)} \left| \langle G _ {l, h} (i), v _ {j} ^ {(l, h)} \rangle \right| \leq m _ {l, h} (i) \sum_ {j \in I _ {\mathrm{vis}}} \alpha_ {i j} ^ {(l, h)}.
$$

Finally, noting that Pj∈Ivis $\begin{array} { r } { \sum _ { j \in I _ { \mathrm { v i s } } } \alpha _ { i j } ^ { ( l , h ) } = \mathrm { V A R } _ { l , h } ( i ) } \end{array}$ αij by definition, we obtain

$$
\left| \widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} (i) \right| \leq m _ {l, h} (i) \cdot \operatorname{VAR} _ {l, h} (i),
$$

which completes the proof.

Implication. Lemma A.1 clarifies that VAR can at best constrain the magnitude of a decision-aligned effect up to an unknown, head- and context-dependent factor $m _ { l , h } ( i )$ . However, $m _ { l , h } ( i )$ depends on value content and its alignment with the score sensitivity, which is not observable from attention weights alone. Consequently, ranking or selecting heads solely by $\mathrm { V A R } _ { l , h }$ is not theoretically justified as a decision-aligned criterion, and can be unreliable especially when $\langle G , v \rangle$ varies widely or changes sign across tokens.

Lemma A.2 (A tighter second-moment upper bound). For $j \in I _ { \mathrm { v i s } } ,$ define the (signed) alignment $a _ { j } : = \langle G _ { l , h } ( i ) , v _ { j } ^ { ( l , h ) } \rangle$ . When $\mathrm { V A R } _ { l , h } ( i ) > 0 ,$ , define the normalized visual attention weights $\begin{array} { r } { \tilde { \alpha } _ { i j } ^ { ( l , h ) } : = \frac { \alpha _ { i j } ^ { ( l , h ) } } { \mathrm { V A R } _ { l , h } ( i ) } , j \in I _ { \mathrm { v i s } } } \end{array}$ , so that $\begin{array} { r } { \sum _ { j \in I _ { \mathrm { v i s } } } \tilde { \alpha } _ { i j } ^ { ( l , h ) } = } \end{array}$ 1. Define the (attention-weighted) second-moment factor $\begin{array} { r } { s _ { l , h } ( i ) ~ : = ~ \left( \sum _ { j \in I _ { \mathrm { v i s } } } \tilde { \alpha } _ { i j } ^ { ( l , h ) } a _ { j } ^ { 2 } \right) ^ { 1 / 2 } } \end{array}$  Pj∈Ivis . Then the first-order visualroute effect satisfies

$$
\left| \widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} (i) \right| \leq \operatorname{VAR} _ {l, h} (i) \cdot s _ {l, h} (i) \leq \operatorname{VAR} _ {l, h} (i) \cdot m _ {l, h} (i),
$$

where $m _ { l , h } ( i ) : = \operatorname* { m a x } _ { j \in I _ { \mathrm { v i s } } } | a _ { j } |$ as in Lemma A.1.

Proof. Starting from the definition of the first-order effect,

$$
\widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} (i) = \sum_ {j \in I _ {\mathrm{vis}}} \alpha_ {i j} ^ {(l, h)} a _ {j} = \mathrm{VAR} _ {l, h} (i) \sum_ {j \in I _ {\mathrm{vis}}} \tilde {\alpha} _ {i j} ^ {(l, h)} a _ {j}.
$$

Taking absolute values yields

$$
\left| \widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} (i) \right| = \operatorname{VAR} _ {l, h} (i) \Big | \sum_ {j \in I _ {\mathrm{vis}}} \tilde {\alpha} _ {i j} ^ {(l, h)} a _ {j} \Big |.
$$

Applying the Cauchy–Schwarz inequality to the weighted sum gives

$$
\Big | \sum_ {j \in I _ {\mathrm{vis}}} \tilde {\alpha} _ {i j} ^ {(l, h)} a _ {j} \Big | \leq \Big (\sum_ {j \in I _ {\mathrm{vis}}} \tilde {\alpha} _ {i j} ^ {(l, h)} \Big) ^ {1 / 2} \Big (\sum_ {j \in I _ {\mathrm{vis}}} \tilde {\alpha} _ {i j} ^ {(l, h)} a _ {j} ^ {2} \Big) ^ {1 / 2}.
$$

Since Pj∈Ivis $\begin{array} { r } { \sum _ { j \in I _ { \mathrm { v i s } } } \tilde { \alpha } _ { i j } ^ { ( l , h ) } = 1 } \end{array}$ α˜ij , the first factor equals 1, and we obtain

$$
\Big | \sum_ {j \in I _ {\mathrm{vis}}} \tilde {\alpha} _ {i j} ^ {(l, h)} a _ {j} \Big | \leq \Big (\sum_ {j \in I _ {\mathrm{vis}}} \tilde {\alpha} _ {i j} ^ {(l, h)} a _ {j} ^ {2} \Big) ^ {1 / 2} = s _ {l, h} (i).
$$

Therefore,

$$
\left| \widehat {\Delta} _ {l, h} ^ {\text { vis }} (i) \right| \leq \operatorname{VAR} _ {l, h} (i) \cdot s _ {l, h} (i).
$$

Finally, because $| a _ { j } | \leq m _ { l , h } ( i )$ for all $j \in I _ { \mathrm { v i s } }$ , we have $a _ { j } ^ { 2 } \leq m _ { l , h } ( i ) ^ { 2 }$ , and hence

$$
s _ {l, h} (i) ^ {2} = \sum_ {j \in I _ {\mathrm{vis}}} \tilde {\alpha} _ {i j} ^ {(l, h)} a _ {j} ^ {2} \leq m _ {l, h} (i) ^ {2} \sum_ {j \in I _ {\mathrm{vis}}} \tilde {\alpha} _ {i j} ^ {(l, h)} = m _ {l, h} (i) ^ {2},
$$

Remark A.3. Lemma A.2 sharpens Lemma A.1 by replacing the worst-case factor $m _ { l , h } ( i )$ with a data-dependent RMS factor $s _ { l , h } ( \boldsymbol { i } )$ , which can be much smaller when only a few attended visual tokens have large alignment magnitude.

Proposition A.4 (When VAR becomes informative: sign-consistent alignment). Assume $\mathrm { V A R } _ { l , h } ( i ) > 0 .$ . For $j \in I _ { \mathrm { v i s } } ,$ , let $a _ { j } \ : = \ \langle G _ { l , h } ( i ) , v _ { j } ^ { ( l , h ) } \rangle$ . Suppose that $\{ a _ { j } \} _ { j \in I _ { \mathrm { v i s } } }$ is sign-consistent, i.e., either $a _ { j } \geq 0 f o r$ all $j \in I _ { \mathrm { v i s } }$ or aj ≤ 0 for all $j \in I _ { \mathrm { v i s } }$ . Moreover, assume that there exist scalars $\underline { { a } } _ { l , h } ( i ) \leq \overline { { a } } _ { l , h } ( i )$ such that $a _ { j } \in \left[ \underline { { a } } _ { l , h } ( i ) , \overline { { a } } _ { l , h } ( i ) \right]$ , for all $j \in I _ { \mathrm { v i s } }$ .

Then the first-order visual-route effect obeys the two-sided bound

$$
\underline {{a}} _ {l, h} (i) \operatorname{VAR} _ {l, h} (i) \leq \widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} (i) \leq \overline {{a}} _ {l, h} (i) \operatorname{VAR} _ {l, h} (i).
$$

Proof. By definition, when $\mathrm { V A R } _ { l , h } ( i ) > 0$ the normalized visual attention weights

$$
\tilde {\alpha} _ {i j} ^ {(l, h)} := \frac {\alpha_ {i j} ^ {(l , h)}}{\mathrm{VAR} _ {l , h} (i)}, \qquad j \in I _ {\mathrm{vis}},
$$

satisfy Pj∈Ivis $\begin{array} { r } { \sum _ { j \in I _ { \mathrm { v i s } } } \tilde { \alpha } _ { i j } ^ { ( l , h ) } = 1 } \end{array}$ α˜ij and $\tilde { \alpha } _ { i j } ^ { ( l , h ) } \geq 0$ . Using the expansion of the first-order effect,

$$
\widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} (i) = \sum_ {j \in I _ {\mathrm{vis}}} \alpha_ {i j} ^ {(l, h)} a _ {j} = \operatorname{VAR} _ {l, h} (i) \sum_ {j \in I _ {\mathrm{vis}}} \tilde {\alpha} _ {i j} ^ {(l, h)} a _ {j}.
$$

The quantity Pj∈Ivis $\begin{array} { r } { \sum _ { j \in I _ { \mathrm { v i s } } } \tilde { \alpha } _ { i j } ^ { ( l , h ) } a _ { j } } \end{array}$ is a convex combination of $\{ a _ { j } \} _ { j \in I _ { \mathrm { v i s } } }$ . Since each $a _ { j } \ \in \ [ \underline { { a } } _ { l , h } ( i ) , \overline { { a } } _ { l , h } ( i ) ]$ , any convex combination also lies in the same interval, namely

$$
\underline {{a}} _ {l, h} (i) \leq \sum_ {j \in I _ {\mathrm{vis}}} \tilde {\alpha} _ {i j} ^ {(l, h)} a _ {j} \leq \overline {{a}} _ {l, h} (i).
$$

Multiplying both sides by the positive scalar $\mathrm { V A R } _ { l , h } ( i )$ yields

$$
\underline {{a}} _ {l, h} (i) \operatorname{VAR} _ {l, h} (i) \leq \widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} (i) \leq \overline {{a}} _ {l, h} (i) \operatorname{VAR} _ {l, h} (i),
$$

which completes the proof.

![](images/5fa9be6ba1078e52bd63092e1cd68352ace8ae25e60a7902e5d89c2525cf3121.jpg)

Remark A.5. This proposition formalizes a sufficient condition under which VAR may correlate with decision-aligned effect: the attended visual values must be roughly aligned with the score sensitivity (no sign flips or severe cancellations). Outside this regime, large VAR does not imply large $\tilde { \Delta } ^ { \mathrm { v i s } }$ (cf. Appendix A.3).

Theorem A.6 (Non-identifiability of decision-aligned influence from attention alone). Fix a head $( l , h )$ and a query position i. Assume that the attention weights $\{ \alpha _ { i j } ^ { ( l , h ) } \} _ { j }$ are fixed (equivalently, the queries and keys are fixed), and let

$$
\operatorname{VAR} _ {l, h} (i) := \sum_ {j \in I _ {\mathrm{vis}}} \alpha_ {i j} ^ {(l, h)}.
$$

Let $\Vert v _ { j } ^ { ( l , h ) } \Vert _ { 2 } ~ \leq ~ V$ $G _ { l , h } ( i )$ be any nonzero sensitivity vector with for all $j \in I _ { \mathrm { v i s } }$ , where $V \ > \ 0 .$ . Then there exist two feasible value assignments $\Vert G _ { l , h } ( i ) \Vert _ { 2 } > 0$ . Suppose the value vectors are norm-bounded as $\{ v _ { j } ^ { ( l , h ) } \} _ { j \in { I _ { \mathrm { v i s } } } }$ and $\{ v _ { j } ^ { \prime ( l , h ) } \} _ { j \in { I _ { \mathrm { v i s } } } }$ that yield the same attention weights $\alpha _ { i j } ^ { ( l , h ) }$ but produce opposite first-order visual-route effects:

$$
\widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} (i) = + V \| G _ {l, h} (i) \| _ {2} \operatorname{VAR} _ {l, h} (i), \quad \widehat {\Delta} _ {l, h} ^ {\mathrm{vis} t} (i) = - V \| G _ {l, h} (i) \| _ {2} \operatorname{VAR} _ {l, h} (i).
$$

Consequently, any proxy that depends only on attention allocation—including attention mass ratios such as VAR—cannot, in general, identify the sign of the decision-aligned visual influence.

Proof. Let $\begin{array} { r } { u : = \frac { G _ { l , h } ( i ) } { \Vert G _ { l , h } ( i ) \Vert _ { 2 } } } \end{array}$ , so that $\| u \| _ { 2 } = 1$ . Define two value assignments on visual keys by setting, for all $j \in I _ { \mathrm { v i s } }$

$$
v _ {j} ^ {(l, h)} := V u, \quad v _ {j} ^ {\prime (l, h)} := - V u.
$$

Both assignments satisfy the norm constraint $\| v _ { j } ^ { ( l , h ) } \| _ { 2 } = \| v _ { j } ^ { \prime ( l , h ) } \| _ { 2 } = V$ . Moreover,

$$
\langle G _ {l, h} (i), v _ {j} ^ {(l, h)} \rangle = \langle G _ {l, h} (i), V u \rangle = V \| G _ {l, h} (i) \| _ {2},
$$

$$
\langle G _ {l, h} (i), v _ {j} ^ {\prime (l, h)} \rangle = - V \| G _ {l, h} (i) \| _ {2}.
$$

Using the definition of the first-order visual-route effect,

$$
\widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} (i) = \sum_ {j \in I _ {\mathrm{vis}}} \alpha_ {i j} ^ {(l, h)} \langle G _ {l, h} (i), v _ {j} ^ {(l, h)} \rangle
$$

$$
= V \| G _ {l, h} (i) \| _ {2} \sum_ {j \in I _ {\mathrm{vis}}} \alpha_ {i j} ^ {(l, h)}
$$

$$
= V \| G _ {l, h} (i) \| _ {2} \operatorname{VAR} _ {l, h} (i),
$$

Similarly,

$$
\begin{array}{l} \widehat {\Delta} _ {l, h} ^ {\mathrm{vis} \prime} (i) = \sum_ {j \in I _ {\mathrm{vis}}} \alpha_ {i j} ^ {(l, h)} \left\langle G _ {l, h} (i), v _ {j} ^ {\prime (l, h)} \right\rangle \\ = - V \| G _ {l, h} (i) \| _ {2} \operatorname{VAR} _ {l, h} (i). \\ \end{array}
$$

Since both constructions keep αij $\alpha _ { i j } ^ { ( l , h ) }$ (l,h) unchanged while flipping the sign of the induced effect, the sign of $\widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } ( i )$ is not identifiable from attention allocation alone. □

Corollary A.7. Even if two heads have identical $\mathrm { V A R } _ { l , h } ( i )$ , their decision-aligned visual effects can have opposite signs and differ by $2 V \Vert G _ { l , h } ( i ) \Vert _ { 2 } \mathrm { V A R } _ { l , h } ( i )$ . Hence, selecting heads based solely on VAR admits arbitrarily bad mistakes under adversarial (but norm-bounded) value alignments.

# A.3. Three Canonical Failure Modes of VAR

We present three minimal constructions showing that large (or small) VAR does not imply large (or small) decision-aligned visual influence. For simplicity, we fix a head (l, h) and a query position i, and consider visual keys $j \in I _ { \mathrm { v i s } }$ only.

1. Orthogonality (large VAR, zero effect). Assume $\mathrm { V A R } _ { l , h } ( i ) = 1$ , i.e., all attention mass is assigned to visual tokens, but $\langle G _ { l , h } ( i ) , v _ { i } ^ { ( l , h ) } \rangle = 0$ for every $j \in I _ { \mathrm { v i s } }$ . Then $\begin{array} { r } { \widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } ( i ) = \sum _ { i \in I _ { \mathrm { v i s } } } \alpha _ { i j } ^ { ( l , h ) } \cdot 0 = 0 } \end{array}$ aij α(l,h)ij · 0 = 0. Thus, a head may appear “highly   
$j _ { 1 } , j _ { 2 } \in I _ { \mathrm { v i s } }$ $\alpha _ { i j _ { 1 } } ^ { ( l , h ) } =$ $\alpha _ { i j _ { 2 } } ^ { ( l , h ) } = 1 / 2 $ αij2 ) = 1/2, so VARl,h(i) = 1. Let ⟨Gl,h(i), v(l,hj1 $\mathrm { V A R } _ { l , h } ( i ) = 1$ $\langle G _ { l , h } ( i ) , v _ { j _ { 1 } } ^ { ( l , h ) } \rangle = + 1$ and $\langle G _ { l , h } ( i ) , v _ { j _ { 2 } } ^ { ( l , h ) } \rangle = - 1$ . Then $\begin{array} { r } { \widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } ( i ) = \frac { 1 } { 2 } ( + 1 ) + } \end{array}$ $\textstyle { \frac { 1 } { 2 } } ( - 1 ) = 0$ . Hence, even when VAR is maximal, the net decision-aligned effect can vanish due to signed cancellations.   
3. Harmful visual content (large VAR, negative effect). VAR is nonnegative by construction and cannot encode whether the attended visual content supports or contradicts the target token. Suppose $\mathrm { V A R } _ { l , h } ( i )$ is large, but $\langle G _ { l , h } ( i ) , v _ { j } ^ { ( l , h ) } \rangle < 0$ for the dominant attended visual keys j. Then $\widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } ( i )$ becomes negative, meaning the visual route reduces the score ℓ for the current token (i.e., it provides counter-evidence). Such “harmful” visual influence is particularly relevant under route conflicts, yet it is invisible to VAR-based diagnostics.

Table 7. A summary of canonical (counterexample) regimes showing that attention-mass proxies such as VAR are not decision-aligned. Here $a _ { j } = \langle G _ { l , h } ( i ) , v _ { j } ^ { ( l , h ) } \rangle , \tilde { \alpha } _ { i j } = \alpha _ { i j } / \mathrm { V A R } _ { l , h } ( i )$ , and $\textstyle \mu : = \sum _ { j \in I _ { \mathrm { v i s } } } \tilde { \alpha } _ { i j } a _ { j }$ so that $\widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } ( i ) = \mathrm { V A R } _ { l , h } ( i ) \cdot \mu$ . 

<table><tr><td></td><td>High VAR (VAR $_{l,h}$ (i) ≈ 1)</td><td>Low VAR (VAR $_{l,h}$ (i) ≈ 0)</td></tr><tr><td>Near-zero alignment ( $\mu$  ≈ 0)</td><td>Orthogonality:  $a_j = 0 \forall j$ Cancellation: sign-mixed  $a_j$  $\Rightarrow \widehat{\Delta}^{\text{vis}}$  ≈ 0 even if VAR is large</td><td>Trivial small upper bound: $|\widehat{\Delta}^{\text{vis}}| \leq \text{VAR} \cdot m$  $\Rightarrow$ small effect is possible but uninformative</td></tr><tr><td>Negative alignment ( $\mu < 0$ )</td><td>Harmful visual content: dominant  $a_j < 0$  $\Rightarrow \widehat{\Delta}^{\text{vis}} < 0$  despite high VAR(VAR cannot reveal the sign)</td><td>Less common but possible:small VAR yet negative influence(rare, bounded by VAR · m)</td></tr></table>

Table 7 summarizes the three canonical failure modes. Taken together, these constructions show that VAR characterizes only attention allocation, while decision-aligned route effects depend on content-sensitive alignment. This motivates using CRE/VRI as the primary signals for diagnosing and selecting intervention targets, rather than relying on attention-mass proxies.

# A.4. Empirical Alignment of VAR with Decision-Aligned Effects

Setup. On POPE-popular, we compute at the answer-critical decoding step the attention-mass proxy $\mathrm { V A R } _ { l , h } ( i ) =$ Pj∈Ivis $\sum _ { j \in I _ { \mathrm { v i s } } } \alpha _ { i j } ^ { ( l , h ) }$ αij , the decision-aligned first-order visual effect $\widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } ( i )$ (from our one-forward/one-gradient estimate), and the relative reliance $\begin{array} { r } { \widehat { \mathrm { V R I } } _ { l , h } ( i ) = \frac { \vert \widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } ( i ) \vert } { \vert \widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } ( i ) \vert + \vert \widehat { \Delta } _ { l , h } ^ { \mathrm { t x t } } ( i ) \vert + \varepsilon } } \end{array}$ .

Metric. We report Spearman rank correlations over all head–example pairs.

Table 8. Spearman rank correlations between VAR and decision-aligned quantities at the answer-critical token. 

<table><tr><td></td><td> $\rho_{1} = \text{corr}_{S}(\text{VAR}, |\widehat{\Delta}^{\text{vis}}|)$ </td><td> $\rho_{2} = \text{corr}_{S}(\text{VAR}, \widehat{\text{VRI}}_{l,h})$ </td></tr><tr><td>POPE (popular)</td><td>0.5766</td><td>0.5300</td></tr></table>

Interpretation. VAR exhibits a moderate monotonic association with decision-aligned magnitudes on this split. However, this does not make VAR a decision-aligned diagnostic: VAR cannot identify the sign of visual influence, nor detect cancellation or harmful visual evidence (Appendix A.3), and thus cannot support reliable head selection under route conflicts. Our theoretical results show that attention allocation alone is insufficient for identifying decision-aligned influence in general (Theorem A.6), motivating CRE/VRI as primary signals for intervention.

Empirical evidence for the “harmful visual content” regime. Consistent with the third failure mode in Appendix A.3 (high VAR but negative alignment), on POPE-popular at the answer-critical token, the visual-route effect is negative in about half of the head–example pairs: $\mathrm { P r } ( \widehat { \Delta } ^ { \mathrm { v i s } } < 0 ) = 0 . 5 0 3$ . Importantly, this does not vanish among heads with the largest visual attention mass: conditioning on the top 10% VAR heads $( \mathrm { V A R } \geq 0 . 1 4 0$ , the 90th percentile), we still have $\operatorname* { P r } ( \widehat { \Delta } ^ { \mathrm { v i s } } < 0 \mid \mathrm { V A R }$ in top 10%) = 0.515. Therefore, even when a head “looks” highly visual under VAR, its visual route can frequently provide counter-evidence (i.e., reduce the decision score), which is invisible to attention-mass proxies.

Table 9. Negative visual effects remain common even among high-VAR heads (POPE-popular). 

<table><tr><td>Split</td><td>Pr( $\widehat{\Delta}^{\text{vis}} < 0$ )</td><td>Pr( $\widehat{\Delta}^{\text{vis}} < 0$  | VAR top 10%)</td><td>90th pct. VAR</td></tr><tr><td>popular</td><td>0.503</td><td>0.515</td><td>0.140</td></tr></table>

# B. Validity of the First-Order Do-Effect Approximation

We provide both theoretical and empirical evidence for the validity of our one-forward/one-gradient first-order approximation to head-level do-effects: (i) we derive a deterministic error bound under a mild Lipschitz-type regularity condition along the gating path, (ii) we give a margin-based sufficient condition for sign reliability, and (iii) we perform an exact sanity check on a small subset by comparing against controlled single-head interventions.

# B.1. First-Order Error Bound

Proposition B.1 (First-order error bound). Assume the map $\tau \mapsto G _ { l , h } ( \tau , 1 )$ is L-Lipschitz on [0, 1], i.e.,

$$
\left\| G _ {l, h} \left(\tau_ {1}, 1\right) - G _ {l, h} \left(\tau_ {2}, 1\right) \right\| _ {F} \leq L \left| \tau_ {1} - \tau_ {2} \right|, \quad \forall \tau_ {1}, \tau_ {2} \in [ 0, 1 ].
$$

Then

$$
\left| \Delta_ {l, h} ^ {\mathrm{vis}} - \widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} \right| \leq \frac {L}{2} \left\| O _ {l, h} ^ {\mathrm{vis}} \right\| _ {F},
$$

and symmetrically for the text route.

Proof. From Proposition 3.1 and the path-integral representation,

$$
\Delta_ {l, h} ^ {\mathrm{vis}} - \widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} = \int_ {0} ^ {1} \left\langle G _ {l, h} (\tau , 1) - G _ {l, h} (1, 1), O _ {l, h} ^ {\mathrm{vis}} \right\rangle d \tau .
$$

Take absolute values and apply Cauchy–Schwarz and triangle inequality:

$$
\left| \Delta_ {l, h} ^ {\mathrm{vis}} - \widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} \right| \leq \| O _ {l, h} ^ {\mathrm{vis}} \| _ {F} \int_ {0} ^ {1} \left\| G _ {l, h} (\tau , 1) - G _ {l, h} (1, 1) \right\| _ {F} d \tau .
$$

Under the Lipschitz condition $\| G _ { l , h } ( \tau , 1 ) - G _ { l , h } ( 1 , 1 ) \| _ { F } \leq L ( 1 - \tau ) .$ .,

$$
\int_ {0} ^ {1} \left\| G _ {l, h} (\tau , 1) - G _ {l, h} (1, 1) \right\| _ {F} d \tau \leq \int_ {0} ^ {1} L (1 - \tau) d \tau = \frac {L}{2}.
$$

Combining yields the stated bound. The proof for the textual route follows by symmetry.

Remark B.2. In practice, we observe $\lVert G _ { l , h } ( \tau , 1 ) - G _ { l , h } ( 1 , 1 ) \rVert _ { F }$ grows approximately linearly as τ moves away from 1 for most heads, supporting the bounded-rate assumption.

# B.2. Sign Reliability

Proposition B.1 provides a deterministic bound on the approximation residual $R _ { l , h } ^ { \mathrm { v i s } } \triangleq \Delta _ { l , h } ^ { \mathrm { v i s } } - \widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } }$ . This immediately yields a sufficient condition under which the sign of the estimated do-effect is guaranteed to be correct.

Lemma B.3 (Sign reliability under bounded error). $I f \left| \Delta _ { l , h } ^ { \mathrm { v i s } } - \widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } \right| \leq \epsilon _ { l , h } ^ { \mathrm { v i s } }$ , then

$$
\widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} > \epsilon_ {l, h} ^ {\mathrm{vis}} \Rightarrow \Delta_ {l, h} ^ {\mathrm{vis}} > 0, \quad \widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} <   - \epsilon_ {l, h} ^ {\mathrm{vis}} \Rightarrow \Delta_ {l, h} ^ {\mathrm{vis}} <   0.
$$

In particular, whenever $\left| \widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } \right| > \epsilon _ { l , h } ^ { \mathrm { v i s } }$ , we have sign $\left( \Delta _ { l , h } ^ { \mathrm { v i s } } \right) = \mathrm { s i g n } ( \widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } )$

Proof. Assume the estimator error is bounded by

$$
\left| \Delta_ {l, h} ^ {\mathrm{vis}} - \widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} \right| \leq \epsilon_ {l, h} ^ {\mathrm{vis}}.
$$

If $\widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } > \epsilon _ { l , h } ^ { \mathrm { v i s } }$ , t hen

$$
\begin{array}{l} \Delta_ {l, h} ^ {\text { vis }} = \widehat {\Delta} _ {l, h} ^ {\text { vis }} + \left(\Delta_ {l, h} ^ {\text { vis }} - \widehat {\Delta} _ {l, h} ^ {\text { vis }}\right) \\ \geq \widehat {\Delta} _ {l, h} ^ {\text { vis }} - \left| \Delta_ {l, h} ^ {\text { vis }} - \widehat {\Delta} _ {l, h} ^ {\text { vis }} \right| \\ \geq \widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} - \epsilon_ {l, h} ^ {\mathrm{vis}} \\ > 0. \\ \end{array}
$$

Similarly, if $\widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } < - \epsilon _ { l , h } ^ { \mathrm { v i s } }$ , then

$$
\begin{array}{l} \Delta_ {l, h} ^ {\text { vis }} = \widehat {\Delta} _ {l, h} ^ {\text { vis }} + \left(\Delta_ {l, h} ^ {\text { vis }} - \widehat {\Delta} _ {l, h} ^ {\text { vis }}\right) \\ \leq \widehat {\Delta} _ {l, h} ^ {\text { vis }} + \left| \Delta_ {l, h} ^ {\text { vis }} - \widehat {\Delta} _ {l, h} ^ {\text { vis }} \right| \\ \leq \widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} + \epsilon_ {l, h} ^ {\mathrm{vis}} \\ <   0. \\ \end{array}
$$

Corollary B.4 (Instantiating Lemma B.3 via Proposition B.1). Under the conditions of Proposition B.1, define

$$
\epsilon_ {l, h} ^ {\mathrm{vis}} = \frac {L}{2} \left\| O _ {l, h} ^ {\mathrm{vis}} \right\| _ {F}, \quad \epsilon_ {l, h} ^ {\mathrm{txt}} = \frac {L}{2} \left\| O _ {l, h} ^ {\mathrm{txt}} \right\| _ {F}.
$$

Then, $i f \left| \widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } \right| > \epsilon _ { l , h } ^ { \mathrm { v i s } }$ , the sign of $\widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } }$ matches that $o f \Delta _ { l , h } ^ { \mathrm { v i s . } } ,$ and symmetrically, $i f \left| \widehat { \Delta } _ { l , h } ^ { \mathrm { t x t } } \right| > \epsilon _ { l , h } ^ { \mathrm { t x t } }$ , the sign of $\widehat { \Delta } _ { l , h } ^ { \mathrm { t x t } }$ matches that of $\Delta _ { l , h } ^ { \mathrm { t x t } }$ .

Implication for sign regimes. If both margins hold simultaneously, $\left. \widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } \right. > \epsilon _ { l , h } ^ { \mathrm { v i s } }$ and $\left. \widehat { \Delta } _ { l , h } ^ { \mathrm { t x t } } \right. > \epsilon _ { l , h } ^ { \mathrm { t x t } }$ , then the sign pattern of $\left( \widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } , \widehat { \Delta } _ { l , h } ^ { \mathrm { t x t } } \right)$  coincides with that of $\left( \Delta _ { l , h } ^ { \mathrm { v i s } } , \Delta _ { l , h } ^ { \mathrm { t x t } } \right)$ , so the head’s regime $( + + , + - , + , - )$ is certified.

# B.3. Exact Validation on a Small Subset

While Proposition B.1 bounds the first-order remainder under a smoothness condition, we further provide an exact sanity check on a small subset to empirically verify that the proposed first-order estimator faithfully captures both the sign and the relative ranking of head-level do-effects.

Protocol. For each example, we consider a fixed decoding step with a fixed prefix (image, question, and generated prefix tokens) and evaluate the effect on a scalar objective $\ell = \log p ( \mathrm { Y e s } ) - \log p ( \mathrm { N o } )$ at that step (the same ℓ used throughout our method). We first run a baseline single-step decode forward with all gates set to 1, obtaining $\ell _ { 0 } \triangleq \ell ( 1 , 1 )$ and caching the route-specific head outputs $O _ { l , h } ^ { \mathrm { v i s } }$ and $O _ { l , h } ^ { \mathrm { t x t } }$ (with the KV-cache intact). We then compute the first-order estimates $\widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } }$ and $\widehat { \Delta } _ { l , h } ^ { \mathrm { t x t } } \mathrm { a t } ( g _ { \mathrm { v i s } } , g _ { \mathrm { t x t } } ) = ( 1 , 1 )$ using the local sensitivity $G _ { l , h } = \partial \ell / \partial \tilde { O } _ { l , h }$ and the inner products in Proposition B.1.

To obtain the exact do-differences, we re-run the same single-step decode forward while intervening on only one head-route at a time: for the visual route, we set $g _ { l , h } ^ { \mathrm { v i s } } = 0$ and keep all other gates (including $g _ { l , h } ^ { \mathrm { t x t } } )$ at 1, yielding $\ell _ { l , h } ^ { \mathrm { v i s - o f f } }$ ; for the text route, we set $g _ { l , h } ^ { \mathrm { t x t } } = 0$ and keep all others at 1, yielding $\ell _ { l , h } ^ { \mathrm { t x t - o f f } }$ . We define the exact do-differences as

$$
\Delta_ {l, h} ^ {\mathrm{vis}} \triangleq \ell_ {0} - \ell_ {l, h} ^ {\mathrm{vis-off}}, \quad \Delta_ {l, h} ^ {\mathrm{txt}} \triangleq \ell_ {0} - \ell_ {l, h} ^ {\mathrm{txt-off}}.
$$

Importantly, all exact evaluations reuse the same prefix and the same KV-cache so that each ∆ reflects a controlled, local intervention at the same decoding step (rather than differences induced by divergent generation trajectories).

Subset selection. We perform this validation on a small subset of $N = 5 0$ examples (POPE). For each example, we evaluate exact do-differences on $K = 3 2$ head-route pairs, consisting of (i) the TopSmallest $- \mathrm { K } / 2$ heads ranked by VRI (most relevant to our gating) and (ii) $K / 2$ uniformly sampled heads (to cover the long tail). This keeps the cost manageable while directly probing the heads that our method is likely to intervene on.

Metrics. We report three complementary metrics: (i) Pearson/Spearman correlation between $\widehat { \Delta }$ and $\Delta$ across evaluated head-route pairs, capturing whether the first-order estimator preserves relative magnitudes and rankings;

Table 10. Correlations between the first-order estimates $\widehat { \Delta }$ and exact do-differences $\Delta$ on a small validation subset. Correlation is higher on large-magnitude heads (Top) that dominate gating decisions. 

<table><tr><td>Route</td><td>Subset</td><td>#Pairs</td><td>Pearson r</td><td>Spearman ρ</td></tr><tr><td>Vis</td><td>All (Top + Random)</td><td>1600</td><td>0.90</td><td>0.88</td></tr><tr><td>Txt</td><td>All (Top + Random)</td><td>1600</td><td>0.86</td><td>0.83</td></tr><tr><td>Vis</td><td>TopSmallest-16 by  $\widehat{VRI}$ </td><td>800</td><td>0.95</td><td>0.93</td></tr><tr><td>Vis</td><td>Random-16</td><td>800</td><td>0.78</td><td>0.74</td></tr><tr><td>Txt</td><td>TopSmallest-16 by  $\widehat{VRI}$ </td><td>800</td><td>0.92</td><td>0.90</td></tr><tr><td>Txt</td><td>Random-16</td><td>800</td><td>0.70</td><td>0.66</td></tr></table>

(ii) sign agreement $\mathbb { I } [ \mathrm { s i g n } ( \widehat { \Delta } ) = \mathrm { s i g n } ( \Delta ) ]$ , which directly validates the reliability of our sign-based regime taxonomy.

Table 11. Sign agreement between $\widehat { \Delta }$ and exact $\Delta$ on a small validation subset. The last row reports the fraction of head-step pairs whose two-route sign pattern is simultaneously correct, directly supporting the regime taxonomy based on $( \mathrm { s i g n } ( \Delta ^ { \mathrm { v i s } } ) , \mathrm { s i g n } ( \Delta ^ { \mathrm { t } \mathrm { x t } } \hat { ) } )$ . 

<table><tr><td>Route</td><td>#Pairs</td><td>Sign agreement (%)</td></tr><tr><td>Vis</td><td>1600</td><td>93.1</td></tr><tr><td>Txt</td><td>1600</td><td>90.4</td></tr><tr><td>Both routes correct (vis ∧ txt)</td><td>1600</td><td>86.7</td></tr></table>

Takeaway. Table 10 and Table 11 show that the first-order estimates $\widehat { \Delta }$ closely track the exact do-differences $\Delta$ on a held-out subset. They preserve both ranking (high Pearson/Spearman correlation) and $s i g n$ (over 90% agreement), with the strongest alignment on large-magnitude heads selected for gating. Moreover, the two-route sign pattern is simultaneously correct for most head-step pairs, directly supporting our sign-based regime taxonomy. Overall, these results validate $\widehat { \Delta }$ as an accurate and efficient proxy for head-level do-effects in our intervention pipeline.

# C. Proofs of Main Results

For readability, we collect proofs of main-text proposition in Appendix C. Proofs of results introduced only in the appendix are provided inline within the corresponding appendix sections. We further justify the validity of our first-order estimates, both theoretically and empirically, in Appendix B.

Proposition 3.1 [Directional-derivative estimator] $A t \left( g _ { \mathrm { v i s } } , g _ { \mathrm { t x t } } \right) = \left( 1 , 1 \right)$ ,

$$
\widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} = \Big \langle G _ {l, h} (1, 1), O _ {l, h} ^ {\mathrm{vis}} \Big \rangle , \qquad \widehat {\Delta} _ {l, h} ^ {\mathrm{txt}} = \Big \langle G _ {l, h} (1, 1), O _ {l, h} ^ {\mathrm{txt}} \Big \rangle ,
$$

Moreover, $\begin{array} { r } { \widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } = \frac { \partial \ell _ { l , h } } { \partial g _ { \mathrm { v i s } } } ( 1 , 1 ) } \end{array}$ ∂gvis and $\begin{array} { r } { \widehat { \Delta } _ { l , h } ^ { \mathrm { t x t } } = \frac { \partial \ell _ { l , h } } { \partial q _ { \mathrm { t x t } } } ( 1 , 1 ) } \end{array}$ ∂ℓl,h∂gtxt (1, 1). The exact do-differences admit the path-integral form, and the gtx approximation error is characterized in Proposition B.1.

Proof. Fix a layer–head pair (l, h). Recall that the route decomposition is computed once from the baseline run at $( g _ { \mathrm { v i s } } , g _ { \mathrm { t x t } } ) = ( 1 , 1 )$ and then kept fixed for all gate values. That is, $O _ { l , h } ^ { \mathrm { v i s } }$ and $O _ { l , h } ^ { \mathrm { t x t } }$ do not depend on $( g _ { \mathrm { v i s } } , g _ { \mathrm { t x t } } )$ ; only their linear combination is gated. By definition of the gated head output

$$
\tilde {O} _ {l, h} (g _ {\mathrm{vis}}, g _ {\mathrm{txt}}) = g _ {\mathrm{vis}} O _ {l, h} ^ {\mathrm{vis}} + g _ {\mathrm{txt}} O _ {l, h} ^ {\mathrm{txt}},
$$

the route activations $O _ { l , h } ^ { \mathrm { v i s } } , O _ { l , h } ^ { \mathrm { t x t } }$ are constants with respect to $( g _ { \mathrm { v i s } } , g _ { \mathrm { t x t } } )$ (they come from the baseline forward pass with a single joint softmax). Let

$$
G _ {l, h} (g _ {\mathrm{vis}}, g _ {\mathrm{txt}}) = \frac {\partial \ell_ {l , h} (g _ {\mathrm{vis}} , g _ {\mathrm{txt}})}{\partial \tilde {O} _ {l , h}} \in \mathbb {R} ^ {L _ {\mathrm{q}} \times d _ {h}}.
$$

The chain rule gives, for any $( g _ { \mathrm { v i s } } , g _ { \mathrm { t x t } } )$ ,

$$
\frac {\partial \ell_ {l , h}}{\partial g _ {\mathrm{vis}}} (g _ {\mathrm{vis}}, g _ {\mathrm{txt}}) = \Big \langle G _ {l, h} (g _ {\mathrm{vis}}, g _ {\mathrm{txt}}), \frac {\partial \tilde {O} _ {l , h}}{\partial g _ {\mathrm{vis}}} \Big \rangle = \Big \langle G _ {l, h} (g _ {\mathrm{vis}}, g _ {\mathrm{txt}}), O _ {l, h} ^ {\mathrm{vis}} \Big \rangle ,
$$

and analogously

$$
\frac {\partial \ell_ {l , h}}{\partial g _ {\mathrm{txt}}} (g _ {\mathrm{vis}}, g _ {\mathrm{txt}}) = \Big \langle G _ {l, h} (g _ {\mathrm{vis}}, g _ {\mathrm{txt}}), O _ {l, h} ^ {\mathrm{txt}} \Big \rangle .
$$

Evaluating at the baseline (1, 1) yields

$$
\frac {\partial \ell_ {l , h}}{\partial g _ {\mathrm{vis}}} (1, 1) = \Big \langle G _ {l, h} (1, 1), O _ {l, h} ^ {\mathrm{vis}} \Big \rangle , \qquad \frac {\partial \ell_ {l , h}}{\partial g _ {\mathrm{txt}}} (1, 1) = \Big \langle G _ {l, h} (1, 1), O _ {l, h} ^ {\mathrm{txt}} \Big \rangle .
$$

By definition, the exact visual do-difference is

$$
\Delta_ {l, h} ^ {\text { vis }} \triangleq \ell_ {l, h} (1, 1) - \ell_ {l, h} (0, 1).
$$

Consider the one-dimensional function obtained by restricting $\ell _ { l , h }$ to the visual path $\tau \mapsto ( \tau , 1 ) , \mathrm { i . e . , } \phi ( \tau ) \triangleq \ell _ { l , h } ( \tau , 1 )$ . By the fundamental theorem of calculus,

$$
\Delta_ {l, h} ^ {\mathrm{vis}} = \phi (1) - \phi (0) = \int_ {0} ^ {1} \phi^ {\prime} (\tau) d \tau .
$$

Since $g _ { \mathrm { t x t } }$ is fixed to 1 on this path, the chain rule gives $\begin{array} { r } { \phi ^ { \prime } ( \tau ) = \frac { \partial \ell _ { l , h } } { \partial g _ { \mathrm { v i s } } } ( \tau , 1 ) } \end{array}$ ∂ℓl,h∂g (τ, 1), hence

$$
\Delta_ {l, h} ^ {\mathrm{vis}} = \int_ {0} ^ {1} \frac {\partial \ell_ {l , h}}{\partial g _ {\mathrm{vis}}} (\tau , 1) d \tau .
$$

This path-integral form is closely related to Integrated Gradients (Sundararajan et al., 2017): both express a finite intervention effect as an integral of a partial derivative along a continuous path. Here the path is over the gating variable, and we adopt a single-point (right-endpoint) discretization for efficiency. We approximate this path integral by its right-endpoint value, yielding the first-order estimator

$$
\widehat {\Delta} _ {l, h} ^ {\mathrm{vis}} \triangleq \frac {\partial \ell_ {l , h}}{\partial g _ {\mathrm{vis}}} (1, 1).
$$

Using the linear gating relation $\tilde { O } _ { l , h } = g _ { \mathrm { v i s } } O _ { l , h } ^ { \mathrm { v i s } } + g _ { \mathrm { t x t } } O _ { l , h } ^ { \mathrm { t x t } }$ and the definition $G _ { l , h } = \partial \ell _ { l , h } / \partial \tilde { O } _ { l , h }$ , we have

$$
\frac {\partial \ell_ {l , h}}{\partial g _ {\mathrm{vis}}} (1, 1) = \Big \langle G _ {l, h} (1, 1), O _ {l, h} ^ {\mathrm{vis}} \Big \rangle ,
$$

which proves the claim. The textual case follows analogously.

Remark C.1 (Multi-point gate integral). Instead of the single right-endpoint approximation, one may use an m-point Riemann estimator ∆b visl,h(m) = 1m Pmk=1 ∂ℓl,h∂gvis ( $\begin{array} { r } { \widehat { \Delta } _ { l , h } ^ { \mathrm { v i s } } ( m ) = \frac { 1 } { m } \sum _ { k = 1 } ^ { m } \frac { \partial \ell _ { l , h } } { \partial g _ { \mathrm { v i s } } } ( k / m , 1 ) } \end{array}$ k/m, 1), which converges to ∆visl,h as m increases. Under the same Lipschitz-type $\Delta _ { l , h } ^ { \mathrm { v i s } }$ condition as Proposition B.1, the discretization error decreases on the order of $1 / m ,$ , trading additional gate-query passes for higher accuracy.

Remark C.2 (General paths). More generally, one can integrate along any continuous path $c ( \tau ) = ( g _ { \mathrm { v i s } } ( \tau ) , g _ { \mathrm { t x t } } ( \tau ) )$ to define joint interventions, with corresponding discretizations.

# D. Implementation Details

# D.1. Benchmark Details

MME. MME (Fu et al., 2025) is a comprehensive benchmark for assessing general multimodal capabilities of LVLMs. It organizes evaluation into two major categories: perception and cognition. The perception category covers fine-grained visual understanding skills, including existence, count, position/location, color, poster, celebrity, scene, landmark, artwork, and OCR. The cognition category focuses on higher-level reasoning and knowledge integration, including commonsense reasoning, numerical calculation, text translation, and code reasoning. Each MME sample consists of an image-question pair, and questions are formatted in a short VQA style with binary (Yes/No) responses, enabling a broad yet unified evaluation across diverse ability dimensions.

POPE. Polling-based Object Probing Evaluation (POPE) (Li et al., 2023) is a binary VQA benchmark designed to assess object-level grounding and object hallucination in LVLMs. Given an image from MS-COCO and a templated query of the form “Is there a <object> in this image?”, the model must answer Yes or No. We evaluate on the three standard POPE subsets—random, popular, and adversarial—each containing 3,000 questions. The three subsets differ in how negative objects are selected: (i) random samples negatives roughly uniformly from the overall vocabulary, so many negatives are less contextually plausible; (ii) popular draws negatives from frequently occurring categories, making language priors stronger and false positives more likely; and (iii) adversarial chooses negatives that are contextually plausible for the image (e.g., objects that commonly co-occur with present objects or fit the scene), making it the most challenging split. We report Accuracy and F1 over Yes/No predictions (treating Yes as the positive class):

$$
\text { Accuracy } = \frac {\# \text { correct }}{\# \text { total }}, \quad \text { Precision } = \frac {T P}{T P + F P}, \quad \text { Recall } = \frac {T P}{T P + F N}, \quad \text { F1 } = \frac {2 \cdot \text { Precision } \cdot \text { Recall }}{\text { Precision } + \text { Recall }}.
$$

CHAIR. Caption Hallucination Assessment with Image Relevance (CHAIR) (Rohrbach et al., 2018) is a benchmark for measuring object hallucination in image captioning. It prompts LVLMs to generate a description for each input image and compares the mentioned objects against the ground-truth object set (e.g., MS-COCO annotations). Using the CHAIR object lexicon to map noun phrases in the caption to object categories, an object is counted as hallucinated if it is mentioned by the model but does not appear in the ground truth. CHAIR quantifies hallucination at both the instance and sentence levels:

$$
\operatorname{CHAIR} _ {I} = \frac {| \{\text { hallucinated   objects } \} |}{| \{\text { all   mentioned   objects } \} |}, \quad \operatorname{CHAIR} _ {S} = \frac {| \{\text { captions   with   hallucinated   objects } \} |}{| \{\text { all   captions } \} |}.
$$

Lower values indicate fewer hallucinations. To evaluate whether captions still cover the necessary visual content, we additionally report instance-level recall:

$$
\text { Recall } = \frac {| \{\text { non -hallucinated   objects } \} |}{| \{\text { all   existing   objects } \} |}.
$$

We also report the average response length (Len), computed as the mean number of generated tokens per caption, to control for potential verbosity changes across methods.

MMHal-Bench. MMHal-Bench (Sun et al., 2024) is a specialized benchmark for evaluating multimodal hallucination in LVLMs, consisting of 96 carefully designed image–question pairs spanning eight categories: object attributes (ATTR), adversarial objects (ADV), comparisons (COMP), counting (COUNT), spatial relations (SPAT), environmental inferences (ENV), holistic descriptions (HOL), and others (OTHER). Following prior work, model outputs are judged by GPT-4 against the reference answer and associated object information, producing an overall score $s _ { i }$ on a 0–5 scale (higher is better). We report the Overall Score and Hallucination rate (Hu.%):

$$
\text {Score} = \frac {1}{N} \sum_ {i = 1} ^ {N} s _ {i}, \quad \text {Hu.} \% = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {I} [ s _ {i} <   3 ] \times 100,
$$

where N is the number of examples and responses with $s _ { i } < 3$ are counted as hallucinations.

AMBER. An LLM-free Multi-dimensional Benchmark (AMBER) (Wang et al., 2023a) evaluates multimodal hallucinations without relying on LLM-based judges. It contains 1,004 manually curated images with structured annotations spanning object existence, attributes (e.g., state/number/action), relations $( \mathrm { e . g . }$ ., direct contact), and per-image hallucinatory target objects, covering 337 object categories. AMBER supports both generative and discriminative evaluation: in the generative setting, models produce free-form captions (e.g., “Describe this image.”), and AMBER extracts mentioned objects with an official lexicon; in the discriminative setting, AMBER constructs 10,586 binary $( \mathrm { Y e s } / \mathrm { N o } )$ probes over existence/attributes/relations and counterfactual hallucinatory targets. For generative evaluation, let $A _ { i }$ be the ground-truth object set, ${ \hat { A } } _ { i }$ the extracted mentioned-object set, $H _ { i } = \hat { A } _ { i } \setminus A _ { i }$ the hallucinated-object $\operatorname { s e t } ,$ , and $T _ { i }$ the hallucinatory target set for image i. AMBER reports:

$$
\mathrm{CHAIR} = \frac {1}{N} \sum_ {i = 1} ^ {N} \frac {| H _ {i} |}{| \hat {A} _ {i} |}, \quad \mathrm{Cover} = \frac {1}{N} \sum_ {i = 1} ^ {N} \frac {| \hat {A} _ {i} \cap A _ {i} |}{| A _ {i} |}, \quad \mathrm{Hal} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {I} [ | H _ {i} | > 0 ], \quad \mathrm{Cog} = \frac {1}{N} \sum_ {i = 1} ^ {N} \frac {| \hat {A} _ {i} \cap T _ {i} |}{| T _ {i} |}.
$$

For discriminative evaluation, AMBER reports standard classification scores (Acc./F1), and further summarizes overall performance with the aggregated AMBER Score:

$$
\mathrm{AMBERScore} = \frac {1}{2} (1 0 0 - \mathrm{CHAIR} + \mathrm{F1}).
$$

# D.2. Models and Experiments Setup

Most experiments are conducted on two NVIDIA A40 GPUs ( 48 GB memory per GPU). For the larger LLaVA-NeXT-34B and Qwen2.5-VL-32B-Instruct models, we use a single NVIDIA RTX PRO 6000 Blackwell GPU (96 GB memory).

Model Architectures. In Table 12, we present the detailed architectures of the LVLMs used in our experiments, where the gray rows correspond to the backbones used in our main experiments. These models are based on the Vision Transformer (ViT) (Dosovitskiy et al., 2021) and employ pre-trained vision encoders from various sources.

Table 12. Details of the LVLM architectures used in our experiments. 

<table><tr><td>Model</td><td>Vision encoder</td><td>LLM</td></tr><tr><td>LLaVA-1.5-7B (Liu et al., 2024b)Qwen-VL-Chat-7B (Bai et al., 2023)Qwen2.5-VL-7B-Instruct (Bai et al., 2025)</td><td>CLIP ViT-L/14 (336px) (Radford et al., 2021)OpenCLIP ViT-bigG (Ilharco et al., 2021)ViT w/ 2D-RoPE &amp; window attention</td><td>Vicuna-v1.5-7B (Chiang et al., 2023)Qwen-7BQwen2.5-7B</td></tr><tr><td>LLaVA-1.5-13B (Liu et al., 2024b)LLaVA-NeXT-34B (Li et al., 2024)Qwen2.5-VL-32B-Instruct (Bai et al., 2025)</td><td>CLIP ViT-L/14 (336px) (Radford et al., 2021)CLIP ViT-L/14 (336px) (Radford et al., 2021)ViT w/ 2D-RoPE &amp; window attention</td><td>Vicuna-v1.5-13B (Chiang et al., 2023)Nous-Hermes-2-Yi-34B (Yi-34B family)Qwen2.5-32B</td></tr></table>

Hyperparameters. Our hyperparameters include the intervened layer range $\left( L _ { \mathrm { s t a r t } } , L _ { \mathrm { e n d } } \right)$ , the number of selected heads k, and the gating strength γ. Concretely, on the POPE popular subset, we intervene on mid-to-upper transformer layers, using $( L _ { \mathrm { s t a r t } } , L _ { \mathrm { e n d } } ) = ( 8 , 1 9 )$ for LLaVA-1.5-7B, (10, 24) for Qwen-VL-Chat, and (9, 17) for Qwen2.5-VL-7B-Instruct. At each intervened layer, we select the top-K heads with the smallest VRI values and apply text-route gating with strength γ. The optimal k and $\gamma$ across benchmarks are summarized in Table 13.

Table 13. Hyperparameter search selected for k and $\gamma$ across benchmarks. 

<table><tr><td>Model</td><td>parameter</td><td>POPE (random)</td><td>POPE (popular)</td><td>POPE (adversarial)</td><td>MME</td><td>CHAIR</td><td>MMHal-Bench</td><td>AMBER-G</td><td>AMBER-D</td></tr><tr><td rowspan="2">LLaVA-1.5-7B</td><td>k</td><td>7</td><td>11</td><td>9</td><td>12</td><td>11</td><td>8</td><td>9</td><td>10</td></tr><tr><td>γ</td><td>0.6</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.7</td><td>0.5</td><td>0.6</td><td>0.7</td></tr><tr><td rowspan="2">Qwen-VL-Chat-7B</td><td>k</td><td>10</td><td>8</td><td>12</td><td>10</td><td>8</td><td>9</td><td>-</td><td>-</td></tr><tr><td>γ</td><td>0.4</td><td>0.7</td><td>0.5</td><td>0.4</td><td>0.6</td><td>0.6</td><td>-</td><td>-</td></tr><tr><td rowspan="2">Qwen2.5-VL-7B-Instruct</td><td>k</td><td>10</td><td>9</td><td>11</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>γ</td><td>0.4</td><td>0.6</td><td>0.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

# D.3. Choice of The Scalar Objective ℓ

To compute decision-aligned route effects, we require a scalar objective ℓ whose gradient reflects the model’s preference for the target decision at the current decoding step. We instantiate ℓ differently for discriminative (binary) and generative benchmarks, matching each evaluation protocol.

Discriminative tasks (binary Yes/No). For discriminative hallucination benchmarks such as POPE, MME and the binary subset of AMBER, the model is prompted to output only “Yes” or “No”. Accordingly, we define the scalar target as the log-odds margin

$$
\ell = \log p (\text { YES } \mid x) - \log p (\text { No } \mid x),
$$

where $p ( \cdot \mid x )$ is the next-token distribution conditioned on the multimodal input x. This choice is both decision-aligned and interpretable: $\ell > 0$ indicates a higher preference for “Yes” than “No”, while $\ell < 0$ indicates the opposite. In particular, when the candidate set is restricted to {YES, NO}, we have

$$
\ell = \log \frac {p (\mathrm{YES} \mid x)}{p (\mathrm{NO} \mid x)},
$$

so changes in ℓ directly correspond to how an intervention shifts the binary decision boundary. Therefore, the sign and magnitude of the estimated causal contribution under this ℓ admit a straightforward interpretation: a positive contribution supports the “Yes” decision, whereas a negative contribution supports $\mathbf { \tilde { \Sigma } } ^ { 6 6 } \mathbf { N } \mathbf { o } ^ { \mathbf { \Sigma } , \mathbf { \vec { \Sigma } } }$ .

Empirical note. We also experimented with using $\ell = \log p ( y _ { t } \mid x )$ for discriminative tasks, but found it consistently inferior to the log-odds margin $\log p ( \mathbf { Y } \mathrm { { E S } } \mid x ) - \log p ( \mathbf { N } \mathrm { { O } } \mid x )$ , likely because the margin explicitly accounts for the competing alternative and thus better matches the binary decision structure.

Generative tasks (open-ended decoding). For open-ended generation, we apply the intervention token-by-token and define the step-wise target as

$$
\ell_ {t} = \log p (y _ {t} | x, y _ {<   t}),
$$

where $y _ { t }$ denotes the token selected at decoding step t (under the chosen decoding rule), and $p ( \cdot \mid x , y _ { < t } )$ is the next-token distribution. Intuitively, $\ell _ { t }$ measures how strongly the model commits to the current token, enabling us to quantify whether the visual/textual routes support or contradict this local decision at step t.

Operationally, at each step t we (i) run a standard one-step decode forward to obtain $p ( \cdot \mid x , y _ { < t } )$ and the selected token $y _ { t } ;$ (ii) compute the local sensitivity for this decision via a single backward pass,

$$
G _ {t} = \frac {\partial \ell_ {t}}{\partial \tilde {O} _ {t}},
$$

where ${ \tilde { O } } _ { t }$ denotes the intervened route/head activations at step t; (iii) aggregate the resulting head-/route-level effects into $r _ { t }$ and apply the gate; and finally (iv) re-run a lightweight one-step forward with the gate applied to produce the actual token emission. This procedure requires two forward passes and one gradient computation per generated token, but the gradient is taken only through a single-step decode (with the prefix KV-cache already built), so its overhead is modest: in our profiling, the autograd component takes 8.77 ms per step, accounting for 6.30% of the total token-generation time, while computing $r _ { t }$ and applying the gate are lightweight. Moreover, the memory footprint is close to regular decoding: we do not retain computation graphs across steps, so the backward-related activations are transient, and the dominant KV-cache growth remains essentially the same as the baseline decode.

Token filtering and object-only intervention. A practical concern in token-by-token intervention is that many generated tokens are weakly semantic (e.g., punctuation such as $^ { 6 6 } , ^ { 6 6 } , ^ { 5 } )$ , for which estimating route effects may be unnecessary and could add noise. We therefore explored two variants: (i) skipping intervention on such non-informative tokens; and (ii) performing intervention only when the current token corresponds to an object word. For (ii), we leverage the CHAIR object vocabulary and apply the intervention only if the selected token $y _ { t }$ matches an entry in this list. While this object-only strategy is straightforward on CHAIR-style evaluation where the object set is predefined, it is less practical in open-world generation because the space of possible objects is unbounded and the relevant lexicon is not known a priori. Empirically, we find that restricting intervention to object tokens yields performance close to our default full-token intervention, suggesting that the gains are largely driven by object-related decisions and that our method is not overly sensitive to intervening on punctuation or other low-content tokens.

# E. Additional Experiments

# E.1. Results on More Advanced LVLMs

To assess whether our intervention generalizes beyond the primary 7B backbones, we further evaluate it on several stronger LVLMs with different architectures and larger model scales. Table 14 reports POPE-average results on more advanced LVLMs, where each entry is Accuracy / F1. Overall, our method yields consistent gains across both LLaVA-style and Qwen-VL-style backbones, including larger-capacity models (LLaVA-1.5-13B, Qwen2.5-VL-32B-Instruct, and LLaVA-NeXT-34B). This suggests that the intervention does not rely on a particular model architecture or a specific pretraining recipe, but instead captures a more general property of how LVLMs mix visual and textual evidence during decision making.

Notably, the improvements persist as model scale increases. While stronger backbones typically improve general instructionfollowing and linguistic priors, they may still exhibit “over-confident yes” tendencies under weak visual evidence in POPE-style settings. Our method counteracts such bias by suppressing visually unsupported routes at decoding time, leading to simultaneous improvements in Accuracy and F1. Importantly, our method improves both metrics for all evaluated models, implying that it does not trade off correctness for conservativeness (or vice versa) on POPE-average. These results further support the robustness and general applicability of our approach beyond the 7B setting.

Table 14. Results on more advanced LVLMs (POPE-average). Each entry is Accuracy / F1-Score. Our method consistently improves both accuracy and F1 across larger backbones. 

<table><tr><td>Method</td><td>LLaVA-1.5-7B</td><td>Qwen2.5-VL-7B-Instruct</td><td>LLaVA-1.5-13B</td><td>Qwen2.5-VL-32B-Instruct</td><td>LLaVA-NeXT-34B</td></tr><tr><td>Regular</td><td>81.37 / 79.65</td><td>83.99 / 84.31</td><td>83.05 / 82.81</td><td>84.25 / 85.31</td><td>83.17 / 84.79</td></tr><tr><td>CRG</td><td>87.71 / 86.61</td><td>89.72 / 88.55</td><td>87.87 / 87.94</td><td>89.92 / 88.93</td><td>89.74 / 87.73</td></tr></table>

# E.2. Implementation with VCD

To examine whether our conflict-aware text-route gating can be combined with existing training-free decoding heuristics, we integrate our method with VCD (Leng et al., 2024), a representative contrastive decoding strategy. Below we briefly summarize VCD and then describe a simple way to apply it on top of our method.

Visual Contrastive Decoding. VCD (Leng et al., 2024) is a training-free decoding method that mitigates hallucinations by contrasting the model’s token preferences under the original image and a visually degraded counterfactual. At each decoding step t, VCD computes logits from the original image I and a perturbed image I− (e.g., blurred or noised), denoted by $z _ { t } ^ { + } = \mathrm { l o g i t s } ( I , y _ { < t } )$ and $z _ { t } ^ { - } = \mathrm { l o g i t s } ( I ^ { - } , y _ { < t } )$ , respectively. It then forms a contrastive logit for sampling/greedy decoding:

$$
\tilde {z} _ {t} = z _ {t} ^ {+} + \alpha (z _ {t} ^ {+} - z _ {t} ^ {-}) = (1 + \alpha) z _ {t} ^ {+} - \alpha z _ {t} ^ {-},
$$

where $\alpha \geq 0$ controls the contrast strength. Intuitively, tokens that remain highly probable even under the weakened visual evidence $I ^ { - }$ are more likely driven by language priors; subtracting $z _ { t } ^ { - }$ suppresses such prior-dominant predictions and promotes image-grounded alternatives.

VCD on top of our conflict-aware text-route gating. We incorporate VCD on top of our Conflict A+B text-route gating by using our method as the positive branch and a regular model as the negative branch. Concretely, for the original image $I ,$ we enable Conflict A+B gating to obtain $z _ { t } ^ { \mathrm { C R G } } = \bar { \log \mathrm { i t s } } ( \mathrm { C R G } ( I ) , y _ { < t } )$ . For the counterfactual image $I ^ { - }$ , we follow the standard VCD construction but disable CRG (i.e., keep the original model unchanged) to preserve a strong language-prior reference, yielding $z _ { t } ^ { \mathrm { R e g , - } } = \mathrm { l o g i t s } ( \mathrm { R e g u l a r } ( I ^ { - } ) , y _ { < t } )$ . We then apply the same contrastive fusion:

$$
\tilde {z} _ {t} = z _ {t} ^ {\mathrm{CRG}} + \alpha (z _ {t} ^ {\mathrm{CRG}} - z _ {t} ^ {\mathrm{Reg,-}}).
$$

This design combines conflict-aware suppression on the original image with contrastive penalization of tokens that are insensitive to visual perturbations, providing potentially complementary signals during generation.

We evaluate the proposed combination on both LLaVA-1.5-7B and Qwen-VL-Chat over the three POPE subsets (random, popular, adversarial). As shown in Table 15, $\mathrm { C R G } _ { + \mathrm { V C D } }$ brings only marginal changes relative to CRG: it slightly improves accuracy in most settings, while the F1 score can be comparable or slightly lower on some subsets.

Table 15. POPE results for LLaVA-1.5-7B and Qwen-VL-Chat, comparing REGULAR, VCD, CRG (ours) and $\mathrm { C R G } _ { + \mathrm { V C D } }$ 

<table><tr><td rowspan="2">Setting</td><td rowspan="2">Method</td><td colspan="2">LLaVA-1.5-7B</td><td colspan="2">Qwen-VL-Chat</td></tr><tr><td>Accuracy</td><td>F1-Score</td><td>Accuracy</td><td>F1-Score</td></tr><tr><td rowspan="4">Random</td><td>Regular</td><td>83.29</td><td>81.33</td><td>84.63</td><td>82.61</td></tr><tr><td>VCD</td><td>87.73</td><td>87.16</td><td>86.93</td><td>85.46</td></tr><tr><td>CRG (ours)</td><td>90.30</td><td>89.51</td><td>89.46</td><td>88.33</td></tr><tr><td> $CRG_{+VCD}$ </td><td>90.43</td><td>89.30</td><td>89.43</td><td>88.63</td></tr><tr><td rowspan="4">Popular</td><td>Regular</td><td>81.88</td><td>80.06</td><td>83.63</td><td>81.53</td></tr><tr><td>VCD</td><td>85.38</td><td>85.06</td><td>85.17</td><td>83.68</td></tr><tr><td>CRG (ours)</td><td>88.40</td><td>86.54</td><td>87.63</td><td>86.55</td></tr><tr><td> $CRG_{+VCD}$ </td><td>88.69</td><td>86.04</td><td>87.66</td><td>86.46</td></tr><tr><td rowspan="4">Adversarial</td><td>Regular</td><td>78.96</td><td>77.57</td><td>81.03</td><td>79.30</td></tr><tr><td>VCD</td><td>80.88</td><td>81.33</td><td>83.10</td><td>82.04</td></tr><tr><td>CRG (ours)</td><td>84.43</td><td>83.77</td><td>84.70</td><td>83.78</td></tr><tr><td> $CRG_{+VCD}$ </td><td>84.97</td><td>83.81</td><td>84.76</td><td>83.84</td></tr></table>

# E.3. Detailed Results of AMBER & MMHal-Bench

Detailed Results of MMHal-Bench. As reported in Table 16, our method achieves the best results on both backbones. On LLaVA-1.5-7B, we improve the score from 2.23 to 2.69 while reducing Hu.% from 65.3 to 50.9; on Qwen-VL-Chat, the score increases from 2.27 to 2.80 with Hu.% dropping from 58.2 to 48.8. Compared with the strongest baseline (VTI), our approach still yields consistent gains in both Score and Hu.%, indicating more informative generations with fewer hallucinations.

Table 16. Results on MMHal-Bench (evaluated by GPT-4) 

<table><tr><td rowspan="2">Method</td><td colspan="2">LLaVA-1.5-7B</td><td colspan="2">Qwen-VL-Chat</td></tr><tr><td>Score↑</td><td>Hu.%↓</td><td>Score↑</td><td>Hu.%↓</td></tr><tr><td>Regular</td><td>2.23</td><td>65.3</td><td>2.27</td><td>58.2</td></tr><tr><td>VCD</td><td>2.31</td><td>58.7</td><td>2.32</td><td>56.5</td></tr><tr><td>OPERA</td><td>2.40</td><td>56.4</td><td>2.49</td><td>55.3</td></tr><tr><td>VTI</td><td>2.51</td><td>53.7</td><td>2.62</td><td>51.4</td></tr><tr><td>CRG</td><td>2.69</td><td>50.9</td><td>2.80</td><td>48.8</td></tr></table>

Table 18. Supplementary POPE results on MS-COCO. Each split reports Accuracy and F1-Score. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Random</td><td colspan="2">Popular</td><td colspan="2">Adversarial</td><td colspan="2">Average</td></tr><tr><td>Acc.</td><td>F1</td><td>Acc.</td><td>F1</td><td>Acc.</td><td>F1</td><td>Acc.</td><td>F1</td></tr><tr><td>ONLY (Wan et al., 2025)</td><td>89.70</td><td>89.10</td><td>86.00</td><td>86.31</td><td>79.40</td><td>81.07</td><td>85.03</td><td>85.49</td></tr><tr><td>CAUSALMM (Zhou et al., 2025)</td><td>88.93</td><td>88.10</td><td>87.13</td><td>87.26</td><td>83.70</td><td>82.78</td><td>86.59</td><td>86.05</td></tr><tr><td>ICT (Chen et al., 2025)</td><td>90.11</td><td>90.03</td><td>87.50</td><td>87.60</td><td>84.43</td><td>83.74</td><td>87.35</td><td>87.12</td></tr><tr><td>DMAS (Yin et al., 2026)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>86.81</td><td>86.79</td></tr><tr><td>CRG (this work)</td><td>90.30</td><td>89.51</td><td>88.40</td><td>86.54</td><td>84.43</td><td>83.77</td><td>87.71</td><td>86.61</td></tr></table>

Detailed Results of AMBER. Table 17 reports the per-metric results on AMBER for LLaVA-1.5-7B. Compared with regular decoding and PAI, our method substantially reduces object hallucination (CHAIR 8.3 to 4.6, Hal 36.7 to 23.2) while improving coverage and decision quality (Cover 49.4 to 53.3, F1 73.7 to 77.5). As a result, it achieves the best overall AMBER score (82.70 to 86.45).

Table 17. Results on AMBER (LLaVA-1.5-7B). The AMBER metric is computed as $( 1 0 0 - \mathrm { C H A I R } + F 1 ) / 2$ . 

<table><tr><td>Model</td><td>Method</td><td>CHAIR↓</td><td>Cover↑</td><td>Hal↓</td><td>Cog↓</td><td>Acc.↑</td><td>Prec.↑</td><td>Rec.↑</td><td>F1↑</td><td>AMBER↑</td></tr><tr><td rowspan="3">LLaVA-1.5-7B</td><td>Regular</td><td>8.3</td><td>49.4</td><td>36.7</td><td>4.4</td><td>72.7</td><td>84.7</td><td>62.1</td><td>73.7</td><td>82.70</td></tr><tr><td>PAI</td><td>6.7</td><td>48.8</td><td>42.3</td><td>1.9</td><td>74.3</td><td>87.2</td><td>73.8</td><td>74.8</td><td>84.05</td></tr><tr><td>CRG</td><td>4.6</td><td>53.3</td><td>23.2</td><td>2.1</td><td>77.4</td><td>92.3</td><td>72.1</td><td>77.5</td><td>86.45</td></tr></table>

# E.4. Supplementary POPE Comparisons with Recent Methods

Table 18 also compares CRG with recent training-free hallucination mitigation methods on the standard POPE benchmark under the MS-COCO setup. These methods include ONLY (Wan et al., 2025), CAUSALMM (Zhou et al., 2025), ICT (Chen et al., 2025), and DMAS (Yin et al., 2026). The Average columns are arithmetic means over the three standard POPE splits: Random, Popular, and Adversarial. The CRG row uses the results reported in this submission.

Relation to process- and attention-based signals. Recent work also suggests that process-level or attention-based signals can be useful. For example, Knowledge Transfer from Interaction Learning (Gao et al., 2025) emphasizes the value of interaction processes rather than only final representations. In hallucination mitigation, Modality Bias in LVLMs (Zheng & Zhang, 2025) uses attention-lens analysis to rebalance modality usage, and AdaIAT (Zhong et al., 2026) adaptively modifies attention to generated text to reduce hallucination while preserving fluency. These directions are complementary to CRG. Our claim is not that attention or interaction signals are useless; rather, attention allocation alone is insufficient as the primary decision-aligned criterion for selective gating. CRG uses route effects to determine whether a route helps or hurts the current decision, while attention/process signals can provide complementary diagnostics or future triggers.

# F. Limitations and More Discussion

# F.1. Scope Boundary of CRG

CRG should be read as a conflict detector and intervention, not as a general hallucination corrector. It targets cases in which visual evidence is available, but a prior-driven text route dominates the current token decision. This is the setting our route decomposition can identify and act on: the visual and text routes push the decision in opposite directions, and suppressing the selected text route can restore a more visually grounded choice. Errors whose cause does not pass through such visual-text route conflict are outside the primary mechanism.

The key non-target case is same-direction bias, which is different from the target failure mode addressed by CRG. CRG is designed for cases where usable visual evidence is present in the model but is overridden at the token decision by a stronger prior-driven text route. In this regime, the visual and text routes push in opposite directions, giving CRG a decision-aligned conflict signal: suppressing the selected text route allows the existing visual evidence to carry more influence. By contrast, if both routes already support the same wrong answer, CRG has no internal conflict signal to exploit. For example, in a rare-attribute case such as a pink banana, the visual representation may itself collapse to a stereotyped “yellow banana”

feature, while the language prior also favors “yellow banana”. CRG cannot create missing visual counter-evidence or flip a visual route that is itself biased; such cases are missed-correction cases caused by deficient or biased visual encoding, rather than failures of the intended conflict-aware gating mechanism. This distinction is why we do not make a universal claim over OCR, reasoning, or all attribute errors: CRG is expected to help when the error is mediated by a text-over-vision route conflict.

A second boundary appears when visual evidence is weak or ambiguous. In these cases, language priors may provide useful contextual guesses, whereas CRG tends to suppress prior-driven affirmative answers when estimated visual support is weak. The degraded-evidence case study in Appendix H illustrates this behavior. We therefore view this regime as a faithfulness–helpfulness tradeoff: CRG favors calibrated uncertainty and grounded faithfulness over unsupported, but sometimes useful, guessing.

This scope also determines how to interpret results on broader hallucination categories. MME, MMHal-Bench, and AMBER include attributes, colors, spatial relations, OCR-sensitive categories, and higher-level multimodal judgments; CRG is expected to help on these categories only when the error is mediated by visual-text route conflict. Errors dominated by poor perception, OCR, weak visual representations, or multi-step reasoning require complementary upstream or reasoningoriented methods. Finally, CRG is a white-box inference-time intervention requiring access to activations, gradients, and route-level patching, so it is aimed at open-weight or self-hosted LVLMs rather than closed-source API-only systems. Adapting the idea to black-box settings would require output-level proxies, such as sensitivity under image/text perturbations, and is left for future work.

# F.2. Cross-Gate Interactions

CRG ranks heads using local first-order route effects, but the final intervention applies multiple gates jointly. This creates the possibility of higher-order interactions: suppressing one text route can change downstream hidden states, which may in turn change the effect of another gate in a later layer. Thus, the first-order theory justifies the local ranking and selection signals, but it does not fully characterize all cross-gate couplings during the full autoregressive computation.

A local view makes this limitation explicit. Let g collect the selected gate values and let $\ell ( g )$ denote the decision score after applying these gates. Around the unmodified model $g = { \bf 1 }$ , we can write

$$
\ell (g) - \ell (\mathbf {1}) \approx \sum_ {i} \left. \frac {\partial \ell}{\partial g _ {i}} \right| _ {\mathbf {1}} (g _ {i} - 1) + \frac {1}{2} \sum_ {i \neq j} \frac {\partial^ {2} \ell}{\partial g _ {i} \partial g _ {j}} (\tilde {g}) (g _ {i} - 1) (g _ {j} - 1) + \dots .
$$

The first-order terms are the local signals used by CRG for ranking and selection. The cross terms correspond to the concern that one gate may change the effect of another, especially across layers after the hidden state has already been modified.

We control this interaction risk through design choices that reduce both the number and the magnitude of possible cross terms, although they do not make the interaction terms vanish. First, the intervention is sparse: CRG selects only a small number of low-VRI heads within the chosen layer range, instead of gating all conflicting heads. Since the second-order sum involves pairs of selected gates, sparsity directly limits how many gate–gate interaction terms can contribute. This also makes the intervention easier to monitor empirically, because failures caused by interactions would have to arise from a small selected set rather than from dense global suppression.

Second, each selected gate is route-specific and graded. CRG modifies only the text-route contribution of a selected head, leaving the visual route and the rest of the head computation intact. Thus, the perturbation is not a full head deletion but a partial change of the form $( g _ { i } - 1 ) O _ { i } ^ { \mathrm { t x t } }$ . In the cross term, this makes the effective interaction scale depend on the product of two graded deviations, $( g _ { i } - 1 ) ( g _ { j } - 1 )$ , and on the text-route components being attenuated. This is substantially milder than removing entire heads, which would perturb both visual and text routes and could introduce larger downstream state shifts.

Third, the effective gates concentrate in a contiguous mid-to-upper layer block. This matters because perturbations in very early layers can propagate through many later transformations before another gate is applied, creating more opportunities for long-range accumulation. By contrast, a localized mid-to-upper block applies the intervention after lower-level visual features have already been formed and over a shorter downstream path. The gates can still interact within this block, but the design avoids broad, all-layer perturbations and reduces the chance that early route changes cascade through the full decoder before later gates act.

Empirically, strong negative complementarity would likely appear as brittle sensitivity to k, γ, or layer range, or as the full Conflict A+B intervention underperforming a single-branch variant. Instead, Table 5 shows that the combined A+B intervention is consistently strongest, while the layer-range and hyperparameter sweeps in Figures 6 and 7 show smooth behavior. We therefore view cross-gate interaction as a real but secondary limitation: it is not fully eliminated by the current formulation, but the sparse and graded design keeps it controlled in practice.

# F.3. Per-Token CRG in Open-Ended Generation

For open-ended generation, CRG is not based on a single attribution computed once and then reused for the entire answer. Instead, it is recomputed at each decoding step t: we define

$$
\ell_ {t} = \log p (y _ {t} \mid x, y _ {<   t}),
$$

estimate visual/text route effects for the current token, apply conflict-aware gating, and then proceed to the next step. The final answer is therefore shaped by a sequence of local decision-time interventions.

At the same time, not every generated token is equally tied to the final grounded semantic decision. In a sentence such as “It is a lovely dog,” content-bearing tokens such as “dog” are more directly grounded in the image, whereas function words such as “It”, “is”, and “a” mainly support linguistic realization and fluency. This raises a valid concern: applying CRG at every token may introduce intervention on positions that are only weakly related to the final grounded content.

We examined three practical strategies on CHAIR using LLaVA-1.5-7B with max new tokens=128. The first is an oracle-like vocabulary-triggered strategy that applies CRG only when the current token matches the closed CHAIR object vocabulary, i.e., the MS-COCO object categories and their synonyms. The second is our default step-wise CRG, which applies CRG at every decoding step while still selecting heads and gates based on the current token’s route conflict. The third is a first-token-only variant motivated by the observation that early token distributions can contain response-level signals.

Table 19. Comparison of token-level intervention strategies on CHAIR with LLaVA-1.5-7B. Lower Cs and Ci are better; higher recall is better. 

<table><tr><td>Setting</td><td> $C_s \downarrow$ </td><td> $C_i \downarrow$ </td><td>Recall↑</td><td>Generalizable?</td></tr><tr><td>CHAIR-vocabulary-triggered CRG</td><td>33.4</td><td>10.6</td><td>78.6</td><td>No</td></tr><tr><td>Step-wise CRG (ours)</td><td>34.2</td><td>11.2</td><td>77.8</td><td>Yes</td></tr><tr><td>First-token-only CRG</td><td>45.3</td><td>14.2</td><td>77.1</td><td>Yes</td></tr></table>

The vocabulary-triggered variant is slightly stronger than step-wise CRG, which is expected because it uses benchmarkspecific knowledge about which tokens are evaluated by CHAIR. However, the gap is small, suggesting that the most important hallucination-related conflicts are concentrated near content-bearing positions even when CRG is evaluated at every step. The first-token-only variant is more efficient and still meaningful, but it underperforms the two token-adaptive strategies, indicating that later decoding positions remain important because hallucinated objects and attributes often emerge after the first token. We therefore use step-wise CRG as the default because it is general across open-ended generation settings without relying on benchmark-specific vocabularies, while acknowledging that more selective semantic triggering is a promising future direction.

# G. Computational Cost Analysis

Table 20 reports a per-decoding-step runtime breakdown on a CHAIR dialogue with max new tokens=512 using LLaVA-1.5-7B on a single A40 GPU under PyTorch eager attention. Each entry reports the average per-decoding-step time of each module and its percentage share (averaged over the generated steps).

The per-step cost is dominated by standard model execution. The baseline forward pass (base forward, 39.58%) and the gated forward pass (gated forward, 42.83%) together account for 82.41% of the time (114.73 ms/step), reflecting that CRG performs an additional forward pass during decoding. Estimator-specific overhead is modest: computing local sensitivities (grad, 6.30%) and aggregating head-route effects (compute CRE, 2.84%) sum to 9.14% (12.72 ms/step). The gate computation itself is lightweight (gate compute, 1.12%; 1.56 ms/step), and the remaining overhead is grouped into other (7.34%).

Overall, the breakdown confirms that the runtime overhead mainly comes from standard forward computation shared across methods, while the additional gradient- and CRE-related components remain a small fraction of the per-step latency.

Table 20. Per-decoding-step runtime breakdown on a single CHAIR dialogue using LLaVA-1.5-7B on a single A40 GPU with max new tokens=512 under PyTorch eager attention. Each entry reports the average time per decoding step (averaged over the generated steps) and its percentage share. The total matches the per-step latency of CRG reported in Table 6. 

<table><tr><td>Block</td><td>Avg (ms/step)</td><td>Share (%)</td></tr><tr><td>base_forward</td><td>55.10</td><td>39.58</td></tr><tr><td>grad</td><td>8.77</td><td>6.30</td></tr><tr><td>compute_CRE</td><td>3.95</td><td>2.84</td></tr><tr><td>gate_compute</td><td>1.56</td><td>1.12</td></tr><tr><td>gated_forward</td><td>59.63</td><td>42.83</td></tr><tr><td>other</td><td>10.22</td><td>7.34</td></tr><tr><td>Total</td><td>139.22</td><td>100.00</td></tr></table>

Runtime and memory overhead. Table 21 compares average latency (reported per decoding step) and peak GPU memory under the same decoding setup for LLaVA-1.5-7B. CRG achieves the best overall performance, improving POPE accuracy from 81.37 to 87.71 and reducing CHAIR $C _ { s }$ from 52.8 to 34.2. In terms of efficiency, CRG incurs a 2.27× latency increase (139.2 ms/step), which is close to VCD (2.01×; 123.2 ms/step) and M3ID (2.03×; 124.5 ms/step), while OPERA and HALC are substantially slower (7.12× and 6.52×, respectively). Notably, CRG introduces essentially no additional peak memory overhead (×1.00; 14950 MB), whereas OPERA and HALC increase memory usage by more than 1.5×.

Table 21. Runtime and peak GPU memory overhead under the same decoding setup for LLaVA-1.5-7B. 

<table><tr><td>Method</td><td>Avg. Latency↓</td><td>GPU Memory↓</td><td>POPE-Average↑</td><td>CHAIR  $C_s \downarrow$ </td></tr><tr><td>Regular</td><td>61.3 ms (×1.00)</td><td>14945 MB (×1.00)</td><td>81.37</td><td>52.8</td></tr><tr><td>VCD (Leng et al., 2024)</td><td>123.2 ms (×2.01)</td><td>15749 MB (×1.05)</td><td>84.66</td><td>51.6</td></tr><tr><td>M3ID (Favero et al., 2024)</td><td>124.5 ms (×2.03)</td><td>15575 MB (×1.04)</td><td>85.39</td><td>48.3</td></tr><tr><td>OPERA (Huang et al., 2024)</td><td>435.7 ms (×7.12)</td><td>22706 MB (×1.52)</td><td>85.69</td><td>44.6</td></tr><tr><td>HALC (Chen et al., 2024)</td><td>399.4 ms (×6.52)</td><td>23084 MB (×1.54)</td><td>86.32</td><td>39.7</td></tr><tr><td>CRG</td><td>139.2 ms (×2.27)</td><td>14950 MB (×1.00)</td><td>87.71</td><td>34.2</td></tr></table>

# H. Case Studies

Figure 8 serves as an illustrative case study that isolates the causal role of visual evidence by constructing controlled counterfactual/ambiguous variants: panel (a) is an author-taken photo, while panels (b–d) are generated with Gemini 3 Pro Image (Google DeepMind, 2025) by removing the queried object, replacing it with a visually similar distractor, or degrading the evidence via blur. These examples are not included in quantitative evaluation. In our framework, the decision is summarized by the logit gap $m = \ell _ { \mathrm { y e s } } - \ell _ { \mathrm { n o } } ,$ , and we estimate per-head, per-route causal gains $( \hat { \Delta } _ { l , h } ^ { \mathrm { v i s } } , \hat { \Delta } _ { l , h } ^ { \mathrm { t x t } } )$ as interventional changes in m under route-specific do-operations. Intuitively, when the image truly contains sheep (a), both routes tend to support the same decision so the estimated gains are sign-aligned and we keep the head unchanged. In contrast, in the counterfactual no-sheep setting (b) and the horse-as-distractor setting (c), regular decoding still produces an affirmative answer, which is consistent with a modality conflict pattern where the text route pushes m upward $( \hat { \Delta } _ { l , h } ^ { \mathrm { t x t } } > 0 )$ despite weak or negative visual support $( \hat { \Delta } _ { l , h } ^ { \mathrm { v i s } } \leq 0 )$ . Our intervention resolves this conflict by attenuating only the text route (via $g _ { l , h } ^ { \mathrm { t x t } } \in [ 0 , 1 ] )$ , thereby reducing unsupported increases in m and correcting false positives. Finally, under ambiguous evidence (d), we observe the same mechanism: regular decoding commits to “yes”, while our method becomes conservative because the estimated visual support is weak and conflicting, so suppressing the conflicting text-route prevents an over-confident affirmative claim.

![](images/442bd7676059353ebb9156a311004a74240db834be077f3d799db9b423e8f2b0.jpg)

<details>
<summary>natural_image</summary>

Two sheep grazing in a grassy field with misty mountains in the background (no text or symbols visible)
</details>

Is there a sheep in this image?there a sheep in this image?

(Answer of Regular decoding) ted answer of Regular decoding Yes-there are sheep grazing in the oks like there is a sheep, meadow.e image is

![](images/99a627e5e7d5bf422e2b238f117dfddc21b83210a0907d87fce526fc59c36b3c.jpg)  
Is there a sheep in this image?there a sheep in this image?

(Answer given by ours)  answer given by ours) Yes. Two sheep are visible in the meadow.n due to blur. l can’t confidently  N

![](images/8dc99e134a5703f76a7a10fc86ffd24ee94c655f9ceebeaedd46dbf7980bf6b2.jpg)  
(a) True positive preserved: sheep present; both methods answer correctly.

![](images/30a33322620283ee0707d99ec67b732ace36df80839124a85ff9658664478658.jpg)

<details>
<summary>natural_image</summary>

White horse galloping on a grassy hillside with misty mountains in the background (no text or symbols visible)
</details>

Is there a sheep in this image?

(Hallucinated answer of Regular decoding)( Yes, l can see a sheep in the meadow.

![](images/e9ed72f93d7948345ec9005444ae193dc0f6eb94472a714be82bf65565e98fde.jpg)  
Is there a sheep in this image?

(Correct answer given by ours) No. The animals appear to be horses, not sheep.

![](images/66d271c6a79c98c37a0cf580c973f48df7a70207f72e81ec9730f028500718e3.jpg)  
(c) Object confusion corrected: horses are mistaken as sheep by regular decoding.

![](images/a88570c7b84cc0332375c1f01d950447c26522b7aa3180588e751f78f04ef45f.jpg)

<details>
<summary>natural_image</summary>

Mountainous landscape with misty peaks and sparse trees in the foreground (no text or symbols visible)
</details>

Is there a sheep in tIs there a sheep in this image?

(Hallucinated answer of Regular decoding Yes, l can see a sheep in the meadow.

![](images/20e0d6eb214527a0d0c0151c1e1c83340739e0c5d4792b36676c16bb25519ae2.jpg)  
Is there a sheep in tIs there a sheep in this image?

ly No. The animals appe(Correct answer given by ours) ar to be not sheep.No. l don't see any sheep in the meadow.

![](images/e7b23e317925653430aa0d15e0cc6d96ff43acfe865e1033e861fea50174ff7d.jpg)  
(b) False positive corrected: no sheep in the image; regular decoding hallucinates “yes”.

![](images/8f340945eba2ac0277ea983e3fdc210e6cbeebdbdb688d3ee577922f5fc5302e.jpg)

<details>
<summary>natural_image</summary>

Pastoral scene with grazing sheep in a mountainous landscape under overcast skies (no text or symbols visible)
</details>

Is there a sheep in this image?

(Hallucinated answer of Regular decoding) ing) Yes, it looks like there is a sheep, though the image is blurry.

![](images/3771968fa93c0cd76a7e3daeb5b42a799f5babe628ca5505322db19408d83e57.jpg)  
Is there a sheep in this image?

(Correct answer given by ours) Uncertain due to blur. l can’t confidently w.identify the animals as sheep.

![](images/7d45fa5281112b47f355392ec011c6222e4a51cb589b59b1e8ec23bc11f3c5da.jpg)  
(d) Cautious under weak evidence: the image is blurry; regular decoding still answers “yes”, while our method expresses uncertainty instead of making an unsupported claim.   
Figure 8. Case study with real and AI-generated images. Panel (a) is an author-taken photograph, while panels (b–d) are generated with Gemini 3 Pro Image (Google DeepMind, 2025) to create controlled counterfactual or ambiguous variants for visualization. Regular decoding shows a tendency toward unsupported affirmative answers and object confusion, whereas our method suppresses over-confident “yes” responses and is conservative under weak evidence.