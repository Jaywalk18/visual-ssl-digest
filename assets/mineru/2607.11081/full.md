Benchmark

# Controlling Motion Transfer in Difusion Transformers via Attention Heads

Sunyoung Jung<sup>1\*</sup> , Jiwoo Park<sup>1,2\*</sup> , Yoonseok Choi<sup>1</sup> , Kyobin Choo<sup>1</sup> , Ming-Hsuan Yang<sup>3</sup> , and Seong Jae Hwang<sup>1†</sup>

<sup>1</sup>Yonsei University <sup>2</sup>LG Electronics <sup>3</sup>University of California, Merced Project page: https://sunyj-hxppy.github.io/halo

![](images/db38fb280442222f2e6887d0f17fe9fdfba691393f568f1c52334d53cb24d648.jpg)  
A red Porsche drives beside a lake at sunrise  
In-the-wild Movie Scene

![](images/c609f395dc6e8c6fd0f77d079466fb6fbab5879740ec4b2ecb2f73258056f867.jpg)  
Two stormtroopers riding mechanical steeds across a concrete field.

Fig. 1: Overview. We present HALO, a head-aware controllable motion transfer framework for video Difusion Transformers, which identifies motion- and structurespecialized attention heads within the model. Leveraging these findings, HALO generates videos that follow the target prompt while remaining motion- and structure-aligned with reference videos, achieving accurate motion transfer.

Abstract. Difusion Transformers (DiTs) have advanced video generation with high-quality, temporally coherent results. However, extending them to motion transfer, which requires following reference motion while aligning with a target prompt, remains challenging due to limited understanding of motion and structure representations within DiTs. We analyze video DiTs at the attention-head level and identify distinct heads specialized for motion and spatial structure. Based on this insight, we propose a head-aware controllable motion transfer framework that requires no parameter updates. Our method refines motion cues from motion-specialized heads via semantic correspondence guidance and preserves structure through selective feature injection. This head-level control not only enables accurate motion transfer but also provides an interpretable foundation for controllable video generation with DiTs.

Keywords: Motion Transfer · Difusion Transformers · Attention Heads

## 1 Introduction

Motion transfer in video generation synthesizes a video that follows the motion of a reference video while adhering to a target prompt. The primary objectives are (1) motion fidelity, ensuring temporal adherence to the reference motion, and (2) structural alignment, maintaining spatial layout of the reference [14,48]. Achieving these goals requires modeling of spatio-temporal dependencies, an area in which recent video Difusion Transformers (DiTs) [25, 40, 46, 47] have shown strong capability. Given their ability to capture spatial structure and temporal dynamics, DiTs have become a natural choice for motion transfer [7, 14, 35].

Existing DiT-based motion transfer approaches, such as noise warping [7], rotary positional embedding manipulation [14], and cross-frame attention optimization [35], ofer varying degrees of motion controllability. However, these methods focus on manipulating motion representations without an understanding of how motion and structure are encoded within DiTs. This lack of understanding is due to the distributed functionality of DiTs, which makes their internal mechanisms challenging to analyze [3, 8, 12]. Consequently, generated videos often exhibit seemingly plausible motion yet inaccurate trajectories or misaligned object structures relative to the reference video. As illustrated in Fig. 1, Go-withthe-flow (GWTF) [7], a state-of-the-art method, exhibits motion deviations, including a car moving straight instead of turning and two stormtroopers with misaligned positions compared to the reference.

Therefore, we conduct a detailed analysis of video DiTs, focusing on the attention heads to answer the fundamental question: How are motion and structural cues internally encoded in video DiTs? To identify the heads specialized in modeling motion, we introduce the first head-level analysis based on displacement maps. The displacement map [35] encodes motion as patch-wise coordinate diferences between frames, making it efective for analyzing motion properties of heads. For structural cues, we compute the visual token attention-map entropy, which represents the uncertainty in patch dependency. By utilizing the entropy, we select heads that reliably capture structural information.

Our analysis identifies two distinct attention head subsets within video DiTs: (1) Motion-Specific Heads. A subset of heads exhibits strong patch correspondences across frames, capturing temporally coherent motion cues in the displacement maps. These heads present clear displacement maps that accurately reflect the motion, demonstrating their superior capability to represent coherent motion flow. (2) Structure-Specialized Heads. Another subset of heads encodes structural information. The attention maps of the heads show low entropy, characterized by sharply diagonal patterns. This attention pattern yields attention features with concentrated structural information, confirming that attention map entropy serves as an indicator of the structural content.

Building on these findings, we propose HALO, a head-driven semantic-structu ral motion transfer framework that simultaneously enforces motion fidelity and structural alignment. By leveraging the properties of attention heads in video DiTs, our approach first constructs inter-frame displacement maps using only motion-specific heads to guide motion optimization. These displacement maps capture the motion dynamics but lack semantic information. This often leads to transferring motion into semantically irrelevant regions, causing inconsistent or reversed object motion, as shown in Fig. 2(a). To overcome this issue, we introduce semantic guidance modules that align motion cues with semantic similarities derived from difusion features, known to encode rich object-level semantics [38, 41]. This refinement produces semantically aligned displacement maps, enabling temporally consistent and semantically coherent motion transfer.

![](images/497bbbcb310effff01772e2f4fa4d1990ef8d1725055ba910bf822114d0aa5dd.jpg)  
Fig. 2: Limitations of displacement-based motion transfer. Comparison between HALO and displacement-only optimization [35]. (a) Lack of semantic alignment causes motion errors. (b) Missing structural preservation leads to spatial misalignment. HALO ensures consistent motion and spatial fidelity

While the refined displacement map provides motion flow information, it remains insuficient for preserving the spatial layout of the reference video. Consequently, the generated results often exhibit structural misalignment with the reference, as illustrated in Fig. 2(b). To overcome this limitation, we leverage our second insight that low-entropy attention heads primarily encode structural information. Specifically, we introduce selective structural feature injection by incorporating attention features from the reference video through these low-entropy heads. This head selection reinforces structural information while suppressing noisy and difuse cues, yielding motion transfer that is structurally aligned with the reference.

## The main contributions of this work are:

– We present a head-level functional analysis of video DiTs, revealing motion-specific and structure-specialized attention heads and validating them as control primitives for controllable video generation.

– We propose HALO, a head-aware controllable motion transfer framework that enhances motion fidelity through semantic-aware displacement optimization and preserves spatial consistency via selective feature injection from structurally informative heads.

– We perform extensive evaluations on standard motion transfer benchmarks and our Movie Scene Dataset, demonstrating that it achieves better motion coherence and structural alignment.

## 2 Related Work

Text-to-Video Generation. Early Text-to-Video (T2V) generation was driven by U-Net-based difusion models [9,23,29], which extended image difusion with temporal attention or 3D convolutions to capture spatio-temporal dynamics [4, 6,15]. More recently, DiTs have emerged as the dominant paradigm [10,16,25,40, 47], significantly improving temporal consistency and visual quality. Motivated by these advances, we build our framework upon a video DiT backbone.

Motion Transfer. Motion transfer aims to generate a target video by extracting and reflecting only the motion information from a reference video [13,33,37,48]. The main challenge of this task lies in efectively decoupling the motion and appearance information from the reference [21, 45].

Early studies have addressed this challenge by extracting motion embedding from the temporal modules of U-Net [27, 42, 50]. Recently, the strong generative capability of DiTs has motivated their adoption in the motion transfer field. For instance, GWTF [7] explicitly warps noise to manipulate the motion, while RoPECraft [14] handles motion by warping the RoPE embeddings using optical flow. Furthermore, DiTFlow [35] derives displacement from cross-frame attention to guide patch movement. While these prior works have explored various ways to handle motion information in DiTs, there has been a lack of direct analysis of how motion and structure are encoded within DiTs’ internal representations.

To bridge this analytical gap, our work analyzes DiT internal representations to identify motion-specific and structure-specialized heads. By leveraging these heads, we achieve training-free motion transfer and extend our approach to controllable video generation.

Attention in Video DiTs. U-Net-based video difusion models [9, 23] explicitly separate spatial and temporal attention, enabling motion extraction from temporal modules [21,27]. In contrast, DiTs [40,47] adopt unified attention that jointly models spatial and temporal information. While this design enhances expressiveness, it obscures the distinction between motion and structure, complicating motion-specific control [28, 35]. In contrast, we posit that this motion information is independently encoded within the unified structure of attention in the DiTs. Consequently, we focus on analyzing this unified attention structure at the level of individual attention heads to identify the distinct encoding of motion and structure in terms of motion transfer.

## 3 Analyzing Attention Heads of Video DiTs

We focus on understanding how motion and structural information are represented within video DiTs [47]. Here, we present a head-level analysis of attention heads to uncover their distinct functional roles in modeling motion and spatial structure, ofering a mechanistic perspective overlooked in prior studies [1,24,44].

Analysis 1: Motion-Specific Heads. We aim to identify the motion-specific properties encoded within individual attention heads. Building on SparseGen [44], which identifies temporal and spatial head patterns for eficient video generation (see Fig. 3(a)), we first classify heads based on their similarity to pattern masks (e.g., cross-frame diagonal for temporal, sparse localized patterns for spatial). Based on this, we construct head-specific displacement maps from cross-frame attention [31,35]. Specifically, cross-frame attention is computed using the queries Q and keys K derived from the latent features $z _ { f } \in \mathbb { R } ^ { h e a d \times N \times d }$ as:

![](images/95d924c300a4e988b22a14ffd8437fda23a0755b78c7536256f0151ae94d5d32.jpg)  
Fig. 3: Head Configuration Comparison. (a) Attention maps show distinct patterns: temporal heads capture cross-frame diagonals, while spatial heads maintain intraframe locality. (b) Quantitative evaluation using directional alignment and correlation shows that temporal heads align more closely with reference motion. (c) Displacement maps further confirm temporal heads more accurately capture motion flow.

![](images/46b728a5b479da3b3b3b80a99dc6b39dc276fa5e64bc661b7351a57736829286.jpg)  
Fig. 4: Relationship between attention maps and structural cues in hidden features. Heads with lower attention entropy show lower feature entropy, indicating stronger structural fidelity. Example: L20 H8 denotes the 8th attention head in the 20th layer.

$$
\mathbf {M} _ {f, f ^ {\prime}} ^ {h} = \mathrm{softmax} \left(\frac {Q _ {f} ^ {h} \cdot K _ {f ^ {\prime}} ^ {h ^ {\top}}}{\sqrt {d}}\right),\tag{1}
$$

where h denotes the attention head, $N = H \times W$ is the number of spatial tokens, and $d$ is the feature dimension. Here, f and $f ^ { \prime }$ represent the frame indices. We then derive a head-specific displacement map $\mathcal { D } _ { f , f ^ { \prime } } ^ { h } \in \mathbb { R } ^ { H \times W \times 2 }$ , which captures the patch-wise motion between frames $( f , f ^ { \prime } )$ , defined as:

$$
\mathrm{I} _ {f, f ^ {\prime}} ^ {h} = \underset {f ^ {\prime}} {\operatorname{argmax}} \left(\mathbf {M} _ {f, f ^ {\prime}} ^ {h}\right), \quad \mathcal {D} _ {f, f ^ {\prime}} ^ {h} (i) = \mathbf {g} \big (\mathrm{I} _ {f, f ^ {\prime}} ^ {h} (i) \big) - \mathbf {g} (i),\tag{2}
$$

where $\mathbf { g } ( \cdot )$ converts a 1D patch index into 2D spatial coordinates and i denotes the patch index.

Using displacement maps, we analyze cross-frame motion patterns across attention heads to quantify motion capture efectiveness, as shown in Fig. 3(c). Temporal heads capture motion variations more accurately than spatial or combined heads, showing stronger alignment with object motion (blue) and background movement (red). To validate these observations, we measure the similarity between predicted displacement maps and ground-truth optical flow [39] through two quantitative metrics: (1) Directional Alignment (DA), calculating cosine similarity between motion directions, and (2) Correlation (Corr), using Pearson correlation for global pattern consistency.

As illustrated in Fig. 3(b), temporal heads consistently achieve higher DA and Corr scores. These results confirm that while spatial heads attend to intraframe patch relationships, temporal heads specialize in cross-frame dependencies, efectively encoding motion-related information within video DiTs.

Analysis 2: Structure-Specialized Heads. To identify attention heads that encode structural information, we analyze the visual token attention maps $A ^ { h }$ corresponding to each head. Our analysis reveals distinct patterns across diferent heads: Heads exhibiting well-defined diagonal alignment patterns with low entropy (e.g., L20 H8) stand in contrast to those with irregular or difuse attention distributions with high entropy (e.g., L20 H6), as shown in Fig. 4. Since the entropy serves to quantify the uncertainty in capturing the relationships between visual tokens, low entropy implies the presence of structural information.

To validate this, we compute the attention entropy [32] and the spatial entropy of corresponding attention features [5,22,34]. Given a head h with attention map $A ^ { h }$ , the attention entropy $\mathcal { H } _ { \mathrm { a t t n } } ^ { h }$ is defined as $\begin{array} { r } { \mathcal { H } _ { \mathrm { a t t n } } ^ { h } = - \sum _ { k = 1 } ^ { N } A _ { k } ^ { h } } \end{array}$ log $A _ { k } ^ { h }$ where lower values indicate a more concentrated attention distribution. To capture spatial properties in attention features, we compute the spatial entropy of PCA-transformed feature embeddings.

As illustrated in Fig. 4, lower entropy heads exhibit clear diagonal attention patterns and yield feature maps maintaining structural layouts. Fig. 4 also provides quantitative evidence of a strong correlation between lower attention entropy ${ \mathcal { H } } _ { \mathrm { a t t n } }$ and lower hidden feature spatial entropy $\mathcal { H } _ { \mathrm { h i d d e n } }$ . Therefore, these diagonal attention patterns indicate strong spatial patch correlations, validating that such heads are specialized in encoding structural information.

## 4 Method

We introduce $H A L O ,$ a head-aware controllable motion transfer framework that jointly achieves motion fidelity and structural alignment with a reference video (see Fig. 5(a)). Building on Sec. 3, we propose (i) semantic-aware motion optimization (Sec. 4.1) and (ii) structure-guided feature injection (Sec. 4.2), illustrated in Fig. 5(b)–(c).

## 4.1 Semantic-Aware Motion Guidance

For motion transfer, we construct cross-frame displacement maps within DiTs. Our analysis (Analysis 1, Sec. 3) shows that motion-specific heads M capture motion more efectively. We therefore aggregate cross-frame attention $\mathbf { M } _ { f , f ^ { \prime } } ^ { h }$ for $h \in { \mathcal { M } }$ , identified via temporal pattern masks to extract a displacement map.

While cross-frame attention provides patch-level correlation cues, it can overem phasize visually similar yet semantically unrelated regions, producing mismatched displacements and degraded motion alignment (e.g., camera-following movements that disregard objects and incorrect object trajectories; see Supp. Sec. B).

![](images/c11cb00a3835496831a268ef9c9fb88d1077223158b916a7042138cca880ef7f.jpg)  
Fig. 5: Overview of HALO. (a) From a reference video, we extract displacement maps and head features. Displacements from motion-specific heads guide motion by optimizing the latent representation, while selected head features from the reference are injected to preserve structure during generation. (b) To enhance motion guidance, semantic correspondence derived from difusion features refines the displacement map, ensuring semantically aligned motion flow. (c) For structural guidance, we inject value features from heads selected via entropy analysis, targeting heads that encode essential spatial information.

To mitigate this, we introduce a semantic-aware motion guidance that refines displacements using semantic similarities from semantically rich difusion features [17, 38, 41], as shown in Fig. 5(b).

Semantic Correspondence Refinement. To refine the reference displacement map, we first select the top-k attention candidates from cross-frame attention rather than relying on a single maximum, as illustrated in Fig. 6(a). We then construct semantic correspondence C using pairwise cosine similarity over difusion features ϕ:

$$
\mathcal {C} _ {f, f ^ {\prime}} = \frac {\phi_ {f} \cdot \phi_ {f ^ {\prime}}}{\| \phi_ {f} \| _ {2} \| \phi_ {f ^ {\prime}} \| _ {2}},\tag{3}
$$

For each patch index, we compute the distances between the most similar semantic indices $\Gamma _ { f , f ^ { \prime } } ^ { \mathrm { c o r } } \in \mathbb { R } ^ { N }$ between frames and the top-k attention candidates. The nearest candidate is then selected as the reference best-match coordinate $\Gamma _ { f , f ^ { \prime } } ^ { \mathrm { r e f } }$ . The reference inter-frame displacement map $\mathcal { D } ^ { \mathrm { r e f } }$ is then constructed from $\Gamma _ { f , f } ^ { \mathrm { r e f } }$ <sub>′</sub> following Eq. 2 (first row in Fig. 7).

Motion Optimization. Given $\mathcal { D } ^ { \mathrm { r e f } }$ , we optimize z<sub>T</sub> to align generated motion with the reference. We apply Semantic Reweighting (SRW) to adjust cross-frame attention via a correspondence-guided bias derived from C (Fig. 6(b)). Using I<sup>cor</sup>,

![](images/9ce9913e56323590ddba3163131eac0c6ec0063b697e2582f83c5c219347499f.jpg)  
Fig. 6: Details of Semantic Correspondence Refinement (SCR) and Semantic Reweighting (SRW). (a) $\Gamma _ { f , f ^ { \prime } } ^ { \mathrm { r e f } }$ is obtained by choosing, among top-k attention candidates, the patch closest to the semantic best match $\Gamma _ { f , f ^ { \prime } } ^ { \mathrm { c o r } } . \ ( \mathrm { b } )$ SRW manipulates target cross-frame attention by adding a correspondence-based bias at $\Gamma _ { f , f ^ { \prime } } ^ { \mathrm { c o r } }$

we form the bias matrix $B _ { f , f ^ { \prime } } \colon$

$$
\mathcal {B} _ {f, f ^ {\prime}} (i, j) = \left\{ \begin{array}{l l} \beta , & \text { if } j = \mathrm{I} _ {f, f ^ {\prime}} ^ {\mathrm{cor}} (i), \\ 0, & \text { otherwise }, \end{array} \right.\tag{4}
$$

where $\beta$ controls bias strength. The refined cross-frame attention scores are $\tilde { E } _ { f , f ^ { \prime } } = E _ { f , f ^ { \prime } } + B _ { f , f ^ { \prime } }$ , and refined attention is $\tilde { M } _ { f , f ^ { \prime } } = \mathrm { s o f t m a x } ( \tilde { E } _ { f , f ^ { \prime } } )$ . We then estimate displacement as an expectation over coordinate diferences,

$$
\Delta x _ {f, f ^ {\prime}} = \sum_ {i, j} (x _ {j} - x _ {i}) \tilde {M} _ {f, f ^ {\prime}} (i, j), \quad \Delta y _ {f, f ^ {\prime}} = \sum_ {i, j} (y _ {j} - y _ {i}) \tilde {M} _ {f, f ^ {\prime}} (i, j),\tag{5}
$$

and stack them in $\mathcal { D } _ { t } ~ = ~ [ \varDelta x _ { f , f ^ { \prime } } , \varDelta y _ { f , f ^ { \prime } } ]$ (second row in Fig. 7). Finally, to optimize the latent for motion alignment, we minimize the semantic motion loss $L _ { S M }$ , defined as the L2 distance between the reference displacement map $\mathcal { D } _ { \mathrm { r e f } }$ and the target displacement map $\mathcal { D } _ { t }$ . Incorporating semantic cues into displacement estimation unifies motion alignment with semantic coherence, yielding consistent motion transfer. Notably, this strategy represents the first attempt to incorporate semantic correspondences within motion representations, facilitating high-fidelity, semantically consistent video generation.

## 4.2 Selective Structural Head Injection

Although displacement maps capture directional motion, they lack suficient intra-frame structure, which can cause spatial misalignment with the reference (Fig. 2, right). To address this, we propose a structural guidance mechanism for preserving the spatial integrity of the reference. We inject reference value features during generation. Naïvely injecting all head features, however, introduces artifacts such as noise and identity leakage [2] (see Supp. Sec. C.4, Supp. Fig. 6).

![](images/29da8d4a74a2d361c32c7f0b798ad73f62263747b67b96991aa70bb0e6388efd.jpg)  
Fig. 7: Efect of SCR and SRW on displacement maps. The refinements improve robustness to fine-grained object motion and better preserve object shape.

We therefore select structurally informative heads via entropy-based analysis (Analysis 2, Sec. 3). Low-entropy heads exhibit strong diagonal attention and produce spatially coherent features; we inject their value features into the corresponding heads, supplying structure-aware guidance without noise or leakage (Fig. 5(c)). This selective injection complements motion optimization and enforces spatial alignment with the reference.

## 5 Experiments

Implementation Details. Following prior works, we evaluate our approach using a standard motion transfer benchmark and settings [14, 36, 48]. Across all experiments, we adopt the video difusion transformer CogVideoX [47] as the base model, using 50 denoising steps, 12 optimization steps $T _ { \mathrm { o p t } }$ , and a guidance scale of 7, consistent with common practice. For additional results with the video DiT model, Wan [40], please refer to Supp. Sec. D. The bias strength β for semantic reweighting is fixed to 0.1, and the entropy range τ for selective structure specialized-head injection is set to $\tau < 7$ , based on 7 being the median entropy value observed in our analysis (validated in Section 5.3). Hyperparameter configurations $( \mathrm { e . g . } , \beta , T _ { \mathrm { o p t } } , \tau , \mathrm { t o p } – k )$ and HALO optimization algorithm are detailed in Supp. Sec. C.1 and C.2, respectively.

Baselines. We compare against representative motion-transfer approaches, including U-Net-based methods (MOFT [45], ConMO [13], and MotionClone [27]) and recent DiT-based methods (DiTFlow [35], RoPECraft [14], and GWTF [7]).

Movie Scene Dataset. Motion transfer has emerged as a pivotal task in video production and digital cinematography, where it is increasingly demanded to replicate sophisticated film-style shot compositions and intricate post-production efects [30]. However, existing benchmarks [27,36] are largely derived from generic internet videos (e.g., DAVIS/WebVid), which do not fully reflect these professional requirements. Therefore, we curate a Movie Scene Dataset specifically for the motion transfer task (see Supp. Sec. E for details). Complementing existing benchmarks, this dataset provides a production-oriented evaluation setting to assess whether methods can achieve faithful motion transfer under real-world cinematic conditions.

Reference  
HALO  
GWTF  
RoPECraft  
DiTFlow  
ConMO  
MotionClone  
![](images/b5db15d9fb992358175278b3921c271af6404a6237eb90f096b5a72fac8f182d.jpg)

The man is riding ahorse on a bridge, passing a roadside sign  
![](images/45516c22d2c6fb06dc209c396948feadc591fb60fc09a57dec6c6421ac0f11f4.jpg)

A jet plane racing through the sky, with smoke trailing behind it.  
![](images/4ab5fc4a20db4de3d9205a8655200b59a304dc40cd2aadea4eed0021acc4b3d7.jpg)  
A child riding a skateboard on a parkpathway.  
Fig. 8: Qualitative comparison between U-Net- and DiT-based baselines and HALO.

## 5.1 Main Results

Qualitative Results. Fig. 8 presents qualitative comparisons with state-of-theart models. Baseline methods often exhibit camera drift, generation failures, or identity leakage (e.g., generating a motorcycle instead of a horse). They also produce semantically inconsistent motion, such as misaligned orientations between correlated objects (e.g., car vs. airplane). In contrast, our method produces semantically coherent motion aligned with the reference while preserving subject identity and structural integrity. Furthermore, in Fig. 9, HALO exhibits remarkable robustness on the Movie Scene Dataset. Even when guided by complex cinematic prompts, HALO faithfully preserves the target’s identity integrity while following the reference dynamics.

Quantitative Results. We evaluate performance using CLIP score (CLIP) [18] for text–video alignment, Temporal Consistency (TC) [49] for frame coherence, and Motion Fidelity (MF) [48] and FTD [14] for reference-motion alignment. As shown in Table 1, HALO achieves notable improvements in CLIP, MF, and FTD, demonstrating that head-level analysis within DiTs enables motion transfer with strong motion fidelity and structural alignment.

Since motion transfer requires following the reference motion while adhering to the target text prompt, CLIP and MF must be jointly evaluated. Prior

![](images/e3499c21a6f5be8d229a95a6e93663635ffe115a1b015176dd7d50da27fae7ca.jpg)  
A SpongeBob-style adventurer rides a bouncing anemone-creature across a smooth open ground.

Fig. 9: Qualitative results in the movie scene dataset.

works exhibit a trade-of between these metrics, as shown in Table 1. In contrast, our method achieves balanced performance, maintaining high CLIP and MF scores. This indicates that our head-analysis-based formulation transfers reference motion efectively while suppressing irrelevant semantic information. Although HALO reports a slightly lower TC than DiTFlow and GWTF in Table 1, TC tends to favor more static outputs. Further analysis of the CLIP–MF balance and the relationship between motion dynamics and temporal consistency is provided in Supp. Sec. F.2. We also report the runtime and peak GPU memory in Supp. Sec. F.3, showing that HALO incurs only marginal overhead relative to the displacement optimization framework.

User Study. To ensure an assessment aligned with human preference, we conduct a user study to evaluate HALO across three criteria: editing accuracy, temporal consistency, and motion accuracy. Twenty participants rated each result on a 5-point Likert scale (1: lowest, 5: highest), with scores subsequently normalized to a 0–100 range. As reported in Table 4, HALO achieves the highest mean score across all criteria, demonstrating a consistent preference among participants.

## 5.2 Ablation Study

We perform ablation studies under conditions identical to the main experiments, using displacement optimization [35] as the baseline.

Efect of Semantic-Aware Motion Guidance. To evaluate the role of semantic correspondence refinement in displacement computation, we directly compute displacement maps from raw cross-frame attention. As shown in Fig. 10 (green

Table 1: Quantitative results with state-of-the-art motion transfer methods. Best and second results are represented with bold and underlined.

<table><tr><td>Model</td><td></td><td></td><td>CLIP ↑</td><td>TC ↑</td><td>MF ↑</td><td>FTD ↓</td></tr><tr><td rowspan="3">U-Net-based</td><td>MoFT [45]</td><td>NeurIPS&#x27;24</td><td>30.9</td><td>85.8</td><td>34.8</td><td>23.0</td></tr><tr><td>ConMO [13]</td><td>CVPR&#x27;25</td><td>29.8</td><td>85.3</td><td>52.0</td><td>17.4</td></tr><tr><td>MotionClone [27]</td><td>ICLR&#x27;25</td><td>30.4</td><td>78.6</td><td>55.6</td><td>19.8</td></tr><tr><td rowspan="4">DiT-based</td><td>DiTFlow [35]</td><td>CVPR&#x27;25</td><td>31.0</td><td>89.5</td><td>59.6</td><td>23.0</td></tr><tr><td>RoPECraft [14]</td><td>NeurIPS&#x27;25</td><td>30.3</td><td>85.9</td><td>58.2</td><td>19.6</td></tr><tr><td>GWTF [7]</td><td>CVPR&#x27;25</td><td>31.6</td><td>88.2</td><td>62.5</td><td>21.6</td></tr><tr><td>HALO</td><td></td><td>31.7</td><td>87.5</td><td>66.2</td><td>19.4</td></tr></table>

Table 2: Quantitative results in the Movie Scene dataset (methods requiring video masks are excluded).

<table><tr><td>Model</td><td>CLIP ↑</td><td>TC ↑</td><td>MF ↑</td></tr><tr><td>MotionClone [27]</td><td>27.1</td><td>74.5</td><td>47.7</td></tr><tr><td>DiTFlow [35]</td><td>29.7</td><td>90.4</td><td>48.4</td></tr><tr><td>RoPECraft [14]</td><td>28.3</td><td>88.2</td><td>49.1</td></tr><tr><td>GWTF [7]</td><td>30.2</td><td>87.6</td><td>46.6</td></tr><tr><td>HALO</td><td>30.5</td><td>88.9</td><td>52.6</td></tr></table>

Table 3: Ablation results of proposed components on performance. “Semantic” denotes the semanticaware motion guidance, and “Injection” denotes the selective structural head injection.  
Table 4: User preference study on motion transfer. The results represent the preference rate across three dimensions.

<table><tr><td>Exp.\#</td><td colspan="2">Semantic Injection</td><td>CLIP ↑</td><td>TC ↑</td><td>MF ↑</td><td>FTD ↓</td></tr><tr><td>1</td><td></td><td></td><td>31.0</td><td>89.5</td><td>59.6</td><td>23.0</td></tr><tr><td>2</td><td></td><td>√</td><td>30.6</td><td>88.3</td><td>61.8</td><td>20.5</td></tr><tr><td>3</td><td>√</td><td></td><td>30.9</td><td>88.5</td><td>61.3</td><td>20.9</td></tr><tr><td>4 (HALO)</td><td>√</td><td>√</td><td>31.7</td><td>87.5</td><td>66.2</td><td>19.4</td></tr></table>

<table><tr><td>Model</td><td>Edit Acc.</td><td>TC</td><td>Motion Acc.</td></tr><tr><td>DiTFlow</td><td>84.1</td><td>83.1</td><td>69.0</td></tr><tr><td>RoPECraft</td><td>72.4</td><td>75.6</td><td>77.6</td></tr><tr><td>GWTF</td><td>84.0</td><td>77.1</td><td>72.8</td></tr><tr><td>HALO</td><td>86.5</td><td>87.8</td><td>93.3</td></tr></table>

![](images/96e4c66592d7478cd09a66affef8de21649d4c33e81d2505b232e395bf3fddb1.jpg)  
The penguin is walking in a zoo enclosure, possibly towards its habitat pool.

Fig. 10: Qualitative results of our method across ablation studies. Exp.# corresponds to Table 3.

box), the absence of semantic refinement causes motion to the background rather than the actual moving object. This demonstrates that cross-frame attention alone captures visually similar but semantically unrelated regions, leading to motion misalignment. Semantic correspondence refinement is therefore crucial for achieving semantically consistent motion transfer.

Efect of Selective Structure-Specialized Head Injection. Removing selective structure-specialized head injection significantly degrades motion fidelity (Exp.#3 in Table 3). As shown in Fig. 10 (yellow box), this omission results in structural distortions and duplicated objects (e.g., two penguins). These findings indicate that injecting low-entropy structural features helps maintain spatial consistency, making explicit structural guidance essential for accurate motion transfer.

Table 5: Performance of motion-transfer methods across diferent motion dificulty levels categorized based on the complexity of motion trajectories.

<table><tr><td rowspan="2">Model</td><td colspan="2">Hard</td><td colspan="2">Medium</td><td colspan="2">Easy</td></tr><tr><td>CLIP ↑</td><td>MF ↑</td><td>CLIP ↑</td><td>MF ↑</td><td>CLIP ↑</td><td>MF ↑</td></tr><tr><td>MotionClone</td><td>26.8</td><td>39.0</td><td>24.8</td><td>37.0</td><td>27.8</td><td>39.0</td></tr><tr><td>ConMo</td><td>30.7</td><td>51.0</td><td>31.0</td><td>54.0</td><td>28.3</td><td>64.0</td></tr><tr><td>DiTFlow</td><td>32.1</td><td>54.0</td><td>32.5</td><td>67.0</td><td>32.1</td><td>66.0</td></tr><tr><td>RoPECraft</td><td>27.1</td><td>55.7</td><td>24.7</td><td>69.9</td><td>24.1</td><td>59.5</td></tr><tr><td>GWTF</td><td>33.6</td><td>56.0</td><td>32.7</td><td>69.0</td><td>30.8</td><td>74.0</td></tr><tr><td>HALO</td><td>33.2</td><td>59.2</td><td>33.8</td><td>72.3</td><td>31.4</td><td>75.5</td></tr></table>

Table 6: Analysis of head configuration and selective injection entropy threshold. (a) Diferent attention heads. (b) Validation of selective structural head injection entropy range τ .

(a) Attention Head  
(b) Entropy Range

<table><tr><td>Head</td><td>MF ↑</td><td>FTD ↓</td></tr><tr><td>All</td><td>63.1</td><td>21.6</td></tr><tr><td>Spatial</td><td>53.6</td><td>22.7</td></tr><tr><td>Motion</td><td>66.2</td><td>19.4</td></tr></table>

<table><tr><td>τ</td><td>MF ↑</td><td>FTD ↓</td></tr><tr><td>7 &lt;</td><td>54.1</td><td>23.5</td></tr><tr><td>&lt; 7</td><td>66.2</td><td>19.4</td></tr></table>

## 5.3 Additional Validations

Complex Motion. Robust motion transfer remains challenging under rapid movements and long-range trajectories. To evaluate the robustness of HALO in such settings, we utilize a benchmark stratified into three dificulty levels (Easy/Medium/Hard) based on trajectory complexity. We assess performance using prompt alignment (CLIP) and motion fidelity (MF), while omitting MoFT due to its lack of competitiveness in these scenarios.

As summarized in Table 5, HALO consistently achieves superior performance across all levels with minimal performance decay, indicating high stability. This robustness is further evidenced qualitatively in Fig. 11(a); our method handles extreme cases efectively, such as occlusions in a motorcycle jump and the highvelocity dynamics of boxing. Overall, these results validate that our approach remains stable across a wide spectrum of complex motions.

Validation of Motion-Specific Heads. To verify motion-head selection, Table 6a compares motion-specific, spatial, and all-head configurations. Spatial heads alone yield significantly lower MF and FTD, indicating a limited motionrelated signal for motion transfer. Although aggregating all heads ofers a slight improvement, it still underperforms motion-specific heads. These results suggest that incorporating spatial heads can dilute relevant motion cues; in contrast, isolating motion-specific heads better captures dynamic correlations essential for motion transfer. Qualitative comparisons are included in Supp. Sec. C.5.

To demonstrate the broader applicability of motion-specific heads as a motioncontrol mechanism, we conduct additional experiments focusing on motion dynamics. Building on Layer Perturbation Guidance [31], we extend this approach to our motion-specific head by introducing head perturbation during inference. Evaluation using VBench metrics [19] on 100 video samples reveals that integrating this method with CogVideoX [47] improves image quality by +14%, aesthetics by +8%, and motion dynamics by +15%. These results suggest that our motion-head analysis provides a lightweight yet efective mechanism to guide motion in DiT-based architectures.

Validation of Structure-Specialized Heads. Table 6b presents results under diferent entropy thresholds for structure-specialized head selection. Based on our analysis showing a median entropy of 7, we adopt τ =7 as the threshold. Injecting features from high-entropy heads (τ >7) yields lower MF and FTD scores, indicating that such heads contain weaker, more difuse structural cues. This validates the relationship between attention-map entropy and structural information in the context of motion transfer. Additional experiments over a wider entropy range and qualitative examples are provided in Supp. Sec. C.1.

![](images/be4c15fbf02d772b1dfa91c7303d2fa61ee82d84e98ac03c5009d716eba98cfb.jpg)  
Fig. 11: Qualitative results on complex motion and video editing. (a) Generated samples illustrating the HALO’s capability in handling complex motion dynamics. (b) Application to video editing using the structure-specialized head.

To further assess the applicability of structure-specialized heads, we extend their utility to video editing, a task where preserving structural integrity is crucial. Specifically, we perform the attention value injection within these selected structure-specialized heads. As shown in Fig. 11(b), our approach successfully modifies subject identity (e.g., rhino→lion/bear) while maintaining the original layout and structural details. These results demonstrate that structurespecialized heads serve as a precise mechanism for structure-preserving video editing, extending their efectiveness beyond motion transfer.

## 6 Conclusion

In this work, we demonstrate for the first time that video DiTs contain distinct motion-specific and structure-specialized heads responsible for temporal dynamics and spatial organization. Based on this insight, we present HALO, a training-free motion transfer framework that leverages these heads for high motion fidelity and structural alignment. Our method integrates semantic correspondences with motion representations and proposes a head-wise injection to enable motion-consistent, structure-aligned transfer. Extensive experiments validate that HALO outperforms prior approaches, highlighting the efectiveness of head-level analysis. Furthermore, by introducing our Movie Scene Dataset, we provide a new direction for future research in practical motion transfer.

Future Work. Our findings reveal that video DiTs inherently disentangle motion and structure across attention heads, opening new directions for controllable video generation. Future research may explore explicit control over motion direction and intensity for fine-grained editing.

## Acknowledgments

This work was supported in part by the IITP RS-2024-00457882 (AI Research Hub Project), IITP 2020-II201361, NRF RS-2024-00345806, and NRF RS-2023- 002620, while the authors are afiliated with the Department of Artificial Intelligence (S.J., J.P, S.J.H.) and the Department of Computer Science (K.C.). This work was supported by the AI Seoul Tech Research Support Program of the Seoul Future Foundation. We would like to express our sincere gratitude to RASCA FX for providing access to the movie dataset used in this work. We also thank the RASCA FX team for their valuable support and collaboration throughout this project.

# Controlling Motion Transfer in Difusion Transformers via Attention Heads Supplementary Material

Sunyoung Jung<sup>1∗</sup> Jiwoo Park<sup>1,2∗</sup> Yoonseok Choi<sup>1</sup> Kyobin Choo<sup>1</sup> Ming-Hsuan Yang<sup>3</sup> Seong Jae Hwang<sup>1</sup> <sup>1</sup>Yonsei University <sup>2</sup>LG Electronics <sup>3</sup>University of California, Merced

## Contents

A Code & Presentation ..... 16
B Details of Problems in Previous Methods ..... 16
C Details of HALO ..... 18
D Application to the Video DiT Model: Wan ..... 23
E Movie Scene Dataset ..... 26
F Additional Experiment Results ..... 27
G Discussion ..... 33

## A Code & Presentation

The code for HALO will be released to the public. In addition, we have prepared a presentation video that introduces HALO in a clear and accessible manner, illustrating qualitative results and representative video demonstrations. This video is included as the file presentation.mp4.

## B Details of Problems in Previous Methods

This section discusses the limitations of existing DiT-based motion transfer approaches [7, 13, 14], which largely arise from the lack of understanding of the functional roles of attention heads. We further examine challenges observed in cross-frame attention-based methods [35]. Although displacement maps can capture motion dynamics, they do not encode semantic correspondence, often resulting in motion being transferred to irrelevant regions and causing inconsistent or even reversed object motion.

Results of DiT-based Methods. We analyze the outputs of prior DiT-based motion transfer approaches and identify several characteristic failure patterns. As illustrated in Fig.12(a), existing methods often produce misaligned or distorted object structures: although the peacock roughly follows the reference motion trajectory, it sufers from severe morphological inconsistency. Furthermore, Fig.12(b) reveals identity leakage from the reference video. In motion transfer, it is essential to preserve the semantics specified by the target prompt while aligning with the reference motion. However, previous methods inherit appearance cues from the reference, indicating insuficient disentanglement between content and motion representations. In contrast, HALO achieves accurate motion transfer, faithfully following the reference motion and adhering to the intended target semantics.

![](images/320cf412578cf7fad6605b62328c2362b932c741765d7c93e98cb4c6fd610d8f.jpg)

![](images/eaa91749b934d31b51f02a33ee26ee628c19bd7cb009194316ae9a32c09eacac.jpg)  
A motorcycle driving on a narrow alley

![](images/d632330dfbbf1cc8b6c7139cbd63aae8acb8316f5f83d99d03c2a8c1c26cb038.jpg)  
Fig. 12: Limitations of existing DiT-based motion transfer methods. (a) The generated subject exhibits structural distortion and misalignment, failing to preserve the spatial layout of the reference. (b) Identity cues from the reference video leak into the generated subject, causing identity corruption and appearance inconsistency across frames.  
Fig. 13: Problems of cross-frame attention-based motion transfer methods. (a) The generated video reproduces the reference camera motion, but the object fails to follow the reference object’s trajectory, leading to degraded motion fidelity. (b) The generated video omits the moving object, and its motion is incorrectly mapped onto the background.

Results of Cross-Frame Attention-based methods. Cross-frame attentionbased motion transfer approaches [35] sufer from a fundamental limitation: because they derive correspondence solely from query-key similarity, the resulting displacement maps tend to reflect coarse attention activation patterns rather than semantic alignment. As shown in Fig.13(a), even when the global camera motion correctly mirrors the reference (e.g., a right-to-left sweep), the generated object often follows an incoherent spatial trajectory, causing the layout to diverge from the reference. This discrepancy arises when spurious attention activations dominate displacement estimation in semantically irrelevant regions. Fig.13(b) illustrates an additional failure mode: the foreground object disappears while its motion is incorrectly transferred to the background. In this case, the intended motion of the foreground subject is improperly transferred to the background, leading to degraded motion alignment.

These shortcomings highlight the necessity of integrating semantic correspondence refinement within the motion transfer pipeline. Therefore, HALO achieves precise and coherent synthesis by grounding motion guidance in semantic alignment rather than coarse feature similarity. To the best of our knowledge, HALO

Table 7: Optimization steps $T _ { \mathrm { o p t } }$ results.

<table><tr><td> $T_{\text{opt}}$ </td><td>CLIP ↑</td><td>MF ↑</td><td>FTD ↓</td></tr><tr><td>10</td><td>30.9</td><td>50.1</td><td>26.2</td></tr><tr><td>12</td><td>31.7</td><td>66.2</td><td>19.4</td></tr><tr><td>15</td><td>30.5</td><td>56.0</td><td>22.7</td></tr></table>

Table 8: Bias strength β analysis results.

<table><tr><td> $\beta$ </td><td>CLIP  $\uparrow$ </td><td>MF  $\uparrow$ </td><td>FTD  $\downarrow$ </td></tr><tr><td>0.1</td><td>31.7</td><td>66.2</td><td>19.4</td></tr><tr><td>0.3</td><td>30.7</td><td>53.6</td><td>23.4</td></tr><tr><td>0.5</td><td>30.7</td><td>53.9</td><td>23.7</td></tr></table>

Table 9: Hyperparameter sensitivity.

<table><tr><td>4 5 6 7 8 9</td><td>0.01 0.05 0.1</td><td>3 4 5</td></tr><tr><td>MF 67.0 67.2 67.1 72.3 70.2 69.1</td><td>MF 66.6 68.1 69.8</td><td>MF 75.4 76.1 73.1</td></tr><tr><td>(a) τ sensitivity</td><td>(b) β sensitivity</td><td>(c) top-k sensitivity</td></tr></table>

represents the first attempt to explicitly bridge the gap between high-level semantic correspondence and low-level attention-based motion representations.

## C Details of HALO

We provide the hyperparameter configurations in Sec. C.1. Our extensive experiments demonstrate that HALO is highly robust to hyperparameter variations, maintaining consistency across diverse videos and model architectures without the need for specific tuning. The full algorithmic procedure for HALO is provided in Sec. C.2.

## C.1 Hyperparameters

Optimization iteration. We set the optimization steps to $T _ { \mathrm { o p t } } { = } 1 2$ . Increasing $T _ { \mathrm { o p t } }$ improves quantitative performance, especially MF, but yields diminishing returns after 12 iterations (see Table 7). We therefore choose $T _ { \mathrm { o p t } } { = } 1 2$ to balance accuracy and eficiency.

Entropy range. Beyond the broad range discussed in the main text, Table 9(a) provides a fine-grained entropy range τ analysis on the curated dataset. MF degrades when the value deviates from 7, indicating that an entropy threshold of 7 optimally distinguishes structural information.

Bias strength. Semantic bias strength $\beta$ scales the correspondence-guided bias on displacement maps in Semantic Reweighting (SRW). Table 8 shows that a small bias strength yields the best overall performance, whereas larger values lead to degraded results. These results indicate that an excessive semantic bias can suppress motion dynamics in the displacement maps, thereby compromising the motion fidelity. Based on this observation, we set β=0.1 in all experiments. A finer-grained sweep on the curated dataset further shows stable performance within the low-bias range, with β=0.1 achieving the best result, as shown in Table 9(b).

Top-k Selection. To construct the reference displacement map,

![](images/06bc885082202d3d6193ff05b4e6d1a6e30897b3fef20216f992bf74cee51587.jpg)  
Fig. 14: Qualitative results with corresponding reference displacement maps $\mathcal { D } _ { \mathrm { r e f } }$ hyperparameter Top-k analysis.  
Table 10: Hyperparameter Top-k analysis.

we extract the top-4 candidates from cross-frame attention and then select the final match by comparing their distances to the semantically bestaligned location. To justify the choice

<table><tr><td>Top-k</td><td>CLIP ↑</td><td>MF ↑</td><td>FTD ↓</td></tr><tr><td>4</td><td>31.7</td><td>66.2</td><td>19.4</td></tr><tr><td>10</td><td>31.2</td><td>45.2</td><td>26.5</td></tr><tr><td>20</td><td>30.8</td><td>39.6</td><td>30.3</td></tr></table>

of k=4, we additionally evaluate larger candidates (k=10, k=20).

As shown in Table 10, increasing k consistently degrades all metrics, indicating that using the top 4 candidates yields the most stable and reliable displacement estimation. This performance drop arises from inaccurate displacement predictions caused by irrelevant cross-frame attention candidates. As illustrated in Fig.14, while HALO with k=4 produces clean, object-aligned displacement maps, larger k settings (k=10, 20) generate noisy and spatially inconsistent maps. These inaccuracies propagate directly into motion transfer, ultimately leading to weaker motion alignment. Additionally, evaluating the performance sensitivity within a narrower top-k range (3, 4, and 5) yields consistent results (Table 9(c)), demonstrating low variance and robustness to this specific setting.

## C.2 Algorithm

Algorithm 1 outlines the inference pipeline of HALO, encompassing both the motion-optimization and the structural feature-injection steps. Notably, this entire procedure is training-free, requiring no additional parameters or model finetuning.

## C.3 Metric Details for Motion-Specific Heads

To verify that our head-selection strategy isolates motion-specific behavior, we compare displacement fields from temporal, spatial, and all-head configurations against ground-truth optical flow (RAFT [39]) using Directional Alignment (DA) and Correlation (Corr). Both metrics operate on displacement maps of shape (T, H, W, 2), where T is the number of frame pairs used for displacement estimation, (H, W ) is the latent spatial resolution, and the final dimension corresponds to the (x, y) flow components.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 HALO inference process
Input: Reference video $x_{\text{ref}}$, text prompt $\mathcal{P}$,
DiT model $\epsilon_{\theta}$, noise scheduler $\mathcal{N}$, encoder $\mathcal{E}$ decoder $\mathcal{G}$
Output: Generated video $x_0$
Note: Semantic extractor DIFT, Attention head h
$\mathcal{T}, \mathcal{S}$: Motion Heads, Structure Heads
1: Extract correspondence map: $\mathcal{C} \leftarrow DIFT$ ▷ cosine similarity
2: Compute attention: $\{Q, K\} \leftarrow \epsilon_{\theta}(z_{\text{ref}}, \emptyset, 0)$ ▷ $z_{\text{ref}} = \mathcal{E}(x_{\text{ref}})$
3: Head Classification for $h \in \mathcal{T}$ ▷ Temporal or Spatial head
4: for each $(i, j)$ where $i, j \in [1, F]$ do
5: if $h \in \mathcal{T}$ then
6: Calculate cross-frame attention $A_{i,j}$
7: Construct displacement matrix $\mathcal{D}_{i,j}$
8: end if
9: end for
10: Construct reference displacement: $\mathcal{D}_{\text{ref}} \leftarrow \text{SCR}(\mathcal{D}_{i,j}, \mathcal{C}_{i,j})$
11: Initialize $z_T \sim \mathcal{N}(0, I)$
12: for $t = T$ to 0 do
13: if $t &gt; T_{\text{opt}}$ then
14: for optimization step $k = 0$ to $N_{\text{opt}}$ do
15: Head Classification for $h \in \mathcal{T}$
16: Entropy Calculation for $h \in \mathcal{S}$
17: $z_t' \leftarrow \mathcal{N}(z_{\text{ref}}, t)$ ▷ Noise Forward
18: if $h \in \mathcal{S}$ then ▷ Structure Heads
19: Store Value: $V' \leftarrow \epsilon_{\theta}(z_t', \mathcal{P}, t)$
20: Inject Value: $V' \leftarrow V$
21: end if
22: if $h \in \mathcal{T}$ then ▷ Motion Heads
23: for each $(i, j)$ where $i, j \in [1, F]$ do
24: Calculate cross-frame attention $A_{i,j}$
25: Reweight $\tilde{A}_{i,j}: \tilde{A}_{i,j} \leftarrow \text{SRW}(A_{i,j}, \mathcal{C}_{i,j})$
26: Construct displacement matrix $\mathcal{D}_{i,j}$
27: end for
28: $\mathcal{L}_{SM} \leftarrow \| \mathcal{D}_{\text{ref}} - \mathcal{D}_t \|_2^2$
29: Update $z_t$ by minimizing $\mathcal{L}_{SM}$
30: end if
31: end for
32: end if
33: $z_{t-1} = f(z_t, \epsilon_{\theta}(z_t, \mathcal{P}, t))$
34: end for
35: Return $x_0 = \mathcal{G}(z_0)$
</div>

Directional Alignment (DA). DA evaluates how well the predicted displacement directions align with the reference optical flow, independent of magnitude. Given the reference flow $f _ { \mathrm { r e f } }$ and generated displacement $f _ { \mathrm { g e n } } , \mathrm { D A }$ is defined as:

![](images/d8246b3021fe9101705b2224d435adce91107f3e524fb09027b8057e818d7d45.jpg)

![](images/caf278c9b7e8a25f4cc4da9f3492a072b53c7c76117e0974be2e149d968806c3.jpg)

![](images/1e85b5ad421b0ed721584011a1846b645a7a793074c8daf58abe9b6e139706ce.jpg)  
Fig. 15: (a) Attention Entropy Distribution: Distribution of attention-map entropy values across samples. (b) Layer-wise Head Entropy Consistency: Distribution of headlevel entropy values across layers, showing consistent patterns across diferent samples. Similarity Metric (SSIM): A heatmap is used to visualize pairwise similarity between samples.

$$
\mathrm{DA} = \mathrm{mean} \Bigl \langle \frac {f _ {\mathrm{ref}}}{\| f _ {\mathrm{ref}} \|}, \frac {f _ {\mathrm{gen}}}{\| f _ {\mathrm{gen}} \|} \Bigr \rangle ,\tag{1}
$$

where ∥ · ∥ denotes the L2 norm and $\langle \cdot , \cdot \rangle$ is the dot product. The computation proceeds as follows: 1) Normalize $f _ { \mathrm { r e f } }$ and $f _ { \mathrm { g e n } }$ at each pixel. 2) Compute perpixel cosine similarity between normalized vectors. 3) Average the directional similarity across all spatial locations and time steps.

Correlation (Corr). Corr measures the linear relationship between the predicted and reference flows, capturing both directional and magnitude consistency. It is computed using the Pearson correlation coeficient:

$$
\mathrm{Corr} = \frac {\mathrm{Cov} (f _ {\mathrm{ref}} , , f _ {\mathrm{gen}})}{\sigma_ {\mathrm{ref}} , \sigma_ {\mathrm{gen}}},\tag{2}
$$

where Cov denotes covariance and σ represents the standard deviation. To compute Corr, we flatten each displacement field into a one-dimensional array of length $T \times H \times W \times 2$ and apply the standard Pearson correlation formula.

Whereas DA focuses solely on directional agreement, Corr reflects overall linear consistency, enabling complementary assessment of both direction- and magnitude-level motion alignment. This multi-dimensional analysis serves as a robust proof of concept for Motion-Specific head, proving that HALO achieves superior motion-aware guidance through a well-validated selection process.

## C.4 Entropy Validation in Structure-Specialized Heads

Entropy Consistency Across Samples. We investigate the consistency of head entropy distributions across diverse video inputs. Fig. 15(a) shows the entropy patterns remain consistent across diferent samples, particularly within layers 14 to 27, which are utilized for structural feature injection. To quantify

Entropy Distribution

![](images/5888736ebbed9f5ce0e15f4568cd230921c448874d694f8c1b8431ccd17f252d.jpg)  
Fig. 16: Entropy distribution across all layers, computed by averaging the entropy of all attention heads within each layer. The orange dashed line denotes the median entropy value across layers, which is 6.78.

Reference  
HALO  
(a)  
(b)  
![](images/ba2f9043e4b61e8ac785b909fa2601181f36f18d99569e289de174b86803caab.jpg)  
A monkey is riding a bicycle on a trail next to a mural-covered wall in an urban park.

Fig. 17: Qualitative results under diferent entropy thresholds. (a) Injecting features only from heads with entropy greater than 7. (b) Injecting features from all heads.

this consistency, we compute similarity metrics using heat maps that visualize entropy values across the entire benchmark (Fig. 15(b)). The resulting SSIM score of 0.988 confirms a highly stable and invariant entropy pattern regardless of the input. This high degree of similarity indicates that the structural information encoded by the low-entropy heads is inherently consistent.

Entropy Threshold. For structural-head selection, we set the entropy threshold to $\tau < 7 .$ This choice is motivated by our empirical observation that the median entropy of attention maps across layers is approximately 7, as shown in Fig.16. Intuitively, lower-entropy heads exhibit more concentrated attention distributions, which aligns with spatially localized, structure-preserving behavior. To derive this distribution, we compute the entropy of attention maps across all 90 samples, 42 layers, and 50 denoising steps. Fig.16 reports the mean entropy aggregated over all samples, layers, and steps, yielding a median value of roughly 7. This provides a principled basis for setting the threshold at $\tau < 7$

![](images/3ff23eefb722682880334e5b0b0b583dea83de252a8be3756afc8d01f14d2e02.jpg)  
Fig. 18: Qualitative results and reference displacement maps $\mathcal { D } _ { \mathrm { r e f } }$ across various $\mathrm { a t - }$ tention head configurations.

We evaluate various entropy ranges to confirm that our selected entropy range, τ , consistently identifies structural-specialized heads. Fig. 17(a), highentropy heads $( \tau > 7 )$ produce difuse attention patterns, resulting in layouts misaligned with the reference. Meanwhile, injecting all heads (b) causes visual artifacts due to the accumulation of disparate spatial features. These results highlight that selective structural-head injection is essential for providing clean structural guidance, allowing HALO to maintain a coherent spatial layout without artifacts.

## C.5 Validation of Motion-Specific Heads

Fig. 18 illustrates how the head analysis translates to the motion transfer results. Motion-Specific Head demonstrates results that faithfully track the motion of the reference video while efectively capturing the movement of the car in the reference displacement map. In contrast, All-Head and Spatial Head fail to properly capture the car in the displacement map, leading to misaligned motion in the generated videos. Consequently, the results validate that Motion-Specific Head plays a crucial role in efectively processing motion information.

## D Application to the Video DiT Model: Wan

To demonstrate the generalizability of HALO, we extend it to Wan [40] video DiT model. Notably, we utilize the identical hyperparameter configurations as those used for CogVideoX [47], achieving consistent performance without any model-specific tuning. Unlike CogVideoX [47], Wan employs a dedicated selfattention module that exclusively processes visual tokens. Accordingly, our analysis is conducted on this specialized attention mechanism.

Head Analysis in Wan. Using the same procedure as in the main model, we identify both Motion-Specific and Structure-Specialized heads within Wan. To analyze motion behavior, we compute displacement maps separately for the temporal and spatial heads. As shown in Fig. 19, temporal heads more accurately capture the reference motion than spatial heads, exhibiting clear distinctions between background motion (red) and object motion (blue). We further compute Correlation (Corr) and Directional Alignment (DA) between RAFT-based optical flow [39] and the displacement maps of each head. Fig. 20 shows that temporal heads consistently achieve higher Corr and DA scores than spatial heads, confirming that temporal heads in Wan correspond to motion-specific heads.

![](images/d1879793b7803114ba494d2def74abecac94b99b7ebb4a0e7e8f2ce79a1d5da4.jpg)  
Fig. 19: Displacement-field comparison between temporal and spatial heads in Wan. Temporal heads better capture reference motion, showing clearer separation of background and object motion.

![](images/b2894fd49abbae3bdf6d169799d4609568fe76a2e9eefae28bb01a8f2102980c.jpg)

![](images/8033247a8e2c6d95c9a3293f56735ad4b84a29a27dd33b4bf15808badb17c3ef.jpg)

![](images/a3a1f7affbf0dcb0e58fcc2ec032e3f88eb3bea27b2d9cf72c1d6791cfefbd9d.jpg)  
Fig. 20: Quantitative comparison of each head in Wan using Directional Alignment (DA) and Correlation (Corr) with groundtruth optical flow.  
Fig. 21: Relationship between attentionmap entropy and spatial entropy in Wan, demonstrating that low-entropy heads correspond to structure-specialized heads.

We also compute attention-map entropy [32] and compare it with the spatial entropy [5, 34] derived from corresponding attention features. As shown in Fig. 21, attention-map entropy is strongly correlated with spatial entropy, consistent with the findings in the main paper. This confirms that low-entropy heads in Wan encode structural information. Together, these results demonstrate that our analysis methodology generalizes efectively across diferent video DiT architectures.

Experimental Results. We conduct experiments on the same dataset [36]

The person is riding a motorbike on a muddy course

![](images/46ed0454300d401937f689f86e2e39ef4d8a0c2123a0d3f1cc38e80af3ed78f4.jpg)  
Fig. 22: Motion transfer results of HALO applied to Wan.

![](images/be2a4ced95b022556ee1c81516968e3ea14750af196801f39633195cf3b86807.jpg)  
Fig. 23: Overview of our movie scene dataset: video samples and corresponding prompts.

used for the primary evaluations in the main paper. For faster experimentation, we generate 21-frame videos using 25-frame reference sequences. Table 11 shows that when applied to

Table 11: Quantitative results of Wan.

<table><tr><td>Model</td><td colspan="4">CLIP ↑ TC ↑ MF ↑ FTD ↓</td></tr><tr><td>WAN + HALO</td><td>30.2</td><td>90.1</td><td>71.6</td><td>18.3</td></tr></table>

Wan, HALO produces videos that accurately reflect both the target prompt and the reference motion, achieving comparable motion fidelity to the results demonstrated in the main model. Qualitative motion transfer examples are shown in Fig. 22.

Cross-Architecture Robustness. While the main paper primarily reports results using CogVideoX [47], HALO is not tied to this specific architecture. To examine whether our entropy-based head selection depends on a particular backbone, we compare attention-entropy statistics between CogVideoX [47] and Wan [40]. We observe highly aligned timestep-wise entropy trends (r=0.97) and strong agreement in mean and median entropy (0.92/0.96), indicating that low-entropy structural heads emerge consistently across both video DiTs. This supports entropy-based head selection as a backbone-robust criterion rather than architecture-specific tuning.

## E Movie Scene Dataset

This section discusses practical applications of motion transfer in Sec. E.1, provides the details of the movie-scene dataset in Sec. E.2, and presents additional results of both our method and baseline models in Sec. E.3.

## E.1 Application Potential of Motion Transfer

Existing visual efects (VFX) and computer graphics (CG) productions involve time-consuming and labor-intensive pipelines, requiring specialized skills across stages, such as modeling, animation, and rendering [11]. With recent advances in video generation accelerating content creation [26, 30], motion transfer has emerged as a practical, controllable video generation with direct utility in filmmaking and game production. The practical demand for such flexible content creation necessitates specialized datasets tailored to these domains. To address this need, we introduce a movie scene dataset that serves as a foundational resource for advancing and benchmarking motion transfer in professional production contexts.

## E.2 Movie Scene Dataset Details

To evaluate our method in more realistic scenarios, we construct a movie scene dataset reflecting real-world video production environments, as shown in Fig. 23. While existing benchmarks [27,36] already utilize real videos, we incorporate this dataset to further assess the generalizability and robustness of our approach in production-level settings. The dataset consists of 20 videos, each paired with five diferent text prompts, totaling 100 samples, and the prompts describe both the primary objects and their backgrounds. We generate prompts using GPT-4o [20] to cover diverse cinematic scenarios. Moreover, the dataset involves complex motion patterns that go beyond simple movements, such as sword fighting, horseback riding, and walking in crowds. Such variety provides a robust testbed for verifying the model’s ability to faithfully transfer motion in dynamic real-world environments.

## E.3 Movie Scene Qualitative Results

We present more qualitative comparisons on the movie scene dataset in Fig. 24, evaluating multiple motion transfer methods. HALO achieves the most consistent motion transfer, exhibiting high motion fidelity and coherent structural alignment. ConMO [13] is excluded from this comparison because it requires an input mask for generation, which is not available for our movie scene dataset.

![](images/39045fee1bf2d37d4abe2d68606fadf40c8d2f6446fa9943bda093eba00b8396.jpg)  
Toy Story’s Woody gallops forward on horseback across a warm sunlit terrain with natural dust.

Fig. 24: Qualitative comparison across diverse motion transfer tasks with the movie scene dataset. HALO is evaluated against U-Net- and DiT-based baselines across multiple subjects and motions.

## F Additional Experiment Results

## F.1 More Qualitative Results

We provide additional qualitative results in Fig. 25 across both the motion transfer benchmark (top) and our movie-scene dataset (bottom). Extended qualitative results in Fig. 26 further show the robustness of HALO across varying motion scales, from routine activities (e.g., walking, bus motion) to challenging high-dynamic scenes (e.g., horse jumping). The results highlight our method’s superior ability to maintain motion fidelity even as the complexity of the scene increases.

![](images/d12452e5c9c30cab5b67bb1e41de9e05373b581dd8865f55f71026503392d7be.jpg)  
Fig. 25: Qualitative results of HALO on benchmark and movie-scene datasets. The top rows illustrate performance on the motion transfer benchmark, while the bottom rows show generalization to movie scene videos.

Reference

HALO

DiTFlow

RoPECraft  
MotionClone  
![](images/b935eabc27d0250692ab109606717c2dd7a776d5cf62210142e91fefa1ef13c2.jpg)  
Fig. 26: Qualitative comparison across diverse motion transfer tasks. HALO is evaluated against U-Net- and DiT-based baselines across multiple subjects and motions.

## F.2 Quantitative Trade-of and metric Analysis

CLIP vs. MF trade-of Given that motion transfer necessitates a delicate balance between preserving target semantics and following reference motion, CLIP and MF serve as complementary metrics that should be interpreted jointly. While baseline methods exhibit a pronounced CLIP–MF trade-of–consistent with prior studies [36,48]–HALO achieves a superior balance, attaining high scores in both metrics simultaneously (Fig. 27). These results suggest that our head-analysis approach efectively transfers motion while reducing irrelevant semantic interference.

TC vs. Motion Dynamics analysis We further analyze temporal consistency by measuring the average optical-flow magnitude across generated videos to quantify motion dynamics. As shown in Table 12, methods with higher automated TC scores can exhibit lower motion dynamics, suggesting that TC may favor relatively static outputs. Therefore, TC should be interpreted jointly with reference-motion alignment metrics such as MF. Although HALO shows a slightly lower TC score than DiTFlow [35] and GWTF [7], it achieves substantially higher MF in the main quantitative results. Moreover, the user study ranks HALO highest in perceived temporal consistency, indicating that the lower automated TC score does not necessarily imply weaker temporal coherence.

![](images/284f029e1382bb1c81ea4a72aace4d4b75095ecb4d9cb53896851991ff539c6d.jpg)  
Fig. 27: Objective comparison across motion-transfer methods in terms of Motion Fidelity (MF) and CLIP Score (CLIP). Baseline methods exhibit a trade-of between MF and CLIP, whereas HALO (Ours) achieves balanced performance.

Table 12: Temporal consistency and motion dynamics analysis.

<table><tr><td></td><td>HALO</td><td>GWTF</td><td>DiTFlow</td></tr><tr><td>Motion Dynamics ↑</td><td>5.79</td><td>4.81</td><td>5.19</td></tr><tr><td>Temporal Consistency ↑</td><td>87.5</td><td>88.2</td><td>89.5</td></tr></table>

## F.3 Computational cost

Table 13: Runtime and Memory Report.  
Table 14: Comparison with Baselines

<table><tr><td></td><td>Base</td><td>+Sem.</td><td>+Inj.</td><td>HALO</td></tr><tr><td>Time(s)</td><td>1076</td><td>1088</td><td>1115</td><td>1125</td></tr><tr><td>Memory(GB)</td><td>38.1</td><td>38.2</td><td>38.2</td><td>38.2</td></tr></table>

<table><tr><td></td><td>DiTFlow</td><td>RoPECraft</td><td>GWTF</td><td>HALO</td></tr><tr><td>Time(s)</td><td>1076</td><td>8187</td><td>456+α</td><td>1125</td></tr><tr><td>Memory(GB)</td><td>38.1</td><td>35.7</td><td>14.8+α</td><td>38.2</td></tr></table>

We report the runtime and memory eficiency of HALO on a single NVIDIA RTX A6000. As shown in Table 13, incorporating semantic correspondence and head-wise injection incurs only a minor runtime overhead; compared to the base model, HALO adds just 49s in inference time and 0.1GB in peak memory. We further compare HALO’s computational cost against other DiT-based methods in Table 14. A direct comparison with GWTF is challenging because its training requirements introduce significant additional overhead α beyond simple inference. HALO demonstrates computational parity with leading training-free methods. Specifically, our approach remains highly competitive with DiTFlow and RopeCraft, achieving faithful motion transfer without sacrificing practical inference scalability.

![](images/d6786c37437b6c8c67b2dbfb5d92b9793c0e44d4788db68bd8a103357917c15c.jpg)  
Fig. 28: Failure cases of our method. In these examples, the output closely follows the reference motion but exhibits unnatural articulated movements (e.g., a penguin walking like a bear), which occurs when semantic correspondence over-aligns structurally incompatible articulations.

## F.4 Limitations

In some cases, the generated motion closely follows the reference yet becomes unnatural (e.g., a penguin walking like a bear), as illustrated in Fig. 28. This failure arises when semantic correspondence over-aligns structurally incompatible articulations. Because our semantic correspondence module refines motion representations via fine-grained correspondences, it can occasionally enforce near one-to-one part alignment, leading to implausible kinematics. Interestingly, these cases can exhibit higher motion fidelity scores because they more strictly track the reference dynamics. This reveals an inherent trade-of between referencemotion following and kinematic plausibility, which is not fully captured by existing evaluations.

Fine-Grained Local Motion We further analyze HALO under fine-grained local motion, such as facial expressions. As shown in Fig. 29, HALO preserves the overall head pose and coarse facial motion of the reference, showing comparable motion alignment to existing methods. However, subtle non-rigid deformations, such as detailed mouth shapes and fine facial expressions, remain challenging. This limitation arises because HALO represents motion using patch-level displacements, which are efective for object-level and medium-scale motion but less precise for very local deformation.

## F.5 Evaluation of Structure-Specialized Heads for Video Editing

As shown in the main paper, we further extend our method to video editing. For quantitative evaluation, we report CLIP score (CLIP), Temporal Consistency (TC), Motion Fidelity (MF), masked PSNR (m.P), and masked LPIPS (m.L) [43]. To compute m.P and m.L, we use the provided segmentation masks and evaluate reconstruction quality on the background region, which is expected to remain unchanged from the reference. As shown in Table 15, HALO achieves strong editing quality while preserving background structure, supporting the efectiveness of structure-specialized heads beyond motion transfer. Additional qualitative results are provided in Fig. 30.

![](images/3198d4ec4227ac1d946ca141fb4e39c778e37d588148ca43cad5d50da4d6160b.jpg)  
Fig. 29: Fine-grained local motion transfer. HALO preserves coarse facial motion and head pose, but subtle non-rigid deformations, such as detailed facial expressions, remain challenging due to the patch-level displacement representation.

![](images/09c5d24374ab4ff26074812165c071f91c2e148ef4263a36580895cebd2354f0.jpg)  
Fig. 30: Application to video editing using the structure-specialized heads.

Table 15: Video editing results.

<table><tr><td></td><td>CLIP ↑</td><td>TC ↑</td><td>MF ↑</td><td>m.P ↑</td><td>m.L ↓</td></tr><tr><td>HALO</td><td>29.3</td><td>93.1</td><td>81.9</td><td>17.2</td><td>0.4</td></tr></table>

## F.6 Structural Consistency in Long Videos

To validate structural consistency in long videos, we evaluate HALO on 112- frame sequences. This setting is substantially longer than the standard generation length used in the main experiments, allowing us to examine whether the selected structure-specialized heads remain stable over extended temporal horizons. As shown in Fig. 31, low-entropy heads preserve consistent structural attention patterns across the sequence, without noticeable drift or degradation over time. This indicates that entropy-based head selection is not limited to short clips but provides stable structural cues for long-horizon motion transfer.

![](images/82e30db852ff17248f776fae45c5eda99688335823c3adee8fc15d813fdb9280.jpg)  
Fig. 31: Structural consistency in long-video generation. We evaluate HALO on 112- frame sequences and visualize the behavior of selected low-entropy heads. The selected heads maintain stable structural attention patterns over extended temporal horizons, supporting long-sequence structural preservation.

## F.7 Scalability of SCR

SCR operates entirely in the latent space, making its computational overhead moderate even at higher output resolutions. As shown in Table 16, increasing the resolution from 720p to 1080p increases the SCR runtime only from 13.2s to 19.1s, while the total inference time increases from 1125s to 1404s. The memory usage of SCR also remains nearly unchanged across the two resolutions. These results indicate that SCR does not introduce a major bottleneck and remains practical for high-resolution motion transfer.

Table 16: Scalability analysis of SCR at higher resolutions.

<table><tr><td></td><td>HALO 720p</td><td>SCR</td><td>HALO 1080p</td><td>SCR</td></tr><tr><td>Time (s)</td><td>1125</td><td>13.2</td><td>1404</td><td>19.1</td></tr><tr><td>Mem. (GB)</td><td>38.2</td><td>18.2</td><td>38.2</td><td>18.4</td></tr></table>

## G User Study

For human evaluation, we conduct a comprehensive user study involving 20 participants with expertise in computer vision. The evaluation is performed on a set of 10 representative video samples. Participants are asked to assess the results based on three key criteria: editing accuracy, temporal consistency, and motion fidelity. The specific evaluation metrics are defined as follows:

– Edit Accuracy: How well does the video content align with the provided text prompt? (e.g., whether the characters, actions, and attributes described in the prompt are accurately reflected in the visual output).

– Temporal Consistency: Do the motion, appearance, and background elements transition naturally throughout the video? (e.g., checking for the absence of flickers or abrupt changes in character identity or background details).

– Motion Fidelity: How closely does the overall motion in the generated video resemble the reference motion? (e.g., whether the pose, flow of action, speed, and timing align with the source motion).

## References

1. Ahn, D., Kang, J., Lee, S., Kim, M., Min, J., Jang, W., Lee, S., Paul, S., Hong, S., Kim, S.: Fine-grained perturbation guidance via attention head selection. arXiv e-prints pp. arXiv–2506 (2025)

2. Atzmon, Y., Gal, R., Tewel, Y., Kasten, Y., Chechik, G.: Motion by queries: Identity-motion trade-ofs in text-to-video generation. arXiv preprint arXiv:2412.07750 (2024)

3. Avrahami, O., Patashnik, O., Fried, O., Nemchinov, E., Aberman, K., Lischinski, D., Cohen-Or, D.: Stable flow: Vital layers for training-free image editing. In: CVPR. pp. 7877–7888 (2025)

4. Bar-Tal, O., Chefer, H., Tov, O., Herrmann, C., Paiss, R., Zada, S., Ephrat, A., Hur, J., Liu, G., Raj, A., et al.: Lumiere: A space-time difusion model for video generation. In: SIGGRAPH Asia 2024 Conference Papers. pp. 1–11 (2024)

5. Batty, M.: Spatial entropy. Geographical analysis 6(1), 1–31 (1974)

6. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al.: Stable video difusion: Scaling latent video difusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023)

7. Burgert, R., Xu, Y., Xian, W., Pilarski, O., Clausen, P., He, M., Ma, L., Deng, Y., Li, L., Mousavi, M., et al.: Go-with-the-flow: Motion-controllable video difusion models using real-time warped noise. In: CVPR. pp. 13–23 (2025)

8. Cai, M., Cun, X., Li, X., Liu, W., Zhang, Z., Zhang, Y., Shan, Y., Yue, X.: Ditctrl: Exploring attention control in multi-modal difusion transformer for tuning-free multi-prompt longer video generation. In: CVPR. pp. 7763–7772 (2025)

9. Chen, H., Zhang, Y., Cun, X., Xia, M., Wang, X., Weng, C., Shan, Y.: Videocrafter2: Overcoming data limitations for high-quality video difusion models. In: CVPR. pp. 7310–7320 (2024)

10. Chen, S., Xu, M., Ren, J., Cong, Y., He, S., Xie, Y., Sinha, A., Luo, P., Xiang, T., Perez-Rua, J.M.: Gentron: Difusion transformers for image and video generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6441–6451 (2024)

11. Du, T., Wu, K., Ma, P., Wah, S., Spielberg, A., Rus, D., Matusik, W.: Difpd: Diferentiable projective dynamics. ACM Transactions on Graphics (ToG) 41(2), 1–21 (2021)

12. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: ICML (2024)

13. Gao, J., Yin, Z., Hua, C., Peng, Y., Liang, K., Ma, Z., Guo, J., Liu, Y.: Conmo: Controllable motion disentanglement and recomposition for zero-shot motion transfer. In: CVPR. pp. 7191–7200 (2025)

14. Gokmen, A.B., Ekin, Y., Bilecen, B.B., Dundar, A.: Ropecraft: Training-free motion transfer with trajectory-guided rope optimization on difusion transformers. arXiv preprint arXiv:2505.13344 (2025)

15. Guo, Y., Yang, C., Rao, A., Liang, Z., Wang, Y., Qiao, Y., Agrawala, M., Lin, D., Dai, B.: Animatedif: Animate your personalized text-to-image difusion models without specific tuning. In: ICLR (2024), https://openreview.net/forum?id= Fx2SbBgcte

16. HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D., Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari, N., Gordon, O., et al.: Ltx-video: Realtime video latent difusion. arXiv preprint arXiv:2501.00103 (2024)

17. Hedlin, E., Sharma, G., Mahajan, S., Isack, H., Kar, A., Tagliasacchi, A., Yi, K.M.: Unsupervised semantic correspondence using stable difusion. NeurIPS 36, 8266– 8279 (2023)

18. Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., Choi, Y.: Clipscore: A referencefree evaluation metric for image captioning. In: Proceedings of the 2021 conference on empirical methods in natural language processing. pp. 7514–7528 (2021)

19. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., Liu, Z.: VBench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)

20. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024)

21. Jeong, H., Park, G.Y., Ye, J.C.: Vmc: Video motion customization using temporal attention adaption for text-to-video difusion models. In: CVPR (2023)

22. Kang, S., Kim, J., Kim, J., Hwang, S.J.: Your large vision-language model only needs a few attention heads for visual grounding. In: CVPR. pp. 9339–9350 (2025)

23. Khachatryan, L., Movsisyan, A., Tadevosyan, V., Henschel, R., Wang, Z., Navasardyan, S., Shi, H.: Text2video-zero: Text-to-image difusion models are zeroshot video generators. In: ICCV. pp. 15954–15964 (2023)

24. Kim, C., Shin, H., Hong, E., Yoon, H., Arnab, A., Seo, P.H., Hong, S., Kim, S.: Seg4dif: Unveiling open-vocabulary segmentation in text-to-image difusion transformers. arXiv preprint arXiv:2509.18096 (2025)

25. Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al.: Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024)

26. Li, B., Zhang, Y., Wang, Q., Ma, L., Shi, X., Wang, X., Wan, P., Yin, Z., Zhuge, Y., Lu, H., Jia, X.: Vfxmaster: Unlocking dynamic visual efect generation via in-context learning (2025), https://arxiv.org/abs/2510.25772

27. Ling, P., Bu, J., Zhang, P., Dong, X., Zang, Y., Wu, T., Chen, H., Wang, J., Jin, Y.: Motionclone: Training-free motion cloning for controllable video generation. arXiv preprint arXiv:2406.05338 (2024)

28. Liu, B., Wang, C., Su, T., Ten, H., Huang, J., Guo, K., Jia, K.: Understanding attention mechanism in video difusion models. arXiv preprint arXiv:2504.12027 (2025)

29. Liu, Y., Zhang, K., Li, Y., Yan, Z., Gao, C., Chen, R., Yuan, Z., Huang, Y., Sun, H., Gao, J., et al.: Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177 (2024)

30. Ma, Y., Feng, K., Hu, Z., Wang, X., Wang, Y., Zheng, M., He, X., Zhu, C., Liu, H., He, Y., et al.: Controllable video generation: A survey. arXiv preprint arXiv:2507.16869 (2025)

31. Nam, J., Son, S., Chung, D., Kim, J., Jin, S., Hur, J., Kim, S.: Emergent temporal correspondences from video difusion transformers. arXiv preprint arXiv:2506.17220 (2025)

32. Pardyl, A., Rypeść, G., Kurzejamski, G., Zieliński, B., Trzciński, T.: Active visual exploration based on attention-map entropy. arXiv preprint arXiv:2303.06457 (2023)

33. Park, G.Y., Jeong, H., Lee, S.W., Ye, J.C.: Spectral motion alignment for video motion transfer using difusion models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 6398–6405 (2025)

34. Peruzzo, E., Sangineto, E., Liu, Y., De Nadai, M., Bi, W., Lepri, B., Sebe, N.: Spatial entropy as an inductive bias for vision transformers. Machine Learning 113(9), 6945–6975 (2024)

35. Pondaven, A., Siarohin, A., Tulyakov, S., Torr, P., Pizzati, F.: Video motion transfer with difusion transformers. In: CVPR. pp. 22911–22921 (2025)

36. Shi, Q., Wu, J., Bai, J., Zhang, J., Qi, L., Tong, Y., Li, X.: Decouple and track: Benchmarking and improving video difusion transformers for motion transfer. In: ICCV. pp. 10995–11005 (2025)

37. Siarohin, A., Lathuilière, S., Tulyakov, S., Ricci, E., Sebe, N.: Animating arbitrary objects via deep motion transfer. In: CVPR. pp. 2377–2386 (2019)

38. Tang, L., Jia, M., Wang, Q., Phoo, C.P., Hariharan, B.: Emergent correspondence from image difusion. NeurIPS 36, 1363–1389 (2023)

39. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: ECCV. pp. 402–419. Springer (2020)

40. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang, X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li, Y., Wu, Y., Liu, Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z., Han, Z., Wu, Z.F., Liu, Z.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025)

41. Wang, J., Ma, Y., Guo, J., Xiao, Y., Huang, G., Li, X.: Cove: Unleashing the difusion feature correspondence for consistent video editing. NeurIPS 37, 96541– 96565 (2024)

42. Wang, L., Mai, Z., Shen, G., Liang, Y., Tao, X., Wan, P., Zhang, D., Li, Y., Chen, Y.C.: Motion inversion for video customization. In: Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. pp. 1–12 (2025)

43. Wang, Y., Wang, L., Ma, Z., Hu, Q., Xu, K., Guo, Y.: Videodirector: Precise video editing via text-to-video models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2589–2598 (2025)

44. Xi, H., Yang, S., Zhao, Y., Xu, C., Li, M., Li, X., Lin, Y., Cai, H., Zhang, J., Li, D., et al.: Sparse videogen: Accelerating video difusion transformers with spatialtemporal sparsity. arXiv preprint arXiv:2502.01776 (2025)

45. Xiao, Z., Zhou, Y., Yang, S., Pan, X.: Video difusion models are training-free motion interpreter and controller. NeurIPS 37, 76115–76138 (2024)

46. Xing, J., Xia, M., Liu, Y., Zhang, Y., Zhang, Y., He, Y., Liu, H., Chen, H., Cun, X., Wang, X., et al.: Make-your-video: Customized video generation using textual and structural guidance. IEEE Transactions on Visualization and Computer Graphics 31(2), 1526–1541 (2024)

47. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., Yin, D., Yuxuan.Zhang, Wang, W., Cheng, Y., Xu, B., Gu, X., Dong, Y., Tang, J.: Cogvideox: Text-to-video difusion models with an expert transformer. In: ICLR (2025), https://openreview.net/forum?id=LQzN6TRFg9

48. Yatim, D., Fridman, R., Bar-Tal, O., Kasten, Y., Dekel, T.: Space-time difusion features for zero-shot text-driven motion transfer. In: CVPR. pp. 8466–8476 (2024)

49. Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L., Shum, H.Y.: Dino: Detr with improved denoising anchor boxes for end-to-end object detection. In: The Eleventh International Conference on Learning Representations (2023)

50. Zhao, R., Gu, Y., Wu, J.Z., Zhang, D.J., Liu, J.W., Wu, W., Keppo, J., Shou, M.Z.: Motiondirector: Motion customization of text-to-video difusion models. In: ECCV. pp. 273–290. Springer (2024)