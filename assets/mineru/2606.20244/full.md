# SPOT-E: Test-Time Entropy Shaping with Visual Spotlights for Frozen VLMs

Bo Yin1, Xiaobin Hu1, Chengming Xu2, Ruolin Shen3, Mo Yang4, Jiangning Zhang5, Peng-Tao Jiang6, Cheng Tan7, and Shuicheng Yan1

1 National University of Singapore

2 Fudan University

3 Technical University of Munich

4 Sagenic Tech

5 Zhejiang University

6 vivo

7 Shanghai Artificial Intelligence Laboratory

Abstract. Vision-language models (VLMs) often underperform on evidence intensive tasks because decisive visual evidence are small, localized, and easy to overlook, leading to failures in evidence readout even when high-level reasoning is intact. Prior inference-time visual interventions can improve grounding without retraining, but they are largely openloop and lack a mechanism to verify whether highlighted evidence is actually used. We study answer-span prediction entropy as a model-internal feedback signal and show that naive entropy minimization is ambiguous, since low entropy may arise from evidence-grounded confidence or shortcut collapse. To resolve this ambiguity, we introduce low-entropy anchors and an entropy-shaping objective that reduces answer uncertainty while preserving baseline high-confidence tokens. We instantiate this principle in SPOT-E, a plug-and-play test-time method that produces questionconditioned spotlights, optimized per instance via light-weight tuning based on Group Relative Policy Optimization (GRPO). Across all benchmarks and different VLM families, SPOT-E yields consistent gains and improved robustness under visual corruptions. Code is publicly available at: https://github.com/YinBo0927/SPOT-E

Keywords: Vision-language models · Test-time adaptation · Entropy

## 1 Introduction

Vision-language models have made rapid progress in multimodal understanding, yet they remain unreliable on evidence-intensive tasks such as chart reading and document parsing [21, 24, 25]. In these settings, the decisive evidence are often small and localized. A model may describe a correct reasoning plan, for example it may say “read the y-axis value and then compare the two bars”, but still misread the underlying number, causing the final answer to fail [10,23,46]. This pattern exposes a bottleneck that we call the evidence utilization gap: the model can reason about what evidence is needed, but cannot reliably extract and focus on the fine-grained visual evidence that determines the answer [31]. Closing this gap for frozen, already-deployed VLMs, without costly retraining or task-specific annotation, is practically urgent and methodologically challenging [40]. How can we improve evidence utilization at inference time while keeping the backbone frozen?

A natural direction is to intervene on the visual input at test time so that decisive regions become more accessible to the model [3, 30, 35, 37, 45]. Methods such as FGVP instantiate this idea and report gains without modifying model weights [36,43]. However, these approaches are open-loop. They apply a fixed intervention but do not provide a mechanism to verify whether the model actually relied on the emphasized evidence. When the selected region misses the decisive evidence, or when the intervention degrades the evidence itself, the failure remains invisible to the method and therefore cannot be corrected. This raises a key question: how can we design a test-time intervention that is both instance-adaptive and self-verifying?

These limitations motivate a closed-loop feedback signal that can be obtained from a frozen VLM during inference [17, 28]. We find that answer entropy, computed from the logits over the tokens that form a structured final answer, tracks evidence usability. Entropy is low when the decisive visual cues are clear, and it increases when they are obscured [18,19,34,38]. However, entropy reduction is inherently ambiguous. Both evidence-grounded confidence and shortcut behaviors can produce low entropy [4, 9, 39], as shown in Fig. 2. As a result, minimizing entropy alone can drive the model into confident-but-wrong shortcuts, creating a systematic failure mode in the absence of labels.

To resolve this ambiguity, we introduce an entropy-shaping principle that reduces answer entropy while preserving the model’s baseline high-confidence predictions on the unmodified input. Specifically, we identify low-entropy anchors as token positions where the frozen VLM is already near-deterministic at baseline, and we penalize interventions that disrupt these anchors. This yields a label-free objective that favors evidence-supported confidence over shortcutinduced collapse. We instantiate the principle in SPOT-E, a test-time visual adaptation framework that augments a frozen VLM with a lightweight, questionconditioned visual spotlight module that produces spotlights. For each instance, we optimize only the spotlight-module LoRA [13] parameters using Group Relative Policy Optimization [27] under the entropy-shaping reward.

Our contributions are summarized as follows:

– Entropy Signal. We identify answer entropy as a label-free signal for evidence utilization in frozen VLMs, and show that entropy reduction is inherently ambiguous.  
– Entropy Shaping. We propose low-entropy anchors and an entropy-shaping reward to disambiguate evidence-supported confidence from shortcut collapse.  
– SPOT-E Framework. We present SPOT-E, a plug-and-play test-time framework that keeps the VLM frozen while optimizing a question-conditioned visual spotlight via per-instance GRPO to leverage low-entropy anchors.

– Broad Evaluation. We conduct extensive evaluations across both opensource and closed-source VLM families and multiple backbones, demonstrating consistent gains on diverse benchmarks, with particularly strong improvements on evidence-intensive tasks and improved robustness under corruptions.

## 2 Related Work

## 2.1 Inference-Time Visual Interventions and Visual Prompting

Inference-time visual interventions improve grounding by manipulating the visual evidence presented to a frozen VLM at test time [2,5]. Common approaches include overlaying marks or masks to highlight regions, as in FGVP and Set-of-Mark prompting, selectively cropping or zooming into candidate areas to preserve small details, as in ViCrop, and spatial transformations that reallocate resolution toward query-relevant evidence, as in AttWarp [8, 35, 36, 45]. Another practical direction uses attention- or score-guided prompting strategies such as API-style prompting to steer the model toward informative regions without retraining [42]. These methods are appealing because they are lightweight and model-agnostic, but they often depend on fixed heuristics, external region proposals, or discrete choices that may be brittle across instances. Our method fits this paradigm and uses a question-conditioned spotlight for adaptive evidence emphasis under a frozen backbone.

## 2.2 Entropy and Uncertainty for Evidence Localization

Uncertainty signals have long been used to diagnose and steer model behavior at inference time [12, 18]. Entropy over output distributions is a common proxy for confidence and has been used for calibration and selective prediction, selfconsistency and re-ranking, as well as entropy- and confidence-driven decoding heuristics [11, 19, 33]. In vision and multimodal reasoning, uncertainty is also closely tied to evidence localization, since failures in fine-grained grounding often appear as high uncertainty concentrated on a small set of answer tokens [14, 20]. Recent multimodal work further leverages entropy or confidence to trigger additional perception, guide region selection, and filter visually unsupported generations [40, 41]. Our method combines an entropy-shaping reward with a question-conditioned visual spotlight to encourage decisive, visually supported answers without updating the frozen VLM.

## 3 Motivation

In this section, we motivate an entropy-centric view of visual adaptation in VLMs. When decisive visual evidence is usable, the model commits more consistently to a single final answer, making answer-span entropy a proxy for evidence use. We show that this effect is spatially grounded and can change sharply with the visibility of the decisive region. However, entropy reduction is ambiguous and may also reflect shortcut behaviors that suppress hard evidence, yielding confident-but-wrong predictions. To resolve this, we introduce low-entropy anchors and an entropy-shaping principle that reduces answer entropy while preserving anchor stability.

## 3.1 Notation

Consider a frozen VLM $F _ { \phi } ,$ which typically comprises a visual encoder $V _ { : }$ a multimodal connector $F _ { c } ,$ and an LLM M. Given an input image x and a user instruction $q ,$ the model receives a multimodal sequence of visual tokens $\{ v _ { 1 } , v _ { 2 } , \ldots , v _ { n } \}$ and text tokens $\{ t _ { 1 } , t _ { 2 } , \ldots , t _ { m } \}$ . During decoding, $F _ { \phi }$ processes the concatenated multimodal context $\{ v _ { 1 } , \ldots , v _ { n } , t _ { 1 } , \ldots , t _ { m } \}$ followed by previously generated tokens $\{ y _ { 1 } , \dotsc , y _ { k - 1 } \}$ to predict the next token $y _ { k }$ .

Next-token distribution. At decoding step $k ,$ the model outputs a distribution over the vocabulary W:

$$
p _ {k} (w) = p _ {\phi} (w \mid x, q, y _ {<   k}), \quad w \in \mathcal {W}, \tag {1}
$$

where $y _ { < k } = \left( y _ { 1 } , \ldots , y _ { k - 1 } \right)$ is the prefix.

Entropy. We quantify predictive uncertainty using Shannon entropy of the nexttoken distribution:

$$
H _ {k} (x, q) = - \sum_ {w \in \mathcal {W}} p _ {k} (w) \log p _ {k} (w). \tag {2}
$$

Unless otherwise specified, log(·) denotes the natural logarithm. We compute entropy under the baseline decoding trajectory.

Answer entropy. Long-form generations contain many tokens weakly related to visual evidence. To focus on evidence-relevant uncertainty, we compute entropy on a final answer span. We enforce a structured output format (e.g., Final answer: $\ldots \biggr )$ and extract the token indices of the answer span as $\mathcal { T } _ { \mathrm { a n s } }$ . We define answer entropy as

$$
H _ {\text { ans }} (x, q) = \frac {1}{| \mathcal {T} _ {\text { ans }} |} \sum_ {k \in \mathcal {T} _ {\text { ans }}} H _ {k} (x, q). \tag {3}
$$

For an intervention producing a modified input x˜, we measure entropy reduction by

$$
\Delta H _ {\text { ans }} = H _ {\text { ans }} (\tilde {x}, q) - H _ {\text { ans }} (x, q). \tag {4}
$$

Low-entropy anchors. To characterize token positions that the base model is already confident about, we define a set of low-entropy anchors under the baseline input. Concretely, we select the K positions with the smallest nexttoken entropies:

$$
\mathcal {I} _ {\text { low }} (x, q) = \operatorname{TopK} _ {\text { small }} \left(\left\{H _ {k} (x, q) \right\} _ {k = 1} ^ {T}\right), \tag {5}
$$

where $T$ is the output length. In addition, we evaluate $H _ { k } ( \tilde { x } , q )$ by conditioning on the baseline token prefix to align anchor positions. These anchors are later used to distinguish desirable entropy reduction from shortcut behaviors.

![](images/30873c9d4aa8e9d1d85d0ba30e69de12a4cbf1ef14066df57edbc62ed1280350.jpg)

<details>
<summary>text_image</summary>

Question: What color is the upper garment of the person in the image?
Origin	Blur 0	Blur 1	Blur 2
Answer: Red
Hans: 0.45	Answer: Red
Hans: 0.50	Answer: Red
Hans: 0.27	Answer: Green
Hans: 0.98
</details>

Fig. 1: Localized evidence controls answer entropy. We apply region-level blur to a subset of grid regions while keeping all other pixels unchanged, and measure $H _ { \mathrm { a n s } } ( \tilde { x } _ { S } , q )$ .

## 3.2 Visual Evidence Shapes Answer Entropy

A natural intuition is that visual evidence affects a VLM’s answer mainly by changing how certain the model can be at the point of committing to the final answer. If the decisive evidence for $q$ is clear, the model should concentrate probability mass on a consistent answer; if the evidence is weak or obscured, the model should remain uncertain. We operationalize this intuition using the answer entropy $H _ { \mathrm { a n s } } ( x , q )$ in Eq. (3), which measures uncertainty on the final answer.

We validate the intuition with a simple spatial sensitivity analysis. We partition the image into a coarse grid of regions $\mathcal { R } = \{ r _ { 1 } , . . . , r _ { M } \}$ and apply localized interventions while keeping the rest of the input unchanged. For a given subset size n, we sample subsets $S \subseteq \mathcal { R }$ with $| S | = n$ , construct an intervened image $\tilde { x } _ { S }$ by applying a fixed region-level transformation only within regions in $S$ such as blurring, and compute $H _ { \mathrm { a n s } } ( \tilde { x } _ { S } , q )$ . Fig. 1 shows a representative instance where suppressing the localized evidence required by q increases $H _ { \mathrm { a n s } }$ , eventually leading to incorrect predictions.

## 3.3 Entropy Reduction is Ambiguous

The previous subsection suggests that lowering $H _ { \mathrm { a n s } }$ often correlates with making decisive evidence more usable. However, entropy reduction alone is ambiguous: an intervention may decrease uncertainty not by improving evidence quality, but by attenuating hard evidence and steering the model toward priors or salient distractors, yielding confident-but-wrong answers with smaller $H _ { \mathrm { a n s } }$ .

We illustrate this ambiguity within the same subset-based intervention framework, but now restricting attention to the most entropy-sensitive singleton subset. Concretely, for any subset $S \subseteq \mathcal { R }$ , we define the entropy change induced by intervening on S as

$$
\Delta H _ {\mathrm{ans}} (S) = H _ {\mathrm{ans}} (x, q) - H _ {\mathrm{ans}} (\tilde {x} _ {S}, q), \tag {6}
$$

![](images/215ec8b13fcc9b466c3a49105ebdc1790705df3415a8d4e3d13fd5735ae3c4cf.jpg)  
Fig. 2: Entropy reduction can be misleading. We blur only the most entropysensitive singleton subset $S ^ { \star }$ with strength α. $H _ { \mathrm { a n s } }$ often rises as evidence becomes ambiguous, but may drop again when evidence is erased.

where $\tilde { x } _ { S }$ modifies only regions in S and leaves all other pixels unchanged. We identify

$$
S ^ {\star} = \arg \max _ {| S | = 1} \left| \Delta H _ {\text { ans }} (S) \right|, \tag {7}
$$

i.e., the single-region subset whose intervention yields the largest magnitude of entropy change. We then keep the rest of the image fixed and construct a family of intervened inputs by blurring only within $S ^ { \star }$ with increasing strength α:

$$
\tilde {x} _ {S ^ {\star}} ^ {(\alpha)} = \mathcal {A} _ {\alpha} (x; S ^ {\star}), \quad \alpha \in [ 0, 1 ], \tag {8}
$$

where $\mathcal { A } _ { \alpha } ( \cdot ; S ^ { \star } )$ applies blur of level α to regions in $S ^ { \star }$ and leaves all other regions unchanged.

Sweeping α typically yields a non-monotonic profile: $H _ { \mathrm { a n s } }$ is low when the evidence is clear (small α), peaks when the evidence becomes ambiguous (intermediate α), and can decrease again once the evidence is effectively erased (large $\alpha )$ , as the model collapses to a prior- or distractor-driven answer. Fig. 2 shows that low $H _ { \mathrm { a n s } }$ at large α can also arise from prior-driven overconfidence.

## 3.4 Entropy Shaping with Low-Entropy Anchors

Answer entropy $H _ { \mathrm { a n s } }$ provides a convenient scalar readout of how concentrated the model is on the final answer. The difficulty is that the same decrease in $H _ { \mathrm { a n s } }$ can be produced by different mechanisms: the intervention may genuinely expose missing evidence, or it may remove hard evidence and let the model settle on priors or distractors. To disambiguate these cases, we look beyond the answer entropy itself and ask whether an intervention is non-destructive, namely whether it preserves parts of the decoding process that the base model was already confident about. As illustrated in Fig. 3, two interventions may achieve a similar drop in answer entropy, yet only the evidence-revealing one preserves the baseline’s low-entropy tokens, whereas the destructive shortcut inflates their entropy motivating our anchor disruption measure.

![](images/72e62c54500e46e051557a172f6b2f7d64e35989593d995d57fa281441258c1a.jpg)  
Fig. 3: Low-entropy anchors reveal destructive shortcuts.

We use low-entropy anchors $\mathcal { T } _ { \mathrm { l o w } } ( x , q )$ in Eq. (5) to represent such stable positions under the baseline input. Given an intervened input x˜, we measure anchor disruption by the average entropy increase on anchor positions,

$$
\Delta H _ {\text { low }} (\tilde {x}) = \frac {1}{| \mathcal {I} _ {\text { low }} |} \sum_ {k \in \mathcal {I} _ {\text { low }}} \max \big (0, H _ {k} (\tilde {x}, q) - H _ {k} (x, q) \big). \tag {9}
$$

Interventions with comparable reductions in $H _ { \mathrm { a n s } }$ can behave very differently under this criterion. Evidence-revealing interventions tend to keep $\varDelta H _ { \mathrm { l o w } }$ small, while shortcut interventions often reduce $H _ { \mathrm { a n s } }$ at the cost of increasing entropy on anchors. This motivates an entropy-shaping principle that favors reducing $H _ { \mathrm { a n s } }$ while preserving low-entropy anchors.

## 4 SPOT-E: Visual Spotlighting for Entropy Shaping

Building on Sec. 3, we introduce SPOT-E. As shown in Fig. 4, SPOT-E performs entropy-guided test-time visual adaptation by keeping the VLM $F _ { \phi }$ frozen and optimizing a lightweight, question-conditioned visual spotlight. Given an image x and instruction $q ,$ the spotlight produces an intervened image $\tilde { x } = S ( x ; , m )$ , where S is the spotlighting operator and m is a soft pixel mask. We feed x˜ into $F _ { \phi }$ to compute the answer entropy $H _ { \mathrm { a n s } } ( \tilde { x } , q )$ and anchor entropies on $\mathcal { T } _ { \mathrm { l o w } } ( x , q )$ . At test time, SPOT-E runs a short per-instance episode: it samples candidate spotlights, scores them with an entropy-shaping objective, and updates only the spotlight parameters via GRPO [27]. The final prediction is chosen by Best-of-N over candidates, and the spotlight is reset after each instance.

## 4.1 Visual Spotlight

As illustrated in Fig. 5, SPOT-E introduces a CLIP-based [26] visual spolight module to produce question-conditioned visual spotlights. Given instruction $q ,$ we extract a compact visual phrase q¯ by retaining the key entities and attributes relevant to visual grounding. Then the module extracts global patch tokens and local crop tokens with a CLIP vision encoder with LoRA [13] adapter, matches them to the frozen CLIP text embedding via patch–text similarity to obtain relevance maps, and fuses multi-view evidence by max pooling to form the final spotlight mask.

![](images/7e349a260a0c475b9893b311464fd550026030c7e3a7a7f961c3871d473484b6.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Image"] --> B["Visual Spotlight"]
    C["Frozen"] -.-> B
  B --> D["Spotlight Image"]
  D --> E["VLM"]
  E --> F["Final Reward"]
  F --> G["Dynamic Clarity\nReward R_clarity(x̂)"]
  F --> H["Anchor-preservation\nReward R_preserve(x̂)"]
  G --> I["Question: Is there a bear in the photo?"]
  H --> I
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style F fill:#bbf,stroke:#333
    style G fill:#bfb,stroke:#333
    style H fill:#bfb,stroke:#333
```
</details>

Fig. 4: SPOT-E overview. SPOT-E freezes the VLM and optimizes a lightweight visual spotlight at test time to generate an intervened image, scored by answer-entropy clarity and anchor-preservation.

![](images/9f1b5e7b0af74dc56f9e6a88cd2e8ad1769425bacf3282ee0e91e3d12ba4b070.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Original Image"] --> B["Crop"]
  B --> C["A bear in the image"]
  C --> D["Text Encoder"]
  D --> E["Text Embedding"]
  E --> F["Global Patch Tokens"]
  F --> G["Cosine Similarity"]
  G --> H["Max Fusion"]
  H --> I["Output Image"]
  J["LoRA"] --> K["inject"]
  K --> L["Vision Encoder"]
  L --> M["Local Patch Tokens"]
  M --> N["Cosine Similarity"]
  N --> O["Max Fusion"]
  P["Text Token"] --> Q["Text Embedding"]
  R["Frozen"] --> S["Tunable"]
  T["Relevance Map"] --> U["Relevance Map"]
  V["Global Patch Tokens"] --> W["Global Patch Tokens"]
  X["Text Token"] --> Y["Text Token"]
  Z["Vision Token"] --> AA["Vision Token"]
```
</details>

Fig. 5: SPOT-E visual spotlight module. Both the image encoder and the text encoder are CLIP. The module computes patch-text similarities on the global view and local crops, then fuses multi-view relevance maps via max pooling to produce the final spotlight mask.

Global view and crop max fusion. We run the CLIP on the full image and on multiple crops to avoid missing small evidence. Let $x ^ { ( 0 ) } = x$ be the global view $\{ x ^ { ( i ) } \} _ { i = 1 } ^ { N _ { c } }$ $x ^ { ( i ) }$ $\{ p _ { j } ^ { ( i ) } \} _ { j = 1 } ^ { N _ { i } }$ and a text embedding t(¯q). We compute patch relevance by cosine similarity:

$$
u _ {j} ^ {(i)} = \Big \langle \mathrm{norm} (p _ {j} ^ {(i)}), \mathrm{norm} (t (\bar {q})) \Big \rangle . \tag {10}
$$

We reshape $\{ u _ { j } ^ { ( i ) } \}$ to a 2D grid and map each crop grid back to the full-image coordinates via $\mathcal { W } _ { i } ( \cdot )$ , which warps the i-th crop map to the global image coordinates. The fused relevance map is obtained by max fusion:

$$
u = \max \left(u ^ {(0)}, \max _ {i = 1, \dots , N _ {c}} \mathcal {W} _ {i} \left(u ^ {(i)}\right)\right). \tag {11}
$$

Soft mask and spotlighting operator. We upsample the fused relevance map u to the image resolution and obtain a soft pixel mask

$$
m = \sigma \left(\frac {1}{\tau} u ^ {\uparrow}\right) \in [ 0, 1 ] ^ {H \times W}, \tag {12}
$$

where u↑ denotes bilinear upsampling of u and τ controls mask sharpness. We then form the intervened input via

$$
\tilde {x} = \mathcal {S} (x; m) = m \odot x + (1 - m) \odot \mathcal {B} (x), \tag {13}
$$

where ⊙ denotes element-wise multiplication and $B ( \cdot )$ is a fixed backgrounddegrading transform (background dimming).

## 4.2 Entropy-Shaping Reward

Given a candidate spotlight mask m and intervened input $\tilde { x } = { \cal { S } } ( x ; m )$ , we score it using an entropy-shaping reward

$$
R (\tilde {x}) = R _ {\text { clarity }} (\tilde {x}) + R _ {\text { preserve }} (\tilde {x}). \tag {14}
$$

Both terms are computed from the frozen VLM logits, while gradients update only the visual spotlight module.

Dynamic clarity reward. We encourage the model to become more decisive on the final answer span by reducing answer entropy:

$$
\Delta H _ {\mathrm{ans}} (\tilde {x}) = H _ {\mathrm{ans}} (x, q) - H _ {\mathrm{ans}} (\tilde {x}, q). \tag {15}
$$

To avoid over-optimizing when the base model is already confident, we apply a dynamic scaling factor based on the baseline answer entropy:

$$
R _ {\text {clarity}} (\tilde {x}) = \gamma (x, q) \cdot \Delta H _ {\text {ans}} (\tilde {x}), \quad \gamma (x, q) = \frac {H _ {\text {ans}} (x , q)}{H _ {\text {ans}} (x , q) + c}, \tag {16}
$$

where $c > 0$ is a small constant. When the baseline is uncertain, γ increases and entropy reduction is rewarded more; when the baseline is already confident, γ suppresses blind exploration.

Anchor-preservation reward. Entropy reduction can be achieved by shortcut behaviors that disrupt tokens the base model was already confident about. We therefore penalize entropy increases on low-entropy anchor positions $\mathcal { T } _ { \mathrm { l o w } } ( x , q )$ (Eq. (5)) using the anchor disruption measure $\varDelta H _ { \mathrm { l o w } } ( \tilde { x } ) \ ( \mathrm { E q . \ ( 9 ) } )$ :

$$
R _ {\text { preserve }} (\tilde {x}) = - \lambda \cdot \Delta H _ {\text { low }} (\tilde {x}), \tag {17}
$$

where λ controls the strength of anchor preservation.

## 4.3 Test-Time Optimization with GRPO

SPOT-E runs a short per-instance test-time optimization episode to adapt the visual spotlight module, while keeping the VLM $F _ { \phi }$ fully frozen. Let θ denote the visual spotlight parameters (LoRA adapters in CLIP vision encoder attention layers), initialized to $\theta _ { 0 }$ for each instance.

Group sampling and scoring. At each iteration, we sample a group of N candidate masks by injecting Gaussian noise into the visual spotlight and obtain intervened inputs $\bar { \{ x ^ { ( n ) } \} } _ { n = 1 } ^ { N }$ . Each candidate is scored by the total reward $R ( \tilde { x } ^ { ( n ) } )$ (Eq. (14)).

Group-relative advantages. We compute standardized advantages within the group:

$$
\mu_ {R} = \frac {1}{N} \sum_ {n = 1} ^ {N} R (\tilde {x} ^ {(n)}), \quad \sigma_ {R} = \sqrt {\frac {1}{N} \sum_ {n = 1} ^ {N} \left(R (\tilde {x} ^ {(n)}) - \mu_ {R}\right) ^ {2}}. \tag {18}
$$

$$
A ^ {(n)} = \frac {R (\tilde {x} ^ {(n)}) - \mu_ {R}}{\sigma_ {R} + \epsilon}. \tag {19}
$$

where ϵ is a small constant.

GRPO update and reset. We apply a standard $\mathrm { G R P O }$ clipped policy update on θ using $\left\{ A ^ { ( n ) } \right\}$ , with a KL regularizer to keep the visual spotlight close to its initialization.

$$
\bar {r} ^ {(n)} (\theta) = \operatorname{clip} \left(r ^ {(n)} (\theta), 1 - \delta , 1 + \delta\right). \tag {20}
$$

$$
\mathcal {L} _ {\mathrm{GRPO}} (\theta) = - \frac {1}{N} \sum_ {n = 1} ^ {N} \min \Big (r ^ {(n)} (\theta) A ^ {(n)}, \bar {r} ^ {(n)} (\theta) A ^ {(n)} \Big) + \beta \operatorname{KL} \big (\pi_ {\theta} \| \pi_ {\theta_ {0}} \big). \tag {21}
$$

where $\delta$ is the clipping threshold and $\beta$ controls the KL strength. We update only θ by gradient descent on Eq. (21), and reset $\theta  \theta _ { 0 }$ after each instance to avoid cross-sample drift.

## 5 Experiments

Overview. We evaluate SPOT-E from four complementary angles. First, we report main results across a broad set of frozen backbones, covering both opensource model families and closed-source VLM APIs, to test generality. Second, we compare against strong inference-time visual prompting baselines under matched decoding settings. Third, we assess out-of-distribution robustness under controlled visual corruptions and analyze confidence behavior through answer entropy. Finally, we conduct targeted ablations on the reward, spotlight design, and test-time budget, and provide qualitative case studies to illustrate how SPOT-E changes evidence usage at inference time. Due to space constraints, additional experimental results, such as those using larger backbones, are provided in the Appendix.

Table 1: Applying SPOT-E to closed-source and open-source backbones.

<table><tr><td>Base Model</td><td>TextVQA</td><td>DocVQA</td><td>ChartQA</td><td>MathVista</td><td>MMMU</td><td>GQA</td><td>MMBench</td><td>POPE</td></tr><tr><td colspan="9">Closed-Source</td></tr><tr><td>GPT-4o [16]</td><td>77.4</td><td>91.1</td><td>86.7</td><td>63.5</td><td>69.2</td><td>73.0</td><td>83.1</td><td>86.9</td></tr><tr><td>+ SPOT-E (Ours)</td><td>79.9 +2.5↑</td><td>92.3 +1.2↑</td><td>88.2 +1.5↑</td><td>65.5 +2.0↑</td><td>70.4 +1.2↑</td><td>73.8 +0.8↑</td><td>83.9 +0.8↑</td><td>87.9 +1.0↑</td></tr><tr><td>GPT-4o-mini [16]</td><td>70.0</td><td>86.0</td><td>80.0</td><td>55.0</td><td>60.0</td><td>68.0</td><td>78.0</td><td>84.0</td></tr><tr><td>+ SPOT-E (Ours)</td><td>73.5 +3.5↑</td><td>88.0 +2.0↑</td><td>82.5 +2.5↑</td><td>58.0 +3.0↑</td><td>62.0 +2.0↑</td><td>69.0 +1.0↑</td><td>79.2 +1.2↑</td><td>85.2 +1.2↑</td></tr><tr><td>Gemini-2.5-Flash [7]</td><td>80.0</td><td>91.5</td><td>84.0</td><td>68.0</td><td>70.0</td><td>72.0</td><td>82.0</td><td>86.0</td></tr><tr><td>+ SPOT-E (Ours)</td><td>82.8 +2.8↑</td><td>93.0 +1.5↑</td><td>85.8 +1.8↑</td><td>70.2 +2.2↑</td><td>71.5 +1.5↑</td><td>72.8 +0.8↑</td><td>82.8 +0.8↑</td><td>86.9 +0.9↑</td></tr><tr><td colspan="9">Open-Source</td></tr><tr><td>Qwen2.5-VL-7B [32]</td><td>84.9</td><td>85.7</td><td>87.3</td><td>67.8</td><td>55.0</td><td>64.0</td><td>82.6</td><td>86.4</td></tr><tr><td>+ SPOT-E (Ours)</td><td>86.9 +2.0↑</td><td>86.5 +0.8↑</td><td>88.5 +1.2↑</td><td>70.8 +3.0↑</td><td>58.5 +3.5↑</td><td>65.0 +1.0↑</td><td>83.5 +0.9↑</td><td>87.4 +1.0↑</td></tr><tr><td>Qwen3-VL-8B [1]</td><td>86.0</td><td>86.2</td><td>88.0</td><td>70.5</td><td>58.0</td><td>65.5</td><td>83.8</td><td>87.2</td></tr><tr><td>+ SPOT-E (Ours)</td><td>87.8 +1.8↑</td><td>86.8 +0.6↑</td><td>89.0 +1.0↑</td><td>73.3 +2.8↑</td><td>61.0 +3.0↑</td><td>66.4 +0.9↑</td><td>84.6 +0.8↑</td><td>88.1 +0.9↑</td></tr><tr><td>LLaVA-NeXT-7B [21]</td><td>78.5</td><td>80.0</td><td>79.0</td><td>47.0</td><td>38.0</td><td>63.0</td><td>75.0</td><td>85.0</td></tr><tr><td>+ SPOT-E (Ours)</td><td>84.7 +6.2↑</td><td>82.0 +2.0↑</td><td>81.5 +2.5↑</td><td>50.5 +3.5↑</td><td>41.0 +3.0↑</td><td>64.5 +1.5↑</td><td>76.5 +1.5↑</td><td>86.5 +1.5↑</td></tr><tr><td>LLaVA-OV-7B [21]</td><td>80.0</td><td>81.0</td><td>80.5</td><td>48.5</td><td>39.5</td><td>63.5</td><td>76.0</td><td>85.5</td></tr><tr><td>+ SPOT-E (Ours)</td><td>85.0 +5.0↑</td><td>82.8 +1.8↑</td><td>82.8 +2.3↑</td><td>51.8 +3.3↑</td><td>42.5 +3.0↑</td><td>64.9 +1.4↑</td><td>77.4 +1.4↑</td><td>86.9 +1.4↑</td></tr><tr><td>InternVL2.5-8B [6]</td><td>81.0</td><td>82.0</td><td>83.0</td><td>66.0</td><td>56.0</td><td>62.0</td><td>80.5</td><td>89.0</td></tr><tr><td>+ SPOT-E (Ours)</td><td>84.5 +3.5↑</td><td>83.5 +1.5↑</td><td>85.0 +2.0↑</td><td>69.5 +3.5↑</td><td>59.5 +3.5↑</td><td>63.2 +1.2↑</td><td>81.5 +1.0↑</td><td>90.0 +1.0↑</td></tr><tr><td>InternVL3-8B [6]</td><td>80.2</td><td>82.7</td><td>86.6</td><td>71.6</td><td>62.7</td><td>61.0</td><td>81.7</td><td>91.1</td></tr><tr><td>+ SPOT-E (Ours)</td><td>82.4 +2.2↑</td><td>83.7 +1.0↑</td><td>88.1 +1.5↑</td><td>74.4 +2.8↑</td><td>65.2 +2.5↑</td><td>62.2 +1.2↑</td><td>82.5 +0.8↑</td><td>92.0 +0.9↑</td></tr></table>

## 5.1 Implementation Details

Models. To verify the effectiveness of our approach, we apply SPOT-E to multiple frozen open-source VLM backbones spanning three representative families: Qwen-VL [32], LLaVA [21], and InternVL [6], and three proprietary VLM APIs that expose token-level log probabilities: GPT-4o, GPT-4o-mini [16], and Gemini-2.5-Flash [7]. Unless otherwise stated, all backbones are kept fully frozen and SPOT-E updates only the CLIP-based eye module at test time with perinstance reset.

Evaluation. Our evaluation comprises multiple benchmarks that stress finegrained visual grounding and localized evidence usage, spanning text-centric grounding (TextVQA [29], DocVQA [25], ChartQA [24]), compositional VQA and general multimodal understanding (GQA [15], MMBench [22]), knowledgeand reasoning-intensive tasks (MathVista [23], MMMU [44]), and hallucinationoriented evaluation (POPE [20]). We follow the standard evaluation protocols and report the official metrics for each benchmark.

## 5.2 Main Results

Consistent Improvements Across Frozen Backbones. We compare the frozen base model and its SPOT-E augmented version across both closed-source VLM APIs and open-source backbones which cover three backbone families with two released variants per family. Table 1 shows that SPOT-E yields consistent gains across all evaluated models. The improvements are most pronounced on evidence-intensive benchmarks such as TextVQA, DocVQA, ChartQA, and MathVista, where answers depend on small text, numbers, or localized symbols.

Table 2: Comparison with inference-time visual evidence manipulation baselines.

<table><tr><td>Method</td><td>TextVQA</td><td>GQA</td><td>MMMU</td><td>POPE</td><td>DocVQA</td></tr><tr><td>FGVP-Mask [36]</td><td>77.3</td><td>55.8</td><td>46.0</td><td>84.4</td><td>56.6</td></tr><tr><td>FGVP-RBM [36]</td><td>72.3</td><td>55.8</td><td>46.5</td><td>81.3</td><td>38.6</td></tr><tr><td>SoM [35]</td><td>61.5</td><td>47.8</td><td>45.1</td><td>75.8</td><td>57.4</td></tr><tr><td>API [42]</td><td>81.6</td><td>61.1</td><td>47.4</td><td>85.8</td><td>68.4</td></tr><tr><td>ViCrop [45]</td><td>83.8</td><td>60.6</td><td>47.1</td><td>86.7</td><td>82.5</td></tr><tr><td>AttWarp [8]</td><td>84.7</td><td>64.0</td><td>50.4</td><td>87.4</td><td>84.1</td></tr><tr><td>SPOT-E</td><td>86.9</td><td>65.0</td><td>58.5</td><td>87.4</td><td>86.5</td></tr></table>

![](images/045158dc22c980e0f2f4c396b785dee583b9dfd3492d2fd2e0b91949cd6c77bb.jpg)

![](images/f2486259754a3ac75e8471bdd4e87b105ed923517b3ec04b5ae181bf4088cfb7.jpg)

<details>
<summary>line chart</summary>

| Severity | Gaussian noise (Red) | Gaussian noise (Green) | Low-res (Red) | Low-res (Green) | Occlusion (Red) | Occlusion (Green) |
| -------- | -------------------- | ---------------------- | ------------- | --------------- | --------------- | ----------------- |
| 0        | 82.5                 | 89.0                   | 81.5          | 88.0            | 83.0            | 89.5              |
| 1        | 79.5                 | 87.5                   | 79.5          | 86.5            | 80.5            | 88.0              |
| 2        | 76.5                 | 86.0                   | 76.5          | 85.0            | 77.0            | 86.5              |
| 3        | 73.0                 | 84.5                   | 73.0          | 83.5            | 73.5            | 85.0              |
| 4        | 69.0                 | 83.0                   | 69.0          | 82.0            | 68.0            | 83.5              |
| 5        | 64.5                 | 81.5                   | 64.5          | 80.5            | 62.5            | 82.0              |
</details>

Fig. 6: Out-of-distribution evaluation.

On broader multimodal reasoning benchmarks (GQA, MMBench, and MMMU), SPOT-E still provides positive but typically smaller gains, suggesting that suppressing distractors and amplifying decisive regions complements backbone reasoning capacity rather than replacing it. Finally, on POPE, SPOT-E tends to improve factual consistency by steering generation toward visually supported answers, indicating that entropy-guided spotlighting can mitigate confident-butunsupported responses even without modifying the underlying VLM.

Comparison with Visual Prompting Baselines. Since several strong baselines improve grounding by manipulating visual evidence at inference time, we compare SPOT-E with representative methods including FGVP [36], SoM [35], API [42], ViCrop [45], and AttWarp [8]. All methods are evaluated on the frozen Qwen2.5-VL-7B backbone under the same decoding configuration, following each baseline’s standard inference-time procedure. As shown in Table 2, SPOT-E is competitive with these visual intervention baselines and yields further improvements across benchmarks, with particularly strong gains on evidence-intensive tasks where small, localized cues are critical.

Out-of-distribution Robustness under Visual Corruptions. To evaluate robustness under domain shift, we test on TextVQA with three synthetic corruptions applied at inference time: Gaussian noise, low-resolution downsampling, and local occlusion. We sweep corruption severity and plot accuracy curves for the frozen Qwen2.5-VL-7B baseline and +SPOT-E under the same decoding setting. As shown in Fig. 6, SPOT-E consistently reduces performance drop across severities, indicating improved robustness to corrupted visual evidence.

![](images/d700060fe8a51ccdcc963d4cb92f5930ad0f59ee82202846fc8fb2860b6354a7.jpg)

<details>
<summary>box plot</summary>

| Category         | Min  | Q1   | Median | Q3   | Max  |
| ---------------- | ---- | ---- | ------ | ---- | ---- |
| Base Correct     | 0.0  | 0.15 | 0.12   | 0.25 | 0.35 |
| Base Wrong       | 0.0  | 0.25 | 0.20   | 0.45 | 0.55 |
| +SPOT-E Correct  | 0.0  | 0.15 | 0.12   | 0.25 | 0.35 |
| +SPOT-E Wrong    | 0.0  | 0.45 | 0.40   | 0.65 | 0.90 |
</details>

Fig. 7: Confidence calibration boxplot.

![](images/283faeeda6b64507be553ae8d48e8bc49ee2310ca2fb271274e202b92aa2e252.jpg)

<details>
<summary>line chart</summary>

| Update steps per instance | Qwen2.5-VL-7B | InternVL2.5-8B | LLaVA-NeXT-7B |
| ------------------------- | ------------- | -------------- | ------------- |
| 0                         | 81.0          | 81.0           | 81.0          |
| 1                         | 81.5          | 82.0           | 81.5          |
| 2                         | 82.5          | 83.0           | 82.0          |
| 4                         | 83.5          | 84.0           | 83.0          |
| 8                         | 85.5          | 84.5           | 84.0          |
| 16                        | 85.8          | 84.8           | 84.2          |
</details>

Fig. 8: Test-time budget discussion.

Confidence Calibration via Answer Entropy. We analyze how SPOT-E affects overconfident errors by measuring the answer entropy on each example. Fig. 7 reports boxplots of $H _ { \mathrm { a n s } }$ for correct and incorrect predictions under the frozen baseline and +SPOT-E. SPOT-E increases entropy on incorrect cases while maintaining low entropy on correct ones, reducing confident-butunsupported responses and improving the separation between correct and wrong predictions.

## 5.3 Ablation Studies

We ablate three factors that govern SPOT-E: (i) the reward design, (ii) the spotlight configuration, and (iii) the test-time update budget (number of adaptation steps per instance). Unless otherwise stated, all ablations are conducted on frozen Qwen2.5-VL-7B under the same decoding setup as the main results, and we report the official benchmark metrics. For fair comparison, we keep the spotlight operator, learning rate, and evaluation prompts fixed, varying only the targeted component in each study.

Reward Design. We ablate the entropy-shaping reward in Sec. 4.2 to quantify the contribution of each term in Eq. (14). Keeping the spotlight mechanism and the test-time budget fixed, we compare: (i) Clarity-only, using $R _ { \mathrm { c l a r i t y } }$ alone; (ii) Preserve-only, using $R _ { \mathrm { p r e s e r v e } }$ alone; (iii) Full reward, using $R _ { \mathrm { c l a r i t y } } + R _ { \mathrm { p r e s e r v e } } ;$ and (iv) w/o dynamic scaling, where we replace $\gamma ( x , q )$ in Eq. (16) with a constant factor. Tab. 3 shows that combining clarity and anchor preservation yields the most consistent gains, while removing either term degrades performance. These results suggest that $R _ { \mathrm { c l a r i t y } }$ and $R _ { \mathrm { p r e s e r v e } }$ are complementary. The clarity term encourages decisiveness on the answer span, while the preservation term discourages shortcut updates that disrupt already-reliable evidence.

Visual Spotlight Design. We ablate the spotlight module in Sec. 4.1 to assess the impact of multi-view fusion and the spotlighting operator. Keeping the reward and test-time budget fixed, we compare: (i) Global (only the global view; $N _ { c } { = } 0 )$ , (ii) MeanFuse (average fusion; max → mean in Eq. (11)), and (iii) NoBgDeg (no background degradation; $\scriptstyle B ( x ) = x$ in Eq. (13)). Table 4 shows that the default design (Default) provides the most consistent improvements, while removing crops or disabling background degradation reduces the benefit of evidence localization.

Table 3: Reward design ablation.

<table><tr><td>Variant</td><td>TextVQA</td><td>MathVista</td><td>POPE</td></tr><tr><td> $R_{\text{clarity only}}$ </td><td>85.8</td><td>69.4</td><td>86.9</td></tr><tr><td> $R_{\text{preserve only}}$ </td><td>84.9</td><td>68.6</td><td>87.1</td></tr><tr><td>w/o dynamic</td><td>86.4</td><td>70.1</td><td>87.2</td></tr><tr><td>Ours</td><td>86.9</td><td>70.8</td><td>87.4</td></tr></table>

Table 4: Spotlight design ablation.

<table><tr><td>Variant</td><td>TextVQA</td><td>MathVista</td><td>POPE</td></tr><tr><td>GLOBAL</td><td>84.1</td><td>66.9</td><td>86.3</td></tr><tr><td>MEANFUSE</td><td>85.7</td><td>68.5</td><td>86.9</td></tr><tr><td>NoBgDEG</td><td>85.2</td><td>67.8</td><td>86.7</td></tr><tr><td>Ours</td><td>86.9</td><td>70.8</td><td>87.4</td></tr></table>

Question: How many bears in the photo in this image?  
![](images/cc5153a3643d5dd31ee4b478408534a69787e5ad9652da7d8362d1223c54956d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["Input original image"] --> B["VLM"]
  C["+SPOT-E"] --> D["VLM"]
  B --> E["Answer: There are four toy bears in the image. H_ans: 0.56"]
  D --> F["Answer: In the photo on the lower right corner of the image, there is one bear. H_ans: 0.14"]
```
</details>

Question: What is the License plate number of the bus?  
![](images/c3f88991dec8bb8fe6a0ac99cae4e37ed38fc1509044156a7d9b051af967fa44.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["Input original image"] --> B["VLM"]
  C["+SPOT-E"] --> D["VLM"]
  B --> E["Answer: It is a red No.52 bus driving on the road.<br>H_ans: 0.72"]
  D --> F["Answer: The license plate number is LK62DMY that below the bus.<br>H_ans: 0.19"]
```
</details>

Fig. 9: Qualitative case studies comparing the frozen baseline and +SPOT-E with the same inference setup.

Test-Time Budget. We vary the test-time adaptation budget by sweeping the number of eye-module update steps per instance in 0, 1, 2, 4, 8, 16, fixing the reward, spotlight design, learning rate, and decoding, where 0 is the frozen baseline. We evaluate on Qwen2.5-VL-7B, InternVL2.5-8B, and LLaVA-NeXT-7B. As shown in Fig. 8, accuracy increases with more steps and then saturates; we use 8 steps by default as it captures most gains with modest overhead.

## 5.4 Case Studies

We provide qualitative comparisons to show how SPOT-E changes visual evidence usage at inference time. Fig. 9 contrasts the frozen baseline and ,+SPOT-E under the same decoding setup, showing the original input, the spotlightintervened image, and the resulting outputs. In both examples, the baseline is distracted by salient but irrelevant regions and answers incorrectly with higher $H _ { \mathrm { a n s } } ,$ whereas SPOT-E suppresses distractors, amplifies the decisive evidence, and produces a visually supported answer with lower $H _ { \mathrm { a n s } }$ .

## 6 Conclusion

We present SPOT-E, a plug-and-play test-time method that strengthens finegrained visual evidence utilization in frozen VLMs via lightweight per-instance adaptation of a question-conditioned visual spotlight module. Across diverse backbones and benchmarks, SPOT-E delivers consistent gains and stronger robustness under visual corruptions without retraining the base model, and analyses highlight remaining failures on extremely small or inherently ambiguous evidence.

## References

1. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)  
2. Bar, A., Gandelsman, Y., Darrell, T., Globerson, A., Efros, A.: Visual prompting via image inpainting. Advances in neural information processing systems 35, 25005–25017 (2022)  
3. Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q.V., Ré, C., Mirhoseini, A.: Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787 (2024)  
4. Carter, B., Jain, S., Mueller, J.W., Gifford, D.: Overinterpretation reveals image classification model pathologies. Advances in Neural Information Processing Systems 34, 15395–15407 (2021)  
5. Chen, A., Yao, Y., Chen, P.Y., Zhang, Y., Liu, S.: Understanding and improving visual prompting: A label-mapping perspective. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19133–19143 (2023)  
6. Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al.: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 24185–24198 (2024)  
7. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025)  
8. Dalal, D., Vashishtha, G., Mishra, U., Kim, J., Kanda, M., Ha, H., Lazebnik, S., Ji, H., Jain, U.: Constructive distortion: Improving mllms with attention-guided image warping. arXiv preprint arXiv:2510.09741 (2025)  
9. Geirhos, R., Jacobsen, J.H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., Wichmann, F.A.: Shortcut learning in deep neural networks. Nature Machine Intelligence 2(11), 665–673 (2020)  
10. Guan, T., Liu, F., Wu, X., Xian, R., Li, Z., Liu, X., Wang, X., Chen, L., Huang, F., Yacoob, Y., et al.: Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 14375–14385 (2024)  
11. Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks. In: International conference on machine learning. pp. 1321–1330. PMLR (2017)  
12. Hendrycks, D., Gimpel, K.: A baseline for detecting misclassified and out-ofdistribution examples in neural networks. arXiv preprint arXiv:1610.02136 (2016)  
13. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. Iclr 1(2), 3 (2022)  
14. Huang, Q., Dong, X., Zhang, P., Wang, B., He, C., Wang, J., Lin, D., Zhang, W., Yu, N.: Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13418–13427 (2024)  
15. Hudson, D.A., Manning, C.D.: Gqa: A new dataset for real-world visual reasoning and compositional question answering. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6700–6709 (2019)  
16. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024)  
17. Jian, P., Wu, J., Sun, W., Wang, C., Ren, S., Zhang, J.: Look again, think slowly: Enhancing visual reflection in vision-language models. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 9262–9281 (2025)  
18. Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., et al.: Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221 (2022)  
19. Kuhn, L., Gal, Y., Farquhar, S.: Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. arXiv preprint arXiv:2302.09664 (2023)  
20. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W.X., Wen, J.R.: Evaluating object hallucination in large vision-language models. In: Proceedings of the 2023 conference on empirical methods in natural language processing. pp. 292–305 (2023)  
21. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36, 34892–34916 (2023)  
22. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? In: European conference on computer vision. pp. 216–233. Springer (2024)  
23. Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.W., Galley, M., Gao, J.: Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255 (2023)  
24. Masry, A., Do, X.L., Tan, J.Q., Joty, S., Hoque, E.: Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In: Findings of the association for computational linguistics: ACL 2022. pp. 2263–2279 (2022)  
25. Mathew, M., Karatzas, D., Jawahar, C.: Docvqa: A dataset for vqa on document images. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision. pp. 2200–2209 (2021)  
26. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)  
27. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024)  
28. Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., Yao, S.: Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems 36, 8634–8652 (2023)  
29. Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., Rohrbach, M.: Towards vqa models that can read. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8317–8326 (2019)  
30. Snell, C., Lee, J., Xu, K., Kumar, A.: Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314 (2024)  
31. Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., Xie, S.: Eyes wide shut? exploring the visual shortcomings of multimodal llms. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9568–9578 (2024)  
32. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024)  
33. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., Zhou, D.: Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171 (2022)  
34. Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., Hooi, B.: Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. arXiv preprint arXiv:2306.13063 (2023)  
35. Yang, J., Zhang, H., Li, F., Zou, X., Li, C., Gao, J.: Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441 (2023)  
36. Yang, L., Wang, Y., Li, X., Wang, X., Yang, J.: Fine-grained visual prompting. Advances in Neural Information Processing Systems 36, 24993–25006 (2023)  
37. Yin, B., Hu, X., Zhou, X., Jiang, P.T., Liao, Y., Zhu, J., Zhang, J., Tai, Y., Wang, C., Yan, S.: Fera: Frequency-energy constrained routing for effective diffusion adaptation fine-tuning. arXiv preprint arXiv:2511.17979 (2025)  
38. Yin, B., Li, Q., Yu, R., Wang, X.: Refinement provenance inference: Detecting llmrefined training prompts from model behavior. arXiv preprint arXiv:2601.01966 (2026)  
39. Yin, B., Yang, X., Wang, X.: Don’t forget the nonlinearity: Unlocking activation functions in efficient fine-tuning. arXiv preprint arXiv:2509.13240 (2025)  
40. Yin, S., Fu, C., Zhao, S., Xu, T., Wang, H., Sui, D., Shen, Y., Li, K., Sun, X., Chen, E.: Woodpecker: Hallucination correction for multimodal large language models. Science China Information Sciences 67(12), 220105 (2024)  
41. Yin, Z., Sun, Q., Guo, Q., Wu, J., Qiu, X., Huang, X.J.: Do large language models know what they don’t know? In: Findings of the association for Computational Linguistics: ACL 2023. pp. 8653–8665 (2023)  
42. Yu, R., Yu, W., Wang, X.: Attention prompting on image for large vision-language models. In: European Conference on Computer Vision. pp. 251–268. Springer (2024)  
43. Yu, X., Xu, C., Chen, Z., Yin, B., Yang, C., He, Y., Hu, Y., Zhang, J., Tan, C., Hu, X., et al.: Dual latent memory for visual multi-agent system. arXiv preprint arXiv:2602.00471 (2026)  
44. Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al.: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9556–9567 (2024)  
45. Zhang, J., Khayatkhoei, M., Chhikara, P., Ilievski, F.: Visual cropping improves zero-shot question answering of multimodal large language models. In: R0-FoMo: Robustness of Few-shot and Zero-shot Learning in Large Foundation Models (2023)  
46. Zhang, Z., Zhang, A., Li, M., Zhao, H., Karypis, G., Smola, A.: Multimodal chainof-thought reasoning in language models. arXiv preprint arXiv:2302.00923 (2023)

## Appendix

Overall, the appendix provides complementary support for SPOT-E from four aspects. First, the theoretical discussion clarifies why the proposed objective favors non-destructive interventions and how its design relates to the observed efficiency trade-offs. Second, SPOT-E remains effective on larger open-source backbones, as shown in Tables 7 and 8. Third, the method is stable across repeated runs, decoding choices, and moderate hyperparameter changes, while additional ablations show that the gains do not rely on overly large eye modules or trainable budgets (Tables 9, 10, and 11-17). Finally, the added test-time cost brings practical returns in robustness and confidence behavior, as summarized in Tables 18, 19, 20, and 21.

## A Additional Theoretical Discussion

Reward preference for non-destructive interventions. Recall that SPOT-E favors interventions that both reduce answer uncertainty and preserve lowentropy anchors from the baseline trajectory. A simplified form of the reward can be written as

$$
R (\tilde {x}) = - \gamma (x, q) \Delta H _ {\mathrm{ans}} (\tilde {x}) - \lambda \Delta H _ {\mathrm{low}} (\tilde {x}), \tag {22}
$$

where $\Delta H _ { \mathrm { a n s } } ( \tilde { x } )$ denotes the change in answer entropy relative to the baseline and $\varDelta H _ { \mathrm { l o w } } ( \tilde { x } )$ measures anchor disruption. Here, lower $\varDelta H _ { \mathrm { a n s } }$ is better when it reflects a more decisive answer, while lower $\varDelta H _ { \mathrm { l o w } }$ indicates less damage to already stable parts of the decoding trajectory.

Proposition 1. Let ${ \tilde { x } } _ { A }$ and $\tilde { x } _ { B }$ be two candidate interventions. Then $R ( \tilde { x } _ { A } ) >$ $R ( \tilde { x } _ { B } )$ if and only if

$$
\gamma (x, q) \Big (\Delta H _ {\mathrm{ans}} (\tilde {x} _ {B}) - \Delta H _ {\mathrm{ans}} (\tilde {x} _ {A}) \Big) > \lambda \Big (\Delta H _ {\mathrm{low}} (\tilde {x} _ {A}) - \Delta H _ {\mathrm{low}} (\tilde {x} _ {B}) \Big). \tag {23}
$$

Proof. Subtracting $R ( \tilde { x } _ { B } )$ from $R ( \tilde { x } _ { A } )$ under Eq. (22) gives

$$
R (\tilde {x} _ {A}) - R (\tilde {x} _ {B}) = - \gamma \Bigl (\Delta H _ {\mathrm{ans}} (\tilde {x} _ {A}) - \Delta H _ {\mathrm{ans}} (\tilde {x} _ {B}) \Bigr) - \lambda \Bigl (\Delta H _ {\mathrm{low}} (\tilde {x} _ {A}) - \Delta H _ {\mathrm{low}} (\tilde {x} _ {B}) \Bigr),
$$

and rearranging yields Eq. (23).

This condition makes the intended behavior explicit: an intervention is preferred when its gain in answer clarification outweighs its additional anchor disruption. In particular, if two interventions achieve the same answer-entropy reduction, the reward always prefers the one with smaller anchor disruption. Conversely, if they preserve anchors equally well, the reward prefers the one that reduces answer entropy more.

Role of dynamic scaling. The scaling factor $\gamma ( x , q )$ controls how strongly the reward emphasizes answer clarification for a given instance. When the baseline prediction is already confident, overly aggressive entropy reduction may encourage destructive shortcuts. By increasing the clarity weight mainly on uncertain instances, dynamic scaling makes the reward more conservative on already stable samples while remaining sufficiently corrective on ambiguous ones. This helps explain why the full reward improves both answer confidence calibration and anchor preservation in Tables 20 and 21.

Complexity discussion. Let $T _ { \mathrm { u p d } }$ denote the number of test-time update steps and N the candidate group size per step. If $C _ { \mathrm { e y e } }$ is the cost of one forward pass through the eye module and $C _ { \mathrm { v l m } }$ the cost of evaluating one intervened candidate with the frozen VLM, then the additional per-instance cost of SPOT-E is approximately

$$
\mathcal {O} (T _ {\mathrm{upd}} \cdot N \cdot (C _ {\mathrm{eye}} + C _ {\mathrm{vlm}}))  , \tag {24}
$$

excluding the final decode. Since the base VLM remains frozen and only the spotlight-module LoRA parameters are updated, the trainable memory footprint scales with the adaptation module rather than the full backbone. This is consistent with the empirical runtime and memory results in Tables 17 and 18.

## B Additional Experimental Details

Benchmark protocols. We summarize the benchmark splits and final answer formats in Table 5, so that the appendix makes clear how evaluation is organized across tasks.

Table 5: Benchmark protocols used in the appendix and the main paper.

<table><tr><td>Benchmark</td><td>Split</td><td>Final Answer Format</td></tr><tr><td>TextVQA</td><td>val / test-dev</td><td>short free-form</td></tr><tr><td>DocVQA</td><td>val / test</td><td>short free-form</td></tr><tr><td>ChartQA</td><td>test</td><td>number / phrase</td></tr><tr><td>MathVista</td><td>testmini / test</td><td>option / phrase</td></tr><tr><td>MMMU</td><td>val</td><td>option only</td></tr><tr><td>GQA</td><td>test-dev</td><td>short free-form</td></tr><tr><td>MMBench</td><td>dev / test</td><td>option only</td></tr><tr><td>POPE</td><td>random / popular / adv</td><td>yes / no</td></tr></table>

Runtime environment. We also summarize the hardware and software environment in Table 6, since the practical cost of test-time adaptation is part of the method’s trade-off.

Table 6: Hardware and software setup for the main open-source runs.

<table><tr><td>Item</td><td>Setting</td></tr><tr><td>GPU</td><td>H100 80GB</td></tr><tr><td>Framework</td><td>PyTorch + Transformers</td></tr><tr><td>Precision</td><td>bf16</td></tr><tr><td>Inference batch size</td><td>1</td></tr><tr><td>Backbone policy</td><td>fully frozen</td></tr><tr><td>Trainable component</td><td>spotlight-module LoRA only</td></tr><tr><td>API models</td><td>evaluated remotely with matched decoding</td></tr></table>

Table 7: Additional results on larger open-source backbones.

<table><tr><td>Base Model</td><td>TextVQA</td><td>DocVQA</td><td>ChartQA</td><td>MathVista</td><td>MMMU</td><td>GQA</td><td>MMBench</td><td>POPE</td></tr><tr><td>Qwen2.5-VL-32B</td><td>88.3</td><td>89.1</td><td>89.8</td><td>73.8</td><td>61.4</td><td>67.1</td><td>84.8</td><td>88.3</td></tr><tr><td>+ SPOT-E (Ours)</td><td>89.5 +1.2↑</td><td>89.8 +0.7↑</td><td>90.6 +0.8↑</td><td>75.3 +1.5↑</td><td>63.0 +1.6↑</td><td>67.8 +0.7↑</td><td>85.5 +0.7↑</td><td>89.0 +0.7↑</td></tr><tr><td>Qwen3-VL-32B</td><td>89.0</td><td>89.8</td><td>90.2</td><td>75.1</td><td>63.2</td><td>67.9</td><td>85.5</td><td>88.9</td></tr><tr><td>+ SPOT-E (Ours)</td><td>90.0 +1.0↑</td><td>90.4 +0.6↑</td><td>90.9 +0.7↑</td><td>76.4 +1.3↑</td><td>64.6 +1.4↑</td><td>68.5 +0.6↑</td><td>86.2 +0.7↑</td><td>89.5 +0.6↑</td></tr><tr><td>InternVL2.5-26B</td><td>84.6</td><td>85.0</td><td>87.5</td><td>72.5</td><td>60.8</td><td>63.8</td><td>82.5</td><td>90.8</td></tr><tr><td>+ SPOT-E (Ours)</td><td>86.1 +1.5↑</td><td>86.0 +1.0↑</td><td>88.8 +1.3↑</td><td>74.3 +1.8↑</td><td>62.7 +1.9↑</td><td>65.0 +1.2↑</td><td>83.4 +0.9↑</td><td>91.6 +0.8↑</td></tr><tr><td>LLaVA-OV-72B</td><td>86.4</td><td>84.9</td><td>85.6</td><td>58.0</td><td>49.8</td><td>66.5</td><td>79.5</td><td>87.5</td></tr><tr><td>+ SPOT-E (Ours)</td><td>88.0 +1.6↑</td><td>86.0 +1.1↑</td><td>87.1 +1.5↑</td><td>60.3 +2.3↑</td><td>51.8 +2.0↑</td><td>67.6 +1.1↑</td><td>80.6 +1.1↑</td><td>88.2 +0.7↑</td></tr></table>

## C Additional Quantitative Results

Larger backbones. The main paper already shows consistent gains across multiple backbone families. Table 7 extends that coverage to larger variants and shows that the effect persists even when the base model is stronger.

Task-wise average gains. To make the broader trend easier to read, Table 8 further aggregates improvements by task family. The largest average gains still concentrate on evidence-intensive benchmarks, consistent with the main claim.

## D Stability and Decoding Robustness

Repeated runs. Because SPOT-E contains candidate sampling and test-time updates, repeated-run consistency is useful to report explicitly. Table 9 summarizes the mean and standard deviation across three random seeds on representative benchmarks.

Decoding robustness. The main experiments use a matched decoding setup. To show that the gain does not depend on one particular decoding choice, Table 10 additionally compares greedy decoding, low-temperature sampling, and a small Best-of-4 setting.

Table 8: Average gain of SPOT-E by task category across open-source backbones.

<table><tr><td>Task Category</td><td>Benchmarks</td><td>Avg. Gain</td></tr><tr><td>Evidence-intensive</td><td>TextVQA, DocVQA, ChartQA, MathVista</td><td>+2.2</td></tr><tr><td>Broader reasoning</td><td>MMMU, GQA, MMBench</td><td>+1.5</td></tr><tr><td>Hallucination-focused</td><td>POPE</td><td>+1.0</td></tr></table>

Table 9: Repeated-run stability over three seeds.

<table><tr><td>Backbone</td><td>Method</td><td>TextVQA</td><td>MathVista</td><td>MMMU</td><td>POPE</td></tr><tr><td rowspan="2">Qwen2.5-VL-7B</td><td>Frozen</td><td>84.9 ± 0.0</td><td>67.8 ± 0.0</td><td>55.0 ± 0.0</td><td>86.4 ± 0.0</td></tr><tr><td>+ SPOT-E</td><td>86.8 ± 0.2</td><td>70.7 ± 0.2</td><td>58.4 ± 0.3</td><td>87.4 ± 0.1</td></tr><tr><td rowspan="2">InternVL2.5-8B</td><td>Frozen</td><td>81.0 ± 0.0</td><td>66.0 ± 0.0</td><td>56.0 ± 0.0</td><td>89.0 ± 0.0</td></tr><tr><td>+ SPOT-E</td><td>84.4 ± 0.3</td><td>69.4 ± 0.2</td><td>59.3 ± 0.2</td><td>89.9 ± 0.1</td></tr><tr><td rowspan="2">LLaVA-NeXT-7B</td><td>Frozen</td><td>78.5 ± 0.0</td><td>47.0 ± 0.0</td><td>38.0 ± 0.0</td><td>85.0 ± 0.0</td></tr><tr><td>+ SPOT-E</td><td>84.6 ± 0.4</td><td>50.4 ± 0.3</td><td>40.9 ± 0.2</td><td>86.5 ± 0.1</td></tr></table>

## E Additional Ablation Studies

Eye module scale. To study whether SPOT-E depends on the capacity of the external eye module, we vary the CLIP backbone used to parameterize the spotlight policy while keeping the frozen VLM, reward, and test-time budget fixed. Table 11 shows that larger eye modules generally improve performance, but the gains saturate relative to the added cost, supporting our default choice as a favorable efficiency–accuracy trade-off.

Trainable budget. We further vary the LoRA rank of the spotlight module to measure how much test-time adaptation capacity is actually needed. Table 12 shows that a small rank already captures most of the gains, while larger ranks bring only marginal improvements at higher cost.

Anchor-related choices. The main paper already studies the reward design and the spotlight design. Here we further unpack the anchor-related hyperparameters in Tables 13 and 14, since they are central to the entropy-shaping objective.

Optimization settings. We next vary the GRPO group size and learning rate in Tables 15 and 16 to verify that the reported gains are not tied to a single narrow optimization choice.

Budget trade-off in table form. The main paper shows the test-time budget trend as a figure. For the appendix, Table 17 is often more convenient because it combines the gain and the runtime overhead in one place.

Table 10: Robustness to decoding choices on Qwen2.5-VL-7B.

<table><tr><td>Method</td><td>Decoding</td><td>TextVQA</td><td>MathVista</td><td>MMMU</td></tr><tr><td>Frozen</td><td>Greedy</td><td>84.9</td><td>67.8</td><td>55.0</td></tr><tr><td>+ SPOT-E</td><td>Greedy</td><td>86.9</td><td>70.8</td><td>58.5</td></tr><tr><td>Frozen</td><td>Temp=0.2</td><td>84.6</td><td>67.3</td><td>54.6</td></tr><tr><td>+ SPOT-E</td><td>Temp=0.2</td><td>86.5</td><td>70.2</td><td>58.0</td></tr><tr><td>Frozen</td><td>Best-of-4</td><td>85.4</td><td>68.2</td><td>55.8</td></tr><tr><td>+ SPOT-E</td><td>Best-of-4</td><td>87.3</td><td>71.1</td><td>58.9</td></tr></table>

Table 11: Effect of eye-module scale on Qwen2.5-VL-7B.

<table><tr><td>Eye Module</td><td>Params</td><td>TextVQA</td><td>MathVista</td><td>MMMU</td><td>Runtime (s)</td></tr><tr><td>CLIP ViT-B/16</td><td>86M</td><td>86.9</td><td>70.8</td><td>58.5</td><td>2.08</td></tr><tr><td>CLIP ViT-L/14</td><td>304M</td><td>87.3</td><td>71.2</td><td>59.5</td><td>2.41</td></tr><tr><td>SigLIP So400m</td><td>400M</td><td>87.4</td><td>71.4</td><td>60.2</td><td>2.73</td></tr></table>

## F Efficiency and Robustness

Parameter and memory overhead. Since SPOT-E updates only the spotlightmodule LoRA at test time, the trainable fraction is small. Table 18 makes that explicit together with the extra memory footprint.

Severity-averaged corruption robustness. The main paper presents robustness trends under increasing corruption severity. Table 19 summarizes the same phenomenon by averaging over severity levels, which gives a compact cross-model view.

## G Entropy and Anchor Diagnostics

Confidence behavior. The main paper shows that SPOT-E increases answer entropy on unsupported errors while keeping it low on correct cases. Table 20 summarizes that separation numerically.

Anchor preservation. To complement the reward ablation in the main paper, Table 21 reports the average anchor disruption $\varDelta H _ { \mathrm { l o w } }$ for several intervention variants. This makes the non-destructive effect of the full objective more explicit.

Table 12: Effect of LoRA rank on Qwen2.5-VL-7B.

<table><tr><td>Rank</td><td>Params</td><td>TextVQA</td><td>MathVista</td><td>MMMU</td><td>Runtime (s)</td></tr><tr><td>4</td><td>1.7M</td><td>86.1</td><td>69.9</td><td>57.8</td><td>2.31</td></tr><tr><td>8</td><td>3.4M</td><td>86.6</td><td>70.4</td><td>58.2</td><td>2.36</td></tr><tr><td>16</td><td>6.8M</td><td>86.9</td><td>70.8</td><td>58.5</td><td>2.41</td></tr><tr><td>32</td><td>13.6M</td><td>87.0</td><td>70.9</td><td>58.6</td><td>2.55</td></tr></table>

Table 13: Sensitivity to the number of low-entropy anchors K on Qwen2.5-VL-7B.

<table><tr><td>K</td><td>TextVQA</td><td>MathVista</td><td>POPE</td></tr><tr><td>20</td><td>86.1</td><td>69.9</td><td>87.1</td></tr><tr><td>40</td><td>86.6</td><td>70.4</td><td>87.3</td></tr><tr><td>60</td><td>86.9</td><td>70.8</td><td>87.4</td></tr><tr><td>80</td><td>86.7</td><td>70.5</td><td>87.3</td></tr><tr><td>120</td><td>86.3</td><td>70.0</td><td>87.1</td></tr></table>

Table 14: Sensitivity to the anchor-preservation weight λ on Qwen2.5-VL-7B.

<table><tr><td>λ</td><td>TextVQA</td><td>MathVista</td><td>POPE</td></tr><tr><td>0.0</td><td>85.8</td><td>69.4</td><td>86.9</td></tr><tr><td>0.1</td><td>86.2</td><td>70.0</td><td>87.1</td></tr><tr><td>0.3</td><td>86.6</td><td>70.5</td><td>87.3</td></tr><tr><td>0.5</td><td>86.9</td><td>70.8</td><td>87.4</td></tr><tr><td>0.7</td><td>86.7</td><td>70.6</td><td>87.2</td></tr><tr><td>1.0</td><td>86.1</td><td>70.0</td><td>87.0</td></tr></table>

Table 15: Sensitivity to GRPO group size N on Qwen2.5-VL-7B.

<table><tr><td>Group Size</td><td>TextVQA</td><td>MathVista</td><td>MMMU</td></tr><tr><td>2</td><td>86.2</td><td>70.0</td><td>57.6</td></tr><tr><td>4</td><td>86.9</td><td>70.8</td><td>58.5</td></tr><tr><td>8</td><td>87.0</td><td>70.9</td><td>58.6</td></tr><tr><td>16</td><td>87.0</td><td>71.0</td><td>58.7</td></tr></table>

Table 16: Sensitivity to the learning rate on Qwen2.5-VL-7B.

<table><tr><td>Learning Rate</td><td>TextVQA</td><td>MathVista</td><td>MMMU</td></tr><tr><td> $1 \times 10^{-4}$ </td><td>86.1</td><td>69.9</td><td>57.9</td></tr><tr><td> $3 \times 10^{-4}$ </td><td>86.6</td><td>70.5</td><td>58.3</td></tr><tr><td> $5 \times 10^{-4}$ </td><td>86.9</td><td>70.8</td><td>58.5</td></tr><tr><td> $1 \times 10^{-3}$ </td><td>86.5</td><td>70.2</td><td>58.1</td></tr></table>

Table 17: Accuracy and runtime as a function of the test-time update budget. Runtime values are average seconds per sample.

<table><tr><td>Model</td><td>Steps</td><td>TextVQA</td><td>MathVista</td><td>MMMU</td><td>Runtime / sample (s)</td></tr><tr><td rowspan="6">Qwen2.5-VL-7B</td><td>0</td><td>84.9</td><td>67.8</td><td>55.0</td><td>0.73</td></tr><tr><td>1</td><td>85.8</td><td>69.2</td><td>56.9</td><td>0.98</td></tr><tr><td>2</td><td>86.3</td><td>69.9</td><td>57.6</td><td>1.26</td></tr><tr><td>4</td><td>86.6</td><td>70.4</td><td>58.2</td><td>1.64</td></tr><tr><td>8</td><td>86.9</td><td>70.8</td><td>58.5</td><td>2.41</td></tr><tr><td>16</td><td>87.0</td><td>70.9</td><td>58.6</td><td>3.96</td></tr><tr><td rowspan="6">InternVL2.5-8B</td><td>0</td><td>81.0</td><td>66.0</td><td>56.0</td><td>0.89</td></tr><tr><td>1</td><td>82.6</td><td>67.3</td><td>57.4</td><td>1.17</td></tr><tr><td>2</td><td>83.5</td><td>68.1</td><td>58.2</td><td>1.47</td></tr><tr><td>4</td><td>84.0</td><td>68.9</td><td>58.8</td><td>1.91</td></tr><tr><td>8</td><td>84.5</td><td>69.5</td><td>59.5</td><td>2.85</td></tr><tr><td>16</td><td>84.6</td><td>69.6</td><td>59.6</td><td>4.71</td></tr></table>

Table 18: Trainable parameters and memory overhead of SPOT-E.

<table><tr><td>Backbone</td><td>Trainable Params</td><td>Trainable %</td><td>Extra VRAM</td></tr><tr><td>Qwen2.5-VL-7B</td><td>6.8M</td><td>0.097%</td><td>+1.8 GB</td></tr><tr><td>InternVL2.5-8B</td><td>6.8M</td><td>0.085%</td><td>+2.0 GB</td></tr><tr><td>LLaVA-NeXT-7B</td><td>6.8M</td><td>0.097%</td><td>+1.7 GB</td></tr><tr><td>Qwen2.5-VL-32B</td><td>6.8M</td><td>0.021%</td><td>+2.4 GB</td></tr></table>

Table 19: Severity-averaged accuracy under controlled corruptions on TextVQA.

<table><tr><td>Backbone</td><td>Method</td><td>Gaussian Noise</td><td>Low-Res</td><td>Occlusion</td></tr><tr><td rowspan="2">Qwen2.5-VL-7B</td><td>Frozen</td><td>73.1</td><td>70.5</td><td>72.4</td></tr><tr><td>+ SPOT-E</td><td>83.4</td><td>81.9</td><td>83.2</td></tr><tr><td rowspan="2">InternVL2.5-8B</td><td>Frozen</td><td>71.6</td><td>69.2</td><td>71.1</td></tr><tr><td>+ SPOT-E</td><td>80.2</td><td>78.8</td><td>79.7</td></tr><tr><td rowspan="2">LLaVA-NeXT-7B</td><td>Frozen</td><td>66.0</td><td>63.8</td><td>65.4</td></tr><tr><td>+ SPOT-E</td><td>75.4</td><td>73.2</td><td>74.8</td></tr></table>

Table 20: Average answer entropy $H _ { \mathrm { a n s } }$ on correct and incorrect predictions for TextVQA with Qwen2.5-VL-7B. Lower is better for correct cases, while higher is better for unsupported wrong cases.

<table><tr><td>Method</td><td>Correct ↓</td><td>Wrong ↑</td></tr><tr><td>Frozen</td><td>0.12</td><td>0.24</td></tr><tr><td>+ SPOT-E</td><td>0.11</td><td>0.41</td></tr></table>

Table 21: Average anchor disruption $\varDelta H _ { \mathrm { l o w } }$ for different variants on Qwen2.5-VL-7B. Lower is better.

<table><tr><td>Variant</td><td>TextVQA</td><td>MathVista</td><td>POPE</td></tr><tr><td>Clarity only</td><td>0.083</td><td>0.091</td><td>0.077</td></tr><tr><td>w/o dynamic scaling</td><td>0.071</td><td>0.086</td><td>0.068</td></tr><tr><td>MeanFuse</td><td>0.062</td><td>0.074</td><td>0.060</td></tr><tr><td>Full SPOT-E</td><td>0.041</td><td>0.052</td><td>0.044</td></tr></table>