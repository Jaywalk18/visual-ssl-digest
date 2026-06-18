# See First, Answer Later: Visual Evidence Pre-Alignment via Sufficiency-Driven RL

Yilian Liu1\* Sicong Leng2\* Guoshun Nan1 Junyi Zhu1 Jiayu Huang1 Minghao Sun1 Xuancheng Zhu1 Yisong Chen3 Zexian Wei1 Xiaofeng Tao1

1Beijing University of Posts and Telecommunications, China

2Nanyang Technological University, Singapore; 3China Telecom, China

{liuyilian,nanguo2021}@bupt.edu.cn; Lengsicong@gmail.com

## Abstract

Multimodal large language models (MLLMs) integrate strong text reasoning with visual inputs, yet their responses can be inconsistent with the underlying images, indicating ineffective utilization of visual evidence during inference. The prevailing training paradigm relies on large-scale caption-based pretraining for general alignment, followed by supervised fine-tuning and reinforcement learning to enable instruction following and complex reasoning. However, such pretraining provides only weak visual grounding: short, coarse captions bias models toward salient objects while neglecting fine-grained visual evidence. In this paper, we introduce Visual Evidence Pre-Alignment (VEPA), an intermediate stage between pretraining and post-training that explores a novel sufficiency-driven objective with Group Relative Policy Optimization (GRPO) to optimize question-conditioned visual evidence descriptions. Extensive experiments across diverse benchmarks show that our VEPA consistently enhances performance on visually demanding evaluations and complements standard supervised post-training. Further analyses show that the income stems from strengthened, transferable visual grounding, rather than from additional task-specific training.

## 1. Introduction

Multimodal large language models (MLLMs) have recently achieved strong performance on diverse tasks such as document and chart comprehension and diagrammatic math reasoning. Despite these advances, MLLMs may produce outputs that are weakly grounded in the underlying image (Luo et al., 2025; Xia et al., 2025), including omissions of critical visual details, incorrect attribution of attributes or relations, and hallucinated content (Li et al., 2025; Xia et al., 2025). Such failures are not confined to specific tasks or domains but instead reflect a pervasive limitation in how visual evidence is exploited during inference. Most existing MLLM training pipelines follow a two-stage paradigm (Liu et al., 2023; Zhu et al., 2024). In the pretraining stage, models are trained on large-scale image–caption corpora to establish coarse vision–language alignment (Li et al., 2020). In the post-training stage, supervised fine-tuning, often combined with reinforcement learning, is applied to improve instruction following and downstream task performance (Christiano et al., 2017; Zhai et al., 2024). However, caption-driven pretraining alone is insufficient to ensure robust visual perception prior to post-training. Captions are typically short and coarse, emphasizing salient objects (Lin et al., 2014) or global scene descriptions while omitting many detailed attributes, relations, and less prominent regions. This supervision biases models toward a narrow subset of visual content and provides limited incentive to encode fine-grained or question-relevant visual evidence, leading models to rely disproportionately on language priors during inference (Goyal et al., 2017).

![](images/26508b3a912e59ce6435504c6e74bd48e75003ce3cfb255f96b0bc3cd5ac9947.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Stage1 Pretrain"] --> B["Word Modeling"]
  B --> C["Course Alignment"]
  C --> D["Q: Describe the image.<br>A: A man and a dog..."]
  E["Stage2 Post-train"] --> F["Instruction Following"]
  F --> G["Task Reasoning"]
  G --> H["Q: Find the value.<br>A: <think>/<answer>"]
  I["Standard Recipe Test"] --> J["Ignore Details"]
  J --> K["Q: How many pins?<br>A: 8 push pins visible"]
  L["Can we teach models to see better before learning to answer?"] --> M["Stage 1.5: VEPA"]
  M --> N["Question-conditioned caption"]
  M --> O["Cap1 Q1"]
  M --> P["Cap2 Q2"]
  M --> Q["Cap3 Q3"]
  M --> R["sufficiency-driven GRPO"]
  S["VEPA Added Test"] --> T["Perception Activated"]
  T --> U["Q: How many pins?<br>A: 6 push pins visible, attached to the bulge"]
```
</details>

Figure 1. Motivation and overview. The standard two-stage recipe often yields coarse alignment and encourages shortcut answering that ignores visual details. We insert VEPA as an intermediate stage that trains the model to produce question-conditioned visual evidence using sufficiency-driven GRPO, with a frozen blind reader (an LLM) answering and verifying whether the evidence suffices to recover the answer. This “see first, answer later” pre-alignment activates perceptual ability and improves downstream visual grounding.

A straightforward remedy is to strengthen pretraining supervision by replacing short captions with dense descriptions (Zheng et al., 2024; Zeng et al., 2025) or augmenting training with perception-oriented datasets (e.g., OCR) that explicitly encode fine-grained visual content. However, this approach is limited both practically and fundamentally. High-quality dense captions and OCR (Chen et al., 2024c) annotations are costly (Dong et al., 2025; Liu et al., 2025; Shen et al., 2025) to collect at scale for diverse web images, and manual annotation pipelines introduce additional biases, omissions, and inconsistencies (Misra et al., 2016; Hu et al., 2023). More fundamentally, the high information density of visual inputs compared to textual representations makes it inherently difficult for any finite description to faithfully encode all objects, relations, and spatial details in a scene. Thus, even dense captions remain a lossy (Dubois et al., 2021) and biased proxy for visual content (Chen et al., 2024a), and simply scaling static image–text pairs is insufficient to achieve the level of perceptual precision required by complex multimodal reasoning tasks.

In this work, we introduce Visual Evidence Pre-Alignment (VEPA), an intermediate training stage that teaches MLLMs to generate question-conditioned visual evidence, textual descriptions that captures the image details needed to solve a given question. As illustrated in Figure 1, VEPA optimizes evidence generation via sufficiency-driven Group Relative Policy Optimization (GRPO) (Shao et al., 2024). During training, a frozen blind reader (an LLM), conditioned only on the question and the generated evidence, serves as an auxiliary evaluator by verifying whether the evidence is sufficient to recover the ground-truth answer. We carefully design the reward to discourage answer leakage and degenerate repetitive evidence, thereby decoupling visual grounding from answer generation. Across diverse evaluation settings, VEPA consistently outperforms the standard training recipe on benchmarks spanning knowledge-intensive and compositional VQA, fine-grained perception, and holistic multimodal evaluation. Ablation studies and qualitative analyses show that these gains arise from strengthened visual perception rather than additional task-level supervision, and that the learned grounding transfers robustly to out-ofdistribution data.

In summary, our contributions are in three aspects:

• We propose Visual Evidence Pre-Alignment (VEPA), a novel intermediate training stage between pretraining and post-training that explicitly strengthens visual perception prior to task-level instruction tuning, encouraging MLLMs to attend to and encode relevant visual evidence before answering.  
• We instantiate VEPA with a GRPO-based reinforcement learning framework that trains models to generate question-conditioned visual evidence. A frozen blind reader evaluates whether the generated evidence suffices to recover the ground-truth answer, decoupling visual grounding from answer generation and enabling training with existing QA data without additional annotation cost.  
• Extensive experiments across diverse benchmarks and model backbones demonstrate consistent improvements on visually demanding evaluations. Ablation studies and qualitative analyses further show that VEPA strengthens visual grounding and induces transferable perceptual capabilities that generalize to out-ofdistribution settings.

## 2. Related Work

Multimodal pretraining and post-training. MLLMs follow a two-stage recipe: large-scale vision–language pretraining for coarse alignment, followed by instruction tuning to improve task-level behaviors (Alayrac et al., 2022; Li et al., 2023a; Liu et al., 2023; 2024; Wang et al., 2024). This paradigm has produced strong systems across late-fusion architectures that connect a frozen or lightly-tuned vision encoder to an LLM as well as increasingly capable end-to-end or early-fusion variants. Recent backbones such as Qwen2- VL (Wang et al., 2024) further improve resolution handling and general visual understanding, strengthening the foundation for downstream multimodal adaptation (Wang et al., 2024). However, empirical evaluations consistently suggest that coarse caption-driven supervision alone does not always yield reliable fine-grained grounding required by complex VQA-style queries (Fu et al., 2025).

Visual grounding. A recurring challenge for MLLMs is the tendency to under-use visual input, leading to omissions or hallucinated content (Li et al., 2025; Xia et al., 2025) when language priors (Goyal et al., 2017) dominate generation. This issue has motivated dedicated benchmarks and diagnostics that quantify visual faithfulness and robustness beyond raw accuracy, including object-hallucination probes and broad-spectrum evaluation suites (Li et al., 2023b; Rohrbach et al., 2018). Complementary lines of work mitigate hallucinations via improved post-training objectives (Zheng et al.,

2025), decoding-time constraints (Li et al., 2025), or auxiliary verification mechanisms that encourage consistency with perceived evidence (Zheng et al., 2025). Despite these advances, many approaches still optimize answer generation directly (Wang et al., 2025), making it difficult to isolate and strengthen the upstream visual extraction process that should support downstream reasoning.

Intermediate evidence and verifier-based reinforcement learning. Generating intermediate representations—such as rationales, descriptions, or other textual evidence—has been explored to improve grounding and interpretability in multimodal reasoning, where models must surface queryrelevant visual details (Li et al., 2018; Rao et al., 2021). When reliable intermediate supervision is unavailable, recent language-model post-training leverages verifier-style or preference-based signals as scalable supervision without token-level labels (Rafailov et al., 2023; Wen et al., 2025). In particular, group-based policy optimization such as GRPO removes an explicit critic and estimates baselines from within-group scores, simplifying RL fine-tuning for long responses (Shao et al., 2024). Our work aligns with this verifier-based direction but targets a different object. Rather than optimizing the final answer directly, we train a question-conditioned evidence channel to be sufficient for solving and resistant to shortcut behaviors, bridging coarse pretraining and downstream post-training.

## 3. Preliminary

## 3.1. Evidence Decomposition for Visual Grounding

The goal of an MLLM is to generate an answer a conditioned on an image v and a question q, ideally approximating $P ( a \mid v , q )$ . In practice, caption-based pretraining provides only coarse vision–language alignment, and answer-level supervision makes improvements in perception mostly a byproduct of reasoning learning. As a result, models may over-rely on language priors, drifting toward $P ( a \mid q )$ and under-utilizing the visual input v.

To address this, we decouple visual perception from reasoning by introducing a latent variable e, termed visual evidence, which serves as an information bottleneck between perception and answer generation. The generation process is formalized as:

$$
P (a \mid v, q) = \sum_ {e \in \mathcal {E}} P (a \mid e, q) \cdot P (e \mid v, q). \tag {1}
$$

Here, $P ( e \mid v , q )$ acts as a visual representation policy (MLLM) that maps the image and question to a textual evidence $e ,$ and $\textstyle P ( a \mid e , q )$ is a reasoning policy (LLM) that produces the answer from the evidence and the question.

Under this decomposition, improving visual alignment amounts to shaping $P ( e \mid v , q )$ such that the induced evidence is both informative for answering and genuinely grounded in the image. We therefore require the evidence e to satisfy two properties:

Sufficiency. Conditioned on the question, the evidence should contain enough information to determine the answer:

$$
P (a \mid v, q) \approx P (a \mid e, q) \quad \text { for } e \sim P (e \mid v, q). \tag {2}
$$

Once e is known, the residual contribution of the raw image v to predicting a should be small.

Visual dependence. The evidence must depend on both the image and the question, rather than being reconstructible from either alone. In particular, for image–question pairs $( v _ { 1 } , q )$ and $( v _ { 2 } , q )$ , or $( v , q _ { 1 } )$ and $( v , q _ { 2 } )$ , that induce different answers, we require:

$$
P (e \mid v _ {1}, q) \neq P (e \mid v _ {2}, q), P (e \mid v, q _ {1}) \neq P (e \mid v, q _ {2}). \tag {3}
$$

When the above properties hold approximately, the model cannot satisfy the objective by ignoring the image. Instead, it must route visual information through the evidence channel $P ( e \mid v , q )$ before producing an answer. The intermediate training stage we introduce is designed to explicitly encourage these properties.

## 3.2. Perception Coverage from Data Diversity

Let D denote the training distribution over triplets $( v , q , a )$ . While the sufficiency and dependence conditions above are defined for individual instances, our objective is to enforce them approximately on average over D.

Concretely, we would like the evidence policy $P ( e \mid v , q )$ to satisfy:

$$
\mathbb {E} _ {(v, q, a) \sim \mathcal {D}} \left[ \mathrm{KL} (P (a \mid v, q) \| P (a \mid e, q)) \right] \leq \varepsilon_ {\text { suff }}, \tag {4}
$$

for a small $\varepsilon _ { \mathrm { s u f f } } \geq 0 .$ , ensuring that answers can be predicted nearly as well from $( e , q )$ as from $( v , q )$ on average. In addition, to prevent degeneracy, the evidence must retain non-trivial information about both the image and the question, for example,

$$
I _ {\mathcal {D}} (e; v \mid q) \geq \delta_ {v}, \quad I _ {\mathcal {D}} (e; q \mid v) \geq \delta_ {q}, \tag {5}
$$

with $\delta _ { v } , \delta _ { q } \ > \ 0 .$ , where $I _ { \mathcal { D } } ( \cdot ; \cdot \mid \cdot )$ denotes conditional mutual information under the joint distribution induced by D and $P ( e \mid v , q )$ .

Rather than supervising e with increasingly dense captions, which remain fundamentally constrained by the information density of text, we leverage the diversity inherent in VQA-style data. When triplets $( v , q , a )$ are sampled from a heterogeneous corpus, different questions probe distinct objects, attributes, relations, and regions within the same image, including details rarely emphasized in caption-based supervision. Encouraging sufficiency and dependence over D therefore pushes the evidence policy to cover a broad range of task-relevant visual details. Each evidence instance need only be sufficient for its own question, while the collection of question–evidence pairs across the dataset provides complementary and diverse textual views of visual content.

## 4. Methodology

## 4.1. Evidence Generation as Perceptual Pre-Alignment

We propose Visual Evidence Pre-Alignment (VEPA), an intermediate training stage that explicitly aligns visual perception before task-level post-training. As illustrated in Figure 2, VEPA trains an MLLM to generate questionconditioned visual evidence, textual descriptions that encode the visual information for a given question, without directly optimizing answer generation. This section first formalizes the evidence-generation objective (§4.1), then presents the VEPA optimization framework (§4.2), and finally describes the blind-reader-based reward design (§4.3).

We instantiate the evidence decomposition introduced in §3.1 as an intermediate training objective. Since groundtruth visual evidence annotations are unavailable at scale and dense captions remain both costly and incomplete, supervised learning over evidence tokens is infeasible. So we cast evidence generation as a policy optimization problem.

Let $\mathcal { D } = \{ ( v _ { i } , q _ { i } , a _ { i } ) \}$ denote a VQA training set. During VEPA, the MLLM is not trained to produce the final answer a. Instead, for each triplet $( v , q , a )$ , it generates an intermediate textual sequence e intended to encode the visual information necessary to answer q. For each rollout, the policy samples a single evidence sequence up to a fixed length. Learning is driven solely by a sequence-level reward computed after the evidence is fully generated.

## 4.2. VEPA Optimization Framework

The proposed VEPA optimizes the evidence policy πθ(e | v, q) using reinforcement learning with a sequence-level reward $R ( e ; v , q , a ^ { * } )$ . Given a dataset D of triplets $( v , q , a ^ { * } )$ , the objective is

$$
J (\theta) = \mathbb {E} _ {(v, q, a ^ {*}) \sim \mathcal {D}} \mathbb {E} _ {e \sim \pi_ {\theta} (\cdot | v, q)} [ R (e; v, q, a ^ {*}) ]. \tag {6}
$$

Optimizing this objective directly is unstable for long, freeform evidence sequences. We therefore adopt sufficiencydriven Group Relative Policy Optimization (GRPO) as a practical optimization strategy1.

At each update step, for a given triplet $( v , q , a ^ { * } )$ , we sample $\{ \stackrel { } { e } _ { g } \} _ { g = 1 } ^ { G }$ from the current policy. Each candidate receives a scalar reward $R _ { g } ,$ from which we compute a group-relative standardized advantage

$$
A _ {g} = \frac {R _ {g} - \bar {R}}{\sigma_ {R} + \delta}, \tag {7}
$$

where $\bar { R }$ and $\sigma _ { R }$ denote the group mean and standard deviation. Policy updates follow a PPO-style clipped surrogate objective with a KL regularization (Schulman et al., 2017) term that constrains deviation from a frozen reference policy $\pi _ { \mathrm { r e f } }$ (the pretrained model before VEPA):

$$
\begin{array}{l} J _ {\mathrm{VEPA}} (\theta) = \mathbb {E} \left[ \frac {1}{G} \sum_ {g = 1} ^ {G} \frac {1}{T _ {g}} \sum_ {t = 1} ^ {T _ {g}} \mathcal {L} _ {g, t} ^ {\text {clip}} (\theta) \right. \tag {8} \\ \left. - \beta \mathbb {D} _ {\mathrm{KL}} (\pi_ {\theta} \| \pi_ {\text {ref}}) _ {t} \right]. \\ \end{array}
$$

This formulation enables stable optimization of long evidence sequences while preserving linguistic coherence.

## 4.3. Blind Reader-Based Sufficiency Reward

To operationalize the sufficiency criterion in §3.1 without evidence annotations, we introduce a frozen blind reader model $f _ { \phi } .$ . The blind reader is an instruction-tuned language model that observes only the generated evidence e and question q, and never accesses the image.

For each training example, the blind reader is prompted to answer the question using the evidence alone and to indicate whether the evidence directly leaks the answer. Let $\hat { a } ( e , q )$ denote the predicted answer after normalization. Based on this answer, we then define a binary solvability score

$$
s (e, q, a ^ {*}) = \mathbb {I} [ \hat {a} (e, q) = a ^ {*} ], \tag {9}
$$

which encourages the evidence to be sufficient for answers.

To discourage trivial solutions that restate the answer, the blind reader emits an honesty flag indicating whether answer leakage is detected. We define

$$
h (e, q) \in \{0, 1 \}, \tag {10}
$$

where h = 1 denotes honest evidence.

Finally, to suppress degenerate repetition, we apply a weak penalty $\gamma ( e ) \in ( 0 , 1 ]$ based on simple repetition statistics. The resulting sequence-level reward can be expressed as

$$
R (e, q, a ^ {*}) = s (e, q, a ^ {*}) \cdot h (e, q) \cdot \gamma (e). \tag {11}
$$

Maximizing the expected reward encourages evidence that is sufficient for a blind reader while remaining imagegrounded and non-degenerate, without requiring any human evidence supervision.

![](images/73dcba991edb042b50541e177c0132c002e0f73bb49ab61ca6326907ade65f84.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Pretrain"] --> B["Policy Model πθ MLLM"]
  C["VEPA"] --> D["Blind Reader Auxiliary Model LLM"]
  E["Post-train"] --> F["Compute Reward"]
  B --> G["Visual Evidence (e)"]
  D --> H["Compute Reward"]
  F --> I["Compute Reward"]
  G --> J["E1"]
  G --> K["E2"]
  G --> L["E3"]
  G --> M["E4"]
  H --> N["Answer (a)"]
  H --> O["Answer (a)"]
  H --> P["Insufficient"]
  N --> Q["Sufficient"]
  O --> Q
  P --> Q
  Q --> R["Verified"]
  R --> S["A1"]
  R --> T["A2"]
  R --> U["A3"]
  R --> V["A4"]
  W["Question: How many push pins are there?"] --> B
  X["Prompt: Your goal is to provide a comprehensive description will be used by a blind person to reason about and answer the question. Therefore, your description must act as the raw visual evidence required for reasoning."] --> B
  Y["q"] --> B
  Z["P(a | v, q)"] --> B
  AA["P(e | v, q)"] --> B
  AB["P(a | e, q)"] --> B
  AC["Update policy"] --> F
  AD["Perception activate?"] --> F
  AE["Group Relative Advantage"] --> F
```
</details>

Figure 2. Framework of VEPA. VEPA is inserted between pretraining and post-training. Given an image v and a question q, the policy MLLM πθ is prompted to generate question-conditioned visual evidence and samples a group of candidate visual evidence {e}. A frozen text-only blind reader (auxiliary LLM) answers using only (q, e). We optimize πθ with a novel sufficiency-driven objective via GRPO.

Overall, the proposed VEPA reframes visual grounding as an explicit pre-alignment problem by isolating evidence generation from answer prediction. By optimizing questionconditioned visual evidence using a blind-reader–based sufficiency signal, VEPA encourages models to encode taskrelevant visual information before engaging in downstream reasoning. This design enables effective perceptual alignment without additional annotations and is complementary to standard supervised post-training, providing a principled mechanism for improving visual grounding in MLLMs.

## 5. Experiments

We conduct experiments to evaluate the effectiveness of our VEPA, and focus on the following questions:

RQ1. Downstream Performance. Does VEPA consistently improve performance across diverse multimodal benchmarks compared to the standard training pipeline?  
RQ2. Visual Grounding. Does VEPA strengthen visual grounding by encouraging the generation of questionconditioned evidence that is sufficient for reasoning?  
RQ3. Visual Reliance. Does VEPA increase reliance on visual inputs over language priors, as reflected by robustness to visual perturbations and reduced hallucination?  
RQ4. Training Dynamics. How does visual perception evolve over the course of VEPA training?

## 5.1. Experimental Setup

Implementation details. We use Qwen2-VL-2B (Wang et al., 2024) as the backbone for all model variants. During the VEPA stage, we optimize the evidence-generation policy using our sufficiency-driven GRPO on a curated subset of FineVision (Wiedmann et al., 2025). Specifically, we uniformly sample 5k training instances from the ScienceQA (Lu et al., 2022), AI2D-Merge (Kembhavi et al., 2016), ChartQA (Masry et al., 2022), Geo3K (Lu et al., 2021), TextVQA (Singh et al., 2019), and CLEVR (Johnson et al., 2017). These datasets jointly cover complementary perception-centric skills, including diagram and scientific reasoning, chart and plot understanding, geographic and map recognition, text understanding in natural images, and compositional visual reasoning.

To better align with the objective of visually grounded evidence generation, we further prioritize questions with high visual dependency, such that the correct answer cannot be reliably inferred from the question alone. This filtering yields diverse evidence patterns that better approximate the target distribution and improves the robustness of the learned evidence policy.

All GRPO training is implemented using VERL. To compute the evidence-sufficiency reward, we employ a frozen text-only blind reader based on Qwen2.5-7B-Instruct (Qwen et al., 2025), which predicts answers conditioned solely on the question and the generated evidence. Unless otherwise specified, we report exact-match accuracy on downstream VQA benchmarks with identical decoding settings across all variants. Additional implementation details and hyperparameters are provided in Appendix A.1.

## 5.2. RQ1: Downstream Performance Across Tasks

To assess whether inserting VEPA between pretraining and standard post-training improves downstream performance, we conduct controlled comparisons in which all variants share the same backbone, supervised fine-tuning (SFT) data, and SFT hyperparameters. Specifically, all models are initialized from Qwen2-VL-2B, trained with standard SFT on 20k examples, and evaluated either directly (SFT) or after an additional VEPA Etage followed by SFT (VEPA+SFT).

We consider four single-domain SFT settings using A-OKVQA (Schwenk et al., 2022), ChartQA (Masry et al., 2022), TextVQA (Singh et al., 2019), and GQA (Hudson & Manning, 2019), as well as a Mixed setting constructed by uniformly sampling 5k examples from each dataset. Models are evaluated on six benchmarks, including in-domain evaluations (marked with †), out-of-domain transfer benchmarks, and two general-purpose multimodal benchmarks (MME and MMStar). Table 1 reports accuracy comparisons.

Overall performance trends. Across all five SFT settings, VEPA consistently improves average performance (+0.5 to +3.8 points) without degrading in-domain accuracy. Indomain results are either preserved or modestly improved, indicating that VEPA does not trade task specialization for robustness. This suggests that the VEPA stage provides complementary supervision signals that are compatible with downstream SFT objectives.

Generalization under domain shift. Performance gains are often more pronounced on out-of-domain benchmarks than on the in-domain metric. In particular, perception-intensive tasks such as ChartQA and TextVQA frequently exhibit larger improvements. For example, under A-OKVQA SFT, VEPA substantially improves ChartQA and TextVQA while also improving A-OKVQA itself; under GQA SFT, VEPA again yields sizable gains on ChartQA and TextVQA, with comparatively small changes on in-domain GQA. Such asymmetric improvements are unlikely to arise from generic additional training, which would be expected to affect all benchmarks more uniformly. Instead, they are consistent with VEPA strengthening transferable, question-conditioned visual grounding.

Complementarity with data diversification. If VEPA merely compensated for limited domain coverage in SFT, its effect should diminish under the Mixed SFT setting. However, VEPA continues to improve average performance and yields a notable gain on GQA even with a stronger, diversified SFT baseline. This indicates that VEPA enforces a distinct inductive bias, greater reliance on visual evidence, rather than functioning solely as a substitute for broader supervised data.

General multimodal evaluation. VEPA also improves performance on MME and MMStar across SFT settings.

Since these benchmarks are not aligned with any specific SFT domain, the gains suggest that VEPA enhances the model’s general ability to ground answers in visual evidence, rather than improving benchmark-specific patterns.

Overall, these results demonstrate that VEPA serves as an effective intermediate alignment stage: it improves average downstream performance, preserves in-domain accuracy, and yields the largest benefits under domain shift, where transferable perceptual grounding is most critical.

Insight I. VEPA consistently improves downstream performance and robustness by activating transferable visual grounding beyond standard supervised fine-tuning.

## 5.3. RQ2: Evidence Sufficiency and Selectivity

While RQ1 establishes that VEPA improves downstream performance, it does not directly verify whether VEPA optimizes the intended objective—namely, producing questionconditioned visual evidence that is sufficient for reasoning. RQ2 therefore evaluates the quality of the learned evidence representations. For each image–question pair, we prompt either the base model or the model after the VEPA stage to generate a question-conditioned visual description, explicitly discouraging direct answer disclosure. We then provide only the question and the generated description to a frozen text-only blind reader (Qwen2.5-7B-Instruct), which attempts to answer POPE and MMStar without access to the image. Because the blind reader never observes visual inputs, higher accuracy directly reflects whether the generated evidence surfaces task-relevant visual information.

Table 2 shows that VEPA consistently improves blind-reader accuracy on both benchmarks, from 42.53% to 44.27% on MMStar and from 78.05% to 79.25% on POPE. Crucially, these gains are accompanied by shorter descriptions: the average output length decreases from 201.97 to 187.86 tokens on MMStar and from 464.21 to 365.16 tokens on POPE. This rules out verbosity as a trivial explanation for the improved solvability.

Consistent with this observation, the mean rollout length exhibits an initial transient increase followed by a gradual decrease and stabilization, rather than unbounded growth (See details in Appendix A.7). After an initial transient increase, it rapidly stabilizes, indicating convergence toward concise yet informative evidence rather than progressively longer descriptions. Taken together, these results indicate that VEPA improves the selectivity and sufficiency of generated evidence, enabling task-critical visual cues to be externalized in a compact form. This provides a direct mechanistic explanation for the stable downstream gains observed in RQ1.

Table 1. Downstream benchmark results under different SFT data settings. Accuracy (%) is reported for each target benchmark. Rows are grouped by the SFT dataset used for post-training (SFT data), comparing the SFT baseline against VEPA+SFT (shaded). Avg. denotes the macro-average over all benchmarks. † marks in-domain evaluation. Subscripts indicate the change relative to SFT (↑ improvement, ↓ decline, - no change).

<table><tr><td>SFT data</td><td>Method</td><td>A-OKVQA</td><td>ChartQA</td><td>TextVQA</td><td>GQA</td><td>MME</td><td>MMStar</td><td>Avg.</td></tr><tr><td rowspan="2">A-OKVQA</td><td>SFT</td><td> $59.70^{\dagger}$ </td><td>51.88</td><td>62.73</td><td>53.70</td><td>75.22</td><td>41.07</td><td>57.38</td></tr><tr><td>VEPA+SFT</td><td> $66.60^{\dagger\uparrow6.9}$ </td><td> $58.96^{\uparrow7.1}$ </td><td> $66.78^{\uparrow4.1}$ </td><td> $54.75^{\uparrow1.1}$ </td><td> $76.95^{\uparrow1.7}$ </td><td> $42.80^{\uparrow1.7}$ </td><td> $61.14^{\uparrow3.8}$ </td></tr><tr><td rowspan="2">ChartQA</td><td>SFT</td><td>59.60</td><td> $69.44^{\dagger}$ </td><td>77.56</td><td>59.70</td><td>81.70</td><td>40.93</td><td>64.82</td></tr><tr><td>VEPA+SFT</td><td> $59.60^{-}$ </td><td> $70.28^{\dagger\uparrow0.8}$ </td><td> $78.96^{\uparrow1.4}$ </td><td> $60.99^{\uparrow1.3}$ </td><td> $82.13^{\uparrow0.4}$ </td><td> $41.93^{\uparrow1.0}$ </td><td> $65.65^{\uparrow0.8}$ </td></tr><tr><td rowspan="2">TextVQA</td><td>SFT</td><td>34.95</td><td>68.08</td><td> $80.66^{\dagger}$ </td><td>38.40</td><td>82.13</td><td>41.87</td><td>57.68</td></tr><tr><td>VEPA+SFT</td><td> $35.50^{\uparrow0.6}$ </td><td> $68.64^{\uparrow0.6}$ </td><td> $81.44^{\dagger\uparrow0.8}$ </td><td> $40.60^{\uparrow2.2}$ </td><td> $80.55^{\downarrow1.6}$ </td><td> $42.40^{\uparrow0.5}$ </td><td> $58.19^{\uparrow0.5}$ </td></tr><tr><td rowspan="2">GQA</td><td>SFT</td><td>42.00</td><td>60.40</td><td>63.49</td><td> $64.41^{\dagger}$ </td><td>78.67</td><td>43.07</td><td>58.67</td></tr><tr><td>VEPA+SFT</td><td> $42.10^{\uparrow0.1}$ </td><td> $64.40^{\uparrow4.0}$ </td><td> $66.15^{\uparrow2.7}$ </td><td> $64.52^{\dagger\uparrow0.1}$ </td><td> $80.55^{\uparrow1.9}$ </td><td> $43.80^{\uparrow0.7}$ </td><td> $60.25^{\uparrow1.6}$ </td></tr><tr><td rowspan="2">Mixed</td><td>SFT</td><td> $62.50^{\dagger}$ </td><td> $70.32^{\dagger}$ </td><td> $79.73^{\dagger}$ </td><td> $51.07^{\dagger}$ </td><td>79.97</td><td>44.53</td><td>64.69</td></tr><tr><td>VEPA+SFT</td><td> $62.70^{\dagger\uparrow0.2}$ </td><td> $70.44^{\dagger\uparrow0.1}$ </td><td> $79.88^{\dagger\uparrow0.2}$ </td><td> $53.69^{\dagger\uparrow2.6}$ </td><td> $80.26^{\uparrow0.3}$ </td><td> $45.07^{\uparrow0.5}$ </td><td> $65.34^{\uparrow0.7}$ </td></tr></table>

Table 2. Blind-reader evaluation of evidence sufficiency. We prompt the base model and the model after the VEPA stage to generate question-conditioned descriptions, and provide only the question and the generated description to a frozen text-only blind reader (Qwen2.5-7B-Instruct) to answer POPE and MMStar without image access. We report Acc. (%, accuracy) and the average description length (tokens).

<table><tr><td rowspan="2">Method</td><td colspan="2">MMStar</td><td colspan="2">POPE</td></tr><tr><td>Acc.</td><td>Length</td><td>Acc.</td><td>Length</td></tr><tr><td>Base</td><td>42.53</td><td>201.97</td><td>78.05</td><td>464.21</td></tr><tr><td>VEPA</td><td>44.27</td><td>187.86</td><td>79.25</td><td>365.16</td></tr></table>

Insight II. VEPA promotes concise yet sufficient visual evidence, making task-relevant visual information recoverable without relying on verbose descriptions.

## 5.4. RQ3: Reliance on Visual Inputs

Table 3. Robustness under image corruption. Models are trained with A-OKVQA SFT. We report clean accuracy (%), retention under three corruption types (Blur, Partial noise, Pure noise), and AUC of the retention curve (lower is better).

<table><tr><td rowspan="2">Metric</td><td colspan="2">ChartQA</td><td colspan="2">GQA</td></tr><tr><td>SFT</td><td>VEPA+SFT</td><td>SFT</td><td>VEPA+SFT</td></tr><tr><td>Clean Acc. (%)</td><td>51.88</td><td>58.96</td><td>52.13</td><td>53.07</td></tr><tr><td>Blur (r)</td><td>0.084</td><td>0.074</td><td>0.838</td><td>0.836</td></tr><tr><td>Partial (r)</td><td>0.753</td><td>0.680</td><td>0.874</td><td>0.867</td></tr><tr><td>Pure (r)</td><td>0.062</td><td>0.052</td><td>0.525</td><td>0.527</td></tr><tr><td>AUC ↓ (over r)</td><td>0.246</td><td>0.220</td><td>0.769</td><td>0.767</td></tr></table>

While RQ1 demonstrates consistent downstream improvements from inserting VEPA, such gains could in principle arise from incidental factors, such as additional optimization or strengthened language priors. RQ3 therefore examines whether VEPA shifts model behavior toward greater reliance on valid visual input, rather than improving performance through language-only shortcuts.

We evaluate visual reliance using two benchmarks. ChartQA (Masry et al., 2022) represents a strongly visiondependent setting that requires reading chart-specific visual content, whereas GQA (Hudson & Manning, 2019) serves as a broad VQA benchmark for which prior work reports substantial question-only accuracy, indicating that language priors can partially support answering. Both models, standard SFT and VEPA followed by SFT, are trained on the same A-OKVQA SFT data to control for the supervised signal. To probe reliance on visual input, we perform counterfactual evaluations by corrupting images at inference time. In addition to clean images, we consider three corruption settings: Gaussian blur, partial noise, and pure noise. Table 3 reports clean accuracy and retention under each corruption type, where retention is defined as $r \triangleq \mathrm { A c c } _ { \mathrm { c o r r u p t } } / \mathrm { A c c } _ { \mathrm { c l e a n } }$ . We further summarize robustness using the AUC computed from $( r _ { \mathrm { b l u r } } , r _ { \mathrm { p a r t i a l } } , r _ { \mathrm { p u r e } } )$ in order; lower AUC indicates stronger reliance on valid visual input. Under clean images, VEPA improves accuracy on both benchmarks.

On ChartQA, image corruption leads to substantial performance degradation, and VEPA exhibits lower retention and lower retention AUC than standard SFT. Given the strong visual dependency of ChartQA, this pattern indicates that VEPA relies more heavily on visual evidence rather than maintaining performance via language priors when the image becomes unreliable. The effect is most pronounced under partial noise, where visual information remains partially informative and grounding is still actionable. On GQA, VEPA closely matches the retention behavior of standard SFT while improving clean accuracy. This is consistent with GQA serving as a general-purpose benchmark in which language priors are known to contribute to performance. Taken together, these results indicate that VEPA selectively increases reliance on visual input when vision is essential, without degrading robustness in broader VQA settings.

![](images/6206acd08b0ed817601852656189a28e2e26e0c1119c2e81b6a49adaa01341a5.jpg)

<details>
<summary>line chart</summary>

| Training Steps | Accuracy (%) |
| -------------- | ------------ |
| 30             | 74.8         |
| 60             | 76.8         |
| 90             | 77.0         |
| 120            | 77.8         |
| 150            | 78.5         |
| 180            | 78.4         |
| 220            | 79.5         |
</details>

![](images/760dd122ea3bca1f38e8847119ead5545ef40b8aa4c42a7c8ba9556d9c94e3ac.jpg)

<details>
<summary>line chart</summary>

| Training Steps | Accuracy (%) |
| -------------- | ------------ |
| 30             | 29.0         |
| 60             | 29.0         |
| 90             | 29.5         |
| 120            | 30.0         |
| 150            | 30.0         |
| 180            | 30.5         |
| 220            | 30.0         |
</details>

Figure 3. Performance evolution during VEPA. (A) illustrates the Accuracy trends on the MME dataset, while (B) presents the corresponding results on the MMStar dataset.

Insight III. Counterfactual image corruption reveals that VEPA increases reliance on valid visual input rather than amplifying language priors.

## 5.5. RQ4. Training Dynamics

To analyze how visual grounding evolves during VEPA, we periodically evaluate intermediate policy checkpoints throughout training. At each checkpoint, we measure zeroshot accuracy on MME and MMStar, using identical evaluation and decoding settings.

Figure 3 shows a clear upward performance trend as training progresses. Accuracy on MME increases from 74.73% to 79.28%, while MMStar improves from 21.53% to 24.87%, with only minor non-monotonic fluctuations. The smooth improvement trajectory indicates that VEPA produces stable, incremental gains rather than relying on abrupt phase transitions. Notably, the rate of improvement is highest during the early stage of training and gradually saturates. This pattern suggests that VEPA first rapidly improves coarse perceptual alignment, after which subsequent updates primarily refine evidence selectivity and training stability.

Overall, these dynamics support the view that VEPA progressively strengthens visual grounding throughout training, consistent with the evidence-sufficiency objective and the blind-reader reward design.

Insight IV. Visual grounding improves progressively during VEPA training, with early rapid gains followed by later-stage refinement, indicating stable and cumulative perceptual alignment.

## 5.6. Sensitivity Analysis

We analyze the sensitivity of VEPA along two dimensions: (i) auxiliary blind-reader capacity and (ii) reinforcement learning data scale.

![](images/180887b34660d11e49776c8b4a7a9b36975eb8037880abfb2b76d7047ccea5d7.jpg)

<details>
<summary>bar-line hybrid chart</summary>

| Data Size | Accuracy (%) |
| :--- | :--- |
| 3k | 42.27 |
| 5k | 42.80 |
| 10k | 43.27 |
</details>

Figure 4. Data size ablation. We scale training data from 3k to 10k samples.The results demonstrate a consistent improvement in accuracy as the data size increases.

Blind-reader capacity. To evaluate dependence on the blind-reader strength, we replace the default Qwen2.5-7B-Instruct auxiliary model with a smaller Qwen2.5-3B-Instruct model while keeping all other VEPA settings unchanged. Table 4 shows that VEPA remains effective with the reduced auxiliary capacity. Without downstream SFT, VEPA improves MMStar accuracy from 28.60% to 32.33% and GQA from 45.07% to 53.40%. When followed by A-OKVQA SFT, both auxiliary choices yield comparable final performance, reaching 43.00% on MMStar and 54.57% on GQA with the 3B auxiliary model, versus 42.80% and 54.75% with the 7B auxiliary model. These results indicate that VEPA does not critically depend on a large auxiliary model and remains robust across auxiliary capacity.

RL data scale. As illustrated in 4 we further study data efficiency by constructing VEPA RL training sets of size N ∈ {3k, 5k, 10k} via subsampling from the same filtered data mixture. VEPA remains effective even with a lightweight RL dataset (3k examples), while performance continues to improve as the dataset scales to 10k. This suggests that VEPA is both data-efficient and scalable, with additional headroom available under larger RL data regimes.

Table 4. Auxiliary model sensitivity. Accuracy (%) on MMStar and GQA for VEPA trained with various blind readers.

<table><tr><td rowspan="2">Dataset</td><td colspan="2">3B aux</td><td colspan="2">7B aux</td></tr><tr><td>VEPA</td><td>VEPA+SFT</td><td>VEPA</td><td>VEPA+SFT</td></tr><tr><td>MMStar</td><td>32.33</td><td>43.00</td><td>30.07</td><td>42.80</td></tr><tr><td>GQA</td><td>53.40</td><td>54.57</td><td>51.46</td><td>54.75</td></tr></table>

## 6. Conclusion

We introduced Visual Evidence Pre-Alignment (VEPA), an intermediate training stage inserted between captionbased pretraining and downstream post-training to explicitly strengthen visual grounding before task-level reasoning. Consistent with the principle of see first, answer later, VEPA encourages models to extract question-conditioned visual evidence prior to answer generation, improving how visual information is utilized during inference. Methodologically, VEPA introduces a sufficiency-driven reinforcement learning objective that enables perceptual alignment using standard VQA supervision, without requiring additional evidence annotations. Across diverse benchmarks and training settings, VEPA consistently improves performance on visually demanding tasks while preserving in-domain accuracy. Further analyses indicate that these improvements are associated with more reliable and transferable visual grounding. Overall, these findings suggest that explicitly separating perceptual alignment from downstream reasoning provides a practical and scalable direction for improving multimodal models. This perspective suggests a promising direction for future multimodal training paradigms that explicitly structure perceptual alignment as a first-class objective alongside reasoning optimization.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for fewshot learning. Advances in neural information processing systems, 35:23716–23736, 2022.  
Chen, D., Cahyawijaya, S., Ishii, E., Chan, H. S., Bang, Y., and Fung, P. What makes for good image captions? In Workshop on Machine Learning and Compression, NeurIPS 2024, 2024a.  
Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37: 27056–27087, 2024b.  
Chen, X., Djolonga, J., Padlewski, P., Mustafa, B., Changpinyo, S., Wu, J., Ruiz, C. R., Goodman, S., Wang, X., Tay, Y., et al. On scaling up a multilingual vision and language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14432–14444, 2024c.  
Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.  
Dong, H., Kang, Z., Yin, W., LiangXiao, L., ChaoFeng, C., and Jiao, R. Scalable vision language model training

via high quality data curation. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 33272– 33293, Vienna, Austria, July 2025. Association for Computational Linguistics.

Dubois, Y., Bloem-Reddy, B., Ullrich, K., and Maddison, C. J. Lossy compression for lossless prediction. Advances in Neural Information Processing Systems, 34:14014– 14028, 2021.

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025.

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6904–6913, 2017.

Hu, Y., Hua, H., Yang, Z., Shi, W., Smith, N. A., and Luo, J. Promptcap: Prompt-guided image captioning for vqa with gpt-3. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2963–2975, 2023.

Hudson, D. A. and Manning, C. D. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6700– 6709, 2019.

Johnson, J., Hariharan, B., Van Der Maaten, L., Fei-Fei, L., Lawrence Zitnick, C., and Girshick, R. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2901–2910, 2017.

Kembhavi, A., Salvato, M., Kolve, E., Seo, M., Hajishirzi, H., and Farhadi, A. A diagram is worth a dozen images. In European conference on computer vision, pp. 235–251. Springer, 2016.

Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pp. 19730–19742. PMLR, 2023a.

Li, Q., Tao, Q., Joty, S., Cai, J., and Luo, J. Vqa-e: Explaining, elaborating, and enhancing your answers for visual questions. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 552–567, 2018.

Li, W., Huang, Z., Li, H., Lu, L., Lu, Y., Tian, X., Shen, X., and Ye, J. Visual evidence prompting mitigates hallucinations in large vision-language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4048–4080, 2025.  
Li, X., Yin, X., Li, C., Zhang, P., Hu, X., Zhang, L., Wang, L., Hu, H., Dong, L., Wei, F., et al. Oscar: Objectsemantics aligned pre-training for vision-language tasks. Lecture Notes in Computer Science, pp. 121–137, 2020.  
Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen, J.-R. Evaluating object hallucination in large visionlanguage models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 292–305, 2023b.  
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L. Microsoft coco: ´ Common objects in context. In European conference on computer vision, pp. 740–755. Springer, 2014.  
Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.  
Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 26296–26306, 2024.  
Liu, Z., Sun, Z., Zang, Y., Dong, X., Cao, Y., Duan, H., Lin, D., and Wang, J. Visual-rft: Visual reinforcement fine-tuning. CoRR, 2025.  
Lu, P., Gong, R., Jiang, S., Qiu, L., Huang, S., Liang, X., and Zhu, S.-C. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165, 2021.  
Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.  
Luo, T., Cao, A., Lee, G., Johnson, J., and Lee, H. Probing visual language priors in vlms. In Forty-second International Conference on Machine Learning, 2025.  
Masry, A., Do, X. L., Tan, J. Q., Joty, S., and Hoque, E. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pp. 2263–2279, 2022.  
Misra, I., Lawrence Zitnick, C., Mitchell, M., and Girshick, R. Seeing through the human reporting bias: Visual classifiers from noisy human-centric labels. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2930–2939, 2016.  
OpenAI. Gpt-5 is here. https://openai.com/ gpt-5, 2025. Accessed: 22 September 2025.  
Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report, 2025.  
Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36: 53728–53741, 2023.  
Rao, V. N., Zhen, X., Hovsepian, K., and Shen, M. A first look: Towards explainable textvqa models via visual and textual explanations. In Proceedings of the Third Workshop on Multimodal Artificial Intelligence, pp. 19– 29, 2021.  
Rohrbach, A., Hendricks, L. A., Burns, K., Darrell, T., and Saenko, K. Object hallucination in image captioning. arXiv preprint arXiv:1809.02156, 2018.  
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
Schwenk, D., Khandelwal, A., Clark, C., Marino, K., and Mottaghi, R. A-okvqa: A benchmark for visual question answering using world knowledge. In European conference on computer vision, pp. 146–162. Springer, 2022.  
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.  
Shen, H., Liu, P., Li, J., Fang, C., Ma, Y., Liao, J., Shen, Q., Zhang, Z., Zhao, K., Zhang, Q., et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.  
Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8317–8326, 2019.  
Wang, H., Qu, C., Huang, Z., Chu, W., Lin, F., and Chen, W. Vl-rethinker: Incentivizing self-reflection of visionlanguage models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025.  
Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.  
Wen, X., Liu, Z., Zheng, S., Ye, S., Wu, Z., Wang, Y., Xu, Z., Liang, X., Li, J., Miao, Z., et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025.  
Wiedmann, L., Zohar, O., Mahla, A., Wang, X., Li, R., Frere, T., von Werra, L., Gosthipaty, A. R., and Marafioti, A. Finevision: Open data is all you need. arXiv preprint arXiv:2510.17269, 2025.  
Xia, J., Zang, Y., Gao, P., Li, S., and Zhou, K. Visionary-r1: Mitigating shortcuts in visual reasoning with reinforcement learning. arXiv preprint arXiv:2505.14677, 2025.  
Yang, Q. A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Dong, G., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y.-C., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., Qiu, Z., Quan, S., and Wang, Z. Qwen2.5 technical report. ArXiv, abs/2412.15115, 2024.  
Zeng, Y., Qi, Y., Zhao, Y., Bao, X., Chen, L., Chen, Z., Huang, S., Zhao, J., and Zhao, F. Enhancing large visionlanguage models with ultra-detailed image caption generation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 26703–26729, 2025.  
Zhai, S., Bai, H., Lin, Z., Pan, J., Tong, P., Zhou, Y., Suhr, A., Xie, S., LeCun, Y., Ma, Y., et al. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. Advances in neural information processing systems, 37:110935–110971, 2024.  
Zheng, K., Zhang, Y., Wu, W., Lu, F., Ma, S., Jin, X., Chen, W., and Shen, Y. Dreamlip: Language-image pre-training with long captions. In European Conference on Computer Vision, pp. 73–90. Springer, 2024.  
Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., and Yu, X. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In The Twelfth International Conference on Learning Representations, 2024.

## A. Appendix

## A.1. Experiment Settings

Dataset and Benchmarks. To strictly evaluate the performance of our proposed VEPA, we conduct experiments on comprehensive benchmarks that assess diverse multimodal capabilities. We first include GQA (Hudson & Manning, 2019) and A-OKVQA (Schwenk et al., 2022) to evaluate compositional visual reasoning and knowledge-intensive question answering, respectively. Although GQA applies bias-mitigation via answer-distribution smoothing, it still retains non-trivial language priors: Hudson (2019) report that a question-only LSTM baseline achieves 42.1% accuracy, indicating that a non-negligible portion of questions can be partially resolved from the question text alone.

To assess the model’s ability to process fine-grained visual information, we utilize TextVQA (Singh et al., 2019) for optical character recognition in natural scenes and ChartQA (Masry et al., 2022) for logical reasoning over complex charts. To further extend the evaluation to scientific and geometric domains, we incorporate ScienceQA (Lu et al., 2022), AI2D-Merge (Kembhavi et al., 2016), and Geo3K (Lu et al., 2021), which challenge the model with textbook-grade diagrams and mathematical reasoning. Additionally, we integrate CLEVR (Johnson et al., 2017) and FineVision (Wiedmann et al., 2025) to strictly test synthetic compositional logic and fine-grained visual discrimination, respectively. For a holistic evaluation of MLLMs, we adopt MME (Fu et al., 2025), which covers a broad range of perception and cognition tasks, and MMStar (Chen et al., 2024b), a benchmark specifically curated to test models on hard samples across multiple disciplines. Finally, we employ POPE (Li et al., 2023b) to specifically measure the object hallucination rates and the robustness of the generated responses.

Models. In our experiments, we primarily utilize Qwen2-VL (Wang et al., 2024) as the backbone architecture for training our VEPA framework. This model is selected for its state-of-the-art performance in visual understanding and its capability to handle arbitrary image resolutions through dynamic resolution support. To assist with auxiliary tasks such as data processing and response refinement, we incorporate a suite of lightweight yet capable language models. Specifically, we employ the 3B and 7B variants of the Qwen2.5 series (Yang et al., 2024), which offer a strong balance between efficiency and reasoning capability. Furthermore, we adopt GPT-5-nano (OpenAI, 2025), the most efficient variant in the GPT-5 family, which is specifically optimized for high-throughput instruction following and low-latency applications, serving as a robust baseline for commercial lightweight systems.

Evaluation Metrics. We report Accuracy as the primary metric across all benchmarks. For open-ended generation tasks, we implement a deterministic matching protocol to address linguistic variations. Specifically, both predictions and ground truths undergo normalization, including case lowering, punctuation removal, and stop-word stripping. To further handle morphological discrepancies, we expand the ground truth into its inflectional variants (e.g., singular and plural forms) to verify semantic equivalence against the prediction. For multiple-choice tasks, we employ a hierarchical parsing strategy that prioritizes the extraction of explicit option labels. When labels are absent, the evaluation falls back to semantic content matching, which verifies the presence of the correct option’s text. Crucially, this content matching enforces an exclusivity constraint: a prediction is considered correct only if it contains the target content without including text from incorrect distractors, thereby preventing false positives from hallucinated candidates.

Implementation Details. We implement our proposed VEPA framework using Group Relative Policy Optimization (GRPO) to fine-tune the Qwen2-VL-2B (Wang et al., 2024) backbone. The training process leverages Fully Sharded Data Parallel (FSDP) with bfloat16 precision to maximize computational efficiency. We employ the AdamW optimizer with a constant learning rate of $1 \times 1 0 ^ { - 6 }$ following a 5% warmup phase, and set the KL divergence coefficient $\beta$ to 0.01 to maintain policy stability. For data generation, we utilize the VLLM engine to sample G = 4 candidate captions for each visual query with a temperature of 0.9. Crucially, our reward mechanism incorporates a lightweight auxiliary judge (Qwen2.5-7B-Instruct) that evaluates responses based on three dimensions: correctness, honesty, and fluency. Specifically, we assign a positive reward (+1.0) for factually correct answers derived from visual evidence, while imposing strict penalties for ”cheating” behaviors—such as outputting the answer directly without description (-1.0)—and for linguistic repetition (-0.5). The system prompt is carefully engineered to simulate a “blind reader” scenario, compelling the model to function as an objective visual analyst rather than a direct question answerer.

## A.2. Prompts Used in Experiments

Below are the prompts used in the experiments.

Caption Generation elicits question-conditioned visual evidence: the model is instructed to produce a comprehensive, objective description of visible elements (including text and spatial relations) that are relevant to the question, while explicitly forbidding direct answers, so that the output can serve as standalone evidence for a blind solver. Auxiliary Judge instantiates our blind-reader mechanism: given the generated evidence and the question, it first performs cheat detection (flagging outputs that answer without visual grounding), and then attempts to solve the question using only the provided description; its structured output is used to reward grounded evidence and penalize answer leakage. Finally, VQA Judge is used for automatic evaluation: it compares the model prediction against the ground-truth answer under semantic-equivalence rules and returns a single Boolean decision (TRUE/FALSE), enabling scalable and reproducible scoring.

## Caption Generation

You are an expert visual analyst. Given an image and a question, your goal is to provide a comprehensive and detailed objective description of the visual content.

Context: This description will be used by a blind person to reason about and answer the question. Therefore, your description must act as the raw visual evidence required for reasoning.

## Strict Guidelines:

1. Focus on Visual Evidence, NOT Conclusions: Describe the visible elements, text, spatial relationships, colors, and specific details that are relevant to the question. Do not just state the final answer.  
2. Be Comprehensive: Do not be concise. Include ALL visual details, no matter how small, if they are necessary to support the reasoning process for the question.  
3. No Direct Answers: You must NOT explicitly output the answer(e.g., do not say “The answer is B”, “It is a cat” or “Select Option A”). Instead, provide all the necessary visual evidence so that a blind person can deduce the answer themselves.  
4. Format: Output ONLY the detailed caption.

## Auxiliary Judge

You are an impartial judge and a blind solver. I will provide a ‘Visual Description’ generated by an AI and a ‘Question’.

## Task 1: Cheat Detection

Check if the description is trying to cheat. ‘Cheating’ means:

\- Directly explicitly stating ‘The answer is X’ WITHOUT visual context.

\- Providing the answer directly (e.g., just saying ‘Yes’, ‘2’, or ‘Blue’) WITHOUT any descriptive sentence.

Important Exception: It is NOT cheating if the description states a visual fact that happens to be the answer.

## Task 2: Solve

Use the description to answer the question briefly.

## Output Format (CRITICAL):

If you detect cheating, start with: “[[CHEAT]]” and then the answer.

If not, start with: “[[honest]]” and give the answer.

Visual Description: {caption}

Question: {question}

## VQA Judge

You are an expert VQA (Visual Question Answering) judge.

## Your Task:

Evaluate if the model’s PREDICTION is correct based on the GROUND TRUTH (GT).

## Judgment Rules (Important):

1. Core Meaning: Focus on semantic meaning.  
2. Ignore Trivialities: Ignore capitalization, punctuation, minor phrasing.  
3. Rationale Check: If GT has explanation and Prediction follows similar reasoning logic, count as CORRECT.

## Data:

- Question: {question}  
- Ground Truth: {gt}  
- Prediction: {prediction}

## Output Requirement:

You must output ONLY one word.

\- Output ‘TRUE’ if the prediction is semantically correct.

\- Output ‘FALSE’ if the prediction is incorrect.

Do NOT output JSON. Do NOT output any explanation.

![](images/6e9ef98baf1002ab223262840b2762f32ef4c22bbc760bb2ef529373a27d570a.jpg)

<details>
<summary>bar chart</summary>

POPE & Qwen2.5-7B-Instruct
| Category | BASE (%) | VEPA (%) |
| :--- | :--- | :--- |
| Adversarial | 77.25 | 77.83 |
| Random | 79.74 | 80.76 |
| Popular | 77.1 | 79.19 |
</details>

(A)

![](images/c023f6b4143af8615c0b024d7b79278d4ddc6c65882fe7c81d829b88bc849feb.jpg)

<details>
<summary>bar chart</summary>

POPE & GPT-5 nano
| Category | BASE (%) | VEPA (%) |
| :--- | :--- | :--- |
| Adversarial | 77.97 | 79.54 |
| Random | 82.09 | 83.26 |
| Popular | 78.23 | 80.97 |
</details>

(B)  
Figure 5. Detailed accuracy comparison on the POPE dataset across three categories: Adversarial, Random, and Popular. (A) displays results for Qwen2.5-7B-Instruct, while (B) shows GPT-5 Nano.

## A.3. More Experiment Results on Evidence Quality

This section reports additional evidence-quality results complementing §5.3. Beyond the aggregate scores, we further decompose POPE into its three subsets (Adversarial, Random, Popular) to assess whether VEPA improves evidence sufficiency across query types.

Figure 5 shows consistent gains in blind-reader accuracy for all POPE subsets. With QWEN2.5-7B-INSTRUCT as the blind reader, VEPA improves accuracy from 77.25% to 77.83% on ADVERSARIAL, from 79.74% to 80.76% on RANDOM, and from 77.10% to 79.19% on POPULAR. The largest improvement occurs on POPULAR (+2.09%), indicating that VEPA particularly enhances the extraction of salient, question-relevant visual cues, while remaining effective under adversarial instances.

These conclusions are robust to the choice of the blind reader. Using GPT5-NANO yields the same pattern: 77.97% to 79.54% on ADVERSARIAL, 82.09% to 83.26% on RANDOM, and 78.23% to 80.97% on POPULAR. The agreement across readers suggests that the improved accuracy reflects stronger evidence sufficiency rather than evaluator-specific artifacts.

Overall, the POPE breakdown reinforces the main finding of RQ2 5.3: VEPA improves the recoverability of task-critical visual cues from generated evidence, thereby strengthening perceptual grounding without relying on dense descriptions.

## A.4. More Experiment results on Performance Analysis

To complement the aggregated results presented in Table 1, we provide a detailed performance analysis on the MME and MMStar benchmarks. While the main text reports average scores to demonstrate overall trends, this section breaks down the performance of the SFT and VEPA+SFT settings across specific sub-tasks to evaluate the consistency and robustness of our method.

We specifically analyze ten fine-grained dimensions, comprising seven categories from MME (Color, Count, Numerical Calculation, OCR, Position, Poster, and Text Translation) and four core capability dimensions from MMStar (Coarse Perception, Logical Reasoning, Math, and Science & Technology).

Figure 6 and Figure 7 visualizes the accuracy comparisons across five distinct SFT data settings (A-OKVQA, ChartQA, TextVQA, GQA, and Mixed). The results reveal the following key observations:

• Consistent Gains: Consistent with the averaged results, VEPA+SFT outperforms the standard SFT baseline in the majority of fine-grained categories. Notably, significant improvements are observed in tasks requiring strong visual dependency, such as Numerical Calculation, OCR, and Coarse Perception. This supports our hypothesis that the VEPA stage enhances the model’s ability to ground textual generation in visual evidence.  
• Robustness: While we observe minor performance fluctuations in a few specific settings (e.g., slight regressions in

select isolated metrics), the overall performance landscape remains stable. VEPA demonstrates robustness across diverse tasks, indicating that the method improves general multimodal capabilities without trading off performance in specific domains.

These fine-grained statistics further corroborate that VEPA serves as an effective intermediate stage, fostering transferable perceptual skills that generalize beyond the specific distribution of the SFT data.

## A.5. More Experiment Results on Sensitivity of Dataset

We vary the number of RL training instances used in VEPA by sampling 3k, 5k, or 10k examples from the same filtered mixture. Figure 4 illustrates accuracy after VEPA followed by A-OKVQA-based SFT. Performance on MMStar increases monotonically from 42.27% to 42.80% and 43.27% as the VEPA data scale grows, suggesting additional headroom from further scaling perception-alignment data. And below is the detail performance comparison on MMStar sub-categories.

Table 5. Detailed performance comparison on MMStar sub-categories. We report the accuracy (%) across varying training scale (3k, 5k, 10k) for both VEPA and VEPA + SFT settings.

<table><tr><td rowspan="2">Category</td><td colspan="3">VEPA</td><td colspan="3">VEPA + SFT</td></tr><tr><td>3k</td><td>5k</td><td>10k</td><td>3k</td><td>5k</td><td>10k</td></tr><tr><td>Coarse Perception</td><td>41.60</td><td>44.80</td><td>46.40</td><td>60.80</td><td>59.20</td><td>57.60</td></tr><tr><td>Fine-grained Perception</td><td>32.40</td><td>34.00</td><td>31.60</td><td>38.40</td><td>36.80</td><td>37.60</td></tr><tr><td>Instance Reasoning</td><td>38.80</td><td>38.80</td><td>39.20</td><td>43.60</td><td>47.20</td><td>46.00</td></tr><tr><td>Logical Reasoning</td><td>28.80</td><td>25.60</td><td>26.80</td><td>41.20</td><td>40.80</td><td>42.00</td></tr><tr><td>Math</td><td>20.00</td><td>20.00</td><td>22.00</td><td>34.80</td><td>37.60</td><td>39.20</td></tr><tr><td>Science &amp; Technology</td><td>17.20</td><td>17.20</td><td>16.80</td><td>34.80</td><td>35.20</td><td>37.20</td></tr></table>

## A.6. Case Study

We present six qualitative examples in Figures 8–13 to probe whether VEPA activates perception. To minimize confounds, the base model in this subsection is the purely pre-trained Qwen2-VL-2B checkpoint (i.e., before any SFT or preference post-training). We then compare it with the corresponding VEPA model obtained by applying only the intermediate VEPA stage on top of this base. Importantly, we do not enforce any extra response format: the model is simply asked to answer each question with the default prompt, without additional instructions that would explicitly demand “look at the image,” “explain,” or “provide evidence.” This design allows us to attribute qualitative differences primarily to changes in visual processing induced by VEPA, instead of compliance with explicit instruction templates.

Under this controlled setting, we observe a consistent shift in the information the model chooses to attend to and verbalize. The base model frequently falls back to under-specified, generic, or prior-driven responses, especially for counting and comparison, where success requires enumerating multiple entities and tracking their relations. Its outputs often omit critical perceptual details (e.g., the exact number of instances, distinguishing attributes, or the relevant subset defined by the question), making the answers difficult to justify from the image. By contrast, the VEPA model is more likely to spontaneously surface question-relevant visual facts—explicitly enumerating objects, mentioning discriminative attributes, reading visible text when needed, and describing spatial/relational cues that enable the downstream reasoning step. Notably, this behavior emerges without being instructed to produce explanations, suggesting that VEPA increases the model’s tendency to consult and extract task-relevant visual evidence during generation.

These cases therefore provide qualitative support for our mechanism-level claim. Since both models share the same architecture and the same pretraining, and since we do not add instruction constraints at inference time, the observed differences are most naturally explained by VEPA strengthening the model’s internal reliance on perceptual signals. In other words, VEPA appears to activate (and make accessible) visual perception for decision making, rather than simply inducing a different style of instruction-following responses.

![](images/686c99c3c0745ccae2eec46675aba8e12f05b86f4016188a514a0bb3b0a593cc.jpg)

<details>
<summary>bar chart</summary>

Color
| Category | SFT (%) | VEPA+SFT (%) |
| :--- | :--- | :--- |
| A-OKVQA | 96.67 | 95.00 |
| ChartQA | 95.00 | 96.67 |
| TextVQA | 96.67 | 98.33 |
| GQA | 88.33 | 98.3 |
| Mixed | 93.33 | 93.33 |
</details>

(a) MME: Color

![](images/4d63b2927411a73690ea607535d25686e895e1850e6184c379969e89a19bcfd0.jpg)

<details>
<summary>bar chart</summary>

Count
| Category | SFT (%) | VEPA+SFT (%) |
| :--- | :--- | :--- |
| A-OKVQA | 76.67 | 75.00 |
| ChartQA | 81.67 | 81.67 |
| TextVQA | 80.00 | 80.00 |
| GQA | 78.33 | 80.00 |
| Mixed | 83.33 | 83.33 |
</details>

(b) MME: Count

![](images/aa4fd2bbf5881f122a90b4dd2c9ee37c98f78d9e154ef5160ec786d1576d2f61.jpg)

<details>
<summary>bar chart</summary>

Numerical Calculation
| Category | SFT (%) | VEPA+SFT (%) |
| :--- | :--- | :--- |
| A-OKVQA | 47.50 | 47.50 |
| ChartQA | 40.00 | 47.50 |
| TextVQA | 45.00 | 47.50 |
| GQA | 45.00 | 47.50 |
| Mixed | 52.50 | 55.00 |
</details>

(c) MME: Numerical Calculation

![](images/f2aad028c20152467662f3582c70a4ae8baa607f4f9b844ac859f63fb06095f9.jpg)

<details>
<summary>bar chart</summary>

OCR
| Category | SFT (%) | VEPA+SFT (%) |
| :--- | :--- | :--- |
| A-OKVQA | 65.00 | 67.50 |
| ChartQA | 60.00 | 60.00 |
| TextVQA | 62.50 | 65.00 |
| GQA | 57.50 | 65.00 |
| Mixed | 55.00 | 55.00 |
</details>

(d) MME: OCR

![](images/a0d923446dc319fb9529860da96758759b14ba5a9a5275b974caa818fbd44484.jpg)

<details>
<summary>bar chart</summary>

Position
| Position | SFT (%) | VEPA+SFT (%) |
| :--- | :--- | :--- |
| A-OKVQA | 71.67 | 71.67 |
| ChartQA | 80.00 | 80.00 |
| TextVQA | 78.33 | 80.00 |
| GQA | 76.67 | 80.00 |
| Mixed | 78.33 | 78.33 |
</details>

(e) MME: Position

![](images/2b683e225740500673becfe257c1c6cb37e4dbe23c95a386407f1bfda70aa522.jpg)

<details>
<summary>bar chart</summary>

Posters
| Category | SFT (%) | VEPA+SFT (%) |
| :--- | :--- | :--- |
| A-OKVQA | 75.51 | 77.89 |
| ChartQA | 87.07 | 87.41 |
| TextVQA | 82.99 | 82.31 |
| GQA | 82.31 | 82.31 |
| Mixed | 82.99 | 82.99 |
</details>

(f) MME: Poster

![](images/d09da85e38652440de84f3f57fa278012963826d946f4a4d46595fa725413a78.jpg)

<details>
<summary>bar chart</summary>

Text Translation
| Task | SFT (%) | VEPA+SFT (%) |
| :--- | :--- | :--- |
| A-OKVQA | 70.00 | 90.00 |
| ChartQA | 90.00 | 87.50 |
| TextVQA | 90.00 | 90.00 |
| GQA | 85.00 | 90.00 |
| Mixed | 90.00 | 90.00 |
</details>

(g) MME: Text Translation  
Figure 6. Fine-grained performance breakdown on MME. We compare the baseline SFT against VEPA+SFT across various sub-tasks. The results demonstrate that VEPA provides consistent improvements across most categories, particularly in tasks requiring precise visual grounding.

![](images/cc679085f6ab5e7da9f5181d2922b8a6ffd0efd3d70bb090c73bac0d95de4cba.jpg)

<details>
<summary>bar chart</summary>

Coarse Perception
| Category | SFT (%) | VEPA+SFT (%) |
| :--- | :--- | :--- |
| A-OKVQA | 57.60 | 59.20 |
| ChartQA | 53.60 | 56.00 |
| TextVQA | 52.40 | 54.00 |
| GQA | 54.40 | 56.40 |
| Mixed | 59.20 | 57.60 |
</details>

(a) MMStar: Coarse Perception

![](images/f309f391b5c04de9ceac75100ef321802e08e068dfa96c31c9b9e80231e080fe.jpg)

<details>
<summary>bar chart</summary>

Logical Reasoning
| Method | SFT (%) | VEPA+SFT (%) |
| :--- | :--- | :--- |
| A-OKVQA | 41.20 | 40.80 |
| ChartQA | 40.40 | 40.40 |
| TextVQA | 42.80 | 42.80 |
| GQA | 44.00 | 44.40 |
| Mixed | 42.80 | 44.00 |
</details>

(b) MMStar: Logical Reasoning

![](images/8363701201c8665a57b17d12446643923a65596effd92c8e0b1f5b724451243f.jpg)

<details>
<summary>bar chart</summary>

Math
| Category | SFT (%) | VEPA+SFT (%) |
| :--- | :--- | :--- |
| A-OKVQA | 36.40 | 37.60 |
| ChartQA | 35.60 | 37.20 |
| TextVQA | 38.40 | 38.00 |
| GQA | 38.40 | 42.40 |
| Mixed | 42.00 | 42.40 |
</details>

(c) MMStar: Math

![](images/6639591523646bf8427fc7a39494c2f69665acf835e421c24ef5024043c5a60d.jpg)

<details>
<summary>bar chart</summary>

Science & Technology
| Category | SFT (%) | VEPA+SFT (%) |
| :--- | :--- | :--- |
| A-OKVQA | 30.40 | 35.60 |
| ChartQA | 25.60 | 28.40 |
| TextVQA | 29.60 | 29.60 |
| GQA | 33.60 | 30.80 |
| Mixed | 32.00 | 32.00 |
</details>

(d) MMStar: Sci-Tech  
Figure 7. Fine-grained performance breakdown on MMStar. We compare the baseline SFT (blue) against VEPA+SFT (green) across various sub-tasks. The results demonstrate that VEPA provides consistent improvements across most categories, particularly in tasks requiring precise visual grounding.

![](images/e003ba301a26d3ea45a978750db0fc462532884c24af18b2c15f1bd58718365e.jpg)

<details>
<summary>text_image</summary>

Answer based on the image:
How many push pins are
there?
Based on the image, there are 8 push pins visible.
They are scattered around the desk and bulletin
board.
There are 6 push pins visible in the image.
These are the push pins that are attached to
the beige bulletin board behind the desk. They
are located at different positions on the
board.
</details>

Figure 8. Case 1 on Counting

![](images/d536326161caaeebc565f706b1287844a2958d9e6802e97668fff273f1a356c9.jpg)

<details>
<summary>text_image</summary>

Answer based on the image:
How many people are in the
photograph?

The photograph shows three people, two of whom
are in green and white jerseys and one in a red
jersey.

There are four people in the photograph. They are
the three men in the foreground, and one person in
the background.
</details>

Figure 9. Case 2 on Counting

![](images/ba295262b4cdef59ab3c62f8513cc36b40351d701968230819028d88a04ecf6b.jpg)

![](images/fdae3a9b0e4b7bf45d6347513790989c9b5f5571a4660553dc057d628e4d869c.jpg)

Answer based on the image: How many horses?What color are these horses respectively?

![](images/13d7c4d520f62b8a18639826ae5960093e967760d86487310fb6aa00164d81df.jpg)

Base

The horses in the image are:1. A brown horse with a white mane and tail.2. A brown horse with a white mane and tail.3. A brown horse with a white mane and tail.So there are three brown horses..

![](images/3159118c60ddf8f07d355c9b82b0902b24a06429670c2eb74a0a6b366b59d0c3.jpg)

VEPA

The image shows 4 horses. The horses are brown, white, and another color that is not fully visible. The horses are spread out in a line, with the brown horse on the left, followed by the white horse, then the brown horse in the middle, and finally the brown horse on the right. The horses are all grazing on the grass. The brown horse has a brown coat, the white horse....

Figure 10. Case 3 on Counting and Attribute  
![](images/46bf669d6e84894c89fe7526c35fbf382a6efb129995943de6e345630b11a724.jpg)

![](images/3aac36a1750e31dabc5f5421b38c2feeb13bff98a0b3e4130f4c75afc1966912.jpg)

Answer based on the image: what color is the train?

![](images/1a5ae6d93ba9a74a20461073f686c0a04be5d643c1962f59be433b779b4a0e09.jpg)

Base

Based on the image, the color of the train is not specified. It is only given that there is a train with a number on it......

![](images/7e6bbea01f374dc145e71011c18f6a5677074418659292d86e7f46370a45d117.jpg)

VEPA

Based on the image, the train is colored in a tan or yellowish-brown color. The specific shade of tan or brown is not clearly visible, but the color is similar to what you would typically see on a train. The train has a number 4790 on its side, indicating it's a locomotive, which is a type of train engine. So, the train is a brown or tan color with the number 4790 on its side.

Figure 11. Case 4 on Attribute  
![](images/8e6ae4496d957828adba87b9a2030e414a0540d4184da35212b64b6b72fccc1e.jpg)

![](images/e9dda11bc636f6f68ad5c2023514926937b6d516ab5243bd3c7f93ef62779522.jpg)

Answer based on the image: Which is longer, the baguette or the croissant?

![](images/c76845f60835e053b1a563ed8df8488be76389b5a2e1e3d3b45dcce4a5720fa8.jpg)

Base

Based on the image, the croissant is longer than the baguette. This can be determined by comparing the size of the croissant to the baguette .

![](images/56b58f2d006a332b985447168a975d618c19fc746ca7e529943b3e4f9c9b76be.jpg)

VEPA

Based on the image, the baguette is longer than the croissant. This can be seen by comparing the length of the baguette with the croissant, which is a shorter bread. The baguette is significantly longer than the croissant, indicating it is indeed longer.

Figure 12. Case 5 on Comparison  
![](images/84fd9e049fd3cef2c4d96b0ae55d0b89d1948c4f7b5566e637b7e76181792a7e.jpg)

![](images/03e1f0c57228dd54ecfedd18f5b661b3b3083992c9dad64f206b71712e916e85.jpg)

Answer based on the image: How many guide signs are there in total in the picture?

![](images/a0f5f25af5c8f58ffa999ffe5789a5e2d9386dfb90e47a51878edfbafad0c0ef.jpg)

Base

The sign pointing to the right says \"Norderstedt\" and \"Langenhorn\". There are a total of 2 guide signs in the picture.

![](images/ebfcf4a5a62ad02e8cc56fceb1300b7c549f76c505689e7aff4cf76f8bdae6f4.jpg)

VEPA

The place name written on the sign pointing to the right is \"Norderstedt\". There are a total of 5 guide signs in the picture.

Figure 13. Case 6 on Counting

## A.7. Training Dynamics

This appendix summarizes additional training-time diagnostics used to monitor the stability and convergence of VEPA optimization. Beyond the downstream evaluations in the main paper, we track two classes of signals during training: (i) rollout statistics, measured by the mean generated response length of the auxiliary model (i.e., the auxiliary model’s output length under the (q, e) prompt), and (ii) optimization signals, measured by the sequence-level reward used for sufficiency-driven GRPO. Since both quantities are computed from stochastic rollouts, the curves can be noisy; we therefore visualize the raw traces together with an exponential moving average (EMA, weight 0.85) to highlight the underlying trend.

Rollout length. Figure 14 reports the evolution of the mean response length over training steps for two auxiliary model settings. This metric serves as a lightweight diagnostic of rollout behavior, helping verify that training does not exhibit degenerate length collapse or uncontrolled length inflation while optimizing the evidence policy.

Reward dynamics. Figures 15–17 plot the reward trajectories for different VEPA configurations. Each figure shows both the per-step reward (raw) and its smoothed counterpart (EMA). Overall, the reward trends provide an at-a-glance view of optimization progress and stability under sequence-level reinforcement learning for free-form evidence generation.

![](images/658e2275cf6934ffdbb9bec69810387e343e3e0aa2886bfa686bf19510c89cdd.jpg)

<details>
<summary>line chart</summary>

| Step | Qwen2.5-3b (Original) | Qwen2.5-3b (Smoothed) | Qwen2.5-7b (Original) | Qwen2.5-7b (Smoothed) |
|------|------------------------|------------------------|------------------------|------------------------|
| 0    | ~38                    | ~38                    | ~38                    | ~39                    |
| 50   | ~78                    | ~77                    | ~76                    | ~75                    |
| 100  | ~28                    | ~29                    | ~40                    | ~45                    |
| 150  | ~35                    | ~38                    | ~60                    | ~55                    |
| 200  | ~30                    | ~32                    | ~50                    | ~50                    |
| 220  | ~28                    | ~30                    | ~48                    | ~50                    |
</details>

Figure 14. Evaluation Metrics of Auxiliary Model Training. The figure presents the evolution of the mean response length (y-axis) over training steps (x-axis). The blue and green lines correspond to the two distinct auxiliary model settings evaluated in this experiment. For each setting, the solid darker lines indicate the smoothed values (exponential moving average, weight 0.85), while the faint background lines represent the raw recorded data points.

Reward Score Trend During Training  
![](images/3f9e88aad953e6e057c3190eb1ca5e73d4878823a6d46f3917e9b2062e4b7704.jpg)

<details>
<summary>line chart</summary>

| Step | Original | Smoothed |
| ---- | -------- | -------- |
| 0    | 0.32     | 0.32     |
| 50   | 0.48     | 0.47     |
| 100  | 0.58     | 0.56     |
| 150  | 0.65     | 0.62     |
| 200  | 0.68     | 0.65     |
| 220  | 0.72     | 0.68     |
</details>

Figure 15. Reward Score Trend During Training (5k-VEPA-3b-aux-critic).

Reward Score Trend During Training  
![](images/0597b17b88789703797cb322631de135547979f418bbd999b2e76b8c84f8a75e.jpg)

<details>
<summary>line chart</summary>

| Step | Original | Smoothed |
| ---- | -------- | -------- |
| 0    | 0.28     | 0.30     |
| 50   | 0.50     | 0.49     |
| 100  | 0.58     | 0.56     |
| 150  | 0.62     | 0.61     |
| 200  | 0.68     | 0.65     |
| 220  | 0.70     | 0.64     |
</details>

Figure 16. Reward Score Trend During Training (5k-VEPA-7b-aux-critic).

Reward Score Trend During Training  
![](images/1e5de52126f7838eb52359f7135dd1b8fad87f0f0d21032b61ee5f699c6987da.jpg)

<details>
<summary>line chart</summary>

| Step | Original | Smoothed |
| ---- | -------- | -------- |
| 0    | 0.25     | 0.24     |
| 50   | 0.45     | 0.43     |
| 100  | 0.52     | 0.51     |
| 150  | 0.58     | 0.57     |
| 200  | 0.62     | 0.61     |
| 250  | 0.65     | 0.64     |
| 300  | 0.68     | 0.67     |
| 350  | 0.69     | 0.68     |
| 400  | 0.71     | 0.69     |
| 450  | 0.70     | 0.68     |
</details>

Figure 17. Reward Score Trend During Training (10k-VEPA-7b-aux-critic).

Table 6. Reproducibility experiments across datasets. We report mean ± SD over 3 independent runs for both SFT and VEPA+SFT settings.

<table><tr><td>Dataset</td><td>Benchmark</td><td>SFT (mean±SD)</td><td>VEPA+SFT (mean±SD)</td><td> $\Delta$ </td></tr><tr><td rowspan="2">GQA</td><td>A-OKVQA</td><td>42.02±0.03</td><td>42.77±0.58</td><td>+0.75</td></tr><tr><td>GQA</td><td>63.86±0.84</td><td>64.61±0.10</td><td>+0.75</td></tr><tr><td rowspan="3">A-OKVQA</td><td>A-OKVQA</td><td>59.07±0.74</td><td>67.53±0.85</td><td>+8.46</td></tr><tr><td>ChartQA</td><td>51.13±0.71</td><td>59.12±0.21</td><td>+7.99</td></tr><tr><td>TextVQA</td><td>62.05±0.70</td><td>66.37±0.37</td><td>+4.32</td></tr><tr><td rowspan="3">Mixed</td><td>A-OKVQA</td><td>62.46±0.08</td><td>62.70±0.05</td><td>+0.24</td></tr><tr><td>ChartQA</td><td>70.06±0.27</td><td>70.53±0.19</td><td>+0.47</td></tr><tr><td>TextVQA</td><td>79.63±0.19</td><td>79.96±0.22</td><td>+0.33</td></tr></table>