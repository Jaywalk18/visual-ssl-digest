# Learning from Fine-Grained Visual Discrepancies: Mitigating Multimodal Hallucinations via In-Context Visual Contrastive Optimization

Haolin Deng 1 2 3 Xin Zou 1 2 3 Zhiwei Jin 2 Chen Chen 2 Haonan Lu 2 Xuming Hu 1 3

# Abstract

Multimodal hallucination remains a persistent challenge for Vision-Language Models (VLMs). Standard textual Direct Preference Optimization (DPO) often fails to mitigate it due to a lack of explicit visual supervision. While existing works introduce visual preference DPO by contrasting original images against negative ones, they suffer from a theoretically inconsistent objective caused by partition function mismatches and rely on coarse-grained negatives that could enable shortcut learning. In this work, we propose In-Context Visual Contrastive Optimization (IC-VCO). By placing contrastive images within a shared multiimage context, IC-VCO ensures a mathematically rigorous objective. We further introduce Visual Contrast Distillation (VCDist), an auxiliary reliability-gated regularizer that encourages consistency between multi-image contrastive training and single-image inference. Finally, we propose a contrastive sample editing strategy that generates hard negatives via precise semantic perturbations. Experiments on five benchmarks demonstrate IC-VCO’s best overall performance and the effectiveness of our sample editing strategy. Code and data are available at https://github.com/ OPPO-Mente-Lab/IC-VCO.

# 1. Introduction

Large Vision-Language Models (LVLMs) have demonstrated unprecedented capabilities in bridging visual perception and linguistic reasoning, revolutionizing tasks from visual question answering to embodied agency (Liu et al., 1The Hong Kong University of Science and Technology (Guangzhou) 2OPPO AI Center 3The Hong Kong University of Science and Technology. Correspondence to: Haolin Deng <hldeng028@gmail.com>, Xuming Hu <xuminghu@hkustgz.edu.cn>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

2023; Li et al., 2024; Bai et al., 2025b; Jin et al., 2025). To align these powerful models with human intent and mitigate toxic or untruthful generations, Direct Preference Optimization (DPO) (Rafailov et al., 2023) and reinforcement learning (Stiennon et al., 2020; Schulman et al., 2017; Guo et al., 2025) have emerged as two standard post-training paradigms. DPO, in particular, is favored for its stability and efficiency, as it does not require additional reward model training and policy roll-out during the training process.

Despite these advancements, LVLMs face a unique challenge: Multimodal Hallucination (Bai et al., 2024; Guan et al., 2024; Zou et al., 2025). Unlike textual hallucinations which often arise from factual (Min et al., 2023) or attribution errors (Deng et al., 2024), multimodal hallucinations frequently show a phenomenon called visual neglect (Wu et al., 2025b; Luo et al., 2025; Lu et al., 2022), where the model fails to ground its response in the provided visual input, relying instead on its language priors (Deletang et al., 2024). Standard DPO on textual preference treats the image merely as a static condition, so it often fails to effectively penalize the model for ignoring visual tokens. This limitation underscores the urgent need for optimization objectives that explicitly enforce visual grounding.

A growing line of multimodal preference optimization methods (Wang et al., 2024a; Xie et al., 2024; Yang et al., 2025; Wu et al., 2025b; Liu et al., 2025b) attempts to inject visual supervision into DPO-style training. One common approach is to construct visual preference pairs by fixing the textual response while contrasting a positive image against a negative one, and then optimize the standard DPO objective (Wang et al., 2024a; Yang et al., 2025; Wu et al., 2025b). The visual preference pairs can be constructed in a symmetrical way (Wu et al., 2025b). Specifically, given a standard image-response triplet (m, x, y), where m, x, y denote the input image, textual prompt and correct response respectively, a contrastive counterpart $( m ^ { \prime } , x , y ^ { \prime } )$ is introduced. m′ is curated to differ from m in specific details. Consequently, the faithful response y for m contradicts the visual content of $m ^ { \prime }$ (and vice versa for $y ^ { \prime } )$ . This setup allows for the formulation of two symmetrical preference relations: $r ( m , x , y ) \succ r ( m ^ { \prime } , x , y )$ and $r ( m ^ { \prime } , x , y ^ { \prime } ) \ \succ \ r ( m , x , y ^ { \prime } )$ , where r denotes the implicit DPO reward. While effective for visual grounding, these approaches are limited by two theoretical and practical constraints:

![](images/525bb29f1309c1fe1448e37591fce6d143ebf5a35c7612731feab1bf0449eb46.jpg)  
Figure 1. Schematic comparison of preference optimization frameworks. (a) Standard DPO optimizes textual preferences (y vs. $y ^ { \prime } )$ while treating the image m merely as a static condition, lacking explicit supervision for visual grounding. (b) Visual Preference DPO attempts to introduce visual rejected samples by changing visual context, e.g. swapping the input images (m vs. m′). However, this approach suffers from a theoretical inconsistency: the partition functions $Z ( m , { \bar { x } } )$ and $Z ( m ^ { \prime } , x )$ do not eliminate, leading to a non-rigorous optimization objective. (c) In-Context VCO (Ours) places both the original and contrastive images within a shared context [m, m′] and applies an anchor prompt extension step to specify the target image for preference labels. This design ensures a theoretically rigorous objective by sharing the partition function. A visual contrast distillation objective is introduced to calibrate the standard single-image DPO optimization with multi-image visual contrastive signals during simultaneous training.

❶ Theoretically Inconsistent Objective. While visual preference DPO methods effectively introduce contrastive visual signals to optimize implicit rewards, their objective functions rest on a loose theoretical approximation. By altering the visual contexts of preference pairs, the partition functions of the reference policy fail to cancel out. This results in a residual partition function ratio that persists as an intractable bias, rendering the optimization objective theoretically inconsistent with the original DPO formulation.   
❷ Coarse-Grained Negatives. Recent works (Xie et al., 2024; Wu et al., 2025b; Liu et al., 2025b) typically construct contrastive images $m ^ { \prime }$ via image retrieval or text-to-image synthesis. These images often exhibit distinct stylistic differences or noticeable semantic gaps compared to the original inputs. Such substantial deviations result in obvious and widespread inconsistencies between the image and contrastive response, making the rejected samples $( m , x , y ^ { \prime } )$ and $( m ^ { \prime } , x , y )$ trivial negatives: the model can minimize the DPO loss easily by exploiting the coarse-grained discrepancies, rather than learning the fine-grained visual facts.

In this work, we propose In-Context Visual Contrastive

Optimization (IC-VCO), a framework which restructures visual alignment by placing contrastive images within a shared multi-image context. By instructing the model to distinguish and respond based on specific targeted images within this context, we ensure that the partition functions of preference pairs remain identical, strictly adhering to the theoretical consistency of DPO. Since IC-VCO constructs preference supervision in a multi-image context, while standard LVLM inference is performed with a single image, we further introduce Visual Contrast Distillation (VCDist) as an auxiliary consistency regularizer. VCDist uses the multiimage preference distribution as a soft reference to calibrate the single-image branch, encouraging the single-image policy to remain compatible with the contrastive training signal, thereby reducing the train-inference context gap.

To further address the challenge of trivial negatives prevalent in existing methods, we introduce a Contrastive Sample Editing strategy. Unlike previous approaches relying on retrieval or global synthesis which often introduce coarse stylistic discrepancies that facilitate shortcut learning, we employ a targeted editing pipeline. We perform precise, localized modifications on original images to generate high-quality hard negatives. These samples maintain strict stylistic consistency with the original visual distribution while embodying specific semantic contradictions, thereby compelling the model to develop fine-grained visual grounding capabilities rather than exploiting low-level shortcuts. Comprehensive experiments on five diverse benchmarks demonstrate that IC-VCO provides a stronger multimodal preference optimization objective than existing baselines. The results also show that contrastive edited samples serve as broadly useful hard negatives, improving multiple preference-optimization methods beyond IC-VCO.

# 2. Preliminaries

# 2.1. Derivation of DPO Objective

DPO derives an analytical reward formulation from the generalized reinforcement learning objective:

$$
r (x, y) = \beta \log \frac {\pi_ {\theta} (y \mid x)}{\pi_ {\mathrm{ref}} (y \mid x)} + \beta \log Z (x), \tag {1}
$$

$$
Z (x) = \sum_ {y} \pi_ {\text { ref }} (y \mid x) \exp \left(\frac {1}{\beta} r (x, y)\right), \tag {2}
$$

where $Z ( x )$ is a partition function dependent on the input context and reference policy. The value of $Z ( x )$ is intractable due to the implicit reward formulation. DPO assumes the reward model to be a Bradley-Terry model:

$$
\begin{array}{l} p (y _ {w} \succ y _ {l}) = \frac {\exp (r (x , y _ {w}))}{\exp (r (x , y _ {w})) + \exp (r (x , y _ {l}))} \\ = \sigma \left(r (x, y _ {w}) - r (x, y _ {l})\right) \\ = \sigma [ \beta (\log \frac {\pi_ {\theta} (y _ {w} \mid x)}{\pi_ {\mathrm{ref}} (y _ {w} \mid x)} + \log Z (x)) \tag {3} \\ \left. - \beta (\log \frac {\pi_ {\theta} (y _ {l} \mid x)}{\pi_ {\mathrm{ref}} (y _ {l} \mid x)} + \log Z (x)) \right]. \\ \end{array}
$$

Eq. 3 shows that the chosen reward and rejected reward share the same value $Z ( x )$ , which can be eliminated. Thus, the DPO objective that tries to maximize the reward margin over preference pairs becomes:

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{DPO}} = - \mathbb {E} _ {(x, y _ {w}, y _ {l}) \sim \mathcal {D}} \\ \left[ \log \sigma \left(\beta \log \frac {\pi_ {\theta} \left(y _ {w} \mid x\right)}{\pi_ {\text { ref }} \left(y _ {w} \mid x\right)} - \beta \log \frac {\pi_ {\theta} \left(y _ {l} \mid x\right)}{\pi_ {\text { ref }} \left(y _ {l} \mid x\right)}\right) \right]. \tag {4} \\ \end{array}
$$

# 2.2. Visual Preference DPO

Given an image m, textual prompt x, and a response pair $( y , y ^ { \prime } )$ , where $r ( m , x , y ) \succ r ( m , x , y ^ { \prime } )$ , visual preference DPO (Wang et al., 2024a; Yang et al., 2025; Wu et al., 2025b) leverages a negative image m′ to optimize visual preference $r ( m , x , y ) \succ r ( m ^ { \prime } , x , y )$ with DPO loss:

$$
\begin{array}{l} \mathcal {L} _ {\text { VisDPO }} = - \mathbb {E} _ {(m, m ^ {\prime}, x, y) \sim \mathcal {D}} \\ \left[ \log \sigma \left(\beta \log \frac {\pi_ {\theta} (y \mid m , x)}{\pi_ {\text { ref }} (y \mid m , x)} - \beta \log \frac {\pi_ {\theta} (y \mid m ^ {\prime} , x)}{\pi_ {\text { ref }} (y \mid m ^ {\prime} , x)}\right) \right]. \tag {5} \\ \end{array}
$$

If $y ^ { \prime }$ is the chosen response of $m ^ { \prime } ,$ an extra visual preference pair $r ( m ^ { \prime } , x , y ^ { \prime } ) \succ r ( m , x , y ^ { \prime } )$ can be leveraged for a symmetrical loss (Wu et al., 2025b):

$$
\begin{array}{l} \mathcal {L} _ {\text { VisDPO }} ^ {\prime} = - \mathbb {E} _ {(m, m ^ {\prime}, x, y, y ^ {\prime}) \sim \mathcal {D}} \\ \left[ \log \sigma \left(\beta \log \frac {\pi_ {\theta} (y ^ {\prime} \mid m ^ {\prime} , x)}{\pi_ {\mathrm{ref}} (y ^ {\prime} \mid m ^ {\prime} , x)} - \beta \log \frac {\pi_ {\theta} (y ^ {\prime} \mid m , x)}{\pi_ {\mathrm{ref}} (y ^ {\prime} \mid m , x)}\right) \right]. \tag {6} \\ \end{array}
$$

The two losses can be jointly optimized.

Issue: Theoretically Inconsistent Objective. Given a visual preference pair $r ( m , x , y ) \succ r ( m ^ { \prime } , x , y )$ , following Eq. 3 and 4, we can derive the theoretical objective:

$$
\begin{array}{l} \mathcal {L} _ {\text { VisDPO }} ^ {*} = - \mathbb {E} _ {(m, m ^ {\prime}, x, y, y ^ {\prime}) \sim \mathcal {D}} \left[ \log \sigma \left(\beta \log \frac {\pi_ {\theta} (y \mid m , x)}{\pi_ {\text { ref }} (y \mid m , x)} \right. \right. \\ \left. - \beta \log \frac {\pi_ {\theta} (y \mid m ^ {\prime} , x)}{\pi_ {\text { ref }} (y \mid m ^ {\prime} , x)} + \beta \log \frac {Z (m , x)}{Z (m ^ {\prime} , x)}\right) \Bigg ]. \tag {7} \\ \end{array}
$$

By generalizing Eq. 2, we can get:

$$
\log \frac {Z (m , x)}{Z (m ^ {\prime} , x)} = \log \frac {\sum_ {y} \pi_ {\text { ref }} (y \mid m , x) \exp \left(\frac {1}{\beta} r (m , x , y)\right)}{\sum_ {y} \pi_ {\text { ref }} (y \mid m ^ {\prime} , x) \exp \left(\frac {1}{\beta} r (m ^ {\prime} , x , y)\right)}. \tag {8}
$$

Eq. 8 cannot be eliminated since $\pi _ { \mathrm { r e f } } ( \cdot | m , x )$ and $\pi _ { \mathrm { r e f } } ( \cdot | m ^ { \prime } , x )$ are different distributions. Eq. 5 and Eq. 6 essentially ignore this residual ratio, thereby optimizing a biased proxy objective. This residual term acts as a uncontrollable offset that shifts the implicit decision boundary arbitrarily for each training sample, ultimately restricting the optimization performance.

# 3. In-Context VCO

To address the theoretical inconsistency in Visual Preference DPO formulations, where distinct visual inputs lead to mismatched partition functions, we introduce a unified In-Context Visual Contrastive Optimization (IC-VCO) framework. Instead of processing images in isolation, we construct a shared multi-image context $M$ that encapsulates both the original image m and the contrastive image $m ^ { \prime } .$ . We define M as a sequence of images $M = [ m , m ^ { \prime } ]$ , containing both the original image m and the contrastive negative $m ^ { \prime }$ , and feed them sequentially into the VLM. Given this multi-image context, the original textual prompt x becomes ambiguous because the model cannot implicitly discern which image to ground its response on. To resolve this, we introduce an anchor prompt extension strategy to direct the model’s attention to a specified target image within M.

Let xˆ and $\hat { x } ^ { \prime }$ denote the extended prompt targeting the original image m and contrastive image $m ^ { \prime }$ respectively. In practice, we explicitly append a positional anchor instruction to x which is one of “respond based on the first image” or “respond based on the second image” depending on the image order in M . To eliminate position bias in optimization, the image order is randomized for each sample. Based on this, we construct two symmetrical preference pairs: $r ( M , { \hat { x } } , y ) \succ r ( M , { \hat { x } } , y ^ { \prime } )$ and $r ( M , \hat { x } ^ { \prime } , y ^ { \prime } ) \succ r ( M , \hat { x } ^ { \prime } , y )$ .

The multi-image objective for $r ( M , { \hat { x } } , y ) \succ r ( M , { \hat { x } } , y ^ { \prime } )$ is:

$$
\begin{array}{l} p _ {\text { multi }} = \sigma \left(\beta \log \frac {\pi_ {\theta} (y \mid M , \hat {x})}{\pi_ {\text { ref }} (y \mid M , \hat {x})} - \beta \log \frac {\pi_ {\theta} (y ^ {\prime} \mid M , \hat {x})}{\pi_ {\text { ref }} (y ^ {\prime} \mid M , \hat {x})}\right), \\ \mathcal {L} _ {\text { Multi }} = - \mathbb {E} _ {(M, \hat {x}, y, y ^ {\prime}) \sim \mathcal {D}} \left[ \log p _ {\text { multi }} \right]. \tag {9} \\ \end{array}
$$

Here $p _ { \mathrm { m u l t i } }$ is the chosen-response win probability under the shared context $( M , { \hat { x } } )$ . Since the chosen and rejected responses share the same condition, the partition function $Z ( M , { \hat { x } } )$ cancels out, yielding a theory-consistent objective.

Following prior multimodal preference optimization methods (Wang et al., 2024a; Yang et al., 2025; Wu et al., 2025b; Liu et al., 2025b), we also use single-image DPO:

$$
\begin{array}{l} p _ {\text { single }} = \sigma \left(\beta \log \frac {\pi_ {\theta} (y \mid m , x)}{\pi_ {\text { ref }} (y \mid m , x)} - \beta \log \frac {\pi_ {\theta} (y ^ {\prime} \mid m , x)}{\pi_ {\text { ref }} (y ^ {\prime} \mid m , x)}\right), \\ \mathcal {L} _ {\text { Single }} = - \mathbb {E} _ {(m, x, y, y ^ {\prime}) \sim \mathcal {D}} \left[ \log p _ {\text { single }} \right]. \tag {10} \\ \end{array}
$$

Visual Contrast Distillation (VCDist). IC-VCO constructs visual preference in a multi-image context M , whereas standard LVLM inference operates with a single image. To reduce the train-inference context gap, we introduce VCDist as an auxiliary consistency regularizer by using the multi-image preference distribution $p _ { \mathrm { m u l t i } }$ as a soft reference to calibrate the single-image distribution $\boldsymbol { p } _ { \mathrm { s i n g l e } }$ . This also encourages the single-image branch to absorb useful contrastive supervision from the multi-image context.

To ensure rigorous alignment, we introduce a dual-gating mechanism that filters the distillation signal based on correctness and relative confidence: a correctness gate filters out unreliable teacher signals $( i . e . , p _ { \mathrm { m u l t i } } > 0 . 5 )$ , while a confidence gate activates distillation only when the student’s confidence falls below the teacher’s $( i . e . , p _ { \mathrm { s i n g l e } } < p _ { \mathrm { m u l t i } } )$ t o prevent reverse penalty. Furthermore, to stabilize the optimization, we apply a stop-gradient operation to the teacher’s logits. We formulate the VCDist objective as:

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{VCDist}} = - \mathbb {E} _ {(M, m, x, y, y ^ {\prime}) \sim \mathcal {D}} \left[ \mathbb {I} \left(p _ {\text { multi }} > 0. 5 \wedge p _ {\text { single }} <   \operatorname{sg} (p _ {\text { multi }})\right) \right. \\ \left. \cdot \left(\operatorname{sg} (p _ {\text { multi }}) \log p _ {\text { single }} + (1 - \operatorname{sg} (p _ {\text { multi }})) \log (1 - p _ {\text { single }})\right) \right], \tag {11} \\ \end{array}
$$

where $\mathbb { I } ( \cdot )$ is the indicator function, and $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator.

Following previous works (Wang et al., 2024a; Yang et al., 2025; Liu et al., 2025b), we also leverage anchor losses to restrain the optimization from decreasing the chosen likelihoods compared to the reference policy. Since IC-VCO contains both single-image and multi-image branches, we define the corresponding anchor terms separately:

$$
\mathcal {L} _ {\text { SingleAnc }} = - \mathbb {E} _ {(m, x, y) \sim \mathcal {D}} \left[ \log \sigma \left(\beta \log \frac {\pi_ {\theta} (y \mid m , x)}{\pi_ {\text { ref }} (y \mid m , x)}\right) \right], \tag {12}
$$

$$
\mathcal {L} _ {\text { MultiAnc }} = - \mathbb {E} _ {(M, \hat {x}, y) \sim \mathcal {D}} \left[ \log \sigma \left(\beta \log \frac {\pi_ {\theta} (y \mid M , \hat {x})}{\pi_ {\text { ref }} (y \mid M , \hat {x})}\right) \right]. \tag {13}
$$

The IC-VCO objective for the original image m is defined by averaging the single-image and multi-image branches:

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{IC-VCO}} = \frac {1}{2} \left[ \underbrace {\lambda_ {1} \left(\mathcal {L} _ {\text {Multi}} + \eta_ {1} \mathcal {L} _ {\text {MultiAnc}}\right)} _ {\text {multi - image branch}} \right. \tag {14} \\ \left. + \underbrace {\lambda_ {2} \left(\mathcal {L} _ {\text {Single}} + \eta_ {2} \mathcal {L} _ {\text {SingleAnc}}\right) + \gamma \mathcal {L} _ {\text {VCDist}}} _ {\text {single - image branch}} \right], \\ \end{array}
$$

where $\lambda _ { 1 } , \lambda _ { 2 } , \eta _ { 1 } , \eta _ { 2 }$ , and γ are hyper-parameters. The anchor weights are coupled with corresponding preference branches, so that the $\mathcal { L } _ { \mathrm { S i n g l e A n c } }$ is scaled together with $\mathcal { L } _ { \mathrm { S i n g l e } }$ , while $\mathcal { L } _ { \mathrm { M u l t i A n c } }$ is scaled together with $\mathcal { L } _ { \mathrm { M u l i } }$ .

Fine-grained Token-level Preference. To further improve the single-image policy’s sensitivity to fine-grained visual discrepancies, we introduce a token mask for the edited response pairs. Specifically, for response tokens that describe the edited visual evidence, we compute the singleimage preference score only over the masked tokens. This token-level preference is applied to the single-image branch, while the multi-image branch still operates on the full response to preserve holistic visual reasoning.

Symmetrical Preference Optimization. Eq. 14 represents the objective when taking m as target image, y as chosen response, and $y ^ { \prime }$ as rejected response. Since the contrastive sample $( m ^ { \prime } , x , y ^ { \prime } )$ is symmetrically constructed, following prior works (Wu et al., 2025b; Liu et al., 2025b), we can define a symmetrical objective $\mathcal { L } _ { \mathrm { { I C - V C O } } } ^ { \prime }$ by taking m′ as the target image and $y ^ { \prime }$ as the chosen response, similar to Eq. 6. The final objective can be expressed as:

$$
\mathcal {L} _ {\text { Total }} = \mathcal {L} _ {\mathrm{IC-VCO}} + \mathcal {L} _ {\mathrm{IC-VCO}} ^ {\prime}. \tag {15}
$$

# 4. Contrastive Sample Editing

To fully activate the theoretical potential of IC-VCO, we prioritize strict distributional alignment to prevent shortcut learning (Geirhos et al., 2020). From a fine-grained causal perspective (Pearl, 2009; Scholkopf et al. ¨ , 2021), we decompose the image generation factors into three components: the target semantic concept $c _ { t g t }$ (the focus of the preference pair), the surrounding semantic context $C _ { c t x }$ (other visual entities), and environmental factors U (e.g., style, lighting). Thus, an image is modeled as $\boldsymbol { m } = f ( c _ { t g t } , C _ { c t x } , U )$ .

![](images/c87a6d396a427204f02f0026f73014a85b592df32f5c5d89dcb17d8ade55e7a1.jpg)  
Figure 2. Qualitative comparison of contrastive images. Synthetic baselines (yellow) exhibit global stylistic shifts, acting as coarsegrained negatives prone to shortcut learning. In contrast, our Contrastive Editing (green) performs surgical, localized interventions while better preserving the visual context, yielding fine-grained hard negatives that compel rigorous visual discrimination.

Existing baselines typically construct negatives via global resampling processes that fail to preserve instance-level consistency. Synthesis-based approaches (Xie et al., 2024; Zhang et al., 2024; Wu et al., 2025b; Liu et al., 2025b), which utilize text-to-image diffusion models (Rombach et al., 2022), suffer from the under-specification problem (D’Amour et al., 2022): since textual descriptions cannot enumerate all background details, the model generates a new scene from the learned distribution, inadvertently altering the context. Retrieval-based approaches (Liu et al., 2025b) also introduce an independent realization of the scene by picking a distinct image m′ from a database. Both paradigms result in coarse-grained negatives exhibiting global semantic drift: $P ( C _ { c t x } , U | m ) \ \neq \ P ( C _ { c t x } , U | m ^ { \prime } )$ . Crucially, due to the simplicity bias of deep neural networks (Shah et al., 2020), these global distributional shifts provide a highly salient discriminative signal. Since distinguishing samples based on global style or background artifacts is easier than verifying fine-grained visual discrepancies, the model is more likely to collapse into a shortcut solution: rejecting m′ simply based on environmental inconsistencies rather than learning the target concept grounding.

In contrast, we define hard negatives as valid contrastive samples generated via surgical intervention: we aim to perform a precise operation $d o ( c _ { t g t } \to c _ { t g t } ^ { \prime } )$ while encouraging invariance on the semantic context and environmental factors: $\{ C _ { c t x } , U \} _ { m } \approx \{ C _ { c t x } , U \} _ { m ^ { \prime } }$ . This formulation ensures that the preference label primarily hinges on the

Table 1. Statistics of the contrastive sample editing dataset. 

<table><tr><td>Hallucination Type</td><td>Scenario A (Realization)</td><td>Scenario B (Injection)</td><td>Total</td></tr><tr><td>Attribute</td><td>7,560</td><td>3,072</td><td>10,632</td></tr><tr><td>Existence</td><td>7,463</td><td>530</td><td>7,993</td></tr><tr><td>Relation</td><td>715</td><td>113</td><td>828</td></tr><tr><td>Total</td><td>15,738</td><td>3,715</td><td>19,453</td></tr></table>

![](images/14f0be1e5b10c8d3c57181f15486ab5555cd48fceccb1105b8c61cbf4721c82c.jpg)

<details>
<summary>area</summary>

| CLIP chosen-rejected image similarity score | SymMPO-Synthetic | Contrastive Sample Editing (Ours) |
| ------------------------------------------ | ---------------- | --------------------------------- |
| 72.88                                      | 0.032            | 0.000                             |
| 94.72                                      | 0.000            | 0.129                             |
</details>

Figure 3. CLIP-based image pair similarity distribution.

fine-grained visual discrepancies of the target concept. To approximate this theoretical objective, we propose a Contrastive Sample Editing Framework utilizing a targeted edit pipeline to generate high-quality contrastive samples.

Editing Pipeline. Given a seed tuple $( m , x , y , y ^ { \prime } )$ , we use QwenVL-Plus (Bai et al., 2025a) as an expert planner to generate an executable editing instruction T . The planner handles two cases: hallucination realization, where an explicit hallucinated detail in $y ^ { \prime }$ is made true in the edited image $m ^ { \prime }$ , and hallucination injection, where a visual detail supporting y is minimally contradicted when $y ^ { \prime }$ does not provide a localized hallucination target. The target edit is categorized as existence, attribute, or relation. We then construct a contrastive response $y _ { \mathrm { n e w } } ^ { \prime }$ by minimally rewriting the original chosen response y according to $\tau _ { \ast }$ , so that $( m ^ { \prime } , y _ { \mathrm { n e w } } ^ { \prime } )$ becomes the positive pair and $( m ^ { \prime } , y )$ becomes the negative pair while preserving the linguistic style of $y .$ . The tokenlevel differences between $y$ and $y _ { \mathrm { n e w } } ^ { \prime }$ are also recorded as fine-grained masks for token-level preference scoring. Next, Qwen-Image-Edit (Wu et al., 2025a) applies $\tau$ to m with a reversible padding procedure that preserves the original aspect ratio and maps the edited output back to the original resolution, keeping non-target regions aligned. Finally, QwenVL-Plus verifies whether $m ^ { \prime }$ faithfully implements $\tau$ and rectifies minor textual mismatches in $\dot { y } _ { \mathrm { n e w } } ^ { \prime }$ ; failed edits or samples with unintended structural changes are discarded. We provide full pipeline details in Appendix B.

Data Source. We use the SymMPO dataset (Liu et al., 2025b) as our seed corpus. This dataset comprises approximately 21.4k samples, originally aggregated from VQA v2 (Goyal et al., 2017), MSCOCO (Lin et al., 2014), and TextVQA (Singh et al., 2019) via the TPO dataset (He et al., 2024). Table 1 shows the core statistics of the contrastive sample editing dataset. We finally construct 19,453 contrastive samples, with an overall success rate of 91%.

Quality Inspection. Figure 2 compares synthetic and edited negatives against the original images. We also use CLIP (Radford et al., 2021) to calculate pair-wise image similarity scores. Figure 3 shows that edited images are more visually similar to the chosen images than synthetic images (mean CLIP image-similarity score: 94.72 vs 72.88). This further validates that our approach is effective at constructing hard negatives with fine-grained visual dependencies.

# 5. Experiment

# 5.1. Experimental Setup

Training Data Statistics. We use the SymMPO-synthetic dataset as our baseline training set. It contains 21.4k symmetrical image-text samples $( m , m ^ { \prime } , x , y , y ^ { \prime } )$ where the contrastive image $m ^ { \prime }$ is generated via text-to-image synthesis. We also use our edited contrastive samples as a comparative training set with 19k symmetrical samples.

Baselines & Architectures. We implement our method on two representative open-source VLMs: LLaVA-NeXT-Interleave-Qwen-7B (Li et al., 2024) and LLaVA-OneVision-Qwen2-7B (Li et al., 2025a). We compare IC-VCO against the base models and leading multimodal preference alignment methods, including mDPO (Wang et al., 2024a), V-DPO (Xie et al., 2024), S-VCO (Wu et al., 2025b), and SymMPO (Liu et al., 2025b)1.

Evaluation Benchmarks. We employ five diverse benchmarks to comprehensively evaluate performance across different hallucination domains: 1) HallusionBench (Guan et al., 2024) assesses language hallucination and visual illusion capabilities. 2) AMBER (Wang et al., 2023) evaluates fine-grained hallucinations, specifically covering object existence, attributes, and spatial relations. 3) CRPE (Wang et al., 2024c) quantitatively tests object recognition and relation comprehension. 4) R-Bench (Li et al., 2025b) measures the model’s robustness against various image corruptions and distortions. 5) BLINK (Fu et al., 2024) evaluates core visual perception abilities across 14 diverse computer vision tasks. To ensure the reproducibility of evaluation results, we use VLMEvalKit (Duan et al., 2024), an open-source evaluation toolkit to perform the standardized evaluations.

See Appendix C for implementation details.

# 5.2. Main Results

As presented in Tables 2 and 3, IC-VCO achieves the best overall score among all compared preference-optimization methods under both contrastive sample sources and both backbone models. On LLaVA-NeXT-Interleave, IC-VCO outperforms the strongest baseline by 1.19 points with synthetic samples and by 1.24 points with contrastive edited samples. On LLaVA-OneVision, IC-VCO surpasses the strongest baseline by 0.26 points and 0.48 points under the synthetic and edited settings, respectively.

The results also verify the general effectiveness of Contrastive Sample Editing. Replacing synthetic samples with edited samples shows improvements across baselines and base models. On LLaVA-NeXT-Interleave, the overall gains for the baselines range from 0.08 to 0.61 points, and IC-VCO improves from 62.83 to 63.35. On LLaVA-OneVision, the baseline gains range from 0.25 to 1.23 points, and IC-VCO improves from 66.26 to 66.82. These consistent improvements validate the advantage of fine-grained editing.

At the metric level, IC-VCO is particularly strong on attribute and existence-oriented grounding. It obtains the best AMBER-Attr score in all settings and also achieves strong AMBER-Exist and CRPE-Exist results. It also consistently improves BLINK, suggesting better general visual perception. However, relation-centric metrics are more mixed: IC-VCO does not always outperform all baselines on AMBER-Rel or CRPE-Rel. This may be partly due to the current edited dataset being skewed toward attribute and existence edits, with relation edits accounting for only 828 of 19,453 samples. We leave richer relation-oriented editing and supervision as future work.

Table 2. Experimental results on LLaVA-NeXT-Interleave-Qwen-7B. We compare the different methods using two distinct contrastive sample sources: Synthetic and Contrastive Sample Editing. The overall score denotes the macro-average across all benchmarks. 

<table><tr><td rowspan="2">Contrastive Sample Source</td><td rowspan="2">Approach</td><td rowspan="2">Overall Score</td><td colspan="3">HallusionBench</td><td colspan="3">AMBER</td><td colspan="2">CRPE</td><td colspan="2">R-Bench</td><td>BLINK</td></tr><tr><td>aAcc</td><td>fAcc</td><td>qAcc</td><td>Attr</td><td>Exist</td><td>Rel</td><td>Exist</td><td>Rel</td><td>Dis</td><td>Ref</td><td>Score</td></tr><tr><td></td><td>LLaVA-NeXT-Interleave-Qwen-7B</td><td>59.14</td><td>55.59</td><td>25.76</td><td>25.90</td><td>79.97</td><td>89.03</td><td>74.51</td><td>92.01</td><td>60.20</td><td>55.96</td><td>59.11</td><td>45.13</td></tr><tr><td rowspan="6">Synthetic (Liu et al., 2025b)</td><td>DPO(Rafailov et al., 2023)</td><td>60.32</td><td>58.69</td><td>26.10</td><td>27.93</td><td>76.92</td><td>86.11</td><td>78.71</td><td>89.55</td><td>65.38</td><td>58.99</td><td>63.16</td><td>44.93</td></tr><tr><td>mDPO(Wang et al., 2024a)</td><td>61.64</td><td>61.51</td><td>30.06</td><td>31.43</td><td>80.27</td><td>88.36</td><td>77.16</td><td>91.79</td><td>64.84</td><td>60.40</td><td>63.77</td><td>44.87</td></tr><tr><td>V-DPO(Xie et al., 2024)</td><td>60.15</td><td>58.26</td><td>26.39</td><td>27.49</td><td>76.48</td><td>86.21</td><td>78.47</td><td>89.41</td><td>65.22</td><td>58.59</td><td>62.15</td><td>45.30</td></tr><tr><td>S-VCO(Wu et al., 2025b)</td><td>60.81</td><td>58.68</td><td>27.17</td><td>28.79</td><td>77.88</td><td>86.17</td><td>79.21</td><td>89.92</td><td>66.00</td><td>59.80</td><td>63.36</td><td>45.19</td></tr><tr><td>SymMPO(Liu et al., 2025b)</td><td>61.50</td><td>60.79</td><td>29.86</td><td>31.23</td><td>80.41</td><td>88.63</td><td>76.54</td><td>91.83</td><td>64.43</td><td>60.20</td><td>63.77</td><td>44.88</td></tr><tr><td>IC-VCO (Ours)</td><td>62.83</td><td>61.94</td><td>30.82</td><td>31.55</td><td>81.81</td><td>90.48</td><td>75.56</td><td>93.16</td><td>65.63</td><td>59.70</td><td>63.87</td><td>48.93</td></tr><tr><td rowspan="6">Contrastive Sample Editing (Ours)</td><td>DPO(Rafailov et al., 2023)</td><td>60.40</td><td>60.46</td><td>27.17</td><td>30.11</td><td>78.16</td><td>92.47</td><td>76.20</td><td>89.35</td><td>63.31</td><td>57.58</td><td>61.34</td><td>44.66</td></tr><tr><td>mDPO(Wang et al., 2024a)</td><td>62.02</td><td>60.25</td><td>29.48</td><td>30.99</td><td>80.31</td><td>92.55</td><td>74.64</td><td>92.27</td><td>65.36</td><td>60.00</td><td>65.79</td><td>45.66</td></tr><tr><td>V-DPO(Xie et al., 2024)</td><td>60.38</td><td>59.94</td><td>26.88</td><td>30.99</td><td>77.77</td><td>91.98</td><td>76.50</td><td>89.20</td><td>62.88</td><td>57.98</td><td>61.34</td><td>44.87</td></tr><tr><td>S-VCO(Wu et al., 2025b)</td><td>61.41</td><td>58.25</td><td>26.88</td><td>29.45</td><td>79.72</td><td>91.41</td><td>79.15</td><td>92.58</td><td>65.89</td><td>58.99</td><td>63.36</td><td>45.03</td></tr><tr><td>SymMPO(Liu et al., 2025b)</td><td>62.11</td><td>60.57</td><td>29.77</td><td>31.65</td><td>80.39</td><td>92.89</td><td>74.52</td><td>92.47</td><td>65.58</td><td>61.01</td><td>65.18</td><td>45.19</td></tr><tr><td>IC-VCO (Ours)</td><td>63.35</td><td>63.51</td><td>33.34</td><td>33.07</td><td>82.24</td><td>92.73</td><td>70.47</td><td>94.15</td><td>64.88</td><td>60.71</td><td>64.67</td><td>49.44</td></tr></table>

Table 3. Experimental results on LLaVA-OneVision-Qwen2-7B. 

<table><tr><td rowspan="2">Contrastive Sample Source</td><td rowspan="2">Approach</td><td rowspan="2">Overall Score</td><td colspan="3">HallusionBench</td><td colspan="3">AMBER</td><td colspan="2">CRPE</td><td colspan="2">R-Bench</td><td>BLINK</td></tr><tr><td>aAcc</td><td>fAcc</td><td>qAcc</td><td>Attr</td><td>Exist</td><td>Rel</td><td>Exist</td><td>Rel</td><td>Dis</td><td>Ref</td><td>Score</td></tr><tr><td></td><td>LLaVA-OneVision-Qwen2-7B</td><td>62.46</td><td>53.84</td><td>25.04</td><td>24.95</td><td>84.05</td><td>91.67</td><td>75.98</td><td>94.52</td><td>65.26</td><td>66.16</td><td>71.96</td><td>44.82</td></tr><tr><td rowspan="6">Synthetic (Liu et al., 2025b)</td><td>DPO(Rafailov et al., 2023)</td><td>64.97</td><td>57.95</td><td>31.01</td><td>31.67</td><td>87.98</td><td>92.37</td><td>82.85</td><td>95.43</td><td>71.76</td><td>63.23</td><td>69.03</td><td>47.20</td></tr><tr><td>mDPO(Wang et al., 2024a)</td><td>65.70</td><td>63.21</td><td>35.93</td><td>36.94</td><td>87.12</td><td>95.68</td><td>74.80</td><td>95.54</td><td>71.24</td><td>63.03</td><td>70.25</td><td>47.25</td></tr><tr><td>V-DPO(Xie et al., 2024)</td><td>65.08</td><td>58.16</td><td>31.01</td><td>31.67</td><td>88.01</td><td>92.02</td><td>83.69</td><td>95.45</td><td>71.71</td><td>63.64</td><td>69.44</td><td>47.09</td></tr><tr><td>S-VCO(Wu et al., 2025b)</td><td>66.00</td><td>62.26</td><td>36.51</td><td>36.94</td><td>88.04</td><td>92.94</td><td>81.71</td><td>95.30</td><td>71.18</td><td>63.44</td><td>69.64</td><td>47.41</td></tr><tr><td>SymMPO(Liu et al., 2025b)</td><td>65.88</td><td>63.10</td><td>36.79</td><td>36.50</td><td>87.21</td><td>95.82</td><td>75.46</td><td>95.67</td><td>71.45</td><td>64.24</td><td>70.45</td><td>46.83</td></tr><tr><td>IC-VCO (Ours)</td><td>66.26</td><td>62.15</td><td>35.55</td><td>36.26</td><td>88.06</td><td>95.98</td><td>73.20</td><td>95.94</td><td>71.75</td><td>65.25</td><td>71.05</td><td>48.92</td></tr><tr><td rowspan="6">Contrastive Sample Editing (Ours)</td><td>DPO(Rafailov et al., 2023)</td><td>66.08</td><td>63.34</td><td>37.06</td><td>37.88</td><td>86.01</td><td>94.24</td><td>80.99</td><td>94.94</td><td>71.53</td><td>64.05</td><td>68.43</td><td>47.75</td></tr><tr><td>mDPO(Wang et al., 2024a)</td><td>66.24</td><td>63.90</td><td>36.36</td><td>37.45</td><td>86.11</td><td>96.28</td><td>76.39</td><td>95.61</td><td>71.93</td><td>64.50</td><td>70.10</td><td>47.99</td></tr><tr><td>V-DPO(Xie et al., 2024)</td><td>66.31</td><td>63.90</td><td>37.80</td><td>38.55</td><td>86.42</td><td>94.65</td><td>81.56</td><td>95.46</td><td>72.19</td><td>64.70</td><td>68.88</td><td>46.63</td></tr><tr><td>S-VCO(Wu et al., 2025b)</td><td>66.34</td><td>63.27</td><td>36.36</td><td>37.45</td><td>86.36</td><td>95.36</td><td>79.10</td><td>95.48</td><td>72.04</td><td>64.30</td><td>69.69</td><td>48.31</td></tr><tr><td>SymMPO(Liu et al., 2025b)</td><td>66.13</td><td>64.00</td><td>36.07</td><td>37.67</td><td>86.06</td><td>96.24</td><td>76.03</td><td>95.48</td><td>71.83</td><td>63.89</td><td>70.30</td><td>47.89</td></tr><tr><td>IC-VCO (Ours)</td><td>66.82</td><td>62.54</td><td>35.50</td><td>35.98</td><td>88.00</td><td>97.12</td><td>73.96</td><td>96.80</td><td>72.52</td><td>66.68</td><td>72.08</td><td>49.01</td></tr></table>

# 5.3. Ablation Study

To validate the design of IC-VCO, we analyze component contributions in Table 4 and training diagnostics in Figure 4 on LLaVA-NeXT-Interleave-Qwen-7B. Compared to the full IC-VCO model on edited samples, removing the single-image branch decreases the overall score to 62.69, showing that single-image DPO is necessary for maintaining inference-time compatibility. Removing $\mathcal { L } _ { \mathrm { V C D i s t } }$ reduces the score to 63.04, indicating that calibrating the single-image branch with the multi-image preference signal is beneficial. Removing the token mask also hurts the overall score, which suggests that focusing the single-image preference signal on edited evidence tokens improves the effectiveness of fine-grained supervision. Removing the anchor loss ${ \mathcal { L } } _ { \mathrm { A n c } }$ reduces overall score to 61.15, confirming that the anchor loss is important for stabilizing preference optimization by preventing the decline of chosen likelihoods.

Multi-Image Branch and VCDist Signal. Figure 4 (a) and (b) show that the multi-image branch consistently achieves higher reward accuracy than the single-image branch. This validates the premise of VCDist: explicit visual comparison in the multi-image context provides a stronger preference signal. Figure 4 (c) further examines whether this stronger branch provides a reliable distillation target. The valid-teacher ratio measures the fraction of training samples for which the teacher passes the correctness gate. The valid KL measures the Bernoulli KL divergence between the teacher and student preference distributions on these valid samples. A stable valid-teacher ratio indicates that VCDist can continuously access a sufficient amount of teacher-aligned supervision, while the low valid KL suggests that the student remains close to the teacher distribution and the distillation target is well-conditioned rather than noisy or unstable. Together, these trends indicate that the VCDist signal is both active and reliable during training.

Robustness of VCDist Design. We further examine the stop-gradient and dual-gating mechanisms in Table 4. Removing stop-gradient slightly decreases the overall score from 63.35 to 63.12, while removing dual-gating slightly decreases it to 63.22. These mechanisms could be viewed as stabilization components rather than the sole source of performance gains: they make VCDist more conservative and robust without heavily changing the optimization objective.

Table 4. Ablation study of IC-VCO. We analyze the impact of different components and the design choices within the VCDist. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Overall Score</td><td colspan="3">HallusionBench</td><td colspan="3">AMBER</td><td colspan="2">CRPE</td><td colspan="2">R-Bench</td><td rowspan="2">BLINK Score</td></tr><tr><td>aAcc</td><td>fAcc</td><td>qAcc</td><td>Attr</td><td>Exist</td><td>Rel</td><td>Exist</td><td>Rel</td><td>Dis</td><td>Ref</td></tr><tr><td>IC-VCO</td><td>63.35</td><td>63.51</td><td>33.34</td><td>33.07</td><td>82.24</td><td>92.73</td><td>70.47</td><td>94.15</td><td>64.88</td><td>60.71</td><td>64.67</td><td>49.44</td></tr><tr><td>w/o single-image branch</td><td>62.69</td><td>61.72</td><td>31.79</td><td>31.21</td><td>82.42</td><td>92.71</td><td>70.19</td><td>94.25</td><td>64.10</td><td>60.20</td><td>64.17</td><td>48.76</td></tr><tr><td>w/o  $\mathcal{L}_{\text{VCDist}}$ </td><td>63.04</td><td>62.68</td><td>31.88</td><td>32.55</td><td>81.97</td><td>92.69</td><td>69.93</td><td>93.92</td><td>64.00</td><td>60.81</td><td>64.37</td><td>49.77</td></tr><tr><td>w/o token mask</td><td>63.10</td><td>61.72</td><td>31.02</td><td>31.53</td><td>82.43</td><td>92.32</td><td>72.58</td><td>94.00</td><td>64.36</td><td>60.30</td><td>64.27</td><td>50.18</td></tr><tr><td>w/o  $\mathcal{L}_{\text{Anc}}$ </td><td>61.15</td><td>63.09</td><td>32.08</td><td>33.85</td><td>77.81</td><td>90.90</td><td>73.56</td><td>89.83</td><td>62.04</td><td>57.58</td><td>61.34</td><td>46.61</td></tr><tr><td colspan="13">VCDist Ablation</td></tr><tr><td>w/o stop-gradient</td><td>63.12</td><td>62.67</td><td>33.05</td><td>32.19</td><td>82.30</td><td>92.99</td><td>70.29</td><td>93.97</td><td>64.41</td><td>59.90</td><td>64.67</td><td>49.65</td></tr><tr><td>w/o dual-gating</td><td>63.22</td><td>62.35</td><td>32.18</td><td>31.75</td><td>82.30</td><td>93.03</td><td>70.71</td><td>94.00</td><td>64.99</td><td>59.70</td><td>64.67</td><td>50.34</td></tr></table>

![](images/1ef1bde3bd2e2ec09676998f19c0075d983831f33a07b52f679a1af3b4530efa.jpg)

<details>
<summary>bar</summary>

| Method     | Synthetic | Edited |
| ---------- | --------- | ------ |
| ℒ_Single   | 0.85      | 0.55   |
| ℒ_Multi    | 0.88      | 0.73   |
</details>

(a)

![](images/7ae042668a29d2f3373679fbe96f9a569cceb3beff9368830040b34904a571f8.jpg)

<details>
<summary>line</summary>

| Training step | ℒ_Single | ℒ_Multi |
| ------------- | -------- | ------- |
| 0             | 0.5      | 0.5     |
| 50            | 0.6      | 0.7     |
| 100           | 0.65     | 0.72    |
| 150           | 0.68     | 0.73    |
| 200           | 0.69     | 0.74    |
| 250           | 0.7      | 0.75    |
| 300           | 0.7      | 0.75    |
</details>

(b)

![](images/ef8e4ae610e5357e7da6934b8822eeebf81dd874c4d96bba4a22b18166b9be63.jpg)

<details>
<summary>line</summary>

| Training step | Valid teacher ratio | Valid KL |
| ------------- | ------------------- | -------- |
| 0             | 0.6                 | 0.006    |
| 100           | 0.7                 | 0.008    |
| 200           | 0.73                | 0.010    |
| 300           | 0.74                | 0.011    |
</details>

(c)

Figure 4. Training diagnostics of IC-VCO on synthetic and edited preference data. (a) Edited preferences are harder to optimize than synthetic preferences under the same response-level IC-VCO setup, yielding lower reward accuracy for both the single-image and multi-image branches. (b) On edited data, the multi-image branch consistently achieves higher reward accuracy than the single-image branch, indicating that multi-image comparison provides a stronger preference signal. (c) The VCDist objective maintains a stable valid-teacher ratio and low valid KL throughout training, showing that its teacher signal is active and well-conditioned.   
![](images/97307e3c9240cd4fc8a6eca97e05fd541515d1ca86ef0e6942aea4e471da0bb2.jpg)  
Figure 5. Partition function bias analysis. We remove the singleimage branch of IC-VCO to form pure multi-image preference optimization. $\mathbf { I C - V C O _ { C r o s s A n c } }$ regroups the preference pairs by creating anchor prompt mismatch. The difference between DPO and VisDPO is shown in Eq. 4 and Eq. 5.

# 5.4. More Analysis

Sample Preference Difficulty. Figure 4 (a) shows that the preference pairs with edited samples are harder to optimize than synthetic preferences, with lower reward accuracy for both the single-image and multi-image branches. This supports our claim that edited samples act as fine-grained hard negatives rather than trivial contrastive examples.

![](images/7e2709ff00ba044019f2f1ded6e556879abf715154e1a8670d59914320ab45fd.jpg)

<details>
<summary>area</summary>

Dual-gating partition
| Training step | Correctness-blocked | Confidence-blocked | Active |
| :--- | :--- | :--- | :--- |
| 0 | 0.2 | 0.2 | 0.2 |
| 50 | 0.3 | 0.3 | 0.3 |
| 100 | 0.4 | 0.4 | 0.4 |
| 150 | 0.5 | 0.5 | 0.5 |
| 200 | 0.6 | 0.6 | 0.6 |
| 250 | 0.7 | 0.7 | 0.7 |
| 300 | 0.8 | 0.8 | 0.8 |
</details>

![](images/f4c17904d8e3a0922a4ca255be2904f32f13bc262cf566a453949d580d3e44a2.jpg)

<details>
<summary>line</summary>

| Training step | Fraction |
| ------------- | -------- |
| Last-100-step mean | 0.13% |
</details>

Figure 6. VCDist dual-gating dynamics. (a) As training progresses, the active fraction increases while the correctness-blocked fraction decreases. (b) The confidence gate provides an additional filtering step among teacher-correct samples by blocking cases where distillation is no longer needed.

Impact of Partition Function Mismatch. Directly estimating log $Z ( m , x ) / Z ( m ^ { \prime } , x )$ is difficult in DPO because the reward is implicit and the partition term is policydependent over an open-ended response space. We therefore quantify its practical effect with a controlled ablation in Figure 5. In the single-image setting, the mismatch-based visual DPO variant underperforms the corresponding nomismatch DPO baseline by 1.97 points on synthetic data and 0.56 points on edited data. In the multi-image setting, IC- $\mathbf { \partial } . \mathsf { V C O } _ { \mathbf { C r o s s A n c } }$ regroups preference pairs so that the same responses are compared across different anchor prompts, i.e., $r ( M , \hat { x } , y ) > r ( M , \hat { x } ^ { \prime } , y )$ and $r ( M , \hat { x } ^ { \prime } , y ^ { \prime } ) > r ( M , \hat { x } , y ^ { \prime } )$ , thereby reintroducing a context mismatch. This also hurts performance, but the degradation is smaller, possibly because the cross-anchor mismatch perturbs the conditioning much less than directly swapping images, and the multiimage visual contrast also provides stronger supervision, making the policy distribution more stable across anchors.

![](images/25fb026a57b06dd02aec23540a1b2679bb08e1106804d49feb9cd344be6c1900.jpg)

<details>
<summary>line</summary>

| Teacher threshold | Score |
| ----------------- | ----- |
| 0.5               | 63.35 |
| 0.6               | 63.12 |
| 0.7               | 63.25 |
</details>

(a)

![](images/95587f138fa1ee8872759afb8b758ae51bee320b5e5ec9fd65c8fd5128623b55.jpg)

<details>
<summary>heatmap</summary>

Metric deltas vs. threshold=0.5
| Metric | Value | Delta score |
| :--- | :--- | :--- |
| HB-aAcc | -0.63 | -0.63 |
| HB-fAcc | -0.58 | -0.58 |
| HB-qAcc | -0.22 | -0.44 |
| AMBER-Attr | +0.02 | +0.16 |
| AMBER-Exist | +0.24 | +0.38 |
| AMBER-Rel | -0.18 | -0.60 |
| CRPE-Exist | -0.20 | -0.13 |
| CRPE-Rel | -0.15 | -0.15 |
| RBench-Dis | -1.22 | -0.41 |
| RBench-Ref | +0.00 | +0.21 |
| BLINK | +0.06 | +0.27 |
</details>

(b)   
Figure 7. Sensitivity analysis of the VCDist teacher threshold. (a) Overall benchmark performance remains stable. (b) Per-metric differences show that the effect of threshold tuning does not lead to a consistent performance shift.

VCDist Gate Dynamics. Figure 6 analyzes how the VCDist gates behave during training. The dual-gating mechanism partitions training samples into active VCDist updates, confidence-blocked and correctness-blocked samples. The active fraction increases over training, while the correctness-blocked fraction decreases. This shows that the multi-image teacher becomes more reliable as training proceeds. The confidence gate is also empirically active: among teacher-correct samples, it filters 13.3% of cases on average in the last 100 steps. Overall, the dual-gating mechanism actively prevents undesired distillation while still allowing useful teacher signals to guide the single-image branch.

Sensitivity of VCDist Threshold. We test the sensitivity of the VCDist correctness threshold by changing it from 0.5 to 0.6 and 0.7. As shown in Figure 7, the overall score remains stable: the maximum change is only 0.23 points and per-metric differences do not show a consistent degradation pattern. This suggests that IC-VCO is not sensitive to the exact threshold choice within a reasonable range.

Position Bias in Multi-Image Context. Because IC-VCO uses positional anchor prompts, we further examine whether the multi-image teacher is affected by image order. We group training samples by whether the anchor-targeted image appears as the first or second image in the context. Figure 8 reports $\Delta = \mathrm { P o s } 2 - \mathrm { P o s } 1$ , where positive values indicate higher values when the anchor-targeted image is placed second. The teacher-side quantities show a detectable position-2 advantage: the last-100-step deltas are +2.0 points for teacher probability, +8.4 points for teacher accuracy, and +12.7 points for VCDist trigger rate. However, the student reward accuracy changes by only +0.6 points, indicating that teacher-side asymmetry does not translate into a substantial single-image policy bias. Since the image order is randomized during training, the position effect is averaged across the dataset, while the dual-gating mechanism further restricts distillation to reliable teacher cases.

![](images/311fdfd85aed8d61ef8b2887a760f4b42866c541e2733c595f4864e9b6b5fc65.jpg)

<details>
<summary>bar</summary>

Last-100-step position bias
| Metric | Value |
| :--- | :--- |
| Teacher prob. | +2.0 |
| Teacher acc. | +8.4 |
| Trigger rate | +12.7 |
| Student acc. | +0.6 |
</details>

![](images/fce115a9a2480135995034f2bfe02dc61293dab0e3c5cffa5354a36a288f5ab7.jpg)

<details>
<summary>line</summary>

| Training step | Trigger rates | Teacher ace | Student ace |
| ------------- | ------------ | ----------- | ----------- |
| 0             | 0.03         | 0.03        | 0.02        |
| 50            | 0.15         | 0.12        | 0.03        |
| 100           | 0.14         | 0.10        | 0.02        |
| 150           | 0.13         | 0.08        | 0.01        |
| 200           | 0.12         | 0.07        | 0.01        |
| 250           | 0.14         | 0.09        | 0.02        |
| 300           | 0.13         | 0.08        | 0.01        |
</details>

Figure 8. Position-effect diagnostics for VCDist. (a) The last-100-step mean deltas for the teacher probability, teacher accuracy, VCDist trigger rate, and student reward accuracy. (b) Representative deltas over training. The teacher-side statistics show a detectable positive position bias toward position 2, while the student single-image branch remains nearly position-symmetric.

# 6. Related Work

To mitigate multimodal hallucinations, recent works (Wang et al., 2024a; Yang et al., 2025; Wu et al., 2025b) incorporate visual constraints into DPO by contrasting positive and negative image pairs. While pioneering, these approaches face two critical limitations. Theoretically, conditioning on distinct visual inputs prevents the cancellation of partition functions, violating the rigorous DPO derivation and introducing intractable bias. Practically, relying on retrieved or synthesized negatives often introduces coarse stylistic discrepancies (Liu et al., 2025b), creating trivial negatives that enable shortcut learning rather than enforcing fine-grained visual grounding. See Appendix A for detailed related works.

# 7. Conclusion

In this work, we present In-Context Visual Contrastive Optimization (IC-VCO), a framework to address the theoretical inconsistencies in multimodal preference optimization for hallucination mitigation. By unifying contrastive images within a shared context, IC-VCO eliminates the intractable partition function bias, establishing a rigorous mathematical foundation for visual preference alignment. We also introduce Visual Contrast Distillation (VCDist), a gated distillation regularizer to calibrate the single-image policy with the visual contrastive multi-image distribution. Furthermore, we propose a Contrastive Sample Editing pipeline which generates high-quality hard negatives to prevent shortcut learning on global distribution shift and enforce fine-grained visual grounding. Empirical results across diverse benchmarks validate the best overall performance of IC-VCO and the benefits from contrastive sample editing.

# Impact Statement

This work introduces a visual preference optimization framework for improving the reliability of Vision-Language Models. By encouraging models to distinguish fine-grained visual evidence through contrastive preference supervision, the method may help reduce visually inconsistent responses in applications such as visual assistants, educational tools, and content understanding systems.

However, our work should not be interpreted as a complete solution to multimodal hallucination or as a replacement for broader safety mechanisms. The method targets a specific post-training setting, namely DPO-style multimodal preference alignment, and is complementary to decoding-time, representation-editing, and human-feedback-based hallucination mitigation approaches. Models trained with our method may still produce incorrect or misleading outputs, especially in open-world or high-stakes scenarios such as medical diagnosis, legal evidence analysis, autonomous driving, or security surveillance, where additional validation and human oversight are necessary.

The proposed contrastive sample editing pipeline also relies on external expert VLMs and image editing models, which may introduce biases, verification errors, or editing artifacts. Moreover, edited counterfactual images could be misused if detached from their intended research context. We therefore recommend using the pipeline and samples only for controlled model training and evaluation. Finally, our approach introduces additional computational cost through multi-image training and model-assisted data construction. We release the materials to help reduce redundant generation costs for future research.

# References

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report, 2025a. URL https://arxiv.org/abs/2511.21631.   
Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2.5- vl technical report. arXiv preprint arXiv:2502.13923, 2025b.   
Bai, Z., Wang, P., Xiao, T., He, T., Han, Z., Zhang, Z., and

Shou, M. Z. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024.

Black Forest Labs, Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., Kulal, S., Lacey, K., Levi, Y., Li, C., Lorenz, D., Muller, J., Podell, D., Rombach, R., Saini, H., Sauer, ¨ A., and Smith, L. Flux.1 kontext: Flow matching for incontext image generation and editing in latent space, 2025. URL https://arxiv.org/abs/2506.15742.

D’Amour, A., Heller, K., Moldovan, D., Adlam, B., Alipanahi, B., Beutel, A., Chen, C., Deaton, J., Eisenstein, J., Hoffman, M. D., et al. Underspecification presents challenges for credibility in modern machine learning. Journal of Machine Learning Research, 23(226):1–61, 2022.

Deletang, G., Ruoss, A., Duquenne, P.-A., Catt, E., Genewein, T., Mattern, C., Grau-Moya, J., Wenliang, L. K., Aitchison, M., Orseau, L., Hutter, M., and Veness, J. Language modeling is compression. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=jznbgiynus.

Deng, H., Wang, C., Xin, L., Yuan, D., Zhan, J., Zhou, T., Ma, J., Gao, J., and Xu, R. Webcites: Attributed queryfocused summarization on chinese web search results with citations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15095–15114, 2024.

Duan, H., Yang, J., Qiao, Y., Fang, X., Chen, L., Liu, Y., Dong, X., Zang, Y., Zhang, P., Wang, J., et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 11198– 11201, 2024.

Fu, J., Huangfu, S., Fei, H., Shen, X., Hooi, B., Qiu, X., and Ng, S.-K. Chip: Cross-modal hierarchical direct preference optimization for multimodal llms. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=7lpDn2MhM2.

Fu, X., Hu, Y., Li, B., Feng, Y., Wang, H., Lin, X., Roth, D., Smith, N. A., Ma, W.-C., and Krishna, R. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pp. 148–166. Springer, 2024.

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., and Wichmann, F. A. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6904–6913, 2017.   
Guan, T., Liu, F., Wu, X., Xian, R., Li, Z., Liu, X., Wang, X., Chen, L., Huang, F., Yacoob, Y., et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14375– 14385, 2024.   
Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X., et al. Deepseekr1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.   
He, L., Chen, Z., Shi, Z., Yu, T., Shao, J., and Sheng, L. A topic-level self-correctional approach to mitigate hallucinations in mllms. arXiv preprint arXiv:2411.17265, 2024.   
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https:// openreview.net/forum?id=nZeVKeeFYf9.   
Jiang, C., Xu, H., Dong, M., Chen, J., Ye, W., Yan, M., Ye, Q., Zhang, J., Huang, F., and Zhang, S. Hallucination augmented contrastive learning for multimodal large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 27036–27046, 2024.   
Jiang, N., Kachinthaya, A., Petryk, S., and Gandelsman, Y. Interpreting and editing vision-language representations to mitigate hallucinations. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=94kQgWXojH.   
Jin, Z., Song, X., Wang, N., Liu, Y., Li, C., Li, X., Wang, R., Li, Z., Qi, Q., Cheng, L., et al. Andesvl technical report: An efficient mobile-side multimodal large language model. arXiv preprint arXiv:2510.11496, 2025.   
Leng, S., Zhang, H., Chen, G., Li, X., Lu, S., Miao, C., and Bing, L. Mitigating object hallucinations in large visionlanguage models through visual contrastive decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13872–13882, 2024.   
Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., and Li, C.

LLaVA-onevision: Easy visual task transfer. Transactions on Machine Learning Research, 2025a. ISSN 2835- 8856. URL https://openreview.net/forum? id=zKv8qULV6n.   
Li, C., Zhang, J., Zhang, Z., Wu, H., Tian, Y., Sun, W., Lu, G., Min, X., Liu, X., Lin, W., et al. R-bench: Are your large multimodal model robust to real-world corruptions? IEEE Journal of Selected Topics in Signal Processing, 2025b.   
Li, F., Zhang, R., Zhang, H., Zhang, Y., Li, B., Li, W., Ma, Z., and Li, C. Llava-next-interleave: Tackling multiimage, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024.   
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L. Microsoft coco: ´ Common objects in context. In European conference on computer vision, pp. 740–755. Springer, 2014.   
Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.   
Liu, S., Ye, H., and Zou, J. Reducing hallucinations in vision-language models via latent space steering. arXiv preprint arXiv:2410.15778, 2024.   
Liu, S., Wang, S., Li, Z., Wang, J., Zeng, C., and Wei, Z. Ovip: Online vision-language preference learning. arXiv preprint arXiv:2505.15963, 2025a.   
Liu, W., Song, X., Li, J., Wei, Y., Zheng, N., Yin, J., and Nie, L. Mitigating hallucination through theoryconsistent symmetric multimodal preference optimization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025b. URL https: //openreview.net/forum?id=tIW29IpCwG.   
Liu, Z., Zang, Y., Dong, X., Zhang, P., Cao, Y., Duan, H., He, C., Xiong, Y., Lin, D., and Wang, J. MIA-DPO: Multi-image augmented direct preference optimization for large vision-language models. In The Thirteenth International Conference on Learning Representations, 2025c. URL https://openreview.net/forum? id=f7WBRSuf9l.   
Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.   
Luo, T., Cao, A., Lee, G., Johnson, J., and Lee, H. Probing visual language priors in VLMs. In Fortysecond International Conference on Machine Learning, 2025. URL https://openreview.net/forum? id=bhTBirS0qi.

Manevich, A. and Tsarfaty, R. Mitigating hallucinations in large vision-language models (lvlms) via languagecontrastive decoding (lcd). In Findings of the Association for Computational Linguistics ACL 2024, pp. 6008–6022, 2024.   
Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W.-t., Koh, P., Iyyer, M., Zettlemoyer, L., and Hajishirzi, H. Factscore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 12076–12100, 2023.   
Pearl, J. Causality. Cambridge university press, 2009.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.   
Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36: 53728–53741, 2023.   
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.   
Sarkar, P., Ebrahimi, S., Etemad, A., Beirami, A., Arik, S. O., and Pfister, T. Mitigating object hallucination in MLLMs via data-augmented phrase-level alignment. In The Thirteenth International Conference on Learning Representations, 2025a. URL https:// openreview.net/forum?id=yG1fW8igzP.   
Sarkar, S., Che, Y., Gavin, A., Beerel, P. A., and Kundu, S. Mitigating hallucinations in vision-language models through image-guided head suppression. In Christodoulopoulos, C., Chakraborty, T., Rose, C., and Peng, V. (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 12481–12500, Suzhou, China, November 2025b. Association for Computational Linguistics. ISBN 979- 8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main. 631. URL https://aclanthology.org/2025. emnlp-main.631/.   
Scholkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalch-¨ brenner, N., Goyal, A., and Bengio, Y. Toward causal representation learning. Proceedings of the IEEE, 109(5): 612–634, 2021.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/ 1707.06347.   
Shah, H., Tamuly, K., Raghunathan, A., Jain, P., and Netrapalli, P. The pitfalls of simplicity bias in neural networks. Advances in Neural Information Processing Systems, 33: 9573–9585, 2020.   
Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8317–8326, 2019.   
Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008–3021, 2020.   
Wang, F., Zhou, W., Huang, J. Y., Xu, N., Zhang, S., Poon, H., and Chen, M. mDPO: Conditional preference optimization for multimodal large language models. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 8078–8088, Miami, Florida, USA, November 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main. 460. URL https://aclanthology.org/2024. emnlp-main.460/.   
Wang, J., Wang, Y., Xu, G., Zhang, J., Gu, Y., Jia, H., Yan, M., Zhang, J., and Sang, J. An llm-free multi-dimensional benchmark for mllms hallucination evaluation. arXiv preprint arXiv:2311.07397, 2023.   
Wang, J., Gao, Y., and Sang, J. Valid: Mitigating the hallucination of large vision language models by visual layer fusion contrastive decoding, 2024b. URL https://arxiv.org/abs/2411.15839.   
Wang, W., Ren, Y., Luo, H., Li, T., Yan, C., Chen, Z., Wang, W., Li, Q., Lu, L., Zhu, X., et al. The all-seeing project v2: Towards general relation comprehension of the open world. arXiv preprint arXiv:2402.19474, 2024c.   
Wang, X., Pan, J., Ding, L., and Biemann, C. Mitigating hallucinations in large vision-language models with instruction contrastive decoding. In Findings of the Association for Computational Linguistics ACL 2024, pp. 15840–15853, 2024d.   
Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.-m., Bai, S., Xu, X., Chen, Y., et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Wu, S., Sun, F.-Y., Wen, K., and Haber, N. Symmetrical visual contrastive optimization: Aligning visionlanguage models with minimal contrastive images. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 30284–30297, Vienna, Austria, July 2025b. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long. 1462. URL https://aclanthology.org/2025. acl-long.1462/.   
Xie, Y., Li, G., Xu, X., and Kan, M.-Y. V-dpo: Mitigating hallucination in large vision language models via visionguided direct preference optimization. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 13258–13273, 2024.   
Yang, Z., Luo, X., Han, D., Xu, Y., and Li, D. Mitigating hallucinations in large vision-language models via dpo: On-policy data hold the key. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 10610–10620, 2025.   
Ye, Z., Li, Q., Feng, X., Qin, L., Huang, Y., Li, B., Jiang, K., Xiang, Y., Zhang, Z., Lu, Y., Tang, D., Tu, D., and Qin, B. CLAIM: Mitigating multilingual object hallucination in large vision-language models with crosslingual attention intervention. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13080–13094, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251- 0. doi: 10.18653/v1/2025.acl-long.640. URL https: //aclanthology.org/2025.acl-long.640/.   
Yu, T., Yao, Y., Zhang, H., He, T., Han, Y., Cui, G., Hu, J., Liu, Z., Zheng, H.-T., Sun, M., et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13807–13816, 2024a.   
Yu, T., Zhang, H., Li, Q., Xu, Q., Yao, Y., Chen, D., Lu, X., Cui, G., Dang, Y., He, T., Feng, X., Song, J., Zheng, B., Liu, Z., Chua, T.-S., and Sun, M. Rlaif-v: Open-source ai feedback leads to super gpt-4v trustworthiness. arXiv preprint arXiv:2405.17220, 2024b.   
Zeng, Y., Liu, G., Ma, W., Yang, N., Zhang, H., and Wang, J. Token-level direct preference optimization. In International Conference on Machine Learning, pp. 58348– 58365. PMLR, 2024.   
Zhang, J., Cai, M., Xie, T., and Lee, Y. J. Countercurate: Enhancing physical and semantic visio-linguistic

compositional reasoning via counterfactual examples. In Findings of the Association for Computational Linguistics ACL 2024, pp. 15481–15495, 2024.

Zou, X., Wang, Y., Yan, Y., Lyu, Y., Zheng, K., Huang, S., Chen, J., Jiang, P., Liu, J., Tang, C., et al. Look twice before you answer: Memory-space visual retracing for hallucination mitigation in multimodal large language models. In International Conference on Machine Learning, pp. 80873–80899. PMLR, 2025.

# A. Extended Related Work

# A.1. Multimodal Hallucination and Mitigation

Despite the remarkable capabilities of Large Vision-Language Models (LVLMs), they frequently exhibit multimodal hallucinations, where generated responses contradict visual content or fabricate non-existent objects (Bai et al., 2024; Guan et al., 2024; Luo et al., 2025). Recent analyses attribute this phenomenon to the model’s over-reliance on language priors (Deletang et al., 2024; Luo et al., 2025) and insufficient visual grounding during the generation process (Lu et al., 2022; Xie et al., 2024; Wu et al., 2025b). Mitigation strategies can be broadly categorized into training-free and training-based approaches.

Training-Free Approaches. These methods intervene during inference or manipulate internal representations without updating model parameters. Decoding-based strategies, such as Visual Contrastive Decoding (VCD) (Leng et al., 2024), Instruction Contrastive Decoding (ICD) (Wang et al., 2024d), and VaLiD (Wang et al., 2024b), construct contrastive or modified decoding distributions to reduce hallucinations from different sources, including language priors and visual encoding distortions. Language-Contrastive Decoding (LCD) (Manevich & Tsarfaty, 2024) further refines this by directly contrasting VLM logits with those of a uni-modal LLM. MemVR (Zou et al., 2025) re-injects visual tokens before answer decoding to enhance visual grounding. Beyond decoding, representation editing methods (Liu et al., 2024; Jiang et al., 2025) manipulate activation patterns or suppress specific attention heads (Sarkar et al., 2025b; Ye et al., 2025) to enhance visual signal reliance. While effective, these methods often incur additional inference latency or require meticulous hyperparameter tuning for different architectures.

Training-Based Approaches. These methods align models via auxiliary visual supervision (Jiang et al., 2024; Sarkar et al., 2025a) or Reinforcement Learning (Stiennon et al., 2020; Yu et al., 2024a;b). Among them, Direct Preference Optimization (DPO) (Rafailov et al., 2023) based approaches have become a dominant paradigm due to its training stability and efficiency. Early multimodal adaptations, such as mDPO (Wang et al., 2024a) and V-DPO (Xie et al., 2024), extended textual DPO by optimizing preference between hallucinated and faithful captions while treating the image as a static condition. To explicitly enforce visual grounding, subsequent works like S-VCO (Wu et al., 2025b) and SymMPO (Liu et al., 2025b) introduced visual contrastive optimization, where the model learns to distinguish between matching and mismatched image-text pairs.

To explicitly enforce visual grounding, mDPO (Wang et al., 2024a) pioneers a visual preference optimization objective, constructing rejected samples by manipulating visual contexts to penalize image-text misalignment. To further mitigate the model’s over-reliance on language priors, Xie et al. (2024) propose V-DPO, which incorporates a Vision-Guided Classifier-Free Guidance (CFG) framework also utilizing visual contrastive samples for optimization. Subsequently, Wu et al. (2025b) argue that utilizing preference pairs symmetrically—by assigning each response with a contradictory image—improves data efficiency. Their proposed S-VCO objective leverages contrastive image samples alongside text-only samples for visual preference modeling. Most recently, Liu et al. (2025b) identified that these prior visual DPO formulations are theoretically non-rigorous due to the partition function mismatch issue. As a theory-consistent alternative, they propose SymMPO, which restores DPO consistency by performing symmetric response-pair optimization under two separate single-image conditions (m, x) and $( m ^ { \prime } , x )$ , coupled by a preference-margin regularizer. In contrast, IC-VCO reformulates the task so that both contrastive images are placed in one shared context $M = [ m , m ^ { \prime } ]$ and the preference comparison is performed under the same full condition (M, x). This allows IC-VCO to retain explicit in-context visual comparison while remaining theory-consistent.

Concurrent works have also explored fine-grained improvements to the DPO objective itself. For instance, TDPO (Zeng et al., 2024) proposes a token-level DPO formulation with forward KL divergence constraints; CHiP (Fu et al., 2025) designs a cross-model hierarchical DPO framework which combines textual preference at response-level, segment-level, and token-level. We note that these fine-grained preference optimizations are orthogonal to our research direction. In this work, we apply single-image token-level DPO using edited samples to all evaluated methods to align the granularity dimension. Furthermore, OPA-DPO (Yang et al., 2025) highlights the importance of on-policy data and aligns the preference data with the initial policy via finetuning. In our experiments, we apply this setting to all evaluated methods. MIA-DPO (Liu et al., 2025c) is similar to our IC-VCO as both works model multi-image DPO. The key difference lies in the supervision geometry. In MIA-DPO, the additional images are inserted as noisy distractors to induce multi-image hallucinations, the extra image mainly acts as a nuisance variable which is expected to be ignored by the model. In contrast, IC-VCO constructs semantically paired images which provide the supervision signal for explicit visual comparison while also supporting symmetrical optimization. MIA-DPO’s approach could not support this, as the additional image is a distractor irrelevant to the question, which cannot ensure $r ( M , \hat { x } , y ^ { \prime } ) > r ( M , \hat { x } , y )$ . To further validate this, we sample 100 samples from the

MIA-DPO dataset2 and use the expert VLM to evaluate the symmetry of preference pairs, and found only 21/100 pairs are symmetric. Additionally, MIA-DPO uses policy-coupled self-generated preference pairs, whereas IC-VCO operates on a fixed offline contrastive dataset shared across methods in our comparisons. IC-VCO also shows that jointly optimizing multi-image and single-image policies brings benefits, and proposes VCDist to bridge them.

# A.2. Visual Contrastive Data Construction

A pivotal challenge in multimodal preference optimization lies in constructing high-quality negative samples—specifically, contrastive images $m ^ { \prime } -$ —that compel the model to attend to fine-grained visual details. Pioneering efforts (Wang et al., 2024a; Yang et al., 2025) typically derived $m ^ { \prime }$ via heuristic augmentations, such as random cropping or noise injection. However, empirical studies (Wu et al., 2025b; Liu et al., 2025b) indicate that these corruption-based techniques yield suboptimal gains. Such drastic structural perturbations often destroy essential semantic information, rendering the negatives too easily distinguishable and prone to shortcut learning. Furthermore, these low-fidelity images prove ineffective under symmetric optimization frameworks, which require high-quality distributional matching.

To address this, subsequent approaches (Xie et al., 2024; Wu et al., 2025b; Liu et al., 2025a) have shifted towards leveraging text-to-image diffusion models (Rombach et al., 2022; Black Forest Labs et al., 2025) to synthesize distinct negatives conditioned on hallucinated captions, or utilizing image retrieval (Liu et al., 2025b) to source hard negatives. While domain-specific heuristics like horizontal flipping (Wu et al., 2025b) have also been explored for spatial relations, they lack generalizability. Crucially, as discussed in § 4, both synthesis-based and retrieval-based paradigms suffer from the under-specification problem (D’Amour et al., 2022), inevitably introducing global semantic drift. These deviations result in coarse-grained negatives, allowing models to bypass visual reasoning. In contrast, our Contrastive Sample Editing framework employs a localized sample editing strategy, implementing precise, surgical manipulations to generate fine-grained negatives that strictly preserve the surrounding context.

# B. Detailed Contrastive Sample Editing Pipeline

This section provides the detailed procedure used to construct the edited contrastive samples in Section 4. Given a seed preference tuple $( m , x , y , y ^ { \prime } )$ , our goal is to produce an edited image $m ^ { \prime }$ and a rewritten contrastive response $y _ { \mathrm { n e w } } ^ { \prime }$ such that $( m , x , y )$ and $( m ^ { \prime } , x , y _ { \mathrm { n e w } } ^ { \prime } )$ form a symmetrical fine-grained contrastive pair, while $( m ^ { \prime } , x , y )$ becomes a hard negative whose error is localized to the edited visual evidence.

Edit Strategy Formulation. The first stage derives an executable image editing instruction T that introduces a precise semantic conflict between the edited image $m ^ { \prime }$ and the original faithful response y. We use QwenVL-Plus (Bai et al., 2025a) as an expert VLM to analyze the seed tuple $( m , x , y , y ^ { \prime } )$ and formulate $\tau$ under two scenarios.

• Scenario A: Hallucination Realization. If the rejected response $y ^ { \prime }$ contains an explicit hallucinated detail that contradicts the original image m and the faithful response $y ,$ the edit instruction T modifies m to make one targeted hallucinated detail in $y ^ { \prime }$ factually true in the edited image $m ^ { \prime }$ . After this intervention, the original chosen response y naturally becomes a negative description for $m ^ { \prime }$ .   
• Scenario B: Hallucination Injection. If $y ^ { \prime }$ does not contain a localized hallucinated detail suitable for editing, we instead target the chosen response $y .$ The edit instruction $\tau$ minimally modifies m to contradict a distinct visual detail described in y. In this case, the visual evidence supporting y is surgically removed or altered, making y a negative description for the edited image $m ^ { \prime }$ .

For both scenarios, the expert VLM classifies the target edit into one of three hallucination types: existence, attribute, and relation. We additionally enforce strict constraints on the editing instruction to avoid structural changes, global style shifts, or background modifications. This ensures that $m ^ { \prime }$ remains visually congruent with m except for the targeted semantic concept.

Contrastive Response Rewriting. To construct a valid symmetrical preference pair, we require a contrastive response $y _ { \mathrm { n e w } } ^ { \prime }$ that faithfully describes the edited image $m ^ { \prime }$ . The original rejected response $y ^ { \prime }$ is not directly used as the positive response for m′ because it may contain multiple hallucinations in Scenario A or may be unrelated to the selected edit target in Scenario B. Therefore, we adopt a minimal intervention rewriting strategy. Specifically, the expert VLM rewrites the original chosen response y by changing only the keywords or short phrases necessary to align with the edit instruction T , while preserving the sentence structure, reasoning pattern, and linguistic style of y.

Table 5. Computation overhead of visual editing: Qwen-Image-Edit-2511 with a fused Lightning LoRA in bf16, 8 inference steps per image, and the reversible padding/unpadding pipeline. Statistics are reported over 100 image-editing runs on a single NVIDIA H20 GPU. 

<table><tr><td>Inference steps per image</td><td>Measured images</td><td>Mean latency</td><td>Median latency</td><td>P90 latency</td><td>Peak GPU memory</td><td>Total time for 100 images</td></tr><tr><td>8</td><td>100</td><td>14.02 s/image</td><td>14.00 s/image</td><td>14.08 s/image</td><td>58.79 GiB</td><td>1402.05 s (23.37 min)</td></tr></table>

This rewriting strategy has three benefits. First, it inherits the coherence and detailed reasoning of the original chosen response, avoiding the noise often present in rejected responses. Second, the resulting pair $( m ^ { \prime } , y _ { \mathrm { n e w } } ^ { \prime } )$ versus $( m ^ { \prime } , y )$ differs only in the atomic concept targeted by the edit, which prevents the model from exploiting textual style, length, or reasoning-format shortcuts. Third, because $y _ { \mathrm { n e w } } ^ { \prime }$ is produced by minimally modifying y, we can directly compute the token-level differences between the two responses and extract fine-grained masks for token-level preference scoring.

Fine-Grained Visual Editing. We use Qwen-Image-Edit (Wu et al., 2025a) to apply the editing instruction $\tau$ to the original image m. In our implementation, the editor is instantiated with the Qwen-Image-Edit-25113 base checkpoint and a fused Lightning LoRA checkpoint, Qwen-Image-Edit-2511-Lightning4. We fuse the LoRA with scale 1.0 in bf16, and use 8 denoising steps with true cfg scale=1. The editor input resolution is fixed to $1 0 2 4 \times 1 0 2 4$ , the random seed is fixed to $4 2$ , and we use an empty negative prompt.

A practical challenge is that image editing models typically operate at a fixed square resolution. Naively resizing the original image to this resolution can distort the aspect ratio and break the spatial alignment between m and $m ^ { \prime } .$ . To avoid such artifacts, we adopt a reversible padding pipeline. We first resize m to fit the 1024 × 1024 canvas while preserving its aspect ratio using Lanczos resampling, then pad the remaining margins with a white background to form a square input. After editing, we crop out the padded margins according to the recorded padding metadata and resize the edited content back to the original resolution. This geometry-aware procedure keeps the original and edited images pixel-aligned in non-edited regions. As a result, the contrastive supervision is concentrated on the localized semantic change specified by T , rather than on resolution artifacts, aspect-ratio distortion, or unintended background shifts.

Validity Verification. Because generative image editing can fail or introduce unintended artifacts, we apply a post-hoc verification loop using QwenVL-Plus. The expert VLM re-evaluates the generated triplet $( m ^ { \prime } , T , y _ { \mathrm { n e w } } ^ { \prime } )$ under two criteria.

First, it performs a visual consistency check to determine whether the edited image m′ faithfully implements the semantic change specified by T . Samples with failed edits, excessive background changes, structural distortion, or unintended modifications to non-target objects are discarded. Second, it performs response rectification by checking whether $y _ { \mathrm { n e w } } ^ { \prime }$ accurately describes the edited image $m ^ { \prime }$ . Minor textual mismatches or residual hallucinations are corrected through minimal editing, while samples requiring substantial rewriting are removed.

The entire pipeline for processing each sample involves a single call for strategy formulation, response rewriting, visual editing, and verification respectively. Table 5 presents the computational overhead for deploying the visual editor on local NVIDIA H20 GPU. For other stages, we relied on API calls via DashSope.5

# C. Implementation Details

All methods are trained on fixed base models with the same training set. Token masks in edited samples are applied to all methods. For methods that do not model symmetrical relations (i.e. DPO, mDPO, V-DPO), we simply split the samples $( m , m ^ { \prime } , x , y , y ^ { \prime } )$ into separate image-text pairs $( m , x , y )$ and $( m ^ { \prime } , x , y ^ { \prime } )$ . The objective of SymMPO (Liu et al., 2025b) includes a term $\mathcal { L } _ { \mathrm { D P O } _ { m } }$ that requires extra annotation of rejected responses for both the original and contrastive images. To ensure fair comparison, we exclude this term which in practice could be integrated into all methods given extra annotation.

We follow the paradigm of OPA-DPO (Yang et al., 2025) to first finetune the base models with LoRA (Hu et al., 2022), using the chosen samples from the training data for on-policy alignment. In practice, we merge synthetic and edited chosen samples to form a unified SFT dataset. For IC-VCO, we add multi-image samples to align the policy with our anchor prompt instruction. We set the LoRA rank to 128 and alpha to 256, finetune for one epoch with a learning rate of 2e-5 and a batch size of 128. Then, we use the finetuned policy as the reference policy for preference optimization, using another initialized LoRA adapter with the same configuration for training. The preference optimization uses a learning rate of 5e-6 and a global batch size of 64, and runs for one epoch. All runs are conducted on 8 NVIDIA H20 GPUs.

We follow prior works (Wang et al., 2024a; Xie et al., 2024; Liu et al., 2025b) for hyper-parameters initialization. For all methods, we set $\beta = 0 . 1$ , and set the anchor loss weight η = 1. For IC-VCO, we set $\lambda _ { 1 } = 0 . 7 5 , \lambda _ { 2 } = 1 . 7 5 , \eta _ { 1 } = \eta _ { 2 } = 1$ , and $\gamma = 0 . 3$ . For all other hyper-parameters in baselines, we use their reported values.

For evaluation, we use greedy decoding for deterministic prediction, with max new tokens=128. For HallusionBench and AMBER, we use Qwen-Flash API via DashScope to map model predictions to yes-or-no labels. For multiple-choice benchmarks including CRPE, R-Bench, and BLINK, we use exact matching for prediction scoring.