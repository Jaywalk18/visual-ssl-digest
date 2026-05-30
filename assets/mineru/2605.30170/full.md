# Unveiling the Visual Counting Bottleneck in Vision-Language Models

Xingzhou Pang \* 1 Yifan Hou \* 1 Junling Wang 1 Mrinmaya Sachan 1

{ xingzhou.pang, yifan.hou, junling.wang, mrinmaya.sachan }@inf.ethz.ch

ETHzurich

# Abstract

While Large Vision-Language Models (VLMs) excel at interpolation, they suffer catastrophic failures in systematic generalization, most notably in visual counting. In this work, we investigate this extrapolation bottleneck by deconstructing visual counting into three cognitive stages: visual individuation, magnitude awareness, and symbolic mapping. Using synthetic Go boards and linear probes, we demonstrate that visual backbones maintain robust, linearly separable representations of quantity well into the extrapolation regime, ruling out perceptual failure. Furthermore, models retain latent magnitude awareness, successfully performing comparative reasoning on quantities they fail to enumerate. We pinpoint the collapse to the symbolic mapping stage, where the model fails to project valid visual magnitudes onto symbolic tokens. Our findings support a fractured magnitude hypothesis: VLMs fail to acquire a universal number space, instead learning disjoint, modality-specific statistical manifolds that prevent cross-modal grounding for unseen quantities. Validated on the state-of-the-art foundation model, our results suggest that bridging this gap requires inductive priors enforcing unified representations, as data scaling alone is insufficient.1

# 1. Introduction

Systematic generalization, the ability to learn a rule from finite examples and apply it to inputs outside the training distribution, remains the central chasm between biological

\*Equal contribution 1Department of Computer Science, ETH Zurich. Correspondence to: Yifan Hou ¨ <yifan.hou@inf.ethz.ch>, Mrinmaya Sachan <mrinmaya.sachan@inf.ethz.ch>.

Proceedings of the 43 rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

1Our code is publicly available here: https://github.c om/Russellpang/semproj.

![](images/0eb190d49696878477b65c229950fc4842c8837a077c236b00c785a8a491f020.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Go Board"] --> B["Vision Module"]
    B --> C["Language Module"]
    C --> D["Answer"]
    E["Question"] --> B
    F["Prediction Number"] --> D
    G["Visual Individuation"] --> H["Magnitude Awareness"]
    H --> I["Symbolic Mapping"]
```
</details>

Figure 1. Deconstructing visual counting. We analyze the counting bottleneck in VLMs via 3 cognitive stages: (1) visual individuation, (2) magnitude awareness, and (3) symbolic mapping.

and artificial intelligence (Fodor & Pylyshyn, 1988; Marcus, 2003; Lake & Baroni, 2018; Hupkes et al., 2020). On the one hand, while Large Vision-Language Models (VLMs) have demonstrated impressive proficiency in describing visual scenes and solving visual reasoning (OpenAI, 2023; Team, 2023; 2025; Bai et al., 2025), others claim they are mere statistical interpolators, often excelling within the support of their training distribution but are error-prone when required to extrapolate (Thrush et al., 2022a; Yuksekg ¨ on¨ ul¨ et al., 2023b; Paiss et al., 2023). This limitation is most clearly evident in the task of visual counting.

Counting serves as a canonical testbed for reasoning as it isolates the problem of extrapolation in its purest form. Grounded in a simple recursive algorithm (n → n + 1), counting allows humans to enumerate arbitrary quantities zero-shot once the principles of cardinality are acquired (Dedekind, 1965; Carey, 2000; Dehaene, 2011; Piantadosi et al., 2012). In contrast, neural models treat counting as a pattern-matching problem, degrading catastrophically when object quantities exceed those observed during training (Wallace et al., 2019; Bender & Koller, 2020; Press et al., 2022; Anil et al., 2022). This failure raises a fundamental diagnostic question: Does the failure to count stem from an inability to perceive distinct objects, an inability to comprehend quantity, or an inability to map that quantity to a label?

We investigate the bottleneck of visual counting in VLMs by deconstructing it into three cognitive stages (Fig. 1): visual individuation (recognition), magnitude awareness (numerosity), and symbolic mapping (articulation). To rigorously isolate architectural biases from the noise inherent in real-world datasets, we construct a synthetic laboratory using Go game boards (§ 2.1). Crucially, we employ a decoupled training curriculum similar to VLM training: the model is trained to visually count only up to N = 49, while its language decoder is pretrained to count textually up to N = 99. This creates a specific visual extrapolation regime (50 − 99), where the model possesses the linguistic labels but with a lack of visual-textual pairings, and a full extrapolation scenario (100 − 120) where both the visual densities and the textual quantities are unseen. We further validate our findings on a state-of-the-art VLM (Qwen3-VL) to confirm that the observed failure mechanisms persist in real-world architectures regardless of pretraining scale (§ 2.2).

To measure the model’s latent counting capability, we operationalize the hidden number via the linear separability of its internal representations. We posit that if feature vectors for distinct quantities remain linearly distinguishable, the model inherently possesses the capacity for visual individuation. Our investigation reveals a counter-intuitive phenomenon: the bottleneck lies neither in visual recognition nor in abstract numerosity. Visual backbones maintain robust, linearly separable representations of quantity well beyond the training distribution, confirming that the model effectively “sees” the objects (§ 3.2). Furthermore, models retain latent magnitude awareness, successfully performing comparative reasoning on quantities far outside the training set (e.g., accurately determining if a visual set of 110 items corresponds to a textual list of 110 characters) even when explicit enumeration fails (§ 3.3). Instead, the generalization collapse is pinpointed strictly to the final symbolic mapping stage (§ 3.4). Validation on foundation models confirms this mechanism: while real-world architectures exhibit minor noise in perception and magnitude processing, the catastrophic failure is consistently driven by the structural disconnect between the preserved latent magnitude and the generated symbolic token (§ 4).

We interpret these findings through the lens of a fractured magnitude hypothesis. We posit that current VLM architectures fail to acquire a universal number space that bridges modalities; instead, they learn disjoint, modality-specific statistical manifolds for visual and textual magnitudes. Consequently, while the language module may possess a robust representation of a number (e.g., 75) and the vision module possesses a precise representation of the corresponding objects, the model fails to map the latter to the former if that specific cross-modal pairing is not observed during training. These findings suggest that solving the extrapolation gap in VLMs is not merely a function of scaling data, but requires architectural or training priors that enforce unified magnitude representations to enable true cross-modal grounding.

# 2. Experimental Framework

To rigorously investigate the visual counting bottleneck, we adopt a two-fold experimental design. First, we construct a synthetic laboratory using a custom VLM trained from scratch, allowing us to strictly control the training distribution and decouple visual exposure from textual priors (§ 2.1). Second, we perform a real-world validation using a state-of-the-art pretrained VLM to confirm that our findings hold in foundation models trained at scale (§ 2.2).

# 2.1. Study 1: The Synthetic Laboratory

Architecture. For initial analysis, we train a lightweight “Toy VLM” composed of standard architectural primitives: a vision Transformer encoder (ViT-Base configuration but only with 2 layers) for visual perception (Dosovitskiy et al., 2021), connected to a causal Transformer decoder (GPT-2 style, 2 layers) for text generation (Radford et al., 2019). This mirrors the architectural bias of modern VLMs (e.g., Qwen-VL (Bai et al., 2025), Gemma (Team, 2025)) while remaining computationally tractable for full retraining.

The Go Board Dataset. We utilize standard 19 × 19 Go game boards as a controlled environment for testing and evaluating our model. Unlike natural images confounded by occlusion or semantic ambiguity, this setup offers precise, deterministic control over object density in the generated images. We generate board configurations where target objects (black stones) are placed at random, non-overlapping coordinates. To preclude solutions based on global pixel statistics (e.g., total foreground area), we introduce distractor objects (white stones) with cardinalities varying by up to ±30 relative to the target; this forces the model to perform specific feature attention rather than global density estimation. Crucially, we calibrate the stone sizes (14 × 14 pixels) to match the VLM’s input patch size, ensuring a theoretically clean mapping between distinct objects and token representations.

Decoupled Training Curriculum. To test whether linguistic priors automatically ground unseen visual inputs, we emulate VLM pretraining dynamics via a decoupled curriculum that creates a deliberate distribution shift between modalities. In Phase 1 (Language Pretraining), we equip the decoder with symbolic counting capabilities by training exclusively on text sequences (length 361) containing up to N = 99 target characters (see Fig. 13 in § D.2.2 for details). This ensures the model possesses the requisite algorithmic priors and output tokens independent of vision. In Phase 2 (Vision Alignment), we finetune the full VLM on visual counting but strictly limit the dataset to N ∈ [0, 49] (see Fig. 14 in § D.2.2 for details). This truncation creates a critical visual extrapolation regime (50 − 99): a zone where the model possesses the labels (from Phase 1) but has never observed their corresponding visual densities. To preserve the textual priors during alignment, we apply a reduced learning rate to the decoder. Full hyperparameters are provided in § D.2.1.

Evaluation Protocol. We evaluate the VLM on three sets: (1) In-Distribution $( I D , 0 - 4 9 )$ : Interpolation within the visual training support. (2) Visual Extrapolation (VE, 50 − 99): The critical regime where visual density is novel but labels are known. Failure here indicates a failure of crossmodal grounding. (3) Full Extrapolation (FE, 100−120): A regime where both visual density and text labels are unseen. For prediction, we employ standard greedy decoding with an End-of-Sequence (EOS) token, treating counting as an open-ended generation task rather than classification.

# 2.2. Study 2: Validation on a State-of-the-Art VLM

Model Selection. We extend our analysis to Qwen3-VL-32B-Instruct (Bai et al., 2025), a state-of-the-art openweights VLM that exhibits strong visual counting capabilities. We focus exclusively on Qwen3-VL because meaningful extrapolation analysis requires a baseline competence; other open-source models failed to achieve sufficient accuracy even in the low-number regime $( N < 1 0 )$ on our diagnostic task. Analyzing this architecture allows us to verify if the “fractured magnitude” phenomenon persists despite trillion-token scale pretraining and instruction tuning.

Dataset Configuration. We adapt the synthetic Go board generator to a $6 \times 6$ grid format. Correspondingly, the target object number ranges from 0 to 20, with distractors (white stones) up to ±5 relative to the target. This resolution is empirically calibrated to the effective capacity of the pretrained model, ensuring the task complexity lies at the boundary of its capabilities. To facilitate precise probing, we align the object dimensions with the model’s patch processing: since Qwen3-VL merges 2 × 2 sets of 16 × 16 patches into a single token, we scale each stone to 32 × 32 pixels to strictly align object boundaries with token inputs.

Prompt Design. We evaluate the model in a zero-shot setting using Chain-of-Thought (CoT) prompting to elicit maximal reasoning performance. Full prompt templates and generated response examples are detailed in § D.2.2.

# 2.3. Diagnostic Tool: Hidden Number Probing

To diagnose why a model fails (blindness vs. decoding failure), we employ linear probing to extract the hidden number $( N _ { H } )$ : the quantity information physically present in the visual token representations.

Let $\textbf { Z } = \ \{ z _ { 1 } , z _ { 2 } , \ldots , z _ { L } \}$ denote the sequence of patch output (d-dimension) embeddings from the visual encoder $\mathcal { M } _ { V } , \mathrm { e . g . } , L = 1 9 \times 1 9 = 3 6 1$ corresponds to the number of visual tokens. We train a binary linear probe $f _ { \mathrm { p r o b e } } :$ $\mathbb { R } ^ { d }  \{ 0 , 1 \}$ to detect the presence of a black stone at the grid position corresponding to token $z _ { i } .$ . Crucially, $f _ { \mathrm { p r o b e } }$ is trained exclusively on the ID dataset $( N _ { G } \le 4 9 )$ to test generalization. The hidden number $( N _ { H } )$ is obtained by aggregating the probe predictions across the full sequence:

$$
N _ {H} = \sum_ {i = 1} ^ {L} f _ {\text { probe }} (z _ {i}). \tag {1}
$$

While defined here for the visual encoder output, this probing methodology can be applied to the hidden states of any layer in the network. By comparing the internal percept $( N _ { H } )$ , the generated symbol $( N _ { P } )$ , and the ground truth $( N _ { G } )$ , we isolate the failure mechanism via two metrics:

• Vision Gap (i.e., recognition error), formally $| N _ { H } - N _ { G } | ,$ which measures the errors brought by the vision module. If $N _ { H } \approx N _ { G } ,$ it means there is no error in vision module.   
• Language Gap (i.e., numerosity and articulation error), formally $| N _ { H } - N _ { P } |$ , which measures the errors brought by the language module. If $N _ { H } \approx N _ { P } ,$ it means there is no such error in the language module.

# 3. Synthetic Extrapolation Laboratory

In this section, we deconstruct the mechanics of visual counting failure using our controlled synthetic environment. We first quantify the extent of the generalization gap (§ 3.1). We then systematically investigate the three hypothesized bottlenecks: visual individuation (§ 3.2), magnitude awareness (§ 3.3), and symbolic mapping (§ 3.4).

# 3.1. The Baseline Paradox

We aim to determine whether a VLM can zero-shot transfer its algorithmic counting knowledge from language to vision. Specifically, if a model has acquired the recursive rule of counting in the text domain (up to $N = 9 9 )$ but has strictly limited visual exposure (up to $N = 4 9 )$ , can it bridge this modality gap to enumerate 50 objects? This establishes the baseline behavior for our Visual Extrapolation (VE) regime.

Experiment Design. We evaluate the synthetic VLM on two parallel tasks across the full range of $N \in [ 0 , 1 2 0 ]$ ] to isolate modality-specific capabilities. For textual counting, the model is prompted to count specific characters within a text sequence, testing its counting ability in the language domain. For visual counting, the model must predict the total number of black stones on a provided Go board image. We report (Exact Match) accuracy across three defined regimes: In-Distribution $( \mathrm { I D } , 0 - 4 9 )$ , Visual Extrapolation $( \mathrm { V E } , 5 0 - 9 9 )$ , and Full Extrapolation (FE, 100 − 120).

![](images/c5acbd04c146aed10eb38b6cf623f52d75b251ea806745eb708ac752c333e0b6.jpg)

<details>
<summary>line</summary>

| Number of Objects | Vision Counting (%) | Text Counting (%) |
| ----------------- | ------------------- | ----------------- |
| 0                 | 100                 | 0                 |
| 50                | 100                 | 100               |
| 100               | 0                   | 100               |
| 120               | 0                   | 0                 |
</details>

Figure 2. The Baseline Paradox: Linguistic competence does not ensure visual ability. While textual counting (bronze area) generalizes perfectly to $N = 9 9 .$ , visual counting (red line) collapses to near-zero immediately outside the training distribution $( N > 4 9 )$ . This stark dissociation in the Visual Extrapolation (VE) regime demonstrates that the model fails to apply its known symbolic labels to unseen visual quantities.

Experiment Results. As illustrated in Fig. 2, the results reveal a stark modular dissociation. The model achieves perfect accuracy (100%) on the textual counting task throughout both the ID and VE regimes, confirming that the language decoder has successfully learned the recursive successor function up to $N = 9 9$ . In contrast, visual counting performance matches this perfection only within the visual training support (ID). Crucially, the moment the quantity exceeds the training boundary $( N > 4 9 )$ , visual accuracy suffers a catastrophic collapse to 0%. The only deviation from this flatline is a singular, anomalous spike at $N = 9 9$ , likely a boundary artifact where the model defaults to the highest number seen in its text training.

Takeaway. These results establish the baseline paradox: the model possesses the algorithmic capacity to count (proven by textual performance) yet appears effectively “blind” in the visual domain. The failure to extrapolate suggests a breakdown in the cross-modal processing chain. To pinpoint the bottleneck, we formulate three competing hypotheses regarding the locus of failure:

• Hypothesis A (stage 1): The bottleneck lies in visual individuation. The visual encoder fails to resolve distinct object features in dense, unseen configurations, preventing the quantity signal from entering the system.   
• Hypothesis B (stage 2): The bottleneck lies in magnitude awareness. While the encoder captures the objects, the abstract quantity signal dissipates or degrades as it propagates through the deep language decoder layers.   
• Hypothesis C (stage 3): The bottleneck lies in symbolic

![](images/4f3781c629880824aae50d09798b92a40d9b2ac8126facddb0deda4fe5b8975e.jpg)

<details>
<summary>line</summary>

| Number of Objects | Vision Gap | Language Gap |
| ----------------- | ---------- | ------------ |
| ID                | 10         | 0            |
| VE                | 10         | 10           |
| FE                | 10         | 10           |
</details>

Figure 3. Visual representations remain robust while symbolic decoding collapses. The Vision Gap $( | N _ { G } - N _ { H } | )$ remains zero across all regimes, proving the linear probe recovers the true count even during extrapolation. In contrast, the Language Gap $( | N _ { P } -$ $N _ { H } | )$ explodes immediately when $N \geq 5 0$ . This confirms the bottleneck is not perceptual blindness, but a failure to map valid visual magnitudes to the correct symbolic tokens.

mapping. The model retains the magnitude in its latent states but lacks the learned projection to map the visual representation to the corresponding symbolic token.

# 3.2. Hypothesis A: Object Individuation

The first logical suspect for the “baseline paradox” is perceptual crowding: the inability of the visual encoder to distinguish individual objects in dense scenes that exceed training examples. If the model cannot “see” 75 objects, it certainly cannot count them. To determine if the synthetic VLM is simply blind to higher quantities, we bypass the language decoder and probe the information content of the visual output embeddings directly.

Experiment Design. We employ the hidden number probing technique defined in § 2.3. We train a linear probe $( f _ { \mathrm { p r o b e } } )$ on the frozen output of the visual encoder to predict object density. Crucially, this probe is trained only on the ID data $( N \leq 4 9 )$ to prevent any information leakage from the extrapolation regimes. We then evaluate this probe across all three sets (ID, VE, FE) to extract the hidden number $( N _ { H } )$ and compute our two diagnostic metrics: (1) The Vision Gap $( | N _ { H } - N _ { G } | ) $ : measuring perceptual error. (2) The Language Gap $( | N _ { H } - N _ { P } | )$ : measuring the alignment error between the encoder’s percept and the final token.

Experiment Results. As illustrated in Fig. 3, the results reveal a striking dissociation between perception and generation. The Vision Gap remains effectively zero across the entire spectrum (0 − 120). The visual encoder maintains a linearly separable representation of quantity well into the FE regime, demonstrating that the visual features for “120 objects” are distinct and readable even though the model has never been trained on images with more than 49 objects. However, the Language Gap explodes precisely at the boundary of the training support $( N = 5 0 )$ . While the encoder “sees” the correct number $( N _ { H } \approx N _ { G } )$ , the language decoder fails to map the recognized objects to the final generated symbolic token $( N _ { P } \neq N _ { H } )$ .

Causality Verification. To confirm that the probed features are the actual causal drivers of counting rather than spurious correlations, we perform an intervention analysis. By masking the latent representations of $k \in [ 1 , 5 ]$ tokens identified as black stones by our probe (within the ID setting), we observe that the model’s prediction drops deterministically to exactly $N _ { G } - k$ . This confirms a direct causal link between the features isolated by our probe and the model’s final counting output. Detailed experimental settings and results are provided in § D.1.1.

Takeaway. We reject Hypothesis A. The visual backbone successfully individuates objects far beyond the training distribution, confirming that the bottleneck is not perceptual. Since the valid quantity signal $( N _ { H } )$ successfully enters the system, the failure must stem from downstream processing rather than blindness. Having established that the model sees the quantity, we proceed to test Hypothesis B: does the model (i.e., language decoder) possess a sense of magnitude, or does the signal dissipate during quantity reasoning?

# 3.3. Hypothesis B: Latent Magnitude Awareness

To detect if the language module possesses a robust sense of numerosity, we investigate whether the quantity signal persists in its hidden layers and if it can utilize these internal magnitude states for logical discrimination tasks, even when it cannot decode them into specific number tokens.

Experiment Design. We evaluate magnitude awareness via a Comparative Counting task, reformulating enumeration as a binary classification problem to bypass symbolic articulation bottlenecks. The model receives two inputs, either two character lists (Text-to-Text) or one image and one character list (Vision-to-Text), and must generate a boolean token (“True” or “False”) indicating if they share the same cardinality (see examples in Figs. 15 and 16). We construct a balanced dataset with equal positive and negative pairs to prevent heuristic guessing. Crucially, consistent with our counting curriculum (§ 2.1), the Vision-to-Text training is restricted to visual quantities $N \leq 4 9$ , while evaluation extends to the unseen VE regime $( N \in [ 5 0 , 9 9 ] )$ . Success in this regime would prove that the model can map a novel visual magnitude to a known textual magnitude, even if it cannot explicitly name it.

![](images/c55bf777e660c1a35132a8e4b49dfc781321ccc11b7f0bc42e6c21c93e603a9a.jpg)

<details>
<summary>line</summary>

| Number of Objects | Vision Counting Accuracy (%) | Text Counting Accuracy (%) |
| ----------------- | ----------------------------- | -------------------------- |
| 0                 | 100                           | 100                        |
| 20                | 100                           | 100                        |
| 40                | 100                           | 100                        |
| 60                | 100                           | 100                        |
| 80                | 95                            | 95                         |
| 100               | 75                            | 75                         |
| 120               | 55                            | 55                         |
</details>

Figure 4. Models can compare quantities they cannot name. In the binary magnitude comparison task, the model maintains high accuracy (> 90%) for visual inputs throughout the VE regime (50 − 99), despite having 0% accuracy in explicit enumeration. This proves the model possesses robust magnitude awareness; the failure is isolated strictly to the generative decoding stage.

Experiment Results. As shown in Fig. 4, the model exhibits remarkable latent robustness. Despite achieving 0% accuracy in explicitly articulating quantities in the VE regime (50 − 99), it successfully performs cross-modal comparison, e.g., correctly matching an image of 75 stones to a text sequence of 75 characters, with near-perfect accuracy. This performance only degrades in the FE regime $( N \ge 1 0 0 )$ , where neither visual density nor textual quantity was observed. To further validate this, we provide auxiliary linear probing results in § D.1.2, which confirm that linearly separable magnitude representations are preserved with 100% fidelity throughout the language decoder layers.

Takeaway. We reject Hypothesis B. The model possesses robust magnitude awareness, successfully utilizing abstract cardinality for quantity reasoning well beyond its training distribution. This proves that the quantity signal does not dissipate during cognitive processing. With perception (Stage 1) and numerosity (Stage 2) cleared, the bottleneck is isolated to the final stage. We thus proceed to investigate Hypothesis C: a failure in symbolic mapping.

# 3.4. Hypothesis C: Symbolic Mapping

Having rejected perceptual failure (§ 3.2) and quantity reasoning loss (§ 3.3), we turn to Hypothesis C (fractured magnitude hypothesis): a structural failure in projecting the valid magnitude onto the correct symbolic token. We investigate this through two complementary analyses: first examining the output topology to see how it fails, and second examining the internal circuits to understand why.

![](images/e55aa9b3ed24ffdfef7da33094c9b9d9345e1df5e7a0f54d115b18e3f8749998.jpg)

<details>
<summary>bar</summary>

| Predicted Number (Np) | Frequency (%) |
| :--- | :--- |
| 9 | 27.79 |
| 49 | 8.59 |
| 56 | 1.84 |
| 58 | 1.42 |
| 59 | 2.87 |
| 90 | 12.02 |
| 95 | 3.09 |
| 96 | 1.44 |
| 97 | 0.85 |
| 98 | 2.07 |
| 99 | 32.41 |
| 949 | 3.91 |
| 999 | 1.63 |
| Others | 8.23 · 10⁻² |
</details>

Figure 5. Decoding failure manifests as mode collapse. Instead of random noise, prediction errors cluster tightly around specific “attractor” tokens: the visual training boundary (49), textual priors (90, 99), and hallucinations (9). This confirms that in the absence of a valid cross-modal path, the model abandons the counting algorithm and reverts to retrieving memorized statistical priors.

# 3.4.1. ANALYSIS I: ERROR TOPOLOGY

Experiment Design. We verify if the mapping failure is stochastic (random noise) or structural (systematic misalignment). We collect the model’s predictions $( N _ { P } )$ across the entire extrapolation set and analyze the error distribution. If the model suffers from uncertainty, errors should be around the ground truth. If the generative mapping path is broken, we expect the model to collapse to specific statistical priors or “attractors” from its training distribution.

Experiment Results. As shown in Fig. 5, the error distribution is highly non-random. Rather than exhibiting Gaussian noise, the model collapses to specific “attractor” tokens: the boundary of the visual training data (49), the upper bound of the textual pretraining (99, 90), or low-frequency hallucinations (e.g., 9). This error pattern explains the anomalous spike in Fig. 2 (at N = 99) and confirms that in the absence of a learned mapping path, the model abandons the counting algorithm entirely, retrieving memorized statistical priors instead.

Takeaway. The non-Gaussian nature of the errors suggests that the model is not merely “unsure”; it is fundamentally ungrounded. When the input magnitude $( N _ { H } )$ falls into the extrapolation regime, the projection layer $f ( N _ { H } )  N _ { P }$ fails to trigger, causing the model to default to known highprobability tokens.

# 3.4.2. ANALYSIS II: CIRCUIT DIVERGENCE

Experiment Design. To determine if this grounding failure stems from physical circuit isolation, we trace the activations of the language module’s attention heads. We employ a greedy pruning strategy to identify “critical heads”: the minimal set of attention heads required to maintain predic-

![](images/1c26652668b96c5caad05b2019c2632b18dcf26e420ef95ef44c42034dc7bad9.jpg)  
Figure 6. Visual and textual counting rely on disjoint neural circuits. Top: Attention heatmaps reveal distinct activation patterns for visual (left) vs. textual (right) counting. Bottom: Quantitative analysis (N = 6400, 12.5% test data) confirms that the set of critical attention heads is non-overlapping in 95.7% of cases. This provides mechanistic evidence for the Fractured Magnitude Hypothesis: the model lacks a unified number space, processing visual and textual quantities on disjoint statistical manifolds.

tion accuracy. We sequentially drop heads (masking their output) from top to bottom layers and measure the performance drop. We then calculate the intersection of these critical head sets between the visual counting task and the textual Ccounting task to quantify their functional overlap.

Experiment Results. The heatmap analysis (Fig. 6) uncovers a startling separation: visual counting and textual counting activate almost entirely disjoint sets of attention heads. In 95.7% of cases, the neural circuits responsible for processing quantity in vision do not overlap with those used for text. This indicates that the model is effectively running two different sub-programs depending on the input modality, with no shared “counting circuit” to bridge them.

Takeaway. These results confirm Hypothesis C (fractured magnitude hypothesis). The VLM has not learned a universal number space. Instead, it has acquired two separate, modality-specific statistical manifolds: a “visual magnitude” and a “textual magnitude”. Because these manifolds reside in disjoint subspaces (evidenced by the disjoint attention heads), the model cannot generalize the mapping function to unseen quantities. While perception generalizes perfectly and magnitude awareness generalizes robustly, the symbolic mapping link is fractured. The “concept of $7 5 ^ { , , }$ in vision has no neural bridge to the “concept of $7 5 ^ { , , }$ in text.

![](images/926668e872b02c2b9a6a850c1c6add7c9047178bcf4a5f2f0aee3e721938702a.jpg)

<details>
<summary>line</summary>

| Number of Objects | Vision Counting | Text Counting |
| ----------------- | --------------- | ------------- |
| 0                 | 100             | 100           |
| 4                 | 60              | 95            |
| 8                 | 30              | 60            |
| 12                | 15              | 50            |
| 16                | 5               | 45            |
| 20                | 0               | 55            |
</details>

Figure 7. The extrapolation gap persists in foundation models. While Text Counting remains robust (> 50%), Vision Counting collapses immediately after the subitizing range $( N > 5 )$ , approaching zero by $N = 1 5$ . This confirms that even trillion-token scale models fail to ground their algorithmic counting rules visually, treating the modalities as disjoint capabilities.

# 4. Real-World Interpolation Validation

To verify whether the bottleneck observed in our synthetic laboratory generalizes to large-scale, pretrained architectures, we evaluate Qwen3-VL. Analyzing this state-of-theart foundation model allows us to test if trillion-token pretraining and RLHF alignment are sufficient to overcome the fractured magnitude phenomenon.

# 4.1. Persistence of the Extrapolation Gap

We first validate if the “baseline paradox” persists in a foundation model. If Qwen3-VL exhibits the same dissociation between textual and visual counting, it suggests the failure is structural rather than an artifact of scale.

Experiment Design. We adopt the evaluation protocol from § 3.1, assessing both visual counting (black stones) and textual counting (characters). The only modification is the dataset: we use 6 × 6 Go boards to align with the model’s effective resolution capacity.

Experiment Results. As shown in Fig. 7, the results strikingly mirror our synthetic findings. Textual counting remains robust (100% accuracy on small numbers), whereas visual counting exhibits a precipitous decline, flatlining near random guessing for $N > 8 .$ . This confirms that the counting bottleneck is structural; linguistic fluency does not automatically ground visual quantities.

Takeaway. We confirm that the baseline paradox is a structural characteristic of VLM architectures, not merely an artifact of scale. Despite trillion-token pretraining, the model exhibits the same dissociation: algorithmic competence in text versus stochastic failure in vision. Having established this baseline, we proceed to verify if the mechanism of failure follows the same diagnostic path (Hypotheses A → C) as our synthetic laboratory.

![](images/f7ab819c6772a2cef18c23b8934ff7a90c331a2173b32de1dfe98c7389d07100.jpg)

<details>
<summary>line</summary>

| Number of Objects | Vision Gap | Language Gap |
| ----------------- | ---------- | ------------ |
| 0                 | 0          | 0            |
| 4                 | 0          | 0.5          |
| 8                 | 0          | 1.5          |
| 12                | 0          | 3.0          |
| 16                | 0          | 5.5          |
| 20                | 0          | 4.0          |
</details>

Figure 8. Visual encoders retain rich signal. The Vision Gap $( | N _ { G } - N _ { H } | )$ remains negligible (< 0.1), indicating the probe recovers the true count even when the model fails. In contrast, the Language Gap $( | N _ { P } - N _ { H } | )$ ) diverges rapidly. This confirms that while real-world encoders introduce minor noise, the primary driver of failure remains the symbolic decoding bottleneck.

# 4.2. Hypothesis A: Object Individuation

A common critique of VLMs is that they suffer from “perceptual crowding”: an inability to distinguish individual features in complex real-world encoders (e.g., SigLIP or CLIP) compared to cleaner synthetic encoders. We investigate whether Qwen3-VL’s failure to count stems from being effectively “blind” to the objects, or if the hidden number $( N _ { H } )$ persists in the latent space despite the incorrect output.

Experiment Design. We replicate the hidden number probing methodology from § 3.2. Specifically, we train a linear regression probe on the feature map outputs to predict object density per token and aggregate them to estimate the hidden quantity ${ N _ { H } } . ^ { 2 }$ We then compare two error metrics: the Vision Gap and the Language Gap.

Experiment Results. The results in Fig. 8 reveal a crucial distinction between noise and failure. Unlike the synthetic setting where recognition was perfectly lossless, the realworld encoder exhibits a non-zero Vision Gap, reflecting the inherent noise of processing high-dimensional natural image features. However, this error is negligible $( < 0 . 1$ on average) compared to the Language Gap, which diverges rapidly as soon as the quantity exceeds the immediate subitizing range $( N > 4 )$ . The probe successfully tracks the hidden number $( N _ { H } \approx N _ { G } )$ , proving that the deviation in the final prediction is driven almost entirely by the language module failure $( | N _ { H } - N _ { P } | )$ . It is worth noting that while our controlled setting isolates numerosity, real-world counting tasks often involve complex object recognition challenges (e.g., occlusion, variance) which may introduce additional perceptual errors not captured here (§ B).

![](images/74ad06aaa490574f3fbea5be1ab9018880671d4f143ebab2a2aaac7559acf21b.jpg)

<details>
<summary>line</summary>

| Layer Index | Probe Accuracy (%) |
| ----------- | ------------------ |
| 0           | 100                |
| 62          | 65                 |
</details>

Figure 9. Magnitude information persists deep into the model. Probing reveals that visual module (99.4%) and the first 40 language module layers (> 99%) maintain precise quantity representations. The signal decay is restricted to the final layers, confirming that the failure is not due to “reasoning drift” or forgetting, but occurs strictly during the final projection to symbolic logits.

Takeaway. We reject Hypothesis A. Consistent with our synthetic findings, the visual encoder is not the bottleneck. While real-world features introduce minor perceptual noise, the hidden number remains distinct and linearly separable $( N _ { H } \approx N _ { G } )$ . The catastrophic drop in performance is not due to blindness; the model successfully encodes the visual quantity but fails to utilize it.

# 4.3. Hypothesis B: Magnitude Awareness

We next investigate the process in language decoder. Does the abstract “sense” of quantity dissipate as information propagates through the deep transformer layers? We probe the network’s depth to determine if the counting failure stems from “awareness drift” or catastrophic forgetting within the language module backbone.

Experiment Design. For visual counting, we train linear probes on the output hidden states of every layer in the architecture: from the final layer of the vision encoder through the first 63 layers of the language decoder.3 This allows us to trace the fidelity of the magnitude signal $( N _ { H } )$ as it flows through the network’s computation graph.

Experiment Results. Fig. 9 presents the probe accuracy across the network depth. The results are unequivocal: magnitude information is robustly preserved for the vast majority of the computation. The visual module passes a precise signal (99.37%) to the language model. Remarkably, the language module maintain this fidelity with near-perfect accuracy (> 99%) for the first 40 layers. We only observe a mild degradation in the final 20 layers, corresponding to the transition from abstract magnitude awareness to the specific

![](images/800d9a58c1862d89b11f6be1a3b67a5b70e6e4ba4bba9c2f20d24dd438838c7c.jpg)

<details>
<summary>scatter</summary>

| Ground Truth (NG) | Predicted Number (NP) |
| ----------------- | --------------------- |
| 0                 | 0                     |
| 5                 | 5                     |
| 10                | 10                    |
| 15                | 15                    |
| 20                | 20                    |
</details>

(a) Visual Counting.

![](images/5c30dfc24cc21766ea0e1677e586c08896fc4390ec60c7d9a6cc1e371b5116a4.jpg)

<details>
<summary>scatter</summary>

| Ground Truth (NG) | Value |
| ----------------- | ----- |
| 0                 | 0     |
| 5                 | 5     |
| 10                | 10    |
| 15                | 15    |
| 20                | 20    |
</details>

(b) Textual Counting.   
Figure 10. Visual errors are stochastic guesses; textual errors are systematic biases. We visualize the prediction error topology. (a) Visual Counting: Errors form a high-variance, diffuse cloud, characteristic of ungrounded estimation. (b) Textual Counting: Errors are tightly clustered along the diagonal with a constant offset, revealing a systematic algorithmic bias. This dichotomy confirms that the model employs fundamentally different, unconnected mechanisms for processing quantity across modalities.

(and fractured) output token space.

Takeaway. We reject Hypothesis B. The model effectively “knows” the cardinality of the scene throughout its decoder. The fact that accuracy remains at ceiling levels for over twothirds of the network depth rules out magnitude awareness capacity or forgetting as the primary bottleneck. While the mild degradation in the final layers suggests some noise is introduced as the model prepares to generate output, this marginal error is insufficient to explain the catastrophic collapse in visual counting performance. This indicates that the failure is not one of intellect, but of articulation: the final symbolic mapping could be broken.

# 4.4. Hypothesis C: Symbolic Mapping

While § 4.1 proves the model can count algorithmically in text, accuracy metrics alone are insufficient to reveal the mechanism of the visual failure. Does the model attempt to run this same counting algorithm on visual inputs (suffering merely from execution noise), or does it abandon the procedure entirely in favor of ungrounded estimation?

Experiment Design. We apply the error topology analysis from § 3.4. We diagnose the failure mechanism by contrasting the statistical signatures of variance (spread) and bias (offset) in the prediction heatmaps (NP vs. $N _ { G } )$ . A tight clustering with consistent bias (e.g., N ± 1) indicates algorithmic execution, implying the rule is active but misaligned. In contrast, a diffuse, cloud-like spread indicates heuristic estimation, implying the model has reverted to ungrounded guessing.

Experiment Results. Fig. 10 illustrates a stark dichotomy in error topology. Visual counting errors form a diffuse, high-variance cloud, where the model frequently over- or under-estimates by wide margins. The mean prediction (Red Line) drifts significantly from the ground truth diagonal, a pattern characteristic of approximate estimation rather than precise enumeration. In sharp contrast, textual counting exhibits errors that are tightly clustered along the diagonal. The model often predicts exactly N ± 1, maintaining the correct slope but suffering from a constant offset. This low-variance, systematic bias indicates a working algorithm estimation with a calibration error (likely due to tokenization artifacts, e.g., counting the start-of-sentence token), rather than a fundamental loss of magnitude awareness.

Takeaway. These results confirm Hypothesis C in a foundation model setting. The analysis of Qwen3-VL validates the fractured magnitude hypothesis in a real-world setting. The model essentially operates with two distinct cognitive processes: a robust textual counter that executes a precise (albeit biased) algorithm, and a stochastic visual estimator that relies on ungrounded heuristics. Crucially, the model possesses both a precise visual magnitude (proven by probing) and a systematic counting algorithm (proven by accuracy), yet it fails to bridge them. The universal number space required to map the visual representation to the textual token is missing, forcing the model to guess stochastically despite having the correct answer encoded in its latent space.

# 5. Conclusion

We diagnose the bottleneck in visual counting, moving from the controlled synthetic laboratory to the state-of-the-art foundation model. Our three-stage decomposition rules out failures in visual individuation and magnitude awareness: models maintain robust latent quantities and support crossmodal comparison even when generation fails. Instead, the collapse is strictly confined to symbolic mapping. This supports a fractured magnitude hypothesis: disjoint visual and textual manifolds prevent the grounding of unseen quantities. Consequently, data scaling is insufficient; systematic generalization requires architectural priors that enforce unified, isomorphic number spaces across modalities.

# Acknowledgment

We thank the reviewers for their constructive feedback. We also thank Jiaoda Li and Mubashara Akhtar for their valuable input during the early stages of this work. Yifan Hou is supported by the Swiss Data Science Center PhD Grant (P22-05). This research was further supported by the ETH AI Center through an ETH AI Center doctoral fellowship to Junling Wang.

# Impact Statement

This paper presents foundational research aimed at understanding the systematic generalization capabilities and failure modes of Multimodal Large Language Models (e.g., VLMs). By diagnosing the “Fractured Magnitude” bottleneck, our work highlights a critical discrepancy between a model’s internal representation (which is often accurate) and its generated output (which can be hallucinated).

Societal Consequences. This work focuses on mechanistic interpretability and does not directly introduce new applications with immediate dual-use risks. However, improving the counting and reasoning reliability of VLMs has positive downstream implications for scientific analysis, automated inventory, and data processing tasks where precision is critical. Conversely, as with all advancements in reasoning capabilities, reliable VLMs could theoretically be employed in autonomous surveillance systems; however, our work identifies current limitations rather than enabling immediate deployment capabilities.

Environmental Impact. Our study utilizes a hybrid approach, leveraging lightweight synthetic models (trained from scratch) to derive hypotheses before validating them on pre-trained foundation models. This methodology significantly reduces the carbon footprint compared to studies that require full-scale retraining of large foundation models.

We do not foresee any specific negative societal consequences or ethical concerns arising directly from the publication of this analysis.

# References

Adi, Y., Kermany, E., Belinkov, Y., Lavi, O., and Goldberg, Y. Fine-grained analysis of sentence embeddings using auxiliary prediction tasks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. URL https: //openreview.net/forum?id=BJh6Ztuxl.

Alain, G. and Bengio, Y. Understanding intermediate layers using linear classifier probes. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Workshop Track Proceedings. OpenReview.net, 2017. URL https: //openreview.net/forum?id=HJ4-rAVtl.

Anil, C., Wu, Y., Andreassen, A., Lewkowycz, A., Misra, V., Ramasesh, V. V., Slone, A., Gur-Ari, G., Dyer, E., and Neyshabur, B. Exploring length generalization in large language models. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems 35: Annual

Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL https://dl.a cm.org/doi/10.5555/3600270.3603063.   
Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3- vl technical report. CoRR, abs/2511.21631, 2025. doi: 10.48550/ARXIV.2511.21631. URL https://doi. org/10.48550/arXiv.2511.21631.   
Bender, E. M. and Koller, A. Climbing towards NLU: on meaning, form, and understanding in the age of data. In Jurafsky, D., Chai, J., Schluter, N., and Tetreault, J. R. (eds.), Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020, pp. 5185–5198. Association for Computational Linguistics, 2020. doi: 10.18653/V1/20 20.ACL-MAIN.463. URL https://doi.org/10 .18653/v1/2020.acl-main.463.   
Beyer, L., Steiner, A., Pinto, A. S., Kolesnikov, A., Wang, X., Salz, D., Neumann, M., Alabdulmohsin, I., Tschannen, M., Bugliarello, E., Unterthiner, T., Keysers, D., Koppula, S., Liu, F., Grycner, A., Gritsenko, A. A., Houlsby, N., Kumar, M., Rong, K., Eisenschlos, J., Kabra, R., Bauer, M., Bosnjak, M., Chen, X., Minderer, M., Voigtlaender, P., Bica, I., Balazevic, I., Puigcerver, J., Papalampidi, P., Henaff, O. J., Xiong, X., Soricut, ´ R., Harmsen, J., and Zhai, X. Paligemma: A versatile 3b VLM for transfer. CoRR, abs/2407.07726, 2024. doi: 10.48550/ARXIV.2407.07726. URL https: //doi.org/10.48550/arXiv.2407.07726.   
Carey, S. The origin of concepts. Journal of Cognition and Development, 1(1):37–41, 2000.   
Csordas, R., Irie, K., and Schmidhuber, J. The devil is ´ in the detail: Simple tricks improve systematic generalization of transformers. In Moens, M., Huang, X., Specia, L., and Yih, S. W. (eds.), Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pp. 619– 634. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.EMNLP-MAIN.49. URL

https://doi.org/10.18653/v1/2021.emn lp-main.49.   
Dedekind, R. Was sind und was sollen die zahlen? In Was sind und was sollen die Zahlen?. Stetigkeit und Irrationale Zahlen, pp. 1–47. Springer, 1965.   
Dehaene, S. The number sense: How the mind creates mathematics. OUP USA, 2011.   
Deletang, G., Ruoss, A., Grau-Moya, J., Genewein, T., Wen- ´ liang, L. K., Catt, E., Cundy, C., Hutter, M., Legg, S., Veness, J., and Ortega, P. A. Neural networks and the chomsky hierarchy. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=WbxH AzkeQcn.   
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=Yicb FdNTTy.   
Falk, T., Mai, D., Bensch, R., C¸ ic¸ek, O., Abdulkadir, A., ¨ Marrakchi, Y., Bohm, A., Deubner, J., J ¨ ackel, Z., Seiwald, ¨ K., et al. U-net: deep learning for cell counting, detection, and morphometry. Nature methods, 16(1):67–70, 2019.   
Fodor, J. A. and Pylyshyn, Z. W. Connectionism and cognitive architecture: A critical analysis. Cognition, 28(1-2): 3–71, 1988.   
Guan, T., Liu, F., Wu, X., Xian, R., Li, Z., Liu, X., Wang, X., Chen, L., Huang, F., Yacoob, Y., Manocha, D., and Zhou, T. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 14375–14385. IEEE, 2024. doi: 10.1109/CVPR52733.2024.01363. URL https://doi.org/10.1109/CVPR52733. 2024.01363.   
Guo, X., Huang, Z., Shi, Z., Song, Z., and Zhang, J. Your vision-language model can’t even count to 20: Exposing the failures of vlms in compositional counting. CoRR, abs/2510.04401, 2025. doi: 10.48550/ARXIV.2510.04 401. URL https://doi.org/10.48550/arXiv .2510.04401.

Gurnee, W., Nanda, N., Pauly, M., Harvey, K., Troitskii, D., and Bertsimas, D. Finding neurons in a haystack: Case studies with sparse probing. Trans. Mach. Learn. Res., 2023, 2023. URL https://openreview.net/f orum?id=JYs1R9IMJr.   
Hewitt, J. and Liang, P. Designing and interpreting probes with control tasks. In Inui, K., Jiang, J., Ng, V., and Wan, X. (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pp. 2733–2743. Association for Computational Linguistics, 2019. doi: 10.18653/V1/D19-1275. URL https://doi.or g/10.18653/v1/D19-1275.   
Hsieh, C., Zhang, J., Ma, Z., Kembhavi, A., and Krishna, R. Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/pap er/2023/hash/63461de0b4cb760fc498e85 b18a7fe81-Abstract-Datasets\_and\_Benc hmarks.html.   
Huang, J., Liu, X., Song, S., Hou, R., Chang, H., Lin, J., and Bai, S. Revisiting multimodal positional encoding in vision-language models. CoRR, abs/2510.23095, 2025. doi: 10.48550/ARXIV.2510.23095. URL https: //doi.org/10.48550/arXiv.2510.23095.   
Hupkes, D., Dankers, V., Mul, M., and Bruni, E. Compositionality decomposed: How do neural networks generalise? J. Artif. Intell. Res., 67:757–795, 2020. doi: 10.1613/JAIR.1.11674. URL https://doi.org/10 .1613/jair.1.11674.   
Jiang, R., Liu, L., and Chen, C. Clip-count: Towards textguided zero-shot object counting. In El-Saddik, A., Mei, T., Cucchiara, R., Bertini, M., Vallejo, D. P. T., Atrey, P. K., and Hossain, M. S. (eds.), Proceedings of the 31st ACM International Conference on Multimedia, MM 2023, Ottawa, ON, Canada, 29 October 2023- 3 November 2023, pp. 4535–4545. ACM, 2023. doi: 10.1145/3581 783.3611789. URL https://doi.org/10.1145/ 3581783.3611789.   
Kazemnejad, A., Padhi, I., Ramamurthy, K. N., Das, P., and Reddy, S. The impact of positional encoding on length generalization in transformers. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine,

S. (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http: //papers.nips.cc/paper\_files/paper/2 023/hash/4e85362c02172c0c6567ce59312 2d31c-Abstract-Conference.html.

Lake, B. M. and Baroni, M. Generalization without systematicity: On the compositional skills of sequence-tosequence recurrent networks. In Dy, J. G. and Krause, A. (eds.), Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmassan, ¨ Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pp. 2879–2888. PMLR, 2018. URL http://proceedings.mlr. press/v80/lake18a.html.

Lempitsky, V. S. and Zisserman, A. Learning to count objects in images. In Lafferty, J. D., Williams, C. K. I., Shawe-Taylor, J., Zemel, R. S., and Culotta, A. (eds.), Advances in Neural Information Processing Systems 23: 24th Annual Conference on Neural Information Processing Systems 2010. Proceedings of a meeting held 6-9 December 2010, Vancouver, British Columbia, Canada, pp. 1324–1332. Curran Associates, Inc., 2010. URL https://proceedings.neurips.cc/paper /2010/hash/fe73f687e5bc5280214e0486b 273a5f9-Abstract.html.

Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen, J. Evaluating object hallucination in large visionlanguage models. In Bouamor, H., Pino, J., and Bali, K. (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pp. 292–305. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.20. URL https: //doi.org/10.18653/v1/2023.emnlp-main. 20.

Marcus, G. F. The algebraic mind: Integrating connectionism and cognitive science. MIT press, 2003.

OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023. doi: 10.48550/ARXIV.2303.08774. URL https: //doi.org/10.48550/arXiv.2303.08774.

Paiss, R., Ephrat, A., Tov, O., Zada, S., Mosseri, I., Irani, M., and Dekel, T. Teaching CLIP to count to ten. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pp. 3147– 3157. IEEE, 2023. doi: 10.1109/ICCV51070.2023.00294. URL https://doi.org/10.1109/ICCV51070. 2023.00294.

Piantadosi, S. T., Tenenbaum, J. B., and Goodman, N. D. Bootstrapping in a language of thought: A formal model of numerical concept learning. Cognition, 123(2):199– 217, 2012.   
Press, O., Smith, N. A., and Lewis, M. Train short, test long: Attention with linear biases enables input length extrapolation. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https: //openreview.net/forum?id=R8sQPpGCv0.   
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.   
Rahmanzadehgervi, P., Bolton, L., Taesiri, M. R., and Nguyen, A. T. Vision language models are blind. In Cho, M., Laptev, I., Tran, D., Yao, A., and Zha, H. (eds.), Computer Vision - ACCV 2024 - 17th Asian Conference on Computer Vision, Hanoi, Vietnam, December 8-12, 2024, Proceedings, Part V, volume 15476 of Lecture Notes in Computer Science, pp. 293–309. Springer, 2024. doi: 10.1007/978-981-96-0917-8\ 17. URL https:// doi.org/10.1007/978-981-96-0917-8\_17.   
Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolutional networks for biomedical image segmentation. In Navab, N., Hornegger, J., III, W. M. W., and Frangi, A. F. (eds.), Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015 - 18th International Conference Munich, Germany, October 5 - 9, 2015, Proceedings, Part III, volume 9351 of Lecture Notes in Computer Science, pp. 234–241. Springer, 2015. doi: 10.1007/978-3-319-24574-4\ 28. URL https://do i.org/10.1007/978-3-319-24574-4\_28.   
Team, G. Gemini: A family of highly capable multimodal models. CoRR, abs/2312.11805, 2023. doi: 10.48550/A RXIV.2312.11805. URL https://doi.org/10.4 8550/arXiv.2312.11805.   
Team, G. Gemma 3 technical report. CoRR, abs/2503.19786, 2025. doi: 10.48550/ARXIV.2503.19786. URL https: //doi.org/10.48550/arXiv.2503.19786.   
Tenney, I., Das, D., and Pavlick, E. BERT rediscovers the classical NLP pipeline. In Korhonen, A., Traum, D. R., and Marquez, L. (eds.), \` Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pp. 4593–4601. Association for Computational Linguistics, 2019. doi: 10.18653/V1/P19-1452. URL https://doi.or g/10.18653/v1/p19-1452.

Thrush, T., Jiang, R., Bartolo, M., Singh, A., Williams, A., Kiela, D., and Ross, C. Winoground: Probing vision and language models for visio-linguistic compositionality. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 5228–5238. IEEE, 2022a. doi: 10.1109/ CVPR52688.2022.00517. URL https://doi.org/ 10.1109/CVPR52688.2022.00517.   
Thrush, T., Jiang, R., Bartolo, M., Singh, A., Williams, A., Kiela, D., and Ross, C. Winoground: Probing vision and language models for visio-linguistic compositionality. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 5228–5238. IEEE, 2022b. doi: 10.1109/ CVPR52688.2022.00517. URL https://doi.org/ 10.1109/CVPR52688.2022.00517.   
Wallace, E., Wang, Y., Li, S., Singh, S., and Gardner, M. Do NLP models know numbers? probing numeracy in embeddings. In Inui, K., Jiang, J., Ng, V., and Wan, X. (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pp. 5306–5314. Association for Computational Linguistics, 2019. doi: 10.18653/V1/D19-1534. URL https://doi.org/10.18653/v1/D19-1534.   
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E. H., Le, Q. V., and Zhou, D. Chain-ofthought prompting elicits reasoning in large language models. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.n ips.cc/paper\_files/paper/2022/hash/9 d5609613524ecf4f15af0f7b31abca4-Abstr act-Conference.html.   
Xu, J., Le, H., Nguyen, V., Ranjan, V., and Samaras, D. Zero-shot object counting. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 15548– 15557. IEEE, 2023. doi: 10.1109/CVPR52729.2023.014 92. URL https://doi.org/10.1109/CVPR52 729.2023.01492.   
Yuksekg ¨ on¨ ul, M., Bianchi, F., Kalluri, P., Jurafsky, D., and ¨ Zou, J. When and why vision-language models behave like bags-of-words, and what to do about it? In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023.

OpenReview.net, 2023a. URL https://openrevi ew.net/forum?id=KRLUvxh8uaX.   
Yuksekg ¨ on¨ ul, M., Bianchi, F., Kalluri, P., Jurafsky, D., and ¨ Zou, J. When and why vision-language models behave like bags-of-words, and what to do about it? In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023b. URL https://openrevi ew.net/forum?id=KRLUvxh8uaX.   
Zhang, S. D., Tigges, C., Zhang, Z., Biderman, S., Raginsky, M., and Ringer, T. Transformer-based models are not yet perfect at learning to emulate structural recursion. Trans. Mach. Learn. Res., 2024, 2024. URL https: //openreview.net/forum?id=Ry5CXXm1sf.   
Zhang, Y., Zhou, D., Chen, S., Gao, S., and Ma, Y. Singleimage crowd counting via multi-column convolutional neural network. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pp. 589–597. IEEE Computer Society, 2016. doi: 10.1109/CVPR.2016.70. URL https://doi.org/10.1109/CVPR.2016.70.   
Zhang, Z., Zhang, A., Li, M., Zhao, H., Karypis, G., and Smola, A. Multimodal chain-of-thought reasoning in language models. CoRR, abs/2302.00923, 2023. doi: 10.48550/ARXIV.2302.00923. URL https://doi. org/10.48550/arXiv.2302.00923.

# A. Related Work

# A.1. Visual Counting and Numerosity

The task of estimating the number of objects in a visual scene has evolved from task-specific engineering to general-purpose reasoning. Traditional approaches treated counting as a density estimation or regression problem, utilizing dedicated architectures like U-Nets or crowd-counting density map regressors to handle high-density scenes (Lempitsky & Zisserman, 2010; Zhang et al., 2016; Ronneberger et al., 2015; Falk et al., 2019). While highly accurate, these models are restricted to specific categories (e.g., people, cars) and lack semantic flexibility. The advent of Vision-Language Models (VLMs) introduced the promise of open-vocabulary zero-shot counting (Jiang et al., 2023; Paiss et al., 2023; Xu et al., 2023). However, recent evaluations consistently demonstrate that while VLMs excel at object recognition, they struggle with precise enumeration, often exhibiting “numerosity blindness” or hallucination when counts exceed small subitizing ranges (N < 5) (Paiss et al., 2023; Hsieh et al., 2023; Rahmanzadehgervi et al., 2024; Guan et al., 2024; Beyer et al., 2024; Guo et al., 2025). Our work extends this line of inquiry by moving beyond performance benchmarking to identifying the precise architectural stage: recognition, numerosity, or articulation, where this capability collapses.

# A.2. Systematic Generalization and Extrapolation

Systematic generalization, the ability to apply learned rules to inputs outside the training distribution, is a known weakness of neural networks. In Natural Language Processing (NLP), this is often framed as “length generalization”: models trained on short arithmetic or logical sequences fail to process longer sequences at test time (Lake & Baroni, 2018; Hupkes et al., 2020; Press et al., 2022; Anil et al., 2022). Theoretical analyses suggest that Transformers tend to overfit to positional correlations rather than learning the recursive underlying algorithms (Csordas et al. ´ , 2021; Deletang et al. ´ , 2023; Kazemnejad et al., 2023; Zhang et al., 2024). In the multimodal domain, this challenge manifests as Visual Extrapolation: the inability to count quantities unseen during training. Unlike prior works that attempt to solve this via data scaling or prompting strategies (Wei et al., 2022; Zhang et al., 2023), we focus on the mechanistic failure of cross-modal grounding, demonstrating that the failure to extrapolate is not a loss of knowledge but a failure of alignment between modalities.

# A.3. Probing Multimodal Representations

Understanding the internal state of “black box” models is critical for diagnosing reasoning failures. Linear probing has emerged as a standard technique to extract latent information from frozen representations, revealing that models often encode features (e.g., syntax, semantics) that are not explicitly supervised (Alain & Bengio, 2017; Adi et al., 2017; Tenney et al., 2019; Hewitt & Liang, 2019). In the context of VLMs, probing has been used to assess spatial awareness, attribute binding, and object existence (Thrush et al., 2022b; Yuksekg ¨ on¨ ul et al. ¨ , 2023a; Li et al., 2023). Closely related to our work, (Wallace et al., 2019; Gurnee et al., 2023) investigated the “number sense” of language models, finding that number neurons exist but are often polysemantic. Our contribution diverges by applying probing specifically to the magnitude extrapolation regime. We provide the first evidence of a “Fractured Magnitude” phenomenon, where the visual encoding of number remains robust and linear even when the symbolic mapping of that number collapses entirely.

# B. Limitations

While our work offers robust mechanistic evidence for the Fractured Magnitude Hypothesis, we acknowledge several limitations in our experimental scope:

Assumption of Perceptual Success. Our evaluation framework focuses strictly on the algorithmic capacity to enumerate quantities, factoring out the confounding variable of object recognition. In wild scenarios, counting failures often stem from detection failures (e.g., occlusion or camouflage). We do not penalize the model for failing to recognize the target object class; however, our Stage 1 probing results confirm that for the chosen domain, the visual signal is present, justifying this isolation of the reasoning bottleneck.

Complexity of Visual Stimuli. We utilize Go game boards as a controlled environment to ensure precise ground truth and strictly enforce identical objects. Real-world visual counting is significantly more complex, requiring semantic filtering (e.g., “count the red apples but ignore the green ones”) and handling high intra-class variance. While our simplified setting yields a necessary upper bound on counting capability, it may not fully capture the noise and ambiguity of natural image datasets.

Model Availability and Scope. Our real-world validation is primarily conducted on Qwen3-VL-32B. While this was necessary because other open-weights models (e.g., LLaVA, previous Qwen iterations) lacked sufficient baseline counting capabilities to meaningfully test extrapolation, our conclusions strictly apply to this architecture. However, given that Qwen3-VL represents the current state-of-the-art in open-source VLMs, we believe these findings are indicative of broader architectural limitations in the Transformer paradigm.

Task Specificity vs. Generalization. Our analysis focuses on counting black stones, a specific feature that may appear with varying frequency in pretraining corpora. It is possible that VLMs exhibit better performance on highly frequent objects (e.g., “people” or “cars”) due to memorized density-label correlations. However, our objective is to measure the model’s expressive power and algorithmic generalization rather than its retrieval capacity. The failure to extrapolate on stones despite having a clear visual signal suggests a fundamental architectural deficit rather than a lack of domain-specific training data.

# C. Use of AI Tools

This work utilized Gemini for linguistic refinement, improving text clarity, formatting LATEX code. AI models were strictly limited to an assistive role and were not involved in research ideation, experimental design, theoretical derivation, or the interpretation of results.

# D. Supplementary

In this section, we provide details about our supplementary experiment results and our experiment settings.

# D.1. Experiment Results

We provide additional experiment results below.

# D.1.1. CAUSAL ANALYSIS

While linear probes confirm that magnitude information $( N _ { H } )$ is linearly separable within the Vision Encoder, probing is correlational; it does not guarantee that the model actively utilizes these specific features for its prediction. We aim to establish a rigorous causal link between the probed visual representations and the generated count $( N _ { P } )$ . If the features detected by our probe are indeed the drivers of counting, surgically suppressing them should result in a deterministic and predictable drop in the model’s output.

Experiment Design. We perform an activation steering (intervention) experiment on the frozen Vision Encoder outputs using the In-Distribution dataset $( N _ { G } \in [ 0 , 4 9 ] )$ , where the model’s baseline counting is 100% accurate. For a given input image, we utilize the probe to identify the patch embeddings corresponding to k specific black stones (where k ranges from 1 to 5). We mask the corresponding tokens for the k black stone in attention calculation. If this token is indeed responsible for the black stone counting, the counting number should be k smaller than the original number. Thus, we define the Steering Accuracy as the percentage of cases where the model’s new prediction $N _ { P } ^ { \prime }$ satisfies the subtraction logic: $N _ { P } ^ { \prime } = N _ { G } - k$ .

Experiment Results. As shown in Fig. 11, the results are definitive. For every intervention size $k \in [ 1 , 4 ]$ , the model achieves almost 100% Steering Accuracy. When $k = 5 .$ , the Steering Accuracy is only 77.2%. We presume this is caused by the attention distribution shift when masking that many tokens. For example, replacing the visual representation of $k = 3$ black stones with background noise causes the model to decrement its count by exactly $N _ { P } ^ { \prime } = N _ { G } - 3$ in every single instance. This indicates a precise, one-to-one functional dependency between the presence of these specific feature vectors and the arithmetic sum computed by the model.

Takeaway. This provides causal evidence that the model is not relying on spurious correlations or global texture statistics. The hidden number features identified by our probes are the true mechanistic inputs to the counting algorithm. Consequently, the downstream extrapolation failure cannot be attributed to the model ignoring the visual signal or attending to the wrong features; the model correctly ingests the “bricks” of the count, confirming the failure occurs strictly in the decoding logic.

![](images/72dc7efe24b50b9fc0f2b48eb9f4d69e4821d5fa334b0a8c0b9b37af4f0d9a22.jpg)

<details>
<summary>bar</summary>

| Intervention Number (k) | Accuracy for N_P' = N_G - k (%) |
| ----------------------- | -------------------------------- |
| 1                       | 100                              |
| 2                       | 100                              |
| 3                       | 100                              |
| 4                       | 95                               |
| 5                       | 75                               |
</details>

Figure 11. Black-stone patch embeddings causally drive counting via attention masking. On the In-Distribution set (baseline counting accuracy: 100%), we randomly sample k specific black stones in each image $( k \in [ 1 , 5 ] )$ , and mask the corresponding tokens in the attention computation. We report Steering Accuracy, defined as the percentage of examples where the intervened prediction matches the new groundtruth $N _ { G } ^ { \prime } = N _ { G } - k .$ . The model achieves near-perfect Steering Accuracy for $\mathsf { \bar { k } } \in [ 1 , 3 ]$ , shows the first noticeable degradation at $k = 4$ , and drops to 77.2% at $k = 5$ , consistent with attention distribution shifts when masking many tokens.

# D.1.2. LAYER-WISE HIDDEN NUMBER PROBING

While the Comparative Counting task (§ 3.3) demonstrates that the model can utilize magnitude information for binary decision-making, it treats the internal representations as a black box. To rigorously verify Hypothesis B, we need to determine if the “Hidden Number” $( N _ { H } )$ is preserved as a linearly separable feature throughout the network’s depth, or if the signal-to-noise ratio degrades as it propagates through the deep Transformer layers of the language decoder.

Experiment Design. We extend the linear probing methodology described in § 2.3 to a layer-wise analysis. We train a separate linear classifier $f _ { \mathrm { p r o b e } } ^ { ( l ) }$ on the output hidden states of every layer l in the architecture:

• Vision Encoder: Layers 1–2 (Final output used for Stage 1 analysis).   
• Language Decoder: Layers 1–2 (The reasoning backbone).

The probes are trained on the In-Distribution (ID) dataset $( N \leq 4 9 )$ and evaluated on the Visual Extrapolation (VE) dataset $( N \in [ 5 0 , 9 9 ] )$ ). High classification accuracy at layer l indicates that the magnitude information remains distinct and retrievable at that depth.

![](images/444f69943bcb1336a89d03eb810c183eb881189aa6375469c54c10910bb464ff.jpg)

<details>
<summary>bar</summary>

| Category | Probe Accuracy (%) |
| :--- | :--- |
| Vision (Last Layer) | 100 |
| Text (Second Last Layer) | 100 |
</details>

Figure 12. Latent quantity representations are perfectly linearly separable. Linear probes trained on both the vision and language modules achieve 100% accuracy. This confirms that distinct magnitude information persists across modalities, ruling out representation collapse as the failure mechanism.

Results. As illustrated in Fig. 12, the probes achieve 100% accuracy across all layers of both the Vision Encoder and the Language Decoder. This result is highly significant: it confirms that the representation of quantity does not “drift” or become entangled as it passes through the cross-modal attention layers. For every number $N \in [ 0 , 9 9 ]$ , the model maintains a precise, linearly separable manifold in the high-dimensional latent space.

Takeaway. These findings provide structural confirmation for the rejection of Hypothesis B. The failure to articulate the number “75” cannot be attributed to a loss of information or reasoning capacity, as the exact integer value is perfectly encoded in the activation space of the very last layer before the final vocabulary projection.

# D.2. Experiment Settings

We introduce details about the synthetic VLM training and the prompt templates below.

# D.2.1. SYNTHETIC VLM TRAINING

To ensure reproducibility, we provide the specific architectural configurations and training hyperparameters used for synthetic experiments. The setup applies to both the generative counting task and the comparative (True/False) reasoning task.

Model Architecture. We utilize a lightweight, custom Vision-Language Model designed to isolate mechanistic behaviors.

• Vision Encoder: A 2-layer Transformer with a patch size of 14 × 14.   
• Language Decoder: A 2-layer Causal Transformer.   
• Dimensions: Both modules use a hidden embedding dimension of $d _ { \mathrm { m o d e l } } = 3 2$ with H = 2 attention heads.   
• Positional Embeddings (M-RoPE): Following the setting of Qwen3-VL-Instruct (Bai et al., 2025), we employ Multimodal Rotary Positional Embeddings (M-RoPE) (Huang et al., 2025) to encode spatiotemporal information. The rotary dimensions are allocated as sections [4, 6, 6], corresponding to the proportions of the head dimension $( d _ { \mathrm { h e a d } } = 1 6 )$ assigned to time (t), height (h), and width (w) respectively.

Tokenization. We employ a custom tokenizer to handle numerical reasoning explicitly:

• Whitespace Separation: Input text is tokenized based on whitespace.   
• Digit Splitting: Crucially, we enable individual digits=True, ensuring that multi-digit numbers are tokenized as sequences of discrete digits $( \mathrm { e . g . , ^ { * * } l 0 0 ^ { * } }  [ ^ { * * } 1 ^ { * } , { ^ * 0 ^ { * } } , { ^ * 0 ^ { * } } ] )$ . This prevents the model from treating large numbers as unique atomic vocabulary items, forcing it to learn place-value counting rules.

Training Protocol. We adopt a two-stage curriculum with a total budget of 20 epochs per stage.

• Data Scale: For both the text-only pretraining and the visual counting training phase, we generate 8,192 training examples for each target number class.   
• Optimization: We train using a batch size of 2,048 with an initial learning rate of $l r = 1 e ^ { - 3 }$ . For the finetuning, the vision encoder has $l r = 1 e ^ { - 3 }$ and language decoder has $l r = 1 e ^ { - 5 }$ .

• Curriculum:

– Phase 1 (Text Priors): Pretraining on text-only numerical sequences to establish counting algorithms.   
– Phase 2 (Visual Alignment): Training on visual counting data to align visual features with the pretrained numerical representations.

# D.2.2. PROMPT

We provide prompt template used in both synthetic and real VLM below.

# Prompt Template for Text Counting with Synthetic VLM

# Input String:

[a a b c a c b c c b b b c b a a b c b b b b b a b a a c b b a b b c b b a b b b a a c b b c c c b b b a b c b c b c a c c a b c c c b a a b b b a c b c b b b b a b c a c a b a b c b a b c c b b a c b a c b a b a b c c b a c a b b b a a c b c b c b b b c c b a a b b c c c b c b a a a c a c b a a b b a b c b a b a b a b a c b b b a c c c c a c b b b c c a a b b a a b a c b c a a c c b b a b b c a c c b b b b c b c b a a c b a c c b b b b c b b a c b a a a a b a b a c a c b b a a a b a b a a a b a b a a a a a c c a b a a b a b b c b b b a b b a a c c b c a a a b c b a a a c a b a a a a b a b a b b a c c c a b c c a a c b b c b a b a a c b c a a b a a c b c b a c c c c b a b a c b c b c b b a c a b a b a b c b a b b]

Question: How many c letters are there in the given string ?

Answer: 96

Figure 13. Prompt template for text counting with synthetic VLM.

# Prompt Template for Vision Counting with Synthetic VLM

Input:

![](images/b6b813625c7d3d27f2a6bb5176cd1513d0288c928d2bdde957b2602758cbe712.jpg)

<details>
<summary>natural_image</summary>

Grid pattern with black and white stones on beige background (no text or symbols)
</details>

Question: How many black stones are there on the chessboard ?

Answer: 41

Figure 14. Prompt template for vision counting with synthetic VLM.

# Prompt Template for Text Counting Comparison with Synthetic VLM

# Input:

[a a b c a c b c c b b b c b a a b c b b b b b a b a a c b b a b b c b b a b b b a a c b b c c c b b b a b c b c b c a c c a b c c c b a a b b b a c b c b b b b a b c a c a b a b c b a b c c b b a c b a c b a b a b c c b a c a b b b a a c b c b c b b b c c b a a b b c c c b c b a a a c a c b a a b b a b c b a b a b a b a c b b b a c c c c a c b b b c c a a b b a a b a c b c a a c c b b a b b c a c c b b b b c b c b a a c b a c c b b b b c b b a c b a a a a b a b a c a c b b a a a b a b a a a b a b a a a a a c c a b a a b a b b c b b b a b b a a c c b c a a a b c b a a a c a b a a a a b a b a b b a c c c a b c c a a c b b c b a b a a c b c a a b a a c b c b a c c c c b a b a c b c b c b b a c a b a b a b c b a b b] [c b a b b b a c c b b b c b b c b c c a a a c c c a a b a a a a a b b a c b b b b b a b b a b b b b b a c a b b a b b c b c c b a b b b c b c c b c b b b b c c b a b c a b a b a b c c a a c b a c a b c a b a c a a a a b c b a a c a a a c a a c c a b c a c a c c a b a a b a a b b b b a a a a b b b b a c a a a a b b a c b b c a c c c b c c a c b b c c b a b c b c c a c b a c b a a b a a b b b c b a a a b b a b b b c c c b c c b b b a b b b a b c b b b a c b c b a b b b b c c a a b b c b a c c a a c b b a a b c a c b b a a b b b c a c b a b a c a b a b a c b b a b a b b b c b c a b a c c b b a b b a c c b c a b b b c b c a c c b c b c b c b a b b a b a a a a c a b c a b c b a a a c b a a b a a c a c b c c a a a b b]

Question: Are the number of c letters in both input strings the same ?

Answer: True

Figure 15. Prompt template for text counting comparison with synthetic VLM.

# Prompt Template for Vision Counting Comparison with Synthetic VLM

Input:

![](images/029ffa5766b5d2640bee5c9110f3ffde790d929b900cd271cd3cd5e9ce2383aa.jpg)

<details>
<summary>natural_image</summary>

Grid pattern with black and white stones (no text or symbols)
</details>

[c b a b b b a c c b b b c b b c b c c a a a c c c a a b a a a a a b b a c b b b b b a b b a b b b b b a c a b b a b b c b c c b a b b b c b c c b c b b b b c c b a b c a b a b a b c c a a c b a c a b c a b a c a a a a b c b a a c a a a c a a c c a b c a c a c c a b a a b a a b b b b a a a a b b b b a c a a a a b b a c b b c a c c c b c c a c b b c c b a b c b c c a c b a c b a a b a a b b b c b a a a b b a b b b c c c b c c b b b a b b b a b c b b b a c b c b a b b b b c c a a b b c b a c c a a c b b a a b c a c b b a a b b b c a c b a b a c a b a b a c b b a b a b b b c b c a b a c c b b a b b a c c b c a b b b c b c a c c b c b c b c b a b b a b a a a a c a b c a b c b a a a c b a a b a a c a c b c c a a a b b]

Question: Is the number of c letters in the given string the same as the number of black stones on the chessboard ?

Answer: False

Figure 16. Prompt template for vision counting comparison with synthetic VLM.

# Prompt Template for Text Counting with Real VLM

System Prompt: You are a precise counting engine. 1. Verification: Before answering, output a compact data check (e.g., list of indices for text, or row-by-row counts for images). Avoid conversational filler. 2. Format: Verification: <Dense Data> → FINAL ANSWER: <Result>.

# Input:

[‘c’, ‘b’, ‘a’, ‘a’, ‘b’, ‘a’, ‘a’, ‘a’, ‘b’, ‘c’, ‘c’, ‘c’, ‘c’, ‘a’, ‘c’, ‘c’, ‘c’, ‘b’, ‘b’, ‘a’, ‘c’, ‘c’, ‘a’, ‘a’, ‘c’, ‘a’, ‘a’, ‘a’, ‘c’, ‘c’, ‘a’, ‘b’, ‘c’, ‘a’, ‘c’, ‘b’]

Question: String length: 36. Count the occurrences of letter ’c’.

Answer: 15

Figure 17. Prompt template for text counting with real VLM.

# Prompt Template for Vision Counting with Real VLM

System Prompt: You are a precise counting engine. 1. Verification: Before answering, output a compact data check (e.g., list of indices for text, or row-by-row counts for images). Avoid conversational filler. 2. Format: Verification: <Dense Data> → FINAL ANSWER: <Result>.

Input:

![](images/359523f41870d4c496cdef2f6439f9283dbfec76e6becde05da4a28bba1f0473.jpg)

Question: Analyze the 6x6 Go board. Count the black stones.

Answer: 10

Figure 18. Prompt template for vision counting with real VLM.