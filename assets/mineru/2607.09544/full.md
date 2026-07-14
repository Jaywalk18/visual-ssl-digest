# The Count Is There, but Misaligned: Understanding and Correcting Counting Failures in VLMs

Ahmed Oumar El-Shangiti $^{1,2}$ Abzal Nurgazy $^{1}$ Hilal AlQuabeh $^{1}$ Nikolai Rozanov $^{3}$ Kentaro Inui $^{1}$

$^{1}$ MBZUAI $^{2}$ DataBayt.AI Labs $^{3}$ Imperial College London Correspondence: ahmed.oumar@mbzuai.ac.ae

## Abstract

Despite strong performance on many multimodal tasks, vision-language models (VLMs) still struggle with basic object counting. We investigate whether this reflects missing internal knowledge or a gap between internal representations and verbalized outputs. Training simple probes on activations from four VLMs across five counting datasets reveals that nonlinear probes can reliably detect counting errors, suggesting that VLMs often encode the correct count even when they output the wrong answer. SVCCA analysis shows that probes trained on ground-truth counts and probes trained on model outputs occupy a partially shared activation subspace but read out along misaligned directions. We further validate our findings using a causal steering intervention, proving that strengthening the direction of count-identified probes does improve model counting performance. Motivated by this result, we propose a detector-guided self-correction method that selectively re-prompts the model only when an internal error detector predicts failure. This simple inference-time intervention improves counting accuracy by up to 15.6% absolute percentage points, without any parameter updates. Our results establish activation-based error probing as both a practical tool for improving VLM counting and a mechanistic lens on the gap between internal knowledge and model outputs.

## 1 Introduction

Vision-language models (VLMs) have achieved strong performance across a wide range of tasks, including image captioning (Li et al., 2023a), visual question answering (Liu et al., 2023a), reasoning (Lu et al., 2024), and web navigation (Koh et al., 2024). However, strong performance on these tasks does not imply reliable quantitative perception. Among such capabilities, counting is especially important because it requires a model to detect relevant objects, distinguish them from distractors, maintain consistent correspondences, and map visual evidence to an exact numerical answer. This makes counting a useful test of whether a VLM truly grounds its predictions in the image rather than relying on superficial correlations or language priors (Vo et al., 2025).

At the same time, counting exposes several core failure modes in VLMs, including hallucination, failures under clutter or occlusion, poor object individuation, and mistakes in translating perceptual representations into language. Recent work has shown that these weaknesses remain substantial even in strong contemporary models (Weng et al., 2025; Vo et al., 2025; Paiss et al., 2023a; Fu et al., 2023).

Most existing work documents that VLMs are weak at counting, but it does so primarily at the behavioral level, through benchmarks and aggregate performance comparisons (Weng et al., 2025; Vo et al., 2025; Paiss et al., 2023a; Fu et al., 2023). These studies are important because they establish counting as a persistent failure mode, yet they largely leave open a more fundamental question: why do VLMs fail at counting? In particular, behavioral evaluations can show when a model is wrong, but they do not reveal whether the correct count is absent from the model's internal representations, whether it is present but poorly aligned with the model's eventual answer, or whether the failure emerges later during decoding.

A smaller body of work begins to examine counting more mechanistically (Hasani et al., 2025; Alghisi et al., 2025), and some methods attempt to improve performance directly. For example, Alghisi et al. (2025) retrain parts of the model, while Sengupta et al. (2025) use attention-based interventions and obtain modest gains. However these approaches primarily target better final behavior or what layers/tokens are involved in the counting process, rather than explaining how count information is internally represented, or why it sometimes fails to appear in the final prediction.

Our work addresses this gap by analyzing counting through multiple probes applied to intermediate VLM representations, with each probe trained to capture a different part of the model's computation. Rather than asking only whether a model produces the correct count, we ask what information about the ground-truth count, the model's predicted count, and the likelihood of error is present during the forward pass, and how these signals relate to one another across depth. This makes it possible to move beyond behavioral failure and study the internal structure of counting errors.

Specifically, we extend (Sun et al., 2025) to VLM counting and train separate probes on the same intermediate representations using different supervision: one to predict the ground-truth count, one to predict the model's own output count, and one to predict whether the model's answer is wrong. This multi-probe setup allows us to move beyond simple decodability and study how different counting-relevant signals are organized internally. Our results suggest that counting errors arise from at least two distinct sources. First, the model's final generated answer does not always faithfully reflect information already present in intermediate representations. Second, even when intermediate representations support the model's own eventual answer, they can remain more strongly aligned with the correct count, revealing a mismatch between internal quantity information and final prediction. To make this comparison explicit, we analyze the alignment of probe subspaces using an adapted version of SVCCA, which lets us test whether ground-truth-, output-related supervision gives rise to shared or distinct counting directions. We also perform causal steering intervention and prove that strengthening the count direction does improve model performance. Our contributions are as follows. (1) We construct four synthetic counting datasets paired with a real-world benchmark, enabling controlled evaluation across diverse counting regimes (Fig 1). (2) Building on Sun et al. (2025)'s multi-probe framework for LLM arithmetic, we extend the paradigm to VLM counting, probing three supervision targets across both vision-encoder and text-decoder representations (Figs 2;2c). (3) We show via SVCCA that ground-truth and output signals occupy substantially misaligned readout subspaces, especially for nonlinear probes (Fig 3), and confirm causal relevance via activation steering: strengthening the count direction improves accuracy while random directions degrade it. (4) We translate these insights into a detector-guided self-correction method that improves counting accuracy by up to 15.6 absolute percentage points without fine-tuning (Table 1).

## 2 Related Work

VLM counting and reliability. Counting has long been used as a stress test for visual reasoning, from VQA-style benchmarks such as TallyQA (Acharya et al., 2019) and everyday-scene counting (Chattopadhyay et al., 2017; Weng et al., 2025) to recent CLIP-era analysis (Paiss et al., 2023b). Broader multimodal evaluations report systematic weaknesses in grounding and object fidelity, including poor counting performance (Guo et al., 2025; Vo et al., 2025), hallucination (Li et al., 2023c; Fu et al., 2023), and weak perception-reasoning alignment (Liu et al., 2023b; Yu et al., 2023; Yue et al., 2023; Vo et al., 2025). Our work is complementary: rather than proposing a new benchmark, we test whether hidden activations already contain recoverable signals that can identify and correct counting failures.

Probing and Inference-time correction. Probing methods are widely used to test which properties are encoded in activations (Alain and Bengio, 2017; Belinkov, 2022). In NLP, probes have been used for syntax (Hewitt and Manning, 2019), pipeline-like linguistic structure (Tenney et al., 2019), and factual associations (Meng et al., 2022). Recent work also studies latent confidence and truth-related structure in activations (Kadavath et al., 2022; Burns et al., 2023), as well as numeracy-specific representations (Heinzerling and Inui, 2024; El-Shangiti et al., 2025). In VLMs, probes were used to both analyze the counting mechanism (Hasani et al., 2025) and localize bottleneck components that limits counting performance (Alghisi et al., 2025). Inference-time intervention has shown that internal signals can improve factuality without parameter updates (Li et al., 2023b). In parallel, test-time reasoning and correction methods such as self-consistency (Wang et al., 2023), Tree-of-Thought (Yao et al., 2023a), ReAct (Yao et al., 2023b), Self-Refine (Madaan et al., 2023), Reflexion (Shinn et al., 2023), and StateAct (Rozanov and Rei, 2025) improve outputs through search or iterative feedback. Closest to our work is (Sun et al., 2025), which probes LLM hidden states for ground-truth, output, and binary-correctness signals during arithmetic reasoning and uses the resulting error detectors for selective reprompting.

![](images/1848a80230a5a4450e0ee47527e38a6fd9b3d657a071d3f4c4a08e258a9144b3.jpg)  
Figure 1: Overview of the activation-based probing, intervention, and self-correction framework. The pipeline extracts VLM hidden states to train probes, analyze representational subspaces via SVCCA, causally steer activations along probe-derived count directions, and selectively trigger inference-time correction using an internal error detector.

## 3 Analysis: Can VLMs internally count objects?

Counting is an important instance of reasoning; it has been proven to be challenging for VLMs (Guo et al., 2025). In this paper, we investigate the counting mechanism in VLMs. We formalize the counting task as an input image I and a textual query q, the task requires the model M to produce a count estimate $\hat{y}$ such that $\hat{y} = \mathcal{M}(I, q)$ , where y denotes the ground-truth count and $y, \hat{y} \in \{1, 2, \ldots, 9\}$ .

## 3.1 Probe Architectures

Let $x_{l} \in \mathbb{R}^{d}$ denote the final text-token representation at layer $l$ , where $d$ is the hidden dimension of the VLM. A probe is a mapping from $x_{l}$ to a target label. We consider three probing tasks. First, a ground-truth count probe maps $x_{l}$ to the true count $y$ . Second, an output-count probe maps the same representation $x_{l}$ to the model's own predicted count $\hat{y}$ . These two probes have the same architecture but are trained with different supervision (we illustrate below what architectures investigated), allowing us to compare what aspects of the hidden state align with the correct answer versus the produced answer. Third, for error detection, we train a separate error probe to predict whether the model's answer is correct: $e = \mathbb{I}[\hat{y} \neq y] \in \{0, 1\}$ , where e = 1 indicates an incorrect counting prediction. Thus, the count probes solve multi-class classification over count labels, while the error probe solves binary classification over correctness labels.

We employ four probe types of increasing expressivity (Sun et al., 2025). The circular probe projects $x_{l}$ onto a learned 2D plane via $w_{1}, w_{2} \in R^{d}$ , encoding the digit in the angle: $\hat{y} = \frac{10}{2\pi}$ atan2( $w_{1}^{\top}x_{l}, w_{2}^{\top}x_{l}$ ). The linear probe applies an affine map $\hat{y} = w^{\top}x_{l} + b$ , trained with $\ell_{2}$ regularization and rounded at inference. The logistic regression probe assigns per-class weights: $\hat{y} = \arg\max_{i}(w_{i}^{\top}x_{l})$ . The MLP probe adds a single hidden layer (ReLU, 512 units): $\hat{y} = \arg\max_{i}(W_{2}^{\top}\text{ReLU}(W_{1}^{\top}x_{l} + b_{1}) + b_{2})$ . All classification probes are trained with cross-entropy; the circular probe with smooth $\ell_{1}$ loss.

## 3.2 Measuring Representational Alignment via SVCCA.

Probe accuracy tells us whether a representation contains information about a target. However, accuracy alone does not tell us whether two targets are decoded from the same internal directions. This distinction is central to our question. A layer may simultaneously contain information about the true count $y$ and the model output $\hat{y}$ , yet the two may rely on different subspaces. In that case, the model may internally encode the correct quantity while organizing its final prediction around a different representational direction. Conversely, if probes trained on $y$ and $\hat{y}$ rely on highly similar subspaces, then the model's final answer is more closely aligned with the internal count information available at that layer.

To test this, we compare pairs of probes with the same architecture trained on the same hidden states but with different supervision: one probe predicts the ground-truth count y, and the other predicts the model output $\hat{y}$ . We then ask whether the two probes read out their information from similar directions (subspaces) in representation space.

We quantify subspace alignment using Singular Vector Canonical Correlation Analysis (SVCCA; (Raghu et al., 2017)); full mathematical details are given in Appendix A.1. Given the weight matrices $W_{\mathrm{gt}}, W_{\mathrm{out}} \in \mathbb{R}^{c \times d}$ of two probes trained on ground-truth and model-output counts respectively, we extract the top- $k$ left singular vectors from each, yielding orthonormal bases $U_{\mathrm{gt}}^{(k)}, U_{\mathrm{out}}^{(k)} \in \mathbb{R}^{d \times k}$ . We then compute the canonical correlations $\rho_1, \ldots, \rho_k = \sigma_1, \ldots, \sigma_k \left( U_{\mathrm{gt}}^{(k)^\top} U_{\mathrm{out}}^{(k)} \right)$ and summarize alignment as: SVCCA( $W_{\mathrm{gt}}, W_{\mathrm{out}}$ ) = $\frac{1}{k} \sum_{i=1}^{k} \rho_i$ .

This score has a simple interpretation. Values near 1 indicate that the ground-truth and output probes rely on nearly the same directions in the hidden representation, suggesting that the model's final count is aligned with the internally available count information at that layer. Values near 0 indicate that the two probes depend on nearly orthogonal subspaces, suggesting a stronger mismatch between information aligned with the correct count and information aligned with the produced answer.

Our use of SVCCA is motivated by the need for a basis-invariant comparison. Directly comparing probe weight matrices is not reliable, because two probes can implement similar subspaces using different but equivalent bases for the same subspace. By comparing the subspaces spanned by the principal singular vectors instead of raw parameters, SVCCA provides a more robust measure of whether ground-truth-aligned and output-aligned counting information are geometrically similar inside the model. While SVCCA was originally introduced to compare activation matrices across layers or networks (Raghu et al., 2017), applying it directly to activations is uninformative in our setting: both probes read from identical hidden states, so activation-level comparison is trivially maximal. The supervision-specific information instead resides in the probes' readout matrices, whose rows live in the same d-dimensional coordinate system of the shared frozen representation and are therefore directly comparable. To our knowledge, we are the first to apply SVCCA to probe weight matrices as a tool for comparing supervision-specific readout subspaces.

This analysis gives us a mechanistic tool for locating where counting failures emerge. If a layer shows strong decodability of the true count but weak alignment with the output probe, this suggests that correct quantity information is present but not faithfully carried into the final answer. In contrast, if the ground-truth count is weakly decodable, the counting failure likely originates earlier in the visual or multimodal representation itself.

## 3.3 Experimental Setup

Datasets. We carefully design and construct four synthetic datasets, each containing 1170 images (512 × 512, white background) with $\{1,\ldots,9\}$ non-overlapping objects placed on a shuffled grid at a fixed size. The four datasets vary the visual diversity of objects along color and shape (e.g., same color/different shape vs. different colors/different shapes, details in A.2 Table 2). Samples are equally distributed across nine count classes with an 80/20 stratified train/test split. We additionally evaluate on CountBench (Paiss et al., 2023a), removing images with more than nine objects and using a 50/50 stratified split to match our experimental setup.

Models. We evaluate four publicly available VLMs: InternVL2-1B, InternVL2-4B (Chen et al., 2024), Qwen3-VL-2B-Instruct, and Qwen3-VL-8B-Instruct (Bai et al., 2025). All models are run with greedy decoding. Models are from HuggingFace Transformers (Wolf et al., 2020). All models are evaluated under zero-shot settings. All probes are trained for 2000 epochs using a weighted cross-entropy loss to mitigate class imbalance. The layer yielding the best performance will be used in the

![](images/d3ded40354cccf6405cebfec82f443515e96e9781dabd9a104d6741b2162d366.jpg)  
(a) Layer-wise probing F1 score across model depths (ground truth objective).

![](images/c3f3647e32b7a29ccc0b1483187910a791e0e30e76795c42d8b8af92fb5577c6.jpg)

(b) Layer-wise probing F1 score across model depths (model output objective).  
![](images/ec2bba1517546b0531347c1f808969fa6a889c47e83cdeba90be618fae357be6.jpg)  
(c) Error detector performance (F1-score) per layer.

Figure 2: Layer-wise probing and detector performance across model depth. Top(a): ground-truth probe F1 by layer. (b): output-supervised probe F1 by layer. Curves are aggregated across models, datasets, and the three random seeds. Bottom (c): Error detection

subsequent experiments of the corresponding probe type.

Training Probes. For each (model, dataset, probe-type, task) combination, we train four probe architectures, sweep over all layers, and characterize each probe by the layer that achieves the highest held-out F1 score. We train each probe on three tasks: ground-truth count (gt\_probe), output-count (output\_probe), and error probe.

This process is repeated for all probes, datasets, layers, all four models, and averaged over three random seeds.

## 3.4 Probing Results

Figure 2 presents the results of training four probes on the activations of four models for two different objectives (ground truth and model output), aggregated across the four synthetic datasets, and three random seeds. Per-dataset results are provided in figs. 6 to 9 in the Appendix A.

The single scalar affine probe is insufficient to decode count reliably, especially in later layers, while multiclass linear probes remain effective. All other architectures exceed 80% F1, reaching near-perfect scores on Qwen3-VL-8B activations for both objectives. The fact that simpler probes such as Logistic and Circular achieve comparable performance to the MLP suggests that the recovered count information reflects genuine structure in the representations rather than an artifact of an expressive classifier learning the task independently (Hewitt and Liang, 2019). A notable temporal asymmetry emerges: ground-truth probes plateau at earlier layers, while output probes peak near the final layers, suggesting the model encodes the correct count well before it commits to a verbalized answer.

The exception is InternVL2-1B, where output probes remain flat around 60 – 65%, yet error detectors on the same activations achieve >90% F1 (Figure 2c). This indicates that predicting whether the model errs can be easier than predicting what it will say; the binary error signal may occupy a more accessible subspace than the full output distribution.

For error detection, we train four detector types: MLP, Logistic-Separately, Circular-Separately, and Circular-Jointly (Sun et al., 2025). The “separately” variants flag an error when two same-architecture probes (one trained on y, one on $\hat{y}$ ) disagree; the joint variant trains a single model on the angular distance between two circular probes. Detectors on InternVL2 activations yield near-perfect F1, while those on Qwen3-VL activations perform lower, consistent with fewer training errors being available. As we show in §5.1 and Figure 5, detector F1 is strongly correlated with downstream correction gain. Similar training and analysis have been done at the level of the last token and average tokens extracted from the vision encoder, more details in appendix C.

## 3.5 SVCCA: Subspace Divergence Between Ground-Truth and Output Probes.

SVCCA measures similarity between learned representations. In Figure 3, we report SVCCA scores computed between the weights (Figure 14a for activations) of the same probe family trained under two objectives: ground-truth count prediction and model-output count prediction. In Figure 3(a), nonlinear probes (Circular and MLP) lie in a near-orthogonal regime (low SVCCA). Linear probes show higher alignment than nonlinear probes, but alignment remains limited, especially for Logistic regression. These findings support a subspace-misalignment hypothesis: count-relevant information is internally encoded, but only weakly coupled to the representation subspace that drives final verbalized outputs. To test whether this pattern is merely a layer-selection artifact, Figure 3(b) plots

![](images/cea2ee69a34a9671d5619249c3b9f1e3195d9816290e786a530e3ad640d24f1c.jpg)  
(a) Mean weight-SVCCA by probe family.

![](images/0d79ed829505d0b57e74b81194621bfb1e61a34c26320785ef9f3a00024aa80c.jpg)  
(b) Weight-SVCCA versus best-layer gap.  
Figure 3: Weight-space SVCCA analysis of gt\_probe versus output\_probe. (Left) Mean SVCCA by probe family across the full 4-model, 4-dataset, 3-seed sweep. MLP and Circular probes lie in the near-orthogonal regime, while Linear probes show substantially higher alignment. (Right) SVCCA versus best-layer gap $|l_{gt} - l_{out}|$ . Larger layer disagreement tends to coincide with lower alignment, but low SVCCA persists even at small layer gaps, indicating that subspace divergence is not reducible to depth mismatch alone.

SVCCA against the best-layer gap. Low SVCCA persists even at small layer gaps, indicating that subspace divergence is not reducible to depth mismatch alone, as discussed in prior findings (Burns et al., 2023; Gekhman et al., 2025)

## 4 Causal Steering Intervention

The probing and SVCCA analyses above establish that VLMs internally encode count information that diverges from their verbalized outputs. These results are correlational: the probe may recover a direction that is predictive of count without being causally involved in generating it. To distinguish these possibilities, we perform a direct activation-steering experiment on the full synthetic test split of all four synthetic datasets.

Method. For each sample we first run the unmodified model to obtain a baseline prediction $\hat{y}$ and record the baseline accuracy (Figure 4). We then rerun the same sample with a single forward-hook intervention applied at the best layer $l^{*}$ of the saved Logistic gt\_probe. Let $x_{l^{*}} \in R^{d}$ denote the final prompt-token hidden state at layer $l^{*}$ , and let $w_{c}$ be the probe weight vector for class c. We define the steering direction as the normalized pairwise difference between the ground-truth and baseline class directions:

$$
d = \frac {w _ {y} - w _ {\hat {y}}}{\| w _ {y} - w _ {\hat {y}} \| _ {2}}, \quad \tilde {x} _ {l ^ {*}} = x _ {l ^ {*}} + \alpha \cdot \mathrm{RMS} (x _ {l ^ {*}}) \cdot d,
$$

$$
\mathrm{RMS} (x) = \frac {\| x \| _ {2}}{\sqrt {d}},\tag{1}
$$

where $\alpha$ controls steering strength and RMS-scaling ensures the perturbation is proportional to the local activation norm. As a control, we run the same experiment replacing d with a randomly sampled unit vector of the same dimensionality. Each $\alpha$ is swept over a range from 5 to 65.

Results. Figure 4 plots model accuracy as a function of $\alpha$ for both the probe-derived (blue) and random-direction (red) interventions. Three observations emerge: (1) Probe-derived steering causally improves counting. On all four datasets, steering along the probe direction lifts accuracy above the unsteered baseline within a broad effective range (green region). The clearest gain appears on diff\_col\_diff\_shape, where accuracy rises from a baseline of $\sim27\%$ to a peak near 46% at moderate $\alpha$ , corresponding to an absolute gain of roughly 19 percentage points. On sing\_col\_sing\_shape and diff\_col\_sing\_shape, more modest but consistent gains ( $\sim6-8$ pp) persist across a range of moderate intervention strengths. These results demonstrate that the direction recovered by the probe is not merely predictive of count but plays a causal role in the model's generation. (2) Random steering monotonically degrades performance. Across all four datasets, the random-direction control either stays near the baseline at small $\alpha$ or steadily degrades as $\alpha$ grows (red curves entering the pink region), confirming that the improvements in (1) are specific to the probe-derived direction rather than an artifact of generic activation perturbation. The widening gap between the blue and red curves as $\alpha$ increases further corroborates the directional specificity of the causal effect. (3) Excessive steering is destructive. On all datasets, there exists a critical $\alpha$ beyond which even probe-derived steering begins to hurt. On sing\_col\_diff\_shape, the blue curve enters the destructive zone around $\alpha = 45$ , and on diff\_col\_sing\_shape, gains plateau early. This non-monotonicity is consistent with a local-linear interpretation: the probe direction is a good first-order approximation of the count subspace, but large perturbations push the hidden state beyond the regime where this linear approximation holds.

![](images/083aaa5b631bd8765bc57dea2c98a3f045d64970552b4e660278f31d3f8da02f.jpg)  
Figure 4: Causal steering results for InternVL2-1B on four synthetic counting datasets. Green shading marks the effective steering zone where probe-derived steering exceeds the baseline; pink shading marks the destructive zone where steering falls below the baseline.

Implications. The causal steering experiment indicates that probe-derived count directions can influence generation, but it also reveals that direct activation intervention depends on two factors: dataset geometry and steering magnitude $\alpha$ . This motivates a less invasive use of the same representational insight: rather than modifying hidden states directly, we use the activation-level error signal to decide when the model should reconsider its answer. Figure 5 supports this choice by showing a strong positive correlation between detector F1 and correction gain, indicating that reliable internal failure prediction translates into effective downstream correction. We therefore turn to a detector-guided self-correction method that treats the probe as an intervention gate rather than as a steering mechanism. Next, we introduce detector-guided self-correction in Section 5

## 5 Detector-Guided Self-Correction Method

Let $x_{l^{*}} \in \mathbb{R}^{d}$ be the activation extracted from last non-padding token from layer $l^{*}$ of VLM $\mathcal{M}$ for image-query pair $(I, q)$ , where $l^{*} = \arg \max_{l} \mathrm{F1}(D^{(l)}, \mathcal{D}_{\mathrm{val}})$ is the layer yielding the best detector performance on a held-out validation set $\mathcal{D}_{\mathrm{val}}$ . Let $\hat{y}^{(1)} = \mathcal{M}(I, q)$ be the first-pass prediction. A detector $D^{(l^{*})} : \mathbb{R}^{d} \to [0, 1]$ estimates the error probability:

$$
s = D ^ {(l ^ {*})} (x _ {l ^ {*}}) \approx \mathbb {P} \Big (\hat {y} ^ {(1)} \neq y \mid x _ {l ^ {*}} \Big)\tag{2}
$$

and triggers a correction when s exceeds a threshold $\tau^{1}$ :

$$
\hat {y} = \left\{ \begin{array}{l l} \mathcal {M} \big (I, f _ {\text { corr }} (q, \hat {y} ^ {(1)}) \big); & \text { if   } s \geq \tau \\ \hat {y} ^ {(1)}; & \text { otherwise } \end{array} \right.\tag{3}
$$

where $f_{\mathrm{corr}}(q,\hat{y}^{(1)})$ is a correction prompt that supplies the original response and asks the model to reconsider its answer. The correction prompt is in the Appendix B.

Baselines. We compare against two baselines. Always Reprompt applies $f_{corr}$ to all samples unconditionally. This is the maximum budget baseline. Random-K matches the detector's intervention budget $K = |\{i : D(x_{l^{*},i}) \geq \tau\}|$ but selects samples at random, isolating the contributions of learned selectivity over random reprompting. An additional baseline (Entropy-guided) is reported in Table B.2.

## 5.1 Proposed Method Results

Table 1 reports the performance of the zero-shot raw model, the Always Reprompt baseline, the Random-K baseline, and our method across four synthetic datasets and four VLMs. In the Always Reprompt setting, every sample is reprompted unconditionally (i.e., 100% of the samples), whereas Random-K reprompts the same number K of samples as our probe flags, but selects them uniformly at random.

We observe that, for almost all models, the dataset with a single color and varying shapes is the most difficult. The only exception is the InternVL2-4B model.

<table><tr><td>Dataset</td><td>Model</td><td>Raw</td><td>Always Repr.</td><td>Random-K</td><td>Ours (Δ)</td></tr><tr><td rowspan="4">DC-DS</td><td>IVL2-1B</td><td>26.50</td><td>23.50</td><td>23.97</td><td>31.80 (+5.31)</td></tr><tr><td>IVL2-4B</td><td>34.62</td><td>50.85</td><td>45.73</td><td>50.21 (+15.60)</td></tr><tr><td>Q3VL-2B</td><td>71.51</td><td>71.51</td><td>71.69</td><td>72.04 (+0.53)</td></tr><tr><td>Q3VL-8B</td><td>79.34</td><td>84.33</td><td>80.31</td><td>83.76 (+4.42)</td></tr><tr><td rowspan="4">DC-SS</td><td>IVL2-1B</td><td>29.77</td><td>28.21</td><td>28.45</td><td>35.43 (+5.66)</td></tr><tr><td>IVL2-4B</td><td>39.89</td><td>44.16</td><td>43.02</td><td>43.95 (+4.06)</td></tr><tr><td>Q3VL-2B</td><td>71.94</td><td>72.93</td><td>72.65</td><td>72.83 (+0.89)</td></tr><tr><td>Q3VL-8B</td><td>92.31</td><td>94.44</td><td>92.59</td><td>94.41 (+2.10)</td></tr><tr><td rowspan="4">SC-DS</td><td>IVL2-1B</td><td>24.22</td><td>24.22</td><td>23.50</td><td>30.88 (+6.66)</td></tr><tr><td>IVL2-4B</td><td>43.02</td><td>47.58</td><td>46.79</td><td>50.61 (+7.59)</td></tr><tr><td>Q3VL-2B</td><td>61.40</td><td>62.11</td><td>61.18</td><td>62.46 (+1.07)</td></tr><tr><td>Q3VL-8B</td><td>66.95</td><td>77.49</td><td>71.94</td><td>77.49 (+10.54)</td></tr><tr><td rowspan="4">SC-SS</td><td>IVL2-1B</td><td>23.65</td><td>30.48</td><td>28.77</td><td>35.11 (+11.47)</td></tr><tr><td>IVL2-4B</td><td>39.74</td><td>44.73</td><td>43.20</td><td>45.37 (+5.63)</td></tr><tr><td>Q3VL-2B</td><td>69.80</td><td>70.94</td><td>70.48</td><td>71.62 (+1.82)</td></tr><tr><td>Q3VL-8B</td><td>90.60</td><td>92.17</td><td>90.92</td><td>92.59 (+1.99)</td></tr><tr><td rowspan="4">CountBench</td><td>IVL2-1B</td><td>40.15</td><td>49.24</td><td>46.78</td><td>46.97 (+6.82)</td></tr><tr><td>IVL2-4B</td><td>51.33</td><td>54.73</td><td>51.66</td><td>55.11 (+3.79)</td></tr><tr><td>Q3VL-2B</td><td>72.92</td><td>78.79</td><td>73.67</td><td>79.31 (+6.39)</td></tr><tr><td>Q3VL-8B</td><td>79.92</td><td>89.02</td><td>81.11</td><td>88.59 (+8.66)</td></tr></table>

Table 1: Mean counting accuracy (%) aggregated over 3 seeds and 4 detectors. Always Repr. is naive always-re-prompt; it does apply correction (reprompting) to 100% of the evaluated samples. Random-K is budget-matched random intervention; Ours is detector-guided correction. $\Delta$ reports only Ours – Before (pp). Values in red indicate performance degradation compared to Raw.

While Always Reprompt outperforms Ours in several configurations, it is not a practical strategy for two reasons. First, it is computationally expensive, as it unconditionally requires a second forward pass on the entire dataset. Second, indiscriminate reprompting is unreliable: it can flip previously correct answers, degrading overall performance (Table 1). In contrast, Ours recovers most of the gains achieved by Always Reprompt while selectively reprompting only a small subset of samples, avoiding both the computational overhead and the risk of degradation. It consistently surpasses the Random-K baseline. Overall, our approach yields improvements of up to 15.6% (avg 5.3%). The largest gains occur for the InternVL2 family, while the smallest gains are observed for Qwen3-VL-2B; these trends align closely with the quality of the error-detection probe (Figure 2c). Figure 5 further illustrates how probe performance on error detection correlates with self-correction gains, showing that the largest improvements are obtained when the probe's F1-score exceeds 90% (with strong correlation $\rho = 0.803$ ). Table 3 provides a summary of correction gains across detector types. We can see several patterns. First, no single detector consistently dominates, suggesting that detector choice is not critical and the method is robust to error detector architecture. Second, the discrepancy between VLM families is larger than it between detectors: InternVL2 benefits substantially (up to +15.95 pp) while Qwen3-VL-2B gains is (+0.43 to +2.42 pp) independent of probe choice, suggesting that the bottleneck for correction is the quality of the error signal in the activation rather than the detector architecture itself. Table A. 4 reports true-positive correction rates and false positive preservation rates for each detector. Additional prompts are reported in B.1. By repeating the full pipeline on the CountBench dataset, we prove that our findings hold beyond synthetic datasets.

## 6 Conclusion

In this paper, we introduce the first study that examines how simple probes can predict the ground-truth count, the model-predicted count, and the model likelihood of error from VLMs' internal representations. We further demonstrate that these simple probes can accurately anticipate when a VLM is about to generate an incorrect count. A complementary SVCCA analysis shows that probes trained on ground-truth counts and those trained on model outputs occupy misaligned readout subspaces, indicating a divergence between true and output count representations during generation. We further validate our findings using a causal steering intervention, proving that strengthening the count-identified probes' direction does improve model counting performance. Finally, we integrate these insights into a method that uses simple probes for error detection and reprompting-based correction, achieving up to a 15.6% improvement in counting performance.

## 7 Limitations and Future Work

The limitations are that we study only models up to 8B parameters due to computational constraints, use synthetic data whose transfer to real-world settings is still unverified, cover only two VLM families, conduct a causal intervention experiment with only one model, and restrict counting tasks to the 1–9 range. Future work should evaluate larger models, real-world generalization, additional VLM families, and harder counting regimes beyond 10 objects.

## References

Manoj Acharya, Kushal Kafle, and Christopher Kanan. 2019. TallyQA: Answering complex counting ques-

tions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 8076–8084.

Guillaume Alain and Yoshua Bengio. 2017. Understanding intermediate layers using linear classifier probes. In International Conference on Learning Representations (ICLR) Workshop. ArXiv:1610.01644.

Simone Alghisi, Gabriel Roccabruna, Massimo Rizzoli, Seyed Mahed Mousavi, and Giuseppe Riccardi. 2025. [delre]constructing vlms' reasoning in counting. Preprint, arXiv:2510.19555.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-vl technical report. Preprint, arXiv:2511.21631.

Yonatan Belinkov. 2022. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219.

Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. 2023. Discovering latent knowledge in language models without supervision. arXiv preprint arXiv:2212.03827.

Prithvijit Chattopadhyay, Ramakrishna Vedantam, Ramprasaath R. Selvaraju, Dhruv Batra, and Devi Parikh. 2017. Counting everyday objects in everyday scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1135–1144.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. 2024. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198.

Ahmed Oumar El-Shangiti, Tatsuya Hiraoka, Hilal AlQuabeh, Benjamin Heinzerling, and Kentaro Inui. 2025. The geometry of numerical reasoning: Language models compare numeric properties in linear subspaces. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 550–561, Albuquerque, New Mexico. Association for Computational Linguistics.

Chaoyou Fu, Yujie Dai, Yinpeng Luo, Liang Li, Shuhuai Ren, Runpeng Zhang, Zihan Wang, Chenyang Zhou, Yadong Shen, Meng Zhang, et al. 2023. MME: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394.

Zorik Gekhman, Eyal Ben David, Hadas Orgad, Eran Ofek, Yonatan Belinkov, Idan Szpektor, Jonathan Herzig, and Roi Reichart. 2025. Inside-out:

Hidden factual knowledge in llms. Preprint, arXiv:2503.15299.

Xuyang Guo, Zekai Huang, Zhenmei Shi, Zhao Song, and Jiahao Zhang. 2025. Your vision-language model can't even count to 20: Exposing the failures of vlms in compositional counting. Preprint, arXiv:2510.04401.

Hosein Hasani, Amirmohammad Izadi, Fatemeh Askari, Mobin Bagherian, Sadegh Mohammadian, Mohammad Izadi, and Mahdieh Soleymani Baghshah. 2025. Understanding counting mechanisms in large language and vision-language models. Preprint, arXiv:2511.17699.

Benjamin Heinzerling and Kentaro Inui. 2024. Monotonic representation of numeric attributes in language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 175–195, Bangkok, Thailand. Association for Computational Linguistics.

John Hewitt and Percy Liang. 2019. Designing and interpreting probes with control tasks. Preprint, arXiv:1909.03368.

John Hewitt and Christopher D. Manning. 2019. A structural probe for finding syntax in word representations. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4129–4138, Minneapolis, Minnesota. Association for Computational Linguistics.

Saurav Kadavath, Rylan Schaeffer, Juhyeon Kwon, Katie Mills, Alyssa Yao, et al. 2022. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. 2024. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. Preprint, arXiv:2401.13649.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. 2023a. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 19730–19742. PMLR.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023b. Inference-time intervention: Eliciting truthful answers from a language model. In Advances in Neural Information Processing Systems, volume 36. Curran Associates, Inc.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, et al. 2023c. Evaluating object hallucination in large vision-language models: The POPE benchmark. arXiv preprint arXiv:2305.10355.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023a. Visual instruction tuning. Preprint, arXiv:2304.08485.

Yang Liu, Wenhai Du, Kai Zhang, Xiaoxin Li, Jinghao Hu, Xiang Liu, Zicheng Zhou, Yuan He, Zejun Qiu, et al. 2023b. MMBench: Is your multimodal model an all-around player? arXiv preprint arXiv:2307.06281.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. 2024. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. Preprint, arXiv:2310.02255.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, volume 36. Curran Associates, Inc.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems, volume 35, pages 17359–17372. Curran Associates, Inc.

Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. 2023a. Teaching clip to count to ten. Preprint, arXiv:2302.12066.

Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. 2023b. Teaching CLIP to count to ten. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3323–3333.

Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. 2017. SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Nikolai Rozanov and Marek Rei. 2025. StateAct: Enhancing LLM base agents via self-prompting and state-tracking. In Proceedings of the 1st Workshop for Research on Agent Language Models (REALM 2025), pages 367–385, Vienna, Austria. Association for Computational Linguistics.

Saurav Sengupta, Nazanin Moradinasab, Jiebei Liu, and Donald E. Brown. 2025. Can vision-language models count? a synthetic benchmark and analysis of attention-based interventions. Preprint, arXiv:2511.17722.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36. Curran Associates, Inc.

Yucheng Sun, Alessandro Stolfo, and Mrinmaya Sachan. 2025. Probing for arithmetic errors in language models. Preprint, arXiv:2507.12379.

Ian Tenney, Dipanjan Das, and Ellie Pavlick. 2019. BERT rediscovers the classical NLP pipeline. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4593–4601, Florence, Italy. Association for Computational Linguistics.

An Vo, Khai-Nguyen Nguyen, Mohammad Reza Taesiri, Vy Tuong Dang, Anh Totti Nguyen, and Daeyoung Kim. 2025. Vision language models are biased. Preprint, arXiv:2505.23941.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations.

Tengjin Weng, Jingyi Wang, Wenhao Jiang, and Zhong Ming. 2025. Visnumbench: Evaluating number sense of multimodal large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 3830–3840.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Karthik Narasimhan, Yuan Cao, et al. 2023a. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023b. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations.

Wenhao Yu, Yiming Chen, Xiang Wu, Zhen He, Yuhang Liu, Xu Zhao, et al. 2023. MM-Vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490.

Xi Yue, Yuan Ni, Kai Zhang, Tong Zheng, Yang Liu, Wenxuan Gao, et al. 2023. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502.

## A Detailed Results

## A.1 SVCCA Details

We quantify this using Singular Vector Canonical Correlation Analysis (SVCCA) (Raghu et al.,

2017). Consider a pair of trained linear classifiers of the same architecture, represented by weight matrices

$$
W _ {\mathrm{gt}} \in \mathbb {R} ^ {c \times d}, \qquad W _ {\mathrm{out}} \in \mathbb {R} ^ {c \times d},\tag{4}
$$

where c is the number of classes and d is the hidden dimension. Each row of a probe weight matrix corresponds to a class-specific readout direction, so the row space of the matrix characterizes the subspace used by that probe for prediction.

To extract a stable low-dimensional basis for each probe, we compute the singular value decomposition

$$
W _ {\mathrm{gt}} ^ {\top} = U _ {\mathrm{gt}} \Sigma_ {\mathrm{gt}} V _ {\mathrm{gt}} ^ {\top}, \qquad W _ {\mathrm{out}} ^ {\top} = U _ {\mathrm{out}} \Sigma_ {\mathrm{out}} V _ {\mathrm{out}} ^ {\top}.\tag{5}
$$

We then retain the top-k left singular vectors from each decomposition, yielding orthonormal basis matrices

$$
U _ {\mathrm{gt}} ^ {(k)}, U _ {\mathrm{out}} ^ {(k)} \in \mathbb {R} ^ {d \times k}.\tag{6}
$$

These bases span the principal readout subspaces used by the ground-truth and output probes, respectively.

Next, we measure how well these two subspaces align by computing the singular values of their cross-projection matrix:

$$
\rho_ {1}, \ldots , \rho_ {k} = \sigma_ {1}, \ldots , \sigma_ {k} \left(U _ {\mathrm{gt}} ^ {(k) ^ {\top}} U _ {\mathrm{out}} ^ {(k)}\right),\tag{7}
$$

where each $\rho_{i} \in [0,1]$ is a canonical correlation. Intuitively, $\rho_{i}$ measures how similar the i-th most aligned directions are across the two probe subspaces. We summarize alignment by the mean canonical correlation:

$$
\mathrm{SVCCA} (W _ {\mathrm{gt}}, W _ {\mathrm{out}}) = \frac {1}{k} \sum_ {i = 1} ^ {k} \rho_ {i}.\tag{8}
$$

## A.2 Synthetic Datasets Details

## B Correction Prompt Template

The following prompt is used during the second pass of the detector-guided self-correction pipeline ( $\S5$ ).

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
$f_{\text{corr}}(q, \hat{y}^{(1)})$

&lt;image&gt;
&lt;q&gt;: How many object(s) are in this image?
Answer:
Your previous answer was: $\hat{y}^{(1)}$
That's definitely wrong. The correct count should be:
</div>

![](images/a6e32a03f9b6b8f668191339f3b7d670f446c51156ac22d25ab0c008b893e6b9.jpg)  
Figure 5: Detector F1 vs. correction gain ( $\Delta$ Acc). Each point is one run (model $\times$ dataset $\times$ detector $\times$ seed). Spearman $\rho = 0.803$ .

The system prompt used for all passes is:

```txt
System Prompt
You are a helpful assistant. Output only the final count in the format <count>.
```

## B.1 Experiments on Additional Prompts

To evaluate the sensitivity of detector-guided self-correction to the specific framing of the intervention prompt, we compare three distinct reprompting strategies. The exact text for each prompt variation is defined as follows:

```txt
stronger
That's definitely wrong. The correct count should be:
strongv2
That's incorrect. Recount every visible item in the image carefully.
counting
I made an error in my previous count—there appears to be a different number of objects in the image. I'll recount carefully, step by step:
```

As shown in Table 5, the counting prompt yields the highest absolute gains on three of the four datasets, peaking at a +16.67 improvement on sing\_col\_diff\_shape. However, stronger remains the most conservative and stable approach. When evaluated by preservation-oriented metrics, stronger maintains the highest average false-positive preservation rates (ranging from 71.95% to 95.21% across the synthetic datasets) and achieves

<table><tr><td>Property</td><td>Values</td></tr><tr><td>Colors</td><td>blue, red, green, yellow, black, orange, purple, cyan, magenta, lime</td></tr><tr><td>Shapes</td><td>circle, square, triangle</td></tr><tr><td>Color mode</td><td>sing_color (uniform per image), diff_color (varied per object)</td></tr><tr><td>Shape mode</td><td>sing_shape (uniform per image), diff_shape (varied per object)</td></tr><tr><td>Object count</td><td>1–9</td></tr><tr><td>Image resolution</td><td>512 × 512</td></tr></table>

Table 2: Properties of the synthetic counting dataset. Pillow python package handles image creation and shape rendering

![](images/071191360ac1d04c2d9233f89b8dcdac6282a6d78208af912f132460e23e0410.jpg)  
(a) Layer-wise probing F1 score across model depths (ground truth objective).

![](images/c7b8be403ad39d4d046a6b5f58e837b0cc18384f40ed0d722f56b0f367ffedcc.jpg)  
(b) Layer-wise probing F1 score across model depths (model output objective).

![](images/086dd2c3580adbd58140b1a4034eff0caadc95d7d1e2703e25783dfa1e2ba773.jpg)  
(c) Error detector performance (F1-score) per layer.

Figure 6: Layer-wise probing and detector performance across model depth. Top: ground-truth probe F1 by layer. Middle: output-supervised probe F1 by layer. Bottom: error-detector F1 by layer. Curves are aggregated across models, and the three random seeds. DC-DS: Different Color Different Shape dataset

the highest combined effectiveness on a majority of splits. This highlights a clear architectural trade-off between aggressive error correction and the preservation of baseline coherence.

## B.2 Comparison of Entropy and Probe-Guided Self-Correction

The results demonstrate that probe-guided self-correction strictly outperforms entropy-based heuristics across all evaluated datasets. Relying purely on output entropy to trigger self-correction is highly unreliable; it actively degrades model performance (yielding net-negative accuracy deltas) on two of

![](images/d4b78152b3113e2fabf14690a40c0f6c84130b2bb3aff58ee95e47c51511d24d.jpg)  
(a) Layer-wise probing F1 score across model depths (ground truth objective).

![](images/4c3ba1d7aa6bac3bae763a69fda52c4e1da80a17b1cf817de79173b078a39043.jpg)  
(b) Layer-wise probing F1 score across model depths (model output objective).

![](images/04f5710245c1df7e6342ff87a4726741eb41349cf4b45a0adca5950e55213ee3.jpg)  
(c) Error detector performance (F1-score) per layer.

Figure 7: Layer-wise probing and detector performance across model depth. Top: ground-truth probe F1 by layer. Middle: output-supervised probe F1 by layer. Bottom: error-detector F1 by layer. Curves are aggregated across models, and the three random seeds. DC-SS: Different Color Single Shape dataset

the four datasets (diff\_col\_diff\_shape and sing\_col\_sing\_shape).

In contrast, the probe-guided approach consistently yields positive gains, achieving a massive +16.24% absolute improvement on the sing\_col\_diff\_shape split. Mechanistically, the probe-guided method succeeds by accurately targeting and correcting flawed reasoning paths that generic, uncertainty-based entropy metrics fail to effectively address.

## C Vision Encoder Probing Results

Testing was conducted across two distinct settings: probes were trained on activations extracted from

![](images/3bf398e4e5be8fdf4628cfa4208f738025f95629d5dbf226c98767037a5d057b.jpg)  
(a) Layer-wise probing F1 score across model depths (ground truth objective).

![](images/7f4118f0ca3e6ee3b036f3479e6d5d8043e08e01eebf9f0fc18135cde01d9bcf.jpg)  
(b) Layer-wise probing F1 score across model depths (model output objective).

![](images/33cd825b375be0341a30a5b027d2642a0bc0347dfac4969686b095d4f4fadac3.jpg)  
(c) Error detector performance (F1-score) per layer.

Figure 8: Layer-wise probing and detector performance across model depth. Top: ground-truth probe F1 by layer. Middle: output-supervised probe F1 by layer. Bottom: error-detector F1 by layer. Curves are aggregated across models, and the three random seeds. SC-DS: Single Color Different Shape dataset.

the last vision token of the VLM vision encoder (results shown in Figures 19, 20, 21 and 22 and on the mean activations across all tokens in each image (results shown in Figures 15, 16, 17 and 18).

The results of the layered analysis of the DC-DS, DC-SS, SC-DS and SC-SS data sets reveal a persistent mechanistic gap between the internal perception of the model and its final verbalized result. Specifically, multilayer perceptrons (MLPs) reach peak performance very early - often by the 5th or 10th layer and maintain this plateau until the final layers. This suggests that the "true" answer is successfully extracted from the visual input and represented within the model early in the forward pass. The model "knows" the answer long before it reaches the final output layers, where the decision to verbalize it is made. It is also worth noting that the ground-truth probes of Qwen-3-VL models begin to fall after the middle layers, which is not observed in the InternVL-2 models. Unlike ground-truth probes, which plateau early in the process, output probes typically reach peak F1 values only in the final layers of the model, indicating that the representational subspace governing actual count generation is fully formed only at the end of the

![](images/94bc38e8f24936d464fbcc76c510b72a7dc80325d3794c6f0259e9c47129c33e.jpg)  
(a) Layer-wise probing F1 score across model depths (ground truth objective).

![](images/91bab966b1f80a503bf66992075f17692eee73227e7a0fd1ea08333ebac33c69.jpg)  
(b) Layer-wise probing F1 score across model depths (model output objective).

![](images/5f2f07b0a86d6eeb53a14b22f4da809d8b1f63cfca85abd2823b424115471c88.jpg)  
(c) Error detector performance (F1-score) per layer.

Figure 9: Layer-wise probing and detector performance across model depth. Top: ground-truth probe F1 by layer. Middle: output-supervised probe F1 by layer. Bottom: error-detector F1 by layer. Curves are aggregated across models, and the three random seeds. SC-SS: Single Color Single Shape dataset.

processing pipeline. This trend persists across varying visual representation complexities, although the performance of these samples is generally lower than that of their ground-truth counterparts.

Furthermore, high F1-score values obtained by error detection algorithms trained on vision encoder activations are likely due to a spurious correlation between scene complexity and the model's error rate, rather than genuine error detection. Because the vision language model (VLM) makes more errors on images with a large number of objects, which also tend to be visually denser and cluttered, the vision encoder activations for "complex" scenes (many objects) and "simple" scenes (few objects) systematically differ. The error detection algorithm exploits this: it learns to predict "error" for activations corresponding to complex scenes and "correct" for simple ones, effectively acting as a scene complexity classifier rather than an error detector. Because complexity and error coexist in the dataset, this leads to inflated F1-score varies that do not reflect the model's true ability to determine whether the model succeeded or failed on any given image.

Comparative analysis shows that probes trained on average activation values consistently outperform probes restricted to the last visual token across all four levels of visual complexity. This performance advantage likely stems from the fact that the critical spatial information required for accurate counting is not concentrated in a single patch or token, but is distributed across the entire image grid. By aggregating features across all patches, averaged representations allow for a more holistic representation of the scene, reducing the information loss inherent in extracting individual tokens and providing a more robust signal for probes to identify internally represented values.

![](images/9e5266d2dabc9a75e1a2f3572bacb9cf74e2887809e30177f7294b71bc910dfe.jpg)  
(a) Layer-wise probing F1 score across model depths (ground truth objective).

![](images/d62ae857c7adbdbd0c964d43cff3ec53b9c1c4e1acc33213a805b121388fd066.jpg)  
(b) Layer-wise probing F1 score across model depths (model output objective).

![](images/ea0825a33bc4b97f76fb004434f49cce963637031f3e25b2225c9d21e22ec0cb.jpg)  
(c) Error detector performance (F1-score) per layer.  
Figure 10: Layer-wise probing and detector performance across model depth. Top: ground-truth probe F1 by layer. Middle: output-supervised probe F1 by layer. Bottom: error-detector F1 by layer. Curves are aggregated across models, and the three random seeds. CountBench dataset

Detector F1 vs Correction Gain  
![](images/fe5cbf140c4cce70f8e54fc0b03089f01654a3ad86f58247a8a3cf86564af200.jpg)  
Figure 11: Detector F1 vs. correction gain ( $\Delta$ Acc). Each point is one run. The dataset is CountBench

<table><tr><td>Dataset</td><td>Model</td><td>MLP</td><td>Log-S</td><td>Circ-J</td><td>Circ-S</td></tr><tr><td rowspan="4">DC-DS</td><td>InternVL2-1B</td><td>+5.27</td><td>+6.41</td><td>+5.13</td><td>+4.42</td></tr><tr><td>InternVL2-4B</td><td>+15.10</td><td>+15.95</td><td>+15.81</td><td>+15.53</td></tr><tr><td>Qwen3-VL-2B</td><td>+0.57</td><td>+0.43</td><td>+0.57</td><td>+0.57</td></tr><tr><td>Qwen3-VL-8B</td><td>+4.13</td><td>+5.27</td><td>+3.85</td><td>+4.42</td></tr><tr><td rowspan="4">DC-SS</td><td>InternVL2-1B</td><td>+5.84</td><td>+5.98</td><td>+4.42</td><td>+6.41</td></tr><tr><td>InternVL2-4B</td><td>+4.13</td><td>+4.42</td><td>+3.70</td><td>+3.99</td></tr><tr><td>Qwen3-VL-2B</td><td>+0.57</td><td>+0.71</td><td>+1.00</td><td>+1.28</td></tr><tr><td>Qwen3-VL-8B</td><td>+1.85</td><td>+2.56</td><td>+1.99</td><td>+1.99</td></tr><tr><td rowspan="4">SC-DS</td><td>InternVL2-1B</td><td>+6.84</td><td>+6.55</td><td>+6.41</td><td>+6.84</td></tr><tr><td>InternVL2-4B</td><td>+7.83</td><td>+7.69</td><td>+7.12</td><td>+7.69</td></tr><tr><td>Qwen3-VL-2B</td><td>+1.14</td><td>+1.28</td><td>+1.14</td><td>+0.71</td></tr><tr><td>Qwen3-VL-8B</td><td>+10.40</td><td>+11.11</td><td>+9.97</td><td>+10.68</td></tr><tr><td rowspan="4">SC-SS</td><td>InternVL2-1B</td><td>+11.40</td><td>+12.25</td><td>+10.97</td><td>+11.25</td></tr><tr><td>InternVL2-4B</td><td>+5.56</td><td>+6.13</td><td>+5.84</td><td>+4.99</td></tr><tr><td>Qwen3-VL-2B</td><td>+1.00</td><td>+2.42</td><td>+2.28</td><td>+1.57</td></tr><tr><td>Qwen3-VL-8B</td><td>+1.71</td><td>+2.28</td><td>+1.99</td><td>+1.99</td></tr></table>

Table 3: Stratified detector-guided gains: mean $\Delta$ Acc (pp) for Ours over Before, computed per model/dataset and averaged over 3 seeds. Detector abbreviations are Log-S (Logistic-Separately), Circ-J (Circular-Jointly), and Circ-S (Circular-Separately). Bold and green mark the best detector(s) per row.

![](images/3bdc5a24784ff510a4e94a492e6637c4589926f5be5a12bd9ce5b14f54f5ddfa.jpg)  
(a) Single color single shape

![](images/f4efedc68290cd8ad979310cbec1415680d7c38af3a18d410aa7881b5af622de.jpg)  
(b) Single color different shape

![](images/0df505c76a34afcfec3f0396d17963079acf859afc4bc4e915ed7622b07bcc03.jpg)  
(c) Different color single shape

![](images/95efb50ccb8509289259c20b921a25dc05d51be9f652b3d6377392e13d32f5e5.jpg)  
(d) Different color different shape  
Figure 12: Layer-wise SVCCA similarity scores for various Vision Language Models and Datasets. High scores indicate strong alignment between ground truth and output probes.

<table><tr><td rowspan="2">Data</td><td rowspan="2">Model</td><td colspan="3">MLP</td><td colspan="3">Log-S</td><td colspan="3">Circ-J</td><td colspan="3">Circ-S</td></tr><tr><td>TP</td><td>FP</td><td> $\Delta \text{Acc (pp)}$ </td><td>TP</td><td>FP</td><td> $\Delta \text{Acc (pp)}$ </td><td>TP</td><td>FP</td><td> $\Delta \text{Acc (pp)}$ </td><td>TP</td><td>FP</td><td> $\Delta \text{Acc (pp)}$ </td></tr><tr><td rowspan="4">DC-DS</td><td>IVL2-1B</td><td>8.9</td><td>86.6</td><td>5.26</td><td>10.1</td><td>88.6</td><td>6.41</td><td>8.8</td><td>85.1</td><td>5.13</td><td>8.4</td><td>79.1</td><td>4.40</td></tr><tr><td>IVL2-4B</td><td>25.9</td><td>78.7</td><td>15.09</td><td>27.5</td><td>76.7</td><td>15.94</td><td>26.5</td><td>83.8</td><td>15.81</td><td>26.2</td><td>80.0</td><td>15.51</td></tr><tr><td>Q3-2B</td><td>4.2</td><td>97.6</td><td>0.56</td><td>5.3</td><td>95.0</td><td>0.43</td><td>4.4</td><td>98.1</td><td>0.56</td><td>5.5</td><td>95.0</td><td>0.56</td></tr><tr><td>Q3-8B</td><td>25.1</td><td>100.0</td><td>4.15</td><td>30.5</td><td>88.9</td><td>5.26</td><td>26.0</td><td>93.0</td><td>3.85</td><td>26.0</td><td>93.3</td><td>4.40</td></tr><tr><td rowspan="4">DC-SS</td><td>IVL2-1B</td><td>9.3</td><td>91.7</td><td>5.85</td><td>9.4</td><td>96.3</td><td>5.98</td><td>7.9</td><td>89.4</td><td>4.40</td><td>10.0</td><td>97.8</td><td>6.41</td></tr><tr><td>IVL2-4B</td><td>9.9</td><td>65.3</td><td>4.15</td><td>10.6</td><td>66.4</td><td>4.40</td><td>9.1</td><td>75.5</td><td>3.72</td><td>9.3</td><td>74.5</td><td>3.97</td></tr><tr><td>Q3-2B</td><td>4.2</td><td>97.3</td><td>0.56</td><td>6.1</td><td>97.4</td><td>0.73</td><td>5.3</td><td>98.4</td><td>0.98</td><td>7.7</td><td>100.0</td><td>1.28</td></tr><tr><td>Q3-8B</td><td>48.9</td><td>100.0</td><td>1.84</td><td>61.0</td><td>100.0</td><td>2.56</td><td>42.2</td><td>99.3</td><td>2.01</td><td>48.3</td><td>83.3</td><td>2.01</td></tr><tr><td rowspan="4">SC-DS</td><td>IVL2-1B</td><td>11.1</td><td>62.3</td><td>6.84</td><td>10.5</td><td>61.1</td><td>6.54</td><td>11.0</td><td>71.2</td><td>6.41</td><td>10.7</td><td>73.5</td><td>6.84</td></tr><tr><td>IVL2-4B</td><td>17.1</td><td>66.8</td><td>7.82</td><td>18.1</td><td>61.5</td><td>7.69</td><td>16.3</td><td>75.9</td><td>7.14</td><td>17.1</td><td>77.8</td><td>7.69</td></tr><tr><td>Q3-2B</td><td>5.0</td><td>96.4</td><td>1.15</td><td>7.3</td><td>91.0</td><td>1.28</td><td>5.2</td><td>95.9</td><td>1.15</td><td>4.6</td><td>92.1</td><td>0.73</td></tr><tr><td>Q3-8B</td><td>34.2</td><td>96.7</td><td>10.38</td><td>35.5</td><td>100.0</td><td>11.11</td><td>34.0</td><td>93.0</td><td>9.96</td><td>34.9</td><td>93.3</td><td>10.68</td></tr><tr><td rowspan="4">SC-SS</td><td>IVL2-1B</td><td>16.0</td><td>86.1</td><td>11.41</td><td>16.5</td><td>100.0</td><td>12.26</td><td>16.0</td><td>85.0</td><td>10.98</td><td>16.0</td><td>92.1</td><td>11.24</td></tr><tr><td>IVL2-4B</td><td>11.9</td><td>77.3</td><td>5.56</td><td>12.8</td><td>72.6</td><td>6.11</td><td>12.2</td><td>75.7</td><td>5.85</td><td>10.9</td><td>76.4</td><td>5.00</td></tr><tr><td>Q3-2B</td><td>9.5</td><td>93.4</td><td>0.98</td><td>14.7</td><td>94.0</td><td>2.44</td><td>13.2</td><td>95.4</td><td>2.26</td><td>12.4</td><td>92.0</td><td>1.58</td></tr><tr><td>Q3-8B</td><td>33.8</td><td>100.0</td><td>1.71</td><td>42.7</td><td>100.0</td><td>2.26</td><td>34.9</td><td>99.0</td><td>2.01</td><td>41.3</td><td>94.4</td><td>2.01</td></tr></table>

Table 4: Full detector analysis by model and dataset, averaged over the three seeds. Within each detector block, columns report TP (true-positive correction rate, %), FP (false-positive preservation rate, %), and $\Delta$ Acc (net accuracy gain in percentage points). Model abbreviations are IVL2 for InternVL2 and Q3 for Qwen3-VL; detector abbreviations are Log-S (Logistic-Separately), Circ-J (Circular-Jointly), and Circ-S (Circular-Separately).

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Raw</td><td colspan="3">stronger</td><td colspan="3">strongv2</td><td colspan="3">counting</td></tr><tr><td>Naive</td><td>Random-K</td><td>Ours (Δ)</td><td>Naive</td><td>Random-K</td><td>Ours (Δ)</td><td>Naive</td><td>Random-K</td><td>Ours (Δ)</td></tr><tr><td>diff_col_diff_shape</td><td>27.35</td><td>22.65</td><td>22.54</td><td>32.05 (+4.70)</td><td>26.92</td><td>25.43</td><td>30.77 (+3.42)</td><td>22.65</td><td>22.97</td><td>36.01 (+8.65)</td></tr><tr><td>diff_col_sing_shape</td><td>29.49</td><td>27.78</td><td>26.71</td><td>36.22 (+6.73)</td><td>30.34</td><td>27.46</td><td>36.64 (+7.16)</td><td>23.93</td><td>23.72</td><td>37.71 (+8.23)</td></tr><tr><td>sing_col_diff_shape</td><td>22.65</td><td>22.22</td><td>20.94</td><td>29.59 (+6.94)</td><td>19.23</td><td>17.52</td><td>26.71 (+4.06)</td><td>27.35</td><td>23.29</td><td>39.32 (+16.67)</td></tr><tr><td>sing_col_sing_shape</td><td>23.93</td><td>31.20</td><td>28.21</td><td>33.97 (+10.04)</td><td>29.06</td><td>25.21</td><td>34.61 (+10.69)</td><td>18.38</td><td>18.80</td><td>28.42 (+4.49)</td></tr></table>

Table 5: Comparison of reprompting strategies for InternVL2-1B synthetic datasets. Results are averaged across four probe architectures (MLP, Circular-Separately, Logistic-Separately, and Circular-Jointly).

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Base Acc.</td><td colspan="2">Entropy-Guided</td><td colspan="2">Probe-Guided (Ours)</td></tr><tr><td>Acc.</td><td>Δ</td><td>Acc.</td><td>Δ</td></tr><tr><td>diff_col_diff_shape</td><td>27.35%</td><td>25.64%</td><td>-1.71%</td><td>34.62%</td><td>+7.26%</td></tr><tr><td>diff_col_sing_shape</td><td>29.49%</td><td>31.20%</td><td>+1.71%</td><td>38.03%</td><td>+8.55%</td></tr><tr><td>sing_col_diff_shape</td><td>22.65%</td><td>29.06%</td><td>+6.41%</td><td>38.89%</td><td>+16.24%</td></tr><tr><td>sing_col_sing_shape</td><td>23.93%</td><td>20.51%</td><td>-3.42%</td><td>28.63%</td><td>+4.70%</td></tr></table>

Table 6: Comparison of Entropy-guided versus Probe-guided self-correction using the counting prompt strategy.

![](images/90d392a1ac3082d9aaa540669d91bedc3f49c0ef3460f69fcfeabfeeace8e93c.jpg)  
(a) Mean weight-SVCCA by probe family.

![](images/eaca2803e235e07b0c60ed896ca33834582a756b8d2c5c280f7c53d368634d76.jpg)  
(b) Weight-SVCCA versus best-layer gap.  
Figure 13: Weight-space SVCCA analysis of gt\_probe versus output\_probe. (Left) Mean SVCCA by probe family across the full 4-model, 3-seed sweep. MLP and Circular probes lie in the near-orthogonal regime, while Linear probes show substantially higher alignment. (Right) SVCCA versus best-layer gap $|l_{gt} - l_{out}|$ . Larger layer disagreement tends to coincide with lower alignment, but low SVCCA persists even at small layer gaps, indicating that subspace divergence is not reducible to depth mismatch alone. CountBench Dataset

![](images/dfbcf137f1477afb2e4b465f78470c87a60036f3ff226b00d5f8fd4be2a9197b.jpg)  
(a) Mean activation-SVCCA by probe family.

![](images/3acd7253c31f09b02f87200cb923554113740cbdcf79333b69e4c01faeae34a0.jpg)  
(b) Activation-SVCCA versus best-layer gap.  
Figure 14: Activation-space SVCCA analysis of gt\_probe versus output\_probe.

![](images/1bbc2b618f3f221b10cebf1836ec6d6ca69fea34054c3d4c64599751f502e4ec.jpg)  
(a) Layer-wise probing F1 scores across vision encoder depths (ground truth objective)

![](images/0a7c2c217fbfd804dc64cf1434628a59e8b1551b71e4f55c29ad0f23d087ca03.jpg)  
(b) Layer-wise probing F1 score across vision encoder depths (model output objective).

![](images/f36cd978f7123605958cacda28bcc7158d066a35df68a6362ce370488a68fb31.jpg)  
(c) Error detection performance (F1-score) in the vision encoder.

Figure 15: This figure illustrates performance aggregated across models and seeds, where activations for each image are computed as the mean of tokens across all image patches. The top plot shows ground-truth probe F1, measuring feature alignment with the actual object count. The middle plot tracks output-supervised probe F1, assessing alignment with the model's first-pass result. The bottom plot displays error-detector F1, indicating the system's reliability in predicting mistakes to trigger the intervention path. DC-DS: Different Color Different Shape

![](images/b7ad5b793cc49881e694e906322f1214372460004c452b3c4a691fa15e305372.jpg)  
(a) Layer-wise probing F1 scores across vision encoder depths (ground truth objective)

![](images/1dfdd5eb444073ef671555c74ff87c3a29226b7d0c00d3906924218643068fea.jpg)  
(b) Layer-wise probing F1 score across vision encoder depths (model output objective).

![](images/0fbdb8f860814874b2ee7b6a62ed6393ffbe73c5bdae999579f6cd89f22c38f4.jpg)  
(c) Error detection performance (F1-score) in the vision encoder.

Figure 16: This figure illustrates performance aggregated across models and seeds, where activations for each image are computed as the mean of tokens across all image patches. The top plot shows ground-truth probe F1, measuring feature alignment with the actual object count. The middle plot tracks output-supervised probe F1, assessing alignment with the model's first-pass result. The bottom plot displays error-detector F1, indicating the system's reliability in predicting mistakes to trigger the intervention path. DC-SS: Different Color Single Shape

![](images/49a4ffce13f50d027656124bbd8a7ff299d2068a0377c472963dd5604e0c29b2.jpg)  
(a) Layer-wise probing F1 scores across vision encoder depths (ground truth objective)

![](images/f2bf1b123848dc45a9f83df2bce32854b11ee74d973a864467ef7f85a6cefd25.jpg)  
(b) Layer-wise probing F1 score across vision encoder depths (model output objective).

![](images/984df2f491b46e2d8929e4e8c8246ab27d0ddf54ef5a784f1d666cc57a4c6b63.jpg)  
(c) Error detection performance (F1-score) in the vision encoder.

Figure 17: This figure illustrates performance aggregated across models and seeds, where activations for each image are computed as the mean of tokens across all image patches. The top plot shows ground-truth probe F1, measuring feature alignment with the actual object count. The middle plot tracks output-supervised probe F1, assessing alignment with the model's first-pass result. The bottom plot displays error-detector F1, indicating the system's reliability in predicting mistakes to trigger the intervention path. SC-DS: Single Color Different Shape

![](images/3e3c58a99dc2d72a1f7358a8d1bcca353e8d213450f82d379262d5d0d01a7201.jpg)  
(a) Layer-wise probing F1 scores across vision encoder depths (ground truth objective)

![](images/0ba00fc1001ed30e887305f2b99c036c9b2bdb5ec20bde064b5b4ff9a56057fd.jpg)  
(b) Layer-wise probing F1 score across vision encoder depths (model output objective).

![](images/50138cb317cebabe9d29fce427e36371019f04576f5c6cb0b66aee77a490f098.jpg)  
(c) Error detection performance (F1-score) in the vision encoder.

Figure 18: This figure illustrates performance aggregated across models and seeds, where activations for each image are computed as the mean of tokens across all image patches. The top plot shows ground-truth probe F1, measuring feature alignment with the actual object count. The middle plot tracks output-supervised probe F1, assessing alignment with the model's first-pass result. The bottom plot displays error-detector F1, indicating the system's reliability in predicting mistakes to trigger the intervention path. SC-SS: Single Color Single Shape

![](images/9d68bd000726bba75c48094487059a5c97231b73e46f3264e831e9aae2956be7.jpg)  
(a) Layer-wise probing F1 scores across vision encoder depths (ground truth objective)

![](images/e77c3211af0074070ae6c3547efeba30f1e9e1470e9f0acfae797f502386396f.jpg)  
(b) Layer-wise probing F1 score across vision encoder depths (model output objective).

![](images/ba9589683d067f873f12e3b87486dbe903591e4086e8858912a2e67f6ae6011e.jpg)  
(c) Error detection performance (F1-score) in the vision encoder.

Figure 19: This figure illustrates performance aggregated across models and seeds, where activations for each image are extracted from the last token of the vision encoder. The top plot shows ground-truth probe F1, measuring feature alignment with the actual object count. The middle plot tracks output-supervised probe F1, assessing alignment with the model's first-pass result. The bottom plot displays error-detector F1, indicating the system's reliability in predicting mistakes to trigger the intervention path. DC-DS: Different Color Different Shape

![](images/7d1c3fd2f9395166552cb8d95883ff04f6461fbd51d17577fbb6100161203eb2.jpg)  
(a) Layer-wise probing F1 scores across vision encoder depths (ground truth objective)

![](images/ed745449ddd3696a1aab847fa8d3f8febf532e83bce1510ec3c1be5bcef53244.jpg)  
(b) Layer-wise probing F1 score across vision encoder depths (model output objective).

![](images/6ef27f9057ba624867d7565f4e8eee9a35140a7112ce224f32095c13f7dbcd2a.jpg)  
(c) Error detection performance (F1-score) in the vision encoder.

Figure 20: This figure illustrates performance aggregated across models and seeds, where activations for each image are extracted from the last token of the vision encoder. The top plot shows ground-truth probe F1, measuring feature alignment with the actual object count. The middle plot tracks output-supervised probe F1, assessing alignment with the model's first-pass result. The bottom plot displays error-detector F1, indicating the system's reliability in predicting mistakes to trigger the intervention path. DC-SS: Different Color Single Shape

![](images/51f9cbf0c6fc4ac933a56f4bd954ea559a057faec59afa6ed38db9b9a87aeb01.jpg)  
(a) Layer-wise probing F1 scores across vision encoder depths (ground truth objective)

![](images/c588907acaf8c993940ca5ffb61a6718e2d164b32f7f585765f03009656a9d34.jpg)  
(b) Layer-wise probing F1 score across vision encoder depths (model output objective).

![](images/537a4bdebd1c49b0aba8fc7eb371dc9e98f184e25bada58eeb0fe93b9e4caef4.jpg)  
(c) Error detection performance (F1-score) in the vision encoder.

Figure 21: This figure illustrates performance aggregated across models and seeds, where activations for each image are extracted from the last token of the vision encoder. The top plot shows ground-truth probe F1, measuring feature alignment with the actual object count. The middle plot tracks output-supervised probe F1, assessing alignment with the model's first-pass result. The bottom plot displays error-detector F1, indicating the system's reliability in predicting mistakes to trigger the intervention path. SC-DS: Single Color Different Shape

![](images/24c803b53be147b3120bbd89a99e3db8b4e5cc14ec4894af9f5ad222fd1d2f9b.jpg)  
(a) Layer-wise probing F1 scores across vision encoder depths (ground truth objective)

![](images/ca4166d8f2b3cf1357a1de36f6f230b6edc0080380eb4a64f67866bd0722252a.jpg)  
(b) Layer-wise probing F1 score across vision encoder depths (model output objective).

![](images/70823b20fcc96c4a1fbefd7138294c059231b4288a8d12fe5442e1cae1e9e281.jpg)  
(c) Error detection performance (F1-score) in the vision encoder.

Figure 22: This figure illustrates performance aggregated across models and seeds, where activations for each image are extracted from the last token of the vision encoder. The top plot shows ground-truth probe F1, measuring feature alignment with the actual object count. The middle plot tracks output-supervised probe F1, assessing alignment with the model's first-pass result. The bottom plot displays error-detector F1, indicating the system's reliability in predicting mistakes to trigger the intervention path. SC-SS: Single Color Single Shape