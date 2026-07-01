# Visual Semantic Entropy: Do Vision Language Models Recognize Visual Ambiguity?

Ta Duc Huy $^{1}$ , Trang Nguyen $^{1}$ Townim Chowdhury $^{1}$ , Ankit Yadav $^{1}$ , Minh-Son To $^{2}$ Zhibin Liao $^{1}$ , Johan W. Verjans $^{1}$ , and Vu Minh Hieu Phan $^{1}$

$^{1}$ Australian Institute for Machine Learning, Adelaide University, Australia $^{2}$ Flinders University, Australia
tdh512194@gmail.com

Abstract. Vision-language models can produce confident answers on visually ambiguous inputs, resulting in biased predictions. Common entropy-based methods, such as Semantic Entropy (SE), rely on output diversity. Yet our analysis shows that overconfident visual embeddings suppress output diversity under stochastic decoding, causing SE to underestimate uncertainty in such cases. Recent methods instead probe output diversity through input perturbations, including textual paraphrasing or joint text-image perturbations, and show improved performance. We study these approaches and reveals that the resulting variability is often dominated by textual changes rather than visual evidence, causing uncertainty estimates to reflect prompt sensitivity rather than visual ambiguity. We therefore propose Visual Semantic Entropy (VSE), which perturbs only the image to probe nearby visual variations while keeping the text query fixed. VSE measures uncertainty by clustering generated answers into semantic prototypes and computing the mass-weighted dispersion among them. Extensive evaluation across five modern vision-language models and five diverse VQA benchmarks demonstrates that VSE effectively captures visual ambiguity, establishing a new state-of-the-art for VLM uncertainty estimation. Code is available: https://github.com/tadeephuy/visual-semantic-entropy

Keywords: vision-language model · visual semantic entropy

## 1 Introduction

Vision-language models (VLMs) excel at visual question answering (VQA), a setting in which uncertainty can arise from ambiguity in the visual evidence. In this work, we study uncertainty estimation for VLMs in VQA. Reliable uncertainty estimates are essential for detecting unreliable predictions and enabling trustworthy deployment of VLMs. Current approaches to uncertainty estimation (UE) in VLMs include verbalized methods that prompt models to report confidence $[5, 19, 21, 25]$ , logit-based methods derived from output probabilities or token entropy $[7,10]$ , and consistency-based methods that measure agreement across sampled generations $[8,13,28]$ . Another widely used class is entropy-based methods, which estimate uncertainty from the semantic distribution of sampled answers, such as Semantic Entropy (SE) and its variants $[3,15,17,27]$ .

![](images/107a5870595f5d560c4e1c3c014fc87370664e72eda453fe7f18bf2e15393dde.jpg)  
Fig. 1: VLMs can be confidently wrong on ambiguous images. Sampling-based methods such as Semantic Entropy estimate uncertainty by repeatedly sampling model outputs and computing the entropy of the resulting answer distribution, where low entropy indicates high certainty. Left (Easy): For a visually clear image, the model correctly predicts “pouch” and repeated sampling consistently outputs “pouch”, producing low entropy that appropriately reflects high certainty. Middle (Ambiguous): When the image is visually biased, the ground truth is “bag” but the model predicts “pouch”. Despite this ambiguity, repeated sampling still yields only “pouch” resulting in low entropy, which is misleading. Right (Ambiguous, perturbed): After perturbing the image to probe alternative local views, sampling produces diverse answers such as “pouch”, “bag”, and “pocket”, increasing entropy and exposing visual uncertainty.

In this paper, we set out to analyze existing UE methods and identify 3 limitations that motivate our approach: (1) Overconfident visual representations can produce highly consistent outputs under sampling, causing entropy-based methods such as SE [3] to underestimate uncertainty. (2) Wording variations among semantically equivalent answers can inflate pairwise semantic distances, leading methods such as SNNE [15] to overestimate uncertainty. (3) Textual perturbations can dominate output variability, causing methods that rely on text paraphrasing or joint text-image perturbations (e.g., VL-Uncertainty [27] and C&U [8]) to reflect prompt sensitivity rather than visual ambiguity.

First, Semantic Entropy $[3]$ estimates uncertainty by sampling multiple outputs under stochastic decoding and computing the entropy of the answer distribution, where diverse answers correspond to high entropy and consistent answers to low entropy. While effective, this approach assumes that epistemic uncertainty manifests through decoding randomness. In VLMs, however, uncertainty can originate from the visual input which SE does not effectively capture (Fig. 1). We show that visually biased images can induce visual embeddings that lead to highly confident predictions (Fig. 3). Under such conditions, repeated stochastic decoding produces little output variation, rendering SE ineffective (Sec. 3.2).

![](images/591c80eb966ecb9c1ddf75c1b54ede50a755505af728b6e03a7d4ad6bad7fc91.jpg)

![](images/4de4a6df71d5c16ddd84451115d94dc0f957397b6e69e0ab8ab32ddbd3401003.jpg)  
Fig. 2: Text perturbations induce large semantic shifts. Left: Given an input image and question, we generate perturbed images and textual paraphrases. In the multimodal embedding space, image perturbations form a tight local cluster around the original representation (gray dashed circle), while text paraphrases produce larger semantic shifts. Right: Cosine distance distributions between perturbed samples and the original representation, measured on AOKVQA dataset [20] using Qwen3-VL-Embedding-8B [9]. Text perturbations lead to substantially larger deviations than image perturbations (mean distance $0.056 \pm 0.014$ vs. $0.012 \pm 0.007$ ).

Second, given the resulting answer variants, SE treats answers as categorical labels and estimates uncertainty by counting their frequencies, ignoring semantic distances between categories. Semantic Nearest Neighbor Entropy (SNNE) $[15]$ addresses this by incorporating semantic distances through pairwise aggregation. However, SNNE operates directly on text outputs. Because language is discrete, answers that share the same meaning but differ in wording may appear separated in embedding space. As a result, distance-based aggregation can mistake wording variation for semantic disagreement and inflate uncertainty (Sec. 3.4).

Third, recent work $[8,27]$ explores input-space perturbations through textual paraphrasing or joint text-image perturbations. As illustrated in Fig. 2, due to prompt sensitivity, text paraphrases can induce non-local, larger semantic shifts than visual perturbations, which typically produce only small, local changes. While such perturbations increase output diversity, the resulting variability is often dominated by textual changes rather than visual evidence, with clustering is largely text-driven (Fig. 5, Sec. 3.3).

To address these limitations, we propose Visual Semantic Entropy (VSE), which perturbs the image to induce alternative answers and measures uncertainty through dispersion among semantic prototypes of the generated responses. First, VSE only perturbs the input image, leaving the text query unchanged, to probe prediction instability that sampling alone fails to capture. Second, VSE clusters semantically similar answers into prototypes so that wording variations do not inflate uncertainty. Uncertainty is measured as the frequency-weighted pairwise distance among prototypes. Finally, by perturbing only the image, VSE avoids modality confounding from textual perturbations and yields uncertainty that better reflects visual ambiguity. Extensive experiments show that VSE is particularly effective on visually adversarial datasets, outperforming methods based on text perturbation and joint image-text perturbations. In summary, our contributions are:

1. Our analysis reveals three failure modes of current VLMs uncertainty estimators: (1) suppressed output variation from overconfident visual embeddings; (2) inflated distances from wording variation; and (3) textual perturbations dominate output variability, causing uncertainty to reflect prompt sensitivity rather than visual ambiguity.

2. We introduce Visual Semantic Entropy (VSE), which perturbs the image, clusters semantically similar answers, and quantifies uncertainty using weighted pairwise distances between prototype embeddings.

3. We present an extensive benchmark across 5 VLMs and 5 datasets, evaluating multiple uncertainty estimation methods and showing that VSE consistently yields more reliable uncertainty estimates.

## 2 Related Work

Uncertainty Estimation in VLMs aims to quantify the reliability of model predictions, which is particularly important for visual question answering (VQA), where ambiguous visual evidence can lead to unreliable answers.

Verbalized methods prompt models to report confidence $[5,19,21,25]$ , while logit-based methods derive uncertainty from output probabilities $[7,10]$ . Both do not explicitly account for visual ambiguity.

Consistency-based approaches, including SelfCheckGPT $[13]$ and C&U $[8]$ , estimate uncertainty by measuring agreement between sampled responses and the original answer. C&U additionally generates question paraphrases before probing agreement. However, when visual embeddings are confident, stochastic decoding can yield highly consistent outputs, causing these methods to underestimate uncertainty.

Entropy-based approaches estimate uncertainty from the semantic distribution of sampled answers. Semantic Entropy (SE) [3] computes entropy over outputs sampled under stochastic decoding. Later methods improve semantic representation, including SNNE [15] and KLE [17], which aggregates pairwise distances across sampled answers. However, similar to consistency-based methods, these approaches rely on diversity in generated text under stochastic decoding and may fail to capture visual uncertainty when visual embeddings are confident, as shown in our analysis (Sec. 3.2). Moreover, pairwise distance aggregation can inflate uncertainty due to wording variations across answers(Sec. 3.4).

Input Perturbation. Recent work probes uncertainty through input perturbations. C&U [8] perturbs the question, while VL-Uncertainty [27] perturbs both the image and the text. Our analysis shows that textual perturbations often dominate output variability due to prompt sensitivity, causing uncertainty estimates to reflect linguistic variation rather than visual ambiguity (Sec. 3.3).

Motivated by these limitations, we focus on uncertainty arising from visual ambiguity. Our design probes uncertainty through visual perturbations while avoiding text perturbations that introduce prompt sensitivity, and aggregates predictions at the semantic level to mitigate wording variations.

## 3 Problem Analysis

We analyze limitations of existing uncertainty estimators in VLMs and formalize two hypotheses and one proposition that guide our method.

## 3.1 Problem Formulation

A visual question instance consists of a question q and an image v. Given $(q,v)$ , a VLM produces an answer $a = \text{VLM}(q, v)$ . Our goal is to estimate the reliability of this prediction. Specifically, we seek an uncertainty estimator U such that $\tilde{u} = U(q, v; \text{VLM})$ , where larger uncertainty score $\tilde{u}$ indicates a higher likelihood that the generated answer a is incorrect. This is of high interests in scenarios where unreliable predictions must be filtered.

## 3.2 Confident visual embeddings induce low semantic entropy

Semantic Entropy estimates uncertainty by sampling multiple outputs under stochastic decoding and computing the entropy of the resulting categorical answer distribution. This approach assumes that epistemic uncertainty manifests as variability induced by decoding randomness. In VLMs, however, predictions are also conditioned on visual embeddings. When these embeddings are highly confident, stochastic decoding produces nearly identical outputs, leading to low SE even if the underlying image is ambiguous.

Hypothesis 1. Confident visual embeddings induce low semantic entropy. If the visual embedding yields a sharply peaked conditional answer distribution, decoding randomness alone cannot generate diverse outputs. As a result, SE may underestimate visual uncertainty.

To test this hypothesis, we utilize the Adversarial split of the VILP dataset [12], which consists of visually ambiguous images. We then filter samples that the VLM answers incorrectly and quantify their visual confidence as follows: Let $\mathbf{v}_i\in \mathbb{R}^D$ denote the embedding of the $i$ -th visual token at the final layer and $W\in \mathbb{R}^{\mathcal{V}\times D}$ the language modeling head with vocabulary size of $\mathcal{V}$ of the VLM. Each visual token is projected into the vocabulary space using LogitLens [18]:

$$
\mathbf {z} _ {i} = W \mathbf {v} _ {i}, \qquad \tilde {\mathbf {z}} _ {i} = \mathrm{softmax} (\mathbf {z} _ {i}).\tag{1}
$$

We compute the entropy of each visual token distribution and average across visual tokens to obtain the visual entropy $H_{vis}$ :

$$
H _ {\mathrm{vis}} (x) = \underset {i} {\text { avg }} \left(- \sum_ {k = 1} ^ {| \mathcal {V} |} \tilde {\mathbf {z}} _ {i} (k) \log \tilde {\mathbf {z}} _ {i} (k)\right).\tag{2}
$$

Lower visual entropy $H_{vis}$ indicates a more confident visual representation. Based on this metric, we partition samples into two groups: Visually confident (low visual entropy) and Visually uncertain (high visual entropy). Importantly, this metric only reflects the model certainty in the visual input, not the predicted answer. We expect high SE (uncertainty) in both groups, since the model answers these samples incorrectly. However, as shown in Fig. 3-Left, standard SE yields high uncertainty only for visually uncertain samples. Visually confident samples exhibit low SE despite the visual ambiguity, indicating that confident visual embeddings suppress output variability, yielding low semantic entropy.

![](images/75d355177bdda32235289aafabd1f81bd894c57d69bbec59c935e0c65ca4c537.jpg)

![](images/8e1a7e7a6215d12efa42d37eca583fdfd271b0ed03d66f8bd0ab57c67d46a78c.jpg)  
Fig. 3: Decoding-based Semantic Entropy underestimates visual ambiguity. On the samples with incorrect answer, models are expected to express high uncertainty (SE). Using LogitLens [18] on the original image, we partition samples into Visually Confident (low visual entropy) and Visually Uncertain (high visual entropy). Left: Under stochastic decoding, high SE appears only for Visually Uncertain samples. Visually Confident samples remain low despite the ambiguous input. Right: After perturbing the image and aggregating predictions across perturbed views, we can induce high SE for Visually Confident samples. Results are shown for Qwen2.5-VL on VILP dataset.

We then perturb the input image to generate alternative local views while preserving semantic content. As shown in Fig. 3-Right, perturbation induces higher semantic entropy for visually confident samples. We note that the grouping is determined by calculating visual entropy $H_{\mathrm{vis}}$ on the original image. More details on experimental settings are included in the Supplementary.

Discussion. These results demonstrate that decoding stochasticity alone fails under confident visual embeddings, while image perturbation induces high semantic entropy for visually ambiguous inputs. This motivates probing uncertainty in the visual input space.

## 3.3 Text perturbation dominates output semantic shifts

Beyond perturbing visual inputs, recent work also perturbs the text query $[8,27]$ . Unlike visual perturbations, which introduce small local changes around the same image representation, textual paraphrases can induce larger semantic shifts. Semantically equivalent sentences may occupy distant regions in embedding space due to the discrete nature of language. As a result, paraphrasing can substantially alter output semantics. This motivates the following hypothesis.

Hypothesis 2. Text perturbations dominate output diversity under joint perturbation. We hypothesize that output diversity is driven primarily by textual variation rather than visual ambiguity. Then uncertainty measured under joint perturbation would mostly reflect prompt sensitivity.

Table 1: Text $(P_{T})$ vs. Image $(P_{I})$ perturbations Purity on VILP dataset for Qwen2.5-VL, Gemma3, Intern3.5-VL and LlaVA-NeXT.

<table><tr><td rowspan="2"></td><td colspan="3">Qwen2.5-VL</td><td colspan="3">Gemma3</td><td colspan="3">Intern3.5-VL</td><td colspan="3">LlaVA-NeXT</td></tr><tr><td>Easy</td><td>Adv.</td><td>All</td><td>Easy</td><td>Adv.</td><td>All</td><td>Easy</td><td>Adv.</td><td>All</td><td>Easy</td><td>Adv.</td><td>All</td></tr><tr><td> $P_T(\%)$ </td><td>51.9</td><td>55.6</td><td>54.3</td><td>35.6</td><td>38.3</td><td>37.4</td><td>46.7</td><td>48.4</td><td>47.8</td><td>54.5</td><td>53.9</td><td>54.1</td></tr><tr><td> $P_I(\%)$ </td><td>36.4</td><td>40.1</td><td>38.8</td><td>30.4</td><td>36.8</td><td>34.6</td><td>37.8</td><td>41.4</td><td>40.2</td><td>41.0</td><td>44.8</td><td>43.4</td></tr></table>

To test this, we analyze several VLMs on the Easy and Adversarial split of the VILP dataset. For each image, we construct an $L \times M$ perturbation grid, where L denotes the number of textual paraphrases and M denotes the number of image perturbations. For each $(l, m)$ pair, we generate a model output and cluster the resulting $L \times M$ responses based on semantic similarity. This allows us to analyze how cluster membership depends on paraphrase identity or image perturbation.

Purity analysis. To quantify the alignment between clusters and perturbation types, we compute cluster purity. For each cluster c, we define text purity $P_{T}(c)$ and image purity $P_{I}(c)$ as the fraction of samples in c originating from the most frequent textual paraphrase and image perturbation. Let $n_{c}$ denote the size of cluster c, $n_{c}(l)$ the number of samples in c generated from paraphrase l, and $n_{c}(m)$ the number generated from image perturbation m. We define

$$
P _ {T} (c) = \frac {\max _ {l} n _ {c} (l)}{n _ {c}}, P _ {I} (c) = \frac {\max _ {m} n _ {c} (m)}{n _ {c}}.\tag{3}
$$

If $P_{T}(c) > P_{I}(c)$ , most samples in cluster c originate from a single paraphrase $l_{max}$ , indicating that cluster membership is primarily determined by the textual input rather than visual perturbations, and vice versa. To obtain an overall text and image purity for each image across clusters, we compute the size-weighted purity: $P_{T} = \sum_{c} \frac{n_{c}}{L \times M} P_{T}(c)$ and $P_{I} = \sum_{c} \frac{n_{c}}{L \times M} P_{I}(c)$ .

Overall, text purity consistently exceeds image purity (Tab. 1), and the gap widens on the Easy split. Furthermore, we estimate the joint density of $(P_{T}, P_{I})$ across images using kernel smoothing and visualize it as a filled contour plot (Fig. 4). The diagonal $P_{T} = P_{I}$ separates text-dominant (above) from image-dominant regions (below). We observe that the density mass concentrates predominantly above this line, meaning that for most images, cluster assignments align more strongly with paraphrase identity than with visual perturbations. This pattern shows that variability under joint perturbation is largely text-driven rather than visual ambiguity. Additional visualizations and details are in Supp.

Ta Duc Huy et al.

![](images/7925ea2899526e0a8c50034e76b92bef85532357480db706bad302670fe40431.jpg)

![](images/f649cffe202b94df5be762778936351625c4193fc333ef9504da25a68a7f2b3d.jpg)

![](images/43dc137bd51083bb1aa19bc90dd08efb07723b7c4ff64e0af8c14def561f6c1e.jpg)

![](images/09fd94f29b8430dd105b85fc5bfb4b8c3eb2ee4196075a40e179660ab92563d4.jpg)

Fig. 4: Kernel density of text $P_{T}$ and image purity $P_{I}$ across models.. The diagonal $P_{T} = P_{I}$ separates text- from image-dominant regions. Density mass lying predominantly above the diagonal indicates that clustering is primarily text-driven.  
![](images/01ff319c0a01dc46b9fc684ad01aff8d71a9102d34c9775d090192346e8df149.jpg)  
Fig. 5: Cluster occupancy map. Each panel shows cluster assignments on the $L \times M$ perturbation grid of a random sample (rows: textual paraphrases; columns: image perturbations). Horizontal stripes indicate invariance across image perturbations, showing that clustering is dominant by textual paraphrase. We provide more results in Supp.

To qualitatively examine this effect, we visualize cluster occupancy on the $L \times M$ perturbation grid. Rows correspond to textual paraphrases and columns to image perturbations. Under joint perturbation, we observe horizontal stripe patterns in the occupancy maps (Fig. 5), indicating that cluster membership is invariant across image perturbations but consistent within a given paraphrase. This pattern confirms that output are primarily conditioned on the textual input. Discussion. The quantitative purity analysis and qualitative examples support Hypothesis 2: under joint image-text perturbation, output diversity is largely driven by textual variation rather than visual ambiguity. Our finding aligns with Image-DPO [12], where the authors perturb only the image while keeping the question fixed to dismiss model over-reliance on language. In uncertainty estimation, prompt sensitivity leads paraphrasing to induce semantic shifts that inflate uncertainty without reflecting ambiguity in visual information. Hence, restricting perturbations to the visual domain avoids artificial prompt sensitivity and yields more faithful visual uncertainty.

## 3.4 Linguistic variations inflates distance-based uncertainty

Other approaches estimate uncertainty by measuring semantic distances among sampled outputs $[15, 17]$ . However, because answers are discrete text, semantically equivalent responses with different wording can be mapped to distant points in embedding space. As a result, distance-based aggregation may yield inflated uncertainty even when there is no true semantic disagreement.

Formally, given n sampling iterations, the model generates answer variants $\{a_{i}\}^{n}$ . A general distance-based estimator computes uncertainty as the average pairwise distance:

$$
U (q) = \frac {1}{n (n - 1)} \sum \mathrm{d} (a, a ^ {\prime}),\tag{4}
$$

where $d(\cdot,\cdot)$ is a semantic distance function.

Proposition 1. Distance inflation under linguistic variations. Suppose all sampled answers express the same semantic meaning. If wording differences induce non-zero embedding distances, $\mathrm{d}(a,a') > 0$ for some $a \neq a'$ , then $U(q) > 0$ despite the absence of semantic disagreement.

Intra- vs. Inter-cluster effect. To separate meaning-level disagreement from wording variation, consider partitioning the answers into semantic clusters, where each cluster groups answers that convey the same meaning. Let c denote the cluster assignment of a. For a distance metric $d(\cdot,\cdot)$ , pairwise aggregation can be decomposed as:

$$
U (q) = \sum_ {k = 1} ^ {K} \mathbb {E} [ \mathrm{d} (a, a ^ {\prime}) \mid c = c ^ {\prime} = k ] + \sum_ {k \neq k ^ {\prime}} \mathbb {E} [ \mathrm{d} (a, a ^ {\prime}) \mid c = k, c ^ {\prime} = k ^ {\prime} ].\tag{5}
$$

The first term captures variation within clusters, which mainly reflects wording differences. The second term captures variation across clusters, corresponding to genuine semantic disagreement. Because distance-based aggregation sums both terms, uncertainty is influenced not only by disagreement between meanings but also by wording variation within each meaning. Therefore, uncertainty estimation should first group semantically equivalent answers and then measure distances across groups rather than between individual text outputs.

## 4 Method

The analysis above shows three limitations of current uncertainty estimators: (1) decoding stochasticity underestimates uncertainty when visual embeddings are overly confident, (2) text perturbations can dominate output variability under joint image-text perturbation, and (3) linguistic variations inflate distance-based estimators. These observations suggest that VQA uncertainty estimation should probe ambiguity in the visual input space, avoid textual perturbations that introduce large semantic shifts, and aggregate outputs at the semantic level.

Guided by these principles, we propose Visual Semantic Entropy (VSE), a method that perturbs the image to elicit alternative visual interpretations (Sec. 4.1), clusters semantically equivalent answers into prototypes, and quantifies uncertainty as the weighted dispersion among prototype embeddings (Sec. 4.2). An overview of VSE is shown in Fig. 6.

![](images/490660c6dc2536320a31386fd09e2071e9400f0ceeb778d98767c4dc438c09bc.jpg)  
Fig. 6: Visual Semantic Entropy for VQA Uncertainty Estimation. (1) We perturb the input image to generate local visual variants. (2) The VLM produces multiple answer samples across perturbed views. (3) We cluster semantically equivalent answers, select a representative prototype for each cluster, and compute uncertainty as the mass-weighted dispersion among prototype embeddings.

## 4.1 Visual Perturbation

To probe visual epistemic uncertainty, we perturb the input image while keeping the question fixed. Unlike textual paraphrasing, which may induce non-local semantic shifts, visual perturbations introduce controlled variations around the original image, enabling us to explore local visual variants.

Let T denote a visual perturbation operator. Given an input image v, we generate M perturbed views:

$$
v _ {m} = \mathcal {T} (v; \xi_ {m}), \quad m = 1, \ldots , M,\tag{6}
$$

where $\xi_{m}$ parameterizes the perturbation. The operator T is designed to preserve high-level semantic content while inducing local variations in pixel space. Each perturbed image $v_{m}$ is paired with the same question q and fed into the VLM to obtain answer samples. Variability across these perturbed views $v_{m}$ serves as a probe of visual uncertainty.

Discussion. Prior work $[27]$ progressively increases perturbation magnitude to generate samples at multiple noise levels. In contrast, we fix a small perturbation scale for two reasons. First, uncertainty should quantify ambiguity conditioned on the original input $(q, v)$ . Progressively increasing perturbations magnitude may move samples outside the local neighborhood of v, so answer variation reflects changes to the input rather than visual ambiguity. Second, sampling-based estimators aggregate samples uniformly. Mixing perturbation magnitudes inappropriately assigns equal weight to local and non-local, large deviations. By using a single fixed scale, the measured dispersion consistently reflects variability within a local neighborhood.

## 4.2 Prototype Semantic Aggregation

Given M perturbed image variants $\{v_{m}\}^{M}$ and the question q, the VLM produces a corresponding set of answer samples $\{a_{m}\}^{M}$ . These samples may differ in wording or semantic meaning. We then perform Prototype Semantic Aggregation (ProtoSem) as follows:

Clustering. We cluster the answer variants $\{a_{m}\}$ into K groups $\{c_{k}\}_{k=1}^{K}$ based on semantic distance base on a function $\mathrm{d}(\cdot,\cdot)$ .

Prototype selection. For each cluster $c_{k}$ , we select a prototype answer $p_{k}$ that minimizes the average distance to other members in the same cluster:

$$
p _ {k} = \arg \min _ {a \in c _ {k}} \sum_ {a ^ {\prime} \in c _ {k}} \mathrm{d} (a, a ^ {\prime}).\tag{7}
$$

This prototype serves as the representative semantic meaning of cluster $k$ .

Prototype dispersion. Let $w_{k} = \frac{|c_{k}|}{M}$ denote the empirical probability of cluster k. We define uncertainty as the expected semantic distance between two independently drawn cluster prototypes:

$$
\tilde {u} = U (q, v; \mathrm{VLM}) = \mathbb {E} _ {k \sim w, k ^ {\prime} \sim w} [ \mathrm{d} (p _ {k}, p _ {k ^ {\prime}}) ] = \sum_ {k \neq k ^ {\prime}} w _ {k} w _ {k ^ {\prime}} \mathrm{d} (p _ {k}, p _ {k ^ {\prime}}).\tag{8}
$$

This measures disagreement among supported semantic meanings: semantic variations are consolidated within each cluster and represented by a single prototype $p_{k}$ , and each prototype is weighted by its mass $w_{k}$ . Uncertainty increases only when multiple high-mass prototypes are separated.

Discussion. SE $[3]$ treats answers as discrete categories and ignores semantic proximity, whereas SNNE $[15]$ measures pairwise distances and is sensitive to linguistic variation. VSE instead groups semantically equivalent outputs and compute weighted dispersion over cluster prototypes, eliminating intra-cluster wording inflation while preserving semantic disagreement across clusters.

## 5 Experiments

## 5.1 Setup

Metrics. Following prior work $[3,15]$ , we evaluate uncertainty estimation using the Area Under the ROC Curve (AUC). Each prediction is labeled as correct or incorrect, and the predicted uncertainty score is used to distinguish between them. AUC measures how well uncertainty scores rank incorrect predictions above correct ones across different thresholds. A higher AUC indicates better alignment between predicted uncertainty and model reliability.

Datasets. We evaluate our method on five benchmarks covering knowledge-based VQA, multimodal reasoning: OKVQA $[14]$ , AOKVQA $[20]$ , MMVet $[26]$ ; and adversarial datasets to expose visual bias: VILP $[12]$ , VLM-are-biased $[22]$ . We detail on the dataset and their splits in the Supplementary material.

Other Baselines. We conduct an extensive benchmark to compare VSE against these uncertainty estimation approaches: Verbalized: Verbalized Uncertainty $[5]$ . Logit-based: Confidence measures derived from output probabilities, including AvgEnt and MaxEnt (token entropy), and AvgProb and MaxProb (token probability) $[10]$ . Consistency-based: SelfCheckGPT $[13]$ and C&U $[8]$ . Entropy-based: Semantic Entropy (SE) $[3]$ , Semantic Nearest Neighbor Entropy (SNNE) $[15]$ , Kernel Language Entropy (KLE) $[17]$ , and VL-Uncertainty $[27]$ . Our method, VSE, also belongs to this category.

Models. We evaluate all approaches on five vision-language models that cover a range of architecture: Qwen2.5-VL-7B [2], Gemma3-4B [4], Intern3.5-VL-8B [23], LLaVA-NeXt-8B [11], and Qwen3-VL-8B [1].

Implementation Details. To perturb the image, we add Gaussian noise with standard deviation $\sigma = 20$ , introducing small visual variations while preserving semantic content. The initial answer a is generated using greedy decoding (T = 0.0). After obtaining this initial answer, we sample M = 10 additional responses with decoding temperature T = 1.0, following the setup of SE [3]. For answer clustering, we use the DeBERTa-v2-xlarge-mnli [6] as the semantic distance function $\mathrm{d}(\cdot, \cdot)$ and apply hierarchical clustering [16]. Additional hyperparameter analysis can be found in supplementary material.

## 5.2 Results

Qualitative Results. We report results on AOKVQA in Tab. 2. Entropy-based methods perform best overall, while logit-based approaches perform poorly across all models. Verbalized uncertainty performs competitively only on Qwen3-VL. Consistency-based methods also show strong performance, particularly C&U with text perturbations. Among entropy-based methods, VSE consistently achieves the best results across all models, reaching AUC scores of 0.783, 0.778, 0.724, 0.798, and 0.792 on Qwen2.5-VL, Gemma3, LLaVA-NeXT, Qwen3-VL, and Intern3.5-VL, and outperforming VL-Uncertainty by up to 9.2% AUC. The same trend holds on MMVet and OKVQA (Tabs. 3 and 4). VSE achieves the best performance across all models, improving over the strongest baselines such as SE (e.g., 0.767 vs. 0.758 on Qwen2.5-VL in OKVQA) and reaching AUC scores of 0.781 and 0.778 on MMVet for Qwen2.5-VL and Gemma3, respectively. We observe that SE and SNNE perform well on MMVet, suggesting that repeated sampling better captures uncertainty in this benchmark, which primarily evaluates reasoning ability rather than visual ambiguity.

Qualitative Results. We provide qualitative result in Fig. 7. More results are provided in Supplementary.

## 5.3 Ablation Study

VSE reflects visual ambiguity more faithfully. We evaluate uncertainty estimation on the visually adversarial datasets VILP and VLM-are-biased (Tab. 5), where images are intentionally designed to be highly ambiguous to probe model biases and test reliance on visual evidence. VSE consistently achieves the best

![](images/23ce1be852ea23ade3dfeeb20d6bc89e348aff6ea72fe1c2b024c49d888e9f45.jpg)

VSE: percentile:
0.638 0.973 ( $\uparrow$ )

Fig. 7: Qualitative Result of Visual Semantic Entropy. Top: VSE is high, showing high uncertainty for wrong prediction. Bottom: VSE is low, showing more certainty for correct prediction. Results are shown for Qwen2.5-VL on AOKVQA.  
Table 2: AOKVQA results. AUC scores of uncertainty estimation methods across Qwen2.5-VL, Gemma3, LLaVA-NeXT, Qwen3-VL, and Intern3.5-VL. Higher is better.

<table><tr><td>Category</td><td>Method</td><td>Venue</td><td>Qwen2.5</td><td>Gemma3</td><td>LLaVA</td><td>Qwen3</td><td>Intern3.5</td></tr><tr><td>Verbalized</td><td>Verb-U [5]</td><td>NAACL&#x27;24</td><td>0.635</td><td>0.654</td><td>0.580</td><td>0.700</td><td>0.630</td></tr><tr><td rowspan="4">Logit</td><td>AvgEnt [10]</td><td>EMNLP&#x27;24</td><td>0.625</td><td>0.530</td><td>0.635</td><td>0.581</td><td>0.613</td></tr><tr><td>MaxEnt [10]</td><td>EMNLP&#x27;24</td><td>0.596</td><td>0.529</td><td>0.638</td><td>0.578</td><td>0.632</td></tr><tr><td>AvgProb [10]</td><td>EMNLP&#x27;24</td><td>0.612</td><td>0.508</td><td>0.634</td><td>0.556</td><td>0.593</td></tr><tr><td>MaxProb [10]</td><td>EMNLP&#x27;24</td><td>0.600</td><td>0.561</td><td>0.522</td><td>0.541</td><td>0.512</td></tr><tr><td rowspan="3">Consistency</td><td>SCG-NLI [13]</td><td>EMNLP&#x27;23</td><td>0.575</td><td>0.576</td><td>0.579</td><td>0.591</td><td>0.618</td></tr><tr><td>SCG-Pr [13]</td><td>EMNLP&#x27;23</td><td>0.734</td><td>0.701</td><td>0.608</td><td>0.708</td><td>0.730</td></tr><tr><td>C&amp;U [8]</td><td>CVPR&#x27;24</td><td>0.731</td><td>0.712</td><td>0.602</td><td>0.720</td><td>0.734</td></tr><tr><td rowspan="4">Entropy</td><td>SE [3]</td><td>ICLR&#x27;23</td><td>0.702</td><td> $\underline{0.732}$ </td><td>0.622</td><td>0.746</td><td>0.775</td></tr><tr><td>SNNE [15]</td><td>ACL&#x27;25</td><td>0.744</td><td>0.700</td><td> $\underline{0.651}$ </td><td>0.747</td><td>0.745</td></tr><tr><td>KLE [17]</td><td>NIPS&#x27;25</td><td> $\underline{0.774}$ </td><td>0.721</td><td>0.541</td><td> $\underline{0.772}$ </td><td>0.764</td></tr><tr><td>VL-U [27]</td><td>arxiv</td><td>0.756</td><td>0.715</td><td>0.616</td><td>0.706</td><td> $\underline{0.781}$ </td></tr><tr><td></td><td>VSE</td><td>Ours</td><td>0.783</td><td>0.778</td><td>0.724</td><td>0.798</td><td>0.792</td></tr></table>

Table 3: MMVet results. AUC scores of uncertainty estimation methods across Qwen2.5-VL, Gemma3. Higher is better.

<table><tr><td>Model</td><td>Verb-U [5]</td><td>A-E [10]</td><td>A-P [10]</td><td>SCG [13]</td><td>C&amp;U [8]</td><td>SE [3]</td><td>SNNE [15]</td><td>VL-U [27]</td><td>VSE</td></tr><tr><td>Qwen2.5-VL</td><td>0.595</td><td>0.687</td><td>0.678</td><td>0.718</td><td>0.587</td><td>0.716</td><td> $\underline{0.758}$ </td><td>0.659</td><td>0.781</td></tr><tr><td>Gemma3</td><td>0.598</td><td>0.607</td><td>0.601</td><td>0.625</td><td>0.646</td><td>0.722</td><td> $\underline{0.740}$ </td><td>0.693</td><td>0.778</td></tr></table>

Table 4: OKVQA results. AUC scores of uncertainty estimation methods across Qwen2.5-VL, Gemma3, LLaVA-NeXT, Qwen3-VL. Higher is better.

<table><tr><td>Category</td><td>Method</td><td>Venue</td><td>Qwen2.5</td><td>Gemma3</td><td>LLaVA</td><td>Qwen3</td></tr><tr><td>Verbalized</td><td>Verb-U [5]</td><td>NAACL&#x27;24</td><td>0.663</td><td>0.647</td><td>0.532</td><td>0.593</td></tr><tr><td rowspan="4">Logit</td><td>AvgEnt [10]</td><td>EMNLP&#x27;24</td><td>0.690</td><td>0.577</td><td> $\underline{0.729}$ </td><td>0.703</td></tr><tr><td>MaxEnt [10]</td><td>EMNLP&#x27;24</td><td>0.714</td><td>0.567</td><td>0.724</td><td>0.715</td></tr><tr><td>AvgProb [10]</td><td>EMNLP&#x27;24</td><td>0.653</td><td>0.570</td><td>0.705</td><td>0.679</td></tr><tr><td>MaxProb [10]</td><td>EMNLP&#x27;24</td><td>0.519</td><td>0.555</td><td>0.542</td><td>0.565</td></tr><tr><td rowspan="3">Consistency</td><td>SCG-NLI [13]</td><td>EMNLP&#x27;23</td><td>0.714</td><td>0.584</td><td>0.658</td><td>0.597</td></tr><tr><td>SCG-Pr [13]</td><td>EMNLP&#x27;23</td><td> $\underline{0.767}$ </td><td>0.657</td><td>0.675</td><td>0.657</td></tr><tr><td>C&amp;U [8]</td><td>CVPR&#x27;24</td><td>0.742</td><td>0.678</td><td>0.718</td><td>0.693</td></tr><tr><td rowspan="4">Entropy</td><td>SE [3]</td><td>ICLR&#x27;23</td><td> $\underline{0.758}$ </td><td> $\underline{0.703}$ </td><td>0.685</td><td> $\underline{0.790}$ </td></tr><tr><td>SNNE [15]</td><td>ACL&#x27;25</td><td>0.751</td><td>0.667</td><td>0.698</td><td>0.711</td></tr><tr><td>KLE [17]</td><td>NIPS&#x27;25</td><td>0.744</td><td>0.673</td><td>0.672</td><td>0.724</td></tr><tr><td>VL-U [27]</td><td>arxiv</td><td>0.731</td><td>0.702</td><td>0.695</td><td>0.731</td></tr><tr><td></td><td>VSE</td><td>Ours</td><td>0.767</td><td>0.715</td><td>0.749</td><td>0.799</td></tr></table>

performance across all settings, improving over the strongest baseline by large margins (e.g., 0.650 vs. 0.535 on VILP with Qwen2.5-VL and 0.826 vs. 0.783 on VLM-are-biased). In contrast, several existing methods degrade substantially on these datasets, suggesting that they are sensitive to language biases or decoding artifacts. These results highlight that explicitly probing visual variations help to capture uncertainty arising from ambiguous visual inputs.

Table 5: AUC scores of uncertainty estimation methods on the VILP and VLM-are-biased benchmarks.

<table><tr><td>Dataset</td><td>Model</td><td>Verb-U [5]</td><td>A-E [10]</td><td>SE [3]</td><td>C&amp;U [8]</td><td>VL-U [27]</td><td>VSE</td></tr><tr><td rowspan="2">VILP</td><td>Qwen2.5-VL</td><td>0.521</td><td>0.507</td><td> $\underline{0.535}$ </td><td>0.515</td><td>0.489</td><td>0.650</td></tr><tr><td>Gemma3</td><td>0.503</td><td>0.528</td><td> $\underline{0.660}$ </td><td>0.540</td><td>0.501</td><td>0.665</td></tr><tr><td rowspan="2">VLM-are-biased</td><td>Qwen2.5-VL</td><td>0.722</td><td>0.537</td><td>0.728</td><td>0.697</td><td> $\underline{0.783}$ </td><td>0.826</td></tr><tr><td>Gemma3</td><td>0.600</td><td>0.694</td><td>0.700</td><td>0.737</td><td> $\underline{0.758}$ </td><td>0.776</td></tr></table>

Effect of Visual Perturbation and Prototype Aggregation. Tab. 6 evaluates two components of our approach. For SE and SNNE, we report the original

estimator (Base) and a variant where we add visual perturbation $(+T)$ . We also evaluate ProtoSem, our prototype semantic aggregation strategy, both with and without visual perturbation.

(1) Comparing Base and $+T$ shows consistent improvements for SE and SNNE across all models, with gains up to +9.5% AUC, demonstrating that visual perturbations are beneficial for uncertainty estimation.

(2) Comparing methods across $+T$ columns (with visual perturbation), ProtoSem consistently achieves the best performance, demonstrating that prototype-based semantic aggregation better captures uncertainty than entropy computed over sampled answers, as used in SE and SNNE.

These results justify the core design of VSE, combining visual perturbation with prototype-based semantic aggregation to improve uncertainty estimation.

Table 6: Effect of visual perturbation $(+T)$ and Prototype Semantic Aggregation (ProtoSem) on entropy-based uncertainty estimation methods on the AOKVQA dataset. The ProtoSem + T combination is our proposed VSE.

<table><tr><td rowspan="2">Model</td><td colspan="3">SE [3]</td><td colspan="3">SNNE [15]</td><td colspan="3">ProtoSem</td></tr><tr><td>Base</td><td> $+\mathcal{T}$ </td><td> $\Delta$ </td><td>Base</td><td> $+\mathcal{T}$ </td><td> $\Delta$ </td><td>Base</td><td> $+\mathcal{T}_{(VSE)}$ </td><td> $\Delta$ </td></tr><tr><td>Qwen2.5-VL</td><td>0.702</td><td>0.761</td><td>+0.059</td><td>0.700</td><td>0.720</td><td>+0.020</td><td>0.711</td><td>0.783</td><td>+0.072</td></tr><tr><td>Gemma3</td><td>0.732</td><td>0.775</td><td>+0.043</td><td>0.651</td><td>0.746</td><td>+0.095</td><td>0.745</td><td>0.778</td><td>+0.033</td></tr><tr><td>Qwen3-VL</td><td>0.747</td><td>0.788</td><td>+0.041</td><td>0.747</td><td>0.757</td><td>+0.010</td><td>0.762</td><td>0.798</td><td>+0.036</td></tr></table>

## 6 Conclusion

We study uncertainty estimation in VLMs for VQA, where ambiguity often arises from the visual input. Our analysis shows that decoding-based estimators can underestimate uncertainty when confident visual embeddings suppress output variation, while text perturbations and wording variations can inflate uncertainty estimates. Motivated by these observations, we propose Visual Semantic Entropy, which probes uncertainty through visual perturbations and aggregates predictions at the semantic prototype level. Experiments across multiple VLMs and VQA benchmarks demonstrate that VSE provides more reliable uncertainty estimates than existing approaches. A limitation of our method is the additional computational cost from perturbation-based sampling, which is shared with other sampling-based uncertainty estimators, as well as sensitivity to the decoding temperature used during generation. Our approach also relies on semantic similarity models to cluster answers, and its performance may depend on the quality of the semantic distance function. While we evaluate VSE across several representative VLMs and benchmarks, extending the analysis to larger model families and additional multimodal tasks remains an important direction for future work. We hope this work encourages future research on uncertainty estimation that explicitly accounts for visual ambiguity.

# Supplementary Material

## A Implementation details

## A.1 Algorithm

We summarize VSE in Alg. 1.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Visual Semantic Entropy (VSE)

Require: Question q, image v, VLM, perturbator T, number of perturbations M

Ensure: Uncertainty score  $\tilde{u}$ 

1:  $a \leftarrow \text{VLM}(q, v; T = 0)$ $\triangleright$  Greedy decoding

2: for  $m = 1, \ldots, M$  do

3:  $v_m \leftarrow \mathcal{T}(v; \xi_m)$ $\triangleright$  Visual perturbation

4:  $a_m \leftarrow \text{VLM}(q, v_m; T &gt; 0)$ $\triangleright$  Sampling with perturbed image

5: end for

6: Cluster  $\{a_m\}_{m=1}^M$  into K clusters  $\{c_k\}_{k=1}^K$  using d( $\cdot, \cdot$ )

7: for  $k = 1, \ldots, K$  do

8:  $p_k \leftarrow \arg\min_{a \in c_k} \sum_{a' \in c_k} d(a, a')$ $\triangleright$  Prototype assignment

9:  $w_k \leftarrow |c_k|/M$ 

10: end for

11:  $\tilde{u} \leftarrow \sum_{k \neq k'} w_k w_{k'} d(p_k, p_{k'})$ 

12: return  $\tilde{u}$ $\triangleright$  Higher  $\tilde{u}$  indicates higher likelihood that a is incorrect
</div>

## A.2 Datasets

The datasets used in this work are:

AOKVQA: A knowledge-intensive VQA dataset where answering questions requires commonsense and world knowledge grounded in the image. It contains 17,056 / 1,145 / 6,702 samples for train/val/test; since test labels are unavailable, we evaluate on the validation split.

MMVet: A challenging benchmark designed to evaluate integrated multimodal reasoning abilities, combining skills such as recognition, OCR, spatial reasoning, and knowledge grounding. The benchmark contains 218 evaluation samples.

OKVQA: A knowledge-based VQA benchmark where answering questions requires external world knowledge beyond what is directly visible in the image, such as object functions, cultural facts, or scientific concepts. The dataset contains 14,055 questions with 9,009 / 5,046 train/test splits, and we report results on the test set.

VILP: A dataset probing language prior bias, where language cues suggest an answer that contradicts the visual evidence. The dataset contains 900 evaluation samples, with 300 Easy images, each of which has 2 Adversarial variants.

VLM-are-biased: A diagnostic benchmark exposing confirmation bias in VLMs through visually misleading examples that encourage reliance on memorized associations. We use the original split of 458 samples.

Together, these benchmarks evaluate uncertainty estimation across diverse VQA settings, with VILP and VLM-are-biased specifically probing visual uncertainty.

## A.3 Confident Visual Embeddings

![](images/89b44b11cbc78c8c09e10ce7c809c4516961389a53d4e4289dd71d60a7bc3373.jpg)  
Fig. 1: Visual entropy estimation. Final-layer visual token embeddings are projected into the vocabulary space via the language model head (LogitLens). The entropy of the resulting token distributions is averaged to obtain the visual entropy $H_{\mathrm{vis}}$ , which measures the model's confidence in the visual input.

As illustrated in Fig. 1, we estimate the model's visual confidence by probing the final-layer visual token representations. Each visual token embedding is projected into the vocabulary space using the language model head via LogitLens [18], producing a token distribution over the vocabulary. We compute the entropy of each projected distribution and average across visual tokens to obtain the visual entropy $H_{\mathrm{vis}}$ , which reflects the model's confidence in the visual representation. This metric allows us to distinguish visually confident inputs from visually uncertain ones, enabling us to analyze cases where semantic entropy fails to reflect uncertainty despite ambiguous visual evidence. We use the top/bottom $20\%$ percentile to define high and low visual entropy. We additionally test top/bot. $10\% \& 30\%$ percentiles and also observe increased entropy under perturbations (Fig. 2-Left).

Robustness. We further analyze the observation in Sec. 3.2 using Qwen2.5-VL and Gemma3 on VILP and VAB (Fig. 2-Right). Quantitatively, we measure the entropy increase and report the mean $\Delta H$ with 1-sided Wilcoxon signed-rank test p-value in Table. 1. We consistently observe positive entropy shifts, with most being statistically significant, suggesting robustness across datasets and models.

![](images/d4ec1f17d188e06d29d118a77eff9c58b1aeebb15a76587f6c98b4b511a95565.jpg)  
Fig. 2: Left: Perturbation effects under 10% & 30% percentiles. Right: Perturbation effects using Gemma3 on VILP and VAB

Table 1: $\Delta H$ under perturbations.

<table><tr><td></td><td>VILP-Qwen</td><td>VILP-Gemma</td><td>VAB-Qwen</td><td>VAB-Gemma</td></tr><tr><td> $\Delta H(p)$ </td><td>0.1625 (0.0203)</td><td>0.0939 (0.0238)</td><td>0.0750 (0.0360)</td><td>0.0923 (0.1057)</td></tr></table>

## B Visual Ambiguity analysis.

We understand that visual ambiguity is semantically related, and therefore analyze whether low-level VSE relates to semantic regions in pixel space (feature-level analysis is included in Sec. E.3). We identify semantic regions using SAM3 to localize objects mentioned in the questions and answers in AOKVQA, then apply Gaussian noise either in/outside these regions. The AUC of Inside, Outside and Full-image VSE are 79.1,75.2,78.3 respectively. Moreover, full-image VSE strongly correlates with semantic-region VSE (r=0.6), suggesting that full-image VSE are driven by semantic regions. Therefore, although Gaussian noise is low-level, VSE reflects visual ambiguity rather than noise-induced instability.

## C Locality of Text and Images

Image vs. Text Embedding shifts. To examine how different perturbations affect the multimodal representation, we compare the embedding shifts induced by textual paraphrases and image perturbations. For each question, we generate semantically equivalent paraphrases using Gemma3-8B [4] following the prompting strategy of [27]. Image perturbations are produced by adding Gaussian noise with standard deviation 20 to the input image. All perturbed inputs are then encoded into a shared multimodal embedding space using Qwen3-VL-Embedding-8B [9]. For each original image–question pair, we generate eight image perturbations and eight question paraphrases, and compute the cosine distance between their embeddings and the embedding of the original pair. Fig. 3 shows the resulting distance distributions across datasets. Image perturbations consistently induce small embedding shifts, indicating that they explore a local neighborhood around the original query. In contrast, textual paraphrases produce substantially larger shifts, suggesting that paraphrasing moves the representation to distant regions of the embedding space rather than sampling locally around the original conditioning.

![](images/e63781cb4afbb208b30eb08ff80a372a69491543df1819683bd6d13915e2ba4f.jpg)

![](images/c24d69ac400e055fda16c5dd8ba6771ad5bb71203ca14c0f748fa33d34c5d014.jpg)

![](images/674696e113c46b7b9f1530e516ef4c462be95b75b522154455827896d73c4064.jpg)  
Fig. 3: Embedding shifts induced by text and image perturbations. For each image–question pair, we generate semantically equivalent paraphrases and image perturbations, then encode all inputs using the Qwen3-VL-Embedding-8B multimodal embedding model. The cosine distance between each perturbed embedding and the original embedding is measured. Across datasets, textual paraphrases produce substantially larger embedding shifts than image perturbations: 0.063 vs. 0.014 on MMVet, 0.119 vs. 0.029 on VLMs Are Biased, and 0.060 vs. 0.015 on VILP.

Image vs. Text Embedding Space. We further visualize the multimodal embedding space using t-SNE. As shown in Fig. 4, image perturbations remain close to the original embedding, whereas question paraphrases induce substantially larger shifts and disperse across the space.

![](images/808a9e6823d866db677e5c040797f53dddb851cef0bf83af2599862869e9ee09.jpg)  
Fig. 4: Image vs. Text Embedding Space. t-SNE projections of multimodal embeddings for several image–question pairs from AOKVQA dataset. Colors indicate the same original image-question pair, while marker shapes distinguish perturbation types (text vs. image). Image perturbations remain tightly clustered around the original embedding, whereas question paraphrases induce larger shifts and spread to distant regions of the embedding space.

## D Text perturbation drives cluster assignments

Such non-local shifts have important implications for uncertainty estimation. When both image and text are perturbed, outputs produced under different paraphrases are likely to occupy distinct regions of the embedding space, even if the visual information remains unchanged. As a result, clustering would group outputs by paraphrase identity rather than by variations in the visual input. To verify this effect, we further provide qualitative examples of the cluster occupancy patterns on the perturbation grid (Fig. 5). We observe horizontal stripe structures where cluster membership is consistent within a paraphrase but invariant across image perturbations. This pattern confirms that cluster assignments are largely driven by the textual input. Hence, these results show that joint image-text perturbation primarily introduces variability through textual paraphrases rather than visual changes. Consequently, uncertainty estimates derived from such perturbations may reflect prompt-induced variability instead of genuine visual ambiguity.

![](images/e3055d8513dc782021e6298cc61916efb3221a95a37936c03f58e2da786fdd19.jpg)  
Fig. 5: Cluster occupancy map. Each panel shows cluster assignments on the $L \times M$ perturbation grid of a random sample (rows: textual paraphrases; columns: image perturbations). Horizontal stripes indicate invariance across image perturbations, showing that clustering is dominant by textual paraphrase. (AOKVQA dataset, Qwen2.5-VL).

## E Hyperparameters Sensitivity

## E.1 Temperature $T$ and Number of samples $M$

Fig. 6 analyzes the impact of decoding temperature T and the number of generated variants M on uncertainty estimation performance. Increasing M consistently improves AUC for both Qwen2.5-VL and Gemma3, with the largest gains observed from 5 to around 15-20 samples. Performance largely saturates once $M \geq 15$ , despite the linear increase in computational cost, indicating diminishing returns from additional sampling. Higher temperatures ( $T \in [0.8, 1.0]$ ) consistently achieve the best results, while lower temperatures suppress output diversity and limit uncertainty signals. Overall, effective uncertainty estimation benefits from sufficiently stochastic decoding and a moderate number of samples, balancing diversity and computational efficiency. Fig. 7 presents the same ablation in line-plot form for clarity. Increasing the number of variants M and decoding temperature T consistently improves AUC, with gains diminishing beyond $M \geq 15$ .

![](images/66d0d428b8f1a693de9e06bd9d627ea0efc1466ce99a674a54035ef8b6d2d5cd.jpg)

![](images/92ca241118a51db6e792f0dae8897f9b65b3b6e93a3a4203fba5f1129030bd59.jpg)  
Fig. 6: Decoding temperature T and Number of variants M ablation. Performance improves with more variants and moderate temperatures. Results on AOKVQA dataset.

![](images/fea9174556f3b1c73d452144bffb19c4033ef71cc16dd74e393b3e93906c49b2.jpg)

![](images/5bea468a1e4a2600621bb99ac9963c2bc14a4e5744b585be05f8512885848291.jpg)  
Fig. 7: Adequate sampling (M) and high temperature (T) demonstrate improved performance, with gains saturating as M increases. Results on AOKVQA dataset.

![](images/4cf05d55bd9a8b6bfeaf824ac40b6d62f2c19baf100da7dc77183bda50dcec3f.jpg)

![](images/c370b839c264606260615adab8c987588cdfeba0fe54b0410ae3bd3ca2299d92.jpg)  
Fig. 8: Visual perturbation strength $\xi$ ablation. Moderate noise improves performance, while stronger perturbations cause slight degradation. Results on AOKVQA dataset.

## E.2 Augmentation strength $\xi$

We vary the visual perturbation strength from 20 to 500 and observe that moderate noise at 20 yields the best performance, while stronger perturbations slightly reduce AUC (Fig. 8). This falls into the behavior of text-driven perturbations: large textual shifts introduce variability that is not grounded in visual evidence and are therefore harmful. Similarly, overly strong visual perturbations create excessive shifts in the visual space, distorting semantic content rather than revealing genuine ambiguity. In both cases, large shifts degrade uncertainty estimation quality.

Table 2: Visual Perturbation Functions T ablation. Results on AOKVQA dataset.

<table><tr><td>Model</td><td>Affine Transform</td><td>Color Transform</td><td>Gaussian Blur</td><td>Gaussian Noise</td></tr><tr><td>Qwen2.5-VL</td><td>0.759</td><td>0.764</td><td>0.749</td><td>0.783</td></tr><tr><td>Gemma3</td><td>0.742</td><td>0.754</td><td>0.782</td><td>0.778</td></tr></table>

## E.3 Visual Perturbation Functions T

Table 2 compares different visual perturbation functions for uncertainty estimation (AUC) on AOKVQA. We evaluate four augmentation types with controlled magnitudes:

Affine Transform: small geometric perturbations including rotation ( $\pm10^{\circ}$ ), translation (up to 5% of image size), scaling within [0.95, 1.05] ratio, and shear ( $\pm10^{\circ}$ ).

\- Color Transform: photometric adjustments implemented via random brightness, contrast, and saturation scaling sampled from [0.6, 1.4], and hue shift sampled from $[-0.02, 0.02]$ in normalized HSV space.

\- Gaussian Blur: spatial smoothing using a Gaussian kernel with a small blur radius (1% image size) and mild spatial jitter (0.1).

\- Gaussian Noise: additive pixel-wise Gaussian noise with standard deviation $\sigma = 20$ , applied independently to each pixel.

Across both Qwen2.5-VL and Gemma3, Gaussian-based perturbations (noise and blur) achieve stronger AUC compared to affine and color transformations. While the differences are moderate, Gaussian noise yields the best result for Qwen2.5-VL and Gaussian blur performs best for Gemma3. Gaussian-based perturbations (noise and blur) consistently achieve stronger performance across both models. This likely stems from their ability to introduce local, distributed variations that preserve the global semantic structure of the image while probing its immediate neighborhood in representation space. We adopt Gaussian noise as the default perturbation function, as it consistently performs well across models and requires tuning only a single parameter, the noise standard deviation $\sigma$ .

Feature Perturbation We also implemented masked patches and feature-level visual-token perturbations, which perform comparably or slightly higher than Gaussian noise. Results on AOKVQA:

Table 3: Feature Perturbation ablation. Results on AOKVQA dataset.

<table><tr><td>Model</td><td>Gaussian Noise</td><td>Feature Perturbation</td><td>Patch Mask</td></tr><tr><td>Qwen2.5-VL</td><td>0.783</td><td>0.785</td><td>0.793</td></tr><tr><td>Gemma3</td><td>0.778</td><td>0.780</td><td>0.783</td></tr></table>

As magnitudes are not directly comparable, we ran each method over a strength sweep (mask ratio $[10\%, 40\%]$ , noise scale $[2\%, 10\%]$ ) and report best result. We observe that:

(1) VSE remains strong across perturbation types and not tied to low-level perturbation

(2) Gaussian noise achieves competitive performance, being a simple and practical probe.

We note that all compared methods are blackbox without feature access, and VSE is designed under the same constraint. We agree that feature-level analysis is important and will include it in the paper.

## E.4 Distance Function d(,)

Table 4 compares SNNE and VSE under different semantic similarity metrics, including:

\- Cosine distance with all-MiniLM-L6-v2 embedding model [24]

Table 4: Semantic Distance Function d(,) ablation. AUC comparison between SNNE and VSE under different semantic distance function d(,). Results on AOKVQA dataset.

<table><tr><td rowspan="2">Model</td><td colspan="3">SNNE</td><td colspan="3">VSE</td></tr><tr><td>Cosine</td><td>BertScore</td><td>DeBERTa</td><td>Cosine</td><td>BertScore</td><td>DeBERTa</td></tr><tr><td>Qwen2.5-VL</td><td>0.651</td><td> $\underline{0.742}$ </td><td>0.744</td><td>0.685</td><td> $\underline{0.754}$ </td><td>0.783</td></tr><tr><td>Gemma3</td><td>0.614</td><td>0.715</td><td> $\underline{0.700}$ </td><td>0.657</td><td> $\underline{0.720}$ </td><td>0.778</td></tr></table>

- BERTScore [29],  
- DeBERTa-v2-xlarge-mnl [6].

Across both Qwen2.5-VL and Gemma3, VSE consistently outperforms SNNE under stronger semantic metrics. In particular, when using DeBERTa, VSE achieves the highest AUC for both models (0.783 for Qwen2.5-VL and 0.778 for Gemma3). While cosine similarity provides weaker alignment signals, performance improves when adopting contextualized semantic metrics such as BERTScore and DeBERTa. The gains are more pronounced for VSE, suggesting that clustering-based uncertainty estimation benefits from richer semantic representations. Overall, these results indicate that combining VSE with a strong semantic similarity measure yields the most reliable uncertainty estimates.

## F Complexity

Complexity of VSE, SE, and SNNE is $O(M^{2})$ . Inference is dominated by VLM sampling, which are all similar, while the aggregation step is negligible: 0.24/0.34/0.21s.

## G Qualitative Examples

We provide qualitative results where high VSE indicates high uncertainty of wrong answers in Fig. 9; low VSE indicates low uncertainty or confident of correct answers in Fig. 10; and failure cases of VSE in Fig. 11. Percentile indicates the rank of the VSE values within the dataset to reflect a value within a bounded range $[0,1]$ for ease of interpretability. Results are shown for AOKVQA dataset using Qwen2.5-VL model.

GT: wii boxing
Pred: wii baseball ✗

![](images/4dfa94b29b692eebe03ae26ab6c952dd27422d95a183cfa245769ae66ccc492e.jpg)  
What sport game is the man playing?  
wii tennis, wii tennis
wii baseball, Wii baseball,
wii baseball, wii sport baseball
Wii sport boxing., boxing,
wii boxing, Wii boxing.

VSE: percentile:
0.638 0.973 ( $\uparrow$ )

![](images/1391156875251ac700d2cae48a4d737c099e51fe7efd0511e206767a0686415f.jpg)  
The name of the street is the same as the last name of what actress?

GT:    Pred:  
Jessica Biel    Eva Green ✗

![](images/eb1d5989d4c6cd4636a8245118dcca367a5ead5f7ffc7ccd91477a7f85d9e6d7.jpg)  
What is the zebra in the center doing?

Sampled answers:  
Eva Mendes, Eva Mendes, Eva Mendes  
eva green, Eva Green.,  
eva green, Eva Green  
Jessica Alba  
Images only has urban traffic scene.  
Jessica Biel.

VSE: percentile:
0.696 0.995 ( $\uparrow$ )

GT: Pred: running standing

Sampled answers:
sleeping
Running, running, Running,
running, running, running
eating
looking around
Activity likely grazing.

VSE: percentile:
0.536 0.908 ( $\uparrow$ )

Fig. 9: Qualitative Results. A high VSE indicates that the model is uncertain about its original prediction, especially when that prediction is incorrect.

![](images/4eaf6c7233f639423277f9abc837a6c1dd34393569c18498c7b78d272ba9fc9c.jpg)  
Fig. 10: Qualitative Results. A low VSE indicates that the model is certain about its original prediction, especially when that prediction is correct.

![](images/b0ca23f879b7c3a965eda20c1c481d592ee53f9439c215bedcd2a2894c73943f.jpg)

![](images/776c6f4325ac0ccdf5e4b39e34c6299bcbca529b1d8d13572a5c026643a47215.jpg)  
The reddish-brown food in the further bowl is what type of food?

![](images/6312578bc2a678cbaf4cf11fac6bd4d3e5798acdd92020049e5931f03447a9eb.jpg)  
What is the mascot of the team whose website address in the background?

![](images/00193616f2f7e8cfeb220965f585409afd2560a431ac352f07335e9eb2bcdfc5.jpg)  
Fig. 11: Qualitative Results. Failure cases where VSE is low for incorrect answer (top) and high for correct answer (bottom). In the bottom example, the visual evidence is highly ambiguous, suggesting that the model arrives at the correct answer largely by chance.

## References

1. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025) 12

2. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025), https://arxiv.org/abs/2502.13923, computer Vision and Pattern Recognition (cs.CV) 12

3. Farquhar, S., Kossen, J., Kuhn, L., Gal, Y.: Detecting hallucinations in large language models using semantic entropy. Nature 630(8017), 625–630 (2024) 2, 4, 11, 12, 13, 14, 15

4. Gemma Team: Gemma 3 technical report. arXiv preprint arXiv:2503.19786 (2025), https://arxiv.org/abs/2503.19786 12, 18

5. Groot, T., Valdenegro-Toro, M.: Overconfidence is key: Verbalized uncertainty evaluation in large language and vision-language models. In: Proceedings of the 4th Workshop on Trustworthy Natural Language Processing (TrustNLP 2024). pp. 145–171 (2024) 1, 4, 12, 13, 14

6. He, P., Liu, X., Gao, J., Chen, W.: Deberta: Decoding-enhanced bert with disentangled attention. arXiv preprint arXiv:2006.03654 (2020) 12, 24

7. Huang, J., Xu, J., Shi, X., Hu, P., Feng, L., Zhu, X.: Revisiting confidence calibration for misclassification detection in vlms. In: The Fourteenth International Conference on Learning Representations 2, 4

8. Khan, Z., Fu, Y.: Consistency and uncertainty: Identifying unreliable responses from black-box vision-language models for selective visual question answering. In: Proceedings of the ieee/cvf conference on computer vision and pattern recognition. pp. 10854–10863 (2024) 2, 3, 4, 6, 12, 13, 14

9. Li, M., Zhang, Y., Long, D., Chen, K., Song, S., Bai, S., Yang, Z., Xie, P., Yang, A., Liu, D., et al.: Qwen3-vl-embedding and qwen3-vl-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720 (2026) 3, 18

10. Li, Q., Geng, J., Lyu, C., Zhu, D., Panov, M., Karray, F.: Reference-free hallucination detection for large vision-language models. In: Findings of the Association for Computational Linguistics: EMNLP 2024. pp. 4542–4551 (2024) 2, 4, 12, 13, 14

11. Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., Lee, Y.J.: Llava-next: Improved reasoning, ocr, and world knowledge (January 2024), https://llava-vl.github.io/blog/2024-01-30-llava-next/12

12. Luo, T., Cao, A., Lee, G., Johnson, J., Lee, H.: Probing visual language priors in vlms. arXiv preprint arXiv:2501.00569 (2024) 5, 8, 11

13. Manakul, P., Liusie, A., Gales, M.: Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. In: Proceedings of the 2023 conference on empirical methods in natural language processing. pp. 9004–9017 (2023) 2, 4, 12, 13, 14

14. Marino, K., Rastegari, M., Farhadi, A., Mottaghi, R.: Ok-vqa: A visual question answering benchmark requiring external knowledge. In: Proceedings of the IEEE/cvf conference on computer vision and pattern recognition. pp. 3195–3204 (2019) 11

15. Nguyen, D., Payani, A., Mirzasoleiman, B.: Beyond semantic entropy: Boosting llm uncertainty quantification with pairwise semantic similarity. In: Findings of the Association for Computational Linguistics: ACL 2025. pp. 4530–4540 (2025) 2, 3, 4, 8, 11, 12, 13, 14, 15

16. Nielsen, F.: Hierarchical clustering. In: Introduction to HPC with MPI for Data Science, pp. 195–211. Springer (2016) 12

17. Nikitin, A., Kossen, J., Gal, Y., Marttinen, P.: Kernel language entropy: Fine-grained uncertainty quantification for llms from semantic similarities. Advances in Neural Information Processing Systems 37, 8901–8929 (2024) 2, 4, 8, 12, 13, 14

18. nostalgebraist: interpreting gpt: the logit lens. LessWrong post (Aug 31 2020), https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens 5, 6, 17

19. Pelucchi, M.: Exploring ChatGPT's accuracy and confidence in high-resource languages. Ph.D. thesis (2023) 1, 4

20. Schwenk, D., Khandelwal, A., Clark, C., Marino, K., Mottaghi, R.: A-okvqa: A benchmark for visual question answering using world knowledge. In: European conference on computer vision. pp. 146–162. Springer (2022) 3, 11

21. Valdenegro-Toro, M.: I find your lack of uncertainty in computer vision disturbing. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1263–1272 (2021) 1, 4

22. Vo, A., Nguyen, K.N., Taesiri, M.R., Dang, V.T., Nguyen, A.T., Kim, D.: Vision language models are biased. arXiv preprint arXiv:2505.23941 (2025) 11

23. Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al.: Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265 (2025) 12

24. Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., Zhou, M.: Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. Advances in neural information processing systems 33, 5776–5788 (2020) 23

25. Xuan, W., Zeng, Q., Qi, H., Wang, J., Yokoya, N.: Seeing is believing, but how much? a comprehensive analysis of verbalized calibration in vision-language models. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 1408–1450 (2025) 1, 4

26. Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., Wang, L.: Mm-vet: Evaluating large multimodal models for integrated capabilities. In: International conference on machine learning. PMLR (2024) 11

27. Zhang, R., Zhang, H., Zheng, Z.: Vl-uncertainty: Detecting hallucination in large vision-language model via uncertainty estimation. arXiv preprint arXiv:2411.11919 (2024) 2, 3, 4, 6, 10, 12, 13, 14, 18

28. Zhang, S., Sambara, S., Banerjee, O., Acosta, J., Fahrner, L.J., Rajpurkar, P.: Radflag: A black-box hallucination detection method for medical vision language models. arXiv preprint arXiv:2411.00299 (2024) 2

29. Zhang, T., Kishore, V., Wu, F., Weinberger, K.Q., Artzi, Y.: Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675 (2019) 24