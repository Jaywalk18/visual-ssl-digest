# Dive into the implicit biases of low-rank vision-language alignment

Mingjia Shi Shuo Wang<sup>1‡</sup> Xiaobo Wang<sup>2†</sup> Sifan Zhou<sup>1</sup> Kai Wang<sup>3</sup> Tianyu Fu<sup>1</sup> Chenxu Zhao<sup>1†</sup> Anyang Su<sup>1</sup> Ping Jiang<sup>1</sup> Minghui Wu<sup>1</sup>

<sup>1</sup> Mininglamp g

2 Shenzhen University of Advanced Technology <sup>3</sup> National University of Singapore {wangshuo.e,zhaochenxu}@mininglamp.com, 3101ihs@gmail.com, wangxiaobo@suat-sz.edu.cn

Abstract. Vision-language alignment, the stage that bridges pretrained vision encoders and large language models, is widely treated as a form of pretraining requiring full-parameter updates. We challenge this view and investigate what happens when low-rank adaptation is applied to the LLM during this stage instead. We find that low-rank alignment not only reduces computational costs but also outperforms full-parameter alignment on most benchmarks. To understand this phenomenon, we systematically characterize the implicit biases introduced by low-rank adaptation during alignment. Empirically, we find that low-rank alignment shifts model behavior from hallucinatory to conservative and preserves per-token linear separability of visual features that full-parameter alignment disrupts, a phenomenon we term LS-curse. Geometrically, lowrank aligned models exhibit more homogeneous and structurally stable visual representations, maintaining modality-specific knowledge rather than prematurely fusing entity-level semantics. Theoretically, we establish two theorems showing that low-rank alignment induces preferences for parameter subspaces with flat gradients and feature subspaces robust to perturbations, providing a principled explanation for the observed structure-preserving behavior. Extensive experiments cover ablation over 100 alignment configurations, three families of low-rank operators, and various rank, encoder, and other settings.

## 1 Introduction

The construction of vision-language models (VLMs) hinges on a critical stage: vision-language alignment, where pretrained vision encoders are bridged with large language models (LLMs) through adapter modules and joint training [5, 8, 25]. As illustrated in Figure 1, this stage precedes instruction tuning and establishes the cross-modal feature correspondence upon which all downstream capabilities depend. Standard practice updates the full LLM parameters during alignment, treating it as a form of pretraining. However, this view constitutes a misconception: unlike LLM pretraining, which builds linguistic representations from scratch, vision-language alignment operates atop pretrained knowledge and reasoning priors: it is, in fact, supervised fine-tuning, precisely the regime for which low-rank adaptation methods were designed [10, 12, 14, 55]. Yet, applying low-rank methods during this stage remains largely unexplored.

![](images/9c2d5a48ae4080593f5200ad47085c17608eb091e1963fd1f4344ebd07914669.jpg)  
Fig. 1: Vision-language alignment framework (default): a minimum achievable VLM with a cross-modal adapter (MLP). How does low-rank alignment shape visual tokens?.

When low-rank adaptation is applied to the LLM during alignment (i.e., low-rank alignment), we observe that not only does the training cost drop substantially, but downstream performance empirically improves across all involved model scales (from 1.4B to 14B). This improvement holds across diverse benchmarks spanning perception, knowledge, reasoning, and hallucination, and generalizes across three families of low-rank operators (LoRA, LoHa, LoKr). This finding raises a natural question: how do the low-rank methods reshape the visual features to yield superior representations, and technically, what implicit biases does the low-rank alignment introduce?

Empirical observations. We investigate this question first at the behavior and feature levels. At the behavior level, hypothesis testing on visual perception benchmarks reveals that low-rank alignment shifts model decision-making from hallucinatory to conservative, reducing overconfident predictions while preserving cautious uncertainty (§2.2). At the feature level, token-wise linear probes uncover that full-parameter alignment disrupts the linear separability of certain visual tokens, a phenomenon we term LS-curse, whereas low-rank alignment preserves the per-token knowledge structure inherited from the pretrained vision tower (§2.3). Together, these observations suggest that the performance gain stems from low-rank alignment’s stronger preservation of modality-specific knowledge structures, rather than the premature entity-level mixing induced by full-parameter updates.

Geometric characterization. We further characterize these biases through geometric analysis of the visual feature space. Fine-grained and coarse-grained linear probes reveal that fully aligned models exhibit greater feature diversity and broader angular coverage across tokens, reflecting aggressive entity-level fusion. In contrast, low-rank aligned models maintain homogeneous and structurally conservative representations. Manifold visualizations corroborate this finding: per-token feature distributions under low-rank alignment are markedly more consistent than those under full alignment (§2.4).

Theoretical analysis. We establish two theorems characterizing the mathematical mechanisms underlying these biases. Theorem 1 reveals that the low-rank gradient flow exhibits a noise-weighted smoothing efect, preferentially reinforcing updates along feature directions robust to stochastic variations, encoding an implicit bias toward flatter regions of the loss landscape. Theorem 2 demonstrates that the steady-state distribution of low-rank parameters concentrates on subspaces with flat gradients and noise-robust features. These two results jointly explain why low-rank alignment preserves general knowledge structures instead of prematurely fusing entity-specific characteristics (§3).

We conduct extensive experiments across over 55 vision-language alignment settings (unique combinations of model, vision encoder, low-rank operator, rank, vision-tower depth, and learning-rate schedule) spanning model architectures from MobileLLaMA-1.4B to Qwen3-14B and three families of low-rank operators, which together yield over 100 configurations when evaluated across benchmark categories. We summarize our contributions as follows:

We demonstrate that low-rank adaptation during vision-language alignment not only improves eficiency but also consistently enhances downstream performance across all evaluated model scales (1.4B–14B), challenging the prevailing assumption that alignment requires full-parameter updates.

We provide a systematic empirical and geometric characterization of the implicit biases introduced by low-rank alignment, identifying the LS-curse in full alignment and revealing how low-rank methods preserve modalityspecific knowledge structures, collectively supporting a progressive fusion view in which general visual representations are retained during alignment and entity-specific fusion is deferred to instruction tuning.

– We establish two theorems identifying the theoretical mechanisms: noiserobust feature preferences during optimization (Theorem 1), and flat-gradient subspace concentration at steady state (Theorem 2), providing a principled explanation for the structure-preserving behavior observed empirically.

– We conduct comprehensive ablations covering rank, operators, model scales, architectures, encoder generalization (CLIP, DINOv2, SigLIPv2) and other details, providing practical guidance for eficient multimodal training.

## 2 Empirical Study: Low-Rank Alignment’s Preferences

Training protocol. As illustrated in Figure 1, during the vision-language alignment stage, we employ perceptual question answering (QA) tasks with general prompting templates. Trainable parameters comprise the LLM backbone and the vision-tower adapter; the vision encoder is frozen by default. For low-rank alignment, we apply adaptation exclusively to all linear modules in the LLM, since the adapters are already lightweight; imposing low-rank constraints on them would force all vision features into a

![](images/da095afb919d061cfd68a66f80383615a6ce1189d816706d80ca1c33248d98f1.jpg)  
Fig. 2: Operators for low-rank alignment: LoRA/LoHa/LoKr (Matrix, Hadamard and Kronecker Products).

low-rank subspace. Both full-parameter and low-rank baselines are tuned under matched hyperparameter budgets to ensure fair comparison (best out of 9 times). For behavioral evaluation, instruction tuning is necessary to ensure proper instruction following. To assess zero-shot capabilities, we divide instruction tuning into two phases: initial tuning with a base visual question answering (VQA) instruction dataset, followed by post-training with an extended instruction dataset covering task-specific and knowledge-intensive data. All benchmarks except GQA and TextVQA are treated as unseen tasks. Extended ablations on encoder choice, frozen vs. unfrozen settings, low-rank placement, learning rate schedules, and overfitting controls are provided in §4.

Low-rank operators. We focus on linear transformations, specifically LoRA [14], LoHa [17], and LoKr [51], which encompass the three principal product-based constructions for low-rank modules: matrix multiplication, Hadamard product, and Kronecker product. The formulations and operator details are in Figure 2.

Linear probe. The procedure extracts features from a classification dataset and trains a linear classifier on them to measure accuracy. This approach enables us to assess both the robustness of the corresponding knowledge and the degree of feature entanglement.

Resources. We conduct experiments across a range of language models spanning compact to commonly used scales, including MobileLLaMA-1.4/2.7B [8], Vicuna-7/13B [7], and Qwen3-8/14B<sup>4</sup> [49]. The vision tower is CLIP-Vision [38]. For vision-language alignment, we utilize datasets including LAION [42], CC [43], SBU [37], SAM [39], and MS-COCO [24]. Instruction tuning leverages default instruction datasets, including LLaVA [25], ShareGPT4V-PT [5], GQA [15], MS-COCO-train-2017, OCRVQA [34], TextVQA [44], and VG [20], along with extended instruction datasets: SBU, ScienceQA [30], MS-COCO, GQA, IconQA [31], SAM, ShareGPT4V-TextVQA [5], Web-Celebrity [29], Web-Landmark [8], and WikiArt [16]. Evaluation is performed on a diverse set of benchmarks and tasks, including GQA, MMBench [28], MME [52], MMMU-Pro-Vision [54], MMVetgpt [53], POPE [22], ScienceQA, and TextVQA.

Default settings. Since this work focuses on the alignment stage, we minimize the VLM pipeline to its essential components, deliberately removing unnecessary architectural tricks to facilitate clean ablation and isolate alignment-stage efects. Unless otherwise specified, instruction fine-tuning uses GQA, TextVQA, and OCRVQA as task-specific datasets, and LoRA with rank 128 by default. For clarity in reporting, we categorize the main benchmarks into four groups: perception, knowledge, reasoning, and hallucination, corresponding to MMEperception, GQA, MME-reasoning, and POPE, respectively. For simplicity, we refer to models by their size rather than their full names. Default visualizations and fine-grained analyses are performed using the 1.4B model to reduce the computational and storage overhead associated with complex processing $( e . g .$ 5 feature extraction and dimensionality reduction).

Table 1: Performance of low-rank vs. full-parameter alignment across model scales (1.4B–13B). Low-rank results report the best configuration across LoRA, LoHa, and LoKr after three rounds of rank tuning. Metrics cover perception (MME-P), knowledge (GQA), reasoning (MME-R), and hallucination (POPE). Training time on 8×A100.

<table><tr><td>size</td><td>perception↑</td><td>Δ ↑</td><td>knwldg.↑</td><td>Δ ↑</td><td>reason↑</td><td>Δ ↑</td><td>hallu.↑</td><td>Δ ↑</td><td>time (h) ↓</td><td>-Δ ↑</td></tr><tr><td>1.4B</td><td>208.1[453.5]</td><td>245.4</td><td>18.9[28.0]</td><td>9.1</td><td>22.9[766.6]</td><td>743.7</td><td>2.5[72.8]</td><td>70.3</td><td>4.53[2.52]</td><td>2.01</td></tr><tr><td>2.7B</td><td>15.7[669.0]</td><td>653.3</td><td>11.6[42.4]</td><td>30.8</td><td>40.4[201.4]</td><td>161.0</td><td>0.5[76.4]</td><td>75.9</td><td>8.61[4.29]</td><td>4.32</td></tr><tr><td>7B</td><td>521.0[965.3]</td><td>444.3</td><td>23.4[38.5]</td><td>15.1</td><td>86.8[332.5]</td><td>245.7</td><td>35.0[80.4]</td><td>45.4</td><td>19.75[7.68]</td><td>12.07</td></tr><tr><td>13B</td><td>236.6[1079.7]</td><td>843.1</td><td>29.5[43.0]</td><td>13.5</td><td>52.5[219.6]</td><td>167.1</td><td>66.9[82.9]</td><td>16.0</td><td>35.32[13.23]</td><td>21.09</td></tr></table>

## 2.1 Pre-Experiments: Performance of Low-Rank Alignment

We evaluate models aligned via low-rank methods under the default instructiontuning setting described above. Table 1 demonstrates that low-rank visionlanguage alignment substantially enhances final model performance across all scales, even when implemented solely through low-rank branches prior to instruction tuning. It raises the question: why and how does low-rank alignment boost performance? We then address this question through both experimental and theoretical analyzes of the mechanism of low-rank alignment.

## 2.2 Behavioral Preferences: From Hallucinatory to Conservative

We first examine whether low-rank alignment alters model decision-making behavior. To this end, we evaluate aligned models on a yes-or-no visual QA dataset and measure Type I and Type II error rates after instruction tuning. In this context, a Type I error occurs when the model incorrectly answers “yes” to a negative-ground-truth question, indicating an overconfident or hallucinated response. A Type II error occurs when the model incorrectly answers “no” to a positive-ground-truth question, reflecting cautious behavior that defaults to rejection under uncertainty. At the sentence level, such cautiousness manifests as increased perplexity in the generated answers.

Figure 3 demonstrates that low-rank adaptation shifts model behavior from halluci natory to conservative. At the 1.4B and 7B scales, LoHa substantially reduces the Type I error rate compared to full fine-tuning, efectively reversing the model’s behavioral tendency from overconfident afirmation toward cautious

![](images/61ad44ac3bda02f075f308195c07f289018a049398405f318f00a830cf34a4fe.jpg)  
Fig. 3: Type -I and -II error rate: models and lowrank operators. The results are on $\mathrm { P O P E , }$ a visual perception hallucination benchmark.

rejection. Through grid search over rank values, we find that certain low-rank operators, such as LoHa, can simultaneously reduce both Type I and Type II error rates without inducing a mere trade-of between the two failure modes. We use conservative to denote a reduced Type I (hallucination) rate and decisive to denote that this reduction is not obtained at the cost of a higher Type II rate, i.e., both error modes decrease jointly rather than trading of. We note that this shift is context-dependent: at the smallest (1.4B) scale the gap narrows, since an under-capacity model already tends toward conservative default decisions, and conservatism is not universally beneficial: settings that require afirmative identification may favor less cautious behavior.

Obs.1 Low-rank alignment is associated with conservative/decisive behaviors.

## 2.3 Feature Preferences: Linear Separability Preservation

We next investigate how the visual features (i.e., the LLM’s visual inputs) are shaped by alignment before multimodal fusion takes place. We employ ImageNet [41] as the test dataset to evaluate general vision capabilities.

We measure per-token linear separability (LS), which assesses whether each visual token independently retains a linearly separable knowledge structure. Concretely, for each token position t in the visual sequence, we train a separate linear classifier on the corresponding C-dimensional feature vectors across all samples and report classification accuracy. A high

![](images/3d44b9010ac56bd7604203c38abba099b2e96eb9b3d4592060829abe98a1f86e.jpg)  
Fig. 4: Broken per-token linear separability of visual tokens, marked in Figure 1, tested by the default settings on ImageNet.

per-token LS indicates that the knowledge encoded at that position remains well-structured and discriminative; a low value suggests that the token’s representation has been entangled with information from other tokens or modalities.

The original vision tower is well-trained and exhibits near-perfect token-wise linear separability. Figure 4 reveals that in a low-level perception setting (i.e., a 1.4B model with full or low-rank alignment using a 2-layer vision-language connector), the vision tokens of aligned models before instruction tuning no longer maintain full linear separability on ImageNet. For the low-rank aligned model (LoRA with rank 128), individual tokens remain largely linearly separable, whereas full-parameter alignment disrupts this property in a subset of tokens, a phenomenon we term LS-curse: certain tokens lose their discriminative structure entirely, with accuracy dropping from near-perfect to as low as 83.6%. As shown in the following table, while some fully aligned tokens retain the knowledge structure from generic image datasets, others exhibit a mixing of representations that destroys linear separability.

<table><tr><td></td><td>origin</td><td>full</td><td>low-rank</td></tr><tr><td>avg. acc. ± std. (%) [min]</td><td>100±0.00</td><td>99.52±1.89 [83.61]</td><td>99.99±0.02</td></tr></table>

This mixing of partial representations is not necessarily detrimental; it can introduce beneficial diversity across token representations. For instance, adversarial samples with similar representations may align semantically, thus appearing non-adversarial in certain tasks or contexts. However, it reflects a fundamentally diferent alignment strategy: full-parameter alignment aggressively restructures the visual feature space, whereas low-rank alignment operates conservatively within a constrained subspace.

## Obs.2 Low-rank alignment preserves per-token linear separability.

Interpretation. The combination of Obs. 1 and 2 reveals a coherent picture of the implicit biases introduced by low-rank alignment. At the behavior level, lowrank aligned models exhibit greater conservatism and reduced susceptibility to hallucination. At the feature level, this conservatism is grounded in the preservation of modality-specific knowledge structures: low-rank alignment maintains the per-token linear separability inherited from the pretrained vision tower, whereas full alignment disrupts this structure through aggressive entity-level mixing. The subspace-constrained nature of low-rank updates limits token-wise diversity, preserving commonalities within each modality and preventing premature fusion of entity-level characteristics, while adjusting the subspace that most strongly affects the majority of features to maximize alignment loss reduction.

<table><tr><td></td><td>full</td><td>low-rank</td></tr><tr><td>(behavior) decisiveness</td><td>hallucinatory</td><td>conservative</td></tr><tr><td>(feature) preference</td><td>entity characteristics</td><td>modal commonality</td></tr></table>

## 2.4 Geometric Characterization

To further characterize the structural diferences between full and low-rank alignment, we conduct two token-level geometric analyses: (1) testing each token’s linear separability on diverse, unseen fine-grained classification datasets (e.g., birds and dogs) to disentangle general from entity-specific knowledge; and (2) examining the angular coverage and manifold structure of token features on general coarse-grained datasets (e.g., ImageNet).

General vs. specific knowledge. If low-rank alignment preserves general knowledge structures without prematurely fusing entity-specific characteristics, the two alignment types should yield comparable per-token LS on fine-grained entity categories while difering substantially on coarse-grained, general-purpose datasets. Table 2 confirms this prediction. On both STF-dogs [19] and CUB [47] (fine-grained datasets targeting dogs and birds, respectively), the fully aligned model exhibits high peak accuracy (e.g., 95.7% on STF-dogs, 94.0% on CUB) but also pronounced inter-token variability, with the worst tokens degrading significantly. The low-rank aligned model, by contrast, maintains narrower and more uniform LS distributions: it trades peak fine-grained separability for consistent preservation of coarse, foundational representations across all tokens.

Table 2: Linear probe results across datasets of varying knowledge granularity. The fully aligned (full) vision tower’s features are more diverse (wider range of linear probe acc. on fine-grained datasets). ImageNet is general classification dataset. STF-dogs and CUB are fine-grained classification dataset of dogs and birds. Ranks of LoRA are $3 2 / 6 4 / 1 2 8 .$ Range is reported by the maximum and minimum.

<table><tr><td>dataset</td><td>fine-grained</td><td>full</td><td>rank=128</td><td>rank=64</td><td>rank=32</td></tr><tr><td>ImageNet</td><td>×</td><td>83.6~100.0</td><td>99.9~100.0</td><td>99.8~100.0</td><td>99.8~100.0</td></tr><tr><td>STF-dogs</td><td>√</td><td>32.8~95.7</td><td>39.2~88.0</td><td>36.3~85.9</td><td>36.7~86.8</td></tr><tr><td>CUB</td><td>√</td><td>49.9~94.0</td><td>61.2~91.7</td><td>60.8~91.5</td><td>61.6~91.5</td></tr></table>

This asymmetry directly sup ports the progressive fusion hypothesis: low-rank alignment retains the general knowl edge backbone while deferring entity-specific fusion to subsequent tuning stages.

![](images/eb0c341d937d2842e0f420f4b50acec98fc91939285bb5ac363278fbd93e9dda.jpg)

Representation geometry. We quantify how alignment reshapes the angular structure of the visual feature space. For each entity sampled from ImageNet, we compute the minimum pairwise cosine angle among its corresponding

Fig. 5: Full aligned vision tokens have larger coverage of each given input images in ImageNet. Each image’s feature angle coverage (i.e., minimal cosine $\begin{array} { r } { \downarrow : = \operatorname* { m i n } _ { i , j } \cos ( i , j ) , i , j \in \dag } \end{array}$ {visualtokens|images}).

visual tokens; a smaller angle indicates that tokens have spread farther apart on the hyperspherical surface. As shown in Figure 5, the horizontal axis tracks alignment progress and the vertical axis reports this angular coverage. The fully aligned model covers a progressively wider angular span, at times exceeding half of the hyperspherical surface, indicating aggressive spread and fusion of token representations. The low-rank aligned model, by contrast, maintains a compact angular footprint throughout, preserving the original structural layout of the visual features.

Feature manifold. Figure 6 visualizes the token-wise feature manifolds (where colors represent diferent tokens), corroborating the angular analysis above. Under full alignment, per-token manifolds are geometrically diverse and spatially dispersed across the feature space, reflecting aggressive restructuring of visual representations. Under low-rank alignment, token clusters are markedly more homogeneous in shape and collectively occupy a more compact region, confirming that the low-rank constraint induces a structural bias toward conservative, stable representations. This geometric regularity is consistent with updates confined to a dominant low-rank subspace, preserving the spatial organization inherited from the pre-trained encoder rather than aggressively overwriting it.

Empirical summary. Lowrank alignment introduces systematic implicit biases that manifest at multiple levels. At the behavior level, it produces conservative and decisive perception. At the feature level, it preserves per-token knowledge structure and prevents the LScurse. Geometrically, it maintains homogeneous and structurally stable visual representations while preventing the

![](images/5fb172418d9b24dad95893e56d0619f886f7155c372de24df06ab7cfe86b8751.jpg)  
Fig. 6: The feature manifolds of each visual token, as depicted in Figure 1, are diverse under full alignment and homogeneous under low-rank alignment (LoRA, rank 128). The surface is fitted using a polynomial, retaining first- and second-order information. Each colored manifold represents the token feature distribution on MS-COCO.

premature spread and overgeneralization of entity-level features observed in full alignment. These converging observations motivate the theoretical analysis in Section 3, where we identify the mathematical mechanisms underlying these biases.

## 3 Theoretical Study: Preferences on Parameter and Representation Spaces, Flatness and Robustness

The empirical study in Section 2 reveals that low-rank alignment exhibits systematic implicit biases: conservative behavior, preservation of per-token knowledge structure, and geometrically stable visual representations. Theoretical analysis is provided to identify the suficient conditions under which such patterns arise and derive their mathematical characterizations. Due to the presence of higherorder tensors beyond second order in our derivations, we adopt the Einstein summation convention by default, wherein duplicated indices imply summation.

Analysis model: unconstrained features. We adopt the Unconstrained Feature Model [35] (UFM) as our analytical framework. UFM assumes that an overparameterized neural network can fit any feature, thereby treating features H as free variables rather than fixed outputs of a specific architecture. This enables us to study global properties of the learned feature structure, such as which directions are preferentially reinforced. In practice, this assumption is well-supported by the over-parameterized nature of modern LLMs used in the alignment stage.

Low-rank modeling. Consider a low-rank adaptation parameterized by θ with forward mapping $O ( \theta , H )$ and total loss

$$
\begin{array}{c} F (\theta) = \mathbb {E} _ {(x, y) \sim \mathcal {D}} \left[ \mathcal {L} (O (\theta , H (x)), y) \right], \\ O ^ {i} = W _ {0} ^ {i} _ {j} H ^ {j} + b ^ {i} + l ^ {i} (H; \theta), \end{array}
$$

where $O ^ { i } \in \mathbb { R } ^ { p }$ and $H ^ { j } \in \mathbb { R } ^ { q }$ denote the output and input features indexed by i and $j ; W _ { 0 j } ^ { i } \in \mathbb { R } ^ { p \times q }$ and $b ^ { i } \in \mathbb { R } ^ { p }$ are the frozen base weight and bias; $l ^ { i } ( H ; \theta )$

is the low-rank adaptation term; D is the training dataset; and $\mathcal { L }$ is the loss function. For LoRA, $l ^ { i } ( H ; \theta ) = B _ { k } ^ { i } A _ { i } ^ { k } H ^ { j }$ with low-rank factors $A ^ { k } { } _ { j } \in \mathbb { R } ^ { r \times q }$ 2 $B ^ { i } { } _ { k } \in \mathbb { R } ^ { p \times { r } }$ , and $\theta = \{ B , A \}$ . Analogous parameterizations for LoHa and LoKr, involving Hadamard factor pairs $B _ { 1 } , B _ { 2 } \in \mathbb { R } ^ { p \times r } , A _ { 1 } , A _ { 2 } \in \mathbb { R } ^ { r \times q }$ and Kronecker factor $C ^ { s } { } _ { t } \in \mathbb { R } ^ { s \times t }$ <sup>t</sup>, are detailed in the Appendix.

## 3.1 Process Preference: Robustness to Feature Noise

Theorem 1 (Feature preferences: noise smoothing.). Under the UFM assumption, the feature space with additive noise exhibits a weighted smoothing efect. Expanding features with zero-mean noise $\xi ,$ we have

$$
\mathbb {E} _ {\xi} \left[ \frac {\partial F}{\partial W _ {0 j} ^ {i}} \right] \simeq \frac {\partial F}{\partial W _ {0 j} ^ {i}} (\bar {H}) + \frac {1}{2} \frac {\partial^ {3} F}{\partial W _ {0 j} ^ {i} \partial H ^ {k} \partial H ^ {l}} \Sigma^ {k l},
$$

where $\Sigma ^ { k l } = \mathbb { E } [ \xi ^ { k } \xi ^ { l } ]$ is the feature noise covariance, F is the alignment loss function, $W _ { 0 }$ is the original weight, H<sup>¯</sup> the mean of $H : = \bar { H } + \xi$

Corollary 1 (LoRA’s<sup>5</sup> noise robustness preference .). Substituting this into the LoRA gradient flow yields

$$
\dot {A} ^ {p} _ {j} \simeq - \eta B ^ {i} _ {p} \left(\frac {\partial F}{\partial W _ {0 j} ^ {i}} (\bar {H}) + \frac {1}{2} \frac {\partial^ {3} F}{\partial W _ {0 j} ^ {i} \partial H ^ {k} \partial H ^ {l}} \Sigma^ {k l}\right),
$$

$$
\dot {B} _ {p} ^ {i} \simeq - \eta \left(\frac {\partial F}{\partial W _ {0 j} ^ {i}} (\bar {H}) + \frac {1}{2} \frac {\partial^ {3} F}{\partial W _ {0 j} ^ {i} \partial H ^ {k} \partial H ^ {l}} \Sigma^ {k l}\right) A _ {j} ^ {p},
$$

demonstrating an implicit smoothing efect along directions with feature noise. This shows that gradient flow preferentially aligns updates along subspaces that are robust to stochastic variations in the features, encoding an implicit bias towards flatter directions in the loss landscape.

Intuition and explanation. The core insight lies in the coupled gradient flow: $\dot { A } \sim B \cdot ( \mathrm { g r a d } + \mathrm { n o i s e } )$ and $\dot { B } \sim ( \mathrm { g r a d } + \mathrm { n o i s e } ) \cdot A .$ , where A<sup>˙</sup> and $\dot { B }$ denote the gradient flows of A and B, respectively, and ∗ and ⊗ denote the Hadamard and Kronecker products where applicable. Since updates to A are scaled by B and vice versa, whichever directions are already dominant receive proportionally larger updates, a self-reinforcing dynamic. The noise-induced third-order term further amplifies this efect along directions with large feature-noise covariance, creating a Matthew efect: strong directions grow stronger while weak directions are suppressed. Consequently, LoRA naturally concentrates updates on noiserobust principal directions in the feature space.

Comparison: full vs. low-rank alignment. In full alignment, the weight matrix W is updated directly and all directions receive gradient updates on an equal footing. In low-rank alignment, A and B act as mutual subspace projectors, confining the efect of third-order smoothing terms to the principal subspace. This confinement favors conservative feature integration along dominant directions while suppressing changes in weaker subspaces, helping to explain the homogeneous manifold structure and preserved linear separability observed in Section 2.3. Here $\delta _ { i } ^ { j }$ denotes the Kronecker delta arising in the projection identities. Analogous conclusions hold for LoHa and LoKr as detailed in the Appendix.

## 3.2 Steady-State Preferences: Flatness and Robustness

Theorem 1 characterizes which directions are reinforced during optimization. We now ask a complementary question: where do the low-rank parameters converge at the end of training?

Theorem 2 (Low-rank steady-state distribution). Let the gradient flow of θ be $\begin{array} { r } { \dot { \theta } ^ { \alpha } = - \dot { \eta } \frac { \partial F } { \partial \theta ^ { \alpha } } } \end{array}$ . Define the efective low-rank subspace mapping $M H = l ( H )$ (for example, $\check { M } = B A$ in LoRA, $M = B _ { 1 } A _ { 1 } * B _ { 2 } A _ { 2 }$ in LoHa, $M = C \otimes B A$ in $\begin{array} { r l } { { L o K r } ) } \end{array}$ and the corresponding efective loss:

$$
F _ {\mathrm{eff}} (M) = F (M) + \frac {\eta \log \det \widetilde {D} (M)}{4} + \frac {\partial^ {2} F}{2 \partial O ^ {i} \partial O ^ {m}} \Sigma^ {j k} (W _ {0 j} ^ {i} + M _ {j} ^ {i}) (W _ {0 k} ^ {m} + M _ {k} ^ {m}),
$$

where $F ( M )$ is the original loss expressed in the low-rank subspace $M ; \widetilde { \cal D } ( M )$ is the covariance of gradient noise projected onto M, encoding flatness preference; $\textstyle \sum ^ { j k }$ is the covariance of feature noise. Then, under small-step stochastic gradient descent or noisy gradient flow, the steady-state M distributes approximately as:

$$
p _ {\mathrm{stat}} (M) \propto \exp \Big (- \frac {2}{\eta} F _ {\mathrm{eff}} (M) \Big).
$$

Remark 1. The efective loss $F _ { \mathrm { e f f } }$ augments F with two penalty terms. The logdeterminant term penalizes subspaces where gradient noise is small or isotropic, biasing convergence toward flat regions of the loss landscape. The feature-noise term penalizes subspaces sensitive to input perturbations, biasing convergence toward noise-robust directions. Together, they ensure that steady-state parameters concentrate on subspaces that are simultaneously flat and robust.

Intuition and takeaways. The two penalty terms reveal a dual implicit bias: lowrank parameters are drawn toward configurations that are flat with respect to the loss and robust with respect to input perturbations. The net efect is that the learned subspace naturally avoids directions sensitive to small variations: the system settles into configurations where outputs remain stable under feature noise, inherently reducing susceptibility to adversarial perturbations and aligning with the conservative behavior observed empirically in Section 2.

Table 3: Comparison of between low-rank operators on Vicuna-13B. Full-parameter alignment serves as the baseline. The selection of low rank operators is highly correlated with rank. A suitable range of rank can be empirically found with better perception.

<table><tr><td></td><td colspan="2">full</td><td colspan="2">LoRA</td><td colspan="3">LoKr</td><td colspan="3">LoHa</td></tr><tr><td>metric</td><td>-</td><td>32</td><td>64</td><td>128</td><td>32</td><td>64</td><td>128</td><td>32</td><td>64</td><td>128</td></tr><tr><td>MME-P</td><td>236.6</td><td>375.9</td><td>288.0</td><td>-</td><td>79.4</td><td>78.5</td><td>449.3</td><td>380.6</td><td>621.4</td><td>1079.7</td></tr><tr><td>POPE</td><td>42.2</td><td>50.0</td><td>19.7</td><td>-</td><td>9.0</td><td>12.7</td><td>46.8</td><td>21.6</td><td>51.5</td><td>82.8</td></tr></table>

Table 4: Ablation on low-rank operators across model scales. Results are reported on 7B / 13B MobileVLM-V2 models. Best results in bold.

<table><tr><td>operators on 7/13B</td><td>perception↑</td><td>knowledge↑</td><td>reasoning↑</td><td>hallucination↑</td></tr><tr><td>full parameter</td><td>521.0/236.6</td><td>23.4/29.5</td><td>86.8/52.5</td><td>35.0/66.9</td></tr><tr><td>LoRA: mat. mul.</td><td>551.8/375.9</td><td>27.0/36.3</td><td>251.4/147.5</td><td>40.9/66.7</td></tr><tr><td>LoHa: Hadamard</td><td>965.3/1079.7</td><td>38.5/43.0</td><td>332.5/219.6</td><td>80.4/82.9</td></tr><tr><td>LoKr: Kronecker</td><td>844.4/449.3</td><td>37.4/40.7</td><td>312.5/62.5</td><td>74.8/70.2</td></tr></table>

## 4 Ablation Study: Low-rank Vision-Language Alignment

Overview. The ablation studies investigate the efects of rank, model scale, alignment depth, low-rank operators, vision encoder choice, frozen vs. unfrozen encoder settings, low-rank placement, learning rate schedules, and overfitting controls. Unless otherwise specified, experiments follow the default settings with targeted modifications. Extended results are provided in the Appendix.

Ranks. In perception-intensive benchmarks, increasing the rank of the low-rank module yields significant performance gains, particularly for larger models. This is evident from the improvements on MME-perception and POPE using the 13B model In Table 3 , in contrast, knowledge-intensive or task-specific benchmarks (e.g., GQA, ScienceQA, TextVQA) show no comparable gains from rank increases, suggesting that the performance bottleneck in these tasks lies elsewhere. We explore this further in the extended-data ablation in the Appendix.

Operators. Ablation of operators is conducted using Vicuna-7B and -13B under default settings with the best-tuned rank for each operator. The results reveal operator-specific strengths across benchmarks, indicating that the choice of lowrank operator is a promising avenue for future optimization.

Model scales and architectures. As shown in Tables 5 and 6, low-rank alignment consistently matches or outperforms full fine-tuning across scales from 1.4B to 14B, and this advantage generalizes to architecturally distinct models including Qwen3-8B/14B under a one-shot protocol. Two exceptions, POPE and MMMU-Pro, appear consistently across both model families, where higher-rank updates partially overwrite pre-trained knowledge structures. The rank dependence of these anomalies suggests that the low-rank constraint acts as a regularizer limiting deviation from the pre-trained weight manifold, a property that becomes increasingly important as rank grows.

Table 5: Performance across model scales and alignment methods. Full-parameter alignment serves as the baseline. MobileVLM-V2 results use rank 64 for all low-rank methods.

<table><tr><td>model</td><td>metric</td><td>Full</td><td>LoRA</td><td>LoKr</td><td>LoHa</td></tr><tr><td rowspan="2">1.4B</td><td>MME-P</td><td>208.1</td><td>568.7</td><td>766.6</td><td>450.8</td></tr><tr><td>POPE</td><td>2.5</td><td>51.5</td><td>72.8</td><td>63.1</td></tr><tr><td rowspan="2">2.7B</td><td>MME-P</td><td>15.8</td><td>52.0</td><td>432.6</td><td>318.3</td></tr><tr><td>POPE</td><td>0.5</td><td>15.0</td><td>71.3</td><td>66.1</td></tr><tr><td rowspan="2">7B</td><td>MME-P</td><td>521.0</td><td>551.8</td><td>825.1</td><td>608.3</td></tr><tr><td>POPE</td><td>35.0</td><td>37.2</td><td>70.7</td><td>67.3</td></tr><tr><td rowspan="2">13B</td><td>MME-P</td><td>236.6</td><td>288.0</td><td>78.5</td><td>621.4</td></tr><tr><td>POPE</td><td>66.9</td><td>44.8</td><td>37.6</td><td>67.4</td></tr></table>

Table 6: Performance of alignment methods on Qwen3. Low-rank alignment works on most of the benchmarks, with notable exceptions on POPE and MMMU-Pro where rank-constrained updates prove insuficient.

<table><tr><td>metric</td><td>full 8B</td><td>LoRA (r=128, 8B)</td><td> $\Delta$  full 14B</td><td>LoRA (r=32, 14B)</td><td> $\Delta$ </td></tr><tr><td>GQA</td><td>28.4</td><td>37.2</td><td>+8.8</td><td>6.6</td><td>20.2</td></tr><tr><td>MMBench</td><td>18.1</td><td>48.7</td><td>+30.6</td><td>12.0</td><td>45.5</td></tr><tr><td>MME-P</td><td>536.0</td><td>752.3</td><td>+216.3</td><td>526.3</td><td>988.9</td></tr><tr><td>MME-R</td><td>212.1</td><td>242.1</td><td>+30.0</td><td>234.3</td><td>308.6</td></tr><tr><td>MMMU-Pro</td><td>12.1</td><td>5.7</td><td>-6.4</td><td>11.8</td><td>11.8</td></tr><tr><td>POPE</td><td>92.7</td><td>25.1</td><td>-67.6</td><td>n/a</td><td>62.3</td></tr><tr><td>ScienceQA</td><td>63.7</td><td>71.3</td><td>+7.6</td><td>59.3</td><td>71.9</td></tr><tr><td>TextVQA</td><td>5.9</td><td>9.7</td><td>+3.8</td><td>4.2</td><td>10.2</td></tr></table>

Other ablation on default settings. These ablations jointly validate the default configuration of our alignment pipeline across four dimensions. (a) Visual encoder. Both DINOv2 [36] and SigLIPv2 [46] backbones benefit from low-rank alignment over full fine-tuning, with consistent gains on GQA and MMBench. The improvement is more pronounced under DINOv2, suggesting that the lowrank constraint interacts more favorably with dense spatial features than with the global token representations of SigLIPv2. (b) Encoder freezing. Freezing the visual encoder during alignment consistently outperforms unfreezing across nearly all metrics, regardless of backbone or alignment method. Unfreezing introduces instability, particularly on TextVQA and MME-p, indicating that gradient updates to the encoder during alignment disrupt pre-trained visual representations rather than refining them. (c) Adaptation scope. Applying low-rank adaptation jointly to both attention and MLP modules yields the best overall performance, with the full configuration outperforming MLP-only and attentiononly variants on MME-p and TextVQA. This suggests that the two module types capture complementary aspects of visual-language alignment and should not be decoupled. (d) Overfitting. Our default low-rank alignment matches or exceeds full fine-tuning with tuned weight decay across all six benchmarks, confirming that the performance advantage is not attributable to overfitting in the full finetuning baseline. The low-rank constraint is an efective overfitting regularizer.

Table 7: Ablation studies on alignment design choices. All configurations use full fine-tuning in the instruction tuning stage with other default settings unchanged. Row colors denote ablation groups: visual encoder , frozen vs. unfrozen encoder , low-rank adaptation scope , and overfitting control .

<table><tr><td>bench.</td><td>GQA</td><td>MME-p</td><td>POPE</td><td>TextVQA</td><td>MMB-dev</td><td>SQA</td></tr><tr><td>dinov2-full</td><td>61.56</td><td>1503.69</td><td>86.86</td><td>51.56</td><td>72.51</td><td>74.27</td></tr><tr><td>dinov2-LoRA</td><td>61.88</td><td>1481.45</td><td>86.52</td><td>55.19</td><td>74.66</td><td>76.15</td></tr><tr><td>siglipv2-full</td><td>62.19</td><td>1406.61</td><td>88.24</td><td>29.58</td><td>72.68</td><td>73.42</td></tr><tr><td>siglipv2-LoRA</td><td>62.73</td><td>1474.70</td><td>87.52</td><td>30.82</td><td>73.37</td><td>72.08</td></tr><tr><td>frz. clip-LoRA</td><td>62.47</td><td>1511.61</td><td>87.52</td><td>44.73</td><td>72.41</td><td>71.42</td></tr><tr><td>un-frz. clip-full</td><td>59.98</td><td>1444.91</td><td>85.79</td><td>40.18</td><td>71.74</td><td>72.53</td></tr><tr><td>un-frz. clip-LoRA</td><td>60.18</td><td>1373.54</td><td>86.49</td><td>42.97</td><td>71.22</td><td>72.53</td></tr><tr><td>un-frz. dinov2-full</td><td>61.78</td><td>1403.20</td><td>87.60</td><td>28.61</td><td>71.82</td><td>72.38</td></tr><tr><td>un-frz. dinov2-LoRA</td><td>63.07</td><td>1481.98</td><td>88.18</td><td>37.27</td><td>76.03</td><td>72.04</td></tr><tr><td>all (attn. + MLP)</td><td>62.47</td><td>1511.61</td><td>87.52</td><td>44.73</td><td>72.41</td><td>71.42</td></tr><tr><td>MLP only</td><td>59.99</td><td>1384.71</td><td>87.95</td><td>41.29</td><td>72.77</td><td>75.41</td></tr><tr><td>attn. only</td><td>62.26</td><td>1391.84</td><td>87.66</td><td>39.38</td><td>69.62</td><td>74.32</td></tr><tr><td>full + weight decay</td><td>61.30</td><td>1432.68</td><td>86.21</td><td>43.55</td><td>70.34</td><td>70.61</td></tr><tr><td>default (LoRA)</td><td>62.47</td><td>1511.61</td><td>87.52</td><td>44.73</td><td>72.41</td><td>71.42</td></tr></table>

## 5 Related Works

Low-rank Adaptation and Vision-Language Models. Low-rank adaptation [14] introduces trainable low-rank matrices into frozen linear layers for parametereficient fine-tuning, with extensions including QLoRA [9], LoHa [17], LoKr [51], and further variants [13, 23, 27, 33, 48, 56]. On the VLM side, Flamingo [1] pioneered visual-language integration, inspiring a broad family of models [6,8,21,26, 50, 57] that difer in scale, architecture, and alignment strategy. In this context, VLoRA [32] suggests that cross-modal information transfer may be intrinsically low-rank, providing empirical motivation for applying low-rank adaptation in the alignment stage. However, whether such constraints introduce systematic implicit biases in visual token representations during alignment remains uncharacterized. Our work targets this gap: rather than proposing a new method, we analyze the alignment dynamics induced by LLM-side low-rank adaptation, a question orthogonal to connector design. Since many open-source VLMs release only trained weights without the full reproduction materials required to replicate their pretraining stage, we adopt a more general alignment pipeline that abstracts away model-specific details while remaining representative of mainstream practice, as evidenced by its continued use in recent popular models such as Kimi2.5-K2 [45]. Our simplified pipeline is therefore deliberately distinct from architecture-specific designs such as Qwen3-VL-style [3] multi-layer visual injection (deepstack), which depends on proprietary training configurations beyond public reproduction, and we cannot reproduce the alignment stages of Qwen3- VL. Moreover, most current open-source models employ a ViT inherited from a previously aligned version rather than trained from unaligned ones, making it dificult to isolate for academic research.

Implicit Bias of Low-Rank Adaptation and Matrix Factorization. A classical line of work studies the implicit regularization of factorized parameterizations: gradient descent on matrix factorization is biased toward low-rank / minimal nuclear-norm solutions [11], an efect that deep factorization intensifies [2] and that need not be captured by any norm [40]; more recent analyses study LoRA from a parameter-eficiency or optimization-landscape standpoint [4, 18]. Our setting departs from this line in four respects: a non-linear LLM backbone, finetuning from a pretrained checkpoint, stochastic (noisy) dynamics, and a crossentropy alignment objective. Consequently, it yields a flat-gradient, noise-robust subspace bias (Sec. 3) rather than the min-norm/low-rank bias of the classical setting; we expand on these four diferences in Appendix 1.4.

## 6 Conclusion

This work investigates the implicit biases introduced by low-rank adaptation during vision-language alignment, a stage widely treated as requiring full-parameter updates. We show that low-rank alignment not only reduces training costs but consistently outperforms full fine-tuning across model scales from 1.4B to 14B parameters. Through behavioral, feature-level, and geometric analyzes, we identify three systematic biases: a shift from hallucinatory to conservative perception, preservation of per-token linear separability (in contrast to full alignment), and structurally homogeneous visual representations. These observations reveal that low-rank alignment retains general modality-specific knowledge during the alignment stage while deferring entity-specific fusion to subsequent instruction tuning. Two theorems ground this behavior in optimization theory, showing that low-rank gradient flow preferentially reinforces noise-robust feature directions and that steady-state parameters concentrate on flat, perturbation-robust subspaces. Extensive experiments across over 100 alignment configurations (spanning model scales, operator families and vision encoders) validate these findings and ofer practical guidance for vision-language alignment.

## Acknowledgements

We thank the ECCV committee and reviewers for their careful reading and constructive feedback. We are also grateful to colleagues for helpful discussions and to the institutions that provided computational resources used in this study. The first author carried out and led this work as an independent researcher.

## References

1. Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: A visual language model for few-shot learning. NeurIPS 35, 23716–23736 (2022)

2. Arora, S., Cohen, N., Hu, W., Luo, Y.: Implicit regularization in deep matrix factorization. In: Advances in Neural Information Processing Systems (NeurIPS) (2019)

3. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)

4. Biderman, D., Portes, J., Gonzalez Ortiz, J.J., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., Blakeney, C., Cunningham, J.P.: LoRA learns less and forgets less. Transactions on Machine Learning Research (TMLR) (2024)

5. Chen, L., Li, J., Dong, X., Zhang, P., He, C., Wang, J., Zhao, F., Lin, D.: Sharegpt4v: Improving large multi-modal models with better captions. In: European Conference on Computer Vision. pp. 370–387. Springer (2024)

6. Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al.: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 24185–24198 (2024)

7. Chiang, W.L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J.E., et al.: Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023) 2(3), 6 (2023)

8. Chu, X., Qiao, L., Zhang, X., Xu, S., Wei, F., Yang, Y., Sun, X., Hu, Y., Lin, X., Zhang, B., et al.: Mobilevlm v2: Faster and stronger baseline for vision language model. arXiv preprint arXiv:2402.03766 (2024)

9. Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L.: Qlora: Eficient finetuning of quantized llms. Advances in Neural Information Processing Systems 36 (2024)

10. Dong, M., Cai, H., Li, J., Zhou, S., Ren, B., Peng, K., Fu, Y.: Visnec: Measuring and leveraging visual necessity for multimodal instruction tuning. arXiv preprint arXiv:2603.01195 (2026)

11. Gunasekar, S., Woodworth, B., Bhojanapalli, S., Neyshabur, B., Srebro, N.: Implicit regularization in matrix factorization. In: Advances in Neural Information Processing Systems (NeurIPS) (2017)

12. Han, Z., Gao, C., Liu, J., Zhang, S.Q., et al.: Parameter-eficient fine-tuning for large models: A comprehensive survey. arXiv preprint arXiv:2403.14608 (2024)

13. Hayou, S., Ghosh, N., Yu, B.: Lora+: Eficient low rank adaptation of large models. arXiv preprint arXiv:2402.12354 (2024)

14. Hu, E.J., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. In: International Conference on Learning Representations

15. Hudson, D.A., Manning, C.D.: Gqa: A new dataset for real-world visual reasoning and compositional question answering. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6700–6709 (2019)

16. HuggAN: WikiArt Dataset. https://huggingface.co/datasets/huggan/wikiart (2022), accessed: 2024-12-19

17. Hyeon-Woo, N., Ye-Bin, M., Oh, T.H.: Fedpara: Low-rank hadamard product for communication-eficient federated learning. In: The Twelfth International Conference on Learning Representations (2022)

18. Jang, U., Lee, J.D., Ryu, E.K.: LoRA training in the NTK regime has no spurious local minima. In: International Conference on Machine Learning (ICML) (2024)

19. Khosla, A., Jayadevaprakash, N., Yao, B., Li, F.F.: Novel dataset for fine-grained image categorization: Stanford dogs. In: Proc. CVPR workshop on fine-grained visual categorization (FGVC). vol. 2 (2011)

20. Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.J., Shamma, D.A., et al.: Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision 123(1), 32–73 (2017)

21. Li, J., Li, D., Savarese, S., Hoi, S.: BLIP-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. ICML pp. 19730– 19742 (2023)

22. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W.X., Wen, J.R.: Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355 (2023)

23. Li, Y., Yu, Y., Liang, C., He, P., Karampatziakis, N., Chen, W., Zhao, T.: Loftq: Lora-fine-tuning-aware quantization for large language models. arXiv preprint arXiv:2310.08659 (2023)

24. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014)

25. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning (2023)

26. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. NeurIPS 36 (2024)

27. Liu, S.Y., Wang, C.Y., Yin, H., Molchanov, P., Wang, Y.C.F., Cheng, K.T., Chen, M.H.: Dora: Weight-decomposed low-rank adaptation. arXiv preprint arXiv:2402.09353 (2024)

28. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? In: European conference on computer vision. pp. 216–233. Springer (2024)

29. Liu, Z., Luo, P., Wang, X., Tang, X.: Deep learning face attributes in the wild. In: Proceedings of International Conference on Computer Vision (ICCV) (December 2015)

30. Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.W., Zhu, S.C., Tafjord, O., Clark, P., Kalyan, A.: Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems 35, 2507–2521 (2022)

31. Lu, P., Qiu, L., Chen, J., Xia, T., Zhao, Y., Zhang, W., Yu, Z., Liang, X., Zhu, S.C.: Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214 (2021)

32. Ma, F., Xue, H., Zhou, Y., Wang, G., Rao, F., Yan, S., Zhang, Y., Wu, S., Shou, M.Z., Sun, X.: Visual perception by large language model’s weights. Advances in Neural Information Processing Systems (2024)

33. Meng, F., Wang, Z., Zhang, M.: PiSSA: Principal singular values and singular vectors adaptation of large language models. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024)

34. Mishra, A., Shekhar, S., Singh, A.K., Chakraborty, A.: Ocr-vqa: Visual question answering by reading text in images. In: 2019 international conference on document analysis and recognition (ICDAR). pp. 947–952. IEEE (2019)

35. Mixon, D.G., Parshall, H., Pi, J.: Neural collapse with unconstrained features. Sampling Theory, Signal Processing, and Data Analysis 20(2), 11 (2022)

36. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

37. Ordonez, V., Kulkarni, G., Berg, T.: Im2text: Describing images using 1 million captioned photographs. Advances in neural information processing systems 24 (2011)

38. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML. pp. 8748–8763. PMLR (2021)

39. Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., et al.: Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024)

40. Razin, N., Cohen, N.: Implicit regularization in deep learning may not be explainable by norms. In: Advances in Neural Information Processing Systems (NeurIPS) (2020)

41. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A.C., Fei-Fei, L.: Imagenet large scale visual recognition challenge (2015), https://arxiv.org/abs/1409.0575

42. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al.: Laion-5b: An open largescale dataset for training next generation image-text models. Advances in neural information processing systems 35, 25278–25294 (2022)

43. Sharma, P., Ding, N., Goodman, S., Soricut, R.: Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In: Proceedings of ACL (2018)

44. Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., Rohrbach, M.: Towards VQA models that can read. In: CVPR. pp. 8317–8326 (2019)

45. Team, K.: Kimi k2.5: Visual agentic intelligence. https://www.kimi.com/blog/ kimi-k2-5 (2025)

46. Tschannen, M., Gritsenko, A., Wang, X., Naeem, M.F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., et al.: Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786 (2025)

47. Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S.: The caltech-ucsd birds-200-2011 dataset (2011)

48. Xu, Y., Xie, L., Gu, X., Chen, X., Chang, H., Zhang, H., Chen, Z., Zhang, X., Tian, Q.: Qa-lora: Quantization-aware low-rank adaptation of large language models. arXiv preprint arXiv:2309.14717 (2023)

49. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Zhou, J., Lin,

J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., Qiu, Z.: Qwen3 technical report (2025), https://arxiv.org/abs/2505.09388

50. Yao, Y., Yu, T., Zhang, A., Wang, C., Cui, J., Zhu, H., Cai, T., Li, H., Zhao, W., He, Z., et al.: Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800 (2024)

51. Yeh, S.Y., Hsieh, Y.G., Gao, Z., Yang, B.B., Oh, G., Gong, Y.: Navigating text-toimage customization: From lycoris fine-tuning to model evaluation. In: The Twelfth International Conference on Learning Representations (2023)

52. Yin, S., Fu, C., Zhao, S., Li, K., Sun, X., Xu, T., Chen, E.: A survey on multimodal large language models. National Science Review 11(12), nwae403 (2024)

53. Yu, W., Yang, Z., Ren, L., Li, L., Wang, J., Lin, K., Lin, C.C., Liu, Z., Wang, L., Wang, X.: Mm-vet v2: A challenging benchmark to evaluate large multimodal models for integrated capabilities. arXiv preprint arXiv:2408.00765 (2024)

54. Yue, X., Zheng, T., Ni, Y., Wang, Y., Zhang, K., Tong, S., Sun, Y., Yu, B., Zhang, G., Sun, H., et al.: Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 15134–15186 (2025)

55. Zhang, S., Dong, L., Li, X., Zhang, S., Sun, X., Wang, S., Li, J., Hu, R., Zhang, T., Wu, F., et al.: Instruction tuning for large language models: A survey. arXiv preprint arXiv:2308.10792 (2023)

56. Zhou, S., Wang, S., Yuan, Z., Shi, M., Shang, Y., Yang, D.: Gsq-tuning: Groupshared exponents integer in fully quantized training for llms on-device fine-tuning. In: Findings of the Association for Computational Linguistics: ACL 2025. pp. 22971–22988 (2025)

57. Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M.: MiniGPT-4: Enhancing visionlanguage understanding with advanced large language models. arXiv preprint arXiv:2304.10592 (2023)

## Appendix

## 1 Implicit Biases: Low-Rank Alignment

## 1.1 Preliminary: First-Order and Symbols

Due to the high-order tensor and Kronecker structures employed in our derivations, we adopt the Einstein summation convention by default. The analysis of the optimized feature H is motivated by the Unconstrained Feature Model (UFM) under the universal approximation assumption and the practical large scale of both data and models.

## Model Definition

$$
O ^ {i} = W _ {0} ^ {i} _ {j} H ^ {j} + b ^ {i} + l ^ {i} (H; \theta),
$$

where $H ^ { j }$ is the input vector (dimension $q )$ , $O ^ { i }$ is the output vector (dimension $p )$ , and \thea denotes, collectively, the parameters of $l ( H ; \theta )$

## Symbols

<table><tr><td>Symbol</td><td>Meaning / Dimension</td></tr><tr><td> $O^{i}, H^{j}$ </td><td>Output/input vectors,  $p, q$ </td></tr><tr><td> $O^{is}, H^{jt}$ </td><td>Kronecker Output/input,  $p \times s, q \times t$ </td></tr><tr><td> $W_{0j}^{i}, b^{i}$ </td><td>Base weight  $p \times q$ , bias  $p$ </td></tr><tr><td> $A^{k}_{j}, B^{i}_{k}$ </td><td>Low-rank factors,  $r \times q, p \times r$ </td></tr><tr><td> $B_{1}, B_{2}, A_{1}, A_{2}$ </td><td>Hadamard-case factors,  $p \times r, r \times q$ </td></tr><tr><td> $C^{s}_{t}$ </td><td>Kronecker factor,  $s \times t$ </td></tr><tr><td> $l^{i}(H; \theta)$ </td><td>Adaptation term with parameters  $\theta$ </td></tr><tr><td> $F(O)$ </td><td>Scalar objective</td></tr><tr><td> $g_{ij}^{(P)}, g_{ij}^{(Q)}$ </td><td>Metric tensors;  $g_{(P)}^{ij}, g_{(Q)}^{ij}$  inverses</td></tr><tr><td> $\delta_{i}^{j}$ </td><td>Kronecker delta</td></tr><tr><td>*, $\otimes$ </td><td>Hadamard, Kronecker products</td></tr><tr><td> $i, j, k$ </td><td>Output, input, latent indices</td></tr><tr><td> $m, n, s, t$ </td><td>Auxiliary summation indices</td></tr><tr><td> $M$ </td><td>Temporary variables for simplification</td></tr></table>

Index and Dimension Conventions

$$
i \in \{1, \dots , p \}, \quad j \in \{1, \dots , q \}, \quad k \in \{1, \dots , r \},
$$

and $m , n , s , t$ are auxiliary summation indices. All indices obey the Einstein summation convention: repeated upper and lower indices are summed.

Metric Tensors Let $g _ { i j } ^ { ( P ) }$ and $g _ { i j } ^ { ( Q ) }$ denote the metric tensors on output and input spaces respectively, with inverses $g _ { ( P ) } ^ { i j }$ and $g _ { ( Q ) } ^ { i j }$ satisfying

$$
g _ {i k} ^ {(P)} g _ {(P)} ^ {k j} = \delta_ {i} ^ {j}, \qquad g _ {i k} ^ {(Q)} g _ {(Q)} ^ {k j} = \delta_ {i} ^ {j}.
$$

For covariant diferentiation we write:

$$
\frac {\partial F}{\partial H _ {j}} = g _ {j m} ^ {(Q)} \frac {\partial F}{\partial H ^ {m}}, \quad \frac {\partial F}{\partial O _ {i}} = g _ {i m} ^ {(P)} \frac {\partial F}{\partial O ^ {m}}.
$$

General Chain Rule

$$
\frac {\partial F}{\partial H ^ {j}} = \frac {\partial F}{\partial O ^ {i}} \frac {\partial O ^ {i}}{\partial H ^ {j}}, \quad \frac {\partial F}{\partial \theta} = \frac {\partial F}{\partial O ^ {i}} \frac {\partial O ^ {i}}{\partial \theta}.
$$

Since

$$
O ^ {i} = W _ {0} ^ {i} _ {j} H ^ {j} + b ^ {i} + l ^ {i} (H; \theta),
$$

it follows that

$$
\frac {\partial O ^ {i}}{\partial H ^ {j}} = W _ {0} ^ {i} _ {j} + \frac {\partial l ^ {i}}{\partial H ^ {j}}.
$$

Case 0 full: l = 0 (Pure Linear Map)

Model.

$$
O ^ {i} = W _ {0} ^ {i} _ {j} H ^ {j} + b ^ {i}
$$

Gradients. Let $\begin{array} { r } { G _ { i } = \frac { \partial F } { \partial O ^ { i } } } \end{array}$ . Then

$$
\begin{array}{c} \frac {\partial F}{\partial H ^ {k}} = G _ {i} W _ {0} ^ {i} _ {k}, \\ \frac {\partial F}{\partial W _ {0} ^ {i} {} _ {j}} = G _ {i} H ^ {j}, \quad \frac {\partial F}{\partial b ^ {i}} = G _ {i}. \end{array}
$$

First-order conditions.

$$
\begin{array}{c} {G _ {i} W _ {0} ^ {i} _ {k} = 0,} \\ {G _ {i} H ^ {j} = 0,} \\ {G _ {i} = 0} \end{array}
$$

Case 1: LoRA l = BAH

Model.

$$
\begin{array}{r} l ^ {i} = B _ {m} ^ {i} A _ {j} ^ {m} H ^ {j} = M _ {j} ^ {i} H ^ {j}, \quad M _ {j} ^ {i} = B _ {m} ^ {i} A _ {j} ^ {m} \\ O ^ {i} = W _ {0 j} ^ {i} H ^ {j} + b ^ {i} + M _ {j} ^ {i} H ^ {j} \end{array}
$$

\frac {ptil F} H^k &= G\_ (W0 + Bm A),   j  qud

$$
\frac {\partial F}{\partial H ^ {k}} = G _ {i} (W _ {0} ^ {i} _ {k} + B ^ {i} _ {m} A ^ {m} _ {k}),
$$

$$
\frac {\partial F}{\partial B _ {m} ^ {i}} = G _ {i} A _ {j} ^ {m} H ^ {j}, \quad \frac {\partial F}{\partial A _ {j} ^ {m}} = G _ {i} B _ {m} ^ {i} H ^ {j}
$$

First-order condition.

$$
\begin{array}{r} G _ {i} (W _ {0} ^ {i} _ {k} + B ^ {i} _ {m} A ^ {m} _ {k}) = 0, \\ G _ {i} A ^ {m} _ {j} H ^ {j} = 0, \\ G _ {i} B ^ {i} _ {m} H ^ {j} = 0 \end{array}
$$

Case 2 LoHa: $l = \left[ \left( B _ { 1 } A _ { 1 } \right) * \left( B _ { 2 } A _ { 2 } \right) \right]$ H (Hadamard)

M\_1^i{}j &= Bm A, \quad 2       l H

$$
\begin{array}{r l} & M _ {1 j} ^ {i} = B _ {1 m} ^ {i} A _ {1 j} ^ {m}, \quad M _ {2 j} ^ {i} = B _ {2 m} ^ {i} A _ {2 j} ^ {m}, \\ & M _ {j} ^ {i} = M _ {1 j} ^ {i} M _ {2 j} ^ {i}, \quad l ^ {i} = M _ {j} ^ {i} H ^ {j} \end{array}
$$

\fracptilFH^k&=G\_(W0+M),B1sAj2qudmn

$$
\frac {\partial F}{\partial H ^ {k}} = G _ {i} (W _ {0} ^ {i} _ {k} + M ^ {i} _ {k}),
$$

$$
\frac {\partial F}{\partial B _ {1 t} ^ {s}} = G _ {s} A _ {1 j} ^ {t} M _ {2 j} ^ {s} H ^ {j}, \quad \frac {\partial F}{\partial B _ {2 t} ^ {s}} = G _ {s} A _ {2 j} ^ {t} M _ {1 j} ^ {s} H ^ {j},
$$

$$
\frac {\partial F}{\partial A _ {1 n} ^ {m}} = G _ {i} B _ {1 m} ^ {i} M _ {2 n} ^ {i} H ^ {n}, \quad \frac {\partial F}{\partial A _ {2 n} ^ {m}} = G _ {i} B _ {2 m} ^ {i} M _ {1 n} ^ {i} H ^ {n}
$$

First-order conditions.

$$
\begin{array}{r} G _ {i} (W _ {0} ^ {i} _ {k} + M _ {1 k} ^ {i} M _ {2 k} ^ {i}) = 0, \\ G _ {i} B _ {1 m} ^ {i} A _ {1 j} ^ {m} H ^ {j} = 0, \\ G _ {i} B _ {2 m} ^ {i} A _ {2 j} ^ {m} H ^ {j} = 0 \end{array}
$$

Case 3 LoKr: $l = [ C \otimes ( B A ) ] H$ (Kronecker)

Model.

$$
M ^ {i} _ {j} = B ^ {i} _ {m} A ^ {m} _ {j}, l ^ {i p} = M ^ {i} _ {j} C ^ {p} _ {q} H ^ {j q}
$$

\frac {ptil F} H^k &= G\_(de   W0 + M C),  Bs  q Aj  u mn

$$
\frac {\partial F}{\partial H ^ {k r}} = G _ {i p} (\delta_ {k} ^ {i} \delta_ {r} ^ {p} W _ {0} ^ {i p} + M _ {k} ^ {i} C _ {r} ^ {p}),
$$

$$
\frac {\partial F}{\partial B _ {t} ^ {s}} = G _ {s p} C _ {q} ^ {p} A _ {j} ^ {t} H ^ {j q}, \quad \frac {\partial F}{\partial A _ {n} ^ {m}} = G _ {i p} C _ {q} ^ {p} B _ {m} ^ {i} H ^ {n q},
$$

$$
\frac {\partial F}{\partial C _ {q} ^ {p}} = G _ {i p} M _ {j} ^ {i} H ^ {j q}
$$

First-order conditions.

$$
\begin{array}{r} G _ {i p} (W _ {0} ^ {i p} + M _ {j} ^ {i} C _ {q} ^ {p}) H ^ {j q} = 0, \\ G _ {i p} A _ {j} ^ {m} B _ {m} ^ {i} H ^ {j q} = 0, \\ G _ {i p} M _ {j} ^ {i} H ^ {j q} = 0 \end{array}
$$

Notes

– Einstein summation convention applies to all repeated indices.

– $\delta _ { j } ^ { i }$ is the Kronecker delta.

$- \ g _ { i j }$ and $h _ { j k }$ are metric tensors for covariant/contravariant transformations. $- \ M , \ M _ { 1 } , \ M _ { 2 }$ denote intermediate low-rank mappings.

## 1.2 LoRA Rank-Weighted Smoothing under Noisy Features

Consider a neural network with weights $W _ { 0 , j } ^ { i }$ and a low-rank update parameterized by $A ^ { p } { } _ { j }$ and $B _ { ~ p } ^ { i }$ . Let the feature vector be $H ^ { j }$ , corrupted by additive noise $\xi ^ { j }$ :

$$
H ^ {j} = \bar {H} ^ {j} + \xi^ {j}, \mathbb {E} [ \xi^ {j} ] = 0, \mathbb {E} [ \xi^ {j} \xi^ {k} ] = \Sigma^ {j k},
$$

where

$\bar { H } ^ { j }$ is the noise-free feature,

$\xi ^ { j }$ is the additive noise,

$\dot { \Sigma } ^ { j k }$ is the noise covariance matrix.

Let the total loss function be $F = F ( W , H )$ . The LoRA gradient flow for A is

$$
\dot {A} ^ {p} _ {j} = - \eta B ^ {i} _ {p} \frac {\partial F}{\partial W _ {0 j} ^ {i}},
$$

where $\eta > 0$ is the learning rate.

Taylor expansion of the gradient Expanding the gradient of F w.r.t W around the noise-free feature H<sup>¯</sup> using a multivariate Taylor expansion:

$$
\begin{array}{l} \frac {\partial F}{\partial W _ {0 j} ^ {i}} (H) = \frac {\partial F}{\partial W _ {0 j} ^ {i}} (\bar {H}) + \frac {\partial^ {2} F}{\partial W _ {0 j} ^ {i} \partial H ^ {k}} \xi^ {k} \\ \qquad + \frac {1}{2} \frac {\partial^ {3} F}{\partial W _ {0 j} ^ {i} \partial H ^ {k} \partial H ^ {l}} \xi^ {k} \xi^ {l} + \mathcal {O} (\| \xi \| ^ {3}), \end{array}
$$

where indices follow Einstein summation convention.

Expectation over noise Taking expectation over the noise $\xi \colon$

$$
\begin{array}{r} \mathbb {E} _ {\xi} \left[ \frac {\partial F}{\partial W _ {0 j} ^ {i}} \right] \simeq \frac {\partial F}{\partial W _ {0 j} ^ {i}} (\bar {H}) + \frac {1}{2} \frac {\partial^ {3} F}{\partial W _ {0 j} ^ {i} \partial H ^ {k} \partial H ^ {l}} \mathbb {E} [ \xi^ {k} \xi^ {l} ] \\ = \frac {\partial F}{\partial W _ {0 j} ^ {i}} (\bar {H}) + \frac {1}{2} \frac {\partial^ {3} F}{\partial W _ {0 j} ^ {i} \partial H ^ {k} \partial H ^ {l}} \Sigma^ {k l}. \end{array}
$$

Substitute into LoRA gradient flow Substituting the expected gradient into the LoRA gradient flow:

$$
\dot {A} ^ {p} _ {j} \simeq - \eta B ^ {i} _ {p} \left(\frac {\partial F}{\partial W _ {0 j} ^ {i}} (\bar {H}) + \frac {1}{2} \frac {\partial^ {3} F}{\partial W _ {0 j} ^ {i} \partial H ^ {k} \partial H ^ {l}} \Sigma^ {k l}\right).
$$

## 1.3 Implicit Biases of Low-Rank Alignment: Gradient Flow

Table 8: Implicit biases summary and comparison: LoRA, LoKr, LoHa

<table><tr><td>Property</td><td>LoRA</td><td>LoKr</td><td>LoHa</td></tr><tr><td>Grad. Flow Subspace</td><td>Rank-r (B, A)</td><td>Rank-r (C ⊗ BA)</td><td>Rank-r1 × r2 ((B1A1) * (B2A2))</td></tr><tr><td>Flatness Bias</td><td>Along BA</td><td>Along C ⊗ BA</td><td>Along (B1A1) * (B2A2)</td></tr><tr><td>Noise Smoothing</td><td>Along H</td><td>Through C ⊗ BA</td><td>Along Hadamard subspace</td></tr><tr><td>Scale Invariance</td><td>Linear in H</td><td>Linear in H</td><td>Linear in H</td></tr></table>

Case $\textit { 1 } L o R A \colon l = B A H$ Consider the LoRA forward mapping:

$$
O ^ {i} = (W _ {0 j} ^ {i} + B _ {p} ^ {i} A _ {j} ^ {p}) H ^ {j},
$$

with a total loss function

$$
F = \mathbb {E} _ {(x, y) \sim \mathcal {D}} \left[ \mathcal {L} (O (x), y) \right].
$$

Assume the feature has additive noise:

$$
H ^ {j} = \bar {H} ^ {j} + \xi^ {j}, \mathbb {E} [ \xi^ {j} ] = 0, \mathbb {E} [ \xi^ {j} \xi^ {k} ] = \Sigma^ {j k}.
$$

Gradient flow dynamics. The gradient flow dynamics for LoRA parameters are:

$$
\dot {A} ^ {p} _ {j} = - \eta \frac {\partial F}{\partial A ^ {p} {} _ {j}} = - \eta B ^ {i} _ {p} \frac {\partial F}{\partial W _ {0 j} ^ {i}},
$$

$$
\dot {B} _ {p} ^ {i} = - \eta \frac {\partial F}{\partial B _ {p} ^ {i}} = - \eta \frac {\partial F}{\partial W _ {0 j} ^ {i}} A _ {j} ^ {p},
$$

with

$$
\frac {\partial F}{\partial W _ {0 j} ^ {i}} = \frac {\partial F}{\partial O ^ {i}} H ^ {j}.
$$

Define the gradient projection:

$$
G ^ {i j} := \mathbb {E} \left[ \frac {\partial F}{\partial O ^ {i}} H ^ {j} \right].
$$

Thus, the expected gradient flow becomes:

$$
\begin{array}{r} \dot {A} _ {j} ^ {p} = - \eta B _ {p} ^ {i} G ^ {i j}, \\ \dot {B} _ {p} ^ {i} = - \eta G ^ {i j} A _ {j} ^ {p}. \end{array}
$$

This shows that the gradient flow is restricted to the rank-r subspace spanned by $B$ and $A ,$ w.r.t the range of $p .$ .

Noise-weighted smoothing. Expanding the gradient under feature noise $\xi \colon$

$$
\mathbb {E} _ {\xi} \left[ \frac {\partial F}{\partial W _ {0 j} ^ {i}} \right] \simeq \frac {\partial F}{\partial W _ {0 j} ^ {i}} (\bar {H}) + \frac {1}{2} \frac {\partial^ {3} F}{\partial W _ {0 j} ^ {i} \partial H ^ {k} \partial H ^ {l}} \Sigma^ {k l}.
$$

Substituting into the gradient flow:

$$
\dot {A} _ {j} ^ {p} \approx - \eta B _ {p} ^ {i} \left(\frac {\partial F}{\partial W _ {0 j} ^ {i}} + \frac {1}{2} \frac {\partial^ {3} F}{\partial W _ {0 j} ^ {i} \partial H ^ {k} \partial H ^ {l}} \Sigma^ {k l}\right),
$$

showing an implicit smoothing efect along noisy feature directions.

Scale invariance. Under feature scaling $H ^ { j }  \alpha H ^ { j }$ , the gradient projection scales linearly:

$$
G ^ {i j} \to \alpha G ^ {i j}, \quad \dot {A} _ {j} ^ {p} \to \alpha \dot {A} _ {j} ^ {p}, \quad \dot {B} _ {p} ^ {i} \to \alpha \dot {B} _ {p} ^ {i}.
$$

This indicates that the implicit flatness preference is invariant to feature scale.

Fokker–Planck approximation and steady-state distribution. Small-step SGD or noisy gradient flow can be approximated as an SDE:

$$
d \theta^ {\alpha} = - \frac {\partial F}{\partial \theta^ {\alpha}} d t + \sqrt {\eta} C ^ {\alpha} _ {\mu} d w _ {t} ^ {\mu},
$$

with noise covariance

$$
D _ {\alpha \beta} = C ^ {\gamma} _ {\alpha} C ^ {\gamma} _ {\beta}.
$$

The corresponding Fokker–Planck equation is:

$$
\partial_ {t} p (\theta , t) = \partial_ {\alpha} \left[ \partial^ {\alpha} F p + \frac {1}{2} \partial_ {\beta} (D ^ {\beta \alpha} p) \right] + \frac {1}{2} \partial_ {\alpha} \partial_ {\beta} (D ^ {\alpha \beta} p).
$$

Assuming no-flux boundary conditions, the steady-state distribution is $\mathrm { a p - }$ proximately:

$$
p _ {\mathrm{stat}} (\theta) \propto \exp \left(- \frac {2}{\eta} F (\theta)\right) \det (D (\theta)) ^ {- 1 / 2}.
$$

LoRA: low-rank subspace marginalization. Define the efective LoRA subspace:

$$
M = B A.
$$

Then the steady-state marginal over M is:

$$
\begin{array}{r l} & p _ {\mathrm{stat}} (M) \propto \exp \left(- \frac {2}{\eta} F (M)\right) \\ & \underbrace {\int_ {B A = M} \det D (B , A) ^ {- 1 / 2} d \mu (B , A)} _ {=: V (M)}. \end{array}
$$

Efective Loss Incorporating Flatness and Noise Define the efective loss in the low-rank subspace:

$$
\begin{array}{r l} & F _ {\mathrm{eff}} (M) = F (M) + \frac {\eta}{4} \log \det \widetilde {D} (M) \\ & \qquad + \frac {1}{2} \Sigma^ {j k} \frac {\partial^ {2} F}{\partial O ^ {i} \partial O ^ {m}} (W _ {0 j} ^ {i} + M _ {j} ^ {i}) (W _ {0 k} ^ {m} + M _ {k} ^ {m}), \end{array}
$$

where the first term is the original loss, the second term encodes gradient-noiseinduced flatness, and the third term encodes feature-noise-weighted smoothing.

Case 2 LoHa: $l = [ ( B _ { 1 } A _ { 1 } ) * ( B _ { 2 } A _ { 2 } ) ] H$ Consider the LoHa forward mapping:

$$
O ^ {i} = (B _ {1 p} ^ {i} A _ {1 j} ^ {p}) * (B _ {2 q} ^ {i} A _ {2 j} ^ {q}) H ^ {j},
$$

with a total loss function

$$
F = \mathbb {E} _ {(x, y) \sim \mathcal {D}} \left[ \mathcal {L} (O (x), y) \right].
$$

Assume additive feature noise:

$$
H ^ {j} = \bar {H} ^ {j} + \xi^ {j}, \mathbb {E} [ \xi^ {j} ] = 0, \mathbb {E} [ \xi^ {j} \xi^ {k} ] = \Sigma^ {j k}.
$$

\dotA\_1^pj&=(B2iq)fraclFOH,.

$$
\begin{array}{r l} & {\dot {A} _ {1 j} ^ {p} = (B _ {2 q} ^ {i} A _ {2 j} ^ {q}) \frac {\partial F}{\partial O ^ {i}} H ^ {j} B _ {1 p} ^ {i},} \\ & {\dot {B} _ {1 p} ^ {i} = ((B _ {2 q} ^ {i} A _ {2 j} ^ {q}) \frac {\partial F}{\partial O ^ {i}} H ^ {j}) A _ {1 j} ^ {p},} \\ & {\dot {A} _ {2 j} ^ {q} = (B _ {1 p} ^ {i} A _ {1 j} ^ {p}) \frac {\partial F}{\partial O ^ {i}} H ^ {j} B _ {2 q} ^ {i},} \\ & {\dot {B} _ {2 q} ^ {i} = ((B _ {1 p} ^ {i} A _ {1 j} ^ {p}) \frac {\partial F}{\partial O ^ {i}} H ^ {j}) A _ {2 j} ^ {q}.} \end{array}
$$

Define gradient projection:

$$
G ^ {i j} := \mathbb {E} \left[ \frac {\partial F}{\partial O ^ {i}} H ^ {j} \right].
$$

\mathb {E}\_xi &lef [ rc p F O^ (B1A \* 2) Hj g ] sq   +  3    n S .

$$
\begin{array}{r l} & {\mathbb {E} _ {\xi} \left[ \frac {\partial F}{\partial O ^ {i}} (B _ {1} A _ {1} * B _ {2} A _ {2}) H ^ {j} \right] \simeq \frac {\partial F}{\partial O ^ {i}} (B _ {1} A _ {1} * B _ {2} A _ {2}) \bar {H} ^ {j}} \\ & {\quad + \frac {1}{2} \frac {\partial^ {3} F}{\partial O ^ {i} \partial H ^ {m} \partial H ^ {n}} (B _ {1} A _ {1} * B _ {2} A _ {2}) \Sigma^ {m n}.} \end{array}
$$

Scale invariance. Under $H ^ { j }  \alpha H ^ { j }$

$$
G ^ {i j} \to \alpha G ^ {i j}, \quad \dot {A} _ {1, 2}, \dot {B} _ {1, 2} \to \alpha \dot {A} _ {1, 2}, \alpha \dot {B} _ {1, 2}.
$$

Fokker–Planck and steady-state.

$$
d \theta^ {\alpha} = - \frac {\partial F}{\partial \theta^ {\alpha}} d t + \sqrt {\eta} C ^ {\alpha} _ {\mu} d w _ {t} ^ {\mu}, D _ {\alpha \beta} = C ^ {\gamma} _ {\alpha} C ^ {\gamma} _ {\beta},
$$

$$
p _ {\mathrm{stat}} (\theta) \propto \exp \left(- \frac {2}{\eta} F (\theta)\right) \det (D (\theta)) ^ {- 1 / 2}.
$$

Efective Loss in LoHa subspace.

$$
M = \left(B _ {1} A _ {1}\right) * \left(B _ {2} A _ {2}\right),
$$

$$
\begin{array}{r l} & F _ {\mathrm{eff}} (M) = F (M) + \frac {\eta}{4} \log \det \widetilde {D} (M) \\ & \qquad + \frac {1}{2} \Sigma^ {j k} \frac {\partial^ {2} F}{\partial O ^ {i} \partial O ^ {m}} (W _ {0 j} ^ {i} + M _ {j} ^ {i}) (W _ {0 k} ^ {m} + M _ {k} ^ {m}). \end{array}
$$

Case 3 LoKr: $l = [ C \otimes ( B A ) ] H$ Consider the LoKr forward mapping:

$$
O ^ {i} = C _ {k} ^ {i} (B _ {p} ^ {k} A _ {j} ^ {p}) H ^ {j},
$$

with a total loss function

$$
F = \mathbb {E} _ {(x, y) \sim \mathcal {D}} \left[ \mathcal {L} (O (x), y) \right].
$$

Assume the feature has additive noise:

$$
H ^ {j} = \bar {H} ^ {j} + \xi^ {j}, \mathbb {E} [ \xi^ {j} ] = 0, \mathbb {E} [ \xi^ {j} \xi^ {k} ] = \Sigma^ {j k}.
$$

Gradient flow dynamics. The gradient flow for LoKr parameters:

$$
\dot {A} ^ {p} _ {j} = - \eta B ^ {k} _ {p} C ^ {i} _ {k} \frac {\partial F}{\partial O ^ {i}} H ^ {j},
$$

$$
\dot {B} _ {p} ^ {k} = - \eta C _ {k} ^ {i} \frac {\partial F}{\partial O ^ {i}} H ^ {j} A _ {j} ^ {p},
$$

$$
\dot {C} _ {k} ^ {i} = - \eta \frac {\partial F}{\partial O ^ {i}} (B _ {p} ^ {k} A _ {j} ^ {p} H ^ {j}).
$$

Define the gradient projection:

$$
G ^ {i j} := \mathbb {E} \left[ \frac {\partial F}{\partial O ^ {i}} H ^ {j} \right].
$$

Expected gradient flow:

$$
\dot {A} ^ {p} _ {j} = - \eta B ^ {k} _ {p} C ^ {i} _ {k} G ^ {i j},
$$

$$
\dot {B} _ {p} ^ {k} = - \eta C _ {k} ^ {i} G ^ {i j} A _ {j} ^ {p},
$$

$$
\dot {C} ^ {i} _ {k} = - \eta G ^ {i j} B ^ {k} _ {p} A ^ {p} _ {j}.
$$

\mathb {E}\_xi lef [ rc p F O^ (Bk Aj) H g ] &sq   + 12 3    n S .

$$
\begin{array}{r l} & {\mathbb {E} _ {\xi} \left[ \frac {\partial F}{\partial O ^ {i}} (B _ {p} ^ {k} A _ {j} ^ {p}) H ^ {j} \right] \simeq \frac {\partial F}{\partial O ^ {i}} (B _ {p} ^ {k} A _ {j} ^ {p}) \bar {H} ^ {j}} \\ & {\qquad + \frac {1}{2} \frac {\partial^ {3} F}{\partial O ^ {i} \partial H ^ {m} \partial H ^ {n}} (B _ {p} ^ {k} A _ {j} ^ {p}) \Sigma^ {m n}.} \end{array}
$$

Scale invariance. Under $H ^ { j }  \alpha H ^ { j }$

$$
\begin{array}{r l} {G ^ {i j} \to \alpha G ^ {i j},} & {\dot {A} ^ {p} _ {j} \to \alpha \dot {A} ^ {p} _ {j},} \\ {\dot {B} ^ {k} _ {p} \to \alpha \dot {B} ^ {k} _ {p},} & {\dot {C} ^ {i} _ {k} \to \alpha \dot {C} ^ {i} _ {k}.} \end{array}
$$

Fokker–Planck approximation and steady-state distribution.

$$
d \theta^ {\alpha} = - \frac {\partial F}{\partial \theta^ {\alpha}} d t + \sqrt {\eta}   C ^ {\alpha} _ {\mu} d w _ {t} ^ {\mu}, \quad D _ {\alpha \beta} = C ^ {\gamma} _ {\alpha} C ^ {\gamma} _ {\beta},
$$

$$
\partial_ {t} p (\theta , t) = \partial_ {\alpha} \left[ \partial^ {\alpha} F p + \frac {1}{2} \partial_ {\beta} (D ^ {\beta \alpha} p) \right] + \frac {1}{2} \partial_ {\alpha} \partial_ {\beta} (D ^ {\alpha \beta} p),
$$

$$
p _ {\mathrm{stat}} (\theta) \propto \exp \left(- \frac {2}{\eta} F (\theta)\right) \det (D (\theta)) ^ {- 1 / 2}.
$$

Efective Loss in LoKr subspace.

$$
M = C \otimes (B A),
$$

$$
\begin{array}{l} F _ {\mathrm{eff}} (M) = F (M) + \frac {\eta}{4} \log \det \widetilde {D} (M) \\ \qquad + \frac {1}{2} \Sigma^ {j k} \frac {\partial^ {2} F}{\partial O ^ {i} \partial O ^ {m}} (W _ {0 j} ^ {i} + M _ {j} ^ {i}) (W _ {0 k} ^ {m} + M _ {k} ^ {m}). \end{array}
$$

## 1.4 Relation to Classical Implicit-Bias Results

A classical line of work studies the implicit regularization of factorized parameterizations: gradient descent on matrix factorization is biased toward low-rank / minimal nuclear-norm solutions [11], an efect that deep factorization intensifies [2] and that need not be captured by any norm [40]. More recent analyses study LoRA from a parameter-eficiency or optimization-landscape standpoint [4, 18]. Our analysis difers along four axes that together yield a qualitatively diferent conclusion: (i) non-linear backbone: prior results target linear or deep-linear networks, whereas we analyze a factorized update acting on a large non-linear LLM; (ii) fine-tuning rather than training from scratch: we study adaptation from a pretrained checkpoint $W _ { 0 } ,$ , so the relevant bias concerns deviation from a pretrained weight manifold rather than convergence from small initialization; (iii) stochastic dynamics: our account explicitly incorporates gradient noise instead of assuming deterministic, full-batch flow; and (iv) alignment objective: we consider a cross-entropy alignment loss rather than squared loss. Under these conditions, our theorems characterize a flat-gradient, noise-robust subspace bias (Sec. 3), distinct from the min-norm/low-rank bias of the classical setting, and it is this bias that underlies the structure-preserving behavior observed empirically.

## 2 More experimental results.

## 2.1 Statement and Limitations

We state the boundaries of our claims explicitly. (i) Controlled alignment pipeline. To isolate alignment-stage efects and keep attribution clean, we deliberately study a minimal pipeline rather than any single proprietary, architecturespecific recipe; our claims are correspondingly scoped to the alignment stage. Within this scope the observed bias is not a single-setup artifact: it holds con sistently across model scales (1.4B–14B), three low-rank operator families, multiple vision encoders, and the architecturally distinct Qwen3 family. Compute constraints further restrict our finest-grained probing to smaller models and to already-post-trained large models. (ii) Explanatory theory. Our theorems are intended as a mechanistic explanation rather than a quantitative bound: they adopt idealized assumptions (an unconstrained-feature view and a small-step, noisy-gradient regime), in the spirit of standard NTK- and neural-collapse-style analyses. Their role is to predict the direction of the observed biases (conservative behavior, homogeneous manifolds, and preserved linear separability), which the experiments corroborate, not to certify exact training trajectories. (iii) Associational evidence. We phrase the link between the feature-level and geometric signatures and downstream gains associationally rather than causally. This is deliberate: the support is a rank-controlled chain in which a single intervention (the low-rank constraint) produces parallel, monotone shifts in per-token linear separability, angular and manifold geometry, Type-I/II behavior, and downstream accuracy, consistently across three operator families and five model scales; we present this as strong, controlled association and do not overstate it as proof of causation. (iv) Boundary cases. The few benchmarks on which low-rank does not lead (e.g., POPE and MMMU-Pro at certain ranks) are consistent with, and predicted by, our regularization account (Sec. 4): they delimit the regime in which the implicit bias is beneficial rather than contradict it.

## 2.2 Why Not Constrain the Connector?

We apply low-rank adaptation to the LLM rather than to the vision–language connector (MLP adapter) by design. The connector holds only a tiny fraction of trainable parameters (<0.3%), so a low-rank constraint there brings negligible compute or memory savings. More importantly, the connector is the sole bottleneck through which all visual tokens pass; forcing it to be low-rank would project every visual feature into one shared low-rank subspace and discard token-specific structure, the opposite of the selective, structure-preserving regularization that LLM-side low-rank provides. We therefore keep the connector at full rank by default and study low-rank adaptation on the LLM.

Table 9: Efect of extended instruction datasets on full vs. low-rank aligned models (MobileVLM-2.7B). ∆zero and ∆Ex. denote gains under zero-shot and extended finetuning settings. Boost ratio = ∆Ex./∆zero − 1. Low-rank results use the best configuration among LoRA/LoHa/LoKr at rank 32/64/128.

<table><tr><td>metric</td><td>full</td><td>low-rank</td><td> $\Delta zero$ </td><td> $\Delta Ex.:full$ </td><td> $\Delta Ex.:low-rank$ </td><td>boost (%)</td></tr><tr><td>GQA</td><td>11.6</td><td>42.4</td><td>30.8↑</td><td>13.7</td><td>3.2</td><td>-34.3</td></tr><tr><td>MMBench</td><td>0.1</td><td>3.6</td><td>3.4↑</td><td>2.7</td><td>30.3</td><td>805.2</td></tr><tr><td>MME-P</td><td>15.7</td><td>668.0</td><td>652.2↑</td><td>148.7</td><td>406.4</td><td>39.6</td></tr><tr><td>MME-R</td><td>40.4</td><td>201.4</td><td>161.1↑</td><td>-19.6</td><td>-7.5</td><td>7.5</td></tr><tr><td>MMMU-Pro</td><td>0.0</td><td>0.2</td><td>0.2↑</td><td>3.8</td><td>5.7</td><td>824.7</td></tr><tr><td>POPE</td><td>0.5</td><td>79.4</td><td>78.8↑</td><td>1.5</td><td>-9.7</td><td>-14.2</td></tr><tr><td>ScienceQA</td><td>1.1</td><td>23.4</td><td>22.3↑</td><td>5.9</td><td>29.2</td><td>104.1</td></tr><tr><td>TextVQA</td><td>0.6</td><td>20.5</td><td>19.8↑</td><td>6.2</td><td>1.4</td><td>-23.9</td></tr></table>

## 2.3 Extended finetuning dataset.

In this work, we primarily focus on zero-shot capabilities. We further investigate how eficiently a model with minimal domain knowledge or limited reasoning ability can absorb new information under both low-rank and full alignment settings, as shown in Table 9. The largest model that is appropriate for this setting is MobileVLM-2.7B. GQA and TextVQA represent previously seen knowledge, whereas MMBench and ScienceQA correspond to novel sub-domain knowledge.

We additionally track changes in perception and reasoning on MME and assess whether the model retains its zero-shot image reasoning ability on MMMU-Pro. POPE is a benchmark for visual perception hallucination.

Analyses. We observe limited gains from previously seen knowledge, such as GQA and TextVQA, although improvements are still present. In contrast, lowrank alignment yields pronounced gains on unseen, knowledge-intensive, or comprehensive benchmarks such as ScienceQA and MMBench. This further supports our hypothesis: low-rank alignment preserves more general knowledge and leaves suficient capacity for subsequent entity-level construction. We also find that new capabilities emerge on unseen tasks such as MMMU-Pro Vision, where accuracy rises from near-zero to meaningful levels, with low-rank alignment exhibiting stronger emergent behavior. On the negative side, both reasoning performance and visual hallucination robustness degrade. The trade-of, however, is an improvement in perceptual ability driven by the increased infusion of visual knowledge.

Table 10: Comparison of full vs. low-rank alignment under full-parameter (F.F.T.) and default (LoRA, rank 128) instruction tuning. Model is MobileVLM-1.4B.

<table><tr><td>metric</td><td>full</td><td>low-rank</td><td>Δdefault</td><td>full F.F.T.</td><td>low-rank F.F.T.</td><td>ΔFFT</td></tr><tr><td>GQA</td><td>18.9</td><td>27.7</td><td>8.8↑</td><td>47.8</td><td>51.9</td><td>4.1↑</td></tr><tr><td>MMBench</td><td>0.0</td><td>0.6</td><td>0.6↑</td><td>2.1</td><td>9.2</td><td>7.1↑</td></tr><tr><td>MME-P</td><td>208.1</td><td>766.6</td><td>558.5↑</td><td>736.9</td><td>743.6</td><td>6.7↑</td></tr><tr><td>MME-R</td><td>22.9</td><td>196.4</td><td>173.6↑</td><td>80.4</td><td>140.0</td><td>59.6↑</td></tr><tr><td>MMMU-Pro</td><td>1.0</td><td>4.4</td><td>3.5↑</td><td>0.2</td><td>0.5</td><td>0.3↑</td></tr><tr><td>POPE</td><td>2.5</td><td>72.8</td><td>70.3↑</td><td>27.6</td><td>58.2</td><td>30.6↑</td></tr><tr><td>ScienceQA</td><td>0.1</td><td>16.4</td><td>16.2↑</td><td>11.0</td><td>25.1</td><td>14.1↑</td></tr><tr><td>TextVQA</td><td>4.2</td><td>8.1</td><td>3.8↑</td><td>24.5</td><td>18.1</td><td>-6.4</td></tr></table>

## 2.4 Full parameter instruction tuning.

This section examines the performance headroom of low-rank aligned models under diferent fine-tuning regimes, including full-parameter tuning. We observe that the smaller model, MobileVLM-1.4B, exhibits far fewer metric collapses when trained with low-rank alignment in Table 10. Even among models that do not collapse, low-rank alignment consistently yields greater robustness.

## 2.5 Linear separability of vision tower’s feature space.

It is observed that the linear separability of CLIP’s tokens is greater than that of those from pretrained VLMs’ adapters on both coarse-grained and fine-grained classification datasets, as shown in Fig 7.

Table 11: Details about dataset for both vision-language alignment and post-training.

<table><tr><td colspan="2">vision-language alignment dataset for MobileVLM-v2</td></tr><tr><td>ShareGPT4V-PT(captioner)</td><td>1.2M</td></tr><tr><td>(subset of following datasets:)</td><td></td></tr><tr><td>· LAION</td><td></td></tr><tr><td>· CC</td><td></td></tr><tr><td>· SBU</td><td></td></tr><tr><td>· MS-COCO</td><td></td></tr><tr><td colspan="2">post-training dataset (same for both)</td></tr><tr><td>LLaVA-Instruction</td><td>665K</td></tr><tr><td>· LLaVA</td><td></td></tr><tr><td>· ShareGPT</td><td></td></tr><tr><td>· MS-COCO</td><td></td></tr><tr><td>· GQA</td><td></td></tr><tr><td>· OCR-VQA</td><td></td></tr><tr><td>· VQAv2</td><td></td></tr><tr><td>· OKVQA</td><td></td></tr><tr><td>· A-OKVQA</td><td></td></tr><tr><td>· TextCaps</td><td></td></tr><tr><td>· RefCOCO</td><td></td></tr><tr><td>· VG</td><td></td></tr></table>

Table 12: Training receipt.

<table><tr><td colspan="2">vision-language alignment</td></tr><tr><td>name</td><td>MobileVLM-v2 (1.7B/3B)</td></tr><tr><td>vision-language alignment datasets</td><td>ShareGPT4V-PT</td></tr><tr><td>trained parameters</td><td>Vision Adapter + LLM</td></tr><tr><td>batchsize</td><td>256</td></tr><tr><td>optimizer</td><td>AdamW</td></tr><tr><td>learning rate</td><td> $1 \times 10^{-3}$ </td></tr><tr><td>scheduler</td><td>cosine</td></tr><tr><td>vision tower</td><td>clip-vit-large-patch14-336</td></tr><tr><td>LLM</td><td>MobileLLaMA-1.4B/2.7B-Chat</td></tr><tr><td>adapter</td><td>ldpnetv2</td></tr><tr><td>max sequence length</td><td>2048</td></tr><tr><td>selected feature layer</td><td>second</td></tr><tr><td colspan="2">post-training</td></tr><tr><td>post-training dataset</td><td>Instruction Tuning from LLaVA-v1</td></tr><tr><td>batchsize</td><td>128</td></tr><tr><td>optimizer</td><td>AdamW</td></tr><tr><td>learning rate</td><td> $4 \times 10^{5}$ </td></tr><tr><td>scheduler</td><td>cosine</td></tr><tr><td>default PEFT</td><td>LoRA</td></tr><tr><td colspan="2">related PEFT setting</td></tr><tr><td>ranks (r) of LoRA</td><td>32/64/128</td></tr><tr><td>α of LoRA</td><td>256 (empirically fixed)</td></tr></table>

## 3 Additional Ablation Experiments with Full Fine-tuning

Depth. In Figure 8, we conduct a grid search over alignment and instructiontuning depths. Intuitively, deeper layers encode higher-level semantic representations, and aligning these may provide a clearer cross-modal bridge. However, the optimal alignment depth varies across model sizes, a phenomenon we term the alignment/fine-tuning depth preference. Specifically, the depth yielding the best alignment is not necessarily optimal for fine-tuning. As illustrated in Figure 8, Vicuna-7B favors deeper layers for alignment and mid-level layers for fine-tuning on perception and hallucination benchmarks, whereas MobileLLaMA-1.4B and -2.7B both favor shallower-to-middle layers for alignment, with divergent finetuning depth preferences.

Learning scheduler. Ablation about learning schedulers is shown in Tabel 13. Consistent results are shown across diferent popular schedulers.

<table><tr><td>bench.</td><td>GQA</td><td>MME-p</td><td>POPE</td><td>TextVQA</td><td>MMB-dev</td><td>SQA</td></tr><tr><td>linear-full</td><td>61.15</td><td>1469.49</td><td>88.09</td><td>43.06</td><td>73.63</td><td>74.77</td></tr><tr><td>linear-pevila</td><td>62.07</td><td>1545.05</td><td>87.36</td><td>46.72</td><td>73.02</td><td>74.15</td></tr><tr><td>wsd-full</td><td>60.92</td><td>1482.24</td><td>87.42</td><td>42.34</td><td>73.54</td><td>74.37</td></tr><tr><td>wsd-pevila</td><td>61.97</td><td>1482.24</td><td>87.09</td><td>47.34</td><td>75.69</td><td>75.41</td></tr></table>

Table 13: Learning rates. Warmup-stable-decay: wsd.

Table 14: Symbols employed in our theoretical analysis.

<table><tr><td>Symbol</td><td>Meaning, Dimension and Shape</td></tr><tr><td>i,j,k,m,n,s,t</td><td>Einstein sum. convention&#x27;s indices</td></tr><tr><td colspan="2">Notice: indices have no meaning but &quot;duplicate→sum&quot;.</td></tr><tr><td> $O^i$ ,  $H^j$ </td><td>Output and input (feature), range: p, q</td></tr><tr><td> $O^{is}$ ,  $H^{jt}$ </td><td>Kronecker Output/input, p × s, q × t</td></tr><tr><td> $W_{0}^{i}{}_{j}, b^{i}$ </td><td>Base weight p × q, bias p</td></tr><tr><td> $A^{k}{}_{j}, B^{i}{}_{k}$ </td><td>Low-rank factors, r × q, p × r</td></tr><tr><td> $B_1$ ,  $B_2$ ,  $A_1$ ,  $A_2$ </td><td>Hadamard-case factors, p × r, r × q</td></tr><tr><td> $C^{s}{}_{t}$ </td><td>Kronecker factor, s × t</td></tr><tr><td> $l^i(H; \theta)$ </td><td>Linear adaptation term, parameter θ</td></tr><tr><td> $F(O; \mathcal{D})$ </td><td>Scalar objective on dataset D</td></tr><tr><td> $\delta_i^j$ ,  $\dot{A}$ </td><td>Kronecker delta, gradient flow of A</td></tr><tr><td>*,⊗</td><td>Hadamard, Kronecker products</td></tr><tr><td>M</td><td>Temporary variables for simplification</td></tr></table>

## 4 Symbols

Our employed symbols are summarized in Table 14

![](images/279b5cc085e89b71724c9b41d045a3945d0f2dcd5fd0a47c477bf6f72b876030.jpg)  
ImageNet (CLIP)

![](images/d8424181bd64840b94c04726099f3fa3d87f061ba381d974446d000a478043a9.jpg)  
STF-dogs (CLIP)

![](images/f0cf7f123bc14dd5395e336b4e540b61c8e05929a35e5009577dbc57eab34c64.jpg)  
CUB (CLIP)

![](images/daf6ddb20ab57482b80d072d6892d0da195470e0190776039714f44973abdd87.jpg)

![](images/0dc56dba5a2bc8385a99239f9207ead58baa84e1ae8bdb7662b9e7f4e50641f9.jpg)

![](images/3268f352e6ff56805d0d4f60b0e22b4e19d88c91abfb60b837e7eee36dff39e7.jpg)

ImageNet (full)  
![](images/1e1a562e5d7860ce9879dfad09165adcb10e00c31ec496820bc7b61821aae1de.jpg)

STF-dogs (full)  
CUB (full)  
![](images/ed78495ebfdbeee74097da5c90faede905f2189cf28a5cba74178c3bfc06295f.jpg)

![](images/a0675b9e64b1826c1abc04c967e2c906f3fc19c0d2f999cb508c761a68581f7b.jpg)  
ImageNet (LoRA)  
STF-dogs (LoRA)  
CUB (LoRA)  
Fig. 7: CLIP, the vision tower of related VLMs, originally has great general and finegrained knowledge structures in each token. Notation CLIP means features in original CLIP of 336×336 resolutions and 14 patch. Notations full/LoRA mean features in each vision token from full-parameter/LoRA pretrained VLMs. Fine-grained classification datasets: STF-dogs and CUB.

![](images/9b85acbc30716f7b32206c050704f10985055711ecc3d74658524f57e3c0c74e.jpg)

![](images/a357450a2b777ffbfb055e989d79dfe5632371cb39cb14bb9cb765402d8527b6.jpg)

![](images/cd5ad98b4f3cec6a13ed6540dbcb656a4eb6fc6fc8617b8d89ab985e5f526c0e.jpg)  
Fig. 8: Depth preferences: the optimal alignment depth depends on the model scale, while patterns in a given model across similar benchmarks remain consistent. Horizontal axis: the vision-tower layer used during the feature alignment stage, sampled from the original 23-layer CLIP encoder at layers {2, 6, 10, 14, 18, 22}. Vertical axis: the visiontower layer used during the instruction fine-tuning stage, selected from layers {2, 10, 18, 22} of the same 23-layer CLIP model. Default: 22 layer. MobileLLaMA-1.4B and -2.7B, Vicuna-7B are used as LLMs backbone.