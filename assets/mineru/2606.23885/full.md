# Mind the Heads: Topological Representation Alignment for Multimodal LLMs

Davide Caffagni<sup>†1</sup> Alberto Compagnoni<sup>†1,2</sup> Federico Melis<sup>†1</sup> Sara Sarto<sup>1</sup> Pier Luigi Dovesi<sup>3</sup> Mark Granroth-Wilding<sup>3</sup> Marcella Cornia<sup>1</sup> Lorenzo Baraldi<sup>1</sup>

<sup>1</sup>University of Modena and Reggio Emilia <sup>2</sup>University of Pisa <sup>3</sup>AMD Silo AI

aimagelab.github.io/HeRA

## Abstract

Representation alignment has emerged as an effective approach to improve Multimodal Large Language Models (MLLMs) by regularizing their internal representations toward those of an external vision encoder. However, existing methods typically align a fixed layer of the language backbone, overlooking the finegrained structure of Transformer models. In this work, we propose Head-Wise Representation Alignment (HeRA), a method that enforces cross-modal alignment at the level of individual attention heads. Our approach is grounded in the Platonic Representation Hypothesis, focusing on preserving the topological structure of representations (i.e., their local neighborhood relationships) across modalities. Following the Mutual K-Nearest Neighbor (MKNN) alignment metric, we introduce a contrastive objective that acts as a differentiable proxy for matching local structures. HeRA applies this objective during multimodal training to specific attention heads in the LLM, selected by their alignment score according to the MKNN metric. Counterintuitively, we find that aligning the least aligned heads yields the largest gains. Extensive evaluations across multiple MLLMs and 18 benchmarks demonstrate that HeRA consistently improves performance on challenging vision-centric tasks and serves as an effective regularizer against visual hallucinations by naturally curbing the over-reliance on linguistic priors. Our code is publicly released.

## 1 Introduction

Multimodal Large Language Models (MLLMs) [1, 19, 34, 41] have emerged as powerful systems capable of solving a wide range of vision-language tasks. Despite their rapid progress, improvements are still largely driven by scaling data, model size, and post-training techniques, rather than by principled changes to their internal mechanisms. While current pipelines have proven highly effective for many applications, MLLMs still exhibit notable limitations in foundational visual reasoning scenarios. Tasks such as confirming the presence of specific objects, accurately counting them, understanding spatial relationships, or parsing dense visual information remain surprisingly challenging [6, 34, 36, 43, 44]. This highlights a severe deficit in visual perception, raising a fundamental question: how can we improve multimodal reasoning by directly intervening on the interaction between vision and language within the model?

A growing line of work addresses this deficiency through representation alignment [2, 38, 46]: during multimodal training, the internal representations of the language model are regularized to match those of an external vision encoder. This can be interpreted as a form of cross-modal distillation, where the MLLM acts as a student and the vision encoder as a teacher, producing aligned representations of the same underlying content across modalities. While this technique has shown promise in improving visual grounding, existing approaches typically align a fixed representation within the language backbone [46], such as the middle layer, without accounting for the internal structure of the model.

![](images/354b68c3d059ab70e31913bf745b82c94acf7dda70a34b655f4d24e407a7ebca.jpg)  
Figure 1: Standard representation alignment imposes strict vision-language feature matching (left), while HeRA (center) matches cross-modal local neighbors, leading to superior VQA results (right).

This limitation is particularly relevant in MLLMs built upon pre-trained LLMs with strong language priors. Unlike diffusion-based models, where representation alignment is applied during training from scratch [15, 47], aligning representations in MLLMs may interact in complex ways with the pre-existing organization of the language model. In this setting, selecting which representation to align becomes a critical design choice.

In this work, we pursue a more principled approach to representation selection, grounded in the Platonic Representation Hypothesis (PRH) [9, 13]. PRH posits that representations learned across different modalities are locally consistent: semantically similar inputs share the same neighborhood structure in their respective latent spaces. This can be interpreted as a form of topological alignment across modalities, where the local geometry of the representation space is preserved, and can be quantified by the Mutual K-Nearest Neighbor (MKNN) metric, which measures the agreement between local neighborhoods. While prior work has established a positive correlation between MKNN alignment and downstream language performance [7, 13], it remains unclear whether explicitly enforcing such alignment leads to improvements in MLLMs.

To address this, we propose Head-Wise Representation Alignment (HeRA), a method that enforces cross-modal alignment at the level of individual attention heads rather than fixed, coarser layers. We use pre-computed MKNN scores as a diagnostic to guide this selection. Counterintuitively, we find that targeting the least aligned heads yields the largest gains, as it strengthens misaligned components of the model while preserving already aligned structures. As outlined in Fig. 1, HeRA applies a contrastive objective to these selected heads, encouraging their representations to match the local topological structure induced by an external vision encoder. This serves as a differentiable proxy for MKNN alignment, promoting cross-modal consistency without imposing rigid feature matching that often conflicts with the language modeling objective.

We evaluate HeRA across multiple MLLMs under the popular LLaVA [19] framework. Extensive evaluations across 18 benchmarks [34] demonstrate that HeRA yields consistent improvements on challenging vision-centric tasks without sacrificing (and often improving) general visual questionanswering performance. Furthermore, the topological alignment enforced by HeRA serves as an effective regularizer against visual hallucinations [10, 39], naturally curbing the models’ tendency to over-rely on linguistic priors.

## 2 Related Work

The Platonic Representation Hypothesis. The Platonic Representation Hypothesis (PRH) [13] posits that models trained across different architectures, modalities, and objectives converge toward structurally similar latent spaces. Crucially, concurrent work [9] highlights that this structural consis tency holds locally rather than globally: semantically equivalent inputs preserve their neighborhood relationships across modalities, while the absolute global geometry may differ. While PRH shows that this local cross-modal alignment naturally emerges with scale and correlates with improved capabilities, it is unclear if a causal relationship can be established. In this work, we investigate whether explicitly enforcing this local neighborhood consistency can lead to better MLLMs.

Vision-Centric Supervision in MLLMs. Recent efforts to boost visual understanding in MLLMs have focused on introducing explicit vision-centric supervision. Several works attempt this by enforcing representation alignment between the MLLM and a teacher vision encoder. However, these methods typically operate directly on the visual features extracted from a fixed, hard-coded layer of the language backbone. For instance, JARVIS [2] reconstructs visual targets using representations from one-quarter of the LLM depth. VIRAL [46] aligns features from the middle layer, and ROSS [38] trains a denoiser using the final layer outputs. In contrast, HeRA takes a fundamentally different approach: we enforce topological alignment within the textual space of the MLLM (conditioned on the multimodal input) rather than strictly matching features from the vision teacher. Furthermore, we abandon the restrictive fixed-layer assumption entirely, instead targeting specific attention heads to preserve local neighborhood structures without conflicting with the language modeling task.

Research on MLLMs is moving fast [1, 41], thanks to massive datasets, post-training, stronger LLM backbones, and natively multimodal models [30, 35]. In this work, we study a novel representation alignment objective on the LLaVA [19] framework to keep experiments computationally tractable, although we also apply it on top of state-of-the-art LLMs, such as the latest Qwen3 [45] family.

## 3 Proposed Method

## 3.1 Background

Multimodal Large Language Models (MLLMs). From an architectural perspective (refer to Fig. 2, left), an MLLM M comprises (i) an LLM G, which constitutes the reasoning backbone and natural language interface of the model; (ii) a pre-trained vision encoder $\nu$ to process visual inputs; and (iii) a projector proj, which aligns the output embedding space of $\nu$ with the input embedding space of $\mathcal { G }$ .

M ingests and generates text x as a sequence of tokens $\mathbf { x } _ { 1 , \dots , T }$ converted into latent vectors by the embedding matrix of ${ \mathcal { G } } .$ On the other hand, visual inputs I are first processed by the vision encoder $\nu ,$ then converted into the input embedding space of $\mathcal { G }$ by the projector: $\mathbf { v } _ { 1 , \dots , V } = \mathrm { p r o j } ( \mathcal { V } ( I ) )$ , and finally concatenated to the sequence of text embeddings. We train $\mathcal { M }$ to minimize the negative log-likelihood of generating token $\mathbf { x } _ { j }$ given the image I and the preceding text $\mathbf { x } _ { 1 , \ldots , j - 1 } { \mathrm { : } }$

$$
\mathcal {L} _ {\mathrm{LM}} \left(I _ {i}, x _ {i}, \mathcal {M}\right) = - \sum_ {j} ^ {T} \log P \left(\mathbf {x} _ {j} \mid \mathbf {v} _ {1, \dots , V}, \mathbf {x} _ {1, \dots , j - 1}; \mathcal {M}\right).\tag{1}
$$

Mutual K-Nearest Neighbor (MKNN) Alignment Metric. MKNN [13] is a kernel alignment metric enabling comparison between different representation functions. In this work, we measure the alignment between textual and visual representations of the same data point. Given an image-text pair $( I , x ) _ { i } \in \mathcal { D }$ , where D is a dataset of aligned image-text pairs, we denote by $\mathcal { G } ( x _ { i } ) \in \mathbb { R } ^ { d _ { \mathcal { G } } }$ a representation of the text extracted from the language model, and by $\mathcal { V } ^ { t } ( I _ { i } ) \in \mathbb { R } ^ { d _ { \nu t } }$ representation of the corresponding image from a teacher vision encoder. Here, $\mathcal { G } ( x _ { i } )$ refers to an internal representation $( e . g .$ , from intermediate layers or attention heads).

For a dataset D, MKNN measures the agreement between the local neighborhood structures induced by the two representation spaces, by computing the average intersection of their k-nearest neighbor sets. We denote by $\mathcal { N } _ { k } ^ { \mathcal { F } } ( \cdot )$ the operator returning the k-nearest neighbors according to maximum dot product similarity in the latent space of the embedding function $\mathcal { F }$ . For instance, in the language space, where we average pool the output embeddings, it is formally defined as follows:

$$
\mathcal {N} _ {k} ^ {\mathcal {G}} (x _ {i}) = \operatorname{argmax} _ {j \neq i} ^ {(k)} \mathcal {G} (x _ {i}) ^ {\top} \mathcal {G} (x _ {j}).\tag{2}
$$

In the visual space, we pool by taking the CLS embeddings at the output of the vision encoder $\mathcal { V } ^ { t }$ The MKNN alignment metric between $\mathcal { G }$ and $\mathcal { V } ^ { t }$ is thus defined as:

$$
m _ {k \mathrm{NN}} (\mathcal {G}, \mathcal {V} ^ {t}, \mathcal {D}) = \underset {(I, x) _ {i} \in \mathcal {D}} {\mathbb {E}} \left[ \frac {1}{k} \left| \mathcal {N} _ {k} ^ {\mathcal {G}} (x _ {i}) \cap \mathcal {N} _ {k} ^ {\mathcal {V} ^ {t}} (I _ {i}) \right| \right] \in [ 0, 1 ],\tag{3}
$$

where $| \cdot |$ denotes set cardinality.

High scores in Eq. 3 reflect that the local topological latent structure generated by $\mathcal { G }$ is preserved in the latent space of $\mathcal { V } ^ { t }$

![](images/82dd07b202da3e0c4827f53cc0e259c37dc12ef1d254241770616f722a571e9b.jpg)  
Figure 2: Overview of HeRA. Alongside the standard language modeling objective $( \mathcal { L } _ { \mathrm { L M } } )$ , HeRA employs a contrastive loss $( \mathcal { L } _ { \mathrm { H e R A } } )$ to pull representations from selected LLM attention heads closer to their k-nearest neighbors (Top-k), computed in the latent space of a frozen teacher vision encoder.

## 3.2 Contrastive Learning as a Proxy for Representation Alignment

For a fixed vision encoder $\mathcal { V } ^ { t }$ $m _ { k \mathrm { N N } }$ has been positively correlated with better performance on language modeling tasks [13]. We want to probe whether a causal effect could exist in a multimodal scenario: can we train a better MLLM by explicitly enforcing alignment with the visual domain?

A natural approach would be to directly maximize $m _ { k \mathrm { N N } }$ during training. However, Eq. 3 depends on discrete neighbor indices and is therefore not differentiable. To address this, we propose a contrastive objective that encourages the multimodal representations produced by M to match the local neighborhood structure induced by the teacher vision encoder.

Given a batch $\boldsymbol { B } = \{ ( I , x ) _ { i } \}$ , let $\mathcal { M } ( I _ { i } , x _ { i } ) \in \mathbb { R } ^ { d _ { \mathcal { G } } }$ denote the multimodal representation obtained by average pooling the text embeddings<sup>1</sup>. For each sample $i ,$ we first identify a set of target neighbors $\mathcal { N } _ { k } ^ { \nu t } ( I _ { i } )$ , corresponding to the k nearest neighbors of $I _ { i }$ in the teacher vision space. We then train M so that its representation $\mathcal { M } ( I _ { i } , x _ { i } )$ is close to the representations of these neighbors, while being separated from the rest of the batch. Formally, this can be achieved via a multi-target variant of the InfoNCE [27] loss:

$$
\mathcal {L} _ {\mathrm{RA}} (I _ {i}, x _ {i}, \mathcal {M}) = - \frac {1}{k} \sum_ {j \in \mathcal {N} _ {k} ^ {\mathcal {V} ^ {t}} (I _ {i})} \log \frac {\exp \left(\frac {\mathcal {M} (I _ {i} , x _ {i}) ^ {\top} \mathcal {M} (I _ {j} , x _ {j})}{\tau}\right)}{\sum_ {z \in \mathcal {B} , z \neq i} \exp \left(\frac {\mathcal {M} (I _ {i} , x _ {i}) ^ {\top} \mathcal {M} (I _ {z} , x _ {z})}{\tau}\right)},\tag{4}
$$

where $\tau$ is a learnable scalar governing the sharpness of the distribution. Minimizing Eq. 4 teaches the student model M to produce multimodal representations sharing the same local neighborhood as the corresponding visual representations from the teacher model $\bar { \nu } ^ { t }$ , which is exactly the property measured by the $m _ { k \mathrm { N N } }$ metric.

## 3.3 Head-Wise Representation Alignment (HeRA)

While the contrastive objective in $\operatorname { E q . }$ 4 enforces alignment at the level of a single pooled representation, it does not account for the internal structure of the language backbone. In particular, G processes the entire multimodal sequence, suggesting that alignment can be more precisely controlled by operating directly on its internal representations.

In principle, G generates multiple representations for a given input. Indeed, we can collect a representation from each Transformer layer of the language backbone. For language modeling $( i . e .$ Eq. 1), we care about the last layer to sample the next token (after passing through the unembedding matrix). Conversely, representation alignment methods [15, 47] typically rely on intermediate layers. However, the choice of which layer(s) to use is typically treated as a fixed hyperparameter (e.g., selecting the middle layer [46]), which does not adapt to the specific structure of a given model.

![](images/625e08e16c4f3f2534d77ab2c392915d62505eb957877fd394a28d2d3eb298a8.jpg)

![](images/b7534c51665e96622b45bbc60f2e2086c1a6e144e85fc1c928bd09d5bbbcd2e2.jpg)  
Figure 3: Left: Alignment with DINOv2-L, measured with the MKNN metric on each layer and attention head of Qwen2.5-3B. Right: MKNN scores of the Worst-5 and Top-5 heads, computed on (i) the base LLM; (ii) after the LLaVA multimodal training; and (iii) after the addition of HeRA.

In this work, we instead probe finer-grained representations within the language model, specifically focusing on the individual attention heads in each multi-head self-attention layer of G. Because different attention heads specialize in different roles within an LLM [25, 26, 40], working at the head level enables a more atomic intervention on the language model, mitigating potential conflicting effects between language modeling and representation alignment.

Head-level Representations. In standard multi-head attention, the final output of a layer is obtained by concatenating the outputs of the individual heads and multiplying them by an output projection matrix $\mathbf { W } _ { O } \in \bar { \mathbb { R } ^ { d _ { g } \times d _ { \mathcal { G } } } }$ . Let $\mathbf { h } _ { l , h } \in \mathbb { R } ^ { d _ { h e a d } }$ be the output of the h-th attention head in layer l, where $d _ { h e a d } = d _ { \mathcal { G } } / H$ . The output projection can be written as:

$$
\operatorname{MultiHead} (\cdot) = \left[ \mathbf {h} _ {l, 1} (\cdot), \dots , \mathbf {h} _ {l, H} (\cdot) \right] \mathbf {W} _ {O}.\tag{5}
$$

Because matrix multiplication is a linear operator, we can decompose $\mathbf { W } _ { O }$ into H distinct blocks along its row dimension, such that $\mathbf { W } _ { O } = \mathbf { \widetilde { \Gamma } } [ \mathbf { W } _ { O , 1 } ^ { \top } , \ldots , \mathbf { W } _ { O , H } ^ { \top } ] ^ { \top }$ , with each $\mathbf { W } _ { O , h } \in \mathbb { R } ^ { d _ { h e a d } \times d _ { \mathcal { G } } }$ The multi-head attention output can then be equivalently written as:

$$
\operatorname{MultiHead} (\cdot) = \sum_ {h = 1} ^ {H} \mathbf {h} _ {l, h} (\cdot) \mathbf {W} _ {O, h}.\tag{6}
$$

This decomposition allows us to isolate the projected contribution of each head before it is summed into the shared residual stream. Given a multimodal input $( I _ { i } , x _ { i } )$ , we define:

$$
\mathcal {M} ^ {l, h} (I _ {i}, x _ {i}) = \mathbf {h} _ {l, h} (I _ {i}, x _ {i}) \mathbf {W} _ {O, h} \in \mathbb {R} ^ {d _ {\mathcal {G}}},\tag{7}
$$

where $\mathcal { M } ^ { l , h }$ denotes the representation extracted from the h-th attentive head in the l-th layer of M during the multimodal forward pass.

Head-wise Alignment Objective. We apply the contrastive alignment loss of Eq. 4 independently to selected head-level representations. For a set of layer-head indices $\mathcal { H } = \{ ( l , \bar { h } ) \}$ }, we define our head-wise representation alignment loss as the average alignment loss over the selected heads:

$$
\mathcal {L} _ {\mathrm{HeRA}} (I _ {i}, x _ {i}, \mathcal {M}, \mathcal {H}) = \underset {(l, h) \in \mathcal {H}} {\mathbb {E}} \left[ \mathcal {L} _ {\mathrm{RA}} (I _ {i}, x _ {i}, \mathcal {M} ^ {l, h}) \right].\tag{8}
$$

The final training objective (illustrated in Fig. 2) is given by the sum of the language modeling and head-wise representation alignment losses:

$$
\mathcal {L} (I _ {i}, x _ {i}, \mathcal {M}, \mathcal {H}) = \mathcal {L} _ {\mathrm{LM}} (I _ {i}, x _ {i}, \mathcal {M}) + \lambda \mathcal {L} _ {\mathrm{HeRA}} (I _ {i}, x _ {i}, \mathcal {M}, \mathcal {H}),\tag{9}
$$

where λ is a fixed hyperparameter to balance the two contributions.

Table 1: Ablation study on the choice of (i) the objective for representation alignment (feature- vs. contrastive-based), and (ii) the granularity of the LLM representation to align (layer- vs. head-level).

<table><tr><td colspan="3">Representation Alignment</td><td colspan="5">LLM: Qwen2.5-3B</td><td colspan="5">LLM: Qwen3-4B</td></tr><tr><td>Objective</td><td>Granularity</td><td>Selection</td><td>General</td><td>Knowledge</td><td>OCR</td><td>Vision</td><td>All</td><td>General</td><td>Knowledge</td><td>OCR</td><td>Vision</td><td>All</td></tr><tr><td>-</td><td>-</td><td>-</td><td>73.5</td><td>46.7</td><td>42.2</td><td>50.5</td><td>54.2</td><td>75.6</td><td>49.6</td><td>43.8</td><td>56.3</td><td>57.4</td></tr><tr><td>Feature</td><td>Layer</td><td>Middle</td><td>72.6</td><td>46.4</td><td>42.3</td><td>50.1</td><td>53.8</td><td>75.0</td><td>48.6</td><td>42.8</td><td>53.4</td><td>56.0</td></tr><tr><td>Feature</td><td>Head</td><td>Worst (5)</td><td>74.0</td><td>46.5</td><td>42.8</td><td>51.6</td><td>54.7</td><td>76.0</td><td>49.3</td><td>44.7</td><td>55.9</td><td>57.6</td></tr><tr><td>Contrastive</td><td>Layer</td><td>Middle</td><td>72.8</td><td>45.7</td><td>41.6</td><td>51.1</td><td>53.8</td><td>75.8</td><td>49.5</td><td>43.7</td><td>57.2</td><td>57.6</td></tr><tr><td>Contrastive</td><td>Layer</td><td>Worst (5)</td><td>73.1</td><td>46.2</td><td>40.9</td><td>51.7</td><td>54.0</td><td>73.2</td><td>47.9</td><td>41.2</td><td>52.1</td><td>54.6</td></tr><tr><td>Contrastive</td><td>Head</td><td>Random (5)</td><td>73.7</td><td>46.6</td><td>42.1</td><td>49.7</td><td>54.0</td><td>76.1</td><td>49.7</td><td>44.0</td><td>56.6</td><td>57.7</td></tr><tr><td>Contrastive</td><td>Head</td><td>Top (5)</td><td>73.8</td><td>46.5</td><td>42.8</td><td>50.2</td><td>54.3</td><td>75.9</td><td>49.9</td><td>45.2</td><td>56.3</td><td>57.8</td></tr><tr><td>Contrastive</td><td>Head</td><td>Worst (1)</td><td>73.5</td><td>46.9</td><td>42.9</td><td>51.2</td><td>54.6</td><td>76.0</td><td>49.7</td><td>44.1</td><td>57.0</td><td>57.8</td></tr><tr><td>Contrastive</td><td>Head</td><td>Worst (3)</td><td>73.9</td><td>47.6</td><td>43.6</td><td>51.0</td><td>55.0</td><td>75.8</td><td>49.9</td><td>44.7</td><td>57.0</td><td>57.9</td></tr><tr><td>Contrastive</td><td>Head</td><td>Worst (10)</td><td>62.6</td><td>45.5</td><td>34.6</td><td>38.9</td><td>46.0</td><td>75.8</td><td>49.7</td><td>44.7</td><td>56.3</td><td>57.7</td></tr><tr><td>Contrastive</td><td>Head</td><td>Worst (5)</td><td>74.5</td><td>47.5</td><td>43.8</td><td>52.9</td><td>55.7</td><td>76.0</td><td>50.1</td><td>44.5</td><td>58.5</td><td>58.4</td></tr></table>

Heads Selection. For an LLM with L layers and H attention heads, the total number of heads is $L \times H$ . In practice, this number is in the order of hundreds, making an extensive search for the optimal H unfeasible. To this end, we propose to exploit the m metric of Eq. 3 to rank the heads by their alignment score with the vision encoder V<sup>t</sup>. We compute this rank using the language model G before the multimodal training, so that its representations are purely textual. Surprisingly, we find that there always exists a set of heads whose alignment score greatly exceeds that of any layer in the same model (see Fig. 3, left). Once the alignment rank is computed, we posit to select a subset of m heads following two reasonable strategies. Specifically, we select either (i) the best aligned heads $( i . e . , \mathcal { H } _ { m } ^ { \mathrm { t o p } } )$ , as forcing the alignment should be easier because they start from an already partially aligned latent space, or (ii) the least aligned heads $( i . e . , \mathcal { H } _ { m } ^ { \mathrm { w o r s t } } )$ ), so to strengthen the components of the model further away from the visual domain. Empirically, we find that choosing $\mathcal { H } _ { m } ^ { \mathrm { { \mathrm { w o r s t } } } }$ works best: it boosts the alignment of poorly aligned heads, while preserving the alignment of the strongest heads, as displayed in Fig. 3, right.

## 4 Experiments

## 4.1 Experimental Settings

Training Details. We train all models following the two-stage LLaVA-1.5 pipeline [19], with the same training data and protocol. As vision encoder V, we adopt SigLIP2 ViT-SO400M/14@384 [37] across all experiments. For the LLM G, we consider a diverse set of architectures, including Vicuna [3], LLama3 [8], Qwen2.5 [29], and Qwen3 [45], ranging from 3B to 14B parameters. We apply the HeRA loss (cf. Eq. 8) in both training stages, using λ equal to 0.01, and k equal to 10. Unless otherwise specified, we use DINOv2 ViT-L [28] as teacher vision encoder V<sup>t</sup>.

Head Selection. We perform head selection prior to multimodal training. Specifically, we compute the $m _ { k \mathrm { N N } }$ alignment score (Eq. 3) for each head using the LLM G, before any multimodal finetuning. This produces a ranking of heads based on their degree of alignment with the visual domain. The scores are computed on 1,000 samples from the GranD dataset [31], which provides highly detailed captions enabling a reliable estimation of cross-modal neighborhood structure. We select the $m = 5$ least aligned heads and restrict the application of HeRA to this subset throughout training.

Evaluation Benchmarks. We primarily evaluate our method using the Cambrian comprehensive benchmark suite [34] covering General, Knowledge, OCR, and Vision tasks. In addition, we evaluate hallucination robustness on CHAIR-MSCOCO [49], AMBER [39], and HallusionBench [10]. The complete details on the evaluation datasets are reported in Appendix B.

## 4.2 Ablation Studies and Analyses

We start by presenting a set of ablation studies in Table 1 designed to understand how key architectural and objective choices influence the behavior of HeRA. For these experiments, we employ Qwen2.5- 3B [29] and Qwen3-4B [45] as the underlying LLMs. As a baseline, we consider a LLaVA model trained on top of the same LLMs without alignment regularization (first row). For all configurations, we use the same training settings used in our approach, as described in Sec. 4.1.

Table 2: VQA results of HeRA applied to the LLaVA training recipe on different LLMs.

<table><tr><td rowspan="2">Model</td><td>General</td><td>Knowledge</td><td>OCR</td><td colspan="6">Vision</td></tr><tr><td>Avg</td><td>Avg</td><td>Avg</td><td>RWQA</td><td>MMVP</td><td>Blink</td><td>V*</td><td>CVBench</td><td>Avg</td></tr><tr><td>Qwen2.5-3B</td><td>73.5</td><td>46.7</td><td>42.2</td><td>55.2</td><td>46.0</td><td>46.8</td><td>44.5</td><td>60.2</td><td>50.5</td></tr><tr><td>+ HeRA (Ours)</td><td>74.5</td><td>47.5</td><td>43.8</td><td>56.3</td><td>48.0</td><td>49.1</td><td>51.3</td><td>59.6</td><td>52.9</td></tr><tr><td>Qwen3-4B</td><td>75.6</td><td>49.6</td><td>43.8</td><td>59.9</td><td>48.0</td><td>55.1</td><td>50.3</td><td>68.3</td><td>56.3</td></tr><tr><td>+ HeRA (Ours)</td><td>76.0</td><td>50.1</td><td>44.5</td><td>61.3</td><td>56.0</td><td>53.7</td><td>52.4</td><td>69.1</td><td>58.5</td></tr><tr><td>Vicuna-7B</td><td>72.2</td><td>44.3</td><td>45.7</td><td>56.5</td><td>38.7</td><td>46.8</td><td>44.5</td><td>62.1</td><td>49.7</td></tr><tr><td>+ HeRA (Ours)</td><td>72.1</td><td>44.5</td><td>45.7</td><td>57.8</td><td>42.7</td><td>47.9</td><td>49.7</td><td>61.9</td><td>52.0</td></tr><tr><td>LLama3-8B</td><td>73.3</td><td>45.0</td><td>43.0</td><td>60.1</td><td>46.0</td><td>49.2</td><td>44.0</td><td>69.5</td><td>53.8</td></tr><tr><td>+ HeRA (Ours)</td><td>74.6</td><td>46.3</td><td>44.7</td><td>60.4</td><td>46.7</td><td>50.2</td><td>51.8</td><td>66.4</td><td>55.1</td></tr><tr><td>Qwen2.5-7B</td><td>76.2</td><td>50.2</td><td>47.9</td><td>59.6</td><td>51.3</td><td>51.7</td><td>50.8</td><td>70.2</td><td>56.7</td></tr><tr><td>+ HeRA (Ours)</td><td>76.5</td><td>50.5</td><td>48.6</td><td>61.3</td><td>54.0</td><td>50.2</td><td>50.3</td><td>71.3</td><td>57.4</td></tr><tr><td>Qwen3-8B</td><td>74.7</td><td>49.7</td><td>43.5</td><td>59.2</td><td>49.3</td><td>52.2</td><td>50.8</td><td>67.8</td><td>55.9</td></tr><tr><td>+ HeRA (Ours)</td><td>76.9</td><td>51.1</td><td>47.6</td><td>60.1</td><td>58.0</td><td>55.5</td><td>50.3</td><td>73.5</td><td>59.5</td></tr><tr><td>Vicuna-13B</td><td>73.4</td><td>45.5</td><td>47.7</td><td>58.7</td><td>44.7</td><td>50.6</td><td>46.1</td><td>63.7</td><td>52.7</td></tr><tr><td>+ HeRA (Ours)</td><td>73.6</td><td>45.7</td><td>47.6</td><td>57.3</td><td>44.0</td><td>52.3</td><td>49.2</td><td>66.5</td><td>53.9</td></tr><tr><td>Qwen2.5-14B</td><td>75.6</td><td>50.7</td><td>44.8</td><td>60.1</td><td>47.3</td><td>52.7</td><td>46.6</td><td>67.9</td><td>54.9</td></tr><tr><td>+ HeRA (Ours)</td><td>77.4</td><td>52.8</td><td>49.3</td><td>60.1</td><td>52.0</td><td>55.2</td><td>52.4</td><td>71.6</td><td>58.3</td></tr><tr><td>Qwen3-14B</td><td>77.4</td><td>52.8</td><td>46.1</td><td>60.3</td><td>57.3</td><td>52.6</td><td>50.3</td><td>70.8</td><td>58.2</td></tr><tr><td>+ HeRA (Ours)</td><td>77.7</td><td>52.6</td><td>47.8</td><td>62.5</td><td>58.0</td><td>52.0</td><td>51.8</td><td>70.2</td><td>58.9</td></tr></table>

Objective and Granularity. First, we consider a standard representation alignment approach [46, 47], where the LLM is trained to minimize the cosine similarity between visual-token features at the middle layer and those from the teacher vision encoder (second row), using a trainable projector. While this is ineffective on both LLMs, switching to our contrastive learning objective using the same representation granularity (fourth row) shows promising results on vision tasks.

Head-Level Alignment and Selection. Sticking with the contrastive objective, we move to a finer granularity, considering the textual representations from specific attentive heads in the LLM. The head selection criterion follows the MKNN alignment score with the vision encoder: we select either the top-5 (seventh row) or worst-5 (last row) according to this ranking. On both LLMs, we record a striking difference favoring alignment on the worst-5 heads. For instance, Qwen2.5-3B boosts its performance on vision-centric tasks by +1.4 points, whereas Qwen3-4B enjoys a +2.3 points gain. As control trials, we also apply the contrastive alignment on a random subset of 5 heads (sixth row), yielding no clear benefit, and experiment with the “worst-5” selection criterion at the layer-level (fifth row), which actually registers a mild performance regression on Qwen2.5-3B and a severe degradation on Qwen3-4B.

Connection to the Platonic Representation Hypothesis. The superiority of the worst-5 strategy corroborates the positive correlation between alignment and performance reported by the PRH [13]. As shown in Fig. 3 (right), after HeRA training, the worst-5 heads drastically increase their visionlanguage alignment without penalizing the alignment of the top-5 heads. Conversely, explicitly forcing alignment on the top-5 heads has no meaningful collateral impact on the worst-5 heads. Interestingly, aligning the visual features from the worst-5 heads (third row) is ineffective, and has little impact on their alignment scores (see the plot in Fig. 5 in Appendix C).

Number of Heads to Align. Finally, we ablate the number of heads to align (Table 1, bottom). Using 3 heads, or even a single one, yields modest gains on both LLMs, while the best overall performance is achieved with 5 heads, particularly on challenging vision benchmarks. Conversely, further scaling up the number of heads to 10 leads to a regression, particularly with Qwen2.5-3B, indicating that aligning too many heads begins to conflict with the core language modeling task.

## 4.3 Main Experimental Results

Results on Cambrian Benchmarks. To assess the generalizability and scalability of our proposed representation alignment regularization, we evaluate HeRA across a diverse suite of language models, as reported in Table 2. We deliberately select models spanning multiple architectural generations to ensure our findings are not isolated to a specific design. This includes established baselines like the Vicuna family, as well as the latest generation of state-of-the-art open-source models, such as Qwen3. Furthermore, we scale the parameter count across our experiments, progressing from compact models (3B and 4B) up to larger reasoning engines (13B and 14B). We remind to Appendix C for the complete breakdown of the General, Knowledge, and OCR categories.

Table 3: Results of HeRA on visual hallucinations benchmarks.

<table><tr><td rowspan="2"></td><td colspan="2">MSCOCO</td><td colspan="4">AMBER (Generative)</td><td colspan="4">AMBER (Discriminative)</td><td colspan="3">HallusionBench</td></tr><tr><td>CHAIRS↓</td><td>CHAIRi↓</td><td>CHAIRi↓</td><td>Cover↑</td><td>HalRate↓</td><td>Cog↓</td><td>Acc↑</td><td>Prec↑</td><td>Rec↑</td><td>F1↑</td><td>qAcc↑</td><td>Easy↑</td><td>Hard↑</td></tr><tr><td>Qwen2.5-3B</td><td>44.2</td><td>12.6</td><td>5.8</td><td>52.0</td><td>30.1</td><td>3.3</td><td>83.6</td><td>83.9</td><td>93.2</td><td>88.3</td><td>23.7</td><td>60.7</td><td>55.2</td></tr><tr><td>+ HeRA (Ours)</td><td>44.0</td><td>11.8</td><td>5.2</td><td>52.3</td><td>27.6</td><td>2.8</td><td>85.6</td><td>86.5</td><td>92.8</td><td>89.5</td><td>24.4</td><td>61.3</td><td>60.2</td></tr><tr><td>Qwen3-4B</td><td>42.0</td><td>11.7</td><td>5.5</td><td>53.0</td><td>28.2</td><td>3.0</td><td>86.5</td><td>89.9</td><td>89.6</td><td>89.7</td><td>22.2</td><td>63.1</td><td>54.6</td></tr><tr><td>+ HeRA (Ours)</td><td>43.8</td><td>11.9</td><td>5.5</td><td>52.5</td><td>27.6</td><td>2.8</td><td>86.5</td><td>90.1</td><td>89.4</td><td>89.7</td><td>18.9</td><td>60.7</td><td>50.3</td></tr><tr><td>Vicuna-7B</td><td>46.6</td><td>12.6</td><td>6.3</td><td>52.2</td><td>30.8</td><td>3.3</td><td>84.2</td><td>90.0</td><td>85.8</td><td>87.8</td><td>15.6</td><td>55.4</td><td>45.7</td></tr><tr><td>+ HeRA (Ours)</td><td>47.6</td><td>12.9</td><td>6.1</td><td>52.6</td><td>31.9</td><td>3.5</td><td>85.9</td><td>90.8</td><td>87.7</td><td>89.2</td><td>16.3</td><td>55.0</td><td>49.6</td></tr><tr><td>LLama3-8B</td><td>44.8</td><td>13.0</td><td>5.5</td><td>51.5</td><td>27.5</td><td>2.9</td><td>85.9</td><td>89.2</td><td>89.6</td><td>89.4</td><td>16.9</td><td>58.7</td><td>49.9</td></tr><tr><td>+ HeRA (Ours)</td><td>40.2</td><td>11.7</td><td>5.6</td><td>51.7</td><td>26.6</td><td>2.7</td><td>85.9</td><td>88.3</td><td>90.8</td><td>89.5</td><td>17.4</td><td>58.0</td><td>49.3</td></tr><tr><td>Qwen2.5-7B</td><td>45.2</td><td>11.9</td><td>5.3</td><td>52.3</td><td>26.7</td><td>2.7</td><td>87.6</td><td>89.6</td><td>91.9</td><td>90.7</td><td>21.5</td><td>66.8</td><td>49.4</td></tr><tr><td>+ HeRA (Ours)</td><td>44.8</td><td>12.4</td><td>4.9</td><td>52.5</td><td>25.1</td><td>2.7</td><td>87.7</td><td>89.1</td><td>92.8</td><td>90.9</td><td>23.3</td><td>66.6</td><td>50.9</td></tr><tr><td>Qwen3-8B</td><td>42.0</td><td>12.3</td><td>5.8</td><td>52.8</td><td>28.9</td><td>3.2</td><td>87.3</td><td>90.3</td><td>90.6</td><td>90.4</td><td>21.1</td><td>61.1</td><td>53.3</td></tr><tr><td>+ HeRA (Ours)</td><td>39.2</td><td>11.0</td><td>5.2</td><td>53.1</td><td>27.9</td><td>3.0</td><td>89.0</td><td>92.4</td><td>90.9</td><td>91.6</td><td>25.7</td><td>66.2</td><td>54.5</td></tr><tr><td>Vicuna-13B</td><td>43.0</td><td>11.9</td><td>6.1</td><td>52.5</td><td>28.4</td><td>3.3</td><td>84.8</td><td>93.9</td><td>82.4</td><td>87.8</td><td>13.0</td><td>55.2</td><td>43.9</td></tr><tr><td>+ HeRA (Ours)</td><td>47.4</td><td>12.2</td><td>6.1</td><td>52.9</td><td>30.3</td><td>3.2</td><td>85.7</td><td>91.7</td><td>86.2</td><td>88.9</td><td>14.7</td><td>56.0</td><td>45.6</td></tr><tr><td>Qwen2.5-14B</td><td>42.6</td><td>12.2</td><td>6.1</td><td>52.3</td><td>29.4</td><td>3.6</td><td>86.2</td><td>87.9</td><td>91.8</td><td>89.8</td><td>23.7</td><td>61.5</td><td>56.5</td></tr><tr><td>+ HeRA (Ours)</td><td>39.0</td><td>10.6</td><td>5.4</td><td>51.9</td><td>28.0</td><td>3.1</td><td>89.0</td><td>90.4</td><td>93.4</td><td>91.9</td><td>25.9</td><td>66.2</td><td>56.4</td></tr><tr><td>Qwen3-14B</td><td>43.2</td><td>11.7</td><td>5.5</td><td>53.1</td><td>28.8</td><td>3.2</td><td>88.8</td><td>91.6</td><td>91.5</td><td>91.5</td><td>23.7</td><td>66.2</td><td>53.0</td></tr><tr><td>+ HeRA (Ours)</td><td>41.2</td><td>11.4</td><td>5.0</td><td>53.0</td><td>26.7</td><td>3.0</td><td>88.9</td><td>90.7</td><td>92.7</td><td>91.7</td><td>27.9</td><td>69.0</td><td>56.8</td></tr></table>

Table 4: VQA comparison of different representation alignment strategies for MLLMs.

<table><tr><td rowspan="2">Alignment</td><td>General</td><td>Knowledge</td><td>OCR</td><td colspan="6">Vision</td></tr><tr><td>Avg</td><td>Avg</td><td>Avg</td><td>RWQA</td><td>MMVP</td><td>Blink</td><td>V*</td><td>CVBench</td><td>Avg</td></tr><tr><td>-</td><td>74.7</td><td>49.7</td><td>43.5</td><td>59.2</td><td>49.3</td><td>52.2</td><td>50.8</td><td>67.8</td><td>55.9</td></tr><tr><td>ROSS [38]</td><td>74.6</td><td>49.4</td><td>44.0</td><td>59.8</td><td>50.3</td><td>52.2</td><td>49.2</td><td>69.7</td><td>56.2</td></tr><tr><td>VIRAL [46]</td><td>73.8</td><td>49.3</td><td>43.6</td><td>57.6</td><td>46.7</td><td>51.3</td><td>46.1</td><td>69.3</td><td>54.2</td></tr><tr><td>JARVIS [2]</td><td>76.8</td><td>49.9</td><td>46.2</td><td>59.2</td><td>54.7</td><td>54.2</td><td>55.5</td><td>69.9</td><td>58.7</td></tr><tr><td>CMAR [7]</td><td>76.4</td><td>51.0</td><td>46.0</td><td>58.8</td><td>50.0</td><td>52.1</td><td>52.9</td><td>70.9</td><td>56.9</td></tr><tr><td>HeRA (Ours)</td><td>76.9</td><td>51.1</td><td>47.6</td><td>60.1</td><td>58.0</td><td>55.5</td><td>50.3</td><td>73.5</td><td>59.5</td></tr></table>

A common risk when forcefully modifying the internal representations of a pre-trained LLM is the potential degradation of its inherent linguistic and reasoning priors. However, the results demonstrate that our head-wise alignment successfully preserves, and frequently improves, the model’s core competencies. Across the General, Knowledge, and OCR task categories, the inclusion of HeRA consistently yields stable or higher average scores compared to the standard LLaVA training recipe. For instance, Qwen2.5-14B sees its General average rise from 75.6 to 77.4, while its Knowledge and OCR averages experience parallel uplifts. This indicates that isolating the alignment to a strategic subset of attention heads successfully mitigates catastrophic interference with the main language modeling objective.

The most substantial impact of HeRA is observed in the Vision-Centric benchmarks, which directly measure visual perception, spatial reasoning, and multimodal grounding. Regardless of the underlying architecture or its release date, our method systematically drives up visual performance. Earlier models like Vicuna-7B experience a robust +2.3 point gain, proving that proper representation alignment can benefit legacy architectures. Simultaneously, modern models equipped with stronger text priors also reap significant benefits; notably, Qwen3-8B achieves the highest individual leap with a +3.6 average improvement.

This trend persists as we scale the LLM backbone. When applied to the largest models in our suite, the representation alignment remains highly effective, with Qwen2.5-14B securing a +3.4 point increase and Qwen3-14B pushing the upper bound of the Vision-Centric average to 58.9.

Results on Hallucination Benchmarks. Although mitigating hallucinations, an open problem in MLLMs, is not explicitly enforced by our contrastive representation alignment loss, we find that HeRA has a positive effect on it, as outlined in Table 3. Across both the CHAIR-MSCOCO and AMBER generative benchmarks, models trained with HeRA consistently lower their hallucination rates (e.g., CHAIR<sub>s</sub> and CHAIR<sub>i</sub>). Crucially, on AMBER, this reduction in hallucinations is achieved while simultaneously improving, or at least maintaining, the cognition (Cog) score, which is a challenging balance, as models often become overly conservative when penalized for hallucinations.

![](images/bf39c807f64deb9eb7dc8cb8257ca74c17c4fa2b240d8c3cab36908d52f56b24.jpg)  
Baseline DINOv2-B DINOv2-L DINOv2-g DINOv3-B DINOv3-L SigLIP2  
Figure 4: VQA results of HeRA with different teacher vision encoders.

In discriminative settings, HeRA yields steady improvements in accuracy and F1 scores on AMBER, indicating a more robust visual grounding. On HallusionBench, the method drives positive gains across nearly all models, significantly improving qAcc, Easy, and Hard metrics. The sole exception is Qwen3-4B, with a drop on this specific benchmark; however, this is vastly offset by the superior performance gains across standard VQA and Vision-Centric benchmarks (as detailed in Table 2). Ultimately, explicitly aligning LLM internal representations with the visual domain naturally curbs the tendency to over-rely on linguistic priors, resulting in more faithful vision-language generations.

Comparison with Previous Representation Alignment Methods. In Table 4, we compare HeRA against recent representation alignment strategies: ROSS [38] trains an auxiliary denoiser network conditioned on LLM visual features to recover visual tokens; VIRAL [46] aligns visual features from the LLM middle layer during the instruction tuning stage; JARVIS [2] reconstructs masked image latents using representations from one quarter of the LLM depth; and CMAR [7] optimizes the CKA alignment metric between textual features from the penultimate LLM layer and the teacher encoder. We run each experiment according to its official implementation. All methods are trained on the same LLaVA [19] dataset, feature Qwen3-8B [1] as the LLM, SigLIP2 ViT-SO400M/14@384 [37] as the vision encoder, and DINOv2-L [28] as the teacher for alignment (with the exception of ROSS).

Compared to these fixed-layer approaches, our targeted head-wise alignment proves significantly more effective. Notably, the strict pointwise feature alignment operated by VIRAL is the only method that registers a regression compared to LLaVA (first row). Furthermore, while CMAR shares our goal of topological alignment between spaces of different modalities, the CKA metric forces global point-wise relationships to match those from vision encoder. By contrast, HeRA focuses strictly on preserving local neighborhood structures, without imposing rigid distance constraints between samples. Ultimately, on the demanding Vision-Centric benchmarks, HeRA yields a +3.6 point average improvement, outperforming the next best method, JARVIS (+2.8), and achieves the highest overall scores across the General, Knowledge, and OCR tasks. Full detailed results for each category are provided in Appendix C.

## 4.4 Varying the Teacher Visual Encoder

In Fig. 4, we experiment with different teacher vision encoders used to extract the targets for the HeRA contrastive loss. All models are trained using Qwen3-8B as the language backbone and SigLIP2 as the primary vision encoder. We observe that using SigLIP2 itself as the teacher is mostly ineffective, in accordance with concurrent work [46, 47] showing that unsupervised vision encoders are better representation teachers than encoders trained with language supervision. Indeed, aligning with DINO-based [28, 32] teachers yields strong and consistent gains, even when using the base models (DINOv2-B and DINOv3-B). However, we note no clear benefits from employing the larger 1B-parameter DINOv2-g, suggesting that base vision encoders may suffice for topological alignment.

## 5 Conclusion

In this work, we introduced Head-Wise Representation Alignment (HeRA), a novel method to enhance Multimodal Large Language Models through topological representation alignment. Guided by the Platonic Representation Hypothesis, HeRA uses a contrastive proxy for the MKNN metric to align specific attention heads with an external vision encoder, demonstrating that targeting the least aligned heads yields the most substantial gains. Evaluations across multiple architectures and benchmarks reveal that our approach significantly benefits demanding vision-centric tasks without compromising, even improving, core linguistic capabilities, while mitigating visual hallucinations.

## Acknowledgments

This work has been supported by the EU Horizon project “ELLIOT” (No. 101214398), by the EuroHPC JU project “MINERVA” (GA No. 101182737), and by the PNRR project “ITSERR” (CUP B53C22001770006) funded by the EU - NextGenerationEU. We also acknowledge EuroHPC JU for awarding the project EHPC-AIF-2025SC04-225 access to LUMI at CSC, Finland.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631, 2025.

[2] Davide Caffagni, Sara Sarto, Marcella Cornia, Lorenzo Baraldi, Pier Luigi Dovesi, Shaghayegh Roohi, Mark Granroth-Wilding, and Rita Cucchiara. Seeing Beyond Words: Self-Supervised Visual Learning for Multimodal Large Language Models. arXiv preprint arXiv:2512.15885, 2025.

[3] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%\* ChatGPT Quality, 2023.

[4] David Fan, Shengbang Tong, Jiachen Zhu, Koustuv Sinha, Zhuang Liu, Xinlei Chen, Michael Rabbat, Nicolas Ballas, Yann LeCun, Amir Bar, et al. Scaling Language-Free Visual Represen tation Learning. In ICCV, 2025.

[5] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv preprint arXiv:2306.13394, 2023.

[6] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. BLINK: Multimodal Large Language Models Can See But Not Perceive. In ECCV, 2024.

[7] Yulu Gan, Kaiya Ivy Zhao, and Phillip Isola. Cross-Modal Alignment Regularization: Enhanc ing Language Models with Vision Model Representations. In ICLR Workshops, 2025.

[8] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783, 2024.

[9] Fabian Gröger, Shuo Wen, and Maria Brbic. Revisiting the Platonic Representation Hypothesis:´ An Aristotelian View. In ICML, 2026.

[10] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models. In CVPR, 2024.

[11] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. VizWiz Grand Challenge: Answering Visual Questions From Blind People. In CVPR, 2018.

[12] Drew A Hudson and Christopher D Manning. GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. In CVPR, 2019.

[13] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. The Platonic Representation Hypothesis. In ICML, 2024.

[14] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, 2016.

[15] Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers. In ICCV, 2025.

[16] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension. arXiv preprint arXiv:2307.16125, 2023.

[17] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating Object Hallucination in Large Vision-Language Models. In EMNLP, 2023.

[18] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common Objects in Context. In ECCV, 2014.

[19] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved Baselines with Visual Instruction Tuning. In CVPR, 2024.

[20] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. MMBench: Is Your Multi-modal Model an All-around Player? In ECCV, 2024.

[21] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models. Sci China Inf Sci, 67(12):220102, 2024.

[22] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts. In ICLR, 2024.

[23] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering. In NeurIPS, 2022.

[24] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In ACL, 2022.

[25] Andrew Joohun Nam, Henry Conklin, Yukang Yang, Thomas L Griffiths, Jonathan D Cohen, and Sarah-Jane Leslie. Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers. In NeurIPS, 2025.

[26] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-Context Learning and Induction Heads. arXiv preprint arXiv:2209.11895, 2022.

[27] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation Learning with Contrastive Predictive Coding. arXiv preprint arXiv:1807.03748, 2018.

[28] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning Robust Visual Features without Supervision. TMLR, pages 1–31, 2024.

[29] Qwen Team. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115, 2024.

[30] Qwen Team. Qwen3.5: Towards Native Multimodal Agents, 2026.

[31] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M. Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S. Khan. GLaMM: Pixel Grounding Large Multimodal Model. In CVPR, 2024.

[32] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. DINOv3. arXiv preprint arXiv:2508.10104, 2025.

[33] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA Models That Can Read. In CVPR, 2019.

[34] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai C Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs. In NeurIPS, 2024.

[35] Shengbang Tong, David Fan, John Nguyen, Ellis Brown, Gaoyue Zhou, Shengyi Qian, Boyang Zheng, Théophane Vallaeys, Junlin Han, Rob Fergus, et al. Beyond Language Modeling: An Exploration of Multimodal Pretraining. arXiv preprint arXiv:2603.03276, 2026.

[36] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs. In CVPR, 2024.

[37] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. arXiv preprint arXiv:2502.14786, 2025.

[38] Haochen Wang, Anlin Zheng, Yucheng Zhao, Tiancai Wang, Zheng Ge, Xiangyu Zhang, and Zhaoxiang Zhang. Reconstructive Visual Instruction Tuning. In ICLR, 2025.

[39] Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Jiaqi Wang, Haiyang Xu, Ming Yan, Ji Zhang, et al. AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation. arXiv preprint arXiv:2311.07397, 2023.

[40] Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small. In ICLR, 2023.

[41] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency. arXiv preprint arXiv:2508.18265, 2025.

[42] Luis Wiedmann, Orr Zohar, Amir Mahla, Xiaohan Wang, Rui Li, Thibaud Frere, Leandro von Werra, Aritra Roy Gosthipaty, and Andrés Marafioti. FineVision: Open Data Is All You Need. arXiv preprint arXiv:2510.17269, 2025.

[43] Penghao Wu and Saining Xie. V\*: Guided Visual Search as a Core Mechanism in Multimodal LLMs. In CVPR, 2024.

[44] xAI. Grok, 2024.

[45] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388, 2025.

[46] Heeji Yoon, Jaewoo Jung, Junwan Kim, Hyungyu Choi, Heeseong Shin, Sangbeom Lim, Honggyu An, Chaehyun Kim, Jisang Han, Donghyun Kim, et al. Visual Representation Alignment for Multimodal Large Language Models. arXiv preprint arXiv:2509.07979, 2025.

[47] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think. In ICLR, 2025.

[48] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. In CVPR, 2024.

[49] Zihao Yue, Liang Zhang, and Qin Jin. Less is More: Mitigating Multimodal Hallucination from an EOS Decision Perspective. In ACL, 2024.

## A Additional Implementation Details

Training Details. Following the two-stage training recipe of LLaVA-1.5 [19], in the first stage, we train only the projector proj, a two-layer MLP, using 558k image-caption pairs, while keeping the language model G frozen. In the second stage, we jointly optimize G and proj for visual instruction tuning on the LLaVA-Instruct-665k dataset. The training settings $( e . g .$ , optimizer, learning rate, and batch size) are kept identical to LLaVA-1.5. All experiments are conducted on AMD MI250x devices, each of which comprises 2 GPUs with 64GB of VRAM. The first training stage runs on 16 GPUs for up to 6 hours, depending on the size of the LLM. The second training stage runs on 32 GPUs, up to 14 hours. We find no noticeable difference in training time with the addition of $\mathcal { L } _ { \mathrm { H e R A } }$ . The MKNN alignment scores can be efficiently computed offline. For instance, for a 7B LLM and DINOv2-L, it takes less than one hour on a single GPU.

Contrastive Learning Details. To generate supervision signals for each batch, we extract the [CLS] token representations from the teacher vision encoder V<sup>t</sup>. We then compute the pairwise dot products between these representations to form a similarity matrix (i.e., a Gram matrix), which captures the visual neighborhood structure of the batch. For every sample, we identify its top-k nearest neighbors to construct multi-positive contrastive targets. This is achieved by assigning a uniform probability of $\frac { 1 } { k }$ to these k neighbors, and a probability of zero to all other samples.

For the student model, we extract the head-wise representations from the selected set of heads (H) and apply average pooling across the embeddings corresponding to text tokens. In the first training stage, text tokens correspond to the caption of the input image, and thus we employ all of them. In the second training stage, text tokens represent multi-turn dialogs, and we pool exclusively over the tokens pertaining to the <ASSISTANT> turn. These are the same tokens contributing to the language modeling loss of Eq. 1.

The temperature parameter τ of Eq. 4 is learned in logarithmic scale, and it is initialized as 0.07.

LLM Details and Selected Heads. We collect in Table 5 the exact checkpoints of each LLM used in this work. All of them are publicly accessible on the Hugging Face Hub. We also report the specific attention heads of each LLM used to compute $\mathcal { L } _ { \mathrm { H e R A } }$ , sorted from left to right by increasing value of the MKNN alignment score. We indicate with LXHY the index of the Y-th head in layer X.

Table 5: Checkpoint reference and list of selected heads for each LLM.

<table><tr><td rowspan="2">LLM</td><td rowspan="2">Hugging Face Page</td><td colspan="5">Selected Heads (H)</td></tr><tr><td>H1</td><td>H2</td><td>H3</td><td>H4</td><td>H5</td></tr><tr><td>Qwen2.5-3B [29]</td><td>Qwen/Qwen2.5-3B-Instruct</td><td>L3H4</td><td>L5H11</td><td>L5H14</td><td>L5H9</td><td>L5H15</td></tr><tr><td>Qwen3-4B [45]</td><td>Qwen/Qwen3-4B-Instruct-2507</td><td>L7H6</td><td>L13H12</td><td>L3H6</td><td>L3H5</td><td>L13H13</td></tr><tr><td>Vicuna-7B [3]</td><td>lmsys/vicuna-7b-v1.5</td><td>L0H14</td><td>L1H1</td><td>L0H1</td><td>L0H3</td><td>L0H30</td></tr><tr><td>Llama3-8B [8]</td><td>meta-llama/Meta-Llama-3-8B-Instruct</td><td>L0H31</td><td>L2H21</td><td>L0H29</td><td>L2H23</td><td>L2H25</td></tr><tr><td>Qwen2.5-7B [29]</td><td>Qwen/Qwen2.5-7B-Instruct</td><td>L14H5</td><td>L4H7</td><td>L14H4</td><td>L4H10</td><td>L4H13</td></tr><tr><td>Qwen3-8B [45]</td><td>Qwen/Qwen3-8B</td><td>L13H12</td><td>L7H6</td><td>L3H5</td><td>L13H13</td><td>L13H15</td></tr><tr><td>Vicuna-13B [3]</td><td>lmsys/vicuna-13b-v1.5</td><td>L0H38</td><td>L1H0</td><td>L0H11</td><td>L0H10</td><td>L0H26</td></tr><tr><td>Qwen2.5-14B [29]</td><td>Qwen/Qwen2.5-14B-Instruct</td><td>L13H27</td><td>L18H13</td><td>L0H9</td><td>L0H7</td><td>L3H8</td></tr><tr><td>Qwen3-14B [45]</td><td>Qwen/Qwen3-14B</td><td>L14H24</td><td>L9H36</td><td>L9H32</td><td>L1H37</td><td>L1H36</td></tr></table>

## B Evaluation Benchmarks

Cambrian Evaluation Suite $[ 3 4 ] . ^ { 2 }$ It comprises a comprehensive suite of benchmarks designed to evaluate diverse capabilities of MLLMs, including general perception, knowledge reasoning, OCR and chart understanding, and core visual abilities. Accordingly, the benchmarks are grouped into four categories: General, Knowledge, OCR, and Vision. In our experiments, we consider 18 benchmarks: GQA [12], POPE [17], MME [5], MMBench (MMB) [20], and SEED-Bench (SEED) [16] for the General category; ScienceQA (SQA) [23], MMMU [48], MathVista [22], and AI2D [14] for Knowledge; ChartQA [24], OCRBench (OCRB) [21], TextVQA [33], and VizWiz [11] for OCR; and MMVP [36], RealWorldQA (RWQA) [44], Blink [6], V\* [43], and CVBench [34] for Vision. When reporting averages, we normalize the MME score by dividing it by 20 to ensure consistency with the scale of the other benchmarks.

Table 6: VQA comparison between DINOv2 and SigLIP2 as the vision encoder for LLaVA.

<table><tr><td rowspan="2">LLM</td><td rowspan="2">Vision Encoder</td><td>General</td><td>Knowledge</td><td>OCR</td><td colspan="6">Vision</td></tr><tr><td>Avg</td><td>Avg</td><td>Avg</td><td>RWQA</td><td>MMVP</td><td>Blink</td><td>V*</td><td>CVBench</td><td>Avg</td></tr><tr><td>Qwen2.5-3B</td><td>DINOv2</td><td>67.1</td><td>43.3</td><td>26.3</td><td>51.5</td><td>30.0</td><td>47.8</td><td>46.1</td><td>58.0</td><td>46.7</td></tr><tr><td>Qwen2.5-3B</td><td>SigLIP2</td><td>73.5</td><td>46.7</td><td>42.2</td><td>55.2</td><td>46.0</td><td>46.8</td><td>44.5</td><td>60.2</td><td>50.5</td></tr><tr><td>+ HeRA (Ours)</td><td>SigLIP2</td><td>74.5</td><td>47.5</td><td>43.8</td><td>56.3</td><td>48.0</td><td>49.1</td><td>51.3</td><td>59.6</td><td>52.9</td></tr><tr><td>Qwen3-4B</td><td>DINOv2</td><td>70.5</td><td>46.1</td><td>24.2</td><td>54.9</td><td>38.7</td><td>49.2</td><td>49.2</td><td>63.1</td><td>51.0</td></tr><tr><td>Qwen3-4B</td><td>SigLIP2</td><td>75.6</td><td>49.6</td><td>43.8</td><td>59.9</td><td>48.0</td><td>55.1</td><td>50.3</td><td>68.3</td><td>56.3</td></tr><tr><td>+ HeRA (Ours)</td><td>SigLIP2</td><td>76.0</td><td>50.1</td><td>44.5</td><td>61.3</td><td>56.0</td><td>53.7</td><td>52.4</td><td>69.1</td><td>58.5</td></tr></table>

Table 7: Effects of $\mathcal { L } _ { \mathrm { H e R A } }$ when applied to the different training stages of LLaVA.

<table><tr><td rowspan="2">Model</td><td colspan="2"> $\mathcal{L}_{\text{HeRA}}$ </td><td>General</td><td>Knowledge</td><td>OCR</td><td colspan="6">Vision</td></tr><tr><td>St.1</td><td>St.2</td><td>Avg</td><td>Avg</td><td>Avg</td><td>RWQA</td><td>MMVP</td><td>Blink</td><td>V*</td><td>CVBench</td><td>Avg</td></tr><tr><td>Qwen2.5-3B</td><td>-</td><td>-</td><td>73.5</td><td>46.7</td><td>42.2</td><td>55.2</td><td>46.0</td><td>46.8</td><td>44.5</td><td>60.2</td><td>50.5</td></tr><tr><td>+ HeRA</td><td>√</td><td>-</td><td>74.2</td><td>47.5</td><td>44.2</td><td>57.4</td><td>44.0</td><td>46.9</td><td>51.3</td><td>60.7</td><td>52.1</td></tr><tr><td>+ HeRA</td><td>-</td><td>√</td><td>73.5</td><td>46.4</td><td>43.0</td><td>54.5</td><td>40.0</td><td>48.8</td><td>44.5</td><td>60.5</td><td>49.7</td></tr><tr><td>+ HeRA (Ours)</td><td>√</td><td>√</td><td>74.5</td><td>47.5</td><td>43.8</td><td>56.3</td><td>48.0</td><td>49.1</td><td>51.3</td><td>59.6</td><td>52.9</td></tr><tr><td>Qwen3-4B</td><td>-</td><td>-</td><td>75.6</td><td>49.6</td><td>43.8</td><td>59.9</td><td>48.0</td><td>55.1</td><td>50.3</td><td>68.3</td><td>56.3</td></tr><tr><td>+ HeRA</td><td>√</td><td>-</td><td>75.6</td><td>49.7</td><td>43.4</td><td>60.7</td><td>51.3</td><td>53.3</td><td>46.1</td><td>66.2</td><td>55.5</td></tr><tr><td>+ HeRA</td><td>-</td><td>√</td><td>75.7</td><td>50.0</td><td>44.6</td><td>59.6</td><td>54.7</td><td>52.7</td><td>49.2</td><td>67.8</td><td>56.8</td></tr><tr><td>+ HeRA (Ours)</td><td>√</td><td>√</td><td>76.0</td><td>50.1</td><td>44.5</td><td>61.3</td><td>56.0</td><td>53.7</td><td>52.4</td><td>69.1</td><td>58.5</td></tr></table>

Hallucination Datasets. We evaluate hallucinatory tendencies on three widely used benchmarks: AMBER [39], CHAIR-MSCOCO [49], and HallusionBench [10]. CHAIR-MSCOCO measures object- and sentence-level hallucination rates (i.e., CHAIR and CHAIR ) on model-generated descriptions for 500 images sampled from the MSCOCO [18] validation set. The AMBER generative task further introduces cognition (Cog), which quantifies the overlap between model- and humanhallucinated objects, and coverage (Cover), which measures object-level recall. Complementarily, the AMBER discriminative task captures a broader set of hallucination types, including attribute and relation hallucinations in addition to object existence, using a ground truth set of 1,004 manually annotated images. For CHAIR-MSCOCO and the AMBER generative task, we use a maximum generation length of 512 tokens with greedy decoding. For the AMBER discriminative task, we append the instruction “Answer only with Yes or No. Use exactly one word. Do not use commas, periods, or symbols.” to each query to enforce binary (Yes/No) responses. We evaluate hallucination robustness on HallusionBench [10] using an exact-match protocol. Specifically, we force the model to output unambiguous responses by appending the following instruction to each query: “Answer the question using a single word or phrase: Yes or No.” We report three standard metrics. qAcc (Question Pair Accuracy) measures group-level consistency. A prediction is counted as correct under qAcc only if the model answers all questions within the same group correctly. In addition, we report Easy accuracy, computed over unmodified questions, and Hard accuracy, computed over adversarially modified or misleading variants designed to induce hallucinations.

## C Additional Experiments

## C.1 Additional Ablation Studies and Results

DINOv2 as Vision Encoder. In this work, we demonstrate the effectiveness of leveraging a teacher vision encoder, e.g., DINOv2-L [28], as a source of supervision for topological representation alignment. It is natural to ask what if we do not perform representation alignment at all, by directly plugging in DINOv2-L as the vision encoder of an MLLM. Table 6 answers this question by comparing DINOv2-L vs SigLIP2 [37], and clearly demonstrates that DINOv2-L is ineffective on its own as a vision encoder for MLLMs. For fair comparison, we feed DINOv2-L with the same image resolution of 384 × 384 pixels as SigLIP2. Despite that, DINOv2-L suffers from severe deficits, especially on OCR tasks. These results agree with prior works [4, 34] showing that unsupervised visual encoders alone fall short against language-supervised encoders on VQA benchmarks. On the other hand, as testified by Fig.4, language-supervised encoders are unsuitable as representation teachers, and that justifies the application of representation alignment methods such as HeRA, where MLLMs benefit from the synergistic effects of language-supervised and unsupervised visual encoders.

Table 8: Detailed VQA results of HeRA applied to the LLaVA training recipe on different LLMs.

<table><tr><td rowspan="2"></td><td colspan="6">General</td><td colspan="5">Knowledge</td><td colspan="5">OCR</td><td colspan="6">Vision</td></tr><tr><td>Avg</td><td>GQA</td><td>POPE</td><td>MME</td><td>MMB</td><td>SEED</td><td>Avg</td><td>SQA</td><td>MMMU</td><td>MathV</td><td>AIZD</td><td>Avg</td><td>CharQA</td><td>OCRB</td><td>TextVQA</td><td>VizWiz</td><td>Avg</td><td>MMVP</td><td>RWQA</td><td>Blink</td><td>V*</td><td>CVBench</td></tr><tr><td>Qwen2.5-3B</td><td>73.5</td><td>63.8</td><td>87.7</td><td>1484.0</td><td>70.9</td><td>71.2</td><td>46.7</td><td>75.4</td><td>40.9</td><td>7.7</td><td>62.9</td><td>42.2</td><td>17.3</td><td>37.1</td><td>60.6</td><td>53.8</td><td>50.5</td><td>46.0</td><td>55.2</td><td>46.8</td><td>44.5</td><td>60.2</td></tr><tr><td>+ HeRA (Ours)</td><td>74.5</td><td>64.1</td><td>87.9</td><td>1525.5</td><td>72.0</td><td>72.0</td><td>47.5</td><td>75.5</td><td>41.3</td><td>7.9</td><td>65.1</td><td>43.8</td><td>20.2</td><td>39.1</td><td>61.7</td><td>54.3</td><td>52.9</td><td>48.0</td><td>56.3</td><td>49.1</td><td>51.3</td><td>59.6</td></tr><tr><td>Qwen3-4B</td><td>75.6</td><td>64.6</td><td>87.3</td><td>1528.4</td><td>75.4</td><td>74.1</td><td>49.6</td><td>77.2</td><td>44.6</td><td>8.4</td><td>68.4</td><td>43.8</td><td>23.3</td><td>42.5</td><td>65.6</td><td>44.0</td><td>56.3</td><td>48.0</td><td>59.9</td><td>55.1</td><td>50.3</td><td>68.3</td></tr><tr><td>+ HeRA (Ours)</td><td>76.0</td><td>64.9</td><td>87.9</td><td>1544.6</td><td>75.6</td><td>74.3</td><td>50.1</td><td>78.8</td><td>44.7</td><td>8.1</td><td>68.9</td><td>44.5</td><td>24.8</td><td>40.5</td><td>65.6</td><td>47.2</td><td>58.5</td><td>56.0</td><td>61.3</td><td>53.7</td><td>52.4</td><td>69.1</td></tr><tr><td>Vicuna-7B</td><td>72.2</td><td>64.9</td><td>87.5</td><td>1469.0</td><td>64.6</td><td>70.8</td><td>44.3</td><td>70.3</td><td>36.3</td><td>11.1</td><td>59.4</td><td>45.7</td><td>20.0</td><td>40.6</td><td>64.3</td><td>58.0</td><td>49.7</td><td>38.7</td><td>56.5</td><td>46.8</td><td>44.5</td><td>62.1</td></tr><tr><td>+ HeRA (Ours)</td><td>72.1</td><td>65.1</td><td>87.8</td><td>1453.4</td><td>64.0</td><td>70.9</td><td>44.5</td><td>70.9</td><td>35.0</td><td>12.4</td><td>59.5</td><td>45.7</td><td>20.6</td><td>40.3</td><td>64.4</td><td>57.6</td><td>52.0</td><td>42.7</td><td>57.8</td><td>47.9</td><td>49.7</td><td>61.9</td></tr><tr><td>LLama3-8B</td><td>73.3</td><td>64.8</td><td>87.5</td><td>1506.0</td><td>67.4</td><td>71.5</td><td>45.0</td><td>72.5</td><td>37.9</td><td>7.2</td><td>62.5</td><td>43.0</td><td>18.8</td><td>40.1</td><td>63.8</td><td>49.5</td><td>53.8</td><td>46.0</td><td>60.1</td><td>49.2</td><td>44.0</td><td>69.5</td></tr><tr><td>+ HeRA (Ours)</td><td>74.6</td><td>65.6</td><td>87.6</td><td>1503.3</td><td>71.7</td><td>72.7</td><td>46.3</td><td>75.9</td><td>38.0</td><td>8.4</td><td>63.1</td><td>44.7</td><td>19.9</td><td>39.9</td><td>64.6</td><td>54.2</td><td>55.1</td><td>46.7</td><td>60.4</td><td>50.2</td><td>51.8</td><td>66.4</td></tr><tr><td>Qwen2.5-7B</td><td>76.2</td><td>64.9</td><td>88.2</td><td>1582.2</td><td>75.3</td><td>73.7</td><td>50.2</td><td>77.9</td><td>45.1</td><td>8.8</td><td>68.8</td><td>47.9</td><td>24.2</td><td>39.9</td><td>64.9</td><td>62.5</td><td>56.7</td><td>51.3</td><td>59.6</td><td>51.7</td><td>50.8</td><td>70.2</td></tr><tr><td>+ HeRA (Ours)</td><td>76.5</td><td>65.3</td><td>88.3</td><td>1574.8</td><td>76.0</td><td>74.1</td><td>50.5</td><td>77.9</td><td>46.9</td><td>7.6</td><td>69.6</td><td>48.6</td><td>23.9</td><td>40.7</td><td>65.6</td><td>64.1</td><td>57.4</td><td>54.0</td><td>61.3</td><td>50.2</td><td>50.3</td><td>71.3</td></tr><tr><td>Qwen3-8B</td><td>74.7</td><td>64.8</td><td>86.5</td><td>1472.8</td><td>75.4</td><td>73.1</td><td>49.7</td><td>77.5</td><td>47.0</td><td>7.3</td><td>67.2</td><td>43.5</td><td>20.2</td><td>37.4</td><td>64.3</td><td>51.9</td><td>55.9</td><td>49.3</td><td>59.2</td><td>52.2</td><td>50.8</td><td>67.8</td></tr><tr><td>+ HeRA (Ours)</td><td>76.9</td><td>66.1</td><td>87.6</td><td>1562.3</td><td>77.6</td><td>74.9</td><td>51.1</td><td>78.8</td><td>46.2</td><td>8.6</td><td>70.8</td><td>47.6</td><td>27.6</td><td>41.4</td><td>67.9</td><td>53.6</td><td>59.5</td><td>58.0</td><td>60.1</td><td>55.5</td><td>50.3</td><td>73.5</td></tr><tr><td>Vicuna-13B</td><td>73.4</td><td>65.5</td><td>87.7</td><td>1491.9</td><td>67.1</td><td>72.2</td><td>45.5</td><td>72.0</td><td>38.0</td><td>9.9</td><td>62.0</td><td>47.7</td><td>22.2</td><td>41.6</td><td>67.2</td><td>59.5</td><td>52.7</td><td>44.7</td><td>58.7</td><td>50.6</td><td>46.1</td><td>63.7</td></tr><tr><td>+ HeRA (Ours)</td><td>73.6</td><td>65.8</td><td>88.0</td><td>1504.7</td><td>67.2</td><td>71.9</td><td>45.7</td><td>72.2</td><td>36.6</td><td>13.2</td><td>60.8</td><td>47.6</td><td>22.0</td><td>41.8</td><td>67.1</td><td>59.4</td><td>53.9</td><td>44.0</td><td>57.3</td><td>52.3</td><td>49.2</td><td>66.5</td></tr><tr><td>Qwen2.5-14B</td><td>75.6</td><td>64.2</td><td>88.0</td><td>1548.8</td><td>75.5</td><td>72.7</td><td>50.7</td><td>76.4</td><td>48.2</td><td>8.3</td><td>69.7</td><td>44.8</td><td>19.6</td><td>33.9</td><td>63.7</td><td>61.9</td><td>54.9</td><td>47.3</td><td>60.1</td><td>52.7</td><td>46.6</td><td>67.9</td></tr><tr><td>+ HeRA (Ours)</td><td>77.4</td><td>66.1</td><td>88.0</td><td>1600.9</td><td>77.5</td><td>75.5</td><td>52.8</td><td>78.6</td><td>49.1</td><td>9.6</td><td>74.1</td><td>49.3</td><td>27.4</td><td>39.0</td><td>67.9</td><td>62.9</td><td>58.3</td><td>52.0</td><td>60.1</td><td>55.2</td><td>52.4</td><td>71.6</td></tr><tr><td>Qwen3-14B</td><td>77.4</td><td>65.6</td><td>87.6</td><td>1609.5</td><td>78.5</td><td>74.6</td><td>52.8</td><td>79.6</td><td>49.6</td><td>10.2</td><td>71.9</td><td>46.1</td><td>26.8</td><td>41.1</td><td>69.7</td><td>47.0</td><td>58.2</td><td>57.3</td><td>60.3</td><td>52.6</td><td>50.3</td><td>70.8</td></tr><tr><td>+ HeRA (Ours)</td><td>77.7</td><td>66.3</td><td>88.0</td><td>1632.5</td><td>78.0</td><td>74.9</td><td>52.6</td><td>79.4</td><td>49.1</td><td>9.2</td><td>72.6</td><td>47.8</td><td>28.1</td><td>41.8</td><td>69.9</td><td>51.4</td><td>58.9</td><td>58.0</td><td>62.5</td><td>52.0</td><td>51.8</td><td>70.2</td></tr></table>

Table 9: Detailed VQA results of different representation alignment strategies for MLLMs.

<table><tr><td rowspan="2">Alignment</td><td colspan="6">General</td><td colspan="5">Knowledge</td><td colspan="5">OCR</td><td colspan="6">Vision</td></tr><tr><td> $Avg$ </td><td>GQA</td><td>POPE</td><td>MME</td><td>MMB</td><td>SEED</td><td> $Avg$ </td><td>SQA</td><td>MMMU</td><td>MathV</td><td>A12D</td><td> $Avg$ </td><td>ChartQA</td><td>OCRB</td><td>TextVQA</td><td>VizWiz</td><td> $Avg$ </td><td>RWQA</td><td>MMVP</td><td>Blink</td><td>V*</td><td>CVBench</td></tr><tr><td>-</td><td>74.7</td><td>64.8</td><td>86.5</td><td>1472.7</td><td>75.4</td><td>73.1</td><td>49.7</td><td>77.5</td><td>47.0</td><td>7.3</td><td>67.2</td><td>43.5</td><td>20.2</td><td>37.4</td><td>64.3</td><td>51.9</td><td>55.9</td><td>59.2</td><td>49.3</td><td>52.2</td><td>50.8</td><td>67.8</td></tr><tr><td>ROSS [38]</td><td>74.6</td><td>64.5</td><td>87.2</td><td>1472.5</td><td>74.9</td><td>72.7</td><td>49.4</td><td>76.4</td><td>46.3</td><td>7.4</td><td>67.4</td><td>44.0</td><td>17.8</td><td>36.8</td><td>64.8</td><td>56.8</td><td>56.2</td><td>59.8</td><td>50.3</td><td>52.2</td><td>49.2</td><td>69.7</td></tr><tr><td>VIRAL [46]</td><td>73.8</td><td>64.2</td><td>87.3</td><td>1396.6</td><td>74.9</td><td>72.6</td><td>49.3</td><td>76.4</td><td>45.7</td><td>8.0</td><td>67.1</td><td>43.6</td><td>19.3</td><td>36.7</td><td>64.2</td><td>54.2</td><td>54.2</td><td>57.6</td><td>46.7</td><td>51.3</td><td>46.1</td><td>69.3</td></tr><tr><td>JARVIS [2]</td><td>76.8</td><td>64.6</td><td>88.2</td><td>1605.1</td><td>76.7</td><td>74.2</td><td>49.9</td><td>77.0</td><td>45.4</td><td>8.4</td><td>68.6</td><td>46.2</td><td>27.1</td><td>40.5</td><td>66.0</td><td>51.2</td><td>58.7</td><td>59.2</td><td>54.7</td><td>54.2</td><td>55.5</td><td>69.9</td></tr><tr><td>CMAR [7]</td><td>76.4</td><td>65.6</td><td>87.4</td><td>1556.6</td><td>76.5</td><td>74.7</td><td>51.0</td><td>78.3</td><td>45.9</td><td>8.6</td><td>71.0</td><td>46.0</td><td>23.3</td><td>41.6</td><td>67.6</td><td>51.6</td><td>56.9</td><td>58.8</td><td>50.0</td><td>52.1</td><td>52.9</td><td>70.9</td></tr><tr><td>HeRA (Ours)</td><td>76.9</td><td>66.1</td><td>87.6</td><td>1562.3</td><td>77.6</td><td>74.9</td><td>51.1</td><td>78.8</td><td>46.2</td><td>8.6</td><td>70.8</td><td>47.6</td><td>27.6</td><td>41.4</td><td>67.9</td><td>53.6</td><td>59.5</td><td>60.1</td><td>58.0</td><td>55.5</td><td>50.3</td><td>73.5</td></tr></table>

HeRA in Different Training Stages. We seamlessly apply the $\mathcal { L } _ { \mathrm { H e R A } }$ on both training stage of LLaVA. However, there are neat differences between them that are worth discussing. For instance, in the first training stage (i.e., St.1), the MLLM is fed with images and their related captions, which represent aligned image-text pairs, i.e., the same concept is expressed in two different modalities. This appears to be a suitable stage for representation alignment, as the student MLLM and the teacher vision encoder process the same underlying concepts. Conversely, image-text pairs in the second training stage (i.e., St.2) are not aligned the same way: the text corresponds to a multi-turn dialog between a user and the assistant, which focuses on the image, but does not exactly mimic the visual content as an image caption. With that in mind, if one had to select a single training stage for $\mathcal { L } _ { \mathrm { H e R A } }$ we would expect a larger impact on St.1 rather than St.2. However, according to Table 7, that is not always the case: with Qwen2.5-3B, $\mathcal { L } _ { \mathrm { H e R A } }$ helps more when applied during St.1, while the opposite holds with Qwen3-4B. Ultimately, our original proposal of regularizing both training stages with $\mathcal { L } _ { \mathrm { H e R A } }$ works best on both models.

Extended Results on All Benchmarks. As we aim to improve MLLMs, we focus on strengthening their visual perception, which is particularly stressed on vision-centric benchmarks. Consequently, in the main paper, we reported detailed scores on specific vision-centric VQA datasets, such as Real-WorldQA, MMVP, Blink, V\*, and CVBench, leaving the average score on the General, Knowledge, and OCR categories. Here, we report the full results over the 18 VQA benchmarks considered in our study. Specifically, we refer to Table 8 for the results of different LLM families, and to Table 9 for a detailed comparison between representation alignment methods on the Qwen3-8B LLM.

## C.2 Additional Analyses

Additional MKNN Head-Wise Analysis. In Fig. 5, we provide a detailed comparison of the MKNN alignment scores for the Worst-5 (upper half) and Top-5 (bottom half) attention heads across different training strategies. These specific heads are identified by computing their MKNN alignment scores on the base Qwen2.5-3B LLM prior to any multimodal training. Interestingly, we observe that the relative alignment of these heads is largely preserved after standard multimodal training (i.e.,

![](images/5266c38f25ca72e85f95658428f21c0924a9e58c35ad309bbb306ffef36302ec.jpg)  
Figure 5: Effect of the multimodal training of LLaVA and representation alignment methods on the Worst-5 and Top-5 heads. Worst-5 and Top-5 heads are selected by the lowest and highest MKNN alignment score with DINOv2-L, computed on the Qwen2.5-3B LLM before multimodal training.

LLaVA): heads that are naturally highly aligned in the base LLM remain highly aligned, whereas poorly aligned heads stay poorly aligned.

When we apply HeRA to the Top-5 heads, their alignment scores further increase, but this intervention has absolutely no impact on the Worst-5 heads. As shown in Tab. 1 (seventh row), this translates into suboptimal downstream performance. Conversely, our proposed strategy, that is HeRA applied to the Worst-5 heads, greatly increases the alignment of the targeted components. Crucially, this massive boost does not sacrifice the integrity of the Top-5 heads, which record MKNN alignment scores remarkably similar to the LLaVA baseline.

Finally, we observe that enforcing representation alignment at the feature level, specifically, via cosine similarity maximization between the MLLM visual features and the teacher vision encoder, is ineffective at modifying the local topological structure (nor at improving performance, see Table 1, third row). As depicted in the plot, this approach has a very small effect on the MKNN alignment scores of the targeted Worst-5 heads, further highlighting the unique contribution of our topologyaware contrastive objective.

MKNN Alignment With Larger Qwen2.5 Models. In Fig. 6 and Fig. 7, we extend the MKNN alignment analysis of Fig. 3 to larger LLMs within the same family, specifically evaluating Qwen2.5- 7B and Qwen2.5-14B against the DINOv2-L teacher. The left parts confirm that our previous observations persist at larger scales: representations of specific individual attention heads consistently exhibit a much higher natural alignment with the visual domain than those obtained out of any layer in the same model. Furthermore, the right parts highlight the atomic nature of our intervention. Applying HeRA to the Worst-5 heads successfully drives a massive boost in their cross-modal alignment, without disrupting the structural alignment of the Top-5 heads that are already naturally aligned in the base LLM.

MKNN Analysis with Different Representation Alignment Methods. In Fig. 8, we compare the MKNN alignment scores of HeRA against existing representation alignment strategies. For this analysis, all methods are trained using Qwen3-8B as the LLM and SigLIP2 as the vision encoder, with the MKNN alignment metric computed with respect to the DINOv2-L teacher. We remind to Table 4 for a quantitative comparison on VQA benchmarks.

Consistent with our previous findings, all methods increase the alignment of the Top-5 heads. However, this is largely a natural consequence of the multimodal training process itself, as most methods achieve scores that closely mirror the unregularized LLaVA baseline. The only method that registers a more significant, distinct impact on the Top-5 heads is CMAR. CMAR shares a conceptual similarity with HeRA in that it enforces cross-modal topological alignment rather than strict feature-level visual matching. However, a key difference lies in their scope: CMAR relies on the CKA metric to match global pairwise relationships across all samples within a training batch, whereas HeRA strictly targets the consistency of local neighborhoods.

![](images/8f151a287e39211b51c09be1d48347bb2180cb90bf83940c89bcc313f8af4ecb.jpg)

![](images/d6784a35a3c883d9d66f6df2018592e939974623538423aa1c88e8ecce1ce944.jpg)  
Worst-5 Heads MKNN Score

![](images/a15da0a40550b824ebfd0be0d2cf3e81b3c8fe53c98e4942e9096248b67090af.jpg)  
LLM LLaVA HeRA (Ours)  
Figure 6: Left: Alignment with DINOv2-L, measured with the MKNN metric on each layer and attention head of Qwen2.5-7B. Right: MKNN scores of the Worst-5 and Top-5 heads, computed on (i) the base LLM; (ii) after the LLaVA multimodal training; and (iii) after the addition of HeRA.

![](images/6396726931bdc399e05486a3c1e1fd98a9ecc3c7bbf9b1a786693265277b681d.jpg)

![](images/d48e643d4fe24c1464c87c878b812d74c3b3a38185dc60d86eb4e314948672bf.jpg)

![](images/647dff1e39b74f3427b70cec556316effec7422e033a29efdae32cd10b87f0e7.jpg)  
LLM LLaVA HeRA (Ours)  
Figure 7: Left: Alignment with DINOv2-L, measured with the MKNN metric on each layer and attention head of Qwen2.5-14B. Right: MKNN scores of the Worst-5 and Top-5 heads, computed on (i) the base LLM; (ii) after the LLaVA multimodal training; and (iii) after the addition of HeRA.

Regarding the analysis on the Worst-5 heads, both CMAR and feature-matching methods fail to induce any meaningful structural changes. Conversely, HeRA is the only method capable of significantly increasing the cross-modal alignment of these initially poorly aligned heads.

## C.3 Qualitative Results

In Fig. 10, we provide a qualitative comparison between the LLaVA [19] baseline, ROSS [38], and HeRA using Qwen3-8B and SigLIP2. The representative samples demonstrate that HeRA consistently delivers more accurate and better-grounded answers across all evaluated categories (General, Knowledge, OCR, and Vision-Centric), effectively correcting various perceptual errors made by the baselines.

Despite these clear improvements, in Fig. 11, we report a few failure cases where our model still struggles. Specifically, HeRA can occasionally misinterpret fine-grained visual details, such as accurately counting multiple small instances, identifying ambiguous materials and shapes, or inferring precise spatial relationships in cluttered scenes.

![](images/b3afd674a535868e7020bfe57f67733a2ddfd25f5cd9ca7f244a6a6ad8a4374d.jpg)

![](images/e221275963fff744fa33c49a1544b274ac731175fa52af822b90b32356ce5add.jpg)

Figure 8: Comparison of MKNN scores of the Worst-5 and Top-5 heads after the second training stage performed with HeRA and our competitors (i.e. LLaVA, VIRAL, JARVIS, ROSS, CMAR).  
![](images/0dd6e67446b276e6b5a10c2749588ec203b9db1a8a42c4b1756587ff8ca7e987.jpg)  
Figure 9: VQA results on Qwen3-VL-4B after fine-tuning, with and without the HeRA objective.

## D Limitations and Societal Impacts

We are aware that the landscape of MLLMs has rapidly evolved beyond LLaVA with the introduction of frontier proprietary models (as acknowledged at the end of Sec. 2). However, we adopted the LLaVA pipeline because it remains computationally tractable, allowing for the extensive ablation studies and rigorous evaluations presented in this work.

To bridge this gap and explore the potential of our method on modern architectures, we conducted a pioneering study applying HeRA directly to Qwen3-VL-4B [1], a state-of-the-art multimodal LLM. We fine-tuned the model using an 83k-sample Cambrian split derived from the FineVision [42] dataset. As shown in Fig. 9, HeRA records promising results, particularly on demanding vision-centric tasks. Crucially, the improvements on visual benchmarks are substantially higher with HeRA than with standard fine-tuning alone; moreover, simple fine-tuning actually registers a performance regression in the General category, which does not manifest in HeRA.

Beyond this architectural constraint, we do not foresee direct negative societal impacts arising specifically from our representation alignment technique. Rather, by improving visual grounding and mitigating object hallucinations, HeRA contributes to the development of more reliable and factual vision-language systems.

Is this artwork titled letizia ramolino bonaparte??

LLaVA A house ROSS A house HeRA A cabin V

![](images/f1549ab1e28e2ab2ae97e8b27a21ee6282962b40f3127ff40be185472cad6f76.jpg)

![](images/4b96c827b5c28b70960dbdc95e6196cc1fa363a52993a7bb5dd974891a640749.jpg)

![](images/01aad081f792a971bddde2ddccd4bc81b372f502a2baca187c75f838dc459b56.jpg)

![](images/4dc62fda09f24dd33db8f01528699cb1186209fddb929ded4e5faed719f6bbff.jpg)

![](images/3baa0604865801f798f4f5e0279f5dd8aeef6fa87d85d013c7f0e6e36753ba88.jpg)

![](images/2f788d1e101ae1e28dc28dae999d8b67948207a30137f2067486471bd23e61ec.jpg)

![](images/6cce15e72d01e5a6b3e7866e2a6ceb00826ce9afa4925ed7c62fb74e6f347417.jpg)

![](images/ff1ab58379e2814f67a1be9a5502d9bbb66535792f47e0f1a411078d455174a0.jpg)

![](images/3a8066c08e5df240491ee4a2ffbb7cba15289e2e88758dc5d743813a52c3c008.jpg)

![](images/79d8e108b5840f3e0b6614813d80a2e98d3363925f407854808602098c771bc5.jpg)  
What color is the jersey of the basketball player dunking the ball?

![](images/19179fd271ea616a58892dc58998caab01bb950188ef2ffc953afcfa9238b0e2.jpg)

![](images/fa86379ea9e88da980a5fe900420a1c8ec57ddba1d533c8c8d90327c25b85d9d.jpg)

![](images/036a0751cca415222f9a6260aaaababd301c21eabb81d1f6feb195fa18e194cf.jpg)

![](images/a0bc5ca344fa9b366012a7c024e291ba879b55c8321b0058344a898cddd54941.jpg)

![](images/255f4999ce1370acdcf6e835b55a472c97a68d95bff2babc95183d12a4cef9c7.jpg)

![](images/d37717f5ebec3804ce4741af46a3dc87339cab4bd5edb052a91b7bde7cd7a47f.jpg)

![](images/c823b63a0426d7f800524b10e5a524c4ba1fa57ec593826eb50f252f96266ce1.jpg)

![](images/3fc125c8798a9a06a392ac6a9ae526d07c8e3663b3f41ef9726270192716b44d.jpg)

![](images/a342589a891c689f04fdc6a50819716fc59494eb1c1209a213a8e83f8c2382a2.jpg)

![](images/b91d743de409f72020f86fdc2fa300243d8e377f31985ca375957203a8b6d937.jpg)

![](images/0adadcebca28f48ba260361734b7418db935dbdc41aa45e7d1df3e6c46e1b0b7.jpg)  
Figure 10: Qualitative comparison of LLaVA [19], ROSS [38], and HeRA using Qwen3-8B and SigLIP2. We present representative samples across all Cambrian categories: General, Knowledge, OCR, and Vision-Centric.

![](images/b1f737e01078074d57db8338d4e59ee1cfa6268d38d24a07c17b6fadf88d1ad3.jpg)

![](images/08eed704707038881d698c7f4a6243f66dd11ea1c8d3640ba68ad6fdea8a8bcd.jpg)

![](images/fe7ec218fe9135c670c35fa80550409cf2fd252cef0f39bc3d18de6eb6cd39d4.jpg)  
Figure 11: Failure cases of HeRA on VQA tasks.

![](images/371f8a96072706da45da7ca239fffc1cb14d0da86c7ce5861076fca99a19be06.jpg)