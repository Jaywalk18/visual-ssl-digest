# Beyond Classification: Dynamic Adapter Routing for Continual Multimodal Retrieval

Alicja Dobrzeniecka∗ NASK National Research Institute

Sebastian Cygert NASK National Research Institute

Filip Szatkowski IDEAS Research Institute Warsaw University of Technology

Szymon Łukasik NASK National Research Institute

Bartłomiej Twardowski Universitat Autonoma de Barcelona IDEAS Research Institute

# Abstract

While retrieval is a core function of vision-language models, continually updating these models for retrieval tasks remains critically underexplored. Existing work often approaches continual retrieval through the lens of class-incremental learning (CIL), evaluating both standard CIL methods and retrieval-oriented adaptations in settings that may not fully capture the retrieval-specific dynamics. To address this, we introduce a new, principled evaluation framework for continual multimodal retrieval (CMR) spanning diverse visual domains, and systematically evaluate common approaches within this setting. Our empirical analysis shows that standard CIL methods fail to yield meaningful gains in our more challenging scenario. Therefore, we propose Dynamic Adapter Routing (DAR), a novel approach based on adapters selected through prototype-based routing and combined via model merging.DAR achieves superior performance over the previous baselines and demonstrates strong generalization under out-of-distribution evaluation. Our results highlights the unique challenges of CMR and encourages further research in this direction.

# 1 Introduction

Retrieval is a core functionality of vision-language models (VLMs) such as CLIP [28], serving as the primary interface for deploying these models in real-world search, recommendation, and indexing systems. Despite its immense practical importance, the problem of continually updating multimodal models for retrieval has received relatively limited attention. Only a small number of works explicitly address continual cross-modal retrieval [18, 4, 24, 16], while the majority of research on continual learning (CL) for VLMs focuses on classification-oriented settings [33, 11, 36, 9, 19, 38], particularly class-incremental learning (CIL). As a result, existing approaches fail to capture the unique dynamics and challenges inherent to multimodal retrieval, which requires maintaining a globally consistent embedding space [32] instead of merely learning discrete decision boundaries. This creates unique failure modes: representation drift can ruin global rankings even if a model learns a new task perfectly, and catastrophic forgetting distorts relative similarities rather than causing simple misclassifications [4, 24] as we present conceptually in Figure 1.

Continual classification (task t)   
![](images/fefed22794a4bc5d1f4289eeee816b49aa7dd32115440b9bffea0fcd1b5894fa.jpg)

<details>
<summary>text_image</summary>

Class: woman
Class: child
Class: man
</details>

Continual classification (task t+1)   
![](images/e14322acf2cd2deb0d562ef75bf757f9487311f0b455e36c72dd1d0ccbc726e2.jpg)

<details>
<summary>text_image</summary>

Class: woman
Class: child
Class: man
</details>

Continual retrieval (task t)   
![](images/4c7701f72fcba4d540f196dc6dc91db7665bfb1421403df273c3a5b71d715d64.jpg)

<details>
<summary>text_image</summary>

"A sketch of a woman (...)"
"A painting of red strawberries (...)"
"A woman in a white dress sits (...)"
"A painting of a woman's face in blue and green colors (...)"
</details>

Continual retrieval (task t+1)   
![](images/790e323cfca72e1281502026d2e68fb988713e18c666f9dab05cac43ce78ff5a.jpg)

<details>
<summary>text_image</summary>

"A sketch of a woman (...)"
"A painting of red strawberries (...)"
"A woman in a white dress sits (...)"
"A painting of a woman's face in blue and green colors (...)"
</details>

Figure 1: Conceptual difference between classification and retrieval in CL scenario. In classification, small perturbations may not affect the result as long as the sample remains within the class boundaries. In retrieval, even small perturbations can alter nearest neighbours and substantially affect retrieval rankings. We argue that continual retrieval requires dedicated evaluation protocols and methods, and introduce a new retrieval-focused benchmark and a novel, state-of-the-art method.

Furthermore, existing works on continual retrieval suffer from limited scale and domain diversity in their evaluations, failing to present sufficiently retrieval-specific scenarios. This in turn produces overly optimistic assessments, ultimately obscuring the true utility of CL approaches. To address this, we introduce a new evaluation framework for continual multimodal retrieval. Our framework includes sequences of heterogeneous, non-overlapping datasets that span various visual domains. This design ensures a sufficiently challenging evaluation, enabling more principled comparison. Additionally, we assess the model performance on out-of-distribution (OOD) data, and demonstrate that improving both in- and out-of-distribution results remains challenging for the commonly used continual methods. We use our framework to conduct a systematic analysis of knowledge transfer, interference and robustness under realistic distribution shifts across common CIL methods and retrieval-oriented approaches from literature. Surprisingly, we find that commonly used CL strategies fail to deliver consistent improvements in our more challenging setup.

Motivated by these findings, we introduce Dynamic Adapter Routing (DAR), a novel method for continual multimodal retrieval. Unlike existing routing and mixture-of-experts approaches designed for classification or task-aware settings, continual retrieval requires preserving a globally consistent embedding space under ambiguous cross-domain queries. DAR addresses this challenge through retrieval-aware prototype routing and uncertainty-triggered adapter merging, enabling adaptive knowledge transfer while limiting representation drift.

We perform exhaustive ablations of our method and demonstrate that it achieves superior results over the prior CL baselines across heterogeneous retrieval benchmarks, while retaining strong flexibility under OOD evaluation. Our main contributions include:

• New, more challenging benchmark for continual multimodal retrieval. We present a CL framework focused on retrieval, comprising heterogeneous, non-overlapping domains, alongside in-distribution and out-of-distribution evaluation protocols.   
• Systematic evaluation of CL methods for continual retrieval. Within our framework, we evaluate common CIL and retrieval-oriented approaches, demonstrating that many existing methods offer limited improvements in more demanding settings. We further analyse the correlation between embedding-space alignment and structure and retrieval performance.   
• Dynamic Adapter Routing (DAR). We introduce a continual retrieval method that uses prototype-guided routing and adaptive merging to mitigate representation drift and cross-task interference. DAR significantly outperforms prior approaches and stands out as the only method that delivers consistent, meaningful improvements.

# 2 Related Work

CL for VLMs. Recent research in CL for VLMs has primarily focused on maintaining classification accuracy and zero-shot capabilities under distribution shifts. Representative approaches include ZSCL [38], which employs distillation from a frozen pre-trained backbone to mitigate forgetting, and various prompt-based methods [30, 36] that utilize learnable tokens to bypass the risks of full fine-tuning. While these methods offer valuable insights into stabilizing classification, they are fundamentally optimized for discriminative objectives, which may not transfer well to retrieval settings, where the preservation of fine-grained, cross-modal alignment is paramount. A specialized subset of literature addresses the unique challenges of continual cross-modal retrieval. Wang et al. [32] establish that maintaining a coherent shared embedding space across tasks is a distinct requirement that separates retrieval from standard CIL. Subsequent methods like DKR [4], Mod-X [24], and C-CLIP [18] have attempted to mitigate representation drift through architectural adaptations, dynamic knowledge updates, or parameter-efficient fine-tuning (PEFT).

Continual Retrieval Benchmarks. Historically, CL for VLMs has primarily relied on CIL benchmarks (e.g., CIFAR [15] or ImageNet [5] splits), which are inherently classification-oriented and therefore inadequately capture retrieval-specific performance. Some works propose the frameworks for retrieval evaluation, but they are fundamentally insufficient. While TiC-CLIP [6] evaluates retrieval on temporal data streams, it focuses on large-scale continual pretraining settings that may be less practical for lightweight and reproducible downstream evaluation. Similarly, FOMO-in-FLUX [31] primarily targets pretraining dynamics rather than practical downstream adaptation. Although C-CLIP [18] recently proposed a framework for retrieval evaluation, its reliance on private datasets and a closed-source implementation hinders reproducibility and broader community adoption. Overall, existing evaluations often rely on datasets with limited domain diversity, potentially obscuring retrieval-specific failure modes and leading to overly optimistic performance estimates. This motivates the need for a more principled evaluation framework, which we introduce and discuss in detail in Section 4.

Parameter-Efficient Adaptation and Model Routing. PEFT methods, such as LoRA [8] and adapter-based modules [27], have emerged as the standard for adapting large-scale models with minimal overhead. To scale these methods to multiple tasks in CL, recent work has explored MoE or adapter-based routing, which maintain task-specific modules selected dynamically at inference [36]. More recently, model merging techniques have been proposed to consolidate multiple task-specific adaptations into a unified model without increasing the inference-time parameter count, and proved effective for CL [31, 20]. However, many existing routing and merging strategies rely on task-aware selection or assume static task boundaries, making them ill-suited for real-world retrieval; this motivates the need for adaptive, task-agnostic routing and merging mechanisms for CL capable of navigating distribution shifts and cross-domain ambiguity.

# 3 Background and Problem Formulation

Multimodal Dual-Encoder Models. We assess cross-modal similarity between image and text pairs using a multimodal embedding model. This model consists of an image encoder $\bar { f ( \cdot ; \theta ) }$ and a text encoder $g ( \cdot ; \phi )$ . Given an input image v and its corresponding text c, the model maps them to a shared latent space to produce embeddings $\mathbf { z } = f ( \mathbf { v } ; \theta )$ and $\mathbf { h } = g ( \mathbf { c } ; \phi )$ . The alignment between the image and text is then evaluated using cosine similarity: $s ( \mathbf v , \mathbf c ) = \mathrm { s i m } ( \mathbf z , \mathbf h )$ .

Cross-Modal Retrieval. A goal of the retrieval system is to find the most relevant reference items from a gallery $\mathcal { R } = \{ r _ { j } \} _ { j = 1 } ^ { N }$ given a query $q _ { i }$ from a query set $\mathcal Q = \{ q _ { i } \} _ { i = 1 } ^ { N }$ . We assume an evaluation dataset with paired annotations, such that the unique ground-truth target for query $q _ { i }$ is $r _ { i }$

The retrieval system computes the similarity score $s ( q _ { i } , r _ { j } )$ between the query and all reference items in $\mathcal { R } ,$ ranking the references in descending order based on this score. Let rank $( q _ { i } , r _ { i } )$ denote the rank of the ground-truth reference $r _ { i }$ among the sorted candidates. For a single query sample $q _ { i } ,$ the Recall@K (R@k) metric acts as an indicator function, evaluating to 1 if the correct reference is present within the top K highest-ranked results, and 0 otherwise. The overall retrieval performance on the dataset is then computed by averaging this single-sample indicator across all N queries:

$$
\text { Recall@K } = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} [ \operatorname{rank} (q _ {i}, r _ {i}) \leq K ]. \tag {1}
$$

Given the multimodal model described above, and an evaluation dataset $\mathcal { D } = \{ ( \mathbf { v } _ { i } , \mathbf { c } _ { i } ) \} _ { i = 1 } ^ { N }$ , we can apply this general formulation to assess cross-modal alignment between images and texts in the dataset in both directions. For Image-to-Text (I2T) retrieval, we treat images as the queries and texts as references $( \mathcal { Q } = \{ \mathbf { v } _ { i } \} _ { i = 1 } ^ { N } , \mathcal { R } = \{ \mathbf { c } _ { j } \} _ { j = 1 } ^ { N } )$ , and rank candidates according to $s ( \mathbf { v } _ { i } , \mathbf { c } _ { j } )$ . Conversely, for Text-to-Image (T2I) retrieval, the text serves as the query against an image gallery $( \mathcal { Q } = \{ \mathbf { c } _ { i } \} _ { i = 1 } ^ { N }$ and $\mathcal { R } = \{ \mathbf { v } _ { j } \} _ { j = 1 } ^ { N } )$ , and ranking is performed based on $s ( \mathbf { c } _ { i } , \mathbf { v } _ { j } )$ .

Continual Learning Setup. We further consider a multimodal dual-encoder model under a continual learning scenario, where it is trained on a sequence of $T$ tasks. We use the index $t \in \{ 1 , \ldots , T \}$ to denote a specific task, where $t = 0$ refers to the pretrained state that serves as the initialization for our continual learning setup. For a given task t, the dataset consists of $N _ { t }$ image-text pairs. We denote the j-th sample within this task as $\bar { ( } \mathbf { v } _ { j } ^ { t } , \mathbf { c } _ { j } ^ { t } )$ , where $j \in \{ 1 , \ldots , N _ { t } \}$ . Consequently, the parameters of the encoders after training on task t are updated to $\theta ^ { t }$ and $\phi ^ { t }$ , and we denote the models obtained at the end of each task as $f ^ { \bar { t } } ( \mathbf { v } ) = f ( \mathbf { v } ; \boldsymbol { \theta } ^ { t } ) , \boldsymbol { \dot { g } } ^ { t } ( \mathbf { c } ) = g ( \mathbf { c } ; \boldsymbol { \phi } ^ { t } )$ .

Table 1: Comparison of existing retrieval benchmarks for CL (see Section H for more discussion). 

<table><tr><td>Benchmark</td><td>Diversity</td><td>Specificity</td><td>Curriculum</td><td>OOD Robustness</td><td>Accessibility</td></tr><tr><td>Wang et al. [32] (2021)</td><td>✕</td><td>√</td><td>✕</td><td>✕</td><td>√</td></tr><tr><td>Ni et al. [24] (2023)</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>√</td></tr><tr><td>Cui et al. [4] (2024)</td><td>√</td><td>√</td><td>✕</td><td>✕</td><td>√</td></tr><tr><td>Garg et al. [6] (2024)</td><td>√</td><td>✕</td><td>√</td><td>√</td><td>✕</td></tr><tr><td>Liu et al. [18] (2025)</td><td>√</td><td>✕</td><td>✕</td><td>√</td><td>✕</td></tr><tr><td>Our Framework</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

After learning task t, the model is evaluated on cross-modal retrieval across all previously seen tasks. To differentiate between the retrieval directions, let $m \in \{ \mathrm { I } 2 \mathrm { T } , \mathrm { T } 2 \mathrm { I } \}$ denote the cross-modality direction and define $R _ { t , k } ^ { m }$ as the R@K performance in direction m on the evaluation dataset of a prior task k (where $k \leq t )$ , computed using the updated encoders $f ^ { t }$ and $g ^ { t }$ . To evaluate the overall performance at the end of the training sequence, we compute the Average Final R@K for each direction, which measures the model’s retention across all $\dot { T }$ tasks:

$$
A _ {T} ^ {m} = \frac {1}{T} \sum_ {k = 1} ^ {T} R _ {T, k} ^ {m}. \tag {2}
$$

Similarly, we track the Average Incremental R@K by averaging the mean recall at every incremental step t. We provide the formal definition of this metric in Section C alongside other supplementary metrics for evaluating continual learning systems.

# 4 A New Principled Framework for Continual Retrieval Evaluation

Standard benchmarks for multimodal CL often emphasize class-incremental proxies (e.g., ImageNet-100) and neglect retrieval-specific requirements like global embedding consistency and fine-grained cross-modal alignment. To address this, we propose a new evaluation framework based on five core design choices to rigorously assess continual multimodal retrieval.

Domain Diversity. To minimize overlap with pre-training data, we propose a sequence of heterogeneous domains: natural images (Flickr30K [35]), AI-generated content (Lexica-SD [37], KreaM [7]), artwork (WikiArt [2]), specialized distributions like cartoons (Flintstones [13, 12]), sketches (Sketch [3, 39]), and medical imaging (ROCOv2 [29]). This diversity ensures realistic distribution shifts and prevents overly optimistic performance assessments.

Semantic Granularity and Specificity. Many common datasets (e.g., Oxford Pets [26] used by Liu et al. [18]) suffer from pair redundancy, where multiple images might match a generic caption or vice versa. Therefore, we select datasets with high semantic precision and distinct targets to ensure cross-modal retrieval metrics (I2T and T2I) provide reliable performance signals.

Difficulty-Calibrated Curriculum. Since task ordering heavily impacts CL performance [22], we use the frozen backbone’s zero-shot performance as a proxy for task complexity, inspired by class-incremental works such as Menabue et al. [23]. This creates a structured progression from in-distribution to challenging OOD domains, allowing us to evaluate plasticity degradation while providing a modular way to integrate future tasks.

OOD Retrieval Generalization. Robust methods should leverage sequentially acquired knowledge to enhance its generalization to entirely unseen domains. Therefore, we evaluate the model’s capacity for OOD retrieval by assessing I2T and T2I performance on held-out, broad-domain benchmarks (COCO [17], NoCaps [1]) to quantify how effectively the model transfers learned cross-modal alignments to novel, open-vocabulary concepts beyond its training trajectory.

Practicality, Accessibility, and Reproducibility. Our framework simulates common practical fine-tuning scenarios and is designed for computational tractability for the broader community, and we fully open source our code to ensure standardized, reproducible comparisons.

We provide a detailed comparison of our framework against existing benchmark suites in Table 1, highlighting how we address critical gaps in prior work. As demonstrated in Section 6, current CL strategies fail to provide consistent gains on this more challenging framework.

# 5 Dynamic Adapter Routing (DAR)

We propose Dynamic Adapter Routing (DAR), a novel continual multimodal learning method that operates on top of the pretrained dual-encoder architecture introduced in Section 3. DAR learns a set of task-specific adapter modules and maintains a set of prototypes that anchor each task $t \in \{ 1 , \ldots , T \}$ in the shared backbone feature space. During inference, it routes each query to the most relevant adapter in a task-agnostic manner, using cosine similarity to the stored prototypes. To facilitate knowledge transfer for ambiguous samples, DAR employs dynamic adapter merging.

Model and Adapter Design. We adopt the frozen pre-trained multimodal model with image encoder $f ( \cdot ; \theta ^ { 0 } )$ and text encoder $g ( \cdot ; \phi ^ { 0 } )$ , where $\theta ^ { 0 }$ and $\phi ^ { 0 }$ denote the initial pretrained parameters (i.e., at state $t = 0 )$ . For each incoming task t, we introduce task-specific LoRA adapters, denoted as $\Delta \theta ^ { t }$ and $\Delta \phi ^ { t }$ . Thus, the encoders adapted for task t are formulated as:

$$
f ^ {t} (\mathbf {v}) = f \left(\mathbf {v}; \theta^ {0} + \Delta \theta^ {t}\right), \quad g ^ {t} (\mathbf {c}) = g \left(\mathbf {c}; \phi^ {0} + \Delta \phi^ {t}\right) \tag {3}
$$

We apply LoRA to the attention output projection and the feed-forward networks (attn.out $\_ { \mathsf { p r o j } }$ ml $\cdot \mathtt { P } \cdot \mathtt { C } _ { - } \mathtt { f c }$ , and ml $\mathtt { p . c \_ p r o j } )$ , obtaining a set of lightweight, task-specific experts that share a common backbone representation. After training on task $t ,$ we freeze $\Delta \dot { \theta } ^ { t }$ and $\Delta \phi ^ { t }$ , and initialize the new adapters for the next task from the pretrained parameters $( \theta ^ { 0 } , \phi ^ { 0 } )$ .

To promote parameter reuse and avoid introducing unnecessary adapters, we reuse the adapters from previously seen similar tasks. Specifically, given a new dataset, we compute its prototype using the frozen backbone features and compare it against the stored prototypes of previously learned adapters; if the similarity to an existing prototype exceeds a predefined threshold, we reuse the corresponding adapter and continue training its LoRA parameters, rather than initializing a new adapter from scratch.

Prototype Memory and Margin of Similarity Score. To enable task-agnostic routing, we summarize each task t using prototype vectors in the shared latent space. After training on dataset $\mathcal { D } _ { t } = \{ ( \mathbf { v } _ { j } ^ { t } , \mathbf { c } _ { j } ^ { t } ) \} _ { j = 1 } ^ { N _ { t } }$ , we compute an image prototypene embeddings over the task datase $\mathbf { p } _ { \mathbf { v } } ^ { t }$ and a text prototype t inference time, giv $\mathbf { p } _ { \mathbf { c } } ^ { t }$ by averaging thean image-text pair $( \mathbf { v } , \mathbf { c } )$ , we obtain normalized backbone embeddings z and h. We then compare the image embedding against the image prototypes and the text embedding against the text prototypes to obtain the routing score $S ^ { t }$ corresponding to the prototype from the task t:

$$
S _ {\mathbf {v}} ^ {t} = \operatorname{sim} (\mathbf {z}, \mathbf {p} _ {\mathbf {v}} ^ {t}), \quad S _ {\mathbf {c}} ^ {t} = \operatorname{sim} (\mathbf {h}, \mathbf {p} _ {\mathbf {c}} ^ {t}), \quad S ^ {t} = \frac {1}{2} \left(S _ {\mathbf {v}} ^ {t} + S _ {\mathbf {c}} ^ {t}\right). \tag {4}
$$

We then take these scores to identify the top-1 and top-2 similarities, denoted as $S ^ { ( 1 ) }$ and $S ^ { ( 2 ) }$ , and compute their similarity margin M as $M = S ^ { ( 1 ) } - S ^ { ( 2 ) }$ . If M falls below a set threshold γ, we interpret this as ambiguity between tasks, and perform the adapter merging.

Continual Adapter Merging. When ambiguity is detected $( M < \gamma )$ , we apply adaptive merging. For a given input query $q ,$ we compute merging weights $w ^ { t }$ for the top-2 adapters using a temperaturescaled softmax over their similarity scores $\dot { S } ^ { t }$ , and merge the adapter parameters as follows:

$$
w ^ {t} = \frac {\exp (S ^ {t} / \tau)}{\sum_ {k \in \text { top - 2 }} \exp (S ^ {k} / \tau)}, \quad \Delta \theta^ {*} = \sum_ {t \in \text { top - 2 }} w ^ {t} \Delta \theta^ {t}, \quad \Delta \phi^ {*} = \sum_ {t \in \text { top - 2 }} w ^ {t} \Delta \phi^ {t}. \tag {5}
$$

By smoothly interpolating between task-specific adapters in such adaptive way, our method improves robustness to distribution shifts and effectively navigates cross-domain ambiguity without increasing the inference-time parameter count, yielding both in-domain stability and generalization.

# 6 Experiments

We use framework described in Section 4 and evaluate DAR (Section 5) method with a diverse set of CL strategies for VLMs, which includes regularization-based and parameter-efficient approaches. We employ simple baselines, such as Zero-shot (ZS) performance of the backbone and Fine-tuning (FT), where the model is sequentially updated without explicit mechanisms to prevent forgetting. Moreover, we include standard CL approaches applicable to retrieval models, such as EWC [14] and L2P [33] and Task Arithmetic (TA) [10]. Finally we also evaluate methods that were created directly for retrieval task such as Mod-X [24], DKR [4] and C-CLIP [18].

Table 2: Cross-modal Recall@1 at the end of continual training on our proposed evaluation framework for CLIP ViT-B/16. Surprisingly, commonly used CL approaches often fail to improve upon finetuning baseline, which highlights how continual retrieval presents its own set of challenges that cannot be addressed simply by adapting CIL methods. Our novel method, DAR, shows robust performance and outperforms the previous approaches by a sizable margin. 

<table><tr><td rowspan="2"></td><td colspan="8">Text → Image</td><td colspan="8">Image → Text</td></tr><tr><td>Flickr</td><td>Lexica</td><td>WikiArt</td><td>KreaM</td><td>Flints</td><td>Sketch</td><td>ROCO</td><td>Avg.</td><td>Flickr</td><td>Lexica</td><td>WikiArt</td><td>KreaM</td><td>Flints</td><td>Sketch</td><td>ROCO</td><td>Avg.</td></tr><tr><td>ZS</td><td>62.3</td><td>52.3</td><td>22.6</td><td>20.0</td><td>16.6</td><td>5.2</td><td>1.8</td><td>25.8</td><td>82.0</td><td>52.0</td><td>20.8</td><td>20.2</td><td>11.1</td><td>4.2</td><td>1.5</td><td>27.4</td></tr><tr><td>FT</td><td>73.5</td><td>63.0</td><td>37.1</td><td>26.7</td><td>38.3</td><td>8.3</td><td>6.5</td><td>36.2</td><td>88.5</td><td>64.8</td><td>38.5</td><td>28.4</td><td>35.4</td><td>8.5</td><td>6.9</td><td>38.7</td></tr><tr><td>TA</td><td>73.2</td><td>62.5</td><td>35.0</td><td>24.1</td><td>32.8</td><td>8.2</td><td>3.2</td><td>34.1</td><td>88.7</td><td>64.9</td><td>35.2</td><td>24.4</td><td>27.5</td><td>7.8</td><td>3.6</td><td>36.0</td></tr><tr><td>EWC</td><td>75.4</td><td>62.0</td><td>36.3</td><td>31.6</td><td>42.3</td><td>12.3</td><td>8.6</td><td>38.4</td><td>89.5</td><td>62.5</td><td>36.5</td><td>33.3</td><td>38.8</td><td>12.4</td><td>8.6</td><td>40.2</td></tr><tr><td>Mod-X</td><td>73.5</td><td>61.0</td><td>36.4</td><td>27.6</td><td>40.1</td><td>9.4</td><td>9.4</td><td>36.8</td><td>88.5</td><td>60.9</td><td>36.8</td><td>28.9</td><td>36.9</td><td>8.6</td><td>9.3</td><td>38.5</td></tr><tr><td>C-CLIP</td><td>73.3</td><td>61.9</td><td>34.3</td><td>25.1</td><td>32.6</td><td>8.9</td><td>3.7</td><td>34.3</td><td>88.3</td><td>65.7</td><td>34.1</td><td>26.1</td><td>26.7</td><td>7.2</td><td>3.3</td><td>35.9</td></tr><tr><td>L2P</td><td>66.3</td><td>51.5</td><td>24.3</td><td>18.4</td><td>20.7</td><td>6.0</td><td>2.9</td><td>27.2</td><td>83.2</td><td>51.1</td><td>20.6</td><td>14.9</td><td>15.5</td><td>4.2</td><td>2.5</td><td>27.4</td></tr><tr><td>DKR</td><td>62.9</td><td>45.0</td><td>24.7</td><td>24.5</td><td>36.6</td><td>9.9</td><td>12.5</td><td>30.9</td><td>78.0</td><td>38.1</td><td>20.4</td><td>22.1</td><td>33.1</td><td>7.1</td><td>12.2</td><td>30.1</td></tr><tr><td>DAR</td><td>81.0</td><td>77.9</td><td>50.2</td><td>45.1</td><td>49.6</td><td>15.8</td><td>13.2</td><td>47.5</td><td>93.4</td><td>77.4</td><td>51.3</td><td>45.3</td><td>48.2</td><td>16.2</td><td>13.3</td><td>49.3</td></tr></table>

![](images/989263cb97c27bc1a8cca6edd7dd2fb24e29b7ad4a2a9e6da2b883f2769e9ce2.jpg)

<details>
<summary>line</summary>

| Task ID | DAR   | Finetuning | Mod-X | C-CLIP |
| ------- | ----- | ---------- | ----- | ------ |
| 0       | 52.0  | 52.0       | 52.0  | 52.0   |
| 1       | 61.5  | 60.0       | 61.0  | 58.0   |
| 2       | 61.5  | 60.0       | 61.0  | 59.0   |
| 3       | 61.5  | 60.0       | 61.0  | 60.0   |
| 4       | 61.5  | 60.0       | 61.0  | 60.0   |
| 5       | 61.5  | 60.0       | 61.0  | 60.0   |
| 6       | 61.5  | 60.0       | 61.0  | 61.0   |
| 7       | 61.5  | 60.0       | 61.0  | 60.0   |
</details>

![](images/3e17e95ca8afc8f92d99a7b6c3271acf07b604c5e17d9987b1ac9361d9be3051.jpg)

<details>
<summary>line</summary>

| Task ID | DAR   | Finetuning | Mod-X | C-CLIP |
| ------- | ----- | ---------- | ----- | ------ |
| 0       | 72.0  | 72.0       | 72.0  | 72.0   |
| 1       | 82.0  | 79.0       | 81.0  | 76.0   |
| 2       | 82.0  | 79.5       | 81.5  | 77.0   |
| 3       | 82.0  | 79.5       | 81.0  | 78.0   |
| 4       | 82.0  | 80.0       | 81.5  | 79.0   |
| 5       | 82.0  | 80.0       | 81.0  | 80.0   |
| 6       | 81.5  | 79.5       | 80.5  | 80.5   |
| 7       | 81.5  | 80.0       | 80.5  | 81.0   |
</details>

![](images/10da4bd1fb6ea61548ecf2a65c664baa5c648fed2cfffb2f08bbf2eb869a30c2.jpg)

<details>
<summary>line</summary>

| Task ID | DAR   | Finetuning | Mod-X | C-CLIP |
| ------- | ----- | ---------- | ----- | ------ |
| 0       | 34.0  | 34.0       | 34.0  | 34.0   |
| 1       | 43.5  | 42.0       | 43.0  | 38.0   |
| 2       | 43.5  | 41.5       | 42.5  | 40.0   |
| 3       | 43.5  | 41.0       | 42.0  | 41.0   |
| 4       | 43.5  | 41.5       | 42.5  | 41.5   |
| 5       | 43.5  | 42.0       | 42.0  | 42.0   |
| 6       | 43.5  | 42.0       | 42.0  | 42.0   |
| 7       | 43.5  | 41.5       | 41.5  | 42.0   |
</details>

![](images/ff32d9a1eeb19d9ada7040a665d4c4aa8cdbda950544ee7d5dc84a6e017f607e.jpg)

<details>
<summary>line</summary>

| Task ID | DAR   | Finetuning | Mod-X | C-CLIP |
| ------- | ----- | ---------- | ----- | ------ |
| 0       | 47.5  | 47.5       | 47.5  | 47.5   |
| 1       | 60.0  | 55.0       | 57.5  | 53.0   |
| 2       | 60.0  | 55.0       | 57.5  | 54.0   |
| 3       | 60.0  | 55.0       | 57.5  | 55.0   |
| 4       | 60.0  | 55.0       | 57.5  | 56.0   |
| 5       | 60.0  | 55.0       | 57.5  | 56.0   |
| 6       | 60.0  | 55.0       | 57.5  | 56.0   |
| 7       | 60.0  | 55.0       | 57.5  | 56.0   |
</details>

Figure 2: (left) Image-to-Text and (right) Text-to-Image Recall@1 performance of various CL methods evaluated on the COCO and NoCaps datasets. DAR consistently outperforms alternative approaches on the held-out datasets throughout continual training on the task suite from Table 2, which highlights the efficacy of knowledge-sharing mechanisms in our method.

Unless explicitly stated otherwise, we build our CL setup on top of a frozen CLIP ViT-B/16 backbone, and report Average Final Recall@1 as described in Equation (2) to measure cross-modal retrieval performance. All methods are trained using AdamW for 20 epochs with a batch size of 256 and learning rate $1 0 ^ { - 4 }$ . We implement DAR using task-specific LoRA adapters inserted into both the visual and text encoders. Unless otherwise specified, we use rank r=8, top-k=2 routing, and trigger adaptive merging when the similarity margin falls below γ=0.05. We also use CoreSpace [25] merging. We provide the full implementation details in Section A and share the exact hyperparameters for our runs in the open-sourced code.

# 6.1 Main Results

In Table 2, we report the Recall@1 for both I2T and T2I retrieval across several CL methods evaluated within our framework. Interestingly, naive fine-tuning emerges as a strong baseline under our evaluation settings, outperforming or closely matching dedicated CL methods such as C-CLIP and Mod-X. Overall, our results demonstrate that existing methods fail to consistently address the challenges of continual retrieval. In contrast, DAR provides a substantial improvement over the baseline, outperforming fine-tuning by approximately 11% for both I2T and T2I. This performance gap widens further when we measure Recall@5 in Section D, underscoring the efficacy of an approach designed specifically for retrieval. Additionally, we evaluate zero-shot retrieval performance on the held-out COCO and NoCaps datasets in Figure 2. DAR consistently achieves the strongest results, demonstrating that, despite continual updates, our method preserves and even improves the global alignment structure required for generalization in retrieval tasks. We provide numerical results for these experiments in Section E.

# 6.2 VLM Backbone Generalization

To assess whether our proposed method generalizes across model scales and representation capacities, we evaluate CL methods using different CLIP backbones under the same protocol as in Section 6.1. Specifically, we compare ViT-B/32, ViT-B/16, and ViT-L/14, providing additional results with

our method applied to larger and smaller models. We present the results for DAR and the bestperforming CL baselines in Table 3. As expected, stronger backbones improve retrieval performance for all methods. However, DAR consistently outperforms the competing approaches across all scales, demonstrating the general applicability and robustness of our method.

Table 3: DAR and VLM backbones. 

<table><tr><td rowspan="2">Method</td><td colspan="2">ViT-B/32</td><td colspan="2">ViT-B/16</td><td colspan="2">ViT-L/14</td></tr><tr><td>T→I</td><td>I→T</td><td>T→I</td><td>I→T</td><td>T→I</td><td>I→T</td></tr><tr><td>FT</td><td>33.4</td><td>31.1</td><td>36.2</td><td>38.7</td><td>44.0</td><td>45.5</td></tr><tr><td>Mod-X</td><td>32.9</td><td>30.6</td><td>36.8</td><td>38.5</td><td>44.0</td><td>45.8</td></tr><tr><td>C-CLIP</td><td>32.1</td><td>29.2</td><td>34.3</td><td>35.9</td><td>41.0</td><td>42.3</td></tr><tr><td>DAR</td><td>42.3</td><td>43.7</td><td>47.5</td><td>49.3</td><td>52.8</td><td>53.7</td></tr></table>

# 6.3 DAR Robustness on Alternative Task Sequences

To evaluate DAR and its adapter sharing mechanism, we extend the training sequences by splitting each task from Table 2 into two consecutive halves that are fed sequentially (e.g., Flickr[:50%], Flickr[50%:], etc.). This modified framework tests the modularity of our method and its stability over longer horizons, where representation drift and routing ambiguity accumulate. In Figure 3, we show the Average Incremental Recall and Image-to-Text Recall@1 on held-out NoCaps dataset throughout this extended sequence.

Even in this more challenging setting, DAR remains the top-performing CL method. Crucially, the adapter sharing mechanism functions as intended, efficiently allocating a single adapter for both corresponding halves of a dataset. This highlights our method’s scalability, robustness, and broad applicability. In Section G, we examine more task sequences, and show that our method consistently proves to be the most effective under diverse CL protocols.

![](images/c4cb96c280f9b9e7942af73dc002feb231165a46a6fd676f834b0e8a599f9ae2.jpg)  
Figure 3: (top) In-distribution AIR and (bottom) I2T R@1 on NoCaps during prolonged continual training.

# 6.4 Retrieval Performance Analysis and Correlation with Cross-modal Alignment

To understand what drives the superior performance of DAR, we take a closer look at the stability-plasticity trade-off inherent in our method. In Figure 4, we compare its performance on the first and third training tasks throughout the continual learning process against selected baselines. Interestingly, we find that the superior performance of our method is not solely driven by its ability to separate tasks, which provides unmatched stability. Our LoRA adapters also exhibit excellent plasticity, enabling the model to effectively learn new incoming tasks and achieve the highest performance on a given task immediately after learning it. The enhanced plasticity is likely a result of absence of regularization employed by other methods, and because we initialize the adapter training for each task from a more general, plastic backbone.

![](images/b670c177950d9619a8ddbd721e63da1af5a06d91ee2f2f8b5ae73052d4c3651b.jpg)  
Figure 4: Continual performance on selected tasks.

To analyse the performance of continual retrieval more deeply, we evaluate selected CL methods after training on our benchmark in Table 4. We report retrieval with additional classification, and alignment metrics averaged across all training tasks. Specifically, we measure symmetric R@1, R@5, and MRR (averaged over I2T and T2I), as well as mean zero-shot accuracy across ImageNet, CIFAR, EuroSAT, and DomainNet. Additionally, we quantify local cross-modal separation using the Top-10 negative margin, hard-negative violation rate, and cross-modal asymmetry gap. Finally, we assess the representation stability and structural integrity via top-10 neighbor overlap between the image and text embeddings and linear CKA. See Section C for the full definitions of the metrics.

Table 4: Final retrieval and cross-modal alignment metrics for selected CL methods evaluated on our benchmark. Arrows indicate whether a higher (↑) or lower (↓) metric is desirable. We observe a strong correlation between modality alignment and retrieval performance. DAR achieves superior results by most effectively preserving this alignment throughout the training process. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Retrieval</td><td>Cls.</td><td colspan="3">Local Alignment</td><td colspan="2">Structure</td></tr><tr><td>R@1 ↑</td><td>R@5 ↑</td><td>MRR ↑</td><td>Acc. ↑</td><td>Negative Margin ↑</td><td>Violation Rate ↓</td><td>Asymmetry Gap ↓</td><td>Neighbour Overlap ↑</td><td>CKA ↑</td></tr><tr><td>ZS</td><td>26.6</td><td>43.2</td><td>0.3</td><td>61.8</td><td>0.00</td><td>0.74</td><td>4.89</td><td>-</td><td>0.42</td></tr><tr><td>FT</td><td>37.5</td><td>57.8</td><td>0.4</td><td>62.3</td><td>0.01</td><td>0.64</td><td>1.58</td><td>0.60</td><td>0.56</td></tr><tr><td>C-CLIP</td><td>35.1</td><td>53.6</td><td>0.39</td><td>63.5</td><td>0.01</td><td>0.66</td><td>2.73</td><td>0.80</td><td>0.52</td></tr><tr><td>Mod-X</td><td>37.7</td><td>59.0</td><td>0.41</td><td>59.1</td><td>0.01</td><td>0.65</td><td>2.19</td><td>0.52</td><td>0.55</td></tr><tr><td>DAR</td><td>48.4</td><td>71.3</td><td>0.50</td><td>58.5</td><td>0.03</td><td>0.52</td><td>1.17</td><td>0.86</td><td>0.6</td></tr></table>

Our analysis reveals how retrieval performance is fundamentally tied to embedding geometry. DAR achieves state-of-the-art continual retrieval by simultaneously improving local alignment and preserving cross-modal structure. However, there is also a distinct classification-retrieval trade-off: DAR’s specialization in fine-grained retrieval inadvertently compromises the broader generalization required for robust zero-shot classification, causing it to underperform compared to methods that were designed with CIL in mind. We present detailed classification results in Appendix F.

# 6.5 Low-Rank Adaptation and Memory Cost

Because DAR relies on low-rank adapters, it is crucial to isolate whether our performance gains stem from the low-rank adaptation itself or from our method design and to which degree the adapter hyperparameters influence our results. To this end, we evaluate standard FT against fine-tuning conducted with LoRA; in both cases, the models use a single parameter set through the learning process, differing by either full or low rank updates. Furthermore, since CL might span multiple tasks, we investigate the LoRA’s inherent expressivity-compression trade-off to determine how much additional memory we require to successfully perform in practice. To address these points, in Table 5 we compare DAR across various LoRA ranks against both Full-FT and LoRA fine-tuning baselines.

Standard LoRA fine-tuning does not outperform full fine-tuning, which confirms that our observed performance improvements are not simply an artifact of the underlying parameter-efficient fine-tuning (PEFT) method, but rather a direct result of DAR’s design. Crucially, DAR also maintains robust performance across all evaluated LoRA ranks and performs well even under aggressive parameter compression $( \mathbf { e . g . } , r = 4 )$ . This indicates that the proposed routing and merging mechanisms can function effectively even with small adapters, allowing our method to scale seamlessly as the number of prototypes increases. In practice, assuming standard model dimensions of $d _ { m } = 7 6 8$ and $d _ { h } = 3 0 7 2$ with a rank of $r = 8$ , DAR adapters require only $2 r ( d _ { m } + d _ { h } )$ parameters per MLP layer and $2 r d _ { m }$ parameters for the output projection in attention. This yields an adapter-to-full-weight parameter ratio of approximately $\frac { r } { d _ { m } }$ per block, meaning we can store roughly 96 task-specific adapters using the exact same memory footprint as a single copy of the full model weights. DAR’s actual memory footprint is also reduced even by adapter reuse via prototype routing. As a result, the memory overhead of our method is substantially lower than that of methods like C-CLIP, Mod-X, or EWC, which store full model copies, unless we approach large numbers of adapters.

# 6.6 Routing and Merging Strategies Ablation

Next, we evaluate the impact of the hyperparameters and design of various DAR components. We first compare different prototype signals and fusion strategies that can be used to select task-specific adapters during inference. We present the results for this ablation in Table 6. We find that combining image and text similarities consistently yields better performance than unimodal routing. Moreover, averaging the similarities from both modalities even outperforms oracle routing based on groundtruth task labels, which highlights how our routing mechanism enables cross-task knowledge transfer and validates our final design choice for DAR routing.

Table 6: DAR routing strategies. 

<table><tr><td></td><td>T→I</td><td>I→T</td></tr><tr><td>Random</td><td>36.7</td><td>32.7</td></tr><tr><td>Oracle</td><td>47.1</td><td>48.6</td></tr><tr><td>Image-only</td><td>46.9</td><td>48.4</td></tr><tr><td>Text-only</td><td>43.0</td><td>45.0</td></tr><tr><td>Image+Text (max)</td><td>46.6</td><td>48.2</td></tr><tr><td>Image+Text (avg)</td><td>46.9</td><td>48.5</td></tr><tr><td>DAR</td><td>47.5</td><td>49.3</td></tr></table>

Next, we ablate our adaptive merging mechanism in Table 7 to assess how the number of merged adapters (k) and routing ambiguity threshold (γ) impact In-Distribution (ID) and Out-of-Distribution (OOD) retrieval reported as R@1 averaged across training and held-out tasks at the end of continual training. Overall, adaptive merging improves performance by combining information across tasks for ambiguous samples, and especially helps to improve OOD performance. However, effective merging requires sufficiently set routing uncertainty threshold: small values yield negligible gains over the no-merging baseline, and overly conservative merging with larger γ degrades ID performance. We highlight that, despite these trade-offs, DAR overall remains highly robust to hyperparameter variations.

Table 7: Adaptive merging ablation. 

<table><tr><td rowspan="2">k</td><td rowspan="2">γ</td><td colspan="2">ID</td><td colspan="2">OOD</td></tr><tr><td>T→I</td><td>I→T</td><td>T→I</td><td>I→T</td></tr><tr><td>1</td><td>-</td><td>46.8</td><td>48.6</td><td>51.8</td><td>71.3</td></tr><tr><td>2</td><td>0.00</td><td>46.9</td><td>48.5</td><td>52.0</td><td>71.2</td></tr><tr><td>2</td><td>0.01</td><td>46.9</td><td>48.5</td><td>52.0</td><td>71.2</td></tr><tr><td>2</td><td>0.05</td><td>47.5</td><td>49.3</td><td>52.2</td><td>71.4</td></tr><tr><td>2</td><td>0.10</td><td>46.8</td><td>48.4</td><td>52.4</td><td>71.6</td></tr><tr><td>3</td><td>0.05</td><td>46.9</td><td>48.3</td><td>52.3</td><td>71.5</td></tr><tr><td>4</td><td>0.05</td><td>46.9</td><td>48.3</td><td>52.3</td><td>71.7</td></tr></table>

While DAR employs CoreSpace [25] merging, its modular design easily supports alternative strategies to combine the routed adapters. In Table 8, we compare several representative merging methods using a fixed top-2 routing configuration and report the final R@1 averaged across training tasks.

Methods designed for full-rank weight merging (Task Arithmetic, DARE-TIES, ISO-C) provide only marginal gains over naive averaging when applied to LoRA adapters. CoreSpace performs best and operates directly in the low-rank subspace of the adapters, aligning shared dominant directions while mitigating destructive interference between unrelated updates.

Table 8: Model merging ablation. 

<table><tr><td></td><td>T→I</td><td>I→T</td></tr><tr><td>No merging</td><td>46.8</td><td>48.6</td></tr><tr><td>Avg (uniform)</td><td>46.8</td><td>48.2</td></tr><tr><td>TA [10]</td><td>46.9</td><td>48.4</td></tr><tr><td>DARE-TIES [34]</td><td>46.8</td><td>48.3</td></tr><tr><td>ISO-C [21]</td><td>46.9</td><td>48.4</td></tr><tr><td>CoreSpace [25]</td><td>47.5</td><td>49.3</td></tr></table>

# 7 Conclusions

In this work, we introduce a principled benchmark for continual multimodal retrieval, establishing how continual retrieval poses fundamentally different challenges than continual classification. By evaluating existing CL methods on our benchmark, we reveal that their design is heavily skewed towards class-incremental settings, leading to suboptimal performance in this more demanding retrieval context. To bridge this gap, we propose Dynamic Adapter Routing (DAR), a novel method that employs lightweight adapters to significantly improve both retrieval performance and robustness across diverse domains. Ultimately, our findings underscore that continual retrieval is a distinct problem, necessitating tailored evaluation protocols and adaptation strategies.

Limitations and future work. Our routing mechanism relies on prototype similarity and the frozen backbone, which may become unreliable when domains strongly overlap or change drastically over time, potentially leading to degraded decisions. The current adaptive merging strategy also depends on manually selected hyperparameters, and more sophisticated strategies could further improve its robustness. Nonetheless, our ablations suggest that DAR is quite robust to the hyperparameter choice. While our evaluation focuses on staged CL with CLIP-style models, extending it to fully online settings and other multimodal architectures could be an important area for future research.

Broader impact. Our work aims to advance the field of continual multimodal retrieval through robust evaluation protocols and novel techniques. Although we acknowledge the broader risks of machine learning misuse and advocate for the responsible development of this technology, we do not see any negative societal impacts specific to our research that would require explicit mention here.

# References

[1] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8948–8957, 2019.   
[2] AterMors. wikiart\_recaption. https://huggingface.co/datasets/AterMors/wikiart\_ recaption, 2024. Hugging Face dataset.   
[3] Pinaki Nath Chowdhury, Aneeshan Sain, Ayan Kumar Bhunia, Tao Xiang, Yulia Gryaditskaya, and Yi-Zhe Song. Fs-coco: Towards understanding of freehand sketches of common objects in context. In European conference on computer vision, pages 253–270. Springer, 2022.   
[4] Zhenyu Cui, Yuxin Peng, Xun Wang, Manyu Zhu, and Jiahuan Zhou. Continual vision-language retrieval via dynamic knowledge rectification. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 11704–11712, 2024.   
[5] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2009), 20-25 June 2009, Miami, Florida, USA, pages 248–255. IEEE Computer Society, 2009.   
[6] Saurabh Garg, Mehrdad Farajtabar, Hadi Pouransari, Raviteja Vemulapalli, Sachin Mehta, Oncel Tuzel, Vaishaal Shankar, and Fartash Faghri. Tic-clip: Continual training of clip models. In The Twelfth International Conference on Learning Representations (ICLR), 2024.   
[7] hahminlew. kream-product-blip-captions. https://huggingface.co/datasets/ hahminlew/kream-product-blip-captions, 2023. Hugging Face dataset.   
[8] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.   
[9] Linlan Huang, Xusheng Cao, Haori Lu, and Xialei Liu. Class-incremental learning with clip: Adaptive representation adjustment and parameter fusion. In European Conference on Computer Vision, pages 214–231. Springer, 2024.   
[10] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In The Eleventh International Conference on Learning Representations, 2023.   
[11] Saurav Jha, Dong Gong, and Lina Yao. CLAP4CLIP: Continual learning with probabilistic finetuning for vision-language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.   
[12] Janak Kapuriya. Flintstonessv\_plus\_plus. https://huggingface.co/datasets/Janak12/ FlintstonesSV\_Plus\_Plus, 2025. Hugging Face dataset.   
[13] Janak Kapuriya and Paul Buitelaar. Flintstonessv++ : Improving story narration using visual scene graph. In Text2Story@ECIR, 2025. URL https://api.semanticscholar.org/ CorpusID:279053465.   
[14] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13): 3521–3526, 2017.   
[15] Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical Report 0, University of Toronto, Toronto, Ontario, 2009. URL https://www.cs. toronto.edu/\~kriz/learning-features-2009-TR.pdf.

[16] Yukun Li, Guansong Pang, Wei Suo, Chenchen Jing, Yuling Xi, Lingqiao Liu, Hao Chen, Guoqiang Liang, and Peng Wang. Coleclip: Open-domain continual learning via joint task prompt and vocabulary learning. IEEE Transactions on Neural Networks and Learning Systems, 36(8):15137–15151, 2025.   
[17] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.   
[18] Wenzhuo Liu, Fei Zhu, Longhui Wei, and Qi Tian. C-clip: Multimodal continual learning for vision-language model. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Learning Representations, pages 46461–46477, 2025.   
[19] Haodong Lu, Xinyu Zhang, Kristen Moore, Jason Xue, Lina Yao, Anton van den Hengel, and Dong Gong. Continual learning on CLIP via incremental prompt tuning with intrinsic textual anchors. Transactions on Machine Learning Research, 2025. ISSN 2835-8856.   
[20] Daniel Marczak, Bartlomiej Twardowski, Tomasz Trzcinski, and Sebastian Cygert. MAGMAX: leveraging model merging for seamless continual learning. In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol, editors, Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LXXXV, Lecture Notes in Computer Science, pages 379–395. Springer, 2024.   
[21] Daniel Marczak, Simone Magistri, Sebastian Cygert, Bartłomiej Twardowski, Andrew D Bagdanov, and Joost van de Weijer. No task left behind: Isotropic model merging with common and task-specific subspaces. In Forty-second International Conference on Machine Learning, 2025.   
[22] Marc Masana, Bartlomiej Twardowski, and Joost van de Weijer. On class orderings for incremental learning. CoRR, abs/2007.02145, 2020.   
[23] Martin Menabue, Emanuele Frascaroli, Matteo Boschini, Enver Sangineto, Lorenzo Bonicelli, Angelo Porrello, and Simone Calderara. Semantic residual prompts for continual learning. In European Conference on Computer Vision, 2024.   
[24] Zixuan Ni, Longhui Wei, Siliang Tang, Yueting Zhuang, and Qi Tian. Continual vision-language representation learning with off-diagonal information. In Proceedings of the 40th International Conference on Machine Learning, ICML’23, 2023.   
[25] Aniello Panariello, Daniel Marczak, Simone Magistri, Angelo Porrello, Bartłomiej Twardowski, Andrew D. Bagdanov, Simone Calderara, and Joost van de Weijer. Accurate and efficient low-rank model merging in core space. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.   
[26] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012.   
[27] Jonas Pfeiffer, Andreas Rücklé, Clifton Poth, Aishwarya Kamath, Ivan Vulic, Sebastian Ruder, ´ Kyunghyun Cho, and Iryna Gurevych. AdapterHub: A framework for adapting transformers. In Qun Liu and David Schlangen, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 46–54, Online, October 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-demos. 7. URL https://aclanthology.org/2020.emnlp-demos.7/.   
[28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 2021.   
[29] Johannes Rückert, Louise Bloch, Raphael Brüngel, Ahmad Idrissi-Yaghir, Henning Schäfer, Cynthia S. Schmidt, Sven Koitka, Obioma Pelka, Asma Ben Abacha, Alba G. Seco de Herrera, Henning Müller, Peter A. Horn, Felix Nensa, and Christoph M. Friedrich. Rocov2: Radiology

objects in context version 2, an updated multimodal image dataset. Scientific Data, 11(1), 2024. ISSN 2052-4463.   
[30] James Seale Smith, Paola Cascante-Bonilla, Assaf Arbelle, Donghyun Kim, Rameswar Panda, David Cox, Diyi Yang, Zsolt Kira, Rogerio Feris, and Leonid Karlinsky. Construct-vl: Data-free continual structured vl concepts learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14994–15004, 2023.   
[31] Vishaal Udandarao, Karsten Roth, Sebastian Dziadzio, Ameya Prabhu, Mehdi Cherti, Oriol Vinyals, Olivier Hénaff, Samuel Albanie, Zeynep Akata, and Matthias Bethge. A practitioner’s guide to real-world continual multimodal pretraining. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 133801–133845. Curran Associates, Inc., 2024.   
[32] Kai Wang, Luis Herranz, and Joost van de Weijer. Continual learning in cross-modal retrieval In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE Computer Society, 2021.   
[33] Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer G. Dy, and Tomas Pfister. Learning to prompt for continual learning. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 139–149, 2021.   
[34] Prateek Yadav, Derek Tam, Leshem Choshen, Colin A Raffel, and Mohit Bansal. Ties-merging: Resolving interference when merging models. Advances in neural information processing systems, 36:7093–7115, 2023.   
[35] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2, 2014.   
[36] Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Ping Hu, Dong Wang, Huchuan Lu, and You He. Boosting continual learning of vision-language models via mixture-of-experts adapters. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 23219–23230, June 2024.   
[37] yuwan0. lexica-stable-diffusion-v1-5. https://huggingface.co/datasets/yuwan0/ lexica-stable-diffusion-v1-5, 2024. Hugging Face dataset.   
[38] Zangwei Zheng, Mingyuan Ma, Kai Wang, Ziheng Qin, Xiangyu Yue, and Yang You. Preventing zero-shot transfer degradation in continual learning of vision-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 19125–19136, October 2023.   
[39] zoheb. sketch-scene. https://huggingface.co/datasets/zoheb/sketch-scene, 2025. Hugging Face dataset.

# Appendix

# A Implementation details

Global training protocol. Unless otherwise stated, all methods are trained with the same optimization, data, and evaluation protocol. We use CLIP ViT-B/16 initialized from the pretrained checkpoint, train for 20 epochs per task, and use a training batch size of 256. Validation and zero-shot evaluation use batch size 512. We use 32 data-loader workers and a maximum text length of 77 tokens. Retrieval evaluation is performed on the same task sequence and validation datasets for all methods.

Table 9: Global hyperparameters shared by all methods. 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Backbone</td><td>CLIP ViT-B/16</td></tr><tr><td>Epochs per task</td><td>20</td></tr><tr><td>Training batch size</td><td>256</td></tr><tr><td>Validation batch size</td><td>512</td></tr><tr><td>Zero-shot batch size</td><td>512</td></tr><tr><td>Max text length</td><td>77</td></tr><tr><td>Number of workers</td><td>32</td></tr><tr><td>Seed</td><td>42</td></tr></table>

DAR-specific hyperparameters. DAR uses task-specific LoRA adapters inserted into both image and text encoders. Unless otherwise stated, we use LoRA rank $r = 8$ , LoRA scaling $\alpha = 1 6 \AA$ , and dropout 0.0. At inference time, we route using the average image-text prototype similarity, select the top-2 adapters, and perform adaptive merging when the similarity margin is below $\gamma = 0 . 0 5$ .

Table 10: DAR-specific hyperparameters. 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $1 \times 10^{-4}$ </td></tr><tr><td>Weight decay</td><td>0.2</td></tr><tr><td>Learning-rate scheduler</td><td>Cosine decay</td></tr><tr><td>Warmup epochs</td><td>5</td></tr><tr><td>LoRA rank</td><td>8</td></tr><tr><td>LoRA alpha</td><td>16</td></tr><tr><td>LoRA dropout</td><td>0.0</td></tr><tr><td>Train logit scale</td><td>False</td></tr><tr><td>Prototype momentum</td><td>0.0</td></tr><tr><td>Normalize prototypes</td><td>True</td></tr><tr><td>Routing top- $k$ </td><td>2</td></tr><tr><td>Margin threshold  $\gamma$ </td><td>0.05</td></tr><tr><td>Evaluation adapter policy</td><td>Margin-based</td></tr></table>

# B Compute resources

All experiments were conducted on a single NVIDIA A100 GPU with 40GB memory. Training each method on the main continual retrieval benchmark with 7 tasks required approximately 5-6 GPU hours per run, while the extended CL sequences consisting of 14 tasks required between 7-11 GPU hours depending on the method. Unless otherwise stated, all methods used the same training protocol described in Section A, including 20 epochs per task, training batch size 256, and the CLIP ViT-B/16 backbone.

# C Additional metric definitions

Following the notation established in Section 3, we define additional metrics helpful for assessing the performance of continual retrieval models.

To capture the model’s performance trajectory and stability throughout the entire CL process, we can compute the Average Incremental R@K for each direction. This metric reflects the area under the performance curve by averaging the mean recall at every incremental step t:

$$
\mathrm{AIR} ^ {m} = \frac {1}{T} \sum_ {t = 1} ^ {T} \left(\frac {1}{t} \sum_ {k = 1} ^ {t} R _ {t, k} ^ {m}\right). \tag {6}
$$

For each query, let $\mathcal { P } ( i )$ denote the set of valid ground-truth targets associated with query i. In datasets with a single paired annotation, $\mathcal { P } ( i )$ contains one element, while datasets such as Flickr30K may contain multiple valid captions per image. All candidates not belonging to $\mathcal { P } ( i )$ are treated as negatives. Unless otherwise stated, all metrics are computed independently for image-to-text (I2T) and text-to-image (T2I) retrieval and reported as symmetric averages across both directions.

Recall@K. Let rank $\left[ q _ { i } , r _ { i } \right)$ ) denote the one-based rank of the ground-truth target among all candidates sorted by decreasing similarity. Recall@K is defined as

$$
\mathrm{R} @ \mathrm{K} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} \{\operatorname{rank} (q _ {i}, r _ {i}) \leq K \}.
$$

Mean Reciprocal Rank. Mean Reciprocal Rank (MRR) measures the inverse rank of the first correct retrieval:

$$
\mathrm{MRR} = \frac {1}{N} \sum_ {i = 1} ^ {N} \frac {1}{\mathrm{rank} (q _ {i} , r _ {i})}.
$$

We report symmetric MRR by averaging the I2T and T2I scores.

Top-k negative margin. To assess local separation in the embedding space, we compare the similarity of the positive pair against the hardest nearby negatives. For a text query $\mathbf { c } _ { i }$ , let

$$
s _ {i} ^ {+} = s (\mathbf {c} _ {i}, \mathbf {v} _ {i})
$$

denote the positive similarity, and let $\mathcal { N } _ { k } ( i )$ contain the indices of the k highest-scoring negative images:

$$
\mathcal {N} _ {k} (i) = \mathrm{TopK} _ {j \neq i} s (\mathbf {c} _ {i}, \mathbf {v} _ {j}).
$$

The Top-k negative margin is

$$
\text { Margin@ } k = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(s _ {i} ^ {+} - \frac {1}{k} \sum_ {j \in \mathcal {N} _ {k} (i)} s (\mathbf {c} _ {i}, \mathbf {v} _ {j})\right).
$$

We compute the same quantity for I2T retrieval and report the symmetric average across directions. In our experiments we use $k = 1 0$ . Larger values indicate better local separation between matched pairs and hard negatives.

Hard-negative violation rate. For a query $\mathbf { c } _ { i } ,$ let

$$
s _ {i} ^ {-} = \max _ {j \neq i} s (\mathbf {c} _ {i}, \mathbf {v} _ {j})
$$

denote the hardest-negative similarity. The violation rate is defined as

$$
\mathrm{Viol} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} \{s _ {i} ^ {-} \geq s _ {i} ^ {+} \}.
$$

We compute the metric symmetrically across retrieval directions. This measures the fraction of queries for which at least one negative candidate scores no lower than the ground-truth match. Lower values indicate stronger local alignment.

I2T/T2I asymmetry gap. To quantify directional imbalance between retrieval modes, we compute

$$
\operatorname{Gap} @ K = \left| \mathrm{R} @ \mathrm{K} _ {\mathrm{I2T}} - \mathrm{R} @ \mathrm{K} _ {\mathrm{T2I}} \right|.
$$

We use $K = 5 ,$ , as R@5 is less sensitive than R@1 in datasets where multiple captions or images may be semantically plausible matches.

Top-k neighbor overlap. To assess retrieval stability across continual-learning checkpoints, let $\mathcal { T } _ { k } ^ { t } ( i )$ denote the set of top-k retrieved candidates for query i after learning task $t ,$ and let $\mathcal { T } _ { k } ^ { t - 1 } ( i )$ denote the corresponding set after the previous task. The neighbor overlap is

$$
\text { Overlap@ } k = \frac {1}{N} \sum_ {i = 1} ^ {N} \frac {| \mathcal {T} _ {k} ^ {t} (i) \cap \mathcal {T} _ {k} ^ {t - 1} (i) |}{k}.
$$

Higher overlap indicates greater local neighborhood stability throughout continual adaptation. We use $k = 1 0$ .

Linear CKA. To evaluate global alignment between image and text representations, we compute linear centered kernel alignment (CKA). Let

$$
X \in \mathbb {R} ^ {N \times d _ {x}}, Y \in \mathbb {R} ^ {N \times d _ {y}}
$$

denote mean-centered image and text feature matrices. Linear CKA is defined as

$$
\operatorname{CKA} (X, Y) = \frac {\| X ^ {\top} Y \| _ {F} ^ {2}}{\| X ^ {\top} X \| _ {F} \| Y ^ {\top} Y \| _ {F}}.
$$

Higher CKA values indicate stronger global alignment between the image and text embedding spaces.

# D Additional quantitative results for the main experimental suite

Table 11: Cross-modal retrieval performance on our proposed evaluation framework measured by Recall@5 at the end of continual training for CLIP ViT-B/16. 

<table><tr><td rowspan="2">Method</td><td colspan="8">Text → Image</td><td colspan="8">Image → Text</td></tr><tr><td>Flickr</td><td>Lexica</td><td>WikiArt</td><td>KreaM</td><td>Flints</td><td>Sketch</td><td>ROCOv2</td><td>Avg.</td><td>Flickr</td><td>Lexica</td><td>WikiArt</td><td>KreaM</td><td>Flints</td><td>Sketch</td><td>ROCOv2</td><td>Avg.</td></tr><tr><td>ZS</td><td>85.7</td><td>74.6</td><td>39.9</td><td>45.2</td><td>42.0</td><td>15.9</td><td>4.5</td><td>44.0</td><td>96.7</td><td>73.5</td><td>40.2</td><td>40.7</td><td>27.0</td><td>13.8</td><td>4.3</td><td>42.3</td></tr><tr><td>FT</td><td>92.7</td><td>83.1</td><td>61.8</td><td>52.8</td><td>75.1</td><td>22.8</td><td>15.4</td><td>57.7</td><td>97.3</td><td>82.9</td><td>63.2</td><td>53.9</td><td>69.6</td><td>22.2</td><td>16.4</td><td>57.9</td></tr><tr><td>EWC</td><td>90.6</td><td>76.6</td><td>55.0</td><td>56.3</td><td>77.6</td><td>27.6</td><td>25.2</td><td>58.4</td><td>96.9</td><td>74.0</td><td>53.6</td><td>56.4</td><td>74.6</td><td>27.4</td><td>24.9</td><td>58.3</td></tr><tr><td>Mod-X</td><td>92.4</td><td>80.7</td><td>61.0</td><td>55.1</td><td>76.7</td><td>25.2</td><td>20.5</td><td>58.8</td><td>97.3</td><td>80.2</td><td>61.6</td><td>55.9</td><td>73.8</td><td>25.4</td><td>20.5</td><td>59.2</td></tr><tr><td>C-CLIP</td><td>92.3</td><td>82.7</td><td>58.1</td><td>49.4</td><td>66.1</td><td>21.2</td><td>8.7</td><td>54.1</td><td>97.9</td><td>83.2</td><td>58.0</td><td>49.5</td><td>53.8</td><td>20.1</td><td>9.0</td><td>53.1</td></tr><tr><td>L2P</td><td>88.2</td><td>73.4</td><td>44.7</td><td>40.6</td><td>49.4</td><td>15.8</td><td>7.4</td><td>45.6</td><td>97.0</td><td>73.9</td><td>40.3</td><td>34.1</td><td>35.5</td><td>13.1</td><td>6.9</td><td>43.0</td></tr><tr><td>DKR</td><td>85.6</td><td>68.2</td><td>46.4</td><td>49.9</td><td>72.5</td><td>26.1</td><td>26.9</td><td>53.7</td><td>93.2</td><td>60.1</td><td>39.9</td><td>47.6</td><td>67.8</td><td>22.6</td><td>26.2</td><td>51.1</td></tr><tr><td>TA</td><td>92.4</td><td>83.2</td><td>59.1</td><td>48.6</td><td>65.5</td><td>21.1</td><td>8.7</td><td>54.1</td><td>97.5</td><td>83.2</td><td>59.5</td><td>47.5</td><td>57.1</td><td>20.0</td><td>9.6</td><td>53.5</td></tr><tr><td>DAR</td><td>95.5</td><td>91.6</td><td>76.5</td><td>78.1</td><td>87.8</td><td>39.5</td><td>28.8</td><td>71.1</td><td>99.5</td><td>91.0</td><td>77.4</td><td>77.3</td><td>86.2</td><td>39.7</td><td>28.8</td><td>71.4</td></tr></table>

Table 12: Cross-modal retrieval performance on a reverse of our main evaluation framework measured by Recall@1 at the end of continual training for CLIP ViT-B/16. 

<table><tr><td rowspan="2">Method</td><td colspan="8">Text → Image</td><td colspan="8">Image → Text</td></tr><tr><td>ROCOv2</td><td>Sketch</td><td>Flints</td><td>KreaM</td><td>WikiArt</td><td>Lexica</td><td>Flickr</td><td>Avg.</td><td>ROCOv2</td><td>Sketch</td><td>Flints</td><td>KreaM</td><td>WikiArt</td><td>Lexica</td><td>Flickr</td><td>Avg.</td></tr><tr><td>ZS</td><td>1.8</td><td>5.3</td><td>16.6</td><td>22.0</td><td>20.6</td><td>53.3</td><td>62.3</td><td>26.0</td><td>1.5</td><td>4.2</td><td>11.1</td><td>20.2</td><td>20.8</td><td>53.0</td><td>82.0</td><td>27.5</td></tr><tr><td>FT</td><td>5.1</td><td>8.9</td><td>35.4</td><td>27.1</td><td>38.1</td><td>68.8</td><td>75.9</td><td>37.1</td><td>5.4</td><td>8.1</td><td>31.6</td><td>27.9</td><td>38.2</td><td>68.8</td><td>90.6</td><td>38.7</td></tr><tr><td>EWC</td><td>7.0</td><td>10.1</td><td>40.1</td><td>32.7</td><td>41.0</td><td>68.3</td><td>79.6</td><td>39.8</td><td>6.2</td><td>8.6</td><td>32.1</td><td>31.7</td><td>37.4</td><td>68.5</td><td>92.3</td><td>39.5</td></tr><tr><td>Mod-X</td><td>6.0</td><td>9.5</td><td>39.0</td><td>29.3</td><td>39.8</td><td>69.6</td><td>77.9</td><td>38.7</td><td>6.4</td><td>8.5</td><td>32.5</td><td>29.7</td><td>38.8</td><td>69.0</td><td>91.9</td><td>9.6</td></tr><tr><td>C-CLIP</td><td>3.2</td><td>8.2</td><td>30.9</td><td>25.5</td><td>34.8</td><td>65.0</td><td>74.1</td><td>34.5</td><td>3.0</td><td>6.6</td><td>24.4</td><td>26.1</td><td>33.5</td><td>67.7</td><td>88.8</td><td>35.7</td></tr><tr><td>L2P</td><td>2.2</td><td>5.9</td><td>22.3</td><td>20.6</td><td>26.8</td><td>55.3</td><td>70.2</td><td>29.0</td><td>1.7</td><td>5.0</td><td>16.9</td><td>18.6</td><td>25.6</td><td>51.5</td><td>86.4</td><td>29.4</td></tr><tr><td>DKR</td><td>6.9</td><td>10.0</td><td>41.2</td><td>33.5</td><td>43.4</td><td>69.4</td><td>80.3</td><td>40.7</td><td>6.0</td><td>8.0</td><td>32.4</td><td>31.4</td><td>39.4</td><td>69.3</td><td>93.4</td><td>40.0</td></tr><tr><td>TA</td><td>3.2</td><td>8.4</td><td>32.5</td><td>24.3</td><td>34.9</td><td>62.5</td><td>73.1</td><td>34.1</td><td>3.7</td><td>7.9</td><td>27.5</td><td>24.5</td><td>35.2</td><td>65.0</td><td>88.3</td><td>36.0</td></tr><tr><td>DAR</td><td>9.7</td><td>13.6</td><td>45.1</td><td>39.8</td><td>42.7</td><td>74.5</td><td>75.1</td><td>42.9</td><td>10.0</td><td>13.5</td><td>43.4</td><td>39.1</td><td>43.6</td><td>74.5</td><td>88.1</td><td>44.6</td></tr></table>

# E Full quantitative results for zero-shot retrieval on our main setup

Table 13: Zero-shot retrieval performance on COCO [17] during continual fine-tuning. We report Recall@1 for Image-to-Text (I2T) and Text-to-Image (T2I), together with performance difference $\Delta .$ 

<table><tr><td rowspan="2">Direction</td><td rowspan="2">Method</td><td colspan="8">Task ID</td><td rowspan="2"> $\Delta$ </td></tr><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td rowspan="8">I2T</td><td>Fine-tuning</td><td>52.4</td><td>59.4</td><td>59.3</td><td>59.7</td><td>60.4</td><td>60.2</td><td>60.0</td><td>61.0</td><td>+8.6</td></tr><tr><td>EWC</td><td>52.4</td><td>62.7</td><td>62.8</td><td>61.7</td><td>61.5</td><td>60.7</td><td>60.1</td><td>59.9</td><td>+7.5</td></tr><tr><td>Mod-X</td><td>52.4</td><td>61.4</td><td>61.9</td><td>60.5</td><td>60.9</td><td>60.2</td><td>59.6</td><td>59.0</td><td>+6.6</td></tr><tr><td>C-CLIP</td><td>52.4</td><td>57.7</td><td>58.2</td><td>58.8</td><td>59.0</td><td>59.7</td><td>60.8</td><td>59.8</td><td>+7.4</td></tr><tr><td>L2P</td><td>52.4</td><td>52.2</td><td>53.0</td><td>53.5</td><td>55.0</td><td>52.2</td><td>53.6</td><td>53.4</td><td>+1.0</td></tr><tr><td>DKR</td><td>52.4</td><td>62.2</td><td>62.3</td><td>59.4</td><td>58.6</td><td>57.8</td><td>54.9</td><td>45.4</td><td>-6.9</td></tr><tr><td>TA</td><td>52.4</td><td>57.4</td><td>57.7</td><td>58.5</td><td>58.9</td><td>59.6</td><td>59.4</td><td>59.7</td><td>+7.3</td></tr><tr><td>DAR</td><td>52.4</td><td>61.7</td><td>61.7</td><td>61.9</td><td>61.8</td><td>61.8</td><td>61.6</td><td>61.9</td><td>+9.5</td></tr><tr><td rowspan="8">T2I</td><td>Fine-tuning</td><td>33.2</td><td>41.6</td><td>41.3</td><td>41.5</td><td>41.9</td><td>41.9</td><td>41.7</td><td>41.6</td><td>+8.4</td></tr><tr><td>EWC</td><td>33.2</td><td>44.3</td><td>43.8</td><td>43.7</td><td>43.4</td><td>42.3</td><td>42.5</td><td>42.4</td><td>+9.2</td></tr><tr><td>Mod-X</td><td>33.2</td><td>43.3</td><td>42.9</td><td>42.3</td><td>42.6</td><td>41.5</td><td>41.8</td><td>40.9</td><td>+7.7</td></tr><tr><td>C-CLIP</td><td>33.2</td><td>38.9</td><td>39.3</td><td>40.9</td><td>41.0</td><td>41.6</td><td>41.9</td><td>41.7</td><td>+8.5</td></tr><tr><td>L2P</td><td>33.2</td><td>37.1</td><td>36.9</td><td>37.8</td><td>37.7</td><td>37.8</td><td>37.4</td><td>37.1</td><td>+3.9</td></tr><tr><td>DKR</td><td>33.1</td><td>44.2</td><td>43.1</td><td>41.7</td><td>40.7</td><td>38.4</td><td>37.2</td><td>31.4</td><td>-1.7</td></tr><tr><td>TA</td><td>33.2</td><td>38.7</td><td>38.9</td><td>40.6</td><td>40.5</td><td>40.6</td><td>40.8</td><td>41.0</td><td>+7.8</td></tr><tr><td>DAR</td><td>33.2</td><td>44.1</td><td>44.1</td><td>44.2</td><td>44.2</td><td>44.2</td><td>44.1</td><td>44.1</td><td>+10.2</td></tr></table>

Table 14: Zero-shot retrieval performance on NoCaps [1] during continual fine-tuning. We report Recall@1 for Image-to-Text (I2T) and Text-to-Image (T2I), together with performance difference $\Delta .$ . 

<table><tr><td rowspan="2">Direction</td><td rowspan="2">Method</td><td colspan="8">Task ID</td><td rowspan="2"> $\Delta$ </td></tr><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td rowspan="8">I2T</td><td>Fine-tuning</td><td>71.7</td><td>78.7</td><td>79.4</td><td>79.4</td><td>79.9</td><td>80.0</td><td>79.4</td><td>80.3</td><td>+8.6</td></tr><tr><td>EWC</td><td>71.7</td><td>82.0</td><td>81.5</td><td>80.8</td><td>80.2</td><td>79.6</td><td>78.3</td><td>76.1</td><td>+4.4</td></tr><tr><td>Mod-X</td><td>71.7</td><td>81.1</td><td>80.9</td><td>80.4</td><td>81.0</td><td>80.2</td><td>80.0</td><td>79.8</td><td>+8.1</td></tr><tr><td>C-CLIP</td><td>71.7</td><td>76.1</td><td>77.2</td><td>78.3</td><td>79.1</td><td>79.9</td><td>80.0</td><td>80.0</td><td>+8.3</td></tr><tr><td>L2P</td><td>71.7</td><td>72.0</td><td>72.7</td><td>73.8</td><td>74.9</td><td>71.9</td><td>73.2</td><td>73.6</td><td>+0.9</td></tr><tr><td>DKR</td><td>71.7</td><td>81.6</td><td>81.8</td><td>80.7</td><td>80.0</td><td>78.9</td><td>75.2</td><td>68.2</td><td>-3.5</td></tr><tr><td>TA</td><td>71.7</td><td>76.2</td><td>76.6</td><td>78.1</td><td>78.4</td><td>79.0</td><td>78.9</td><td>79.7</td><td>+8.0</td></tr><tr><td>DAR</td><td>71.7</td><td>81.6</td><td>81.8</td><td>81.9</td><td>81.9</td><td>81.8</td><td>81.4</td><td>81.4</td><td>+9.7</td></tr><tr><td rowspan="8">T2I</td><td>Fine-tuning</td><td>46.7</td><td>56.1</td><td>55.9</td><td>56.8</td><td>57.4</td><td>57.4</td><td>56.8</td><td>56.6</td><td>+9.9</td></tr><tr><td>EWC</td><td>46.7</td><td>59.9</td><td>58.8</td><td>57.8</td><td>57.3</td><td>56.1</td><td>55.6</td><td>52.3</td><td>+5.6</td></tr><tr><td>Mod-X</td><td>46.7</td><td>58.2</td><td>57.8</td><td>57.8</td><td>58.2</td><td>57.4</td><td>57.1</td><td>55.9</td><td>+9.2</td></tr><tr><td>C-CLIP</td><td>46.7</td><td>53.1</td><td>53.6</td><td>56.0</td><td>56.0</td><td>56.6</td><td>56.7</td><td>56.3</td><td>+9.6</td></tr><tr><td>L2P</td><td>46.7</td><td>51.3</td><td>51.1</td><td>52.5</td><td>52.4</td><td>53.0</td><td>52.2</td><td>51.3</td><td>+4.6</td></tr><tr><td>DKR</td><td>46.7</td><td>59.7</td><td>58.8</td><td>57.9</td><td>57.0</td><td>55.0</td><td>52.2</td><td>45.4</td><td>-1.3</td></tr><tr><td>TA</td><td>46.7</td><td>53.0</td><td>53.2</td><td>55.1</td><td>55.3</td><td>55.3</td><td>55.3</td><td>55.6</td><td>+8.9</td></tr><tr><td>DAR</td><td>46.7</td><td>60.4</td><td>60.4</td><td>60.7</td><td>60.6</td><td>60.6</td><td>60.2</td><td>60.2</td><td>+13.5</td></tr></table>

# F Full quantitative results for zero-shot classification

Table 15: Zero-shot accuracy on ImageNet during continual fine-tuning. 

<table><tr><td>Method</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td> $\Delta$ </td></tr><tr><td>Fine-tuning</td><td>68.1</td><td>68.8</td><td>68.8</td><td>67.4</td><td>67.8</td><td>65.9</td><td>65.6</td><td>66.5</td><td>-1.6</td></tr><tr><td>EWC</td><td>68.1</td><td>67.9</td><td>67.6</td><td>66.6</td><td>66.1</td><td>63.7</td><td>62.5</td><td>63.1</td><td>-5.0</td></tr><tr><td>Mod-X</td><td>68.1</td><td>68.6</td><td>68.3</td><td>66.2</td><td>66.4</td><td>63.2</td><td>62.7</td><td>63.2</td><td>-4.9</td></tr><tr><td>C-CLIP</td><td>68.1</td><td>69.2</td><td>69.1</td><td>68.9</td><td>68.7</td><td>67.9</td><td>67.7</td><td>67.6</td><td>-0.5</td></tr><tr><td>L2P</td><td>68.1</td><td>65.2</td><td>66.5</td><td>66.5</td><td>67.4</td><td>64.8</td><td>65.9</td><td>66.3</td><td>-1.8</td></tr><tr><td>DKR</td><td>68.1</td><td>68.0</td><td>66.5</td><td>63.7</td><td>62.2</td><td>56.6</td><td>53.9</td><td>48.2</td><td>-19.9</td></tr><tr><td>TA</td><td>68.1</td><td>69.0</td><td>68.9</td><td>68.5</td><td>68.5</td><td>67.6</td><td>67.3</td><td>67.0</td><td>-1.1</td></tr><tr><td>DAR</td><td>68.1</td><td>64.3</td><td>64.2</td><td>64.2</td><td>64.2</td><td>64.2</td><td>64.2</td><td>64.2</td><td>-3.9</td></tr></table>

Table 16: Zero-shot accuracy on CIFAR100 during continual fine-tuning. 

<table><tr><td>Method</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td> $\Delta$ </td></tr><tr><td>Fine-tuning</td><td>68.4</td><td>68.7</td><td>68.4</td><td>69.7</td><td>69.9</td><td>68.8</td><td>69.6</td><td>71.4</td><td>+3</td></tr><tr><td>EWC</td><td>68.4</td><td>68.2</td><td>68.5</td><td>69.2</td><td>67.7</td><td>66.9</td><td>65.4</td><td>68.3</td><td>-0.1</td></tr><tr><td>Mod-X</td><td>68.4</td><td>69.0</td><td>68.8</td><td>69.5</td><td>69.1</td><td>67.0</td><td>67.8</td><td>67.7</td><td>-0.7</td></tr><tr><td>C-CLIP</td><td>68.4</td><td>69.7</td><td>69.7</td><td>70.7</td><td>70.6</td><td>71.1</td><td>72.0</td><td>72.7</td><td>+4.3</td></tr><tr><td>L2P</td><td>68.4</td><td>65.9</td><td>66.8</td><td>66.6</td><td>68.0</td><td>63.5</td><td>66.8</td><td>64.4</td><td>-4.0</td></tr><tr><td>DKR</td><td>68.4</td><td>68.4</td><td>67.3</td><td>67.5</td><td>65.4</td><td>60.2</td><td>59.7</td><td>45.2</td><td>-23.2</td></tr><tr><td>TA</td><td>68.4</td><td>68.7</td><td>68.6</td><td>69.3</td><td>69.4</td><td>69.4</td><td>69.7</td><td>70.1</td><td>+1.7</td></tr><tr><td>DAR</td><td>68.4</td><td>65.3</td><td>65.4</td><td>65.4</td><td>65.4</td><td>65.4</td><td>65.9</td><td>65.8</td><td>0.0</td></tr></table>

Table 17: Zero-shot accuracy on EuroSAT during continual fine-tuning. 

<table><tr><td>Method</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td> $\Delta$ </td></tr><tr><td>Fine-tuning</td><td>54.0</td><td>54.3</td><td>56.9</td><td>52.6</td><td>56.5</td><td>50.6</td><td>51.1</td><td>55.8</td><td>+1.8</td></tr><tr><td>EWC</td><td>54.0</td><td>51.4</td><td>56.1</td><td>53.1</td><td>56.2</td><td>47.4</td><td>47.7</td><td>55.0</td><td>+1.0</td></tr><tr><td>Mod-X</td><td>54.0</td><td>51.5</td><td>56.3</td><td>49.7</td><td>54.9</td><td>43.9</td><td>46.8</td><td>50.4</td><td>-3.6</td></tr><tr><td>C-CLIP</td><td>54.0</td><td>53.5</td><td>56.4</td><td>56.1</td><td>59.2</td><td>57.1</td><td>58.0</td><td>56.9</td><td>+2.9</td></tr><tr><td>L2P</td><td>54.0</td><td>48.3</td><td>51.4</td><td>52.1</td><td>59.3</td><td>53.7</td><td>52.5</td><td>54.9</td><td>+0.1</td></tr><tr><td>DKR</td><td>54.0</td><td>50.9</td><td>54.9</td><td>50.2</td><td>47.3</td><td>33.4</td><td>33.8</td><td>32.8</td><td>-21.2</td></tr><tr><td>TA</td><td>54.0</td><td>54.1</td><td>56.4</td><td>55.4</td><td>55.6</td><td>52.1</td><td>53.8</td><td>53.6</td><td>-0.4</td></tr><tr><td>DAR</td><td>54.0</td><td>48.9</td><td>49.0</td><td>49.0</td><td>49.0</td><td>49.0</td><td>49.0</td><td>49.0</td><td>-5.0</td></tr></table>

Table 18: Zero-shot accuracy on DomainNet during continual fine-tuning. 

<table><tr><td>Method</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td> $\Delta$ </td></tr><tr><td>Fine-tuning</td><td>56.8</td><td>56.8</td><td>56.8</td><td>56.1</td><td>56.1</td><td>56.1</td><td>56.3</td><td>55.6</td><td>-1.2</td></tr><tr><td>EWC</td><td>56.8</td><td>56.1</td><td>55.6</td><td>55.5</td><td>55.0</td><td>55.0</td><td>55.0</td><td>54.3</td><td>-2.5</td></tr><tr><td>Mod-X</td><td>56.8</td><td>56.6</td><td>56.2</td><td>55.5</td><td>55.3</td><td>55.3</td><td>55.4</td><td>55.0</td><td>-1.8</td></tr><tr><td>C-CLIP</td><td>56.8</td><td>56.9</td><td>56.8</td><td>56.6</td><td>56.6</td><td>56.7</td><td>56.9</td><td>56.7</td><td>-0.1</td></tr><tr><td>L2P</td><td>56.8</td><td>56.1</td><td>56.5</td><td>56.3</td><td>56.8</td><td>55.7</td><td>56.4</td><td>54.6</td><td>-2.2</td></tr><tr><td>DKR</td><td>56.8</td><td>56.3</td><td>55.4</td><td>54.5</td><td>53.1</td><td>52.0</td><td>50.6</td><td>45.9</td><td>-10.9</td></tr><tr><td>TA</td><td>56.8</td><td>56.9</td><td>57.0</td><td>56.7</td><td>56.7</td><td>56.5</td><td>56.6</td><td>56.2</td><td>-0.6</td></tr><tr><td>DAR</td><td>56.8</td><td>54.6</td><td>54.6</td><td>54.6</td><td>54.7</td><td>54.7</td><td>54.9</td><td>54.9</td><td>-1.9</td></tr></table>

# G Full quantitative results alternative tasks sequences

Table 19: Recall@1 after continual training on a longer 14-task sequence using CLIP ViT-B/16. Each original dataset is split into two consecutive subtasks, denoted by suffixes 0 and 1, and the sequence groups both halves of each domain together (e.g., Flickr0, Flickr1, Lexica0, Lexica1, ...). Results are reported for both Text-to-Image and Image-to-Text retrieval, with averages computed across all subtasks. This setting evaluates whether methods remain robust when each domain is revisited immediately through a second split.

<table><tr><td colspan="16">Text → Image</td></tr><tr><td>Method</td><td>Flickr0</td><td>Flickr1</td><td>Lexica0</td><td>Lexica1</td><td>WikiArt0</td><td>WikiArt1</td><td>KreaM0</td><td>KreaM1</td><td>Flints0</td><td>Flints1</td><td>Sketch0</td><td>Sketch1</td><td>ROCO0</td><td>ROCO1</td><td>Avg.</td></tr><tr><td>ZS</td><td>62.8</td><td>62.3</td><td>60.1</td><td>60.0</td><td>26.4</td><td>26.9</td><td>29.2</td><td>28.3</td><td>25.4</td><td>24.0</td><td>7.6</td><td>8.8</td><td>2.5</td><td>2.5</td><td>30.3</td></tr><tr><td>FT</td><td>73.4</td><td>74.3</td><td>69.9</td><td>71.4</td><td>44.6</td><td>44.9</td><td>35.5</td><td>34.0</td><td>51.3</td><td>50.6</td><td>11.3</td><td>14.6</td><td>9.4</td><td>9.02</td><td>42.4</td></tr><tr><td>EWC</td><td>67.6</td><td>69.6</td><td>59.7</td><td>60.0</td><td>36.0</td><td>36.2</td><td>34.9</td><td>34.7</td><td>51.3</td><td>50.7</td><td>14.7</td><td>14.1</td><td>15.3</td><td>15.3</td><td>42.2</td></tr><tr><td>Mod-X</td><td>72.8</td><td>74.2</td><td>66.1</td><td>69.4</td><td>43.6</td><td>44.1</td><td>36.5</td><td>35.4</td><td>53.0</td><td>53.8</td><td>12.0</td><td>14.6</td><td>12.7</td><td>12.3</td><td>43.2</td></tr><tr><td>C-CLIP</td><td>73.4</td><td>74.4</td><td>67.8</td><td>69.6</td><td>42.5</td><td>43.1</td><td>32.9</td><td>32.0</td><td>46.3</td><td>45.5</td><td>10.5</td><td>12.8</td><td>6.2</td><td>6.4</td><td>41.0</td></tr><tr><td>L2P</td><td>65.8</td><td>67.6</td><td>61.1</td><td>61.6</td><td>31.8</td><td>32.7</td><td>27.3</td><td>26.8</td><td>29.7</td><td>27.4</td><td>7.5</td><td>10.7</td><td>4.5</td><td>4.1</td><td>32.8</td></tr><tr><td>DKR</td><td>57.7</td><td>59.2</td><td>46.6</td><td>46.3</td><td>24.5</td><td>25.4</td><td>28.8</td><td>28.5</td><td>44.5</td><td>46.1</td><td>12.8</td><td>12.2</td><td>15.7</td><td>15.4</td><td>32.7</td></tr><tr><td>TA</td><td>70.0</td><td>70.0</td><td>65.7</td><td>66.6</td><td>39.4</td><td>38.8</td><td>27.4</td><td>26.8</td><td>39.3</td><td>39.0</td><td>9.4</td><td>12.2</td><td>3.9</td><td>4.0</td><td>36.6</td></tr><tr><td>DAR</td><td>79.9</td><td>81.7</td><td>83.1</td><td>81.5</td><td>58.3</td><td>58.3</td><td>54.0</td><td>53.9</td><td>63.1</td><td>63.1</td><td>21.1</td><td>21.1</td><td>16.6</td><td>16.8</td><td>55.2</td></tr></table>

<table><tr><td colspan="16">Image → Text</td></tr><tr><td>Method</td><td>Flickr0</td><td>Flickr1</td><td>Lexica0</td><td>Lexica1</td><td>WikiArt0</td><td>WikiArt1</td><td>KreaM0</td><td>KreaM1</td><td>Flints0</td><td>Flints1</td><td>Sketch0</td><td>Sketch1</td><td>ROCO0</td><td>ROCO1</td><td>Avg.</td></tr><tr><td>ZS</td><td>72.9</td><td>74.9</td><td>59.2</td><td>59.6</td><td>26.7</td><td>26.8</td><td>27.0</td><td>25.8</td><td>14.6</td><td>16.2</td><td>6.8</td><td>6.8</td><td>2.3</td><td>2.2</td><td>30.1</td></tr><tr><td>FT</td><td>81.7</td><td>81.1</td><td>71.0</td><td>70.0</td><td>46.2</td><td>46.7</td><td>36.5</td><td>35.9</td><td>47.7</td><td>46.2</td><td>12.9</td><td>12.1</td><td>9.7</td><td>9.9</td><td>43.4</td></tr><tr><td>EWC</td><td>75.6</td><td>77.2</td><td>56.9</td><td>56.3</td><td>32.7</td><td>34.3</td><td>34.7</td><td>35.8</td><td>47.1</td><td>48.2</td><td>12.6</td><td>12.7</td><td>14.7</td><td>15.2</td><td>39.2</td></tr><tr><td>Mod-X</td><td>80.3</td><td>80.8</td><td>67.1</td><td>67.5</td><td>44.0</td><td>44.4</td><td>37.1</td><td>37.0</td><td>49.1</td><td>49.7</td><td>13.7</td><td>12.7</td><td>12.7</td><td>12.5</td><td>43.8</td></tr><tr><td>C-CLIP</td><td>80.7</td><td>80.8</td><td>70.1</td><td>69.9</td><td>43.3</td><td>43.4</td><td>33.7</td><td>32.9</td><td>40.3</td><td>39.7</td><td>11.2</td><td>11.2</td><td>6.5</td><td>6.9</td><td>40.8</td></tr><tr><td>L2P</td><td>75.7</td><td>77.8</td><td>60.1</td><td>59.1</td><td>29.0</td><td>29.3</td><td>23.9</td><td>23.5</td><td>22.5</td><td>23.3</td><td>7.3</td><td>7.6</td><td>3.6</td><td>3.6</td><td>32.5</td></tr><tr><td>DKR</td><td>64.9</td><td>65.3</td><td>39.6</td><td>39.3</td><td>20.0</td><td>20.8</td><td>25.8</td><td>26.1</td><td>38.0</td><td>39.1</td><td>10.9</td><td>10.0</td><td>14.6</td><td>14.5</td><td>29.9</td></tr><tr><td>Meging-TA</td><td>78.7</td><td>79.6</td><td>67.5</td><td>64.5</td><td>38.6</td><td>39.5</td><td>27.7</td><td>27.5</td><td>37.7</td><td>36.1</td><td>10.8</td><td>12.0</td><td>4.8</td><td>5.2</td><td>38.6</td></tr><tr><td>DAR</td><td>87.2</td><td>88.4</td><td>81.7</td><td>81.4</td><td>59.6</td><td>58.6</td><td>53.4</td><td>54.2</td><td>61.1</td><td>62.3</td><td>20.8</td><td>20.3</td><td>16.0</td><td>16.4</td><td>55.8</td></tr></table>

Table 20: Recall@1 after continual training on a longer 14-task interleaved sequence using CLIP ViT-B/16. Each original dataset is split into two subtasks, denoted by suffixes 0 and 1, but all first splits are seen before the corresponding second splits (Flickr0, Lexica0, ..., ROCO0, followed by Flickr1, Lexica1, ..., ROCO1). Results are reported for both Text-to-Image and Image-to-Text retrieval, with averages computed across all subtasks. This setting tests robustness to delayed domain revisitation and evaluates whether methods retain knowledge across a longer gap before seeing the second split of each domain.

<table><tr><td colspan="16">Text → Image</td></tr><tr><td>Method</td><td>Flickr0</td><td>Lexica0</td><td>WikiArt0</td><td>KreaM0</td><td>Flints0</td><td>Sketch0</td><td>ROCO0</td><td>Flickr1</td><td>Lexica1</td><td>WikiArt1</td><td>KreaM1</td><td>Flints1</td><td>Sketch1</td><td>ROCO1</td><td>Avg.</td></tr><tr><td>ZS</td><td>62.8</td><td>60.1</td><td>26.4</td><td>29.2</td><td>25.4</td><td>7.6</td><td>2.5</td><td>62.3</td><td>60.0</td><td>26.9</td><td>28.3</td><td>24.0</td><td>8.8</td><td>2.5</td><td>30.5</td></tr><tr><td>FT</td><td>74.7</td><td>72.6</td><td>46.7</td><td>36.9</td><td>52.3</td><td>11.6</td><td>9.1</td><td>75.6</td><td>74.3</td><td>47.1</td><td>35.4</td><td>51.2</td><td>14.3</td><td>8.8</td><td>43.6</td></tr><tr><td>EWC</td><td>73.9</td><td>68.7</td><td>46.8</td><td>41.9</td><td>57.5</td><td>17.2</td><td>15.2</td><td>76.5</td><td>70.8</td><td>47.6</td><td>40.8</td><td>57.7</td><td>17.8</td><td>15.4</td><td>46.6</td></tr><tr><td>Mod-X</td><td>75.5</td><td>73.2</td><td>48.6</td><td>39.6</td><td>56.2</td><td>13.9</td><td>12.4</td><td>76.6</td><td>75.5</td><td>49.4</td><td>38.8</td><td>54.5</td><td>16.0</td><td>12.2</td><td>45.5</td></tr><tr><td>C-CLIP</td><td>74.7</td><td>70.6</td><td>43.6</td><td>32.8</td><td>46.9</td><td>11.7</td><td>5.9</td><td>75.2</td><td>71.4</td><td>44.2</td><td>32.1</td><td>46.1</td><td>13.0</td><td>6.1</td><td>41.7</td></tr><tr><td>L2P</td><td>67.6</td><td>62.3</td><td>31.6</td><td>26.9</td><td>32.1</td><td>7.6</td><td>4.0</td><td>68.9</td><td>62.9</td><td>32.7</td><td>25.9</td><td>30.1</td><td>9.9</td><td>4.0</td><td>33.9</td></tr><tr><td>DKR</td><td>69.6</td><td>62.5</td><td>41.8</td><td>38.8</td><td>54.0</td><td>17.3</td><td>15.4</td><td>70.5</td><td>64.5</td><td>42.7</td><td>38.7</td><td>54.5</td><td>17.2</td><td>16.4</td><td>43.1</td></tr><tr><td>TA</td><td>71.8</td><td>62.8</td><td>39.4</td><td>27.6</td><td>39.3</td><td>10.0</td><td>3.6</td><td>71.7</td><td>62.3</td><td>38.5</td><td>26.4</td><td>38.9</td><td>12.2</td><td>3.8</td><td>36.9</td></tr><tr><td>DAR</td><td>79.5</td><td>82.4</td><td>55.2</td><td>48.7</td><td>61.8</td><td>18.9</td><td>14.1</td><td>79.9</td><td>81.4</td><td>55.7</td><td>48.7</td><td>60.5</td><td>20.3</td><td>14.2</td><td>51.4</td></tr></table>

<table><tr><td colspan="16">Image → Text</td></tr><tr><td>Method</td><td>Flickr0</td><td>Lexica0</td><td>WikiArt0</td><td>KreaM0</td><td>Flints0</td><td>Sketch0</td><td>ROCO0</td><td>Flickr1</td><td>Lexica1</td><td>WikiArt1</td><td>KreaM1</td><td>Flints1</td><td>Sketch1</td><td>ROCO1</td><td>Avg.</td></tr><tr><td>ZS</td><td>72.9</td><td>59.2</td><td>26.6</td><td>27.0</td><td>14.6</td><td>6.1</td><td>2.3</td><td>74.9</td><td>59.6</td><td>26.8</td><td>25.8</td><td>16.2</td><td>6.8</td><td>2.2</td><td>30.1</td></tr><tr><td>FT</td><td>81.7</td><td>73.6</td><td>49.2</td><td>37.1</td><td>48.5</td><td>12.9</td><td>9.3</td><td>82.9</td><td>72.5</td><td>49.4</td><td>36.5</td><td>48.5</td><td>12.3</td><td>9.8</td><td>44.6</td></tr><tr><td>EWC</td><td>82.3</td><td>67.7</td><td>46.4</td><td>41.5</td><td>55.3</td><td>17.5</td><td>14.8</td><td>82.6</td><td>68.7</td><td>47.0</td><td>42.2</td><td>54.8</td><td>16.9</td><td>15.2</td><td>49.2</td></tr><tr><td>Mod-X</td><td>82.8</td><td>73.0</td><td>50.4</td><td>40.0</td><td>51.0</td><td>15.1</td><td>12.3</td><td>83.7</td><td>73.1</td><td>50.2</td><td>39.8</td><td>54.9</td><td>14.5</td><td>12.6</td><td>46.7</td></tr><tr><td>C-CLIP</td><td>82.2</td><td>71.5</td><td>44.2</td><td>34.1</td><td>41.0</td><td>10.6</td><td>6.1</td><td>82.1</td><td>72.0</td><td>44.5</td><td>33.8</td><td>39.1</td><td>10.5</td><td>6.5</td><td>41.6</td></tr><tr><td>L2P</td><td>76.3</td><td>61.2</td><td>29.6</td><td>23.0</td><td>22.6</td><td>6.9</td><td>3.2</td><td>79.6</td><td>60.5</td><td>30.2</td><td>22.5</td><td>23.1</td><td>7.6</td><td>3.2</td><td>32.8</td></tr><tr><td>DKR</td><td>75.8</td><td>57.1</td><td>37.2</td><td>36.7</td><td>49.5</td><td>17.0</td><td>14.5</td><td>78.8</td><td>58.8</td><td>38.8</td><td>36.9</td><td>52.1</td><td>16.6</td><td>14.6</td><td>41.7</td></tr><tr><td>TA</td><td>79.3</td><td>66.2</td><td>38.6</td><td>27.1</td><td>36.4</td><td>10.7</td><td>4.6</td><td>80.0</td><td>63.4</td><td>38.9</td><td>27.0</td><td>35.3</td><td>11.6</td><td>5.1</td><td>37.6</td></tr><tr><td>DAR</td><td>86.1</td><td>82.2</td><td>57.3</td><td>49.1</td><td>60.2</td><td>18.9</td><td>13.8</td><td>87.7</td><td>80.6</td><td>57.4</td><td>49.3</td><td>60.4</td><td>19.4</td><td>13.7</td><td>52.6</td></tr></table>

# H Summary of related work evaluation protocols

Table 21: Comparison of Continual Retrieval Benchmarks, evaluated each suite against our proposed desiderata: Div. (Domain Diversity), Spec. (Semantic Specificity), Curr. (Calibrated Curriculum), Rob. (OOD Robustness) Acc. (Accessibility). 

<table><tr><td>Benchmark</td><td>Div.</td><td>Spec.</td><td>Curr.</td><td>Rob.</td><td>Acc.</td></tr><tr><td>Wang et al.[32]</td><td> $\times$ Natural images only (COCO/VG)</td><td>✓High-information target pairs</td><td> $\times$ Only 3 sequential tasks</td><td> $\times$ No zero-shot/OOD testing</td><td>✓Small-scale and accessible</td></tr><tr><td>Ni et al.[24]</td><td> $\times$ Limited to Natural/E-commerce</td><td> $\times$ High overlap in Flickr/COCO</td><td> $\times$ Streaming focus; no curriculum</td><td> $\times$ No cross-domain stability eval</td><td>✓Lightweight fine-tuning</td></tr><tr><td>Cui et al.[4]</td><td>✓Diverse (Natural, e-commerce, etc.)</td><td>✓Precise semantic targets</td><td> $\times$ No structured task ordering</td><td> $\times$ Lacks OOD classification</td><td>✓Publicly available</td></tr><tr><td>Garg et al.[6]</td><td>✓Diverse although the domain boundaries are not clearly defined.</td><td> $\times$ Significant web-scale noise</td><td>✓Temporal-based ordering</td><td>✓Extensive OOD suite</td><td> $\times$ Prohibitive compute costs</td></tr><tr><td>Liu et al.[18]</td><td>✓Diverse (Flickr, WikiArt, etc.)</td><td> $\times$ Redundant pairs (e.g., Oxford Pets)</td><td> $\times$ Arbitrary/Random task sequence</td><td>✓HaVG &amp; Cls.</td><td> $\times$ Computationally tractable but no code</td></tr><tr><td>Our Framework</td><td>✓7 heterogeneous domains (Med, Art, etc.)</td><td>✓Curated high-density captions</td><td>✓ZS-calibrated difficulty order</td><td>✓Multi-dataset Ret. &amp; Cls.</td><td>✓Moderate scale; Open-source</td></tr></table>

Table 22: Summary of training and evaluation datasets across selected vision-language CL papers. 

<table><tr><td>Paper</td><td>Training Data</td><td>Training Domains Approximation (no. unique vs total)</td><td>Eval ID</td><td>Eval OOD Classification</td><td>Eval OOD Retrieval</td></tr><tr><td>Our framework</td><td>Retrieval: Flickr30K, Lexica-SD, WikiArt, Kream, Flintstones, Sketch, ROCOv2</td><td>7/7: Natural images, synthetic, art, fashion, cartoons, sketches, medical</td><td>Same datasets</td><td>ImageNet, CIFAR100, EuroSAT, DomainNet</td><td>COCO [17], NoCaps [1]</td></tr><tr><td>C-CLIP [18]</td><td>Retrieval: Flickr30K, COCO, Pets, Lexica, Simpsons, WikiArt, Kream, Sketch</td><td>6/8: Natural images, synthetic, cartoons, art, e-commerce, sketches</td><td>Same datasets</td><td>CIFAR100, ImageNet, Flowers, DTD, Food101, Stanford Cars</td><td>HAVG</td></tr><tr><td>DKR [4]</td><td>Retrieval: MS-COCO, Flickr30K, IAPR TC-12, EC, RSICD</td><td>4/5: Natural images, remote sensing, e-commerce/products, general web images</td><td>Same datasets</td><td>-</td><td>Train on EC splits → test on unseen MS-COCO, Flickr30K</td></tr><tr><td>TIC-CLIP [6]</td><td>Retrieval: TIC-DataComp, TIC-YFCC, TIC-RedCaps</td><td>2/3: Large-scale web data, diverse real-world domains</td><td>Classification: TIC-DataComp-Net; Retrieval: TIC-DataComp-Retrieval, TIC-YFCC Retrieval and TIC-RedCaps</td><td>Over 28 datasets: ImageNet, Food101, MNIST, Oxford-Flowers, Stanford Cars, SUN-97, Oxford-Pet, ObjectNet, and more.</td><td>Flickr30k</td></tr><tr><td>Cross-modal retrieval [32]</td><td>Re- Retrieval: Sequential Visual Genome (SeViGe), Sequential MS-COCO (SeCOCO)</td><td>1/2: Natural images</td><td>Same datasets</td><td>-</td><td>-</td></tr><tr><td>Mod-X [24]</td><td>Retrieval: COCO (initial training) + Flickr30K (streaming); also ECommerce-T2I in extended experiments</td><td>2/3: Natural images, e-commerce/products</td><td>Same datasets</td><td>-</td><td>Train on COCO/Flickr → test on unseen ECommerce-T2I</td></tr></table>

# NeurIPS Paper Checklist

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction state the paper’s main contributions: a continual multimodal retrieval evaluation framework, a systematic empirical comparison of CL methods, and Dynamic Adapter Routing (DAR). These claims are supported by the main results in Section 6 and Table 2.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.   
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.   
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.   
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We include the limitation discussion in Section 7.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.   
• The authors are encouraged to create a separate “Limitations” section in their paper.   
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.   
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.   
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.   
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.   
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.   
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

# Answer: [N/A]

Justification: The paper does not present theoretical results, theorems, or formal proof claims.

# Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.   
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.   
• All assumptions should be clearly stated or referenced in the statement of any theorems.   
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.   
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.   
• Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

# Answer: [Yes]

Justification: The paper specifies the benchmark construction, task sequence, datasets, evaluation metrics, baselines, model backbone, optimization protocol, and DAR hyperparameters in Sections 4, 6, and Appendix A.

# Guidelines:

• The answer [N/A] means that the paper does not include experiments.   
• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.   
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.   
• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.   
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example   
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.   
(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.   
(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).   
(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The paper provides an anonymized repository for the evaluation framework and implementation, and the benchmark is constructed from publicly accessible datasets.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.   
• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.   
• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).   
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.   
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.   
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.   
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).   
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

# 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: The experimental setup specifies the datasets, continual task ordering, backbone, optimizer, number of epochs, batch sizes, learning rate, LoRA rank, routing top-k, margin threshold, and evaluation metrics in Section 6 and Appendix A.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.   
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.   
• The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: The paper reports results across several datasets, backbones, and ablations, but does not currently include error bars, confidence intervals, or statistical significance tests for the main experiments.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.   
• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).   
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)   
• The assumptions made should be given (e.g., Normally distributed errors).   
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.   
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.   
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).   
• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We discuss this in detail in Section B. We also specify the shared training configuration, including the CLIP ViT-B/16 backbone, number of epochs, batch size, optimizer, and other implementation details in Section A.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.   
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.   
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.   
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

# 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research uses existing public datasets and pretrained models for benchmark and method evaluation, and we are not aware of any aspect of the work that violates the NeurIPS Code of Ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.   
• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.   
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

# 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper includes a broader impact discussion in Section 7, where we discuss the responsible development of continual multimodal retrieval systems and acknowledge potential misuse risks associated with vision-language models and retrieval technologies. Section 7.

# Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.   
• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.   
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.   
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.   
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.   
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper does not release a high-risk generative model, scraped dataset, or other asset with a direct high risk for misuse; it releases an evaluation framework and method built on existing public assets.

# Guidelines:

• The answer [N/A] means that the paper poses no such risks.   
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.   
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.   
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper properly credits the pretrained models, datasets, and prior methods used throughout the experiments through citations and references, and uses publicly available assets under their respective terms of use.

# Guidelines:

• The answer [N/A] means that the paper does not use existing assets.   
• The authors should cite the original paper that produced the code package or dataset.   
• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

# 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The paper releases a new evaluation framework and implementation through an anonymized repository. This repository serves as a practical documentation with setup instructions, dataset preparation, expected outputs, licenses, and limitations.

# Guidelines:

• The answer [N/A] means that the paper does not release new assets.   
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.   
• The paper should discuss whether and how consent was obtained from people whose asset is used.   
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing or new research with human subjects.

# Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.   
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.   
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

# 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing or new research with human subjects, so IRB approval is not applicable.

# Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.   
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.   
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.   
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

# 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: The core method development and experiments do not involve LLMs as an important, original, or non-standard component.

# Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.   
• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.