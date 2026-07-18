# U-shaped Multi-granularity Learning for Vision-Language Models

Biao Chen University of Electronic Science and Technology of China Chengdu, China Hithink Research, Hangzhou, China chenbiao2@myhexin.com

Zhongshu Chen University of Electronic Science and Technology of China Chengdu, China zschen@std.uestc.edu.cn

Yunqian Yu University of Electronic Science and Technology of China Chengdu, China yuyunqian2022@163.com

Mengmeng Jing University of Electronic Science and Technology of China Chengdu, China jingmeng1992@gmail.com

Xiangxu Zhao University of Electronic Science and Technology of China Chengdu, China xxzhao@std.uestc.edu.cn

Lin Zuo University of Electronic Science and Technology of China Chengdu, China linzuo@uestc.edu.cn

## Abstract

The prompt learning paradigm for vision-language models is effective yet faces a granularity dilemma: global prompts lack finegrained semantic awareness, while local prompts ignore contextual associations, limiting cross-task generalization. This dilemma exists in dense prediction tasks. Inspired by U-Net, which unifies multi level representations across granularities, we propose UPrompt, a U-shaped multi-granularity prompt learning framework for vision language models. Similar to how U-Net integrates fine and coarse features through symmetric encoder-decoder pathways with cross level connections, UPrompt constructs parallel multi-granularity representations in both visual and textual modalities, where coarse to-fine cascaded enhancement propagates global context to refine local details, while fine-to-coarse hierarchical supervision en sures semantic consistency across scales. Extensive experiments on 17 benchmarks validate our efectiveness. UPrompt outperforms MAMET and VPKE by 4.1 and 7.3 rSum on MSCOCO, surpasses CoCoA-Mix by 5.09% in base-to-novel generalization, while main taining competitive performance with minimal overhead (coarsegrained) and matching PSRC with 1/3 cost (medium-grained). Our code is available at https://github.com/JustCoolPig/UPrompt.

## CCS Concepts

<sup>•</sup> Computing methodologies → Machine learning approaches<sup>.</sup>

## Keywords

Vision Language Model, Cross-modal Retrieval

ACM Reference Format: Biao Chen, Yunqian Yu, Xiangxu Zhao, Zhongshu Chen, Mengmeng Jing, and Lin Zuo. 2018. U-shaped Multi-granularity Learning for Vision-Language <sup>Models.</sup> <sup>In</sup> Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). <sup>ACM,</sup> <sup>New</sup> <sup>York,</sup> NY, USA, 20 pages. https://doi.org/XXXXXXX.XXXXXXX

## 1 Introduction

Prompt learning has emerged as an efective paradigm for Vision-Language Model (VLM) adaptation by optimizing learnable prompt tokens [3, 22, 63]. However, existing methods [2, 42, 56, 60] mainly optimize at a single, fixed granularity, creating an inherent tradeof between capturing broad context and preserving fine-grained visual details [12, 59]. Global prompting strategies cannot encode local features for fine-grained reasoning: CoOp [63], using a single global prompt, underperforms fine-grained TAP [10] by 11.31% on the FGVCAircraft dataset. Conversely, finely structured prompts struggle to integrate suficient global context or model cross-region compositional relationships, as illustrated in Fig. 1. This granularity bottleneck significantly constrains adaptation performance and generalization across diverse vision-language tasks.

In view of this problem, we turn to hierarchical architectures that have demonstrated remarkable success in dense prediction tasks, exemplified by U-Net [41]. U-Net enables efective multi-scale modeling through its symmetric encoder-decoder design and skip connections, jointly preserving high-level semantic context and fine-grained local details. However, transferring these principles to prompt learning is non-trivial. A fundamental paradigm gap exists: U-Net operates on spatially structured pixel grids [49], whereas prompt learning functions in an abstract embedding space [21, 26], where textual prompts lack inherent geometric structure. This raises two core challenges: constructing meaningful multi-granularity representations that embody semantic hierarchy in both modalities, and establishing bidirectional information flow between granularities to ensure cross-level consistency and complementary learning.

To address these challenges, we propose UPrompt learning, a U-shaped multi-granularity prompt learning framework that introduces a structured hierarchy into vision-language adaptation, as shown in Fig. 1(c). Inspired by U-Net’s multi-scale feature fusion mechanisms, such as skip connections [41], UPrompt constructs parallel granularity pathways in vision and language, using progressive spatial pooling for images and iterative semantic enrichment for text. Moreover, the framework incorporates a bidirectional connection mechanism analogous to U-Net’s skip connections: a coarse-to-fine cascaded mechanism that injects global context into fine-grained features using cross-granularity attention, and fine-to coarse hierarchical supervision that distributes semantic knowledge from the finest to the coarsest levels via distillation. This ensures not only enhanced representational capacity at each level, but also consistent semantics across granularities. Comprehensive evalua tion across diverse benchmarks validates the efectiveness of our approach. UPrompt bridges hierarchical representation learning principles from U-Net to prompt-based VLMs, establishing a unified framework that resolves the limitation of single granularity through bidirectional information flow. This architecture provides flexible multi-granularity alignment while ensuring semantic con sistency across representation scales, ofering a principled solution for vision-language adaptation. Our contributions are as follows:

![](images/8093c7e49bd4d4996b7c1c17d4714e3cdb014a1afd050f753b28d5e408f96beb.jpg)  
Figure 1: The granularity trade-of in prompt learning. (a) Global prompting captures broad context but lacks fine-grained details, while (b) fine-grained prompting preserves local features but loses global information. (c) UPrompt learning addresses the trade-of through multi-granularity hierarchical modeling, achieving both global understanding and local precision.

<sub>•</sub> We introduce UPrompt, a U-Net-inspired framework for prompt learning that leverages hierarchical multi-granularity representa tions across vision-language modalities to overcome single-scale adaptation limitations.

<sub>•</sub> We introduce bidirectional connection, establishing bidirectional information flow across multi-granularity hierarchies. Coarseto-fine enhancement injects global context into fine-grained rep resentations for improved local modeling, while Fine-to-Coarse supervision leverages finest-level alignment to regularize coarser granularities, ensuring semantic consistency.

<sub>•</sub> Experiments on 17 benchmarks demonstrate UPrompt’s superior ity in cross-modal retrieval, few-shot classification, base-to-novel generalization, and out-of-distribution scenarios, while its hierarchical design enables flexible performance-eficiency trade-ofs.

## 2 Related Work

<sub>Prompt</sub> <sub>Learning</sub> <sub>in</sub> <sub>VLMs.</sub> CoOp [63] introduced prompt learn ing to CLIP [39], which was later extended to both visual and textual modalities [5, 22]. To overcome single-global-prompt limitations, subsequent methods explore multi-granularity representations. Gal LoP [24] uses dual prompts for global and local features, TAP [10] derives diverse prompts from attribute trees, and SurPL [32] generates dynamic features. HiCroPL [62] injects prompts at multiple levels, while SPTR [7] employs diverse fixed prompt ensembles. However, these methods largely treat granularities as independent modules fused only at the final stage, without unified modeling of cross-scale dependencies and information flow. Our U-Net-inspired framework instead introduces bidirectional connections for progressive integration and semantic consistency across the hierarchy. <sub>Hierarchical</sub> <sub>Representation.</sub> Multi-scale feature fusion, established by FPN [30] and U-Net [41], is fundamental to visual understanding and has been extended to diverse architectures. UNet++ [64] uses nested skip connections to bridge encoder-decoder semantic gaps, while GraphFPN [61] constructs data-dependent feature pyramids for graph neural networks. HGFormer [9] performs part-whole grouping in Vision Transformers, and MSVMamba [43] incorporates hierarchical design into State Space Models, confirming its continued relevance. VLMs must capture both coarse context and fine details, requiring corresponding textual representations at each visual granularity. However, text lacks inherent spatial structure, making direct hierarchical construction dificult. We therefore build and align multi-granularity hierarchies across modalities.

<sub>Cross-Level</sub> <sub>Interaction.</sub> Hierarchical representations commonly follow coarse-to-fine and fine-to-coarse paradigms. Coarse-to-fine methods, such as Stacked Hourglass Networks [35] and RefineNet [29], propagate global context to refine local details. Fine-to-coarse methods, including GroupViT [53] and HVQ [33], aggregate low-level features to maintain high-level consistency. NeRD-Rain [4] further enables bidirectional flow, refining features with coarser context while enriching them with finer details. Inspired by these strategies, we address the isolated optimization of hierarchical prompts in VLMs through bidirectional connections that enable progressive integration and semantic consistency across the hierarchy.

## 3 Methodology

## 3.1 Preliminaries

<sub>CLIP.</sub> Contrastive Language-Image Pre-training (CLIP) [39] uses an image encoder <sub>F</sub> and a text encoder <sub>G</sub> to map image ?? and text ?? into a shared space: $\mathbf { z } _ { v } \ = \ \mathrm { n o r m } ( \mathcal { F } ( x ) ) , \mathbf { z } _ { t } \ = \ \mathrm { n o r m } ( \mathcal { G } ( t ) )$ A symmetric contrastive loss aligns matched pairs and separates mismatched ones, enabling zero-shot classification and cross-modal retrieval. With temperature $\tau ,$ the probability is:

![](images/2b743abcd61c879db4c78649b376ac0e0398f2dfa4a76c8231e15919f62f4fae.jpg)  
Figure 2: Method overview of UPrompt. UPrompt learning constructs hierarchical vision-language alignment via multi granularity pathways with learnable prompts. Bidirectional connection operates through coarse-to-fine cascaded enhancement that injects global context into fine-grained embeddings via cross-attention, and fine-to-coarse hierarchical supervision that guides coarser levels using finest-grained representations.

$$
p (k | x) = \frac {\exp (\mathrm{sim} (\mathbf {z} _ {v} , \mathbf {z} _ {t , k}) / \tau)}{\sum_ {j} \exp (\mathrm{sim} (\mathbf {z} _ {v} , \mathbf {z} _ {t , j}) / \tau)}.\tag{1}
$$

Prompt learning methodology. Prompt learning eficiently op timizes VLMs by incorporating learnable prompt tokens instead of full fine-tuning approaches [22, 63]. The textual and visual in put token sequences at transformer layer ?? are formally defined as: $T _ { i n p u t } ^ { ( i ) } = \{ t _ { b o s } , P _ { t } ^ { ( i ) }$ , ??<sub>??????????</sub>, ??<sub>?????? }</sub> and $V _ { i n p u t } ^ { ( i ) } = \{ v _ { c l s } , E _ { p a t c h } , P _ { v } ^ { ( i ) } \}$ where $P _ { t } ^ { ( i ) } = \{ p _ { t } ^ { 1 } , p _ { t } ^ { 2 } , . . . , p _ { t } ^ { \eta } \}$ and $P _ { v } ^ { ( i ) } = \{ p _ { v } ^ { 1 } , p _ { v } ^ { 2 } , . . . , p _ { v } ^ { M } \}$ are learnable prompt vectors with dimensions $\mathbb { R } ^ { \eta }$ and $\mathbb { R } ^ { M }$ respectively.

<sub>U-shaped</sub> <sub>architecture.</sub> U-shaped networks (e.g., U-Net [41]) con sist of a symmetric encoder-decoder design with skip connections between corresponding layers. Let the network have ?? levels; the encoder at level ?? outputs features $\mathbf { h } ^ { ( i ) }$ , and the decoder at level ?? fuses them with the upsampled features from level $i + 1 { : }$

$$
\tilde {\mathbf {h}} ^ {(i)} = \phi^ {(i)} \left(\mathbf {h} ^ {(i)}, \mathrm{up} (\tilde {\mathbf {h}} ^ {(i + 1)})\right),\tag{2}
$$

where $\phi ^ { ( i ) } ( \cdot )$ is a cross-level fusion operator and up<sub>(·)</sub> denotes up sampling. This enables multi-level information propagation, main taining global context while preserving fine details.

## 3.2 U-Shaped Multi-Granularity Prompting

<sub>UPrompt</sub> <sub>Learning</sub> <sub>Paradigm.</sub> To address the limitation of trade of between global context and local details in single-granularity prompt learning, we draw inspiration from U-Net’s hierarchical processing where $\tilde { \mathbf { h } } ^ { ( i ) } \ = \ \phi ^ { ( i ) } \bar { ( } \mathbf { h } ^ { ( i ) } , \mathrm { u p } ( \tilde { \mathbf { h } } ^ { ( i + 1 ) } ) )$ fuses multi-level features, maintaining both global context and fine-grained details across scales. Existing multi-level methods (e.g., TAP, HiCroPL) treat granularities as independent modules or operate within net work depths, lacking cross-scale dependencies across semantic hier archies. We propose U-shaped multi-granularity prompting, dubbed as UPrompt learning, that constructs parallel hierarchical semantic structures with explicit cross-granularity information flow across modalities. We extend CLIP encoders <sub>F</sub> and <sub>G</sub> to multi-granularity versions $\mathcal { F } ^ { ( k ) } , \mathcal { G } ^ { ( k ) } { } _ { k = 1 } ^ { K }$ where $k \in [ 1 , K ]$ spans coarsest to finest granularities.

For visual modality, we construct nested patch hierarchies through progressive downsampling pooling in embedding space. The input image is first processed by CLIP’s patch embedding layer to extract the finest-grained patch tokens $E _ { p a t c h } ^ { ( K ) } .$ Coarser representations are derived via recursive pooling: $\mathring { E } _ { p a t c h } ^ { ( k ) } = \mathrm { P o o l } ^ { ( k ) } ( E _ { p a t c h } ^ { ( k + 1 ) } )$ for $k = K - 1 , \ldots , 1 .$ ensuring $| E _ { p a t c h } ^ { ( k ) } | < | \dot { E } _ { p a t c h } ^ { ( k + 1 ) } | . \Lambda$ t each granularity level $k ,$ we concatenate the pooled patch embeddings with learnable prompts $P _ { v } ^ { ( k ) } \in \mathbb { R } ^ { M \times d }$ and feed them into CLIP’s vision encoder to obtain visual features: $\mathbf { z } _ { v } ^ { ( k ) } = \mathcal { F } ^ { ( k ) } ( [ { E } _ { p a t c h } ^ { ( k ) } ; { P } _ { v } ^ { ( k ) } ] )$ .

For textual modality, we construct semantic hierarchies via progressive enrichment:

$$
T _ {e m b e d} ^ {(1)} = \Phi_ {\mathrm{abstract}} (t),\tag{3}
$$

$$
T _ {e m b e d} ^ {(k)} = T _ {e m b e d} ^ {(k - 1)} \oplus \Phi_ {\mathrm{refine}} ^ {(k)} (t), \quad k = 2, \ldots , K,\tag{4}
$$

where $\Phi _ { \mathrm { a b s t r a c t } } ( \cdot )$ extracts core semantics, $\Phi _ { \mathrm { r e f i n e } } ^ { ( k ) } ( \cdot )$ generates granularityspecific elaborations, and <sub>⊕</sub> is semantic expansion ensuring nesting $\hat { T } _ { e m b e d } ^ { ( k ) } \subset T _ { e m b e d } ^ { ( k + 1 ) }$ . Text representations integrate prompts $P _ { t } ^ { ( k ) }$ ∈ $\begin{array} { r } { \mathbb { R } ^ { \eta \times d } , \mathrm { i . e . , } \mathbf { z } _ { t } ^ { ( k ) } = \mathcal { G } ^ { ( k ) } ( [ P _ { t } ^ { ( k ) } ; T _ { e m b e d } ^ { ( k ) } ] ) } \end{array}$ . Operators $\Phi _ { \mathrm { a b s t r a c t } } ( \cdot )$ and $\Phi _ { \mathrm { r e f i n e } } ^ { ( k ) } ( \cdot )$ are instantiated via LLMs with specific prompts for multilevel text generation (implementation details in Sec. 4).

Cross-modal alignment at each granularity ?? is achieved through similarity $\begin{array} { r } { \mathbf { S } ^ { ( k ) } ( x , t ) = \frac { \mathbf { z } _ { v } ^ { ( k ) } \cdot \mathbf { z } _ { t } ^ { ( k ) } } { \| \mathbf { z } _ { v } ^ { ( k ) } \| \| \mathbf { z } _ { t } ^ { ( k ) } \| } } \end{array}$ . This U-shaped architecture enables multi-granularity vision-language alignment through hierarchical prompt learning across symmetric pathways, with visual branch providing spatial representations and textual branch ofering semantic specifications.

## 3.3 Bidirectional Connection for UPrompt Learning

Simple granularity stacking in UPrompt learning lacks inter-granularity interaction, causing fine-grained context deficiency and coarsegrained optimization inconsistency that limit multi-granularity representation. To address these challenges, we propose bidirectional

connection for UPrompt learning, which employs coarse-to-fine cas caded enhancement during forward propagation and fine-to-coarse hierarchical supervision during backward optimization (Fig. 2).

Coarse-to-Fine Cascaded Enhancement (CE). <sup>To</sup> <sup>address</sup> <sup>con</sup> text deficiency where fine-grained embeddings lack global contextual guidance for modeling local information relationships, we propose cascaded enhancement that injects coarse-grained contex tual information into finer embeddings. For embeddings $X ^ { ( k ) } ~ \in$ $\{ E _ { p a t c h } ^ { ( k ) } , T _ { e m b e d } ^ { ( k ) } \}$ at granularity level $k ,$ the enhancement operation:

$$
\hat {X} ^ {(k)} = X ^ {(k)} \odot \mathcal {A} (X ^ {(k)}, \hat {X} ^ {(k - 1)}),\tag{5}
$$

where $\hat { X }$ are enhanced embeddings, <sub>⊙</sub> is element-wise product, $\mathcal { A } ( \cdot , \cdot )$ is cross-granularity attention:

$$
\begin{array}{c} \mathcal {A} (X ^ {(k)}, \hat {X} ^ {(k - 1)}) = \text {softmax} \left(\frac {X ^ {(k)} \mathbf {W} _ {q} (\hat {X} ^ {(k - 1)} \mathbf {W} _ {k}) ^ {\top}}{\sqrt {d}}\right) \\ \cdot \hat {X} ^ {(k - 1)} \mathbf {W} _ {v}, \end{array}\tag{6}
$$

where $\mathbf { W } _ { q , k , v } ~ \in ~ \mathbb { R } ^ { d \times d }$ are query, key, and value projection matrices, ?? is the embedding dimension. Enhanced embeddings are fed into encoders to obtain $\mathbf { z } _ { v } ^ { ( k ) } = \mathcal { F } ^ { ( k ) } ( [ \hat { E } _ { p a t c h } ^ { ( k ) } ; P _ { v } ^ { ( k ) } ] )$ and $\mathbf { \Delta z } _ { t } ^ { ( k ) } =$ $\boldsymbol { \mathcal { G } } ^ { ( k ) } ( [ P _ { t } ^ { ( k ) } ; \hat { T } _ { e m b e d } ^ { ( k ) } ] )$ . Fine-grained embeddings can extract contextually relevant information from global representations, enhancing local information modeling with global contextual guidance.

Proposition 3.1 (CE Directional Alignment Effect). <sub>Let</sub> $\hat { X } ^ { ( k ) }$ be the fine-grained representation at level <sup>??</sup> enhanced $b y$ coarseto-fine cascaded enhancement (CE, Eq. $( 5 ) - ( 6 ) )$ , which leverages contextual guidance from the coarser representation $\hat { X } ^ { ( k - 1 ) }$ . Let $\bar { X } ^ { ( k ) }$ be its unenhanced counterpart. Under the mild assumption that the coarse context is informative, CE provably strengthens alignment between fine-grained features and coarse-grained guidance in expectation:

$$
\mathbb {E} \left[ \frac {\langle \hat {X} ^ {(k)} , \hat {X} ^ {(k - 1)} \rangle}{\| \hat {X} ^ {(k)} \| \| \hat {X} ^ {(k - 1)} \|} \right] \geq \mathbb {E} \left[ \frac {\langle X ^ {(k)} , \hat {X} ^ {(k - 1)} \rangle}{\| X ^ {(k)} \| \| \hat {X} ^ {(k - 1)} \|} \right].\tag{7}
$$

<sub>Proof.</sub> Cascaded enhancement injects coarse contextual guidance into fine representations via cross-attention, provably improving directional alignment in expectation. See Appendix $\mathrm { A . 1 }$

Fine-to-Coarse Hierarchical Supervision (HS). <sup>To</sup> <sup>address</sup> <sup>opti</sup> mization inconsistency from semantic drift at coarse granularities, we propose fine-to-coarse hierarchical supervision using the supe rior alignment of finest-grained features. The finest-level features $( \mathbf { z } _ { v } ^ { ( K ) } , \mathbf { \bar { z } } _ { t } ^ { ( K ) } )$ achieve optimal cross-modal correspondence through rich representational capacity, serving as teacher signals for coarser levels. The finest-level cross-modal similarity matrix ${ \mathsf S } ^ { ( K ) }$ provides the teacher distribution:

$$
\mathbf {S} _ {i j} ^ {(K)} = \frac {\mathbf {z} _ {v , i} ^ {(K)} \cdot \mathbf {z} _ {t , j} ^ {(K)}}{\| \mathbf {z} _ {v , i} ^ {(K)} \| \| \mathbf {z} _ {t , j} ^ {(K)} \|}.\tag{8}
$$

All coarser levels are supervised via knowledge distillation from the detached finest-level representations, preventing degradation of the teaching signal by coarse-grained semantic drift:

$$
\begin{array}{c} \mathcal {L} _ {\text {guide}} = \frac {1}{K - 1} \sum_ {k = 1} ^ {K - 1} \mathbb {E} _ {(i, j)} \left[ D _ {\mathrm{KL}} \Big (\text {softmax} \left(\mathrm{S} _ {i,:} ^ {(k)} / \tau_ {d}\right) \right. \\ \left. \left. \| \text {softmax} \left(\mathrm{S} _ {i,:} ^ {(K)} / \tau_ {d}\right)\right) \right]. \end{array}\tag{9}
$$

where $\tau _ { d }$ is the distillation temperature. Detaching $\mathsf { S } ^ { ( K ) }$ prevents gradients from coarse-level training flowing back to the finest layer, ensuring that fine-grained misalignment does not corrupt coarse representations. Concurrently, CE injects global contextual guidance into fine-grained features, enabling self-correction across granularities. This hierarchical supervision enforces semantic consistency across all granularity levels, preventing coarse-grained drift while keeping the complementary nature of multi-granularity representations enables more efective cascaded enhancement.

Proposition 3.2 (HS Consistency and Substitutability). <sub>Let</sub> $S ^ { ( k ) }$ and <sup>?? (?? )</sup> be similarity matrices from Eq. (8), and define $p _ { \tau _ { d } } ^ { ( k ) } ( j | i ) =$ softmax $S _ { i , : } ^ { ( k ) } / \tau _ { d } ) _ { j }$ ?? and $q _ { \tau _ { d } } ^ { ( K ) } ( j | i ) = s o f t m a x ( S _ { i , : } ^ { ( K ) } / \tau _ { d } ) _ { j }$ ?? where teacher $q ^ { ( K ) }$ is detached as in Eq. (9). Assuming HS aligns coarse-grained distributions with fine-grained teachers, HS bounds semantic drift and enables performance-preserving coarse inference:

$$
\begin{array}{r l} & {\mathbb {E} _ {(x, t), i} \bigg [ \mathrm{KL} \Big (q _ {\tau_ {d}} ^ {(K)} (\cdot | i) \| p _ {\tau_ {d}} ^ {(k)} (\cdot | i) \Big) \bigg ] \leq \varepsilon} \\ & {\implies \mathbb {E} _ {(x, t), i} \big [ \big | \Phi \Big (p _ {\tau_ {d}} ^ {(k)} (\cdot | i) \Big) - \Phi \Big (q _ {\tau_ {d}} ^ {(K)} (\cdot | i) \Big) \big | \big ] \leq L \sqrt {\varepsilon / 2},} \end{array}\tag{10}
$$

for any <sup>??</sup>-Lipschitz functional Φ w.r.t. total variation distance. The detach operation ensures gradient isolation: $\partial L _ { q u i d e } / \partial z ^ { ( K ) } = 0$

<sub>Proof.</sub> Hierarchical supervision constrains KL divergence between coarse and fine distributions, yielding bounded substitutability via Pinsker’s inequality. See Appendix $\mathrm { A } . 2 .$

<sub>Overall</sub> <sub>Objective.</sub> The UPrompt learning framework combines contrastive alignment loss across all ?? granularity levels with hierarchical supervision for cross-modal alignment and inter-granularity consistency. Contrastive losses are averaged for stable optimization:

(11)

During inference, we can flexibly select granularity levels based on performance-eficiency trade-ofs. We default to finest-grained features $( \mathbf { z } _ { v } ^ { ( K ) } , \mathbf { z } _ { t } ^ { ( K ) } )$ for optimal performance. When prioritizing eficiency, coarser levels ofer reduced token requirements and lower costs while preserving semantic consistency via our fine-tocoarse hierarchical supervision that prevents coarse-grained drift.

## 4 Experiments

<sub>Datasets.</sub> For cross-modal retrieval, we evaluate on Flickr30K [57] with 31,783 images and MSCOCO-5K [31] with 123,287 images, each annotated with 5 captions. For classification tasks, we use 11 datasets: ImageNet [8], Caltech101 [11], OxfordPets [38], StanfordCars [23], Flowers102 [36], Food101 [1], FGVCAircraft [34], SUN397 [51], UCF101 [44], DTD [6] and EuroSAT [16]. For out-ofdistribution evaluation, we use ImageNet-A [18], ImageNet-R [17], ImageNet-Sketch [46], and ImageNet-V2 [40].

<sub>Implementation</sub> <sub>details.</sub> For multi-granularity construction, visual modality applies progressive spatial pooling to patch embeddings: from original 14<sub>×</sub>14 to 7<sub>×</sub>7, 4<sub>×</sub>4, and 1<sub>×</sub>1 tokens. Classification uses all 4 scales while retrieval uses the first 3 scales. For textual modality, we employ Llama 3-8B to generate hierarchical representations. In classification, we construct 4-level prompts:“<sub>a</sub> <sub>photo</sub> <sub>of</sub> <sub>a</sub> <sub>{cls}</sub>” (Level 1), progressively enriched with representative with coarse granularity created via prompt“<sub>shorten</sub> <sub>the</sub> <sub>caption</sub> <sub>and</sub> <sub>keep</sub> <sub>important</sub> <sub>information</sub>”, and fine granularity enhanced via “<sub>add</sub> details from other captions to enhance the original caption and keep <sub>original</sub> <sub>meaning</sub> <sub>unchanged</sub>”. Text hierarchies (via LLM) are precomputed rather than computed in real-time, avoiding significant computational overhead. We use CLIP ViT-B/16 as backbone by default (Ablation studies on diferent backbones and VLMs are in Appendix D.3 and D.4). We employ 3- and 4-granularity configurations for retrieval and classification respectively, with distinct learnable prompts of length 4 for each granularity level. During inference, we default to the finest-grained features, which consistently yield optimal alignment.

Table 1: Base-to-novel generalization. Bold values indicate the best results. HM: Harmonic Mean.

<table><tr><td rowspan="2">Method</td><td colspan="3">Average</td><td colspan="3">ImageNet</td><td colspan="3">Caltech101</td><td colspan="3">OxfordPets</td></tr><tr><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td> $CoOp_{(IJCV'22)}$ </td><td>82.69</td><td>63.22</td><td>71.66</td><td>76.47</td><td>67.88</td><td>71.92</td><td>96.00</td><td>89.81</td><td>93.73</td><td>93.67</td><td>95.29</td><td>94.47</td></tr><tr><td> $PSRC_{(ICCV'23)}$ </td><td>84.26</td><td>76.10</td><td>79.97</td><td>77.60</td><td>70.73</td><td>74.01</td><td>98.10</td><td>94.03</td><td>96.02</td><td>95.33</td><td>97.30</td><td>96.30</td></tr><tr><td> $TAP_{(ICLR'25)}$ </td><td>84.75</td><td>77.63</td><td>81.04</td><td>77.97</td><td>70.40</td><td>73.99</td><td>98.90</td><td>95.50</td><td>97.17</td><td>95.80</td><td>97.73</td><td>96.76</td></tr><tr><td> $CLIP-AST_{(CVPR'25)}$ </td><td>85.64</td><td>76.99</td><td>81.06</td><td>78.44</td><td>70.22</td><td>74.10</td><td>98.71</td><td>94.00</td><td>96.30</td><td>96.23</td><td>97.37</td><td>96.80</td></tr><tr><td> $SurPL-G_{(ICML'25)}$ </td><td>86.37</td><td>76.32</td><td>81.03</td><td>78.74</td><td>70.49</td><td>74.39</td><td>98.77</td><td>95.16</td><td>96.93</td><td>96.37</td><td>97.41</td><td>96.89</td></tr><tr><td> $CoCoA-Mix_{(ICML'25)}$ </td><td>79.31</td><td>75.10</td><td>77.03</td><td>75.47</td><td>68.92</td><td>72.04</td><td>98.02</td><td>94.39</td><td>96.17</td><td>95.16</td><td>97.60</td><td>96.36</td></tr><tr><td>UPrompt(Ours)</td><td>86.35</td><td>78.29</td><td>82.12</td><td>78.65</td><td>71.24</td><td>74.76</td><td>98.78</td><td>95.84</td><td>97.29</td><td>96.41</td><td>97.92</td><td>97.16</td></tr><tr><td rowspan="2">Method</td><td colspan="3">StanfordCars</td><td colspan="3">Flowers102</td><td colspan="3">Food101</td><td colspan="3">FGVCAircraft</td></tr><tr><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td> $CoOp_{(IJCV'22)}$ </td><td>78.12</td><td>60.40</td><td>68.13</td><td>97.60</td><td>59.67</td><td>74.06</td><td>88.33</td><td>82.26</td><td>85.19</td><td>40.44</td><td>22.30</td><td>28.75</td></tr><tr><td> $PSRC_{(ICCV'23)}$ </td><td>78.27</td><td>74.97</td><td>76.58</td><td>98.07</td><td>76.50</td><td>85.95</td><td>90.67</td><td>91.53</td><td>91.10</td><td>42.73</td><td>37.87</td><td>40.15</td></tr><tr><td> $TAP_{(ICLR'25)}$ </td><td>80.70</td><td>74.27</td><td>77.35</td><td>97.90</td><td>75.37</td><td>85.30</td><td>90.97</td><td>91.83</td><td>91.40</td><td>40.40</td><td>36.50</td><td>40.06</td></tr><tr><td> $CLIP-AST_{(CVPR'25)}$ </td><td>84.21</td><td>74.05</td><td>78.80</td><td>97.91</td><td>77.73</td><td>86.66</td><td>90.57</td><td>91.11</td><td>90.84</td><td>48.98</td><td>38.21</td><td>42.93</td></tr><tr><td> $SurPL-G_{(ICML'25)}$ </td><td>83.57</td><td>72.77</td><td>77.80</td><td>98.90</td><td>72.88</td><td>83.92</td><td>90.92</td><td>91.81</td><td>91.36</td><td>49.20</td><td>36.93</td><td>42.19</td></tr><tr><td> $CoCoA-Mix_{(ICML'25)}$ </td><td>73.09</td><td>74.97</td><td>74.01</td><td>91.04</td><td>77.37</td><td>83.64</td><td>90.09</td><td>90.93</td><td>90.50</td><td>33.51</td><td>34.15</td><td>33.83</td></tr><tr><td>UPrompt(Ours)</td><td>83.58</td><td>74.57</td><td>78.82</td><td>98.54</td><td>78.43</td><td>87.34</td><td>91.20</td><td>92.16</td><td>91.68</td><td>49.33</td><td>39.25</td><td>43.72</td></tr><tr><td rowspan="2">Method</td><td colspan="3">SUN397</td><td colspan="3">DTD</td><td colspan="3">EuroSAT</td><td colspan="3">UCF101</td></tr><tr><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td> $CoOp_{(IJCV'22)}$ </td><td>80.60</td><td>65.89</td><td>72.51</td><td>79.44</td><td>41.18</td><td>54.24</td><td>93.19</td><td>54.74</td><td>68.69</td><td>84.69</td><td>56.05</td><td>67.46</td></tr><tr><td> $PSRC_{(ICCV'23)}$ </td><td>82.67</td><td>78.47</td><td>80.52</td><td>83.37</td><td>62.97</td><td>71.75</td><td>92.90</td><td>73.90</td><td>82.32</td><td>87.10</td><td>78.80</td><td>82.74</td></tr><tr><td> $TAP_{(ICLR'25)}$ </td><td>82.87</td><td>79.53</td><td>81.17</td><td>84.20</td><td>68.00</td><td>75.24</td><td>90.70</td><td>82.17</td><td>86.22</td><td>87.90</td><td>82.43</td><td>85.08</td></tr><tr><td> $CLIP-AST_{(CVPR'25)}$ </td><td>83.05</td><td>78.12</td><td>80.51</td><td>84.03</td><td>65.34</td><td>73.52</td><td>95.90</td><td>81.72</td><td>88.24</td><td>87.38</td><td>79.12</td><td>83.05</td></tr><tr><td> $SurPL-G_{(ICML'25)}$ </td><td>83.43</td><td>78.96</td><td>81.13</td><td>86.07</td><td>62.04</td><td>72.11</td><td>94.63</td><td>81.33</td><td>87.48</td><td>89.44</td><td>79.74</td><td>84.31</td></tr><tr><td> $CoCoA-Mix_{(ICML'25)}$ </td><td>78.51</td><td>76.60</td><td>77.54</td><td>72.80</td><td>64.29</td><td>68.25</td><td>83.49</td><td>69.11</td><td>75.54</td><td>81.28</td><td>77.75</td><td>79.47</td></tr><tr><td>UPrompt(Ours)</td><td>83.77</td><td>80.05</td><td>81.87</td><td>85.60</td><td>67.23</td><td>75.31</td><td>94.82</td><td>82.68</td><td>88.33</td><td>89.21</td><td>81.83</td><td>85.36</td></tr></table>

Table 2: Cross-modal retrieval performance of diferent CLIP fine-tuning methods. “ZS” denotes zero-shot and “FT” is finetuned. rSum is the sum of all R@1, R@5, and R@10 scores. Best results highlighted in first , second .

<table><tr><td rowspan="3">Methods</td><td colspan="7">Flickr30K</td></tr><tr><td colspan="3">Image-to-Text</td><td colspan="3">Text-to-Image</td><td rowspan="2">rSum</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>CLIP $_{ZS}$ </td><td>81.3</td><td>96.4</td><td>98.5</td><td>62.2</td><td>85.7</td><td>91.7</td><td>515.8</td></tr><tr><td>CLIP $_{FT}$ </td><td>91.7</td><td>99.0</td><td>99.5</td><td>79.1</td><td>95.2</td><td>97.6</td><td>562.1</td></tr><tr><td>DoPL(ACL&#x27;25)</td><td>69.8</td><td>90.7</td><td>95.0</td><td>66.9</td><td>89.0</td><td>93.6</td><td>505.0</td></tr><tr><td>MAMET(TCSVT&#x27;25)</td><td>92.7</td><td>99.3</td><td>99.7</td><td>79.8</td><td>95.2</td><td>97.2</td><td>563.9</td></tr><tr><td>VPKE(TCSVT&#x27;25)</td><td>93.7</td><td>99.2</td><td>99.8</td><td>82.0</td><td>95.7</td><td>98.2</td><td>568.6</td></tr><tr><td>UPrompt</td><td>93.8</td><td>99.4</td><td>99.6</td><td>83.6</td><td>96.3</td><td>98.4</td><td>571.1</td></tr><tr><td rowspan="3">Methods</td><td colspan="7">MSCOCO</td></tr><tr><td colspan="3">Image-to-Text</td><td colspan="3">Text-to-Image</td><td rowspan="2">rSum</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>CLIP $_{ZS}$ </td><td>52.5</td><td>76.6</td><td>84.7</td><td>33.1</td><td>58.4</td><td>69.0</td><td>374.3</td></tr><tr><td>CLIP $_{FT}$ </td><td>66.9</td><td>88.3</td><td>93.6</td><td>51.5</td><td>78.0</td><td>86.1</td><td>464.4</td></tr><tr><td>DoPL(ACL&#x27;25)</td><td>63.2</td><td>86.7</td><td>91.8</td><td>49.6</td><td>76.3</td><td>85.2</td><td>452.8</td></tr><tr><td>MAMET(TCSVT&#x27;25)</td><td>66.0</td><td>88.4</td><td>93.6</td><td>52.4</td><td>79.3</td><td>87.3</td><td>467.0</td></tr><tr><td>VPKE(TCSVT&#x27;25)</td><td>69.2</td><td>89.0</td><td>94.2</td><td>52.8</td><td>78.5</td><td>86.5</td><td>470.2</td></tr><tr><td>UPrompt</td><td>70.1</td><td>89.8</td><td>94.8</td><td>52.6</td><td>79.1</td><td>87.9</td><td>474.3</td></tr></table>

nouns (Level 2), attribute phrases (Level 3), and detailed descriptions (Level 4). In retrieval, original captions serve as medium granularity,

## 4.1 Comparative Results

Base-to-novel generalization. <sup>UPrompt</sup> <sup>achieves</sup> <sup>an</sup> <sup>82.12%</sup> <sup>har-</sup> monic mean (HM) across 11 datasets (Table 1), +1.06% improvement over the second best method. It outperforms recent multi-level methods like TAP (81.04%) [10], which constructs concept-attribute hierarchies, and SurPL-G (81.03%) [32], which generates diverse features across granularities, as well as latest methods like CLIP-AST (81.06%) [21] and CoCoA-Mix (77.03%) [19]. While TAP and SurPL-G treat granularities as independent modules, UPrompt’s U-shaped architecture establishes bidirectional information flow across hierarchical levels, providing more robust generalization with notable gains, including +0.79% HM on the challenging FGV-CAircraft. Appendix C.3 provides error bar analysis.

![](images/4a91014798ff1ff8b4f5337e4972dafca2e468186f417de4f948dbc0af448189.jpg)

![](images/4000f37699f374894954cfa6470c607d36f6f76af7d5a6e39d226e905bbc797c.jpg)

Figure 3: Few-shot classification. Performance on 11-dataset average and ImageNet. Remaining results are in the Appendix C.1.  
![](images/2cb5d205656b189d7feeb97aafcefa51a9597af04e97d00f58e43e9d59cf399d.jpg)  
Figure 4: Accuracy-Eficiency Trade-of on ImageNet. Static models (grey circles) operate at fixed granularities. Adaptive UPrompt (red star) matches fine-grained accuracy while re ducing computational cost to 48%, achieving 2.1× speedup.

<sub>Few-shot</sub> <sub>classification.</sub> UPrompt achieves a leading 85.13% averaged accuracy across 11 datasets at 16-shot (Fig. 3), surpassing recent methods: GalLoP [24] (84.50%), which learns from separate global and local features; ProVP-Ref [52] (83.07%), which builds progressive layer-wise connections; and MMA [55] (82.76%), which adapts only higher-level representations. On ImageNet, our method performs best across all shots. Unlike these approaches lacking systematic cross-level interaction, UPrompt’s bidirectional connection creates richer information flow across multi-granularity hierarchies, providing consistent gains from 1-shot to 16-shot.

<sub>Adaptive</sub> <sub>Classification.</sub> We exploit the hierarchical architecture of UPrompt to enable adaptive inference via a cascaded early-exit strategy. Inference proceeds sequentially from the coarsest granu larity (?? <sub>=</sub> 1) to the finest. At each level, the process terminates if the prediction confidence (maximum softmax probability) exceeds a calibrated threshold $\tau _ { k } ;$ otherwise, it continues to the next level via cascaded enhancement. To avoid data leakage, thresholds <sub>{</sub>??<sub>?? }</sub> are determined via grid search on a held-out calibration set. For each threshold candidate, we compute the classification accuracy and average computational cost over the calibration set. We select thresholds that maximize early exits while maintaining accuracy comparable to the static fine-grained model. Eficiency is quantified <sup>using</sup> Average Relative Cost<sup>:</sup> <sup>Avg.</sup> $\begin{array} { r } { \mathrm { C o s t } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { C _ { k _ { i } } } { C _ { K } } } \end{array}$ , where $C _ { k _ { i } }$ is the cumulative FLOPs required to reach exit level $k _ { i } ,$ and $C _ { K }$ is the cost of the full model.

Table 3: Out-of-distribution (OOD) generalization. ‘\*’ means reproduced results. Best results highlighted in first second .

<table><tr><td rowspan="2">Method</td><td>Source</td><td colspan="5">Target</td></tr><tr><td>ImgNet</td><td>-V2</td><td>-S</td><td>-A</td><td>-R</td><td>OOD</td></tr><tr><td>CoOp</td><td>71.51</td><td>64.44</td><td>47.61</td><td>49.53</td><td>74.98</td><td>59.14</td></tr><tr><td>PSRC</td><td>71.27</td><td>64.35</td><td>49.55</td><td>50.90</td><td>77.80</td><td>60.65</td></tr><tr><td>GalLoP*</td><td>71.14</td><td>64.32</td><td>49.56</td><td>50.83</td><td>77.42</td><td>60.53</td></tr><tr><td>SPTR</td><td>70.05</td><td>64.40</td><td>48.78</td><td>51.30</td><td>77.90</td><td>60.59</td></tr><tr><td>MMRL</td><td>72.03</td><td>64.47</td><td>49.13</td><td>51.20</td><td>77.53</td><td>60.58</td></tr><tr><td>HiCroPL</td><td>71.22</td><td>64.33</td><td>49.47</td><td>50.79</td><td>77.15</td><td>60.44</td></tr><tr><td>UPrompt</td><td>72.25</td><td>65.06</td><td>50.43</td><td>51.33</td><td>78.04</td><td>61.22</td></tr></table>

Table 4: Ablation study on Flickr30K. CE: coarse-to-fine Cascaded Enhancement; HS: fine-to-coarse Hierarchical Supervision. Baseline uses original image-text pairs; Finegrained only extends baseline with finest-granularity features; $\mathbf { ^ { * } g r a y } ^ { * }$ is our default set.

<table><tr><td rowspan="2">Granularity</td><td rowspan="2">Method</td><td colspan="2">Strategy</td><td colspan="2">I2T</td><td colspan="2">T2I</td></tr><tr><td>CE</td><td>HS</td><td>R@1</td><td>R@5</td><td>R@1</td><td>R@5</td></tr><tr><td rowspan="2">Single</td><td>Baseline</td><td>✕</td><td>✕</td><td>91.2</td><td>98.1</td><td>80.0</td><td>94.6</td></tr><tr><td>Fine-grained only</td><td>✕</td><td>✕</td><td>92.2</td><td>98.3</td><td>81.1</td><td>95.2</td></tr><tr><td rowspan="8">Multiple</td><td>(Effectiveness of CE)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vision only</td><td>√</td><td>✕</td><td>93.0</td><td>98.8</td><td>82.6</td><td>95.7</td></tr><tr><td>Text only</td><td>√</td><td>✕</td><td>92.7</td><td>98.6</td><td>82.4</td><td>95.5</td></tr><tr><td>Vision &amp; Text</td><td>√</td><td>✕</td><td>93.3</td><td>99.1</td><td>83.0</td><td>95.9</td></tr><tr><td>(Effectiveness of HS)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Coarse-to-fine</td><td>✕</td><td>√</td><td>86.8</td><td>95.1</td><td>75.3</td><td>91.5</td></tr><tr><td>Fine-to-coarse</td><td>✕</td><td>√</td><td>92.4</td><td>98.3</td><td>81.2</td><td>95.3</td></tr><tr><td>Full</td><td>√</td><td>√</td><td>93.8</td><td>99.4</td><td>83.6</td><td>96.3</td></tr></table>

As illustrated in Figure 4, Adaptive UPrompt achieves a favorable accuracy-eficiency trade-of. It attains accuracy comparable to the static fine-grained baseline (74.8% vs. 75.6%) while operating at only 48% of the average relative cost, translating to approximately 2.1<sub>×</sub> speedup. Notably, we observe that most samples exit at coarse granularities $( k \ = \ 1 , 2 )$ , validating that our hierarchical design efectively routes easy samples to computationally cheaper levels while reserving fine-grained resources for challenging instances. <sub>Cross-modal</sub> <sub>retrieval.</sub> We evaluate on Flickr30K and MSCOCO using Recall@K (K=1,5,10) and rSum (Table 2). UPrompt achieves rSum scores of 571.1 and 474.3 on Flickr30K and MSCOCO, outperforming recent CLIP-based fine-tuning methods. It surpasses DoPL [14] (505.0, 452.8), which generates layer-wise prompts for alignment, MAMET [48] (563.9, 467.0), which distills knowledge from multiple embeddings, and VPKE [47] (568.6, 470.2), which uses external visual knowledge. UPrompt’s superior results, with R@1 scores of 93.8% on Flickr30K and 70.1% on MSCOCO, stem from its unique architecture. Its multi-granularity prompting captures vision-language alignments across various semantic levels, while hierarchical contextual guidance leads to significant improvement in retrieval precision. Appendix B further explores adaptive retrieval strategies.

![](images/5418ef96eea148ea15b912988407d05ccdbf59ee09febc624ae46daf67f2be6e.jpg)

![](images/26999a411c07b2340f6ba3f2282c4bf578960e6d05cc700ae3361a7e15c90ab8.jpg)  
Table 5: Text hierarchy robustness across diferent LLMs on Flickr30K.

<table><tr><td colspan="4">Image-to-text</td></tr><tr><td>Models</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>Qwen3-4B</td><td>93.4</td><td>99.4</td><td>99.5</td></tr><tr><td>Llama 3-8B</td><td>93.8</td><td>99.4</td><td>99.6</td></tr><tr><td>Qwen3-14B</td><td>93.7</td><td>99.5</td><td>99.8</td></tr></table>

Figure 5: Granularity number and prompt length efects on classification. Left: varied granularity number (1-4). Right: varied prompt length (1-8).  
Figure 6: Eficiency-performance trade-ofs across granularity levels. C, M, F denote coarse, medium, and fine configurations respectively.

<table><tr><td>Models</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>Qwen3-4B</td><td>83.4</td><td>96.2</td><td>98.6</td></tr><tr><td>Llama 3-8B</td><td>83.6</td><td>96.3</td><td>98.4</td></tr><tr><td>Qwen3-14B</td><td>83.9</td><td>96.2</td><td>98.7</td></tr></table>

Table 6: Resolution×granularity level ablation on Flickr30K (I2T R@1). Starting from the coarsest level, we progressively add finer granularity levels.

<table><tr><td>Resolution</td><td>Level 1</td><td>Level 2</td><td>Level 3</td></tr><tr><td> $224 \times 224$ </td><td>82.8</td><td>89.4</td><td>93.8</td></tr><tr><td> $336 \times 336$ </td><td>83.3</td><td>92.2</td><td>95.1</td></tr></table>

Out-of-distribution Generalization. <sup>Domain</sup> <sup>shift</sup> <sup>evaluation</sup> <sup>ex</sup> amines semantic preservation (Table 3). GalLoP [24] sparse feature selection loses cross-domain information, SPTR [7] optimal trans port maintains single-granularity stability, MMRL [13] representa tion learning preserves generalization through decoupling, while HiCroPL [62] bidirectional refinement focuses on task-specific alignment rather than domain robustness (60.44%). UPrompt achieves 61.22% through multi-granularity architecture with cascaded en hancement providing global guidance and hierarchical supervision preventing semantic drift.

## 4.2 Ablation study

<sub>Component</sub> <sub>Ablation.</sub> We conduct an ablation study on Flickr30K (Table 4) to isolate the contributions of our core components: coarseto-fine cascaded enhancement (CE) and fine-to-coarse hierarchical supervision (HS). We use the “Fine-grained only” model (92.2% I2T R@1) as our single-scale reference. Without CE, fine-grained fea tures in multi-granularity architectures are very similar to singlegranularity features. CE improves performance by maintaining global context while preserving fine details in “Vision only” mode (93.0% I2T R@1), providing progressive textual enhancement in “Text only” mode (92.7% I2T R@1), and enabling comprehensive local relationship modeling under global guidance when combined for “Vision & Text” (93.3% I2T R@1). The directional nature of HS is critical; reversed coarse-to-fine supervision severely degrades performance by forcing detailed representations to model ambigu ous signals. Conversely, fine-to-coarse hierarchical supervision in isolation ofers negligible improvement, as enforcing semantic consistency at coarse levels lacks a mechanism to refine independently optimized fine-grained prompts. The full UPrompt model achieves best results (93.8% I2T R@1), revealing crucial connection where CE establishes cross-scale feature dependencies, allowing multi-level semantic consistency enforced by HS to efectively improve the entire representation hierarchy.

Granularity Number and Prompt Length. <sup>We</sup> <sup>analyze</sup> <sup>sensitiv-</sup> ity of UPrompt to the number of granularities and prompt lengths (Fig. 5). Starting from the finest level and progressively adding coarser levels, performance consistently improves as the number increases from 1 to 4, with harmonic mean reaching 82.12% versus 80.33% at single finest granularity, though improvement rate gradually slows with increasing computational overhead. We therefore adopt 4 granularity levels balancing performance and eficiency. For the downsampling strategy between granularities, following U-Net’s 2<sub>×</sub>2 downsampling principle [41], we employ approximate halving of visual token resolution (14 <sub>×</sub> 14 <sub>→</sub> 7 <sub>×</sub> 7 <sub>→</sub> 4 <sub>×</sub> 4 for retrieval), which demonstrates superior performance over sparser intervals as analyzed in Appendix D.2. For prompt length, performance in Fig. 5 (right) peaks at length 4 and remains stable at length 8, thus we set prompt length to 4.

<sub>Resolution</sub> <sub>Analysis.</sub> To validate that multi-granularity improvements stem from semantic hierarchy rather than token density compensation, we conduct resolution-scale ablation on Flickr30K (Table 6). Results show that increasing the granularity level (Level 1<sub>→</sub>3) yields substantial gains at both 224<sub>×</sub>224 (82.8%<sub>→</sub>93.8%) and 336<sub>×</sub>336 (83.3%<sub>→</sub>95.1%), while resolution alone improves marginally (82.8%<sub>→</sub>83.3%). This demonstrates multi-granularity benefits dominate across resolutions, confirming our method addresses semantic hierarchy independent of spatial token density. Appendix D.3 further validates consistent improvements with denser backbones (ViT-L), demonstrating framework generalization across model scales. Analysis of Performance-Cost. <sup>Fig.</sup> <sup>6</sup> <sup>reveals</sup> <sup>the</sup> <sup>performance</sup> and cost across diferent granularities, where UPrompt-C, UPrompt-M, and UPrompt-F are coarse (1<sub>×</sub>1), medium (7<sub>×</sub>7), and fine (14<sub>×</sub>14) visual tokens with corresponding textual granularities. UPrompt-F outperforms existing prompt learning methods with 97.22% average HM across OxfordPets and Caltech101 with limited additional cost. UPrompt-M achieves comparable performance (95.97% average HM) and matches PSRC’s accuracy using only 1/3 of PSRC’s FLOPs.

![](images/6ac6ed005a679197b030abc94182318468b3bdec39ffb157cd234c6aab9d339e.jpg)  
Figure 7: Multi-granularity visual attention visualization. Heat maps show bidirectional connection efects with cascaded enhancement preserving global context in fine-grained attention while hierarchical supervision reduces semantic drift across diferent levels.

Table 7: Efectiveness of Cascaded Enhancement (CE) and Hierarchical Supervision (HS) on cross-modal retrieval on Flickr30K with diferent granularity sets.

<table><tr><td rowspan="2">Granularity</td><td rowspan="2">Method</td><td colspan="3">Image-to-text</td><td colspan="3">Text-to-image</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td rowspan="2">Coarse-grained</td><td>w/o HS</td><td>75.4</td><td>89.9</td><td>92.8</td><td>60.4</td><td>82.3</td><td>86.8</td></tr><tr><td>w/ HS</td><td>82.8</td><td>92.4</td><td>93.5</td><td>67.6</td><td>87.5</td><td>90.3</td></tr><tr><td rowspan="2">Fine-grained</td><td>w/o CE</td><td>92.7</td><td>98.5</td><td>99.2</td><td>81.7</td><td>95.8</td><td>98.0</td></tr><tr><td>w/ CE</td><td>93.8</td><td>99.4</td><td>99.6</td><td>83.6</td><td>96.3</td><td>98.4</td></tr></table>

UPrompt-C requires minimal resources while preserving reasonable performance at 91.47% average HM. This flexibility stems from our architecture where fine-to-coarse supervision enables coarse levels to benefit from detailed representations, allowing adaptive granularity selection based on resource constraints.

<sub>Robustness</sub> <sub>Analysis.</sub> We evaluate the sensitivity of our text hi erarchy generation to diferent LLMs by conducting experiments with Qwen3-4B, Qwen3-14B [54] and Llama3-8B on Flickr30K cross modal retrieval (Table 5). The performance remains stable across models with varying architectures and scales (4B to 14B parameters), confirming our method is not sensitive to the specific LLM used. Appendix E validates that our bidirectional connection mechanism remains efective even with simple rule-based text hierarchies. Additional robustness validation across diferent backbones and VLM architectures is provided in Appendix D.3 and D.4.

Analysis of Bidirectional Information Flow. <sup>We</sup> <sup>conduct</sup> <sup>an</sup> ablation study on Flickr30K to isolate our bidirectional connection components, with results in Table 7. Fine-to-coarse Hierarchical Supervision (HS) substantially boosts coarse-grained performance (I2T R@1 from 75.4% to 82.8%), addressing the semantic drift caused by optimizing on the ambiguous signals inherent in simplified rep resentations. Coarse-to-fine Cascaded Enhancement (CE) improves fine-grained performance (I2T R@1 from 92.7% to 93.8%), resolving context deficiency from isolated local detail modeling, without an understanding of their role within the global scene. Both compo nents are integral: HS maintains semantic consistency for coarse representations, while CE provides contextual guidance for fine ones. To verify fine-grained supervision reliability, we compared single fine-layer supervision against mixed fine and medium-layer supervision (Appendix D.1). Results confirm fine-layer supervision alone achieves comparable performance, validating its suficiency as the teacher signal.

## 4.3 Visualization

Visualization of Multi-Granularity Attention. <sup>Fig.</sup> <sup>7</sup> <sup>validates</sup> UPrompt’s bidirectional connection. Cascaded Enhancement enables fine-grained attention (14<sub>×</sub>14) to maintain global coherence while capturing details like winter gear and textures. Hierarchical Supervision ensures coarser levels (4<sub>×</sub>4, 7<sub>×</sub>7) focus on semantically relevant regions, preventing background noise interference and semantic drift. The progressive refinement from global to local demonstrates efective multi-scale integration, where each granularity captures complementary information while maintaining semantic consistency. Fig. 9 in Appendix F.1 further shows our multi-granularity design across retrieval.

Visualization of Bidirectional Connection Components. <sup>Fig.</sup> <sup>10</sup> and Fig. 11 in Appendix F.2, F.3 visualize Coarse-to-Fine Enhancement injects global context into fine-grained representations for improved local modeling, and Fine-to-Coarse Supervision leverages finest-level alignment to regularize and maintain consistency across coarser granularities.

## 5 Conclusion

In this work, we present UPrompt, a simple yet efective framework that addresses the limitation of single granularity in visionlanguage prompt learning. Inspired by U-Net, our U-shaped framework constructs parallel multi-granularity representations with bidirectional connections to facilitate information flow across scales. This consists of coarse-to-fine enhancement that injects global context into local details, and fine-to-coarse supervision that ensures semantic consistency. Extensive experiments demonstrate efectiveness across cross-modal retrieval, base-to-novel generalization, and few-shot classification. Despite its efectiveness, UPrompt’s hierarchical depth remains limited, constraining its capacity to model richer semantic structures. Future work will explore deeper multi-granularity architectures with more hierarchical levels.

## References

[1] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. 2014. Food-101–mining discriminative components with random forests. In <sub>Computer</sub> <sub>vision–ECCV</sub> <sub>2014:</sub> 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, <sub>part</sub> <sub>VI</sub> <sub>13</sub>. Springer, 446–461.

[2] Biao Chen, Kunbin He, Zhikun Zheng, Mengmeng Jing, and Lin Zuo. 2025. Chain of-Thought Guided Semantic Debiasing for Low-Shot Vision-Language Tasks. In Proceedings of the 33rd ACM International Conference on Multimedia<sup>.</sup> <sup>4600–4609.</sup>

[3] Biao Chen, Lin Zuo, Mengmeng Jing, Kunbin He, and Yuchen Wang. 2026. Dropout Prompt Learning: Towards Robust and Adaptive Vision-Language Mod <sup>els.</sup> <sup>In</sup> Proceedings of the AAAI Conference on Artificial Intelligence<sup>,</sup> <sup>Vol.</sup> <sup>40.</sup> <sup>19987–</sup> 19995.

[4] Xiang Chen, Jinshan Pan, and Jiangxin Dong. 2024. Bidirectional multi-scale implicit neural representations for image deraining. In <sub>Proceedings</sub> <sub>of</sub> <sub>the</sub> <sub>IEEE/CVF</sub> conference on computer vision and pattern recognition<sup>.</sup> <sup>25627–25636.</sup>

[5] Eulrang Cho, Jooyeon Kim, and Hyunwoo J Kim. 2023. Distribution-aware prompt <sup>tuning</sup> <sup>for</sup> <sup>vision-language</sup> <sup>models.</sup> <sup>In</sup> Proceedings of the IEEE/CVF international conference on computer vision<sup>.</sup> <sup>22004–22013.</sup>

[6] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and An drea Vedaldi. 2014. Describing textures in the wild. In <sub>Proceedings</sub> <sub>of</sub> <sub>the</sub> <sub>IEEE</sub> conference on computer vision and pattern recognition<sup>.</sup> <sup>3606–3613.</sup>

[7] Fangming Cui, Jan Fong, Rongfei Zeng, Xinmei Tian, and Jun Yu. 2025. A Similarity Paradigm Through Textual Regularization Without Forgetting. In Proceedings of the AAAI Conference on Artificial Intelligence<sup>,</sup> <sup>Vol.</sup> <sup>39.</sup> <sup>16100–16108.</sup>

[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. Imagenet: A large-scale hierarchical image database. In <sub>2009</sub> <sub>IEEE</sub> <sub>conference</sub> <sub>on</sub> <sub>computer</sub> vision and pattern recognition<sup>.</sup> <sup>Ieee,</sup> <sup>248–255.</sup>

[9] Jian Ding, Nan Xue, Gui-Song Xia, Bernt Schiele, and Dengxin Dai. 2023. Hg former: Hierarchical grouping transformer for domain generalized semantic <sup>segmentation.</sup> <sup>In</sup> Proceedings of the IEEE/CVF conference on computer vision and pattern recognition<sup>.</sup> <sup>15413–15423.</sup>

[10] Tong Ding, Wanhua Li, Zhongqi Miao, and Hanspeter Pfister. 2025. Tree of attributes prompt learning for vision-language models. In <sub>International</sub> <sub>Conference</sub> on Learning Representations<sup>,</sup> <sup>Vol.</sup> <sup>2025.</sup> <sup>50495–50514.</sup>

[11] Li Fei-Fei, Rob Fergus, and Pietro Perona. 2004. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 <sup>object</sup> <sup>categories.</sup> <sup>In</sup> 2004 conference on computer vision and pattern recognition <sub>workshop</sub>. IEEE, 178–178.

[12] Feng Guo, Biao Chen, Zhongshu Chen, Zhikun Zheng, Lin Zuo, and Wen Li. 2025. DviT: Debiased variational inference for multi-modal mutual prompt tuning. Knowledge-Based Systems <sup>324</sup> <sup>(2025),</sup> <sup>113798.</sup>

[13] Yuncheng Guo and Xiaodong Gu. 2025. Mmrl: Multi-modal representation learning for vision-language models. In <sub>Proceedings</sub> <sub>of</sub> <sub>the</sub> <sub>Computer</sub> <sub>Vision</sub> <sub>and</sub> Pattern Recognition Conference<sup>.</sup> <sup>25015–25025.</sup>

[14] Yongbin Guo, Shuzhen Li, Zhulin Liu, Tong Zhang, and CL Philip Chen. 2025. A Parameter-Eficient and Fine-Grained Prompt Learning for Vision-Language <sup>Models.</sup> <sup>In</sup> Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)<sup>.</sup> <sup>31346–31359.</sup>

[15] Fusheng Hao, Fengxiang He, Fuxiang Wu, Tichao Wang, Chengqun Song, and Jun Cheng. 2025. Task-Aware Clustering for Prompting Vision-Language Models. <sup>In</sup> Proceedings of the Computer Vision and Pattern Recognition Conference<sup>.</sup> <sup>14745–</sup> 14755.

[16] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. 2019. Eurosat: A novel dataset and deep learning benchmark for land use and land <sup>cover</sup> <sup>classification.</sup> IEEE Journal of Selected Topics in Applied Earth Observations <sub>and</sub> <sub>Remote</sub> <sub>Sensing</sub> 12, 7 (2019), 2217–2226.

[17] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. 2021. The many faces of robustness: A critical analysis of out-of-distribution generalization. <sup>In</sup> Proceedings of the IEEE/CVF international conference on computer vision<sup>.</sup> <sup>8340–</sup> 8349.

[18] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. <sup>2021.</sup> <sup>Natural</sup> <sup>adversarial</sup> <sup>examples.</sup> <sup>In</sup> Proceedings of the IEEE/CVF conference on computer vision and pattern recognition<sup>.</sup> <sup>15262–15271.</sup>

[19] Dasol Hong, Wooju Lee, and Hyun Myung. [n. d.]. CoCoA-Mix: Confusion-and Confidence-Aware Mixture Model for Context Optimization. ([n. d.]).

[20] Chen Huang, Skyler Seto, Samira Abnar, David Grangier, Navdeep Jaitly, and Joshua Susskind. 2024. Aggregate-and-adapt natural language prompts for down <sup>stream</sup> <sup>generalization</sup> <sup>of</sup> <sup>clip.</sup> Advances in Neural Information Processing Systems 37 (2024), 81077–81104.

[21] Xin Huang, Shilong Wang, Tong Jia, Zhihang Gou, and Jingjing Li. 2025. Adaptive prompt-based semantic embedding with inspire potential of implicit knowledge <sup>for</sup> <sup>cross-modal</sup> <sup>retrieval.</sup> <sup>In</sup> Proceedings of the AAAI Conference on Artificial <sub>Intelligence</sub>, Vol. 39. 17485–17493

[22] Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. 2023. Maple: Multi-modal prompt learning. In <sub>Pro-</sub> ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition<sup>.</sup>

19113–19122.

[23] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 2013. 3d object representations for fine-grained categorization. In <sub>Proceedings</sub> <sub>of</sub> <sub>the</sub> <sub>IEEE</sub> <sub>international</sub> conference on computer vision workshops<sup>.</sup> <sup>554–561.</sup>

[24] Marc Lafon, Elias Ramzi, Clément Rambour, Nicolas Audebert, and Nicolas Thome. 2024. Gallop: Learning global and local prompts for vision-language models. In European Conference on Computer Vision<sup>.</sup> <sup>Springer,</sup> <sup>264–282</sup>

[25] Gen Li, Nan Duan, Yuejian Fang, Ming Gong, and Daxin Jiang. 2020. Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training. In Proceedings of the AAAI conference on artificial intelligence<sup>,</sup> <sup>Vol.</sup> <sup>34.</sup> <sup>11336–11344.</sup>

[26] Jiahao Li, Yang Lu, Yuan Xie, and Yanyun Qu. 2024. Relationship prompt learning is enough for open-vocabulary semantic segmentation. <sub>Advances</sub> <sub>in</sub> <sub>Neural</sub> Information Processing Systems <sup>37</sup> <sup>(2024),</sup> <sup>74298–74324</sup>

[27] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. 2020. Oscar: Object-semantics aligned pre-training for vision-language tasks. In <sub>European</sub> <sub>conference</sub> <sub>on</sub> <sub>computer</sub> <sub>vision</sub>. Springer, 121–137.

[28] Yilun Li, Miaomiao Cheng, Xu Han, and Wei Song. 2025. Divergence-enhanced knowledge-guided context optimization for visual-language prompt tuning. In The Thirteenth International Conference on Learning Representations<sup>.</sup>

[29] Guosheng Lin, Anton Milan, Chunhua Shen, and Ian Reid. 2017. Refinenet: Multi-path refinement networks for high-resolution semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition<sup>.</sup> 1925–1934.

[30] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. 2017. Feature pyramid networks for object detection. In <sub>Proceed-</sub> ings of the IEEE conference on computer vision and pattern recognition<sup>.</sup> <sup>2117–2125.</sup>

[31] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In <sub>European</sub> <sub>conference</sub> <sub>on</sub> <sub>computer</sub> <sub>vision</sub>. Springer, 740–755.

[32] Liangchen Liu, Nannan Wang, Xi Yang, Xinbo Gao, and Tongliang Liu. [n. d.]. Surrogate Prompt Learning: Towards Eficient and Diverse Prompt Learning for <sup>Vision-Language</sup> <sup>Models.</sup> <sup>In</sup> Forty-second International Conference on Machine Learning<sup>.</sup>

[33] Ruiying Lu, YuJie Wu, Long Tian, Dongsheng Wang, Bo Chen, Xiyang Liu, and Ruimin Hu. 2023. Hierarchical vector quantized transformer for multi-class <sup>unsupervised</sup> <sup>anomaly</sup> <sup>detection.</sup> Advances in Neural Information Processing <sub>Systems</sub> 36 (2023), 8487–8500.

[34] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. 2013. Fine-grained visual classification of aircraft. <sub>arXiv</sub> <sub>preprint</sub> <sub>arXiv:1306.5151</sub> (2013).

[35] Alejandro Newell, Kaiyu Yang, and Jia Deng. 2016. Stacked hourglass networks for human pose estimation. In <sub>European</sub> <sub>conference</sub> <sub>on</sub> <sub>computer</sub> <sub>vision</sub>. Springer, 483–499.

[36] Maria-Elena Nilsback and Andrew Zisserman. 2008. Automated flower classifica-<sup>tion</sup> <sup>over</sup> <sup>a</sup> <sup>large</sup> <sup>number</sup> <sup>of</sup> <sup>classes.</sup> <sup>In</sup> 2008 Sixth Indian conference on computer vision, graphics & image processing<sup>.</sup> <sup>IEEE,</sup> <sup>722–729.</sup>

[37] Wonpyo Park, Dongju Kim, Yan Lu, and Minsu Cho. 2019. Relational knowledge <sup>distillation.</sup> <sup>In</sup> Proceedings of the IEEE/CVF conference on computer vision and pattern recognition<sup>.</sup> <sup>3967–3976.</sup>

[38] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. 2012. <sup>Cats</sup> <sup>and</sup> <sup>dogs.</sup> <sup>In</sup> 2012 IEEE conference on computer vision and pattern recognition<sup>.</sup> IEEE, 3498–3505.

[39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. <sup>In</sup> International conference on machine learning<sup>.</sup> <sup>PmLR,</sup> <sup>8748–8763.</sup>

[40] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. 2019. Do imagenet classifiers generalize to imagenet?. In <sub>International</sub> <sub>conference</sub> <sub>on</sub> <sub>machine</sub> <sub>learning</sub>. PMLR, 5389–5400.

[41] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-net: Convolutional networks for biomedical image segmentation. In <sub>International</sub> <sub>Conference</sub> <sub>on</sub> Medical image computing and computer-assisted intervention<sup>.</sup> <sup>Springer,</sup> <sup>234–241.</sup>

[42] Shuvendu Roy and Ali Etemad. 2024. Consistency-guided Prompt Learning for <sup>Vision-Language</sup> <sup>Models.</sup> <sup>In</sup> International Conference on Learning Representations<sup>.</sup>

[43] Yuheng Shi, Minjing Dong, and Chang Xu. 2024. Multi-scale vmamba: Hierarchy <sup>in</sup> <sup>hierarchy</sup> <sup>visual</sup> <sup>state</sup> <sup>space</sup> <sup>model.</sup> Advances in Neural Information Processing <sub>Systems</sub> 37 (2024), 25687–25708

[44] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. 2012. UCF101: A dataset of 101 human actions classes from videos in the wild. <sub>arXiv</sub> <sub>preprint</sub> arXiv:1212.0402 <sup>(2012).</sup>

[45] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. 2011. The caltech-ucsd birds-200-2011 dataset. (2011).

[46] Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. 2019. Learning robust global representations by penalizing local predictive power. <sub>Advances</sub> <sub>in</sub> Neural Information Processing Systems <sup>32</sup> <sup>(2019)</sup>

[47] Hengchang Wang, Li Liu, Huaxiang Zhang, Lei Zhu, Xiaojun Chang, and Hao Du. 2025. VisualRAG: Knowledge-Guided Retrieval Augmentation for Image-Text

<sup>Matching.</sup> IEEE Transactions on Circuits and Systems for Video Technology <sup>(2025).</sup>

[48] Pengzhe Wang, Lei Zhang, Zhendong Mao, Nenan Lyu, and Yongdong Zhang. 2025. Matryoshka Learning with Metric Transfer for Image-text Matching. <sub>IEEE</sub> Transactions on Circuits and Systems for Video Technology <sup>(2025).</sup>

[49] Christopher Williams, Fabian Falck, George Deligiannidis, Chris C Holmes, Ar naud Doucet, and Saifuddin Syed. 2023. A unified framework for U-Net design <sup>and</sup> <sup>analysis.</sup> Advances in Neural Information Processing Systems <sup>36</sup> <sup>(2023),</sup> <sup>27745–</sup> 27782.

[50] Yongqin Xian, Bernt Schiele, and Zeynep Akata. 2017. Zero-shot learning-the <sup>good,</sup> <sup>the</sup> <sup>bad</sup> <sup>and</sup> <sup>the</sup> <sup>ugly.</sup> <sup>In</sup> Proceedings of the IEEE conference on computer vision and pattern recognition<sup>.</sup> <sup>4582–4591.</sup>

[51] Jianxiong Xiao, Krista A Ehinger, James Hays, Antonio Torralba, and Aude Oliva. 2016. Sun database: Exploring a large collection of scene categories. <sub>International</sub> Journal of Computer Vision <sup>119</sup> <sup>(2016),</sup> <sup>3–22.</sup>

[52] Chen Xu, Yuhan Zhu, Haocheng Shen, Boheng Chen, Yixuan Liao, Xiaoxin Chen, and Limin Wang. 2025. Progressive visual prompt learning with contrastive <sup>feature</sup> <sup>re-formation.</sup> International Journal of Computer Vision <sup>133,</sup> <sup>2</sup> <sup>(2025),</sup> 511–526.

[53] Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, and Xiaolong Wang. 2022. Groupvit: Semantic segmentation emerges from text <sup>supervision.</sup> <sup>In</sup> Proceedings of the IEEE/CVF conference on computer vision and pattern recognition<sup>.</sup> <sup>18134–18144.</sup>

[54] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical <sup>report.</sup> arXiv preprint arXiv:2505.09388 <sup>(2025).</sup>

[55] Lingxiao Yang, Ru-Yuan Zhang, Yanchen Wang, and Xiaohua Xie. 2024. Mma: Multi-modal adapter for vision-language models. In <sub>Proceedings</sub> <sub>of</sub> <sub>the</sub> <sub>IEEE/CVF</sub> Conference on Computer Vision and Pattern Recognition<sup>.</sup> <sup>23826–23837.</sup>

[56] Hantao Yao, Rui Zhang, and Changsheng Xu. 2023. Visual-language prompt tun ing with knowledge-guided context optimization. In <sub>Proceedings</sub> <sub>of</sub> <sub>the</sub> <sub>IEEE/CVF</sub> conference on computer vision and pattern recognition<sup>.</sup> <sup>6757–6767.</sup>

[57] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. 2014. From image descriptions to visual denotations: New similarity metrics for semantic <sup>inference</sup> <sup>over</sup> <sup>event</sup> <sup>descriptions.</sup> Transactions of the association for computational <sub>linguistics</sub> 2 (2014), 67–78.

[58] Fei Yu, Jiji Tang, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. 2021. Ernie-vil: Knowledge enhanced vision-language representations through <sup>scene</sup> <sup>graphs.</sup> <sup>In</sup> Proceedings of the AAAI conference on artificial intelligence<sup>,</sup> Vol. 35. 3208–3216.

[59] Yunqian Yu, Biao Chen, Yunya Zhang, Tonglan Xie, Mengmeng Jing, and Lin Zuo. 2026. Instruction-guided cross-modal clustering for training-free visual token pruning in vision-language models. In <sub>Proceedings</sub> <sub>of</sub> <sub>the</sub> <sub>AAAI</sub> <sub>Conference</sub> on Artificial Intelligence<sup>,</sup> <sup>Vol.</sup> <sup>40.</sup> <sup>12213–12221.</sup>

[60] Yunqian Yu, Feng Guo, Xianlong Tian, Biao Chen, Mengmeng Jing, and Lin Zuo. 2025. Visual residual aggregation network for visual-language prompt tuning. Applied Intelligence <sup>55,</sup> <sup>15</sup> <sup>(2025),</sup> <sup>988.</sup>

[61] Gangming Zhao, Weifeng Ge, and Yizhou Yu. 2021. GraphFPN: Graph feature <sup>pyramid</sup> <sup>network</sup> <sup>for</sup> <sup>object</sup> <sup>detection.</sup> <sup>In</sup> Proceedings of the IEEE/CVF international conference on computer vision<sup>.</sup> <sup>2763–2772.</sup>

[62] Hao Zheng, Shunzhi Yang, Zhuoxin He, Jinfeng Yang, and Zhenhua Huang. 2025. Hierarchical Cross-modal Prompt Learning for Vision-Language Models. <sub>arXiv</sub> preprint arXiv:2507.14976 <sup>(2025).</sup>

[63] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. 2022. Learning <sup>to</sup> <sup>prompt</sup> <sup>for</sup> <sup>vision-language</sup> <sup>models.</sup> International Journal of Computer Vision 130, 9 (2022), 2337–2348.

[64] Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, and Jianming Liang. 2019. Unet++: Redesigning skip connections to exploit multiscale features in image segmentation. <sub>IEEE</sub> <sub>transactions</sub> <sub>on</sub> <sub>medical</sub> <sub>imaging</sub> 39, 6 (2019), 1856– 1867.

## Appendix

The appendix provides supplementary materials including theoretical proofs, extended experiments, ablation studies, and visualizations. The contents are structured as follows:

<sub>•</sub> Theoretical proofs for Propositions A.1 and A.2 (Appendix A).

<sub>•</sub> Adaptive strategies for retrieval (Appendix B).

<sub>•</sub> Extended experiments on few-shot classification, cross-dataset transfer and error bar analysis (Appendix C).

<sub>•</sub> Ablation studies on supervision reliability, granularity interval strategy and backbone architectures (Appendix D).

<sub>•</sub> Sensitivity analysis on LLM-generated priors for text hierarchy construction (Appendix E).

<sub>•</sub> Visualizations of multi-granularity retrieval, cascaded enhancement, and hierarchical supervision (Appendix F).

## A Proof for the Proposition

## A.1 Proof for Proposition 3.1

Proposition A.1 (CE Directional Alignment Effect). <sub>Let</sub> $\hat { X } ^ { ( k ) }$ be the fine-grained representation at level <sup>??</sup> enhanced by coarse-to-fine cascaded enhancement (CE, Eq. (5)–(6)), which leverages contextual guidance from the coarser representation $\hat { X } ^ { ( k - 1 ) }$ . Let $X ^ { ( k ) }$ be its unenhanced counterpart. Under the mild assumption that the coarse context is informative, CE provably strengthens the alignment between fine-grained features and their coarse-grained guidance in expectation:

$$
\mathbb {E} \left[ \frac {\langle \hat {X} ^ {(k)} , \hat {X} ^ {(k - 1)} \rangle}{\| \hat {X} ^ {(k)} \| \| \hat {X} ^ {(k - 1)} \|} \right] \geq \mathbb {E} \left[ \frac {\langle X ^ {(k)} , \hat {X} ^ {(k - 1)} \rangle}{\| X ^ {(k)} \| \| \hat {X} ^ {(k - 1)} \|} \right].
$$

Proof. <sup>Let</sup> $u \triangleq \hat { X } ^ { ( k - 1 ) } / \Vert \hat { X } ^ { ( k - 1 ) } \Vert$ be the unit coarse direction. Denote element-wise absolute value by <sub>|</sub> <sub>·</sub> <sub>|</sub>, and define the “sign-adjusted” vector $\boldsymbol { v } \in \mathbb { R } ^ { d }$ by $v _ { i } \triangleq \operatorname { s i g n } ( X _ { i } ^ { ( k ) } u _ { i } )$ <sub>|</sub>??<sub>?? |</sub>. Then for any nonnegative gate ?? <sub>∈</sub> $\mathbb { R } _ { \geq 0 } ^ { d }$ we can rewrite the cosine with ?? as

$$
\frac {\left\langle X ^ {(k)} \odot a , u \right\rangle}{\| X ^ {(k)} \odot a \|} = \frac {\left\langle | X ^ {(k)} | \odot a , v \right\rangle}{\left\| | X ^ {(k)} | \odot a \right\|}.
$$

Under CE $\left( \operatorname { E q . } \left( 5 \right) - \left( 6 \right) \right)$ , the enhanced representation takes the form $\hat { X } ^ { ( k ) } { = } X ^ { ( k ) } \odot A \big ( X ^ { ( k ) } , \hat { X } ^ { ( k - 1 ) } \big )$ with an <sub>element-wise</sub> nonnegative map ?? produced from cross-granularity softmax attention and a value mixing of $\hat { X } ^ { ( k - 1 ) }$ ; we treat $a { = } A { \left( X ^ { ( k ) } , \hat { X } ^ { ( k - 1 ) } \right) }$ as the random gate induced by CE.

<sub>Monotone</sub> <sub>informative-gate</sub> <sub>(MIG)</sub> <sub>assumption</sub> [37]. Formally, we instantiate the “coarse context is informative” condition as:

$\mathbb { E } \big [ a _ { i } \big | X ^ { ( k ) } , u \big ]$ is coordinatewise nondecreasing in $r _ { i } \triangleq \frac { v _ { i } } { | X _ { i } ^ { ( k ) } | }$

(12)

i.e., coordinates that are better aligned with ?? (larger ??<sub>??</sub> ) receive, in expectation, larger CE weights.<sup>1</sup>

Directional-derivative lemma. <sup>Define</sup> $f ( a ) \triangleq { \frac { \langle | X ^ { ( k ) } | \odot a , \ v \rangle } { \left\| | X ^ { ( k ) } | \odot a \right\| } }$ . A direct calculation gives, for any coordinate $i ,$

$$
\frac {\partial f}{\partial a _ {i}} (a) = \frac {| X _ {i} ^ {(k)} | v _ {i}}{\left\| | X ^ {(k)} | \odot a \right\|} - f (a) \cdot \frac {a _ {i} | X _ {i} ^ {(k)} | ^ {2}}{\left\| | X ^ {(k)} | \odot a \right\|}.
$$

Evaluated at the identity gate $a = 1 ,$ , letting $f _ { 0 } \triangleq f ( 1 ) = \frac { \langle | X ^ { ( k ) } | , v \rangle } { \| X ^ { ( k ) } \| }$ , we obtain the directional derivative along any perturbation ??:

$$
\left. \frac {d}{d \alpha} f (\mathbf {1} + \alpha w) \right| _ {\alpha = 0} = \frac {1}{\| X ^ {(k)} \|} \sum_ {i = 1} ^ {d} w _ {i} | X _ {i} ^ {(k)} | \left(\frac {v _ {i}}{| X _ {i} ^ {(k)} |} - f _ {0}\right) = \frac {1}{\| X ^ {(k)} \|} \sum_ {i = 1} ^ {d} w _ {i} | X _ {i} ^ {(k)} | (r _ {i} - f _ {0}).
$$

Hence the first-order increase of $f$ at <sub>1</sub> is nonnegative whenever ?? is (on average) positively associated with the alignment score ??.

<sub>Homotopy</sub> <sub>argument.</sub> Consider the linear path $a ( t ) = 1 + t \left( a - 1 \right)$ for $t \in \left[ 0 , 1 \right]$ . Diferentiating along the path,

$$
\frac {d}{d t} f (a (t)) = \sum_ {i = 1} ^ {d} \left(a _ {i} - 1\right) \frac {\partial f}{\partial a _ {i}} (a (t)).
$$

By continuity of $\frac { \partial f } { \partial a _ { i } }$ and the preceding lemma, it sufices that the <sub>expected</sub> increment $( a _ { i } - 1 )$ remain positively associated with the local alignment score along the path. The MIG assumption (12) guarantees exactly this: conditioned on $( X ^ { ( k ) } , u )$ , the coordinates with larger $r _ { i }$ receive larger expected weights at every ??, so $\begin{array} { r } { \mathbb { E } \big [ \frac { d } { d t } f \big ( a ( t ) \big ) \big | X ^ { ( k ) } , u \big ] \geq 0 } \end{array}$ for all $t \in [ 0 , 1 ]$ . Integrating from $t = 0$ to ?? <sub>=</sub>1 yields

$$
\mathbb {E} \big [ f (a)   \big |   X ^ {(k)}, u \big ]   \geq   f (\mathbf {1})   =   \frac {\langle X ^ {(k)} , u \rangle}{\| X ^ {(k)} \|}.
$$

Finally, taking expectation over $( X ^ { ( k ) } , \hat { X } ^ { ( k - 1 ) } )$ proves

$$
\mathbb {E} \left[ \frac {\langle \hat {X} ^ {(k)} , \hat {X} ^ {(k - 1)} \rangle}{\| \hat {X} ^ {(k)} \| \| \hat {X} ^ {(k - 1)} \|} \right] \geq \mathbb {E} \left[ \frac {\langle X ^ {(k)} , \hat {X} ^ {(k - 1)} \rangle}{\| X ^ {(k)} \| \| \hat {X} ^ {(k - 1)} \|} \right].
$$

<sub>Remarks.</sub> (i) The proof explicitly uses CE’s <sub>element-wise</sub> form (5)–(6); additive/value-replacement assumptions are unnecessary. (ii) The MIG condition is a precise, verifiable suficient condition tailored to element-wise gating, addressing the need to clarify when an element-wise CE improves directional alignment (and avoiding overly strong orthogonal-leakage assumptions).

## A.2 Proof of Proposition A.2

Proposition A.2 (HS Consistency and Substitutability). <sub>Let</sub> $S ^ { ( k ) }$ and $S ^ { ( K ) }$ be similarity matrices from $E q . \ ( 8 ) ,$ and define $p _ { \tau _ { d } } ^ { ( k ) } ( j | i ) =$ $s o f t m a x ( S _ { i , : } ^ { ( k ) } / \tau _ { d } ) _ { j }$ and $q _ { \tau _ { d } } ^ { ( K ) } ( j | i ) = s o f t m a x ( S _ { i , : } ^ { ( K ) } / \tau _ { d } ) _ { j }$ where teacher $q ^ { ( K ) }$ is detached as in $E q .$ (9). Assuming HS aligns coarse-grained distributions with fine-grained teachers, HS bounds semantic drift and enables performance-preserving coarse inference:

$$
\mathbb {E} _ {(x, t), i} \Big [ \mathrm{KL} \Big (q _ {\tau_ {d}} ^ {(K)} (\cdot | i) \| p _ {\tau_ {d}} ^ {(k)} (\cdot | i) \Big) \Big ] \leq \varepsilon \implies \mathbb {E} _ {(x, t), i} \big [ \big | \Phi \Big (p _ {\tau_ {d}} ^ {(k)} (\cdot | i) \Big) - \Phi \Big (q _ {\tau_ {d}} ^ {(K)} (\cdot | i) \Big) \big | \big ] \leq L \sqrt {\varepsilon / 2}\tag{13}
$$

for any <sup>??</sup>-Lipschitz functional Φ w.r.t. total variation distance. The detach operation ensures gradient isolation: $\partial L _ { g u i d e } / \partial z ^ { ( K ) } = 0$

<sub>Proof.</sub> Fix <sub>(</sub>??, ??<sub>)</sub> and anchor index $i ,$ and set $Q : = q _ { \tau _ { d } } ^ { ( K ) } ( \cdot \mid i )$ and $P : = p _ { \tau _ { d } } ^ { ( k ) } ( \cdot \mid i )$ . By Pinsker’s inequality,

$$
\operatorname{TV} (P, Q) \leq \sqrt {\frac {1}{2} \operatorname{KL} (Q \| P)}.
$$

For any functional <sub>Φ</sub> that is ??-Lipschitz w.r.t. total variation,

$$
\left| \Phi (P) - \Phi (Q) \right| \leq L \operatorname{TV} (P, Q) \leq L \sqrt {\frac {1}{2} \operatorname{KL} (Q \| P)}.
$$

Taking expectation over <sub>(</sub>??, ??<sub>)</sub>, ?? and applying Jensen’s inequality (since $\sqrt { \cdot }$ is concave) yields

$$
\mathbb {E} _ {(x, t), i} \big [ \big | \Phi (P) - \Phi (Q) \big | \big ] \leq L \mathbb {E} \left[ \sqrt {\frac {1}{2} \operatorname{KL} (Q \| P)} \right] \leq L \sqrt {\frac {1}{2}} \mathbb {E} [ \operatorname{KL} (Q \| P) ] \leq L \sqrt {\varepsilon / 2},
$$

which proves the stated consistency/substitutability bound.

For the gradient isolation, write the HS guidance loss as

$$
\mathcal {L} _ {\text { guide }} = \mathbb {E} _ {(x, t), i} \left[ \mathrm{KL} \left(q _ {\tau_ {d}} ^ {(K)} (\cdot | i) \| p _ {\tau_ {d}} ^ {(k)} (\cdot | i)\right) \right],
$$

where the teacher $q _ { \tau _ { d } } ^ { ( K ) }$ is detached as in Eq. (8). Hence $q _ { \tau _ { d } } ^ { ( K ) }$ is treated as a constant and

$$
\frac {\partial \mathcal {L} _ {\mathrm{guide}}}{\partial z ^ {(K)}} = 0.
$$

Equivalently, gradients flow only to the coarse head via the similarity logits $S ^ { ( k ) }$ from Eq. (7): if $P = \operatorname { s o f t m a x } ( S _ { i , : } ^ { ( k ) } / \tau _ { d } )$ , then

$$
\frac {\partial \mathcal {L} _ {\mathrm{guide}}}{\partial S _ {i , :} ^ {(k)}} = \frac {1}{\tau_ {d}} (P - Q),
$$

which is the standard soft-target distillation gradient scaled by $1 / \tau _ { d }$

## B Adaptive Retrieval Strategy

We implement a multi-stage cascaded pruning protocol to optimize the trade-of between retrieval precision and computational overhead. The inference pipeline operates progressively across granularity levels $k = 1$ to 3, dynamically reducing the gallery search space $\Omega _ { k }$ based on prediction confidence. Initially, all ?? gallery samples are evaluated at the coarse level, denoted as $| \Omega _ { 1 } | = N . { \mathrm { A t } }$ each stage $k ,$ we compute similarity scores for the current candidates and evaluate the retrieval certainty using the Relative Score Gap (RSG), defined as $\gamma _ { k } = \big ( s _ { ( 1 ) } - s _ { ( 1 + m ) } \big ) \big / s _ { ( 1 ) }$ , where $s _ { ( r ) }$ denotes the similarity score of the ??-th ranked candidate and ??<sub>=</sub>10 controls the rank gap for measuring confidence spread. If $\dot { \gamma } _ { k }$ exceeds a calibrated threshold $\tau _ { k } .$ , the process terminates early, outputting the current top-1 prediction. Conversely, $\mathrm { i f } \gamma _ { k } \le \tau _ { k }$ and $k < 3$ , we retain the top-scoring half of $\Omega _ { k }$ to construct $\Omega _ { k + 1 }$ , yielding $\lvert \Omega _ { 2 } \rvert = N / 2$ and $ { \left| \Omega _ { 3 } \right| } { = } N / 4$ progressively.

To ensure generalization, the decision thresholds $\{ \tau _ { 1 } , \tau _ { 2 } \}$ are calibrated via grid search on a held-out validation set. We select optimal threshold pairs that minimize average FLOPs subject to varying constraints on performance degradation. By adjusting this tolerance margin, we derive three representative operating points: <sub>Aggressive</sub> (lower thresholds), <sub>Balanced</sub>, and <sub>Conservative</sub> (higher thresholds), corresponding to relaxed, moderate, and strict accuracy constraints, respectively. Specific configurations are detailed in Table 8. The results demonstrate that the <sub>Balanced</sub> setting with $\tau _ { 1 } { = } 0 . 5 0$ and ??<sub>2=</sub>0.65 achieves 91.3% R@1, retaining 97.3% of the fine-grained performance (93.8% from static Level 3), while reducing computational cost to 41.3%.

Table 8: Adaptive retrieval on Flickr30K image-to-text task. Exit Dist. format: Coarse(%)/Medium(%)/Fine(%). Thresholds $( \tau _ { 1 } , \tau _ { 2 } )$ denote RSG cutofs at Levels 1 and 2 with margin $m = 1 0 .$

<table><tr><td>Method</td><td>R@1</td><td>R@5</td><td>R@10</td><td>Rel. Cost</td><td>Exit Dist.</td><td>Thresholds</td></tr><tr><td>Static Level 1</td><td>82.8</td><td>92.4</td><td>93.5</td><td>14.2%</td><td>100/0/0</td><td>-</td></tr><tr><td>Static Level 3</td><td>93.8</td><td>99.4</td><td>99.6</td><td>100.0%</td><td>0/0/100</td><td>-</td></tr><tr><td>Adaptive Aggressive</td><td>87.3</td><td>95.8</td><td>97.9</td><td>24.7%</td><td>76/18/6</td><td>(0.30, 0.45)</td></tr><tr><td>Adaptive Balanced</td><td>91.3</td><td>98.2</td><td>99.2</td><td>41.3%</td><td>52/28/20</td><td>(0.50, 0.65)</td></tr><tr><td>Adaptive Conservative</td><td>93.1</td><td>98.9</td><td>99.4</td><td>67.1%</td><td>18/30/52</td><td>(0.75, 0.85)</td></tr></table>

## C Additional Experiments

## C.1 Individual Dataset Few-shot Classification

Results in Figure 8 confirm UPrompt’s efectiveness across diverse domains. On fine-grained tasks like StanfordCars and FGVCAircraft, our method outperforms CoOp and PSRC by capturing both specific details and global patterns through multi-granularity representations. For DTD texture classification, UPrompt surpasses MMA and ProVP-Ref via bidirectional connection that enables fine-grained modeling guided by coarse context. On SUN397 and EuroSAT, hierarchical supervision prevents semantic drift while maintaining competitive performance with GalLoP. Consistent improvements across shot configurations validate that our U-shaped architecture addresses granularity trade-ofs in single-scale methods.

![](images/49d140750356e8b6a4dbad9d0709ca35d7a16f3dc29834c2f49fb68b7791044b.jpg)

(%)  
![](images/445b592efbc39b027dcc744b5a02f0d86edc3027f55fdfad07ebf4c483f527da.jpg)

![](images/60a35c0c2b66dff73553a6f0128d87d656be265e787f5d1102e2b6780245eef8.jpg)

(%)  
![](images/c67155e1a52c7e9fdde33aa74fb1b4e36b9ba93a1820ce5eca0337a072896837.jpg)

(%)  
![](images/f777d57e4dcc1288c65a08c928a74381e96b2be8b92d4f563e735f3d01881e69.jpg)

![](images/5266c4719a5cd393d82e3ee24079052d6c2b305546c079c6287e7473c7235476.jpg)

![](images/58aa876b37a934c6c2f7c34ec32874d89afa583cc5b4d67b4bb54f0bb8531660.jpg)

![](images/d7f3ed8c0d227cfc12d412532472be311f8959eb07925f96e9e0fec91484880c.jpg)

![](images/1658bd975a4435058428806bdb60b727518197d16179bb7ebc67259e8a6e6e76.jpg)

![](images/e253405dd0d5a4e72397fe688c95d7c4087593b457481baf64acd5d4b0dd4855.jpg)  
Figure 8: Few-shot classification results on individual datasets. Detailed performance breakdown across 10 evaluation datasets with 1, 2, 4, 8, and 16 shots per class.

## C.2 Cross-dataset evaluation

As presented in Table 9, our UPrompt framework demonstrates robust domain generalization capabilities when transferred from ImageNet to 10 downstream datasets. It achieves the highest average accuracy of 67.55%, underscoring the efectiveness of its architecture in adapting to new data distributions. We compare UPrompt with other methods that also leverage multi-level or hierarchical representations. For instance, TAP [10] constructs an explicit "concept-attribute-description" hierarchy, while HiCroPL [62] establishes knowledge flow across network layers. Although these approaches are competitive, particularly HiCroPL on datasets like FGVC and SUN, our UPrompt’s U-Net-inspired

Table 9: Cross-dataset evaluation. Domain transfer against recent prompt learning methods. Trained on ImageNet, evaluated on 10 datasets. Best results highlighted in first , second .

<table><tr><td rowspan="2"></td><td>Source</td><td colspan="11">Target</td></tr><tr><td>ImgNet</td><td>Cal101</td><td>Pets</td><td>Cars</td><td>Flowers</td><td>Food</td><td>FGVC</td><td>SUN</td><td>DTD</td><td>SAT</td><td>UCF</td><td>Avg</td></tr><tr><td> $CoOp_{(IJCV'22)}$ </td><td>71.51</td><td>93.70</td><td>89.14</td><td>64.51</td><td>68.71</td><td>85.30</td><td>18.47</td><td>64.15</td><td>41.92</td><td>46.39</td><td>66.55</td><td>63.88</td></tr><tr><td> $PSRC_{(ICCV'23)}$ </td><td>71.27</td><td>93.60</td><td>90.25</td><td>65.70</td><td>70.25</td><td>86.15</td><td>23.90</td><td>67.10</td><td>46.87</td><td>45.50</td><td>68.75</td><td>65.81</td></tr><tr><td> $DeKgTCP_{(ICLR'25)}$ </td><td>72.33</td><td>94.73</td><td>90.02</td><td>65.49</td><td>72.39</td><td>86.59</td><td>25.05</td><td>67.19</td><td>44.47</td><td>51.37</td><td>68.78</td><td>66.61</td></tr><tr><td> $TAP_{(ICLR'25)}$ </td><td>72.30</td><td>94.30</td><td>90.70</td><td>65.60</td><td>70.93</td><td>86.10</td><td>24.57</td><td>68.30</td><td>50.20</td><td>46.00</td><td>68.90</td><td>66.56</td></tr><tr><td> $TAC_{(CVPR'25)}$ </td><td>72.77</td><td>94.53</td><td>90.67</td><td>65.30</td><td>72.20</td><td>85.83</td><td>23.53</td><td>67.63</td><td>47.57</td><td>48.07</td><td>70.00</td><td>66.53</td></tr><tr><td> $HiCroPL_{(ICCV'25)}$ </td><td>70.84</td><td>94.48</td><td>90.13</td><td>65.68</td><td>72.03</td><td>86.46</td><td>26.58</td><td>68.78</td><td>53.19</td><td>49.19</td><td>70.31</td><td>67.38</td></tr><tr><td> $CoCoA-Mix_{(ICML'25)}$ </td><td>70.85</td><td>93.46</td><td>89.07</td><td>65.59</td><td>68.72</td><td>85.78</td><td>24.10</td><td>63.61</td><td>46.41</td><td>48.18</td><td>67.78</td><td>65.27</td></tr><tr><td> $UPrompt_{(Ours)}$ </td><td>72.25</td><td>94.75</td><td>90.97</td><td>66.09</td><td>72.41</td><td>86.60</td><td>26.44</td><td>68.64</td><td>47.51</td><td>52.08</td><td>70.05</td><td>67.55</td></tr></table>

bidirectional multi-granularity learning leads to more consistent and superior performance across a wider range of tasks, securing the top results on 6 of the 10 target datasets. Furthermore, UPrompt outperforms other recent domain generalization methods like TAC [15] and DeKgTCP [28], validating that our explicit modeling of coarse-to-fine semantic levels is highly efective for robust cross-dataset transfer.

Table 10: Error bar analysis on base-to-novel generalization. Results report mean accuracy and standard deviation across three independent runs.

<table><tr><td rowspan="2">Method</td><td colspan="3">Average</td><td colspan="3">ImageNet</td><td colspan="3">Caltech101</td></tr><tr><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CoOp</td><td>82.69</td><td>63.22</td><td>71.66</td><td>76.47</td><td>67.88</td><td>71.92</td><td>96.00</td><td>89.81</td><td>93.73</td></tr><tr><td>UPrompt</td><td>86.35</td><td>78.29</td><td>82.12</td><td> $78.65 \pm 0.09$ </td><td> $71.24 \pm 0.11$ </td><td>74.76</td><td> $98.78 \pm 0.04$ </td><td> $95.84 \pm 0.13$ </td><td>97.29</td></tr><tr><td rowspan="2">Method</td><td colspan="3">OxfordPets</td><td colspan="3">StanfordCars</td><td colspan="3">Flowers102</td></tr><tr><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CoOp</td><td>93.67</td><td>95.29</td><td>94.47</td><td>78.12</td><td>60.40</td><td>68.13</td><td>97.60</td><td>59.67</td><td>74.06</td></tr><tr><td>UPrompt</td><td> $96.41 \pm 0.32$ </td><td> $97.92 \pm 0.23$ </td><td>97.16</td><td> $83.58 \pm 0.22$ </td><td> $74.57 \pm 0.18$ </td><td>78.82</td><td> $98.54 \pm 0.72$ </td><td> $78.43 \pm 0.40$ </td><td>87.34</td></tr><tr><td rowspan="2">Method</td><td colspan="3">Food101</td><td colspan="3">FGVCAircraft</td><td colspan="3">SUN397</td></tr><tr><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CoOp</td><td>88.33</td><td>82.26</td><td>85.19</td><td>40.44</td><td>22.30</td><td>28.75</td><td>80.60</td><td>65.89</td><td>72.51</td></tr><tr><td>UPrompt</td><td> $91.20 \pm 0.17$ </td><td> $92.16 \pm 0.22$ </td><td>91.68</td><td> $49.33 \pm 0.30$ </td><td> $39.25 \pm 0.16$ </td><td>43.72</td><td> $83.77 \pm 0.19$ </td><td> $80.05 \pm 0.32$ </td><td>81.87</td></tr><tr><td rowspan="2">Method</td><td colspan="3">DTD</td><td colspan="3">EuroSAT</td><td colspan="3">UCF101</td></tr><tr><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CoOp</td><td>79.44</td><td>41.18</td><td>54.24</td><td>93.19</td><td>54.74</td><td>68.69</td><td>84.69</td><td>56.05</td><td>67.46</td></tr><tr><td>UPrompt</td><td> $85.60 \pm 1.06$ </td><td> $67.23 \pm 1.64$ </td><td>75.31</td><td> $94.82 \pm 2.04$ </td><td> $82.68 \pm 2.37$ </td><td>88.33</td><td> $89.21 \pm 0.62$ </td><td> $81.83 \pm 0.44$ </td><td>85.36</td></tr></table>

## C.3 Error bar analysis

We conducted error bar analysis across both cross-dataset and base-to-novel evaluation settings, performing three independent runs to ensure robust statistical evaluation. For base-to-novel generalization (Table 10), UPrompt maintains consistent performance with low standard deviations across most datasets. While slightly higher variance appears on EuroSAT (<sub>±</sub>2.37) and DTD (<sub>±</sub>1.64) due to limited training samples, UPrompt still substantially outperforms CoOp, confirming the reliability of our bidirectional multi-granularity framework. For cross-dataset evaluation (Table 11), UPrompt demonstrates remarkable stability with low variance on Food101 (<sub>±</sub>0.03), StanfordCars (<sub>±</sub>0.08), and DTD (<sub>±</sub>0.08), validating robust cross-domain generalization.

Table 11: Error bar analysis on cross-dataset evaluation. Trained on ImageNet, evaluated on 10 datasets. Results report mean accuracy and standard deviation across three independent runs.

<table><tr><td>Method</td><td>Caltech101</td><td>OxfordPets</td><td>StanfordCars</td><td>Flowers102</td><td>Food101</td></tr><tr><td>CoOp</td><td>93.70</td><td>89.14</td><td>64.51</td><td>68.71</td><td>85.30</td></tr><tr><td>UPrompt</td><td> $94.51 \pm 0.11$ </td><td> $90.75 \pm 0.17$ </td><td> $65.98 \pm 0.08$ </td><td> $72.19 \pm 0.23$ </td><td> $86.57 \pm 0.03$ </td></tr><tr><td>Method</td><td>FGVCAircraft</td><td>SUN397</td><td>DTD</td><td>EuroSAT</td><td>UCF101</td></tr><tr><td>CoOp</td><td>18.47</td><td>64.15</td><td>41.92</td><td>46.39</td><td>66.55</td></tr><tr><td>UPrompt</td><td> $26.14 \pm 0.13$ </td><td> $68.42 \pm 0.21$ </td><td> $47.35 \pm 0.08$ </td><td> $53.24 \pm 1.18$ </td><td> $69.93 \pm 0.20$ </td></tr></table>

## D Other ablation studies

## D.1 Reliability of fine-grained supervision.

To validate that fine-grained representations provide reliable supervision signals for coarser levels, we compared single fine-layer supervision with mixed supervision combining fine and medium-granularity teachers on Flickr30K and MSCOCO. Results in Table 12 demonstrate that fine-layer supervision alone achieves comparable or superior performance across both datasets (Flickr30K I2T R@1: 93.8% vs 93.7%; MSCOCO I2T R@1: 70.1% vs 69.8%), confirming its suficiency and stability as the primary teacher signal without requiring multi-layer aggregation.

Table 12: Fine-Layer Supervision Reliability. Comparison of fine-layer versus fine + medium-layer supervision on Flickr30K and MSCOCO. rSum denotes sum of all R@1, R@5, R@10 scores.

<table><tr><td rowspan="3">Teacher Strategy</td><td colspan="6">Flickr30K</td><td colspan="6">MSCOCO</td></tr><tr><td colspan="3">Image-to-Text</td><td colspan="3">Text-to-Image</td><td rowspan="2">rSum</td><td colspan="3">Image-to-Text</td><td colspan="2">Text-to-Image</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td></tr><tr><td>Fine + Medium (soft mixing)</td><td>93.7</td><td>99.2</td><td>99.3</td><td>83.9</td><td>96.3</td><td>98.2</td><td>570.6</td><td>69.8</td><td>89.4</td><td>94.5</td><td>52.5</td><td>78.8</td></tr><tr><td>Fine only (Ours)</td><td>93.8</td><td>99.4</td><td>99.6</td><td>83.6</td><td>96.3</td><td>98.4</td><td>571.1</td><td>70.1</td><td>89.8</td><td>84.8</td><td>52.6</td><td>79.1</td></tr></table>

## D.2 Granularity Interval Strategy

Beyond the number of granularity levels, we examine diferent downsampling interval strategies between adjacent granularities. Table 13 compares two strategies on Flickr30K I2T retrieval: our adopted approach following U-Net’s principle [41] of approximate halving (14 <sub>×</sub> $1 4 \to 7 \times 7 \to 4 \times 4 )$ versus a sparser interval $( 1 4 \times 1 4 \to 5 \times 5 \to 2 \times 2 ) .$ Both strategies achieve comparable performance at the fine-grained level (93.6% vs 93.8%), but the sparse interval shows degradation at coarser levels, particularly at the coarse granularity (78.6% vs 82.8%).

Table 13: Granularity interval strategy comparison on Flickr30K I2T retrieval using R@1.

<table><tr><td>Strategy</td><td>Coarse</td><td>Medium</td><td>Fine</td></tr><tr><td>14 × 14 → 5 × 5 → 2 × 2</td><td>78.6</td><td>88.3</td><td>93.6</td></tr><tr><td>14 × 14 → 7 × 7 → 4 × 4</td><td>82.8</td><td>89.4</td><td>93.8</td></tr></table>

This performance gap is attributed to the reduced expressive capacity at coarser levels under sparser downsampling, demonstrating the efectiveness of our graduated interval design.

Table 14: Cross-modal retrieval results on ViT-B/32 backbone. rSum is the sum of all R@1, R@5, and R@10 scores. Best results highlighted in first , second .

<table><tr><td rowspan="3">Methods</td><td colspan="6">Flickr30K</td><td rowspan="3">rSum</td><td colspan="6">MSCOCO</td></tr><tr><td colspan="3">Image-to-Text</td><td colspan="3">Text-to-Image</td><td colspan="3">Image-to-Text</td><td colspan="3">Text-to-Image</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td> $MAMET_{(TCSVT'25)}$ </td><td>87.7</td><td>97.5</td><td>99.6</td><td>73.5</td><td>93.0</td><td>96.5</td><td>547.8</td><td>61.5</td><td>86.2</td><td>92.5</td><td>48.6</td><td>76.3</td><td>85.3</td></tr><tr><td> $APSE-IPIK_{(AAAI'25)}$ </td><td>86.3</td><td>97.6</td><td>99.4</td><td>72.0</td><td>92.5</td><td>95.1</td><td>542.9</td><td>59.1</td><td>85.7</td><td>94.6</td><td>45.1</td><td>72.8</td><td>82.5</td></tr><tr><td> $UPrompt_{(Ours)}$ </td><td>88.9</td><td>97.6</td><td>99.7</td><td>74.0</td><td>93.2</td><td>96.8</td><td>550.2</td><td>62.4</td><td>86.8</td><td>93.8</td><td>49.7</td><td>76.9</td><td>85.7</td></tr></table>

Table 15: Image-to-text retrieval on ViT-L/14 backbone. Results highlighted in first , second .

<table><tr><td rowspan="2">Methods</td><td colspan="3">MSCOCO</td><td colspan="3">Flickr30K</td><td rowspan="2">rSum</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>Unicoder-VL</td><td>62.3</td><td>87.1</td><td>92.8</td><td>86.2</td><td>96.3</td><td>99.0</td><td>523.7</td></tr><tr><td>Oscar</td><td>73.5</td><td>92.2</td><td>96.0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ERNIE-ViL</td><td>-</td><td>-</td><td>-</td><td>88.7</td><td>98.0</td><td>99.2</td><td>-</td></tr><tr><td>AAPE</td><td>76.7</td><td>94.5</td><td>97.4</td><td>94.9</td><td>99.3</td><td>99.7</td><td>561.8</td></tr><tr><td>UPrompt</td><td>77.8</td><td>94.9</td><td>97.4</td><td>95.1</td><td>99.7</td><td>99.8</td><td>564.7</td></tr></table>

## D.3 Cross-modal retrieval with diferent backbones

To validate UPrompt’s generalizability across diferent architectures, we evaluate on ViT-B/32 and ViT-L/14 backbones (Tables 14 and 15). On ViT-B/32, UPrompt achieves 550.2 and 455.3 rSum on Flickr30K and MSCOCO respectively, outperforming MAMET [48] (547.8, 450.4) and APSE-IP1K [21] (542.9, 439.8). The consistent improvements across diferent model scales demonstrate the robustness of our multi-granularity framework. On the larger ViT-L/14 backbone for image-to-text retrieval, UPrompt achieves 564.7 rSum, outperforming AAPE [20] (561.8), Unicoder-VL [25] (523.7), Oscar [27], and ERNIE-ViL [58], with particularly strong performance on Flickr30K (95.1% R@1) and MSCOCO (77.8% R@1). These results confirm that our bidirectional connection mechanisms efectively leverage increased mode capacity, with hierarchical supervision preventing semantic drift across granularities regardless of backbone architecture.

Table 16: Base-to-novel generalization on alternative VLM architectures. UPrompt consistently outperforms baselines across SigLIP and EVA-CLIP.

<table><tr><td>Methods</td><td>Backbone</td><td>Cars</td><td>Flowers</td><td>FGVC</td></tr><tr><td>CoOp</td><td>EVA-CLIP</td><td>71.33</td><td>77.39</td><td>34.72</td></tr><tr><td>UPrompt</td><td>EVA-CLIP</td><td>79.42</td><td>87.36</td><td>43.86</td></tr><tr><td>CoOp</td><td>SigLIP</td><td>92.33</td><td>89.42</td><td>38.27</td></tr><tr><td>UPrompt</td><td>SigLIP</td><td>94.67</td><td>93.26</td><td>46.34</td></tr></table>

Table 17: Rule-based hierarchies on CUB-200 and AWA2 in generalized zero-shot learning.

<table><tr><td>Dataset</td><td>Method</td><td>Level</td><td>Base</td><td>New</td><td>HM</td></tr><tr><td rowspan="4">AWA2</td><td>CoOp</td><td>Single</td><td>95.32</td><td>72.68</td><td>82.47</td></tr><tr><td rowspan="3">Uprompt</td><td>Level 1</td><td>93.24</td><td>70.27</td><td>80.14</td></tr><tr><td>Level 2</td><td>95.81</td><td>73.13</td><td>82.95</td></tr><tr><td>Level 3</td><td>96.70</td><td>74.62</td><td>84.24</td></tr><tr><td rowspan="4">CUB-200</td><td>CoOp</td><td>Single</td><td>63.78</td><td>49.23</td><td>55.57</td></tr><tr><td rowspan="3">UPrompt</td><td>Level 1</td><td>61.36</td><td>46.84</td><td>53.13</td></tr><tr><td>Level 2</td><td>64.65</td><td>50.42</td><td>56.65</td></tr><tr><td>Level 3</td><td>66.12</td><td>51.26</td><td>57.75</td></tr></table>

## D.4 Generalization across VLM architecture

To validate UPrompt’s generalizability across diverse vision-language models, we evaluate on alternative architectures including SigLIP and EVA-CLIP with ViT-B/16 backbones. Table 16 presents base-to-novel generalization results on three representative datasets. UPrompt achieves consistent improvements over CoOp across both architectures: on EVA-CLIP, gains of +8.09%, +9.97%, and +9.14% HM on StanfordCars, Flowers102, and FGVCAircraft respectively; on SigLIP, gains of +2.34%, +3.84%, and +8.07% HM respectively. Notably, the improvements are particularly pronounced on the fine-grained FGVCAircraft dataset (+9.14% on EVA-CLIP, +8.07% on SigLIP), demonstrating that our multi-granularity framework efectively enhances fine-grained recognition across diferent VLM architectures. These results confirm that our bidirectional connection mechanisms operate efectively at the prompt level, independent of the underlying vision-language model architecture.

## E Rule-Based Text Construction.

To demonstrate that our framework’s efectiveness does not rely on LLM-generated priors, we conduct experiments using rule-based text construction on CUB-200 [45] and AWA2 [50] datasets, which provide dense attribute annotations. We construct three granularity levels: Level 1 (coarse) uses "a photo of a class", Level 2 (medium) adds the highest-certainty attribute (e.g., "a photo of a class with black bill"), and Level 3 (fine) incorporates two highest-certainty attributes (e.g., "a photo of a class with black bill and white breast"), where attributes are ranked by certainty scores from dataset metadata. Visually, the three levels correspond to 4<sub>×</sub>4 pooled tokens, 7<sub>×</sub>7 pooled tokens, and the original 14<sub>×</sub>14 tokens respectively. We compare against CoOp, which uses "a photo of a class" with 14<sub>×</sub>14 visual tokens as the single-granularity baseline. Table 17 presents results on generalized zero-shot learning (GZSL). The progressive improvements from coarse to fine levels validate that our bidirectional connection mechanism efectively integrates multi-scale representations, even without external knowledge sources like LLM generated content.

![](images/192553f470069348ca668d392ca4db631e66eb4a4f7d6bd6c5ea3b9ad9472132.jpg)  
Figure 9: Cross-modal retrieval results on Flickr30K dataset. “Original Single Granularity” refers to baseline model using fixed single-scale visual and textual representations. ○× indicate retrieval failures, ○<sup>✓</sup> indicate successful retrievals.

## F Visualizations

## F.1 Granularity-Specific Retrieval Efectiveness.

The retrieval results in Fig. 9 validate our bidirectional connection mechanisms across granularity levels. Fine-grained representations consistently excel in both Text-to-Image and Image-to-Text tasks, resolving challenging cases requiring precise semantic understanding,

Input Image  
![](images/d3da34c2ec5f9d648bb958617700018d84d2082ce655414715b331c0fed17b28.jpg)

Two men sit at a diner, one reading a newspaper, the other smoking a cigarette.

-Grained

Input Image  
![](images/90a9c2b04fe2ebf9a75b958c3672f7ef2ed9751ead8c93d0057b2ca74860ea66.jpg)

A woman at a fish market is looking at the fish that are frozen.

Fine
-Grained

w/o cascaded enhancement  
![](images/9c327f578cf954115620dbaccd3b6279f14a7c0579ed794789b6409c6ef91284.jpg)  
w/ cascaded enhancement

## w/o cascaded enhancement

![](images/534e2fe33b52b39da162f41e83e80a10036bc677f21b93296c1d220647fa898f.jpg)  
Input Image

A man sits at a restaurant table reading a newspaper, with another man seated at the table next to him, one smoking a cigarette, as people wait in the background.

![](images/2fee27944d7b9feb46298ff547323df8514fb63eb0c06405ca786316a0358b9e.jpg)

![](images/d59acdb29cbe4c7a849816aa513fb30341ad30a5ed3bd17b33a0c31923ea2218.jpg)

## w/ cascaded enhancement

![](images/3c825e64f3508a64ca6fd9ddbcbbd8cdcefc258610dbe4c778793725380ffb1c.jpg)

Two guitarist are preparing for a show on stage in front of a waiting crowd.

A woman wearing a short-sleeved striped shirt examines fish on ice at a bustling street market, looking at the frozen fresh catch while shopping for her purchase.

## w/o cascaded enhancement

![](images/5c4e97276744b0dee4786e9b90ffee997890c3c035f29b1f423a3a90f9a7120f.jpg)

## w/ cascaded enhancement

![](images/1cc3bebf3766247428dd78b917e57040db179422460fbc611a7b8a1286e2e3ad.jpg)

Fine
-Grained

Two men, one sitting, one standing, with one wearing orange glasses, play guitars on stage in front of a large, waiting crowd of onlookers, preparing for a show.

Figure 10: Visual validation of Coarse-to-Fine Cascaded Enhancement (CE). Our CE module addresses the context deficiency of fine-grained embeddings by injecting global contextual guidance. The comparison demonstrates that without CE (middle column), fine-grained attention struggles to model local information relationships. With CE (right column), our model achieves precise, contextually-aware alignment for complex fine-grained descriptions.

such as distinguishing “carrying a toy in its mouth” from general dog activities. This capability stems from cascaded enhancement providing global contextual guidance, preventing attention from focusing solely on isolated details while capturing comprehensive information. Medium-grained representations outperform the single-granularity baseline while using fewer visual tokens. Coarse-grained representations achieve comparable performance despite using substantially fewer visual and textual tokens, enabled by hierarchical supervision that prevents semantic drift and preserves alignment quality with reduced representational capacity. These results confirm flexible performance-eficiency trade-ofs while keeping semantic consistency across hierarchical structure.

![](images/2970dd1a9365f144d4fd013fc748a8e925232ad091370b801b8d2840279be3f7.jpg)  
Figure 11: Visual validation of Fine-to-Coarse Hierarchical Supervision (HS). HS prevents semantic drift in coarse-grained representations. Without HS (middle), attention maps show common failures: missing key objects (second bicycle), poor component grounding (pole), or drift to irrelevant backgrounds. With HS (right), fine-level supervision guides coarse models to maintain semantic consistency, producing well-localized attention that accurately reflects textual descriptions.

## F.2 Visual Analysis of Cascaded Enhancement

Fig. 10 provides an analysis to visually validate the eficacy of our Coarse-to-Fine Cascaded Enhancement (CE) module. The comparison demonstrates that without CE (middle column), fine-grained attention struggles with context deficiency, failing to accurately ground complex descriptions involving multiple entities or specific attributes. For instance, it cannot disambiguate the "man reading a newspaper" from the one "smoking a cigarette," nor can it precisely locate the "striped shirt" or the "orange glasses." Conversely, by injecting global contextual guidance, our CE module (right column) resolves these ambiguities, enabling precise, contextually-aware alignment. The resulting attention maps successfully disentangle parallel actions and ground fine-grained attributes to their corresponding image regions. This visual evidence substantiates our claim that CE is crucial for addressing the limitations of isolated fine-grained processing, enabling robust alignment for complex, multi-faceted image-text pairs

## F.3 Visual Analysis of Hierarchical Supervision

Our Fine-to-Coarse Hierarchical Supervision (HS) plays a crucial role in mitigating semantic drift at coarser granularity levels, as visually validated in Fig. 11. Without HS, coarse-grained models trained on simplified text-image pairs often produce flawed alignments; the attention may drift to background noise (e.g., the water wake instead of the girl), omit less salient objects mentioned in the text (e.g., the second bicycle), or fail to ground all relevant components (e.g., ignoring the pole). These inconsistencies arise because coarser levels are optimized in isolation with ambiguous supervision. Our HS mechanism addresses this by using the finest-level alignment as a teacher distribution to regularize the learning process across the hierarchy. As demonstrated in the right column, this forces the coarse-grained representations to maintain semantic consistency, resulting in well-localized attention and complete object coverage that correctly reflects the underlying semantics.