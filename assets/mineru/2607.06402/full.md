# What Images Cannot Say: Language-Guided Olfactory Representation Learning

Eleftherios Tsonis, Xi Wang, and Vicky Kalogeiton

LIX, École Polytechnique, IP Paris, CNRS

{firstname.lastname}@polytechnique.edu

https://www.lix.polytechnique.fr/vista/projects/2026\_scent\_tsonis/

Abstract. Images tell us what a scene looks like, but rarely what it would feel like to be there. While recent datasets pair visual scenes with electronic-nose measurements, aligning smell signals with images remains challenging because many olfactory cues arise from contextual environmental factors that are not directly visible in pixels. We introduce SCENT, a multimodal framework that uses language guidance as a semantic bridge between vision and olfaction. Our approach leverages Vision-Language Models (VLMs) to generate scene descriptors capturing objects, environmental context, and plausible ambient smell cues suggested by the visual scene. These descriptors provide semantic guidance for learning olfactory representations. We train a smell encoder that maps electronic-nose signals into a shared embedding space aligned with both visual and textual representations, and introduce a languageguided latent decomposition that separates object-specific odors from contextual environmental contributions. Experiments on the New York Smells dataset demonstrate that SCENT significantly improves crossmodal retrieval compared to vision-only baselines, achieving state-of-theart performance on smell-to-image and smell-to-text retrieval tasks. In addition, our framework produces interpretable olfactory representations that enable the disentanglement of complex smell mixtures. Our results reveal the importance of contextual semantic information for grounding olfactory perception in multimodal learning and pave the way for future research in this area.

Keywords: Cross-modal Retrieval · Olfactory Representation Learning · Vision-Language Models

## 1 Introduction

“A picture is worth a thousand words” (att. Frederick R. Barnard) ... but it rarely tells us what it feels like to be there..

In real environments, perception extends beyond vision. A photograph of a busy street may evoke the smell of vehicle exhaust; a metro entrance may imply the metallic scent of ventilation air. Although such environmental cues are rarely visible directly, humans routinely infer them from visual context. Despite years of progress in computer vision, AI systems today primarily capture what a scene looks like, while remaining largely unaware of what it might feel like to be there.

![](images/211c8bfb46c4c689aa23c8e996c36947aa347b383405195d2bb10240da423dde.jpg)  
Fig. 1: Not everything we can smell is visible. Given only View 1 (in-sample) during training, the true olfactory context may lie outside the field of view, in View 2 (out-of-sample). A VLM can bridge this gap by inferring plausible smells from semantic context alone. We use these language-derived signals as supervision to learn richer smell representations that go beyond what is directly seen.

Recent advances in Vision-Language Models (VLMs) suggest that this longstanding idea may finally be realized: modern systems can generate rich descriptions of visual scenes [34, 35, 67], answer questions about images [19, 64, 71], and reason about complex environments through language [5, 30]. Their extensive world knowledge opens the possibility of inferring what exists in a scene, beyond what is visible.

Among sensory modalities, olfaction is particularly important. Smell conveys rich semantic information that is dificult to infer from other modalities [22, 54], including food quality and safety [1, 55], environmental conditions and airborne hazards, such as pollution [7] and toxic substances [8, 63], as well as social cues related to health and emotional state [31, 40].

Advances in electronic noses (e-noses) [63] enable the capture of high dimensional sensor measurements [18,20,46] that characterize the chemical composition of air. Mapping these raw signals to semantic representations is, however, fundamentally dificult. First, the image paired with each recording captures only a partial view of the scene: smells difuse freely and may originate from sources entirely outside the camera’s field of view, so visual supervision alone cannot fully account for what was measured (Figure 1). Second, a smell recording reflects a mixture of contributions from multiple odor sources in the scene, making it hard to disentangle individual semantic factors.

VLMs trained on large-scale data develop world knowledge that extends well beyond visual appearence, learning to reason about physical, acoustic, and semantic properties of scenes [27, 60, 69]. This extends to olfaction, as Large Language Models (LLM) can classify odors, predict smell descriptors, and identify sources from natural language descriptions, demonstrating that smell associations are encoded through semantic context [39]. Inherited from LLMs, modern VLMs also hold natural knowledge priors which are useful for our problem: given a scene image, a VLM can generate plausible olfactory descriptions that go beyond what the camera captures, directly compensating for the partial observability of visual supervision.

Building on this insight, we propose SCENT: Semantic Context-aware e-Nose Transformer, a framework that integrates e-nose measurements with visual and linguistic representations. Our approach first uses a VLM to generate structured textual descriptors of a scene, capturing objects, environmental context, and plausible ambient smell cues. These descriptions serve as a semantic bridge between vision and smell: they encode what a scene implies about its olfactory environment, extending supervision beyond what the camera alone can observe. The descriptors are encoded and used as semantic anchors for representation learning. We train a smell encoder that maps e-nose signals into a shared embedding space aligned with both visual and textual representations, enabling multimodal representation of olfactory environments.

To further structure the learned representation, we introduce a latent decomposition that separates object-specific odor signals from contextual environmental contributions. This decomposition is guided by the text descriptions produced by the VLM and encourages the model to disentangle the components of complex olfactory mixtures.

We evaluate our approach on the New York Smells (NYS) dataset [46] and demonstrate that integrating language guidance significantly improves multimodal alignment compared to vision-only baselines. Our method achieves stateof-the-art performance on cross-modal retrieval tasks, including smell-to-image and smell-to-text retrieval, while also producing more interpretable olfactory representations. In summary, our contributions are:

– We introduce the idea of using language guidance as a semantic bridge between visual scenes and olfactory signals, enabling the interpretation of smell measurements through contextual world knowledge.

– We present SCENT, a multimodal framework for learning olfactory representations that aligns e-nose, visual, and textual embeddings.

– We propose a language-guided decomposition of olfactory representations that separates object-related odors from environmental contributions.

## 2 Related Work

Cross-modal and multimodal supervision. Using one modality to supervise another has a long history, spanning early self-supervised approaches [12] and deep multimodal models [43], to more recent methods that exploit naturally cooccurring sensor data for audio-visual [2,45] and visual-tactile [14,68,70] learning. Models such as CLIP [47] and ALIGN [28] learn joint image-text representations, while CLAP [17] extends this paradigm to audio and text. Similar approaches have since been applied to video–text [38] and point cloud-text [72] representation learning. Works such as CLIP4VLA [50], VALOR [36], VAST [9], mPLUG-2 [65], UMT [37], InternVideo2 [61], and ConFu [32] align three modalities at once. For higher order settings, existing methods often designate one modality as an anchor and align the remaining modalities through pairwise contrastive objectives [41]. For example, ImageBind [24] uses images as the central bridge, and LanguageBind [75] uses text. OneLLM [26] aligns all modalities to a frozen language model, and UniAlign [74] encodes modalities in a unified mixture-ofexperts architecture. More recently, alternative formulations have been explored, including geometric approaches such as GRAM [11] and TRIANGLE [10], and information-theoretic objectives such as Symile [52] and CoMM [16].

Language as a semantic bridge across modalities Machine generated language provides a scalable alternative to manual annotation, enabling semantic supervision of modalities that are otherwise expensive to label. Early neural image captioning [58] established that vision and language can be jointly modelled to produce descriptive text from visual inputs. Later work leverages this ability as a supervisory signal across diverse modalities. ULIP-2 [66] generates holistic language descriptions from rendered views of 3D shapes, enabling scalable trimodal pre-training over point clouds, images, and text without any human annotation. LAVILA [73] similarly uses LLMs as narrators to densely annotate video, yielding supervision for contrastive video–text learning. Moving beyond vision, WineSensed/FEAST [4] demonstrates that text and images can be aligned to human sensory perception of taste, establishing that sensory modalities can be grounded in shared multimodal representations. Building on these precedents, we use VLM-generated descriptors to inject world knowledge and contextual cues into olfactory representation learning.

Machine olfaction and olfactory representations. Machine olfaction has historically focused on constrained settings using specialized or laboratory-grade sensing [13, 20, 42, 49, 57, 62], enabling applications such as scent design [51], disease detection [21, 23], and security [56]. At the molecular level, prior work has used psychophysical data and graph-based models to predict perceptual attributes [15, 29] and to define a principal odor map (POM) [33], while mixture and similarity studies remain limited [48, 53]. Recent findings from neuroscience emphasize the importance of studying olfaction under natural concentration ranges [59]. Concurrently, recent work examines how well large language models can reason about smell [39].

Moving beyond lab-based limitations, the New York Smells (NYS) dataset [46] is a large-scale multimodal benchmark designed for "in-the-wild" smell perception. It comprises 7,000 image-olfactory pairs, collected across diverse urban environments ranging from indoor libraries to outdoor parks. NYS frames olfactory learning as direct cross-modal alignment between visual imagery and e-nose signals. Our approach difers from NYS by using language as a semantic bridge to inject prior world knowledge into smell representation learning, capturing olfactory cues that visual appearence alone cannot express. To our knowledge, no existing work combines olfactory sensor signals with vision and text together.

![](images/21997409b8915b59bb5894332ea0591881258141769e82a33ab2084511edec72.jpg)  
Fig. 2: Textual scene descriptors via VLM inference. Given an image, a pretrained VLM generates object, contextual, and inferred smell descriptors, which serve as language guidance for olfactory representation learning.

## 3 Method

We introduce SCENT: Semantic Context-aware e-Nose Transformer, a framework for learning olfactory representations from e-nose measurements, guided by vision and language. Our key insight is that an olfactory signal $\mathbf { X } \in \dot { \mathbb { R } } ^ { \check { C } \times T }$ (a C-channel e-nose recording over T time steps) reflects odor contributions from the entire scene, yet paired visual observations capture only a subset of these factors (Figure 1). In other words, smell information is only partially observable from visual input. To bridge this gap, we leverage the strong world priors of Vision-Language Models (VLMs) [34, 67], which can infer plausible olfactory cues from visual context.

SCENT therefore combines three stages: (1) VLM-based semantic scene augmentation (Figure 2, Section 3.1); (2) multimodal olfactory representation learning (Figure 3(a), Section 3.2); and (3) latent smell disentanglement that separates object and contextual odor components (Figure 3(b), Section 3.3).

## 3.1 VLM-based Semantic Scene Augmentation

The NYS dataset provides images I and corresponding smell measurements X. To expose the environmental factors that shape olfactory perception, we query a pretrained VLM through a structured prompt, eliciting its prior world knowledge at three semantic levels. Given an image I, the VLM produces a set of language descriptors:

$$
T (I) = \{t _ {\mathrm{obj}}, t _ {\mathrm{ctx}}, t _ {\mathrm{smell}} ^ {(1)}, \ldots , t _ {\mathrm{smell}} ^ {(K)} \} \quad ,
$$

where $t _ { \mathrm { o b j } }$ denotes a textual description of the primary object in the scene, $t _ { \mathrm { c t x } }$ describes the surrounding environmental context, and $\{ t _ { \mathrm { s m e l l } } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ are textual descriptors referring to plausible ambient smell cues suggested by the scene context. These descriptors provide semantic cues that may relate to the measured olfactory signal but are not directly encoded in the image alone. As shown in Figure 2, we explore several prompting schemes and adopt the rightmost one, which explicitly encourages the VLM to infer potential olfactory elements.

These descriptors are combined and prompted into a textual description and encoded using the frozen CLIP [47] text encoder: $z ^ { T } = f _ { T } ( \mathrm { p r o m p t } ( T ) ) \in \mathbb { R } ^ { 5 1 2 }$

![](images/bd388d018d2fc36df3ea89cdf5e98201b091be3d98e51bfbcc422b71b1b152aa.jpg)  
Fig. 3: Overall framework for multimodal olfactory representation learning and disentanglement. (a) Smell Representation Learning: We train a Smell Encoder to align smell fingerprints with CLIP visual $\left( z ^ { I } \right)$ and textual $( z ^ { T } )$ embeddings. The dual projection heads yield $z _ { I } ^ { S }$ and $z _ { T } ^ { S }$ . (b) Latent Smell Disentanglement: Leveraging the pretrained encoder, we disentangle the smell representation into object $( z _ { \mathrm { o b j } } ^ { S } )$ and context $( z _ { \mathrm { c t x } } ^ { S } )$ components. These components are aligned with their respective linguistic descriptors via $\mathcal { L } _ { \mathrm { o b j } }$ and ${ \mathcal L } _ { \mathrm { c t x } }$

The resulting representation $z ^ { T }$ serves as language supervision in the multimodal alignment stage (Section 3.3), helping compensate for information missing from visual observations.

## 3.2 Learning Multi-modal Olfactory Representations

With visual observations I and augmented smell descriptors T (I) available, we learn an olfactory representation aligned with both visual and textual modalities.

Olfactory Encoder. The smell signal X is processed using a Transformer-based encoder $f _ { S }$ that models temporal dependencies across sensor measurements, yielding $z ^ { S } = f _ { S } ( \mathbf { X } )$ . This embedding captures the global structure of the olfactory signal. To enable alignment with visual and textual modalities, we apply two modality-specific projection heads:

$$
z _ {I} ^ {S} = \phi_ {I} (z ^ {S}) \in \mathbb {R} ^ {5 1 2}, \qquad z _ {T} ^ {S} = \phi_ {T} (z ^ {S}) \in \mathbb {R} ^ {5 1 2},
$$

which map the olfactory representation toward the visual and textual embedding spaces, respectively (Figure 3a).

Visual and Textual Encoders. We obtain visual embeddings $z ^ { I }$ using CLIP’s [47] image encoder, and textual embeddings $z ^ { T }$ using CLIP’s text encoder, as described in Section 3.1. We have:

$$
z ^ {I} = f _ {I} (I) \in \mathbb {R} ^ {5 1 2}, \qquad z ^ {T} = f _ {T} \left(\operatorname{prompt} (t _ {\mathrm{obj}}, t _ {\mathrm{ctx}}, t _ {\mathrm{smell}} ^ {(1)}, \ldots , t _ {\mathrm{smell}} ^ {(K)})\right) \in \mathbb {R} ^ {5 1 2}.
$$

While the text encoder remains frozen to preserve semantic structure, the image encoder is fine-tuned to adapt its features towards olfactory-relevant cues.

Multimodal Alignment. We train the smell encoder by aligning the projected smell representations with their corresponding visual and textual embeddings using symmetric InfoNCE losses [44, 47]. Let sim(·, ·) denote cosine similarity. For a batch of N samples we compute

$$
\mathcal {L} _ {I S} = - \frac {1}{2 N} \sum_ {i = 1} ^ {N} \left(\log \frac {\exp (\mathrm{sim} (z _ {I , i} ^ {S} , z _ {i} ^ {I}) / \tau)}{\sum_ {j = 1} ^ {N} \exp (\mathrm{sim} (z _ {I , i} ^ {S} , z _ {j} ^ {I}) / \tau)} + \log \frac {\exp (\mathrm{sim} (z _ {I , i} ^ {S} , z _ {i} ^ {I}) / \tau)}{\sum_ {j = 1} ^ {N} \exp (\mathrm{sim} (z _ {I , j} ^ {S} , z _ {i} ^ {I}) / \tau)}\right)
$$

where τ is a learnable temperature parameter. $\mathcal { L } _ { S T }$ is computed analogously using $( z _ { T } ^ { S } , z ^ { T } )$ . The overall representation learning objective is

$$
\mathcal {L} _ {\mathrm{total}} = \lambda_ {I S} \mathcal {L} _ {I S} + \lambda_ {S T} \mathcal {L} _ {S T}.\tag{1}
$$

This stage produces an olfactory embedding that is jointly structured by the smell measurements, the visual scene, and the augmented language descriptors generated by the VLM.

## 3.3 Latent Smell Disentanglement

The representation $z ^ { S }$ learned in the previous stage captures the overall olfactory signal present in the measurement. However, smell recordings often contain a mixture of sources, including odors emitted from the target object as well as background environmental factors. To disentangle these contributions, we introduce a latent decomposition of the olfactory representation into object-specific and background-related components. This step is illustrated in Figure 3b.

Latent Decomposition. Our goal is to decompose the learned embedding into two components:

$$
z _ {\mathrm{obj}} ^ {S} = \phi_ {\mathrm{obj}} (z _ {T} ^ {S}) \qquad \mathrm{and} \qquad z _ {\mathrm{ctx}} ^ {S} = \phi_ {\mathrm{ctx}} (z _ {T} ^ {S}) \quad ,
$$

where $z _ { \mathrm { o b j } } ^ { S }$ and $z _ { \mathrm { c t x } } ^ { S }$ exclusively represent object-related odor factors and the remaining contextual environmental contributions, respectively.

Semantic Alignment. The decomposition is enabled by the practical compositionality of language descriptors, i.e. we have separate text encodings for an ‘object’ and the ‘context’ of the scene it is located in. During training, each decoded smell component is aligned with its corresponding textual descriptor obtained in Section 3.1 to provide explicit supervision: the object component $z _ { \mathrm { o b j } } ^ { S }$ is aligned with the object descriptor $t _ { \mathrm { o b j } }$ , while the contextual component $z _ { \mathrm { c t x } } ^ { S }$ is aligned with the context descriptor $t _ { \mathrm { c t x } }$

Denoting $z _ { \mathrm { o b j } } ^ { T }$ and $z _ { \mathrm { c t x } } ^ { T }$ the corresponding CLIP text embeddings. We enforce this decomposition through contrastive losses. Let sim $( \cdot , \cdot )$ denote cosine similarity. For a batch of N samples, we compute:

$$
\mathcal {L} _ {o b j} = - \frac {1}{N} \sum_ {i = 1} ^ {N} \log \frac {\exp (\mathrm{sim} (z _ {o b j , i} ^ {S} , z _ {o b j , i} ^ {T}) / \tau)}{\sum_ {j = 1} ^ {N} \exp (\mathrm{sim} (z _ {o b j , i} ^ {S} , z _ {o b j , j} ^ {T}) / \tau)}
$$

$$
\mathcal {L} _ {c t x} = - \frac {1}{N} \sum_ {i = 1} ^ {N} \log \frac {\exp (\mathrm{sim} (z _ {c t x , i} ^ {S} , z _ {c t x , i} ^ {T}) / \tau)}{\sum_ {j = 1} ^ {N} \exp (\mathrm{sim} (z _ {c t x , i} ^ {S} , z _ {c t x , j} ^ {T}) / \tau)}.
$$

Reconstruction Constraint. To ensure that the decomposition preserves the information contained in the original smell signal without representation collapse, we also implement a reconstruction objective. A decoder $d ( \cdot )$ predicts the smell signal from the concatenated latents:

$$
\mathcal {L} _ {\mathrm{rec}} = \left\| \mathbf {X} - d ([ z _ {\mathrm{obj}} ^ {S}, z _ {\mathrm{ctx}} ^ {S} ]) \right\| ^ {2}.
$$

The disentanglement stage is therefore a weighted combination of these losses:

$$
\mathcal {L} _ {\mathrm{dis}} = \mathcal {L} _ {\mathrm{obj}} + \mathcal {L} _ {\mathrm{ctx}} + \lambda_ {\mathrm{rec}} \mathcal {L} _ {\mathrm{rec}}.\tag{2}
$$

$\lambda _ { r e c }$ is a weighting factor for the reconstruction loss, which acts as a regularizer to prevent embedding collapse. This decomposition encourages SCENT to separate object-specific odor signals from contextual environmental contributions while remaining consistent with the underlying sensor measurements.

## 4 Experiments

## 4.1 Implementation details and experiment settings

Dataset. We evaluate SCENT on the New York Smells (NYS) dataset [46], the largest multimodal benchmark for in-the-wild olfactory perception. NYS contains 7,000 paired image-smell samples across 3,500 distinct object categories. We use 5,996 training and 936 validation samples; all baselines and ablations share this split.

Implementation Details. Each olfactory sample comprises a 32-channel resistance signal from an e-nose array, consisting of a baseline recording $( \mathbf { B } \in \mathbb { R } ^ { C \times T } )$ of ambient air and a sample recording $( \mathbf { S } \in \mathbb { R } ^ { C \times T } )$ taken near the object, where $C = 3 2$ is the number of sensor channels and T denotes the temporal dimension (typically 14 timesteps). Following [46], we use a unified olfactory input X formed by the temporal concatenation of the two signals: $\mathbf { X } = [ \mathbf { B } ; \mathbf { S } ] \in \bar { \mathbb { R } ^ { C \times 2 T } }$ Our olfactory encoder $f _ { S }$ is a Transformer with 6 layers and 8 attention heads. The input signal X is projected to a 448 dimensional embedding and augmented with sinusoidal positional encodings, the resulting olfactory representation $z ^ { S }$ is then mapped to the 512-dimensional CLIP visual and textual spaces using two MLP projection heads. For semantic augmentation, we use Qwen3VL-30B [67] to generate descriptors $\{ t _ { \mathrm { o b j } } , t _ { \mathrm { c t x } } , t _ { \mathrm { s m e l l } } \}$ . We employ CLIP ViT-B/16 [47] as our image-language encoder backbone. During training, the text encoder remains frozen, while the vision encoder is trainable, unless stated otherwise in specific ablation studies. Additional architectural and training details are provided in the supplementary material.

Evaluation Metrics and Retrieval Tasks. We evaluate SCENT across two primary categories, single-modality and joint-modality retrieval, to assess the alignment of olfactory features with individual and combined semantic spaces. We report standard retrieval metrics: Recall@k (%) (R@k for $k \in \{ 1 , 5 , 1 0 , 2 0 \}$ ).

Single-modality Retrieval (S2I, S2T): These tasks evaluate the model’s ability to align olfactory signals with a single target modality. In Smell-to-Image (S2I) and Smell-to-Text (S2T), a query smell signal is used to rank and retrieve candidates from a set of images or textual descriptors, respectively, based on cross-modal similarity.

Joint-modality Retrieval (S2IT): This task evaluates the model’s performance in a fully multimodal setting. The query is a singular olfactory signal, and the search database consists of (Image, Text) pairs. For each candidate pair in the gallery, the model must compute a joint similarity score that accounts for both the visual and textual components. We compute this score via latent-level fusion of the dual olfactory projections; full details and a comparison with an alternative fusion strategy are provided in the supplementary material.

Baselines. As the current state-of-the-art on the NYS dataset [46] is a visiononly model, it serves as our primary comparison for the S2I task. However, since this baseline lacks a native textual projection, it cannot be directly applied to tasks involving language. Moreover, the original work does not provide publicly available code or pretrained weights, requiring us to reproduce the model based on the descriptions in the paper. To establish a rigorous baseline for text-based tasks, we reproduce the NYS model and implement NYS (adapted) for multimodal retrieval tasks, which utilizes a two-stage “Image Bridge” (IB) protocol:

– S2T Retrieval: The model first retrieves the top-1 most visually similar image to the query smell using its olfactory-visual head. This “bridge” image is then encoded via a frozen CLIP ViT-B/16 [47] model and used to rank textual candidates in the CLIP latent space.

– Joint Retrieval (S2IT): For S2IT, a composite score is formed by summing the smell-image similarity (from the NYS model) with the pre-existing image-text alignment score (from CLIP) for each candidate pair.

## 4.2 The Necessity of Language Guidance

We first motivate SCENT by evaluating the impact of diferent supervision levels while keeping visual and textual backbones frozen (Table 1). Aligning smell solely with images (S2I) achieves a baseline R@5 of 12.1. When aligning strictly with text (S2T), we observe that performance is highly dependent on the semantic granularity of the descriptors. Using only isolated object labels (O) results in poor alignment (8.8 R@5), as these labels fail to capture the environmental factors present in the sensor data. However, as we incrementally add contextual descriptors (Ctx) and VLM-inferred ambient smells (S), performance improves significantly, eventually surpassing the image-only baseline. The most robust representations are formed through the joint Image&Text strategy. By aligning the smell encoder with both modalities simultaneously using our semantic-rich template $\mathbf { \Gamma } ( \mathbf { O } + \mathbf { C } \mathbf { t x } + \mathbf { S } )$ , we achieve 14.3 R@5. This confirms that language guidance ofers a critical semantic bridge that visual features alone cannot provide.

Table 1: Impact of semantics on olfactory alignment. We evaluate the retrieval performance of our olfactory encoder when aligned with frozen CLIP visual and textual spaces. Results indicate that simple object labels (O) provide insuficient supervision, whereas incorporating context (Ctx), and inferred ambient smells (I) yields the better cross-modal alignment across both single and joint retrieval tasks. Best results in bold.

<table><tr><td rowspan="2">Align. Strategy</td><td colspan="3">Textual Info</td><td colspan="3">Smell Retrieval</td></tr><tr><td>O</td><td>Ctx</td><td>S</td><td>R@5</td><td>R@10</td><td>R@20</td></tr><tr><td>Image (S2I)</td><td></td><td></td><td></td><td>12.1</td><td>18.3</td><td>29.2</td></tr><tr><td rowspan="3">Text (S2T)</td><td>√</td><td></td><td></td><td>8.8 (-3.3)</td><td>14.4 (-3.9)</td><td>24.9 (-4.3)</td></tr><tr><td>√</td><td>√</td><td></td><td>10.8 (-1.3)</td><td>18.3 (0.0)</td><td>28.1 (-1.1)</td></tr><tr><td>√</td><td>√</td><td>√</td><td>12.4 (+0.3)</td><td>19.2 (+0.9)</td><td>29.0 (-0.2)</td></tr><tr><td>Image &amp; Text (S2IT)</td><td>√</td><td>√</td><td>√</td><td>14.3 (+2.2)</td><td>22.5 (+4.2)</td><td>35.1 (+5.9)</td></tr></table>

## 4.3 Comparison to State-of-the-Art

In Table 2, we compare SCENT against the state-of-the-art NYS benchmark.

S2I Retrieval. We observe that language guidance improves performance even on purely visual tasks. In the S2I setting, SCENT outperforms the reproduced NYS baseline (R@5 from 20.0 to 22.1) even only with object labels O. This confirms that the semantic structure of language helps the olfactory encoder extract more discriminative features than vision alone. As contextual (Ctx) and inferred smell (S) descriptors are added, S2I performance peaks at 23.0 R@5, indicating that a more complete semantic description of the scene complements the visual signal and improves the olfactory representation.

S2T and S2IT Retrieval. On tasks involving language, the advantage of our proposed architecture is even more apparent. The NYS (adapted) baseline struggles with the semantic gap, as its “Image Bridge” is limited by what is explicitly visible. In contrast, SCENT leverages its native textual projection head to achieve 11.9 R@5 on S2T retrieval, a 47% relative improvement over the best adapted baseline. Furthermore, our joint-modality retrieval (S2IT) results show that SCENT efectively fuses visual and textual cues, outperforming the baseline by over 7% in R@5.

![](images/3dbf70d43606dfe8c01fab9e5021ea0ee6a19ff4cb63a41049a81bc874e92ddf.jpg)  
Fig. 4: Mean cosine similarity between the ground-truth and top-k retrieved items, in frozen CLIP image space (orange) and learned olfactory space (teal), for $k \in \{ 1 , 5 , 5 0 \}$ Solid lines: SCENT; dashed: NYS.

Visual vs. Olfactory Discriminativity. Figure 4 plots the mean cosine similarity between the ground-truth item and the top-k retrieved items, measured in frozen CLIP image space (orange) and in the learned olfactory space (teal), for $k \in \{ 1 , 5 , 5 0 \}$ . Image features are weakly discriminative: scores remain nearly flat across retrieval ranks, failing to separate good retrievals (R@1) from bad ones (R@50). In contrast, olfactory features drop sharply with retrieval rank; SCENT’s smell curve (solid) decays faster than NYS’s (dashed), confirming both that smell is the necessary modality for this task and that SCENT learns a more discriminative representation. Notably, NYS achieves similar S2T visual similarity to SCENT yet 32% lower R@5, showing that visual proximity does not imply correct olfactory match.

Qualitative Analysis. We visualize the S2IT retrieval performance in Figure 5. Each row corresponds to one example and the columns illustrate the top-5 images from the retrieved (Image,Text) pairs. Despite the inherent dificulty of the olfactory-to-visual mapping, SCENT consistently ranks the correct semantic category within the top-5 results. Notably, the model handles fine-grained diferences (e.g., distinguishing between diferent types of foliage or tree barks) and identifies objects regardless of their visual state, such as the peeled banana or the sliced pizza, by leveraging the underlying olfactory cue.

## 4.4 Ablation Studies

Impact of Semantic Granularity. We first evaluate how the complexity of the textual descriptors impacts olfactory alignment. As shown in Table 2, we observe a consistent performance gain across all tasks as the supervision transitions from simple object labels (O) to scene-aware descriptions that include environmental context (Ctx) and inferred ambient smells (S). In the S2I task, expanding the semantic scope beyond basic labels improves R@5 from 22.1 to 23.0, suggesting that language context complements the missing information from the visual modality during olfactory representation training. This trend is even more evident in language-centric tasks: for S2T, moving from O to the full O + Ctx + S template yields an improvement from 8.0 to 11.9 R@5. Notably, the inclusion of VLM-inferred smells provides the best performance, validating our hypothesis that while visual features may miss the olfactory essence of a scene, high-level language inferring can recover these unobservable cues to better align the olfactory latent space.

![](images/433ae5c947dc29778a325d4824d09841fc625baab92b9be89033f9ab66e42ee3.jpg)  
Fig. 5: Qualitative results for S2IT retrieval. For a given olfactory query, we show the ground-truth image in the Query column and the top-5 images from the retrieved (Image, Text) pairs. The Correct retrieval is highlighted. SCENT demonstrates robust cross-modal alignment across diverse categories, successfully retrieving the correct match even when the visual appearance varies significantly from the query (e.g., the pizza box in Rank 1 vs. the open pizza in query).

Image Bridge Ablation. To isolate the benefit of our three-stream architecture from the VLM-generated language guidance, we apply the Image Bridge protocol to SCENT: we disable the ST head and route S2T/S2IT retrieval via the IS head and a frozen CLIP proxy, mirroring the NYS<sup>†</sup> (adapted) baseline (Table 3). S2T and S2IT degrade, confirming that the gains stem from the three-stream design and not solely from VLM-enriched training data. Notably, SCENT with IB still outperforms NYS<sup>†</sup> (adapted) with IB, showing that co-training with the ST loss also improves the IS head.

VLM Sensitivity Analysis We evaluate how the choice of Vision-Language Model afects retrieval performance, keeping all other components of SCENT fixed. Results are shown in Table 5. Performance scales with VLM reasoning quality rather than parameter count alone: Qwen3-VL-30B outperforms the larger Qwen2.5- VL-72B across all tasks, consistent with the multi-hop reasoning required to infer plausible olfactory cues from visual context. Smaller models (Qwen3-VL-8B, Gemma 4 E4B) produce weaker annotations, yielding lower retrieval performance, particularly on language-centric tasks (S2T). These results confirm that annotation richness, and therefore downstream alignment quality, is driven primarily by the model’s reasoning capability rather than its scale alone.

Table 2: State-of-the-art comparison on the NYS dataset. We compare SCENT against the vision-only NYS baseline [46] across single-modality (S2I, S2T) and jointmodality (S2IT) tasks. To enable a comparison on language-based tasks, we implement NYS (adapted) using an "Image Bridge" protocol. Our method consistently outperforms the adapted baseline, demonstrating that native three-stream alignment with inferred semantic cues captures nuanced olfactory information that is unobservable through visual features alone.

<table><tr><td rowspan="2">Method</td><td colspan="3">Textual Info</td><td colspan="4">S2I</td><td colspan="4">S2T</td><td colspan="4">S2IT</td></tr><tr><td>O</td><td>Ctx</td><td>S</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td></tr><tr><td>NYS* [46]</td><td></td><td></td><td></td><td>—</td><td>16.5</td><td>29.6</td><td>43.1</td><td></td><td>—</td><td></td><td></td><td></td><td>—</td><td></td><td></td></tr><tr><td>NYS $^{\dagger}$ </td><td></td><td></td><td></td><td>5.4</td><td>20.0</td><td>29.9</td><td>42.0</td><td></td><td>—</td><td></td><td></td><td></td><td>—</td><td></td><td></td></tr><tr><td rowspan="3">NYS $^{\dagger}$  (adapted)</td><td>√</td><td></td><td></td><td></td><td></td><td>—</td><td></td><td>1.7</td><td>6.2</td><td>10.4</td><td>14.4</td><td>3.9</td><td>15.6</td><td>25.8</td><td>39.1</td></tr><tr><td>√</td><td>√</td><td></td><td></td><td></td><td>—</td><td></td><td>2.5</td><td>8.7</td><td>12.7</td><td>18.3</td><td>3.9</td><td>15.9</td><td>25.2</td><td>38.7</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td></td><td>—</td><td></td><td>2.6</td><td>8.1</td><td>12.9</td><td>19.2</td><td>4.9</td><td>18.5</td><td>27.5</td><td>40.7</td></tr><tr><td rowspan="3">Ours</td><td>√</td><td></td><td></td><td>5.6</td><td>22.1</td><td>32.7</td><td>42.4</td><td>1.6</td><td>8.0</td><td>13.8</td><td>21.5</td><td>5.9</td><td>22.0</td><td>32.4</td><td>42.6</td></tr><tr><td>√</td><td>√</td><td></td><td>5.9</td><td>21.8</td><td>32.4</td><td>45.2</td><td>2.8</td><td>10.7</td><td>17.3</td><td>28.1</td><td>6.1</td><td>21.3</td><td>32.5</td><td>45.7</td></tr><tr><td>√</td><td>√</td><td>√</td><td>6.0</td><td>23.0</td><td>33.5</td><td>43.6</td><td>3.3</td><td>11.9</td><td>19.8</td><td>29.2</td><td>6.8</td><td>23.3</td><td>32.7</td><td>42.5</td></tr></table>

<sup>∗</sup> Results reported in the original NYS paper [46].  
<sup>†</sup> Results reproduced.

## 4.5 Annotation quality.

To verify that VLM-generated descriptors infer real olfactory context rather than plausible hallucinations, we use a second VLM, the “judge”. We provide a held-out second view of the same scene <sup>1</sup>, and the judge scores each annotation on two criteria: plausibility (is the annotation contextually reasonable given both views?), reaching 98.6%; and View-2 confirmation (does the annotation specifically predict content hidden in View 1 but visible in View 2?), reaching 32.2%. This confirms that our descriptors capture real concealed scene content rather than generic guesses, as illustrated in Figure 6.

## 4.6 Latent Smell Disentanglement

We evaluate the ability of SCENT to decompose a complex olfactory signal into its constituent parts: the target object and the environmental context.

Evaluation Protocol: Recombination Retrieval. To evaluate whether the latent decomposition $\phi _ { \mathrm { o b j } }$ and $\phi _ { \mathrm { c t x } }$ efectively isolates object-specific and environmental olfactory components, we propose a zero-shot recombination retrieval task. We identify (Object, Context) pairs in the validation set that never appear together in the training data, although the individual objects and contexts are present separately across diferent training samples.

Table 3: Image Bridge (IB) ablation. Replacing SCENT’s native ST head with an Image Bridge proxy confirms that performance gains stem from the three-stream architecture.  
Table 4: Recombination retrieval. Decoded Synthesis (ours) outperforms Raw Recombination on zeroshot (Object, Context) pairings unseen during training.

<table><tr><td rowspan="2">Method</td><td rowspan="2">IB</td><td colspan="2">S2T</td><td colspan="2">S2IT</td></tr><tr><td>R@5</td><td>R@20</td><td>R@5</td><td>R@20</td></tr><tr><td>NYS $^{\dagger}$  (adapted)</td><td>✓</td><td>8.1</td><td>19.2</td><td>18.5</td><td>40.7</td></tr><tr><td>SCENT</td><td>✓</td><td>9.1</td><td>18.8</td><td>21.4</td><td>43.8</td></tr><tr><td>SCENT (ours)</td><td>✘</td><td>11.9</td><td>29.2</td><td>23.3</td><td>42.5</td></tr></table>

<table><tr><td>Query Type</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td></tr><tr><td>Raw Recombination</td><td>3.9</td><td>5.9</td><td>7.8</td><td>9.8</td></tr><tr><td>Decoded Synthesis (ours)</td><td>2.0</td><td>9.8</td><td>11.8</td><td>13.7</td></tr></table>

Table 5: VLM sensitivity analysis. All variants are SCENT trained with O+Ctx+ S descriptors; only the VLM used for annotation difers. MMLU score is reported as a proxy for general reasoning capability.

<table><tr><td rowspan="2">VLM</td><td rowspan="2">MMLU</td><td colspan="2">S2I</td><td colspan="2">S2T</td><td colspan="2">S2IT</td></tr><tr><td>R@5</td><td>R@20</td><td>R@5</td><td>R@20</td><td>R@5</td><td>R@20</td></tr><tr><td>Gemma 4 E4B [25]</td><td>69.4</td><td>19.6</td><td>43.3</td><td>7.7</td><td>19.8</td><td>20.2</td><td>44.4</td></tr><tr><td>Qwen2.5-VL-72B [3]</td><td>71.2</td><td>20.0</td><td>42.1</td><td>8.1</td><td>20.5</td><td>19.7</td><td>41.9</td></tr><tr><td>Qwen3-VL-8B [67]</td><td>71.6</td><td>19.1</td><td>39.9</td><td>6.5</td><td>22.0</td><td>20.3</td><td>40.0</td></tr><tr><td>Qwen3-VL-30B [67] (ours)</td><td>77.8</td><td>23.0</td><td>43.6</td><td>11.9</td><td>29.2</td><td>23.3</td><td>42.5</td></tr></table>

For each such pair, we construct a synthetic query by pairing an object factor $z _ { \mathrm { o b j } } ^ { S }$ from one training sample (e.g., “cofee”) with a context factor $z _ { \mathrm { c t x } } ^ { S }$ from a distinct training sample with the target background $( \mathrm { e . g . }$ , “outdoor park”). We then utilize the decoder $g ( \cdot )$ to synthesize a novel smell fingerprint $\hat { \textbf { X } } =$ $g ( [ z _ { \mathrm { o b j } } ^ { S } ; z _ { \mathrm { c t x } } ^ { S } ] )$ . The task is to retrieve the ground-truth validation samples where this specific combination occurred.

We compare our Decoded Synthesis against a Raw Recombination baseline, which creates a query by manually concatenating the raw sensor baseline (B) and sample (S) signals from two diferent training exemplars. This protocol tests the model’s ability to perform compositional generalization.

Quantitative Analysis. As shown in Table 4, we evaluate the retrieval performance of these synthetic queries within our latent space, by passing the query smell fingerprint X through our smell encoder. Our Decoded Synthesis outperforms the Raw Recombination baseline. While the raw baseline plateaus at 9.8 R@20, our synthesized fingerprints achieve 13.7 R@20.

![](images/7f5575c729a0102fd37dc221bebf94f6bc2d10a319e2954a0299d5a85238df92.jpg)  
Fig. 6: Held-out view validation setup. A VLM infers plausible smells from View 1; a second VLM judge uses a held-out View 2 to verify the inferences, distinguishing grounded predictions from hallucinations.

## 5 Limitations and Conclusion

Despite the performance gains of SCENT, several challenges remain. The reliance on e-nose sensors introduces inherent stochasticity, as these hardware devices are prone to temporal drift and cross-sensitivity to environmental factors such as humidity and temperature. Furthermore, while the NYS dataset is a significant milestone, a substantial gap remains between olfactory benchmarks and the webscale datasets used to train vision-language backbones. Narrowing this gap may benefit from synthetic data augmentation [6].

In this paper, we addressed the fundamental challenge of aligning highdimensional electronic-nose signals with visual data, recognizing that images often fail to capture the pervasive environmental context of olfaction. We introduced SCENT, a multimodal framework that employs language guidance as a semantic bridge, leveraging the extensive world knowledge embedded in VLMs to infer plausible ambient olfactory cues. Through a language-guided latent decomposition, our model efectively disentangles object-specific odors from broader environmental factors. Extensive experiments on the New York Smells (NYS) dataset demonstrate that this semantic grounding enhances cross-modal retrieval performance across both single and joint modality tasks. Moroever, our language-guidance mechanism achieves decompositionality of olfactory information, facilitating generalization to out-of-distribution (OOD) samples.

## Acknowledgements

This work is supported by Hi! Paris and the ANR/France 2030 program (ANR-23-IACL-0005), a Hi! Paris grant and fellowship, a CIEDS grant, and a Google DeepMind academic gift. Computing resources were provided by GENCI through access to the IDRIS High-Performance Computing facilities under allocations 2026-AD011014300R3 and 2025-AD011015893R1, and by Google Gemini. We sincerely thank Mathieu Aubry and the anonymous reviewers for their insightful discussions that contributed to this work. We are also grateful to Julie Mordacq and Robin Courant for their meticulous proofreading.

## References

1. Ali, M.M., Hashim, N., Abd Aziz, S., Lasekan, O.: Principles and recent advances in electronic nose for quality inspection of agricultural and food products. Trends in Food Science & Technology (2020)

2. Aytar, Y., Vondrick, C., Torralba, A.: Soundnet: Learning sound representations from unlabeled video. In: NeurIPS (2016)

3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. arXiv preprint arXiv: 2502.13923 (2025)

4. Bender, T., Sørensen, S., Kashani, A., Eldjarn Hjorleifsson, K., Hyldig, G., Hauberg, S., Belongie, S., Warburg, F.: Learning to taste: A multimodal wine dataset. In: NeurIPS (2023)

5. Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al.: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164 (2024)

6. Boudier, L., Manganelli, L., Tsonis, E., Dufour, N., Kalogeiton, V.: Training-free synthetic data generation with dual ip-adapter guidance. In: BMVC (2025)

7. Brattoli, M., De Gennaro, G., De Pinto, V., Loiotile, A.D., Lovascio, S., Penza, M.: Odour detection methods: Olfactometry and chemical sensors. Sensors (2011)

8. Chen, H., Huo, D., Zhang, J.: Gas recognition in e-nose system: A review. IEEE transactions on biomedical circuits and systems (2022)

9. Chen, S., Li, H., Wang, Q., Zhao, Z., Sun, M.T., Zhu, X., Liu, J.: VAST: A visionaudio-subtitle-text omni-modality foundation model and dataset. In: NeurIPS (2023)

10. Cicchetti, G., Grassucci, E., Comminiello, D.: A triangle enables multimodal alignment beyond cosine similarity. In: NeurIPS (2025)

11. Cicchetti, G., Grassucci, E., Sigillo, L., Comminiello, D.: Gramian multimodal representation learning and alignment. In: ICLR (2025)

12. De Sa, V.: Learning classification with unlabeled data. In: NeurIPS (1993)

13. Debnath, T., Nakamoto, T.: Predicting human odor perception represented by continuous values from mass spectra of essential oils resembling chemical mixtures. PloS one (2020)

14. Dou, Y., Yang, F., Liu, Y., Loquercio, A., Owens, A.: Tactile-augmented radiance fields. In: CVPR (2024)

15. Dravnieks, A.: Atlas of Odor Character Profiles. ASTM Special Technical Publication (1985)

16. Dufumier, B., Castillo Navarro, J., Tuia, D., Thiran, J.P.: What to align in multimodal contrastive learning? In: ICLR (2025)

17. Elizalde, B., Deshmukh, S., Al Ismail, M., Wang, H.: CLAP: learning audio concepts from natural language supervision. In: ICASSP (2023)

18. Erlangga, F., Wijaya, D.R., Wikusna, W.: Electronic nose dataset for classifying rice quality using neural network. In: International Conference on Information and Communication Technology (ICoICT) (2021)

19. Fang, X., Mao, K., Duan, H., Zhao, X., Li, Y., Lin, D., Chen, K.: Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. NeurIPS (2024)

20. Feng, D., Dai, W., Li, C., Pernigo, A., Wen, Y., Liang, P.P.: Smellnet: A large-scale dataset for real-world smell recognition. In: ICLR (2026)

21. Fundurulic, A., Faria, J.M., Inácio, M.L.: Advances in electronic nose sensors for plant disease and pest detection. Engineering Proceedings (2023)

22. Furizal, F., Ma’arif, A., Firdaus, A.A., Rahmaniar, W.: Future potential of enose technology: A review. International Journal of Robotics and Control Systems (2023)

23. Ghazaly, C., Biletska, K., Thevenot, E.A., Devillier, P., Naline, E., Grassin-Delyle, S., Scorsone, E.: Assessment of an e-nose performance for the detection of covid-19 specific biomarkers. Journal of Breath Research (2023)

24. Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K.V., Joulin, A., Misra, I.: Imagebind: One embedding space to bind them all. In: CVPR (2023)

25. Google DeepMind: Gemma model documentation. https://ai.google.dev/ gemma/docs/core (2026), accessed: 2026-06-30

26. Han, J., Gong, K., Zhang, Y., Wang, J., Zhang, K., Lin, D., Qiao, Y., Gao, P., Yue, X.: Onellm: One framework to align all modalities with language. In: CVPR (2024)

27. Huh, M., Cheung, B., Wang, T., Isola, P.: The platonic representation hypothesis. arXiv preprint arXiv:2405.07987 (2024)

28. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: ICML (2021)

29. Keller, A., Gerkin, R.C., Guan, Y., Dhurandhar, A., Turu, G., Szalai, B., Mainland, J.D., Ihara, Y., Yu, C.W., Wolfinger, R., et al.: Predicting human olfactory perception from chemical features of odor molecules. Science (2017)

30. Kim, M.J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E.P., Sanketi, P.R., Vuong, Q., et al.: Openvla: An open-source vision-language-action model. In: 8th Annual Conference on Robot Learning (2025)

31. Kontaris, I., East, B.S., Wilson, D.A.: Behavioral and neurobiological convergence of odor, mood and emotion: A review. Frontiers in Behavioral Neuroscience (2020)

32. Koutoupis, S., Zervou, M.A., Kontras, K., De Vos, M., Tsakalides, P., Tsagkatakis, G.: The more, the merrier: Contrastive fusion for higher-order multimodal alignment. In: CVPR (2026)

33. Lee, B.K., Mayhew, E.J., Sanchez-Lengeling, B., Wei, J.N., Qian, W.W., Little, K.A., Andres, M., Nguyen, B.B., Moloy, T., Yasonik, J., et al.: A principal odor map unifies diverse tasks in olfactory perception. Science (2023)

34. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al.: Llava-onevision: Easy visual task transfer. IEEE Transactions on Machine Learning Research (2025)

35. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: ICML (2023)

36. Liu, J., Chen, S., He, X., Guo, L., Zhu, X., Wang, W., Tang, J.: Valor: Vision-audiolanguage omni-perception pretraining model and dataset. IEEE Transactions on Pattern Analysis and Machine Intelligence (2024)

37. Liu, Y., Li, S., Wu, Y., Chen, C.W., Shan, Y., Qie, X.: Umt: Unified multi-modal transformers for joint video moment retrieval and highlight detection. In: CVPR (2022)

38. Luo, H., Ji, L., Zhong, M., Chen, Y., Lei, W., Duan, N., Li, T.: Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing (2022)

39. Makri, E., Nakis, N., Sisson, L., Minsky, G., Tassiulas, L., Satarifard, V., Christakis, N.A.: Benchmark for assessing olfactory perception of large language models. arXiv preprint arXiv:2604.00002 (2026)

40. Moein, S.T., Hashemian, S.M., Mansourafshar, B., Khorram-Tousi, A., Tabarsi, P., Doty, R.L.: Smell dysfunction: a biomarker for covid-19. In: International forum of allergy & rhinology (2020)

41. Mordacq, J., Milecki, L., Vakalopoulou, M., Oudot, S., Kalogeiton, V.: Adapt: Multimodal learning for detecting physiological changes under missing modalities. In: Medical Imaging with Deep Learning (2024)

42. Mueller, P., Salminen, K., Nieminen, V., Kontunen, A., Karjalainen, M., Isokoski, P., Rantala, J., Savia, M., Väliaho, J., Kallio, P., et al.: Scent classification by k nearest neighbors using ion-mobility spectrometry measurements. Expert systems with applications (2019)

43. Ngiam, J., Khosla, A., Kim, M., Nam, J., Lee, H., Ng, A.Y., et al.: Multimodal deep learning. In: ICML (2011)

44. Oord, A.v.d., Li, Y., Vinyals, O.: Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018)

45. Owens, A., Isola, P., McDermott, J., Torralba, A., Adelson, E.H., Freeman, W.T.: Visually indicated sounds. In: CVPR (2016)

46. Ozguroglu, E., Liang, J., Liu, R., Chiquier, M., DeTienne, M., Qian, W.W., Horowitz, A., Owens, A., Vondrick, C.: New york smells: A large multimodal dataset for olfaction. arXiv preprint arXiv:2511.20544 (2025)

47. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021)

48. Ravia, A., Snitz, K., Honigstein, D., Finkel, M., Zirler, R., Perl, O., Secundo, L., Laudamiel, C., Harel, D., Sobel, N.: A measure of smell enables the creation of olfactory metamers. Nature (2020)

49. Rodríguez, J., Durán, C., Reyes, A.: Electronic nose for quality control of colombian cofee through the detection of defects in “cup tests”. Sensors (2009)

50. Ruan, L., Hu, A., Song, Y., Zhang, L., Zheng, S., Jin, Q.: Accommodating audio modality in clip for multimodal processing. In: AAAI (2023)

51. Sanchez-Lengeling, B., Wei, J.N., Lee, B.K., Gerkin, R.C., Aspuru-Guzik, A., Wiltschko, A.B.: Machine learning for scent: Learning generalizable perceptual representations of small molecules. arXiv preprint arXiv:1910.10685 (2019)

52. Saporta, A., Puli, A.M., Goldstein, M., Ranganath, R.: Contrasting with symile: Simple model-agnostic representation learning for unlimited modalities. In: NeurIPS (2024)

53. Snitz, K., Yablonka, A., Weiss, T., Frumin, I., Khan, R.M., Sobel, N.: Predicting odor perceptual similarity from odor structure. PLoS computational biology (2013)

54. Stevenson, R.J.: An initial evaluation of the functions of human olfaction. Chemical senses (2010)

55. Tan, J., Xu, J.: Applications of electronic nose (e-nose) and electronic tongue (e-tongue) in food quality-related properties determination: A review. Artificial Intelligence in Agriculture (2020)

56. Torres-Tello, J., Guaman, A.V., Ko, S.B.: Improving the detection of explosives in a mox chemical sensors array with lstm networks. IEEE Sensors Journal (2020)

57. Vergara, A., Vembu, S., Ayhan, T., Ryan, M.A., Homer, M.L., Huerta, R.: Chemical gas sensor drift compensation using classifier ensembles. Sensors and Actuators B: Chemical (2012)

58. Vinyals, O., Toshev, A., Bengio, S., Erhan, D.: Show and tell: A neural image caption generator. In: CVPR (2015)

59. Wachowiak, M., Dewan, A., Bozza, T., O’Connell, T.F., Hong, E.J.: Recalibrating olfactory neuroscience to the range of naturally occurring odor concentrations. Journal of Neuroscience (2025)

60. Wang, W., Nie, A., Zhou, W., Kai, Y., Hu, C.: Teaching physical awareness to llms through sounds. In: ICML (2025)

61. Wang, Y., Li, K., Li, X., Yu, J., He, Y., Chen, G., Pei, B., Zheng, R., Wang, Z., Shi, Y., et al.: Internvideo2: Scaling foundation models for multimodal video understanding. In: ECCV (2024)

62. Wijaya, D.R., Sarno, R., Zulaika, E.: Dwtlstm for electronic nose signal processing in beef quality monitoring. Sensors and Actuators B: Chemical (2021)

63. Wilson, A.D.: Review of electronic-nose technologies and algorithms to detect hazardous chemicals in the environment. Procedia Technology (2012)

64. Xiao, J., Yao, A., Li, Y., Chua, T.S.: Can i trust your answer? visually grounded video question answering. In: CVPR (2024)

65. Xu, H., Ye, Q., Yan, M., Shi, Y., Ye, J., Xu, Y., Li, C., Bi, B., Qian, Q., Wang, W., et al.: mplug-2: A modularized multi-modal foundation model across text, image and video. In: ICML (2023)

66. Xue, L., Yu, N., Zhang, S., Panagopoulou, A., Li, J., Martín-Martín, R., Wu, J., Xiong, C., Xu, R., Niebles, J.C., et al.: Ulip-2: Towards scalable multimodal pretraining for 3d understanding. In: CVPR (2024)

67. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al.: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025)

68. Yang, F., Ma, C., Zhang, J., Zhu, J., Yuan, W., Owens, A.: Touch and go: Learning from human-collected vision and touch. In: NeurIPS Datasets and Benchmarks Track (2022)

69. Yu, J., Wang, X., Tu, S., Cao, S., Zhang-Li, D., Lv, X., Peng, H., Yao, Z., Zhang, X., Li, H., et al.: Kola: Carefully benchmarking world knowledge of large language models. In: ICLR (2024)

70. Yuan, W., Wang, S., Dong, S., Adelson, E.: Connecting look and feel: Associating the visual and tactile properties of physical materials. In: CVPR (2017)

71. Zhang, B., Li, K., Cheng, Z., Hu, Z., Yuan, Y., Chen, G., Leng, S., Jiang, Y., Zhang, H., Li, X., et al.: Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106 (2025)

72. Zhang, R., Guo, Z., Zhang, W., Li, K., Miao, X., Cui, B., Qiao, Y., Gao, P., Li, H.: Pointclip: Point cloud understanding by clip. In: CVPR (2022)

73. Zhao, Y., Misra, I., Krähenbühl, P., Girdhar, R.: Learning video representations from large language models. In: CVPR (2023)

74. Zhou, B., Li, L., Wang, Y., Liu, H., Yao, Y., Wang, W.: Unialign: Scaling multimodal alignment within one unified model. In: CVPR (2025)

75. Zhu, B., Lin, B., Ning, M., Yan, Y., Cui, J., HongFa, W., Pang, Y., Jiang, W., Zhang, J., Li, Z., et al.: Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. In: ICLR (2024)

# Supplementary Material

What Images Cannot Say: Language-Guided Olfactory Representation Learning

This supplementary material provides further technical depth and empirical evidence to support our main findings. Specifically, we include: (1) comprehensive implementation details; (2) details on the semantic augmentation pipeline for extracting inference smell information; (3) additional quantitative results across diverse benchmarks; (4) ablation studies on key design choices; and (5) further qualitative examples and visualizations.

A. Implementation Details

B. Semantic Augmentation Details B.1. Prompt Templates B.2. Annotation Statistics

C. Quantitative results C.1. Inverse Retrieval Tasks: I2S and T2S C.2. Classification C.3. Performance by Scene Depth

D. Ablation Studies

D.1. Scene Inference with Diferent Language Models

D.2. Fusion Strategies for Joint-Modality Retrieval (S2IT)

D.3. Two-Stage Training Strategy

D.4. Annotation Granularity

E. Qualitative Results

## A Implementation Details

Dataset and Custom Splits. Since the original splits for the New York Smells (NYS) dataset [46] have not been released, we established an independent partitioning to facilitate our experiments. We use a training set of 5,996 samples and a validation set of 936 samples. All reported results, including the reproduced NYS baseline and our proposed SCENT, are evaluated using these identical splits to ensure a fair and consistent comparison.

Baseline Reproduction. To ensure a rigorous evaluation, we reproduce the NYS baseline [46] using our standardized data splits and the same Transformer-based smell encoder architecture employed in SCENT. By adjusting the backbone design and optimization strategy, our reproduced baseline achieves performance metrics that exceed those originally reported in [46], providing a more challenging and fair point of comparison for our multimodal approach.

Architectural Specifications. Our smell encoder $f _ { S }$ is a Transformer with a model dimension d = 448, 6 layers, and 8 attention heads. The input e-nose signal consists of 32 sensor channels across 28 timesteps (14 for the baseline B and 14 for the sample S). While most recordings in the NYS dataset follow this duration, any samples exceeding 14 seconds per phase are truncated to maintain a uniform temporal resolution. This input is projected into the embedding space via a linear layer and augmented with sinusoidal positional encodings. For the visual and textual backbones, we utilize CLIP ViT-B/16 [47]. While the textual encoder remains frozen to preserve its pre-trained semantic structure, the vision encoder is fine-tuned to align its representations with the olfactory features. The CLIP vision backbone contains 85.8M trainable parameters, while the smell encoder contains 14.9M trainable parameters. The two modality-specific projection heads, ϕ<sub>I</sub> and ϕ<sub>T</sub> , are implemented as MLPs that map the 448-dimensional olfactory vector into the 512-dimensional CLIP latent space.

Training and Optimization. Models are trained using the AdamW optimizer with a batch size of 512. To improve the robustness of the visual representations, we apply strong random resized crops and horizontal flips during training. All training is conducted in FP16 mixed precision on a single NVIDIA H100 GPU.

## B Semantic Augmentation Details

Terminology Clarification. To avoid confusion with the original NYS taxonomy [46], we distinguish between the closed-set categories used for evaluation and the open-set descriptors used for model training.

– Object (O), Context (Ctx) & Smell (S): Open-set natural language descriptors generated by our VLM pipeline. These are used directly as linguistic supervision in SCENT.

– Item (I) & Environment (E): Closed-set categorical labels (43 and 7 classes, respectively) following the taxonomy shown in [46]. These are used only for the classification probing tasks in Section C.2 of the supplementary material, and nowhere else.

## B.1 Prompt Templates

We utilize Qwen3VL-30B [67] to extract semantic anchors from scene images. To ensure the generated descriptors are relevant, we use a structured prompt that forces the model to decouple the primary olfactory information.

Usage of Descriptors. While the VLM generates five distinct fields, they serve diferent roles in our study. The open-set Object (O) and Context (Ctx), as well as the inferred Smell (S) descriptors are used to construct the languageguided supervision for SCENT. The closed-set Item and Environment fields are used exclusively as ground-truth labels for the linear probing tasks (Section C.2) to maintain compatibility with the original NYS taxonomy.

## Primary VLM Prompt.

Analyze this image for environmental olfactory context. The yellow tip of the probe (the "snout") is sampling a "smell print" from a specific object or surface.

1. ITEM (I): The general category of the object the yellow tip is touching. Choose ONE: {items\_list}.

2. OBJECT (O): A specific, open-set name for the exact object or surface the yellow tip is touching.

3. ENVIRONMENT (E): The general environment/setting. Choose ONE: {environments\_list}.

```txt
4. CONTEXT (Ctx): A short, open-set phrase describing the background/setting (e.g. park, garage, kitchen, office, street, garden, workshop). Anything that describes where the scene is.
5. INFERRED SMELLS (S): Infer likely smells from the surrounding environment ONLY.
***CRITICAL: EXCLUDE the smell of the OBJECT (O) itself.
*** Focus on "invisible" scents that are likely in the air (e.g., traffic exhaust, humidity, air conditioning, distant greenery).
```

```txt
Respond in this exact format (one field per line):
ITEM: [exactly one from the item list]
OBJECT: [name]
ENVIRONMENT: [exactly one from the environment list]
CONTEXT: [one short open-set phrase]
SMELLS: [ambient smell 1], [ambient smell 2]
```

```txt
EXAMPLE:
ITEM: plants flower ornamental
OBJECT: white daffodil
ENVIRONMENT: Campus Outdoors
BACKGROUND: a breezy concrete walkway adjacent to a freshly mowed lawn
SMELLS: fresh cut grass, concrete dust, distant vehicle exhaust --
```

Scene Inference with Diferent Language Models (Two-Stage Pipeline). In Section D.1, we evaluate an alternative pipeline to determine whether olfactory reasoning is best performed by grounded vision-language models or via decoupled language reasoning. In this variant, we replace the single-stage VLM inference with a two-stage process:

1. Stage 1 (Visual Captioning): The VLM generates a dense, generic description of the scene image.

2. Stage 2 (Linguistic Reasoning): This description is passed to a separate Large Language Model (Qwen3-30B [67]) which performs the final olfactory inference based on the text alone.

By decoupling visual perception from olfactory inference, we can test whether olfactory knowledge is more efectively extracted via direct visual grounding or through the high-level semantic priors encoded in LLMs. The prompts used for this two-stage pipeline are provided below:

## Stage 1 VLM Prompt:

Caption this image. Ignore the blue and yellow tool.

## Stage 2 LLM Prompt:

You are given a short textual description of an image. Based only on this description, provide four outputs:

1. ITEM (I): The main item/object that would be shown. Choose exactly ONE from this list: {items\_list}

2. OBJECT (O): A specific, open-set name for the exact object.

3. ENVIRONMENT (E): The general environment/setting. Choose exactly ONE from this list: {environments\_list}

4. CONTEXT (Ctx): A short, open-set phrase describing the background/setting (e.g. park, garage, kitchen, office, street, garden, workshop). Anything that describes where the scene is. 5. INFERRED SMELLS (S): Infer likely smells from the surrounding environment ONLY.

\*\*\* Focus on ’invisible’ scents that are likely in the air (e.g., traffic exhaust, humidity, air conditioning, distant greenery).

Respond in this exact format (one field per line):

ITEM: [exactly one from the item list]

OBJECT: [name]

ENVIRONMENT: [exactly one from the environment list]

CONTEXT: [one short open-set phrase]

SMELLS: [ambient smell 1], [ambient smell 2]

EXAMPLE:

ITEM: plants flower ornamental

OBJECT: white daffodil

ENVIRONMENT: Campus Outdoors

CONTEXT: a breezy concrete walkway adjacent to a freshly mowed lawn

SMELLS: fresh cut grass, concrete dust, distant vehicle exhaust

Description of the image: {caption}

![](images/1f3a94261ded96d0ca5fb6cdd85f6a4d018e7f06557c33770bf3899a1469bca2.jpg)  
(a) Environment Categories (7 classes)

Item Category Distribution (43 Classes)  
![](images/f08a5488216d04078e2f28b76cd49c30acbc1edc0825cb94838cef6c0502cee4.jpg)  
(b) Item Categories (43 classes)  
Fig. 7: Taxonomy distribution of the NYS benchmark. We visualize the label frequency for the two classification probing tasks: (a) the 7 Environment categories and (b) the 43 Item categories. Note the long-tail distribution in the Item classes, which challenges the model to learn robust categorical representations from imbalanced enose signals.

## B.2 Annotation Statistics

We analyze the distribution of the VLM-generated pseudo-labels used for our probing tasks. As shown in Figure 7, the taxonomy covers a broad spectrum of in-the-wild olfactory scenes.

## C Quantitative results

## C.1 Inverse Retrieval Tasks: I2S and T2S

To further validate the bidirectional consistency of our learned latent spaces, we evaluate SCENT on the inverse retrieval tasks: Image-to-Smell (I2S) and Text-to-Smell (T2S). These experiments assess the model’s ability to identify a specific smell fingerprint given a visual or textual query.

Baselines and Protocols. As with the forward tasks, we compare against a reproduced version of the NYS baseline [46]. For the T2S task, since the baseline lacks a native textual head, we implement the Image Bridge protocol described in Section 4.1 of the main paper: a query text is first mapped to its most similar image in the gallery via zero-shot CLIP [47] similarity, and this retrieved image is then used to query the olfactory database.

Analysis of Results. As shown in Table 6, SCENT outperforms the baseline in I2S retrieval, improving R@5 from 18.5 to 22.3. This confirms that the semantic information provided by our VLM-augmented language guidance (O+Ctx+S) during training helps the olfactory encoder learn features that are more efectively aligned with the visual manifold.

In the T2S task, we observe that the NYS (adapted) baseline achieves slightly higher recall (9.9 R@5) than our direct T2S approach (7.2 R@5). This result is expected due to the structural nature of the Image Bridge protocol. Specifically, the adapted baseline leverages the massive, web-scale prior of the frozen CLIP model to align text queries with visual candidates before performing the final I2S step. In contrast, our model performs direct native retrieval without ever accessing the image modality during the query process.

The fact that our model maintains competitive performance (22.6% R@20) while relying solely on the alignment between the e-nose signal and textual descriptors demonstrates the strength of our native olfactory-textual projection. While a visual bridge provides a shortcut for retrieval, our end-to-end alignment proves that the olfactory encoder is capable of capturing high-level semantic concepts purely through linguistic supervision.

## C.2 Classification

To further evaluate the richness of the learned olfactory representations, we perform a linear probing analysis on two classification tasks: Item and Environment. Since the original NYS dataset [46] provides raw images and sensor data without discrete categorical labels, we utilize a pretrained VLM (Qwen3VL-30B [67]) to generate a set of closed-set pseudo-labels for the entire dataset.

Table 6: Inverse retrieval performance on the NYS dataset. We evaluate the bidirectional consistency of our olfactory latent space via Image-to-Smell (I2S) and Text-to-Smell (T2S) tasks. For T2S, the NYS (adapted) baseline utilizes an Image Bridge protocol, leveraging CLIP’s pre-existing image-text alignment to rank candidates. In contrast, SCENT (Ours) performs direct retrieval using its native textual projection head. Our method outperforms the baseline in I2S and maintains competitive performance in T2S without requiring visual proxies during the query process.

<table><tr><td rowspan="2">Method</td><td colspan="3">Text Used</td><td colspan="4">I2S</td><td colspan="4">T2S</td></tr><tr><td>O</td><td>Ctx</td><td>S</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td></tr><tr><td>NYS (reproduced)</td><td></td><td></td><td></td><td>4.2</td><td>18.5</td><td>28.2</td><td>39.6</td><td></td><td>—</td><td></td><td></td></tr><tr><td>NYS (adapted)</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>—</td><td></td><td>2.1</td><td>9.9</td><td>16.7</td><td>26.0</td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>√</td><td>5.3</td><td>22.3</td><td>32.9</td><td>46.2</td><td>1.2</td><td>7.2</td><td>12.8</td><td>22.6</td></tr></table>

Label Generation. We prompt the VLM to classify each scene into one of the 43 item categories and 7 environment types defined in the taxonomy of the NYS benchmark. The specific prompts used for this labeling process are detailed in Section B.1. These VLM-generated labels serve as the ground truth for our classification experiments.

Linear Probing Protocol. We freeze the parameters of the smell encoder $f _ { S }$ to ensure that no further representational learning occurs during this task. We extract the penultimate representation $z ^ { S }$ (the encoder output prior to the modalityspecific projection heads) for all samples in the training and validation sets. A single linear layer is then trained as a probe for each task using a standard cross-entropy loss and the Adam optimizer. This protocol measures the degree to which item-specific and context-specific semantic information is linearly separable within the fixed olfactory latent space.

Results. We report Top-1 Accuracy on the held-out validation set. As shown in Table 7, we evaluate the categorical richness of the learned olfactory latent space through linear and non-linear (MLP) probing. Both SCENT and the NYS baseline significantly outperform the chance-level baseline, confirming that enose signals carry substantial discriminative information about both the target object and its surroundings.

When employing a linear probe, SCENT reaches 18.48% and 48.08% accuracy for Item and Environment classification, respectively. The introduction of a nonlinear MLP probe yields a performance boost across both tasks, with our method achieving scores of 25.43% (Item) and 56.84% (Environment). This suggests that our multimodal alignment, incorporating VLM-augmented language descriptors, produces a feature space that is rich in semantic content and resilient to the inherent noise of urban olfactory measurements.

Table 7: Probing olfactory representations. We evaluate the linear and nonlinear (MLP) separability of the frozen smell encoder f<sub>S</sub> across two classification tasks: Item Category (43 classes) and Environmental Context (7 classes). Both models utilize VLM-generated closed-set pseudo-labels as ground truth. While both methods significantly outperform the chance baseline, SCENT achieves superior performance in Environmental classification (56.84%), demonstrating that our multimodal alignment creates a latent space that better captures the global semantic context of the olfactory scene.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Probe</td><td>Item</td><td>Environment</td></tr><tr><td>Acc</td><td>Acc</td></tr><tr><td>Chance</td><td>—</td><td>2.33</td><td>14.29</td></tr><tr><td rowspan="2">NYS</td><td>Linear</td><td>18.16</td><td>50.11</td></tr><tr><td>MLP</td><td>26.71</td><td>55.88</td></tr><tr><td rowspan="2">Ours</td><td>Linear</td><td>18.48</td><td>48.08</td></tr><tr><td>MLP</td><td>25.43</td><td>56.84</td></tr></table>

## C.3 Performance by Scene Depth

Although SCENT does not use depth as an input, the NYS dataset provides co-registered depth measurements for every sample. We exploit this to probe whether retrieval performance is afected by the physical geometry of the captured scene. Specifically, we partition the validation set into enclosed and open environments based on the depth reading, and evaluate each subset against its own gallery.

As shown in Table 8, SCENT outperforms the NYS<sup>†</sup> baseline in both subsets across all tasks and metrics. The improvement is larger for open scenes, where SCENT achieves the best performance overall. We attribute this to richer visual context in open environments: a VLM observing a street scene or a park can infer a more diverse and precise set of ambient smells, producing stronger language supervision and better downstream smell–scene alignment. Enclosed scenes, by contrast, ofer less contextual diversity and may constrain the range of inferred ambient odors. Designing annotation strategies that remain efective in visually sparse, enclosed environments is a promising direction for future work.

## D Ablation Studies

## D.1 Scene Inference with Diferent Language Models

We investigate the impact of the semantic source on olfactory representation learning by comparing our primary VLM-driven pipeline (Ours (VLM)) with a two-stage decoupled pipeline (Ours (LLM)) utilizing Qwen3-30B for reasoning. As shown in Table 9, we observe two key trends:

Table 8: Performance by scene depth (enclosed vs. open). We split the validation set by depth into enclosed and open scenes and evaluate each subset against its own gallery. SCENT improves over the baseline in both subsets, with the largest gains on open scenes.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Subset</td><td colspan="2">S2I</td><td colspan="2">S2T</td><td colspan="2">S2IT</td></tr><tr><td>R@5</td><td>R@20</td><td>R@5</td><td>R@20</td><td>R@5</td><td>R@20</td></tr><tr><td rowspan="3">NYS $^{\dagger}$ </td><td>all</td><td>20.0</td><td>42.0</td><td>8.1</td><td>19.2</td><td>20.0</td><td>40.9</td></tr><tr><td>enclosed</td><td>19.4</td><td>41.2</td><td>6.9</td><td>19.0</td><td>19.4</td><td>40.9</td></tr><tr><td>open</td><td>20.0</td><td>42.5</td><td>9.2</td><td>19.1</td><td>20.0</td><td>40.6</td></tr><tr><td rowspan="3">SCENT (ours)</td><td>all</td><td>23.0</td><td>43.6</td><td>11.9</td><td>29.2</td><td>23.3</td><td>42.5</td></tr><tr><td>enclosed</td><td>20.7</td><td>42.7</td><td>10.8</td><td>29.0</td><td>21.4</td><td>41.8</td></tr><tr><td>open</td><td>24.9</td><td>44.2</td><td>13.0</td><td>29.1</td><td>24.7</td><td>42.9</td></tr></table>

Table 9: Ablation on Scene Inference with Diferent Language Models: Visual Grounding vs. Two-Stage LLM Pipeline. We compare the performance of SCENT when trained using semantics directly from a VLM (Qwen3-VL) versus those generated by an LLM (Qwen3-30B) based on visual captions. This comparison measures the trade-of between the direct visual grounding of a VLM and the common-sense priors of a large-scale instruction-tuned LLM.

<table><tr><td rowspan="2">Method</td><td colspan="3">Textual Info</td><td colspan="2">S2I</td><td colspan="2">S2T</td><td colspan="2">S2IT</td></tr><tr><td>O</td><td>Ctx</td><td>S</td><td>R@5</td><td>R@20</td><td>R@5</td><td>R@20</td><td>R@5</td><td>R@20</td></tr><tr><td rowspan="3">Ours (VLM)</td><td>√</td><td></td><td></td><td>22.1</td><td>42.4</td><td>8.0</td><td>21.5</td><td>22.0</td><td>42.6</td></tr><tr><td>√</td><td>√</td><td></td><td>21.8</td><td>45.2</td><td>10.7</td><td>28.1</td><td>21.3</td><td>45.7</td></tr><tr><td>√</td><td>√</td><td>√</td><td>23.0</td><td>43.6</td><td>11.9</td><td>29.2</td><td>23.3</td><td>42.5</td></tr><tr><td rowspan="3">Ours (LLM)</td><td>√</td><td></td><td></td><td>21.2</td><td>42.4</td><td>6.6</td><td>18.2</td><td>21.5</td><td>42.5</td></tr><tr><td>√</td><td>√</td><td></td><td>21.9</td><td>42.7</td><td>8.7</td><td>24.1</td><td>22.1</td><td>42.9</td></tr><tr><td>√</td><td>√</td><td>√</td><td>22.0</td><td>44.1</td><td>9.1</td><td>23.7</td><td>22.2</td><td>44.2</td></tr></table>

The Superiority of Visual Grounding. Across all retrieval tasks, SCENT with VLM-generated annotations consistently outperforms the LLM-based variant. For the most complete language setting (O+Ctx+S), the VLM pipeline achieves 29.2% R@20 in S2T retrieval, compared to 23.7% for the LLM pipeline. This gap suggests that when a VLM directly "sees" the scene, it captures nuanced environmental cues that are lost when the scene is first compressed into a generic text caption for an LLM. Visual grounding ensures that the inferred ambient smells (S) are spatially and contextually anchored to the specific instance, rather than being generalized "common sense" guesses.

Robustness of the Two-Stage Pipeline. Despite the performance gap, the LLM pipeline remains highly competitive, particularly in the S2I task (44.1% R@20). The fact that the LLM-based pipeline still outperforms the vision-only NYS baseline (42.0% R@20) demonstrates that the "olfactory common sense" encoded in large language models is a powerful prior. Even without direct visual access, the LLM is able to successfully infer a plausible olfactory manifold based solely on a textual description of the scene. However, for precise multimodal alignment, the end-to-end visual grounding provided by the VLM remains the optimal choice.

Table 10: Ablation of fusion strategies and modality weighting. We evaluate the sensitivity of joint-modality retrieval (S2IT) to diferent fusion paradigms and visual-textual weighting coeficients (a). Results are reported using the full semantic template $\left( \mathbf { O } + \mathbf { C } \mathbf { t x } + \mathbf { S } \right)$

<table><tr><td>Fusion</td><td>a</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td></tr><tr><td rowspan="3">Similarity-level</td><td>0.3</td><td>5.0</td><td>20.3</td><td>32.5</td><td>44.3</td></tr><tr><td>0.5</td><td>5.8</td><td>22.1</td><td>32.2</td><td>44.1</td></tr><tr><td>0.7</td><td>6.3</td><td>22.0</td><td>34.0</td><td>44.6</td></tr><tr><td rowspan="3">Latent-level</td><td>0.3</td><td>4.7</td><td>17.5</td><td>27.6</td><td>40.7</td></tr><tr><td>0.5</td><td>6.1</td><td>20.8</td><td>31.2</td><td>42.9</td></tr><tr><td>0.7</td><td>6.2</td><td>22.9</td><td>32.9</td><td>42.9</td></tr></table>

## D.2 Fusion Strategies for Joint-Modality Retrieval (S2IT)

To perform joint retrieval, we evaluate two distinct fusion paradigms to unify the image-aligned $\textstyle ( z _ { I } ^ { S } )$ and textual-aligned $\left( z _ { T } ^ { S } \right)$ smell embeddings.

1. Latent-level Fusion: We construct a single blended representation z˜ before computing similarity. This is our primary method:

$$
\tilde {z} _ {i} = \frac {\alpha z _ {I , i} ^ {S} + (1 - \alpha) z _ {T , i} ^ {S}}{\left\| \alpha z _ {I , i} ^ {S} + (1 - \alpha) z _ {T , i} ^ {S} \right\| _ {2}}
$$

The retrieval score is then calculated by comparing this unified vector to both targets:

$$
\mathrm{score} _ {i j} = \alpha \cdot \mathrm{sim} (\tilde {z} _ {i}, z _ {j} ^ {I}) + (1 - \alpha) \cdot \mathrm{sim} (\tilde {z} _ {i}, z _ {j} ^ {T})
$$

2. Similarity-level Fusion: A late-fusion approach where the joint score is a weighted combination of independent similarity scores:

$$
\mathrm{score} _ {i j} = \alpha \cdot \mathrm{sim} (z _ {I, i} ^ {S}, z _ {j} ^ {I}) + (1 - \alpha) \cdot \mathrm{sim} (z _ {T, i} ^ {S}, z _ {j} ^ {T})
$$

Latent-level fusion forces the model to find a single point in the latent manifold that satisfies both visual and linguistic constraints simultaneously. Unless otherwise specified, we use Latent-level fusion for all joint retrieval experiments, with the fusion weight α selected via grid search on the validation set.

Table 11: Ablation of two-stage training strategy. We compare standard joint training against a two-stage schedule that disables the S2T loss for the first half of training before switching it on at full weight. Results use the full semantic template $\left( \mathbf { O } + \mathbf { C } \mathbf { t x } + \mathbf { S } \right)$

<table><tr><td rowspan="2">Method</td><td rowspan="2">Two-Stage</td><td colspan="4">S2I</td><td colspan="4">S2T</td><td colspan="4">S2IT</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td></tr><tr><td>SCENT</td><td>√</td><td>5.0</td><td>19.4</td><td>30.2</td><td>41.2</td><td>2.4</td><td>9.7</td><td>16.6</td><td>25.1</td><td>5.6</td><td>18.9</td><td>30.4</td><td>41.5</td></tr><tr><td>SCENT (ours)</td><td>X</td><td>6.0</td><td>23.0</td><td>33.5</td><td>43.6</td><td>3.3</td><td>11.9</td><td>19.8</td><td>29.2</td><td>6.8</td><td>23.3</td><td>32.7</td><td>42.5</td></tr></table>

Ablation. We evaluate the sensitivity of our joint retrieval (S2IT) to diferent fusion paradigms and modality weightings (a). As shown in Table 10, both Similarity-level and Latent-level fusion exhibit a similar upward trend as the visual weighting a increases. While Similarity-level fusion provides slightly better performance at lower visual weights, Latent-level fusion demonstrates a more significant performance gain as the modalities are balanced, reaching an R@5 of 22.9. This suggests that while Similarity fusion acts as a "greedy" ensemble of independent heads, our chosen Latent-level fusion succeeds in forcing the olfactory embedding into a singular, semantically coherent point in the CLIP manifold that satisfies both visual and textual constraints simultaneously.

## D.3 Two-Stage Training Strategy

We investigate whether warming up smell-image alignment before introducing textual supervision benefits training. In our two-stage schedule, Phase 1 trains with only the S2I loss active. At the midpoint, the S2T loss is switched on and training continues to completion with both losses active.

The two-stage schedule falls short of joint training across all tasks and metrics, as shown in Table 11. The S2T gap is the most pronounced (−2.2 R@5), but S2I and S2IT also decline. This outcome suggests that the olfactory encoder benefits from simultaneous co-alignment pressure from both objectives from the very start of training. When Phase 1 optimizes exclusively for the visual manifold, the shared encoder backbone develops representations biased toward visual features. Introducing the textual loss mid-training then forces the textual projection head to adapt to an already-committed encoder, limiting its ability to discover a complementary textual alignment. Joint training, by contrast, allows the encoder to continuously balance both objectives, yielding stronger olfactory representations.

## D.4 Annotation Granularity

We test whether eliciting more inferred smells per image improves retrieval. Using the same VLM, we re-annotate the dataset with a prompt that yields a higher number of ambient smell descriptors per image rather than our default prompt. As shown in Table 12, performance degrades consistently across all tasks. We attribute this to annotation quality collapsing past the VLM’s natural saturation point: once the most salient, visually-grounded smells have been listed, additional entries are increasingly speculative. Training against these low-confidence descriptors introduces label noise.

Table 12: Ablation on smell annotation granularity. We compare our default prompt against a high-granularity variant that elicits a larger number of inferred ambient smells per image using the same VLM. Both variants use the full O + Ctx + S template and identical architecture.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Increased Inferred Smells</td><td colspan="4">S2I</td><td colspan="4">S2T</td><td colspan="4">S2IT</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td></tr><tr><td>SCENT</td><td>√</td><td>4.9</td><td>19.7</td><td>28.6</td><td>41.1</td><td>2.4</td><td>7.1</td><td>12.4</td><td>20.1</td><td>4.9</td><td>19.9</td><td>29.2</td><td>40.3</td></tr><tr><td>SCENT (ours)</td><td>✗</td><td>6.0</td><td>23.0</td><td>33.5</td><td>43.6</td><td>3.3</td><td>11.9</td><td>19.8</td><td>29.2</td><td>6.8</td><td>23.3</td><td>32.7</td><td>42.5</td></tr></table>

## E Qualitative Results

We provide extended qualitative results for the Smell-to-Image-Text (S2IT) retrieval task in Figure 8. Each row displays the query olfactory signal alongside the top-5 retrieved (Image, Text) pairs.

Semantic Consistency and Error Analysis. In several instances (e.g., Row 1: Pizza, Row 4: Library Books), SCENT achieves perfect or near-perfect retrieval, correctly identifying both the object and the environment. Notably, the textual descriptors for Row 4 reveal that the model distinguishes between specific attributes like "aged paper" and "faint leather binding".

Ambient Environment Overlap. The textual descriptors explain many of the "near-misses." In Row 2 (Plastic Cup), while the Rank 2 result is a cardboard box, the text reveals a shared environmental context: "high-rise apartment balcony/window with a city view" and shared ambient smells of "city air pollution" and "trafic exhaust." This confirms that even when the primary object difers, the model successfully identifies the broader olfactory scene, aligning the e-nose signal with the background atmosphere of the environment.

![](images/cb33f2c7d9857e143ac6569bf0fc9c554786fb361708874ad22cfeff7a63d52a.jpg)  
Fig. 8: Extended S2IT Qualitative Results with Semantic Textual Descriptors. We visualize the top-5 retrieved (Image, Text) pairs for a query olfactory signal (orange). The cyan border indicates the Rank-1 result. The accompanying text, decomposed into Object (O), Context (Ctx), and Inferred Smells (S), demonstrates that SCENT aligns olfactory signals with complex scenes.