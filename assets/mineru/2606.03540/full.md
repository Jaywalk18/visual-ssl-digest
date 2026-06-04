# Attend to Anything: Foundation Model for Unified Human Attention Modeling

Wenzhuo Zhao 1 Ronghao Xian 1 Keren Fu 1 2 Qijun Zhao 1 2

# Abstract

Existing human attention (saliency) modeling methods persist as highly fragmented across modalities, scenes, and task formulations. Consequently, even with increasing model capacity and data scale, current models predominantly remain scene-dependent and task-specific, failing to practically generalize in real-world applications. To address the fundamental limitations, we present the Attend to Anything Model (AAM), a multi-modal foundation model that unifies attention modeling across various image, video, and audio-visual tasks and scenes. AAM reformulates attention as a cognitive entailment relationship organized in a general-to-specific hierarchy, implemented through language prompts with hierarchical embeddings in hyperbolic space. Furthermore, to unify static image and dynamic video attention, we adopt a fluid-dynamics perspective, formulating video-frame attention as a diffusive temporal evolution governed by the Fokker–Planck equation. Extensive experiments on 16 benchmarks demonstrate that AAM consistently outperforms state-of-the-art methods by an average of 6% across various scenarios, while achieving approximately a 4× speedup in video inference. Overall, these results demonstrate that AAM provides a principled foundation for future research on attention and saliency-related tasks. The dataset and code will be available at https://github. com/wz-zhao/Attend-to-Anything.

# 1. Introduction

Human visual attention modeling (saliency prediction) aims to predict where humans look in visual stimuli and constitutes a fundamental component of multimedia understanding (Mishra et al., 2021), marketing analytics (Jiang et al.,

XXXXXXXXXXXXXXXXXXXXXXX.

Human Attention Modeling   
![](images/48a781a31fe282209225174fb3f4b28c446703001fb5155604943fee350cb84b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Guided by speaking faces."] --> B["Condition"]
    B --> C["AAM (Ours)"]
    C --> D["Semantic Information"]
    D --> E["General Attention"]
    E --> F["WebPage"]
    E --> G["Movies"]
    E --> H["Music"]
    E --> I["Talking"]
    E --> J["Natural"]
    E --> K["Music"]
    E --> L["Natural"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#ffc,stroke:#333
```
</details>

Figure 1. The Attend to Anything Model (AAM) unifies fragmented human attention tasks into a coherent foundation. By learning a general-to-specific hierarchy in hyperbolic space, AAM effectively models attention patterns across images, videos, and audio-visual scenarios, spanning from low-level visual saliency to complex, semantic-driven dynamic interactions.

2023), and robotic perception (Samani et al., 2023). Despite decades of progress across image, video, and audiovisual settings, existing attention models remain highly fragmented: different modalities, scenes, and task conditions are typically studied as isolated problems with dedicated architectures and training protocols (Zhou et al., 2023; Tang et al., 2025). This fragmentation stands in stark contrast to recent advances in computer vision, where unified foundation models have demonstrated strong generalization across data domains and tasks. To date, attention modeling still lacks a unified foundation model capable of generalizing across modalities, scenes, and tasks, which severely limits its generalization ability and real-world applicability. The persistent gap in cross-dataset generalization, despite increasing model capacity and data scale (Kummerer et al.¨ , 2025), points to a deeper issue: the current formulation of attention as disjoint tasks fails to capture its underlying unified cognitive process. The absence of a unified foundation model in attention modeling can be traced to two fundamental challenges rooted in current problem formulations:

I) Cross-Scene generalization. From a neuroscientific perspective (Rao & Ballard, 1999), human visual attention is hierarchically modulated by cognitive context and tasks, a property widely acknowledged in attention modeling research (Yarbus, 2013; Lou et al., 2022). However, existing methods (Droste et al., 2020; Chen et al., 2023a) predominantly attribute condition-dependent variations to datasetspecific statistical biases, thereby collapsing the hierarchical modulation (from general to specific) into dataset-specific differences. Under this paradigm, attention models are typically trained and evaluated on individual datasets (Xie et al., 2024; Jin et al., 2025), leading to a performance plateau on existing benchmarks (Kummerer & Bethge ¨ , 2023). Recent studies further reveal a substantial performance drop (around 40%) when models trained on one dataset are applied to another (Kummerer et al. ¨ , 2025), a discrepancy that cannot be resolved simply by scaling training data.

Although several works attempt to alleviate this issue via joint training across multiple datasets (Droste et al., 2020; Hosseini et al., 2025b), they rely on dataset-specific priors or isolated parameters (e.g., normalization statistics or Gaussian maps), fundamentally limiting their ability to generalize beyond the statistics of observed datasets.

II) Cross-Task modeling. A second challenge arises from the heterogeneous formulation of attention modeling across modalities and temporal scales. Image and video attention have long been treated as distinct tasks, despite arising from the same underlying attentional mechanisms. Existing video models operate on fixed-window clips and encode temporal information through optical flow (Lai et al., 2019) or spatiotemporal convolutions and transformers (Zhou et al., 2023), only outputting the final frame. Such formulations impose task boundaries, restrict frame-wise inference, and incur substantial computational overhead, limiting modeling flexibility and inference efficiency.

To address these challenges, we propose the Attend to Anything Model (AAM), the first unified multi-modal foundation model for attention modeling across images, videos, and audio-visual scenarios. ❶ To tackle cross-scene generalization, we structure the cognitive evolution from general priors to specific tasks as an asymmetric hierarchical implication (Fig. 1). Since hyperbolic space naturally supports hierarchical structures (Li et al., 2025a; Liu et al., 2025), we model attention differences as entailment relations induced by text prompts in hyperbolic space. Notably, compared to existing methods based on parameter isolation, AAM introduces a cognitively motivated paradigm for modeling attention variation, offering theoretical and empirical insights for the future development of holistic visual perception systems. ❷ To tackle cross-task modeling, we introduce the Fokker–Planck Dynamics Module (FPD), which models video attention as the fluid dynamics of static attention over a spatiotemporal manifold. FPD models attention as

![](images/8fe228ed9d71e57c234c11f54fee3904ef4d9bcd535463061f4a3e8ebefbf161.jpg)

<details>
<summary>text_image</summary>

Human
Observers
15 Research Progress
Years (2009-2023)
300+
AUDIO-TASGAL
IMAGE
VIDEO
1.75M Eye Fixation
Maps
Dataset-Level
Condition Prompts
[Stimulus Type] [scene/domain] [task setting] [attention bias]
Top-Down
Semantic Modulation
Bottom-Up
Dataset-Level Cognitive Prompts
Dynamic audio-visual content under free-viewing,
which is guided by sound sources and speaking faces.
Static natural photographs under free-viewing, which
is guided by people, readable text, animals, vehicles.
Dynamic movie action sequences under task-driven
viewing, which is guided by movie cinematography.
...
Static user webpage screenshot under task-driven
viewing, which is guided by layout structure and text.
</details>

Figure 2. Attention-1.75M surpasses existing datasets in both scale and diversity, and is equipped with dataset-level texts derived from the dataset acquisition protocols (Appendix B for details).

the continuous transport governed by the Fokker–Planck equation, decomposing the evolution process into advection and diffusion. This physics-informed formulation unifies static and dynamic attention, enabling efficient frame-wise inference. ❸ To support the foundation model training, we curate Attention-1.75M, a standardized corpus unifying over 1.75M fixation instances across images, videos, and audio-visual scenarios equipped with dataset-level texts. Our principal contributions are summarized as follows:

❶ Problem Identification. We identify a fundamental mismatch between unified human attentional mechanisms and existing fragmented attention modeling formulations, revealing that the absence of a unified foundation model severely limits generalization and real-world applicability.   
❷ Paradigm Reformulation. We propose Attend to Anything Model (AAM), the first unified multi-modal foundation model for attention modeling across image, video, and audio-visual scenarios. AAM achieves cross-scene generalization via hierarchical entailment in hyperbolic space and resolves static–dynamic task incompatibility through Fokker–Planck physical temporal dynamics, unifying image and video attention in a continuous framework.   
❸ Experimental Validation. We evaluate AAM on 16 benchmarks, demonstrating consistent state-of-the-art (SOTA) performance with an average improvement of 6% and approximately a 4× speedup in video inference, supported by a large-scale training corpus of 1.75M instances.   
❹ Conceptual Insights. Extensive inductive experiments demonstrate that AAM introduces a cognitively aligned paradigm for modeling attentional differences, contributing theoretical understanding and empirical evidence to the future development of holistic visual perception systems.

# 2. Related Works

Evolution of Attention Modeling. The field has evolved from heuristic contrast features (Riche et al., 2013) to deep representation learning across three tracks. Image modeling employs CNN and Transformer backbones (Huang et al., 2015; Lou et al., 2022) to learn distribution mappings, but typically ignores the hierarchical nature of cognitive tasks. Temporal modeling extends these to videos via motion cues (Jain et al., 2021) or spatiotemporal encoders such as 3D-CNNs and Video Swin Transformers (Bellitto et al., 2021; Jin et al., 2025), which are often constrained by fixed-window and high complexity. Audio-visual modeling integrates auditory streams through sound localization or modality-specific fusion branches (Aytar et al., 2016; Xiong et al., 2024). Despite recent joint-training attempts (Droste et al., 2020; Hosseini et al., 2025b; Kummerer et al.¨ , 2025), existing methods still rely on parameter isolation or specialized modules to handle heterogeneous data. In contrast, AAM provides a unified paradigm by modeling attention in hyperbolic space and bridging static-dynamic transitions through a continuous fluid-dynamics formulation.

Biological and Cognitive Foundations of Attention. Existing studies (Jiang et al., 2024) in neuroscience and cognitive science provide important theoretical support for modeling attention in a cognitively aligned manner. First, the hierarchical organization of the visual cortex, from low-level edge responses in V1 to high-level semantic representations in IT, suggests that visual perception is inherently structured across multiple levels of abstraction. Prior work has shown that visual hypercolumns can be naturally modeled with hyperbolic geometry (Chossat & Faugeras, 2009), and that spiking activity patterns in V1/V2 exhibit intrinsic hyperbolic structure (Guidolin et al., 2022). These findings motivate the use of hyperbolic space to capture the hierarchical attention. Second, the drift-diffusion model (DDM) has been widely used to explain perceptual decision-making and saccadic latency in cognitive science (Ratcliff, 1978; Bogacz et al., 2006). Its macroscopic dynamics are closely related to the Fokker–Planck equation, which describes the evolution of probability density over time. This connection provides a biologically grounded interpretation of attention shifts: drift captures top-down task-driven bias, diffusion reflects bottom-up visual exploration, and the concentration of probability mass corresponds to decision-boundary crossing (Shinn et al., 2020). Together, these biological and cognitive studies motivate AAM as a formulation that bridges static geometry and temporal attention dynamics.

# 3. Methodology

# 3.1. Overview of AAM

In this section, we present an overview of the proposed AAM, which represents human attention as a shared latent process across modalities, scenes, and time by modeling hierarchical semantic specialization and temporal dynamics, as shown in Fig. 3. Visual inputs are encoded by a frozen self-supervised backbone (DINOv3 (Simeoni et al.´ , 2025)) with LoRA for adaptation to attention modeling (see Appendix A for detailed hyperparameters and configurations). Text prompts and audio signals are encoded using frozen CLIP (Radford et al., 2021) and Wav2CLIP (Wu et al., 2022) encoders, respectively, with audio mapped into the visual semantic space. Audio-visual fusion is performed through a relevance-gated cross-attention mechanism, ensuring that audio cues contribute only when semantically aligned (Appendix E.2). For video inputs, frame-wise attention representations are refined by a Fokker–Planck Dynamics (FPD) module that models attention evolution over a spatiotemporal manifold (Sec. 3.4). Visual and textual representations are lifted into hyperbolic space via hierarchical entailment learning for explicit hierarchy enforcement (Sec. 3.2). Finally, a geometry-aware hyperbolic decoder projects structured representations back to Euclidean space to generate spatial attention maps (Sec. 3.3).

# 3.2. Hierarchical Human Attention Modeling in Hyperbolic Space

# 3.2.1. PRELIMINARIES

Hyperbolic geometry, a non-Euclidean geometry characterized by constant negative curvature and exponential volume growth, naturally encodes the degree of semantic specialization through the distance from the origin and represents the scope of refinement via angular regions. This geometric structure makes it an ideal choice for learning representations of data with inherent hierarchical structures (Krioukov et al., 2010; Sarkar, 2011; Nickel & Kiela, 2017). Specifically, the Lorentz model $\mathbb { L } _ { \kappa } ^ { n }$ with curvature $- \kappa \in \mathbb { R }$ is defined as an n-dimensional manifold represented as the upper sheet of a two-sheeted hyperboloid in (n + 1)-dimensional Minkowski spacetime, which is described as

$$
\mathbb {L} _ {\kappa} ^ {n} = \left\{\mathbf {z} \in \mathbb {R} ^ {n + 1}: \langle \mathbf {z}, \mathbf {z} \rangle_ {\mathbb {L}} = - \frac {1}{\kappa}, \mathbf {z} _ {0} = \sqrt {\frac {1}{\kappa} + \| \tilde {\mathbf {z}} \| _ {2} ^ {2}} \right\}, \tag {1}
$$

where $\langle . , . \rangle _ { \mathbb { I } }$ denotes the Lorentzian inner product for ${ \bf z } , { \bf z } ^ { \prime } \in $ $\mathbb { L } _ { \kappa } ^ { n }$ , defined as:

$$
\langle \mathbf {z}, \mathbf {z} ^ {\prime} \rangle_ {\mathbb {L}} = - \mathbf {z} _ {0} \mathbf {z} _ {0} ^ {\prime} + \langle \tilde {\mathbf {z}}, \tilde {\mathbf {z}} ^ {\prime} \rangle_ {\mathbb {E}}, \tag {2}
$$

with $\langle . , . \rangle _ { \mathbb { E } }$ denoting the Euclidean inner product. For each vector $\mathbf { z } ,$ the first dimension is taken as the time-axis, denoted $\mathbf { z } _ { 0 } .$ , and the remaining n dimensions as the spatialcoordinates, denoted $\tilde { \textbf { z } } \in \mathbb { R } ^ { n }$ . The Lorentzian distance between two points in $\mathbb { L } _ { \kappa } ^ { n }$ , is the length of the shortest path (geodesic), which can be computed as:

$$
d _ {\mathbb {L}} \left(\mathbf {z}, \mathbf {z} ^ {\prime}\right) = \sqrt {\frac {1}{\kappa}} \cosh^ {- 1} \left(- \kappa \langle \mathbf {z}, \mathbf {z} ^ {\prime} \rangle_ {\mathbb {L}}\right), \quad \mathbf {z}, \mathbf {z} ^ {\prime} \in \mathbb {L} _ {\kappa} ^ {n}, \tag {3}
$$

which inducing Lorentzian norm $\| \mathbf { z } \| _ { \mathbb { L } } = \langle \mathbf { z } , \mathbf { z } \rangle _ { \mathbb { L } }$ . Based on this geometry, any vector $\mathbf { v } \in T _ { \mathbf { z } } \mathbb { L } _ { \kappa } ^ { n }$ in the tangent space

![](images/347c125a1151283bf333f33603da1e7947253a7f0a5989bb5833f07e2d0152db.jpg)  
Figure 3. (a) Overview of AAM. (b) The general-specific attention ordering between $( \mathbf { z } _ { \mathrm { a n c } } , \mathbf { z } _ { \mathrm { t x t } } ) , ( \mathbf { z } _ { \mathrm { t x t } } , \mathbf { z } _ { \mathrm { i m g } } )$ is enforced in hyperbolic space using entailment cones. The external angle ϕ of a specific condition $\mathbf { \sigma } ( \mathbf { z } _ { \mathrm { t x t } } )$ is pushed to be within the aperture threshold ηω of the general attention $( \mathbf { z } _ { \mathrm { a n c } } ) . ~ ( \mathrm { c } )$ The Fokker–Planck Dynamics (FPD) Modeling illustrates the drift, diffusion, and correction processes used to model the transition of attention over the video frame axis.

Text Encodercan be projected onto the hyperboloid using the exponential map (Khrulkov et al., 2020):

$$
\exp_ {\mathbf {z}} ^ {\kappa} (\mathbf {v}) = \cosh \left(\sqrt {\kappa} \| \mathbf {v} \| _ {\mathbb {L}}\right) \mathbf {z} + \frac {\sinh \left(\sqrt {\kappa} \| \mathbf {v} \| _ {\mathbb {L}}\right)}{\sqrt {\kappa} \| \mathbf {v} \| _ {\mathbb {L}}} \mathbf {v}. \tag {4}
$$

# 3.2.2. HIERARCHICAL ENTAILMENT LEARNING

We model human visual attention as a hierarchical process. Specifically, attention is refined from general attention to specific visual attention distribution as conditions are imposed. To formalize these hierarchical entailment relations, we introduce a partial order relation in hyperbolic space:

$$
\mathbf {z} _ {\mathrm{img}} \preceq \mathbf {z} _ {\mathrm{txt}} \preceq \mathbf {z} _ {\mathrm{anc}}, \tag {5}
$$

where $\mathbf { z } _ { \mathrm { a n c } }$ and $\mathbf { z } _ { \mathrm { t x t } }$ denote the general attention and text prompts, respectively, with $\mathbf { z } _ { \mathrm { i m g } }$ serving as their specific instantiation within the visual input. We map the text embedding $\mathbf { v _ { \mathrm { t x t } } }$ generated by the text encoder into the Lorentz manifold via the exponential map defined at the origin:

$$
\mathbf {z} _ {\mathrm{txt}} = \exp_ {\mathbf {o}} ^ {\kappa} (\mathbf {v} _ {\mathrm{txt}}) \in \mathbb {L} _ {\kappa} ^ {n}. \tag {6}
$$

Similarly, we map the visual feature embedding $\mathbf { v } _ { \mathrm { i m g } }$ into the Lorentz manifold to obtain its hyperbolic representation $\mathbf { z } _ { \mathrm { i m g } } ,$ , and introduce a learnable general attention anchor $\mathbf { z } _ { \mathrm { a n c } }$ .

Based on these hyperbolic representations, we employ hyperbolic entailment cones (Ganea et al., 2018; Pal et al., 2025) to transform hierarchical attention entailment relations into optimizable geometric constraints. As illustrated in Fig. 3 (b), entailment cones define a region $\Re _ { \mathbf { z } _ { \mathrm { a n c } } }$ for

every possible point $\mathbf { z } _ { \mathrm { a n c } }$ in the space such that all points $\mathbf { z } _ { \mathrm { t x t } } \in \Re _ { \mathbf { z } _ { \mathrm { a n c } } }$ are semantically linked to $\mathbf { z } _ { \mathrm { a n c } }$ as its child concepts. As such, points in $\Re _ { \mathbf { z } _ { \mathrm { a n c } } }$ are expected to contain specific condition for the general concept $( \mathbf { z } _ { \mathrm { t x t } } \preceq \mathbf { z } _ { \mathrm { a n c } } )$ . The half-aperture of these conical regions is formulated by (Le et al., 2019; Desai et al., 2023) as: $\begin{array} { r } { \omega ( \mathbf { z } ) = \sin ^ { - 1 } \left( \frac { 2 K } { \sqrt { \kappa } \| \tilde { \mathbf { z } } \| } \right) } \end{array}$ .

To learn partial orders in the Lorentz space, we employ a thresholded entailment loss based on angular residuals (Le et al., 2019; Desai et al., 2023):

$$
\mathcal {L} _ {\text { ent }} ^ {*} (\mathbf {z} _ {\text { anc }}, \mathbf {z} _ {\text { txt }}) = \max (0, \phi (\mathbf {z} _ {\text { anc }}, \mathbf {z} _ {\text { txt }}) - \eta \omega (\mathbf {z} _ {\text { anc }})), \tag {7}
$$

where $\eta$ is a threshold scaling factor used to adjust the tightness of the entailment constraint, $\phi ( \mathbf { z } , \mathbf { z } ^ { \prime } )$ denotes the exterior angle of the child node deviating from the boundary of the parent entailment cone:

$$
\phi (\mathbf {z}, \mathbf {z} ^ {\prime}) = \cos^ {- 1} \left(\frac {z _ {0} + z _ {0} ^ {\prime} \kappa \langle \mathbf {z} , \mathbf {z} ^ {\prime} \rangle_ {\mathbb {L}}}{\| \tilde {\mathbf {z}} ^ {\prime} \| \sqrt {(\kappa \langle \mathbf {z} , \mathbf {z} ^ {\prime} \rangle_ {\mathbb {L}}) ^ {2} - 1}}\right). \tag {8}
$$

Hence, the hierarchical attention entailment (HAE) loss of human attention would comprise both anchor-text conditional entailments and text-image entailments as:

$$
\mathcal {L} _ {\mathrm{HAE}} = \mathcal {L} _ {\text { ent }} ^ {*} (\mathbf {z} _ {\text { anc }}, \mathbf {z} _ {\text { txt }}) + \lambda \mathcal {L} _ {\text { ent }} ^ {*} (\mathbf {z} _ {\text { txt }}, \mathbf {z} _ {\text { img }}), \tag {9}
$$

The final loss $\mathcal { L } _ { \mathrm { t o t a l } }$ (Droste et al., 2020) is composed of task and HAE loss:

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {\mathrm{KLD}} - \mathcal {L} _ {\mathrm{CC}} - \mathcal {L} _ {\mathrm{SIM}} + \mathcal {L} _ {\mathrm{HAE}}. \tag {10}
$$

$\mathcal { L } _ { \mathrm { K L D } } , \mathcal { L } _ { \mathrm { C C } } , \mathcal { L } _ { \mathrm { S I M } }$ represent the task losses for attention modeling, as detailed in Appendix A.1

# 3.3. Decoding Attention from Hyperbolic Manifolds

Building on the hierarchical entailment constraints regarding text specialization depth and cone positioning, the hyperbolic decoder employs scale modulation and spatial focusing to map hyperbolic geometry into Euclidean attention. The visual features $\mathbf { X } \in \mathbb { R } ^ { C \times H \times W }$ are thus adaptively modulated by ${ \bf z } _ { \mathrm { i m g } } , { \bf z } _ { \mathrm { t x t } }$ (Architecture details in Appendix 1).

❶ Specialization depth-driven scale modulation. Given that geodesic distance in hyperbolic space characterizes semantic specialization, we define the specialization depth of the condition as $r _ { \mathrm { t x t } } = d _ { \mathbb { L } } ( \mathbf { z } _ { \mathrm { t x t } } , \mathbf { o } )$ , which regulates the relative focus of the decoder between global structure and fine-grained local details. We introduce a set of operators $\{ S _ { k } \} _ { k = 1 } ^ { K }$ with K levels alongside anchors $\pmb { \mu _ { k } } \in \mathbb { L } _ { \kappa } ^ { n }$ to perform conditional weighted fusion of multi-scale features. The scale weights are determined by the hyperbolic distances between the text condition and scale anchor:

$$
w _ {k} = \operatorname{softmax} _ {k} \left(- d _ {\mathbb {L}} \left(\mathbf {z} _ {\mathrm{txt}}, \boldsymbol {\mu} _ {k}\right)\right). \tag {11}
$$

The overall modulation intensity is governed by a monotonic function of the specialization depth, $\begin{array} { r l } { \alpha ( r _ { \mathrm { t x t } } ) } & { { } = } \end{array}$ sof $\mathrm { \ p l u s } ( r _ { \mathrm { t x t } } )$ , while the final scale response is computed as a weighted combination of the operators:

$$
\mathbf {X} _ {s} = \sum_ {k = 1} ^ {K} w _ {k}   \mathcal {S} _ {k} (\mathbf {X}). \tag {12}
$$

❷ Relative geodesic direction-driven spatial focusing. To characterize the relative semantic relationship between the condition and the visual instance, we define the relative geodesic direction in the tangent space at the origin:

$$
\Delta = \log_ {\mathbf {o}} ^ {\kappa} (\mathbf {z} _ {\mathrm{img}}) - \log_ {\mathbf {o}} ^ {\kappa} (\mathbf {z} _ {\mathrm{txt}}). \tag {13}
$$

∆ captures the semantic deviation direction of the visual instance relative to the condition center, thereby providing semantic guidance for pixel-level spatial focusing. The aperture of the entailment cone corresponding to the text condition, denoted as $\omega ( \mathbf { z } _ { \mathrm { t x t } } )$ , reflects its degree of semantic generalization and regulates the intensity of spatial focusing. We define the focusing temperature as $\beta = \beta _ { 0 } \mathopen { } \mathclose \bgroup \left( \omega \mathopen { } \mathclose \bgroup \left( \mathbf { z } _ { \mathrm { t x t } } \aftergroup \egroup \right) + \varepsilon \aftergroup \egroup \right)$ , where $\beta _ { 0 }$ is a temperature scaling hyperparameter. A larger cone aperture corresponds to a more generalized semantic condition, inducing a broader spatial attention pattern. Let $\mathbf { u } _ { i , j }$ denote the channel-wise feature vector of $\mathbf { X }$ at position $( i , j )$ ; the spatial weight is defined by its consistency with the relative geodesic direction $\Delta \colon$

$$
m _ {i, j} = \operatorname{softmax} _ {i, j} \left(\frac {\langle \mathbf {u} _ {i , j} , \Delta \rangle}{\| \mathbf {u} _ {i , j} \| \| \Delta \| + \varepsilon} \cdot \frac {1}{\beta}\right). \tag {14}
$$

❸ Joint decoding. Scale selection and spatial focusing are integrated via a unified residual formulation:

$$
\mathbf {X} ^ {\prime} = \mathbf {X} + \alpha (r _ {\mathrm{txt}}) \left(\mathbf {M} \odot \mathbf {X} _ {s}\right), \tag {15}
$$

where $\mathbf { M } = \{ m _ { i , j } \}$ denotes the attention map induced by the relative geodesic direction. This residual fusion ensures that the hyperbolic entailment structure is consistently preserved within the pixel-wise attention distribution.

# 3.4. Fokker–Planck Dynamics (FPD) Modeling

For video sequences of arbitrary length, let $u _ { t } ^ { \mathrm { o b s } }$ denote the attention distribution derived from the visual encoder output $\mathbf { v } _ { \mathrm { i m g } } .$ , defined on the discrete spatial domain $\Omega =$ $\{ 1 , \ldots , H \bar { \} } \times \{ 1 , \ldots , W \}$ and subject to a normalization constraint. We introduce the Fokker–Planck (FP) dynamics equation to model the temporal evolution of attention:

$$
\frac {\partial u}{\partial t} = \underbrace {- \nabla_ {\tau} \cdot (\mathbf {v} u)} _ {\mathcal {A} _ {\text { drift }}} + \underbrace {\nabla_ {\tau} \cdot (D \nabla_ {\tau} u)} _ {\mathcal {A} _ {\text { diffusion }}} + \underbrace {\lambda (u ^ {\mathrm{obs}} - u)} _ {\mathcal {A} _ {\text { correction }}}, \tag {16}
$$

where the operators $\nabla _ { \tau }$ are defined along the temporal axis, characterizing attention drifting, smoothing, and correction, respectively. The parameter λ balances the dynamic prediction with the initial information. We employ the Lie–Trotter operator splitting scheme to discretize the FP dynamics into sequential sub-operator updates within a time step ∆t:

$$
u (t + \Delta t) \approx (\mathcal {S} _ {\text { drift }} \circ \mathcal {S} _ {\text { diffusion }} \circ \mathcal {S} _ {\text { correction }} \circ \mathcal {P} _ {\text { proj }}) [ u (t) ]. \tag {17}
$$

We denote by $u _ { t } ^ { ( k ) }$ the intermediate state after applying the k-th sub-operator, with u(0t $u _ { t } ^ { ( 0 ) } \equiv u _ { t }$ .

# 3.4.1. DRIFT EVOLUTION OPERATOR: SDRIFT

The drift equation is given by $\begin{array} { r } { \frac { \partial u } { \partial t } = - \nabla _ { \tau } \cdot ( \mathbf { v } u ) } \end{array}$ . To simulate this physical process within a discrete feature space, we utilize bidirectional temporal self-attention to parameterize the drift propagator $A _ { x }$ . We define the discrete transition kernel $A _ { x } ( t \gets t ^ { \prime } )$ from source time $t ^ { \prime }$ to target time t as:

$$
A _ {x} ^ {(h)} (t \leftarrow t ^ {\prime}) = \frac {\exp \left(\langle q _ {t} ^ {(h)} (x) , k _ {t ^ {\prime}} ^ {(h)} (x) \rangle / (\sqrt {d} \beta))\right)}{\sum_ {\tau^ {\prime} = 1} ^ {T} \exp \left(\langle q _ {t} ^ {(h)} (x) , k _ {\tau^ {\prime}} ^ {(h)} (x) \rangle / (\sqrt {d} \beta)\right)}. \tag {18}
$$

Here, the transition kernel $A _ { x }$ is normalized along the temporal axis to facilitate cross-time information aggregation, constituting a Markov transition (Detailed specifications are provided in Appendix D.2). Following the Lagrangian transport, $\tilde { u } _ { t }$ denotes the aggregated state derived from information integration across the full temporal domain:

$$
\tilde {u} _ {t} (x) = \sum_ {t ^ {\prime} = 1} ^ {T} A _ {x} (t \leftarrow t ^ {\prime}) u _ {t ^ {\prime}} (x). \tag {19}
$$

Subsequently, the drift operator $ { S _ { \mathrm { d r i f t } } }$ adopts a residual Euler update scheme to evolve the state $u _ { t } ^ { ( 0 ) }$ into the first-stage intermediate state $u _ { t } ^ { ( 1 ) }$ :

$$
u _ {t} ^ {(1)} (x) = u _ {t} ^ {(0)} (x) + \Delta t \cdot \big (\tilde {u} _ {t} (x) - u _ {t} ^ {(0)} (x) \big). \tag {20}
$$

The full-temporal attention mechanism enables cross-time information backflow, effectively mitigating the issue of insufficient motion cues in the early stages.

# 3.4.2. DIFFUSION OPERATOR: SDIFFUSION

To regularize the high-frequency noise introduced during the drift process, we employ a second-order central finite difference approximation based on the intermediate state $u _ { t } ^ { ( 1 ) }$ to compute the temporal diffusion term:

$$
u _ {t} ^ {(2)} (x) = u _ {t} ^ {(1)} (x) + \nu_ {t} (x) \Delta t \frac {u _ {t - 1} ^ {(1)} (x) - 2 u _ {t} ^ {(1)} (x) + u _ {t + 1} ^ {(1)} (x)}{(\Delta t) ^ {2}}. \tag {21}
$$

This second-order central difference approximation performs temporal smoothing, where the learnable intensity $\nu _ { t } ( x )$ adapts by reducing diffusion in dynamic regions to preserve edges and increasing it in static areas for stability.

# 3.4.3. CORRECTION OPERATOR: SCORRECTION

Finally, we employ a relaxed Euler scheme to refine the prediction based on the original state:

$$
u _ {t} ^ {(3)} (x) = (1 - \lambda_ {t} (x)) u _ {t} ^ {(2)} (x) + \lambda_ {t} (x) u _ {t} ^ {\mathrm{obs}} (x). \tag {22}
$$

The coefficient $\lambda _ { t } ( x ) \in [ 0 , 1 ]$ is a learnable gating parameter activated via a Sigmoid function. This term functions analogously to a Kalman gain, dynamically correcting between the dynamic prediction $\bar { u } _ { t } ^ { ( 2 ) }$ and the global static features $u _ { t } ^ { \mathrm { o b s } }$ , thereby suppressing error accumulation. To address simplex deviations caused by the correction term, we project u(3t $\bar { u } _ { t } ^ { ( 3 ) }$ via $\mathcal { P } _ { \mathrm { p r o j } }$ to ensure numerical stability for the next iteration.

$$
u _ {t} (x) \leftarrow \frac {u _ {t} ^ {(3)} (x) + \epsilon}{\sum_ {x ^ {\prime} \in \Omega} (u _ {t} ^ {(3)} (x ^ {\prime}) + \epsilon)}. \tag {23}
$$

# 4. Experiments

# 4.1. Experimental Setup

Datasets. AAM is trained on Attention-1.75M, a standardized corpus unifying over 1.75M fixation instances across 8 image, 4 video, and 6 audio-visual datasets. Dataset specifications and textual conditions are provided in Appendix B. Covering diverse scenarios, Attention-1.75M substantially exceeds the training scale of existing methods. Implementation Details. AAM is implemented with the NPU-PyTorch CANN on four Ascend Snt9B3 NPUs. All inputs are resized to 448 × 448. We employ a phased training strategy: AAM is first trained on image and video data (with a sampling ratio of 4:6), during which the general attention anchor $z _ { \mathrm { a n c } }$ is warm-started by free-viewing datasets. Audio-visual data is then added after 10 epochs (Detailed configuration in Appendix 6).

![](images/0054bd4d4fe4923b7143d2068a6ab8aeaace0c9d717c5ede6d746f0b16333ec7.jpg)

<details>
<summary>text_image</summary>

Input
GT
AAM
SUM
UNISAL
</details>

(a) Image Attention Modeling

![](images/153305eeaa575bf6b678ad8a872216fcefad24813f3b7949bc8ebce1ee347d63.jpg)

<details>
<summary>natural_image</summary>

Grid of 24 grayscale thermal or heatmaps showing heat signatures over a dark background, no text or symbols present.
</details>

(b) Video Attention Modeling   
Figure 4. Visual comparison against SOTA methods.

Baselines. We compare our foundation model with stateof-the-art attention modeling methods including (i) scenedependent, task-specific models and (ii) jointly trained models with parameter isolation across partial datasets, as discussed in Section 1 (e.g., SUM and UNISAL).

# 4.2. Performance Analysis

Extensive experiments across 16 datasets demonstrate that AAM consistently outperforms SOTA methods using a single model. As in prior work (Hosseini et al., 2025b), we use the evaluation metrics AUC-Judd (AUC), Similarity Metric (SIM), Linear Correlation Coefficient (CC), and Normalized Scanpath Saliency (NSS). Specifically, averaging improvements across all evaluation metrics, we achieve average metric gains of 5.2%, 5.8% (including LEDOV in Appendix 21), and 6.0% on image (Table 1), audio-visual (Table 2), and video (Table 3) tasks, respectively (see Appendix F.4 for detailed comparison). Notably, diverging from existing methods that rely on computationally intensive multi-to-single-frame paradigms, AAM employs an efficient frame-wise prediction via FPD. This architectural distinction not only secures robust performance within a unified framework but also eliminates inherent redundancy. Consequently, AAM achieves an approximate 4× inference speedup compared to fixed-window methods (Table 4), with only 21M trainable parameters (324M total). Qualitative visual comparisons are presented in Fig. 4.

# 4.3. Ablation Study

# 4.3.1. ANALYSIS OF JOINT TRAINING

As shown in Fig. 5, all results are averaged over datasets. Under our hierarchical conditional modeling setting, we conduct a systematic ablation study to evaluate the effect of unified multimodal joint training. For image joint training (A), A1 evaluates cross-dataset generalization trained on single-dataset. A2 is the in-domain single-dataset baseline, where the model is trained and evaluated separately on each dataset. A3 performs full images joint training, and A4 further incorporates video data for cross-modal joint training. For audio-visual training (B), B1 is the in-domain singledataset baseline, B2 adds video data, and B3 applies full multimodal unified training, leading to consistent improvements. For video joint training (C), C1 is the in-domain single-dataset baseline, C2 adds video data, C3 combines video and image data, and C4 trains on all data jointly, yielding stable gains across benchmarks. Overall, these results support our hypothesis that unified multimodal training enhances generalization under hierarchical attention modeling.

Table 1. Quantitative comparison on image datasets. The best results are shown in red, and the second best in blue. 

<table><tr><td colspan="10">Image Attention Modeling</td></tr><tr><td>Method</td><td>CC ↑</td><td>KLD ↓</td><td>AUC ↑</td><td>SIM ↑</td><td>Method</td><td>CC ↑</td><td>KLD ↓</td><td>AUC ↑</td><td>SIM ↑</td></tr><tr><td>Dataset: MIT1003 (Natural)</td><td colspan="9">Dataset: U-EYE (Web page)</td></tr><tr><td>UNISAL (Droste et al., 2020)</td><td>0.734</td><td>1.014</td><td>0.902</td><td>0.597</td><td>TransalNet (Lou et al., 2022)</td><td>0.696</td><td>0.616</td><td>0.839</td><td>0.598</td></tr><tr><td>TransalNet (Lou et al., 2022)</td><td>0.722</td><td>0.660</td><td>0.903</td><td>0.592</td><td>UMSI++ (Jiang et al., 2023)</td><td>0.670</td><td>0.860</td><td>0.830</td><td>0.580</td></tr><tr><td>SUM (Hosseini et al., 2025b)</td><td>0.768</td><td>0.563</td><td>0.913</td><td>0.630</td><td>SUM (Hosseini et al., 2025b)</td><td>0.731</td><td>0.544</td><td>0.846</td><td>0.630</td></tr><tr><td>AAM (Ours)</td><td>0.831</td><td>0.446</td><td>0.923</td><td>0.674</td><td>AAM (Ours)</td><td>0.743</td><td>0.524</td><td>0.847</td><td>0.635</td></tr><tr><td>Dataset: CAT2000 (Natural)</td><td colspan="9">Dataset: SalECI (E-Commercial)</td></tr><tr><td>UNISAL (Droste et al., 2020)</td><td>0.842</td><td>0.530</td><td>0.876</td><td>0.721</td><td>EML-NET (Jiang et al., 2023)</td><td>0.510</td><td>1.220</td><td>0.807</td><td>0.536</td></tr><tr><td>TransalNet (Lou et al., 2022)</td><td>0.877</td><td>0.287</td><td>0.882</td><td>0.744</td><td>Hosseini (Hosseini et al., 2025a)</td><td>0.750</td><td>0.578</td><td>0.892</td><td>0.645</td></tr><tr><td>SUM (Hosseini et al., 2025b)</td><td>0.882</td><td>0.270</td><td>0.888</td><td>0.754</td><td>SUM (Hosseini et al., 2025b)</td><td>0.789</td><td>0.473</td><td>0.899</td><td>0.680</td></tr><tr><td>AAM (Ours)</td><td>0.906</td><td>0.235</td><td>0.890</td><td>0.769</td><td>AAM (Ours)</td><td>0.797</td><td>0.450</td><td>0.899</td><td>0.678</td></tr><tr><td>Dataset: SALICON (Natural)</td><td colspan="9">Dataset: OSIE (Natural)</td></tr><tr><td>TransalNet (Lou et al., 2022)</td><td>0.890</td><td>0.220</td><td>0.867</td><td>0.783</td><td>TransalNet (Jiang et al., 2023)</td><td>0.791</td><td>0.667</td><td>0.923</td><td>0.651</td></tr><tr><td>Temp-Sal (Aydemir et al., 2023)</td><td>0.911</td><td>0.195</td><td>0.869</td><td>0.800</td><td>UniAR (Li et al., 2024)</td><td>0.754</td><td>0.547</td><td>0.867</td><td>0.647</td></tr><tr><td>SUM (Hosseini et al., 2025b)</td><td>0.909</td><td>0.192</td><td>0.876</td><td>0.804</td><td>SUM (Hosseini et al., 2025b)</td><td>0.861</td><td>0.340</td><td>0.924</td><td>0.727</td></tr><tr><td>AAM (Ours)</td><td>0.925</td><td>0.163</td><td>0.876</td><td>0.819</td><td>AAM (Ours)</td><td>0.901</td><td>0.243</td><td>0.933</td><td>0.760</td></tr></table>

Table 2. Quantitative comparison on audio-visual datasets. The best results are shown in red, and the second best in blue. 

<table><tr><td colspan="16">Audio-Video Attention Modeling</td></tr><tr><td rowspan="2">Method</td><td colspan="3">DIEM</td><td colspan="3">ETMD</td><td colspan="3">SumMe</td><td colspan="3">Coutrot1</td><td colspan="3">Coutrot2</td></tr><tr><td>CC↑</td><td>NSS↑</td><td>AUC↑</td><td>CC↑</td><td>NSS↑</td><td>AUC↑</td><td>CC↑</td><td>NSS↑</td><td>AUC↑</td><td>CC↑</td><td>NSS↑</td><td>AUC↑</td><td>CC↑</td><td>NSS↑</td><td>AUC↑</td></tr><tr><td>CASP (Xiong et al., 2023)</td><td>0.655</td><td>2.61</td><td>0.906</td><td>0.620</td><td>3.34</td><td>0.940</td><td>0.499</td><td>2.60</td><td>0.907</td><td>0.561</td><td>2.65</td><td>0.889</td><td>0.788</td><td>6.34</td><td>0.963</td></tr><tr><td>DAVS (Zhu et al., 2024)</td><td>0.580</td><td>2.29</td><td>0.884</td><td>0.600</td><td>2.96</td><td>0.932</td><td>0.423</td><td>2.29</td><td>0.889</td><td>0.482</td><td>2.19</td><td>0.869</td><td>0.734</td><td>4.98</td><td>0.960</td></tr><tr><td>MSPI (Xie et al., 2024)</td><td>0.653</td><td>2.62</td><td>0.907</td><td>0.601</td><td>3.24</td><td>0.937</td><td>0.482</td><td>2.49</td><td>0.901</td><td>0.567</td><td>2.76</td><td>0.895</td><td>0.783</td><td>6.28</td><td>0.963</td></tr><tr><td>TAVDiff (Yu et al., 2025)</td><td>0.670</td><td>2.75</td><td>0.909</td><td>0.613</td><td>3.15</td><td>0.937</td><td>0.500</td><td>2.51</td><td>0.904</td><td>0.607</td><td>2.85</td><td>0.892</td><td>0.798</td><td>6.52</td><td>0.963</td></tr><tr><td>AAM (Ours)</td><td>0.710</td><td>2.88</td><td>0.919</td><td>0.655</td><td>3.66</td><td>0.945</td><td>0.550</td><td>2.90</td><td>0.920</td><td>0.626</td><td>3.22</td><td>0.911</td><td>0.887</td><td>7.46</td><td>0.971</td></tr></table>

# 4.3.2. COMPONENT ABLATION

We conduct a systematic ablation study on the proposed components. For the backbone ablation (D), we compare different visual encoder configurations, including DINOv3 (D1–D3, small→large) and SAM2 (Ravi et al., 2025) (D4 base, D5 large). The results indicate that stronger selfsupervised visual representations further improve overall performance. For the temporal modeling ablation (E), the model progresses from removing the temporal module entirely (E1), to adopting a standard time-dimension selfattention (Chen et al., 2025) (E2), and finally to our proposed FPD temporal module (E3/E4), which uses 16-frame and 32-frame clips as input, respectively. The full temporal design achieves the best performance, supporting our motivation to model video attention as a continuous evolution over a spatiotemporal manifold. For the hyperbolic representation ablation (F), performance consistently improves from removing hyperbolic learning (F1), to introducing a hyperbolic loss constraint (F2), and further to incorporating a hyperbolic decoder enhancement (F3), with more pronounced gains in complex scenarios exhibiting stronger hierarchical structure. In contrast, comparisons with the joint-training settings in (A) suggest that naive direct joint training may introduce domain conflicts and lead to degraded performance.

# 4.3.3. INDUCTIVE ANALYSIS

To analyze whether AAM effectively models the hierarchical attention of cognitive modulation, we conduct a series of ablation studies under varying semantic conditions, as shown in Fig. 6 (see Appendix F for detailed setups). We assess the cognitive regulation from three perspectives:

❶ Condition swap. We evaluate three configurations: Correct (task-aligned), Generic (general attention), and Wrong (mismatched tasks). We observe a strict performance hierarchy: Correct > Generic > Wrong, despite identical visual inputs. Such sensitivity to condition swapping demonstrates that AAM treats attention as a dynamic cognitive process rather than a static, stimulus-driven pixel mapping.

Table 3. Quantitative comparison on video datasets. The best results are shown in red, and the second best in blue. 

<table><tr><td colspan="13">Video Attention Modeling</td></tr><tr><td rowspan="2">Method</td><td colspan="4">DHF1K (Natural)</td><td colspan="4">Hollywood2 (Movies)</td><td colspan="4">UCF (Sports)</td></tr><tr><td>AUC↑</td><td>SIM↑</td><td>CC↑</td><td>NSS↑</td><td>AUC↑</td><td>SIM↑</td><td>CC↑</td><td>NSS↑</td><td>AUC↑</td><td>SIM↑</td><td>CC↑</td><td>NSS↑</td></tr><tr><td>UNISAL (Droste et al., 2020)</td><td>0.901</td><td>0.390</td><td>0.490</td><td>2.776</td><td>0.934</td><td>0.542</td><td>0.673</td><td>3.380</td><td>0.917</td><td>0.498</td><td>0.636</td><td>3.189</td></tr><tr><td>VSSM (Lu et al., 2023)</td><td>0.915</td><td>0.383</td><td>0.521</td><td>3.027</td><td>0.939</td><td>0.583</td><td>0.729</td><td>3.927</td><td>0.936</td><td>0.560</td><td>0.705</td><td>3.908</td></tr><tr><td>MSFF-Net (Zhang et al., 2023)</td><td>0.913</td><td>0.392</td><td>0.534</td><td>3.066</td><td>0.940</td><td>0.574</td><td>0.723</td><td>3.952</td><td>0.933</td><td>0.557</td><td>0.698</td><td>3.769</td></tr><tr><td>TFS-Net (Li et al., 2025b)</td><td>0.912</td><td>0.412</td><td>0.527</td><td>2.953</td><td>0.934</td><td>0.580</td><td>0.725</td><td>3.953</td><td>0.930</td><td>0.558</td><td>0.664</td><td>3.653</td></tr><tr><td>AAM (Ours)</td><td>0.919</td><td>0.421</td><td>0.563</td><td>3.272</td><td>0.944</td><td>0.599</td><td>0.742</td><td>4.055</td><td>0.943</td><td>0.584</td><td>0.736</td><td>3.892</td></tr></table>

(A) Image Joint Training   
![](images/18290196520b8d5707f4139693d0878cc25ee4bff923ebe826c032298548f31a.jpg)

<details>
<summary>bar</summary>

| Setting | A1   | A2   | A3   | A4   | KLD  | CC   |
|---------|------|------|------|------|------|------|
| Score   | 0.6  | 0.5  | 0.4  | 0.7  | 0.6  | 0.8  |
</details>

(B) Audio-Visual Joint Training   
![](images/db8b7816c9aff15750f70e391cb4c2b1709cc89c8cbda5d09c607ef0c09e61a5.jpg)

<details>
<summary>bar</summary>

| Metric | Value |
|--------|-------|
| B1     | 1     |
| B2     | 2     |
| B3     | 3     |
| AUC    | 1     |
| CC     | 2     |
| Metric | 3     |
</details>

(C) Video Joint Training   
![](images/ccbcd497d11288a464e3a4978b73c8a8c2bbe80d45b73f9b204dfaa3453be05c.jpg)

<details>
<summary>bar</summary>

| Category | Score |
| -------- | ----- |
| C1       | 0.5   |
| C2       | 0.4   |
| Setting  | 0.3   |
| C3       | 0.2   |
| C4       | 0.1   |
| SIM      | 0.0   |
| Metric   | 0.1   |
</details>

(D) Backbone Ablation   
![](images/ceb6d027c0bd18ec163822ac887ec9653c204ce85a79b870ec54e938b72e3f3b.jpg)

<details>
<summary>bar</summary>

| Setting | KLD | Score |
| ------- | --- | ----- |
| D1      | 0.4 | 0.6   |
| D2      | 0.6 | 0.8   |
| D3      | 0.8 | 0.6   |
| D4      | 0.6 | 0.4   |
| D5      | 0.4 | 0.2   |
| CC      | 0.8 | 0.6   |
| Metric   | 0.6 | 0.4   |
| KLD     | 0.4 | 0.2   |
</details>

(E) Temporal Module Ablation   
![](images/8d60ee80a0d58754daab101c9c825cf4d5b9ebac6a0e127af8022f92d84c8312.jpg)

<details>
<summary>bar</summary>

| Setting | UCF-CC | DHF1K-CC | DHF1K-SIM |
| ------- | ------ | -------- | --------- |
| E1      | 0.6    | 0.6      | 0.6       |
| E2      | 0.6    | 0.6      | 0.6       |
| E3      | 0.6    | 0.6      | 0.6       |
| E4      | 0.6    | 0.6      | 0.6       |
</details>

(F) Hyperbolic Ablation   
![](images/14d405c15b621d27e1650e077b9076a1d829969fc9ffa9fabcdfd25e330a67ec.jpg)

<details>
<summary>bar</summary>

| Category | Value |
| -------- | ----- |
| F1       | 0.6   |
| F2       | 0.7   |
| F3       | 0.8   |
</details>

Figure 5. Ablation experiments: (A) Different image training settings, (B) Different audio-visual training settings, (C) Different video training settings, (D) Different backbone settings, (E) Temporal module variants, (F) Hyperbolic component ablations.

(a) Condition swap   
![](images/a2fd5ef3902a9f10b8da663f889b4ac856df37578c05394b12fffb52a96eb0f0.jpg)

(b) Paraphrase invariance   
![](images/c7f35b53a0e61e0fc945ca0e7155e98813cedaad7766194da2e3cfec3e95c29a.jpg)

<details>
<summary>scatter</summary>

|        | CC     | SIM    |
| ------ | ------ | ------ |
| DHF1K  | 0.0020 | 0.0028 |
| DIEM   | 0.0045 | 0.0011 |
| SumMe  | 0.0064 | 0.0166 |
| Salicon| 0.0049 | 0.010   |
| OSIE   | 0.0062 | 0.0067 |
| UI     | 0.0380 | 0.0188 |
</details>

![](images/145f8d709a3798a087bd84b9846773701a17ba63513daf68df7f693bdf81d826.jpg)

<details>
<summary>scatter</summary>

| Group | Mean Value | Sigma |
|-------|-------------|-------|
| NS    | 0.06        | 0.0   |
| NS    | 0.15        | 0.0   |
| NS    | 0.74        | 0.0   |
| NS    | 0.021       | 0.0   |
| NS    | 0.039       | 0.0   |
| NS    | 0.085       | 0.0   |
| KLD   | 0.09        | 0.0   |
| KLD   | 0.017       | 0.0   |
| KLD   | 0.062       | 0.0   |
| KLD   | 0.042       | 0.0   |
| KLD   | 0.039       | 0.0   |
| KLD   | 0.085       | 0.0   |
</details>

(c) Prompt granularity   
![](images/c37c5a26b52e794e35a3f79cf195da71f6e372cf404ed1d21ef1c502e6ee5952.jpg)  
Figure 6. Inductive analysis of conditional attention modulation: (a) Condition swap, (b) Paraphrase invariance, (c) Prompt granularity.

Table 4. Comparison of complexity and efficiency metrics including Backbone, Input Length (Fixed/Arbitrary), Inference Speed (FPS), and Trainable Parameters. 

<table><tr><td>Method</td><td>Backbone</td><td>Input</td><td>FPS (img/s)</td><td>Param (M)</td></tr><tr><td>TASED (Min &amp; Corso, 2019)</td><td>3D Conv</td><td>Fixed</td><td>17</td><td>82</td></tr><tr><td>STSANet (Wang et al., 2021)</td><td>Video Swin</td><td>Fixed</td><td>28</td><td>643</td></tr><tr><td>TMFI-Net (Zhou et al., 2023)</td><td>Video Swin</td><td>Fixed</td><td>30</td><td>234</td></tr><tr><td>AAM (Ours)</td><td>DINOv3</td><td>Arbi</td><td>111</td><td>21.4</td></tr></table>

❷ Paraphrase invariance. To verify that AAM responds to semantic content rather than specific lexical cues, we generated multiple paraphrases for each prompt, varying in wording and syntactic structure. The model exhibits invariance to surface-level linguistic variations, evidenced by the negligible performance variance (CC standard deviation < 0.01) across paraphrases. Such robustness indicates that AAM captures the underlying semantics of cognitive cues instead of being overfitted to specific phrasing.   
❸ Prompt granularity. We refine prompts from general (G1) to highly specific (G4): (general→modality→viewing intention→detailed context) to probe hierarchical attention. As shown in Fig. 6 (c), performance on various tasks im-

![](images/eac27ac61682ec2beb40d563f8084d098f7fe39d94bf60663d5ca6ed076fa146.jpg)

<details>
<summary>scatter</summary>

| Point | X Coordinate | Y Coordinate |
|-------|--------------|--------------|
| 1     | 0.2          | 0.8          |
| 2     | 0.3          | 0.75         |
| 3     | 0.4          | 0.7          |
| 4     | 0.5          | 0.65         |
| 5     | 0.6          | 0.6          |
| 6     | 0.7          | 0.55         |
| 7     | 0.8          | 0.5          |
| 8     | 0.9          | 0.45         |
| 9     | 1.0          | 0.4          |
| 10    | 1.1          | 0.35         |
| 11    | 1.2          | 0.3          |
| 12    | 1.3          | 0.25         |
| 13    | 1.4          | 0.2          |
| 14    | 1.5          | 0.15         |
| 15    | 1.6          | 0.1          |
| 16    | 1.7          | 0.05         |
| 17    | 1.8          | 0.0          |
| 18    | 1.9          | -0.05        |
| 19    | 2.0          | -0.1         |
| 20    | 2.1          | -0.15        |
| 21    | 2.2          | -0.2         |
| 22    | 2.3          | -0.25        |
| 23    | 2.4          | -0.3         |
| 24    | 2.5          | -0.35        |
| 25    | 2.6          | -0.4         |
| 26    | 2.7          | -0.45        |
| 27    | 2.8          | -0.5         |
| 28    | 2.9          | -0.55        |
| 29    | 3.0          | -0.6         |
| 30    | 3.1          | -0.65        |
| 31    | 3.2          | -0.7         |
| 32    | 3.3          | -0.75        |
| 33    | 3.4          | -0.8         |
| 34    | 3.5          | -0.85        |
| 35    | 3.6          | -0.9         |
| 36    | 3.7          | -0.95        |
| 37    | 3.8          | -1.0         |
| 38    | 3.9          | -1.05        |
| 39    | 4.0          | -1.1         |
| 40    | 4.1          | -1.15        |
| 41    | 4.2          | -1.2         |
| 42    | 4.3          | -1.25        |
| 43    | 4.4          | -1.3         |
| 44    | 4.5          | -1.35        |
| 45    | 4.6          | -1.4         |
| 46    | 4.7          | -1.45        |
| 47    | 4.8          | -1.5         |
| 48    | 4.9          | -1.55        |
| 49    | 5.0          | -1.6         |
| 50    | 5.1          | -1.65        |
| 51    | 5.2          | -1.7         |
| 52    | 5.3          | -1.75        |
| 53    | 5.4          | -1.8         |
| 54    | 5.5          | -1.85        |
| 55    | 5.6          | -1.9         |
| 56    | 5.7          | -1.95        |
| 57    | 5.8          | -2.0         |
| 58    | 5.9          | -2.05        |
| 59    | 6.0          | -2.1         |
| 60    | 6.1          | -2.15        |
| 61    | 6.2          | -2.2         |
| 62    | 6.3          | -2.25        |
| 63    | 6.4          | -2.3         |
| 64    | 6.5          | -2.35        |
| 65    | 6.6          | -2.4         |
| 66    | 6.7          | -2.45        |
| 67    | 6.8          | -2.5         |
| 68    | 6.9          | -2.55        |
| 69    | 7.0          | -2.6         |
| 70    | 7.1          | -2.65        |
| 71    | 7.2          | -2.7         |
| 72    | 7.3          | -2.75        |
| 73    | 7.4          | -2.8         |
| 74    | 7.5          | -2.85        |
| 75    | 7.6          | -2.9         |
| 76    | 7.7          | -2.95        |
| 77    | 7.8          | -3.0         |
| 78    | 7.9          | -3.05        |
| 79    | 8.0          | -3.1         |
| 80    | 8.1          | -3.15        |
| 81    | 8.2          | -3.2         |
| 82    | 8.3          | -3.25        |
| 83    | 8.4          | -3.3         |
| 84    | 8.5          | -3.35        |
| 85    | 8.6          | -3.4         |
| 86    | 8.7          | -3.45        |
| 87    | 8.8          | -3.5         |
| 88    | 8.9          | -3.55        |
| 89    | 9.0          | -3.6         |
| 90    | 9.1          | -3.65        |
| 91    | 9.2          | -3.7         |
| 92    | 9.3          | -3.75        |
| 93    | 9.4          | -3.8         |
| 94    | 9.5          | -3.85        |
| 95    | 9.6          | -3.9         |
| 96    | 9.7          | -3.95        |
| 97    | 9.8          | -4.0         |
| 98    | 9.9          | -4.05        |
| 99    | 10.0         | -4.1         |
| Note: The data is randomly generated and may vary each time the code is run due to the use of random number generation.
</details>

![](images/33fb108a7dc59684bd32c574965688fbea8f3e7ec7dff5ef5b98d0bf4c216f28.jpg)

<details>
<summary>scatter</summary>

| Category | Value |
|---|---|
| Zanc | 0 |
| mean | 0.5 |
| log10(2000) | 0.3 |
| log10(2000) | 0.4 |
| log10(2000) | 0.6 |
| log10(2000) | 0.7 |
| log10(2000) | 0.8 |
| log10(2000) | 0.9 |
| log10(2000) | 1.0 |
| log10(2000) | 1.1 |
| log10(2000) | 1.2 |
| log10(2000) | 1.3 |
| log10(2000) | 1.4 |
| log10(2000) | 1.5 |
| log10(2000) | 1.6 |
| log10(2000) | 1.7 |
| log10(2000) | 1.8 |
| log10(2000) | 1.9 |
| log10(2000) | 2.0 |
| log10(2000) | 2.1 |
| log10(2000) | 2.2 |
| log10(2000) | 2.3 |
| log10(2000) | 2.4 |
| log10(2000) | 2.5 |
| log10(2000) | 2.6 |
| log10(2000) | 2.7 |
| log10(2000) | 2.8 |
| log10(2000) | 2.9 |
| log10(2000) | 3.0 |
| log10(2000) | 3.1 |
| log10(2000) | 3.2 |
| log10(2000) | 3.3 |
| log10(2000) | 3.4 |
| log10(2000) | 3.5 |
| log10(2000) | 3.6 |
| log10(2000) | 3.7 |
| log10(2000) | 3.8 |
| log10(2000) | 3.9 |
| log10(2000) | 4.0 |
| log10(2000) | 4.1 |
| log10(2000) | 4.2 |
| log10(2000) | 4.3 |
| log10(2000) | 4.4 |
| log10(2000) | 4.5 |
| log10(2000) | 4.6 |
| log10(2000) | 4.7 |
| log10(2000) | 4.8 |
| log10(2000) | 4.9 |
| log10(2000) | 5.0 |
| log10(2000) | 5.1 |
| log10(2000) | 5.2 |
| log10(2000) | 5.3 |
| log10(2000) | 5.4 |
| log10(2000) | 5.5 |
| log10(2000) | 5.6 |
| log10(2000) | 5.7 |
| log10(2000) | 5.8 |
| log10(2000) | 5.9 |
| log10(2000) | 6.0 |
| log1o (mean, log1o) | - |
| log1o (mean, log1o) | - |
| log1o (mean, log1o) | - |
| log1o (mean, log1o) | - |
| log1o (mean, log1o) | - |
| log1o (mean, log1o) | - |
| log1o (mean, log1o) | - |
| log1o (mean, log1o) | - |
|
| log1o (mean, log1o) | - |
| log1o (mean, log1o) | - |
| log1o (mean, log1o) | - |
| log1o (mean, log1o) | - |
| log1o (mean, log1o) | - |
| log1o (mean, log1o) | - |
|
| log1o (mean, log1o) | - |
|
| log1o (mean, log1o) | - |
|
| log1o (mean, log1o) | - |
|
| log1o (mean, log1o) | - |
|
| log1o (mean, log1o) | - |
|
| log1o (mean, log1o) | - |
|
| log1o (mean, log1e-2, -) | - |
|
| log1o (mean, log1e-2, -) | - |
|
| log1o (mean, log1e-2, -) | - |
|
| log1o (mean, log1e-2, -) | - |
|
| log1o (mean, log1e-2, -) | - |
|
| log1o (mean, log1e-2, -) | - |
| log1o (mean, log1e-2, -) | - |
|
| log1o (mean, log1e-2, -) | - |
|
|
| log1o (mean, log1e-2, -) | - |
|
|
| log1o (mean, log1e-2, -) | - |
|
|
| log1o (mean, log1e-2, -) | - |
|
|
|
| log1o (mean, log1e-2, -) | - |
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
</details>

![](images/7c376676e1c2a42304e6260449fc3f146e6c5259ef7f4cacc65a9efc93f5b50d.jpg)  
Figure 7. The learned hyperbolic representations: (a) HoroPCA (Chami et al., 2021) and (b) CO-SNE (Guo et al., 2022) visualizations of the latent space in L2.

proves with granularity, whereas dynamic and task-driven datasets plateau early. This diminishing marginal utility mirrors cognitive neuroscience findings, where dominant task contexts often override fine-grained linguistic nuances.

Zero-Shot Generalization. We evaluate the zero-shot generalization of our AAM on AVAD, which is unseen during training, and on LEDOV, which is trained without text prompts. On AVAD, AAM achieves strong performance when provided with appropriate text prompts, indicating high transferability of the learned attention representations. For LEDOV, our method also yields substantial performance gains without prompt input (Appendix 17 for details).

Geometric Transfer Explanation. Fig. 7 suggests that AAM organizes attention modulation on a shared hyperbolic manifold rather than independent task predictors. Generic prompts remain close to the origin, while increasingly specific task intents extend outward along hierarchical branches, providing a geometric explanation for strong zero-shot generalization across semantic conditions.

# 5. Conclusions

This paper argues that the persistent generalization failure in attention modeling is not a capacity issue, but a formulation issue: although human attention follows a unified cognitive mechanism, existing models fragment it into task-, scene-, and modality-specific problems. We address this mismatch by reframing attention variation as a hierarchical entailment process, modeling how general priors progressively specialize into task intent. Embedding this structure in hyperbolic space and modeling temporal evolution as continuous transport provide a unified geometric and physical interpretation of attention across space and time. Beyond performance gains, our results suggest that effective attention modeling requires explicit structure and dynamics, rather than isolated predictors. We hope this perspective encourages cognitively grounded foundation models that treat attention as a coherent process underlying visual perception.

# Acknowledgments

This research was supported by HUAWEI’s Al Hundred Schools Program and was carried out using the Ascend AI technology stack.

# Impact Statement

Ethical Considerations. We believe that our proposed AAM raises no ethical concerns regarding its motivation, design, implementation, or data usage. The method is designed to provide a unified foundation model for modeling human attention with cross-modal and cross-scene generalization while adhering to ethical guidelines in AI research.

Societal Implications. AAM introduces a unified perspective on human attention modeling. Unlike existing taskspecific and scene-dependent methods, it captures hierarchical aspects of human attention in a cognitively motivated manner. This perspective facilitates application to real-world scenarios, improving inference efficiency while preserving output quality. Moreover, AAM provides theoretical insights and empirical evidence for the future development of holistic visual perception systems, and can serve as a basis for downstream tasks related to attention modeling and visual saliency.

# References

Aydemir, B., Hoffstetter, L., Zhang, T., Salzmann, M., and Susstrunk, S. Tempsal-uncovering temporal information ¨ for deep saliency prediction. In CVPR, pp. 6461–6470, 2023.   
Aytar, Y., Vondrick, C., and Torralba, A. Soundnet: Learning sound representations from unlabeled video. Advances in neural information processing systems, 29, 2016.   
Bellitto, G., Proietto Salanitri, F., Palazzo, S., Rundo, F., Giordano, D., and Spampinato, C. Hierarchical domainadapted feature learning for video saliency prediction. IJCV, 129(12):3216–3232, 2021.   
Bogacz, R., Brown, E., Moehlis, J., Holmes, P., and Cohen, J. D. The physics of optimal decision making: a formal analysis of models of performance in two-alternative forced-choice tasks. Psychological review, 113(4):700, 2006.   
Borji, A. and Itti, L. Cat2000: A large scale fixation dataset for boosting saliency research. arXiv preprint arXiv:1505.03581, 2015.   
Bylinskii, Z., Isola, P., Bainbridge, C., Torralba, A., and Oliva, A. Intrinsic and extrinsic effects on image memorability. Vision research, 116:165–178, 2015.   
Chami, I., Gu, A., Nguyen, D. P., and Re, C. Horopca: ´ Hyperbolic dimensionality reduction via horospherical projections. In ICML, pp. 1419–1429. PMLR, 2021.   
Chang, Q. and Zhu, S. Temporal-spatial feature pyramid for video saliency detection. arXiv preprint arXiv:2105.04213, 2021.   
Chang, Q. and Zhu, S. Human vision attention mechanisminspired temporal-spatial feature pyramid for video saliency detection. Cognitive Computation, 15(3):856– 868, 2023.   
Chen, S., Jiang, M., and Zhao, Q. What do deep saliency models learn about visual attention? Advances in Neural Information Processing Systems, 36:9543–9555, 2023a.   
Chen, S., Valliappan, N., Shen, S., Ye, X., Kohlhoff, K., and He, J. Learning from unique perspectives: User-aware saliency modeling. In CVPR, pp. 2701–2710, 2023b.   
Chen, S., Guo, H., Zhu, S., Zhang, F., Huang, Z., Feng, J., and Kang, B. Video depth anything: Consistent depth estimation for super-long videos. In CVPR, pp. 22831– 22840, 2025.   
Chossat, P. and Faugeras, O. Hyperbolic planforms in relation to visual edges and textures perception. PLoS computational biology, 5(12):e1000625, 2009.

Cornia, M., Baraldi, L., Serra, G., and Cucchiara, R. Predicting human eye fixations via an lstm-based saliency attentive model. IEEE TIP, 27(10):5142–5154, 2018.   
Coutrot, A. and Guyader, N. How saliency, faces, and sound influence gaze in dynamic social scenes. Journal of vision, 14(8):5–5, 2014.   
Coutrot, A. and Guyader, N. An efficient audiovisual saliency model to predict eye positions when looking at conversations. In 2015 23rd European signal processing conference (EUSIPCO), pp. 1531–1535. IEEE, 2015.   
Desai, K., Nickel, M., Rajpurohit, T., Johnson, J., and Vedantam, S. R. Hyperbolic image-text representations. In ICML, pp. 7694–7731. PMLR, 2023.   
Droste, R., Jiao, J., and Noble, J. A. Unified image and video saliency modeling. In ECCV, pp. 419–435. Springer, 2020.   
Fosco, C., Casser, V., Bedi, A. K., O’Donovan, P., Hertzmann, A., and Bylinskii, Z. Predicting visual importance across graphic design types. In Proceedings of the 33rd Annual ACM Symposium on User Interface Software and Technology, pp. 249–260, 2020.   
Ganea, O., Becigneul, G., and Hofmann, T. Hyperbolic ´ entailment cones for learning hierarchical embeddings. In ICML, pp. 1646–1655. PMLR, 2018.   
Guidolin, A., Desroches, M., Victor, J. D., Purpura, K. P., and Rodrigues, S. Geometry of spiking patterns in early visual cortex: a topological data analytic approach. Journal of the Royal Society Interface, 19(196), 2022.   
Guo, Y., Guo, H., and Yu, S. X. Co-sne: Dimensionality reduction and visualization for hyperbolic data. In CVPR, pp. 21–30, 2022.   
Gygli, M., Grabner, H., Riemenschneider, H., and Van Gool, L. Creating summaries from user videos. In ECCV, pp. 505–520. Springer, 2014.   
Harel, J., Koch, C., and Perona, P. Graph-based visual saliency. Advances in neural information processing systems, 19, 2006.   
Hossein Khatoonabadi, S., Vasconcelos, N., Bajic, I. V., and Shan, Y. How many bits does it take for a stimulus to be salient? In CVPR, pp. 5501–5510, 2015.   
Hosseini, A., Hooshanfar, K., Omrani, P., Toosi, R., Toosi, R., Ebrahimian, Z., and Akhaee, M. A. Brand visibility in packaging: A deep learning approach for logo detection, saliency-map prediction, and logo placement analysis. Discover Applied Sciences, 7(6):537, 2025a.

Hosseini, A., Kazerouni, A., Akhavan, S., Brudno, M., and Taati, B. Sum: Saliency unification through mamba for visual attention modeling. In WACV, pp. 1597–1607. IEEE, 2025b.   
Hu, F. and McGuinness, K. Fastsal: A computationally efficient network for visual saliency prediction. In ICPR, pp. 9054–9061. IEEE, 2021.   
Huang, X., Shen, C., Boix, X., and Zhao, Q. Salicon: Reducing the semantic gap in saliency prediction by adapting deep neural networks. In ICCV, pp. 262–270, 2015.   
Jain, S., Yarlagadda, P., Jyoti, S., Karthik, S., Subramanian, R., and Gandhi, V. Vinet: Pushing the limits of visual modality for audio-visual saliency prediction. In IEEE IROS, pp. 3520–3527. IEEE, 2021.   
Jia, S. and Bruce, N. D. Eml-net: An expandable multilayer network for saliency prediction. Image and vision computing, 95:103887, 2020.   
Jiang, L., Xu, M., Liu, T., Qiao, M., and Wang, Z. Deepvs: A deep learning based video saliency prediction approach. In ECCV, pp. 602–617, 2018.   
Jiang, L., Wang, Z., Xu, M., and Wang, Z. Image saliency prediction in transformed domain: A deep complex neural network method. In AAAI, volume 33, pp. 8521–8528, 2019.   
Jiang, L., Li, Y., Li, S., Xu, M., Lei, S., Guo, Y., and Huang, B. Does text attract attention on e-commerce images: A novel saliency prediction dataset and method. In CVPR, pp. 2088–2097, 2022.   
Jiang, Y., Leiva, L. A., Rezazadegan Tavakoli, H., RB Houssel, P., Kylmal¨ a, J., and Oulasvirta, A. Ueyes: Under- ¨ standing visual saliency across user interface types. In Proceedings of the 2023 CHI conference on human factors in computing systems, pp. 1–21, 2023.   
Jiang, Y., Yan, X., Ji, G.-P., Fu, K., Sun, M., Xiong, H., Fan, D.-P., and Khan, F. S. Effectiveness assessment of recent large vision-language models. Visual Intelligence, 2(1): 17, 2024.   
Jin, Y., Zhou, X., Zhang, Z., Fang, H., Shi, R., and Xu, X. Hierarchical spatiotemporal feature interaction network for video saliency prediction. Image and Vision Computing, 154:105413, 2025.   
Judd, T., Ehinger, K., Durand, F., and Torralba, A. Learning to predict where humans look. In ICCV, pp. 2106–2113. IEEE, 2009.   
Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I., and Lempitsky, V. Hyperbolic image embeddings. In CVPR, pp. 6418–6428, 2020.

Koutras, P., Katsamanis, A., and Maragos, P. Predicting eyes’ fixations in movie videos: Visual saliency experiments on a new eye-tracking database. In International Conference on Engineering Psychology and Cognitive Ergonomics, pp. 183–194. Springer, 2014.   
Krioukov, D., Papadopoulos, F., Kitsak, M., Vahdat, A., and Boguna, M. Hyperbolic geometry of complex net- ´ works. Physical Review E—Statistical, Nonlinear, and Soft Matter Physics, 82(3):036106, 2010.   
Kroner, A., Senden, M., Driessens, K., and Goebel, R. Contextual encoder–decoder network for visual saliency prediction. Neural Networks, 129:261–270, 2020.   
Kummerer, M. and Bethge, M. Predicting visual fixations. ¨ Annual Review of Vision Science, 9(1):269–291, 2023.   
Kummerer, M., Khanuja, H. S., and Bethge, M. Modeling ¨ saliency dataset bias. In ICCV, pp. 22077–22088, 2025.   
Lai, Q., Wang, W., Sun, H., and Shen, J. Video saliency prediction using spatiotemporal residual attentive networks. IEEE TIP, 29:1113–1126, 2019.   
Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings of the 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.   
Le, M., Roller, S., Papaxanthos, L., Kiela, D., and Nickel, M. Inferring concept hierarchies from text corpora via hyperbolic embeddings. In ACL, pp. 3231–3241, 2019.   
Leboran, V., Garcia-Diaz, A., Fdez-Vidal, X. R., and Pardo, X. M. Dynamic whitening saliency. IEEE TPAMI, 39(5): 893–907, 2016.   
Li, J., Wang, J., Tan, C., Lian, N., Chen, L., Wang, Y., Zhang, M., Xia, S.-T., and Chen, B. Enhancing partially relevant video retrieval with hyperbolic learning. In ICCV, pp. 23074–23084, 2025a.   
Li, L., Dong, L., Zhang, H., Qin, J., Zhang, Z., and Sun, M. Tfs-net: Temporal first simulation network for video saliency prediction. Expert Systems with Applications, pp. 127652, 2025b.   
Li, P., He, J., Li, G., Bhargava, R., Shen, S., Valliappan, N., Liang, Y., Gu, H., Ramachandran, V., Farhadi, G., et al. Uniar: A unified model for predicting human attention and responses on visual content. Advances in Neural Information Processing Systems, 37:106346–106369, 2024.   
Linardos, A., Kummerer, M., Press, O., and Bethge, M.¨ Deepgaze iie: Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling. In ICCV, pp. 12919–12928, 2021.

Liu, Y., He, Z., and Han, K. Hyperbolic category discovery. In CVPR, pp. 9891–9900, 2025.   
Lou, J., Lin, H., Marshall, D., Saupe, D., and Liu, H. Transalnet: Towards perceptually relevant visual saliency prediction. Neurocomputing, 494:455–467, 2022.   
Lu, F., Lian, Y., Jin, B., and Gu, W. Visual saliency assistance mechanism based on visually impaired navigation systems. Displays, 79:102482, 2023.   
Mathe, S. and Sminchisescu, C. Actions in the eye: Dynamic gaze datasets and learnt saliency models for visual recognition. IEEE TPAMI, 37(7):1408–1424, 2014.   
Min, K. and Corso, J. J. Tased-net: Temporally-aggregating spatial encoder-decoder network for video saliency detection. In ICCV, pp. 2394–2403, 2019.   
Min, X., Zhai, G., Gu, K., and Yang, X. Fixation prediction through multimodal analysis. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 13(1):1–23, 2016.   
Mishra, D., Singh, S. K., Singh, R. K., and Kedia, D. Multiscale network (mssg-cnn) for joint image and saliency map learning-based compression. Neurocomputing, 460: 95–105, 2021.   
Mital, P. K., Smith, T. J., Hill, R. L., and Henderson, J. M. Clustering of gaze during dynamic scene viewing is predicted by motion. Cognitive computation, 3(1):5–24, 2011.   
Nickel, M. and Kiela, D. Poincare embeddings for learning ´ hierarchical representations. Advances in neural information processing systems, 30, 2017.   
Pal, A., van Spengler, M., di Melendugno, G. M. D., Flaborea, A., Galasso, F., and Mettes, P. Compositional entailment learning for hyperbolic vision-language models. In ICLR, 2025.   
Pan, J., Sayrol, E., Nieto, X. G.-i., Ferrer, C. C., Torres, J., McGuinness, K., and OConnor, N. E. Salgan: Visual saliency prediction with adversarial networks. In CVPR, 2017.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In ICML, pp. 8748–8763. PmLR, 2021.   
Rao, R. P. and Ballard, D. H. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature neuroscience, 2(1):79–87, 1999.

Ratcliff, R. A theory of memory retrieval. Psychological review, 85(2):59, 1978.   
Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Radle, R., Rolland, C., Gustafson, L., et al. ¨ Sam 2: Segment anything in images and videos. In ICLR, 2025.   
Reddy, N., Jain, S., Yarlagadda, P., and Gandhi, V. Tidying deep saliency prediction architectures. In IEEE IROS, pp. 10241–10247. IEEE, 2020.   
Riche, N., Mancas, M., Duvinage, M., Mibulumukini, M., Gosselin, B., and Dutoit, T. Rare2012: A multi-scale rarity-based saliency detection with its comparative statistical analysis. Signal Processing: Image Communication, 28(6):642–658, 2013.   
Rudoy, D., Goldman, D. B., Shechtman, E., and Zelnik-Manor, L. Learning video saliency from human gaze using candidate selection. In CVPR, pp. 1147–1154, 2013.   
Samani, M. A., Hooshanfar, K., Jey, H. S., and Esmailzadeh, S. M. Eye-tracking based control of a robotic arm and wheelchair for people with severe speech and motor impairment (ssmi). In 2023 11th RSI International Conference on Robotics and Mechatronics (ICRoM), pp. 35–41. IEEE, 2023.   
Sarkar, R. Low distortion delaunay embedding of trees in hyperbolic plane. In International symposium on graph drawing, pp. 355–366. Springer, 2011.   
Shen, C. and Zhao, Q. Webpage saliency. In ECCV, pp. 33–46. Springer, 2014.   
Shinn, M., Lam, N. H., and Murray, J. D. A flexible framework for simulating and fitting generalized drift-diffusion models. ELife, 9:e56938, 2020.   
Simeoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, ´ M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.   
Tang, Y., Zhan, G., Yang, L., Liao, Y., and Xu, C. Cardiff: Video salient object ranking chain of thought reasoning for saliency prediction with diffusion. In AAAI, volume 39, pp. 7302–7310, 2025.   
Tsiami, A., Koutras, P., and Maragos, P. Stavis: Spatiotemporal audiovisual saliency network. In CVPR, pp. 4766–4776, 2020.   
Wang, W. and Shen, J. Deep visual attention prediction. IEEE TIP, 27(5):2368–2378, 2017.

Wang, W., Shen, J., Guo, F., Cheng, M.-M., and Borji, A. Revisiting video saliency: A large-scale benchmark and a new model. In CVPR, pp. 4894–4903, 2018.   
Wang, Z., Liu, Z., Li, G., Wang, Y., Zhang, T., Xu, L., and Wang, J. Spatio-temporal self-attention network for video saliency prediction. IEEE TMM, 25:1161–1174, 2021.   
Woo, C., Lee, S., Park, S. M., and Kim, B. H. Recsal-net: Recursive saliency network for video saliency prediction. Neurocomputing, 650:130822, 2025.   
Wu, H.-H., Seetharaman, P., Kumar, K., and Bello, J. P. Wav2clip: Learning robust audio representations from clip. In ICASSP, 2022.   
Xie, J., Liu, Z., Li, G., and Song, Y. Audio-visual saliency prediction with multisensory perception and integration. Image and Vision Computing, 143:104955, 2024.   
Xiong, J., Wang, G., Zhang, P., Huang, W., Zha, Y., and Zhai, G. Casp-net: Rethinking video saliency prediction from an audio-visual consistency perceptual perspective. In CVPR, pp. 6441–6450, 2023.   
Xiong, J., Zhang, P., You, T., Li, C., Huang, W., and Zha, Y. Diffsal: Joint audio and video learning for diffusion saliency prediction. In CVPR, pp. 27273–27283, 2024.   
Xu, J., Jiang, M., Wang, S., Kankanhalli, M. S., and Zhao, Q. Predicting human gaze beyond pixels. Journal of vision, 14(1):28–28, 2014.   
Xu, M., Jiang, L., Sun, X., Ye, Z., and Wang, Z. Learning to detect video saliency with hevc features. IEEE TIP, 26 (1):369–385, 2016.   
Xue, H., Sun, M., and Liang, Y. Ecanet: Explicit cyclic attention-based network for video saliency prediction. Neurocomputing, 468:233–244, 2022.   
Yarbus, A. L. Eye movements and vision. Springer, 2013.   
Yu, L., Sun, X., Zhou, W., and Gabbouj, M. Text-audiovisual-conditioned diffusion model for video saliency prediction. arXiv preprint arXiv:2504.14267, 2025.   
Zhang, J. and Sclaroff, S. Exploiting surroundedness for saliency detection: A boolean map approach. IEEE TPAMI, 38(5):889–902, 2016. doi: 10.1109/TPAMI.2015. 2473844.   
Zhang, Y., Zhang, T., Wu, C., and Tao, R. Multi-scale spatiotemporal feature fusion network for video saliency prediction. IEEE TMM, 26:4183–4193, 2023.   
Zhou, X., Wu, S., Shi, R., Zheng, B., Wang, S., Yin, H., Zhang, J., and Yan, C. Transformer-based multi-scale feature integration network for video saliency prediction. IEEE TCSVT, 33(12):7696–7707, 2023.

Zhu, D., Zhang, K., Zhu, K., Zhang, N., Ding, W., Zhai, G., and Yang, X. From discrete representation to continuous modeling: A novel audio-visual saliency prediction model with implicit neural representations. IEEE TETCI, 8(6):4059–4074, 2024.

Contents 

<table><tr><td>Appendix A</td><td>Implementation Details ...... Page 14Hardware infrastructure (Huawei Ascend), training hyperparameters, and specific configurations for the Visual Backbone (DINOv3 + LoRA).</td></tr><tr><td>Appendix B</td><td>Dataset Details &amp; Texts ...... Page 15Detailed specifications of the Attention-1.75M dataset and the conditional text prompts for image, video, and audio-visual tasks.</td></tr><tr><td>Appendix C</td><td>Theoretical Proofs ...... Page 18Proof of Proposition regarding Discrete Mass Conservation and stability analysis of the drift-diffusion dynamics.</td></tr><tr><td>Appendix D</td><td>Methodology Details ...... Page 20Mathematical definitions of Loss Functions ( $\mathcal{L}_{\text{KLD}}, \mathcal{L}_{\text{CC}}, \mathcal{L}_{\text{SIM}}$ ); architectural details of the Hyper-bolic Decoder and Audio Fusion Module; specifications of the Drift Evolution Operator.</td></tr><tr><td>Appendix E</td><td>Additional Results ...... Page 22Comprehensive benchmark tables (Image, Video, AV); extensive ablation studies and component-wise analysis; additional comparisons with more baseline methods; detailed setups for Inductive Analysis (Condition Swap &amp; Paraphrase construction); and analysis of Zero-shot Generalization.</td></tr><tr><td>Appendix F</td><td>Future Directions ...... Page 24Discussion on integrating Large Language Models (LLMs) for fine-grained semantic granularities and extending the AAM paradigm to unify broad downstream attention-centric tasks.</td></tr></table>

# A. Implementation Details

The computing infrastructure specifications are detailed in Table 5. The hyperparameter settings for AAM training are listed in Table 6.

Table 5. Computing infrastructure for experiments on Huawei Ascend platform. 

<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>CPU</td><td>Huawei Kunpeng-920 (ARM64, 192 cores)</td></tr><tr><td>NPU</td><td>Huawei Ascend 910B3 (64GB HBM)</td></tr><tr><td>RAM</td><td>1.5TB</td></tr><tr><td>OS</td><td>EulerOS 2.0 (SP10)</td></tr><tr><td>NPU Firmware</td><td>23.0.6</td></tr><tr><td>CANN Version</td><td>6.3.2</td></tr><tr><td>Language</td><td>Python 3.7</td></tr><tr><td>Framework</td><td>MindSpore 2.1.0</td></tr><tr><td>Dependencies</td><td>torch 2.7.1, torchvision 0.22.1, numpy 1.26.4</td></tr></table>

# A.1. Task Losses and Metrics

Following prior work (Wang & Shen, 2017; Wang et al., 2018; Li et al., 2025b), we utilize standard metrics for optimization and evaluation. Let P, $\mathbf { G } \in \mathbb { R } ^ { H \times W }$ denote the predicted and ground-truth density maps (sum to 1), and $\mathbf { B } \in \{ 0 , 1 \} ^ { H \times W }$ denote the binary fixation map.

Table 6. Hyperparameter settings for AAM training. 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Training Dynamics</td></tr><tr><td>Input Resolution</td><td> $448 \times 448$ </td></tr><tr><td>Batch Size</td><td>32</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Base Learning Rate</td><td> $5 \times 10^{-4}$ </td></tr><tr><td>Weight Decay</td><td> $5 \times 10^{-3}$ </td></tr><tr><td>LR Scheduler</td><td>Cosine Annealing ( $T_{max} = 10, \eta_{min} = 10^{-5}$ )</td></tr><tr><td>Precision</td><td>BF16 (BFloat16)</td></tr><tr><td colspan="2">Model Architecture</td></tr><tr><td>Visual Backbone</td><td>DINOv3 ViT-L (Frozen)</td></tr><tr><td>Text Encoder</td><td>CLIP ViT-L/14 (Frozen)</td></tr><tr><td>Audio Encoder</td><td>Wav2CLIP Res-Net18(Frozen)</td></tr><tr><td>LoRA Rank ( $r$ )</td><td>32</td></tr><tr><td>LoRA Alpha ( $\alpha$ )</td><td>64</td></tr><tr><td>LoRA Dropout</td><td>0.05</td></tr><tr><td>LoRA Adaptation Scope</td><td>Last 24 Transformer Blocks</td></tr><tr><td colspan="2">Specialized Optimization</td></tr><tr><td>Router Weight Decay</td><td> $0.1 \times \text{Base Weight Decay} (5 \times 10^{-4})$ </td></tr><tr><td>Gradient Clipping</td><td>None</td></tr></table>

Kullback-Leibler Divergence (KLD). Minimizes the information loss between distributions:

$$
\mathcal {L} _ {\mathrm{KLD}} (\mathbf {P}, \mathbf {G}) = \sum_ {i} \mathbf {G} _ {i} \log \left(\frac {\mathbf {G} _ {i}}{\mathbf {P} _ {i} + \epsilon}\right). \tag {24}
$$

Correlation Coefficient (CC). Measures the linear correlation strength:

$$
\mathcal {L} _ {\mathrm{CC}} (\mathbf {P}, \mathbf {G}) = \frac {\operatorname{cov} (\mathbf {P} , \mathbf {G})}{\sigma_ {\mathbf {P}} \cdot \sigma_ {\mathbf {G}}}. \tag {25}
$$

Similarity Metric (SIM). Quantifies the histogram intersection (overlap):

$$
\mathcal {L} _ {\mathrm{SIM}} (\mathbf {P}, \mathbf {G}) = \sum_ {i} \min (\mathbf {P} _ {i}, \mathbf {G} _ {i}). \tag {26}
$$

AUC-Judd (AUC-J). Evaluates saliency as a binary classification task. It calculates the area under the ROC curve (TPR vs. FPR) by varying a threshold τ on P to classify fixation locations in B.

# B. Dataset Details and Texts

To foster community research and facilitate the development of downstream tasks, we will publicly release the AAM codebase alongside Attention-1.75M. To support foundation model training, we curated Attention-1.75M, a standardized corpus unifying over 1.75M fixation instances across image, video, and audio-visual modalities, each equipped with dataset-level prompts. Specifically, this corpus encompasses a diverse spectrum of scenarios—including e-commerce, webpages, UIs, natural scenes, photography, portraits, and cinematic content—spanning both free-viewing and task-driven attention paradigms, as illustrated in Table. 7, Table. 8, and Table. 9. The corresponding text conditions were systematically annotated based on the detailed metadata and acquisition protocols of the source datasets, as illustrated in Fig. 8, Fig. 9, and Fig. 10. In addition, we provide summary statistics of Attention-1.75M, along with details on annotation quality control and consistency measures, in Tables 10 and 11.

Table 7. Summary of Image-based Attention Modeling Datasets 

<table><tr><td colspan="6">Image Attention</td></tr><tr><td>Dataset</td><td>Publication</td><td>Domain</td><td>Images</td><td>Resolution</td><td>Task</td></tr><tr><td>SALICON (Huang et al., 2015)</td><td> $CVPR_{15}$ </td><td>Natural scenes</td><td>15,000</td><td>640 × 480</td><td>Free-view</td></tr><tr><td>MIT1003 (Judd et al., 2009)</td><td> $ICCV_{09}$ </td><td>Natural scenes</td><td>1,003</td><td>Varied</td><td>Free-view</td></tr><tr><td>CAT2000 (Borji &amp; Itti, 2015)</td><td> $arXiv_{15}$ </td><td>Natural scenes</td><td>2,000</td><td>1080 × 1920</td><td>Free-view</td></tr><tr><td>OSIE (Xu et al., 2014)</td><td>Journal of  $Vision_{14}$ </td><td>Natural scenes</td><td>700</td><td>800 × 600</td><td>Free-view</td></tr><tr><td>FIGRIM (Bylinskii et al., 2015)</td><td>Vision  $Research_{15}$ </td><td>Natural scenes</td><td>2,787</td><td>1366 × 768</td><td>Free-view</td></tr><tr><td>U-EYE (Jiang et al., 2023)</td><td>ACM  $CHI_{23}$ </td><td>Web pages</td><td>1,583</td><td>Varied</td><td>Task-driven</td></tr><tr><td>FiWI (Shen &amp; Zhao, 2014)</td><td> $ECCV_{14}$ </td><td>Web pages</td><td>149</td><td>1366 × 768</td><td>Task-driven</td></tr><tr><td>SalECI (Jiang et al., 2022)</td><td> $CVPR_{22}$ </td><td>E-commerce</td><td>871</td><td>720 × 720</td><td>Task-driven</td></tr></table>

Table 8. Summary of Video-based Attention Modeling Datasets 

<table><tr><td colspan="8">Video Attention</td></tr><tr><td>Dataset</td><td>Publication</td><td>Domain</td><td>Videos</td><td>Resolution</td><td>Frames</td><td>Viewer</td><td>Task</td></tr><tr><td>DHF1K (Wang et al., 2018)</td><td> $CVPR_{18}$ </td><td>Natural scenes</td><td>1,000</td><td>640 × 360</td><td>582,605</td><td>17</td><td>Free-view</td></tr><tr><td>Hollywood-2 (Mathe &amp; Sminchisescu, 2014)</td><td> $TPAMI_{15}$ </td><td>Movies</td><td>1,707</td><td>720 × 480</td><td>487,207</td><td>19</td><td>Task-driven</td></tr><tr><td>UCF Sports (Mathe &amp; Sminchisescu, 2014)</td><td> $TPAMI_{15}$ </td><td>Sports</td><td>150</td><td>720 × 480</td><td>9,900</td><td>19</td><td>Task-driven</td></tr><tr><td>LEDOV (Jiang et al., 2018)</td><td> $ECCV_{18}$ </td><td>Natural scenes</td><td>538</td><td>1280 × 720</td><td>179,336</td><td>32</td><td>Free-view</td></tr></table>

Table 9. Summary of Audio-Visual Attention Modeling Datasets 

<table><tr><td colspan="8">Audio-Video Attention</td></tr><tr><td>Dataset</td><td>Publication</td><td>Domain</td><td>Videos</td><td>Resolution</td><td>Frames</td><td>Viewers</td><td>Task</td></tr><tr><td>DIEM (Mital et al., 2011)</td><td>Cognit. Comput. $_{11}$ </td><td>Movies</td><td>84</td><td> $1280 \times 720$ </td><td>240,452</td><td>50</td><td>Free-view</td></tr><tr><td>Coutrot-1 (Coutrot &amp; Guyader, 2014)</td><td>Jour. of Vision $_{12}$ </td><td>People</td><td>60</td><td> $1280 \times 720$ </td><td>9,564</td><td>72</td><td>Task-driven</td></tr><tr><td>Coutrot-2 (Coutrot &amp; Guyader, 2015)</td><td>IJCV $_{14}$ </td><td>Natural scenes</td><td>40</td><td> $1280 \times 720$ </td><td>25,223</td><td>40</td><td>Task-driven</td></tr><tr><td>AVAD (Min et al., 2016)</td><td>TOMM $_{18}$ </td><td>Action events</td><td>60</td><td> $1920 \times 1280$ </td><td>17,134</td><td>16</td><td>Free-view</td></tr><tr><td>ETMD (Koutras et al., 2014)</td><td>SPIC $_{20}$ </td><td>Movies</td><td>30</td><td> $1280 \times 720$ </td><td>109,788</td><td>25</td><td>Free-view</td></tr><tr><td>SumMe (Gygli et al., 2014)</td><td>ECCV $_{14}$ </td><td>Sports</td><td>25</td><td> $640 \times 360$ </td><td>52,744</td><td>15</td><td>Task-driven</td></tr></table>

<table><tr><td colspan="2">IMAGE</td></tr><tr><td>Datasets</td><td>Language-Based Conditional Cognitive Prompts</td></tr><tr><td>Salicon</td><td>Static natural image from a large-scale object-in-context dataset, covering diverse everyday scenes and environments; free-viewing saliency data with weak to moderate center bias, designed for general-purpose saliency learning.</td></tr><tr><td>CAT2000</td><td>Static high-resolution image from a wide variety of categories including natural scenes and artificial patterns such as cartoons, sketches, fractals and abstract textures; free-viewing saliency with strong variability in category-dependent center bias.</td></tr><tr><td>MIT1003</td><td>Static natural photograph with strong photographer-style center framing; free-viewing eye-tracking strongly attracted to faces, people, readable text, animals and vehicles, exhibiting a strong center bias.</td></tr><tr><td>OSIE</td><td>Static everyday indoor or outdoor scene containing multiple interacting objects and rich semantic relationships; free-viewing eye-tracking with moderate center bias, commonly attracted to faces, gaze direction, text and object interactions.</td></tr><tr><td>U-EYE</td><td>Static user interface screenshot showing webpages or mobile apps, with menus, buttons, icons and dense small text; free-viewing eye-tracking where attention is strongly biased to the top-left, typical of browsing UI layouts and posters.</td></tr><tr><td>fiwi</td><td>Static webpage screenshot containing mixed visual elements such as text blocks, images, icons and faces in structured page layouts; free-viewing eye-tracking where attention is strongly guided by text regions typical of webpage browsing.</td></tr><tr><td>SalEC</td><td>Static e-commerce product image with packaging, brand logos, price tags and dense short text blocks; free-viewing eye-tracking dominated by text and logo-driven attention over retail and shopping items.</td></tr></table>

Figure 8. Image attention modeling datasets and their corresponding text-based cognitive conditional prompts

<table><tr><td colspan="2">VIDEO</td></tr><tr><td>Datasets</td><td>Language-Based Conditional Cognitive Prompts</td></tr><tr><td>DHF1K</td><td>Dynamic free-viewing video across diverse scenes and camera motions, containing multiple moving objects and complex backgrounds; saliency-style eye-tracking with dispersed attention and weak center bias.</td></tr><tr><td>Hollywood</td><td>Dynamic cinematic movie clip with actors, dialogues, shot changes and film-style camera motion;task-driven action recognition with strong center framing and typical movie cinematography.</td></tr><tr><td>UCF</td><td>Dynamic broadcast sports video showing athletes on courts or fields, often with uniforms, scoreboards and audience context; task-driven action recognition with strong center tracking of the main athlete and sports action</td></tr></table>

Figure 9. Video attention modeling datasets and their corresponding text-based cognitive conditional prompts

<table><tr><td colspan="2">AUDIO-VISUAL</td></tr><tr><td>Datasets</td><td>Language-Based Conditional Cognitive Prompts</td></tr><tr><td>DIEM</td><td>Audio-visual real-world and cinematic video clips including film, television, online video and everyday events; free-viewing eye-tracking capturing how gaze behavior shapes visual perception, memory and emotional experience over time.</td></tr><tr><td>AVAD</td><td>Audio-visual video clips centered on moving sound-generating objects, such as speaking faces, ; free-viewing eye-tracking emphasizing how tightly coupled audio and motion cues jointly guide visual attention over time.</td></tr><tr><td>Coutrot_db1</td><td>Audio-visual conversational video clips featuring multiple interacting people in realistic social environments; free-viewing eye-tracking revealing attention toward talking faces over time.</td></tr><tr><td>Coutrot_db2</td><td>Audio-visual video clips spanning landscapes, moving objects, and social conversations; free-viewing eye-tracking capturing how auditory cues modulate attention toward sound sources and talking faces.</td></tr><tr><td>ETMD_av</td><td>Audio-visual video cinematic clips from Oscar-winning films featuring high-motion action, dialogues, and complex visual storytelling; free-viewing eye-tracking showing attention guided by semantic cues such as faces over long durations.</td></tr><tr><td>SumMe</td><td>Audio-visual video clips depicting events like sports and holidays with static, moving, and egocentric camera motions; task-driven video summarization capturing human consensus on interestingness via explicit selection of important segments.</td></tr></table>

Figure 10. Audio-visual attention modeling datasets and their corresponding text-based cognitive conditional prompts

Table 10. Analysis of Dataset: Summary statistics of Attention-1.75M. More detailed quantitative analysis is presented in Appendix B. 

<table><tr><td>Aspect</td><td>Quantitative Bias / Key Findings</td></tr><tr><td>Modality</td><td>Static images dominate (88.0%), while video (11.3%) and audio-visual (0.7%) data are scarce.</td></tr><tr><td>Image Scene</td><td>Natural scenes account for 88.3% (&gt;70% landscape-style), while UI/web/e-commerce data contribute 11.7%.</td></tr><tr><td>Video Scene</td><td>Professional media (49.6%) and daily videos (43.0%) dominate, with limited sports (4.1%) and meetings (3.3%).</td></tr><tr><td>Demographic</td><td>Aged 18–36 (≈65–70% in 18–25; &lt;1% above 36; &gt;50% annotations rely on mouse-tracking proxies.</td></tr></table>

Table 11. Taxonomy of Attention Levels, corresponding descriptions, and representative datasets. Building upon this hierarchy, we construct prompts using a unified standardized template. 

<table><tr><td>Level</td><td>Attention</td><td>Description</td><td>Template Components</td><td>Datasets</td></tr><tr><td>High</td><td>Top-down</td><td>Guided by explicit goals and task objectives.</td><td>[task setting] + [bias]</td><td>Hollywood, UCF, SumMe</td></tr><tr><td>Mid</td><td>Semantic Mod.</td><td>Influenced by objects, interactions and semantic relationships.</td><td>[key elements]</td><td>OSIE, FIWI, UI, SalEC, DIEM, AVAD, Coutrot_db, ETMD_av</td></tr><tr><td>Low</td><td>Bottom-up</td><td>Driven by visual stimuli such as contrast and scene statistics.</td><td>[stimulus type] + [domain]</td><td>MIT1003, CAT2000, SALICON, DHF1K</td></tr></table>

\_ Standardized Prompt Template: 

<table><tr><td>[stimulus type] + [scene/domain] + [key elements] + [task setting] + [attention bias]</td></tr></table>

# C. Data Provenance and Licensing

We carefully reviewed the terms of use for all datasets included in Attention-1.75M. To ensure transparency and responsible use, we provide a data provenance and licensing summary in this appendix, documenting the original license, usage restrictions, and redistribution permissions associated with each dataset.

Datasets under CC BY 4.0. The following datasets are released under the Creative Commons Attribution 4.0 International license (CC BY 4.0): SALICON, U-EYE, DHF1K, Hollywood-2, and UCF. These datasets allow reuse and redistribution with appropriate attribution, subject to the terms of the license.

Research-only datasets. Several datasets are made available for research purposes only, following the terms specified in their original publications or official release pages. These include MIT300, CAT2000, FIGRIM, FiWI, LEDOV, Coutrot db, DIEM, ETMD, and SumMe. For these datasets, we follow the original usage restrictions and do not claim any additional redistribution rights.

Datasets under the MIT License. OSIE and SalECI are released under the MIT License, which permits reuse, modification, and redistribution under the conditions specified by the license.

Aggregate benchmark policy. Attention-1.75M is constructed as an aggregate benchmark from datasets with heterogeneous licensing conditions. To ensure compliance with the most restrictive components, Attention-1.75M will be released under a research-only policy.

Licensing matrix. Table 12 summarizes the licensing status and redistribution policy for each dataset.

# D. Theoretical Proof and Analysis

# D.1. Theoretical Properties of the Discretized FP Dynamics

The Fokker–Planck equation provides a physically grounded framework for modeling temporal saliency evolution, while enforcing a key structural constraint via mass conservation. This constraint stabilizes training and promotes meaningful, interpretable attention dynamics.

Table 12. Data provenance and licensing matrix for the datasets included in Attention-1.75M. 

<table><tr><td>Dataset</td><td>License / Usage Policy</td><td>Our Redistribution Policy</td></tr><tr><td>SALICON</td><td>CC BY 4.0</td><td>Refer to original license terms</td></tr><tr><td>U-EYE</td><td>CC BY 4.0</td><td>Refer to original license terms</td></tr><tr><td>DHF1K</td><td>CC BY 4.0</td><td>Refer to original license terms</td></tr><tr><td>Hollywood-2</td><td>CC BY 4.0</td><td>Refer to original license terms</td></tr><tr><td>UCF</td><td>CC BY 4.0</td><td>Refer to original license terms</td></tr><tr><td>MIT300</td><td>Research only</td><td>Not redistributed</td></tr><tr><td>CAT2000</td><td>Research only</td><td>Not redistributed</td></tr><tr><td>FIGRIM</td><td>Research only</td><td>Not redistributed</td></tr><tr><td>FiWI</td><td>Research only</td><td>Not redistributed</td></tr><tr><td>LEDOV</td><td>Research only</td><td>Not redistributed</td></tr><tr><td>Coutrot_db</td><td>Research only</td><td>Not redistributed</td></tr><tr><td>DIEM</td><td>Research only</td><td>Not redistributed</td></tr><tr><td>ETMD</td><td>Research only</td><td>Not redistributed</td></tr><tr><td>SumMe</td><td>Research only</td><td>Not redistributed</td></tr><tr><td>OSIE</td><td>MIT License</td><td>Refer to original license terms</td></tr><tr><td>SalECI</td><td>MIT License</td><td>Refer to original license terms</td></tr></table>

In video saliency prediction, standard metrics (e.g., KL, CC, SIM) implicitly assume attention to be a conserved finite resource. Accordingly, we model the saliency state at time t as a probability density on Ω and enforce the simplex constraint:

$$
u _ {t} \in \Delta = \left\{u \succeq 0 \middle | \sum_ {x \in \Omega} u (x) = 1 \right\}. \tag {27}
$$

Without this conservation law, the temporal module could trivially reduce loss by globally rescaling saliency magnitude, rather than capturing motion-driven drift and diffusion. Mass conservation removes this degree of freedom, ensuring stable and physically meaningful evolution.

In this appendix, we formally analyze the theoretical properties of the resulting discrete system. Specifically, we show that the Lie–Trotter operator splitting scheme (Sec. 3.4) preserves the probabilistic interpretation of attention distributions and ensures numerical stability.

Probability simplex. Recall that each saliency state $u _ { t } ~ \in ~ \mathbb { R } ^ { | \Omega | }$ is defined on the probability simplex $\Delta : = \quad$ $\{ u \succeq 0 , \mathbf { 1 } ^ { \top } u = 1 \}$ .

Proposition D.1 (Simplex Invariance and Stability). Assume periodic or zero-flux boundary conditions along the temporal axis. If the initial state satisfies $u _ { t } \in \Delta$ and the diffusion step size obeys the CFL condition

$$
\Delta t \nu_ {t} (x) \leq \frac {1}{2}, \quad \forall t, x, \tag {28}
$$

then the discretized FP evolution satisfies:

1. Non-negativity: all intermediate states remain element-wise non-negative.   
2. Mass boundedness: the $\ell _ { 1 }$ mass remains bounded and is strictly normalized after projection.   
3. Simplex invariance: the projection operator guarantees $u _ { t } \in \Delta$ at every iteration.

Proof Sketch. Drift step. The temporal attention kernel $A _ { x } ( t \gets t ^ { \prime } )$ is row-stochastic, thus defining a valid Markov transition. Consequently, the drift update is a convex combination of non-negative states and preserves non-negativity.

Diffusion step. The explicit second-order diffusion update can be rewritten as a convex interpolation of neighboring temporal states. Under the CFL constraint $\Delta t \nu _ { t } \le 1 / 2$ , all coefficients remain non-negative, ensuring stability and preventing oscillatory divergence.

![](images/61e10951e9316035df8eb51bf8ee9ee1b0db4ce38612bd40069266b22234368d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Visual Features X_in [B, C, H, W"]] --> B["Euclidean Path"]
    C["Text Euclidean Emb [B, L, D"]] --> B
    B --> D["MLP"]
    D --> E["GroupNorm"]
    E --> F["AdaGN (Mouulation γ, β)"]
    F --> G["Main Feature X"]
    G --> H["Map to Hyperbolic exp_map(X)"]
    H --> I["Hyperfobic Image Feats Z_img"]
    I --> J["Scale Weights {w_k}"]
    J --> K["Weighted Sum Σ w_k * S_k(X)"]
    K --> L["Scale Response X_s"]
    L --> M["Spatial Modulation ⊙(M)"]
    M --> N["Strength Modulation * (α)"]
    N --> O["Residual Add ⊕ (X + ...)"]
    O --> P["SiLU"]
    P --> Q["Refined Features X'"]
    Q --> R["Output"]

    subgraph Hyperbolic Geometry Guidance Branch
        S1["Hyperbolic Distance d_L(Z_txt, o)"] --> T1["Origin o"]
        T1 --> U1["Specialization Depth r_txt"]
        U1 --> V1["Modulation Strength α"]
        U1 --> W1["Scale Anchors {μ_k}"]
        W1 --> X1["Softmax"]
        X1 --> Y1["Scale Weights {w_k}"]
        Y1 --> Z1["Relative Geodesic Direction-Driven Spatial Focus"]
        Z1 --> AA1["Aperture Func ω(Z_txt)"]
        AA1 --> AB1["Focus Degree β"]
        AB1 --> AC1["Spatial Attention M"]
    end

    subgraph Joint Decoding
        AD["Op S₁ Conv 3x3"] --> AE["Op S₂ Dilated Conv ..."]
        AE --> AF["Op S₃ Global Pool"]
    end

    subgraph Specialization Depth-Driven Scale Modulation
        AG["Z_txt"] --> AH["Hyperbolic Distance d_L(Z_txt, o)"]
        AH --> AI["Origin o"]
        AI --> AJ["Specialization Depth r_txt"]
        AJ --> AK["Modulation Strength α"]
        AJ --> AL["Scale Anchors {μ_k}"]
        AL --> AM["Softmax"]
        AM --> AN["Scale Weights {w_k}"]
    end

    subgraph Relative Geodesic Direction-Driven Spatial Focus
        AO["LogMap at Origin log_o(·)"] --> AP["Relative Geodesic Dir. Δ = log(Z_img) - log(Z_txt)"]
        AP --> AQ["Aperture Func ω(Z_txt)"]
        AQ --> AR["Spatial Attention M"]
        AR --> AS["Focus Degree β"]
    end

    subgraph Output
        AT["Output"] --> AU["Output"]
    end
```
</details>

Figure 11. The architecture of audio fusion module

Correction step. The correction operator interpolates between the predicted state and the observed distribution, hence preserving non-negativity.

Projection. Finally, the simplex projection explicitly enforces $\mathbf { 1 } ^ { \top } u _ { t } = 1$ , eliminating accumulated numerical drift and ensuring a valid probability distribution for the next iteration. □

# D.2. Supplementary Derivation Details for FPD Parameters

The drift equation is given by utilize bidirectional temporal s $\begin{array} { r } { \frac { \partial u } { \partial t } = - \nabla _ { \tau } \cdot ( \mathbf { v } u ) } \end{array}$ . To simulate this physical meterize the drift propagator ess within a discrete feature space, we. We define the discrete transition kernel $A _ { x }$ $A _ { x } ( t \gets t ^ { \prime } )$ from source time $t ^ { \prime }$ to target time t as:

$$
A _ {x} ^ {(h)} (t \leftarrow t ^ {\prime}) = \frac {\exp \left(\langle q _ {t} ^ {(h)} (x) , k _ {t ^ {\prime}} ^ {(h)} (x) \rangle / (\sqrt {d} \beta)\right)}{\sum_ {\tau^ {\prime} = 1} ^ {T} \exp \left(\langle q _ {t} ^ {(h)} (x) , k _ {\tau^ {\prime}} ^ {(h)} (x) \rangle / (\sqrt {d} \beta)\right)}. \tag {29}
$$

Here, $x \in \Omega$ denotes a fixed spatial location, t represents the target time step for the current update, and $t ^ { \prime }$ indicates the source time step providing information. $q _ { t } ^ { ( h ) }$ and $k _ { t ^ { \prime } } ^ { ( h ) }$ refer to the query and key projections of the h-th attention head, respectively. The summation term in the denominator functions as a normalization factor, where $\tau ^ { \prime }$ is a dummy index traversing the entire temporal axis $1 , \ldots , T .$ . Finally, $\beta$ serves as a temperature coefficient to modulate the entropy of the transition distribution.

# E. Methodology Details

# E.1. Hyperbolic Decoder

As shown in Fig. 11 and Algorithm 1, the decoder modulates the backbone feature $X \in \mathbb { R } ^ { B \times C \times H \times W }$ using the text tangent vector t and hyperbolic condition embedding $z _ { \mathrm { c o n d } }$ . First, we compute the condition depth $r _ { \mathrm { c o n d } } = d _ { \mathbb { L } } ( z _ { \mathrm { c o n d } } , 0 )$ to control the modulation strength. This scalar guides the pixel-level hyperbolic gating, where we apply a log–gate–exp transformation: $\tilde { X } = \mathrm { H y p P i x G a t e } ( X , r _ { \mathrm { c o n d } } )$ . Next, for multi-scale extraction, we obtain structural responses $\{ Y _ { k } \} _ { k = 1 } ^ { K } = \{ \mathcal { O } _ { k } ( \tilde { X } ) \}$ using local, dilated, and global operators. These scales are fused via an anchor-based module:

$$
y = \text { HypScaleFuse } (\{Y _ {k} \}, t, z _ {\text { cond }}), \tag {30}
$$

where weights depend on text features and hyperbolic distances. Finally, the fused term y is injected into the upsampling blocks as $\hat { X } = \mathrm { U p B l o c k } ( X , t , y , r _ { \mathrm { c o n d } } )$ , and the final saliency map is generated by a joint Gaussian bias head:

$$
S = \text { GaussianBiasHead } (\hat {X}, t, z _ {\text { cond }}, r _ {\text { cond }}). \tag {31}
$$

Algorithm 1 Hyperbolic Multi-Scale Decoder (Data Flow)   
Input: backbone feature map X
Input: text feature t
Input: hyperbolic condition point $z_{cond}$ Output: saliency map S

Step 1: Condition depth $r_{cond} \leftarrow d_{\mathbb{L}}(z_{cond}, 0)$ Step 2: Pixel-level hyperbolic modulation $\tilde{X} \leftarrow \text{HypPixGate}(X, r_{cond}) \{\log-\text{gate}-\exp \text{ conditional gating}\}$ Step 3: Multi-scale structural extraction $\{Y_k\}_{k=1}^K \leftarrow \text{MultiScaleOps}(\tilde{X}) \{\text{local / dilated / global operators}\}$ Step 4: Hyperbolic scale fusion $y \leftarrow \text{HypScaleFuse}(\{Y_k\}, t, z_{cond}) \{\text{anchor-distance scale gating}\}$ Step 5: Upsampling with depth-controlled injection $\hat{X} \leftarrow \text{UpBlock}(X, t, y, r_{cond}) \{\text{Ada-style modulation + residual injection}\}$ Step 6: Joint Gaussian bias prediction $S \leftarrow \text{GaussianBiasHead}(\hat{X}, t, z_{cond}, r_{cond})$ Return: S

![](images/3dacf8cb5164a6b9675322446d33b5471d8c2bfdea125f2430cb18ffa09526af.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Raw Audio"] --> B["Wav2CLIP (Frozen ResNet-18)"]
    C["Visual Features"] --> B
    D["Global Feats"] --> E["Relevance Gate MLP (Bias Init = -3.0)"]
    B --> F["Cross-Attention (Spatial Alignment)"]
    F --> G["FiLM Generator (Linear + Tanh)"]
    G --> H["Global Pooling"]
    H --> I["Fused Features F_out"]
    I --> J["FiLM Modulation"]
    J --> K["F_v"]
    G --> L["Shift (β)"]
    L --> H
    style B stroke-dasharray: 5 5
    style H stroke-dasharray: 5 5
    style I stroke-dasharray: 5 5
```
</details>

Figure 12. The architecture of audio-visual fusion branch

# E.2. Audio-Visual Branch with Robust Feature Modulation

To effectively exploit auditory semantics while mitigating the interference from irrelevant background noise $( \mathrm { i . e . }$ , the audio-visual mismatch problem), we introduce a lightweight yet robust audio-visual fusion branch, as illustrated in Fig. 12 and Algorithm 2.

Audio Representation. To ensure semantic alignment with the visual modality under a limited computational budget, we avoid using computationally expensive Transformer-based audio encoders. Instead, we adopt Wav2CLIP (Wu et al., 2022) as our audio feature extractor. Wav2CLIP employs a ResNet-18 backbone distilled from the CLIP image encoder, projecting the raw audio waveform $\mathcal { A } \in \mathbb { R } ^ { T \times L }$ into a semantic embedding sequence $\mathbf { F } _ { a } \in \mathbb { R } ^ { T \times C _ { a } }$ , where $C _ { a } = 5 1 2$ . This design naturally places the extracted audio features in a shared latent space with the visual representations, facilitating cross-modal interaction.

Spatial Alignment via Cross-Attention. Given the visual feature map from the backbone network, denoted as $\mathbf { F } _ { v } \in$ $\mathbf { \nabla } \mathbf { \times } \mathbf { \times } \mathbf { \times } \mathbf { \times } \mathbf { \times } \mathbf { \times } \mathbf { \times } \mathbf { \times }$ , we first establish spatial correspondence across modalities. Specifically, we employ multi-head cross-attention (MHCA), where the flattened visual features $\mathbf { F } _ { v }$ serve as the Query, while the audio embeddings ${ \mathbf { F } } _ { a }$ act as both the $K e y$ and Value. This operation produces a spatially aligned auditory representation $\hat { { \mathbf { F } } } _ { a }$ , which guides auditory context toward visually relevant regions.

Stabilized FiLM Fusion. Na¨ıve additive fusion often leads to the phenomenon of modality laziness, where one modality dominates while the other is ignored. Instead, we adopt Feature-wise Linear Modulation (FiLM) to conditionally modulate the visual statistics using audio cues. To improve training stability, we propose a Residual Tanh-FiLM mechanism.

Algorithm 2 Robust Audio-Visual Fusion Branch with Stabilized Modulation   
Input: audio waveform $\mathcal{A} \in \mathbb{R}^{T \times L}$ Input: visual features $\mathbf{F}_v \in \mathbb{R}^{T \times HW \times C_v}$ Output: fused visual features $\mathbf{F}_{\text{out}}$ Step 1: Audio representation (Wav2CLIP) $\mathbf{F}_a \leftarrow \text{Wav2CLIP}(\mathcal{A})$ $\{\mathbf{F}_a \in \mathbb{R}^{T \times C_a}, \text{aligned with CLIP space}\}$ Step 2: Spatial alignment via cross-attention $\hat{\mathbf{F}}_a \leftarrow \text{MHCA}(\mathbf{Q} = \mathbf{F}_v, \mathbf{K} = \mathbf{F}_a, \mathbf{V} = \mathbf{F}_a)$ {audio context aligned to visual spatial locations}
Step 3: Modality gating (audio-visual relevance) $\alpha \leftarrow \sigma(\text{MLP}_{\text{gate}}([\text{Pool}(\mathbf{F}_v) \| \text{Pool}(\mathbf{F}_a)])$ $\{\alpha \in [0,1]$ , initialized with visual-prior bias}
Step 4: Stabilized Residual Tanh-FiLM fusion $(\gamma, \beta) \leftarrow \text{Linear}(\hat{\mathbf{F}}_a)$ {zero-initialized projection} $\mathbf{F}_{\text{out}} \leftarrow \mathbf{F}_v \odot (1 + \alpha \cdot \tanh(\gamma)) + \alpha \cdot \beta$ {bounded modulation prevents gradient explosion}
Step 5: Negative Sample Attack (training only)
if with probability $p = 0.3$ then
Replace $\mathcal{A}$ with Gaussian noise
end if
Add sparsity regularization loss: $\mathcal{L}_{\text{gate}} = \| \alpha \|_2$ Return: $\mathbf{F}_{\text{out}}$

Concretely, we generate the scaling and shifting parameters $( \gamma , \beta )$ from $\hat { { \mathbf { F } } } _ { a }$ through a zero-initialized linear projection layer. The modulation is formulated as:

$$
\mathbf {F} _ {\text { out }} = \mathbf {F} _ {v} \odot (1 + \alpha \cdot \tanh (\gamma)) + (\alpha \cdot \beta), \tag {32}
$$

where ⊙ denotes element-wise multiplication. The tanh(·) function bounds the scaling factor within a stable range, effectively preventing gradient explosion during early training.

Modality Gating and Negative Sample Attack. To explicitly handle audio-visual mismatch, we introduce a learnable relevance gate $\alpha \in [ 0 , 1 ]$ , computed by a lightweight MLP that takes global audio-visual embeddings as input. We initialize the gate bias to −3.0 (yielding an initial $\alpha \approx 0 . 0 5 )$ ), thereby imposing a visual-prior at the beginning of training. Furthermore, we adopt a Negative Sample Attack (NSA) strategy: with probability $p = 0 . 3 ,$ the audio input is replaced by Gaussian noise, and a sparsity regularization term $\lVert \alpha \rVert _ { 2 }$ is applied to encourage suppression of unreliable auditory signals. This adversarial setup forces the network to explicitly recognize and reject irrelevant or noisy audio cues.

# F. Additional Results

Synthesizing the ablation studies discussed in the main text, we draw the following conclusions:

1. Given visual input, the attention distribution is not unique but systematically modulated by viewing conditions and cognitive contexts.   
2. Language prompts provide an effective and interpretable mechanism for characterizing this modulation.   
3. Through hierarchical and unified modeling, shared structures across different modalities and attention domains are effectively captured.   
4. Collectively, these designs significantly enhance the model’s generalization capabilities across datasets and zero-shot scenarios.

In this appendix, we provide more detailed quantitative results and analyses.

# F.1. Detailed Ablation Studies

Joint Training Efficacy. Tables 13 and 14 present detailed results for image and audio-visual joint training, respectively. The results on image datasets demonstrate that, under our hierarchical modeling framework, joint multi-source attention training further boosts performance. Conversely, unconditional modeling leads to a significant performance degradation.

Backbone and Modality Analysis. Table 15 provides a quantitative comparison of different visual backbones, while Table 16 details the ablation results for the audio component, confirming that the incorporation of audio cues effectively aids attention prediction in multimodal scenarios.

# F.2. Inductive Analysis

To verify whether AAM effectively captures the hierarchical behaviors of cognitive modulation, we analyze the mechanism from three complementary perspectives and validate the model’s generalization ability (see Table 18 for comprehensive results): ❶ Condition Swap, ❷ Paraphrase Invariance, ❸ Prompt Granularity.

❶ Condition Swap: Attention Is Not Input-Only. For each dataset, we compare three condition settings: i) Correct condition (prompts aligned with the dataset protocol); ii) No condition (generic free-viewing prompts); and iii) Wrong condition (mismatched tasks).   
❷ Paraphrase Invariance: Conditioning on Semantics. To verify that the linguistic conditioning operates at a semantic level rather than relying on fixed lexical cues, we generated multiple paraphrases for each prompt. These paraphrases vary in wording, length, and syntactic structure but preserve the underlying semantic meaning (see Fig. 13).   
❸ Prompt Granularity: Hierarchical Cognitive Conditioning. We further investigate whether attention modulation exhibits a hierarchical structure by progressively refining prompts from general to highly specific. For each dataset, we constructed a four-level prompt hierarchy (see Fig. 14):

• G1 (General): Free-viewing without task or context specification.   
• G2 (Modality): Specifying image, video, or audio-visual input.   
• G3 (Viewing Intention): Coarse alignment with the dataset’s viewing protocol.   
• G4 (Detailed Context): Full scene description and dominant attention drivers.

Across extensive benchmarks, we observe a consistent trend: semantics- and layout-driven datasets (e.g., Salicon, OSIE, U-EYE, MIT1003, SalEC) show significant performance gains as prompt granularity increases. Dynamic and audio-visual datasets (e.g., DHF1K, DIEM) exhibit modest but stable gains, indicating that linguistic modulation is non-negligible. Task-driven datasets (e.g., UCF, SumMe) display dataset-specific optimal granularities, reflecting the dominance of task structure in guiding attention. Performance improves when prompt granularity matches the true viewing protocol and cognitive context, whereas over-specification or mis-specification yields diminishing returns. This behavior aligns with findings in cognitive neuroscience, suggesting that attention is jointly shaped by general viewing priors and specific task goals.

# F.3. Zero-shot Generalization

As shown in Table 17, we evaluate the zero-shot generalization of AAM on AVAD (unseen during training) and LEDOV (trained without text inputs). On AVAD, AAM matches the performance of state-of-the-art task-specific models under both text-conditioned and unconditional settings. On the diverse and large-scale LEDOV benchmark, AAM achieves significant performance gains without text input (see Table 21), demonstrating robust real-world applicability and generalization ability.

# F.3.1. VISUALIZATION OF HYPERBOLIC SPACE CHARACTERISTICS

Fig. 7 visualizes the learned hyperbolic embeddings, revealing a distinct “general → context → instance” radial hierarchy. Driven by the hierarchical ordering constraints, the unconditional anchors stably cluster near the origin, while conditional text embeddings (▲) and specific instance embeddings (•/■/♦) are pushed progressively toward the boundary.

Ablation: Paraphrase Invariance 

<table><tr><td colspan="2">U-EYE</td><td><img src="images/14aae645d96870c9f513cbaa35f286f0781fc4b1b379add337ecfe72aca7922d.jpg"/></td><td colspan="2">OSIE</td><td><img src="images/601f77f39db290606e338141c39108d445ee003237783318a8759de9382cf0b5.jpg"/></td></tr><tr><td>Text</td><td colspan="2">Static user interface screenshot showing webpages or mobile apps, with menus, buttons, icons and dense small text; free-viewing eye-tracking where attention is strongly biased to the top-left, typical of browsing UI layouts and posters.</td><td>Text</td><td colspan="2">Static everyday indoor or outdoor scene containing multiple interacting objects and rich semantic relationships; free-viewing eye-tracking with moderate center bias, commonly attracted to faces, gaze direction, text and object interactions.</td></tr><tr><td>Paraphrase 1</td><td colspan="2">&quot;Image. Free-viewing. An everyday scene with multiple objects; predict gaze attracted by people and interactions.&quot;</td><td>Paraphrase 1</td><td colspan="2">&quot;Image. Free-viewing. An everyday scene with multiple objects; predict gaze attracted by people and interactions.&quot;</td></tr><tr><td>Paraphrase 2</td><td colspan="2">&quot;Image. Unconstrained viewing of a complex scene. Predict fixations guided by semantic cues.&quot;</td><td>Paraphrase 2</td><td colspan="2">&quot;Image. Unconstrained viewing of a complex scene. Predict fixations guided by semantic cues.&quot;</td></tr><tr><td>Paraphrase 3</td><td colspan="2">&quot;Free viewing of an indoor or outdoor scene with rich context.&quot;</td><td>Paraphrase 3</td><td colspan="2">&quot;Free viewing of an indoor or outdoor scene with rich context.&quot;</td></tr><tr><td>Paraphrase 4</td><td colspan="2">&quot;No task is given. Predict saliency driven by faces, objects, and interactions.&quot;</td><td>Paraphrase 4</td><td colspan="2">&quot;No task is given. Predict saliency driven by faces, objects, and interactions.&quot;</td></tr><tr><td>Paraphrase 5</td><td colspan="2">&quot;Image. Natural viewing. Predict attention influenced by object relations and faces.&quot;</td><td>Paraphrase 5</td><td colspan="2">&quot;Image. Natural viewing. Predict attention influenced by object relations and faces.&quot;</td></tr><tr><td>Paraphrase 6</td><td colspan="2">&quot;Predict human fixation density in a complex everyday scene.&quot;</td><td>Paraphrase 6</td><td colspan="2">&quot;Predict human fixation density in a complex everyday scene.&quot;</td></tr><tr><td colspan="2">Salicon</td><td><img src="images/ee569f3b323d6a6119577a8e92a098ac14d96e4924a37b94fc0465561611269c.jpg"/></td><td colspan="2">DHF1K</td><td><img src="images/ddb23b12273e93beb524cd7243add393724f8d17289f9028ce069a9f2b829657.jpg"/></td></tr><tr><td>Text</td><td colspan="2">Static natural image from a large-scale object-in-context dataset, covering diverse everyday scenes and environments; free-viewing saliency data with weak to moderate center bias, designed for general-purpose saliency learning.</td><td>Text</td><td colspan="2">Dynamic free-viewing video across diverse scenes and camera motions, containing multiple moving objects and complex backgrounds; saliency-style eye-tracking with dispersed attention and weak center bias.</td></tr><tr><td>Paraphrase 1</td><td colspan="2">&quot;Image. Free-viewing. Predict where people naturally look without any task.&quot;</td><td>Paraphrase 1</td><td colspan="2">&quot;Video. Free-viewing. Predict gaze over time in dynamic scenes with motion.&quot;</td></tr><tr><td>Paraphrase 2</td><td colspan="2">&quot;Image. The observer views freely; estimate the human fixation density map.&quot;</td><td>Paraphrase 2</td><td colspan="2">&quot;Video. The viewer watches freely; estimate frame-wise human fixations.&quot;</td></tr><tr><td>Paraphrase 3</td><td colspan="2">&quot;Free viewing of a natural image. Predict typical human gaze locations.&quot;</td><td>Paraphrase 3</td><td colspan="2">&quot;Unconstrained viewing of a video. Predict saliency dynamics.&quot;</td></tr><tr><td>Paraphrase 4</td><td colspan="2">&quot;No explicit goal is given. Predict saliency under natural viewing.&quot;</td><td>Paraphrase 4</td><td colspan="2">&quot;No explicit task in the video. Predict natural gaze trajectories.&quot;</td></tr><tr><td>Paraphrase 5</td><td colspan="2">&quot;Predict general visual attention for free-viewing on the image.&quot;</td><td>Paraphrase 5</td><td colspan="2">&quot;Predict video saliency under free viewing.&quot;</td></tr><tr><td>Paraphrase 6</td><td colspan="2">&quot;Estimate human gaze distribution on an image in unconstrained viewing.&quot;</td><td>Paraphrase 6</td><td colspan="2">&quot;Estimate human gaze distribution for each frame during free viewing.&quot;</td></tr><tr><td colspan="2">DIEM</td><td><img src="images/29ad6f2338b15cd1230c237f58a4db5d31b1969022e139d9297ec7df99cc8f32.jpg"/></td><td colspan="2">SumMe</td><td><img src="images/cb61df26ff2a05379375857343b9b3e7b616a0f157014299503e4dacd29205eb.jpg"/></td></tr><tr><td>Text</td><td colspan="2">Audio-visual real-world and cinematic video clips including film, television, online video and everyday events; free-viewing eye-tracking capturing how gaze behavior shapes visual perception, memory and emotional experience over time.</td><td>Text</td><td colspan="2">Audio-visual video clips depicting events like sports and holidays with static, moving, and egocentric camera motions; task-driven video summarization capturing human consensus on interestingness via explicit selection of important segments.</td></tr><tr><td>Paraphrase 1</td><td colspan="2">&quot;Unconstrained viewing of an audio-visual clip. Predict attention shifts modulated.&quot;</td><td>Paraphrase 1</td><td colspan="2">&quot;Video. Task-driven viewing. Select important moments for video summarization.&quot;</td></tr><tr><td>Paraphrase 2</td><td colspan="2">&quot;Audio-visual video. Free-viewing. Predict gaze where sound may influence attention.&quot;</td><td>Paraphrase 2</td><td colspan="2">&quot;Video summarization task. Predict attention to key events.&quot;</td></tr><tr><td>Paraphrase 3</td><td colspan="2">&quot;AV video. The viewer watches freely; estimate fixations guided by audio cues.&quot;</td><td>Paraphrase 3</td><td colspan="2">&quot;Task: summarize the video. Focus on informative segments.&quot;</td></tr><tr><td>Paraphrase 4</td><td colspan="2">&quot;No task is given. Predict AV saliency influenced by auditory signals.&quot;</td><td>Paraphrase 4</td><td colspan="2">&quot;Goal-directed viewing for summarization. Attend to important actions.&quot;</td></tr><tr><td>Paraphrase 5</td><td colspan="2">&quot;Predict gaze dynamics for AV video under free viewing.&quot;</td><td>Paraphrase 5</td><td colspan="2">&quot;Predict gaze under summarization intent.&quot;</td></tr><tr><td>Paraphrase 6</td><td colspan="2">&quot;Estimate frame-wise gaze in AV content where sound can attract attention.&quot;</td><td>Paraphrase 6</td><td colspan="2">&quot;During summarization, attend to key events and informative regions.&quot;</td></tr></table>

Figure 13. Inductive analysis of conditional attention modulation: (b) Paraphrase invariance.

This radial stratification utilizes the exponential volume growth of hyperbolic space to prevent collapse and accommodate the vast diversity of specific samples. Crucially, the angular distribution exhibits an “intertwined yet separable” structure governed by multi-factor interactions: while modalities (Image/Video/Audio-Visual) form high-level directional branches, distinct viewing intentions (free-viewing vs. task-driven) induce significant intra-modal separation, and shared attention cues (e.g., faces or motion) create local cross-modal proximity without merging. This structure is quantitatively corroborated by the norm distribution analysis (Fig. 7), which confirms that our hyperbolic entailment cones successfully encode attention modeling as a coherent geometric process capturing radial specialization, angular semantic containment, and hierarchical differentiation.

# F.4. More Comparison Results

Due to space constraints, we present only a subset of state-of-the-art (SOTA) methods in the main text. For a more comprehensive comparison, we here evaluate our model against a broader range of baselines across Image Attention Modeling (Table 19), Video Attention Modeling (Tables 20 and 21), and Audio-Visual Modeling (Table 22). Additionally, we include qualitative visualization results under diverse scenarios and task settings in Fig. 15, Fig. 16

# G. Future Work.

LLM-Driven Hierarchical Refinement. Currently, textual conditions are primarily derived from fixed dataset labels or coarse captions. In future work, we plan to integrate Large Language Models (LLMs) to synthesize fine-grained, open-vocabulary descriptions. By leveraging the reasoning capabilities of LLMs, we aim to construct richer semantic trees, enabling the model to capture more subtle, granular hierarchical relationships in human attention modulation.

<table><tr><td colspan="12">Ablation: Prompt Granularity</td></tr><tr><td colspan="2">U-EYE</td><td><img src="images/33a369872991def1e472197a0c1736ef2805100fa6f0ed67f3d52ee6cccfd2a4.jpg"/></td><td><img src="images/011ee60e2bab7b42bc72d59bc9828f5be000c08e3e4bfa42ccaf0ac248930e20.jpg"/></td><td><img src="images/68dabfe7e4f5b4e3c83703418db659723aace9bc52a2ec59f59f6c1b7446634d.jpg"/></td><td><img src="images/43787f3a075f48440feb6c173e63b2c3b827c8b420f79c4dbd3f79d380e1ed63.jpg"/></td><td><img src="images/e9f9e5142ca2a710f9985d875c847e7bdbab1c7c5a85aeda1e9a9c9511609ff6.jpg"/></td><td colspan="2">OSIE</td><td><img src="images/e34901cd5854959a761737a7e33dc376574d42f8405fa17e34f805d75d5d7174.jpg"/></td><td><img src="images/cfe0b339c4e45265b1e9669c5b535ede6b38a6620d3b2a1f92817c30251050d8.jpg"/></td><td><img src="images/87728daed0dd2eca0059131d70911c7a49045e74b04e2158b54bdfbb7be189cd.jpg"/></td></tr><tr><td>Text</td><td colspan="6">Static user interface screenshot showing webpages or mobile apps, with menus, buttons, icons and dense small text; free-viewing eye-tracking where attention is strongly biased to the top-left, typical of browsing UI layouts and posters.</td><td>Text</td><td colspan="4">Static everyday indoor or outdoor scene containing multiple interacting objects and rich semantic relationships; free-viewing eye-tracking with moderate center bias, commonly attracted to faces, gaze direction, text and object interactions.</td></tr><tr><td>G1</td><td colspan="6">&quot;Free-viewing.P predict general human visual attention.&quot;</td><td>G1</td><td colspan="4">&quot;Free-viewing. Predict general human visual attention.&quot;</td></tr><tr><td>G2</td><td colspan="6">&quot;Static image observed under free-viewing conditions.&quot;</td><td>G2</td><td colspan="4">&quot;Static image observed under free-viewing conditions.&quot;</td></tr><tr><td>G3</td><td colspan="6">&quot;Static user interface image observed under free-viewing conditions.&quot;</td><td>G3</td><td colspan="4">&quot;Static product image observed under free-viewing conditions.&quot;</td></tr><tr><td>G4</td><td colspan="6">&quot;Static user interface screenshot showing webpages or mobile apps, with menus, buttons, icons and dense small text; free-viewing eye-tracking where attention is guided by layout structure, text regions, and interface elements.&quot;</td><td>G4</td><td colspan="4">&quot;Static e-commerce product image with packaging, brand logos, price tags and dense short text blocks; free-viewing eye-tracking dominated by text and logo-driven attention over retail items.&quot;</td></tr><tr><td colspan="2">MIT1003</td><td><img src="images/85a08be92e9b8a6901ad1a2c7d3a10921c007ba7572a8af5653269fd52c97ddb.jpg"/></td><td><img src="images/d69dc5e0165b3dada2af628afcf810b0083a2350a398b15f8ed80df15471ea52.jpg"/></td><td><img src="images/e12ce422b75665814dccccae0161fcbf40db6d63d17d8215e867fe8e86ac014d.jpg"/></td><td><img src="images/16572a4c3c6cfe8832f8d3bf751d6d3fc07d93e78c5a7dd7174b7fc9afd3210e.jpg"/></td><td><img src="images/59c7cd337d7cdfe2bae90082c025dc6598289d02635180deab953fcd700ba3f2.jpg"/></td><td colspan="2">SalEC</td><td><img src="images/c4cbd98bdbfffe84ea0c00978e6a7e940949739ef89b4f9a3f676e804454aa0e.jpg"/></td><td><img src="images/f5bf86a72b0c790be4728a5b1564b910fb37f5f3deafec5d01ae0c733f45b427.jpg"/></td><td><img src="images/073595a15da20faa4ed6fe5c7573c00bf5eebfcef531193b990a05c125debef4.jpg"/></td></tr><tr><td>Text</td><td colspan="6">Static natural photograph with strong photographer-style center framing; free-viewing eye-tracking strongly attracted to faces, people, readable text, animals and vehicles, exhibiting a strong center bias.</td><td>Text</td><td colspan="4">Static e-commerce product image with packaging, brand logos, price tags and dense short text blocks; free-viewing eye-tracking dominated by text and logo-driven attention over retail and shopping items.</td></tr><tr><td>G1</td><td colspan="6">&quot;Free-viewing. Predict general human visual attention.&quot;</td><td>G1</td><td colspan="4">&quot;Free-viewing. Predict general human visual attention.&quot;</td></tr><tr><td>G2</td><td colspan="6">&quot;Static image observed under free-viewing conditions.&quot;</td><td>G2</td><td colspan="4">&quot;Static image observed under free-viewing conditions.&quot;</td></tr><tr><td>G3</td><td colspan="6">&quot;Static natural photograph observed under free-viewing conditions.&quot;</td><td>G3</td><td colspan="4">&quot;Static product image observed under free-viewing conditions.&quot;</td></tr><tr><td>G4</td><td colspan="6">&quot;Static natural photograph containing people, faces, readable text, animals and vehicles; free-viewing eye-tracking attracted to semantically salient objects.&quot;</td><td>G4</td><td colspan="4">&quot;Static e-commerce product image with packaging, brand logos, price tags and dense short text blocks; free-viewing eye-tracking dominated by text and logo-driven attention over retail items.&quot;</td></tr><tr><td colspan="2">Salicon</td><td><img src="images/d39d1ac2d857c58999d3cba29d3bd8362cb266e47dd839e70b304a7e08d91157.jpg"/></td><td><img src="images/4959546822f20130ffc040926633981fa4597a9c902142ec0b9e9381c445dfe5.jpg"/></td><td><img src="images/6e0304521d8a8a8eabb9ee80932a9dd04c5e97955b489d5ce4466e17d0a5ce86.jpg"/></td><td><img src="images/18bcc5de63c33dd5bf0eca2bc5fc5813f9dc0b17298fdc8a1cbe5e78a76d260c.jpg"/></td><td><img src="images/2820fbeb2bbd913287ca429b19167f2f8e4d1ee2e0ae79a8c2d72136700723ab.jpg"/></td><td colspan="2">Coutrot_db1</td><td><img src="images/8d4376ee2f9a3d6214f67753dd885520b2870c0dc6158ce6f5ecc13ebcd199ff.jpg"/></td><td><img src="images/9d538c7897edb277e504c07df9abe45b345b63035377f00c24eed673c45cd86a.jpg"/></td><td><img src="images/1a883678adb1245202116cb3d82bdab7953b74d4f7667c54adb920165ad6b36a.jpg"/></td></tr><tr><td>Text</td><td colspan="6">Static natural image from a large-scale object-in-context dataset, covering diverse everyday scenes and environments; free-viewing saliency data with weak to moderate center bias, designed for general-purpose saliency learning.</td><td>Text</td><td colspan="4">Audio-visual conversational video clips featuring multiple interacting people in realistic social environments; free-viewing eye-tracking revealing attention toward talking faces over time.</td></tr><tr><td>G1</td><td colspan="6">&quot;Free-viewing. Predict general human visual attention.&quot;</td><td>G1</td><td colspan="4">&quot;Free-viewing. Predict general human visual attention.&quot;</td></tr><tr><td>G2</td><td colspan="6">&quot;Static image observed under free-viewing conditions.&quot;</td><td>G2</td><td colspan="4">&quot;Audio-visual video observed under free-viewing conditions.&quot;</td></tr><tr><td>G3</td><td colspan="6">&quot;Static natural image observed under free-viewing conditions.&quot;</td><td>G3</td><td colspan="4">&quot;Audio-visual conversational video observed under free-viewing conditions.&quot;</td></tr><tr><td>G4</td><td colspan="6">&quot;Static natural image covering diverse everyday scenes and environments; free-viewing saliency driven by semantically meaningful objects such as people, animals, vehicles and readable text.&quot;</td><td>G4</td><td colspan="4">&quot;Audio-visual conversational video featuring multiple interacting people; free-viewing eye-tracking revealing attention toward talking faces over time.&quot;</td></tr></table>

Figure 14. Inductive analysis of conditional attention modulation: (c) Prompt granularity.   
![](images/58fae28fe7dc4491085ac7feda3a5b201bbbfcad1f28bc83a765d714c1596aac.jpg)

<details>
<summary>text_image</summary>

Input
GT
AAM
SUM
UNSAL
3D View
</details>

(a) Image Attention Modeling

![](images/7e80261177b98d38571d57f11b4cd5c668cf80944e5e3574bdcc6be6ad75a657.jpg)

<details>
<summary>text_image</summary>

Input
GT
AAM
TMFI
UNSSAL
</details>

(b) Video Attention Modeling

![](images/865f99e89522359bcab483f14b4ea1fd8ca7dc739c657a1966f0eb246b6c06bb.jpg)

<details>
<summary>text_image</summary>

Input
GT
AAM
TMFI
UNSAL
</details>

(c) Audio-Visual Modeling   
Figure 15. Visual comparison against SOTA methods.

Universal Foundation for Downstream Tasks. Furthermore, we envision AAM serving as a versatile foundation model for the broader landscape of human attention and saliency-related research. We propose that diverse downstream tasks can be effectively unified under our paradigm, allowing for consistent modeling across different applications. Establishing AAM as a general-purpose backbone for these tasks represents a pivotal direction for our future research.

Table 13. Quantitative comparison of different training strategies on six image saliency datasets. Results demonstrate the effectiveness of joint training and the proposed hyperbolic components. Best results are highlighted in bold.   
ë Image Joint training 

<table><tr><td rowspan="2">Method</td><td colspan="3">Salicon</td><td colspan="3">OSIE</td><td colspan="3">U-EYE</td><td colspan="3">MIT1003</td><td colspan="3">SalECI</td><td colspan="3">CAT2000</td></tr><tr><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td></tr><tr><td>Single Training</td><td>0.911</td><td>0.806</td><td>0.175</td><td>0.872</td><td>0.727</td><td>0.312</td><td>0.733</td><td>0.630</td><td>0.539</td><td>0.781</td><td>0.622</td><td>0.537</td><td>0.788</td><td>0.673</td><td>0.469</td><td>0.879</td><td>0.633</td><td>0.270</td></tr><tr><td>Cross-Dataset Test</td><td>0.786</td><td>0.625</td><td>0.485</td><td>0.743</td><td>0.596</td><td>0.567</td><td>0.365</td><td>0.424</td><td>1.222</td><td>0.696</td><td>0.471</td><td>0.901</td><td>0.611</td><td>0.461</td><td>0.950</td><td>0.723</td><td>0.585</td><td>0.617</td></tr><tr><td>Joint Image (w/o Hyp)</td><td>0.907</td><td>0.796</td><td>0.191</td><td>0.870</td><td>0.724</td><td>0.331</td><td>0.639</td><td>0.578</td><td>0.709</td><td>0.777</td><td>0.604</td><td>0.571</td><td>0.763</td><td>0.663</td><td>0.521</td><td>0.820</td><td>0.708</td><td>0.361</td></tr><tr><td>Joint Image (w/ Hyp)</td><td>0.917</td><td>0.808</td><td>0.179</td><td>0.881</td><td>0.740</td><td>0.288</td><td>0.741</td><td>0.633</td><td>0.531</td><td>0.794</td><td>0.643</td><td>0.509</td><td>0.790</td><td>0.673</td><td>0.462</td><td>0.887</td><td>0.679</td><td>0.255</td></tr><tr><td>Joint Image (w/ Hyp + Dec)</td><td>0.924</td><td>0.816</td><td>0.166</td><td>0.900</td><td>0.760</td><td>0.249</td><td>0.746</td><td>0.635</td><td>0.527</td><td>0.824</td><td>0.674</td><td>0.469</td><td>0.796</td><td>0.677</td><td>0.459</td><td>0.909</td><td>0.771</td><td>0.238</td></tr><tr><td>Image+Video (w/o Hyp)</td><td>0.901</td><td>0.789</td><td>0.201</td><td>0.866</td><td>0.724</td><td>0.337</td><td>0.607</td><td>0.522</td><td>0.852</td><td>0.779</td><td>0.611</td><td>0.572</td><td>0.674</td><td>0.455</td><td>0.887</td><td>0.806</td><td>0.698</td><td>0.363</td></tr><tr><td>Image+Video (w/ Hyp)</td><td>0.925</td><td>0.819</td><td>0.163</td><td>0.901</td><td>0.762</td><td>0.245</td><td>0.745</td><td>0.635</td><td>0.523</td><td>0.830</td><td>0.674</td><td>0.445</td><td>0.796</td><td>0.678</td><td>0.452</td><td>0.907</td><td>0.769</td><td>0.241</td></tr></table>

Table 14. Ablation study of audio-video joint training across five audio-visual saliency datasets. We progressively incorporate additional video and image data into the training process. Results show that multi-source joint training consistently improves performance. Best results are highlighted in bold.   
Ð Y Audio-Video Joint training 

<table><tr><td rowspan="2">Method</td><td colspan="3">DIEM</td><td colspan="4">Coutrot_db1</td><td colspan="4">Coutrot_db2</td><td colspan="4">ETMD_av</td><td colspan="4">SumMe</td><td></td></tr><tr><td>AUC↑</td><td>CC↑</td><td>SIM↑</td><td>NSS↑</td><td>AUC↑</td><td>CC↑</td><td>SIM↑</td><td>NSS↑</td><td>AUC↑</td><td>CC↑</td><td>SIM↑</td><td>NSS↑</td><td>AUC↑</td><td>CC↑</td><td>SIM↑</td><td>NSS↑</td><td>AUC↑</td><td>CC↑</td><td>SIM↑</td><td>NSS↑</td></tr><tr><td>AV</td><td>0.910</td><td>0.662</td><td>0.535</td><td>2.69</td><td>0.902</td><td>0.589</td><td>0.464</td><td>2.95</td><td>0.970</td><td>0.886</td><td>0.698</td><td>7.360</td><td>0.939</td><td>0.531</td><td>0.482</td><td>3.485</td><td>0.910</td><td>0.521</td><td>0.393</td><td>2.744</td></tr><tr><td>AV + Video</td><td>0.916</td><td>0.684</td><td>0.489</td><td>2.51</td><td>0.906</td><td>0.625</td><td>0.489</td><td>3.15</td><td>0.971</td><td>0.888</td><td>0.706</td><td>7.410</td><td>0.941</td><td>0.546</td><td>0.500</td><td>3.596</td><td>0.915</td><td>0.535</td><td>0.406</td><td>2.832</td></tr><tr><td>AV + Video + Image</td><td>0.919</td><td>0.710</td><td>0.572</td><td>2.88</td><td>0.911</td><td>0.626</td><td>0.496</td><td>3.22</td><td>0.971</td><td>0.887</td><td>0.697</td><td>7.46</td><td>0.945</td><td>0.550</td><td>0.504</td><td>3.66</td><td>0.920</td><td>0.550</td><td>0.420</td><td>2.90</td></tr></table>

Table 15. Ablation study of different backbones. The best results are shown in bold.

â Ablation Study of Foundation Models 

<table><tr><td rowspan="2">Model</td><td colspan="3">Salicon</td><td colspan="3">OSIE</td><td colspan="3">U-EYE</td><td colspan="3">MIT1003</td><td colspan="3">SalECI</td><td colspan="3">CAT2000</td></tr><tr><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td></tr><tr><td>DINOv3_small</td><td>0.918</td><td>0.811</td><td>0.177</td><td>0.857</td><td>0.727</td><td>0.317</td><td>0.726</td><td>0.624</td><td>0.550</td><td>0.800</td><td>0.650</td><td>0.508</td><td>0.800</td><td>0.686</td><td>0.447</td><td>0.894</td><td>0.760</td><td>0.256</td></tr><tr><td>DINOv3_base</td><td>0.920</td><td>0.810</td><td>0.172</td><td>0.869</td><td>0.734</td><td>0.303</td><td>0.729</td><td>0.624</td><td>0.550</td><td>0.811</td><td>0.656</td><td>0.477</td><td>0.795</td><td>0.678</td><td>0.443</td><td>0.897</td><td>0.762</td><td>0.247</td></tr><tr><td>DINOv3_large</td><td>0.924</td><td>0.816</td><td>0.166</td><td>0.900</td><td>0.760</td><td>0.249</td><td>0.746</td><td>0.635</td><td>0.527</td><td>0.824</td><td>0.674</td><td>0.469</td><td>0.796</td><td>0.677</td><td>0.459</td><td>0.909</td><td>0.771</td><td>0.238</td></tr><tr><td>SAM2_base</td><td>0.871</td><td>0.763</td><td>0.243</td><td>0.811</td><td>0.722</td><td>0.289</td><td>0.719</td><td>0.613</td><td>0.566</td><td>0.790</td><td>0.641</td><td>0.500</td><td>0.729</td><td>0.613</td><td>0.397</td><td>0.870</td><td>0.726</td><td>0.245</td></tr><tr><td>SAM2_large</td><td>0.884</td><td>0.788</td><td>0.211</td><td>0.824</td><td>0.739</td><td>0.266</td><td>0.731</td><td>0.624</td><td>0.545</td><td>0.798</td><td>0.644</td><td>0.515</td><td>0.742</td><td>0.625</td><td>0.404</td><td>0.881</td><td>0.739</td><td>0.266</td></tr></table>

Table 16. Ablation study of the audio modality.

Ð Impact of Audio Modality 

<table><tr><td rowspan="2">Setting</td><td colspan="3">DIEM</td><td colspan="3">Coutrot_db1</td><td colspan="3">Coutrot_db2</td><td colspan="3">ETMD_av</td><td colspan="3">SumMe</td></tr><tr><td>CC↑</td><td>SIM↑</td><td>NSS↑</td><td>CC↑</td><td>SIM↑</td><td>NSS↑</td><td>CC↑</td><td>SIM↑</td><td>NSS↑</td><td>CC↑</td><td>SIM↑</td><td>NSS↑</td><td>CC↑</td><td>SIM↑</td><td>NSS↑</td></tr><tr><td>w/o Audio</td><td>0.703</td><td>0.571</td><td>2.880</td><td>0.615</td><td>0.492</td><td>3.180</td><td>0.862</td><td>0.689</td><td>7.40</td><td>0.648</td><td>0.422</td><td>3.660</td><td>0.549</td><td>0.422</td><td>2.910</td></tr><tr><td>w/ Audio</td><td>0.710</td><td>0.572</td><td>2.880</td><td>0.626</td><td>0.496</td><td>3.223</td><td>0.887</td><td>0.697</td><td>7.46</td><td>0.655</td><td>0.504</td><td>3.656</td><td>0.550</td><td>0.420</td><td>2.895</td></tr></table>

Table 17. Performance on Zero-shot Generalization.

 Zero-shot Generalization 

<table><tr><td>Method</td><td>AUC-J↑</td><td>NSS↑</td><td>CC↑</td><td>SIM↑</td></tr><tr><td colspan="5">AVAD Dataset</td></tr><tr><td>MSPI (Xie et al., 2024)</td><td>0.935</td><td>3.870</td><td>0.697</td><td>0.529</td></tr><tr><td>TAVDiff (Yu et al., 2025)</td><td>0.949</td><td>4.290</td><td>0.729</td><td>0.550</td></tr><tr><td>AAM (Ours) (w/ Text)</td><td>0.936</td><td>4.101</td><td>0.712</td><td>0.557</td></tr><tr><td>AAM (Ours) (Zero-shot, w/o Text)</td><td>0.933</td><td>4.100</td><td>0.695</td><td>0.542</td></tr><tr><td>Method</td><td>AUC-J↑</td><td>NSS↑</td><td>CC↑</td><td>KLD↑</td></tr><tr><td colspan="5">LEDOV Dataset</td></tr><tr><td>Sal-DCNN (Jiang et al., 2019)</td><td>0.892</td><td>2.838</td><td>0.573</td><td>1.304</td></tr><tr><td>AAM (Ours)(w/o Text)</td><td>0.941</td><td>3.652</td><td>0.708</td><td>0.888</td></tr></table>

Table 18. Prompt-related ablations and robustness analyses. (a) Condition Swap / Text Prompt Impact. (b) Paraphrase Invariance. (c) Prompt Granularity. Best results are in bold. 

<table><tr><td colspan="20">(a) Condition Swap / Text Prompt Impact</td></tr><tr><td rowspan="2">Setting</td><td colspan="3">Salicon</td><td colspan="3">OSIE</td><td colspan="3">UI</td><td colspan="3">MIT1003</td><td colspan="3">SalECI</td><td colspan="3">CAT2000</td><td>DHF1K</td></tr><tr><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑</td><td>SIM↑</td><td>KLD↓</td><td>CC↑ SIM↑ NSS↑</td></tr><tr><td>Correct Text Input</td><td>0.925</td><td>0.819</td><td>0.163</td><td>0.901</td><td>0.760</td><td>0.243</td><td>0.743</td><td>0.635</td><td>0.524</td><td>0.831</td><td>0.674</td><td>0.446</td><td>0.797</td><td>0.678</td><td>0.450</td><td>0.906</td><td>0.769</td><td>0.235</td><td>0.579 0.421 3.272</td></tr><tr><td>No Text Input (General)</td><td>0.873</td><td>0.759</td><td>0.247</td><td>0.851</td><td>0.703</td><td>0.375</td><td>0.480</td><td>0.494</td><td>1.060</td><td>0.783</td><td>0.620</td><td>0.549</td><td>0.699</td><td>0.589</td><td>0.618</td><td>0.833</td><td>0.719</td><td>0.361</td><td>0.523 0.368 2.960</td></tr><tr><td>Wrong Prompt (Cross-Task)</td><td>0.849</td><td>0.736</td><td>0.278</td><td>0.833</td><td>0.686</td><td>0.389</td><td>0.485</td><td>0.498</td><td>0.968</td><td>0.780</td><td>0.614</td><td>0.551</td><td>0.740</td><td>0.612</td><td>0.557</td><td>0.853</td><td>0.736</td><td>0.308</td><td>0.409 0.288 2.306</td></tr></table>

(b)  Paraphrase Invariance 

<table><tr><td>Metric</td><td>DHF1K</td><td>DIEM</td><td>SumMe</td><td>Salicon</td><td>OSIE</td><td>UI</td></tr><tr><td>CC</td><td>0.5768 ± 0.0020</td><td>0.6933 ± 0.0045</td><td>0.5458 ± 0.0064</td><td>0.7141 ± 0.0119</td><td>0.7158 ± 0.0082</td><td>0.4883 ± 0.0380</td></tr><tr><td>SIM</td><td>0.4302 ± 0.0028</td><td>0.5702 ± 0.0011</td><td>0.3971 ± 0.0166</td><td>0.6395 ± 0.0110</td><td>0.5961 ± 0.0067</td><td>0.5008 ± 0.0188</td></tr><tr><td>NSS</td><td>3.3121 ± 0.0158</td><td>2.8782 ± 0.0145</td><td>2.7465 ± 0.0738</td><td>1.4653 ± 0.0211</td><td>2.4431 ± 0.0390</td><td>1.1225 ± 0.0854</td></tr><tr><td>KLD</td><td>1.2227 ± 0.0092</td><td>0.7708 ± 0.0167</td><td>1.2722 ± 0.0223</td><td>0.5325 ± 0.0420</td><td>0.6748 ± 0.0393</td><td>0.9564 ± 0.0849</td></tr></table>

(c) Prompt Granularity Ablation (G1 → G4) 

<table><tr><td colspan="5">CC↑</td><td colspan="5">NSS ↑ (Static)</td><td colspan="5">NSS ↑ (Video)</td></tr><tr><td>Dataset</td><td>G1</td><td>G2</td><td>G3</td><td>G4</td><td>Dataset</td><td>G1</td><td>G2</td><td>G3</td><td>G4</td><td>Dataset</td><td>G1</td><td>G2</td><td>G3</td><td>G4</td></tr><tr><td>Salicon</td><td>0.805 →</td><td>0.848 →</td><td>0.867 →</td><td>0.913</td><td>Salicon</td><td>1.722 →</td><td>1.804 →</td><td>1.865 →</td><td>1.958</td><td>DHF1K</td><td>0.576 →</td><td>0.577 →</td><td>0.577 →</td><td>0.579</td></tr><tr><td>OSIE</td><td>0.803 →</td><td>0.818 →</td><td>0.839 →</td><td>0.883</td><td>OSIE</td><td>2.797 →</td><td>2.828 →</td><td>2.864 →</td><td>3.291</td><td>UCF</td><td>0.742 →</td><td>0.742 →</td><td>0.743 →</td><td>0.741</td></tr><tr><td>UI</td><td>0.478 →</td><td>0.477 →</td><td>0.586 →</td><td>0.734</td><td>UI</td><td>1.097 →</td><td>1.104 →</td><td>1.347 →</td><td>1.676</td><td>DIEM</td><td>0.700 →</td><td>0.698 →</td><td>0.697 →</td><td>0.702</td></tr><tr><td>MIT1003</td><td>0.756 →</td><td>0.766 →</td><td>0.795 →</td><td>0.824</td><td>MIT1003</td><td>2.624 →</td><td>2.638 →</td><td>2.762 →</td><td>2.974</td><td>SumMe</td><td>0.547 →</td><td>0.550 →</td><td>0.549 →</td><td>0.552</td></tr><tr><td>SalECI</td><td>0.726 →</td><td>0.721 →</td><td>0.767 →</td><td>0.807</td><td>SalECI</td><td>1.861 →</td><td>1.833 →</td><td>1.954 →</td><td>2.038</td><td>CAT2000</td><td>0.871 →</td><td>0.863 →</td><td>0.866 →</td><td>0.886</td></tr></table>

Table 19. Quantitative comparison on image attention modeling datasets. The best results are shown in red, and the second best in blue. 

<table><tr><td colspan="10">Image Attention Modeling</td></tr><tr><td>Method</td><td>CC ↑</td><td>KLD ↓</td><td>AUC ↑</td><td>SIM ↑</td><td>Method</td><td>CC ↑</td><td>KLD ↓</td><td>AUC ↑</td><td>SIM ↑</td></tr><tr><td>Dataset: MIT1003 (Natural)</td><td colspan="9">Dataset: U-EYE (Web page)</td></tr><tr><td>DVA (Wang &amp; Shen, 2017)</td><td>0.699</td><td>0.753</td><td>0.897</td><td>0.566</td><td>SAM (Cornia et al., 2018)</td><td>0.580</td><td>1.490</td><td>0.811</td><td>0.520</td></tr><tr><td>SAM-ResNet (Cornia et al., 2018)</td><td>0.746</td><td>1.247</td><td>0.902</td><td>0.597</td><td>UMSI (Fosco et al., 2020)</td><td>0.562</td><td>1.580</td><td>0.805</td><td>0.510</td></tr><tr><td>UNISAL (Droste et al., 2020)</td><td>0.734</td><td>1.014</td><td>0.902</td><td>0.597</td><td>TransalNet (Lou et al., 2022)</td><td>0.696</td><td>0.616</td><td>0.839</td><td>0.598</td></tr><tr><td>FastSal (Hu &amp; McGuinness, 2021)</td><td>0.590</td><td>1.036</td><td>0.875</td><td>0.478</td><td>SAM++ (Jiang et al., 2023)</td><td>0.580</td><td>1.190</td><td>0.800</td><td>0.530</td></tr><tr><td>TransalNet (Lou et al., 2022)</td><td>0.722</td><td>0.660</td><td>0.903</td><td>0.592</td><td>UMSI++ (Jiang et al., 2023)</td><td>0.670</td><td>0.860</td><td>0.830</td><td>0.580</td></tr><tr><td>SUM (Hosseini et al., 2025b)</td><td>0.768</td><td>0.563</td><td>0.913</td><td>0.630</td><td>SUM (Hosseini et al., 2025b)</td><td>0.731</td><td>0.544</td><td>0.846</td><td>0.630</td></tr><tr><td>AAM (Ours)</td><td>0.831</td><td>0.446</td><td>0.923</td><td>0.674</td><td>AAM (Ours)</td><td>0.743</td><td>0.524</td><td>0.847</td><td>0.635</td></tr><tr><td>Dataset: CAT2000 (Natural)</td><td colspan="9">Dataset: SalECI (E-Commercial)</td></tr><tr><td>DVA (Wang &amp; Shen, 2017)</td><td>0.861</td><td>0.449</td><td>0.878</td><td>0.734</td><td>SSM (Cornia et al., 2018)</td><td>0.720</td><td>0.599</td><td>0.830</td><td>0.611</td></tr><tr><td>SAM-ResNet (Cornia et al., 2018)</td><td>0.870</td><td>0.670</td><td>0.878</td><td>0.739</td><td>DeepGaze (Linardos et al., 2021)</td><td>0.560</td><td>0.995</td><td>0.842</td><td>0.399</td></tr><tr><td>MSI-Net (Kroner et al., 2020)</td><td>0.866</td><td>0.428</td><td>0.881</td><td>0.730</td><td>SSwin (Jiang et al., 2022)</td><td>0.687</td><td>0.652</td><td>0.868</td><td>0.606</td></tr><tr><td>UNISAL (Droste et al., 2020)</td><td>0.842</td><td>0.530</td><td>0.876</td><td>0.721</td><td>EML-NET (Jiang et al., 2023)</td><td>0.510</td><td>1.220</td><td>0.807</td><td>0.536</td></tr><tr><td>MDNSal (Reddy et al., 2020)</td><td>0.889</td><td>0.293</td><td>0.878</td><td>0.751</td><td>TransalNet (Jiang et al., 2023)</td><td>0.717</td><td>0.873</td><td>0.824</td><td>0.534</td></tr><tr><td>FastSal (Hu &amp; McGuinness, 2021)</td><td>0.721</td><td>0.552</td><td>0.860</td><td>0.603</td><td>Temp-Sal (Aydemir et al., 2023)</td><td>0.719</td><td>0.712</td><td>0.813</td><td>0.629</td></tr><tr><td>TransalNet (Lou et al., 2022)</td><td>0.877</td><td>0.287</td><td>0.882</td><td>0.744</td><td>Hosseini (Hosseini et al., 2025a)</td><td>0.750</td><td>0.578</td><td>0.892</td><td>0.645</td></tr><tr><td>SUM (Hosseini et al., 2025b)</td><td>0.882</td><td>0.270</td><td>0.888</td><td>0.754</td><td>SUM (Hosseini et al., 2025b)</td><td>0.789</td><td>0.473</td><td>0.899</td><td>0.680</td></tr><tr><td>AAM (Ours)</td><td>0.906</td><td>0.235</td><td>0.890</td><td>0.769</td><td>AAM (Ours)</td><td>0.797</td><td>0.450</td><td>0.899</td><td>0.678</td></tr><tr><td>Dataset: SALICON (Natural)</td><td colspan="9">Dataset: OSIE (Natural)</td></tr><tr><td>MDNSal (Reddy et al., 2020)</td><td>0.899</td><td>0.217</td><td>0.868</td><td>0.797</td><td>SAM-ResNet (Cornia et al., 2018)</td><td>0.758</td><td>0.480</td><td>0.860</td><td>0.648</td></tr><tr><td>MSI-Net (Kroner et al., 2020)</td><td>0.899</td><td>0.307</td><td>0.865</td><td>0.784</td><td>UMSI (Fosco et al., 2020)</td><td>0.746</td><td>0.513</td><td>0.856</td><td>0.631</td></tr><tr><td>UNISAL (Droste et al., 2020)</td><td>0.879</td><td>0.354</td><td>0.864</td><td>0.775</td><td>EML-NET (Jia &amp; Bruce, 2020)</td><td>0.717</td><td>0.537</td><td>0.854</td><td>0.619</td></tr><tr><td>DeepGaze (Linardos et al., 2021)</td><td>0.872</td><td>0.285</td><td>0.869</td><td>0.733</td><td>Hosseini (Chen et al., 2023b)</td><td>0.761</td><td>0.506</td><td>0.860</td><td>0.652</td></tr><tr><td>TransalNet (Lou et al., 2022)</td><td>0.890</td><td>0.220</td><td>0.867</td><td>0.783</td><td>TransalNet (Jiang et al., 2023)</td><td>0.791</td><td>0.667</td><td>0.923</td><td>0.651</td></tr><tr><td>Temp-Sal (Aydemir et al., 2023)</td><td>0.911</td><td>0.195</td><td>0.869</td><td>0.800</td><td>UniAR (Li et al., 2024)</td><td>0.754</td><td>0.547</td><td>0.867</td><td>0.647</td></tr><tr><td>SUM (Hosseini et al., 2025b)</td><td>0.909</td><td>0.192</td><td>0.876</td><td>0.804</td><td>SUM (Hosseini et al., 2025b)</td><td>0.861</td><td>0.340</td><td>0.924</td><td>0.727</td></tr><tr><td>AAM (Ours)</td><td>0.925</td><td>0.163</td><td>0.876</td><td>0.819</td><td>AAM (Ours)</td><td>0.901</td><td>0.243</td><td>0.933</td><td>0.760</td></tr></table>

Table 20. Quantitative comparison on video attention modeling datasets. The best results are shown in red, and the second best in blue. 

<table><tr><td colspan="13">Video Attention Modeling</td></tr><tr><td rowspan="2">Method</td><td colspan="4">DHF1K</td><td colspan="4">Hollywood2</td><td colspan="4">UCF</td></tr><tr><td>AUC↑</td><td>SIM↑</td><td>CC↑</td><td>NSS↑</td><td>AUC↑</td><td>SIM↑</td><td>CC↑</td><td>NSS↑</td><td>AUC↑</td><td>SIM↑</td><td>CC↑</td><td>NSS↑</td></tr><tr><td>DeepVS (Jiang et al., 2018)</td><td>0.856</td><td>0.256</td><td>0.344</td><td>1.911</td><td>0.887</td><td>0.356</td><td>0.446</td><td>2.313</td><td>0.870</td><td>0.321</td><td>0.405</td><td>2.089</td></tr><tr><td>ACLNet (Wang et al., 2018)</td><td>0.890</td><td>0.315</td><td>0.434</td><td>2.354</td><td>0.913</td><td>0.542</td><td>0.623</td><td>3.086</td><td>0.897</td><td>0.406</td><td>0.510</td><td>2.567</td></tr><tr><td>TASED-Net (Min &amp; Corso, 2019)</td><td>0.895</td><td>0.361</td><td>0.470</td><td>2.667</td><td>0.918</td><td>0.507</td><td>0.646</td><td>3.302</td><td>0.899</td><td>0.469</td><td>0.582</td><td>2.920</td></tr><tr><td>UNISAL (Droste et al., 2020)</td><td>0.901</td><td>0.390</td><td>0.490</td><td>2.776</td><td>0.934</td><td>0.542</td><td>0.673</td><td>3.380</td><td>0.917</td><td>0.498</td><td>0.636</td><td>3.189</td></tr><tr><td>HD2S (Bellitto et al., 2021)</td><td>0.908</td><td>0.406</td><td>0.503</td><td>2.812</td><td>0.936</td><td>0.551</td><td>0.670</td><td>3.352</td><td>0.904</td><td>0.507</td><td>0.604</td><td>3.114</td></tr><tr><td>ViNet (Jain et al., 2021)</td><td>0.908</td><td>0.381</td><td>0.511</td><td>2.872</td><td>0.930</td><td>0.550</td><td>0.693</td><td>3.730</td><td>0.924</td><td>0.522</td><td>0.673</td><td>3.620</td></tr><tr><td>STSANet (Wang et al., 2021)</td><td>0.913</td><td>0.383</td><td>0.529</td><td>3.010</td><td>0.938</td><td>0.579</td><td>0.721</td><td>3.974</td><td>0.938</td><td>0.563</td><td>0.698</td><td>3.889</td></tr><tr><td>ECANet (Xue et al., 2022)</td><td>0.903</td><td>0.385</td><td>0.500</td><td>2.814</td><td>0.929</td><td>0.526</td><td>0.673</td><td>3.910</td><td>0.923</td><td>0.561</td><td>0.685</td><td>3.698</td></tr><tr><td>VSSM (Lu et al., 2023)</td><td>0.915</td><td>0.383</td><td>0.521</td><td>3.027</td><td>0.939</td><td>0.583</td><td>0.729</td><td>3.927</td><td>0.936</td><td>0.560</td><td>0.705</td><td>3.908</td></tr><tr><td>TSFP-Net (Chang &amp; Zhu, 2023)</td><td>0.912</td><td>0.392</td><td>0.517</td><td>2.967</td><td>0.936</td><td>0.571</td><td>0.711</td><td>3.930</td><td>0.939</td><td>0.563</td><td>0.710</td><td>3.913</td></tr><tr><td>MSFF-Net (Zhang et al., 2023)</td><td>0.913</td><td>0.392</td><td>0.534</td><td>3.066</td><td>0.940</td><td>0.574</td><td>0.723</td><td>3.952</td><td>0.933</td><td>0.557</td><td>0.698</td><td>3.769</td></tr><tr><td>TFS-Net (Li et al., 2025b)</td><td>0.912</td><td>0.412</td><td>0.527</td><td>2.953</td><td>0.934</td><td>0.580</td><td>0.725</td><td>3.952</td><td>0.930</td><td>0.558</td><td>0.664</td><td>3.653</td></tr><tr><td>RecSal-Net (Woo et al., 2025)</td><td>0.913</td><td>0.414</td><td>0.547</td><td>3.135</td><td>0.938</td><td>0.606</td><td>0.737</td><td>3.901</td><td>0.918</td><td>0.523</td><td>0.644</td><td>3.381</td></tr><tr><td>AAM (Ours)</td><td>0.919</td><td>0.421</td><td>0.563</td><td>3.272</td><td>0.944</td><td>0.599</td><td>0.742</td><td>4.055</td><td>0.943</td><td>0.584</td><td>0.736</td><td>3.892</td></tr></table>

Table 21. Quantitative comparison on the LEDOV (video) dataset. The best results are shown in red, and the second best in blue. 

<table><tr><td colspan="5">LEDOV</td></tr><tr><td>Method</td><td>AUC-J↑</td><td>NSS↑</td><td>CC↑</td><td>KLD↓</td></tr><tr><td>GBVS (Harel et al., 2006)</td><td>0.839</td><td>1.541</td><td>0.322</td><td>1.824</td></tr><tr><td>Rudoy et al. (Rudoy et al., 2013)</td><td>0.799</td><td>1.454</td><td>0.320</td><td>2.421</td></tr><tr><td>SAILICON (Huang et al., 2015)</td><td>0.851</td><td>2.332</td><td>0.437</td><td>1.635</td></tr><tr><td>OBDL (Hossein Khatoonabadi et al., 2015)</td><td>0.801</td><td>1.545</td><td>0.315</td><td>2.053</td></tr><tr><td>AWS-D (Leboran et al., 2016)</td><td>0.795</td><td>1.365</td><td>0.294</td><td>2.023</td></tr><tr><td>Xu et al. (Xu et al., 2016)</td><td>0.827</td><td>1.475</td><td>0.382</td><td>1.652</td></tr><tr><td>BMS (Zhang &amp; Sclaroff, 2016)</td><td>0.757</td><td>0.979</td><td>0.214</td><td>2.225</td></tr><tr><td>SalGAN (Pan et al., 2017)</td><td>0.868</td><td>2.193</td><td>0.428</td><td>1.680</td></tr><tr><td>DVA (Wang &amp; Shen, 2017)</td><td>0.885</td><td>2.840</td><td>0.557</td><td>1.323</td></tr><tr><td>DeepVS (Jiang et al., 2018)</td><td>0.902</td><td>2.999</td><td>0.586</td><td>1.222</td></tr><tr><td>ACLNet (Wang et al., 2018)</td><td>0.897</td><td>2.872</td><td>0.570</td><td>1.445</td></tr><tr><td>Sal-DCNN (Jiang et al., 2019)</td><td>0.892</td><td>2.838</td><td>0.573</td><td>1.304</td></tr><tr><td>AAM (Ours)</td><td>0.941</td><td>3.652</td><td>0.708</td><td>0.888</td></tr></table>

Table 22. Quantitative comparison on audio-visual attention modeling datasets. 

<table><tr><td colspan="16">Audio-Video Attention Modeling</td></tr><tr><td rowspan="2">Method</td><td colspan="3">DIEM</td><td colspan="3">ETMD</td><td colspan="3">SumMe</td><td colspan="3">Coutrot1</td><td colspan="3">Coutrot2</td></tr><tr><td>CC↑</td><td>NSS↑</td><td>AUC↑</td><td>CC↑</td><td>NSS↑</td><td>AUC↑</td><td>CC↑</td><td>NSS↑</td><td>AUC↑</td><td>CC↑</td><td>NSS↑</td><td>AUC↑</td><td>CC↑</td><td>NSS↑</td><td>AUC↑</td></tr><tr><td>TSFP (Chang &amp; Zhu, 2021)</td><td>0.651</td><td>2.62</td><td>0.906</td><td>0.576</td><td>3.07</td><td>0.932</td><td>0.464</td><td>2.30</td><td>0.894</td><td>0.571</td><td>2.73</td><td>0.895</td><td>0.743</td><td>5.31</td><td>0.959</td></tr><tr><td>STAViS (Tsiami et al., 2020)</td><td>0.580</td><td>2.29</td><td>0.885</td><td>0.585</td><td>3.09</td><td>0.934</td><td>0.427</td><td>2.10</td><td>0.888</td><td>0.497</td><td>2.29</td><td>0.868</td><td>0.732</td><td>5.64</td><td>0.961</td></tr><tr><td>ViNet (Jain et al., 2021)</td><td>0.632</td><td>2.53</td><td>0.899</td><td>0.571</td><td>3.08</td><td>0.928</td><td>0.463</td><td>2.41</td><td>0.897</td><td>0.560</td><td>2.73</td><td>0.889</td><td>0.754</td><td>5.95</td><td>0.951</td></tr><tr><td>CASP (Xiong et al., 2023)</td><td>0.655</td><td>2.61</td><td>0.906</td><td>0.620</td><td>3.34</td><td>0.940</td><td>0.499</td><td>2.60</td><td>0.907</td><td>0.561</td><td>2.65</td><td>0.889</td><td>0.788</td><td>6.34</td><td>0.963</td></tr><tr><td>DAVS (Zhu et al., 2024)</td><td>0.580</td><td>2.29</td><td>0.884</td><td>0.600</td><td>2.96</td><td>0.932</td><td>0.423</td><td>2.29</td><td>0.889</td><td>0.482</td><td>2.19</td><td>0.869</td><td>0.734</td><td>4.98</td><td>0.960</td></tr><tr><td>MSPI (Xie et al., 2024)</td><td>0.653</td><td>2.62</td><td>0.907</td><td>0.601</td><td>3.24</td><td>0.937</td><td>0.482</td><td>2.49</td><td>0.901</td><td>0.567</td><td>2.76</td><td>0.895</td><td>0.783</td><td>6.28</td><td>0.963</td></tr><tr><td>TAVDiff (Yu et al., 2025)</td><td>0.670</td><td>2.75</td><td>0.909</td><td>0.613</td><td>3.15</td><td>0.937</td><td>0.500</td><td>2.51</td><td>0.904</td><td>0.607</td><td>2.85</td><td>0.892</td><td>0.798</td><td>6.52</td><td>0.963</td></tr><tr><td>AAM (Ours)</td><td>0.710</td><td>2.88</td><td>0.919</td><td>0.655</td><td>3.66</td><td>0.945</td><td>0.550</td><td>2.90</td><td>0.920</td><td>0.626</td><td>3.22</td><td>0.911</td><td>0.887</td><td>7.46</td><td>0.971</td></tr></table>

![](images/4ae7b08d9b104fd5d93aa7557bf460bbb7fa5bc09917b224c815aee57a8fa494.jpg)  
Figure 16. Prediction results of AAM (Ours) across various task scenarios, where GT denotes the human attention distribution.