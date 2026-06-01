# How can embedding models bind concepts?

# Arnas Uselis\* 1 Darina Koishigarina\* 1 Seong Joon Oh 1 2

# Abstract

Humans easily determine which color belongs to which shape in multi-object scenes, an ability known as concept binding. Vision–language embedding models such as CLIP struggle with binding: they recognize individual concepts but fail to represent which concepts form which objects. Although CLIP behaves like a bag-of-concepts model in cross-modal retrieval, object information is recoverable from its image and text embeddings separately. We study this tension through the binding function, which maps concepts to scene embeddings. We find that scene embeddings decompose additively into object representations, explaining why uni-modal probes can recover object information. However, CLIP’s binding function is high-complexity, which likely prevents the image and text encoders from learning a shared binding mechanism that generalizes to unseen concept combinations. We then ask whether this limitation is fundamental. We show that it is not. In controlled transformer models trained from scratch, binding generalization emerges with sufficient data coverage. These models learn lowcomplexity binding functions characterized by multiplicative interactions between concepts, enabling systematic generalization. Code is publicly available at GitHub.

# 1. Introduction

Given a picture of a scene with two objects of different colors and shapes, humans can easily determine which color is associated with which shape, an ability often referred to as concept binding (Treisman & Gelade, 1980; Treisman, 1998). While straightforward for humans, this has proved challenging for modern vision-language embedding systems such as CLIP (Radford et al., 2021). These models often behave like bag-of-concepts systems: they reliably recognize

\*Equal contribution. 1University of Tubingen, T ¨ ubingen ¨ AI Center 2KAIST AI. Email to: <arnas.uselis, darina.koishigarina@uni-tuebingen.de>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

individual concepts (e.g., red, square), yet struggle to correctly represent their combinations in multi-object scenes (e.g., distinguishing red square vs. blue square) (Yuksekgonul et al., 2023; Lewis et al., 2024; Tang et al., 2023). At the same time, within individual modalities of vision and language, object-related information can often be recovered from embeddings using probes (Koishigarina et al., 2026). In other words, despite their failures in multi-object settings, these models appear to internally encode object information to some extent. This tension motivates a closer examination of the geometry of CLIP’s binding: if object information is present, how is it represented, and why does it fail to align across modalities?

We begin by analyzing how multi-object scenes are represented in CLIP. We find that scene embeddings exhibit a clear additive structure: they decompose into sums of object representations. This structure explains why uni-modal probes can recover object information—object-level components are explicitly present in the embedding space and can be manipulated directly. However, this geometric structure alone does not explain binding. We show that CLIP’s binding function, which maps concepts to object representations, is high-complexity in the sense that it cannot be captured by a simple MLP mapping. Instead, the mapping from concepts to objects differs across concept combinations, effectively requiring memorization of object identities. This prevents generalization to unseen objects and makes it difficult for the image and text encoders to learn a shared binding mechanism, despite exhibiting uni-modal binding.

We then ask whether this limitation is fundamental. We show that it is not. In controlled transformer-based dual-encoder models trained from scratch on synthetic multi-object data, binding generalization emerges with sufficient data coverage. These models learn low-complexity binding functions that reuse structure across concept combinations and generalize to unseen objects. Finally, we analyze the structure of these generalizing binding functions. We find that models that generalize implement multiplicative interactions between concepts, rather than purely additive composition. This multiplicative structure enables generalization and explains how different modalities can produce aligned object representations for unseen concept combinations.

Together, these results show that binding failures in CLIP arise not from the absence of object structure, but from the complexity of the binding function it learns. Embedding models are capable of generalizable binding, but doing so requires learning a low-complexity, systematic concept-toobject mapping that can be shared across modalities.

# 2. Related work

Binding. The binding problem asks how models associate features that belong to the same object (Greff et al., 2020; Feng & Steinhardt, 2024). Surveys and empirical studies examine binding limits and emergent symbolic mechanisms (Campbell et al., 2024; Assouel et al., 2026; Jeong et al., 2026). In autoregressive language models, mechanistic studies have identified position-independent binding IDs that link related tokens (Feng & Steinhardt, 2024; Feng et al., 2025). Binding at patch-level of DINO and CLIP has also been studied (Li et al., 2026), showing that vision models naturally develop an ability to bind at a patch level. However, such mechanisms operate at the token level and it remains unclear how embedding models that produce a single vector per input could implement analogous structure. In contrast, we study how binding is realized in the geometry of embedding models: we ask how object representations are constructed from concepts, and argue that the complexity of this mapping explains cross-modal binding failures.

Binding in embedding models. CLIP has been observed to exhibit uni-modal binding (Koishigarina et al., 2026) but fail cross-modally (Lewis et al., 2024; Tang et al., 2023). These failures have been attributed to encoder-level weaknesses such as difficulty with fine-grained details, negations, and spatial reasoning (Tong et al., 2024b;a; Kamath et al., 2023; Li et al., 2025) and over-reliance on concept-level information (Huang et al., 2025). Several works propose fine-tuning approaches to improve CLIP’s compositional understanding, including word order sensitivity and attribute-object association (Yuksekgonul et al., 2023; Ma et al., 2023; Hsieh et al., 2024) and argue of the importance of data (Gurung et al., 2025). Some works suggest that concept recognition and object recognition (binding) are fundamentally at odds, suggesting a trade-off (Kang et al., 2025). In contrast, we show that both concept and object recognition can coexist in the same embedding space, explain how uni-modal binding can succeed via linear probes, and argue that cross-modal binding failures stem from the complexity of how concepts are combined into objects, not from an inherent incompatibility.

Compositional generalization. Research on compositional generalization investigates how models can generalize to novel combinations of known parts. Many approaches attempt to inject this ability into neural networks (Wang et al., 2024; Mahajan et al., 2025; Jarvis et al., 2024), while others study conditions under which it can be achieved (Lippl & Stachenfeld, 2025; Montero et al., 2021; Dittadi et al., 2021; Liang et al., 2025; Schott et al., 2022; Kempf et al., 2025; Okawa et al., 2023; Uselis et al., 2026). However, this literature largely focuses on single-object settings with factorized concept spaces. For CLIP specifically, generalization of concept recognition to novel combinations has been studied (Uselis et al., 2025), but generalizing binding, object-level recognition, to novel combinations remains difficult (Lewis et al., 2024). In contrast, we study multi-object scenes and find that generalization of binding is possible and emerges with data scale: at small scales, object recognition fails to generalize (consistent with prior findings), but concept recognition of novel objects remains robust; at larger scales, both concept and object recognition generalize.

Geometry of learned representations. Many works study the shape of learned features. General notions of the Linear Representation Hypothesis mostly report linearity in language models (Jiang et al., 2024; Park et al., 2023). Abbasi et al. (2024) find evidence of disentanglement in CLIP models, Trager et al. (2023) show additive decomposition in CLIP text encoder, and Uselis et al. (2025); Berasi et al. (2025) show it in vision encoder. However, these analyses typically characterize geometry in terms of single objects. In contrast, we extend this to multi-object scenes and show that scene embeddings admit a two-level additive decomposition into objects and concepts.

# 3. Formalisation of binding

Before analyzing how embedding models represent multiobject scenes, we formalize the problem. We define what it means to recognize concepts, recognize objects, and bind them together. This formalization guides §4 and §5.

We illustrate the setup in Fig. 1. We assume images and captions depict scenes. A scene contains one or more objects. Each object is specified by values of a fixed set of concepts (e.g. color and shape). We follow prior work (Okawa et al., 2023) in defining objects as combinations of concept values (Fig. 1(a)) and extend this formalisation to scenes containing multiple objects (Fig. 1(b)).

Definition 3.1 (Concept space). Let the C concepts be indexed by $i \in \{ 1 , \ldots , C \}$ , where concept i takes values in a set $\mathcal { C } _ { i }$ (often $| { \mathcal { C } } _ { i } | = V$ for all i, where V is the number of values per concept). The concept space is the Cartesian product $\mathcal { C } = \mathcal { C } _ { 1 } \times \cdots \times \mathcal { C } _ { C }$ .

An object is a vector $\pmb { o } = ( c _ { 1 } , \dots , c _ { C } ) \in \mathcal { C }$ , where $c _ { i } \in$ $\mathcal { C } _ { i }$ specifies the value for concept i. An object therefore corresponds to a particular combination of concept values.

Definition 3.2 (Scene space). Let $O _ { \mathrm { m a x } } \in$ N denote the maximum number of objects in a scene. For each $m \in$ $\{ 1 , \dots , O _ { \mathrm { m a x } } \}$ , let ${ \mathcal { C } } ^ { m }$ be the set of ordered m-tuples of objects.

(a) Concept space C   
![](images/3c6be685737fea111fd893ceed598cb436964749da2999919b4e13347b44d476.jpg)

(b)Scene space S   
![](images/c0ecc9a42c8303006a191ce4d269de40a4539cac4924520d077d9e970a9e4204.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["S = {■ ▲ ▲ ... ●} "] --> B["s = (o1, o2)"]
    B --> C["o1 = (red, square) ■"]
    B --> D["o2 = (blue, circle) ●"]
```
</details>

(c）Concept recognition   
![](images/ec0948a8fabcb1888197aa5227d1d36af6f71e5560c4449b5aaa9ac1cd757d7a.jpg)

<details>
<summary>heatmap</summary>

| x | Present | Concept | Similarity | Absent | Object | Similarity |
|---|---|---|---|---|---|---|
| ● | red | 1.0 | 1.0 | ● | 1.0 | 1.0 |
| ● | blue | 1.0 | 1.0 | ■ | 1.0 | 1.0 |
| ● | □ | 1.0 | 1.0 | ● | 1.0 | 1.0 |
| ● | ○ | 1.0 | 1.0 | ● | 1.0 | 1.0 |
| △ | Present | 0.0 | 0.0 | ● | 0.0 | 0.0 |
| △ | Absent | 0.0 | 0.0 | ● | 0.0 | 0.0 |
| △ | Green | 0.0 | 0.0 | ■ | 0.0 | 0.0 |
The image contains a small diagram of three colored circles (red, blue, green) above the grid, but no explicit numerical values are provided for the data.
</details>

Object recognition   
Figure 1. Schematic of the binding setup. (a) Example concept space C (e.g., color × shape), where each object is a tuple of concept values. (b) Example scene space S, where a scene s is a tuple of objects (here, two objects $( o _ { 1 } , o _ { 2 } ) )$ . (c) Example of the two recognition criteria: concept recognition ranks present concept values above absent ones (Def. 3.3), while object recognition ranks present objects above absent concept combinations (Def. 3.4).

The scene space is

$$
\mathcal {S} = \bigcup_ {m = 1} ^ {O _ {\max}} \mathcal {C} ^ {m}
$$

A scene is a tuple $\pmb { \mathscr { s } } = ( \pmb { o } _ { 1 } , \pmb { o } _ { 2 } , . . . , \pmb { o } _ { m } ) \in \mathcal { S } ,$ , describing a collection of m objects. We assume that each scene s corresponds to some datapoint $\pmb { x _ { s } } \in \mathcal { X }$ (e.g. an image).

Models. We study embedding models that encode inputs and queries independently. An input corresponds to a scene, such as an image, while a query specifies the information to retrieve, such as a concept value or an object (e.g. CLIP).

We denote the input encoder by $f : \mathcal { X } \to \mathbb { R } ^ { d }$ and the query encoder by $q : \mathcal { V } \to \mathbb { R } ^ { d }$ . Given an input $\scriptstyle { \mathbf { { \mathbf { x } } } } _ { s }$ and a query $\textbf {  { y } } \in \mathcal { V }$ , the matching score is given by cosine similarity, $\begin{array} { r } { s ( \pmb { x } _ { s } , \pmb { y } ) = \frac { f ( \pmb { x } _ { s } ) ^ { \top } q ^ { \top } ( \pmb { y } ) } { \| f ( \pmb { x } _ { s } ) \| \| q ( \pmb { y } ) \| } } \end{array}$ . We refer to $( f , q )$ as a model.

For a scene $\pmb { \mathscr { s } } ~ = ~ ( \mathbf { o } ^ { ( 1 ) } , \dots , \mathbf { o } ^ { ( m ) } ) ~ \in ~ \mathcal { S }$ , we define the set of values taken by concept i across all objects in the scene as $V _ { i } ( s ) \subsetneq \{ c _ { i } ^ { ( 1 ) } , \ldots , c _ { i } ^ { ( m ) } \} \subseteq { \mathcal { C } } _ { i }$ {c(1)i , . c(m)i } ⊆ Ci. We also define the set of objects present in the scene as $\begin{array} { r l } { O ( s ) } & { { } : = } \end{array}$ $\{ \mathbf { o } ^ { ( 1 ) } , \ldots , \mathbf { o } ^ { ( m ) } \} \subseteq \mathcal { C }$ .

Binding ability. A model exhibits binding ability when it has (Fig. 1(c)): (1) concept-level recognition: recognize which concept values are present in a scene, (2) object-level recognition: recognize which objects are present in a scene.

Definition 3.3 (Concept recognition). A model $( f , q )$ recognizes concepts if for any scene $s \in S$ and any concept $i \in [ C ]$ , the model assigns higher scores to all concept values present in the scene than to any absent ones:

$$
\min _ {v \in V _ {i} (\boldsymbol {s})} s \left(\boldsymbol {x} _ {\boldsymbol {s}}, \boldsymbol {y} _ {i, v}\right) > \max _ {v ^ {\prime} \in \mathcal {C} _ {i} \backslash V _ {i} (\boldsymbol {s})} s \left(\boldsymbol {x} _ {\boldsymbol {s}}, \boldsymbol {y} _ {i, v ^ {\prime}}\right). \tag {1}
$$

While concept recognition captures the presence of individual concept values, it does not test whether the model can identify which values belong together. This distinction is formalized by object recognition.

Definition 3.4 (Object recognition). A model $( f , q )$ recognizes objects if for any scene $s \in { \mathcal { S } }$ , the model assigns higher scores to all objects present in the scene than to any objects not present:

$$
\min _ {\mathbf {o} \in O (\boldsymbol {s})} s \left(\boldsymbol {x} _ {\boldsymbol {s}}, \boldsymbol {y} _ {\mathbf {o}}\right) > \max _ {\mathbf {o} ^ {\prime} \in \mathcal {C} \backslash O (\boldsymbol {s})} s \left(\boldsymbol {x} _ {\boldsymbol {s}}, \boldsymbol {y} _ {\mathbf {o} ^ {\prime}}\right). \tag {2}
$$

Concept recognition and object recognition capture complementary aspects of how a model represents scenes. Binding concerns the ability to support both levels consistently.

Definition 3.5 (Binding). A model exhibits binding if it satisfies both concept recognition (Def. 3.3) and object recognition (Def. 3.4), i.e., it can identify which concept values are present in a scene and which values belong together.

The definitions above characterize binding as a behavioral property. We distinguish cross-modal binding, where the model $( f , q )$ directly satisfies concept and object recognition via cosine similarity, from uni-modal binding, where a probe must be trained on top of embeddings within a single modality. Prior work has shown that CLIP exhibits unimodal binding but fails cross-modally (Koishigarina et al., 2026). To understand why, we examine how embeddings are constructed from scenes.

Definition 3.6 (Binding function). For a dual-encoder model $( f , q )$ , we define the binding functions $B _ { \mathrm { i m g } } : S $ $\mathbb { R } ^ { d }$ and $B _ { \mathrm { t x t } } : S  \mathbb { R } ^ { d }$ as

$$
B _ {\mathrm{img}} (\boldsymbol {s}) := f \left(\boldsymbol {x} _ {\boldsymbol {s}}\right), \quad B _ {\mathrm{txt}} (\boldsymbol {s}) := q \left(\boldsymbol {y} _ {\boldsymbol {s}}\right), \tag {3}
$$

mapping scenes to embeddings within each modality.

Intuitively, cross-modal binding requires $B _ { \mathrm { i m g } }$ and $B _ { \mathrm { t x t } }$ to produce compatible embeddings for the same scene, including novel ones. By Occam’s razor, among functions consistent with the training scenes, those with simpler, more reusable structure are more likely to generalize identically off-domain (Elmoznino et al., 2025): if both binding functions admit a low-complexity, compositional rule, fewer mappings are consistent with the observations, so the two encoders are more likely to converge on the same rule and remain aligned on unseen objects. If they are high-complexity, each modality may instead fit the training domain in a combination-specific way and disagree off-domain, preventing cross-modal alignment.

In the following sections, we first study the geometry of CLIP’s binding functions: how scene embeddings decompose into objects and concepts (§4). We then investigate whether these functions can be approximated using concept indices and study controlled models in which binding generalizes (§5).

![](images/6ca61f48511dde3cc62101d1a9327044d409d9bd553a49aacf4bddc76e0592ed.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["BSRS"] --> B["BS+RS"]
    A --> C["BS"]
    A --> D["BSBS"]
    E["BSRC"] --> F["BC+RS"]
    E --> G["BC+RC"]
    H["BCRRC"] --> I["RC"]
    J["BC+RC"] --> K["RC+RC"]
    L["BC+RC"] --> M["BC"]
    N["BC+RC"] --> O["BC+RC"]
    P["BSBC"] --> Q["BS+RC"]
    R["BS+RS"] --> S["BS+RS"]
    T["BS+RS"] --> U["BS"]
    V["BS+RS"] --> W["BS"]
    X["BS+RS"] --> Y["BS"]
    Z["BS+RS"] --> AA["BS"]
    AB["BS+RS"] --> AC["BS"]
    AD["BS+RS"] --> AE["BS"]
    AF["BS+RS"] --> AG["BS"]
    AH["BS+RS"] --> AI["BS"]
    AJ["BS+RS"] --> AK["BS"]
    AL["BS+RS"] --> AM["BS"]
    AN["BS+RS"] --> AO["BS"]
    AP["BS+RS"] --> AQ["BS"]
    AR["BS+RS"] --> AS["BS"]
    AT["BS+RS"] --> AU["BS"]
    AV["BS+RS"] --> AW["BS"]
    AX["BS+RS"] --> AY["BS"]
    AZ["BS+RS"] --> BA["BS"]
    BB["BS+RS"] --> BC["BS"]
    BD["BS+RS"] --> BE["BS"]
    BF["BS+RS"] --> BG["BS"]
    BH["BS+RS"] --> BI["BS"]
    BJ["BS+RS"] --> BK["BS"]
    BL["BS+RS"] --> BM["BS"]
    BN["BS+RS"] --> BO["BS"]
    BP["BS+RS"] --> BQ["BS"]
    BR["BS+RS"] --> BSQ["BS"]
    BT["BS+RS"] --> BU["BS"]
    BV["BS+RS"] --> BW["BS"]
    BX["BS+RS"] --> BYQ["BS"]
    BZ["BS+RS"] --> CA["BS"]
    CB["BS+RS"] --> CC["BS"]
    DD["BS+RS"] --> DJ["BS"]
    DK["BS+RS"] --> DL["BS"]
    DV["BS+RS"] --> DW["BS"]
    DX["BS+RS"] --> DXQ["BS"]
    DXQ --> DXQ
    DXQ --> DXQ
    DXQ --> DXQ
    DXQ --> DXQ
    DXQ --> DXQ
    DXQ --> DXQ
    DXQ --> DXQ
    DXQ --> DXQ
    DXQ --> DXQ
    DXQ --> DXQ
    DXQ --> DXQ
    DXQ --> DXQ
    DXQ --> DXQ
    DXQ --> DXQ
    DXQ --> DXQ
    DG["BCRRC"] --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DG --> DC
    DN["R-S"] --> RS
    DN --> RS + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + MC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    DN --> RC + RC
    
    style A fill:#f9f,stroke:#333
    style B fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style J fill:#f9f,stroke:#333
    style K fill:#f9f,stroke:#333
    style L fill:#f9f,stroke:#333
    style M fill:#f9f,stroke:#333
    style N fill:#f9f,stroke:#333
    style O fill:#f9f,stroke:#333
```
</details>

Figure 2. Additive structure in two-object scene embeddings. MDS projection of CLIP embeddings for single- and two-object scenes that vary in color and shape (distances in the plot approximate embedding distances). Labels use $R / B$ for red/blue and $C / S$ for cube/sphere; points correspond to embeddings of the associated single concepts, single-object scenes, and their two-object combinations. Two-object scene embeddings lie near the line segments between the corresponding single-object embeddings; the highlighted triplet (blue outline) illustrates this geometry for one pair of single-object scenes and their combined two-object scene.

# 4. Geometry of embedding models’s binding

CLIP embeddings have been shown to encode binding information within each modality, while this information is misaligned across modalities (Koishigarina et al., 2026). However, it remains unclear how this binding information is represented in the embedding space and how it becomes accessible to probes. To address this, we examine the geometry of CLIP embeddings and find that scene embeddings decompose into object components, which further decompose into concept components (§4.1). We show that these object components are directly responsible for object recognition (§4.2), explaining how binding information can be recovered uni-modally. While prior work mostly focused on single-object representations (Trager et al., 2023; ?; Berasi et al., 2025), we study embeddings in the multi-object setting. Metric definitions are provided in §C.2. Descriptions of the visual datasets and text compositions are given in §C.1 and §C.3.

# 4.1. Embeddings decompose into objects and concepts

To gain intuition about the geometry of CLIP embeddings, we visualize them in Fig. 2. It shows an MDS (Borg & Groenen, 2005) projection of embeddings for two-object scenes. Two-object scene embeddings lie roughly between the embeddings of the corresponding single-object scenes, suggesting that scene embeddings are approximately sums of their constituent object embeddings. Motivated by this geometric structure, we study how multi-object scenes are represented in CLIP. We consider scenes with two objects. Throughout the section, we will let $\pmb { s } = ( o _ { 1 } , o _ { 2 } )$ denote a scene and $\scriptstyle { \mathbf { { \mathit { x } } } } _ { s }$ denote the corresponding input.

Table 1. Estimators for object embeddings $\mathbf { \Delta } \mathbf { u } _ { o }$ and concept embeddings ${ \textbf { \em u } } _ { c } .$ Each method averages the image encoder $f$ over a subset of scenes $s$ containing the target object or concept value. The AVG+POS variant conditions on the position k of the object in the scene tuple $\pmb { s } = \left( \pmb { o } _ { 1 } , \dots , \pmb { o } _ { m } \right)$ . 

<table><tr><td>Method</td><td>Subset</td><td>Estimator</td></tr><tr><td>AVG</td><td> $\mathcal{D}_{o} = \{ s \in \mathcal{S} : o \in O(s) \}$ </td><td> $u_{o}^{(\text{avg})} = \mathbb{E}_{s \sim \mathcal{D}_{o}}[f(x_{s})]$ </td></tr><tr><td>AVG+POS</td><td> $\mathcal{D}_{o}^{(k)} = \{ (o_{1}, \ldots, o_{m}) \in \mathcal{S} : o_{k} = o \}$ </td><td> $u_{o,k}^{(\text{avg+pos})} = \mathbb{E}_{s \sim \mathcal{D}_{o}^{(k)}}[f(x_{s})]$ </td></tr><tr><td>SINGLE-OBJ</td><td> $\mathcal{D}_{o}^{(1)} = \{ s \in \mathcal{S} : s = (o) \}$ </td><td> $u_{o}^{(\text{single-obj})} = \mathbb{E}_{s \sim \mathcal{D}_{o}^{(1)}}[f(x_{s})]$ </td></tr><tr><td>CONCEPT</td><td> $\mathcal{D}_{c} = \{ s \in \mathcal{S} : c \in V_{i}(s) \}$ </td><td> $u_{c} = \mathbb{E}_{s \sim \mathcal{D}_{c}}[f(x_{s})]$ </td></tr></table>

We hypothesize that the embedding space exhibits a hierarchical additive structure, where scenes decompose into objects and objects further decompose into concepts:

1. (Level-I) A scene embedding approximately decomposes into a sum of object embeddings.   
2. (Level-II) An object embedding approximately decomposes into a sum of concept embeddings.

Under this hypothesis, the embedding of a two-object scene can be written as follows. Writing each object as a tuple of concept values, $\pmb { o } _ { 1 } ~ = ~ ( c _ { 1 , 1 } , \ldots , c _ { 1 , C } )$ and $_ { 2 } =$ $( c _ { 2 , 1 } , \ldots , c _ { 2 , C } )$ :

$$
f (\boldsymbol {x} _ {\boldsymbol {s}}) \approx \boldsymbol {u} _ {\boldsymbol {o} _ {1}} + \boldsymbol {u} _ {\boldsymbol {o} _ {2}} \approx \sum_ {i = 1} ^ {C} \boldsymbol {u} _ {c _ {1, i}} + \sum_ {i = 1} ^ {C} \boldsymbol {u} _ {c _ {2, i}}. \tag {4}
$$

We estimate object and concept embeddings using the methods summarized in Tab. 1. We empirically evaluate both levels of this decomposition below.

Scenes decompose into object components (Level-I). We test the first claim of the Level-I decomposition: that scene embeddings decompose into object-level components. We estimate object embeddings $\mathbf { \Delta } \mathbf { u } _ { o }$ according to Tab. 1. We then reconstruct scene embeddings by summing the corresponding object components. We evaluate the reconstruction using the retrieval score, probing accuracy, and $R ^ { 2 }$ .

Tab. 2 shows that CLIP scene embeddings are well approximated by sums of object components (text: $R ^ { 2 } = { }$ 0.90/0.92; image PUG:SPARE CLIP: $R ^ { 2 } = 0 . 7 5 / 0 . 8 4 )$ , supporting the proposed object-level decomposition. The same decomposition also extends to 3-object CLEVR scenes (including with occlusions; §D.3) and to natural images generated by Gemini Nano Banana 2 (§D.4).

Editing scene embeddings via object replacement. The Level-I decomposition in (4) has a concrete implication. If a scene embedding is well approximated by a sum of object components, then editing a multi-object embedding by removing one object component and inserting another should produce a meaningful counterfactual representation.

Table 2. Object-based reconstruction preserves most of the scene variance and retains predictive structure. We report $R ^ { 2 }$ , retrieval accuracy, and probing accuracy across models. Entries for $R ^ { 2 }$ are AVG / AVG+POS; retrieval and probing are AVG+POS only. 

<table><tr><td>Dataset</td><td>Model</td><td> $R^{2}$ </td><td>Retrieval</td><td>Probing</td></tr><tr><td rowspan="3">Text</td><td>Guess</td><td>—</td><td>1 / 160K</td><td>1 / 400</td></tr><tr><td>Random</td><td>0.47 / 0.69</td><td>0.42</td><td>0.82</td></tr><tr><td>CLIP</td><td>0.90 / 0.92</td><td>0.97</td><td>0.99</td></tr><tr><td rowspan="3">PUG:SPARE</td><td>Random</td><td>0.47 / 0.53</td><td>0.25</td><td>0.90</td></tr><tr><td>CLIP</td><td>0.75 / 0.84</td><td>0.93</td><td>0.98</td></tr><tr><td>DINOv2</td><td>0.78 / 0.86</td><td>0.86</td><td>0.98</td></tr><tr><td rowspan="2">CLEVR-2D</td><td>CLIP</td><td>0.75 / 0.77</td><td>0.82</td><td>0.99</td></tr><tr><td>DINOv2</td><td>0.78 / 0.79</td><td>0.68</td><td>0.97</td></tr><tr><td rowspan="2">CLEVR</td><td>CLIP</td><td>0.78 / 0.83</td><td>0.94</td><td>0.96</td></tr><tr><td>DINOv2</td><td>0.85 / 0.88</td><td>0.86</td><td>0.96</td></tr></table>

Table 3. Object-level decomposition enables controlled edits of scene embeddings. Probing and retrieval performance under object replacement interventions. Each entry corresponds to AVG, AVG+POS, and SINGLE-OBJ object embeddings. 

<table><tr><td>Dataset</td><td>Model</td><td>Probing</td><td>Retrieval</td></tr><tr><td>Text</td><td>CLIP</td><td>0.99 / 0.99 / 0.69</td><td>0.99 / 0.99 / 0.96</td></tr><tr><td rowspan="2">CLEVR</td><td>CLIP</td><td>0.98 / 0.98 / 0.86</td><td>1.00 / 1.00 / 0.97</td></tr><tr><td>DINO</td><td>0.97 / 0.98 / 0.77</td><td>0.89 / 0.95 / 0.86</td></tr><tr><td rowspan="2">CLEVR-2D</td><td>CLIP</td><td>0.98 / 0.98 / 0.92</td><td>0.99 / 0.99 / 0.97</td></tr><tr><td>DINO</td><td>0.97 / 0.98 / 0.72</td><td>0.97 / 0.97 / 0.88</td></tr><tr><td rowspan="2">PUG:SPARE</td><td>CLIP</td><td>0.94 / 0.95 / -</td><td>0.86 / 0.94 / -</td></tr><tr><td>DINO</td><td>0.97 / 0.97 / -</td><td>0.48 / 0.76 / -</td></tr></table>

We examine this through linear operations in embedding space as depicted in Fig. 3.

![](images/e926e3b8b1f7e9129cc29951cbed067e92d94675c049803afe1aeedb50253242.jpg)

<details>
<summary>text_image</summary>

f( ) - f( ) + f( ) ≈ f( )
</details>

Figure 3. Scene embeddings support object-level editing via linear operations in embedding space. Subtracting one object embedding and adding another produces a counterfactual embedding corresponding to the edited scene.

Setup. Let $\pmb { s } = ( \pmb { o } _ { 1 } , \pmb { o } _ { 2 } )$ be a two-object scene, and let ${ \pmb o } _ { 1 } ^ { \prime }$ be a counterfactual object obtained by changing one concept value of $\mathbf { o } _ { 1 }$ (e.g. color). We construct an edited embedding by substituting the object component of $\scriptstyle { o _ { 1 } }$ with that of ${ \pmb o } _ { 1 } ^ { \prime }$ :

$$
\tilde {z} := f (x _ {s}) - u _ {o _ {1}} + u _ {o _ {1} ^ {\prime}}. \tag {5}
$$

We evaluate edited embeddings z˜ with retrieval and probing metrics; full details are in §D.2. We report results for text encoders (§C.3) and for image encoders on CLEVR, CLEVR-2D, and PUG:SPARE (§C.1).

Results. As shown in Tab. 3, replacing an object in a multiobject scene embedding with another object produces embeddings that behave like the intended counterfactual scenes under both probing and retrieval (e.g., CLIP on CLEVR achieves 1.00 retrieval with scene-averaged object embeddings). Notably, object embeddings obtained from singleobject scenes can be directly used to edit multi-object scene embeddings (e.g., retrieval accuracy 0.97 for CLEVR and CLEVR-2D with CLIP). This supports the Level-I decomposition hypothesis that scene embeddings are composed of additive object-level components.

Objects decompose into concepts (Level-II). Having established scene-level decomposition, we now analyse whether objects themselves decompose into concepts as in (4). A purely additive concept-level decomposition corresponds to a bag-of-concepts model, which cannot encode within-object associations required for binding. We therefore quantify how much variance is explained by object-only and concept-only components using $R ^ { 2 }$ .

Table 4. Concepts explain most scene variance, but objects capture additional structure. We report $R ^ { 2 }$ for predicting (i) the scene embedding from only objects and (ii) the object and scene embeddings from only concepts, for PUG:SPARE. Values are shown for AVG / AVG+POS object embeddings. 

<table><tr><td>Model</td><td>Scenes from objects  $R^{2}$ </td><td>Objects from concepts  $R^{2}$ </td><td>Scenes from concepts  $R^{2}$ </td></tr><tr><td>Random-embeds</td><td>0.01</td><td>0.01</td><td>-0.03</td></tr><tr><td>Random-init (txt)</td><td>0.47 / 0.69</td><td>0.56 / 0.80</td><td>0.39 / 0.56</td></tr><tr><td>CLIP-B/32 (txt)</td><td>0.90 / 0.92</td><td>0.88 / 0.90</td><td>0.83 / 0.84</td></tr><tr><td>Random-init (img)</td><td>0.47 / 0.53</td><td>0.25 / 0.57</td><td>0.14 / 0.34</td></tr><tr><td>CLIP-B/32 (img)</td><td>0.75 / 0.84</td><td>0.35 / 0.83</td><td>0.30 / 0.71</td></tr></table>

Results. From Tab. 4, we observe that a bag-of-concepts representation explains a large fraction of embedding variance, up to 84% for the text encoder and 71% for the image encoder. Representing scenes as bags of objects explains roughly 10% more variance. Thus, most variance is captured by concepts, with object representations contributing additional structure beyond a bag-of-concepts.

Takeaway §4.1: Scene embeddings decompose into object components, which explain most of the variance and support object edits that remain consistent under probing and retrieval.

# 4.2. Uni-modal binding is explained by object decomposition

In the previous section, we showed that CLIP embeddings decompose into object and concept components. In this section, we demonstrate that these components are directly responsible for the corresponding object and concept recognition in scene embeddings.

Setup. We perform targeted component removal on the scene embedding and then fit linear probes on the residuals. Using the decomposition in (4), we define the concept part of a scene as $\begin{array} { r } { \sum _ { i } \pmb { u } _ { c _ { 1 , i } } + \sum _ { i } \pmb { u } _ { c _ { 2 , i } } } \end{array}$ , and the object part as $\pmb { u _ { o _ { 1 } } } + \pmb { u _ { o _ { 2 } } }$ . We then form intervened embeddings by subtracting these components, e.g. $\begin{array} { r } { f ( \pmb { x } _ { s } ) - \sum _ { i } \pmb { u } _ { c _ { 1 , i } } - \sum _ { i } \pmb { u } _ { c _ { 2 , i } } } \end{array}$ i (remove concepts) or $f ( \pmb { x } _ { s } ) - \pmb { u } _ { o _ { 1 } } - \pmb { u } _ { o _ { 2 } }$ (remove objects), and train probes on the resulting residual embeddings. As a control, we repeat the same removals using permuted-ID concept/object components (details in §D.1).

![](images/1c86180bc549f936018da26e7a75b8ef71a5ca781ca797bf8dcbb29b9a4c5fae.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input x"] --> B["OR &quot;A red sphere and a blue cube&quot;"]
    B --> C["VLM f(·) (e.g CLIP)"]
    C --> D["f(x)"]
    D --> E["min g ||f(x) - g(o1, o2)||2"]
    E --> F["Binding approximator g(·)"]
    F --> G["g(o1, o2)"]
    H["Scene description (O1, O2)"] --> I["Object 1: O1 = [c11, c21"], Object 2: O2 = [c21, c22]]
    I --> J["Embedder"]
    J --> K["MLP"]
    K --> L["Linear"]
    L --> M["g(o1, o2)"]
```
</details>

(b)   
![](images/80b58b943862acce0f7fdd303db70b005cd0ceb18aae0d0ebc7e05a8ac4ec92a.jpg)

<details>
<summary>line</summary>

| Fraction of data | CLIP B/32 (image) - Objects | CLIP B/32 (image) - Concepts | DINOv2 B/16 (image) - Objects | DINOv2 B/16 (image) - Concepts | CLIP B/32 (text) - Objects | CLIP B/32 (text) - Concepts |
| ---------------- | --------------------------- | ---------------------------- | ----------------------------- | ----------------------------- | -------------------------- | -------------------------- |
| 0.1              | ~0.0                        | ~0.0                         | ~0.0                          | ~0.0                          | ~0.0                       | ~0.0                       |
| 0.3              | ~0.0                        | ~0.8                         | ~0.0                          | ~0.8                          | ~0.0                       | ~0.8                       |
| 0.5              | ~0.0                        | ~0.9                         | ~0.0                          | ~0.9                          | ~0.0                       | ~0.9                       |
| 0.7              | ~0.0                        | ~0.95                        | ~0.0                          | ~0.95                         | ~0.0                       | ~0.95                      |
| 0.9              | ~0.1                        | ~0.98                        | ~0.0                          | ~0.98                         | ~0.0                       | ~0.98                      |
</details>

Figure 4. CLIP’s binding function is high-complexity. (a) We train a binding approximator $g ( o _ { 1 } , o _ { 2 } )$ ) (a single-layer MLP) to predict CLIP scene embeddings f(x) from concept indices describing the objects present, minimizing (6). (b) Maximum accuracy achieved across all MLP capacities and training coverages for concept vs. object recognition on scenes composed of held-out objects. Predicting concepts is high, confirming that additive structure suffices to capture concept-level information. Object recognition is worse, indicating that predicting how concepts combine into objects requires high model complexity.

Results. Tab. 5 demonstrates that concept recognition and object recognition in scenes depend on their corresponding components in the scene embeddings. For CLIP-B/32, subtracting concept components nearly eliminates concept decoding (text: 1.00 → 0.06, image: $0 . 9 4  0 . 0 5 )$ ) while largely preserving object decoding (text: 1.00 → 0.99, image: $0 . 9 6  0 . 8 5 )$ . In contrast, subtracting object components collapses both object and concept decoding (text: 0.04/0.05, image: 0.01/0.02). Permuted-ID controls do not produce comparable degradation. Removing object components that are not present in the scene has little effect on decoding accuracy, confirming that the performance drop is specific to the correct object component.

These results explain why prior work observes binding in CLIP embeddings. Object components in a scene embedding encode concept information jointly, allowing probes to recover the correct combinations of concepts.

Takeaway §4.2: Object components in scene embeddings are responsible for object recognition, which explains why the binding is observed uni-modally via probes.

# 5. Binding generalization

Having established that scene embeddings decompose into objects, explaining uni-modal binding, we next examine why cross-modal binding fails through the complexity and generalization of the underlying binding functions.

# 5.1. CLIP’s binding function does not generalize

Inspired by prequential/MDL perspectives that interpret simplicity as fast generalization from limited data (Blier & Ollivier, 2018; Elmoznino et al., 2025), we measure how generalization of a binding function (Def. 3.6) depends on training-set size and MLP capacity (width). Concretely, we study the sample-capacity requirements of a 1-hidden-layer ReLU MLP whose width is varied, trained to approximate a binding map from discrete concept values to scene embeddings. We use MLPs because SGD-trained MLPs favor simple, compressible solutions (Wilson, 2025), so if a simple compositional binding rule exists, we expect this approximator family to find it.

Table 5. Removing object or concept components reduces recognition accuracy for the corresponding prediction task. We report probe accuracy after subtracting components from the embedding (remove concepts/objects) or their permuted-ID controls, for both text and image encoders and their random-init counterparts. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Intervention</td><td colspan="2">Text</td><td colspan="2">Image</td></tr><tr><td>Conc.</td><td>Obj.</td><td>Conc.</td><td>Obj.</td></tr><tr><td rowspan="5">Random</td><td>None</td><td>0.94</td><td>0.60</td><td>0.76</td><td>0.71</td></tr><tr><td>Subtract concepts</td><td>0.02</td><td>0.26</td><td>0.06</td><td>0.66</td></tr><tr><td>Subtract objects</td><td>0.01</td><td>0.01</td><td>0.04</td><td>0.00</td></tr><tr><td>Permute concepts</td><td>0.87</td><td>0.68</td><td>0.74</td><td>0.72</td></tr><tr><td>Permute objects</td><td>0.83</td><td>0.93</td><td>0.78</td><td>0.88</td></tr><tr><td rowspan="5">CLIP-B/32</td><td>None</td><td>1.00</td><td>1.00</td><td>0.94</td><td>0.96</td></tr><tr><td>Subtract concepts</td><td>0.06</td><td>0.99</td><td>0.05</td><td>0.85</td></tr><tr><td>Subtract objects</td><td>0.05</td><td>0.04</td><td>0.02</td><td>0.01</td></tr><tr><td>Permute concepts</td><td>0.92</td><td>0.99</td><td>0.99</td><td>0.97</td></tr><tr><td>Permute objects</td><td>0.96</td><td>1.00</td><td>0.86</td><td>0.92</td></tr></table>

Setup. We train an approximator $g : \mathcal { C } ^ { 2 }  \mathbb { R } ^ { d }$ to predict scene embeddings from concept indices (Fig. 4(a)). For a scene $\pmb { \mathscr { s } } = \left( \pmb { o } _ { 1 } , \pmb { o } _ { 2 } \right)$ where each object $o _ { i } = ( c _ { i 1 } , c _ { i 2 } )$ is specified by concept values, we minimize

$$
\min _ {g} \sum_ {\boldsymbol {s} \in \mathcal {S} _ {\text { train }}} \| f (\boldsymbol {x} _ {\boldsymbol {s}}) - g (\boldsymbol {o} _ {1}, \boldsymbol {o} _ {2}) \| ^ {2}, \tag {6}
$$

where $f ( x _ { s } )$ is the CLIP embedding of scene s and $ { S } _ { \mathrm { t r a i n } }$ is the set of training scenes. We vary (1) MLP hidden layer width {64, 256, 1024, 4096}, and (2) the fraction of objects seen during training $\{ 1 0 \% , \ldots , 9 0 \% \}$ . At test time, we evaluate on scenes containing only held-out objects. We measure whether the predicted embeddings support both concept recognition (Def. 3.3) and object recognition (Def. 3.4) by using the linear probes trained on the scene embeddings. See details in §C.4.

![](images/bb9b99e06e8dc84008263a6f58a1d5570a6650d0e61e22a7c7fda5fdf8c4363f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Training fraction"] --> B["Concept 1"]
    A --> C["Concept 2"]
    A --> D["..."]
    A --> E["Concept 3"]
    A --> F["..."]
    A --> G["Concept 1"]
    A --> H["Concept 2"]
    I["Training scenes"] --> J["O1 X O2"]
    K["Testing scenes"] --> L["O1 X O2"]
```
</details>

![](images/6744b0b5388d7611182b59f2f25a06d6a60da3a45f3f0cbdf4da9d969180d9e0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Objects"] --> B["&quot;A [1,2,3"]"]
    A --> C["&quot;A [1,1,3"]"]
    B --> D["Text encoder"]
    C --> E["Text encoder"]
    D --> F["t(o[1,2,3"])]
    E --> G["t(c[1,-,-"])]
    F --> H["t(o[1,1,3"])]
    G --> I["t(c[-,2,-"])]
    J["Concepts"] --> K["&quot;A [1,1,3"] and_a["0,0,1"]"]
    J --> L["&quot;A [1,2,3"] and_a["3,0,1"]"]
    K --> M["Vision encoder"]
    L --> N["v(s[1,1,3"],[0,0,1])]
    N --> O["v(s[1,2,3"],[3,0,1])]
    P["(d)"] --> Q["t(o[1,2,3"]) t(o["1,1,3"]) t(c["1,-,-"]) t(c["-,2,-"])]
    Q --> R["0 1 1 0"]
    Q --> S["1 0 1 0"]
```
</details>

Figure 5. Controlled setup for studying generalizable binding. We train transformer-based embedding models on synthetic multi-object data to test whether binding can generalize to entirely unseen objects. (a) Data design: We vary the training coverage $\rho _ { \mathrm { t r a i n } }$ from 0.1 to 0.9, controlling what fraction of the object space the model observes during training. (b) Scene construction: Training scenes are composed of objects from the training split (blue); test scenes are composed of objects from the held-out split (red). Crucially, test objects have concept configurations that never appeared during training. (c) Encoder design: Following CLIP’s architecture, we use two independent encoders: a “text” encoder that embeds individual objects or concepts, and a “vision” encoder that embeds full scenes. Both produce embeddings in $\mathbb { R } ^ { 5 1 2 }$ . (d) Training objective: We optimize a contrastive retrieval loss using cosine similarity. The table shows a training batch where rows are scene embeddings v(s) and columns are object/concept embeddings t(·); green indicates matching pairs, red indicates mismatches.

Results. Fig. 4(b) reports the best accuracy across the MLP configurations. Concept recognition yields ≥ 80% accuracy among all conditons where the fraction of data exceeds 0.3, whereas object recognition is substantially lower, including at large widths and high training coverage. This gap indicates that correctly predicting object-level structure is markedly harder than concept-level structure, suggesting a higher effective complexity of object-level binding under this probe family. Among encoders, the text encoder performs slightly better than the image encoders; CLIP image embeddings outperform DINO but remain far below concept recognition, never exceeding 20% accuracy. The result does not depend on the approximator family: swapping MLPs for XGBoost or Random Forest yields a similar qualitative pattern on CLIP and DINOv2 (§D.6). These results indicate that CLIP’s binding function is high-complexity: it resists approximation even by high-capacity MLPs and fails to generalize to novel object combinations.

Takeaway §5.1: For pretrained embedding models, even highcapacity MLPs fail to generalize to unseen concept combinations. This may explain cross-modal binding failures: a complex, non-generalizing mapping prevents encoders from aligning on novel objects.

# 5.2. It is possible to have a generalizable binding model

The previous section showed that CLIP’s binding function is hard to approximate and does not generalize to unseen concept combinations. This raises a natural question: is this failure inherent, or can embedding models learn binding that does generalize when the data and training signal are controlled? To answer this, we train CLIP-style dual-encoder transformers on synthetic multi-object scenes where we can precisely vary object-space coverage (Fig. 5).

Setup. We illustrate the process in Fig. 5. We use synthetic scenes with up to 2 objects; each object is defined by C concepts with V values $( | O | = V ^ { C } )$ , and we vary C, V , and the training fraction of seen objects from 0.1 to 0.9. Train scenes draw only from the training split; test scenes are composed entirely of held-out objects. We use two transformer (Vaswani et al., 2023) encoders (query and scene); we refer to them as “text” and “vision” by analogy only since both process tokenized sequences, differing only in whether they encode object/concept queries or full scenes. Both encoders output embeddings in $\mathbf { \mathbb { R } ^ { 5 1 2 } }$ and have ≈20M parameters each. We train end-to-end with AdamW (Loshchilov & Hutter, 2017) on a contrastive retrieval objective (Radford et al., 2021); see §B for full details.

Metrics. We evaluate both concept and object recognition. For concepts, we test against all $C \cdot V$ values. For objects, testing against all $\bar { V } ^ { C }$ is infeasible; instead, we use $\mathrm { S w a p } ( o _ { 1 } , o _ { 2 } )$ , objects formed by exchanging concept values between $\scriptstyle { o _ { 1 } }$ and $\mathbf { o } _ { 2 }$ , which directly tests binding by ranking correct object identities against negatives.

Results. We observe two key findings (Fig. 6). First, compositional generalization emerges: object recognition on heldout objects starts near the bag-of-concepts baseline but rises sharply with coverage, reaching near-perfect accuracy despite all test objects being unseen. Second, concepts are eas-

![](images/5b198e54b90ad1cb40c0ddc9d9a8cb0cb6247e73093a7c63243a01e302119dc5.jpg)  
Concepts . Objects Bag of Concepts

Figure 6. Binding generalization emerges with scale. Test accuracy on held-out objects as a function of training coverage. Each panel varies object complexity (C concepts, V values, $| \boldsymbol { \bar { O } } | = V ^ { C }$ objects). Concept recognition (orange) generalizes readily; object recognition (blue; binding) requires more coverage but reaches high accuracy, surpassing the bag-of-concepts baseline (dashed).

ier than objects: concept recognition generalizes early, while object recognition requires substantially more coverage. When the number of possible objects is small, generalization requires observing more objects, consistent with prior findings in low-combinatorial regimes (Lewis et al., 2024). As the number of possible objects increases, the fraction of objects required for reliable object recognition decreases substantially. For example, when $| O | = 4 0 0$ , high accuracy is achieved only after observing roughly 50% of objects, whereas for $| O | \geq 2 { , } 5 0 0$ , generalization emerges at around 30% coverage. In the largest setting $( | O | = 1 2 5 , 0 0 0 )$ , increasing coverage from 30% to 40% yields a sharp transition from below-chance to near-perfect object recognition. Notably, this occurs despite the corresponding scene space being extremely large (e.g., $( 5 0 ^ { 3 } ) ^ { 2 }$ ≈ 15 billion two-object scenes for $C = 3 , V = 5 0 )$ .

Takeaway §5.2: When models are trained from scratch on sufficient data, binding does generalize: object recognition (binding) can approach perfect accuracy.

# 5.3. Binding functions of generalizing models tend to have low complexity functions

In §5.1 we operationalized binding complexity via generalization of an MLP approximator: even wide MLPs struggled to predict CLIP embeddings on held-out objects, revealing a high-complexity, non-generalizing binding map. Here we apply the same diagnostic to the controlled models from §5.2 and test the converse prediction: models that do generalize binding should admit a low-complexity (easy-toapproximate) concept-to-object mapping.

Setup. We measure binding complexity exactly as in §5.1: we train MLPs of varying width to predict scene embeddings from discrete concept indices and evaluate on scenes composed of held-out objects, reporting concept vs. object recognition on the predicted embeddings. If binding is low-complexity, small MLPs should generalize; if it is high-complexity (as in CLIP), generalization should require large capacity and still fail. We further test this in the visual domain by training scene encoders from scratch on pixel inputs, including noisy and occluded settings (see §E).

![](images/097ee66b9ac7bdb16ccb451759c3b94a89cd95d953ec2a4040c33a7fbc21b91a.jpg)  
Figure 7. Binding functions can be approximated by lowcapacity models. Even single-layer MLPs with small hidden dimensions achieve high accuracy, suggesting that the learned binding operation has low computational complexity. Each panel varies the number of concepts C and concept values V .

Results. Fig. 7 shows that for controlled models that generalize, the binding map is easy to approximate in the sense that even small nonlinear MLPs achieve strong accuracy. At the same time, as the object space grows (larger C and V ), the performance gap between the smallest nonlinear MLP and the largest-capacity MLP increases, showing that the binding map does become moderately more complex with scale. These findings are analogous when the scene encoder is trained from scratch directly on pixel inputs (§E). This contrasts with §5.1, where even high-capacity MLPs fail to generalize object recognition on CLIP embeddings.

Takeaway §5.3: In generalizing transformer-based models, binding function tends to be low-complexity: even small nonlinear MLPs generalize well.

# 5.4. Multiplicative structure best explains binding

We next ask what functional form the learned binding takes. We compare simple, structured probes that predict scene embeddings from concept values via additive and multiplicative composition. The Additive probe is the bag-of-concepts baseline. The Per-obj. products and Global product probes introduce the simplest possible departure from additivity: product interactions between concept embeddings, which can produce a unique representation for each object combination. In this subsection, we consider scenes of two ordered objects, $\pmb { s } = ( o _ { 1 } , o _ { 2 } )$ , where each object is described by two concept types (e.g., color and shape), so $o _ { i } = ( c _ { i 1 } , c _ { i 2 } )$ with $c _ { i k } \in \mathcal { C } _ { k }$ the value of concept type k in the i-th object. The parameters we train are concept-value embeddings: ${ \pmb u } _ { k , c }$ is indexed by concept type k and value $c \in { \mathcal { C } } _ { k } .$ shared across the two object positions. The multiplicative embedding $v _ { i , k , c }$ carries an additional index $i \in \{ 1 , 2 \}$ for the object’s position, so its products in Per-obj. products (within an object) and Global product (across both objects) yield a distinct vector for every combination of concept values — the binding signal that pure addition cannot express. Training and testing data are constructed following §C.4.

![](images/fbbd643ce4719dc62441f155aa297edff9fee83116fb40e029fbfb23b7a70230.jpg)  
Figure 8. Multiplicative structure best explains binding. Accuracy of the Additive (—), Per-obj. products (—), and Global product (—) probes for predicting scene embeddings from concept indices, evaluated on held-out objects. Dashed: concept recognition; solid: object recognition.

$$
\begin{array}{l} \underline {{\text { Additive: }}} \quad \sum_ {i = 1} ^ {2} \sum_ {k = 1} ^ {2} \boldsymbol {u} _ {k, c _ {i k}} \\ \text { Per - obj.   products: } \quad \sum_ {i = 1} ^ {2} \sum_ {k = 1} ^ {2} \boldsymbol {u} _ {k, c _ {i k}} + \sum_ {i = 1} ^ {2} \prod_ {k = 1} ^ {2} \boldsymbol {v} _ {i, k, c _ {i k}} \\ \text { Global   product: } \quad \sum_ {i = 1} ^ {2} \sum_ {k = 1} ^ {2} \boldsymbol {u} _ {k, c _ {i k}} + \prod_ {i = 1} ^ {2} \prod_ {k = 1} ^ {2} \boldsymbol {v} _ {i, k, c _ {i k}} \\ \end{array}
$$

Results. Fig. 8 shows that all probes do well at concept recognition, but the Additive baseline fails at object recognition. Adding multiplicative interactions recovers binding generalization, with the Global product form performing best, especially in larger object spaces.

Multiplicative structure underlies generalization. To test whether this relationship holds more broadly than on the four considered models, we examine the correlation between a model’s generalization ability and how well its embeddings are approximated by the Global product probe. For each of ∼500 models trained with varying hyperparameters across all object space sizes, we fit the Global product probe using 50% of the object space and measure both (1) the model’s object recognition accuracy on held-out objects, and (2) the probe’s accuracy at predicting scene embeddings. Fig. 9 shows a strong positive correlation: models that generalize well are precisely those whose embeddings are wellapproximated by the Global product form. This suggests that generalizing models are internally implementing something akin to multiplicative binding—a simple functional form that composes concept representations systematically. The correlation weakens slightly for the largest object space $( | O | = 1 2 5 \mathbf { k } )$ , but the pattern remains clear: the multiplicative form is sufficient to capture how concepts combine, which is why it generalizes to unseen combinations. Applying the same Global product probe to pretrained CLIP and DINOv2 embeddings instead recovers concept recognition but leaves object recognition near zero (§D.7), consistent with CLIP’s failure to bind concepts cross-modally.

![](images/62e515dd5ce9c49b764431e9c274bb47d8d2e60e92008b45541070492644ef1e.jpg)  
Figure 9. Generalization correlates with multiplicative structure. Each point is a trained model; x-axis shows object recognition accuracy on scenes composed of held-out objects, y-axis shows how well the Global product probe (trained on 50% of objects) approximates scene embeddings. Colors indicate object space size. Models that generalize also admit simpler (multiplicative) binding.

Takeaway §5.4: Generalizable binding corresponds to multiplicative structure in the embedding space: models whose embeddings admit simple multiplicative composition are precisely those that generalize to unseen object combinations.

# 6. Discussion

In this work we formalized concept binding by distinguishing concept recognition from object recognition, and then analyzed the geometry of multi-object embeddings through this lens. Scene embeddings show a clear additive objectlevel structure (supporting uni-modal decodability and direct embedding edits), but the concept-to-object mapping appears to be high-complexity and combination-specific, which blocks generalization and makes cross-modal alignment brittle. In controlled transformer dual-encoders trained from scratch, binding generalization emerges with enough data, and the generalizing mapping is best explained by low-complexity structure with multiplicative interactions between concepts.

Limitations. Our analysis uses synthetic datasets. To our knowledge, no real-world dataset currently provides the controlled, combinatorially complete structure required to study binding in our setting. In addition, our notion of complexity is necessarily relative to a chosen approximator family, since Kolmogorov complexity is not computable in practice.

# Acknowledgments

The authors thank Declan Campbell for helpful discussions and comments on initial drafts of this work, Alexander Rubinstein for insightful discussions, and the anonymous reviewers for their valuable feedback. This work was supported by the Tubingen AI Center. Arnas Uselis and Da- ¨ rina Koishigarina were supported by the International Max Planck Research School for Intelligent Systems (IMPRS-IS). Seong Joon Oh was supported by the Institute for Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2019-II190075, Artificial Intelligence Graduate School Program gram(KAIST)).

# Impact statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

# References

Abbasi, R., Rohban, M. H., and Baghshah, M. S. Deciphering the role of representation disentanglement: Investigating compositional generalization in clip models. arXiv preprint arXiv:2407.05897, 2024. 2   
Assouel, R., Campbell, D. I., Bengio, Y., and Webb, T. W. Visual symbolic mechanisms: Emergent symbol processing in vision language models. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=3RQ863cRbx. 2   
Berasi, D., Farina, M., Mancini, M., Ricci, E., and Strisciuglio, N. Not only text: Exploring compositionality of visual representations in vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 24917–24927, 2025. 2, 4   
Bergsma, S., Dey, N. S., Gosal, G., Gray, G., Soboleva, D., and Hestness, J. Straight to zero: Why linearly decaying the learning rate to zero works best for LLMs. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=hrOlBgHsMI. 16   
Blier, L. and Ollivier, Y. The description length of deep learning models, 2018. URL https://arxiv.org/ abs/1802.07044. 6, 25   
Borg, I. and Groenen, P. J. Modern multidimensional scaling: Theory and applications. Springer, 2005. 4, 24

Campbell, D., Rane, S., Giallanza, T., De Sabbata, N., Ghods, K., Joshi, A., Ku, A., Frankland, S. M., Griffiths, T. L., Cohen, J. D., et al. Understanding the limits of vision language models through the lens of the binding problem. arXiv preprint arXiv:2411.00238, 2024. 2, 23, 24

Dittadi, A., Trauble, F., Locatello, F., W ¨ uthrich, M., ¨ Agrawal, V., Winther, O., Bauer, S., and Scholkopf, B. ¨ On the transfer of disentangled representations in realistic settings, 2021. URL https://arxiv.org/abs/ 2010.14407. 2

Elmoznino, E., Jiralerspong, T., Bengio, Y., and Lajoie, G. Towards a formal theory of compositionality, 2025. URL https://openreview.net/forum? id=hKMPz3wkPV. 3, 6

Feng, J. and Steinhardt, J. How do language models bind entities in context? In The Twelfth International Conference on Learning Representations, 2024. URL https:// openreview.net/forum?id=zb3b6oKO77. 2

Feng, J., Russell, S., and Steinhardt, J. Monitoring latent world states in language models with propositional probes. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=0yvZm2AjUr. 2

Greff, K., Van Steenkiste, S., and Schmidhuber, J. On the binding problem in artificial neural networks. arXiv preprint arXiv:2012.05208, 2020. 2

Gurung, B., Hoffmann, D. T., and Brox, T. Common data properties limit object-attribute binding in clip. In DAGM German Conference on Pattern Recognition, pp. 353–368. Springer, 2025. 2

Hsieh, C.-Y., Zhang, J., Ma, Z., Kembhavi, A., and Krishna, R. Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality. Advances in neural information processing systems, 36, 2024. 2

Huang, L., Li, Y., Chang, Y., Salehi, S., and Kording, K. The binding problem in vision models: Geometric, functional, and behavioral approaches. In NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations, 2025. URL https://openreview.net/forum? id=rUK0P1Ejxl. 2

Jarvis, D., Klein, R., Rosman, B., and Saxe, A. M. On the specialization of neural modules, 2024. URL https: //arxiv.org/abs/2409.14981. 2

Jeong, Y., Uselis, A., Laina, I., Oh, S. J., and Rohrbach, A. When do diffusion models learn to generate multiple objects?, 2026. URL https://arxiv.org/abs/ 2605.00273. 2

Jiang, Y., Rajendran, G., Ravikumar, P. K., Aragam, B., and Veitch, V. On the origins of linear representations in large language models. In Forty-first International Conference on Machine Learning, 2024. URL https:// openreview.net/forum?id=otuTw4Mghk. 2   
Johnson, J., Hariharan, B., van der Maaten, L., Fei-Fei, L., Zitnick, C. L., and Girshick, R. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1988–1997, 2017. doi: 10.1109/CVPR.2017.215. 17   
Kamath, A., Hessel, J., and Chang, K.-W. Text encoders bottleneck compositionality in contrastive vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 4933– 4944, 2023. 2   
Kang, R., Song, Y., Gkioxari, G., and Perona, P. Is clip ideal? no. can we fix it? yes! ICCV, 2025. 2   
Kempf, E., Schrodi, S., Argus, M., and Brox, T. When and how does CLIP enable domain and compositional generalization? In Forty-second International Conference on Machine Learning, 2025. URL https://openreview. net/forum?id=Lktwi30g63. 2   
Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization, 2017. URL https://arxiv.org/abs/ 1412.6980. 20   
Koishigarina, D., Uselis, A., and Oh, S. J. CLIP behaves like a bag-of-words model cross-modally but not uni-modally. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview. net/forum?id=DldwXCCP25. 1, 2, 3, 4, 17   
Lewis, M., Nayak, N., Yu, P., Merullo, J., Yu, Q., Bach, S., and Pavlick, E. Does clip bind concepts? probing compositionality in large image models. In Findings of the Association for Computational Linguistics: EACL 2024, pp. 1487–1500, 2024. 1, 2, 8   
Li, S., Koh, P. W., and Du, S. S. Exploring how generative mllms perceive more than clip with the same vision encoder. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 10101–10119, 2025. 2   
Li, Y., Salehi, S., Ungar, L., and Kording, K. P. Does object binding naturally emerge in large pretrained vision transformers?, 2026. URL https://arxiv.org/abs/ 2510.24709. 2   
Liang, Q., Qian, D., Ziyin, L., and Fiete, I. R. Compositional generalization via forced rendering of disentangled

latents. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview. net/forum?id=rkHCHI5H5W. 2   
Lippl, S. and Stachenfeld, K. When does compositional structure yield compositional generalization? a kernel theory. In The Thirteenth International Conference on Learning Representations, 2025. URL https:// openreview.net/forum?id=FPBce2P1er. 2   
Loshchilov, I. and Hutter, F. Fixing weight decay regularization in adam. CoRR, abs/1711.05101, 2017. URL http://arxiv.org/abs/1711.05101. 7   
Ma, Z., Hong, J., Gul, M. O., Gandhi, M., Gao, I., and Krishna, R. Crepe: Can vision-language foundation models reason compositionally? In CVPR, pp. 10910–10921, 2023. 2   
Mahajan, D., Pezeshki, M., Arnal, C., Mitliagkas, I., Ahuja, K., and Vincent, P. Compositional risk minimization, 2025. URL https://arxiv.org/abs/2410. 06303. 2   
Montero, M. L., Ludwig, C. J., Costa, R. P., Malhotra, G., and Bowers, J. The role of disentanglement in generalisation. In International Conference on Learning Representations, 2021. URL https://openreview.net/ forum?id=qbH974jKUVy. 2   
Okawa, M., Lubana, E. S., Dick, R., and Tanaka, H. Compositional abilities emerge multiplicatively: Exploring diffusion models on a synthetic task. Advances in Neural Information Processing Systems, 36:50173–50195, 2023. 2   
Park, K., Choe, Y. J., and Veitch, V. The linear representation hypothesis and the geometry of large language models. arXiv preprint arXiv:2311.03658, 2023. 2   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML), pp. 8748–8763. PMLR, 2021. URL https://proceedings.mlr.press/ v139/radford21a.html. 1, 7, 14   
Ren, Y. and Sutherland, D. J. Understanding simplicity bias towards compositional mappings via learning dynamics. arXiv preprint, 2024. 25   
Schott, L., von Kugelgen, J., Tr ¨ auble, F., et al. Visual rep- ¨ resentation learning does not generalize strongly within the same domain, 2022. URL https://arxiv.org/ abs/2107.08221. 2

Tang, Y., Yamada, Y., Zhang, Y., and Yildirim, I. When are lemons purple? the concept association bias of visionlanguage models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 14333–14348, 2023. 1, 2   
Tong, S., Jones, E., and Steinhardt, J. Mass-producing failures of multimodal systems with language models. NIPS, 36, 2024a. 2   
Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., and Xie, S. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In CVPR, pp. 9568–9578, 2024b. 2   
Trager, M., Perera, P., Zancato, L., Achille, A., Bhatia, P., and Soatto, S. Linear spaces of meanings: compositional structures in vision-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15395–15404, 2023. 2, 4   
Treisman, A. Feature binding, attention and object perception. Philosophical Transactions of the Royal Society of London. Series B: Biological Sciences, 353(1373):1295– 1306, 1998. 1   
Treisman, A. M. and Gelade, G. A feature-integration theory of attention. Cognitive Psychology, 12(1):97–136, 1980. 1   
Uselis, A., Dittadi, A., and Oh, S. J. Does data scaling lead to visual compositional generalization? In Fortysecond International Conference on Machine Learning, 2025. URL https://openreview.net/forum? id=M2WMUuwoh5. 2   
Uselis, A., Dittadi, A., and Oh, S. J. Compositional generalization requires linear, orthogonal representations in vision embedding models, 2026. URL https://arxiv. org/abs/2602.24264. 2   
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need, 2023. URL https://arxiv.org/ abs/1706.03762. 7   
Wang, H., Si, H., Shao, H., and Zhao, H. Enhancing compositional generalization via compositional feature alignment. arXiv preprint arXiv:2402.02851, 2024. 2   
Wilson, A. G. Deep learning is not so mysterious or different. arXiv preprint, 2025. 6, 25   
Yuksekgonul, M., Bianchi, F., Kalluri, P., Jurafsky, D., and Zou, J. When and why vision-language models behave like bags-of-words, and what to do about it? In ICLR, 2023. 1, 2

# Appendix

A Notation summary 14   
B Details on the controlled models 14

B.1 Scene space and tokenization 14   
B.2 Encoders 15   
B.3 Queries, labels, and loss . 15   
B.4 Optimization and controlled scenarios 16

C Experimental details 17

C.1 Dataset details . 17   
C.2 CLIP geometry experiments 18   
C.3 Text encoder decomposition 19   
C.4 Details on approximating pre-trained models’ binding function 20

D Additional results 20

D.1 Interventions 20   
D.2 Editing scene embeddings with object embeddings 21   
D.3 Three-object scenes and occlusions . 23   
D.4 Extension to natural images . 24   
D.5 MDS visualization 24   
D.6 Robustness of the binding-complexity diagnostic to the approximator family 25   
D.7 Multiplicative probe applied to CLIP and DINOv2 26

E Vision encoders trained from scratch 26

# A. Notation summary

Table 6. Core notation used in the main text. 

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $C$ </td><td>Number of concepts per object;  $i \in \{1, \dots, C\}$  indexes concepts.</td></tr><tr><td> $\mathcal{C}_{i}$ </td><td>Set of values for concept  $i$ ;  $V$  denotes  $|\mathcal{C}_{i}|$  when uniform.</td></tr><tr><td> $\mathcal{C}$ </td><td>Concept space  $\mathcal{C}_{1} \times \cdots \times \mathcal{C}_{C}$ .</td></tr><tr><td> $\mathbf{o} = (c_{1}, \dots, c_{C})$ </td><td>Object (concept-value tuple);  $c_{i} \in \mathcal{C}_{i}$ .</td></tr><tr><td> $O_{\max}$ </td><td>Maximum number of objects in a scene.</td></tr><tr><td> $m$ </td><td>Number of objects in a scene.</td></tr><tr><td> $\mathcal{S}_{m}$ </td><td>Set of ordered  $m$ -object scenes;  $\mathcal{S} = \bigcup_{m=1}^{O_{\max}} \mathcal{S}_{m}$ .</td></tr><tr><td> $\mathbf{s}$ </td><td>Scene (tuple of objects).</td></tr><tr><td> $\boldsymbol{x}_{\mathbf{s}}$ </td><td>Datapoint (e.g., image) depicting scene  $\mathbf{s}$ .</td></tr><tr><td> $\mathcal{X}, \mathcal{Y}$ </td><td>Input space and query space.</td></tr><tr><td> $f, g$ </td><td>Input encoder  $f : \mathcal{X} \to \mathbb{R}^{d}$  and query encoder  $g : \mathcal{Y} \to \mathbb{R}^{d}$ .</td></tr><tr><td> $d$ </td><td>Embedding dimension.</td></tr><tr><td> $s(\boldsymbol{x}_{\mathbf{s}}, \boldsymbol{y})$ </td><td>Cosine similarity between scene and query embeddings.</td></tr><tr><td> $\boldsymbol{y}_{i,v}, \boldsymbol{y}_{\mathbf{o}}$ </td><td>Query for concept  $i$  value  $v$ ; query for object  $\mathbf{o}$ .</td></tr><tr><td> $V_{i}(\mathbf{s})$ </td><td>Set of concept- $i$  values present in scene  $\mathbf{s}$ .</td></tr><tr><td> $O(\mathbf{s})$ </td><td>Set of objects present in scene  $\mathbf{s}$ .</td></tr><tr><td> $\boldsymbol{u}_{\mathbf{o}}$ </td><td>Object embedding;  $\boldsymbol{u}_{c_{j,i}}$  is the embedding of the value of concept  $i$  for object  $\mathbf{o}_{j}$ .</td></tr><tr><td> $\tilde{\boldsymbol{z}}$ </td><td>Counterfactual (edited) scene embedding.</td></tr><tr><td> $R^{2}, \boldsymbol{f}$ </td><td>Variance explained;  $\boldsymbol{f}$  is the mean scene embedding.</td></tr><tr><td> $\pi_{\mathrm{c}}, \pi_{\mathrm{o}}$ </td><td>Permutations of concept/object IDs in control ablations.</td></tr><tr><td> $\boldsymbol{W}, l$ </td><td>Linear model weights and input dimension in additive regressions.</td></tr><tr><td> $g(\mathbf{o}_{1}, \mathbf{o}_{2})$ </td><td>Binding-function approximator (MLP) mapping objects to embeddings.</td></tr></table>

# B. Details on the controlled models

We train controlled models from scratch on synthetic multi-object scenes represented as token sequences. We use two transformer encoders: one for scenes and one for queries (either a single concept value or an entire object). We optimize a retrieval objective that makes matching scene-query pairs more similar than mismatched pairs (akin to CLIP’s contrastive loss (Radford et al., 2021)). Because both scenes and queries are generated procedurally, we can vary the object-space size $V ^ { C }$ , the maximum number of objects per scene $O _ { \mathrm { m a x } } .$ , and whether evaluation uses held-out concept combinations. Table 7 summarizes the key parameters used throughout.

Table 7. Notation and key configuration parameters for the controlled models. 

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $C$ </td><td>Number of concepts per object.</td></tr><tr><td> $V$ </td><td>Number of values per concept.</td></tr><tr><td> $O_{\text{max}}$ </td><td>Maximum number of objects per scene.</td></tr><tr><td> $L_{\text{max}}$ </td><td>Maximum sequence length,  $L_{\text{max}} = (C + 2)O_{\text{max}} + 1$ .</td></tr><tr><td> $d_{\text{model}}$ </td><td>Transformer width.</td></tr><tr><td> $H$ </td><td>Number of attention heads.</td></tr><tr><td> $L$ </td><td>Number of transformer layers.</td></tr><tr><td> $d_{\text{out}}$ </td><td>Output embedding dimension.</td></tr><tr><td> $K_{\text{concept}}$ </td><td>Max number of concept queries per step.</td></tr><tr><td> $K_{\text{obj}}$ </td><td>Max number of object queries per step.</td></tr><tr><td> $M_{\text{perturb}}$ </td><td>Perturbation negatives per positive object.</td></tr><tr><td> $M_{\text{swap}}$ </td><td>Swap negatives generated per scene.</td></tr><tr><td> $\rho_{\text{train}}$ </td><td>Fraction of objects available during training (object-space split).</td></tr><tr><td> $s_{\text{split}}$ </td><td>Seed for the object-space split.</td></tr><tr><td> $\text{sim}$ </td><td>Similarity function ( $\cos$  or  $\text{dot}$ ).</td></tr></table>

# B.1. Scene space and tokenization

We parameterize the synthetic environment by the number of concepts $C ,$ the number of values per concept V , and the maximum number of objects per scene $O _ { \mathrm { m a x } }$ (max num objects). Each object is represented as a C-tuple of discrete concept values $\pmb { v } = ( v _ { 1 } , \dots , v _ { C } )$ with $v _ { c } \in \{ 0 , \ldots , V - 1 \}$ , and a scene is an ordered list of $n \in \{ 1 , \ldots , O _ { \operatorname* { m a x } } \}$ such objects.

Scene sampling. We draw the scene length $n \sim \operatorname { U n i f } \{ 1 , \dots , O _ { \operatorname* { m a x } } \}$ and then sample n objects independently. By default, each component $v _ { c }$ is drawn uniformly from $\{ 0 , \ldots , V - 1 \}$ . To implement train/test splits, we instead sample objects from a restricted list $\mathtt { a l l }$ owed objects.

Tokenization. Table 8 summarizes the sequence format. We map each concept/value pair $( c , v )$ to a token id $t ( c , v ) =$ $( c - 1 ) V + v ,$ yielding $C V$ concept-value tokens, and add four extra tokens: SOO (start-of-object), EOO (end-of-object), EOS (end-of-scene), and PAD. An object $\pmb { v } = ( v _ { 1 } , \dots , v _ { C } )$ is encoded as

$$
[ \mathrm{SOO}, t (1, v _ {1}), \dots , t (C, v _ {C}), \mathrm{EOO} ],
$$

and a scene is the concatenation of its objects followed by EOS, padded to the fixed length

$$
L _ {\mathrm{max}} = (C + 2) O _ {\mathrm{max}} + 1.
$$

Scenes are generated on the fly by an “infinite” iterator, so training is naturally described in gradient steps rather than epochs.

Table 8. Token IDs and sequence formats used for scenes and probe queries. 

<table><tr><td>Component</td><td>Definition</td></tr><tr><td>Concept-value token</td><td> $t(c,v) = (c-1)V + v$  for concept  $c \in \{1,\dots,C\}$ , value  $v \in \{0,\dots,V-1\}$ .</td></tr><tr><td>Special tokens</td><td> $\text{SOO} = CV, \text{EOO} = CV + 1, \text{EOS} = CV + 2, \text{PAD} = CV + 3.$ </td></tr><tr><td>Object sequence</td><td> $[\text{SOO}, t(1,v_1),\dots,t(C,v_C), \text{EOO}]$ .</td></tr><tr><td>Scene sequence</td><td>Concatenate  $n$  objects, append EOS, pad with PAD to length  $L_{\text{max}}$ .</td></tr><tr><td>Concept query</td><td> $[t(c,v), \text{EOS}]$  padded to  $L_{\text{max}}$ .</td></tr><tr><td>Object query</td><td> $[\text{SOO}, t(1,v_1),\dots,t(C,v_C), \text{EOO}, \text{EOS}]$  padded to  $L_{\text{max}}$ .</td></tr></table>

Example. With $C = 2$ and $V = 1 0$ , the object $v = ( 3 , 7 ) ~ ( \mathrm { i . e . , } v _ { 1 } = 3 , v _ { 2 } = 7 )$ is encoded as [SOO, 3, 17, EOO] since $t ( 1 , 3 ) = 3$ and $t ( 2 , 7 ) = 1 0 + 7 = 1 7$ .

# B.2. Encoders

Scenes and queries are embedded by two transformer encoders with the same architecture (SceneEncoder) but independent parameters: a scene encoder $f _ { \mathrm { s c e n e } }$ and a probe encoder $f _ { \mathrm { p r o b e } }$ . Each maps a padded length- $L _ { \mathrm { m a x } }$ token sequence to a single embedding via token embeddings, positional encoding, and an L-layer Transformer encoder with H attention heads (masking PAD). The scene encoder outputs $\boldsymbol { s } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } }$ . The probe encoder outputs $\pmb q \in \mathbb { R } ^ { d _ { \mathrm { o u t } } }$ .

For a minibatch of B scenes and T queries, let $\pmb { S } \in \mathbb { R } ^ { B \times d _ { \mathrm { o u t } } }$ and $Q \in \mathbb { R } ^ { T \times d _ { \mathrm { o u t } } }$ t collect their embeddings row-wise. We form a logit matrix $\boldsymbol { Z } \in \mathbb { R } ^ { B \times T }$ . We then optimze the embeddings $\pmb { \hat { x } } = \pmb { x } / ( \lVert \pmb { x } \rVert _ { 2 } + 1 0 ^ { - 6 } )$ and define Sˆ, Qˆ by row-wise normalization. We use $\tau = \exp ( \tilde { \tau } )$ as a learned temperature parameter, akin to CLIP (clamped to $\tau \leq 1 0 0 )$ .

# B.3. Queries, labels, and loss

At each training step we sample a batch of scenes and construct a set of query sequences. We form a multi-hot label matrix $\pmb { Y } \in \{ 0 , 1 \} ^ { B \times \breve { T } }$ , where $Y _ { i j } = 1$ if query j matches scene i. We use two query types:

1. Concept retrieval. Concept queries specify a single concept value. We collect the unique concept-value tokens that appear in the batch (excluding SOO/EOO/EOS/PAD), subsample up to num concept values take max, and tokenize each as a concept query.   
2. Object retrieval. Object queries specify a full object. We parse object blocks from the tokenized scenes, take the set of unique objects present in the batch, subsample up to 1024, and tokenize each as an object query. We then augment the query set with hard negatives (random/perturbed objects and attribute swaps).

Concept queries probe whether the model can retrieve scenes based on single attributes, while object queries probe multi-attribute binding.

Given logits Z and labels Y , we treat multiple positives by normalizing each row to a target distribution $\begin{array} { r } { P _ { i \cdot } = Y _ { i \cdot } / ( \sum _ { j } Y _ { i j } + } \end{array}$ $1 0 ^ { - 1 0 } )$ and minimizing the cross-entropy

$$
\mathcal {L} = - \frac {1}{B} \sum_ {i = 1} ^ {B} \sum_ {j = 1} ^ {T} \boldsymbol {P} _ {i j} \log \operatorname{softmax} (\boldsymbol {Z} _ {i.}) _ {j}.
$$

Scenes with $\textstyle \sum _ { j } Y _ { i j } = 0$ (none of the sampled queries match) have $\mathbf { \nabla } P _ { i } . = 0$ and contribute zero loss. When both concept and object queries are enabled we sum their losses and backpropagate through both encoders and τ˜.

# B.4. Optimization and controlled scenarios

Training uses AdamW over the parameters of both encoders and τ˜, with a linear warmup (first 10% of steps) followed by linear decay (Bergsma et al., 2025). The code can optionally take multiple optimizer steps per sampled minibatch (num grad steps). Table 9 lists the default hyperparameters and numerical-stability choices.

Table 9. Training hyperparameters and numerical-stability choices (defaults unless stated otherwise). 

<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Training steps</td><td> $T = 50,000$  gradient steps</td></tr><tr><td>Batch size</td><td> $B = 1024$ scenes per step</td></tr><tr><td>Optimizer</td><td>AdamW over both encoders and  $\tilde{\tau}$ </td></tr><tr><td>Learning rate</td><td> $3 \times 10^{-4}$ </td></tr><tr><td>Weight decay</td><td>0.001</td></tr><tr><td>LR schedule</td><td>Linear warmup for  $0.1T$  steps, then linear decay to 0</td></tr><tr><td>Temperature init/clamp</td><td>Initialize  $\tilde{\tau} = 1.15$ , use  $\tau = \exp(\tilde{\tau})$  clamped to  $\tau \leq 100$ </td></tr></table>

# C. Experimental details

# C.1. Dataset details

We use three real-image datasets: PUG:SPARE (Koishigarina et al., 2026) (Fig. 10) and CLEVR (Johnson et al., 2017) (Fig. 11). We additionally introduce CLEVR-2D (Fig. 12), a 2D adaptation of CLEVR in which 3D shapes are replaced by their flat 2D counterparts (cube → square, sphere → circle, cylinder → triangle). Each scene contains two objects; objects vary along two concepts—color (8 values) and shape (3 values)—yielding 576 two-object scenes. Paired captions follow the same template as CLEVR (e.g., “a purple square and a gray circle”).

a Red Crow and a Blue Fox   
![](images/b6d44815a117637279978556ded19ca93a366789844d5282e1446db835d83f1e.jpg)

<details>
<summary>natural_image</summary>

Two stylized animals, one red and one blue, running across a desert landscape under a cloudy sky (no text or symbols)
</details>

a Red Crow and a Green Fox   
![](images/16dce2e83b9d8d5eaf4b00afc5b1ab84b620a6672b6883935fc1178101340a60.jpg)

<details>
<summary>natural_image</summary>

Two stylized animals, one red and one green, running across a barren, sandy landscape under a cloudy sky (no text or symbols)
</details>

a Red Crow and a Yellow Fox   
![](images/e755ddb4c7f6681a99b64f8ed11d946907a0b46d21b8b40de636fbfe0939ec7c.jpg)

<details>
<summary>natural_image</summary>

Two stylized animals, one red and one yellow, running across a barren, sandy landscape under a cloudy sky (no text or symbols)
</details>

a Red Crow and a Brown Fox   
![](images/1930433626da24ef4fc92cf25e5512a353ebc40295527de1dedb05df01479818.jpg)

<details>
<summary>natural_image</summary>

Two animals running in a desert landscape under a cloudy sky (no text or symbols visible)
</details>

a Red Crow and a Purple Fox   
![](images/2f92d91782b7828ac6e64cadf570cf233fceb0cfbf2a1d85e4271b38d41231f1.jpg)

<details>
<summary>natural_image</summary>

Two pink figures running in a desert landscape under a cloudy sky (no text or symbols)
</details>

a Red Crow and a Cyan Fox   
![](images/186684adb68e9c4c0ba01c581c48087274229e64a129af8ebaaa0f729dc59f5b.jpg)

<details>
<summary>natural_image</summary>

Two stylized animals, one red and one green, running across a barren, reddish-brown landscape under a cloudy sky (no text or symbols)
</details>

a Red Crow and a Gray Fox   
![](images/17a0e464548a8cfcae028de7c314306c0aace90418819b9b6e541062979a20df.jpg)

<details>
<summary>natural_image</summary>

Desert landscape with a red flag and a gray wolf running across the sand (no text or symbols)
</details>

a Blue Crow and a Red Fox   
![](images/39d5d25358c20e4cab16f2ee9fcb225a493e04c91d47a6cff3f3e09e51294560.jpg)

<details>
<summary>natural_image</summary>

Two stylized animals, one blue and one red, running across a barren, sandy landscape under a cloudy sky (no text or symbols)
</details>

Figure 10. PUG:SPARE dataset samples. Photorealistic scenes with two objects varying in color and species (12 colors × 8 animals, 7392 scenes total).

![](images/353e276d7cf34391ff317e402b7c83b747c296bbf841bbf174d178b20fb9d3f0.jpg)

Figure 11. CLEVR dataset samples. 3D-rendered scenes with two objects varying in color and shape (8 colors × 3 shapes, 576 scenes).   
![](images/1df85ab2730810a4045748dea2fede63969f959108455289867dc2c180408075.jpg)  
Figure 12. CLEVR-2D dataset samples. Our 2D adaptation of CLEVR, replacing 3D shapes with flat 2D equivalents. Same concept structure as CLEVR (8 colors × 3 shapes, 576 scenes).

# C.2. CLIP geometry experiments

This section specifies how we compute the retrieval and probing metrics reported for the Level-I/II decomposition experiments (Tab. 2) and for the intervention experiments (§4.1). To evaluate whether binding-relevant information is present in embeddings within a modality, we use trained linear probes on frozen embeddings.

Probing metric (concept/object recognition accuracy). Let $z _ { s } = f ( \pmb { x } _ { s } ) \in \mathbb { R } ^ { d }$ denote an embedding of a scene s (image or text, depending on the encoder). For each concept $i \in [ C ]$ , we train a linear probe $h _ { i } ( z ) = W _ { i } z + \bar { b _ { i } } \in \mathbb { R } ^ { | \mathcal { C } _ { i } | }$ that outputs logits over the values of concept i. We also train an object probe $h _ { \mathrm { o b j } } ( z ) = W _ { \mathrm { o b j } } z + b _ { \mathrm { o b j } } \in \mathbb { R } ^ { | \mathcal { C } | }$ that outputs logits over the object space (concept tuples). Because scenes contain multiple objects, the target for a probe is generally multi-hot (multiple values/objects are present). We train probes on frozen embeddings by minimizing cross-entropy to the normalized multi-positive target distribution (as in §B).

At evaluation time, we evaluate using the metrics in Def. 3.3 and Def. 3.4. For concept i, let $h _ { i } ( z _ { s } ) _ { \tau }$ be the logit for value $v \in \mathcal { C } _ { i }$ . We count the scene as correct for concept i if

$$
\min _ {v \in V _ {i} (\boldsymbol {s})} h _ {i} (\boldsymbol {z} _ {\boldsymbol {s}}) _ {v} > \max _ {v \in \mathcal {C} _ {i} \setminus V _ {i} (\boldsymbol {s})} h _ {i} (\boldsymbol {z} _ {\boldsymbol {s}}) _ {v}.
$$

Concept recognition accuracy is the mean of this indicator over scenes and then averaged across concepts. Object recognition accuracy is computed analogously from $h _ { \mathrm { o b j } } ( z _ { s } )$ , using the set of objects $O ( s )$ .

Retrieval metric (nearest-neighbor scene retrieval). For a fixed dataset of two-object scenes $\{ s _ { n } \} _ { n = 1 } ^ { N }$ , we precompute their embeddings $z _ { n } = f ( \pmb { x } _ { s _ { n } } )$ . Given a predicted embedding $\tilde { z } _ { n } \left( \mathrm { e . g } \right.$ ., a reconstruction from object means, or an edited embedding from §4.1), retrieval is treated as a nearest-neighbor problem in embedding space:

$$
\hat {n} = \arg \max _ {m \in [ N ]} \frac {\langle \tilde {\boldsymbol {z}} _ {n} , \boldsymbol {z} _ {m} \rangle}{\| \tilde {\boldsymbol {z}} _ {n} \| _ {2} \| \boldsymbol {z} _ {m} \| _ {2}}.
$$

Retrieval is correct if the nearest neighbor corresponds to the intended scene. When multiple scenes share the same underlying object tuple (e.g., because of positional variation), we treat any scene with the intended object tuple as correct. We report top-1 accuracy over all evaluated scenes.

Reconstruction metric. For the $R ^ { 2 }$ score, we measure how much variance is explained by the addition of the two object embeddings. In other words, the score measures how well $V ^ { 2 }$ vectors can reconstruct all of the $V ^ { 4 }$ embeddings. The $R ^ { 2 }$ score is defined as

$$
R ^ {2} := 1 - \frac {\sum_ {\boldsymbol {x} _ {s}} \| f (\boldsymbol {x} _ {s}) - \boldsymbol {u} _ {\boldsymbol {o} _ {0}} - \boldsymbol {u} _ {\boldsymbol {o} _ {1}} \| _ {2} ^ {2}}{\sum_ {\boldsymbol {x} _ {s}} \| f (\boldsymbol {x} _ {s}) - \boldsymbol {f} \| _ {2} ^ {2}}. \tag {7}
$$

# C.3. Text encoder decomposition

Setup. We generate a synthetic caption dataset for scenes by specifying a finite concept space

$$
\mathcal {C} := \mathcal {C} _ {1} \times \mathcal {C} _ {2},
$$

where $\mathcal { C } _ { 1 }$ contains adjective tokens (e.g., color/size/material) and $\mathcal { C } _ { 2 }$ contains noun tokens $( \mathrm { e . g . }$ , shape/category). In our experiments we generate 10,000 two-object scenes plus 100 single-object scenes; each scene is a list of objects, and each object is represented by a pair of integer indices $( i _ { 1 } , i _ { 2 } )$ selecting tokens from $\mathcal { C } _ { 1 }$ and $\mathcal { C } _ { 2 }$ .

Caption construction. For each scene, we linearize objects into a string of the form $\mathbf { \ddot { a } }$ ⟨attr⟩ ⟨obj⟩ and a $\langle \mathrm { a t t r } \rangle ~ \langle \mathrm { o b j } \rangle ^ { \flat }$ . The following snippet shows the exact procedure used.

```python
concept_space = [
    ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "gray", "black",
    "small", "large", "tiny", "round", "sharp", "rough", "bumpy", "smooth", "matte", "glossy"],
    ["circle", "square", "triangle", "star", "heart", "spiral", "sphere", "cylinder", "cone", "pyramid",
    "car", "cat", "dog", "man", "woman", "tree", "flower", "rock", "bird", "fish"],
]

captions_dataset = []

for scene in scene_dataset:
    scene_caption = "a" 
```

```python
for obj_i, obj in enumerate(scene):
    for concept_i, concept in enumerate(obj):
    scene_caption += f"{concept_space[concept_i][concept]}"
    if obj_i != len(scene) - 1:
    scene_caption += "_and_a"
    captions_dataset.append(scene_caption) 
```

# C.4. Details on approximating pre-trained models’ binding function

This section provides experimental details for the binding function approximation experiments on pre-trained models (§5.1). Our aim is to determine whether CLIP’s binding function can be approximated by a learnable function, rather than to find the most efficient approximator. To ensure that any failure to generalize reflects genuine complexity of the binding function rather than insufficient model capacity, we deliberately over-parameterize the approximator and sweep over a wide range of configurations.

MLP architectures. We use ReLU MLPs that take as input a concatenation of one-hot encodings of concept indices for both objects in the scene $( \mathbf { i . e . , 2 } \times C \times V$ binary features, where C is the number of concepts and V is the number of values per concept). We experiment with 1-layer MLPs with hidden widths ∈ {64, 128, 256, 512, 1024, 2048, 4096};

The output dimension matches the embedding dimension d of the encoder (e.g., d = 512 for CLIP ViT-B/32). Results reported in the main text correspond to the best performance achieved across all architectures for each training fraction, ensuring that we report an upper bound on what the approximator family can achieve.

Training. We use the Adam (Kingma & Ba, 2017) optimizer with learning rates in $\{ 1 0 ^ { - 2 } , 1 0 ^ { - 3 } , 1 0 ^ { - 4 } \}$ , batch size 4096, for 10,000 steps. We select the best configuration based on test object accuracy. We split the $V ^ { C }$ possible objects into train/test sets, varying training coverage over $\{ 1 0 \% , 2 0 \% , . . . , 9 0 \% \}$ . Test scenes contain only held-out objects, i.e., objects whose concept configurations never appeared during training.

Evaluation. We apply linear probes (trained on true embeddings) to the predicted embeddings and measure concept/object recognition accuracy. This tests whether predicted embeddings preserve functional binding properties, not just reconstruction fidelity. For text: we use synthetic captions (§C.3); for images: CLEVR, CLEVR-2D, PUG:SPARE. We report the best accuracy across all MLP architectures and learning rates for each configuration.

# D. Additional results

# D.1. Interventions

In §4.2, we test how concept and object components each contribute to concept and object recognition in scene embeddings.

Setup. We estimate object and concept embeddings as described in Tab. 6. We use the same retrieval and probing metrics defined in §C.2. For interventions, the target of an edited embedding is the intended counterfactual scene (or, when multiple scenes share the same object tuple due to positional variation, any scene with that object tuple is treated as correct).

Permuted-ID controls. To ensure that effects are not explained by subtracting an arbitrary but similarly-sized signal, we also run permuted-ID controls for both concept and object ablations. Concretely, let $N _ { \mathrm { c } } : = \dot { \sum } _ { i = 1 } ^ { C } | \mathcal { C } _ { i } |$ be the total number of concept values (often $N _ { \mathrm { c } } = C \cdot V )$ , and let $\begin{array} { r } { N _ { \mathrm { o } } : = | \mathcal { C } | = \prod _ { i = 1 } ^ { C } | \mathcal { C } _ { i } | } \end{array}$ be the number of objects (concept tuples). We sample uniform random permutations $\pi _ { \mathrm { c } } : [ N _ { \mathrm { c } } ]  [ N _ { \mathrm { c } } ]$ and $\pi _ { \mathrm { o } } : [ N _ { \mathrm { o } } ] \to [ N _ { \mathrm { o } } ]$ . When applying $\pi _ { \mathrm { c } } .$ , we treat each concept value (i, v) as a unique ID in $[ N _ { \mathrm { c } } ] ;$ ; when applying $\pi _ { \mathrm { o } }$ , we treat each object $\pmb { o } \in \mathcal { C }$ as a unique ID in $[ N _ { \mathrm { o } } ]$ . We then form the same residual embeddings as in the main text, but with permuted components, e.g.,

$$
f (\boldsymbol {x} _ {\boldsymbol {s}}) - \sum_ {i} \boldsymbol {u} _ {\pi_ {\mathrm{c}} (c _ {1, i})} - \sum_ {i} \boldsymbol {u} _ {\pi_ {\mathrm{c}} (c _ {2, i})} \quad \text { or } \quad f (\boldsymbol {x} _ {\boldsymbol {s}}) - \boldsymbol {u} _ {\pi_ {\mathrm{o}} (\boldsymbol {o} _ {1})} - \boldsymbol {u} _ {\pi_ {\mathrm{o}} (\boldsymbol {o} _ {2})}.
$$

We evaluate these permuted residuals with the same probes.

Table 10. Decodability results. We report probe accuracy after subtracting components from the embedding (remove concepts/objects) or their permuted-ID controls, for the image encoder and its random-init counterpart for nonpositional / positional variations. 

<table><tr><td>Intervention</td><td>Conc. (↑)</td><td>Obj. (↑)</td></tr><tr><td colspan="3">CLIP ViT-B/32 (CLEVR-2D)</td></tr><tr><td>None</td><td>1.00 / 1.00</td><td>0.99 / 0.99</td></tr><tr><td>Subtract concepts</td><td>0.32 / 0.30</td><td>0.78 / 0.79</td></tr><tr><td>Subtract objects</td><td>0.28 / 0.29</td><td>0.08 / 0.10</td></tr><tr><td>Permute concepts</td><td>0.83 / 0.83</td><td>0.95 / 0.95</td></tr><tr><td>Permute objects</td><td>0.92 / 0.91</td><td>0.86 / 0.86</td></tr><tr><td colspan="3">CLIP ViT-B/32 (CLEVR)</td></tr><tr><td>None</td><td>1.00 / 1.00</td><td>0.93 / 0.93</td></tr><tr><td>Subtract concepts</td><td>0.28 / 0.27</td><td>0.77 / 0.79</td></tr><tr><td>Subtract objects</td><td>0.26 / 0.23</td><td>0.01 / 0.03</td></tr><tr><td>Permute concepts</td><td>0.83 / 0.81</td><td>0.91 / 0.93</td></tr><tr><td>Permute objects</td><td>0.93 / 0.91</td><td>0.84 / 0.82</td></tr><tr><td colspan="3">DINOv2 ViT-B/14 (CLEVR-2D)</td></tr><tr><td>None</td><td>1.00 / 1.00</td><td>0.97 / 0.96</td></tr><tr><td>Subtract concepts</td><td>0.26 / 0.27</td><td>0.79 / 0.79</td></tr><tr><td>Subtract objects</td><td>0.25 / 0.23</td><td>0.03 / 0.06</td></tr><tr><td>Permute concepts</td><td>0.79 / 0.78</td><td>0.94 / 0.93</td></tr><tr><td>Permute objects</td><td>0.91 / 0.90</td><td>0.85 / 0.86</td></tr><tr><td colspan="3">DINOv2 ViT-B/14 (CLEVR)</td></tr><tr><td>None</td><td>0.99 / 0.99</td><td>0.92 / 0.92</td></tr><tr><td>Subtract concepts</td><td>0.20 / 0.20</td><td>0.61 / 0.65</td></tr><tr><td>Subtract objects</td><td>0.18 / 0.20</td><td>0.01 / 0.00</td></tr><tr><td>Permute concepts</td><td>0.77 / 0.76</td><td>0.92 / 0.92</td></tr><tr><td>Permute objects</td><td>0.87 / 0.88</td><td>0.83 / 0.81</td></tr></table>

# D.2. Editing scene embeddings with object embeddings

![](images/737f6177bb50f75ca9b97a2a6159053d28db776a7fcd7e13fe7564ecbfe475ae.jpg)  
Figure 13. Effect of intervention strength on object replacement. We vary intervention strength k with position-independent (AVG) and position-dependent (AVG+POS) object embeddings for CLEVR and PUG:SPARE. We distinguish between samples in which two objects have different concepts (solid line) and those with a shared concept (dashed line).

Table 11. Decodability results. We report probe accuracy after subtracting components from the embedding (remove concepts/objects) or their permuted-ID controls, for both text and image encoders and their random-init counterparts for nonpositional / positional variations on the PUG:SPARE dataset. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Intervention</td><td colspan="2">Text</td><td colspan="2">Image</td></tr><tr><td>Conc. (↑)</td><td>Obj. (↑)</td><td>Conc. (↑)</td><td>Obj. (↑)</td></tr><tr><td rowspan="5">Random</td><td>None</td><td>0.95 / 0.95</td><td>0.59 / 0.60</td><td>0.78 / 0.78</td><td>0.71 / 0.71</td></tr><tr><td>Subtract concepts</td><td>0.02 / 0.02</td><td>0.27 / 0.43</td><td>0.06 / 0.08</td><td>0.66 / 0.68</td></tr><tr><td>Subtract objects</td><td>0.01 / 0.01</td><td>0.00 / 0.00</td><td>0.04 / 0.05</td><td>0.00 / 0.01</td></tr><tr><td>Permute concepts</td><td>0.88 / 0.88</td><td>0.69 / 0.75</td><td>0.80 / 0.81</td><td>0.78 / 0.76</td></tr><tr><td>Permute objects</td><td>0.83 / 0.84</td><td>0.93 / 0.98</td><td>0.78 / 0.80</td><td>0.88 / 0.88</td></tr><tr><td rowspan="5">CLIP-B/32</td><td>None</td><td>1.00 / 1.00</td><td>1.00 / 1.00</td><td>0.95 / 0.95</td><td>0.88 / 0.88</td></tr><tr><td>Subtract concepts</td><td>0.07 / 0.07</td><td>0.99 / 0.99</td><td>0.06 / 0.08</td><td>0.73 / 0.73</td></tr><tr><td>Subtract objects</td><td>0.05 / 0.05</td><td>0.02 / 0.03</td><td>0.04 / 0.04</td><td>0.00 / 0.00</td></tr><tr><td>Permute concepts</td><td>0.92 / 0.93</td><td>1.00 / 1.00</td><td>0.84 / 0.84</td><td>0.91 / 0.91</td></tr><tr><td>Permute objects</td><td>0.96 / 0.96</td><td>1.00 / 1.00</td><td>0.82 / 0.83</td><td>0.92 / 0.93</td></tr><tr><td rowspan="5">DINOv2-B/14</td><td>None</td><td>-</td><td>-</td><td>0.95 / 0.94</td><td>0.79 / 0.79</td></tr><tr><td>Subtract concepts</td><td>-</td><td>-</td><td>0.04 / 0.04</td><td>0.62 / 0.70</td></tr><tr><td>Subtract objects</td><td>-</td><td>-</td><td>0.03 / 0.04</td><td>0.00 / 0.00</td></tr><tr><td>Permute concepts</td><td>-</td><td>-</td><td>0.77 / 0.77</td><td>0.80 / 0.79</td></tr><tr><td>Permute objects</td><td>-</td><td>-</td><td>0.83 / 0.83</td><td>0.87 / 0.87</td></tr></table>

![](images/2683302886279afb7caf90e56343d55882637fd16fa90df70db0531dfa2fcc1b.jpg)

<details>
<summary>line</summary>

| Model          | k    | Probing, different | Probing, shared | Retrieval, different | Retrieval, shared |
| -------------- | ---- | ------------------ | ---------------- | --------------------- | ------------------ |
| CLIP ViT-B/32  | 0.2  | 0.0                | 0.0              | 0.0                   | 0.0                |
| CLIP ViT-B/32  | 0.4  | 0.6                | 0.8              | 0.4                   | 0.5                |
| CLIP ViT-B/32  | 0.6  | 0.9                | 0.9              | 0.6                   | 0.7                |
| CLIP ViT-B/32  | 0.8  | 0.9                | 0.9              | 0.8                   | 0.9                |
| CLIP ViT-B/32  | 1.0  | 0.9                | 0.9              | 0.9                   | 0.9                |
| CLIP ViT-L/14  | 0.2  | 0.0                | 0.0              | 0.0                   | 0.0                |
| CLIP ViT-L/14  | 0.4  | 0.2                | 0.4              | 0.2                   | 0.3                |
| CLIP ViT-L/14  | 0.6  | 0.6                | 0.8              | 0.6                   | 0.7                |
| CLIP ViT-L/14  | 0.8  | 0.9                | 0.9              | 0.8                   | 0.9                |
| CLIP ViT-L/14  | 1.0  | 0.9                | 0.9              | 0.9                   | 0.9                |
| DINO ViT-B/14   | 0.2  | 0.0                | 0.0              | 0.0                   | 0.0                |
| DINO ViT-B/14   | 0.4  | 0.2                | 0.4              | 0.2                   | 0.3                |
| DINO ViT-B/14   | 0.6  | 0.6                | 0.8              | 0.6                   | 0.7                |
| DINO ViT-B/14   | 0.8  | 0.9                | 0.9              | 0.8                   | 0.9                |
| DINO ViT-B/14   | 1.0  | 0.9                | 0.9              | 0.9                   | 0.9                |
</details>

Figure 14. Object editing with SINGLE-OBJ embeddings. Intervention strength k is varied for CLEVR image embeddings using object embeddings estimated from single-object scenes.

![](images/28b848c591a36b83e5cf221681ec4209057d7c37c91fe515692d5474dc3f0475.jpg)

<details>
<summary>line</summary>

| k    | Probing, different | Probing, shared | Retrieval, different | Retrieval, shared |
|------|--------------------|-----------------|----------------------|-------------------|
| 0.2  | 0.0                | 0.0             | 0.0                  | 0.0               |
| 0.4  | 0.2                | 0.1             | 0.1                  | 0.1               |
| 0.6  | 0.8                | 0.7             | 0.7                  | 0.6               |
| 0.8  | 1.0                | 0.9             | 0.9                  | 0.8               |
| 1.0  | 1.0                | 1.0             | 1.0                  | 1.0               |
</details>

Figure 15. Object editing for text embeddings. We vary the intervention strength k of object replacement for text encoder embeddings.

We provide additional results for the object-replacement interventions introduced in §4.1. These experiments test whether object embeddings can be reused across scenes to produce counterfactual scene embeddings via simple arithmetic.

Setup. Given a two-object scene $\pmb { s } = ( o _ { 1 } , o _ { 2 } )$ with embedding $f ( x _ { s } )$ , we construct an edited embedding by replacing $\scriptstyle { o _ { 1 } }$ with a counterfactual object ${ \pmb o } _ { 1 } ^ { \prime }$ :

$$
\tilde {z} = f \left(x _ {\boldsymbol {s}}\right) - k \boldsymbol {u} _ {\boldsymbol {o} _ {1}} + k \boldsymbol {u} _ {\boldsymbol {o} _ {1} ^ {\prime}}, \tag {8}
$$

where k controls the strength of the intervention. Setting k = 0 leaves the embedding unchanged, while larger k increases the contribution of the replacement. In practice, performance varies with $k ,$ and we report full curves over k in Figure 13. Unless stated otherwise, results in Tab. 12 use k = 1.0.

We distinguish between two settings. In the different-concepts setting, the two objects in the scene do not share any concept values (e.g., a red cube and a blue sphere). In the shared-concepts setting, the objects share at least one concept value (e.g., a red cube and a red sphere). The shared-concepts setting is more challenging because the two objects occupy overlapping concept directions in the representation.

Results. Table 12 reports probing and retrieval performance for object replacement interventions at full strength (k = 1.0) across datasets and models. In most settings, replacing an object embedding yields scene embeddings that support both object probing and retrieval consistent with the intended counterfactual scene. Object embeddings obtained via scene averaging and position-specific averaging perform similarly across datasets, indicating that positional effects are limited. When available, embeddings derived from single-object scenes also enable effective edits, though with reduced performance in some settings.

Table 12. Intervention results summary at $k = 1 . 0 .$ . Object embeddings are constructed by object averaging (AVG), position-wise averaging (AVG+POS), or from single-object scenes (SINGLE-OBJ). 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td rowspan="2">Concepts</td><td colspan="3">Probing (↑)</td><td colspan="3">Retrieval (↑)</td></tr><tr><td>AVG</td><td>AVG+POS</td><td>SINGLE-OBJ</td><td>AVG</td><td>AVG+POS</td><td>SINGLE-OBJ</td></tr><tr><td>Text</td><td>CLIP ViT-B/32</td><td>Different</td><td>0.99</td><td>0.99</td><td>0.69</td><td>0.99</td><td>0.99</td><td>0.96</td></tr><tr><td>Text</td><td>CLIP ViT-B/32</td><td>Shared</td><td>0.86</td><td>0.86</td><td>0.23</td><td>0.73</td><td>0.73</td><td>0.10</td></tr><tr><td>CLEVR</td><td>CLIP ViT-B/32</td><td>Different</td><td>0.98</td><td>0.98</td><td>0.86</td><td>1.00</td><td>1.00</td><td>0.97</td></tr><tr><td>CLEVR</td><td>CLIP ViT-B/32</td><td>Shared</td><td>0.77</td><td>0.78</td><td>0.25</td><td>0.38</td><td>0.39</td><td>0.08</td></tr><tr><td>CLEVR</td><td>CLIP ViT-L/14</td><td>Different</td><td>0.98</td><td>0.98</td><td>0.90</td><td>0.98</td><td>0.99</td><td>0.98</td></tr><tr><td>CLEVR</td><td>CLIP ViT-L/14</td><td>Shared</td><td>0.80</td><td>0.80</td><td>0.28</td><td>0.10</td><td>0.11</td><td>0.05</td></tr><tr><td>CLEVR</td><td>DINO ViT-B/14</td><td>Different</td><td>0.97</td><td>0.98</td><td>0.77</td><td>0.89</td><td>0.95</td><td>0.86</td></tr><tr><td>CLEVR</td><td>DINO ViT-B/14</td><td>Shared</td><td>0.95</td><td>0.94</td><td>0.29</td><td>0.30</td><td>0.42</td><td>0.13</td></tr><tr><td>CLEVR-2D</td><td>CLIP ViT-B/32</td><td>Different</td><td>0.98</td><td>0.98</td><td>0.92</td><td>0.99</td><td>0.99</td><td>0.97</td></tr><tr><td>CLEVR-2D</td><td>CLIP ViT-B/32</td><td>Shared</td><td>0.68</td><td>0.67</td><td>0.26</td><td>0.14</td><td>0.14</td><td>0.05</td></tr><tr><td>CLEVR-2D</td><td>CLIP ViT-L/14</td><td>Different</td><td>0.98</td><td>0.98</td><td>0.90</td><td>0.95</td><td>0.95</td><td>0.90</td></tr><tr><td>CLEVR-2D</td><td>CLIP ViT-L/14</td><td>Shared</td><td>0.72</td><td>0.72</td><td>0.36</td><td>0.13</td><td>0.14</td><td>0.07</td></tr><tr><td>CLEVR-2D</td><td>DINO ViT-B/14</td><td>Different</td><td>0.97</td><td>0.98</td><td>0.72</td><td>0.97</td><td>0.97</td><td>0.88</td></tr><tr><td>CLEVR-2D</td><td>DINO ViT-B/14</td><td>Shared</td><td>0.87</td><td>0.88</td><td>0.29</td><td>0.14</td><td>0.16</td><td>0.03</td></tr><tr><td>PUG:SPARE</td><td>CLIP ViT-B/32</td><td>Different</td><td>0.94</td><td>0.95</td><td>-</td><td>0.86</td><td>0.94</td><td>-</td></tr><tr><td>PUG:SPARE</td><td>CLIP ViT-L/14</td><td>Different</td><td>0.96</td><td>0.96</td><td>-</td><td>0.58</td><td>0.74</td><td>-</td></tr><tr><td>PUG:SPARE</td><td>DINO ViT-B/14</td><td>Different</td><td>0.97</td><td>0.97</td><td>-</td><td>0.48</td><td>0.76</td><td>-</td></tr></table>

Performance drops substantially when the two objects in the scene share concepts, particularly for retrieval. This effect is consistent across datasets and encoders. A plausible explanation is feature interference: when objects share concept values, their contributions overlap in representation space, making it harder to selectively remove and insert a single object without affecting the other (Campbell et al., 2024).

We further analyze the role of intervention strength in Figs. 13 to 15. Varying the intervention coefficient k reveals that performance typically improves with stronger interventions before saturating. However, in shared-concept settings, optimal performance is often achieved at smaller values of $k ,$ suggesting that partial edits mitigate interference effects.

# D.3. Three-object scenes and occlusions

To check that the additive decomposition of §4.1 is not limited to two-object scenes, we generate CLEVR scenes with three objects and matching text descriptions (Fig. 16). We evaluate $R ^ { 2 }$ reconstruction, probing accuracy, and retrieval as in Tab. 2 and Tab. 3.

![](images/1b4032fe0b687fcdf95570ef488f2350ec037fa73079db442ca7b9a84401d769.jpg)

<details>
<summary>text_image</summary>

CLEVR-2D
CLEVR
CLEVR (with occlusions)
</details>

Figure 16. Example CLEVR scenes with 3 objects. Used for the 3-object decomposition evaluation in Tab. 13.

Table 13. Object decomposition extends to 3-object scenes (Fig. 16). $R ^ { 2 }$ reconstruction from concepts and objects, probing accuracy, and retrieval accuracy for scenes with 3 objects (as in Tab. 2 and Tab. 3). Reconstruction from objects remains stronger than from concepts. Occlusions degrade retrieval but probing stays high. 

<table><tr><td>Dataset</td><td>Concepts  $R^{2}$ </td><td>Objects  $R^{2}$ </td><td>Probing</td><td>Retrieval</td></tr><tr><td>Text</td><td>0.83</td><td>0.86</td><td>0.86</td><td>0.93</td></tr><tr><td>CLEVR</td><td>0.67</td><td>0.72</td><td>0.93</td><td>0.76</td></tr><tr><td>CLEVR (occlusions)</td><td>0.65</td><td>0.70</td><td>0.91</td><td>0.65</td></tr><tr><td>CLEVR-2D</td><td>0.67</td><td>0.73</td><td>0.82</td><td>0.76</td></tr></table>

The additive decomposition extends to 3 objects: probing accuracy remains 0.91–0.93 on CLEVR even with occlusions, and reconstruction from objects consistently exceeds reconstruction from concepts, matching the pattern in Tab. 2. Retrieval drops under occlusion (0.76 → 0.65), as expected, consistent with prior findings that VLM binding degrades as the number of objects grows (Campbell et al., 2024).

# D.4. Extension to natural images

Setup. To assess whether the additive decomposition extends beyond synthetic datasets, we generate images using Gemini Nano Banana 2 (gemini-3.1-flash-image-preview), spanning 5 object types and 5 patterns (625 samples). Unlike the synthetic datasets, these images exhibit variation in size, shape, color, and pattern realization, making the task substantially more challenging. We evaluate object decomposition and editing following the same protocol as in Tab. 2 and Tab. 3, using retrieval-based metrics: full retrieval among all 625 scenes, 4-way retrieval (correct scene, permuted concept combination, and two random distractors), and binary retrieval between the original and edited scene embeddings.

![](images/ffaa96736f467f26eb613fd5f4d039795b6d8c35f3a03ea5da475031a2347070.jpg)  
Figure 17. Example images spanning 5 objects and 5 patterns. Images were generated with Gemini Nano Banana 2 (gemini-3.1-flash-image-preview) at 512 × 512 resolution using the prompt template: “A square studio product photo of exactly two objects: a {obj1} with a {pattern1} print on its surface, next to a {obj2} with a {pattern2} print on its surface. Exactly two objects only. The pattern must appear only as the surface design of each object, not as separate objects, contents, decorations, or background elements.” Objects: mug, backpack, umbrella, vase, chair. Patterns: striped, polka dot, checkerboard, floral, zigzag. This yields $5 ^ { 2 } \times 5 ^ { 2 } = 6 2 5$ unique two-object scenes. The images more closely resemble natural images; objects vary in size, shape, and color, and patterns are realized differently across samples. See Tab. 14 for results.

Results. Results are well above chance despite the increased difficulty, suggesting that the additive structure extends, to some degree, to more naturalistic scenes. We note that with a larger sample size, object embedding estimates would likely improve further.

Table 14. Object decomposition and editing on the natural images. We evaluate additive decomposition (as in Tab. 2) and object editing (as in Tab. 3) via retrieval on the 625 generated scenes. Results are above chance despite the increased visual complexity. 

<table><tr><td>Task</td><td>Setting</td><td>Retrieval</td><td>Random chance</td></tr><tr><td>Decomposition</td><td>All (625)</td><td>54%</td><td>1/625 ≈ 0.16%</td></tr><tr><td>Decomposition</td><td>4-way</td><td>81%</td><td>25%</td></tr><tr><td>Object editing</td><td>All (625)</td><td>68%</td><td>1/625 ≈ 0.16%</td></tr><tr><td>Object editing</td><td>Binary</td><td>95%</td><td>50%</td></tr></table>

# D.5. MDS visualization

To build intuition about the embedding geometry, we visualize MDS projections for a small set of concepts (Borg & Groenen, 2005). We consider two shapes (C, S) and two colors (R, B). This yields objects such as RC (‘red cube’) and BS (‘blue sphere’), and two-object scenes such as RCBS (‘red cube and blue sphere’). Object and concept embeddings are estimated as averages of embeddings of scenes that contain the corresponding component. Purple points represent sums of object embeddings (e.g. RC+BS). We connect the embeddings of the corresponding two-object scenes (e.g. RCBS) to these summed vectors.

Results. Across datasets and models, the projections exhibit a partial grid-like structure reflecting color and shape composition. This structure is most pronounced for text embeddings, while for CLIP image embeddings, it is more distorted. In DINOv2, objects with different shapes are far apart in embedding space, whereas objects with similar colors remain close.

![](images/80aa4669d7d7fde2547e316fe845c2c8e35ae675b27c5d3e35a397e525532c9f.jpg)  
Figure 18. MDS shows an approximate additive structure using averaged object embeddings. MDS projections of CLIP and DINOv2 embeddings for scenes from CLEVR, CLEVR-2D, and PUG (distances approximate embedding distances). Object embeddings (e.g. RC for ‘red cube’) and concept embeddings (e.g. R for ‘red’) are estimated as averages of embeddings of scenes containing the corresponding component. Two-object scenes are denoted by concatenation (e.g. RCBS for ‘red cube and blue sphere’). Purple points correspond to sums of object embeddings (e.g. RC+BS). Line segments connect these summed vectors to the corresponding two-object scene embeddings.

# D.6. Robustness of the binding-complexity diagnostic to the approximator family

The binding-complexity analyses in §5.1 and §5.3 uses a 1-hidden-layer MLP as the approximator. Our choice is motivated by the view that MLPs trained with SGD tend to favor simple, compressible solutions (Blier & Ollivier, 2018; Wilson, 2025), and that compositional mappings are precisely the kind of solutions this bias should favor (Ren & Sutherland, 2024). If a simple compositional binding function exists, this approximator family should find it.

To verify that our findings are not specific to the MLP family, we additionally fit XGBoost and Random Forest approximators with the same setup as §5.1 (predict CLIP/DINOv2 scene embeddings from discrete concept indices, evaluate on held-out objects). For XGBoost we sweep n estimators∈ {100, 500, 1000}, max depth∈ {6, 8, 16, 32}, lr∈ {0.1, 0.01}. For Random Forest we sweep n estimators∈ {100, 500, 1000}, max depth∈ {None, 8, 16, 32}. We report the best accuracy across configurations.

Table 15. Approximator family does not change the conclusion. Best object / concept recognition accuracy on held-out objects when approximating scene embeddings from concept indices, for different approximator families. Across all three families, concept recognition is high while object recognition stays near zero, indicating that the bottleneck is the binding structure rather than the choice of approximator. 

<table><tr><td>Approximator</td><td>CLIP B/32 (img)Obj. / Conc.</td><td>DINOv2 B/16 (img)Obj. / Conc.</td><td>CLIP B/32 (txt)Obj. / Conc.</td></tr><tr><td>MLP</td><td>0.17 / 0.97</td><td>0.10 / 1.00</td><td>0.21 / 1.00</td></tr><tr><td>XGBoost</td><td>0.00 / 0.94</td><td>0.03 / 1.00</td><td>0.18 / 1.00</td></tr><tr><td>RF</td><td>0.00 / 0.75</td><td>0.00 / 1.00</td><td>0.04 / 0.90</td></tr></table>

In all three cases the approximator succeeds at concept-level prediction but fails at object-level binding. MLPs perform slightly better than XGBoost and Random Forest, though this is not an exhaustive search. The qualitative conclusion of §5.1 is unchanged: CLIP’s binding function does not generalize to unseen objects across this set of approximator families.

# D.7. Multiplicative probe applied to CLIP and DINOv2

We complement §5.4 by fitting the same multiplicative Global product probe to CLIP and DINOv2 embeddings (Fig. 19). Concept recognition recovers with more training data, but object recognition stays near zero across encoders. Unlike the from-scratch models (§5.4, §E), neither CLIP nor DINOv2 admits a multiplicative binding function, consistent with their known cross-modal binding failures.

![](images/e66cc20158adcaea7863c5fb24062668094c714a1951e2d8467397a0d820d37f.jpg)

<details>
<summary>line</summary>

| Fraction of train data | Accuracy |
| ---------------------- | -------- |
| 0.1                    | 0.00     |
| 0.3                    | 0.00     |
| 0.5                    | 0.00     |
| 0.7                    | 0.75     |
</details>

![](images/624557084edaff7db91a1d6b2e4cf6f7e10cb51417e82b138879d560ac5729e1.jpg)

<details>
<summary>line</summary>

| Fraction of train data | Value |
| ---------------------- | ----- |
| 0.1                    | 0.0   |
| 0.3                    | 0.0   |
| 0.5                    | 0.0   |
| 0.7                    | 0.0   |
</details>

![](images/6a3631e063cb3bb8a729dc482467109719f9a3e7df4d11b0d448eaefb6e02865.jpg)

<details>
<summary>line</summary>

| Fraction of train data | CLIP B/32 (text) |
| ---------------------- | ---------------- |
| 0.1                    | 0.0              |
| 0.3                    | 0.5              |
| 0.5                    | 1.0              |
| 0.7                    | 1.0              |
</details>

§c + ¦c (objects) concepts

Figure 19. Multiplicative structure does not hold for CLIP and DINOv2. We fit the Global product probe on CLIP and DINOv2 embeddings. Concept recognition (dashed) recovers with more training data, but object recognition (solid) stays near zero for all encoders. Unlike the from-scratch models trained on pixels (Fig. 21, bottom), CLIP’s and DINOv2’s binding functions are too complex to be captured by the multiplicative probe, consistent with their failure to bind concepts cross-modally.

# E. Vision encoders trained from scratch

The controlled experiments in §5.2–§5.4 use multi-hot token sequences as input to the scene encoder. To verify that the conclusions are not an artifact of the discrete input format, we replicate the entire pipeline (§5.2–§5.4) with pixel inputs and a from-scratch vision encoder.

Setup. We replace the multi-hot embedding layer of the scene encoder with a convolutional front-end and increase the number of transformer layers from 6 to 8 (we found this configuration to generalize better). The training pipeline, evaluation protocol, and binding analyses are otherwise identical to the main paper. Each object is defined by two concepts (square color and border color) with V = 50 values, giving up to $6 . 5 \times 1 0 ^ { 6 }$ object combinations. To stress the visual difficulty, we consider three levels of pixel-space complexity (Fig. 20): (i) noise-free, non-overlapping objects; (ii) speckled noise with non-overlapping objects; and (iii) noisy and overlapping objects.

![](images/32b8c4fc8c435a4d57b8d0aad1814403a841f6a1a1d09490941307ce85bdb936.jpg)

<details>
<summary>heatmap</summary>

| Group | o1   | o2   |
|-------|------|------|
| Top   | (12, 11) | (9, 16) |
| Top   | (11, 10) | (7, 5) |
| Top   | (0, 14) | (2, 12) |
| Top   | (4, 1) | (4, 11) |
| Top   | (2, 0) | (6, 0) |
| Top   | (15, 4) | (16, 2) |
| Bottom | (12, 11) | (9, 16) |
| Bottom | (11, 10) | (7, 5) |
| Bottom | (0, 14) | (2, 12) |
| Bottom | (4, 1) | (4, 11) |
| Bottom | (2, 0) | (6, 0) |
| Bottom | (15, 4) | (16, 2) |
| Bottom | (12, 11) | (9, 16) |
| Bottom | (11, 10) | (7, 5) |
| Bottom | (0, 14) | (2, 12) |
| Bottom | (4, 1) | (4, 11) |
| Bottom | (2,0) | (6, 0) |
| Bottom | (15, 4) | (16, 2) |
| Bottom | (9, 16) | (9, 16) |
| Bottom | (7, 5) | (7, 5) |
| Bottom | (4, 1) | (4, 11) |
| Bottom | (6, 0) | (6, 0) |
| Bottom | (15, 4) | (16, 2) |
The image displays a grid of colored squares with labels for 'o1' and 'o2', likely representing coordinate or position coordinates.
</details>

Figure 20. Synthetic pixel samples used to train vision encoders from scratch. Three levels of pixel-space complexity: (top) noise-free, non-overlapping objects; (middle) speckled noise, non-overlapping objects; (bottom) noisy and overlapping objects. Each object is defined by two concepts (square color and border color). Samples shown correspond to the $C = 2 , \bar { V } = 5 0$ setting, giving up to $6 . 5 \times 1 0 ^ { 6 }$ object combinations; the number of unique scenes is larger due to random positions and noise. See Fig. 21 for the corresponding binding-complexity and multiplicative-probe results.

Results. The two main findings from §5.3–§5.4 replicate on pixel inputs (Fig. 21). The binding maps of generalizing from-scratch vision models are well approximated by shallow MLPs (top row), and the multiplicative Global product probe recovers object recognition on held-out objects (bottom row), even under speckled noise and overlapping objects. Contrast with Fig. 19, where the same probe fails on the pretrained CLIP and DINOv2 encoders.

![](images/6354849f248eb2f2dee460fb1a671ba2c20d8cc6f7e883ec148d0f0265666e2e.jpg)

![](images/7f3ca3c2400c95272dccc3ae9147be4aec2a95dcef2619105aa696d861e46c6f.jpg)

<details>
<summary>line</summary>

| Fraction of train data | Accuracy (Solid Line) | Accuracy (Dashed Line) |
| ---------------------- | --------------------- | ---------------------- |
| 0.1                    | 0.00                  | 0.00                   |
| 0.3                    | 0.05                  | 0.25                   |
| 0.5                    | 0.35                  | 0.85                   |
| 0.7                    | 0.85                  | 0.95                   |
</details>

![](images/f03a23b9f4fb24d47925f5c943d915d8981bc641f1c04244fc75ac62bf304d0e.jpg)

<details>
<summary>line</summary>

| Fraction of train data | Solid Line | Dashed Line |
| ---------------------- | ---------- | ----------- |
| 0.1                    | 0.0        | 0.0         |
| 0.3                    | 0.0        | 0.2         |
| 0.5                    | 0.4        | 0.6         |
| 0.7                    | 0.8        | 0.9         |
</details>

![](images/ee2a4733e1f07d97d92bcef964e55223bba67ca4e48873a7ad63ed1f0c9a5852.jpg)

<details>
<summary>line</summary>

| Fraction of train data | Noise, overlap (solid line) | Noise, overlap (dashed line) |
| ---------------------- | ---------------------------- | ----------------------------- |
| 0.1                    | ~0                           | ~0                            |
| 0.3                    | ~0.2                         | ~0.5                          |
| 0.5                    | ~0.6                         | ~0.8                          |
| 0.7                    | ~0.9                         | ~1.0                          |
</details>

§c + ¦c (objects) concepts original model

Figure 21. Vision-domain results corresponding to Fig. 20. Columns correspond to the three visual complexity levels. (Top) Bindingcomplexity results, analogous to §5.3: even small MLPs approximate the binding function well, confirming that from-scratch vision encoders learn low-complexity binding functions. (Bottom) Multiplicative-probe results, analogous to §5.4: the Global product probe recovers object recognition on held-out objects, confirming that the multiplicative structure observed on multi-hot inputs extends to the vision setting. Contrast with Fig. 19, where the same probe fails on the pretrained CLIP and DINOv2 encoders.