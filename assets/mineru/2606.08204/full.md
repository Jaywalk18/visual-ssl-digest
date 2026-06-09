# Neural Field Tokenizations with Hierarchy and Spatial Locality Priors

Alonso Urbano1∗ David W. Romero2 Max Zimmer1 Sebastian Pokutta1,3

1 Department for AI in Society, Science, and Technology, Zuse Institute Berlin (ZIB), Germany

2 Cartesia AI, San Francisco, CA, USA

3 Institute of Mathematics, Technische Universität Berlin, Germany

{urbano,zimmer,pokutta}@zib.de dwromerog@gmail.com

## Abstract

Neural fields parameterize data as functions from coordinates to values, providing a unified framework for representation learning across modalities. Existing approaches are dominated by per-sample meta-learning, which scales poorly due to memory-intensive inner-loop optimization. The natural alternative –feed-forward encoding– typically introduces modality-specific assumptions, sacrificing the generality that makes learning with neural fields attractive. We argue that locality and hierarchy are useful priors for learning field representations that can be injected without compromising modality-agnosticism. We propose LH-NeF, a framework to learn general-purpose tokenized representations of continuous signals. A localitypreserving hierarchical encoder maps raw coordinate-value field observations to structured tokens, from which the field is reconstructed during training. By replacing meta-learning’s inner loop with a single forward pass, LH-NeF uses 42× less memory and supports 133× larger batches than the strongest modality-agnostic baseline. Across images, 3D shapes, and climate fields, our learned representations match or exceed performance of modality-agnostic, modality-specific, and specialized generative neural field baselines on both reconstruction and downstream tasks.

## 1 Introduction

Neural fields represent data as continuous functions from coordinates to values parameterized by neural networks that can be queried at arbitrary resolution, e.g. an image as a mapping $\dot { f _ { \theta } } : [ 0 , 1 ] ^ { 2 } \to \mathbb { R } ^ { 3 }$ . Learning over entire datasets is enabled by conditional neural fields, where a shared network $f _ { \theta } ( x ; z )$ is conditioned on latent variables z representing each instance in the data (Park et al., 2019; Mescheder et al., 2019). While many methods focused on specific modalities (e.g. 2D grids or radiance fields), Dupont et al. (2022a) proposed Functa, a unified framework to learn latent representations $z ~ \in ~ \mathbb { R } ^ { n }$ of neural fields across modalities. The resulting latents are then used as data surrogates for arbitrary downstream tasks such as generation and classification (Bauer et al., 2024; Wessels et al., 2025). However, existing modalityagnostic neural field methods face a fundamental tradeoff between introducing useful structure into these representations and preserving the modality-agnosticism that makes them attractive.

In standard representation learning, structure is introduced through encoder design, e.g., local receptive fields in CNN (LeCun et al., 1998), patch tokenization in ViT (Dosovitskiy et al., 2021), message passing in GCN (Kipf and Welling, 2017). These design choices shape the latent space in ways that improve efficiency and generalization (Bronstein et al., 2021), but typically assume specific input structure, which constrains the modality-agnosticism of neural fields when used to obtain the conditioning latent z. As a result, current modality-agnostic neural field methods (Functa and its descendants) avoid obtaining latents through structured encoders entirely, and resort to decoder-only setups where the latents are obtained per-sample through meta-learning (Finn et al., 2017; Dupont et al., 2022a) or auto-decoding (Park et al., 2019). While effective, this reliance on per-sample optimization has practical and theoretical costs.

![](images/2433aad951cd8d83570b9b3e6fc44028fdc368b598715b2a842ebfc69c23bdec.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Any modality f : X → ℝ^C_out"] --> B["Embed observations {(x_i, v_i)} + Locality-preserving re-order"]
  B --> C["N × C_emb"]
  C --> D["Encoder\nHierarchical grouped attention\nG_1 > G_2 > ... > G_L"]
  D --> E["Grouped tokens Y^(L)"]
  F["w_g(x) = softmax(-d_X(x, μ_g)^2 / 2σ_θ^2)"] --> G["Σ h(x)"]
  G --> H["ICA"]
  H --> I["KA V"]
  H --> J["KA V"]
  H --> K["KA V"]
  H --> L["KA V"]
  H --> M["KA V"]
  H --> N["KA V"]
  H --> O["KA V"]
  H --> P["KA V"]
  H --> Q["KA V"]
  H --> R["KA V"]
  H --> S["KA V"]
  H --> T["KA V"]
  H --> U["KA V"]
  H --> V["KA V"]
  H --> W["KA V"]
  H --> X["KA V"]
  H --> Y["KA V"]
  H --> Z["KA V"]
  H --> AA["KA V"]
  H --> AB["KA V"]
  H --> AC["KA V"]
  H --> AD["KA V"]
  H --> AE["KA V"]
  H --> AF["KA V"]
  H --> AG["KA V"]
  H --> AH["KA V"]
  H --> AI["KA V"]
  H --> AJ["KA V"]
  H --> AK["KA V"]
  H --> AL["KA V"]
  H --> AM["KA V"]
  H --> AN["KA V"]
  H --> AO["KA V"]
  H --> AP["KA V"]
  H --> AQ["KA V"]
  H --> AR["KA V"]
  H --> AS["KA V"]
  H --> AT["KA V"]
  H --> AU["KA V"]
  H --> AV["KA V"]
  H --> AW["KA V"]
  H --> AX["KA V"]
  H --> AY["KA V"]
  H --> AZ["KA V"]
  H --> BA["KA V"]
  H --> BB["KA V"]
  H --> BC["KA V"]
  H --> BD["KA V"]
  H --> BE["KA V"]
  H --> BF["KA V"]
  H --> BG["KA V"]
  H --> BH["KA V"]
  H --> BI["KA V"]
  H --> BJ["KA V"]
  H --> BK["KA V"]
  H --> BL["KA V"]
  H --> BM["KA V"]
  H --> BN["KA V"]
  H --> BO["KA V"]
  H --> BP["KA V"]
  H --> BQ["KA V"]
  H --> BR["KA V"]
  H --> BS["KA V"]
  H --> BT["KA V"]
  H --> BU["KA V"]
  H --> BV["KA V"]
  H --> BW["KA V"]
  H --> BX["KA V"]
  H --> BY["KA V"]
  H --> BZ["KA V"]
  H --> CA["KA V"]
  H --> CB["KA V"]
  H --> CC["KA V"]
  H --> CD["KA V"]
  H --> CE["KA V"]
  H --> CF["KA V"]
  H --> CG["KA V"]
  H --> CH["KA V"]
  H --> CI["KA V"]
  H --> CJ["KA V"]
  H --> CK["KA V"]
  H --> CL["KA V"]
  H --> CM["KA V"]
  H --> CN["KA V"]
  H --> CO["KA V"]
  H --> CP["KA V"]
  H --> CQ["KA V"]
```
</details>

Figure 1: LH-NeF overview. (a) Observations from an input are embedded into the $C _ { \mathrm { e m b } }$ -dimensional tokenizer input space (Appx. B.1) and sorted by a locality-preserving ordering, then processed by L grouped attention blocks, yielding the tokenized representation $\bar { \mathbf { Y } } ^ { ( L ) }$ . The locality-preserving ordering ensures each group’s spatial support covers a compact region of the coordinate space $( \sqsupset / \sqsupset )$ . (b) To render the field ${ \bar { f } } _ { \theta }$ at any coordinate $x ,$ the renderer routes x to the k nearest groups, weighting each group’s contribution via a learnable Gaussian kernel. Per-group cross-attention outputs are then aggregated into $\mathbf { h } ( x )$ via a weighted sum, FiLM-conditioned on the query’s relative coordinate ${ \tilde { x } } ,$ and decoded into the field value $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ .

First, the resulting optimization-derived latent space lacks explicit structure, which makes learning inefficient. Recent methods address this by introducing structure through geometry-grounded mechanisms in the decoding process (Wessels et al., 2025) or modality-specific latent layouts (Bauer et al., 2024). However, these approaches remain tied to optimization-based latent fitting and in many instances sacrifice modality-agnosticism at the cost of adding structure –see Table 1 for an overview. Second, meta-learning approaches –the dominant paradigm on state-of-the-art methods– require storing the full computation graph in memory across all inner steps to compute second-order gradients. At moderate resolutions, this severely limits batch sizes even at moderate resolutions (Table 5), a major bottleneck for scaling to high resolutions or large datasets. Overall, current approaches face a trade-off between (i) structural priors that aid learning, (ii) modality-agnosticism, and (iii) the scalability bottleneck of training with per-sample optimization.

We argue that hierarchy (fine-to-coarse organization) and locality (correlation between nearby coordinates) are inductive biases that can be used to learn continuous signal tokenizations without compromising modality-agnosticism –both have in fact been proved to be excellent priors across images (LeCun et al., 1998; Vahdat and Kautz, 2020), 3D geometry (Qi et al., 2017; Wang et al., 2023) and continuous signals in general (Mallat, 1989). To this end, we propose LH-NeF (Locality-preserving Hierarchical Neural Fields). LH-NeF consists of a tokenizer that builds upon general perception systems exploiting hierarchy through layered grouped attention (Hierarchical Perceiver (Carreira et al., 2022)), and modifies them to respect coordinate-space locality. As a result, the spatial support of each group corresponds to a compact region in the domain (Figure 2). To render field values from the resulting tokenization, LH-NeF defines a renderer that exploits this structure through Gaussian soft group routing and group-wise cross-attention. The same architecture handles any coordinate domain, requiring only a locality-preserving ordering for the domain geometry, and infers neural field representations in a single forward pass –eliminating the memory bottleneck that limits the scalability of existing modality-agnostic neural field methods.

## Our contributions are:

• We identify a tradeoff between scalability, structural priors and modality-agnosticism in neural field representation learning, and show that hierarchy and locality can be introduced without modality-specific architectural assumptions.

Table 1: Methods for learning on datasets of neural fields. OPT = optimization-based. † labels used during meta-learning.

<table><tr><td>Method</td><td>Latent structure</td><td>Latent inference</td><td>Modality</td><td>Objective</td></tr><tr><td>GEM (Du et al., 2021)</td><td>Vector</td><td>OPT (autodec.)</td><td>Any</td><td></td></tr><tr><td>GASP (Dupont et al., 2022b)</td><td>Vector</td><td>—</td><td>Any</td><td>Generation</td></tr><tr><td>DPF (Zhuang et al., 2023)</td><td>—</td><td>—</td><td>Any</td><td></td></tr><tr><td>MWT (Gielisse and van Gemert, 2025)</td><td>Weights</td><td>OPT (MAML)</td><td>Any</td><td>Classif. $^{\dagger}$ </td></tr><tr><td>NF2Vec (Ramirez et al., 2024)</td><td>Vector</td><td>OPT (fit+enc.)</td><td>3D</td><td></td></tr><tr><td>3DS2VS (Zhang and Wonka, 2023)</td><td>Vector set</td><td>Forward pass</td><td>3D</td><td></td></tr><tr><td>Spatial Functa (Bauer et al., 2024)</td><td>2D vector grid</td><td>OPT (MAML)</td><td>2D</td><td>Repr. learn</td></tr><tr><td>Functa (Dupont et al., 2022a)</td><td>Vector</td><td>OPT (MAML)</td><td>Any</td><td></td></tr><tr><td>ENF (Wessels et al., 2025)</td><td>Point cloud</td><td>OPT (MAML)</td><td>Any</td><td></td></tr><tr><td>LH-NeF (Ours)</td><td>Structured tokens</td><td>Forward pass</td><td>Any</td><td>Repr. learn</td></tr></table>

• We propose LH-NeF, a modality-agnostic framework for learning neural field representations. The LH-NeF tokenizer produces a spatially structured grouped tokenization that conditions a renderer, which decodes the field at any coordinate via soft group routing and cross-attention aggregation.  
• We demonstrate empirically that LH-NeF stays competitive or exceeds state-of-the-art reconstruction and downstream performance on images, 3D shapes, and climate data, while using ∼42× less memory and fitting 133× larger batch sizes than the strongest modality-agnostic baseline.

Our code is publicly available at link-hidden-for-double-blind-review.

## 2 Related work

Neural fields and conditional neural fields. Neural fields parameterize signals as continuous functions $f _ { \theta } : \mathbb { R } ^ { d }  \mathbb { R } ^ { c }$ via coordinate-based neural networks. Early works fit a separate network per signal (NeRF (Mildenhall et al., 2020), SIREN (Sitzmann et al., 2020)), with acceleration via hash grids (Müller et al., 2022) and other spatial structures (Chen et al., 2022; Barron et al., 2021). To extend to datasets, DeepSDF (Park et al., 2019) introduced Conditional Neural Fields with auto-decoding: a single shared MLP conditioned on per-shape latent codes optimized jointly with network weights. Functa (Dupont et al., 2022a) extended this paradigm to learning modalityagnostic representations for downstream tasks. This paradigm has been widely adopted, e.g., for medical imaging (Friedrich et al., 2026), video (Wolleb et al., 2025b,a), PDEs (Jo et al., 2025; Wessels et al., 2024), and generation (Du et al., 2021; Dupont et al., 2022b; Zhuang et al., 2023) –with diffusion models trained on conditioning variable spaces (Peebles and Xie, 2023; Ho et al., 2020; Chen et al., 2024; Kim et al., 2023). All modality-agnostic methods in this family rely on per-sample optimization (MAML or auto-decoding) to obtain latents. Several works observe that the resulting latent vectors are difficult to classify directly (Wolleb et al., 2025b; Gielisse and van Gemert, 2025), and Gielisse and van Gemert (2025) showed that more fitting steps improve reconstruction but hurt classification. Recent modality-agnostic works improve performance by adding structure: 2D latent grids (Bauer et al., 2024), geometry-grounded point clouds with symmetry-specific biinvariants and Gaussian-weighted conditioning (Wessels et al., 2025) or weight-space embeddings directly (Ramirez et al., 2024; Navon et al., 2023; Gielisse and van Gemert, 2025). Akin to our method, 3DShape2VecSet (Zhang and Wonka, 2023) encodes shapes as unstructured latent vector sets, but lacks spatial structure and is modality-specific. At query time, several local conditioning methods read from multiple features via interpolation or nearest neighbors on feature grids (Chen et al., 2021; Lee and Jin, 2022; Yu et al., 2021; Peng et al., 2020). Our renderer applies the same principle to modality-agnostic grouped tokens under a locality-preserving ordering.

Forward-pass inference of neural field representations. Encoder-based methods that bypass per-sample optimization of the field representation exist, but typically in modality-specific settings. An early example is Occupancy Networks (Mescheder et al., 2019), which pair a PointNet/ResNet encoder with a shared occupancy decoder; amortized, but restricted to 3D shapes. In 2D, LIIF (Chen et al., 2021) and LTE (Lee and Jin, 2022) pair CNN encoders with local coordinate-conditioned MLP decoding, conditioning each query on its nearest grid features with distance-based weighting. INFD (Chen et al., 2024) uses a CNN encoder for neural field diffusion and leverages neural field properties to obtain state-of-the-art multi-scale generation. In 3D, pixelNeRF (Yu et al., 2021)

and LRM (Hong et al., 2024) condition radiance/tri-plane representations on image features with camera models. ConvOccNet (Peng et al., 2020) uses trilinear interpolation from volumetric feature grids and 3DILG (Zhang and Wonka, 2022) conditions on irregular latent point sets with Gaussianweighted aggregation. For PDEs, AROMA (Serrano et al., 2024) uses a Perceiver-style encoder. In general, these methods are either tied to a specific input modality or lack a more general structure that enables learning across modalities. LH-NeF bridges this gap. It encodes the field in grouped tokens by operating on the field’s coordinate-value observations, without relying on modalityspecific components. Outside of neural fields literature, modality-agnostic models exist. Perceiver IO (Jaegle et al., 2022) uses cross-attention to learn in a modality agnostic way. However, its learned latent bottleneck lacks spatial structure and locality. Building on Perceiver IO, Hierarchical Perceiver (HiP) (Carreira et al., 2022) adds hierarchy but violates locality, which our ablations find to significantly hurt performance for data defined on metric spaces that carry a corresponding geometry (Table 4). These methods learn representations in modality-agnostic settings, but can not query inputs at arbitrary resolutions as neural fields allow.

Hierarchy, locality, and dynamic tokenization. Hierarchy and locality are fundamental inductive biases in deep learning. CNNs learn fine-to-coarse features through stacked layers of local filters with increasing receptive fields (LeCun et al., 1998), ViTs introduce locality through explicit patch-based tokenization before applying global self-attention (Dosovitskiy et al., 2021). Swin Transformer (Liu et al., 2021) combines both via hierarchical shifted windows over image patches and HiP (Carreira et al., 2022) generalizes hierarchical processing to arbitrary modalities via grouped hierarchical attention. However, HiP’s grouping follows a fixed raster scan that ignores coordinate-space geometry, violating locality. Our approach respects locality and produces tokenized representations whose geometry is input-adaptive and locality-preserving (Figure 7). The benefits of hierarchy extend beyond continuous signals. Recently, H-Net (Hwang et al., 2025) used dynamic chunking to replace fixed tokenization with input-adaptive hierarchical grouping and showed that learned hierarchical representations improve efficiency in large language models. Other work on dynamic (or adaptive) tokenization addresses this in vision settings. TokenLearner (Ryoo et al., 2021) selects tokens via spatial attention, ALIT (Duggal et al., 2025) and ElasticTok (Yan et al., 2025) adapt token count to input complexity, and GPSToken (Zhang et al., 2025) parameterizes tokens as 2D Gaussians. However, all these methods assume fixed modalities (text and images, respectively). LH-NeF combines input-adaptive spatial grouping with hierarchical attention and exploits this structure explicitly at decoding time, creating spatially coherent receptive fields at every level without grid assumptions or modality-specific windowing (Appx. A.4).

## 3 Method

We consider the problem of representing a signal as a continuous field $f _ { \theta } : \mathcal { X } \to \mathbb { R } ^ { C _ { \mathrm { o u t } } }$ , where the coordinate domain $( \mathcal { X } , d _ { \mathcal { X } } )$ is a metric space of dimension $d = \dim ( { \mathcal { X } } )$ equipped with a metric dX (e.g., for images, $\dot { \mathcal { X } } = [ - 1 , 1 ] ^ { 2 }$ with the Euclidean distance; for climate data, $\chi = S ^ { 2 }$ with the Riemannian distance). Given $N$ observations $\{ ( x _ { i } , v _ { i } ) \} _ { i = 1 } ^ { N }$ 1 of $f$ with coordinates $x _ { i } \in \mathcal X$ and values $v _ { i } = f ( x _ { i } ) \in \mathbb { R } ^ { C _ { \mathrm { o u t } } }$ , our goal is to learn a (conditional) neural field $f _ { \theta }$ that enables querying the input at any coordinate $x \in \mathcal { X }$ .

Overview. Our method has two main components (Fig. 1). First, the LH-NeF tokenizer takes $\{ ( x _ { i } , v _ { i } ) \} _ { i = 1 } ^ { N }$ of a field $f$ and produces a grouped token representation $\mathbf { Y } ^ { ( L ) }$ with spatial metadata $( \ S 3 . 1 )$ , representing the field $f .$ . Then, we define a conditional neural field: given a tokenization $\mathbf { Y } ^ { ( L ) }$ of $f$ and a query coordinate x, we route x to the spatially relevant groups of $\mathbf { Y } ^ { ( L ) }$ and aggregate information via cross-attention to decode the field value $f _ { \theta } ( \stackrel { \cdot } { x } ) ( \ S 3 . 2 )$ . The pipeline is trained end-to-end by minimizing $\ell _ { 1 }$ reconstruction loss between $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ and the ground truth $f ( x )$ .

## 3.1 LH-NeF Tokenizer

Our tokenizer builds on Hierarchical Perceivers (HiP) (Carreira et al., 2022), which process input sequences by splitting tokens into contiguous groups, applying attention within each group, and merging groups into coarser ones across blocks. The original formulation operates on generic token sequences with a fixed (flattened) rasterization order. In the neural field setting, each element of the sequence of field observations carries a coordinate $x _ { i } \in \mathcal X$ that lives on a metric space $( \boldsymbol { \mathcal { X } } , d _ { \boldsymbol { \mathcal { X } } } )$ . As a result, the contiguous grouping of HiP leads to each group’s coordinates occupying a spatial region of X over which attention exchanges information – essentially receptive fields. Under $\bar { \mathrm { H i P ^ { \circ } s } }$ default raster ordering, these regions induce slices over $\mathcal { X } , \mathrm { e . g . }$ , consecutive rows of pixels in a 2D image, or slices in a 3D voxelization as shown in Figure 2 (left). This violates the spatial locality of $( \boldsymbol { \mathcal { X } } , d _ { \boldsymbol { \mathcal { X } } } )$ as coordinates that are far in X share the same token group. Notably, the initial sequence ordering determines each group’s spatial support and, by extension, the geometry of the entire downstream hierarchy (Figure 5).

Building on this observation, we reorder input sequences before grouping using a locality-preserving permutation $\pi : \{ 1 , . . . , N \} \to \{ 1 , . . . , N \}$ derived from space-filling curves (Bader, 2013) on the input coordinates. This reordering provides contiguous groups that correspond to spatially compact regions on on X rather than slices ( / in Figure 1a), and allows locality to propagate through the hierarchy by merging nearby groups into progressively coarser spatial regions – see visualizations in Appx. A.4.

Locality-preserving permutation π. We define a localitypreserving key $\kappa : \mathcal { X }  \mathbb { N }$ and sort tokens by ascending $\kappa ( \boldsymbol { x } _ { i } )$ . For Euclidean domains $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ , we quantize coordinates $x _ { i } \in [ - 1 , 1 ] ^ { d }$ into a discrete grid of 2b bins per dimension, and use Morton ordering (bit-interleaving) as the locality key $\kappa \left( \mathrm { A p p x . \ A . 1 } \right)$ . Other Euclidean locality-preserving keys apply equally (Table 4). For non-Euclidean domains, we can construct locality-preserving keys based on transformed Hilbert curves (Ai et al., 2025), which are defined for any Riemannian manifold – $\mathrm { e . g . }$ , we use $S ^ { 2 }$ cell indices for the sphere (Appx. A.2). The permutation π is then defined by sorting tokens in ascending order of $\kappa ( x _ { i } )$ . This yields an ordering where nearby coordinates remain close in the reordered sequence (Figure 2).

After reordering, we embed the sequence as $e _ { i } = \mathrm { P r o j } _ { v } ( v _ { \pi ( i ) } ) +$ $\mathrm { P E } ( x _ { \pi ( i ) } )$ , where $\mathrm { P r o j } _ { v }$ is a learned projection and PE is a sinu-

soidal positional encoding (Vaswani et al. (2017)) on d dimensions. We then form groups by taking contiguous chunks of this reordered sequence and process it through L grouped attention blocks just $\mathcal { T } _ { g } ^ { ( \ell ) } \subseteq \{ 1 , . . . , N \}$ the set of (reordered) sequence indices that fall into group g at block ℓ.

Grouped tokens with spatial metadata. The output of each block ℓ is a set of grouped tokens $\mathbf { Y } ^ { ( \ell ) } { = } \{ \mathbf { Y } _ { g } ^ { ( \ell ) } \} _ { q { = } 1 } ^ { G _ { \ell } } , \mathbf { Y } _ { g } ^ { ( \ell ) } \in \mathbb { R } ^ { K _ { \ell } \times C _ { \ell } }$ . Note that each group g is a spatially local subset of the observed coordinate-value input pairs and is therefore input-adaptive. The final block output $\mathbf { Y } ^ { ( L ) }$ is the tokenized representation of the field. In contrast to standard HiP, which collapses onto a single group at the final block $( G _ { L } = 1 )$ , we retain multiple groups at the last block to control the granularity of the resulting tokenization $( \mathbf { e } . \mathbf { g } . , G _ { L } { = } 3 2 )$ . We summarize the spatial support of groups $g \in \{ 1 , . . . , G _ { L } \}$ at the final block by their centroid $\mu _ { g }$ and extent $\lambda _ { g } \mathbf { \dot { \sigma } }$ :

$$
\mu_ {g} = \underset {y \in \mathcal {X}} {\arg \min} \sum_ {i \in \mathcal {I} _ {g}} d _ {\mathcal {X}} (y, x _ {i}) ^ {2}, \quad \lambda_ {g} = \underset {i \in \mathcal {I} _ {g}} {\max} d _ {\mathcal {X}} (\mu_ {g}, x _ {i}), \tag {1}
$$

$\mathcal { T } _ { g } = \mathcal { T } _ { g } ^ { ( L ) }$ denotes the indices assigned to group g at the final encoder block. The centroid is the Fréchet mean of the group’s observed coordinates (the arithmetic mean on Euclidean domains $\mathcal { X } \subseteq$ $\mathbb { R } ^ { d } )$ , and the extent is the group radius. On Euclidean domains, we replace the scalar radius with the axis-aligned bounding-box extent $\begin{array} { r } { \lambda _ { g } = \operatorname* { m a x } _ { i } x _ { i } - \operatorname* { m i n } _ { i } x _ { i } \in \mathbb { R } _ { > 0 } ^ { d } , } \end{array}$ , which provides richer information that is not available on general metric spaces. Together with $\mathbf { Y } ^ { ( L ) }$ , the pairs $\{ ( \mu _ { g } , \lambda _ { g } ) \} _ { g = 1 } ^ { G _ { L } }$ (routing metadata) condition the field renderer as we explain next.

## 3.2 LH-NeF Renderer

Given conditioning variables $( \mathbf { Y } ^ { ( L ) } , \{ ( \mu _ { g } , \lambda _ { g } ) \} _ { g = 1 } ^ { G _ { L } } )$ of f and a query coordinate $x \in \mathcal { X }$ , the LH-NeF renderer produces the field value $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ in three steps: (i) routing identifies which token groups are spatially relevant to $x ,$ and assign soft weights to them, (ii) aggregation uses cross-attention to read out information of each relevant group and (iii) modulation combines per-group outputs and applies a geometry-conditioned modulation to produce the final field value $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ .

![](images/497a668656c89017ac0a8bd784353ffee488a2185eb849542e40fd8435660f2e.jpg)  
Figure 2: Group assignments on a 3D chair. Top: all groups. Bottom: routed groups for two queries (⋆).

Table 2: Reconstruction quality across modalities. Test set PSNR (dB, ↑) for images, IoU (↑) for 3D shapes, MSE (↓) for climate temperatures, and field parameter count (#Param, ↓). Bold: best.

<table><tr><td rowspan="3">Method</td><td colspan="6">Images</td><td colspan="2">3D Shapes</td><td colspan="2">Data on Manifolds</td></tr><tr><td colspan="2">CIFAR-10</td><td colspan="2">CelebA-HQ 64 $^{2}$ </td><td colspan="2">ImageNet1k 256 $^{2}$ </td><td colspan="2">ShapeNet16</td><td colspan="2">ERA5</td></tr><tr><td>PSNR↑</td><td>#Param</td><td>PSNR↑</td><td>#Param</td><td>PSNR↑</td><td>#Param</td><td>IoU↑</td><td>#Param</td><td> $T_{t}$ -MSE↓</td><td>#Param</td></tr><tr><td>Functa</td><td>38.1</td><td>2.6M</td><td>28.0</td><td>3.4M $^{\ddagger}$ </td><td>-</td><td>-</td><td>92.1</td><td>4.0M $^{\ddagger}$ </td><td>5.75E-05</td><td>4.1M $^{\ddagger}$ </td></tr><tr><td>Spatial Functa $^{\dagger}$ </td><td>39.0</td><td>425K $^{\ddagger}$ </td><td>-</td><td>-</td><td>38.4/28.3 $^{\S}$ </td><td>1.4M $^{\ddagger}$ </td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ENF</td><td>42.2</td><td>522K</td><td>34.6</td><td>3.2M $^{\ddagger}$ </td><td>27.5</td><td>817K $^{\ddagger}$ </td><td>92.9</td><td>813K $^{\ddagger}$ </td><td>8.04E-06</td><td>817K $^{\ddagger}$ </td></tr><tr><td>LH-NeF (Ours)</td><td>44.4±0.3</td><td>198K</td><td>36.1±0.2</td><td>657K</td><td>28.3</td><td>1.8M</td><td>93.1±0.2</td><td>649K</td><td>3.57E-05</td><td>307K</td></tr></table>

† 2D-grid data only, non modality-agnostic. ‡ Not reported; reproduced from provided code (cf. Appx. E).  
§ Spatial Functa uses 65K conditioning dimensions vs. ≤14K for ENF / LH-NeF; at comparable latent budget (16K), they report 28.3 dB.

(i) k-Group soft routing. We identify the k groups whose centroids are nearest to x:

$$
\mathcal {N} (x) = \underset {S \subseteq \{1, \dots , G \}, | S | = k} {\arg \min} \sum_ {g \in S} d _ {\mathcal {X}} \left(x, \mu_ {g}\right) ^ {2}, \tag {2}
$$

according to the distance $d _ { \mathcal { X } }$ in $\mathcal { X } ,$ , and each selected group gets a weight assigned via a Gaussian kernel with learnable bandwidth $\sigma _ { \theta } \in \mathbb { R } _ { > 0 } ;$ :

$$
w _ {g} (x) = \frac {\exp \left(- d _ {\mathcal {X}} (x , \mu_ {g}) ^ {2} / 2 \sigma_ {\theta} ^ {2}\right)}{\sum_ {g ^ {\prime} \in \mathcal {N} (x)} \exp \left(- d _ {\mathcal {X}} (x , \mu_ {g ^ {\prime}}) ^ {2} / 2 \sigma_ {\theta} ^ {2}\right)}, \quad g \in \mathcal {N} (x). \tag {3}
$$

The motivation for multi-group routing is continuity. With hard routing (k=1), queries would switch abruptly from one group to another as a query moves across the boundary of their receptive fields. This would cause potential discontinuities in the reconstructed field. Multi-group routing ensures $f _ { \theta }$ changes smoothly across $\mathcal { X } .$ .

While bounded receptive fields already provide coarse locality (each group only “sees” tokens from a spatial region), Gaussian weighting adds distance-dependent relevance during inference: groups whose centroids are farther away from the queried coordinate contribute less. The bandwidth $\sigma _ { \theta }$ defines a characteristic scale –the contribution of groups beyond $\sim 2 \sigma _ { \theta }$ away from the query becomes negligible. We learn $\sigma _ { \theta }$ during training and parameterize it as $\sigma _ { \theta } = \exp ( \log \sigma _ { \theta } )$ to ensure positivity.

(ii) Cross-attention aggregation. We embed the query coordinate into a query vector:

$$
\mathbf {q} (x) = \mathrm{MLP} _ {q} (\mathrm{PE} (x)) \in \mathbb {R} ^ {D} \tag {4}
$$

and use cross-attention to extract relevant information from the k-nearest groups according to the queried coordinate x. In particular, for each routed group $g \in { \mathcal { N } } ( x )$ , we apply cross-attention with $\mathbf { Y } _ { g }$ as keys/values:

$$
\mathbf {h} _ {g} (x) = \operatorname{CrossAttn} \left(\mathbf {q} (x), \mathbf {Y} _ {g}\right) \in \mathbb {R} ^ {D}. \tag {5}
$$

Each group’s contribution is then aggregated into a feature vector via a weighted sum:

$$
\mathbf {h} (x) = \sum_ {g \in \mathcal {N} (x)} w _ {g} (x) \cdot \mathbf {h} _ {g} (x). \tag {6}
$$

(iii) Geometry-conditioned modulation. Next, we use FiLM-style modulations (Perez et al., 2018) to condition h(x) based on the position of the query x within its nearest group. In general metric spaces, the query’s extent-normalized position relative to its nearest group center $\mu _ { g ^ { * } }$ is:

$$
\tilde {x} = \frac {\log_ {\mu_ {g ^ {*}}} (x)}{\lambda_ {g ^ {*}}}, \tag {7}
$$

where $\begin{array} { r } { g ^ { * } \textstyle { = } \arg \operatorname* { m i n } _ { g \in \mathcal { N } ( x ) } d _ { \mathcal { X } } ( x , \mu _ { g } ) } \end{array}$ is the nearest group, and $\log _ { \mu _ { q ^ { * } } } : \mathcal { X } \to T _ { \mu _ { g ^ { * } } } \mathcal { X }$ is the logarithmic map mapping x into the tangent space at $\mu _ { g ^ { * } }$ (which reduces to $x - \mu _ { g ^ { * } }$ on Euclidean spaces). This normalized coordinate x˜ then modulates $\mathbf { \bar { h } } ( x )$ as:

$$
(\gamma , \beta) = \mathrm{MLP} _ {\mathrm{FiLM}} (\mathrm{PE} (\tilde {x})), \quad \mathbf {h} ^ {\prime} (x) = (1 + \gamma) \odot \mathbf {h} (x) + \beta , \tag {8}
$$

which lets the renderer adapt its output based on intra-group position (e.g. near vs. far from group boundaries). The final field value is

$$
f _ {\theta} (x) = \mathrm{MLP} _ {\text { out }} (\mathbf {h} ^ {\prime} (x)) \in \mathbb {R} ^ {C _ {\text { out }}}. \tag {9}
$$

Table 3: Downstream tasks on frozen latent representations. Bold: best among modality-agnostic representation learning methods. Underline: best overall.

<table><tr><td rowspan="2">Method</td><td colspan="2">Generation (FID↓)</td><td colspan="2">Classification (Acc↑)</td><td>Forecasting (MSE↓)</td></tr><tr><td>CIFAR-10</td><td>CelebA-HQ 642</td><td>CIFAR-10</td><td>ShapeNet16</td><td>ERA5</td></tr><tr><td>GEM‡</td><td>23.8</td><td>30.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GASP‡</td><td>-</td><td>13.5</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DPF‡</td><td>15.1</td><td>13.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Spatial Functa†</td><td>16.5</td><td>-</td><td>90.3%</td><td>-</td><td>-</td></tr><tr><td>Functa</td><td>78.2</td><td>40.4</td><td>68.3%</td><td>90.3%</td><td>3.45E-03</td></tr><tr><td>ENF</td><td>23.5</td><td>33.8</td><td>82.1%</td><td>96.6%</td><td>9.44E-06</td></tr><tr><td>LH-NeF (Ours)</td><td>18.5</td><td>9.7</td><td>91.2%</td><td>96.8%</td><td>4.42E-05</td></tr></table>

† 2D-grid data only, non modality-agnostic. ‡ Generative only, non general repr. learning.

## 3.3 LH-NeF properties

The design choices above give rise to several properties worth highlighting:

• Locality enters at three scales: the tokenizer’s coordinate-ordered grouping summarizes each input into spatially compact regions; Gaussian-weighted routing at query time blends information from the k nearest groups; and FiLM modulation provides finer adaptation based on where within the nearest group’s receptive field the query lies. Because x˜ is group-normalized, this modulation is invariant to translations and axis-aligned rescalings of the group’s support, automatically sharing the same conditioning across groups (Appx. A.5, visualized in Figure 8).  
• Hierarchy. The tokenizer’s layered blocks induce a multi-scale hierarchy that is aligned with coordinate-space geometry; e.g., on dense grids, successive layers produce progressively coarser compact spatial tilings (Appendix A.4 visualizes this for 2D inputs and 3D shapes).  
• Dynamic tokenization. Because grouping operates over observations rather than over the coordinate space itself, the group support $( \mu _ { g } , \lambda _ { g } )$ adapts to the sampling density and geometry of the input. For signals with uniform sampling (e.g., images, voxelized data), the support of groups is consistent for all instances. For signals with non-uniform sampling however (e.g. point clouds), our tokenization adapts to the geometry of each instance (see Figure 7).  
• Generality. Our framework applies to any modality expressed as coordinate-value observations over a coordinate domain X . That is, the natural scope of neural fields. The method only requires distinguishing between Euclidean and non-Euclidean domains to provide a correct locality key.  
• Efficiency. Our approach replaces second-order MAML’s O(K · Mem(∇L)) inner-loop cost (Rajeswaran et al., 2019) with a single forward-backward pass, yielding ∼42× memory reduction and 133× larger batch sizes in practice at 64×64 resolution (§4.3). At inference, k-group routing maintains querying costs at $\mathcal { O } ( \bar { k } K _ { L } D )$ rather than $\mathcal { O } ( G K _ { L } D )$ (often $k \in \{ 2 , 4 \} ,$ ).

## 4 Experiments

We evaluate LH-NeF across three modalities: 2D images (CIFAR-10 322, CelebA-HQ 642, ImageNet1k 2562), 3D voxel occupancy (ShapeNet16, 323), and global climate fields (ERA5). The coordinate dimension d and locality-preserving key κ are set by the corresponding coordinate domain. Other hyperparameters are chosen based on best validation loss (full configurations in Appx. B).

We follow the usual two-stage pipeline: (i) we fit LH-NeF to the data through a reconstruction objective, and then (ii) use the learned tokenization, obtained through a forward pass on the frozen LH-NeF tokenizer, to solve downstream tasks. We compare against three functa-style neural field representation learning methods: Functa (Dupont et al., 2022a), Spatial Functa (Bauer et al., 2024) and ENF (Wessels et al., 2025). For generation, we additionally compare against dedicated generative neural field based methods (GEM (Du et al., 2021), GASP (Dupont et al., 2022b), DPF (Zhuang et al., 2023)).

## 4.1 Reconstruction

Table 2 summarizes reconstruction quality across modalities. On both images and 3D shapes, LH-NeF consistently achieves state of the art performance among modality-agnostic neural field methods, while using significantly fewer parameters –up to 5× less on CelebA 642. On ImageNet1k 2562,

Table 4: Component ablation on CelebA-HQ 642 (recon PSNR ↑, gen FID ↓) and ShapeNet16 (recon IoU ↑, cls acc ↑). †k-d tree on CelebA, Morton on ShapeNet.

<table><tr><td rowspan="2">Configuration</td><td colspan="2">CelebA-HQ 64 $^{2}$ </td><td colspan="2">ShapeNet16</td></tr><tr><td>PSNR ↑</td><td>FID ↓</td><td>IoU ↑</td><td>Acc ↑</td></tr><tr><td>Full model</td><td>36.3</td><td>9.7</td><td>93.2</td><td>96.8</td></tr><tr><td>LH-NeF Tokenizer → HiP</td><td>29.9 (−6.4)</td><td>25.7 (+16.0)</td><td>76.3 (−16.9)</td><td>95.6 (−1.2)</td></tr><tr><td>Other coord. ordering $^{\dagger}$ </td><td>35.5 (−0.8)</td><td>10.7 (+1.0)</td><td>92.5 (−0.7)</td><td>96.0 (−0.8)</td></tr><tr><td>Gaussian → power weight.</td><td>34.1 (−2.3)</td><td>13.4 (+3.7)</td><td>85.5 (−7.7)</td><td>96.8 (0.0)</td></tr><tr><td>k → 1 (hard routing)</td><td>35.6 (−0.7)</td><td>10.2 (+0.5)</td><td>84.8 (−8.4)</td><td>96.7 (−0.1)</td></tr><tr><td>No FiLM modulation</td><td>34.0 (−2.3)</td><td>13.0 (+3.3)</td><td>88.9 (−4.3)</td><td>97.0 (+0.2)</td></tr></table>

Table 5: Training efficiency on CelebA-HQ $6 4 ^ { 2 ^ { - } }$ (H100 47 GB MIG). Mem: peak step memory at B=1; Max BS: largest fitting batch.

<table><tr><td></td><td>Functa</td><td>ENF</td><td>LH-NeF</td></tr><tr><td>Training step</td><td>MAML</td><td>MAML</td><td>Fwd. + Bwd.</td></tr><tr><td>Mem (MB)</td><td>1,583</td><td>7,347</td><td>173</td></tr><tr><td>Max BS</td><td>18</td><td>2</td><td>266</td></tr></table>

Spatial Functa reaches 38.4 dB but uses a 2D conditioning latent grid of size 32×32×64=65, 536. At a comparable conditioning budget of size 8×8×256=16, 384 (vs. ours of 14, 336), we obtain competitive scores while providing a general modality-agnostic formulation (Spatial Functa is restricted to 2D signals). On ERA5, LH-NeF improves over Functa while remaining behind ENF, although we note that ENF relies on an equivariant formulation that needs prior knowledge of a symmetry group.

## 4.2 Downstream tasks

We evaluate whether the tokenizations learned by LH-NeF support downstream tasks they were never trained for, by training off-the-shelf models on the frozen tokenizations. Detailed experimental descriptions are provided in Appx. D.

Generation. We train a Diffusion Transformer (Peebles and Xie, 2023) on the (frozen) LH-NeF tokenizations. On CelebA-HQ $6 4 ^ { 2 }$ , LH-NeF achieves state of the art generation (FID↓), notably outperforming specialized generative methods and modality-agnostic neural field learning baselines (Table 3, samples in Figure 3). On CIFAR-10, we achieve the best generation score among modality-agnostic representation learning methods, while remaining behind Spatial Functa and Diffusion Probabilistic Fields (DPF), which benefit from domain and task-specific advantages respectively.

Classification. On ShapeNet16, a lightweight classifier on frozen latents reaches 96.8% accuracy, outperforming ENF (96.6%) and Functa (90.3%). On CIFAR-10, a ConvNeXt (Liu et al., 2022) classifier Fi trained on the data’s learned tokenizations reaches 10 91.2% accuracy, outperforming all modality-agnostic vis baselines (ENF 82.1%, Functa 68.3%) as well as Spatial Functa (Bauer et al., 2024) (90.3%, modality-specific).

![](images/f56c07f6893c1de97342a90949853fc4951518c0060b07dc06b27b5f5dda9790.jpg)

<details>
<summary>text_image</summary>

Functa
ENF
LH-NeF
Functa
ENF
LH-NeF
</details>

gure 3: Unconditional generation on CIFAR-(top) and CelebA-HQ 642 (bottom). Larger ualizations in Appx. 9.

Global temperature forecasting. On ERA5, we follow the ENF protocol: consecutive hourly snapshots are tokenized, a temporal predictor maps $\mathbf { Y } _ { t } { \mapsto } \mathbf { Y } _ { t + 1 }$ in latent space, and predictions are decoded through the frozen renderer (details in Appx. D.3). LH-NeF improves over Functa but trails ENF, consistent with the reconstruction gap observed in Table 2; downstream predictions decode through the frozen renderer and are therefore upper-bounded by its fidelity.

## 4.3 Ablations and efficiency assessment

Table 4 ablates each component of LH-NeF on CelebA-HQ $6 4 ^ { 2 }$ and ShapeNet16. Locality-preserving ordering is the most important component. Replacing it with raster ordering (vanilla HiP) produces −6.4 dB PSNR and +16.0 FID on CelebA; and −16.9 IoU points on ShapeNet. Morton and k-d tree are near-interchangeable (≤ 0.8 dB / IoU swap cost), confirming both locality-preserving orderings are effective. On the renderer side, Gaussian weighting and FiLM modulation are the most impactful for CelebA reconstruction (−2.3 dB each). Multi-group routing (k>1) contributes little on CelebA (−0.7 dB) but is the second-largest factor on ShapeNet (−8.4 IoU). Classification is largely unaffected by these changes, since it is a task that pools the spatial information and is therefore less affected by our locality ablation components. An extended discussion is provided in Appx. C including parallels with Spatial Functa (Bauer et al., 2024) and LIIF (Chen et al., 2021).

We benchmark the computational cost of LH-NeF according to the protocol described in Appx. F (Table 5). We use the official JAX implementations with XLA JIT compilation for Functa and ENF, which is favorable to the baselines as XLA fuses the MAML inner loop into a single optimized program. Even so, LH-NeF uses ∼42× less memory (173 MB vs. 7.3 GB) and supports 133× larger batch sizes (266 vs. 2) than ENF on the same GPU. This results from second-order MAML methods requiring maintaining the full inner-loop computation graph across all K inner steps, inflating persample memory by a factor of K relative to a single forward-backward pass. Our method fits latents in one forward pass and still matches or improves over ENF in quality.

## 5 Conclusion

We presented LH-NeF, a framework for learning tokenizations of neural fields across metric coordinate domains that uses hierarchy and locality as inductive biases in its encoding and decoding process. LH-NeF achieves competitive or state-of-the-art performance on reconstruction and downstream tasks across 2D images, 3D shapes, and spherical climate data. By replacing per-sample meta-learning with a learned tokenization, our method uses significantly less memory and facilitates much larger batch sizes. Altogether, LH-NeF serves as a scalable modality-agnostic foundational method for neural field representation learning.

Limitations and broader perspective. Fundamentally, our choice of leveraging hierarchy and locality biases is a deductive approach motivated by structural properties observed in data across modalities. A complementary, more inductive approach is learning which biases generalize across tasks from a distribution over modalities and tasks (Baxter, 2000); this is a promising future work direction, as well as an active area of research, particularly on symmetry-related priors (Romero and Lohit, 2022; van der Linden et al., 2024; Urbano et al., 2026). Generally, characterizing the full set of beneficial inductive biases across modalities, and understanding when learned biases should replace hand-designed ones, remains an open question. Another natural direction comes from defining a multi-hierarchy conditioning method that exploits information at all encoder levels rather than the last one, akin in spirit to multi-resolution hash encodings (Müller et al., 2022) or hierarchical latent diffusion (Kim et al., 2023). On signals with very irregular sampling (e.g. LiDAR), our locality guarantee holds only in expectation: consecutive points in the locality-preserving order may fall into different groups, and soft routing in the renderer mitigates this only partially (see Appx. A.3). A systematic study of the framework’s behavior in that regime, along with scaling to substantially higher-resolution signals remains future work.

## Acknowledgments and Disclosure of Funding

We thank David R. Wessels and David M. Knigge for their valuable discussions during the early stages of this research, and Carlos Saavedra Luque for his contributions to the design of the main figure of this paper. This research was partially supported by the Deutsche Forschungsgemeinschaft (DFG) through the DFG Cluster of Excellence MATH+ (EXC-2046/1, EXC-2046/2, project id 390685689), as well as by the German Federal Ministry of Research, Technology and Space (research campus Modal, fund number 05M14ZAM, 05M20ZBM) and the VDI/VDE Innovation + Technik GmbH (fund number 16IS23025B).

## References

Bijan Afsari. Riemannian lp center of mass: Existence, uniqueness, and convexity. Proceedings of the American Mathematical Society, 139(2):655–673, 2011. doi: 10.1090/S0002-9939-2010-10541-5.  
Mingyao Ai, Yunfan Yang, and Xiangshun Kong. Space-filling designs on riemannian manifolds. Journal of Complexity, 86:101899, 2025. ISSN 0885-064X. doi: https://doi.org/10.1016/j.jco. 2024.101899.  
Michael Bader. Space-Filling Curves: An Introduction with Applications in Scientific Computing, volume 9 of Texts in Computational Science and Engineering. Springer, 2013.  
Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-NeRF: A multiscale representation for anti-aliasing neural radiance fields. In IEEE/CVF International Conference on Computer Vision, 2021.  
Matthias Bauer, Emilien Dupont, Andrew Brock, Dan Rosenbaum, Jonathan Schwarz, and Hyunjik Kim. Spatial functa: Scaling functa to ImageNet classification and generation. In International Conference on Learning Representations, 2024.  
Jonathan Baxter. A model of inductive bias learning. Journal of Artificial Intelligence Research, 12: 149–198, 2000.  
Michael M. Bronstein, Joan Bruna, Taco Cohen, and Petar Velickovi ˇ c. Geometric deep learning: ´ Grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478, 2021.  
João Carreira, Skanda Koppula, Daniel Zoran, Adrià Recasens, Catalin Ionescu, Olivier Henaff, Evan Shelhamer, Relja Arandjelovic, Matt Tompson, Andrew Brock, and Andrew Zisserman. HiP: Hierarchical perceiver. arXiv preprint arXiv:2202.10890, 2022.  
Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. TensoRF: Tensorial radiance fields. In European Conference on Computer Vision, 2022.  
Yinbo Chen, Sifei Liu, and Xiaolong Wang. Learning continuous image representation with local implicit image function. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.  
Yinbo Chen, Oliver Ren, and Vincent Sitzmann. Image neural field diffusion models. In CVPR, 2024.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.  
Yilun Du, Katherine M. Collins, Joshua B. Tenenbaum, and Vincent Sitzmann. Learning signalagnostic manifolds of neural fields. In NeurIPS, 2021.  
Shivam Duggal et al. Adaptive length image tokenization via recurrent allocation. In ICLR, 2025.  
Emilien Dupont, Hyunjik Kim, S. M. Ali Eslami, Danilo Jimenez Rezende, and Dan Rosenbaum. From data to functa: Your data point is a function and you can treat it like one. In International Conference on Machine Learning, 2022a.  
Emilien Dupont, Yee Whye Teh, and Arnaud Doucet. Generative models as distributions of functions. In International Conference on Artificial Intelligence and Statistics, 2022b.  
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In ICML, 2017.  
Paul Friedrich, Florentin Bieder, Julian McGinnis, Julia Wolleb, Daniel Rueckert, and Philippe C. Cattin. MedFuncta: A unified framework for learning efficient medical neural fields. In MIDL, 2026.  
Sander Gielisse and Jan van Gemert. End-to-end implicit neural representations for classification. In CVPR, 2025.  
Google. S2 Geometry Library. https://s2geometry.io, 2024. Hierarchical decomposition of the sphere via Hilbert curves on the six faces of an enclosing cube.  
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.  
Aaron Holland. s2cell: Minimal python s2 cell id, s2point and lat/lon conversion library. https: //pypi.org/project/s2cell/, 2024.  
Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. LRM: Large reconstruction model for single image to 3D. In International Conference on Learning Representations, 2024.  
Sukjun Hwang, Brandon Wang, and Albert Gu. Dynamic chunking for end-to-end hierarchical sequence modeling. arXiv preprint arXiv:2507.07955, 2025.  
Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Oriol Vinyals, Andrew Zisserman, and João Carreira. Perceiver IO: A general architecture for structured inputs & outputs. In International Conference on Learning Representations, 2022.  
Minju Jo, Woojin Cho, Uvini Balasuriya Mudiyanselage, Seungjun Lee, Noseong Park, and Kookjin Lee. PDEfuncta: Spectrally-aware neural representation for PDE solution modeling. In NeurIPS, 2025.  
Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. In Advances in Neural Information Processing Systems, 2022.  
Seung Wook Kim, Bradley Brown, Kangxue Yin, Karsten Kreis, Katja Schwarz, Daiqing Li, Robin Rombach, Antonio Torralba, and Sanja Fidler. NeuralField-LDM: Scene generation with hierarchical latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.  
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In ICLR, 2017.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.  
Jaewon Lee and Kyoung Mu Jin. Local texture estimator for implicit representation function. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.  
Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In IEEE/CVF International Conference on Computer Vision, 2021.  
Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.  
Stéphane G. Mallat. A theory for multiresolution signal decomposition: The wavelet representation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 11(7):674–693, 1989.  
Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3D reconstruction in function space. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.  
Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision, 2020.  
Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics, 2022.  
Aviv Navon, Aviv Shamsian, Idan Achituve, Ethan Fetaya, Gal Chechik, and Haggai Maron. Equivariant architectures for learning in deep weight spaces. In International Conference on Machine Learning, 2023.  
Samuele Papa. Spatial functa — unofficial JAX implementation. https://github.com/ samuelepapa/spatial\_functa, 2024. Unofficial reimplementation.  
Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. DeepSDF: Learning continuous signed distance functions for shape representation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.  
William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.  
Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc Pollefeys, and Andreas Geiger. Convolutional occupancy networks. In ECCV, 2020.  
Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. FiLM: Visual reasoning with a general conditioning layer. In AAAI Conference on Artificial Intelligence, 2018.  
Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. PointNet++: Deep hierarchical feature learning on point sets in a metric space. In Advances in Neural Information Processing Systems, 2017.  
Markus N. Rabe and Charles Staats. Self-attention does not need $O ( n ^ { 2 } )$ memory. arXiv preprint arXiv:2112.05682, 2021.  
Aravind Rajeswaran, Chelsea Finn, Sham Kakade, and Sergey Levine. Meta-learning with implicit gradients. In NeurIPS, 2019.  
Pierluigi Zama Ramirez, Luca De Luigi, Daniele Sirocchi, Adriano Cardace, Riccardo Spezialetti, Francesco Ballerini, Samuele Salti, and Luigi Di Stefano. Deep learning on object-centric 3D neural fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.  
David W. Romero and Suhas Lohit. Learning partial equivariances from data. In Advances in Neural Information Processing Systems, 2022.  
Michael S Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. Tokenlearner: Adaptive space-time tokenization for videos. In NeurIPS, 2021.  
Louis Serrano, Leon Migus, Yuan Yin, Jocelyn Ahmed Mazari, and Patrick Gallinari. AROMA: Preserving spatial structure for latent PDE modeling with local neural fields. In NeurIPS, 2024.  
Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. In Advances in Neural Information Processing Systems, 2020.  
Alonso Urbano, David W. Romero, Max Zimmer, and Sebastian Pokutta. RECON: Robust symmetry discovery via explicit canonical orientation normalization. In International Conference on Learning Representations, 2026.  
Arash Vahdat and Jan Kautz. NVAE: A deep hierarchical variational autoencoder. In Advances in Neural Information Processing Systems, 2020.  
Putri A. van der Linden, Alejandro García-Castellanos, Sharvaree Vadgama, Thijs P. Kuipers, and Erik J. Bekkers. Learning symmetries via weight-sharing with doubly stochastic tensors. In Advances in Neural Information Processing Systems, 2024.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.  
Peng Wang, Haoxi Gan, Yonghong Liu, Ruigang Zhang, and He Wang. OctFormer: Octree-based transformers for 3D point clouds. In ACM Transactions on Graphics (SIGGRAPH), 2023.  
David Wessels, David M. Knigge, Erik J. Bekkers, and Patrick Forré. Grounding continuous representations in geometry: Equivariant neural fields. In International Conference on Learning Representations, 2025.  
David R. Wessels, David M. Knigge, Riccardo Valperga, Samuele Papa, Sharvaree Vadgama, Efstratios Gavves, and Erik J. Bekkers. Space-time continuous PDE forecasting using equivariant neural fields. In Advances in Neural Information Processing Systems (NeurIPS), 2024.  
Julia Wolleb, Cristiana Baloescu, Alicia Durrer, Hemant D. Tagare, and Xenophon Papademetris. Low-rank-modulated functa: Exploring the latent space of implicit neural representations for interpretable ultrasound video analysis. In MICCAI, 2025a.  
Julia Wolleb, Florentin Bieder, Paul Friedrich, Hemant D. Tagare, and Xenophon Papademetris. VidFuncta: Towards generalizable neural representations for ultrasound videos. In MICCAI, 2025b.  
Wilson Yan et al. ElasticTok: Adaptive tokenization for image and video. In ICLR, 2025.  
Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelNeRF: Neural radiance fields from one or few images. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.  
Biao Zhang and Peter Wonka. 3DILG: Irregular latent grids for 3D generative modeling. In NeurIPS, 2022.  
Biao Zhang and Peter Wonka. 3DShape2VecSet: A 3D shape representation for neural fields and generative diffusion models. In ACM SIGGRAPH, 2023.  
Hongyi Zhang, Moustapha Cissé, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In International Conference on Learning Representations (ICLR), 2018.  
Zhengqiang Zhang, Rongyuan Wu, Lingchen Sun, and Lei Zhang. GPSToken: Gaussian parameterized spatially-adaptive tokenization for image representation and generation. arXiv preprint arXiv:2509.01109, 2025.  
Peiye Zhuang, Samira Abnar, Jiatao Gu, Alexander Schwing, Joshua M. Susskind, and Miguel Ángel Bautista. Diffusion probabilistic fields. In International Conference on Learning Representations, 2023.

## A Coordinate Ordering Details

## A.1 Euclidean Locality-preserving keys

Morton / Z-order key Let $x \in [ - 1 , 1 ] ^ { d }$ . Define the per-dimension integer quantization $q _ { k } ( x ) =$  x(k)+12 (2b − 1) ∈ {0, . . . , 2b − 1}, where b is the number of quantization bits. Let bitj (qk(x)) ∈ ${ \bigl \lfloor } { \frac { x ^ { ( k ) } + 1 } { 2 } } \left( 2 ^ { b } - 1 \right) { \bigr \rfloor } \in \left\{ 0 , \ldots , 2 ^ { b } - 1 \right\}$ ${ \mathrm { b i t } } _ { j } ( q _ { k } ( x ) ) \in$ $\{ 0 , \bar { 1 } \}$ denote the j-th least significant bit of $q _ { k } ( x )$ . The Morton key is obtained by interleaving bits across dimensions:

$$
\kappa (x) = \sum_ {j = 0} ^ {b - 1} \sum_ {k = 1} ^ {d} \mathrm{bit} _ {j} (q _ {k} (x))   2 ^ {d j + (k - 1)}. \tag {10}
$$

Sorting tokens by κ produces a space-filling curve that visits points in a spatially coherent order. For d=2 this is the standard Z-curve; for d=3 it generalizes to a 3D Z-curve. On regular grids, Morton ordering produces roughly square (2D) or cubic (3D) groups that respect spatial locality at every scale.

k-d tree linearization For sparse or irregularly sampled data, Morton ordering can produce groups that span empty regions of the coordinate space (see Section A.3). An alternative key is the k-d tree linearization:

1. Given a set of N points with coordinates in $\mathbb { R } ^ { d }$ , compute the bounding box and identify the axis of maximum spread.  
2. Sort the points along that axis and split at the median.  
3. Recurse on each half until each subset contains a single point.  
4. Concatenate the leaves in depth-first, left-before-right order.

The resulting permutation places spatially nearby points at adjacent positions in the sequence.

## A.2 Non-euclidean Locality-preserving keys

For data living on a non-Euclidean manifold, the same locality-preserving idea applies. The construction is the same as in the Euclidean case (Section A.1): pick any sort key whose level sets are spatially compact on the manifold, then use the resulting permutation as the input ordering for the contiguous-chunk grouping in the LH-NeF tokenizer.

![](images/7dc22c023d713b74893c480cfec2f314de8e701884150d777cdd924fb13a07b3.jpg)

<details>
<summary>natural_image</summary>

Abstract geometric pattern with overlapping colored circles forming a stylized oval (no text or symbols)
</details>

![](images/662d7156ab21911abbcda9338e66d52ed9d147b289826b2b7586c2df5dc615fb.jpg)

<details>
<summary>text_image</summary>

HiP
</details>

![](images/9c66ceb74c041902ce67792df772ccb51bfc7199af67fb9f4b2cecf8a0c91972.jpg)

<details>
<summary>natural_image</summary>

Colorful horizontal striped pattern inside an oval frame (no text or symbols)
</details>

![](images/c6ef4f28980bd8bebe8daa540ff3b943f9b44d2dcf9c75eef5f0c45c951a3d1e.jpg)

<details>
<summary>natural_image</summary>

Colorful pixelated globe illustration with no text or symbols
</details>

$\operatorname { B l o c k } 1 \left( G = 2 8 8 \right)$

![](images/ff7bea9eaa16d02a232f7b366fc418d3506c6776536c1355f6b1fb24bd3b00d3.jpg)

<details>
<summary>text_image</summary>

LH-NeF (S2 Hilbert ordering)
</details>

$\operatorname { B l o c k } 2 \left( G = 1 4 4 \right)$

![](images/308e35c19fef8717c62f156cfdf9926a135d1ca47b420a11c2e8fc6497747e4b.jpg)

<details>
<summary>natural_image</summary>

Colorful pixelated globe illustration with no text or symbols
</details>

Block 3 (G=48)  
Figure 4: Receptive fields of the tokenizer hierarchy for the ERA5 (46×90 lat-lon grid on $S ^ { 2 } )$ LH-NeF tokenizer configuration. HiP (top): groups follow flat lat-lon raster order, producing latitude bands that lump unrelated climate regions together. LH-NeF (bottom): the S2 cell sort key produces compact, geographically coherent regions at every level. Each color represents one group.

Spherical data (ERA5). For ERA5, points live on the unit sphere $S ^ { 2 }$ embedded as $( x , y , z ) =$ (cos θ cos $\varphi ,$ cos θ sin $\varphi ,$ sin θ), which naturally handles periodicity at the antimeridian and convergence at the poles. We sort points using the S2 cell hierarchy (Google, 2024): each point is mapped to its S2 cell token at a level chosen so that the total number of cells is at least N (so that most cells contain at most one point), and the token strings are sorted lexicographically. Because S2 cell IDs are constructed from a face index plus a Hilbert-curve position within that face, a lexicographic sort over tokens directly yields a Hilbert-curve traversal of the sphere. We implement it using the s2cell Python package (Holland, 2024).

## A.3 Ordering on observations vs. coordinate space

Both orderings operate on the observed coordinate-value tokens, not on a fixed discretization of the coordinate space. Group boundaries are therefore determined by the distribution of observations. As a result, for uniformly sampled, regular data (e.g., images where every pixel is observed), the groups approximate a fixed spatial tiling and the induced regions are locality-preserving (compact) by construction. For sparse or irregularly sampled data (e.g., point clouds), denser regions receive finer partitions while sparser regions receive coarser ones, adapting to the sampling density. This enables the same architecture to handle variable input sizes and sampling density without modification.

However, we note that on this case of non-uniformly sampled observations, locality-preserving holds only in expectancy. For example, on a 3D point cloud with very irregular sampling, consecutive points in Morton order may be assigned to a different group based on the total number of observations and number of groups used. While the use of k-soft group routing alleviates this partial locality violation, a systematic study of how this affects performance on highly irregularly sampled data such as LiDAR is left for future work.

## A.4 Receptive field hierarchy

Figure 5 visualizes the receptive fields induced by the LH-NeF tokenizer hierarchy on a 32×32 CIFAR-10 input. Under HiP raster ordering (top row), each group covers a horizontal stripe, i.e. information from distant coordinates must be encoded in the same token group. Under Morton ordering (bottom row), groups form compact, spatially coherent patches at every hierarchical level: from 128 fine patches at block 1, through 64 medium patches at block 2, to 32 coarse regions at block 3. Note that groups at blocks 1 and 3 appear as 2:1 rectangles rather than perfect squares: this is because the Z-curve alternates which spatial dimension it doubles at each recursion level, so contiguous chunks of $2 ^ { 2 m }$ tokens form squares while chunks of $2 ^ { 2 m + 1 }$ tokens form 2:1 rectangles. Groups remain spatially compact however, and soft group routing aggregates group information smoothly regardless.

![](images/4431ce7da00033f7cc4eda48fc8ac06a8f0209d169b0653376927f0154b93681.jpg)  
Figure 5: Receptive fields of the tokenizer hierarchy for the CIFAR-10 (32×32) LH-NeF tokenizer configuration. HiP (top): groups are horizontal stripes with no 2D locality. LH-NeF (bottom): groups are spatially compact patches that coarsen hierarchically. Each color represents one group.

Figure 6 shows the same visualization on a ShapeNet chair $( 3 2 ^ { 3 }$ voxels) using k-d tree ordering. Since grouping depends on the observed voxel distribution, groups adapt to the shape geometry, i.e. different inputs get different group distributions (and therefore routing metadata).  
![](images/5f4523506ee39a6a84b07f1d75da61a6a39ba1df96a1ba70ec61220ce28c1fd5.jpg)

<details>
<summary>natural_image</summary>

3D rendered mechanical part with stepped structure and mounting feet (no text or symbols)
</details>

Input

![](images/737325b0334f6b0bcb8ea9424e6213445917da2e781e505bd0e50a526a8e05ee.jpg)

<details>
<summary>natural_image</summary>

Pixelated 3D model of a stylized animal figure with colorful color blocks (no text or symbols)
</details>

$\mathrm { B l o c k } 1 ( G { = } 2 5 6 )$

![](images/9b14e5df9442005139b2ad4686919eb3a65937feaabe422d17116cd18e226e6c.jpg)

<details>
<summary>natural_image</summary>

Pixelated 3D model of a stylized animal figure with colorful segmentation (no text or symbols)
</details>

$\operatorname { B l o c k } 2 \left( G = 1 2 8 \right)$

![](images/c7b50c13b2ffcc4ff62167fb5492c60877ebe950d91017bb61fc8668aa679a1b.jpg)

<details>
<summary>natural_image</summary>

3D pixelated abstract geometric structure with multicolored blocks (no text or symbols)
</details>

$\mathrm { B l o c k } 3 ( G { = } 6 4 )$  
Figure 6: Receptive field LH-NeF tokenizer hierarchy on a ShapeNet chair $( 3 2 ^ { 3 }$ voxels, k-d tree ordering).  
Figure 4 shows the same visualization for ERA5 global temperature (46×90 lat-lon grid on the unit sphere $S ^ { 2 } )$ under our LH-NeF tokenizer with the S2 cell ordering described in Section A.2. Under flat lat-lon raster ordering each group is a horizontal latitude band that spans the entire globe; under S2 ordering, groups become compact, geographically coherent patches at every level of the hierarchy.

Spatial partition on X induced by the LH-NeF tokenizer The LH-NeF renderer assigns each query coordinate to its k-nearest group centroids. Note that this mechanism induces a partition on the entire continuous coordinate space, not just the observed points. This matters on e.g. point clouds. We visualize these partitions on different ShapeNet point clouds, illustrating how (arbitrary) query coordinates in unoccupied, distant regions would be routed to nearby surface groups (Figure 7). Colored points show observations belonging to each highlighted group, while the semi-transparent colored cells show the full 3D Voronoi-like region that would assign queries to that group. Cells on the object surface extend into the surrounding empty space: any query in that region is routed to the nearest surface group(s). Each shape induces a different partition since group centroids depend on the distribution of observed samples.

![](images/bee24eb925587fe4501b86c14c0bd1bc242181fc4ac1142c67d105d883f569e3.jpg)

<details>
<summary>natural_image</summary>

Four abstract 3D wireframe models of furniture and objects, including a chair, a box, a table, and a stand, with no visible text or symbols.
</details>

Figure 7: Voronoi partition of 3D space induced by k-d tree group centroids (G=64, four groups highlighted per shape). Colored points: observed surface samples belonging to each highlighted group. Semi-transparent colored cells: the corresponding Voronoi regions extending into unoccupied space. Groups adapt each input’s point cloud geometry. From left to right: chair, bag, table, lamp.

## A.5 Invariance of the FiLM coordinate frame

The FiLM modulation in the LH-NeF renderer (Section 3.2) conditions on the group-normalized relative coordinate x˜. We show that x˜ is invariant under natural families of transformations, so that the FiLM output $( \gamma , \beta ) = \mathrm { M L P } _ { \mathrm { F i L M } } ( \mathrm { P E } ( \tilde { x } ) )$ is identical for the same local pattern at any absolute position or scale.

Euclidean domains. On $\mathcal { X } \subseteq \mathbb { R } ^ { d } .$ , the group-normalized coordinate is $\tilde { x } ^ { g } = ( x - \mu _ { g } ) / \lambda _ { g }$ with $\begin{array} { r } { \mu _ { g } = \frac { 1 } { | { \mathcal { T } } _ { g } | } \sum _ { i } x _ { i } } \end{array}$ and $\lambda _ { g } = \operatorname* { m a x } _ { i } x _ { i } - \operatorname* { m i n } _ { i } x _ { i } \in \mathbb { R } _ { > 0 } ^ { d }$ (Section 3.1). Under any transformation $T _ { t , S } : x \mapsto S x + t$ with $t \in \mathbb { R } ^ { d }$ and $S = \mathrm { d i a g } ( s _ { 1 } , \ldots , s _ { d } ) , s _ { k } > 0$ , applied jointly to all observations and the ${ \mathrm { q u e r y } } , ^ { 2 }$ the centroid transforms as $\mu _ { g } ^ { \prime } = S \mu _ { g } + t$ and the extent as $\lambda _ { g } ^ { \prime } = S \lambda _ { g }$ (element-wise, since $S$ is a positive diagonal), giving

![](images/57694e0232e57d4149d03f66b182c2ef58c3881be14c680b187304b5b122d1f9.jpg)  
Figure 8: FiLM scale parameter $\gamma$ (six different channels) evaluated on a $3 2 \times 3 2$ query grid with the trained CIFAR10 checkpoint. The modulation pattern repeats identically across all spatial groups (see overlay in Figure 5 Block 3), showing how the group-normalized coordinate frame $\tilde { x } = ( x - \mu _ { g } ) / \lambda _ { g }$ makes the FiLM output invariant to the spatial position of the group. Since all groups share the same spatial extent on uniform grids, this figure demonstrates translation invariance; on variable-geometry data (e.g. point clouds), the same property additionally ensures scale invariance across groups of different sizes.

$$
\tilde {x} ^ {\prime} = \frac {(S x + t) - (S \mu_ {g} + t)}{S \lambda_ {g}} = \frac {S (x - \mu_ {g})}{S \lambda_ {g}} = \frac {x - \mu_ {g}}{\lambda_ {g}} = \tilde {x}. \tag {11}
$$

This covers translations $( S = I )$ , uniform scaling $( S = s I )$ , and anisotropic per-axis rescaling (distinct $s _ { k } )$ . The per-axis bounding-box extent is what enables the anisotropic case: each coordinate dimension normalizes independently, so the diagonal entries of S cancel component-wise.

General Riemannian manifolds. On a complete Riemannian manifold $( \mathcal { X } , d _ { \mathcal { X } } )$ with scalar group radius $\lambda _ { g } = \operatorname* { m a x } _ { i } d _ { \mathcal X } ( \mu _ { g } , x _ { i } )$ , the analogous result holds for geodesic dilations $\begin{array} { r } { D _ { s } ( x ) = \exp _ { \mu _ { a } } ( s \cdot } \end{array}$ $\log _ { \mu _ { g } } ( x ) ) , s > 0 .$ . Let $v _ { i } = \log _ { \mu _ { q } } ( x _ { i } )$ and $v = \log _ { \mu _ { q } } ( x )$ . The Fréchet mean is preserved because $D _ { s }$ scales all tangent vectors by $s ,$ so the optimality condition $\textstyle \sum _ { i } v _ { i } = 0$ becomes s $\boldsymbol { \Sigma } _ { i } \boldsymbol { v } _ { i } = 0$ , which still holds under the same uniqueness conditions (Afsari, 2011). The radius scales to $s \lambda _ { g }$ (since $d _ { \mathcal { X } } ( \mu _ { g } , \mathrm { e x p } _ { \mu _ { g } } ( s v _ { i } ) ) = s \Vert v _ { i } \Vert _ { \mu _ { g } } ) _ { : }$ , and $\log _ { \mu _ { g } } ( D _ { s } ( x ) ) = s v .$ , giving

$$
\tilde {x} ^ {\prime} = \frac {s v}{s \lambda_ {g}} = \frac {v}{\lambda_ {g}} = \tilde {x}. \tag {12}
$$

On $\mathbb { R } ^ { d } .$ , geodesic dilation reduces to $x \mapsto \mu _ { g } + s ( x - \mu _ { g } )$ , i.e. the uniform-scaling subcase $S = s I$ of the Euclidean result above. The Euclidean formulation is strictly stronger: translations and anisotropic rescalings are additionally available because the per-axis bounding-box extent normalizes each coordinate dimension independently (structure that is unavailable on a general manifold).

The invariance provides an inductive bias analogous to weight sharing in convolutional networks: the FiLM MLP learns a single mapping from relative position to modulation parameters, and the normalization by $( \mu _ { g } , \lambda _ { g } )$ ensures this mapping generalizes across groups at different absolute locations and of different spatial extents without requiring the network to learn position- or scale-dependent features. The ablation in Table 4 confirms the empirical impact: removing FiLM modulation costs 2.3 dB PSNR on CelebA-HQ and 4.3 IoU on ShapeNet.

## B LH-NeF Implementation Details

## B.1 LH-NeF tokenizer configuration

The LH-NeF tokenizer processes re-ordered inputs with Hierarchical Perceiver (Rabe and Staats, 2021) grouped attention blocks. Each block is characterized by its number of groups G, tokens per group K, channels C, and number of self-attention layers. Tokens are progressively regrouped from many small groups (fine locality) to fewer, larger groups (coarser, more global context). The conditioning representation is the output of the final block: a structured token set of shape [B, G, K, C] that serves as the per-sample representation for the LH-NeF renderer. Unlike the original HiP, our bottleneck representation is multi-group (rather than single group), since we aim to exploit the locality of each group at query time.

Table 6 shows the block hierarchy for each dataset. Compute is concentrated at the final block (18-36 self-attention layers) with lighter processing at earlier stages (1-3 layers).

Table 6: LH-NeF tokenizer block hierarchy per dataset. Each row shows the LH-NeF tokenizer variant used on each dataset; the conditioning representation is extracted from the final block. G: groups, K: tokens/group, C: channels, SA: self-attention layers. ImageNet uses 4 blocks (block 0–3); all others use 3 (block 0–2).

<table><tr><td rowspan="2">Dataset</td><td colspan="3">Block 0</td><td colspan="3">Block 1</td><td colspan="3">Block 2</td><td colspan="3">Block 3</td><td rowspan="2">Cond. dims</td></tr><tr><td>G</td><td>K/C</td><td>SA</td><td>G</td><td>K/C</td><td>SA</td><td>G</td><td>K/C</td><td>SA</td><td>G</td><td>K/C</td><td>SA</td></tr><tr><td>CIFAR-10</td><td>128</td><td>4/64</td><td>1</td><td>64</td><td>4/96</td><td>2</td><td>32</td><td>4/16</td><td>18</td><td>-</td><td>-</td><td>-</td><td>2,048</td></tr><tr><td>CelebA-HQ  $64^2$ </td><td>256</td><td>4/96</td><td>1</td><td>128</td><td>4/128</td><td>3</td><td>64</td><td>2/24</td><td>36</td><td>-</td><td>-</td><td>-</td><td>3,072</td></tr><tr><td>ImageNet1k  $256^2$ </td><td>1,024</td><td>4/64</td><td>1</td><td>512</td><td>4/96</td><td>2</td><td>256</td><td>4/128</td><td>24</td><td>128</td><td>4/28</td><td>6</td><td>14,336</td></tr><tr><td>ShapeNet16</td><td>256</td><td>4/64</td><td>1</td><td>128</td><td>4/32</td><td>2</td><td>64</td><td>3/8</td><td>24</td><td>-</td><td>-</td><td>-</td><td>1,536</td></tr><tr><td>ERA5</td><td>288</td><td>4/96</td><td>1</td><td>144</td><td>4/128</td><td>3</td><td>48</td><td>2/24</td><td>36</td><td>-</td><td>-</td><td>-</td><td>2,304</td></tr></table>

Input embedding. All variants use sinusoidal positional encoding on input coordinates before projection into the HiP embedding space. Table 7 lists the embedding hyperparameters.

Table 7: LH-NeF input embedding and ordering configuration per dataset.

<table><tr><td></td><td>CIFAR-10</td><td>CelebA-HQ 642</td><td>ImageNet1k 2562</td><td>ShapeNet16</td><td>ERA5</td></tr><tr><td>Coord. dim d</td><td>2</td><td>2</td><td>2</td><td>3</td><td>3</td></tr><tr><td>Value dim</td><td>3 (RGB)</td><td>3 (RGB)</td><td>3 (RGB)</td><td>1 (occ.)</td><td>1 (temp.)</td></tr><tr><td>Fourier bands</td><td>16</td><td>32</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Max frequency</td><td>40</td><td>40</td><td>40</td><td>20</td><td>20</td></tr><tr><td>Embedding channels</td><td>64</td><td>64</td><td>64</td><td>64</td><td>128</td></tr><tr><td>Coord. ordering</td><td>Morton</td><td>Morton</td><td>Morton</td><td>k-d tree</td><td>S2 cell</td></tr></table>

## B.2 LH-NeF renderer configuration

Table 8 lists the LH-NeF renderer hyperparameters per dataset. All configurations render from the LH-NeF tokenization, with a point MLP head, Gaussian-windowed KNN routing with learnable σ, and FiLM modulation.

Conditioning dimensions. The $G \times K \times C$ token representation is the primary per-sample conditioning. In addition, the LH-NeF renderer receives per-group routing metadata: the group center $\boldsymbol { \mu _ { g } } \in \mathbb { R } ^ { \bar { d } }$ (mean coordinate of input points assigned to group g) and group scale $\lambda _ { g } \in \mathbb { R } ^ { \breve { d } }$ (boundingbox extent), for a total of 2Gd additional dimensions. These are computed from the input coordinates only (i.e. they are not learnable parameters); the renderer uses them for KNN group routing and relative-coordinate normalization in the FiLM path. For fixed-grid data (images, ERA5, voxels), $\mu _ { g }$ and $\lambda _ { g }$ are constant across samples and carry zero bits of per-sample information. For variable-input data (e.g. point clouds), the routing metadata varies per sample. Table 9 reports all dimensions per dataset.

Table 8: LH-NeF renderer hyperparameters per dataset.

<table><tr><td></td><td>CIFAR-10</td><td>CelebA-HQ 64 $^{2}$ </td><td>ImageNet1k 256 $^{2}$ </td><td>ShapeNet16</td><td>ERA5</td></tr><tr><td> $d_{model}$  (cross-attn)</td><td>128</td><td>128</td><td>256</td><td>256</td><td>128</td></tr><tr><td>Num. heads</td><td>8</td><td>8</td><td>8</td><td>8</td><td>4</td></tr><tr><td>k(KNN groups)</td><td>4</td><td>4</td><td>4</td><td>8</td><td>2</td></tr><tr><td> $σ_{init}$  (Gaussian)</td><td>0.25</td><td>0.35</td><td>0.5</td><td>0.75</td><td>0.35</td></tr><tr><td>MLP hidden</td><td>128</td><td>384</td><td>768</td><td>256</td><td>192</td></tr><tr><td>MLP depth</td><td>3</td><td>5</td><td>4</td><td>4</td><td>3</td></tr><tr><td>FiLM hidden</td><td>256</td><td>256</td><td>256</td><td>256</td><td>512</td></tr><tr><td>FiLM PE dim</td><td>16</td><td>16</td><td>16</td><td>8</td><td>8</td></tr><tr><td>Coord PE dim</td><td>32</td><td>16</td><td>16</td><td>32</td><td>32</td></tr></table>

Table 9: Per-sample representation dimensions: token representation $( G \times K \times C )$ and routing metadata (2Gd dimensions for group centers and scales).

<table><tr><td>Dataset</td><td>d</td><td>Tokens dim. (GKC)</td><td>Routing dim. (2Gd)</td><td>Total</td></tr><tr><td>CIFAR-10</td><td>2</td><td>2,048</td><td>128*</td><td>2,176</td></tr><tr><td>CelebA-HQ 64 $^{2}$ </td><td>2</td><td>3,072</td><td>256*</td><td>3,328</td></tr><tr><td>ImageNet1k 256 $^{2}$ </td><td>2</td><td>14,336</td><td>512*</td><td>14,848</td></tr><tr><td>ShapeNet16</td><td>3</td><td>1,536</td><td>384*</td><td>1,920</td></tr><tr><td>ERA5</td><td>3</td><td>2,304</td><td>288*</td><td>2,592</td></tr></table>

∗ Constant across samples (fixed-grid, uniformly sampled data).

## B.3 Training configuration

All datasets use AdamW for the tokenizer and Adam for the renderer, with cosine learning rate decay. Checkpoints are selected by the primary reconstruction metric: PSNR for images, IoU for 3D occupancy, MSE for ERA5.

Table 10: LH-NeF training configuration per dataset.

<table><tr><td></td><td>CIFAR-10</td><td>CelebA-HQ 64 $^{2}$ </td><td>ImageNet1k 256 $^{2}$ </td><td>ShapeNet16</td><td>ERA5</td></tr><tr><td>Tokenizer observations N</td><td>1,024</td><td>4,096</td><td>65,536</td><td>32,768</td><td>4,140</td></tr><tr><td>Batch size</td><td>384</td><td>128</td><td>52</td><td>40</td><td>64</td></tr><tr><td>Epochs</td><td>1,000</td><td>1,000</td><td>100</td><td>300</td><td>1,000</td></tr><tr><td>Tokenizer LR</td><td>1.2e-3</td><td>2.8e-4</td><td>5.5e-4</td><td>1.3e-3</td><td>1.8e-3</td></tr><tr><td>Tokenizer weight decay</td><td>5.3e-3</td><td>1.6e-4</td><td>1.7e-2</td><td>2.1e-3</td><td>7.0e-3</td></tr><tr><td>Renderer LR</td><td>1.8e-4</td><td>9.0e-4</td><td>1.6e-4</td><td>4.9e-4</td><td>2.7e-4</td></tr><tr><td>Loss</td><td>L1</td><td>L1</td><td>L1</td><td>L1</td><td>MSE</td></tr><tr><td>Selection metric</td><td>PSNR ↑</td><td>PSNR ↑</td><td>PSNR ↑</td><td>IoU ↑</td><td>MSE ↓</td></tr><tr><td>GPUs</td><td>1×A40</td><td>1×A100</td><td>2×A100</td><td>2×H200</td><td>1×A40</td></tr><tr><td>Training time</td><td>~18 h</td><td>~29 h</td><td>~87 h</td><td>~25 h</td><td>~64 h</td></tr></table>

## C Detailed Component Ablation Analysis

We extend the discussion of Table 4 from the main text.

Locality-preserving ordering. Locality dominates the ablations on dense tasks (reconstruction, generation), but classification is the exception (−1.2 pp without locality). This is because our classifier pools the spatial structure of the LH-NeF tokens via a global [CLS] readout (Appx. D.2), which discards geometric structure from the representation. Dense tasks like reconstruction and generation rely directly on the spatial structure of the tokens, where locality has a much larger effect.

Renderer components on CelebA-HQ reconstruction. Gaussian weighting and relativecoordinate FiLM modulation tie at −2.3 dB each, while multi-group routing (k>1) contributes less (−0.7 dB). The latter result mirrors Spatial Functa (Bauer et al., 2024), which reports that single-neighbor latent lookup matches or beats bilinear interpolation on images, and LIIF (Chen et al., 2021), which reports small but consistent gains from using four local neighbors for super-resolution – a task in which the field is queried often close to group boundaries, where continuity through multiple neighbors is more important.

Renderer components on CelebA-HQ generation. For generation, every renderer component except k>1 becomes significant, and both locality-preserving orderings remain effective with minor gains for Morton over k-d tree.

Renderer components on ShapeNet16. The ranking flips on ShapeNet: k>1 and Gaussian weighting become the second most impactful (−8.4 and −7.7 IoU). Multiple neighbors help on ShapeNet because the renderer queries a full 323 grid that includes many empty-space voxels: queries near shape boundaries can borrow representational capacity from empty-space group tokens via multi-group routing. Renderer ablations barely move ShapeNet classification (<1 pp), by the same global-readout argument as above.

## D Downstream Task Implementation Details

All downstream models are trained on frozen tokenizations with no gradient flow to the tokenizer or renderer.

## D.1 Generation (Latent Diffusion)

Generation uses a diffusion transformer trained on the (frozen) tokenizations learned during the LH-NeF training (reconstruction stage). Training and evaluation follows three steps: (i) preprocessing (tokenization extraction and normalization), (ii) diffusion model training, and (iii) sampling and rendering for FID computation.

Tokenization extraction and normalization. Given a frozen LH-NeF model, we run the LH-NeF tokenizer on every training and validation sample, extracting the grouped token representation $\mathbf { Y } ^ { ( L ) } \in \mathbb { R } ^ { G _ { L } \times K _ { L } \times C _ { L } }$ from the final block (for brevity we write $G , K , C$ for $G _ { L } , K _ { L } , C _ { L }$ throughout this section). Along with each token set we store the group routing metadata (centroids $\mu _ { g } \in \mathbb { R } ^ { \breve { d } }$ and extents $\lambda _ { g } \in \mathbb { R } _ { > 0 } ^ { d } )$ . Tokenizations are cast to float16 and written to sharded archives for efficient loading. We compute running per-channel statistics (mean, std) across all training samples; during diffusion training, tokenizations are normalized as $\hat { \mathbf { Y } } ^ { ( L ) } = ( \mathbf { Y } ^ { ( L ) } - \mathbf { m } ) / \mathbf { s }$ where m, $\mathbf { s } \in \mathbb { R } ^ { C }$ are the per-channel mean and standard deviation (broadcast over groups and tokens).

Diffusion model architecture. The target of the latent diffusion model is the (normalized) tokenization $\hat { \mathbf { Y } } ^ { ( L ) } \in \mathbb { R } ^ { G \times K \times C }$ . We use a DiT variant adapted to the grouped structure: self-attention within each group’s K tokens, and regrouping across blocks via cross-attention from learnable latent queries $( G \to G / 4 \to 1 )$ , so that a single global-attention layer at the coarsest level provides full cross-group communication. A symmetric decoder path with skip connections expands back to the original grouping (U-Net style). The grouping reduces self-attention cost from $\mathcal { O } ( ( G K ) ^ { 2 } )$ to $\mathcal { O } ( G \breve { K ^ { 2 } } + \breve { G } ^ { 2 } )$ across the hierarchy.

Following DiT (Peebles and Xie, 2023), all sublayers use adaLN-Zero conditioning from a noise-level embedding (the $\mathrm { E D M } c _ { \mathrm { n o i s e } } ( \sigma )$ passed through random Fourier features), with zero-initialized gates. Continuous coordinate embeddings are computed from the group centroids $\mu _ { g }$ via a random Fourier feature projection. Since all K tokens within a group share the same positional embedding (their group centroid $\mu _ { g } )$ , we add a learned per-position token-ID embedding $( j \in \{ 1 , \ldots , K \} )$ to break within-group permutation symmetry.

Training. We adopt the EDM formulation (Karras et al., 2022). Writing ${ \bf z } _ { 0 } = \hat { \bf Y } ^ { ( L ) }$ for the clean (normalized) tokenization, we perturb it with continuous, variance-exploding noise ${ \bf z } _ { \sigma } = { \bf z } _ { 0 } + \sigma \epsilon .$ and wrap the network with Karras preconditioning so that the denoised estimate is

$$
\hat {\mathbf {z}} _ {0} = c _ {\text { skip }} (\sigma) \mathbf {z} _ {\sigma} + c _ {\text { out }} (\sigma) F _ {\theta} \left(c _ {\text { in }} (\sigma) \mathbf {z} _ {\sigma}, c _ {\text { noise }} (\sigma)\right), \quad c _ {\text { noise }} (\sigma) = \frac {1}{4} \ln \sigma , \tag {13}
$$

with the standard $c _ { \mathrm { s k i p } } , c _ { \mathrm { o u t } } , c _ { \mathrm { i n } }$ coefficients of Karras et al. (2022). Noise levels are sampled as ln $\sigma \sim \mathcal { N } ( P _ { \mathrm { m e a n } } , P _ { \mathrm { s t d } } ^ { 2 } ) \stackrel { } { ( } P _ { \mathrm { m e a n } } = - 1 . 2 , P _ { \mathrm { s t d } } = 1 . 2 )$ , and we minimize the EDM-weighted denoising loss

$$
\mathcal {L} _ {\mathrm{dm}} = \mathbb {E} _ {\sigma , \epsilon} \left[ \lambda (\sigma) \| \hat {\mathbf {z}} _ {0} - \mathbf {z} _ {0} \| ^ {2} \right], \quad \lambda (\sigma) = \frac {\sigma^ {2} + \sigma_ {\text { data }} ^ {2}}{\left(\sigma \sigma_ {\text { data }}\right) ^ {2}}, \tag {14}
$$

where $\sigma _ { \mathrm { d a t a } }$ is estimated from the normalized tokenizations (≈ 1). We use AdamW and maintain an exponential moving average (EMA) of model weights for sampling.

Sampling. We generate samples with the deterministic ODE sampler of Karras et al. (2022), using 18 steps over the Karras noise schedule $( \rho { = } 7 , \sigma _ { \operatorname* { m i n } } { = } 0 . 0 0 2 , \sigma _ { \operatorname* { m a x } } { = } 8 0$ , no stochastic churn). Sampled tokenizations are denormalized $( \mathbf { Y } ^ { ( L ) } = \hat { \mathbf { Y } } ^ { ( L ) } { \cdot } \mathbf { s } + \mathbf { m } )$ , reshaped to $[ B , G , K , C ] ,$ , and passed through the frozen LH-NeF renderer to produce images. The renderer receives the sampled tokens as its tokenizer block output and queries a full coordinate grid at the target resolution. For evaluation, we generate 50,000 samples and compute FID against the real training set.

Notes on generation on variable geometries. Rendering a generated sample requires not only the grouped token representation but also the routing metadata: group centroids $\mu _ { g } \in \mathbb { R } ^ { d }$ and extents $\lambda _ { g } \in \mathbb { R } _ { > 0 } ^ { d } .$ the diffusion model only needs to generate the grouped token representation, and rendering uses the fixed-grid routing metadata. For data on variable geometries (e.g. on point clouds), the group partition depends on the distribution of observed points for each input, which differs per sample. The routing metadata must therefore be generated alongside the token representation, e.g. by concatenating it to the diffusion target.

Hyperparameters. Table 11 lists the diffusion model architecture and training configuration for each dataset.

## D.2 Classification

ShapeNet16. The ShapeNet16 classifier is a transformer encoder operating on the tokenization extracted with a frozen LH-NeF tokenizer forward pass and per-channel normalized as in §D.1. We add two kinds of blocks: intra-group blocks self-attend within each group’s K tokens, and inter-group blocks pool each group to a single vector via mean, self-attend across the G group representations, and broadcast back. This factorization is added for efficiency, with the added benefit of matching the grouped structure of our tokenization. Both block types alternate for a total of 8 layers. We then prepend a learnable CLS token to the per-group pooled representation and apply a 2-layer transformer encoder; the CLS output is fed to a linear classification head. Positional information is injected via random Fourier feature embeddings of the group centers $\mu _ { g }$ and a learned per-slot token-ID embedding to break within-group permutation symmetry.

CIFAR-10. Following the protocol of Spatial Functa (Bauer et al., 2024) and ENF (Wessels et al., 2025), we extract frozen tokenizations on a 50×-augmented training set (random 32×32 crops of 40×40 zero-padded images, horizontal flips) and an unaugmented val/test set, apply per-channel normalization as in §D.1, and train a classifier on the resulting tokenizations.

The tokenization $\mathbf { Y } ^ { ( L ) } \in \mathbb { R } ^ { G \times K \times C }$ is reshaped into a 2D feature map for convolutions: we order the G groups in row-major order by their centroids $\mu _ { g } ,$ arrange the K slots within each group as an $H _ { k } \times W _ { k }$ sub-grid, and concatenate to obtain $\mathbf { X } \in \overset { \sigma } { \mathbb { R } } ^ { C \times H \times \overline { { W } } }$ with $H = H _ { g } H _ { k } , W = W _ { g } W _ { k }$ (so $G = H _ { g } W _ { g }$ and $K = H _ { k } W _ { k } )$ . We then apply a 2-stage ConvNeXt (Liu et al., 2022) classifier: a 1×1 stem to projection dim $D _ { 0 }$ (no patchify, since X is already low-resolution), ConvNeXt blocks with stochastic depth, a strided downsampling between stages, global-average pooling, and a linear head. We additionally apply two latent-space augmentations during classifier training: Mixup (Zhang et al., 2018) on the feature map and random zeroing of spatial cells. Both are disabled at evaluation.

Table 11: Diffusion model hyperparameters per dataset. Input structure (G, K, C) is inherited from the tokenizer output representation.

<table><tr><td></td><td>CIFAR-10</td><td>CelebA-HQ 642</td></tr><tr><td colspan="3">Input structure (from LH-NeF tok.)</td></tr><tr><td>Groups G</td><td>32</td><td>64</td></tr><tr><td>Tokens/group K</td><td>4</td><td>2</td></tr><tr><td>Token dim C</td><td>16</td><td>24</td></tr><tr><td>Sequence length L = GK</td><td>128</td><td>128</td></tr><tr><td colspan="3">Architecture</td></tr><tr><td>Hidden dim (per level)</td><td>[384, 384, 480, 576]</td><td>[384, 384, 480, 576]</td></tr><tr><td>Depth (enc/proc/dec blocks)</td><td>3 (7 total)</td><td>3 (7 total)</td></tr><tr><td>Self-attend layers/block</td><td>[1, 2, 4, 4, 4, 2, 1]</td><td>[1, 2, 4, 4, 4, 2, 1]</td></tr><tr><td>Attention heads</td><td>6</td><td>6</td></tr><tr><td>MLP widening factor</td><td>4</td><td>4</td></tr><tr><td>Dropout</td><td>0</td><td>0</td></tr><tr><td colspan="3">Diffusion (EDM)</td></tr><tr><td>Parameterization</td><td>precond.  $\hat{\mathbf{z}}_0$ </td><td>precond.  $\hat{\mathbf{z}}_0$ </td></tr><tr><td> $\sigma_{\text{data}}$ </td><td>≈ 1 (measured)</td><td>≈ 1 (measured)</td></tr><tr><td> $P_{\text{mean}} / P_{\text{std}}$ </td><td>-1.2 / 1.2</td><td>-1.2 / 1.2</td></tr><tr><td> $\sigma_{\text{min}} / \sigma_{\text{max}}$ </td><td>0.002 / 80</td><td>0.002 / 80</td></tr><tr><td>ρ</td><td>7</td><td>7</td></tr><tr><td>EMA rate</td><td>0.99995</td><td>0.99995</td></tr><tr><td colspan="3">Training</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $1.0 \times 10^{-4}$ </td><td> $1.53 \times 10^{-4}$ </td></tr><tr><td>Weight decay</td><td>0.01</td><td>0</td></tr><tr><td>Batch size</td><td>128</td><td>128</td></tr><tr><td>Epochs</td><td>5000</td><td>6000</td></tr><tr><td colspan="3">Sampling</td></tr><tr><td>Sampler</td><td>Heun (EDM ODE)</td><td>Heun (EDM ODE)</td></tr><tr><td>Steps</td><td>18</td><td>18</td></tr><tr><td>Stochastic churn  $S_{\text{churn}}$ </td><td>0</td><td>0</td></tr><tr><td>Samples for FID</td><td>50,000</td><td>50,000</td></tr></table>

Table 12: ShapeNet16 classification hyperparameters.

<table><tr><td></td><td>ShapeNet16</td></tr><tr><td> $d_{model}$ </td><td>256</td></tr><tr><td>Depth (blocks)</td><td>8</td></tr><tr><td>Attention heads</td><td>8</td></tr><tr><td>MLP ratio</td><td>4.0</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td>Label smoothing</td><td>0.0</td></tr><tr><td>Pooling</td><td>CLS token</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>1.5e-3</td></tr><tr><td>Weight decay</td><td>0.29</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Epochs</td><td>100</td></tr></table>

## D.3 ERA5 forecasting

We follow the ENF protocol (Wessels et al., 2025), where consecutive hourly ERA5 temperature snapshots are tokenized and per-channel normalized (as in §D.1), and a temporal predictor $F _ { \psi }$ maps $\mathbf Y _ { t } \mapsto \mathbf Y _ { t + 1 }$ in tokenization space. Training uses a function-space loss: the predicted tokenization $\hat { \mathbf { Y } } _ { t + 1 } = F _ { \psi } ( \mathbf { Y } _ { t } )$ and the ground-truth tokenization $\mathbf { Y } _ { t + 1 }$ are both decoded through the frozen LH-NeF renderer, and the loss is the MSE between the decoded temperature fields.

Table 13: CIFAR-10 classification hyperparameters (frozen-latent ConvNeXt classifier).

<table><tr><td></td><td>CIFAR-10</td></tr><tr><td> $D_0, D_1$  (per-stage widths)</td><td>192, 384</td></tr><tr><td>Blocks per stage</td><td>4, 4</td></tr><tr><td>Depthwise kernel size</td><td>3×3</td></tr><tr><td>MLP ratio</td><td>4</td></tr><tr><td>Stochastic depth (drop_path)</td><td>0.2</td></tr><tr><td>LayerScale init</td><td> $10^{-6}$ </td></tr><tr><td>Head dropout</td><td>0.1</td></tr><tr><td>Group grid ( $H_g, W_g$ )</td><td>(4, 8)</td></tr><tr><td>Slot grid ( $H_k, W_k$ )</td><td>(2, 2)</td></tr><tr><td>Mixup  $\alpha$ </td><td>0.5</td></tr><tr><td>Token drop  $p$ </td><td>0.1</td></tr><tr><td>Label smoothing</td><td>0.1</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate (peak)</td><td> $1.24 \times 10^{-3}$ </td></tr><tr><td>Weight decay</td><td> $4.15 \times 10^{-2}$ </td></tr><tr><td>LR schedule</td><td>cosine, 1000-iter warmup</td></tr><tr><td>Batch size</td><td>1024</td></tr><tr><td>Epochs</td><td>200</td></tr></table>

Our temporal predictor uses the DiT architecture used for generation (§D.1), with the only difference being the conditioning mechanism: the diffusion model uses adaLN-Zero from a noise-level embedding, while the forecaster uses standard pre-norm (no timestep conditioning, as this is a deterministic prediction task). The model predicts a residual $\Delta \mathbf Y = \dot { F } _ { \boldsymbol \psi } ( \mathbf Y _ { t } )$ , and the forecast is $\hat { \mathbf { Y } } _ { t + 1 } = \mathbf { Y } _ { t } + \Delta \mathbf { Y }$ . ENF uses a PΘNITA MPNN (3 layers, hidden dim 256) as their temporal predictor.

Table 14: ERA5 forecasting model hyperparameters.

<table><tr><td></td><td>LH-NeF</td><td>ENF</td></tr><tr><td>Hidden dim</td><td>256</td><td>256</td></tr><tr><td>Depth (blocks)</td><td>3</td><td>3 (layers)</td></tr><tr><td>Self-attention layers per block</td><td>4</td><td>-</td></tr><tr><td>Attention heads</td><td>4</td><td>-</td></tr><tr><td>Predict residual</td><td>yes</td><td>yes</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>Adam</td></tr><tr><td>Learning rate</td><td>8.4e-5</td><td>-</td></tr><tr><td>Weight decay</td><td>3.9e-3</td><td>-</td></tr><tr><td>Batch size</td><td>128</td><td>32</td></tr><tr><td>Epochs</td><td>500</td><td>1,000</td></tr></table>

## E Baseline Reproduction

Parameter counts (marked with ‡ in Table 2) and configurations for the efficiency benchmark are reproduced from reported hyperparameters using the authors’ public code where possible.

Functa (Dupont et al., 2022a). We use the authors’ official JAX/Haiku implementation directly, with the per-dataset hyperparameters reported in their paper. All Functa models use a SIREN with hidden width 512 and $\omega _ { 0 } = 3 0$ .

Spatial Functa (Bauer et al., 2024). Spatial Functa does not provide publicly available code. We use an unofficial public JAX reimplementation (Papa, 2024) to compute parameter counts. Because this reimplementation is not validated against the authors’ original and to avoid reporting inaccurate results, we exclude Spatial Functa from the efficiency benchmark and report only parameter counts.

Table 15: ENF reproduction hyperparameters per dataset. CIFAR-10 hyperparameters and parameter count (522K) are reported by the authors. For all other datasets, parameter counts are computed from the authors’ official JAX implementation using reported hyperparameters.

<table><tr><td></td><td>CelebA-HQ 64 $^{2}$ </td><td>ImageNet1k 256 $^{2}$ </td><td>ShapeNet16</td><td>ERA5</td></tr><tr><td>Hidden dim</td><td>256</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Latent dim</td><td>64</td><td>64</td><td>32</td><td>64</td></tr><tr><td>Num. latents</td><td>36</td><td>169</td><td>27</td><td>36</td></tr><tr><td>Num. heads</td><td>3*</td><td>3</td><td>3</td><td>3</td></tr><tr><td>k (nearest)</td><td>4</td><td>4</td><td>4</td><td>4</td></tr><tr><td> $σ_{q},σ_{v}$ </td><td>2.0, 10.0</td><td>2.0, 10.0</td><td>2.0, 10.0</td><td>2.0, 8.0</td></tr><tr><td>Params</td><td>3.2M</td><td>817K</td><td>813K</td><td>817K</td></tr></table>

∗Not reported for CelebA-HQ; we use the same number of heads as in all other datasets (3).

Table 16: Functa reproduction hyperparameters per dataset.

<table><tr><td></td><td>CIFAR-10</td><td>CelebA-HQ 64 $^{2}$ </td><td>ShapeNet16</td><td>ERA5</td></tr><tr><td>Hidden width</td><td>512</td><td>512</td><td>512</td><td>512</td></tr><tr><td>Hidden layers</td><td>10</td><td>13</td><td>15</td><td>16</td></tr><tr><td> $ω_0$ </td><td>30</td><td>30</td><td>30</td><td>30</td></tr><tr><td>Params</td><td>2.6M</td><td>3.4M</td><td>4.0M</td><td>4.1M</td></tr></table>

ENF (Wessels et al., 2025). For CIFAR-10, the authors report 522K parameters. For all other datasets, we instantiate $\mathrm { E N F } \mathbf { \vec { s } }$ official JAX implementation with the hyperparameters reported in their paper and count parameters directly.

Table 17: Spatial Functa hyperparameters per dataset.

<table><tr><td></td><td>CIFAR-10</td><td>ImageNet1k 256 $^{2}$ </td></tr><tr><td>Latent grid</td><td>8×8</td><td>32×32</td></tr><tr><td>Latent channels</td><td>16</td><td>64</td></tr><tr><td>Conditioning dims</td><td>1,024</td><td>65,536</td></tr><tr><td>SIREN width</td><td>256</td><td>256</td></tr><tr><td>SIREN layers</td><td>6</td><td>8</td></tr><tr><td> $ω_0$ </td><td>10</td><td>10</td></tr><tr><td>Modulation</td><td>shift-only</td><td>shift-only</td></tr><tr><td>Lat.-to-mod. map</td><td>1×1 Conv</td><td>3×3 Conv</td></tr><tr><td>Interpolation</td><td>1-NN</td><td>1-NN</td></tr><tr><td>MetaSGD lrs</td><td>Yes</td><td>Yes</td></tr><tr><td>Params</td><td>425K</td><td>1.4M</td></tr></table>

## F Efficiency Benchmark Protocol

Table 5 benchmarks the cost of a single training step on CelebA-HQ $6 4 ^ { 2 }$ . Measurements are performed on a single NVIDIA H100 NVL GPU (47 GB MIG partition). We report parameter count, peak memory at batch size 1 and maximum batch size.

Functa. Benchmarked using the authors’ official JAX/Haiku implementation (Dupont et al., 2022a) with the reported CelebA-HQ 642 hyperparameters (Table 16). The training step is a 3-step MAML inner loop $\scriptstyle ( \eta = 1 0 ^ { - 2 }$ inner-SGD on the per-sample shift modulations) followed by a final SIREN forward pass and an outer-loop backward pass through the SIREN weights, fully fused as a single JIT-compiled XLA program (matching the ENF setup above). Maximum batch size is found via the same binary-search protocol as ENF.

ENF. Benchmarked using the authors’ official JAX implementation, with JAX 0.9.2 and XLA JIT compilation. The training step is a 3-step MAML inner loop followed by an outer-loop backward pass through all model parameters (second-order MAML), compiled as a single fused XLA program. This setup is favorable to ENF: XLA fuses the full inner-loop and outer-gradient computation, eliminating the overhead of retained computation graphs that arises in eager-mode frameworks such as PyTorch. Maximum batch size is found via binary search over [1, 4096] with 3 validation iterations per candidate.

LH-NeF. Benchmarked in PyTorch 2.10 (CUDA 12.8). The training step is a single forward pass through the tokenizer and renderer followed by a backward pass through all parameters. Maximum batch size is found via binary search over [1, 2048] with 5 validation iterations per candidate.

## G Qualitative Results

## G.1 CelebA-HQ $6 4 ^ { 2 }$ generation samples

![](images/b9bff1b647d404406b5236f6166dda9ef8ef62d4b04c1f569c2dd14b404000fb.jpg)  
Figure 9: Additional unconditional generation samples on CelebA-HQ $6 4 ^ { 2 }$ from a diffusion transformer trained on the frozen LH-NeF tokenizations.

## G.2 ShapeNet16 reconstructions

![](images/4da7720a79c6fc30717b19841bd46ef076db4059a5b1a58d61d7659a88971c9b.jpg)

![](images/8fd65dc81ea7c9663878aa2f5e57028da4791b1857b53b49bd70585e558b4d67.jpg)

![](images/5b426ae43472d84c40853cfd3aa317be7e400ce445d6af87a10695d81dafb053.jpg)

![](images/ef134edb080337a547f6ac6afa2057ec7c3ac7c69d70ec5ab7d8a4a32645d981.jpg)

![](images/c74b8268e95e869c1e63b73197b7e1fe6bae9254d4223553fb68e62f2b0caf9e.jpg)

![](images/27156bf2da619fcbcec93cba2c54313a336a8d8c2283c913ff5e9f36f663d725.jpg)  
Airplane

![](images/cf78a064c8b847fd5fa3a0a39cc13f4ff4070bac1b0aa6d7dfdaf8c50cc71d09.jpg)  
Chair

![](images/d0738963761f58060a9199050252a5ea8312841e62077fd4fb69f6db7d459b68.jpg)  
Table

![](images/f21654f47958ea36ae3f46b2bc94a92a3bb4f70777077639e5d54feebdfa4721.jpg)  
Pistol

![](images/196575b57d6c214e3c6ce604cbda3febc272719f93e61390f61997b3cbe058e6.jpg)  
Lamp  
Figure 10: ShapeNet16 voxel occupancy reconstructions at $3 2 ^ { 3 }$ resolution. Top: ground-truth occupancy grids. Bottom: LH-NeF reconstructions.

## Broader Impact

As a representation learning method that can be paired with generative models on images, it shares the standard misuse risks of image generation systems, e.g. the synthesis of unauthorized or misleading visual content. Positive impacts of modality-agnostic representation learning could include its use in scientific applications. Besides this, we do not anticipate broader societal impacts beyond those typical for foundational representation learning research.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: Every contribution claimed in the introduction and abstract is well justified by the experiments and derivations.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.  
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.  
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.  
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Last section in the main body (Section 5) discusses main limitations and future work directions.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.  
• The authors are encouraged to create a separate “Limitations” section in their paper.  
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.  
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.  
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.  
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.  
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.  
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Justification: The only claims that need proof – about the invariance of the FiLM modulation – are derived formally in Section A.5. All assumptions are clearly stated and formalized in Section 3.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.  
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.  
• All assumptions should be clearly stated or referenced in the statement of any theorems.  
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.  
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.  
• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [Yes]

Justification: Exhaustive experimental information to reproduce our results is provided throughout the Appendix. In addition, results are reproducible with the code provided at the https url in the Introduction.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.  
• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.  
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.  
• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.  
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.  
(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.  
(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).  
(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: All our results are reproducible through the code provided at the https url in the Introduction; all datasets used in our experiment are publicly available.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.  
• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.  
• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).  
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.  
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.  
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.  
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).  
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Exhaustive experimental information to reproduce our results is provided throughout the Appendix. In addition, results are reproducible with the code provided at the https url in the Introduction.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.  
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.  
• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report mean ± standard deviation over 3 independent runs with different random seeds for the main results in Table 2, with the exception of the ImageNet and ERA5 entries. These remaining entries and the downstream task experiments are reported single-seed; full multi-seed results for all experiments will be included in the camera-ready version.

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

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Reconstruction stage compute is reported per dataset in Appendix Table 10 (GPU type and count, batch size, training epochs, and approximate wall-clock training time). The training-efficiency benchmark in main text Table 5 (peak step memory and largest fitting batch size) is performed on a single NVIDIA H100 NVL GPU (47 GB MIG partition); the protocol is detailed in Appendix F.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.  
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.  
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.  
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: No conflicts with Code of Ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.  
• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.  
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

## Answer: [Yes]

Justification: A brief broader impact discussion is included in Appendix G.2. As a representation learning method that can be paired with generative models on images, LH-NeF shares the standard misuse risks of image generation systems (e.g., synthesis of unauthorized or misleading visual content). Positive impacts of modality-agnostic representation learning could include its use in scientific applications. Besides this, we do not anticipate broader societal impacts beyond those typical for foundational representation learning research.

## Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.  
• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.  
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.  
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.  
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.  
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

## Answer: [N/A]

Justification: Our pre-trained checkpoints do not have high risk of misuse, therefore no safeguards are considered.

## Guidelines:

• The answer [N/A] means that the paper poses no such risks.  
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.  
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.  
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

## Answer: [Yes]

Justification: Every asset is properly referenced.

## Guidelines:

• The answer [N/A] means that the paper does not use existing assets.  
• The authors should cite the original paper that produced the code package or dataset.  
• The authors should state which version of the asset is used and, if possible, include a URL.  
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: All assets are well documented.

## Guidelines:

• The answer [N/A] means that the paper does not release new assets.  
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.  
• The paper should discuss whether and how consent was obtained from people whose asset is used.  
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: Not involved.

## Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.  
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.  
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

## Answer: [N/A]

Justification: Not involved.

## Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.  
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.  
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.  
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

## Answer: [Yes]

Justification: Declared in the submission.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.  
• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.