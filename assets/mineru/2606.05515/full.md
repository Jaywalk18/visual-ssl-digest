# BRepCLIP: Contrastive Multimodal Pretraining on BRep Primitives for CAD Understanding

Muhammad Usama1,2

muhammad.usama@dfki.de

Didier Stricker1,2

didier.stricker@dfki.de

Mohammad Sadil Khan1,2∗†

mohammad.khan@dfki.de

Muhammad Zeshan Afzal2∗

muhammad.zeshan.afzal@dfki.de

1DFKI, Germany

2RPTU Kaiserslautern-Landau, Germany

# Abstract

Learning representations of CAD models is a largely open problem. While 3D representation learning has flourished around point clouds and meshes, the native format of CAD - boundary representations (BReps), which encodes exact parametric surfaces, curves, and their topology, has received little attention as a representation learning substrate. We introduce BRepCLIP, the first framework to align BRep geometry with language and image embeddings through contrastive pretraining. We model each CAD object as a sequence of face and edge tokens with separate discrete vocabularies for surface and curve geometry, augmented with spatial and semantic descriptors that capture surface types (e.g., cylindrical, torus, NURBS) and curve primitives (e.g., line, arc, B-spline). A transformer encoder aggregates these tokens into a global BRep embedding, aligned with CLIP’s text and image encoders via a joint contrastive objective. BRepCLIP generates more discriminative and semantically grounded embeddings than existing point-based alternatives, improving Top-1 retrieval over OpenShape by 40.4%, 22.0%, and 23.9% on ABC, CADParser, and Automate, respectively, and improving zero-shot classification on FabWave by 15% in Top-1 score. We further demonstrate its utility as a CAD-aware similarity metric for evaluating textand image-conditioned CAD generation, establishing the importance of structureaware pretraining for multimodal CAD understanding. Project page is available at https://muhammadusama100.github.io/BrepClip2026/

# 1 Introduction

Computer-aided design (CAD) is the backbone of modern engineering, underpinning the design of everything from consumer electronics to aerospace components [5, 12]. CAD models are represented as a BRep structure, which provides exact, parametric descriptions of geometry organized into faces, edges, and their topological adjacencies [18]. Unlike generic 3D assets, BRep geometry is precise by construction. Every surface has an analytic type, every edge has a defined curve, and the topology encodes how parts connect and bind one another. In practice, engineers rarely design from scratch. They search large internal repositories to find and reuse existing parts, adapting them to new specifications. This process, known as CAD retrieval, is central to reducing design time, avoiding redundant modeling, and ensuring manufacturing consistency across product lines [8]. Despite its industrial importance, learning general-purpose representations that support open-vocabulary CAD retrieval remains a largely open problem.

Existing multimodal 3D alignment methods [40, 21] learn powerful joint representations of point clouds, images, and text, demonstrating strong performance on generic 3D object understanding. However, these methods are fundamentally designed around point cloud representations and cannot be directly applied to CAD models without first discarding their native BRep structure. Converting a BRep to a point cloud reduces a precisely structured boundary representation to an unordered set of coordinates, erasing the analytic surface types, curve primitives, and topological adjacency that are intrinsic to CAD geometry. For generic 3D assets, this approximation may be acceptable, but for CAD, it is a fundamental information loss. The geometric features most critical for engineering interpretation, such as small holes, chamfered and filleted edges, sharp boundaries, face-to-face adjacency, and exact surface curvature, are precisely what point clouds fail to encode (Figure 1). A representation that cannot distinguish a cylindrical bore from a planar pocket, or a filleted edge from a sharp one, cannot support fine-grained CAD retrieval or reliable generation evaluation.

We introduce BRepCLIP, the first contrastive representation learning framework to operate directly on BRep primitives. Each CAD model is represented as a set of BRep face and edge primitives, where each primitive is encoded through sampled local geometry together with its semantic type and topological grouping. We learn discrete surface and curve tokens from these faces and edge points using a dVAE model [41]. In addition to spatial descriptors, these tokens also encode semantic descriptors. A transformer aggregates these tokens into BRep-aware tokens, which are aligned with frozen CLIP text and image encoders via a symmetric contrastive objective.

Operating directly on BReps presents a core challenge: unlike point clouds, BReps have no canonical ordering, vary in the number of faces and edges

![](images/6ce8481ae76ecd2577d434c05062edb5942dfdf683559d4074dbfdd099a12896.jpg)

<details>
<summary>text_image</summary>

Original
CAD Model
Previous Methods
Ours
Missing
small holes
Preserving
small holes
Face
Points
Edge
Points
Groundplate: wide rectangular plate with 37 holes, including
central slot, eight large circular cutouts, and symmetric side
tabs. Width-to-length ratio 1.3.
</details>

Figure 1: Compared to point clouds, our BRepaware representations (edge, face points) preserve both geometry and fine-grained structures (e.g., holes, rounded corners) for accurate CAD representation learning.

across models, and contain heterogeneous geometry types such as planes, cylinders, tori, NURBS surfaces, or lines, arcs, and B-Splines, within a single model. We address this with a hybrid dual-DVAE tokenization scheme, training separate discrete autoencoders for faces and edges to produce dedicated codebooks for surface and curve geometry. This prevents geometrically dissimilar primitives from sharing a vocabulary and allows each branch to specialize to its own geometric domain. Each token is further augmented with semantic descriptors derived from primitive type, so the transformer reasons over typed CAD entities rather than anonymous geometric patches.

We evaluate BRepCLIP on three tasks. On text-to-CAD retrieval, BRepCLIP outperforms all pointbased baselines on ABC, CADParser, and Automate. On zero-shot CAD classification, we transfer directly to FabWave [4] without fine-tuning, again exceeding point-cloud counterparts. Finally, we introduce BRepCLIP-Score, a geometry-aware metric for evaluating text- and image-conditioned CAD generation, and show it correlates more reliably with human expert judgments than CLIP score [30] or Chamfer Distance. Our contributions are as follows:

• First contrastive representation learning framework operating natively on BRep primitives, bridging CAD geometry with language and image modalities.   
• Hybrid dual-dVAE tokenization with separate discrete tokens for face and edge geometry, enabling semantically typed tokenization of heterogeneous BRep primitives.   
• State-of-the-art results on text-to-CAD retrieval and zero-shot CAD classification.   
• BRepCLIP score, a new CAD-aware similarity metric for evaluating text- and imageconditioned CAD generation, validated against human expert judgments.

# 2 Related Work

3D representation learning and CAD. 3D representation learning has progressed from PointNet’s hierarchical point aggregation [28, 29] to transformer-based self-supervised pretraining with Point-BERT [41], and more recently to multimodal alignment. ULIP [39] aligned point clouds, images, and text against a frozen CLIP space, ULIP-2 [40] scaled this through automatic caption generation from rendered views, and OpenShape [21] pushed further with multi-dataset ensembling and stronger backbones for open-world recognition. Despite their strength in generic 3D assets, these methods are ill-suited for CAD retrieval. CAD retrieval is not a coarse semantic matching problem. It requires discriminating between shapes that look globally similar but differ in the details that matter most for engineering: a threaded hole versus a smooth bore, a chamfered edge versus a fillet, an extruded pocket versus a boss. Point clouds reduce geometry to an unordered set of surface samples, discarding the analytic surface types, curve primitives, and topological adjacency that are intrinsic to CAD [9, 18]. This structural information is erased at the point of conversion. No downstream architecture can recover what was discarded at the input.

Recognizing this, a line of work has moved toward learning directly on BRep structure. BReps are the native format of CAD models, organizing geometry into typed faces such as planes, cylinders, tori, and NURBS surfaces, and typed edges such as lines, arcs, and B-splines, connected through an explicit topological graph. This structure is not incidental. It is the primary carrier of engineering semantics. UV-Net introduced UV-domain surface sampling with graph-based topology learning [9], while BRepNet exploited native BRep connectivity through message passing over faces, edges, and coedges [18]. BRep-BERT applied masked modeling over BRep subgraphs using a GNN tokenizer [23], and BRT brought attention-based encoding to boundary representations [44]. MultiCAD proposed contrastive representation learning between point clouds and CAD sequences [24], and BrepCoder aligned BRep geometry with structured CAD code for multi-task reasoning [16]. However, all of these methods target recognition, segmentation [26, 1], reconstruction [14], or within-CAD structural pretraining. None learn language or image-aligned representations over native BRep primitives, a prerequisite for open-vocabulary retrieval that BRepCLIP is the first to address.

CAD retrieval, generation, and evaluation. CAD retrieval is a practically critical but surprisingly underexplored problem. In engineering workflows, retrieval enables part reuse, design search, and manufacturing planning. These tasks demand fine-grained geometric discrimination rather than coarse object-level similarity. As surveyed in [31, 8], learning-based CAD retrieval has largely relied on shape signatures, voxel descriptors, or rendered silhouettes, none of which capture the topological and parametric richness of BReps. Early work on scan-to-CAD retrieval focused on aligning clean CAD models to noisy RGB-D scans [3], and FastCAD extended this to real-time retrieval and alignment using contrastive shape embeddings [19]. However, both operate on generic 3D representations and target scene-level alignment rather than language-driven engineering retrieval. Jones et al. proposed self-supervised pretraining directly on BRep geometry using a hybrid implicit/explicit surface representation, demonstrating strong few-shot transfer on BRep benchmarks [11]. Yet this work focuses on within-CAD recognition tasks and does not align BRep geometry with language or image modalities. OSCAR studied open-set CAD retrieval from language and image prompts [27], and CAD-RAG introduced a retrieval-augmented generation framework combining multiple modalities [2]. However, both operate on non-native representations and are not designed for large-scale contrastive pretraining over BRep structure. The recent release of CADCAP-1M from DreamCAD [15] is the largest CAD captioning dataset to date and finally makes large-scale multimodal BRep representation learning tractable. BRepCLIP is the first method to exploit it through native BRep pretraining.

The dominant direction in multimodal CAD research has meanwhile been generation. DeepCAD established the sequence-modeling view of parametric CAD [37], and subsequent work extended this to reconstruction and generation from point clouds, BReps, text, and images [22, 42, 13, 38, 20, 34, 36, 32, 6]. Yet as generative CAD has grown, evaluation has not kept pace. Generated models are typically assessed with Chamfer Distance or CLIP score. These metrics are borrowed from point cloud and vision-language literature and are blind to BRep structure. A model that produces the correct overall silhouette but wrong surface topology, missing holes, or incorrect edge types will score well on these metrics while failing every engineering criterion that matters. BRepCLIP-Score addresses this directly. It is a CAD-aware similarity metric grounded in BRep embeddings, validated against human expert judgments on outputs from six recent text-to-CAD models.

# 3 BRepCLIP Architecture

We present BRepCLIP, a multimodal CAD representation learning framework that aligns native BRep geometry with text and images through contrastive pretraining. Unlike generic multimodal 3D encoders built on point clouds, BRepCLIP operates directly on CAD primitives and treats faces and edges as first-class entities throughout the pipeline. Each CAD model is represented by face $( G _ { f } )$ and edge $( G _ { e } )$ point sets together with primitive-type semantics. We tokenize these primitives with separate face and edge tokenizers, producing dedicated discrete tokens for surface and curve geometry. The resulting face-edge token sequence is then enriched with spatial and semantic cues and processed by a transformer encoder, whose learnable [CLS] token yields a global BRep embedding for multimodal alignment.

# 3.1 Hybrid Face-Edge Tokenization

We encode BRep geometry through a tokenization scheme over faces and edges as shown in Figure 2. Unlike Point-BERT [41], which uses local point neighborhoods to group points and generate tokens, we instead use corresponding face and edge segmentation to group points semantically. We train two separate dVAEs for faces and edges. The face dVAE uses a PointNetstyle encoder with face tokenizer $F _ { T }$ and folding-based decoder $F _ { D }$ to reconstruct surface geometry. The edge dVAE uses a lightweight 1D convolutional encoder with edge tokenizer $E _ { T }$ and decoder $E _ { D }$ to reconstruct ordered curve geometry. Separate codebooks are essential as faces and edges exhibit fundamentally different geometric structures. Each dVAE is trained by minimizing a reconstruction loss with a KL regularization term as

![](images/53487809acc47354f48b154aea2eaf584e9cc0baa58cd6115a488423d7f29722.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph_FacePoints_Gf["Face Points Gf"]
        A1["Embedding"] --> B1["Face Tokenizer FT"]
        A2["Embedding"] --> B2["PointNet"]
        A3["Embedding"] --> B3["Embedding"]
        B1 --> C1["Face Decoder FD"]
        B2 --> C2["Face Decoder FD"]
        B3 --> C3["Face Decoder FD"]
        C1 --> D1["Reconstruction Rf"]
        C2 --> D2["Reconstruction Rf"]
        C3 --> D3["Reconstruction Rf"]
    end

    subgraph_EdgePoints_Ge["Edge Points Ge"]
        E1["Embedding"] --> F1["Edge Tokenizer ET"]
        E2["Embedding"] --> F2["Edge Tokenizer ET"]
        E3["Embedding"] --> F3["Edge Tokenizer ET"]
        E4["Embedding"] --> F4["Edge Tokenizer ET"]
        E5["Embedding"] --> F5["Edge Tokenizer ET"]
        E6["Embedding"] --> F6["Edge Tokenizer ET"]
        E7["Embedding"] --> F7["Edge Tokenizer ET"]
        E8["Embedding"] --> F8["Edge Tokenizer ET"]
        E9["Embedding"] --> F9["Edge Tokenizer ET"]
        E10["Embedding"] --> F10["Edge Tokenizer ET"]
        E11["Embedding"] --> F11["Edge Tokenizer ET"]
        E12["Embedding"] --> F12["Edge Tokenizer ET"]
        E13["Embedding"] --> F13["Edge Tokenizer ET"]
        E14["Embedding"] --> F14["Edge Tokenizer ET"]
        E15["Embedding"] --> F15["Edge Tokenizer ET"]
        E16["Embedding"] --> F16["Edge Tokenizer ET"]
        E17["Embedding"] --> F17["Edge Tokenizer ET"]
        E18["Embedding"] --> F18["Edge Tokenizer ET"]
        E19["Embedding"] --> F19["Edge Tokenizer ET"]
        E20["Embedding"] --> F20["Edge Tokenizer ET"]
        E21["Embedding"] --> F21["Edge Tokenizer ET"]
        E22["Embedding"] --> F22["Edge Tokenizer ET"]
        E23["Embedding"] --> F23["Edge Tokenizer ET"]
        E24["Embedding"] --> F24["Edge Tokenizer ET"]
        E25["Embedding"] --> F25["Edge Tokenizer ET"]
        E26["Embedding"] --> F26["Edge Tokenizer ET"]
        E27["Embedding"] --> F27["Edge Tokenizer ET"]
        E28["Embedding"] --> F28["Edge Tokenizer ET"]
        E29["Embedding"] --> F29["Edge Tokenizer ET"]
        E30["Embedding"] --> F30["Edge Tokenizer ET"]
        E31["Embedding"] --> F31["Edge Tokenizer ET"]
        E32["Embedding"] --> F32["Edge Tokenizer ET"]
        E33["Embedding"] --> F33["Edge Tokenizer ET"]
        E34["Embedding"] --> F34["Edge Tokenizer ET"]
        E35["Embedding"] --> F35["Edge Tokenizer ET"]
        E36["Embedding"] --> F36["Edge Tokenizer ET"]
        E37["Embedding"] --> F37["Edge Tokenizer ET"]
        E38["Embedding"] --> F38["Edge Tokenizer ET"]
        E39["Embedding"] --> F39["Edge Tokenizer ET"]
        E40["Embedding"] --> F40["Edge Tokenizer ET"]
        E41["Embedding"] --> F41["Edge Tokenizer ET"]
        E42["Embedding"] --> F42["Edge Tokenizer ET"]
        E43["Embedding"] --> F43["Edge Tokenizer ET"]
        E44["Embedding"] --> F44["Edge Tokenizer ET"]
        E45["Embedding"] --> F45["Edge Tokenizer ET"]
        E46["Embedding"] --> F46["Edge Tokenizer ET"]
        E47["Embedding"] --> F47["Edge Tokenizer ET"]
        E48["Embedding"] --> F48["Edge Tokenizer ET"]
        E49["Embedding"] --> F49["Edge Tokenizer ET"]
        E50["Embedding"] --> F50["Edge Tokenizer ET"]
    end

    subgraph Reconstruction_Rf
        Rf_1
    end

    note1["CD(Rf,Gf) + DKL"]
    note2["1D Conv"]
    note3["Re(Re,Ge) + DKL"]
    note4["Re(Re,Ge) + DKL"]
    note5["Re(Re,Ge) + DKL"]
    note6["Re(Re,Ge) + DKL"]
    note7["Re(Re,Ge) + DKL"]
    note8["Re(Re,Ge) + DKL"]
    note9["Re(Re,Ge) + DKL"]
    note10["Re(Re,Ge) + DKL"]
    note11["Re(Re,Ge) + DKL"]
    note12["Re(Re,Ge) + DKL"]
    note13["Re(Re,Ge) + DKL"]
    note14["Re(Re,Ge) + DKL"]
    note15["Re(Re,Ge) + DKL"]
    note16["Re(Re,Ge) + DKL"]
    note17["Re(Re,Ge) + DKL"]
    note18["Re(Re,Ge) + DKL"]
    note19["Re(Re,Ge) + DKL"]
    note20["Re(Re,Ge) + DKL"]
    note21["Re(Re,Ge) + DKL"]
    note22["Re(Re,Ge) + DKL"]
    note23["Re(Re,Ge) + DKL"]
    note24["Re(Re,Ge) + DKL"]
    note25["Re(Re,Ge) + DKL"]
    note26["Re(Re,Ge) + DKL"]
    note27["Re(Re,Ge) + DKL"]
    note28["Re(Re,Ge) + DKL"]
    note29["Re(Re,Ge) + DKL"]
    note30["Re(Re,Ge) + DKL"]
    note31["Re(Re,Ge) + DKL"]
    note32["Re(Re,Ge) + DKL"]
    note33["Re(Re,Ge) + DKL"]
    note34["Re(Re,Ge) + DKL"]
    note35["Re(Re,Ge) + DKL"]
    note36["Re(Re,Ge) + DKL"]
    note37["Re(Re,Ge) + DKL"]
    note38["Re(Re,Ge) + DKL"]
    note39["Re(Re,Ge) + DKL"]
    note40["Re(Re,Ge) + D KL"]
    note41["Re(Re,Ge) + D KL"]
    note42["Re(Re,Ge) + D KL"]
    note43["Re(Re,Ge) + D KL"]
    note44["Re(Re,Ge) + D KL"]
    note45["Re(Re,Ge) + D KL"]
    note46["Re(Re,Ge) + D KL"]
    note47["Re(Re,Ge) + D KL"]
    note48["Re(Re,Ge) + D KL"]
    note49["Re(Re,Ge) + D KL"]
    note50["Re(Re,Ge) + D KL"]
```
</details>

Figure 2: Hybrid dual-dVAE tokenization. Face and edge points are tokenized independently using separate discrete VAEs with dedicated codebooks.

$$
\mathcal {L} _ {x} = C D (R _ {x}, G _ {x}) + D _ {K L}, \tag {1}
$$

where x can be either face or edge, and $C D ( R _ { x } , G _ { x } )$ denotes the Chamfer Distance between sampled points from the reconstructed and ground-truth geometry, and $D _ { K L }$ is the KL divergence regularizing the discrete latent space [41].

# 3.2 Structure-Aware Global BRep Encoding

After obtaining discrete face tokens $F _ { T } ( G _ { f } )$ and edge tokens $E _ { T } ( G _ { e } )$ , we construct a unified BRep sequence by concatenating them with a learnable [CLS] token as

$$
\mathbf {Z} ^ {\mathbf {B}} = \left[ \mathrm{CLS}; F _ {T} \left(G _ {f}\right) + f _ {m} + f _ {s} + f _ {d}; E _ {T} \left(G _ {e}\right) + e _ {m} + e _ {s} + e _ {d} \right] \tag {2}
$$

where $f _ { m }$ and $e _ { m }$ are modality indicators distinguishing faces from edges, $f _ { s }$ and $e _ { s }$ are spatial descriptors derived from primitive centroids, and $f _ { d }$ and $e _ { d }$ are semantic descriptors encoding primitive type. The geometry and modality terms form the content embedding of each token, while the spatial and semantic terms form its positional embedding. This sequence is processed by a transformer encoder, and the final [CLS] representation serves as the global BRep embedding, capturing both 3D structure and fine-grained surface and curve semantics.

# 3.3 Multimodal Contrastive Alignment

BRepCLIP aligns BRep geometry with text and image modalities through a three-branch contrastive framework consisting of a structure-aware BRep encoder, a frozen CLIP text encoder, and a frozen CLIP image encoder [30]. The BRep branch encodes the face-edge token sequence with a transformer encoder, producing a global shape embedding $\mathbf { Z } ^ { B } \in \mathbb { R } ^ { d }$ from the [CLS] token via a lightweight MLP projection head. In parallel, the frozen CLIP text and image encoders produce embeddings ${ \mathbf Z } ^ { T }$ and $\bar { \mathbf Z } ^ { I }$ respectively, each projected into the same shared latent space. Only the BRep branch is trained; the text and image encoders remain frozen throughout. Training is driven by two symmetric InfoNCE contrastive objectives: a BRep-text loss $\mathcal { L } _ { b t }$ and a BRep-image loss $\mathcal { L } _ { b i }$ . For a batch of N matched CAD-text pairs, $\mathcal { L } _ { b t }$ is defined as:

![](images/15893a17768a73cfc7e7bb568465aa0447c4ab6d25b31c22987be2552631db3c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Text Description"] --> B["CLIP"]
    B --> C["Multi-View Images"]
    C --> D["CLIP"]
    D --> E["BRepClip"]
    E --> F["Transformer"]
    F --> G["Face Tokenizer"]
    G --> H["Face Semantics"]
    H --> I["Edge Semantics"]
    I --> J["Edge Tokenizer"]
    J --> K["Edge Ge"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#cfc,stroke:#333
    style H fill:#fcc,stroke:#333
    style I fill:#cfc,stroke:#333
    style J fill:#fcc,stroke:#333
    style K fill:#cfc,stroke:#333
```
</details>

Figure 3: BRepCLIP. Face and edge point sets, $G _ { f }$ and $G _ { e }$ , are tokenized by frozen face $( F _ { T } )$ and edge $( E _ { T } )$ tokenizers and encoded by a transformer with modality, spatial, and semantic cues to produce a global BRep embedding. Frozen CLIP text and image encoders provide caption and multi-view image embeddings for BRep–text and BRep–image contrastive training.

$$
\mathcal {L} _ {b t} = - \frac {1}{2 N} \sum_ {i = 1} ^ {N} \left[ \log \frac {\exp (\mathbf {Z} _ {i} ^ {B} \cdot \mathbf {Z} _ {i} ^ {T} / \tau)}{\sum_ {j = 1} ^ {N} \exp (\mathbf {Z} _ {i} ^ {B} \cdot \mathbf {Z} _ {j} ^ {T} / \tau)} + \log \frac {\exp (\mathbf {Z} _ {i} ^ {T} \cdot \mathbf {Z} _ {i} ^ {B} / \tau)}{\sum_ {j = 1} ^ {N} \exp (\mathbf {Z} _ {i} ^ {T} \cdot \mathbf {Z} _ {j} ^ {B} / \tau)} \right] \tag {3}
$$

where $\tau$ is a learnable temperature parameter and all embeddings are ℓ2-normalized prior to similarity computation. The BRep-image loss $\mathcal { L } _ { b i }$ is defined analogously by substituting ${ \mathbf Z } ^ { T }$ with ${ \bf Z } ^ { I }$ . The total training objective is:

$$
\mathcal {L} = \mathcal {L} _ {b t} + \mathcal {L} _ {b i} \tag {4}
$$

This design keeps the multimodal alignment framework simple and compatible with existing retrieval pipelines, while the BRep encoder learns representations grounded in both language semantics and visual appearance.

# 4 Experiments

Datasets. We primarily use CADCap-1M from DreamCAD [15], specifically its high-quality ABC subset, which provides CAD models paired with captions and multiview renderings from CADCap-1M [15]. From this subset, we use 400K samples for training and 10K for validation. These data are used to train both the primitive tokenizers and the full BRepCLIP model. For each sample, we extract a structured BRep from the STEP file using a PythonOCC pipeline extended from BRepNet [18]. We also sample dense point clouds for point-based baselines and use the DreamCAD multiview renderings for image supervision.

Implementation. Training proceeds in two stages. In the first stage, we train separate dVAEs for faces and edges on 4 NVIDIA A100 GPUs using AdamW with cosine decay and warmup, and annealing schedules for both Gumbel-Softmax temperature and KL-divergence weight. The face dVAE is trained for 100 epochs with a codebook size of 8192, and the edge dVAE for 200 epochs with a codebook size of 2048, both with latent dimension 256. In the second stage, BRepCLIP is trained for 38 epochs on a single NVIDIA A100 GPU using AdamW with learning rate $\mathrm { i 0 ^ { - 3 } }$ and weight decay 0.05. The BRep transformer encoder is a 12-layer transformer with hidden dimension 384 and 6 attention heads, projected to a 512-dimensional shared embedding space. We use frozen OpenCLIP ViT-bigG-14 encoders for text and image, and optimize a weighted sum of BRep–text and BRep–image contrastive losses with equal weights. Training uses mixed precision, gradient checkpointing, and gradient clipping with an effective batch size of 200.

![](images/7958b2db76e03e6888f1731326db47ff2ed6ba38e4b1cc364de25927f8ce59bb.jpg)

Figure 4: Qualitative retrieval results. Given a text query, BRepCLIP retrieves CAD models that faithfully match fine-grained geometric details such as hole count, edge topology, and surface type compared to Point-based baselines.   
Table 1: Zero-shot text-to-CAD retrieval results across different CAD databases. Retrieval performance is reported using Top-k accuracy and Chamfer Distance (CD). Chamfer Distance (CD) is scaled by 103 

<table><tr><td rowspan="2">Method</td><td colspan="5">ABC</td><td colspan="5">CADParser</td><td colspan="5">Automate</td></tr><tr><td>Top-1</td><td>Top-5</td><td>Top-10</td><td>Top-20</td><td>CD ↓</td><td>Top-1</td><td>Top-5</td><td>Top-10</td><td>Top-20</td><td>CD ↓</td><td>Top-1</td><td>Top-5</td><td>Top-10</td><td>Top-20</td><td>CD ↓</td></tr><tr><td>Point-BERT [41]</td><td>2.60</td><td>9.36</td><td>15.80</td><td>22.72</td><td>61.56</td><td>1.10</td><td>4.40</td><td>7.00</td><td>10.90</td><td>45.12</td><td>0.91</td><td>3.55</td><td>6.04</td><td>9.64</td><td>71.58</td></tr><tr><td>PointNet [28]</td><td>3.31</td><td>12.07</td><td>19.38</td><td>29.60</td><td>62.27</td><td>0.40</td><td>1.90</td><td>3.50</td><td>6.40</td><td>64.80</td><td>3.33</td><td>10.72</td><td>16.41</td><td>23.96</td><td>68.13</td></tr><tr><td>PointMLP [25]</td><td>0.90</td><td>3.50</td><td>6.00</td><td>9.50</td><td>68.43</td><td>1.10</td><td>3.70</td><td>6.70</td><td>9.30</td><td>54.55</td><td>1.02</td><td>3.79</td><td>6.39</td><td>10.20</td><td>75.51</td></tr><tr><td>BRepEncoder</td><td>4.30</td><td>16.30</td><td>24.70</td><td>33.90</td><td>61.11</td><td>2.10</td><td>8.20</td><td>13.20</td><td>19.00</td><td>40.32</td><td>4.82</td><td>14.30</td><td>21.06</td><td>29.55</td><td>68.48</td></tr><tr><td>MixCon3D [7]</td><td>1.20</td><td>2.10</td><td>4.20</td><td>8.12</td><td>74.18</td><td>0.19</td><td>1.74</td><td>2.33</td><td>5.12</td><td>69.83</td><td>0.18</td><td>2.33</td><td>3.88</td><td>6.54</td><td>94.15</td></tr><tr><td>ULIP [39]</td><td>2.30</td><td>4.00</td><td>7.40</td><td>12.20</td><td>63.48</td><td>0.70</td><td>2.90</td><td>4.60</td><td>7.10</td><td>67.33</td><td>0.92</td><td>3.11</td><td>5.06</td><td>7.95</td><td>91.41</td></tr><tr><td>OpenShape [21]</td><td>6.12</td><td>18.17</td><td>24.88</td><td>34.36</td><td>71.63</td><td>4.10</td><td>13.40</td><td>19.70</td><td>29.30</td><td>43.33</td><td>7.60</td><td>19.86</td><td>27.58</td><td>36.45</td><td>79.82</td></tr><tr><td>BRepCLIP</td><td>8.59</td><td>24.52</td><td>35.08</td><td>47.89</td><td>58.16</td><td>5.00</td><td>15.08</td><td>22.12</td><td>30.60</td><td>35.28</td><td>9.42</td><td>24.18</td><td>32.86</td><td>42.83</td><td>60.32</td></tr></table>

Experimental Setup. We train on the ABC split from DreamCAD [15], which provides 400K CAD models paired with captions and multiview renderings, with 10K samples held out for validation. For each model, we extract both a structured BRep representation and a point cloud, paired with the corresponding caption. Multiview images are additionally used for multimodal baselines. To support large-scale BRep processing, we extend the BRepNet [18] extraction pipeline to dataset-wide feature extraction. All baselines follow their original training configurations; when adaptation to CAD data is required, we keep the original recipe fixed and only replace the input representation and dataset. We evaluate on three downstream tasks: zero-shot text-to-CAD retrieval , zero-shot CAD classification and generative CAD evaluation.

# 4.1 Text-to-CAD Retrieval Task

In the text-to-CAD retrieval task, the goal is to retrieve the most relevant CAD model from a gallery given a text query, using cosine similarity in the shared embedding space. We consider a zero-shot setting in which all gallery instances are unseen during training.

![](images/940b9b14b5fa6293a318c24668197b423225337a8f3a23bd6dc5965135847240.jpg)

<details>
<summary>text_image</summary>

Zero-Shot Classification
pipe fittings
grommets
hex head screws
boxes
BRepCLIP-Score
Oval plate with two
circular holes and a central
rectangular cutout. Height
is 8.3 times the length.
Circular plate with one
circular holes and a central
rectangular cutout. Height
is 8.3 times the length.
Correct
0.59
0.39
Incorrect
Correct
0.48
Spur gear with central through-
hole and ten equally spaced
teeth; nearly tall profile (width-
to-height ratio 0.3).";
Spur gear with central through-
hole and fifteen equally spaced
teeth; nearly tall profile (width-
to-height ratio 0.3).";
Incorrect
0.45
Incorrect
</details>

Figure 5: Qualitative results for zero-shot classification and BRepCLIP-Score. Left: BRepCLIP supports zero-shot CAD classification via class-level text matching. Right: BRepCLIP-Score assigns higher similarity to prompt-faithful CAD outputs and lower similarity to mismatched ones.

Task Dataset and Protocol. All methods are trained on the same ABC split from CADCap-1M [15], using 400K samples for training and 10K for validation. Retrieval is evaluated on three held-out datasets: a 91K held-out ABC split, CADParser [43], and Automate [10]. The held-out ABC split serves as the in-domain retrieval benchmark, while CADParser and Automate are used for zero-shot retrieval transfer, since neither dataset is seen during training. For all datasets, BRep embeddings are precomputed offline for the gallery models. Concretely, the retrieval benchmarks contain 91K query model pairs for ABC, 40K for CADParser, and 65K for Automate, where each text query is evaluated against the full corresponding CAD gallery.

Baselines. We compare against point-based 3D encoders (PointNet [28], PointMLP [25], Point-BERT [41]), multimodal alignment frameworks (ULIP [39], MixCon3D [7], OpenShape [21]), and our proposed BRep-based models. Our method is evaluated in two forms: BRepEncoder, which uses our BRep-native encoder with text supervision only, and BRepCLIP, which further adds image supervision. To ensure a fair comparison, all baselines are retrained on the same 400K ABC split used for BRepCLIP. We preserve the original training recipes of the respective methods whenever applicable, including encoder architecture, optimizer settings, learning-rate schedule, and contrastive batch size, and only replace the input representation and dataset where necessary.

Metrics. We report Top-k retrieval accuracy for k ∈ {1, 5, 10, 20} together with Chamfer Distance (CD). To measure geometric similarity beyond exact instance matching, we additionally compute CD on a random subset of 10K queries from each dataset. For each query, we retrieve the top-5 candidates, compute the Chamfer Distance between the ground-truth CAD model and each retrieved candidate, average over the top-5 retrieved results, and then average again over all query samples to obtain a single dataset-level CD score.

Results. Table 1 shows that BRepCLIP achieves the strongest retrieval performance across all three datasets. It attains the best Top-k accuracy at every reported value of k and also the lowest Chamfer Distance, indicating that the retrieved CAD models are both more semantically relevant and more geometrically faithful to the ground-truth targets. On ABC, BRepCLIP improves Top-1 accuracy from 6.12 to 8.59, a relative gain of 40.4%, while reducing CD from 0.071 to 0.058. On CADParser, it reaches 5.00 Top-1 and 30.60 Top-20, outperforming OpenShape by 22.0% and 4.4%, respectively, and achieves the best CD of 0.035. On Automate, BRepCLIP achieves 9.42 Top-1 and 42.83 Top-20, corresponding to relative improvements of 23.9% and 17.5% over OpenShape, while also lowering CD from 0.079 to 0.060. Notably, the text-only BRepEncoder already outperforms generic point-based encoders, confirming the importance of native BRep structure for CAD retrieval, and the full BRepCLIP further improves over it through multimodal alignment with both text and image supervision. Qualitative examples in Figure 4 further show that BRepCLIP better captures fine-grained engineering properties such as hole count, edge topology, and surface type, whereas point-based baselines often retrieve only globally similar shapes.

# 4.2 Zero-Shot Classification

Task. Given a BRep embedding, we perform zero-shot CAD classification by matching each model against class-level text descriptors without any fine-tuning.

Setup. We evaluate on FabWave [4], which is not used during training and is treated as a zero-shot transfer benchmark for CAD classification. The original manifest contains

4,421 samples across 45 categories. After filtering out 43 broken or incomplete assets, the final benchmark contains 4,378 valid samples spanning 39 engineering-oriented categories. All models are trained on ABC and transferred di-

rectly to FabWave without further fine-tuning. For evaluation, we define class-level text descriptors for the 39 valid categories and perform zero-shot classification by matching each CAD embedding to these class embeddings.

Baselines and Metrics. We compare against the same baselines as in retrieval and report Top-1, Top-5, and Top-10 accuracy.

Results. Table 2 shows that BRepCLIP achieves the best performance overall, reaching 38.62 Top-1, 70.28 Top-5, and 86.71 Top-10 accuracy. Among 3D-only

Table 2: Zero-shot classification (FabWave) 

<table><tr><td>Method</td><td>Top-1</td><td>Top-5</td><td>Top-10</td></tr><tr><td>Point-BERT [41]</td><td>17.34</td><td>40.21</td><td>56.04</td></tr><tr><td>PointNet [28]</td><td>15.74</td><td>38.78</td><td>54.37</td></tr><tr><td>PointMLP [25]</td><td>18.80</td><td>41.00</td><td>59.02</td></tr><tr><td>BRepEncoder</td><td>21.81</td><td>43.40</td><td>60.74</td></tr><tr><td>ULIP [39]</td><td>21.65</td><td>47.28</td><td>60.62</td></tr><tr><td>MixCon3D [7]</td><td>34.10</td><td>63.93</td><td>78.18</td></tr><tr><td>OpenShape [21]</td><td>33.58</td><td>68.73</td><td>81.73</td></tr><tr><td>BRepCLIP</td><td>38.62</td><td>70.28</td><td>86.71</td></tr></table>

encoders, BRepEncoder performs best, with 21.81 Top-1 accuracy, outperforming all point-based alternatives. These results indicate that CAD-aware BRep encoding yields more transferable semantic representations for zero-shot CAD classification.

# 4.3 BRepCLIP Score for Generative CAD Evaluation

Motivation. Evaluating text-to-CAD generation requires more than visual similarity.

A model that looks correct when rendered may still be missing holes, chamfers, or correct edge topology. CLIP score operates on 2D projections and cannot capture these details. Chamfer Distance measures global shape proximity but is insensitive to local topology. BRepCLIP-Score addresses both limitations by grounding evaluation directly in BRep embeddings, where surface types, edge primitives, and topological structure are explicitly represented.

BRepCLIP-Score. Given a text prompt t and a generated CAD model x, we define

![](images/9562ec82dd9dc853b9c9c60bc4a5d55ea869d0959b027b163c6f866877509154.jpg)  
Figure 6: Score Sensitivity to prompt corruption.

$$
\operatorname{BRepCLIP} - \text { Score } (t, x) = \cos \left(f _ {\text { text }} (t), f _ {3 \mathrm{D}} (x)\right) \tag {5}
$$

where $f _ { \mathrm { t e x t } } ( t )$ is the text embedding of the prompt and f3D(x) is the BRep embedding produced by our encoder. To test sensitivity to semantic mismatch, we sample 10,000 CAD models from CADParser, Automate, and a held-out ABC split, and compare scores under three conditions: the original caption, a mildly corrupted GPT-generated caption, and a fully mismatched caption.

Sensitivity to prompt corruption. Figure 6 shows that BRepCLIP-Score is substantially more sensitive to prompt corruption than image-based similarity metrics.

Under mild corruption, it drops by 17.71%, compared with 2.78% for CLIP score and 4.54% for LongCLIP. Under full mismatch, the drop increases to 104.17%, compared with 25.00% and 18.18%, respectively. This indicates that BRepCLIP-Score better reflects semantic inconsistencies that arise from incorrect geometry, rather than rewarding only visual resemblance.

For benchmarking generative models, we evaluate on 15,000 examples from the ABC dataset using outputs from recent text-to-CAD methods, including DeepCAD [37], Text2CAD [13], CADRille [17] Text2CQ [33], and CADFusion [35]. In addition to automatic metrics, we conduct both human and GPT-based evaluation following the protocol used in DreamCAD. Specifically, for each prompt, evaluators are shown multiview renderings of CAD generations from all competing methods together with the input text, and assign a score from 0 to 10 based on semantic similarity between the generated CAD model and the caption. Human evaluation is performed by five CAD designers, and the final human score is obtained by averaging their ratings. GPT evaluation uses the same multiview renderings and caption, and assigns scores under the same 0–10 semantic-faithfulness criterion. The resulting human and GPT scores are therefore preference-style measures of prompt faithfulness grounded in caption-to-geometry consistency rather than reconstruction accuracy alone. As shown in Table 3, BRepCLIP-Score aligns more closely with both human and GPT judgments than CLIP score, indicating that it provides a more faithful evaluator of text-conditioned CAD generation quality.

Table 3: Benchmarks of text-to-CAD methods. 

<table><tr><td>Method</td><td>CD ↓</td><td>CLIP Score ↑</td><td>Human Score ↑</td><td>GPT Score ↑</td><td>BRepCLIP Score ↑</td></tr><tr><td>Ground Truth</td><td>-</td><td>0.37</td><td>9.7</td><td>9.8</td><td>0.61</td></tr><tr><td>DeepCAD [37]</td><td>86.54</td><td>0.24</td><td>2.2</td><td>2.4</td><td>0.15</td></tr><tr><td>Text2CAD [13]</td><td>86.54</td><td>0.26</td><td>3.6</td><td>3.5</td><td>0.16</td></tr><tr><td>Cadrille [17]</td><td>155.80</td><td>0.26</td><td>3.5</td><td>3.7</td><td>0.16</td></tr><tr><td>Text2CQ (Q3B) [33]</td><td>68.15</td><td>0.33</td><td>5.0</td><td>4.9</td><td>0.31</td></tr><tr><td>Text2CQ (GL) [33]</td><td>71.27</td><td>0.32</td><td>4.6</td><td>4.5</td><td>0.25</td></tr><tr><td>Text2CQ (CG) [33]</td><td>77.91</td><td>0.31</td><td>4.1</td><td>3.9</td><td>0.22</td></tr><tr><td>CADFusion [35]</td><td>56.36</td><td>0.29</td><td>5.5</td><td>5.8</td><td>0.35</td></tr></table>

# 4.4 Ablation Study

BRepCLIP Modality components. We ablate the contribution of each BRep primitive branch by comparing edge-only, face-only, and full BRepCLIP variants. As shown in Table 4, both reduced variants perform substantially worse than the full model. Using only face primitives lowers Top-1 retrieval from 8.59 to 3.40, a drop of 60.4%, while using only edge primitives further reduces it to 1.26, corresponding to an 85.3% drop. Similar trends hold for Top-20, where the face-only and edge-only variants fall by 44.9% and 61.6%, respectively. These results confirm that surface and boundary geometry provide complementary cues, and that jointly encoding both is essential for discriminative CAD retrieval.

Table 4: Ablation of BRepCLIP components on ABC retrieval. 

<table><tr><td>Method</td><td>Top-1</td><td>Top-5</td><td>Top-10</td><td>Top-20</td></tr><tr><td>Edge-only</td><td>1.26</td><td>6.44</td><td>10.24</td><td>18.39</td></tr><tr><td>Face-only</td><td>3.40</td><td>13.12</td><td>19.24</td><td>26.39</td></tr><tr><td>BRepCLIP</td><td>8.59</td><td>24.52</td><td>35.08</td><td>47.89</td></tr></table>

Batch Size. Since BRepCLIP uses cross-modal contrastive learning, larger batches enlarge the in-batch negative pool and improve alignment quality. As shown in Table 5, increasing the batch size from 128 to 200 yields substantial gains of 172.7% and 68.5% in Top-1 and Top-20 accuracy, respectively, whereas further increasing it to 400 brings only marginal improvements of 0.23% and 0.02%. We therefore adopt a batch size of 200, which achieves near-identical performance to 400 while requiring roughly half the GPU memory, about ∼30 GB compared with ∼55 GB, consistent with findings in OpenShape [21].

Table 5: Effect of batch size on BRepCLIP for ABC retrieval. 

<table><tr><td>Batch</td><td>Top-1</td><td>Top-5</td><td>Top-10</td><td>Top-20</td></tr><tr><td>128</td><td>3.15</td><td>10.79</td><td>18.22</td><td>28.42</td></tr><tr><td>200</td><td>8.59</td><td>24.52</td><td>35.08</td><td>47.89</td></tr><tr><td>400</td><td>8.61</td><td>24.53</td><td>35.11</td><td>47.90</td></tr></table>

Multimodal Supervision. As shown in Table 6, BReponly training already provides a strong retrieval baseline. Adding single-view image supervision improves Top-1 and Top-20 by 54.4% and 26.0% respectively, confirming that visual supervision is complementary to native BRep geometry. Replacing single-view with multi-view supervision yields further gains of 29.4%

Table 6: Ablation of multimodal supervision on ABC retrieval. 

<table><tr><td>BRep</td><td>Image</td><td>MultiView</td><td>Top-1</td><td>Top-5</td><td>Top-10</td><td>Top-20</td></tr><tr><td>√</td><td>✗</td><td>✗</td><td>4.30</td><td>16.30</td><td>24.70</td><td>33.90</td></tr><tr><td>√</td><td>√</td><td>✗</td><td>6.64</td><td>20.78</td><td>31.36</td><td>42.73</td></tr><tr><td>√</td><td>√</td><td>√</td><td>8.59</td><td>24.52</td><td>35.08</td><td>47.89</td></tr></table>

on Top-1 and 12.1% on Top-20, indicating that richer visual coverage strengthens alignment between BRep structure and image-text semantics.

# 5 Limitation

BRepCLIP has two main limitations. First, faces and edges are tokenized at a fixed geometric resolution, which may be insufficient for complex CAD models with finer local detail or denser primitive counts, increasing memory and compute at scale. Second, semantic descriptors are limited to a fixed taxonomy of face and edge types, which does not cover the full diversity of primitives and topology encountered in real-world engineering data. Extending both the resolution and the semantic vocabulary are important directions for future work.

# 6 Conclusion

We presented BRepCLIP, the first multimodal contrastive pretraining framework built directly on BRep primitives for CAD understanding. By modeling faces and edges as distinct geometric entities, learning separate discrete token vocabularies for surface and curve geometry, and aligning the resulting BRep representation with text and image embeddings, BRepCLIP captures fine-grained

CAD semantics that are typically lost in point-based representations. Across zero-shot text-to-CAD retrieval and zero-shot CAD classification, BRepCLIP consistently outperforms generic point-based encoders and strong multimodal baselines. We further showed that the learned embedding supports CAD-aware generation evaluation through BRepCLIP-Score, providing a more structure-sensitive alternative to image-based similarity metrics such as CLIP-Score. These results establish native BRep structure as a strong foundation for multimodal CAD representation learning, and open a new direction toward BRep-native foundation models for retrieval, evaluation, and broader engineering design workflows.

# 7 Acknowledgements

This work was co-funded by the European Union under Horizon Europe, grant number 101135724, project LUMINOUS. However, the views and opinions expressed are those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible.

# References

[1] Sk Aziz Ali, Mohammad Sadil Khan, and Didier Stricker. Brep boundary and junction detection for cad reverse engineering. In 2024 IEEE 3rd International Conference on Computing and Machine Intelligence (ICMI), 2024. 3   
[2] A Ananthakrishnan. A multi-modal retrieval augmented framework for user editable 3d cad model generation. 2025. 3   
[3] Armen Avetisyan, Manuel Dahnert, Angela Dai, Manolis Savva, Angel X. Chang, and Matthias Nießner. Scan2CAD: Learning CAD model alignment in RGB-D scans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2614–2623, 2019. 3   
[4] Akshay Bharadwaj, Yang Xu, Atin Angrish, Yong Chen, and Binil Starly. Development of a pilot manufacturing cyberinfrastructure with an information rich mechanical cad 3d model repository. In International Manufacturing Science and Engineering Conference, 2019. 2, 7   
[5] Polly Ann Brown. Cad: Do computers aid the design process after all? Intersect: The Stanford Journal of Science, Technology and Society, 2:52–66, 2009. 1   
[6] Jiali Chen, Xusen Hei, Hongfei Liu, Yuancheng Wei, Zikun Deng, Jiayuan Xie, Yi Cai, and Li Qing. Cadreview: Automatically reviewing cad programs with error detection and correction. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9909–9927, 2025. 3   
[7] Yipeng Gao, Zeyu Wang, Wei-Shi Zheng, Cihang Xie, and Yuyin Zhou. Sculpting holistic 3d representation in contrastive language-image-3d pre-training. In CVPR, 2024. 6, 7, 8   
[8] Negar Heidari and Alexandros Iosifidis. Geometric deep learning for computer-aided design: A survey. IEEE Access, 13:119305–119334, 2024. 1, 3   
[9] Pradeep Kumar Jayaraman, Aditya Sanghi, Joseph G Lambourne, Karl DD Willis, Thomas Davies, Hooman Shayani, and Nigel Morris. Uv-net: Learning from boundary representations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11703–11712, 2021. 3   
[10] Benjamin Jones, Dalton Hildreth, Duowen Chen, Ilya Baran, Vladimir G Kim, and Adriana Schulz. Automate: A dataset and learning approach for automatic mating of cad assemblies. ACM Transactions on Graphics (TOG), 2021. 7   
[11] Benjamin T. Jones, Michael Hu, Vladimir G. Kim, and Adriana Schulz. Self-supervised representation learning for CAD. arXiv preprint arXiv:2210.10807, 2022. 3   
[12] David Kasik, William Buxton, and David Ferguson. Ten cad challenges. IEEE computer graphics and applications, 25:81–92, 03 2005. 1

[13] Mohammad S Khan, Sankalp Sinha, Talha U Sheikh, Didier Stricker, Sk A Ali, and Muhammad Z Afzal. Text2cad: Generating sequential cad designs from beginner-to-expert level text prompts. Advances in Neural Information Processing Systems, 37:7552–7579, 2024. 3, 8   
[14] Mohammad Sadil Khan, Elona Dupont, Sk Aziz Ali, Kseniya Cherenkova, Anis Kacem, and Djamila Aouada. Cad-signet: Cad language inference from point clouds using layer-wise sketch instance guided attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4713–4722, June 2024. 3   
[15] Mohammad Sadil Khan, Muhammad Usama, Rolandos Alexandros Potamias, Didier Stricker, Muhammad Zeshan Afzal, Jiankang Deng, and Ismail Elezi. Dreamcad: Scaling multi-modal cad generation using differentiable parametric surfaces. Arxiv, 2026. 3, 5, 6, 7   
[16] Mingi Kim, Yongjun Kim, Jungwoo Kang, and Hyungki Kim. Brepcoder: A unified multimodal large language model for multi-task b-rep reasoning. arXiv preprint arXiv:2602.22284, 2026. 3   
[17] Maksim Kolodiazhnyi, Denis Tarasov, Dmitrii Zhemchuzhnikov, Alexander Nikulin, Ilya Zisman, Anna Vorontsova, Anton Konushin, Vladislav Kurenkov, and Danila Rukhovich. cadrille: Multi-modal cad reconstruction with reinforcement learning. In The Fourteenth International Conference on Learning Representations, 2025. 8   
[18] Joseph G Lambourne, Karl DD Willis, Pradeep Kumar Jayaraman, Aditya Sanghi, Peter Meltzer, and Hooman Shayani. Brepnet: A topological message passing system for solid models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12773–12782, 2021. 1, 3, 5, 6   
[19] Florian Langer, Jihong Ju, Georgi Dikov, Gerhard Reitmayr, and Mohsen Ghafoorian. FastCAD: Real-time CAD retrieval and alignment from scans and videos. In Proceedings of the European Conference on Computer Vision (ECCV), 2024. 3   
[20] Jiahao Li, Weijian Ma, Xueyang Li, Yunzhong Lou, Guichun Zhou, and Xiangdong Zhou. Cad-llama: leveraging large language models for computer-aided design parametric 3d model generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18563–18573, 2025. 3   
[21] Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xuanlin Li, Shizhong Han, Hong Cai, Fatih Porikli, and Hao Su. Openshape: Scaling up 3d shape representation towards open-world understanding. Advances in neural information processing systems, 36:44860–44879, 2023. 2, 3, 6, 7, 8, 9   
[22] Yujia Liu, Anton Obukhov, Jan Dirk Wegner, and Konrad Schindler. Point2cad: Reverse engineering cad models from 3d point clouds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3763–3772, 2024. 3   
[23] Yunzhong Lou, Xueyang Li, Haotian Chen, and Xiangdong Zhou. Brep-bert: Pre-training boundary representation bert with sub-graph node contrastive learning. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, pages 1657–1666, 2023. 3   
[24] Weijian Ma, Minyang Xu, Xueyang Li, and Xiangdong Zhou. Multicad: Contrastive representation learning for multi-modal 3D computer-aided design models. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM). ACM, 2023. 3   
[25] Xu Ma, Can Qin, Haoxuan You, Haoxi Ran, and Yun Fu. Rethinking network design and local geometry in point cloud: A simple residual mlp framework. In International Conference on Learning Representations, 2022. 6, 7, 8   
[26] Dimitrios Mallis, Ali Sk Aziz, Elona Dupont, Kseniya Cherenkova, Ahmet Serdar Karadeniz, Mohammad Sadil Khan, Anis Kacem, Gleb Gusev, and Djamila Aouada. Sharp challenge 2023: Solving cad history and parameters recovery from point clouds and 3d scans. overview, datasets, metrics, and baselines. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1786–1795, 2023. 3

[27] Tessa Pulli, Jean-Baptiste Weibel, Peter Hönig, Matthias Hirschmanner, Markus Vincze, and Andreas Holzinger. Oscar: Open-set cad retrieval from a language prompt and a single image. arXiv preprint arXiv:2601.07333, 2026. 3   
[28] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 652–660, 2017. 2, 6, 7, 8   
[29] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems, 30, 2017. 2   
[30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021. 2, 4   
[31] C. Schinko, T. Vosgien, T. Prante, T. Schreck, and T. Ullrich. Search and retrieval in cad databases - a user-centric state-of-the-art overview. In Proceedings of the 12th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, 2017. 12th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications : VISAPP 2017, VISIGRAPP ; Conference date: 27-02-2017 Through 01-03-2017. 3   
[32] Sankalp Sinha, Mohammad Sadil Khan, Muhammad Usama, Shino Sam, Didier Stricker, Sk Aziz Ali, and Muhammad Zeshan Afzal. Marvel-40m+: Multi-level visual elaboration for high-fidelity text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8105–8116, 2025. 3   
[33] Yuhao Sun, Hao Cheng, Shang Zheng, Hualong Yu, and Haitao Zou. Balancing speed and executability in interactive text-to-cad code generation for early-stage parametric cad ideation. Journal of King Saud University Computer and Information Sciences, 2026. 8   
[34] Muhammad Usama, Mohammad Sadil Khan, Didier Stricker, and Muhammad Zeshan Afzal. Nurbgen: High-fidelity text-to-cad generation through llm-driven nurbs modeling. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 9603–9611, 2026. 3   
[35] Ruiyu Wang, Yu Yuan, Shizhao Sun, and Jiang Bian. Text-to-cad generation through infusing visual feedback in large language models. arXiv preprint arXiv:2501.19054, 2025. 8   
[36] Siyu Wang, Cailian Chen, Xinyi Le, Qimin Xu, Lei Xu, Yanzhou Zhang, and Jie Yang. Cad-gpt: Synthesising cad construction sequence with spatial reasoning-enhanced multimodal llms. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 7880–7888, 2025. 3   
[37] Rundi Wu, Chang Xiao, and Changxi Zheng. Deepcad: A deep generative network for computeraided design models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6772–6782, 2021. 3, 8   
[38] Jingwei Xu, Chenyu Wang, Zibo Zhao, Wen Liu, Yi Ma, and Shenghua Gao. Cad-mllm: Unifying multimodality-conditioned cad generation with mllm. arXiv preprint arXiv:2411.04954, 2024. 3   
[39] Le Xue, Mingfei Gao, Chen Xing, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1179–1189, 2023. 2, 6, 7, 8   
[40] Le Xue, Ning Yu, Shu Zhang, Artemis Panagopoulou, Junnan Li, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, et al. Ulip-2: Towards scalable multimodal pre-training for 3d understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27091–27101, 2024. 2, 3

[41] Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, and Jiwen Lu. Point-bert: Pre-training 3d point cloud transformers with masked point modeling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19313–19322, 2022. 2, 4, 6, 7, 8   
[42] Shuming Zhang, Zhidong Guan, Hao Jiang, Tao Ning, Xiaodong Wang, and Pingan Tan. Brep2seq: a dataset and hierarchical deep learning network for reconstruction and generation of computer-aided design models. Journal of Computational Design and Engineering, 11(1):110– 134, 2024. 3   
[43] Shengdi Zhou, Tianyi Tang, and Bin Zhou. Cadparser: a learning approach of sequence modeling for b-rep cad. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, 2023. 7   
[44] Qiang Zou and Lizhen Zhu. Bringing attention to cad: Boundary representation learning via transformer. Computer-Aided Design, 189:103940, December 2025. 3

# Supplementary Material

# A Dataset Analysis

In this section, we provide additional analysis of the datasets used for training and evaluation. Our training data is built from the high-quality ABC subset of CADCap-1M, from which we use 400K CAD models for training and 10K for validation. Retrieval is evaluated on a held-out ABC split with 91K samples, while zero-shot retrieval is evaluated on two unseen CAD datasets: Automate and CADParser. For zero-shot classification, we use FabWave. The original FabWave manifest contains 45 categories, but after filtering 43 broken or incomplete assets, the final benchmark contains 4,378 valid CAD models across 39 categories.

# A.1 Training Data Statistics

![](images/86026ba40e0530b27b1b35de5910115db7814815a79e412106a044297180b438.jpg)

<details>
<summary>bar_line</summary>

| Number of edges | CAD models per bin (%) |
| --------------- | ---------------------- |
| 0               | 3.0                    |
| 20              | 6.0                    |
| 40              | 11.0                   |
| 60              | 8.0                    |
| 80              | 6.0                    |
| 100             | 4.0                    |
| 120             | 2.0                    |
| 140             | 1.5                    |
| 160             | 1.0                    |
| 180             | 0.8                    |
| 200             | 0.6                    |
| 220             | 0.5                    |
| 240             | 0.4                    |
| 260             | 0.3                    |
| 280             | 0.2                    |
| 300             | 0.1                    |
| 320             | 0.1                    |
| 340             | 0.1                    |
| 360             | 0.1                    |
| 380             | 0.1                    |
| 400             | 0.1                    |
| 420             | 0.1                    |
| 440             | 0.1                    |
| 460             | 0.1                    |
| 480             | 0.1                    |
| 500             | 0.1                    |
| 520             | 0.1                    |
| 540             | 0.1                    |
| 560             | 0.1                    |
| 580             | 0.1                    |
| 600             | 0.1                    |
</details>

![](images/62aff2e1422c9697a4ef89d0ada03e9615818170484e0c67437782b6c7bb7e0f.jpg)

<details>
<summary>bar_line</summary>

| Number of faces | CAD models per bin (%) |
| --------------- | ---------------------- |
| 0               | 4%                     |
| 25              | 8%                     |
| 50              | 4%                     |
| 75              | 2%                     |
| 100             | 1%                     |
| 125             | 0.5%                   |
| 150             | 0.3%                   |
| 175             | 0.2%                   |
| 200             | 0.1%                   |
| 225             | 0.05%                  |
| 250             | 0%                     |
</details>

Figure 7: Distributions of the number of edges per CAD model (left), the number of faces per CAD model (middle), and the average number of edges per face (right) in the 400K ABC training set.

We first analyze the geometric complexity of the 400K ABC training set used for BRepCLIP pretraining. Figure 7 summarizes three complementary statistics: the number of edges per CAD model, the number of faces per CAD model, and the average number of edges per face. All three provide a compact view of the structural diversity present in the training set.

The face and edge distributions are both strongly right-skewed, showing that most CAD models contain a relatively small to moderate number of primitives, while a smaller subset contains substantially more complex geometry. For faces, the mean number per CAD model is 47.8, the median is 27.0, and the 95th percentile is 165.0. This indicates that the training set contains many simple and medium-complexity mechanical parts, but also a substantial long tail of models with rich surface decomposition. For edges, the distribution is broader and heavier-tailed, with a mean of 115.9, a median of 69.0, and a 95th percentile of 408.0. This is expected, since edges capture local boundaries, transitions, and fine geometric details more densely than faces. The much heavier tail in the edge distribution confirms that many CAD models contain rich boundary structure, which motivates treating edges as first-class primitives rather than relying only on surface-level information.

To further characterize local BRep structure, Figure 7 also plots the distribution of the average number of edges per face. This distribution is concentrated around 2.5, with a mean of 2.5, a median of 2.5, and a 95th percentile of 3.0. This indicates that most faces in the training set are bounded by a small number of edges, reflecting the predominance of regular engineering surfaces such as planar, cylindrical, and smoothly connected analytic patches. At the same time, the spread toward higher values suggests the presence of more irregular or highly segmented face boundaries in complex models.

Taken together, these statistics show that the ABC training split spans a broad range of CAD complexity, from simple low-face parts to highly structured objects with many faces and edges. This diversity is important for training BRepCLIP, since it exposes the model to both regular mechanical primitives and harder long-tail geometries.

# A.2 Evaluation Split Overview

Figure 8 summarizes the data usage across training and evaluation. The main training set consists of 400K ABC models. For in-domain retrieval testing, we use a held-out ABC split of 91K CAD models. For zero-shot retrieval transfer, we evaluate on two unseen datasets: Automate with 65K models and CADParser with 40K models. This setup clearly separates indomain retrieval from zero-shot transfer evaluation, allowing us to test both memorization-free retrieval within the same CAD source and generalization to different CAD repositories.   
![](images/3d2ff702667b3bde6ce5e1d2bf3cd83e584ce944a95b21798a1d84b1742e230a.jpg)

<details>
<summary>bar</summary>

| Category       | Number of CAD models |
| -------------- | -------------------- |
| ABC Train      | 400K                 |
| ABC Held-out   | 91K                  |
| Automate       | 65K                  |
| CADParser      | 40K                  |
</details>

Figure 8: Overview of training, in-domain retrieval, and zero-shot retrieval splits used in our experiments.

For zero-shot classification, we use FabWave after filtering invalid assets. The final benchmark contains 4,378 valid samples across 39 categories and is never used during training. This makes FabWave a strict zero-shot transfer benchmark for category-level CAD recognition.

# A.3 Primitive Type Statistics

We further analyze the distribution of BRep primitive types in the 400K ABC training set. Our extraction pipeline assigns semantic labels to both faces and edges. For faces, we extract Plane, Cylinder, Cone, Sphere, Torus, and Rational NURBS. For edges, we extract Line, Circle, Ellipse, Non-rational B-spline, and Rational B-spline. We also extract edge relation attributes, including Convex, Concave, Smooth, and Closed.

![](images/01d6c58759b03abadc9bdac7fabbb7de4e2208b21e4ddf9c9ec8445b5832cf67.jpg)

<details>
<summary>bar</summary>

| Face type | Percentage of all face primitives (%) |
| :--- | :--- |
| Plane | 61.3 |
| Cylinder | 28.9 |
| Torus | 3.4 |
| Cone | 3.1 |
| Rational NURBS | 2.3 |
| Sphere | 1.0 |
</details>

![](images/ebabf55bd5589200ac1dee1477c7576c927fd5ef5fff6cb3fd38a8dc2384d215.jpg)

<details>
<summary>bar</summary>

| Edge curve type | Percentage (%) |
|---|---|
| Line | 58.4 |
| Circle | 22.2 |
| Non-rational B-spline | 17.4 |
| Ellipse | 1.5 |
| Rational B-spline | 0.5 |
</details>

Figure 9: Distribution of face primitive types (left) and edge curve types (right) in the 400K ABC training set.

Figure 9 shows that the dataset is dominated by analytic CAD geometry. For faces, planes account for 61.3% of all face primitives, followed by cylinders at 28.9%. Torus, cone, rational NURBS, and sphere faces are much less frequent. For edges, lines are the most common primitive at 58.4%, followed by circles at 22.2% and non-rational B-splines at 17.4%, while ellipses and rational B-splines are relatively rare. Overall, this confirms that most CAD models in ABC are composed of planar and cylindrical surfaces bounded by straight and circular edges, with a smaller long tail of more complex free-form geometry.

![](images/0911f178a339f0bbf9771187003d7b8ee41f5855d012343593b7d1136c1f7451.jpg)

<details>
<summary>bar</summary>

| Edge Relation | Percentage of all edge relation attributes (%) |
| :--- | :--- |
| Concave | 18.0 |
| Convex | 53.1 |
| Smooth | 23.1 |
| Closed | 5.8 |
</details>

Figure 10: Distribution of edge relation attributes in the 400K ABC training set.

Figure 10 shows the distribution of edge relation attributes. Convex edges are the most common at 53.1%, followed by smooth edges at 23.1% and concave edges at 18.0%. Closed edges account for the remaining 5.8%. This indicates that most

![](images/8161ee7fb44adaa1e7b72863584b0a414fb82aea5155e04221dcdaca3a7a18ee.jpg)

<details>
<summary>text_image</summary>

PointBert
PointNet
PointMLP
Mixcon3D
ULIP
(Point-BERT)
OpenShape
(Point-BERT)
BRepClip
(Ours)
Cylindrical connector with a knurled flange and hollow interior. Width is slightly greater than height, maintaining a balanced proportion.
</details>

![](images/af59a80a713e557dbcfa2150408ade3eb060af21f852515364e577d512217eee.jpg)

<details>
<summary>natural_image</summary>

Collection of 3D-printed mechanical parts and hexagonal shapes, no text or symbols visible
</details>

Flat octagonal plate, very wide (length-to-height 9.9), uniform thickness, eight equal edges with slight chamfers, no holes or additional features.   
![](images/e081d2666312a681ae9e65daecca9ece036afd130d699a735d640efde1b04795.jpg)

<details>
<summary>natural_image</summary>

Collection of various colored plastic and metal components (no text or symbols visible)
</details>

Rectangular bracket with rounded ends, three circular holes; two through top, one lateral. Height-to-length ratio approximately 1.5.

![](images/afb1f38f3dac5e97ab7753de97903d1c20f8f8e27a8c818c948d96c7aeeff1b7.jpg)

<details>
<summary>natural_image</summary>

Collection of various plastic mechanical parts and fittings, including T-shaped connectors, gears, and brackets (no text or symbols visible)
</details>

Flat symmetric connector plate with central circular through-hole and two rounded side tabs; parallel faces, perpendicular hole axis.   
![](images/4c62e6097204bf61d66ea2f409a05256584c03990a39a723708bfcec3d629909.jpg)

<details>
<summary>natural_image</summary>

Collection of various plastic mechanical parts and accessories, including washers, flanges, and washers (no text or symbols visible)
</details>

Round base fixture with one recessed pocket, two wedge ramps, and a central small cylindrical post on top.   
Figure 11: Additional qualitative text-to-CAD retrieval results. Given a text query, BRepCLIP retrieves CAD models that better preserve fine-grained geometric details such as hole layout, boundary structure, and surface composition than point-based baselines.

CAD parts are dominated by regular outward boundaries and smooth transitions, while concave and closed structures occur less frequently but remain important for engineering geometry.

Taken together, these primitive statistics support our use of primitive-aware tokenization and semantic descriptors, since face and edge types provide useful structural cues beyond raw point samples alone.

More Qualitative results. In Figure 11, 12, 13 we provided more qualitative samples on retrieval task, BRepCLIP-Score and zero-shot classification.

# BRepCLIP-Score

![](images/4e87fb41326bde7ac433b048ef0203a8b243b1b5fb627fa291f5c76b30064807.jpg)

<details>
<summary>text_image</summary>

Wide rectangular housing
with shallow recessed cavity,
central internal boss, and six
peripheral mounting tabs with
circular holes. Height-to-length
ratio 0.1; width-to-length 0.9.

Correct
0.244

Wide cylindrical housing with
shallow recessed cavity, central
internal boss, and six peripheral
mounting tabs with circular holes.
Height-to-length ratio 0.1; width-
-to-length 0.9.

Incorrect
0.046
</details>

![](images/af676b998d909744acb45aaf6bdde4810094cebca8f26e40c4177c3c05dfdc9f.jpg)

<details>
<summary>text_image</summary>

Forked U-shaped
bracket with two
parallel rounded
arms, curved base,
and two circular
mounting holes
on the base.
Correct
0.31
0.25
Incorrect
Forked U-shaped
bracket with two
parallel sharp-edged
arms, curved base,
and three circular
mounting holes on
the base.
</details>

![](images/d8ed19dc6b5a3ff6d7d9997433fef66f1c11db0ebdaa629fc0cfcec5b3adc3e3.jpg)

<details>
<summary>bar</summary>

| Plate Type | Value |
| :--- | :--- |
| Correct | 0.45 |
| Incorrect | 0.23 |
</details>

![](images/065c8dd7889d2165098d5619ef9fca1133bf38fa71e9406988046da375c155de.jpg)

<details>
<summary>other</summary>

| Bracket Type | Correct | Incorrect |
| ------------ | ------- | --------- |
| Rectangular Base | 0.244 | - |
| Arched Rib, and Single Cylindrical Boss Centered Above the Base. Width-to-length 0.8, height-to-length 0.5. | - | 0.046 |
</details>

![](images/58cac36deb0ec7ed593306eff1a0740d2a34373bcb642aeb44831f7dcac0bb42.jpg)

<details>
<summary>text_image</summary>

Universal tool mount:
wide cylindrical plate
with rectangular side
tab, twenty circular holes
including central bore
and patterned perimeter
holes.

Correct
0.54

Dual rectangular U-
brackets with rounded
inner cutouts and
seven circular holes
total; parallel
sidewalls, filleted
edges, and symmetric
hole placement.

Incorrect
-0.11
</details>

![](images/84c8a22495bed15a4f12ec9d87c353aaee594c9e6ef67e2e04b128c5896bd63f.jpg)

<details>
<summary>text_image</summary>

Mounting bar, long rectangular with rounded ends, nine evenly spaced through-holes. Length is 10× width; height is half the width.
Correct
0.51
-0.101
Mounting bracket with cylindrical boss and base; three circular holes total. Height is 1.2 times the length.
Incorrect
</details>

Figure 12: Additional qualitative results for BRepCLIP-Score. Higher scores are assigned to CAD models that better match the input text in geometry and structure, while semantically inconsistent generations receive lower scores.

![](images/9834341b51461477e4fe91c6849f44ceaaa06cbe8a1d753339fb32fb0cbd1273.jpg)  
Figure 13: Additional qualitative results for zero-shot classification on FabWave. BRepCLIP produces more semantically accurate class predictions for engineering CAD models than point-based and multimodal baselines.