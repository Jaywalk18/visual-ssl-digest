# Intra-Modal Neighbors Never Lie: Rectifying Inter-Modal Noisy Correspondence via Graph-Based Intra-Modal Reasoning

Yang Liu 1 2 Wentao Feng 1 Shu-Dong Huang 1 Yalan Ye 3 Jiancheng Lv 1

# Abstract

Large-scale web-harvested datasets have fueled the progress of cross-modal retrieval but inevitably suffer from noisy correspondence, which severely degrades model generalization. Existing methods primarily address this by filtering out noise or seeking a substitute label, yet they predominantly remain bound by a “Discrete Selection” paradigm. We argue that relying on a single discrete proxy induces Single-Point Fragility and Discretization Error. To overcome these limitations, we propose a novel framework, Intra-modal Neighbor-aware Noise Rectification (IN2R), which shifts the paradigm from searching for a substitute to synthesizing a reliable supervision target. Leveraging the intrinsic geometric stability of intra-modal data, IN2R employs a Graph Refiner to perform relational reasoning over neighbors retrieved from a dynamic Cross-Model Memory. Instead of propagating discrete labels, our method synthesizes a continuous, soft prototype that reflects the consensus of“A gold the local semantic neighborhood, effectively rectifying inter-modal misalignment. Extensive exper-“A beagle iments on Flickr30K, MS-COCO, and CC152K demonstrate that IN2R significantly outperforms state-of-the-art methods. Our code and pre-green law trained models are publicly available at https: //github.com/liuyyy111/IN2R.

# 1. Introduction

Effective visual-semantic alignment has emerged as a cornerstone of diverse vision-language applications, encompass-

![](images/536a275d866695691d2254da7d506015ce163177fdb102852f78a5b56113804a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Image\nNoisy Caption\n“A sleeping cat”"] --> B["Traditional Discrete Selection\nClean Set Search:\nFind Nearest Neighbor"]
    B --> C["Nearest Neighbor\nProxy Caption\n“A golden retrieval\nin a living room”"]
```
</details>

![](images/413886b9246767abf6af4a690d035e6bcc085e25aa0e81963bb82b450f8dce80.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Image\nNoisy Caption\n“A sleeping cat”"] --> B["Proposed Continuous Rectification"]
    B --> C["Intra-Modal Retrieval & Graph-guided Rectification"]
    C --> D["Graph Refiner"]
    D --> E["Synthesized Continuous Prototype"]
    
    subgraph Inputs
        F1["A gold retriever..."]
        F2["A beagle..."]
        F3["A sprawling green lawn..."]
    end
    
    subgraph Outputs
        D
        E
    end
```
</details>

Figure 1. Comparison between the Traditional Discrete Selection paradigm and our proposed Continuous Rectification (IN2R). While discrete selection (top) seeks a single substitute proxy from a finite dataset, often suffering from discretization error or selecting noisy neighbors (e.g., retrieving an imperfect caption), our approach (bottom) leverages the intrinsic topological structure. By retrieving intra-modal neighbors and aggregating them via a Graph Refiner, we synthesize a robust, continuous prototype that rectifies the semantic misalignment (e.g., correcting “A sleeping cat” using the visual consensus of dog-related features).

ing cross-modal retrieval, visual question answering, and broader multi-modal reasoning. The prevailing paradigm relies on contrastive learning to align visual and textual representations into a shared semantic space, typically requiring large-scale, high-quality image-text pairs. However, realworld datasets derived from web sources inevitably suffer from noisy correspondence, leading to images being paired with mismatched or irrelevant captions. Recent studies have revealed that even widely used benchmarks like Conceptual Captions (Huang et al., 2021) contain a non-negligible ratio of mismatched pairs. Such noise fundamentally corrupts the training signal, causing the model to memorize erroneous associations and severely degrading retrieval performance.

To combat this, existing approaches primarily diverge into three paradigms. Sample Selection methods (Huang et al., 2021; Qin et al., 2022) and Consistency-based approaches (Yang et al., 2023; Zha et al., 2025; Zhao et al., 2024) often falter due to intrinsic data waste (i.e., discarding informative hard positives) or a circular dependency on corrupted inter-modal signals for re-weighting. Consequently, recent research has shifted towards Correction-based strategies (Han et al., 2023; Li et al., 2024; Han et al., 2024), which attempt to rectify noisy labels by retrieving new targets. However, we argue these methods remain bound by a “Discrete Selection” paradigm, seeking a substitute proxy from the existing finite dataset. This reliance on discrete proxies fundamentally limits precision: it induces Single-Point Fragility when the selected neighbor is noisy, and inevitably introduces Discretization Error by forcing continuous semantic truths to align with imperfect, discrete samples.

To overcome the fragility of discrete selection, we exploit the intrinsic topological structure of the data. A key observation, illustrated in Figure 1, motivates our design: while noise disrupts the explicit alignment between modalities, the implicit geometric structure within each modality remains robust. For instance, as shown in the upper path of Figure 1, an image of a golden retriever might be wrongly paired with a noisy caption ”a sleeping cat”. Traditional methods risk selecting another discrete proxy that may itself be imperfect. However, the image still maintains correct semantic proximity to other dog images (e.g., “beagle”) in the visual feature space. This implies that reliable semantic information is not attached to any single instance—which might itself be noisy—but is effectively preserved within the collective consensus of the local neighborhood. Crucially, we posit that the semantic “truth” of a noisy sample is not a discrete point waiting to be found in the dataset, but is best modeled as a continuous prototype synthesized from this local consensus. This perspective drives a paradigm shift from searching for a substitute to synthesizing a target, offering two decisive advantages. First, regarding robustness, it mitigates the risk of selecting a noisy neighbor; while individual samples may be unreliable, their collective distribution statistically marginalizes such noise. Second, regarding precision, it transcends the limitations of discrete selection. Unlike reusing a neighbor’s existing label—which inevitably introduces discretization error and merely recycles old data—our method synthesizes a new, continuous supervision signal. This synthesized target fills the semantic gaps between discrete samples, providing finergrained supervision that more accurately approximates the true semantic center of the manifold.

To instantiate this continuous rectification paradigm, we propose a novel framework termed Intra-modal Neighboraware Noise Rectification (IN2R). Recognizing that reliable “synthesis” requires a pristine source of information, we build our framework upon a co-training backbone, employing dual peer networks that maintain Cross-Model Memory Queues to dynamically curate high-confidence clean samples. This cross-model design is pivotal: it decouples the source of retrieval from the model being trained, ensuring that the topological reference remains unbiased. The core innovation lies in our Graph-Guided Rectification mechanism. Rather than treating the retrieved neighbors as a bag of discrete candidates, we model them as a local semantic graph. Specifically, we employ a learnable Graph Refiner that performs relational reasoning over the retrieved neighbors. Through attention-based aggregation, the graph refiner captures the subtle geometric dependencies within the neighborhood and synthesizes a refined prototype. This prototype serves as a robust, continuous supervision target that is both topologically faithful (derived from the clean manifold) and statistically precise (marginalizing discrete noise), effectively transforming the supervision of noisy data from “imitation of a proxy” to “alignment with a synthesized truth.”

The main contributions are summarized as follows:

• Paradigm Shift: We identify the “Single-Point Fragility” in discrete selection methods and propose a shift towards continuous rectification. Our approach leverages intramodal topology to synthesize robust supervision targets, avoiding discretization errors.   
• Methodological Innovation: We propose the IN2R framework, which integrates a Cross-Model Memory with a Graph Refiner. This design effectively decouples noise sources and employs relational reasoning to generate finegrained supervision for noisy samples.   
• State-of-the-Art Performance: Extensive experiments on three benchmarks demonstrate that IN2R significantly outperforms existing methods, particularly in high-noise scenarios, validating the efficacy of our strategy.

# 2. Related Works

Image-Text Matching. Classical image–text retrieval approaches can be broadly categorized into two groups based on the granularity of the matching similarity: global-level matching and local-level matching methods. Global-level matching methods project images and texts into a shared embedding space, where similarity is computed using improved loss functions (Faghri et al., 2018; Chun et al., 2021) to bring semantically aligned pairs closer together. To enhance the quality of the embedding space, recent studies have introduced more sophisticated network architectures, such as graph convolutional networks (Liu et al., 2020; Wang et al., 2020), generalized pooling operators (Chen et al., 2021), and other advanced model designs (Huang et al., 2018; Li et al., 2019; 2022). To achieve closer alignment between semantically corresponding image–text pairs, several methods (Chun et al., 2021; Kim et al., 2023; Chun, 2023; Liu et al., 2025b;a) have been proposed to address the inherent discrepancy in information density between visual and textual modalities. In contrast, local-level matching methods (Diao et al., 2021; Liu et al., 2023; Wang et al.,

2019; Wei et al., 2020; Zhang et al., 2022; 2020; Chen et al., 2020; Qu et al., 2021) focus on fine-grained region-level alignment. These approaches typically capture detailed correspondences between image regions and textual phrases through cross-modal interaction networks.

Learning with Noisy Correspondence (NCL). Existing NCL methods primarily diverge into three streams: selection, consistency, and correction. Sample Selection methods (Huang et al., 2021; Qin et al., 2022) partition data into clean and noisy subsets based on loss distributions. While effective for low noise, these methods inevitably suffer from data waste by discarding ”hard positives” that exhibit high losses. To mitigate this, Consistency-based methods (Yang et al., 2023; Zha et al., 2025; Zhao et al., 2024) re-weight samples by enforcing cross-modal or intramodal consistency. However, they face a circular dependency: consistency computed from corrupted inter-modal signals is inherently unreliable under high noise ratios. Recently, Correction-based approaches have emerged, attempting to rectify noisy labels by learning meta-similarity (Han et al., 2023), mining consistency cues across views (Ma et al., 2024), suppressing soft-margin contributions of suspicious pairs (Yang et al., 2024), retrieving nearest neighbors (Li et al., 2024), propagating pseudo-label consistency across peers (Liu et al., 2026), or seeking optimal transport plans (Han et al., 2024). Crucially, these methods predominantly follow a “Discrete Selection” paradigm—seeking a substitute proxy from the finite dataset. We argue this induces discretization error, as the true semantic target often lies on a continuous manifold and may not exist in the discrete candidate pool. Unlike these works, our method shifts from finding a discrete proxy to synthesizing a continuous prototype.

Graph-based Reasoning in NCL. Graph Neural Networks (GNNs) have recently been introduced to Noisy Correspondence Learning (NCL) to capture high-order structural information. Recent works like GLP (Li et al., 2025) and SPS (Xie et al., 2025) construct neighbor graphs to refine representations. However, these approaches typically treat the graph as a tool for passive label propagation or smoothing, averaging out discriminative signals alongside noise. In contrast, our framework employs GNNs for active reasoning, utilizing attention mechanisms to dynamically detect structural conflicts and synthesize robust supervision signals from the local consensus, rather than merely propagating existing labels.

# 3. Method

# 3.1. Overview and Problem Formulation

We consider the cross-modal retrieval task given a dataset $\mathcal { D } = \{ ( I _ { i } , T _ { i } ) \} _ { i = 1 } ^ { N }$ 1, which inevitably contains noisy correspondence. Our goal is to learn robust encoders $f _ { \theta } ( \cdot )$ and $g _ { \phi } ( \cdot )$ by transforming noisy labels into continuous, highfidelity supervision signals.

We propose the Intra-modal Neighbor-aware Noise Rectification $\mathbf { ( I N ^ { 2 } R ) }$ framework. As illustrated in Figure 2, our method adopts a co-training paradigm with two peer networks, $\mathcal { M } _ { A }$ and $\mathcal { M } _ { B } ,$ , to decouple the noise identification from the rectification process.

To initiate the robust training, we strictly leverage the “Small-Loss” hypothesis. Specifically, the training proceeds in two phases:

• Warm-up Phase: We first warm up the networks on the full dataset D. Following L2RM (Han et al., 2024), we adopt the Symmetric Cross Entropy (SCE) as the robust objective to prevent overfitting. This strategy helps establish a preliminary discriminative feature space, effectively separating the loss distributions of clean and noisy samples.

To formalize this, let H(·, ·) denote the cross-entropy operator. We define the general SCE loss between a target distribution q and a prediction distribution p as:

$$
\mathcal {L} _ {s c e} (\mathbf {q}, \mathbf {p}) = \alpha \mathbb {H} (\mathbf {q}, \mathbf {p}) + \beta \mathbb {H} (\mathbf {p}, \mathbf {q}) \tag {1}
$$

where the first term is the standard InfoNCE and the second is the Reverse Cross Entropy (RCE). Here, p denotes the temperature-scaled softmax distribution of similarity scores, while the target q is set to the one-hot ground-truth y. Note that we apply ϵ-smoothing to q in the RCE term for numerical stability. The bidirectional robust loss is derived as $\begin{array} { r } { \mathcal { L } _ { r o b u s t } = \dot { \frac { 1 } { 2 } } [ \mathcal { L } _ { s c e } ( \mathbf { y } , \mathbf { p } ^ { i 2 t } ) + \mathcal { L } _ { s c e } ( \mathbf { y } , \mathbf { p } ^ { t 2 i } ) ] } \end{array}$ .

• Co-training Phase with Dynamic Partition: Following warm-up, at each epoch, we fit a two-component Gaussian Mixture Model (GMM) to the per-sample loss distributions. Based on the posterior probabilities, we dynamically partition D into a labeled clean subset $\mathcal { D } _ { c l e a n }$ and an unlabeled noisy subset $\mathcal { D } _ { { n o i s y } }$ .

# 3.2. Manifold Stabilization via Intra-modal Constraints

For the identified clean subset $\mathcal { D } _ { c l e a n }$ , our goal is twofold: (1) to align the semantic representations across modalities, and (2) to explicitly consolidate the geometric structure within each modality to support reliable neighbor retrieval.

Formally, consider a batch of clean samples $\begin{array} { r l } { B } & { { } = } \end{array}$ $\{ ( I _ { i } , T _ { i } ) \} _ { i = 1 } ^ { B }$ sampled from $\mathcal { D } _ { c l e a n }$ . Let $\mathbf { v } _ { i } = f _ { \theta } ( I _ { i } )$ and $\mathbf { u } _ { i } = g _ { \phi } ( T _ { i } )$ denote the normalized image and text embeddings. We define the standard bi-directional Triplet

![](images/048c755748047bd0abd1c7be9a1ee2a27970bcdebb9222863750583ebe37cbf2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["A golden retrieval in a living room"] --> B["fθ"]
    A --> C["gφ"]
    B --> D["Manifold Stabilization<br>L_clean = L_inter + L_intra"]
    C --> D
    D --> E["High-Confidence Cross-Model Memory"]
    E --> F["Image Queue ... FIFO"]
    E --> G["Text Queue ... FIFO"]
    B --> H["enqueue"]
    C --> I["enqueue"]
    H --> D
    I --> D
```
</details>

![](images/991ef24d0241d912c5a465788331fbb74abcfefa6fb25c49984461e6271cbf65.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["A sleeping cat"] --> B["fθ"]
    A --> C["gφ"]
    B --> D["Text Queue"]
    C --> E["Image Queue"]
    D --> F["Rectification Lrect"]
    E --> G["Rectification Lrect"]
    F --> H["Graph Refiner"]
    G --> I["Graph Refiner"]
    H --> J["Top-K Neighbors"]
    I --> K["Top-K Neighbors"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#ccf,stroke:#333
    style D fill:#cfc,stroke:#333
    style E fill:#cfc,stroke:#333
    style F fill:#fcc,stroke:#333
    style G fill:#fcc,stroke:#333
    style H fill:#cff,stroke:#333
    style I fill:#cff,stroke:#333
    style J fill:#ffc,stroke:#333
    style K fill:#ffc,stroke:#333
```
</details>

Figure 2. The overall framework of Intra-modal Neighbor-aware Noise Rectification $( \mathbf { I N } ^ { 2 } \mathbf { R } ) .$ . (Top) Manifold Stabilization: For identified clean pairs, we minimize ${ \mathcal { L } } _ { \mathrm { c l e a n } }$ (combining inter-modal alignment and intra-modal constraints) to consolidate the geometric structure, while pushing high-confidence representations into the Cross-Model Memory. (Bottom) Graph-Guided Continuous Rectification: For noisy pairs, we retrieve the Top-K intra-modal neighbors from the memory queue. A learnable Graph Refiner then performs relational reasoning over these neighbors to synthesize a continuous, robust soft prototype. This synthesized target provides fine-grained supervision via ${ \mathcal { L } } _ { \mathrm { r e c t } } .$ , correcting the noisy correspondence.

Ranking Loss $\mathcal { L } _ { t r i p l e t } \mathrm { a s } \mathrm { : }$

$$
\begin{array}{l} \mathcal {L} _ {\text { triplet }} (\mathbf {X}, \mathbf {Y}) = \sum_ {i = 1} ^ {B} \left([ \alpha - S (\mathbf {x} _ {i}, \mathbf {y} _ {i}) + S (\mathbf {x} _ {i}, \mathbf {y} _ {i} ^ {-}) ] _ {+} \right. \\ \left. + \left[ \alpha - S (\mathbf {x} _ {i}, \mathbf {y} _ {i}) + S (\mathbf {x} _ {j} ^ {-}, \mathbf {y} _ {i}) \right] _ {+}\right) \tag {2} \\ \end{array}
$$

where $S ( \cdot , \cdot )$ computes the cosine similarity, α is the margin parameter, $\begin{array} { r c l } { [ x ] _ { + } } & { = } & { \operatorname* { m a x } ( 0 , x ) } \end{array}$ , and $\begin{array} { r l } { { \bf y } _ { i } ^ { - } } & { { } = } \end{array}$ arg $\textstyle \operatorname* { m a x } _ { j \neq i } S ( \mathbf { x } _ { i } , \mathbf { y } _ { j } )$ and $\mathbf { x } _ { j } ^ { - } = \mathrm { a r g }$ max $\mathbf { \chi } _ { : j \neq i } S ( \mathbf { x } _ { j } , \mathbf { y } _ { i } )$ denote the hardest negatives within the batch.

# 3.2.1. INTER-MODAL ALIGNMENT

To bridge the modality gap and align semantic semantics, we apply the ranking loss to the paired image and text embeddings:

$$
\mathcal {L} _ {\text { inter }} = \mathcal {L} _ {\text { triplet }} (\mathbf {V}, \mathbf {U}) \tag {3}
$$

where V and U represent the sets of embeddings in the batch.

# 3.2.2. INTRA-MODAL GEOMETRIC CONSISTENCY

Relying solely on inter-modal alignment often leads to semantic misalignment within individual modalities, where semantically similar instances may not be clustered effectively. Inspired by ConVSE (Liu et al., 2022), we introduce explicit intra-modal constraints to regularize the feature space.

Following the strategy in (Liu et al., 2022), we employ Random Dropout as a data augmentation technique to generate positive pairs without requiring external data. Specifically, we forward the same batch of images (or texts) through the encoder twice with different dropout masks, yielding two views for each sample: original views $\{ \mathbf { v } _ { i } , \mathbf { u } _ { i } \}$ and augmented views $\{ \mathbf { v } _ { i } ^ { \prime } , \mathbf { u } _ { i } ^ { \prime } \}$ .

We then impose the ranking constraints within each modality to pull these intrinsic positive pairs together while pushing

away different instances:

$$
\mathcal {L} _ {i m g} = \mathcal {L} _ {t r i p l e t} (\mathbf {V}, \mathbf {V} ^ {\prime}), \quad \mathcal {L} _ {t x t} = \mathcal {L} _ {t r i p l e t} (\mathbf {U}, \mathbf {U} ^ {\prime}) \tag {4}
$$

By explicitly enforcing $\mathcal { L } _ { i m g }$ and $\mathcal { L } _ { t x t } .$ , we ensure that the visual (or textual) proximity faithfully reflects semantic similarity, preventing the manifold from collapsing due to noise.

# 3.2.3. OPTIMIZATION OBJECTIVE FOR CLEAN DATA

The final objective for the clean subset integrates both constraints:

$$
\mathcal {L} _ {\text { clean }} = \mathcal {L} _ {\text { inter }} + \lambda_ {\text { intra }} (\mathcal {L} _ {\text { img }} + \mathcal {L} _ {\text { txt }}) \tag {5}
$$

where $\lambda _ { i n t r a }$ balances the contribution of the manifold regularization. This structure-first approach ensures a robust topological backbone for the subsequent rectification of noisy data.

# 3.3. High-Confidence Cross-Model Memory

To rectify noisy samples, we require a pristine source of semantic reference. Relying solely on the current minibatch is insufficient due to its limited scope, while using a static dataset fails to track the evolving feature space. Therefore, we maintain a Dynamic Cross-Model Memory Bank Q to store a history of reliable representations.

# 3.3.1. ELITIST ROLLING UPDATE STRATEGY

Standard memory banks typically enqueue all clean samples. However, the identified clean set $\mathcal { D } _ { c l e a n }$ may still contain “hard” noise with borderline confidence. Moreover, as the encoders evolve, stored features from early epochs become stale. To address these issues, we employ an Elitist Rolling Update mechanism:

• Dynamic High-Confidence Filtering: We compute a dynamic threshold τ (t)dyn $\tau _ { d y n } ^ { ( t ) }$ at epoch t as the average confidence of the current clean set. Only samples satisfying $p _ { i } > \tau _ { d y n } ^ { ( t ) }$ are considered as elite candidates.   
• FIFO Maintenance with Gradient Detachment: We maintain the memory as a First-In-First-Out (FIFO) queue. In each iteration, the embeddings of these elite candidates are detached from the computation graph and pushed into the queue, replacing the oldest entries.

This ensures that the memory bank $\mathcal { Q }$ preserves only the most trustworthy and up-to-date representations of the manifold.

# 3.3.2. CROSS-MODEL DECOUPLING

To prevent confirmation bias—where a network reinforces its own erroneous predictions—we leverage the co-training architecture to decouple the retrieval source from the query. Specifically, network $\mathcal { M } _ { A }$ retrieves neighbors exclusively from the memory queue of its peer $\mathcal { M } _ { B }$ , denoted as $\mathcal { Q } _ { B }$ , and vice versa:

$$
\mathcal {Q} _ {B} = \left\{\left(\mathbf {v} _ {k}, \mathbf {u} _ {k}\right) \right\} _ {k = 1} ^ {M} \tag {6}
$$

where M is the memory capacity. By querying the peer’s historical consensus, the model avoids verifying its own potential hallucinations.

# 3.4. Graph-Guided Continuous Rectification

For the identified noisy subset $\mathcal { D } _ { n o i s y } .$ , the original labels are deemed unreliable. We propose to rectify them by synthesizing continuous supervision signals derived from the clean manifold. Remark on Symmetry: Our rectification process is designed to be bidirectional and symmetric. For a noisy pair $( I _ { i } , T _ { i } )$ , we simultaneously rectify the imageto-text direction (synthesizing a Soft Textual Prototype ˆt) and the text-to-image direction (synthesizing a Soft Visual Prototype vˆ). For brevity, we detail the generation of ˆt given a noisy image query $\mathbf { q } _ { v } = f _ { \theta } ( I _ { i } )$ ; the generation of vˆ from a noisy text query $\mathbf { q } _ { u } = g _ { \phi } ( T _ { i } )$ follows an identical symmetric procedure.

# 3.4.1. INTRA-MODAL NEIGHBOR RETRIEVAL

We first query the peer memory bank $\mathcal { Q } _ { B }$ to identify the local semantic support set. Specifically, we retrieve the Top-K nearest neighbors based on the cosine similarity between the query $\mathbf { q } _ { v }$ and the stored visual keys $\{ \mathbf { v } _ { k } \} \in \mathcal { Q } _ { B }$ . Crucially, to bridge the modality gap, we access the paired textual values of these neighbors, denoted as $\mathcal { U } _ { n e i g h b o r } =$ $\left\{ { \bf u } _ { 1 } , \ldots , { \bf u } _ { K } \right\}$ , to serve as the candidate semantic targets.

# 3.4.2. PROTOTYPE SYNTHESIS VIA GRAPH REFINER

A naive approach would be to average $\mathcal { U } _ { n e i g h b o r }$ (mean pooling) or select the top-1 candidate (discrete selection). However, the retrieved neighborhood may contain outliers or irrelevant samples. To mitigate this, we treat $\mathcal { U } _ { n e i g h b o r }$ as nodes in a fully connected graph and employ a learnable Graph Refiner to synthesize a robust prototype.

The Graph Refiner consists of a Multi-Head Self-Attention (MHSA) layer followed by a feed-forward aggregation. Let $\mathbf { H } \in \mathbb { R } ^ { K \times \dot { d } }$ denote the stacked features of $\mathcal { U } _ { n e i g h b o r }$ . We first compute the intra-neighborhood attention to model the relational density:

$$
\operatorname{Attn} (\mathbf {H}) = \text { Softmax } \left(\frac {\left(\mathbf {H} \mathbf {W} _ {Q}\right) \left(\mathbf {H} \mathbf {W} _ {K}\right) ^ {T}}{\sqrt {d _ {k}}}\right) \left(\mathbf {H} \mathbf {W} _ {V}\right) \tag {7}
$$

where $\mathbf { W } _ { Q } , \mathbf { W } _ { K } , \mathbf { W } _ { V }$ are learnable projection matrices. This mechanism assigns higher weights to neighbors that form a semantic consensus and suppresses outliers. The refined features are then obtained via a residual connection and layer normalization:

Table 1. Image-Text Retrieval on Flickr30K and MS-COCO 1K datasets under different noise ratios. \* indicates the noise robust method. The best indicators are represented in bold. 

<table><tr><td rowspan="3">Noise</td><td rowspan="3">Method</td><td colspan="7">Flickr30k 1K Test</td><td colspan="7">MS-COCO 5-fold 1K Test</td></tr><tr><td colspan="3">Text Retrieval</td><td colspan="3">Image Retrieval</td><td rowspan="2">rSum</td><td colspan="3">Text Retrieval</td><td colspan="3">Image Retrieval</td><td rowspan="2">rSum</td></tr><tr><td>R1</td><td>R5</td><td>R10</td><td>R1</td><td>R5</td><td>R10</td><td>R1</td><td>R5</td><td>R10</td><td>R1</td><td>R5</td><td>R10</td></tr><tr><td rowspan="8">0.2</td><td>NCR (NeurIPS&#x27;21)</td><td>73.5</td><td>93.2</td><td>96.6</td><td>56.9</td><td>82.4</td><td>88.5</td><td>491.1</td><td>76.6</td><td>95.6</td><td>98.2</td><td>60.5</td><td>88.8</td><td>95.0</td><td>515.0</td></tr><tr><td>BiCro (CVPR&#x27;23)</td><td>78.1</td><td>94.4</td><td>97.5</td><td>60.4</td><td>84.4</td><td>89.9</td><td>504.7</td><td>78.8</td><td>96.1</td><td>98.6</td><td>63.7</td><td>90.3</td><td>95.7</td><td>523.2</td></tr><tr><td>L2RM (CVPR&#x27;24)</td><td>77.9</td><td>95.2</td><td>97.8</td><td>59.8</td><td>83.6</td><td>89.5</td><td>503.8</td><td>80.2</td><td>96.3</td><td>98.5</td><td>64.2</td><td>90.1</td><td>95.4</td><td>524.7</td></tr><tr><td>CREAM (TIP&#x27;24)</td><td>77.4</td><td>95.0</td><td>97.5</td><td>58.7</td><td>84.1</td><td>89.8</td><td>502.3</td><td>78.9</td><td>96.3</td><td>98.7</td><td>63.3</td><td>90.5</td><td>95.3</td><td>523.0</td></tr><tr><td>ESC (CVPR&#x27;24)</td><td>79.0</td><td>94.8</td><td>97.5</td><td>59.1</td><td>83.8</td><td>89.1</td><td>503.3</td><td>79.2</td><td>97.0</td><td>99.1</td><td>64.8</td><td>90.7</td><td>96.0</td><td>526.8</td></tr><tr><td>GSC (CVPR&#x27;24)</td><td>78.3</td><td>94.6</td><td>97.8</td><td>60.1</td><td>84.5</td><td>90.5</td><td>505.8</td><td>79.5</td><td>96.4</td><td>98.9</td><td>64.4</td><td>90.6</td><td>95.9</td><td>525.7</td></tr><tr><td>SPS (IJCAI&#x27;25)</td><td>79.5</td><td>95.0</td><td>98.0</td><td>60.5</td><td>84.3</td><td>89.8</td><td>507.1</td><td>79.8</td><td>96.4</td><td>98.6</td><td>64.3</td><td>90.5</td><td>95.8</td><td>525.5</td></tr><tr><td>PCSR (AAAI&#x27;26)</td><td>78.7</td><td>95.1</td><td>97.9</td><td>60.9</td><td>83.7</td><td>89.4</td><td>505.7</td><td>80.5</td><td>97.3</td><td>98.9</td><td>63.8</td><td>89.7</td><td>95.1</td><td>525.4</td></tr><tr><td></td><td>IN $^{2}$ R</td><td>80.0</td><td>95.5</td><td>97.7</td><td>60.5</td><td>86.2</td><td>91.7</td><td>511.6</td><td>81.6</td><td>96.3</td><td>98.8</td><td>64.2</td><td>90.5</td><td>95.9</td><td>527.3</td></tr><tr><td rowspan="8">0.4</td><td>NCR (NeurIPS&#x27;21)</td><td>75.3</td><td>92.1</td><td>95.2</td><td>56.2</td><td>80.6</td><td>87.4</td><td>486.8</td><td>76.5</td><td>95.0</td><td>98.2</td><td>60.7</td><td>88.5</td><td>95.0</td><td>513.9</td></tr><tr><td>BiCro (CVPR&#x27;23)</td><td>74.6</td><td>92.7</td><td>96.2</td><td>55.5</td><td>81.0</td><td>87.4</td><td>487.5</td><td>77.0</td><td>95.9</td><td>98.3</td><td>61.1</td><td>89.2</td><td>94.9</td><td>517.1</td></tr><tr><td>L2RM (CVPR&#x27;24)</td><td>75.8</td><td>93.2</td><td>96.9</td><td>56.3</td><td>81.0</td><td>87.3</td><td>490.5</td><td>77.5</td><td>95.8</td><td>98.5</td><td>62.0</td><td>89.1</td><td>94.9</td><td>517.7</td></tr><tr><td>CREAM (TIP&#x27;24)</td><td>76.3</td><td>93.3</td><td>97.1</td><td>57.0</td><td>82.6</td><td>88.7</td><td>495.0</td><td>76.5</td><td>95.0</td><td>98.2</td><td>61.7</td><td>89.4</td><td>95.6</td><td>516.8</td></tr><tr><td>ESC (CVPR&#x27;24)</td><td>76.1</td><td>93.1</td><td>96.4</td><td>56.0</td><td>80.8</td><td>87.2</td><td>489.6</td><td>78.6</td><td>96.6</td><td>99.0</td><td>63.2</td><td>90.6</td><td>95.9</td><td>523.9</td></tr><tr><td>GSC (CVPR&#x27;24)</td><td>76.5</td><td>94.1</td><td>97.6</td><td>57.5</td><td>82.7</td><td>88.9</td><td>497.3</td><td>78.2</td><td>95.9</td><td>98.2</td><td>62.5</td><td>89.7</td><td>95.4</td><td>519.9</td></tr><tr><td>SPS (IJCAI&#x27;25)</td><td>77.8</td><td>93.6</td><td>97.1</td><td>57.3</td><td>83.5</td><td>89.6</td><td>498.9</td><td>79.2</td><td>95.9</td><td>98.5</td><td>63.3</td><td>89.8</td><td>95.4</td><td>522.1</td></tr><tr><td>PCSR (AAAI&#x27;26)</td><td>76.6</td><td>94.5</td><td>97.3</td><td>57.4</td><td>82.4</td><td>88.9</td><td>497.1</td><td>77.7</td><td>96.0</td><td>98.1</td><td>63.1</td><td>89.4</td><td>95.5</td><td>519.8</td></tr><tr><td></td><td>IN $^{2}$ R</td><td>78.3</td><td>94.5</td><td>97.2</td><td>58.7</td><td>84.7</td><td>90.7</td><td>504.1</td><td>79.8</td><td>95.8</td><td>98.6</td><td>63.4</td><td>90.1</td><td>95.4</td><td>523.2</td></tr><tr><td rowspan="8">0.6</td><td>NCR (NeurIPS&#x27;21)</td><td>68.7</td><td>89.9</td><td>95.5</td><td>52.0</td><td>77.6</td><td>84.9</td><td>468.6</td><td>72.7</td><td>94.0</td><td>97.6</td><td>57.9</td><td>87.0</td><td>94.1</td><td>503.3</td></tr><tr><td>BiCro (CVPR&#x27;23)</td><td>67.6</td><td>90.0</td><td>94.4</td><td>51.2</td><td>77.6</td><td>84.7</td><td>466.3</td><td>73.9</td><td>94.4</td><td>97.8</td><td>58.3</td><td>87.2</td><td>93.5</td><td>505.5</td></tr><tr><td>L2RM (CVPR&#x27;24)</td><td>70.0</td><td>90.8</td><td>95.4</td><td>51.3</td><td>76.4</td><td>83.7</td><td>467.6</td><td>75.4</td><td>94.7</td><td>97.9</td><td>59.2</td><td>87.4</td><td>93.6</td><td>508.4</td></tr><tr><td>CREAM (TIP&#x27;24)</td><td>70.6</td><td>91.2</td><td>96.1</td><td>53.3</td><td>79.2</td><td>87.0</td><td>477.4</td><td>74.7</td><td>94.7</td><td>98.0</td><td>59.7</td><td>88.0</td><td>94.6</td><td>509.9</td></tr><tr><td>ESC (CVPR&#x27;24)</td><td>72.6</td><td>90.9</td><td>94.6</td><td>53.0</td><td>78.6</td><td>85.3</td><td>475.0</td><td>77.2</td><td>95.1</td><td>98.1</td><td>61.1</td><td>88.6</td><td>94.9</td><td>515.0</td></tr><tr><td>GSC (CVPR&#x27;24)</td><td>70.8</td><td>91.1</td><td>95.9</td><td>53.6</td><td>79.8</td><td>86.8</td><td>478.0</td><td>75.6</td><td>95.1</td><td>98.0</td><td>60.0</td><td>88.3</td><td>94.6</td><td>511.7</td></tr><tr><td>SPS (IJCAI&#x27;25)</td><td>73.4</td><td>92.7</td><td>96.3</td><td>53.7</td><td>80.2</td><td>87.7</td><td>484.1</td><td>77.6</td><td>95.7</td><td>98.3</td><td>61.6</td><td>89.0</td><td>95.1</td><td>517.2</td></tr><tr><td>PCSR (AAAI&#x27;26)</td><td>72.8</td><td>91.9</td><td>95.8</td><td>54.8</td><td>79.7</td><td>86.5</td><td>480.5</td><td>76.3</td><td>94.8</td><td>97.9</td><td>61.7</td><td>87.5</td><td>94.0</td><td>512.2</td></tr><tr><td></td><td>IN $^{2}$ R</td><td>75.1</td><td>94.0</td><td>97.2</td><td>56.5</td><td>82.9</td><td>89.5</td><td>495.2</td><td>78.8</td><td>95.2</td><td>98.3</td><td>61.4</td><td>89.2</td><td>95.1</td><td>518.0</td></tr><tr><td rowspan="5">0.8</td><td>NCR (NeurIPS&#x27;21)</td><td>1.5</td><td>6.2</td><td>9.9</td><td>0.3</td><td>1.0</td><td>2.1</td><td>21.0</td><td>0.1</td><td>0.3</td><td>0.4</td><td>0.1</td><td>0.5</td><td>1.0</td><td>2.4</td></tr><tr><td>BiCro (CVPR&#x27;23)</td><td>2.3</td><td>9.2</td><td>17.2</td><td>2.6</td><td>10.2</td><td>16.8</td><td>58.3</td><td>62.2</td><td>88.6</td><td>94.6</td><td>47.4</td><td>79.2</td><td>88.5</td><td>460.5</td></tr><tr><td>L2RM (CVPR&#x27;24)</td><td>55.7</td><td>80.8</td><td>87.8</td><td>39.4</td><td>65.4</td><td>74.9</td><td>404.0</td><td>69.0</td><td>91.9</td><td>96.4</td><td>52.6</td><td>82.4</td><td>90.3</td><td>482.6</td></tr><tr><td>CREAM (TIP&#x27;24)</td><td>58.7</td><td>83.3</td><td>90.1</td><td>40.8</td><td>67.1</td><td>76.3</td><td>416.3</td><td>68.6</td><td>92.0</td><td>96.4</td><td>52.4</td><td>84.8</td><td>92.8</td><td>487.2</td></tr><tr><td>PCSR (AAAI&#x27;26)</td><td>63.1</td><td>87.1</td><td>93.2</td><td>44.6</td><td>70.9</td><td>78.6</td><td>437.5</td><td>71.6</td><td>92.9</td><td>96.7</td><td>55.3</td><td>83.9</td><td>93.3</td><td>493.7</td></tr><tr><td></td><td>IN $^{2}$ R</td><td>68.5</td><td>88.1</td><td>93.0</td><td>48.2</td><td>76.4</td><td>84.7</td><td>458.8</td><td>73.2</td><td>93.8</td><td>97.4</td><td>57.1</td><td>86.4</td><td>93.4</td><td>501.3</td></tr></table>

$$
\mathbf {H} ^ {\prime} = \text { LayerNorm } (\mathbf {H} + \text { Dropout } (\text { Linear } (\text { Attn } (\mathbf {H})))) \tag {8}
$$

Finally, we perform mean pooling over the refined nodes H′ to synthesize the continuous Soft Textual Prototype ˆt:

$$
\hat {\mathbf {t}} = \frac {1}{K} \sum_ {k = 1} ^ {K} \mathbf {H} _ {k} ^ {\prime} \tag {9}
$$

Symmetrically, for a noisy text query, we obtain the Soft Visual Prototype vˆ using the same Graph Refiner shared across modalities.

# 3.4.3. RECTIFICATION OBJECTIVE

To utilize the embedding prototype tˆ for supervision, we convert it into a soft target distribution $q ^ { i 2 t } ( { \hat { t } } )$ by computing its softmax-normalized similarity against the current batch embeddings U:

$$
q ^ {i 2 t} (\hat {t}) = \text { Softmax } \left(\frac {\hat {t} \cdot U ^ {\top}}{\tau}\right) \tag {10}
$$

Symmetrically, we derive $q ^ { t 2 i } ( { \hat { v } } )$ for the text-to-image direction. We then employ the robust objective from Eq. (1), utilizing these calibrated distributions as targets:

$$
\mathcal {L} _ {\text { rect }} = \frac {1}{2} \left[ \mathcal {L} _ {s c e} (q ^ {i 2 t} (\hat {t}), \mathbf {p} ^ {i 2 t}) + \mathcal {L} _ {s c e} (q ^ {t 2 i} (\hat {v}), \mathbf {p} ^ {t 2 i}) \right] \tag {11}
$$

This objective aligns noisy samples with the intra-modal consensus while regularizing against estimation bias via the RCE term.

# 3.5. Overall Optimization

The final training objective integrates the structural constraints from clean data and the rectified supervision from noisy data. For each network $( \mathbf { e } . \mathbf { g } . , \mathcal { M } _ { A } )$ , the total loss is defined as a weighted sum:

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {\text { clean }} + \gamma \mathcal {L} _ {\text { rect }} \tag {12}
$$

where $\mathcal { L } _ { c l e a n } \left( \mathrm { E q . ~ } 5 \right)$ consolidates the manifold structure using the hard-margin ranking loss, and $\mathcal { L } _ { r e c t }$ (Eq. 11) guides the rectification using the robust symmetric loss. The hyperparameter $\gamma$ balances the contribution of the synthesized supervision.

Table 2. Comparisons with real-world NCs on CC152K. The best performance is highlighted in bold. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Text Retrieval</td><td colspan="3">Image Retrieval</td><td rowspan="2">rSum</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>NCR</td><td>39.5</td><td>64.5</td><td>73.5</td><td>40.3</td><td>64.6</td><td>73.2</td><td>355.6</td></tr><tr><td>BiCro</td><td>40.8</td><td>67.2</td><td>76.1</td><td>42.1</td><td>67.6</td><td>76.4</td><td>370.2</td></tr><tr><td> $PC^2$ </td><td>39.3</td><td>66.4</td><td>75.4</td><td>39.8</td><td>66.4</td><td>76.8</td><td>364.1</td></tr><tr><td>L2RM</td><td>43.0</td><td>67.5</td><td>75.7</td><td>42.8</td><td>68.0</td><td>77.2</td><td>374.2</td></tr><tr><td>ESC</td><td>42.8</td><td>67.3</td><td>76.9</td><td>44.8</td><td>68.2</td><td>75.9</td><td>375.9</td></tr><tr><td>GSC</td><td>42.1</td><td>68.4</td><td>77.7</td><td>42.2</td><td>67.6</td><td>77.1</td><td>375.1</td></tr><tr><td>SPS</td><td>40.8</td><td>67.9</td><td>77.7</td><td>42.4</td><td>69.5</td><td>78.0</td><td>376.3</td></tr><tr><td>PCSR</td><td>43.7</td><td>67.7</td><td>77.2</td><td>43.1</td><td>67.7</td><td>76.3</td><td>375.3</td></tr><tr><td> $IN^2R$ </td><td>45.2</td><td>69.0</td><td>76.8</td><td>43.3</td><td>68.3</td><td>78.2</td><td>380.8</td></tr></table>

# 4. Experiments

# 4.1. Datasets

We evaluate on three datasets: Flickr30K (Plummer et al., 2015), MS-COCO (Lin et al., 2014), and Conceptual Captions (CC) (Huang et al., 2021). Flickr30K contains 31K images (5 captions each); following (Faghri et al., 2018), we use 1K images each for validation and testing. MS-COCO comprises 123K images (5 captions each); we utilize 566K pairs for training, with 25K pairs each allocated for validation and testing. CC is a noisy, web-harvested dataset (1 caption each). We use the CC152K subset, consisting of 150K training images and 1K each for validation and testing.

Noise Simulation and Evaluation. Following (Huang et al., 2021), we assess retrieval performance using Recall at K (R@K), which quantifies the percentage of relevant items correctly identified within the top K retrieved results. We report R@1, R@5, R@10, and the cumulative recall score (rSum) for bidirectional matching tasks.

# 4.2. Implementation Details

Following standard protocols in noisy correspondence learning (Huang et al., 2021; Han et al., 2024) we utilize preextracted features to ensure a fair comparison. For images, we use the bottom-up attention features extracted from a pre-trained Faster R-CNN (2048-d). For text, we use a Bi-GRU to extract sentence embeddings. These features are projected into a common D-dimensional metric space (e.g., $D = 1 0 2 4 )$ via the learnable encoders $f _ { \theta }$ and $g _ { \phi }$ . More implementation details are provided in the supplementary material.

# 4.3. Main Results

In this section, we carry out a comprehensive evaluation to present the effectiveness of $\mathrm { I N ^ { 2 } R }$ , benchmarking it against state-of-the-art (SOTA) baselines across three widely-used datasets. The baselines comprise NCR (Huang et al., 2021), BiCro (Yang et al., 2023), L2RM (Han et al., 2024), CREAM (Ma et al., 2024), ESC (Yang et al., 2024), GSC (Zhao et al., 2024), SPS (Xie et al., 2025), and PCSR (Liu et al., 2026). We evaluate $\mathrm { I N ^ { 2 } R }$ on Flickr30K and MS-COCO with simulated noise rates of 20%–80% generated by random shuffling. Additionally, we validate performance on the real-world CC152K dataset, which contains inherent web noise. All reported test results are based on the optimal validation checkpoint.

Evaluation under Simulated Noise. Table 1 presents the comprehensive comparison between our proposed $\mathrm { I N ^ { 2 } R }$ and state-of-the-art methods on Flickr30K and MS-COCO datasets under varying symmetric noise ratios (20% to 80%). The results demonstrate that $\mathrm { I N ^ { 2 } R }$ consistently outperforms existing baselines across all noise levels and metrics.

Robustness in High-Noise Regimes: $\mathrm { I N ^ { 2 } R }$ excels as noise increases. At 80% noise, where NCR collapses (21.0 rSum), our method maintains remarkable stability. On Flickr30K, $\mathrm { I N ^ { 2 } R }$ achieves 458.8 rSum, surpassing PCSR by 21.3 points. Similarly, on MS-COCO, it sets a new SOTA of 501.3 (+7.6 points). This validates that our Graph-Guided Continuous Rectification effectively synthesizes reliable supervision even under extreme corruption.

Improvements in Low-Noise Regimes: Even at lower noise ratios (20% and 40%), where the clean data signal is stronger, $\mathrm { I N ^ { 2 } R }$ continues to refine the boundary of performance. For instance, at 20% noise on Flickr30K, our method achieves an rSum of 511.6, outperforming the correction-based method SPS (507.1). This indicates that our manifold stabilization strategy and intra-modal reasoning are beneficial not just for correcting errors, but for learning a more discriminative feature space overall.

# Evaluation under Real-World Noise.

We evaluate on Conceptual Captions (CC152K) to verify robustness against heterogeneous web noise. $\mathrm { I N ^ { 2 } R }$ generalizes remarkably well, achieving a state-of-the-art rSum of 380.8, surpassing SPS (376.3) by 4.5 points. Notably, it attains a Text Retrieval R@1 of 45.2%, significantly outperforming PCSR (43.7%). While baselines like ESC show isolated strengths, IN2R delivers the most balanced performance across modalities. This confirms that our intra-modal rectification effectively handles naturally occurring mismatches without requiring manual noise priors.

# 4.4. Ablation Studies

To provide a comprehensive understanding of the proposed framework, we conduct extensive ablation studies on the Flickr30K dataset. Unless otherwise specified, we report results under the 60% symmetric noise setting, as high-noise scenarios best highlight the robustness of our rectification

mechanism.

Impact of Key Components. We investigate the contribution of each module in $\mathrm { I N ^ { 2 } R }$ by incrementally adding them to the baseline on the Flickr30K dataset under 60% noise. The baseline, trained solely with the inter-modal ranking loss $\mathcal { L } _ { i n t e r } ,$ , yields an rSum of 469.1. This relatively low performance highlights the difficulty of learning robust representations from heavily corrupted data without explicit correction.

Incorporating the intra-modal geometric constraints $( \mathcal { L } _ { i n t r a } )$ improves the rSum to 475.2, a gain of 6.1 points. This suggests that stabilizing the feature manifold prevents the model from overfitting to noisy correspondence, providing a better initialization for retrieval. Furthermore, applying the Graph-Guided Rectification $( \mathcal { L } _ { r e c t } )$ directly to the baseline yields a significant boost, reaching 483.7 rSum. Finally, the full $\mathrm { I N ^ { 2 } R }$ framework, which integrates both manifold stabilization and continuous rectification, achieves the best performance with an rSum of 495.2. This represents a substantial total improvement of 26.1 points over the baseline, confirming that the two components work synergistically: $\mathcal { L } _ { i n t r a }$ constructs a reliable geometric basis, while $\mathcal { L } _ { r e c t }$ actively synthesizes precise supervision signals from it.

Table 3. Component-wise ablation study on Flickr30K (60% noise). $\mathcal { L } _ { i n t e r } \colon$ Inter-modal alignment; $\mathcal { L } _ { i n t r a } { : }$ Intra-modal geometric constraints; $\mathcal { L } _ { r e c t } { : }$ Graph-guided rectification. 

<table><tr><td colspan="3">Components</td><td colspan="2">Text Retrieval</td><td colspan="2">Image Retrieval</td><td></td><td></td></tr><tr><td> $\mathcal{L}_{inter}$ </td><td> $\mathcal{L}_{intra}$ </td><td> $\mathcal{L}_{rect}$ </td><td>R@1</td><td>R@10</td><td>R@1</td><td>R@10</td><td>rSum</td><td> $\Delta$ </td></tr><tr><td>√</td><td></td><td></td><td>68.7</td><td>95.5</td><td>52.4</td><td>84.7</td><td>469.1</td><td>-</td></tr><tr><td>√</td><td>√</td><td></td><td>71.2</td><td>96.1</td><td>52.8</td><td>85.3</td><td>475.2</td><td>+[6.1]</td></tr><tr><td>√</td><td></td><td>√</td><td>73.6</td><td>96.8</td><td>53.7</td><td>86.8</td><td>483.7</td><td>+[14.6]</td></tr><tr><td>√</td><td>√</td><td>√</td><td>75.1</td><td>97.2</td><td>56.5</td><td>89.5</td><td>495.2</td><td>+[26.1]</td></tr></table>

Continuous Rectification vs. Discrete Selection. To validate that continuous synthesis outperforms discrete selection, we compare our Graph Refiner with three strategies on Flickr30K (60% noise): None (discarding noise), Hard Selection (Top-1 neighbor), and Mean Pooling (Top-K average). As shown in Table 4, Hard Selection (479.6) suffers from Single-Point Fragility, while Mean Pooling (485.2) fails to filter outliers. In contrast, our Graph Refiner achieves an rSum of 495.2. By synthesizing a robust soft prototype via dynamic re-weighting, it outperforms discrete selection and mean pooling by 15.6 and 10.0 points, respectively.

Impact of Refiner Architecture and Memory Decoupling. We validate our architectural choices on Flickr30K (60% noise) by comparing our Multi-Head Self-Attention (MHSA) refiner against GCN and GAT variants. As shown in Table 5, MHSA achieves the best rSum of 495.2, significantly outperforming GCN (486.3) and GAT (490.5). This superiority suggests that MHSA’s dense, fully-connected attention models the global neighborhood consensus more effectively than the fixed or local topologies of the baselines. Additionally, replacing the Cross-Model Memory with a “Self-Memory” variant leads to a performance drop, confirming that decoupling the query and retrieval networks is essential to mitigate confirmation bias.

Table 4. Comparison of different rectification strategies on Flickr30K (60% noise). Our Graph Refiner (Continuous) outperforms Discrete Selection (Top-1) and naive averaging. 

<table><tr><td rowspan="2">Rectification Strategy</td><td colspan="2">Text Retrieval</td><td colspan="2">Image Retrieval</td><td rowspan="2">rSum</td></tr><tr><td>R@1</td><td>R@10</td><td>R@1</td><td>R@10</td></tr><tr><td>None (Discard Noise)</td><td>70.0</td><td>93.9</td><td>50.6</td><td>85.5</td><td>467.7</td></tr><tr><td>Hard Selection (Top-1)</td><td>73.1</td><td>95.1</td><td>53.3</td><td>87.0</td><td>479.6</td></tr><tr><td>Mean Pooling (Top-K)</td><td>74.0</td><td>95.8</td><td>55.1</td><td>87.7</td><td>485.2</td></tr><tr><td>Graph Refiner (Ours)</td><td>75.1</td><td>97.2</td><td>56.5</td><td>89.5</td><td>495.2</td></tr></table>

![](images/fc690c34efd37b5737b9068ec03c93f386239b7d831bd01cb8a4e12484410417.jpg)

<details>
<summary>line</summary>

| Memory Queue Size (M) | Text R@1 | Image R@1 |
| --------------------- | -------- | --------- |
| 2k                    | 73       | 54        |
| 4k                    | 74       | 55        |
| 8k                    | 74       | 56        |
| 16k                   | 75       | 57        |
| 32k                   | 75       | 57        |
| 65k                   | 75       | 57        |
</details>

![](images/c832c3b41a5361e025c391b8de13666a0169701cf56330ee54336c03adee206a.jpg)

<details>
<summary>line</summary>

| Number of Neighbors (K) | Text R@1 | Image R@1 |
| ----------------------- | -------- | --------- |
| 1                       | 71       | 53        |
| 3                       | 74       | 55        |
| 5                       | 75       | 56        |
| 10                      | 74       | 56        |
| 20                      | 73       | 56        |
</details>

Figure 3. Hyperparameter Sensitivity Analysis on R@1. We report the R@1 performance for both Text Retrieval (Blue) and Image Retrieval (Orange). (Left) Performance improves with memory size M and saturates at $M = 3 2 k$ . (Right) The retrieval accuracy peaks at $K = 5 ,$ demonstrating that a moderate neighbor count effectively balances semantic consensus and noise introduction.

Table 5. Comparison of different Graph Refiner architectures on Flickr30K (60% noise). 

<table><tr><td rowspan="2">Rectification Strategy</td><td colspan="2">Text Retrieval</td><td colspan="2">Image Retrieval</td><td rowspan="2">rSum</td></tr><tr><td>R@1</td><td>R@10</td><td>R@1</td><td>R@10</td></tr><tr><td>GCN (Fixed Graph)</td><td>74.9</td><td>95.1</td><td>53.7</td><td>87.6</td><td>486.1</td></tr><tr><td>GAT (Graph Attn)</td><td>75.3</td><td>96.6</td><td>55.7</td><td>88.1</td><td>490.5</td></tr><tr><td>MHSA</td><td>75.1</td><td>97.2</td><td>56.5</td><td>89.5</td><td>495.2</td></tr></table>

Hyperparameter Sensitivity Analysis. We analyze the sensitivity of IN2R to memory size M and neighbor count K on Flickr30K (60% noise). Figure 3 shows that performance improves with M and saturates around 32k, confirming that a sufficiently large memory captures the global distribution needed for high-fidelity retrieval. We thus adopt $M \ =$ 65, 536. Regarding K, performance peaks at $K = 5$ . This configuration represents the optimal trade-off: it aggregates sufficient local consensus to mitigate “Single-Point Fragility” (K = 1) while avoiding the semantic dilution caused by distant outliers in larger neighborhoods $( K > 1 0 )$ .

# 5. Conclusion

In this paper, we proposed the Intra-modal Neighboraware Noise Rectification (IN2R) framework to address noisy correspondence. Departing from the limitations of discrete selection, our work pioneers a shift towards continuous prototype synthesis, leveraging intra-modal geometric constraints to reconstruct reliable supervision targets. Extensive experiments on Flickr30K, MS-COCO, and Conceptual Captions demonstrate the superiority of our approach. IN2R not only establishes new state-of-the-art results but also exhibits remarkable stability in extreme noise regimes (up to 80% noise) and real-world web-noise scenarios.

# Acknowledgments

This work was partially supported by the the National Science Foundation of China under Grant 62376175 and 22494712, the 111 Project under Grant B21044, the Science Fund for Creative Research Groups of Sichuan Province Natural Science Foundation under Grant 2024NSFTD0035, and the National Science Foundation of Sichuan Province under Grant 2025ZNSFSC0480.

# Impact Statement

This paper presents a method for training robust visionlanguage models using noisy datasets derived from the web. Our work contributes to the democratization of AI research by reducing the reliance on expensive, human-annotated datasets, thereby making large-scale pre-training more accessible. However, we acknowledge that our approach relies on the topological consensus of local neighborhoods to rectify noisy labels. If the underlying data distribution contains societal biases or stereotypes, reliance on local consensus could potentially amplify these biases by smoothing out minority but valid representations. While our method focuses on correcting correspondence noise, future work should consider how such rectification mechanisms interact with data fairness and bias mitigation.

# References

Chen, H., Ding, G., Liu, X., Lin, Z., Liu, J., and Han, J. Imram: Iterative matching with recurrent attention memory for cross-modal image-text retrieval. In CVPR, pp. 12655–12663, 2020.   
Chen, J., Hu, H., Wu, H., Jiang, Y., and Wang, C. Learning the best pooling strategy for visual semantic embedding. In CVPR, pp. 15789–15798, 2021.   
Chun, S. Improved probabilistic image-text representations. arXiv preprint arXiv:2305.18171, 2023.   
Chun, S., Oh, S. J., De Rezende, R. S., Kalantidis, Y., and Larlus, D. Probabilistic embeddings for cross-modal retrieval. In CVPR, pp. 8415–8424, 2021.   
Diao, H., Zhang, Y., Ma, L., and Lu, H. Similarity rea-

soning and filtration for image-text matching. In AAAI, volume 35, pp. 1218–1226, 2021.   
Faghri, F., Fleet, D. J., Kiros, J. R., and Fidler, S. Vse++: Improving visual-semantic embeddings with hard negatives. In BMVC, 2018.   
Han, H., Miao, K., Zheng, Q., and Luo, M. Noisy correspondence learning with meta similarity correction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7517–7526, 2023.   
Han, H., Zheng, Q., Dai, G., Luo, M., and Wang, J. Learning to rematch mismatched pairs for robust cross-modal retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26679– 26688, 2024.   
Huang, Y., Wu, Q., Song, C., and Wang, L. Learning semantic concepts and order for image and sentence matching. In CVPR, pp. 6163–6171, 2018.   
Huang, Z., Niu, G., Liu, X., Ding, W., Xiao, X., Wu, H., and Peng, X. Learning with noisy correspondence for cross-modal matching. Advances in Neural Information Processing Systems, 34:29406–29419, 2021.   
Kim, D., Kim, N., and Kwak, S. Improving cross-modal retrieval with set of diverse embeddings. In CVPR, pp. 23422–23431, 2023.   
Li, K., Zhang, Y., Li, K., Li, Y., and Fu, Y. Visual semantic reasoning for image-text matching. In ICCV, pp. 4654– 4662, 2019.   
Li, K., Zhang, Y., Li, K., Li, Y., and Fu, Y. Image-text embedding learning via visual and textual semantic reasoning. TPAMI, 45(1):641–656, 2022.   
Li, R., Wu, X., and Yang, Y. Noise self-correction via relation propagation for robust cross-modal retrieval. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 4748–4757, 2025.   
Li, Y., Huang, H., Xu, J., and Huang, S.-L. Nac: Mitigating noisy correspondence in cross-modal matching via neighbor auxiliary corrector. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 6815–6819. IEEE, 2024.   
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L. Microsoft ´ coco: Common objects in context. In ECCV, pp. 740–755, 2014.   
Liu, C., Mao, Z., Zhang, T., Xie, H., Wang, B., and Zhang, Y. Graph structured network for image-text matching. In CVPR, pp. 10921–10930, 2020.

Liu, Y., Liu, H., Wang, H., and Liu, M. Regularizing visual semantic embedding with contrastive learning for imagetext matching. IEEE SPL, 2022.   
Liu, Y., Liu, H., Wang, H., Meng, F., and Liu, M. Bcan: Bidirectional correct attention network for cross-modal retrieval. TNNLS, 2023.   
Liu, Y., Feng, W., Liu, Z., Huang, S., and Lv, J. Aligning information capacity between vision and language via dense-to-sparse feature distillation for image-text matching. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 21679–21688, October 2025a.   
Liu, Y., Liu, M., Huang, s., and Lv, J. Asymmetric visual semantic embedding framework for efficient visionlanguage alignment. In AAAI, 2025b.   
Liu, Z., Liu, Y., Feng, W., and Huang, S. Pcsr: Pseudo-label consistency-guided sample refinement for noisy correspondence learning. In AAAI, 2026.   
Ma, X., Yang, M., Li, Y., Hu, P., Lv, J., and Peng, X. Crossmodal retrieval with noisy correspondence via consistency refining and mining. IEEE transactions on image processing, 33:2587–2598, 2024.   
Plummer, B. A., Wang, L., Cervantes, C. M., Caicedo, J. C., Hockenmaier, J., and Lazebnik, S. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In ICCV, pp. 2641–2649, 2015.   
Qin, Y., Peng, D., Peng, X., Wang, X., and Hu, P. Deep evidential learning with noisy correspondence for crossmodal retrieval. In Proceedings of the 30th ACM International Conference on Multimedia, pp. 4948–4956, 2022.   
Qu, L., Liu, M., Wu, J., Gao, Z., and Nie, L. Dynamic modality interaction modeling for image-text retrieval. In SIGIR, pp. 1104–1113, 2021.   
Wang, H., Zhang, Y., Ji, Z., Pang, Y., and Ma, L. Consensusaware visual-semantic embedding for image-text matching. In ECCV, pp. 18–34, 2020.   
Wang, Z., Liu, X., Li, H., Sheng, L., Yan, J., Wang, X., and Shao, J. Camp: Cross-modal adaptive message passing for text-image retrieval. In ICCV, pp. 5764–5773, 2019.   
Wei, J., Xu, X., Yang, Y., Ji, Y., Wang, Z., and Shen, H. T. Universal weighting metric learning for cross-modal matching. In CVPR, pp. 13005–13014, 2020.   
Xie, Y., Cai, S., Tong, T., Hu, P., and Zhu, X. Seeking proxy point via stable feature space for noisy correspondence learning. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, 2025.

Yang, S., Xu, Z., Wang, K., You, Y., Yao, H., Liu, T., and Xu, M. Bicro: Noisy correspondence rectification for multimodality data via bi-directional cross-modal similarity consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19883– 19892, 2023.   
Yang, Y., Wang, L., Yang, E., and Deng, C. Robust noisy correspondence learning with equivariant similarity consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17700– 17709, 2024.   
Zha, Q., Liu, X., Peng, S.-J., Cheung, Y.-m., Xu, X., and Wang, N. Recon: Enhancing true correspondence discrimination through relation consistency for robust noisy correspondence learning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 29680– 29689, 2025.   
Zhang, K., Mao, Z., Wang, Q., and Zhang, Y. Negativeaware attention framework for image-text matching. In CVPR, pp. 15661–15670, 2022.   
Zhang, Q., Lei, Z., Zhang, Z., and Li, S. Z. Context-aware attention network for image-text retrieval. In CVPR, pp. 3536–3545, 2020.   
Zhao, Z., Chen, M., Dai, T., Yao, J., Han, B., Zhang, Y., and Wang, Y. Mitigating noisy correspondence by geometrical structure consistency learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 27381–27390, 2024.

# A. More Implementation Details

Datasets and Features. Following standard protocols in noisy correspondence learning (Huang et al., 2021; Han et al., 2024) we utilize pre-extracted features to ensure a fair comparison. For images, we use the bottom-up attention features extracted from a pre-trained Faster R-CNN (2048-d). For text, we use a Bi-GRU to extract sentence embeddings. These features are projected into a common D-dimensional metric space $( { \mathrm { e . g . , } } D = 1 0 2 4 )$ via the learnable encoders $f _ { \theta }$ and $g _ { \phi }$ . More implementation details are provided in the supplementary material.

Training Settings. Our framework is implemented in PyTorch and trained on a single NVIDIA RTX 3090 GPU. We employ the Adam optimizer with a mini-batch size of 128. The initial learning rate is set to $5 e ^ { - 4 }$ , with a cosine annealing decay schedule. The training process consists of two phases: a warm-up phase for the first $E _ { w a r m } = 5$ epochs to initialize the feature space, followed by the co-training phase for another 40 epochs.

Hyper-parameters. For the $\mathrm { I N ^ { 2 } R }$ specific components:

The Cross-Model Memory Bank size M is set to 65536 for Flickr30K and 448000 for MS-COCO. For Neighbor Retrieval, we retrieve $K = 5$ visual neighbors to construct the semantic graph. The Graph Refiner is implemented as a single-layer Transformer encoder with 4 attention heads. The Loss Weights are empirically set as follows: the intra-modal constraint weight $\lambda _ { i n t r a } = 0 . 1$ , and the rectification weight $\gamma = 1 . 0$ . The Temperature τ in the robust symmetric loss is set to 0.05, and the balancing factor $\alpha / \beta$ for SCE is set to $1 . 0 / 1 . 0$ . The margin α in the ranking loss is set to 0.2.

For synthetic noise experiments, we follow the standard noise generation protocol (Huang et al., 2021) to inject symmetric noise at ratios ranging from 20% to 80%.

Training Procedure. The framework is optimized in an end-to-end manner. In the co-training phase, the peer networks $\mathcal { M } _ { A }$ and $\mathcal { M } _ { B }$ are updated simultaneously but interactively. Specifically, $\mathcal { M } _ { A }$ rectifies its noisy samples by retrieving semantic evidence from the frozen memory queue of $\mathcal { M } _ { B } \left( \mathcal { Q } _ { B } \right)$ , and vice versa. This cross-model interaction ensures that the error flow from one network does not directly propagate to the other, strictly enforcing the decoupling principle required for robust learning.

Inference Efficiency. It is worth emphasizing that the proposed auxiliary modules—including the Cross-Model Memory Bank and the Graph Refiner—are exclusively constructed to facilitate robust training. During the inference phase, these modules are discarded. The model functions strictly as a standard dual-encoder, utilizing only the learned encoders $f _ { \theta } ( \cdot )$ and $g _ { \phi } ( \cdot )$ to compute image-text similarity. Consequently, $\mathrm { I N ^ { 2 } R }$ incurs no additional computational overhead or memory cost during deployment compared to standard baselines.

# B. Justification for Backbone Selection

Table 6. Performance comparison of pure backbones on Flickr30K (Clean setting). GPO achieves competitive performance comparable to the interaction-based SGRAF, yet maintains the efficiency of a dual-encoder architecture. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Text Retrieval</td><td colspan="3">Image Retrieval</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>GPO (Chen et al., 2021)</td><td>74.8</td><td>93.5</td><td>97.0</td><td>55.1</td><td>83.8</td><td>89.4</td></tr><tr><td>SGR (Diao et al., 2021)</td><td>75.2</td><td>93.3</td><td>96.6</td><td>56.2</td><td>81.0</td><td>86.5</td></tr><tr><td>SAF (Diao et al., 2021)</td><td>73.7</td><td>93.3</td><td>96.3</td><td>56.1</td><td>81.5</td><td>88.0</td></tr><tr><td>SGRAF (Diao et al., 2021)</td><td>77.8</td><td>94.1</td><td>97.4</td><td>58.5</td><td>83.0</td><td>88.8</td></tr></table>

In our main experiments, we adopt the Generalized Pooling Operator (GPO) (Chen et al., 2021) as the visual-semantic backbone. This decision is driven by the inherent design philosophy of our $\mathrm { I N ^ { 2 } R }$ framework, which prioritizes the “Index-and-Search” paradigm over computationally expensive cross-modal interactions.

Efficiency Necessity for Intra-Modal Retrieval. While interaction-based methods like SGRAF (Diao et al., 2021) achieve superior performance on standard benchmarks (Table 6), they rely on complex graph reasoning and cross-attention mechanisms to compute similarity. This fine-grained interaction requires heavy computation for every image-text pair $( O ( N ^ { 2 } )$ complexity), prohibiting the use of offline indexing. In contrast, our $\mathrm { I N ^ { 2 } R }$ framework is built upon retrieving topological neighbors from a large-scale dynamic memory bank. GPO, as a dual-encoder, maps images and texts into compact global vectors, allowing for highly efficient nearest neighbor search via simple dot products. This efficiency is critical for the scalability of our rectification mechanism.

Performance-Efficiency Trade-off. As shown in Table 6, GPO delivers competitive performance comparable to SGRAF on clean datasets, serving as a strong baseline. Therefore, we select GPO to demonstrate that our performance gains stem from the proposed rectification strategy rather than a heavy backbone.

Universality of $\mathbf { I N ^ { 2 } R } .$ To further demonstrate that our method is backbone-agnostic, we integrated $\mathrm { I N ^ { 2 } R }$ with the SGRAF backbone on Flickr30K (60% noise). As presented in Table 7, IN2R (SGRAF) consistently outperforms $\mathrm { I N ^ { 2 } R }$ (GPO), benefiting from the stronger representation power. However, this comes at a significant cost: the training time increases from 3.5 min/epoch to 30.0 min/epoch. We maintain that for practical large-scale retrieval, the efficiency of GPO is more valuable. Thus, we prioritize GPO in this work to highlight the efficiency and scalability of our approach.

Table 7. Universality of $\mathbf { I N } ^ { 2 } \mathbf { R }$ across different backbones on Flickr30K (60% noise). While $\mathrm { I N } ^ { 2 } \mathrm { R }$ boosts the performance of SGRAF, we default to GPO to balance accuracy with significantly lower training costs. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Text Retrieval</td><td colspan="3">Image Retrieval</td><td rowspan="2">Time/Epoch</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>IN2R (GPO)</td><td>75.1</td><td>94.0</td><td>97.2</td><td>56.5</td><td>82.9</td><td>89.5</td><td>3.5 min</td></tr><tr><td>IN2R (SGRAF)</td><td>75.8</td><td>94.6</td><td>97.3</td><td>57.4</td><td>83.1</td><td>89.5</td><td>30.0 min</td></tr></table>

# C. More Ablation Studies

Hyperparameter Sensitivity on Loss Weights. Since the impact of the neighbor count K and memory size M has been discussed in the main text, we focus here on the sensitivity of the loss balancing terms: the Intra-modal Constraint Weight $\lambda _ { i n t r a }$ and the Rectification Weight γ. Table 8 presents the results on Flickr30K (60% noise).

Impact of Intra-modal Weight $\lambda _ { i n t r a }$ . As shown in the left subtable, setting $\lambda _ { i n t r a } = 0 ( \mathrm { i . e }$ ., removing manifold stabilization) leads to a clear performance drop, verifying the necessity of geometric constraints. The performance peaks at $\lambda _ { i n t r a } = 0 . 5$ . Increasing it further (e.g., to 1.0) degrades the results, likely because an overly strong intra-modal constraint may dominate the optimization, interfering with the primary objective of inter-modal alignment.

Impact of Rectification $W e i g h t \gamma _ { \cdot }$ . Regarding γ, the model remains stable across a wide range [0.5, 1.5], demonstrating that our synthesized supervision is reliable and does not require delicate tuning to balance with the clean supervision.

Table 8. Sensitivity analysis of loss weights on Flickr30K (60% noise). We analyze the trade-off between intra-modal constraints $( \lambda _ { i n t r a } )$ and rectification strength $( \gamma )$ .   
(a) Varying Intra-modal Weight $\lambda _ { i n t r a }$ 

<table><tr><td> $\lambda_{intra}$ </td><td>Text-R@1</td><td>Image-R@1</td><td>rSum</td></tr><tr><td>0.0</td><td>68.7</td><td>52.4</td><td>469.1</td></tr><tr><td>0.1</td><td>71.5</td><td>53.2</td><td>478.0</td></tr><tr><td>0.5</td><td>75.1</td><td>56.5</td><td>495.2</td></tr><tr><td>1.0</td><td>73.2</td><td>54.8</td><td>488.3</td></tr></table>

(b) Varying Rectification Weight γ 

<table><tr><td>γ</td><td>Text-R@1</td><td>Image-R@1</td><td>rSum</td></tr><tr><td>0.1</td><td>70.5</td><td>52.0</td><td>472.3</td></tr><tr><td>0.5</td><td>73.2</td><td>55.1</td><td>488.0</td></tr><tr><td>1.0</td><td>75.1</td><td>56.5</td><td>495.2</td></tr><tr><td>1.5</td><td>74.5</td><td>56.2</td><td>493.8</td></tr></table>

Impact of Memory and Retrieval Mechanism. We investigate the design of the retrieval mechanism, specifically the necessity of the Cross-Model Decoupling strategy. In our proposed $\mathrm { I N ^ { 2 } R }$ , network $\mathcal { M } _ { A }$ retrieves neighbors from the memory queue of its peer $\mathcal { M } _ { B } \left( \mathcal { Q } _ { B } \right)$ . We compare this against a Self-Memory baseline, where each network queries its own history $( \mathcal { Q } _ { A } )$ . Table 9 demonstrates the results:

• Self-Memory: Relying on self-generated history leads to suboptimal performance (rSum 482.7). This suggests that without decoupling, the model tends to reinforce its own errors, leading to confirmation bias.   
• Cross-Memory (Ours): By leveraging the peer network’s consensus, our approach effectively breaks this self-reinforcing loop, improving rSum by 12.5 points.

![](images/dfc19ae31ea108091746020ff5c3f826f1f6f766fde673560c53accd80d82996.jpg)

<details>
<summary>scatter</summary>

| Category | Count |
| -------- | ----- |
| Image    | 120   |
| Text     | 120   |
</details>

Figure 4. t-SNE visualization of the learned feature embeddings on the Flickr30K test set. (Left) Our proposed IN2R: The feature distribution exhibits a more compact and structured manifold, indicating that our intra-modal geometric constraints successfully stabilized the feature space. (Right) Discrete Selection Baseline: The feature space appears more scattered and disordered. Compared to the discrete selection paradigm, IN2R achieves better visual-semantic alignment, where image (blue) and text (orange) features of similar semantics form tighter clusters.

• Queue Strategy: We also validate the Elitist Rolling Update (filtering low-confidence samples). Removing this filter (i.e., storing all samples) drops performance to 488.1, confirming that maintaining a high-confidence ”elite” memory is vital for pristine retrieval.

Table 9. Ablation of Memory Construction and Retrieval Strategy on Flickr30K (60% noise). ”Decoupling” indicates whether peer-memory is used. 

<table><tr><td>Memory Strategy</td><td>Decoupling</td><td>Text R@1</td><td>Image R@1</td><td>rSum</td></tr><tr><td>Self-Memory (Standard)</td><td></td><td>72.5</td><td>53.8</td><td>482.7</td></tr><tr><td>Full Queue (No Filtering)</td><td>√</td><td>73.4</td><td>54.9</td><td>488.1</td></tr><tr><td>Elitist Cross-Memory (Ours)</td><td>√</td><td>75.1</td><td>56.5</td><td>495.2</td></tr></table>

# C.1. Qualitative Analysis of Feature Manifolds.

To intuitively verify the impact of our proposed method on the feature space, we visualize the learned image and text embeddings on the Flickr30K test set using t-SNE. Figure 4 provides a side-by-side comparison between our $\mathrm { I N ^ { 2 } R }$ (Left) and the baseline trained with discrete selection (Right). As observed on the right side of Figure 4, the feature space learned by the discrete selection paradigm appears relatively dispersed, with blurred boundaries between semantic clusters. This visual scattering corroborates our hypothesis that assigning discrete, noisy proxies introduces discretization error, preventing the model from learning a sharp semantic structure. In contrast, the manifold learned by $\mathrm { I N ^ { 2 } R }$ on the left side exhibits significantly higher intra-class compactness and inter-class separability. The image (blue) and text (orange) features form tighter, more distinct clusters, indicating a superior cross-modal alignment. This structural improvement demonstrates that our Graph-Guided Continuous Rectification effectively filters out feature-level noise and synthesizes reliable supervision targets, thereby regularizing the manifold towards a more discriminative geometric structure.