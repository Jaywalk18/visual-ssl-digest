# From Extrinsic to Intrinsic: Geodesic-Guided Representation Learning for 3D Geometric Data

Yuming Zhao 1 Junhui Hou 1 Qijian Zhang 2 Jia Qin 3 Ying He 4

# Abstract

Geometric analysis fundamentally distinguishes between extrinsic and intrinsic perspectives. The dominant paradigm in current 3D representation learning relies on either extrinsic spatial structures or high-level semantics, struggling to capture the essence of shape identity and underlying manifold topology. To bridge this gap, we introduce a novel 3D representation learning paradigm, namely PRISM, for Pre-training, which learns isometric embeddings by Recovering the Intrinsic Surface geodesic Metric. PRISM incorporates a topology-enforcing objective that explicitly constrains the structure of latent space, alongside a specialized two-stage training recipe mitigating sample imbalance inherent in the distribution of geodesic distances. Experiments demonstrate that our approach shows satisfactory accuracy, robustness, and high efficiency in geodesic distance prediction and achieves superior performance across diverse downstream tasks, including shape recognition, surface parameterization, and non-rigid correspondence. The code will be publicly available at https://github.com/ AidenZhao/PRISM.

# 1. Introduction

Self-supervised representation learning has emerged as a pivotal research direction in processing and understanding 3D geometric data. Developing powerful backbone feature extractors through pre-training is a critical prerequisite for building scalable foundation models and achieving superior generalization across diverse downstream applications.

1Department of Computer Science, City University of Hong Kong, Hong Kong, China 2Bambu Lab, Shenzhen, China 3Meshy AI, California, USA 4College of Computing and Data Science, Nanyang Technological University, Singapore. Correspondence to: Junhui Hou <jh.hou@cityu.edu.hk>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

In recent years, a rich variety of 3D pre-training frameworks have been investigated to continuously push the boundary of representation quality and transferability, built upon contrastive learning (Xie et al., 2020), masked modeling (Wang et al., 2021; Pang et al., 2022), cross-domain interaction (Afham et al., 2022; Zhang & Hou, 2024), etc. Generally, existing pre-training objectives are driven by either extrinsic 3D spatial structures and/or high-level semantic information, producing superior performance on downstream tasks such as classification and segmentation. However, due to the lack of modeling and learning of intrinsic geometric properties, existing frameworks still struggle to capture the essence of shape identity, understand the underlying manifold topology, and produce high-quality fine-grained geometric features. Consequently, these approaches typically suffer from suboptimal performance when evaluated on geometry-sensitive tasks that are grounded in intrinsic manifold structures or require robustness against deformations and pose variations, such as surface parameterization and non-rigid shape correspondence.

As a fundamental Riemannian metric, geodesic distance serves as the definitive intrinsic characterization of geometry on 3D curved surfaces, remaining invariant under isometric deformations. Accurately and efficiently computing geodesic distances is a central and long-standing problem in the field of computational geometry. While conventional geometry processing techniques deliver high precision and provide theoretical guarantees, they typically suffer from prohibitive computational bottlenecks. More recently, learningbased approaches have emerged as a promising alternative, offering substantially faster query speeds as well as greater flexibility. Despite the remarkable advances, existing approaches are still faced with obvious limitations in terms of robustness (GeGNN (Pang et al., 2023)) and cumbersome pre-computation (NeuroGF (Zhang et al., 2023)).

Drawing inspiration from Nash Embedding Theorem (Nash, 1956) (i.e., any Riemannian manifold can be isometrically embedded into a higher-dimensional Euclidean space while preserving geodesic distances), we introduce geodesic distance prediction as a core pretext task for 3D geometric data pre-training. Our objective is to learn a latent feature space that approximates isometry with respect to the original intrinsic geometry, such that the backbone feature extractor is enforced to capture both intrinsic manifold structure and fine-grained geometric details.

Technically, we employ a powerful point transformer (Wu et al., 2024) architecture with high efficiency and scalability as the target backbone. For geodesic-guided representation learning, we resort to a two-stage workflow that combines a pre-training stage constrained with geodesic structure consistency and a fine-tuning stage introducing importance sampling to mitigate sample imbalance. The pre-trained model achieves high-precision geodesic distance prediction with only 3D point clouds as input.

Empirically, we evaluate the effectiveness of our proposed geodesic-guided 3D pre-training framework by performing task-specific fine-tuning. For high-level semantic-oriented tasks, we conduct shape classification and part segmentation. For fine geometric details-centric tasks, we conduct fixedboundary surface parameterization and non-rigid shape correspondence. Extensive experiments demonstrate that our approach consistently delivers competitive performance.

In essence, the main contributions of this work can be summarized as follows.

• We propose a novel 3D representation learning paradigm guided by intrinsic geodesic properties for learning robust and fine-grained geometric features.   
• Our framework serves as a high-precision geodesic distance predictor with unstructured point clouds as input.   
• Particularly, thanks to intrinsic geometry pre-training, we achieve the first successful training of a feed-forward surface parameterization model.

# 2. Related Work

# 2.1. Representation Learning of 3D Geometry

Contrastive Learning. The essential goal of contrastive learning approaches is to measure the feature-level similarity and dissimilarity between positive and negative samples. PointContrast (Xie et al., 2020) constructs two point clouds from different perspectives and performs pre-training by measuring the feature similarity between corresponding points. Subsequent studies further investigate more aggressive data augmentation strategies (Wu et al., 2023; Zhang et al., 2021), self-distillation (Wu et al., 2025), cross-modal interaction (Afham et al., 2022; Zhang & Hou, 2024; Zhang et al., 2025a), etc.

Masked Modeling. Inspired by the success of masked 2D image autoencoders (He et al., 2022), a variety of adaptations are explored for 3D data pre-training. The representative works of Point-BERT (Yu et al., 2022) and Point-MAE (Pang et al., 2022) perform masked reconstruction on point clouds. TAP (Wang et al., 2023) and Ponder (Huang et al., 2023) drive pre-training by generating 2D projections of 3D point clouds. Point-M2AE (Zhang et al., 2022) introduces a hierarchical architecture progressively modeling geometric and feature information from local to global scales. Joint-MAE (Guo et al., 2023) designs joint encoders and decoders between 2D image and 3D point cloud modalities.

In addition, there also emerge more novel self-supervision paradigms such as adopting denoising mechanisms (Zheng et al., 2024; Zhang et al., 2025b; Chen et al., 2025).

# 2.2. Geodesics Computation

Computing geodesic distances and paths on 3D surfaces has been extensively studied for several decades (Mitchell et al., 1987), resulting in a large body of literature (Bose et al., 2011; Crane et al., 2020). Existing methods can be broadly categorized into two categories: traditional methods and learning-based methods.

Traditional Methods. Classical exact algorithms compute polyhedral geodesics by propagating continuous wavefronts, commonly known as continuous Dijkstra methods (Mitchell et al., 1987; Chen & Han, 1990; Xin et al., 2012; Surazhsky et al., 2005; Ying et al., 2014; Xu et al., 2015; Qin et al., 2016; Sharp & Crane, 2020). These methods are generally robust and capable of handling low-quality meshes, but they are computationally expensive. In applications that require frequent point-to-point queries, preprocessing techniques such as geodesic graphs or Steiner-point constructions (Ying et al., 2013; Adikusuma et al., 2020; Meng et al., 2022) are commonly used to spread out the computational cost. PDEbased techniques compute distance fields by solving or approximating the Eikonal equation. Representative methods include fast marching on triangulated domains (Kimmel & Sethian, 1998; Sethian & Vladimirsky, 2000) and heat-flowbased approaches (Crane et al., 2013; Sharp et al., 2019; Tao et al., 2021). PDE methods typically provide only firstorder approximations of geodesic distances and often rely on high-resolution and high-quality input data to achieve satisfactory accuracy.

Learning-based Methods. More recently, learning-based geodesic computation approaches are increasingly revealing new potential. NeuroGF (Zhang et al., 2023) introduces a deep implicit neural geodesic field that encodes the geodesic structure of a given 3D shape, enabling fast inference-time queries of arbitrary point-to-point geodesic distances and shortest paths, together with extensions to generalizable frameworks with learned 3D shape encoders. GeGNN (Pang et al., 2023) employs a graph neural network that maps mesh vertices into the high-dimensional embedding space, and answers distance queries by applying a lightweight decoder to the embeddings of queried points. More recently, LiteGE (Adikusuma et al., 2026) challenges the need for costly shape encoders. Instead, it canonicalizes input shapes and builds compact, category-aware descriptors by applying PCA to unsigned distance field samples at informative voxels and then predicts geodesic distances using a small inference network.

![](images/9b21cfb19129a4b18039c36828b1b800277a24710b5f67b27d061fce49dcae6a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Coordinate"] --> B["Geodesic Embedding Encoder"]
    C["Normal"] --> B
    B --> D["Point Transformer V3"]
    D --> E["Point-wise Feature"]
    E --> F["Metric Decoder"]
    F --> G["MLPs"]
```
</details>

![](images/f5a73fde655c612bfebc42ed5992eaf65f9b1ccd95425590b8ac1890cb1c3f27.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Training Objectives"] --> B["Geodesic Structure"]
    B --> C["Predicted Geodesic"]
    B --> D["GT Geodesic"]
    C --> E["MRE & L1 Loss"]
    D --> E
```
</details>

![](images/d6bc32b0dd4d932bf49802d3c481b4db4169f2047942ae892f04277a8f9a19f2.jpg)

<details>
<summary>text_image</summary>

Fine Geometric Details
Fix-Boundary Parameterization
Shape Correspondence
</details>

![](images/4e3d5124cf8c7b4628555e92d8776d2410d19fde85bed801cb730a8bd17bd06f.jpg)

<details>
<summary>text_image</summary>

High-Level Semantics
"Sofa"
Classification
Part Segmentation
</details>

Figure 1. The overview of PRISM, including an intrinsic geometry-aware foundation model and a geodesic-driven training objective composed of geodesic structure and prediction. Our PRISM effectively facilitates downstream tasks that focus on fine geometric details and high-level semantics.

# 3. Proposed Method

We introduce a novel 3D representation learning paradigm, namely PRISM, for Pre-training, which learns isometric embeddings by Recovering the Intrinsic Surface geodesic Metric. An overview of PRISM, is shown in Figure 1.

# 3.1. Motivation

Geodesic Embedding. The fundamental goal of representation learning on 3D surfaces is to find a mapping that preserves geometric information. The Nash Embedding Theorem (Nash, 1956) provides a rigorous theoretical foundation for this pursuit. It states that every Riemannian manifold can be isometrically embedded into a Euclidean space of sufficiently high dimension.

Inspired by this theorem, we seek to learn a non-linear mapping $\Phi : \mathcal { M }  \mathbb { R } ^ { k }$ that embeds the input manifold M (represented by a 3D point cloud) into a high-dimensional feature space $\mathbb { R } ^ { k }$ . Our objective is to ensure that this embedding is approximately isometric, meaning the metric structure of the feature space reflects the intrinsic metric of the input surface.

Geodesic Distance as the Intrinsic Metric. A 3D point cloud P is often treated as a set of coordinates in $\mathbb { R } ^ { 3 }$ . However, these coordinates usually describe the extrinsic geometry. The essence of the shape lies in its intrinsic geometry properties that depend only on the surface itself, not on how it is folded or bent in space.

The geodesic distance $d _ { G } ( \cdot , \cdot )$ is the natural realization of the Riemannian metric on the manifold. Unlike the classic Euclidean distance $d _ { E } ( \cdot , \cdot )$ which cuts through the ambient space, $d _ { G }$ measures the shortest path along the surface. By training our model to predict geodesic distances, we explicitly force the learned feature space to respect the underlying manifold structure. This serves as a robust foundation for fine-grained geometric processing tasks, where understanding topological connectivity is crucial.

# 3.2. Network Architecture

We employ Point Transformer V3 (PTv3) (Wu et al., 2024) as our feature representation backbone. PTv3 is chosen for its efficiency in processing large-scale point clouds and its ability to capture both local geometric details and global context through self-attention mechanisms. Given an input point cloud $\bar { \mathcal { P } } \in \mathbb { R } ^ { N \times 3 }$ of N points, the backbone maps each point $p _ { i }$ to a high-dimensional feature vector $\mathbf { f } _ { i } \in \mathbb { R } ^ { k }$ .

$$
\mathbf {F} = \mathrm{PTv3} (\mathcal {P}), \quad \text { where } \mathbf {F} = \{\mathbf {f} _ {1}, \dots , \mathbf {f} _ {N} \}. \tag {1}
$$

Geodesic Prediction Head. We designed a metric decoder to represent the metric between two points on $\mathbb { R } ^ { k }$ , constraining it to be as close as possible to the geodesic distance between the two points on M. To predict the intrinsic distance between any pair of points $( \mathbf { p } _ { i } , \mathbf { p } _ { j } )$ , we employ a symmetric predictor that operates on the feature difference. Specifically, we compute the absolute difference between their feature vectors:

$$
\mathbf {h} _ {i j} = \left| \mathbf {f} _ {i} - \mathbf {f} _ {j} \right| \in \mathbb {R} ^ {k}. \tag {2}
$$

This difference vector is then fed into a 3-layer Multi-Layer Perceptron (MLP) that maps the feature difference to a scalar value $R _ { i j }$ , representing the estimated geodesic distance, formulated as:

$$
\hat {d} _ {i j} = \mathrm{MLP} (\mathbf {h} _ {i j}). \tag {3}
$$

Using the absolute difference ensures the prediction is symmetric, i.e., $\hat { d } _ { i j } = \hat { d } _ { j i }$ i, respecting the symmetry of the distance metric.

# 3.3. Pre-training Objective

Geodesic Regression Loss $( \mathcal { L } _ { L 1 } )$ . First, we apply a standard L1 loss to directly minimize the absolute error between the predicted and the ground truth geodesic distances:

$$
\mathcal {L} _ {L 1} = \frac {1}{| \mathcal {S} |} \sum_ {(i, j) \in \mathcal {S}} | \hat {d} _ {i j} - d _ {G} (p _ {i}, p _ {j}) |, \tag {4}
$$

which ensures that the model learns the correct scale of the manifold.

Mean Relative Error Loss $( \mathcal { L } _ { M R E } )$ . Since geodesic distances can vary significantly in magnitude (from local neighborhoods to global extrema), an L1 loss may be dominated by large distances. To ensure the model captures fine-grained local geometry as well as global structure, we incorporate a relative error term:

$$
\mathcal {L} _ {M R E} = \frac {1}{| \mathcal {S} |} \sum_ {(i, j) \in \mathcal {S}} \frac {\left| \hat {d} _ {i j} - d _ {G} \left(p _ {i} , p _ {j}\right) \right|}{d _ {G} \left(p _ {i} , p _ {j}\right) + \epsilon}, \tag {5}
$$

where ϵ is a small constant for numerical stability.

Geodesic Structure Consistency Loss $( \mathcal { L } _ { s t r u c t } )$ . Beyond regressing exact values, we explicitly constrain the structure of the latent space to reflect the relative order of distances on the manifold. We employ a continuous ordinal loss to align the pairwise feature distances with the geodesic distances. Specifically, for two pairs of points $( i , j )$ and $( u , v )$ , we define the geodesic difference $\Delta d _ { G } =$ $d _ { G } ( p _ { i } , p _ { j } ) - d _ { G } ( p _ { u } , p _ { v } )$ and the feature distance difference $\Delta d _ { \Phi } = \| { \bf f } _ { i } - { \bf f } _ { j } \| - \| { \bf f } _ { u } - { \bf f } _ { v } \|$ .

This loss enforces that the sign of the feature difference matches the sign of the corresponding geodesic difference. To ensure differentiability, we approximate the sign function using a scaled hyperbolic tangent (tanh). The loss is formulated as the L1 distance between the true sign of geodesic differences and the soft sign of feature differences:

$$
\mathcal {L} _ {\text { struct }} = \frac {1}{| \mathcal {Q} |} \sum_ {((i, j), (u, v)) \in \mathcal {Q}} | \operatorname{sgn} (\Delta d _ {G}) - \tanh (\alpha \cdot \Delta d _ {\Phi}) |, \tag {6}
$$

where α is a scaling factor that controls the steepness of the approximation, acting as a soft-sign function. This ensures that if pair $( i , j )$ is geodesically closer than pair $( u , v )$ , their representations in feature space will also be closer, preserving the ordinal structure of the manifold.

# 3.4. Training Strategy

To enhance convergence speed and address the imbalance in geodesic distance distributions, we propose a two-stage workflow: (1) geodesic structure warm-up and (2) importance sampling fine-tuning.

Geodesic Structure Warm-Up. In the initial training phase, the primary goal is to rapidly guide the model towards a topologically meaningful feature space. To this end, we incorporate the geodesic structure consistency loss $( \mathcal { L } _ { s t r u c t } )$ with a time-decaying weight. By placing a heavy emphasis on structural consistency early on, we enforce a global ordering constraint that prevents the model from collapsing into local minima driven solely by distance regression. We define a dynamic weight $\lambda _ { 3 } ( t )$ for $\mathcal { L } _ { s t r u c t }$ that decreases over epochs:

$$
\lambda_ {3} (t) = \lambda_ {i n i t} \cdot \left(1 - \frac {t}{T _ {w a r m u p}}\right), \tag {7}
$$

where t is the current epoch and $T _ { w a r m u p }$ is the duration of the warm-up phase. This strategy accelerates convergence by first establishing the correct ”shape” of the latent manifold before refining the precise metric values.

In all, the overall training objective can be formulated as:

$$
\mathcal {L} = \lambda_ {1} \cdot \mathcal {L} _ {L 1} + \lambda_ {2} \cdot \mathcal {L} _ {M R E} + \lambda_ {3} (t) \cdot \mathcal {L} _ {\text { struct }}. \tag {8}
$$

Importance Sampling Fine-Tuning. As illustrated in Fig. 2, the distribution of geodesic distances between paired points in 3D shapes typically follows a normal-like distribution, where mid-range distances are abundant, but shortrange and long-range distances are rare. Hence, standard uniform sampling causes the model to underfit these distant pairs, leading to inaccurate global feature representations. To mitigate this, we introduce an Importance Sampling Fine-Tuning phase. We pre-compute the empirical probability density function $P ( d )$ of the geodesic distances in the training set. During finetuning, we sample point pairs $( i , j )$ with probability inversely proportional to their occurrence:

![](images/05db033f184edd293290ea92710ae0de32dbea090226fe688ccebd54eea56f28.jpg)

<details>
<summary>histogram</summary>

| Geodesic Distance Range | Proportion |
| ------------------------ | ---------- |
| 0.0 - 0.1                | 0.001      |
| 0.1 - 0.2                | 0.005      |
| 0.2 - 0.3                | 0.010      |
| 0.3 - 0.4                | 0.015      |
| 0.4 - 0.5                | 0.020      |
| 0.5 - 0.6                | 0.025      |
| 0.6 - 0.7                | 0.030      |
| 0.7 - 0.8                | 0.035      |
| 0.8 - 0.9                | 0.038      |
| 0.9 - 1.0                | 0.040      |
| 1.0 - 1.1                | 0.039      |
| 1.1 - 1.2                | 0.037      |
| 1.2 - 1.3                | 0.035      |
| 1.3 - 1.4                | 0.032      |
| 1.4 - 1.5                | 0.028      |
| 1.5 - 1.6                | 0.024      |
| 1.6 - 1.7                | 0.020      |
| 1.7 - 1.8                | 0.016      |
| 1.8 - 1.9                | 0.012      |
| 1.9 - 2.0                | 0.008      |
| 2.0 - 2.1                | 0.005      |
| 2.1 - 2.2                | 0.003      |
| 2.2 - 2.3                | 0.002      |
| 2.3 - 2.4                | 0.001      |
| 2.4 - 2.5                | 0.001      |
| 2.5 - 2.6                | 0.001      |
| 2.6 - 2.7                | 0.001      |
| 2.7 - 2.8                | 0.001      |
| 2.8 - 2.9                | 0.001      |
| 2.9 - 3.0                | 0.001      |
</details>

Figure 2. The distribution of geodesic distance values.

$$
w _ {s a m p l e} \propto \frac {1}{P (d _ {G} (p _ {i} , p _ {j}))}. \tag {9}
$$

Table 1. Quantitative comparison of geodesic distances estimated by different methods. For all metrics, the smaller, the better. The best results are highlighted in bold. Note that [S/M/L] NeuroGF is per-scene overfitting, where for each 3D shape, a portion of ground-truth geodesic distances have to be pre-computed to train the network. 

<table><tr><td>Method</td><td>Input</td><td>MRE (%)</td><td>L1 (%)</td><td>Time (s)</td></tr><tr><td colspan="5">Traditional Methods</td></tr><tr><td>HM</td><td>Mesh</td><td>2.52</td><td>2.71</td><td>79.7</td></tr><tr><td>DGG</td><td>Mesh</td><td>0.25</td><td>0.27</td><td>37.8</td></tr><tr><td>EEM</td><td>Mesh</td><td>8.73</td><td>8.41</td><td>45.7</td></tr><tr><td>FPGDC</td><td>Mesh</td><td>2.49</td><td>2.31</td><td>12.1</td></tr><tr><td colspan="5">Learning-based Methods</td></tr><tr><td>LiteGE</td><td>10.81</td><td>3.91</td><td>0.3</td><td></td></tr><tr><td>[O] GeGNN</td><td>Mesh</td><td>23.2</td><td>20.8</td><td>1.4</td></tr><tr><td>[R] GeGNN</td><td>Mesh</td><td>4.43</td><td>3.26</td><td>18.9</td></tr><tr><td>[S] NeuroGF</td><td>Points</td><td>3.12</td><td>2.93</td><td>60</td></tr><tr><td>[M] NeuroGF</td><td>Points</td><td>1.84</td><td>1.69</td><td>120</td></tr><tr><td>[L] NeuroGF</td><td>Points</td><td>0.52</td><td>0.48</td><td>600</td></tr><tr><td>Ours</td><td>Points</td><td>3.87</td><td>2.75</td><td>0.5</td></tr></table>

In this phase, we disable the structure loss (setting $\lambda _ { 3 } = 0 )$ and focus exclusively on the regression objectives $( \mathcal { L } _ { L 1 }$ and $\mathcal { L } _ { M R E } )$ . By prioritizing rare, long-range distances, we finetune the model to achieve high precision across the entire spectrum of the manifold.

# 4. Experiments

# 4.1. Implementation Details

We collected a refined subset of the classic ShapeNet (Chang et al., 2015) repository in the self-supervised pre-training phase, consisting of 15,000 training samples and 2,000 testing samples. To ensure high-quality geometric supervision, all the original mesh models were pre-processed through watertight manifold reconstruction and uniform remeshing, then normalized into a unit sphere. We applied the classic MMP (Mitchell et al., 1987) algorithm to produce groundtruth geodesics. For each training shape, we randomly sampled 500 source points to deduce geodesic distance fields.

The overall pre-training process is composed of two stages. In the first 1,000 epochs, we performed the Geodesic Structure Warm-Up. Then we performed fine-tuning with Importance Sampling for the subsequent 200 epochs. We adopted the AdamW optimizer with an initial learning rate of 1e-4, which decays to 1e-6 following a Cosine Annealing schedule. Our model is trained on 8 NVIDIA H100 GPUs with the total batch size of 64.

# 4.2. Geodesic Distance Prediction

To evaluate our performance of geodesic distance prediction, we made comparisons with traditional computational approaches (HM (Crane et al., 2013), DGG (Adikusuma et al., 2020), EEM (Panozzo et al., 2013), FPGDC (Shamai et al., 2018)) and state-of-the-art learning-based frameworks (GeGNN (Pang et al., 2023), NeuroGF (Zhang et al., 2023)). We quantitatively measured the metrics of Mean Relative Error (MRE), L1 Error, and average time cost (2000 queries per testing shape model). Considering the specific characteristics of the two learning-based competitors, we adopted different evaluation protocols. Given the sensitivity of GNNs to mesh density, we evaluated GeGNN on both our original testing data ([O] GeGNN) and a separate version remeshed to match the density used in its paper ([R] GeGNN). As NeuroGF relies on per-scene overfitting, the duration of the training stage significantly impacts performance. Therefore, we configured three training durations for comparison: short (1 minute, [S] NeuroGF), medium (2 minutes, [M] NeuroGF), and long (10 minutes, [L] NeuroGF). We reported quantitative comparison results in Table 1.

![](images/ddd6140543a312d0cbdbb7e5f83ef469050f9e387a374ca223d54058c950660c.jpg)

<details>
<summary>line</summary>

| Epoch | w/ Geo. Struct. Consis. | w/o Geo. Struct. Consis. |
|-------|--------------------------|---------------------------|
| 0     | 0.6                      | 0.6                       |
| 200   | 0.1                      | 0.15                      |
| 400   | 0.08                     | 0.1                       |
| 600   | 0.07                     | 0.09                      |
| 800   | 0.06                     | 0.08                      |
| 1000  | 0.05                     | 0.07                      |
</details>

(a) LMRE

![](images/1dd6c5b5004023682e68bd0de73e3a6082faf64cd53d46aa9868ea62b3df1693.jpg)

<details>
<summary>line</summary>

| Epoch | w/ Geo. Struct. Consis. | w/o Geo. Struct. Consis. |
|-------|--------------------------|---------------------------|
| 0     | 0.3                      | 0.3                       |
| 200   | 0.08                     | 0.09                      |
| 400   | 0.06                     | 0.07                      |
| 600   | 0.05                     | 0.06                      |
| 800   | 0.04                     | 0.05                      |
| 1000  | 0.03                     | 0.04                      |
</details>

(b) $\mathcal { L } _ { L 1 }$   
Figure 3. Ablation of Geodesic Structure Consistency results on LMRE and $\mathcal { L } _ { L 1 }$ .

Table 2. Ablation of importance sampling fine-tuning results on LMRE of different geodesic distance in (0, 1], (1, 2] and (2, 3] . 

<table><tr><td>Setting</td><td>(0, 1]</td><td>(1, 2]</td><td>(2, 3]</td></tr><tr><td>w/o Fine-tuning</td><td>5.1%</td><td>1.8%</td><td>2.9%</td></tr><tr><td>w/ Fine-tuning</td><td>3.9%</td><td>1.7%</td><td>2.4%</td></tr></table>

Traditional computation- and optimization-based methods can usually achieve the highest accuracy and demonstrate the most stable and reliable performance across various scenarios, but they are slower, limited by pre-computation time and relatively slow geodesic query speed. Among learningbased methods, GeGNN is quite sensitive to mesh density due to its GNN architecture. On our dataset, GeGNN almost completely fails, with MRE reaching 23.2%. However, when we remesh the test data to the same resolution as the training data, GeGNN’s performance can recover to a normal level. In contrast, NeuroGF, as a neural field method that performs per-scene overfitting for individual models, requires longer training time, but its prediction accuracy can gradually approach extremely high levels comparable to traditional methods. Compared with other methods, our approach achieves high accuracy and efficient geodesic distance prediction. Fig. 5 shows a comparison between the geodesic lines predicted by our method and the ground truth, as well as the visualization of the obtained features after t-SNE dimensionality reduction. Our method was trained only on object shapes from ShapeNet, yet it still exhibits a certain degree of generalization capability to inputs of scene type.

![](images/6b37f2fda6359a845086c9875dc563c7aaf6f09492990869c52e604c29406aa0.jpg)  
Geodesic Distance 0 1 2 3

Figure 4. Visual results on ablation of importance sampling finetuning. (a) w/ importance sampling fine-tuning, (b) w/o importance sampling fine-tuning.   
Table 3. Comparison of different feature dimensions of our backbone on geodesic distance estimation. 

<table><tr><td>Setting</td><td># Params.</td><td>k</td><td>GFLOPs</td><td>MRE (%)</td><td>L1 (%)</td></tr><tr><td>Small</td><td>31M</td><td>256</td><td>82.87</td><td>6.5</td><td>5.2</td></tr><tr><td>Base</td><td>124M</td><td>512</td><td>328.95</td><td>4.4</td><td>3.2</td></tr><tr><td>Large</td><td>494M</td><td>768</td><td>1295.81</td><td>3.9</td><td>2.8</td></tr></table>

In Figure 7, we present a set of results featuring nonwatertight and noisy inputs. As illustrated, our method demonstrates strong robustness against both topological defects and sensor noise. In Figure 8, we show a heatmap of geodesic distance and feature Euclidean distance between 50 points on Bunny. It can be seen that the distribution and scale ordinary between geodesics and features are similar.

# 4.3. Ablation Study

To evaluate the necessity and effectiveness of our core designs, we conducted ablation studies on three components: geodesic structure consistency, importance sampling finetuning, and different feature dimensions.

![](images/dd06229e94116f8474e1cceb1e8bdd2c10a017944fb7461ed2fa5bc0ee202fc4.jpg)  
Geodesic Distance 0 2 3

Figure 5. Visualization of geodesic prediction results by our methods. (a) Ground Truth by MMP (Mitchell et al., 1987), (b) Geodesic prediction by our method, (c) Point-wise feature visualization by t-SNE. It can be observed that the point-wise features exhibit a structure aligned with the geodesic distance, as demonstrated by the t-SNE dimensionality reduction visualization.

Geodesic Structure Consistency. We targetedly observed the training curves of MRE and L1 error under the same pretraining setup with and without geodesic structure warm-up. As shown in Figure 3, its integration effectively accelerates the decrease of both MRE and L1 error.

Importance Sampling Fine-Tuning. We compared the distribution curves of MRE across different geodesic distance lengths before and after applying importance sampling finetuning. As shown in Table 2 and Fig. 4, the accuracy for both short and long geodesic distances can greatly improve after such fine-tuning, effectively mitigating the prediction bias caused by the natural imbalance in the distance distribution.

Feature Dimension. Regarding scaling, we evaluated three models of different feature dimensions. As shown in Table 3, increasing the model capacity steadily improves geodesic distance prediction accuracy.

Geodesic Prediction Head. In Figure 9, we present a comparative analysis of training dynamics. By comparing our method with a variant that calculates feature-wise Euclidean distance without an MLP, it becomes evident that the latter suffers from convergence issues, whereas our full model converges smoothly.

![](images/1a282ba370b13e4686eccdd9b282200babaf0beb5ff31ff8b8c42ef468f5e38e.jpg)

Figure 6. Visual comparison of fixed-boundary parameterization results by different methods. From left to right: Ours, BFF (Sawhney & Crane, 2017), and Flexpara (Zhao et al., 2025).   
![](images/bf873f755db3be83598112098ed609e304783bfc0f0c3d3224abf6cf15411c96.jpg)  
(a) Ground truth.

![](images/b0d706f7624184343328cc1b5ae2d3b3034fcf2f96a043f064beea7267d08c8e.jpg)

<details>
<summary>natural_image</summary>

3D rendered illustration of a snail with a color-coded stress or deformation visualization (no text or symbols)
</details>

(b) Non-manifold.

![](images/3e7403c1f9514204733972083f4c4d9e634da5c5e88747779cb3c3fdff339fdc.jpg)

<details>
<summary>natural_image</summary>

3D rendered snail with a color gradient from green to blue, no text or symbols present
</details>

(c) Noise.

Figure 7. Geodesic prediction on different input.   
![](images/0cfd0b5e3dd16cd27d714ff31283e3a8273108a1232ee24aa18a60dc2e168565.jpg)

<details>
<summary>heatmap</summary>

| Sorted Sample Index | Sorted Sample Index | Geodesic Distance |
|---------------------|---------------------|-------------------|
| 0                   | 0                   | High              |
| 10                  | 10                  | Medium            |
| 20                  | 20                  | Low               |
| 30                  | 30                  | Medium            |
| 40                  | 40                  | Low               |
</details>

![](images/aa24c92aac2ec7e75c183a895354852b52582e73cfafd7f84b9183cf1f44a6f8.jpg)  
Figure 8. Geodesic and feature euclidean distance heatmap.

# 4.4. Downstream Tasks

Generally, we focus on two categories of downstream task scenarios respectively dominated by fine geometric details and high-level semantics. For the former, we evaluated the novel fixed-boundary surface parameterization task and the challenging non-rigid shape correspondence task. For the latter, we evaluated the standard tasks of shape classification and object part segmentation.

Fixed-Boundary Surface Parameterization. Parameterization refers to mapping a 3D surface onto a 2D plane while

![](images/dad4a89b457883c43b45ccb3f99088008f20ee4a0ad7ca94690b68ce4354ebdb.jpg)

<details>
<summary>line</summary>

| Epoch | w/ MLP. | w/o MLP. |
|-------|---------|----------|
| 0     | 0.65    | 0.65     |
| 100   | 0.15    | 0.38     |
| 200   | 0.10    | 0.36     |
| 300   | 0.08    | 0.35     |
| 400   | 0.07    | 0.34     |
| 500   | 0.06    | 0.34     |
| 600   | 0.06    | 0.34     |
| 700   | 0.06    | 0.34     |
| 800   | 0.06    | 0.34     |
| 900   | 0.06    | 0.34     |
| 1000  | 0.06    | 0.34     |
</details>

Figure 9. Training loss curve on w/ MLP and w/o MLP.

Table 4. Quantitative comparison of fixed-boundary surface parameterization. The best results are highlighted in bold. Note that FlexPara is per-scene overfitting method, which needs a long time to train. “M+S”: Mesh + Seams. 

<table><tr><td>Method</td><td>Input</td><td>Error</td><td>Time</td></tr><tr><td>BFF (Sawhney &amp; Crane, 2017)</td><td>M+S</td><td>7.27</td><td>4.3</td></tr><tr><td>FlexPara (Zhao et al., 2025)</td><td>Points</td><td>4.45</td><td>423</td></tr><tr><td>Ours</td><td>Points</td><td>5.34</td><td>0.5</td></tr></table>

maintaining injectivity (one-to-one mapping) and low shape distortion. We made the first successful attempt to apply a pre-trained model to the fixed-boundary parameterization task and achieved the first feed-forward parameterization. It is worth noting that feed-forward parameterization is a highly challenging task. Even the state-of-the-art neural method FlexPara operates only in an overfitting framework and struggles to achieve global feed-forward parameterization. We experimented with a wide range of feed-forward neural network architectures as well as methods based on various pre-trained models, but none of them could effectively work on the parameterization task.

![](images/2569541602fd7fc1150a5f0acd9a750a0deedee0d1dc8a9ae7e97a6e0aecbd46.jpg)

<details>
<summary>text_image</summary>

FM
ZoomOut
ULRSM
SMS
Ours
FM
ZoomOut
ULRSM
SMS
Ours
</details>

Figure 10. Visual comparison of correspondence results by different methods. From left to right: FM (Ovsjanikov et al., 2012), ZoomOut (Melzi et al., 2019), ULRSM (Cao et al., 2023), SMS (Cao et al., 2024), and Ours.

Table 5. Quantitative comparison of non-rigid shape correspondence on FAUST. The best results are highlighted in bold. 

<table><tr><td>Method</td><td>Input</td><td>Error</td><td>Time(s)</td></tr><tr><td>FM (Ovsjanikov et al., 2012)</td><td>Mesh</td><td>21.2</td><td>3.2</td></tr><tr><td>ZoomOut (Melzi et al., 2019)</td><td>Mesh</td><td>6.3</td><td>7.4</td></tr><tr><td>ULRSM (Cao et al., 2023)</td><td>Mesh</td><td>1.6</td><td>1.7</td></tr><tr><td>SMS (Cao et al., 2024)</td><td>Mesh</td><td>1.4</td><td>0.5</td></tr><tr><td>Ours</td><td>Points</td><td>1.4</td><td>0.5</td></tr></table>

We made comparisons with the representative traditional geometry optimization method, BFF (Sawhney & Crane, 2017), and the representative neural network-based method, FlexPara (Zhao et al., 2025). Visual comparisons of parameterization results are shown in Figure 13. Our method, together with BFF and FlexPara, produces convincing parameterization outcomes. For quantitative evaluation, we use isometric loss to compare the results. The quantitative comparisons are presented in Table 4. Our method achieves an isometric loss better than BFF and slightly worse than FlexPara. However, thanks to the feed-forward mode, our method achieves considerably faster runtime. Moreover, traditional methods such as BFF require mesh input along with cut seams, whereas our method operates solely on unstructured point clouds.

Non-Rigid Shape Correspondence. We experimented with non-rigid 3D shape correspondence on the FAUST (Bogo et al., 2014) dataset. We made comparisons with traditional methods Functional Map (FM) (Ovsjanikov et al., 2012) and ZoomOut (Melzi et al., 2019), as well as learning-based methods ULRSM (Cao et al., 2023) and SMS (Cao et al., 2024). We used the first 80 pairs of the FAUST dataset for training and the last 20 pairs for testing. We reported the average geodesic loss and runtime in Table 5. Our method, using only point cloud as input, achieves the best geodesic distance error while achieving substantially faster runtime, since our approach does not need to compute the Laplace-Beltrami operator. Visualization results are shown in Figure 10.

Shape Classification. We experimented with real-scanned 3D object classification on the ScanObjectNN (Uy et al., 2019) benchmark dataset. We made comparisons with existing advanced 3D geometric pre-training frameworks, including OcCo (Wang et al., 2021), CrossPoint (Afham et al., 2022), Point-BERT (Yu et al., 2022), Point-MAE (Pang et al., 2022), Point-M2AE (Zhang et al., 2022), Point-Dif (Zheng et al., 2024), Point-DAE (Zhang et al., 2025b) and PointSD (Chen et al., 2025). To highlight the quality of pre-trained features, we chose to freeze the pre-trained backbone and train the classification head. As shown in Table 6, in the OBJ-BG and OBJ-ONLY settings, our method achieves the near best overall performance. In the PD-T50- RS setting, which includes random rotation data augmentation, our method also reaches the near best performance, indicating that our approach can more effectively capture intrinsic geometric features. To further verify rotation robustness, we set up the PB-T50-RS-Test-Only setting: using weights trained on OBJ-BG, and then testing directly on the PB-T50-RS data which includes rotation augmentation. The results demonstrate that our method exhibits significantly better rotation robustness. additionally, we combined the features from Point-MAE with those from our pre-trained model for a classification task, resulting in observable performance gains.

Table 6. Quantitative comparison of classification on ScanObjectNN. The best results are highlighted in bold. Note that OcCo and CrossPoint do not provide the results on OBJ-ONLY and PB-T50-RS. 

<table><tr><td>Method</td><td>OBJ-BG</td><td>OBJ-ONLY</td><td>PB-T50-RS</td><td>PB-T50-RS-Test-Only</td></tr><tr><td>OcCo (Wang et al., 2021)</td><td>84.5</td><td>-</td><td>-</td><td>54.1</td></tr><tr><td>CrossPoint (Afham et al., 2022)</td><td>86.2</td><td>-</td><td>-</td><td>55.7</td></tr><tr><td>Point-BERT (Yu et al., 2022)</td><td>88.1</td><td>87.4</td><td>83.1</td><td>53.7</td></tr><tr><td>Point-MAE (Pang et al., 2022)</td><td>90.0</td><td>88.3</td><td>85.2</td><td>55.2</td></tr><tr><td>Point-M2AE (Zhang et al., 2022)</td><td>91.2</td><td>88.8</td><td>86.5</td><td>58.1</td></tr><tr><td>PointDif (Zheng et al., 2024)</td><td>93.3</td><td>91.9</td><td>87.6</td><td>64.2</td></tr><tr><td>Point-DAE (Zhang et al., 2025b)</td><td>93.9</td><td>93.1</td><td>88.7</td><td>-</td></tr><tr><td>PointSD (Chen et al., 2025)</td><td>95.2</td><td>93.6</td><td>90.1</td><td>-</td></tr><tr><td>PTv3 from Scratch</td><td>89.9</td><td>88.2</td><td>84.9</td><td>-</td></tr><tr><td>Ours+Point-MAE</td><td>93.9</td><td>92.7</td><td>89.9</td><td>-</td></tr><tr><td>Ours</td><td>93.5</td><td>92.1</td><td>89.7</td><td>72.1</td></tr></table>

Table 7. Quantitative comparison of part segmentation on ShapeNetPart. The best results are highlighted in bold. Note that OcCo and CrossPoint do not provide standalone pretrained weights that are decoupled from downstream task heads, and PointDif does not include part segmentation fine-tuning results on ShapeNetPart in its original evaluation. 

<table><tr><td>Method</td><td>mIou Freeze</td><td>mIou Finetune</td></tr><tr><td>OcCo (Wang et al., 2021)</td><td>-</td><td>85.2</td></tr><tr><td>CrossPoint (Afham et al., 2022)</td><td>-</td><td>85.5</td></tr><tr><td>Point-Bert (Yu et al., 2022)</td><td>83.9</td><td>85.6</td></tr><tr><td>Point-MAE (Pang et al., 2022)</td><td>84.3</td><td>86.1</td></tr><tr><td>Point-M2AE (Zhang et al., 2022)</td><td>84.5</td><td>86.5</td></tr><tr><td>PointDif (Zheng et al., 2024)</td><td>84.5</td><td>-</td></tr><tr><td>Point-DAE (Zhang et al., 2025b)</td><td>-</td><td>86.4</td></tr><tr><td>PointSD (Chen et al., 2025)</td><td>-</td><td>86.1</td></tr><tr><td>Ours</td><td>85.1</td><td>86.5</td></tr></table>

Part Segmentation. We experimented with 3D object part segmentation on the ShapeNetPart (Yi et al., 2016) benchmark dataset under two evaluation protocols: full fine-tuning and training with backbone frozen. The results are shown in Table 7. Under full fine-tuning, our method achieves stateof-the-art performance compared to other methods. Under the frozen backbone setting, our method significantly outperforms other approaches, demonstrating that our pre-trained model provides higher-quality features.

# 5. Conclusion and Discussion

We proposed a novel 3D pre-training paradigm that prioritizes intrinsic geometric properties over extrinsic spatial structures. By defining the pretext task as Geodesic Distance Prediction and enforcing a Geodesic Structure Consistency loss, we isometrically embedded the Riemannian manifold structure into the high-dimensional feature space. We conducted comprehensive experiments to demonstrate that our approach not only achieves high-precision geodesic distance prediction but also shows superior performance on downstream tasks, including fixed-boundary surface parameterization, non-rigid shape correspondence, 3D shape classification and part segmentation.

Our in-depth explorations indicate a highly promising direction of intrinsic geometry learning. Although our method has achieved some progress in rotation robustness, it is still far from sufficient. In the future, we will explore more advanced intrinsic representation learning and attempt to investigate multi-task-driven 3D representation pre-training, further pushing the boundaries of geodesic-guided 3D representation learning.

# Acknowledgment

This work was supported in part by the NSFC under Grant 62422118, and in part by the Hong Kong RGC under Grants 11219324 and N CityU1114/25.

# Impact statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

# References

Adikusuma, Y. Y., Fang, Z., and He, Y. Fast construction of discrete geodesic graphs. ACM Trans. Graph., 39(2): 14:1–14:14, 2020.   
Adikusuma, Y. Y., Huang, Q., and He, Y. Litege: Lightweight geodesic embedding for efficient geodesics computation and non-isometric shape correspondence. In AAAI, 2026.   
Afham, M., Dissanayake, I., Dissanayake, D., Dharmasiri, A., Thilakarathna, K., and Rodrigo, R. Crosspoint: Selfsupervised cross-modal contrastive learning for 3d point cloud understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9902–9912, June 2022.   
Bogo, F., Romero, J., Loper, M., and Black, M. J. FAUST: Dataset and evaluation for 3D mesh registration. In Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Piscataway, NJ, USA, June 2014. IEEE.   
Bose, P., Maheshwari, A., Shu, C., and Wuhrer, S. A survey of geodesic paths on 3D surfaces. Computational Geometry, 44(9):486–498, 2011.   
Cao, D., Roetzer, P., and Bernard, F. Unsupervised learning of robust spectral shape matching. ACM Trans. Graph., 42(4), July 2023. ISSN 0730-0301. doi: 10.1145/3592107. URL https://doi.org/10. 1145/3592107.   
Cao, D., Eisenberger, M., El Amrani, N., Cremers, D., and Bernard, F. Spectral meets spatial: Harmonising 3d shape matching and interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3658–3668, June 2024.   
Chang, A. X., Funkhouser, T., Guibas, L., Hanrahan, P., Huang, Q., Li, Z., Savarese, S., Savva, M., Song, S., Su, H., et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.   
Chen, J. and Han, Y. Shortest paths on a polyhedron. In SoCG, pp. 360–369, 1990.   
Chen, Y., Zhao, S., Duan, L., Ding, C., and Tao, D. Harnessing text-to-image diffusion models for point cloud self-supervised learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 26156– 26166, 2025.   
Crane, K., Weischedel, C., and Wardetzky, M. Geodesics in heat: A new approach to computing distance based on heat flow. ACM Transactions on Graphics, 32(5), 2013. doi: 10.1145/2516971.2516977.

Crane, K., Livesu, M., Puppo, E., and Qin, Y. A survey of algorithms for geodesic paths and distances. CoRR, abs/2007.10430, 2020. URL https://arxiv.org/ abs/2007.10430.

Guo, Z., Zhang, R., Qiu, L., Li, X., and Heng, P.-A. Jointmae: 2d-3d joint masked autoencoders for 3d point cloud pre-training. arXiv preprint arXiv:2302.14007, 2023.

He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick, ´ R. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009, 2022.

Huang, D., Peng, S., He, T., Yang, H., Zhou, X., and Ouyang, W. Ponder: Point cloud pre-training via neural rendering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16089–16098, 2023.

Kimmel, R. and Sethian, J. A. Computing geodesic paths on manifolds. Proceedings of the National Academy of Sciences, 95(15):8431–8435, 1998. doi: 10.1073/pnas. 95.15.8431.

Liu, L., Ye, C., Ni, R., and Fu, X.-M. Progressive parameterizations. ACM Transactions on Graphics(SIGGRAPH), 37(4):41:1–41:12, 2018.

Melzi, S., Ren, J., Rodola, E., Sharma, A., Wonka, P., and Ovsjanikov, M. Zoomout: Spectral upsampling for efficient shape correspondence. arXiv preprint arXiv:1904.07865, 2019.

Meng, W., Xin, S., Tu, C., Chen, S., He, Y., and Wang, W. Geodesic tracks: Computing discrete geodesics with track-based steiner point propagation. IEEE Transactions on Visualization and Computer Graphics, 28(12):4887– 4901, 2022.

Mitchell, J. S., Mount, D. M., and Papadimitriou, C. H. The discrete geodesic problem. SICOMP, 16(4):647–668, 1987.

Nash, J. The imbedding problem for riemannian manifolds. Annals of mathematics, 63(1):20–63, 1956.

Ovsjanikov, M., Ben-Chen, M., Solomon, J., Butscher, A., and Guibas, L. Functional maps: a flexible representation of maps between shapes. ACM Transactions on Graphics (ToG), 31(4):1–11, 2012.

Pang, B., Zheng, Z., Wang, G., and Wang, P.-S. Learning the geodesic embedding with graph neural networks. ACM Transactions on Graphics (TOG), 42(6):1–12, 2023.

Pang, Y., Wang, W., Tay, F. E., Liu, W., Tian, Y., and Yuan, L. Masked autoencoders for point cloud self-supervised learning. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part II, pp. 604–621. Springer, 2022.

Panozzo, D., Baran, I., Diamanti, O., and Sorkine-Hornung, O. Weighted averages on surfaces. ACM Transactions on Graphics (TOG), 32(4):1–12, 2013.   
Qin, Y., Han, X., Yu, H., Yu, Y., and Zhang, J. Fast and exact discrete geodesic computation based on triangle-oriented wavefront propagation. ACM Trans. Graph., 35(4), 2016.   
Sawhney, R. and Crane, K. Boundary first flattening. ACM Transactions on Graphics (ToG), 37(1):1–14, 2017.   
Sethian, J. A. and Vladimirsky, A. Fast methods for the Eikonal and related Hamilton-Jacobi equations on unstructured meshes. Proceedings of National Academy of Sciences, 97:5699–5703, 2000.   
Shamai, G., Zibulevsky, M., and Kimmel, R. Efficient intergeodesic distance computation and fast classical scaling. IEEE transactions on pattern analysis and machine intelligence, 42(1):74–85, 2018.   
Sharp, N. and Crane, K. You can find geodesic paths in triangle meshes by just flipping edges. ACM Trans. Graph., 39(6):249:1–249:15, 2020.   
Sharp, N., Soliman, Y., and Crane, K. The vector heat method. ACM Trans. Graph., 38(3), June 2019. ISSN 0730-0301. doi: 10.1145/3243651. URL https:// doi.org/10.1145/3243651.   
Surazhsky, V., Surazhsky, T., Kirsanov, D., Gortler, S. J., and Hoppe, H. Fast exact and approximate geodesics on meshes. ACM Transactions on Graphics, 24(3):553–560, 2005. doi: 10.1145/1073204.1073228.   
Tao, J., Zhang, J., Deng, B., Fang, Z., Peng, Y., and He, Y. Parallel and scalable heat methods for geodesic distance computation. IEEE Trans. Pattern Anal. Mach. Intell., 43(2):579–594, February 2021. ISSN 0162-8828. doi: 10.1109/TPAMI.2019.2933209. URL https://doi. org/10.1109/TPAMI.2019.2933209.   
Uy, M. A., Pham, Q.-H., Hua, B.-S., Nguyen, T., and Yeung, S.-K. Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1588–1597, 2019.   
Wang, H., Liu, Q., Yue, X., Lasenby, J., and Kusner, M. J. Unsupervised point cloud pre-training via occlusion completion. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9782–9792, 2021.   
Wang, Z., Yu, X., Rao, Y., Zhou, J., and Lu, J. Take-a-photo: 3d-to-2d generative pre-training of point cloud models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5640–5650, 2023.

Wu, X., Wen, X., Liu, X., and Zhao, H. Masked scene contrast: A scalable framework for unsupervised 3d representation learning. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pp. 9415–9424, 2023.   
Wu, X., Jiang, L., Wang, P.-S., Liu, Z., Liu, X., Qiao, Y., Ouyang, W., He, T., and Zhao, H. Point transformer v3: Simpler faster stronger. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4840–4851, 2024.   
Wu, X., DeTone, D., Frost, D., Shen, T., Xie, C., Yang, N., Engel, J., Newcombe, R., Zhao, H., and Straub, J. Sonata: Self-supervised learning of reliable point representations. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22193–22204, 2025.   
Xie, S., Gu, J., Guo, D., Qi, C. R., Guibas, L., and Litany, O. Pointcontrast: Unsupervised pre-training for 3d point cloud understanding. In European conference on computer vision, pp. 574–591. Springer, 2020.   
Xin, S.-Q., He, Y., and Fu, C.-W. Efficiently computing exact geodesic loops within finite steps. IEEE Transactions on Visualization and Computer Graphics, 18(6):879–889, 2012.   
Xu, C.-X., Wang, T. Y., Liu, Y.-J., Liu, L., and He, Y. Fast wavefront propagation (FWP) for computing exact geodesic distances on meshes. IEEE Transactions on Visualization and Computer Graphics, 21(7):822–834, 2015.   
Yi, L., Kim, V. G., Ceylan, D., Shen, I.-C., Yan, M., Su, H., Lu, C., Huang, Q., Sheffer, A., and Guibas, L. A scalable active framework for region annotation in 3d shape collections. ACM Transactions on Graphics (ToG), 35(6):1–12, 2016.   
Ying, X., Wang, X., and He, Y. Saddle vertex graph (SVG): A novel solution to the discrete geodesic problem. ACM Trans. Graph., 32(6):170:1–12, 2013.   
Ying, X., Xin, S.-Q., and He, Y. Parallel Chen-Han (PCH) algorithm for discrete geodesics. ACM Transactions on Graphics, 33(1):9:1–9:11, 2014.   
Yu, X., Tang, L., Rao, Y., Huang, T., Zhou, J., and Lu, J. Point-bert: Pre-training 3d point cloud transformers with masked point modeling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 19313–19322, 2022.   
Zhang, Q. and Hou, J. Pointvst: Self-supervised pre-training for 3d point clouds via view-specific point-to-image translation. IEEE Transactions on Visualization and Computer Graphics, 30(10):6900–6912, 2024.

Zhang, Q., Hou, J., Adikusuma, Y., Wang, W., and He, Y. Neurogf: A neural representation for fast geodesic distance and path queries. Advances in Neural Information Processing Systems, 36:19485–19501, 2023.   
Zhang, R., Guo, Z., Gao, P., Fang, R., Zhao, B., Wang, D., Qiao, Y., and Li, H. Point-m2ae: multi-scale masked autoencoders for hierarchical point cloud pre-training. Advances in neural information processing systems, 35: 27061–27074, 2022.   
Zhang, Y., Hou, J., Ren, S., Wu, J., Yuan, Y., and Shi, G. Self-supervised learning of lidar 3d point clouds via 2d-3d neural calibration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025a.   
Zhang, Y., Lin, J., Li, R., Jia, K., and Zhang, L. Point-dae: Denoising autoencoders for self-supervised point cloud learning. IEEE Transactions on Neural Networks and Learning Systems, 2025b.   
Zhang, Z., Girdhar, R., Joulin, A., and Misra, I. Selfsupervised pretraining of 3d features on any point-cloud. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10252–10263, 2021.   
Zhao, Y., Zhang, Q., Hou, J., Xia, J., Wang, W., and He, Y. Flexpara: Flexible neural surface parameterization. IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–18, 2025. doi: 10.1109/TPAMI.2025. 3628727.   
Zheng, X., Huang, X., Mei, G., Hou, Y., Lyu, Z., Dai, B., Ouyang, W., and Gong, Y. Point cloud pre-training with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22935–22945, 2024.

# A. Model Structure

We employ PTv3 as the backbone and design three scaled versions (Small, Base, and Large) to investigate the scaling behavior of the proposed approach. The detailed configurations are summarized in Table 8. Following the PTv3 backbone, we attach a lightweight MLP regression head to predict a single positive scalar distance. The head takes the extracted feature vector as input and passes it through three successive linear layers. The first two layers are followed by ReLU activations and dropout regularization, while the final linear layer is succeeded by a Softplus activation.

Table 8. Architectural configurations of the Small, Base, and Large backbone variants. 

<table><tr><td>Component</td><td>Small</td><td>Base</td><td>Large</td></tr><tr><td rowspan="4">Encoder</td><td> $depths = (2,2,2,6,2)$ </td><td>(2,2,2,6,2)</td><td>(3,3,3,12,3)</td></tr><tr><td> $channels = (32,64,128,256)$ </td><td>(32,64,128,256,512)</td><td>(64,128,256,512,768)</td></tr><tr><td> $num\_heads = (2,4,8,16)$ </td><td>(2,4,8,16,32)</td><td>(4,8,16,32,48)</td></tr><tr><td> $patch\_size = 1024 (all)$ </td><td>1024 (all)</td><td>1024 (all)</td></tr><tr><td rowspan="4">Decoder</td><td> $depths = (2,2,2,2)$ </td><td>(2,2,2,2)</td><td>(3,3,3,3)</td></tr><tr><td> $channels = (256,256,256)$ </td><td>(512,512,512,512)</td><td>(768,768,768,768)</td></tr><tr><td> $num\_heads = (2,4,8)$ </td><td>(2,4,8,16)</td><td>(4,8,16,32)</td></tr><tr><td> $patch\_size = 1024 (all)$ </td><td>1024 (all)</td><td>1024 (all)</td></tr></table>

# B. Implementation of Downstream Tasks

# B.1. Fixed-Boundary Surface Parameterization

![](images/20345aa1d29dbc254d4bf30cd92cfcac559bc480da983dbcbc604cef6ae47950.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Coordinate"] --> B["Geodesic Embedding Encoder"]
    C["Normal"] --> B
    B --> D["Point Transformer V3"]
    D --> E["Point-wise Feature"]
    E --> F["Unwrapping-Net"]
    F --> G["Parameterization Result"]
    G --> H["Wrapping-Net"]
    H --> I["Reconstructed Point-wise Feature"]
```
</details>

Figure 11. The overall pipeline of our fixed-boundary surface parameterization framework.

In the fixed-boundary surface parameterization task, we adopt the unwrapping-wrapping architecture from FlexPara. The raw point cloud is first fed into our pre-trained model to obtain per-point features. These point-wise features are then passed through a lightweight unwrapping module to produce per-point UV coordinates. Subsequently, the UV coordinates are fed into a wrapping module to reconstruct high-dimensional per-point features. Both the unwrapping and wrapping modules are implemented as lightweight MLPs. The framework is shown in Figure 11.

Regarding the loss functions, following FlexPara, we employ a consistency loss to constrain the reconstruction quality after wrapping, and an isometric loss to regularize the deformation in the UV space. Additionally, since this is a fixed-boundary task, we introduce an extra Chamfer Distance (CD) loss ${ \mathcal { L } } _ { w }$ between the predicted UV shape and a regular grid to enforce boundary shape conformity.

$$
\mathcal {L} _ {c} = \left\| \mathbf {F} - \mathbf {F} _ {\text { recon }} \right\| _ {1} \tag {10}
$$

$$
\ell_ {\mathrm{iso}} = \sum_ {\Omega_ {\mathbf {P}} \in \mathcal {X}} \sum_ {i} \| \theta_ {i} - \beta_ {i} \| _ {1}. \tag {11}
$$

$$
\ell_ {\text { global }} = \alpha_ {w} \cdot \ell_ {w} + \alpha_ {c} \cdot \ell_ {c} + \alpha_ {\text { iso }} \cdot \ell_ {\text { iso }}, \tag {12}
$$

As a baseline method, we extend the global parameterization of FlexPara by incorporating the same UV-shape constraint via the CD loss between UV and the grid, while keeping all other settings and model architecture identical to FlexPara. The BFF approach naturally produces rectangular parameterization boundaries. Since it requires an input mesh along with corresponding cut seams, we utilize the pre-provided cut seams from the dataset in Progressive Parameterizations (Liu et al., 2018).

![](images/4657a71410c9ecc95d11665e7f3db583a794705de1fe2edfdd471fa84811da85.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Human Model"] --> B["Geodesic Embedding Encoder"]
    C["Human Model"] --> B
    B --> D["Point Transformer V3"]
    D --> E["Correspondence Head"]
    E --> F["PointNet"]
    F --> G["Human Model"]
    F --> H["Human Model"]
```
</details>

Figure 12. The overall pipeline of our 3D shape correspondence framework.

![](images/7dab674fed1db6302f9ee540812818203c3ca84313431185568c2b1555ae448f.jpg)

<details>
<summary>text_image</summary>

Ours
BFF
FlexPara
Ours
BFF
FlexPara
Ours
BFF
FlexPara
</details>

Figure 13. More visualization of fixed-boundary surface parameterization results produced by different approaches. From left to right: Ours, BFF and FlexPara.

# B.2. 3D Shape Correspondence

In the shape correspondence task, we utilize a simple PointNet as the decoder head. Experiments were conducted on the FAUST dataset, using the first 80 objects for training and the remaining 20 for testing. The framework is shown in Figure 12. The network takes a 3D point cloud as input and outputs per-point features. A similarity matrix C is computed between the per-point features of two shapes. The model is trained to minimize the discrepancy between C and the identity matrix I.

# B.3. 3D Shape Classification

We evaluate our approach on the ScanObjectNN dataset for the 3D object classification task. In our method, we employ a simple PointNet-style classification head on top of the pre-trained features.

To more fairly demonstrate the quality of features provided by our pre-trained model and to exclude interference from

Table 9. Quantitative comparison of fixed-boundary surface parameterization. The best results are highlighted in bold. 

<table><tr><td>Model</td><td>BFF</td><td>FlexPara</td><td>Ours</td></tr><tr><td>Input</td><td>Mesh + Seams</td><td>Points</td><td>Points</td></tr><tr><td>Bear1</td><td>15.16</td><td>7.20</td><td>5.33</td></tr><tr><td>Bear2</td><td>17.15</td><td>6.69</td><td>5.13</td></tr><tr><td>Fish</td><td>18.31</td><td>4.52</td><td>6.15</td></tr><tr><td>Head1</td><td>7.72</td><td>5.29</td><td>4.31</td></tr><tr><td>Head2</td><td>11.34</td><td>5.07</td><td>6.71</td></tr><tr><td>Head3</td><td>10.23</td><td>4.20</td><td>4.18</td></tr><tr><td>Box</td><td>24.38</td><td>4.97</td><td>4.15</td></tr><tr><td>Box2</td><td>18.76</td><td>5.99</td><td>3.78</td></tr><tr><td>Nail</td><td>7.43</td><td>6.23</td><td>4.11</td></tr><tr><td>Bird</td><td>13.48</td><td>6.41</td><td>5.62</td></tr><tr><td>Tongs</td><td>21.74</td><td>4.86</td><td>8.11</td></tr><tr><td>Starfish</td><td>16.41</td><td>5.41</td><td>8.29</td></tr><tr><td>Girl</td><td>8.90</td><td>6.69</td><td>3.80</td></tr><tr><td>Curl</td><td>17.01</td><td>7.05</td><td>3.53</td></tr><tr><td>Pensive</td><td>12.08</td><td>5.95</td><td>3.75</td></tr><tr><td>Screwdriver</td><td>15.48</td><td>5.71</td><td>5.75</td></tr><tr><td>Sofa</td><td>5.63</td><td>6.19</td><td>5.29</td></tr><tr><td>Mouse</td><td>20.71</td><td>4.34</td><td>3.95</td></tr></table>

PTv3’s inherently more advanced modeling capacity, we freeze the pre-trained PTv3 backbone during training and only optimize the lightweight PointNet classification head. In contrast, for all baseline methods, we follow their original settings and perform full fine-tuning of the entire model.

# B.4. 3D Part Segmentation

We evaluate the part segmentation task on the ShapeNetPart dataset. For our method, we adopt a simple MLP as the segmentation head built upon the pre-trained features. In the comparative experiments, we conduct evaluations under two distinct settings: full fine-tuning and frozen backbone. Full fine-tuning represents the common practice for applying pre-trained models to part segmentation tasks. To more rigorously compare the feature quality provided by different pre-training methods and to minimize interference from variations in network architecture, we additionally freeze the pre-trained backbone for each method and train only the segmentation head. This setup isolates the contribution of the learned features to the downstream segmentation performance. Note that OcCo and CrossPoint do not provide standalone pre-trained weights that are decoupled from downstream task heads; therefore, results under the frozen-backbone setting are not available for these methods. Similarly, PointDif does not include part segmentation fine-tuning results on ShapeNetPart in its original evaluation, so no full fine-tuning results are reported for it.

# C. Additional Experimental Results

# C.1. Fixed-Boundary Surface Parameterization

More visual comparison results for fixed-boundary parameterization are presented in Figure 13, with the corresponding isometric quantitative results given in Table 9. It can be observed that, across different types of 3D shapes, our method consistently achieves good fixed-boundary parameterization results, comparable to methods like FlexPara that require long per-scene overfitting times.

# C.2. 3D Shape Correspondence

More visual comparison results for 3D shape correspondence are presented in Figure 14. It can be observed that our method consistently delivers accurate shape correspondence results across different human bodies and a wide variety of poses, achieving performance on par with state-of-the-art task-specific approaches.

![](images/77296e3984ac61ca85964c2d7419a0de0e04c4736e563ae7dce2220c85dcd7ac.jpg)

<details>
<summary>other</summary>

| Method       | Group 1 | Group 2 | Group 3 |
| ------------ | ------- | ------- | ------- |
| Functional Map | High    | Medium  | Low     |
| ZoomOut      | Medium  | Low     | Low     |
| ULRSM        | Medium  | Low     | Low     |
| SMS          | Low     | Medium  | Low     |
| Ours         | Low     | Medium  | Low     |
</details>

Figure 14. More visualization of non-rigid 3D shape correspondence results produced by different approaches. From top to down: FM, ZoomOut, ULRSM, SMS, and Ours. 16 16