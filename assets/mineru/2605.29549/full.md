# Learning Representations from 3D Gaussian Splats

Julia Farganus ⋆, Krzysztof Żurawicki ⋆, Arkadiusz Gaweł ⋆, Weronika Jakubowska, and Halina Kwaśnicka

Department of Artificial Intelligence, Wrocław University of Science and Technology, Wrocław, Poland

julia.farganus@student.pwr.edu.pl, krzysztof.zurawicki@pwr.edu.pl, weronika.jakubowska@pwr.edu.pl

Abstract. 3D Gaussian Splatting (3DGS) is a recent approach for scene rendering. Although primarily designed for view synthesis, its potential for scene understanding tasks remains underexplored. In this work, we conduct a comparative evaluation of various geometric deep learning architectures for the classification of 3D scenes represented using Gaussian Splatting. We benchmark point-based and graph-based models across both traditional point cloud datasets and dedicated Gaussian Splatting datasets. Scenes are embedded into latent representations, which are evaluated through end-to-end classification, linear probing, and clustering analysis. Our study provides insight into the suitability of different geometry-aware architectures and input feature configurations for learning effective 3D Gaussian Splat representations. The results highlight consistent differences between architectural families and reveal the impact of Gaussian-specific attributes on the quality of representation.

Keywords: 3D Gaussian Splatting · Latent representations · Geometric Deep Learning.

# 1 Introduction

As AI becomes increasingly integral to daily life, machines must not only process, but truly understand data that describe the physical world. While natural phenomena often yield complex, high-dimensional data, research demonstrates [1] that these representations can be compressed into fewer dimensions for downstream tasks. A primary example is 3D structures, whose unordered and sparse nature poses significant challenges for standard neural networks. To address these complexities, we focus on 3D point-based representations, exploring various architectural approaches to the classification task and analyzing their learned embedding space. The importance of learning embeddings is grounded in the hypothesis that high-dimensional data actually resides on a lower-dimensional manifold [1]. Specifically, within the 3D domain, expressive embeddings that preserve semantic distances and similarities can be utilized to bypass resource-intensive computations in the native 3D space. Although creation of representations has been previously approached by masked autoencoders [14] or multi-task learning [9], we frame this task in a more direct manner, revisiting the efficacy of fully supervised learning. Our primary focus lies in comparing the underlying data structures and connectivity patterns inherent to the selected architectures.

We base our research on the 3D Gaussian Splatting (3DGS) [10] that has been up to date explored mostly for rendering and visualization. Crucially, like 3D point clouds, 3DGS consists of a set of unordered centroids, suggesting a natural compatibility with existing geometric neural architectures. However, unlike raw points, these primitives exhibit spatial extent and directionality. While this extended feature set is optimized strictly for visual rasterization rather than geometric fidelity – and could theoretically introduce noise in downstream tasks – pioneering work [28] has shown that incorporating auxiliary features like opacity, scale, and rotation can actually enhance shape recognition and yield more separable latent spaces. Intuitively, incorporating purely visual features like opacity and color can significantly enhance class separation. For instance, textureless or solid objects are typically represented by a small number of large, opaque primitives with uniform color. Conversely, intricate objects with wire-like or porous structures require elongated primitives with varied opacities – to account for holes – and differing colors.

Unlike purely centroid-based approaches, learning on 3DGS representations enables networks to differentiate between classes that share near-identical spatial geometry but possess distinct visual cues. Motivated by the sparsity of research in this direction, we pose a fundamental question: Can neural networks successfully extract meaningful geometric representations from features originally optimized purely for visual fidelity? To systematically evaluate this, we model 3DGS scenes as graphs, allowing us to benchmark three distinct architectural paradigms: global connectivity, self-connectivity, and local connectivity (explained in Section 3.1). We evaluate the quality of the resulting internal representations through a suite of downstream tasks, including linear probing alongside hard and soft clustering.

This geometric interpretation aligns our framework directly with Geometric Deep Learning [3]. This approach incorporates spatial symmetry and scale separation principles directly into neural architectures, effectively mitigating the curse of dimensionality. By applying graph embedding techniques, it maps sparse graphs with high-dimensional features into dense, low-dimensional vector spaces while strictly preserving structural properties. In this work, we provide an evaluation of such embeddings, derived as a by-product of the classification task. Our contributions are as follows:

1. Comprehensive evaluation of embeddings through supervised classification and unsupervised clustering tasks.   
2. Establishment of Graph Neural Network baselines for classification on 3D Gaussian Splatting datasets.

3. Quantitative benchmarking of state-of-the-art architectures with follow-up recommendations.

# 2 Related work

Existing classification frameworks adopt one of three primary representation strategies: projection-based, voxel-based, and point-based methods, each offering a unique strategy to encode 3D geometric structures into latent embeddings.

Point cloud classification is a practical problem encountered in object detection, medicine, and 3D reconstruction [27]. The early solutions include projectionbased methods [18, 26] and voxel-based methods [31, 26, 5]. Nevertheless, these methods ignore geometric representation and fail to learn recognizing abstract shapes. More efficient solutions work directly with the unordered structure of 3D point cloud. PointNet [15] introduces an architecture that transforms each point independently. Follow-ups include PointNet++ [16], which processes points hierarchically, and PointNeXt [17], which extends PointNet++ by introducing residual connections and separable MLPs.

Graph Neural Networks extend point-based methods by explicitly modeling local and non-local relationships between points through graph structures. Although many approaches are based on spectral graph theory, they struggle with generalization. SplineCNN [8] mitigates this limitation by operating directly in the spatial domain. Its convolution operator aggregates node features using a trainable kernel function based on B-spline bases. Another approach proposed in Graph Attention Network [19] (GAT) is to utilize masked attention layers, gaining efficiency from parallelization and shared weights for all edges. Previously mentioned methods use static neighborhood that may hamper the ability to fuse information from distant nodes. Dynamically built graphs were introduced in the Dynamic Graph Convolutional Neural Network [20] (DGCNN), through an EdgeConv operation that uses feature space to find semantically similar elements, even though they are distant in the 3D coordinate space. Other approaches for integrating contextual relationships involve pair-point impact function [30] or superpoint graph [12]. The work done in [21], [24] addresses the extension of convolutional architectures to the 3D point cloud domain.

Gaussian Splatting [10] is a novel technique for 3D visualization that has found use cases, for example, in medical imaging [2] and autonomous driving [11]. Let $\mathcal { G } = \{ g _ { i } \} _ { i = 1 } ^ { N }$ denote a 3D Gaussian Splatting representation consisting of N Gaussian primitives. According to the formulation [10], each primitive is characterized by its spatial position $\mathbf { p } _ { i } ,$ , the anisotropic scaling factors $\mathbf { s } _ { i }$ , the orientation represented as a unit quaternion $\mathbf { q } _ { i }$ , opacity $\alpha _ { i }$ , and view-dependent color represented by spherical harmonic coefficients sh $. i \cdot \ i$

$$
g _ {i} = (\mathbf {p} _ {i}, \mathbf {s} _ {i}, \mathbf {q} _ {i}, \alpha_ {i}, \mathbf {s h} _ {i}) \in \left(\mathbb {R} ^ {3}, \mathbb {R} ^ {3}, \mathbb {R} ^ {4}, \mathbb {R} ^ {1}, \mathbb {R} ^ {4 8}\right).
$$

In particular, considering only splat centroids $\mathbf { p } _ { i }$ makes them equivalent to a point cloud. As each feature carries a different meaning and lives in a separate domain, we also refer to them as heterogeneous attributes later on. The technique has been mainly employed for high quality rendering purposes with limited research exploring its potential as a replacement for point cloud representations. Work done in [29] shows that Gaussian Splatting objects perform better than point clouds on the 3D classification task. Their enhanced representation mitigates ambiguities in distinguishing wire-like and flat surfaces, as well as transparent or reflective objects. Gaussian-MAE [13] is the first to perform self-supervised pretraining on a large-scale Gaussian Splatting dataset. They show that pre-training on Gaussian Splats can encode shape priors that enhance classification performance on point cloud inputs.

![](images/591a34298bf5f542b5d2cc7fa15c2466d36ab251bc06054d1227d6ef8b3e4014.jpg)  
(a)

![](images/ae9788a5448ac3c492a83b29a38301c5dba64b0a3ab7168026089f96f42073a0.jpg)  
(b)

![](images/f226a83c6bfab89c278467550e5b65cfa774693b2380b711d7d09a1a23fd7ce0.jpg)  
(c)

![](images/9f63f636f2924f3cd80ead3c2b43c5ddbcb9671de98e3dbe6e8fa952509c3616.jpg)  
(d)   
Fig. 1: Samples of airplane class from datasets (a) ShapeSplat, (b) MACGS, (c) ModelNet40 and (d) ShapeNet. Gaussian Splatting representation provides is generally more detailed, but can suffer from artifacts as it’s visible in (b). Mesh representations are smooth, but can’t be directly processed by CNN’s.

Current research overlooks the potential of learned representations for 3D Gaussian Splats, largely due to the absence of dedicated mechanisms for sampling and aggregating splat features. To date, only [13] has addressed this by introducing specialized grouping and pooling layers. However, their approach is intrinsically tied to a Transformer architecture and suffers from unbalanced feature weighting, limiting its generalization to standard point-based models. Furthermore, recent work by [23] highlights the challenges of directly utilizing splat parameters due to their non-unique mapping and numerical heterogeneity.

In this work, we evaluate the impact of Gaussian features on the inherent mechanisms of Graph Neural Networks (GNNs) to assess their ability to mitigate these issues. We hypothesize that geometry-aware processing can yield robust embeddings suitable for downstream applications. Motivated by the multi-task learning approach in [9], we evaluate the suitability of these embeddings for both supervised and unsupervised tasks. To our knowledge, no prior study has systematically examined the latent representations learned by these architectures in the domain of Gaussian Splatting. We also include point cloud architectures to compare different connectivity paradigms.

# 3 Used models and 3D Embeddings

We select several point-based methods and adapt their input layers and featureprocessing mechanisms to handle the attributes of 3D Gaussian Splats. All models are trained end-to-end on classification tasks using both Gaussian Splatting and standard point cloud datasets, the latter serving as a baseline of uniformly sampled points – a geometric distribution that cannot be reproduced by simply discarding the additional Gaussian Splat features. We then provide a brief explanation and categorization of the considered point-processing approaches.

# 3.1 Approach Taxonomy

Global Connectivity. Multi-Layer Perceptrons (MLPs) process flattened point clouds completely oblivious to their underlying geometric layout. In this setting, the network establishes global connectivity where each layer’s output depends on the combination of all input points regardless of spatial proximity. Consequently, it lacks permutation invariance, meaning that reordering points or features drastically alters the output.

Self-Connectivity. The PointNet family – including PointNet, PointNet++, and PointNeXt – processes individual points independently using shared weights, a mechanism we define as self-connectivity. Permutation invariance is strictly enforced by aggregating these isolated point features via a symmetric pooling function (e.g., max-pooling). While the original PointNet operates globally, PointNet++ and PointNeXt introduce hierarchical grouping to capture expanding local neighborhoods, using relative distance encodings to improve spatial awareness.

Local Connectivity. Graph Neural Networks (GNNs) – specifically SplineCNN, DGCNN, and GAT – explicitly model local neighborhoods by constructing edge relations among points. In this framework, each Gaussian primitive serves as a node enriched with an extended feature vector, and local spatial relationships define the edges. This local connectivity allows for highly flexible graph construction; for instance, we define neighborhoods using k = 20 nearest neighbors, which DGCNN dynamically updates in the latent feature space. Figure 2 illustrates the general schema for these GNN architectures.

# 3.2 Embeddings

We formalize each network as a composition of a feature extractor $h _ { \theta }$ and a classifier $g _ { \gamma }$ . While a standard point cloud input is a set of 3D coordinates $V = \{ v _ { i } \} _ { i = 1 } ^ { n }$ where $\boldsymbol { v } _ { i } = ( x , y , z )$ , the extended 3DGS input expands each primitive to a 14- dimensional vector $v _ { i } = ( x , y , z , s _ { [ 1 : 3 ] } , q _ { [ 1 : 4 ] } , \alpha , \mathrm { s h } _ { [ 1 : 3 ] } )$ containing scaling, rotation, opacity, and color features. The final classification is computed as:

$$
\hat {y} = g _ {\gamma} \left(\mathrm{agg} \left(h _ {\theta} (V)\right)\right),
$$

where agg denotes a channel-wise max-pooling operator that aggregates the perprimitive feature matrix into a single vector. Crucially, we define the final scene embedding as the fixed-size vector extracted immediately after this aggregation step (discarding $g _ { \gamma } )$ . This choice captures the global structure synthesized by the encoder while allowing direct control over latent space expressiveness via the bottleneck dimension.

![](images/9ec883cc7abc3a87636bab689c1dc414ba9a29ff1e651df710f4b42bbef666de.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["x, y, z position ∈ R³"] --> B["Input"]
    C["scale ∈ R³ quaternion ∈ R⁴"] --> B
    D["opacity ∈ R¹"] --> B
    B --> E["Graph NN"]
    E --> F["max (f₁,k)"]
    E --> G["..."]
    E --> H["(4)"]
    F --> I["Class Scores"]
    G --> I
    H --> I
```
</details>

Fig. 2: Gaussian Splatting processing via GNNs. The input comprises spatial coordinates and a subset of auxiliary splat attributes (excluding spherical harmonics). The object is formulated as a graph and processed using the general message passing paradigm (1). Global feature aggregation (2) is performed via max pooling, resulting in latent vector (3), followed by an MLP classifier (4).

# 4 Experiments

Our experiments assess end-to-end classification performance across both Gaussian Splatting datasets and standard point cloud benchmarks. Furthermore, we evaluate the quality and discriminative power of the learned embeddings through downstream tasks.

# 4.1 Datasets

We selected two point clouds datasets ModelNet40 [22] and ShapeNet [4], as well as two Gaussian Splatting datasets: ShapeSplat [13] and MACGS [29]. Although these datasets are semantically similar, containing common objects such as a bag, airplane, or can (see examples in Figure 1), they differ significantly in structure. This diversity allows us to evaluate the performance of the network in varying data qualities. Specifically, ShapeSplat was trained on 360-degree views, resulting in objects that are structurally complete and visually well-defined from all angles. In contrast, MACGS relies on single-view supervision, leading to partial reconstructions where we anticipate decreased network performance. Finally, ModelNet40 and ShapeNet were converted to point cloud representations via uniform sampling from their original mesh formats. Regarding data set sizes, we operate on ModelNet40 – 9,843 samples, 40 classes, ShapeSplat – 30,843 samples, 55 classes, and MACGS – 4,288 samples, 30 classes. Additionally, we use ShapeNet-29 – 16,318 samples, a subset restricted to 29 classes for computational efficiency.

# 4.2 Experiments Setup

Evaluation framework. all models are trained end-to-end for the classification task without any explicit regularization on the embedding space. The quality of the learned representations is evaluated on a series of downstream tasks, following the protocols in [9, 13, 6].

Linear probing trains a linear classifier on top of the frozen embeddings. The

![](images/fc46ccca1bd520d49671b91f857b263b381e53a68d7f80f5b8d8539b9bd9b07b.jpg)

<details>
<summary>scatter</summary>

| x    | y    | cluster |
| ---- | ---- | ------- |
| -3.5 | 4.0  | 1       |
| -2.8 | 3.5  | 1       |
| -2.0 | 3.0  | 1       |
| -1.5 | 2.5  | 1       |
| -1.0 | 2.0  | 1       |
| -0.5 | 1.5  | 1       |
| 0.0  | 1.0  | 1       |
| 0.5  | 0.5  | 1       |
| 1.0  | 0.0  | 1       |
| 1.5  | -0.5 | 1       |
| 2.0  | -1.0 | 1       |
| 2.5  | -1.5 | 1       |
| 3.0  | -2.0 | 1       |
| -3.5 | -4.0 | 2       |
| -2.8 | -3.5 | 2       |
| -2.0 | -3.0 | 2       |
| -1.5 | -2.5 | 2       |
| -1.0 | -2.0 | 2       |
| -0.5 | -1.5 | 2       |
| 0.0  | -1.0 | 2       |
| 0.5  | -0.5 | 2       |
| 1.0  | 0.0  | 2       |
| 1.5  | 0.5  | 2       |
| 2.0  | 1.0  | 2       |
| 2.5  | 1.5  | 2       |
| 3.0  | 2.0  | 2       |
| -3.5 | -6.0 | 3       |
| -2.8 | -5.5 | 3       |
| -2.0 | -5.0 | 3       |
| -1.5 | -4.5 | 3       |
| -1.0 | -4.0 | 3       |
| -0.5 | -3.5 | 3       |
| 0.0  | -3.0 | 3       |
| 0.5  | -2.5 | 3       |
| 1.0  | -2.0 | 3       |
| 1.5  | -1.5 | 3       |
| 2.0  | -1.0 | 3       |
| 2.5  | -0.5 | 3       |
| 3.0  | 0.0  | 3       |
| -3.5 | -8.0 | 4       |
| -2.8 | -7.5 | 4       |
| -2.0 | -7.0 | 4       |
| -1.5 | -6.5 | 4       |
| -1.0 | -6.0 | 4       |
| -0.5 | -5.5 | 4       |
| 0.0  | -5.0 | 4       |
| 0.5  | -4.5 | 4       |
| 1.0  | -4.0 | 4       |
| 1.5  | -3.5 | 4       |
| 2.0  | -3.0 | 4       |
| 2.5  | -2.5 | 4       |
| 3.0  | -2.0 | 4       |
| -3.5 | -9.0 | 5       |
| -2.8 | -8.5 | 5       |
| -2.0 | -8.0 | 5       |
| -1.5 | -7.5 | 5       |
| -1.0 | -7.0 | 5       |
| -0.5 | -6.5 | 5       |
| 0.0  | -6.0 | 5       |
| 0.5  | -5.5 | 5       |
| 1.0  | -5.0 | 5       |
| 1.5  | -4.5 | 5       |
| 2.0  | -4.0 | 5       |
| 2.5  | -3.5 | 5       |
| 3.0  | -3.0 | 5       |
| -3.5 | -10.0|    |
| -2.8 | -9.5 |    |
| -2.0 | -9.0 |    |
| -1.5 | -8.5 |    |
| -1.0 | -8.0 |    |
| -0.5 | -7.5 |    |
| 0.0  | -7.0 |    |
| 0.5  | -6.5 |    |
| 1.0  | -6.0 |    |
| 1.5  | -5.5 |    |
| 2.0  | -5.0 |    |
| 2.5  | -4.5 |    |
| 3.0  | -4.0 |    |
|     |     nan   |         |
</details>

(a)

![](images/483ccbbb05431a20982de52276f5ca76b4725e35fef9760a30ec5f88c7c24605.jpg)  
(b)

![](images/43c477e8f998113becf10c2637011019a407266844bab9e7f8ad03e4b0a7920d.jpg)

<details>
<summary>scatter</summary>

| x    | y    | cluster |
| ---- | ---- | ------- |
| -10  | 5    | 1       |
| -8   | 4    | 2       |
| -6   | 3    | 3       |
| -4   | 2    | 4       |
| -2   | 1    | 5       |
| 0    | 0    | 6       |
| 2    | -1   | 7       |
| 4    | -2   | 8       |
| 6    | -3   | 9       |
| 8    | -4   | 10      |
| 10   | -5   | 11      |
| 12   | -6   | 12      |
| 14   | -7   | 13      |
| 16   | -8   | 14      |
| 18   | -9   | 15      |
| 20   | -10  | 16      |
| 22   | -11  | 17      |
| 24   | -12  | 18      |
| 26   | -13  | 19      |
| 28   | -14  | 20      |
| 30   | -15  | 21      |
| 32   | -16  | 22      |
| 34   | -17  | 23      |
| 36   | -18  | 24      |
| 38   | -19  | 25      |
| 40   | -20  | 26      |
| 42   | -21  | 27      |
| 44   | -22  | 28      |
| 46   | -23  | 29      |
| 48   | -24  | 30      |
| 50   | -25  | 31      |
| 52   | -26  | 32      |
| 54   | -27  | 33      |
| 56   | -28  | 34      |
| 58   | -29  | 35      |
| 60   | -30  | 36      |
| 62   | -31  | 37      |
| 64   | -32  | 38      |
| 66   | -33  | 39      |
| 68   | -34  | 40      |
| 70   | -35  | 41      |
| 72   | -36  | 42      |
| 74   | -37  | 43      |
| 76   | -38  | 44      |
| 78   | -39  | 45      |
| 80   | -40  | 46      |
| 82   | -41  | 47      |
| 84   | -42  | 48      |
| 86   | -43  | 49      |
| 88   | -44  | 50      |
| 90   | -45  | 51      |
| 92   | -46  | 52      |
| 94   | -47  | 53      |
| 96   | -48  | 54      |
| 98   | -49  | 55      |
| 100  | -50  | 56      |
| -10  | -5   | 6       |
| -8   | -4   | 7       |
| -6   | -3   | 8       |
| -4   | -2   | 9       |
| -2   | -1   | 10      |
| 0    | 0    | 11      |
| 2    | 1    | 12      |
| 4    | 2    | 13      |
| 6    | 3    | 14      |
| 8    | 4    | 15      |
| 10   | 5    | 16      |
| 12   | 6    | 17      |
| 14   | 7    | 18      |
| 16   | 8    | 19      |
| 18   | 9    | 20      |
| 20   | 10   | 21      |
| 22   | 11   | 22      |
| 24   | 12   | 23      |
| 26   | 13   | 24      |
| 28   | 14   | 25      |
| 30   | 15   | 26      |
| 32   | 16   | 27      |
| 34   | 17   | 28      |
| 36   | 18   | 29      |
| 38   | 19   | 30      |
| 40   | 20   |        |
| -10.5| -5.5±0.5|     |
| -9.5| -5.0±0.5|     |
| -8.5| -4.5±0.5|     |
| -7.5| -4.0±0.5|     |
| -6.5| -3.5±0.5|     |
| -5.5| -3.0±0.5|     |
| -4.5| -2.5±0.5|     |
| -3.5| -2.0±0.5|     |
| -2.5| -1.5±0.5|     |
| -1.5| -1.0±0.5|     |
| -0.5| -0.5±0.5|     |
| 0.5+         (additional points) are not provided in the image; the actual values may vary based on the random seed and data generation. The chart is a standard representation of the data being plotted.
</details>

(c)

![](images/9d438b57d76a3ca705364c185a520e2f5ca777c0b7a75516239733a7258033ac.jpg)

<details>
<summary>scatter</summary>

| x    | y    | cluster |
| ---- | ---- | ------- |
| -40  | -10  | red     |
| -35  | -8   | green   |
| -30  | -6   | blue    |
| -25  | -4   | purple  |
| -20  | -2   | pink    |
| -15  | 0    | brown   |
| -10  | 2    | gray    |
| -5   | 4    | cyan    |
| 0    | 6    | magenta |
| 5    | 8    | navy    |
| 10   | 10   | violet  |
| 15   | 12   | maroon  |
| 20   | 14   | gold    |
| 25   | 16   | silver  |
| 30   | 18   | zpink   |
| 35   | 20   | pink    |
| 40   | 22   | yellow  |
| 45   | 24   | cyan    |
| 50   | 26   | magenta |
| 55   | 28   | navy    |
| 60   | 30   | violet  |
| 65   | 32   | maroon  |
| 70   | 34   | gold    |
| 75   | 36   | cyan    |
| 80   | 38   | magenta |
| 85   | 40   | navy    |
| 90   | 42   | yellow  |
| 95   | 44   | cyan    |
| 100  | 46   | magenta |
| 105  | 48   | navy    |
| 110  | 50   | yellow  |
| 115  | 52   | cyan    |
| 120  | 54   | magenta |
| 125  | 56   | navy    |
| 130  | 58   | yellow  |
| 135  | 60   | cyan    |
| 140  | 62   | magenta |
| 145  | 64   | navy    |
| 150  | 66   | yellow  |
| 155  | 68   | cyan    |
| 160  | 70   | magenta |
| 165  | 72   | navy    |
| 170  | 74   | yellow  |
| 175  | 76   | cyan    |
| 180  | 78   | magenta |
| 185  | 80   | navy    |
| 190  | 82   | yellow  |
| 195  | 84   | cyan    |
| 200  | 86   | magenta |
| 205  | 88   | navy    |
| 210  | 90   | yellow  |
| 215  | 92   | cyan    |
| 220  | 94   | magenta |
| 225  | 96   | navy    |
| 230  | 98   | yellow  |
| 235  | 100  | cyan    |
| 240  | 102  | magenta |
| 245  | 104  | navy    |
| 250  | 106  | yellow  |
| 255  | 108  | cyan    |
| 260  | 110  | magenta |
| 265  | 112  | navy    |
| 270  | 114  | yellow  |
| 275  | 116  | cyan    |
| 280  | 118  | magenta |
| 285  | 120  | navy    |
| 290  | 122  | yellow  |
| 295  | 124  | cyan    |
| 300  | 126  | magenta |
| 305  | 128  | navy    |
| 310  | 130  | yellow  |
| 315  | 132  | cyan    |
| 320  | 134  | magenta |
| 325  | 136  | navy    |
| 330  | 138  | yellowing|
| -40   | -9   | red     |
| -35   | -7   | green   |
| -30   | -5   | blue    |
| -25   | -3   | purple  
|
| -20   | -1   | brown  
|
| -15   | -1   | cyan  
|
| -10   | -3   | magenta|
|
| -5    | -5   | navy  
|
| -10   | -7   | blue  
|
| -5    | -9   | magenta|
|
| -10   | -11                  | navy  
|
| -5    | -13                  | yellowing|
|
| -10   | -15                  | cyan  
|
| -5    | -17                  | magenta|
|
| -10   | -19                  | navy  
|
| -5    | -21                  | yellowing|
|
| -10   | -23                  | cyan  
|
| -5    | -25                  | magenta|
|
| -10   | -27                  | navy  
|
| -5    | -29                  | yellowing|
|
| -10   | -31                  | cyan  
|
| -5    | -33                  | magenta|
|
| -10   | -35                  | navy  
|
| -5    | -37                  | yellowing|
|
| -10   | -39                  | cyan  
|
| -5    | -41                  | magenta|
|
| -10   | -43                  | navy  
|
| -5    | -45                  | yellowing|
|
| -10   | -47                  | cyan  
|
| -5    | -49                  | magenta|
|
| -10   | -51                  | navy  
|
| -5    | -53                  | yellowing|
|
| -10   | -55                  | cyan  
|
| -5    | -57                  | magenta|
|
| -10   | -59                  | navy  
|
| -5    | -61                  | yellowing|
|
| -10   | -63                  | cyan  
|
| -5    | -65                  | magenta|
|
| -10   | -67                  | navy  
|
| -5    | -69                  | yellowing|
|
| -10   | -71                  | cyan  
|
| -5    | -73                  | magenta|
|
| -10   | -75                  | navy  
|
| -5    | -77                  | yellowing|
|
| -10   | -79                  | cyan  
|
| -5    | -81                  | magenta|
|
| -10   | -83                  | navy  
|
| -5    | -85                  | yellowing|
|
| -10   | -87                  | cyan  
|
| -5    | -89                  | magenta|
|
| -10   | -91                  | navy  
|
| -5    | -93                  | yellowing|
|
| -10   | -95                  | cyan  
|
| -5    | -97                  | magenta|
|
| -10   | -99                  | navy  
|
</details>

(d)   
Fig. 3: T-SNE visualization of ShapeSplat. Images, from left to right present architecture/features: (a) PointNet++ / p, (b) PointNet++ / psqo, (c) Spline p and (d) Spline / psqo. PointNet++ obtains visually more compact clusters (a), especially after including Gaussian splat features (b). On the other hand, Spline struggles to separate classes (c), with more observable spread when incorporating extra features (d).

Table 1: Model complexity and embedding dimensions. P. (M) denotes the number of parameters in millions; Emb. len. refers to the embedding vector length. Parentheses indicate standard deviation. 

<table><tr><td>Charact.</td><td>MLP</td><td>PointNet</td><td>PointNet++</td><td>PointNeXt</td><td>DGCNN</td><td>Spline</td><td>GAT</td></tr><tr><td>P. (M)</td><td>2.50 (0.30)</td><td>3.50</td><td>1.70</td><td>1.37 (0.01)</td><td>1.85 (0.25)</td><td>3.00</td><td>0.78 (0.03)</td></tr><tr><td>Emb. len.</td><td>1024</td><td>1024</td><td>1024</td><td>1024</td><td>1024</td><td>512</td><td>256</td></tr></table>

F1-score reveals whether the discriminative power originates from the classification head or the embeddings.

To evaluate the structure of the latent space, we utilize two clustering approaches. K-Means clustering (hard clustering) uses Adjusted Mutual $I n f o r \mathrm { - }$ mation (AMI) score to measure how well generated clusters align with the true class distribution, correcting for chance. Gaussian Mixture Models (soft clustering) model each class as a single multivariate Gaussian with a diagonal covariance matrix. It provides insights into the structure of the latent space, under the intuition that semantic classes should map to compact clusters. We evaluate the fit by the mean $\left( \mu _ { \mathrm { l o g } } \right)$ and standard deviation $\left( \sigma _ { \mathrm { l o g } } \right)$ of the log-likelihood of test set embeddings. To enable cross-dataset comparison independent of the absolute values of the mean log-likelihood $( \mu _ { l o g } )$ and its standard deviation $\left( \sigma _ { l o g } \right)$ , we introduce the Mean Log Ratio, defined as:

$$
\mathrm{MeanLogRatio} = \frac {\sum_ {i = 1} ^ {C} \mu_ {\log} ^ {(i)}}{\sum_ {i = 1} ^ {C} \sigma_ {\log} ^ {(i)}},
$$

where C denotes the number of classes.

Implementation Details. We train all models once, minimizing the standard Cross-Entropy loss using a batch size of 32, except for GAT, where the batch size is reduced to 16 due to memory constraints imposed by the attention mechanism. Optimization is performed using AdamW with an initial learning rate of $1 0 ^ { - 3 }$ , a weight decay of $1 0 ^ { - 4 }$ , and a Cosine annealing scheduler. To avoid overfitting, we employ early stopping with a patience of 3 epochs; the average training duration was approximately 12 ± 2 hours. The architecture implementations for PointNet and PointNet++ are adapted from [25], while PointNeXt follows the official codebase [17]. Graph-based models (SplineCNN, DGCNN, GAT) and the MLP baseline were constructed using geometric deep learning primitives from [7]. all experiments were conducted on a single NVIDIA A100. Network sizes are given in Table 1.

Data Preparation. Input features underwent a specific normalization, appropriate for each domain. Point coordinates were standardized to zero mean and unit variance. Gaussian opacities o were mapped through a sigmoid function, followed by standardization. To resolve antipodal ambiguity, quaternions were L2-normalized and aligned to the upper hemisphere by multiplying with the sign of the real component. Scale features were standardized globally. ShapeSplat and MACGS point clouds were downsampled to N = 1024 points via Farthest Point Sampling (FPS) [16]. For mesh-based datasets (ShapeNet-29, ModelNet40), we first performed uniform surface sampling, followed by FPS to obtain a canonical set of 1024 points. We use 80 : 20 train/test split, with a further 90 : 10 train/validation split used during the training phase.

Points attributes. For Gaussian Splats, we focus on comparing its two representations: centroids only (p) and full splats which include position, scale (s), quaternion (q), and opacity (o) attributes – we denote the complete set as psqo. We also conduct an experiment with attributes exclusive to Gaussian Splats (sqo) to assess their discriminative power in isolation. With coordinate-only input, we utilize p to represent position as feature attributes. The color (expressed as spherical harmonics in 3DGS) is omitted, consistent with [20] and [29], to ensure a fair comparison with traditional point clouds.

# 4.3 Results

Classification Performance. accuracy results are detailed in Table 2.

MACGS dataset. Spline achieves the best performance (≈ 85%), competing closely with PointNet++. Incorporating full splat attributes (psqo) generally improves performance over coordinates alone (p). This gain is most prominent for PointNeXt (+33.8 pp), DGCNN (+13.9 pp) and GAT (+13.3 pp). Overall, PointNet, PointNet++, and Spline achieve the best performances (in the range of ≈ 76-84%), with PointNeXt, DGCNN, and GAT typically achieving < 70%. MLP, as expected, yields the worst results.

ShapeSplat dataset. The best performance is achieved by similarly scoring Spline, PointNet, and PointNet++ in the range of 84-86%. DGCNN performs better than PointNeXt, scoring ≈ 79-83%, which is a notable improvement over the MACGS results. a similar enhancement is observed for GAT with respect to the corresponding experiments on MACGS. For PointNeXt and GAT, we observe a noticeable improvement (+10 pp) over the coordinate-only setting (p). MLP, even though scoring behind, improves with the enhanced quality of the splats.

Point Cloud Benchmarks. On ModelNet40 and ShapeNet-29, PointNet++ achieves the highest accuracy, closely followed by Spline and PointNet. although ModelNet40 results on PointNet and PointNeXt (72-78%) are slightly lower than reported in [17] this is likely due to the specific split or FPS preprocessing. Notably, PointNeXt exhibits moderate degradation compared to the leading architectures on ShapeNet-29, while for GAT and DGCNN, the performance gap is significant. While the MACGS dataset typically provides lower accuracy than point cloud benchmarks, PointNet, Spline, and GAT exhibit improved performance relative to ModelNet40. In contrast, the ShapeSplat dataset provides distinct gains, most prominently for DGCNN and GAT, while PointNet and Spline also surpass their ModelNet40 baselines.

Table 2: Accuracy results for end-to-end training. Numbers in bold denote the best result within a given dataset and feature. Underlined numbers indicate the best result within a given dataset and architecture. although accuracy does not reflect class imbalance, it remains a common metric reported in related studies such as [29], [20], and [13] 

<table><tr><td>Dataset</td><td>Attr.</td><td>MLP</td><td>PointNet</td><td>PointNet++</td><td>PointNeXt</td><td>DGCNN</td><td>Spline</td><td>GAT</td></tr><tr><td>ModelNet40</td><td>p</td><td>4.052</td><td>78.201</td><td>86.791</td><td>72.639</td><td>75.810</td><td>80.713</td><td>40.438</td></tr><tr><td>ShapeNet-29</td><td>p</td><td>19.761</td><td>86.737</td><td>89.130</td><td>66.796</td><td>3.639</td><td>88.276</td><td>11.602</td></tr><tr><td rowspan="3">MACGS</td><td>p</td><td>14.510</td><td>77.170</td><td>79.760</td><td>35.016</td><td>57.024</td><td>78.928</td><td>46.858</td></tr><tr><td>psqo</td><td>23.660</td><td>79.570</td><td>84.658</td><td>68.806</td><td>70.980</td><td>84.935</td><td>60.166</td></tr><tr><td>sqo</td><td>13.216</td><td>76.250</td><td>82.348</td><td>69.850</td><td>51.017</td><td>78.281</td><td>42.144</td></tr><tr><td rowspan="3">ShapeSplat</td><td>p</td><td>40.901</td><td>85.730</td><td>86.323</td><td>57.208</td><td>80.951</td><td>86.685</td><td>68.268</td></tr><tr><td>psqo</td><td>56.658</td><td>85.729</td><td>86.504</td><td>67.678</td><td>83.611</td><td>85.794</td><td>80.021</td></tr><tr><td>sqo</td><td>42.942</td><td>85.006</td><td>84.773</td><td>63.711</td><td>79.259</td><td>84.915</td><td>70.657</td></tr></table>

Linear Probing. We report the weighted average F1-score to account for class imbalance. The results are summarized in Table 3.

MACGS dataset. While PointNet and PointNet++ achieve robust performance (≈ 80%), PointNeXt exceeds the GNN baselines only when using the psqo and sqo features (≈ 60%). among GNNs, Spline yields the best results, yet it attains only half the accuracy of the PointNet family. DGCNN and GAT score is only 20%, indicating a severe loss of generalization compared to their end-toend accuracy. Incorporating Gaussian attributes generally drives improvement; as shown in Figure 5a, PointNeXt exhibits a relative improvement over 90%, with sqo slightly outperforming psqo. Conversely, Spline is the only architecture that suffers a relative regression (≈ 10%) with Gaussian features.

Table 3: Linear probing F1 Score. Numbers in bold denote the best result within a given dataset and a feature. Underlined numbers indicate the best result within a given dataset and an architecture. 

<table><tr><td>Dataset</td><td>Attr.</td><td>MLP</td><td>PointNet</td><td>PointNet++</td><td>PointNeXt</td><td>DGCNN</td><td>Spline</td><td>GAT</td></tr><tr><td>ModelNet40</td><td>p</td><td>0.316</td><td>81.418</td><td>85.380</td><td>76.322</td><td>53.331</td><td>54.181</td><td>42.810</td></tr><tr><td>ShapeNet-29</td><td>p</td><td>6.521</td><td>86.366</td><td>86.682</td><td>79.198</td><td>6.521</td><td>69.338</td><td>12.521</td></tr><tr><td rowspan="3">MACGS</td><td>p</td><td>3.355</td><td>78.091</td><td>76.247</td><td>31.724</td><td>19.555</td><td>44.004</td><td>14.928</td></tr><tr><td>psqo</td><td>5.456</td><td>79.674</td><td>81.317</td><td>60.076</td><td>20.333</td><td>39.303</td><td>18.162</td></tr><tr><td>sqo</td><td>4.729</td><td>77.091</td><td>80.932</td><td>63.119</td><td>11.498</td><td>33.945</td><td>13.668</td></tr><tr><td rowspan="3">ShapeSplat</td><td>p</td><td>6.001</td><td>84.868</td><td>83.988</td><td>77.349</td><td>43.153</td><td>62.050</td><td>15.355</td></tr><tr><td>psqo</td><td>6.958</td><td>85.412</td><td>84.104</td><td>79.925</td><td>42.512</td><td>61.034</td><td>20.506</td></tr><tr><td>sqo</td><td>6.630</td><td>81.961</td><td>82.180</td><td>77.370</td><td>34.482</td><td>54.008</td><td>19.069</td></tr></table>

ShapeSplat dataset. The PointNet family outperforms GNN-based architectures, showing greater robustness to class imbalance. PointNet and PointNet++ achieve 81–85% accuracy, with PointNeXt close behind ≈ 77-79%. Spline exhibits a substantial drop relative to its fully supervised performance, lagging more than 20 percentage points behind the PointNet family. DGCNN and GAT perform poorly, although DGCNN roughly doubles its MACGS performance (≈ 20% and ≈ 40% respectively). As shown in Figure 5b, Gaussian features typically yield modest gains (1–2 pp), but for DGCNN and Spline, the coordinateonly input (p) performs slightly better. Overall, the results indicate that GNNs may struggle to model interactions between heterogeneous feature types.

Point Cloud Benchmarks. On point cloud benchmarks, PointNet++ consistently secures the highest scores, closely followed by PointNet and PointNeXt. In contrast, GNNs exhibit notable degradation, often yielding near-random results (with a minor exception for Spline on ShapeNet-29). Compared to the ModelNet40 baseline, the MACGS dataset exhibits a performance drop across all architectures, with the degradation being most visible in GNNs. In the case of ShapeSplat, however, we observe improvements for PointNet, PointNeXt, and Spline. While the 3DGS format does not outperform the ShapeNet-29 baseline for the PointNet family, it notably enhances the performance of GAT and DGCNN architectures.

Clustering. This evaluation assesses the geometric properties of the embedding space, specifically focusing on intra-class compactness and inter-class separability. The results are provided in Tables 4 and 5.

MACGS dataset. PointNet achieves the highest AMI (≈ 0.6), followed by PointNet++ (≈ 0.5), while PointNeXt, Spline, and DGCNN obtain suboptimal scores (≈ 0.2-0.4), with GAT and MLP results being negligible. Incorporating splat attributes (psqo) generally enhances AMI – most notably for PointNeXt (+0.2) – yet this does not necessarily translate to improved Log Ratio. Specifically, Spline is the only architecture where adding Gaussian features to positions improves the Log Ratio; conversely, PointNet and PointNeXt peak using standalone Gaussian features (sqo), while PointNet++, DGCNN, and GAT favor coordinate-only input (p). as illustrated in Figure 4, there is a trade-off between these metrics: PointNet (sqo and psqo) defines the Pareto frontier, whereas for PointNeXt, relying solely on coordinates degrades both metrics. Meanwhile, Spline with full feature set (psqo) offers a favorable compromise, sacrificing negligible AMI for a significantly improved Log Ratio compared to the baseline.

Table 4: Clustering - AMI results. Numbers in bold denote the best result within a given dataset and a feature. Underlined numbers indicate the best result within a given dataset and an architecture. 

<table><tr><td>Dataset</td><td>Attr.</td><td>MLP</td><td>PointNet</td><td>PointNet++</td><td>PointNeXt</td><td>DGCNN</td><td>Spline</td><td>GAT</td></tr><tr><td>ModelNet40</td><td>p</td><td>0.000</td><td>0.650</td><td>0.665</td><td>0.487</td><td>0.371</td><td>0.569</td><td>0.339</td></tr><tr><td>ShapeNet-29</td><td>p</td><td>0.006</td><td>0.558</td><td>0.643</td><td>0.494</td><td>0.006</td><td>0.626</td><td>0.008</td></tr><tr><td rowspan="3">MACGS</td><td>p</td><td>0.008</td><td>0.619</td><td>0.510</td><td>0.223</td><td>0.148</td><td>0.294</td><td>0.085</td></tr><tr><td>psqo</td><td>0.016</td><td>0.651</td><td>0.527</td><td>0.430</td><td>0.152</td><td>0.235</td><td>0.096</td></tr><tr><td>sqo</td><td>0.012</td><td>0.633</td><td>0.509</td><td>0.370</td><td>0.137</td><td>0.209</td><td>0.076</td></tr><tr><td rowspan="3">ShapeSplat</td><td>p</td><td>0.017</td><td>0.732</td><td>0.679</td><td>0.603</td><td>0.601</td><td>0.645</td><td>0.273</td></tr><tr><td>psqo</td><td>0.024</td><td>0.735</td><td>0.711</td><td>0.640</td><td>0.643</td><td>0.620</td><td>0.244</td></tr><tr><td>sqo</td><td>0.016</td><td>0.648</td><td>0.675</td><td>0.594</td><td>0.479</td><td>0.603</td><td>0.167</td></tr></table>

Table 5: Clustering - Mean Log Ratio results. Numbers in bold denote the best result within a given dataset and a feature. Underlined numbers indicate the best result within a given dataset and an architecture. Results missing for ModelNet40 and ShapeNet-29 come from the fact that the GMM may fail to converge. 

<table><tr><td>Dataset</td><td>Attr.</td><td>MLP</td><td>PointNet</td><td>PointNet++</td><td>PointNeXt</td><td>DGCNN</td><td>Spline</td><td>GAT</td></tr><tr><td>ModelNet40</td><td>p</td><td>-</td><td>2.219</td><td>2.965</td><td>2.039</td><td>1.794</td><td>2.337</td><td>2.218</td></tr><tr><td>ShapeNet-29</td><td>p</td><td>-</td><td>2.253</td><td>2.609</td><td>3.385</td><td>-</td><td>2.245</td><td>1.961</td></tr><tr><td rowspan="3">MACGS</td><td>p</td><td>1.454</td><td>2.755</td><td>2.577</td><td>1.905</td><td>2.079</td><td>1.729</td><td>1.734</td></tr><tr><td>psqo</td><td>1.672</td><td>2.390</td><td>2.055</td><td>2.109</td><td>2.008</td><td>2.397</td><td>1.581</td></tr><tr><td>sqo</td><td>1.841</td><td>2.846</td><td>2.170</td><td>2.250</td><td>1.618</td><td>1.841</td><td>1.720</td></tr><tr><td rowspan="3">ShapeSplat</td><td>p</td><td>1.704</td><td>2.889</td><td>2.803</td><td>2.442</td><td>1.764</td><td>1.946</td><td>1.656</td></tr><tr><td>psqo</td><td>1.725</td><td>2.575</td><td>2.708</td><td>2.419</td><td>2.015</td><td>2.112</td><td>1.950</td></tr><tr><td>sqo</td><td>1.732</td><td>2.377</td><td>2.954</td><td>1.593</td><td>1.578</td><td>1.820</td><td>1.889</td></tr></table>

ShapeSplat dataset. PointNet and PointNet++ dominate the clustering benchmarks, achieving high AMI scores for coordinate-based (p) and full (psqo) configurations (≈ 0.7), with only minor degradation observed for the sqo subset. DGCNN, Spline, and PointNeXt follow closely, exceeding an AMI threshold of 0.6 in the best case. although integrating splat attributes generally yields performance gains, except for Spline and GAT, relying solely on Gaussian features (sqo) typically causes regression. In terms of Log Ratio, PointNet and Point-Net++ score top values, similarly as with AMI results. Interestingly, for the

![](images/9435f09877d3892a98ca9429deb3edb92069d3341bc923dd3b4a1a6a42af4501.jpg)  
Fig. 4: Clustering Evaluation. Relation between AMI from k-Means and standard deviation of mean Log Probabilities from GMM. Marker colors represent a unique architecture, and their style a given feature.

PointNet architectures, augmenting positions with Gaussian features (psqo) provides no significant benefit over the baseline (p), although PointNet++ notably achieves its peak Log Ratio using sqo alone. Conversely, GNNs demonstrate a clear dependency on Gaussian attributes to surpass the MLP baseline. As visualized in Figure 4, PointNet and PointNet++ occupy the optimal upper-right quadrant, striking a balance between cluster quality and probabilistic fit. In contrast, PointNeXt and DGCNN exhibit degradation in the sqo setting, while GAT remains suboptimal, despite improving its MACGS performance. Overall, ShapeSplat yields consistently robust AMI scores across most architectures, with the primary optimization challenge lying in the Log Ratio. We provide a visual assessment of the learned embedding space in Figure 3.

Point Cloud Benchmarks. For both datasets, PointNet++, PointNet and Spline attain good performance for both clustering types. For DGCNN and GAT we can observe a degradation, especially for ShapeNet-29, remaining nearzero scores obtained by MLP baseline. On the MACGS dataset, clustering performance generally degrades compared to point cloud baselines. In contrast, ShapeSplat achieves improvements for PointNet and DGCNN, as well as for PointNet++ and PointNeXt, although with some regressions in the Mean Log Ratio metric.

# 5 Conclusions

Contrary to initial intuition, GNNs generally struggle with end-to-end classification, with the notable exception of SplineCNN, which retains accuracy comparable to the PointNet family. However, these networks suffer from significant performance drops in linear probing and clustering, suggesting that standard message-passing mechanisms require adaptation for the Gaussian Splat domain. Since Gaussian Splats are non-uniformly distributed in space (in contrast to uniformly sampled point clouds), the constructed k-NN neighborhoods may fail to reflect semantically relevant connections. Examining the architectures, DGCNN appears to struggle with consistently capturing local geometric structure due to the dynamic recomputation of the k-NN graph at every layer. As it computes distances in feature space, the heterogeneity of attributes likely hinders the learning of meaningful connections. Conversely, GAT operates on fixed neighborhoods but discards explicit structural information in favor of a masked attention mechanism. Additionally, it aggregates features via linear combinations that, as previously hypothesized, may be suboptimal for this data type. SplineCNN employs local processing mechanisms that closely resemble the operations of Convolutional Neural Networks (CNNs). Its weights are locally constrained within a defined kernel, operating directly on relative distances. This structural inductive bias likely explains its classification performance, although its generalization capability degrades significantly in downstream tasks compared to the PointNet family. As anticipated, the global processing schema of the MLP consistently provides the lowest scores; however, the use of 3DGS representations improves performance compared to point cloud baselines.

![](images/ca12f0215d6ba3415adcafd7565ed0bead3de98db1f0a8fa7520ce30e156511e.jpg)

<details>
<summary>bar</summary>

| Model          | p (baseline) | p (+2.0%) | p (+1.3%) | p (+6.7%) | p (+6.1%) | p (+89.4%) | p (+98.0%) | p (+4.0%) | p (+1.3%) | p (+7.7%) | p (+2.9%) | p (+1.7%) | p (+8.4%) |
| -------------- | ------------ | --------- | --------- | --------- | --------- | ---------- | ---------- | --------- | --------- | --------- | --------- | --------- | --------- |
| MLP            | 0.05         | 0.06      | 0.05      | 0.05      | 0.05      | 0.05       | 0.05       | 0.05      | 0.05      | 0.05      | 0.05      | 0.05      | 0.05      |
| PointNet       | 0.78         | 0.80      | 0.78      | 0.80      | 0.80      | 0.80       | 0.80       | 0.80      | 0.80      | 0.80      | 0.80      | 0.80      | 0.80      |
| PointNet++     | 0.75         | 0.82      | 0.75      | 0.82      | 0.82      | 0.82       | 0.82       | 0.82      | 0.82      | 0.82      | 0.82      | 0.82      | 0.82      |
| PointNextX Network | 0.32        | 0.62      | 0.32      | 0.62      | 0.62      | 0.62       | 0.62       | 0.62      | 0.62      | 0.62      | 0.62      | 0.62      | 0.62      |
| DGCNN          | 0.22         | 0.42      | 0.22      | 0.42      | 0.42      | 0.42       | 0.42       | 0.42      | 0.42      | 0.42      | 0.42      | 0.42      | 0.42      |
| Spline         | 0.45         | 0.75      | 0.45      | 0.75      | 0.75      | 0.75       | 0.75       | 0.75      | 0.75      | 0.75      | 0.75      | 0.75      | 0.75      |
| GAT            | 0.15         | 0.35      | 0.15      | 0.35      | 0.35      | 0.35       | 0.35       | 0.35      | 0.35      | 0.35      | 0.35      | 0.35      | 0.35      |
</details>

(a) Linear probing - MACGS

![](images/65ba856a9d4c0281f6817546d6c73e121dcd666f6492bf4b5a57daef3aa5645d.jpg)

<details>
<summary>bar</summary>

| Model       | p (baseline) | psqo   | sqo    |
|-------------|--------------|--------|--------|
| MLP         | +15%         | +10.5% | +10.5% |
| PointNet    | +0.6%        | -3.4%  | -0.6%  |
| PointNet++  | +0.1%        | -2.2%  | -0.1%  |
| PointNextNetwork | +2.3%      | +2.0%  | +2.0%  |
| DGCNN       | -1.5%        | -20.1% | -1.5%  |
| Spline      | +1.6%        | +31.0% | +1.6%  |
| GAT         | +3.5%        | +22.2% | +3.5%  |
</details>

(b) Linear probing - ShapeSplat   
Fig. 5: Linear probing F1 Score for each feature configuration. Percentage value above the bar containers denote the relative gain or lost when using p.

Future directions. We argue that incorporating dedicated message-passing mechanisms could greatly benefit the Gaussian Splat representation, as each input feature carries a distinct semantic meaning. Although some progress has been made in this direction through Splat Grouping and Pooling layers [13], it still does not enhance the mathematical interpretation and possible interactions between features. Furthermore, the current standard for point cloud preprocessing utilizes Farthest Point Sampling (FPS). Since FPS considers only spatial distance, it may discard splats that contain auxiliary features critical to object representation. Consequently, we suggest that considering the probabilistic interpretation of Gaussian Splats could drive the implementation of a specialized, feature-aware sampling routine. Finally, the robust, generalizable performance of the PointNet family in our experiments underscores the need to maintain a balance between self- and local connectivity in future architectural designs.

Acknowledgments. The work of W. Jakubowska was supported by the National Centre of Science (Poland) Grant No. 2023/50/E/ST6/00068. We gratefully acknowledge the Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computational facilities and support under computational grant No. PLG/2025/018862. Authors would like to acknowledge Maciej Zięba for review and his insightful comments on paper.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

# References

1. Altman, N., Krzywinski, M.: The curse(s) of dimensionality. Nature Methods 15, 399 – 400 (2018), https://api.semanticscholar.org/CorpusID:44115671   
2. Bonilla, S., Zhang, S., Psychogyios, D., Stoyanov, D., Vasconcelos, F., Bano, S.: Gaussian pancakes: Geometrically-regularized 3d gaussian splatting for realistic endoscopic reconstruction (2024), https://arxiv.org/abs/2404.06128   
3. Bronstein, M.M., Bruna, J., Cohen, T., Veličković, P.: Geometric deep learning: Grids, groups, graphs, geodesics, and gauges (2021), https://arxiv.org/abs/2104.13478   
4. Chang, A.X., Funkhouser, T., Guibas, L., Hanrahan, P., Huang, Q., Li, Z., Savarese, S., Savva, M., Song, S., Su, H., et al.: Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012 (2015)   
5. Chen, Y., Liu, J., Zhang, X., Qi, X., Jia, J.: Voxelnext: Fully sparse voxelnet for 3d object detection and tracking. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 21674–21683 (2023)   
6. Chopin, J., Dahyot, R.: Performance of gaussian mixture model classifiers on embedded feature spaces. arXiv preprint arXiv:2410.13421 (2024)   
7. Fey, M., Lenssen, J.E.: Fast graph representation learning with PyTorch Geometric. In: ICLR Workshop on Representation Learning on Graphs and Manifolds (2019)   
8. Fey, M., Lenssen, J.E., Weichert, F., Müller, H.: Splinecnn: Fast geometric deep learning with continuous b-spline kernels. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 869–877 (2018)   
9. Hassani, K., Haley, M.: Unsupervised multi-task feature learning on point clouds. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 8160–8171 (2019)   
10. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering (2023), https://arxiv.org/abs/2308.04079   
11. Khan, M., Fazlali, H., Sharma, D., Cao, T., Bai, D., Ren, Y., Liu, B.: Autosplat: Constrained gaussian splatting for autonomous driving scene reconstruction. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 8315–8321 (2025). https://doi.org/10.1109/ICRA55743.2025.11127564   
12. Landrieu, L., Simonovsky, M.: Large-scale point cloud semantic segmentation with superpoint graphs. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition pp. 4558–4567 (2017), https://api.semanticscholar.org/CorpusID:4396837   
13. Ma, Q., Li, Y., Ren, B., Sebe, N., Konukoglu, E., Gevers, T., Van Gool, L., Paudel, D.P.: A large-scale dataset of gaussian splats and their self-supervised pretraining. In: 2025 International Conference on 3D Vision (3DV). pp. 145–155. IEEE (2025), https://arxiv.org/abs/2408.10906

14. Pang, Y., Tay, E.H.F., Yuan, L., Chen, Z.: Masked autoencoders for 3d point cloud self-supervised learning. World Scientific Annual Review of Artificial Intelligence 1, 2440001 (2023)   
15. Qi, C.R., Su, H., Mo, K., Guibas, L.J.: Pointnet: Deep learning on point sets for 3d classification and segmentation (2017), https://arxiv.org/abs/1612.00593   
16. Qi, C.R., Yi, L., Su, H., Guibas, L.J.: Pointnet++: Deep hierarchical feature learning on point sets in a metric space (2017), https://arxiv.org/abs/1706.02413   
17. Qian, G., Li, Y., Peng, H., Mai, J., Hammoud, H., Elhoseiny, M., Ghanem, B.: Pointnext: Revisiting pointnet++ with improved training and scaling strategies. Advances in neural information processing systems 35, 23192–23204 (2022)   
18. Su, H., Maji, S., Kalogerakis, E., Learned-Miller, E.: Multi-view convolutional neural networks for 3d shape recognition (2015), https://arxiv.org/abs/1505.00880   
19. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., Bengio, Y.: Graph attention networks. In: International Conference on Learning Representations (2018)   
20. Wang, Y., Sun, Y., Liu, Z., Sarma, S.E., Bronstein, M.M., Solomon, J.M.: Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog) 38(5), 1–12 (2019)   
21. Wu, W., Qi, Z., Fuxin, L.: Pointconv: Deep convolutional networks on 3d point clouds (2020), https://arxiv.org/abs/1811.07246   
22. Wu, Z., Song, S., Khosla, A., Yu, F., Zhang, L., Tang, X., Xiao, J.: 3d shapenets: A deep representation for volumetric shapes. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1912–1920 (2015)   
23. Xin, Y., Liu, Y., Xie, X., Li, X.: Learning unified representation of 3d gaussian splatting (2025), https://arxiv.org/abs/2509.22917   
24. Xu, Y., Fan, T., Xu, M., Zeng, L., Qiao, Y.: Spidercnn: Deep learning on point sets with parameterized convolutional filters (2018), https://arxiv.org/abs/1803.11527   
25. Yan, X.: Pointnet/pointnet++ pytorch (2019)   
26. Yang, Z., Goldsztein, G.: Classification using 3d point cloud and 2d image on abstract objects. Journal of Student Research 10 (11 2021)   
27. Zhang, H., Wang, C., Tian, S., Lu, B., Zhang, L., Ning, X., Bai, X.: Deep learningbased 3d point cloud classification: A systematic survey and outlook. Displays 79, 102456 (2023). https://doi.org/https://doi.org/10.1016/j.displa.2023.102456, https://www.sciencedirect.com/science/article/pii/S0141938223000896   
28. Zhang, R., Zhu, H., Zhao, J., Zhang, Q., Cao, X., Ma, Z.: Mitigating ambiguities in 3d classification with gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2025)   
29. Zhang, R., Zhu, H., Zhao, J., Zhang, Q., Cao, X., Ma, Z.: Mitigating ambiguities in 3d classification with gaussian splatting. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 27275–27284 (2025)   
30. Zhao, H., Jiang, L., Fu, C.W., Jia, J.: Pointweb: Enhancing local neighborhood features for point cloud processing. In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5560–5568 (2019)   
31. Zhou, Y., Tuzel, O.: Voxelnet: End-to-end learning for point cloud based 3d object detection (2017), https://arxiv.org/abs/1711.06396