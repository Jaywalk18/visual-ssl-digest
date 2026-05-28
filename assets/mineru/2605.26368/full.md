# Unified Panoramic Geometry Estimation via Multi-View Foundation Models

Vukasin Bozic

ETH Zürich

vukasin.bozic@ethz.ch

Isidora Slavkovic

isidora.slavkovic@gmail.com

Dominik Narnhofer

ETH Zürich

dnarnhofer@ethz.ch

Nando Metzger

Athlence Sports

nando.metzger@athlencesports.com

Denis Rozumny

Meta

rozumden@gmail.com

Konrad Schindler

ETH Zürich

schindler@ethz.ch

# Nikolai Kalischek

Google

nikolai.kalischek@gmail.com

![](images/a03a74e5f3ce7f690bfed437887f59fa42b6eb79130c24de51e9eb2052ede5b0.jpg)  
Figure 1: Different geometric modalities predicted by PaGeR. Given only a single monocular panoramic image, our framework simultaneously reconstructs highly detailed scale-invariant depth, absolute metric depth, surface normals, and sky segmentation masks across both indoor and outdoor environments. Spatial measurements are presented in meters.

# Abstract

Geometry estimation from perspective images has greatly advanced, maturing to the point where off-the-shelf foundation models are able to reconstruct 3D scene structure not only from multi-view imagery, but even from a single view. A natural extension is 3D reconstruction from panoramas, with the exciting prospect of recovering a full 360◦ scene from a single panoramic image. In this work, we introduce PaGeR (Panoramic Geometry Reconstruction), a framework to lift powerful 3D foundation models designed for perspective imagery to the panorama domain. Our strategy is to start from a pre-trained transformer for 3D reconstruction and turn it into a unified high-performance model that predicts scale-invariant depth, metric depth, surface normals, and sky masks from both perspective and omnidirectional images, in a single forward pass. By keeping architectural changes to a minimum and mixing perspective and panoramic images during training, PaGeR retains the rich 3D prior of the underlying foundation model while learning to also estimate geometrically consistent 360◦ scenes from single panoramas. We extensively test our method in both indoor and outdoor environments and find that it delivers state-of-the-art performance and excellent zero-shot performance across a wide range of scenes.

# 1 Introduction

Sensing and understanding the 3D structure of the surrounding world is important in many applications, ranging from virtual and augmented reality to autonomous driving and robotics. Scene depth and surface normals are two central geometric properties in that context: together, they describe the position and the local surface orientation at any point of the scene, providing a complete representation that supports graphics tasks like rendering and relighting as well as high-level perception tasks like spatial reasoning and path planning.

A particularly attractive, but also heavily ill-posed variant is to recover depth or surface normals from a single RGB image [11, 19], obviating the need for multi-view capture and camera pose estimation. Early attempts relied on limited datasets and convolutional backbones [5, 8], but large-scale data collection [35] and advances in neural architectures, most notably vision transformers and denoising diffusion models [50, 19], have greatly advanced monocular geometry estimation. Most recently, this trend has converged with learning-based multi-view reconstruction, leading to foundation feedforward models [43, 25] capable of zero-shot, dense 3D reconstruction. From their massive training datasets, captured under diverse imaging conditions, these models acquire not only an understanding of multi-view geometry but also an elaborate prior of the world’s 3D surface structure, which supports detailed, dense depth estimation from single views.

Yet, these models are designed for perspective images such that every image covers only a limited field of view, and many viewpoints must be aggregated and fused to build up spatial context and perceive the complete environment. Panoramic images, by construction, provide a full 360° view around the camera location, offering rich global context for holistic 3D understanding. However, high-quality panoramic datasets with metrically accurate depth and surface normals, needed to train panoramic reconstruction models, are laborious to collect and remain scarce. As a result, existing models tend to overfit to comparatively small datasets and struggle to generalize to unseen scenes. Another limitation is that existing models commonly represent panoramic images in equirectangular projection, which introduces serious geometric distortions. On the one hand, this means an extremely uneven sampling of the ray space (and, after unwarping, of the 3D environment). On the other hand, and perhaps more importantly, it means that one cannot easily employ transfer learning from models trained with perspective images.

We take a different route and explore cubemaps as our panorama representation. Rather than designing custom architectures applicable specifically to panoramas, we use this parametrization to repurpose state-of-the-art perspective foundation models for the panoramic domain. The cubemap representation has been used to adapt pre-trained, diffusion-based image generators to 360° imagery [17]. For our purposes, we prefer to build on top of deterministic, feed-forward foundation models. Besides avoiding the computational overhead of diffusion and the practical limitations of operating in a compressed latent space, models like DA3 [25] are already designed for (perspective) multi-view input. They offer a natural synergy with the cubemap format, as their geometric prior includes the integration of multiple viewing directions and is fundamentally stronger than previous, purely monocular schemes. To properly anchor the spatial context, we explicitly condition the architecture on camera parameters and introduce targeted modifications of the decoder to ensure distortion-free and seamless reconstruction across face boundaries. Furthermore, we propose to use a mixed training regime with both synthetic panoramas and real perspective imagery. This strategy allows the network to adapt to the 360° setting while remaining firmly grounded in real-world image statistics, thus preventing overfitting to peculiarities of synthetic data and preserving the prior of the pre-trained foundation model.

Taken together, we introduce PaGeR, a unified geometry estimation framework for panoramic (and perspective) images. Its latent representation, inherited from the foundation geometry model, is holistic and allows for simultaneous decoding of multiple scene properties. We exploit this property and equip the backbone with multiple, coupled task heads: Scale-Invariant (SI) depth, metric scale estimation, surface normals, and sky segmentation. That design allows the network to jointly reason about related geometric properties and to efficiently extract a comprehensive 3D representation in a single forward pass (see Fig. 1). By reparameterizing panoramic geometry as a structured multi-view problem, we achieve high-resolution, metrically accurate predictions that set a new state of the art for several benchmarks. Furthermore, to address the lack of benchmark data for rigorous evaluation in long-range, outdoor scenarios, we curate and introduce ZüriPano, a novel dataset of real-world outdoor panoramas with associated high-accuracy LiDAR scans. In summary, our contributions are:

• A novel strategy to adapt foundation geometry models to panorama geometry. Our scheme is built around the cubemap representation consisting of six perspective images, and combines it with a hybrid training strategy to seamlessly transfer 3D scene priors to the $3 6 0 ^ { \circ }$ panorama setting, while sidestepping degradations caused by equirectangular distortion.   
• PaGeR, a unified panoramic geometry estimation model, featuring a shared transformer backbone (adopted from DA3) and specialized task heads to enable holistic reconstruction in a single forward pass.   
• Zero-shot generalization to unseen indoor and outdoor scenes, outperforming methods limited to a specific setting; and the new ZüriPano benchmark for zero-shot evaluation.

# 2 Method

This section outlines the geometric preliminaries of panoramic representations and our perspective backbone architecture. We then introduce the panoramic adaptation layers and the hybrid training strategy designed to bridge the perspective and spherical domains. Finally, we detail the unified multi-task architecture and formalize the loss objectives for each geometric modality.

# 2.1 Preliminaries

Panoramic Image Representations. Panoramas capture a holistic $3 6 0 ^ { \circ } \times 1 8 0 ^ { \circ }$ environment on the unit sphere $\mathbb { S } ^ { \breve { 2 } } .$ The standard Equirectangular Projection (ERP) maps spherical coordinates, i.e., longitude $\theta \in [ - \pi , \pi ]$ and latitude $\bar { \phi } \in [ - \pi / 2 , \pi / 2 ]$ , to a 2D planar grid $( \dot { u } , v ) \in [ 0 , 1 ] ^ { 2 }$ via:

$$
u = \frac {\theta}{2 \pi} + 0. 5, \quad v = \frac {\phi}{\pi} + 0. 5 \tag {1}
$$

While structurally simple, ERP introduces severe nonlinear distortions. The horizontal sampling density scales by $\sec ( \phi )$ , causing extreme stretching near the poles $( \phi  \pm \pi / 2 )$ . This domain shift degrades the efficacy of translation-invariant architectures optimized for perspective imagery.

To mitigate polar distortions, the cubemap projection maps $\mathbb { S } ^ { 2 }$ onto the six faces of a circumscribed unit cube $\mathcal { C } .$ Each cube face constitutes a standard $9 0 °$ FoV perspective image. A 3D ray $\mathbf { p } =$ $( x , y , z ) \in \mathbb { S } ^ { 2 }$ is mapped to local face coordinates via gnomonic projection $( \mathrm { e . g . , } u _ { c } = x / z , v _ { c } = y / z$ for the front face where $z = 1 )$ . This piecewise perspective formulation offers uniform sampling and directly aligns with the inductive priors of models trained on perspective data. However, partitioning the continuous sphere introduces geometric and photometric discontinuities at face boundaries, requiring custom adaptations of the architecture to maintain global consistency.

Geometry Transformer Backbone. Our framework is compatible with any multi-view transformer architecture. We instantiate our model using Depth Anything 3 (DA3) [25], which couples a vision transformer encoder [29] with a dense prediction transformer decoder [36]. Given a set of S perspective views $\mathcal { T } = \{ I _ { i } \} _ { i = 1 } ^ { S }$ , the encoder tokenizes the inputs and routes them through interleaved intra-image and cross-image attention layers. This global attention mechanism can optionally be conditioned on explicit camera parameters, namely intrinsic matrices $\mathbf { K } _ { i } \in \mathbb { R } ^ { 3 \times 3 }$ and extrinsic poses $\mathbf { E } _ { i } \in S E ( 3 )$ , to guide spatial cross-view reasoning. The encoder yields hierarchical feature maps $\mathcal { F } = \{ F ^ { ( \ell ) } \}$ across transformer layers $\ell ,$ which the decoder progressively upsamples and fuses to output dense spatial predictions.

![](images/df788850b257ccb2e13d52c12a7354f80b0520dd7507fbb121cb8dfc7bc71b07.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["RGB-Panorama"] --> B["Backbone"]
    B --> C["Features"]
    B --> D["Depth Head"]
    C --> E["Skay Mask"]
    D --> F["Scale Invariant Depth"]
    E --> G["Normals Head"]
    F --> H["Coarse Metric Head"]
    G --> I["normals"]
    H --> J["normals"]
    I --> K["normals"]
    J --> L["normals"]
    K --> M["normals"]
    L --> N["normals"]
    M --> O["normals"]
    N --> P["normals"]
    O --> Q["normals"]
    P --> R["normals"]
    Q --> S["normals"]
    R --> T["normals"]
    S --> U["normals"]
    T --> V["normals"]
    U --> W["normals"]
    V --> X["normals"]
    W --> Y["normals"]
    X --> Z["normals"]
    Y --> AA["normals"]
    Z --> AB["normals"]
```
</details>

Figure 2: PaGeR Architecture. An input RGB panorama is processed by a shared geometry transformer backbone to predict a sky mask, scale-invariant (SI) depth, surface normals, and coarse metric depth. In the metric branch, the final absolute depth is obtained by aligning the SI depth with coarse metric predictions and masking the sky. In the normal branch, predicted orientations are masked by the sky segmentation to produce the final surface normal map.

# 2.2 Panoramic Adaptation and Joint Training

To adapt the multi-view architecture for holistic 360◦ estimation, we format the panoramic input as a six-face cubemap and supply fixed camera matrices K alongside axis-aligned extrinsics $\mathbf { \bar { E } } _ { i } ,$ $i = 1 , \ldots , 6$ . While these geometric parameters explicitly define the spatial configuration, naively assembling independent face predictions into an equirectangular projection yields pronounced discontinuities at the boundaries. Furthermore, training exclusively on synthetic panoramas can cause the model to quickly diverge from its pre-training weights. We resolve these challenges through structural adaptations that favor global feature extraction and local decoding, complemented by a regularized joint training regime.

Implicit Encoder Synchronization. We fine-tune the ViT encoder on panoramic data without any structural modifications. Guided by the fixed camera tokens, face positional embeddings, and cross-view attention layers, the network naturally learns to route context and synchronize features across adjacent cubemap faces. The fine-tuning allows the backbone to adapt to the spherical topology while preserving the rich perspective priors learned during pre-training.

Spherically Aware Decoder Padding. Although global synchronization occurs in the encoder, local boundary artifacts can still emerge during dense upsampling in the decoder. To ensure continuous spherical sampling, we integrate cross-face valid padding into all convolutional and interpolation operations within the decoder architecture [14]. Instead of standard zero padding, this layer dynamically extracts features from geometrically adjacent cubemap faces, enforcing seamless geometric and photometric transitions across all boundaries.

Mixed Panoramic / Perspective Co-Training. To preserve the rich priors inherited from the perspective backbone and mitigate the sim-to-real domain gap, we employ a training strategy that alternates between two data streams. For panoramic batches, the network processes the full six-face configuration (S = 6) with active cross-face padding. For perspective batches, we isolate a single real-world image (S = 1), warp it to a $9 0 ^ { \circ }$ field of view to match the imaging geometry of cubemap faces, and assign it the extrinsics of a random equatorial face. Cross-face padding is dynamically disabled for these perspective samples, with the layer reverting to standard zero padding. The dualstream training protects the model from catastrophic forgetting while teaching it to handle continuous spherical observations.

# 2.3 Multi-Task Geometric Decoding

Existing geometric foundation models are typically confined to a single modality, such as scaleinvariant depth estimation. For a comprehensive 3D understanding of 360◦ environments, we add multi-task decoding to the unified backbone. Specialized prediction heads simultaneously decode depth, surface orientation, and sky masks in a single forward pass (see Fig. 2), always operating on the planar cubemap faces to benefit from the underlying perspective prior.

Scale-Invariant Depth. The model is supervised with the local, orthogonal per-face log-planar depth $z ^ { * } = \log ( Z _ { \mathrm { p l a n a r } } ^ { * } )$ to compress metric variance and avoid optimization bias from distant background objects. We remove the final exponential activation of the decoder to work directly in its native log space. The head outputs both the predicted scale-invariant log depth $\hat { z } _ { \mathrm { S I } }$ and an aleatoric confidence map $c _ { p }$ . To isolate relative shape from metric size, we dynamically compute an optimal log-space shift $\begin{array} { r } { \bar { \beta } ^ { * } = \arg \operatorname* { m i n } _ { \beta } \| ( \hat { z } _ { \mathrm { S I } } + \beta ) - z ^ { * } \| _ { 2 } ^ { 2 } } \end{array}$ and optimize the aligned predictions $\hat { z } _ { \mathrm { a l i g n e d } } = \hat { z } _ { \mathrm { S I } } + \beta ^ { * }$ We supervise the scale-invariant depth branch using a composite loss function $\mathcal { L } _ { \mathrm { d e p t h } }$ that balances per-pixel precision, local smoothness, and surface alignment:

$$
\mathcal {L} _ {\text { depth }} = \lambda_ {L _ {1}} \mathcal {L} _ {1} + \lambda_ {\text { grad }} \mathcal {L} _ {\text { grad }} + \lambda_ {\text { norm }} \mathcal {L} _ {\text { norm }} \tag {2}
$$

where $\begin{array} { r } { \mathcal { L } _ { 1 } = \frac { 1 } { N } \sum _ { p } \left( c _ { p } \left| \hat { z } _ { \mathrm { a l g n e d } , p } - z _ { p } ^ { * } \right| - \lambda _ { c } \log c _ { p } \right) , \mathcal { L } _ { \mathrm { g r a d } } = \frac { 1 } { N } \sum _ { p } \sum _ { i \in \{ x , y \} } \left| \partial _ { i } \hat { z } _ { \mathrm { a l g n e d } , p } - \partial _ { i } z _ { p } ^ { * } \right| } \end{array}$ and $\begin{array} { r } { \mathcal { L } _ { \mathrm { n o r m } } = \frac { 1 } { N } \sum _ { p } \left( 1 - \hat { \mathbf { n } } _ { p } \cdot \mathbf { n } _ { p } ^ { * } \right) } \end{array}$ . Here, N denotes the total number of valid pixels. The primary loss $\mathcal { L } _ { 1 }$ measures the absolute discrepancy scaled by the predicted aleatoric confidence $c _ { p } .$ . We complement this with an edge-aware gradient penalty $\mathcal { L } _ { \mathrm { g r a d } }$ to preserve discontinuities at object boundaries, and a normal consistency loss $\mathcal { L } _ { \mathrm { { n o r m } } }$ that enforces geometric alignment via the cosine similarity between ground-truth orientations $\mathbf { n } ^ { * }$ and surface normals nˆ, derived analytically from the predicted depth maps.

Surface Normals. We instantiate a dedicated, parallel decoding branch for normals. It is initialized with the pre-trained depth weights to benefit from the close connection between depth and normals. The final layer is modified to output three-dimensional unit vectors nˆ. Training utilizes a joint objective $\dot { \mathcal { L } } _ { \mathrm { n o r m a l } } = \lambda _ { \mathrm { c o s } } \mathcal { L } _ { \mathrm { c o s } } + \lambda _ { \mathrm { p e r c } } \dot { \mathcal { L } } _ { \mathrm { p e r c } } .$ , which combines a pixel-wise cosine similarity loss with a VGG-based perceptual loss [39]. The latter serves to prevent over-smoothing and promote sharp edges.

Metric Scale. To reconstruct an absolute scale without disrupting the reconstruction of relative local geometry, we decouple metric estimation from the high-resolution, scale-invariant branch. A parallel, coarse decoder predicts a low-resolution metric log-depth map $\hat { z } _ { \mathrm { m } }$ alongside an aleatoric confidence map $\hat { c } _ { \mathrm m }$ . From that map, we infer a global scale factor $\hat { \beta }$ as the median difference between the coarse metric log-depth and an average-pooled version of its scale-invariant counterpart, computed over a lower-resolution grid of spatial anchors a:

$$
\hat {\beta} = \operatorname{median} _ {\mathbf {a}} \left(\hat {z} _ {\mathrm{m}} (\mathbf {a}) - \operatorname{pool} \left[ \hat {z} _ {\mathrm{SI}} \right] (\mathbf {a})\right) \tag {3}
$$

The median filters out localized geometric discrepancies. The final, absolute metric depth is recovered as $\hat { Z } _ { \mathrm { m } } = \exp ( \hat { \beta } ) \hat { Z } _ { \mathrm { S I } }$ . The metric head is trained with a coverage-weighted version of the confidenceaware $\mathcal { L } _ { 1 }$ loss against appropriately downsampled ground-truth targets, ensuring that invalid regions do not corrupt the scale estimation.

Sky Segmentation. Modeling infinite depth directly destabilizes metric regression. We explicitly decouple unbounded regions by introducing a lightweight sky segmentation branch, such that the primary depth heads can focus on structures with finite depth. The branch reads out geometric cues from intermediate decoder features and fuses them with semantic tokens extracted from the deep encoder layers and passed through a small, fully connected network. The concatenated feature maps are mapped to binary sky probabilities $\hat { Y }$ with a shallow convolutional decoder. This head is trained with a combination of binary cross-entropy, focal [26], and dice losses [41] w.r.t. the ground-truth mask. Its outputs serve to mask sky regions with undefined geometry in the depth and normal outputs.

# 3 Experiments

We evaluate PaGeR across diverse quantitative and qualitative experiments on both indoor and outdoor environments. We compare against existing state-of-the-art panoramic geometry estimators and provide detailed ablation studies to isolate and validate the individual structural adaptations and joint training choices of our approach.

# 3.1 Training Details

We initialize our framework from pre-trained DA3 weights [25] featuring a DINOv2 backbone [29]. Optimization proceeds in two sequential stages. First, we jointly train the scale-invariant depth and surface normal decoders, adapting the backbone features to support both geometric modalities. Second, we freeze these components and independently train the metric scale and sky segmentation heads using the frozen feature representations. We optimize using AdamW [28] with an exponentially decaying learning rate schedule initialized at $3 \cdot 1 0 ^ { - 4 }$ and an Exponential Moving Average decay of 0.999. The first stage requires 12 hours of training on 8 NVIDIA H200 GPUs, while the second stage completes in an additional 8 hours.

![](images/d4ffd063f9816069c3ed9535f3208fc78cb71f322d11544e50397ef23cb35f8e.jpg)

<details>
<summary>text_image</summary>

Input
DAP
DA²
Ours
Ground Truth
Matterport3D360
Stanford2D3DS
ZüriPano
</details>

Figure 3: Qualitative comparison of panoramic depth estimation. Visual results from PaGeR, DAP [27], and $\mathrm { D A } ^ { 2 }$ [23] (the strongest metric and scale-invariant baselines) alongside the RGB input and ground-truth depth on Matterport3D360, Stanford2D3DS, and ZüriPano. Our framework recovers sharper boundaries and more accurate global structures than competitors. Additional examples are in the appendix. Best viewed zoomed in.

<table><tr><td rowspan="2">Method</td><td colspan="3">Matterport3D360</td><td colspan="3">Stanford2D3DS</td><td colspan="3">ZüriPano</td></tr><tr><td>AbsRel ↓</td><td>RMSE ↓</td><td> $\delta_1 \uparrow$ </td><td>AbsRel ↓</td><td>RMSE ↓</td><td> $\delta_1 \uparrow$ </td><td>AbsRel ↓</td><td>RMSE ↓</td><td> $\delta_1 \uparrow$ </td></tr><tr><td>DreamCube* [14]</td><td>30.45</td><td>108.81</td><td>54.26</td><td>28.45</td><td>75.30</td><td>56.39</td><td>29.57</td><td>484.91</td><td>53.07</td></tr><tr><td>DepthAnyCamera [9]</td><td>25.62</td><td>94.06</td><td>68.11</td><td>18.35</td><td>53.73</td><td>76.23</td><td>21.98</td><td>487.66</td><td>65.41</td></tr><tr><td>MoGe [45]</td><td>18.12</td><td>81.68</td><td>77.11</td><td>18.12</td><td>81.68</td><td>77.11</td><td>19.25</td><td>484.18</td><td>77.12</td></tr><tr><td>EGformer* [54]</td><td>16.74</td><td>96.15</td><td>79.32</td><td>13.64</td><td>58.95</td><td>84.64</td><td>55.49</td><td>721.83</td><td>31.21</td></tr><tr><td>DAP [27]</td><td>15.84</td><td>85.18</td><td>82.44</td><td>9.76</td><td>46.69</td><td>92.37</td><td>19.86</td><td>583.42</td><td>72.09</td></tr><tr><td>RPG360 [16]</td><td>15.40</td><td>79.40</td><td>82.40</td><td>11.91</td><td>46.60</td><td>87.73</td><td>18.27</td><td>455.28</td><td>78.41</td></tr><tr><td>UniK3D†[31]</td><td>14.82</td><td>69.27</td><td>83.67</td><td>9.93</td><td>40.69</td><td>93.34</td><td>31.00</td><td>832.13</td><td>38.78</td></tr><tr><td>PanDA* [4]</td><td>13.84</td><td>82.82</td><td>84.26</td><td>10.69</td><td>55.72</td><td>90.73</td><td>24.64</td><td>515.48</td><td>58.31</td></tr><tr><td>DA2[23]</td><td>11.06</td><td>67.72</td><td>89.18</td><td>7.64</td><td>37.29</td><td>95.63</td><td>61.22</td><td>869.08</td><td>2.17</td></tr><tr><td>PaGeR (Ours)</td><td>9.67</td><td>64.69</td><td>90.87</td><td>5.93</td><td>35.34</td><td>96.10</td><td>9.36</td><td>299.61</td><td>94.75</td></tr></table>

Table 1: Quantitative comparison between PaGeR and state-of-the-art baselines across indoor (Matterport3D360, Stanford2D3DS) and outdoor (ZüriPano). Best and second-best results are indicated in bold and underlined, respectively. Methods optimized using in-domain training are marked with †, and affine-invariant methods are denoted with \*.

Our mixed data regime balances 80k synthetic panoramas from Structured3D [56] and our PanoInfinigen dataset with 10k real perspective images from ScanNet++ [51] and ARKitScenes [2] to mitigate the sim-to-real domain gap. Following standard practice [27, 49], we train independent metric scale heads for indoor and outdoor environments to accommodate distinct spatial layouts. We maintain a training resolution of 504 × 504 pixels per cubemap face, which assembles into a 2K equirectangular panorama. At inference time, our unified framework processes a full 2K panorama in 0.5 seconds while consuming 12.8 GB of memory, allowing for deployment on a single consumer-grade GPU. The choices of hyperparameters are given in Tab. 7.

<table><tr><td rowspan="2">Method</td><td colspan="3">Matterport3D360</td><td colspan="3">Stanford2D3DS</td><td colspan="3">ZüriPano</td></tr><tr><td>AbsRel ↓</td><td>RMSE ↓</td><td> $\delta_1 \uparrow$ </td><td>AbsRel ↓</td><td>RMSE ↓</td><td> $\delta_1 \uparrow$ </td><td>AbsRel ↓</td><td>RMSE ↓</td><td> $\delta_1 \uparrow$ </td></tr><tr><td>UniK3D $^{\dagger}$  [31]</td><td>33.43</td><td>143.24</td><td>46.48</td><td>24.37</td><td>64.71</td><td>64.50</td><td>35.81</td><td>1345.48</td><td>34.86</td></tr><tr><td>RPG360 [16]</td><td>29.26</td><td>136.99</td><td>35.33</td><td>29.35</td><td>90.73</td><td>17.36</td><td>40.36</td><td>772.16</td><td>2.14</td></tr><tr><td>DepthAnyCamera [9]</td><td>25.16</td><td>132.66</td><td>62.25</td><td>17.15</td><td>59.29</td><td>77.98</td><td>33.23</td><td>716.38</td><td>37.83</td></tr><tr><td>DAP $^{\ddagger}$  [27]</td><td>30.16</td><td>168.88</td><td>80.19</td><td>10.97</td><td>53.39</td><td>90.64</td><td>47.00</td><td>839.72</td><td>24.02</td></tr><tr><td>PaGeR $^{\ddagger}$  (Ours)</td><td>21.83</td><td>123.48</td><td>69.50</td><td>10.94</td><td>45.43</td><td>90.94</td><td>31.97</td><td>530.85</td><td>39.30</td></tr></table>

Table 2: Quantitative comparison of panoramic metric depth models. Methods optimized using in-domain training are marked with †, while those using separate indoor or outdoor prediction heads are marked with ‡. Bold font marks the best result, underlined the second-best.

# 3.2 Evaluation Protocol

Datasets and Evaluation Ranges. Consistent with panoramic benchmarks [4, 44, 23], we evaluate scale-invariant and metric depth on the real-world indoor datasets Matterport3D360 [37] and Stanford2D3DS [40]. To address the indoor bias of existing literature, we also introduce ZüriPano, a custom outdoor urban LiDAR dataset tailored for long-range geometric evaluation (see Sec. B). For all depth evaluations, we enforce a broad range constraint of [0, 75] m to ensure a thorough assessment of global structures. This avoids the evaluation bias of prior works [4, 23] that use a narrow [0, 5] m window, which inadvertently masks far-field errors. For surface orientation, we benchmark on the Structured3D dataset [56], where all compared baselines are trained to guarantee a fair comparison.

Metrics and Processing. Following established conventions [50], depth accuracy is measured via Absolute Relative Error (AbsRel), Root Mean Squared Error (RMSE), and the threshold percentage $\delta _ { 1 } $ . Scale-invariant depth maps are adjusted using a standard least-squares alignment prior to scoring, whereas metric depth is evaluated directly without modifications. For surface normals, we report the Mean Angular Error, Mean Squared Error (MSE), and the fraction of pixels with errors below $\delta _ { \theta } \in \{ 5 ^ { \circ } , 2 2 . 5 ^ { \circ } \}$ [12], with all predictions normalized to unit length before evaluation.

# 3.3 Quantitative Comparison

As demonstrated in Table 1, PaGeR consistently outperforms existing methods across all datasets. While it demonstrates notable improvements on indoor-biased benchmarks, its primary advantage lies in its cross-domain generalization. On the challenging outdoor ZüriPano dataset, PaGeR reduces the Absolute Relative Error (AbsRel) from the previous best 18.27 for RPG360 to 9.36, nearly cutting it in half. This substantial improvement confirms that our framework enhances structural geometry globally and does not need to trade off indoor vs. outdoor accuracy.

This balanced capability extends directly to absolute scale recovery, as detailed in Table 2. By decoupling metric scale estimation from the structural backbone, our independent domain heads successfully specialize to indoor or outdoor structures while sharing the same underlying transformer features. On the ZüriPano metric benchmark, PaGeR establishes a commanding lead with an RMSE of 530.85 compared to 716.38 for the next best, DepthAnyCamera. At the same time, it maintains high accuracy indoors, outperforming recent baselines such as UniK3D and DAP on both indoor datasets.

To evaluate higher-order geometric consistency, we report surface normal estimation on the Structured3D dataset in Table 3. PaGeR sets a new state of the art, outperforming specialized architectures including PanoNormal and HyperSphere. Specifically, our framework achieves a Mean Angular Error of $\bar { 5 } . 4 9 ^ { \circ }$ and an MSE of 174.9, which represents a major reduction from the 246.6 MSE of the previous state of the art. This validates our multi-task architecture choice, demonstrating that joint learning across separate heads improves the recovery of fine-grained surface structures.

<table><tr><td></td><td colspan="3">Replica360_4K(Seam Eval.)</td><td colspan="3">Stanford2D3DS(SI Depth Eval.)</td></tr><tr><td>Method</td><td>SDD ↓</td><td>SP ↓</td><td>SS ↓</td><td>AbsRel ↓</td><td>RMSE ↓</td><td> $\delta_{1}$  ↑</td></tr><tr><td>DA3</td><td>33.71</td><td>44.29</td><td>68.97</td><td>6.58</td><td>35.79</td><td>95.94</td></tr><tr><td>Ours</td><td>0.94</td><td>0.38</td><td>2.31</td><td>5.93</td><td>35.34</td><td>96.10</td></tr></table>

(a) Comparison against DA3 baseline.

<table><tr><td>Method</td><td>Mean ↓</td><td>MSE ↓</td><td> $\delta_{5^{\circ}}$  ↑</td><td> $\delta_{22.5^{\circ}}$  ↑</td></tr><tr><td>Frozen backbone</td><td>6.21</td><td>251.1</td><td>77.00</td><td>90.52</td></tr><tr><td>W/o perceptual</td><td>5.92</td><td>198.2</td><td>77.82</td><td>90.96</td></tr><tr><td>Rand. init. DPT</td><td>6.02</td><td>191.9</td><td>78.02</td><td>89.76</td></tr><tr><td>Ours</td><td>5.49</td><td>174.9</td><td>79.91</td><td>92.83</td></tr></table>

(b) Surface Normals on Structured3D.

Table 4: Ablations for vanilla DA3 and surface normals.

# 3.4 Qualitative Comparison

Visual comparisons across both scale-invariant and metric depth settings are presented in Figure 3. While baselines such as DAP and DA2 suffer from structural over-smoothing and fail to resolve fine details, PaGeR delivers sharp geometric boundaries alongside a globally continuous scene layout.

This structural precision is further highlighted in the point cloud reconstructions illustrated in Figure 9. Here, competing approaches frequently introduce warped geometries and surface discontinuities along the boundaries. In contrast, our framework accurately recovers complex outdoor topographies while preserving intricate indoor structural arrangements. Furthermore, Figure 4 demonstrates the high-resolution detail of our estimated surface normals; for more qualitative examples, see Appendix G.

# 3.5 Ablation Studies

Comparison to Baseline. We evaluate PaGeR against the vanilla DA3 baseline using depth accuracy and cross-face geometric consistency metrics (Table 4a; formal definitions in Section C). PaGeR significantly improves upon both. While preserving the robust geometric priors of the base model, our framework resolves depth misalignment across face boundaries to yield seamless reconstructions, as qualitatively illustrated in Figure 8.

<table><tr><td>Method</td><td>AbsRel ↓</td><td>RMSE ↓</td><td> $\delta_1 \uparrow$ </td></tr><tr><td>W/o camera cond.</td><td>10.12</td><td>43.91</td><td>90.01</td></tr><tr><td>Linear-depth</td><td>6.96</td><td>39.12</td><td>94.98</td></tr><tr><td>W/o sky segm. head</td><td>6.88</td><td>40.74</td><td>93.59</td></tr><tr><td>Euclidean-depth</td><td>6.34</td><td>36.91</td><td>95.48</td></tr><tr><td>W/o valid padding</td><td>6.45</td><td>37.00</td><td>95.73</td></tr><tr><td>W/o structural losses</td><td>6.11</td><td>36.72</td><td>95.93</td></tr><tr><td>Vanilla L1 loss</td><td>6.15</td><td>36.02</td><td>96.00</td></tr><tr><td>W/o perspective data</td><td>6.45</td><td>37.98</td><td>95.45</td></tr><tr><td>Ours</td><td>5.93</td><td>35.34</td><td>96.10</td></tr></table>

(a) Scale-invariant depth.

<table><tr><td>Method</td><td>AbsRel ↓</td><td>RMSE ↓</td><td> $\delta_1 \uparrow$ </td></tr><tr><td>Scalar</td><td>12.67</td><td>47.14</td><td>88.05</td></tr><tr><td>Dense, F=1</td><td>11.35</td><td>46.31</td><td>88.35</td></tr><tr><td>Dense, F=2</td><td>11.34</td><td>45.98</td><td>90.01</td></tr><tr><td>Dense, F=8</td><td>12.15</td><td>47.09</td><td>89.92</td></tr><tr><td>Dense, F=4 (Ours)</td><td>10.94</td><td>45.43</td><td>90.94</td></tr></table>

(b) Metric depth.

Table 5: Ablations for scale-invariant and metric depth on Stanford2D3DS.

Scale-Invariant Depth. Table 5a isolates the impact of individual architectural choices on the Stanford2D3DS dataset. Omitting explicit camera conditioning causes the most severe performance drop, increasing absolute relative error. Our log-space formulation also outperforms alternative linear-depth targets. Furthermore, decoupling infinite depth is critical; removing the auxiliary sky segmentation head causes a major spike in RMSE. Finally, simplifying the training objective or omitting structural data refinements, such as cross-face valid padding and joint perspective training, consistently degrades overall precision.

<table><tr><td>Method</td><td>Mean ↓</td><td>MSE ↓</td><td> $\delta_{5^{\circ}}$  ↑</td><td> $\delta_{22.5^{\circ}}$  ↑</td></tr><tr><td>UniFuse [15]</td><td>8.25</td><td>453.1</td><td>76.24</td><td>87.55</td></tr><tr><td>PanoFormer [38]</td><td>16.92</td><td>1053.5</td><td>59.13</td><td>75.50</td></tr><tr><td>OmniFusion [24]</td><td>20.70</td><td>832.0</td><td>28.51</td><td>63.55</td></tr><tr><td>MonoViT [55]</td><td>5.92</td><td>277.8</td><td>78.93</td><td>90.58</td></tr><tr><td>HyperSphere [18]</td><td>5.79</td><td>253.4</td><td>78.38</td><td>90.73</td></tr><tr><td>MTL [13]</td><td>8.98</td><td>469.0</td><td>72.51</td><td>86.02</td></tr><tr><td>PanoNormal [38]</td><td>5.56</td><td>246.6</td><td>79.18</td><td>91.01</td></tr><tr><td>PaGeR (Ours)</td><td>5.49</td><td>174.9</td><td>79.91</td><td>92.83</td></tr></table>

Table 3: Quantitative comparison of surface normals on the Structured3D dataset. All evaluated methods are in-domain architectures optimized directly on Structured3D.

![](images/fa3a6c9f12bf7bacaaaf5ce1d4c08ebf7311cbfdb3f9361bfdf9e01a183d7b5a.jpg)

<details>
<summary>text_image</summary>

RGB
MTL
Ours
GT
</details>

Figure 4: Qualitative comparison of panoramic surface normals estimation. Results from PaGeR and MTL (best available baseline method), shown alongside the RGB input and ground-truth depth on panoramas from the Structured3D dataset. (Best viewed zoomed in.)

Surface Normals. We validate our surface normal setup on the Structured3D dataset (Table 4b). Altering the encoder configuration proves highly detrimental; freezing the backbone causes a major spike in MSE, while optimizing a randomly initialized decoder yields the poorest threshold accuracy. Omitting the perceptual loss from the training objective causes a clear regression across all metrics, confirming that feature matching regularizers are necessary to prevent over-smoothing and recover sharp geometric boundaries.

Metric Scale. Table 5b evaluates our metric supervision strategy on the Stanford2D3DS dataset. Relying strictly on global scalar estimation rather than dense supervision across distributed anchor points yields the poorest baseline performance. Within our dense supervision framework, tuning the feature downsampling factor F is critical; both insufficient (F = 1) and extreme (F = 8) downsampling undermine accuracy. Our final configuration successfully balances these factors to achieve optimal metric depth reconstruction.

# 4 Related Work

Perspective Geometry Estimation. Monocular depth estimation has evolved into general-purpose architectures capable of robust zero-shot 3D geometry inference [35, 32, 50], recently augmented by diffusion priors [19, 6, 7] and unified geometric models [11, 45] for joint depth and normal prediction. Despite high structural fidelity, recovering absolute metric scale remains challenging. Naively regressing dense metric depth or localized scales [3, 48, 52] disrupts perspective priors, while directly regressing a single global scale factor [57] suffers from sparse gradients. However, for panoramic imagery with fixed 360° intrinsics, metric reconstruction mathematically reduces to estimating a single global scale. Exploiting this, we propose a framework that transfers rich perspective priors to panoramas, outputting scale-invariant geometry alongside a robustly decoupled global metric scale.

Visual Geometry Foundation Models. Visual Geometry Foundation Models (VGFMs) [46, 22, 43, 25] have shifted 3D reconstruction from task-specific estimators to unified, feed-forward architectures. By treating reconstruction as dense correspondence regression, they bypass traditional iterative Structure-from-Motion. A key advantage of VGFMs is their input flexibility, seamlessly transitioning from monocular to multi-view regimes and natively incorporating varying camera intrinsics or extrinsics. We build directly upon these robust, flexible priors, adapting their multi-view feed-forward capabilities to the panoramic domain.

Panoramic Geometry Estimation. Panoramic geometry estimation historically struggles with equirectangular distortions and data scarcity. Previous methods range from designing specialized architectures [42, 38, 12, 20] to adapting perspective models via multi-projection formats [44, 1]. Recently, training-free cubemap adaptations [17, 14, 16] mitigate distortions by predicting mutually consistent faces, though they often struggle to merge views back into a seamless equirectangular projection (ERP). Consequently, the field faces a trade-off: training-free multi-face methods exhibit seam artifacts, while fully-supervised continuous ERP models [23, 27] incur prohibitive training costs. Even concurrent efforts adapting VGFMs to 360° panoramas [10, 53] require extensive retraining. In contrast, our approach retains and reuses the original perspective VGFM priors, extending them to panoramas without requiring massive retraining or sacrificing expressivity.

# 5 Conclusion

We have presented PaGeR, a unified framework for panoramic geometry estimation that successfully lifts the robust representations of perspective depth models into the spherical domain. By combining a synchronized cubemap representation with separate decoding heads, our architecture transfers established perspective priors to multiple 360° geometric tasks without requiring extensive retraining. To guide this adaptation, we introduced a joint panoramic/perspective training regime. Extensive evaluations demonstrate that PaGeR establishes a new state of the art in zero-shot panoramic reconstruction across diverse indoor and outdoor environments. Although our instantiation leverages DA3, the architectural principles of our framework maintain compatibility with alternative geometric transformers [43, 47], offering a generalized methodology for bridging the domain gap between perspective and panoramic 3D scene understanding.

# 6 Limitations

Despite its strong performance, PaGeR shares several inherent constraints common to monocular geometry estimation. The model can produce unreliable predictions when encountering specular, reflective, or transparent surfaces, and it remains susceptible to depth ambiguities caused by complex material variations across different datasets.

Furthermore, while our cubemap adaptation implements cross-face padding and global attention mechanisms to harmonize features across the spherical manifold, subtle geometric or photometric misalignments can still emerge at face boundaries in rare, structurally complex scenes. Although these boundary artifacts are minimal and do not disrupt the global geometric layout, they indicate that unconstrained data-driven attention cannot always guarantee absolute boundary consistency, highlighting an avenue for future work involving explicit mathematical continuity constraints.

# Acknowledgments

We sincerely thank Hothifa Smair and the Parametra team for granting us written authorization to use the iCity Blender add-on for data generation, to train our models on the resulting renderings, and to release those renderings to the community under a non-commercial academic license; this contribution made the urban split of PanoInfinigen possible. We are also grateful to Veljko Bozic for his help in assembling the urban scenes with the iCity tool, which we subsequently rendered to produce the training data.

# References

[1] Hao Ai and Lin Wang. Elite360D: Towards efficient 360 depth estimation via semantic-and distance-aware bi-projection fusion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.   
[2] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Yuri Feigin, Peter Fu, Thomas Gebauer, Daniel Kurz, Tal Dimry, Brandon Joffe, Arik Schwartz, and Elad Shulman. ARKitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile RGB-D data. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2021.   
[3] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller. Zoedepth: Zero-shot transfer by combining relative and metric depth. preprint arXiv:2302.12288, 2023.   
[4] Zidong Cao, Jinjing Zhu, Weiming Zhang, Hao Ai, Haotian Bai, Hengshuang Zhao, and Lin Wang. PanDA: Towards panoramic depth anything with unlabeled panoramas and mobius spatial augmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.   
[5] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. In Advances in Neural Information Processing Systems (NeurIPS), 2014.   
[6] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. In European Conference on Computer Vision (ECCV), 2024.   
[7] Gonzalo Martin Garcia, Karim Abou Zeid, Christian Schmidt, Daan De Geus, Alexander Hermans, and Bastian Leibe. Fine-tuning image-conditional diffusion models is easier than you think. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025.   
[8] Clément Godard, Oisin Mac Aodha, and Gabriel J. Brostow. Unsupervised monocular depth estimation with left-right consistency. In CVPR, 2017.   
[9] Yuliang Guo, Sparsh Garg, S Mahdi H Miangoleh, Xinyu Huang, and Liu Ren. Depth any camera: Zero-shot metric depth estimation from any camera. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.   
[10] Yijing Guo, Mengjun Chao, Luo Wang, Tianyang Zhao, Haizhao Dai, Yingliang Zhang, Jingyi Yu, and Yujiao Shi. PanoVGGT: Feed-forward 3d reconstruction from panoramic imagery. preprint arXiv:2603.17571, 2026.   
[11] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zeroshot metric depth and surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.   
[12] Kun Huang, Fanglue Zhang, and Neil A Dodgson. PanoNormal: Monocular indoor 360◦ surface normal estimation. preprint arXiv:2405.18745, 2024.   
[13] Kun Huang, Fanglue Zhang, Fangfang Zhang, Yu-Kun Lai, Paul L Rosin, and Neil A Dodgson. Multi-task geometric estimation of depth and surface normal from monocular 360◦ images. preprint arXiv:2411.01749, 2024.   
[14] Yukun Huang, Yanning Zhou, Jianan Wang, Kaiyi Huang, and Xihui Liu. DreamCube: 3d panorama generation via multi-plane synchronization. preprint arXiv:2506.17206, 2025.   
[15] Hualie Jiang, Zhe Sheng, Siyu Zhu, Zilong Dong, and Rui Huang. Unifuse: Unidirectional fusion for 360◦ panorama depth estimation. IEEE Robotics and Automation Letters, 2021.   
[16] Dongki Jung, Jaehoon Choi, Yonghan Lee, and Dinesh Manocha. RPG360: Robust 360 depth estimation with perspective foundation models and graph optimization. In Advances in Neural Information Processing Systems (NeurIPS), 2026.   
[17] Nikolai Kalischek, Michael Oechsle, Fabian Manhardt, Philipp Henzler, Konrad Schindler, and Federico Tombari. CubeDiff: Repurposing diffusion-based image models for panorama generation. In International Conference on Learning Representations (ICLR), 2025.   
[18] Antonis Karakottas, Nikolaos Zioulis, Stamatis Samaras, Dimitrios Ataloglou, Vasileios Gkitsas, Dimitrios Zarpalas, and Petros Daras. 360◦ surface regression with a hyper-sphere loss. preprint arXiv:1909.07043, 2019.

[19] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.   
[20] Jongsung Lee, Harin Park, Byeong-Uk Lee, and Kyungdon Joo. HUSH: Holistic panoramic 3d scene understanding using spherical harmonics. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.   
[21] Leica Geosystems. Leica RTC360 3D Reality Capture Solution System Specification. Hexagon AB, 2026. Accessed: 2026-05-02.   
[22] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with MASt3R. In European Conference on Computer Vision (ECCV), 2024.   
[23] Haodong Li, Wangguangdong Zheng, Jing He, Yuhao Liu, Xin Lin, Xin Yang, Ying-Cong Chen, and Chunchao Guo. DA2: Depth anything in any direction. In International Conference on Learning Representations (ICLR), 2026.   
[24] Yuyan Li, Yuliang Guo, Zhixin Yan, Xinyu Huang, Ye Duan, and Liu Ren. OmniFusion: 360 monocular depth estimation via geometry-aware fusion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.   
[25] Haotong Lin, Sili Chen, Junhao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth Anything 3: Recovering the visual space from any views. In International Conference on Learning Representations (ICLR), 2026.   
[26] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.   
[27] Xin Lin, Meixi Song, Dizhe Zhang, Wenxuan Lu, Haodong Li, Bo Du, Ming-Hsuan Yang, Truong Nguyen, and Lu Qi. Depth Any Panoramas: A foundation model for panoramic depth estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.   
[28] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. preprint arXiv:1711.05101, 2019.   
[29] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 1, 2024.   
[30] Parametra. iCity, professional procedural city generation add-on for blender. https://parametra.net/, 2026. Accessed: 2026-05-02.   
[31] Luigi Piccinelli, Christos Sakaridis, Mattia Segu, Yung-Hsu Yang, Siyuan Li, Wim Abbeloos, and Luc Van Gool. UniK3D: Universal camera monocular 3d estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.   
[32] Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, and Luc Van Gool. Unidepthv2: Universal monocular metric depth estimation made simpler. preprint arXiv:2502.20110, 2025.   
[33] Alexander Raistrick, Lahav Lipson, Zeyu Ma, Lingjie Mei, Mingzhe Wang, Yiming Zuo, Karhan Kayan, Hongyu Wen, Beining Han, Yihan Wang, Alejandro Newell, Hei Law, Ankit Goyal, Kaiyu Yang, and Jia Deng. Infinite photorealistic worlds using procedural generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.   
[34] Alexander Raistrick, Lingjie Mei, Karhan Kayan, David Yan, Yiming Zuo, Beining Han, Hongyu Wen, Meenal Parakh, Stamatis Alexandropoulos, Lahav Lipson, Zeyu Ma, and Jia Deng. Infinigen indoors: Photorealistic indoor scenes using procedural generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.   
[35] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(3):1623–1637, 2020.

[36] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In IEEE/CVF International Conference on Computer Vision (ICCV), 2021.   
[37] Manuel Rey-Area, Mingze Yuan, and Christian Richardt. Matterport3D 360◦ RGBD dataset. https: //researchdata.bath.ac.uk/1126/, 2022.   
[38] Zhijie Shen, Chunyu Lin, Kang Liao, Lang Nie, Zishuo Zheng, and Yao Zhao. PanoFormer: panorama transformer for indoor 360◦ depth estimation. In European Conference on Computer Vision (ECCV), 2022.   
[39] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations (ICLR), 2015.   
[40] Stanford Doerr School of Sustainability Data Repository. Stanford 2D-3D-Semantics dataset (2D-3D-S). https://sdss.redivis.com/datasets/f304-a3vhsvcaf?v=1.0, 2024.   
[41] Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, and M. Jorge Cardoso. Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations, page 240–248. Springer International Publishing, 2017.   
[42] Fu-En Wang, Yu-Hsuan Yeh, Yi-Hsuan Tsai, Wei-Chen Chiu, and Min Sun. Bifuse++: Self-supervised and efficient bi-projection fusion for 360 depth estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5):5448–5460, 2022.   
[43] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.   
[44] Ning-Hsu Albert Wang and Yu-Lun Liu. Depth anywhere: Enhancing 360 monocular depth estimation via perspective distillation and unlabeled data augmentation. In Advances in Neural Information Processing Systems (NeurIPS), 2024.   
[45] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. MoGe: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.   
[46] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3d vision made easy. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.   
[47] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π3: Permutation-equivariant visual geometry learning, 2026.   
[48] Chengrui Wei, Meng Yang, Lei He, and Nanning Zheng. FS-Depth: Focal-and-scale depth estimation from a single image in unseen indoor scene. IEEE Transactions on Circuits and Systems for Video Technology, 34(11), 2024.   
[49] Tao Wen, Jiepeng Wang, Yabo Chen, Shugong Xu, Chi Zhang, and Xuelong Li. Metric-solver: Sliding anchored metric depth estimation from a single image, 2025.   
[50] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.   
[51] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. ScanNet++: A high-fidelity dataset of 3d indoor scenes. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.   
[52] Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3d: Towards zero-shot metric 3d prediction from a single image. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023.   
[53] Jiayi Yuan, Haobo Jiang, De Wen Soh, and Na Zhao. VGGT-360: Geometry-consistent zero-shot panoramic depth estimation. preprint arXiv:2603.18943, 2026.   
[54] Ilwi Yun, Chanyong Shin, Hyunku Lee, Hyuk-Jae Lee, and Chae Eun Rhee. EGformer: Equirectangular geometry-biased transformer for 360 depth estimation. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[55] Chaoqiang Zhao, Youmin Zhang, Matteo Poggi, Fabio Tosi, Xianda Guo, Zheng Zhu, Guan Huang, Yang Tang, and Stefano Mattoccia. MonoViT: Self-supervised monocular depth estimation with a vision transformer. In 2022 International Conference on 3D Vision (3DV), 2022.   
[56] Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3D: A large photo-realistic dataset for structured 3d modeling. In European Conference on Computer Vision (ECCV), 2020.   
[57] Ruijie Zhu, Chuxin Wang, Ziyang Song, Li Liu, Jianfeng He, Jiacheng Deng, Tianzhu Zhang, and Yongdong Zhang. ScaleDepth: Decomposing metric depth estimation into semantic-aware scale prediction and adaptive relative depth estimation. IEEE Transactions on Circuits and Systems for Video Technology, page 1–1, 2026.

# A PanoInfinigen

High-quality datasets are the cornerstone of robust monocular depth estimation. While perspective depth estimation has benefited immensely from large-scale, diverse data collections, the panoramic domain suffers from limited training data. Existing standard benchmarks, such as Stanford2D3DS [40] and Matterport3D360 [37], rely on real-world scanners. Consequently, they are restricted to static indoor environments and often contain acquisition artifacts, such as missing depth values in reflective or distant regions. Synthetic alternatives like Structured3D [56] offer dense, complete ground truth but are similarly confined to low resolutions and indoor scenes with limited variability. To address this gap, we introduce PanoInfinigen, a large-scale synthetic dataset designed to unlock the potential of high-resolution, general-purpose panoramic depth estimation.

Data Generation and Characteristics. PanoInfinigen provides complete, high-quality ground truth for both depth and surface normals across a variety of indoor and outdoor scenes. The dataset is intended to support the development of high-performance models; moreover, it is extensible through our accompanying open-source generation tool. To construct PanoInfinigen, we build upon Infinigen [33, 34], a procedural content generation framework capable of synthesizing realistic, unbounded 3D environments. In contrast to methods that depend on fixed 3D asset libraries, Infinigen procedurally generates both geometry and textures, thereby enabling effectively unlimited variability in scene layout, object placement, and illumination conditions. We extend its rendering pipeline to support 360° equirectangular projection and generate 70,000 unique panoramas from 20,000 distinct scenes.

To extend coverage to urban environments, we employ the iCity [30] procedural city-generation add-on for Blender, which automates the creation of realistic and diverse synthetic urban scenes. We synthesize 20 urban environments with varying configurations and visual styles, and render approximately 400 panoramas per city using our existing Infinigen-based auto-rendering tool. This results in roughly 7,000 high-quality outdoor panoramas, each accompanied by dense depth and surface normal maps.

The resulting dataset spans a vast range of semantic domains, from standard indoor environments (e.g., kitchens, bedrooms) to complex natural landscapes (e.g., forests, deserts) and urban surroundings (e.g., skyscrapers, small houses, historical buildings, parks). Crucially, every sample is rendered at native 4K resolution with complete, pixel-perfect ground truth for metric depth and surface normals. Examples of the dataset are shown in Fig. 5.

# B ZüriPano

Outdoor panoramic data collection introduces significant complexities absent in indoor environments, primarily due to dynamic occlusions (e.g., pedestrians, vehicular traffic, and urban activity) and the limited availability of high-resolution, long-range sensing hardware. To date, these challenges have hindered the development of a reliable, LiDAR-based outdoor panoramic evaluation benchmark.

To address this gap, we employ the Leica RTC360 LiDAR scanner [21], a high-performance reality capture system capable of generating panoramas at 8K resolution. The device features an effective operating range of 130 meters and utilizes advanced HDR imaging and automated double-scan routines to resolve transient occlusions and disocclusions effectively. Our collection encompasses 100 panoramic scans across 11 distinct urban locations in Zürich, Switzerland, capturing a diverse array of architectural styles and open-space geometries.

During the post-processing stage, we meticulously filter the data to ensure metric reliability. Infinite depth regions, such as the sky, and areas of high specular reflectance (e.g., glass facades) are masked out, resulting in a dense depth map and a corresponding validity mask for every panorama. We believe this dataset provides a rigorous testbed for evaluating the robustness of panoramic depth estimation models, specifically regarding long-range accuracy and structural consistency in complex outdoor environments.

![](images/f7e697182faad266945b49520dfb0dd1fc34c51e037b1a8169be5b62590382a3.jpg)

<details>
<summary>natural_image</summary>

Grid of 3D-rendered architectural and interior designs with varying lighting and color schemes (no text or symbols)
</details>

Figure 5: Examples of RGB, depth, and surface normal panoramas from our PanoInfinigen dataset.

![](images/493bc5b08d1f45bb089c6c802d9dbb1afeedcd396052d3753068f41a1508f9e4.jpg)

<details>
<summary>natural_image</summary>

Four-panel composite image showing urban and commercial architectural designs with 3D overlays of trees, buildings, and a parking lot (no text or symbols)
</details>

Figure 6: Samples of our ZüriPano dataset.

# C Seam Consistency Metrics

To quantify the geometric consistency of our cubemap projections, we evaluate seam artifacts at three distinct granularities: Seam Defect Density (SDD), Seam Prevalence (SP), and Seam Severity (SS). These metrics monitor depth discontinuities across the $N = 1 2$ shared face boundaries of the cubemap.

Let $\mathcal { E }$ be the set of all pixel pairs $( p , q )$ that are spatially adjacent across a cubemap boundary, and let $\{ E _ { k } \} _ { k = 1 } ^ { N }$ be the partition of $\mathcal { E }$ into N disjoint sets representing each edge. For any pair $( p , q ) \in \mathcal { E }$ we define the log-depth jump as:

$$
\Delta_ {p q} = \left| \log \hat {d} _ {p} - \log \hat {d} _ {q} \right| \tag {4}
$$

where $\hat { d } _ { p }$ and $\hat { d } _ { q }$ are the Euclidean ERP linear depths at pixels p and $q ,$ respectively.

Seam Defect Density (SDD) measures the global frequency of artifacts by calculating the fraction of total boundary pixels whose depth jump exceeds a tolerance threshold τ :

$$
\mathrm{SDD} = \frac {1}{| \mathcal {E} |} \sum_ {(p, q) \in \mathcal {E}} \mathbb {I} (\Delta_ {p q} > \tau) \tag {5}
$$

Seam Prevalence (SP) assesses the distribution of defects across the cubemap structure. An edge $E _ { k }$ is considered prevalent with defects if more than 10% of its constituent pixels exceed τ . The $\mathrm { S P }$ metric is the fraction of such "corrupted" edges:

$$
\mathrm{SP} = \frac {1}{N} \sum_ {k = 1} ^ {N} \mathbb {I} \left(\frac {1}{| E _ {k} |} \sum_ {(p, q) \in E _ {k}} \mathbb {I} (\Delta_ {p q} > \tau) > 0. 1\right) \tag {6}
$$

Seam Severity (SS) captures systemic geometric misalignment by measuring the fraction of edges whose mean jump across the entire boundary exceeds a strict magnitude threshold $\gamma \colon$

$$
\mathrm{SS} = \frac {1}{N} \sum_ {k = 1} ^ {N} \mathbb {I} \left(\frac {1}{| E _ {k} |} \sum_ {(p, q) \in E _ {k}} \Delta_ {p q} > \gamma\right) \tag {7}
$$

where I(·) is the indicator function that equals 1 if the condition is true and 0 otherwise.

In plain words, Seam Defect Density (SDD) measures the local density of pixel-level defects; Seam Prevalence (SP) captures geometric coverage by identifying how many of the twelve edges show visible artifacts; and Seam Severity (SS) identifies systemic collapses by measuring the fraction of edges with high average depth jumps. This triplet allows us to distinguish between widespread, mild jitter and isolated catastrophic failures.

# D Performance Trade-offs of Joint vs. Independent Model Training

While our unified panorama estimation model provides the significant advantage of simultaneously outputting scale-invariant (SI) depth, metric depth, surface normals, and sky segmentation from a shared ViT backbone, this multi-task setting inherently requires a compromise in individual task performance. To quantify the impact of joint training and analyze the network’s capacity limits, we conducted an ablation study comparing our unified framework against independently trained models.

For this ablation, we trained isolated models for SI depth, surface normals, and a full metric depth model. The independent full metric depth model was trained using dense supervision without depth alignment, thereby directly outputting metric depth.

As expected in multi-task architectures, the results demonstrate that the independent SI depth and surface normal models modestly outperform their counterparts in the unified model. However, the most substantial performance gap is observed between our independent full metric depth model and the metric scale head within the unified setting.

<table><tr><td rowspan="2">Model</td><td colspan="3">Matterport3D360</td><td colspan="3">Stanford2D3DS</td><td colspan="3">ZüriPano</td></tr><tr><td>AbsRel↓</td><td>RMSE↓</td><td> $\delta_1\uparrow$ </td><td>AbsRel↓</td><td>RMSE↓</td><td> $\delta_1\uparrow$ </td><td>AbsRel↓</td><td>RMSE↓</td><td> $\delta_1\uparrow$ </td></tr><tr><td colspan="10">SI Depth</td></tr><tr><td>Unified</td><td>9.67</td><td>64.69</td><td>90.87</td><td>5.93</td><td>35.34</td><td>96.10</td><td>9.36</td><td>299.61</td><td>94.75</td></tr><tr><td>Specialized</td><td>9.57</td><td>63.22</td><td>91.02</td><td>5.87</td><td>34.07</td><td>96.15</td><td>9.07</td><td>274.92</td><td>95.31</td></tr><tr><td colspan="10">Metric Depth</td></tr><tr><td>Unified</td><td>21.83</td><td>123.48</td><td>69.50</td><td>10.94</td><td>45.43</td><td>90.94</td><td>31.97</td><td>530.85</td><td>39.30</td></tr><tr><td>Specialized</td><td>17.71</td><td>91.27</td><td>86.56</td><td>10.01</td><td>46.28</td><td>92.07</td><td>18.19</td><td>335.90</td><td>77.90</td></tr></table>

(a) Depth Estimation (SI and Metric)

<table><tr><td rowspan="2">Model</td><td colspan="4">Structured3D</td></tr><tr><td>Mean↓</td><td>MSE↓</td><td>5°↑</td><td>22.5°↑</td></tr><tr><td colspan="5">Surf. Normals</td></tr><tr><td>Unified</td><td>5.49</td><td>174.9</td><td>79.91</td><td>92.83</td></tr><tr><td>Specialized</td><td>4.96</td><td>157.4</td><td>82.18</td><td>93.53</td></tr></table>

(b) Surface Normals   
Table 6: Unified vs. Specialized Models. We compare our multi-task unified PaGeR model against individually trained specialized models for each modality. (a) Evaluation of Scale-Invariant (SI) and Metric depth across three datasets. (b) Surface normal estimation evaluated on the Structured3D dataset. Top-performing metrics for each modality pair are highlighted in bold.

This discrepancy is directly tied to gradient flow and backbone freezing constraints. For the independent full metric model, the ViT backbone is fully unlocked, allowing the network to extract metric-specific cues early in the encoder stages. This deep integration greatly enhances metric depth accuracy and effectively obviates the need for separate, heavy metric heads.

Conversely, in the unified model, the ViT backbone must remain frozen with respect to the metric scale head. We found that allowing gradients from the metric scale head to flow back into the shared ViT during joint training fundamentally conflicts with the optimization of the other tasks, leading to a severe degradation in the quality of both SI depth and surface normal predictions. Thus, keeping the ViT frozen preserves the integrity of the SI depth and normals but creates a bottleneck for metric depth performance.

These findings suggest that while our shared representation is highly efficient, it is currently capacityconstrained when balancing the extraction of both scale-invariant and purely metric cues. A careful adaptation of the ViT backbone—such as introducing task-specific routing or slightly expanding capacity to better accommodate these competing gradients—represents a highly promising direction for future work.

# E Loss Weights and Hyperparameters

To facilitate full reproducibility and provide transparency regarding our training setup, we detail the complete configuration and computational infrastructure of our framework. This includes scaling factors for the scale-invariant depth, metric scale, surface normal, and sky segmentation heads. For clarity and ease of reference, we present the precise loss weights λ, optimization schedules, learning rates, and architectural choices across all modalities in Table 7.

<table><tr><td>Modality</td><td>Loss Term</td><td>Symbol</td><td>Weight</td></tr><tr><td rowspan="4">Depth</td><td>Confidence L1 Loss</td><td> $\lambda_{L1}$ </td><td>1.0</td></tr><tr><td>Lambda Confidence</td><td> $\lambda_c$ </td><td>0.2</td></tr><tr><td>Gradient Loss</td><td> $\lambda_{grad}$ </td><td>40.0</td></tr><tr><td>Normals Consistency Loss</td><td> $\lambda_{norm}$ </td><td>0.6</td></tr><tr><td rowspan="2">Normals</td><td>Cosine Similarity Loss</td><td> $\lambda_{cos}$ </td><td>1.0</td></tr><tr><td>Perceptual Loss</td><td> $\lambda_{perc}$ </td><td>0.5</td></tr><tr><td rowspan="3">Metric Scale</td><td>Confidence L1 Loss</td><td> $\lambda_{L1}$ </td><td>1.0</td></tr><tr><td>Lambda Confidence</td><td> $\lambda_c$ </td><td>0.2</td></tr><tr><td>DPT Subsampling Factor</td><td>F</td><td>4.0</td></tr><tr><td rowspan="3">Sky Segmentation</td><td>Binary Cross-Entropy Loss</td><td> $\lambda_{BCE}$ </td><td>1.0</td></tr><tr><td>Focal Loss</td><td> $\lambda_{Focal}$ </td><td>0.4</td></tr><tr><td>Dice Loss</td><td> $\lambda_{Dice}$ </td><td>1.0</td></tr></table>

Table 7: Hyperparameter configuration for our default model.

# F Computational Efficiency and Resource Benchmark

To evaluate the practical utility and scalability of PaGeR, we benchmark its computational footprint against other baselines. Table 8 provides a comprehensive comparison across four key performance dimensions: inference runtime, peak memory consumption during evaluation, native processing resolution, and total training data requirements.

<table><tr><td>Method name</td><td>Runtime [s]</td><td>Peak Memory [GB]</td><td>Resolution [px]</td><td>Data size [ $10^3$ ]</td></tr><tr><td>DreamCube</td><td>6.06</td><td>15.2</td><td>1024×2048</td><td>N/A</td></tr><tr><td>DepthAnyCamera</td><td>0.11</td><td>0.3</td><td>512×1024</td><td>800</td></tr><tr><td>MoGe</td><td>36.05</td><td>3.6</td><td>1024×2048</td><td>8860</td></tr><tr><td>DAP</td><td>0.10</td><td>3.7</td><td>512×1024</td><td>1700</td></tr><tr><td>RPG360</td><td>5.83</td><td>13.2</td><td>512×1024</td><td>N/A</td></tr><tr><td>EGformer</td><td>0.31</td><td>1.2</td><td>512×1024</td><td>70</td></tr><tr><td>UniK3D</td><td>0.41</td><td>3.3</td><td>560×1106</td><td>700</td></tr><tr><td>PanDA</td><td>0.63</td><td>3.5</td><td>504×1008</td><td>120</td></tr><tr><td> $DA^2$ </td><td>0.05</td><td>2.8</td><td>546×1092</td><td>606</td></tr><tr><td>Ours</td><td>0.48</td><td>12.8</td><td>1008×2016</td><td>100</td></tr></table>

Table 8: Comparison of various methods based on performance metrics.

# G Additional Qualitative Examples

![](images/20fae0016ae17c9f1e5638d609bb0d9f90b5cdfc093f6ee625be42b322fffb50.jpg)

<details>
<summary>text_image</summary>

Matterport3D360
Stanford2D3DS
ZüriPano
Input
PanDA
UniK3D
DAP
DA²
Ours
GT
</details>

Figure 7: Qualitative comparison of panoramic depth estimation.. Visual results from PaGeR, compared to the subset of the best evaluated baselines, shown alongside the RGB input and groundtruth depth on Matterport3D360, Stanford2D3DS, and ZüriPano panoramas. (Best viewed zoomed in.)

![](images/31d0822d283efa40021ddb521c981013573eb29e883d0989a67eabd2bc29f64c.jpg)

<details>
<summary>natural_image</summary>

Six-panel comparison of interior lighting and thermal imaging, showing different room layouts and heatmaps (no text or symbols)
</details>

Figure 8: Comparison to vanilla DA3.

![](images/5a05a48bb0c3cdeed6003c3cfebe7a3db944e6df5ac40fbafc9ed1f163151fea.jpg)

Figure 9: Qualitative point-cloud comparison. Indoor scenes (top) and an outdoor scene (bottom) are rendered as point clouds alongside the corresponding panoramic input images for competitors and our method. For the indoor examples, we show our point cloud reconstruction with zoomed-in novel-view rendering comparison to the main competitors, highlighted by red boxes.   
![](images/60df714175efc8dd00cafaed25c0562d0c8583523c46ac6b576e81a93451eeae.jpg)  
Figure 10: Examples of measured distances in our predicted point cloud. The measures are given in meters.

![](images/305264bc67ae982ca7cd10637e71a211cc1cf7bbdf8ed707f7b7ff42ae265374.jpg)  
Figure 11: Qualitative comparison of panoramic surface normals estimation. Visual results from PaGeR and MTL (best available baseline method), shown alongside the RGB input and ground-truth depth on panoramas from the Structured3D dataset. (Best viewed zoomed in.)