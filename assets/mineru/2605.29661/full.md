# Geometry-Guided Modeling of Foundation Features Enables Generalizable Object Shape Deformation Learning

Yiyao Ma 1 Kai Chen 1 Zhongxiang Zhou 2 Zhuheng Song 1 Dongsheng Xie 1 Zelong Tan 1 Rong Xiong 2 3 Qi Dou 1

# Abstract

Monocular 3D shape recovery is fundamental to geometric understanding, yet achieving robust generalization across arbitrary viewpoints and unseen object categories remains a significant challenge. In this paper, we present a generalizable deformation learning framework that reconstructs 3D objects by explicitly deforming a category-level shape template to match the target observation. To address complex shape variations between the template and the target, we introduce a geometry-guided feature modeling mechanism. This process first enriches foundation features with template topology to yield a geometry-aware representation, which is then explicitly correlated with the target observation to guide precise deformation. Furthermore, to bridge the disparity between the fixed template and arbitrary target views, we propose a view-adaptive feature aggregation module. This module leverages multi-view template features and their corresponding camera poses to enrich the canonical template representation, ensuring robust feature alignment regardless of the target’s perspective. Extensive experiments demonstrate that our approach significantly outperforms state-of-theart methods in handling large shape variations and diverse viewpoints, exhibiting strong generalization to novel categories and effectively supporting downstream real-world dexterous robotic manipulation tasks. Project homepage: https: //GODeform.github.io/

1Department of Computer Science and Engineering, The Chinese University of Hong Kong, Hong Kong SAR, China 2Zhejiang Innovation Center for Humanoid Robotics, Ningbo, China 3State Key Laboratory of Industrial Control and Technology, Zhejiang University, Hangzhou, China. Correspondence to: Kai Chen <kaichen@cuhk.edu.hk>.

Preprint. May 29, 2026.

# 1. Introduction

Recovering object shape from a monocular image is fundamental to geometric understanding and spatial reasoning (Mescheder et al., 2019; Fahim et al., 2021). However, relying on a single viewpoint impedes the reconstruction of plausible geometry for invisible regions, while the vast structural complexity of the real world demands strong generalization capabilities to handle shapes beyond the training distribution (Wang et al., 2024; Jiang et al., 2025). Despite growing attention in recent years (Hong et al., 2024; Long et al., 2024; Wang et al., 2025), robust and generalizable shape recovery across viewpoints and diverse object categories remains a critical challenge (Cho et al., 2025).

Recent advances leverage generative models to achieve highfidelity reconstruction (Long et al., 2024). However, their performance is often sensitive to viewpoint variations and partial observability. In self-occluded regions, they frequently hallucinate geometry that is unrealistic or inconsistent with visible surfaces (Huang et al., 2025; Wu et al., 2025). While some works further incorporate a categorylevel template to inject structural priors (Xiu et al., 2022; Zheng et al., 2021; Wang et al., 2025), they typically treat the template merely as a source of feature conditioning rather than a geometric starting point. Without explicitly establishing point-wise relationships to the template, they struggle to effectively leverage the template’s topology to regularize the reconstruction, and thus remain prone to structural degradation in invisible parts (Xiu et al., 2023).

To address these structural issues, another line of research focuses on explicitly deforming a template shape to match the target observation (Wang et al., 2018; Wen et al., 2019; Sommer et al., 2025; Shuai et al., 2023). By warping a known geometry, these methods leverage the template’s structural priors to ensure plausibility in occluded regions, while inherently preserving the template’s topology to maintain dense correspondence (Groueix et al., 2018; Chen et al., 2025; Liu et al., 2023a). However, these approaches typically rely on visual encoders trained from scratch on limited datasets, which struggle to extract robust semantic features for unseen categories (Wallace & Hariharan, 2019). Furthermore, even within seen categories, they lack a deep understanding of the intrinsic geometric relationship between the template and diverse novel objects, and thus can fail to predict accurate deformation fields when the target shape deviates significantly from the template (Pan et al., 2019; Zhang et al., 2024; Di et al., 2024). These limitations motivate the need for a deformation learning framework that generalizes across viewpoints, large template-target shape variations, and object categories beyond training.

![](images/cbf0497c432359be57aa0e830cedad70abcfa9c4dbb0f85826fa9d9a20e0b411.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Object Shape Deformation Learning"] --> B["Target Image"]
    A --> C["Template"]
    A --> D["Deformation"]
    A --> E["Reconstruction"]
    F["Generalizable to Large Shape Variations"] --> G["Baseline"]
    F --> H["Ours"]
    I["Viewpoint Changes"] --> J["Baseline"]
    I --> K["Ours"]
    L["Novel Object Categories"] --> M["GT"]
    L --> N["Ours"]
    
    O["Applications"] --> P["Image-based Shape Reconstruction"]
    O --> Q["Contact Map Transfer"]
    O --> R["Grasp Transfer"]
    
    S["Dexterous Grasping"] --> T["Dustpan"]
    S --> U["Bowl"]
    S --> V["Lotion pump"]
    S --> W["Milk"]
    
    X["Real-world Robotic Manipulation"] --> Y["Dustpan"]
    X --> Z["Bowl"]
```
</details>

Figure 1. The proposed object shape deformation learning framework can handle large template-target shape variations, remains robust to diverse camera viewpoints, and generalizes to unseen categories. It enables various downstream applications, and effectively supports generalizable dexterous robotic manipulation in the real world.

In this paper, we present a generalizable deformation learning framework enabled by geometry-guided modeling of foundation features. Our core insight is to harness the robust semantic similarity provided by 2D foundation models (Simeoni et al. ´ , 2025) to guide 3D shape deformation. The pre-trained foundation features offer a consistent representation across categories and domains, strengthening robustness under distribution shifts and supporting generalization to unseen categories. To handle complex shape variations between the template and diverse target objects, we introduce a geometry-guided feature propagation and alignment module. Specifically, we build a geometry-aware template representation by enriching 2D foundation features with template shape features and propagating them over the template surface to cover both visible and occluded regions. We then explicitly model the relationship between the resulting template representation and the target observation to condition the inference of a per-point deformation field. Together, these components enable the model to capture fine-grained similarities between the template and diverse targets, yielding deformation predictions that generalize across viewpoints, shape variations, and object categories.

However, the efficacy of such feature guidance is challenged when there is a significant viewpoint disparity between the fixed template view and the target observation. Such misalignment diminishes the semantic correspondence captured by foundation models, potentially degrading deformation quality. To achieve robustness across arbitrary observation

angles, we introduce a view-adaptive feature aggregation module to upgrade the template representation. Instead of relying on a static template view, this module dynamically identifies a primary view from a pre-sampled support set. We then enhance this primary representation by injecting complementary cues from auxiliary template views via a pose-aware attention mechanism that explicitly accounts for relative camera geometry. The resulting fused representation can effectively mitigate perspective-induced feature drift and stabilize deformation prediction even under large viewpoint changes. As illustrated in Figure 1, our proposed deformation learning framework proves robust to large shape variations, viewpoint changes, and novel object categories, thereby facilitating a wide range of downstream applications. Our contributions are summarized as follows:

• We propose a generalizable deformation learning framework that reconstructs 3D objects by explicitly deforming category-level templates, leveraging the robust visual representations of 2D foundation models to generalize to unseen categories.   
• We introduce a geometry-guided feature modeling mechanism that enriches foundation features with template topology to establish precise point-wise correspondences, effectively addressing complex shape variations between the template and the target.   
• We design a view-adaptive feature aggregation module that dynamically incorporates camera pose embeddings to compensate for perspective-induced geometric misalignments, ensuring robustness against significant viewpoint disparities.   
• Extensive experiments show that our method generalizes well across diverse viewpoints, large shape variations, and novel object categories, while effectively facilitating generalizable dexterous robotic manipulation in the real world.

# 2. Related Work

# 2.1. Object Shape Reconstruction via Deformation

A prevalent paradigm in category-level 3D reconstruction involves deforming a canonical shape template to recover the target geometry. Early approaches primarily focused on sparse deformation mechanisms, employing lowdimensional controls such as cages (Yifan et al., 2020), semantic keypoints (Jakab et al., 2021), or volumetric warps (Jack et al., 2018; Kurenkov et al., 2018) to constrain the solution space, while others explored part-based deformation (Shuai et al., 2023; Paschalidou et al., 2021) to handle structural variations. To capture more fine-grained geometric details beyond these sparse controls, subsequent research shifted towards dense deformation strategies, predicting high-dimensional fields like point-wise offsets (Wang et al., 2019) or pixel-aligned displacements (Wang et al., 2018; Wen et al., 2019) to refine local surface variations. To further improve reconstruction fidelity, recent works propose retrieval-augmented pipelines that first retrieve shape templates visually similar to the target from a library, followed by deformation to fit the observed images (Di et al., 2023; Zhang et al., 2024; Di et al., 2024; Uy et al., 2021). However, the performance of these methods is heavily contingent on the geometric similarity between the retrieved template and the target object. Furthermore, these deformation models typically fail to generalize beyond their training categories, limiting their practical usage in open-world scenarios. In contrast, our method explicitly models foundation features to guide the shape deformation learning, thereby enabling superior generalizability across diverse shape variations and unseen object categories.

# 2.2. Object Representation with Foundation Models

Large-scale pre-trained 2D foundation models have demonstrated strong generalization across categories and domains (Radford et al., 2021; Oquab et al., 2024; Simeoni´ et al., 2025), and their representations have been widely adopted as generic visual descriptors for diverse downstream tasks (Chen et al., 2024b; Barsellotti et al., 2025; Lin et al., 2024). Despite these advances in 2D, extending such capabilities to 3D remains challenging. Although recent efforts explore native 3D foundation models (Pang et al., 2023; Xue et al., 2023; Liu et al., 2023b), their scalability is fundamentally constrained by the scarcity and limited diversity of high-quality 3D data. As a result, they often exhibit limited generalization ability and a constrained understanding of complex semantic concepts compared to their 2D counterparts (Thengane et al., 2025). To bridge this gap, a growing body of work adapts pre-trained 2D foundation models for 3D understanding by lifting and aligning 2D features with geometric representations, achieving promising results in scene understanding (Knaebel et al., 2026; Zhu et al., 2024; 2025; Zhu et al.), pose estimation (Caraffa et al., 2024; Chen et al., 2024a), and robotic manipulation (Huang et al., 2023; Chen et al., 2026). Inspired by these successes, we study generalizable object shape deformation by making 2D foundation features geometry-aware on the template surface, enabling robust point-wise template-target correspondence reasoning.

# 3. Methodology

We formulate the task as a geometry-guided shape deformation problem. Given a single RGB image $I _ { T }$ of a target object and a category-level 3D template $\bar { \mathcal { S } } \in \mathbb { R } ^ { N \times 3 }$ , our goal is to predict a per-point deformation field $\mathcal { D } \in \mathbb { R } ^ { N \times 3 }$ that aligns the template geometry with the observed target shape. Unlike standard reconstruction approaches, this formulation yields both the recovered 3D shape and explicit point-wise correspondences between the template and the target. To achieve this, we model the deformation process as a conditional flow matching problem. We define the conditioning context as $\mathbf { c } = \{ I _ { T } , I _ { S } \}$ , where $I _ { \mathcal { S } }$ denotes the associated template image. Our objective is to learn a continuous deformation mapping Φ that transports the geometric distribution of the template $s$ to the target ${ \hat { \mathcal { T } } } = \Phi ( S \mid \mathbf { c } )$ . This ensures that the reconstruction originates from a topologically plausible prior while effectively recovering instance-specific fine details through the learned flow.

# 3.1. Shape Deformation Learning via Flow Matching

# 3.1.1. FLOW MATCHING FORMULATION

Flow matching provides a framework for learning continuous deformation paths from a template S to a target T under context c. The flow is governed by an ordinary differential equation (ODE) $d \phi _ { t } / d t = \mathbf { v } _ { t } ( \phi _ { t } \mid \mathbf { c } _ { t }$ ), transporting the distribution from $p ( \mathbf { x } \mid S )$ to $p ( \mathbf { x } \mid \tau )$ . Following (Albergo & Vanden-Eijnden, 2023; Liu et al., 2023d), we construct a linear interpolation path ${ \bf x } _ { t } = ( 1 - t ) { \bf x } _ { 0 } + t { \bf x } _ { 1 }$ between correspondences, which implies a constant target velocity ${ \bf u } _ { t } = { \bf x } _ { 1 } - { \bf x } _ { 0 }$ . We train a network $v _ { \theta } ( \mathbf { x } _ { t } , t , \mathbf { c } )$ to approximate this velocity field. At inference time, we achieve efficiency by fixing t = 0 and directly predicting the deformation D:

$$
\mathcal {D} = v _ {\theta} (\mathcal {S}, 0, \mathbf {c}), \quad \hat {\mathcal {T}} = \mathcal {S} + \mathcal {D}. \tag {1}
$$

This enables single-step inference while maintaining the theoretical guarantees of continuous flow matching.

# 3.1.2. GEOMETRY-GUIDED FOUNDATION FEATURE MODELING

Guiding object shape deformation using 2D images of template and target objects presents a significant challenge, as the conditioning process requires robustness to target variations and the complex discrepancies between template and target geometries. Our core insight lies in leveraging the consistent representations of 2D foundation models pre-trained on large-scale datasets, modeling these foundation features via geometry-guided mechanisms. By explicitly aligning robust 2D features with 3D geometry, we aim to maintain a stable and informative conditioning signal, thereby facilitating generalizable deformation learning across diverse object categories.

![](images/2b3ae7291a12bc8565738d8ed3dffb575741e38b63073ad01752cca30df7e677.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Target Observation"] --> B["2D Foundation Model"]
    C["Template 3D shape"] --> D["Shape Encoder"]
    D --> E["Feature Propagation"]
    E --> F["Feature Alignment"]
    F --> G["Point Transformer V3"]
    G --> H["Deformation Field"]
    H --> I["Reconstruction"]
    I --> J["Training Objectives"]
    J --> K["Deformation Module"]
    
    subgraph Feature Aggregation
        L["View-Adaptive Feature Aggregation"]
        M["Main view selection"]
        N["Feature Lifting"]
        O["Pose-aware Template Feature"]
    end
    
    style A fill:#99ccff,stroke:#333
    style C fill:#99ccff,stroke:#333
    style D fill:#99ccff,stroke:#333
    style E fill:#99ccff,stroke:#333
    style F fill:#99ccff,stroke:#333
    style G fill:#99ccff,stroke:#333
    style H fill:#99ccff,stroke:#333
    style I fill:#99ccff,stroke:#333
    style J fill:#99ccff,stroke:#333
    style K fill:#99ccff,stroke:#333
    style L fill:#99ccff,stroke:#333
    style M fill:#99ccff,stroke:#333
    style N fill:#99ccff,stroke:#333
    style O fill:#99ccff,stroke:#333
    style P fill:#99ccff,stroke:#333
    style Q fill:#99ccff,stroke:#333
    style R fill:#99ccff,stroke:#333
    style S fill:#99ccff,stroke:#333
    style T fill:#99ccff,stroke:#333
    style U fill:#99ccff,stroke:#333
    style V fill:#99ccff,stroke:#333
    style W fill:#99ccff,stroke:#333
    style X fill:#99ccff,stroke:#333
    style Y fill:#99ccff,stroke:#333
    style Z fill:#99ccff,stroke:#333
    style AA fill:#99ccff,stroke:#333
    style AB fill:#99ccff,stroke:#333
    style AC fill:#99ccff,stroke:#333
    style AD fill:#99ccff,stroke:#333
    style AE fill:#99ccff,stroke:#333
    style AF fill:#99ccff,stroke:#333
    style AG fill:#99ccff,stroke:#333
    style AH fill:#99ccff,stroke:#333
    style AI fill:#99ccff,stroke:#333
    style AJ fill:#99ccff,stroke:#333
    style AK fill:#99ccff,stroke:#333
    style AL fill:#99ccff,stroke:#333
    style AM fill:#99ccff,stroke:#333
    style AN fill:#99ccff,stroke:#333
    style AO fill:#99ccff,stroke:#333
    style AP fill:#99ccff,stroke:#333
    style AQ fill:#99ccff,stroke:#333
    style AR fill:#99ccff,stroke:#333
    style AS fill:#99ccff,stroke:#333
    style AT fill:#99ccff,stroke:#333
    style AU fill:#99ccff,stroke:#333
    style AV fill:#99ccff,stroke:#333
    style AW fill:#99ccff,stroke:#333
```
</details>

Figure 2. Overview of our proposed framework. The core of our approach is a conditional flow-matching module that warps a template shape toward a target via a continuous trajectory. This deformation is conditioned on the geometry-guided modeling of 2D foundation features. To ensure these features are spatially aligned and robust to varying observation angles, we introduce two key components: (1) a geometry-guided feature modeling process, which diffuses lifted 2D features across the 3D template surface to bridge the domain gap; and (2) a view-adaptive feature aggregation module, which synthesizes a pose-aware, viewpoint-invariant feature map to compensate for self-occlusions.

Geometry-Guided Feature Propagation. Bridging 2D image semantics with 3D geometry introduces the challenge of partial observability. Image-based foundation features are inherently limited to the visible surface of the template, whereas deformation requires holistic guidance for the entire shape. To address this, we propose a geometry-guided propagation mechanism that diffuses semantic cues from visible regions to the full 3D structure. This approach relies on the insight that geometrically correlated regions often share semantic attributes, allowing us to infer features for occluded areas based on their structural affinity with visible counterparts. Specifically, we first extract features Fvis ∈ RM×D $\mathbf { F } _ { \mathrm { v i s } } \in \mathbb { R } ^ { M \times D }$ corresponding to the subset of M visible points on the template. Concurrently, a 3D encoder processes the complete point cloud to yield geometric embeddings G ∈ RN×d, $\mathbf { G } \in \mathbb { R } ^ { N \times d }$ capturing the structural context of all N template points. These embeddings facilitate the computation of a pairwise affinity matrix, where the geometric similarity between a point j in the full shape and a visible point i is defined as:

$$
S _ {j i} = \frac {\mathbf {g} _ {j} \cdot \mathbf {g} _ {i}}{\| \mathbf {g} _ {j} \| \| \mathbf {g} _ {i} \|}, \tag {2}
$$

where $\mathbf { g } \in \mathbf { G }$ represents the geometric feature vector. Leveraging these affinities as a structural bridge, we propagate the available 2D semantics to the entire shape via a weighted aggregation:

$$
\mathbf {f} _ {j} ^ {\text { complete }} = \sum_ {i = 1} ^ {M} \frac {\exp (S _ {j i} / \tau)}{\sum_ {k = 1} ^ {M} \exp (S _ {j k} / \tau)} \mathbf {f} _ {i} ^ {\text { vis }}. \tag {3}
$$

This formulation enables occluded points to acquire semantic representations from geometrically related visible points, yielding the comprehensive feature set $\mathbf { F } _ { \mathrm { c o m p l e t e } } \in \bar { \mathbb { R } } ^ { N \times D }$ . This provides the deformation network with spatially consistent conditioning signals regardless of initial visibility.

Semantic-Aware Feature Alignment. While the propagated template features $\mathbf { F } _ { \mathrm { c o m p l e t e } }$ provide comprehensive semantic guidance regarding the template structure, effective deformation necessitates establishing robust correspondences between the target object and the template. The target image features ${ \bf { F } } { \boldsymbol { \tau } }$ encode the desired shape semantics within a 2D patch-based representation, whereas the template features operate on 3D point embeddings. To bridge this gap, we introduce an alignment module that leverages cross-attention to dynamically retrieve target semantics relevant to each template point. Specifically, we formulate the template features $\mathbf { F } _ { \mathrm { c o m p l e t e } } \in \mathbf { \bar { \mathbb { R } } } ^ { N \times D }$ as queries and the flattened target image features $\mathbf { F } _ { \mathcal { T } } \in \mathbb { R } ^ { H W \times D }$ as keys and values. This allows us to synthesize the aligned features by aggregating target information via:

$$
\mathbf {F} _ {\text { aligned }} = \operatorname{softmax} \left(\frac {\left(\mathbf {F} _ {\text { complete }} \mathbf {W} _ {Q}\right) \left(\mathbf {F} _ {\mathcal {T}} \mathbf {W} _ {K}\right) ^ {T}}{\sqrt {D}}\right) \left(\mathbf {F} _ {\mathcal {T}} \mathbf {W} _ {V}\right), \tag {4}
$$

where $\mathbf { W } _ { Q } , \mathbf { W } _ { K } , \mathbf { W } _ { V }$ are learnable projections. This mechanism enables template points to attend specifically to semantically corresponding regions in the target image, effectively overcoming geometric misalignments. The resulting aligned features $\mathbf { F } _ { \mathrm { a l i g n e d } }$ are further refined through self-attention layers to enforce local consistency, ultimately producing a conditioning signal c that guides the deformation network to predict shape changes that are both geometrically plausible and semantically faithful to the target.

![](images/daa1f53ac850ce9e889a6c6170adf9db4df8ee4359ddbc97f122ce1c29f62e8d.jpg)  
Figure 3. Qualitative comparison with existing shape deformation methods on novel target objects under the Random Template setting.

# 3.2. View-Adaptive Feature Aggregation

While geometry-guided modeling effectively densifies the semantic signals into $\mathbf { F } _ { \mathrm { c o m p l e t e } }$ , the underlying information remains rooted in a specific 2D observation. Consequently, these foundation features inherently encode viewpointdependent biases, where the same 3D structure may manifest differently in the feature space when observed from distinct camera poses. To address this, we propose a viewadaptive feature aggregation mechanism that dynamically incorporates camera pose embeddings to explicitly discriminate perspective-induced geometric misalignments from actual shape deformations, thereby disentangling viewpoint effects from intrinsic object semantics. This enables a viewpoint-invariant representation, providing stable guidance for the deformation learning.

# 3.2.1. MULTI-VIEW CAMERA POSE ENCODING

To align the multi-view features within a unified geometric context, we establish a canonical reference frame centered on the template view most semantically consistent with the target. Formally, given a set of template views $\mathcal { V } _ { S } = \{ I _ { S } ^ { k } \} _ { k = 1 } ^ { \bar { K } }$ 1 and their camera extrinsics $\{ \mathbf { E } _ { S } ^ { k } \} _ { k = 1 } ^ { K }$ , we identify the primary view $I _ { s } ^ { * }$ by maximizing the foundation feature cosine similarity with the target image $I _ { T }$ . The remaining views $\{ I _ { S } ^ { j } \} _ { j \neq }$ ∗ serve as auxiliary sources, offering complementary structural details from varying viewpoints.

To capture the spatial configuration of the auxiliary views relative to the primary anchor, we compute the relative camera transformations. Specifically, we define the relative pose $\mathbf { P } _ { \mathrm { r e l } } ^ { k }$ of the k-th view with respect to the primary view ∗ as:

$$
\mathbf {P} _ {\text { rel }} ^ {k} = \left(\mathbf {E} _ {\mathcal {S}} ^ {*}\right) ^ {- 1} \mathbf {E} _ {\mathcal {S}} ^ {k}, \tag {5}
$$

where $\mathbf { E } _ { \mathcal { S } }$ represents the camera-to-world transformation matrix. This formulation explicitly encodes the geometric offset between the auxiliary and primary perspectives. We flatten the rotation and translation components of $\mathbf { P } _ { \mathrm { r e l } } ^ { k }$ into a vector $\mathbf { p } ^ { k } \in \mathbb { R } ^ { 1 2 }$ and project it into the feature space:

$$
\mathbf {e} ^ {k} = \mathbf {W} _ {\text { pose }} \mathbf {p} ^ {k} + \mathbf {b} _ {\text { pose }}, \tag {6}
$$

where $\mathbf { W } _ { \mathrm { p o s e } } \in \mathbb { R } ^ { D \times { 1 2 } }$ and $\mathbf { b } _ { \mathrm { p o s e } } \in \mathbb { R } ^ { D }$ are learnable parameters. The resulting embeddings $\mathbf { e } ^ { k }$ provide the network with viewpoint-invariant geometric cues, facilitating the disentanglement of pose-induced discrepancies from intrinsic shape deformations.

Table 1. Performance on seen and unseen categories with retrieved vs. random templates. Our-SV and Our-MV refer to our method using single-view template and multi-view template feature fusion, respectively. The best results are in bold. 

<table><tr><td rowspan="3">Methods</td><td colspan="6">Seen Categories Unseen Objects</td><td colspan="6">Unseen Categories</td></tr><tr><td colspan="3">Retrieved Template</td><td colspan="3">Random Template</td><td colspan="3">Retrieved Template</td><td colspan="3">Random Template</td></tr><tr><td>CD $(10^{-3})$  ↓</td><td>EMD $(10^{-2})$  ↓</td><td>S-IoU(%) ↑</td><td>CD $(10^{-3})$  ↓</td><td>EMD $(10^{-2})$  ↓</td><td>S-IoU(%) ↑</td><td>CD $(10^{-3})$  ↓</td><td>EMD $(10^{-2})$  ↓</td><td>S-IoU(%) ↑</td><td>CD $(10^{-3})$  ↓</td><td>EMD $(10^{-2})$  ↓</td><td>S-IoU(%) ↑</td></tr><tr><td>ShapeMatcher(Di et al., 2024)</td><td>5.92</td><td>6.43</td><td>40.47</td><td>13.02</td><td>8.82</td><td>34.36</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>KP-RED(Zhang et al., 2024)</td><td>3.05</td><td>5.23</td><td>46.73</td><td>5.10</td><td>6.35</td><td>42.05</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Our-SV</td><td>2.45</td><td>4.76</td><td>48.45</td><td>2.61</td><td>4.94</td><td>46.78</td><td>4.31</td><td>5.84</td><td>50.31</td><td>4.94</td><td>6.15</td><td>49.36</td></tr><tr><td>Our-MV</td><td>2.38</td><td>4.69</td><td>48.79</td><td>2.46</td><td>4.86</td><td>47.31</td><td>3.69</td><td>5.42</td><td>52.38</td><td>4.24</td><td>5.65</td><td>52.57</td></tr></table>

# 3.2.2. POSE-AWARE CROSS-VIEW FEATURE AGGREGATION

Building upon the pose-encoded representations, we introduce an aggregation strategy designed to distill viewpointinvariant semantics while exploiting complementary structural details from diverse orientations.

To explicitly inject geometric context, we first modulate the semantic features with their corresponding pose embeddings. For each visible point i in the primary view, the feature $\mathbf { f } _ { \mathrm { p a r t i a l } } ^ { i } \in \mathbb { R } ^ { D }$ is added to the canonical pose embedding:

$$
\tilde {\mathbf {f}} _ {\text { partial }} ^ {i} = \mathbf {f} _ {\text { partial }} ^ {i} + \mathbf {e} ^ {*}, \tag {7}
$$

where $\mathbf { e } ^ { * }$ represents the embedding of the canonical reference frame. Similarly, for the M visible points in each auxiliary view $j ,$ we modulate their features $\mathbf { f } _ { \mathrm { a u x } } ^ { j , i }$ with the relative pose embedding $\mathbf { e } ^ { j } \mathrm { : }$ :

$$
\tilde {\mathbf {f}} _ {\text { aux }} ^ {j, i} = \mathbf {f} _ {\text { aux }} ^ {j, i} + \mathbf {e} ^ {j}. \tag {8}
$$

This operation aligns the semantic features with their geometric acquisition conditions.

To further enrich the primary view with context from auxiliary perspectives, we employ a cross-view attention mechanism. We designate the pose-modulated primary features as queries, while the concatenation of all view features serves as the keys and values. This configuration allows the primary view to selectively query semantically relevant regions across all available viewpoints. Formally, we stack all modulated features into a unified memory bank $\mathbf { F } _ { \mathrm { a l l } } \in \mathbb { R } ^ { ( K \times M ) \times D }$ and compute:

$$
\mathbf {F} _ {\text { fused }} = \text { Attention } (\mathbf {Q} = \tilde {\mathbf {F}} _ {\text { primary }}, \mathbf {K} = \mathbf {F} _ {\text { all }}, \mathbf {V} = \mathbf {F} _ {\text { all }}), \tag {9}
$$

where $\tilde { \mathbf { F } } _ { \mathrm { p r i m a r y } } \ \in \ \mathbb { R } ^ { M \times D }$ denotes the matrix of primary features. The final aggregated representation is obtained via a residual connection:

$$
\tilde {\mathbf {F}} _ {\text { partial }} = \mathbf {F} _ {\text { fused }} + \tilde {\mathbf {F}} _ {\text { primary }}. \tag {10}
$$

This formulation preserves the semantic integrity of the primary view while integrating complementary cues from auxiliary angles. The resulting features $\tilde { \mathbf { F } } _ { \mathrm { p a r t i a l } }$ are robust to viewpoint variations and serve as the input for the subsequent geometry-guided propagation module.

# 3.3. Training Objectives

Training the deformation network requires supervising the predicted flow field to ensure both geometric accuracy and structural preservation. The primary objective is the flow matching loss $\mathcal { L } _ { \mathrm { F M } }$ , which aligns the predicted deformation field $v _ { \theta } ( \mathbf { x } _ { t } , t , \mathbf { c } )$ with the target velocity field derived from the interpolation path. To prevent artifacts such as non-smooth deformations or excessive distortions, we incorporate several geometric regularizers: Chamfer distance loss $\mathcal { L } _ { \mathrm { C D } }$ for global shape alignment, Laplacian smoothness loss $\mathcal { L } _ { \mathrm { L a p } }$ for local continuity, ARAP loss $\mathcal { L } _ { \mathrm { A R A P } }$ for preserving local rigidity (Sorkine & Alexa, 2007), a regularization term $\mathcal { L } _ { \mathrm { r e g } }$ to constrain deformation magnitude, and a silhouette loss $\mathcal { L } _ { \mathrm { s i l } }$ for multi-view geometric supervision. The final training objective combines these terms:

$$
\begin{array}{l} \mathcal {L} = \lambda_ {\mathrm{FM}} \mathcal {L} _ {\mathrm{FM}} + \lambda_ {\mathrm{CD}} \mathcal {L} _ {\mathrm{CD}} + \lambda_ {\mathrm{Lap}} \mathcal {L} _ {\mathrm{Lap}} \tag {11} \\ + \lambda_ {\mathrm{ARAP}} \mathcal {L} _ {\mathrm{ARAP}} + \lambda_ {\mathrm{reg}} \mathcal {L} _ {\mathrm{reg}} + \lambda_ {\mathrm{sil}} \mathcal {L} _ {\mathrm{sil}}, \\ \end{array}
$$

where $\lambda .$ are weights balancing the contribution of each term. Please refer to Sec. A.1 for more details on training objectives.

# 4. Experiments

In this section, we conduct extensive experiments to evaluate the effectiveness of our proposed method, including comparisons with deformation-based and 3D generative methods, ablation studies, and demonstrations of downstream applications.

![](images/9c0ccbdd8f598bba14012dd893c654d6131929ac8dc9bb4ff10a2db1c3f836cb.jpg)

![](images/bd4b7d79c976c6ea0d83b64efa890755eb0525dcda7415dbbd607028c0e2c29e.jpg)

![](images/f61300f1af124ffed359e3e1a3393c915729d847db93aa5e64cbd10906a7d5f1.jpg)  
Figure 4. Quantitative and qualitative comparisons with existing 3D generative methods on single-view shape reconstruction. LRM-small, LRM-base, and LRM-large denote different model sizes. Phidias-Image and Phidias-3D refer to models conditioned solely on target images and those conditioned on additional 3D templates, respectively.

# 4.1. Experimental Setting

Datasets. Following prior works (Uy et al., 2021; Di et al., 2024; Zhang et al., 2024), we use ShapeNetv2 (Chang et al., 2015) as the training source and consider seven seen categories: chair, table, airplane, car, cabinet, bowl and bottle. For each category, we randomly sample 500 shape models, from which 50 models are further randomly selected as template objects; the remaining models are randomly split into training and test targets with a 9:1 ratio. To evaluate generalization to unseen categories, we additionally use OakInk (Yang et al., 2022) and select four categories: mug, teapot, lotion pump, and camera. For each category, we randomly choose 5 objects as templates and 20 objects as test targets, covering both real and synthetic instances. For data generation, we render RGB observations by sampling camera viewpoints on an object-centered upper hemisphere. We uniformly sample 16 camera poses around the template object to obtain multi-view template images, while for each target object, we randomly sample one viewpoint to obtain a single target observation, which naturally introduces challenges such as self-occlusion or partial visibility.

Competing Methods and Evaluation Metrics. We compare our approach against two representative deformation learning methods, ShapeMatcher (Di et al., 2024) and KP-RED (Zhang et al., 2024), and three 3D generative methods, LRM (Hong et al., 2024), Wonder3D (Long et al., 2024), and Phidias (Wang et al., 2025). We quantitatively evaluate the performance using Chamfer Distance (CD), Earth Mover’s Distance (EMD), and silhouette IoU (S-IoU). Please refer to Sec. A.2 and Sec. A.3 for more details.

# 4.2. Main Results

We assess different methods on novel objects to evaluate shape deformation learning quality, employing two template selection strategies for each target observation. The Retrieved Template setting selects the template from the library that maximizes the cosine similarity in DINOv3 features between the canonical view and the target image. The Random Template setting selects a template arbitrarily and may introduce large shape variations. Regarding the baselines ShapeMatcher and KP-RED, we train separate models for each category while setting the target occlusion ratio to 50% and utilizing ground truth depth values to recover the target partial point cloud. In contrast, our method employs a unified model across all categories without relying on target observation depth information. We ensure fairness by using identical training data and testing pairs across all methods.

Evaluation on Large Shape Variations. Table1 presents quantitative results on novel objects within seen and unseen categories. As can be seen, our method consistently achieves lower CD and EMD scores alongside higher S-IoU values to demonstrate superior deformation quality. Notably, the performance of the two baselines drops significantly when using random templates, whereas our method maintains performance levels similar to the retrieved template setting. This resilience shows that our geometry-guided modeling effectively captures the intricate similarity between the template and target observation to enhance robustness against shape variations. Figure 3 illustrates qualitative comparisons under the Random Template setting. It can be observed that baseline methods struggle to match the target when significant topological changes occur, such as transforming a four-legged chair into a sofa or simplifying a two-tier table into a single-layer one. In contrast, our method handles these structural variations effectively.

Evaluation on Unseen Categories. For the challenging scenario where both template and target belong to unseen categories, ShapeMatcher and KP-RED are not applicable to this setting due to their requirement for per-category training. Consequently, we only evaluate our method and the results indicate that it continues to achieve high-quality deformations and demonstrates strong generalization capabilities for unseen categories. Figure 5 presents qualitative results of our method on unseen categories. Please refer to Sec. A.4 and Sec. A.6 for more results.

Evaluation on Viewpoint Variations. During dataset construction, we randomly sampled viewpoints around the target shape to render observations, inevitably introducing challenging scenarios such as severe self-occlusion. Figure 4 presents comparisons with 3D generative methods. As observed, when distinctive structures like a mug handle or table legs are partially occluded in the input view, the baselines tend to generate implausible shapes, often failing to reconstruct these missing parts. In contrast, our proposed deformation learning approach effectively leverages geometric priors from the template, preserving the essential topology of the template and ensuring the generated shape remains semantically plausible and consistent with the category.

# 4.3. Ablation Study

In this section, we perform an ablation study to verify the necessity of each proposed module. All model variants were retrained on the same dataset and evaluated on novel objects to assess deformation learning performance. The experimental results are summarized in Table 2.

(i) Replacing the flow matching framework with a direct regression objective (w/o FM) resulted in a performance drop. This indicates that modeling the deformation as a continuous trajectory via flow matching is superior to deterministic regression for capturing complex shape variations. (ii) We demonstrated that our geometry-guided foundation feature modeling strategy is essential for effective deformation learning. We validated this through two specific experiments: (1) by assigning features only to visible projected points on the template and setting the remaining points to the mean value (w/o Prop.); (2) by removing the relational modeling entirely, where target features were globally pooled and broadcast via FiLM to condition the template features (w/o Rel.). In both scenarios, the performance deteriorated, indicating that our geometry-guided design effectively captures the intricate spatial and semantic relationships between the template and target foundation features for precise deformation. (iii) View-adaptive feature aggregation module

Table 2. Quantitative ablation study results on novel objects. 

<table><tr><td rowspan="2">Methods</td><td colspan="3">Retrieved Template</td><td colspan="3">Random Template</td></tr><tr><td>CD $(10^{-3}) \downarrow$ </td><td>EMD $(10^{-2}) \downarrow$ </td><td>S-IoU $(\%) \uparrow$ </td><td>CD $(10^{-3}) \downarrow$ </td><td>EMD $(10^{-2}) \downarrow$ </td><td>S-IoU $(\%) \uparrow$ </td></tr><tr><td>w/o FM</td><td>2.66</td><td>4.87</td><td>45.78</td><td>2.74</td><td>4.95</td><td>43.57</td></tr><tr><td>w/o Prop.</td><td>2.74</td><td>4.96</td><td>44.19</td><td>2.95</td><td>5.18</td><td>41.10</td></tr><tr><td>w/o Rel.</td><td>2.56</td><td>4.85</td><td>45.36</td><td>2.70</td><td>5.06</td><td>44.67</td></tr><tr><td>w/o PrimSel.</td><td>2.64</td><td>4.87</td><td>44.50</td><td>2.84</td><td>5.08</td><td>44.40</td></tr><tr><td>w/o PoseAware.</td><td>2.60</td><td>4.87</td><td>44.98</td><td>2.79</td><td>5.06</td><td>44.47</td></tr><tr><td>Our-MV</td><td>2.38</td><td>4.69</td><td>48.79</td><td>2.46</td><td>4.86</td><td>47.31</td></tr></table>

further enhances deformation learning. We compared our full model against using a single-view template reference (our-SV), removing the primary view selection by using the mean of all features as the aggregation query (w/o PrimSel.), and excluding camera pose encoding by directly averaging multi-view features (w/o PoseAware.). All these variations led to suboptimal results. Notably, we observed that ineffective utilization of multi-view foundation features (w/o PrimSel. and w/o PoseAware.) can yield results inferior to using a single view (our-SV). This confirms that the proposed multi-view fusion module plays a crucial role in constructing a robust, holistic template representation that mitigates occlusion and viewpoint bias.

# 4.4. Applications

Robotic dexterous grasp generation serves as a fundamental prerequisite for complex manipulation in humanoid robotics (Shu et al., 2024; Duan et al., 2022). While existing approaches include analytical (Liu et al., 2021; Wang et al., 2023; Ma et al., 2025a) and generative (Liu et al., 2023c; Lu et al., 2024) techniques, transfer-based methods (Yang et al., 2022; Ma et al., 2025b) typically exhibit stronger generalization by adapting grasp representations from known templates to novel targets via learned correspondences. However, the success of such methods critically depends on the quality of the underlying shape mapping.

Addressing this challenge, a distinct advantage of our deformation-based framework is the inherent preservation of dense point-wise correspondence, which naturally facilitates downstream tasks such as dexterous grasp transfer. As illustrated in Figure 5, we leverage the predicted deformation fields to warp contact maps from templates to novel targets, guiding the optimization of grasp configurations for arbitrary robotic hands. This transfer-based dexterous grasp generation process requires only about 0.67 s for deformation prediction and 15 s for contact-based grasp recovery. We validate this capability in Isaac Gym simulations using the Shadow Hand on the OakInk dataset (Yang et al., 2022). Compared to existing dexterous grasp generation methods, our method achieves highly competitive grasp quality while significantly reducing computational overhead and bypassing the need for complete 3D target shapes. Furthermore, as shown in Figure 6, we demonstrate the practical applicability of our approach through real-world physical experiments on a NAVIAI AW-1 humanoid robot. Evaluated across four categories (bowl, bottle, mug, and lotion pump), our method achieves an impressive overall physical grasp success rate of 77%, demonstrating robust generalization to unseen objects in realistic settings. More detailed experimental setups, baseline comparisons, and quantitative metrics are provided in Sec. A.4.

![](images/e52da3e356485a334c9d7e11b0a7f80e0e9ac7d0ab0f5fe3cc1632e25e06ed68.jpg)

Figure 5. Qualitative results of contact map and grasp transfer for diverse objects across multiple robotic hands. Red and blue regions in the contact maps denote high and low contact values, respectively.   
![](images/66fd6b4064710653a43c1c0ee56c4742aa21d165be96ba8bb8fd99ef0315e0eb.jpg)  
Figure 6. Qualitative results of generalizable dexterous manipulation in the real world.

# 5. Conclusion

In this paper, we presented a novel deformation learning framework for generalizable single-view 3D shape recovery. We introduced a geometry-guided feature modeling mechanism that enriches foundation features with template topology, enabling the model to capture fine-grained correspondences and handle complex shape variations across unseen

categories. To mitigate performance degradation caused by viewpoint disparities, we developed a view-adaptive feature aggregation module that dynamically synthesizes a viewpoint-invariant template representation, ensuring consistent alignment between the fixed template and arbitrary target views. Extensive experiments demonstrate that our approach outperforms state-of-the-art methods in deformation and reconstruction quality, while also proving its practical efficacy in dexterous robotic manipulation tasks.

While our method shows promising results, single-view reconstruction remains an inherently ill-posed problem. When key regions of the target are fully occluded, the lack of explicit 2D deformation cues may lead to geometric discrepancies (please refer to Sec. A.8 for more details). In future work, we plan to extend our framework to support multi-view target inputs, thereby enriching the geometric information for more accurate deformation learning. Moreover, incorporating semantic priors from vision-language models can provide enhanced guidance for the reconstruction of ambiguous structures, which represents a promising direction for future investigation.

# Impact Statement

This paper presents a geometry-guided framework for monocular 3D object shape recovery via template deformation, leveraging semantically robust 2D foundation features to improve generalization across viewpoints and shape variations. The resulting capabilities may advance downstream research in areas such as affordance transfer and robotic manipulation, contributing to broader developments in autonomous systems. We do not foresee immediate severe risks or negative societal implications related to our contributions.

# Acknowledgment

This wok was supported in part by the National Natural Science Foundation of China Project No. 62322318, and in part by a grant from the NSFC/RGC Joint Research Scheme sponsored by the Research Grants Council of the Hong Kong Special Administrative Region, China and the National Natural Science Foundation of China (Project No. N CUHK410/23), and in part by the Joint Funds of the National Natural Science Foundation of China (Grant No. U24A20128).

# References

Shadow dexterous hand series - research and development tool. https://www.shadowrobot.com/ dexterous-hand-series/, 2005.   
Albergo, M. S. and Vanden-Eijnden, E. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations (ICLR), 2023.   
Barsellotti, L., Bianchi, L., Messina, N., Carrara, F., Cornia, M., Baraldi, L., Falchi, F., and Cucchiara, R. Talking to dino: Bridging self-supervised vision backbones with language for open-vocabulary segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22025–22035, 2025.   
Cai, E., Donca, O., Eisner, B., and Held, D. Non-rigid relative placement through 3d dense diffusion. In Conference on Robot Learning, pp. 1268–1289. PMLR, 2025.   
Caraffa, A., Boscaini, D., Hamza, A., and Poiesi, F. Freeze: Training-free zero-shot 6d pose estimation with geometric and vision foundation models. In European Conference on Computer Vision, pp. 414–431. Springer, 2024.   
Carion, N., Gustafson, L., Hu, Y.-T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K. V., Khedr, H., Huang, A., et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.

Chang, A. X., Funkhouser, T., Guibas, L., Hanrahan, P., Huang, Q., Li, Z., Savarese, S., Savva, M., Song, S., Su, H., et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.

Chen, K., Ma, Y., Lin, X., James, S., Zhou, J., Liu, Y.-H., Abbeel, P., and QI, D. Vision foundation model enables generalizable object pose estimation. Advances in Neural Information Processing Systems (NeurIPS), 37:19975– 20002, 2024a.

Chen, K., Li, C., Tu, C., Pan, J., Ma, Y., Chen, W., Zhou, Z., Xu, X., James, S., Fu, C.-W., Xiong, R., Abbeel, P., Liu, Y.-H., and Dou, Q. A retrieval-augmented framework enabling vlm spatial awareness for object-centric robot manipulation. Science Robotics, 11(113):eaea2092, 2026. doi: 10.1126/scirobotics.aea2092.

Chen, X., Huang, L., Liu, Y., Shen, Y., Zhao, D., and Zhao, H. Anydoor: Zero-shot object-level image customization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 6593–6602, 2024b.

Chen, Z., Jiang, P., and Huang, R. Dv-matcher: Deformation-based non-rigid point cloud matching guided by pre-trained visual features. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 27264–27274, 2025.

Cho, J., Youwang, K., Yang, H., and Oh, T.-H. Robust 3d shape reconstruction in zero-shot from a single image in the wild. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pp. 22786–22798, 2025.

Di, Y., Zhang, C., Zhang, R., Manhardt, F., Su, Y., Rambach, J., Stricker, D., Ji, X., and Tombari, F. U-red: Unsupervised 3d shape retrieval and deformation for partial point clouds. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8884–8895, 2023.

Di, Y., Zhang, C., Wang, C., Zhang, R., Zhai, G., Li, Y., Fu, B., Ji, X., and Gao, S. Shapematcher: Self-supervised joint shape canonicalization segmentation retrieval and deformation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21017–21028, 2024.

Duan, H., Wang, P., Li, Y., Li, D., and Wei, W. Learning human-to-robot dexterous handovers for anthropomorphic hand. IEEE Transactions on Cognitive and Developmental Systems, 15(3):1224–1238, 2022.

Fahim, G., Amin, K., and Zarif, S. Single-view 3d reconstruction: A survey of deep learning methods. Computers & Graphics, 94:164–190, 2021.

Groueix, T., Fisher, M., Kim, V. G., Russell, B. C., and Aubry, M. 3d-coded: 3d correspondences by deep deformation. In Proceedings of the european conference on computer vision (ECCV), pp. 230–246, 2018.   
Hong, Y., Zhang, K., Gu, J., Bi, S., Zhou, Y., Liu, D., Liu, F., Sunkavalli, K., Bui, T., and Tan, H. Lrm: Large reconstruction model for single image to 3d. In The Twelfth International Conference on Learning Representations (ICLR), 2024.   
Huang, W., Wang, C., Zhang, R., Li, Y., Wu, J., and Fei-Fei, L. Voxposer: Composable 3d value maps for robotic manipulation with language models. In Conference on Robot Learning (CoRL), pp. 540–562. PMLR, 2023.   
Huang, Z., Boss, M., Vasishta, A., Rehg, J. M., and Jampani, V. Spar3d: Stable point-aware reconstruction of 3d objects from single images. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pp. 16860–16870, 2025.   
Jack, D., Pontes, J. K., Sridharan, S., Fookes, C., Shirazi, S., Maire, F., and Eriksson, A. Learning free-form deformations for 3d object reconstruction. In Asian conference on computer vision (ACCV), pp. 317–333. Springer, 2018.   
Jakab, T., Tucker, R., Makadia, A., Wu, J., Snavely, N., and Kanazawa, A. Keypointdeformer: Unsupervised 3d keypoint discovery for shape control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12783–12792, 2021.   
Jiang, H., Huang, Q., and Pavlakos, G. Real3d: Towards scaling large reconstruction models with real images. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5821–5833, 2025.   
Knaebel, K., Yilmaz, K., de Geus, D., Hermans, A., Adrian, D., Linder, T., and Leibe, B. DINO in the room: Leveraging 2D foundation models for 3D segmentation. In 2026 International Conference on 3D Vision (3DV), 2026.   
Kurenkov, A., Ji, J., Garg, A., Mehta, V., Gwak, J., Choy, C., and Savarese, S. Deformnet: Free-form deformation network for 3d shape reconstruction from a single image. In 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 858–866. IEEE, 2018.   
Lin, Y., Chen, Y.-W., Tsai, Y.-H., Jiang, L., and Yang, M.- H. Text-driven image editing via learnable regions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 7059–7068, 2024.   
Liu, D., Yu, X., Ye, M., Zhangli, Q., Li, Z., Zhang, Z., and Metaxas, D. N. Deformer: Integrating transformers with deformable models for 3d shape abstraction from

a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision (CVPR), pp. 14236–14246, 2023a.   
Liu, M., Shi, R., Kuang, K., Zhu, Y., Li, X., Han, S., Cai, H., Porikli, F., and Su, H. Openshape: Scaling up 3d shape representation towards open-world understanding. Advances in neural information processing systems, 36: 44860–44879, 2023b.   
Liu, S., Zhou, Y., Yang, J., Gupta, S., and Wang, S. ContactGen: Generative Contact Modeling for Grasp Generation. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 20552–20563, Paris, France, 2023c. IEEE. ISBN 9798350307184. doi: 10.1109/ICCV51070.2023.01884.   
Liu, T., Liu, Z., Jiao, Z., Zhu, Y., and Zhu, S.-C. Synthesizing diverse and physically stable grasps with arbitrary hand structures using differentiable force closure estimator. IEEE Robotics and Automation Letters, 7(1):470–477, 2021.   
Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations (ICLR), 2023d.   
Long, X., Guo, Y.-C., Lin, C., Liu, Y., Dou, Z., Liu, L., Ma, Y., Zhang, S.-H., Habermann, M., Theobalt, C., et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9970–9980, 2024.   
Lu, J., Kang, H., Li, H., Liu, B., Yang, Y., Huang, Q., and Hua, G. Ugg: Unified generative grasping. In European Conference on Computer Vision (ECCV), pp. 414–433. Springer, 2024.   
Ma, Y., Chen, K., Xu, X., Zhou, Z., Xie, L., Xiong, R., and Dou, Q. Coladex: Contact-guided optimization and vlm-assisted selection for task-oriented dexterous grasp generation. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 19562– 19569, 2025a.   
Ma, Y., Chen, K., Zheng, K., and Dou, Q. Contact map transfer with conditional diffusion model for generalizable dexterous grasp generation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS), 2025b.   
Makoviychuk, V., Wawrzyniak, L., Guo, Y., Lu, M., Storey, K., Macklin, M., Hoeller, D., Rudin, N., Allshire, A., Handa, A., et al. Isaac gym: High performance gpu based physics simulation for robot learning. In NeurIPS Datasets and Benchmarks, 2021.

Mescheder, L., Oechsle, M., Niemeyer, M., Nowozin, S., and Geiger, A. Occupancy networks: Learning 3d reconstruction in function space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 4460–4470, 2019.   
Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research Journal, pp. 1–31, 2024.   
Pan, J., Han, X., Chen, W., Tang, J., and Jia, K. Deep mesh reconstruction from single rgb images via topology modification networks. In Proceedings of the IEEE/CVF international conference on computer vision (ICCV), pp. 9964–9973, 2019.   
Pang, Y., Tay, E. H. F., Yuan, L., and Chen, Z. Masked autoencoders for 3d point cloud self-supervised learning. World Scientific Annual Review of Artificial Intelligence, 1:2440001, 2023.   
Paschalidou, D., Katharopoulos, A., Geiger, A., and Fidler, S. Neural parts: Learning expressive 3d shape abstractions with invertible neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3204–3215, 2021.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning (ICML), pp. 8748–8763. PmLR, 2021.   
Shu, X., Ni, F., Fan, X., Yang, S., Liu, C., Tu, B., Liu, Y., and Liu, H. A versatile humanoid robot platform for dexterous manipulation and human–robot collaboration. CAAI Transactions on Intelligence Technology, 9(2):526– 540, 2024.   
Shuai, Q., Zhang, C., Yang, K., and Chen, X. Dpf-net: Combining explicit shape priors in deformable primitive field for unsupervised structural reconstruction of 3d objects. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14321–14329, 2023.   
Simeoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, ´ M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.   
Sommer, L., Dunkel, O., Theobalt, C., and Kortylewski, A.¨ Common3d: Self-supervised learning of 3d morphable models for common objects in neural feature space. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pp. 6468–6479, 2025.

Sorkine, O. and Alexa, M. As-rigid-as-possible surface modeling. In Symposium on Geometry processing, volume 4, pp. 109–116, 2007.   
Team, G., Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican, K., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.   
Thengane, V., Zhu, X., Bouzerdoum, S., Phung, S. L., and Li, Y. Foundational models for 3d point clouds: A survey and outlook. arXiv preprint arXiv:2501.18594, 2025.   
Uy, M. A., Kim, V. G., Sung, M., Aigerman, N., Chaudhuri, S., and Guibas, L. J. Joint learning of 3d shape retrieval and deformation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11713–11722, 2021.   
Wallace, B. and Hariharan, B. Few-shot generalization for single-image 3d reconstruction via priors. In Proceedings of the IEEE/CVF international conference on computer vision (ICCV), pp. 3818–3827, 2019.   
Wang, N., Zhang, Y., Li, Z., Fu, Y., Liu, W., and Jiang, Y.-G. Pixel2mesh: Generating 3d mesh models from single rgb images. In Proceedings of the European conference on computer vision (ECCV), pp. 52–67, 2018.   
Wang, R., Zhang, J., Chen, J., Xu, Y., Li, P., Liu, T., and Wang, H. DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General Objects Based on Simulation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 11359–11366, London, United Kingdom, 2023. IEEE. ISBN 9798350323658. doi: 10.1109/ICRA48891.2023.10160982.   
Wang, W., Ceylan, D., Mech, R., and Neumann, U. 3dn: 3d deformation network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (ICCV), pp. 1038–1046, 2019.   
Wang, Y., Lira, W., Wang, W., Mahdavi-Amiri, A., and Zhang, H. Slice3d: Multi-slice occlusion-revealing single view 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9881–9891, 2024.   
Wang, Z., Wang, T., He, Z., Hancke, G., Liu, Z., and Lau, R. W. Phidias: A generative model for creating 3d content from text, image, and 3d conditions with referenceaugmented diffusion. In 13th International Conference on Learning Representations (ICLR), 2025.   
Wei, Y.-L., Jiang, J.-J., Xing, C., Tan, X., Wu, X.-M., Li, H., Cutkosky, M., and Zheng, W.-S. Grasp as you say: Language-guided dexterous grasp generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS), 2024.

Wen, C., Zhang, Y., Li, Z., and Fu, Y. Pixel2mesh++: Multiview 3d mesh generation via deformation. In Proceedings of the IEEE/CVF international conference on computer vision (ICCV), pp. 1042–1051, 2019.   
Wu, J. Z., Zhang, Y., Turki, H., Ren, X., Gao, J., Shou, M. Z., Fidler, S., Gojcic, Z., and Ling, H. Difix3d+: Improving 3d reconstructions with single-step diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pp. 26024–26035, 2025.   
Wu, X., Jiang, L., Wang, P.-S., Liu, Z., Liu, X., Qiao, Y., Ouyang, W., He, T., and Zhao, H. Point transformer v3: Simpler faster stronger. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4840–4851, 2024.   
Xiu, Y., Yang, J., Tzionas, D., and Black, M. J. Icon: Implicit clothed humans obtained from normals. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13286–13296. IEEE, 2022.   
Xiu, Y., Yang, J., Cao, X., Tzionas, D., and Black, M. J. Econ: Explicit clothed humans optimized via normal integration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 512–523, 2023.   
Xue, L., Gao, M., Xing, C., Mart´ın-Mart´ın, R., Wu, J., Xiong, C., Xu, R., Niebles, J. C., and Savarese, S. Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1179–1189, 2023.   
Yang, L., Li, K., Zhan, X., Wu, F., Xu, A., Liu, L., and Lu, C. Oakink: A large-scale knowledge repository for understanding hand-object interaction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20953–20962, 2022.   
Yifan, W., Aigerman, N., Kim, V. G., Chaudhuri, S., and Sorkine-Hornung, O. Neural cages for detail-preserving 3d deformations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 75–83, 2020.   
Zhang, R., Zhang, C., Di, Y., Manhardt, F., Liu, X., Tombari, F., and Ji, X. Kp-red: Exploiting semantic keypoints for joint 3d shape retrieval and deformation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 20540–20550, 2024.   
Zheng, Z., Yu, T., Liu, Y., and Dai, Q. Pamir: Parametric model-conditioned implicit representation for imagebased human reconstruction. IEEE transactions on pat-

tern analysis and machine intelligence (TPAMI), 44(6): 3170–3184, 2021.   
Zhu, R., Hui, K.-H., Liu, Z., Wu, Q., Tang, W., Qiu, S., Heng, P.-A., and Fu, C.-W. Cos3d: Collaborative openvocabulary 3d segmentation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.   
Zhu, R., Qiu, S., Wu, Q., Hui, K.-H., Heng, P.-A., and Fu, C.- W. Pcf-lift: Panoptic lifting by probabilistic contrastive fusion. In European Conference on Computer Vision, pp. 92–108. Springer, 2024.   
Zhu, R., Qiu, S., Liu, Z., Hui, K.-H., Wu, Q., Heng, P.-A., and Fu, C.-W. Rethinking end-to-end 2d to 3d scene segmentation in gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3656–3665, 2025.

# A. Appendix

In this supplementary material, we provide additional details, experiments, and results to support the main paper:

A1. Details on Training Objectives   
A2. Details on Evaluation Metrics   
A3. Implementation Details   
A4. Additional Application Results   
A5. Additional Comparison Results   
A6. Additional Qualitative Results   
A7. Discussion on Robustness to Cross-Category Template Assignments   
A8. Failure Analysis

# A.1. Details on Training Objectives

As formulated in the main paper, our total training objective L is a weighted combination of the flow matching loss and geometric regularization terms. In this section, we provide the detailed definitions for each component.

First, to supervise the continuous deformation trajectory, we minimize the difference between the predicted vector field $v _ { \theta }$ and the target straight-line velocity field via the Flow Matching Loss $( \mathcal { L } _ { \mathrm { F M } } )$ . This loss is computed as an expectation over time t and data pairs:

$$
\mathcal {L} _ {\mathrm{FM}} = \mathbb {E} _ {t, \mathbf {x} _ {0}, \mathbf {x} _ {1}} \left[ \| v _ {\theta} (\mathbf {x} _ {t}, t, \mathbf {c}) - (\mathbf {x} _ {1} - \mathbf {x} _ {0}) \| _ {2} ^ {2} \right], \tag {12}
$$

where ${ \bf x } _ { t } = ( 1 - t ) { \bf x } _ { 0 } + t { \bf x } _ { 1 }$ represents the interpolated state. To ensure the deformed point cloud $\mathcal { P } _ { \mathrm { p r e d } }$ geometrically aligns with the target shape ${ \mathcal P } _ { \mathrm { g t } }$ , we employ the Chamfer Distance Loss $( \mathcal { L } _ { \mathrm { C D } } )$ , calculated as the symmetric sum of squared distances:

$$
\mathcal {L} _ {\mathrm{CD}} = \sum_ {\mathbf {p} \in \mathcal {P} _ {\text { pred }}} \min _ {\mathbf {q} \in \mathcal {P} _ {\mathrm{gt}}} \| \mathbf {p} - \mathbf {q} \| _ {2} ^ {2} + \sum_ {\mathbf {q} \in \mathcal {P} _ {\mathrm{gt}}} \min _ {\mathbf {p} \in \mathcal {P} _ {\text { pred }}} \| \mathbf {q} - \mathbf {p} \| _ {2} ^ {2}. \tag {13}
$$

To prevent high-frequency artifacts and maintain local surface smoothness, we incorporate the Laplacian Smoothness Loss $( \mathcal { L } _ { \mathrm { L a p } } )$ . Let $\delta _ { i }$ be the Laplacian coordinate of vertex i (the difference between the vertex and the centroid of its neighbors). We minimize:

$$
\mathcal {L} _ {\text { Lap }} = \sum_ {i} \| \delta_ {i} ^ {(\text { pred })} - \delta_ {i} ^ {(\text { src })} \| _ {2} ^ {2}, \tag {14}
$$

which encourages the local geometry of the deformed mesh to remain consistent with the source mesh topology. Furthermore, to preserve structural rigidity and avoid unnatural distortions (e.g., shearing), we enforce the As-Rigid-As-Possible (ARAP) Loss $( \mathcal { L } _ { \mathrm { A R A P } } )$ (Sorkine & Alexa, 2007). This term minimizes the deviation of local transformations from rigid rotations:

Table 3. Quantitative comparison with different dexterous grasp generation methods. The best results are in bold. 

<table><tr><td>Methods</td><td>SR(%) ↑</td><td>Pen.(mm) ↓</td><td>Cov.(%) ↑</td><td>Time(s) ↓</td></tr><tr><td colspan="5">Analytical Method</td></tr><tr><td>DFC (Liu et al., 2021)</td><td>65.00%</td><td>8.01</td><td>30.07%</td><td>&gt;1000</td></tr><tr><td colspan="5">Generative Methods</td></tr><tr><td>ConGen (Liu et al., 2023c)</td><td>47.59%</td><td>3.05</td><td>23.19%</td><td>17.1</td></tr><tr><td>UGG (Lu et al., 2024)</td><td>58.00%</td><td>8.37</td><td>36.65%</td><td>76.0</td></tr><tr><td colspan="5">Transfer-Based Methods</td></tr><tr><td>Tink (Yang et al., 2022)</td><td>61.96%</td><td>4.60</td><td>27.91%</td><td>87.6</td></tr><tr><td>cmtDiff (Ma et al., 2025b)</td><td>69.59%</td><td>2.70</td><td>31.17%</td><td>62.2</td></tr><tr><td>Our-MV</td><td>76.92%</td><td>2.86</td><td>28.39%</td><td>15.7</td></tr></table>

$$
\mathcal {L} _ {\mathrm{ARAP}} = \sum_ {i} \sum_ {j \in \mathcal {N} (i)} w _ {i j} \| (\mathbf {p} _ {i} ^ {\prime} - \mathbf {p} _ {j} ^ {\prime}) - \mathbf {R} _ {i} (\mathbf {p} _ {i} - \mathbf {p} _ {j}) \| _ {2} ^ {2}, \tag {15}
$$

where $\mathbf { p } , \mathbf { p ^ { \prime } }$ denote vertex positions before and after deformation, $\mathcal { N } ( i )$ are neighbors of vertex i, $w _ { i j }$ are cotangent weights, and $\mathbf { R } _ { i }$ is the optimal rotation matrix for the local neighborhood.

We also apply a Regularization Loss $( \mathcal { L } _ { \mathrm { r e g } } )$ to constrain the magnitude of the deformation vectors $\mathbf { d } _ { i }$ and encourage minimal necessary displacement:

$$
\mathcal {L} _ {\text { reg }} = \frac {1}{N} \sum_ {i = 1} ^ {N} \| \mathbf {d} _ {i} \| _ {2} ^ {2}. \tag {16}
$$

Finally, to leverage multi-view supervision, we utilize the Silhouette Loss $( \mathcal { L } _ { \mathrm { s i l } } )$ by rendering the deformed shape into 2D silhouette s S(k) $S _ { \mathrm { p r e d } } ^ { ( k ) }$ from K viewpoints and comparing them with ground truth masks Sgt $S _ { \mathrm { g t } } ^ { ( k ) }$ :

$$
\mathcal {L} _ {\mathrm{sil}} = \sum_ {k = 1} ^ {K} \| S _ {\text { pred }} ^ {(k)} - S _ {\mathrm{gt}} ^ {(k)} \| _ {2} ^ {2}. \tag {17}
$$

# A.2. Details on Evaluation Metrics

We quantitatively evaluate the reconstruction quality using three standard metrics: Chamfer Distance (CD), Earth Mover’s Distance (EMD), and Silhouette IoU (S-IoU). Chamfer Distance (CD) measures the geometric accuracy between the predicted point cloud P and the ground truth G without requiring point-to-point correspondence. We compute the symmetric Chamfer distance using the L2 norm, defined as the sum of the average nearest neighbor distances in both directions: $\begin{array} { r } { \mathbf { C D } ( \mathcal { P } , \mathcal { G } ) = \frac { 1 } { | \mathcal { P } | } \sum _ { \mathbf { p } \in \mathcal { P } } \operatorname* { m i n } _ { \mathbf { g } \in \mathcal { G } } \| \mathbf { p } - \mathbf { \updownarrow } } \end{array}$ $\begin{array} { r } { \mathbf { g } \| _ { 2 } ^ { 2 } + \frac { 1 } { | \mathcal { G } | } \sum _ { \mathbf { g } \in \mathcal { G } } \operatorname* { m i n } _ { \mathbf { p } \in \mathcal { P } } \| \mathbf { g } - \mathbf { p } \| _ { 2 } ^ { 2 } } \end{array}$ . Lower CD values indicate better surface alignment. Earth Mover’s Distance (EMD) captures both geometry and point density by solving an optimal transport problem. It seeks a bijection $\phi : \mathcal { P }  \mathcal { G }$ that minimizes the average distance: $\begin{array} { r } { \mathrm { E M D } ( \mathcal { P } , \mathcal { G } ) = \operatorname* { m i n } _ { \phi } \frac { 1 } { | \mathcal { P } | } \sum _ { \mathbf { p } \in \mathcal { P } } \| \mathbf { p } - \phi ( \mathbf { p } ) \| _ { 2 } } \end{array}$ , where lower values imply a more accurate reconstruction of the shape distribution. Finally, Silhouette IoU (S-IoU) evaluates the visual fidelity from a specific 2D perspective. We render binary masks M from the observed viewpoint and calculate the Intersection-over-Union between the predicted and ground truth projections: $\begin{array} { r } { \mathrm { S - I o U } = \frac { | M _ { \mathrm { p r e d } } \cap M _ { \mathrm { g t } } ^ { \star } | } { | M _ { \mathrm { p r e d } } \cup M _ { \mathrm { g t } } | } } \end{array}$ Higher S-IoU values indicate better structural correctness in the projected 2D space.

![](images/fa53b84dc44ba1324bc9c2e62d0e335df84c02882fb91fb01098d575df013b70.jpg)  
Target Observation

Deformation Reconstruction   
![](images/c9dcc17ffe0286f135766173fefc05b80bde9d0d8d089b1fcb0bfc8d24776ede.jpg)  
w/o FM

Deformation Reconstruction   
![](images/8964a0d21c7353c18496d8a760b03ed316a6ca5054affbeb41547b8cda283886.jpg)  
w/o Prop.

Deformation Reconstruction   
![](images/abc9e3d04bd6add029442376419ee6ecd698f8ba5b88a150860b9730f95e5662.jpg)  
w/o Rel.

![](images/e4197ed4e0521f2acccfccbf68d0b2531e50387ad9b45267c760eedbbe9ca512.jpg)  
GT

![](images/a504085544bee2b847f29c16c1f05d3408cc39d241d4921bd907eeb05762085e.jpg)  
Template Shape

Deformation Reconstruction   
![](images/4e92528c460f9ad477fb3a8d1984209cd2face72ca5f3ebb1082100d58df386e.jpg)  
w/o PrimSel.

Deformation Reconstruction   
![](images/542854a6217abcf54bcd132556bd50eeda8ca229e39fb9df39454b089a2d867e.jpg)  
w/o PoseAware.

Deformation Reconstruction   
![](images/1889e5f526e9031d0e0a8f888b88e8e9459fd3af4f2cba389cbf42a716150c0c.jpg)  
Our\_SV

Deformation Reconstruction   
![](images/e6414c77c4e69a9b1987f7c4590d15a8ff36f95308aa115affadc4879ca9c0c5.jpg)  
Our\_MV

![](images/e16cb789b390db56ef0673db1d8247382380f707df457adb680f2fee04be3488.jpg)  
Target Observation

Deformation Reconstruction   
![](images/0915800e1889c76e3ad257739344c691ae4f3456d21bdcecc5cef2b9af106f16.jpg)  
w/o FM

Deformation Reconstruction   
![](images/05ff85efa95e765d4f8ac8881a5aff64c63b014dbd954a4a74970e8905bfd891.jpg)  
w/o Prop.

Deformation Reconstruction   
![](images/ab85faa21f4720199514c44403c87d587062f980b5ca408139c139c3fae180d6.jpg)  
w/o Rel.

![](images/73f5993964144c1e89ddddcaee219c2094f80fe351829ccacc8264170b9cd174.jpg)

![](images/6ac24dd2b180fd996c33b4a2f4c7208ca2a933c883036d31bdff4ba1b9a6470b.jpg)  
Template Shape

Deformation Reconstruction   
![](images/b3c7b335d27149a0014ad789b1410ffa3222786b4fc38344d9a73edd72c21f87.jpg)  
w/o PrimSel.

Deformation Reconstruction   
![](images/540051b92586e86db434a512e87884d0ab6e0ce80d6c92c0b5ef3b21aa99c32a.jpg)  
w/o PoseAware.

![](images/6cf82f5784d596310d58d3116b8be9e9474e8063e0b192815e44e6f0236fed30.jpg)  
Our\_SV

Deformation Reconstruction   
![](images/facde180284584a7bf76b68d4f46327fd1a29ff6ff2fbb7ecd54e8601ab51154.jpg)  
Our\_MV   
Figure 7. Qualitative comparison of deformation and reconstruction results under different ablation settings.

# A.3. Implementation Details

We represent each template shape as a point cloud with N = 1024 points and render K = 16 support views, with images at resolution $H = W = 5 1 2 ;$ for partial observations, we set the number of visible points to $M = 5 1 2$ We extract patch-level features from the final transformer block (layer 12) of a pretrained DINOv3 ViT-B/16 model, yielding a feature dimension of D = 768. During geometryguided feature lifting, we adopt Point-MAE (Pang et al., 2023) to capture structural relationships within the complete point cloud, using a 12-layer Transformer encoder with embedding dimension 384 and 6 attention heads, and

a decoder that outputs point-wise features of dimension $d = 2 5 6$ . For both the feature alignment module and the multi-view camera feature fusion module, we integrate contextual information into a 512-dimensional query stream via an initial cross-attention block, followed by a stack of self-attention layers with 8 heads for feature refinement. We adopt PointTransformerV3 (Wu et al., 2024) as our deformation backbone, which employs a hierarchical encoderdecoder architecture with serialized attention mechanisms to process point clouds organized into patches. We set the loss weights to $\lambda _ { F M } = \lambda _ { L a p } = \lambda _ { A R A P } = \lambda _ { r e g } = 1 . 0 ,$ , $\lambda _ { C D } = 1 0 0 . 0$ , and $\lambda _ { s i l } = 5 . 0 .$ . We train the model from scratch using Adam with batch size 8, 4 dataloader workers, and an initial learning rate of $1 0 ^ { - 5 }$ annealed at 50% of the training epochs using a cosine schedule. All models are trained for 100 epochs on two NVIDIA H800 GPUs, which takes approximately 36 hours.

# A.4. Additional Application Results

In this section, we validate the practical utility of our framework by leveraging the predicted template-to-target deformation fields for dexterous grasp transfer. Robotic dexterous grasp generation aims to synthesize stable and physically plausible hand poses for manipulating objects, serving as a fundamental prerequisite for enabling complex dexterous manipulation in humanoid robotics (Shu et al., 2024; Duan et al., 2022). Existing approaches can be broadly categorized into analytical (Liu et al., 2021; Wang et al., 2023), generative (Liu et al., 2023c; Lu et al., 2024), and transferbased methods (Yang et al., 2022; Ma et al., 2025b). Among them, transfer-based methods typically exhibit stronger generalization capabilities by adapting grasp-related representations from known templates to novel targets via learned correspondences. However, the success of such methods critically depends on the quality of the underlying shape mapping. Here, we demonstrate that our framework provides an effective solution for transferring known human grasps defined on template shapes to diverse, unseen target objects, supporting arbitrary multi-finger robot hands.

Table 4. Quantitative results under cross-category template assignments. We evaluate our method under three template-selection regimes: (i) Same sub-category, where the template is randomly drawn from a same sub-category (ii) Same super-category, where the template is drawn from a different sub-category within the same super-category, and (iii) Cross super-category, where the template is sampled from a different super-category. 

<table><tr><td rowspan="3">Test category</td><td colspan="6">Ours</td><td colspan="2">Phidias-3D</td></tr><tr><td colspan="2">Same-sub</td><td colspan="2">Same-super</td><td colspan="2">Cross-super</td><td colspan="2">Same-sub</td></tr><tr><td>CD $(10^{-3}) \downarrow$ </td><td>EMD $(10^{-2}) \downarrow$ </td><td>CD $(10^{-3}) \downarrow$ </td><td>EMD $(10^{-2}) \downarrow$ </td><td>CD $(10^{-3}) \downarrow$ </td><td>EMD $(10^{-2}) \downarrow$ </td><td>CD $(10^{-3}) \downarrow$ </td><td>EMD $(10^{-2}) \downarrow$ </td></tr><tr><td>Furniture</td><td>3.41</td><td>5.66</td><td>7.88</td><td>7.21</td><td>15.86</td><td>10.31</td><td>11.22</td><td>8.34</td></tr><tr><td>Container</td><td>2.21</td><td>4.50</td><td>7.51</td><td>7.52</td><td>8.81</td><td>8.39</td><td>8.07</td><td>7.45</td></tr><tr><td>Vehicle</td><td>1.36</td><td>3.88</td><td>5.94</td><td>7.13</td><td>11.57</td><td>9.90</td><td>5.78</td><td>6.93</td></tr><tr><td>Overall</td><td>2.56</td><td>4.88</td><td>7.36</td><td>7.30</td><td>12.42</td><td>9.52</td><td>8.31</td><td>7.64</td></tr></table>

![](images/772b04abf488d83d718d9aed3c86a59a33b7d96b68c14e59df939ce4a8484346.jpg)  
using Same sub-category templates (left), Same super-category templates (center), and Cross super-category templates (right). The selected template category Figure 8. Qualitative results under cross-category template assignments. For each test instance, we show reconstructions produced by our is indicated in each panel. As the template becomes less related to the target object, reconstructions progressively deviate from the correct part structure and method using Same sub-category templates (left), Same super-category templates (center), and Cross super-category templates (right). geometry, illustrating the model’s failure mode under severe template mismatch.The selected template category is indicated in each panel. As the template becomes less related to the target object, reconstructions progressively deviate from the correct part structure and geometry, illustrating the model’s failure mode under severe template mismatch.

# A.4.1. SIMULATION EXPERIMENTS

Methodology. Specifically, following the protocols established in (Liu et al., 2023c; Ma et al., 2025b), we represent hand-object interaction using an object-centric contact map. We utilize objects from the OakInk dataset (Yang et al., 2022), which provides parametric human grasps as templates, and compute a contact map on each template shape. This map is defined as a point-wise contact probability in [0, 1], where a higher value indicates a higher likelihood of hand contact. Given an RGB observation of a target object, our model predicts a template-to-target deformation field to reconstruct the target shape. Leveraging the induced pointwise correspondence, we directly warp the template contact map onto the reconstructed target geometry. Finally, we recover the grasp configuration for various robotic hands on the target object by optimizing for the transferred contact information.

![](images/ff37581ddff87e4b5d9cca8216ad33383cbf6ad506b0a7e70a1fc0f055e44812.jpg)

<details>
<summary>other</summary>

| Category | Baseline Type | Deformation | Reconstruction |
|---|---|---|---|
| Chair | Ours | 0.1 | 0.2 |
| Chair | Ours | 0.15 | 0.25 |
| Chair | Ours | 0.2 | 0.3 |
| Chair | Ours | 0.25 | 0.35 |
| Chair | Ours | 0.3 | 0.4 |
| Chair | Ours | 0.35 | 0.45 |
| Chair | Ours | 0.4 | 0.5 |
| Chair | Ours | 0.45 | 0.55 |
| Chair | Ours | 0.5 | 0.6 |
| Chair | Ours | 0.55 | 0.65 |
| Chair | Ours | 0.6 | 0.7 |
| Chair | Ours | 0.65 | 0.75 |
| Chair | Ours | 0.7 | 0.8 |
| Chair | Ours | 0.75 | 0.85 |
| Chair | Ours | 0.8 | 0.9 |
| Chair | Ours | 0.85 | 0.95 |
| Chair | Ours | 0.9 | 1.0 |
| Chair | Ours | 0.95 | 1.05 |
| Chair | Ours | 1.0 | 1.1 |
| Chair | Ours | 1.05 | 1.15 |
| Chair | Ours | 1.1 | 1.2 |
| Chair | Ours | 1.15 | 1.25 |
| Chair | Ours | 1.2 | 1.3 |
| Chair | Ours | 1.25 | 1.35 |
| Chair | Ours | 1.3 | 1.4 |
| Chair | Ours | 1.35 | 1.45 |
| Chair | Ours | 1.4 | 1.5 |
| Chair | Ours | 1.45 | 1.55 |
| Chair | Ours | 1.5 | 1.6 |
| Chair | Ours | 1.55 | 1.65 |
| Chair | Ours | 1.6 | 1.7 |
| Chair | Ours | 1.65 | 1.75 |
| Chair | Ours | 1.7 | 1.8 |
| Chair | Ours | 1.75 | 1.85 |
| Chair | Ours | 1.8 | 1.9 |
| Chair | Ours | 1.85 | 1.95 |
| Chair | Ours | 1.9 | 2.0 |
| Chair | Ours | 1.95 | 2.05 |
| Chair | Ours | 2.0 | 2.1 |
| Chair | Ours | 2.05 | 2.2 |
| Chair | Ours | 2.1 | 2.3 |
| Chair | Ours | 2.15 | 2.4 |
| Chair | Ours | 2.2 | 2.5 |
| Chair | Ours | 2.25 | 2.6 |
| Chair | Ours | 2.3 | 2.7 |
| Chair | Ours | 2.35 | 2.8 |
| Chair | Ours | 2.4 | 2.9 |
| Chair | Ours | 2.45 | 3.0 |
| Chair | Ours | 2.5 | 3.1 |
| Chair | Ours | 2.55 | 3.2 |
| Chair | Ours | 2.6 | 3.3 |
| Chair | Ours | 2.65 | 3.4 |
| Chair | Ours | 2.7 | 3.5 |
| Chair | Ours | 2.75 | 3.6 |
| Chair | Ours | 2.8 | 3.7 |
| Chair | Ours | 2.85 | 3.8 |
| Chair | Ours | 2.9 | 3.9 |
| Chair | Ours | 2.95 | 4.0 |
| Chair | Ours | 3.0 | 4.1 |
| Chair - Box Top-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box- Box Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box-Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box- Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Box Box - Total: ~3,4,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99; Total: ~34,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78; Total: ~34,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68; Total: ~34,46,47,48,49,50,51,52,53[ ]
</details>

Figure R5. Qualitative comparison with TAX3D on shape deformation. Our method achieves superior geometric Figure 9. Qualitative comparison with TAX3D on shape deformation. Our method achieves superior geometric consistency and consistency and demonstrates greater robustness to large template-target variationsdemonstrates greater robustness to large template-target variations compared to TAX3D.

In detail, we reconstruct the robotic hand mesh from joint angles q using differentiable forward kinematics. Beyond the dense contact map, we also leverage sparse contact cues by identifying the five fingertip positions relative to the template shape. Using the predicted deformation field, we map these keypoints to their corresponding target locations, denoted as $\mathcal { P } _ { \mathrm { t g t } } \in \mathbb { R } ^ { 5 \times 3 }$ . We then iteratively update q by minimizing a composite loss function:

$$
\mathcal {L} _ {\text { syn }} = \lambda_ {\text { cont }} E _ {\text { cont }} + \lambda_ {\text { tip }} E _ {\text { tip }} \tag {18}
$$

where ${ \cal E } _ { \mathrm { c o n t } } = { \mathcal L } _ { \mathrm { M S E } } ( \hat { \mathbf c } , \overline { { { \mathbf c } } } )$ represents the mean squared map c derived from error between the transferred contact map ˆc and the current $\mathbf { q } .$ The term $\begin{array} { r } { E _ { \mathrm { t i p } } = \sum _ { i = 1 } ^ { 5 } \| \mathbf { p } _ { i } ( \mathbf { q } ) - \mathbf { \tau } } \end{array}$ $\mathbf { p } _ { \mathrm { t g t } , i } \Vert _ { 2 } ^ { 2 }$ enforces the Euclidean distance between the current fingertip positions ${ \bf p } _ { i } ( { \bf q } )$ and their target locations $\mathbf { p } _ { \mathrm { t g t } , i }$ .

Evaluation Metrics. To comprehensively evaluate grasp quality, we benchmark our method against state-of-the-art analytical, generative, and transfer-based approaches (Wei et al., 2024; Liu et al., 2023c; Ma et al., 2025b). Experiments are conducted using the Shadow Hand (sha, 2005) across three object categories unseen during training: mug, teapot, and lotion pump. Each category contains 10 target objects with 5 different template grasps. We employ three standard metrics to quantify the performance. Success Rate (SR) measures the ability to stably lift an object in the Isaac-Gym simulator (Makoviychuk et al., 2021). Under standard gravity $( - 9 . 8 m / s ^ { 2 } )$ , a grasp is deemed successful if the object is lifted above 10 cm and shifts less than 2 cm after 60 steps. We report the average success rate over 10 trials with randomized object poses. Penetration Depth (Pen.) evaluates collision artifacts by measuring the maximum distance any object point penetrates the hand mesh. Contact Coverage (Cov.) assesses the percentage of hand vertices located within a ±2mm threshold of the object surface.

Results. Table 3 reports the quantitative results for dexterous grasp generation. As expected, previous transferbased methods generally achieve higher grasp success rates and better grasp quality compared to generative baselines. However, they require access to complete target shapes to transfer contacts and typically incur substantial runtime (at least 60 s), which limits practical deployment. In contrast, our method predicts the template-to-target deformation directly from a single target RGB observation by modeling 3D shape variation with 2D foundation features. This enables rapid deformation inference (approx. 0.67 s) followed by a 15 s optimization to recover the grasp configuration, thus offering improved generalization and usability in realistic settings. Figure 5 in the main paper presents qualitative examples of grasp transfer onto diverse objects using multiple robotic hands (Shadow Hand, Inspire Hand, NAVIAI Hand, and a self-developed dexterous hand), demonstrating that our method provides accurate dense correspondence to effectively enhance dexterous grasp generation.

# A.4.2. REAL-WORLD ROBOTIC EXPERIMENTS

We further conducted real-world experiments to evaluate the effectiveness of our method in practical scenarios. The experiments were performed on a NAVIAI AW-1 humanoid robot equipped with a dexterous hand featuring 15 degrees of freedom (DoFs) and 6 active joints. As shown in Figure 6, the setup consists of a table placed in front of the robot.

![](images/a3e7385c4015cb7aae6a697d48564fa1adc6e48f19624a4ac6c8f46ded66c2cd.jpg)

<details>
<summary>text_image</summary>

Target Observation
Template Shape
Front View Side View
Ours
Discrepancy due to occlusion
Front View Side View
GT
Side view of the target object (unobserved)
</details>

Figure 10. Failure Analysis. When key regions are fully occluded in the target observation, our method tends to preserve the template’s corresponding structure. This leads to a discrepancy with the true target shape if the unobserved geometry differs from the template.

Given the target observation captured by the robot’s builtin camera, we first employ SAM3 (Carion et al., 2025) to segment the target object, which is then fed into our model to predict the deformation from the template. Subsequently, we transfer the human grasp from the template to the robotic hand following the optimization method described above. For grasp execution, we follow the protocol in (Wei et al., 2024): the robot arm first moves to the predicted 6-DoF pose of the hand’s root, after which the dexterous hand actuates its joint angles based on the predicted grasp configuration.

We evaluated our method on four object categories: bowl, bottle, mug, and lotion pump. We conducted five trials for each object instance to compute the average success rate. A grasp is considered successful only if the robot can lift the object by at least 20 cm and maintain a stable hold for a duration of 10 seconds. Figure 6 illustrates qualitative results from the physical experiments, highlighting our method’s ability to accurately deform template shapes to match realworld objects captured by robot perception. These results demonstrate effective generalization to novel objects and categories in real-world settings. Quantitative results show an overall success rate of 77% (60% for bowls, 90% for bottles, 80% for mugs, and 80% for lotion pumps), confirming the practical applicability of our approach and its robustness to real-world observations.

# A.5. Additional Comparison Results

In addition to the state-of-the-art deformation methods evaluated in the main paper, we further compare our approach with TAX3D (Cai et al., 2025), a recent diffusion-based deformation field generation method. Specifically, we set the template object as the initial action point cloud $( P _ { A } )$ and the target object as the anchor point cloud $( P _ { B } )$ to predict the per-point displacement flow from $P _ { A }$ to $P _ { B }$ . We used the ground-truth depth to recover the target partial point cloud, consistent with the evaluation of other baselines in the main paper. We trained TAX3D from scratch on the same training dataset used for our method, following the default hyperparameter from its original paper.

Table 5. Quantitative comparison of shape deformation performance between our method and TAX3D on unseen objects. 

<table><tr><td>Method</td><td>CD ↓</td><td>EMD ↓</td><td>S-IoU ↑</td><td>Time (s) ↓</td></tr><tr><td>TAX3D(Cai et al., 2025)</td><td>5.83</td><td>6.49</td><td>40.11</td><td>3.23</td></tr><tr><td>Ours</td><td>2.46</td><td>4.86</td><td>47.31</td><td>0.64</td></tr></table>

Both quantitative and qualitative results are presented in Table 5 and Figure 9. As demonstrated, our method achieves lower CD and EMD errors, as well as better qualitative shape consistency after deformation. Because single-view target observations are inherently partial, using them directly as a condition (as TAX3D does) leads to information loss. In contrast, our method leverages the robust semantic similarity provided by 2D foundation models to guide 3D shape deformation learning. This makes our approach much more robust to large template-target shape variations and yields superior generalization to unseen objects. Additionally, our single-step rectified flow matching framework is computationally more efficient than the iterative denoising process required by the diffusion model in TAX3D.

# A.6. Additional Qualitative Results

Figure 11 presents additional deformation and reconstruction results on diverse objects, utilizing target observations either rendered from the test set or synthesized by the image generation model (Team et al., 2023). Despite significant shape variations between the template and target, our method yields smooth, consistent deformation fields and faithfully reconstructs the target geometry. Figure 12 illustrates the deformation and reconstruction outcomes on realworld unseen object categories. Additionally, we demonstrate the transfer of human grasps and template contact maps to the target via the estimated deformation fields, enabling the generation of robotic dexterous grasps across diverse robot hand models. Figure 7 presents the qualitative results under different ablation settings. It can be observed that removing any key component leads to varying degrees of performance degradation, such as discontinuous deformations, compromised structural integrity in the generated shapes, and poor reconstruction of unobserved regions. These visual artifacts confirm the necessity of each module for achieving high-quality deformation and reconstruction.

# A.7. Discussion on Robustness to Cross-Category Template Assignments

In the main paper, we demonstrated the effectiveness of our deformation-based approach at the category level. To further explore its generalization capabilities and systematically investigate the robustness of our framework against template mismatches, we evaluate its performance under cross-category template assignments. Specifically, we group the test objects into three broad super-categories: Furniture (chair, table, cabinet), Container (bowl, bottle, mug, teapot, lotion pump), and Vehicle (car, airplane). Without any retraining, we assess the model under three distinct template selection settings: 1) Same-sub (template from the exact same sub-category), 2) Same-super (template from a different sub-category but within the same super-category), and 3) Cross-super (template from a completely different super-category).

As illustrated in Table 4 and Figure 8, transitioning from the Same-sub to the Same-super setting leads to a moderate increase in CD and EMD. However, our method remains highly competitive with template-augmented 3D generative baselines, such as Phidias-3D. This demonstrates the model’s robustness and highlights its underlying mechanism: while 2D observations provide strong target guidance, the template’s topology acts as a structural regularizer during shape deformation. For instance, in Same-super cases (e.g., deforming a bowl template to a bottle target), the model successfully matches the overall silhouette of the bottle while preserving the open-top topology of the bowl. Conversely, moving to the Cross-super setting marks the failure threshold of our model. In these scenarios, the extreme topological discrepancy (e.g., deforming a car template to a cup) exceeds the inherent deformation capacity of the network, resulting in a substantial performance drop.

# A.8. Failure Analysis

Our method leverages the template shape, template 2D foundation features, and their correlation with the target’s 2D features to enrich the shape representation. While this approach ensures robustness to viewpoint variations by preserving the template’s structure in unobserved regions, it may fail to recover the accurate target geometry when the invisible parts differ significantly from the template. As shown in Figure 10, when the handle of the mug is totally occluded, our method successfully recovers the cup body but retains the handle structure of the template. This ambiguity is an inherent challenge in single-view reconstruction. In future work, we plan to extend our model to support multi-view target inputs, thereby achieving more accurate deformation learning and reconstruction.

![](images/90f4995eb6f7498fbf80d6085d9512ac9fb41c0ca9950b52a3392dd31b550ef7.jpg)

Figure 11. Visualization of shape and corresponding deformation field generated by our proposed method on diverse novel objects.   
![](images/63082681fd06758afb57c8280c3f4748e57506ed2d48c7bcc59e5b6f9d47cee4.jpg)  
Target Object Observation

![](images/d560e5bbc0a3ec2b7317998ed2599e181113aa37c59e157c26ad9a248f3d0e72.jpg)  
Template

![](images/c67e6d534c30edfd9d5a5f4524ce9c2cb897addf648e150789e664f7737662c4.jpg)  
Deformation

![](images/fd95a3aa0892fb53fa86f63d2db3c0c217cb7ccc2bfb08ab52da33c86dbdc5d4.jpg)  
Reconstruction

Human Grasp   
![](images/82943e9697ce31b0eb5a692419fff5157459cb0ff7773e9d93a467a9ccdfb837.jpg)

Template Contact Ma   
![](images/e90e3f5a6062cd7012deb2af650045caf6cffe9a703dff4c3eb87332b7be1417.jpg)

![](images/3a8571253ec52fba5b59c538bd8c894b9e3bc38b83f4bc7181f4840f5743971f.jpg)

![](images/770674872c42436a19d1e615acedd3e406b28acc27367d464dbc3be0b35a8199.jpg)

Transferred Contact Map and Grasps   
![](images/f867db8eb5bdd0483ab922d931c51eae24b9a8d128833da22160949ca95331e8.jpg)

![](images/bbd642eb9a971be6347ad0423d1107f70fa9bdac5bca2714676cb30ca18351dc.jpg)

![](images/dfdaaef9f8f483c9fd28ac5e43d69f2180573433e3184488e45da5c42524a844.jpg)

![](images/65635fa62fb610b42d19eee7d211661f4d2c7fda4f97e3d20c1baeed90abc2cc.jpg)

![](images/478fbcc22455d5d72f3a6528fb285b19fc1ae8b35058f7bf2e13d79f87fca25b.jpg)

![](images/9e2f0bf36393d2dda0638977a371c2a978195941a8a61ef08ece27ce069d2715.jpg)

![](images/760b2ca470b302ea05468e81717134d4170ed111aaf2a1f7772c540f86434646.jpg)

![](images/6657948d72b79f9cf253e20b99c817675656590b9c526553a5b83967333b1e25.jpg)

![](images/fdbdb5461088c4ba4ec8729e306048e5e206d38828d9b9b7a0e4ddecabc484b7.jpg)

![](images/0b707fcf8ba7b42b722b710e0cf35994079fd6bec11b0fe14b27e24dbdda54b3.jpg)  
Reconstruction

![](images/517991fbc1609c97b7a88effd6b910f4c7ad049a7c54ec3dc4558e98433521fd.jpg)

![](images/ba9938f60756d78ffb88134417cb1dbc75b3b03c5363fc990784ee2d0efa942e.jpg)

![](images/b9041e7f5be1044aced2b1f3dc5b7daac0c384df239d1cc043abc1c92426bf03.jpg)

![](images/a40c69388b5b468956de948f37caeb1d1fc41b8a1b57344406da8fc1a02e448f.jpg)

![](images/bdc58080f4d01e9ce513756d606e0c5a637e85a4e7b059d6bf9c237ca4cf2e7f.jpg)

![](images/59a9ea0fbb801be6df18ab5d3a6da5122c1d5dd5e2085fd01e2edba94b306fa4.jpg)

![](images/2fc81b48498e0fca0fe609e3bcd73ad349793dec21f01bf46371ec2f248197a7.jpg)

![](images/5ee807a5414d3ad584c79dafba88dd6ef66d289fec549eaf5644ecbf2cce83a9.jpg)  
Reconstruction

![](images/3da9ad530c66c406f6d04184b1a04cd3957d45e37965a213a7104c0a32731411.jpg)

![](images/ca25ae7c3cf56cf427297b15eae083bbaa0d356210b93fb0e0d198d4ba592f97.jpg)

![](images/10f105562ce53b558a62db55ac32def3904c2c5bd18d62bccba87aedc0e20cf9.jpg)

![](images/00e64c4d1898c11cf9a460f99f5f6c81b155ff7a4811995c259e86e3f8df772a.jpg)

![](images/17af898297c1388f78b58c6c49fda92b04b0447f04d758da18487689a3247247.jpg)

![](images/733139299bc9b77b08e7661b2439457e3db23101169d41cbb2dbfb877d1ae013.jpg)

![](images/502f9f5962a04097de7214b86a8e56a42570ce20f545dede04108d6d16d57134.jpg)

![](images/45fb45fa65cce118b38f9691f8301002c9f3381503a5be32bb98336aed3c00a6.jpg)  
Reconstruction

![](images/8a43da268cffb6f8a54a9fcca2241df66953207e6279430469c6b1acb8827a63.jpg)

![](images/b1449ca5603458df52dfa2e281b341528d986b58eb079ba5faf8886a008f4a4e.jpg)

![](images/a54a494df7bdf3cecffd059c9825343063e845eba94c9a3229300dfdb198196c.jpg)

![](images/0aa0a78b998a7b01f87874600b33704d359e77864c67d9599e4f8898a55bc926.jpg)

![](images/a17a2da91d19447ae9706a6d06dbe1ed8ed1085eae8a4b8aaf8ebbe6170298d7.jpg)

![](images/ba2bafb740f92e8afcb23e1097f163fe64392274ecee669041876a244215ce3b.jpg)  
Figure 12. Qualitative results of deformation and reconstruction on real-world unseen object categories, along with transferred contact maps and robotic dexterous grasps.