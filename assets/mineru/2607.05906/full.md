# GaussFusion: Towards Multimodal 3D Gaussian Pretraining

Zhixuan You<sup>a,∗</sup>, Jihua Zhu<sup>a,∗</sup>, Yiding Sun<sup>a,∗</sup>, Zihao Guo<sup>a</sup>, Haozhe Cheng<sup>a</sup>, Dongxu Zhang<sup>a</sup>, Lin Chen<sup>a</sup>, Hainan Luo<sup>b</sup>

<sup>a</sup>School of Software Engineering, Xi’an Jiaotong University, Xi’an, 710100, Shaanxi, China <sup>b</sup>Wuhu HIT Robot Technology Research Institute Co., Ltd., Wuhu, Anhui, China

## Abstract

3D Gaussian Splatting provides an explicit representation that jointly models geometry and appearance, serving as a scalable foundation for 3D representation learning. Existing pre-training methods for Gaussian representations, such as masked Gaussian reconstruction, primarily capture local structures but ofer limited semantic supervision. In this paper, we propose GaussFusion, a multimodal pre-training framework for 3D Gaussian representations. GaussFusion integrates image and text supervision into masked Gaussian modeling through cross-modal semantic alignment, enabling the Gaussian encoder to learn both visual and language-level semantic information during pre-training. To better adapt masked modeling to the non-uniform distribution of Gaussian primitives, we further propose Gaussian Salience-guided Multi-scale Hole Masking (GSHM). GSHM constructs spatially continuous masked regions based on Gaussian salience. By applying hole masks at multiple scales, GSHM encourages the encoder to capture both fine-grained local patterns and broader structural depen dencies. Extensive experiments on downstream tasks demonstrate that GaussFusion improves the transferability of Gaussian representations. Notably, GaussFusion outperforms Gaussian-MAE on ModelNet40 and ScanObjectNN (PB-T50-RS) by 0.61% and 3.85%, respectively.

Keywords: 3D Gaussian Splatting, Multimodal fusion, Self-supervised Learning, 3D Representation Learning

## 1. Introduction

3D Gaussian Splatting (3DGS)[1] represents 3D objects and scenes with anisotropic Gaussian primitives. Unlike point clouds, which typically encode discrete geometric samples, 3DGS assigns opacity, scale, rotation, and color to each primitive, thereby modeling geometry and appearance within a unified explicit representation. With high-quality rendering capability and eficient optimization, 3DGS has become an important paradigm for 3D visual representation. However, most existing studies use 3DGS primarily for reconstruction and rendering, leaving the structural and semantic information in the Gaussian parameter space insuficiently explored for 3D understanding.

This gap is important for 3D representation learning, where transferable models are still limited by the scarcity of large-scale, high-quality annotated 3D data. Although PointNet[2] and PointNet++[3] show that deep neural networks can learn effective features directly from point clouds, scaling point cloud pre-training remains dificult because point cloud datasets require costly acquisition, cleaning, and annotation. In contrast, 3DGS data can be automatically generated from existing 3D models, multi-view images, and rendering pipelines[4, 5, 6], ofering greater scalability for self-supervised learning. Therefore, learning structural and appearance priors from large-scale 3DGS data and transferring the learned representations to downstream tasks, such as point cloud classification, segmentation, and few-shot recognition, provides a feasible way to reduce the dependence on annotated 3D data.

Existing 3D pre-training methods are mainly designed for point clouds and can generally be divided into generative and contrastive approaches. Generative methods, such as Point-BERT[7] and Point-MAE[8], learn contextual 3D information through masked point patch modeling or masked autoencoding reconstruction. Contrastive and cross-modal methods further introduce supervision from 2D images, text, and pretrained vision models to enhance point cloud representations. For example, CrossPoint[9],

![](images/ab514d82337cf4bc448028a6b92b504b26999b0a05ef86444246009a10564810.jpg)  
Fig. 1. Motivation and overview of GaussFusion. Compared with basic masked Gaussian modeling, GaussFusion introduces image-text semantic alignment and GSHM masking to select salient, spatially co herent Gaussian regions, improving local structure learning and semantic transferability.

ACT[10], and ReCon[11] leverage cross-modal information to improve the semantic expressiveness of point cloud features. These studies show that mature 2D visionlanguage models can provide efective semantic priors for 3D representation learning. Recently, several studies have started to perform pre-training directly on 3DGS representations. Gaussian-MAE, proposed in ShapeSplat[12], introduces the masked autoencoder paradigm into Gaussian primitive modeling. By randomly masking and reconstructing Gaussian attributes, it enables self-supervised learning of Gaussian representations and achieves strong transfer performance on multiple downstream point cloud tasks. SceneSplat[13] further extends this direction to scene-level 3D representation learning and improves global modeling capability with large-scale scene data.

Existing 3DGS-oriented pre-training methods have shown that Gaussian representations can be efectively transferred to downstream 3D tasks. However, most of these methods are still driven by masked Gaussian reconstruction alone. This objective helps the model recover local geometry and primitive attributes, but it provides limited guidance for learning semantic distinctions across object categories and scenes. When objects share similar shapes or local structures, or when observations are heavily occluded, reconstruction based only on Gaussian parameters may fail to learn suficiently discriminative representations. Meanwhile, pre-trained image and vision-language models[14, 15] have shown strong generalization capability in visual semantic modeling. This motivates the integration of image appearance cues, textual semantics, and 3DGS parameters, so that Gaussian representations can learn both local structural details and global semantic information for more transferable 3D representation learning.

Beyond the design of the training objective, multimodal Gaussian pre-training also requires a masking strategy that is compatible with cross-modal semantic alignment. In multimodal pre-training, image and text features can provide semantic supervision for Gaussian representations. However, such supervision cannot be fully exploited if the masked Gaussian regions are selected randomly. Unlike uniformly sampled points, Gaussian primitives are irregularly distributed and exhibit large variations in opacity, scale, and spatial coverage. Random masking may either remove many lowinformation primitives that contribute little to semantic learning or disrupt a small number of visually salient primitives that correspond to key object parts. As a result, the reconstruction targets may be weakly associated with the semantic cues provided by images and text, leading to large variations in reconstruction dificulty and limiting the efective use of multimodal supervision. These challenges indicate that efective multimodal Gaussian pre-training requires a masking mechanism that selects visually salient and structurally informative regions while preserving spatial coherence across scales.

To address these limitations, this paper proposes GaussFusion, a multimodal self supervised pre-training framework for 3D Gaussian representations. Gaussian primitives are used as the fundamental input representation. Local grouping and masked reconstruction are employed to learn 3D structural and attribute information, while image and text encoders are introduced to provide external semantic supervision, enabling Gaussian representations to remain aligned with visual and linguistic features during pre-training. Furthermore, a salience-guided multi-scale hole masking strat egy, termed Gaussian Salience-guided Multi-scale Hole Masking (GSHM), is proposed. This strategy estimates the importance of local regions according to Gaussian attributes, such as opacity and scale, and constructs spatially meaningful masked regions, thereby encouraging the model to focus on Gaussian regions that are more critical for semantic and structural understanding during reconstruction. Through the joint optimization of reconstruction learning and cross-modal alignment, GaussFusion learns 3D representations that preserve geometric structure, encode appearance information, and support semantic discrimination, as shown in Fig. 1.

The main contributions of this paper are summarized as follows:

• We identify that existing Gaussian pre-training methods mainly rely on reconstructing masked Gaussian attributes, which limits their ability to capture highlevel semantic information for 3D understanding. To address this problem, we propose GaussFusion, a multimodal self-supervised pre-training framework that integrates images, text, and 3D Gaussian parameters for semantic-aware Gaussian representation learning.

• To improve the compatibility between masked Gaussian modeling and multi modal semantic alignment, we propose Gaussian Salience-guided Multi-scale Hole Masking (GSHM). GSHM adapts the masking process to the non-uniform distribution and structural importance of Gaussian primitives by constructing spatially continuous masks over salient regions at multiple scales.

• Extensive experiments on multiple downstream point cloud tasks demonstrate that GaussFusion improves the transferability and robustness of Gaussian representations under challenging and data-limited scenarios. These results suggest that 3DGS can serve not only as a rendering-oriented representation, but also as a scalable representation space for multimodal 3D pre-training and general 3D understanding.

## 2. Related Work

## 2.1. Point Cloud Representation Learning

Point cloud representation learning aims to extract discriminative features from unordered 3D point sets to support downstream tasks such as classification, segmen tation, and retrieval [16, 17]. Early studies mainly follow the supervised learning paradigm. PointNet[2] first enables end-to-end modeling of point sets and addresses the permutation invariance of point clouds through symmetric functions. Based on this foundation, PointNet++[3] further introduces hierarchical local region modeling to enhance the perception of local geometric structures. These methods mark a shift from global point-set aggregation toward hierarchical local structure modeling in point cloud representation learning. However, their performance gains usually rely on large-scale and high-quality annotated data, while the high annotation cost of 3D data limits their further deployment in practical scenarios.

To alleviate the shortage of annotated data, self-supervised point cloud pre-training has become an important direction in 3D representation learning[18, 19, 20, 21]. Point-BERT[7] divides point clouds into local point patches and learns contextual representations through masked point modeling. Point-MAE[8] adopts a masked autoencoder framework and directly reconstructs the masked local point patches. Point-M2AE[22] further introduces multi-scale structures to strengthen hierarchical geometric modeling. Recent studies have also explored teacher-guided masking and contrastive patch-graph learning to improve the semantic quality of point cloud representations[23, 24]. Mamba-based and parameter-eficient adaptation methods further improve the ef ficiency of point cloud foundation models, as shown by PointMamba[25] and Mantis[26]. These studies show that masked reconstruction-based pre-training and eficient sequence modeling provide efective supervision signals for 3D understanding tasks. In contrast, this work performs pre-training on 3DGS data, which is more scalable and contains richer attributes, and then transfers the learned structural and appearance priors to point cloud understanding tasks.

## 2.2. Gaussian Representation Learning

3D Gaussian Splatting[1] represents 3D objects or scenes as a set of 3D Gaussian primitives and achieves high-quality real-time rendering through diferentiable rasterization. Early studies on 3DGS mainly focus on reconstruction quality, rendering eficiency, and geometric consistency. For example, SuGaR[27] improves the alignment between Gaussians and object surfaces through surface alignment constraints and supports mesh extraction. Compressed 3D Gaussian Splatting[28] reduces the storage cost of Gaussian representations through compression and quantization strategies.

As the application scope of 3DGS continues to expand, recent studies have increasingly explored its capacity for semantic representation. Feature 3DGS[29] distills features from 2D foundation models into Gaussian representations, enabling Gaussian fields to carry semantic features. LangSplat[30] embeds language features into 3DGS for open-vocabulary 3D queries. Furthermore, ShapeSplat[12] constructs a large-scale object-level Gaussian dataset and proposes Gaussian-MAE, which learns self-supervised representations by reconstructing masked Gaussian attributes. SceneSplat[13] extends Gaussian representation learning to scene-level understanding and incorporates vision-language pre-training to enhance semantic modeling. These studies show that 3DGS is not only an explicit representation for rendering, but also a learnable representation for 3D understanding. However, existing object-level Gaussian pretraining still mainly relies on a single reconstruction objective, and dedicated masking mechanisms for the non-uniform distribution of Gaussian primitives remain underexplored.

## 2.3. Multimodal-Guided 3D Representation Methods

The development of 2D vision-language models[14] provides new forms of supervision for 3D representation learning. Compared with 3D annotated data, 2D images and text are easier to obtain, and large-scale vision-language models have learned strong appearance and semantic priors. Therefore, many studies attempt to enhance point cloud representations with 2D or language information[31]. Joint representation learning has been used to connect text and point clouds through an intermedi ate image space[32], while cross-modal knowledge transfer provides complementary guidance for point cloud representation learning[33]. CrossPoint[9] improves the discriminative ability of point cloud features through cross-modal contrastive learning between point clouds and rendered images. Related methods introduce cross-modal information bottlenecks or hyperbolic contrastive objectives to strengthen semantic correspondence between images and point clouds[34, 35]. ACT[10] uses pre-trained 2D models as teachers to guide 3D networks toward more semantic feature representations. ReCon[11] further combines reconstruction learning with cross-modal contrastive learning, allowing local geometry recovery and global semantic discrimination to complement each other. ULIP[36] maps language, images, and point clouds into a unified representation space, improving open-vocabulary semantic understanding in point cloud models. However, most existing methods take point clouds as the 3D input and mainly focus on aligning point cloud features with image or text features. Research that directly introduces image and text supervision into 3DGS pre-training to improve Gaussian representation learning remains relatively limited.

![](images/f1c93084c41679f2044364a285257b25418ef5d9f2ef421cd3f3a096221f2134.jpg)  
Fig. 2. Overall framework of GaussFusion. GaussFusion generates Gaussian tokens from 3D splats and applies Gaussian Salience-guided Multi-scale Hole Masking (GSHM) to local Gaussian groups. Visible Gaussian tokens are combined with learnable image and text tokens in a multimodal encoder. The decoder reconstructs masked Gaussian groups, while frozen image and text encoders provide cross-modal supervision for learning transferable 3D representations.

## 3. GaussFusion

The goal is to learn transferable representations from 3D Gaussian primitives. To this end, a pre-training framework is constructed by jointly integrating local Gaussian structure modeling, masked reconstruction learning, and cross-modal semantic align ment. The overall framework of the proposed GaussFusion is shown in Fig. 2. The input 3D Gaussian primitives are first divided into local Gaussian groups and encoded as tokens. Then, the GSHM strategy is applied to mask a subset of Gaussian regions. During the reconstruction of the masked geometric and attribute information, image and text features are introduced for cross-modal semantic alignment.

Gaussian Token Generation Following Gaussian-MAE[12], each Gaussian primitive is treated as an attributed point and is formally analogous to a point cloud input. Given a scene composed of N Gaussian primitives, the input is denoted as

$$
\mathcal {G} = \{g _ {i} \} _ {i = 1} ^ {N}, \quad g _ {i} = (\mu_ {i}, \alpha_ {i}, s _ {i}, q _ {i}, S H _ {i}) \in \mathbb {R} ^ {5 9},\tag{1}
$$

where $\mu _ { i } \in \mathbb { R } ^ { 3 }$ denotes the position of the Gaussian center, $\alpha _ { i } \in \mathbb { R } ^ { 1 }$ denotes opacity, $s _ { i } = ( s _ { i , x } , s _ { i , y } , s _ { i , z } ) \in \mathbb { R } ^ { 3 }$ denotes the scale along the principal axes, $q _ { i } \in \mathbb { R } ^ { 4 }$ denotes the rotation parameter in quaternion form, and $S H _ { i } \in \mathbb { R } ^ { 4 8 }$ denotes the spherical harmonics coeficients.

GaussFusion first organizes the unordered Gaussian set $\mathcal { G }$ into local structural units $N _ { j }$ . Gaussian group centers are selected by farthest point sampling over Gaussian attributes, denoted as $c _ { j } = { \mathrm { F P S } } ( { G } )$ . Then, according to the selected attribute indices, a local neighborhood is constructed around each center in the attribute space as $N _ { j } =$ $\mathrm { K N N } ( \mathcal { G } _ { \mathrm { a t t r } } , c _ { j , \mathrm { a t t r } } )$ . The Gaussian attributes in each local neighborhood are then fed into a geometric encoder to obtain the Gaussian token:

$$
z _ {j} = E _ {g} (\mathcal {N} _ {j}) + P (c _ {j}),\tag{2}
$$

where $E _ { g } ( \cdot )$ denotes the Gaussian local encoder and $P ( \cdot )$ denotes the positional embedding function.

Multimodal Encoding GaussFusion applies the GSHM masking strategy to Gaussian tokens and feeds only the unmasked visible tokens into the Transformer encoder[37] Meanwhile, image tokens and text tokens are inserted at the beginning of the sequence to receive semantic supervision from external modalities. Image and text semantics are therefore not used as auxiliary information in a post-processing stage, but directly participate in the formation of 3D representations during pre-training, as detailed in Section 3.2.

Cross-Modal Alignment and Reconstruction The reconstruction branch constrains the model to infer missing Gaussian parameters from local context. The crossmodal branch aligns Gaussian representations with semantic features produced by frozen image and text encoders, enabling 3D representations to absorb view-level ap pearance information and category-level semantic information. The two branches are jointly optimized within the same Gaussian encoding space.

## 3.1. Gaussian Salience-guided Multi-scale Hole Masking

In masked pre-training for 3D Gaussian representations, the masking strategy afects not only reconstruction learning but also the use of multimodal supervision. Random masking usually selects local groups independently and produces scattered missing regions, which is not well suited to the nonuniform distribution of Gaussian primitives. In 3DGS, dif-

![](images/4ab198bc6673583938ee4260447c87b5f1ccaa2793b6f1fa777ecaed68dd1901.jpg)  
Fig. 3. Overview of the proposed GSHM strategy. Gauss-Fusion builds local patches from 3D Gaussian splats, computes salience scores for group centers, and allocates mask quotas by the salience distribution. Selected centers form multi-scale hole regions, where masked patches are replaced by mask tokens and the remaining patches serve as visible tokens for reconstruction.

ferent primitives contribute unequally to geometry and appearance. Primitives with high opacity tend to have a stronger influence on rendered appearance, while primitives with large scales often cover broader local structures. Randomly removing isolated groups may therefore mask many low-information primitives or disrupt visually important regions, making the reconstruction target less useful for image- and text-guided semantic learning.

To address this issue, we propose Gaussian Salience-guided Multi-scale Hole Masking (GSHM), as shown in Fig. 3. GSHM constructs spatially continuous masked regions by jointly considering Gaussian salience and spatial coverage, thereby encouraging the encoder to recover semantically meaningful local structures from surrounding context.

GSHM first estimates the salience of each local Gaussian group from the attributes of its primitives. For the local group $N _ { j }$ with K Gaussian primitives, opacity reflects its contribution to rendered appearance, while the scale parameters provide a proxy for the spatial coverage of the group. The salience score is computed as

$$
a _ {j} = \left(\frac {1}{K} \sum_ {k = 1} ^ {K} \alpha_ {j, k}\right) \cdot \left(\frac {1}{K} \sum_ {k = 1} ^ {K} s _ {j, k} ^ {x} s _ {j, k} ^ {y} s _ {j, k} ^ {z}\right) ^ {\frac {1}{3}}.\tag{3}
$$

The first term measures the average visibility of the group, and the second term measures its average spatial extent using the geometric mean of the Gaussian scales. The salience scores are normalized within each sample and then mixed with a uniform distribution. This mixture prevents the masking process from collapsing to only a few highly salient regions, while still assigning higher sampling priority to groups that are more informative for appearance and structure.

After obtaining the salience-guided sampling weights $w _ { j }$ , GSHM selects multiple hole centers by salience-weighted farthest point sampling. The first center is sampled according to $w _ { j }$ . Given the previously selected hole centers $Q _ { h - 1 }$ , the next center is selected by

$$
q _ {h} = \arg \max _ {j} w _ {j} \cdot \min _ {q \in Q _ {h - 1}} \left\| \mathbf {p} _ {j} - \mathbf {p} _ {q} \right\| _ {2}, \quad h = 2, \dots , H,\tag{4}
$$

where $\mathbf { p } _ { j }$ denotes the spatial position of the j-th Gaussian group and H is the number of hole centers. This criterion favors groups that are both salient and spatially distant from existing centers, producing diverse masked regions instead of concentrating all masks in a single local area.

Given the selected hole centers, each Gaussian group is assigned to its nearest center, forming a spatial partition over local Gaussian groups. The partition associated with the h-th hole center is defined as

$$
\mathcal {A} _ {h} = \left\{j \left| h = \arg \min _ {r \in \{1, \dots , H \}} \left\| \mathbf {p} _ {j} - \mathbf {p} _ {q _ {r}} \right\| _ {2} \right. \right\}.\tag{5}
$$

GSHM then allocates diferent mask quotas to diferent hole centers. Instead of using identical hole sizes, a random scale perturbation is applied to the initial quota of each hole:

$$
\tilde {b} _ {h} = \left\lfloor \frac {\rho G}{H} \cdot \eta_ {h} \right], \quad \eta_ {h} \sim \mathcal {U} (1 - \gamma , 1 + \gamma),\tag{6}
$$

where $\rho$ denotes the mask ratio, $G$ is the number of local Gaussian groups, and $\gamma$ controls the degree of scale variation. The perturbed quotas are further adjusted so that the total number of masked groups matches the predefined mask ratio. This design produces multi-scale holes with diferent spatial extents and makes the reconstruction task closer to realistic incomplete observations.

Within each partition $\mathcal { A } _ { h } .$ , GSHM selects the groups closest to the corresponding hole center as masked targets:

$$
\mathcal {M} _ {h} = \mathrm{TopK} _ {j \in \mathcal {A} _ {h}} \left(- \left\| \mathbf {p} _ {j} - \mathbf {p} _ {q _ {h}} \right\| _ {2}, b _ {h}\right),\tag{7}
$$

$$
\mathcal {M} = \bigcup_ {h = 1} ^ {H} \mathcal {M} _ {h},\tag{8}
$$

where $b _ { h }$ is the adjusted mask quota of the h-th hole. If the union does not exactly satisfy the required number of masked groups because of empty partitions or quota perturbation, GSHM performs a final correction according to the group salience scores. The resulting mask therefore satisfies the predefined mask ratio while preserving spatial continuity and salience awareness. Compared with independent random masking, GSHM creates structured missing regions that encourage the model to infer semantically meaningful and spatially coherent Gaussian structures from visible context.

## 3.2. Cross-Modal Semantic Alignment

To improve the semantic discriminability of Gaussian representations, GaussFusion introduces image and text modalities as external supervision during pre-training. The image modality provides appearance-level cues from rendered views, while the text modality provides category-level semantic cues from textual descriptions. These modalities are used only as supervision targets. During downstream fine-tuning and inference, the model takes only 3D inputs and does not require additional image or text data.

Specifically, given a 3D Gaussian input $\mathcal { G } _ { i }$ , GaussFusion first divides it into local Gaussian groups and encodes the visible groups as a Gaussian token sequence $\mathbf { x } _ { i } ^ { g }$ To inject multimodal supervision into the Gaussian encoder, two learnable tokens are prepended to the token sequence: an image alignment token $\mathbf { x } _ { i } ^ { i m g }$ and a text alignment token $\mathbf { x } _ { i } ^ { t x t }$ . These two tokens do not contain raw image or text content. Instead, they serve as learnable query tokens that collect 3D contextual information from visible

Gaussian tokens through self-attention and are later constrained by external modality features. The input sequence of the Transformer encoder is defined as

$$
\mathbf {X} _ {i} ^ {(0)} = [ \mathbf {x} _ {i} ^ {i m g}; \mathbf {x} _ {i} ^ {t x t}; \mathbf {x} _ {i} ^ {g} ] + \mathbf {P} _ {i},\tag{9}
$$

where $\mathbf { P } _ { i }$ denotes the positional embedding. The encoded sequence is obtained by applying L Transformer layers:

$$
\mathbf {X} _ {i} ^ {(\ell)} = \operatorname{Trm} _ {\ell} \left(\mathbf {X} _ {i} ^ {(\ell - 1)}\right), \quad \ell = 1, \dots , L.\tag{10}
$$

The final output sequence is denoted as

$$
\mathbf {U} _ {i} = [ \mathbf {u} _ {i} ^ {i m g}; \mathbf {u} _ {i} ^ {t x t}; \mathbf {u} _ {i} ^ {g} ],\tag{11}
$$

where $\mathbf { u } _ { i } ^ { i m g }$ and $\mathbf { u } _ { i } ^ { t x t }$ are the output features corresponding to the image and text alignment tokens, respectively.

The role of the alignment tokens can be interpreted from the self-attention operation. Let $m \in \{ i m g , t x t \}$ denote one of the two alignment tokens, and let $\mathcal { N } _ { i }$ denote the visible Gaussian token indices of the i-th sample. In one attention head of the -th Transformer layer, the contribution collected by the modality token from visible Gaussian tokens can be written as

$$
\mathbf {c} _ {i, m} ^ {(\ell)} = \sum_ {j \in \mathcal {V} _ {i}} \omega_ {i, m j} ^ {(\ell)} \mathbf {x} _ {i, j} ^ {(\ell - 1)} W _ {V} ^ {(\ell)},\tag{12}
$$

where the attention weight is computed as

$$
\omega_ {i, m j} ^ {(\ell)} = \frac {\exp \left(\frac {\big (\mathbf {x} _ {i , m} ^ {(\ell - 1)} W _ {Q} ^ {(\ell)} \big) \big (\mathbf {x} _ {i , j} ^ {(\ell - 1)} W _ {K} ^ {(\ell)} \big) ^ {\top}}{\sqrt {d}}\right)}{\sum_ {r \in \Omega_ {i}} \exp \left(\frac {\big (\mathbf {x} _ {i , m} ^ {(\ell - 1)} W _ {Q} ^ {(\ell)} \big) \big (\mathbf {x} _ {i , r} ^ {(\ell - 1)} W _ {K} ^ {(\ell)} \big) ^ {\top}}{\sqrt {d}}\right)},\tag{13}
$$

with $\Omega _ { i } ~ = ~ \{ i m g , t x t \} \cup \mathcal { V } _ { i }$ . This formulation shows that the image and text alignment tokens aggregate information from visible Gaussian tokens through the same self-attention operation, rather than receiving image or text content directly inside the Gaussian encoder.

Each Gaussian sample $\mathcal { G } _ { i }$ is paired with a rendered image $I _ { i }$ and a text description $T _ { i } .$ . The image is encoded by a frozen ViT image encoder[15], and the text description is encoded by a frozen CLIP text encoder[14]. Their output features are used as fixed semantic targets:

$$
\mathbf {z} _ {i} ^ {i m g} = \mathrm{stopgrad} \left(F _ {i m g} (I _ {i})\right),\tag{14}
$$

$$
\mathbf {z} _ {i} ^ {t x t} = \mathrm{stopgrad} \left(F _ {t x t} (T _ {i})\right),\tag{15}
$$

where $F _ { i m g } ( \cdot )$ and $F _ { t x t } ( \cdot )$ denote the frozen image and text encoders, respectively. Since the image and text tokens produced by the Gaussian encoder are not directly in the same feature spaces as the external modality features, two linear projection layers $\phi _ { i m g } ( \cdot )$ and $\phi _ { t x t } ( \cdot )$ are used for alignment:

$$
\hat {\mathbf {z}} _ {i} ^ {i m g} = \phi_ {i m g} (\mathbf {u} _ {i} ^ {i m g}), \quad \hat {\mathbf {z}} _ {i} ^ {t x t} = \phi_ {t x t} (\mathbf {u} _ {i} ^ {t x t}).\tag{16}
$$

The projected image and text tokens are encouraged to match their corresponding external modality features through feature-level alignment. In this way, the image alignment token receives appearance supervision from rendered views, while the text alignment token receives semantic supervision from category descriptions. After pretraining, the image and text encoders are discarded, and the learned Gaussian encoder can be transferred to downstream 3D tasks without additional multimodal inputs.

## 3.3. Reconstruction Target

The reconstruction target of GaussFusion consists of a Gaussian masked reconstruction objective and a cross-modal semantic alignment objective.

Gaussian reconstruction preserves the core mechanism of Gaussian-MAE[12]. The reconstruction objective contains multiple attribute terms, including position, opacity, scale, rotation, and spherical harmonics color. The reconstruction of spatial coordinates is constrained by Chamfer Distance, while the remaining attributes are constrained by the $\ell _ { 1 }$ loss:

$$
\mathcal {L} _ {r e c} = \mathcal {L} _ {x y z} + \mathcal {L} _ {\alpha} + \mathcal {L} _ {s} + \mathcal {L} _ {q} + \mathcal {L} _ {S H},\tag{17}
$$

where $\mathcal { L } _ { x y z }$ denotes the coordinate reconstruction loss, while $\mathcal { L } _ { \alpha } , \mathcal { L } _ { s } , \mathcal { L } _ { q }$ , and $\mathcal { L } _ { S H }$ denote the reconstruction losses for opacity, scale, rotation, and color attributes, respectively. The complete Gaussian reconstruction loss can be written as

$$
\mathcal {L} _ {r e c} = \mathcal {L} _ {x y z} + \mathcal {L} _ {a t t r}.\tag{18}
$$

This reconstruction objective forces the decoder to recover not only the spatial layout of masked regions, but also the appearance-related attributes carried by Gaussian primitives.

The cross-modal semantic alignment objective constrains the image-aligned and text-aligned representations produced by the Gaussian branch. Frozen image and text encoders are used, and their output features serve as semantic targets. The Gaussian branch predicts the semantic features of the corresponding modalities through the image token and the text token, and Smooth L1 loss is used to reduce the discrepancy between the predicted features and the target features. The image alignment loss and the text alignment loss are defined as

$$
\mathcal {L} _ {i m g} = \text { SmoothL1 } \left(\mathbf {z} _ {i} ^ {i m g}, \hat {\mathbf {z}} _ {i} ^ {i m g}\right),\tag{19}
$$

$$
\mathcal {L} _ {t x t} = \text { SmoothL1 } \left(\mathbf {z} _ {i} ^ {t x t}, \hat {\mathbf {z}} _ {i} ^ {t x t}\right).\tag{20}
$$

Finally, the overall optimization objective of GaussFusion is defined as

$$
\mathcal {L} = \mathcal {L} _ {r e c} + \lambda_ {\mathrm{img}} \mathcal {L} _ {i m g} + \lambda_ {\mathrm{text}} \mathcal {L} _ {t x t}.\tag{21}
$$

## 4. Experiments

We evaluate GaussFusion by testing whether the pre-trained Gaussian representations can transfer to diferent 3D understanding tasks. The evaluation covers realworld scanned object classification, clean CAD object classification, fine-grained part segmentation, and few-shot classification. ScanObjectNN[38] tests robustness under background clutter and perturbations, ModelNet10/40[39] measure standard objectlevel recognition, ShapeNetPart[40] examines local part understanding, and few-shot classification reflects transferability under limited annotations.

The experiments also isolate the main design choices in GaussFusion. The ablation study examines how image and text supervision, GSHM masking, encoder selection, and multimodal loss weights afect downstream performance.

## 4.1. Pre-training Setup

GaussFusion is pre-trained on the ShapeSplat dataset in a self-supervised manner. ShapeSplat[12] is a large-scale 3DGS dataset that contains approximately 52,000 high-quality 3D Gaussian Splatting models. Each object is represented in the form of Gaussian parameters, including Gaussian attributes such as center position, opacity, scale, rotation quaternion, and spherical harmonics color coeficients.

Table 1: Object classification on ScanObjectNN. Overall accuracy (%) is reported. <sup>†</sup> denotes reproduced results under our implementation.

<table><tr><td>Method</td><td>OBJ-BG</td><td>OBJ-ONLY</td><td>PB-T50-RS</td></tr><tr><td colspan="4">with Point Cloud Pre-training Representation</td></tr><tr><td>PointNet[2]</td><td>73.30</td><td>79.20</td><td>68.00</td></tr><tr><td>SpiderCNN[41]</td><td>77.10</td><td>79.50</td><td>73.70</td></tr><tr><td>PointNet++[3]</td><td>82.30</td><td>84.30</td><td>77.90</td></tr><tr><td>DGCNN[42]</td><td>82.80</td><td>86.20</td><td>78.10</td></tr><tr><td>PointCNN[43]</td><td>86.10</td><td>85.50</td><td>78.50</td></tr><tr><td>Transformer[7]</td><td>79.86</td><td>80.55</td><td>77.24</td></tr><tr><td>Transformer + OcCo[7]</td><td>84.85</td><td>85.54</td><td>78.79</td></tr><tr><td>Point-MAE[8]</td><td>90.02</td><td>88.29</td><td>85.18</td></tr><tr><td>Point-M2AE[22]</td><td>91.22</td><td>88.81</td><td>86.43</td></tr><tr><td>Mamba3D[44]</td><td>92.94</td><td>92.08</td><td>91.81</td></tr></table>

with 3D Gaussian Pre-training Representation

<table><tr><td>Gaussian-MAE $^{\dagger}$ [12]</td><td>84.30</td><td>86.23</td><td>78.87</td></tr><tr><td>GaussFusion</td><td>88.98(+4.68)</td><td>88.47(+2.24)</td><td>82.72(+3.85)</td></tr></table>

To enable multimodal learning, each object in ShapeSplat is also equipped with image and text semantic information. Multi-view images are rendered from 16 predefined viewpoints and contain object silhouettes, texture cues, occlusion relations, and appearance consistency under viewpoint changes. Text descriptions are composed of category names and templated attribute phrases, providing explicit category and at tribute semantics for Gaussian representations.

In GaussFusion, 1024 Gaussian primitives are randomly sampled from each object during training. The input Gaussians are divided into 128 local groups, each containing 32 primitives. Grouping is performed based on xyz coordinates, while the local encoder uses complete Gaussian attributes for feature extraction. The Transformer encoder has a feature dimension of 384 and consists of 12 layers with 6 attention heads. The decoder consists of 4 Transformer blocks with 6 attention heads.

The mask ratio is set to 0.6 during pre-training. GSHM generates 4 spatial mask ing regions for each sample, and the salience mixing coeficient and multi-scale jitter coeficient are set to 0.7 and 0.35, respectively. The model reconstructs the masked xyz, opacity, scale, rotation, and SH attributes. Chamfer Distance is used for xyz reconstruction, while the $\ell _ { 1 }$ loss is used for the remaining attributes. Cross-modal supervision uses a frozen ViT-B/32 image encoder[15] and a frozen CLIP text encoder[14]. Both image and text alignment losses adopt Smooth L1 loss with a weight of 1.0.

![](images/b3ec41d927f63b78c9c912c38bb6ce027b69879d3c10b60cdd1fbfa131981a58.jpg)  
Fig. 4. Visualization of feature distributions. The t-SNE plots compare the feature embeddings of Gaussian-MAE and GaussFusion after pre-training and after fine-tuning on ModelNet40 and ScanObjectNN. Diferent colors represent diferent object categories.

All pre-training experiments use the AdamW optimizer with an initial learning rate of 0.001 and a weight decay of 0.05. The model is trained for 300 epochs with a cosine learning rate schedule and 10 warm-up epochs. The total batch size is 128.

## 4.2. Downstream Tasks

For downstream evaluation, only the pre-trained Gaussian encoder is transferred. The image and text encoders are used during pre-training only and are removed during fine-tuning and inference. This protocol ensures that GaussFusion does not introduce additional multimodal inputs or inference cost in downstream tasks.

Table 2: Object classification on clean ModelNet datasets. Input indicates the representation used by each method. Overall accuracy (%) is reported. <sup>†</sup> denotes reproduced results under our implementation.

<table><tr><td>Method</td><td>Input</td><td>ModelNet10</td><td>ModelNet40</td></tr><tr><td colspan="4">with Point Cloud Pre-training Representation</td></tr><tr><td>PointNet[2]</td><td>xyz</td><td>94.40</td><td>89.20</td></tr><tr><td>PointNet++[3]</td><td>xyz+normal</td><td>94.10</td><td>91.90</td></tr><tr><td>DGCNN[42]</td><td>xyz</td><td>95.00</td><td>92.20</td></tr><tr><td>PointCNN[43]</td><td>xyz</td><td>-</td><td>92.50</td></tr><tr><td>PCNN[45]</td><td>xyz</td><td>94.90</td><td>92.30</td></tr><tr><td>A-CNN[46]</td><td>xyz+normal</td><td>95.50</td><td>92.60</td></tr><tr><td>PointASNL[47]</td><td>xyz</td><td>95.70</td><td>92.90</td></tr><tr><td>P2SResLNet[48]</td><td>xyz</td><td>-</td><td>90.60</td></tr><tr><td>Point Cloud Mamba[49]</td><td>xyz</td><td>-</td><td>93.40</td></tr><tr><td>Mamba3D[44]</td><td>xyz</td><td>-</td><td>93.40</td></tr></table>

with 3D Gaussian Pre-training Representation

<table><tr><td>Gaussian-MAE $^{\dagger}$ [12]</td><td>Gaussian</td><td>95.37</td><td>92.46</td></tr><tr><td>GaussFusion</td><td>Gaussian</td><td>95.82(+0.45)</td><td>93.07(+0.61)</td></tr></table>

## 4.2.1. Classification Experiments

Object Classification on Real-World Datasets. For real-world object classification, ScanObjectNN[38] is used to evaluate the classification ability of diferent models on real scanned data. Unlike clean CAD models, this dataset contains background clutter, occlusion, and pose perturbations, making it more suitable for testing the robustness of pre-trained representations in real scenarios. It also helps examine whether a method overfits standard data characteristics or learns more general semantic representations. Table 1 reports the classification results on the OBJ-BG, OBJ-ONLY, and PB-T50-RS splits. Fig. 4 provides a t-SNE visualization of the feature distributions after pre-training and after fine-tuning on ModelNet40 and ScanObjectNN, illustrating the evolution from mixed embeddings to more discriminative category clusters.

Compared with point cloud pre-training methods such as Point-MAE[8], Gauss-Fusion still shows a performance gap. This is mainly related to the diference in input representation and pre-training data form. Point-MAE is pre-trained and transferred directly in the point cloud space, whereas GaussFusion uses 3D Gaussian attributes as the pre-training target and then transfers the learned representation to point cloud classification. This setting introduces the dificulty of cross-representation transfer. Therefore, the comparison with Gaussian-MAE[12] under the same Gaussian representation setting is more central to this work.

![](images/a5a4667e2e5a037729641ca9bf80200a8d7af2b88987dddb9b053973d9ff1fed.jpg)  
Fig. 5. Qualitative comparison of part segmentation results on ShapeNetPart. Diferent colors represent predicted semantic parts. Compared with Gaussian-MAE, GaussFusion produces more spatially coherent predictions and clearer part boundaries, particularly for thin structures and geometrically complex regions.

Under the Gaussian representation setting, GaussFusion outperforms Gaussian MAE[12] on the splits with background clutter and strong perturbations. In particular, it achieves 82.72% on PB-T50-RS, the most challenging setting, improving over Gaussian-MAE by 3.85%. This result indicates that multimodal supervision and the GSHM masking strategy help the model learn more transferable Gaussian representations.

Object Classification on Clean Object Datasets. ModelNet10 and ModelNet40[39] are used to evaluate classification performance on clean object data. These datasets mainly consist of regular CAD models with limited background noise and scanning incompleteness. They therefore serve as a complement to ScanObjectNN and are used to examine whether GaussFusion remains efective on objects with clear structures. Table 2 presents comparisons with classical point cloud classification methods and Gaussian-MAE.

On both ModelNet datasets, GaussFusion outperforms Gaussian-MAE[12]. This result shows that the proposed pre-training strategy not only improves robustness in

Table 3: Part segmentation results on the ShapeNetPart dataset. We report the mean IoU across all part categories mIoU<sub>C</sub> (%) and the mean IoU across all instances mIoU<sub>I</sub> (%), as well as the IoU (%) for each category. <sup>†</sup> denotes reproduced results under our implementation.

<table><tr><td>Method</td><td> $mIoU_C$ </td><td> $mIoU_I$ </td><td>aero lamp</td><td>bag laptop</td><td>cap motor</td><td>car mug</td><td>chair pistol</td><td>e-phone rocket</td><td>guitar s-board</td><td>knife table</td></tr><tr><td rowspan="2">PointNet[2]</td><td rowspan="2">80.39</td><td rowspan="2">83.7</td><td>83.4</td><td>78.7</td><td>82.5</td><td>74.9</td><td>89.6</td><td>73.0</td><td>91.5</td><td>85.9</td></tr><tr><td>80.8</td><td>95.3</td><td>65.2</td><td>93.0</td><td>81.2</td><td>57.9</td><td>72.8</td><td>80.6</td></tr><tr><td rowspan="2">PointNet++[3]</td><td rowspan="2">81.85</td><td rowspan="2">85.1</td><td>82.4</td><td>79.0</td><td>87.7</td><td>77.3</td><td>90.8</td><td>71.8</td><td>91.0</td><td>85.9</td></tr><tr><td>83.7</td><td>95.3</td><td>71.6</td><td>94.1</td><td>81.3</td><td>58.7</td><td>76.4</td><td>82.6</td></tr><tr><td rowspan="2">DGCNN[42]</td><td rowspan="2">82.33</td><td rowspan="2">85.2</td><td>84.0</td><td>83.4</td><td>86.7</td><td>77.8</td><td>90.6</td><td>74.7</td><td>91.2</td><td>87.5</td></tr><tr><td>82.8</td><td>95.7</td><td>66.3</td><td>94.9</td><td>81.1</td><td>63.5</td><td>74.5</td><td>82.6</td></tr><tr><td rowspan="2">Transformer[7]</td><td rowspan="2">83.42</td><td rowspan="2">85.1</td><td>82.9</td><td>85.4</td><td>87.7</td><td>78.8</td><td>90.5</td><td>80.8</td><td>91.1</td><td>87.7</td></tr><tr><td>85.3</td><td>95.6</td><td>73.9</td><td>94.9</td><td>83.5</td><td>61.2</td><td>74.9</td><td>80.6</td></tr><tr><td rowspan="2">Point-BERT[7]</td><td rowspan="2">84.11</td><td rowspan="2">85.6</td><td>84.3</td><td>84.8</td><td>88.0</td><td>79.8</td><td>91.0</td><td>81.7</td><td>91.6</td><td>87.9</td></tr><tr><td>85.2</td><td>95.6</td><td>75.6</td><td>94.7</td><td>84.3</td><td>63.4</td><td>76.3</td><td>81.5</td></tr><tr><td rowspan="2">Point-MAE[8]</td><td rowspan="2">84.19</td><td rowspan="2">86.1</td><td>84.3</td><td>85.0</td><td>88.3</td><td>80.5</td><td>91.3</td><td>78.5</td><td>92.1</td><td>87.4</td></tr><tr><td>86.1</td><td>96.1</td><td>75.2</td><td>94.6</td><td>84.7</td><td>63.5</td><td>77.1</td><td>82.4</td></tr><tr><td rowspan="2">Gaussian-MAE $^†$ [12]</td><td rowspan="2">83.65</td><td rowspan="2">85.8</td><td>84.8</td><td>84.8</td><td>90.2</td><td>80.2</td><td>90.8</td><td>73.1</td><td>91.8</td><td>87.6</td></tr><tr><td>84.4</td><td>95.9</td><td>76.6</td><td>95.9</td><td>85.3</td><td>61.5</td><td>73.9</td><td>81.9</td></tr><tr><td rowspan="2">GaussFusion</td><td rowspan="2">84.43</td><td rowspan="2">86.3</td><td>85.2</td><td>84.2</td><td>87.7</td><td>80.4</td><td>91.4</td><td>72.6</td><td>92.2</td><td>87.5</td></tr><tr><td>84.4</td><td>95.9</td><td>78.8</td><td>95.5</td><td>84.8</td><td>65.8</td><td>75.6</td><td>82.2</td></tr></table>

real-world scenarios, but also enhances category discrimination on clean object data. Since some point cloud methods in the table use diferent input forms and training protocols, they are mainly included as reference baselines. A more direct comparison is between Gaussian-MAE and GaussFusion, as both adopt the same Gaussian input setting and thus more clearly reflect the efects of multimodal guidance and the GSHM masking strategy.

## 4.2.2. Segmentation Experiments

ShapeNetPart[40] is used to evaluate fine-grained local structure understanding. Unlike object classification, part segmentation requires the model not only to recognize the object category, but also to predict the corresponding semantic part label for each point. Therefore, this task further examines whether the pre-trained Gaussian representations preserve local geometry and part-level semantic information.

The experiments use the same dataset split and part annotations as those used by point cloud methods. Since the input is a 3D Gaussian representation while the supervision signals in ShapeNetPart are defined on point cloud positions, Gaussian data are aligned with the original point cloud annotations according to the mapping files, and segmentation results are computed at the annotated point cloud positions. This protocol avoids unfair comparisons caused by diferent supervision locations. Table 3 reports the segmentation results of diferent methods on ShapeNetPart. Fig. 5 provides qualitative visualization results of our part segmentation predictions.

Table 4: Few-shot object classification on ModelNet40. We report the average accuracy (%) and standard deviation (%) of 10 independent experiments.  denotes reproduced results under our implementation. Bold denotes the best result among Gaussian-representation methods.

<table><tr><td>Method</td><td>5-way, 10-shot</td><td>5-way, 20-shot</td><td>10-way, 10-shot</td><td>10-way, 20-shot</td></tr><tr><td colspan="5">with Point Cloud Pre-training Representation</td></tr><tr><td>DGCNN-rand[50]</td><td>31.6 ± 2.8</td><td>40.8 ± 4.6</td><td>19.9 ± 2.1</td><td>16.9 ± 1.5</td></tr><tr><td>+OcCo[50]</td><td>90.6 ± 2.8</td><td>92.5 ± 1.9</td><td>82.9 ± 1.3</td><td>86.5 ± 2.2</td></tr><tr><td>Transformer[7]</td><td>87.8 ± 5.2</td><td>93.3 ± 4.3</td><td>84.6 ± 5.5</td><td>89.4 ± 6.3</td></tr><tr><td>+OcCo[7]</td><td>94.0 ± 3.6</td><td>95.9 ± 2.3</td><td>89.4 ± 5.1</td><td>92.4 ± 4.6</td></tr><tr><td>Point-BERT[7]</td><td>94.6 ± 3.1</td><td>96.3 ± 2.7</td><td>91.0 ± 5.4</td><td>92.7 ± 5.1</td></tr><tr><td>Point-MAE[8]</td><td>96.3 ± 2.5</td><td>97.8 ± 1.8</td><td>92.6 ± 4.1</td><td>95.0 ± 3.0</td></tr><tr><td>Mamba3D[44]</td><td>96.4 ± 2.2</td><td>98.2 ± 1.2</td><td>92.4 ± 4.1</td><td>95.2 ± 2.9</td></tr><tr><td colspan="5">with 3D Gaussian Pre-training Representation</td></tr><tr><td>Gaussian-MAE†[12]</td><td>94.0 ± 3.5</td><td>95.1 ± 3.6</td><td>87.3 ± 6.1</td><td>92.2 ± 4.7</td></tr><tr><td>GaussFusion</td><td>96.1(+2.1) ± 2.4</td><td>96.5(+1.4) ± 2.3</td><td>88.9(+1.6) ± 5.9</td><td>93.1(+0.9) ± 4.9</td></tr></table>

The results show that GaussFusion outperforms Gaussian-MAE in both mIoU<sub>C</sub> and mIoU<sub>I</sub>, indicating improved transferability of the Gaussian encoder to part-level prediction. Across categories, GaussFusion achieves better results on categories containing thin parts or local structural variations, suggesting that the GSHM masking strategy encourages the model to focus on more discriminative local regions. Meanwhile, GaussFusion does not surpass Gaussian-MAE on several simple categories. A possible reason is that, for categories with simple shapes, limited training samples, or ambiguous part boundaries, image- and text-level supervision tends to emphasize global semantics and provides limited help for fine-grained part boundaries.

## 4.2.3. Few-shot Classification

Few-shot classification experiments evaluate the transferability of pre-trained rep resentations in low-annotation scenarios and directly assess whether the model learns transferable category semantics. The experiments are conducted on ModelNet40[39] under 5-way and 10-way settings, with 10-shot and 20-shot training samples, respectively. Each setting is independently run 10 times, and the mean accuracy and standard deviation are reported. Table 4 presents the few-shot classification results.

Since the point cloud methods in Table 4 are pre-trained and transferred directly in the point cloud space, their results are included mainly as reference baselines. The primary comparison is therefore conducted with Gaussian-MAE[12], which adopts the same 3D Gaussian representation and downstream evaluation protocol as Gauss-Fusion. GaussFusion consistently outperforms Gaussian-MAE across all four fewshot settings, indicating that multimodal semantic supervision and GSHM improve the transferability of Gaussian representations under limited annotations. The improvement is more pronounced under the 5-way settings, which may indicate that when the number of categories is smaller, the semantic priors provided by image and text supervision can more efectively help the model form clear inter-class boundaries. Therefore, the few-shot experiments further demonstrate that the benefits of GaussFusion come not only from fine-tuning under fully annotated conditions, but also from transfer in low-annotation scenarios.

## 4.3. Ablation Study

## 4.3.1. Efect of Multimodal Supervision

The roles of image and text supervision are evaluated under both random masking and GSHM. As shown in Table 5, introducing either modality consistently improves classification performance, while combining the two leads to further gains under both masking strategies. Text supervision brings a slightly larger improvement than image supervision. One possible reason is that the text input is derived from category descriptions, which directly provide object-level semantic constraints. In contrast, the image input is rendered from a single view or a limited number of views and is more susceptible to variations in viewpoint, occlusion, and appearance. The best result is achieved when image and text supervision are used together, indicating that the two modalities provide complementary guidance and jointly encourage the Gaussian en coder to learn more transferable representations.

Table 6  
Table 5  
Ablation study of GSHM and cross-modal supervision on ScanObjectNN OBJ-BG. Overall accuracy (%) is reported.

<table><tr><td rowspan="2">GSHMMask</td><td colspan="2">Cross-modal Inputs</td><td rowspan="2">ScanObjectNN</td></tr><tr><td>Image Input</td><td>Text Input</td></tr><tr><td>X</td><td>X</td><td>X</td><td>84.30</td></tr><tr><td>X</td><td>√</td><td>X</td><td>85.89</td></tr><tr><td>X</td><td>X</td><td>√</td><td>86.23</td></tr><tr><td>X</td><td>√</td><td>√</td><td>87.95</td></tr><tr><td>√</td><td>X</td><td>X</td><td>85.03</td></tr><tr><td>√</td><td>√</td><td>X</td><td>86.91</td></tr><tr><td>√</td><td>X</td><td>√</td><td>87.43</td></tr><tr><td>√</td><td>√</td><td>√</td><td>88.98</td></tr></table>

Ablation study of diferent image encoders on ScanObjectNN. The text encoder is fixed to CLIP-B. Overall accuracy (%) is reported.

<table><tr><td>Image Encoder</td><td>Acc. (%)</td></tr><tr><td>BEiT-B[51]</td><td>87.61</td></tr><tr><td>ResNet-50[52]</td><td>87.95</td></tr><tr><td>CLIP-B[14]</td><td>88.30</td></tr><tr><td>Swin-B[53]</td><td>88.64</td></tr><tr><td>ViT-B[15]</td><td>88.98</td></tr></table>

## 4.3.2. Efect of Masking Strategy

The efectiveness of GSHM is evaluated by comparing it with random masking under the same cross-modal supervision settings. Table 5 shows that GSHM consistently improves classification performance across all image and text input combinations, indicating that spatially coherent masking provides more efective reconstruction targets for multimodal Gaussian pre-training. Fig. 6(a) further shows that performance gradually improves as the masking ratio increases from 0.2 to 0.6 and decreases when the ratio is further increased. This result suggests that a moderate masking ratio provides an appropriate balance between reconstruction dificulty and visible context. Therefore, the masking ratio is set to 0.6 in the main experiments.

## 4.3.3. Efect of Diferent Encoders

This section analyzes the influence of diferent image encoders on GaussFusion. The Gaussian reconstruction objective, text encoder[14], and GSHM masking strategy are kept unchanged, while only the pre-trained encoder in the image branch is replaced. Table 6 shows that all evaluated visual encoders provide efective supervision for Gaussian pre-training, with ViT achieving the best result.

Although the text branch uses the CLIP text encoder[14], the image branch adopts ViT[15] as the image encoder rather than the CLIP image encoder. It is observed that the image and text encoders in CLIP are jointly trained in the same image-text alignment space, so their supervision signals may have strong overlap. In contrast, the ViT image encoder provides more independent visual structural features. When combined with the CLIP text encoder, the image and text branches form a clearer complementary relationship.

![](images/d0d60fa44fc2ab02d5b2fefd550d2dda89b21c5e1fe4caf1ddf0672772ad5886.jpg)  
(a) Masking Ratio

![](images/52498bbad8cf8cbb9154e1915eed35d85acd03e0b923661708fbe2b04fe07b11.jpg)  
(b) Image and Text Loss Weights  
Fig. 6. Ablation study on masking ratio and multimodal loss weights. (a) shows the efect of GSHM masking ratios on ScanObjectNN OBJ-BG. (b) shows the efect of image and text loss weights, with the best accuracy of 88.98% at $\lambda _ { \mathrm { i m g } } = \lambda _ { \mathrm { t e x t } } = 1$

## 4.3.4. Hyperparameter Analysis

The sensitivity to multimodal loss weights is studied by varying the image alignment weight $\lambda _ { \mathrm { i m g } }$ and the text alignment weight $\lambda _ { \mathrm { t e x t } }$ . The Gaussian reconstruction objective, GSHM masking strategy, and other training settings are kept unchanged. Fig. 6 shows the performance landscape under diferent weight combinations.

The results show that when the two cross-modal loss weights are small, the model mainly relies on the Gaussian reconstruction objective and receives insuficient semantic supervision, leading to relatively low classification performance on ScanObjectNN[38]. As $\lambda _ { \mathrm { i m g } }$ and $\lambda _ { \mathrm { t e x t } }$ increase, the model performance gradually improves, indicating that image and text features provide efective semantic constraints for Gaussian representation learning. The best result is achieved when both weights are set to 1. Further increasing the weights leads to performance degradation, suggesting that overly strong cross-modal alignment shifts the training objective toward the image-text semantic space and weakens the focus on local Gaussian geometry and attribute reconstruction.

Based on this analysis, both $\lambda _ { \mathrm { i m g } }$ and $\lambda _ { \mathrm { t e x t } }$ are set to 1 in the main experiments.

## 5. Conclusion

In this paper, we propose GaussFusion, a multimodal self-supervised pre-training framework for 3D Gaussian representations. By combining masked Gaussian reconstruction with image and text supervision, GaussFusion learns representations that preserve local Gaussian structure while absorbing high-level semantic information from external modalities. We further introduce GSHM, a salience-guided multi scale masking strategy designed for the non-uniform distribution of Gaussian primitives. Experiments on classification, segmentation, and few-shot recognition show that GaussFusion improves the transferability of Gaussian representations across diferent downstream tasks. Current multimodal supervision is mainly derived from rendered images and category-level text, and future work will explore richer semantic annotations and larger scene-level Gaussian representations.

## CRediT authorship contribution statement

Zhixuan You: Conceptualization, Methodology, Writing - Original Draft. Jihua Zhu: Software, Investigation. Yiding Sun: Resources, Visualization. Zihao Guo: Investigation. Haozhe Cheng: Writing - Review and Editing. Dongxu Zhang: Writing - Review and Editing. Lin Chen: Writing - Review and Editing. Hainan Luo: Writing - Review and Editing.

## References

[1] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, 3d gaussian splatting for real-time radiance field rendering, ACM Transactions on Graphics 42 (4) (2023) 139:1–139:14. doi:10.1145/3592433.

[2] C. R. Qi, H. Su, K. Mo, L. J. Guibas, Pointnet: Deep learning on point sets for 3d classification and segmentation, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 652–660.

[3] C. R. Qi, L. Yi, H. Su, L. J. Guibas, Pointnet++: Deep hierarchical feature learning on point sets in a metric space, in: Advances in Neural Information Processing Systems, Vol. 30, 2017.

[4] J. L. Schönberger, J.-M. Frahm, Structure-from-motion revisited, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4104–4113.

[5] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, R. Ng, NeRF: Representing scenes as neural radiance fields for view synthesis, in: European Conference on Computer Vision (ECCV), Springer, 2020, pp. 405–421.

[6] T. Müller, A. Evans, C. Schied, A. Keller, Instant neural graphics primitives with a multiresolution hash encoding, ACM Transactions on Graphics 41 (4) (2022) 102:1–102:15.

[7] X. Yu, L. Tang, Y. Rao, T. Huang, J. Zhou, J. Lu, Point-bert: Pre-training 3d point cloud transformers with masked point modeling, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 19313–19322.

[8] Y. Pang, W. Wang, F. E. H. Tay, W. Liu, Y. Tian, L. Yuan, Masked autoencoders for point cloud self-supervised learning, in: European Conference on Computer Vision (ECCV), Springer, 2022, pp. 604–621.

[9] M. Afham, I. Dissanayake, D. Dissanayake, A. Dharmasiri, K. Thilakarathna, R. Rodrigo, Crosspoint: Self-supervised cross-modal contrastive learning for 3d point cloud understanding, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 9902–9912.

[10] R. Dong, Z. Qi, L. Zhang, J. Zhang, J. Sun, Z. Ge, L. Yi, K. Ma, Autoencoders as cross-modal teachers: Can pretrained 2d image transformers help 3d representation learning?, in: International Conference on Learning Representations (ICLR), 2023.

[11] Z. Qi, R. Dong, G. Fan, Z. Ge, X. Zhang, K. Ma, L. Yi, Contrast with reconstruct: Contrastive 3d representation learning guided by generative pretraining, in: Proceedings of the 40th International Conference on Machine Learning, Vol. 202 of Proceedings of Machine Learning Research, PMLR, 2023, pp. 28223–28243.

[12] Q. Ma, Y. Li, B. Ren, N. Sebe, E. Konukoglu, T. Gevers, L. Van Gool, D. P. Paudel, A large-scale dataset of gaussian splats and their self-supervised pretraining, in: 2025 International Conference on 3D Vision (3DV), IEEE, 2025, pp. 145–155. doi:10.1109/3DV66043.2025.00019.

[13] Y. Li, Q. Ma, R. Yang, H. Li, M. Ma, B. Ren, N. Popovic, N. Sebe, E. Konukoglu, T. Gevers, L. Van Gool, M. R. Oswald, D. P. Paudel, Scenesplat: Gaussian splatting-based scene understanding with vision-language pretraining, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 4961–4972.

[14] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, I. Sutskever, Learning transferable visual models from natural language supervision, in: Proceedings of the 38th International Conference on Machine Learning, Vol. 139 of Proceedings of Machine Learning Research, PMLR, 2021, pp. 8748–8763.

[15] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby, An image is worth 16x16 words: Transformers for image recognition at scale, in: International Conference on Learning Representations (ICLR), 2021.

[16] Y. Wang, Y. Sun, Q. Wang, P. Li, C. Lu, D. Zhang, Pointrft: Explicit reinforcement fine-tuning for point cloud few-shot learning, in: IEEE International Conference on Multimedia and Expo (ICME2026), 2026.

[17] D. Zhang, Y. Wang, Y. Sun, H. Xu, P. Fan, J. Zhu, Cmhanet: A cross-modal hybrid attention network for point cloud registration, Neurocomputing (2026).

[18] X. Han, Y. Sun, C. Lu, Rethinking regressor in 3d gaussian pretraining, in: Pattern Recognit. Comput. Vis., 2026, pp. 177–190.

[19] Y. Sun, J. Zhu, H. Cheng, C. Lu, Z. Yang, L. Chen, Y. Wang, Align then adapt: Rethinking parameter-eficient transfer learning in 4d perception, IEEE Trans. Multimedia (2026).

[20] Y. Sun, H. Cheng, C. Lu, Z. Li, M. Wu, H. Lu, J. Zhu, Hyperpoint: Multimodal 3d foundation model in hyperbolic space, Pattern Recognit. 173 (2026) 112800.

[21] P. Li, Y. Sun, H. Cheng, Pointdico: Contrastive 3d representation learning guided by difusion models, arXiv preprint arXiv:2512.08330 (2025).

[22] R. Zhang, Z. Guo, P. Gao, R. Fang, B. Zhao, D. Wang, Y. Qiao, H. Li, Point-m2ae: Multi-scale masked autoencoders for hierarchical point cloud pretraining, in: Advances in Neural Information Processing Systems, Vol. 35, 2022, pp. 27061–27074.

[23] H. Cheng, J. Zhu, N. Hu, J. Chen, W. Yan, PTM: Torus masking for 3d representation learning guided by robust and trusted teachers, IEEE Transactions on Circuits and Systems for Video Technology 34 (12) (2024) 12158–12170. doi:10.1109/TCSVT.2024.3430904.

[24] J. Zhou, Y. Song, C. Chiu, Y. Xiong, Y. Luo, S. Song, CPG: Contrastive patchgraph learning for 3d point cloud, Pattern Recognition 169 (2026) 111954. doi:10.1016/j.patcog.2025.111954.

[25] D. Liang, X. Zhou, W. Xu, X. Zhu, Z. Zou, X. Ye, X. Tan, X. Bai, Pointmamba: A simple state space model for point cloud analysis, in: Advances in Neural Information Processing Systems, Vol. 37, 2024, pp. 32653–32677. doi:10.52202/079017-1026.

[26] Z. Guo, J. Zhu, J. Liu, A. S. Mian, Mantis: Mamba-native tuning is eficient for 3d point cloud foundation models, arXiv preprint arXiv:2605.03438 (2026). doi:10.48550/arXiv.2605.03438.

[27] A. Guédon, V. Lepetit, SuGaR: Surface-aligned gaussian splatting for eficient 3d mesh reconstruction and high-quality mesh rendering, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 5354–5363.

[28] J. C. Lee, D. Rho, X. Sun, J. H. Ko, E. Park, Compact 3d gaussian representation for radiance field, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21719–21728.

[29] S. Zhou, H. Chang, S. Jiang, Z. Fan, Z. Zhu, D. Xu, P. Chari, S. You, Z. Wang, A. Kadambi, Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21676–21685.

[30] M. Qin, W. Li, J. Zhou, H. Wang, H. Pfister, Langsplat: 3d language gaussian splatting, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20051–20060.

[31] X. Zhu, R. Zhang, B. He, Z. Guo, Z. Zeng, Z. Qin, S. Zhang, P. Gao, Pointclip v2: Prompting clip and gpt for powerful 3d open-world learning, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 2639–2650.

[32] R. Huang, X. Pan, H. Zheng, H. Jiang, Z. Xie, C. Wu, S. Song, G. Huang, Joint representation learning for text and 3d point cloud, Pattern Recognition 147 (2024) 110086. doi:10.1016/j.patcog.2023.110086.

[33] H. Zhang, L. Yu, G. Wang, S. Tian, Z. Yu, W. Li, X. Ning, Cross-modal knowl edge transfer for 3d point clouds via graph ofset prediction, Pattern Recognition 162 (2025) 111351. doi:10.1016/j.patcog.2025.111351.

[34] H. Cheng, X. Han, P. Shi, J. Zhu, Z. Li, Multi-trusted cross-modal information bottleneck for 3d self-supervised representation learning, Knowledge-Based Systems 283 (2024) 111217. doi:10.1016/j.knosys.2023.111217.

[35] N. Hu, H. Cheng, Y. Xie, P. Shi, J. Zhu, Hyperbolic image-and-pointcloud contrastive learning for 3d classification, in: 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE, 2024, pp. 4973–4979. doi:10.1109/IROS58592.2024.10802543.

[36] L. Xue, M. Gao, C. Xing, R. Martín-Martín, J. Wu, C. Xiong, R. Xu, J. C. Niebles, S. Savarese, Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 1179–1189.

[37] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention is all you need, in: Advances in Neural Information Processing Systems, Vol. 30, 2017, pp. 5998–6008.

[38] M. A. Uy, Q.-H. Pham, B.-S. Hua, T. Nguyen, S.-K. Yeung, Revisiting point cloud classification: A new benchmark dataset and classification model on realworld data, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 1588–1597.

[39] Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, J. Xiao, 3d shapenets: A deep representation for volumetric shapes, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 1912– 1920.

[40] L. Yi, V. G. Kim, D. Ceylan, I.-C. Shen, M. Yan, H. Su, C. Lu, Q. Huang, A. Shefer, L. Guibas, A scalable active framework for region annotation in 3d shape collections, ACM Transactions on Graphics 35 (6) (2016) 210:1–210:12.

[41] Y. Xu, T. Fan, M. Xu, L. Zeng, Y. Qiao, Spidercnn: Deep learning on point sets with parameterized convolutional filters, in: Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 87–102.

[42] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, J. M. Solomon, Dynamic graph cnn for learning on point clouds, ACM Transactions on Graphics (tog) 38 (5) (2019) 146:1–146:12. doi:10.1145/3326362.

[43] Y. Li, R. Bu, M. Sun, W. Wu, X. Di, B. Chen, Pointcnn: Convolution on xtransformed points, in: Advances in Neural Information Processing Systems, Vol. 31, 2018.

[44] X. Han, Y. Tang, Z. Wang, X. Li, Mamba3d: Enhancing local features for 3d point cloud analysis via state space model, in: Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 4995–5004. doi:10.1145/3664647.3681173.

[45] M. Atzmon, H. Maron, Y. Lipman, Point convolutional neural networks by extension operators, ACM Transactions on Graphics 37 (4) (2018) 71:1–71:12.

[46] A. Komarichev, Z. Zhong, J. Hua, A-cnn: Annularly convolutional neural networks on point clouds, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 7421–7430.

[47] X. Yan, C. Zheng, Z. Li, S. Wang, S. Cui, Pointasnl: Robust point clouds processing using nonlocal neural networks with adaptive sampling, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 5589–5598.

[48] Q. Wu, Q. Zhang, C. Tan, Y. Zhou, C. Sun, Point-to-spike residual learning for energy-eficient 3d point cloud classification, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38, 2024, pp. 6092–6099. doi:10.1609/aaai.v38i6.28425.

[49] T. Zhang, H. Yuan, L. Qi, J. Zhang, Q. Zhou, S. Ji, S. Yan, X. Li, Point cloud mamba: Point cloud learning via state space model, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, 2025, pp. 10121–10130. doi:10.1609/aaai.v39i10.33098.

[50] H. Wang, Q. Liu, X. Yue, J. Lasenby, M. J. Kusner, Unsupervised point cloud pre-training via occlusion completion, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 9782–9792.

[51] H. Bao, L. Dong, S. Piao, F. Wei, BEiT: BERT pre-training of image transformers, in: International Conference on Learning Representations (ICLR), 2022.

[52] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.

[53] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, B. Guo, Swin transformer: Hierarchical vision transformer using shifted windows, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 10012–10022.