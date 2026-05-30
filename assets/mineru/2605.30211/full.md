# Cycle Consistency in Video Object-Centric Learning

Rongzhen Zhao1

rongzhen.zhao@aalto.fi

Zhiyuan Li1

6 zhiyuan.li@aalto.fi

2Ruonan Wei2

2ruonan2765@gmail.com

yJuho Kannala3,4

ajuho.kannala@aalto.fi

MJoni Pajarinen1

joni.pajarinen@aalto.fi

1 Department of Electrical Engineering and Automation, Aalto University, Espoo, Finland

2 School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, Wuhan, China

3 Department of Computer Science, Aalto University, Espoo, Finland

4 Center for Machine Vision and Signal Analysis, University of Oulu, Oulu, Finland

# Abstract

Self-supervised video Object-Centric Learning (OCL) aims to discover distinct objects and associate them across time, whereas self-supervised Multi-Object Tracking (MOT) focuses on associating pre-defined object detections or segmentations. Although well-established in MOT, Cycle Consistency (CC) cannot naively or explicitly apply to the latent slot space of OCL. Unlike the deterministic and ideal object representations in MOT, OCL slots are inherently stochastic and ambiguous due to non-unique scene decompositions. Enforcing explicit cycle consistency (ECC) on slots imposes rigid mean seeking. This severely penalizes the model for exploring alternative but equally valid decompositions, thereby driving towards feature collapse. To resolve this dilemma, we propose Implicit Cycle Consistency (ICC), which shifts the cycle-consistency constraint from the restrictive slot space to the continuous reconstruction manifold, encouraging slots to reach a soft consensus on collectively interpreting the visual scene rather than forcing rigid point-to-point feature alignment. Extensive experiments on complex video OCL benchmarks demonstrate that ICC avoids feature collapse and outperforms ECC baselines. Our source code is provided as the supplement.

# 1 Introduction

Self-supervised video representation learning has advanced along two representative directions: Multi-Object Tracking (MOT) [18] and Object-Centric Learning (OCL) [16]. Selfsupervised MOT associates object identities across frames without trajectory labels given ideal object detections or segmentations. Whereas, OCL decomposes each video frame into objects (and background) and associate them temporally to represent the visual scene with minimal information loss. Understanding their connection can support the improvement of video OCL for visual scene representation and understanding [20].

![](images/c53d06ea609303cbb255e9ab3fbfc8b1dc8f6dbdf82c17a77b425cec0892af26.jpg)

<details>
<summary>text_image</summary>

t-th frame
ECC
ICC
forward
AcaWhips Net
AcaWhips Net
AcaWhips Net
AcaWhips Net
</details>

Figure 1: Decomposition Divergence. We visualize the attention masks of frame t from Forward and Backward streams. (middle) Explicit Cycle Consistency (ECC) forces alignment, causing conflict and blurriness. (right) Implicit Cycle Consistency (ICC) allows the Forward stream to clearly segment the car, land, tree and sky, while decomposing the car into different sets of the car’s parts. Gaussian-smoothed for better presentation.

Although MOT has been studied extensively, its insights have rarely been utilized for improving video OCL. A cornerstone of MOT is Cycle Consistency (CC), which enforces agreement between forward and backward object associations [12]: an object’s trajectory tracked from frame t to t + k and back to t should return to the original object, or their forward-backward trajectories should overlap. Intuitively, this utilizes the video’s inherent temporal coherence: if an object can be tracked reversibly with negligible drift, the model learns robust associations directly from raw video. Incorporating this principle into video OCL suggests a promising route to unify object discovery and temporal association.

However, we demonstrate that naively applying explicit CC (ECC) to video OCL is fundamentally ill-posed. We identify a core conflict: unlike MOT’s ideal object representations, OCL’s slot representations are inherently stochastic and ambiguous. The self-supervised decomposition of a visual scene can have different possibilities. When modeling a car, different decompositions may partition it into body + wheels, or alternatively into windshield + hood + the remaining. So the forward and backward streams often converge to distinct, yet equally valid, decompositions [5]. Consequently, enforcing hard alignment in the slot space (latent space) penalizes the model for exploring these valid variations. As shown in Figure 1 middle and Section 3.2, this drives representations toward a collapsed average, smoothing out discriminative features essential for object discovery.

To address this, we propose Implicit CC (ICC) for video OCL. We align the forward and backward streams on the reconstruction manifold (observation space), enforcing a soft consensus. As shown in Figure 1 right and Section 3.3, while allowed to diverge in the latent space to accommodate stochasticity and ambiguity, forward-backward slots must reach a consensus on explaining the visual scene, i.e., the reconstruction. This leverages the temporal self-supervision of CC without suffering from feature collapse.

In this work we make the following contributions: (c1) We articulate a fundamental conflict when adapting cycle consistency from MOT to video OCL, showing that rigid latent alignment suppresses the slots’ capacity to handle scene-decomposition ambiguities. (c2)

We enforce temporal consistency implicitly on the reconstruction manifold, allowing slots to maintain optimization flexibility while capturing robust temporal correlations across frames. (c3) ICC improves object discovery on complex video datasets, outperforming explicit alignment baselines while demonstrating strong resistance to feature degradation.

# 2 Related Work

We discuss related work organized into three key directions, omitting the attributive “selfsupervised” for brevity.

# 2.1 Image Object-Centric Learning

Image OCL serves as the spatial foundation for video OCL. Most existing methods adopt an encode–aggregate–decode architecture [9, 24]. The encoder usually employs some Vision Foundation Models (VFMs) for feature extraction. The aggregator is typically based on slot attention [9], which aggregates VFM features into mutually competing slots to obtain object-level representations, i.e., slots; the attention maps of these slots can be used for object segmentation, i.e., object discovery. The decoder reconstructs the input from slots, providing self-supervision that encourages each slot to capture as much information as possible.

Recent advances can be categorized by the affected modules. Methods for improving aggregation include [2, 6, 25]. Methods for improving decoding include [7, 21, 26]. Methods for improving reconstruction include [14, 15, 24].

# 2.2 Video Object-Centric Learning

Video OCL is a temporal extension of image OCL, performing image OCL on each video frame while connecting frames recurrently via a transition module [16]. Namely, video OCL not only decomposes each frame into objects but also associate them across time.

Recent advances can be categorized by the affected modules. Methods improving reconstruction include: SAVi and SAVi++ [3, 8] using flow and depth for weakly-supervised object separation signal; VideoSAUR [22] predicting patch movement for temporal consistency. Methods improving transitioning include: SlotContrast [11] introducing slot contrastive loss for temporal consistency; RandSF.Q [23] predicting next queries from a random slot-feature pair with relative time information for implicit transition dynamics modeling.

Despite these advances, existing Video OCL methods typically rely on forward-only stream, lacking verifications on forward-backward consistency.

# 2.3 Multi-Object Tracking

To learning how to associate objects through time without trajectory labels, i.e., selfsupervised MOT, some form of consistency has to be utilized. Cross-view consistency [1] requires invariance across view perturbations. Path consistency [10] enforces stability across variable temporal strides. Graph-based methods like [13] utilize temporal appearance graphs to maintain coherence. Temporal consistency [4] maintains similar appearance in a tracking of one object. We specifically focus on cycle-consistency (CC) [12, 27], where an object trajectory in forward and backward streams should overlap.

In whichever case, explicit consistency regularization is enforced on object representations directly. This is effective because object representations are extracted from ideal detections or segmentations. However, this does not hold in video OCL, where naively applying explicit CC is actually harmful.

# 3 Proposed Method

In this section, we present our framework for integrating cycle consistency into unsupervised video Object-Centric Learning (OCL). We first outline the video OCL formulation based on the most recent state-of-the-art methods, RandSF.Q [23] and SmoothSA [25], where our method is built upon. We then analyze the limitations of applying explicit cycle constraints (common in MOT) to the OCL setting. Finally, we introduce our Implicit Bi-directional Consensus mechanism, which leverages stochastic slot dynamics to enforce temporal consistency without hindering object discovery.

# 3.1 Preliminary: Video Object-Centric Learning

Given a video clip of T frames, we extract visual features $\{ \pmb { F } _ { t } \in \mathbb { R } ^ { h \times w \times c } \} _ { t = 1 } ^ { T }$ using a pre-trained encoder, as formalized in prior work [24]. The goal of video OCL is to map these features to a set of S slot vectors $\pmb { S } _ { t } \in \mathbb { R } ^ { S \times c }$ for each time step t, representing objects and background in the t-th video frame.

Denote the typical video OCL process as forward stream $\mathcal { T } _ { \mathrm { f w } }$ , which operates as below:

$$
\text { initialization } \quad Q _ {1} = \boldsymbol {\phi} _ {\mathrm{n}} (\boldsymbol {C}) \quad t = 1 \tag {1a}
$$

$$
\text { transition } \quad \boldsymbol {Q} _ {t} = \boldsymbol {\phi} _ {\mathrm{r}} (\boldsymbol {S} _ {t - 1}, \boldsymbol {F} _ {t}) + \boldsymbol {\varepsilon} \quad t > 1 \tag {1b}
$$

$$
\text { aggregation } \quad \boldsymbol {S} _ {t}, \boldsymbol {A} _ {t} = \boldsymbol {\phi} _ {\mathrm{a}} (\boldsymbol {Q} _ {t}, \boldsymbol {F} _ {t}) \tag {1c}
$$

$$
\text { decoding } \quad \hat {\boldsymbol {F}} _ {t} = \boldsymbol {\phi} _ {\mathrm{d}} (\boldsymbol {S} _ {t}) \tag {1d}
$$

Initialization: if $t = 1$ , initializer $\phi _ { \mathrm { n } }$ transforms clue $c , \mathrm { e . g . }$ ., learned Gaussian samplings [9] or object bounding boxes [8], into slot queries $\pmb { Q } _ { 1 } \in \mathbb { R } ^ { S \times c }$ . Transition: if $t > 1$ , transitioner $\phi _ { \mathrm { r } }$ recurrently transforms previous frame’s slots $\pmb { S } _ { t - 1 }$ into next queries $\pmb { Q } _ { t }$ , with training stochasticity ε utilized in the most recent state-of-the-art, RandSF.Q [23] and SmoothSA [25]. Aggregation: aggregator $\pmb { \phi } _ { \mathrm { a } }$ , a Slot Attention module or its variants, iteratively aggregates information in feature $\pmb { F } _ { t }$ into queries $\pmb { Q } _ { t }$ , producing slots $\pmb { S } _ { t }$ , along with byproduct attention maps $\pmb { A } _ { t } \in \mathbb { R } ^ { S \times h \times w }$ , which can be binarized as object segmentation masks. Decoding: decoder $\phi _ { \mathrm { d } }$ decodes slots $\pmb { S } _ { t }$ into feature reconstruction $\hat { \pmb { F } } _ { t } \in \mathrm { \mathbb { R } } ^ { h \times w \times c }$ .

The self-supervised is achieved by minimizing the reconstruction error of all time steps:

$$
\mathcal {L} _ {\text { recon }} = \sum_ {t = 1} ^ {T} \| \hat {\boldsymbol {F}} _ {t} - \boldsymbol {F} _ {t} \| ^ {2} \tag {2}
$$

Note that there might be some auxiliary losses [11, 22], which are ignored for brevity.

# 3.2 Explicit Cycle Consistency

A naive intuition is to borrow technique Cycle Consistency (CC) from self-supervised Multiple Object Tracking (MOT) [10, 12], to enforce Cycle Consistency, which crucial in

![](images/b1c47a1ae634c56079d66aeac24be3e604ed5d4a91c354bbd0506bcf16221c30.jpg)  
F frame feature Qt St slot queries slot vectors feature recon.   
ˆS't Q't backward slot queries backward slot vectors backward feature recon.   
explicit cycle consistency loss implicit cycle consistency loss

Figure 2: Cycle Consistency in video OCL. (left) Baseline video OCL with forward-only stream. (middle) Explicit Cycle Consistency (ECC) applies a loss directly on forwardbackward slots, forcing hard latent alignment. We demonstrate this is ill-posed and leads to feature collapse due to decomposition ambiguity. (right) Implicit Cycle Consistency (ICC) applies the loss on forward-backward feature reconstruction. By aligning on the observation manifold rather than the latent space, ICC enforces explanation power while preserving latent representation diversity.

associating objects across time without external supervision:

$$
\text { initialization } \quad \boldsymbol {Q} _ {T - 1} ^ {\prime} = \boldsymbol {\phi} _ {\mathrm{r}} ^ {\prime} (\boldsymbol {S} _ {T}) + \boldsymbol {\varepsilon} \quad t = T - 1 \tag {3a}
$$

$$
\text { transition } \quad \boldsymbol {Q} _ {t} ^ {\prime} = \boldsymbol {\phi} _ {\mathrm{r}} ^ {\prime} (\boldsymbol {S} _ {t + 1} ^ {\prime}, \boldsymbol {F} _ {t}) + \boldsymbol {\varepsilon} \quad t <   T - 1 \tag {3b}
$$

$$
\text { aggregation } \quad \boldsymbol {S} _ {t} ^ {\prime}, \boldsymbol {A} _ {t} ^ {\prime} = \boldsymbol {\phi} _ {\mathrm{a}} (\boldsymbol {Q} _ {t} ^ {\prime}, \boldsymbol {F} _ {t} ^ {\prime}) \tag {3c}
$$

Initialization: if $t = T - 1$ , backward transitioner $\pmb { \phi } _ { \mathrm { r } } ^ { \prime }$ transforms the last forward slots ${ \pmb S } _ { T }$ into the first backward queries ${ \pmb Q } _ { T - 1 } ^ { \prime }$ . Transition: if $t < T - 1$ , transitioner $\pmb { \phi } _ { \mathrm { r } } ^ { \prime }$ recurrently transforms previous backward slots $\pmb { S } _ { t + 1 } ^ { \prime }$ into the next backward queries $\pmb { Q } _ { t } ^ { \prime }$ . Aggregation: aggregator $\pmb { \phi } _ { \mathrm { a } }$ iteratively aggregates information in feature $\pmb { F } _ { t }$ into backward queries $\pmb { Q } _ { t }$ , producing backward slots $\pmb { S } _ { t } ^ { \prime }$ , along with byproduct backward attention maps ${ \pmb A } _ { t } ^ { \prime } .$ .

Through CC, an object’s forward and backward associations through time should have overlapped trajectories. Mathematically, this implies an explicit regularization loss:

$$
\mathcal {L} _ {\mathrm{ECC}} = \sum_ {t = 1} ^ {T - 1} \left\| \boldsymbol {S} _ {t} ^ {\prime} - \boldsymbol {S} _ {t} \right\| ^ {2} \tag {4}
$$

where the explicit CC loss $\mathcal { L } _ { \mathrm { E C C } }$ is added into the original losses for joint optimization.

Intuition 1 (Feature Collapse from Decomposition Divergence) Applying explicit cycle consistency to video OCL is ill-posed. The unsupervised decomposition of a scene is inherently ambiguous, i.e., existing multiple valid ways to segment complex objects. Explicit alignment penalizes the model for exploring different decompositions, forcing collapse to averaged representation.

Formalism 1 In the forward stream $\mathcal { T } _ { f w ; }$ , let the set of slots $\pmb { S } _ { t }$ represent a decomposition hypothesis $\pmb { H } _ { t } \in \mathbb { R } ^ { S \times c }$ derived from the feature map $\pmb { F } _ { t }$ , where each $\pmb { H } _ { t , s }$ corresponds to a discovered visual entity. In the backward stream $\mathcal { T } _ { b w }$ , starting from T and propagating back, the accumulated stochastic noise ε leads the model to sample a different decomposition mode $\pmb { H } _ { t } ^ { \prime }$ . Crucially, given the self-supervision, $\pmb { H } _ { t }$ and $\pmb { H } _ { t } ^ { \prime }$ are likely to differ not just in permutation, but in content, e.g., $\pmb { H } _ { t , s }$ captures “rider + bike rear wheel” while $\pmb { H } _ { t , s } ^ { \prime }$ captures “rider + bike front wheel”. The explicit consistency loss imposes:

$$
\mathcal {L} _ {E C C} = \left\| \boldsymbol {S} _ {t} - \boldsymbol {S} _ {t} ^ {\prime} \right\| ^ {2} \approx \sum_ {s} \left\| \boldsymbol {H} _ {t, s} - \boldsymbol {H} _ {t, \pi (s)} ^ {\prime} \right\| ^ {2} \tag {5}
$$

where π is an implicit matching. Since $\pmb { H } _ { t } \neq \pmb { H } _ { t } ^ { \prime }$ due to decomposition divergence, the gradient acts to minimize the distance between two distinct visual concepts:

$$
\nabla_ {\boldsymbol {S} _ {t}} \mathcal {L} _ {E C C} = 2 (\boldsymbol {S} _ {t} - \boldsymbol {S} _ {t} ^ {\prime}) \approx 2 (\boldsymbol {H} _ {t} - \boldsymbol {H} _ {t} ^ {\prime}) \tag {6}
$$

This creates a “mean-seeking” force: Instead of refining the segmentation, this force pulls the slot representations towards the average of the diverging hypotheses $\mathbb { E } [ \pmb { H } _ { t , : } ]$ , smoothing out discriminative features and leading to the observed representation collapse (blurriness or background absorption).

# 3.3 Implicit Cycle Consistency

To effectively utilize temporal information without the drawbacks of explicit alignment, we introduce a symmetric backward stream $\mathcal { T } _ { \mathrm { b w } }$ and an implicit consensus objective.

Subsequent the forward stream, we instantiate a backward stream that processes the video frames inversely from $t = T$ down to 1:

$$
\text { initialization } \quad \boldsymbol {Q} _ {T - 1} ^ {\prime} = \boldsymbol {\phi} _ {\mathrm{r}} ^ {\prime} (\boldsymbol {S} _ {T}) + \boldsymbol {\varepsilon} \quad t = T - 1 \tag {7a}
$$

$$
\text { transition } \quad \boldsymbol {Q} _ {t} ^ {\prime} = \boldsymbol {\phi} _ {\mathrm{r}} ^ {\prime} (\boldsymbol {S} _ {t + 1} ^ {\prime}, \boldsymbol {F} _ {t}) + \boldsymbol {\varepsilon} \quad t <   T - 1 \tag {7b}
$$

$$
a g g r e g a t i o n \quad \boldsymbol {S} _ {t} ^ {\prime}, \boldsymbol {A} _ {t} ^ {\prime} = \boldsymbol {\phi} _ {\mathrm{a}} (\boldsymbol {Q} _ {t} ^ {\prime}, \boldsymbol {F} _ {t} ^ {\prime}) \tag {7c}
$$

$$
\text { decoding } \quad \hat {\boldsymbol {F}} _ {t} ^ {\prime} = \boldsymbol {\phi} _ {\mathrm{d}} (\boldsymbol {S} _ {t} ^ {\prime}) \tag {7d}
$$

Initialization: if $t = T - 1$ , backward transitioner $\pmb { \phi } _ { \mathrm { r } } ^ { \prime }$ transforms the last forward slots ${ \pmb S } _ { T }$ into the first backward queries ${ \pmb Q } _ { T - 1 } ^ { \prime }$ . Transition: if $t < T - 1$ , transitioner $\pmb { \phi } _ { \mathrm { r } } ^ { \prime }$ recurrently transforms previous backward slots $\pmb { S } _ { t + 1 } ^ { \prime }$ into the next backward queries $\pmb { Q } _ { t } ^ { \prime }$ . Aggregation: aggregator $\pmb { \phi } _ { \mathrm { a } }$ iteratively aggregates information in feature $\pmb { F } _ { t }$ into backward queries $\pmb { Q } _ { t }$ , producing backward slots $\pmb { S } _ { t } ^ { \prime }$ , along with byproduct backward attention maps ${ \pmb A } _ { t } ^ { \prime }$ . Decoding: decoder $\phi _ { \mathrm { d } }$ decodes backward slots $\pmb { S } _ { t } ^ { \prime }$ into backward feature reconstruction $\hat { \pmb F } ^ { \prime }$ .

Crucially, in the backward stream $\mathcal { T } _ { \mathrm { b w } }$ , modules aggregator $\pmb { \phi } _ { \mathrm { a } }$ and decoder $\phi _ { \mathrm { d } }$ are rigidly shared from the forward stream $\mathcal { T } _ { \mathrm { f w } }$ , while transitioner $\pmb { \phi } _ { \mathrm { r } } ^ { \prime }$ shares most weights with its forward counterpart $\phi _ { \imath }$ r to ensure the model learns time-consistent physical rules, differing only in relative time embeddings [23].

Instead of forcing $\pmb { S } _ { t }$ and $\pmb { S } _ { t } ^ { \prime }$ to be similar, we enforce that both representations must yield the same reconstruction of the scene. The implicit consensus is achieved by joint minimizing the forward reconstruction loss Equation (2) and the following backward reconstruction loss:

$$
\mathcal {L} _ {\mathrm{ICC}} = \sum_ {t = 1} ^ {T - 1} \| \hat {\boldsymbol {F}} _ {t} ^ {\prime} - \boldsymbol {F} _ {t} \| ^ {2} \tag {8}
$$

We demonstrate that our proposed implicit objective maximizes the mutual consistency of the bi-directional streams in the observation space, bypassing the permutation problem.

![](images/10bde0e85cf76674a7b1d609326e0ce532722aef827553333cf99bc09cb97908.jpg)

<details>
<summary>text_image</summary>

YTWIS Video
ground-truth
Rad5F Q
+10C
</details>

![](images/4d48039c2d929eeb70199d69d980a0dde08e0a6e23e6a2a4a1b1787059b75c0e.jpg)

<details>
<summary>text_image</summary>

MOVI-C video
ground-truth
SmoothSA
+TCC
</details>

Figure 3: Qualitative results of object discovery on videos. Our ICC improves basis methods, RandSF.Q [23] and SmoothSA [25], consistently. Note that the differences of segmentation colors have no semantic meaning.

Intuition 2 (Decomposition Consensus on Reconstruction Manifold) Unlike explicit alignment, our implicit regularization imposes consistency on the observation manifold, allowing slots to diverge in vector space, accommodating Permutation Drift, provided they remain functionally equivalent in scene reconstruction.

Formalism 2 Let us view the video OCL process as a Variational Autoencoder (VAE) framework. We aim to maximize the log-likelihood of video data log $p ( \pmb { F } _ { t } )$ . The forward and backward streams approximate the intractable posterior using two distinct variational distributions, $q _ { 1 } ( \pmb { S } | \pmb { F } )$ and $q _ { 2 } ( { \pmb S } ^ { \prime } | { \pmb F } )$ . The proposed objective, Equations (2) and (8), can be viewed as maximizing the Evidence Lower Bound (ELBO) for both streams simultaneously:

$$
\mathcal {J} = \mathbb {E} _ {\boldsymbol {S} \sim q _ {1}} [ \log p _ {\boldsymbol {\phi}} (\boldsymbol {F} | \boldsymbol {S}) ] + \mathbb {E} _ {\boldsymbol {S} ^ {\prime} \sim q _ {2}} [ \log p _ {\boldsymbol {\phi}} (\boldsymbol {F} | \boldsymbol {S} ^ {\prime}) ] - \mathrm{KL} (\dots) \tag {9}
$$

Let $\mathcal { M } = \{ \hat { \pmb { F } } | \exists \pmb { S } , \hat { \pmb { F } } = \pmb { \phi } _ { d } ( \pmb { S } ) \}$ be the manifold of all possible image reconstructions decodable from the slot space. The implicit loss requires that:

$$
\boldsymbol {\phi} _ {d} (\boldsymbol {S} _ {t}) \approx \boldsymbol {F} _ {t} \quad a n d \quad \boldsymbol {\phi} _ {d} (\boldsymbol {S} _ {t} ^ {\prime}) \approx \boldsymbol {F} _ {t} \tag {10}
$$

Let $\begin{array} { r } { S _ { F } = \pmb { \phi } _ { d } ^ { - 1 } ( \pmb { F } _ { t } ) } \end{array}$ be the set of all valid slot configurations (permutations and decompositions) that can reconstruct frame $\pmb { F } _ { t }$ . By minimizing the reconstruction error for both streams, we actually enforce:

$$
\boldsymbol {S} _ {t} \in \mathcal {S} _ {\boldsymbol {F}} \quad a n d \quad \boldsymbol {S} _ {t} ^ {\prime} \in \mathcal {S} _ {\boldsymbol {F}} \tag {11}
$$

Crucially, since decoder $\phi _ { d }$ is permutation invariant regarding slots [19], the condition $\pmb { S } _ { t } , \pmb { S } _ { t } ^ { \prime } \in S _ { F }$ does not imply $\pmb { S } _ { t } = \pmb { S } _ { t } ^ { \prime } .$ . Instead, it implies consistency in explanatory power. The stochastic transitions ε allow $\pmb { S } _ { t }$ and $\pmb { S } _ { t } ^ { \prime }$ to explore different regions of SF (the solution space), preventing the model from getting stuck in local minima, while the joint reconstruction objective ensures both paths remain valid explanations of the visual scene.

# 4 Experiment

We evaluate Implicit Cycle Consistency (ICC) across a hierarchy of visual understanding: unsupervised object discovery (Section 4.1) and downstream object recognition (Section 4.2). All experiments are conducted with the same set of three random seeds whenever applicable. Note that as our ICC is designed as a hyperparameter-free plugin, which can be integrated into state-of-the-art basis methods, there is no need to do any ablation study.

<table><tr><td rowspan="2"></td><td>ARI</td><td> $ARI_{fg}$ </td><td>mBO</td><td>mIoU</td><td>ARI</td><td> $ARI_{fg}$ </td><td>mBO</td><td>mIoU</td><td>ARI</td><td> $ARI_{fg}$ </td><td>mBO</td><td>mIoU</td></tr><tr><td colspan="4">MOVi-C #slot=11, conditional</td><td colspan="4">MOVi-E #slot=21, conditional</td><td colspan="4">YTVIS-HQ #slot=7</td></tr><tr><td>VideoSAUR</td><td>41.9±1.1</td><td>53.3±2.1</td><td>16.1±0.4</td><td>14.8±0.4</td><td>17.4±2.5</td><td>34.6±20.7</td><td>8.3±4.9</td><td>7.5±4.3</td><td>33.8±0.7</td><td>49.2±0.5</td><td>29.9±0.4</td><td>29.7±0.4</td></tr><tr><td>SlotContrast</td><td>64.6±9.4</td><td>59.9±5.3</td><td>27.7±3.0</td><td>25.8±2.9</td><td>29.9±4.9</td><td>70.6±3.8</td><td>20.7±1.4</td><td>19.3±1.2</td><td>37.2±0.6</td><td>49.4±1.1</td><td>33.0±0.2</td><td>32.8±0.1</td></tr><tr><td>RandSF.Q</td><td>65.4±10.7</td><td>67.4±2.1</td><td>29.2±3.8</td><td>26.8±3.7</td><td>30.5±1.2</td><td>82.1±3.1</td><td>23.0±1.2</td><td>21.6±1.4</td><td>40.1±0.4</td><td>58.0±1.0</td><td>37.6±0.4</td><td>37.2±0.4</td></tr><tr><td>+ ECC</td><td>53.8±3.4</td><td>46.6±3.7</td><td>20.5±1.1</td><td>17.6±1.7</td><td>34.0±4.2</td><td>45.1±2.3</td><td>13.3±5.6</td><td>11.8±6.0</td><td>40.0±1.6</td><td>57.2±3.5</td><td>37.0±0.7</td><td>36.2±1.8</td></tr><tr><td>+ ICC</td><td>73.2±0.7</td><td>67.4±1.0</td><td>32.9±0.4</td><td>30.3±0.3</td><td>41.6±7.5</td><td>80.5±4.1</td><td>26.3±1.3</td><td>24.8±1.1</td><td>40.6±1.0</td><td>60.1±3.6</td><td>39.2±0.3</td><td>38.9±0.4</td></tr><tr><td>SmoothSA</td><td>50.9±1.6</td><td>69.0±0.3</td><td>31.7±0.8</td><td>30.2±0.8</td><td>36.7±0.6</td><td>73.6±0.6</td><td>28.6±0.1</td><td>27.4±0.1</td><td>42.4±0.8</td><td>63.0±3.4</td><td>38.9±0.7</td><td>38.3±0.6</td></tr><tr><td>+ ECC</td><td>44.1±1.6</td><td>70.4±0.1</td><td>30.6±0.8</td><td>29.2±0.8</td><td>35.0±0.8</td><td>65.2±0.5</td><td>23.9±0.3</td><td>22.6±0.2</td><td>40.2±0.3</td><td>59.8±1.2</td><td>38.0±0.3</td><td>37.2±0.3</td></tr><tr><td>+ ICC</td><td>52.1±2.3</td><td>71.2±0.9</td><td>33.4±0.7</td><td>32.1±0.8</td><td>35.4±1.2</td><td>74.0±0.3</td><td>28.7±0.5</td><td>27.5±0.3</td><td>42.1±0.9</td><td>60.2±0.5</td><td>39.6±0.7</td><td>39.2±0.7</td></tr></table>

Table 1: Object discovery on videos. MOVi-C / E – synthetic datasets; YTVIS-HQ – real-world. ARI – mostly background segmentation accuracy; $\mathbf { A R I } _ { \mathrm { f g } }$ – foreground large objects; mBO – best-matched segmentations, normalized by area; mIoU – Hugarian-matched segmentations, normalized by area. ECC and ICC are our explicit and implicit cycle consistency respectively. Input resolution is 224×224; DINO2 ViT-S/14 is employed for encoding; Using random seeds 42, 43 and 44.

# 4.1 Video Object Discovery

Metrics. Video object discovery performance intuitively reflects the quality of slots. We use standard unsupervised object segmentation metrics for the OCL setting: Adjusted Rand Index (ARI) 1, Foreground ARI $( \mathrm { A R I _ { f g } } )$ , Mean Best Overlap (mBO) [17] and Mean Intersection over Union (mIoU) 2. We do not adopt Mean Average Precision (mAP) or similar metrics because in OCL there is no confidence-based precision-recall tradeoff.

Datasets. We evaluate on multiple standard benchmarks. MOVi-C and MOVi-E 3: synthetic videos featuring complex object dynamics (C, E) and camera movements (E). YTVIS 4 the High-Quality version 5: real-world YouTube videos with complex backgrounds, occlusions, motion / encoding blurs and textures.

Baselines. We compare our method against recent representative video OCL methods. VideoSAUR [22]: a classical method. SlotContrast [11]: SOTA in the year 2025. RandSF.Q [23] and SmoothSA [25]: the most recent SOTA methods that surpass SlotContrast further by a large margin. To evaluate our implicit design, we incorporate our Implicit CC design into the two strongest baselines, denoted as RandSF.Q + ICC and SmoothSA + ICC. We also evaluate ECC via RandSF.Q + ECC and SmoothSA + ECC.

<table><tr><td rowspan="2">per epoch; V100</td><td colspan="2">training</td><td colspan="2">evaluation</td></tr><tr><td>GB</td><td>min</td><td>GB</td><td>min</td></tr><tr><td>RandSF.Q @ MOVi-E</td><td>24.2</td><td>8.0</td><td>7.9</td><td>1.2</td></tr><tr><td>+ECC</td><td>24.6</td><td>8.1</td><td>8.0</td><td>1.3</td></tr><tr><td>+ICC</td><td>24.7</td><td>8.4</td><td>8.4</td><td>1.3</td></tr></table>

Table 2: Computation overhead in space and time. Spatial overhead is measured in peak VRAM consumption GB while temporal overhead is measured in time consumption minutes.

<table><tr><td rowspan="2"></td><td>Top-1</td><td>Top-3</td><td>IoU</td><td>#match</td></tr><tr><td colspan="4">YTVIS-HQ #slot=7</td></tr><tr><td>RandSF.Q+MLP</td><td> $90.5_{\pm 0.3}$ </td><td> $97.9_{\pm 0.3}$ </td><td> $50.6_{\pm 0.4}$ </td><td> $8979_{\pm 123}$ </td></tr><tr><td>+ ICC +MLP</td><td> $91.6_{\pm 0.2}$ </td><td> $97.7_{\pm 0.3}$ </td><td> $52.5_{\pm 0.5}$ </td><td> $9233_{\pm 61}$ </td></tr><tr><td>SmoothSA+MLP</td><td> $90.4_{\pm 0.2}$ </td><td> $97.6_{\pm 0.1}$ </td><td> $42.6_{\pm 1.4}$ </td><td> $8957_{\pm 34}$ </td></tr><tr><td>+ ICC +MLP</td><td> $91.5_{\pm 0.0}$ </td><td> $97.9_{\pm 0.2}$ </td><td> $47.0_{\pm 0.6}$ </td><td> $9112_{\pm 48}$ </td></tr></table>

Table 3: Object recognition on videos. Top-1 / Top-3: category classification accuracy; IoU: bounding box regression accuracy; #match: number of matched objects. By training a linear probe (MLP) on frozen slots from Table 1; Using random seeds 42, 43 and 44.

Codebase. Experiments are conducted using codebase object-centric-bench 6 7 , which has reproduced many representative OCL methods with identical advanced data augmentation and training recipes [3], ensuring fair, strong comparisons. It also provides model checkpoints and logs for all three standard random seeds, supporting reproducibility and efficient experimentation.

Results. Table 1 reports the results of all methods across the datasets and metrics. The results show that ICC consistently improves ARI, ARI-FG, and mIoU across all datasets, with particularly strong gains in complex dynamic scenes. Further analysis reveals that ICC yields general performance improvements when integrated with RandSF.Q across all datasets and metrics, while on SmoothSA, slight degradations are observed on a few metrics. Note that these results are upon two SOTA methods that already substantially outperform prior methods, making further improvements inherently difficult. In contrast, the ECC objective on the basis methods always degenerates the performance.

Efficiency. As shown in Table 2, ICC introduces more computation overhead in both space and time than both the basis method and ECC.

# 4.2 Video Object Recognition

Video object recognition performance directly measures the quality of slots. To verify if semantics beyond low-level texture are captured, we evaluate performance using metrics including Top-1 and Top-3 accuracy for category classification, IoU for bounding box regression, and the count of successfully matched objects (#match). We conduct these evaluations on the real-world video dataset YTVIS-HQ. For baselines, we compare the vanilla counterparts of state-of-the-art methods RandSF.Q and SmoothSA against our ICC-integrated versions to demonstrate the improvements in latent space separation. All experiments are implemented within the same codebase as in object discovery, following the standard protocol: freezing the OCL model and training a lightweight 2-layer MLP to predict class labels and bounding boxes from the slot representations.

<table><tr><td>@MOVi-C</td><td>ARI</td><td>ARI $_{fg}$ </td><td>mBO</td><td>mIoU</td></tr><tr><td>RandSF.Q</td><td>65.4±10.7</td><td>67.4±2.1</td><td>29.2±3.8</td><td>26.8±3.7</td></tr><tr><td>+ ECC</td><td>53.8±3.4</td><td>46.6±3.7</td><td>20.5±1.1</td><td>17.6±1.7</td></tr><tr><td>+ Hungarian ECC</td><td>59.4±2.7</td><td>47.2±5.3</td><td>22.4±2.5</td><td>18.3±2.8</td></tr><tr><td>+ Non-Chain Recon.</td><td>68.3±1.5</td><td>65.1±8.8</td><td>29.1±1.4</td><td>27.4±1.5</td></tr><tr><td>+ ICC</td><td>73.2±0.7</td><td>67.4±1.0</td><td>32.9±0.4</td><td>30.3±0.3</td></tr></table>

Table 4: Isolating the source of performance gains. Hungarian ECC isolates permutation drift from decomposition divergence; Non-Chain Reconstruction represents doubled reconstruction without a temporal chain as in ICC.

As shown in Table 3 object recognition results, models trained with ICC outperform their vanilla counterparts in Top-1/Top-3 accuracy and box IoU. This indicates that our consensus objective forces the slots to retain more discriminative identity features rather than just lowlevel texture information, thereby facilitating better separation of object categories.

# 4.3 Ablation Study

Does the performance gain come from the extra backward reconstruction? – No. We design a new experiment item, Non-Chain Reconstruction (NCR), which replaces the chained initialization in Equation (7a) with the default initialization similar to Equation (1a). Namely, dependence between the forward and backward streams are removed; they are just two parallel streams in inverse direction. In this setting, there is still the extra backward reconstruction. But as shown in Table 4, NCR shows no consistent superiority.

Is that ECC performs worse than ICC due to its naive implementation? – No. We design a stronger ECC baseline, Hungarian ECC. We enforce the ECC loss on Hungarian matched slot pairs, rather than on slot pairs that have identical indexes in two sets of slots, as what is conducted in Equation (4) originally. The match metric is cosine similarity. This experiment item can handle the identity switch along time, thus can perform better. As shown in Table 4, although the Hungarian ECC is a bit better but still much worse than ICC.

# 5 Dissection: Mechanism of Implicit Consensus

# 5.1 Quantifying Feature Collapse vs. Diversity

We argue in Section 3.2 that ECC hard alignment forces slots towards a mean-collapsed state. To quantify this, we measure the slot variance, defined as the average variance of slot features across the temporal dimension for a tracked object, and the slot diversity, measuring the cosine distance between slots within a frame.

As shown in Table 5, applying Explicit CC to RandSF.Q results in a sharp drop in slot diversity, i.e., 0.934 → 0.904, confirming that the slots become “averaged” and lose discriminative identity. In contrast, our Implicit CC maintains high diversity 0.300 comparable to the baseline 0.298, without worsening reconstruction error. This proves that ICC aggregates slots through time without sacrificing representation distinctiveness.

<table><tr><td rowspan="2">@YTVIS-HQ ×100</td><td colspan="2">slot</td><td rowspan="2">recon. error↓</td><td rowspan="2"></td></tr><tr><td>diversity↑</td><td>variance↑</td></tr><tr><td>RandSF.Q</td><td>93.4±0.9</td><td>29.8±1.5</td><td>51.5±3.1</td><td></td></tr><tr><td>+ ECC</td><td>90.4±1.2</td><td>19.5±1.3</td><td>67.3±6.4</td><td>collapse</td></tr><tr><td>+ ICC</td><td>91.6±0.9</td><td>30.0±2.4</td><td>53.7±5.9</td><td></td></tr></table>

Table 5: Quantifying representation collapse. ECC drastically reduces slot diversity (feature collapse), whereas ICC maintains spatial diversity and temporal variance without worsening reconstruction error.

![](images/203265bf72855eac2b9432c96b1b1fa5a918307f47ec93e08217e9b8efc8da9c.jpg)

<details>
<summary>scatter</summary>

| Category | x_value | y_value |
| -------- | ------- | ------- |
| RandSF.Q | 0.85    | 0.20    |
| RandSF.Q | 0.90    | 0.18    |
| RandSF.Q | 0.95    | 0.22    |
| RandSF.Q | 1.00    | 0.25    |
| RandSF.Q | 1.05    | 0.23    |
| RandSF.Q | 1.10    | 0.24    |
| RandSF.Q | 1.15    | 0.26    |
| RandSF.Q | 1.20    | 0.27    |
| +ECC     | 0.40    | 0.05    |
| +ECC     | 0.45    | 0.06    |
| +ECC     | 0.50    | 0.07    |
| +ECC     | 0.55    | 0.08    |
| +ECC     | 0.60    | 0.09    |
| +ECC     | 0.65    | 0.10    |
| +ECC     | 0.70    | 0.11    |
| +ECC     | 0.75    | 0.12    |
| +ECC     | 0.80    | 0.13    |
| +ECC     | 0.85    | 0.14    |
| +ECC     | 0.90    | 0.15    |
| +ECC     | 0.95    | 0.16    |
| +ECC     | 1.00    | 0.17    |
| +ECC     | 1.05    | 0.18    |
| +ECC     | 1.10    | 0.19    |
| +ECC     | 1.15    | 0.20    |
| +ICC     | 0.45    | 0.04    |
| +ICC     | 0.50    | 0.05    |
| +ICC     | 0.55    | 0.06    |
| +ICC     | 0.60    | 0.07    |
| +ICC     | 0.65    | 0.08    |
| +ICC     | 0.70    | 0.09    |
| +ICC     | 0.75    | 0.10    |
| +ICC     | 0.80    | 0.11    |
| +ICC     | 0.85    | 0.12    |
| +ICC     | 0.90    | 0.13    |
| +ICC     | 0.95    | 0.14    |
| +ICC     | 1.00    | 0.15    |
| +ICC     | 1.05    | 0.16    |
| +ICC     | 1.10    | 0.17    |
| +ICC     | 1.15    | 0.18    |
| +ICC     | 1.20    | 0.19    |
</details>

Figure 4: Manifold Alignment Analysis. Each dot represents a video frame. ICC achieves high consensus on reconstruction (low Y-axis) despite allowing slots to diverge in the latent space (high X-axis), validating that we align on the reconstruction manifold.

# 5.2 The Manifold Alignment Hypothesis

We argue in Section 3.3 that ICC align streams on the observation manifold, achieving consensus in the observation / reconstruction space while maintaining diversity in the latent / slot space. We analyze the relationship between them using latent distance, measuring slot distance between forward and backward streams, and reconstruction distance, measuring the reconstruction error between forward and backward streams. As there is no backward stream in the baseline, we run the baseline model on videos played inversely; Similarly as ECC has no backward stream reconstruction, we reuse the decoder to decode the backward slots into quasi-backward reconstruction.

Figure 4 plots these two metrics for YTVIS-HQ videos. ECC results cluster in the bottom-left: low reconstruction disagreement but at the cost of latent collapse. Baseline RandSF.Q is scattered: high latent distance, high reconstruction disagreement. ICC forms a unique cluster in the bottom-right: high latent diversity and better reconstruction consensus. This empirically proves that our method successfully decouples latent similarity from semantic consistency, allowing the model to navigate the solution space flexibly.

# 5.3 Visualizing Decomposition Divergence

A key motivation for our method is that the Forward and Backward streams may generate distinct but equally valid segmentations. ECC penalizes this valid ambiguity.

In Figure 1, we visualize the attention masks from the Forward and Backward streams at the timestep t. (middle) ECC forces the masks to be identical. Since the streams disagree on the “correct” decomposition, the model outputs blurry, uncertain masks, failing to capture the object. (right) For ICC, both streams clearly separate the car, land, trees and sky, which is crucial. But specifically for the car, these streams have different decompositions schema. By aligning on the reconstruction rather than slots, ICC allows such semantic flexibility.

# 6 Conclusion

We demonstrate that applying explicit cycle consistency to OCL is ill-posed, as rigid latent alignment conflicts with the stochastic nature of scene decomposition, leading to feature collapse. To handle this, we propose Implicit Cycle Consistency (ICC), which aligns forward-backward streams on the reconstruction manifold. This approach enforces temporal coherence while allowing slot representations to diverge, successfully reconciling object stability with valid decomposition ambiguity. This work sets a good starting point for more techniques from unsupervised MOT to be explored in video OCL.

# References

[1] Favyen Bastani, Songtao He, and Samuel Madden. Self-Supervised Multi-Object Tracking with Cross-Input Consistency. Advances in Neural Information Processing Systems, 34:13695–13706, 2021.   
[2] Ondrej Biza, Sjoerd Van Steenkiste, Mehdi SM Sajjadi, Gamaleldin Fathy Elsayed, Aravindh Mahendran, and Thomas Kipf. Invariant Slot Attention: Object Discovery with Slot-Centric Reference Frames. In International Conference on Machine Learning, pages 2507–2527. PMLR, 2023.   
[3] Gamaleldin Elsayed, Aravindh Mahendran, Sjoerd Van Steenkiste, et al. SAVi++: Towards End-to-End Object-Centric Learning from Real-World Videos. Advances in Neural Information Processing Systems, 35:28940–28954, 2022.   
[4] Iñaki Erregue, Kamal Nasrollahi, and Sergio Escalera. YOLO11-JDE: Fast and Accurate Multi-Object Tracking with Self-Supervised Re-ID. In Proceedings of the Winter Conference on Applications of Computer Vision, pages 824–833, 2025.   
[5] Ke Fan, Zechen Bai, Tianjun Xiao, Tong He, Max Horn, Yanwei Fu, Francesco Locatello, and Zheng Zhang. Adaptive Slot Attention: Object Discovery with Dynamic Slot Number. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23062–23071, 2024.   
[6] Baoxiong Jia, Yu Liu, and Siyuan Huang. Improving Object-centric Learning with Query Optimization. In The Eleventh International Conference on Learning Representations, 2023.   
[7] Ioannis Kakogeorgiou, Spyros Gidaris, Konstantinos Karantzalos, and Nikos Komodakis. Spot: Self-Training with Patch-Order Permutation for Object-Centric Learning with Autoregressive Transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22776–22786, 2024.

[8] Thomas Kipf, Gamaleldin Elsayed, Aravindh Mahendran, et al. Conditional Object-Centric Learning from Video. International Conference on Learning Representations, 2022.   
[9] Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, et al. Object-Centric Learning with Slot Attention. Advances in Neural Information Processing Systems, 33: 11525–11538, 2020.   
[10] Zijia Lu, Bing Shuai, Yanbei Chen, Zhenlin Xu, and Davide Modolo. Self-Supervised Multi-Object Tracking with Path Consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19016–19026, 2024.   
[11] Anna Manasyan, Maximilian Seitzer, Filip Radovic, Georg Martius, and Andrii Zadaianchuk. Temporally consistent object-centric learning by contrasting slots. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5401–5411, 2025.   
[12] Sha Meng, Dian Shao, Jiacheng Guo, and Shan Gao. Tracking Without Label: Unsupervised Multiple Object Tracking via Contrastive Similarity Learning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 16264–16273, 2023.   
[13] Mattia Segu, Luigi Piccinelli, Siyuan Li, Luc Van Gool, Fisher Yu, and Bernt Schiele. Walker: self-Supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs. In European Conference on Computer Vision, pages 1–18. Springer, 2024.   
[14] Maximilian Seitzer, Max Horn, Andrii Zadaianchuk, et al. Bridging the Gap to Real-World Object-Centric Learning. International Conference on Learning Representations, 2023.   
[15] Gautam Singh, Fei Deng, and Sungjin Ahn. Illiterate DALL-E Learns to Compose. International Conference on Learning Representations, 2022.   
[16] Gautam Singh, Yi-Fu Wu, and Sungjin Ahn. Simple Unsupervised Object-Centric Learning for Complex and Naturalistic Videos. Advances in Neural Information Processing Systems, 35:18181–18196, 2022.   
[17] Jasper RR Uijlings, Koen EA Van De Sande, Theo Gevers, and Arnold WM Smeulders. Selective Search for Object Recognition. International Journal of Computer Vision, 104:154–171, 2013.   
[18] Ning Wang, Yibing Song, Chao Ma, Wengang Zhou, Wei Liu, and Houqiang Li. Unsupervised deep tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.   
[19] Thaddäus Wiedemer, Jack Brady, Alexander Panfilov, Attila Juhos, Matthias Bethge, and Wieland Brendel. Provable Compositional Generalization for Object-Centric Learning. In The Twelfth International Conference on Learning Representations, 2024.   
[20] Ziyi Wu, Nikita Dvornik, Klaus Greff, Thomas Kipf, and Animesh Garg. SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models. International Conference on Learning Representations, 2023.

[21] Ziyi Wu, Jingyu Hu, Wuyue Lu, Igor Gilitschenski, and Animesh Garg. SlotDiffusion: Object-Centric Generative Modeling with Diffusion Models. Advances in Neural Information Processing Systems, 36:50932–50958, 2023.   
[22] Andrii Zadaianchuk, Maximilian Seitzer, and Georg Martius. Object-Centric Learning for Real-World Videos by Predicting Temporal Feature Similarities. Advances in Neural Information Processing Systems, 36, 2024.   
[23] Rongzhen Zhao, Jian Li, Juho Kannala, and Joni Pajarinen. Predicting Video Slot Attention Queries from Random Slot-Feature Pairs. arXiv preprint arXiv:2508.22772, 2025.   
[24] Rongzhen Zhao, Vivienne Wang, Juho Kannala, and Joni Pajarinen. Vector-Quantized Vision Foundation Model for Object-Centric Learning. In ACM Multimedia, 2025.   
[25] Rongzhen Zhao, Wenyan Yang, Juho Kannala, and Joni Pajarinen. Smoothing Slot Attention Iterations and Recurrences. arXiv preprint arXiv:2508.05417, 2025.   
[26] Rongzhen Zhao, Yi Zhao, Juho Kannala, and Joni Pajarinen. Slot Attention with Re-Initialization and Self-Distillation. In ACM Multimedia, 2025.   
[27] Zixu Zhao, Jiaze Wang, Max Horn, Yizhuo Ding, Tong He, Zechen Bai, Dominik Zietlow, Carl-Johann Simon-Gabriel, Bing Shuai, Zhuowen Tu, et al. Object-Centric Multiple Object Tracking. In Proceedings of the IEEE/CVF international conference on computer vision, pages 16601–16611, 2023.