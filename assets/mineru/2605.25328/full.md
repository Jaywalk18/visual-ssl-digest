# DIVA: Harnessing the Representation Divergence in Unified Multimodal Models for Mutual Reinforcement

Renjie Lu \* 1 2 Xulong Zhang \* 1 Xiaoyang Qu 1 Shangfei Wang 2 Jianzong Wang † 1

# Abstract

Unified Multimodal models (UMMs) built on a single architecture have shown impressive performance in both understanding and generation. We identify a fundamental challenge that lies in inductive biases induced by distinct supervision signals: generation branch prefers high-fidelity, finegrained representations capable of reconstruction, while the understanding favours semantically discriminative embeddings that remain invariant to task-irrelevant factors. Consequently, optimizing these complementary but non-equivalent objectives within a monolithic backbone leads to mutual impairment instead of enhancement. In this paper, we first analyze the root cause of this interference in unified backbones and reveal a complementary structure in their internal representations. Motivated by the observation, we propose DIVA, a self-improved post-training framework that transforms the representation divergence into interior synergy. By explicitly factorizing the visual representation into shared and unique components based on two complementary information flow, DIVA enables both the understanding and generation branches to achieve beneficial transferring while preserving the integrity of unique information from cross-flow interference via mutual information estimation. Despite its generality, our method consistently achieves improvements across visual understanding (+7.82%) and generation (+8.46%). The official code is available at: https://github.com/Jayyy-H/DIVA.

\*Equal contribution . 1Ping An Technology (Shenzhen) Co., Ltd., Shenzhen, China 2University of Science and Technology of China, Hefei, China. Correspondence to: Jianzong Wang <jzwang@188.com>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

# 1. Introduction

Unified Multimodal Models (UMMs) have recently demonstrated impressive capability in both visual understanding

![](images/5d404aee46ae69e18a45de473ba3345e20a5be7d354f970f931d836322fdd452.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Visual & Text"] --> B["UFFR"]
    B --> C["UFFR"]
    C --> D["UFFR"]
    D --> E["UFFR"]
    E --> F["UFFR"]
    F --> G["UFFR"]
    G --> H["UFFR"]
    H --> I["UFFR"]
    I --> J["UFFR"]
    J --> K["UFFR"]
    K --> L["UFFR"]
    L --> M["UFFR"]
    M --> N["UFFR"]
    N --> O["UFFR"]
    O --> P["UFFR"]
    P --> Q["UFFR"]
    Q --> R["UFFR"]
    R --> S["UFFR"]
    S --> T["UFFR"]
    T --> U["UFFR"]
    U --> V["UFFR"]
    V --> W["UFFR"]
    W --> X["UFFR"]
    X --> Y["UFFR"]
    Y --> Z["UFFR"]
    Z --> AA["UFFR"]
    AA --> AB["UFFR"]
    AB --> AC["UFFR"]
    AC --> AD["UFFR"]
    AD --> AE["UFFR"]
    AE --> AF["UFFR"]
    AF --> AG["UFFR"]
    AG --> AH["UFFR"]
    AH --> AI["UFFR"]
    AI --> AJ["UFFR"]
    AJ --> AK["UFFR"]
    AK --> AL["UFFR"]
    AL --> AM["UFFR"]
    AM --> AN["UFFR"]
    AN --> AO["UFFR"]
    AO --> AP["UFFR"]
    AP --> AQ["UFFR"]
    AQ --> AR["UFFR"]
    AR --> AS["UFFR"]
    AS --> AT["UFFR"]
    AT --> AU["UFFR"]
    AU --> AV["UFFR"]
    AV --> AW["UFFR"]
    AW --> AX["UFFR"]
    AX --> AY["UFFR"]
    AY --> AZ["UFFR"]
    AZ --> BA["UFFR"]
    BA --> BB["UFFR"]
    BB --> BC["UFFR"]
    BC --> BD["UFFR"]
    BD --> BE["UFFR"]
    BE --> BF["UFFR"]
    BF --> BG["UFFR"]
    BG --> BH["UFFR"]
    BH --> BI["UFFR"]
    BI --> BJ["UFFR"]
    BJ --> BK["UFFR"]
    BK --> BL["UFFR"]
    BL --> BM["UFFR"]
    BM --> BN["UFFR"]
    BN --> BO["UFFR"]
    BO --> BP["UFFR"]
    BP --> BQ["AFFR"]
    BQ --> BR["AFFR"]
    BR --> BS["AFFR"]
    BS --> BT["AFFR"]
    BT --> BU["AFFR"]
    BU --> BV["AFFR"]
    BV --> BW["AFFR"]
    BW --> BX["AFFR"]
    BX --> BY["AFFR"]
    BY --> BZ["AFFR"]
    BZ --> CA["AFFR"]
    CA --> CB["AFFR"]
    CB --> CC["AFFR"]
    CC --> CD["AFFR"]
    CD --> CE["AFFR"]
    CE --> CF["AFFR"]
    CF --> CG["AFFR"]
    CG --> CH["AFFR"]
    CH --> CI["AFFR"]
    CI --> CJ["AFFR"]
    CJ --> CK["AFFR"]
```
</details>

Figure 1. Illustration of the gap and base for synergy within UMMs. While the conflict induced by inductive biases from understanding and generation exists, the information flows constructed from same image-text pairs share the semantic anchor, providing the basis for transforming the conflict into mutual reinforcement.

and image generation with a unified architecture (Team, 2024; Pan et al., 2025a; Ge et al., 2024; Wang et al., 2024b; Chen et al., 2025b). While UMMs aim to interleave different tasks within a single backbone and obtain performance improved, existing methods rely on increasingly complex architecture designs, fall short of delivering intrinsic synergy between the capabilities of understanding and generation.

Most existing works (Chen et al., 2025a; Pan et al., 2025b; Chen et al., 2025b) frequently report that optimizing generative objectives negatively degrades the understanding capability. To mitigate this, others (Liao et al., 2025; Qu et al., 2025; Deng et al., 2025) choose to decouple the model component to varying degrees, including separating visual encoders or distinct backbones for different tasks. However, we argue that such separation compromises the fundamental promise of UMMs. As indicated by (Gu et al., 2025), an unified architectures and embedding training is essential for integrating the complementary strengths of different branches, enabling the beneficial transfer between understanding and generation. Therefore, the imperative is to resolve the internal conflict within a fully shared architecture to unlock the potential for mutual reinforcement.

As illustrated in Figure 1, although the representation divergence induced by distinct inductive biases often leads to performance degradation, we argue that these different properties actually offer a unique opportunity for conditional mutual reinforcement. The fundamental basis for this synergy is the shared anchor: when understanding and generation tasks are constructed from the same data sample, they essentially represent the identical underlying physical reality, despite differing in input-output modalities. The inductive biases can be transformed from conflicts into complementary assets - the semantic-invariance information from the understanding branch provides high-level guidance for faithful synthesis, while the structural sensitivity from the generation branch grounds abstract concepts into finegrained details. Specifically, we conduct related experiments in Sec. 2 and the results further validated this analysis.

![](images/681013760b9b52f6a81bdfcafe945d5a7ce8a9eb29b1c55d4f880036ad22d38c.jpg)

<details>
<summary>line</summary>

| Layer Index | Emu3 (NTP) | Show-o (NTP & MTP) |
| ----------- | ---------- | ------------------ |
| 0           | 0.25       | 0.32               |
| 6           | 0.10       | 0.40               |
| 12          | 0.15       | 0.18               |
| 18          | 0.12       | 0.20               |
| 24          | 0.25       | 0.32               |
</details>

(a) Inductive Bias Conflict

![](images/042df56e2379cc572672169bb4f0e0292aaaf14d8facbb42da2177e75821ff6a.jpg)

<details>
<summary>line</summary>

| Layer Index | rate_1=0.2 | rate_2=0.4 | rate_3=0.6 |
| ----------- | ---------- | ---------- | ---------- |
| 4           | 0.25       | 0.30       | 0.35       |
| 6           | 0.55       | 0.60       | 0.65       |
| 12          | 0.75       | 0.78       | 0.80       |
| 18          | 0.70       | 0.72       | 0.75       |
| 24          | 0.65       | 0.68       | 0.70       |
</details>

(b) Geometric Subspace Divergence   
![](images/4376e9309dc0929d24de5eb09b1ccdbf91e3a7885fb96fe6a36e8a4468adc3aa.jpg)

<details>
<summary>line</summary>

| Layer Index | rate_1=0.2 | rate_2=0.4 | rate_3=0.6 |
| ----------- | ---------- | ---------- | ---------- |
| 6           | 50         | 52         | 54         |
| 12          | 70         | 72         | 74         |
| 18          | 55         | 57         | 60         |
| High Shared | 50         | 52         | 54         |
</details>

![](images/6509fbde6970d58568936deaf203e7566f76dc80165d24f55f22dfed5649c9f1.jpg)

<details>
<summary>line</summary>

| Layer Index | Understanding Flow | Generation Flow |
| ----------- | ------------------- | --------------- |
| 0           | 0.9                 | 0.9             |
| 6           | 0.3                 | 0.7             |
| 12          | 0.2                 | 0.7             |
| 18          | 0.2                 | 0.6             |
| 24          | 0.1                 | 0.5             |
</details>

(c)Frequency Analysis   
Figure 2. Visualization of the representation divergence and synergy. (a) shows the severe conflicts occurs in the shallow and deep layers while the mitigation is observed in the middle layers. Meantime, based on the two information flows that are described in Sec. 3.1, the effective rank between different flows increases in the middle layers and decrease again in the deep layers as presented in (b). And we conduct a frequency analysis in (c) to explore the distinct preferences for information extraction and modeling between understanding and generation branches. The discovery of these phenomena forms the basis of DIVA.

In this paper, motivated by these insights, we propose DIVA, a self-improved post-training framework that transforms the conflict between understanding and generation into mutual reinforcement. The core idea is to explicitly factorize the visual representation into shared components that facilitate cross-task transfer, and unique components that preserve task-specific inductive biases. Based on two information flows constructed from understanding and generation branches, we first introduce a collaborative decomposition mechanism. Specifically, we freeze the backbone and training lightweight encoders via factorized logit injection, where the shared encoder learns to transfer semantic skeletons from the counter-flow, and the unique encoder is compelled to capture the remaining flow-specific residuals, constrained by orthogonality. Subsequently, we post-train the UMMs via mutual-information estimation, aligning the shared information while disentangling the unique factors from dual flows across the specific layers. By integrating these constraints with native task supervision, DIVA effectively unlocks the internal synergistic effects within the unified architecture. The main contributions of this paper can be summarized as follows:

• We reveal that the representation divergence induced by inductive biases is not limitation but holds the potential for mutual reinforcement based on same anchor.   
• We propose DIVA as a self-improved framework, that transforms internal conflict into mutual reinforcement by leveraging controllable transfer between shared and unique information.   
• DIVA yields consistent improvements across image understanding, generation and editing, demonstrating its effectiveness and robustness.

# 2. Observation

Point 1: Task-specfic inductive biases between understanding and generation branch. Traditional UMMs are commonly optimized by jointly minimizing an understanding (Und) and generation (Gen) objectives. Formally:

$$
\begin{array}{l} \mathcal {L} _ {\text {Und}} = \mathcal {L} \left(f _ {\theta} \left(\operatorname{concat} \left(t _ {\text {question}}, h _ {v}\right)\right), t _ {\text {answer}}\right) \\ \mathcal {L} _ {\text {Und}} = \mathcal {L} \left(f _ {\theta} \left(\operatorname{concat} \left(t _ {\text {question}}, h _ {v}\right)\right), t _ {\text {answer}}\right) \end{array} \tag {1}
$$

$$
\mathcal {L} _ {\mathrm{Gen}} = \mathcal {L} (f _ {\theta} (\operatorname{concat} (t _ {\text {prompt}}, h _ {v})), I _ {\mathrm{gt}}),
$$

where $f _ { \theta }$ is the shared UMM backbone and $h _ { v }$ is the visual embedding extracted by the visual encoder. The textual variables $t _ { \mathrm { q u e s t i o n } } , t _ { \mathrm { a n s w e r } }$ , and $t _ { \mathrm { p r o m p t } }$ correspond to the question, response, and generation prompt, respectively, and $I _ { \mathrm { g t } }$ denotes the target image. The overall training objective is θ∗ = arg minθ $( \gamma \mathcal { L } _ { \mathrm { U n d } } + \lambda \mathcal { L } _ { \mathrm { G e n } } )$ .

The two objectives impose distinct representational preferences, and prior studies(Niu et al., 2025; Pan et al., 2025b) have observed that strengthening one capability (e.g., visual generation fidelity) may degrade the other (e.g., multimodal understanding accuracy), suggesting a persistent form of negative transfer in shared transformers (Team et al., 2025).

Point 2: Is it possible to transform the conflict into synergy? To investigate the internal interactions, we conducted gradient, geometric, and spectral analyses on shared transformers (Wang et al., 2024b; Xie et al., 2024). Gradient analysis in Figure 2.(a) reveals a inverted parabolic-shaped pattern, that the conflicts are eased in the middle layers while become severe in the shallow and deep layers. To explore the internal interactions, we constructed paired information flows rooted in a common anchor (detailed construction of information flows are in Sec. 3.1). Specifically, for a given image-text pair, we extracted layer-wise hidden states from the understanding and generation branches.

![](images/89fac8e86545e019773f0c296f67f56864fed5e37dd9ab65685be47d592df8a4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Visual Understanding"] --> B["Original Sample Y"]
    C["Image Inpainting"] --> D["Mask"]
    D --> E["A brown-grey tabby cat lies on a weathered blue-green wooden windowsill, its front paws resting on the edge of the window. Its gaze is fixed on the camera, its bright green eyes focused and curious. Its erect ears ..."]
    E --> F["Unified Multimodal Models"]
    F --> G["Middle Layers"]
    G --> H["H_U^(ℓ)"]
    G --> I["K_th layer"]
    G --> J["K_th layer"]
    H --> K["E_U^sha"]
    I --> L["E_G^uni"]
    K --> M["X_1^sha"]
    L --> N["X_2^sha"]
    M --> O["Shared Space"]
    N --> O
    O --> P["Image & Text Detokenizer"]
    P --> Q["Minimize"]
    Q --> R["I (X_1^uni, X_2^uni; Y)"]
    R --> S["Unique Space"]
    S --> T["Shared Space"]
    T --> U["X_1^uni"]
    U --> V["E_U^uni"]
    V --> W["X_2^uni"]
    W --> X["Shared Space"]
    X --> Y["X_1^sha, X_2^sha; Y"]
    Y --> Z["Shared Space"]
    Z --> AA["X_1^sha, X_2^sha; Y"]
    AA --> AB["Shared Space"]
    AB --> AC["X_1^sha, X_2^sha; Y"]
    AC --> AD["Shared Space"]
    AD --> AE["X_1^sha, X_2^sha; Y"]
    AE --> AF["Shared Space"]
    AF --> AG["X_1^sha, X_2^sha; Y"]
    AG --> AH["Shared Space"]
    AH --> AI["X_1^sha, X_2^sha; Y"]
    AI --> AJ["Shared Space"]
    AJ --> AK["X_1^sha, X_2^sha; Y"]
    AK --> AL["Shared Space"]
    AL --> AM["X_1^sha, X_2^sha; Y"]
    AM --> AN["Shared Space"]
    AN --> AO["X_1^sha, X_2^sha; Y"]
    AO --> AP["Shared Space"]
    AP --> AQ["X_1^sha, X_2^sha; Y"]
    AQ --> AR["Shared Space"]
    AR --> AS["X_1^sha, X_2^sha; Y"]
    AS --> AT["Shared Space"]
    AT --> AU["X_1^sha, X_2^sha; Y"]
    AU --> AV["Shared Space"]
    AV --> AW["X_1^sha, X_2^sha; Y"]
    AW --> AX["Shared Space"]
    AX --> AY["X_1^sha, X_2^sha; Y"]
    AY --> AZ["Shared Space"]
    AZ --> BA["X_1^sha, X_2^sha; Y"]
    BA --> BB["Shared Space"]
    BB --> BC["X_1^sha, X_2^sha; Y"]
    BC --> BD["Shared Space"]
    BD --> BE["X_1^sha, X_2^sha; Y"]
    BE --> BF["Shared Space"]
    BF --> BG["X_1^sha, X_2^sha; Y"]
    BG --> BH["Shared Space"]
    BH --> BI["X_1^sha, X_2^sha; Y"]
    BI --> BJ["Shared Space"]
    BJ --> BK["X_1^sha, X_2^sha; Y"]
    BK --> BL["Shared Space"]
    BL --> BM["X_1^sha, X_2^sha; Y"]
    BM --> BN["Shared Space"]
    BN --> BO["X_1^sha, X_2^sha; Y"]
    BO --> BP["Shared Space"]
    BP --> BQ["X_1^sha, X_2^sha; Y"]
    BQ --> BR["Shared Space"]
    BR --> BS["X_1^sha, X_2^sha; Y"]
    BS --> BT["Shared Space"]
    BT --> BU["X_1^sha, X_2^sha; Y"]
    BU --> BV["Shared Space"]
    BV --> BW["X_1^sha, X_2^sha; Y"]
    BW --> BX["Shared Space"]
    BX --> BY["X_1^sha, X_2^sha; Y"]
    BX --> BZ["Shared Space"]
    BZ --> CA["X_1^sha, X_2^sha; Y"]
    CA --> CB["Shared Space"]
    CB --> CC["X_1^sha, X_2^sha; Y"]
    CC --> CD["Shared Space"]
    CD --> CE["X_1^sha, X_2^sha; Y"]
    CE --> CF["Shared Space"]
    CF --> CG["X_1^sha, X_2^sha; Y"]
    CG --> CH["Shared Space"]
    CH --> CI["X_1^sha, X_2^sha; Y"]
    CI --> CJ["Shared Space"]
    CJ --> CK["X_1^sha, X_2^sha; Y"]
    CK --> CL["Shared Space"]
    CL --> CM["X_1^sha, X_2^sha; Y"]
    CM --> CN["Shared Space"]
    CN --> CO["X_1^sha, X_2^sha; Y"]
    CO --> CP["Shared Space"]
    CP --> CQ["X_1^sha, X_2^sha; Y"]
    CQ --> CR["Shared Space"]
    CR --> CS["X_1^sha, X_2^sha; Y"]
    CS --> CT["Shared Space"]
    CT --> CU["X_1^sha, X_2^sha; Y"]
    CU --> CV["Shared Space"]
    CV --> CW["X_1^sha, X_2^sha; Y"]
    CW --> CX["Shared Space"]
    CX --> CY["X_1^sha, X_2^sha; Y"]
    CY --> CZ["Shared Space"]
    CZ --> DA["X_1^sha, X_2^sha; Y"]
    DA --> DB["Shared Space"]
    DB --> DC["X_1^sha, X_2^sha; Y"]
    DC --> DD["Shared Space"]
    DD --> DE["X_1^sha, X_2^sha; Y"]
    DE --> DF["Shared Space"]
    DF --> DG["X_1^sha, X_2^sha; Y"]
    DG --> DH["Shared Space"]
    DH --> DI["X_1^sha, X_2^sha; Y"]
    DI --> DJ["Shared Space"]
    DJ --> DK["X_1^shi, X_2^shi; Y"]
```
</details>

Figure 3. Overview of the self-improved mutual reinforcement (DIVA) pipeline. We propose a post-training paradigm that explicitly align the shared information, while preserve the integrity of unique information between the understanding and generation flows. Both flows are constructed base on the same sample pair to ensure the shared anchor.

Specifically, we employ two geometry-based metrics: Reconstruction Residual and Effective-Rank Increment.The Reconstruction Residual measures the components in the subspace of two information flows that cannot be explained by the information contained in either flow:

$$
\mathcal {R} _ {\mathrm{res}} (G \mid U) \triangleq \frac {\| G - \Pi_ {U} G \| _ {F} ^ {2}}{\| G \| _ {F} ^ {2}} \tag {2}
$$

where ΠU denotes the orthogonal projection operator defined by the PCA basis of information flow. And the Effective-Rank Increment ∆ER can be written as:

$$
\Delta \mathrm{ER} (H _ {G}; X \mid H _ {U}) \triangleq \mathrm{ER} (H _ {U, G}) - \mathrm{ER} (H _ {U}) \tag {3}
$$

As shown in Figure 2.(b), the representations from two flows start with high similarity, significantly diverge into distinct subspaces in the middle layers, and exhibit partial recoupling in the final layers which is attributed to the shared semantic anchor. This indicates that despite shared weights, the model spontaneously learns to separate task-specific information in the intermediate stages. Frequency analysis Figure 2.(c) further explains this divergence: the understanding flow exhibits a low-frequency bias, rapidly discarding noise to capture global semantics, whereas the generation flow maintains high-frequency energy throughout the depth to preserve fine-grained features.

Point 3: Analysis and motivation. The observations reveal a critical duality: the middle layers spontaneously decouple to accommodate conflicting inductive biases, yet the deep layers re-align, confirming the existence of a shared semantic anchor. This insight drives a fundamental shift: rather than enforcing a monolithic compromise, we propose to explicitly factorize the representations into a shared space for the semantic consensus and unique spaces for taskspecific information. By structuring this decomposition, we can transform the internal interference into a mechanism of controllable mutual reinforcement.

# 3. Methodology

In this section, we present DIVA as a bi-directional selfsupervision paradigm for UMMs. Given an image–text pair, first we construct an understanding and generation flows with complementary supervision in Sec. 3.1. Second, we propose a factorized framework that decomposes the dual-view representations into shared and unique components, facilitating bidirectional inductive-bias transfer while mitigating cross-task interference in Sec. 3.2. Finally, we presented DIVA as a two stage post-train method in Sec. 3.3. The overall pipeline is shown in Figure 3.

# 3.1. Preliminary

We first construct two complementary task-induced information flows within the shared transformer backbone $f _ { \theta }$ from the same image-text pair (I, T ).

(1) Understanding Information Flow. Since the ${ \mathcal { L } } _ { \mathrm { U n d } }$ aims to project visual features into a language-aligned semantic space, it encourages representations that emphasize global semantics and structural coherence from the visual information. Therefore, we combine the raw image I with a prompt template $t _ { p r o m p t } \ : ( \mathrm { { e . g . } }$ , "Please describe this image in detail."), and use it as a captioning instruction to elicit a detailed description from the UMM. This captioning supervision induces a low-frequency, global semantic bias in the resulting information flows.

(2) Generation Information Flow. To construct this flow, we leverage a self-supervised inpainting task. We apply a random mask M with a ratio $r \in [ 0 . 2 , 0 . 6 ]$ to the original image I, yielding a corrupted image $I _ { m a s k } = I \odot ( 1 - M )$ . Using the original paired text T as the semantic condition, the model is asked to reconstruct the missing regions. The ${ \mathcal { L } } _ { \mathrm { G e n } }$ objective induces a high-frequency, detail-preserving bias in the generation flow.

Motivated by the observation in section 2, we target the middle layers ${ \mathcal { T } } _ { m i d } = \{ l ~ | ~ l _ { s t a r t } \leq l \leq l _ { e n d } \}$ where taskspecific biases are most distinguished. Formally, for any layer $l \in \mathcal { T } _ { m i d } ,$ we extract the image-token hidden states H img,ℓU , $\bar { H _ { U } ^ { i m g , \ell } } , H _ { G } ^ { i \overleftarrow { m g , \ell } } \in \mathbb { R } ^ { N \times d }$ from the understanding and generation flows, respectively. These complementary representations serve as the basis for our mutual improvement paradigm.

# 3.2. Implicit Synergy Via Mutual-Information

Information Factorization. Given two task-induced information flows $X _ { \mathbf { i } }$ and $X _ { \mathrm { j } }$ from the same sample pairs Y derived from the same physical anchor $Y ~ ( \mathrm { i . e . }$ ., the imagetext pair), we assume that the task-relevant information can be factorized into two types: shared information $\Pi _ { \mathrm { s h } }$ and unique information $\Pi _ { \mathrm { u n i } }$ . The former denotes information that is common across dual flow, while the latter captures information specific to individual flow. Both types of information flow are essential for accurately modeling the unified target Y . This factorization can be formalized as follows:

$$
I \left(X _ {1}, X _ {2}; Y\right) \triangleq \underbrace {\Pi_ {\mathrm{sh}}} _ {\text { Shared   Info }} + \underbrace {\Pi_ {\mathrm{uni}} ^ {i} + \Pi_ {\mathrm{uni}} ^ {j}} _ {\text { Unique   Info }} + \epsilon_ {\text { noise }} \tag {4}
$$

where $\Pi _ { \mathrm { u n i } } ^ { k }$ represent the task-relevant information of two information flow, $\epsilon _ { \mathrm { n o i s e } }$ accounts for irrelevant residuals. This motivates us to align shared factors while preserving unique ones.

As shown in Fig. 3, to compute $\Pi _ { \mathrm { s h } }$ and $\Pi _ { \mathrm { u n i } } ^ { k }$ in Equation $^ { 4 , }$ we factorize the image-token representations extracted from $\mathcal { T } _ { m i d }$ . For each flow $i \in \{ U , G \}$ and layer $\ell \in \mathcal { T } _ { m i d } .$ we first pool image-token hidden states into a layer-wise vector $h _ { i } ^ { ( \ell ) } = \mathrm { P o o l } \left( H _ { i } ^ { i m g , \ell } \right) \in \mathbb { R } ^ { d }$ , forming a set of layerwise features shared encod $\{ h _ { i } ^ { ( l ) } , \dot { h _ { i } ^ { ( l + 1 ) } } , \dot { ~ } \cdot \cdot \cdot \}$ . Then weue encoder duced thewhich is $E _ { s h a } ^ { i }$ $E _ { u n i } ^ { i }$ composed of 3-layer Gated MLPs for each branch, and obtain the shared information $z _ { \mathrm { s h } } ^ { \ell , i }$ and unique information $z _ { \mathrm { u n i } } ^ { \ell , i }$ as follows:

$$
\begin{array}{l} z _ {\mathrm{sh}} ^ {\ell , i} = g _ {\mathrm{sh}} ^ {(i)} (\ell) \odot \phi_ {\mathrm{sh}} \Big (h _ {i} ^ {(\ell)} \Big), g _ {\mathrm{sh}} ^ {(i)} (\ell) = \sigma \Big (W _ {\mathrm{sh}} ^ {i} h _ {i} ^ {(\ell)} \Big), \\ z _ {\mathrm{uni}} ^ {\ell , i} = g _ {\mathrm{uni}} ^ {(i)} (\ell) \odot \phi_ {\mathrm{uni}} \left(h _ {i} ^ {(\ell)}\right), g _ {\mathrm{uni}} ^ {(i)} (\ell) = \sigma \left(W _ {\mathrm{uni}} ^ {i} h _ {i} ^ {(\ell)}\right), \tag {5} \\ \end{array}
$$

where $g _ { ( \cdot ) } ^ { ( i ) } ( \ell )$ is an element-wise soft gate predicted from $h _ { i } ^ { ( \ell ) } , \phi _ { \mathrm { s h } } ( \cdot )$ and $\phi _ { \mathrm { u n i } } ( \cdot )$ are MLPs projections. The training process is presented in Sec. 3.3 which is crucial.

Mutual Enhancement. To effectively enable the bidirectional transfer of complementary information between the understanding and generation flows, while preserve the integrity of their unique components, we introduce a mutualinformation based learning framework. Let $X _ { i } ^ { s } , X _ { j } ^ { s }$ denote the shared features produced by Eq. (5) from the two flows, and $X _ { i } ^ { u } , X _ { j } ^ { u }$ denote the corresponding unique features.

Specifically, we aim to maximize a lower bound on the mutual information between shared representations:

$$
I _ {s h a} (X _ {i} ^ {s}; X _ {j} ^ {s}) = \mathbb {E} _ {\substack {x _ {i}, x _ {j} ^ {+} \sim p (x _ {i}, x _ {j}) \\ x _ {j} ^ {-} \sim p (x _ {j})}} \left[ \log \frac {\exp f (x _ {i} , x _ {j} ^ {+})}{\sum_ {k} \exp f (x _ {i} , x _ {j} ^ {-})} \right], \tag{6}
$$

where $f ( x _ { i } , x _ { j } ^ { + } )$ is the optimal critic, and $\boldsymbol { x } _ { j } ^ { + }$ refers to the shared features of another information flow from the same sample as $x _ { i } .$ , while $x _ { j } ^ { - } )$ denotes the shared features from a different sample.

Maximizing shared information alignment solely is insufficient, as the shared subspace may inadvertently absorb taskspecific factors, or the unique subspace may redundantly encode shared semantics, leading to information leakage. To strictly enforce the disentanglement of $\Pi _ { \mathrm { u n i } }$ between $X _ { \mathbf { i } }$ and $X _ { \mathrm { j } }$ , we propose to minimizes the expected upper bound on the unique features $z _ { \mathrm { u n i } } ^ { \ell , i }$ and $z _ { \mathrm { u n i } } ^ { \ell , j }$ by utilizing the NCE-CLUB (Liang et al., 2023):

$$
\begin{array}{l} I _ {u n i} (X _ {i} ^ {u}; X _ {j} ^ {u}) = \mathbb {E} _ {x _ {i}, x _ {j} ^ {+} \sim p (x _ {i}, x _ {j})} \left[ f ^ {*} (x _ {i}, x _ {j} ^ {+}) \right] \\ - \underset { \begin{array}{c} x _ {j} ^ {-} \sim p (x _ {j}) \end{array} } {\mathbb {E}} \left[ f ^ {*} \left(x _ {i}, x _ {j} ^ {-}\right) \right], \tag {7} \\ \end{array}
$$

where $f ^ { * } ( x _ { i } , x _ { j } ^ { + } )$ is the optimal critic from $I _ { N C E }$ , used within the $I _ { C L U B }$ (Cheng et al., 2020). In practice, we propose an asymmetric alignment design to stabilize optimization and avoid one-sided dominance of information flow; the exact instantiation is illustrated in Eq. (6) and Eq. (7) together encourage transferable information to concentrate in shared factors while confining view-specific biases to unique factors, enabling the implicit bidirectional synergy under a single backbone. In the following section, we will transition from the theoretical analysis presented above to the practical implementation.

Table 1. Comparison on widely used image understanding and generation benchmarks. Scores marked with (\*) are our reproduced results using 8 random seeds. Models incorporating the DIVA are denoted with +DIVA. Detailed scores of GenEval and WISE are provided in Appendix’s Sec.C. 

<table><tr><td>Model</td><td># Params</td><td>Types</td><td>MMMU</td><td>MME</td><td>MMBench</td><td>MMVet</td><td>POPE</td><td>GenEval</td><td>DPG-Bench</td><td>WISE</td></tr><tr><td colspan="11">Understanding Only Models</td></tr><tr><td>LlaVA-v1.5 (Liu et al., 2024a)</td><td>7B</td><td>AR</td><td>35.4</td><td>1488.0</td><td>78.3</td><td>-</td><td>84.1</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Qwen2.5-VL (Bai et al., 2025)</td><td>20B</td><td>AR</td><td>58.6</td><td>-</td><td>83.1</td><td>66.4</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>InstructBLIP (Dai et al., 2023)</td><td>7B</td><td>AR</td><td>-</td><td>1365.9</td><td>-</td><td>53.2</td><td>79.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="11">Generation Only Models</td></tr><tr><td>SDXL (Podell et al., 2023)</td><td>2.6B</td><td>Diff</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.55</td><td>73.75</td><td>0.43</td></tr><tr><td>Qwen-Image (Wu et al., 2025a)</td><td>8B+20B</td><td>AR+Diff</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.86</td><td>88.14</td><td>0.55</td></tr><tr><td>SD3-medium (Esser et al., 2024)</td><td>2B</td><td>Diff</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.74</td><td>83.81</td><td>0.42</td></tr><tr><td>Infinity (Han et al., 2025)</td><td>8B</td><td>VAR</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.79</td><td>86.26</td><td>0.45</td></tr><tr><td colspan="11">Unified Multimodal Models</td></tr><tr><td>Janus-Pro* (Chen et al., 2025b)</td><td>7B</td><td>AR</td><td>40.6</td><td>-</td><td>69.5</td><td>49.9</td><td>86.7</td><td>0.80</td><td>84.22</td><td>0.35</td></tr><tr><td>BLIP3-o* (Chen et al., 2025a)</td><td>7B+1.4B</td><td>AR+Diff</td><td>56.9</td><td>1466.2</td><td>82.5</td><td>66.3</td><td>-</td><td>0.81</td><td>80.56</td><td>0.31</td></tr><tr><td>Bagel* (Deng et al., 2025)</td><td>8B+8B</td><td>AR+Diff</td><td>54.5</td><td>-</td><td>84.8</td><td>67.1</td><td>-</td><td>0.84</td><td>85.04</td><td>0.52</td></tr><tr><td>OmniGen2 (Wu et al., 2025b)</td><td>3B+4B</td><td>AR+Diff</td><td>52.6</td><td>1247.4</td><td>78.1</td><td>-</td><td>82.4</td><td>0.80</td><td>83.59</td><td>0.36</td></tr><tr><td>Emu3 (Wang et al., 2024b)</td><td>8B</td><td>AR</td><td>30.7</td><td>1220.3</td><td>61.4</td><td>37.1</td><td>78.7</td><td>0.64</td><td>79.82</td><td>0.33</td></tr><tr><td>Nexus-Gen (Zhang et al., 2025)</td><td>7B</td><td>AR</td><td>43.5</td><td>1279.1</td><td>70.7</td><td>45.2</td><td>83.6</td><td>0.77</td><td>81.30</td><td>0.39</td></tr><tr><td>+DIVA</td><td>7B</td><td>AR</td><td>49.4 (+5.9)</td><td>1355.3 (+76.2)</td><td>74.9 (+4.2)</td><td>46.6 (+1.4)</td><td>87.4 (+3.8)</td><td>0.83 (+0.06)</td><td>87.87 (+6.57)</td><td>0.45 (+0.06)</td></tr><tr><td>Show-o* (Xie et al., 2024)</td><td>1.5B</td><td>AR</td><td>26.3</td><td>1097.7</td><td>48.7</td><td>32.5</td><td>73.1</td><td>0.57</td><td>69.81</td><td>0.29</td></tr><tr><td>+DIVA</td><td>1.5B</td><td>AR</td><td>32.4 (+6.1)</td><td>1206.1 (+108.4)</td><td>51.0 (+2.3)</td><td>33.8 (+1.3)</td><td>79.1 (+6.0)</td><td>0.64 (+0.07)</td><td>76.03 (+6.22)</td><td>0.34 (+0.05)</td></tr><tr><td>Liquid* (Wu et al., 2026)</td><td>7B</td><td>AR</td><td>30.2</td><td>1321.7</td><td>57.2</td><td>36.9</td><td>77.4</td><td>0.70</td><td>80.63</td><td>0.41</td></tr><tr><td>+DIVA</td><td>7B</td><td>AR</td><td>34.0 (+3.8)</td><td>1434.9 (+113.2)</td><td>58.9 (+1.7)</td><td>37.8 (+0.9)</td><td>84.5 (+7.1)</td><td>0.81 (+0.11)</td><td>83.47 (+2.84)</td><td>0.44 (+0.03)</td></tr></table>

# 3.3. Training Paradigm

In this section, we will transition from the theoretical analysis presented above to the practical implementation of DIVA, a two-stage post-training paradigm. By using native task supervision with cross-task conditioning, we obtain the shared / unique encoders $E _ { i } ^ { s }$ and $E _ { i } ^ { u }$ in stage 1; Then in Stage 2 we freeze the learned encoders and refines the UMM backbone $f _ { \theta }$ via the proposed asymmetric objectives.

Stage 1:Task-Driven Encoder Warmup. We first introduce a Cross-Task Conditioning mechanism to exclusively train the $E _ { s } ^ { i }$ and $E _ { u } ^ { i }$ while freeze the $f _ { \theta }$ . The key idea is to inject factorized representations as logit biases: the shared factors provide transferable signals, while the unique factors are encouraged to correct the remaining task-specific residual.

Let t and v index the text-token and image-token positions used in the corresponding losses, $h _ { \theta } ( \cdot )$ denotes the logit network of UMMs. For the understanding and generation flows target the same sample, we extract the task-supervised logit blocks by slicing the output logits: $s _ { U } : = h _ { \theta } ( . ) [ $ [: , $\mathbf { i } \overline { { \mathbf { l } } } \in \mathbb { R } ^ { V _ { t } \times L }$ and $s _ { G } : = h _ { \theta } ( \cdot ) [ : , \overline { { \mathrm { v } } } ] \in \bar { \mathbb { R } } ^ { V _ { v } \times M }$ , where t and v index the text-token and image-token positions used in the corresponding losses, respectively. Then we obtain the shared factors zℓ,Ush $z _ { \mathrm { s h } } ^ { \ell , \bigtriangledown }$ and $z _ { \mathrm { s h } } ^ { \ell , G }$ zsh via Eq. (5), and inject them as:

$$
\begin{array}{l} \tilde {s} _ {U} = s _ {U} + A _ {U} z _ {\mathrm{sh}} ^ {\ell , G} + B _ {U} z _ {\mathrm{uni}} ^ {\ell , U}, \\ \tilde {s} _ {G} = s _ {G} + A _ {G} z _ {\mathrm{sh}} ^ {\ell , U} + B _ {G} z _ {\mathrm{uni}} ^ {\ell , G}, \tag {8} \\ \end{array}
$$

where $A _ { U } , A _ { G } , B _ { U } , B _ { G }$ are low-rank matrix shared across all layers in $\mathcal { T } _ { m i d }$ and learned together with the encoders.

We train $E _ { u n i }$ and $E _ { s h a }$ by minimizing the native task losses computed on s˜(ℓ)U a $\tilde { s } _ { U } ^ { ( \ell ) }$ nd $\tilde { s } _ { G } ^ { ( \ell ) }$ . Specifically, to prevent the unique encoder $E _ { u n i }$ from redundantly encoding shared factors, we add the orthogonality constraints:

$$
\mathcal {L} _ {\perp} = \sum_ {i \in \{U, G \}} \left\| (\mathbf {z} _ {\mathrm{sh}} ^ {i}) ^ {\top} \mathbf {z} _ {\mathrm{uni}} ^ {i} \right\| _ {F} ^ {2}. \tag {9}
$$

In practice, we adopt a simple schedule that warms up the shared-only conditioning before enabling the uniqueresidual injection. The details about the training process can be seen in Appendix’s Sec. A.

Stage 2: Backbone Fine-Tuning. After obtain the encoders, we unfreeze the backbone $f _ { \theta }$ and refine it using the mutualinformation objectives in Eq. (6) and (7). In practice, directly applying symmetric alignment can be unstable, as the losses of different tasks may differ significantly in scale, leading to one-sided dominance. To avoid this, we adopt an asymmetric alignment with stop-gradient, yielding two directed objectives:

$$
\mathcal {L} _ {U \rightarrow G} = - \log \frac {\exp (\text { sim } (z _ {\text { sh }} ^ {U} , \text { sg } [ z _ {\text { sh }} ^ {G} ]) / \tau)}{\sum_ {j} \exp (\text { sim } (z _ {\text { sh }} ^ {U} , \text { sg } [ z _ {\text { sh }} ^ {G , j} ]) / \tau)},
$$

$$
\mathcal {L} _ {G \to U} = - \log \frac {\exp (\mathrm{sim} (z _ {\mathrm{sh}} ^ {G} , \mathrm{sg} [ z _ {\mathrm{sh}} ^ {U} ]) / \tau)}{\sum_ {j} \exp (\mathrm{sim} (z _ {\mathrm{sh}} ^ {G} , \mathrm{sg} [ z _ {\mathrm{sh}} ^ {U , j} ]) / \tau)}.
$$

The stop-gradient operator sg[·] prevents the target view from being updated within each directed term, improving optimization stability under heterogeneous task scales. Overall, we combine the above losses to optimize the UMMs:

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {U \rightarrow G} + \mathcal {L} _ {G \rightarrow U} + \mathcal {L} _ {\text { uni }} + \mathcal {L} _ {U \text { nd }} + \mathcal {L} _ {\text { Gen }} \tag {11}
$$

where $\mathcal { L } _ { u n i }$ denotes the minimization of upper bound function presented in Eq. 7.

![](images/b6be017262dba1f9547f0b22e4d5f5501ffb3edf9ba3409670e537b8b9d5b3c4.jpg)

Figure 4. Qualitative results on image generation. We use Nexus-Gen as baseline for comparsion. It can be observed that after post-train with DIVA, the model’ s ability of handling the complex attribute, spatial layouts and multiple objectives has significant improved.   
Table 2. Quantitative comparison for the ablation study about the impact of data quality and method effectiveness, training mechanism of DIVA, sensitivity to middle-layer range, architecture of shared/unique encoders, and mask patterns, where Bold values denote the best result within each group. 

<table><tr><td>Configs</td><td>MMMU</td><td>POPE</td><td>GenEval</td><td>DPG-Bench</td></tr><tr><td colspan="5">Data quality and method effectiveness</td></tr><tr><td>Base</td><td>26.3</td><td>73.1</td><td>0.69</td><td>69.81</td></tr><tr><td>Base+SFT</td><td>26.8</td><td>74.5</td><td>0.67</td><td>70.75</td></tr><tr><td>Base+DIVA</td><td>32.4</td><td>79.1</td><td>0.75</td><td>76.03</td></tr><tr><td colspan="5">Mechanism of DIVA</td></tr><tr><td>w/o  $I_{uni}$ </td><td>28.3</td><td>75.8</td><td>0.70</td><td>71.58</td></tr><tr><td>w/o sg[·]</td><td>31.7</td><td>78.2</td><td>0.73</td><td>74.92</td></tr><tr><td colspan="5">Sensitivity to middle-layer range</td></tr><tr><td>Mid-Layer (9–17)</td><td>31.5</td><td>78.4</td><td>0.72</td><td>73.36</td></tr><tr><td>Mid-Layer (8–18)</td><td>32.4</td><td>79.1</td><td>0.75</td><td>76.03</td></tr><tr><td>Mid-Layer (7–17)</td><td>32.2</td><td>78.7</td><td>0.72</td><td>74.70</td></tr><tr><td>Mid-Layer (7–19)</td><td>32.5</td><td>79.0</td><td>0.74</td><td>75.09</td></tr><tr><td colspan="5">Architecture of shared/unique encoders</td></tr><tr><td>Linear+LN</td><td>29.4</td><td>75.9</td><td>0.71</td><td>72.37</td></tr><tr><td>Transformer</td><td>32.1</td><td>79.2</td><td>0.74</td><td>75.65</td></tr><tr><td colspan="5">Mask patterns</td></tr><tr><td>Contiguous</td><td>24.7</td><td>69.6</td><td>0.70</td><td>68.22</td></tr></table>

# 4. Experiments and Results

# 4.1. Experimental Setup

Baselines. The selected baselines include: (1) Shared Architecture (Wang et al., 2024b; Zhang et al., 2025; Xie et al.,

Table 3. Robustness analysis of DIVA on Show-o under different post-training data sources and scales. 

<table><tr><td>Training Data</td><td>Scale</td><td>MMMU</td><td>MME</td><td>GenEval</td><td>DPG-Bench</td></tr><tr><td>JourneyDB</td><td>70K</td><td>31.7</td><td>1175.8</td><td>0.62</td><td>75.71</td></tr><tr><td>Mixed-Dataset</td><td>70K</td><td>31.9</td><td>1193.2</td><td>0.62</td><td>75.83</td></tr><tr><td>Mixed-Dataset</td><td>200K</td><td>32.4</td><td>1206.1</td><td>0.64</td><td>76.03</td></tr></table>

2024; Wu et al., 2026), which unifies the visual encoder and backbone for both understanding and generation tasks; (2) Mixture-of-Transformers (MoT) (Deng et al., 2025), assigning a separate generation-oriented transformer while retaining the original language backbone mainly for understanding; (3) Hybrid Architecture (Wu et al., 2025b; Chen et al., 2025b;a), including hybrid encoding (e.g., CLIP or SigLIP for understanding and VAE for generation) or hybrid modeling (e.g., fused autoregressive (AR) and diffusion).

Implementation Details. We instantiate DIVA on three representative single-backbone UMMs: Nexus-Gen (Zhang et al., 2025), show-o (Xie et al., 2024) and Liquid (Wu et al., 2026). Detailed hyperparameters and optimization settings are summarized in Appendix’ s Sec. B.

Training Data. Due to the limited availability of paired UMM post-training data, we construct a 200K image-text dataset from both understanding-oriented and generationoriented sources. Specifically, it contains: (1) 60K qualityfiltered samples from CapsFusion-120M (Yu et al., 2024) and Infinity-MM (Li et al., 2025), where we preserve the original image-text pairing and refine the captions with

Table 4. Sensitivity analysis of DIVA on Show-o under different weights of the unique-information regularization term. 

<table><tr><td>Config</td><td>POPE</td><td>MMMU</td><td>DPG-Bench</td><td>GenEval</td></tr><tr><td>Base</td><td>73.1</td><td>26.3</td><td>69.81</td><td>0.69</td></tr><tr><td>Base+SFT</td><td>74.5</td><td>26.8</td><td>70.75</td><td>0.67</td></tr><tr><td> $\lambda_{uni} = 0.4$ </td><td>78.3</td><td>31.9</td><td>74.92</td><td>0.74</td></tr><tr><td> $\lambda_{uni} = 0.6$ </td><td>79.1</td><td>32.4</td><td>76.03</td><td>0.75</td></tr><tr><td> $\lambda_{uni} = 0.8$ </td><td>78.7</td><td>32.2</td><td>75.50</td><td>0.75</td></tr></table>

Qwen2.5-VL-32B (Bai et al., 2025); (2) 70K samples from JourneyDB (Sun et al., 2023) with their original text annotations; and (3) 70K samples from MidjourneyV6 (CortexLM, 2024), for which we regenerate image-grounded captions using Qwen2.5-VL-32B. For both information flows, the supervision is constructed from the caption or text prompt associated with the same image-text sample. The understanding flow takes the image with a captioning prompt to elicit semantic descriptions, while the generation flow uses the same associated text as the semantic condition for masked-image reconstruction. This design ensures that the two flows are rooted in the same visual-textual anchor, rather than being optimized with unrelated supervision signals. Further details about the construction of training data are provided in Appendix’s Sec. B.

# 4.2. Benchmark Evaluation

Multimodal Understanding. We assess visual understanding on five standard benchmarks - MMMU (Yue et al., 2024), MMBench (Liu et al., 2024b), MMVP (Tong et al., 2024), MMVet (Yu et al., 2023) and POPE (Li et al., 2023) - to comprehensively assess the model’s capabilities in reasoning, perception, and hallucination robustness. As presented in Table 1, our methods demonstrates consistent improvements over the standard single-backbone baseline across all metrics. Notably, the most significant gains are observed on POPE and MME.

Text-to-Image Generation. Following the evaluation protocol of Janus-Pro (Chen et al., 2025b), we evaluate image generation with Geneval (Ghosh et al., 2023) and DPG-Bench (Hu et al., 2024). As shown in Table 1, applying DIVA to Nexus-Gen, Show-o, and Liquid leads to stable improvements on these compositional generation tasks. We further include WISE (Niu et al., 2025), a benchmark built from 1,000 knowledge-puzzle prompts that probe whether generated images reflect implicit factual knowledge. Our strategy conducted on WISE yields consistent gains on Show-o and Nexus-Gen, while Liquid shows smaller improvements. Though DIVA is not designed to enhance the model’s ability to learn and master world knowledge, the generation branch can learn to better utilize world knowledge attributd to the enhancements in global information consistency and spatial structure perception. Addition results are provided in Appendix’s Sec. A.

![](images/3527f4ad337b0dca4fd015daec5779f95865b7652211ad22390a3121f1d0906d.jpg)

<details>
<summary>scatter</summary>

| Group | X     | Y     |
|-------|-------|-------|
| S0    | 0.35  | 0.48  |
| S1    | 0.15  | 0.35  |
| S2    | 0.80  | 0.85  |
| S3    | 0.65  | 0.38  |
| S4    | 0.45  | 0.82  |
| S5    | 0.55  | 0.90  |
| S6    | 0.80  | 0.60  |
| S7    | 0.10  | 0.12  |
| S8    | 0.25  | 0.58  |
| S9    | 0.40  | 0.20  |
</details>

Figure 5. The t-SNE visualization of our extracted shared factors. Points of different colors indicate different samples. "Und" implies the shared information from understanding flows and "Gen" represents the shared factors from generation.

Image Editing. In addition to Bagel, we conduct experiments on AnyEdit (Yu et al., 2025), UltraEdit (Zhao et al., 2024b) and FLUX.1-Kontext (Labs et al., 2025). As shown in Table 5, DIVA conducted on Nexus-Gen consistently outperforms existing baselines across all tasks. It outperforms Bagel, Anyedit and UltraEdit on ImgEdit (Ye et al., 2025), and also obtain improvement on Edit-Bench-EN (Liu et al., 2025). This demonstrates that the improvement of perceiving global information and spatial structure by DIVA can enhance the model’s image editing capabilities.

# 4.3. More Results

Qualitative Results. Figure 4 demonstrate improvements after conducted on DIVA. The original model often struggles with prompts involving multiple entities, attribute binding, and spatial relations, whereas the DIVA-enhanced model produces images that better follow these constraints. For dense prompts, DIVA more faithfully preserves fine-grained details, reducing the omissions and ambiguous visual bindings observed in the baseline. Additional qualitative results are presented in Appendix’s Sec. A.

Trade-off Analysis. Figure 6 illustrates the validation losses of understanding and generation tasks under varying weights. We set the weight of understanding loss to 1, and change the weight on the generation loss. The baseline suffers from distinct task conflict, where prioritizing generation performance leads to a significant degradation in understanding. In contrast, the model trained with DIVA consistently obtain lower losses across metrics.

Visualization of Factorization. We first randomly select 10 types of test set, each set consists of four sample pairs with similar semantic anchors (distinguishing only from a few attributes). Then we extract the shared factors across all off the understanding flows and generation flows constructed based on these samples. By t-SNE we visualize these factors in Figure 5. The presented results demonstrate our method’ s ability to obtain the shared information between two flows constructed on the same anchor.

Table 5. Image editing results on ImgEdit and GEdit-Bench-EN benchmarks. We conducted DIVA on Nexus-Gen to compare with previous methods. The scores of GPT-4o on both benchmarks are reported in (Deng et al., 2025). 

<table><tr><td rowspan="2">Method</td><td rowspan="2"># Params</td><td colspan="10">ImgEdit</td><td colspan="3">GEdit-Bench-EN</td></tr><tr><td>Rep.</td><td>Style</td><td>Act.</td><td>Ext.</td><td>Rem.</td><td>Bg.</td><td>Add</td><td>Comp.</td><td>Adj.</td><td>Ovr.</td><td>SC</td><td>PQ</td><td>Overall</td></tr><tr><td>GPT-4o</td><td>-</td><td>4.35</td><td>4.93</td><td>4.89</td><td>2.90</td><td>3.66</td><td>4.57</td><td>4.61</td><td>3.96</td><td>4.33</td><td>4.20</td><td>7.85</td><td>7.62</td><td>7.53</td></tr><tr><td>AnyEdit</td><td>4B</td><td>2.41</td><td>2.91</td><td>2.67</td><td>1.88</td><td>2.26</td><td>2.27</td><td>3.22</td><td>1.63</td><td>2.94</td><td>2.67</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UltraEdit</td><td>4B</td><td>2.86</td><td>3.81</td><td>2.98</td><td>2.16</td><td>1.43</td><td>2.84</td><td>3.48</td><td>1.93</td><td>2.81</td><td>2.99</td><td>-</td><td>-</td><td>-</td></tr><tr><td>FLUX.1-kontext</td><td>12B</td><td>4.12</td><td>4.55</td><td>4.10</td><td>1.79</td><td>2.91</td><td>3.72</td><td>3.69</td><td>2.91</td><td>3.55</td><td>3.48</td><td>6.67</td><td>7.03</td><td>6.01</td></tr><tr><td>BAGEL</td><td>8B+8B</td><td>3.78</td><td>4.46</td><td>4.13</td><td>1.49</td><td>3.01</td><td>3.35</td><td>3.62</td><td>2.50</td><td>3.56</td><td>3.24</td><td>7.54</td><td>6.42</td><td>6.64</td></tr><tr><td>Nexus-Gen</td><td>7B</td><td>3.03</td><td>3.52</td><td>2.85</td><td>2.23</td><td>1.50</td><td>3.08</td><td>3.41</td><td>1.96</td><td>2.42</td><td>2.98</td><td>5.32</td><td>4.55</td><td>4.61</td></tr><tr><td>+DIVA</td><td>7B</td><td>3.67 (+0.64)</td><td>3.93 (+0.41)</td><td>3.21 (+0.36)</td><td>2.72 (+0.49)</td><td>1.79 (+0.29)</td><td>3.25 (+0.17)</td><td>3.73 (+0.32)</td><td>2.25 (+0.29)</td><td>2.75 (+0.33)</td><td>3.35 (+0.37)</td><td>5.63 (+0.31)</td><td>4.73 (+0.18)</td><td>4.92 (+0.31)</td></tr></table>

Empirical Study. We further compare DIVA with recent post-training methods for UMMs, including RecA (Xie et al., 2025) and UAE (Yan et al., 2025), on both understanding and generation benchmarks. As shown in Table 6, DIVA achieves stronger improvements than RecA under the same SFT-type post-training setting. Specifically, DIVA improves MME by +108.4 and POPE by +6.0, which are substantially larger than the gains obtained by RecA. On the generation side, DIVA also achieves a higher GenEval score, indicating that the proposed factorized mutual-reinforcement objective does not merely enhance visual understanding, but also benefits text-to-image generation. Compared with reconstructionoriented alignment, DIVA explicitly models the shared and unique information between understanding and generation flows, which enables more balanced improvements across tasks. For UAE, comparable results under the same evaluation setting are not publicly available, so we leave the corresponding entries blank to avoid unfair comparison.

# 4.4. Ablation Study

Considering the computational overhead required for training, we selected Show-o to perform the ablation experiments. The modest scale of this model facilitates a more agile training process, thereby making it feasible to extensively verify the contribution of each module in our proposed method and the results is presented in Table 2.

Data Quality and Method Effectiveness. As shown in Table 2, fine-tuning the baseline with our data using standard Supervised Fine-tuning (SFT) brings only marginal changes, suggesting that the post-training dataset itself does not introduce substantial performance gains. In contrast, adding DIVA yields consistent and consistent improvements on both understanding and generation metrics. This gap between Base+SFT and Base+DIVA indicates that the observed gains are largely attributed to our training strategy rather than data quality or additional fine-tuning alone.

Table 6. Comparison with other post-training methods for UMMs. Entries marked with “-” indicate that comparable results under the same evaluation setting are not publicly available, and therefore cannot be fairly reproduced in our setup. 

<table><tr><td>Method</td><td>Types</td><td>MME</td><td>POPE</td><td>GenEval</td></tr><tr><td>RecA</td><td>SFT</td><td>1134.8 (+37.1)</td><td>75.7 (+2.6)</td><td>0.63 (+0.06)</td></tr><tr><td>UAE</td><td>SFT+RL</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DIVA</td><td>SFT</td><td>1206.1 (+108.4)</td><td>79.1 (+6.0)</td><td>0.64 (+0.07)</td></tr></table>

Mechanism of DIVA. To evaluate the importance of key components in our post-training paradigm DIVA, we perform ablations on (i) the unique-information regularization term $I _ { \mathrm { u n i } }$ and (ii) the stop-gradient design used in the shared MI alignment. The results in Table 2 prove the necessity of them. Removing $I _ { \mathrm { u n i } } \left( \mathrm { w } / \mathrm { o } I _ { u n i } \right)$ consistently harms both understanding and generation tasks, indicating that explicitly suppressing cross-flow leakage of unique factors is indispensable for achieving genuine mutual gains. Without this constraint, the optimization is prone to entangle task-specific information and allow shortcut correlations to seep into the shared subspace, which in turn weakens cross-task transfer and leads to broader degradation rather than a single-sided drop. Besides, ablating stop-gradient (w/o sg[·]) also yields a noticeable but milder decline, suggesting that its primary role is to stabilize the bi-directional alignment and mitigate gradient interference between the understanding and generation objectives. Together, these ablations support our design rationale: $I _ { u n i }$ is the key mechanism that enforces a clean shared/unique decomposition to prevent negative transfer, while stop-gradient acts as an important stabilizer that makes mutual-information based sculpting reliably trainable in a unified backbone.

Sensitivity to Middle-layer Range.Since DIVA applies the shared/unique factorization and mutual-information objectives on the middle layers where task-specific divergence is most pronounced, we further study its sensitivity to the selected layer range. As shown in Table 2, DIVA performs consistently across a reasonable middle-layer region. The default range of 8–18 achieves the best overall performance, while nearby choices such as 7–17 and 7–19 remain competitive. Meanwhile, indiscriminately enlarging the range does not bring further gains and also increases the training cost. This verifies that our layer selection is guided by the diagnostic observations in Sec. 2, rather than being a fragile hyperparameter tuned for a single setting.

Architecture of Shared/Unique Encoders. The factorization encoders $E _ { i } ^ { s }$ and $E _ { i } ^ { u }$ play an essential role in earlystage feature mapping. To assess the impact of this design choice, we replace our default Gated-MLP encoders with a standard Linear+LayerNorm mapping and a more heavy Transformer encoder. The results show a clear capacity–stability trade-off. when adopt Linear+LayerNorm as projector shows a notable performance degradation, suggesting that a purely affine mapping with normalization is not expressive enough to capture the non-linear factorization required by the shared/unique information decomposition. In contrast, the Transformer variant performs close to the original solution on most metrics and even achieves a marginal improvements in the POPE benchmark. However,given its substantially higher complexity and optimization burden, this phenomenon indicates that the bottleneck is not simply encoder capacity; Rather, the Gated-MLP already provides sufficient non-linearity to realize effective factorization, while remaining lightweight and stable for posttraining. These findings support our architectural choice: a Gated-MLP strikes the right balance between representational power and trainability, making it a practical and effective instantiation of $E _ { i } ^ { s }$ and $E _ { i } ^ { u }$ for DIVA.

![](images/d10557b7b8e9f64da6ac581d7e1fe2d9f4c1251622df89120e7ea8d59a076fc2.jpg)

<details>
<summary>line</summary>

| Generation Capability (Validation loss) | Ours (Baseline + DIVA) | Baseline |
| ---------------------------------------- | ---------------------- | -------- |
| 4.2                                      | 0.80                   | 0.90     |
| 4.6                                      | 0.78                   | 0.88     |
| 4.8                                      | 0.70                   | 0.85     |
| 5.0                                      | 0.64                   | 0.75     |
| 5.2                                      | 0.53                   | 0.65     |
| 5.4                                      | -                      | 0.56     |
</details>

Figure 6. Visualization of breakthrough of capability between understanding and generation branches under unified training.

Mask Patterns. The results in Table 2 proves that using dispersed random masks to construct the information flow of generation branch is important for mutually enhance. We replace the default random local masking with a same-ratio contiguous block mask and find that performance drops consistently across both understanding and generation tasks. This indicates that block masking weakens the quality and diversity of supervisory signals provided by the generation branch: masking a single continuous region encourages the model to rely more on coarse spatial continuity and local texture propagation, rather than integrating globally distributed semantic and structural features.

# 4.5. Sensitivity and Robustness Analysis

Robustness to post-training data.To further disentangle the effect of DIVA from the specific data mixture or captioning pipeline, we additionally evaluate DIVA under different post-training data sources and scales in Table 3. As shown in Table 3, using raw JourneyDB-70K already yields competitive performance, while replacing it with a mixed 70K subset leads to only marginal changes. Increasing the mixed dataset to 200K brings further gains, but the improvement is moderate rather than abrupt. These results are consistent with the Base+SFT versus Base+DIVA comparison in Table 2, suggesting that the observed gains are not mainly attributed to a particular data source or additional fine-tuning alone, but to the proposed factorized mutual-reinforcement training strategy.

Sensitivity to unique-information regularization. Although Fig. 6 already shows that DIVA consistently alleviates the conflict frontier between understanding and generation losses, we further conduct an explicit sensitivity study on the weight of the unique-information regularization term. In our main experiments, we keep the original task loss weights of the corresponding base models unchanged, and only use $\lambda _ { \mathrm { u n i } }$ to control the strength of the NCE-CLUB based unique-information regularization. As shown in Table 4, DIVA consistently outperforms both the base model and the SFT baseline across different values of $\lambda _ { \mathrm { u n i } }$ . The best overall performance is obtained at $\lambda _ { \mathrm { u n i } } = 0 . 6$ , while nearby settings still maintain clear gains on both understanding and generation benchmarks. These results indicate that DIVA does not rely on a narrowly tuned loss weight to converge or achieve improvements, supporting the robustness of the proposed factorized mutual-reinforcement objective.

# 5. Conclusion and Limitation

DIVA is a self-improved post-training framework designed for achieving synergy in UMMs. It consistently achieves better performance across image understanding, generation, and editing tasks, highlighting the great potential of optimizing UMMs through their internal complementary structures.

Limitation. Our current evaluation primarily focuses on models in the 1.5B to 8B parameter range. While we observe consistent gains, validating the scalability of DIVA on larger-scale models remains an important direction to confirm whether our method follows scaling laws. Besides, extending DIVA to broader multimodal settings, such as video and interleaved generation, is worth future exploration.

# Acknowledgements

This work was supported by Shenzhen-Hong Kong Joint Funding Project (Category A) under grant No. SGDX20240115103359001.

# Impact Statement

This work aims to improve unified multimodal models by enabling visual understanding and generation to reinforce each other within a shared backbone. Such models may benefit applications in multimodal assistants, creative content generation, and visual reasoning systems. At the same time, stronger image generation and editing capabilities may also amplify risks such as synthetic-content misuse, biased generation, or visually plausible but incorrect outputs. We therefore encourage responsible deployment with provenance tracking, safety filtering, and careful evaluation under real-world usage scenarios.

# References

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.   
Chen, J., Xu, Z., Pan, X., Hu, Y., Qin, C., Goldstein, T., Huang, L., Zhou, T., Xie, S., Savarese, S., et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025a.   
Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., and Ruan, C. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025b.   
Cheng, P., Hao, W., Dai, S., Liu, J., Gan, Z., and Carin, L. Club: A contrastive log-ratio upper bound of mutual information. In International conference on machine learning, pp. 1779–1788. PMLR, 2020.   
CortexLM. Cortexlm/midjourney-v6. https: //huggingface.co/datasets/CortexLM/ midjourney-v6, 2024.   
Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P. N., and Hoi, S. Instructblip: Towards generalpurpose vision-language models with instruction tuning. Advances in neural information processing systems, 36: 49250–49267, 2023.   
Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Ge, Y., Zhao, S., Zhu, J., Ge, Y., Yi, K., Song, L., Li, C., Ding, X., and Shan, Y. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

Grill, J.-B., Strub, F., Altché, F., Tallec, C., Richemond, P., Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.

Gu, T., Yang, K., Feng, Z., Wang, X., Zhang, Y., Long, D., Chen, Y., Cai, W., and Deng, J. Breaking the modality barrier: Universal embedding learning with multimodal llms. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 2860–2869, 2025.

Han, J., Liu, J., Jiang, Y., Yan, B., Zhang, Y., Yuan, Z., Peng, B., and Liu, X. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 15733–15744, 2025.

Hu, X., Wang, R., Fang, Y., Fu, B., Cheng, P., and Yu, G. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Jin, W., Niu, Y., Liao, J., Duan, C., Li, A., Gao, S., and Liu, X. Srum: Fine-grained self-rewarding for unified multimodal models. arXiv preprint arXiv:2510.12784, 2025.

Labs, B. F., Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., and Li, C. Llavaonevision: Easy visual task transfer. Trans. Mach. Learn. Res., 2025, 2025.

Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, X., and Wen, J.-R. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pp. 292–305, 2023.   
Liang, P. P., Deng, Z., Ma, M. Q., Zou, J. Y., Morency, L.-P., and Salakhutdinov, R. Factorized contrastive learning: Going beyond multi-view redundancy. Advances in Neural Information Processing Systems, 36:32971–32998, 2023.   
Liao, C., Liu, L., Wang, X., Luo, Z., Zhang, X., Zhao, W., Wu, J., Li, L., Tian, Z., and Huang, W. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.   
Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.   
Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 26296–26306, 2024a.   
Liu, S., Han, Y., Xing, P., Yin, F., Wang, R., Cheng, W., Liao, J., Wang, Y., Fu, H., Han, C., et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.   
Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pp. 216–233. Springer, 2024b.   
Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
Niu, Y., Ning, M., Zheng, M., Jin, W., Lin, B., Jin, P., Liao, J., Feng, C., Ning, K., Zhu, B., et al. Wise: A world knowledge-informed semantic evaluation for textto-image generation. arXiv preprint arXiv:2503.07265, 2025.   
Pan, K., Lin, W., Yue, Z., Ao, T., Jia, L., Zhao, W., Li, J., Tang, S., and Zhang, H. Generative multimodal pretraining with discrete diffusion timestep tokens. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26136–26146, 2025a.   
Pan, X., Shukla, S. N., Singh, A., Zhao, Z., Mishra, S. K., Wang, J., Xu, Z., Chen, J., Li, K., Juefei-Xu, F., et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025b.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., and Rombach, R. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.   
Qu, L., Zhang, H., Liu, Y., Wang, X., Jiang, Y., Gao, Y., Ye, H., Du, D. K., Yuan, Z., and Wu, X. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2545–2555, 2025.   
Sun, K., Pan, J., Ge, Y., Li, H., Duan, H., Wu, X., Zhang, R., Zhou, A., Qin, Z., Wang, Y., et al. Journeydb: A benchmark for generative image understanding. Advances in neural information processing systems, 36:49659–49678, 2023.   
Team, C. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.   
Team, N., Han, C., Li, G., Wu, J., Sun, Q., Cai, Y., Peng, Y., Ge, Z., Zhou, D., Tang, H., et al. Nextstep-1: Toward autoregressive image generation with continuous tokens at scale. arXiv preprint arXiv:2508.10711, 2025.   
Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., and Xie, S. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9568–9578, 2024.   
Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.   
Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024b.   
Wang, Y., Yang, S., Zhao, B., Zhang, L., Liu, Q., Zhou, Y., and Xie, C. Gpt-image-edit-1.5 m: A millionscale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033, 2025.   
Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.-m., Bai, S., Xu, X., Chen, Y., et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.   
Wu, C., Zheng, P., Yan, R., Xiao, S., Luo, X., Wang, Y., Li, W., Jiang, X., Liu, Y., Zhou, J., et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025b.   
Wu, J., Jiang, Y., Ma, C., Liu, Y., Zhao, H., Yuan, Z., Bai, S., and Bai, X. Liquid: Language models are scalable and unified multi-modal generators. International Journal of Computer Vision, 134(1):39, 2026.

Xie, J., Mao, W., Bai, Z., Zhang, D. J., Wang, W., Lin, K. Q., Gu, Y., Chen, Z., Yang, Z., and Shou, M. Z. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.   
Xie, J., Darrell, T., Zettlemoyer, L., and Wang, X. Reconstruction alignment improves unified multimodal models. arXiv preprint arXiv:2509.07295, 2025.   
Yan, Z., Lin, K., Li, Z., Ye, J., Han, H., Wang, Z., Liu, H., Lin, B., Li, H., Xu, X., et al. Can understanding and generation truly benefit together–or just coexist? arXiv preprint arXiv:2509.09666, 2025.   
Ye, Y., He, X., Li, Z., Lin, B., Yuan, S., Yan, Z., Hou, B., and Yuan, L. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.   
Yu, Q., Sun, Q., Zhang, X., Cui, Y., Zhang, F., Cao, Y., Wang, X., and Liu, J. Capsfusion: Rethinking image-text data at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14022– 14032, 2024.   
Yu, Q., Chow, W., Yue, Z., Pan, K., Wu, Y., Wan, X., Li, J., Tang, S., Zhang, H., and Zhuang, Y. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26125–26135, 2025.   
Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., and Wang, L. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.   
Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.   
Zhang, H., Duan, Z., Wang, X., Zhao, Y., Lu, W., Di, Z., Xu, Y., Chen, Y., and Zhang, Y. Nexus-gen: A unified model for image understanding, generation, and editing. arXiv e-prints, pp. arXiv–2504, 2025.   
Zhao, C., Song, Y., Wang, W., Feng, H., Ding, E., Sun, Y., Xiao, X., and Wang, J. Monoformer: One transformer for both diffusion and autoregression. arXiv preprint arXiv:2409.16280, 2024a.   
Zhao, H., Ma, X. S., Chen, L., Si, S., Wu, R., An, K., Yu, P., Zhang, M., Li, Q., and Chang, B. Ultraedit: Instructionbased fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024b.

Zhou, C., Yu, L., Babu, A., Tirumala, K., Yasunaga, M., Shamis, L., Kahn, J., Ma, X., Zettlemoyer, L., and Levy, O. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.

# A. Related work.

# A.1. Unified Multimodal Models (UMMs)

Vision-Language Models (VLMs) have demonstrated remarkable progress in multimodal understanding and reasoning, enabled by combining Large Language Models (LLMs) with powerful visual encoders (Liu et al., 2023; Team, 2024; Wang et al., 2024a). Motivated by this success, recent research has sought to extend VLMs with image generation capabilities, resulting in the development of Unified Multimodal Models (UMMs) (Pan et al., 2025a; Ge et al., 2024; Wang et al., 2024b; Chen et al., 2025b). UMMs aim to combine multimodal understanding and generation within a single backbone, enabling the capacity between understanding and generation interleaved and resulting the improvement of performance across various tasks.

Recent studies can be categorized into the following two types. To combine the inference capabilities of LLMs with the high generative quality of diffusion models, some researches (Ge et al., 2024; Zhou et al., 2024; Zhao et al., 2024a) employ a hybrid strategy of using AR for understanding and diffusion for generation. However, these methods typically introduce additional semantic encoders or complex two-stage designs, sacrificing the uniformity of the architecture and the reciprocal potential of parameter sharing. Furthermore, when performing auto-regressive predictions in continuous embedding spaces, the problem of error accumulation is often encountered, leading to a decrease in generation quality with sequence length.

Other works (Xie et al., 2024; Wang et al., 2024b; Team, 2024; Wu et al., 2026) select discretize visual data into a sequence of tokens, and then jointly model it with text in the same Transformer. Despite its simple architecture, experiments (Zhang et al., 2025; Deng et al., 2025) show that a fully shared Transformer can cause severe gradient conflicts at shallow and deep layers when processing text and images, due to the huge differences in their underlying statistical properties (such as entropy), hindering the effective convergence of the model. Despite the performance are boosted by increasingly complex system designs, the gap between understanding and generation branches within UMMs are the fundamental challenge.

# A.2. Post-Training strategy for UMMs

Supervised fine-tuning with high-quality data (Chen et al., 2025a; Wang et al., 2025) used to be a common and direct practice by utilizing advanced closed-source models (e.g., GPT-4o) to generate large-scale, high-quality image-text pairs. However, this method is limited to its high cost and the risk of distribution shift about generated data. Recently some works choose to explore different techniques to enhance the generation branch of UMMs, given that the understanding branch performs better. For instance, RecA (Xie et al., 2025) leverages reconstruction alignment by conditioning generation on understanding embeddings and using reconstruction losses to bring representations closer. In addition, SRUM (Jin et al., 2025) proposes a fine-grained self-reward framework. Its core lies in utilizing understanding branch as an internal evaluator. By constructing a dual reward system encompassing both global and local dimensions, it guides and optimizes the performance of the generating branch without requiring additional manually labeled data. Compared to these methods, UAE (Yan et al., 2025) introduces a training paradigm based on the auto-encoder perspective: treating the understanding task as an encoder (image to text) and the generation task as a decoder (text to image). By maximizing the fidelity of image reconstruction, it forces the establishment of a bidirectional information flow between understanding and generation, thereby achieving mutual promotion. However, it forcibly uses discrete text as an intermediate information bottleneck, which inevitably leads to the loss of a large amount of pixel-level details. This makes it difficult to perfectly reconstruct the original image by relying solely on text descriptions, thus limiting potential of the model’s capabilities.

# B. Implementation Details.

# B.1. Architecture

In Stage 1, we inject factorized representations into task-supervised logit blocks via lightweight readouts. To keep conditioning parameter-efficient, we parameterize each readout matrix as a rank-r factorization:

$$
A = P Q ^ {\top}, \tag {12}
$$

where $P \in \mathbb { R } ^ { V \times r }$ and $Q \in \mathbb { R } ^ { d \times r }$ , with $V$ denoting the target logit dimension (text vocabulary size $V _ { t }$ or visual-token vocabulary size $V _ { v } )$ and d the factor dimension $( d _ { \mathrm { s h } } \ o r d _ { \mathrm { u n i } } )$ . We use four readouts in total: $A _ { U } , A _ { G }$ for cross-flow shared injection and $B _ { U } , B _ { G }$ for self-flow unique injection,

$$
A _ {U} = P _ {U} Q _ {U} ^ {\top}, A _ {G} = P _ {G} Q _ {G} ^ {\top}, B _ {U} = R _ {U} S _ {U} ^ {\top}, B _ {G} = R _ {G} S _ {G} ^ {\top},
$$

and share the same readout parameters across all $\ell \in \mathcal { T } _ { m i d }$ to avoid layer-specific adapters. We set the low-rank dimension to $r = 2 4$ in all experiments.

Gated-MLP factorization encoders. For each flow $i \in \{ U , G \}$ and each selected middle layer $\ell \in \mathcal { T } _ { m i d } .$ , we first pool the image-token hidden states $H _ { i } ^ { i m g , \ell } \in \mathbb { R } ^ { N \times d }$ into a single vector $h _ { i } ^ { ( \ell ) } \in \mathbb { R } ^ { d }$ . Both the shared encoder and unique encoder adopt a gated-MLP form:

$$
z = \mathrm{LN} (g \odot \phi (h)), \quad g = \sigma (W h), \tag {13}
$$

where $\phi ( \cdot )$ is a 3-layer MLPs with a nonlinearity (GELU in our implementation), W is a linear projection producing an element-wise sigmoid gate $g \in ( 0 , 1 ) ^ { d ^ { \prime } }$ , ⊙ denotes element-wise product, and $\mathrm { L N } ( \cdot )$ stabilizes the factor scale.

Table 7. Hyperparameters and Settings in the stage 1 of post-training. 

<table><tr><td></td><td>Nexus-Gen</td><td>Show-o</td><td>Liquid</td></tr><tr><td colspan="4">Optimization</td></tr><tr><td>Optimizer</td><td>AdamW + EMA</td><td>AdamW + EMA</td><td>AdamW + EMA</td></tr><tr><td>Learning rate</td><td>2e-4</td><td>2e-4</td><td>1.5e-4</td></tr><tr><td>LR scheduler</td><td>Cosine</td><td>Cosine</td><td>Cosine</td></tr><tr><td>EMA decay</td><td>(0.99,0.999)</td><td>(0.99,0.999)</td><td>(0.99,0.999)</td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>Warmup steps</td><td>300</td><td>200</td><td>500</td></tr><tr><td>Training steps</td><td>3K</td><td>2K</td><td>5K</td></tr><tr><td>Grad. accumulation</td><td>5</td><td>5</td><td>8</td></tr><tr><td>Per-GPU batch size</td><td>6</td><td>6</td><td>6</td></tr><tr><td rowspan="2">Trainable modules</td><td>Shared &amp; unique encoders</td><td>Shared &amp; unique encoders</td><td>Shared &amp; unique encoders</td></tr><tr><td>Low-rank readouts</td><td>Low-rank readouts</td><td>Low-rank readouts</td></tr><tr><td>Frozen backbone  $f_{\theta}$ </td><td>√</td><td>√</td><td>√</td></tr><tr><td colspan="4">Loss weights / schedule</td></tr><tr><td> $\lambda_{und}$ </td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td> $\lambda_{gen}$ </td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td> $\lambda_{\perp}$ (Equation 9)</td><td>0.2</td><td>0.2</td><td>0.2</td></tr><tr><td>Schedule</td><td>shared-only → shared+unique</td><td>shared-only → shared+unique</td><td>shared-only → shared+unique</td></tr></table>

# B.2. Training details

Formally DIVA is a two-stage post-training paradigm:

(1) In stage 1 we introduce a cross-task logit biases conditioning mechanism combined with the native task losses to train the shared encoders $E _ { s h a } ^ { i }$ and the unique encoders $E _ { u n i } ^ { i }$ . We freeze the backbone and only optimize the factorization encoders and the low-rank logit readouts via the native losses of understanding and generation, together with the orthogonality regularizer. We follow a simple schedule: first by enabling shared-only cross-task conditioning to stabilize the shared encoder $E _ { s h a } ^ { i } .$ , and then turn on the unique-residual injection so that the unique encoder $E _ { u n i } ^ { i }$ learns to correct the remaining task-specific residuals. The hyperparameters for Stage 1 are summarized in Table 7.   
(2) In stage 2 we freeze the factorization encoders and post-train the UMM’ s backbone using Equation 11, which including the shared alignment (Equation 10) and unique-information regularization (Equation 7) with the native losses of different tasks. As reported in Table 8, we use AdamW with EMA for optimization (with cosine learning-rate schedule), and linearly ramp $\lambda _ { \mathrm { s h a } }$ and $\lambda _ { \mathrm { u n i } }$ from 0 to 0.6 to improve early-stage stability under different objectives.

For Nexus-Gen, we proportionally resize images to approximately $5 1 2 \times 5 1 2$ resolution for both understanding and generation tasks. Under the DeepSpeed ZeRO-3 framework, the entire post-training process for the 7B model took approximately 74 hours with 8 NVIDIA RTX4090 (24GB) GPUs. For show-o, we proportionally resize images to approximately $2 5 6 \times 2 5 6$ resolution for both understanding and generation tasks. Under the DeepSpeed ZeRO-2 framework, the entire posttraining process for the 8B model took approximately 46 hours with 8 NVIDIA RTX4090 (24GB) GPUs. For Liquid, we proportionally resize images to approximately 512 × 512 resolution for both understanding and generation tasks. Under the DeepSpeed ZeRO-2 framework, the entire post-training process for the 7B model took approximately 89 hours with 8 NVIDIA RTX4090 (24GB) GPUs. Specifically, We use AdamW (Loshchilov & Hutter, 2017) for optimization and adopt EMA (Grill et al., 2020) to provide stable targets for bidirectional alignment during post-training.

Table 8. Hyperparameters and Settings in the stage 2 of post-training. 

<table><tr><td></td><td>Nexus-Gen</td><td>Show-o</td><td>Liquid</td></tr><tr><td colspan="4">Optimization</td></tr><tr><td>Optimizer</td><td>AdamW + EMA</td><td>AdamW + EMA</td><td>AdamW + EMA</td></tr><tr><td>Learning rate</td><td>5e-5</td><td>3e-5</td><td>2e-5</td></tr><tr><td>LR scheduler</td><td>Cosine</td><td>Cosine</td><td>Cosine</td></tr><tr><td>EMA decay</td><td>(0.99,0.999)</td><td>(0.99,0.999)</td><td>(0.99,0.999)</td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>Warmup steps</td><td>1000</td><td>800</td><td>1300</td></tr><tr><td>Training steps</td><td>15K</td><td>12K</td><td>20K</td></tr><tr><td>Grad. accumulation</td><td>10</td><td>12</td><td>18</td></tr><tr><td>Per-GPU batch size</td><td>6</td><td>6</td><td>6</td></tr><tr><td colspan="4">Trainable modules</td></tr><tr><td>Trainable layers</td><td>layer 8 - 18</td><td>layer 8 - 18</td><td>layer 9 - 22</td></tr><tr><td colspan="4">Loss weights</td></tr><tr><td> $\lambda_{und}$ </td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td> $\lambda_{gen}$ </td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td> $\lambda_{uni}$ </td><td>0 → 0.6</td><td>0 → 0.6</td><td>0 → 0.6</td></tr><tr><td> $\lambda_{sha}$ </td><td>0 → 0.6</td><td>0 → 0.6</td><td>0 → 0.6</td></tr></table>

# B.3. Evaluation details

We briefly introduce the benchmarks we adopted:

MMMU: Which is designed to evaluate multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning, including four challenges: (1) comprehensiveness: 11.5K college-level problems across six broad disciplines and 30 college subjects; (2) highly heterogeneous image types; (3) interleaved text and images; (4) expert-level perception and reasoning rooted in deep subject knowledge

MME: A comprehensive evaluation benchmark for multimodal large language models, measures both perception and cognition abilities on a total of 14 subtasks.

MMBench: Contains 2974 multiple-choice questions, covering 20 ability dimensions including: coarse perception, fine-grained single-instance perception, attribute reasoning, relation reasoning and logic reasoning.

MMVet: Focuses on the integration of different core vision-language capabilities, including recognition, OCR, knowledge, language generation, spatial awareness, and math.

POPE: The POPE benchmark quantifies hallucination rates in object existence verification tasks. It transforms hallucination evaluation into a set of binary classification tasks. Essentially, the MLLMs are posed Yes-or-No questions about the existence of some particular objects in the images, such as “Is there a car in the image?”

GenEval: An object-focused framework to evaluate compositional image properties such as object co-occurrence, position, count, and color with 553 prompts.

DPG: A specialized evaluation framework for text-to-image models, consisting of 1,065 lengthy and dense prompts that describe multiple objects with complex attributes and relationships. It measures a model’s semantic alignment by decomposing these complex instructions into fine-grained evaluation metrics.

WISE: The world-knowledge informed T2I evaluation with 1000 structured prompts across 25 subdomains.

Table 9. The detailed results in GenEval Benchmark.. 

<table><tr><td>Model</td><td># Params</td><td>Single Object</td><td>Two Object</td><td>Counting</td><td>Colors</td><td>Position</td><td>Color Attribute</td><td>Overall</td></tr><tr><td>Janus-Pro</td><td>7B</td><td>0.99</td><td>0.89</td><td>0.59</td><td>0.90</td><td>0.79</td><td>0.66</td><td>0.80</td></tr><tr><td>BLIP3-o</td><td>7B+1.4B</td><td>0.99</td><td>0.91</td><td>0.62</td><td>0.87</td><td>0.84</td><td>0.65</td><td>0.81</td></tr><tr><td>Bagel</td><td>8B+8B</td><td>1.00</td><td>0.95</td><td>0.82</td><td>0.89</td><td>0.66</td><td>0.65</td><td>0.84</td></tr><tr><td>OmniGen2</td><td>3B+4B</td><td>1.00</td><td>0.95</td><td>0.64</td><td>0.88</td><td>0.55</td><td>0.76</td><td>0.80</td></tr><tr><td>Emu3</td><td>8B</td><td>0.97</td><td>0.80</td><td>0.39</td><td>0.76</td><td>0.44</td><td>0.47</td><td>0.64</td></tr><tr><td>Nexus-Gen</td><td>7B</td><td>0.98</td><td>0.86</td><td>0.53</td><td>0.84</td><td>0.77</td><td>0.61</td><td>0.77</td></tr><tr><td>+DIVA</td><td>7B</td><td>0.98 (+0.00)</td><td>0.95 (+0.09)</td><td>0.60 (+0.07)</td><td>0.89 (+0.05)</td><td>0.84 (+0.07)</td><td>0.70 (+0.09)</td><td>0.83 (+0.06)</td></tr><tr><td>Show-o</td><td>1.5B</td><td>0.95</td><td>0.53</td><td>0.51</td><td>0.82</td><td>0.13</td><td>0.28</td><td>0.57</td></tr><tr><td>+DIVA</td><td>1.5B</td><td>0.96 (+0.01)</td><td>0.65 (+0.12)</td><td>0.54 (+0.03)</td><td>0.84 (+0.02)</td><td>0.27 (+0.14)</td><td>0.39 (+0.11)</td><td>0.64 (+0.07)</td></tr><tr><td>Liquid</td><td>7B</td><td>0.97</td><td>0.84</td><td>0.57</td><td>0.83</td><td>0.44</td><td>0.56</td><td>0.70</td></tr><tr><td>+DIVA</td><td>7B</td><td>0.98 (+0.01)</td><td>0.91 (+0.07)</td><td>0.66 (+0.09)</td><td>0.91 (+0.08)</td><td>0.71 (+0.27)</td><td>0.70 (+0.14)</td><td>0.82 (+0.11)</td></tr></table>

Table 10. The detailed results in WISE Benchmark. 

<table><tr><td>Model</td><td># Params</td><td>Cultural</td><td>Time</td><td>Space</td><td>Biology</td><td>Physics</td><td>Chemistry</td><td>Overall</td></tr><tr><td>Janus-Pro</td><td>7B</td><td>0.30</td><td>0.37</td><td>0.49</td><td>0.36</td><td>0.42</td><td>0.26</td><td>0.35</td></tr><tr><td>BLIP3-o</td><td>7B+1.4B</td><td>0.33</td><td>0.34</td><td>0.31</td><td>0.27</td><td>0.28</td><td>0.20</td><td>0.31</td></tr><tr><td>Bagel</td><td>8B+8B</td><td>0.43</td><td>0.52</td><td>0.67</td><td>0.45</td><td>0.60</td><td>0.46</td><td>0.52</td></tr><tr><td>OmniGen2</td><td>3B+4B</td><td>0.34</td><td>0.40</td><td>0.47</td><td>0.34</td><td>0.53</td><td>0.31</td><td>0.36</td></tr><tr><td>Emu3</td><td>8B</td><td>0.29</td><td>0.41</td><td>0.40</td><td>0.31</td><td>0.37</td><td>0.23</td><td>0.33</td></tr><tr><td>Nexus-Gen</td><td>7B</td><td>0.35</td><td>0.43</td><td>0.50</td><td>0.41</td><td>0.42</td><td>0.32</td><td>0.39</td></tr><tr><td>+DIVA</td><td>7B</td><td>0.35 (+0.00)</td><td>0.47 (+0.04)</td><td>0.64 (+0.14)</td><td>0.46 (+0.05)</td><td>0.53 (+0.11)</td><td>0.34 (+0.02)</td><td>0.45 (+0.06)</td></tr><tr><td>Show-o</td><td>1.5B</td><td>0.27</td><td>0.35</td><td>0.39</td><td>0.22</td><td>0.32</td><td>0.21</td><td>0.29</td></tr><tr><td>+DIVA</td><td>1.5B</td><td>0.29 (+0.02)</td><td>0.35 (+0.00)</td><td>0.47 (+0.08)</td><td>0.26 (+0.04)</td><td>0.44 (+0.13)</td><td>0.23 (+0.02)</td><td>0.34 (+0.05)</td></tr><tr><td>Liquid</td><td>7B</td><td>0.35</td><td>0.47</td><td>0.50</td><td>0.43</td><td>0.47</td><td>0.29</td><td>0.41</td></tr><tr><td>+DIVA</td><td>7B</td><td>0.35 (+0.00)</td><td>0.45 (-0.02)</td><td>0.60 (+0.10)</td><td>0.44 (+0.01)</td><td>0.53 (+0.06)</td><td>0.32 (+0.03)</td><td>0.44 (+0.03)</td></tr></table>

ImgEdit: Consists of 1.2 million high-quality image-editing pairs, including 1.1 million single-turn and 110,000 multi-turn samples. The benchmark specifically evaluates models across three dimensions—instruction adherence, editing quality, and detail preservation.

GEdit-Bench-EN: Designed to reflect real-world user requirements, covering 11 diverse editing tasks such as background change, subject removal, and text modification. It contains approximately 600 high-quality image-instruction pairs (within a broader dataset scale of 1K−10K samples) and utilizes advanced MLLMs like GPT-4o as automatic evaluators for metrics.

# C. More Experiment Results.

# C.1. The Detailed results on GenEval and WISE

We provide the qualitative analysis in detail across Geneval and WISE benchmark. Table 9 shows DIVA’ s consistent performance imporvements across all evaluated aspects. The detailed WISE benchmark results in Table ?? indicates that DIVA primarily enhances the model’ s ability to maintain global information consistency while showing modest improvements in reasoning-intensive tasks.

# C.2. Quantitative results

We provide more cases to demonstrate our method’ s superiority regarding the performance of generation in Figure 7. Experimental results demonstrate that by guiding the understanding end to maintain global information consistency and possessing strong capabilities in spatial structure layout and complex attribute allocation, the performance of the generation end is enhanced. The concrete text prompts is provided as follows:

(1) A white four-seater sofa.   
(2) The bathtub in the bathroom was full of bananas which also existed on the green sofa next to it.

(3) The computer desk space is decorated with mock farm animals on shelves   
(4) A lemon-flavored birthday cake.   
(5) A cat sits in the foreground of the grass, while other cats walk past behind it.   
(6) Two golden dogs lay together on the ground beside the woods.   
(7) A red and blue airplane is flying in a field.   
(8) Giraffe lying in bed with white pillows.   
(9) A red bus. Cartoon style.   
(10) A photo of a camera which is angled towards the lens.   
(11) A sailboat is trapped in a glass bottle on the ocean.   
(12) A clock is placed on the head of a sheep.   
(13) Some horses wandered under the Eiffel Tower in Paris.   
(14) Two cats standing on snowboards in the Big Ben and London Bridge.   
(15) On the table was a white sign with the word "DIVA" written in black lettering.   
(16) Two crows standing close to each other. In painting style

![](images/ae796bb67cbbf85442bc85f1465a2c3466602fa583185374f1f76a79ae50c330.jpg)  
Figure 7. Image Generation results. The generating process encompasses multiple dimensions, including world knowledge acquisition, multi-objective scenarios, complex attribute control, spatial layout, and counterfactual generation.