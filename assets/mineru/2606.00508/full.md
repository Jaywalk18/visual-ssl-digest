# Jungin Park 1 Jiyoung Lee 2 \* Kwanghoon Sohn 1 \*

# Abstract

This study introduces an intriguing phenomenon in Video LLMs: rather than merely translating frames into textual embeddings, Video LLMs establish a continuous manifold, token interface, allowing visual tokens to operate as standalone entities within the architecture. Exploiting this discovery, we propose V-LynX, a scalable framework that integrates novel modalities into Video LLMs by repurposing the internalized interface. Departing from conventional paradigms that necessitate heavy modality-specific encoders or paired supervision, V-LynX employs a lightweight auxiliary pathway in parallel with the frozen vision encoder. Our method integrates new sensory inputs with intrinsic video priors by aligning both attention responses and statistical distributions using unpaired unimodal data sets. This ensures manifold compatibility while preserving the integrity of the Video LLMs. Extensive benchmarks demonstrate that V-LynX achieves SOTA and efficiency across audio-visual QA, 3D reasoning, high-frame-rate, and multi-view video understanding. The code is available at project site.

# 1. Introduction

The advent of video large language models (Video LLMs) (Li et al., 2025a; Zhang et al., 2023; Cheng et al., 2024; Wang et al., 2024) highlights remarkable capabilities on sophisticated scene understanding by capturing longrange temporal dependencies. Nonetheless, despite their apparent multimodality, most existing Video LLMs have predominantly relied on RGB frames (optionally with text) while neglecting other rich sensory signals found in realworld environments. In existing designs, extending Video LLMs to new modalities (Cheng et al., 2024; Liu et al., 2025) typically necessitates large-scale modality-specific

![](images/d549c927ca72ee65dae3c0de8a50197e3cdc6b193d4995230c1b2b7e0a007481.jpg)

<details>
<summary>radar</summary>

| Model              | AVSD  | AVQA  | MUSIC-AVQA | ScanQA | SQA3D | VideoMME | MVBench | MLVU  | EgoExo4D |
| ------------------ | ----- | ----- | ---------- | ------ | ----- | -------- | ------- | ----- | -------- |
| LLaVA-OV-7B-ZS     | 113.0 | 92.5  | 77.5       | 105.0  | 51.0  | 61.0     | 59.5    | 66.0  | 42.5     |
| LLaVA-OV-7B-FT     | 113.0 | 90.0  | 70.0       | 100.0  | 51.0  | 61.0     | 59.5    | 63.5  | 35.0     |
| PAVE-7B            | 145.0 | 145.0 | 145.0      | 145.0  | 145.0 | 145.0    | 145.0   | 145.0 | 145.0    |
| LynX-7B            | 145.0 | 145.0 | 145.0      | 145.0  | 145.0 | 145.0    | 145.0   | 145.0 | 145.0    |
</details>

(a) Performance comparisons across 9 multimodal tasks   
![](images/8e7517c8a45d86db904c0a41caf4ecc8dfcd73020683d777ecdc6fe7da6b5585.jpg)

<details>
<summary>bar</summary>

| Category | PAVE-7B (M) | Ours-7B (M) | PAVE-0.5B (M) | Ours-0.5B (%) |
| :--- | :--- | :--- | :--- | :--- |
| For Audio | 256.7 | 195.0 | 127.6 | -47 |
| For 3D | 475.0 | 195.0 | 345.9 | -59 |
| For Video | 500.5 | 195.0 | 371.4 | -61 |
| For Audio: PAVE-7B; For 3D: PAVE-0.5B; For Video: PAVE-7B; For Audio: Ours-7B; For 3D: Ours-0.5B; For Video: Ours-0.5B; For Audio: PAVE-7B; For 3D: Ours-7B; For Video: PAVE-0.5B; For Audio: Ours-0.5B; For 3D: Ours-0.5B; For Video: Ours-0.5B; For Audio: Ours-0.5B; For 3D: Ours-0.5B; For Video: Ours-0.5B; For Audio: Ours-0.5B; For 3D: Ours-0.5B; For Video: Ours-0.5B; For Audio: Ours-0.5B; For 3D: Ours-0.5B: Ours-0.5B; For Video: Ours-0.5B: Ours-0.5B; For Audio: Ours-0.5B: Ours-0.5B; For 3D: Ours-0.5B: Ours-0.5B; For Video: Ours-0.5B: Ours-0.5B; For Audio: Ours-0.5B: Ours-0.5B; For 3D: Ours-0.5B: Ours-0.5B; For Video: Ours-0.5B; For Audio: Ours-0.5B: Ours-0.5B; For 3D: Ours-0.5B: Ours-0.5B; For Video: Ours-0.5B: Ours-0.5B; For Audio: Ours-0.5B: Ours-0.5B; For 3D: Ours-0.5B, Ours-0.5B, Ours-0.5B, Ours-0.5B, Ours-0.5B, For Audio: Ours-0.5B: Ours-0.5B: Ours-0.5B; For 3D: Ours-0.5B: Ours-0.5B: Ours-0.5B; For Video: Ours-0.5B: Ours-0.5B: Ours-0.5B; For Audio: Ours-0.5B: Ours-0.5B: Ours-0.5B; For 3D: Ours-0.5B: Ours-0.5B: Ours-0.5B; For Video: Ours-0.5B: Ours-0.5B: Ours-0.5B; For Audio: Ours-0.5B: Ours-0.18M; For 3D: Ours-0.18M; For Video: Ours-0.18M; For Audio: Ours-0.18M; For 3D: Ours-0.18M; For Video: Ours-0.18M; For Audio: Ours-0.18M; For 3D: Ours-0.18M; For Video: Ours-0.18M; For Audio: Ours-0.18M; For 3D: Ours-0.18M<nl>
</details>

(b) Extra number of parameters for each new modality   
Figure 1. V-LynX enables efficient modality expansion of pretrained Video LLMs. (a) V-LynX achieves state-of-the-art performance across diverse multimodal benchmarks with audio, 3D, and additional video, while (b) requiring significantly fewer extra parameters than PAVE (Liu et al., 2025).

encoders, complex fusion mechanisms, and paired supervision. Such designs significantly increase computational cost and architectural complexity, and degrade scalability.

This work investigates a fundamental question: How can we effectively repurpose the internalized visual pathway in Video LLMs for novel modalities? Our investigation yields a key insight that the visual encoder and projector in the Video LLM do not merely map frames onto existing vocabulary embeddings. Instead, the visual pathway carves out a continuous geometric space. This emergent space, illustrated in Figure 2, functions as a bridge that decouples sensory perception from fixed vocabulary constraints, effectively allowing the LLM to process continuous visual signals as distinct, non-symbolic entities. We term such an emergent manifold as token interface. Like ‘soft token’ view in parameter-efficient prompting (Li & Liang, 2021; Lester et al., 2021), these visual tokens occupy a geometry internalized during video-language alignment training.

![](images/e54616e611aa8cf783d2ee55860321f80f7ad1fba068b77ea2710f556fd7c918.jpg)  
Figure 2. t-SNE visualization of frame embeddings and vocabulary embeddings from the pretrained LLaVA-OV (Li et al., 2025a). We randomly sample 2,000 frames from each of the six benchmarks and 10,000 token embeddings from LLaVA-OV.

This perspective suggests a streamlined route for multimodal scaling: rather than retraining with a heavily connected modality encoder and projector, one needs only to map new sensory inputs into this existing token interface. Building on this, we introduce a novel token interface alignment method, V-LynX, that establishes a lightweight auxiliary pathway parallel to the frozen vision backbone. To ensure seamless integration, we propose a distributional alignment strategy, i.e.aligning both the attention responses and the statistical distributions of the new modality with the intrinsic video priors on unpaired unimodal data, for a more flexible adaptation to the target manifold without imposing overconstraints that may disrupt semantic coherence (Sun & Saenko, 2016; Gretton et al., 2012).

V-LynX shows the surprising modality expansion achievements on four new input types, including audio, 3D, highframe-rate videos, and egocentric videos. Across all benchmarks, V-LynX consistently yields strong gains, indicating that the video interface is reliably adapted to diverse modalities. Notably, even with a compact LLaVA-OV-0.5B backbone, V-LynX outperforms PAVE (Liu et al., 2025), the prior state-of-the-art efficient multimodal alignment method, establishing a new efficient and scalable frontier.

# 2. Related Work

Video LLMs. Video LLMs (Maaz et al., 2024; Li et al., 2023b) have emerged to understand and reason spatiotemporal visual instruction. Subsequent works advanced the video representation and cross-modal alignment strategy (e.g., Video-LLaVA (Lin et al., 2024)), alongside efforts that scaled training data, optimization recipes, and evaluation protocols (e.g., VideoChat2 (Li et al., 2024)). More recent studies such as LLaVA-OV (Li et al., 2025a), LLaVA-Video (Zhang et al., 2024), Qwen2.5-VL (Bai et al., 2025), and InternVL2.5 (Chen et al., 2024b) aim to unify image and video capabilities within a single family, and broaden task and domain coverage. In parallel, efficiency-oriented designs (Xu et al., 2024a; Weng et al., 2024) reduce tokenization overhead and computational cost for long-form video understanding. However, most works remain largely video-dominant, which limits their scalability to new modalities beyond visual inputs. In contrast, this work explores an emergent token interface, a connected bridge of visual and semantic representation spaces learned by Video LLMs for an efficient modality adaptation pathway.

Video-to-multimodal LLMs. Incorporating non-RGB signals, such as audio and 3D, into Video LLMs has recently attracted increasing research interest for richer multimodal understanding. For instance, Video-LLaMA (Zhang et al., 2023) leverages ImageBind (Girdhar et al., 2023)’s audio encoder to build a siamese audio branch to video branch, and VideoLLaMA2 (Cheng et al., 2024) strengthens audio capability by integrating a cutting-edge audio encoder (Chen et al., 2023b). While Meerkat (Chowdhury et al., 2024) targets finer spatiotemporal grounding, Video Salmonn2 (Tang et al., 2025) bootstraps language-driven audiovisual alignment by direct preference optimisation (DPO) (Rafailov et al., 2023). Instruction-tuned 3D LLMs typically rely on dedicated 3D encoders and paired supervision (Xu et al., 2024b; Chen et al., 2024a). PAVE (Liu et al., 2025) represented the closest line of work to ours in that it augments Video LLMs with an external encoder and cross-attention block-based alignment trained on multimodal pairing. Our interesting ideas rely on reusing the video pathway to LLM to adapt the distribution of new modality inputs into the video-induced token interface using only unimodal data and minimal learnable parameters.

# 3. Method

A standard Video LLM architecture typically comprises a vision encoder $g _ { \psi } ,$ a projector module $p _ { \theta }$ , and an LLM $f _ { \phi } .$ Given a video $\mathbf { X } _ { v } ,$ the visual pathway generates a sequence of latent tokens $\mathbf { Z } _ { v }$ that the LLM can interpret:

$$
\mathbf {Z} _ {v} = p _ {\theta} (g _ {\psi} (\mathbf {X} _ {v})). \tag {1}
$$

![](images/e9c314b302a32362dddf26e7458754f45f691bfb047e60857b08e1512abcc393.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Reference videos"] --> B["Vision Encoder"]
    B --> C["Projection"]
    C --> D["Distribution Regularization"]
    D --> E["Attention Response Alignment"]
    E --> F["Attention"]
    F --> G["Qm"]
    F --> H["Km"]
    F --> I["Vm"]
    G --> J["Projection"]
    H --> J
    I --> J
    J --> K["V-LynX"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#cfc,stroke:#333
    style H fill:#cfc,stroke:#333
    style I fill:#cfc,stroke:#333
    style J fill:#fcc,stroke:#333
    subgraph New modality data
        K
        L
        M
        N
        O
        P
        Q
        R
        S
        T
        U
        V
        W
        X
        Y
        Z
        AA
        AB
        AC
        AD
        AE
        AF
        AG
        AH
        AI
        AJ
        AK
        AL
        AM
        AN
        AO
        AP
        AQ
        AR
        AS
        AT
        AU
        AV
        AW
        AX
        AY
        AZ
        BA
        BB
        BC
        BD
        BE
        BF
        BG
        BH
        BI
        BJ
        BK
        BL
        BM
        BN
        BO
        BP
        BPB
        BZ
    end
```
</details>

(a) Interface guidance   
(b) Interface alignment with unpaired data

![](images/bc3d93802c19713b0d1a7bc652b0d118d7acbc56870c6832ebed9a433c5d01d6.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Projector"] --> B["V-LynX"]
    B --> C["Vision Encoder"]
    C --> D["Large Language Model"]
    D --> E["LoRA"]
    E --> F["Tokenizer"]
    F --> G["Select the best answer to the following question:"]
    H["Answer"] --> D
    I["Multimodal instruction data"] --> B
```
</details>

(c) Instruction tuning   
Figure 3. Overall framework of our V-LynX. (a) We first extract interface guidance from a set of available videos and (b) learn LoRAs in the vision encoder to adapt the interface to given new modality data through attention response alignment and distribution regularization. (c) We then train additional LoRAs in the LLM on diverse instruction datasets.

These visual tokens $\mathbf { Z } _ { v }$ inhabit a functional token interface within the LLM’s high-dimensional space. The existence of this interface suggests that the pretrained visual pathway $( \theta , \psi )$ has already internalized a geometric prior for LLM-compatible sensory tokens. Consequently, extending the model to a new modality $\mathbf { X } _ { m }$ does not necessitate a complete architectural overhaul or paired multimodal supervision. Instead, a key challenge is to make tokens from the new modality compatible with the model’s native video behavior, including the attention responses inside the encoder and the token distribution expected by the projector and LLM. To this end, our V-LynX repurposes the frozen visual backbone to accommodate novel sensory inputs, where the distributional alignment preserves the attention dynamics and statistical properties internalized during video-language training. The overall procedure of V-LynX is shown in Figure 3 and Algorithm 1.

# 3.1. Shared video-path for novel modality

Integrating new modality in Video LLMs is fundamentally constrained by two factors: the dependence on paired crossmodal datasets (Akbari et al., 2021) and the risk of catastrophic forgetting when reusing or adapting existing encoders (Zhou et al., 2025a). While paired data enables explicit alignment, it is costly and inflexible, and encoder adaptation often compromises previously acquired knowledge. To overcome these limitations, V-LynX reuses the frozen vision encoder with a small set of learnable parameters $\Delta \psi ,$ , implemented via low-rank adaptation modules (i.e., LoRA (Hu et al., 2022)) within self-attention layers. Namely, the visual pathway by routing visual tokens through the frozen parameters ψ during inference, while selectively activating additional learnable parameters $\Delta \psi$ only for the new modality. By reusing the same architectural path for both modalities, the model supports efficient modality extension without a separate interface (Liu et al., 2025).

# 3.2. Interface alignment with unpaired unimodal data

While the shared-path architecture ensures parameterefficient adaptation, effective integration of a new modality further requires aligning its representations with the token interface expected by the LLM. Conventional alignments (Akbari et al., 2021; Girdhar et al., 2023) rely on paired crossmodal supervision (e.g., 3D-video-text, audio-video-text), which is often prohibitively scarce or unavailable for diverse target modalities. To circumvent these constraints, V-LynX learns additional parameters solely on unimodal data (e.g., audio, 3D, and multi-view).

Video-derived interface guidance. To anchor modality adaptation to the interface expected by the LLM, the behavior of the pretrained Video LLM is first characterized on its native modality (i.e., videos). Specifically, a set of available unlabeled videos, V, is used to estimate reference statistics that describe how visual tokens are processed within the encoder and subsequently projected to the LLM’s token space. At the encoder level, we extract averaged Key and Value embeddings at each attention layer:

$$
K _ {v} ^ {(l)} = \mathbb {E} _ {\mathbf {X} _ {v} \sim \nu} [ K _ {\psi} ^ {(l)} (\mathbf {X} _ {v}) ], \tag {2}
$$

$$
V _ {v} ^ {(l)} = \mathbb {E} _ {\mathbf {X} _ {v} \sim \mathcal {V}} [ V _ {\psi} ^ {(l)} (\mathbf {X} _ {v}) ],
$$

where $\mathbf { X } _ { v }$ is an input video sampled from V, $K _ { \psi } ^ { ( l ) }$ and $V _ { \psi } ^ { ( l ) }$ are projections producing Key and Value at the l-th layer, respectively. These mean embeddings capture the typical attention space induced by videos and serve as stable anchors for encoder-level alignments. Simultaneously, the distribution of latent video tokens is characterized at the projector level. Let ${ \bf Z } _ { v } = p _ { \theta } \big ( g _ { \psi } ( { \bf X } _ { v } ) \big )$ ) denote the projector output. The mean $\mu _ { v }$ and variance $\sigma _ { v } ^ { 2 }$ of the projected video embeddings are computed as

$$
\mu_ {v} = \mathbb {E} _ {\mathbf {X} _ {v} \sim \mathcal {V}} [ \mathbf {Z} _ {v} ], \quad \sigma_ {v} ^ {2} = \mathbb {E} _ {\mathbf {X} _ {v} \sim \mathcal {V}} [ (\mathbf {Z} _ {v} - \mu_ {v}) ^ {2} ]. \tag {3}
$$

The pre-computed reference serves as the target statistic to enable new modality tokens to be compatible with LLM.

Attention response alignment. The proposed attention alignment objective is introduced to ensure that inputs from a new modality activate the shared visual pathway in a manner compatible with existing video priors. In Video LLMs, the vision encoder constitutes the earliest stage at which heterogeneous inputs are processed through a common computational structure, and its internal attention dynamics largely determine how information is selected, aggregated, and propagated to downstream modules. We insist that while the projector and LLM operate on encoder outputs, they work on summarized token-level reasoning (Li et al., 2025b). For effective modality integration, alignment must therefore be applied at the level of encoder attention, where functional computation is formed.

Given a new modality input $\mathbf { X } _ { m }$ of a set of newly introduced target modality data M, Query, Key, and Value embeddings are obtained from the encoder with original and learnable parameters $( i . e . , \psi + \Delta \psi )$ at each layer. The target attention response $O _ { m } ^ { ( l ) }$ of $\mathbf { X } _ { m }$ is,

$$
O _ {m} ^ {(l)} = \mathrm{Attn} (Q _ {m} ^ {(l)}, K _ {m} ^ {(l)}, V _ {m} ^ {(l)}). \tag {4}
$$

The reference response $\tilde { O } _ { m } ^ { ( l ) }$ is computed via video-derived Key $K _ { v } ^ { ( l ) }$ and Value $V _ { v } ^ { ( l ) }$ as references, providing a stable and well-calibrated attention behavior:

$$
\tilde {O} _ {m} ^ {(l)} = \mathrm{Attn} (Q _ {m} ^ {(l)}, K _ {v} ^ {(l)}, V _ {v} ^ {(l)}). \tag {5}
$$

The Key-Value embeddings define how tokens are matched and aggregated within the shared attention framework, which directly shapes the functional operation of the encoder. By conditioning on the same Query embedding $Q _ { m } ^ { ( l ) }$ while replacing the Key-Value pairs with video, the reference response specifies how the new modality should interact with the existing attention mechanism to remain compatible with the video-derived interface. The attention alignment loss minimizes the discrepancy between the target and reference attention responses:

$$
\mathcal {L} _ {\text { attn }} = \sum_ {l} | | O _ {m} ^ {(l)} - \tilde {O} _ {m} ^ {(l)} | | _ {1}. \tag {6}
$$

This objective promotes internal cross-modal alignment rather than raw feature similarity. Therefore, pairindependent modality adaptation is achieved while preserving the original vision-language interface.

Distribution regularization. Attention alignment alone does not guarantee that the projector’s outputs lie in the distribution the LLM expects. To constrain the distribution (i.e., mean and variance) of new modality, we compute a statistic of projected modality tokens ${ \bf Z } _ { m } = p _ { \theta } ( g _ { \psi + \Delta \psi } ( \tilde { x } _ { m } ) )$ obtained by the projector pθ:

$$
\mu_ {m} = \mathbb {E} _ {\mathbf {X} _ {m} \sim \mathcal {M}} [ \mathbf {Z} _ {m} ], \quad \sigma_ {m} ^ {2} = \mathbb {E} _ {\mathbf {X} _ {m} \sim \mathcal {M}} [ (\mathbf {Z} _ {m} - \mu_ {m}) ^ {2} ]. \tag {7}
$$

We align the token distributions by applying the meansquared error between a reference distribution $\mathbf { Z } _ { v }$ and the learned distribution $\mathbf { Z } _ { m }$ :

$$
\mathcal {L} _ {\text { stat }} = | | \mu_ {v} - \mu_ {m} | | _ {2} + | | \sigma_ {v} ^ {2} - \sigma_ {m} ^ {2} | | _ {2}. \tag {8}
$$

Overall objective. We train the LoRA parameters $\Delta \psi$ in the encoder with the following objective:

$$
\mathcal {L} _ {\mathrm{V-LynX}} = \mathcal {L} _ {\text { attn }} + \beta \cdot \mathcal {L} _ {\text { stat }}, \tag {9}
$$

where $\beta$ controls the trade-off between attention alignment and training stability. Given that we do not require an additional modality-specific encoder and paired multimodal data, our V-LynX is data- and parameter-efficient solution to establish multimodal LLMs.

# 3.3. Instruction tuning

After alignment learning, $p _ { \theta } \big ( g _ { \psi + \Delta \psi } ( \cdot ) \big )$ produces the tokens from new modality data in a form that the LLM can interpret. Conditioned on the embeddings from visual and new modality data $( i . e . , \mathbf { Z } _ { v }$ and $\mathbf { Z } _ { m } )$ , we perform supervised fine-tuning by applying additional LoRA layers to the LLM. Specifically, we train a set of LoRA parameters $\Delta \phi$ to enable the LLM to maximize the likelihood of the autoregressively generated answer:

$$
\mathcal {L} _ {\mathrm{sft}} = - \sum_ {n = 1} ^ {N} \log P (\mathbf {a} _ {n} | \mathbf {A} _ {<   n}, \mathbf {Q}, \mathbf {Z} _ {v}, \mathbf {Z} _ {m}), \tag {10}
$$

where N is the number of tokens in the answer, $\mathbf { A } _ { < n } =$ $\{ \mathbf { a } _ { 1 } , . . . , \mathbf { a } _ { n - 1 } \}$ denotes the sequence of tokens prior to the autoregressive decoding step n, and Q is a set of instruction tokens.

# 3.4. Interpretation of V-LynX.

Recently, Huh et al. (2024) suggests that representations learned across different models and modalities may share structural regularities of the underlying world. From the Platonic representation perspective (Huh et al., 2024), the token interface can be interpreted as the organized structural regularity for the LLM. We speculate that V-LynX’s formulation works on such an interpretation: if a new modality contains a world structure that overlaps with video, it can be adapted by learning a modality-specific pathway into this existing interface. Accordingly, V-LynX aligns new modality inputs to the attention behavior and projector-level token statistics of the pretrained token interface, enabling the LLM to interpret them through the same operational regime while preserving the original video-language pathway. We provide more analysis in Section B.1.

Table 1. Performance comparison on audio-visual QA. We report CIDEr score on AVSD and the accuracy (Acc.) on AVQA and MUSIC-AVQA, respectively. ‘∆Params.’ indicates the number of additional parameters than LLaVA-OV-0.5B/-7B. 

<table><tr><td rowspan="2">Method</td><td>AVSD</td><td>AVQA</td><td colspan="4">MUSIC-AVQA</td><td rowspan="2">ΔParams.</td></tr><tr><td>CIDEr</td><td>Acc.</td><td>Audio Acc.</td><td>Visual Acc.</td><td>Audio-Visual Acc.</td><td>Overall Acc.</td></tr><tr><td colspan="8">Zero-shot Video LLMs</td></tr><tr><td>CAT-7B (Ye et al., 2024)</td><td>79.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>48.6</td><td>-</td></tr><tr><td>LLaVA-OV-0.5B (Li et al., 2025a)</td><td>65.1</td><td>77.4</td><td>60.0</td><td>57.1</td><td>48.5</td><td>52.8</td><td>-</td></tr><tr><td>LLaVA-OV-7B (Li et al., 2025a)</td><td>70.6</td><td>85.6</td><td>68.8</td><td>70.6</td><td>52.8</td><td>60.4</td><td>-</td></tr><tr><td colspan="8">Task-specific models &lt; 7B</td></tr><tr><td>COST (Pham et al., 2022)</td><td>108.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PSTP-Net (Li et al., 2023a)</td><td>-</td><td>90.2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>VAST (Chen et al., 2023a)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>80.7</td><td>-</td></tr><tr><td>LLaVA-OV-0.5B-FT (Li et al., 2025a)</td><td>117.6</td><td>86.4</td><td>69.6</td><td>76.3</td><td>62.8</td><td>67.6</td><td>35.2M</td></tr><tr><td>PAVE-0.5B (Liu et al., 2025)</td><td>134.5</td><td>90.4</td><td>77.3</td><td>89.8</td><td>74.1</td><td>78.8</td><td>127.6M</td></tr><tr><td>V-LynX-0.5B (Ours)</td><td>145.7</td><td>93.1</td><td>78.9</td><td>92.2</td><td>76.5</td><td>81.1</td><td>68.7M</td></tr><tr><td colspan="8">Task-specific models ≥ 7B</td></tr><tr><td>CAT-7B-FT (Ye et al., 2024)</td><td>-</td><td>92.0</td><td>84.9</td><td>86.1</td><td>83.2</td><td>84.3</td><td>-</td></tr><tr><td>LLaVA-OV-7B-FT (Li et al., 2025a)</td><td>124.9</td><td>90.8</td><td>75.4</td><td>89.3</td><td>72.3</td><td>77.4</td><td>161.5M</td></tr><tr><td>PAVE-7B (Liu et al., 2025)</td><td>152.9</td><td>93.8</td><td>79.7</td><td>93.0</td><td>78.0</td><td>82.3</td><td>256.7M</td></tr><tr><td>V-LynX-7B (Ours)</td><td>163.0</td><td>94.2</td><td>80.8</td><td>93.5</td><td>78.8</td><td>83.0</td><td>195.0M</td></tr></table>

# 4. Experiment

# 4.1. Default configuration

Video LLM backbone. LLaVA-OneVision (LLaVA-OV) (Li et al., 2025a) is our base Video LLM. LLaVA-OV employs SigLIP (Zhai et al., 2023) as the vision encoder $g _ { \psi }$ , Qwen-2 (Team et al., 2024) as the LLM $f _ { \phi } ,$ and a 2-layer MLP as the projector $p _ { \theta }$ . To validate the scalability of our V-LynX, we primarily employ LLaVA-OV-0.5B and -7B.

Reference videos. For a video-derived interface guidance, we first gather a set of reference videos V from training sets of all benchmarks, including AVSD (Alamri et al., 2019), AVQA (Yang et al., 2022), MUSIC-AVQA (Li et al., 2022), ScanNet (Dai et al., 2017), a subset of LLaVA-Video-178K (Zhang et al., 2024), and Ego-Exo4D (Grauman et al., 2024), with a total number of videos of ∼117k. Given $\nu ,$ we sample frames with 1 fps and extract Key, Value, and video token distribution are computed across all video features from the pretrained vision encoder and projector. In this regard, we will explore the effectiveness according to the scale of V in Table 7.

LoRAs in $\Delta \psi$ and $\Delta \phi .$ . We set rank r to 64 for LoRAs in both the vision encoder $( \Delta \psi )$ and LLM $( \Delta \phi )$ . Meanwhile, α is set to 128 and 16 for $\Delta \psi$ and $\Delta \phi .$ , respectively. We keep the identical settings across the modality.

Baselines. We mainly compare our V-LynX with two approaches: (1) LLaVA-OV-FT, fine-tuned through instruction tuning by LoRA without target modality data; and (2) PAVE (Liu et al., 2025), which employs a target modality encoder and incorporates it via the cross-attention module. They share the same Video LLM backbone, allowing for a fair comparison. In addition, we provide zero-shot performance of LLaVA-OV-0.5B, and -7B.

# 4.2. Audio-visual QA

Audio, while inherently synchronized with video in nature, remains a structurally heterogeneous modality that presents a significant representation gap for vision-centric models. We extend Video LLMs by assessing integrated sensory reasoning through audio-visual QA.

Datasets. We evaluate our V-LynX on three audio-visual QA benchmarks: AVSD (Alamri et al., 2019) consists of 79k open-ended QA pairs with 7.9k videos for training and 1k audio-visual questions for evaluation. We report the CIDEr score. AVQA (Yang et al., 2022) contains 40k videos with each closed-form QA pair for training. We evaluate with 17k questions and report the accuracy. MUSIC-AVQA (Li et al., 2022) provides questions, which are categorized into visual, audio, and audio-visual questions. We use 32k QA pairs from 9.2k videos for training and measure the accuracy on 9.1k questions.

Preprocessing. We resample the audio to 16 kHz and convert waveforms into normalized log-mel spectrograms.

Additional baselines. We consider task-specific models that are fine-tuned on the target dataset, including COST (Pham et al., 2022), PSTP-Net (Li et al., 2023a), CAT-7B (Ye et al., 2024), and VAST (Chen et al., 2023a); and zero-shot performance from CAT-7B.

Results. Table 1 shows audio-visual reasoning performance evaluated on three benchmarks. Our V-LynX consistently improves over both zero-shot and task-specific baselines, indicating that the video-induced token interface can be reliably transferred to audio. Notably, V-LynX-0.5B outperforms LLaVA-OV-0.5B-FT and PAVE-0.5B across all benchmarks. Comparison between V-LynX-7B and PAVE-7B further highlights the effectiveness of V-LynX, improving AVSD by +10.1 CIDEr and MUSIC-AVQA by +0.7% with 24% fewer additional parameters. Even though CAT-7B-FT was tailored to audio-visual LLM with aligned audio and video encoders (Girdhar et al., 2023), V-LynX-7B achieves the best score in AVQA.

Table 2. 3D reasoning on 3D QA benchmarks. We report CIDEr, BLEU-4, METEOR, ROUGE scores for ScanQA, and top-1 Exact Match (EM@1) and (refined EM@1) for both ScanQA and SQA3D, respectively. 

<table><tr><td rowspan="2">Method</td><td colspan="5">ScanQA</td><td>SQA3D</td><td rowspan="2">ΔParams.</td></tr><tr><td>CIDEr</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE</td><td>EM@1</td><td>EM@1</td></tr><tr><td colspan="8">Zero-shot Video LLMs</td></tr><tr><td>VideoChat2-7B (Li et al., 2024)</td><td>49.2</td><td>9.6</td><td>9.5</td><td>28.2</td><td>19.2</td><td>37.3</td><td>-</td></tr><tr><td>LLaVA-OV-0.5B (Li et al., 2025a)</td><td>17.2</td><td>1.2</td><td>13.7</td><td>18.4</td><td>0.2 (28.0)</td><td>0.8 (43.0)</td><td>-</td></tr><tr><td>LLaVA-OV-7B (Li et al., 2025a)</td><td>91.0</td><td>5.3</td><td>18.2</td><td>45.9</td><td>26.7</td><td>8.3</td><td>-</td></tr><tr><td colspan="8">Task-specific models &lt; 7B</td></tr><tr><td>LLaVA-OV-0.5B-FT (Li et al., 2025a)</td><td>70.5</td><td>6.5</td><td>14.3</td><td>36.9</td><td>20.1 (36.3)</td><td>44.1 (45.7)</td><td>35.2M</td></tr><tr><td>PAVE-0.5B (Liu et al., 2025)</td><td>84.2</td><td>13.1</td><td>17.0</td><td>42.1</td><td>23.1 (40.0)</td><td>51.1 (52.8)</td><td>345.9M</td></tr><tr><td>V-LynX-0.5B (Ours)</td><td>87.1</td><td>14.3</td><td>17.2</td><td>43.8</td><td>26.4 (44.2)</td><td>52.2 (54.2)</td><td>68.7M</td></tr><tr><td colspan="8">Task-specific models ≥ 7B</td></tr><tr><td>3D-LLM-7B (Hong et al., 2023)</td><td>74.5</td><td>12.9</td><td>15.1</td><td>37.5</td><td>21.2</td><td>49.8</td><td>-</td></tr><tr><td>LEO-7B (Huang et al., 2024)</td><td>101.4</td><td>13.2</td><td>20.0</td><td>49.2</td><td>24.5 (47.6)</td><td>50.0 (52.4)</td><td>-</td></tr><tr><td>Scene-LLM-7B (Fu et al., 2025b)</td><td>80.0</td><td>12.0</td><td>16.6</td><td>40.0</td><td>27.2</td><td>54.2</td><td>-</td></tr><tr><td>LLaVA-3D-7B (Zhu et al., 2025)</td><td>91.7</td><td>14.5</td><td>20.7</td><td>50.1</td><td>27.0 (45.0)</td><td>55.6 (57.6)</td><td>-</td></tr><tr><td>LLaVA-OV-7B-FT (Li et al., 2025a)</td><td>95.1</td><td>13.5</td><td>19.1</td><td>47.4</td><td>27.4 (46.3)</td><td>55.8 (58.1)</td><td>161.5M</td></tr><tr><td>PAVE-7B (Liu et al., 2025)</td><td>103.4</td><td>16.0</td><td>19.9</td><td>49.0</td><td>29.1 (48.5)</td><td>59.0 (61.4)</td><td>475.0M</td></tr><tr><td>V-LynX-7B (Ours)</td><td>107.4</td><td>16.7</td><td>20.8</td><td>50.3</td><td>29.7 (48.6)</td><td>60.5 (62.6)</td><td>195.0M</td></tr></table>

# 4.3. 3D QA

We now consider 3D information as new modality data and evaluate the model on 3D QA tasks. The goal of 3D QA is to answer questions about the objects in a 3D scene and their relationships, such as relative spatial positions.

Datasets. We evaluate the 3D QA performance on two benchmarks that share the same 3D scanning dataset, i.e., ScanNet (Dai et al., 2017). Since the following two benchmarks share most of the videos, the vision encoder is trained to share, and LoRAs in LLM are separately learned in instruction tuning stages. ScanQA (Azuma et al., 2022) contains 25k QA pairs for training and 4.6k questions for evaluation. We report the CIDEr, BLEU-4, METEOR, ROUGE, and top-1 Exact Match (EM@1) scores. SQA3D (Ma et al., 2023) includes 26k QA pairs and 3.5k questions for training and evaluation, respectively. We report the EM@1 score.

Preprocessing. Following (Girdhar et al., 2022), the depth map is converted into disparity maps, then processed as a 3-channel image with an RGB LookUp Table. Different from the previous works (Zhu et al., 2025; Liu et al., 2025), which take geometry-aggregated multi-view features, our V-LynX requires only a depth map for 3D QA.

Additional baselines. As baselines, we consider finetuned 3D-specific models, such as 3D-LLM (Hong et al., 2023), LEO (Huang et al., 2024), Scene-LLM (Fu et al., 2025b), and LLaVA-3D (Zhu et al., 2025), and the zero-shot performance from VideoChat2 (Li et al., 2024).

Results. As shown in Table 2, V-LynX-0.5B attains the strongest performance among sub-7B baselines, achieving 87.1 CIDEr and 26.4 EM@1 on ScanQA, and 52.2 EM@1 on SQA3D while introducing only 68.7M additional parameters. Scaling to 7B further improves accuracy across the baselines: V-LynX-7B delivers the best overall results, reaching 107.4 CIDEr and 29.7 EM@1 on ScanQA, and 60.5 EM@1 on SQA3D. Notably, it surpasses PAVE-7B (Liu et al., 2025) while requiring 59% fewer added parameters (195.0M vs. 475.0M), and consistently outperforms LLaVA-OV-7B-FT (Li et al., 2025a) (161.5M) with only a modest increase in adaptation cost. Collectively, these results indicate that V-LynX sustains the benefits of scaling while keeping modality adaptation lightweight, outperforming heavier patching-based alternatives at both model sizes. In addition, it is worth noting that our V-LynX outperforms baselines without geometry-aggregated multi-view features in (Zhu et al., 2025; Liu et al., 2025). We provide additional results evaluated on SQA3D with different backbones, including Qwen2.5-VL-3B (Bai et al., 2025) and InternVL-2.5-4B (Chen et al., 2024b) in Section B.2.

# 4.4. Enhanced video QA

In video understanding (VU), high-frame-rate video offers fine-grained motion cues and short-lived temporal dynamics, capturing fast actions and subtle interactions (Feichtenhofer et al., 2019; Park et al., 2023). We rethink such videos as additional information and evaluate the model on multidomain generalized VU tasks.

Table 3. High-frame-rate video understanding on diverse video QA benchmarks. Accuracy scores are reported. For MVBench, we report the performance evaluated on stage change (SC), fine-grained pose (FGP), and object shuffle (OS) subsets, following Liu et al. (2025). 

<table><tr><td rowspan="2">Method</td><td colspan="4">VideoMME</td><td colspan="4">MVBench</td><td>MLVU</td><td rowspan="2">ΔParams.</td></tr><tr><td>Short</td><td>Median</td><td>Long</td><td>Avg.</td><td>SC</td><td>FGP</td><td>OS</td><td>Avg.</td><td>Acc.</td></tr><tr><td colspan="11">Task-specific models &lt; 7B</td></tr><tr><td>LLaVA-OV-0.5B (Li et al., 2025a)</td><td>53.4</td><td>41.2</td><td>37.3</td><td>44.0</td><td>37.5</td><td>49.0</td><td>33.0</td><td>45.5</td><td>50.3</td><td>-</td></tr><tr><td>PAVE-0.5B (Liu et al., 2025)</td><td>57.8</td><td>42.7</td><td>37.4</td><td>46.0</td><td>40.0</td><td>54.0</td><td>35.5</td><td>46.6</td><td>51.6</td><td>371.4M</td></tr><tr><td>V-LynX-0.5B (Ours)</td><td>63.1</td><td>50.7</td><td>44.6</td><td>52.8</td><td>45.5</td><td>54.5</td><td>37.5</td><td>53.7</td><td>55.0</td><td>68.7M</td></tr><tr><td colspan="11">Task-specific models ≥ 7B</td></tr><tr><td>LLaVA-OV-7B (Li et al., 2025a)</td><td>70.1</td><td>56.6</td><td>48.9</td><td>58.2</td><td>52.0</td><td>53.0</td><td>35.5</td><td>56.7</td><td>64.7</td><td>-</td></tr><tr><td>PAVE-7B (Liu et al., 2025)</td><td>71.1</td><td>59.4</td><td>49.2</td><td>59.9</td><td>51.5</td><td>54.5</td><td>39.0</td><td>58.0</td><td>67.0</td><td>500.5M</td></tr><tr><td>V-LynX-7B (Ours)</td><td>73.0</td><td>61.2</td><td>53.8</td><td>62.7</td><td>53.5</td><td>54.0</td><td>42.0</td><td>61.2</td><td>68.4</td><td>195.0M</td></tr></table>

Datasets. Following Liu et al. (2025), we train the model only on LLaVA-Video-178K and report the accuracy on remaining evaluation benchmarks. LLaVA-Video-178K (Zhang et al., 2024) is a large-scale video instructiontuning dataset. In line with (Liu et al., 2025), we train on a 114k QA subset, consisting of 57k videos (each > 1 min) with two QA pairs per video. VideoMME (Fu et al., 2025a) provides a comprehensive multi-domain video QA benchmark with 900 videos and 2.7k four-option multiple-choice questions. MVBench (Li et al., 2024) contains 20 VU subtasks (e.g., object shuffle and fine-grained pose), with about 3.9k videos and 4k questions in total. MLVU (Zhou et al., 2025b) focuses on long-video understanding and includes 1.3k long videos and 2.1k questions.

Preprocessing. Processing high-frame-rate videos inevitably incurs more computational overhead. To mitigate this, we adopt a frame-stacking strategy (Park et al., 2023) that aggregates temporal information into a single spatial representation. We downsample the spatial dimensions of four consecutive frames by a factor of 0.5 and tile them into a single frame. Consequently, the vision encoder processes ×4 temporal context with the same computational cost.

Results. Table 3 demonstrates that V-LynX-0.5B delivers the best performance across all benchmarks while remaining markedly parameter-efficient. Our V-LynX outperforms PAVE by +6.8%, +7.1%, and +3.4% on VideoMME, MVBench, and MLVU, respectively, despite introducing 81% fewer additional parameters. Moreover, V-LynX-7B attains the top overall accuracy, i.e., 62.7 on VideoMME, 61.2 on MVBench, and 68.4 on MLVU. The only exception is MVBench fine-grained pose (FGP), where V-LynX underperforms by 0.5%; we attribute this to resolution-sensitive cues being attenuated by the downsampling used in preprocessing. Furthermore, while V-LynX maintains a minimal

Table 4. Multi-view video understanding on DPE benchmark. 

<table><tr><td>Method</td><td>Acc.</td><td>ΔParams.</td></tr><tr><td colspan="3">Zero-shot Video LLMs</td></tr><tr><td>LLaVA-OV-0.5B (Li et al., 2025a)</td><td>23.6</td><td>-</td></tr><tr><td>LLaVA-OV-7B (Li et al., 2025a)</td><td>23.6</td><td>-</td></tr><tr><td colspan="3">Task-specific models</td></tr><tr><td>LLaVA-OV-0.5B-FT (Li et al., 2025a)</td><td>28.2</td><td>35.2M</td></tr><tr><td>LLaVA-OV-7B-FT (Li et al., 2025a)</td><td>29.8</td><td>161.5M</td></tr><tr><td>TimeSFormer (Bertasius et al., 2021)</td><td>43.7</td><td>-</td></tr><tr><td>PAVE-0.5B (Liu et al., 2025)</td><td>32.4</td><td>41.4M</td></tr><tr><td>PAVE-7B (Liu et al., 2025)</td><td>44.2</td><td>170.5M</td></tr><tr><td>V-LynX-0.5B (Ours)</td><td>38.6</td><td>68.7M</td></tr><tr><td>V-LynX-7B (Ours)</td><td>46.9</td><td>195.0M</td></tr></table>

parameter footprint, PAVE’s reliance on modality-specific backbones–such as 330M for video encoder from Zhu et al. (2023)–leads to a prohibitive parameter explosion as the number of supported modalities increases.

# 4.5. Multi-view video understanding

Due to distinct FoV and motion patterns (Grauman et al., 2024; Park et al., 2025), we treat egocentric (first-person) videos as a separate modality. Since they naturally match the vision encoders input format, we process them directly.

Datasets. Following Liu et al. (2025), we employ demonstrator proficiency estimation (DPE) benchmark from Ego-Exo4D (Grauman et al., 2024) that aims to classify human action proficiency into one of four skill levels from a timesynchronized multi-view videos (one ego video and optionally four exo videos). We report accuracy scores.

Additional baselines. We include TimeSFormer (Bertasius et al., 2021) trained on paired ego-exo videos.

Results. In Table 4, the zero-shot performance of the baselines suggests that, despite strong video-language priors, pretrained LLMs lack the egocentric-specific cues required for reliable proficiency estimation. Comparisons between task-specific models show that our V-LynX consistently outperforms TimeSFormer and PAVE. While V-LynX introduces a marginal parameter increment to capture the domain shift in egocentric videos, unlike PAVE, which relies on a shared encoder for disparate perspectives, our V-LynX achieves remarkable improvement in both 0.5B and 7B models by +6.2% and +2.7%, respectively.

Table 5. Component analysis in V-LynX on ScanQA. 

<table><tr><td>Method</td><td>C</td><td>B-4</td><td>M</td><td>R</td><td>EM@1</td></tr><tr><td>V-LynX</td><td>87.1</td><td>14.3</td><td>17.2</td><td>43.8</td><td>26.4 (44.2)</td></tr><tr><td>- Attn. Align.</td><td>81.0</td><td>11.8</td><td>16.3</td><td>41.2</td><td>23.5 (40.4)</td></tr><tr><td>- Dist. Reg.</td><td>86.2</td><td>13.4</td><td>17.1</td><td>43.5</td><td>25.6 (43.0)</td></tr><tr><td>- Interface Adapt.</td><td>77.3</td><td>10.9</td><td>15.7</td><td>39.1</td><td>22.4 (39.9)</td></tr></table>

Table 6. Different rank r of LoRAs in ∆ψ on ScanQA. 

<table><tr><td>r</td><td>C</td><td>B-4</td><td>M</td><td>R</td><td>EM@1</td><td>ΔParams.</td></tr><tr><td>8</td><td>86.1</td><td>13.1</td><td>16.2</td><td>42.8</td><td>24.9 (42.7)</td><td>39.4M</td></tr><tr><td>16</td><td>86.8</td><td>13.6</td><td>17.1</td><td>43.0</td><td>25.3 (43.3)</td><td>43.6M</td></tr><tr><td>32</td><td>86.9</td><td>13.7</td><td>17.2</td><td>43.3</td><td>26.3 (44.0)</td><td>51.9M</td></tr><tr><td>64</td><td>87.1</td><td>14.3</td><td>17.2</td><td>43.8</td><td>26.4 (44.2)</td><td>68.7M</td></tr></table>

# 4.6. Ablation studies

All experiments in this section are conducted on ScanQA.

Objective. We compare V-LynX variants by ablating each objective used for interface alignment, i.e., attention response alignment and distribution regularization. As shown in Table 5, removing attention alignment yields the larger performance drop, indicating that aligning attention responses is a primary component to adapt the pretrained model to new modality data. In contrast, ablating distribution regularization causes a minor degradation, suggesting the regularization plays a complementary role by stabilizing token statistics rather than defining the actual mapping. Finally, training LoRAs of the vision encoder and LLM through only instruction tuning (–Interface Align.) leads to a substantial collapse, demonstrating that interface alignment is essential for transferring new modality representations into the LLM-compatible token interface.

LoRA rank in vision encoder. Table 6 indicates that even low-rank adapters achieve strong performance. Increasing capacity achieves diminishing yet consistent improvements, where r = 64 provides the best results of 87.1 CIDEr and 26.4 EM@1 scores with 68.7M parameters. Overall, V-LynX is robust to the choice of r and remains parameterefficient, retaining most gains at small ranks.

Source and scale of reference V. As shown in Table 7, V-LynX is remarkably resilient to distribution shifts. Using out-of-distribution audio-related videos (57k) achieves 87.7 CIDEr and 25.9 EM@1, while a minimal, 3D set of 563 clips remains highly competitive at 87.8 CIDEr and 26.3 EM@1. Overall results underscore that V-LynX achieves accurate interface adaptation without requiring large or strictly in-domain reference V, in that the averaged reference statistics provide a stable target across diverse sources.

Table 7. Reference V choices on ScanQA. |V| indicate the number of videos. 

<table><tr><td>Source</td><td>C</td><td>B-4</td><td>M</td><td>R</td><td>EM@1</td><td> $|\mathcal{V}|$ </td></tr><tr><td>Audio</td><td>87.7</td><td>14.2</td><td>17.3</td><td>43.8</td><td>25.9 (43.6)</td><td>57k</td></tr><tr><td>3D</td><td>87.8</td><td>14.8</td><td>17.3</td><td>43.7</td><td>26.3 (43.9)</td><td>563</td></tr><tr><td>Video</td><td>87.8</td><td>14.3</td><td>17.2</td><td>43.7</td><td>25.8 (43.5)</td><td>59k</td></tr><tr><td>All</td><td>87.1</td><td>14.3</td><td>17.2</td><td>43.8</td><td>26.4 (44.2)</td><td>117k</td></tr></table>

# 4.7. Visualization

Qualitative results. We provide qualitative comparisons between our V-LynX and PAVE (Liu et al., 2025) on 3D QA and audio-visual QA in Section B.3.

Attention visualization. In Figure 4, we extract attention maps between question tokens and the corresponding modality tokens from a given scene-question pair from ScanQA. Especially for the 3D inputs, we illustrate the attention maps with and without our V-LynX to demonstrate the actual functionality of the modality-specific pathway. The results indicate that the model with V-LynX attends to consistent, question-relevant regions across modalities (e.g., focusing on the target object areas for white pillow or monitor), suggesting that the adapted modality representations participate in the same functional routing used for reasoning.

# 5. Conclusion

In this work, we introduced the token interface, an emergent and functional manifold within Video LLMs. Leveraging this insight, we proposed V-LynX, a highly efficient framework for multimodal expansion. By aligning new modalities to this internalized interface space using only unimodal data and a lightweight auxiliary pathway, V-LynX achieves stateof-the-art performance across diverse settings, including audio-visual QA, 3D reasoning, high-frame-rate video understanding, and multi-view proficiency estimation.

Broader impact. V-LynX could broaden access to multimodal reasoning for data-scarce domains and resourceconstrained deployments. Potential positive outcomes include improved assistive systems that jointly leverage vision with audio or geometry, and more modular multimodal pipelines that reduce redundant pretraining and associated energy costs. In addition, expanding the token-interface methodology offers a new lens through which to analyze the modality gap (Liang et al., 2022). By defining a measurable interface gap based on geometric statistics and functional attention divergence, researchers could perform principled diagnostics of multimodal adaptation. This framework provides a robust foundation for identifying failure modes, optimizing reference set selection, and supporting modelfairness by quantifying representation variations across domains and demographic factors.

![](images/a3f1283aa5f45e8a79f870bc11f684141f64af2cfbbfd6eee0f5abaee2ae8555.jpg)

<details>
<summary>natural_image</summary>

Interior view of a hotel room with a white bed and two pillows, no visible text or symbols
</details>

RGB input

![](images/a4dd8300df6782778b749ab1ea92f1b7f0eff87ed3dcf2faa056b7561714432e.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a bedroom with heat signatures indicating temperature distribution (no text or symbols visible)
</details>

Attention

![](images/d2edacff2b6c39ab33442a2be9e3cc9d9f8d19db7c8096acf525f6074739a78a.jpg)

<details>
<summary>natural_image</summary>

Abstract blue gradient background with no text, symbols, or identifiable objects
</details>

3D input

![](images/4c232167f1d74ca2a9676fe22028f3379813a860fd4ff0b62b846cae9207ed59.jpg)

<details>
<summary>natural_image</summary>

Thermal image of a handgun with red heat signature on blue background (no text or symbols)
</details>

Attention w/V-LynX

![](images/33cec193784552ce7c6b66cc9c7030e6fc3ff74395025a47178d42103636610d.jpg)

<details>
<summary>natural_image</summary>

Thermal or heat map image showing heat distribution across a dark blue background with no visible text or symbols
</details>

Attention w/o V-LynX

(a) Question: What is placed up another white pillow?   
![](images/70e22119f5ee1f4689c1bfd2069c2e87e5f4538d0ee14365511cba4d52d366a5.jpg)

<details>
<summary>natural_image</summary>

Interior view of a cluttered office desk with computer, keyboard, and chair (no visible text or symbols)
</details>

RGB input

![](images/fff7ba977c1d8524381819f4de2cc1b838d54cf8e8e6d16bf8d104b109e68b1f.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view of a person operating at a desk with a computer monitor in the background (no visible text or symbols)
</details>

Attention

![](images/f15595bdf954330c346ae9ea2b56f5ce14dd8c0cd5e639f6ae73e07e491184ef.jpg)

<details>
<summary>natural_image</summary>

Interior view of a dimly lit room with blue lighting and furniture (no visible text or symbols)
</details>

3D input

![](images/69ad5e0f704f3c0f6eab10231a7963d86e89fae95d4c875e56ce4d7ca616560a.jpg)

<details>
<summary>natural_image</summary>

Thermal or heat map image showing heat distribution with red hotspots and blue cooler areas (no text or symbols)
</details>

Attention w/V-LynX

![](images/3dd609c0f1c31516d815d98c7582b61d2e561ff8671f468627adbb9fb5d8e41b.jpg)

<details>
<summary>natural_image</summary>

Thermal imaging view showing heat distribution with blue and yellow gradients, no text or symbols visible
</details>

Attention w/o V-LynX   
(b) Question: Where is the monitor with a dark screen located?   
Figure 4. Attention visualization on ScanQA. We depict the RGB inputs and the corresponding attention maps. For the 3D inputs, we provide the attention maps with and without our V-LynX.

# Impact Statement

This paper presents work whose goal is to advance multimodal machine learning by making pretrained Video LLMs more transferable to new modalities under unimodal supervision. The approach may reduce computational cost and data constraints for modality expansion, but it may also increase the potential for privacy-invasive deployments and uneven reliability across settings. We anticipate that responsible use will require careful data governance, evaluation under distribution shifts, and safeguards for sensitive audio and first-person data.

# Acknowledgements

This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (RS-2025-16065706) and (RS-2025- 02216328).

# References

Akbari, H., Yuan, L., Qian, R., Chuang, W.-H., Chang, S.-F., Cui, Y., and Gong, B. Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text. In NeurIPS, 2021.   
Alamri, H., Cartillier, V., Das, A., Wang, J., Cherian, A., Essa, I., Batra, D., Marks, T. K., Hori, C., Anderson, P., et al. Audio visual scene-aware dialog. In CVPR, 2019.   
Azuma, D., Miyanishi, T., Kurita, S., and Kawanabe, M. Scanqa: 3d question answering for spatial scene understanding. In CVPR, 2022.   
Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.   
Bertasius, G., Wang, H., and Torresani, L. Is space-time attention all you need for video understanding? In ICML, 2021.   
Chen, S., Li, H., Wang, Q., Zhao, Z., Sun, M., Zhu, X., and Liu, J. Vast: A vision-audio-subtitle-text omni-modality foundation model and dataset. In NeurIPS, 2023a.   
Chen, S., Wu, Y., Wang, C., Liu, S., Tompkins, D., Chen, Z., Che, W., Yu, X., and Wei, F. Beats: Audio pre-training with acoustic tokenizers. In ICML, 2023b.   
Chen, S., Chen, X., Zhang, C., Li, M., Yu, G., Fei, H., Zhu, H., Fan, J., and Chen, T. Ll3da: Visual interactive

instruction tuning for omni-3d understanding reasoning and planning. In CVPR, 2024a.   
Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024b.   
Cheng, Z., Leng, S., Zhang, H., Xin, Y., Li, X., Chen, G., Zhu, Y., Zhang, W., Luo, Z., Zhao, D., and Bing, L. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024.   
Chowdhury, S., Nag, S., Dasgupta, S., Chen, J., Elhoseiny, M., Gao, R., and Manocha, D. Meerkat: Audio-visual large language model for grounding in space and time. In ECCV, 2024.   
Dai, A., Chang, A. X., Savva, M., Halber, M., Funkhouser, T., and Nießner, M. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017.   
Feichtenhofer, C., Fan, H., Malik, J., and He, K. Slowfast networks for video recognition. In ICCV, 2019.   
Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In CVPR, 2025a.   
Fu, R., Liu, J., Chen, X., Nie, Y., and Xiong, W. Scene-llm: Extending language model for 3d visual understanding and reasoning. In WACV, 2025b.   
Girdhar, R., Singh, M., Ravi, N., Van Der Maaten, L., Joulin, A., and Misra, I. Omnivore: A single model for many visual modalities. In CVPR, 2022.   
Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K. V., Joulin, A., and Misra, I. Imagebind: One embedding space to bind them all. In CVPR, 2023.   
Grauman, K., Westbury, A., Torresani, L., Kitani, K., Malik, J., Afouras, T., Ashutosh, K., Baiyya, V., Bansal, S., Boote, B., et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In CVPR, 2024.   
Gretton, A., Borgwardt, K. M., Rasch, M. J., Scholkopf, B., ¨ and Smola, A. A kernel two-sample test. JMLR, 13(1): 723–773, 2012.   
Hong, Y., Zhen, H., Chen, P., Zheng, S., Du, Y., Chen, Z., and Gan, C. 3d-llm: Injecting the 3d world into large language models. In NeurIPS, 2023.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. In ICLR, 2022.   
Huang, J., Yong, S., Ma, X., Linghu, X., Li, P., Wang, Y., Li, Q., Zhu, S.-C., Jia, B., and Huang, S. An embodied generalist agent in 3d world. In ICML, 2024.   
Huh, M., Cheung, B., Wang, T., and Isola, P. Position: The platonic representation hypothesis. In ICML, 2024.   
Lester, B., Al-Rfou, R., and Constant, N. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021.   
Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., and Li, C. LLaVAonevision: Easy visual task transfer. TMLR, 2025a.   
Li, G., Wei, Y., Tian, Y., Xu, C., Wen, J.-R., and Hu, D. Learning to answer questions in dynamic audio-visual scenarios. In CVPR, 2022.   
Li, G., Hou, W., and Hu, D. Progressive spatiotemporal perception for audio-visual question answering. In ACM MM, 2023a.   
Li, K., He, Y., Wang, Y., Li, Y., Wang, W., Luo, P., Wang, Y., Wang, L., and Qiao, Y. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023b.   
Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Liu, Y., Wang, Z., Xu, J., Chen, G., Luo, P., Wang, L., and Qiao, Y. Mvbench: A comprehensive multi-modal video understanding benchmark. In CVPR, 2024.   
Li, W., Tang, R., Li, C., Zhang, C., Vulic, I., and Søgaard, A. Lost in embeddings: Information loss in vision-language models. In EMNLP Findings, 2025b.   
Li, X. L. and Liang, P. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021.   
Liang, V. W., Zhang, Y., Kwon, Y., Yeung, S., and Zou, J. Y. Mind the gap: Understanding the modality gap in multimodal contrastive representation learning. In NeurIPS, 2022.   
Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., and Yuan, L. Video-llava: Learning united visual representation by alignment before projection. In EMNLP, 2024.   
Liu, Z., Li, Y., Nguyen, K. D., Zhong, Y., and Li, Y. Pave: Patching and adapting video large language models. In CVPR, 2025.   
Loshchilov, I. and Hutter, F. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In ICLR, 2019.   
Ma, X., Yong, S., Zheng, Z., Li, Q., Liang, Y., Zhu, S.-C., and Huang, S. Sqa3d: Situated question answering in 3d scenes. In ICLR, 2023.   
Maaz, M., Rasheed, H., Khan, S., and Khan, F. Videochatgpt: Towards detailed video understanding via large vision and language models. In ACL, 2024.   
Park, J., Lee, J., and Sohn, K. Dual-path adaptation from image to video transformers. In CVPR, 2023.   
Park, J., Lee, J., and Sohn, K. Bootstrap your own views: Masked ego-exo modeling for fine-grained view-invariant video representations. In CVPR, 2025.   
Pham, H.-A., Le, T. M., Le, V., Phuong, T. M., and Tran, T. Video dialog as conversation about objects living in space-time. In ECCV, 2022.   
Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS, 2023.   
Su, Y., Lan, T., Li, H., Xu, J., Wang, Y., and Cai, D. Pandagpt: One model to instruction-follow them all. In Proceedings of the 1st Workshop on Taming Large Language Models: Controllability in the era of Interactive Assistants!, 2023.   
Sun, B. and Saenko, K. Deep coral: Correlation alignment for deep domain adaptation. In ECCV, 2016.   
Sun, G., Yu, W., Tang, C., Chen, X., Tan, T., Li, W., Lu, L., Ma, Z., Wang, Y., and Zhang, C. video-salmonn: Speech-enhanced audio-visual large language models. arXiv preprint arXiv:2406.15704, 2024.   
Tang, C., Yu, W., Sun, G., Chen, X., Tan, T., Li, W., Lu, L., Ma, Z., and Zhang, C. Salmonn: Towards generic hearing abilities for large language models. In ICLR, 2024.   
Tang, C., Li, Y., Yang, Y., Zhuang, J., Sun, G., Li, W., Ma, Z., and Zhang, C. video-salmonn 2: Captioningenhanced audio-visual large language models. arXiv preprint arXiv:2506.15220, 2025.   
Team, Q. et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.   
Wang, Y., Li, K., Li, X., Yu, J., He, Y., Chen, G., Pei, B., Zheng, R., Wang, Z., Shi, Y., et al. Internvideo2: Scaling foundation models for multimodal video understanding. In ECCV, 2024.

Weng, Y., Han, M., He, H., Chang, X., and Zhuang, B. Longvlm: Efficient long video understanding via large language models. In ECCV, 2024.   
Xu, M., Gao, M., Gan, Z., Chen, H.-Y., Lai, Z., Gang, H., Kang, K., and Dehghan, A. Slowfast-llava: A strong training-free baseline for video large language models. arXiv preprint arXiv:2407.15841, 2024a.   
Xu, R., Wang, X., Wang, T., Chen, Y., Pang, J., and Lin, D. Pointllm: Empowering large language models to understand point clouds. In ECCV, 2024b.   
Yang, P., Wang, X., Duan, X., Chen, H., Hou, R., Jin, C., and Zhu, W. Avqa: A dataset for audio-visual question answering on videos. In ACM MM, 2022.   
Yang, Y., Zhuang, J., Sun, G., Tang, C., Li, Y., Li, P., Jiang, Y., Li, W., Ma, Z., and Zhang, C. Audio-centric video understanding benchmark without text shortcut. In EMNLP, 2025.   
Ye, Q., Yu, Z., Shao, R., Xie, X., Torr, P., and Cao, X. Cat: Enhancing multimodal large language model to answer questions in dynamic audio-visual scenarios. In ECCV, 2024.   
Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In ICCV, 2023.   
Zhang, H., Li, X., and Bing, L. Video-llama: An instructiontuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.   
Zhang, Y., Wu, J., Li, W., Li, B., Ma, Z., Liu, Z., and Li, C. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.   
Zhou, D.-W., Zhang, Y., Wang, Y., Ning, J., Ye, H.-J., Zhan, D.-C., and Liu, Z. Learning without forgetting for visionlanguage models. IEEE TPAMI, pp. 4489–4504, 2025a.   
Zhou, J., Shu, Y., Zhao, B., Wu, B., Liang, Z., Xiao, S., Qin, M., Yang, X., Xiong, Y., Zhang, B., et al. Mlvu: Benchmarking multi-task long video understanding. In CVPR, 2025b.   
Zhu, B., Lin, B., Ning, M., Yan, Y., Cui, J., Wang, H., Pang, Y., Jiang, W., Zhang, J., Li, Z., et al. Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. arXiv preprint arXiv:2310.01852, 2023.   
Zhu, C., Wang, T., Zhang, W., Pang, J., and Liu, X. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. In ICCV, 2025.

# A. Implementation Details

# A.1. Algorithm

Overall procedure of our V-LynX is described in Algorithm 1.

Algorithm 1 V-LynX   
Require: Pre-trained video LLM: Vision encoder $g_{\psi}$ , projector $p_{\theta}$ , LLM $f_{\phi}$ .
Require: Unlabeled videos V; unlabeled modality data M; instruction data D.
Require: Weights $\beta$ .
Ensure: Vision encoder LoRA $\Delta\psi$ ; LLM LoRA $\Delta\phi$ .

1: Reference extraction
2: for each layer l in encoder do
3: $\bar{K}_{v}^{(l)} \leftarrow \mathbb{E}_{\mathbf{X}_{v} \sim \mathcal{V}} \left[ K^{(l)}(\mathbf{X}_{v}) \right]$ 4: $\bar{V}_{v}^{(l)} \leftarrow \mathbb{E}_{\mathbf{X}_{v} \sim \mathcal{V}} \left[ V^{(l)}(\mathbf{X}_{v}) \right]$ 5: end for
6: $\mathbf{Z}_{v} \leftarrow p_{\theta}(g_{\psi}(\mathbf{X}_{v}))$ 7: $\mu_{v} \leftarrow \mathbb{E}_{\mathbf{X}_{v} \sim \mathcal{V}}[\mathbf{Z}_{v}], \quad \sigma_{v}^{2} \leftarrow \mathbb{E}_{\mathbf{X}_{v} \sim \mathcal{V}} \left[ (\mathbf{Z}_{v} - \mu_{v})^{2} \right]$ 8: Stage 1: Unimodal training (optimize $\Delta\psi$ )
9: Freeze $p_{\theta}$ and $f_{\phi}$ ; insert LoRA into $g_{\psi}$ to obtain $g_{\psi + \Delta\psi}$ 10: repeat
11: Sample $X_{m} \sim M$ 12: Run $g_{\psi + \Delta\psi}(\mathbf{X}_{m})$ and collect $\{Q_{m}^{(l)}, K_{m}^{(l)}, V_{m}^{(l)}\}_{(l)}$ 13: $L_{attn} \leftarrow 0$ 14: for each layer l in encoder do
15: $O_{m}^{(l)} \leftarrow \text{Attn}\left(Q_{m}^{(l)}, K_{m}^{(l)}, V_{m}^{(l)}\right)$ 16: $\tilde{O}_{m}^{(l)} \leftarrow \text{Attn}\left(Q_{m}^{(l)}, \bar{K}_{v}^{(l)}, \bar{V}_{v}^{(l)}\right)$ 17: $L_{attn} \leftarrow L_{attn} + ||O_{m}^{(l)} - \tilde{O}_{m}^{(l)}||_{1}$ 18: end for
19: $Z_{m} \leftarrow p_{\theta}(g_{\psi + \Delta\psi}(\mathbf{X}_{m}))$ 20: $\mu_{m} \leftarrow E[Z_{m}], \quad \sigma_{m}^{2} \leftarrow E[(Z_{m} - \mu_{m})^{2}]$ 21: $L_{stat} \leftarrow \| \mu_{m} - \mu_{v} \|_{2} + \| \sigma_{m}^{2} - \sigma_{v}^{2} \|_{2}$ 22: $L_{V-LynX} \leftarrow L_{attn} + \beta L_{stat}$ 23: Update $\Delta\psi$ by minimizing $L_{V-LynX}$ 24: until convergence
25: Stage 2: Instruction tuning (optimize $\Delta\phi$ )
26: Freeze $g_{\psi + \Delta\psi}$ and $p_{\theta}$ ; insert LoRA into $f_{\phi}$ to obtain $f_{\phi-1}$ .
27: repeat
28: Sample $(\mathbf{Q}, X_{m}, A) \sim D$ (and optional $X_{v}$ if available)
29: $Z_{m} \leftarrow p_{\theta}(g_{\psi + \Delta\psi}(X_{m})) \quad (\text{and } Z_{v} \leftarrow p_{\theta}(g_{\psi}(X_{v})) \text{ if } Q_{v} = Q_{w}(x))^{\frac{n}{2}})$ 30: $L_{sft} \leftarrow -\sum_{n=1}^{N} log P(a_n | a < n, Q, Z_v, Z_m)$ 31: Update $\Delta\phi$ by minimizing $L_{sft}$ 32: until convergence
33: return $\Delta\psi, \Delta\phi$

# A.2. Processing data

We transform new modality data to enable the pretrained vision encoder to process them accordingly. We provide examples for each modality data in Figure A1.

![](images/ff1268722fce4ec47393091e46f6957ac491172389db28e230c12bf66c97b3cb.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern auditorium with two speakers at podiums and a large display panel showing text sheets (no readable signage or symbols)
</details>

(a) Audio-visual

![](images/5e735e084c0365f008abe14d914545a53a385db638abcc4f6bb1d450bb29301e.jpg)

<details>
<summary>natural_image</summary>

Four-panel image showing a table, chairs, and a blue-lit interior scene with furniture (no visible text or symbols)
</details>

(b) 3D

![](images/5a83e29fc87ed238a81278b4a3634fce7d533029c6c90b02e886e31488295cbc.jpg)

<details>
<summary>natural_image</summary>

Six-panel collage showing a person seated at desks in an office, with no visible text or symbols.
</details>

(c) High frame rate videos

![](images/728e4c7d82115b3261f61e00fcc31f3f2654c40ec56d5fd4cf0ec0f5fd0e3f8d.jpg)  
(d) Ego-Exo   
Figure A1. Examples of input transformation for each new modality. (a) From a given video, we sample an audio signal and transform it into a normalized log-mel spectrogram; (b) Given a depth map, we convert it into a 3-channel disparity map; (c) We rescale and stack multiple frames to obtain a single frame. While the transformed frame has the size of the original frame, we depict it with a scaled-up size for visibility; (d) Egocentric videos are fed into the model without any transformation.

AnswerAnswerTable B1. Embedding analysis on the LLMs’ input space. We present (1) the averaged pairwise cosine distance, (2) the L2 norm of the Projector ProjectorAttention Response Alignment Projector Large Language Modelmodality-wise mean embedding, and (3) a scale-invariant modality gap (Liang et al., 2022) between frame and vocabulary embeddings. 

<table><tr><td rowspan="2">Model</td><td colspan="3">Cosine Distance</td><td colspan="2"> $\ell-2$  Norm</td><td rowspan="2"> $\Delta_{\text{gap}}$ .</td></tr><tr><td>Vocab.-Vocab.</td><td>Frame-Frame</td><td>Vocab.-Frame</td><td>Vocab.</td><td>Frame</td></tr><tr><td>LLaVA-OV-0.5B (Li et al., 2025a)</td><td>0.71</td><td>0.10</td><td>0.96</td><td>0.19</td><td>26.30</td><td>1.0081</td></tr><tr><td>LLaVA-OV-7B (Li et al., 2025a)</td><td>0.78</td><td>0.11</td><td>0.99</td><td>0.10</td><td>45.42</td><td>0.9499</td></tr><tr><td>Qwen2.5-VL-3B (Bai et al., 2025)</td><td>0.79</td><td>0.35</td><td>1.02</td><td>0.47</td><td>31.65</td><td>0.9539</td></tr><tr><td>InternVL-2.5-4B (Chen et al., 2024b)</td><td>0.92</td><td>0.11</td><td>1.01</td><td>0.29</td><td>41.71</td><td>0.9930</td></tr></table>

# Reference videos Reference videos ference videos A.3. Training details

) Interface guidance (b) Interface alignment with unpaired data (c) Instruction tuning Interface guidance (b) Interface alignment with unpaired data (c) Instruction tuningterface guidance (b) Interface alignment with unpaired data (c) Instruction tuningWe optimize all models with AdamW (Loshchilov & Hutter, 2019). For both interface alignment and instruction tuning, we apply a linear warm-up over the first 3% of iterations and use cosine annealing for learning rate decay (Loshchilov & Hutter, 2016). We set the regularization parameter β to 0.01. All experiments are conducted on two NVIDIA RTX PRO 6000 Blackwell 96GB GPUs.

Audio-visual QA. For interface alignment, we train V-LynX for 10 epochs with batch size 8 across all audio-visual QA benchmarks, followed by instruction tuning for 1 epoch on AVSD and 2 epochs on AVQA and MUSIC-AVQA. For V-LynX-0.5B, the base learning rates are 2e-5 for interface alignment and 1e-4 for instruction tuning. For V-LynX-7B, we use 5e-4 for interface alignment and 1e-4 for instruction tuning.

3D QA. ScanQA and SQA3D provide 562 and 518 training videos, respectively, with 517 videos overlapping. We therefore train a single set of vision-encoder LoRAs using the ScanNet training split, and then perform instruction tuning of LLM LoRAs for 1 epoch on ScanQA and 2 epochs on SQA3D. We use the same learning rate settings as in the audio-visual QA experiments.

Enhanced video QA. We perform interface alignment for 5 epochs and instruction tuning for 2 epochs on LLaVA-Video-178K, and evaluate on VideoMME, MVBench, and MLVU without training on the target benchmarks. For interface alignment, we set the base learning rate to 1e-5 for V-LynX-0.5B and 5e-5 for V-LynX-7B, while keeping the remaining schedule unchanged.

Multi-view video understanding. We follow the same training protocol as audio-visual QA, performing 10 epochs of interface alignment and then instruction tuning for 1 epoch on AVSD and 2 epochs on AVQA and MUSIC-AVQA. We use the learning rate settings from the enhanced video QA configuration.

# B. Additional Analysis

# B.1. Analysis on token interface

Existence of token interface. We provide additional analysis to present that token interfaces are a common phenomenon in Video LLM. Specifically, we quantitatively analyze the LLM’s input space using three statistics: (1) the averaged pairwise cosine distance, (2) the ℓ-2 norm of the modality-wise mean embedding, and (3) a scale-invariant modality gap (Liang et al., 2022) between frame and vocabulary embeddings across four Video LLMs, including LLaVA-OV-0.5B, -7B (Li et al., 2025a), Qwen2.5-VL-3B (Bai et al., 2025), and InternVL2.5-4B (Chen et al., 2024b), as shown in Table B1. Across all backbones, projected frame embeddings show much smaller pairwise cosine distances than vocabulary embeddings, indicating that visual tokens form a more compact geometric regime. At the same time, the cosine distance between vocabulary and frame embeddings is close to one, suggesting that the two embedding groups are nearly orthogonal rather than intermingled. The consistently large scale-invariant modality gap further indicates that this separation cannot be explained by simple norm differences. Importantly, this separated region is not an invalid out-of-distribution space, but an operationally compatible token interface that can be processed by the LLMs. This interpretation is supported by our empirical results: new modalities can be aligned to this region without paired cross-modal supervision during the interface alignment stage, while removing interface alignment substantially degrades performance, as shown in Table 5.

Table B2. Mean and variance analysis at 26-layers of the vision tower in LLaVA-OV-0.5B (Li et al., 2025a). 

<table><tr><td rowspan="2"></td><td rowspan="2">Entity</td><td colspan="13">Layers</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td></tr><tr><td rowspan="3">Key</td><td> $||K_v^{(l)}||^2$ </td><td>0.675</td><td>0.615</td><td>1.243</td><td>1.373</td><td>0.918</td><td>1.210</td><td>0.930</td><td>1.097</td><td>0.951</td><td>0.872</td><td>0.804</td><td>0.746</td><td>0.725</td></tr><tr><td>trace( $\Sigma_K^{(l)}$ )</td><td>0.002</td><td>0.003</td><td>0.002</td><td>0.002</td><td>0.003</td><td>0.003</td><td>0.004</td><td>0.003</td><td>0.004</td><td>0.005</td><td>0.004</td><td>0.006</td><td>0.006</td></tr><tr><td> $R_K$ </td><td>369.0</td><td>211.7</td><td>816.0</td><td>786.2</td><td>267.0</td><td>427.1</td><td>263.6</td><td>329.8</td><td>219.2</td><td>191.2</td><td>181.6</td><td>131.3</td><td>114.9</td></tr><tr><td rowspan="3">Value</td><td> $||V_v^{(l)}||^2$ </td><td>0.015</td><td>0.028</td><td>0.056</td><td>0.041</td><td>0.164</td><td>0.167</td><td>0.108</td><td>0.056</td><td>0.074</td><td>0.074</td><td>0.054</td><td>0.065</td><td>0.052</td></tr><tr><td>trace( $\Sigma_V^{(l)}$ )</td><td> $2\times 10^{-5}$ </td><td> $3\times 10^{-4}$ </td><td> $3\times 10^{-4}$ </td><td> $5\times 10^{-4}$ </td><td> $1\times 10^{-3}$ </td><td> $1\times 10^{-3}$ </td><td> $2\times 10^{-3}$ </td><td> $1\times 10^{-3}$ </td><td> $3\times 10^{-3}$ </td><td> $2\times 10^{-3}$ </td><td> $2\times 10^{-3}$ </td><td> $2\times 10^{-3}$ </td><td> $2\times 10^{-3}$ </td></tr><tr><td> $R_V$ </td><td>783.0</td><td>92.7</td><td>175.2</td><td>81.7</td><td>136.6</td><td>132.9</td><td>70.7</td><td>43.3</td><td>26.1</td><td>43.5</td><td>31.8</td><td>30.5</td><td>22.1</td></tr><tr><td rowspan="2"></td><td rowspan="2">Entity</td><td colspan="13">Layers</td></tr><tr><td>14</td><td>15</td><td>16</td><td>17</td><td>18</td><td>19</td><td>20</td><td>21</td><td>22</td><td>23</td><td>24</td><td>25</td><td>26</td></tr><tr><td rowspan="3">Key</td><td> $||K_v^{(l)}||^2$ </td><td>0.711</td><td>0.685</td><td>0.646</td><td>0.721</td><td>0.689</td><td>0.633</td><td>0.596</td><td>0.621</td><td>0.673</td><td>0.647</td><td>0.657</td><td>0.651</td><td>0.608</td></tr><tr><td>trace( $\Sigma_K^{(l)}$ )</td><td>0.007</td><td>0.007</td><td>0.008</td><td>0.008</td><td>0.008</td><td>0.008</td><td>0.009</td><td>0.010</td><td>0.010</td><td>0.010</td><td>0.010</td><td>0.010</td><td>0.008</td></tr><tr><td> $R_K$ </td><td>108.2</td><td>95.1</td><td>85.0</td><td>92.3</td><td>83.4</td><td>75.3</td><td>68.1</td><td>64.9</td><td>67.1</td><td>63.2</td><td>63.8</td><td>63.0</td><td>74.8</td></tr><tr><td rowspan="3">Value</td><td> $||V_v^{(l)}||^2$ </td><td>0.053</td><td>0.045</td><td>0.035</td><td>0.035</td><td>0.040</td><td>0.051</td><td>0.048</td><td>0.052</td><td>0.093</td><td>0.109</td><td>0.158</td><td>0.170</td><td>0.293</td></tr><tr><td>trace( $\Sigma_V^{(l)}$ )</td><td> $3\times 10^{-3}$ </td><td> $4\times 10^{-3}$ </td><td> $3\times 10^{-3}$ </td><td> $4\times 10^{-3}$ </td><td> $4\times 10^{-3}$ </td><td> $6\times 10^{-3}$ </td><td> $6\times 10^{-3}$ </td><td> $6\times 10^{-3}$ </td><td> $7\times 10^{-3}$ </td><td> $8\times 10^{-3}$ </td><td> $1\times 10^{-2}$ </td><td> $1\times 10^{-2}$ </td><td> $1\times 10^{-2}$ </td></tr><tr><td> $R_V$ </td><td>21.3</td><td>12.4</td><td>12.0</td><td>9.2</td><td>9.3</td><td>8.0</td><td>8.3</td><td>8.3</td><td>13.6</td><td>13.3</td><td>15.4</td><td>16.3</td><td>25.2</td></tr></table>

![](images/6c25c17517f199dc787a08b5e5e6a9c06f562713c9c7f9dca7789ede113c8d77.jpg)

![](images/aeb29c502a613c12d0d5d04291474ec0516f730a72b300260c2bf1471a047b78.jpg)  
Figure B2. Mean, variance, and corresponding dominance score for the Key and Value of the interface guidance.

Video-derived interface guidance. For interface alignment, we first estimate reference statistics by extracting averaged Key and Value embeddings at each attention layer from V, as shown in Equation (2). While we demonstrate that they successfully represent the behavior of the pretrained Video LLM, they are potentially less representative since V is gathered from the six benchmarks. We further measure the variance of the Key and Value across the reference videos, and compare with the averaged Key and Value embeddings to verify the stability of the interface guidance.

Let $\gamma _ { s }$ is the s-th benchmark in V. We first obtain the mean Key and Value embeddings for each benchmark at each layer:

$$
K _ {s} ^ {(l)} = \mathbb {E} _ {\mathbf {X} _ {v} \sim \nu_ {s}} K _ {\psi} ^ {(l)} (\mathbf {X} _ {v}), \quad V _ {s} ^ {(l)} = \mathbb {E} _ {\mathbf {X} _ {v} \sim \nu_ {s}} V _ {\psi} ^ {(l)} (\mathbf {X} _ {v}). \tag {11}
$$

With the mean Key and Value embeddings, we can obtain the variance matrix $\Sigma _ { K }$ and $\Sigma _ { V } { : }$ :

$$
\Sigma_ {K} ^ {(l)} = \frac {1}{S - 1} \sum_ {s} (K _ {s} ^ {(l)} - K _ {v} ^ {(l)}) (K _ {s} ^ {(l)} - K _ {v} ^ {(l)}) ^ {\top}, \quad \Sigma_ {V} ^ {(l)} = \frac {1}{S - 1} \sum_ {s} (V _ {s} ^ {(l)} - V _ {v} ^ {(l)}) (V _ {s} ^ {(l)} - V _ {v} ^ {(l)}) ^ {\top}, \tag {12}
$$

where $K _ { v } ^ { ( l ) } , V _ { v } ^ { ( l ) }$ are the reference Key and Value in Equation (2) and S is the number of benchmarks. We define a

Table B3. Performance comparisons with different backbones. We report top-1 Exact Match (EM@1) and (refined EM@1) on SQA3D. 

<table><tr><td>Method</td><td>EM@1</td><td>ΔParams.</td></tr><tr><td colspan="3">With LLaVA-OV</td></tr><tr><td>LLaVA-OV-0.5B (Li et al., 2025a)</td><td>0.8 (43.0)</td><td>-</td></tr><tr><td>LLaVA-OV-7B (Li et al., 2025a)</td><td>8.3</td><td>-</td></tr><tr><td>V-LynX-0.5B (Ours)</td><td>52.2 (54.2)</td><td>68.7M</td></tr><tr><td>V-LynX-7B (Ours)</td><td>60.5 (62.6)</td><td>195.0M</td></tr><tr><td colspan="3">With Qwen2.5-VL</td></tr><tr><td>Qwen2.5-VL-3B (Bai et al., 2025)</td><td>15.1</td><td>-</td></tr><tr><td>V-LynX-3B (Ours)</td><td>59.7 (60.0)</td><td>165.5M</td></tr><tr><td colspan="3">With InternVL-2.5</td></tr><tr><td>InternVL-2.5-4B (Chen et al., 2024b)</td><td>44.0 (50.6)</td><td>-</td></tr><tr><td>V-LynX-4B (Ours)</td><td>61.1 (63.5)</td><td>144.9M</td></tr></table>

Table B4. Performance comparisons on AV-Human of AVUT (Yang et al., 2025). We report the accuracy (Acc.). 

<table><tr><td>Method</td><td>Acc. (%)</td></tr><tr><td colspan="2">Visual MLLMs</td></tr><tr><td>GPT-4o</td><td>56.62</td></tr><tr><td>Qwen2-VL-7B</td><td>58.38</td></tr><tr><td>LLaVA-Video-7B</td><td>56.52</td></tr><tr><td>InternVL2-8B</td><td>45.9</td></tr><tr><td>VILA-1.5-8B</td><td>44.48</td></tr><tr><td>VideoLLaVA-7B</td><td>33.14</td></tr><tr><td colspan="2">Audio MLLMs</td></tr><tr><td>SALMONN-13B</td><td>36.48</td></tr><tr><td colspan="2">Audio-visual MLLMs</td></tr><tr><td>Gemini 1.5 Pro</td><td>78.34</td></tr><tr><td>VideoLLaMA2-7B</td><td>44.90</td></tr><tr><td>video-SALMONN-13B</td><td>38.33</td></tr><tr><td>PandaGPT-13B</td><td>25.38</td></tr><tr><td>V-LynX-0.5B</td><td>46.91</td></tr></table>

dominance score R as the ratio of the magnitude of the reference Key and Value to the total variance:

$$
R _ {K} ^ {(l)} = \frac {| | K _ {v} ^ {(l)} | | ^ {2}}{\operatorname{trace} \left(\Sigma_ {K} ^ {(l)}\right)}, \quad R _ {V} ^ {(l)} = \frac {| | V _ {v} ^ {(l)} | | ^ {2}}{\operatorname{trace} \left(\Sigma_ {V} ^ {(l)}\right)}. \tag {13}
$$

In Table B2 and Figure B2, we depict the magnitude of the reference Key and Value embeddings, the total variance of the Key and Value embeddings across benchmarks, and the corresponding dominance scores derived from each layer of the vision tower in LLaVA-OV-0.5B (Li et al., 2025a). The result demonstrates that the global reference is stable: when averaged across layers, the variance is 0.006 for the Key and 0.004 for the Value, while the corresponding means are 0.80 and 0.08, respectively. Consequently, the dominance scores of the Key and Value are both much higher than 1, indicating that the reference Key and Value statistics are tightly concentrated across videos rather than being dominated by large sample-to-sample fluctuations.

# B.2. Additional experiments

Performance with different backbones. To further demonstrate the scalability of V-LynX in backbones, we train V-LynX with Qwen2.5-VL-3B (Bai et al., 2025) and InternVL-2.5-4B (Chen et al., 2024b), and evaluate them on SQA3D. As shown in Table B3, V-LynX consistently improves all baselines: V-LynX-0.5B and V-LynX-7B achieve 52.2 and 60.5 EM@1 with LLaVA-OV, while V-LynX-3B improves Qwen2.5-VL-3B from 15.1 to 59.7 EM@1. InternVL-2.5-4B already provides a strong baseline of 44.0 EM@1, partly because ScanQA was included in its fine-tuning data (Chen et al., 2024b). Nevertheless, V-LynX-4B further improves it to 61.1 EM@1 and 63.5 refined EM@1 with only 144.9M additional parameters. These results demonstrate that the proposed interface alignment generalizes beyond a specific Video LLM backbone.

![](images/2944a9685adfd2130335e942c0635b7fc1439aa3a77cf975ef36ddc5ff41a0d8.jpg)

<details>
<summary>natural_image</summary>

Collage of dark-framed photos showing interior scenes with no visible text or symbols
</details>

Question: Where is the wall-mounted shelf located?

GT: Above printer / Left to window

Baseline: Above counter

Ours: Above printer

![](images/02cb5845c169f4433aca3cf73dbbecad7548c1ed4080e55172b7434d8c7e8014.jpg)

<details>
<summary>natural_image</summary>

Collage of office furniture photos showing room layouts and a chair, no visible text or symbols
</details>

Question: Is the shape of table behind me round,square or rectangular?

GT: Rectangular

Baseline: Square

Ours: Rectangular

(a) 3D QA examples from ScanQA (left) and SQA3D (right)   
![](images/63b441b7a21057e71f37de169aaed82e1cf9b2b916b46dfc192f9d8735b4c25c.jpg)

<details>
<summary>natural_image</summary>

Stack of six black-and-white photos showing a person wearing a T-shirt with 'HUAAN ZHI' and a face mask, arranged in a row (no visible text or symbols)
</details>

Question: Does he speak to the camera or is he silent?

GT: He does not say anything

Baseline: He does not speak to the camera.

Ours: He is silent the whole time

![](images/bc3f9d7b9182720e34250a99f29ba801e8f9b9e6f32652bf2f34d2c0eaefd4dd.jpg)

<details>
<summary>natural_image</summary>

Sequence of black-framed photos showing a person standing in front of a wooden deck, no visible text or symbols
</details>

Question: What did he do in the video?

GT: He was fixing the vacum

Baseline: He vacuumed the floor

Ours: He was fixing the vacuum

(b) Audio-visual examples from AVSD   
![](images/94499d980bfdb0ee83d13dc796dfeb423b84545b6153493a929db2bf500410ca.jpg)

<details>
<summary>natural_image</summary>

Group of people performing on stage with city skyline backdrop (no visible text or symbols)
</details>

Question: What is the first instrument that comes in?

CGT: Piano

Baseline: Flute

COurs: Flute

![](images/f4c626554f8b7b14134b28ba3160b3504735012284768672592dc2538877e948.jpg)

<details>
<summary>text_image</summary>

DEPENDE...
</details>

Question: What is the first instrument that comes in?

GT: Piano

Baseline: Acoustic guitar

Ours: Acoustic guitar

(c) Failure cases from MUSIC-AVQA

Figure B3. Qualitative examples for (a) 3D QA from ScanQA (left) and SQA3D (right), and (b) audio-visual QA from AVSD. We also provide (c) failure cases from MUSIC-AVQA.

Experiment on less visually grounded task. To further examine V-LynX on tasks where a new modality data (i.e., audio) plays a central role, we conduct an additional experiment on AVUT (Yang et al., 2025). AVUT is an audio-centric video understanding benchmark designed to reduce text shortcuts and evaluate both audio content understanding and audio-visual alignment across diverse video domains. It consists of AV-Gemini, a larger Gemini-augmented training split, and AV-Human, a human-annotated evaluation split. Specifically, we train V-LynX on AV-Gemini, and evaluate it on AV-Human.

As shown in Table B4, V-LynX-0.5B achieves 46.91% accuracy on AV-Human. Although this task is less aligned with our video-induced token interface than visually grounded audio-visual QA, V-LynX still outperforms several audio and audio-visual MLLMs, including SALMONN-13B (Tang et al., 2024), VideoLLaMA2-7B (Cheng et al., 2024), video-SALMONN-13B (Sun et al., 2024), and PandaGPT-13B (Su et al., 2023). This result suggests that the proposed interface alignment can transfer audio information to the Video LLM beyond strongly visually grounded settings. At the same time, the remaining gap to Gemini 1.5 Pro and strong visual MLLMs indicates that purely audio-centric reasoning remains challenging when the new modality is routed through a video-induced interface, which is consistent with the limitation discussed in Section C.

# B.3. Qualitative analysis

In Figure B3a, the results show that V-LynX produces more spatially grounded answers than the baseline, which often defaults to course category priors and misses geometric relations (e.g., relative position such as above or behind). For audio-visual QA, our V-LynX yields responses that better reflect subtle action and state cues, as shown in Figure B3b. Notably, even under spurious audio signals (e.g., the vacuuming sound), V-LynX can still reach correct conclusions by appropriately integrating visual evidence with the language query. We also include failure cases from MUSIC-AVQA in Figure B3c, where both the baseline and V-LynX struggle when the target instruments are not given as visual cues (e.g., piano present only as background music). Although similar limitations have been reported in (Liu et al., 2025), we attribute this to an inherent limitation of our V-LynX, which is to align a new modality to the visual token interface.

# C. Limitation

Inherent limitation. Our approach adapts a new modality by aligning it to the video-induced token interface. This design choice bounds what the adapted modality can express to what is representable through the visual interface that the backbone has internalized. In practice, when the target concept is weakly or not at all grounded in visual evidence, alignment to the visual token interface can be insufficient. This behavior is visible in the MUSIC-AVQA failure cases where the target instrument is only present as background audio without a corresponding visual cue, leading both LynX and prior baselines to fail on purely audio-driven discrimination.

Input transformation. A second limitation arises from the modality-to-vision preprocessing that enables the reuse of the frozen vision encoder. While aligning heterogeneous signals into a unified visual manifold ensures seamless integration, it introduces a subtle trade-off between modality-specific granularity and cross-modal compatibility. For high-frame-rate video, the frame stacking strategy (Park et al., 2023) introduces downsampling that attenuates resolution-sensitive cues, which aligns with the observed degradation on fine-grained pose in MVBench. Similarly, depth-to-disparity conversion and audio-to-log-mel transformation may remove information that is useful for downstream reasoning, such as fine geometry, phase, or transient structure, and the model cannot recover what is lost at the interface input.