# CLIP-Guided SAM: Parameter-Efficient Semantic Conditioning for Promptable Segmentation

Shayan Jalilian and Abdul Bais

University of Regina, Regina, SK, Canada sjs949@uregina.ca, Abdul.Bais@uregina.ca

Abstract. Promptable foundation models such as the Segment Anything Model (SAM) produce high-quality masks but remain semantically blind, relying on external prompts to specify categories. Existing vision– language approaches address this limitation by using external prompt coupling, in which a vision–language model generates spatial prompts for SAM as a separate stage.

We propose CLIP-Guided SAM, a parameter-efficient segmentation framework built on internal semantic conditioning. Instead of using semantic signals only to generate prompts, we inject CLIP-derived text, vision, and similarity features directly into SAM’s image encoder via lightweight multi-modal semantic adapters. These adapters condition SAM’s internal feature representations, allowing semantic information to influence mask prediction while preserving SAM’s original promptable interface.

Our framework is designed for low labelled-data settings and applies to both general-domain benchmarks and specialized downstream tasks. It supports two operating modes: Manual mode, for interactive segmentation with both text and spatial prompts, and Semi-Automatic (text-only) mode, for applications that require concept-specific segmentation using only textual input. We show that robustness depends on aligning training with the type of prompts used at inference, making train–test prompt consistency an important design principle.

Through extensive experiments and ablations, we evaluate our method against SAM+PEFT baselines without semantic conditioning, vision– language + SAM pipelines, SAM 3, and strong semi-supervised segmentation methods that rely on large amounts of unlabelled data. Across these settings, CLIP-Guided SAM consistently achieves superior or competitive performance while remaining parameter-efficient in both training and deployment.

Keywords: Promptable Segmentation · Segment Anything · CLIP · Parameter-Efficient Fine-Tuning · Vision-Language Models

# 1 Introduction

Prompt-based segmentation, popularized by the Segment Anything Model (SAM) [15], has significantly expanded the flexibility of visual localization. With minimal spatial input—such as a point, box, or mask—SAM can produce high-quality object masks across diverse scenes. However, SAM is intentionally class-agnostic: it excels at grouping visual structures but does not encode semantic category information. As a result, identifying what to segment remains an external problem. Furthermore, SAM is sometimes adapted via Parameter-Efficient Fine-Tuning (PEFT) methods for specific tasks to enhance performance and address label scarcity or computational constraints.

Vision–language models (VLMs) such as Contrastive Language–Image Pretraining (CLIP) [28] provide complementary strengths, offering semantic representations aligned with natural language but lacking the spatial precision required for dense segmentation. This complementarity has motivated VLM+SAM systems based on external prompt coupling, where a VLM generates spatial prompts—points, boxes, or masks—that are provided to a frozen or lightly adapted SAM.

While effective in some settings, this stage-wise and shallow integration keeps semantic reasoning and spatial segmentation architecturally separate. Semantic information influences SAM only indirectly through sparse or noisy prompts, leaving its internal feature representations semantically uninformed. This limitation is even more pronounced in low-label regimes and text-only scenarios, where supervision is limited and spatial cues are suboptimal. Figure 1 illustrates the difference between existing approaches—such as SAM with PEFT or training-free VLM+SAM pipelines—and our proposed framework, which injects semantic signals directly into SAM’s internal representations.

In this work, we propose CLIP-Guided SAM, a parameter-efficient framework that unifies internal semantic conditioning with external spatial prompting. Rather than relying solely on prompt-level guidance, we inject CLIP-derived text, vision, and similarity features directly into SAM’s image encoder via semantic adapters. This design allows semantic information to influence feature formation during segmentation, while spatial prompts remain complementary for boundary refinement and localization.

A central component of our framework is the joint co-adaptation of SAM and CLIP. Through systematic ablations, we show that effective semantic conditioning requires optimizing CLIP not merely as a prompt generator, but as an internal semantic guide tailored to SAM’s feature space. Freezing CLIP or prefine-tuning it independently beforehand to improve spatial prompt quality leads to inferior performance, whereas end-to-end co-adapting both models enables CLIP to learn to produce semantic signals useful for segmentation and better aligned with SAM’s feature space. We further observe asymmetric sensitivity in tuning budgets: performance saturates quickly with additional SAM backbone tuning, whereas increasing the number of trainable parameters in CLIP produces larger gains.

CLIP-Guided SAM is designed to support multiple deployment scenarios. In an interactive mode, the framework combines text prompts with user-provided point inputs, modelling user-in-the-loop segmentation or annotation workflows. In a text-only (semi-automatic) mode, segmentation is driven solely by class descriptions, without human spatial input. In this setting, spatial prompts serve as model-generated inputs rather than supervision. We therefore introduce a traintest-prompt-aligned training strategy that trains directly on noisy prompts sampled from CLIP similarity maps. This alignment mitigates the train–test mismatch introduced by idealized ground-truth clicks and yields substantially more robust inference under text-only prompting.

![](images/b56057c5c2718c4f5f0e38cd55b1cd5c9054f818124b4bbc479517f7722558c5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph SAM
        A1["Image"] --> B1["SAM"]
        A2["Spatial Prompts"] --> B2["PEFT"]
        B1 --> C1["Output Mask"]
        B2 --> C2["Error"]
    end

    subgraph VL + SAM
        D1["Image"] --> E1["VL Model"]
        E1 --> F1["Spatial prompts"]
        F1 --> G1["SAM"]
        G1 --> H1["Output Mask"]
        G2["Text Prompt"] --> E1
        E1 --> I1["CLIP"]
        I1 --> J1["PEFT"]
        J1 --> K1["Dense prompt"]
        K1 --> L1["Vision, Text, Similarity Features"]
        L1 --> M1["Output Mask"]
    end

    subgraph CLIP-Guided SAM
        N1["Image"] --> O1["CLIP"]
        O1 --> P1["PEFT"]
        P1 --> Q1["Visual/Text Similarity Features"]
        Q1 --> R1["Output Mask"]
        R1 --> S1["Error"]
        S1 --> T1["Text Prompt"]
        T1 --> U1["CLIP"]
        U1 --> V1["PEFT"]
        V1 --> W1["Dense prompt"]
        W1 --> X1["Vision, Text Similarity Features"]
        X1 --> Y1["Output Mask"]
    end

    style SAM fill:#f9f,stroke:#333
    style VL + SAM fill:#ccf,stroke:#333
    style CLIP-Guided SAM fill:#cfc,stroke:#333
```
</details>

Fig. 1: CLIP-Guided SAM Overview. Comparison of (top) SAM with PEFT, (middle) typical VLM+SAM pipelines, and (bottom) our CLIP-Guided SAM. Existing approaches either rely on spatial prompting alone or couple a VLM to SAM externally via prompt generation, leaving SAM’s internal representations semantically uninformed. Our approach introduces internal semantic conditioning by injecting CLIP-derived features directly into SAM’s image encoder through multi-modal adapters, enabling joint SAM–CLIP co-adaptation.

We evaluate CLIP-Guided SAM on general semantic segmentation benchmarks (COCO, ADE20K, PASCAL VOC) and specialized downstream tasks such as camouflaged object detection. Under limited supervision, our method consistently outperforms SAM-based parameter-efficient baselines and achieves competitive performance relative to heavier vision–language systems. With only a few hundred labelled images, our modular co-adaptation framework approaches the zero-shot performance of large-scale models such as SAM 3, while using substantially fewer parameters and enabling efficient tuning on commodity GPUs.

Contributions. In summary, we make the following contributions:

– We introduce a framework for internal semantic conditioning of SAM, shifting VLM+SAM integration from external prompt coupling to joint feature-level co-adaptation.

– We analyze co-adaptation dynamics between SAM and CLIP, showing that adapting the semantic guide is critical, whereas additional tuning of the SAM backbone yields diminishing returns.   
– We propose a train-test prompt alignment strategy for text-only segmentation that improves robustness by matching training and inference prompt distributions.   
– We present a parameter-efficient system applicable to both general and specialized segmentation tasks, supporting interactive (text + spatial prompts) and text-only deployment modes under low-label and modest-compute settings.   
– We demonstrate strong performance on different comparisons and experiments, along with comprehensive ablations, highlighting an efficient and strong alternative to SAM-PEFT baselines, VLM+SAM pipelines, large foundation models, and even strong semi-supervised methods in certain contexts.   
– We demonstrate strong empirical performance across diverse comparisons and ablations, providing an efficient and competitive alternative to SAM-PEFT baselines, VLM+SAM pipelines, large segmentation foundation models, and, in certain regimes, semi-supervised methods.

# 2 Related Works

# 2.1 SAM and Parameter-Efficient Fine-Tuning

SAM is a powerful prompt-based segmentation foundation model, but often requires adaptation for specialized downstream tasks [5]. Because full fine-tuning is costly in low-label regimes, parameter-efficient fine-tuning (PEFT) methods such as adapters [11], LoRA [13], and prompt tuning [19] are widely used.

Prior work adapts SAM via lightweight modules or modified heads, e.g., SAM-Adapter [4], Conv-Meets-LoRA [42], SU-SAM [32], and TS-SAM [40], while others provide light semantic conditioning through text embeddings, e.g., SAM-PTx [14]. In contrast, we keep SAM’s prompt encoder and mask decoder intact and inject CLIP-derived text, vision, and similarity features into the image encoder via semantic adapters, enabling text-prompted semantic conditioning with minimal architectural change.

# 2.2 Vision–Language Models for Semantic Segmentation

Vision–language models such as CLIP [28] align images and text in a shared embedding space and have been widely used for segmentation. Zero-shot and weakly supervised approaches derive activation maps or grouped regions from CLIP representations [3, 6, 10, 26, 30, 34, 43], but often suffer from coarse boundaries due to limited spatial precision [12].

Open-vocabulary segmentation methods instead train large models with language supervision to generalize to unseen classes [7,9,20,23,35,44]. SemiVL [12] is an example that combines limited dense labels with a large unlabelled pool for semi-supervised learning.

CLIPSurgery [22] improves CLIP’s spatial localization by modifying its inference pathway to produce sharper similarity maps, which can be used to sample spatial prompts for SAM, and is a training-free method.

Unlike methods that treat CLIP either as the primary segmentation engine or solely as an external prompt generator, we use CLIP as a provider of semantic priors, whose features are injected directly into SAM’s encoder and jointly optimized with SAM.

# 2.3 Integration of SAM and CLIP for Text-to-Mask Segmentation

Existing VLM+SAM systems generally follow two strategies. The first relies on cascaded pipelines where a VLM converts text prompts into spatial prompts for SAM. Examples include GroundedSAM [29], which uses GroundingDINO [25] to generate box prompts, and CLIP-based methods that derive points or masks from similarity maps [16,17,21,22]. Other approaches use SAM to propose masks, which are then filtered or scored using CLIP [1,38]. In most cases, SAM and the VLM operate as separate stages.

A smaller body of work explores tighter feature-level coupling between SAM and VLMs by injecting language signals directly into SAM’s internal representations [24, 27, 33, 39, 41]. These approaches typically fuse text embeddings or activation maps through additional tokens, modified decoders, or auxiliary modules, often departing from the original SAM architecture or requiring substantial supervision and training, and are sometimes tailored to specific domains.

Our approach bridges these directions. We build on CLIPSurgery-style prompt strategy while also injecting CLIP-derived semantic features directly into SAM’s encoder via adapters, enabling internal semantic conditioning and joint SAM and CLIP co-adaptation with minimal architectural changes.

# 3 Method

We propose CLIP-Guided SAM, which uses CLIP to produce (i) spatial semantic prompts and (ii) feature-level conditioning prompts that are injected into SAM’s image encoder via semantic adapters. We support two usage modes depending on whether point prompts come from CLIP (text-only inference) or from user/GT clicks (interactive setting).

# 3.1 Generating Semantic Prompts with CLIP

Given an image and a class prompt, CLIP produces: (i) patch-level image embeddings $\mathbf { V } \in \mathbb { R } ^ { N \times C _ { c } }$ (after discarding the [CLS] token, where $N = H W )$ , and (ii) a global text embedding $\mathbf { t } \in \mathbb { R } ^ { C _ { c } }$ .

We compute patch-wise cosine similarity:

$$
\mathbf {s} _ {p} = \left\langle \frac {\mathbf {V} _ {p}}{\| \mathbf {V} _ {p} \|}, \frac {\mathbf {t}}{\| \mathbf {t} \|} \right\rangle , \quad p = 1, \dots , N, \tag {1}
$$

![](images/12794a47418d1fb59ed7d3fc9e97881614d1c54a25cf2446036e00f6e743a3bc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Image"] --> B["CLIP Text Encoder"]
    C["Class Name"] --> D["CLIP Image Encoder"]
    B --> E["Text Embedding"]
    D --> F["Image Embedding"]
    E --> G["Similarity Features"]
    F --> G
    G --> H["Min-Max norm Reshape Interpolate"]
    H --> I["Similarity Maps"]
    I --> J["Concat"]
    K["Similarity Maps"] --> L["Mode: Manual or Semi-automatic"]
    M["GT Labels"] --> L
    L --> N["Point Sampling"]
    N --> O["Point Prompts"]
    O --> J
    J --> P["Semantic Prompts"]
    Q["Vision+Similarity Features"] --> H
```
</details>

Fig. 2: Semantic Prompt Generation with CLIP. Shows the process of how we use CLIP to generate our semantic prompts, which include similarity maps (dense mask prompt), vision+similarity features, text features, and point prompts. Point prompts are sampled either from similarity maps or GT labels, depending on the design mode (manual vs semi-automatic)

yielding similarity scores $\mathbf { s } \in \mathbb { R } ^ { N \times 1 }$ .

Reshaping s to $H \times W$ gives a similarity map S. We upsample and min–max normalize S, then threshold it to obtain a binary prompt mask B.

We use these CLIP semantic signals in three ways (Fig. 2):

– Dense prompt: resize B to SAM’s mask-prompt resolution and feed it to the prompt encoder.   
– Point prompts: sample K=5 positive points from B (semi-automatic mode); in manual mode, points are sampled from GT masks.   
– Feature injection: inject the text embedding t and fused vision–similarity features into SAM’s image encoder via semantic adapters.

# 3.2 Two Design Modes

We provide two design modes in our integration framework (Fig. 3):

Semi-automatic (text-only). Points and dense prompts are derived from B during both training and inference, yielding train–test consistency under imperfect CLIP localization.

Manual (interactive). CLIP feature injection remains enabled and similarity maps are still used as dense prompts, but point prompts come from GT/user clicks (oracle) during both training and inference.

# 3.3 Semantic Adapters

We insert semantic adapters in parallel to the MLP of each ViT block (regular adapters remain parallel to attention). See Fig. 4 for an overview of adapter placement and the internal structure of the semantic adapters.

Let $\mathbf { F } _ { \ell } \in \mathbb { R } ^ { H _ { \ell } \times W _ { \ell } \times C _ { \ell } }$ denote the feature map entering the MLP at layer ℓ. CLIP provides patch embeddings $\mathbf { V } \in \mathbb { R } ^ { N \times C _ { c } }$ , a text embedding $\mathbf { t } \in \mathbb { R } ^ { C _ { c } }$ , and similarity scores $\mathbf { s } \in \mathbb { R } ^ { N \times 1 }$ . We fuse vision and similarity via broadcasted addition,

![](images/30f960aee91e2771bd1186452409d0cd6d0c381b64cba1e8a10deed22283e2a5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Image"] --> B["CLIP"]
    C["Text Prompt (Class Name)"] --> D["CLIP Image Encoder"]
    B --> E["Semantic Prompt Generation"]
    D --> E
    E --> F["Image Encoder"]
    F --> G["SAM"]
    G --> H["Mask Decoder"]
    H --> I["Output mask"]
    B --> J["CLIP Text Encoder"]
    D --> K["Attention blocks"]
    D --> L["Similarity Mask Prompts"]
    F --> M["VIT Block"]
    F --> N["Attention"]
    F --> O["Adapter"]
    F --> P["MLP"]
    F --> Q["Semantic Adapter"]
    F --> R["Attention + Similarity embeddings"]
    F --> S["Point Prompts"]
    F --> T["Similarity Mask Prompts"]
    G --> U["Prompt Encoder"]
```
</details>

Fig. 3: Overall integration framework. CLIP extracts text and image embeddings, which produce similarity features and maps via cosine similarity. Points prompts are derived either from similarity masks or GT labels, depending on our design mode (manual or semi-automatic), while text, vision, and similarity features are injected into SAM through our semantic adapter modules.

$$
\mathbf {U} = \mathbf {V} + \mathbf {s},
$$

project to SAM’s channel dimension,

$$
\mathbf {U} ^ {\prime} = \mathbf {U W} _ {v} \in \mathbb {R} ^ {N \times C _ {\ell}},
$$

reshape to $H \times W \times C _ { \ell }$ , and interpolate to obtain $\mathbf { U } _ { \ell } \in \mathbb { R } ^ { H _ { \ell } \times W _ { \ell } \times C _ { \ell } }$ . The text embedding is projected and activated,

$$
\mathbf {t} ^ {\prime} = \operatorname{GELU} (\mathbf {W} _ {t} \mathbf {t}) \in \mathbb {R} ^ {C _ {\ell}},
$$

and broadcast spatially to $\mathbf { T } _ { \ell } \in \mathbb { R } ^ { H _ { \ell } \times W _ { \ell } \times C _ { \ell } }$ . Semantic conditioning is injected through residual addition:

$$
\hat {\mathbf {F}} _ {\ell} = \mathbf {F} _ {\ell} + \mathbf {U} _ {\ell} + \mathbf {T} _ {\ell},
$$

after which the adapter output is merged back into the main pathway by adding it to the output of the MLP block.

# 3.4 Joint Optimization & Training Objective

We train SAM and CLIP end-to-end using supervision only on SAM’s output masks. Since thresholding used for binary similarity mask prompt is nondifferentiable, gradients reach CLIP only through the injected feature pathway. We optimize a standard segmentation loss $\mathcal { L } = \mathcal { L } _ { \mathrm { B C E } } + \mathcal { L } _ { \mathrm { D i c e } } + \mathcal { L } _ { \mathrm { I o U } }$ . We fine-tune only CLIP’s vision encoder’s attention blocks while freezing the MLP blocks and the text encoder; for SAM, we fine-tune the prompt encoder, mask decoder, and the inserted adapters.

# 3.5 Efficiency and Trainable Parameter Accounting

Table 1 reports total vs. trainable parameters. Overall, we train 49.0M parameters and deploy SAM+CLIP-vision (197.1M) under a closed-set assumption where class text embeddings are cached.

![](images/f89bdff0d31b72dd4ab68775a81441240943994857ed3daaf2d6a51668e9e1f8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Window Partition"] --> B["Layer Norm"]
    B --> C["Attention"]
    C --> D["Regular Adapter"]
    D --> E["MLP"]
    E --> F["Semantic Adapter"]
    F --> G["Text,Vision,Similarity embeds"]
    G --> H["Layer Norm"]
    H --> I["Window Unpartition"]
    I --> J["Attention"]
    J --> K["Regular Adapter"]
    K --> L["MLP"]
    L --> M["Semantic Adapter"]
    M --> N["Text,Vision,Similarity embeds"]
    N --> O["Layer Norm"]
    O --> P["Attention"]
    P --> Q["Regular Adapter"]
    Q --> R["MLP"]
    R --> S["Semantic Adapter"]
    S --> T["Text,Vision,Similarity embeds"]
    T --> U["Layer Norm"]
    U --> V["Attention"]
    V --> W["Regular Adapter"]
    W --> X["MLP"]
    X --> Y["Semantic Adapter"]
    Y --> Z["Text,Vision,Similarity embeds"]
```
</details>

![](images/662da51c0f8da9227d63568d6909be0b94c058718375e493a98a72e74564d819.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Vision-Similarity embeddings"] --> B["Vision projection"]
    B --> C["Resize"]
    D["Text embeddings"] --> E["Text projection"]
    E --> F["act"]
    F --> G["+"]
    H["x"] --> G
    I["Semantic Adapter"] --> G
    G --> J["Fc1"]
    G --> K["act"]
    G --> L["Fc2"]
```
</details>

Fig. 4: Adapter Placement and Internal Structure. The figure on the left shows the structure of a ViT block in SAM’s image encoder, and how our adapters are placed in parallel to the Attention and MLP blocks. The semantic adapter is in parallel to the MLP and receives the semantic inputs generated from CLIP. The right side shows the design of our semantic adapters. Inside each semantic adapter, text features are projected and fused through GELU, while vision–similarity (vis–sim) features are projected, spatially aligned, and added to SAM’s representation.

# 4 Experiments

We evaluate on (i) camouflaged object detection (COD10K, CAMO, and CHAM-ELEON) and (ii) low-label multi-class segmentation (COCO, ADE20K, and Pascal VOC), with four experiments: SAM-PEFT baselines, VLM+SAM pipelines, comparison to SAM 3, and comparison to modern multi-class segmentation (semi-supervised and labelled-only). Unless noted, we use SAM ViT-B, and use the CLIPSurgery [22] architecture for CLIP except in a few ablations; bold and underline mark best/second-best. We use the semi-automatic (text-only) mode in all experiments except Experiment 1, where we use both modes. For COCO, ADE20K, and Pascal VOC datasets, we use low-labelled-data splits commonly used in semi-supervised segmentation settings to simulate low-labelled-data scenarios [12, 36, 37, 45].

Experiment 1: SAM PEFT baselines. We follow standard COD protocols on COD10K [8], CAMO [18], and CHAMELEON [31]: train on COD10K+CAMO train splits; evaluate on their test splits and CHAMELEON; report MAE, $S _ { \alpha } ,$ $E _ { \phi }$ , and $F _ { \beta } ^ { w }$ following [8].

Table 2 shows that our manual and text-only variants achieve the best overall results, outperforming SAM-only PEFT baselines with and without manual prompts, supporting the value of active semantic supervision (both vision and text encoders of CLIP, and fine-tuning CLIP) and encoder-level semantic conditioning of SAM.

Experiment 2: VLM+SAM baselines. We compare against representative vision–language SAM pipelines, including GroundedSAM [29] and CLIPSurgery [22] with SAM, across COCO, ADE20K, PASCAL VOC, and camouflaged object benchmarks. GroundedSAM performs external grounding via GroundingDINO [25], while CLIPSurgery uses CLIP-based similarity maps to generate prompts for SAM.

For CLIPSurgery, we report the original zero-shot model as well as progressively stronger variants that add dense mask prompts from similarity maps,

Table 1: Parameter accounting of CLIP-Guided SAM (in millions). “Trainable” denotes parameters updated during training. “Deployment” refers to the closed-set inference configuration where CLIP text embeddings are cached and only CLIP-vision is required at runtime. 

<table><tr><td>Component</td><td>Total (M)</td><td>Trainable (M)</td></tr><tr><td>SAM (overall)</td><td>110.3</td><td>20.6</td></tr><tr><td>Image encoder (incl. adapters)</td><td>106.2</td><td>16.6</td></tr><tr><td>Mask decoder</td><td>4.1</td><td>4.1</td></tr><tr><td>CLIP (overall)</td><td>150.3</td><td>28.4</td></tr><tr><td>CLIP-vision</td><td>86.8</td><td>28.3</td></tr><tr><td>CLIP-text</td><td>63.4</td><td>0</td></tr><tr><td>Full system total</td><td>260.5</td><td>49.0</td></tr><tr><td>Deployment total (SAM + CLIP-vision)</td><td>197.1</td><td>49.0</td></tr></table>

Adapter overhead. The SAM image-encoder adapters introduce 16.6M parameters, corresponding to an 18.5% increase relative to the vanilla SAM backbone. In closedset inference, CLIP text embeddings are precomputed once and the text encoder is not required at runtime.

decoder fine-tuning, and adapters. For COCO/ADE20K/VOC, our model is trained on the split with the least supervision.

Table 3 shows that our jointly trained framework consistently outperforms CLIPSurgery variants across all datasets and GroundedSAM on all benchmarks except COCO (by a small margin). Notably, even when CLIPSurgery is strengthened with decoder fine-tuning and adapters, our method achieves substantially better performance despite using the same CLIP backbone, indicating that the gains arise from internal semantic conditioning and joint SAM–CLIP coadaptation rather than from model size or training alone.

The largest improvements appear on camouflaged object benchmarks, where purely spatial grouping is insufficient. While fine-tuning improves CLIPSurgery variants, external semantic grounding alone remains insufficient; jointly adapting SAM and CLIP through internal semantic conditioning proves critical for robust segmentation in these challenging scenarios.

Experiment 3: Direct Comparison with SAM 3. SAM 3 [2] is a large-scale segmentation foundation model with built-in text-prompt capability trained on massive datasets. Unlike SAM 3, which is trained end-to-end for concept segmentation on massive datasets, our method relies on parameter-efficient semantic injection and co-adaptation using limited supervision.

We evaluate SAM 3 both zero-shot and with light fine-tuning on COCO 1/512, training only its segmentation head and transformer-based fusion components (23M parameters). Even under this limited fine-tuning regime, SAM 3 achieves strong performance. For fair comparison, predictions are converted to per-class binary masks and evaluated using the same class-wise mIoU protocol as our method.

Table 2: First-experiment results on camouflaged object datasets: CHAMELEON, CAMO, and COD10K. 

<table><tr><td rowspan="2">Method</td><td colspan="4">CHAMELEON</td><td colspan="4">CAMO</td><td colspan="4">COD10K</td></tr><tr><td> $S_{\alpha}$ (↑)</td><td> $E_{\phi}$ (↑)</td><td> $F_{\beta}^{w}$ (↑)</td><td>MAE(↓)</td><td> $S_{\alpha}$ (↑)</td><td> $E_{\phi}$ (↑)</td><td> $F_{\beta}^{w}$ (↑)</td><td>MAE(↓)</td><td> $S_{\alpha}$ (↑)</td><td> $E_{\phi}$ (↑)</td><td> $F_{\beta}^{w}$ (↑)</td><td>MAE(↓)</td></tr><tr><td colspan="13">(A) Methods requiring manual point prompts</td></tr><tr><td>SAM [15]</td><td>0.727</td><td>0.734</td><td>0.639</td><td>0.081</td><td>0.684</td><td>0.687</td><td>0.606</td><td>0.132</td><td>0.783</td><td>0.798</td><td>0.701</td><td>0.050</td></tr><tr><td>SU-SAM-Series [32]</td><td>0.762</td><td>0.824</td><td>0.552</td><td>0.083</td><td>0.671</td><td>0.779</td><td>0.489</td><td>0.139</td><td>0.720</td><td>0.729</td><td>0.455</td><td>0.073</td></tr><tr><td>SU-SAM-Mix [32]</td><td>0.915</td><td>0.918</td><td>0.819</td><td>0.029</td><td>0.861</td><td>0.895</td><td>0.763</td><td>0.065</td><td>0.881</td><td>0.868</td><td>0.751</td><td>0.029</td></tr><tr><td>SU-SAM-Parallel [32]</td><td>0.930</td><td>0.940</td><td>0.856</td><td>0.024</td><td>0.890</td><td>0.925</td><td>0.815</td><td>0.053</td><td>0.915</td><td>0.909</td><td>0.818</td><td>0.020</td></tr><tr><td>SU-SAM-LoRA [32]</td><td>0.689</td><td>0.764</td><td>0.384</td><td>0.130</td><td>0.641</td><td>0.747</td><td>0.379</td><td>0.168</td><td>0.671</td><td>0.669</td><td>0.324</td><td>0.097</td></tr><tr><td>SAM-PTx [14]</td><td>0.930</td><td>0.940</td><td>0.853</td><td>0.024</td><td>0.891</td><td>0.921</td><td>0.810</td><td>0.053</td><td>0.915</td><td>0.906</td><td>0.813</td><td>0.020</td></tr><tr><td>Ours (manual)</td><td>0.927</td><td>0.967</td><td>0.903</td><td>0.017</td><td>0.896</td><td>0.944</td><td>0.855</td><td>0.035</td><td>0.914</td><td>0.965</td><td>0.874</td><td>0.015</td></tr><tr><td colspan="13">(B) Methods without manual point prompts</td></tr><tr><td>†SINetV2 [8]</td><td>0.888</td><td>0.942</td><td>0.816</td><td>0.030</td><td>0.820</td><td>0.882</td><td>0.743</td><td>0.070</td><td>0.815</td><td>0.887</td><td>0.680</td><td>0.037</td></tr><tr><td>SAM-Adapter [4]</td><td>0.896</td><td>0.919</td><td>0.824</td><td>0.033</td><td>0.847</td><td>0.873</td><td>0.765</td><td>0.070</td><td>0.883</td><td>0.918</td><td>0.801</td><td>0.025</td></tr><tr><td>Conv-LoRA [42]</td><td>0.891</td><td>0.922</td><td>0.790</td><td>0.031</td><td>0.832</td><td>0.865</td><td>0.741</td><td>0.073</td><td>0.861</td><td>0.903</td><td>0.747</td><td>0.031</td></tr><tr><td>TS-SAM [40]</td><td>0.912</td><td>0.947</td><td>0.849</td><td>0.023</td><td>0.826</td><td>0.862</td><td>0.753</td><td>0.073</td><td>0.863</td><td>0.909</td><td>0.771</td><td>0.029</td></tr><tr><td>Ours (text-only)</td><td>0.923</td><td>0.970</td><td>0.905</td><td>0.018</td><td>0.877</td><td>0.924</td><td>0.845</td><td>0.044</td><td>0.899</td><td>0.951</td><td>0.854</td><td>0.018</td></tr></table>

† Note: SINetV2 is not SAM-based but is included as a key baseline for camouflaged object detection, as it was introduced by the authors of the COD10K dataset and remains a strong benchmark reference.

Table 3: Second-experiment results across datasets. Class mIoU is reported for PASCAL, ADE20K, and COCO; MAE is reported for COD10K, CAMO, and CHAMELEON. 

<table><tr><td>Method</td><td>Fine-tuning SAM?</td><td>VLM Size</td><td>PASCAL (mIoU %)</td><td>ADE20K (mIoU %)</td><td>COCO (mIoU %)</td><td>COD10K (MAE)</td><td>CAMO (MAE)</td><td>CHAMELEON (MAE)</td></tr><tr><td>GroundedSAM [29]</td><td>No</td><td>172M (GroundingDINO)</td><td>72.0</td><td>42.4</td><td>61.3</td><td>0.500</td><td>0.146</td><td>0.249</td></tr><tr><td>CLIPSurgery+SAM (vanilla) [22]</td><td>No</td><td>87M (CS-CLIP-Base)</td><td>41.2</td><td>20.0</td><td>23.9</td><td>0.533</td><td>0.478</td><td>0.253</td></tr><tr><td> $\dagger$ CLIPSurgery+SAM (modified) [22]</td><td>No</td><td>87M (CS-CLIP-Base)</td><td>48.6</td><td>23.4</td><td>28.0</td><td>0.287</td><td>0.527</td><td>0.501</td></tr><tr><td>CLIPSurgery+SAM [22]</td><td>Yes (decoder)</td><td>87M (CS-CLIP-Base)</td><td>58.8</td><td>23.1</td><td>34.6</td><td>0.045</td><td>0.100</td><td>0.046</td></tr><tr><td>CLIPSurgery+SAM [22]</td><td>Yes (adapters + decoder)</td><td>87M (CS-CLIP-Base)</td><td>61.4</td><td>24.8</td><td>36.2</td><td>0.026</td><td>0.063</td><td>0.020</td></tr><tr><td>CLIP-Guided SAM (ours)</td><td>Yes</td><td>87M (CS-CLIP-Base)</td><td>78.5</td><td>47.9</td><td>60.5</td><td>0.018</td><td>0.044</td><td>0.018</td></tr></table>

† Note: the modified, training-free CLIPSurgery+SAM baseline includes binary similarity maps as mask prompts in addition to points prompts.   
The fine-tuned results use the dataset with the least amount of data for COCO/ADE/PASCAL, which are 1/512, 1/64, and 1/16 respectively.

Table 4 shows that with limited supervision, our smaller system (197M deployment parameters) approaches or surpasses SAM 3 zero-shot performance and remains competitive as supervision increases.

To examine scaling behaviour, we also evaluate two larger backbone variants: CLIP(ViT-L/14)+SAM(ViT-B) and CLIP(ViT-B/16)+SAM(ViT-H). Increasing CLIP capacity yields a substantial improvement (65.0 mIoU on COCO 1/512), whereas increasing SAM capacity alone provides a smaller gain (61.2 mIoU). This suggests that strengthening the semantic guide is more beneficial than scaling the segmentation backbone within our framework.

Experiment 4: Comparison to multi-class segmentation. This comparison is not task-identical: prior methods predict all classes jointly, whereas we predict a binary mask per class prompt and assume image-level class presence via text prompts. Nevertheless, it provides a useful reference point for the extent to which prompt-conditioned segmentation can go when class intent is available.

Table 4: Direct comparison with SAM 3 [2] on COCO. Results are mIoU. 

<table><tr><td>Method</td><td>Total Params</td><td>Trainable Params</td><td>Zero-shot</td><td>1/512 (232)</td><td>1/256 (463)</td><td>1/128 (925)</td><td>1/64 (1849)</td><td>1/32 (3697)</td></tr><tr><td>SAM 3</td><td>840M</td><td>23M*</td><td>63.7</td><td>68.8</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Ours (CLIP-B + SAM-B)</td><td> $197M^†$ </td><td>49M</td><td>28.0</td><td>60.5</td><td>63.2</td><td>65.8</td><td>67.7</td><td>69.2</td></tr><tr><td>Ours (CLIP-L + SAM-B)</td><td>543M</td><td>126M</td><td>-</td><td>65.0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Ours (CLIP-B + SAM-H)</td><td>809M</td><td>50M</td><td>-</td><td>61.2</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

\*SAM 3 fine-tuning trains the segmentation head and transformer layers only. †197M corresponds to SAM-B + added adapters + CLIP vision backbone at inference. Text embeddings can be precomputed under closed-set assumptions.

Table 5: Experiment 4: Comparison to multi-class segmentation. Class mIoU (%) on COCO, ADE20K, and PASCAL VOC under varying label fractions. Baselines perform joint multi-class prediction, whereas our method predicts a binary mask per class prompt using labelled data only (no unlabelled data). 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Unlab data?</td><td rowspan="2">Img-level prompts?</td><td rowspan="2">Encoder</td><td rowspan="2"># Params</td><td colspan="5">COCO (mIoU %)</td><td colspan="5">ADE20K (mIoU %)</td><td colspan="5">PASCAL VOC (mIoU %)</td></tr><tr><td>1/512(232)</td><td>1/256(463)</td><td>1/128(925)</td><td>1/64(1849)</td><td>1/32(3697)</td><td>1/64(316)</td><td>1/32(631)</td><td>1/16(1262)</td><td>1/8(2526)</td><td>1/4(5052)</td><td>1/16(92)</td><td>1/8(183)</td><td>1/4(366)</td><td>1/2(732)</td><td>Full(1464)</td></tr><tr><td>SemiVL [12]</td><td>Yes</td><td>No</td><td>CLIP-Base</td><td>88M</td><td>50.1</td><td>52.8</td><td>53.6</td><td>55.4</td><td>56.5</td><td>33.7</td><td>35.1</td><td>37.2</td><td>39.4</td><td>-</td><td>84.0</td><td>85.6</td><td>86.0</td><td>86.7</td><td>87.3</td></tr><tr><td>UniMatchV2 [37]</td><td>Yes</td><td>No</td><td>DINOv2-B</td><td>97.5M</td><td>47.9</td><td>55.8</td><td>58.7</td><td>60.4</td><td>63.3</td><td>38.7</td><td>45.0</td><td>46.7</td><td>49.8</td><td>52.0</td><td>86.3</td><td>87.9</td><td>88.9</td><td>90.0</td><td>90.8</td></tr><tr><td>UniMatchV2 [37] (labelled-only)</td><td>No</td><td>No</td><td>DINOv2-B</td><td>97.5M</td><td>36.8</td><td>45.8</td><td>52.1</td><td>56.2</td><td>59.5</td><td>26.1</td><td>39.3</td><td>42.8</td><td>46.4</td><td>49.0</td><td>76.9</td><td>82.1</td><td>85.3</td><td>87.2</td><td>88.3</td></tr><tr><td>Our Method (labelled-only)</td><td>No</td><td>Yes</td><td>CLIP-B+SAM-B</td><td>197M</td><td>60.5</td><td>63.2</td><td>65.8</td><td>67.7</td><td>69.2</td><td>47.9</td><td>51.6</td><td>55.2</td><td>57.7</td><td>61.1</td><td>78.3</td><td>81.7</td><td>82.9</td><td>83.9</td><td>85.0</td></tr></table>

Table 5 shows strong gains on COCO and ADE20K, despite our method using no unlabelled data while several baselines rely on large semi-supervised training sets (SemiVL [12], UniMatchV2 [37]). Compared to the labelled-only variant of UniMatchV2, our results highlight the effectiveness of prompt-conditioned segmentation even without additional data. Importantly, these improvements are not explained by backbone capacity alone: a CLIP+SAM baseline with similar backbones but without semantic adapters performs substantially worse (see Experiment 2 and Table 3).

Performance differences are smaller on PASCAL VOC, a simpler dataset where prompt-based conditioning offers less advantage and factors such as backbone capacity and semi-supervised training become more influential.

# 4.1 Ablation Studies

Unless stated otherwise, ablations use PASCAL VOC 1/16 in the semi-automatic setting and are evaluated using mIoU.

Which injected modality matters? Table 6 shows that similarity features provide the dominant signal: removing similarity causes a sharp performance drop, while similarity-only remains strong (77.5 mIoU) and almost matches the full method. In contrast, text-only and vision-only variants perform substantially worse. This means that similarity features carry the main semantic localization, and text and vision features refine it slightly. Furthermore, adapters without semantic inputs provide minimal benefit (67.1 vs. 66.6 without adapters), confirming that semantic injection—rather than having regular adapters alone—is the main source of improvement.

Table 6: Ablation on feature modalities injected into adapters (VOC 1/16). 

<table><tr><td>Text</td><td>Vision</td><td>Sim.</td><td>Adapt.</td><td>Config</td><td>mIoU</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>Full fusion</td><td>78.3</td></tr><tr><td>√</td><td>√</td><td>✗</td><td>√</td><td>Text+Vision</td><td>69.5</td></tr><tr><td>√</td><td>✗</td><td>√</td><td>√</td><td>Text+Similarity</td><td>78.1</td></tr><tr><td>✗</td><td>√</td><td>√</td><td>√</td><td>Vision+Similarity</td><td>77.5</td></tr><tr><td>√</td><td>✗</td><td>✗</td><td>√</td><td>Text only</td><td>67.2</td></tr><tr><td>✗</td><td>√</td><td>✗</td><td>√</td><td>Vision only</td><td>68.8</td></tr><tr><td>✗</td><td>✗</td><td>√</td><td>√</td><td>Sim. only</td><td>77.5</td></tr><tr><td>✗</td><td>✗</td><td>✗</td><td>√</td><td>Parallel only</td><td>67.1</td></tr><tr><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>No adapters</td><td>66.6</td></tr></table>

Table 7: Co-adaptation budget study on PASCAL VOC 1/16 (mIoU %). (C, S) denotes the number of CLIP vision blocks (attention weights fine-tuned) and SAM blocks equipped with trainable semantic adapters. 

<table><tr><td>SAM/CLIP</td><td>C=4</td><td>C=8</td><td>C=12</td></tr><tr><td>S=4</td><td>77.4</td><td>77.2</td><td>78.2</td></tr><tr><td>S=8</td><td>77.0</td><td>77.6</td><td>78.0</td></tr><tr><td>S=12</td><td>77.1</td><td>77.17</td><td>78.1</td></tr></table>

Co-adaptation budget. We vary (C, S): the number of trainable CLIP vision blocks (attention) and the number of SAM blocks equipped with semantic adapters. Table 7 reveals an asymmetric effect: performance is more sensitive to CLIP adaptation than to increasing SAM semantic depth. While increasing the number of trainable CLIP blocks consistently improves or stabilizes performance, expanding semantic adapters across SAM layers yields only marginal gains, suggesting that SAM requires only modest semantic capacity once strong semantic guidance is available. This trend is consistent with the scaling results in Experiment 3, where increasing CLIP capacity (CLIP-L) provided larger gains than scaling the SAM backbone (SAM-H), indicating that the semantic encoder plays a dominant role in the framework’s performance.

Train-test prompt alignment and dense similarity prompting. Using similarity maps as dense mask prompts provides additional spatial conditioning beyond point prompts (78.3→77.6). More importantly, training with CLIP-sampled points significantly improves robustness: models trained on GT points but evaluated on CLIP prompts drop sharply (78.3→69.8), supporting our strategy of training with noisy prompts for text-only inference.

Analyzing CLIP adaptation. We analyze how the choice and training strategy of the semantic guide (CLIP) affects performance. Replacing CLIPSurgery with standard CLIP reduces accuracy substantially (78.3→72.8 mIoU), confirming that spatially aligned similarity maps are critical for effective conditioning. Freezing CLIPSurgery also degrades performance (73.7 mIoU), indicating that jointly optimizing the semantic guide and the segmentation backbone is important.

Table 8: Effect of pre-finetuning CLIP on our CLIP+SAM framework (VOC 1/16). Values are mIoU. 

<table><tr><td>Setup</td><td>Stage</td><td>Start</td><td>Final</td><td> $\Delta$ </td></tr><tr><td>CLIP pre-finetune</td><td>100 epochs</td><td>-</td><td>58</td><td>-</td></tr><tr><td>SAM+CLIP (no pre)</td><td>Joint</td><td>38.7</td><td>78.3</td><td>+35.7</td></tr><tr><td>SAM+CLIP (with pre)</td><td>Joint</td><td>54.1</td><td>67.5</td><td>+12.4–14.4</td></tr></table>

Table 9: Ablation on loss function (VOC 1/16). Values are mIoU (%). 

<table><tr><td>Loss</td><td>mIoU (%)</td></tr><tr><td>BCE</td><td>76.1</td></tr><tr><td>BCE + Dice</td><td>77.7</td></tr><tr><td>BCE + IoU</td><td>78.1</td></tr><tr><td>Dice + IoU</td><td>76.8</td></tr><tr><td>BCE + Dice + IoU</td><td>78.3</td></tr></table>

Interestingly, pre-fine-tuning CLIPSurgery on the target dataset improves its standalone mask quality but leads to worse final performance after joint training (Table 8). Although the pre-adapted model starts from a stronger initialization, it converges to a substantially lower final accuracy (67.5 mIoU vs. 78.3), suggesting that strong prior specialization can hinder SAM–CLIP co-adaptation and may even underperform weaker semantic guides such as standard CLIP (72.8) when integrated into the joint system. This suggests that maximizing the standalone performance of a vision–language model does not necessarily translate to better segmentation performance when the model is later integrated into a jointly trained system.

Overall, these results indicate that effective semantic conditioning emerges from joint optimization rather than from independently maximizing the quality of the semantic encoder.

Loss. Table 9 shows that combining BCE, Dice, and IoU yields the best performance.

# 4.2 Qualitative Analysis

Figure 5 shows qualitative comparisons on ADE20K (1/16 split) across vanilla SAM, CLIP- and CLIPSurgery-driven variants, and our CLIP-Guided SAM with and without fine-tuning. The jointly fine-tuned model produces more complete and semantically consistent masks, recovering fine object boundaries and cluttered regions where baselines either miss objects or leak into distractors, corroborating the quantitative gains of our multi-modal encoder injection and joint SAM-CLIP training.

![](images/7e4b9ab8d12c924d12b6991b439f7c1ab9ea1b4203d30bb23586d6cf0e9707dc.jpg)

<details>
<summary>text_image</summary>

Image
Ground
Truth
(GT)
SAM (Vanilla)
No CLIP
GT Points
SAM (Vanilla)
CLIP (Vanilla)
No CLIP FT
No GT Points
SAM (Vanilla)
CLIP (Surgery)
No CLIP FT
No GT Points
SAM (Ours - FT)
CLIP (Surgery)
No CLIP FT
No GT Points
SAM (Ours - FT)
CLIP (Surgery)
With CLIP FT
No GT Points

AID:206.1/16 - Selected Comparisons
Prompt: ceiling
Original image
GT mask
SAM (Vanilla) no CLIP with GT points
SAM (Vanilla) with CLIP (Vanilla)
SAM (Vanilla) with CLIPSurgery (NoFT)
SAM (FT) with CLIPSurgery (NoFT)
SAM (FT) with CLIPSurgery (FT)

Prompt: grass
Original image
GT mask
SAM (Vanilla) no CLIP with GT points
SAM (Vanilla) with CLIP (Vanilla)
SAM (Vanilla) with CLIPSurgery (NoFT)
SAM (FT) with CLIPSurgery (NoFT)
SAM (FT) with CLIPSurgery (FT)

Prompt: light
Original image
GT mask
SAM (Vanilla) no CLIP with GT points
SAM (Vanilla) with CLIP (Vanilla)
SAM (Vanilla) with CLIPSurgery (NoFT)
SAM (FT) with CLIPSurgery (NoFT)
SAM (FT) with CLIPSurgery (FT)

Prompt: signboard
Original image
GT mask
SAM (Vanilla) no CLIP with GT points
SAM (Vanilla) with CLIP (Vanilla)
SAM (Vanilla) with CLIPSurgery (NoFT)
SAM (FT) with CLIPSurgery (NoFT)
SAM (FT) with CLIPSurgery (FT)

Prompt: ceiling
Original image
GT mask
SAM (Vanilla) no CLIP with GT points
SAM (Vanilla) with CLIP (Vanilla)
SAM (Vanilla) with CLIPSurgery (NoFT)
SAM (FT) with CLIPSurgery (NoFT)
SAM (FT) with CLIPSurgery (FT)
</details>

Fig. 5: Qualitative analysis on ADE20K (1/16 split). Comparison of vanilla SAM, CLIP-based prompting variants, and our CLIP-Guided SAM before and after fine-tuning.

# 5 Conclusion

We introduced CLIP-Guided SAM, a parameter-efficient framework for enabling semantic conditioning in the Segment Anything Model under limited supervision and compute. Instead of treating a vision–language model as an external prompt generator, our approach injects CLIP-derived semantic signals directly into SAM’s image encoder through semantic adapters, allowing semantics to influence feature formation while preserving SAM’s promptable design. The same mechanism supports both interactive segmentation with user-provided points and a semi-automatic text-only regime, in which spatial prompts are derived from CLIP similarity maps.

Our experiments show that effective semantic conditioning requires joint coadaptation between the segmentation backbone and its semantic guide. Freezing or independently pre-training CLIP degrades performance, while end-to-end coadaptation enables CLIP to serve as a task-aligned semantic guide for SAM. Across camouflaged object benchmarks and low-label segmentation datasets, including COCO, ADE20K, and Pascal VOC, CLIP-Guided SAM consistently outperforms SAM-based PEFT methods and VLM+SAM pipelines while remaining competitive with powerful promptable segmentation foundation models such as SAM 3 under limited compute and supervision.

Overall, these results suggest that internal semantic co-adaptation provides a practical alternative to shallow prompt-based coupling when adapting promptable segmentation models under limited data and compute.

# References

1. Aleem, S., Wang, F., Maniparambil, M., Arazo, E., Dietlmeier, J., Curran, K., Connor, N.E., Little, S.: Test-time adaptation with SALIP: A cascade of SAM and CLIP for zero-shot medical image segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5184–5193 (2024)   
2. Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., Lei, J., Ma, T., Guo, B., Kalla, A., Marks, M., Greer, J., Wang, M., Sun, P., Rädle, R., Afouras, T., Mavroudi, E., Xu, K., Wu, T.H., Zhou, Y., Momeni, L., Hazra, R., Ding, S., Vaze, S., Porcher, F., Li, F., Li, S., Kamath, A., Cheng, H.K., Dollár, P., Ravi, N., Saenko, K., Zhang, P., Feichtenhofer, C.: Sam 3: Segment anything with concepts (2025), https://arxiv. org/abs/2511.16719   
3. Cha, J., Mun, J., Roh, B.: Learning to generate text-grounded mask for openworld semantic segmentation from only image-text pairs. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11165– 11174 (2023)   
4. Chen, T., Zhu, L., Deng, C., Cao, R., Wang, Y., Zhang, S., Li, Z., Sun, L., Zang, Y., Mao, P.: Sam-adapter: Adapting segment anything in underperformed scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3367–3375 (2023)   
5. Chen, T., Zhu, L., Ding, C., Cao, R., Wang, Y., Li, Z., Sun, L., Mao, P., Zang, Y.: Sam fails to segment anything?–sam-adapter: Adapting sam in underperformed scenes: Camouflage, shadow, medical image segmentation, and more. arXiv preprint arXiv:2304.09148 (2023)   
6. Cho, S., Shin, H., Hong, S., Arnab, A., Seo, P.H., Kim, S.: Cat-seg: Cost aggregation for open-vocabulary semantic segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4113–4123 (2024)   
7. Ding, J., Xue, N., Xia, G.S., Dai, D.: Decoupling zero-shot semantic segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11583–11592 (2022)   
8. Fan, D.P., Ji, G.P., Cheng, M.M., Shao, L.: Concealed object detection. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(10), 6024–6042 (Oct 2022). https://doi.org/10.1109/tpami.2021.3085766, http://dx.doi.org/10. 1109/TPAMI.2021.3085766   
9. Ghiasi, G., Gu, X., Cui, Y., Lin, T.Y.: Scaling open-vocabulary image segmentation with image-level labels. In: European conference on computer vision. pp. 540–557. Springer (2022)   
10. Gu, X., Lin, T.Y., Kuo, W., Cui, Y.: Open-vocabulary object detection via vision and language knowledge distillation. arXiv preprint arXiv:2104.13921 (2021)

11. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., Gelly, S.: Parameter-efficient transfer learning for nlp. In: International conference on machine learning. pp. 2790–2799. PMLR (2019)   
12. Hoyer, L., Tan, D.J., Naeem, M.F., Van Gool, L., Tombari, F.: Semivl: semisupervised semantic segmentation with vision-language guidance. In: European Conference on Computer Vision. pp. 257–275. Springer (2024)   
13. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. ICLR 1(2), 3 (2022)   
14. Jalilian, S., Bais, A.: Sam-ptx: Text-guided fine-tuning of sam with parameterefficient, parallel-text adapters. IEEE Access 14, 31732–31746 (2026). https:// doi.org/10.1109/ACCESS.2026.3668182   
15. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4015–4026 (2023)   
16. Koleilat, T., Asgariandehkordi, H., Rivaz, H., Xiao, Y.: Medclip-samv2: Towards universal text-driven medical image segmentation. arxiv 2024. arXiv preprint arXiv:2409.19483 (2024)   
17. Kollias, D., Arsenos, A., Wingate, J., Kollias, S.: Sam2clip2sam: Vision language model for segmentation of 3d ct scans for covid-19 detection. arXiv preprint arXiv:2407.15728 (2024)   
18. Le, T.N., Nguyen, T.V., Nie, Z., Tran, M.T., Sugimoto, A.: Anabranch network for camouflaged object segmentation. Computer vision and image understanding 184, 45–56 (2019)   
19. Lester, B., Al-Rfou, R., Constant, N.: The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691 (2021)   
20. Li, B., Weinberger, K.Q., Belongie, S., Koltun, V., Ranftl, R.: Language-driven semantic segmentation. arXiv preprint arXiv:2201.03546 (2022)   
21. Li, S., Cao, J., Ye, P., Ding, Y., Tu, C., Chen, T.: ClipSAM: CLIP and SAM collaboration for zero-shot anomaly segmentation. Neurocomputing 618, 129122 (2025)   
22. Li, Y., Wang, H., Duan, Y., Zhang, J., Li, X.: A closer look at the explainability of contrastive language-image pre-training. Pattern Recognition 162, 111409 (2025)   
23. Liang, F., Wu, B., Dai, X., Li, K., Zhao, Y., Zhang, H., Zhang, P., Vajda, P., Marculescu, D.: Open-vocabulary semantic segmentation with mask-adapted clip. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 7061–7070 (2023)   
24. Liu, K., Wang, J., Jin, R., Hwang, W., Chung, T.S.: Schnet: Sam marries clip for human parsing. arXiv preprint arXiv:2503.22237 (2025)   
25. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., et al.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In: European conference on computer vision. pp. 38–55. Springer (2024)   
26. Luo, H., Bao, J., Wu, Y., He, X., Li, T.: Segclip: Patch aggregation with learnable centers for open-vocabulary semantic segmentation. In: International Conference on Machine Learning. pp. 23033–23044. PMLR (2023)   
27. Ma, X., Fu, J., Liao, W., Zhang, S., Wang, G.: Clisc: Bridging clip and sam by enhanced cam for unsupervised brain tumor segmentation. In: 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI). pp. 1–5. IEEE (2025)   
28. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from

natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)   
29. Ren, T., Liu, S., Zeng, A., Lin, J., Li, K., Cao, H., Chen, J., Huang, X., Chen, Y., Yan, F., Zeng, Z., Zhang, H., Li, F., Yang, J., Li, H., Jiang, Q., Zhang, L.: Grounded sam: Assembling open-world models for diverse visual tasks (2024)   
30. Shin, G., Xie, W., Albanie, S.: Reco: Retrieve and co-segment for zero-shot transfer. Advances in Neural Information Processing Systems 35, 33754–33767 (2022)   
31. Skurowski, P., Abdulameer, H., Błaszczyk, J., Depta, T., Kornacki, A., Kozieł, P.: Animal camouflage analysis: Chameleon database 2(6), 7   
32. Song, Y., Zhou, Q., Lu, X., Shao, Z., Ma, L.: Su-sam: A simple unified framework for adapting segment anything model in underperformed scenes. arXiv preprint arXiv:2401.17803 (2024)   
33. Wang, H., Vasu, P.K.A., Faghri, F., Vemulapalli, R., Farajtabar, M., Mehta, S., Rastegari, M., Tuzel, O., Pouransari, H.: Sam-clip: Merging vision foundation models towards semantic and spatial understanding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3635–3647 (2024)   
34. Xu, J., De Mello, S., Liu, S., Byeon, W., Breuel, T., Kautz, J., Wang, X.: Groupvit: Semantic segmentation emerges from text supervision. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 18134– 18144 (2022)   
35. Xu, M., Zhang, Z., Wei, F., Lin, Y., Cao, Y., Hu, H., Bai, X.: A simple baseline for open-vocabulary semantic segmentation with pre-trained vision-language model. In: European Conference on Computer Vision. pp. 736–753. Springer (2022)   
36. Yang, L., Qi, L., Feng, L., Zhang, W., Shi, Y.: Revisiting weak-to-strong consistency in semi-supervised semantic segmentation (2023), https://arxiv.org/abs/2208. 09910   
37. Yang, L., Zhao, Z., Zhao, H.: Unimatch v2: Pushing the limit of semi-supervised semantic segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)   
38. Yang, X., Gong, X.: Foundation model assisted weakly supervised semantic segmentation. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision. pp. 523–532 (2024)   
39. Yu, X., Elazab, A., Ge, R., Jin, H., Jiang, X., Jia, G., Wu, Q., Shi, Q., Wang, C.: Ich-scnet: Intracerebral hemorrhage segmentation and prognosis classification network using clip-guided sam mechanism. In: 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). pp. 2795–2800. IEEE (2024)   
40. Yu, Y., Xu, C., Wang, K.: Ts-sam: Fine-tuning segment-anything model for downstream tasks. In: 2024 IEEE International Conference on Multimedia and Expo (ICME). pp. 1–6. IEEE (2024)   
41. Yuan, H., Li, X., Zhou, C., Li, Y., Chen, K., Loy, C.C.: Open-vocabulary sam: Segment and recognize twenty-thousand classes interactively. In: European Conference on Computer Vision. pp. 419–437. Springer (2024)   
42. Zhong, Z., Tang, Z., He, T., Fang, H., Yuan, C.: Convolution meets lora: Parameter efficient finetuning for segment anything model. arXiv preprint arXiv:2401.17868 (2024)   
43. Zhou, C., Loy, C.C., Dai, B.: Extract free dense labels from clip. In: European conference on computer vision. pp. 696–712. Springer (2022)   
44. Zhou, Z., Lei, Y., Zhang, B., Liu, L., Liu, Y.: Zegclip: Towards adapting clip for zero-shot semantic segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11175–11185 (2023)

45. Zou, Y., Zhang, Z., Zhang, H., Li, C.L., Bian, X., Huang, J.B., Pfister, T.: Pseudoseg: Designing pseudo labels for semantic segmentation. arXiv preprint arXiv:2010.09713 (2020)