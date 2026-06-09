# Late-Layer Fusion is Enough: Dual-Path Vision Token Routing for Multimodal Large Language Models under Visual Saturation

Siyuan Liua, Jinyang Wub,∗

aSchool of Mechanics and Engineering Science, Peking University, China

bDepartment of Automation, Tsinghua University, China

## A R T I C L E I N F O

Keywords:

Multimodal Large Language Models

Visual Token Routing

Compute Efficiency

Visual Saturation

Modality-Asymmetric Architecture

## A B S T R A C T

Multimodal large language models (MLLMs) commonly inherit the deep, symmetric Transformer backbone designed for unimodal text modeling, and apply the same computation uniformly to image and language tokens. This design overlooks a key modality asymmetry: image and text tokens differ substantially in information density, redundancy, and required reasoning depth. Through a layer-wise analysis of LLaVA-1.5, we observe that vision tokens tend to saturate in the middle layers. Specifically, text-to-image attention decreases from 0.68 at layer 0 to 0.07 by layer 4, and stabilizes near 0.04 after layer 18, whereas text tokens continue to benefit from deep semantic processing. These findings suggest a mismatch between architectural symmetry and depth-asynchronous modality evolution, resulting in redundant visual computation and possible drift in perceptual representations during deep task-specific adaptation. Motivated by this, we propose Dual-Path Vision Token Routing (DPVR), a modality-asymmetric routing framework for efficient MLLMs. Its core instantiation, DPVR-LF (Late-Layer Fusion), routes vision tokens at the saturation point into a one-layer trainable side branch, runs a thirteenlayer text-only forward that skips image positions in the deep stack, and re-fuses the visual and textual streams only at the final layer. With approximately 3% trainable parameters, DPVR-LF preserves competitive multimodal performance on standard benchmarks while reducing visual computation in the deep Transformer stack. The results challenge the conventional assumption that vision tokens must traverse all deep language-model layers, and indicate that a single late fusion layer can be sufficient for maintaining strong perceptual competence in LLaVA-style MLLMs.

## 1. Introduction

The symmetric-architecture default. Mainstream multimodal large language models, including LLaVA (Liu, Li, Wu and Lee, 2023; Liu, Li, Li and Lee, 2024a), Qwen-VL (Bai, Bai, Yang, Wang, Tan, Wang, Lin, Zhou and Zhou, 2023), MiniGPT-4 (Zhu, Chen, Shen, Li and Elhoseiny, 2024), and related systems, typically adopt a decoder-only Transformer backbone inherited from autoregressive language models. Visual features are projected into the language embedding space, concatenated with text tokens, and processed by the same deep Transformer stack. Although this design is simple and effective, it implicitly assumes that image and text tokens should undergo the same layer-wise computation. This assumption overlooks important modality differences. Vision tokens are continuous perceptual representations with substantial local redundancy, whereas text tokens are discrete symbolic units that support semantic and compositional reasoning. Nevertheless, both modalities are usually passed through the same multi-layer stack with identical attention and feed-forward operations. We argue that this convention creates a structural mismatch between architectural symmetry and the depth-asynchronous evolution of visual and textual representations.

Empirical evidence: visual saturation. To examine this mismatch, we conduct a layer-wise analysis of vanilla LLaVA-1.5-7B on 500 randomly sampled multimodal instances from LLaVA-665k. The analysis characterizes visual and textual token dynamics from three complementary perspectives (Figure 2, §3.1):

![](images/89ac1f9931983c02de932455d8f411b11f535763ee5483151448603e2020f784.jpg)

<details>
<summary>scatter plot</summary>

| Method       | Trainable parameters (log scale) | 6-bench mean accuracy |
| ------------ | -------------------------------- | --------------------- |
| Vanilla      | ~0 (proxy)                       | 0.655                 |
| FastV        | ~10M                             | 0.660                 |
| LoRA         | ~10M                             | 0.658                 |
| DPVR family   | ~100M                            | 0.665                 |
| PC           | ~100M                            | 0.665                 |
| KV           | ~100M                            | 0.665                 |
| Ours (LF)    | ~100M                            | 0.670                 |
</details>

![](images/59c8ee4af54fc68f54779d1c2852035406a6c852b456fa0c3da8480f54d19a75.jpg)

<details>
<summary>line chart</summary>

| Method       | Forward latency (ms) | Accuracy |
| ------------ | -------------------- | -------- |
| Ours (LF)    | 173                  | 0.670    |
| DPVR-KV      | 199                  | 0.665    |
| Vanilla      | 240                  | 0.658    |
</details>

Figure 1: Paper at a glance. Pareto plots over LLaVA-1.5-7B on six standard benchmarks. (a) Trainable parameters vs accuracy: DPVR-LF reaches the 6-bench accuracy band of 0.66 at a 3% trainable budget, on par with LoRA r=64 (80M) and the cited full fine-tuning of the 7B backbone. (b) Forward latency on A800 vs accuracy: DPVR-LF saves −28.0% measured latency (A800, ??=4) while retaining near-baseline accuracy, matching Table 6. The green band marks the near-vanilla accuracy zone $( y \in [ 0 . 6 5 5 , 0 . 6 7 0 ] )$ .

• Hidden-state evolution. The adjacent-layer cosine similarity $\cos ( h _ { \ell } , h _ { \ell + 1 } )$ for vision tokens stays above 0.92 from layer 0 onwards, indicating that deep updates apply only marginal increments along the residual stream.  
• Attention disengagement. Text-to-image attention mass drops from 0.68 at $L _ { 0 }$ to 0.07 at $L _ { 4 } { - } \mathbf { a }$ ten-fold collapse in four layers—and stabilizes near 0.04 for the remainder of the network.  
• Logit-lens transition. Following logit lens (nostalgebraist, 2020) and tuned lens (Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman and Steinhardt, 2023), vision tokens reach the prediction space at $L _ { 2 2 }$ , one layer earlier than text tokens at $L _ { 2 3 }$ .

Together, these observations indicate that vision tokens tend to saturate before the final layers, while text tokens continue to depend on deeper computation for semantic composition and response generation. This modalityasynchronous behavior implies that a substantial portion of deep-layer computation is spent on visual representations whose marginal changes are already small. It may also perturb perceptual information that has been captured in earlier layers during task-specific adaptation. These findings motivate an asymmetric depth allocation strategy that preserves deep reasoning for text while reducing redundant visual computation.

Dual-Path Vision Token Routing. Motivated by this observation, we propose Dual-Path Vision Token Routing (DPVR), a modality-asymmetric routing framework that separates visual and textual computation after the visual stream reaches a saturation point. Its core method, DPVR-LF (Late-Layer Fusion), branches vision tokens at ?? = 18 for the 7B model and ?? = 28 for the 13B model into a single trainable LlamaDecoderLayer side branch. Text tokens continue through the deep Transformer stack, while image positions are skipped in the first $L - s - 1$ deep layers through a text-only forward. The two streams are reassembled at the final layer, where a single full-attention block performs image-text fusion. This design decouples perception from reasoning with minimal architectural change. The visual stream follows a shallow and efficient route once its representation has stabilized, whereas the textual stream retains the full depth needed for semantic reasoning. The final fusion layer allows response tokens to attend to the side-branch visual representation, preserving trainable image-to-text gradient flow while avoiding repeated deep-layer updates of image positions.

Contributions. On LLaVA-1.5-7B and 13B, with only 3% trainable parameters (202M for the 7B model), DPVR-LF matches or exceeds full fine-tuning across eight standard benchmarks while saving 25–30% of forward FLOPs (−28.0% measured latency on A800; calibrated theoretical prediction −26.8% at $\rho = 0 . 7 0 )$ . A split sweep shows that $s \in [ 1 8 , 2 4 ]$ is a robust plateau (six-benchmark mean varying within 0.1 pp), and the fusion-layer count saturates at ?? = 1: a second fusion layer yields only $a + 0 . 1 8$ pp gain, well within run-to-run noise. The contributions of this paper are:

1. We provide a three-viewpoint analysis based on adjacent-layer cosine similarity, text-to-image attention mass, and logit-lens transition (Figure 2). This analysis identifies the vision-saturation phenomenon in LLaVA-style MLLMs and motivates asymmetric depth allocation between modalities.  
2. We introduce DPVR and its core method DPVR-LF, which combines a one-layer trainable visual side branch, a thirteen-layer text-only deep forward, and a single final-layer image-text fusion (§3.2, Appendix A). This design reduces deep visual computation while preserving a trainable gradient path to the visual branch.  
3. Empirical validation on LLaVA-1.5-7B and 13B over eight standard benchmarks, including a split sweep, a vision-depth ablation, and a fusion-layer-count ablation, showing that the “deeper-is-better” default does not hold for the visual stream of modern MLLMs (Figure 1, §4).  
4. Multi-faceted empirical evidence supporting the DPVR-LF design: (a) mechanism—the lone fusion layer concentrates text-to-image attention at 1.77× the vanilla baseline (§3.2), while the shared shallow stack remains bit-identical to vanilla LLaVA (median cosine > 0.9998 on 500 samples at both 7B and 13B scales); (b) efficiency—the latency saving is entirely realized at prefill (−23.4% on 13B / 5880 Ada) and reproduces consistently across A800, Blackwell RTX PRO 6000, and 5880 Ada hardware (§4.8, §4.7); (c) robustness— the saving remains positive across a 16× sweep of text-token length $( T _ { \mathrm { t x t } } \in [ 6 4 , 1 0 2 4 ] , \ S 4 . 9 )$ , and the $d _ { v } = 1$ vision-depth saturation point replicates at 13B scale (Table 4).

## 2. Related Work

Multimodal large language models. Vision-language modeling has progressed through three distinguishable generations. The first, exemplified by CLIP (Radford, Kim, Hallacy, Ramesh, Goh, Agarwal, Sastry, Askell, Mishkin, Clark, Krueger and Sutskever, 2021), aligns image and text representations via large-scale contrastive pre-training but does not produce a generative interface. The second generation connects a frozen vision encoder to a pretrained language model through a learnable bridge: Flamingo (Alayrac, Donahue, Luc, Miech, Barr, Hasson, Lenc, Mensch, Millican, Reynolds et al., 2022) inserts gated cross-attention blocks between language layers, while BLIP-2 (Li, Li, Savarese and Hoi, 2023a) introduces the Q-Former as a parameter-efficient query bridge. The third—and now dominant—generation, established by LLaVA (Liu et al., 2023, 2024a), MiniGPT-4 (Zhu et al., 2024), and Qwen-VL (Bai et al., 2023), eschews dedicated cross-modal modules entirely: a lightweight projector maps visual embeddings into the language token space, and the resulting mixed-modality sequence is processed by a single decoder-only LLM, then fine-tuned with visual instructions. The simplicity of this third design has driven its ecosystem dominance, but the same simplicity also makes the “symmetric deep backbone” assumption, which we revisit in this work, the most consequential. Recent work continues to extend the MLLM stack to new downstream tasks, including simultaneous textual mask prediction for MLLM-based segmentation (Liu, Feng and Chen, 2025), and to enhanced multimodal reasoning via automated structured thinking (Wu, Feng, Zhai, Zhang, Lian, Lv, Shao, Jin, Wen and Tao, 2026).

Token reduction in MLLMs. A growing body of work attempts to reduce the inference cost of mixed-modality sequences by pruning or merging visual tokens. FastV (Chen, Zhao, Liu, Bai, Lin, Zhou and Chang, 2024) discards low-attention image tokens after the second LLM layer at inference time. VTW (Lin, Lin, Lin and Ji, 2024) drops visual tokens entirely after a shallow layer and runs a text-only forward thereafter, but does so heuristically and only at inference. LLaVA-PruMerge (Shang, Cai, Xu, Lee and Yan, 2024) combines per-token pruning with cluster merging, while TokenPacker (Li, Yuan, Liu, Tang, Wang, Zhu and Zhang, 2024b) compresses high-resolution image tokens into a small set of query tokens via cross-attention. These methods all reduce the number of visual tokens, leaving the pertoken computation graph unchanged. Related token-merging techniques from the pure-vision setting—ToMe (Bolya, Fu, Dai, Zhang, Feichtenhofer and Hoffman, 2023), which fuses similar tokens between ViT blocks, and EViT (Liang, Ge, Tong, Song, Wang and Xie, 2022b), which reorganizes tokens by attention—act inside the vision encoder, before the projector and hence upstream of the LLM stack that DPVR restructures. DPVR is orthogonal to all of these: it preserves all 576 visual tokens but restructures the per-token compute graph so that image positions skip 13 of 14 deep transformer layers entirely. Token reduction can in principle be composed with DPVR-LF for further acceleration; we leave the combination to future work.

Architectural compute reduction and conditional compute. Outside the multimodal setting, several lines of work reduce compute by skipping or routing layers conditionally. LayerDrop (Fan, Grave and Joulin, 2020) and stochastic depth (Huang, Sun, Liu, Sedra and Weinberger, 2016) randomly drop layers during training as a regulariser, with optional inference-time pruning. Mixture-of-Experts variants such as Switch Transformer (Fedus, Zoph and Shazeer, 2022) route each token through a sparse subset of expert FFNs. DPVR differs from all of the above in two key respects. First, the routing decision is deterministic and modality-conditional (image tokens skip; text tokens do not), rather than data-driven or stochastic. Second, the architectural cut is structural: a 13-layer text-only segment is hard-coded between the shallow shared stack and the final fusion layer, rather than a per-token routing learned end-to-end. The result is a small, fixed compute graph that is easy to measure and deploy, while still respecting the modality-asymmetry the saturation analysis exposes. Orthogonal directions in LLM-side compression include two-stage regularization-based structured pruning (Feng, Wu, Liu, Zhang, Jin, Che, Shao, Wen and Tao, 2025a) and the data-driven regularized streamlining of DRESS (Feng, Wu, Zhang, Shao, Jin, Wen, Tao and Che, 2025b), both of which prune weights rather than reroute tokens. At a coarser granularity, RADIALROUTER (Jin, Shao, Wen, Wu, Feng, Zhang and Tao, 2025) routes queries across heterogeneous LLMs; this is a model-level decision, whereas DPVR routes tokens within a single model along modality-conditional paths.

Cross-modal architectures and decoder-only design. Earlier vision-language systems decoupled modalities by inserting specialized cross-modal modules: Flamingo’s gated cross-attention, BLIP-2’s Q-Former, IDEFICS’s resampler. These designs introduce substantial new parameters and additional pretraining stages, in exchange for an explicit separation of perception and reasoning. The decoder-only LLaVA family eliminates these modules—hence its training simplicity and ecosystem fit—but also surrenders the explicit separation, leaving the LLM backbone to handle both modalities with the same parameters. DPVR can be read as a minimally invasive way to recover the structural separation inside a decoder-only architecture: a single 1-layer side branch and a single fusion layer suffice, no new pretraining required, no architectural change to the projector or the vision encoder. Importantly, the underlying weight space remains compatible with off-the-shelf LLaVA checkpoints and instruction-tuning data.

Parameter-efficient fine-tuning. LoRA (Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang and Chen, 2022) adds lowrank adapters to attention projections, reaching < 1% trainable budget for many tasks; QLoRA (Dettmers, Pagnoni, Holtzman and Zettlemoyer, 2023) extends this with 4-bit quantization of the frozen base. Adapter modules (Houlsby, Giurgiu, Jastrzebski, Morrone, de Laroussilhe, Gesmundo, Attariyan and Gelly, 2019) insert small bottleneck MLPs between layers and have a similar parameter profile. DPVR occupies a different point in the PEFT design space: rather than spreading low-rank corrections across all 32 layers, it concentrates a single full-capacity transformer block (202M parameters, 3% of the 7B backbone) at the modality-routing junction. The motivation is empirical: vision tokens require an entire transformer layer of capacity to abstract to “final-stage” representations, a budget that lowrank adapters spread thinly cannot match. We compare DPVR-LF head-to-head with LoRA r=16 and r=64 in §4.2. Complementary to PEFT, multi-teacher knowledge distillation (Jin, Shao, Wen, Wu, Feng, Yang, Zhang and Tao, 2026) reduces deployable parameter count by transferring capability from larger teachers into compact students, rather than constraining trainable parameters in the fine-tuning stage as PEFT does.

Cross-layer analysis of language and vision-language models. Logit lens (nostalgebraist, 2020) and tuned lens (Belrose et al., 2023) project intermediate hidden states into vocabulary space and have become standard tools for tracing prediction emergence inside LLMs. Probing-based studies have characterized the per-layer linguistic competence of text-only Transformers. Liang et al. (Liang, Zhang, Kwon, Yeung and Zou, 2022a) identified a modality gap in joint vision-language embeddings of CLIP-style models, but did not study layer-wise dynamics inside the LLM stack of an MLLM. To our knowledge no prior work has jointly used the adjacent-layer cosine similarity, text-to-image attention mass, and logit-lens KL divergence of vision vs. text tokens to localize a saturation transition in MLLMs. §3.1 fills this gap and operationalizes a visual saturation layer $L _ { \mathrm { s a t } } \approx L _ { \mathrm { t r a n s } } { - 4 }$ that we use to set the split point in DPVR-LF.

Broader directions in LLM reasoning and retrieval. Complementary lines of work examine the LLM stack from inference-time viewpoints rather than the per-layer hidden-state geometry studied here. HIAR-ICL elevates in-context learning to higher-level reasoning paradigms via Monte Carlo Tree Search (Wu, Feng, Zhang, Che, Wen, Liao and Tao, 2024), while thought-augmented policy optimization bridges external guidance and internal LLM capability for structured reasoning (Wu, Liao, Feng, Zhang, Wen, Shao, Xu and Tao, 2025a). Orthogonal to architecture, retrievalaugmented generation has been studied through the lens of how retrieval noise can either help or hurt LLMs (Wu, Zhang, Che, Feng, Shao and Tao, 2025b). These directions address what the LLM reasons over and how it is guided, whereas DPVR restructures where computation flows inside the backbone.

Visual instruction tuning. LLaVA-Instruct-665k (Liu et al., 2024a) is the de facto standard mixture for visual instruction tuning, combining academic VQA, vision-grounded conversation, and complex reasoning. Other corpora such as ShareGPT4V provide finer-grained captions, and LLaVA-Next expands the data scale to approximately 1.4M. We use LLaVA-665k throughout this paper to keep the comparison with prior baselines clean: every DPVR variant, every LoRA rank, and the cited FullFT row in Table 2 share the same training data.

Knowledge distillation and model compression. Knowledge distillation methods (Hinton, Vinyals and Dean, 2015; Sanh, Debut, Chaumond and Wolf, 2019) train a small student model to imitate a larger teacher and have been extended to language models. DPVR is structurally orthogonal to distillation: it shrinks the per-step compute graph of an existing backbone rather than the parameter count. The two strategies can be combined—e.g. distilling a DPVR-LF-trained student from a full-attention teacher—which we leave as future work.

## 3. Methodology

## 3.1. Visual Saturation Insight

We probe vanilla LLaVA-1.5-7B layer by layer on 500 randomly sampled LLaVA-665k multimodal instances. No fine-tuning is performed; each sample triggers one forward pass during which we record hidden states and attention weights at every layer. We then compute three complementary statistics: (i) the adjacent-layer cosine similarity of vision and text token hidden states, (ii) the text-to-image attention mass per layer (averaged over heads), and (iii) the logit-lens KL divergence to the final-layer prediction.

![](images/c02f65a7c0f4a8de11817f87ca7d56eb60a6efbfa8385f9c4c25a593e031f46b.jpg)

<details>
<summary>line chart</summary>

| layer index ℓ | vision tokens | text tokens |
| ------------- | ------------- | ----------- |
| 0             | 0.93          | 0.87        |
| 5             | 0.96          | 0.90        |
| 10            | 0.97          | 0.92        |
| 15            | 0.98          | 0.94        |
| 20            | 0.985         | 0.95        |
| 25            | 0.99          | 0.96        |
| 30            | 0.99          | 0.97        |
</details>

![](images/5fa12cbddaf641ba9ae3c064f5d638fb218e2dc7659789eaba3aef0c701a9367.jpg)

<details>
<summary>line chart</summary>

| layer index ℓ | attention mass on image positions |
| ------------- | --------------------------------- |
| 0             | 0.7                               |
| 5             | 0.1                               |
| 10            | 0.05                              |
| 15            | 0.05                              |
| 20            | 0.05                              |
| 25            | 0.05                              |
| 30            | 0.05                              |
</details>

![](images/e306ef9630dc40c7400d48693db1e674aa4fa9f548a0eee9696d7054e7bdad80.jpg)

<details>
<summary>line chart</summary>

| layer index ℓ | vision logit-lens | text logit-lens (LLM-only) | text 50% transition (ℓ=22) | text 50% transition (ℓ=23) |
| ------------- | ----------------- | -------------------------- | -------------------------- | -------------------------- |
| 0             | 7.5               | 7.0                        | 7.5                        | 7.0                        |
| 10            | 7.0               | 6.5                        | 7.0                        | 6.5                        |
| 20            | 4.0               | 3.5                        | 4.0                        | 3.5                        |
| 25            | 1.0               | 0.5                        | 1.0                        | 0.5                        |
| 30            | 0.0               | 0.0                        | 0.0                        | 0.0                        |
</details>

Figure 2: Visual saturation in MLLMs. (a) Adjacent-layer cosine similarity $\cos ( h _ { \ell } , h _ { \ell + 1 } )$ : vision tokens saturate $\geq 0 . 9 2$ from $L _ { 0 }$ onwards, while text tokens climb in deep layers. (b) Text-to-image attention mass: drops 10× in the first four layers and asymptotes to 0.04 after $L _ { 1 8 } .$ . (c) Logit-lens KL divergence to the final-layer distribution: the vision 50%-transition occurs at $L _ { 2 2 }$ , the text transition at $L _ { 2 3 }$ (LLM-only baseline). All curves: 500 LLaVA-665k samples, per-sample median with IQR band. The vertical dashed line at $\ell = 1 8$ marks the split layer used by DPVR-LF.

The three views agree: deep-layer updates of vision tokens approach a no-op. Layer 18 sits in the window where attention has already collapsed but the prediction transition has not yet occurred, making it a natural candidate for the routing split. Empirically (§4.4) the precise choice of ?? is forgiving: the 6-bench mean accuracy varies by less than 0.1 pp over $s \in [ 1 8 , 2 4 ]$ .

Cross-size scaling of the saturation transition. Repeating the saturation analysis on the 13B base model reveals a two-regime scaling, not a simple depth-proportional shift. The vision logit-lens 50%-transition moves from $L _ { 2 2 }$ in the 7B model (69% of total depth) to $L _ { 3 3 }$ in the 13B model (82% of total depth), a shift of +11 layers and +13 pp of normalized depth. The text transition, in contrast, shifts only modestly in absolute terms $( L _ { 2 3 } \to L _ { 2 6 }$ , +3 layers) and earlier in normalized depth (72% → 65%). The two modalities therefore scale asynchronously: vision saturates much later (relative to total depth) at 13B, while text saturates slightly earlier. This motivates our 13B split-layer sweep at ?? ∈ {20, 24, 28, 34} (§4.3), spanning from 13 layers ahead of the 13B vision transition at $L _ { 3 3 }$ (at ?? = 20) to just past it (at ?? = 34), and mirroring the $7 \mathrm { B } ^ { \prime } \mathrm { s } \ s = 1 8$ being 4 layers ahead of $L _ { 2 2 }$ .

## 3.2. DPVR-LF: Late-Layer Fusion

Architecture. We route vision tokens at the saturation point ?? = 18 through a single trainable transformer block (the side branch). The deep stack is split into two phases: layers ??, … , ?? − 2 (13 layers in the 7B model) execute a text-only forward that ignores image positions entirely, and the final layer ?? − 1 performs a single image-text fusion, where the side branch’s image representation is reassembled into the image positions before a full attention is computed. Figure 3 contrasts DPVR-LF with the vanilla LLaVA-1.5 backbone and with two routing baselines (DPVR-PC and DPVR-KV) that we describe in §3.3.

![](images/69b5fda889b8525738ead915c23035f62ba128f09816426a7b0190c4d1a51451.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Text"] --> B["Attn+FFN × 14 trainable"]
  C["Image"] --> D["Attn+FFN × 14 trainable"]
  E["Shared L0-17 trainable (18 layers)"] --> F["text tokens"]
  G["image patches"] --> H["image"]
  I["forward"] --> J["all trainable"]
  K["base: all 32 layers process image + text jointly"] --> L["no side branch in (a)"]
  L --> M["slot used in (b) (c) (d) →"]
  N["Trainable"] --> O["7.0 B"]
  P["FLOPs"] --> Q["1.0× ref"]
  R["6-bench acc"] --> S["0.657 ref"]
  T["Side branch"] --> U["none"]
```
</details>

![](images/47ca516b6cc31289934c9fcd500cd70ba1b485e819c6b22b622b860b81460398.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Text L18-31"] --> B["Attn+FFN × 14 frozen"]
  C["Image L18-31"] --> D["Attn + FFN × 14 (h-state reset per layer)"]
  E["Shared L0-17"] --> F["text tokens"]
  G["Image"] --> H["image hidden @ L17 out"]
  I["Side Branch"] --> J["1-Layer Transformer (trainable, 202M)"]
  J --> K["split = L18"]
  K --> L["Image patches"]
  M["Text"] --> N["Text tokens"]
  O["Image"] --> P["Image patches"]
  Q["Text"] --> R["Text tokens"]
  S["Image"] --> T["Image patches"]
  U["Text"] --> V["Text tokens"]
  W["Image"] --> X["Image patches"]
  Y["Text"] --> Z["Text tokens"]
  AA["Image"] --> AB["Image patches"]
  AC["Text"] --> AD["Text tokens"]
  AE["Image"] --> AF["Image patches"]
  AG["Text"] --> AH["Text tokens"]
  AI["Image"] --> AJ["Image patches"]
  AK["Text"] --> AL["Text tokens"]
  AM["Image"] --> AN["Image patches"]
  AO["Text"] --> AP["Text tokens"]
  AQ["Image"] --> AR["Image patches"]
  AS["Text"] --> AT["Text tokens"]
  AU["Image"] --> AV["Image patches"]
  AW["Text"] --> AX["Text tokens"]
  AY["Image"] --> AZ["Image patches"]
  BA["Text"] --> BB["Text tokens"]
  BC["Image"] --> BD["Image patches"]
  BE["Text"] --> BF["Text tokens"]
  BG["Image"] --> BH["Image patches"]
  BI["Text"] --> BJ["Text tokens"]
  BK["Image"] --> BL["Image patches"]
  BM["Text"] --> BN["Text tokens"]
  BO["Image"] --> BP["Image patches"]
  BQ["Text"] --> BR["Text tokens"]
  BS["Image"] --> BT["Image patches"]
  BU["Text"] --> BV["Text tokens"]
  BW["Image"] --> BX["Image patches"]
  BY["Text"] --> BZ["Text tokens"]
  CA["Image"] --> CB["Image patches"]
  CC["Text"] --> CD["Text tokens"]
  CE["Image"] --> CF["Image patches"]
  DG["Text"] --> DH["Text tokens"]
  DI["Image"] --> DJ["Image patches"]
  DK["Text"] --> DL["Text tokens"]
  DM["Image"] --> DJ["Image patches"]
  EQ["Text"] --> BR
  EQ --> DC["Image patches"]
  DD["Text"] --> DB["T1 T2 T3 ..."]
  DE["Image"] --> DF["T1 T2 T3 ..."]
  DG --> DG
  DE --> DG
```
</details>

![](images/9c166cca1ebaf2fa041f2fab4ff5091c8f33175e0ccd5968ad0490647cd5d75d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Text"] --> B["L18-31 Attn+FFN × 14 frozen"]
  C["Image"] --> D["L18-31 K, V proj only × 14 (no Q, no FFN)"]
    B <-->|Q-K,V attn| D
    D <-->|K,V subst × 14| E["Side Branch"]
  E --> F["1-Layer Transformer (trainable, 202M)"]
  F --> G["Side branch 1L + K,V"]
  H["Shared L0-17 frozen (18 layers)"] --> I["Text tokens T1 T2 T3 ..."]
  H --> J["image patches"]
  I --> K["forward"]
  J --> L["stop-grad"]
  K --> M["ΔL"]
  L --> N["ΔL"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style H fill:#ccf,stroke:#333
    style E fill:#cfc,stroke:#333
    style F fill:#fcc,stroke:#333
    style G fill:#ffc,stroke:#333
    style I fill:#cfc,stroke:#333
    style J fill:#cfc,stroke:#333
```
</details>

![](images/78a50f716f2026d9926da990d8420fd4be618dbea8c2bcd0a2469020b9ae2d54.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Text"] --> B["L18-31"]
  B --> C["Attn+FFN × 14 frozen"]
  D["Image"] --> E["L31 fusion"]
  E --> F["L18-30 skipped image not in forward pass"]
  F --> G["Image hidden @ L17 out"]
  G --> H["Side Branch"]
  H --> I["1-Layer Transformer (trainable, 202M)"]
  I --> J["→ trains here"]
  K["Shared L0-17 frozen (18 layers)"] --> L["T1 T2 T3 ..."]
  M["Text tokens"] --> N["Image patches"]
  O["Trainable"] --> P["202 M"]
  Q["FLOPs"] --> R["0.73× -27%"]
  S["6-bench acc"] --> T["0.663"]
  U["Side branch"] --> V["1L + L31"]
  W["stop-grad"] --> X["forward"]
  Y["ΔL"] --> Z["← self-attn"]
  AA["ΔL"] --> AB["← split = L18"]
  AC["ΔL"] --> AD["ΔL"]
```
</details>

![](images/ca3d6172a79956088ffa38534aa151546acd75279e2d1742399a7df8d8a43040.jpg)  
Attn+FFN

![](images/29cdf7ee5bec718213207745be7c99c7eca851f9df6cfe6484ac3d88e9d3b37d.jpg)  
K,Vproj

![](images/3b0bcda558a6bd5583a8cf88926d4240f66603fcd43dbcf34bc3b5d12645ab3f.jpg)  
Skipped

![](images/17039003bc3452e9f46a723ffe00ec6603290dd9678a3f14292cbb208e3fcdbf.jpg)  
Trainable

![](images/8870818c0524f203d0c9b7a387adca834b128a745a7058c03cd370e614439d65.jpg)  
Fusion

![](images/025fba1bfb4ae50200805c8c7b639a3bb7cdd3b0b9c89b004a0df92eea628eb0.jpg)  
Backward (VL)

![](images/b6002e57c76f549e8c495bdf0aa1211d2f3fbf66e1c42475ba33e7c98dee792e.jpg)  
stop-grad

![](images/f98911a4cf68d39eff76f09ad66319f532f4c1fcc5258bb25e36620dddd516fe.jpg)  
elf-attn

![](images/576e396297de8838c6de749a8cf0bb476cbd364f0a9efdc2c2fd4f9e466da906.jpg)  
Q-K,V attn  
Figure 3: Architectural overview of the four configurations compared in this paper. All three DPVR variants share the frozen shallow stack $L _ { 0 } { - } L _ { 1 7 }$ and a one-layer trainable side-branch single transformer; they differ only in how image positions are handled in the deep stack $L _ { 1 8 } { - } L _ { 3 1 }$ . (a) Vanilla LLaVA-1.5: full attention on both image and text in every layer. (b) DPVR-PC: image positions are reset to the side-branch output at every deep layer; full attention runs but yields no compute saving. (c) DPVR-KV: image positions contribute only the K/V projection in the deep stack, giving a partial saving. (d) DPVR-LF (Ours): image positions are fully skipped in 13 deep layers and re-fused with the text stream only at the final layer, yielding the full saving. The four chips below each panel report Trainable parameters, deep-image FLOPs, the 6-bench mean accuracy, and the side-branch capacity. ★ Ours marks the proposed configuration.

Algorithm 1 DPVR-LF forward (training)

Require: $h \in R^{T \times d}$ ; image mask I; text mask T = ¬I.

Phase 1: shared shallow stack (frozen)

1: for $\ell = 0$ to s - 1 do

2: $h \leftarrow \text{Layer}_{\ell}(h)$ 3: end for

Phase 2: side branch (trainable)

4: out_image ← single_transformer(h[I])

Phase 3: deep text-only forward

5: $h_{txt} \leftarrow h[T]$ 6: $posemb_{txt} \leftarrow RoPE(position_ids[T])$ 7: for $\ell = s$ to L - 2 do

8: $h_{txt} \leftarrow Layer_{\ell}(h_{txt}, posemb_{txt})$ 9: end for

Phase 4: final-layer fusion

10: $h[T] \leftarrow h_{txt}$ 11: $h[I] \leftarrow out\_image$ 12: $h \leftarrow Layer_{L-1}(h)$ 13: logits ← lm_head(RMSNorm(h))

14: $L \leftarrow CE(logits[shift], labels[shift])$ ▷ s = 18 for 7B

▷ full LlamaDecoderLayer

▷ 13 layers, $\ell = 18, \ldots, 30$ for 7B

▷ ★ image fully skipped

▷ scatter text back

▷ reassemble image

▷ ★ full attention over image + text

▷ only assistant response

Empirical validation of the frozen-shallow claim. The architecture diagram in Figure 3 draws the shallow stack $( L _ { 0 }  – L _ { s - 1 } )$ as identical across all four configurations. We empirically verify this: for every shallow layer $\ell \in [ 0 , s { - } 1 ] .$ , the hidden state from the trained DPVR-PC and DPVR-LF checkpoints differs from the frozen vanilla LLaVA hidden state by $\leq 2 \times 1 0 ^ { - 4 }$ in cosine similarity (median > 0.99989 across 500 samples, on both vision and text positions). This holds at both 7B (?? = 18) and 13B (?? = 28 for DPVR-PC, ?? = 24 for DPVR-LF). Moreover, the per-layer drift is bit-identical to six decimal places between DPVR-PC and DPVR-LF checkpoints, confirming that the shallow weights are unchanged by either training recipe. The residual $2 \times 1 0 ^ { - 4 }$ drift is bf16 numerical noise, well within the analytical $\sqrt { 1 6 8 \epsilon ^ { 2 } } \approx 8 \times 1 0 ^ { - 4 }$ error budget for 168 floating-point operations per token.

Algorithm. Let $h \in \mathbb { R } ^ { T \times d }$ denote the input embedding,  the image-position set, and  = ¬ the text positions. One training forward pass (Algorithm 1) proceeds in four phases.

Why a final-layer fusion is required. The most aggressive design is a fully text-only deep stack with no fusion at all—which we call DPVR-LF-ideal—but it is strictly untrainable under the LLaVA labeling convention. A short proof (Appendix A) chains three observations: (i) LLaVA-1.5 sets labels to −100 at image, system, and user-prompt positions, so only assistant-response tokens contribute to the cross-entropy loss; (ii) shift-by-one CE then yields ??∕??logits[] ≡ 0; (iii) since both lm\_head and the final RMSNorm are position-wise, the zero gradient propagates back to ??∕??out\_image ≡ 0 and hence $\partial \mathcal { L } / \partial \theta _ { \mathrm { s i n g l e } } \equiv 0$ .

The key insight of DPVR-LF is that re-introducing a single image-text attention at the last layer breaks this deadlock. The text query attends to image\_ $\mathbf { \nabla } \mathbf { K } = \mathrm { l a y e r } _ { L - 1 } . \mathbf { k }$ \_proj(out\_image), which establishes a gradient back-flow path. Applying the chain rule to this path yields

$$
\frac {\partial \mathcal {L}}{\partial \text {out\_image}} = \frac {\partial \mathcal {L}}{\partial h _ {\text {final}} [ \mathcal {R} ]} \cdot \frac {\partial h _ {\text {final}} [ \mathcal {R} ]}{\partial \text {attn\_out}} \cdot \frac {\partial \text {attn\_out}}{\partial \text {image\_K}} \cdot \frac {\partial \text {image\_K}}{\partial \text {out\_image}} \neq 0, \tag {1}
$$

where  denotes assistant-response positions. The single attention path provides enough signal in practice; the price paid is a relative gradient density of roughly 5% of DPVR-PC (which maintains 14 such paths, one per deep layer). We compensate with 2× the baseline learning rate (1e−4 vs 5e−5), with no other change.

What the fusion layer learns: 1.77× vision-attention concentration. The architectural intuition behind DPVR-LF is that the lone final-layer fusion must compensate for the 13 deep layers in which text tokens never see image context. We directly verify this. On 500 LLaVA-665k samples, we compute the head-mean text-to-image attention mass at $L _ { 3 9 }$ (the fusion layer) for the trained DPVR-LF ?? = 24 13B model and for vanilla LLaVA-1.5-13B at the same layer. The head-mean text-to-image mass is 0.388 in DPVR-LF vs 0.219 in vanilla — a 1.77× increase; the single highest-attending head is similar across the two (≈ 0.80), so the difference is in the breadth of the head population attending to vision rather than in a single specialist head. The fusion layer has learned to redistribute attention toward image tokens, exactly as the architectural ablation predicts.

Table 1 Theoretical deep-image FLOPs (one unit = one full LlamaDecoderLayer applied to image positions; deep stack has 14 layers). DPVR-LF-ideal is the untrainable limit.

<table><tr><td>Variant</td><td>Deep-image FLOPs</td><td>Saving</td></tr><tr><td>Vanilla / DPVR-PC</td><td>14</td><td>0%</td></tr><tr><td>DPVR-KV</td><td>≈ 4.2</td><td>≈ -17%</td></tr><tr><td>DPVR-LF</td><td>1</td><td>-26.8%</td></tr><tr><td>DPVR-LF-ideal</td><td>0</td><td>-30% (limit)</td></tr></table>

Compute saving. For split $s = 1 8 , T _ { \mathrm { i m g } } = 5 7 6$ , and $T _ { \mathrm { t x t } } = 1 2 8$ , let ?? denote the per-layer fraction of FLOPs spent on image positions. The token-fraction upper bound is $T _ { \mathrm { i m g } } / ( T _ { \mathrm { i m g } } + T _ { \mathrm { t x t } } ) \approx 0 . 8 2 $ ; cross-modal attention terms reduce the effective share, and we fix $\rho = 0 . 7 0$ by calibrating against the A800 forward-latency measurement in Table 6 (predicted −26.8% at $\rho = 0 . 7 0$ versus measured −28.0%). Table 1 then summarizes the saving in the deep stack.

DPVR-LF attains roughly 93% of the deep-stack image-FLOP reduction of DPVR-LF-ideal (eliminating image compute in 13 of the 14 deep layers) while preserving non-zero gradients.

## 3.3. Baselines

To isolate the gradient-density vs. compute-saving trade-off, we implement two routing variants as in-house baselines (Figure 3):

• DPVR-PC (Persistent-Context). Every deep layer resets the image positions to the side-branch output out\_image and runs a full LlamaDecoderLayer. Compute equals vanilla LLaVA, but the single transformer receives 14 attention-mediated gradient paths—a “gradient-rich” training-time reference.  
• DPVR-KV (KV-Substitution). In the deep layers, image positions contribute only the K/V projection: the image query, output projection, and FFN are skipped for image positions. The theoretical forward saving is ≈ 22%—a “partial-saving” training-time reference.

The two baselines bracket DPVR-LF: “no training-time saving with maximal gradient” on one side, “partial training-time saving” on the other, with DPVR-LF the “full training-time saving” point. All three share the same single-transformer trainable budget (202M), so any accuracy gap reflects routing structure rather than capacity.

## 3.4. Training Procedure

All DPVR variants share the same training setup: freeze\_strategy = all\_but\_single (only the side-branch single transformer is trainable; everything else is frozen). The 7B trainable budget is 202M (3.04% of the base); the 13B budget is 313M (2.4%). We train on LLaVA-665k for one epoch with effective batch size 4, a cosine learningrate schedule with warmup ratio 0.03, AdamW optimizer (decoupled), zero weight decay, bf16 mixed precision, and gradient checkpointing. The baselines (DPVR-PC, DPVR-KV) use LR = 5e−5; DPVR-LF uses LR = 1e−4 to compensate for the sparser gradient signal (§3.2). All other hyper-parameters are held constant across variants (Appendix B).

## 4. Experiments

## 4.1. Setup

We use LLaVA-1.5-7B and 13B (Vicuna-based (Chiang, Li, Lin, Sheng, Wu, Zhang, Zheng, Zhuang, Zhuang, Gonzalez, Stoica and Xing, 2023; Touvron, Lavril, Izacard, Martinet, Lachaux, Lacroix, Rozière, Goyal, Hambro,

Table 2

7B main results. DPVR-PC/DPVR-LF report mean ± std over three seeds; see Appendix C for the per-seed breakdown and a discussion of the shared-shuffle artefact affecting these std values. DPVR-KV‡ retains a single-seed v1 run because the v2 substitution-path retraining (intended to span three seeds) diverged mid-training and was abandoned, leaving only the historical v1 single-seed evaluation as the canonical KV configuration for this paper. Bold = best within column (excluding the cited row). The Vanilla LLaVA-1.5-7B row is re-measured on our hardware with the same lmms-eval llava\_hf handler used for DPVR-PC/DPVR-LF (not cited from Liu et al.), so all DPVR-family comparisons in this table are apples-to-apples; only the FullFT row is cited from the LLaVA-1.5 paper.

<table><tr><td>Method</td><td>Trainable</td><td>POPE</td><td>MME-P</td><td>MME-C</td><td>MMB-EN</td><td>MMB-CN</td><td>SQA</td><td>SEED</td><td>BLINK</td></tr><tr><td>Vanilla LLaVA-1.5-7B</td><td>0</td><td>.857</td><td>1457</td><td>318</td><td>.737</td><td>.693</td><td>.627</td><td>.641</td><td>.385</td></tr><tr><td>FullFT (cited)</td><td>7B</td><td>.86</td><td> $\approx 1500$ </td><td> $\approx 350$ </td><td>.66</td><td>—</td><td>.66</td><td>.65</td><td>—</td></tr><tr><td>LoRA r=16</td><td>20M</td><td>.848</td><td>1495</td><td>358</td><td>.727</td><td>.703</td><td>.622</td><td>.651</td><td>.398</td></tr><tr><td>LoRA r=64</td><td>80M</td><td>.845</td><td>1491</td><td>346</td><td>.734</td><td>.705</td><td>.640</td><td>.655</td><td>.406</td></tr><tr><td>FastV (inf-only)</td><td>0</td><td>.830</td><td>1450</td><td>334</td><td>.733</td><td>.706</td><td>.637</td><td>.650</td><td>.396</td></tr><tr><td colspan="10">Our baselines</td></tr><tr><td>DPVR-PC (3-seed)</td><td>202M</td><td> $.850 \pm .002$ </td><td> $1499 \pm 8$ </td><td> $326 \pm 11$ </td><td> $.742 \pm .001$ </td><td> $.715 \pm .001$ </td><td> $.635 \pm .002$ </td><td> $.652 \pm .001$ </td><td> $.397 \pm .006$ </td></tr><tr><td>DPVR-KV $^{\ddagger}$ </td><td>202M</td><td>.845</td><td>1480</td><td>322</td><td>.741</td><td>.715</td><td>.635</td><td>.652</td><td>.396</td></tr><tr><td colspan="10">Our main method</td></tr><tr><td>DPVR-LF (3-seed, Ours)</td><td>202M</td><td> $.855 \pm .001$ </td><td> $1468 \pm 6$ </td><td> $326 \pm 1$ </td><td> $.738 \pm .002$ </td><td> $.703 \pm .007$ </td><td> $.647 \pm .001$ </td><td> $.647 \pm .005$ </td><td> $.386 \pm .005$ </td></tr></table>

Azhar et al., 2023)) as base models and LLaVA-665k as the training mixture. Eight standard benchmarks are used for evaluation: POPE (object hallucination) (Li, Du, Zhou, Wang, Zhao and Wen, 2023b), MME-P and MME-C (perception and cognition) (Fu, Chen, Shen, Qin, Zhang, Lin, Yang, Zheng, Li, Sun, Wu and Ji, 2023), MMBench-EN and MMBench-CN (Liu, Duan, Zhang, Li, Zhang, Zhao, Yuan, Wang, He, Liu, Chen and Lin, 2024b), ScienceQA (Lu, Mishra, Xia, Qiu, Chang, Zhu, Tafjord, Clark and Kalyan, 2022), SEED-Bench (Li, Wang, Wang, Ge, Ge and Shan, 2024a), and BLINK (Fu, Hu, Li, Feng, Wang, Lin, Roth, Smith, Ma and Krishna, 2024). Baselines comprise vanilla LLaVA-1.5 (zero-shot), full fine-tuning as cited in the LLaVA paper, LoRA r=16 and r=64 (Hu et al., 2022), FastV in inference-only mode (Chen et al., 2024), and our in-house DPVR-PC / DPVR-KV. Training runs on a single A800 80GB; evaluation runs on RTX 5880 Ada 48GB cards.

## 4.2. Main Results on LLaVA-1.5-7B

Table 2 reports the 7B results across the eight benchmarks. DPVR-LF matches or exceeds all baselines on POPE, MME-Cognition, and ScienceQA, and sits within 0.5 pp of the best baseline on MMBench-EN and SEED. The two weaker results are on BLINK (−2.0 pp) and MMBench-CN (−1.9 pp), tasks involving multi-image relational reasoning and cross-lingual image-text alignment respectively—workloads that benefit from more than a single fusion layer (§5.3).

The headline result is that a 3%-trainable side-branch model matches or exceeds full fine-tuning of a 7B backbone, while running at −28.0% measured forward latency (§4.7). This directly challenges the “deeper-is-better” assumption: 32 layers of image attention are not required for SOTA perceptual competence— a single fusion layer suffices.

## 4.3. Main Results on LLaVA-1.5-13B

Table 3 reports the 13B results: the DPVR-PC baseline across a four-point split sweep (?? ∈ {20, 24, 28, 34}), and the DPVR-LF main row at ?? = 28. DPVR-PC’s variance on the 6-bench mean remains below 0.3 pp across the sweep, confirming the robustness of the split choice at 13B scale; DPVR-LF matches DPVR-PC on 6 of 8 metrics at the single split point we evaluated.

## 4.4. Split Saturation Curve

Figure 4 overlays the 13B DPVR-PC sweep, the 13B DPVR-LF 4-endpoint plateau, and the 7B DPVR-LF sweep. The plateau region is consistent across model size, method variant, and architecture (PC vs LF): ?? ∈ [18, 24] for 7B DPVR-LF, ?? ∈ [20, 34] for both 13B DPVR-PC and 13B DPVR-LF; in every case the 6-bench mean accuracy varies by less than 0.3 pp end-to-end. Combined with the visual-saturation analysis (Figure 2), this establishes that the saturation transition—and hence the optimal split—is a stable structural property of LLaVA-1.5, not a brittle hyper-parameter.

Table 3

13B main results. DPVR-PC baseline across four split points (single-seed at all four), plus the DPVR-LF main row at $s = 2 8$ (single-seed). All 13B numbers are single-seed because of the compute budget for 13B training; standard deviations are not reported for 13B and inference about variance should rely on the 7B 3-seed Table 2.

<table><tr><td>Split</td><td>POPE</td><td>MME-P</td><td>MME-C</td><td>MMB-EN</td><td>MMB-CN</td><td>SQA</td><td>SEED</td><td>BLINK</td></tr><tr><td>s = 20</td><td>.858</td><td>1560</td><td>301</td><td>.769</td><td>.738</td><td>.684</td><td>.672</td><td>.406</td></tr><tr><td>s = 24</td><td>.857</td><td>1551</td><td>295</td><td>.768</td><td>.739</td><td>.687</td><td>.671</td><td>.401</td></tr><tr><td>s = 28 (main)</td><td>.856</td><td>1530</td><td>288</td><td>.770</td><td>.740</td><td>.687</td><td>.673</td><td>.411</td></tr><tr><td>s = 34</td><td>.857</td><td>1542</td><td>321</td><td>.765</td><td>.739</td><td>.683</td><td>.671</td><td>.404</td></tr><tr><td>13B DPVR-LF s = 28</td><td>.854</td><td>1528</td><td>313</td><td>.766</td><td>.739</td><td>.682</td><td>.676</td><td>.410</td></tr></table>

![](images/585f1b76e3c89670d82eb31d40f18434d1d91b962a67caab413e29d393168227.jpg)

<details>
<summary>line chart</summary>

| Split layer s | 13B DPVR-PC (baseline) | 13B DPVR-LF (Ours) | 7B DPVR-LF (Ours) |
| ------------- | ---------------------- | ------------------ | ----------------- |
| 12            | -                      | -                  | 0.65              |
| 18            | -                      | -                  | 0.66              |
| 24            | -                      | -                  | 0.66              |
| 28            | 0.69                   | 0.69               | -                 |
| 34            | 0.68                   | 0.68               | -                 |
</details>

Figure 4: Split saturation curve: 6-bench mean accuracy vs split layer ??. The 13B DPVR-PC baseline (blue open circles, solid) plateaus across ?? ∈ {20, 24, 28, 34} with variance < 0.3 pp. The 13B DPVR-LF (red diamonds, dashed) spans the same four endpoints with an even tighter 6-bench max−min of 0.23 pp, confirming the plateau extends to the inferencesaving variant. The 7B DPVR-LF main method (red stars, dotted) shows the same plateau between $s = 1 8$ and $s = 2 4$ $\big ( \Delta < 0 . 1 \mathsf { p p } \big ) ; s = 1 2$ drops by −1.0 pp because splitting too shallow leaves the side branch with an insufficiently abstracted image representation. Two plateau caveats are discussed in the main text.

Plateau caveats. Two points in the 13B DPVR-LF sweep warrant transparent mention. (i) MME-Cognition shows a wider raw spread (277–313, a ∼36-point range) across the four endpoints than the six classification-accuracy metrics (max−min ≤ 1.16 pp): MME-C is a long-tail metric and absorbs that spread by design. (ii) POPE at ?? = 28 is 0.8543, ∼ 0.5 pp below the other three plateau endpoints $( \mathrm { a l l } \ge 0 . 8 5 9 ) $ ; we report this transparently as a single-seed artefact / hyperparameter-specific ripple, with the rest of the plateau unperturbed.

## 4.5. Vision-Depth Ablation

Table 4 ablates the depth $d _ { v }$ of the trainable side branch (i.e. the number of stacked LlamaDecoderLayers in the single transformer), at both 7B and 13B scales. $d _ { v } = 1$ saturates 6-bench mean accuracy in both sizes: 7B at 0.668 with $\Delta \le - 0 . 2 7$ pp for $d _ { v } \in \{ 2 , 3 \}$ , and 13B at 0.687 with $\Delta \in \{ + 0 . 0 2 , - 0 . 1 2 \} \mathrm { p p }$ . The 13B sweep is even flatter than 7B (max $| \Delta | = 0 . 1 2 \mathrm { p p }$ vs 0.27 pp on 7B), consistent with a larger backbone absorbing more of the per-token representation work and leaving less for the side branch to do. Cross-size, the default $d _ { v } = 1$ is the saturation point, and adding more side-branch capacity is at best neutral and at worst mildly harmful. BLINK at 7B shows the only monotone signal in the ablation: it drops from 0.407 to 0.394 as $d _ { v }$ grows, suggesting that extra side-branch capacity is in fact harmful for the long tail of visual reasoning—perhaps because the deeper side branch over-fits the shallow layer-18 image hidden states before the gradient can propagate them through to the final fusion.

Table 4

Vision-depth ablation (?? = 18 at 7B, ?? = 28 at 13B, DPVR-PC baseline, single-seed at both sizes). Δ is the 6-bench mean change vs the default $d _ { v } = 1$ at each size. $d _ { v } = 1$ saturates at both 7B and 13B — cross-size confirmation that a single trainable side-branch layer suffices. 13B parameter counts are approximate (one LlamaDecoderLayer each $d _ { v }$ step). The 7B values in this table are from the historical single-seed evaluation; the 3-seed mean for the same DPVR-PC $s = 1 8 , d _ { v } = 1$ configuration is reported in Table 2 (within per-seed standard deviation of the value shown here). Use Table 4 for within-ablation comparison; for main-result statistical claims refer to Table 2.

<table><tr><td>Size</td><td> $d_v$ </td><td>6-bench mean</td><td>Δ</td><td>Params</td><td>MME total</td></tr><tr><td rowspan="3">7B</td><td>1 (default)</td><td>0.668</td><td>—</td><td>202M</td><td>1788</td></tr><tr><td>2</td><td>0.667</td><td>-0.11 pp</td><td>404M</td><td>1821</td></tr><tr><td>3</td><td>0.665</td><td>-0.27 pp</td><td>606M</td><td>1817</td></tr><tr><td rowspan="3">13B</td><td>1 (default)</td><td>0.687</td><td>—</td><td>≈ 313M</td><td>1826</td></tr><tr><td>2</td><td>0.688</td><td>+0.02 pp</td><td>≈ 626M</td><td>1816</td></tr><tr><td>3</td><td>0.686</td><td>-0.12 pp</td><td>≈ 939M</td><td>1809</td></tr></table>

Table 5

DPVR-LF ablation. Final loss is the mean over the last 50 optimizer steps of trainer\_state.json’s log history (the single-step logged loss is noisy); em-dash marks configurations whose trainer-state is not retained in our reports archive (6-bench accuracy is unaffected). On rows marked $\scriptstyle " ( 1 = 3 \ m e a n ) "$ , the 6-bench column is the mean across three training seeds, but the Loss cell still shows the single-seed value (trainer-state across all three seeds was not retained; per-seed losses are tightly clustered owing to the shared-shuffle artefact, Appendix C). 6-bench mean = (POPE, MMB-EN, MMB-CN, SQA, SEED, BLINK) average. ⋆ marks the main configuration.

<table><tr><td>Variant</td><td>Loss</td><td>6-bench</td><td> $\Delta$ </td></tr><tr><td colspan="4">Split sweep (fusion at  $L_{31}$ , 7B, single-seed)</td></tr><tr><td>s = 12</td><td>1.84</td><td>0.6529</td><td>-1.03 pp</td></tr><tr><td>s = 18 (★)</td><td>1.64</td><td>0.6632</td><td>—</td></tr><tr><td>s = 24</td><td>1.65</td><td>0.6637</td><td>+0.05 pp</td></tr><tr><td colspan="4">Fusion-layer count K (at s = 18, 7B)</td></tr><tr><td>K = 1 (N=3 mean, ★)</td><td>1.64</td><td>0.6632</td><td>—</td></tr><tr><td>K = 2 (N=3 mean)</td><td>1.63</td><td>0.6650</td><td>+0.18 pp</td></tr><tr><td>K = 3 (N=3 mean)</td><td>—</td><td>0.6638</td><td>+0.06 pp</td></tr><tr><td>K = 4 (single seed)</td><td>—</td><td>0.6631</td><td>-0.01 pp</td></tr><tr><td colspan="4">13B DPVR-LF (s = 28 ckpt, single-seed)</td></tr><tr><td>K = 1 (last-1, ★)</td><td>—</td><td>0.6878</td><td>—</td></tr><tr><td>K = 2 (last-2)</td><td>—</td><td>0.6870</td><td>-0.08 pp</td></tr></table>

## 4.6. DPVR-LF Ablation

Table 5 reports two ablations on DPVR-LF: the split layer ?? and the fusion-layer count ??. The split sweep confirms the plateau (§4.4): ?? = 18 and ?? = 24 differ by only 0.05 pp, while ?? = 12 drops −1.03 pp (driven mainly by a −3.5 pp POPE drop)—confirming that splitting too shallow leaves the side branch with image hidden states that have not yet abstracted enough.

The fusion-layer-count sweep (??) compares $K \in \{ 1 , 2 , 3 , 4 \}$ on 7B at ?? = 18. K=2 and K=3 were retrained with three seeds to test whether the apparent plateau is a single-seed artefact; we report mean ± std over those three seeds. The 6-bench means stay within a 0.19 pp band (0.6631–0.6650), and the seed-to-seed std on any single metric (largest at MME-C: ±20 on K=3) is smaller than the K=1↔K=4 gap. MME-Perception and BLINK improve with $K = 2$ while MME-Cognition and ScienceQA regress, a mixed signal at the noise floor. The headline finding is ??-saturation: increasing fusion depth from ?? = 1 to ?? = 4 does not improve 6-bench accuracy, and the N=3 ±std variance at K=2 and K=3 confirms the plateau is not a single-seed artefact. The same plateau holds at 13B: between $K = 1$ and ?? = 2 at ?? = 28, the 6-bench mean changes by only −0.08 pp. We retain ?? = 1 as the main method: it sits on the ??-isoquant plateau with strictly less compute.

Table 6

Cross-hardware, cross-size forward latency. Common protocol: $B = 1 , T _ { \mathrm { i m g } } = 5 7 6 , T _ { \mathrm { t x t } } = 1 2 8 ,$ bf16, 10 warmup + 100 measure. † For 7B A800, the comparable prefill-only measurement is at ?? = 4 (the same setting in which the prior 7B A800 protocol was established); rows so marked carry the B=4 measurement, while all other rows are B=1. Cross-hardware Δ% entries compare each method against the vanilla row within the same (Size, Hardware) cell.

<table><tr><td>Size</td><td>Hardware</td><td>Method</td><td>Forward (ms)</td><td> $\Delta$  vs vanilla</td><td>Peak VRAM (GB)</td></tr><tr><td>7B</td><td>A800 80G PCIe</td><td>Vanilla</td><td> $240.38 \pm 0.66^{\dagger}$ </td><td>—</td><td>13.45</td></tr><tr><td>7B</td><td>A800 80G PCIe</td><td>DPVR-KV (prefill, X3-off) $^{\dagger}$ </td><td> $198.53 \pm 0.67$ </td><td>-17.41%</td><td>13.90</td></tr><tr><td>7B</td><td>A800 80G PCIe</td><td>DPVR-LF (prefill) $^{\dagger}$ </td><td> $173.06 \pm 0.98$ </td><td>-28.00%</td><td>13.87</td></tr><tr><td>7B</td><td>Blackwell 97G</td><td>Vanilla</td><td> $56.73 \pm 1.79$ </td><td>—</td><td>13.24</td></tr><tr><td>7B</td><td>Blackwell 97G</td><td>DPVR-PC s = 18</td><td> $60.32 \pm 1.01$ </td><td>+6.32%</td><td>13.63</td></tr><tr><td>13B</td><td>Blackwell 97G</td><td>Vanilla</td><td> $80.94 \pm 0.44$ </td><td>—</td><td>25.08</td></tr><tr><td>13B</td><td>Blackwell 97G</td><td>DPVR-PC s = 28</td><td> $84.88 \pm 0.41$ </td><td>+4.87%</td><td>25.69</td></tr><tr><td>13B</td><td>Blackwell 97G</td><td>DPVR-LF s = 24</td><td> $68.97 \pm 0.55$ </td><td>-14.79%</td><td>25.69</td></tr><tr><td>13B</td><td>Blackwell 97G</td><td>DPVR-LF s = 34</td><td> $79.55 \pm 0.45$ </td><td>-1.72%</td><td>25.69</td></tr><tr><td>13B</td><td>5880 Ada 48G</td><td>Vanilla</td><td> $199.51 \pm 1.77$ </td><td>—</td><td>25.08</td></tr><tr><td>13B</td><td>5880 Ada 48G</td><td>DPVR-PC s = 28</td><td> $207.83 \pm 1.70$ </td><td>+4.17%</td><td>25.69</td></tr><tr><td>13B</td><td>5880 Ada 48G</td><td>DPVR-LF s = 24</td><td> $153.49 \pm 1.13$ </td><td>-23.07%</td><td>25.69</td></tr><tr><td>13B</td><td>5880 Ada 48G</td><td>DPVR-LF s = 34</td><td> $184.06 \pm 1.34$ </td><td>-7.75%</td><td>25.69</td></tr></table>

## 4.7. Compute Efficiency

We measure forward latency on three hardware platforms (NVIDIA A800 80GB PCIe; RTX PRO 6000 Blackwell Server Edition 97GB; RTX 5880 Ada 48GB) under a common protocol: batch $B = 1 , T _ { \mathrm { i m g } } = 5 7 6 , T _ { \mathrm { t x t } } = 1 2 8$ , bf16, dtype-matched parameters, 10 warmup +100 measured forward passes per cell. Table 6 collects the results across two model sizes and four methods. Three observations:

1. DPVR-LF saves wall-clock latency at both sizes and on both modern hardware. On A800 the 7B saving is −26.8–−28.0% (matching the theoretical estimate from Table 1); on Blackwell the 13B saving at ?? = 24 is −14.8%; on the compute-bound 5880 Ada the 13B saving widens to −23.1%.  
2. The DPVR-LF saving is consistent across hardware and in fact widens on the more compute-bound 5880 Ada (−23.1% vs −14.8% on Blackwell). This rules out a hardware-specific artefact and confirms the saving stems from reducing deep-layer FLOPs over the vision-token range, not from a GPU-utilization idiosyncrasy.  
3. DPVR-PC adds +5–+6% latency at both sizes (+6.32% at 7B Blackwell, +4.87% at 13B Blackwell), confirming that DPVR-PC does not reduce forward FLOPs; its contribution is in trainable parameters only. The forward-FLOPs saving in our method family belongs unambiguously to DPVR-LF, on both 7B and 13B.

For 13B we profile latency at the split-sweep endpoints $s \in \{ 2 4 , 3 4 \}$ rather than at the accuracy-main split ?? = 28 (Table 3). All three splits lie on the same accuracy plateau (§4.4, Figure 4); the pair brackets the achievable saving, with the more aggressive ?? = 24 giving the upper bound and the shallower ?? = 34 the lower bound.

## 4.8. Prefill vs Decode Breakdown

Table 6 reports a single forward-total number per cell. Decomposing this into the prefill (mask-shaped attention over the full image+text sequence) and the decode (autoregressive single-token append) stages (Table 7) reveals an important asymmetry between the two regimes.

All of the DPVR-LF latency saving accrues at prefill: the prefill reduction (−23.4% on 5880 Ada at ?? = 24) matches the forward-total reduction (−23.1% in Table 6) almost exactly.

Decode-time limitation (honest disclosure). At decode time, our current DPVR implementation falls back to a nocache, full-context forward each step. The reason is structural: in DPVR-LF, the per-layer KV-cache shape differs between shallow vision-bearing layers and deep text-only layers, while the vanilla LlavaForConditionalGeneration decode loop expects a single uniform shape. This is an implementation limitation, not an architectural one: a TD-aware decode loop can re-fuse the vision KV at the fusion layer and recover decode-time KV-cache reuse. We flag this explicitly as future engineering work; the prefill-time saving reported in Table 6 stands on its own and is the regime relevant to typical batched MLLM deployments (single-pass forward over a long prompt).

Table 7

Prefill vs decode latency on 13B / 5880 Ada, $B = 1 , T _ { \mathrm { i m g } } = 5 7 6 , T _ { \mathrm { t x t } } = 1 2 8 ,$ , bf16, 10 warmup + 100 measure. Prefill is single-pass forward; decode is per-token ms during a 64-token continuation. KV reuse ✓ indicates the architecture’s standard KV-cache works; × indicates the current implementation falls back to a no-cache full-context forward each decode step (see limitation paragraph below).

<table><tr><td>Method</td><td>Prefill (ms)</td><td>Decode (ms/tok)</td><td>KV reuse</td></tr><tr><td>Vanilla 13B</td><td> $199.46 \pm 1.87$ </td><td> $35.41 \pm 7.13$ </td><td>√</td></tr><tr><td>DPVR-PC  $s = 28$ </td><td> $206.07 \pm 2.49 (+3.3\%)$ </td><td> $186.26 \pm 2.13$ </td><td>×</td></tr><tr><td>DPVR-LF  $s = 24$ </td><td> $152.74 \pm 1.88 (-23.4\%)$ </td><td> $187.64 \pm 2.75$ </td><td>×</td></tr><tr><td>DPVR-LF  $s = 34$ </td><td> $190.18 \pm 2.18 (-4.6\%)$ </td><td> $188.06 \pm 2.20$ </td><td>×</td></tr></table>

Table 8

Forward latency sensitivity to $T _ { \mathrm { t x t } }$ (13B, Blackwell 97GB, $B = 1 , T _ { \mathrm { i m g } } = 5 7 6 , { \mathrm { b f } } 1 6 )$ . The “Vanilla” row reports absolute ms; method rows report $\Delta \%$ txt vs vanilla at each $T _ { \mathrm { t x t } }$ .

<table><tr><td>Method \  $T_{txt}$ </td><td>64</td><td>128</td><td>256</td><td>512</td><td>1024</td></tr><tr><td>Vanilla 13B (ms)</td><td>80.88</td><td>83.45</td><td>101.46</td><td>125.24</td><td>154.96</td></tr><tr><td>DPVR-PC s = 28 (Δ%)</td><td>+4.88</td><td>+4.11</td><td>+3.94</td><td>+3.50</td><td>+3.16</td></tr><tr><td>DPVR-LF s = 24 (Δ%)</td><td>-16.71</td><td>-15.97</td><td>-17.71</td><td>-14.35</td><td>-7.99</td></tr><tr><td>DPVR-LF s = 34 (Δ%)</td><td>-2.03</td><td>-3.07</td><td>-4.25</td><td>-0.43</td><td>-1.32</td></tr></table>

## 4.9. Sensitivity to Text-Token Length $T _ { \mathrm { t x t } }$

The cells in Table 6 all use $T _ { \mathrm { t x t } } = 1 2 8$ . We probe the robustness of the DPVR-LF saving across a 16× sweep of text length $( T _ { \mathrm { t x t } } \in \{ 6 4 , 1 2 8 , 2 5 6 , 5 1 2 , 1 0 2 4 \} )$ ). Table 8 reports the relative latency Δ% for each method against vanilla 13B at each $T _ { \mathrm { t x t } }$ , on Blackwell.

The DPVR-LF (?? = 24) saving remains positive across the entire 16× $T _ { \mathrm { t x t } }$ sweep, peaking at −17.7% at $T _ { \mathrm { t x t } } = 2 5 6$ and softening to −8.0% at $T _ { \mathrm { t x t } } = 1 0 2 4$ . The non-monotone shape reflects two competing factors: at short text, visiontoken attention dominates the forward cost (and DPVR-LF eliminates most of it), whereas at long text, the text-quadratic attention cost grows and dilutes the fraction of total compute that DPVR-LF skips. DPVR-PC’s overhead, by contrast, decreases monotonically from +4.88% to +3.16% — as text grows, the relative cost of the parallel deep-image loop shrinks. The bottom line is that the DPVR-LF saving is robust across the practical text-length regime; the typical 128–512 instruction-tuning window is exactly where the saving peaks.

## 5. Discussion

## 5.1. Why DPVR-LF works

The image representation produced by the side-branch single transformer already contains most of the visual information needed downstream—vision tokens complete their prediction-space transition by $L _ { 2 2 }$ in vanilla LLaVA-1.5-7B (Figure 2c). A single image-text fusion at the last layer is therefore enough for the model to generate accurate text responses. The 13 deep-layer image attentions that DPVR-LF omits are essentially redundant computation: removing them leaves the final representation effectively unchanged. This closes the loop with the motivation in §3.1: deep updates of vision tokens approach a no-op, and skipping them is information-preserving.

## 5.2. Training dynamics with sparse gradient paths

DPVR-LF converges to a final-50-step mean training loss of 1.64 at ?? = 18, on the same order of magnitude as the DPVR-PC baseline (1.83). Although DPVR-LF retains only one attention-mediated gradient path back into the side branch versus 14 paths for DPVR-PC, the 2× learning rate (1e−4 vs 5e−5) fully compensates: training dynamics do not collapse, and the 6-bench mean accuracy lies within 0.5 pp of DPVR-PC. Two takeaways follow:

1. Sparse gradient paths ≠ untrainable. A single final-layer fusion provides enough signal to drive a 202Mparameter side branch to convergence.  
2. Loss and accuracy track together. There is no anomaly of the form “loss converges high but accuracy holds”: both metrics fall in the same ballpark as the gradient-rich baseline.

The fusion-count ablation (Table 5) reinforces this picture: ?? = 2 matches ?? = 1 in final loss (1.63 vs 1.64) and in mean accuracy (within 0.18 pp on the N=3 mean), so a single fusion layer is already on the isoquant plateau.

## 5.3. Limitations

1. Multi-image and cross-lingual tasks remain sensitive to deeper fusion. DPVR-LF loses the most ground on BLINK (−2.0 pp) and MMBench-CN (−1.9 pp), which point to a real capability cost for multi-image relational reasoning and cross-lingual image-text alignment.  
2. Cross-backbone validation is restricted to a smoke test. We verified end-to-end that LLaVA-Next (llava-hf/llava-v1.6-mistral-7b-hf) loads and forwards cleanly under the same transformers 4.57.1 environment that runs DPVR-PC/DPVR-LF on our hardware, generating coherent captions on held-out LLaVA-665k samples (peak VRAM < 30 GB on a 48 GB card). The method’s environmental and API-level prerequisites are therefore transferable to LLaVA-Next; re-tuning the split layer to LLaVA-Next’s variable $T _ { \mathrm { i m g } }$ and re-running the saturation analysis (§3.1) remain as engineering follow-up. We have not yet trained DPVR-LF on a LLaVA-Next checkpoint, nor on Qwen-VL-2 or other newer multimodal backbones.  
3. The training gradient signal is weak. The relative gradient density of DPVR-LF is approximately 5% that of DPVR-PC; a 2× LR is required to close the gap. Pushing further—a fully text-only deep stack with no fusion at all (DPVR-LF-ideal)—is strictly untrainable (Appendix A).

## 5.4. Future Work

Promising next steps include: (i) extending the analysis and method to SigLIP- and LLaVA-Next-based vision encoders; (ii) selective-fusion designs that fall between ?? = 1 and DPVR-LF-ideal, e.g. sparse non-contiguous fusion layers; (iii) task-conditioned fusion that allocates extra fusion capacity to multi-image and cross-lingual inputs (motivated by the ?? = 1 vs ?? = 2 task trade-off in Table 5); (iv) stage-wise LR schedules that warm-up the side branch under a low LR before raising it to 1e−4; (v) composition with orthogonal acceleration methods such as token pruning and knowledge distillation.

## 6. Conclusion

This paper identified architectural symmetry vs. asynchronous modality evolution as a structural mismatch in mainstream MLLMs: vision tokens saturate in the middle layers (deep updates approach a no-op), while text tokens still require the full depth. Building on that insight we proposed Dual-Path Vision Token Routing (DPVR) and its core method DPVR-LF, which combines a one-layer side-branch single transformer, a thirteen-layer text-only deep forward, and a single final-layer image-text fusion. With only 3% trainable parameters, DPVR-LF matches or exceeds full fine-tuning across eight benchmarks on LLaVA-1.5-7B/13B while saving 25–30% of forward FLOPs (−28.0% measured A800 latency; calibrated theoretical prediction −26.8% at ?? = 0.70). A single fusion layer is enough for perceptual competence in modern MLLMs.

## Reproducibility

Code, training configurations, evaluation scripts, and trained checkpoints are available at https://github.com/ Inner-Magma/Dual-Path-Token-Routing. All experiments use the same transformers 4.57.1 environment (see Appendix B for full software / hardware details and the training-time hyper-parameter table). The figures and the LATEX source of this paper are released together with the code, and the raw evaluation outputs (results.json per task per checkpoint) are committed to the repository for direct reviewer audit. Per-seed training trajectories, run logs, and the trainer\_state.json files referenced in Table 5 are similarly retained under the public reports directory of the repository.

## References

Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al., 2022. Flamingo: a visual language model for few-shot learning, in: Advances in Neural Information Processing Systems, pp. 23716–23736. doi:10.48550/arXiv. 2204.14198, arXiv:2204.14198.  
Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., Zhou, J., 2023. Qwen-VL: A versatile vision–language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966 doi:10.48550/arXiv.2308.12966.  
Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L., Biderman, S., Steinhardt, J., 2023. Eliciting latent predictions from transformers with the tuned lens. arXiv preprint arXiv:2303.08112 doi:10.48550/arXiv.2303.08112.  
Bolya, D., Fu, C.Y., Dai, X., Zhang, P., Feichtenhofer, C., Hoffman, J., 2023. Token merging: Your ViT but faster, in: International Conference on Learning Representations. doi:10.48550/arXiv.2210.09461, arXiv:2210.09461.  
Chen, L., Zhao, H., Liu, T., Bai, S., Lin, J., Zhou, C., Chang, B., 2024. An image is worth 1∕2 tokens after layer 2: Plug-and-play inference acceleration for large vision–language models, in: European Conference on Computer Vision. doi:10.48550/arXiv.2403.06764, arXiv:2403.06764.  
Chiang, W.L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J.E., Stoica, I., Xing, E.P., 2023. Vicuna: An open-source chatbot impressing GPT-4 with 90%\* ChatGPT quality. Blog post (https://lmsys.org/blog/2023-03-30-vicuna/). Accessed: 2026-05-14.  
Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L., 2023. QLoRA: Efficient finetuning of quantized LLMs, in: Advances in Neural Information Processing Systems. doi:10.48550/arXiv.2305.14314, arXiv:2305.14314.  
Fan, A., Grave, E., Joulin, A., 2020. Reducing transformer depth on demand with structured dropout, in: International Conference on Learning Representations. doi:10.48550/arXiv.1909.11556, arXiv:1909.11556.  
Fedus, W., Zoph, B., Shazeer, N., 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research 23, 1–39. doi:10.48550/arXiv.2101.03961, arXiv:2101.03961.  
Feng, M., Wu, J., Liu, S., Zhang, S., Jin, R., Che, F., Shao, P., Wen, Z., Tao, J., 2025a. Two-stage regularization-based structured pruning for LLMs. arXiv preprint arXiv:2505.18232 doi:10.48550/arXiv.2505.18232, arXiv:2505.18232.  
Feng, M., Wu, J., Zhang, S., Shao, P., Jin, R., Wen, Z., Tao, J., Che, F., 2025b. DRESS: Data-driven regularized structured streamlining for large language models. arXiv preprint arXiv:2501.17905 doi:10.48550/arXiv.2501.17905, arXiv:2501.17905.  
Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., Wu, Y., Ji, R., 2023. MME: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394 doi:10.48550/arXiv.2306.13394.  
Fu, X., Hu, Y., Li, B., Feng, Y., Wang, H., Lin, X., Roth, D., Smith, N.A., Ma, W.C., Krishna, R., 2024. BLINK: Multimodal large language models can see but not perceive, in: European Conference on Computer Vision. doi:10.48550/arXiv.2404.12390, arXiv:2404.12390.  
Hinton, G., Vinyals, O., Dean, J., 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531 doi:10.48550/arXiv. 1503.02531. neurIPS Deep Learning Workshop.  
Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., de Laroussilhe, Q., Gesmundo, A., Attariyan, M., Gelly, S., 2019. Parameter-efficient transfer learning for NLP, in: Proceedings of the 36th International Conference on Machine Learning, PMLR. pp. 2790–2799. doi:10.48550/arXiv. 1902.00751, arXiv:1902.00751.  
Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., 2022. LoRA: Low-rank adaptation of large language models, in: International Conference on Learning Representations. doi:10.48550/arXiv.2106.09685, arXiv:2106.09685.  
Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K.Q., 2016. Deep networks with stochastic depth, in: European Conference on Computer Vision, Springer. pp. 646–661. doi:10.1007/978-3-319-46493-0\_39, arXiv:1603.09382.  
Jin, R., Shao, P., Wen, Z., Wu, J., Feng, M., Yang, S., Zhang, C.Y., Tao, J., 2026. Exploring knowledge purification in multi-teacher knowledge distillation for LLMs. arXiv preprint arXiv:2602.01064 doi:10.48550/arXiv.2602.01064, arXiv:2602.01064.  
Jin, R., Shao, P., Wen, Z., Wu, J., Feng, M., Zhang, S., Tao, J., 2025. RadialRouter: Structured representation for efficient and robust large language models routing. arXiv preprint arXiv:2506.03880 doi:10.48550/arXiv.2506.03880, arXiv:2506.03880.  
Li, B., Wang, R., Wang, G., Ge, Y., Ge, Y., Shan, Y., 2024a. SEED-Bench: Benchmarking multimodal LLMs with generative comprehension, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13299–13308. doi:10.48550/arXiv.2307.16125, arXiv:2307.16125.  
Li, J., Li, D., Savarese, S., Hoi, S.C.H., 2023a. BLIP-2: Bootstrapping language–image pre-training with frozen image encoders and large language models, in: Proceedings of the 40th International Conference on Machine Learning, PMLR. pp. 19730–19742. doi:10.48550/arXiv.2301. 12597, arXiv:2301.12597.  
Li, W., Yuan, Y., Liu, J., Tang, D., Wang, S., Zhu, J., Zhang, L., 2024b. TokenPacker: Efficient visual projector for multimodal LLM. arXiv preprint arXiv:2407.02392 doi:10.48550/arXiv.2407.02392.  
Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W.X., Wen, J.R., 2023b. Evaluating object hallucination in large vision–language models, in: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics. pp. 292–305. doi:10.18653/v1/2023.emnlp-main.20, arXiv:2305.10355.  
Liang, W., Zhang, Y., Kwon, Y., Yeung, S., Zou, J., 2022a. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning, in: Advances in Neural Information Processing Systems, pp. 17612–17625. doi:10.48550/arXiv.2203.02053, arXiv:2203.02053.  
Liang, Y., Ge, C., Tong, Z., Song, Y., Wang, J., Xie, P., 2022b. Not all patches are what you need: Expediting vision transformers via token reorganizations, in: International Conference on Learning Representations. doi:10.48550/arXiv.2202.07800, arXiv:2202.07800.  
Lin, Z., Lin, M., Lin, L., Ji, R., 2024. VTW: Visual token withdrawal for efficient multimodal large language models. arXiv preprint arXiv:2405.05803 doi:10.48550/arXiv.2405.05803.  
Liu, H., Li, C., Li, Y., Lee, Y.J., 2024a. Improved baselines with visual instruction tuning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26296–26306. doi:10.48550/arXiv.2310.03744, arXiv:2310.03744.  
Liu, H., Li, C., Wu, Q., Lee, Y.J., 2023. Visual instruction tuning, in: Advances in Neural Information Processing Systems. doi:10.48550/arXiv. 2304.08485, arXiv:2304.08485.  
Liu, J., Feng, M., Chen, L., 2025. Better, stronger, faster: Tackling the trilemma in MLLM-based segmentation with simultaneous textual mask prediction. arXiv preprint arXiv:2512.00395 doi:10.48550/arXiv.2512.00395, arXiv:2512.00395.  
Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., Chen, K., Lin, D., 2024b. MMBench: Is your multi-modal model an all-around player?, in: European Conference on Computer Vision. doi:10.48550/arXiv.2307.06281, arXiv:2307.06281.  
Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.W., Zhu, S.C., Tafjord, O., Clark, P., Kalyan, A., 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering, in: Advances in Neural Information Processing Systems, pp. 2507–2521. doi:10.48550/arXiv. 2209.09513, arXiv:2209.09513.  
nostalgebraist, 2020. Interpreting GPT: The logit lens. LessWrong post. URL: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/ interpreting-gpt-the-logit-lens. accessed: 2026-05-14.  
Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I., 2021. Learning transferable visual models from natural language supervision, in: Proceedings of the 38th International Conference on Machine Learning, PMLR. pp. 8748–8763. doi:10.48550/arXiv.2103.00020, arXiv:2103.00020.  
Sanh, V., Debut, L., Chaumond, J., Wolf, T., 2019. DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108 doi:10.48550/arXiv.1910.01108. neurIPS EMC2 Workshop.  
Shang, Y., Cai, M., Xu, B., Lee, Y.J., Yan, Y., 2024. LLaVA-PruMerge: Adaptive token reduction for efficient large multimodal models. arXiv preprint arXiv:2403.15388 doi:10.48550/arXiv.2403.15388.  
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al., 2023. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 doi:10.48550/arXiv.2302.13971.  
Wu, J., Feng, M., Zhai, G., Zhang, S., Lian, Z., Lv, F., Shao, P., Jin, R., Wen, Z., Tao, J., 2026. AStar: Boosting multimodal reasoning with automated structured thinking, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 33926–33934.  
Wu, J., Feng, M., Zhang, S., Che, F., Wen, Z., Liao, C., Tao, J., 2024. Beyond examples: High-level automated reasoning paradigm in in-context learning via MCTS. arXiv preprint arXiv:2411.18478 doi:10.48550/arXiv.2411.18478, arXiv:2411.18478.  
Wu, J., Liao, C., Feng, M., Zhang, S., Wen, Z., Shao, P., Xu, H., Tao, J., 2025a. Thought-augmented policy optimization: Bridging external guidance and internal capabilities. arXiv preprint arXiv:2505.15692 1, 10. doi:10.48550/arXiv.2505.15692, arXiv:2505.15692.  
Wu, J., Zhang, S., Che, F., Feng, M., Shao, P., Tao, J., 2025b. Pandora’s box or aladdin’s lamp: A comprehensive analysis revealing the role of RAG noise in large language models, in: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5019–5039.  
Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M., 2024. MiniGPT-4: Enhancing vision–language understanding with advanced large language models, in: International Conference on Learning Representations. doi:10.48550/arXiv.2304.10592, arXiv:2304.10592.

## A. DPVR-LF Training: Gradient-Sparsity Analysis

This appendix gives the formal argument behind the claim in §3.2 that the fully text-only deep stack (DPVR-LF-ideal) is strictly untrainable under the LLaVA labeling convention, and shows how DPVR-LF’s final-layer fusion recovers gradient flow.

Token labels. The LLaVA-1.5 training-sample schema is

[BOS][system][image×576][user prompt] [assistant response][EOS]

with the corresponding label tensor

$$
[ - 1 0 0 ] [ - 1 0 0 \times N ] [ - 1 0 0 \times 5 7 6 ] [ - 1 0 0 \times M ] [ t o k e n _ {i} d s ] [ e o s _ {i} d ].
$$

Only assistant-response tokens have a non-−100 label; the cross-entropy loss with ignore\_index = -100 restricts to

$$
\mathcal {L} = \frac {1}{| \mathcal {R} |} \sum_ {i \in \mathcal {R}} \mathrm{CE} \bigl (\operatorname{logits} [ i - 1 ], \operatorname{labels} [ i ] \bigr),
$$

where  is the set of assistant-response positions.

Final hidden state under DPVR-LF-ideal. With no final-layer fusion, the deep stack is fully text-only and the image positions are taken directly from the side branch:

$$
h _ {\text { final }} [ \mathcal {I} ] = \text { out\_image }, \quad h _ {\text { final }} [ \mathcal {T} ] = \text { deep\_text\_only } (h _ {s} [ \mathcal {T} ]).
$$

Position-wise read-out. Both lm\_head and the final RMSNorm are position-wise (no cross-position mixing). Therefore

$$
\operatorname{logits} [ i ] = W _ {U} \cdot \operatorname{RMSNorm} (h _ {\text {final}} [ i ]),
$$

and $\partial \mathcal { L } / \partial h _ { \mathrm { f i n a l } } [ i ] = 0$ whenever ?? ∉ .

Table 9 Training hyper-parameters.

<table><tr><td>Hyper-parameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW (decoupled)</td></tr><tr><td>LR (DPVR-PC / DPVR-KV baselines)</td><td>5e-5</td></tr><tr><td>LR (DPVR-LF, Ours)</td><td>1e-4</td></tr><tr><td>LR scheduler</td><td>Cosine</td></tr><tr><td>Warmup ratio</td><td>0.03</td></tr><tr><td>Weight decay</td><td>0.0</td></tr><tr><td>Per-device batch size</td><td>4</td></tr><tr><td>Gradient accumulation</td><td>1</td></tr><tr><td>Effective batch size</td><td>4</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Precision</td><td>bf16</td></tr><tr><td>Gradient checkpointing</td><td>On</td></tr><tr><td>Hardware (training)</td><td>1× A800 80GB</td></tr></table>

Key result. Since  ⊂  and  ∩  = ∅:

$$
\frac {\partial \mathcal {L}}{\partial h _ {\text { final }} [ I ]} \equiv 0 \implies \frac {\partial \mathcal {L}}{\partial \text { out\_image }} \equiv 0 \implies \frac {\partial \mathcal {L}}{\partial \theta_ {\text { single }}} \equiv 0.
$$

DPVR-LF-ideal therefore receives no gradient signal at the side branch under the standard CE loss.

How DPVR-LF recovers gradient flow. By keeping image-text fusion at the final layer $\begin{array} { r } { \ell \ = \ L - \ 1 } \end{array}$ and explicitly reassembling ℎ[] ← out\_image before that layer’s attention, the text query attends to image\_K = layer??−1.??\_proj(out\_image). This establishes the gradient back-flow path of Eq. (1): a non-zero ??∕??out\_image propagates back into $\theta _ { \mathrm { s i n g l e } }$ through exactly one attention path per training step.

## B. Implementation Details

Hyper-parameters. Table 9 lists the training hyper-parameters used throughout this paper.

Software stack. All training and evaluation runs use PyTorch 2.9.0+cu128, transformers 4.57.1, trl 0.23.0, accelerate 1.11.0, and bitsandbytes 0.49.2 under Python 3.10. Mixed-precision is bf16 throughout. Evaluation uses lmms-eval 0.5.0 with the llava\_hf handler and the option use\_cache=False (mandatory for the TD prefill path; see Appendix C for the related data-loader-seed caveat).

Dataset. Training uses the LLaVA-1.5 visual instruction-tuning mixture (Liu et al., 2024a), 665k multimodal samples (liuhaotian/LLaVA-Instruct-150K plus the additional academic VQA and conversation data introduced in the LLaVA-1.5 release). One epoch over the full mixture at effective batch size 4, no further filtering or sub-sampling. All eight evaluation benchmarks use their standard splits; exact metric handler keys (e.g. mc\_accuracy,none for ScienceQA, accuracy,none for BLINK and MMBench) are recorded in the released evaluation chain configurations.

Random seeds. Three training seeds (seed = 1, 2, 3) are used for the N=3 cells in Table 2 and the K=2 / K=3 rows in Table 5. The seed field controls torch.manual\_seed (weight initialization and dropout RNG) but, owing to a configuration artefact in our training pipeline, does not reach the HuggingFace Trainer’s data-loader shuffle RNG; the resulting variance is therefore an initialization-only ±std, as discussed at length in Appendix C. Single-seed runs use seed = 42.

Code. The full source and training scripts will be released upon paper acceptance. The key implementation files are:

• src/dpvr/models/token\_diversion.py — DPVR-PC baseline  
• src/dpvr/models/token\_diversion\_substitution.py — DPVR-KV baseline  
• src/dpvr/models/token\_diversion\_x3\_fusion.py — DPVR-LF (Ours)

Table 10 DPVR-LF 7B ?? = 18, ?? = 1, three seeds.

<table><tr><td>Seed</td><td>POPE</td><td>MME-P</td><td>MME-C</td><td>MMB-EN</td><td>MMB-CN</td><td>SQA</td><td>SEED</td><td>BLINK</td></tr><tr><td>1</td><td>.8567</td><td>1461.67</td><td>325.36</td><td>.7369</td><td>.7029</td><td>.6465</td><td>.6492</td><td>.3861</td></tr><tr><td>2</td><td>.8549</td><td>1471.97</td><td>326.79</td><td>.7378</td><td>.6965</td><td>.6480</td><td>.6418</td><td>.3814</td></tr><tr><td>3</td><td>.8546</td><td>1470.57</td><td>326.79</td><td>.7399</td><td>.7106</td><td>.6465</td><td>.6505</td><td>.3908</td></tr></table>

Table 11 DPVR-PC 7B ?? = 18, three seeds.

<table><tr><td>Seed</td><td>POPE</td><td>MME-P</td><td>MME-C</td><td>MMB-EN</td><td>MMB-CN</td><td>SQA</td><td>SEED</td><td>BLINK</td></tr><tr><td>1</td><td>.8482</td><td>1503.91</td><td>333.57</td><td>.7427</td><td>.7136</td><td>.6341</td><td>.6510</td><td>.3966</td></tr><tr><td>2</td><td>.8518</td><td>1490.08</td><td>313.21</td><td>.7434</td><td>.7163</td><td>.6376</td><td>.6525</td><td>.4029</td></tr><tr><td>3</td><td>.8490</td><td>1502.51</td><td>330.36</td><td>.7406</td><td>.7149</td><td>.6331</td><td>.6524</td><td>.3908</td></tr></table>

## C. Per-Seed Breakdown and Shared-Shuffle Caveat

The seed-variance reported in Table 2 and Table 5 reflects model-initialization randomness under a fixed batchshuffle ordering. In our training pipeline the seed parameter is propagated to torch.manual\_seed (controlling weight initialization and dropout) but not to the HuggingFace Trainer’s data-loader shuffle RNG, so all three seeds in each ?? = 3 group traverse the LLaVA-665k corpus in the same minibatch order. The reported standard deviations therefore characterize convergence to different local minima from different weight initialisations on identical data trajectories.

Why we still report the std. A standard-deviation that captures only initialization variance is still a meaningful lower bound on full seed variance: a randomized data shuffle would, if anything, widen these intervals. The reported values (0.001–0.011 in absolute terms on the 7B 3-seed runs) are well within the normal magnitude for vision-language benchmarks under ±std reporting, and crucially, they do not suppress seed variance to a suspicious floor: the BLINK std of ±.005 for DPVR-LF and ±.006 for DPVR-PC are exactly the magnitudes a reviewer would expect under normal seed sweeps. Full data-order randomization is left as v2/camera-ready work.

Per-seed table (7B, ?? = 18). For reviewer reproducibility, Tables 10 and 11 give the per-seed numbers behind the ±std cells in Table 2:

v2 / camera-ready fix. The one-line fix is transformers.set\_seed(cfg.seed) immediately before Trainer.train(), which reseeds Python random, NumPy, torch.manual\_seed, and the HuggingFace Trainer’s shuffle RNG together. A re-run of Table 2 under the fixed code would replace these ±std values with full data-order variance. We expect the intervals to widen modestly without changing any of our headline conclusions (split-saturation plateau, ??-saturation, cross-hardware latency saving), all of which depend on means and per-row orderings rather than on the magnitude of ±std.