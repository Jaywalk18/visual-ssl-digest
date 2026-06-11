# Echo-Memory: A Controlled Study of Memory in Action World Models

Wayne King1, Zeyue Xue2,∗, Yuxuan Bian3, Jie Huang2, Haoran Li2, Yaowei Li4, Yuwei Niu4, Sihan Xu8, Junhao Zhuang6, Haoyang Huang2, Nan Duan2

1The University of Hong Kong; 2Joy Future Academy, JD; 3The Chinese University of Hong Kong; 4Peking University; 5Fudan University; 6Tsinghua University; 7The Hong Kong University of Science and Technology; 8University of Michigan.

∗Project leader. †Project advisor.

## Abstract

We present Echo-Memory, a controlled study of memory mechanisms in action-conditioned world models. These models generate multi-segment videos from a first frame, text prompt, and camera-action sequence, but their central failure is often memory rather than local image synthesis: after the camera leaves and returns, the scene or salient object may silently change. Existing memory designs are hard to compare because gains are entangled with backbone, training, retrieval, and evaluation differences. Echo-Memory fixes the action-to-video interface and varies only how history is stored and read by the generator. Under a shared video diffusion backbone, optimizer, camera-action representation, sampler, and evaluation pipeline, we compare raw context, compression-based memory, spatial summaries with different read-out paths, and state-space recurrence. This matched matrix separates four otherwise conflated axes: capacity, compression, read-out, and recurrence. We also evaluate memory through a three-branch protocol: replay quality, in-domain loop revisit, and open-domain return probes. The branches routinely disagree, showing that replay fidelity is not a sufficient proxy for remembering a world. Three findings follow. Raw context is a strong capacity baseline and improves open-domain return far more than it improves replay metrics. Compactness is not a free substitute for capacity: aggressive spatial and hybrid-compression memories lose the salient evidence needed for return. Finally, block-wise state-space recurrence is the strongest open-domain return mechanism in our matrix, showing that the structure of implicit memory matters as much as the decision to use it. These results provide a compact protocol for studying memory in action world models beyond isolated replay metrics.

Date: June 9, 2026

Project Page: https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/

Code: https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-Memory

## 1 Introduction

A central frontier for video generation is the move from producing a single plausible clip to producing a controlled world rollout. An action world model receives a first frame, a text prompt, and a sequence of camera actions. It must generate the next chunk, then the next, and continue doing so while preserving geometry, object identity, and camera obedience across revisits. The visible failures are familiar: the camera returns to the starting pose but the scene has silently changed, the salient object is replaced by a plausible impostor, or a chunk boundary wipes away the accumulated context. At their root, these are not generic generation failures. They are memory failures, and they are the failure mode that decides whether a system has actually modelled a world rather than merely extrapolated a clip.

Recent work has proposed several ways to supply this missing memory. Context-based methods retrieve previous observations and append them as visual evidence for the current segment [24, 31, 57, 62]. Compressionbased methods reweight or shorten the retrieved history to reduce cost [5, 28, 58, 71]. Spatial memories replace the full temporal stack with compact scene tokens [20, 25, 55, 57, 68], whereas state-space variants carry history implicitly through recurrence [9, 34, 44, 64, 66]. The same pressure appears in plug-and-play memory tokens, multi-shot keyframe memories, real-time interactive world models, persistent-world editors, and efficient long-form video diffusion systems [3, 7, 11, 39, 40, 45–49, 51, 56, 61, 65, 70]. These studies make clear that memory is now a central design axis for action-conditioned video generation. They also make comparison difficult: reported gains are often entangled with changes in backbone, training recipe, context budget, sampling procedure, and evaluation protocol.

![](images/7dcf825b9094f201eea8260f80d0a7318d80892181b821e9d23821df791886b6.jpg)

<details>
<summary>text_image</summary>

Input
Text Descriptions
The scene is traditional...
History Information
Previous Observation & State
Action State
Camera Pose
Atomic Manipulation
</details>

![](images/6aea5132991c06ab049915ea0dce086c93d40a836de68280286dfc960db6d90d.jpg)

<details>
<summary>text_image</summary>

Memory
Context Memory
CaM StoryMem
OneStory
Spatial Memory
SPMem VMem
Spatia EvoWorld
</details>

![](images/d4d301cbafcb71d48b49ee2118984777cdce57f4b5e536bb856c0e7487f66ec8.jpg)

<details>
<summary>text_image</summary>

Design
Compression Memory
FAR, Framepack
Infinite-World
State Memory
RAD
VideoSSM
LCSSVWM
</details>

![](images/62d71037728ede80743b430c9dfc809625171288f4944da0c49ce75049186d4f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Chunk 1"] --> B["Memory Management"]
  B --> C["C2"]
  B --> D["Memory Management"]
  D --> E["C3"]
  F["F0"] --> G["Memory Evaluation"]
  H["F242"] --> I["Semantic Align Scene Align"]
    style A fill:#f9f,stroke:#333
    style B fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style F fill:#ccf,stroke:#333
    style H fill:#ccf,stroke:#333
    style I fill:#ccf,stroke:#333
```
</details>

Figure 1 Abstract teaser and workflow of Echo-Memory. Given a text description, historical observations, and the camera/action state, an action world model must generate chunk-wise video while carrying memory across revisits. The figure positions Context, Compression, Spatial, and State-Space families as representative designs for preserving a revisitable world.

This paper studies memory from a controlled, vision-centric perspective. Rather than introducing another memory module, Echo-Memory uses a fixed video diffusion-transformer backbone, data interface, cameraaction encoding, sampler, and evaluation pipeline, and varies only the memory representation. This setting turns memory design into a common interface question: what information is stored, how is it compressed, how is it read by the generator, and whether it can survive a return motion after the camera leaves the visible support. The resulting comparison is intentionally narrower than the space of all world models, but it exposes differences that are otherwise easy to hide behind unrelated system changes.

Several methodological concerns motivate this design. Larger raw context may improve performance simply because it exposes more evidence, not because it provides a better memory mechanism; compression can appear efficient when token cost is hidden inside an operator whose retained evidence is never measured; and in-domain trajectories can overstate memory quality, because familiar textures allow plausible local reconstruction even when object identity is lost under distribution shift. We therefore evaluate every memory row through the same three branches: ground-truth replay, in-domain loop closure, and open-domain return. This also forces a separation between storage and read-out: a memory can store evidence without making it accessible to the generator, and one of our spatial read-out ablations shows that this distinction changes replay quality without improving revisit consistency.

Our study is organized around four memory families and three evaluation branches. The main empirical results are as follows:

• Raw context is a strong capacity baseline. Increasing context from K 1 to K 20 improves the replay image bundle and raises open-domain VLM return from 12.25 to 58.63.  
• Compact memory is not automatically semantic memory. Spatial Memory is competitive on replay PSNR but weak on open-domain return, while hybrid compression loses much of the signal preserved

by simpler length compression.

• Read-out matters as much as storage. Storing tokens is insufficient unless the generator can use them at the moment of return; withheld and dedicated spatial read-out variants expose this difference.  
• Structured recurrence is the strongest current open-domain bias. Block-wise State-Space memory reaches an open-domain VLM score of 69.00 despite lower replay PSNR, illustrating why replay alone is not an adequate memory score.

The rest of the main paper gives a compact literature review, defines the shared interface, describes the three-branch evaluation, and reports the controlled comparisons. Extended related work, training details, and the full ablation analysis are provided in the appendix.

## 2 Related Work

Action world models. World models have been studied as latent imagination, predictive state dynamics, and joint-embedding prediction. Recent video and embodied systems move this idea into generated visual rollouts conditioned on prompts, camera motion, or agent actions. In this setting, memory is not only a cachemanagement problem: the model must preserve scene geometry, object identity, and action consistency after the camera leaves and later revisits a region.

Memory mechanisms. Existing long-horizon video systems typically use one of three memory interfaces. Retrieval methods append previous frames or features chosen by camera overlap, scene geometry, entity identity, or temporal proximity. Compression methods reduce the cost of history by weighting tokens, pooling temporal windows, packing frames, or replacing full frame stacks with compact summaries. Implicit methods carry history through recurrent or state-space computation instead of explicit visual tokens. These designs are usually compared under different backbones, budgets, schedules, and metrics, making it hard to isolate the memory mechanism itself.

Evaluation. Standard video metrics such as PSNR, SSIM, LPIPS, FID, and FVD remain useful health checks, but they do not directly measure revisit consistency. Recent action-world-model work has begun to report loop-closure and return metrics, while VLM-as-judge protocols provide a way to verify semantic scene and identity preservation in open-domain settings. Echo-Memory combines these ideas in a controlled matrix: Replay measures camera-following quality, while in-domain and open-domain return probes measure whether the same world survives after a leave-and-return motion. A fuller literature review is provided in section A.

## 3 Method: The Echo-Memory Design Space

This section formalizes the memory design space studied by Echo-Memory. The point is not a new backbone, but a clean factorization of how an action world model can represent the past: Context tokens, Compression operators, Spatial summaries, or a State-Space state. All rows share the same training path and the same evaluation path; the row name changes only the memory/context profile.

## 3.1 Preliminaries

Backbone. Let $\mathbf { x } _ { 1 : T } \in \mathbb { R } ^ { T \times H \times W \times C }$ be a target video segment of T frames at spatial resolution H×W . A pre-trained video diffusion-transformer [52, 53] models the conditional velocity field $v _ { \theta } \big ( \mathbf { z } _ { t } ; t , \mathbf { c } _ { \mathrm { t e x t } } , \mathbf { c } _ { \mathrm { c t x } } , \mathbf { c } _ { \mathrm { a c t } } \big )$ , where $\mathbf { z } _ { t }$ is the noised VAE-encoded latent at flow-matching timestep $t , \ \mathbf { c } _ { \mathrm { t e x t } }$ is a text embedding, $\mathbf { c } _ { \mathrm { c t x } }$ is a memory context, and $\mathbf { c } _ { \mathrm { a c t } } \in \mathbb { R } ^ { T \times 1 2 }$ is a per-frame relative-RT camera-action sequence. The single-stage training objective is the rectified-flow regression loss applied only on the target frames:

$$
\mathcal {L} (\theta) = \mathbb {E} _ {t, \mathbf {x}, \mathbf {c}} \left\| v _ {\theta} (\mathbf {z} _ {t}; t, \mathbf {c} _ {\mathrm{text}}, \mathbf {c} _ {\mathrm{ctx}}, \mathbf {c} _ {\mathrm{act}}) - v ^ {\star} \right\| _ {2} ^ {2}, \qquad v ^ {\star} = \mathbf {z} _ {1} - \mathbf {z} _ {0}. \tag {1}
$$

Per-frame VAE context. We use a per-frame VAE context representation, which produces one latent token group per video frame. This keeps the temporal dimension of $\mathbf { c } _ { \mathrm { c t x } }$ semantically aligned with the target tokens, and makes compression-based memory operations well defined as transformations on the temporal axis of a fixed-length latent stack.

Action representation. We use a 12-dimensional relative camera RT with 9 rotation entries and 3 translation entries, expressed in the reference frame of the first frame of the segment. Every action in this study is relative; absolute action encodings are out of scope.

Context-conditioned generation. For a current segment with start frame s and end frame $e ,$ the model receives the target segment together with a context set of K historical observations:

$$
\mathcal {C} _ {s: e} = \left\{\left(\mathbf {x} _ {f _ {k}}, \mathbf {p} _ {f _ {k}}\right) \right\} _ {k = 0} ^ {K - 1}, \quad f _ {0} = s, \quad f _ {1.. K - 1} \subset \{0, \dots , s - 1 \}, \tag {2}
$$

where $\mathbf { p } _ { f _ { k } }$ denotes the associated camera pose/action metadata. The first context element is the anchor observation for the current segment; the remaining K 1 elements come from a retriever  over historical frames. With probability $p _ { \mathrm { d r o p } } = 0 . 1$ we drop the retrieved frames and keep only the anchor, simulating a cold-start regime.

## 3.2 Memory Design Space

Echo-Memory treats memory as a pair of operations attached to an otherwise fixed video DiT and our design can be summerized as in 2.

![](images/80260456a926231de966f9066b3ae4f6146ca1c678ed42a3fffd0c521a226337.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["DiT Backbone"] --> B["Retrieval/Compress"]
  B --> C["Noisy Latent"]
  B --> D["Pre. Chunk w/ State"]
  E["DiT Backbone"] --> F["Geometry Information"]
  F --> G["Geometry Extractor"]
  F --> H["Noisy Latent"]
  F --> I["Pre. Chunk w/ State"]
  J["DiT Backbone"] --> K["SSM Update"]
  K --> L["Noisy Latent"]
  K --> M["Pre. Chunk w/ State"]
```
</details>

Figure 2 Overview of four representative approaches to memory in action world models. Under a shared video diffusion backbone and a shared camera-action interface, the approaches differ only in how historical information is stored and read. We compare four families throughout: Context, Compression, Spatial, and State-Space.

## 3.3 Concrete Variants

The representative rows instantiate the four families above under the same backbone and action interface: anchor-only I2V, raw Context at different lengths, Compression, Spatial Memory, and State-Space recurrence. Focused ablations then vary one mechanism at a time: Spatial read-out, Compression type, recurrence structure, or raw context length. The empirical tables introduce the exact rows where they are needed, so the method section can stay focused on the shared interface rather than on a separate registry.

Default configuration. Unless otherwise stated, every row uses relative-RT camera conditioning, a fixed flow-matching timestep shift, target-frame-only supervision, and the same inference-time memory profile as its training configuration.

## 3.4 A Unified Memory Interface

To keep the comparison fair, the training loop, sampler, and evaluation pipeline expose the same memory interface to every variant. Each row is specified by a small memory profile: storage type, context length, compression rule, read-out path, and recurrence structure when applicable. Mutual-exclusion constraints are enforced at the profile level; for example, the two State-Space instantiations cannot be enabled simultaneously, and length compression is kept separate from pure token weighting unless the row is explicitly marked as hybrid. This unified interface is what makes Sec. C a controlled comparison rather than a juxtaposition of independently tuned systems.

Table 1 Single-stage hyper-parameters shared by all variants. Only context length and memory module vary across rows.

<table><tr><td rowspan="13"></td><td colspan="2">Setting</td><td colspan="5">Value</td></tr><tr><td colspan="2">Backbone</td><td colspan="5">Video DiT (per-frame VAE)</td></tr><tr><td colspan="2">Resolution</td><td colspan="5">352 × 640</td></tr><tr><td colspan="2">Segment length</td><td colspan="5">81 frames</td></tr><tr><td colspan="2">Context length K</td><td colspan="5">{1, 5, 20}, default 5</td></tr><tr><td colspan="2">Memory module</td><td colspan="5">Context, Compression, Spatial, or State-Space</td></tr><tr><td colspan="2">Optimizer</td><td colspan="5">AdamW</td></tr><tr><td colspan="2">Learning rate</td><td colspan="5"> $5 \times 10^{-5}$  ( $1 \times 10^{-4}$  at K=1)</td></tr><tr><td colspan="2">GPUs</td><td colspan="5">8 A100-80G</td></tr><tr><td colspan="2">Total steps</td><td colspan="5">5k</td></tr><tr><td colspan="2">Target-frame-only supervision</td><td colspan="5">enabled</td></tr><tr><td colspan="2">Relative-RT action encoding</td><td colspan="5">enabled</td></tr><tr><td colspan="2">10% overlap-drop policy</td><td colspan="5">enabled</td></tr><tr><td></td><td>f0</td><td>f40</td><td>f80</td><td>f121</td><td>f161</td><td>f202</td><td>f242</td></tr><tr><td>GT</td><td><img src="images/363603ae1bc158624bef01e34498e54f970d61760f15fbf1b91a201ea4fcb346.jpg"/></td><td><img src="images/1fc6710f9de52b38a3367f8f2abf9a97096f0fd873676a6c544510fecbf40e74.jpg"/></td><td><img src="images/5b6c023aa880ec3bdc582b31ce486d1f0b3d912d0e71eba206590489a5626968.jpg"/></td><td><img src="images/7fd084e9bc3688b9cfb37c4d8590da8e6b6df727381a0828e2a4860c6874f236.jpg"/></td><td><img src="images/873e23fcee409dd59258421d79148180611f29040d3653ca507076bf8c2dc568.jpg"/></td><td><img src="images/0a5b2a632c33b9737cd4b7f1760ca9b3b8c36a3848b2b97e2a8bce19d3602c34.jpg"/></td><td><img src="images/69615d423afbc2031012d11493db7fa0459bc658e1e8a00b14efdabef85771c0.jpg"/></td></tr><tr><td>Ctx-5</td><td><img src="images/cf237185ad2dc788c47391fe08973b414f503ce77632473ed45ef9be78b218fa.jpg"/></td><td><img src="images/d4c4b189d1552f9e4faf129321caa6fee3e4f548c53ea6091e386fb3094d5841.jpg"/></td><td><img src="images/b0cb22d745fbb9ea32c50088bc901883a3d7f79490f27042f40d9cc37b610c75.jpg"/></td><td><img src="images/e804b5316439a027c88e30f2e47d214bd9f793abe184ace9969633cd78e41ed3.jpg"/></td><td><img src="images/4a49ab2d1c0ef8b6a06f097386141e6052b9cd55d98982063442cf4cd6d775c8.jpg"/></td><td><img src="images/9cc67a53118958418ea48fd03672b4b2b4bacf3783a667dedd2337eb937aba07.jpg"/></td><td><img src="images/a868b35696eca28272fadf059ac224bc53197d8f0050f7c0972045acf4bfd337.jpg"/></td></tr><tr><td>Ctx-20</td><td><img src="images/cf8c6a7956edf8f9ea70496110be8776da0266727c9f239e512250de136e58c2.jpg"/></td><td><img src="images/2c57088b7817a0043221767d358f0b0d7d586cd93e5f354d5125c4e95920e7d8.jpg"/></td><td><img src="images/7a7610b4c9924e7ed033fbf007e8510b929e41864941b8aa9edd31d2eaca2fe6.jpg"/></td><td><img src="images/674011591b87c74ae4dd140f137762a536de20b83d53b7dfa9b2d532b39070c3.jpg"/></td><td><img src="images/4457a8dd3c831f6bc7db75955a582819a937e7a5dca3d01b938836206a242937.jpg"/></td><td><img src="images/7abd8ec056be9664fe06ff494c60baf6d2e1e40369dcc8721ee97d7b9b2a7aaf.jpg"/></td><td><img src="images/e469f5d5921bf209703551682f74228f4b2d3818bae6c58c5a74b0be7890bb28.jpg"/></td></tr></table>

Figure 3 Replay progression on a fixed GT camera trajectory. The diagnostic samples compare generated multisegment replay against a dataset trajectory at matched time indices. The panel reveals where background structure, object layout, and boundary continuity begin to drift before the final revisit probes are run.

## 4 Training Protocol

All variants are trained under a single shared protocol. The backbone, optimizer, schedule, sampler, data interface, action representation, and evaluation path are fixed; only the memory profile changes. This is the main control that lets the later tables be read as storage, compression, read-out, or recurrence effects rather than as hidden changes in training recipe.

Data and actions. We train on the Context-as-Memory dataset, which provides long camera trajectories, per-frame poses, prompts, and source videos. Each target sample is an 81-frame segment at 352×640 resolution. The camera signal is a per-frame 12-dimensional relative-RT vector, expressed in the reference frame of the first target frame. Context frames are selected by a fixed field-of-view retrieval policy, and the same reference-frame convention is used during training and evaluation.

Replay diagnostics. During training we sample fixed replay cases at matched intervals. These samples use the same generation path as evaluation, including the same context retriever, RT-relative action constructor, and memory profile. The scalar denoising loss is therefore supplemented by a visual diagnostic that exposes boundary continuity, identity drift, and action-visual alignment before final evaluation. Full training and sampling details are deferred to section B.

Long-Horizontal Consistency  
![](images/c8200a97f9dff6b467b9c7165cc7af31fa2b880bd331982d28f66b320cdb92b3.jpg)  
Probe Quality Changes via Replay  
Metric: PSNR/SSIM

![](images/0ce518d79263ade71b12204b32d82addc45b5d5d482c4026ab4a6fef6dd8da6f.jpg)

In-Domain Consistency  
![](images/49f18742fb8a1705456bda9158ccf7941127d1a16089b31b5b2a6b49b0aedddd.jpg)

![](images/a2df140989f7c36bcf2c477af023c3e1e0a925fb302983a6a1c23b5606f23ad5.jpg)  
Probe Scene Changes via Revisit  
Metric: PSNR/VLM

Open-Domain Consistency  
![](images/2330538d76412636f7d211c6f97a1b98547891a4215a1214fbd7b7aff9eada46.jpg)  
Probe Object Changes via Revisit  
Metric: PSNR/VLM

![](images/73a5a39a516efb491559ab1c0e048ad01526eaaa719d0f8b107611848377440e.jpg)

Figure 4 Evaluation taxonomy used in the study. Replay measures long-horizon image quality under GT camera motion. In-domain and open-domain return probes measure whether visual evidence survives a leave-and-return trajectory.  
![](images/a0068d12d2da570b48b4acdf0efb907e641092843d04add3dc09d94e4d7aa56e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["01 Prompt bank\n20 held-out scene prompts"] --> B["02 Qwen Edit\ninsert identity anchor object"]
  B --> C["03 Variants\n8 first frames per prompt"]
  C --> D["04 Return pool\n20 × 8 held-out first frames"]
```
</details>

Figure 5 Qwen Edit construction of open-domain first frames. Each held-out environment prompt is edited into identity-anchored first frames, which seed the open-domain return probe.

The replay diagnostic is useful precisely because the scalar flow-matching loss is a weak indicator of memory behavior. Two runs can have nearly identical denoising loss while diverging sharply at chunk boundaries or during return motion. Conversely, a model with slightly worse loss can preserve identity better if its memory path carries useful evidence across segments. We therefore treat replay sampling as a fixed in-training protocol rather than as an informal visualization: the held-out first frames, action constructor, context retriever, and generation code path are all shared across variants.

This protocol also provides a bridge between training and evaluation. Replay samples do not replace the final return metrics, because they still follow dataset-backed camera trajectories. However, they expose the first point at which a memory mechanism becomes unreliable: a context row may drift only after several generated chunks, a spatial row may preserve background layout but lose object identity, and a state-space row may carry scene evidence while paying a local reconstruction cost. These qualitative differences motivate the three-branch evaluation used in the next section.

## 5 Evaluation

The central evaluation challenge is that action world models mix two capabilities that should not be collapsed into one score. A model must follow camera actions, but it must also preserve the scene after the camera leaves and later returns. We therefore report three branches: Replay, which follows dataset camera trajectories and reports PSNR/SSIM/LPIPS; in-domain return, which uses a GT-backed 180○ loop and reports pixel and VLM consistency; and open-domain return, which uses held-out edited first frames and a compact 45○ leave/return probe.

Samples. The in-domain split provides first frames, metadata prompts, GT poses, and GT images. This supports direct loop-closure PSNR and VLM checks for fine-grained scene and identity details. The opendomain split has no pixel-level GT trajectory, so we avoid treating pixel differences as geometric truth and instead use controlled first-frame probes scored by semantic verification.

Open-domain first-frame pool. We generate held-out first frames with Qwen Edit. For each of 20 gamestyle environment prompts, the edit model inserts a distinctive identity anchor while preserving the surrounding scene. Eight variants per prompt yield a 20 8 pool whose objects are easy to name, localize, and revisit.

Metrics. Replay uses mean PSNR/SSIM/LPIPS over three generated chunks. In-domain return matches mirrored frames along the outgoing and returning trajectory and averages the pairwise image scores. Opendomain return uses a visual-return PSNR proxy plus a Qwen3-VL-30B-A3B semantic score for Scene and Special Identity. The VLM prompt weights object appearance and presence more heavily than background plausibility:

![](images/ba13e6473771efa649cf58764d36b6512dc05cafa5cce6a62ce2d506db779782.jpg)

<details>
<summary>natural_image</summary>

Night scene of a traditional Chinese courtyard with illuminated gate and stone bridge, featuring a yellow bear statue in the foreground (no visible text or symbols)
</details>

![](images/3bae041de4933d9f0154ccb5cd857dd36f2f941e871feb809dcd89a1c05e1000.jpg)

<details>
<summary>natural_image</summary>

Orange teddy bear sitting on a wooden table in an urban plaza with buildings and trees in the background (no visible text or symbols)
</details>

![](images/fc61982d37395dd99ddea1c5583ec92a96c4e1ad5c53dd650584a88524559455.jpg)

<details>
<summary>natural_image</summary>

Illustration of a traditional Mozambique pyramid with a turquoise tree and misty mountains in the background (no text or symbols)
</details>

![](images/2aa4d4c47cc8feedd0fa3514575de23651b6e998f739f6397e73c29b8cbeae88.jpg)

<details>
<summary>natural_image</summary>

Red teddy bear standing on a stone path beside a stone gate, with a distant castle and greenery in the background (no text or symbols visible)
</details>

![](images/f8e7547806c90671edd09186546e57b409376652eb5063ab8b3e093945173174.jpg)

<details>
<summary>natural_image</summary>

A turquoise teddy bear sitting on a paved surface with industrial storage tanks and power lines in the background (no visible text or symbols)
</details>

![](images/7b740ef7d39bbbf4b9d19e07c08f89ec18228c376b70c9760a6bca1bed06e666.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a historic stone courtyard with a central fountain and arched colonnades, surrounded by trees and potted plants (no signage or text visible)
</details>

![](images/d364f805bf8ebdcec2d32dace55125d89bd6a23822a675f123234726be88b5c4.jpg)

<details>
<summary>natural_image</summary>

Colorful teddy bear statue in front of modern glass buildings and flowerbeds (no visible text or symbols)
</details>

![](images/6ffc286caf07fe93e7d5098fd85e7147a167efb7ca21bc99de20ff9e045c588a.jpg)

<details>
<summary>natural_image</summary>

Colorful teddy bear sitting on a sandy beach with palm trees and ocean in background (no text or symbols)
</details>

Figure 6 Open-domain revisit source panel. The 2×4 grid shows representative first-frame sources from the Qwenedit open-domain pool. The full pool is generated by editing distinctive toy-like objects into held-out game-style environments; a practical construction uses 20 scene prompts and 8 edited variants per prompt. These objects serve as deliberately simple identity anchors for the VLM judge: after the camera leaves and returns, the judge scores whether the same object appearance, subject presence, camera viewpoint, and background scene are preserved.

Table 2 Cross-judge agreement for open-domain scoring. Mean scores over a shared open-domain subset. ∆ is the mean absolute difference from Qwen3-VL-30B-A3B, and ρ is Pearson correlation with Qwen3-VL.

<table><tr><td>Judge</td><td>Mean score</td><td> $\Delta$  vs. Qwen</td><td> $\rho$  vs. Qwen</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>19.6±13.9</td><td>-</td><td>1.00</td></tr><tr><td>Claude Opus 4.6</td><td>20.7±11.1</td><td>3.1</td><td>0.93</td></tr><tr><td>GPT-5.5</td><td>21.1±11.4</td><td>2.7</td><td>0.94</td></tr><tr><td>Human</td><td>20.9±11.9</td><td>1.3</td><td>0.96</td></tr></table>

$$
0. 4 5 s _ {\mathrm{appearance}} + 0. 2 5 s _ {\mathrm{presence}} + 0. 2 0 s _ {\mathrm{view}} + 0. 1 0 s _ {\mathrm{scene}},
$$

with the 0, 5 weighted sum rescaled to 0, 100 . A cross-judge sanity check is reported in section 6.1.

## 6 Additional Evaluation Details

## 6.1 Robustness of the VLM Judge

Because the open-domain branch relies on a single VLM judge, we run a sanity check on whether the choice of judge changes our conclusions. The probe is a constrained recognition task: the judge compares the source frame with revisit-tail frames and decides whether the salient object, viewpoint, and scene are preserved. We re-score a stratified subset of open-domain cases with Qwen3-VL-30B-A3B, two stronger closed-source judges, and a human anchor, all receiving the same images and prompt. The alternative judges stay within a few points of Qwen3-VL on average, remain correlated above 0.90, and preserve the relative reading of the subset. Changing the judge therefore adds at most a small offset rather than reordering the memory profiles.

## 7 Results

The experiments are organized around the failure mode that motivates the paper: a model can generate plausible local video while still failing to remember the world at revisit time. Each table reports the same evidence bundle: Replay PSNR/SSIM/LPIPS (R-P/R-S/R-L), in-domain return PSNR/SSIM/LPIPS (ID-P/ID-S/ID-L), and open-domain VLM score (O-V). The first two branches verify camera-following and loop closure; the last branch is the strongest semantic memory stress test.

Table 3 Representative memory-family comparison. R-P/R-S/R-L denote Replay PSNR/SSIM/LPIPS; ID-P/ID-S/ID-L denote in-domain closure PSNR/SSIM/LPIPS; O-V denotes open-domain VLM.

<table><tr><td></td><td>R-P↑</td><td>R-S↑</td><td>R-L↓</td><td>ID-P↑</td><td>ID-S↑</td><td>ID-L↓</td><td>O-V↑</td></tr><tr><td>I2V baseline</td><td>10.03</td><td>0.398</td><td>0.534</td><td>10.32</td><td>0.291</td><td>0.643</td><td>12.25</td></tr><tr><td>Spatial Memory baseline</td><td>13.60</td><td>0.411</td><td>0.554</td><td>10.04</td><td>0.260</td><td>0.617</td><td>6.00</td></tr><tr><td>Compression weight-only</td><td>8.65</td><td>0.174</td><td>0.752</td><td>10.67</td><td>0.154</td><td>0.645</td><td>22.38</td></tr><tr><td>State-Space (legacy hybrid)</td><td>12.69</td><td>0.344</td><td>0.581</td><td>12.23</td><td>0.298</td><td>0.584</td><td>34.75</td></tr><tr><td>State-Space (block-wise)</td><td>9.59</td><td>0.282</td><td>0.698</td><td>11.95</td><td>0.280</td><td>0.535</td><td>69.00</td></tr><tr><td>Context learning, K=5</td><td>11.92</td><td>0.408</td><td>0.501</td><td>10.72</td><td>0.307</td><td>0.596</td><td>50.75</td></tr><tr><td>Context learning, K=20</td><td>12.54</td><td>0.449</td><td>0.496</td><td>11.07</td><td>0.359</td><td>0.543</td><td>58.63</td></tr></table>

![](images/dc427aa4d9183571e59a5173bd032db552fa36113532ebe1ec17dbf89886a2f3.jpg)

<details>
<summary>bar chart</summary>

| Method   | Normalized score |
| -------- | ---------------- |
| I2V      | 0.65             |
| Spatial  | 0.1              |
| Comp.    | 0.85             |
| SSM-L    | 0.25             |
| SSM-B    | 0.7              |
| Ctx-5    | 0.45             |
| Ctx-20   | 1.0              |
|        | 0.8              |
|        | 0.7              |
|        | 0.9              |
|        | 0.85             |
</details>

![](images/14bfbbe73c56327be3a170bf4db81f1546da42a495c32b6cad3412f96bd4a323.jpg)

<details>
<summary>scatterplot</summary>

| Replay bundle | Open-domain VLM |
| ------------- | --------------- |
| 0.0           | 22              |
| 0.25          | 68              |
| 0.7           | 34              |
| 0.85          | 50              |
| 0.9           | 58              |
| 0.65          | 12              |
| 0.9           | 6               |
</details>

![](images/e2f8875972efce9986ae0ec9c50bf07593d31bd495f5080b624ac0d403ff27ea.jpg)

<details>
<summary>bar chart</summary>

| Category | Return rank - replay rank |
| :--- | :--- |
| Spatial better | 4 |
| Ctx-20 | 1 |
| I2V | 1 |
| Ctx-5 | 0 |
| SSM-L | 0 |
| Comp. | -2 |
| SSM-B | -4 |
</details>

Figure 7 Replay is a health signal, not the final memory score. The normalized replay and return views disagree across memory families, showing that low-level trajectory fidelity does not determine semantic revisit consistency.

## 7.1 Memory Families

The headline comparison shows why replay alone is insufficient. Different columns select different winners: Spatial Memory has the best Replay PSNR, long raw Context has the best Replay SSIM/LPIPS, and blockwise State-Space has the strongest open-domain VLM score. This inversion means that memory quality is not the same thing as image quality under GT motion. Replay tells us whether the generator can follow an observed trajectory; open-domain return tells us whether it can put the same object and scene back after leaving the view.

This disagreement is not noise in the table; it is the central diagnostic signal. Replay rewards the model for matching a known camera trajectory and reconstructing visible regions with low distortion. A return probe removes part of that support. The model must leave the relevant evidence behind, generate intervening frames, and then reconstruct the same world state when the viewpoint comes back. The Spatial–State-Space inversion is therefore informative: Spatial Memory can improve local reconstruction, but the open-domain probe asks it to recover a distinctive object from a compact scene summary. Block-wise State-Space produces less faithful pixels under GT replay, but its recurrent state appears to preserve the semantic identity needed for return.

The Context rows are the key capacity baseline. Moving from the anchor-only I2V baseline to K 5 raises open-domain VLM from 12.25 to 50.75, while K=20 reaches 58.63. The replay image bundle improves much less dramatically. This asymmetry suggests that raw history mainly helps semantic return rather than merely improving local pixel fidelity. Compact memories should therefore be compared against raw Context, not only against I2V.

The alignment plot turns the headline table into a diagnostic rather than a ranking. Replay catches broken generators that cannot follow a camera trajectory, so it remains a useful and inexpensive health signal. However, the rank shift from replay to return shows why it cannot be used as the final memory score: the strongest replay rows can still forget the salient object, while rows with weaker replay fidelity can carry scene

![](images/7b37fce5ad5b9a0af597a2409e739cdab5b9703740586bdf29892d97fef35c1b.jpg)

<details>
<summary>text_image</summary>

K=1
Ctx-5
Comp-L
Spatial
SS-Leg.
SS-Block
Revisit
GT
</details>

Figure 8 Representative open-domain revisit case. Matched revisit frames expose whether the salient object survives the leave-and-return motion. The qualitative panel illustrates why high Replay fidelity need not imply high semantic return.

Table 4 Spatial Memory read-out ablation.

<table><tr><td></td><td>Read-out</td><td>R-P↑</td><td>R-S↑</td><td>R-L↓</td><td>ID-P↑</td><td>ID-S↑</td><td>ID-L↓</td><td>O-V↑</td></tr><tr><td>Spatial baseline</td><td>default</td><td>13.60</td><td>0.411</td><td>0.554</td><td>10.04</td><td>0.260</td><td>0.617</td><td>6.00</td></tr><tr><td>Spatial inject-none</td><td>none</td><td>14.66</td><td>0.417</td><td>0.541</td><td>10.38</td><td>0.309</td><td>0.631</td><td>15.50</td></tr><tr><td>Spatial concat-text</td><td>text KV concat</td><td>13.44</td><td>0.412</td><td>0.572</td><td>9.35</td><td>0.272</td><td>0.614</td><td>10.25</td></tr><tr><td>Spatial cross-attn RO</td><td>separate cross-attn</td><td>13.52</td><td>0.409</td><td>0.557</td><td>8.49</td><td>0.232</td><td>0.664</td><td>17.12</td></tr></table>

evidence through the excursion.

Finding 1: Open-domain return is the most discriminative memory stress test. Replay and indomain return are necessary filters, but the open-domain VLM column is where memory families separate most clearly.

## 7.2 Spatial Read-out

Spatial summaries are efficient, but the ablation shows that efficiency does not imply semantic memory. The inject-none row obtains the best Replay bundle despite withholding the stored tokens from the generator, which indicates that replay improvements can come from regularization rather than usable memory. Dedicated cross-attention improves open-domain VLM to 17.12, but the family remains far below raw Context and block-wise State-Space. The bottleneck is therefore shared between storage bandwidth and read-out access: a compact grid can encode where things are, but it does not reliably encode which object must be regenerated after return.

This distinction is important because spatial memories are often attractive precisely when long raw context is too expensive. The replay numbers make them look promising: the family can reconstruct local trajectory frames well, and the inject-none row even improves the low-level image bundle. The return probe changes the interpretation. When the salient object must be regenerated from memory, a low-bandwidth scene grid under-specifies identity. The result is not that spatial memory is useless; it is that a scene summary needs object-aware retention and an explicit read-out path before it can replace raw visual evidence.

Finding 2: Spatial summaries are efficient but not yet reliable as semantic memory. The opendomain gap points jointly to storage bandwidth and read-out design; replay efficiency alone cannot identify which factor is binding.

The practical implication is that future spatial memories should be tested at two interfaces. First, the write path should be asked whether it retains object-level evidence rather than only coarse layout. Second, the read path should be asked whether the generator can select and use the relevant token at return time. A single replay score mixes these questions together. In our matrix, the dedicated read-out row improves the semantic column but does not close the gap, which suggests that both sides need to change: the storage grid should be more identity-aware, and the read-out should expose object evidence in a form the diffusion transformer can directly condition on.

Table 5 Compression ablation.

<table><tr><td></td><td>Mechanism</td><td>R-P↑</td><td>R-S↑</td><td>R-L↓</td><td>ID-P↑</td><td>ID-S↑</td><td>ID-L↓</td><td>O-V↑</td></tr><tr><td>Weight-only baseline</td><td>token weighting</td><td>8.65</td><td>0.174</td><td>0.752</td><td>10.67</td><td>0.154</td><td>0.645</td><td>22.38</td></tr><tr><td>Length r=2</td><td>temporal length</td><td>7.84</td><td>0.162</td><td>0.746</td><td>8.82</td><td>0.183</td><td>0.654</td><td>24.00</td></tr><tr><td>Length r=4</td><td>temporal length</td><td>9.88</td><td>0.183</td><td>0.714</td><td>9.53</td><td>0.163</td><td>0.617</td><td>43.25</td></tr><tr><td>Hybrid r=2 + weight</td><td>both</td><td>9.10</td><td>0.181</td><td>0.734</td><td>7.87</td><td>0.277</td><td>0.695</td><td>8.75</td></tr><tr><td>Hybrid r=4 + weight</td><td>both</td><td>9.63</td><td>0.215</td><td>0.730</td><td>8.81</td><td>0.156</td><td>0.625</td><td>9.00</td></tr></table>

Table 6 State-Space and raw-context ablations.

<table><tr><td></td><td>Setting</td><td>R-P↑</td><td>R-S↑</td><td>R-L↓</td><td>ID-P↑</td><td>ID-S↑</td><td>ID-L↓</td><td>O-V↑</td></tr><tr><td>State-Space</td><td>legacy hybrid</td><td>12.69</td><td>0.344</td><td>0.581</td><td>12.23</td><td>0.298</td><td>0.584</td><td>34.75</td></tr><tr><td>State-Space</td><td>block-wise</td><td>9.59</td><td>0.282</td><td>0.698</td><td>11.95</td><td>0.280</td><td>0.535</td><td>69.00</td></tr><tr><td>Context</td><td>K=1</td><td>10.03</td><td>0.398</td><td>0.534</td><td>10.32</td><td>0.291</td><td>0.643</td><td>12.25</td></tr><tr><td>Context</td><td>K=5</td><td>11.92</td><td>0.408</td><td>0.501</td><td>10.72</td><td>0.307</td><td>0.596</td><td>50.75</td></tr><tr><td>Context</td><td>K=20</td><td>12.54</td><td>0.449</td><td>0.496</td><td>11.07</td><td>0.359</td><td>0.543</td><td>58.63</td></tr></table>

## 7.3 Compression

Compression is not a monotone capacity dial. Weight-only compression preserves every temporal position, yet length compression with r=4 performs much better on open-domain VLM. A shorter but cleaner context can be easier for the generator to read than a full temporal stack with diffuse learned weights. The hybrid rows are the clearest failure case: pooling removes temporal evidence before weighting can decide which views matter, so the returned object is often under-specified. This result argues against treating all compression operators as interchangeable memory-saving tricks.

The compression table also explains why “same budget” comparisons are not enough. A method can preserve many tokens but still spread attention over frames that are only weakly useful for the return. Conversely, a more aggressive temporal reduction can keep fewer but cleaner slots, which makes the evidence easier for the generator to consume. The hybrid rows show a negative interaction between two individually plausible operators: first reducing temporal support, then learning weights over the reduced sequence, can erase the very view that should have anchored the return. A useful compressed memory therefore needs an objective for what to retain, not only a smaller token count.

Finding 3: Compression must preserve return-critical evidence, not just reduce cost. Length compression can beat token weighting, but hybrid compression shows that stacking budget-saving operators can destroy the identity evidence needed for revisit.

## 7.4 Implicit and Raw Memory

The State-Space comparison is the largest controlled open-domain jump in the matrix. The legacy hybrid row has better replay PSNR and strong in-domain PSNR, but block-wise recurrence nearly doubles opendomain VLM. This suggests that implicit memory is not a single design choice: the state must be structurally integrated enough that the network cannot bypass it when the camera leaves visible support.

The replay cost of block-wise recurrence is also informative. A non-bypassable state competes with local reconstruction capacity: the network must summarize history, update that summary, and use it during generation. This can hurt pixel metrics on GT trajectories, where the best strategy is often to focus on local visual fidelity. The same structural commitment becomes valuable in open-domain return, where the model has no pixel-aligned target and must rely on carried evidence. This trade-off is exactly why a single scalar would be misleading. A design that looks worse under replay can be better at the task that motivated memory in the first place.

![](images/701c4ebf3bdd75385b85e54d5fa0777619840d88228c7d65bbf380f752affeb2.jpg)

<details>
<summary>line chart</summary>

| Raw context length | Replay PSNR | In-domain PSNR | Open VLM judge score |
| ------------------ | ----------- | -------------- | -------------------- |
| K=1                | 10.0        | 9.5            | 10                   |
| K=5                | 12.0        | 10.0           | 50                   |
| K=20               | 12.5        | 13.0           | 60                   |
</details>

![](images/3e01442a71341e5a86018a77053d2aaedbf69aaaed1c2858ef1df942236831e9.jpg)

<details>
<summary>scatterplot</summary>

| Method       | Normalized GPU-hour / step | Normalized Replay PSNR |
| ------------ | --------------------------- | ---------------------- |
| SS-Leg.      | 0.05                        | 0.8                    |
| Ctx-5        | 0.15                        | 0.6                    |
| K=1          | 0.1                         | 0.1                    |
| SS-Block     | 0.0                         | 0.0                    |
| Ctx-20       | 0.7                         | 0.75                   |
| Comp-W outlier | 0.9                         | 0.8                    |
| Comp-L       | 1.0                         | 0.1                    |
</details>

Figure 9 Context scaling and training efficiency. Left: increasing raw context improves open-domain semantic return much more than the replay image bundle. Right: normalized replay PSNR versus normalized GPU-hour per step illustrates that replay efficiency and semantic memory are different optimization targets.

The raw-context rows calibrate the cost of compactness. Most of the open-domain gain appears by K=5, and K=20 adds further improvement while giving the best replay SSIM/LPIPS. This is a conservative but demanding baseline: specialized memory mechanisms should be judged by how much of the K=20 return benefit they retain. Under that bar, only block-wise State-Space exceeds raw Context on open-domain VLM.

The context curve also prevents an overly pessimistic reading of compact memory. Raw context is strong, but the gain is sublinear: most of the open-domain improvement arrives by K 5, with K 20 adding a smaller but still meaningful margin. This shape suggests that there is room for compact memories to match raw context if they retain the right evidence. The current matrix does not show that compression or spatial summaries are hopeless; it shows that generic compression and generic scene grids are not yet the right abstractions for return. The target is now clearer: preserve high-value views and object anchors while avoiding the linear cost of long raw context.

Finding 4: Block-wise recurrence is the strongest current open-domain memory bias. The gain over legacy recurrence shows that the structure of implicit memory matters, not merely the decision to make memory recurrent.

Finding 5: Raw context is the capacity baseline, and the gain it buys is overwhelmingly on revisit rather than on Replay. Compact and implicit memories should be judged by how much of the K 20 open-domain benefit they retain, not by how much they beat I2V.

## 7.5 Scaling and Efficiency

The context-scaling curve makes the capacity result more explicit. The open-domain VLM jump from K=1 to K=5 is much larger than the corresponding replay gain, and K=20 continues to improve return while only modestly improving low-level trajectory metrics. This slope mismatch is one of the cleanest signs that revisit consistency is a separate capability. If the problem were only local reconstruction, the replay and open-domain curves would move together. Instead, raw evidence primarily helps the model restore the same object and scene after the camera leaves.

The efficiency panel prevents the opposite over-correction. Raw context is a strong baseline, but it is not free: increasing K raises memory and training cost. Spatial summaries and compression remain worth studying because they target this cost, even though the current instantiations do not yet preserve enough semantic evidence. Block-wise State-Space occupies a different point in the trade-off: it is not the most replay-efficient row, but it is the only compact or implicit mechanism that exceeds raw Context K 20 on open-domain VLM. The practical goal is therefore not simply to maximize replay quality per GPU-hour, but to find mechanisms that preserve return-critical evidence at a cost below raw context.

This trade-off matters for how the matrix should be used. A researcher who only wants short GT-trajectory replay might choose a different row from a researcher who wants revisitable open-domain scenes. Spatial Memory is attractive in a replay-efficiency view because it provides strong local reconstruction with a compact representation. Context K=20 is attractive as a robustness baseline because it exposes the generator to many uncompressed observations. Block-wise State-Space is attractive as a memory mechanism because it carries scene evidence without appending a long explicit history. None of these rows dominates all axes, and that is the point: memory design is a multi-objective problem involving local fidelity, revisit consistency, and cost.

The current results also suggest how to design the next controlled matrix. First, compressed methods should be optimized against a return-aware signal, not only against replay loss or token count. Second, spatial methods should report object-level retention and read-out access separately. Third, recurrent methods should vary how deeply the state is integrated into the DiT rather than treating “state-space memory” as one monolithic option. Finally, every compact method should be compared against a raw-context scaling curve. Without that curve, it is too easy to mistake a win over I2V for a win over a strong memory baseline.

## 7.6 Cross-cutting Observations

Three patterns are consistent across the ablations. First, low-level image quality is necessary but not sufficient. Rows that fail Replay are not useful world models, but rows that win Replay can still forget the scene. Second, memory storage and memory read-out should be evaluated separately. The Spatial rows show that storing a compact representation does not guarantee that the generator can use it at the return point. Third, capacity baselines should be strong. The I2V baseline is useful as a floor, but the real reference for compact memories is raw Context K=20, because it shows what the same generator can do when given uncompressed evidence under the same retrieval and action interface.

These observations explain why the paper reports a matrix rather than a single winning model. The rows expose different failure modes: Spatial Memory preserves local replay structure but loses identity; Compression can make history cheaper but may delete the decisive view; legacy recurrence smooths trajectories without carrying enough open-domain evidence; block-wise recurrence carries that evidence but pays a replay-quality cost; and raw Context remains a stubborn capacity baseline. A memory mechanism for action world models should therefore be judged by how it moves along all three axes: trajectory following, in-domain return, and open-domain semantic return.

## 7.7 Design Implications

The controlled matrix suggests a concrete checklist for future memory modules. A new memory mechanism should first be compared against raw Context at multiple K values. This avoids the common but weak claim that a method improves over anchor-only I2V. The stronger question is whether the method retains the semantic return benefit of raw context while using fewer tokens, less compute, or a more stable state. Second, the mechanism should report write capacity and read-out capacity separately. Spatial Memory shows that a model can store a compact scene summary without making the right evidence available to the generator at return time. Third, compression should be evaluated by what evidence it preserves, not only by how many tokens it removes. A good compressor for world models is not merely a smaller context; it is a selective memory that keeps views and objects likely to become load-bearing later.

The results also argue for training signals that are closer to revisit quality. The current single-stage objective supervises target-frame denoising, while revisit consistency is measured after generation. This gap explains why replay can improve without semantic return improving. One natural direction is to add return-aware auxiliary supervision: object-level retention losses for spatial summaries, contrastive alignment between source and return frames, or VLM-guided selection signals for compression. Another direction is to expose recurrent states to explicit object or view prediction during training, so that the state is pressured to encode the evidence that will be needed after the camera leaves visible support.

Finally, the evaluation should remain multi-branch even when stronger models arrive. Better video backbones will raise replay quality and may make longer open-domain loops feasible, but they will not remove the distinction between following and remembering. If anything, stronger generators make the distinction more important: a highly capable model can hallucinate plausible replacements for missing objects, making qualitative failures harder to notice without a return-specific semantic check. The three-branch protocol is therefore not only a diagnostic for the present matrix; it is a guardrail for future systems that may look visually convincing while still failing to preserve the same world.

## 8 Discussion

We have presented Echo-Memory, a controlled study of memory in action world models. By fixing the backbone, optimizer, sampler, and evaluation protocol, the study makes the memory/context profile the main experimental variable. The resulting matrix separates a model’s ability to follow camera actions from its ability to preserve the world that those actions leave behind.

Main findings. Three lessons stand out. First, raw context is a strong capacity baseline: increasing context improves open-domain revisit more reliably than it improves replay metrics. Second, compactness alone is not enough: Spatial Memory and hybrid compression are efficient, but lose identity evidence without objectaware retention and read-out. Third, implicit memory depends on structure; block-wise recurrence preserves open-domain return much better than lightweight recurrent smoothing.

Evaluation lesson. Replay quality and revisit quality are not monotonically aligned. Several rows that look competitive under low-level replay metrics fall behind under semantic revisit scores. Selecting models by Replay alone would therefore favor the wrong family for interactive, revisitable world generation. This is why we report replay, in-domain return, and open-domain return as separate evidence rather than collapsing them into one scalar.

Limitations. The study is still bounded in scope. We train on a single dataset, so the ranking of mechanisms may change with noisier poses, different camera statistics, larger compute, or multi-stage curricula. The open-domain scores also rely on a VLM judge; our cross-judge check (section 6.1) is encouraging, but broader human calibration would make the comparison stronger. Finally, revisit quality is not yet available as a cheap training-time signal, so model selection still has to lean on imperfect proxies. The reported rows should therefore be read as representative points in a controlled matrix, not as the final implementations of each memory family.

Outlook. The next step is to make memory design explicitly revisit-aware: compression should preserve high-value object evidence, spatial summaries should expose object-level read-out to the generator, and structured recurrence should be evaluated as a first-class design axis. More broadly, Echo-Memory argues that world models should be measured not only by how smoothly they continue a trajectory, but by whether they can return to the same world after leaving it.

## References

[1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.  
[2] Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.  
[3] Zhaochong An, Menglin Jia, Haonan Qiu, Zijian Zhou, Xiaoke Huang, Zhiheng Liu, Weiming Ren, Kumara Kahatapitiya, Ding Liu, Sen He, Chenyang Zhang, Tao Xiang, Fanny Yang, Serge Belongie, and Tian Xie. Onestory: Coherent multi-shot video generation with adaptive memory. arXiv preprint arXiv:2512.07802, 2025.  
[4] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14834–14844, 2025.  
[5] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, Maneesh Agrawala, Lu Jiang, and Gordon Wetzstein. Mixture of contexts for long video generation. arXiv preprint arXiv:2508.21058, 2025.  
[6] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024.  
[7] Jingxi Chen, Zongxia Li, Zhichao Liu, Guangyao Shi, Xiyang Wu, Fuxiao Liu, Cornelia Fermuller, Brandon Y Feng, and Yiannis Aloimonos. First frame is the place to go for video content customization. arXiv preprint arXiv:2511.15700, 2025.  
[8] Kaijin Chen, Dingkang Liang, Xin Zhou, Yikang Ding, Xiaoqiang Liu, Pengfei Wan, and Xiang Bai. Out of sight but not out of mind: Hybrid memory for dynamic video world models. arXiv preprint arXiv:2603.25716, 2026.  
[9] Taiye Chen, Zihan Ding, Anjian Li, Christina Zhang, Zeqi Xiao, Yisen Wang, and Chi Jin. Recurrent autoregressive diffusion: Global memory meets local attention. arXiv preprint arXiv:2511.12940, 2025.  
[10] Taiye Chen, Xun Hu, Zihan Ding, and Chi Jin. Learning world models for interactive video generation. arXiv preprint arXiv:2505.21996, 2025.  
[11] Yabo Chen, Yuanzhi Liang, Jiepeng Wang, Tingxi Chen, Junfei Cheng, Zixiao Gu, Yuyang Huang, Zicheng Jiang, Wei Li, Tian Li, Weichen Li, Zuoxin Li, Guangce Liu, Jialun Liu, Junqi Liu, Haoyuan Wang, Qizhen Weng, Xuan’er Wu, Xunzhi Xiang, Xiaoyan Yang, Xin Zhang, Shiwen Zhang, Junyu Zhou, Chengcheng Zhou, Haibin Huang, Chi Zhang, and Xuelong Li. Teleworld: Towards dynamic multimodal synthesis with a 4d world model. arXiv preprint arXiv:2601.00051, 2026.  
[12] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.  
[13] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025.  
[14] Google DeepMind. Veo 3, 2025. URL https://deepmind.google/technologies/veo.  
[15] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for stateof-the-art vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 91–104, 2025.  
[16] Zicheng Duan, Jiatong Xia, Zeyu Zhang, Wenbo Zhang, Gengze Zhou, Chenhui Gou, Yefei He, Feng Chen, Xinyu Zhang, and Lingqiao Liu. Liveworld: Simulating out-of-sight dynamics in generative video world models. arXiv preprint arXiv:2603.07145, 2026.  
[17] Tuo Feng, Wenguan Wang, and Yi Yang. A survey of world models for autonomous driving. arXiv preprint arXiv:2501.11260, 2025.  
[18] Xiao Fu, Shitao Tang, Min Shi, Xian Liu, Jinwei Gu, Ming-Yu Liu, Dahua Lin, and Chen-Hsuan Lin. Plenoptic video generation. arXiv preprint arXiv:2601.05239, 2026.  
[19] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.  
[20] Samuel Garcin, Thomas Walker, Steven McDonagh, Tim Pearce, Hakan Bilen, Tianyu He, Kaixin Wang, and Jiang Bian. Beyond pixel histories: World models with persistent 3d state. arXiv preprint arXiv:2603.03482, 2026.  
[21] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2(3), 2018.  
[22] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.  
[23] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.  
[24] Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, Kalyan Sunkavalli, Feng Liu, Zhengqi Li, and Hao Tan. Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040, 2025.  
[25] JiaKui Hu, Jialun Liu, Liying Yang, Xinliang Zhang, Kaiwen Li, Shuang Zeng, Yuanwei Li, Haibin Huang, Chi Zhang, and Yanye Lu. Geometry-as-context: Modulating explicit 3d in scene-consistent video generation to geometry context. arXiv preprint arXiv:2602.21929, 2026.  
[26] Yuqi Hu, Longguang Wang, Xian Liu, Ling-Hao Chen, Yuwei Guo, Yukai Shi, Ce Liu, Anyi Rao, Zeyu Wang, and Hui Xiong. Simulating the real world: A unified survey of multimodal generative models. arXiv preprint arXiv:2503.04641, 2025.  
[27] Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, and Li Jiang. Memory forcing: Spatio-temporal memory for consistent scene generation on minecraft. arXiv preprint arXiv:2510.03198, 2025.  
[28] Sihui Ji, Xi Chen, Shuai Yang, Xin Tao, Pengfei Wan, and Hengshuang Zhao. Memflow: Flowing adaptive memory for consistent and efficient long video narratives. arXiv preprint arXiv:2512.14699, 2025.  
[29] Yann LeCun et al. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62, 2022.  
[30] Dohun Lee, Chun-Hao Paul Huang, Xuelin Chen, Jong Chul Ye, Duygu Ceylan, and Hyeonho Jeong. Memory-v2v: Memory-augmented video-to-video diffusion for consistent multi-turn editing. arXiv preprint arXiv:2601.16296, 2026.  
[31] JoungBin Lee, Jaewoo Jung, Jisang Han, Takuya Narihira, Kazumi Fukuda, Junyoung Seo, Sunghwan Hong, Yuki Mitsufuji, and Seungryong Kim. 3d scene prompting for scene-consistent camera-controllable video generation. arXiv preprint arXiv:2510.14945, 2025.  
[32] Lingen Li, Guangzhi Wang, Xiaoyu Li, Zhaoyang Zhang, Qi Dou, Jinwei Gu, Tianfan Xue, and Ying Shan. Cubecomposer: Spatio-temporal autoregressive 4k 360° video generation from perspective video. arXiv preprint arXiv:2603.04291, 2026.  
[33] Xinqing Li, Xin He, Le Zhang, Min Wu, Xiaoli Li, and Yun Liu. A comprehensive survey on world models for embodied ai. arXiv preprint arXiv:2510.16732, 2025.  
[34] Hansen Jin Lillemark, Benhao Huang, Fangneng Zhan, Yilun Du, and Thomas Anderson Keller. Flow equivariant world models: Memory for partially observed dynamic environments. arXiv preprint arXiv:2601.01075, 2026.  
[35] Kai Liu, Wei Li, Lai Chen, Shengqiong Wu, Yanhao Zheng, Jiayi Ji, Fan Zhou, Rongxin Jiang, Jiebo Luo, Hao Fei, et al. Javisdit: Joint audio-video diffusion transformer with hierarchical spatio-temporal prior synchronization. arXiv preprint arXiv:2503.23377, 2025.  
[36] Xiaoxiao Long, Qingrui Zhao, Kaiwen Zhang, Zihao Zhang, Dingrui Wang, Yumeng Liu, Zhengjie Shu, Yi Lu, Shouzheng Wang, Xinzhe Wei, et al. A survey: Learning embodied intelligence from physical simulators and world models. arXiv preprint arXiv:2507.00917, 2025.  
[37] Yawen Luo, Xiaoyu Shi, Jianhong Bai, Menghan Xia, Tianfan Xue, Xintao Wang, Pengfei Wan, Di Zhang, and Kun Gai. Camclonemaster: Enabling reference-based camera control for video generation. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–10, 2025.  
[38] Wufei Ma, Yu-Cheng Chou, Qihao Liu, Xingrui Wang, Celso de Melo, Jianwen Xie, and Alan Yuille. Spatialreasoner: Towards explicit and generalizable 3d spatial reasoning. arXiv preprint arXiv:2504.20024, 2025.  
[39] Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang. Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744, 2025.  
[40] Yihao Meng, Hao Ouyang, Yue Yu, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Hanlin Wang, Yixuan Li, Cheng Chen, Yanhong Zeng, Yujun Shen, and Huamin Qu. Holocine: Holistic generation of cinematic multi-shot long video narratives. arXiv preprint arXiv:2510.20822, 2025.  
[41] OpenAI. Sora, 2024. URL https://openai.com/sora.  
[42] OpenAI. Gpt-5, 2025. URL https://openai.com/gpt-5.  
[43] OpenAI. Sora 2: Video generation model, 2025. URL https://openai.com/sora.  
[44] Ryan Po, Yotam Nitzan, Richard Zhang, Berlin Chen, Tri Dao, Eli Shechtman, Gordon Wetzstein, and Xun Huang. Long-context state-space video world models. arXiv preprint arXiv:2505.20171, 2025.  
[45] Ryan Po, David Junhao Zhang, Amir Hertz, Gordon Wetzstein, Neal Wadhwa, and Nataniel Ruiz. Multigen: Level-design for editable multiplayer worlds in diffusion game engines. arXiv preprint arXiv:2603.06679, 2026.  
[46] Georgy Savva, Oscar Michel, Daohan Lu, Suppakit Waiwitlikhit, Timothy Meehan, Dhairya Mishra, Srivats Poddar, Jack Lu, and Saining Xie. Solaris: Building a multiplayer video world model in minecraft. arXiv preprint arXiv:2602.22208, 2026.  
[47] Selena Song, Ziming Xu, Zijun Zhang, Kun Zhou, Jiaxian Guo, Lianhui Qin, and Biwei Huang. Learning plugand-play memory for guiding video diffusion models. arXiv preprint arXiv:2511.19229, 2025.  
[48] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.  
[49] FSVideo Team, Qingyu Chen, Zhiyuan Fang, Haibin Huang, Xinwei Huang, Tong Jin, Minxuan Lin, Bo Liu, Celong Liu, Chongyang Ma, Xing Mei, Xiaohui Shen, Yaojie Shen, Fuwen Tan, Angtian Wang, Xiao Yang, Yiding Yang, Jiamin Yuan, Lingxi Zhang, and Yuxin Zhang. Fsvideo: Fast speed video diffusion model in a highly-compressed latent space. arXiv preprint arXiv:2602.02092, 2026.  
[50] InSpatio Team, Donghui Shen, Guofeng Zhang, Haomin Liu, Haoyu Ji, Jialin Liu, Jing Guo, Nan Wang, Siji Pan, Weihong Pan, Weijian Xie, Xiaojun Xiang, Xiaoyu Zhang, Xianbin Liu, Yifu Wang, Yipeng Chen, Zhewen Le, Zhichao Ye, and Ziqiang Zhao. Inspatio-worldfm: An open-source real-time generative frame model. arXiv preprint arXiv:2603.11911, 2026.  
[51] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, et al. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.  
[52] Alibaba Tongyi. Wan 2.5: Unified multi-modal video generation framework, 2025. URL https://tongyi.aliyun. com/wan.  
[53] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.  
[54] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.  
[55] Zun Wang, Han Lin, Jaehong Yoon, Jaemin Cho, Yue Zhang, and Mohit Bansal. Anchorweave: World-consistent video generation with retrieved local spatial memories. arXiv preprint arXiv:2602.14941, 2026.  
[56] Ruiqi Wu, Xuanhua He, Meng Cheng, Tianyu Yang, Yong Zhang, Zhuoliang Kang, Xunliang Cai, Xiaoming Wei, Chunle Guo, Chongyi Li, et al. Infinite-world: Scaling interactive world models to 1000-frame horizons via pose-free hierarchical memory. arXiv preprint arXiv:2602.02393, 2026.  
[57] Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory. arXiv preprint arXiv:2506.05284, 2025.  
[58] Xiaofei Wu, Guozhen Zhang, Zhiyong Xu, Yuan Zhou, Qinglin Lu, and Xuming He. Pack and force your memory: Long-form and consistent video generation. arXiv preprint arXiv:2510.01784, 2025.  
[59] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025.  
[60] Tianxing Xu, Zixuan Wang, Guangyuan Wang, Li Hu, Zhongyi Zhang, Peng Zhang, Bang Zhang, and Song-Hai Zhang. Ucm: Unifying camera control and memory with time-aware positional encoding warping for world models. arXiv preprint arXiv:2602.22960, 2026.  
[61] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, Song Han, and Yukang Chen. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.  
[62] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141, 2025.  
[63] Jiwen Yu, Yiran Qin, Haoxuan Che, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Hao Chen, and Xihui Liu. A survey of interactive generative video. arXiv preprint arXiv:2504.21853, 2025.  
[64] Yifei Yu, Xiaoshan Wu, Xinting Hu, Tao Hu, Yangtian Sun, Xiaoyang Lyu, Bo Wang, Lin Ma, Yuewen Ma, Zhongrui Wang, and Xiaojuan Qi. Videossm: Autoregressive long video generation with hybrid state-space memory. arXiv preprint arXiv:2512.04519, 2025.  
[65] Kaiwen Zhang, Liming Jiang, Angtian Wang, Jacob Zhiyuan Fang, Tiancheng Zhi, Qing Yan, Hao Kang, Xin Lu, and Xingang Pan. Storymem: Multi-shot long video storytelling with memory. arXiv preprint arXiv:2512.19539, 2025.  
[66] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T Freeman, and Hao Tan. Test-time training done right. arXiv preprint arXiv:2505.23884, 2025.  
[67] Yifu Zhang, Hao Yang, Yuqi Zhang, Yifei Hu, Fengda Zhu, Chuang Lin, Xiaofeng Mei, Yi Jiang, Bingyue Peng, and Zehuan Yuan. Waver: Wave your way to lifelike video generation. arXiv preprint arXiv:2508.15761, 2025.  
[68] Jinjing Zhao, Fangyun Wei, Zhening Liu, Hongyang Zhang, Chang Xu, and Yan Lu. Spatia: Video generation with updatable spatial memory. arXiv preprint arXiv:2512.15716, 2025.  
[69] Dingcheng Zhen, Xu Zheng, Ruixin Zhang, Zhiqi Jiang, Yichao Yan, Ming Tao, and Shunshun Yin. Soulxliveact: Towards hour-scale real-time human animation with neighbor forcing and convkv memory. arXiv preprint arXiv:2603.11746, 2026.  
[70] Jinsong Zhou, Yihua Du, Xinli Xu, Luozhou Wang, Zijie Zhuang, Yehang Zhang, Shuaibo Li, Xiaojun Hu, Bolan Su, and Ying-cong Chen. Videomemory: Toward consistent video generation via memory integration. arXiv preprint arXiv:2601.03655, 2026.  
[71] Tianrui Zhu, Shiyi Zhang, Zhirui Sun, Jingqi Tian, and Yansong Tang. Memorize-and-generate: Towards longterm consistency in real-time video generation. arXiv preprint arXiv:2512.18741, 2025.  
[72] Zheng Zhu, Xiaofeng Wang, Wangbo Zhao, Chen Min, Bohan Li, Nianchen Deng, Min Dou, Yuqi Wang, Botian Shi, Kai Wang, et al. Is sora a world simulator? a comprehensive survey on general world models and beyond. arXiv preprint arXiv:2405.03520, 2024.

## A Action World Models: Preliminaries and Related Work

The key components of an action world model are the video diffusion backbone, the action representation and its injection site, the memory subsystem, the retrieval policy, the training schedule, and the evaluation and benchmarking protocol. Each component has its own intricacies, and the interactions among them are non-trivial. Our study investigates them under a single backbone and a single training budget, with the explicit goal of isolating the contribution of memory.

World models have been studied as latent imagination, predictive state dynamics, and joint-embedding prediction [21, 22, 29]. Recent video and embodied systems extend this view from compact latent states to generated visual rollouts conditioned on prompts, camera motion, or agent actions [17, 26, 33, 36, 63, 72]. This shift makes memory a practical modeling question: a rollout must not only synthesize the next plausible clip, but also preserve scene and object evidence after the camera leaves and later revisits a region.

Diffusion-Transformer Video Backbones. A modern video generator typically follows the latent-diffusion recipe: an autoencoder maps each frame to a low-dimensional latent, and a denoising network learns the conditional velocity field over a stack of latent frames. Recent work has converged on diffusion-transformer backbones [52, 53], which scale favorably and natively expose per-frame token positions. Compared to U-Netbased 3D diffusion models, DiT backbones are particularly well suited to action-controlled generation: any per-frame signal (camera RT, action labels) can be attached to the corresponding frame token and propagated through self-attention, and the choice of injection site directly determines how strongly the signal reaches the spatial attention. Throughout this paper we use an off-the-shelf video DiT [53] as the common substrate; all design-space choices we study are local edits to this substrate. The broader video-generation ecosystem has converged on increasingly powerful diffusion-transformer backbones, including Imagen Video [23], Sora [41, 43], Veo [14], Wan [52, 53], Seedance [19], Waver [67], joint audio-video DiT [35], and camera-controllable generative re-rendering [4, 37]. Native multimodal models such as Emu3.5 [13] explore unified token-based modeling across images, video, and language. Cosmos is better described as a world-foundation-model platform for Physical AI, with video foundation models for prediction and conditional world generation rather than a single native multimodal model [1, 2].

Action and Camera Conditioning. A second axis, orthogonal to the memory subsystem, is the representation of action signals. Camera-controllable video diffusion has explored a spectrum of representations: world-frame extrinsics in context-based action-world models [62], Plücker embeddings, dense optical flow, and discretized action tokens for game-style control as in game-style world models. Time-aware positional encoding [60], cube-map and panoramic parametrizations [32], and iterative video editing on the same scene [30] further illustrate alternative action and viewpoint representations. We adopt the most straightforward representation, 12-dimensional relative-RT camera poses, and treat the injection site as a first-class design variable: post-attention, pre-norm, pre-QKV, or gated pre-QKV (section 3.2). This mirrors a finding from the multimodal-LLM literature: a seemingly small plumbing decision (e.g., where exactly the connector reads from) often dominates large architectural ones.

Memory for Long-Horizon Video Generation. Long-horizon video generation has increasingly treated memory as an interface between the current denoising window and observations that are no longer visible. One direct strategy is retrieval: previous frames or features are selected by camera overlap, entity identity, temporal proximity, or scene geometry, and then appended as context for the next chunk [24, 25, 31, 62, 68]. This line has been extended to streaming and interactive settings, where the system must decide what to keep in a fast cache, what to compress, and when to refresh evidence during rollout [10, 27, 48, 56]. Other work reduces the cost of carrying history by weighting tokens, packing temporal windows, pooling latent frames, or replacing a full frame stack with compact spatial summaries [5, 28, 57, 58]. Keyframe memories and multi-shot generation systems make a similar trade-off at the level of scenes or shots rather than individual tokens [3, 40, 65]. These methods differ in implementation, but they share the same underlying trade-off: the model must preserve enough visual evidence for a later return while keeping the current generation window affordable.

A complementary direction avoids explicit retrieval at inference time and carries history in a recurrent or learned state. State-space video world models propagate a causal hidden state through selected DiT blocks, whereas recurrent gates, test-time fast weights, flow-equivariant states, and plug-and-play memory tokens offer lighter or more modular variants [9, 34, 47, 64, 66]. Persistent-scene editors and spatial world memories similarly emphasize that what matters is not merely whether information is stored, but whether the generator can read it back when the viewpoint returns [11, 18, 45, 46, 50]. We use this literature to define a controlled comparison rather than a new memory module: all rows share the same backbone, camera interface, training budget, sampler, and evaluation protocol, while the memory representation and read-out path are varied.

Evaluation and Benchmarking. Static image and single-clip video metrics such as PSNR, SSIM, LPIPS, FID, and FVD remain the lingua franca of video generation, but on their own they do not capture longhorizon revisit consistency. Context-based action-world-model work has begun reporting specialized revisit metrics on rotation loops [24, 27, 48, 62], while the broader vision-language community has shifted toward model-as-judge protocols using strong VLMs (e.g., GPT-4V or Qwen3-VL-30B-A3B [54, 59]). Multi-shot storytelling benchmarks [3, 40, 65, 70], streaming long-video efficiency benchmarks [61, 69, 71], and out-ofsight dynamics evaluation [8, 16] further enrich this evaluation landscape. Spatial reasoning benchmarks and general VLM/LLM backbones also inform model-as-judge protocols for semantic verification [6, 12, 15, 38, 42, 59]. Our evaluation (section 5) combines both: long-horizon GT replay for PSNR/SSIM/LPIPS image quality, GT-backed in-domain $1 8 0 ^ { \circ }$ loop closure with revisit PSNR and VLM scoring, and open-domain $4 5 ^ { \circ }$ return probes measured by revisit PSNR proxies and VLM scores for Scene and Special Identity consistency.

How Echo-Memory differs. Our contribution is neither a new storage mechanism, nor a new retriever, nor a new conditioning trick. It consists of (i) a common interface into which all of the above can be plugged without modifying the training loop, sampler, or evaluation protocol; (ii) a matched ablation protocol that exercises the full design space under a single-stage training budget; and (iii) a three-branch evaluation stack that links trajectory replay, in-domain revisit, and open-domain revisit consistency. Rather than introducing yet another isolated model, the study is intended to make the interactions among storage, compression, read-out, recurrence, and evaluation visible within a single experimental matrix.

## B Training and In-training Diagnostics

By fixing the backbone, optimizer, schedule, and inference path across variants (section 3.4), the training setup is a constant of the study: the only thing that varies across the rows of table 8 is the memory module. This is what lets the later differences be read as storage, compression, read-out, or recurrence effects rather than as hidden protocol changes. We describe that constant below, and then promote in-training replay sampling from a debugging convenience to a diagnostic signal.

## B.1 Data

We train on the Context-as-Memory dataset [62], which contains real-world camera trajectories paired with per-frame poses and textual prompts. Each training sample is an 81-frame segment at 352×640 resolution, drawn from a long source video. For each sample, the data interface provides (i) the target segment $\mathbf { x } _ { s : e }$ , (ii) the corresponding 12-dim relative-RT camera trajectory, and (iii) a context buffer of K−1 historical frames retrieved from pre-computed field-of-view overlap labels keyed on the mid-frame of the segment.

Train/eval reference-frame alignment. A non-trivial implementation invariant is that the same notion of reference frame is used at training and evaluation time when retrieving context: training uses $r _ { \mathrm { t r a i n } } = { \left( s + e \right) } / 2$ , while in-domain evaluation uses $r _ { \mathrm { e v a l } } = N _ { \mathrm { f r a m e s } } / 2$ on videos that always start at frame 0. This alignment is not a headline ablation, but we still treat it as a hard protocol invariant.

## B.2 Single-stage Optimization

All variants in this study are trained in a single stage, of identical length, with the optimizer settings listed in Table 7. We deliberately avoid multi-stage curricula and stage-1 / stage-2 splits: in pilot experiments at this scale, we were unable to find a multi-stage schedule that consistently outperformed the single-stage one, and the simpler protocol makes the ablations easier to read.

Table 7 Single-stage hyper-parameters shared by all variants. Only the columns marked (varies) are perturbed across variants.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Backbone</td><td>Video DiT (per-frame VAE)</td></tr><tr><td>Resolution</td><td>352 × 640</td></tr><tr><td>Segment length T</td><td>81 frames</td></tr><tr><td>Context length K</td><td>(varies) - {1, 5, 20}, default 5</td></tr><tr><td>Memory module</td><td>(varies) - Context, Compression, Spatial, or State-Space</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $5 \times 10^{-5}$  ( $1 \times 10^{-4}$  at K=1)</td></tr><tr><td>Per-device batch / grad accumulation</td><td>1 / 1</td></tr><tr><td>GPUs</td><td>8 (A100-80G)</td></tr><tr><td>Total steps</td><td>5k</td></tr><tr><td>Timestep shift</td><td>15</td></tr><tr><td>Spike rejection threshold</td><td>15.0</td></tr><tr><td>Target-frame-only supervision</td><td>enabled</td></tr><tr><td>Flow noise shift</td><td>1.0</td></tr><tr><td>Relative-RT action encoding</td><td>enabled</td></tr><tr><td>Context placement</td><td>suffix</td></tr><tr><td>10% overlap-drop policy</td><td>enabled</td></tr></table>

## B.3 Replay Sampling Methodology

A practical observation from pilot experiments is that the scalar flow-matching loss is a poor progress indicator for action-world-model training. Two runs with visually different boundary behavior can have nearly indistinguishable denoising-loss curves; conversely, a run with a misconfigured camera-action condition can look stable on the loss until a return trajectory exposes the error. We therefore use replay sampling as an in-training methodology, not as an informal debugging video. The replay protocol fixes the sample pool, the action construction, and the generation code path, so changes in the sampled videos can be attributed to the memory profile under training, and boundary continuity, identity preservation, and action–visual alignment become directly comparable across variants well before final evaluation.

Sample pool and segment length. Every variant in Sec. C is configured to sample, at fixed step intervals, videos built from the same held-out first frames. Each generated segment contains 81 frames, matching the training target length, and multi-segment replay concatenates several such segments without changing the memory interface between segments. This keeps replay close to the actual inference regime, rather than measuring a short single-clip proxy.

Camera-trajectory construction. Replay actions are produced by the same relative-RT constructor used in evaluation. For ground-truth replay, the constructor reads the dataset camera trajectory and converts each frame into a 12-dim relative RT vector. For return probes, it synthesizes symmetric yaw motions, including 180○/360○ loops and the shorter 45○ open-domain revisit used when no GT pose is available. The important invariant is that the trajectory is generated before the model sees the target frames, so replay remains a test of action conditioning and memory rather than a post-hoc alignment procedure.

Shared generation path. Sampling uses the same generation procedure as evaluation, with the same context retriever, RT-relative actions, and memory profile. The sampled videos are therefore a faithful preview of evaluation behavior, not a divergent path.

Loss and metric granularity. The training loss in Eq. 1 is computed only on target frames within the current segment; context frames are conditioning evidence, not supervision targets. Replay diagnostics are stored at both per-frame and per-chunk granularity. Per-frame PSNR/SSIM/LPIPS reveal where drift begins, while chunk means and retention ratios summarize whether quality decays as memory must carry information farther from the anchor frame.

Table 8 Representative memory-family comparison. R-P/R-S/R-L denote Replay PSNR/SSIM/LPIPS; ID-P/ID-S/ID-L denote in-domain closure PSNR/SSIM/LPIPS; O-V denotes open-domain VLM.

<table><tr><td></td><td>R-P↑</td><td>R-S↑</td><td>R-L↓</td><td>ID-P↑</td><td>ID-S↑</td><td>ID-L↓</td><td>O-V↑</td></tr><tr><td>I2V baseline</td><td>10.03</td><td>0.398</td><td>0.534</td><td>10.32</td><td>0.291</td><td>0.643</td><td>12.25</td></tr><tr><td>Spatial Memory baseline</td><td>13.60</td><td>0.411</td><td>0.554</td><td>10.04</td><td>0.260</td><td>0.617</td><td>6.00</td></tr><tr><td>Compression weight-only</td><td>8.65</td><td>0.174</td><td>0.752</td><td>10.67</td><td>0.154</td><td>0.645</td><td>22.38</td></tr><tr><td>State-Space (legacy hybrid)</td><td>12.69</td><td>0.344</td><td>0.581</td><td>12.23</td><td>0.298</td><td>0.584</td><td>34.75</td></tr><tr><td>State-Space (block-wise)</td><td>9.59</td><td>0.282</td><td>0.698</td><td>11.95</td><td>0.280</td><td>0.535</td><td>69.00</td></tr><tr><td>Context learning, K=5</td><td>11.92</td><td>0.408</td><td>0.501</td><td>10.72</td><td>0.307</td><td>0.596</td><td>50.75</td></tr><tr><td>Context learning, K=20</td><td>12.54</td><td>0.449</td><td>0.496</td><td>11.07</td><td>0.359</td><td>0.543</td><td>58.63</td></tr></table>

Multi-segment sampling as an early memory metric. Multi-segment replay is a useful preliminary metric because it stresses the same failure mode that revisit evaluation later quantifies: the model must preserve scene evidence after the immediate visual context has been replaced by generated chunks. The boundary continuity, identity preservation, and action–visual alignment visible in these samples are qualitative evidence that we inspect before interpreting the full evaluation bundle.

## B.4 Protocol Invariants

Every model is evaluated under the following hard-fail invariants: (i) context handling matches between training and inference; (ii) context noise levels match between training and inference; (iii) K matches between training and inference; (iv) target-frame-only supervision and the flow-noise shift remain fixed; (v) cameraaction parameters are loaded before evaluation; and (vi) mutual exclusion between the two State-Space instantiations, and between Compression weighting and Compression length reduction, is enforced unless a row is explicitly marked as hybrid.

## C Empirical Analysis

The ablations are organized around the failure mode that motivates the paper: a model can generate plausible local video while still failing to remember the world at revisit time. Each sub-section isolates one axis of the memory design space and reads the resulting numbers through the same three-branch evidence stack established in section 5. The headline table compares four memory families: Context, Compression, Spatial, and State-Space. The mechanism tables then isolate three choices: Spatial read-out, Compression design, and State-Space recurrence. A final Context table reports the benefit of longer raw history. Unless stated otherwise, revisit numbers are the current material snapshot from eight cases per row: in-domain uses the 180○ return loop, while open-domain uses the 45○ return probe. Replay columns come from the GT-trajectory replay export and report mean PSNR/SSIM/LPIPS over three chunks.

How to read the tables. The tables should be read horizontally before they are read vertically. Each row reports three different views of the same model. Replay PSNR/SSIM/LPIPS measures whether the row can follow an observed trajectory with acceptable low-level fidelity; in-domain revisit asks whether the model can return to a known scene under a controlled loop; and open-domain VLM asks whether the same memory mechanism preserves a salient object and scene when no pixel-aligned reference is available. A row that wins on one column but fails on another is therefore not an outlier to be explained away. It is evidence that the memory mechanism is solving only part of the action-world-model problem, and the particular column that breaks tells us which part. This is the diagnostic logic we will use throughout the section.

## C.1 Memory Families under Matched Evaluation

The headline result immediately shows why replay alone is insufficient. Different low-level metrics select different winners: Spatial Memory is strongest on replay PSNR, long raw Context is strongest on replay SSIM/LPIPS, and block-wise State-Space remains strongest on in-domain LPIPS and open-domain VLM. This inversion means that memory quality is not the same thing as image quality under GT motion. Replay rewards the ability to synthesize frames that match a dataset trajectory, while open-domain revisit rewards the ability to carry object identity and scene layout through a generated excursion. The Context rows occupy a useful middle ground: increasing raw history improves semantic return much more clearly than it improves the full replay or in-domain image-metric bundle.

The Spatial–State-Space inversion is the central finding of the table. Spatial Memory wins replay PSNR and remains competitive on the replay image bundle, but collapses on open-domain VLM; block-wise State-Space loses the replay bundle and yet wins open-domain VLM by a wide margin. Two rows on the same backbone, optimizer, sampler, and evaluation pipeline thus arrive at opposite conclusions about which is the better world model, depending only on which column one reads. This is exactly the kind of evidence that a single-metric report would suppress. It also says something stronger: replay PSNR cannot be treated as a low-cost proxy for revisit quality. A model that reconstructs GT trajectories at sub-pixel cost can still fail to preserve a single salient object across a 45○ excursion, while a model that produces less faithful pixels can nonetheless carry the same scene through it. Replay measures how well a model follows; revisit measures how well a model remembers; these are different capabilities, and they are not monotonically aligned.

The in-domain column tells a third, intermediate story. Several rows cluster in the same in-domain VLM range, and the best in-domain PSNR comes from the legacy State-Space variant, while the best in-domain LPIPS comes from block-wise recurrence. This should not be read as a contradiction. In-domain return has pixel-aligned views and familiar data statistics, so a model with strong local reconstruction can partly succeed simply by tracking the trajectory back to a visually similar view. Once the open-domain probe removes that support, the picture changes: object preservation becomes the dominant criterion, and the block-wise row separates from the legacy variant. Reading the two columns together is therefore informative in a way that neither column is on its own. It exposes which rows were quietly relying on in-distribution texture statistics and which were carrying scene evidence in a form robust enough to survive shift.

What about Context? The Context rows are easy to under-rate because they contain no new machinery. Yet K 20 posts the second-best in-domain VLM (29.40) and a solid open-domain VLM (58.63), within striking distance of block-wise State-Space, while staying mid-pack on Replay. In a field that prefers learned bottlenecks to brute capacity, this is a counter-result: raw retrieval, with no learned compression and no specialized read-out, beats every compact memory mechanism in the matrix on open-domain semantic return except block-wise recurrence. We will see in section C.5 that the gain is not linear in K either: most of the open-domain jump happens by K=5, with K=20 supplying further refinement rather than a step change.

Treating the headline table as a metric bundle. For the reasons above, the paper treats this table not as a ranking but as a metric bundle. Replay filters out broken generators that cannot follow a trajectory at all; in-domain return checks controlled loop closure under familiar statistics; open-domain VLM provides the strongest evidence of whether the memory survives distribution shift. The most informative rows are precisely the ones where these three views disagree: Spatial Memory and block-wise State-Space disagree by sixty-three VLM points on open-domain return alone. The remainder of the section explains why.

Why the alignment plot matters. figure 7 turns the table into a diagnostic question. Replay is not meaningless: it catches whether a row can synthesize coherent frames under GT camera motion, so it is a dense and cheap health signal for intermediate model selection. But the same plot also shows why mean replay quality cannot be promoted into the final memory score. The strongest replay rows can still fail opendomain return, while rows with weaker replay fidelity can carry scene evidence through the excursion. Replay tells us whether the generator is healthy enough to trust; revisit tells us whether the world was remembered.

## C.2 Spatial Storage and Read-out

Setup. Spatial Memory asks whether a compact scene summary can replace raw temporal context. The ablation separates writing from reading: storing tokens but never injecting them is a lower bound; appending them to text cross-attention and reading them with a dedicated cross-attention block are the active variants. This distinction matters because a compact store can look useful for two different reasons. It may help by providing the generator with a stable scene representation, or it may simply regularize the context pathway and improve local reconstruction without carrying the object evidence needed for revisit. The read-out rows are designed to separate these cases: withheld read-out tests whether storage alone changes the generator, text KV concatenation tests whether the existing cross-attention pathway can consume the summary, and dedicated read-out tests whether the summary needs its own access path.

Table 9 Spatial Memory read-out ablation. R-P/R-S/R-L denote Replay PSNR/SSIM/LPIPS; ID-P/ID-S/ID-L denote in-domain closure PSNR/SSIM/LPIPS; O-V denotes open-domain VLM.

<table><tr><td></td><td>Read-out</td><td>R-P↑</td><td>R-S↑</td><td>R-L↓</td><td>ID-P↑</td><td>ID-S↑</td><td>ID-L↓</td><td>O-V↑</td></tr><tr><td>Spatial baseline</td><td>default</td><td>13.60</td><td>0.411</td><td>0.554</td><td>10.04</td><td>0.260</td><td>0.617</td><td>6.00</td></tr><tr><td>Spatial inject-none</td><td>none</td><td>14.66</td><td>0.417</td><td>0.541</td><td>10.38</td><td>0.309</td><td>0.631</td><td>15.50</td></tr><tr><td>Spatial concat-text</td><td>text KV concat</td><td>13.44</td><td>0.412</td><td>0.572</td><td>9.35</td><td>0.272</td><td>0.614</td><td>10.25</td></tr><tr><td>Spatial cross-attn RO</td><td>separate cross-attn</td><td>13.52</td><td>0.409</td><td>0.557</td><td>8.49</td><td>0.232</td><td>0.664</td><td>17.12</td></tr></table>

The replay numbers are encouraging but also cautionary. Spatial inject-none obtains the strongest replay PSNR/SSIM/LPIPS bundle among the spatial rows, even though it withholds the stored tokens from the generator. The open-domain VLM column shows that stronger read-out can help: dedicated cross-attention reaches 17.12, above the default spatial baseline at 6.00. Yet the same column also shows that the family is still weak as semantic memory. Even the best read-out row remains far below raw Context K 20 and block-wise State-Space in table 8, and inject-none remains surprisingly competitive at 15.50. The spatial store is therefore not simply missing a read-out path; its compact representation also appears to discard object evidence needed for return.

Why does withheld read-out improve Replay? The most striking number in table 9 is the inject-none row: 14.66 Replay PSNR, 0.417 SSIM, and 0.541 LPIPS, all better than the active read-out variants. Withholding the stored tokens makes the network strictly less expressive at inference, yet it makes replay look strictly better. The simplest explanation consistent with the rest of the matrix is that the spatial summary is not, at this scale, behaving as a memory at all: it is behaving as an additional training-time conditioning channel that regularises the network and quietens the per-chunk reconstruction error. Once we remove the tokens at inference, the network reverts to the cleaner I2V-style prediction it has implicitly been prepared for, and replay PSNR ticks up. The lesson is that replay improvements from a memory module do not, on their own, certify that the module carries memory.

Storage cannot be evaluated without read-out. The read-out rows span a useful gradient of access strength. Withheld read-out is a hard zero: tokens exist but are unreachable. Text-KV concatenation is a soft access path: the spatial tokens are dumped into the same cross-attention slots that already serve the text prompt. Dedicated cross-attention read-out is a harder access path: the spatial tokens get their own query mechanism and a zero-initialised gate that the optimizer can open. The measured VLM scores indicate that access matters, but not enough. Dedicated read-out is best among the spatial rows, while text-KV concat improves only mildly over the default baseline. Since all three read-out variants remain far below the strongest context and State-Space rows, the failure is shared between access and storage bandwidth, rather than being only missing bookkeeping in the read-out module.

Why Spatial loses open-domain VLM by so much. A finer reading of the score helps explain the magnitude. The open-domain probe is constructed so that the salient object is distinctive in shape, colour, and category; in effect, it is chosen to be a high-entropy identity anchor. A G×G grid summary is a lowbandwidth representation of the scene; it can encode where things roughly are, but it has limited capacity to encode which thing. When the camera leaves and returns, the generator must place the same object back into the frame, and a generic spatial code under-determines that decision. Replay does not expose this gap because the GT trajectory keeps the object visible; revisit exposes it because the object must be regenerated from memory alone. The Spatial family is therefore not weak in general; it is weak exactly where compact summaries should be weak, which is where compact memory bottlenecks are most attractive to use.

Table 10 Compression ablation. R-P/R-S/R-L denote Replay PSNR/SSIM/LPIPS; ID-P/ID-S/ID-L denote indomain closure PSNR/SSIM/LPIPS; O-V denotes open-domain VLM.

<table><tr><td></td><td>Mechanism</td><td>R-P↑</td><td>R-S↑</td><td>R-L↓</td><td>ID-P↑</td><td>ID-S↑</td><td>ID-L↓</td><td>O-V↑</td></tr><tr><td>Weight-only baseline</td><td>token weighting</td><td>8.65</td><td>0.174</td><td>0.752</td><td>10.67</td><td>0.154</td><td>0.645</td><td>22.38</td></tr><tr><td>Length r=2</td><td>temporal length</td><td>7.84</td><td>0.162</td><td>0.746</td><td>8.82</td><td>0.183</td><td>0.654</td><td>24.00</td></tr><tr><td>Length r=4</td><td>temporal length</td><td>9.88</td><td>0.183</td><td>0.714</td><td>9.53</td><td>0.163</td><td>0.617</td><td>43.25</td></tr><tr><td>Hybrid r=2 + weight</td><td>both</td><td>9.10</td><td>0.181</td><td>0.734</td><td>7.87</td><td>0.277</td><td>0.695</td><td>8.75</td></tr><tr><td>Hybrid r=4 + weight</td><td>both</td><td>9.63</td><td>0.215</td><td>0.730</td><td>8.81</td><td>0.156</td><td>0.625</td><td>9.00</td></tr></table>

## C.3 Compression and Temporal Resolution

Setup. The Compression family asks whether memory should be compacted by importance or by time. Token weighting preserves the number of historical observations but changes their relative influence; length compression discards temporal resolution. The hybrid row tests whether the two operations are complementary or redundant. These variants are often grouped together as “compression”, but they make different assumptions about what history contains. Token weighting assumes that all temporal positions can remain available while their importance varies with age or relevance. Temporal length reduction assumes that neighboring observations can be pooled without destroying the memory signal. The hybrid row is the strictest test: it asks whether the model can first reduce temporal support, and then still learn a useful importance profile over the reduced support.

The table shows that compression has no monotonic ordering by apparent capacity. Weight-only compression is the most conservative operation because it keeps every temporal position, yet it is not the best opendomain semantic row. Length r=4 outperforms weight-only on open-domain VLM, indicating that a shorter but cleaner context can sometimes be preferable to a full-length context with learned weights. However, the hybrid rows perform poorly, especially on open-domain VLM. The likely explanation is that the two operations remove different kinds of signal: pooling reduces temporal evidence before the model can decide which observations matter, and weighting then operates on an already coarsened sequence. When the return probe depends on a particular object or view, this order can erase the evidence that weighting would otherwise have preserved.

Compression is not a monotone capacity dial. Naive intuition says that “less compression” should mean “more memory”. The numbers in table 10 actively disagree. Weight-only compression preserves every temporal position (r 1 in length terms) and yet posts an open-domain VLM of only 22.38. Length r 4 throws away three out of every four temporal positions, and reaches 43.25 on the same metric. The model that sees less history does better on the harder column. This is counter-intuitive only if one assumes that all temporal frames carry equal information; once we drop that assumption, the result becomes mechanistic. A length-pooled context concentrates the remaining capacity on fewer, cleaner temporal slots, which makes the per-frame signal easier for the generator to read; a weight-only context preserves every frame but spreads the model’s attention thinly across them, and the attention budget for any specific frame stays low. The lesson is structural: how the context is shaped can matter more than how much of it survives.

Why hybrid compression backfires. The hybrid rows are the most diagnostic numbers in the table. They combine the two operations one would naively assume to be complementary, and they perform worse than either operation alone (8.75 and 9.00 on open-domain VLM, well below both pure weighting and pure length pooling). This is not a small effect: hybrid compression discards three-quarters of the open-domain return signal that pure length pooling preserves. The order of operations explains why. Length pooling acts first, on the raw context. It collapses neighbouring frames into a single pooled slot without consulting any importance signal, so any frame that happens to share a window with a forgettable neighbour is averaged out before the model can object. Weighting then learns importance on top of this coarsened sequence, where the high-value frames are no longer individually addressable. The two operations do not compose like filters; they compose like a destructive quantization followed by a re-encoding of the resulting noise. figure 8 makes the failure mode visible: hybrid rows tend to preserve the existence of an object after return, but not the identity of the object.

Table 11 State-Space memory comparison. R-P/R-S/R-L denote Replay PSNR/SSIM/LPIPS; ID-P/ID-S/ID-L denote in-domain closure PSNR/SSIM/LPIPS; O-V denotes open-domain VLM.

<table><tr><td></td><td>Structure</td><td>R-P↑</td><td>R-S↑</td><td>R-L↓</td><td>ID-P↑</td><td>ID-S↑</td><td>ID-L↓</td><td>O-V↑</td></tr><tr><td>State-Space (legacy hybrid)</td><td>legacy hybrid</td><td>12.69</td><td>0.344</td><td>0.581</td><td>12.23</td><td>0.298</td><td>0.584</td><td>34.75</td></tr><tr><td>State-Space (block-wise)</td><td>paper-aligned recurrence</td><td>9.59</td><td>0.282</td><td>0.698</td><td>11.95</td><td>0.280</td><td>0.535</td><td>69.00</td></tr></table>

Cost-control, not capacity substitution. The replay columns reinforce the same interpretation. Length r=4 improves the Replay PSNR/LPIPS bundle over weight-only, but the absolute values remain below the strongest Context and Spatial rows. Compression therefore appears useful as a cost-control mechanism, a way to trade some replay quality for cheaper steps, not as a free substitute for historical capacity. The right reading of table 10 is thus less about which compression operator wins and more about what compression is. In the current matrix, it is a budget-management tool that has not yet learned what to budget for. A stronger compression memory would need an explicit objective for what to retain: an object-aware pooling rule, a learned criterion tied to revisit success, or a mechanism that preserves a small set of high-value frames while compressing only the redundant background evidence around them. The current operators do none of these things, and the numbers reflect that absence.

## C.4 Implicit State-Space Memory

Setup. State-Space memory asks whether an implicit recurrent state can carry the scene history that explicit context provides. We report two instantiations as one family: a legacy VideoSSM hybrid, and a paper-aligned block-wise recurrence. The point is not to crown a new State-Space variant, but to test whether implicit memory is a credible answer to revisit consistency. The comparison is deliberately narrow. Both rows use the same evaluation bundle and are treated as instantiations of an implicit-memory family, but they differ in where and how recurrence is structured. The legacy hybrid behaves like a lightweight recurrent supplement to the video backbone; the block-wise row makes recurrence a more regular part of the temporal computation. If both rows behaved similarly, the main conclusion would be that implicit memory is largely a family-level choice. Their separation instead suggests that recurrence design matters as much as the decision to use recurrence at all.

This is the clearest example of metric separation in the paper. Legacy hybrid recurrence has the best indomain PSNR and strong replay PSNR, but block-wise recurrence dominates open-domain VLM. The likely reason is that in-domain return still offers familiar textures, motions, and scene statistics, so a model with strong local reconstruction can look good even when its state is not robust under distribution shift. Opendomain identity preservation is less forgiving. The model must carry a compact description of the salient object and environment through the generated excursion, and the block-wise recurrence appears better suited to this role.

This does not mean that State-Space memory is universally superior. Its replay PSNR is not the highest, and the family still needs comparison at longer open-domain horizons. What the current evidence supports is more specific: a recurrent state can be a real memory mechanism for revisit, provided that its structure is strong enough to preserve scene evidence rather than merely smoothing the temporal computation. This motivates future SSM variants that expose the state to explicit object- or view-level supervision during replay and return probes.

The largest single jump in the paper. The two State-Space rows separate sharply on open-domain VLM using the same backbone, optimizer, sampler, and data, while their replay metrics move in the opposite direction and their in-domain PSNRs stay close. We are not aware of another row pair in the matrix that opens a comparable gap on open-domain VLM with so few controlled differences. The headline of section C.4 is therefore that the choice of recurrence structure, including where the state is updated, how often it is read, and how it is gated, is at least as consequential for revisit consistency as the older choice between explicit and implicit memory. Treating “State-Space memory” as a single design choice obscures the biggest variance the matrix has shown so far.

Table 12 Context length ablation. R-P/R-S/R-L denote Replay PSNR/SSIM/LPIPS; ID-P/ID-S/ID-L denote indomain closure PSNR/SSIM/LPIPS; O-V denotes open-domain VLM.

<table><tr><td></td><td>K</td><td>R-P↑</td><td>R-S↑</td><td>R-L↓</td><td>ID-P↑</td><td>ID-S↑</td><td>ID-L↓</td><td>O-V↑</td></tr><tr><td>I2V baseline</td><td>1</td><td>10.03</td><td>0.398</td><td>0.534</td><td>10.32</td><td>0.291</td><td>0.643</td><td>12.25</td></tr><tr><td>Context learning, K=5</td><td>5</td><td>11.92</td><td>0.408</td><td>0.501</td><td>10.72</td><td>0.307</td><td>0.596</td><td>50.75</td></tr><tr><td>Context learning, K=20</td><td>20</td><td>12.54</td><td>0.449</td><td>0.496</td><td>11.07</td><td>0.359</td><td>0.543</td><td>58.63</td></tr></table>

Why does block-wise recurrence preserve more under shift? A useful contrast is the legacy hybrid, which behaves like a lightweight recurrent supplement: it adds a recurrent gate around an otherwise standard temporal attention path. That gate is sufficient to smooth in-distribution motion and is sufficient to post a strong in-domain PSNR, but its state is shallow in the sense that any single block can largely bypass it. Block-wise recurrence, in contrast, makes the recurrent step a regular, non-bypassable part of the temporal computation: every n-th DiT block reads the carried state, updates it, and writes it back. The state thus accumulates evidence across the chunk boundary in a way that the legacy variant cannot, and it is this carried evidence that becomes load-bearing once the camera leaves the visible support. From the generator’s point of view, the block-wise state is the only place where information about the salient object can survive between segments without an explicit retrieval step.

Why does block-wise lose Replay? The same structural choice has a cost. A non-bypassable recurrent state forces the network to commit some representational capacity to summarising the past, and that capacity is not available for fitting the local trajectory pixel-by-pixel. The result is the 9.59 Replay PSNR and 0.698 Replay LPIPS, which on their own would put block-wise recurrence near the bottom of the headline table. Reading Replay alongside open-domain VLM is therefore essential here in a way that it is not for any other row. Block-wise recurrence is precisely the kind of design that a single-metric report would dismiss.

Implication: implicit memory is a real mechanism, not a smoother. The most robust statement we can make from table 11 is narrow: a recurrent state can act as a genuine memory mechanism for revisit, provided that its structure is strong enough to preserve scene evidence rather than merely smooth the temporal computation. This is a stronger claim than “recurrence helps”. It rules out the lightweight-gate interpretation of state-space memory in this regime, and it sets a structural minimum: the state has to be carried through a non-trivial fraction of the network’s depth before it begins to behave like memory. Future SSM variants that expose this state to explicit object- or view-level supervision during replay and return probes are a natural next step, and the gap between 34.75 and 69.00 suggests there is significant room above the current ceiling.

## C.5 Raw Context Capacity

Setup. Context learning asks how much raw history is enough when the memory is not compressed into a special module. Here the I2V baseline contains only the current anchor frame; K 5 and K 20 add retrieved historical observations. This is the simplest ablation in the section, and also the most important reference point. Raw context has no learned compression bottleneck and no special read-out module, so it answers a direct capacity question: how much does the generator benefit when it can see more historical observations under the same retrieval and action-conditioning interface? Any compact memory should be compared against this trend, because a method that only beats the anchor-only baseline may still be weaker than ordinary uncompressed history.

Both replay and return improve with longer raw context, but the open-domain VLM trend is much steeper than the replay trend. Replay PSNR, SSIM, and LPIPS improve from the anchor-only baseline to K=20, while open-domain VLM makes most of its gain by K=5 and continues to improve at K=20. This asymmetry is meaningful: raw history improves the GT-trajectory image bundle, but it buys much more on the semantic return axis, where the model must restore an object and scene after leaving the view. In other words, raw context is not merely extra conditioning; it is a memory-capacity baseline.

The result also calibrates the rest of the study. Spatial Memory and compression are attractive only if they preserve enough of this raw capacity at lower cost, while State-Space memory is attractive if it matches or exceeds raw context in robustness under distribution shift. The current table suggests that raw context remains difficult to beat as a simple solution. The best compact or implicit memory mechanisms should therefore be evaluated against Context K=20, not only against the I2V baseline.

The capacity asymmetry between Replay and revisit. The single most informative pattern in table 12 is the slope mismatch between the replay image bundle and open-domain return. Replay PSNR moves by 2.51 points from K=1 to K=20, while open-domain VLM moves by 46.38 points over the same range. Raw history does improve pixel-level reconstruction, but its larger effect is on the ability to put the same object back into the frame after the camera has left it. This is the cleanest evidence in the paper that revisit consistency is a different capability from trajectory following, and that the two scale with the available history at different rates. Any compact memory whose argument rests on “matching Replay quality at a fraction of the tokens” has, by this measurement, only matched the cheaper of the two quantities.

The gain is sublinear in K. Most of the open-domain VLM jump happens by K=5 (+38.50 points over the I2V baseline), with K=20 adding a further 7.88 points. The marginal value of an extra retrieved frame is therefore high near the cold-start regime and decays quickly. Two consequences follow. First, the I2V baseline is severely under-resourced: it is not a calibrated reference point for what a generation model can do, because moving from K=1 to K=5 alone roughly quadruples the open-domain VLM score on the same backbone. Any memory method that primarily reports “beats I2V” should be re-read with this factor in mind. Second, the asymptote of raw context appears to be near, not at, K=20: the cost grows linearly with K, the benefit grows sublinearly, and the curve is already flattening. This is exactly where compressed memories should help. The fact that they do not in the current matrix is the strongest invitation for future work that the paper makes.

Raw context as a calibration tool. The right way to read K=20 Context is therefore not as a candidate winner of the matrix, but as a calibration tool. It tells us what a generator can do when it is given uncompressed evidence under the same retrieval and action-conditioning interface as the specialized methods. Any compact or implicit memory should be evaluated against this reference point, not against the I2V floor. Under that more demanding bar, the current matrix shows exactly one mechanism, block-wise State-Space, that exceeds Context K=20 on open-domain VLM, and most of the others fall short of even K=5. This runs against the field’s default direction of travel toward more elaborate compressed memories.

## C.6 Quality–Time Trade-off

Setup. The headline table ranks quality after training, but a practical memory baseline must also justify its training cost. We parse the latest training logs for each representative row, convert the average step time to GPU-hour per optimization step, and join it with the final Replay PSNR reported in materials/training\_- time\_by\_baseline.csv. The scatter below normalizes both axes: smaller x means cheaper steps, larger y means stronger replay fidelity, and the diagonal marks equal normalized quality and cost. The efficiency panel should be read as a diagnostic rather than as a replacement for the evaluation bundle. It measures how much replay quality is obtained per unit of step cost, which is useful for deciding which memory rows are practical to scale. However, the axis is replay PSNR, not open-domain semantic return: a method can look efficient because it reconstructs GT trajectories cheaply while still failing to preserve object identity after a return. Conversely, a method with weaker replay efficiency can still be scientifically important if it is the only one that preserves open-domain memory.

This distinction explains why the final conclusion is not a simple efficiency ranking. Spatial Memory is attractive in a replay-efficiency view because it delivers strong low-level reconstruction at a compact cost Block-wise State-Space has the opposite profile: it does not dominate replay PSNR, yet it is the most reliable semantic-return mechanism in the open-domain probe. Raw Context is the third corner of the trade-off: it provides capacity and strong semantic return, but its cost grows with the amount of retrieved history. The practical memory-design problem is therefore a three-way trade-off among replay fidelity, step-time efficiency, and revisit generalization.

Three corners, three different design recipes. It is worth stating the three corners of this trade-off explicitly because they map onto three different downstream uses of an action world model. (i) If the application is high-fidelity replay of an observed trajectory at the lowest possible step cost, such as telemetry-driven re-rendering of a recorded camera path, the replay-efficiency corner argues for compact spatial summaries. (ii) If the application is interactive camera control over a scene that the camera will leave and re-enter many times, the canonical world-model use case, the open-domain VLM corner argues for block-wise state-space recurrence, and the rest of the matrix is at best a second choice. (iii) If neither cost nor open-domain robustness is the binding constraint, raw context at K=20 is the safest default, because it dominates the in-domain VLM column and is strictly within reach of block-wise State-Space on open-domain return. figure 9 should be read with this map in mind: points above the diagonal are good for replay alone, and they are not necessarily good for the use case that motivated the world-model framing.

Why a single efficiency number would mislead. Suppose one collapsed the matrix into a single “quality per GPU-hour” scalar by averaging Replay PSNR with a normalised cost. Spatial Memory would win, blockwise State-Space would look unappealing, and the headline of the paper would be that compact summaries are the practical answer. Yet the open-domain VLM column disagrees by a factor of more than 11 between the two (69.00 vs. 6.00). The single-number ranking would not be wrong about the cost; it would be wrong about what the cost is buying. This is the structural reason the paper insists on reporting the metric bundle rather than collapsing it.

## C.7 Cross-cutting Observations

Read together, the sub-sections support three statements that no single row establishes on its own.

(A) Replay quality and revisit quality are different axes that routinely disagree. Spatial Memory dominates Replay and bottoms out on open-domain VLM; block-wise State-Space does the reverse; length r 4 is worse than its hybrid counterpart on Replay but far better on revisit. Across the matrix the rank correlation between Replay PSNR and open-domain VLM is weak and sometimes inverted within a family, because the two metrics measure two sub-problems: following an action sequence and remembering a world. A memory paper that does not separate these axes cannot tell whether its mechanism helps generation or memory.  
(B) Storage and read-out are separate design choices. The Spatial read-out ablation shows this most directly: withholding the stored tokens makes Replay PSNR go up, not down. Raw context makes most of its open-domain gain from capacity under a trivial read-out, while block-wise State-Space gains from read-out structure at fixed capacity. The two families move along different axes of the storage–read-out plane, so a useful minimum is to report one ablation that fixes storage and varies read-out, and one that does the reverse.  
(C) Compactness is not free. The two most aggressively compact mechanisms, Spatial Memory and hybrid compression, post the two lowest open-domain VLM scores (6.00 and 8.75), while raw context at K=20 reaches 58.63 and block-wise State-Space reaches 69.00. What separates the strong rows is not whether the memory is compact, but whether the compact representation is read out in a way that lets the generator put the same object back after the camera leaves it. Future compact designs should therefore be benchmarked against Context K=20 and block-wise State-Space on open-domain VLM, not against I2V.

These claims sit within clear limits. With eight cases per row, small open-domain VLM differences (within roughly five points) should not be over-interpreted, and the matrix uses a single training budget and a single dataset, so conclusions about which mechanism wins may shift at larger compute even if those about which axes matter are more durable. section 8 returns to these points.