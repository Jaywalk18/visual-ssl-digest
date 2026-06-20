# Timage: A Generative Text-in-Image Paradigm for Fine-Tuning Vision-Language Models

Yifeng Wu1,2,3, Huimin Huang3, Ruiluo Wu2, Chunyi Lin2, Guanhua Chen4 Xian Wu3, Wang Song2, Ruize Han1⋆

1Fudan University, 2Shenzhen University of Advanced Technology, 3Tencent Jarvis Lab, 4Southern University of Science and Technology

Abstract. Multimodal Large Language Models (MLLMs) often lose track of the right image regions during fine-grained spatial reasoning, because a textual query rarely carries any explicit geometric anchor into the pixel domain. Prevailing remedies either rewire the model’s weights or pad the prompt with verbose instructions, yet neither reliably pins the language to the correct visual coordinates without eroding the backbone’s general competence. We introduce Timage, a paradigm that recasts multimodal understanding as an alignment problem solved at the input: the query is drawn, as a typeset overlay, onto the image itself. The placement and appearance of this overlay are produced by a Constrained Schr¨odinger Bridge (cSB), an entropic optimal-transport sampler that factorizes layout synthesis into two coupled stochastic stages. The first stage, Region Search, transports noise toward query-aligned image zones while obeying a hard occlusion barrier that protects salient foreground content; the second stage, Appearance Shaping, sizes the glyphs through an “ink-budget” regularizer so that the rendered text stays legible and visually balanced. The resulting overlay behaves as an explicit attention beacon that channels the model’s focus along spatial semantics. On the VMCBench suite, Timage paired with a modest 7B backbone clearly overtakes far larger proprietary systems as well as parameter-tuned baselines. The study positions deliberate input reconstruction as a powerful, architecture-neutral lever for strengthening multimodal reasoning.

Keywords: Multimodal Reasoning · Schr¨odinger Bridge · Multimodal Fine-Tuning · Representation Learning

## 1 Introduction

Multimodal Large Language Models (MLLMs) have reshaped how machines couple visual perception with linguistic reasoning, and they now span a wide task surface: answering questions about cluttered scenes, parsing documents, reading scientific charts, and more. Beneath these successes, however, lies a persistent structural mismatch–the query and the picture never truly share a representational footing. A textual instruction (a question, a directive, a reasoning cue) is tokenized into a discrete symbol stream, whereas the image enters as a dense, continuous tensor. The two only meet later, through implicit feature blending inside cross-attention layers. As a result, the network is forced to silently infer a mapping from abstract phrases–“to the left of,” or a positional reference inside a packed layout–onto concrete pixel regions. When the task demands fine spatial discrimination, dense text reading, or chained logical steps, this inference frequently slips, and the model’s attention drifts away from the region the query actually concerns.

![](images/88beda170b16bd1703d736bedf077e59aafc0db977698390b10e37d5b694a9ad.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["(a) No Visual Prompt"] --> B["VMCBench: No"]
  B --> C["Answer the questions in the image"]
  C --> D["(b) Subtitles Prompt"]
  D --> E["Accuracy: 83.5%"]
  E --> F["(c) Random Overlay"]
  F --> G["Accuracy: 79.8%"]
  G --> H["(d) Text-in-image"]
  H --> I["Accuracy: 87.7% ↑"]
    style A fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
```
</details>

Fig. 1. (a-c) Prior approaches stumble on feature-level decoupling, spatial misplacement, or visual occlusion. (d) Timage instead synthesizes query-conditioned semantic overlays via the cSB sampler. By binding the instruction to the image manifold at the input stage, Timage lifts complex spatial reasoning and returns better-grounded multimodal answers.

Existing attempts to close this gap tend to fall into two families, each with its own ceiling. The first family, weight-space adaptation, is typified by parameterefficient fine-tuning such as LoRA [19] and adapter modules [4,47,58,11,20]. These methods bend the model’s internal parameters toward a task distribution. They work well inside a fixed domain, but the price is steep: the pre-trained model’s breadth degrades, foundational knowledge is partially overwritten, robustness to distribution shift weakens, and both compute and storage grow as task variants accumulate. In effect, a representational defect is patched by overfitting parameters rather than by repairing the signal that enters the model. The second family, input-space prompting, led by Visual Prompt Tuning (VPT), optimizes continuous learnable vectors appended to the input. Yet these vectors are semantically opaque perturbations. Even geometry-aware variants only reshuffle where abstract feature tokens sit; they never inject readable content. Such prompts therefore cannot tap the backbone’s latent character-recognition skill or its pre-aligned vision-language grounding, because the prompt itself means nothing in human terms.

These observations expose a missing piece in multimodal representation learning: there is no mechanism that explicitly turns the query text into a harmonious cross-modal cue co-located with the visual signal. We fill this gap with Text-in-Image (Timage), which reframes multimodal reasoning as input-level alignment on a shared manifold. Rather than feeding the query as an external token stream, Timage paints the natural-language instruction onto the image as a typeset overlay, producing a query-conditioned visual prompt (Figure 1(d)). As a paradigm, Timage offers three distinguishing properties. ➊ In contrast to the standard recipe of Figure 1(a), which embeds text and vision through separate branches, Timage routes everything through a single visual input. ➋ Timage acts as an external adapter that leaves the MLLM untouched. ➌ Timage is simultaneously human-legible and machine-actionable: the cross-modal cue is literally visible, which is also a step toward interpretable multimodal AI.

Realizing this is not trivial. As Figure 1(b-c) shows, naively stamping the text as a fixed caption [52] or at a random spot is either too rigid or too unstable. We therefore cast the problem generatively, as a Constrained Schr¨odinger Bridge (cSB). We split text-in-image inlaying into two questions–where to write and how to write–and solve them in sequence. For where, we pose two objectives: pull the writing region toward query-relevant content (a low semantic potential) while honoring a hard prior that no salient foreground object may be covered. For how, we auto-tune glyph size through an “ink usage” budget so the text fits gracefully. Both objectives become control conditions that steer a Schr¨odinger-Bridge diffusion process generating the overlay layer step by step.

We benchmark Timage on VMCBench [60], which consolidates 20 diverse VQA datasets. With only a 7B backbone, Timage reaches a state-of-the-art mean accuracy of 87.7%. It clears massive proprietary systems by wide margins (e.g., +7.4% over GPT-4o and +2.7% over Qwen2-VL-72B) and, strikingly, beats a full fine-tuning baseline by +2.5%. These results argue that rebuilding the input can deliver more capability than adapting parameters. Cross-architecture tests confirm generality, with gains of +4.8% to +8.6% spanning backbones from LLaVA-1.5 to the recent Qwen3. Our contributions are:

1. We present Timage, a paradigm that fuses linguistic semantics with the visual signal by rendering the query as an optimized overlay–single-modality at the input, plug-and-play without model edits, and visibly interpretable.  
2. We cast overlay-layout synthesis as a Constrained Schr¨odinger Bridge, yielding a principled generative procedure whose energy functional trades off semantic pull against hard physical constraints.  
3. We set new state-of-the-art numbers on VMCBench in a backbone-frozen, training-free regime, with stronger generalization and efficiency than both large proprietary models and parameter-efficient fine-tuning.

## 2 Related Work

Fine-Tuning of Vision-Language Models. Adapting Vision-Language Models (VLMs) is presently dominated by three parameter-efficient strands [18,6,54]. Adapter-style methods [47,58,11,20,4] graft small task-specific bottlenecks onto a frozen backbone to allow modular updates. Low-Rank Adaptation (LoRA) and its descendants [19,32,1,51,30,57,34] express weight updates through low-rank factors, trading expressive range for a small storage footprint. In parallel, Visual Prompt Tuning (VPT) and relatives [17,5,22,43,12] prepend learnable tokens to the input across transformer depths. Effective as they are at tuning weights or latent vectors, these techniques live almost entirely in latent feature space; their abstract codes resist human reading and can falter when the task needs tight alignment with explicit spatial-semantic constraints. Departing from such feature-centric edits, our work pursues a different route–steering model behavior by writing explicit semantic cues straight into the input pixels.

Generation in Service of Multimodal Understanding. Folding generative steps into the understanding pipeline has become a fertile direction. The literature ranges from textual Chain-of-Thought extensions [53,10,59] to visual-generative reasoning [49,3,35,39,8,50,38,56,55], where models emit intermediate sketches, heatmaps, or masks to scaffold hard logic. Other lines exploit interactive generation [24,28,44,45,33,40,34,42,7], producing iterative cues that sharpen spatial grounding and cross-modal alignment downstream. Most of this work, however, treats the reasoning trace as a side modality or a separate step kept apart from the original visual input. While that decoupling cleanly separates perception from reasoning, it can introduce a mismatch between the reasoning context and the primary scene. Motivated by this, we ask whether the reasoning prompt can be inscribed directly into the original image, yielding a selfcontained, instruction-aware context.

Schr¨odinger Bridges in Diffusion Models. The Schr¨odinger Bridge (SB) framework has recently entered diffusion modeling as a way to solve entropic optimaltransport problems. This body of work includes theoretical advances [14,9,37] that characterize optimal stochastic paths between distributions, alongside practical deployments [46,26,25,31,41,48] in image translation and molecular conformer search. Newer studies push SB toward structured optimization [29,15,16,27], weaving geometric or physical priors into the diffusion to enable constrained synthesis. Extending SB to structured layout generation such as text overlays raises fresh difficulties: classical SB targets unconstrained transport, whereas a legible prompt must respect strict geometric and visibility limits so that essential image content is never occluded. Reconciling these competing objectives inside an SB process is still open, and we take a step toward it with a constrained generative formulation.

## 3 Method

We propose Timage, a paradigm for visual-language interaction that writes the natural-language query into the image as a semantically aligned, spatially admissible overlay. Whereas conventional systems carry text as an external token sequence detached from image geometry, Timage produces a query-conditioned visual prompt through a controllable stochastic generator. This couples linguistic meaning with spatial context directly: the overlay nudges a downstream model toward task-relevant regions while a hard constraint forbids it from covering salient foreground content.

## 3.1 Overview and Notation

Take an image x0 and a query q; our aim is to inscribe q onto x0, much as one would letter a caption onto a painting. Two decisions govern the result: where the text lives and how it looks. We encode the first as a placement field and the second as a style field, and we score any candidate glyph layout T with a single writing energy

![](images/bb91a5659c8b44b32284bcb08a758ff32d65470d6f3cf2dd58d40c23efd53a0a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Input Image"] --> B["Saliency Map"]
  A --> C["Attention Map"]
  A --> D["Heat Map"]
  B --> E["Constrained Schrödinger Bridge (cSB)"]
  C --> E
  D --> E
  E --> F["Timage"]
  F --> G["MLLMs"]
  G --> H["Snow."]
  H --> I["Answer"]
  J["What is on the top of the mountain?"] --> A
```
</details>

Fig. 2. Overview of Timage. From an image and its question we build a feasible manifold Ω through an Admissible Mask that merges semantic relevance with a hard non-occlusion prior. A Constrained Schr¨odinger Bridge then renders the inlaid text via a projected stochastic process, guaranteeing both geometric validity and downstream accuracy.

$$
E (\mathcal {T}; x _ {0}, q) = \lambda_ {\mathrm{zon}} E _ {\mathrm{zon}} (\alpha (\mathcal {T}); x _ {0}, q) + \lambda_ {\mathrm{sty}} E _ {\mathrm{sty}} (\beta (\mathcal {T})), \tag {1}
$$

where $\alpha \in [ 0 , 1 ] ^ { H \times W }$ is the placement field marking where $\tau$ is written, and $\beta$ summarizes its appearance. The two terms read as:

– Placement energy $E _ { \mathrm { z o n } } \mathrm { : }$ rewards regions α that are query-relevant yet do not sit on top of the image content the query refers to.  
– Style energy $E _ { \mathrm { s t y } } { \mathrm { : } }$ : imposes appearance priors–an ink budget and spatial smoothness– so that $\tau$ stays readable. Note that $\beta$ inherits constraints from the chosen placement α.

Sections 3.2 make $E _ { \mathrm { z o n } }$ and $E _ { \mathrm { s t y } }$ precise.

Why a distribution, not a point. As with lettering a painting, neither the region nor the style is uniquely determined: many placements and many sizes are acceptable. We therefore want a method that samples from the set of admissible layouts rather than collapsing onto one. This is exactly what motivates the Schr¨odinger-Bridge formulation of Section 3.3.

## 3.2 Building the Energy Function

Given the pair $( x _ { 0 } , q )$ , we synthesize $x _ { 1 } = x _ { 0 }$ ⊕ $\tau _ { \ast }$ , where $\tau$ renders the content of $q .$ Writing $\tau$ requires a placement field $\alpha ( \mathcal { T } )$ and an appearance field $\beta ( \tau )$ subject to the following constraints.

Placement Constraint $E _ { \mathbf { z o n } }$ . We want α (shorthand for $\alpha ( \mathcal { T } ) )$ to be near the image evidence implicated by q but off of it. Proximity helps the MLLM associate the inscribed T with the relevant region; non-coverage protects the very content the query depends on, especially query-related foreground objects. We make this concrete in two parts.

➊ Semantic pull $( s o f t )$ . To express soft alignment, we first form a queryrelevance heatmap $H _ { q }$ . With $e _ { q } = \operatorname { E n c o d e r } _ { \operatorname { t e x t } } ( q )$ , we read cross-attention between $e _ { q }$ and the visual patches of a frozen VLM (e.g., Qwen2.5-VL) to produce $H _ { q }$ . The placement α should overlap with the spatial focus indicated by $\textstyle e _ { q } ,$ giving

$$
E _ {\mathrm{sem}} (\alpha) = - \int \alpha (u) H _ {q} (u) d u, \tag {2}
$$

with u a spatial coordinate; stronger overlap means lower energy.

➋ Non-occlusion (hard). We deliberately treat occlusion as a barrier rather than a soft penalty. We assemble an Admissible Mask $M _ { \mathrm { c a n d } } \in \{ 0 , 1 \} ^ { H \times W }$ that delimits the feasible region by fusing object and saliency evidence:

$$
M _ {\text { cand }} (u) = \sigma \left(\mathbf {I} - \left(\bigcup_ {i} O _ {i} (u)\right) - \gamma   S (u)\right), \tag {3}
$$

where I is the all-ones $W \times H$ matrix, $O _ { i } ( u )$ is the instance mask of object i from a segmenter (e.g., Mask R-CNN), S(u) is a dense saliency map, γ weights the saliency term, and σ binarizes via thresholding. Non-occlusion is enforced hard: we require $\alpha \subset M _ { \mathrm { c a n d } }$ , so feasibility is guaranteed rather than merely encouraged.

Style Constraint $E _ { \mathrm { s t y } }$ . With α fixed, we render T strictly inside it and shape its appearance through β. In practice the dominant knob is font size, which we govern with a Query-Adaptive Ink Budget: just as physical writing consumes ink, we cap the rendered ink so glyph size adapts to the text and the scene. The target ink volume is

$$
\operatorname{ink} (\mathcal {T}) = \kappa \cdot \operatorname{Length} (q) \cdot \operatorname{Scale} (\alpha), \tag {4}
$$

where Length counts tokens, Scale is the resolution factor of the placement region, and κ is the nominal ink per character per unit area.

## 3.3 Constrained Schr¨odinger Bridge

We synthesize the overlay through a Constrained Schr¨odinger Bridge (cSB): an optimal stochastic transport from a noise prior $\mu _ { 0 } = \delta$ to the inscribed text T .

Constrained Dynamics. We first transport the placement field α, keeping the trajectory inside the feasible manifold $M _ { \mathrm { c a n d } }$ of Section 3.2. Unlike unconstrained SB, our energy target rewards only semantic alignment (Eq. (2)); spatial feasibility is delegated to the membership constraint $\alpha \subset M _ { \mathrm { c a n d } }$ from Eq. (3).

Following [45], generation obeys a drift-diffusion SDE driven by a timedependent velocity field $v _ { \theta } \colon$ :

$$
d \mu_ {t} = v _ {\theta} (t, \mu_ {t}, \mathbf {c}) d t + \sqrt {2 \mathcal {F} (t)} d W _ {t}, \quad t \in [ 0, 1 ], \tag {5}
$$

where $\mathcal { F } ( t )$ is a fixed noise schedule and c bundles the conditions of Section 3.2. Concretely $\mathbf { c } = \{ x _ { 0 } , q , H _ { q } , M _ { \mathrm { c a n d } } \}$ gathers the original input, the query-semantic cue, and the admissible region; it acts as the control signal throughout generation.

To merge these cues, at each step the network forms the composite input $[ \mu _ { t } ; \pmb { \varPhi } ( \phi ( t ) ) ; \pmb { \varPsi } ( \mathbf c ) ]$ , where $\phi ( t )$ are Fourier time features broadcast to spatial size by $\varPhi .$ , and $\varPsi ( \mathbf { c } )$ injects the semantic features $( e _ { q } , H _ { q } )$ through cross-attention with $e _ { q } = \operatorname { E n c o d e r } _ { \operatorname { t e x t } } ( q )$ . To keep the path inside $M _ { \mathrm { c a n d } }$ , we discretize with a Projected Euler–Maruyama scheme [36]; each step of size ∆t reads

$$
\mu_ {t + \Delta t} = \mathcal {P} _ {\Omega} \Big (\mu_ {t} + v _ {t}   \Delta t + \sqrt {2 \mathcal {F} (t)}   \Delta W _ {t} \Big), \tag {6}
$$

where $v _ { t } = v _ { \theta } ( t , \mu _ { t } , \mathbf { c } )$ is the predicted instantaneous drift and $\mathcal { P } _ { \varOmega }$ projects back onto the feasible set $\Omega = M _ { \mathrm { c a n d } }$ .

Equation (6) cleanly separates two responsibilities: the conditioned drift $v _ { t }$ learns the semantic direction, while the projection $\mathcal { P } _ { \varOmega }$ enforces spatial feasibility. Running this to completion yields $\mu _ { 1 }$ , a layout that satisfies both the semantic objective and the non-occlusion barrier; we set the placement field $\alpha = \mu _ { 1 }$ .

Next we generate the appearance field of T . During this stage the ink budget enters the style energy $E _ { \mathrm { s t y } }$ as a regularizer; tuning $\lambda _ { \mathrm { s t y } }$ penalizes deviation so that the activated pixel count $\| \mathcal { T } \| _ { 1 }$ drives toward ink(T ). The glyph size thus adapts automatically–long queries draw a larger ink allotment (and may wrap to multiple lines), short queries stay compact–so the text settles naturally into the available space without crowding. Given region and size, line breaking follows automatically. Font and color are likewise carried through a Projected Euler– Maruyama solver: each iteration computes an intermediate noisy state seeded from default font and color values, which the drift then refines.

Training Objective. We learn $v _ { \theta }$ with a two-part loss, $\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { c S B } } + \lambda \mathcal { L } _ { \mathrm { t a s k } }$ jointly fitting the constraint energy and the downstream task.

Constrained flow-matching term $( \mathcal { L } _ { c S B } )$ . Assuming a straight interpolation between $\mu _ { 0 }$ and $\mu _ { 1 }$ , this term regresses the velocity onto the constant transport vector

$$
\mathcal {L} _ {\mathrm{cSB}} = \mathbb {E} _ {t} \left[ \left| \left| v _ {\theta} (t, \mu_ {t}, \mathbf {c}) - (\alpha^ {*} - \mu_ {0}) \right| \right| _ {2} ^ {2} \right], \tag {7}
$$

where $\alpha ^ { * }$ is a pseudo-target for the placement field. Since the placement has no unique optimum, we obtain $\alpha ^ { * }$ by approximate sampling from the energy $E _ { \mathrm { z o n } }$ to build an empirical distribution $\tilde { \mu } _ { 1 } ;$ ; we use Langevin dynamics [13] for this, with details in the supplement.

Task-guided term $( \mathcal { L } _ { t a s k } )$ . To maximize downstream accuracy $( \mathrm { e . g . , V Q A ) }$ , we draw supervision from a frozen MLLM M. With the rendered composite $x _ { 1 } =$ $x _ { 0 }$ ⊕ T and query q, we minimize the negative log-likelihood

$$
\mathcal {L} _ {\mathrm{task}} = - \log P \big (y _ {\mathrm{gt}}, y _ {\mathrm{pred}} = \mathcal {M} (x _ {1}) \big). \tag {8}
$$

Gradients flow back through the discrete trajectory of Eq. (6) using a straightthrough estimator (STE) [2] to bypass the non-differentiable projection, enabling end-to-end refinement of vθ so that the layout is not only geometrically valid but also semantically optimal for the task.

Remark on the hybrid loss. The two terms are complementary. The task-guided term is direct: it back-propagates the downstream signal to improve placement and style. The constraint term is indispensable too: it supplies a placement prior that sharply shrinks the search space during diffusion and speeds convergence.

## 3.4 Implementation Details

We train for 100 epochs on 16 NVIDIA H20 GPUs (batch size 128) with AdamW $( l r = 1 0 ^ { - 4 }$ , weight decay 10−2) and cosine annealing. The ink-budget scale is κ = 80 pixels/character. To dampen layout bias and sampling variance, we apply self-consistency voting: we run K independent inferences per image-query pair and select the final layout by majority vote, with K = 5 in all experiments. Timage is built on PyTorch 2.1, Diffusers 0.24.0, and Qwen2.5-VL.

## 4 Experiments

## 4.1 Datasets and Protocol

We evaluate on VMCBench, which folds 20 diverse VQA datasets (e.g., VQAv2, MathVista, DocVQA) into one multiple-choice format of 9,450 items. The suite covers four axes–General Scene Understanding, Complex Reasoning, Text-Rich & OCR, and Structured Document Understanding–probing robustness under varied spatial and semantic load. Using the unified format, we report Accuracy as the main metric for stability. For each item Timage samples K overlays and votes for the final answer. We present Overall Accuracy and category-wise scores. All runs follow a strict zero-shot protocol with no task-specific fine-tuning, treating cSB as a plug-and-play, inference-time visual-prompting module on frozen MLLMs.

## 4.2 Comparison with Baselines

Table 1 shows that Timage holds a clear, uniform edge across every visual domain, attesting to the robustness of cSB. Despite a compact 7B backbone, Timage reaches a state-of-the-art mean of 87.7%, edging past efficient peers such as Qwen3-8B (87.3%) and decisively beating massive proprietary systems like GPT-4o (+7.4%) and Qwen2-VL-72B (+2.7%). The jump is sharpest where precise spatial alignment matters most: Reasoning (+7.6%) and Doc&Chart (+6.1%), where query-conditioned overlays shrink the attentional search by physically tying intent to relevant regions while avoiding occlusion. Gains also carry to text-dense and general perception, with new bests in OCR (98.5%, +3.2%) and General (+5.7%). Together these results indicate that inscribing the query into the visual geometry through our Constrained Schr¨odinger Bridge yields a stronger reasoning signal than simply enlarging the parameter count, bridging language and vision across domains.

Table 1. Performance comparison on VMCBench. Timage denotes our framework applied to Qwen2.5-VL-7B, with ensemble voting (K = 5).

<table><tr><td>Model</td><td colspan="2">General Reasoning</td><td>OCR</td><td>Doc&amp;Chart</td><td>Avg.</td></tr><tr><td colspan="6">Proprietary / Large-Scale Models (δ30B)</td></tr><tr><td>Qwen2-VL-72B</td><td>88.5</td><td>72.6</td><td>96.8</td><td>90.1</td><td>85.0</td></tr><tr><td>GPT-4o</td><td>85.2</td><td>66.9</td><td>96.4</td><td>83.1</td><td>80.3</td></tr><tr><td>Claude-3.5-Sonnet</td><td>81.3</td><td>62.8</td><td>93.4</td><td>84.6</td><td>77.8</td></tr><tr><td>Molmo-72B</td><td>82.9</td><td>66.6</td><td>94.7</td><td>81.1</td><td>78.7</td></tr><tr><td colspan="6">Open-Source Mid-Scale Models (10B-40B)</td></tr><tr><td>Qwen2.5-VL-14B</td><td>89.8</td><td>72.6</td><td>96.5</td><td>89.9</td><td>85.9</td></tr><tr><td>Cambrian-34B</td><td>83.7</td><td>65.9</td><td>95.7</td><td>73.3</td><td>77.0</td></tr><tr><td>VILA1.5-40B</td><td>82.5</td><td>65.3</td><td>93.2</td><td>67.4</td><td>74.7</td></tr><tr><td>CogVLM2-19B</td><td>78.1</td><td>55.6</td><td>92.3</td><td>72.6</td><td>71.4</td></tr><tr><td colspan="6">Efficient Models (≤8B) &amp; Our Method</td></tr><tr><td>Qwen3-8B</td><td>87.3</td><td>71.8</td><td>96.4</td><td>89.5</td><td>87.3</td></tr><tr><td>Qwen2.5-VL-7B(Base)</td><td>85.6</td><td>68.2</td><td>95.3</td><td>86.5</td><td>79.1</td></tr><tr><td>Qwen2-VL-7B</td><td>84.5</td><td>62.7</td><td>96.4</td><td>80.1</td><td>78.1</td></tr><tr><td>Molmo-7B-D</td><td>73.2</td><td>55.5</td><td>91.7</td><td>72.1</td><td>69.5</td></tr><tr><td>Cambrian-8B</td><td>77.9</td><td>56.4</td><td>91.0</td><td>65.4</td><td>69.6</td></tr><tr><td>Phi-3-Vision</td><td>74.1</td><td>56.4</td><td>90.6</td><td>73.8</td><td>70.3</td></tr><tr><td>LLaVA1.5-7B</td><td>63.6</td><td>44.7</td><td>74.0</td><td>35.0</td><td>51.8</td></tr><tr><td>Timage (Ours)</td><td>91.3</td><td>75.8</td><td>98.5</td><td>92.6</td><td>87.7</td></tr><tr><td colspan="6">(Base: Qwen2.5-VL-7B, K=5) (+5.7) (+7.6) (+3.2) (+6.1) (+8.6)</td></tr></table>

## 4.3 Comparison with Alternative Strategies

To probe Timage from every angle, we line it up against a broad slate of adaptation paradigms on the Qwen2.5-VL-7B backbone over VMCBench. Table 2 groups the baselines into four families: (1) plain text prompting, (2) heuristic spatial overlays, (3) existing (noise-based) visual prompting, and (4) parameterefficient fine-tuning (PEFT). Notably, every baseline except plain text either needs task-specific gradient updates (PEFT, VPT) or leans on ad-hoc placement rules, whereas Timage runs strictly zero-shot and training-free, letting the Constrained Schr¨odinger Bridge produce semantic-aware layouts.

Semantics beat heuristics. Does merely stamping text on the image suffice? Random Text Overlay (79.8%) and Top-Left Placement (79.4%) barely move the needle over text-only (79.1%), and Saliency-Based Placement reaches only 80.5%. Bounding Box Prompts, carrying no semantic text, also stay below 80%. So while location matters, semantic alignment is decisive. Timage’s 87.7% (+7.2% over the best heuristic) confirms that cSB locates regions that are at once spatially admissible and semantically on-point–exactly where geometric rules fail.

Table 2. Comparison with representative fine-tuning paradigms on VMCBench.

<table><tr><td>Method</td><td>Category</td><td>Training Required?</td><td colspan="2">General Reasoning</td><td colspan="3">OCR Doc&amp;Chart Avg.</td></tr><tr><td colspan="8">Standard Baselines</td></tr><tr><td>Base Model (Text Only)</td><td>Text Prompt</td><td>No</td><td>85.6</td><td>68.2</td><td>95.3</td><td>86.5</td><td>79.1</td></tr><tr><td>+ Chain-of-Thought (CoT)</td><td>Text Prompt</td><td>No</td><td>86.1</td><td>69.5</td><td>95.8</td><td>87.2</td><td>80.3</td></tr><tr><td colspan="8">Heuristic &amp; Spatial Visual Prompts</td></tr><tr><td>Random Text Overlay</td><td>Heuristic Prompt</td><td>No</td><td>86.9</td><td>69.5</td><td>95.8</td><td>86.1</td><td>79.8</td></tr><tr><td>Top-Left Text Placement</td><td>Heuristic Prompt</td><td>No</td><td>86.2</td><td>69.0</td><td>95.5</td><td>87.0</td><td>79.4</td></tr><tr><td>Saliency-Based Placement</td><td>Heuristic Prompt</td><td>No</td><td>86.1</td><td>68.5</td><td>96.0</td><td>88.2</td><td>80.5</td></tr><tr><td>Bounding Box Prompt</td><td>Spatial Marker</td><td>No</td><td>86.8</td><td>69.9</td><td>95.2</td><td>86.0</td><td>79.4</td></tr><tr><td colspan="8">Existing Visual Prompting</td></tr><tr><td>VPT-shallow[23]</td><td>Visual Prompt</td><td>Yes (Few-shot)</td><td>86.8</td><td>70.8</td><td>96.1</td><td>88.4</td><td>81.9</td></tr><tr><td>VPT-deep[23]</td><td>Visual Prompt</td><td>Yes (Few-shot)</td><td>87.5</td><td>72.1</td><td>96.4</td><td>89.6</td><td>82.7</td></tr><tr><td>CVPT[21]</td><td>Visual Prompt</td><td>Yes (Few-shot)</td><td>88.4</td><td>73.5</td><td>96.9</td><td>90.8</td><td>83.5</td></tr><tr><td colspan="8">Parameter-Efficient Fine-tuning</td></tr><tr><td>Adapter[4]</td><td>PEFT</td><td>Yes (Full)</td><td>88.2</td><td>72.9</td><td>96.8</td><td>90.1</td><td>83.4</td></tr><tr><td>LoRA[19] (Rank=64)</td><td>PEFT</td><td>Yes (Full)</td><td>89.1</td><td>74.2</td><td>97.2</td><td>91.5</td><td>84.5</td></tr><tr><td>Full Fine-tuning</td><td>Fine-tuning</td><td>Yes (Full)</td><td>89.8</td><td>75.1</td><td>97.5</td><td>92.3</td><td>85.2</td></tr><tr><td>Timage (Ours)</td><td>Text-in-Image</td><td>No</td><td>91.3</td><td>75.8</td><td>98.5</td><td>92.6</td><td>87.7</td></tr></table>

Versus visual prompting. We inscribe natural language, not noise vectors. Constrained CVPT (83.5%) improves on vanilla VPT (82.7%) via geometry, yet still trails Timage by +4.2%. The gap is telling: even optimal spatial constraints cannot supply the semantic grounding that hard reasoning needs. By writing the actual question into the visual manifold, Timage taps the MLLM’s native OCR and grounding priors without any weight update–evidence that semantic fidelity matters more than abstract-representation tuning.

Versus parameter tuning. Strikingly, zero-shot Timage beats every PEFT variant and the full fine-tuning baseline: +3.2% over LoRA (84.5%) and +2.5% over Full FT (85.2%) on average, with clear gains in Reasoning (+1.6%) and Doc&Chart (+0.3%). Unlike PEFT and Full FT, which need task data and risk overfitting across heterogeneous distributions, Timage assembles an optimal visual context at inference. Reforming the input via cSB thus proves more effective than adapting parameters–efficient and broadly generalizable.

## 4.4 Ablation Study

Table 3 isolates each component. From a random-placement base, adding Semantic Pull $( E _ { \mathrm { { s e m } } } )$ lifts accuracy by +2.5%, confirming the value of query-relevant alignment. Activating the Admissible Mask $( M _ { \mathrm { c a n d } } )$ gives the single largest gain $( + 3 . 0 \% )$ , underscoring that avoiding occlusion is critical in dense scenes. Style Regularization $( E _ { \mathrm { s t y } } )$ adds legibility and physical validity (+1.4% combined). Finally, the Stochastic Ensemble (K = 5) captures layout multi-modality for a closing +2.2%, reaching 87.7%.

Table 3. Incremental impact of Timage components on VMCBench.

<table><tr><td>Configuration</td><td>Sem.  $(E_{\text{sem}})$ </td><td>Mask  $(M_{\text{cand}})$ </td><td>Style  $(E_{\text{sty}})$ </td><td>Ens.  $(K = 5)$ </td><td>Avg.</td></tr><tr><td>Random Placement (Base)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>79.1</td></tr><tr><td>+ Semantic Pull</td><td>√</td><td>-</td><td>-</td><td>-</td><td>81.6</td></tr><tr><td>+ Admissible Mask</td><td>√</td><td>√</td><td>-</td><td>-</td><td>84.6</td></tr><tr><td>+ Style Regularization</td><td>√</td><td>√</td><td>√</td><td>-</td><td>86.0</td></tr><tr><td>Timage</td><td>√</td><td>√</td><td>√</td><td>√</td><td>87.7</td></tr></table>

![](images/9e3536daad889d0356851f8ecfa7ac408294c169b6b9827adb4fc66b0d91f525.jpg)

<details>
<summary>natural_image</summary>

Orange tabby cat resting indoors near a window with potted plants and bookshelf in background (no visible text or symbols)
</details>

![](images/cfd559e132fb4bdd318c123ca00a482b27768d5e74143e8880fb2d672bddf6a8.jpg)

<details>
<summary>text_image</summary>

What is by the
window
</details>

Random

![](images/11218405fd30c0fa68d677dc6b3a50fd800838aa64214e56671c8c4df7e1d6ee.jpg)

<details>
<summary>natural_image</summary>

Interior scene with a cat sitting near a window, bookshelves in background (no visible text or symbols)
</details>

A:Book

![](images/05278dc7cb3d5d67331712804afd7b72b06ab4765384e2ce543cfdadfc77b1e8.jpg)

<details>
<summary>text_image</summary>

What is by the window
</details>

Timage

![](images/e25c53fab885cb8f28e28af9acdfa6b5a4e83f112cb40215c071ccd1fe59b816.jpg)

<details>
<summary>natural_image</summary>

Thermal image of a cat with visible heat signature, surrounded by warm lighting and bookshelves in the background (no text or symbols)
</details>

A:Cat  
Fig. 3. Qualitative comparison showing the benefit of Timage. Left: random question embedding covers irrelevant content and misleads the model to “Book”. Right: cSB renders a semantically grounded, non-occluding overlay that preserves foreground saliency and guides attention to the window region, yielding the correct “Cat”.

## 4.5 Qualitative Results

Figure 3 illustrates how Timage steers MLLM reasoning. The baseline’s random inscription happens to cover key content and pulls attention toward an irrelevant background object (answering “Book”). Timage instead lays down a semantically grounded, non-occluding overlay: by seating the query beside the target region it keeps foreground saliency intact and redirects the attention map to the correct object, so the model answers “Cat”. This corroborates that an optimized text layout works as a potent visual prompt for accurate reasoning.

## 5 Analysis

Generalization. To test universality, we attach Timage to five distinct backbones, from the classic LLaVA-1.5 to the recent Qwen3. Table 4 shows uniform improvement, averaging +6.1%. Even on the strong Qwen3-8B (87.3% base), Timage climbs to 92.1% (+4.8%), so the method does more than patch older models–it raises the ceiling of advanced VLMs. Gains concentrate on hard tasks, averaging +6.8% in Reasoning and +5.6% in Doc&Chart. That this holds across CLIP-based, hybrid-ViT, and dynamic-resolution designs points to a shared bottleneck in current VLMs: the disconnect between linguistic queries and visual geometry. By rebuilding the input to align intent with spatial context,

Table 4. Cross-architecture generalization of Timage.

<table><tr><td>Backbone Model</td><td>Base Acc. +</td><td>Timage (Ours) Improvement</td><td> $(\Delta)$  Reasoning</td><td> $\Delta$  Doc&amp;Chart</td><td> $\Delta$ </td></tr><tr><td>LLaVA-1.5-7B</td><td>51.8</td><td>57.4</td><td>+5.6</td><td>+6.8</td><td>+5.2</td></tr><tr><td>Qwen2-VL-7B</td><td>78.1</td><td>83.9</td><td>+5.8</td><td>+7.1</td><td>+6.4</td></tr><tr><td>InternVL-2-8B</td><td>72.5</td><td>78.3</td><td>+5.8</td><td>+6.5</td><td>+5.9</td></tr><tr><td>Qwen2.5-VL-7B</td><td>79.1</td><td>87.7</td><td>+8.6</td><td>+7.6</td><td>+6.1</td></tr><tr><td>Qwen3-8B</td><td>87.3</td><td>92.1</td><td>+4.8</td><td>+5.9</td><td>+4.5</td></tr><tr><td>Average Improvement</td><td>-</td><td>-</td><td>+6.1%</td><td>+6.8%</td><td>+5.6%</td></tr></table>

Timage behaves as a robust, model-agnostic, plug-and-play module that unlocks latent ability in any frozen MLLM.

Efficiency. To gauge parameter efficiency, we compare Timage with LoRA and CVPT across trainable-parameter budgets. As Figure 4 shows, Timage attains higher VMCBench accuracy with far fewer trainable parameters. Even scaled to 40M parameters, LoRA only inches to 88.3, while Timage keeps its lead at minimal cost–exposing the diminishing returns of brute-force parameter growth and reinforcing the value of our constrained, geometry-aware adaptation.

![](images/6aaed0b6f5148c1fb9dd8eb6769729f64dd1af91f21e90b08756f31d4a8962dd.jpg)  
Fig. 4. Efficiency-accuracy trade-offs on VMCBench.

## Limitations and Future Work.

As Figure 5 indicates, while Timage strikes a good performance-efficiency balance, two minor artifacts surface in extreme cases. First, the rendered cue may occasionally be partially incomplete (e.g., a missing stroke); our ex-

![](images/304b1b6c9b753341be47dfaf4d680ab4885911671f97a51e2ac24dcc3c7a6b19.jpg)

<details>
<summary>text_image</summary>

sanyone skiing
</details>

Predict: Yes; GT: Yes

![](images/6ab9764b116aaae45d705a6b625cee008cd54ce1f98322744d4c9319fb8a418c.jpg)

<details>
<summary>natural_image</summary>

A zebra grazing on a grassy field, no visible text or symbols.
</details>

Predict: No; GT: Yes  
Fig. 5. Failure cases of our method.

periments show this rarely dents reasoning accuracy. Second, in heavily textured or low-contrast areas the overlay’s transparency can be suboptimal, letting the cue blend into the background. Future work will refine the renderer to ensure both the completeness and the distinctiveness of the cue without sacrificing efficiency.

## 6 Conclusion

We presented Timage, a paradigm that resolves the representational mismatch between linguistic queries and visual geometry by writing instructions onto the image as semantically grounded, spatially constrained overlays. Casting layout synthesis as a Constrained Schr¨odinger Bridge, Timage aligns intent with visual context while strictly preserving foreground saliency, removing the attentional drift built into token-decoupled designs. On VMCBench, this input-level manifold alignment surpasses both massive proprietary models and full fine-tuning, and it transfers robustly across heterogeneous backbones. The takeaway is that rebuilding the input signal through optimal transport can be a more efficient and potent path to multimodal reasoning than parameter adaptation–opening fresh directions for geometry-aware visual-language interaction.

## References

1. Agiza, A., Neseem, M., Reda, S.: Mtlora: Low-rank adaptation approach for efficient multi-task learning. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16196–16205 (2024)  
2. Bengio, Y., L´eonard, N., Courville, A.: Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432 (2013)  
3. Chen, M., Radford, A., Child, R., Wu, J., Jun, H., Luan, D., Sutskever, I.: Generative pretraining from pixels. In: International Conference on Machine Learning. pp. 1691–1703. PMLR (2020)  
4. Chen, Z., Duan, Y., Wang, W., He, J., Lu, T., Dai, J., Qiao, Y.: Vision transformer adapter for dense predictions. arXiv preprint arXiv:2205.08534 (2022)  
5. Das, R., Dukler, Y., Ravichandran, A., Swaminathan, A.: Learning expressive prompting with residuals for vision transformers. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3366–3377 (2023)  
6. Ding, N., Qin, Y., Yang, G., Wei, F., Yang, Z., Su, Y., Hu, S., Chen, Y., Chan, C.M., Chen, W., et al.: Parameter-efficient fine-tuning of large-scale pre-trained language models. Nature Machine Intelligence 5(3), 220–235 (2023)  
7. Du, Y., Li, S., Mordatch, I.: Compositional visual generation with energy based models. Advances in Neural Information Processing Systems 33, 6637–6647 (2020)  
8. Esser, P., Rombach, R., Ommer, B.: Taming transformers for high-resolution image synthesis. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12873–12883 (2021)  
9. Fang, Z., Hsu, D., Lee, G.H., Lee, G.H.: Neuralized markov random field for interaction-aware stochastic human trajectory prediction. In: International Conference on Learning Representations (2025)  
10. Feng, G., Zhang, B., Gu, Y., Ye, H., He, D., Wang, L.: Towards revealing the mystery behind chain of thought: a theoretical perspective. Advances in Neural Information Processing Systems 36, 70757–70798 (2023)  
11. Gao, P., Geng, S., Zhang, R., Ma, T., Fang, R., Zhang, Y., Li, H., Qiao, Y.: Clipadapter: Better vision-language models with feature adapters. International journal of computer vision 132(2), 581–595 (2024)  
12. Gao, Y., Shi, X., Zhu, Y., Wang, H., Tang, Z., Zhou, X., Li, M., Metaxas, D.N.: Visual prompt tuning for test-time domain adaptation. arXiv preprint arXiv:2210.04831 (2022)  
13. Gillespie, D.T.: The chemical langevin equation. The Journal of Chemical Physics 113(1), 297–306 (2000)  
14. Gu, T., Chen, G., Li, J., Lin, C., Rao, Y., Zhou, J., Lu, J.: Stochastic trajectory prediction via motion indeterminacy diffusion. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 17113–17122 (2022)  
15. Gushchin, N., Kholkin, S., Burnaev, E., Korotin, A.: Light and optimal schr¨odinger bridge matching. In: International Conference on Machine Learning (2024)  
16. Gushchin, N., Selikhanovych, D., Kholkin, S., Burnaev, E., Korotin, A.: Adversarial schr¨odinger bridge matching. Advances in Neural Information Processing Systems 37, 89612–89651 (2024)  
17. Han, C., Wang, Q., Cui, Y., Cao, Z., Wang, W., Qi, S., Liu, D.: Eˆ 2vpt: An effective and efficient approach for visual prompt tuning. arXiv preprint arXiv:2307.13770 (2023)  
18. Han, Z., Gao, C., Liu, J., Zhang, J., Zhang, S.Q.: Parameter-efficient fine-tuning for large models: A comprehensive survey (2024), https://arxiv.org/abs/2403. 14608  
19. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. International Conference on Learning Representations 1(2), 3 (2022)  
20. Hu, Z., Wang, L., Lan, Y., Xu, W., Lim, E.P., Bing, L., Xu, X., Poria, S., Lee, R.: Llm-adapters: An adapter family for parameter-efficient fine-tuning of large language models. In: Conference on Empirical Methods in Natural Language Processing. pp. 5254–5276 (2023)  
21. Huang, L., Mao, J., Yi, J., Tao, Z., Wang, Y.: Cvpt: Cross visual prompt tuning. In: IEEE/CVF International Conference on Computer Vision. pp. 848–858 (2025)  
22. Huang, Q., Dong, X., Chen, D., Zhang, W., Wang, F., Hua, G., Yu, N.: Diversityaware meta visual prompting. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10878–10887 (2023)  
23. Jia, M., Tang, L., Chen, B.C., Cardie, C., Belongie, S., Hariharan, B., Lim, S.N.: Visual prompt tuning. In: European conference on computer vision. pp. 709–727. Springer (2022)  
24. JonathanHo, A., Abbeel, P.: Denoising diffusion probabilistic models. arXiv preprint arXiv:2006.11239 (2020)  
25. Kim, B., Kwon, G., Kim, K., Ye, J.C.: Unpaired image-to-image translation via neural schr¨odinger bridge. arXiv preprint arXiv:2305.15086 (2023)  
26. Kim, J., Kim, B., Ye, J.C.: Latent schrodinger bridge: Prompting latent diffusion for fast unpaired image-to-image translation. arXiv preprint arXiv:2411.14863 (2024)  
27. Kim, J.H., Kim, S., Moon, S., Kim, H., Woo, J., Kim, W.Y.: Discrete diffusion schr¨odinger bridge matching for graph transformation. In: International Conference on Learning Representations. pp. 82925–82971 (2025)  
28. Kodaira, A., Xu, C., Hazama, T., Yoshimoto, T., Ohno, K., Mitsuhori, S., Sugano, S., Cho, H., Liu, Z., Tomizuka, M., et al.: Streamdiffusion: A pipeline-level solution for real-time interactive generation. In: IEEE/CVF International Conference on Computer Vision. pp. 12371–12380 (2025)  
29. Li, C., Chen, Z., Wang, L., Zhu, J.: Audio super-resolution with latent bridge models. arXiv preprint arXiv:2509.17609 (2025)  
30. Liang, Y.S., Li, W.J.: Inflora: Interference-free low-rank adaptation for continual learning. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 23638–23647 (2024)  
31. Liu, G.H., Vahdat, A., Huang, D.A., Theodorou, E.A., Nie, W., Anandkumar, A.: I2SB: Image-to-Image Schr¨odinger Bridge. arXiv preprint arXiv:2302.05872 (2023)  
32. Liu, S.Y., Wang, C.Y., Yin, H., Molchanov, P., Wang, Y.C.F., Cheng, K.T., Chen, M.H.: Dora: Weight-decomposed low-rank adaptation. In: Forty-first International Conference on Machine Learning (2024)  
33. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems 35, 5775–5787 (2022)  
34. Luo, S., Tan, Y., Patil, S., Gu, D., Von Platen, P., Passos, A., Huang, L., Li, J., Zhao, H.: Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556 (2023)  
35. Van den Oord, A., Kalchbrenner, N., Espeholt, L., Vinyals, O., Graves, A., et al.: Conditional image generation with pixelcnn decoders. Advances in Neural Information Processing Systems 29 (2016)  
36. Pierret, F.: A non-standard-euler–maruyama scheme. Journal of Difference Equations and Applications 22(1), 75–98 (2016)  
37. Qiu, X., Yang, M., Ma, X., Li, F., Liang, D., Luo, G., Wang, W., Wang, K., Li, S.: Finding local diffusion schrodinger bridge using kolmogorov-arnold network. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition Conference. pp. 23227–23236 (2025)  
38. Razavi, A., Van den Oord, A., Vinyals, O.: Generating diverse high-fidelity images with vq-vae-2. Advances in Neural Information Processing Systems 32 (2019)  
39. Reed, S., Oord, A., Kalchbrenner, N., Colmenarejo, S.G., Wang, Z., Chen, Y., Belov, D., Freitas, N.: Parallel multiscale autoregressive density estimation. In: International Conference on Machine Learning. pp. 2912–2921 (2017)  
40. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Conference on Computer Vision and Pattern Recognition. pp. 10684–10695 (2022)  
41. Shi, Y., De Bortoli, V., Campbell, A., Doucet, A.: Diffusion schr¨odinger bridge matching. Advances in neural information processing systems 36, 62183–62223 (2023)  
42. Shih, A., Belkhale, S., Ermon, S., Sadigh, D., Anari, N.: Parallel sampling of diffusion models. Advances in Neural Information Processing Systems 36, 4263–4276 (2023)  
43. Sohn, K., Chang, H., Lezama, J., Polania, L., Zhang, H., Hao, Y., Essa, I., Jiang, L.: Visual prompt tuning for generative transfer learning. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19840–19851 (2023)  
44. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020)  
45. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 (2020)  
46. Su, X., Song, J., Meng, C., Ermon, S.: Dual diffusion implicit bridges for imageto-image translation. arXiv preprint arXiv:2203.08382 (2022)  
47. Sung, Y.L., Cho, J., Bansal, M.: Vl-adapter: Parameter-efficient transfer learning for vision-and-language tasks. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5227–5237 (2022)  
48. Tang, Z., Hang, T., Gu, S., Chen, D., Guo, B.: Simplified diffusion schr\” odinger bridge. arXiv preprint arXiv:2403.14623 (2024)  
49. Tian, K., Jiang, Y., Yuan, Z., Peng, B., Wang, L.: Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems 37, 84839–84865 (2024)  
50. Van Den Oord, A., Vinyals, O., et al.: Neural discrete representation learning. Advances in neural information processing systems 30 (2017)  
51. Wang, S., Yu, L., Li, J.: Lora-ga: Low-rank adaptation with gradient approximation. Advances in Neural Information Processing Systems 37, 54905–54931 (2024)  
52. Wang, Z., Wang, Y., Cai, Y.: Cure or poison? embedding instructions visually alters hallucination in vision-language models. arXiv preprint arXiv:2508.01678 (2025)  
53. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems 35, 24824–24837 (2022)  
54. Xu, L., Xie, H., Qin, S.J., Tao, X., Wang, F.L.: Parameter-efficient fine-tuning methods for pretrained language models: A critical review and assessment. IEEE Transactions on Pattern Analysis and Machine Intelligence (2026)  
55. Yu, J., Li, X., Koh, J.Y., Zhang, H., Pang, R., Qin, J., Ku, A., Xu, Y., Baldridge, J., Wu, Y.: Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627 (2021)  
56. Yu, J., Xu, Y., Koh, J.Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B.K., et al.: Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789 2(3), 5 (2022)  
57. Zanella, M., Ben Ayed, I.: Low-rank few-shot adaptation of vision-language models. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1593–1603 (2024)  
58. Zhang, R., Fang, R., Zhang, W., Gao, P., Li, K., Dai, J., Qiao, Y., Li, H.: Tip-adapter: Training-free clip-adapter for better vision-language modeling. arXiv preprint arXiv:2111.03930 (2021)  
59. Zhang, X., Du, C., Pang, T., Liu, Q., Gao, W., Lin, M.: Chain of preference optimization: Improving chain-of-thought reasoning in llms. Advances in Neural Information Processing Systems 37, 333–356 (2024)  
60. Zhang, Y., Su, Y., Liu, Y., Wang, X., Burgess, J., Sui, E., Wang, C., Aklilu, J., Lozano, A., Wei, A., et al.: Automated generation of challenging multiplechoice questions for vision language model evaluation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition Conference. pp. 29580–29590 (2025)