# Cycle-World: Mitigating Error Accumulation in Long-term Video World Models via Reverse-Prediction Cycle Consistency

Zihan Su<sup>1⋆</sup> , Teng Hu<sup>1⋆</sup> , Jiangning Zhang<sup>2</sup> , Ruiyan Wang<sup>1</sup> , Ran Yi<sup>1†</sup> , Lizhuang Ma<sup>1†</sup> , and Dacheng Tao<sup>3</sup>

School of Computer Science, Shanghai Jiao Tong University, Shanghai, China 2 Institute of Cyber-Systems and Control, Zhejiang University, Hangzhou, China <sup>3</sup> Nanyang Technological University, Singapore https://szhcz.github.io/projects/Cycle-World/

Abstract. Autoregressive difusion models have enabled high-quality video generation, yet their sequential nature inherently sufers from error accumulation. In long-horizon video synthesis, minor prediction deviations compound over time, inevitably leading to unconstrained generative drift, structural collapse, and severe visual degradation. To address this, we propose Cycle-World, a novel framework designed for stable and temporally consistent long-video generation. Our approach tackles error drift by enforcing strict temporal reversibility across both the training and inference phases. Theoretically, we demonstrate that forward generative drift can be strictly bottlenecked by a cycle-consistency objective. During training, we integrate an eficient reverse-prediction model to implicitly embed causal constraints into the forward generator, compelling it to produce reversible sequences that tightly adhere to the natural video manifold. At inference time, we repurpose this frozen reverse model as a runtime corrector. Through gradient-based cycle guidance, it iteratively refines the generated latent representations, actively suppressing accumulated errors before they are committed to the historical context. Extensive experiments on the VBench benchmark demonstrate that Cycle-World’s dual-phase synergy significantly mitigates error drift, achieving state-of-the-art overall generation quality and long-horizon temporal consistency in 60-second synthesis.

Keywords: Video generation · Cycle consistency · Error accumulation

## 1 Introduction

The field of video generation has witnessed unprecedented advancements recently, driven by powerful models such as Sora [36, 37], Seedance [12, 41], and

![](images/1b39c2f3a44681393128e5ff7e0f3a22ac611721cc9832510a9c23394f36a5c0.jpg)  
Fig. 1: High-fidelity, long-horizon video generation with Cycle-World. By efectively bottlenecking generative drift via temporal reversibility, our framework strictly suppresses structural hallucinations inherent in forward-only models. Cycle-World maintains state-of-the-art visual quality, strict physical conservation, and temporally consistent object states over extended generation horizons.

Kling [44], alongside pioneering open-source eforts like Wan [45], Hunyuan-Video [29]. While these models primarily rely on bidirectional or full-sequence architectures that achieve remarkable visual quality, their non-causal nature fundamentally limits their flexibility for open-ended, sequential, and interactive generation. As the community pivots towards the paradigm of World Models [17, 25, 34, 42], there is a critical consensus that real-time interactivity, continuous generation, and causal reasoning are indispensable. Consequently, the field is experiencing a paradigm shift towards causal, autoregressive generation models [16, 23, 57, 61].

However, this shift towards causal autoregressive models introduces severe bottlenecks in long video generation. A primary challenge is the train-inference mismatch, often leading to rapid accumulation of errors. Traditionally, models are trained using Teacher-Forcing, which strictly relies on ground-truth past frames, causing significant exposure bias: during inference, the autoregressive model relies on its own past predictions, where minor deviations — unseen during training — propagate and amplify. To bridge this gap and mitigate drift, the community has continuously evolved towards more robust training paradigms. This progression spans from the introduction of Difusion Forcing [5] to recent advanced train-inference alignment strategies, such as Self-Forcing [23] and LongLive [51].

Nevertheless, we observe that even with these sophisticated mitigations, forward-only generation still sufers from a more fundamental and destructive issue: structural hallucinations. While existing progressive methods efectively suppress generic noise accumulation, they operate strictly in a unidirectional, forward-time manner. Crucially, they lack temporal cycle consistency—the explicit temporal constraints required to ensure that a generated state logically allows its past to have happened, such that past states could be reversely predicted from future states. Without this bidirectional verification, the models lack the physical constraints necessary to maintain continuous object states. Consequently, they are prone to severe video distortion and non-physical artifacts— such as characters clipping through solid objects, entities spontaneously appearing, or objects vanishing without a trace. Unlike generic noise, these structural hallucinations represent irreversible violations of real-world physics that abruptly destroy the integrity of the generated sequence.

To fundamentally address these irreversible structural hallucinations, we propose Cycle-World, a unified framework that enforces physical consistency via temporal reversibility. Our core motivation is grounded in a fundamental premise: if a generated causal sequence obeys real-world dynamics, it must be temporally reversible. Based on this insight, we first establish the Cycle-Bounded Drift (CBD) theorem, formally proving that the unconstrained error accumulation in forward autoregressive synthesis can be strictly bottlenecked by minimizing the reverse reconstruction error. Guided by this theoretical guarantee, we translate the mathematical bound into a practical Cycle-Consistent Learning (CCL) paradigm. By introducing a reverse-prediction branch, we explicitly constrain the forward generator to produce inherently reversible latents, enabling it to foresee and suppress non-physical artifacts during training. Furthermore, to extend the applicability of our theory to pre-trained models and resource-constrained scenarios, we propose Cycle-Guided Inference (CGI). This inference-time strategy repurposes the reverse model as a runtime critic to iteratively refine latents, ofering a plug-and-play solution that significantly boosts stability without the need for expensive architectural modifications.

Extensive experiments conducted on VBench validate the efectiveness of our framework. Cycle-World significantly mitigates error drift, achieving stateof-the-art visual fidelity, semantic consistency, and overall temporal stability in long-video generation, as shown in Fig. 1.

In summary, our main contributions are threefold:

– We establish the Cycle-Bounded Drift (CBD) theorem, a theoretical framework demonstrating that unconstrained generative drift and structural hallucinations in forward autoregressive synthesis can be strictly bottlenecked by enforcing temporal reversibility.

– We propose a Cycle-Consistent Learning (CCL) paradigm. By introducing a novel reverse-prediction cycle-consistency loss $( \mathcal { L } _ { \it c y c l e } )$ , we explicitly constrain the forward causal generator to internalize physical conservation and maintain long-term structural integrity.

– We introduce Cycle-Guided Inference (CGI), a zero-shot runtime optimization strategy. By repurposing the frozen reverse model as a runtime critic, CGI utilizes iterative gradient-based latent refinement to actively rectify non-physical artifacts, significantly enhancing long-term generation quality without architectural modifications to the forward model.

## 2 Related Work

## 2.1 Video Generation

Difusion models [14,35,43], particularly Difusion Transformers (DiT) [38], have established the prevailing paradigm for video generation [21,29,45,52,60], achieving remarkable visual fidelity. This rapid progress spans various specialized capabilities, including multimodal customized generation [4, 19, 20], and native high-resolution synthesis [22,48], and spatiotemporally consistent video processing [27, 32]. Furthermore, recent advancements have expanded into joint audiovisual generation, leveraging cross-modal interactions and cross-task synergy to achieve synchronized multi-sensory synthesis [18, 30, 33, 46, 58]. However, de spite these visual and multimodal achievements, their reliance on non-causal, bidirectional attention for concurrent frame denoising restricts their flexibility for open-ended generation. Conversely, pure autoregressive (AR) models [15,28] ofer sequential flexibility via discrete next-token prediction but sufer from irreversible information loss during latent compression and suboptimal generation diversity.

To bridge this gap, recent works explore hybrid Autoregressive Difusion architectures [10, 11, 23, 26, 57], which temporally decompose generation by conditioning future frames on past context. Despite their promise, these models sufer from severe exposure bias. During iterative inference, conditioning on imperfect prior predictions causes minor deviations to amplify continuously. This catastrophic error accumulation constitutes the primary bottleneck our work addresses.

## 2.2 Long Video Generation

Early long-video approaches relied on generating overlapping clips [8, 39, 47] or performing temporal interpolation between sparse keyframes [3, 54]. While extending the generation window, these heuristic designs fail to achieve true, infinitely long streaming synthesis. Consequently, the focus has shifted toward native causal modeling. However, traditional training paradigms for these models often employ Teacher Forcing [40, 50], which inevitably introduces a severe distribution discrepancy between the training and inference stages. To alleviate this issue, Difusion Forcing [5] proposes joint denoising optimization for tokens with independent noise levels, which SkyReels-V2 [6] further integrates with Multi-modal Large Language Models to facilitate infinite-length, cinematic video synthesis. CausVid [57] extends distribution matching distillation [55, 56] to the video domain to mitigate error accumulation in AR generation. To resolve exposure bias and bridge the train-test gap, Self-Forcing [23] innovatively simulates inference conditions directly during training by executing autoregressive rollouts with a Key-Value cache, optimizing the model conditioned on its own historically generated outputs. Most recent cutting-edge works build upon this foundation. LongLive [51], for instance, adopts streaming long-sequence finetuning to strictly maintain end-to-end "train-long-test-long" consistency.

![](images/c6facc76824137d3ab282943ec55d30b863a6c3a3c5a088941a8d58ec9062ad8.jpg)

![](images/ee402020dbe653550d50d3e279335f1e929e69441ddd1333a328066050678f71.jpg)  
(b) Cycle-Guided Inference(CGI) Runtime Optimization  
Fig. 2: Overview of Cycle-World. We mitigate generative drift in autoregressive video synthesis by enforcing temporal reversibility. (a) Training (CCL): The forward generator $G _ { \theta }$ and reverse model $R _ { \phi }$ are jointly optimized. A latent cycle-consistency loss $( \mathcal { L } _ { c y c l e } )$ explicitly penalizes physically irreversible trajectories. (b) Inference (CGI): The frozen $R _ { \phi }$ acts as a runtime corrector. By evaluating cycle discrepancy (D), it performs iterative gradient-based latent refinement to actively prune accumulated errors before they enter the historical context.

## 3 Methodology

The overarching goal of Cycle-World is to mitigate the unconstrained generative drift and structural hallucinations inherent in long-term autoregressive video synthesis. To achieve this, we introduce a novel framework grounded in the principle of temporal reversibility—the intuition that physically valid and structurally sound video dynamics must be accurately reversible. As illustrated in Figure 2, our approach addresses this challenge through a cohesive pipeline encompassing theoretical grounding, cycle-consistent learning, and inference-time optimization. Specifically, we first establish the Cycle-Bounded Drift (CBD) theorem (Sec. 3.1), a theoretical foundation demonstrating that the forward generative error can be strictly constrained by minimizing a reverse-prediction cycle-consistency error. Guided by this insight, we propose a Cycle-Consistent Learning (CCL) paradigm (Sec. 3.2). In this phase, we introduce a reverseprediction model $R _ { \phi }$ to enforce a cycle-consistency loss $( \mathcal { L } _ { c y c l e } )$ on the forward causal generator $G _ { \theta } ,$ , explicitly penalizing irreversible structural hallucinations.

Finally, to combat out-of-distribution perturbations during open-ended generation, we introduce Cycle-Guided Inference (Sec. 3.3). Under this strategy, we repurpose the frozen reverse model as a runtime corrector. By actively evaluating cycle discrepancy, it performs iterative gradient-based latent refinement to prune non-physical artifacts before they are committed to the historical context.

## 3.1 Theoretical Foundation: Bounding Generative Drift via Temporal Reversibility

Let $Z = \{ z _ { 1 } , z _ { 2 } , \dots , z _ { N } \}$ denote the ground-truth latent sequence of a natural video, where each $z _ { n }$ is a latent chunk. In standard autoregressive synthesis, a forward causal generator $G _ { \theta }$ sequentially predicts $\hat { z } _ { n }$ conditioned on the previously generated context $\hat { z } _ { < n }$ . Because this sequential generation is inherently unconstrained, minor prediction deviations compound at each step n, leading to unbounded generative drift and structural collapse in long video synthesis.

We theoretically argue that this generative drift can be strictly constrained by enforcing temporal reversibility. Because natural video dynamics obey physical laws and spatiotemporal causality, they are inherently reversible. If a generated frame $\hat { z } _ { n }$ severely deviates from the natural manifold, it loses the causal information necessary to reconstruct its history.

Let $e _ { n } = \| \hat { z } _ { n } - z _ { n } \|$ be the accumulated generative drift at step n. To constrain this, we introduce a reverse-prediction model $R _ { \phi }$ mapping a state at n back to $n - 1$ . We establish our theoretical guarantees based on two mild assumptions:

Assumption 1 (Reverse Predictability of Natural Dynamics) The reverse model $R _ { \phi }$ is comprehensively trained under a self-forcing paradigm [23]. Given this aligned training scheme, its approximation error for short-horizon reverse prediction is bounded by a small constant $\epsilon _ { R } > 0$ . That is, $\| R _ { \phi } ( z _ { n } ) - z _ { n - 1 } \| \leq \epsilon _ { R }$

Assumption 2 (Reverse-Lipschitz Continuity) The learned reverse mapping $R _ { \phi }$ preserves the distance properties within the relevant latent manifold, satisfying a reverse-Lipschitz condition. There exists a constant $C > 0$ such that for any two states $z _ { a }$ and $z _ { b } , \| z _ { a } - z _ { b } \| \leq C \| R _ { \phi } ( z _ { a } ) - R _ { \phi } ( z _ { b } ) \|$

Under these conditions, we demonstrate that the forward generative error $e _ { n }$ is constrained by the cycle-consistency objective and the error from the previous steps. We formally define the single-step cycle-consistency distance as $d _ { c y c l e } ^ { ( n ) } =$ $\lVert \hat { z } _ { n - 1 } - R _ { \phi } ( \hat { z } _ { n } ) \rVert$

Due to space constraints, all detailed proofs for the theoretical results presented in this section are deferred to Supplementary Material.

Theorem 1 (Cycle-Bounded Drift). Under Assumptions 1 and 2, the forward generative drift $e _ { n }$ satisfies:

$$
e _ {n} \leq C \left(d _ {c y c l e} ^ {(n)} + e _ {n - 1} + \epsilon_ {R}\right).\tag{1}
$$

To understand the compounding drift over a long horizon, we recursively unroll this step-wise recurrence.

Corollary 1 (Long-Horizon Error Bound). Assuming a worst-case upper limit on the single-step cycle-consistency distance, ma $\mathfrak { c } _ { i } d _ { c y c l e } ^ { ( i ) } \le \delta _ { c y c l e }$ , the total generative drift at step n (for $C \neq 1 )$ is explicitly governed by:

$$
e _ {n} \leq C ^ {n} e _ {0} + (\delta_ {c y c l e} + \epsilon_ {R}) C \frac {C ^ {n} - 1}{C - 1}.\tag{2}
$$

Building on Corollary 1, next we show Cycle-World framework is theoretically superior to unconstrained baselines. Let $\begin{array} { r } { \delta _ { u n c } = \operatorname* { m a x } _ { i } \| \hat { z } _ { i - 1 } ^ { u n c } - R _ { \phi } ( \hat { z } _ { i } ^ { u n c } ) \| } \end{array}$ ∥ represent the maximum local cycle error of a standard, unconstrained generator, and $\delta _ { c y c l e }$ represent our constrained error, where $\delta _ { c y c l e } \ll \delta _ { u n c }$

Proposition 1 (Theoretical Advantage over Unconstrained Baselines). Given the identical initial drift condition $e _ { 0 }$ and the same pre-trained reverse model $R _ { \phi }$ (with properties defined in Assumptions 1 and 2), let $E _ { n } ^ { u n c }$ and $E _ { n } ^ { o u r s }$ denote the theoretical upper limits of the generative drift at step n for the unconstrained baseline and our constrained method, respectively. The explicit drift reduction (the gap between these theoretical limits) achieved by our method is:

$$
\varDelta E _ {n} = E _ {n} ^ {u n c} - E _ {n} ^ {o u r s} = (\delta_ {u n c} - \delta_ {c y c l e}) C \frac {C ^ {n} - 1}{C - 1} > 0.\tag{3}
$$

Remark. Proposition 1 is the theoretical cornerstone of our method. The term $\left( \delta _ { u n c } - \delta _ { c y c l e } \right)$ represents the single-step advantage gained by explicitly enforcing temporal reversibility. Crucially, the multiplier $\overset { \smile } { C } \frac { C ^ { n } - 1 } { C - 1 }$ implies that this advantage is not merely additive, but scales with the sequence length n. This proves that while standard generation and our method might perform similarly for very short clips, the unconstrained baseline will inevitably sufer from a much faster error explosion over time. By minimizing the single-step cycle error upper bound $\delta _ { c y c l e }$ , Cycle-World efectively suppresses the base magnitude of the accumulated generative drift, theoretically guaranteeing significantly better structural preservation in long-video generation.

## 3.2 Cycle-Consistent Learning: Enforcing Cycle Consistency via Reverse Prediction

Constraining the Forward Model via Temporal Cycle Consistency. Building upon the theoretical guarantees established in Section 3.1—specifically Proposition 1, which proves that bounding single-step cycle error mathematically curtails compounding generative drift—we translate this insight into a practical learning framework. we extend a forward autoregressive video generator $G _ { \theta }$ predicting latent chunks $\hat { z } _ { n }$ given history $\hat { z } _ { < n }$ . Since standard maximum likelihood training optimizes solely forward prediction, unconstrained reverse consistency causes unbounded error accumulation scaling with $C { \frac { C ^ { n } - 1 } { C - 1 } }$ over sequence length n (Corollary 1). To minimize this bound and mitigate structural hallucinations, we introduce a reverse-prediction model $R _ { \phi }$ enforcing temporal cycle consistency.

The Pixel-Latent Mismatch Bottleneck. Implementing $R _ { \phi }$ naively by training a separate autoregressive model on reversed videos incurs severe dual distillation overhead and faces a prohibitive structural barrier: pixel-latent mismatch. Because Video VAEs temporally compress continuous frame sequences into a compact latent representation, the latent features of a forward sequence do not exhibit simple temporal symmetry with those of a reversed sequence. This mismatch precludes the direct computation of the cycle-consistency distance, $d _ { c y c l e } ^ { ( n ) } = \| \hat { z } _ { n - 1 } - R _ { \phi } ( \hat { z } _ { n } ) \|$ defined in Theorem 1. Aligning the forward-ordered chunk $\hat { z } _ { n - 1 }$ and a reverse-predicted chunk residing in disjoint manifolds requires an exorbitant Decode-Flip-Encode loop of decoding, temporally flipping, and re-encoding, rendering joint training computationally infeasible.

Our Solution: Intrinsic Latent Reversibility. To fundamentally circumvent this bottleneck and the inherent feature mismatch, we propose Intrinsic Latent Reversibility, redefining the cycle consistency objective directly on the intrinsic latent manifold rather than mapping back to the extrinsic pixel domain. $R _ { \phi }$ need not predict decoded, temporally flipped frames. Instead, we formulate the reverse task as learning the inverse transition dynamics within the latent space. We set the optimization target of $R _ { \phi } ( \hat { z } _ { n } )$ to the forward-ordered history $\hat { z } _ { n - 1 }$ , learning the inverse probability $P ( z _ { n - 1 } | \hat { z } _ { n } )$ directly. This approach eliminates the need for “Decode-Flip-Encode" loop. Consequently, the cycle-consistency distance $d _ { c y c l \epsilon } ^ { ( n ) }$ e reduces to a direct vector subtraction in the shared latent space. This allows us to enforce strict structural constraints with negligible computational overhead, making joint training feasible.

Forward Generation via Self-Forcing. To bridge the train-inference discrepancy, we train $G _ { \theta }$ via self-forcing to predict the current chunk $\hat { z } _ { n }$ given its generated history $\hat { z } _ { < n }$ instead of ground truth. Since Distribution Matching Distillation (DMD) lacks paired regression constraints, optimizing this process updates the forward objective $\mathcal { L } _ { f w d }$ using its score-diference gradient:

$$
\nabla_ {\theta} \mathcal {L} _ {f w d} = - \mathbb {E} _ {t, \epsilon} \left[ (s _ {r e a l} (\hat {z} _ {n, t}, t) - s _ {f a k e} (\hat {z} _ {n, t}, t)) \frac {\partial G _ {\theta} (\hat {z} _ {<   n})}{\partial \theta} \right],\tag{4}
$$

where $\hat { z } _ { n , t }$ is the noisy latent at difusion timestep t. While this gradient update ensures high-fidelity chunk generation, it lacks explicit penalties for the structural drift that accumulates over long sequences.

Latent Cycle-Consistency Objective. To bound this generative drift, we implement the proposed Intrinsic Latent Reversibility via a reverse-prediction model $R _ { \phi }$ . As established, $R _ { \phi }$ operates directly on the latent manifold, tasked with reconstructing the preceding latent $z _ { n - 1 }$ given the current generation ${ \hat { z } } _ { n } .$ Formally, let $\hat { z } _ { n }$ be the latent chunk synthesized by the forward generator. The reverse model attempts to predict the immediate history $\tilde { z } _ { n - 1 } = R _ { \phi } ( \hat { z } _ { n } )$ . The cycle-consistency objective is defined as the expected squared Euclidean distance between the actual autoregressive conditioning context used by the forward model and the reconstructed history inferred by the reverse model:

$$
\mathcal {L} _ {c y c l e} = \mathbb {E} _ {\hat {z} _ {n} \sim G _ {\theta}} \left[ \| \hat {z} _ {n - 1} - R _ {\phi} (\hat {z} _ {n}) \| _ {2} ^ {2} \right].\tag{5}
$$

Minimizing this objective explicitly enforces the invertibility assumption (Assumption 2), ensuring that the generated $\hat { z } _ { n }$ retains suficient causal information to recover its origin, thereby preventing error accumulation.

Joint Optimization. The final training objective seamlessly integrates the distribution-level supervision with our structural cycle constraint:

$$
\mathcal {L} _ {t o t a l} = \mathcal {L} _ {f w d} + \lambda \mathcal {L} _ {c y c l e},\tag{6}
$$

where λ is a hyperparameter scaling the penalty strength. Crucially, during backpropagation, the gradients from $\mathcal { L } _ { \mathit { c y c l e } }$ flow through the reverse model $R _ { \phi }$ and back into the forward generator $G _ { \theta }$ . This mechanism implicitly endows $G _ { \theta }$ with foresight—it penalizes the generation of physically implausible artifacts (like disappearing objects) that, while locally reasonably under ${ \mathcal { L } } _ { f w d } ,$ fail to accurately reconstruct their history, explicitly pruning divergent trajectories during the learning phase.

## 3.3 Cycle-Guided Inference: Optimizing Generative Latents via Runtime Guidance

While the proposed Cycle-Consistent Learning paradigm ensures the generator intrinsically preserves temporal reversibility, it requires full-scale parameter retraining. In the era of large-scale foundation models, such retraining is often computationally prohibitive or practically infeasible due to closed-source weights. To address this limitation and extend the benefits of temporal reversibility to broader scenarios, we introduce Cycle-Guided Inference (CGI), a zeroshot, training-free latent optimization strategy. Crucially, this strategy is modelagnostic: as long as the target model and the reverse corrector operate within the same latent manifold (sharing the same Video VAE), CGI can be seamlessly plugged into any of-the-shelf autoregressive video generator, enabling structural hallucinations mitigation without updating a single model parameter.

Cycle Guidance in Latent Space. During the autoregressive inference phase, the weights of both the forward generator $G _ { \theta }$ and the reverse model $R _ { \phi }$ are strictly frozen. While earlier sections abstract the generation of the n-th block as a single output $z _ { n } ,$ , actual synthesis in difusion models involves an iterative denoising process over timesteps $\left\{ t _ { T } , \ldots , t _ { 1 } \right\}$ . Unlike conventional decoding that passively accepts forward predictions, we actively rectify accumulated errors at intermediate difusion timesteps committing them to the historical context bufer (KV cache). Let $z _ { n , t }$ denote the latent state at difusion timestep t. First, the forward generator $G _ { \theta }$ predicts the corresponding clean latent $\hat { z } _ { n | t } = G _ { \theta } ( z _ { n , t } , t , \mathcal { H } )$ based on the historical context H. To evaluate the physical plausibility of this prediction, we compute the cycle discrepancy D. Specifically, the reverse evaluation is formulated as a two-stage autoregressive process. The predicted clean latent is temporally flipped, denoted by the operator $\mathcal F ( \cdot )$ , and processed by the reverse model at a fixed context noise level $t _ { \mathrm { c t x } }$ to construct a reverse contextual cache ${ \mathcal { H } } _ { \mathrm { r e v } }$ . Subsequently, the reverse model utilizes this newly constructed cache to predict the clean predecessor state from standard Gaussian noise $\epsilon ,$ conditioned on the initial difusion timestep $t _ { T }$ . The discrepancy is defined as the Euclidean distance between the causal condition $\hat { z } _ { n - 1 }$ (the confirmed output of the previous block) and the time-flipped reverse reconstruction:

$$
\begin{array}{r l} & R _ {\phi} (\mathcal {F} (\hat {z} _ {n | t} ^ {(k)}), t _ {\mathrm{ctx}}, \mathcal {H} _ {\mathrm{rev}} ^ {(k)}), \quad \tilde {z} _ {n - 1} ^ {(k)} = \mathcal {F} (R _ {\phi} (\epsilon , t _ {T}, \mathcal {H} _ {\mathrm{rev}} ^ {(k)})), \\ & \qquad \mathcal {D} (z _ {n, t} ^ {(k)}) = \left\| \hat {z} _ {n - 1} - \tilde {z} _ {n - 1} ^ {(k)} \right\| _ {2} ^ {2}, \end{array}\tag{7}
$$

where $\hat { z } _ { n | t } ^ { ( k ) } = G _ { \theta } ( z _ { n , t } ^ { ( k ) } , t , \mathcal { H } ) , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ , and k is optimization iteration index.

Gradient-Based Latent Refinement. Since forward prediction and reverse reconstruction are fully diferentiable processes, we can iteratively refine the latent state via cycle guidance. To maximize eficiency and structural impact, this optimization is strategically applied only during a specific window of early denoising timesteps (from $T _ { \mathrm { s t a r t } }$ to $T _ { \mathrm { e n d } } )$ ). We compute the gradient of the cycle discrepancy with respect to the current state $z _ { n , t } ^ { ( k ) }$ and perform gradient descent:

$$
z _ {n, t} ^ {(k + 1)} = z _ {n, t} ^ {(k)} - \eta \nabla_ {z _ {n, t} ^ {(k)}} \mathcal {D} (z _ {n, t} ^ {(k)}),\tag{8}
$$

where η is the optimization step size. After K iterations, the refined state $z _ { n , t } ^ { ( K ) }$ is detached from the computation graph. This optimized state is then passed to the standard difusion transition function Ψ to obtain the latent state for the subsequent timestep. Upon reaching the final denoising step $t _ { 1 } .$ , the resulting clean latent $\hat { z } _ { n }$ is appended to the historical context bufer to guide the generation of the next block. This active rectification mechanism acts as a structural bottleneck, preventing inherent drift from manifesting as visual degradation. The complete inference procedure is summarized in Supplementary Material.

## 4 Experiments

## 4.1 Experiment Settings

Baselines. We evaluate the proposed method against several video generation baselines, categorized by their architectural paradigms. For bidirectional difusion and transformer models, we compare with LTX-Video [13] and Wan2.1 [45]. Within the autoregressive family, we evaluate general-purpose models of varying scales, including NOVA [10] (0.6B), SkyReels-V2 [6] (1.3B), Pyramid Flow [26] (2B), and MAGI-1 [11] (4.5B). To ensure a direct comparison with models sharing our foundational architecture and eficient training setup, we include the 1.3B parameter distilled few-step generators CausVid [57] and chunk-wise Self Forcing [23]. Finally, to assess performance over extended sequences, we compare against models designed for long video generation: LongLive [51], Self Forcing++ [9], Rolling Forcing [31], Infinity-RoPE [53], and Context Forcing [7]. Evaluation Metrics. We report the performance on VBench [24, 59] following [7, 23, 51]. To assess physical consistency, we use two metrics. The first is

Table 1: Quantitative comparison on the VBench for 5s and 60s video generation.

<table><tr><td rowspan="2">Model</td><td rowspan="2">#Params</td><td colspan="3">Evaluation scores on 5s ↑</td><td colspan="3">Evaluation scores on 60s ↑</td></tr><tr><td colspan="2">Total Quality</td><td>Semantic</td><td colspan="2">Total Quality</td><td>Semantic</td></tr><tr><td colspan="8">Bidirectional models</td></tr><tr><td>LTX-Video [13]</td><td>1.9B</td><td>80.00</td><td>82.30</td><td>70.79</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Wan2.1 [45]</td><td>1.3B</td><td>84.26</td><td>85.30</td><td>80.09</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="8">Autoregressive models</td></tr><tr><td>SkyReels-V2 [6]</td><td>1.3B</td><td>82.67</td><td>84.70</td><td>74.53</td><td>70.47</td><td>75.30</td><td>51.15</td></tr><tr><td>MAGI-1 [11]</td><td>4.5B</td><td>79.18</td><td>82.04</td><td>67.74</td><td>69.87</td><td>76.12</td><td>44.87</td></tr><tr><td>CausVid [57]</td><td>1.3B</td><td>81.20</td><td>84.05</td><td>69.80</td><td>71.04</td><td>76.80</td><td>48.01</td></tr><tr><td>NOVA [10]</td><td>0.6B</td><td>80.12</td><td>80.39</td><td>79.05</td><td>65.25</td><td>70.25</td><td>45.24</td></tr><tr><td>Pyramid Flow [26]</td><td>2B</td><td>81.72</td><td>84.74</td><td>69.62</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Self Forcing, chunk-wise [23]</td><td>1.3B</td><td>84.31</td><td>85.07</td><td>81.28</td><td>71.86</td><td>77.20</td><td>50.51</td></tr><tr><td colspan="8">Long autoregressive models</td></tr><tr><td>LongLive [51]</td><td>1.3B</td><td>83.72</td><td>85.42</td><td>76.95</td><td>82.62</td><td>84.53</td><td>74.97</td></tr><tr><td>Self Forcing++ [9]</td><td>1.3B</td><td>83.11</td><td>83.79</td><td>80.37</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Rolling Forcing [31]</td><td>1.3B</td><td>81.22</td><td>84.08</td><td>69.78</td><td>79.31</td><td>81.87</td><td>67.69</td></tr><tr><td>Infinity-RoPE [53]</td><td>1.3B</td><td>81.79</td><td>83.27</td><td>75.87</td><td>79.99</td><td>80.81</td><td>74.30</td></tr><tr><td>Context Forcing [7]</td><td>1.3B</td><td>83.44</td><td>84.98</td><td>77.29</td><td>82.45</td><td>83.55</td><td>76.10</td></tr><tr><td>Ours</td><td>1.3B</td><td>84.36</td><td>86.14</td><td>77.25</td><td>82.88</td><td>84.39</td><td>76.86</td></tr></table>

Physical Commonsense (PC) from the VideoPhy benchmark [1, 2], which measures adherence to real-world physical laws. The second is the Physical Alignment and Consistency Evaluation (PACE), an LLM-as-a-Judge metric powered by Gemini. PACE scores videos from 0 to 100, penalizing physical hallucinations based on prompt compliance and four criteria: gravity and mass representation, collision dynamics, motion continuity (avoiding sudden teleportation or unnatural warping), and long-term temporal coherence.

## 4.2 Comparison Results

Quantitative Evaluation on VBench. We comprehensively evaluate our method against state-of-the-art bidirectional, standard autoregressive, and longvideo specific autoregressive models on the VBench benchmark. As reported in Table 1, we evaluate both short-horizon (5s) and long-horizon (60s) video generation to demonstrate our model’s robustness against generative drift.

For 5-second generation, our model achieves the highest Total score and Quality score, outperforming strong baselines such as LongLive and Self-Forcing. The superiority of our approach becomes overwhelmingly evident in the extremely long-horizon (60s) setting. Standard autoregressive models sufer from catastrophic error accumulation over extended contexts; for instance, the Total score of Self-Forcing plummets from 84.31 (5s) to 71.86 (60s), and SkyReels-V2 drops from 82.67 to 70.47. In stark contrast, our method efectively bounds this generative drift, maintaining a remarkable Total score and achieving the highest Semantic score at 60 seconds. This minimal performance degradation over time confirms that our cycle-consistency framework successfully preserves the structural integrity and visual fidelity of the generated sequence over infinite horizons.

Physical Consistency Evaluation. To further validate whether our model accurately captures the underlying physical rules of the visual world, we evalu-

Wide-angle ballet dancer on mirror salt flat, sky to purple sunset

8K slow-motion white Andalusian stallion, cool morning light, Alpine meadow to lake

![](images/fbedd4b1e1045e76f290216b953fe6a5407b39759149008674123ee3ff9e4c10.jpg)  
Fig. 3: Qualitative comparison of long video generation. We compare our proposed method against several baseline models. The figure displays sampled frames at 0, 30, and 60 seconds for two distinct scenes. While the baseline methods struggle with subject loss, severe structural distortion, or identity shifts over the extended timeframe, our method successfully preserves temporal consistency, visual quality, and subject identity throughout the entire 60-second duration.

ate it on Physical Commonsense (PC) and Physical Alignment and Consistency Evaluation (PACE) metrics. As shown in Table 2, our method significantly outperforms all baseline approaches, achieving the highest average score of 75.66. By enforcing temporal reversibility, our model inherently internalizes causal constraints, endowing it with a superior understanding of complex physical dynamics compared to standard autoregressive predictors.

Qualitative Comparison. Figure 3 visualizes the 60-second generation quality of our model compared to the baselines. As the autoregressive steps accumulate, the baseline methods exhibit severe structural distortion, loss of the main subject, and prominent identity shifts. Our Cycle-World framework, however, acts as a strict temporal regularizer. It consistently preserves the subject’s identity, maintains sharp background details, and ensures temporal continuity from the first frame to the very last, translating the quantitative resilience observed in Table 1 into striking visual stability.

Table 2: Quantitative evaluation of physical consistency. Our proposed method outperforms all baseline approaches across both metrics, demonstrating a comprehensive understanding of complex physical dynamics.  
Table 3: Ablation study of the proposed components on the VBench benchmark. We evaluate the individual and synergistic efects of the trainingtime cycle loss $( \mathcal { L } _ { c y c l e } )$ and the inference-time cycle guidance on 5s video generation. The best results are highlighted in bold.

<table><tr><td>Method</td><td>PC</td><td>PACE</td><td>Avg.</td></tr><tr><td>Self-Forcing</td><td>55.97</td><td>78.57</td><td>67.27</td></tr><tr><td>LongLive</td><td>61.94</td><td>78.03</td><td>69.99</td></tr><tr><td>RollingForcing</td><td>67.91</td><td>75.05</td><td>71.48</td></tr><tr><td>Infinity-Rope</td><td>60.45</td><td>78.58</td><td>69.52</td></tr><tr><td>Ours</td><td>69.40</td><td>81.91</td><td>75.66</td></tr></table>

<table><tr><td>Method</td><td>Tot.</td><td>Qual.</td><td>Sem.</td><td>PC</td><td>PACE</td></tr><tr><td>Baseline</td><td>83.72</td><td>85.42</td><td>76.95</td><td>61.94</td><td>78.03</td></tr><tr><td>+ CCL</td><td>83.89</td><td>85.72</td><td>76.55</td><td>68.70</td><td>79.5</td></tr><tr><td>+ CGI</td><td>84.51</td><td>86.37</td><td>77.05</td><td>65.67</td><td>78.80</td></tr><tr><td>Cycle-World</td><td>84.36</td><td>86.14</td><td>77.25</td><td>69.40</td><td>81.91</td></tr></table>

## 4.3 Ablation Studies

Efectiveness of Cycle-World Components. To evaluate the contributions of our proposed modules, we conduct an ablation study on visual quality and physical consistency. Table 3 reports the quantitative metrics on 5-second video generation, while Figure 4 illustrates the qualitative long-horizon stability over 30 seconds.

Quantitative Synergy in Short-Horizon Generation. As shown in Table 3, the Baseline achieves a Total VBench score of 83.72 while scoring 61.94 on PC. Adding the Cycle-Consistent Learning (+ CCL) provides a learned parametric prior. While it moderately improves general visual metrics, its primary benefit is in physical consistency, raising PC by nearly 7 points. This suggests that penalizing reverse-prediction errors embeds causal constraints into the generator. Conversely, applying the runtime corrector solely during inference (+ CGI) acts as an instance-level regularizer against visual degradation, achieving the highest VBench Quality and Total scores. However, relying purely on gradient-based inference optimization without a learned parametric physical prior yields suboptimal physical consistency.

The full Cycle-World model combines both mechanisms. While its general VBench scores are marginally lower than the CGI-only variant, it achieves the highest performance in physical consistency. This combined strategy ensures the generated videos maintain both visual quality and physical accuracy.

Qualitative Results in Long-Horizon Generation. The benefit of this combination is more evident when extending the generation to 30 seconds (Fig. 4). In long-horizon synthesis, the Baseline exhibits rapid trajectory drift and background degradation. The +CCL variant preserves the core structural identity but struggles to suppress high-frequency temporal artifacts over extended autoregressive steps. The +CGI variant maintains aesthetic coherence locally but eventually drifts from the physical manifold due to the lack of an intrinsic causal prior. The full Cycle-World model combines these strengths. By using CCL to keep initial forward proposals close to the reversible manifold, the runtime corrector (CGI) performs more accurate latent refinement. This approach eliminates spatial distortion and temporal identity shifts, producing visually stable and physically grounded long videos.

32yo barista trains 2 trainees in bright modern café, latte art, smooth actions, realistic fluid, no clipping  
![](images/d19db767ca1a604db6e8de6b6322c6d807858cca79a245282fae481b63f10cd5.jpg)  
Fig. 4: Qualitative ablation study of the proposed Cycle-World components. We compare the visual quality and temporal consistency of diferent model variants. While the training cycle loss $( \mathcal { L } _ { c y c l e } )$ establishes a robust parametric foundation for structural stability, and the runtime corrector acts as an active safeguard against temporal artifacts, their combination achieves a powerful dual-phase synergy. The full Cycle-World model efectively prevents trajectory drift and background degradation, yielding superior results in long video generation.

## 5 Conclusion

In this paper, we presented Cycle-World, a novel autoregressive video generation framework designed to tackle the pervasive issue of unconstrained generative drift in long-horizon synthesis. Based on the theoretical insight that forward prediction errors can be strictly bottlenecked by enforcing temporal reversibility, we proposed a unified strategy that maintains causal consistency across both the training and inference phases. During training, we introduce a reverse-prediction cycle loss alongside the Distribution Matching Distillation (DMD) objective. This explicitly embeds causal constraints into the generator’s parametric weights, establishing a robust foundation for structurally stable frame generation. During inference, we repurpose the frozen reverse model as a fully diferentiable runtime corrector. By leveraging gradient-based cycle guidance, this mechanism acts as an active, instance-level safeguard, dynamically pruning out-of-distribution trajectory drift and temporal artifacts before they compound. Extensive experiments on the VBench benchmark demonstrate that this dual-phase synergy efectively prevents the structural collapse and aesthetic degradation typically observed in prolonged autoregressive generation. Consequently, Cycle-World achieves state-of-the-art performance in producing highly consistent, smooth, and high-fidelity 60-second videos.

## Acknowledgement

This work was supported by National Natural Science Foundation of China (No. 62302297, 625B2115, 62272447, 62472285, 72192821, 62472285), the Fundamental Research Funds for the Central Universities (YG2023QNB17, YG2024QNA44). [Re Prof Tao] This project is supported by the National Research Foundation, Singapore, under its NRF Professorship Award No. NRF-P2024-001.

## References

1. Bansal, H., Lin, Z., Xie, T., Zong, Z., Yarom, M., Bitton, Y., Jiang, C., Sun, Y., Chang, K.W., Grover, A.: Videophy: Evaluating physical commonsense for video generation. arXiv preprint arXiv:2406.03520 (2024) 11, 23

2. Bansal, H., Peng, C., Bitton, Y., Goldenberg, R., Grover, A., Chang, K.W.: Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800 (2025) 11, 23

3. Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S.W., Fidler, S., Kreis, K.: Align your latents: High-resolution video synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 22563–22575 (2023) 4

4. Cai, Y., Zhang, H., Chen, X., Xing, J., Hu, Y., Zhou, Y., Zhang, K., Zhang, Z., Kim, S.Y., Wang, T., Zhang, Y., Yang, X., Lin, Z., Yuille, A.: Omnivcus: Feedforward subject-driven video customization with multimodal control conditions. In: Belgrave, D., Zhang, C., Lin, H., Pascanu, R., Koniusz, P., Ghassemi, M., Chen, N. (eds.) Advances in Neural Information Processing Systems. vol. 38, pp. 115404– 115423. Curran Associates, Inc. (2025), https://proceedings.neurips.cc/ paper \_ files / paper / 2025 / file / a79054a9da91d73ed3cb1a9e87d7cd2d - Paper - Conference.pdf 4

5. Chen, B., Martí Monsó, D., Du, Y., Simchowitz, M., Tedrake, R., Sitzmann, V.: Difusion forcing: Next-token prediction meets full-sequence difusion. Advances in Neural Information Processing Systems 37, 24081–24125 (2024) 2, 4, 22

6. Chen, G., Lin, D., Yang, J., Lin, C., Zhu, J., Fan, M., Zhang, H., Chen, S., Chen, Z., Ma, C., et al.: Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074 (2025) 4, 10, 11

7. Chen, S., Wei, C., Sun, S., Nie, P., Zhou, K., Zhang, G., Yang, M.H., Chen, W.: Context forcing: Consistent autoregressive video generation with long context. arXiv preprint arXiv:2602.06028 (2026) 10, 11

8. Chen, X., Wang, Y., Zhang, L., Zhuang, S., Ma, X., Yu, J., Wang, Y., Lin, D., Qiao, Y., Liu, Z.: Seine: Short-to-long video difusion model for generative transition and prediction. In: The Twelfth International Conference on Learning Representations (2023) 4

9. Cui, J., Wu, J., Li, M., Yang, T., Li, X., Wang, R., Bai, A., Ban, Y., Hsieh, C.J.: Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283 (2025) 10, 11

10. Deng, H., Pan, T., Diao, H., Luo, Z., Cui, Y., Lu, H., Shan, S., Qi, Y., Wang, X.: Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169 (2024) 4, 10, 11

11. Dobrosotskaya, I.Y., James, G.L.: Magi-1 interacts with β-catenin and is associated with cell–cell adhesion structures. Biochemical and biophysical research communications 270(3), 903–909 (2000) 4, 10, 11

12. Gao, Y., Guo, H., Hoang, T., Huang, W., Jiang, L., Kong, F., Li, H., Li, J., Li, L., Li, X., et al.: Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113 (2025) 1

13. HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D., Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari, N., Gordon, O., Panet, P., Weissbuch, S., Kulikov, V., Bitterman, Y., Melumian, Z., Bibi, O.: Ltx-video: Realtime video latent difusion. arXiv preprint arXiv:2501.00103 (2024) 10, 11

14. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020) 4

15. Hong, W., Ding, M., Zheng, W., Liu, X., Tang, J.: Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868 (2022) 4

16. Hou, S., Wang, C., Zhuang, W., Chen, Y., Wang, Y., Bao, H., Chai, J., Xu, W.: A causal convolutional neural network for multi-subject motion modeling and generation. Computational Visual Media 10(1), 45–59 (2024). https://doi.org/10. 1007/s41095-022-0307-3 2

17. Hu, T., Lu, M., Wang, Y., Zhang, J., Hao, J., Pan, Y., Yi, R., Ma, L., Tao, D.: Metaworld: Scaling multi-agent video world model from single-view video data (2026), https://arxiv.org/abs/2606.02753 2

18. Hu, T., Yu, Z., Zhang, G., Su, Z., Zhou, Z., Zhang, Y., Zhou, Y., Lu, Q., Yi, R.: Harmony: Harmonizing audio and video generation through cross-task synergy. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16085–16095 (June 2026) 4

19. Hu, T., Yu, Z., Zhou, Z., Liang, S., Zhou, Y., Lin, Q., Lu, Q.: Hunyuancustom: A multimodal-driven architecture for customized video generation (2025), https: //arxiv.org/abs/2505.04512 4

20. Hu, T., Yu, Z., Zhou, Z., Zhang, J., Zhou, Y., Lu, Q., Yi, R.: Polyvivid: Vivid multi-subject video generation with cross-modal interaction and enhancement. In: Belgrave, D., Zhang, C., Lin, H., Pascanu, R., Koniusz, P., Ghassemi, M., Chen, N. (eds.) Advances in Neural Information Processing Systems. vol. 38, pp. 49394–49420. Curran Associates, Inc. (2025), https://proceedings.neurips.cc/ paper \_ files / paper / 2025 / file / 4683beb6bab325650db13afd05d1a14a - Paper - Conference.pdf 4

21. Hu, T., Zhang, J., Huang, H., Yi, R., Su, Z., Weng, J., Xue, Z., Ma, L., Yang, M.H., Tao, D.: Evolution of video generative foundations (2026), https://arxiv. org/abs/2604.06339 4

22. Hu, T., Zhang, J., Su, Z., Yi, R.: Ultragen: High-resolution video generation with hierarchical attention (2025), https://arxiv.org/abs/2510.18775 4

23. Huang, X., Li, Z., He, G., Zhou, M., Shechtman, E.: Self forcing: Bridging the train-test gap in autoregressive video difusion. arXiv preprint arXiv:2506.08009 (2025) 2, 4, 6, 10, 11, 22

24. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., Liu, Z.: VBench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024) 10

25. HunyuanWorld, T.: Hy-world 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency. arXiv preprint (2025) 2

26. Jin, Y., Sun, Z., Li, N., Xu, K., Jiang, H., Zhuang, N., Huang, Q., Song, Y., Mu, Y., Lin, Z.: Pyramidal flow matching for eficient video generative modeling. arXiv preprint arXiv:2410.05954 (2024) 4, 10, 11

27. Karacan, L., Sarıgül, M.: Full-frame video stabilization via spatiotemporal transformers. Computational Visual Media 11(3), 655–667 (2025). https://doi.org/ 10.26599/CVM.2025.9450416 4

28. Kondratyuk, D., Yu, L., Gu, X., Lezama, J., Huang, J., Schindler, G., Hornung, R., Birodkar, V., Yan, J., Chiu, M.C., et al.: Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125 (2023) 4

29. Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al.: Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024) 2, 4

30. Liu, K., Zheng, Y., Wang, K., Wu, S., Zhang, R., Luo, J., Hatzinakos, D., Liu, Z., Fei, H., Chua, T.S.: Javisdit++: Unified modeling and optimization for joint audio-video generation. In: The Fourteenth International Conference on Learning Representations (2026) 4

31. Liu, K., Hu, W., Xu, J., Shan, Y., Lu, S.: Rolling forcing: Autoregressive long video difusion in real time. arXiv preprint arXiv:2509.25161 (2025) 10, 11

32. Liu, Y., Zhao, H., Chan, K.C.K., Wang, X., Loy, C.C., Qiao, Y., Dong, C.: Temporally consistent video colorization with deep feature propagation and self-regularization learning. Computational Visual Media 10(2), 375–395 (2024). https://doi.org/10.1007/s41095-023-0342-8 4

33. Low, C., Wang, W., Katyal, C.: Ovi: Twin backbone cross-modal fusion for audiovideo generation (2025), https://arxiv.org/abs/2510.01284 4

34. Mao, X., Li, Z., Li, C., Xu, X., Ying, K., He, T., Pang, J., Qiao, Y., Zhang, K.: Yume-1.5: A text-controlled interactive world generation model. arXiv preprint arXiv:2512.22096 (2025) 2

35. Nichol, A.Q., Dhariwal, P.: Improved denoising difusion probabilistic models. In: International conference on machine learning. pp. 8162–8171. PMLR (2021) 4

36. OpenAI: Sora. https://openai.com/sora (2024) 1

37. OpenAI: Sora 2. https://openai.com/index/sora-2/ (2025) 1

38. Peebles, W., Xie, S.: Scalable difusion models with transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4195–4205 (2023) 4

39. Qiu, H., Xia, M., Zhang, Y., He, Y., Wang, X., Shan, Y., Liu, Z.: Freenoise: Tuningfree longer video difusion via noise rescheduling. arXiv preprint arXiv:2310.15169 (2023) 4

40. Rasul, K., Seward, C., Schuster, I., Vollgraf, R.: Autoregressive denoising difusion models for multivariate probabilistic time series forecasting. In: International conference on machine learning. pp. 8857–8868. PMLR (2021) 4

41. Seed, B.: Seedance 2.0. https://seed.bytedance.com/en/seedance2\_0 (2026) 1

42. Skywork AI Matrix-Game Team: Matrix-game 3.0: Real-time and streaming interactive world model with long-horizon memory. Technical report (2026), https: //github.com/SkyworkAI/Matrix- Game/blob/main/Matrix- Game- 3/assets/ pdf/report.pdf 2

43. Song, J., Meng, C., Ermon, S.: Denoising difusion implicit models. arXiv preprint arXiv:2010.02502 (2020) 4

44. Technology, K.: Kling. https://kling.kuaishou.com/ (2025) 2

45. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., et al.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025) 2, 4, 10, 11, 22

46. Wang, D., Zuo, W., Li, A., Chen, L.H., Liao, X., Zhou, D., Yin, Z., Dai, X., Jiang, D., Yu, G.: Universe-1: Unified audio-video generation via stitching of experts. arXiv preprint arXiv:2509.06155 (2025) 4

47. Wang, F.Y., Chen, W., Song, G., Ye, H.J., Liu, Y., Li, H.: Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264 (2023) 4

48. Wang, H., Ma, C.Y., Liu, Y.C., Hou, J., Xu, T., Wang, J., Juefei-Xu, F., Luo, Y., Zhang, P., Hou, T., et al.: Lingen: Towards high-resolution minute-length textto-video generation with linear computational complexity. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 2578–2588 (2025) 4

49. Wang, W., Yang, Y.: Vidprom: A million-scale real prompt-gallery dataset for textto-video difusion models (2024), https://openreview.net/forum?id=pYNl76onJL 22

50. Williams, R.J., Zipser, D.: A learning algorithm for continually running fully recurrent neural networks. Neural computation 1(2), 270–280 (1989) 4

51. Yang, S., Huang, W., Chu, R., Xiao, Y., Zhao, Y., Wang, X., Li, M., Xie, E., Chen, Y., Lu, Y., et al.: Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622 (2025) 2, 4, 10, 11, 22

52. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al.: Cogvideox: Text-to-video difusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024) 4

53. Yesiltepe, H., Meral, T.H.S., Akan, A.K., Oktay, K., Yanardag, P.: Infinity-rope: Action-controllable infinite video generation emerges from autoregressive selfrollout. arXiv preprint arXiv:2511.20649 (2025) 10, 11

54. Yin, S., Wu, C., Yang, H., Wang, J., Wang, X., Ni, M., Yang, Z., Li, L., Liu, S., Yang, F., et al.: Nuwa-xl: Difusion over difusion for extremely long video generation. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 1309–1320 (2023) 4

55. Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., Freeman, B.: Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems 37, 47455–47487 (2024) 4

56. Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T., Park, T.: One-step difusion with distribution matching distillation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6613– 6623 (2024) 4

57. Yin, T., Zhang, Q., Zhang, R., Freeman, W.T., Durand, F., Shechtman, E., Huang, X.: From slow bidirectional to fast autoregressive video difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22963–22974 (2025) 2, 4, 10, 11, 22

58. Zhang, G., Zhou, Z., Hu, T., Peng, Z., Zhang, Y., Chen, Y., Zhou, Y., Lu, Q., Wang, L.: Uniavgen: Unified audio and video generation with asymmetric crossmodal interactions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1950–1960 (2026) 4

59. Zheng, D., Huang, Z., Liu, H., Zou, K., He, Y., Zhang, F., Zhang, Y., He, J., Zheng, W.S., Qiao, Y., Liu, Z.: VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755 (2025) 10

60. Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., You, Y.: Open-sora: Democratizing eficient video production for all. arXiv preprint arXiv:2412.20404 (2024) 4

61. Zhu, H., Zhao, M., He, G., Su, H., Li, C., Zhu, J.: Causal forcing: Autoregressive difusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214 (2026) 2

62. Zhu, J.Y., Park, T., Isola, P., Efros, A.A.: Unpaired image-to-image translation using cycle-consistent adversarial networks. In: Proceedings of the IEEE international conference on computer vision. pp. 2223–2232 (2017) 29

## A Overview

This supplementary material provides further theoretical derivations, algorithmic details, rigorous experimental setups, and extensive qualitative and quantitative evaluations to thoroughly support the claims made in the main manuscript. The document is logically organized as follows:

– Section B presents the detailed mathematical proofs for the theoretical bounds established in the main text (Theorem 1, Corollary 1, and Proposition 1).

Section C provides the complete pseudocode and a step-by-step breakdown of our proposed zero-shot Cycle-Guided Inference (CGI) strategy.

– Section D outlines the comprehensive implementation details, including data preparation, model construction, and hyperparameter configurations.

– Section E details the exact definitions and prompt templates used for our Physical Consistency evaluation metrics (PC and PACE).

– Section F provides extended quantitative comparisons and ablation studies on the challenging 60-second video generation tasks.

– Section G.1 & G.2 showcase additional qualitative comparisons, highlighting Cycle-World’s robustness in interactive generation and single-prompt extended synthesis.

Section I presents a thorough sensitivity analysis of key training and inference hyperparameters.

– Section J further validates the plug-and-play extensibility of CGI on standard forward-only baselines.

Section K clarifies the fundamental distinctions between our temporal cycleconsistency framework and traditional spatial cycle-consistency models.

– Section L concludes with a rigorous discussion on the theoretical boundaries and broader applicability of the Cycle-World framework to other sequential modalities.

## B Detailed Proofs of Theoretical Bounds

This section provides the complete mathematical derivations for the theoretical claims established in the main manuscript. We rely on the definitions of generative drift $e _ { n } = \| \hat { z } _ { n } - z _ { n } \|$ and the single-step cycle-consistency distance $d _ { c y c l e } ^ { ( n ) } = \| \hat { z } _ { n - 1 } - R _ { \phi } ( \hat { z } _ { n } ) \|$

## B.1 Proof of Theorem 1 (Cycle-Bounded Drift)

Proof. Applying the reverse-Lipschitz condition (Assumption 2) to the generated state $\hat { z } _ { n }$ and the ground-truth state $z _ { n }$ , we have:

$$
e _ {n} = \left\| \hat {z} _ {n} - z _ {n} \right\| \leq C \left\| R _ {\phi} (\hat {z} _ {n}) - R _ {\phi} (z _ {n}) \right\|.
$$

By applying the triangle inequality and adding/subtracting both the generated history $\hat { z } _ { n - 1 }$ and the ground-truth history $z _ { n - 1 }$ inside the norm on the right side, we obtain:

$$
e _ {n} \leq C \left(\| R _ {\phi} (\hat {z} _ {n}) - \hat {z} _ {n - 1} \| + \| \hat {z} _ {n - 1} - z _ {n - 1} \| + \| z _ {n - 1} - R _ {\phi} (z _ {n}) \|\right).
$$

Substituting the definitions of the cycle-consistency distance $d _ { c y c l e } ^ { ( n ) }$ , the previous step’s generative drift $e _ { n - 1 }$ , and the inherent reverse approximation error $\epsilon _ { R }$ (Assumption 1) yields the step-wise recurrence:

$$
e _ {n} \leq C \left(d _ {\text { cycle }} ^ {(n)} + e _ {n - 1} + \epsilon_ {R}\right).
$$

## B.2 Proof of Corollary 1 (Long-Horizon Error Bound)

Proof. Starting from Theorem 1, we recursively substitute the recurrence for $e _ { n - 1 } { : }$

$$
\begin{array}{r l} & e _ {n} \leq C e _ {n - 1} + C \left(d _ {c y c l e} ^ {(n)} + \epsilon_ {R}\right) \\ & \quad \leq C \left[ C e _ {n - 2} + C \left(d _ {c y c l e} ^ {(n - 1)} + \epsilon_ {R}\right) \right] + C \left(d _ {c y c l e} ^ {(n)} + \epsilon_ {R}\right) \\ & \quad = C ^ {2} e _ {n - 2} + C ^ {2} \left(d _ {c y c l e} ^ {(n - 1)} + \epsilon_ {R}\right) + C \left(d _ {c y c l e} ^ {(n)} + \epsilon_ {R}\right). \end{array}
$$

Continuing this expansion down to the initial condition $e _ { 0 }$ yields the summation: $\begin{array} { r } { e _ { n } \leq C ^ { n } e _ { 0 } + \sum _ { i = 1 } ^ { n } C ^ { n - i + 1 } \left( d _ { c y c l e } ^ { ( i ) } + \epsilon _ { R } \right) } \end{array}$ . Factoring out the maximum local loss $\delta _ { c y c l e }$ resolves it into the closed-form geometric expression in $\mathrm { E q . 2 }$ of the main text.

## B.3 Proof of Proposition 1 (Theoretical Advantage over Unconstrained Baselines)

Proof. Applying Corollary 1 to both generation pipelines, their respective drift limits are:

$$
\begin{array}{r} E _ {n} ^ {u n c} = C ^ {n} e _ {0} + (\delta_ {u n c} + \epsilon_ {R}) C \frac {C ^ {n} - 1}{C - 1}, \\ E _ {n} ^ {o u r s} = C ^ {n} e _ {0} + (\delta_ {c y c l e} + \epsilon_ {R}) C \frac {C ^ {n} - 1}{C - 1}. \end{array}
$$

Subtracting $E _ { n } ^ { o u r s }$ from $E _ { n } ^ { u n c }$ immediately yields the reduction gap:

$$
\varDelta E _ {n} = (\delta_ {u n c} - \delta_ {c y c l e}) C \frac {C ^ {n} - 1}{C - 1}.
$$

Since $\delta _ { u n c } > \delta _ { c y c l e }$ and $C \neq 1$ , it is evident that $\varDelta E _ { n } > 0$

## C Algorithm for Cycle-Guided Inference

As detailed in the main manuscript, Cycle-Guided Inference (CGI) serves as a zero-shot, training-free latent optimization strategy to enforce temporal reversibility in frozen foundational models. Algorithm 1 provides the complete step-by-step pseudocode for this procedure.

Specifically, during a designated optimization window $[ T _ { \mathrm { s t a r t } } , T _ { \mathrm { e n d } } ]$ within the autoregressive difusion loop, CGI computes the Euclidean cycle discrepancy D using the frozen reverse corrector $R _ { \phi }$ . Because the reverse reconstruction is fully diferentiable within the latent space, the gradient of D is iteratively backpropagated to actively refine the intermediate noisy latent $z _ { n , t }$ This runtime rectification acts as a structural bottleneck, pruning non-physical artifacts before the final latent is permanently committed to the historical context bufer.

## D Implementation Details

In this section, we provide comprehensive implementation details of our proposed framework, covering dataset preparation, model configuration, training strategies, and hyperparameter settings.

Data Preparation. For the Ordinary Diferential Equation (ODE) initialization phase in the difusion forcing [5] process, the ground-truth ODE latents are acquired using the Wan2.1 14B model [45], as provided by CausVid [57]. For the subsequent DMD training phase, we utilize a filtered and augmented version of the VidProM [23, 49] prompt dataset.

Reverse Model Construction. The reverse prediction model is initialized from the pre-trained Wan2.1 1.3B model. During the ODE initialization phase, we temporally flip the latent trajectories of the forward videos to initialize the reverse model. Subsequently, the model is trained utilizing the self-forcing paradigm. A key eficiency of our implementation is that, during training, we simply reverse the streaming latent outputs along the temporal dimension. This elegant operation allows us to directly reuse the bidirectional Wan model—which originally serves the forward prediction model—as both the teacher model and the critic, thereby eliminating the prohibitive computational cost of training a dedicated reverse teacher model.

Forward Model Tuning. The forward generator is built upon the pre-trained LongLive [51] framework. We adapt streaming long tuning [51] on 60-second video sequences, where each sequence contains a single prompt switch to encourage dynamic context transition. The forward model is fine-tuned using Low-Rank Adaptation (LoRA) on the LongLive weights. The training process spans a total of 3,000 iterations with a batch size of 4.

Hyperparameter Configurations. During the joint training phase, the weight for the cycle-consistency loss (λ) is empirically set to 0.1. During the inference phase, the cycle guidance is applied across all denoising timesteps. For the runtime corrector, we set the number of optimization iterations per timestep to

K = 1, with a optimization step size of $\eta = 1 0$ . All other unmentioned architectural and optimization hyperparameters strictly adhere to the default configurations of LongLive.

## E Physical Consistency Evaluation Metrics

To rigorously assess the physical realism of the generated videos, we utilize two complementary metrics: Physical Commonsense (PC) and Physical Alignment and Consistency Evaluation (PACE). We randomly sample 150 prompts from the VideoPhy [1, 2] prompt set to synthesize the test videos for our evaluation protocol. For short-horizon evaluation on 5-second videos, the metrics are computed directly on the entire clip. For long-horizon evaluation on 60-second videos, we divide each generated sequence into non-overlapping 5-second chunks and randomly sample three distinct chunks per video for assessment.

Physical Commonsense (PC). The PC metric employs the automatic scoring model introduced in the VideoPhy benchmark, which assigns a discrete physical consistency score ranging from 1 to 5 to each video clip. To establish a stringent baseline for physical realism, we define the final PC score as the percentage of evaluated clips that achieve a score of 4 or higher.

Physical Alignment and Consistency Evaluation (PACE). To obtain a granular, human-aligned assessment of complex physical dynamics, we introduce PACE, a Multimodal-LLM-as-a-Judge metric. We utilize the Gemini model to evaluate the videos based on a dedicated prompt. The model is instructed to act as an expert assessor and output a comprehensive score from 0 to 100. The evaluation explicitly penalizes physical hallucinations by scrutinizing basic prompt compliance alongside four core physical dimensions: the realistic representation of gravity and mass, the natural dynamics of collisions, motion continuity (avoiding sudden teleportation or unnatural warping), and long-term temporal coherence. To facilitate automated parsing, the model is constrained to return the evaluation result strictly in JSON format. The exact prompt template provided to the model is as follows:

Act as an expert in video quality assessment and physics. Evaluate the provided video based on these criteria: (1) Prompt Compliance: Does the video content strictly follow the intended action and description? (2) Physical Consistency: Does the video adhere to real-world physical laws? Look for gravity and weight (do objects fall or move with realistic mass?), collisions (do interactions between objects look natural?), motion continuity (is there any sudden teleportation or unnatural warping?), and temporal coherence (does the scene remain consistent over time?). Provide a score from 0 to 100 and a concise justification for your rating. Return the result strictly in JSON format with ’score’ and ’reason’ keys.

## F Extended Evaluation on Physical Consistency

To comprehensively validate the efectiveness of our proposed framework in extremely long sequences, we first present the extended quantitative comparisons and ablation studies focusing on the 60-second video generation tasks.

Table S1: Quantitative comparison of physical consistency on 60-second video generation.

<table><tr><td>Method</td><td>PC</td><td>PACE</td><td>Average</td></tr><tr><td>Self Forcing</td><td>65.67</td><td>49.96</td><td>57.82</td></tr><tr><td>LongLive</td><td>59.45</td><td>75.71</td><td>67.58</td></tr><tr><td>Rolling Forcing</td><td>61.69</td><td>74.59</td><td>68.14</td></tr><tr><td>Infinity-RoPE</td><td>63.18</td><td>72.77</td><td>67.98</td></tr><tr><td>Ours</td><td>70.90</td><td>75.42</td><td>73.16</td></tr></table>

Superiority in Long-Horizon Physical Consistency. To evaluate the robustness of our method over extended sequences, we compare its physical consistency against state-of-the-art baselines on 60-second generation, as detailed in Table S1. The results demonstrate that standard autoregressive approaches like Self Forcing struggle with complex physical dynamics over time, yielding an exceptionally low PACE score of 49.96 despite maintaining a moderate physical commonsense score. While long-context models such as LongLive and Rolling Forcing improve PACE, their overall physical consistency remains bounded around an average of 68. Our method efectively harmonizes prompt adherence with real-world physical laws, achieving the highest average score of 73.16. This superiority confirms that enforcing temporal reversibility acts as a critical mechanism to prevent physical hallucinations in extremely long videos.

Table S2: Ablation study of physical consistency on 60-second video generation.

<table><tr><td>Method</td><td>PC</td><td>PACE</td><td>Average</td></tr><tr><td>Baseline</td><td>59.45</td><td>75.71</td><td>67.58</td></tr><tr><td>+ CCL</td><td>70.40</td><td>74.94</td><td>72.67</td></tr><tr><td>+ CGI</td><td>62.77</td><td>76.67</td><td>69.72</td></tr><tr><td>Ours (Cycle-World)</td><td>70.90</td><td>75.42</td><td>73.16</td></tr></table>

Synergistic Efects of Cycle Constraints. The 60-second ablation study reveals the distinct and complementary roles of the proposed modules. The baseline model exhibits a noticeable imbalance over extended horizons, maintaining a relatively high PACE score of 75.71 but struggling with foundational physical commonsense, as evidenced by a low PC score of 59.45. Incorporating the training-time cycle-consistent learning immediately provides a robust parametric foundation, which drastically raises the PC score to 70.40, albeit with a slight reduction in PACE to 74.94. This indicates that the parametric prior efectively enforces rigid structural laws but may slightly constrain unbridled dynamic variance. Conversely, applying the cycle guidance inference alone primarily preserves the high PACE metric at 76.67 but only marginally improves the PC score to 62.77, demonstrating that runtime optimization without a learned physical prior is insuficient to fundamentally correct structural physical violations. The full model integrates both mechanisms to achieve a powerful dual-phase synergy, maximizing the PC score to 70.90 and recovering the PACE score to 75.42, which culminates in the highest overall average performance of 73.16.

![](images/e7b2669ea5cbf5bdc255eaa3b1ec27755d92e639dee65fdeb07f632b30b91cbf.jpg)  
Fig. S1: Qualitative comparison of interactive long video generation over a 60-second horizon. The textual prompt is dynamically updated every 10 seconds. Cycle-World seamlessly interpolates new instructions while maintaining strict identity and background consistency, whereas baselines sufer from severe physical hallucinations and abrupt transitions.

## G More Qualitative Comparisons

## G.1 Interactive Long Video Generation

Beyond continuous prediction, a true video world model must support interactive, open-ended generation, allowing users to dynamically alter the future trajectory based on a shared historical context. However, dynamically changing text conditions during autoregressive generation often exacerbates structural collapse in standard models, as the sudden semantic shift disrupts the already fragile temporal continuity.

Leveraging the robust physical grounding provided by our Cycle-World framework, we evaluate its performance in interactive generation scenarios over a 60-second horizon, where the textual prompt is updated every 10 seconds. In this highly challenging setting, standard baselines such as Infinity-RoPE and LongLive struggle significantly. When confronted with semantic shifts or largescale subject movements, these models frequently exhibit abrupt, unnatural scene transitions. Furthermore, they sufer from severe identity degradation and physical hallucinations, such as the sudden appearance or vanishing of background objects and characters.

As illustrated in Fig. S1, Cycle-World exhibits exceptional adaptability and structural resilience. The runtime cycle critic explicitly penalizes physically impossible transitions, ensuring that the background remains stable and the subject’s identity is strictly preserved despite the semantic branch. Unlike forwardonly baselines that hallucinate entirely new entities when the prompt shifts, our model seamlessly interpolates the fluid dynamics required by the new instruction while maintaining strict temporal reversibility. This demonstrates that enforcing cycle consistency establishes a highly robust latent manifold, unlocking stable and interactive control for open-ended world simulation.

## G.2 Single-Prompt Long Video Generation

To further demonstrate the robustness of our framework, we compare Cycle-World against state-of-the-art long video autoregressive models, namely LongLive, Infinity-RoPE, and Rolling Forcing, under a 60-second single-prompt generation setting. Generating extended sequences without intermediate text guidance exposes the critical vulnerabilities of existing methods over time.

As shown in Fig. S2, LongLive sufers from physical hallucinations, including the spontaneous manifestation of non-existent subjects and unnatural object interpenetration (structural clipping). Infinity-RoPE, while attempting to maintain structural coherence, exhibits significantly degraded motion dynamics and fails to preserve subject consistency over extended periods. Similarly, Rolling Forcing exhibits a limited dynamic range and tends to hallucinate abrupt, outof-context entities.

In stark contrast, our Cycle-World framework successfully achieves an optimal balance between visual consistency and rich motion dynamics. By rigorously enforcing temporal reversibility, our model prevents unconstrained generative drift. It ensures that the primary subject, background integrity, and natural physical interactions are preserved throughout the entire 60-second duration, without sacrificing the amplitude and realism of the generated motion.

## H More Qualitative Results

We provide additional qualitative results in Fig. S3 to showcase the versatility and high visual fidelity of Cycle-World across diverse scenes and complex physical motions. These examples further corroborate the eficacy of our cycle-consistency framework. By inherently bounding the autoregressive prediction errors, Cycle-World is capable of producing highly stable, structurally sound, and aesthetically pleasing long-horizon video simulations across a wide variety of open-domain prompts.

Red fox playfully chases autumn leaves in sunlit golden forest, energetic and lively, medium tracking shot.  
![](images/4fff89fc47d0637a4461a62121e93b2c342308b6535329b27c4bf838a4ff2c79.jpg)  
Fig. S2: Qualitative comparison of 60-second single-prompt video generation. Compared to LongLive, Infinity-RoPE, and Rolling Forcing, our method efectively prevents object interpenetration and hallucinatory artifacts, achieving an optimal balance between long-term visual consistency and rich motion dynamics.

## I Hyperparameter Analysis

We analyze the key hyperparameters of our framework to understand their impact on the trade-of between visual quality and physical consistency. Specifically, we examine the training cycle loss weight alongside the temporal distribution and step size of the inference cycle guidance.

Table S3: Impact of the cycle loss weight during training (evaluated on 5-second generation).

<table><tr><td>Weight (λ)</td><td>Total</td><td>Quality</td><td>Semantic</td><td>PC</td><td>PACE</td></tr><tr><td>0.2</td><td>82.89</td><td>84.80</td><td>75.27</td><td>68.66</td><td>75.80</td></tr><tr><td>0.1</td><td>84.05</td><td>85.83</td><td>76.93</td><td>64.18</td><td>79.04</td></tr><tr><td>0.05</td><td>84.47</td><td>86.56</td><td>76.13</td><td>56.72</td><td>76.53</td></tr></table>

Table S3 shows the efect of the cycle loss weight during training on 5-second video generation. The results reveal a trade-of between aesthetic quality and structural adherence. A lower weight of 0.05 improves visual and semantic scores but causes a sharp decline in physical commonsense to 56.72. This indicates that the model struggles to stay on the physical manifold. A higher weight of 0.2 enforces strict cycle consistency and raises the PC score to 68.66. However, this regularization penalizes visual fidelity, reducing the total VBench score to 82.89. A weight of 0.1 balances these aspects, yielding the highest PACE score of 79.04 while preserving competitive video quality.

Table S4: Efect of the temporal distribution of optimization steps during inference.

<table><tr><td>Optimization Strategy</td><td>Total</td><td>Quality</td><td>Semantic</td><td>PC</td><td>PACE</td></tr><tr><td>1 step across all 4 timesteps</td><td>84.36</td><td>86.14</td><td>77.25</td><td>69.40</td><td>81.91</td></tr><tr><td>2 steps on the first 2 timesteps</td><td>84.22</td><td>85.89</td><td>77.52</td><td>69.40</td><td>81.77</td></tr><tr><td>4 steps on the 1st timestep</td><td>84.07</td><td>85.90</td><td>76.78</td><td>68.66</td><td>76.79</td></tr></table>

Table S4 details the efect of distributing a fixed budget of four optimization steps across the denoising timesteps during inference. Concentrating all four steps at the initial timestep results in the weakest performance, particularly lowering the PACE score to 76.79. Distributing the steps evenly by applying one optimization per timestep yields the best results across both visual and physical metrics. Continuous gradient guidance along the generation trajectory is more efective for maintaining temporal coherence than isolated early intervention.

Table S5: Sensitivity analysis of the inference gradient guidance step size.

<table><tr><td>Step</td><td>Size (η)</td><td>Total</td><td>Quality</td><td>Semantic</td><td>PC</td><td>PACE</td></tr><tr><td>5</td><td></td><td>84.20</td><td>85.86</td><td>77.58</td><td>68.66</td><td>76.79</td></tr><tr><td>10</td><td></td><td>84.36</td><td>86.14</td><td>77.25</td><td>69.40</td><td>81.91</td></tr><tr><td>15</td><td></td><td>83.92</td><td>85.62</td><td>77.12</td><td>70.15</td><td>82.27</td></tr></table>

The gradient guidance step size controls the strength of the runtime corrector. As shown in Table S5, increasing the step size from 5 to 15 steadily improves the model’s adherence to physical constraints, with PC and PACE reaching 70.15 and 82.27. An aggressive step size of 15 compromises general generation quality, leading to lower total and semantic VBench scores. A step size of 10 provides an optimal configuration that maintains physical alignment without sacrificing the visual and semantic integrity of the sequence.

## J Plug-and-Play Extensibility of Cycle-Guided Inference

A practical benefit of Cycle-Guided Inference (CGI) is its extensibility. Since the runtime corrector uses gradient-based latent refinement without altering the architecture of the forward generator, it integrates directly into existing autoregressive video models. This inference strategy applies to any forward-generation framework that operates within the same VAE latent space as the pre-trained reverse model.

To evaluate this adaptability, we apply the frozen reverse-prediction corrector to two standard forward-only baselines, CausVid and Self-Forcing. As Figure S4 shows, adding CGI consistently improves both models in the 60-second longhorizon setting.

Qualitative results highlight several improvements from this integration. CGI enhances the global spatiotemporal consistency of the generated sequences, preserving subject identities and background details over extended frames. It also corrects physical anomalies common in unconstrained autoregressive models. For example, in a dynamic chasing scene, unmodified baselines often generate a background that incorrectly moves forward relative to the running subjects. By enforcing temporal reversibility, the CGI module corrects this motion error, ensuring the background recedes naturally.

Furthermore, while standard forward-only models sufer from generative drift over long contexts, applying CGI delays structural collapse. By removing accumulated artifacts at each autoregressive step, this strategy extends the efective generation length of the underlying baselines. This results in longer, structurally stable video sequences without requiring additional model retraining.

## K Distinctions from Traditional Cycle Consistency

The concept of cycle consistency has been widely explored in computer vision, most notably in unpaired image-to-image translation frameworks such as CycleGAN [62]. These traditional methods introduce a cycle-consistency objective to learn bijective mappings between two distinct spatial or stylistic domains. By ensuring that an image translated to a target domain can be accurately reconstructed back to its original domain, these models efectively bypass the requirement for strictly paired training data. The constraint primarily operates spatially, focusing on preserving texture, geometry, and structural content across diferent artistic or sensor modalities.

Our Cycle-World framework fundamentally diverges from these traditional applications in both its core objective and operational domain. Rather than mapping between diferent visual domains, our method enforces cycle consistency strictly along the temporal axis within a single continuous domain. The primary goal is not stylistic translation, but rather bounding the compounding generative drift inherent in long-horizon autoregressive synthesis. We conceptualize cycle consistency as a fundamental physical constraint, grounded in the observation that valid natural dynamics and causal events must be temporally reversible.

Functionally, this distinction translates to divergent architectural implementations. While traditional frameworks employ two symmetric cross-domain spatial generators, Cycle-World pairs a forward autoregressive generator with a temporal reverse-prediction model. This configuration constructs a step-wise cycle across sequential states. By actively minimizing the discrepancy between a historical state and its reverse-predicted reconstruction from a future state, our approach serves as an intrinsic physical regularizer. This temporal cycle explicitly penalizes structural hallucinations and non-physical motion artifacts, thereby maintaining long-term causal coherence rather than mere spatial fidelity.

## L Applicability to General Temporal Modalities

While this study grounds the Cycle-World framework in long-horizon video synthesis, its theoretical formulation is intrinsically modality-agnostic. The core mechanism of bounding autoregressive drift via cycle consistency relies entirely on the topological properties of the latent manifold rather than the specific visual nature of the data. Consequently, this framework can be extended to other sequential domains, provided the underlying data distribution strictly adheres to the principle of local temporal reversibility.

Audio generation represents a highly compatible domain for this extension. Acoustic signals, whether speech, music, or environmental sounds, are continuous physical waveforms governed by mechanical laws and temporal causality. Current autoregressive audio models frequently experience compounding errors that manifest as rhythmic degradation, phase shifts, or the gradual loss of speaker identity over extended contexts. Because acoustic dynamics preserve short-term historical information within their local temporal window, training a reverse acoustic predictor is mathematically well-posed. Applying cycle-guided inference to audio latent spaces could actively correct these deviations, ensuring the generated sequence remains anchored to a natural acoustic manifold without requiring architectural changes to the base audio model.

Beyond perceptual modalities, the cycle-consistency paradigm holds significant potential for general temporal forecasting tasks in physically grounded environments. Predictive models in autonomous driving, robotic kinematics, and molecular dynamics simulate spatial-temporal states that are primarily governed by classical mechanics. In these Newtonian systems, state transitions are inherently time-reversible. A valid future state must contain suficient deterministic information to deduce its immediate predecessor. By utilizing cycle guidance as a runtime physical regularizer, predictive models would be compelled to respect these mechanical constraints, thereby preventing the unconstrained divergence of simulated trajectories over long time horizons.

However, a rigorous theoretical boundary limits the universal application of this framework to all autoregressive tasks. The eficacy of Cycle-World is strictly contingent upon the reverse-predictability assumption. This mechanism cannot be generalized to highly entropic, discrete, or lossy sequential processes where the arrow of time introduces severe information collapse. In domains such as abstract text generation or financial market forecasting, state transitions often represent many-to-one mappings where multiple distinct past contexts can converge into an identical current state. Under such macroscopic irreversible conditions, the backward mapping becomes fundamentally ill-posed, and the reverse approximation error would violate the theoretical bounds required for our theorem to hold. Therefore, the applicability of Cycle-World is rigorously confined to continuous, physically grounded, or information-preserving latent manifolds where temporal inversion remains locally deterministic.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Cycle-Guided Inference with Runtime Corrector
Require: Forward generator $G_{\theta}$, Reverse corrector $R_{\phi}$, Diffusion transition function $\Psi$, Initial context $\hat{z}_{0}$, Sequence length $N$, Timesteps $\{t_{T},\ldots,t_{1}\}$, Optimization window $[T_{\text{start}},T_{\text{end}}]$, Context noise $t_{\text{ctx}}$, Refinement iterations $K$, Step size $\eta$.
Ensure: Generated video latents $\hat{Z}=\{\hat{z}_{1},\ldots,\hat{z}_{N}\}$
1: Initialize history buffer $\mathcal{H}=\{\hat{z}_{0}\}$
2: Initialize output sequence $\hat{Z}=\emptyset$
3: for $n=1$ to $N$ do
4: Sample initial noise $z_{n,t_{T}}\sim\mathcal{N}(0,\mathbf{I})$
5: for $j=T$ down to 1 do
6: Let $t=t_{j}$
7: if $n&gt;1$ and $t\in[T_{\text{start}},T_{\text{end}}]$ then ▷ Cycle Guidance Optimization Window
8: $z_{n,t}^{(0)}\leftarrow z_{n,t}$
9: Sample target noise $\epsilon\sim\mathcal{N}(0,\mathbf{I})$
10: for $k=0$ to $K-1$ do
11: $\hat{z}_{n|t}\leftarrow G_{\theta}(z_{n,t}^{(k)},t,\mathcal{H})$ ▷ Predict clean latent
12: $R_{\phi}(\mathcal{F}(\hat{z}_{n|t}),t_{\text{ctx}},\mathcal{H}_{\text{rev}})$ ▷ Construct reverse context
13: $\tilde{z}_{n-1}\leftarrow\mathcal{F}(R_{\phi}(\epsilon,t_{T},\mathcal{H}_{\text{rev}}))$ ▷ Predict predecessor from noise
14: $\mathcal{D}\leftarrow\|\hat{z}_{n-1}-\tilde{z}_{n-1}\|_{2}^{2}$ ▷ Compute cycle discrepancy
15: $z_{n,t}^{(k+1)}\leftarrow z_{n,t}^{(k)}-\eta\nabla_{z_{n,t}^{(k)}}\mathcal{D}$ ▷ Gradient descent update
16: end for
17: $z_{n,t}\leftarrow\text{Detach}(z_{n,t}^{(K)})$ ▷ Final refined state
18: end if
19: $\hat{z}_{n|t}\leftarrow G_{\theta}(z_{n,t},t,\mathcal{H})$ ▷ Forward prediction for transition
20: if $j=1$ then ▷ Final Denoising Step
21: $\hat{z}_{n}\leftarrow\hat{z}_{n|t}$ ▷ Final generated clean latent
22: $\mathcal{H}\leftarrow\mathcal{H}\cup\{\hat{z}_{n}\}$ ▷ Update KV cache
23: $\hat{Z}\leftarrow\hat{Z}\cup\{\hat{z}_{n}\}$ ▷ Update output sequence
24: else ▷ Intermediate Denoising Step
25: Sample $\epsilon_{\text{trans}}\sim\mathcal{N}(0,\mathbf{I})$
26: $z_{n,t_{j-1}}\leftarrow\Psi(\hat{z}_{n|t},\epsilon_{\text{trans}},t_{j-1})$ ▷ Standard diffusion transition
27: end if
28: end for
29: end for
30: return $\hat{Z}$
</div>

Professional ballet dancers in white tutus perform synchronized, elegant moves on grand stage under spotlights.

Male fitness instructor leads HIIT in bright gym, energetic, clear instructions, professional shots.

Cute cartoon penguins slide joyfully down snowy Antarctic slope, sunny day, icebergs, lively tracking shot.

![](images/4f9c019995ed5fc26276a275aabb0cd4fa47e58311b0d4692bd5f52132de2f42.jpg)  
Gothic horror animation: Woman with candelabra explores haunted Victorian mansion on stormy night.  
Cute otter plays red ukulele on sunlit riverbank, natural fresh vibe, medium close-up.

Fig. S3: Additional qualitative results of Cycle-World across diverse scenes and complex motions, demonstrating its high visual fidelity and robust spatiotemporal consistency.

A person is sweeping florr.  
![](images/319a913e8aa9ad5754fe312f7a1656b0bbade84da605bdf25980bf3b52e7b3db.jpg)

Spider-Man & Green Goblin chase in neon NYC night, Into the Spider-Verse cel-shaded style, dynamic action.  
![](images/6fc9b244470e58612290064e61d3cfc40c75ded9a3f1ea697616a27c73f80088.jpg)  
Fig. S4: Qualitative results of integrating Cycle-Guided Inference (CGI) into CausVid and Self-Forcing. The integration improves spatiotemporal consistency and corrects physical motion anomalies without model retraining.