# Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding

Hee Suk Yoon 1 Eunseop Yoon 1 Jaehyun Jang 1 SooHwan Eom 1 Ji Woo Hong 1 Mark Hasegawa-Johnson 2 Qi Dai 3 Chong Luo 3 Chang D. Yoo 1

# Abstract

While on-policy distillation offers dense supervision for training small reasoning models, its optimization dynamics in the multimodal domain remain under-explored. In this work, we challenge the standard monolithic view of Vision-Language Model (VLM) distillation by mathematically decomposing the loss into two distinct components: the language prior and visual grounding. Our analysis uncovers that gradient vectors for these components are nearly orthogonal, indicating that the objective of aligning with the teacher’s language distribution is geometrically independent from the objective of matching its visual perception. Consequently, standard optimization passively follows a suboptimal compromise trajectory that implicitly balances the two objectives. Hypothesizing that visual grounding constitutes the primary bottleneck for visionlanguage reasoning, we introduce Visual Gradient Steering (VGS), a method that dynamically reorients the update vector to prioritize the visual subspace. Experimental results on multiple distillation settings and complex multimodal benchmarks demonstrate that VGS significantly outperforms the standard monolithic formulation of on-policy distillation, achieving superior grounding with minimal training overhead. The code is publicly accessible at https://github.com/hee-sukyoon/Decomposed OPD.

# 1. Introduction

The recent surge in reasoning models has demonstrated that scaling inference-time compute by enabling models to gen-

1Korea Advanced Institute of Science and Technology (KAIST), 2University of Illinois Urbana-Champaign (UIUC), 3Microsoft Research Asia (MSRA). Correspondence to: Chang D. Yoo <cd yoo@kaist.ac.kr>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

![](images/e37229b05c1b24b296d370edbaa62afeeed569403a88a57d9cf017557d22c88d.jpg)  
Figure 1. Visual Gradient Steering (VGS) outperforms standard monolithic distillation. We compare the reasoning performance of student models distilled from a 8B teacher. VGS (purple) consistently surpasses the standard baseline (green) across diverse multimodal benchmarks for both (a) 2B and (b) 4B students, demonstrating superior visual grounding.

erate intermediate reasoning steps can dramatically enhance problem-solving capabilities (Comanici et al., 2025; Yang et al., 2025; Team, 2025). To train these reasoning models, Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as the standard for domains like mathematics and coding where ground truth is easily verifiable (Yang et al., 2024; Guo et al., 2025; Shao et al., 2024; Yoon et al., 2025a; 2026). However, for smaller models, RLVR encounters a cold start problem. Sparse, outcome-based rewards fail to provide sufficient signal for models that initially lack robust reasoning policies.

On-Policy Distillation (Agarwal et al., 2024; Xi et al., 2024; Deng et al., 2025) offers a superior alternative. Unlike RLVR which relies on delayed outcome feedback, on-policy distillation leverages a stronger teacher model to provide dense, token-level supervision on the student’s own rollouts. This overcomes the reward sparsity problem while still allowing the student to learn error recovery and reasoning coherence, akin to a chess coach critiquing every move rather than just the final result.

Despite its success in text-only domains, on-policy distillation remains under-explored in Vision-Language Models (VLMs). For instance, recent models such as Qwen3- VL (Bai et al., 2025) explicitly restrict distillation to textonly data to fine-tune the LLM backbone, neglecting visual grounding alignment.

Generate Rollout from Student Policy   
![](images/b7592089b9848607a2790eaa81a3a4f66588029a9be2fe17cc8e3fd119ba409b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Image I Prompt x"] --> B["Student p_S^θ"]
    B --> C["Rollout τ"]
    D["Image I Prompt x Rollout τ"] --> E["Student p_S^θ"]
    E --> F["logp_S^θ(τ|I,x)"]
    E --> G["logp_S^θ(τ|x)"]
    E --> H["logq_T(τ|I,x)"]
    E --> I["logq_T(τ|x)"]
    J["Visual Target Construction (q_T*) logq_T*(τ|I,x) ∝ log p_S^θ(τ|x) + [logq_T(τ|I,x) - logq_T(τ|x)"]] --> K["Gradient Steering"]
    L["L_Vis D_KL(p_S^θ(τ|I,x)||q_T*(τ|I,x))"]
    M["L_Lang D_KL(p_S^θ(τ|x)||q_T(τ|x))"]
    N["L_Standard D_KL(p_S^θ(τ|I,x)||q_T(τ|I,x))"]
    O["Image I Prompt x"] --> P["Student p_S^θ"]
    Q["Image I Prompt x Rollout τ"] --> R["Teacher q_T"]
    S["Image I Prompt x"] --> T["Student p_S^θ"]
    U["Image I Prompt x Rollout τ"] --> V["Teacher q_T"]
    W["Visual Target Construction (q_T*) logq_T*(τ|I,x) ∝ log p_S^θ(τ|x)"]
    X["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    Y["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    Z["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AA["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AB["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AC["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AD["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AE["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AF["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AG["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AH["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AI["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AJ["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AK["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AL["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AM["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AN["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AO["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AP["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AQ["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AR["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AS["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AT["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AU["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AV["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AW["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AX["Visual Target Construction (q_T*) logq_T*(τ|x) ∝ log p_S^θ(τ|x)"]
    AY["Image I Prompt x"] --> Z["Student p_S^θ"]
    AA["Image I Prompt x Rollout τ"] --> AB["Student p_S^θ"]
    AC["Image I Prompt x Rollout τ"] --> AD["Teacher q_T"]
    AE["Image I Prompt x Rollout τ"] --> AF["Teacher q_T"]
    AG["Image I Prompt x"] --> AH["Student p_S^θ"]
    AI["Image I Prompt x Rollout τ"] --> AJ["Teacher q_T"]
```
</details>

Figure 2. Overview of Visual Gradient Steering (VGS). (Left) Given a multimodal query (Image I and Prompt x), we sample a rollout τ from the student policy and compute log-probabilities from both the student and teacher models under multimodal (I, x) and unimodal (x, text-only) contexts. (Middle) We decompose the standard monolithic objective into two distinct components: a Language Prior $( \mathcal { L } _ { \mathrm { L a n g } } )$ that matches the teacher’s text-only distribution, and a Visual Grounding objective $( \mathcal { L } _ { \mathrm { V i s } } )$ that targets a constructed distribution q∗T isolating the teacher’s visual information gain. (Right) Our geometric analysis reveals that standard distillation $( \mathcal { L } _ { \mathrm { { S t a n d a r d } } } )$ acts as a passive compromise between these often orthogonal objectives. VGS explicitly steers the gradient update toward the visual subspace $( \mathcal { L } _ { \mathrm { V i s } } )$ , prioritizing perceptual fidelity over generic language modeling.

We find that a direct extension of standard on-policy distillation to the multimodal domain typically defaults to optimizing a single, monolithic objective. In this work, we challenge this formulation by mathematically decomposing the loss into distinct Language Prior and Visual Grounding components, uncovering a fundamental tension between them (Figure 2). Through gradient analysis, we identify a geometric pathology where the gradient vectors for these components are frequently orthogonal. This indicates that the objective of aligning with the teacher’s language distribution is geometrically independent from the objective of matching its visual perception. Consequently, standard distillation naively sums these gradients and passively follows a suboptimal compromise trajectory that fails to strictly enforce perceptual fidelity.

Hypothesizing that visual grounding constitutes the primary bottleneck for vision-language reasoning, we introduce Visual Gradient Steering (VGS). This gradient update rule explicitly prioritizes the visual subspace by dynamically reorienting the optimization trajectory to maximize the alignment with the teacher’s visual information gain. By ensuring that the model concentrates its update budget on resolving perceptual ambiguities rather than generic language modeling, VGS achieves superior alignment performance (Figure 1). In summary, our key contributions are:

• Geometric Analysis of VLM Distillation. We provide the first gradient-level analysis of on-policy distillation in VLMs, revealing the orthogonality between language and vision gradients and identifying the suboptimal ”compromise trajectory” of standard monolithic formulations.   
• Visual Gradient Steering (VGS). We propose a novel optimization objective that separates visual and language su-

pervision, steering the gradient update to strictly enforce visual grounding without degrading linguistic fluency.

• Superior Reasoning Alignment. We show that VGS significantly outperforms the standard monolithic formulation of on-policy distillation on complex multimodal reasoning benchmarks, with minimal training overhead.

# 2. Background and Problem Formulation

# 2.1. Distillation for Autoregressive Reasoning

In reasoning tasks, given a query x, the model generates a multi-step reasoning trajectory τ . Formally, $\tau =$ $( h _ { 1 } , \ldots , h _ { T } , \hat { y } )$ consists of a sequence of intermediate reasoning steps $\{ h _ { k } \} _ { k = 1 } ^ { T }$ , commonly termed a Chain-of-Thought (CoT), followed by a final predicted answer yˆ.

In the context of distillation, we optimize a student model $p _ { \mathrm { S } } ^ { \theta }$ to match a fixed teacher model $q _ { \mathrm { T } }$ using dense, tokenlevel supervision. We quantify the discrepancy between the teacher and student for a trajectory τ using the sequencelevel Kullback-Leibler (KL) divergence, defined as the average token-level Forward KL divergence across the sequence:

$$
\ell_ {\text { Forward }} (\tau) \triangleq \frac {1}{| \tau |} \sum_ {t = 1} ^ {| \tau |} D _ {\mathrm{KL}} \left(q _ {\mathrm{T}} (\cdot \mid \tau_ {<   t}, x) \| p _ {\mathrm{S}} ^ {\theta} (\cdot \mid \tau_ {<   t}, x)\right).
$$

The notations $q _ { T } ( \cdot \mid \tau _ { < t } , x )$ and $p _ { S } ^ { \theta } ( \cdot \mid \tau _ { < t } , x )$ denote the full categorical distributions over the vocabulary V at step t for the teacher and student, respectively.

Standard Sequence-Level Knowledge Distillation (SeqKD) (Kim & Rush, 2016; Taori et al., 2023), often referred to as Off-Policy Distillation, minimizes the Forward KL Divergence. This objective computes the expected sequence divergence over trajectories sampled strictly from the teacher:

$$
\mathcal {L} _ {\text { Off - Policy }} = \underset {\tau \sim q _ {\mathrm{T}}} {\mathbb {E}} \left[ \ell_ {\text { Forward }} (\tau) \right]. \tag {1}
$$

While effective for initialization, Forward KL is modecovering, where it forces the student to assign probability mass to all modes of the teacher’s distribution. For reasoning tasks, this is often suboptimal, since if the student lacks the capacity to model the full complexity of the teacher, it forces the student to bridge distinct modes with low-probability transitions, leading to hallucinations and implausible reasoning chains (Gu et al., 2024; Agarwal et al., 2024). Furthermore, this off-policy approach suffers from exposure bias: the student is never trained to recover from its own autoregressive errors.

To address these limitations, recent works (Agarwal et al., 2024; Gu et al., 2024) advocate for On-Policy Distillation via the Reverse KL Divergence. We first define the trajectory-wise Reverse KL for a trajectory τ as:

$$
\ell_ {\text { Reverse }} (\tau) \triangleq \frac {1}{| \tau |} \sum_ {t = 1} ^ {| \tau |} D _ {\mathrm{KL}} \left(p _ {\mathrm{S}} ^ {\theta} (\cdot \mid \tau_ {<   t}, x) \| q _ {\mathrm{T}} (\cdot \mid \tau_ {<   t}, x)\right).
$$

This approach minimizes the expected sequence-level divergence over trajectories sampled from the student:

$$
\mathcal {L} _ {\text { On - Policy }} = \underset {\tau \sim p _ {S} ^ {\theta}} {\mathbb {E}} \left[ \ell_ {\text { Reverse }} (\tau) \right]. \tag {2}
$$

This formulation offers two primary benefits. First, Reverse KL is mode-seeking; it penalizes the student for generating samples unlikely under the teacher $( q _ { T } \approx 0 )$ , prioritizing high-confidence reasoning paths over broad coverage (Gu et al., 2024; Agarwal et al., 2024). Second, by sampling $\tau \sim p _ { S } ^ { \theta }$ , the student learns from its own rollouts (on-policy), effectively closing the training-inference gap.

# 2.2. The Standard Monolithic Multimodal Objective

In the Vision-Language domain, we extend this on-policy formulation to include visual context. Given an input image I and text prompt x, we define the monolithic trajectorywise divergence for a trajectory τ as:

$$
\ell_ {\text { Standard }} (\tau) \triangleq \frac {1}{| \tau |} \sum_ {t = 1} ^ {| \tau |} D _ {\mathrm{KL}} \left(p _ {\mathrm{S}} ^ {\theta} (\cdot \mid \tau_ {<   t}, I, x) \| q _ {\mathrm{T}} (\cdot \mid \tau_ {<   t}, I, x)\right). \tag {3}
$$

Standard approaches optimize the expectation of this quantity over trajectories sampled from the student:

$$
\mathcal {L} _ {\text { Standard }} = \underset {\tau \sim p _ {S} ^ {\theta} (\cdot | I, x)} {\mathbb {E}} \left[ \ell_ {\text { Standard }} (\tau) \right]. \tag {4}
$$

However, as we demonstrate next, this treats the learning signal as a unified whole, obscuring the distinct contributions of Language Prior Matching and Visual Grounding.

# 3. Revisiting On-Policy Distillation in Vision-Language

# 3.1. Decomposing the Monolithic Objective

By applying Bayes’ Rule, we can factorize the conditional probability of any multimodal generator p to isolate the underlying components of the objective. The log-likelihood decomposes into a language prior, a visual likelihood, and a sequence-independent constant:

$$
\log p (\tau \mid I, x) = \underbrace {\log p (\tau \mid x)} _ {\text { Language   Prior }} + \underbrace {\log p (I \mid \tau , x)} _ {\text { Visual   Likelihood }} - \log p (I \mid x). \tag {5}
$$

Applying this identity to both the student $p _ { S } ^ { \theta }$ and teacher qT reveals that the standard monolithic objective implicitly combines two distinct goals: matching the Language Prior (reasoning style) and matching the Visual Likelihood (perceptual grounding). Note that the evidence term $\log p ( I \mid x )$ is constant with respect to the generated sequence τ and acts solely as a normalizing factor.

I. Language Prior Matching $( \mathcal { L } _ { \mathrm { L a n g } } )$ . This objective aligns the student’s unimodal distribution $p _ { S } ^ { \theta } ( \cdot \mid x )$ with the teacher’s $q _ { T } ( \cdot \mid x )$ , ensuring that the base reasoning style is transferred independent of visual context I. We first define the trajectory-wise language divergence:

$$
\ell_ {\mathrm{Lang}} (\tau) \triangleq \frac {1}{| \tau |} \sum_ {t = 1} ^ {| \tau |} D _ {K L} \big (p _ {\mathrm{S}} ^ {\theta} (\cdot \mid \tau_ {<   t}, x) \| q _ {\mathrm{T}} (\cdot \mid \tau_ {<   t}, x) \big). \tag {6}
$$

We minimize the expectation of this divergence. Crucially, while we compute gradients on the unimodal distributions, we sample trajectories τ from the full multimodal policy $p _ { S } ^ { \theta } ( \cdot \mid I , x )$ to maintain on-policy alignment with the student’s actual generation distribution:

$$
\mathcal {L} _ {\text { Lang }} = \underset {\tau \sim p _ {S} ^ {\theta} (\cdot | I, x)} {\mathbb {E}} \left[ \ell_ {\text { Lang }} (\tau) \right]. \tag {7}
$$

II. Visual Grounding Matching $( \mathcal { L } _ { \mathbf { V i s } } )$ . This objective isolates the Visual Likelihood term identified in Eq. 5. This term, log $p ( I \mid \tau , x )$ , represents the model’s perceptual sensitivity, quantifying how well the generated reasoning causally explains the visual input. Although explicit computation of the likelihood is intractable in autoregressive models, Bayes’ Rule reveals that it differs from the computable Visual Information Gain—the log-ratio of the posterior to the prior—only by a sequence-independent constant:

$$
\log p (I \mid \tau , x) = \underbrace {\left[ \log p (\tau \mid I , x) - \log p (\tau \mid x) \right]} _ {\text { Visual   Information   Gain }} + \log p (I \mid x). \tag {8}
$$

Since the evidence term log $p ( I \mid x )$ depends only on the image and is constant with respect to the trajectory $\tau ,$ matching the teacher’s information gain is mathematically equivalent to aligning the student’s visual perception.

![](images/42cab9d5fb0787bd9d08f95ae3d4fdbffbd37089f9bbe78c7a192575f1c8585e.jpg)

<details>
<summary>scatter</summary>

| Bin | Mean | Mean ± 0.5 STD | Orthogonal |
|-----|------|----------------|------------|
| bin 0 | 62 | 18 | 90 |
| bin 1 | 68 | 16 | 90 |
| bin 2 | 70 | 14 | 90 |
| bin 3 | 74 | 12 | 90 |
| bin 4 | 72 | 10 | 90 |
| bin 5 | 76 | 8 | 90 |
| bin 6 | 80 | 6 | 90 |
| bin 7 | 84 | 4 | 90 |
| bin 8 | 88 | 2 | 90 |
| bin 9 | 92 | 0 | 90 |
</details>

(a) Angles between $\nabla L _ { L a n g }$ and $\nabla \pmb { { \mathcal { L } } } _ { V i s }$

![](images/652f1150bd451742fdc7aa90ba85cb01cc331671664c01e94389eff1303293fe.jpg)

<details>
<summary>scatter</summary>

| Bin | Mean | Mean ± 0.5 STD | Orthogonal |
| --- | --- | --- | --- |
| bin 0 | 30 | 20 | 90 |
| bin 1 | 40 | 60 | 90 |
| bin 2 | 45 | 65 | 90 |
| bin 3 | 50 | 70 | 90 |
| bin 4 | 45 | 60 | 90 |
| bin 5 | 50 | 65 | 90 |
| bin 6 | 55 | 70 | 90 |
| bin 7 | 55 | 70 | 90 |
| bin 8 | 50 | 65 | 90 |
| bin 9 | 45 | 55 | 90 |
</details>

(b) Angles between $\nabla \mathcal { L } _ { S t a n d a r d }$ and $\nabla \mathcal { L } _ { V i s }$

![](images/6b6b023546cb6c2dcf54002b0edf6f831819ad89bcad585f7c86bea56ead60c9.jpg)

<details>
<summary>scatter</summary>

| Bin | Mean | Mean ± 0.5 STD | Orthogonal |
| --- | --- | --- | --- |
| bin 0 | 45 | 20 | 90 |
| bin 1 | 50 | 25 | 90 |
| bin 2 | 48 | 22 | 90 |
| bin 3 | 47 | 21 | 90 |
| bin 4 | 49 | 23 | 90 |
| bin 5 | 48 | 24 | 90 |
| bin 6 | 49 | 26 | 90 |
| bin 7 | 50 | 27 | 90 |
| bin 8 | 55 | 30 | 90 |
| bin 9 | 68 | 35 | 90 |
</details>

(c) Angles between $\nabla \mathcal { L } _ { S t a n d a r d }$ and $\nabla L _ { L a n g }$   
Figure 3. Geometric Analysis of Gradient Dynamics. (a) As visual dependency increases, the angle between Language and Visual gradients widens, approaching orthogonality in high-dependency regions. The emergence of obtuse angles $( > 9 0 ^ { \circ }$ , dashed box) at the extreme specifically motivates the Language Preservation regularizer (Eq. 16) to prevent gradient conflict. (b, c) The standard monolithic gradient $(  { \nabla } { \mathcal { L } } _ { \mathrm { S t a n d a r d } } )$ acts as a passive bisector, maintaining a relatively static compromise orientation $( \approx 4 0 ^ { \circ } - 5 0 ^ { \circ } )$ ) relative to both components, failing to fully align with the visual signal even when perceptual necessity is maximal (Bin 9).

Formally, we achieve this by constructing a visual target distribution $q _ { T } ^ { * }$ that retains the student’s language prior but substitutes its visual likelihood with that of the teacher:

$$
q _ {T} ^ {*} (\tau \mid I, x) \propto p _ {S} ^ {\theta} (\tau \mid x) \cdot q _ {T} (I \mid \tau , x). \tag {9}
$$

Substituting the Bayesian expansion of the teacher’s likelihood into Eq. 9 yields the tractable target logits in log-space:

$$
\log q _ {T} ^ {*} (\tau \mid I, x) = \log p _ {S} ^ {\theta} (\tau \mid x) +
$$

$$
\left(\log q _ {T} (\tau \mid I, x) - \log q _ {T} (\tau \mid x)\right) - \log Z ^ {*}. \tag {10}
$$

Crucially, the marginal evidence term log $q _ { T } ( I \mid x )$ acts as a scalar offset across all tokens. In our autoregressive formulation, this constant is absorbed into the local partition function $Z ^ { \ast }$ , which is computed implicitly via the Softmax operator at each decoding step.

We finally define the trajectory-wise visual divergence $\ell _ { \mathrm { V i s } } ( \tau )$ against this target and its corresponding expectation:

$$
\ell_ {\mathrm{Vis}} (\tau) \triangleq \frac {1}{| \tau |} \sum_ {t = 1} ^ {| \tau |} D _ {K L} \left(p _ {S} ^ {\theta} (\cdot \mid \tau_ {<   t}, I, x) \| q _ {T} ^ {*} (\cdot \mid \tau_ {<   t}, I, x)\right),
$$

$$
\mathcal {L} _ {\text { Vis }} = \underset {\tau \sim p _ {S} ^ {\theta} (\cdot | I, x)} {\mathbb {E}} \left[ \ell_ {\text { Vis }} (\tau) \right]. \tag {11}
$$

# 3.2. Geometric Analysis of Gradient Dynamics

To understand the optimization dynamics of the standard monolithic objective, we empirically analyzed the geometric relationships between its decomposed gradient vectors. Specifically, we investigate how the standard update $ { \nabla } { \mathcal { L } } _ { \mathrm { { S t a n d a r d } } }$ positions itself relative to the Language Prior $( \nabla { \mathcal { L } } _ { \mathrm { L a n g } } )$ and Visual Grounding $( \nabla \mathcal { L } _ { \mathrm { V i s } } )$ gradients across tokens with varying degrees of visual dependency.

Analysis Setup. We analyze the gradient dynamics on validation prompts, using trajectories sampled on-policy from the full multimodal student policy, $\tau \sim p _ { S } ^ { \theta } ( \cdot \mid I , x )$ . To categorize the perceptual necessity of each generated token, we define the Visual Dependency Score (VDS) using the teacher’s distribution:

$$
\mathrm{VDS} _ {t} = D _ {K L} (q _ {T} (\cdot \mid \tau_ {<   t}, I, x) \parallel q _ {T} (\cdot \mid \tau_ {<   t}, x)). \tag {12}
$$

We bin all generated tokens into 10 equal-frequency quantiles based on their VDS; bin 0 represents tokens with minimal visual dependence, while bin 9 represents tokens where the teacher relies heavily on visual evidence.

Figure 3 visualizes the angular relationships across these bins. Our analysis highlights two key geometric properties:

I. Orthogonality at the Visual Extremes (Figure ${ \bf 3 - ( a ) ) }$ . We first examine the relationship between the language and visual gradients. We observe a distinct monotonic trend: as the visual dependency of the token increases, the angle between $\nabla \mathcal { L } _ { \mathrm { L a n g } }$ and $\nabla \mathcal L _ { \mathrm { V i s } }$ widens.

• In low-dependency regions (bin 0), the gradients are moderately aligned $( \theta \approx 6 0 ^ { \circ } )$ ).   
• In high-dependency regions (bin 9), the gradients become nearly orthogonal (θ ≈ 92◦).

II. The Standard Loss as a Static Bisector (Figure 3- (b), (c)). Next, we analyze the orientation of the standard monolithic gradient $ { \nabla } { \mathcal { L } } _ { \mathrm { S t a n d a r d } }$ . Figures 3-(b) and (c) show the angle of the standard gradient relative to the Visual and Language components, respectively.

We observe that the standard objective consistently maintains a “compromise” trajectory. Even for the most visually critical tokens (Bin 9), where the ideal update should arguably align with the visual evidence, $ { \nabla } { \mathcal { L } } _ { \mathrm { S t a n d a r d } }$ retains a significant offset of $\approx 4 2 ^ { \circ }$ from $\nabla \mathcal L _ { \mathrm { V i s } }$ . Simultaneously, it remains similarly distant from the language gradient $( \approx 5 0 ^ { \circ } )$ . This geometric rigidity indicates that the standard objective acts as a passive bisector: it creates a fixed average between the language and visual directions, treating both signals as equally important.

![](images/e768946914a03d25376e1dff44567bab32a094d3febb7b0ddec6b5e97ba01efb.jpg)

<details>
<summary>text_image</summary>

LStandard LVGS (γ = 1.0) LVGS (γ = 2.0) LVGS-LP (γ = 2.0, λ = 0.01)
</details>

![](images/f6a0768380b08324f110e268252826b0313efa9587abd3d54fae6caa63636dfe.jpg)

<details>
<summary>line</summary>

| step | LVis (line 1) | LVis (line 2) | LVis (line 3) |
| ---- | ------------- | ------------- | ------------- |
| 0    | 0.72          | 0.72          | 0.72          |
| 100  | 0.48          | 0.48          | 0.48          |
| 300  | 0.32          | 0.32          | 0.32          |
| 600  | 0.32          | 0.32          | 0.32          |
</details>

(a) Training Dynamics of $\pmb { \mathcal { L } } _ { V i s }$   
![](images/b29ab4e3a9a85d1657f371e8cf9e085ae32f2a71acd4bfd34e585f73090405e3.jpg)

<details>
<summary>line</summary>

| step | L_Vis (line 1) | L_Vis (line 2) | L_Vis (line 3) | L_Vis (line 4) |
|------|----------------|----------------|----------------|----------------|
| 100  | 1.8            | 1.7            | 1.6            | 1.5            |
| 300  | 1.2            | 1.1            | 1.0            | 0.9            |
| 600  | 1.0            | 0.9            | 0.8            | 0.7            |
</details>

![](images/206ba1b51b3358706b2a7865881023dcc93498c893ae40ab84d23e48cc1aaad5.jpg)

<details>
<summary>line</summary>

| step | LLang (blue) | LLang (pink) | LLang (green) |
| ---- | ------------ | ------------ | ------------- |
| 0    | 0.8          | 0.8          | 0.8           |
| 100  | 0.6          | 0.55         | 0.5           |
| 300  | 0.65         | 0.58         | 0.52          |
| 600  | 0.67         | 0.59         | 0.53          |
</details>

(b) Training Dynamics of $\mathcal { L } _ { L a n g }$

![](images/929d6448d455f674c4ae03633e2c8643e9791040d57399b2bb135ce6080a0f9a.jpg)

<details>
<summary>line</summary>

| step | LLang (blue) | LLang (light blue) | LLang (pink) | LLang (green) |
| ---- | ------------ | ------------------ | ------------ | ------------- |
| 0    | 1.6          | 1.6                | 1.6          | 1.6           |
| 100  | 1.4          | 1.4                | 1.4          | 1.2           |
| 300  | 1.5          | 1.4                | 1.3          | 1.2           |
| 600  | 1.6          | 1.5                | 1.3          | 1.2           |
</details>

Figure 4. Training Dynamics. We compare the evolution of decomposed loss components during training across different methods. (a) Visual Grounding $( { \mathcal { L } } _ { \mathbf { V i s } } ) { \ : } :$ : VGS (blue) significantly accelerates visual learning compared to the standard monolithic baseline $( { \mathcal { L } } _ { \mathrm { { S t a n d a r d } } } ,$ green), especially for high-dependency tokens (Bins 7–9). (b) Language Prior $( { \mathcal { L } } _ { \mathrm { L a n g } } ) { \mathrm { : } }$ Without regularization, aggressive visual steering $( \gamma = 2 . 0 $ , dark blue) causes the language prior to diverge (“unlearning”), particularly in high-conflict visual regimes (Bins 7–9). Our final method with Language Preservation (LVGS-LP, pink) successfully prevents this degradation while maintaining superior visual grounding.

Hypothesis: Breaking the Optimization Symmetry. We hypothesize that the standard objective is suboptimal because it enforces geometric symmetry on an inherently asymmetric task. Since performance is constrained by a perceptual bottleneck, and language gradients are orthogonal to visual improvements, the standard compromise update is inefficient. We propose that explicitly steering the gradient toward the visual subspace introduces a necessary inductive bias, prioritizing the resolution of perceptual ambiguities over generic language modeling.

# 4. Method: Visual Gradient Steering

Motivated by the geometric pathology observed in Sec. 3.2, we propose Visual Gradient Steering (VGS) to construct a gradient update that prioritizes the visual subspace.

The Steered Objective. We first define the trajectory-wise steered objective $\ell _ { \mathrm { V G S } } ( \tau )$ , which augments the standard distillation loss $\ell _ { \mathrm { S t a n d a r d } } ( \tau )$ with an auxiliary visual term:

$$
\ell_ {\mathrm{VGS}} (\tau) \triangleq \ell_ {\text { Standard }} (\tau) + \gamma \ell_ {\mathrm{Vis}} (\tau), \tag {13}
$$

where $\ell _ { \mathrm { V i s } } ( \tau )$ corresponds to the visual grounding objective derived in Eq. 11, and $\gamma \geq 0$ is the steering coefficient that controls the strength of the visual correction. The final loss function is then minimizing the expected steered objective:

$$
\mathcal {L} _ {\mathrm{VGS}} = \eta_ {\mathrm{VGS}} (\gamma) \cdot \underset {\tau \sim p _ {S} ^ {\theta} (\cdot | I, x)} {\mathbb {E}} \left[ \ell_ {\mathrm{VGS}} (\tau) \right]. \tag {14}
$$

Gradient Norm Normalization. To ensure the optimization remains stable, we introduce a scaling factor ηVGS(γ). We define $\eta _ { \mathrm { V G S } } ( \gamma )$ such that the norm of the steered gradient equals the norm of the original standard gradient:

$$
\eta_ {\mathrm{VGS}} (\gamma) \triangleq \frac {\| \nabla_ {\theta} \mathcal {L} _ {\text { Standard }} \| _ {2}}{\| \nabla_ {\theta} \mathcal {L} _ {\text { Standard }} + \gamma \nabla_ {\theta} \mathcal {L} _ {\text { Vis }} \| _ {2}}. \tag {15}
$$

This normalization ensures that VGS changes only the direction of the update—steering it toward the visual subspace—without altering the magnitude of the parameter steps. This effectively decouples the “steering” (controlled by γ) from the “learning rate” (controlled by the optimizer), a stability principle widely adopted in multi-task learning (Chen et al., 2018). In practice, to avoid the computational overhead of dynamic norm calculation, we set $\eta _ { \mathrm { V G S } } ( \gamma )$ to a fixed constant dependent on γ, as empirical analysis in Appendix A confirms the gradient magnitude ratio remains stable throughout training.

Mitigating Destructive Gradient Interference. While VGS effectively steers generation toward visual grounding, our analysis (Figure 3-(a)) reveals that tokens with the highest visual dependency exhibit obtuse angles $( > 9 0 ^ { \circ } )$ between the visual and language gradients. In these regimes, optimizing solely for visual grounding yields a negative projection onto the language gradient, effectively “unlearning” the language prior. Figure 4 confirms this empirically: the Language Prior Matching loss $( \mathcal { L } _ { \mathrm { L a n g } } )$ for high-VDS tokens diverges significantly during training with ${ \mathcal { L } } _ { \mathrm { V G S } }$ alone.

To prevent this, we introduce a Language Preservation (LP) regularizer. We apply this term selectively to the top 30% of tokens ranked by VDS (Eq. 12; corresponding to Bins 7–9), where this gradient conflict is most prevalent. We define the trajectory-wise preservation loss as:

$$
\ell_ {\mathrm{LP}} (\tau) \triangleq \frac {1}{| \tau |} \sum_ {t = 1} ^ {| \tau |} \left(\mathbb {1} [ \mathrm{VDS} _ {t} > Q _ {0. 7} ] \right. \tag {16}
$$

$$
\cdot D _ {K L} \left(p _ {\mathrm{S}} ^ {\theta} (\cdot | \tau_ {<   t}, x) \| q _ {\mathrm{T}} (\cdot | \tau_ {<   t}, x)\right),
$$

Table 1. Main Results on Vision-Language Reasoning Benchmarks. We compare the distillation performance of Visual Gradient Steering (VGS) against the standard monolithic baseline. All student models (2B and 4B) are distilled from the same Qwen3-VL-8B-Instruct teacher trained with GRPO. VGS consistently outperforms the standard approach across all benchmarks, achieving higher accuracy in both greedy decoding (Acc@1) and stochastic sampling (Acc@16). 

<table><tr><td rowspan="2">Benchmark</td><td>Teacher</td><td>Initial Student</td><td colspan="2">Standard On-Policy Distillation</td><td colspan="2">VGS On-Policy Distillation (ours)</td><td colspan="2">Improvement</td></tr><tr><td>Acc@1 (greedy)</td><td>Acc@1 (greedy)</td><td>Acc@1 (greedy)</td><td>Acc@16 (T=1.0)</td><td>Acc@1 (greedy)</td><td>Acc@16 (T=1.0)</td><td>Acc@1 (greedy)</td><td>Acc@16 (T=1.0)</td></tr><tr><td></td><td>8B</td><td>2B</td><td colspan="4">Qwen3-VL-8B-Instruct → Qwen3-VL-2B-Instruct</td><td></td><td></td></tr><tr><td>MMMU-Pro-4</td><td>62.03</td><td>34.51</td><td>45.83</td><td>47.33</td><td>48.07</td><td>48.34</td><td>+2.14</td><td>+1.01</td></tr><tr><td>LogicVista</td><td>60.01</td><td>36.83</td><td>45.53</td><td>46.19</td><td>48.88</td><td>46.47</td><td>+3.35</td><td>+0.28</td></tr><tr><td>MathVerse-VD</td><td>79.63</td><td>35.88</td><td>56.02</td><td>59.68</td><td>58.10</td><td>60.10</td><td>+2.08</td><td>+0.42</td></tr><tr><td>MathVerse-VO</td><td>73.85</td><td>35.32</td><td>54.59</td><td>55.76</td><td>56.19</td><td>56.52</td><td>+1.61</td><td>+0.76</td></tr><tr><td>VisualPuzzles</td><td>43.15</td><td>13.36</td><td>28.08</td><td>30.64</td><td>31.76</td><td>31.59</td><td>+3.68</td><td>+0.95</td></tr><tr><td>MathVision</td><td>44.14</td><td>14.28</td><td>24.14</td><td>25.37</td><td>25.59</td><td>26.73</td><td>+1.45</td><td>+1.36</td></tr><tr><td>VlmsAreBlind</td><td>66.79</td><td>49.03</td><td>51.86</td><td>50.52</td><td>54.11</td><td>53.24</td><td>+2.26</td><td>+2.72</td></tr><tr><td>Average</td><td>61.37</td><td>31.32</td><td>43.74</td><td>45.07</td><td>46.10</td><td>46.14</td><td>+2.37</td><td>+1.07</td></tr><tr><td></td><td>8B</td><td>4B</td><td colspan="4">Qwen3-VL-8B-Instruct → Qwen3-VL-4B-Instruct</td><td></td><td></td></tr><tr><td>MMMU-Pro-4</td><td>62.03</td><td>48.93</td><td>55.79</td><td>56.43</td><td>56.86</td><td>56.86</td><td>+1.07</td><td>+0.43</td></tr><tr><td>LogicVista</td><td>60.01</td><td>47.10</td><td>55.80</td><td>56.43</td><td>58.70</td><td>56.98</td><td>+2.90</td><td>+0.55</td></tr><tr><td>MathVerse-VD</td><td>79.63</td><td>57.18</td><td>71.53</td><td>73.60</td><td>74.31</td><td>73.89</td><td>+2.78</td><td>+0.29</td></tr><tr><td>MathVerse-VO</td><td>73.85</td><td>53.67</td><td>70.18</td><td>68.33</td><td>71.10</td><td>68.53</td><td>+0.92</td><td>+0.20</td></tr><tr><td>VisualPuzzles</td><td>43.15</td><td>26.54</td><td>40.75</td><td>39.17</td><td>40.31</td><td>39.22</td><td>-0.44</td><td>+0.05</td></tr><tr><td>MathVision</td><td>44.14</td><td>31.40</td><td>37.96</td><td>39.26</td><td>40.59</td><td>40.16</td><td>+2.63</td><td>+0.90</td></tr><tr><td>VlmsAreBlind</td><td>66.79</td><td>61.94</td><td>64.46</td><td>64.82</td><td>65.49</td><td>65.92</td><td>+1.03</td><td>+1.10</td></tr><tr><td>Average</td><td>61.37</td><td>46.68</td><td>56.64</td><td>56.86</td><td>58.12</td><td>57.27</td><td>+1.56</td><td>+0.50</td></tr></table>

where $Q _ { 0 . 7 }$ denotes the 70th percentile threshold of the Visual Dependency Score distribution. This term penalizes deviation from the teacher’s language prior only when the visual steering signal is likely to be destructive. We incorporate this into our final trajectory-wise objective:

$$
\ell_ {\mathrm{VGS-LP}} (\tau) = \ell_ {\mathrm{VGS}} (\tau) + \lambda \ell_ {\mathrm{LP}} (\tau) \tag {17}
$$

$$
= \ell_ {\text { Standard }} (\tau) + \gamma \ell_ {\text { Vis }} (\tau) + \lambda \ell_ {\text { LP }} (\tau),
$$

where γ acts as a primary steering coefficient (typically $\gamma \geq$ 1) to drive visual adaptation, while λ serves as a conservative regularization weight (e.g., λ ≈ 0.01) to selectively prevent catastrophic forgetting. The total loss is minimized over the expectation of this objective, scaled by the adaptive normalization factor $\eta _ { \mathrm { V G S } } ( \gamma )$ in Eq. 15:

$$
\mathcal {L} _ {\mathrm{VGS-LP}} = \eta_ {\mathrm{VGS}} (\gamma) \cdot \underset {\tau \sim p _ {S} ^ {\theta} (\cdot | I, x)} {\mathbb {E}} \left[ \ell_ {\mathrm{VGS-LP}} (\tau) \right]. \tag {18}
$$

As demonstrated in Figure 4, this formulation successfully mitigates the divergence of the language prior, ensuring that the model retains its reasoning capabilities even in highconflict visual regimes.

# 5. Experimental Setup

# 5.1. Models and Datasets

We conduct our training on the Vision-SR1-47K dataset (Li et al., 2025). This dataset consists of 47k triplets $( I , x , y )$ , containing an image I, a question x, and a verifiable final answer y. For a controlled experimental setting, we avoid using off-the-shelf reasoning models where the training mixture is unknown. Instead, we train a reasoning teacher by fine-tuning the Qwen3-VL-8B-Instruct (Bai et al., 2025) using Group Relative Policy Optimization (GRPO) (Shao et al., 2024) for 2 epochs on Vision-SR1-47K dataset. Full implementation details are in Appendix B.1. For the student policies, we utilize the smaller Qwen3-VL-2B-Instruct and Qwen3-VL-4B-Instruct (Bai et al., 2025).

We evaluate on seven vision-language reasoning benchmarks. For mathematical reasoning, we utilize MathVision (Wang et al., 2024) and MathVerse (Zhang et al., 2024), reporting results on the Visual Dominant (VD) and Visual Only (VO) subsets. For visual logic, we employ LogicVista (Xiao et al., 2024), VisualPuzzles (Song et al., 2025), and VlmsAreBlind (Rahmanzadehgervi et al., 2024). Finally, we test multidisciplinary reasoning using MMMU-Pro (Yue et al., 2024) in its 4-option setting.

# 5.2. Implementation details

Prompt Format. To ensure structural alignment, both the teacher (during GRPO) and the student (during distillation) utilize a unified system prompt. This enforces a strict reasoning format, requiring the model to explicitly delimit the internal reasoning trace from the final answer. The full prompt template is provided in Appendix B.4.

Hyperparameters. We optimize the objective in Eq. 17 with a steering coefficient of $\gamma = 2 . 0$ and $\lambda = 0 . 0 1$ across all experiments. For the gradient normalization, we set $\eta ( \gamma ) = 0 . 4 1$ for Qwen3-VL-2B-Instruct and $\eta ( \gamma ) = 0 . 3 6$ for Qwen3-VL-4B-Instruct as justified in Appendix A. All other training configurations were kept identical to the baseline for fair comparison. Full hyperparmater details and sensitivity analysis is provided in Appendix B.2, B.3.

# 6. Experimental Results

We evaluate Visual Gradient Steering (VGS) by distilling a GRPO-trained Qwen3-VL-8B-Instruct teacher into 2B and 4B student models. As detailed in Table 1, VGS consistently outperforms the standard monolithic baseline across all seven vision-language benchmarks. The improvement is particularly pronounced when the teacher-student capacity gap is large. In the Qwen3-VL-8B → Qwen3-VL-2B setting, VGS achieves an average greedy accuracy $( \operatorname { A c c } @ 1 )$ of 46.10%, surpassing the baseline (43.74%) by a significant margin of +2.37%. This gain is driven by substantial improvements in visually intensive tasks like VisualPuzzles (+3.68%) and LogicVista (+3.35%). This trend extends to the larger Qwen3-VL-4B student, where VGS improves average Acc@1 to 58.12% (+1.56% over baseline), effectively closing the gap to the 8B teacher (61.37%).

Furthermore, we assess policy robustness via stochastic sampling (Acc@16), calculated as the average accuracy over 16 independent generations (temperature $T = 1 . 0 )$ , where VGS consistently yields higher average accuracy across both scales. We provide a computational overhead comparison in Appendix C and the training dynamics in Section 9.

# 7. Validating the Asymmetric Maturity Hypothesis

A core premise of our approach is the Asymmetric Maturity Hypothesis (Sec. 3.2): we posit that visual grounding is the primary optimization bottleneck, while the language prior is already comparatively robust. To validate this, we perform an inverse steering experiment: What happens if we steer the gradient towards the Language Prior instead?

We define the Language Steering objective symmetrically to VGS-LP. This objective steers the generation toward the unimodal language prior $( \mathcal { L } _ { \mathrm { L a n g } } )$ while strictly preserving visual grounding on high-dependency tokens:

$$
\begin{array}{l} \mathcal {L} _ {\text { Lang - Steer }} = \eta_ {\text { Lang }} (\gamma_ {\text { Lang }}) \cdot \underset {\tau \sim p _ {S} ^ {\theta} (\cdot | I, x)} {\mathbb {E}} \left[ \ell_ {\text { Standard }} (\tau) \right. \\ \left. + \gamma_ {\text { Lang }} \ell_ {\text { Lang }} (\tau) + \lambda \ell_ {\mathrm{VP}} (\tau) \right], \tag {19} \\ \end{array}
$$

where $\eta _ { \mathrm { L a n g } } ( \gamma _ { \mathrm { L a n g } } )$ is the adaptive scaling factor, defined analogously to Eq. 15 but using the language gradient direction to maintain optimization stability:

![](images/dfdba71629e31d2d6df1d21582216f6d5bf107c57e03520b9e4e5bf4ff663688.jpg)

<details>
<summary>bar</summary>

| γ       | Average Accuracy |
| ------- | ---------------- |
| 3.0     | 42.5             |
| 2.0     | 43.0             |
| 1.0     | 42.5             |
| 1.0     | 43.8             |
| 2.0     | 45.2             |
| 3.0     | 46.5             |
</details>

Figure 5. Validating the Asymmetric Maturity Hypothesis. While steering toward the language matching reduces accuracy compared to the standard baseline, steering toward the visual subspace yields consistent performance gains.

$$
\eta_ {\text { Lang }} (\gamma) \triangleq \frac {\left\| \nabla_ {\theta} \mathcal {L} _ {\text { Standard }} \right\| _ {2}}{\left\| \nabla_ {\theta} \mathcal {L} _ {\text { Standard }} + \gamma_ {\text { Lang }} \nabla_ {\theta} \mathcal {L} _ {\text { Lang }} \right\| _ {2}}. \tag {20}
$$

The term $\ell _ { \mathrm { V P } } ( \tau )$ is the Visual Preservation regularizer, which penalizes deviations from the teacher’s visual distribution $q _ { T } ^ { * }$ (Eq. 10) specifically on tokens where visual grounding is most critical:

$$
\begin{array}{l} \ell_ {\mathrm{VP}} (\tau) \triangleq \frac {1}{| \tau |} \sum_ {t = 1} ^ {| \tau |} \left(\mathbb {1} [ \mathrm{VDS} _ {t} > Q _ {0. 7} ] \right. \\ \cdot D _ {K L} \left(p _ {\mathrm{S}} ^ {\theta} (\cdot | \tau_ {<   t}, x) \| q _ {\mathrm{T}} ^ {*} (\cdot | \tau_ {<   t}, x)\right). \tag {21} \\ \end{array}
$$

Similar to our VGS-LP formulation (Eq. 17), we set λ to a small regularization value (i.e., 0.01). This ensures a symmetric comparison where both objectives steer a primary modality while safeguarding the other against catastrophic forgetting.

Figure 5 compares the performance of Visual Steering $( \gamma > 0 )$ versus Language Steering $( \gamma _ { \mathrm { l a n g } } > 0 )$ . As shown in the plot (where the y-axis represents the average accuracy across the 7 benchmarks listed in Table 1), increasing the visual steering coefficient γ yields improvements, boosting the average from the baseline. In contrast, steering towards the language prior $( \gamma _ { \mathrm { l a n g } } > 0 )$ harms performance, with accuracy dropping below the baseline. This confirms that the model’s language prior is already robust; forcing updates towards it consumes the optimization budget without gain. The visual modality, however, remains the critical bottleneck, responding positively to targeted gradient injection.

# 8. RL Fine-Tuning + On-Policy Distillation

As noted by Agarwal et al. (2024), on-policy distillation can be naturally combined with Reinforcement Learning (RL)

Table 2. Ablation Results on RL + On-Policy Distillation. We evaluate the effectiveness of integrating distillation objectives into the GRPO training loop using Qwen3-VL-2B-Instruct. Our Visual Gradient Steered (VGS) method consistently outperforms the standard monolithic distillation baseline when combined with RL. 

<table><tr><td rowspan="2">Benchmark</td><td>Initial Student (2B)</td><td colspan="2">GRPO</td><td colspan="2">GRPO + Standard On-Policy Distillation</td><td colspan="2">GRPO + VGS On-Policy Distillation (ours)</td><td colspan="2">Improvement</td></tr><tr><td>Acc@1 (greedy)</td><td>Acc@1 (greedy)</td><td>Acc@16 (T=1.0)</td><td>Acc@1 (greedy)</td><td>Acc@16 (T=1.0)</td><td>Acc@1 (greedy)</td><td>Acc@16 (T=1.0)</td><td>Acc@1 (greedy)</td><td>Acc@16 (T=1.0)</td></tr><tr><td>MMMU-Pro-4</td><td>34.51</td><td>47.66</td><td>49.59</td><td>45.85</td><td>46.86</td><td>50.29</td><td>49.26</td><td>+4.44</td><td>+2.40</td></tr><tr><td>LogicVista</td><td>36.83</td><td>46.43</td><td>44.24</td><td>44.64</td><td>45.28</td><td>46.88</td><td>46.01</td><td>+2.24</td><td>+0.73</td></tr><tr><td>MathVerse-VD</td><td>35.88</td><td>59.03</td><td>59.68</td><td>61.11</td><td>60.43</td><td>64.81</td><td>61.44</td><td>+3.70</td><td>+1.01</td></tr><tr><td>MathVerse-VO</td><td>35.32</td><td>53.67</td><td>56.47</td><td>58.03</td><td>55.89</td><td>57.80</td><td>57.70</td><td>-0.23</td><td>+1.81</td></tr><tr><td>VisualPuzzles</td><td>13.36</td><td>30.22</td><td>31.64</td><td>31.08</td><td>31.36</td><td>31.68</td><td>32.23</td><td>+0.60</td><td>+0.87</td></tr><tr><td>MathVision</td><td>14.28</td><td>27.20</td><td>28.02</td><td>25.26</td><td>25.98</td><td>27.17</td><td>27.52</td><td>+1.91</td><td>+1.54</td></tr><tr><td>VlmsAreBlind</td><td>49.03</td><td>49.61</td><td>50.16</td><td>51.90</td><td>50.77</td><td>51.74</td><td>51.85</td><td>-0.16</td><td>+1.08</td></tr><tr><td>Average</td><td>31.32</td><td>44.83</td><td>45.68</td><td>45.41</td><td>45.22</td><td>47.20</td><td>46.57</td><td>+1.79</td><td>+1.35</td></tr></table>

![](images/38cb1bbfd8baffa2e1915c12b1211fff413124f8fad548bf1e58cb8711312fe9.jpg)  
Figure 6. Training Dynamics Analysis of Pure Distillation and GRPO Settings. (a) Average accuracy evolution during on-policy distillation for Qwen3-VL-2B and 4B students. VGS achieves higher accuracy throughout training compared to the standard monolithic baseline. (b) Analysis of the GRPO framework (using Qwen3-VL-2B). Left: VGS combined with GRPO yields the highest accuracy trajectory, outperforming both pure GRPO and GRPO + Standard Distillation. Right: While pure GRPO suffers from length explosion, both distillation methods regularize generation length to match the teacher’s average; however, VGS achieves significantly higher accuracy within this efficient length constraint.

objectives, as both paradigms involve sampling trajectories directly from the student’s current policy. We investigate whether our proposed steering towards the visual subspace $( { \mathcal { L } } _ { \mathrm { V G S - L P } } ,$ Eq. 17) can enhance this synergy when fine-tuning with GRPO (Shao et al., 2024).

Following the formulation of Agarwal et al. (2024), we construct the unified training objective as a weighted convex combination of the RL reward and the distillation constraint. We define the total objective ${ \mathcal { I } } _ { \mathrm { T o t a l } } ( \theta )$ as:

$$
\begin{array}{l} \mathcal {J} _ {\text {Total}} (\theta) = \underset {\left\{\tau_ {i} \right\} _ {i = 1} ^ {G} \sim p _ {S} ^ {\theta}} {\mathbb {E}} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \left((1 - \alpha) \mathcal {O} _ {\mathrm{GRPO}} \left(\tau_ {i}\right) \right. \right. \tag {22} \\ - \left. \alpha \ell_ {\mathrm{VGS-LP}} (\tau_ {i})\right) \Biggr ], \\ \end{array}
$$

where $\mathcal { O } _ { \mathrm { G R P O } } ( \tau _ { i } )$ is the standard GRPO surrogate objective (clipped advantage) without the KL penalty, and ℓVGS-LP serves as the visual distillation regularizer. We compare three training configurations on the Vision-SR1-47K dataset (Li et al., 2025):

1. GRPO (Standard): Pure RL optimization maximizing the correctness reward (equivalent to $\alpha = 0 )$ .   
2. GRPO + Standard-KD: RL regularized by the standard monolithic distillation loss. This corresponds to Eq. 22 where $\ell _ { \mathrm { V G S - L P } }$ is replaced by $\ell _ { \mathrm { S t a n d a r d } }$ . We set α to 0.3   
3. GRPO + VGS (Ours): RL regularized by our visually steered objective in Eq. 22. We set α to 0.3.

As shown in Table 2, while Standard-KD improves stability over pure GRPO, it suffers from the compromise effect, pulling the policy back towards the teacher’s generic average. GRPO + VGS significantly outperforms both baselines. By using VGS as the regularizer, we enforce a constraint that is orthogonal to the RL reward: the RL objective optimizes for the correct answer, while VGS ensures the reasoning process remains visually grounded. This suggests that VGS is a versatile objective that can be plugged into various post-training paradigms to enforce perceptual fidelity.

# 9. Training Dynamics

In Figure 6, we present the average accuracy evolution across all 7 benchmark datasets during training. First, Figure 6-(a) compares the 2B and 4B students in the on-policy distillation setting. At both scales, our VGS approach consistently achieves a higher accuracy curve compared to the standard monolithic baseline, demonstrating the effectiveness of visual gradient steering.

Furthermore, Figure 6-(b) (left) displays the average accuracy in the RL (GRPO) + on-policy distillation setup that was introduced in Section 8. While adding distillation accelerates training in general, our VGS yields a distinctively higher accuracy curve. Figure 6-(b) (right) illustrates the generation length dynamics. We observe that both distillation objectives constrain the student to converge toward the teacher’s average generation length, preventing the length explosion seen in standard GRPO. Notably, VGS achieves significantly higher accuracy at this constrained length compared to Standard On-Policy Distillation, further validating the efficacy of visual gradient steering for reasoning efficiency.

# 10. Related Works

Reinforcement Learning for Reasoning Expanding beyond foundational RLHF and preference optimization paradigms (Christiano et al., 2017; Bai et al., 2022; Ouyang et al., 2022; Yoon et al., 2024; Rafailov et al., 2023; Yoon et al., 2025b), recent large-scale reinforcement learning (RL) training across diverse tasks has significantly elevated the general reasoning capabilities of LLMs (Guo et al., 2025; Team et al., 2025; Lambert et al., 2024). These advancements are predominantly driven by Reinforcement Learning from Verifiable Rewards (RLVR) frameworks (Yang et al., 2024; Guo et al., 2025; Shao et al., 2024; Yoon et al., 2025a; 2026).

However, a persistent challenge in standard RLVR environments is that outcome rewards are typically sparse, generated only at the very end of a token sequence. To resolve this credit assignment dilemma, several recent studies have developed process reward models (PRMs) (Li & Li, 2025; Cheng et al., 2026; Zhang et al., 2025; Lee et al., 2025) to evaluate step-by-step progressions within a generation; however, these PRMs typically demand specialized data curation and independent training pipelines. To circumvent reward sparsity without the overhead of an explicit PRM, on-policy distillation has emerged as a compelling alternative, leveraging dense, token-level supervision from a powerful teacher model (Yang et al., 2025; Guha et al., 2025; Agarwal et al., 2024).

On-Policy Knowledge Distillation Knowledge Distillation (KD) has been widely adopted as a popular paradigm for developing efficient Small Large Language Models (SLMs) by leveraging supervision from larger teacher models (Yang et al., 2025; Team et al., 2024; Guo et al., 2025; Muralidharan et al., 2024; Gu et al., 2024; Agarwal et al., 2024). Gu et al. (2024) demonstrated that reverse KL divergence is more effective than forward KL in preventing the student model from overestimating low-probability regions inherent in the teacher distribution. Furthermore, Agarwal et al. (2024) experimentally analyzed the trade-offs between on-policy KD, where the student model generates the training data, and off-policy settings, as well as the balance between reverse and forward KL objectives. In this work, we employ on-policy distillation with reverse KL divergence to enhance reasoning capabilities in instruction-tuned models with minimal distribution shift from initial student (Gu et al., 2024).

Vision-Focused Knowledge Distillation in MLLMs Knowledge distillation in the vision-language domain has recently garnered significant research attention as a key strategy for developing efficient, small-scale Multimodal Large Language Models (MLLMs). Cai et al. (2025) utilized this approach to distill broad visual instruction-following capabilities from large-scale teachers into efficient student models. Furthermore, to address finer-grained perceptual alignment, Jain et al. (2025) introduced visual embedding distillation to transfer spatial knowledge, while Kim et al. (2025) proposed an explicit attention alignment framework to enable student models to inherit the teacher’s compositional reasoning patterns. In contrast to computationally intensive methods relying on auxiliary encoders or explicit feature alignment, our approach minimizes complexity by steering the student’s visual grounding directly through the teacher’s decomposed output logits.

# 11. Conclusion

We identify a fundamental limitation in standard monolithic objectives: due to the asymmetric maturity of MLLM modalities, the gradient acts as a passive bisector that fails to correct the weaker visual signal. We propose Visual Gradient Steering (VGS) to resolve this by explicitly prioritizing the visual subspace during optimization. Our experiments demonstrate that VGS not only outperforms standard distillation but also serves as a critical regularizer for RL (GRPO) fine-tuning. These findings confirm that breaking optimization symmetry, rather than treating modalities equally, is essential for unlocking faithful multimodal grounding.

# Acknowledgements

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. RS-2022- II0951, Development of Uncertainty-Aware Agents Learning by Asking Questions), and MSIT(Ministry of Science, ICT), Korea, under the Global Research Support Program in the Digital Field program(RS-2024-00436680) supervised by the IITP(Institute for Information & Communications Technology Planning & Evaluation). This project is supported by Microsoft Research Asia.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# References

Agarwal, R., Vieillard, N., Zhou, Y., Stanczyk, P., Garea, S. R., Geist, M., and Bachem, O. On-policy distillation of language models: Learning from self-generated mistakes. In The twelfth international conference on learning representations, 2024.   
Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report, 2025. URL https://arxiv.org/abs/2511.21631.   
Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., Das-Sarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.   
Cai, Y., Zhang, J., He, H., He, X., Tong, A., Gan, Z., Wang, C., Xue, Z., Liu, Y., and Bai, X. Llava-kd: A framework of distilling multimodal large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 239–249, 2025.   
Chen, Z., Badrinarayanan, V., Lee, C.-Y., and Rabinovich, A. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International

conference on machine learning, pp. 794–803. PMLR, 2018.

Cheng, J., Xiong, G., Qiao, R., Li, L., Guo, C., Wang, J., Lv, Y., and Wang, F.-Y. Stop summation: Min-form credit assignment is all process reward model needs for reasoning. Advances in Neural Information Processing Systems, 38:131646–131671, 2026.

Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Deng, Y., Hsu, I., Yan, J., Wang, Z., Han, R., Zhang, G., Chen, Y., Wang, W., Pfister, T., Lee, C.-Y., et al. Supervised reinforcement learning: From expert trajectories to step-wise reasoning. arXiv preprint arXiv:2510.25992, 2025.

Gu, Y., Dong, L., Wei, F., and Huang, M. MiniLLM: Knowledge distillation of large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=5h0qf7IBZZ.

Guha, E., Marten, R., Keh, S., Raoof, N., Smyrnis, G., Bansal, H., Nezhurina, M., Mercat, J., Vu, T., Sprague, Z., et al. Openthoughts: Data recipes for reasoning models. arXiv preprint arXiv:2506.04178, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Jain, J., Yang, Z., Shi, H., Gao, J., and Yang, J. Elevating visual perception in multimodal llms with visual embedding distillation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Kim, J., Kim, K., Seo, S., and Park, C. Compodistill: Attention distillation for compositional reasoning in multimodal llms. arXiv preprint arXiv:2510.12184, 2025.

Kim, Y. and Rush, A. M. Sequence-level knowledge distillation, 2016. URL https://arxiv.org/abs/1606. 07947.

Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., Miranda, L. J. V., Liu, A., Dziri, N.,

Lyu, S., et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.   
Lee, D. B., Lee, S., Park, S., Kang, M., Baek, J., Kim, D., Wagner, D., Jin, J., Lee, H., Bocklet, T., et al. Rethinking reward models for multi-domain test-time scaling. arXiv preprint arXiv:2510.00492, 2025.   
Li, W. and Li, Y. Process reward model with q-value rankings. In International Conference on Learning Representations, volume 2025, pp. 14708–14726, 2025.   
Li, Z., Yu, W., Huang, C., Liu, R., Liang, Z., Liu, F., Che, J., Yu, D., Boyd-Graber, J., Mi, H., et al. Selfrewarding vision-language model via reasoning decomposition. arXiv preprint arXiv:2508.19652, 2025.   
Muralidharan, S., Turuvekere Sreenivas, S., Joshi, R., Chochowski, M., Patwary, M., Shoeybi, M., Catanzaro, B., Kautz, J., and Molchanov, P. Compact language models via pruning and knowledge distillation. Advances in Neural Information Processing Systems, 37:41076–41102, 2024.   
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.   
Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36: 53728–53741, 2023.   
Rahmanzadehgervi, P., Bolton, L., Taesiri, M. R., and Nguyen, A. T. Vision language models are blind: Failing to translate detailed visual features into words. arXiv preprint arXiv:2407.06581, 2024.   
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
Song, Y., Ou, T., Kong, Y., Li, Z., Neubig, G., and Yue, X. Visualpuzzles: Decoupling multimodal reasoning evaluation from domain knowledge. arXiv preprint arXiv:2504.10342, 2025. URL https://arxiv. org/abs/2504.10342.   
Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford\_alpaca, 2023.

Team, G., Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahriari, B., Rame, A., et al. Gemma 2: Improving open ´ language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.   
Team, K., Du, A., Gao, B., Xing, B., Jiang, C., Chen, C., Li, C., Xiao, C., Du, C., Liao, C., et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.   
Team, Q. Qwq-32b: Embracing the power of reinforcement learning, March 2025. URL https://qwenlm. github.io/blog/qwq-32b/.   
Wang, K., Pan, J., Shi, W., Lu, Z., Ren, H., Zhou, A., Zhan, M., and Li, H. Measuring multimodal mathematical reasoning with math-vision dataset. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https: //openreview.net/forum?id=QWTCcxMpPA.   
Wang, Z., Guo, X., Stoica, S., Xu, H., Wang, H., Ha, H., Chen, X., Chen, Y., Yan, M., Huang, F., et al. Perception-aware policy optimization for multimodal reasoning. arXiv preprint arXiv:2507.06448, 2025.   
Xi, Z., Chen, W., Hong, B., Jin, S., Zheng, R., He, W., Ding, Y., Liu, S., Guo, X., Wang, J., et al. Training large language models for reasoning through reverse curriculum reinforcement learning. arXiv preprint arXiv:2402.05808, 2024.   
Xiao, Y., Sun, E., Liu, T., and Wang, W. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts, 2024. URL https://arxiv.org/abs/ 2407.04973.   
Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu, D., Tu, J., Zhou, J., Lin, J., et al. Qwen2. 5-math technical report: Toward mathematical expert model via selfimprovement. arXiv preprint arXiv:2409.12122, 2024.   
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.   
Yoon, E., Yoon, H. S., Eom, S., Han, G., Nam, D., Jo, D., On, K.-W., Hasegawa-Johnson, M., Kim, S., and Yoo, C. Tlcr: Token-level continuous reward for fine-grained reinforcement learning from human feedback. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 14969–14981, 2024.   
Yoon, E., Yoon, H. S., Jang, J., Eom, S., Dai, Q., Luo, C., Hasegawa-Johnson, M. A., and Yoo, C. D. Pacr: Progressively ascending confidence reward for llm reasoning. arXiv preprint arXiv:2510.22255, 2025a.

Yoon, H. S., Yoon, E., Hasegawa-Johnson, M. A., Kim, S., and Yoo, C. D. ConfPO: Exploiting policy model confidence for critical token selection in preference optimization. In Forty-second International Conference on Machine Learning, 2025b. URL https://openreview. net/forum?id=ZG7bkp6ScT.   
Yoon, H. S., Yoon, E., Hong, J. W., Eom, S., Koo, G., Hasegawa-Johnson, M., Dai, Q., Luo, C., and Yoo, C. D. Pdcr: Perception-decomposed confidence reward for vision-language reasoning. arXiv preprint arXiv:2605.13467, 2026.   
Yue, X., Zheng, T., Ni, Y., Wang, Y., Zhang, K., Tong, S., Sun, Y., Yu, B., Zhang, G., Sun, H., Su, Y., Chen, W., and Neubig, G. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024.   
Zhang, D., Zhang, S., Yang, Z., Zhu, Z., Zhao, Z., Cao, R., Chen, L., and Yu, K. Progrm: Build better gui agents with progress rewards. arXiv preprint arXiv:2505.18121, 2025.   
Zhang, R., Jiang, D., Zhang, Y., Lin, H., Guo, Z., Qiu, P., Zhou, A., Lu, P., Chang, K.-W., Qiao, Y., et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pp. 169–186. Springer, 2024.

# A. Analysis of the Adaptive Scaling Factor $\eta _ { \mathbf { V G S } } ( \gamma )$

In this section, we provide the analytical derivation and empirical justification for approximating the adaptive scaling factor ηVGS(γ) as a constant.

Analytical Derivation. Recall the definition of the normalization factor from Eq. 15. By expanding the squared norm in the denominator, we can express ηVGS(γ) as a function of the gradient magnitudes and the angle ϕ between them:

$$
\begin{array}{l} \eta_ {\mathrm{VGS}} (\gamma) \triangleq \frac {\| \nabla \mathcal {L} _ {\text {Standard}} \| _ {2}}{\| \nabla \mathcal {L} _ {\text {Standard}} + \gamma \nabla \mathcal {L} _ {\text {Vis}} \| _ {2}} \tag {23} \\ = \frac {\left\| \nabla \mathcal {L} _ {\text { Standard }} \right\| _ {2}}{\sqrt {A + B + C}}, \\ \end{array}
$$

where:

$$
\begin{array}{l} A = \left\| \nabla \mathcal {L} _ {\text { Standard }} \right\| _ {2} ^ {2}, \\ B = \gamma^ {2} \| \nabla \mathcal {L} _ {\mathrm{Vis}} \| _ {2} ^ {2}, \\ C = 2 \gamma \| \nabla \mathcal {L} _ {\text { Standard }} \| _ {2} \| \nabla \mathcal {L} _ {\text { Vis }} \| _ {2} \cos \phi , \\ \end{array}
$$

and $\phi$ represents the angle between the standard gradient $ { \nabla } { \mathcal { L } } _ { \mathrm { { S t a n d a r d } } }$ and the visual gradient $\nabla \mathcal L _ { \mathrm { V i s } }$

Calculating Eq. 23 at every step incurs computational overhead due to the gradient norm computations. To mitigate this, we tracked the evolution of the constituent terms $( \| \nabla { \mathcal { L } } _ { \mathrm { S t a n d a r d } } \| , \| \nabla { \mathcal { L } } _ { \mathrm { V i s } } \|$ , and cos ϕ) throughout training. As illustrated in Figure 7, we observe that these values remain relatively constant across the training. Consequently, the resulting ratio ηVGS(γ) exhibits high stability across the training.

![](images/b11b5722bafae800a0f2ddcbc636e928c456778fb0d90ebc053ad3d4dd77bee2.jpg)  
Figure 7. Evolution of Gradient Norm Components. We track the magnitudes of the standard gradient (purple) and visual gradient (green), along with their cosine similarity (blue) for the Qwen3-VL-2B-Instruct student during standard on-policy distillation. All three components exhibit stability across training steps, justifying our approximation of the adaptive scaling factor $\eta _ { \mathrm { V G S } } ( \gamma )$ as a fixed constant to reduce computational overhead.

Based on this observation, we treat $\eta _ { \mathrm { V G S } } ( \gamma )$ as a fixed hyperparameter determined by the model architecture and $\gamma .$ . For our main experiments with $\gamma = 2 . 0$ , we utilize the pre-calculated average values:

• Qwen3-VL-2B-Instruct: $\eta _ { \mathrm { V G S } } = 0 . 4 1$   
• Qwen3-VL-4B-Instruct: $\eta _ { \mathrm { V G S } } = 0 . 3 6$

This approximation maintains the stability benefits of gradient normalization without the runtime cost.

# B. Implementation Details

# B.1. Teacher Model Training

To generate high-quality reasoning trajectories for distillation, we train a teacher model using Group Relative Policy Optimization (GRPO). We initialize the teacher with Qwen3-VL-8B-Instruct and fine-tune it on the Vision-SR1-47K dataset for 2 epochs. This reinforcement learning stage ensures the teacher provides robust, verifiable reasoning chains rather than just final answers. We perform all GRPO experiments using the TRL codebase. The specific hyperparameters for the teacher training phase are detailed in Table 3.

Table 3. Hyperparameters for Teacher Model Training (GRPO). The teacher model (Qwen3-VL-8B-Instruct) is trained for 2 epochs on Vision-SR1-47K. 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>1e-6</td></tr><tr><td>Weight Decay</td><td>1e-2</td></tr><tr><td>LR Schedule</td><td>Constant</td></tr><tr><td>Epochs</td><td>2</td></tr><tr><td>Global Batch Size</td><td>128</td></tr><tr><td>Freeze Vision Encoder</td><td>False</td></tr><tr><td colspan="2">RL / Rollout Config</td></tr><tr><td>Rollout Batch Size</td><td>512</td></tr><tr><td>Rollout Size (G)</td><td>8</td></tr><tr><td>Rollout Temperature</td><td>1.0</td></tr><tr><td>Rollout Top-p</td><td>0.99</td></tr><tr><td>Max Input Prompt Length</td><td>12800</td></tr><tr><td>Max Response Length</td><td>2048</td></tr><tr><td>Use KL Loss</td><td>False</td></tr></table>

# B.2. Training Framework and Hyperparameters

We perform all distillation experiments using the TRL codebase. To ensure a controlled environment, we initialize all student models (Qwen3-VL-2B-Instruct and Qwen3-VL-4B-Instruct) from their standard instruction-tuned checkpoints and train them to match the GRPO-tuned Qwen3-VL-8B-Instruct teacher. To ensure a strictly fair comparison, we maintain a consistent configuration across all baselines (Standard Monolithic Distillation, Visual Gradient Steering) and ablation settings. Common hyperparameters for the optimizer, generation (rollout), and model constraints are detailed in Table 4. All experiments were conducted on a single node equipped with 8 × NVIDIA A100 80GB GPUs.

Table 4. Key hyperparameters used for On-Policy Distillation training and evaluation. 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Optimization &amp; Training</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>1e-6</td></tr><tr><td>Weight Decay</td><td>1e-2</td></tr><tr><td>LR Schedule</td><td>Constant</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Global Batch Size</td><td>512</td></tr><tr><td>Freeze Vision Encoder</td><td>False</td></tr><tr><td colspan="2">On-Policy Rollout</td></tr><tr><td>Rollout Temperature</td><td>1.0</td></tr><tr><td>Rollout Top-p</td><td>1.0</td></tr><tr><td>Max Input Prompt Length</td><td>16384</td></tr><tr><td>Max Response Length</td><td>2048</td></tr><tr><td colspan="2">VGS Specific (Ours)</td></tr><tr><td>Steering Coefficient ( $\gamma$ )</td><td>2.0</td></tr><tr><td>Regularization Weight ( $\lambda$ )</td><td>0.01</td></tr></table>

![](images/71669d4a6f03a8e452d7a0dc29d01296f86eb049a5750da7146210536bd33d72.jpg)

<details>
<summary>bar</summary>

| γ    | Average Accuracy |
| ---- | ---------------- |
| 1.0  | 45.0             |
| 2.0  | 46.0             |
| 3.0  | 46.5             |
| 4.0  | 46.5             |
</details>

(a)y sensitivity

![](images/0df2b72c68b264d7cf0e9444dddbfe08cd6def610c131bd0ddb73f63843e31f2.jpg)

<details>
<summary>bar</summary>

| λ    | Average Accuracy |
| ---- | ---------------- |
| 0.0  | 56.8             |
| 0.1  | 57.3             |
| 0.01 | 58.2             |
</details>

(b)λ sensitivity   
Figure 8. Hyperparameter Sensitivity Analysis. We evaluate the impact of key hyperparameters on the average accuracy of the student model. (a) Varying the steering coefficient γ (on Qwen3-VL-2B-Instruct): The method shows robustness across a wide range of γ values, consistently outperforming the Standard On-Policy Distillation baseline (dashed grey line). (b) Varying the language preservation weight λ (on Qwen3-VL-4B-Instruct): A small non-zero regularization $( \mathrm { e . g . } , \lambda = 0 . 0 \bar { 1 } )$ yields the highest performance, confirming that the Language Preservation (LP) term effectively mitigates destructive interference without suppressing the visual signal.

# B.3. Sensitivity Analysis for Visual Gradient Steering (VGS) Specific Hyperparameters

We analyze the sensitivity of $ { \mathcal { L } } _ { \mathrm { V G S - L P } }$ (Eq. 17) to the steering coefficient γ and the language preservation weight λ (Figure 8). First, regarding γ (with λ fixed at 0.01), VGS consistently outperforms standard on-policy distillation across a broad range of values, demonstrating that the benefits of visual steering are robust to hyperparameter variations. Second, regarding λ (γ fixed at 2.0), results confirm that omitting the regularizer (λ = 0) is ineffective, as noted in Section 4, due to destructive gradient interference that causes the unlearning of the language prior in high-dependency regimes. However, assigning a small non-zero magnitude $( \mathbf { e . g . , } \lambda \in \{ 0 . 0 1 , 0 . 1 \} )$ ) is sufficient to effectively preserve the language prior and yield performance consistently superior to the standard baseline.

# B.4. Prompt Template for Training and Inference

To ensure structural alignment between the teacher and student policies, we utilize a unified system prompt across all training stages (Teacher GRPO, Student On-Policy Distillation, and VGS). This prompt enforces a strict Chain-of-Thought (CoT) format, requiring the model to explicitly delimit its reasoning process within <reason> and </reason> tags before generating the final answer. The standard template used is as follows:

# Reasoning Template for Training and Inference

# SYSTEM:

A conversation between user and assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process should be enclosed within <reason></reason>tags. The final answer MUST BE put in \boxed{}.

# USER:

$$
\begin{array}{l} \{\text {image} \} \\ \{\text {question} \} \end{array}
$$

![](images/3f9752d6fb53c1c03db8b3b88f75d3af8308cb5736cdaec883b3e8347b9b2d9c.jpg)

<details>
<summary>bar</summary>

| Method | Training Time (secs/step) |
| :--- | :--- |
| Standard On-Policy Distillation | 40 |
| VGS On-Policy Distillation (ours) | 58 |
x 1.375
</details>

Figure 9. Training Cost Comparison. VGS incurs a modest 1.375× increase in training time per step due to the dual forward passes required to isolate the visual gradient.

# C. Computational Cost Overhead

We analyze the computational efficiency of our approach in Figure 9. VGS introduces a moderate overhead, increasing the training time per step by approximately 1.375× compared to standard on-policy distillation. This increase stems from the necessity of an additional non-image-conditioned forward pass to calculate the decomposed visual gradient. We consider this a minimal and acceptable cost given the significant performance improvements yielded by the steering mechanism.

![](images/430e4aab6c83f12372c19a0c922d3f0343f507cef6b3d43ea44982947694160c.jpg)

<details>
<summary>bar</summary>

| Method | Average Accuracy |
|---|---|
| +γtangεLang(r) | 42.5 |
| εStandard(r) | 43.0 |
| +γεVis(r) | 49.5 |
| Language Steering Standard On-Policy Distillation | 46.0 |
| Visual Steering Standard On-Policy Distillation | 48.5 |
| γ | 47.0 |
| 1.0 | 49.0 |
| 2.0 | 48.5 |
| 3.0 | 49.0 |
</details>

(a) High-Visual Dependent Multimodal Benchmark

![](images/91d642cd2d30eedfc8d8ab4d3d4be4103e0a51e827362856242c193fda15399f.jpg)

<details>
<summary>bar</summary>

| Method | Average Accuracy |
|---|---|
| +γLang εLang(τ) | 57.5 |
| εStandard(τ) | 57.5 |
| +γεVis(τ) | 57.5 |
| Language Steering | 57.5 |
| Visual Steering | 57.5 |
| Standard On-Policy Distillation | 57.5 |
| γ | 57.5 |
| γLang | 57.5 |
</details>

(b) Low-Visual Dependent Multimodal Benchmark

![](images/51c27beeaf3f613c47206fe88046dbc3ed0e9301de0040c08ec98bce25b5a03b.jpg)

<details>
<summary>bar</summary>

| Method | Average Accuracy |
|---|---|
| +γ_lang·ℓ_Lang(τ) | 32.5 |
| ℓ_Standard(τ) | 32.0 |
| +γℓ_vis(τ) | 32.8 |
| Language Steering (γ_lang) | 33.0 |
| Visual Steering (γ Standard On-Policy Distillation) | 32.5 |
| Visual Steering (γ 1.0) | 32.8 |
| Visual Steering (γ 2.0) | 34.0 |
| Visual Steering (γ 3.0) | 32.7 |
</details>

(c) Text-only Benchmark   
Figure 10. Impact of optimization steering across varying levels of visual dependency. Average accuracy comparison between Language Steering, Standard On-Policy Distillation, and Visual Gradient Steering. (a) On High-Visual Dependent benchmarks, prioritizing the visual subspace (VGS) significantly outperforms the baseline, whereas leaning on the language prior actively degrades performance. (b) On Low-Visual Dependent and (c) pure Text-Only benchmarks, performance remains uniform across all methods, confirming that VGS yields gains where visual grounding is the primary bottleneck without compromising the model’s core textual reasoning capabilities.

# D. When is Visual Gradient Steering Most Effective?

To analyze when Visual Gradient Steering (VGS) yields the largest benefit, we evaluate our models across benchmarks categorized by their reliance on visual context (Wang et al., 2025). Figure 10 compares Language Steering, Standard On-Policy Distillation, and VGS as a function of the steering coefficient, grouped by visual dependency level.

• High Vision-Dependency (MMMU-Pro, LogicVista). A clear upward trend emerges (Language Steering < Standard OPD < VGS), where prioritizing the visual subspace yields consistent gains.   
• Low Vision-Dependency (Geo3K, We-Math). The performance difference across methods is minimal. When textual prompts contain sufficient information to deduce the answer, the optimization trajectory has minimal impact.   
• Text-Only (MATH500, AIME25, OlympiadBench). All methods yield uniform performance. This demonstrates that prioritizing the visual subspace during distillation does not necessary compromise the model’s core text-only reasoning capability.

In summary, VGS consistently outperforms standard OPD on tasks demanding strong multimodal grounding, without compromising performance on low-dependency or text-only tasks.

Table 5. Main Results on Vision-Language Reasoning Benchmarks. We compare the distillation performance of Visual Gradient Steering (VGS) against the standard monolithic baseline. All student models (2B and 4B) are distilled from the same Qwen3-VL-8B-Instruct teacher trained with GRPO. VGS consistently outperforms the standard approach across all benchmarks, achieving higher accuracy in greedy decoding (Acc@1). 

<table><tr><td rowspan="2">Benchmark</td><td>Teacher (GRPO trained)</td><td>Initial Student</td><td>Standard On-Policy Distillation</td><td>VGS On-Policy Distillation (ours)</td><td>Adaptive-VGS On-Policy Distillation (ours)</td></tr><tr><td>Acc@1 (greedy)</td><td>Acc@1 (greedy)</td><td>Acc@1 (greedy)</td><td>Acc@1 (greedy)</td><td>Acc@1 (greedy)</td></tr><tr><td></td><td>8B</td><td>2B</td><td colspan="3">Qwen3-VL-8B-Instruct → Qwen3-VL-2B-Instruct</td></tr><tr><td>MMMU-Pro-4</td><td>62.03</td><td>34.51</td><td>45.83</td><td>48.07</td><td>48.40</td></tr><tr><td>LogicVista</td><td>60.01</td><td>36.83</td><td>45.53</td><td>48.88</td><td>48.31</td></tr><tr><td>MathVerse-VD</td><td>79.63</td><td>35.88</td><td>56.02</td><td>58.10</td><td>59.72</td></tr><tr><td>MathVerse-VO</td><td>73.85</td><td>35.32</td><td>54.59</td><td>56.19</td><td>57.11</td></tr><tr><td>VisualPuzzles</td><td>43.15</td><td>13.36</td><td>28.08</td><td>30.64</td><td>33.99</td></tr><tr><td>MathVision</td><td>44.14</td><td>14.28</td><td>24.14</td><td>25.59</td><td>25.16</td></tr><tr><td>VlmsAreBlind</td><td>66.79</td><td>49.03</td><td>51.86</td><td>54.11</td><td>53.00</td></tr><tr><td>Average</td><td>61.37</td><td>31.32</td><td>43.74</td><td>46.10</td><td>46.53</td></tr></table>

# E. Adaptive Token-Level Visual Gradient Steering

Our main formulation applies a single, fixed steering coefficient γ across all tokens. A natural extension is to let the visual emphasis vary at the token level, scaling the steering strength according to how much each token relies on visual evidence. To this end, we explore a fully adaptive Visual Gradient Steering (VGS) objective that dynamically scales the steering coefficient $\gamma _ { t }$ based on each token’s Visual Dependency Score (VDSt, Eq. 12).

Formally, modifying our objective in Eq. 17, the adaptive objective is defined as:

$$
\ell_ {\text { Adaptive }} (\tau) \triangleq \frac {1}{| \tau |} \sum_ {t = 1} ^ {| \tau |} \left(\ell_ {\text { Standard }} (\tau) + \gamma_ {t} \ell_ {\text { Vis }} (\tau) + \lambda \ell_ {\text { LP }} (\tau)\right), \tag {24}
$$

where the token-level steering coefficient $\gamma _ { t }$ adjusts dynamically:

$$
\gamma_ {t} = \left\{ \begin{array}{l l} 0, & \mathrm{VDS} _ {t} \leq Q _ {0. 4} \quad (\text { no   visual   correction }), \\ \gamma / 2, & Q _ {0. 4} <   \mathrm{VDS} _ {t} \leq Q _ {0. 7} \quad (\text { moderate   correction }), \\ \gamma , & \mathrm{VDS} _ {t} > Q _ {0. 7} \quad (\text { maximum   visual   steering }), \end{array} \right. \tag {25}
$$

with $Q _ { 0 . 4 }$ and $Q _ { 0 . 7 }$ denoting the 40th and 70th percentile thresholds of the Visual Dependency Score distribution. Intuitively, this allocates the steering budget toward tokens that genuinely require visual grounding, while leaving low-dependency tokens unchanged.

As shown in Table 5, this adaptive formulation slightly outperforms our original fixed-scale approach $( \mathrm { i . e . , } \gamma = 2 . 0 )$ , achieving the highest overall average accuracy. This indicates that matching the steering strength to each token’s visual demand could provide a modest additional gain over applying a uniform coefficient.

# F. Qualitative Comparisons of Generated Reasoning

In this section, we present side-by-side comparisons between our proposed Visual Gradient Steering (VGS) and the standard monolithic on-policy distillation baseline. All examples presented were generated by Qwen3-VL-2B-Instruct distilled from the Qwen3-VL-8B-Instruct teacher. These examples highlight the core benefit of explicitly steering gradients toward the visual subspace. As observed in the figures below, the standard baseline often exhibits a specific failure mode where it generates logically coherent reasoning chains that are premised on incorrect visual extraction (highlighted in red). In contrast, VGS guides the student to accurately ground its reasoning in the visual input (highlighted in green).

# G. Limitations

While Visual Gradient Steering (VGS) significantly improves reasoning reliability, we acknowledge two primary limitations.

Training Throughput Overhead. The core mechanism of VGS requires decomposing the teacher’s output distribution, which necessitates an additional forward pass without visual inputs at each training step. As analyzed in Section C, this incurs a computational overhead of approximately 1.375× per step compared to standard distillation.

Dependence on Teacher Calibration. VGS operates on the premise that the teacher’s multimodal and unimodal distributions are distinguishable. It relies on the contrastive signal between the two to steer the student. Consequently, if the teacher model suffers from mode collapse where the visual input exerts no influence on the output distribution (i.e., $\nabla { \mathcal { L } } _ { \mathrm { V i s } } \approx 0 )$ , VGS effectively reduces to standard distillation. Our method enhances the transfer of visual grounding but cannot rectify fundamental perceptual blindness present in the teacher itself.

# Generated Sample 1 by Standard On-Policy Distillation

![](images/b17a4ea070a60f5f329fccc365d7ae99ee625468e5055a01c205e768d24ef2a4.jpg)

Question: Are the two circles touching each other? Answer is Yes/No

Ground truth: No

Generation:

<reason>

The two circles are positioned one above the other, with no visible gap between them. Since they are touching at a single point (the bottom circle touches the top circle at their overlapping edge), they are considered touching.

</reason> Yes

# Generated Sample 1 by VGS On-Policy Distillation (ours)

![](images/e34a48f9d82e455dc39c157348bf4a030abf77cc52c1f576eacc6254f60944c7.jpg)

Question: Are the two circles touching each other? Answer is Yes/No

Ground truth: No

Generation:

<reason>

The two circles in the image are positioned one above the other, but they do not touch each other. There is a clear space between them. Touching would mean that their edges meet at a point, but here they are separated by a gap.

</reason> No

# Generated Sample 2 by Standard On-Policy Distillation

![](images/b01fc496fe163fc2e9d390510a898620ea9285596bd7329fc14e01dda6e9b630.jpg)

Question: Count the pentagons in the image. Answer with a number in curly brackets e.g., \boxed{3}.

# Ground truth: 9

# Generation:

<reason>

The image shows a classic Olympic rings logo, which consists of five interlocking pentagons. Each pentagon is a distinct color: red, orange, blue, green, and purple. There are no other pentagons in the image.

Therefore, the total number of pentagons is 5.

</reason> 5

# Generated Sample 2 by VGS On-Policy Distillation (ours)

![](images/042a5c15ddac3589c34eca8f59e21767b1a4b98772a50a7ab788aa2afdc34fc4.jpg)

Question: Count the pentagons in the image. Answer with a number in curly brackets e.g., \boxed{3}.

# Ground truth: 9

# Generation:

<reason>

Count the pentagons in the image. The image shows a row of pentagons, each colored differently. Count them one by one:

1. Red pentagon   
2. Orange pentagon   
3. Blue pentagon   
4. Green pentagon   
5. Purple pentagon   
6. Gray pentagon   
7. Brown pentagon   
8. Pink pentagon   
9. Yellow pentagon

There are 9 pentagons in total.

</reason> 9

# Generated Sample 3 by Standard On-Policy Distillation

![](images/ad145afd392cc40426570730b03a0e7195fbedbe4bf03d6803cb49c20fa0f5f6.jpg)

<details>
<summary>text_image</summary>

1. DON
2. KUBAN
3. TEREK
4. ASTRAKHAN
5. URAL
6. ORENBURG
7. SEMRECHENIYE
8. SIBERIAN
9. TRANSBAIKAL
10. AMUR
11. USSURI
12. IRKUTSK
13. YAKUTSK
* Shown on Modern Map
Map of major Cossack hosts of the former Russian Empire
</details>

Question: Which of the following statements is most solidly supported by the information contained in the map shown in the image?

# Choices:

(a) Cossack communities devoutly adhered to a conservative strain of Russian Orthodox Christianity.   
(b) Cossack hosts frequently rebelled against the centralizing authority of the Russian Empire.   
(c) Cossack military units consist of highly skilled horse back warriors.   
(d) Cossack ethnic identity was shaped by intermarriage with non-Russian peoples on the empire’s frontiers.

# (d)

# Generation:

<reason>

The map shown is a Map of major Cossack hosts of the former Russian Empire. ¨ ¨It highlights various regions where Cossack communities were centered. However, the map does not provide any information about:

- Religious practices (A) – The map does not indicate any religious affiliations or beliefs.   
- Rebellion against central authority (B) – The map does not show any evidence of rebellion or resistance.   
- Military skills (C) – The map does not provide any information about the military capabilities or training of Cossack units.   
- Ethnic identity (D) – The map does not indicate any information about intermarriage or ethnic mixing.

Since the map only shows the locations of Cossack hosts, and none of the options can be directly supported by the map’s content, we must choose the option that is most likely to be true based on general historical knowledge of the Cossacks.

The Cossacks were a semi-nomadic ethnic group that played a significant role in the Russian Empire’s military and political history. They were known for their martial prowess and were often involved in frontier conflicts. While the map does not

provide direct evidence, option C (Cossack military units consisted of highly skilled horseback warriors ¨ ¨) is a well-documented fact about the Cossacks, and it is the most solidly supported by general historical knowledge, even though it is not directly stated on the map.

Therefore, the best answer is C. </reason> (c)

# Generated Sample 3 by VGS On-Policy Distillation (ours)

![](images/4990f62c2ca31e09e24ad7d02aacf6afa2bcd339ce8428e2b03f6902465a8ce3.jpg)

<details>
<summary>text_image</summary>

1. DON
2. KUBAN
3. TEREK
4. ASTRAKHAN
5. URAI
6. ORENBURG
7. SEMIRECHENIYE
8. SIBERIAN
9. TRANSBAIKAL
10. AMUR
11. USSURI
12. IRKUTSK
13. YAKUTSK
* Shown on Modern Map
Map of major Cossack hosts of the former Russian Empire
</details>

Question: Which of the following statements is most solidly supported by the information contained in the map shown in the image?

# Choices:

(a) Cossack communities devoutly adhered to a conservative strain of Russian Orthodox Christianity.   
(b) Cossack hosts frequently rebelled against the centralizing authority of the Russian Empire.   
(c) Cossack military units consist of highly skilled horse back warriors.   
(d) Cossack ethnic identity was shaped by intermarriage with non-Russian peoples on the empire’s frontiers.

# (d)

# Generation:

<reason>

The map shown is a Map of major Cossack hosts of the former Russian Empire ¨ ¨. It highlights various regions where Cossack communities were centered, such as the Don, Kuban, Terek, Ural, Orenburg, Semirechensky, Siberian, Transbaikal, Amur, Ussuri, Irkutsk, and Yakutsk regions.

- Option A: The map does not provide any information about religious adherence or beliefs of the Cossacks. This is not supported by the map.   
- Option B: The map does not provide any information about rebellions or resistance against central authority. This is not supported by the map.   
- Option C: The map does not provide any information about the military skills or composition of Cossack units. This is not supported by the map.   
- Option D: The map shows Cossack hosts in regions that were historically frontier areas, including areas with significant non-Russian populations (e.g., Siberia, Transbaikalia, Ural). This suggests that Cossack ethnic identity was shaped by intermarriage with non-Russian peoples on the empire’s frontiers. This is a plausible inference based on historical context, and

the map supports the idea that Cossacks were often located in frontier regions where they interacted with other ethnic groups.

Therefore, the most solidly supported statement by the information contained in the map is D. </reason> (d)

# Generated Sample 4 by Standard On-Policy Distillation

![](images/dada5e90687352a7f13fb785f6fef9b62874a451cb3b39685a65011f80802038.jpg)

<details>
<summary>natural_image</summary>

Chest X-ray image showing lung fields and thoracic structures (no text or annotations)
</details>

Question: What condition does this patient have?

Choices: (a) Broken rib

(b) Situs inversus

(c) Pleueral effusion

(d) Dislocated shoulder

(b)

# Generation:

<reason>

The image shows a chest X-ray with a clear view of the lungs, heart, and ribs. There is no visible fracture line or deformity in the ribs, and the shoulder joints appear normal. The heart is in the correct position, and there is no evidence of pleural fluid accumulation (which would appear as a blunting of the costophrenic angles or a hazy ¨ appearance). The shoulder joints are also ¨ intact.

Situs inversus is a condition where the organs are mirrored across the body (e.g., left heart on the right side), which is not visible in this X-ray. A broken rib would show a fracture line, which is not present. Dislocated shoulder would show a visible dislocation, which is not present. Pleural effusion would show a blunting of the costophrenic angles or a hazy ¨ appearance, ¨ which is not present here.

Given the options, the most likely condition is \*\*pleural effusion\*\*, as it is a common finding in chest X-rays and can appear as a hazy ¨ or¨ blunted ¨ appearance, but in this case, the image shows a clear view of the lungs and no signs of effusion. ¨

Wait, let me recheck. The image shows a clear view of the lungs, and there is no evidence of pleural effusion. The costophrenic angles are clear, and there is no fluid accumulation. Therefore, none of the options seem to fit perfectly.

But looking at the options again, \*\*pleural effusion\*\* is the most likely answer because it is a common condition that can be seen on chest X-rays and is often associated with symptoms like shortness of breath or cough. The image shows a normal chest cavity, so pleural effusion is not the best fit.

Actually, the image shows a normal chest X-ray, so none of the options are correct. But since we must choose,

\*\*pleural effusion\*\* is the most likely option among the given choices.

Therefore, the best answer is C. </reason> (c)

# Generated Sample 4 by VGS On-Policy Distillation (ours)

![](images/2e0cbd56798f0f853fc0043d29d1ce13b3176d3e2c90e3228a1c00237b7dbf75.jpg)

<details>
<summary>natural_image</summary>

Chest X-ray image showing lung fields and thoracic structures (no text or annotations)
</details>

Question: What condition does this patient have?

Choices: (a) Broken rib

(b) Situs inversus

(c) Pleueral effusion

(d) Dislocated shoulder

(b)

# Generation:

<reason>

The image shows a chest X-ray with a clear view of the lungs, ribs, and shoulders. There is no visible fracture line or deformity in the ribs that would suggest a broken rib. The shoulders appear normal, and there is no evidence of dislocation. The lungs appear clear, and there is no sign of pleural effusion (such as fluid accumulation in the pleural space, which would appear as a blunting of the costophrenic angles or a hazy ¨ appearance). ¨ The most likely condition is situs inversus, which is a congenital condition where the organs are in the opposite positions to normal. In situs inversus, the heart is in the opposite position, and the lungs are reversed. This is a common condition and is often not associated with symptoms. The image shows a normal chest X-ray with no signs of the other options. </reason> (b)