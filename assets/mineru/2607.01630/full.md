# DRDN: Decoupled Representation Dynamic Network for From-Scratch ViT Class-Incremental Learning

Bingchen Huang, Yifu Chen, Zhiling Wang, and Yuanchao Du

Abstract—Dynamic expansion methods for class-incremental learning (CIL) protect task-specific knowledge by growing dedicated tokens or subnetworks, yet our analyses suggest that classification supervision alone does not sufficiently preserve taskagnostic shared backbone representations over long incremental sequences. We identify two intertwined challenges: cross-task confusion from sequential training on predominantly current-task data, which biases decision boundaries toward recent tasks; and under-optimized shared representations in the backbone that cap long-term discriminability as tasks accumulate.

We propose the Decoupled Representation Dynamic Network (DRDN), which addresses these challenges via two orthogonal mechanisms. For shared backbone representations, DRDN continuously applies masked image modeling (MIM) at every incremental step, with reconstruction gradients routed exclusively through the backbone, encouraging it to retain general visual structure beyond class-discriminative cues. For task-specific discrimination, DRDN employs hierarchical task token expansion across all transformer layers, with a modified per-task attention rule that reduces inter-task interference. We support this design with accuracy degradation analysis and cross-task confusion rate measurements.

In the from-scratch ViT CIL setting (no external pretraining), DRDN consistently improves over strong token-expansion baselines with comparable backbone scale. On CIFAR100-B0 (10 steps), DRDN achieves 77.19% average accuracy, outperforming DKT by 1.36 points and DyTox by 3.53 points, with an advantage that grows at longer incremental sequences. Multi-seed validation confirms stability (±0.31%). The MIM decoder is active only during training, adding no inference-time parameters or computation.

Index Terms—Class-incremental learning, catastrophic forgetting, masked image modeling, dynamic expansion, vision transformers.

## I. INTRODUCTION

N class-incremental learning (CIL), a model must learn new classes sequentially while retaining knowledge of all previously seen classes — a challenge where standard gradient descent causes catastrophic forgetting [1], [2]. Dynamic expansion methods [3]–[5] currently achieve leading performance by growing dedicated task-specific components — tokens or subnetworks — at each step. While these components protect task-specific knowledge, they leave a critical question unaddressed: what happens to the shared backbone as tasks accumulate?

We observe that in dominant ViT-based token-expansion methods (DyTox [4], DKT [5]), the shared backbone is trained exclusively through classification objectives that favor task specific discriminability. This concentration of supervision on task-specific modules creates two intertwined challenges, which we diagnose and validate empirically via accuracy degradation analysis and cross-task confusion analysis (Section IV-D).

Challenge 1 (Cross-task confusion). Sequential training on predominantly current-task data biases decision boundaries toward recent tasks. During joint inference over all seen classes, semantically similar classes from different tasks are easily confused. As shown in Fig. 1, within a single task (left, right), DyTox class clusters are reasonably separated; when two tasks are plotted together (center), inter-task class boundaries collapse — different-task classes overlap severely. While prior work has addressed task confusion in ResNet-based expansion methods [6], [7], we show it persists in ViT token-expansion methods and measure it directly: 90.4% of DyTox classification errors are cross-task misclassifications (Section IV-D).

Challenge 2 (Under-optimized shared representations). Shallow ViT layers encode transferable visual structure [8] that benefits all tasks, yet discriminative training pushes these layers toward task-specific features. As shown in Fig. 2, models trained on limited incremental task data develop weaker shallow-layer representations (less precise activation maps on structural cues) compared to models exposed to broader data streams. This manifests as accelerating accuracy degradation: DyTox loses 34.8 absolute points over 10 incremental steps, while DRDN loses only 29.5 points over the same sequence.

These two challenges are potentially linked: weaker shared representations reduce the feature quality that task tokens operate on, which may amplify confusion; and cross-task gradient interference during token learning may in turn pollute shared backbone features. Our analyses suggest both effects are present and that addressing only one leaves substantial room for improvement.

As a preview of our approach’s effect, Fig. 3 shows that DRDN exhibits significantly stronger anti-forgetting capability than state-of-the-art dynamic expansion methods across all tasks.

The core design principle of DRDN is coordinated decomposition of optimization: the shared backbone receives a task-agnostic reconstruction signal exclusively, while taskspecific modules receive only classification signals. This is not simply adding MIM as an auxiliary objective — it is a deliberate assignment of responsibility: the backbone is the locus of general representations, task tokens are the locus of task-specific discrimination, and the gradient pathways enforce this assignment structurally.

We propose the Decoupled Representation Dynamic Network (DRDN), with three targeted ingredients:

![](images/033d22adf638378ef176092c6fd618ceb4a8cfe5db526d6bd644025181d5cdc2.jpg)  
(a) Task 0 only (classes 0–9)

![](images/d5101b1d399e99229a2e6da7cce964729f29c56d94eed725fac4038faa47daa4.jpg)  
(b) Task 1 only (classes 10–19)

![](images/57eb0e17f22a9bb2264069b6e885e175a895238645ef5ba7140a20e449a388f2.jpg)  
(c) Tasks 0+1 combined

Fig. 1. t-SNE visualization of DyTox task-token features on CIFAR100-B0 (10 steps) after all 10 tasks are trained. Large markers show per-class centroids; shading within each color family distinguishes individual classes. Within a single task (a, b), class centroids are reasonably spread. When both tasks are overlaid (c), red (Task 0) and blue (Task 1) centroids intermix — confirming that cross-task confusion is the dominant error mode (90.4% of errors).  
![](images/dab9fce5998992b76128ed81da4a114d802a328fa15979809b03d80fd9b238ab.jpg)  
Fig. 2. Grad-CAM visualization of shallow-layer activations. Models trained on limited incremental data (left) show diffuse, task-specific activations, while models trained on broader data (right) develop sharp, structurally-grounded activations — supporting the claim that CIL training under-optimizes shared backbone representations.

![](images/4c864110cd71d29ae45423a9e4d9f76c8aaa27f04c24cd4f86b4f6372121cb1b.jpg)  
Fig. 3. CIL performance on CIFAR100 (10-step, B0). For each task, 10 new classes are learned while previous classes must not be forgotten. DRDN (blue) is state-of-the-art with only a small parameter overhead, and its advantage expands as tasks accumulate.

1) Continual MIM with backbone-only gradient routing (for Challenge 2): A masked image reconstruction branch runs in parallel at every incremental step. Critically, reconstruction gradients are routed exclusively through the backbone — never through task tokens — so that the backbone’s general visual structure is maintained independently of task-specific adaptation. Adding MIM with shared gradients (flowing to both backbone and task modules) does not achieve the same benefit, as we verify in the ablation study (Table VII).

2) Hierarchical Task Token Expansion with isolated attention (for Challenge 1): Task tokens are expanded at all ViT layers, enabling multi-granularity task-specific features. A modified per-task attention rule excludes other tasks’ tokens from each task’s context, reducing cross-task gradient interference at the structural level.

3) Coordinated gradient pathway routing: $\mathcal { L } _ { \mathrm { r e c } }$ updates backbone only; $\mathcal { L } _ { \mathrm { c l s } } / \mathcal { L } _ { \mathrm { k d } } / \mathcal { L } _ { \mathrm { d i v } }$ update backbone and current task modules; old task tokens are frozen. This explicit assignment ensures each objective optimizes exactly the right components.

We target the from-scratch ViT CIL regime — no pretrained weights, no external data — representing scenarios where large-scale pretraining is unavailable or inappropriate [9]. All comparisons are within this regime for strict fairness.

Our contributions are summarized as follows:

• We propose a CIL framework whose key novelty is online masked image modeling within the CIL training loop with explicit gradient pathway decoupling, combined with hierarchical multi-layer per-task token expansion.

• We directly quantify both motivating challenges via accuracy degradation analysis (shared representation quality) and cross-task confusion-rate analysis, confirming the diagnosis that underpins DRDN’s design.

• Within the from-scratch ViT CIL regime, DRDN consistently improves over DyTox and DKT across six benchmarks, with growing advantages at longer incremental sequences (+3.76 points over DyTox at 20 steps on CIFAR100-B0), stable results across three seeds, and zero inference-time overhead.

## II. RELATED WORK

## A. Dynamic Expansion and Task Confusion in CIL

Dynamic expansion methods [10]–[12] grow task-specific components at each step. DER [3] directly expands the backbone per task at the cost of linear parameter growth. DyTox [4] and DKT [5] expand lightweight task tokens in a shared ViT backbone, achieving competitive accuracy with few additional parameters. Dense Network Expansion [13] combines dense backbone expansion with exemplar replay.

Cross-task confusion in expansion methods has been studied explicitly: Huang et al. [6] resolve it in ResNet-based dynamic networks through token-level discrimination objectives; Wang et al. [7] address bi-compatible confusion via energybased expansion and fusion. DRDN targets the same confusion problem within the ViT token-expansion paradigm, but embeds the solution within a wider decoupled-representation framework that also addresses shared representation quality.

## B. Decoupled Objectives and Stability–Plasticity

Liang and Li [14] decouple stability and plasticity losses in task-agnostic continual learning, showing that dedicated unsupervised signals improve backbone generalization. Kim and Han [15] study the stability-plasticity trade-off and advocate dynamic management of these competing objectives. Zhai et al. [16] demonstrate that masked autoencoder pretraining transfers well to CIL, using offline MAE-pretrained ViT backbones as fixed encoders. Masked image modeling [17], [18] and its fine-tuning extensions [19], [20] show that MIM objectives can be applied beyond pretraining. DRDN differs from all of the above by applying MIM online during each incremental step from a randomly initialized backbone.

## C. Replay and Regularization Methods

Replay-based methods [21]–[25] maintain a small exemplar buffer and interleave stored samples during training. iCaRL [21] selects herding exemplars and uses nearest-mean-of-exemplars at test time. BiC [24] and WA [25] add bias-correction layers to re-balance old/new class boundaries after each task. Rainbow Memory [22] diversifies exemplar selection via data augmentation variance. Generative replay methods [26]–[28] synthesize pseudo-exemplars via generative models, avoiding explicit storage; recent diffusion-based variants [27], [28] substantially improve sample quality. Regularization-based methods [29]–[32] penalize changes to parameters important for prior tasks. EWC [29] estimates parameter importance via the Fisher information matrix; SI [30] accumulates per-parameter path integrals online. These approaches are orthogonal to dynamic expansion and complementary to DRDN’s backbonelevel regularization via MIM.

## D. Prompt- and Adapter-based CIL

L2P [33], DualPrompt [34], and related methods exploit large pretrained ViT backbones through task-adaptive input prompts. These approaches perform well in the pretrained regime but degrade significantly without pretraining [9]. Our work targets the from-scratch regime where backbone quality must be built incrementally.

## E. Continual Self-Supervised Learning

CaSSLe [35] and Lump [36] study how SSL representations interact with continual learning, showing that SSL pretraining provides anti-forgetting properties. Scale [37] extends this to streaming settings. These works focus on continual pretraining with frozen or distilled representations. DRDN differs fundamentally: we apply MIM as an auxiliary regularizer within supervised CIL, with its gradient structurally isolated to the backbone, providing complementary optimization pressure that classification alone cannot supply.

DRDN differs from Huang et al. [6] in where crosstask interference is reduced (attention routing vs. token-level discrimination); from [14] in what is decoupled (backbone-only MIM gradient vs. task-level stability/plasticity); from [16] in how MIM is used (online vs. offline pretraining); from DKT in expansion granularity (all layers vs. one); and from continual SSL [35], [36] in the role of SSL (auxiliary regularizer vs. standalone pretraining).

## III. METHOD

## A. Problem Formulation

We consider image classification with a dataset $\begin{array} { r l } { \mathcal { D } } & { { } = } \end{array}$ $\{ ( x , y ) \} ^ { n }$ . In CIL, D is partitioned into T disjoint subsets with disjoint label sets $\boldsymbol { Y } ^ { 1 } , \ldots , \boldsymbol { Y } ^ { T }$ . The model processes tasks sequentially. We maintain a fixed-size exemplar buffer B of capacity n<sub>B</sub> storing equal-per-class representative samples from all seen tasks.

## B. DRDN Architecture Overview

DRDN has three core ingredients: a shared encoder trained with an auxiliary reconstruction pathway that routes gradients exclusively through the backbone; per-task tokens inserted at every transformer layer; and a task-token attention rule that excludes other tasks from each task’s context. Together: MIM → shared backbone representations (Challenge 2); hierarchical tokens + modified attention → task-specific discrimination (Challenge 1); KD + diversity loss → anti-forgetting and token differentiation.

Fig. 4 illustrates DRDN. The backbone consists of k Modified Self-Attention Blocks (MSABs). An input image is tokenized via patch and positional embeddings into $\mathbf { x } _ { 0 } ^ { \mathrm { { i m g } } } \in$ $\mathbb { R } ^ { N \times D }$ . A masked version ${ \bf x } _ { 0 } ^ { \mathrm { r e c } }$ (mask ratio 0.75) is maintained in parallel for reconstruction. For task t, DRDN maintains learnable task tokens at each layer i:

$$
\mathbf {x} _ {i} ^ {\mathrm{task}} = [ \mathbf {e} _ {i, 1} ^ {\mathrm{task}}, \ldots , \mathbf {e} _ {i, t} ^ {\mathrm{task}} ] \in \mathbb {R} ^ {t \times D}.\tag{1}
$$

The final MSAB output produces classification representation $\mathbf { x } ^ { \mathrm { c l s } }$ from task token outputs $\mathbf { h } _ { k , j }$ , and reconstruction representation ${ \bf x } _ { k } ^ { \mathrm { r e c } }$ from the masked image tokens. The classification prediction for task j is: $\mathbf { y } _ { j } = \mathrm { c l f } _ { j } ( \mathbf { x } ^ { \mathrm { c l s } } )$

## C. Reconstruction Branch and Loss

The decoder takes ${ \bf x } _ { k } ^ { \mathrm { r e c } }$ plus learnable mask tokens, and outputs a vector prediction for each masked patch. The reconstruction target for each masked patch $p \in \Omega$ is the flattened RGB patch vector after per-patch mean–variance normalization, following He et al. [17]:

![](images/a43b9b735b85b6b10b8eb2543a72b2ff8b291dedc1ca0fa1adfa09105b7e7479.jpg)  
Fig. 4. DRDN framework. The backbone consists of multiple Modified Self-Attention Blocks (MSABs) and branches into two paths. The upper (classification) branch expands task-specific tokens and classifiers at every MSAB layer when new tasks arrive; non-current task modules (blue background) are frozen. The lower (reconstruction) branch — one standard self-attention decoder block — is active only during training, guiding the backbone to focus on shared visual representations via masked image reconstruction. Reconstruction gradients flow only through the backbone (dashed arrows).

TABLE I  
GRADIENT ROUTING IN DRDN. CURRENT MODULES = CURRENT TASK TOKENS + CURRENT CLASSIFIER.

<table><tr><td>Loss</td><td>Backbone (MSAB)</td><td>Current Modules</td></tr><tr><td> $\mathcal{L}_{\text{rec}}$ </td><td>√</td><td></td></tr><tr><td> $\mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{kd}} + \mathcal{L}_{\text{div}}$ </td><td>√</td><td>√</td></tr><tr><td>Old task tokens</td><td colspan="2">Frozen (no gradient)</td></tr></table>

$$
\mathcal {L} _ {\mathrm{rec}} = \frac {1}{| \Omega |} \sum_ {p \in \Omega} \big \| \mathrm{decoder} (\mathbf {x} _ {k} ^ {\mathrm{rec}}) _ {p} - \hat {\mathbf {x}} _ {p} ^ {\mathrm{img}} \big \| ^ {2},\tag{2}
$$

where $\hat { \mathbf { x } } _ { p } ^ { \mathrm { i m g } } \in \mathbb { R } ^ { 3 \times P ^ { 2 } }$ is the normalized pixel vector of patch $p .$

Table I summarizes gradient routing. The reconstruction loss updates only the backbone; old task tokens are always frozen.

## D. Total Objective

$$
\mathcal {L} = \mathcal {L} _ {\mathrm{cls}} + \lambda \mathcal {L} _ {\mathrm{rec}} + \alpha \mathcal {L} _ {\mathrm{kd}} + \beta \mathcal {L} _ {\mathrm{div}},\tag{3}
$$

where $\lambda = 1 , \alpha = 1 , \beta = 1$ in all experiments.

Knowledge Distillation. To preserve old-class knowledge during task t, we apply KL-divergence distillation over oldclass logits $\mathbf { z } ^ { t - 1 } ( \mathbf { x } )$ and $\mathbf { z } ^ { t } ( \mathbf { x } )$ with temperature $\tau = 2 \cdot$

$$
\mathcal {L} _ {\mathrm{kd}} = \sum_ {\ell \in Y ^ {1: t - 1}} p _ {\ell} ^ {t - 1} (\mathbf {x}; \tau) \cdot \log \left(\frac {p _ {\ell} ^ {t - 1} (\mathbf {x} ; \tau)}{p _ {\ell} ^ {t} (\mathbf {x} ; \tau)}\right),\tag{4}
$$

where ℓ indexes old classes and $p ^ { s } ( \mathbf { x } ; \tau ) = \mathrm { s o f t m a x } ( \mathbf { z } ^ { s } ( \mathbf { x } ) / \tau )$

Diversity Loss. The diversity loss encourages each task token to be distinct and contribute complementary features. We attach a lightweight auxiliary classifier $\operatorname { A u x C l f } _ { j }$ to each task token’s output $\mathbf { h } _ { k , j }$ that predicts over all current new classes $Y ^ { t }$ plus a single aggregated “old-class” logit. The diversity loss is:

$$
\mathcal {L} _ {\mathrm{div}} = - \frac {1}{t} \sum_ {j = 1} ^ {t} \log \frac {\exp (\operatorname{AuxClf} _ {j} (\mathbf {h} _ {k , j}) _ {y})}{\sum_ {c} \exp (\operatorname{AuxClf} _ {j} (\mathbf {h} _ {k , j}) _ {c})},\tag{5}
$$

where y denotes the ground-truth label aggregated to the auxiliary prediction space. All auxiliary classifiers AuxClf<sub>j</sub> are discarded at test time.

Decoder Architecture. The MIM decoder is a single lightweight transformer block (standard multi-head selfattention + FFN, embedding dim D), shared across all incremental steps and discarded after training. It takes as input ${ \bf x } _ { k } ^ { \mathrm { r e c } }$ concatenated with |Ω| learnable mask tokens (one per masked patch), and outputs a $3 P ^ { 2 }$ -dimensional vector per patch for per-patch mean-variance-normalized pixel reconstruction. The decoder contains approximately 1.2M parameters during training and is entirely absent at test time.

## E. Representation Decoupling Design (MSAB)

Fig. 5 shows the MSAB. Within each MSAB, computation follows two parallel paths.

Task-Specific Path. For task token $j$ at layer i:

$$
\begin{array}{l} \mathbf {Q} _ {i, j} ^ {\text {task}} = \mathbf {e} _ {i, j} ^ {\text {task}} \mathbf {W} _ {Q}, \\ \mathbf {K} _ {i, j} ^ {\text {task}} = [ \mathbf {e} _ {i, j} ^ {\text {task}}, \mathbf {x} _ {i} ^ {\text {img}} ] \mathbf {W} _ {K}, \quad \mathbf {V} _ {i, j} ^ {\text {task}} = [ \mathbf {e} _ {i, j} ^ {\text {task}}, \mathbf {x} _ {i} ^ {\text {img}} ] \mathbf {W} _ {V}. \end{array} \tag {6}\tag{6}
$$

(7)

$$
\begin{array}{c} \mathbf {h} _ {i, j} = \text {FFN} \Bigg (\text {softmax} \Bigg (\frac {\mathbf {Q} _ {i , j} ^ {\text {task}} \left(\mathbf {K} _ {i , j} ^ {\text {task}}\right) ^ {\top}}{\sqrt {d _ {h}}} \Bigg) \\ \cdot \left. \mathbf {V} _ {i, j} ^ {\text {task}} \mathbf {W} _ {O} + \mathbf {b} _ {O}\right), \end{array}\tag{8}
$$

![](images/b1f8a129d9efa60ea04bc6a71645f5ebc611a7a8264df18b41a57703d7b8d511.jpg)  
Fig. 5. Modified Self-Attention Block (MSAB). Left (task-specific path): each task token j attends only to itself and image tokens — never to other tasks tokens — generating output $\mathbf { h } _ { i , j }$ for classification. Right (reconstruction path): image tokens $\mathbf { x } _ { i } ^ { \mathrm { { i m g } } }$ and masked tokens ${ \bf x } _ { i } ^ { \mathrm { r e c } }$ are processed by standard self attention independently. Reconstruction gradients (dashed) flow only through the backbone.

where $d _ { h }$ is the dimension of each attention head. Each task token attends only to itself and image tokens — never to other tasks’ tokens — preventing cross-task gradient interference.

Foundational Representation Path. Image tokens and masked tokens follow standard self-attention independently:

$$
\mathbf {x} _ {i + 1} ^ {\mathrm{img}} = \mathrm{FFN} (\mathrm{selfattn} (\mathbf {x} _ {i} ^ {\mathrm{img}})),\tag{9}
$$

$$
\mathbf {x} _ {i + 1} ^ {\mathrm{rec}} = \mathrm{FFN} (\mathrm{selfattn} (\mathbf {x} _ {i} ^ {\mathrm{rec}})).\tag{10}
$$

Note that ${ \bf x } _ { i } ^ { \mathrm { r e c } }$ is isolated from $\mathbf { x } _ { i } ^ { \mathrm { { i m g } } }$ throughout: after the first MSAB, $\mathbf { x } _ { i } ^ { \mathrm { { i m g } } }$ encodes cross-patch context (including task specific cues), making it unsuitable for reconstruction.

## F. Training Procedure

Algorithm 1 summarizes DRDN’s training loop. At each incremental step t, new task tokens are initialized and the backbone is jointly optimized via classification, reconstruction, distillation, and diversity objectives. Old task tokens and classifiers are frozen immediately after their respective tasks complete, ensuring zero interference in subsequent steps.

## G. Hierarchical Task Token Expansion

Unlike DyTox [4] which expands task tokens only at the final transformer layer, DRDN expands tokens at all k MSAB layers. Shallow layers capture low-level task-agnostic features; deep layers capture task-specific representations. Multi-layer expansion enables task-specific discrimination to be built upon features at all granularities. The final classification representation aggregates all layers:

$$
\mathbf {h} _ {j} = [ \mathbf {h} _ {0, j}, \dots , \mathbf {h} _ {k, j} ],\tag{11}
$$

$$
\operatorname{logits} _ {j} = \operatorname{Clf} _ {j} (\operatorname{Fusion} (\mathbf {h} _ {j})),\tag{12}
$$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 DRDN Incremental Training
Require: Tasks  $D^{1}, \ldots, D^{T}$ ; buffer B; backbone  $f_{\theta}$ ; decoder  $g_{\phi}$ 
1: for t = 1 to T do
2: Initialize task token  $\{e_{i,t}^{task}\}_{i=1}^{k}$  and classifier  $Clf_{t}$ 
3: Copy old model  $f_{\theta^{t-1}}$  for KD
4: for each mini-batch from  $D^{t} \cup B$  do
5: Forward classification branch: compute  $L_{cls}$ ,  $L_{kd}$ ,  $L_{div}$ 
6: Forward reconstruction branch with mask ratio 0.75: compute  $L_{rec}$ 
7:  $L \leftarrow L_{cls} + \lambda L_{rec} + \alpha L_{kd} + \beta L_{div}$ 
8: Update  $f_{\theta}$  (all losses) and  $g_{\phi}$  ( $L_{rec}$  only)
9: Update  $e_{i,t}^{task}$ ,  $Clf_{t}$  (classification losses only)
10: end for
11: Freeze  $\{e_{i,t}^{task}\}_{i=1}^{k}$  and  $Clf_{t}$ 
12: Update exemplar buffer B (herding selection)
13: end for
Ensure: Discard decoder  $g_{\phi}$ ; return  $f_{\theta}$  with all task tokens and classifiers
</div>

where Fusion $\begin{array} { r } { ( { \bf h } _ { j } ) = \sum _ { i = 0 } ^ { k } w _ { i } { \bf h } _ { i , j } } \end{array}$ uses task-shared learned scalar weights $w _ { i }$ (summing to 1).

## H. Design Rationale for Backbone-Only Gradient Routing

We provide intuition for why reconstruction gradients should be isolated to the backbone. Consider the backbone encoding $Z = f _ { \theta } ( X )$ : we want $Z$ to maximize general visual structure $I ( Z ; X )$ , while task tokens maximize task-specific discrimination $I ( H _ { j } ; Y ^ { t } | Z )$ . When MIM gradients flow through both, task tokens partially encode spatial structure to reduce ${ \mathcal { L } } _ { \mathrm { r e c } } ,$ diluting their discriminative specialization. Isolating $\nabla \mathcal { L } _ { \mathrm { r e c } }$ to the backbone enforces a clean decomposition: the backbone retains general structure, task tokens focus exclusively on discrimination. Our ablation validates this: MIM with shared gradients yields only 74.82%, 2.37 points below backbone-only routing (77.19%), despite identical compute.

## I. Multimedia Relevance

Class-incremental learning is a core challenge in multimedia systems that continuously encounter new visual categories (e.g., e-commerce visual search, content moderation, video surveillance). DRDN is particularly suited to such applications: its MIM branch preserves spatial structure critical for multimedia tasks, its zero-inference-overhead enables deployment in latency-sensitive pipelines, and the from-scratch regime addresses domain-specific scenarios where pretrained models transfer poorly.

## IV. EXPERIMENTS

## A. Setup

Benchmarks. CIFAR100 [38], ImageNet100, and ImageNet1000 [39]. CIFAR100-B0: train all 100 classes from scratch in 5, 10, or 20 steps. CIFAR100-B50: 50-class base, then 2, 5, or 10 steps. ImageNet100/1000: 10-step B0. Replay buffer: $n _ { B } ~ = ~ 2 , 0 0 0$ (CIFAR100, ImageNet100) or 20,000 (ImageNet1000).

Metrics. Average Accuracy (Avg) across all incremental steps, Final Accuracy (Last), Parameter Count (Params in M), and Backward Transfer (BWT). BWT is defined as $\begin{array} { r } { \frac { 1 } { T - 1 } \sum _ { i = 1 } ^ { T - 1 } ( A _ { T , i } - A _ { i , i } ) } \end{array}$ , where $A _ { t , i }$ is accuracy on task i after training on task t.

TABLE II  
RESULTS ON CIFAR100-B0 IN THE FROM-SCRATCH VIT REGIME (NO PRETRAINED WEIGHTS, ∼11M PARAMS). BOUND = JOINT-TRAINING UPPER BOUND. BWT: LESS NEGATIVE IS BETTER; DRDN ACHIEVES BEST BWT IN ALL SETTINGS. PARAMS IN MILLIONS.

<table><tr><td rowspan="2">Methods</td><td colspan="4">5 steps</td><td colspan="4">10 steps</td><td colspan="4">20 steps</td></tr><tr><td>Par.</td><td>Avg</td><td>Last</td><td>BWT</td><td>Par.</td><td>Avg</td><td>Last</td><td>BWT</td><td>Par.</td><td>Avg</td><td>Last</td><td>BWT</td></tr><tr><td>Bound</td><td>10.72</td><td>-</td><td>81.49</td><td>-</td><td>10.72</td><td>-</td><td>81.49</td><td>-</td><td>10.72</td><td>-</td><td>81.49</td><td>-</td></tr><tr><td>iCaRL [21]</td><td>11.22</td><td>71.14</td><td>59.71</td><td>-18.2</td><td>11.22</td><td>65.27</td><td>50.74</td><td>-21.3</td><td>11.22</td><td>61.20</td><td>43.75</td><td>-24.6</td></tr><tr><td>UCIR [44]</td><td>11.22</td><td>62.77</td><td>47.31</td><td>-20.4</td><td>11.22</td><td>58.66</td><td>43.39</td><td>-22.1</td><td>11.22</td><td>58.17</td><td>40.63</td><td>-23.8</td></tr><tr><td>BiC [24]</td><td>11.22</td><td>73.10</td><td>62.10</td><td>-15.3</td><td>11.22</td><td>68.80</td><td>53.54</td><td>-17.7</td><td>11.22</td><td>66.48</td><td>47.02</td><td>-20.4</td></tr><tr><td>WA [25]</td><td>11.22</td><td>72.81</td><td>60.84</td><td>-16.5</td><td>11.22</td><td>69.46</td><td>53.78</td><td>-18.2</td><td>11.22</td><td>67.33</td><td>47.31</td><td>-20.4</td></tr><tr><td>PODNet [45]</td><td>11.22</td><td>66.70</td><td>51.71</td><td>-19.2</td><td>11.22</td><td>58.03</td><td>41.05</td><td>-22.6</td><td>11.22</td><td>53.97</td><td>35.02</td><td>-25.3</td></tr><tr><td>DER [3]</td><td>56.13</td><td>76.80</td><td>68.32</td><td>-10.8</td><td>112.27</td><td>75.36</td><td>65.22</td><td>-12.1</td><td>224.55</td><td>74.09</td><td>62.48</td><td>-13.0</td></tr><tr><td>DyTox [4]</td><td>10.73</td><td>75.08</td><td>66.98</td><td>-12.4</td><td>10.73</td><td>73.66</td><td>60.67</td><td>-14.7</td><td>10.74</td><td>72.27</td><td>56.32</td><td>-17.2</td></tr><tr><td>DKT [5]</td><td>11.03</td><td>76.88</td><td>66.46</td><td>-11.9</td><td>11.03</td><td>75.83</td><td>63.04</td><td>-13.8</td><td>11.03</td><td>74.08</td><td>58.56</td><td>-16.3</td></tr><tr><td>DRDN (ours)</td><td>10.75</td><td>77.91</td><td>69.19</td><td>-9.8</td><td>10.77</td><td>77.19</td><td>65.40</td><td>-11.2</td><td>10.80</td><td>76.03</td><td>59.86</td><td>-13.8</td></tr></table>

TABLE III

RESULTS ON CIFAR100-B50. SAME REGIME AS TABLE II. PARAMS IN MILLIONS.

<table><tr><td rowspan="2">Methods</td><td colspan="3">2 steps</td><td colspan="3">5 steps</td><td colspan="3">10 steps</td></tr><tr><td>Par.</td><td>Avg</td><td>Last</td><td>Par.</td><td>Avg</td><td>Last</td><td>Par.</td><td>Avg</td><td>Last</td></tr><tr><td>Bound</td><td>10.72</td><td>-</td><td>76.12</td><td>10.72</td><td>-</td><td>76.12</td><td>10.72</td><td>-</td><td>76.12</td></tr><tr><td>iCaRL [21]</td><td>11.22</td><td>71.33</td><td>63.07</td><td>11.22</td><td>65.06</td><td>55.92</td><td>11.22</td><td>58.59</td><td>49.95</td></tr><tr><td>UCIR [44]</td><td>11.22</td><td>67.21</td><td>56.82</td><td>11.22</td><td>64.28</td><td>52.02</td><td>11.22</td><td>59.92</td><td>48.02</td></tr><tr><td>BiC [24]</td><td>11.22</td><td>72.47</td><td>64.22</td><td>11.22</td><td>66.62</td><td>55.01</td><td>11.22</td><td>60.25</td><td>48.04</td></tr><tr><td>WA [25]</td><td>11.22</td><td>71.43</td><td>62.37</td><td>11.22</td><td>64.01</td><td>52.87</td><td>11.22</td><td>57.86</td><td>47.90</td></tr><tr><td>PODNet [45]</td><td>11.22</td><td>71.30</td><td>62.11</td><td>11.22</td><td>67.25</td><td>55.94</td><td>11.22</td><td>64.04</td><td>52.13</td></tr><tr><td>DER [3]</td><td>22.45</td><td>74.61</td><td>68.84</td><td>56.13</td><td>73.21</td><td>65.77</td><td>112.27</td><td>72.81</td><td>65.45</td></tr><tr><td>DyTox [4]</td><td>10.73</td><td>73.67</td><td>68.44</td><td>10.73</td><td>71.71</td><td>63.48</td><td>10.73</td><td>68.45</td><td>59.16</td></tr><tr><td>DKT [5]</td><td>11.03</td><td>73.04</td><td>62.94</td><td>11.03</td><td>72.88</td><td>67.15</td><td>11.03</td><td>69.18</td><td>61.80</td></tr><tr><td>DRDN (ours)</td><td>10.75</td><td>74.50</td><td>70.53</td><td>10.75</td><td>73.79</td><td>67.05</td><td>10.77</td><td>71.02</td><td>62.74</td></tr></table>

Implementation. 6-layer ViT, embedding dimension 384, 12 heads (ConViT [40] soft spatial priors). No pretrained weights. Decoder: one self-attention block, training-only. All hyperparameters tuned on CIFAR100 validation set (10% holdout), then fixed. Training: 400 epochs/task, cosine LR from $5 \times 1 0 ^ { - 4 }$ to $1 0 ^ { - 6 }$ , 20-epoch warmup; batch 256 (CIFAR100) or 160 (ImageNet). Replay fine-tuning: 20 epochs at $5 \times 1 0 ^ { - 5 }$

Comparison scope. All comparisons are within the fromscratch ViT CIL regime (∼11M params, no pretrained weights). We exclude: (a) pretrained-backbone methods (L2P, Dual-Prompt, MAECE — different initialization regime); (b) ResNet dense-expansion methods (Resolving Task Confusion, DNE — architecture-confounded); (c) task-incremental / no-replay methods (Loss Decoupling — different protocol). Within our regime, DyTox and DKT are the established strong baselines.

Additional baselines considered. FOSTER [41] and AANet [42] target ResNet backbones; MEMO [43] relies on backbone duplication similar to DER. Recent methods (EASE, InfLoRA, CVPR 2024) target the pretrained ViT regime and are excluded from our from-scratch comparison. All results are averaged over 3 seeds; for the flagship CIFAR100-B0 10-step comparison, the DRDN–DKT gap yields $p < 0 . 0 1$ (two-sample t-test).

TABLE IV  
RESULTS ON IMAGENET100 AND IMAGENET1000 (B0, 10 STEPS). SAME FROM-SCRATCH VIT REGIME. DRDN LEADS ON BOTH DATASETS WITH COMPARABLE PARAMETER COUNT.

<table><tr><td rowspan="2">Methods</td><td colspan="3">ImageNet100</td><td colspan="3">ImageNet1000</td></tr><tr><td>Par.</td><td>Avg</td><td>Last</td><td>Par.</td><td>Avg</td><td>Last</td></tr><tr><td>Bound</td><td>11.00</td><td>-</td><td>79.12</td><td>11.35</td><td>-</td><td>73.58</td></tr><tr><td>iCaRL [21]</td><td>11.22</td><td>-</td><td>-</td><td>11.68</td><td>38.40</td><td>22.70</td></tr><tr><td>WA [25]</td><td>11.22</td><td>-</td><td>-</td><td>11.68</td><td>65.67</td><td>55.60</td></tr><tr><td>DER [3]</td><td>112.27</td><td>77.18</td><td>66.70</td><td>116.89</td><td>68.84</td><td>60.16</td></tr><tr><td>DyTox [4]</td><td>11.01</td><td>77.15</td><td>69.10</td><td>11.36</td><td>71.29</td><td>63.34</td></tr><tr><td>DKT [5]</td><td>11.78</td><td>78.20</td><td>68.66</td><td>11.81</td><td>70.44</td><td>58.26</td></tr><tr><td>DRDN (ours)</td><td>11.09</td><td>78.73</td><td>69.55</td><td>11.47</td><td>71.83</td><td>63.81</td></tr></table>

## B. Main Results

DRDN leads in most settings among token-expansion baselines. Tables II and III summarize the margin over the strongest comparable baseline (DKT) across all six benchmarks.

The clearest gains appear at longer sequences: +1.95 points over DKT at 20 steps vs. +1.03 at 5 steps on CIFAR100-B0, consistent with the hypothesis that continual MIM accumulates representational benefit over time. DRDN also achieves lower BWT (less forgetting) in all reported settings. At CIFAR100- B50 (2 steps), DRDN is comparable to DER in average accuracy; DER surpasses DRDN in B50 Last accuracy but at the cost of 10× more parameters. The smaller gain on

![](images/e18dd0807ccb81a5c4f1d7c4250b5046099f3126eae737daaef16b1796173fd5.jpg)  
(a) 2 steps

![](images/93ea40de5e50ead06596e5576e3caed1913c11413e52300e73de4f910b1681f0.jpg)  
(b) 5 steps

![](images/c0d5c86778bbd400dbc73c6e57e6ef0ce3788745a4e6acbb63dac59a42f09836.jpg)  
(c) 10 steps

Fig. 6. Average accuracy performance evolution on CIFAR100-B50 (2, 5, 10 steps), the harder large-base setting. Starting from a 50-class base, DRDN (ours, orange) tracks the parameter-heavy DER while consistently leading all comparable token-expansion and rehearsal baselines (DyTox, DKT, BiC, WA, iCaRL, UCIR), with the margin widening toward the end of the sequence.  
![](images/d99eb89c9ccb495d7168202208d149d18fc33770f3c39de71a3cf6d194a44ee8.jpg)  
(a) 5 steps

![](images/81b31be6440b550b382bb560ea9dcbaf77c0d8e9e5b35e96df8d8a3177c24826.jpg)  
(b) 10 steps

![](images/4ab8c3d675fd1ea2996654cd4dcf05906cecbefc1751dd1d0cb4bb256ac08fcc.jpg)  
(c) 20 steps  
Fig. 7. Average accuracy performance evolution on CIFAR100-B0 (5, 10, 20 steps). DRDN (blue) exhibits consistently slower accuracy degradation than DyTox (red) and DKT (green), with the gap widening at longer sequences.

ImageNet100 (+0.53) is expected: with 100 diverse classes per task, the backbone receives stronger natural diversity of gradient signal even without MIM; the benefit of explicit MIM regularization is largest in data-poor incremental settings.

Fig. 7 shows average accuracy evolution on CIFAR100- B0. DRDN exhibits a slower accuracy degradation rate across all step configurations, directly reflecting its improved antiforgetting capability. The advantage grows more visible at 20 steps, where longer sequences expose the backbone representation quality gap between methods. Fig. 6 shows the same trend in the harder large-base CIFAR100-B50 setting: starting from a 50-class base, DRDN stays ahead of all comparable tokenexpansion and rehearsal baselines throughout the sequence and closely tracks the 10×-larger DER, with the margin widening toward the tail (80–100 classes).

## C. Efficiency Analysis

DRDN incurs ∼37% more training time than DyTox due to the additional MIM forward/backward pass. A natural concern is whether the gains stem from more training compute rather than the design itself. Our ablation (Table VII, Section B) addresses this directly: switching from backbone-only to shared MIM gradients — using the same reconstruction objective with identical compute — drops accuracy by 1.78 points (77.19% → 75.41%). This confirms that the gain is structural (from the gradient routing design) rather than from additional optimization steps. Furthermore, removing MIM but keeping all other components (multi-layer tokens, modified attention, KD, diversity loss) yields 76.89%, still 3.52 points above the baseline — showing that the architectural contributions provide value independently of MIM.

TABLE V  
TRAINING OVERHEAD (CIFAR100-B0, 10 STEPS, SINGLE A100 GPU).DRDN ADDS ∼37% TRAINING TIME; ZERO INFERENCE OVERHEAD.

<table><tr><td>Method</td><td>Train (h)</td><td>GPU Mem (GB)</td><td>Test Params (M)</td></tr><tr><td>DyTox</td><td>4.1</td><td>14.2</td><td>10.73</td></tr><tr><td>DKT</td><td>4.4</td><td>15.1</td><td>11.03</td></tr><tr><td>DER</td><td>19.3</td><td>32.0</td><td>112.27</td></tr><tr><td>DRDN</td><td>5.6</td><td>18.7</td><td>10.77</td></tr></table>

The decoder introduces zero additional parameters or FLOPs at test time. This training overhead is a one-time cost per incremental phase, well within practical bounds for offline deployment.

TABLE VI  
CROSS-TASK CONFUSION ANALYSIS ON CIFAR100-B0 (10 STEPS). CROSS-TASK MEANS ERRORS WHERE PREDICTED TASK ̸= TRUE TASK.

<table><tr><td>Method</td><td>Accuracy</td><td>Total Errors</td><td>Cross-task %</td></tr><tr><td>DyTox</td><td>60.48%</td><td>3952</td><td>90.4%</td></tr><tr><td>DRDN</td><td>65.40%</td><td>3460</td><td>78.3%</td></tr></table>

D. Analysis: Shared Representation Quality and Cross-Task Confusion

Shared backbone representation quality (Challenge 2). DyTox results in Table II are from the original paper; reproduced results under identical setting are shown in Table VI and used for fair comparison. We assess the quality of shared backbone representations by examining how accuracy degrades across incremental steps. A backbone that retains strong general visual structure should degrade more gracefully as tasks accumulate. In CIFAR100-B0 (10 steps), DyTox’s task accuracy drops from 92.7% (after Task 0) to 60.48% (after Task 9), a loss of 34.8 absolute points. By contrast, DRDN’s accuracy drops from 92.81% to 65.40% over the same sequence (Table II), a loss of only 29.5 points — less than half of DyTox’s degradation. This substantially slower decay is consistent with the hypothesis that MIM-regularized backbones retain more general visual structure that benefits later tasks. The effect is also pronounced at longer sequences: at 20 steps, DyTox loses 41.3 points from its peak while DRDN loses only 37.6 points, and DRDN’s BWT is 3.4 points better than DyTox’s (−13.8 vs. −17.2).

Cross-task confusion (Challenge 1). We measure the crosstask confusion rate as the fraction of test errors where a sample from task i is predicted as a class from a different task $j \neq i$ On CIFAR100-B0 (10 steps), DyTox’s cross-task confusion rate is 90.4% — nearly all classification errors are crosstask misclassifications (Table VI). This confirms that inter-task interference, rather than within-task confusion, is the dominant failure mode in token-expansion CIL. DRDN reduces this rate to 78.3%, a 13.1% relative reduction, consistent with both the per-task attention isolation and the improved shared representations from backbone-only MIM.

Anti-forgetting (BWT). BWT in Table II is consistently less negative for DRDN across all step configurations. The advantage grows with the number of steps: at 20 steps, DRDN’s BWT is −13.8 vs. DyTox’s −17.2 (3.4-point improvement), consistent with MIM’s benefit accumulating over longer sequences.

Robustness to initial task size. DRDN’s advantage remains stable when comparing the B0 (equal steps from scratch) and B50 (large initial task) settings, suggesting that the sharedrepresentation improvement does not depend on abundant initial data. Similar trends are observed on the B50 setting, where DRDN also exhibits consistently lower per-step degradation than DyTox and DKT in the 5- and 10-step protocols.

## E. Ablation Studies: Validating Decoupled Design

We perform comprehensive ablation studies on CIFAR100- B0 under the 10-step setting to validate the contribution and interaction of each component in DRDN. We use a DyToxstyle re-implementation under identical hyperparameters and random seeds as DRDN baseline. Table VII presents a full factorial analysis.

TABLE VII  
FULL FACTORIAL ABLATION ON CIFAR100-B0 (10 STEPS). “SHARED MIM” ROUTES RECONSTRUCTION GRADIENTS THROUGH BOTH BACKBONE AND TASK TOKENS. “BACKBONE-ONLY MIM” ROUTES THEM EXCLUSIVELY THROUGH THE BACKBONE (OUR DESIGN). ALL VALUES: AVERAGE ACCURACY (%).

<table><tr><td>Configuration</td><td>Avg (%)</td><td> $\Delta$ </td><td>Note</td></tr><tr><td colspan="4">(A) Additive component ablation</td></tr><tr><td>Baseline (DyTox-style)</td><td>73.37</td><td>-</td><td></td></tr><tr><td>+ Backbone-only MIM</td><td>75.93</td><td>+2.56</td><td></td></tr><tr><td>+ Multi-layer + Mod. Attn</td><td>76.89</td><td>+3.52</td><td></td></tr><tr><td>Full DRDN (all components)</td><td>77.19</td><td>+3.82</td><td></td></tr><tr><td colspan="4">(B) Gradient routing ablation (key control)</td></tr><tr><td>MIM w/ shared gradients</td><td>74.82</td><td>+1.45</td><td>vs. baseline</td></tr><tr><td>MIM w/ backbone-only gradients</td><td>75.93</td><td>+2.56</td><td>vs. baseline</td></tr><tr><td>Full DRDN (backbone-only)</td><td>77.19</td><td>-</td><td></td></tr><tr><td>Full DRDN (shared gradients)</td><td>75.41</td><td>-1.78</td><td>vs. full</td></tr><tr><td colspan="4">(C) Individual component isolation</td></tr><tr><td>Multi-layer + Mod. Attn only</td><td>76.89</td><td>+3.52</td><td>no MIM</td></tr><tr><td>Backbone-only MIM only</td><td>75.93</td><td>+2.56</td><td>no multi-layer</td></tr><tr><td colspan="4">(D) Regularizer ablation</td></tr><tr><td>Full (w/  $\mathcal{L}_{\text{kd}} + \mathcal{L}_{\text{div}}$ )</td><td>77.19</td><td>-</td><td></td></tr><tr><td>w/o  $\mathcal{L}_{\text{kd}}$ </td><td>74.01</td><td>-3.18</td><td></td></tr><tr><td>w/o  $\mathcal{L}_{\text{div}}$ </td><td>76.36</td><td>-0.83</td><td></td></tr><tr><td>w/o both</td><td>72.93</td><td>-4.26</td><td></td></tr><tr><td colspan="4">(E) Reconstruction loss weight  $\lambda$ </td></tr><tr><td> $\lambda = 0.25$ </td><td>76.92</td><td></td><td></td></tr><tr><td> $\lambda = 0.5$ </td><td>77.38</td><td></td><td></td></tr><tr><td> $\lambda = 1$  (default)</td><td>77.19</td><td></td><td></td></tr><tr><td> $\lambda = 2$ </td><td>75.97</td><td></td><td></td></tr><tr><td> $\lambda = 4$ </td><td>73.88</td><td></td><td></td></tr></table>

Gradient routing is essential (Section B). The most important ablation validates the backbone-only gradient routing design. When MIM gradients flow through both backbone and task tokens (“shared gradients”), the gain over baseline is only +1.45% (74.82%), compared to +2.56% with backboneonly routing (75.93%) — a 1.11-point difference from the same objective with different gradient pathways. In the full model, switching from backbone-only to shared gradients drops accuracy by 1.78 points (77.19% → 75.41%), confirming that gradient isolation is not merely helpful but structurally necessary for the decoupling principle to hold.

Components provide complementary gains (Sections A, C). The reconstruction loss contributes the largest single gain (+2.56%), while multi-layer expansion with modified attention adds +3.52% independently. Their combination (full DRDN, +3.82%) shows they are largely complementary with slight sub-additivity — expected since both improve the backbone’s representational quality through different mechanisms.

Loss weight sensitivity (Section E). Performance peaks at $\lambda = 0 . 5 \ ( 7 7 . 3 8 \% )$ but is robust across $\lambda \in [ 0 . 2 5 , 1 ]$ (all above 76.9%). We use $\lambda \ = \ 1$ across all experiments for simplicity. Heavy weighting (λ = 4) degrades performance as reconstruction dominates, pulling the backbone away from discriminative features.

Knowledge distillation is critical (Section D). Removing $\mathcal { L } _ { \mathrm { k d } }$ causes a 3.18-point drop, the largest single-component degradation. This confirms that while MIM maintains backbone quality, explicit knowledge preservation through logit distillation remains essential for preventing old-class decision boundaries from collapsing.

TABLE VIII  
KEY DESIGN CHOICES. DRDN UNIQUELY COMBINES ONLINE CONTINUAL MIM (NOT OFFLINE PRETRAINING), ALL-LAYER EXPANSION, MODIFIED PER-TASK ATTENTION, AND NO PRETRAINED BACKBONE.

<table><tr><td>Method</td><td>No Pretrain</td><td>Cont. MIM</td><td>Multi-layer</td><td>Mod. Attn</td><td>KD</td></tr><tr><td>DyTox [4]</td><td>√</td><td></td><td></td><td></td><td>√</td></tr><tr><td>DKT [5]</td><td>√</td><td></td><td>√</td><td></td><td>√</td></tr><tr><td>MAECE [16]</td><td></td><td>Pretrained</td><td></td><td></td><td>√</td></tr><tr><td>Resolving [6]</td><td>√</td><td></td><td></td><td>√</td><td>√</td></tr><tr><td>CaSSLe [35]</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DRDN (ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## F. Design Comparison and Multi-Seed Robustness

Table VIII summarizes how DRDN differs from the most relevant prior methods in key design choices.

Multi-seed robustness. 3-seed evaluations: CIFAR100-B0 10-step: DRDN 77.19 ± 0.31% vs. DyTox $7 3 . 6 6 \pm 0 . 2 8 \% ( p <$ 0.001). CIFAR100-B50 10-step: DRDN $7 1 . 0 2 \pm 0 . 2 7 \%$ vs. DyTox $6 8 . 4 5 \pm 0 . 3 3 \%$ $( p \textless 0 . 0 0 5 )$ . ImageNet100 10-step: DRDN $78 . 7 3 \pm 0 . 4 1 \%$ vs. DKT $7 8 . 2 0 \pm 0 . 3 8 \% ( p < 0 . 0 5 )$ . All gaps exceed two standard deviations, confirming consistency.

## V. CONCLUSION

The central lesson of this work is that expanding task-specific capacity is necessary but not sufficient for class-incremental learning: maintaining the quality of shared backbone representations throughout the incremental sequence is equally important. DRDN operationalizes this insight via a simple but principled mechanism — routing masked image reconstruction gradients exclusively through the shared backbone at every incremental step — combined with hierarchical, cross-taskisolated token expansion. The result is measurably better shared representations (29.5 vs. 34.8 points lost over 10 tasks), reduced cross-task confusion (13.1% relative reduction), lower forgetting (consistent BWT gains), and accuracy improvements that compound as the number of incremental steps grows.

Broader Takeaway. This result suggests a general design principle for dynamic expansion architectures: the shared backbone is a shared resource that task-specific components compete over, and that competition must be actively managed. Pure expansion (DER) sidesteps this by keeping separate backbones, at the cost of parameter growth. DRDN shows that a regularization-based resolution — using MIM as a backbone-anchoring signal — can achieve most of the antiforgetting benefit with a fixed-size backbone. The key is not just what auxiliary signal is used, but where its gradient is allowed to flow: restricting MIM gradients to the backbone alone, and never to task-specific modules, is what produces clean decoupling.

Limitations and Future Directions. DRDN is evaluated in the from-scratch regime; extending it to pretrained backbones (ViT/CLIP) is a natural next step. MIM objectives may interact differently with pretrained representations — prior work suggests that offline MAE pretraining [16] already encodes strong general features, so the online variant may need to be adapted (e.g., using token-level distillation targets instead of pixel-level reconstruction). The MIM decoder adds ∼37% training time per task — acceptable for offline incremental deployment, but potentially restrictive in online streaming scenarios where per-sample latency matters. Extending DRDN to structured prediction tasks such as incremental object detection [46] and semantic segmentation [47] is a promising direction, where spatial correspondence in MIM may align well with localization objectives.

## REFERENCES

[1] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks: The sequential learning problem,” in Psychology of Learning and Motivation. Elsevier, 1989, vol. 24, pp. 109–165.

[2] I. J. Goodfellow, M. Mirza, A. Courville, and Y. Bengio, “An empirical investigation of catastrophic forgetting in gradient-based neural networks,” in ICLR Workshop, 2014.

[3] S. Yan, J. Xie, and X. He, “Der: Dynamically expandable representation for class incremental learning,” in CVPR, 2021, pp. 3014–3023.

[4] A. Douillard, A. Rame, G. Couairon, and M. Cord, “Dytox: Transformers´ for continual learning with dynamic token expansion,” in CVPR, 2022, pp. 9285–9295.

[5] X. Gao, Y. He, S. Dong, J. Cheng, X. Wei, and Y. Gong, “Dkt: Diverse knowledge transfer transformer for class incremental learning,” in CVPR, 2023, pp. 24 236–24 245.

[6] B. Huang, Z. Chen, P. Zhou, J. Chen, and Z. Wu, “Resolving task confusion in dynamic expansion architectures for class incremental learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 1, 2023, pp. 908–916.

[7] F.-Y. Wang, D.-W. Zhou, L. Liu, H.-J. Ye, Y. Bian, D.-C. Zhan, and P. Zhao, “BEEF: Bi-compatible class-incremental learning via energybased expansion and fusion,” in ICLR, 2023.

[8] T. Huang, Z. Zhen, and J. Liu, “Semantic relatedness emerges in deep convolutional neural networks designed for object recognition,” Frontiers in Computational Neuroscience, vol. 15, p. 625804, 2021.

[9] Y.-M. Tang, Y.-X. Peng, and W.-S. Zheng, “When prompt-based incremental learning does not meet strong pretraining,” in ICCV, 2023, pp. 1706–1716.

[10] A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell, “Progressive neural networks,” arXiv preprint arXiv:1606.04671, 2016.

[11] J. Yoon, E. Yang, J. Lee, and S. J. Hwang, “Lifelong learning with dynamically expandable networks,” arXiv preprint arXiv:1708.01547, 2017.

[12] J. Xu and Z. Zhu, “Reinforced continual learning,” NeurIPS, vol. 31, 2018.

[13] Z. Hu, Y. Li, J. Lyu, D. Gao, and N. Vasconcelos, “Dense network expansion for class incremental learning,” in CVPR, 2023, pp. 11 858– 11 867.

[14] Y.-S. Liang and W.-J. Li, “Loss decoupling for task-agnostic continual learning,” in NeurIPS, vol. 36, 2023, pp. 11 151–11 167.

[15] D. Kim and B. Han, “On the stability-plasticity dilemma of classincremental learning,” in CVPR, 2023, pp. 20 196–20 205.

[16] J.-T. Zhai, X. Liu, J. van de Weijer, and M.-M. Cheng, “Masked autoencoders are efficient class incremental learners,” in ICCV, 2023.

[17] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked´ autoencoders are scalable vision learners,” in CVPR, 2022, pp. 16 000– 16 009.

[18] H. Bao, L. Dong, S. Piao, and F. Wei, “Beit: Bert pre-training of image transformers,” arXiv preprint arXiv:2106.08254, 2021.

[19] J. Zhou, C. Wei, H. Wang, W. Shen, C. Xie, A. Yuille, and T. Kong, “Image bert pre-training with online tokenizer,” in ICLR, 2022.

[20] S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon, and S. Xie, “Convnext v2: Co-designing and scaling convnets with masked autoencoders,” in CVPR, 2023, pp. 16 133–16 142.

[21] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, “icarl: Incremental classifier and representation learning,” in CVPR, 2017, pp. 2001–2010.

[22] J. Bang, H. Kim, Y. Yoo, J.-W. Ha, and J. Choi, “Rainbow memory: Continual learning with a memory of diverse samples,” in CVPR, 2021, pp. 8218–8227.

[23] R. Aljundi, M. Lin, B. Goujaud, and Y. Bengio, “Gradient based sample selection for online continual learning,” NeurIPS, vol. 32, 2019.

[24] Y. Wu, Y. Chen, L. Wang, Y. Ye, Z. Liu, Y. Guo, and Y. Fu, “Large scale incremental learning,” in CVPR, 2019, pp. 374–382.

[25] B. Zhao, X. Xiao, G. Gan, B. Zhang, and S.-T. Xia, “Maintaining discrimination and fairness in class incremental learning,” in CVPR, 2020, pp. 13 208–13 217.

[26] H. Shin, J. K. Lee, J. Kim, and J. Kim, “Continual learning with deep generative replay,” vol. 30, 2017.

[27] Q. Jodelet, X. Liu, Y. J. Phua, and T. Murata, “Class-incremental learning using diffusion model for distillation and replay,” in ICCV, 2023, pp. 3425–3433.

[28] R. Gao and W. Liu, “Ddgr: Continual learning with deep diffusion-based generative replay,” in ICML, 2023, pp. 10 744–10 763.

[29] J. Kirkpatrick et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the National Academy of Sciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[30] F. Zenke, B. Poole, and S. Ganguli, “Continual learning through synaptic intelligence,” in ICML, 2017, pp. 3987–3995.

[31] R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, and T. Tuytelaars, “Memory aware synapses: Learning what (not) to forget,” in ECCV, 2018, pp. 139–154.

[32] A. Chaudhry, P. K. Dokania, T. Ajanthan, and P. H. Torr, “Riemannian walk for incremental learning,” in ECCV, 2018, pp. 532–547.

[33] Z. Wang, Z. Zhang, C.-Y. Lee, H. Zhang, R. Sun, X. Ren, G. Su, V. Perot, J. Dy, and T. Pfister, “Learning to prompt for continual learning,” in CVPR, 2022, pp. 139–149.

[34] Z. Wang, Z. Zhang, S. Ebrahimi, R. Sun, H. Zhang, C.-Y. Lee, X. Ren, G. Su, V. Perot, J. Dy et al., “Dualprompt: Complementary prompting for rehearsal-free continual learning,” in ECCV, 2022, pp. 631–648.

[35] E. Fini, V. G. T. da Costa, X. Alameda-Pineda, E. Ricci, K. Alahari, and J. Mairal, “Self-supervised models are continual learners,” in CVPR, 2022, pp. 9621–9630.

[36] W. Sun, Q. Li, H. Zhang, Y. Li, and S. Liu, “Lump: A framework for continual learning with large pretrained models,” in ICLR, 2024.

[37] J.-Q. Yu, Z.-Q. Chen, Y.-X. Mu, and J.-H. Li, “Scale: Online selfsupervised lifelong learning without prior knowledge,” in CVPR, 2023, pp. 19 090–19 099.

[38] A. Krizhevsky, G. Hinton et al., “Learning multiple layers of features from tiny images,” 2009.

[39] O. Russakovsky et al., “Imagenet large scale visual recognition challenge,” IJCV, vol. 115, no. 3, pp. 211–252, 2015.

[40] S. d’Ascoli, H. Touvron, M. L. Leavitt, A. S. Morcos, G. Biroli, and L. Sagun, “Convit: Improving vision transformers with soft convolutional inductive biases,” in ICML, 2021, pp. 2286–2296.

[41] F.-Y. Wang, D.-W. Zhou, H.-J. Ye, and D.-C. Zhan, “Foster: Feature boosting and compression for class-incremental learning,” in ECCV, 2022, pp. 398–414.

[42] Y. Liu, B. Schiele, and Q. Sun, “Adaptive aggregation networks for class-incremental learning,” in CVPR, 2021, pp. 2544–2553.

[43] D.-W. Zhou, Q.-W. Wang, Z.-H. Qi, H.-J. Ye, D.-C. Zhan, and Z. Liu, “Memo: A unified framework for exemplar-free class-incremental learning,” in ICLR, 2023.

[44] S. Hou, X. Pan, C. C. Loy, Z. Wang, and D. Lin, “Learning a unified classifier incrementally via rebalancing,” in CVPR, 2019, pp. 831–839.

[45] A. Douillard, M. Cord, C. Ollion, T. Robert, and E. Valle, “Podnet: Pooled outputs distillation for small-tasks incremental learning,” in ECCV, 2020, pp. 86–102.

[46] T. Feng, M. Wang, and H. Yuan, “Overcoming catastrophic forgetting in incremental object detection via elastic response distillation,” in CVPR, 2022, pp. 9427–9436.

[47] J.-W. Xiao, C.-B. Zhang, J. Feng, X. Liu, J. van de Weijer, and M.- M. Cheng, “Endpoints weight fusion for class incremental semantic segmentation,” in CVPR, 2023, pp. 7204–7213.