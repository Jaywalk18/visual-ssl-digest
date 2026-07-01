# ExPLoRe: Expert Patch-Level Loss Routing for Multi-Objective Masked Image Modeling

Konstantinos Georgiou, Maofeng Tang, and Hairong Qi

Min H. Kao Department of Electrical Engineering and Computer Science, The University of Tennessee, Knoxville, TN 37996, USA {kgeorgio, mtang4}@vols.utk.edu, hqi@utk.edu

Abstract. Multi-objective masked image modeling (MIM) combines com plementary learning signals (token distillation, CLS alignment, and pixel reconstruction) but existing methods weight these objectives with global scalars, ignoring spatial heterogeneity across patches. We present Ex-PLoRe (Expert Patch-Level Loss Routing), which repurposes Soft Mixture of Experts (MoE) dispatch weights as learned, per-patch loss coefficients. The key mechanism is loss-coupling: allowing loss gradients to flow through dispatch weights to the router enables content-dependent specialization, where diferent patches receive diferent emphases across objectives. A detach ablation confirms loss-coupling as the core mechanism, degrading performance by 1.6% when gradients are blocked. On ImageNet-1K with ViT-Base, ExPLoRe improves over non-MoE baselines on two objective combinations (Token+CLS: +0.5% k-NN, +4.4% linear probe; Token+Pixel: +2.2% k-NN), achieving 80.6% linear probe and 85.3% finetuning accuracy, competitive with published methods. For downstream transfer, we develop adaptation recipes (Freeze Routing, Expert Dropout, and Freeze Attention) that improve MoE finetuning by +1.5% over the vanilla MoE, and close a 2.5–2.9 mIoU segmentation gap so that MoE models match or exceed non-MoE baselines on ADE20K.

Keywords: Masked Image Modeling · Mixture of Experts · Self-Supervised Learning · Knowledge Distillation · Multi-Task Learning

## 1 Introduction

Masked image modeling (MIM), inspired by BERT’s masked language modeling [10], has emerged as a powerful self-supervised learning paradigm for vision [3,15]. Teacher-guided approaches [12,16,22] leverage frozen feature extractors such as CLIP [25] to provide semantic targets for student encoders. Recent works combine multiple learning signals including token-level distillation, CLS alignment, and pixel reconstruction; but how to weight these diverse objectives optimally remains an open question that is largely independent of the specific teacher used.

The Patch-Level Heterogeneity Problem. Existing multi-task learning methods [8, 19] balance losses at the global level, applying a single scalar weight per objective uniformly across all spatial locations. However, diferent image regions benefit from diferent learning signals: semantically rich patches are better served by teacher distillation, while texture-heavy background regions benefit from pixel reconstruction. Global methods cannot capture this spatial variation. As we demonstrate, gradient-based balancing (GradNorm [8]) can even fail catastrophically when objectives have spatially heterogeneous importance. To our knowledge, no prior work addresses patch-level loss weighting for masked image modeling.

![](images/41687d80d61c50e3a4df2c2168d055ae904d153b929c6e5ff8504df7473d5ecb.jpg)  
Fig. 1: Expert dispatch-weight visualization. Per-patch dispatch weights from a trained 2-expert ExPLoRe model overlaid on input images (warm = high weight, cool = low weight). The two experts learn complementary spatial specialization without explicit supervision: one expert assigns higher loss emphasis to foreground regions while the other focuses on background and context.

We present ExPLoRe<sup>1</sup> (Expert Patch-Level Loss Routing), which addresses patch-level heterogeneity by repurposing Soft Mixture of Experts (Soft-MoE) [24] dispatch weights as learned, per-patch loss coeficients. Unlike traditional MoE methods that use routing for capacity scaling [13, 26, 29] or task-specific expert assignment [5, 34], ExPLoRe uses continuous, sample-adaptive weighting through a loss-coupling mechanism: gradients from each loss objective flow back through the dispatch weights to the router, enabling the model to learn contentdependent specialization. We evaluate two objective combinations (Token+CLS and Token+Pixel) with expert scaling from 2 to 64 under sparse encoding [15], and develop downstream transfer recipes for competitive finetuning. Figure 1 visualizes the resulting routing patterns.

ExPLoRe provides content-dependent per-patch loss emphasis that prior methods lack; competitive results across classification and dense prediction demonstrate this mechanism does not sacrifice representation quality. Our contributions center on a single mechanism with two supporting analyses:

– Core contribution: dispatch-weight loss weighting. We introduce a mechanism that repurposes Soft-MoE dispatch weights as per-patch loss coeficients for multi-objective MIM. Loss-coupling, where loss gradients flow through dispatch weights to update the router, is the core innovation enabling learned per-patch specialization. We demonstrate consistent improvements over non-MoE baselines across two objective combinations, achieving 80.6% linear probe accuracy competitive with published methods (Table 6).

Routing dynamics analysis. We empirically characterize routing behavior: entropy regularization governs a sharp transition between stable routing and catastrophic collapse; optimal regularization is expert-count dependent; and a detach ablation validates loss-coupling as the mechanism driving specialization (Sections 4.2 and 4.2).

– Downstream transfer recipes. We develop complementary adaptation strategies, Freeze Routing (FR), Expert Dropout (ExD), and Freeze Attention (FA), that compose naturally. The combined FR+FA+ExD recipe exceeds the non-MoE baseline on both classification (+0.5%) and semantic segmentation, closing a 2.5–2.9 mIoU gap (Section 4.4).

## 2 Related Work

Masked Image Modeling. MIM extends masked language modeling (MLM) from NLP to vision. BEiT [3] uses dense encoders processing all patches with mask tokens, while MAE [15] uses sparse encoders processing only visible patches with shufling for eficiency. SimMIM [33] uses simple linear projections, iBOT [38] combines masking with self-distillation, and MaskFeat [30] uses HOG features as targets.

Self-distillation approaches include data2vec [1, 2] (EMA representations), CAE/CAEv2 [6,36] (context autoencoding), SdAE [7] (pixel + self-distillation), and BootMAE [11] (bootstrapping).

Teacher-Guided MIM with CLIP. CLIP [25] enables MIM methods with semantic targets. MaskDistill [22] provides a unified view of masked image modeling, formulating MIM as $\dot { \mathcal { L } } ( \dot { \mathcal { N } } ( \mathcal { T } ( I ) ) , \mathcal { H } ( \mathcal { S } ( I _ { m } ) ) )$ with teacher $\tau ,$ student $s ,$ projector heads ${ \mathcal { N } } / { \mathcal { H } } ,$ and shows CLIP features outperform pixels or discrete tokens as reconstruction targets. Our base architecture builds upon MaskDistill’s framework. BEiT v2 [21] combines tokenizers with CLIP distillation. MI-LAN [16] uses CLIP attention for semantic masking and combines CLS with patch distillation. MaskCLIP [12] explores masked self-distillation within CLIP.

Multi-Task Learning and Loss Balancing. Multiple methods balance competing objectives: uncertainty weighting [18] models task-dependent uncertainty, GradNorm [8] normalizes gradient magnitudes, PCGrad [35] projects conflicting gradients, MGDA [28] seeks Pareto-optimal solutions, and random weighting [19] samples task weights stochastically. Crucially, all operate at the global level, applying a single scalar weight per objective uniformly across all spatial locations. This ignores the spatial structure of images, where diferent regions may benefit from diferent loss emphasis. To our knowledge, no prior work addresses per-patch adaptive loss weighting for masked image modeling, which we identify as a key gap.

Mixture of Experts. Sparse MoE methods [13, 29] route tokens to expert subsets for conditional computation but introduce load balancing challenges. Soft-MoE [24] addresses this through fully-diferentiable soft routing, where dispatch and combine weights computed via softmax avoid discrete decisions and their associated training dificulties. V-MoE [26] applied sparse MoE to vision transformers; subsequent work [14, 20] compared routing strategies, and multitask applications [5,34] demonstrated MoE efectiveness in vision. CR-MoE [17] first applied MoE to self-supervised learning (contrastive), discovering routing consistency challenges. However, applications to masked image modeling with dynamic task weighting have not been investigated. MoE downstream transfer remains understudied; ViMoE [14] finds that only 2–5 MoE layers sufice but focuses on supervised training. In contrast to prior work on capacity scaling [13, 26, 29] or task-specific expert assignment [5, 34], ExPLoRe repurposes Soft-MoE dispatch weights for per-patch loss weighting through continuous, losscoupled routing, and addresses the MoE transfer gap with dedicated adaptation recipes (Section 4.4).

## 3 Method

ExPLoRe uses Soft-MoE to learn per-patch loss weighting in multi-objective masked image modeling, repurposing dispatch weights as dynamic loss coeficients (Figure 2). Given an input image divided into patches with a binary mask partitioning them into visible set V and masked set M, the framework consists of: (1) a student encoder processing visible patches, (2) a frozen CLIP teacher providing semantic targets, and (3) optional auxiliary networks (decoder for pixel reconstruction, projection heads for CLS alignment).

## 3.1 Multi-Objective Learning Framework

Our framework combines three complementary objectives at diferent spatial scales. Per-image losses are defined below; MoE-weighted variants (Section 3.2) average over the batch.

Token-Level Distillation: Distills CLIP teacher features t into student predictions <sup>ˆ</sup>t using Huber loss (β = 1.0) with layer-normalized teacher features:

$$
\mathcal {L} _ {\mathrm{token}} = \frac {1}{| \mathcal {M} |} \sum_ {i \in \mathcal {M}} \ell_ {\mathrm{huber}} (\hat {\mathbf {t}} _ {i}, \mathrm{LN} (\mathbf {t} _ {i}))\tag{1}
$$

Global CLS Alignment: Aligns CLS tokens of student and teacher:

$$
\mathcal {L} _ {\mathrm{cls}} = 1 - \frac {\hat {\mathbf {z}} _ {\mathrm{cls}} \cdot \mathbf {t} _ {\mathrm{cls}}}{\left\| \hat {\mathbf {z}} _ {\mathrm{cls}} \right\| \left\| \mathbf {t} _ {\mathrm{cls}} \right\|}\tag{2}
$$

Pixel Reconstruction: A decoder reconstructs normalized pixel values:

$$
\mathcal {L} _ {\mathrm{pixel}} = \frac {1}{| \mathcal {M} |} \sum_ {i \in \mathcal {M}} \| \hat {\mathbf {x}} _ {i} - \bar {\mathbf {x}} _ {i} \| _ {2} ^ {2}\tag{3}
$$

![](images/823c13a8778a77f8fedb5bab55114867cf97a4767349d0ebce7a4a34a37423b3.jpg)  
Fig. 2: ExPLoRe Framework Overview. Soft Mixture of Experts (Soft-MoE) is integrated into the student encoder for patch-level adaptive loss weighting. The student encoder (ViT-Base with alternating MoE blocks at layers {1,3,5,7,9,11}) processes patches while a frozen CLIP teacher provides semantic targets. Soft-MoE dispatch weights D serve as per-patch loss coeficients: each expert weights a diferent training objective. The two routing weight types play distinct roles: dispatch weights D (normalized over patches per expert) form the loss-coupling pathway that carries loss gradients back to the router, whereas combine weights C (normalized over experts per patch) only mix expert outputs in the forward pass. Loss-coupling, where loss gradients flow through D to the router, is the key mechanism enabling learned specialization.

where $\hat { \mathbf { x } } = d _ { \omega } ( \mathbf { z } )$ are reconstructed patches and x¯ are normalized targets. Ex-PLoRe uses Soft MoE dispatch weights as learned per-patch loss coeficients for the combined objective (Section 3.2).

Encoding Strategy and Decoder. We adopt sparse encoding (MAEstyle [15]) for our main model, processing only visible patches for eficiency. Sparse encoding yields dispatch weights only for visible patches, determining which losses can be MoE-weighted (Section 3.2). Dense encoding (BEiT-style [3]) is evaluated as an ablation. A lightweight 8-block decoder reconstructs masked patches; full details in supplementary Section E.

## 3.2 Soft Mixture of Experts for Patch-Level Dynamic Loss Weighting

Global methods apply uniform weights across patches, but diferent patches benefit from diferent signals: semantic regions need distillation, texture regions need reconstruction. We integrate Soft-MoE [24] into the student encoder for learned patch-level adaptive weighting (Figure 2). The central idea is that Soft-MoE produces two types of weights from shared logits: dispatch weights D (softmax over patches, normalized per expert) that determine how patches contribute to each expert, and combine weights C (softmax over experts, normalized per patch) that determine the output mixture. We repurpose dispatch weights, rather than combine weights, as loss coeficients, for reasons detailed below.

Architecture and Routing Mechanism: We adopt the Soft-MoE formulation [24] (full derivation in supplementary Section G). Soft-MoE replaces selected MLP layers with E expert networks, each a standard MLP with independent parameters. Given patch representations $\mathbf { X } \in \mathbb { R } ^ { B \times N \times d }$ , learnable expert parameters $\Phi \in \mathbb { R } ^ { d \times E }$ , and a learned scale s, routing logits are computed as:

$$
\mathbf {L} = s \cdot \frac {\mathbf {X}}{\| \mathbf {X} \| _ {2}} \cdot \frac {\boldsymbol {\Phi}}{\| \boldsymbol {\Phi} \| _ {2}}\tag{4}
$$

These shared logits yield two weight types via softmax along diferent dimensions: dispatch weights $\mathbf { D } = \mathrm { s o f t m a x } _ { \mathrm { p a t c h e s } } ( \mathbf { L } )$ with $\begin{array} { r } { \sum _ { n } \mathbf { D } _ { b , n , e } = 1 } \end{array}$ per expert, and combine weights $\mathbf { C } = \mathrm { s o f t m a x } _ { \mathrm { e x p e r t s } } ( \mathbf { L } )$ with $\begin{array} { r } { \sum _ { e } \mathbf { C } _ { b , n , e } = 1 } \end{array}$ per patch. Dispatch weights aggregate patches into expert inputs $\mathbf { S } _ { \mathrm { i n } } = \mathbf { D } ^ { \top } \mathbf { X }$ , each expert processes its slot, and combine weights reconstruct per-patch outputs $\mathbf { Y } = \mathbf { C } \cdot \mathbf { S } _ { \mathrm { o u t } } .$ The critical distinction for loss weighting is that dispatch weights are normalized over patches (per expert) while combine weights are normalized over experts (per patch).

Dispatch-Weight-Based Loss Weighting: Our key innovation is using dispatch weights D (not combine weights) for patch-level loss weighting. Unlike combine weights which represent per-token expert mixtures with constraint $\begin{array} { r } { \sum _ { e } \mathbf { C } _ { b , n , e } = 1 } \end{array}$ (normalized over experts for each patch), dispatch weights represent how much each patch contributes to each expert with constraint $\begin{array} { r } { \sum _ { n } \mathbf { D } _ { b , n , e } = } \end{array}$ 1 (normalized over patches for each expert). This expert-centric perspective prevents a critical degeneracy issue: with combine weights, the router could minimize loss by setting all weights to zero for dificult patches, causing the loss to vanish. With dispatch weights, the normalization constraint over patches prevents this collapse, made concrete by the per-image normalization in Eq. 5 below. The router must distribute each expert’s "attention" across patches and cannot zero out the entire loss.

For our main configuration (Token+CLS, sparse encoding), Expert 0 weights the token distillation loss on visible patches V. Each patch’s contribution is scaled by its dispatch weight, averaged over the batch:

$$
\mathcal {L} _ {\mathrm{token}} ^ {\mathrm{MoE}} = \frac {1}{B} \sum_ {b = 1} ^ {B} \frac {\sum_ {i \in \mathcal {V}} \mathbf {D} _ {b , i , 0} \cdot \ell_ {\mathrm{huber}} (\hat {\mathbf {t}} _ {b , i} , \mathrm{LN} (\mathbf {t} _ {b , i}))}{\sum_ {i \in \mathcal {V}} \mathbf {D} _ {b , i , 0}}\tag{5}
$$

Per-image normalization (dividing by the sum of dispatch weights) computes a weighted average for each image independently, preventing the router from minimizing loss by scaling weights down across the batch.

When $E > K$ (more experts than loss-weighted objectives), only K experts receive direct loss-coupling; the rest provide MoE capacity through forward-pass gradients and entropy regularization (supplementary Section H). The pixel loss variant applies analogous weighting (supplementary Section D); this requires dense encoding since sparse mode produces dispatch weights only for visible patches V, while pixel loss is computed on masked patches M.

Entropy Regularization for Uniform Token Distribution: While dispatch weights prevent loss degeneracy, they don’t guarantee balanced expert utilization. Without regularization, one expert might receive contributions from all patches while another receives none, leading to expert collapse. Per-expert entropy regularization encourages uniform token distributions.

For each expert e, we compute the entropy of its dispatch-weight distribution across patches:

$$
H _ {e} = - \frac {1}{B} \sum_ {b = 1} ^ {B} \sum_ {n = 1} ^ {N} p _ {b, n, e} \log p _ {b, n, e}\tag{6}
$$

where $p _ { b , n , e } = \mathbf { D } _ { b , n , e ; }$ , which already forms a distribution over patches per expert since Soft-MoE normalizes dispatch weights over the token axis $\begin{array} { r } { ( \sum _ { n } \mathbf { D } _ { b , n , e } = 1 ) } \end{array}$ . Higher entropy indicates a more uniform distribution across patches.

The entropy loss encourages high entropy (uniformity) through minimization:

$$
\mathcal {L} _ {\mathrm{entropy}} = - \sum_ {e = 0} ^ {E - 1} \lambda_ {e} \cdot H _ {e}\tag{7}
$$

where $\lambda _ { e }$ are per-expert weights. For Token+CLS, we use a symmetric weight λ for all experts; through hyperparameter search, we find $\lambda = 5 . 0$ provides stable training as default, while $\lambda = 0 . 5$ is optimal for 2 experts. Critically, optimal entropy weight is expert-count dependent: $\lambda = 0 . 5$ improves 2-expert routing but degrades 64-expert performance (Section 4.2). Token+Pixel requires asymmetric per-expert weights (supplementary Section C). Regularization is computed at the last MoE block (Block 11), validated via block selection ablation (Section 4.2). Entropy regularization and loss-coupling are opposing forces: entropy pushes dispatch weights toward uniformity (no specialization), while losscoupling pushes them toward content-dependent specialization; the coeficient λ governs the trade-of, with collapse at $\lambda = 0 . 0 1$ , stable specialization at the default, and over-regularization (degradation) at $\lambda \gg 5$ (Section 4.2).

MoE Placement and Implementation: MoE replaces MLP layers in alternating blocks {1, 3, 5, 7, 9, 11} of ViT-Base, balancing capacity (6 MoE blocks) with eficiency (6 standard blocks) at approximately 1.5× computation. Expert networks use hidden dimension 3072 with GELU activation; $E = 2$ experts increase parameters by 35% (86M→116M), while E = 64 reaches 1.86B. We use $S = 1$ slot per expert for direct expert-to-loss mapping. Full design rationale, memory analysis, and dense/sparse integration details are in the supplementary material (Sections D–G).

Loss-Coupling Mechanism: Because dispatch weights are diferentiable, gradients from each loss objective flow back through D to the router parameters Φ and s, creating a loss-coupling efect that enables learned specialization. A detach ablation confirms this as the core mechanism (Section 4.2).

Table 1: MoE loss weighting configurations. Which losses receive dispatch-weight modulation in each configuration. In sparse encoding, dispatch weights exist only for visible patches, so pixel loss (computed on masked patches) cannot be MoE-weighted.

<table><tr><td>Configuration</td><td>Encoding</td><td>Token Loss</td><td>CLS Loss</td><td>Pixel Loss</td></tr><tr><td>Token+CLS</td><td>Sparse</td><td>MoE (Exp. 0)</td><td>Global  $w_{\text{cls}}$ </td><td>—</td></tr><tr><td>Token+Pixel</td><td>Sparse</td><td>MoE (Exp. 0)</td><td>—</td><td>Uniform</td></tr></table>

Total Training Objective: Consolidating the per-objective losses (Eqs. 1– 3) under dispatch-weight coupling, the full training objective is

$$
\mathcal {L} _ {\mathrm{total}} = \sum_ {e} w _ {e} \cdot \frac {1}{| \mathcal {P} _ {e} |} \sum_ {n \in \mathcal {P} _ {e}} \mathbf {D} _ {b ^ {\star}, n, e} \cdot \mathcal {L} _ {e} (n),\tag{8}
$$

where $\mathcal { L } _ { e } ( n )$ is the per-patch loss for objective e over its patch set $\mathcal { P } _ { e } , \ w _ { e }$ the global per-objective weight, and $\mathbf { D } _ { b ^ { \star } , n , e }$ the dispatch weight from loss block $b ^ { \star }$ (Eq. 5; D ≡ 1 for un-coupled objectives). The product $\mathbf { D } _ { b ^ { \star } , n , e } \mathcal { L } _ { e } ( n )$ is the perpatch loss coeficient named in the abstract; entropy regularization (Eq. 7) is added during training.

## 4 Experiments

## 4.1 Experimental Setup

Dataset. All experiments use ImageNet-1K [9] (1.28M training images, 1000 classes).

Pretraining. ViT-Base/16 student encoder with frozen CLIP ViT-B/16 teacher, 300 epochs, batch size 4096 on 4× H100 GPUs with block masking at 40% ratio and sparse encoding. Full hyperparameters are in supplementary Tables A1–A2.

Evaluation. We evaluate across four complementary tasks: (1) k-NN (k = 20, cosine similarity) as a training-free representation quality metric; (2) Linear probe on frozen [CLS] features; (3) ImageNet finetuning with layer-wise LR decay [3]; and (4) Semantic segmentation on ADE20K [37] with UperNet [32]. MoE models use the adaptation recipes from Section 4.4. Full evaluation protocols are in supplementary Section A.3.

Model Configurations. We evaluate two multi-objective combinations, summarized in Table 1. Token+CLS is our primary configuration because it achieves the highest downstream accuracy and supports the full expert scaling range (2–64) without the inverted scaling observed in Token+Pixel at 64 experts.

The Multi-Objective Challenge. Existing global weighting methods fail on spatially heterogeneous MIM objectives. Despite an extensive hyperparameter search (supplementary Table A3), GradNorm [8] cannot match its own static baseline (72.8% vs. 74.2% k-NN for Token+CLS) and collapses entirely on Token+Pixel (32.5%), underperforming even its own dense baseline. Random loss weighting (RLW) [19] helps modestly (75.8% k-NN) but applies uniform perpatch weights, unable to capture spatial variation.

Table 2: MoE design ablations on 2-expert Token+CLS (k-NN@20). ∆ relative to baseline (λ = 5.0, Block 11).

<table><tr><td>Entropy Reg.</td><td> $\lambda=0.5$ </td><td> $\lambda=5.0$ </td><td> $\lambda=0.1$ </td><td> $\lambda=0.01$ </td></tr><tr><td>k-NN@20</td><td>75.5</td><td>75.4</td><td>75.3</td><td>66.9</td></tr><tr><td> $\Delta$ </td><td>+0.1</td><td>—</td><td>-0.1</td><td>-8.5</td></tr><tr><td colspan="5">Mech. Ablations Combine reg. Importance ( $w=2$ ). Detach weights. No entropy</td></tr><tr><td>k-NN@20</td><td>75.4</td><td>75.3</td><td>73.8</td><td>2.1</td></tr><tr><td> $\Delta$ </td><td>+0.0</td><td>-0.1</td><td>-1.6</td><td>-73.3</td></tr><tr><td>Loss Block Sel.</td><td>Block 11</td><td>Block 9</td><td>Block 7</td><td>Block 5</td></tr><tr><td>k-NN@20</td><td>75.4</td><td>75.3</td><td>75.4</td><td>75.2</td></tr><tr><td> $\Delta$ </td><td>—</td><td>-0.1</td><td>+0.0</td><td>-0.2</td></tr></table>

These failures motivate per-patch loss weighting: diferent image regions benefit from diferent loss emphasis, and global methods that apply a single scalar weight per objective cannot capture this spatial variation.

## 4.2 MoE Mechanism Validation

Having established the need for per-patch loss weighting, we validate the core design decisions of our MoE approach. Table 2 summarizes ablations on the 2-expert Token+CLS configuration.

Dispatch weights prevent degeneracy. Combine-weight loss weighting collapses to 2.1% k-NN (−73.3 points), confirming the dispatch-weight design choice (Section 3.2).

Loss-coupling is the core mechanism. We validate loss-coupling through a detach ablation: computing MoE-weighted losses with dispatch weights detached from the computation graph (i.e., treated as fixed constants during backpropagation so that loss gradients cannot update the router). This degrades k-NN by 1.6% (Table 2), confirming that the router’s ability to receive gradient signal from loss objectives is essential for learned specialization. The result is reinforced by the 2-expert unweighted variant (74.1% k-NN, Table 3), which underperforms the non-MoE baseline (75.7%) by 1.6%: MoE capacity alone is not merely insuficient but actively harmful without loss-coupling, making the coupling mechanism essential rather than incremental.

Importance loss targets the wrong distribution. Importance-based regularization (w = 2) penalizes dispatch-weight imbalance across experts, but Soft-MoE’s softmax already ensures balanced dispatch. The resulting scale parameter collapse (s : 0.90 → 0.22 at the loss block) degrades performance by 0.1%. Supplementary Section D provides visualizations of dispatch-weight patterns across all six MoE blocks, confirming spatially coherent, content-dependent routing.

## 4.3 Expert Scaling Analysis

Table 3 and Figure 3 present pretraining results across expert counts for both objective combinations. We include our MaskDistill [22] reproduction (token distillation only, dense encoding) as a reference point. For Token+CLS, starting from this MaskDistill baseline (75.6% k-NN), adding CLS alignment without MoE improves k-NN to 75.7%. Introducing 2-expert ExPLoRe with dispatchweight loss weighting reveals an instructive pattern: k-NN decreases slightly to 75.4%, but linear probe accuracy jumps to 79.6% (+3.4% over No MoE), indicating that the routing mechanism reshapes the feature space toward improved linear separability.

k-NN scales monotonically from 2 to 64 experts, reaching 76.2% at 64 experts, with smooth convergence across all configurations (Figure 3a). Linear probe, evaluated at the 2- and 64-expert endpoints, shows the same trend: 79.6% → 80.6%. Critically, comparing weighted vs. unweighted variants at the same expert count isolates loss weighting from the parameter efect (Figure 3b). The 2-expert unweighted variant (74.1%) demonstrates that loss weighting provides the +1.3% gain that makes ExPLoRe competitive. The 64-expert unweighted variant (75.3%) confirms that loss weighting contributes +0.9% at this scale. The relative gain decreases from +1.3% (2 experts) to +0.9% (64 experts) as inherent diversity from more expert parameters provides partial dispatch variation independently of loss-coupling.

For Token+Pixel: no MoE (71.5%) → 32-expert without dispatch weighting (73.0%, +1.5%) → 32-expert with dispatch weighting (73.7%, +2.2% total). Dispatch weighting adds +0.7% over the unweighted variant at 32 experts, confirming the mechanism’s efectiveness across objective combinations. This +2.2% total improvement is substantially larger than Token+CLS’s +0.5%, indicating that dispatch-weight loss weighting is particularly efective when objectives have diferent spatial characteristics.

However, Token+Pixel exhibits an inverted scaling pattern: k-NN peaks at 32 experts then drops at 64 experts (69.3%), a phenomenon not observed in Token+CLS. We attribute this to weaker gradient signals from pixel reconstruction, which may be insuficient to train 64 expert sets. MoE variants use the default pixel loss weight $( w _ { \mathrm { p i x e l } } = 1 . 0 )$ ; the non-MoE baseline uses tuned weighting $( w _ { \mathrm { p i x e l } } = 0 . 0 1 )$ ). While this 100× diference could partially explain the Token+Pixel gap, the comparison isolates the MoE contribution: dispatch weights provide implicit loss modulation that partially substitutes for manual weight tuning, and the weighted vs. unweighted comparison (+0.7% at 32 experts) controls for this confound since both use $w _ { \mathrm { p i x e l } } = 1 . 0$

Parameter cost. Expert scaling increases parameters substantially: 86M (no MoE) → 116M (2 experts, +35%) → 1.86B (64 experts, 21.6×). The weighted vs. unweighted comparisons at matched expert counts (Table 3) isolate loss weighting from the parameter efect; a detailed mechanism isolation analysis is in supplementary Section I. We view the 2-expert configuration as the practical recommendation (35% parameter increase for 79.6% linear probe), while the 64-expert scale demonstrates that dispatch-weight loss weighting improves representation quality at scale.

Table 3: Expert scaling analysis. Token+CLS scales smoothly to 64 experts (see also Figure 3a); Token+Pixel peaks at 32. All MoE models use sparse encoding, 300 epochs. W=dispatch-weight loss weighting. ⋆ Dense encoding (BEiT-style [3]). † Tuned pixel loss weight $( w _ { \mathrm { p i x e l } } = 0 . 0 1 )$

<table><tr><td colspan="7">Token + CLS MaskDistill* No MoE 2 2+W 64 64+W</td></tr><tr><td>k-NN@20</td><td>75.6</td><td>75.7</td><td>74.1</td><td>75.4</td><td>75.3</td><td>76.2</td></tr><tr><td>Linear</td><td>75.7</td><td>76.2</td><td>—</td><td>79.6</td><td>—</td><td>80.6</td></tr><tr><td rowspan="3"></td><td colspan="6">Token + Pixel No MoE† 32 32+W 64+W</td></tr><tr><td>k-NN@20</td><td>71.5</td><td>73.0</td><td>73.7</td><td>69.3</td><td></td></tr><tr><td>Linear</td><td>—</td><td>—</td><td>79.4</td><td>—</td><td></td></tr></table>

![](images/e26ec6c533497e10739548f289921cc4cc3511a226aad3099a000c0b1d9deec1.jpg)

![](images/f48811a152878b436f6bf9b98ae186fd326bcc97ed7192a637ef03034555c929.jpg)  
Fig. 3: Expert scaling and mechanism isolation (Token+CLS). (a) k-NN@20 trajectories over epochs 200–300 for No MoE and 2/16/32/64-expert configurations (all with dispatch-weight loss weighting). Stars mark peak accuracy per configuration; more experts yield higher final accuracy. (b) Mechanism isolation: weighted (W) vs. unweighted (no W) at 2 and 64 experts. Dispatch-weight loss weighting contributes +1.3% at 2 experts and +0.9% at 64 experts at matched parameter counts, confirming the mechanism’s efect beyond the parameter contribution.

Entropy Regularization and Stability Transitions. We sweep entropy regularization weight λ on the 2-expert Token+CLS configuration (Table 2). We use two diagnostics: the dispatch-weight coeficient of variation (CV) across patches, and the weighted-to-unweighted loss ratio (MoE-weighted loss divided by the same loss computed with uniform weights; 1.0 indicates neutral routing, $\ll 1 . 0$ indicates the router is redistributing weight away from dificult patches). A sharp transition exists between $\lambda = 0 . 0 1$ (collapsed, 66.9% k-NN) and λ = 0.1 (stable, 75.3%): CV jumps 32× and the loss ratio drops to 0.13, indicating deceptive loss minimization rather than meaningful specialization.

At λ = 0.5 (optimal for 2 experts), k-NN reaches 75.5% (+0.1% over the default λ = 5.0). Critically, optimal entropy weight is expert-count dependent: while λ = 0.5 helps 2 experts (+0.1%), it degrades 64-expert performance (−0.2% vs. λ = 5.0). With 64 experts, inherent diversity from more expert parameters provides natural dispatch variation, and lower regularization leads to instability rather than improved contrast.

Loss block selection. We apply dispatch-weight loss weighting at the last MoE block (Block 11). Ablations over blocks {5, 7, 9, 11} show Block 11 performs best (75.4% k-NN), with a narrow 0.2% spread across blocks, confirming that the last block captures the most task-relevant routing patterns. Multi-block fusion (combining weights from blocks 7+9+11) underperforms single-block selection, suggesting that intermediate blocks add noise rather than complementary signal.

CLS token specialization. At Block 11 (the loss block), Expert 1 concentrates 91–95% of its dispatch weight on the CLS token, efectively becoming a CLS specialist. This concentration emerges exclusively at the loss-coupled block; non-loss blocks show no such pattern. The phenomenon provides direct evidence that loss-coupling drives expert specialization: the router learns to route the CLS token heavily to the expert weighting CLS-aligned loss.

## 4.4 Downstream Transfer Recipes

While MoE loss weighting improves pretraining representations, transferring MoE models to downstream tasks is non-trivial: standard finetuning can overwrite pretrained routing patterns, and extra expert parameters increase overfitting risk. We evaluate three complementary strategies: Freeze Routing (FR) freezes router parameters (Φ, s) to preserve pretrained patch-expert assignments [39]; Freeze Attention (FA) additionally freezes attention weights [39]; and Expert Dropout (ExD) applies dropout (p= 0.4) to expert outputs [13]. Full rationale is in supplementary Section J.

Table 4 presents our recipe campaign on the Token+CLS 64-expert model, building each step on the previous best. Without any adaptation recipe, standard finetuning with unfrozen routing achieves 83.8%, as the router overwrites pretrained routing patterns with task-specific shortcuts. Freeze Routing (FR) improves to 84.2% (+0.4%) by preserving pretrained patch-expert assignments. Adding Freeze Attention (FA) yields 84.3%, a modest further gain. Adding Expert Dropout to FR provides the largest single gain (+0.7%), yielding FR+ExD at 84.9% by preventing over-reliance on dominant experts. The full FR+FA+ExD recipe reaches 85.3%, a +1.5% improvement over vanilla MoE finetuning (83.8%), exceeding the non-MoE baseline (84.8%) by +0.5%.

The recipe transfers across objective combinations and expert counts: the Token+Pixel 32-expert model improves from 83.6% to 84.8% with FR+FA+ExD (+1.2%), exceeding the Token+Pixel non-MoE baseline (84.6%). The 2-expert Token+CLS model with FR alone achieves 84.1%, below the non-MoE baseline (84.8%), indicating that the full recipe stack is needed to overcome the additional overfitting risk from expert parameters even at the 2-expert scale.

Table 4: Downstream adaptation recipes. FR=Freeze Routing, FA=Freeze Attention, ExD=Expert Dropout $( p = 0 . 4 )$ . Top: Token+CLS 64-expert recipe ablation. Bottom: cross-configuration transfer.

<table><tr><td>Configuration</td><td colspan="3">FR FA ExD Top-1</td></tr><tr><td colspan="4">Token+CLS 64-Expert</td></tr><tr><td>Vanilla (unfrozen routing)</td><td></td><td></td><td>83.8</td></tr><tr><td>FR only</td><td>√</td><td></td><td>84.2</td></tr><tr><td>FR+FA</td><td>√</td><td>√</td><td>84.3</td></tr><tr><td>FR+ExD</td><td>√</td><td></td><td>84.9</td></tr><tr><td>FR+FA+ExD</td><td>√</td><td>√</td><td>85.3</td></tr><tr><td colspan="4">Cross-Configuration Transfer</td></tr><tr><td>Token+CLS (no MoE)</td><td>—</td><td>—</td><td>84.8</td></tr><tr><td>Token+CLS 2-expert (FR)</td><td>√</td><td></td><td>84.1</td></tr><tr><td>Token+Pixel 32-expert (Vanilla)</td><td></td><td></td><td>83.6</td></tr><tr><td>Token+Pixel 32-expert (FR+FA+ExD)</td><td>√</td><td>√</td><td>84.8</td></tr><tr><td>Token+Pixel (no MoE)</td><td>—</td><td>—</td><td>84.6</td></tr></table>

Table 5: Semantic segmentation on ADE20K (UperNet, 160K iters, 32- expert). T+P=Token+Pixel, T+C=Token+CLS. Def=default finetuning (no recipe), W=dispatch-weight loss weighting, All=FR+FA+ExD.

<table><tr><td>T+P</td><td>No MoE</td><td>Def</td><td>Def+W</td><td>FR+W</td><td>All+W</td></tr><tr><td>mIoU</td><td>52.5</td><td>49.6</td><td>50.1</td><td>50.8</td><td>52.8</td></tr></table>

<table><tr><td colspan="6">T+C No MoE Def Def+W FR+W All+W</td></tr><tr><td>mIoU</td><td>50.8</td><td>48.3</td><td>48.9</td><td>50.3</td><td>51.1</td></tr></table>

Semantic Segmentation Table 5 evaluates semantic segmentation on ADE20K for both objective combinations at 32 experts, enabling controlled comparison at a common scale (the optimal expert count for Token+Pixel, Table 3). MoE models present a well-known challenge for downstream transfer [13, 39]: routing patterns optimized during pretraining can conflict with the uniform spatial coverage required for dense prediction. Without adaptation recipes, our MoE models underperform non-MoE baselines by 2.5–2.9 mIoU, consistent with prior findings that MoE finetuning is non-trivial [39]. Even in this default setting, dispatch-weight loss weighting helps modestly: +0.5 mIoU (Token+Pixel) and +0.6 mIoU (Token+CLS) over unweighted variants.

The adaptation recipes progressively close this gap. Freeze Routing (FR) alone recovers 0.7–1.4 mIoU by preserving pretrained routing patterns. The full FR+FA+ExD recipe fully closes the gap, with MoE models reaching parity or slight improvement over non-MoE baselines: Token+Pixel reaches 52.8 mIoU (+0.3 over non-MoE 52.5), and Token+CLS reaches 51.1 (+0.3 over non-MoE 50.8). This recovery arc, from MoE hurting dense prediction to matching baselines with proper adaptation, demonstrates that the recipes are essential for realizing MoE benefits beyond classification.

Table 6: Comparison with published methods on ImageNet-1K. All use ViT-Base encoders. GFLOPs=inference cost (analytical MACs at N = 197 tokens, crossvalidated against fvcore within 1.1%). Lin.=linear probe top-1, FT=finetuning top-1, Seg.=ADE20K mIoU. “—” = not reported. ∗ Our reproduction with sparse Token+CLS encoding; Lin./FT are ours, Seg. from [22]. † 400 epochs. ‡ YFCC-15M pretrained. § Token+Pixel 32-exp with FR+FA+ExD (Section 4.4).

<table><tr><td>Method</td><td>Teacher</td><td>Params</td><td>GFLOPs</td><td>Ep.</td><td>Lin.</td><td>FT</td><td>Seg.</td></tr><tr><td>MAE [15]</td><td>None</td><td>86M</td><td>17.45</td><td>1600</td><td>68.0</td><td>83.6</td><td>48.1</td></tr><tr><td>SdAE [7]</td><td>Self</td><td>86M</td><td>17.45</td><td>300</td><td>—</td><td>84.1</td><td>48.6</td></tr><tr><td>BootMAE [11]</td><td>Self</td><td>86M</td><td>17.45</td><td>800</td><td>—</td><td>83.6</td><td>—</td></tr><tr><td>BEiT [3]</td><td>DALL-E</td><td>86M</td><td>17.45</td><td>800</td><td>56.7</td><td>83.0</td><td>45.6</td></tr><tr><td>MVP [31]</td><td>CLIP-B</td><td>86M</td><td>17.45</td><td>300</td><td>75.4</td><td>84.4</td><td>52.4</td></tr><tr><td>MaskDistill* (repr.) [22]</td><td>CLIP-B</td><td>86M</td><td>17.45</td><td>300</td><td>76.2</td><td>84.8</td><td>53.8</td></tr><tr><td>BEiT v2 [21]</td><td>CLIP-B</td><td>86M</td><td>17.45</td><td>300</td><td>80.1</td><td>85.0</td><td>52.7</td></tr><tr><td> $MILAN^†$  [16]</td><td>CLIP-B</td><td>86M</td><td>17.45</td><td>400</td><td>79.9</td><td>85.4</td><td>52.7</td></tr><tr><td>CAE v2 [36]</td><td>CLIP-B</td><td>86M</td><td>17.45</td><td>300</td><td>80.5</td><td>85.3</td><td>52.9</td></tr><tr><td> $MaskCLIP^‡$  [12]</td><td>CLIP-B</td><td>86M</td><td>17.45</td><td>25</td><td>73.7</td><td>83.6</td><td>50.5</td></tr><tr><td>ExPLoRe (ours, 2-exp)</td><td>CLIP-B</td><td>116M</td><td>11.93</td><td>300</td><td>79.6</td><td>84.1</td><td>—</td></tr><tr><td>ExPLoRe (ours, 64-exp)</td><td>CLIP-B</td><td>1.86B</td><td>13.86</td><td>300</td><td>80.6</td><td>85.3</td><td>52.8§</td></tr></table>

Comparison with Published Methods Table 6 compares ExPLoRe against published methods using ViT-Base encoders. Our 2-expert model (116M) achieves 79.6% linear probe, +3.4% over MaskDistill (76.2%). Scaling to 64 experts reaches 80.6% linear probe, the highest among compared methods, with 85.3% finetuning accuracy matching CAE v2. Semantic segmentation with the full recipe reaches 52.8% mIoU (Token+Pixel 32-expert, Table 5), competitive with BEiT v2 (52.7%) and MILAN (52.7%). MaskDistill’s 53.8% segmentation is from the original paper [22]; our controlled baselines in Table 5 provide the direct comparison.

Although the 64-expert model has 1.86B parameters, Soft-MoE with one slot per expert decouples parameters from inference cost: each MoE layer routes all tokens into only E expert slots, so the expert MLPs process E rather than N inputs and only the routing projections scale with E. Both ExPLoRe configurations therefore use fewer inference GFLOPs than every ViT-B/16 comparator (11.93 and 13.86 vs. 17.45; Table 6). Per-FLOP, ExPLoRe is favorably positioned: E=2 trades −32% cost for a −0.9% linear-probe gap to CAE v2, and E=64 matches CAE v2 at −21% cost (cost–quality plot in the supplementary).

Robustness and Additional Probing Protocols Seed and mask-ratio robustness. Across two additional seeds and a 50% mask ratio, the 2-expert Token+CLS model varies tightly: standard deviation 0.078 (k-NN), 0.037 (linear probe), 0.039 (Eficient Probing), with mask=50% within this band (40% is the established block-masking optimum [3,22]). Our mechanism-isolation deltas (+1.3% weighted vs. unweighted, −1.6% detach) compare configurations sharing pretraining randomness and are not single-run artifacts; full per-seed numbers are in the supplementary.

Eficient Probing and fine-grained transfer. Under Eficient Probing (EP) [23], a lightweight attentive protocol recovering patch-level information that CLSaggregated probes under-measure, the 2-expert model reaches 80.19% vs. 78.77% for the non-MoE baseline $( + 1 . 4 1 \% , \sim 3 \times$ the k-NN gain on the same backbones). On fine-grained Food-101 [4], EP gives 89.55% vs. 87.68% (+1.87%), indicating the routing emphasis transfers beyond ImageNet.

## 5 Conclusion

We presented ExPLoRe, a framework that repurposes Soft-MoE dispatch weights as learned per-patch loss coeficients for multi-objective masked image modeling. Three findings emerge from our study. (1) Loss-coupling drives specialization: detaching this connection degrades performance by 1.6%, and MoE without loss-coupling underperforms the non-MoE baseline (74.1% vs. 75.7%), confirming the mechanism is essential. While absolute k-NN gains are modest (+0.5% for Token+CLS, +2.2% for Token+Pixel), the +4.4% linear probe improvement suggests loss-coupled routing reshapes the feature space in ways not fully captured by nearest-neighbor evaluation. (2) Routing dynamics exhibit sharp transitions: entropy regularization governs a sharp transition between stable routing and catastrophic collapse, with the optimal weight being expertcount dependent (λ = 0.5 for 2 experts, λ = 5.0 for 64). (3) MoE transfer requires dedicated recipes: our FR+FA+ExD recipe improves MoE finetuning by +1.5% over vanilla MoE (83.8% → 85.3%), exceeding the non-MoE baseline (84.8%). Without recipes, MoE models underperform on segmentation by 2.5–2.9 mIoU; the full recipe closes this gap (+0.3 mIoU for both objective combinations).

Limitations and Future Work. Our experiments use ViT-Base with up to 64 experts (GPU-memory bounded) and a frozen CLIP teacher, though the mechanism is teacher-agnostic and applies to any multi-objective MIM framework. Natural extensions include a dedicated loss-weighting router separate from the feature router (enabling top-k routing across all experts per loss), multiple slots per expert (S >1), learned block aggregation of dispatch weights across MoE layers, and scaling to larger models and longer schedules.

## References

1. Baevski, A., Babu, A., Hsu, W.N., Auli, M.: data2vec 2.0: Highly eficient selfsupervised learning for vision, speech and language. In: International Conference on Machine Learning. pp. 1694–1714 (2023)

2. Baevski, A., Hsu, W.N., Xu, Q., Babu, A., Gu, J., Auli, M.: data2vec: A general framework for self-supervised learning in speech, vision and language. In: International Conference on Machine Learning. pp. 1298–1312 (2022)

3. Bao, H., Dong, L., Piao, S., Wei, F.: Beit: Bert pre-training of image transformers. In: International Conference on Learning Representations (2022)

4. Bossard, L., Guillaumin, M., Van Gool, L.: Food-101 – mining discriminative components with random forests. In: European Conference on Computer Vision (ECCV) (2014)

5. Chen, T., Chen, X., Du, X., Rashwan, A., Yang, F., Chen, H., Wang, Z., Li, Y.: Adamv-moe: Adaptive multi-task vision mixture-of-experts. In: IEEE/CVF International Conference on Computer Vision (ICCV). pp. 17346–17357 (2023)

6. Chen, X., Ding, M., Wang, X., Xin, Y., Mo, S., Wang, Y., Han, S., Luo, P., Zeng, G., Wang, J.: Context autoencoder for self-supervised representation learning. International Journal of Computer Vision 132(1), 208–223 (2024)

7. Chen, Y., Liu, Y., Jiang, D., Zhang, X., Dai, W., Xiong, H., Tian, Q.: Sdae: Selfdistillated masked autoencoder. In: European Conference on Computer Vision. pp. 108–124 (2022)

8. Chen, Z., Badrinarayanan, V., Lee, C.Y., Rabinovich, A.: Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In: International Conference on Machine Learning. pp. 794–803 (2018)

9. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: 2009 IEEE Conference on Computer Vision and Pattern Recognition. pp. 248–255 (2009)

10. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics. pp. 4171–4186 (2019)

11. Dong, X., Bao, J., Zhang, T., Chen, D., Zhang, W., Yuan, L., Chen, D., Wen, F., Yu, N.: Bootstrapped masked autoencoders for vision bert pretraining. In: European Conference on Computer Vision. pp. 247–264 (2022)

12. Dong, X., Bao, J., Zheng, Y., Zhang, T., Chen, D., Yang, H., Zeng, M., Zhang, W., Yuan, L., Chen, D., et al.: Maskclip: Masked self-distillation advances contrastive language-image pretraining. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10995–11005 (2023)

13. Fedus, W., Zoph, B., Shazeer, N.: Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity. Journal of Machine Learning Research 23(120), 1–39 (2022)

14. Han, X., Wei, L., Dou, Z., Wang, Z., Qiang, C., He, X., Sun, Y., Han, Z., Tian, Q.: Vimoe: An empirical study of designing vision mixture-of-experts. arXiv preprint arXiv:2410.15732 (2024)

15. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16000–16009 (2022)

16. Hou, Z., Sun, F., Chen, Y.K., Xie, Y., Kung, S.Y.: Milan: Masked image pretraining on language assisted representation. arXiv preprint arXiv:2208.06049 (2022)

17. Jiang, Z., Zheng, G., Cheng, Y., Awadallah, A.H., Wang, Z.: Cr-moe: Consistent routed mixture-of-experts for scaling contrastive learning. Transactions on Machine Learning Research (2024)

18. Kendall, A., Gal, Y., Cipolla, R.: Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 7482–7491 (2018)

19. Lin, B., Ye, F., Zhang, Y., Tsang, I.W.: Reasonable efectiveness of random weighting: A litmus test for multi-task learning. Transactions on Machine Learning Research (2022), arXiv:2111.10603

20. Liu, T., Blondel, M., Riquelme, C., Puigcerver, J.: Routers in vision mixture of experts: An empirical study. Transactions on Machine Learning Research (TMLR) (2024)

21. Peng, Z., Dong, L., Bao, H., Ye, Q., Wei, F.: Beit v2: Masked image modeling with vector-quantized visual tokenizers. arXiv preprint arXiv:2208.06366 (2022)

22. Peng, Z., Dong, L., Bao, H., Ye, Q., Wei, F.: A unified view of masked image modeling. arXiv preprint arXiv:2210.10615 (2022)

23. Psomas, B., Christopoulos, D., Baltzi, E., Kakogeorgiou, I., Aravanis, T., Komodakis, N., Karantzalos, K., Avrithis, Y., Tolias, G.: Attention, please! revisiting attentive probing through the lens of eficiency. In: International Conference on Learning Representations (ICLR) (2026)

24. Puigcerver, J., Riquelme, C., Mustafa, B., Houlsby, N.: From sparse to soft mixtures of experts. In: International Conference on Learning Representations (ICLR) (2024)

25. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning. pp. 8748–8763 (2021)

26. Riquelme, C., Puigcerver, J., Mustafa, B., Neumann, M., Jenatton, R., Susano Pinto, A., Keysers, D., Houlsby, N.: Scaling vision with sparse mixture of experts. In: Advances in Neural Information Processing Systems (NeurIPS) (2021)

27. Rousseeuw, P.J.: Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics 20, 53–65 (1987)

28. Sener, O., Koltun, V.: Multi-task learning as multi-objective optimization. In: Advances in Neural Information Processing Systems. vol. 31 (2018)

29. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J.: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In: International Conference on Learning Representations (ICLR) (2017)

30. Wei, C., Fan, H., Xie, S., Wu, C.Y., Yuille, A., Feichtenhofer, C.: Masked feature prediction for self-supervised visual pre-training. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14668–14678 (2022)

31. Wei, L., Xie, L., Zhou, W., Li, H., Tian, Q.: Mvp: Multimodality-guided visual pre-training. arXiv preprint arXiv:2203.05175 (2022)

32. Xiao, T., Liu, Y., Zhou, B., Jiang, Y., Sun, J.: Unified perceptual parsing for scene understanding. In: European Conference on Computer Vision. pp. 418–434 (2018)

33. Xie, Z., Zhang, Z., Cao, Y., Lin, Y., Bao, J., Yao, Z., Dai, Q., Hu, H.: Simmim: A simple framework for masked image modeling. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9653–9663 (2022)

34. Yang, X., Lu, J., Qiu, H., Li, S., Li, H.: Astrea: A moe-based visual understanding model with progressive alignment. arXiv preprint arXiv:2503.09445 (2025)

35. Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., Finn, C.: Gradient surgery for multi-task learning. In: Advances in Neural Information Processing Systems. vol. 33, pp. 5824–5836 (2020)

36. Zhang, X., Yuan, J., Wei, X., Wei, Y., Hong, S., Wang, J.: Cae v2: Context autoencoder with clip target. arXiv preprint arXiv:2211.09799 (2024)

37. Zhou, B., Zhao, H., Puig, X., Fidler, S., Barber, A., Torralba, A.: Scene parsing through ade20k dataset. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 633–641 (2017)

38. Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., Kong, T.: ibot: Image bert pre-training with online tokenizer. In: International Conference on Learning Representations (2022)

39. Zoph, B., Bello, I., Kumar, S., Du, N., Huang, Y., Dean, J., Shazeer, N., Fedus, W.: St-moe: Designing stable and transferable sparse expert models. arXiv preprint arXiv:2202.08906 (2022)

## A Training and Evaluation Hyperparameters

## A.1 Pretraining Configuration

Table A1 summarizes the pretraining hyperparameters shared across all model configurations. All experiments use identical optimization settings to ensure fair comparison; only the loss objectives and MoE configuration vary between runs.

Table A1: Pretraining hyperparameters. Shared across all configurations.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td colspan="2">Architecture</td></tr><tr><td>Encoder</td><td>ViT-Base/16 (86M params)</td></tr><tr><td>Teacher</td><td>Frozen CLIP ViT-B/16</td></tr><tr><td>Decoder</td><td>8 blocks, dim 512</td></tr><tr><td>Position embeddings (encoder)</td><td>Absolute (learnable)</td></tr><tr><td>Position embeddings (decoder)</td><td>Learnable</td></tr><tr><td colspan="2">Masking</td></tr><tr><td>Strategy</td><td>Block masking</td></tr><tr><td>Mask ratio</td><td>40%</td></tr><tr><td>Patch shuffling</td><td>Enabled (sparse mode)</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>Optimizer</td><td>AdamW ( $\beta_1=0.9$ ,  $\beta_2=0.95$ )</td></tr><tr><td>Weight decay</td><td>0.05</td></tr><tr><td>Peak learning rate</td><td> $1.5 \times 10^{-3}$ </td></tr><tr><td>Minimum learning rate</td><td> $1 \times 10^{-6}$ </td></tr><tr><td>Schedule</td><td>Cosine decay</td></tr><tr><td>Warmup epochs</td><td>40</td></tr><tr><td>Total epochs</td><td>300</td></tr><tr><td>Batch size</td><td>4096 (across 4× H100 GPUs)</td></tr><tr><td>Gradient clipping</td><td>3.0</td></tr><tr><td>Mixed precision</td><td>FP16</td></tr><tr><td>Seed</td><td>42</td></tr></table>

## A.2 MoE Configuration Details

Table A2 details the MoE-specific parameters for each model configuration evaluated in the main paper.

## A.3 Downstream Evaluation Protocols

k-NN Classification. We use k = 20 nearest neighbors with cosine similarity on [CLS] token representations. Features are extracted from the last encoder block without any learned components, providing a training-free measure of representation quality. We evaluate checkpoints at epochs {100, 150, 180, 200, 220, 240, 250, 260, 270, 280, 290, 295, 300} and report the best.

Table A2: MoE configuration details. Parameters specific to each ExPLoRe variant.

<table><tr><td>Parameter</td><td>Token+CLS</td><td>Token+Pixel</td></tr><tr><td colspan="3">Objectives</td></tr><tr><td>Token distillation</td><td>√</td><td>√</td></tr><tr><td>CLS alignment (w=0.4)</td><td>√</td><td>-</td></tr><tr><td>Pixel reconstruction</td><td>-</td><td>√</td></tr><tr><td colspan="3">MoE Architecture</td></tr><tr><td>MoE placement</td><td>Alternating (blocks 1,3,5,7,9,11)</td><td>Same</td></tr><tr><td>Slots per expert</td><td>1</td><td>1</td></tr><tr><td>Routing type</td><td>Soft (fully differentiable)</td><td>Same</td></tr><tr><td>Scale init</td><td>1.0 (learned)</td><td>Same</td></tr><tr><td>Expert init</td><td>Kaiming uniform</td><td>Same</td></tr><tr><td colspan="3">Regularization</td></tr><tr><td>Entropy weight (dispatch)</td><td>5.0 (default)</td><td>5.0</td></tr><tr><td>Loss weighting block</td><td>Block 11 (last MoE)</td><td>Same</td></tr><tr><td>Per-image normalization</td><td>√</td><td>√</td></tr><tr><td>Loss weighting method</td><td>Dispatch weights</td><td>Dispatch weights</td></tr></table>

Linear Probing. A single linear layer is trained on frozen [CLS] token features for 100 epochs with batch size 2048, learning rate $2 \times 1 0 ^ { - 3 }$ with cosine decay, 10-epoch warmup, label smoothing 0.1, mixup 0.8, cutmix 1.0, and RandAugment (9, 0.5).

ImageNet Finetuning. Full model finetuning for 100 epochs with layerwise learning rate decay (base $5 \times 1 0 ^ { - 4 }$ , decay factor 0.65), batch size 1024, AdamW with weight decay 0.05, warmup 5 epochs, mixup 0.8, cutmix 1.0, drop path 0.1. MoE models use the FR+FA+ExD recipe (Section 4.4 of the main paper) unless noted otherwise.

Semantic Segmentation. UperNet [32] on ADE20K with 160K iterations, learning rate $8 \times 1 0 ^ { - 5 }$ , layer-wise decay 0.85, batch size 16. MoE models freeze routing parameters and use expert dropout $\left( p = 0 . 4 \right)$

## B GradNorm Hyperparameter Search Details

Table A3 presents the complete GradNorm hyperparameter search across both objective combinations. For Token+CLS (dense encoding), we explored $\alpha \in$ {0.12, 0.5, 1.0, 1.5} and learning rates $\in [ 1 0 ^ { - 4 } , 0 . 1 5 ]$ across 5 configurations; the best result (72.8% k-NN) substantially underperforms the static baseline (74.2%). For Token+Pixel (dense encoding), we tested 12 configurations spanning α ∈ {0.0, 0.5, 1.0} and learning rates ∈ [0.00125, 0.025]; all configurations exhibited training instability and collapsed to ≤32.5% k-NN, far below the non-MoE baseline (71.5%). The consistent failure across diverse hyperparameters confirms that GradNorm’s global gradient-based balancing is fundamentally incompatible with spatially heterogeneous MIM objectives, motivating the per-patch approach of ExPLoRe.

Table A3: Complete GradNorm hyperparameter search. Dense encoding throughout. All Token+CLS configurations underperform the static baseline (74.2% k-NN). All Token+Pixel configurations collapse. Representative results from 12 Token+Pixel configurations shown.

<table><tr><td>α</td><td>Learning Rate</td><td>Objectives</td><td>k-NN@20</td><td>Status</td></tr><tr><td colspan="5">Token + CLS (baseline: 74.2%)</td></tr><tr><td>0.12</td><td>0.15</td><td>Token + CLS</td><td>72.8</td><td>Best</td></tr><tr><td>0.12</td><td>0.0001</td><td>Token + CLS</td><td>69.5</td><td></td></tr><tr><td>1.0</td><td>0.025</td><td>Token + CLS</td><td>68.7</td><td></td></tr><tr><td>1.5</td><td>0.025</td><td>Token + CLS</td><td>68.3</td><td></td></tr><tr><td>0.5</td><td>0.00015</td><td>Token + CLS</td><td>66.4</td><td>Worst</td></tr><tr><td colspan="5">Token + Pixel (baseline: 71.5%)</td></tr><tr><td>1.0</td><td>0.025</td><td>Token + Pixel</td><td>32.5</td><td>Best</td></tr><tr><td>0.5</td><td>0.025</td><td>Token + Pixel</td><td>30.8</td><td></td></tr><tr><td>0.0</td><td>0.00125</td><td>Token + Pixel</td><td>28.3</td><td></td></tr></table>

## C Entropy Regularization Analysis

We conducted a grid search to identify optimal entropy regularization weights for the 2-expert Token+Pixel configuration (dense encoding). For Expert 0 (token distillation) we explored $\lambda _ { 0 } \in \{ 0 . 1 , 0 . 5 , 1 . 0 , 2 . 0 , 3 . 0 , 5 . 0 \}$ , and for Expert 1 (pixel reconstruction) we explored $\lambda _ { 1 } \in \{ 5 . 0 , 1 0 . 0 , 1 5 . 0 , 2 0 . 0 , 2 4 . 0 , 3 0 . 0 \}$

The optimal configuration uses asymmetric weights: $\lambda _ { 0 } = 5 . 0 , \lambda _ { 1 } = 2 4 . 0$ The 4.8× asymmetry compensates for weaker gradient signals from pixel reconstruction compared to semantic distillation. Configurations with $\lambda _ { 1 } < 2 0 . 0$ consistently exhibited expert collapse, with Expert 1 receiving <10% of tokens. For the Token+CLS configuration (sparse encoding), symmetric weights $( \lambda { = } 5 . 0$ for both experts) are suficient since both objectives involve semantic features.

## D Expert Routing Visualizations

This section provides visual and quantitative evidence that loss-coupled dispatch weights develop content-dependent expert specialization. We compare the coupled model (ExPLoRe with $\lambda _ { \mathrm { e n t r o p y } } = 0 . 5$ , the best 2-expert configuration from Section 4.2) against the detach ablation (trained with identical $\lambda _ { \mathrm { e n t r o p y } } = 0 . 5 )$ ， which uses identical dispatch weights for the MoE forward pass but detaches gradients before loss weighting, removing the loss-coupling signal.

## D.1 Dispatch Weight Heatmaps

Figure A1 visualizes per-patch dispatch weights for four natural images. In the coupled model, Expert 0 shows spatially coherent activation regions aligned with salient image content (e.g., the dog’s body, the horse, the bird), while Expert 1 exhibits complementary patterns. In the detach model, both experts show more scattered, less content-aligned activation patterns. Per-model normalization is applied so that each coupled/detach pair uses its full color range, enabling fair comparison of spatial structure rather than absolute magnitude.

![](images/8a4d4200d66dbc80a109f635dd5004568f1430a58ecf4f57a7af249af9bb13ed.jpg)  
Block 9 dispatch weights  
Fig. A1: Dispatch weight heatmaps. Each row shows a diferent input image. Left pair: Coupled (ExPLoRe) Expert 0 and Expert 1 dispatch weights. Right pair: Detach ablation. The coupled model produces spatially coherent, content-dependent routing; the detach model shows more scattered patterns. Dashed line separates the two models. Per-model color normalization.

## D.2 Expert Cluster Separation Across Blocks

To quantify routing specialization, we compute the silhouette coeficient [27] (ranging from −1 to +1, where higher values indicate better-separated clusters) at each MoE block, using per-token input embeddings as features and argmax(dispatch weights) as cluster labels.

Figure A2 shows that both models exhibit similar, modest specialization at early blocks (1–5), consistent with generic feature extraction. The models diverge at later blocks: at Block 9, the coupled model reaches a silhouette of 0.17 vs. 0.15 for detach. At Block 11 (the loss block), the divergence is dramatic: the coupled model achieves 0.505 while the detach model collapses to 0.0 (all tokens assigned to a single expert). This confirms that loss-coupling drives expert specialization specifically at the loss block. Without gradient flow through dispatch weights, the entropy regularization pushes the router toward uniformity with no counteracting signal, eliminating specialization entirely.

![](images/d05ddeb02b7d4677d763bb0ecb9a74db278547da3d95270343b2769050717e96.jpg)  
Fig. A2: Silhouette coeficient across MoE blocks. Dispatch-based cluster assignments on 200 ImageNet validation images (50K tokens subsampled). The coupled model develops strong expert specialization at Block 11 (the loss block), while the detach ablation collapses to zero specialization at this block. Early blocks show similar routing structure in both models.

## D.3 Dispatch Weight–Loss Correlation

To understand how the coupled router specializes, we examine the relationship between Expert 0’s dispatch weight and per-token distillation loss at Block 11. Figure A3 shows scatter plots for both models. In the coupled model, there is a clear negative Pearson correlation (r = −0.601): patches receiving higher Expert 0 dispatch weight tend to have lower distillation loss, indicating that the router learns to upweight patches where it can reduce loss most efectively. In the detach model, this correlation nearly vanishes (r =−0.191), confirming that loss-coupling is necessary for the router to develop loss-relevant specialization.

![](images/8c50b7350f7ed1af1cd8fcfce3cc07bc0646b9a80ab7cc05e3b92e8ca4a72c72.jpg)  
Fig. A3: Dispatch weight vs. per-token distillation loss at Block 11. Each point is one token from 10K subsampled tokens across ImageNet validation images. Left: Coupled model shows strong negative correlation $( r = - 0 . 6 0 1 )$ , indicating the router upweights patches where it reduces loss. Right: Detach ablation shows weak correlation $( r = - 0 . 1 9 1 )$ . Dashed line: linear regression (clipped at $y = 0 )$

## E Encoding and Decoder Details

This section provides the full encoding equations summarized in Section 3.1 of the main paper, along with decoder architecture details.

Dense Encoding (BEiT-style). Following BEiT [3], masked patches are replaced with learnable mask tokens $\mathbf { e } _ { \mathrm { m a s k } } \in \mathbb { R } ^ { d }$ , processing the complete sequence of N patches:

$$
\mathbf {x} _ {\mathrm{dense}} = \mathbf {x} _ {\mathrm{vis}} \odot (1 - \mathbf {m}) + \mathbf {e} _ {\mathrm{mask}} \odot \mathbf {m}\tag{A1}
$$

where $\odot$ denotes element-wise multiplication. Dense encoding yields dispatch weights for all patches including masked ones, enabling MoE-weighted pixel reconstruction.

Sparse Encoding (MAE-style). Following MAE [15], only visible patches are processed, reducing computation by factor (1 − r) where r is the mask ratio. Position embeddings are added before masking to preserve spatial information:

$$
\mathbf {x} _ {\mathrm{pos}} = \mathbf {x} + \mathbf {p} _ {\mathrm{emb}}\tag{A2}
$$

where $\mathbf { p } _ { \mathrm { e m b } } \in \mathbb { R } ^ { N \times d }$ are position embeddings. This pre-masking addition is crucial: once patches are masked and shufled, their original spatial positions would be lost without embedded positional information. To prevent the model from exploiting positional regularities in the visible subset, patches undergo random shufling:

$$
\pi = \mathrm{randperm} (N), \quad \mathbf {x} _ {\mathrm{shuffle}} = \mathbf {x} _ {\mathrm{pos}} [ \pi ]\tag{A3}
$$

$$
\mathbf {x} _ {\mathrm{sparse}} = \mathbf {x} _ {\mathrm{shuffle}} [ \neg \mathbf {m} [ \pi ] ]\tag{A4}
$$

Without shufling, the encoder could learn spurious correlations $( e . g .$ ., “patches at positions 0–49 are always visible for 75% masking”). The inverse permutation $\pi ^ { - \bar { 1 } }$ is stored for decoder reconstruction.

Loss Application by Encoding Mode:

– Dense: Token distillation on masked positions $\mathcal { L } _ { \mathrm { t o k e n } } ( \hat { \mathbf { t } } [ \mathbf { m } ] , \mathbf { t } [ \mathbf { m } ] )$ ), pixel reconstruction on masked positions $\mathcal { L } _ { \mathrm { p i x e l } } ( \hat { \mathbf { x } } [ \mathbf { m } ] , \mathbf { x } [ \mathbf { m } ] )$

– Sparse: Token distillation on visible positions $\mathcal { L } _ { \mathrm { t o k e n } } ( \hat { \mathbf { t } } [ \mathbf { \bar { \tau } } \mathbf { \cdot } \mathbf { m } ] , \mathbf { t } [ \mathbf { \bar { \tau } } \mathbf { \cdot } \mathbf { m } ] )$ , pixel reconstruction on masked positions $\mathcal { L } _ { \mathrm { p i x e l } } ( \hat { \mathbf { x } } [ \mathbf { m } ] , \mathbf { x } [ \mathbf { m } ] )$

The encoding mode determines which patches receive MoE dispatch weights and thus which losses can be MoE-weighted.

Decoder Architecture. The decoder is a lightweight 8-block transformer with embedding dimension 512 and 16 attention heads. In sparse mode, the decoder receives the encoder’s visible-patch representations concatenated with learnable mask tokens at the masked positions, using the stored inverse permutation $\pi ^ { - 1 }$ to restore spatial ordering. Learnable position embeddings are added before decoding. A linear projection maps decoder outputs to the pixel target dimension $( 1 6 \times 1 6 \times 3 = 7 6 8 )$ for reconstruction. The decoder is discarded after pretraining and is not used during downstream evaluation.

## F Notation Reference

Table A4 provides a compact reference for the notation used throughout the paper.

Table A4: Notation reference.

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\mathbf{X} \in \mathbb{R}^{B \times N \times d}$ </td><td>Patch representations (batch, patches, dimension)</td></tr><tr><td> $\boldsymbol{\Phi} \in \mathbb{R}^{d \times E}$ </td><td>Expert routing parameters</td></tr><tr><td>s</td><td>Learned routing scale parameter</td></tr><tr><td>E</td><td>Number of experts</td></tr><tr><td> $\mathbf{L} \in \mathbb{R}^{B \times N \times E}$ </td><td>Routing logits (patch-expert affinity)</td></tr><tr><td> $\mathbf{D} \in \mathbb{R}^{B \times N \times E}$ </td><td>Dispatch weights (softmax over patches)</td></tr><tr><td> $\mathbf{C} \in \mathbb{R}^{B \times N \times E}$ </td><td>Combine weights (softmax over experts)</td></tr><tr><td> $\mathcal{V}, \mathcal{M}$ </td><td>Visible and masked patch sets</td></tr><tr><td> $\pi, \pi^{-1}$ </td><td>Shuffle permutation and its inverse</td></tr><tr><td> $\lambda, \lambda_e$ </td><td>Entropy regularization weight (global / per-expert)</td></tr><tr><td> $H_e$ </td><td>Per-expert dispatch entropy</td></tr><tr><td> $w_{\text{cls}}, w_{\text{pixel}}$ </td><td>CLS alignment / pixel reconstruction loss weights</td></tr></table>

## G Detailed Soft-MoE Routing Derivation

This section provides the full step-by-step derivation of the Soft-MoE routing mechanism summarized in Section 3.2 of the main paper. We follow the formulation of Puigcerver et al. [24].

For a given MoE layer, we maintain E learnable expert parameters $\Phi \in \mathbb { R } ^ { d \times E }$ that define what type of content each expert responds to, along with E expert networks each implemented as a standard MLP: $\mathrm { M L P } ( x ) = W _ { 2 } \cdot \mathrm { G E L U } ( W _ { 1 } \cdot x )$ 2 where $W _ { 1 } \ \in \ \mathbb { R } ^ { d \times 4 d }$ and $W _ { 2 } ~ \in ~ \mathbb { R } ^ { 4 d \times d }$ . Each expert has independent parameters, allowing specialization through training. A learned scalar s controls the sharpness of the routing distribution, similar to temperature in softmax.

The routing mechanism operates as follows. Given patch representations $\mathbf { X \in }$ <sup>RB×N×d</sup> from the attention layer:

(1) Logits measure patch-expert afinity through normalized similarity:

$$
\mathbf {L} = s \cdot \frac {\mathbf {X}}{\| \mathbf {X} \| _ {2}} \cdot \frac {\boldsymbol {\Phi}}{\| \boldsymbol {\Phi} \| _ {2}}\tag{A5}
$$

where L2 normalization stabilizes training and prevents logit saturation.

(2) Dispatch weights via softmax over the patch dimension aggregate patches to experts:

$$
\mathbf {D} = \mathrm{softmax} _ {\mathrm{dim=1}} (\mathbf {L}) \in \mathbb {R} ^ {B \times N \times E}\tag{A6}
$$

where $\begin{array} { r } { \sum _ { n } \mathbf { D } _ { b , n , e } = 1 } \end{array}$ for each expert e. These weights represent how much each patch contributes to each expert and are used for patch-level loss weighting.

(3) Combine weights via softmax over the expert dimension determine output composition:

$$
\mathbf {C} = \mathrm{softmax} _ {\mathrm{dim=2}} (\mathbf {L}) \in \mathbb {R} ^ {B \times N \times E}\tag{A7}
$$

where $\begin{array} { r } { \sum _ { e } \mathbf { C } _ { b , n , e } = 1 } \end{array}$ for each patch n. Dispatch and combine correspond to the input-to-slot and slot-to-output transformations in Soft-MoE; these are analogous to “routing” or “gating” weights in sparse MoE literature [13, 29].

(4) Aggregation transforms patch representations into expert-specific inputs using dispatch weights:

$$
\mathbf {S} _ {\mathrm{in}} = \mathbf {D} ^ {\top} \mathbf {X} \in \mathbb {R} ^ {B \times E \times d}\tag{A8}
$$

(5) Each expert processes its inputs: ${ \bf S } _ { \mathrm { o u t } } ^ { ( e ) } = \mathrm { E x p e r t } _ { e } ( { \bf S } _ { \mathrm { i n } } ^ { ( e ) } )$ for $e = 0 , \ldots , E -$ 1.

(6) Outputs are combined back to per-patch representations:

$$
\mathbf {Y} = \mathbf {C} \cdot \mathbf {S} _ {\mathrm{out}} \in \mathbb {R} ^ {B \times N \times d}\tag{A9}
$$

Parameter and Memory Cost. Each MoE block replaces one MLP $( 2 \times$ $d \times 4 d = 2 \times 7 6 8 \times 3 0 7 2 \approx 4 . 7 \mathrm { { M } }$ parameters) with E independent expert MLPs, adding $( E - 1 ) \times 4 . 7 \mathrm { M }$ parameters per block across 6 alternating blocks. Total model parameters: 86M (no MoE), 116M $( E { = } 2 , + 3 5 \% )$ , 537M (E = 16), 993M (E = 32), 1.86B (E = 64, 21.6×). Peak GPU memory scales linearly with E due to expert parameters and activations; E = 64 requires approximately 72 GB per GPU with batch size 1024 per GPU (FP16), fitting within 80 GB H100 HBM3. Routing overhead (logits, dispatch, combine weights) is negligible: $3 \times B \times N \times E$ floats per MoE block.

## H Extended Expert Scaling Discussion

This section provides detailed analysis of expert behavior when the number of experts E exceeds the number of loss-weighted objectives K, complementing the scaling results in Section 4.3 of the main paper.

Two experts $( E = 2 , K = 1 )$ . Expert 0 directly modulates the token loss through dispatch weights (loss-coupling). Expert 1 specializes through complementary gradient pressure: the shared logit matrix means that when Expert 0’s dispatch weights are pushed toward foreground patches by loss gradients, Expert 1’s weights shift toward the complementary set. Empirically, this manifests as CLS concentration (91–95% dispatch weight on the CLS token at Block 11; Section 4.2 of the main paper).

Many experts $( E = 6 4 , \ K = 1 )$ . Only one expert receives direct losscoupling; the remaining 63 provide capacity through standard MoE forwardpass gradients and entropy regularization. Even at this scale, dispatch-weight loss weighting provides a consistent benefit (+0.9% k-NN over the unweighted variant), though the relative gain decreases from +1.3% (2 experts) to +0.9% (64 experts) as inherent diversity from more expert parameters provides partial dispatch variation independently of loss-coupling.

## I Extended Mechanism Isolation Analysis

This section details how the evaluation structure in Table 3 of the main paper isolates the loss-weighting mechanism from the parameter efect at each expert scale.

The isolation relies on three complementary comparisons:

First, comparing non-MoE to MoE quantifies the combined efect of routing capacity and loss weighting. For Token+CLS: 75.7% (no $\mathrm { M o E } )  7 5 . 4 \%$ (2- expert weighted) → 76.2% (64-expert weighted).

Second, comparing weighted vs. unweighted variants at the same expert count (identical parameters) isolates loss weighting: +1.3% at 2 experts (74.1% → 75.4%), +0.9% at 64 experts (75.3% → 76.2%).

Third, the unweighted 2-expert model (74.1%) underperforms the non-MoE baseline (75.7%) despite having 35% more parameters (116M vs. 86M), demonstrating that parameters alone do not explain the gains; the loss-coupling mechanism is essential.

The 2-expert configuration ofers the most eficient tradeof: 35% parameter increase for 79.6% linear probe (vs. 80.6% at 64 experts with 21.6× parameters). We view the 64-expert scale as demonstrating that dispatch-weight loss weighting improves representation quality at scale, while the 2-expert configuration is the practical recommendation.

## J Downstream Adaptation Strategies

While MoE-based loss weighting improves pretraining representations (as measured by k-NN and linear probe), transferring MoE models to downstream tasks presents unique challenges. Standard finetuning can disrupt the routing patterns learned during pretraining, and the additional expert parameters increase overfitting risk. Three complementary adaptation strategies address these challenges:

Freeze Routing (FR). Following ST-MoE [39], we freeze all routing parameters (Φ, s) during finetuning, preserving pretrained patch-expert assignments while allowing expert MLP parameters to adapt. Without freezing, finetuning overwrites routing patterns with task-specific shortcuts.

Expert Dropout (ExD). Following Switch Transformer [13], we apply dropout (p = 0.4) to expert outputs during finetuning, preventing over-reliance on dominant experts and encouraging redundancy across the expert ensemble. This is analogous to standard dropout but operates at the expert granularity rather than the neuron level.

Freeze Attention (FA). Following ST-MoE [39], we freeze all attention parameters (QKV and output projections), finetuning only MLP/expert parameters, layer norms, and the classification head. Pretrained attention patterns encode transferable visual structure.

These compose naturally: the best recipe FR+FA+ExD freezes routing and attention to preserve pretrained representations while expert dropout regularizes during task-specific adaptation. The recipe ablation and cross-configuration transfer results are in Section 4.4 of the main paper.

## K Inference Cost and Cost–Quality Trade-of

Table 6 of the main paper reports inference GFLOPs as analytical MACs at N = 197 tokens, cross-validated against fvcore within 1.1%. Because Soft-MoE with one slot per expert routes all tokens into only E expert slots, the expert MLPs process E rather than N inputs, so parameter count and inference cost decouple: both ExPLoRe configurations use fewer GFLOPs than every ViT-B/16 comparator (11.93 at E = 2, 13.86 at E = 64, vs. 17.45). Figure A4 plots the resulting cost–quality trade-of: ExPLoRe sits to the upper-left of the comparator cluster, achieving comparable or higher linear-probe accuracy at lower inference cost. This directly addresses whether the mechanism provides value beyond simply adding parameters: on a per-FLOP basis it does.

## L Seed and Mask-Ratio Robustness

To assess statistical reliability, we re-trained the 2-expert Token+CLS model under two additional random seeds and a higher 50% mask ratio, evaluating each under k-NN, linear probe (LP), and Eficient Probing (EP). Table A5 reports the results: variation is tight across all three protocols (standard deviation 0.078, 0.037, 0.039 respectively), and the 50% mask ratio stays within this noise band (40% is the established block-masking optimum [3, 22]). The mechanismisolation deltas reported in the main paper compare configurations sharing the same pretraining randomness and are therefore not single-run artifacts.

![](images/e300680bb71c88b1abbbd49aacb4eebd945e89a4a9c0bbb0168dcc886a8c9728.jpg)  
Fig. A4: Cost–quality trade-of on ImageNet-1K. Inference GFLOPs (ViT-B/16 backbone, N = 197 tokens) vs. linear-probe top-1 accuracy. ExPLoRe E=2 (triangle) and E=64 (star) lie to the left of all compared methods, which cluster at 17.45 GFLOPs.

Table A5: Seed and mask-ratio robustness for the 2-expert Token+CLS model. Seeds 1–2 and mask=0.5 vs. the seed-42, mask-0.4 baseline. k-NN at k = 20.

<table><tr><td></td><td>Seed 1</td><td>Seed 2</td><td>Std</td><td>Mask=0.5</td><td>Baseline</td></tr><tr><td>k-NN</td><td>75.40</td><td>75.23</td><td>0.078</td><td>75.34</td><td>75.39</td></tr><tr><td>LP</td><td>79.54</td><td>79.63</td><td>0.037</td><td>79.23</td><td>79.60</td></tr><tr><td>EP</td><td>80.21</td><td>80.28</td><td>0.039</td><td>79.80</td><td>80.19</td></tr></table>

## M Token+CLS+Pixel Three-Objective Extension

The main paper studies two-objective combinations (Token+CLS, Token+Pixel). We additionally trained the three-loss variant (2 experts, sparse, $\lambda { = } 0 . 5 , w _ { \mathrm { p i x e l } } { = }$ 0.1, otherwise matching the practical Token+CLS configuration). It reaches 75.15% k-NN top-1, comparable to the matched non-MoE three-loss baseline (75.16%). The third loss introduces gradient-magnitude conflicts that destabilize MoE specialization at default settings: $w _ { \mathrm { p i x e l } } = 1 . 0$ underperforms by 0.67%, and asymmetric per-expert entropy (the Token+Pixel-dense recipe) destabilizes dense-mode routing. At matched representation quality, per-patch routing in this configuration does not yet provide a measurable benefit over symmetric three-loss training; efective three-loss MoE requires per-loss expert-specialization analysis, which we leave to future work.