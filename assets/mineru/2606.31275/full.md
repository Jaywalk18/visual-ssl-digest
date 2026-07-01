# CLIMB: CENTROID-BASED HIERARCHICAL MEMORY FOR ONLINE CONTINUAL SELF-SUPERVISED LEARNING

Julien Lefebvre Universite Claude Bernard Lyon 1, CNRS, INSA Lyon, LIRIS, UMR 5205, 69622 Villeurbanne, France julien.lefebvre@liris.cnrs.fr

Stefan Duffner

INSA Lyon, CNRS, Universite Claude Bernard Lyon 1, LIRIS, UMR 5205, 69621 Villeurbanne, France stefan.duffner@liris.cnrs.fr

Mathieu Lefort Universite Claude Bernard Lyon 1 / Univ. Rennes, Inria, CNRS LIRIS UMR 5205 / IRISA UMR 6074, France mathieu.lefort@liris.cnrs.fr

## ABSTRACT

Online Continual Self-Supervised Learning (OCSSL) aims to learn representations from a continuous stream of unlabeled data, without knowledge of task boundaries and under memory constraints. Existing methods rely either on replay buffers that exploit latent space structure, or on regularization alone. We present CLIMB (Continual Learning with Intelligent Memory Bank), which combines both simultaneously. Our method introduces a hierarchical centroid-based memory, bounded in total number of stored images, combined with knowledge distillation on replayed examples to limit representation drift. The memory groups similar images into centroids, providing hard-to-discriminate examples for contrastive learning while covering the diversity of observed distributions. Experiments on Split CIFAR-100 and Split ImageNet-100, on standard benchmarks from the state-of-the-art as well as a new protocol with irregular task distributions show that CLIMB outperforms state-of-the-art OCSSL methods. Source code is available at : https://github.com/lefebvju/climb

## 1 INTRODUCTION

Self-Supervised Learning (SSL) has established itself as the dominant paradigm for representation learning from unlabeled data (Chen et al., 2020; Grill et al., 2020; Chen & He, 2021). SSL methods exploit the intrinsic structure of data through augmentations. Contrastive approaches (Chen et al., 2020) attract different views of the same image while repelling those of different images. Non-contrastive approaches (Grill et al., 2020; Chen & He, 2021) prevent representational collapse through architectural asymmetry without requiring negative examples. However, most SSL methods assume that all training data is jointly available, which is not realistic when data evolves over time. In such settings, these systems either require full retraining from scratch at each update, or suffer from catastrophic forgetting, the tendency of neural networks to overwrite weights associated with past tasks when learning new ones (Kirkpatrick et al., 2017). The domain of Continual Learning (CL) addresses this challenge by enabling models to acquire new knowledge incrementally without erasing prior representations (Wang et al., 2024).

A sub-branch of continual learning focuses on unannotated data, making it possible to exploit far larger data volumes by removing the need for labels. The intersection of CL and SSL defines Continual Self-Supervised Learning (CSSL), an emerging field explored by several recent works (Fini et al., 2022; Gomez-Villa et al., 2022; Zhang et al., 2024). Generally in CSSL, methods assume that multiple epochs can be performed on each task and that task boundaries are known, allowing catastrophic forgetting mitigation mechanisms to be triggered explicitly at each transition. We focus on a subdomain of CSSL, Online Continual Self-Supervised Learning (OCSSL) (Yu et al., 2023), where the model performs a single pass over the incoming data stream using mini-batches, with no access to task boundaries. This setting is particularly relevant for applications requiring rapid adaptation and continuous representation evaluation, for example an autonomous agent exploring a novel environment. Fair comparison between methods requires controlling both memory capacity and computational budget simultaneously, a constraint barely addressed in the existing literature.

In this paper, we present CLIMB, which addresses the OCSSL setting by combining a novel strictly bounded hierarchical centroid-based memory with a classical EMA-based regularization. Our contributions are:

1. A hierarchical centroid-based memory inspired by PCMC (Taylor et al., 2024), compatible with the OC-SSL setting under strict memory constraints. Unlike PCMC, CLIMB operates with full-image representations rather than patches, removes both the offline pretraining phase and the sleep phases, and imposes a strict global bound on memory size. Combined with knowledge distillation on replayed examples following CLA (Cignoni et al., 2025), this memory improves performance in most configurations, with a particularly pronounced advantage when data is observed in a more fragmented fashion, i.e., when each task covers a smaller subset of classes.

2. A new evaluation protocol with irregular task distributions for the OCSSL setting, where the number of classes per task varies randomly, following irregular distributions previously explored in supervised continual learning (Koh et al., 2022), in order to measure the robustness of OCSSL methods against more varied continual learning configurations than the balanced tasks usually considered.

The related work is presented in Section 2, followed by a detailed description of our approach (Section 3). Experimental protocols and analysis of results are then developed in Sections 4 and 5, before concluding and discussing perspectives in Section 6.

## 2 RELATED WORK

In continual learning, three main families of methods have been developed.

Regularization-based methods constrain network weight updates based on their importance to previous tasks. EWC (Kirkpatrick et al., 2017) exploits the Fisher information matrix to estimate the importance of each parameter and penalizes their modification when learning new tasks. SI (Zenke et al., 2017) extends this idea by evaluating parameter importance along the full learning trajectory. Both methods operate in a supervised setting. A second regularization family targets model outputs rather than parameters, using a frozen previous model as a teacher via a distillation loss. LwF (Li & Hoiem, 2017) is the founding work of this approach. CaSSLe (Fini et al., 2022) adapts it to the offline self-supervised setting via a dedicated projection head.

Architecture-based methods dynamically adapt model structure to allocate capacity to new tasks. Progressive Neural Networks (Rusu et al., 2016) dedicate a new backbone to each task while adding lateral connections to reuse past representations. U-TELL (Solomon et al., 2024) extends this paradigm to the unsupervised setting by dedicating a new expert module to each incoming task.

Replay-based methods revisit past data during training. They divide into two main categories: generative approaches, which use a generative model to synthesize past examples without requiring explicit storage of input (Cywinski et al.´ , 2024; Solomon et al., 2024), and episodic memory approaches, which maintain a buffer containing a subset of past data for replay during training (Purushwalkam et al., 2022; Taylor et al., 2024; Cignoni et al., 2025). These approaches have been developed in both supervised (Cywinski et al.´ , 2024) and self-supervised settings (Purushwalkam et al., 2022; Solomon et al., 2024; Cignoni et al., 2025).

These three families are not mutually exclusive and are frequently combined to leverage their complementarity. In a supervised setting, methods such as iCaRL (Rebuffi et al., 2017) combine replay and distillation to better preserve past knowledge. In a self-supervised setting, replayed examples are also frequently combined with knowledge distillation regularization, where a frozen previous model constrains current representations from drifting excessively (Fini et al., 2022; Gomez-Villa et al., 2022; Cignoni et al., 2025). U-TELL (Solomon et al., 2024) combines architectural modifications with generative replay in an unsupervised setting.

Online supervised continual learning has been extensively studied (Buzzega et al., 2020; Koh et al., 2022; Caccia et al., 2022), where the single-pass constraint requires updating representations from each mini-batch only once without storing the full dataset. Online Continual Self-Supervised Learning (OCSSL) extends this challenge to the unlabeled setting, combining the absence of annotations, a single-pass data stream, and unknown task boundaries. Existing methods augment a standard SSL backbone with mechanisms from the families described above: a regularization loss constraining representation drift, an episodic memory for replaying past examples, or a combination of both.

Comparing OCSSL methods equitably is non-trivial, as performance depends on both memory capacity and compu tational budget. A method with access to more stored examples benefits from more diverse replay, while a method performing more gradient updates exploits that diversity more effectively. Without controlling both simultaneously, comparisons become meaningless: a well-organized memory paired with a larger budget would trivially outperform a simpler method under tighter constraints. CLA (Cignoni et al., 2025) identifies this open problem and introduces Cumulative Backward Passes (CBP) as a principled metric, representing the total number of images that undergo backpropagation during training. We will adopt this metric throughout our experiments to ensure fair comparison across all methods.

Replay-based methods maintain a buffer of past examples used during training. Minred (Purushwalkam et al., 2022) keeps a subset of maximally decorrelated examples, discarding the most redundant at each insertion, and uses only memory examples in the SSL loss without additional regularization. Osiris-R (Zhang et al., 2024) uses two parallel projection heads: one dedicated to plasticity, optimized on current stream examples, and one dedicated to cross-task consolidation, optimized on both stream and replay examples via a contrastive loss. Memory relies on reservoir sampling (Vitter, 1985) for the buffer. PCMC (Taylor et al., 2024) takes a different approach, maintaining a centroid-based memory that approximates the distribution of data representations, storing raw image patches to avoid representation drift due to encoder updates. Clusters of similar patches in representation space are distributed between a short-term memory (STM) and a long-term memory (LTM). The method proceeds in two phases: a wake phase during which the model generates clusters in the STM and fills them with patches. In the middle of each task, a sleep phase where the encoder is retrained in offline SSL mode on patches stored in STM and LTM, then centroids are recalculated with the new encoder to reposition them in latent space. PCMC’s hierarchical centroid structure represents a promising approach to organizing replay memory, motivating its adaptation to the OCSSL setting. Although PCMC presents itself as an OCSSL method, several design choices imply substantially different constraints: the LTM grows without bound, and sleep phases involve 200 epochs of offline training on the entire memory, making CBP computation impossible since the memory has no defined size. Furthermore, since memory capacity is unbounded, optimizing the method effectively amounts to maximizing the number of stored images through hyperparameter tuning. PCMC also requires an initial offline pretraining task, constituting prior knowledge absent in other OCSSL methods. These differences make PCMC incompatible with the two constraints we impose for fair comparison, a fixed memory capacity and a measurable CBP budget.

Methods combining replay with regularization constrain representation drift in addition to replaying past examples. SCALE (Yu et al., 2023) organizes memory via the Part and Select Algorithm (Salomon et al., 2013) to maximize representational diversity, and penalizes divergence between the current model’s similarity matrix and that of the previous training step. CLA (Cignoni et al., 2025) also combines replay with a distillation loss, but proposes a more stable reference by aligning current representations with past ones on replayed examples only, preserving plasticity for new data. Two variants are proposed: CLA-E uses an EMA encoder updated after each mini-batch as a stable distillation reference, while CLA-R stores embeddings from the first pass through the encoder directly as alignment targets. CLA achieves state-of-the-art performance among the OCSSL methods evaluated in their paper, with CLA-E and CLA-R each leading on different metrics and configurations.

Faced with these observations, we propose CLIMB, which builds on the idea of a hierarchical centroid-based memory to replace the FIFO buffer of CLA, the current state-of-the-art OCSSL method. Rather than retaining only the most recent examples as CLA does, CLIMB maintains a structured memory that covers the entire stream, raising the question of what to preserve and discard under a fixed capacity constraint, while preserving CLA’s distillation mechanism as a complementary component to limit representation drift.

## 3 METHOD

## 3.1 ARCHITECTURE

CLIMB builds on a standard SSL architecture comprising a backbone $f _ { \theta }$ and a projection head $g _ { \theta }$ optimizing a selfsupervised pretext task, as illustrated in Figure 1. Two mechanisms to limit forgetting are integrated: a memory module for replaying past examples, and a knowledge distillation loss using a frozen EMA model as reference to constrain representation drift. These mechanisms thus combine two of the main families identified in the previous Section 2: memory and regularization.

During training, each stream mini-batch $b _ { s }$ is encoded by the backbone and projection head, and subjected to a novelty detection step to update the hierarchical memory (STM/LTM, Section 3.2). A replay batch $b _ { r }$ is sampled from memory and concatenated with the current batch to form the final batch $b = b _ { s } \cup b _ { r }$ , used for SSL optimization, while knowledge distillation is applied on $b _ { r }$ only (Section 3.3).

## 3.2 MEMORY

CLIMB’s memory is a centroid-based structure that maintains a compact yet representative summary of the observed latent space under a strictly bounded capacity. Each centroid groups up to M raw images and is represented in latent space by the exponential moving average of their embeddings. The underlying intuition is that close neighbors in latent space are hard to discriminate under the contrastive loss and likely belong to the same mode of the feature distribution, grouping them into a shared centroid therefore yields a set of anchors that jointly cover the diversity of observed data throughout the stream. Distances are computed in the projected space after g<sub>θ</sub>, i.e., the space in which $\mathcal { L } _ { \mathrm { S S I } }$ is optimized, ensuring coherence between the centroid structure and the space in which the model learns to discriminate embeddings: hard examples identified by clustering are directly relevant to the training loss.Rather than maximizing raw diversity, CLIMB organizes its memory into semantically coherent and well-separated clusters that jointly cover the diversity of the observed stream, a quantitative analysis of this structure relative to alternative buffer strategies is provided in Appendix E.

![](images/6605c059b810681729b578c3c58b90984a499ea12c07f9dd2a57b0864bb5752a.jpg)  
Figure 1: Overview of CLIMB’s architecture. At each step, a stream mini-batch $b _ { s }$ is combined with a replay batch $b _ { r }$ sampled from the hierarchical centroid memory to form the final batch $b = b _ { s } \cup b _ { r }$ . The online network $( f _ { \boldsymbol { \theta } } , g _ { \boldsymbol { \theta } } )$ processes b under two augmented views, producing embeddings $z = g _ { \boldsymbol { \theta } } ( f _ { \boldsymbol { \theta } } ( t ( \boldsymbol { b } ) ) )$ ) for the contrastive loss ${ \mathcal { L } } _ { \mathrm { S S L } }$ . The alignment loss $\mathcal { L } _ { \mathrm { a l i g n } }$ is computed as the negative cosine similarity between present representations of the replay subset, passed through projection head $a _ { \phi } .$ , and past representations $z _ { r } ^ { \prime } \overset { \cdot } { = } g _ { \theta ^ { \prime } } ( f _ { \theta ^ { \prime } } ( \bar { t } ( b _ { r } ) ) )$ produced by the frozen EMA target network $( f _ { \theta ^ { \prime } } , g _ { \theta ^ { \prime } } )$ . Embeddings of replay examples $z _ { r }$ are also used to update the centroid positions in memory.

Inspired by the hierarchical designs of STAM (Smith et al., 2021) and PCMC (Taylor et al., 2024), memory is split into a short-term memory (STM, up to L centroids) acting as a staging area for newly discovered concepts, and a long-term memory (LTM, up to K centroids) consolidating mature, well-populated centroids (Figure 2). Our main contribution on the memory side, relative to PCMC, is to strictly bound total capacity, both in number of centroids and in number of stored images, while preserving a diverse, up-to-date summary of the stream. The complete update procedure is given in Algorithm 1.

Novelty is detected via an adaptive threshold $\tau ,$ recomputed at each step as the $p \cdot$ th percentile $( p \ : = \ : 0 . 9 5 )$ of the last $w = 1 0 0 0$ observed minimum distances, this removes the need to hand-tune a fixed distance cutoff and lets the threshold track the geometry of the representation as it evolves. A sample whose minimum cosine distance to all existing centroids exceeds τ instantiates a new STM centroid, replacing the least recently updated one if the STM is full. Otherwise the sample is assigned to its nearest centroid. STM assignments accumulate toward the promotion threshold M and update the centroid value via EMA with a fixed coefficient $\alpha _ { \mathrm { s t m } }$ . LTM assignments are accepted with probability 0.5, in which case they replace a random example in the matched centroid, allowing the LTM to incorporate recent content under bounded storage. When an STM centroid reaches M examples, it is promoted to the LTM. If the LTM is full, the two most similar centroids are merged and M examples are retained by random selection, bounding the number of long-term centroids while preserving latent-space coverage.

Two additional mechanisms enforce overall capacity control. First, a global pruning is triggered when the total number of stored images exceeds N: all STM examples are deleted except one anchor per centroid, the image that triggered its creation, retained so that the centroid remains a valid landmark and can accommodate similar incoming samples if they reappear later in the stream. An ablation of this pruning strategy is provided in Appendix D. This mechanism is distinct from LTM merging and targets image count rather than centroid count. Second, centroid values are updated after each gradient step using, for each replayed example, the mean of its two augmented-view embeddings produced by the updated encoder, ensuring that the memory structure remains aligned with the current state of the encoder as representations drift during training. A detailed analysis of these dynamics over a representative training run is provided in Appendix C.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Input: Stream mini-batch: images $x_i$ and their projected embeddings $z_i = g_\theta(f_\theta(x_i))$

Step 1 — Nearest centroid.

foreach image $x_i$ in the mini-batch do

| Compute the cosine distance between $z_i$ and every centroid in STM and LTM.
| Record the closest centroid $j^\star$ and its distance $d_i^\star$.

Step 2 — Assignment.

foreach image $x_i$ in the mini-batch do

| if $d_i^\star &gt; \tau$ (novel sample) then
| if STM is full ($|\text{STM}| = L$) then
| Remove the least recently updated STM centroid and its associated images.
| Instantiate a new centroid with value $z_i$ and associated image $x_i$.

else
| if $j^\star \in \text{STM}$ then
| Add $x_i$ to the centroid's examples (up to $M$).
| Update its value via EMA: $\mu_{j^\star} \leftarrow (1 - \alpha_{\text{stm}}) \mu_{j^\star} + \alpha_{\text{stm}} z_i$.

| else
| With probability 0.5, add $x_i$ to the LTM centroid's examples, replacing a randomly chosen example.

Step 3 — Threshold update.

Append the distances $d_i^\star$ to a sliding window of size $w$, and recompute $\tau$ as the $p$-th percentile of the window.

Step 4 — STM → LTM promotion.

foreach STM centroid $j^\star$ that received an example in Step 2 do

| if $|\mathcal{E}_{j^\star}| = M$ then
| Promote $j^\star$ to the LTM with its associated images; free its STM slot.
| if $|\text{LTM}| &gt; K$ then
| Merge the two LTM centroids with highest cosine similarity: pool their examples and retain $M$ by random selection.

Step 5 — Memory pruning.

if total stored images &gt; N then
| Delete all STM examples except the first one per centroid, retained as an anchor.

Step 6 — Centroid value update.

After the gradient step, update each centroid $j$ that contributed a replay image via EMA with coefficient $\alpha_j = 0.5 / |\mathcal{E}_j|$, using the mean embedding of the two augmented views of that image.

Algorithm 1: CLIMB Memory Update (per stream mini-batch)
</div>

![](images/980d3de5373c53abe2be85bf0de780f5fd44264ff8d913ef01eabf4a8438e120.jpg)  
Figure 2: Overview of CLIMB’s hierarchical memory. $c _ { i }$ denotes centroid positions in the projected latent space, and $d _ { i }$ denotes the cosine distances between each centroid and the current image embedding. The short-term memory (STM) is shown in orange and the long-term memory (LTM) in green. ⃝1 If min $( d _ { i } ) > \tau ,$ a new centroid is instantiated from the current image. $\textcircled{2}$ Otherwise, the image is assigned to the nearest centroid if it belongs to the STM. ⃝3 When a centroid reaches $M$ examples, it is promoted to the LTM and removed from the STM.

## 3.3 REPLAY-BASED KNOWLEDGE DISTILLATION

As illustrated in Figure 1, for each stream mini-batch $b _ { s }$ of size $| b _ { s } | ,$ , a replay batch $b _ { r }$ of size $\left| b _ { r } \right|$ is sampled in equal parts from the LTM and the STM, forming the final batch $b = b _ { s } \cup b _ { r }$ . This batch is transformed according to the augmentations of the chosen SSL method, then passed through the backbone $f _ { \theta }$ followed by projection head $g _ { \theta }$ to compute loss $\mathcal { L } _ { \mathrm { S S I } }$ We use SimCLR (Chen et al., 2020) as the base SSL method, whose loss enforces similarity between representations of two augmented views of the same image while pushing apart those of different images in the batch using the same encoder for both views. SimCLR has been extensively studied in the CSSL literature (Fini et al., 2022; Cignoni et al., 2025; Taylor et al., 2024; Zhang et al., 2024), enabling direct comparison with existing methods.

To limit catastrophic forgetting without task boundaries, an alignment loss is computed on examples from $b _ { r }$ , following CLA (Cignoni et al., 2025). The intuition is to constrain representations produced by the current model to remain aligned with those produced by a reference model, preventing excessively rapid drift of the latent space. This constraint is complementary to $\mathrm { C L I M B } ^ { \circ } \mathrm { s }$ structured replay: whereas CLA uses a FIFO buffer that retains only the most recent examples, CLIMB’s hierarchical centroid memory maintains examples representative of the entire stream, enabling replay to combat forgetting over the long term rather than only in the short term. The reference model is implemented as an EMA encoder $f _ { \theta ^ { \prime } }$ and its projector $g _ { \boldsymbol { \theta ^ { \prime } } }$ , updated after each mini-batch as $\theta ^ { \prime }  \tau _ { \mathrm { e m a } } \theta ^ { \prime } + ( 1 - \tau _ { \mathrm { e m a } } ) \theta .$ , where $\tau _ { \mathrm { e m a } } \in$ [0, 1] controls the update speed. A dedicated projection head $a _ { \phi }$ is applied to $z _ { r }$ to align them with representations $\begin{array} { r } { \dot { z } _ { r } ^ { \mathrm { e m a } ^ { \bullet } } = g _ { \theta ^ { \prime } } ( f _ { \theta ^ { \prime } } ( b _ { r } ) ) } \end{array}$ produced by the EMA model. The alignment loss is defined as the negative average cosine similarity between $a _ { \phi } ( z _ { r } )$ and $z _ { r } ^ { \mathrm { e m a , } }$

$$
\mathcal {L} _ {\mathrm{align}} = - \frac {1}{| b _ {r} |} \sum_ {i = 1} ^ {| b _ {r} |} \frac {a _ {\phi} (z _ {i}) \cdot z _ {i} ^ {\mathrm{ema}}}{\| a _ {\phi} (z _ {i}) \| \| z _ {i} ^ {\mathrm{ema}} \|}\tag{1}
$$

where $z _ { i } = g _ { \theta } ( f _ { \theta } ( t ( x _ { i } ) ) )$ is the representation of replay example $x _ { i } \in b _ { r } , a _ { \phi } ( z _ { i } )$ is the projection head applied to the representation of $x _ { i } ,$ , and $z _ { i } ^ { \mathrm { e m a } } = \bar { g _ { \theta ^ { \prime } } } ( f _ { \theta ^ { \prime } } ( t ( x _ { i } ) ) )$ ) its representation from the EMA model. The total loss is then:

$$
\mathcal {L} = \mathcal {L} _ {\mathrm{SSL}} + \lambda \mathcal {L} _ {\mathrm{align}}\tag{2}
$$

where λ is a hyperparameter controlling the weight of the alignment loss relative to the SSL loss.

## 4 EXPERIMENTS

## 4.1 DATASETS AND TASK CONFIGURATIONS

Experiments are conducted on Split CIFAR-100 (Krizhevsky et al., 2009) and Split ImageNet-100 (Deng et al., 2009), the reference benchmarks used in CLA (Cignoni et al., 2025) and CaSSLe (Fini et al., 2022). All experiments use a class-incremental protocol.

Regular task distribution. Following Cignoni et al. (2025), the standard configuration uses 20 tasks of 5 classes drawn randomly per seed, ensuring that all methods face identical class sequences for a given seed. To evaluate method behavior under more challenging conditions, configurations with 50 and 100 tasks are also tested, reducing the number of classes per task while preserving the same total number of classes. This increases task fragmentation, making classes seen in early tasks more susceptible to forgetting.

Irregular task distribution. Inspired by Koh et al. (2022), who introduced irregular task distributions in the supervised continual learning setting, to evaluate the robustness of methods against varied task sequences, we conduct experiments with irregular distributions of the number of classes per task, on the 20-task and 50-task configurations. The 100-task configuration is excluded as it would force exactly one class per task, preventing any irregularity. For the 20-task (resp. 50-task) configuration, the number of classes per task is drawn randomly between 1 and 12 (resp. 5), subject to the constraint that the sum of classes across all tasks remains equal to the total number of classes in the dataset. Since task distribution introduces an additional factor of variability beyond model initialization, 20 seeds are used for these experiments, each controlling both the class-to-task assignment, the number of classes per task and model initialization. Full distributions across all seeds are provided in Appendix A.

## 4.2 MODELS AND EVALUATION PROTOCOL

All experiments use a ResNet-18 encoder (He et al., 2016). The computational budget is measured by CBP (Cumula tive Backward Passes) as defined by Cignoni et al. (2025):

$$
\mathrm{CBP} = n _ {v} \times n _ {\text { steps }} \times b, \quad n _ {\text { steps }} = n _ {p} \times \frac {| \mathcal {D} |}{b _ {s}},\tag{3}
$$

where ${ n } _ { v } = 2$ is the number of views required by SimCLR, $b = b _ { s } + b _ { \tau }$ <sub>r</sub> is the total mini-batch size, $n _ { p }$ is the number of sequential passes per incoming mini-batch, and |D| is the total number of training samples. We follow the high CBP setting of CLA (Cignoni et al., 2025), with $b _ { s } = 1 0 , \dot { b _ { r } } = 1 2 8$ and $n _ { p } = 3 { \mathrm { ; } }$ , yielding $\mathrm { { C B P } = 3 . 7 \times 1 0 ^ { 6 } }$ for Split CIFAR-100 and $\mathrm { C B P = 9 . 5 \times 1 0 ^ { 6 } }$ for Split ImageNet-100. This budget is maintained equivalent across all compared methods. The total memory capacity is likewise fixed at $N = 2 5 0 0$ images for all methods. For regular task distribution and ablation experiments, 5 runs with fixed seeds are conducted. For irregular task distribution experiments, 20 seeds are used as described in previous section.

CLIMB is compared against MinRed, Osiris-R, SCALE, CLA-E, and CLA-R. Results with SimSiam as an alternative base SSL method are reported in Appendix G. PCMC (Taylor et al., 2024) is included as an indicative reference without its offline pretraining phase, the only modification made to ensure a minimal basis for comparison. Even in this reduced setting, its effective CBP remains substantially higher than the budget used for all other methods due to its offline sleep phases, and its memory grows without bound, reaching between 6000 and 15000 stored patches. Note that these patches are $6 0 \times 6 0$ crops rather than full images, making direct memory capacity comparison with other methods difficult.

Performance is evaluated via a linear classifier trained on representations produced by the frozen encoder $f _ { \theta }$ on all classes seen so far, and we report the mean and standard deviation of two metrics: Final Accuracy (FA), measured at the end of the training stream on all classes, and Continual Accuracy (CA), computed by averaging accuracies evaluated at the end of each task on all classes seen up to that point, reflecting representation quality throughout the stream. Forgetting metrics for all methods are reported in appendix H. Statistical significance is assessed by a Student’s t-test at threshold $p < 0 . 0 5$ . All baseline results are obtained using the publicly available implementation of Cignoni et al. (2025), with hyperparameters set as in their original paper.

## 4.3 ABLATION STUDY

An ablation study is conducted to analyze the respective contribution of each CLIMB component. Always preserving an equivalent CBP, four variants are first evaluated: an SSL-only model $( \mathcal { L } _ { \mathrm { S S L } } )$ , corresponding to plain Sim-CLR without any continual learning mechanism, to which the hierarchical centroid memory is added successively $( \mathcal { L } _ { \mathrm { S S L } } \mathrm { + M e m o r y ) }$ , then the alignment loss $\mathcal { L } _ { \mathrm { a l i g n } }$ computed on replayed examples with the model at the previous training step $( \dot { \mathcal { L } } _ { \mathrm { S S L } ^ { + } \mathrm { M e m o r y } + \dot { \mathcal { L } } _ { \mathrm { a l i g n } } ) }$ , then with the EMA model $( \bar { \mathcal { L } } _ { \mathrm { S S L } } \mathrm { + M e m o r y + } \bar { \mathcal { L } } _ { \mathrm { a l i g n } } \mathrm { + E M A ) }$ , constituting complete CLIMB. To isolate the contribution of the centroid-based memory structure, three additional variants replace it with alternative buffer strategies while keeping all other components identical: a FIFO buffer $( \mathcal { L } _ { \mathrm { S S L } } + \mathrm { F I F O } + \bar { \mathcal { L } } _ { \mathrm { a l i g n } } + \mathrm { E M A } )$ which corresponds to CLA-E (Cignoni et al., 2025) trained with CLIMB’s learning rate and alignment loss weight $\lambda ,$ a MinRed buffer (Purushwalkam et al., 2022) $( \mathcal { L } _ { \mathrm { S S L } } + \mathrm { M i n R e d } + \mathcal { L } _ { \mathrm { a l i g n } } + \mathrm { E M A } )$ , and a reservoir buffer (Vitter, 1985) (L +Reservoir+ $\cdot { \mathcal { L } } _ { \mathrm { a l i g n } } { + } \mathrm { E M A } )$

## 4.4 MODEL CONFIGURATION

CLIMB’s hyperparameters are configured as follows. The EMA reference model is updated with coefficient $\tau _ { \mathrm { e m a } } =$ 0.999 after each mini-batch, following CLA (Cignoni et al., 2025). The total memory budget is fixed at $N = 2 5 0 0$ images for all methods

The remaining hyperparameters were selected via grid search on the corresponding dataset with 20 tasks. The learning rate, alignment loss weight λ and STM centroid EMA coefficient $\alpha _ { \mathrm { { s t m } } }$ were searched jointly over lr ∈ $\{ 0 . 0 1 , \bar { 0 } . 0 5 , 0 . 1 , \bar { 0 } . 3 \} , \lambda \in \{ 0 . 1 , \bar { 0 . 5 } , 1 . 0 , 2 . 0 , 5 . 0 \}$ and $\alpha _ { \mathrm { s t m } } \in \{ 0 . 0 5 , 0 . 1 , 0 . 3 , 0 . 5 \}$ using SGD. For Split ImageNet-100 ,yielding ${ \mathrm { l r } } = { \dot { 0 } } . 1 , \lambda = { \dot { 1 } } . 0$ and $\alpha _ { \mathrm { s t m } } = 0 . 1$ . A separate grid search on Split CIFAR-100 yielded $\mathrm { { l r } = 0 . 3 }$ and $\lambda = 2 . 0$ , the remaining hyperparameters are shared across both datasets. The hierarchical memory parameters (M, $L , K )$ were then searched under the constraint that total allocated capacity remains consistent with $\bar { N } = 2 5 0 0$ , over $M \in \{ 1 0 , 2 0 , 3 0 , 5 0 \}$ images per centroid and STM/LTM centroid capacities $( L , K )$ accordingly, yielding $M = 3 0 .$ $L = \mathrm { i } 0 0 , K = 6 0$ . A sensitivity analysis of the memory architecture and update-rule parameters is provided in Appendix B.

Table 1: Classification performances of the OCSSL methods on 20, 50, and 100 tasks class-incremental settings.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2">CIFAR-100</td><td colspan="2">ImageNet-100</td></tr><tr><td>CA</td><td>FA</td><td>CA</td><td>FA</td></tr><tr><td></td><td>i.i.d.</td><td>—</td><td>50.45±2.03</td><td>—</td><td>53.93±1.52</td></tr><tr><td rowspan="7">20 tasks</td><td>SCALE</td><td>27.88±1.30</td><td>31.32±0.40</td><td>28.70±1.45</td><td>33.43±0.59</td></tr><tr><td>Osiris-R</td><td>34.13±1.29</td><td>37.65±0.57</td><td>37.06±1.80</td><td>42.72±2.00</td></tr><tr><td>MinRed</td><td>39.34±1.14</td><td>43.89±1.44</td><td>35.87±1.99</td><td>43.34±1.71</td></tr><tr><td>CLA-R</td><td>39.87±0.88</td><td>42.89±1.72</td><td>42.86±1.48</td><td>49.89±1.12</td></tr><tr><td>CLA-E</td><td>37.59±1.14</td><td>40.95±0.98</td><td>45.52±1.22</td><td>51.03±1.61</td></tr><tr><td>CLIMB</td><td>41.33±0.72</td><td>44.09±0.30</td><td>47.46±1.76</td><td>52.92±1.14</td></tr><tr><td>PCMC</td><td>39.26±1.13</td><td>43.01±0.77</td><td>40.11±1.15</td><td>39.42±1.73</td></tr><tr><td rowspan="7">50 tasks</td><td>SCALE</td><td>27.40±0.86</td><td>31.23±0.50</td><td>29.26±3.64</td><td>31.74±1.86</td></tr><tr><td>Osiris-R</td><td>31.94±1.08</td><td>37.19±0.74</td><td>35.64±1.80</td><td>39.02±0.93</td></tr><tr><td>MinRed</td><td>38.14±1.61</td><td>43.62±1.48</td><td>34.83±2.29</td><td>42.70±1.11</td></tr><tr><td>CLA-R</td><td>38.68±1.26</td><td>41.67±1.85</td><td>40.41±1.64</td><td>46.06±0.78</td></tr><tr><td>CLA-E</td><td>36.60±1.38</td><td>40.65±1.25</td><td>43.39±0.96</td><td>46.78±1.30</td></tr><tr><td>CLIMB</td><td>38.68±1.04</td><td>43.15±0.58</td><td>46.22±1.27</td><td>50.34±0.61</td></tr><tr><td>PCMC</td><td>38.31±1.77</td><td>43.13±4.16</td><td>42.65±1.54</td><td>44.46±2.33</td></tr><tr><td rowspan="7">100 tasks</td><td>SCALE</td><td>27.25±1.15</td><td>31.14±0.73</td><td>22.87±3.09</td><td>28.08±1.86</td></tr><tr><td>Osiris-R</td><td>32.91±1.43</td><td>35.48±1.51</td><td>35.04±1.71</td><td>39.44±1.47</td></tr><tr><td>MinRed</td><td>38.30±1.42</td><td>43.46±1.48</td><td>35.07±1.90</td><td>41.74±0.83</td></tr><tr><td>CLA-R</td><td>38.84±1.05</td><td>42.28±2.26</td><td>39.57±1.43</td><td>45.81±0.88</td></tr><tr><td>CLA-E</td><td>36.70±1.31</td><td>41.23±1.27</td><td>42.20±1.29</td><td>46.78±1.30</td></tr><tr><td>CLIMB</td><td>38.60±1.22</td><td>43.37±0.84</td><td>44.58±1.13</td><td>50.21±0.40</td></tr><tr><td>PCMC</td><td>36.72±1.64</td><td>39.48±2.79</td><td>40.53±1.13</td><td>38.12±5.64</td></tr></table>

## 5 RESULTS

## 5.1 REGULAR TASK DISTRIBUTION

Results on the 20, 50, and 100 tasks configurations are presented in Table 1. On CIFAR-100, CLIMB significantly outperforms all methods in CA on 20 tasks, and remains statistically indistinguishable from CLA-R and MinRed on 50 and 100 tasks, no method significantly dominates in FA across all configurations. The absence of clear separation is consistent with observations from the CLA article (Cignoni et al., 2025), where methods show a more pro nounced advantage on complex datasets such as ImageNet-100. However, CLIMB outperforms all OCSSL methods on ImageNet-100 across all configurations, in both Final Accuracy (FA) and Continual Accuracy (CA). The gap with CLA-E, the state-of-the-art method, remains around 2 percentage points on 20 tasks but widens as the number of tasks increases, highlighting a limitation of CLA’s FIFO buffer. By favoring recent examples, it tends to progressively forget older distributions as task count grows, whereas CLIMB’s hierarchical centroid memory maintains more representative coverage of the entire stream. PCMC, reported without its pretraining phase, remains competitive on CIFAR-100 but lags significantly behind CLIMB on ImageNet-100, despite operating under a substantially higher effective CBP due to its offline sleep phases and unbounded memory. Wall-clock time comparisons across all methods are reported in Appendix F.

## 5.2 IRREGULAR TASK DISTRIBUTION

Results on the irregular task distribution configuration are presented in Table 2. These experiments are conducted on both Split CIFAR-100 and Split ImageNet-100 and exclude SCALE due to its inferior performance in the regular task distribution experiments. On CIFAR-100, CLIMB, CLA-R, and MinRed are statistically indistinguishable in CA across both configurations, while CLIMB and MinRed lead in FA, with CLA-R significantly below in both configurations. This result is consistent with the regular task distribution setting, confirming that CLIMB achieves competitive performance on CIFAR-100. On ImageNet-100, CLIMB outperforms all OCSSL methods in both evalu ated configurations (20 and 50 tasks), in both Final Accuracy and Continual Accuracy. The gap with CLA-E, the most competitive method, is comparable to that observed in the regular configuration, confirming that CLIMB’s advantage on ImageNet-100 does not depend on the regularity of the task distribution. MinRed and Osiris-R exhibit markedly inferior performance, suggesting that their less structured replay strategies struggle to cover the diversity of encoun tered distributions when task fragmentation is variable. PCMC shows no clear advantage in the irregular setting either, achieving results comparable to the weakest OCSSL methods on ImageNet-100.

Table 2: Classification performances of OCSSL methods with irregular task distribution on Split CIFAR-100 and Split ImageNet-100.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2">CIFAR-100</td><td colspan="2">ImageNet-100</td></tr><tr><td>CA</td><td>FA</td><td>CA</td><td>FA</td></tr><tr><td rowspan="6">20 tasks</td><td>Osiris-R</td><td>33.37±1.41</td><td>36.94±1.77</td><td>37.52±2.71</td><td>40.68±2.61</td></tr><tr><td>MinRed</td><td>38.82±1.18</td><td>43.42±1.97</td><td>36.47±2.46</td><td>42.46±2.22</td></tr><tr><td>CLA-R</td><td>39.27±1.36</td><td>42.42±2.44</td><td>43.70±2.41</td><td>48.74±2.65</td></tr><tr><td>CLA-E</td><td>37.04±1.42</td><td>41.16±1.81</td><td>46.30±2.33</td><td>49.68±2.87</td></tr><tr><td>CLIMB</td><td>39.00±1.58</td><td>43.59±1.54</td><td>47.99±2.57</td><td>52.19±2.55</td></tr><tr><td>PCMC</td><td>38.50±1.46</td><td>39.65±4.75</td><td>38.44±1.49</td><td>37.07±4.70</td></tr><tr><td rowspan="6">50 tasks</td><td>Osiris-R</td><td>32.46±1.48</td><td>36.41±1.38</td><td>37.35±1.79</td><td>40.07±2.07</td></tr><tr><td>MinRed</td><td>38.61±1.46</td><td>43.32±1.20</td><td>36.99±1.94</td><td>42.18±2.64</td></tr><tr><td>CLA-R</td><td>38.89±1.62</td><td>42.40±1.53</td><td>42.68±1.70</td><td>46.53±1.70</td></tr><tr><td>CLA-E</td><td>36.79±1.65</td><td>41.03±1.62</td><td>45.28±1.83</td><td>47.49±1.70</td></tr><tr><td>CLIMB</td><td>38.93±1.74</td><td>43.22±1.68</td><td>47.94±1.94</td><td>50.82±1.65</td></tr><tr><td>PCMC</td><td>37.35±1.50</td><td>36.98±2.65</td><td>40.58±2.31</td><td>38.85±5.80</td></tr></table>

## 5.3 ABLATION STUDY

The ablation study results, presented in Table 3, clearly confirm the contribution of each CLIMB component and highlight the central role of the proposed intelligent memory. First, the SSL-only model achieves limited performance (19.64 ± 1.06 and 22.63 ± 1.22), confirming that online self-supervised learning, without an explicit information preservation mechanism, suffers strongly from catastrophic forgetting.

Using the hierarchical centroid memory yields a highly significant performance improvement $( 4 6 . 0 9 \pm 1 . 6 9$ and $5 0 . 8 4 \pm 0 . 9 5 )$ , demonstrating that the proposed memory alone already effectively stabilizes learned representations in a continual setting. This substantial improvement validates the idea that CLIMB relies first and foremost on a structured memory capable of preserving relevant information over time.

When the alignment loss $\mathcal { L } _ { \mathrm { a l i g n } }$ is introduced, performance improves further $( 4 6 . 5 6 \pm 1 . 5 8$ and $5 1 . 4 5 \pm 2 . 2 2 )$ , confirming that knowledge distillation on replayed examples helps reduce representation drift beyond what memory alone achieves, though this gain is not statistically significant. The choice of reference model, whether the previous training step model or an EMA-updated encoder as described in Section 3.3, has no statistically significant impact (47.46±1.76 and 52.92 ± 1.14), suggesting that any stable reference suffices in practice. The EMA variant nonetheless achieves the highest absolute performance of all configurations, and we therefore retain it as the reference in the final CLIMB configuration, as it additionally provides a smoother reference trajectory without adding computational overhead.

Finally, replacing the centroid-based memory with alternative buffer strategies while keeping all other components identical yields a significant performance drop in FA, confirming that CLIMB’s advantage is attributable to the memory design. Notably, the $\mathcal { L } _ { \mathrm { S S L } } + \mathrm { M e m o r y }$ variant alone achieves performance statistically equivalent to FIFO, MinRed, and Reservoir buffers combined with distillation, highlighting the importance of memory quality in the OCSSL setting.

## 6 CONCLUSION AND PERSPECTIVES

In this paper, we have presented CLIMB, an online continual self-supervised learning method that learns representations from a continuous stream of unlabeled data, without knowledge of task boundaries and under memory constraints. The central contribution of CLIMB is a hierarchical centroid-based memory, bounded in total number of stored images, designed to maintain a representative set of the latent space of the stream. Combined with replay-based knowledge distillation, this memory enables CLIMB to tackle catastrophic forgetting while maintaining representative coverage of the entire stream.

Table 3: Ablation study on Split ImageNet-100, 20 tasks.

<table><tr><td>Method</td><td>CA</td><td>FA</td></tr><tr><td> $\mathcal{L}_{\text{SSL}}$ </td><td>19.64±1.06</td><td>22.63±1.22</td></tr><tr><td> $\mathcal{L}_{\text{SSL}} + \text{Memory}$ </td><td>46.09±1.69</td><td>50.84±0.95</td></tr><tr><td> $\mathcal{L}_{\text{SSL}} + \text{Memory} + \mathcal{L}_{\text{align}}$ </td><td>46.56±1.58</td><td>51.45±2.22</td></tr><tr><td> $\mathcal{L}_{\text{SSL}} + \text{Memory} + \mathcal{L}_{\text{align}} + \text{EMA (CLIMB)}$ </td><td>47.46±1.76</td><td>52.92±1.14</td></tr><tr><td> $\mathcal{L}_{\text{SSL}} + \text{FIFO} + \mathcal{L}_{\text{align}} + \text{EMA}$ </td><td>45.15±1.53</td><td>50.98±0.96</td></tr><tr><td> $\mathcal{L}_{\text{SSL}} + \text{Minred} + \mathcal{L}_{\text{align}} + \text{EMA}$ </td><td>45.73±1.39</td><td>50.98±0.94</td></tr><tr><td> $\mathcal{L}_{\text{SSL}} + \text{Reservoir} + \mathcal{L}_{\text{align}} + \text{EMA}$ </td><td>44.78±1.82</td><td>48.64±2.00</td></tr></table>

Experiments on Split ImageNet-100 in 20, 50, and 100 tasks configurations show that CLIMB outperforms state-ofthe-art methods. Experiments with irregular task distributions further confirm the robustness of CLIMB across varied learning regimes. Ablation results show that the hierarchical centroid memory alone already matches the performance of flat buffer strategies combined with distillation, and that combining it with knowledge distillation yields a further significant gain in FA, confirming that both components are necessary to achieve the best performance.

Several directions for improvement can be envisioned. The balance between STM and LTM capacity, as well as the centroid merging strategy, could be refined using more informative indicators than cosine similarity alone, incorporating for instance usage frequency. The use of SSL methods that promote a better-structured latent space could reinforce the relevance of centroid groupings and thus improve replay quality. Finally, rather than storing raw images, the memory could preserve compressed representations or leverage a generative model, further reducing memory footprint, although this would need to find a way to tackle the representation drift issue that storing raw images currently avoids.

## ACKNOWLEDGMENTS

This work was granted access to the HPC resources of IDRIS under the allocation 2025-AD011016434 and 2025- AD011014045R2 granted by GENCI on the V100 partition of the Jean Zay supercomputer. This work was funded by Lyon 1 Universite and the Soutien aux ENSeignants-chercheurs (SENS) call for projects.´

## REFERENCES

Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. Advances in neural information processing systems, 33:15920–15930, 2020.

Lucas Caccia, Rahaf Aljundi, Nader Asadi, Tinne Tuytelaars, Joelle Pineau, and Eugene Belilovsky. New insights on reducing abrupt representation change in online continual learning. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=N8MaByOzUfb.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pp. 1597–1607. PmLR, 2020.

Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15750–15758, 2021.

Giacomo Cignoni, Andrea Cossu, Alexandra Gomez-Villa, Joost van de Weijer, and Antonio Carta. Cla: Latent alignment for online continual self-supervised learning. In Conference on Lifelong Learning Agents, 2025.

Bartosz Cywinski, Kamil Deja, Tomasz Trzci´ nski, Bartłomiej Twardowski, and Łukasz Kuci´ nski. Guide: Guidance-´ based incremental learning with diffusion models. In ECAI 2025, pp. 3614–3621. IOS Press, 2024.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Enrico Fini, Victor G. Turrisi Da Costa, Xavier Alameda-Pineda, Elisa Ricci, Karteek Alahari, and Julien Mairal. Self-Supervised Models are Continual Learners. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9611–9620, New Orleans, LA, USA, June 2022. IEEE. doi: 10.1109/CVPR52688.2022. 00940.

Dan Friedman and Adji Bousso Dieng. The vendi score: A diversity evaluation metric for machine learning. Transactions on Machine Learning Research, 2023. ISSN 2835-8856.

Alex Gomez-Villa, Bartlomiej Twardowski, Lu Yu, Andrew D Bagdanov, and Joost Van de Weijer. Continually learning self-supervised representations with projected functional regularization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3867–3877, 2022.

Jean-Bastien Grill, Florian Strub, Florent Altche, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Do-´ ersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521–3526, March 2017. ISSN 0027-8424, 1091-6490. doi: 10.1073/pnas.1611835114.

Hyunseo Koh, Dahyun Kim, Jung-Woo Ha, and Jonghyun Choi. Online continual learning on class incremental blurry task configuration with anytime inference. In ICLR, 2022.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947, 2017.

Senthil Purushwalkam, Pedro Morgado, and Abhinav Gupta. The challenges of continuous self-supervised learning. In European conference on computer vision, pp. 702–721. Springer, 2022.

Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 2001–2010, 2017.

Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016.

Shaul Salomon, Gideon Avigad, Alex Goldvard, and Oliver Schutze. Psa–a new scalable space partition based se-¨ lection algorithm for moeas. In EVOLVE-A Bridge between Probability, Set Oriented Numerics, and Evolutionary Computation II, pp. 137–151. Springer, 2013.

James Smith, Cameron Taylor, Seth Baer, and Constantine Dovrolis. Unsupervised progressive learning and the STAM architecture. arXiv preprint arXiv:1904.02021, 2021. Accepted for publication at IJCAI 2021.

Indu Solomon, Aye Phy Phyu Aung, Uttam Kumar, and Senthilnath Jayavelu. U-tell: Unsupervised task expert lifelong learning. In 2024 IEEE International Conference on Image Processing (ICIP), pp. 1057–1063. IEEE, 2024.

Cameron Ethan Taylor, Vassilis Vassiliades, and Constantine Dovrolis. Patch-based contrastive learning and memory consolidation for online unsupervised continual learning. In Vincenzo Lomonaco, Stefano Melacci, Tinne Tuytelaars, Sarath Chandar, and Razvan Pascanu (eds.), Proceedings of The 3rd Conference on Lifelong Learning Agents, volume 274 of Proceedings of Machine Learning Research, pp. 938–958. PMLR, 29 Jul–01 Aug 2024.

Jeffrey S Vitter. Random sampling with a reservoir. ACM Transactions on Mathematical Software (TOMS), 11(1): 37–57, 1985.

Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning: Theory, method and application. IEEE transactions on pattern analysis and machine intelligence, 46(8):5362–5383, 2024.

Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International conference on machine learning, pp. 9929–9939. PMLR, 2020.

Xiaofan Yu, Yunhui Guo, Sicun Gao, and Tajana Rosing. SCALE: Online Self-Supervised Lifelong Learning without Prior Knowledge. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Work shops (CVPRW), pp. 2484–2495, Vancouver, BC, Canada, June 2023. IEEE. ISBN 979-8-3503-0249-3. doi: 10.1109/CVPRW59228.2023.00247.

Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In International conference on machine learning, pp. 3987–3995. Pmlr, 2017.

Yipeng Zhang, Laurent Charlin, Richard Zemel, and Mengye Ren. Integrating present and past in unsupervised continual learning. In Conference on Lifelong Learning Agents, pp. 388–409. PMLR, 2024.

## A IRREGULAR TASK DISTRIBUTION

Figure 3 shows the frequency distribution of the number of classes per task, pooled across all 20 seeds, for the 20-task and 50-task configurations. These distributions are identical for Split CIFAR-100 and Split ImageNet-100, as the same seeds control the class-to-task assignment in both cases.

![](images/4b981ffde31fc0fbb09e01b941e7886dc59f90828156c3c23d3b3aae556ab59d.jpg)

![](images/877f2f26b259e9402558c45455a70ed20755fb5e198f2fc21f94a3af07db7635.jpg)  
Figure 3: Frequency histogram of the number of classes per task pooled over all 20 seeds, for the 20-task (left) and 50-task (right) configurations. The dashed line indicates the mean value, which corresponds to the fixed number of classes per task in the regular protocol.

## B HYPERPARAMETER SENSITIVITY ANALYSIS

Tables 4 and 5 report the sensitivity of CLIMB to its memory architecture and update-rule parameters, respectively, on Split ImageNet-100 with 20 tasks over 5 seeds

Coupling constraints. To perform these experiments, we vary the values of N, L, K, M, and the update-rule parameters independently. However, some of these memory parameters are not independent under a fixed total budget N. When varying N, the number of LTM centroids K is adjusted proportionally as $K = \lvert 6 0 \times N / 2 5 0 0 \rvert$ to ensure that the LTM capacity K × M remains consistent with the total budget, M is kept fixed at 30. When varying M, K is adjusted so that the total LTM image capacity $K \times M$ remains approximately constant at 1800, corresponding to the LTM capacity at the reference budget N = 2500. The L and K experiments do not require coupling adjustments.

Memory architecture parameters (Table 4). Performance increases monotonically with N up to the reference value of 2500, beyond which it levels off, suggesting that $N = 2 5 0 0$ is sufficient for the memory to maintain adequate stream coverage, noting that the remaining hyperparameters were also tuned at this budget. Regarding L, performance is relatively stable across the tested range, with the reference value $L = 1 0 0$ achieving the best result, both smaller and larger values lead to slightly lower performance. For K, performance peaks at the reference $K = 6 0$ , with $K =$ 50 achieving statistically indistinguishable results, suggesting robustness to moderate reductions in LTM capacity. Performance degrades at $K = 8 0$ in FA, note that due to the dependence between K, M, and N, with $K = \bar { 8 } 0$ and

Table 4: Sensitivity to memory architecture parameters on Split ImageNet-100 (20 tasks, 5 seeds). Underlined: reference configuration.

<table><tr><td>Parameter</td><td>Value</td><td>CA</td><td>FA</td></tr><tr><td rowspan="6">N (total images)</td><td>1000</td><td>43.23±2.61</td><td>47.39±1.09</td></tr><tr><td>1500</td><td>44.69±1.84</td><td>48.55±0.87</td></tr><tr><td>2000</td><td>45.71±2.46</td><td>50.41±1.41</td></tr><tr><td>2500</td><td>47.46±1.76</td><td>52.92±1.14</td></tr><tr><td>3500</td><td>47.18±1.94</td><td>52.78±1.02</td></tr><tr><td>5000</td><td>47.05±1.98</td><td>53.18±1.35</td></tr><tr><td rowspan="5">L (STM centroids)</td><td>50</td><td>45.88±1.55</td><td>52.03±0.78</td></tr><tr><td>75</td><td>46.58±1.71</td><td>51.70±1.10</td></tr><tr><td>100</td><td>47.46±1.76</td><td>52.92±1.14</td></tr><tr><td>150</td><td>46.13±1.65</td><td>51.12±1.04</td></tr><tr><td>200</td><td>46.52±1.85</td><td>52.26±1.01</td></tr><tr><td rowspan="5">K (LTM centroids)</td><td>20</td><td>44.97±1.49</td><td>50.29±1.35</td></tr><tr><td>35</td><td>46.31±1.87</td><td>51.58±0.63</td></tr><tr><td>50</td><td>46.98±1.38</td><td>52.42±0.35</td></tr><tr><td>60</td><td>47.46±1.76</td><td>52.92±1.14</td></tr><tr><td>80</td><td>45.95±1.82</td><td>50.91±1.11</td></tr><tr><td rowspan="4">M (images/centroid)</td><td>10</td><td>45.86±1.23</td><td>51.66±0.60</td></tr><tr><td>20</td><td>46.00±1.89</td><td>51.09±1.16</td></tr><tr><td>30</td><td>47.46±1.76</td><td>52.92±1.14</td></tr><tr><td>50</td><td>46.19±1.99</td><td>51.32±2.03</td></tr></table>

Table 5: Sensitivity to update-rule parameters on Split ImageNet-100 (20 tasks, 5 seeds). Underlined: reference configuration.

<table><tr><td>Parameter</td><td>Value</td><td>CA</td><td>FA</td></tr><tr><td rowspan="5">Novelty percentile p</td><td>0.75</td><td>42.67±1.33</td><td>46.88±0.42</td></tr><tr><td>0.85</td><td>43.93±1.86</td><td>49.47±0.42</td></tr><tr><td>0.90</td><td>45.95±2.59</td><td>51.68±1.76</td></tr><tr><td>0.95</td><td>47.46±1.76</td><td>52.92±1.14</td></tr><tr><td>0.99</td><td>45.98±2.09</td><td>51.24±1.23</td></tr><tr><td rowspan="4">EMA coefficient τema</td><td>0.990</td><td>46.39±2.24</td><td>51.38±1.29</td></tr><tr><td>0.995</td><td>46.44±2.13</td><td>51.26±1.18</td></tr><tr><td>0.999</td><td>47.46±1.76</td><td>52.92±1.14</td></tr><tr><td>0.9995</td><td>45.77±1.76</td><td>51.04±1.56</td></tr><tr><td rowspan="5">STM EMA coeff. αstm</td><td>0.01</td><td>45.40±2.41</td><td>51.08±1.87</td></tr><tr><td>0.05</td><td>45.72±2.01</td><td>50.07±1.19</td></tr><tr><td>0.1</td><td>47.46±1.76</td><td>52.92±1.14</td></tr><tr><td>0.3</td><td>45.87±1.89</td><td>51.54±0.84</td></tr><tr><td>0.5</td><td>46.07±1.68</td><td>51.78±0.88</td></tr></table>

$M = 3 0 ,$ , only 100 images remain available for the STM out of the total budget N = 2500, drastically reducing its staging capacity. For M, performance peaks at the reference $M = 3 0$ and slightly decreases for both smaller and larger values, while remaining statistically comparable in CA, reflecting a trade-off between centroid granularity and per-centroid example diversity.

Update-rule parameters (Table 5). The novelty percentile p controls how selective the threshold is: lower values make the threshold more permissive, creating many small centroids, while higher values are more conservative. As a slight tendency, performance degrades below $p = 0 . 9 0$ , while $p = 0 . 9 0 , p = 0 . 9 5$ , and $p = 0 . 9 9$ remain statistically comparable, confirming the stability of the method across a wide range of threshold selectivity. Performances are globally similar across all tested update-rule parameters, always statistically indistinguishable in CA and most often in FA, illustrating the robustness of CLIMB to these hyperparameters. As a slight tendency, the EMA coefficient $\tau _ { \mathrm { e m a } }$ shows that slower update speeds (higher $\tau _ { \mathrm { e m a } } )$ benefit performance up to the reference value of 0.999, and the STM EMA coefficient $\alpha _ { \mathrm { { s t m } } }$ exhibits a similar pattern, with the reference value $\alpha _ { \mathrm { s t m } } = 0 . 1$ providing the best trade-off between centroid responsiveness and stability.

## C STM/LTM MEMORY DYNAMICS

Figures 4 and 5 illustrate the internal dynamics of CLIMB’s hierarchical memory over a representative run on Split ImageNet-100 with 20 tasks (SimCLR, ResNet-18, seed 1486). Vertical dashed lines mark task boundaries.

The number of images stored in the STM (Figure 4, left) stabilizes around 700 from task 9 onwards, oscillating between ∼ 100 (immediately after a global pruning event) and ∼ 700 (just before the next pruning). These fluctuations reflect global pruning events triggered when total stored images exceed $N = 2 5 0 0$ , which reset STM centroids to a single anchor image each, followed by rapid refilling until the next pruning event. The count of the images in LTM (Figure 4, right) increases monotonically from 0 and saturates at $K \times M = 6 0 \times 3 0 = 1 8 0 0$ images around task 9, after which every new promotion triggers a merge that maintains the total LTM count at its maximum.

A total of 74 STM→LTM promotions occur over the full run (Figure 5, left), each corresponding to an STM centroid that accumulated M = 30 examples. Most promotions occur early in training, before the global pruning mechanism activates between tasks 2 and 3, once pruning begins, STM centroids are regularly reset to a single anchor image, delaying their accumulation toward the promotion threshold and resulting in visible flat segments in the cumulative curve.

The adaptive novelty threshold τ (Figure 5, right) rises during the first seven to ten tasks before stabilizing, oscillating within a narrow range around 0.6–0.7 over the remainder of training. This stabilization reflects that the encoder has built a sufficiently structured latent space: once representations are well-organized, the distribution of minimum distances to existing centroids becomes stationary, and the threshold tracks this stable geometry rather than continuously drifting upward. The early rise corresponds to the period during which the encoder is still learning discriminative features and the representation space is reorganizing.

![](images/674cff687ad7512a19dc56e31b9ec613a8545e80f4a12dedce4e4fefa60b0be6.jpg)

![](images/ada4c04c37dcb4cecf19da04ca54c8dddb554a90f54afb9146004d85b6567175.jpg)  
Figure 4: Number of images stored in the STM (left) and LTM (right) as a function of task.

## D GLOBAL PRUNING STRATEGY ABLATION

Table 6 compares two strategies for the global pruning step triggered when the total number of stored images exceeds N. The Anchor strategy, used by default in CLIMB, deletes all STM examples except one anchor image per centroid, retained as a landmark to accommodate future similar samples. The Selective strategy instead removes examples incrementally from the least recently updated STM centroid until the budget is met, preserving more recent content at the cost of potentially depleting individual centroids entirely.

Neither strategy is significantly better than the other across both task configurations $( p > 0 . 0 5 .$ , Student’s t-test), confirming that CLIMB is robust to this design choice. The Anchor strategy nonetheless achieves consistently higher absolute performance and is therefore retained as the default.

![](images/501b93e690886b08ef7e2b3411ea47f3807284f3ec3313925cbf49d5f85bad94.jpg)

![](images/24578e38a4928b88c57cf18d4d8bb38394b9c35e046899a567dd45636fe4b3f0.jpg)  
Figure 5: Left: cumulative number of STM→LTM promotions (74 total). Right: adaptive novelty threshold τ , computed as the 95th percentile of the last w = 1000 observed minimum distances.

Table 6: Global pruning strategy ablation on Split ImageNet-100 (5 seeds). Anchor: all STM examples are deleted except one anchor per centroid. Selective: examples are removed from the least recently updated STM centroid until the budget is met.

<table><tr><td>Strategy</td><td>Tasks</td><td>CA</td><td>FA</td></tr><tr><td>Anchor</td><td>20</td><td>47.46±1.76</td><td>52.92±1.14</td></tr><tr><td>Selective</td><td>20</td><td>47.36±1.35</td><td>52.21±1.15</td></tr><tr><td>Anchor</td><td>50</td><td>46.22±1.27</td><td>50.34±0.61</td></tr><tr><td>Selective</td><td>50</td><td>45.50±1.08</td><td>50.19±1.33</td></tr></table>

## E MEMORY DIVERSITY ANALYSIS

Table 7 compares the memory structure of CLIMB against MinRed, Reservoir, and FIFO at the end of a representative training run on Split ImageNet-100 with 20 tasks. To isolate the contribution of the memory module, all four buffers are evaluated within CLIMB’s learning pipeline under identical CBP budget, thus differing only in how images are selected and organized in memory. We report the Vendi score (Friedman & Dieng, 2023) as a measure of global memory diversity, the uniformity metric of Wang & Isola (2020) as a measure of coverage of the unit hypersphere, and an Inter/Intra ratio measuring the semantic separability of stored examples.

Table 7: Memory diversity metrics at end of training on Split ImageNet-100 (20 tasks). All buffers operate under capacity N = 2500.

<table><tr><td>Metric</td><td>CLIMB</td><td>MinRed</td><td>Reservoir</td><td>FIFO</td></tr><tr><td>Stored images (end of training)</td><td>2121</td><td>2500</td><td>2500</td><td>2500</td></tr><tr><td>Vendi score (Friedman &amp; Dieng, 2023) (↑)</td><td>43.1</td><td>35.3</td><td>53.3</td><td>29.2</td></tr><tr><td>Uniformity (Wang &amp; Isola, 2020) (↓)</td><td>-3.550</td><td>-3.582</td><td>-3.624</td><td>-3.419</td></tr><tr><td>Intra-cluster dist (↓)</td><td>0.264</td><td>0.453±0.003</td><td>0.324±0.002</td><td>0.288±0.002</td></tr><tr><td>Inter/Intra ratio (↑)</td><td>3.796</td><td>2.212±0.017</td><td>3.101±0.020</td><td>3.485±0.036</td></tr><tr><td>Linear probe accuracy (↑)</td><td>0.510</td><td>0.491</td><td>0.485</td><td>0.501</td></tr></table>

The Vendi score is defined as $\mathrm { V S } = \exp ( - \operatorname { t r } ( \mathbf { K } / n \log \mathbf { K } / n ) )$ , where $\mathbf { K } \in \mathbb { R } ^ { n \times n }$ is the cosine similarity matrix computed over the encoder embeddings of all n images stored in the buffer (Friedman & Dieng, 2023). Intuitively, VS can be read as the effective number of distinct elements in the buffer, ranging from 1 when all embeddings are identical to n when all are mutually orthogonal. The uniformity metric is $\mathcal { L } _ { \mathrm { u n i f o r m } } = \log \mathbb { E } _ { x , y } \big [ e ^ { - t \| f ( x ) - f ( y ) \| ^ { 2 } } \big ]$ with t = 2, where f denotes the frozen encoder and the expectation is over all pairs of buffered images, more negative values indicate more uniform coverage of the unit hypersphere (Wang & Isola, 2020). The Inter/Intra ratio is obtained by running k-means with k = 100, matching the number of classes in the dataset, on a subsample of 2121 encoder embeddings drawn from each buffer, matching CLIMB’s image count at end of training to ensure comparability, since cluster metrics depend on sample size, repeated 10 times for the baselines to account for sampling variance, reported values are means and standard deviations over these repetitions. Higher values indicate clusters that are simultaneously tight and well-separated.

Reservoir achieves the highest raw diversity, both in Vendi score (53.3) and uniformity (−3.624), yet obtains the lowest linear probe accuracy (0.485). This dissociation indicates that raw diversity alone is not the relevant criterion for replay quality: without structural organization, the stored examples do not provide the semantically coherent and well-separated groups that benefit contrastive learning.

FIFO achieves the worst uniformity score (−3.419) and the lowest Vendi score (29.2), consistent with its buffer being concentrated on the most recent task only. Its competitive Inter/Intra ratio (3.485) should reflects locally coherent clusters restricted to recent data.

CLIMB achieves the best Inter/Intra ratio (3.796), meaning its clusters are simultaneously internally coherent (intracluster distance 0.264) and mutually well-separated, while maintaining the second-highest Vendi score (43.1) despite storing fewer images than MinRed (35.3) and FIFO (29.2). This combination of local structure and global stream coverage is consistent with the highest linear probe accuracy (0.510), suggesting that structured diversity, semantically coherent and well-separated clusters covering the full stream, may be more beneficial for replay quality than raw diversity alone.

## F WALL-CLOCK TIME

![](images/5d34c292675d6774b4121ff436e27bbe7eb1e4b91425db869b040b9be7bfb9e9.jpg)  
Figure 6: Total training time (hours, log scale) on Split ImageNet-100 as a function of the number of tasks, measured on a single NVIDIA V100 32GB GPU.

Figure 6 reports the total training time in hours for each method on Split ImageNet-100, measured on a single NVIDIA V100 32GB GPU across the 20, 50, and 100 task configurations, using identical seeds for all methods.

All methods scale quasi-linearly with the number of tasks, a growth driven primarily by the increasing number of linear classifier evaluations: across the 20, 50, and 100 task configurations, the total number of training images is identical, distributed differently across tasks, so the additional cost comes almost entirely from the growing number of evaluation steps performed at task boundaries. CLA-R, Osiris-R, MinRed, and CLA-E form a tight cluster ranging from 4 to 14 hours across all configurations. CLIMB presents a moderate but constant overhead relative to this cluster: the gap with CLA-E remains stable at approximately 2.6 to 3.0 hours regardless of the number of tasks. SCALE constitutes a clea outlier, requiring 3 to 4 times more training time than CLIMB, likely due to its costly memory update procedure.

## G CLIMB WITH SIMSIAM

Table 8: Classification performances with SimSiam on Split CIFAR-100 and Split ImageNet-100 (5 seeds).

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2">CIFAR-100</td><td colspan="2">ImageNet-100</td></tr><tr><td>CA</td><td>FA</td><td>CA</td><td>FA</td></tr><tr><td></td><td>i.i.d.</td><td>—</td><td>43.12±0.56</td><td>—</td><td>49.75±0.98</td></tr><tr><td rowspan="4">20 tasks</td><td>MinRed</td><td>33.11±1.12</td><td>37.34±2.50</td><td>34.58±2.33</td><td>42.72±1.78</td></tr><tr><td>CLA-R</td><td>32.28±1.61</td><td>33.21±3.79</td><td>36.49±1.56</td><td>38.99±4.65</td></tr><tr><td>CLA-E</td><td>31.01±0.88</td><td>33.31±3.52</td><td>34.23±2.46</td><td>41.25±1.72</td></tr><tr><td>CLIMB</td><td>31.62±1.34</td><td>35.88±1.00</td><td>39.21±1.75</td><td>44.40±1.30</td></tr><tr><td rowspan="4">50 tasks</td><td>MinRed</td><td>32.57±1.42</td><td>38.49±0.81</td><td>33.88±2.32</td><td>41.61±1.46</td></tr><tr><td>CLA-R</td><td>31.80±1.55</td><td>32.45±4.14</td><td>35.39±2.31</td><td>39.82±1.94</td></tr><tr><td>CLA-E</td><td>29.58±1.17</td><td>31.57±3.77</td><td>33.48±2.42</td><td>41.11±1.21</td></tr><tr><td>CLIMB</td><td>31.35±1.15</td><td>37.04±0.78</td><td>37.51±2.13</td><td>41.79±1.97</td></tr></table>

Tables 8 report the performance of CLIMB and the main baselines using SimSiam (Chen & He, 2021) as the base SSL method on Split CIFAR-100 and Split ImageNet-100 respectively. Hyperparameters for CLIMB were selected via grid search on each dataset independently, yielding lr = 0.2 and λ = 0.1 in both cases. Hyperparameters for all baselines correspond to their best-performing configurations.

On Split CIFAR-100, all methods remain far below the i.i.d. upper bound, and CLA-R and CLA-E exhibit high variance in FA (standard deviations of 3–4%), suggesting instability of SimSiam under these methods’ distillation dynamics. CLIMB achieves competitive CA alongside MinRed and CLA-R at both 20 and 50 tasks, and ranks among the top methods in FA at 20 tasks.

On Split ImageNet-100, CLIMB consistently outperforms all other methods in CA at both 20 and 50 tasks, and leads in FA at 20 tasks alongside MinRed, mirroring the pattern observed under SimCLR. At 50 tasks, all methods are statistically indistinguishable in FA while CLIMB and CLA-R form the top group in CA. The consistency between the two SSL methods suggests that CLIMB’s advantage stems from its memory structure rather than from properties specific to contrastive learning.

## H FORGETTING METRIC

Table 9: Forgetting metric (%) on Split CIFAR-100 and Split ImageNet-100 (5 seeds, lower is better).

<table><tr><td></td><td>Method</td><td>CIFAR-100</td><td>ImageNet-100</td></tr><tr><td rowspan="6">20 tasks</td><td>SCALE</td><td>-4.03±0.84</td><td>-4.46±1.87</td></tr><tr><td>Osiris-R</td><td>-3.79±0.91</td><td>-4.26±1.14</td></tr><tr><td>MinRed</td><td>-4.24±1.04</td><td>-5.64±0.65</td></tr><tr><td>CLA-R</td><td>-1.92±1.70</td><td>-3.06±0.71</td></tr><tr><td>CLA-E</td><td>-2.47±1.18</td><td>-1.62±1.09</td></tr><tr><td>CLIMB</td><td>-3.26±0.86</td><td>-3.06±1.12</td></tr><tr><td rowspan="6">50 tasks</td><td>SCALE</td><td>-3.30±3.03</td><td>-2.96±2.78</td></tr><tr><td>Osiris-R</td><td>-4.59±0.88</td><td>-2.39±1.02</td></tr><tr><td>MinRed</td><td>-5.16±1.01</td><td>-7.29±1.12</td></tr><tr><td>CLA-R</td><td>-2.42±1.70</td><td>-4.12±1.45</td></tr><tr><td>CLA-E</td><td>-3.52±1.74</td><td>-1.74±1.61</td></tr><tr><td>CLIMB</td><td>-4.37±0.70</td><td>-3.66±0.95</td></tr><tr><td rowspan="6">100 tasks</td><td>SCALE</td><td>-4.58±1.32</td><td>-3.74±2.96</td></tr><tr><td>Osiris-R</td><td>-4.39±1.60</td><td>-3.62±1.58</td></tr><tr><td>MinRed</td><td>-5.57±1.87</td><td>-7.01±1.06</td></tr><tr><td>CLA-R</td><td>-3.25±1.98</td><td>-5.37±1.59</td></tr><tr><td>CLA-E</td><td>-4.35±1.28</td><td>-3.14±1.87</td></tr><tr><td>CLIMB</td><td>-4.66±0.60</td><td>-5.13±0.58</td></tr></table>

Table 9 reports the forgetting metric for all methods, defined as:

$$
\text { Forgetting } = \frac {1}{T - 1} \sum_ {i = 1} ^ {T - 1} (a _ {i, i} - a _ {T, i})\tag{4}
$$

where $a _ { i , j }$ is the evaluation of task $j$ after learning task i, and $a _ { T , i }$ its accuracy at the end of the stream. Negative values indicate that representations of past tasks improved over training rather than degraded. All methods exhibit negative forgetting across all configurations, indicating that representations of past tasks continue to improve throughout training rather than degrade. This suggests that replay not only prevents catastrophic forgetting but also enables backward transfer, where learning new tasks benefits the representations of previously seen ones. Note that a strongly negative forgetting score does not imply superior overall performance: it reflects a large accuracy gain on past tasks between their initial and final evaluation, but says nothing about the absolute level of those final accuracies. A method that starts from a low initial accuracy on each task and improves substantially over time may exhibit strong negative forgetting while still achieving lower CA and FA than a method that learns better representations from the start.