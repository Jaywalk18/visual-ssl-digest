# Lifelong Representations: A Survey on Continual Self-Supervised Learning for Vision Models

Sergi Masip<sup>∗</sup>, Alicja Dobrzeniecka<sup>†</sup>, Jonathan Swinnen<sup>∗</sup>, Joachim Collin<sup>∗</sup>, Bartłomiej Twardowski <sup>‡</sup>, Szymon Łukasik<sup>†</sup> and Tinne Tuytelaars<sup>∗</sup>

<sup>∗</sup>PSI, KU Leuven, Leuven, Belgium

Email: {sergi.masipcabeza, jonathan.swinnen, joachim.collin, tinne.tuytelaars}@kuleuven.be

<sup>†</sup>NASK - National Research Institute, Warsaw, Poland

Email: {alicja.dobrzeniecka, szymon.lukasik}@nask.pl <sup>‡</sup> Computer Vision Center, Universitat Autonoma de Barcelona, Barcelona, Spain IDEAS Research Institute, Warsaw, Poland Email: btwardowski@cvc.uab.cat

Abstract—Traditionally, continual learning has assumed access to labeled data, yet many real-world applications—such as lifelong robotics—require models to adapt continuously from unlabeled streams. This has led to the development of continual self-supervised learning (CSSL), a rapidly growing area that lacks a dedicated, systematic review. In this work, we present a comprehensive survey of CSSL for vision, with connections to emerging vision-language settings. First, we analyze existing evaluation protocols and highlight inconsistencies that hinder fair comparison. We then examine why self-supervised objectives exhibit improved robustness to catastrophic forgetting, relating this to task-agnostic representations and smoother loss landscapes. Next, we organize existing methods into a unified taxonomy based on their forgetting-mitigation strategies, including distillation, replay, regularization, architectural approaches, model merging, and objective-level adaptation. Finally, we identify open challenges such as scalability and the need for fast adaptability. We argue that advancing CSSL requires moving beyond small-scale benchmarks towards continual pre-training paradigms for large-scale systems.

Index Terms—self-supervised learning, representation learning, continual learning, online learning, pretraining, computer vision

## I. INTRODUCTION

Most machine learning models are trained under a static paradigm, assuming the entire training dataset is available upfront. However, many real-world deployment scenarios challenge this assumption. Consider, for example, a robot navigation policy powered by a vision foundation model backbone that has been trained for logistics and delivery in North America and Europe. These environments are well represented in current datasets [1]. However, when deployed in, for example, Sub-Saharan Africa, the model encounters new object appearances, road layouts and lighting conditions that were largely absent from its training data. The industry standard is to periodically retrain the model from scratch, which is wasteful. Yet the alternative–continually adapting the model– poses a fundamental challenge: how can new knowledge be incorporated without overwriting the old?

Continual Learning (CL) addresses this challenge by enabling models to learn from data as it arrives over time, while also reducing the risk of catastrophic forgetting. [2]. However, most existing studies on CL focus on supervised settings, such as class-incremental learning [3], thereby sidestepping a key practical issue: in real-world deployments, labels are scarce, expensive, and often unavailable. A robot adapting to a new environment cannot pause to wait for human annotators. A foundation model ingesting a new data stream cannot assume curated, task-annotated inputs. Unlabeled visual data, by contrast, is generated at massive scales wherever these systems are deployed. Self-supervised learning, which extracts supervisory signals from the structure of the data itself [4], is therefore a natural alternative for continual adaptation of models in the wild.

![](images/71ce5c9820cb14d18c519e68a9911e62b2b8bf88d6da9fc8750ee6ec4e5bf7c5.jpg)  
Fig. 1: A conceptual visualization of CSSL. The bottom part represents tasks T arriving in sequence, and the top part shows an illustration of an SSL objective (in this example, non-contrastive).

This has driven growing interest in continual self-supervised learning (CSSL), in which a model is continually pre-trained as new streams of data arrive, without explicit labels (Figure 1). This setting remains underexplored relative to its supervised counterpart and is almost entirely unexplored at foundationmodel scale [5]. While continual learning has been surveyed for large language models [6] and for the finetuning of visionlanguage models (VLMs) [7], CSSL in vision and VLMs lacks a dedicated review. The closest works [8], [9] treat CSSL only as a secondary topic and do not discuss concrete protocols, metrics, benchmarks, theory, or recent advances. Because modern applications increasingly rely on multimodal streams rather than isolated unimodal encoders, restricting our analysis to pure vision would ignore the reality of deployed systems. To fill this gap, we present the first systematic review of CSSL for vision models, while naturally extending our analysis to vision-language models (VLMs) where multimodal streams intersect with visual representations. Our contributions are as follows:

• Evaluation analysis. We gather and review typical evaluation protocols, benchmarks, and metrics (Section II).

• Insight into forgetting in SSL. We examine why selfsupervised objectives appear more robust to catastrophic forgetting than supervised ones (Section III).

• Taxonomy of methods. We categorize CSSL approaches by their forgetting-mitigation strategies (Section IV). Additionally, we examine how SSL can be a useful component within hybrid, supervised CL pipelines (Section V).

• Identifying open questions. We finally identify the key challenges that must still be addressed, and discuss how these approaches can be scaled to larger foundation model regimes (Section VI).

With this work, our aim is not only to map the current landscape, but also to position CSSL as a promising foundation for lifelong, large-scale, deployed systems.

## II. OBJECTIVES, PROTOCOLS AND EVALUATION

This section summarizes the main families of self-supervised objectives, training protocols, and evaluation metrics.

## A. Self-supervised learning objectives

Self-supervised learning methods train an encoder to produce useful representations without requiring manual labels by solving pretext tasks whose supervisory signal is derived directly from the data [4]. Modern SSL approaches are commonly categorized into contrastive, non-contrastive, and masked modeling methods, which correspond broadly to contrastive, predictive, and generative paradigms in prior surveys [10]. In our survey, we focus on continual learning with objectives within these families.

a) Contrastive learning: Contrastive methods learn representations by aligning augmented views of the same image (positive pairs) while distinguishing them from different samples (negative pairs). These approaches were popularized by SimCLR [11] and MoCo [12], [13] with the InfoNCE objective [14].

b) Non-contrastive learning: Explicit negatives for alignment are not strictly necessary to learn useful representations. Non-contrastive approaches avoid collapse via alternative mechanisms: self-distillation methods (BYOL [15], SimSiam [16]) match views using asymmetric predictors and stop-gradients;

redundancy-reduction techniques impose statistical constraints, such as minimizing cross-correlation off-diagonals (Barlow Twins [17]) or decorrelating features (VICReg [18]); and clustering methods (SwAV [19]) map representations to learnable prototypes to predict one view’s cluster assignment from another.

c) Masked modeling: Inspired by NLP, masked image modeling (MIM) captures global structure by reconstructing missing parts of the input. Masked Autoencoders (MAE) [20] process visible patches via a ViT encoder, training a lightweight decoder to reconstruct masked regions.

## B. Training protocols

We define the training protocols based on whether task boundaries are defined and how.

a) Offline CSSL: The standard evaluation protocol in continual self-supervised learning. Data arrive as a sequence of discrete experiences, each presented in full before the next begins. Task boundaries are typically explicit, indicating when a new experience starts, and models are allowed to train on this experience for either a set amount of epochs or until convergence. Most methods in the literature adopt this setting and evaluate representation quality after each experience.

b) Online / task-free CSSL: In online CSSL, data arrive as a single stream of mini-batches, typically processed in a one-pass fashion without revisiting samples. Task identities are not provided, and boundaries may be absent or blurred. This setting introduces additional computational constraints and is therefore more challenging. The computational budget can be measured with, for example, the Cumulative Backward Passes (CBP) [21] metric, a compute-normalized metric for comparing methods under strict online training budgets.

## C. Evaluation metrics

Evaluating CSSL typically involves either probing representations or analyzing their structural stability. The most common probe types are linear probing, which measures linear separability, and k-NN accuracy, which evaluates whether samples from the same class form coherent local neighborhoods.

a) Standard CL metrics: Average Accuracy (AA) measures the final mean performance across all T tasks. Forgetting (F) and Backward Transfer (BWT) evaluate the impact of learning new tasks on past tasks; F captures strict performance degradation, while signed BWT captures both interference and beneficial transfer. Conversely, Forward Transfer (FWT) measures how prior knowledge facilitates learning new tasks.

b) Multimodal metrics: In large-scale vision-language settings, metrics like Average Knowledge Accumulation (AKA) and Average Zero-Shot Retention (AZS) jointly capture the plasticity–stability trade-off across held-in and held-out task sets [22], [23]. Both metrics are typically derived from accuracy and retrieval performance.

c) Representation stability (CKA): To diagnose where in the network forgetting occurs [24], several works analyze the learned representation space using Centered Kernel Alignment (CKA) [25]. CKA measures the similarity of activation Gram matrices between models or checkpoints; values near 1 indicate nearly identical representations up to orthogonal transformation, while values near 0 indicate substantial divergence.

TABLE I: Common benchmarks used in continual selfsupervised learning, grouped by evaluation setting.

<table><tr><td>Benchmark</td><td>Protocol</td><td>Modality</td><td>Scale</td></tr><tr><td colspan="4">Vision Benchmarks</td></tr><tr><td>CIFAR-10/100 [26]</td><td>Offline/Online</td><td>Vision</td><td>Small</td></tr><tr><td>ImageNet-100 [27]</td><td>Offline/Online</td><td>Vision</td><td>Medium</td></tr><tr><td>TinyImageNet [27]</td><td>Offline/Online</td><td>Vision</td><td>Medium</td></tr><tr><td colspan="4">Multimodal Benchmarks</td></tr><tr><td>TiC-(DataComp, YFCC, RedCaps) [23]</td><td>Offline</td><td>Vision-Language</td><td>Large</td></tr><tr><td>FoMo-in-Flux [22]</td><td>Offline</td><td>Vision-Language</td><td>Large</td></tr></table>

## D. Benchmarks

The most widely used benchmarks split standard classification datasets into a sequence of tasks by class. In Table I, we have classified the most common benchmarks based on the modality they evaluate.

## III. PROPERTIES OF SSL IN CONTINUAL LEARNING SETUPS

A recurrent observation when running self-supervised methods in a continual learning setup is that the representations seem to be more robust to forgetting than when using supervised losses [28]–[30]. Studies have empirically shown, using CKA, that the representations remain remarkably stable throughout training. Before examining the continual learning approaches applied to CSSL, we review why self-supervised learning is less susceptible to catastrophic forgetting per se.

## A. Explaining the robustness of SSL representations

Empirical and analytical studies suggest that the robustness of SSL models to forgetting stems from two core properties: their representations are task-agnostic, and their loss landscapes are smoother and flatter. This perspective clarifies why forgetting is less problematic in CSSL and highlights the key challenges that remain.

a) Supervision, representation collapse, and tunnel effect: During supervised classification training, representations often suffer from a loss of informational richness, manifesting in two related phenomena. First, representations often exhibit neural collapse, where samples of the same class converge to a single point in representation space, and class means form a highly symmetric structure [31], [32]. While this geometry improves linear separability for the training task, it suppresses withinclass variability and discards features that may be useful for other tasks [33]. Second, a related phenomenon, known as the tunnel effect [34], has also been observed in supervised models, where deeper representations collapse to a progressively lower rank subspace. This effect, which worsens when training with few classes (as is common in traditional class-incremental learning), reduces out-of-distribution accuracy and causes significant forgetting in continual learning [35].

b) Self-supervision learns task-agnostic features: On the other hand, SSL objectives encourage the network to learn task-agnostic, descriptive features about the data distribution itself [36]. Self-supervised representation geometry seems to have higher effective dimensionality and less class-wise collapse than supervised models [37], [38]. In fact, many SSL methods are explicitly designed to prevent representational collapse during training [15]–[18], [39], [40], and a higher rank tends to correlate with better downstream performance [41].

c) Flatter minima in the loss landscape: The robustness to forgetting can also be explained by looking at the loss landscape. Several works [28], [42], [43] have observed that self-supervised objectives sculpt a smoother loss landscape compared to supervised objectives. On top of this, the solutions reached by these objectives are generally flatter, which has benefits against forgetting, as it increases the tolerance to parameter shifts.

## B. The challenges of continual self-supervised learning

a) Data and compute hunger: A first limitation shared with SSL in general is that these methods are data-hungry, requiring more iterations and data to converge than their supervised counterparts [44]–[46]. This is further aggravated in the continual setting, where small per-task datasets have been shown to cause significant instability [30]. Compounding this, SSL carries a substantial computational overhead: long training schedules, large batch sizes, and auxiliary components such as momentum encoders.

b) Stability vs. plasticity dilemma: Regarding continual learning, catastrophic forgetting remains an open problem. The performance gap with respect to joint training—commonly used as an upper bound—is still present and has been observed to widen as the number of tasks grows [29], [30], [47], [48]. At the same time, several works argue that plasticity deserves at least equal attention as a research focus [48], [49], as the two objectives are inherently in tension and the absence of a supervised signal makes balancing the stability-plasticity trade-off particularly challenging in CSSL.

c) Risk of overfitting: Rehearsal-based methods (see Section IV-C) face an additional challenge: because SSL requires far more training epochs than supervised learning, memory samples are revisited many more times, increasing the risk of overfitting to the replay buffer and undermining its purpose [30], [50].

d) Cross-modal drift in multimodal settings: In multimodal continual learning, particularly for vision–language models, an additional challenge arises from maintaining the alignment of modalities over time. As models are updated with new data, the shared embedding space between modalities can drift. This leads to a decline in cross-modal retrieval performance and zero-shot generalization [22], [23]. Unlike in unimodal settings, where forgetting primarily affects feature representations, multimodal models must preserve both intramodal structure and inter-modal consistency.

Figure 2 summarises these key challenges and highlights the methods proposed to address each of them.

![](images/89fe623535be499c75d7f61b62f2cc058ab06b020cbb7b6ede95436a5db6a2c6.jpg)  
Fig. 2: Challenges in CSSL and representative solutions.

## IV. CONTINUAL SELF-SUPERVISED LEARNING

CSSL methods aim to maintain stable and transferable representations while expanding the model’s knowledge. As discussed in Section III, despite increased representational stability, models still suffer from gradual forgetting and loss of plasticity when trained through CSSL. In this section, we gather and discuss papers that are strictly self-supervised, i.e., no labels were used during training. We identify six main families of approaches: distillation, weight regularization, replay, architectural methods, model merging, and objectivelevel adaptation. We illustrate the different approaches in Figure 3.

## A. Distillation

Knowledge distillation has emerged as a promising strategy for mitigating catastrophic forgetting in CSSL. Traditionally, distillation aimed to maintain stability by preserving the feature structure learned in previous tasks.

MedCoSS [53] explored training a shared MAE encoder sequentially on different medical data modalities, using a distillation loss on representations of samples from past tasks. Contrastive Continual Learning (CCL) [58], on the other hand, distills similarity matrices of sample representations directly from a momentum teacher network. They found that this improves performance, although the gains are modest in general. In vision-language domains, distillation is commonly used to preserve cross-modal alignment over time. Most approaches adopt cross-modal similarity distillation, matching the softmax-normalized similarity distributions of a previous model using KL divergence on both image-to-text and textto-image pairs [23], [55], [56], thereby preserving the shared embedding space structure and mitigating cross-modal drift. Mod-X [57] operates directly on non-normalized similarity matrices to enforce strong geometric consistency. CTP [56] extends similarity distillation to intra-modal pairs to preserve modal structure, and adds a contrastive loss between the current model and an EMA teacher to stabilize representation drift.

In unimodal vision settings, it has been observed that naive distillation, where features are matched directly, severely restricts model plasticity by anchoring representations too rigidly to the past, stripping the model’s ability to adapt to new data distributions [21], [30], [47]. To overcome this limitation, recent methodologies have adopted projected distillation, incorporating an auxiliary projector or predictor network into the distillation process, which acts as a buffer against distribution shifts [29]. This preserves the encoder’s plasticity as long as new representations can be mapped back to older ones [47], a representational benefit significant enough even to improve standard supervised baselines [59].

![](images/fbc093a163537b82bf7920602908a80f308b35f29d6d4ede5609178862d317d3.jpg)  
Fig. 3: The diagram illustrates the different approaches to CSSL: (A) A distillation loss $( \mathcal { L } ^ { \mathrm { { D i s } } } )$ is used to align representations of the current model with those of the previous one $( T - 1 )$ with an optional projector; (B) a weight regularization loss $( \mathcal { L } ^ { \mathrm { R e g } } )$ penalizes changes to parameters that were important for past tasks, computed by comparing the current model (T ) against the frozen previous model $( T - 1 ) \colon$ ; (C) incoming data at step T consists of new images combined with a replay memory buffer of past samples; (D) a parameter-isolated architecture separates frozen and trainable components, enabling taskspecific plasticity while preserving prior knowledge; (E) a model merging module combines the previous and current model checkpoints; (F) the overall training objective optionally adapts $( \mathcal { L } ^ { \mathrm { { a d } } } )$ the self-supervised loss $( \mathcal { L } ^ { \mathrm { { S S L } } } )$ to embed continual learning constraints directly into the SSL objective.

Pioneering frameworks such as CaSSLe [30] and Projected Functional Regularization (PFR) [47] map current representations back to the previous task model via a projector, thereby explicitly freeing the encoder to learn new features. Building on CaSSLe, Pseudo-Negative Regularization (PNR) [60] adapts projected distillation by incorporating the current model’s outputs as pseudo-negatives, explicitly preventing new representations from overlapping with previously learned ones. Extensions like Plasticity-Optimized Complementary Networks [48] apply this to transfer knowledge from a taskspecific feature extractor to a cumulative knowledge extractor in a complementary learning system. One limitation is that these methods depend on task boundaries. This motivates Continuous Latent Alignment (CLA) [21] to adapt the CaSSLe framework to the online setting by replacing the previous task model with an EMA of the model. However, CLA scales poorly with increased compute and primarily benefits early pretraining, acting more as a representational regularizer than a complete continual learning solution.

Projected distillation has also been adapted across different modalities. For example, in remote sensing, CoSMAE [54] continually pretrains an MAE while applying a model mixup strategy, interpolating between the current and previous models, and aligning the current model with this mixed teacher via a projector. In vision-language domains, C-CLIP [61] performs contrastive distillation in a projected joint embedding space to avoid over-constraining the updated features.

Takeaway: These works demonstrate that augmenting distillation with a projector is a highly effective approach for balancing stability and plasticity in CSSL. However, practitioners must note that this family of approaches increases the computational overhead by adding a teacher network and a projector, requiring at a minimum one additional forward pass. Improving the efficiency of these methods is an open question.

## B. Weight regularization

Weight regularization techniques, such as EWC [62], SI [63], and MAS [64], mitigate catastrophic forgetting by penalizing changes to parameters deemed critical for previous tasks.

EWC is frequently utilized to consolidate weights during unsupervised incremental phases. For instance, VINIL [65] combines EWC with memory replay to manage long-horizon visual instance learning without manual labels. In remote sensing, Continual Barlow Twins [66] trains a Barlow Twins backbone [17] and employs EWC to preserve representations during incremental pretraining on heterogeneous earth observation datasets. SI performs particularly well when integrated with unsupervised visual representation methods like SimSiam and Barlow Twins [28]. Similarly, MAS parameter regularization has been applied to MoCo-v2 [13] and BYOL [15] to close the performance gap between sequential and joint training on streaming data [42].

These methods have also been benchmarked in large-scale multimodal continual pretraining, where EWC and SI are frequently used as baselines [22], [23], [56], [57], [61], [67]. However, empirical studies show that they underperform compared to alternative approaches.

Takeaway: Although weight regularization may occasionally improve knowledge retention over other methods, these gains come at the cost of poor plasticity.

## C. Replay methods

Replay-based approaches maintain a memory buffer of past samples for reuse during the training of new tasks. The key differences between methods lie both in how these replay samples are used to mitigate forgetting and how they are selected and maintained over time.

A straightforward strategy is to directly concatenate some replay samples to each training batch. For instance, Singh et al. [68] continually pre-trained an MAE in this manner using Experience Replay (ER) [69], and observed low forgetting when combined with an infinite learning rate schedule. In multimodal models [7], such as for example TiC-CLIP [23], ER has also been shown to help significantly. In CTP [56] and RetCoP [67] they also observe that the performance of their method improves a lot when including replay. IncCLIP [55] also uses ER, and additionally generates difficult pseudonegative text samples for their contrastive loss through model inversion. Finally, in FoMo-in-Flux [22], they experiment with an infinite buffer size in addition to access to the original pretraining pool, but they only allow finite compute. They show that in this setting, performance depends strongly on the training mixture: replaying buffered data is critical, replaying the pretraining pool has a limited effect, and increasing the ratio of replay samples versus new task data to make training more IID improves the accumulation-retention trade-off.

Rather than naively adding replay samples to each training batch, many methods incorporate replay samples through additional objectives, most commonly via distillation or contrastive losses. In CCL [58], for example, replay samples are used both for distillation and for an auxiliary contrastive loss that separates representations of old and new tasks. Their buffer is constructed by selecting representative samples through K-Means clustering in embedding space. Similarly, MedCoSS [53] also adopts a K-Means-based buffer for feature distillation, and further applies MixUp [70] augmentation between samples of the same modality within the buffer.

A different line of work focuses on mixing replay and current samples via interpolation. A first work, LUMP [28], uses MixUp to interpolate between samples from new tasks and the replay buffer. Similarly, Cro-Mo Mixup [52] adopts this same MixUp scheme and combines it with a Cross-Model Feature Mixup loss. CoSMAE [54] also incorporates this data mixup strategy alongside distillation. Finally, ASCL [50] also builds on LUMP, combining it with their adaptive sharpnessaware minimization (ASAM) to reduce overfitting, as well as their adaptive memory enhancement (AME), which applies augmentations of varying strength to increase buffer diversity.

Beyond how replay is used, several works investigate strategies for selecting informative samples. ASR [71], for example, selects a mixture of representative samples, whose representations remain stable under augmentation, and discriminative samples, whose representations vary the most. These are then used in their C<sup>2</sup>ASR contrastive loss to maintain consistency with past tasks while separating them from new ones. CUCL [72] instead jointly trains their SSL objective along with additional codebooks for soft vector quantization of the representations. The samples whose representations are furthest away from their corresponding codewords are deemed the most difficult and informative, and are thus selected for replay. Finally, EDSR [73] selects a small subset of replay samples per task by maximizing their Mutual Information with the full task dataset from an information-theoretic perspective.

They also use these replay samples for distillation and mitigate potential overfitting on the replay buffer by making the replayed teacher outputs noisy.

In online CSSL settings, replay buffers serve multiple roles. First, they allow samples to be revisited multiple times instead of being observed only once. Second, they help shuffle and diversify the incoming data stream, which is particularly important when dealing with highly non-IID data such as video streams. Importantly, in this setting, sample selection must happen online, without access to the full task split and often without clear task boundaries, which requires strategies different from those used in offline CL. In a work by Imai et al. [46], for example, they keep a diverse buffer by retaining only samples with the lowest cosine similarity in feature space, reducing biased updates caused by non-IID data. In contrast, CLA [21] finds that a simple FIFO buffer performs better for their method. They argue that this makes the expected number of training iterations for each sample equal, which improves convergence and reduces overfitting on buffer samples, but this could come at the cost of a higher recency bias. Memory Storyboard [74] also uses a short-term FIFO buffer, but adds a larger long-term reservoir sampling memory on top of it to reduce such recency bias.

Takeaway: Replay has often been shown to be an important component in CSSL pipelines. However, it should be noted that this comes at a cost. Old samples must be stored, which could cause problems in memory-constrained settings or when there are privacy concerns. Additionally, processing these extra replay samples can cost a significant amount of additional training compute.

## D. Architectural methods

Architectural methods mitigate catastrophic forgetting at the parameter level by reducing interference between tasks. Rather than forcing a single shared set of weights to accommodate all tasks, these approaches freeze subsets of parameters or introduce task-specific components. This preserves previously learned representations while enabling controlled plasticity for new data.

In the context of CSSL, a representative example is Branch-Tuning [24], which adds temporary, task-specific branches to existing network layers. During training on a new task, a parallel branch is added to the frozen backbone, and its outputs are combined. After training, these additional parameters can be incorporated into the original network, thus enabling knowledge integration without overwriting prior representations directly.

Some works adapt the model capacity online to match the evolving data distribution. For example, KIERA [75] introduces a self-evolving deep clustering architecture, in which layers, nodes, and clusters are dynamically added during training. This enables the model to allocate new representational capacity to novel data while retaining previously learned structures.

Another approach exploits the hierarchical nature of deep networks by selectively freezing subsets of parameters. For example, PTLF [76] progressively freezes layers with high inter-task correlation, thereby reducing both forgetting and computational cost. This approach builds on the idea that initial representations in SSL remain consistent across tasks.

Finally, in vision-language continual learning, C-CLIP [61] applies LoRA [77] to both encoders of CLIP [78], bounding weight deviation to limit forgetting but restricting plasticity in the process. FoMo-in-Flux [22] draws a similar conclusion: the number of trainable parameters strongly impacts the stability–plasticity trade-off, with parameter-selective methods showing strong retention but very limited accumulation, and parameter-efficient tuning approaches, such as LoRA, achieving a more favorable trade-off, but still exhibiting limited knowledge accumulation.

Takeaway: Architectural methods reduce catastrophic forgetting by isolating parameters through freezing or improve plasticity by introducing task-specific modules. However, this often comes at the cost of limited scalability and reduced knowledge integration. Methods that freeze parts of the model prioritize stability over plasticity by restricting which parameters are updated, which makes adapting to new data more difficult. Furthermore, many methods depend on distinct task boundaries and may perform poorly in online settings. Overall, these methods demonstrate that freezing parameter updates is effective for retention and expanding the architecture improves plasticity, but both require careful design to balance flexibility and efficiency.

## E. Model merging

Model merging has emerged as a computationally lightweight strategy to consolidate knowledge across fine-tuned models by directly aggregating their parameters. While most merging methods assume that all expert models are available simultane ously, continual learning introduces a sequential setting where models must be integrated over time.

In continual pretraining of CLIP models, TiC-CLIP [23] evaluates a recursive merging strategy, where models are sequentially combined using interpolation coefficients tuned to preserve performance on previous tasks, but showed limited gains compared to replay-based approaches. FoMo-in-Flux [22] shows that, together with replay, merging yields the strongest performance among evaluated methods. In particular, EMA merging and continual zero-shot merging achieve strong knowledge accumulation while maintaining neutral or improved zero-shot retention, especially over shorter training horizons. Building on these findings, TIME [51] provides a systematic analysis of temporal model merging, studying its use at both initialization and evaluation. They show that EMA-based strategies offer the best plasticity–stability trade-off, with limited sensitivity to the specific merging rule, and scale well across model sizes, compute budgets, and task sequence lengths.

Takeaway: Model merging appears to be a promising and computationally efficient approach, including at scale, showing favorable behavior across both short and long update horizons, but has so far received limited attention in CSSL.

## F. Objective-level adaptation

While most CSSL methods rely on mechanisms such as replay or distillation, a nascent line of work explores modifying the self-supervised objective itself to improve stability on nonstationary data streams, thereby reducing reliance on external components such as memory buffers or additional parameters in the form of teacher models.

A recent work, Continual Multi-Patch Learning (CMP) [79], generates multiple patches from a single input and enforces consistency among their representations while preventing collapse through the Total Coding Rate (TCR) regularizer. Prior work by Imai et al. [46] used a similar setup in combination with replay, but CMP was shown to be effective even without requiring stored past samples, highlighting how objective design can substitute for explicit memory mechanisms.

In the spirit of tweaking the SSL objective, InfoUCL [80] introduces an InfoDrop contrastive loss to mitigate the texture bias inherent in CSSL methods, improving generalization and reducing forgetting. Similarly, PNR [60] mathematically redefines both contrastive and non-contrastive objectives to explicitly push new representations away from pseudo-negatives derived from the previous model. While both approaches embed continual learning constraints directly into the representation learning phase, they still fundamentally rely on external continual learning strategies.

Takeaway: Objective-level adaptation represents a promising and underexplored direction for CSSL. By embedding CL constraints directly into the self-supervised objective, it offers an elegant pathway that bridges the two fields more tightly than approaches that address forgetting through external mechanisms.

## V. SSL AS A COMPONENT IN HYBRID PIPELINES

The robustness properties of SSL described in Section III suggest a natural question: can self-supervised objectives improve supervised continual learning, even when labels are available? Empirical evidence consistently answers yes.

An early work by Zhang et al. [81] showed that augmenting cross-entropy training with auxiliary self-supervised pretext tasks yields more generalizable, task-agnostic features that serve as a superior foundation for subsequent incremental classes. DualNet [82] formalizes this intuition through a complementary learning system: a slow learner continuously optimizes a Barlow Twins objective to build general representations, while a fast learner adapts these for specific supervised classification tasks. Similarly, SscFA [83] combines a similar architecture with a MAS weight regularization loss. Kaizen [84] uses a setup similar to the fully self-supervised CaSSLe, training a self-supervised representation with replay and knowledge distillation. In addition, it adds a classifier on top. This classifier is trained during the task sequence whenever a sample has a label and also uses knowledge distillation and replay to mitigate forgetting.

MAEs have proven particularly versatile in hybrid settings as well, as their decoder can serve double duty: MAE-CIL [85] exploits the MAE reconstruction to compress and decompress replay buffers, effectively increasing their replay buffer size without additional storage cost. In pMAE [86], a similar idea is used in federated CL using prompt tuning.

Collectively, these works suggest that the benefits of SSL are not exclusive to fully unsupervised pipelines; they transfer meaningfully to supervised settings as well.

## VI. OPEN CHALLENGES AND FUTURE DIRECTIONS

Despite rapid progress, CSSL is still in an early stage of development, with many core challenges remaining unresolved. While existing methods often demonstrate promising results on small-scale benchmarks, it is unclear whether these approaches can be applied to long-horizon or foundation model settings. A key issue is that much of the current literature adapts techniques from supervised CL without fully accounting for the unique properties of self-supervised objectives and representation learning. Consequently, fundamental questions remain regarding how to evaluate, scale, and control representation dynamics in non-stationary environments. We identify several critical challenges that define the current limitations of CSSL and outline directions for future research.

## A. Rethinking evaluation protocols and metrics

A key limitation of current CSSL research is the lack of standardised evaluation protocols. As discussed in Section II, existing works differ substantially in terms of task construction (class-incremental versus stream-based), evaluation settings (offline versus online), and probe strategies (e.g., linear versus k-NN versus fine-tuning). This leads to inconsistent and often incomparable results. This variation makes it challenging to determine whether the reported improvements are due to methodological enhancements or variations in evaluation frameworks.

Furthermore, commonly used metrics such as linear probing or k-NN accuracy only offer a partial view of representation quality (linear separability and clustering), particularly in continual settings where adaptability, robustness to distribution shifts, and long-term retention are essential. This limitation is particularly evident in methods such as masked image modelling, where fine-tuning is a more accurate reflection of performance than probing [20].

Crucially, forgetting and plasticity should also be analyzed from a purely representational perspective so they can be evaluated even under label scarcity. For example, one could measure uniformity and alignment [87] to track the structural health of representations over time, and monitor feature rank [41] to assess whether the model maintains a high plasticity.

Finally, we highlight the need for the development of unified, multidimensional evaluation protocols that assess representation quality, transferability, robustness, and efficiency consistently. Such benchmarks will be essential for driving meaningful progress and enabling CSSL to transition from controlled experimental setups to real-world deployment.

## B. Rethinking the stability–plasticity trade-off

Although SSL objectives are more robust to forgetting than those of supervised learning, forgetting still accumulates over long training periods, particularly when there are significant shifts in the data distribution (see Section III). Consequently, much of the CSSL literature prioritizes stability over plasticity. The empirical observations reviewed in Section IV suggest that stability-prioritizing methods, such as weight regularization or naive feature distillation, often reduce plasticity by overly constraining representation updates.

However, a growing body of work challenges this framing directly: studies on projected distillation [47] and complementary network architectures [48] each argues that overly rigid alignment with past representations is itself a failure mode, one that prevents the model from integrating genuinely novel information. Therefore, we argue that plasticity should be regarded as a problem of equal importance to forgetting in CSSL. This also opens a more precise research question: rather than uniformly minimizing forgetting, how can models distinguish between representations worth preserving and those that should be allowed to evolve, especially in unsupervised settings? Current methods treat all past knowledge as equally important, lacking mechanisms to identify transferable structure versus task-specific or obsolete features. Promising directions include parameter importance estimation conditioned on distribution shift, modular or compositional representations, and data-driven criteria for selectively updating features over time.

## C. Generalization of findings across SSL objectives

A largely overlooked question in CSSL is whether empirical findings generalize across SSL objective families. As discussed in Section II-A, contrastive, non-contrastive, and masked modeling approaches differ in their stability properties, yet many CSSL studies commit to a single family per contribution (e.g. [47], [54], [58]). Works that test across multiple SSL objectives, such as CaSSLe [30] and Branch-Tuning [24], reveal familyspecific behaviors rather than uniform responses to the same CL strategy. CaSSLe observes that BYOL is unstable under naive fine-tuning while other methods are not, and that SwAV’s prototype-based structure causes its representations to behave differently under k-NN probing compared to linear evaluation. Furthermore, distillation losses using L2-normalization conflict with Barlow Twins, which relies on standardization, leading to measurable performance degradation. In online settings, CLA [21] additionally highlights that contrastive objectives are substantially more sensitive to minibatch size than noncontrastive ones, an important constraint when replay buffers are limited. Yet despite observing these interactions, neither CaSSLe nor Branch-Tuning systematically analyze why the same CL strategy produces different outcomes across families. Establishing which CSSL findings are objective-agnostic and which are family-specific is a prerequisite for building generalpurpose continual pretraining approaches.

## D. Scaling to foundation models and long-horizon settings

A significant shortcoming of the existing literature is the discrepancy between experimental settings and real-world deployment scenarios. Except in multimodal settings, most CSSL methods are evaluated on small-scale benchmarks with simplified task constructions, and often rely on relatively small architectures such as ResNet-18 and datasets like CIFAR or TinyImageNet. There has been limited investigation into largescale or foundation model regimes in unimodal vision settings.

This raises concerns about the data and architecture scalability of existing approaches. As discussed in Section III-B and Section IV, memory overfitting is a problem in CSSL, and at larger scales, the computational costs of current approaches could be prohibitive within constrained budgets. Even if compute were not a constraint, it is uncertain whether current solutions would improve with it. CLA [21] is a concrete case where their approach does not scale with more compute, unless more replay data is used. Beyond raw compute, traditional SSL methods do not always translate seamlessly to ViT architectures [10]. Future research must investigate how modern ViT-native self-supervised paradigms beyond CLIP, such as DINO [39] or JEPA [40], interact with continual data streams. Progress in this area will require developing parameter-efficient update mechanisms and compute-constrained strategies aligned with the specific challenges of modern foundation models.

Nonetheless, scaling is not only a computational challenge: it also exposes a deeper underspecification in current work. Deployed systems like the delivery robot from the introduction (Section I) would need to continually update their representations and knowledge to adapt to new environments or task requirements. However, most methods optimize for short stream benchmarks (see Section II-D) where retaining recent task performance is sufficient, without addressing how representations should evolve over genuinely long horizons, selectively discarding outdated features while preserving transferable structure. Closing this gap requires not just larger experiments, but theoretical frameworks that characterize what reliable, reusable representations look like under persistent non-stationarity.

Finally, another promising direction would look beyond label scarcity to address missing modalities in multimodal scenarios, establishing frameworks that maintain robust representations when a modality (e.g., text captions) is intermittently or permanently unavailable in new incoming training data.

## VII. CONCLUSIONS

This survey has examined CSSL for learning representations from unlabeled data in dynamic environments. We have reviewed training protocols, evaluation strategies, and methods for mitigating forgetting in sequential data streams. Although self-supervised objectives produce more general and stable representations than supervised approaches—a robustness we traced to task-agnostic features and flatter loss landscapes— they do not fully solve continual adaptation. Mechanisms such as projected feature distillation, replay, and architectural methods are still necessary, but raise scalability concerns that become more acute as model and data scale increase. Looking forward, we argue that progress requires moving beyond treating CSSL as supervised continual learning without labels. The lack of unified evaluation protocols and open questions around cross-family generalization of findings or the behavior of ViT-native objectives under non-stationary streams each demand approaches tailored to the specific properties of SSL. Addressing these challenges is essential to unlock the potential of CSSL as a backbone for lifelong foundation models that can adapt continuously to an ever-changing world.

## ACKNOWLEDGEMENTS

We would like to thank Emanuele Sansone and Alvaro Budria for their insightful comments. This paper has received funding from the Flemish Government under the Methusalem Funding Scheme (grant agreement $\mathfrak { n } ^ { \circ }$ METH/24/009).

## REFERENCES

[1] V. V. Ramaswamy et al., “Geode: a geographically diverse evaluation dataset for object recognition,” Advances in Neural Information Process ing Systems, vol. 36, pp. 66 127–66 137, 2023.

[2] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks: The sequential learning problem,” ser. Psychology of Learning and Motivation, G. H. Bower, Ed. Academic Press, 1989, vol. 24, pp. 109–165.

[3] G. M. Van de Ven and A. S. Tolias, “Three continual learning scenarios,” in NeurIPS Continual Learning Workshop, vol. 1, no. 9, 2018, p. 4.

[4] L. Jing and Y. Tian, “ Self-Supervised Visual Feature Learning With Deep Neural Networks: A Survey ,” IEEE TPAMI, vol. 43, no. 11, pp. 4037–4058, Nov. 2021.

[5] J. Bell et al., “The future of continual learning in the era of foundation models: Three key directions,” 2025. [Online]. Available: https://arxiv.org/abs/2506.03320

[6] H. Shi et al., “Continual learning of large language models: A comprehensive survey,” ACM Comput. Surv., vol. 58, no. 5, Nov. 2025.

[7] Y. Liu et al., “Continual learning for vlms: A survey and taxonomy beyond forgetting,” 2025. [Online]. Available: https: //arxiv.org/abs/2508.04227

[8] B. Bagus, A. Gepperth, and T. Lesort, “Beyond supervised continual learning: a review,” 2022. [Online]. Available: https: //arxiv.org/abs/2208.14307

[9] M. Kilickaya, J. van de Weijer, and Y. M. Asano, “Towards label-efficient incremental learning: A survey,” 2023. [Online]. Available: https://arxiv.org/abs/2302.00353

[10] A. Khan et al., “A survey of the self supervised learning mechanisms for vision transformers,” 2025.

[11] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in Int. Conf. on Machine Learning, 2020, pp. 1597–1607.

[12] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for unsupervised visual representation learning,” in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 2020, pp. 9729–9738.

[13] X. Chen, H. Fan, R. Girshick, and K. He, “Improved baselines with momentum contrastive learning,” 2020. [Online]. Available: https://arxiv.org/abs/2003.04297

[14] A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” 2019. [Online]. Available: https://arxiv.org/abs/1807.03748

[15] J.-B. Grill et al., “Bootstrap your own latent-a new approach to selfsupervised learning,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 21 271–21 284.

[16] X. Chen and K. He, “Exploring simple siamese representation learning,” in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 2021, pp. 15 750–15 758.

[17] J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny, “Barlow twins: Self-supervised learning via redundancy reduction,” in Int. Conf. on Machine Learning. PMLR, 2021, pp. 12 310–12 320.

[18] A. Bardes, J. Ponce, and Y. LeCun, “Vicreg: Variance-invariancecovariance regularization for self-supervised learning,” in Int. Conf. on Learning Representations, 2022.

[19] M. Caron et al., “Unsupervised learning of visual features by contrasting cluster assignments,” vol. 33, 2020, pp. 9912–9924.

[20] K. He et al., “Masked autoencoders are scalable vision learners,” in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 2022, pp. 16 000–16 009.

[21] G. Cignoni, A. Cossu, A. Gomez-Villa, J. van de Weijer, and A. Carta, “Cla: Latent alignment for online continual self-supervised learning,” 2025. [Online]. Available: https://arxiv.org/abs/2507.10434

[22] V. Udandarao et al., “A practitioner’s guide to continual multimodal pretraining,” in Advances in Neural Information Processing Systems, 2024.

[23] S. Garg et al., “Tic-clip: Continual training of clip models,” in Int. Conf. on Learning Representations, 2024.

[24] W. Liu, F. Zhu, and C.-L. Liu, “Branch-tuning: balancing stability and plasticity for continual self-supervised learning,” IEEE Transactions on Neural Networks and Learning Systems, 2025.

[25] S. Kornblith, M. Norouzi, H. Lee, and G. Hinton, “Similarity of neural network representations revisited,” in Int. Conf. on Machine Learning. PMlR, 2019, pp. 3519–3529.

[26] A. Krizhevsky, “Learning multiple layers of features from tiny images,” pp. 32–33, 2009. [Online]. Available: https://www.cs.toronto.edu/\~kriz/ learning-features-2009-TR.pdf

[27] J. Deng et al., “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE Conf. on Computer Vision and Pattern Recognition. Ieee, 2009, pp. 248–255.

[28] D. Madaan, J. Yoon, Y. Li, Y. Liu, and S. J. Hwang, “Representational continuity for unsupervised continual learning,” in Int. Conf. on Learning Representations, 2022.

[29] Z. Ni, S. Tang, and Y. Zhuang, “Self-supervised class incremental learning,” 2021. [Online]. Available: https://arxiv.org/abs/2111.11208

[30] E. Fini et al., “Self-supervised models are continual learners,” in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 2022, pp. 9621–9630.

[31] V. Papyan, X. Han, and D. L. Donoho, “Prevalence of neural collapse during the terminal phase of deep learning training,” Proc. of the National Academy of Sciences, vol. 117, no. 40, pp. 24 652–24 663, 2020.

[32] T. Tirer, H. Huang, and J. Niles-Weed, “Perturbation analysis of neural collapse,” in Int. Conf. on Machine Learning. PMLR, 2023, pp. 34 301– 34 329.

[33] S. Kornblith, T. Chen, H. Lee, and M. Norouzi, “Why do better loss functions lead to less transferable features?” Advances in Neural Information Processing Systems, vol. 34, pp. 28 648–28 662, 2021.

[34] W. Masarczyk et al., “The tunnel effect: Building data representations in deep neural networks,” Advances in Neural Information Processing Systems, vol. 36, pp. 76 772–76 805, 2023.

[35] Y. Harun, K. Lee, J. Gallardo, G. Krishnan, and C. Kanan, “What variables affect out-of-distribution generalization in pretrained models?” Advances in Neural Information Processing Systems, vol. 37, pp. 56 479– 56 525, 2024.

[36] U. Ozbulak, et al., “Know your self-supervised learning: A survey on image-based generative and discriminative training,” Transactions on Machine Learning Research, 2024.

[37] I. Ben-Shaul, R. Shwartz-Ziv, T. Galanti, S. Dekel, and Y. LeCun, “Reverse engineering self-supervised learning,” Advances in Neural Information Processing Systems, vol. 36, pp. 58 324–58 345, 2023.

[38] Z. Song, W. Qiang, C. Zheng, F. Sun, and H. Xiong, “On the discriminability of self-supervised representation learning,” Information Sciences, p. 122556, 2025.

[39] M. Caron et al., “Emerging properties in self-supervised vision transformers,” in Proc. of the IEEE/CVF Int. Conf. on Computer Vision, 2021, pp. 9650–9660.

[40] M. Assran et al., “Self-supervised learning from images with a joint embedding predictive architecture,” in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 2023, pp. 15 619–15 629.

[41] Q. Garrido, R. Balestriero, L. Najman, and Y. Lecun, “Rankme: Assessing the downstream performance of pretrained self-supervised representations by their rank,” in Int. Conf. on Machine Learning. PMLR, 2023, pp. 10 929–10 974.

[42] D. Hu et al., “How well does self-supervised pre-training perform with streaming data?” in Int. Conf. on Learning Representations, 2022.

[43] P. S. Bhat, B. Zonooz, and E. Arani, “Task agnostic representation consolidation: a self-supervised based continual learning approach,” in Conf. on Lifelong Learning Agents. PMLR, 2022, pp. 390–405.

[44] S. Addepalli, K. Bhogale, P. Dey, and R. V. Babu, “Towards efficient and effective self-supervised learning of visual representations,” in European Conf. on Computer Vision. Springer, 2022, pp. 523–538.

[45] M. T. Koçyigit, T. M. Hospedales, and H. Bilen, “Accelerating self-˘ supervised learning via efficient training strategies,” in Proc. of the

IEEE/CVF Winter Conf. on Applications of Computer Vision, 2023, pp. 5654–5664.

[46] K. Imai, N. Hayashi, T. Hirakawa, T. Yamashita, and H. Fujiyoshi, “Faster convergence and uncorrelated gradients in self-supervised online continual learning,” in Proc. of the Asian Conf. on Computer Vision, 2024, pp. 436–453.

[47] A. Gomez-Villa, B. Twardowski, L. Yu, A. D. Bagdanov, and J. Van de Weijer, “Continually learning self-supervised representations with projected functional regularization,” in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 2022, pp. 3867–3877.

[48] A. Gomez-Villa, B. Twardowski, K. Wang, and J. Van de Weijer, “Plasticity-optimized complementary networks for unsupervised continual learning,” in Proc. of the IEEE/CVF Winter Conf. on Applications of Computer Vision, 2024, pp. 1690–1700.

[49] A. Cossu et al., “Continual pre-training mitigates forgetting in language and vision,” Neural Networks, vol. 179, p. 106492, 2024.

[50] L. Wu, Z. Wang, and J. Liu, “Adaptive self-supervised continual learning,” in ECAI 2023. IOS Press, 2023, pp. 2680–2687.

[51] S. Dziadzio et al., “How to merge your multimodal models over time?” in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 2025, pp. 20 479–20 491.

[52] E. Mushtaq et al., “Cromo-mixup: Augmenting cross-model representations for continual self-supervised learning,” in European Conf. on Computer Vision. Springer, 2024, pp. 311–328.

[53] Y. Ye et al., “Continual self-supervised learning: Towards universal multimodal medical data representation learning,” in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 2024, pp. 11 114– 11 124.

[54] L. Möllenbrok, B. Rasti, and B. Demir, “Continual self-supervised learning with masked autoencoders in remote sensing,” IEEE Geoscience and Remote Sensing Letters, 2025.

[55] S. Yan et al., “Generative negative text replay for continual visionlanguage pretraining,” in European Conf. on Computer Vision. Springer, 2022, pp. 22–38.

[56] H. Zhu, Y. Wei, X. Liang, C. Zhang, and Y. Zhao, “Ctp: Towards visionlanguage continual pretraining via compatible momentum contrast and topology preservation,” in Proc. of the IEEE/CVF Int. Conf. on Computer Vision, 2023, pp. 22 257–22 267.

[57] Z. Ni, L. Wei, S. Tang, Y. Zhuang, and Q. Tian, “Continual visionlanguage representation learning with off-diagonal information,” in Proc. of the 40th Int. Conf. on Machine Learning. PMLR, 2023, pp. 26 129– 26 149.

[58] Z. Lin, Y. Wang, and H. Lin, “Continual contrastive learning for image classification,” in 2022 IEEE International conference on multimedia and expo (ICME). IEEE, 2022, pp. 1–6.

[59] D. Marczak, S. Cygert, T. Trzcinski, and B. Twardowski, “Revisiting ´ supervision for continual representation learning,” in European Conf. on Computer Vision. Springer, 2024, pp. 181–197.

[60] S. Cha, K. Cho, and T. Moon, “Regularizing with pseudo-negatives for continual self-supervised learning,” in Int. Conf. on Machine Learning. PMLR, 2024, pp. 6048–6065.

[61] W. Liu, F. Zhu, L. Wei, and Q. Tian, “C-clip: Multimodal continual learning for vision-language model,” in Int. Conf. on Learning Representations, 2025.

[62] J. Kirkpatrick et al., “Overcoming catastrophic forgetting in neural networks,” Proc. of the national academy of sciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[63] F. Zenke, B. Poole, and S. Ganguli, “Continual learning through synaptic intelligence,” in Int. Conf. on Machine Learning. Pmlr, 2017, pp. 3987–3995.

[64] R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, and T. Tuytelaars, “Memory aware synapses: Learning what (not) to forget,” in Proc. of the European Conf on Computer Vision, 2018, pp. 139–154.

[65] M. Kilickaya and J. Vanschoren, “Are labels needed for incremental instance learning?” in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 2023, pp. 2401–2409.

[66] V. Marsocci and S. Scardapane, “Continual barlow twins: continual selfsupervised learning for remote sensing semantic segmentation,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 16, pp. 5049–5060, 2023.

[67] Y. Yao, R. Wu, Y. Zhou, and T. Zhou, “Continual retinal vision-language pre-training upon incremental imaging modalities,” in Medical Image Computing and Computer Assisted Intervention. Springer, 2025, pp. 111–121.

[68] V. Singh et al., “Beyond cosine decay: On the effectiveness of infinite learning rate schedule for continual pre-training,” Fourth Conf. on Lifelong Learning Agents, 2025.

[69] D. Rolnick, A. Ahuja, J. Schwarz, T. Lillicrap, and G. Wayne, “Experience replay for continual learning,” Advances in Neural Information Processing Systems, vol. 32, 2019.

[70] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, “mixup: Beyond empirical risk minimization,” Int. Conf. on Learning Representations, 2018.

[71] H. Cheng et al., “Contrastive continuity on augmentation stability rehearsal for continual self-supervised learning,” in Proc. of the IEEE/CVF Int. Conf. on Computer Vision, 2023, pp. 5707–5717.

[72] C. Cheng et al., “Cucl: Codebook for unsupervised continual learning,” in Proc. of the 31st ACM Int. Conf. on Multimedia, 2023, pp. 1729–1737.

[73] L. Hanmo et al., “Effective data selection and replay for unsupervised continual learning,” in IEEE 40th Int. Conf. on Data Engineering. IEEE, 2024, pp. 1449–1463.

[74] Y. Yang and M. Ren, “Memory storyboard: Leveraging temporal segmentation for streaming self-supervised learning from egocentric videos,” Fourth Conference on Lifelong Learning Agents, 2025.

[75] M. Pratama, A. Ashfahani, and E. Lughofer, “Unsupervised continual learning via self-adaptive deep clustering approach,” in Int. Workshop on Continual Semi-Supervised Learning. Springer, 2021, pp. 48–61.

[76] L. Yang, S. Lin, F. Zhang, J. Zhang, and D. Fan, “Efficient self-supervised continual learning with progressive task-correlated layer freezing,” in 26th Int. Symposium on Quality Electronic Design. IEEE, 2025, pp. 1–8.

[77] E. Hu et al., “Lora: Low-rank adaptation of large language models,” in Int. Conf. on Learning Representations, 2022.

[78] A. Radford et al., “Learning transferable visual models from natural language supervision,” CoRR, vol. 139, pp. 8748–8763, 2021. [Online]. Available: https://arxiv.org/abs/2103.00020

[79] G. Cignoni, A. Cossu, A. Gomez-Villa, J. van de Weijer, and A. Carta, “Replay-free online continual learning with self-supervised multipatches,” 2025. [Online]. Available: https://arxiv.org/abs/2502.09140

[80] L. Zhang, J. Zhao, Q. Wu, L. Pan, and H. Li, “Infoucl: Learning informative representations for unsupervised continual learning,” IEEE Transactions on Multimedia, vol. 26, pp. 10 779–10 791, 2024.

[81] S. Zhang, G. Shen, J. Huang, and Z.-H. Deng, “Self-supervised learning aided class-incremental lifelong learning,” 2020. [Online]. Available: https://arxiv.org/abs/2006.05882

[82] Q. Pham, C. Liu, and S. Hoi, “Dualnet: Continual learning, fast and slow,” Advances in Neural Information Processing Systems, vol. 34, pp. 16 131–16 144, 2021.

[83] S. Dang et al., “Self-supervised continual learning for sar-atr: A local feature adaptation framework,” IEEE Transactions on Aerospace and Electronic Systems, vol. 62, pp. 1378–1393, 2025.

[84] C. I. Tang et al., “Kaizen: Practical self-supervised continual learning with continual fine-tuning,” in Proc. of the IEEE/CVF Winter Conf. on Applications of Computer Vision, 2024, pp. 2841–2850.

[85] J.-T. Zhai, X. Liu, A. D. Bagdanov, K. Li, and M.-M. Cheng, “Masked autoencoders are efficient class incremental learners,” in Proc. of the IEEE/CVF Int. Conf. on Computer Vision, 2023, pp. 19 104–19 113.

[86] Y. He and X. Wang, “Masked autoencoders are parameter-efficient federated continual learners,” in 2024 IEEE Int. Conf. on Big Data (BigData). IEEE, 2024, pp. 3682–3691.

[87] T. Wang and P. Isola, “Understanding contrastive representation learning through alignment and uniformity on the hypersphere,” in Int. Conf. on Machine Learning. PMLR, 2020, pp. 9929–9939.