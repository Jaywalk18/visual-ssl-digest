# Symbiosis-Inspired Knowledge Distillation for Incremental Object Detection

Mingyue Zeng <sup>1</sup> De Cheng<sup>B</sup> <sup>1</sup> Zhipeng Xu <sup>1</sup> Huaijie Wang <sup>2</sup> Nannan Wang <sup>1</sup> Xinbo Gao <sup>2</sup>

## Abstract

Incremental object detection (IOD) aims to extend detectors to new categories while retaining previously acquired knowledge. Existing methods often adopt a class incremental learning perspective, separating feature spaces to sharpen decision boundaries. However, this separation-oriented paradigm may overlook object symbiosis in detection, where co-occurrence and occlusion introduce spatial and semantic dependencies that benefit from shared representations. Ignoring these dependencies distorts the shared representations, exacerbates confusion between old and new classes, and accelerates catastrophic forgetting. To address this, we propose Symbiosis-Inspired Knowledge Distillation (SIKD), which explicitly leverages object symbiosis at two complementary levels. Spatial Symbiosis Distillation (SpSD) focuses on symbiotic regions where the old model responds with high overlap to objects in the new task. It preserves generalizable old class cues, sup presses class-specific bias and redundancy, and distills the refined evidence to the new model at matched spatial locations with slot-aligned supervision. Semantic Symbiosis Distillation (SeSD) maintains class level structure by forming confidence weighted prototypes for old classes and aligning their inter class soft ranks over the old class logits, which stabilizes the semantic topology during adaptation. Extensive experiments demonstrate the effectiveness and superiority of the proposed method.

## 1. Introduction

Object detection has advanced from two-stage frameworks (Girshick, 2015; Ren et al., 2016) to efficient onestage detectors (Ge et al., 2021; Tian et al., 2019) and end-to-end transformer architectures (Zhu et al., 2020; Liu et al., 2024). Large-scale pretraining (Dai et al., 2021) and stronger benchmarks have further improved accuracy. However, in real deployments (Cheng et al., 2024; Xu et al., 2025), label spaces evolve as new categories appear. Retraining from scratch for every update is costly and often infeasible when prior data cannot be stored or accessed due to privacy or licensing. Incremental object detection (IOD) addresses this setting by learning new categories while preserving knowledge of learned ones using only the annotations available at each task.

![](images/b15108e00f56a2e41b989359a6845c56028bac702af8c1f2a76aadce9ed9adb6.jpg)  
Figure 1. Illustration of (a) object symbiosis in IOD, (b) existing methods in new task, and (c) our method in new task. In (b) and (c), arrows indicate how regional features are classified. In (c), the new-task class apple shares coarse features with the old class orange while also retaining class-specific cues.

Unlike class incremental classification (Zhou et al., 2024a; Masana et al., 2022), where each task provides complete labels for its classes, IOD trains on images that may still contain old objects while only the current categories are annotated. This mismatch pushes unlabeled old-class objects to be treated as background or drift toward new-class labels during training, which accelerates catastrophic forgetting.

To address this issue, existing IOD methods (Liu et al., 2023b; Kang et al., 2023; Mo et al., 2024; Kim et al., 2024; Wang et al., 2025c; Zhang et al., 2025a; Wang et al., 2025d) rely on the old model to mine old-class signals in the current data. As illustrated in Fig. 1(b), they retain only highconfidence old-class detections with low Intersection over Union (IoU) to new-class ground truth. These “clean” predictions are treated as the only source of old-task knowledge. This design follows a classification mindset that separates old and new features to reduce entanglement and sharpen decision boundaries (Rebuffi et al., 2017; Li et al., 2024). However, it overlooks the essential property of object symbiosis in detection. As shown in Fig. 1(a), objects naturally co-occur in shared contexts (e.g., orange and apple as fruits) and occlude one another (e.g., person riding horse), which create spatial and semantic dependencies that call for a unified feature space. Filtering supervision to only “clean” cases erodes symbiosis-bearing signals and biases the model toward the newly annotated categories, which increases old–new confusion and forgetting.

Instead, we propose Symbiosis-Inspired Knowledge Distillation (SIKD), a framework that maintains a unified feature space by leveraging object symbiosis across both spatial and semantic dimensions. As shown in Fig. 1(c), the old model processing new-task images reveals two symbiotic patterns. Unseen objects are mapped to semantically similar old classes, while partially visible old objects retain detection despite occlusion. We treat these as symbiotic regions encoding shared knowledge rather than noise. By preserving the consistent feature patterns presented in these regions, the model sustains a unified feature space across incremental tasks.

Concretely, SIKD distills symbiotic cues at two levels: instance-level spatial structure and class-level semantic topology, through Spatial Symbiosis Distillation (SpSD) and Semantic Symbiosis Distillation (SeSD). SpSD focuses on co-occurrence and occlusion regions, where it applies a Consistent Feature Enhancement (CFE) module to stabilize overlap-heavy features by reinforcing transferable patterns and suppressing spurious old-class activations. The enhanced features are then distilled to the new model via slot-aligned supervision to preserve spatial dependencies. In parallel, SeSD constructs confidence-weighted prototypes from both symbiotic and non-symbiotic regions and preserves their relative ordering in the old-class subspace via soft rank alignment, thereby maintaining the old-class semantic structure during incremental updates. Together, these two components improve knowledge retention across incremental steps. Our contributions are summarized as follows:

• We reinterpret IOD through object symbiosis (cooccurrence and occlusion), exposing the limits of classification-style feature separation in detection.

• We propose Symbiosis-Inspired Knowledge Distillation (SIKD), with Spatial Symbiosis Distillation (SpSD) to preserve spatial dependencies in symbiotic regions and Semantic Symbiosis Distillation (SeSD) to preserve old-class semantic topology.

• Extensive experiments achieve state-of-the-art performance, and ablations and visual analyses support our method.

## 2. Related Work

## 2.1. Incremental Learning

Incremental learning aims to acquire new categories over time while preserving prior knowledge, with catastrophic forgetting as the core challenge. Existing methods can be categorized into four main groups. First, output-level distillation (He et al., 2025b; Wang et al., 2025b; Rebuffi et al., 2017) transfers the old model’s logits and features to the new one to curb prediction drift. Second, parameter regularization (Wang et al., 2025a; Jung et al., 2020), exemplified by elastic weight consolidation (Kirkpatrick et al., 2017), penalizes changes to important weights so new learning does not overwrite old knowledge. Third, replay or exemplar memory (Aljundi et al., 2019a;b; Zhou et al., 2024b) stores a small set of representative samples or uses generative replay to stabilize the decision boundary. Fourth, parameter isolation and structural expansion (He et al., 2026; 2025a; Yan et al., 2021; Li et al., 2019b) allocate task-specific capacity through masks, sub-networks, or expandable branches to reduce interference between old and new knowledge.

## 2.2. Incremental Object Detection

Incremental Object Detection (IOD) adapts detectors to new categories while retaining previously learned knowledge. Unlike continual learning for classification, where each task uses a fixed label set, IOD operates on images that contain both old and new objects while only the new categories are annotated. This annotation mismatch causes unlabeled old-class instances to be suppressed as background or misassigned to new classes.

Most incremental object detection methods follow a consistent paradigm across different detector architectures. Singlestage detectors like GFL (Li & Hoiem, 2017; Li et al., 2019a; Peng et al., 2021; Feng et al., 2022; Wang et al., 2025c; Zhang et al., 2024), two-stage frameworks such as Faster R-CNN (Liu et al., 2023a; Mo et al., 2024), and transformer-based methods (Liu et al., 2023b; Kang et al., 2023; Zhang et al., 2024), all employ pseudo-labeling from previous models to identify old-class instances while filtering out regions potentially containing new categories. In transformer-based methods, CL-DETR (Liu et al., 2023b) selects reliable pseudo labels through dual filtering on IoU and confidence, and BPF (Mo et al., 2024) adopts a similar strategy with multiple teachers on Faster R-CNN (Ren et al., 2016). Subsequent work enhances this foundation through synthetic exemplar generation using Stable Diffusion (Kim et al., 2024) and improved pseudo-label filtering techniques (Wang et al., 2025c). DCA (Zhang et al., 2025a) introduces a localization-then-recognition paradigm that decouples localization from recognition to reduce forgetting, and GCD (Wang et al., 2025d) incorporates language priors through textual grounding. However, these methods share a fundamental limitation: they treat high-confidence, low-IoU detections as exclusively clean old-class evidence, thereby overlooking the inherent object symbiosis in detection scenarios. In contrast, our SIKD framework explicitly embraces object symbiosis, maintaining a unified feature space and modeling inter-object dependencies rather than suppressing them.

![](images/1e0da807e4f62e793fc6e909bbf89132c91d0bd66e535d0e30f36beb7c283a1b.jpg)  
(a)  
(b)  
Figure 2. Statistical analysis on COCO 2017 under the 70+10 setting, using old-detector predictions with $\mathrm { I o U } > 0 . 7$ to new-class ground truth. (a) Confidence distribution and mean IoU of the old detector’s old-class predictions on new-class ground-truth instances. (b) Per-class proportion of new-class ground-truth instances misclassified as old classes by the old detector.

## 3. Methodology

## 3.1. Problem Formulation

In incremental object detection, the detector is trained over T tasks. The class domain is $\textstyle C = \bigcup _ { i = 1 } ^ { T } { \mathcal { C } } ^ { i }$ with ${ \mathcal { C } } ^ { i } \cap { \mathcal { C } } ^ { j } = \emptyset$ for different tasks i and $j .$ . The dataset is $\begin{array} { r } { \mathcal { D } = \bigcup _ { i = 1 } ^ { T } \mathcal { D } ^ { i } } \end{array}$ where each $\mathcal { D } ^ { i }$ provides annotations $\mathcal { V } ^ { i }$ only for classes in ${ \mathcal { C } } ^ { i }$ . At task t, the model $\mathcal { M } ^ { t - 1 }$ is updated to $\mathcal { M } ^ { t }$ using only $\mathcal { D } ^ { t }$ and $\mathcal { V } ^ { t }$ . Images in $\mathcal { D } ^ { t }$ may still contain unlabeled instances from previously learned classes $\textstyle { \mathcal { C } } ^ { 1 : t - 1 } = \bigcup _ { i = 1 } ^ { t - 1 } { \mathcal { C } } ^ { i }$ The objective is to learn the new classes $\mathcal { C } ^ { t }$ while maintaining performance on $\mathcal { C } ^ { 1 : t }$ without accessing earlier data $\{ \mathcal { D } ^ { 1 } , \ldots , \mathcal { D } ^ { t - 1 } \}$

## 3.2. Transformer-based Detectors

Following CL-DETR (Liu et al., 2023b), we adopt Deformable DETR (Zhu et al., 2020) as the architecture. In Deformable DETR, a transformer encoder processes image features, and the decoder operates on a set of n learnable object queries $\mathcal { Q } = [ \mathbf { q } _ { 1 } , \hdots , \mathbf { q } _ { n } ] ^ { \top } \in \mathbb { R } ^ { n \times d }$ . Each query hypothesizes one object and gathers evidence from the encoded features via cross-attention. A prediction head maps the decoded queries to class logits $\mathbf { z } _ { i } \in \mathbb { R } ^ { e }$ and a classagnostic box $\mathbf { b } _ { i } \in \mathbb { R } ^ { 4 }$ , where e is the number of categories.

The decoder of DETR has L layers. At each layer, queries are refined by self-attention, multi-scale deformable crossattention, and a feed-forward block. Learned reference points are updated across layers, enabling progressive localization and classification refinement. Intermediate predictions are produced at every layer, and the final outputs after $L$ layers are the refined query embeddings $\mathcal { Q } ^ { ( L ) }$ together with logits $\{ { \bf z } _ { i } \} _ { i = 1 } ^ { n }$ and boxes $\{ { \bf b } _ { i } \} _ { i = 1 } ^ { n }$

## 3.3. Symbiosis-aware Query Partitioning

As shown in Fig. 2, incremental object detection naturally exhibits object symbiosis, where old and new categories co-occur and occlude each other in the current training data. When the old model $\mathcal { M } ^ { t - 1 }$ processes current data $\mathcal { D } ^ { t }$ , these relationships emerge as structured patterns in the query space. New objects often activate queries of semantically similar old classes, while partially visible old objects still trigger relevant query responses based on visible cues and context.

At step t, we use $\mathcal { M } ^ { t - 1 }$ predictions on $\mathcal { D } ^ { t }$ to guide the new model $\mathcal { M } ^ { t }$ . Let the old model’s queries be ${ \mathcal { Q } } ^ { t - 1 } =$ $[ \mathbf { q } _ { 1 } ^ { t - 1 } , \dots , \mathbf { q } _ { n } ^ { t - 1 } ] ^ { \top }$ with outputs $\mathbf { z } _ { i } ^ { t - 1 }$ for logits and $\mathbf { b } _ { i } ^ { t - 1 }$ for the box of each query $\mathbf q _ { i } ^ { t - 1 }$ . The ground truth for $\mathcal { D } ^ { \dot { t } }$ is

$$
\mathcal {Y} ^ {t} = \{(\mathbf {g} _ {j} ^ {t}, o _ {j} ^ {t}) \} _ {j = 1} ^ {| \mathcal {Y} ^ {t} |},\tag{1}
$$

where $\mathbf { g } _ { j } ^ { t }$ denotes a bounding box with label $o _ { j } ^ { t }$ .

Let $\sigma ( \cdot )$ denote the sigmoid. For each query we define old class confidence and the maximum overlap as

$$
\begin{array}{l} s _ {i} ^ {t - 1} = \max \sigma (\mathbf {z} _ {i} ^ {t - 1}), \\ v _ {i} ^ {t - 1} = \max _ {\forall (\mathbf {g} _ {j} ^ {t}, o _ {j} ^ {t}) \in \mathcal {Y} ^ {t}} \mathrm{IoU} (\mathbf {b} _ {i} ^ {t - 1}, \mathbf {g} _ {j} ^ {t}). \end{array}\tag{2}
$$

With confidence threshold $\gamma$ and IoU threshold τ , we parti-

![](images/4825c6c89995af426991f3d70edb39f90cf9dd0e3b7bed6a9a0fd70c40ee94b1.jpg)  
Figure 3. Overview of our proposed SIKD. (a) Training pipeline. The frozen old model $\mathcal { M } ^ { t - 1 }$ produces queries $\nsubseteq { \tau } ^ { t - 1 }$ on $\mathcal { D } ^ { t }$ . CFE refines queries of symbiotic regions under anchor-prototype guidance, yielding ${ \mathcal { Q } } ^ { \mathrm { E } }$ that reduces old-class bias and removes redundancy. SpSD distills anchor logits and boxes, which enforces confidence-weighted, layer-wise logit consistency over all queries. SeSD builds confidence-weighted, $L _ { 2 } .$ -normalized old-class prototypes from the last decoder layer of both models and aligns their classifier ranks to preserve the topology of old classes. (b) SpSD module. CFE contains multi-head self-attention (MHA) and an MLP, which is optimized with a prototype-guided cosine loss $\mathcal { L } _ { \mathrm { C F E } }$ on ${ \mathcal { Q } } ^ { \mathrm { E } }$ , and is discarded at inference.

tion the queries into index sets ${ \mathcal { A } } , S , { \mathcal { R } } \subseteq \{ 1 , \dots , n \}$

$$
\begin{array}{l} \mathcal {A} = \{i \mid s _ {i} ^ {t - 1} \geq \gamma \land v _ {i} ^ {t - 1} <   \tau \}, \\ \mathcal {S} = \{i \mid v _ {i} ^ {t - 1} \geq \tau \}, \\ \mathcal {R} = \{i \mid s _ {i} ^ {t - 1} <   \gamma \land v _ {i} ^ {t - 1} <   \tau \}. \end{array}\tag{3}
$$

This partition makes object symbiosis explicit at the query level where DETR performs inference. The set A contains high-confidence, low-overlap detections of old classes, which serve as stable anchors. The set S collects queries from overlapping scenarios caused by co-occurrence and occlusion. It includes cases where unseen new objects are misclassified as semantically similar old classes and cases where partially visible old objects remain detectable. These queries form symbiotic regions that encapsulate spatial and semantic dependencies. In contrast to prior methods that often discard high overlap predictions, we retain and exploit them as valuable supervisory signals. The set R comprises low-confidence queries that still carry weak relational cues and are utilized with reduced weighting.

## 3.4. Overall Framework

We propose SIKD for incremental object detection with DETR-style detectors. At step t, the frozen model $\mathcal { M } ^ { t - 1 }$ runs on $\mathcal { D } ^ { t }$ to produce queries $\mathcal { Q } ^ { t - 1 }$ and predictions. Following Sec. 3.3, we partition the queries into stable anchors A, symbiotic regions S and residual queries R. The core of SIKD consist of two complementary distillation pathway.

Spatial Symbiosis Distillation refines S with CFE while keeping A fixed and yields the enhanced set ${ \mathcal { Q } } ^ { \mathrm { E } }$ . SpSD then applies slot aligned, confidence weighted supervision by feeding ${ \mathcal { Q } } ^ { \mathrm { E } }$ to the frozen old decoder and $\mathcal { Q } ^ { t - 1 }$ to the new decoder. This promotes instance-level consistency in space. Semantic Symbiosis Distillation aggregates the last layer decoder outputs of both models into confidence weighted, $L _ { 2 }$ normalized prototypes for each old class. It evaluates these prototypes in the old-class logit subspace and aligns their soft ranks to preserve semantic structure during adaptation. Together, SpSD and SeSD convert overlap-driven signals into reliable supervision and maintain a unified feature space across incremental tasks.

## 3.5. Spatial Symbiosis Distillation

Spatial Symbiosis Distillation (SpSD) leverages object symbiosis in the spatial dimension by exploiting co-occurrence and occlusion patterns that manifest as distinctive signatures in the feature space. As identified in Sec. 3.3, these symbiotic patterns concentrate in the overlap-driven set S, while the set A provides semantically reliable anchors of old-class knowledge. Directly transferring knowledge from all old model queries causes two issues: It inflates old class confidence on unseen categories, and it suppresses responses to partially occluded old class objects, which leads to misclassification or background assignment. SpSD preserves anchor integrity and transforms symbiotic queries into anchor aligned representations, which maintains feature consistency across spatially related instances.

Consistent Feature Enhancement. We aim to enhance spatially consistent features in symbiotic regions while preventing anchor drift. This requires modeling contextual relationships among queries to enable ambiguous symbiotic slots to aggregate evidence from reliable anchors. We refine queries using multi-head self-attention followed by an MLP while preserving anchors:

$$
\begin{array}{c} \Delta \mathcal {Q} ^ {t - 1} = \mathrm{MLP} \big (\mathrm{MHA} (\mathcal {Q} ^ {t - 1}) \big), \\ \mathbf {q} _ {i} ^ {\mathrm{E}} = \mathbf {q} _ {i} ^ {t - 1} + \mathbb {1} _ {i \notin \mathcal {A}}   \Delta \mathbf {q} _ {i} ^ {t - 1}. \end{array}\tag{4}
$$

Here $\Delta \boldsymbol { \mathcal { Q } } ^ { t - 1 } = [ \Delta \mathbf { q } _ { 1 } ^ { t - 1 } , \dots , \Delta \mathbf { q } _ { n } ^ { t - 1 } ] ^ { \top } , \Delta \mathbf { q } _ { i } ^ { t - 1 }$ is the i-th row of $\Delta \mathcal { Q } ^ { t - 1 }$ , and $\mathbb { 1 } _ { i \notin \mathcal { A } } = 1$ if i $\notin$ A and 0 otherwise.

To guide this enhancement, we construct class-consistent prototypes from anchors. For each enhanced query $\mathbf { q } _ { i } ^ { \mathrm { E } }$ with predicted class $y _ { i } ^ { t - 1 }$ , the confidence-weighted prototype is

$$
\mathbf {p} _ {i} = \mathrm{norm} \left(\frac {\sum_ {j \in \mathcal {P} _ {\mathcal {A}} (i)} s _ {j} ^ {t - 1} \mathbf {q} _ {j} ^ {\mathrm{E}}}{\sum_ {j \in \mathcal {P} _ {\mathcal {A}} (i)} s _ {j} ^ {t - 1} + \varepsilon}\right) \in \mathbb {R} ^ {d},\tag{5}
$$

where $\mathcal { P } _ { A } ( i ) = \{ j \in \mathcal { A } \mid y _ { i } ^ { t - 1 } = y _ { i } ^ { t - 1 } , j \neq i \}$ . We then align enhanced symbiotic queries to their prototypes with a cosine objective:

$$
\mathcal {L} _ {\mathrm{CFE}} = \frac {1}{| \mathcal {S} |} \sum_ {i \in \mathcal {S}} \Big (1 - \mathrm{norm} (\mathbf {q} _ {i} ^ {\mathrm{E}}) ^ {\top} \mathbf {p} _ {i} \Big).\tag{6}
$$

Spatially Aligned Distillation. Let $\mathcal { Q } ^ { \mathrm { E } } = [ \mathbf { q } _ { 1 } ^ { \mathrm { E } } , \dots , \mathbf { q } _ { n } ^ { \mathrm { E } } ] ^ { \intercal }$ denote the enhanced query set. We feed ${ \mathcal { Q } } ^ { \mathrm { E } }$ to the decoder of the frozen old model $\mathcal { M } ^ { t - 1 }$ and $\mathcal { Q } ^ { t - 1 }$ to the decoder of the new model M<sup>t</sup>, producing logits $\mathbf { z } _ { i } ^ { \mathrm { E } } , \hat { \mathbf { z } } _ { i } ^ { t } \in \mathbb { R } ^ { m }$ and boxes $\mathbf { b } _ { i } ^ { \mathrm { E } } , \hat { \mathbf { b } } _ { i } ^ { t } \in \mathbb { R } ^ { 4 }$ , where $m = | \mathcal { C } ^ { 1 : t - 1 } |$ denotes the number of old classes. For anchors we distill semantics and geometry:

$$
\begin{array}{l} \mathcal {L} _ {A} = \frac {1}{| \mathcal {A} |} \sum_ {i \in \mathcal {A}} \Big [ \mathcal {L} _ {\mathrm{KL}} \big (\hat {\mathbf {z}} _ {i} ^ {t}, \mathbf {z} _ {i} ^ {\mathrm{E}} \big) + \lambda_ {1} \mathcal {L} _ {L 1} \big (\hat {\mathbf {b}} _ {i} ^ {t}, \mathbf {b} _ {i} ^ {\mathrm{E}} \big) \\ \qquad + \lambda_ {2} \mathcal {L} _ {\mathrm{GIoU}} \big (\hat {\mathbf {b}} _ {i} ^ {t}, \mathbf {b} _ {i} ^ {\mathrm{E}} \big) \Big ]. \end{array}\tag{7}
$$

Here ${ \mathcal { L } } _ { \mathrm { K L } }$ is the KL divergence on the old-class logit subspace, $\mathcal { L } _ { L 1 }$ is the $\ell _ { 1 }$ loss on box coordinates, and ${ \mathcal { L } } _ { \mathrm { G I o U } }$ is the generalized IoU loss.

To maintain consistency across decoder layers while handling prediction uncertainty, we employ layer-specific confidence weights at layer ℓ as:

$$
w _ {i} ^ {(\ell)} = \frac {s _ {i} ^ {t - 1 , (\ell)}}{\sum_ {j} s _ {j} ^ {t - 1 , (\ell)} + \varepsilon},\tag{8}
$$

where $s _ { i } ^ { t - 1 , ( \ell ) }$ denotes the old model confidence for query i at decoder layer $\ell .$ The layer wise distillation objective enforces progressive feature alignment:

$$
\mathcal {L} _ {\mathrm{ID}} = \sum_ {\ell = 1} ^ {L} \sum_ {i} w _ {i} ^ {(\ell)} \left\| \mathbf {z} _ {i} ^ {\mathrm{E}, (\ell)} - \hat {\mathbf {z}} _ {i} ^ {t, (\ell)} \right\| _ {2} ^ {2}.\tag{9}
$$

The overall spatial distillation objective combines both components:

$$
\mathcal {L} _ {\mathrm{SpSD}} = \mathcal {L} _ {\mathrm{A}} + \alpha \mathcal {L} _ {\mathrm{ID}},\tag{10}
$$

where $\alpha$ balances the contributions from anchor distillation and layer-wise feature alignment. With this weighting, SpSD preserves spatial coherence by aligning symbiotic regions during incremental learning and reduces representation drift by turning object co-occurrence and occlusion into useful supervision.

## 3.6. Semantic Symbiosis Distillation

While SpSD preserves instance-level relationships, its effectiveness diminishes as query assignments shift during incremental training. Semantic Symbiosis Distillation (SeSD) addresses this limitation by transitioning to class-level structure preservation, maintaining the semantic topology of old classes through prototype-based alignment that remains robust to instance-level correspondence changes.

SeSD constructs stable class representations by aggregating features into confidence-weighted prototypes. For each old class $c \in \mathcal { C } ^ { 1 : t - 1 }$ across both model states:

$$
\mathbf {p} _ {c} ^ {\pi} = \text { norm } \left(\frac {\sum_ {j \in \mathcal {P} ^ {\pi} (c)} s _ {j} ^ {\pi} \mathbf {q} _ {j} ^ {\pi , (L)}}{\sum_ {j \in \mathcal {P} ^ {\pi} (c)} s _ {j} ^ {\pi} + \varepsilon}\right), \quad \pi \in \{t - 1, t \},\tag{11}
$$

where π indexes the model state (t − 1 for old model, t for new model), ${ \mathcal { P } } ^ { \pi } ( c )$ denotes the set of queries assigned to old class c by model $\mathcal { M } ^ { \pi }$ , and $s _ { j } ^ { \pi }$ represents the corresponding confidence score derived from old-class logits.

To construct semantic relations among old classes, we project prototypes through the classifier heads $\mathcal { H } ^ { \pi } ( \cdot )$ (π ∈ $\{ t - 1 , t \} )$ and normalize the resulting score vectors:

$$
\tilde {\mathbf {s}} _ {c} ^ {\pi} = \frac {\sigma \big (\mathcal {H} ^ {\pi} (\mathbf {p} _ {c} ^ {\pi}) \big) _ {1 : m}}{\max \left[ \sigma \big (\mathcal {H} ^ {\pi} (\mathbf {p} _ {c} ^ {\pi}) \big) _ {1 : m} \right]} \in \mathbb {R} ^ {m},\tag{12}
$$

where $m = | \mathcal { C } ^ { 1 : t - 1 } |$ | denotes the total number of old classes, and the new model is restricted to these old-class dimensions to prevent interference from new categories.

We align semantic structures by the distillation objective:

$$
\mathcal {L} _ {\mathrm{SeSD}} = \frac {1}{(m) ^ {2}} \sum_ {c = 1} ^ {m} \left\| \mathrm{rank} (\tilde {\mathbf {s}} _ {c} ^ {t}) - \mathrm{rank} (\tilde {\mathbf {s}} _ {c} ^ {t - 1}) \right\| _ {1},\tag{13}
$$

where rank $\begin{array} { r } { \mathbf { \cdot } ( \tilde { \mathbf { s } } _ { c } ^ { t } ) _ { k } = \sum _ { j = 1 } ^ { m } \sigma \big ( - \big ( \tilde { s } _ { c , k } ^ { t } - \tilde { s } _ { c , j } ^ { t } \big ) \big ) } \end{array}$ computes the soft rank position of class k within the score vector, representing its relative semantic ordering. This rank-based alignment preserves topological relationships independent of absolute confidence values (Tao et al., 2020; Liu et al., 2022), focusing on the essential semantic structure rather than magnitude variations.

Table 1. Experimental results (%) on the COCO 2017 two-task settings. Best results are in bold. Methods marked with \* use exemplars.

<table><tr><td>Setting</td><td>Method</td><td>Baseline</td><td> $AP$ </td><td> $AP_{50}$ </td><td> $AP_{75}$ </td><td> $AP_S$ </td><td> $AP_M$ </td><td> $AP_L$ </td></tr><tr><td rowspan="12">70 + 10</td><td>LwF (Li &amp; Hoiem, 2017)</td><td>GFLv1</td><td>7.1</td><td>12.4</td><td>7.0</td><td>4.8</td><td>9.5</td><td>10.0</td></tr><tr><td>RILOD (Li et al., 2019a)</td><td>GFLv1</td><td>24.5</td><td>37.9</td><td>25.7</td><td>14.2</td><td>27.4</td><td>33.5</td></tr><tr><td>SID (Peng et al., 2021)</td><td>GFLv1</td><td>32.8</td><td>49.0</td><td>35.0</td><td>17.1</td><td>36.9</td><td>44.5</td></tr><tr><td>ERD (Feng et al., 2022)</td><td>GFLv1</td><td>34.9</td><td>51.9</td><td>37.4</td><td>18.7</td><td>38.8</td><td>45.5</td></tr><tr><td>TLR (Zhang et al., 2024)</td><td>GLIP</td><td>42.9</td><td>59.2</td><td>45.2</td><td>24.3</td><td>45.1</td><td>54.1</td></tr><tr><td>CL-DETR* (Liu et al., 2023b)</td><td>Deformable DETR</td><td>40.4</td><td>58.0</td><td>43.9</td><td>23.8</td><td>43.6</td><td>53.5</td></tr><tr><td>DyQ-DETR* (Zhang et al., 2025b)</td><td>Deformable DETR</td><td>42.4</td><td>60.4</td><td>46.3</td><td>24.5</td><td>45.7</td><td>57.5</td></tr><tr><td>CL-DETR (Liu et al., 2023b)</td><td>Deformable DETR</td><td>35.8</td><td>53.5</td><td>39.5</td><td>19.4</td><td>41.5</td><td>46.1</td></tr><tr><td>ACF (Kang et al., 2023)</td><td>Deformable DETR</td><td>37.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DCA (Zhang et al., 2025a)</td><td>Deformable DETR</td><td>41.3</td><td>59.2</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DyQ-DETR (Zhang et al., 2025b)</td><td>Deformable DETR</td><td>39.5</td><td>56.4</td><td>43.1</td><td>22.5</td><td>43.1</td><td>53.0</td></tr><tr><td>SIKD (Ours)</td><td>Deformable DETR</td><td>44.3</td><td>62.9</td><td>47.8</td><td>28.2</td><td>47.7</td><td>59.5</td></tr><tr><td rowspan="12">40 + 40</td><td>LwF (Li &amp; Hoiem, 2017)</td><td>GFLv1</td><td>17.2</td><td>25.4</td><td>18.6</td><td>7.9</td><td>18.4</td><td>24.3</td></tr><tr><td>RILOD (Li et al., 2019a)</td><td>GFLv1</td><td>29.9</td><td>45.0</td><td>32.0</td><td>15.8</td><td>33.0</td><td>40.5</td></tr><tr><td>SID (Peng et al., 2021)</td><td>GFLv1</td><td>34.0</td><td>51.4</td><td>36.3</td><td>18.4</td><td>38.4</td><td>44.9</td></tr><tr><td>ERD (Feng et al., 2022)</td><td>GFLv1</td><td>36.9</td><td>54.5</td><td>39.6</td><td>21.3</td><td>40.4</td><td>47.5</td></tr><tr><td>TLR (Zhang et al., 2024)</td><td>GLIP</td><td>40.4</td><td>57.4</td><td>43.9</td><td>23.3</td><td>44.7</td><td>54.5</td></tr><tr><td>CL-DETR* (Liu et al., 2023b)</td><td>Deformable DETR</td><td>42.0</td><td>60.1</td><td>45.9</td><td>24.0</td><td>45.3</td><td>55.6</td></tr><tr><td>DyQ-DETR* (Zhang et al., 2025b)</td><td>Deformable DETR</td><td>42.4</td><td>60.5</td><td>45.9</td><td>23.9</td><td>46.3</td><td>56.7</td></tr><tr><td>CL-DETR (Liu et al., 2023b)</td><td>Deformable DETR</td><td>39.2</td><td>56.1</td><td>42.6</td><td>21.0</td><td>42.8</td><td>52.6</td></tr><tr><td>ACF (Kang et al., 2023)</td><td>Deformable DETR</td><td>39.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DCA (Zhang et al., 2025a)</td><td>Deformable DETR</td><td>42.8</td><td>58.4</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DyQ-DETR (Zhang et al., 2025b)</td><td>Deformable DETR</td><td>41.4</td><td>59.7</td><td>44.9</td><td>24.1</td><td>45.2</td><td>54.3</td></tr><tr><td>SIKD (Ours)</td><td>Deformable DETR</td><td>43.3</td><td>61.7</td><td>46.7</td><td>26.4</td><td>46.6</td><td>57.0</td></tr></table>

This semantic alignment preserves the relative logit structure of old classes, anchoring $\mathcal { Q } ^ { t , ( L ) }$ to $\mathcal { Q } ^ { t - 1 , ( L ) }$ even when instance correspondences break down. By complementing spatial distillation with semantic structure preservation, SeSD stabilizes the feature space throughout tasks.

## 3.7. Training Objective

The training objective combines our symbiotic distillation terms with the standard detection loss. This joint objective maintains performance across incremental steps:

$$
\mathcal {L} _ {\text { total }} = \underbrace {\mathcal {L} _ {\text { det }} + \mathcal {L} _ {\text { SpSD }} + \beta \mathcal {L} _ {\text { SeSD }}} _ {\mathcal {L} _ {\text { model }}} + \mathcal {L} _ {\text { CFE }}.\tag{14}
$$

Here $\mathcal { L } _ { \mathrm { d e t } }$ is the standard DETR detection loss, $\mathcal { L } _ { \mathrm { S p S D } }$ maintains spatial consistency through query-level alignment, $\mathcal { L } _ { \mathrm { S e S D } }$ preserves semantic structure via prototypebased ranking, and $\mathcal { L } _ { \mathrm { C F E } }$ enhances features in symbiotic regions. The model parameters are updated using $\mathcal { L } _ { \mathrm { m o d e l } } ,$ while only the CFE module is optimized via $\mathcal { L } _ { \mathrm { C F E } }$ , with gradients isolated between these components.

## 4. Experiments

## 4.1. Experimental Settings

Datasets and Evaluation Metrics. Consistent with prior work (Liu et al., 2023b; Kim et al., 2024; Zhang et al., 2025b;a), we adopt the standard COCO 2017 (Lin et al., 2014) evaluation protocol and incremental-setting notation. We evaluate on COCO 2017, which contains 80 object categories with 118k training images and 5k validation images. Performance follows the standard COCO metrics. The primary metric, AP , is the average precision averaged over IoU thresholds from 0.50 to 0.95 in steps of 0.05. We also report $A P _ { 5 0 }$ and $A P _ { 7 5 }$ at single IoU thresholds of 0.50 and 0.75. Scale-specific scores $A P _ { S } , A P _ { M }$ , and $A P _ { L }$ evaluate small, medium, and large objects, where small means area $< 3 2 ^ { 2 }$ pixels, medium means $3 2 ^ { 2 } \leq \tt { a r e a } < 9 6 ^ { 2 }$ , and large means area $\geq 9 6 ^ { 2 }$ pixels. For incremental settings denoted $\mathbf { A } + \mathbf { B }$ , we follow prior work: the initial step contains A classes, and each subsequent step adds B new classes.

Implementation Details. We implement our method within MMDetection on Deformable DETR with a ResNet-50 backbone pre-trained on ImageNet. All experiments run on four RTX 4090 GPUs, and the basic training settings follow the official implementation (Chen et al., 2019). We use fixed hyperparameters across all settings. Following prior work (Wang et al., 2024), we set $\lambda _ { 1 } = 5 . 0$ and $\lambda _ { 2 } = 2 . 0$ We set $\alpha = 1 . 0$ and $\beta = 6 . 0$ as our choices. To ensure comparability and reproducibility, we randomize the category order using the predefined random seed released with CL-DETR (Liu et al., 2023b) and adopt the resulting order. The pseudo-label selection threshold in each incremental phase is 0.4, and the IoU threshold is 0.7.

Table 2. Experimental results $( A P / A P _ { 5 0 } ,$ %) on the COCO 2017 multi-task settings. Methods marked with \* use exemplars.

<table><tr><td rowspan="2">Method</td><td rowspan="2">(1-40)</td><td colspan="4">40+10+10+10+10</td><td colspan="2">40+20+20</td></tr><tr><td>+(40-50)</td><td>+(50-60)</td><td>+(60-70)</td><td>+(70-80)</td><td>+(40-60)</td><td>+(60-80)</td></tr><tr><td>RILOD (Li et al., 2019a)</td><td>45.7/66.3</td><td>25.4/38.9</td><td>11.2/17.3</td><td>10.5/15.6</td><td>8.4/12.5</td><td>27.8/42.8</td><td>15.8/4.0</td></tr><tr><td>SID (Peng et al., 2021)</td><td>45.7/66.3</td><td>34.6/52.1</td><td>24.1/38.0</td><td>14.6/23.0</td><td>12.6/23.3</td><td>34.0/51.8</td><td>23.8/36.5</td></tr><tr><td>ERD (Feng et al., 2022)</td><td>45.7/66.3</td><td>36.4/53.9</td><td>30.8/46.7</td><td>26.2/39.9</td><td>20.7/31.8</td><td>36.7/54.6</td><td>32.4/48.6</td></tr><tr><td>CL-DETR* (Liu et al., 2023b)</td><td>46.5/68.6</td><td>-/-</td><td>-/-</td><td>-/-</td><td>28.1/-</td><td>-/-</td><td>35.3/-</td></tr><tr><td>ACF (Kang et al., 2023)</td><td>48.0/-</td><td>39.1/-</td><td>35.4/-</td><td>32.0/-</td><td>30.3/-</td><td>39.3/-</td><td>36.6/-</td></tr><tr><td>DCA (Zhang et al., 2025a)</td><td>48.0/68.9</td><td>44.0/61.2</td><td>41.1/56.5</td><td>39.2/53.8</td><td>37.2/49.6</td><td>42.7/59.6</td><td>40.3/54.1</td></tr><tr><td>SIKD (Ours)</td><td>45.4/64.7</td><td>43.6/62.3</td><td>41.1/59.9</td><td>39.8/57.8</td><td>38.1/55.5</td><td>43.4/62.1</td><td>40.8/58.8</td></tr></table>

Table 3. Ablations on COCO 2017 (70+10) with Deformable DETR. “All categories” reports the $A P$ of the final-phase model over all 80 categories. “Old categories” reports the AP of the final-phase model on the 70 categories introduced in phase 1. $\mathrm { \displaystyle { \tilde { \Gamma } P P } } ^ { 6 6 }$ is the $A P$ difference between the phase-1 model and the final-phase model on those 70 categories (lower is better). Idx 5 corresponds to our method.

<table><tr><td rowspan="2">Idx</td><td rowspan="2">Raw KD</td><td rowspan="2">SpSD</td><td rowspan="2">SeSD</td><td colspan="3">All categories ↑</td><td colspan="3">Old categories ↑</td><td colspan="3">FPP ↓</td></tr><tr><td>AP</td><td> $AP_{50}$ </td><td> $AP_{75}$ </td><td>AP</td><td> $AP_{50}$ </td><td> $AP_{75}$ </td><td>AP</td><td> $AP_{50}$ </td><td> $AP_{75}$ </td></tr><tr><td>1</td><td></td><td></td><td></td><td>41.2</td><td>59.6</td><td>44.4</td><td>41.5</td><td>60.1</td><td>44.7</td><td>4.9</td><td>5.4</td><td>5.4</td></tr><tr><td>2</td><td>√</td><td></td><td></td><td>41.1</td><td>59.4</td><td>44.3</td><td>41.3</td><td>59.9</td><td>44.5</td><td>5.1</td><td>5.6</td><td>5.6</td></tr><tr><td>3</td><td></td><td>√</td><td></td><td>42.3</td><td>60.8</td><td>45.8</td><td>42.8</td><td>61.6</td><td>46.2</td><td>3.6</td><td>3.9</td><td>3.9</td></tr><tr><td>4</td><td></td><td></td><td>√</td><td>43.9</td><td>62.6</td><td>47.4</td><td>44.4</td><td>63.4</td><td>47.9</td><td>2.0</td><td>2.1</td><td>2.2</td></tr><tr><td>5</td><td></td><td>√</td><td>√</td><td>44.3</td><td>62.9</td><td>47.8</td><td>45.1</td><td>64.0</td><td>48.7</td><td>1.3</td><td>1.5</td><td>1.4</td></tr></table>

Two-task settings. Table 1 compares our method with prior state-of-the-art methods on the COCO 2017 two-task splits. Compared with DyQ-DETR and DCA, both implemented on Deformable DETR, our method improves AP by 1.9 and 3.0 points on the 70+10 split and by 0.9 and 0.5 points on the 40+40 split. The corresponding $A P _ { 5 0 }$ gains are 2.5 and 3.7 points on 70+10 and 1.2 and 3.3 points on 40+40. It also surpasses the GLIP-based TLR, with gains of 1.5 and 2.9 AP points and 3.7 and 4.3 $A P _ { 5 0 }$ points on 70+10 and 40+40, respectively.

## 4.2. Comparison with the State-of-the-Arts

Multi-task settings. Table 2 reports results on COCO 2017 under the multi-task settings. In the initial base training phase, our implementation achieves slightly lower performance than some prior methods. However, our method establishes new state-of-the-art results in all subsequent incremental phases. Compared with DCA, under the 40+10+10+10+10 setting we improve AP by 0.9 and $A P _ { 5 0 }$ by 5.9 points. Under the 40+20+20 setting we improve $A P$ by 0.5 and $A P _ { 5 0 }$ by 4.7. The improvements reflect superior retention of prior knowledge coupled with efficient acquisition of new concepts.

## 4.3. Results and Analysis

Results on DIOR dataset. We further evaluate our method on the DIOR dataset (Li et al., 2020). DIOR is a large-scale optical remote sensing benchmark with 20 categories and strong variation in scale, viewpoint, object density, and background, where co-occurrence and occlusion are common. We follow three two-task settings 10+10, 15+5, and 19+1 and report $A P _ { 5 0 }$ as summarized in Table 4. Compared with the replay-based method CL-DETR\*, SIKD improves the All score by 6.2 points in the 10+10 setting, by 7.3 points in the 15+5 setting, and by 9.0 points in the 19+1 setting.

Ablation study of SIKD. We evaluate the contribution of each component under the 70+10 setting, with results summarized in Table 3. Idx 1 is the baseline, which adopts a standard pseudo-labeling strategy without any of our proposed modules. Idx 2 (Raw KD) incorporates high-IoU queries directly into the distillation process. Idx 3 employs only SpSD, and Idx 4 uses only SeSD. Idx 5 combines both SpSD and SeSD, representing our full SIKD method.

Compared to the baseline in Idx 1, directly integrating highoverlap queries in Index 2 causes a drop of 0.1 AP , suggesting that such queries introduce noise and bias when used naively. Using SpSD alone (Idx 3) improves $A P$ by 1.1 points, while SeSD alone (Idx 4) brings a more substantial gain of 2.7 points, highlighting the individual efficacy of each distillation pathway. The full SIKD model (Idx 5) achieves the best performance of 44.3 AP , demonstrating that combining spatial and semantic distillation yields complementary benefits and the highest overall accuracy.

![](images/f39015661a937f6a10128448f9ccb2bc33b8c5107ed9cad146e1ef7d4270f92a.jpg)

Table 4. Experimental results (%) on the DIOR dataset under the two-task settings. $A P _ { 5 0 }$ is reported for Old, New, All and taskaveraged (Avg). Best results are in bold. Methods marked with \* use exemplars.

<table><tr><td>Setting</td><td>Methods</td><td>Old</td><td>New</td><td>All</td><td>Avg</td></tr><tr><td rowspan="3">10+10</td><td>CL-DETR</td><td>42.2</td><td>63.9</td><td>53.1</td><td>53.1</td></tr><tr><td>CL-DETR*</td><td>64.4</td><td>61.5</td><td>63.0</td><td>63.0</td></tr><tr><td>SIKD (Ours)</td><td>72.0</td><td>66.4</td><td>69.2</td><td>69.2</td></tr><tr><td rowspan="3">15+5</td><td>CL-DETR</td><td>43.0</td><td>63.8</td><td>48.2</td><td>53.4</td></tr><tr><td>CL-DETR*</td><td>64.3</td><td>60.4</td><td>63.4</td><td>62.4</td></tr><tr><td>SIKD (Ours)</td><td>71.8</td><td>67.5</td><td>70.7</td><td>69.6</td></tr><tr><td rowspan="3">19+1</td><td>CL-DETR</td><td>47.6</td><td>44.0</td><td>47.5</td><td>45.8</td></tr><tr><td>CL-DETR*</td><td>57.3</td><td>53.0</td><td>57.1</td><td>55.2</td></tr><tr><td>SIKD (Ours)</td><td>65.3</td><td>79.7</td><td>66.1</td><td>72.5</td></tr></table>

![](images/31a69850c38f0bda3819067c54d39497187efdfe1646ab65ccf066debb7d0264.jpg)

![](images/c6dee2d734c2685405eb84156d02e89589496266cb26f3e3b19dc44200908b57.jpg)  
Figure 4. Ablations of SpSD and SeSD on COCO 2017 (70+10). In (a), the inner bars denote the baseline, with the full bars showing absolute values and labels indicating improvements over baseline.

Analysis of SpSD and SeSD. In Fig. 4(a), Anchor-Only Distillation (AOD) using Eq. 7 raises AP by 0.3, indicating a modest gain. SpSD without CFE improves AP by 0.8, showing that overlap-driven evidence is useful. Adding CFE to SpSD lifts the gain to 1.1, which further stabilizes instance-level consistency. SeSD without CFE improves $A P$ by 2.5, highlighting the value of preserving class-level topology. SeSD with CFE achieves 2.7, confirming that feature enhancement also benefits the class-level objective. In Fig. 4(b) and (c), CFE contributes 0.4 AP to SpSD on the old task and 0.5 AP to SeSD on the new task.

Analysis of the balance weight. As shown in Table 5, we ablate the balancing coefficients $\beta$ and α on COCO 2017 (70+10). For $\beta ,$ increasing the weight from 4 to 8 raises All

Table 5. Ablation of the balance weights on COCO 2017 (70+10). $A P$ is reported for Old, New, All and task-averaged (Avg).

<table><tr><td>Setting</td><td>Old</td><td>New</td><td>All</td><td>Avg</td></tr><tr><td> $\beta = 4$ </td><td>44.7</td><td>39.1</td><td>44.0</td><td>41.9</td></tr><tr><td> $\beta = 6$ </td><td>45.1</td><td>38.8</td><td>44.3</td><td>42.0</td></tr><tr><td> $\beta = 8$ </td><td>45.2</td><td>38.5</td><td>44.4</td><td>41.9</td></tr><tr><td> $\alpha = 0.1$ </td><td>42.1</td><td>39.9</td><td>41.9</td><td>41.0</td></tr><tr><td> $\alpha = 1$ </td><td>42.8</td><td>38.8</td><td>42.3</td><td>40.8</td></tr><tr><td> $\alpha = 10$ </td><td>43.2</td><td>34.9</td><td>42.2</td><td>39.1</td></tr></table>

![](images/b8ce8ce0a56afe4890a408ba2199ebc75eabcfd50b93b8f85ebfd7ffe4fe39c4.jpg)

![](images/fa1dd5f9a5686739f393c635499bfb0e751eb81150a6cf914cfaa55a10b847fe.jpg)  
Figure 5. Comparison of confusion matrices for old and new Classes between the baseline and our method on COCO (70+10).

AP by 0.4 and Old AP by 0.5, while reducing New AP by 0.6. Setting $\beta = 6$ yields the best task-averaged AP (42.0) and provides a favorable trade-off between retention and plasticity, so we adopt $\beta = 6$ in the main experiments. For $\alpha ,$ larger values bias training toward preserving old knowledge. Old $A P$ increases by 1.1 as α grows from 0.1 to 10, whereas New $A P$ drops by 5.0 and Avg AP decreases by 1.9. Setting $\alpha = 1$ achieves the highest All AP (42.3) with a reasonable balance between old and new, and is therefore our default.

Visualizations. As shown in Fig. 5, we compare the confusion matrices between old and new categories for SIKD and the baseline on COCO 2017 (70+10). Relative to the baseline, SIKD reduces old-to-new confusion while not increasing new-to-old errors. These results support our design of maintaining a unified feature space across old and new classes, thereby mitigating the tendency of old-class knowledge to overfit toward new classes. The appendix includes additional experiments and visual analyses. Table 9 reports efficiency results, Table 6 provides further analysis of SeSD, Table 7 ablates the hyperparameters γ and τ , and Table 8 reports results under the multi-task setting. The appendix also contains Fig. 7 and additional qualitative visualizations.

## 5. Conclusion

We revisit IOD through object symbiosis and show that a unified feature space reduces confusion between old and new classes and curbs forgetting. SIKD models spatial symbiosis and semantic symbiosis. SpSD captures spatial symbiosis by refining symbiotic queries under anchor guidance and enforcing slot-aligned instance consistency. SeSD preserves semantic symbiosis by building confidence-weighted prototypes and aligning their ranks within the old-class subspace. Together, they convert overlap responses into reliable supervision and stabilize both spatial and semantic representations. Extensive experiments show consistent gains over state-of-the-art methods. In future work, we will explore using LLMs (Cheng et al., 2026; Xu et al., 2026) to inject contextual priors to further strengthen symbiosis-aware distillation.

## Acknowledgments

This work was supported in part by the National Key R&D Program of China under Grant No.2023YFA1008600, in part by the National Natural Science Foundation of China under Grants 62576262, U22A2096, in part by the Key Research and Development Program of Shaanxi Province under grant 2024SF-YBXM-647, in part by the Fundamental Research Funds for the Central Universities under Grant QTZX25083, QTZX23042.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## References

Aljundi, R., Belilovsky, E., Tuytelaars, T., Charlin, L., Caccia, M., Lin, M., and Page-Caccia, L. Online continual learning with maximal interfered retrieval. Advances in neural information processing systems, 32, 2019a.

Aljundi, R., Lin, M., Goujaud, B., and Bengio, Y. Gradient based sample selection for online continual learning. Advances in neural information processing systems, 32, 2019b.

Chen, K., Wang, J., Pang, J., Cao, Y., Xiong, Y., Li, X., Sun, S., Feng, W., Liu, Z., Xu, J., et al. Mmdetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155, 2019.

Cheng, D., Xu, Z., Jiang, X., Wang, N., Li, D., and Gao, X. Disentangled prompt representation for domain generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23595– 23604, 2024.

Cheng, D., Xu, Z., Jiang, X., Li, D., Wang, N., and Gao, X. Prompt disentanglement via language guidance and rep-

resentation alignment for domain generalization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

Dai, Z., Cai, B., Lin, Y., and Chen, J. Up-detr: Unsupervised pre-training for object detection with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1601–1610, 2021.

Feng, T., Wang, M., and Yuan, H. Overcoming catastrophic forgetting in incremental object detection via elastic response distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9427–9436, 2022.

Ge, Z., Liu, S., Wang, F., Li, Z., and Sun, J. Yolox: Exceeding yolo series in 2021. arXiv preprint arXiv:2107.08430, 2021.

Girshick, R. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision, pp. 1440–1448, 2015.

He, L., Cheng, D., Ma, Z., Wang, H., Zhang, D., Wang, N., and Gao, X. Ckaa: Cross-subspace knowledge alignment and aggregation for robust continual learning. arXiv preprint arXiv:2507.09471, 2025a.

He, L., Cheng, D., Wang, H., and Wang, N. Harnessing textual semantic priors for knowledge transfer and refinement in clip-driven continual learning. arXiv preprint arXiv:2508.01579, 2025b.

He, L., Cheng, D., Wang, H., Zhu, X., Yang, X., Wang, N., and Gao, X. Task-driven subspace decomposition for knowledge sharing and isolation in lora-based continual learning. In Forty-third International Conference on Machine Learning, 2026.

Jung, S., Ahn, H., Cha, S., and Moon, T. Continual learning with node-importance based adaptive group sparse regularization. Advances in neural information processing systems, 33:3647–3658, 2020.

Kang, M., Zhang, J., Zhang, J., Wang, X., Chen, Y., Ma, Z., and Huang, X. Alleviating catastrophic forgetting of incremental object detection via within-class and between-class knowledge distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 18894–18904, 2023.

Kim, J., Cho, H., Kim, J., Tiruneh, Y. Y., and Baek, S. Sddgr: Stable diffusion-based deep generative replay for class incremental object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28772–28781, 2024.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.

Li, D., Tasci, S., Ghosh, S., Zhu, J., Zhang, J., and Heck, L. Rilod: Near real-time incremental learning for object detection at the edge. In Proceedings of the 4th ACM/IEEE Symposium on Edge Computing, pp. 113–126, 2019a.

Li, K., Wan, G., Cheng, G., Meng, L., and Han, J. Object detection in optical remote sensing images: A survey and a new benchmark. ISPRS journal of photogrammetry and remote sensing, 159:296–307, 2020.

Li, Q., Peng, Y., and Zhou, J. Fcs: Feature calibration and separation for non-exemplar class incremental learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 28495–28504, 2024.

Li, X., Zhou, Y., Wu, T., Socher, R., and Xiong, C. Learn to grow: A continual structure learning framework for overcoming catastrophic forgetting. In International conference on machine learning, pp. 3925–3934. PMLR, 2019b.

Li, Z. and Hoiem, D. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947, 2017.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L. Microsoft coco:´ Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pp. 740– 755. Springer, 2014.

Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pp. 38–55. Springer, 2024.

Liu, Y., Hong, X., Tao, X., Dong, S., Shi, J., and Gong, Y. Model behavior preserving for class-incremental learning. IEEE Transactions on Neural Networks and Learning Systems, 34(10):7529–7540, 2022.

Liu, Y., Cong, Y., Goswami, D., Liu, X., and Van De Weijer, J. Augmented box replay: Overcoming foreground shift for incremental object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 11367–11377, 2023a.

Liu, Y., Schiele, B., Vedaldi, A., and Rupprecht, C. Continual detection transformer for incremental object detection.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23799–23808, 2023b.

Masana, M., Liu, X., Twardowski, B., Menta, M., Bagdanov, A. D., and Van De Weijer, J. Class-incremental learning: survey and performance evaluation on image classification. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5):5513–5533, 2022.

Mo, Q., Gao, Y., Fu, S., Yan, J., Wu, A., and Zheng, W.-S. Bridge past and future: Overcoming information asymmetry in incremental object detection. In European Conference on Computer Vision, pp. 463–480. Springer, 2024.

Peng, C., Zhao, K., Maksoud, S., Li, M., and Lovell, B. C. Sid: Incremental learning for anchor-free object detection via selective and inter-related distillation. Computer vision and image understanding, 210:103229, 2021.

Rebuffi, S.-A., Kolesnikov, A., Sperl, G., and Lampert, C. H. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 2001–2010, 2017.

Ren, S., He, K., Girshick, R., and Sun, J. Faster r-cnn: Towards real-time object detection with region proposal networks. IEEE transactions on pattern analysis and machine intelligence, 39(6):1137–1149, 2016.

Tao, X., Chang, X., Hong, X., Wei, X., and Gong, Y. Topology-preserving class-incremental learning. In European conference on computer vision, pp. 254–270. Springer, 2020.

Tian, Z., Shen, C., Chen, H., and He, T. Fcos: Fully convolutional one-stage object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9627–9636, 2019.

Wang, H., Cheng, D., He, L., Li, Y., Li, J., Wang, N., and Gao, X. Ekpc: Elastic knowledge preservation and compensation for class-incremental learning. arXiv preprint arXiv:2506.12351, 2025a.

Wang, H., Cheng, D., Li, G., Xu, Z., He, L., Li, J., Wang, N., and Gao, X. Stpr: Spatiotemporal preservation and routing for exemplar-free video class-incremental learning. arXiv preprint arXiv:2505.13997, 2025b.

Wang, Q., Chen, Z., Yang, C., Liu, J., Li, Z., and Zhao, F. Psedet: Revisiting the power of pseudo label in incremental object detection. In The Thirteenth International Conference on Learning Representations, 2025c.

Wang, X., Wang, Z., and Lin, Z. Gcd: Advancing visionlanguage models for incremental object detection via

global alignment and correspondence distillation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 8015–8023, 2025d.

Wang, Y., Li, X., Weng, S., Zhang, G., Yue, H., Feng, H., Han, J., and Ding, E. Kd-detr: Knowledge distillation for detection transformer with consistent distillation points sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16016– 16025, 2024.

Xu, Z., Cheng, D., Jiang, X., Wang, N., Li, D., and Gao, X. Adversarial domain prompt tuning and generation for single domain generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18584–18595, 2025.

Xu, Z., Wang, Z., Jiang, X., Li, D., Cheng, D., and Wang, N. Reasoning-driven multimodal LLM for domain generalization. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=psJiUopUt7.

Yan, S., Xie, J., and He, X. Der: Dynamically expandable representation for class incremental learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3014–3023, 2021.

Zhang, A., Yang, D., Liu, C., Hong, X., Shang, M., and Zhou, Y. Dca: Dividing and conquering amnesia in incremental object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 9851– 9859, 2025a.

Zhang, H., Gao, B.-B., Zeng, Y., Tian, X., Tan, X., Zhang, Z., Qu, Y., Liu, J., and Xie, Y. Learning task-aware language-image representation for class-incremental object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 7096–7104, 2024.

Zhang, J., Li, W., Cheng, S., Li, Y., and Wang, S. Dynamic object queries for transformer-based incremental object detection. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2025b.

Zhou, D.-W., Wang, Q.-W., Qi, Z.-H., Ye, H.-J., Zhan, D.- C., and Liu, Z. Class-incremental learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024a.

Zhou, Y., Yao, J., Hong, F., Zhang, Y., and Wang, Y. Balanced destruction-reconstruction dynamics for memoryreplay class incremental learning. IEEE Transactions on Image Processing, 2024b.

Zhu, X., Su, W., Lu, L., Li, B., Wang, X., and Dai, J. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020.

## A. Training pipeline for SIKD

Algorithm 1 illustrates the training details of our method.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 SIKD training in new task t

Require: Frozen old model  $M^{t-1}$ , new model  $M^{t}$ , current dataset  $D^{t}$ , thresholds  $\gamma, \tau$ , weights  $\alpha, \beta$ .

1: for each mini-batch  $(X, \mathcal{Y}^{t})$  in  $D^{t}$  do

2: Old-model forward and statistics

3:  $Q^{t-1}, \{z_{i}^{t-1}\}_{i=1}^{n}$ , and  $\{b_{i}^{t-1}\}_{i=1}^{n} \leftarrow \mathcal{M}^{t-1}(X)$ 

4: for i = 1 to n do

5:  $s_{i}^{t-1} \leftarrow \max \sigma(z_{i}^{t-1})$ , and  $v_{i}^{t-1} \leftarrow \max_{j} \text{IoU}(b_{i}^{t-1}, g_{j}^{t})$ 

6: end for

7:  $A \leftarrow \{i \mid s_{i}^{t-1} \geq \gamma \wedge v_{i}^{t-1} &lt; \tau\}$ ,  $S \leftarrow \{i \mid v_{i}^{t-1} \geq \tau\}$ , and  $R \leftarrow \{i \mid s_{i}^{t-1} &lt; \gamma \wedge v_{i}^{t-1} &lt; \tau\}$ 

8: Consistent Feature Enhancement (CFE)

9:  $\Delta Q^{t-1} \leftarrow \text{MLP}(MHA(Q^{t-1}))$ 

10: for i = 1 to n do

11: if  $i \in A$  then

12:  $q_{i}^{E} \leftarrow q_{i}^{t-1}$ 

13: else

14:  $q_{i}^{E} \leftarrow q_{i}^{t-1} + \Delta q_{i}^{t-1}$ 

15: end if

16: end for

17:  $Q^{E} \leftarrow [q_{1}^{E}, \ldots, q_{n}^{E}]^{\top}$ 

18: Build anchor-based prototypes  $\{p_{i}\}$  and compute  $L_{CFE}$  as in Eq. (6)

19: Spatial Symbiosis Distillation (SpSD)

20:  $\{z_{i}^{E}, b_{i}^{E}\}_{i=1}^{n} \leftarrow \text{decoder of } M^{t-1}(Q^{E})$ , and  $\{\hat{z}_{i}^{t}, \hat{b}_{i}^{t}\}_{i=1}^{n} \leftarrow \text{decoder of } M^{t}(Q^{t-1})$ 

21: Compute anchor loss  $L_{A}$  using Eq. (7)

22: Compute layer-wise loss  $L_{ID}$  with  $w_{i}^{(\ell)}$  from Eq. (9)

23:  $L_{SpSD} \leftarrow L_{A} + \alpha L_{ID}$ 

24: Semantic Symbiosis Distillation (SeSD)

25: Build confidence-weighted prototypes  $\{p_{c}^{t-1}, p_{c}^{t}\}$  for each old class c

26: Obtain  $\tilde{s}_{c}^{t-1}, \tilde{s}_{c}^{t}$  and compute  $L_{SeSD}$  via Eq. (13)

27: Total losses

28:  $\hat{Y}^{t} \leftarrow M^{t}(X)$ 

29:  $L_{det} \leftarrow L_{det}(\hat{Y}^{t}, Y^{t})$ , and  $L_{model} \leftarrow L_{det} + L_{SpSD} + \beta L_{SeSD}$ 

30: Update the new detector with  $L_{model}$  and the CFE module with  $L_{CFE}$ , while blocking gradients from each loss to the other branch.

31: end for

32: Output: updated new model  $M^{t}$
</div>

## B. Effect of rank alignment in SeSD

Table 6 compares rank alignment with direct score distillation on COCO 2017 (70+10). Using rank alignment improves all metrics. Old rises from 42.4 to 44.4 (+2.0), New from 39.6 to 40.4 (+0.8), All from 42.1 to 43.9 (+1.8), and Avg from 41.0 to 42.4 (+1.4). These gains indicate that preserving the relative ordering of class scores is more robust than matching raw scores. Rank alignment reduces sensitivity to calibration and scale differences between models, which helps maintain the semantic topology of old classes while adapting to new ones.

## C. Sensitivity to γ and τ

We analyze the sensitivity of SIKD to the two thresholds $\gamma$ and τ used in our symbiosis-aware query partitioning. As shown in Table 7, the performance is stable across a reasonable range of values. For $\gamma ,$ setting it to 0.4 yields the best overall performance, while values 0.3 and 0.5 lead to only minor changes. For $\tau ,$ , the default choice $\tau = 0 . 7$ achieves the highest

Table 6. SeSD ablation on COCO 2017 (70+10) comparing rank alignment with direct score distillation. $A P \left( \% \right)$ is reported for Old, New, All, and task averaged (Avg).

<table><tr><td>Methods</td><td>Old</td><td>New</td><td>All</td><td>Avg</td></tr><tr><td>SeSD w/o rank</td><td>42.4</td><td>39.6</td><td>42.1</td><td>41.0</td></tr><tr><td>SeSD</td><td>44.4</td><td>40.4</td><td>43.9</td><td>42.4</td></tr></table>

Table 7. Sensitivity analysis of thresholds γ and τ on DIOR under the 15+5 incremental setting. $A P _ { 5 0 } ( \% )$ is reported for old classes (Old), new classes (New), and all classes (All).

<table><tr><td>Setting</td><td>Old</td><td>New</td><td>All</td></tr><tr><td> $\gamma = 0.3$ </td><td>70.9</td><td>66.9</td><td>69.9</td></tr><tr><td> $\gamma = 0.4$ </td><td>71.8</td><td>67.5</td><td>70.7</td></tr><tr><td> $\gamma = 0.5$ </td><td>71.4</td><td>68.2</td><td>70.6</td></tr><tr><td> $\tau = 0.5$ </td><td>71.2</td><td>67.9</td><td>70.4</td></tr><tr><td> $\tau = 0.7$ </td><td>71.8</td><td>67.5</td><td>70.7</td></tr><tr><td> $\tau = 0.9$ </td><td>71.1</td><td>68.2</td><td>70.3</td></tr></table>

A $P _ { 5 0 }$ (70.7), and both lower (0.5) and higher (0.9) thresholds result in small degradations. Overall, these results indicate that our method is not overly sensitive to threshold selection, and we use $\gamma = 0 . 4$ and $\tau = 0 . 7$ as default in all experiments.

## D. Multi-task evaluation on DIOR

We further evaluate SIKD under multi-task class-incremental settings on DIOR. Specifically, we consider three settings: 10+5+5, 5+5+5+5, and 10+2+2+2+2+2, and report the final-phase $A P _ { 5 0 }$ in Table 8. Across all settings, SIKD consistently outperforms CL-DETR and CL-DETR\*. Notably, the performance gap becomes larger as the number of incremental phases increases, suggesting that SIKD better mitigates forgetting and maintains effective knowledge transfer over longer learning sequences.

## E. More visualization

## E.1. t-SNE visualization

To qualitatively assess representation drift in class-incremental detection, we visualize the learned object features using t-SNE on the COCO 2017 validation set under the 70+10 setting in Figure 6. The baseline shows fragmented and less cohesive clusters, indicating unstable representations and aggravated old–new confusion after incremental updates. In contrast, our SIKD yields a more compact and well-structured embedding space that more closely resembles the joint-training reference. This suggests that the proposed symbiosis-inspired distillation better preserves old-class feature geometry while effectively incorporating new classes. Overall, this qualitative observation aligns with our quantitative results, supporting that SIKD alleviates catastrophic forgetting and improves feature consistency across incremental tasks.

## E.2. CFE module visualization

In Fig. 7, we visualize the per-detection transition in predicted class and confidence before and after applying CFE. Arrows indicate the mapping from the pre-CFE state to the post-CFE state.

## F. Efficiency of our method

Table 9 reports inference and per-image training GFLOPs and parameter counts on COCO 2017 (70+10) with input size $( 1 0 6 4 \times 8 0 0 )$ . All methods keep inference at 125 GFLOPs since no test-time modules are added. The baseline uses one forward of the frozen old model together with one forward–backward of the new detector, totaling 500 GFLOPs for training. Raw KD increases the training cost to 644 GFLOPs by adding an extra forward through the new decoder and the associated

Table 8. Experimental results $( A P _ { 5 0 } , \% )$ on the DIOR dataset under the multi-task settings. Best results are in bold.

<table><tr><td>Methods</td><td>10+5+5</td><td>5+5+5+5</td><td>10+2+2+2+2+2</td></tr><tr><td>CL-DETR</td><td>42.9</td><td>39.4</td><td>35.2</td></tr><tr><td>CL-DETR*</td><td>47.1</td><td>37.1</td><td>34.1</td></tr><tr><td>SIKD (Ours)</td><td>66.5</td><td>61.4</td><td>61.4</td></tr></table>

Table 9. Efficiency on COCO 2017 (70+10). GFLOPs<sup>1</sup> denotes inference FLOPs; GFLOPs<sup>2</sup> denotes training FLOPs per image.

<table><tr><td>Method</td><td>GFLOPs $^{1}$ </td><td>GFLOPs $^{2}$ </td><td>#Params (M)</td></tr><tr><td>baseline</td><td>125</td><td>500</td><td>82.41</td></tr><tr><td>Raw KD</td><td>125</td><td>644</td><td>82.41</td></tr><tr><td>CFE</td><td>-</td><td>2.169</td><td>1.05</td></tr><tr><td>SIKD</td><td>125</td><td>694.169</td><td>83.46</td></tr></table>

KD losses. SIKD retains these costs and further adds the training-only CFE block with both forward and backward, plus one additional forward of the frozen old decoder using enhanced queries for slot-aligned supervision, reaching 694.169 GFLOPs in training. CFE has no test-time cost, so inference remains unchanged.

## G. Analysis of detection predictions

As shown in Fig. 8, we qualitatively compare detection predictions of the baseline and our SIKD on COCO 2017 under the 70+10 setting. In Fig. 8(a), SIKD prevents the old class oven from being confused with the visually similar new class microwave. In Fig. 8(b), SIKD avoids misclassifying the old class carrot as the new class apple and successfully detects hard instances that are missed by the baseline. In Fig. 8(c), SIKD reduces forgetting of the old class orange, whereas the baseline misses many orange objects. In Fig. 8(d) and Fig. 8(e), on images containing only old classes such as orange or boat, SIKD suppresses spurious predictions of new classes while reducing missed detections of old-class objects. These results demonstrate that SIKD simultaneously alleviates old–new confusion and preserves detection quality on old classes.

![](images/c03361678c511439b34507f463a9182326c3a29f31caf7b7b177619a5155d23e.jpg)  
(a) Baseline

![](images/e7042f12f8ea5cee1bcad006ec0d5d6636d1e3252bc6fec286ae2c15993dc726.jpg)  
(b) Joint Training

![](images/4f6b9b5e5195f8666e0befa3e334450dd5b8737ebfa4b35f78f0bb1f6c534454.jpg)  
(c) Our method  
Figure 6. t-SNE visualization of object features on the COCO 2017 validation set. (a) Baseline after the final incremental task on COCO 2017 (70+10) setting, (b) Joint Training as an upper-bound model trained once on the union of old and new classes with full annotations (non-incremental), and (c) our method (SIKD) after the final incremental task on COCO 2017 (70+10) setting.

![](images/9762e360e5c5c86ee51466a44f986c553f87d344208243d8c03e408bacd1fd9f.jpg)  
Figure 7. Visualization of the CFE module. For each detection, class id and confidence are shown as $\mathrm { \ddot { \hbar } o l d \mathrm { \ - } \vec { \Omega } \mathrm { n e w } ^ { \mathrm { \prime } \mathrm { * } } }$ with the left value before CFE and the right value after CFE.

Unannotated images

![](images/11ace00f6b6a943128944400286794178181b8fa9db638fb3fd51eb7882ffe8a.jpg)

![](images/a4f5454281c575cbaf537630045a4d5d2596c860098ee3c9c30e7f07c00b92ce.jpg)

![](images/c2aaca1f56769932d922de6b6afe89a6a93b9867a97f4113cecd53feb748a572.jpg)

![](images/f912b813cb6d7198bef6f8c0af4499a1df130a7a7c9b583c089b1983f7bdd79a.jpg)

![](images/9c3e79bb4504c31af75b5e067043f502219751a332936795042f8f57e6baaf89.jpg)

Ground truth

![](images/51a6f3fd9d120dd5c43df18afe0378447602f72bd3502c1d12383553b8f0a8b2.jpg)

![](images/ccc93af71009eb682f38d1dfd4185aea6c46c46cbf94b3e96b47f82252f7a4c8.jpg)

![](images/004b602c22d92da65636d062eb06851cee0557e3eb8241352d4f467eabe4880f.jpg)

![](images/bcd2a6ca6cf78a8fc242791a814a48a1d4d96168f5cc9a705a2176dc938f8511.jpg)

![](images/7896f1e3a6cbc0e6df26e6977094c878d9eda6f439e1489acb48c9677a36e6f4.jpg)

Baseline

![](images/ea5bf826fa2af199e3f29c7f3dde74e1eb525575f799a62ccd7c740a4e0725c3.jpg)

![](images/42fbd4203f00e0f068eaf8527051695c367a9d0ba8a1c62cb731b89104968f14.jpg)

![](images/4f82adc4d120919b1ef3a5bd23b019fb80dcfc2cebcabdd8e54a65d81b186df7.jpg)

![](images/686add7379dd64274a03ec4e2a8221ba8996cc0140ec0834f15d46f253c87088.jpg)

![](images/44e83d5b52160b4f5ddf4b0d30938fb283266cdbb5dad92e2c109d2742cf738f.jpg)

Our method

![](images/034cb462d6c262a453373be9ff0c43380b6d85880398814d5655b288180636dd.jpg)  
(a)

![](images/e22e25603dbd5b06729b10cb51b728781d80eb798f5ab000d3b3287e20ec1824.jpg)  
(b)

![](images/2eaa52ca455bdd18ea87137da6d244b3aeed26c043a3e87b98a13c7b983051c9.jpg)  
(c)

![](images/54e21358e1b56b32e4d1704528eec8369c20a3b5f1913cf1b21f4be388b31905.jpg)  
(d)

![](images/3b79521160ce6f83f95dcb7b990ca03c04f63c5aefeca027e95ea4d65b949a85.jpg)  
(e)  
Figure 8. Qualitative visualization of detection predictions from the baseline and our method (SIKD) on COCO 2017 (70+10) setting. In (a), oven is an old class and microwave is a new class. In (b), carrot is an old class and apple is a new class. In (c), both orange and person are old classes. In (d), orange is an old class, whereas banana and apple are new classes. In (e), boat is an old class.