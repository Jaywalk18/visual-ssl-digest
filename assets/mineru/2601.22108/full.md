# Learning What to Predict: Downstream-Guided Task Design for Continued Pretraining

Shuqi Ke

Department of ECE

Carnegie Mellon University

Pittsburgh, PA 15213, USA

shuqik@andrew.cmu.edu

Giulia Fanti

Department of ECE

Carnegie Mellon University

Pittsburgh, PA 15213, USA

gfanti@andrew.cmu.edu

## Abstract

Continued pretraining is typically optimized with a fixed self-supervised task, yet selected and justified by downstream performance. This creates a coarse feedback loop: practitioners evaluate checkpoints, revise data mixtures or objectives, and restart pretraining runs, while individual pretraining updates remain blind to whether they help the capabilities of interest. We ask whether a small set of verifiable downstream examples can provide step-level feedback during continued pretraining without direct learner supervision. We introduce V-pretraining, which separates a learner trained only by a self-supervised loss from a lightweight task designer that constructs targets or views for unlabeled batches. Given the current learner and an unlabeled batch, V-pretraining estimates the downstream value of a candidate target or view construction. It does this by predicting the first-order drop in downstream loss caused by a self-supervised update. The designer is trained to increase this value; the learner then applies the resulting self-supervised update, with targets or views detached, so downstream labels never directly update learner parameters. We instantiate V-pretraining to learn adaptive top-K soft targets for language modeling and learned views or masks for self-supervised vision. We observe that V-pretraining can significantly improve target capabilities without harming generalization on vision and language modalities. For instance, under wallclock-matched continued pretraining, V-pretraining improves GSM8K Pass@1 for Qwen models using 1,024 GSM8K examples only as feedback, including a +7.4 point single-run gain for Qwen2.5-0.5B. In vision, V-pretraining improves DINOv3 transfer to ADE20K semantic segmentation and NYUv2 depth estimation while preserving ImageNet linear accuracy, indicating that feedback-guided task construction improves target downstream capabilities without collapsing general-purpose representations.

## 1 Introduction

Continued pretraining is a standard way to adapt foundation models to new domains and capabilities [25, 24, 47]. For instance, a language model may be further trained on mathematical text to improve reasoning [37], or a visual backbone may be further trained with self-supervised objectives to improve dense prediction [67]. In both cases, the training signal is usually a fixed proxy task [34]: next-token prediction for text [7, 45, 76], reconstruction/view-based or joint-embedding objectives for images [11, 2, 67]. These proxy tasks scale because they do not require dense human annotation [26, 7]. Yet a pretraining run is not judged by the proxy loss itself. It is judged by downstream performance.

This creates a mismatch between the unit of optimization and the unit of selection. Optimization happens one self-supervised update at a time; selection happens after entire continued-pretraining runs are evaluated downstream [42]. In current practice, downstream feedback enters through a coarse outer loop: practitioners evaluate checkpoints, revise the corpus, objective, curriculum, or augmentation recipe, and launch another run. This makes task-recipe design an expensive black-box search over full pretraining trajectories.

![](images/0f1d382fd86bf35a8c7119681b4df2f9a6a781e4b938d0f33e5b4a2ce3868608.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph a_Task_construction["(a) Task construction"]
  A["Unlabeled x ~ Du"] --> B["Task Construction c"]
  B --> C["Task (xc, y, m)"]
  C --> D["Learner hθ"]
  D --> E["Loss l(hθ(xc, m), y)"]
    end

    subgraph b_Text["(b) Text"]
  F["Input xc: The cat sat on the mat"] --> G["Text: Fixed c: standard NTP (one-hot)<br>[0 0 0 ... 1 0"]]
        H["Side info m: predict token 6 1 2 3 4 5 6"]
        I["Target y: Fixed c: adaptive soft NTP<br>[0 0 0.1 ... 0.7 0.1"]]
    end

    subgraph c_Image["(c) Image"]
        J["View 1: Fixed c: random augmentation / mask"]
        K["View 2: Target y: Adaptive augmentation / mask"]
        L["Image: Fixed c: standard pretraining"]
        M["Image: Learnable cφ: V-pretraining"]
    end
```
</details>

Figure 1: Task construction as the control surface in continued pretraining. A construction rule c maps each unlabeled example $x \sim \mathcal { D } _ { u }$ into a self-supervised prediction problem $( x _ { c } , y , m )$ ), such as one-hot next-token prediction in language modeling or paired views and targets in $\mathrm { D I N O - s t y l { \epsilon } }$ vision SSL. Standard continued pretraining fixes this rule before training, whereas V-pretraining replaces it with a feedback-trained designer $c _ { \phi }$ while keeping the learner update self-supervised.

This motivates our question of interest: given a pretrained foundation model, an unlabeled stream for continued pretraining, and a small set of verifiable examples that specify desired downstream capabilities, can we use this feedback during continued pretraining to improve desired capabilities under a limited compute budget, while preserving the generalization capabilities of the pretrained model?

A natural first-pass solution is to fine-tune on the feedback examples. However, fine-tuning is not necessarily the most efficient use of downstream samples when the feedback set is small. SFT [73], preference optimization [57], and RL [48, 65, 1] use downstream examples as learner supervision: labels, preferences, or rewards directly define the learner update. These methods are powerful and complementary, but they answer a different question: how to update the learner on downstream supervision. They do not tell us which unlabeled self-supervised pretraining update was worth taking during a continued-pretraining run; they replace that update with a downstream supervised or reward-based one. Our goal is instead to use downstream examples as a value signal over candidate pretraining updates, without applying downstream gradients or rewards as learner updates.

We introduce V-pretraining: value-based continued pretraining with downstream feedback, a framework for the downstream-guided design of self-supervised pretraining tasks (Figure 1). Vpretraining complements the target learner, with parameters θ, with a lightweight task designer, with parameters ϕ. The learner is trained only by a self-supervised pretraining loss. The designer instead dynamically controls how the self-supervised task is constructed during continued pretraining. For example, in language modeling, it shapes the next-token target distribution over a small candidate set; in vision, it selects instance-wise views or masks for self-supervised learning. A feedback batch defines a downstream evaluator gradient, but that gradient is not used to update the learner’s parameters directly; rather, it is used only to train the task designer.

The primary technical challenge is that the ideal designer objective is long-horizon and bilevel: minimize a downstream task loss function $L _ { \mathrm { d o w n } } ,$ i.e. choose task constructions whose induced learner trajectory performs well downstream [38, 19]. This is computationally infeasible at pretraining scale. So V-pretraining replaces this objective with a one-step value estimate [32, 53]. If a proposed pretraining task induces gradient $g _ { \mathrm { p r e } } ( \theta ; \phi )$ and the feedback set induces downstream gradient $g _ { \mathrm { d o w n } } ( \theta )$ , then for a small enough $\eta > 0$

$$
L _ {\text { down }} (\theta - \eta g _ {\text { pre }}) \approx L _ {\text { down }} (\theta) - \eta g _ {\text { down }} ^ {\top} g _ {\text { pre }}. \tag {1}
$$

Thus $g _ { \mathrm { d o w n } } ^ { \top } g _ { \mathrm { p r e } }$ estimates whether the unlabeled update would reduce downstream loss. V-pretraining trains the designer to maximize this value, then detaches the resulting targets or views and updates the learner with the usual pretraining loss. The pretraining loss has a target selected based on the output of the task designer. We instantiate V-pretraining in both language and vision modalities. For language models, the designer learns adaptive top-K soft targets for continued next-token pretraining. For vision backbones, it learns instance-wise views for DINO-style self-supervised learning. In both cases, V-pretraining changes the task construction, not the learner architecture, optimizer, unlabeled stream, or downstream evaluation protocol.

Contributions. We make the following contributions.

• Conceptual: We propose V-pretraining, a novel framework for continued pretraining. Notably, we formulate downstream-guided continued pretraining as learnable task construction, which introduces a new channel for guidance during continued pretraining. This guidance comes without directly supervising the pretraining task, but implicitly injects supervision via the task designer.  
• Algorithmic: We propose a step-level objective for V-pretraining that can be easily and efficiently optimized, and instantiated for different modalities. In particular, we instantiate the same principle as adaptive top-K target construction for language pretraining and learned view construction for DINO/I-JEPA style vision SSL. In both cases, feedback changes the self-supervised task, not the learner objective.  
• Empirical: We conduct extensive compute-matched experiments to demonstrate the utility of V-pretraining. We demonstrate on multiple modalities that V-pretraining improves downstream performance on target tasks without harming generalization. Under wall-clock-matched continued pretraining, V-pretraining achieves a +7.4 point gain or 33% relative improvement on GSM8K Pass@1 in language. In vision, V-pretraining improves DINOv3-ViT-L transfer from 51.33 to 52.40 mIoU on ADE20K and reduces NYUv2 RMSE from 0.5752 to 0.5522. Ablations, one-step probes, decontamination, transfer checks, multitask feedback, and overhead measurements show that the gains require task-relevant feedback and are not explained by generic smoothing, self-distillation, contamination, or direct post-training feedback.

## 2 V-pretraining Framework

Self-supervised pretraining and continued pretraining do not only specify a loss; they choose how an unlabeled sample is turned into a prediction problem [34]. We call this choice the task construction rule. Given an unlabeled example $x \sim \mathcal { D } _ { u }$ from the pretraining data distribution $\mathcal { D } _ { u } , { \bf a }$ (possibly randomized) task construction rule c(·) produces a modified input $x _ { c } , \mathrm { a }$ target y, and optional side information $m ,$ i.e., $( x _ { c } , y , m ) \sim c ( x )$ . The learner predicts $y$ by outputting $\hat { y } = h _ { \theta } ( x _ { c } , m )$ and minimizes

$$
L _ {\text { pre }} (\theta ; c) = \mathbb {E} _ {x \sim \mathcal {D} _ {u}} \mathbb {E} _ {(x _ {c}, y, m) \sim c (x)} \ell (h _ {\theta} (x _ {c}, m), y). \tag {2}
$$

In next-token pretraining [7], given a token sequence $x = ( w _ { 1 } , \ldots , w _ { n } )$ , c selects a position $t \in$ $\{ 1 , \ldots , n \}$ , sets $x _ { c } = w _ { < t }$ , uses side information $m = t .$ , and sets the target to the one-hot next token $y = \delta _ { w _ { t } }$ . In vision masked modeling [26, 75, 2], c chooses which tokens or patches to hide and defines the reconstruction targets. In DINO-style [67] vision ${ \mathrm { S S L } } , c$ samples crops [9], augmentations, and masks; the target is the teacher representation induced by the paired view, and the learner predicts it from the student view.

Standard continued pretraining fixes c before the run. This does not mean the constructed tasks are deterministic: token positions, masks, crops, and augmentations can still be random [10, 26]. It means the distribution that constructs inputs and targets is chosen in advance and does not depend on downstream feedback during training. Our language baseline fixes c to one-hot next-token prediction on the continued-pretraining text stream. Our vision baseline fixes c to the DINO view-generation pipeline. Downstream performance may motivate these recipes, but once training starts, each learner update is optimized for the fixed proxy task rather than for an explicit estimate of downstream value.

V-pretraining alters the construction rule c and define c as a function of the unlabeled data batch (Figure 1). It keeps the learner architecture, optimizer, loss family, and unlabeled data stream fixed, but replaces the fixed rule c with a designer-controlled rule $c _ { \phi } .$ In natural language, $c _ { \phi }$ changes the target distribution for next-token prediction while keeping the context and text stream fixed. In vision, $c _ { \phi }$ changes instance-wise views or masks while keeping the SSL objective fixed. Thus the controlled object is not the learner itself, but the self-supervised task that generates the learner’s next update.

## 2.1 Indirect feedback through task construction

Let ${ \mathcal { D } } _ { \mathrm { f b } }$ be a small feedback set with labels, rewards, or other verifiable downstream signals, and let $\mathcal { D } _ { \mathrm { e v a l } }$ be a held-out evaluation set. Direct post-training methods use ${ \mathcal { D } } _ { \mathrm { f b } }$ as learner supervision: SFT [48, 73], preference optimization [40, 57], or RL [1, 65] supply a downstream loss or reward update to θ. V-pretraining uses the same kind of information during pretraining phase. The feedback set may train the task designer but we do not use it to directly train the learner.

Formally, every learner update must remain a self-supervised pretraining update on an unlabeled batch $B _ { t } ^ { u } \subset D _ { u }$ :

$$
g _ {t} ^ {\text { learn }} = \nabla_ {\theta} L _ {\text { pre }} \left(\theta_ {t}; \operatorname{sg} (c _ {\phi_ {t}} (B _ {t} ^ {u}))\right), \quad \theta_ {t + 1} = \theta_ {t} - \eta_ {t} g _ {t} ^ {\text { learn }}. \tag {3}
$$

Here $\operatorname { s g } ( \cdot )$ denotes stop-gradient through the constructed targets or views. We write parameter updates using gradient descent for clarity; other optimizers like AdamW [36] can simply replace $- \eta _ { t } g _ { t }$ with the adaptive optimizer step, and all comparisons use the same learner optimizer and schedule. The key invariant is the gradient supplied to the learner: no term of the form $\dot { \nabla } _ { \theta } L _ { \mathrm { d o w n } } ( \theta _ { t } ; \mathcal { D } _ { \mathrm { f b } } )$ is applied to θ.

This distinction is essential. Fine-tuning answers how the learner should fit the feedback examples [73]. V-pretraining asks which unlabeled self-supervised update is worth taking before the run is over (Figure 1). The feedback signal is therefore direct-to-designer but indirect-to-learner: it can change $c _ { \phi } ( \bar { B } _ { t } ^ { u } )$ , and hence the pretraining gradient, but it cannot replace that gradient with a downstream supervised or reward gradient.

## 2.2 A local value objective for task design

The ideal designer would choose a construction rule whose full continued-pretraining trajectory performs well downstream:

$$
\min _ {\phi} L _ {\text { down }} (\theta_ {T} (\phi)), \tag {4}
$$

$$
\theta_ {t + 1} (\phi) = \theta_ {t} (\phi) - \eta_ {t} \nabla_ {\theta} L _ {\text { pre }} \left(\theta_ {t} (\phi); c _ {\phi} (B _ {t} ^ {u})\right), \quad t = 0, \dots , T - 1. \tag {5}
$$

Differentiating through this trajectory is impractical at pretraining scale [18, 59, 19, 29]. V-pretraining instead asks a local question at each step: among the tasks that can be constructed from the current unlabeled batch, which one induces a learner gradient that would most decrease the feedback loss? For an unlabeled batch $B ^ { u }$ and a labeled feedback batch $B ^ { f }$ , the designer induces a candidate pretraining gradient

$$
g _ {\text { pre }} (\theta ; \phi , B ^ {u}) = \nabla_ {\theta} L _ {\text { pre }} \big (\theta ; c _ {\phi} (B ^ {u}) \big), \tag {6}
$$

and the feedback batch defines an evaluator gradient

$$
g _ {\text { down }} (\theta ; B ^ {f}) = \nabla_ {\theta} L _ {\text { down }} (\theta ; B ^ {f}). \tag {7}
$$

The candidate learner update is $\theta ^ { + } = \theta - \eta g _ { \mathrm { p r e } } .$ A first-order expansion [32, 53] gives

$$
L _ {\text { down }} (\theta^ {+}; B ^ {f}) \approx L _ {\text { down }} (\theta ; B ^ {f}) - \eta g _ {\text { down }} (\theta ; B ^ {f}) ^ {\top} g _ {\text { pre }} (\theta ; \phi , B ^ {u}). \tag {8}
$$

We therefore define the value of the proposed pretraining update as the inner product

$$
V (\phi ; \theta , B ^ {u}, B ^ {f}) = g _ {\text { down }} (\theta ; B ^ {f}) ^ {\top} g _ {\text { pre }} (\theta ; \phi , B ^ {u}), \tag {9}
$$

and train the designer with $L _ { \mathrm { m e t a } } ( \phi ) = - V ( \phi ; \theta , B ^ { u } , B ^ { f } )$ . Gradients through $g _ { \mathrm { d o w n } }$ are stopped; the meta-gradient changes ϕ according to how its task construction changes the learner gradient. For efficiency, we compute the dot product on a parameter subset S such as the last transformer blocks or task-relevant projection layers. After the designer update, the task is reconstructed and detached, and the learner applies the gradient-descent pretraining update above. Under smoothness, maximizing V maximizes the first-order term in a lower bound on one-step downstream improvement; the proofs and full algorithm are in Appendix A and Appendix B.

## 2.3 One-step justification

We now state the local guarantee underlying the value objective. The result is not a long-horizon convergence theorem. Instead, it shows that the alignment objective maximizes a first-order lower bound on one-step downstream improvement.

Proposition 2.1 (1-step downstream improvement lower-bound). Let $L _ { \mathrm { d o w n } }$ be L-smooth in θ. Let

$$
\theta^ {+} = \theta - \eta g _ {\text { pre }} (\theta ; \phi) \tag {10}
$$

for step size $\eta > 0 ,$ , and define

$$
V (\phi ; \theta) = g _ {\text { down }} (\theta) ^ {\top} g _ {\text { pre }} (\theta ; \phi), \quad g _ {\text { down }} (\theta) = \nabla_ {\theta} L _ {\text { down }} (\theta). \tag {11}
$$

Then

$$
L _ {\text { down }} (\theta) - L _ {\text { down }} (\theta^ {+}) \geq \eta V (\phi ; \theta) - \frac {L \eta^ {2}}{2} \| g _ {\text { pre }} (\theta ; \phi) \| _ {2} ^ {2}. \tag {12}
$$

The proposition explains the role of V . For sufficiently small learner steps, or when the pretraining update norm is controlled, increasing $g _ { \mathrm { d o w n } } ^ { \top } g _ { \mathrm { p r e } }$ increases a certified lower bound on one-step downstream loss decrease. Equivalently, define the one-step bilevel objective

$$
J (\phi ; \theta) = L _ {\text { down }} \left(\theta - \eta \nabla_ {\theta} L _ {\text { pre }} (\theta ; \phi)\right). \tag {13}
$$

A Taylor expansion gives $J ( \phi ; \theta ) = L _ { \mathrm { d o w n } } ( \theta ) - \eta V ( \phi ; \theta ) + O ( \eta ^ { 2 } )$ . Thus maximizing V is equivalent to minimizing the first-order approximation of the one-step downstream objective.

This local result is deliberately modest. It does not claim that every high-value step guarantees monotonic downstream improvement over a long training trajectory. Stochastic gradients, optimizer state, distribution shift between feedback and evaluation, and interactions across steps can all affect the final trajectory. The purpose of the value objective is to provide a scalable online signal for task design. Section 3 evaluates whether it improves downstream performance over full continued-pretraining runs. We leave the detailed proofs to Appendix B.3.

## 2.4 Instantiations

Language: adaptive top-K soft targets. For a token sequence ${ \boldsymbol x } = ( w _ { 1 } , w _ { 2 } , \ldots )$ , standard nexttoken pretraining uses context $w _ { < t }$ and one-hot target $y = \delta _ { w _ { t } }$ V-pretraining keeps the context, token positions, and text stream fixed, but lets the designer replace the one-hot target with a bounded soft target over a small candidate set $C _ { t }$ :

$$
C _ {t} = \{w _ {t} \} \cup \text { TopK } _ {K - 1} \big (\text { sg } (p _ {\theta} (\cdot \mid w _ {<   t})) \setminus \{w _ {t} \} \big). \tag {14}
$$

The true next token is always included. The task construction rule $c _ { \phi }$ outputs a distribution $r _ { \phi , t } \in$ $\Delta ( C _ { t } )$ and a gating constant $\alpha _ { \phi , t } \in [ 0 , \alpha _ { \mathrm { m a x } } ]$ , producing weights for each element $v \in C _ { t } .$ :

$$
q _ {\phi , t} (v) = \left(1 - \alpha_ {\phi , t}\right) \mathbf {1} [ v = w _ {t} ] + \alpha_ {\phi , t} r _ {\phi , t} (v), \quad v \in C _ {t}. \tag {15}
$$

The learner minimizes cross-entropy to detached $q _ { \phi , t }$ on the continued-pretraining text. Feedback examples such as GSM8K [15] problems define $L _ { \mathrm { d o w n } }$ for the designer, but they are not inserted as supervised learner examples.

Vision: learned views for self-supervised learning. We take DINO-style [10, 46, 67] continued SSL $V _ { 0 } ( x ) = \{ v _ { 1 } ^ { 0 } , . . . , v _ { M } ^ { 0 } \}$ V-pretraining keeps the DINO loss and student–teacher learner fixed, but lets the designer modify selected views through instance-wise masks or view parameters, producing $V _ { \phi } ( x )$ . The learner uses the same SSL loss as the baseline,

$$
L _ {\text { pre }} ^ {\text { vis }} (\theta ; \phi , B ^ {u}) = \frac {1}{| B ^ {u} |} \sum_ {x \in B ^ {u}} \ell_ {\text { SSL }} (\theta , \bar {\theta}; V _ {\phi} (x)), \tag {16}
$$

where $\bar { \theta }$ is the EMA teacher used to anchor the DINO SSL pipeline. The feedback evaluator uses lightweight downstream task heads like segmentation and depth estimation on top of the backbone,

$$
L _ {\text { down }} ^ {\text { vis }} = \omega_ {\text { seg }} L _ {\text { seg }} (H _ {\text { seg }} (F _ {\theta} (x)), y _ {\text { seg }}) + \omega_ {\text { dep }} L _ {\text { dep }} (H _ {\text { dep }} (F _ {\theta} (x)), y _ {\text { dep }}) + \dots , \tag {17}
$$

Segmentation [81] or depth estimation [44] labels train the task designer through $g _ { \mathrm { d o w n } } ;$ the backbone update remains the detached SSL loss on unlabeled ImageNet [17] images. By simply replacing the task construction rule $c ,$ this instantiation also applies to other SSL method backends like I-JEPA [2].

## 3 Experiments

We evaluate whether downstream feedback can steer continued pretraining through task construction rather than direct learner supervision. The main experiments ask: (i) Does V-pretraining improve target capabilities under a strict compute comparison? (ii) Does feedback steering over-specialize the model to the feedback task and harm broader transfer? (iii) Are the gains caused by leakage or shortcut supervision, where the learner effectively sees benchmark or feedback labels? (iv) Does the gradient-alignment value signal matter, or are the gains explained by generic smoothing, self-distillation, or additional stochasticity? (v) How does indirect feedback compare with direct post-training, and what overhead does it introduce? (vi) Can the same feedback channel scale across model sizes and support multitask downstream feedback?

## 3.1 Protocol

All main comparisons are matched by wall-clock training time on the same hardware configuration. For each model size, modality, and method, training is run for the same elapsed time under the same hardware and software configuration [31, 39, 33]. The initial learner, unlabeled stream, sequence length or crop configuration, optimizer, numerical precision, and data-loading protocol are held fixed. If V-pretraining introduces additional per-step overhead, it must pay that cost within the same elapsed time. Thus the baseline is allowed to complete as many learner updates and process as many unlabeled tokens or images as its faster training loop permits. V-pretraining receives no extra time to compensate for task-designer computation.

Language. We continue pretraining Qwen1.5 [72] and Qwen2.5 [54] base models on NuminaMath-CoT [35] in main experiments. The baseline is standard next-token continued pretraining with one-hot targets. V-pretraining uses the same learner configuration and text stream, but replaces one-hot targets with designer-shaped top-K targets. The feedback set contains 1,024 GSM8K training examples, used only to compute $g _ { \mathrm { d o w n } }$ for the designer. Evaluation is GSM8K test Pass@1 with greedy decoding. See Appendix F for more details and hyper-parameters.

Vision. We continue self-supervised training of DINOv3 ViT-B and ViT-L backbones [67] on ImageNet-1K [17] with a DINO-style student-teacher objective. The baseline uses the fixed viewgeneration pipeline of vanilla DINOv3. V-pretraining uses learned instance-wise views with the same SSL objective and learner schedule. Dense-task feedback comes from small labeled ADE20K and NYUv2 pools. We report ADE20K mIoU, NYUv2 RMSE, ImageNet linear accuracy, and transfer diagnostics [44, 81, 55].

## 3.2 V-pretraining improves target capabilities

Table 1 reports the main wall-clock-matched results. In this part, we focus on tasks that are aligned with the capabilities of the small feedback dataset used to train the task designer. In language, V-pretraining improves GSM8K Pass@1 across the tested Qwen models. The largest gain is the Qwen2.5-0.5B single-run result, improving from 22.20 to 29.60 (a 33% gain). In the replicated Qwen1.5 runs, V-pretraining improves 0.5B, 4B, and 7B models, with the strongest replicated gain at 4B. The gains decrease with model size, suggesting that larger learners already extract more math-relevant signal from standard continued next-token prediction.

In vision, we provide downstream feedback from dense prediction tasks for V-pretraining. Vpretraining improves dense prediction while preserving global recognition (a task that was not used to train the task designer). For DINOv3 ViT-B and ViT-L, ADE20K mIoU increases and NYUv2 RMSE decreases; ImageNet linear accuracy also slightly improves. These results support the feasibility claim: downstream feedback can improve continued pretraining through task construction, even though the learner update remains self-supervised and the training time is matched.

## 3.3 V-pretraining avoids over-specialization

Because the feedback sets are small and task-specific, V-pretraining could improve the target metric by steering the learner too narrowly. We therefore evaluate tasks that are not used as feedback.

Table 2 evaluates whether feedback steering harms tasks not used as feedback. For language, the OMEGA benchmark [70] measures math reasoning under distribution shift, while MMLU benchmark [27] measures broader zero-shot multiple-choice knowledge and reasoning. V-pretraining improves OMEGA for the 4B model, remains similar for 0.5B and 7B, and leaves MMLU nearly unchanged for 4B and 7B. The 0.5B model, however, drops in MMLU. This indicates that V-pretraining is a steering mechanism, not a guarantee of monotonic improvement on every unrelated task; small learners can be more sensitive to narrow feedback.

<table><tr><td>Modality</td><td>Benchmark</td><td>Model</td><td>Baseline</td><td>V-pretraining</td></tr><tr><td rowspan="4">Language</td><td>MATH Pass@1↑</td><td>Qwen2.5-0.5B</td><td>22.20</td><td>29.6</td></tr><tr><td>GSM8K Pass@1↑</td><td>Qwen1.5-0.5B</td><td> $19.15 \pm 1.16$ </td><td> $22.67 \pm 1.05$ </td></tr><tr><td>GSM8K Pass@1↑</td><td>Qwen1.5-4B</td><td> $56.48 \pm 1.56$ </td><td> $58.98 \pm 1.03$ </td></tr><tr><td>GSM8K Pass@1↑</td><td>Qwen1.5-7B</td><td> $65.26 \pm 1.06$ </td><td> $66.17 \pm 0.63$ </td></tr><tr><td rowspan="9">Vision</td><td>NYUv2 RMSE↓</td><td>DINOv3-ViT-B</td><td>0.5888</td><td>0.5697</td></tr><tr><td>NYUv2 RMSE↓</td><td>DINOv3-ViT-L</td><td>0.5752</td><td>0.5522</td></tr><tr><td>NYUv2 RMSE↓</td><td>I-JEPA-ViT-H</td><td>0.7835</td><td>0.7797</td></tr><tr><td>ADE20K mIoU↑</td><td>DINOv3-ViT-B</td><td>48.82</td><td>49.68</td></tr><tr><td>ADE20K mIoU↑</td><td>DINOv3-ViT-L</td><td>51.33</td><td>52.47</td></tr><tr><td>ADE20K mIoU↑</td><td>I-JEPA-ViT-H</td><td>28.03</td><td>28.19</td></tr><tr><td>ImageNet-1K Acc.↑</td><td>DINOv3-ViT-B</td><td>80.74</td><td>81.01</td></tr><tr><td>ImageNet-1K Acc.↑</td><td>DINOv3-ViT-L</td><td>84.07</td><td>84.59</td></tr><tr><td>ImageNet-1K Acc.↑</td><td>I-JEPA-ViT-H</td><td>92.40</td><td>92.50</td></tr></table>

Table 1: Main target-capability results under matched wall-clock training budgets. Feedback examples train only the task designer; the learner is updated only by a self-supervised loss on the unlabeled stream. Qwen2.5-0.5B is a single-run diagnostic; Qwen1.5 results report mean and standard deviation.

<table><tr><td>Benchmark</td><td>Model</td><td>Baseline</td><td>V-pretraining</td></tr><tr><td>OMEGA Acc. ↑</td><td>0.5B</td><td>0.65</td><td>0.65</td></tr><tr><td>OMEGA Acc. ↑</td><td>4B</td><td>1.44</td><td>1.88</td></tr><tr><td>OMEGA Acc. ↑</td><td>7B</td><td>1.52</td><td>1.50</td></tr><tr><td>MMLU Acc. ↑</td><td>0.5B</td><td>38.08</td><td>35.01</td></tr><tr><td>MMLU Acc. ↑</td><td>4B</td><td>53.32</td><td>53.51</td></tr><tr><td>MMLU Acc. ↑</td><td>7B</td><td>58.81</td><td>58.68</td></tr></table>

Table 2: Language generalization beyond the GSM8K feedback task. V-pretraining does not over-specialize larger models onto a single downstream task, but the 0.5B model shows a generalization drop on MMLU.

<table><tr><td>Dataset</td><td>Protocol</td><td>Baseline</td><td>V-pretraining</td></tr><tr><td>R-Oxford5k mAP ↑</td><td>Easy</td><td>0.5268</td><td>0.6048</td></tr><tr><td>R-Oxford5k mAP ↑</td><td>Medium</td><td>0.4072</td><td>0.4557</td></tr><tr><td>R-Oxford5k mAP ↑</td><td>Hard</td><td>0.0867</td><td>0.0820</td></tr><tr><td>R-Paris6k mAP ↑</td><td>Easy</td><td>0.5433</td><td>0.5973</td></tr><tr><td>R-Paris6k mAP ↑</td><td>Medium</td><td>0.6332</td><td>0.7005</td></tr><tr><td>R-Paris6k mAP ↑</td><td>Hard</td><td>0.2208</td><td>0.2509</td></tr></table>

Table 3: Vision transfer beyond the dense feedback tasks. We evaluate frozen ViT-L representations on instance retrieval. Dense-task feedback improves most retrieval protocols, with a small decrease on Oxford Hard.

For vision, we evaluate frozen ViT-L representations on Oxford5k & Paris6k instance retrieval [55], which are not used as feedback tasks. Table 3 shows that dense-task feedback improves most retrieval protocols, including Oxford Easy/Medium and all Paris protocols. Oxford Hard is slightly lower. Overall, dense feedback does not collapse the representation onto merely segmentation or depth, though improvements are not uniform across all retrieval settings.

## 3.4 Does the task designer create a shortcut for the learner?

A separate concern is that V-pretraining might improve the target metric through a shortcut rather than through better self-supervised updates. There are two possible shortcuts. First, the unlabeled continued-pretraining stream might contain near-duplicates of evaluation examples. Second, the task designer might effectively turn feedback examples into learner supervision. We test the first possibility experimentally and rule out the second by the training channel.

Decontamination. We decontaminate NuminaMath-CoT by removing similar samples of GSM8K and MATH using MinHash LSH and n-gram Jaccard similarity [22, 14]. We provide details of this decontamination in Appendix C.2. We then retrain the Qwen1.5-4B baseline and V-pretraining models under the same learner update budget. Table 4 shows that V-pretraining remains above the baseline after decontamination. The margin is smaller than in the original run, but the result suggests that the main gain is not primarily caused by memorizing benchmark-like examples in the unlabeled stream.

<table><tr><td>Unlabeled stream</td><td>Baseline</td><td>V-pretraining</td></tr><tr><td>Original NuminaMath-CoT</td><td>56.48</td><td>58.98</td></tr><tr><td>Decontaminated NuminaMath-CoT</td><td>56.7</td><td>57.5</td></tr></table>

Table 4: GSM8K Pass@1 for Qwen1.5-4B before and after removing near-duplicates of GSM8K and MATH from the continued-pretraining stream. Baseline and V-pretraining are compared under the same wall-clock training budget. Decontamination does not remove the advantage of V-pretraining.

<table><tr><td>Method</td><td>GSM8K Pass@1 ↑</td></tr><tr><td>Baseline</td><td>56.48</td></tr><tr><td>Rand-feedback</td><td>54.31</td></tr><tr><td>Uni-smoothing</td><td>54.58</td></tr><tr><td>Self-distillation</td><td>57.61</td></tr><tr><td>V-pretraining</td><td>58.98</td></tr></table>

Table 5: Ablations in the Qwen1.5- 4B setting. V-pretraining is the only variant that trains the designer with $g _ { \mathrm { d o w n } } ^ { \top } g _ { \mathrm { p r e } } .$ .

<table><tr><td colspan="4">Vision multitask feedback</td></tr><tr><td>Method</td><td>ADE20K mIoU ↑</td><td>NYUv2 RMSE ↓</td><td>-</td></tr><tr><td>Baseline</td><td>48.85</td><td>0.4139</td><td>-</td></tr><tr><td>V-pretraining</td><td>49.49</td><td>0.4135</td><td>-</td></tr><tr><td colspan="4">Language multitask feedback</td></tr><tr><td>Method</td><td>GSM8K Acc. ↑</td><td>MMLU Acc. ↑</td><td>MMBP Acc. ↑</td></tr><tr><td>Baseline</td><td>44.28</td><td>41.90</td><td>4.00</td></tr><tr><td>V-pretraining</td><td>45.26</td><td>42.17</td><td>4.00</td></tr></table>

Table 6: Multitask feedback under wall-clock matching. Each V-pretraining row is a single run using a combined feedback gradient rather than separately tuned single-task feedback.

No shortcut learner supervision. The second shortcut is ruled out by construction. During a V-pretraining learner step, the learner never receives a gradient of the form $\nabla _ { \theta } L _ { \mathrm { d o w n } } ( \theta ; D _ { \mathrm { f b } } )$ , and feedback examples are not inserted into the learner’s pretraining batch. The designer can change the target distribution in language or the views/masks in vision, but these constructed targets or views are detached before the learner update.

## 3.5 Downstream task signal matters for V-pretraining

The main results could have several alternative explanations, including generic label smoothing, self-distillation, or additional stochasticity. We therefore isolate the role of downstream-aligned value feedback in the Qwen1.5-4B setting. Random feedback replaces $g _ { \mathrm { d o w n } }$ with random vectors [12]. Uniform top-K smoothing uses soft targets without downstream feedback [21, 52]. Self top-K distillation trains on the learner’s own candidate distribution [20]. Table 5 shows that these alternatives do not match V-pretraining under the same wall-clock budget. Random feedback and uniform smoothing fall below the one-hot baseline, and self-distillation improves over baseline but remains below downstream-aligned value feedback. Thus the gain is not explained by soft labels or extra meta-gradient noise; it requires task-relevant feedback.

## 3.6 V-pretraining as a lower-cost feedback before post-training

We show that V-pretraining can be a low-cost complement to post-training. Direct fine-tuning methods such as SFT [48, 73] or GRPO [65] use downstream labels or rewards as learner updates. V-pretraining uses them only to train the task designer.

We compare V-pretraining with the common direct fine-tuning alternatives under a different shared math data source (Figure 2) to show that the trend is robust to data distribution. SFT and V-pretraining both train on OpenMathReasoning [43]: SFT uses the dataset as direct learner supervision, while V-pretraining uses it as the continued training stream with feedback-guided target construction. GRPO also trains on math questions in OpenMathReasoning. So the direct-feedback baselines and V-pretraining are exposed to the same source of math problems but use the signal through different training channels. This comparison is therefore a data-source-controlled diagnostic.

Figure 2 shows that V-pretraining reaches higher GSM8K Pass@1 than SFT runs with comparable or larger additional GPU hours, and also outperforms GRPO alone in this setting. The SFT+GRPO pipeline achieves the highest Pass@1, but at larger training cost. These results support the intended interpretation: V-pretraining can inject useful math capability during continued training before direct post-training, making it a compute-efficient complement to SFT or RL.

GSM8K Pass@1 vs GPU Hours  
![](images/64e54a0dcb96328cc50f1ac787ea637daf5a6c043945fe9b86e0a305cd22cb90.jpg)

<details>
<summary>line chart</summary>

| Training GPU Hours | VPT   | SFT   | GRPO  | SFT+GRPO |
| ------------------ | ----- | ----- | ----- | -------- |
| 0.25               | 79.5  | 77.5  | -     | -        |
| 1.00               | -     | 78.0  | 78.0  | 78.0     |
| 2.00               | -     | -     | -     | 82.0     |
</details>

Figure 2: Comparing direct and indirect feedback approaches under a shared math data source. SFT and V-pretraining train on OpenMathReasoning CoT traces, while GRPO trains on OpenMathReasoning questions. V-pretraining uses downstream feedback indirectly through target construction, whereas SFT and GRPO apply supervised or reward feedback directly to the learner. The comparison is diagnostic rather than an apples-to-apples post-training benchmark: SFT+GRPO achieves the highest Pass@1, while V-pretraining provides a lower-cost pre-post-training feedback channel.

<table><tr><td>Method</td><td>Grad. accum.</td><td>Tokens/step</td><td>Step time</td><td>Tokens/s ↑</td><td>Peak memory</td></tr><tr><td>Baseline</td><td>32</td><td>65,536</td><td>2.45s</td><td>26.7k</td><td>31,302 MiB</td></tr><tr><td>V-pretraining</td><td>20</td><td>40,960</td><td>1.69s</td><td>24.2k</td><td>36,440 MiB</td></tr></table>

Table 7: Runtime profiling for a representative current language run. Both methods use microbatch size 4 and sequence length 512. V-pretraining uses a smaller gradient-accumulation factor to fit the designer and meta-gradient computation, so throughput rather than raw step time is the primary comparison.

Runtime overhead. The preceding comparisons measure downstream performance per reported GPUhour. We also profile the implementation overhead of the indirect channel itself in a representative language run. This answers a separate question: how much throughput and memory does V-pretraining add relative to standard continued pretraining, before any downstream accuracy is considered?

The V-pretraining run uses a lightweight task designer. The value objective aligns the last two learner layers and performs a meta update every 8 learner optimizer steps. Raw optimizer-step time is not the right cost metric because the meta run uses smaller gradient accumulation to fit the designer and meta-gradient computation. With microbatch size 4 and sequence length 512, the baseline processes $4 \times 3 2 \times 5 1 2 = 6 5 { , } 5 3 6$ tokens per optimizer step, while V-pretraining processes $4 \times 2 0 \times 5 1 2 = 4 0 { , } 9 6 0$ tokens per optimizer step.

Table 7 shows that the configured task designer adds measurable cost. Token throughput decreases by 9.4%, and peak memory increases by 5,138 MiB, or 16.4%. The shorter optimizer-step time for V-pretraining is therefore not evidence of lower cost, since each optimizer step processes fewer tokens. This overhead is already charged in our wall-clock-matched main results: the baseline is allowed to process more tokens or learner steps within the same elapsed time. Appendix C reports feedback-set coverage, Pass@k, token-efficiency curves, and additional language diagnostics.

## 3.7 Capacity scaling and multitask control

Table 1 tested whether the feedback channel transfers across learner scales (measured in number of parameters). A stricter test is whether this channel can compose multiple downstream value signals in one continued-pretraining run. For feedback tasks $j = 1 , \dots , J ,$ , we use a weighted evaluator gradient

$$
g _ {\text { down }} = \sum_ {j = 1} ^ {J} \omega_ {j} g _ {j}, \tag {18}
$$

where $g _ { j }$ is the feedback gradient for task $j$ and $\omega _ { j }$ controls its relative weight. The designer is then trained to construct self-supervised updates whose learner gradients align with this combined value signal.

For the language multitask run, we use a shared continued-pretraining mixture with three domains: math, code, and general instruction following. Each domain contains a raw component and a highquality instruction component: OpenWebMath [50] and MetaMathQA [79] for math, codeparrot-clean [6] and Magicoder [74] for code, and C4 [58] and Alpaca [71] for general text (see Appendix E). The mixture assigns 70% probability mass to raw data and 30% to high-quality instruction data, balanced equally across domains. Both the baseline and V-pretraining use the same mixture. Feedback batches are drawn only from the high-quality domain-matched pools and are used to compute the combined evaluator gradient, not to update the learner directly. We present the results in Table 6. V-pretraining improves capabilities for both GSM8K (44.28 → 45.26) and MMLU (41.9 → 42.17).

## 4 Contributions and Limitations

Our contributions can be summarized as follows. (1) We formulate continued pretraining with downstream feedback as task design rather than learner supervision: a small verifier trains a controller that constructs pretraining targets or views, while the learner remains trained only by a self-supervised loss. (2) We derive a scalable step-level value objective, $g _ { \mathrm { d o w n } } ^ { \top } g _ { \mathrm { p r e } } ,$ showing that it is the first-order surrogate for one-step downstream improvement and avoids differentiating through full pretraining trajectories. (3) We instantiate the same principle in language and vision, as adaptive top-K target construction for next-token pretraining and learned view construction for self-supervised vision. (4) We provide compute-matched evidence and controls showing that downstream feedback improves value per learner update and that the effect is not explained by random feedback, fixed smoothing, or self-distillation.

V-pretraining provides a local value signal, not a long-horizon optimality guarantee. The alignment objective can be noisy, depends on feedback quality, and may over-steer small learners toward narrow feedback tasks. The method also introduces additional computation for task generation and value-gradient estimation, although our main comparisons match wall-clock time.

## References

[1] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce-style optimization for learning from human feedback in llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12248–12267, 2024.  
[2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a jointembedding predictive architecture. arXiv preprint arXiv:2301.08243, 2023.  
[3] Stephen H. Bach, Bryan He, Alexander Ratner, and Christopher Ré. Learning the structure of generative models without labeled data, 2017.  
[4] Gregor Bachmann and Vaishnavh Nagarajan. The pitfalls of next-token prediction, 2025.  
[5] Wele Gedara Chaminda Bandara, Naman Patel, Ali Gholami, Mehdi Nikkhah, Motilal Agrawal, and Vishal M. Patel. Adamae: Adaptive masking for efficient spatiotemporal learning with masked autoencoders. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14507–14517, June 2023.  
[6] Loubna Ben Allal and Leandro von Werra. codeparrot-clean dataset. https://huggingface. co/datasets/codeparrot/codeparrot-clean, 2022.  
[7] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec  
Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc.  
[8] Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, and Jeff Wu. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision, 2023.  
[9] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 9912–9924. Curran Associates, Inc., 2020.  
[10] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the International Conference on Computer Vision (ICCV), 2021.  
[11] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations, 2020.  
[12] Jeonghwan Cheon, Sang Wan Lee, and Se-Bum Paik. Pretraining with random noise for fast and robust learning without weight transport. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 13748–13768. Curran Associates, Inc., 2024.  
[13] Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems (NeurIPS), 2017.  
[14] Karl Cobbe et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.  
[15] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.  
[16] Roi Cohen, Konstantin Dobler, Eden Biran, and Gerard de Melo. I don't know: Explicit modeling of uncertainty with an [idk] token. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 10935–10958. Curran Associates, Inc., 2024.  
[17] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.  
[18] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks, 2017.  
[19] Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi, and Massimilano Pontil. Bilevel programming for hyperparameter optimization and meta-learning, 2018.  
[20] Arvid Frydenlund, Gagandeep Singh, and Frank Rudzicz. Language modelling via learning to rank. Proceedings of the AAAI Conference on Artificial Intelligence, 36(10):10636–10644, Jun. 2022.  
[21] Camille Garcin, Maximilien Servajean, Alexis Joly, and Joseph Salmon. Stochastic smoothing of the top-k calibrated hinge loss for deep imbalanced classification. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 7208–7222. PMLR, 17–23 Jul 2022.  
[22] Aristides Gionis, Piotr Indyk, and Rajeev Motwani. Similarity search in high dimensions via hashing. In Proceedings of the 25th International Conference on Very Large Data Bases (VLDB), 1999.  
[23] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning, 2020.  
[24] Kshitij Gupta, Benjamin Thérien, Adam Ibrahim, Mats Leon Richter, Quentin Gregory Anthony, Eugene Belilovsky, Irina Rish, and Timothée Lesort. Continual pre-training of large language models: How to re-warm your model? In Workshop on Efficient Systems for Foundation Models @ ICML2023, 2023.  
[25] Suchin Gururangan, Ana Marasovic, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, ´ and Noah A. Smith. Don’t stop pretraining: Adapt language models to domains and tasks. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8342–8360, Online, July 2020. Association for Computational Linguistics.  
[26] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16000–16009, June 2022.  
[27] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.  
[28] Dan Hendrycks et al. Measuring mathematical problem solving with the MATH dataset. arXiv preprint arXiv:2103.03874, 2021.  
[29] Kaiyi Ji, Junjie Yang, and Yingbin Liang. Bilevel optimization: Convergence analysis and enhanced design, 2021.  
[30] Wei Jin, Xiaorui Liu, Xiangyu Zhao, Yao Ma, Neil Shah, and Jiliang Tang. Automated self-supervised learning for graphs, 2022.  
[31] Aaron Klein, Zhenwen Dai, Frank Hutter, Neil Lawrence, and Javier Gonzalez. Meta-surrogate benchmarking for hyperparameter optimization. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.  
[32] Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International Conference on Machine Learning (ICML), 2017.  
[33] Sameer Kumar, Yu Wang, Cliff Young, James Bradbury, Naveen Kumar, Dehao Chen, and Andy Swing. Exploring the limits of concurrency in ml training on google tpus. In A. Smola, A. Dimakis, and I. Stoica, editors, Proceedings of Machine Learning and Systems, volume 3, pages 81–92, 2021.  
[34] Yann LeCun. Predictive learning. Invited talk at the 30th Conference on Neural Information Processing Systems (NIPS), December 2016.  
[35] Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath. [https://huggingface.co/AI-MO/NuminaMath-CoT](https://github.com/ project-numina/aimo-progress-prize/blob/main/report/numina\_dataset.pdf), 2024.  
[36] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.  
[37] Zimu Lu, Aojun Zhou, Ke Wang, Houxing Ren, Weikang Shi, Junting Pan, Mingjie Zhan, and Hongsheng Li. Mathcoder2: Better math reasoning from continued pretraining on modeltranslated mathematical code. In The Thirteenth International Conference on Learning Representations, 2025.  
[38] Dougal Maclaurin, David Duvenaud, and Ryan P. Adams. Gradient-based hyperparameter optimization through reversible learning. In International Conference on Machine Learning (ICML), 2015.  
[39] Peter Mattson, Christine Cheng, Gregory Diamos, Cody Coleman, Paulius Micikevicius, David Patterson, Hanlin Tang, Gu-Yeon Wei, Peter Bailis, Victor Bittorf, David Brooks, Dehao Chen, Debo Dutta, Udit Gupta, Kim Hazelwood, Andy Hock, Xinyuan Huang, Daniel Kang, David Kanter, Naveen Kumar, Jeffery Liao, Deepak Narayanan, Tayo Oguntebi, Gennady Pekhimenko, Lillian Pentecost, Vijay Janapa Reddi, Taylor Robie, Tom St John, Carole-Jean Wu, Lingjie Xu, Cliff Young, and Matei Zaharia. Mlperf training benchmark. In I. Dhillon, D. Papailiopoulos, and V. Sze, editors, Proceedings of Machine Learning and Systems, volume 2, pages 336–349, 2020.  
[40] Yu Meng, Mengzhou Xia, and Danqi Chen. SimPO: Simple preference optimization with a reference-free reward. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.  
[41] Mike Mintz, Steven Bills, Rion Snow, and Daniel Jurafsky. Distant supervision for relation extraction without labeled data. In Keh-Yih Su, Jian Su, Janyce Wiebe, and Haizhou Li, editors, Proceedings of the Joint Conference of the 47th Annual Meeting of the ACL and the 4th International Joint Conference on Natural Language Processing of the AFNLP, pages 1003–1011, Suntec, Singapore, August 2009. Association for Computational Linguistics.  
[42] Warren Morningstar, Alex Bijamov, Chris Duvarney, Luke Friedman, Neha Kalibhat, Luyang Liu, Philip Mansfield, Renan Rojas-Gomez, Karan Singhal, Bradley Green, and Sushant Prakash. Augmentations vs algorithms: What works in self-supervised learning, 2024.  
[43] Ivan Moshkov, Darragh Hanley, Ivan Sorokin, Shubham Toshniwal, Christof Henkel, Benedikt Schifferer, Wei Du, and Igor Gitman. Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset. arXiv preprint arXiv:2504.16891, 2025.  
[44] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012.  
[45] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike,

Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024.

[46] Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023.  
[47] Yixin Ou, Yunzhi Yao, Ningyu Zhang, Hui Jin, Jiacheng Sun, Shumin Deng, Zhenguo Li, and Huajun Chen. How do LLMs acquire new knowledge? a knowledge circuits perspective on continual pre-training. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 19889–19913, Vienna, Austria, July 2025. Association for Computational Linguistics.  
[48] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc.  
[49] Long Ouyang, Jeff Wu, Xu Jiang, et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2022.  
[50] Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text, 2023.  
[51] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4195–4205, October 2023.  
[52] Ruotian Peng, Yi Ren, Zhouliang Yu, Weiyang Liu, and Yandong Wen. Simko: Simple pass@k policy optimization, 2025.  
[53] Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. In Advances in Neural Information Processing Systems (NeurIPS), 2020.  
[54] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025.  
[55] F. Radenovic, A. Iscen, G. Tolias, Y. Avrithis, and O. Chum. Revisiting oxford and paris: ´ Large-scale image retrieval benchmarking. In CVPR, 2018.  
[56] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems (NeurIPS), 2023.  
[57] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2024.  
[58] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020.  
[59] Aravind Rajeswaran, Chelsea Finn, Sham Kakade, and Sergey Levine. Meta-learning with implicit gradients, 2019.  
[60] Neil Rathi and Alec Radford. Shaping capabilities with token-level data filtering, 2026.  
[61] Alexander Ratner, Stephen H. Bach, Henry Ehrenberg, Jason Fries, Sen Wu, and Christopher Ré. Snorkel: rapid training data creation with weak supervision. Proceedings of the VLDB Endowment, 11(3):269–282, November 2017.  
[62] Colorado J Reed, Sean Metzger, Aravind Srinivas, Trevor Darrell, and Kurt Keutzer. Selfaugment: Automatic augmentation policies for self-supervised learning. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2673–2682, 2021.  
[63] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation, 2015.  
[64] Chenze Shao, Darren Li, Fandong Meng, and Jie Zhou. Continuous autoregressive language models, 2025.  
[65] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024.  
[66] Yuge Shi, N Siddharth, Philip HS Torr, and Adam R Kosiorek. Adversarial masking for self-supervised learning. In International Conference on Machine Learning, 2022.  
[67] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. DINOv3, 2025.  
[68] Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A Raffel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li. Fixmatch: Simplifying semisupervised learning with consistency and confidence. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 596–608. Curran Associates, Inc., 2020.  
[69] Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. Learning from noisy labels with deep neural networks: A survey, 2022.  
[70] Yiyou Sun, Shawn Hu, Georgia Zhou, Ken Zheng, Hannaneh Hajishirzi, Nouha Dziri, and Dawn Song. Omega: Can llms reason outside the box in math? evaluating exploratory, compositional, and transformative generalization, 2025.  
[71] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford\_alpaca, 2023.  
[72] Qwen Team. Introducing qwen1.5, February 2024.  
[73] Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations (ICLR), 2022.  
[74] Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. Magicoder: Empowering code generation with OSS-instruct. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 52632–52657. PMLR, 21–27 Jul 2024.  
[75] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In International Conference on Computer Vision and Pattern Recognition (CVPR), 2022.  
[76] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025.  
[77] Yuning You, Tianlong Chen, Yang Shen, and Zhangyang Wang. Graph contrastive learning automated. arXiv preprint arXiv:2106.07594, 2021.  
[78] Yuning You, Tianlong Chen, Zhangyang Wang, and Yang Shen. Bringing your own view: Graph contrastive learning without prefabricated data augmentations, 2022.  
[79] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023.  
[80] Jingqing Zhang, Yao Zhao, Mohammad Saleh, and Peter J. Liu. Pegasus: Pre-training with extracted gap-sentences for abstractive summarization, 2019.  
[81] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

## Appendix Contents

## A Algorithm 18

A.1 V-pretraining algorithm . . . 18  
A.2 Information bottlenecks . . . 18

## B One-step Value Objective 19

B.1 Derivation . . 19  
B.2 Parameter subset used for alignment 19  
B.3 One-step justification and bounds . . . . 19  
B.4 Relation to the actual optimizer . . . 20

## C Additional Language Results 20

C.1 One-step value probe . . . . 20  
C.2 Decontamination 21  
C.3 Generalization beyond the GSM8K verifier 21  
C.4 Feedback-set coverage 21  
C.5 Pass@k . . . 21  
C.6 Token-efficiency diagnostic . . . . 21  
C.7 Additional computation cost comparison . . . 22

## D Related Work 22

## E Multitask Feedback Data Construction 23

## F Main Language Experiment Setup 25

F.1 Single-task math continued pretraining . . . . 25  
F.2 Language implementation branches 26

## G Main Vision Experiment Setup 28

G.1 Baseline continued DINO pretraining . . 28  
G.2 V-pretraining with downstream-aware learned augmentations . . . . 28  
G.3 Dense evaluation protocol . . . . 30  
G.4 Hyperparameter selection . . . . 30

## H Declarations and Impacts 31

## I Compute Resources and Total Compute 31

Algorithm 1 V-pretraining  
Require: Initial learner $\theta_0$ , task designer $\phi_0$ , unlabeled stream $\mathcal{D}_u$ , feedback set $\mathcal{D}_{\mathrm{fb}}$ , learner budget $T$ , meta-update period $r$ , alignment parameter subset $\mathcal{S}$ 1: for $t = 0, \ldots, T - 1$ do

2:    Sample unlabeled batch $B_t^u \sim \mathcal{D}_u$ .

3:    if $t \mod r = 0$ then

4:    Sample feedback batch $B_t^f \sim \mathcal{D}_{\mathrm{fb}}$ .

5:    Construct task $c_{\phi_t}(B_t^u)$ with gradients enabled for $\phi_t$ .

6:    Compute $L_{\mathrm{pre}}^{\mathrm{meta}}(\theta_t; \phi_t, B_t^u)$ .

7: $g_{\mathrm{pre},\mathcal{S}} \leftarrow \nabla_{\theta_{\mathcal{S}}} L_{\mathrm{pre}}^{\mathrm{meta}}(\theta_t; \phi_t, B_t^u)$ .

8: $g_{\mathrm{down},\mathcal{S}} \leftarrow \mathrm{sg}\left( \nabla_{\theta_{\mathcal{S}}} L_{\mathrm{down}}(\theta_t; B_t^f) \right)$ .

9: $L_{\mathrm{meta}} \leftarrow -g_{\mathrm{down},\mathcal{S}}^\top g_{\mathrm{pre},\mathcal{S}}$ .

10: $\phi_{t + \frac{1}{2}} \leftarrow \mathrm{Opt}_{\phi}(\phi_t, \nabla_{\phi} L_{\mathrm{meta}})$ .

11:    else

12: $\phi_{t + \frac{1}{2}} \leftarrow \phi_t$ .

13:    end if

14:    Construct task $c_{\phi_{t + \frac{1}{2}}}(B_t^u)$ and detach its targets, masks, or views.

15: $L_{\mathrm{pre}}^{\mathrm{learn}} \leftarrow L_{\mathrm{pre}}\left( \theta_t; \mathrm{sg}\left( c_{\phi_{t + \frac{1}{2}}}(B_t^u) \right) \right)$ .

16: $\theta_{t+1} \leftarrow \mathrm{Opt}_{\theta}(\theta_t, \nabla_\theta L_{\mathrm{pre}}^{\mathrm{learn}})$ .

17: $\phi_{t+1} \leftarrow \phi_{t+\frac{1}{2}}$ .

18: end for

## A Algorithm

## A.1 V-pretraining algorithm

Algorithm 1 gives the generic V-pretraining procedure. Each iteration has two conceptually separate updates. First, the feedback batch is used to update the task designer through the value objective. Second, the learner is updated only on the resulting pretraining loss over the unlabeled batch. The regularizer $R ( \phi )$ is optional and instantiation-specific. It can restrict the designer to stay close to a base pretraining recipe, for example by limiting the amount of probability mass moved away from the true next token [16] or by enforcing mask sparsity and spatial smoothness [66]. These constraints prevent the designer from constructing arbitrary adversarial tasks and keep the learner update anchored to the unlabeled example.

## A.2 Information bottlenecks

The algorithm is defined by four information bottlenecks. (1) Feedback labels appear only in the evaluator. The feedback set ${ \mathcal { D } } _ { \mathrm { f b } }$ is used to compute $L _ { \mathrm { d o w n } }$ and $g _ { \mathrm { d o w n } }$ . These labels or rewards are not inserted into the learner’s pretraining batch. (2) The downstream gradient is not a learner update. Although $g _ { \mathrm { d o w n } }$ is a gradient with respect to learner parameters, it is never passed to the learner optimizer. It is detached and used only as an evaluator vector for updating $\phi .$ (3) The learner update is self-supervised. The learner step minimizes $L _ { \mathrm { p r e } } ^ { \mathrm { l e a r n } }$ on unlabeled data. The targets, masks, or views are produced by the designer but detached before the learner update. Thus the learner receives no supervised downstream loss. (4) Feedback reaches the learner only through task construction. The only path from downstream feedback to θ is

$$
\mathcal {D} _ {\mathrm{fb}} \rightarrow g _ {\text { down }} \rightarrow \phi \rightarrow c _ {\phi} (B _ {u}) \rightarrow \nabla_ {\theta} L _ {\text { pre }}. \tag {19}
$$

In practice, computing the alignment over all learner parameters can be expensive. We therefore allow the value to be computed on a subset of learner parameters $s ,$ such as adapters, the last transformer blocks, or task-relevant projection layers:

$$
V _ {\mathcal {S}} (\phi ; \theta , B _ {u}, B _ {f}) = g _ {\text { down }, \mathcal {S}} ^ {\top} g _ {\text { pre }, \mathcal {S}}. \tag {20}
$$

When $s$ is the full parameter set, this recovers the original $V .$ When $s$ is restricted, the objective is a computationally cheaper estimator of the same alignment principle. The derivation above is written for an SGD-style step to expose the basic mechanism. For adaptive optimizers, the same Taylor argument applies to the actual optimizer-induced update direction. Our implementation uses gradient alignment as a lightweight surrogate under a fixed learner optimizer and validates this surrogate empirically with a one-step probe

## B One-step Value Objective

## B.1 Derivation

At a training iteration, let $B ^ { u } \subset \mathcal { D } _ { u }$ be an unlabeled pretraining batch and $B ^ { f } \subset D _ { \mathrm { f b } }$ be a feedback batch. The designer induces a candidate pretraining gradient

$$
g _ {\text { pre }} (\theta ; \phi , B ^ {u}) = \nabla_ {\theta} L _ {\text { pre }} \big (\theta ; c _ {\phi} (B ^ {u}) \big), \tag {21}
$$

and the feedback batch defines an evaluator gradient

$$
g _ {\text { down }} (\theta ; B ^ {f}) = \nabla_ {\theta} L _ {\text { down }} (\theta ; B ^ {f}). \tag {22}
$$

For the candidate learner update

$$
\theta^ {+} = \theta - \eta g _ {\text { pre }} (\theta ; \phi , B ^ {u}), \tag {23}
$$

a first-order expansion of the feedback loss gives

$$
L _ {\text { down }} (\theta^ {+}; B ^ {f}) \approx L _ {\text { down }} (\theta ; B ^ {f}) - \eta   g _ {\text { down }} (\theta ; B ^ {f}) ^ {\top} g _ {\text { pre }} (\theta ; \phi , B ^ {u}). \tag {24}
$$

Thus the value score

$$
V (\phi ; \theta , B ^ {u}, B ^ {f}) = g _ {\text { down }} (\theta ; B ^ {f}) ^ {\top} g _ {\text { pre }} (\theta ; \phi , B ^ {u}) \tag {25}
$$

estimates the predicted one-step decrease in downstream loss induced by the proposed self-supervised update. V-pretraining trains the designer by minimizing

$$
L _ {\text { meta }} (\phi) = - V (\phi ; \theta , B ^ {u}, B ^ {f}) + \lambda R (\phi), \tag {26}
$$

where R is an optional regularizer that keeps the constructed task close to the base pretraining recipe. When updating $\phi , g _ { \mathrm { d o w n } }$ is treated as a detached evaluator vector:

$$
\nabla_ {\phi} L _ {\text { meta }} (\phi) = - \nabla_ {\phi} \left[ \operatorname{sg} (g _ {\text { down }} (\theta ; B ^ {f})) ^ {\top} \nabla_ {\theta} L _ {\text { pre }} (\theta ; c _ {\phi} (B ^ {u})) \right] + \lambda \nabla_ {\phi} R (\phi). \tag {27}
$$

This is a mixed Hessian-vector product. The designer is updated according to how its construction changes the learner gradient, as judged by the downstream evaluator gradient.

## B.2 Parameter subset used for alignment

Computing the alignment over all learner parameters can be expensive. We therefore compute the value objective on a subset S of learner parameters:

$$
V _ {S} (\phi ; \theta , B ^ {u}, B ^ {f}) = g _ {\text { down }, S} (\theta ; B ^ {f}) ^ {\top} g _ {\text { pre }, S} (\theta ; \phi , B ^ {u}). \tag {28}
$$

In language experiments, S is chosen from the final learner blocks or other task-relevant parameters. In the current profiled meta run, the alignment uses the last two learner layers. In vision experiments, S is chosen from the final backbone blocks used by the downstream evaluator heads. This reduces the cost of the mixed derivative while keeping the value signal tied to features used by the feedback tasks.

## B.3 One-step justification and bounds

We now state the local guarantee underlying the value objective. The result is not a long-horizon convergence theorem. Instead, it shows that the alignment objective maximizes a first-order lower bound on one-step downstream improvement.

Proposition B.1 (1-step downstream improvement lower-bound). Let $L _ { \mathrm { d o w n } }$ be L-smooth in θ. Let

$$
\theta^ {+} = \theta - \eta g _ {\text { pre }} (\theta ; \phi) \tag {29}
$$

for step size $\eta > 0 ,$ , and define

$$
V (\phi ; \theta) = g _ {\text { down }} (\theta) ^ {\top} g _ {\text { pre }} (\theta ; \phi), \quad g _ {\text { down }} (\theta) = \nabla_ {\theta} L _ {\text { down }} (\theta). \tag {30}
$$

Then

$$
L _ {\text { down }} (\theta) - L _ {\text { down }} (\theta^ {+}) \geq \eta V (\phi ; \theta) - \frac {L \eta^ {2}}{2} \| g _ {\text { pre }} (\theta ; \phi) \| _ {2} ^ {2}. \tag {31}
$$

Proof of Theorem B.1. By L-smoothness,

$$
L _ {\text { down }} (\theta^ {+}) \leq L _ {\text { down }} (\theta) + \nabla_ {\theta} L _ {\text { down }} (\theta) ^ {\top} (\theta^ {+} - \theta) + \frac {L}{2} \| \theta^ {+} - \theta \| _ {2} ^ {2}. \tag {32}
$$

Substituting $\theta ^ { + } - \theta = - \eta g _ { \mathrm { p r e } } ( \theta ; \phi )$ gives

$$
L _ {\text { down }} (\theta^ {+}) \leq L _ {\text { down }} (\theta) - \eta g _ {\text { down }} (\theta) ^ {\top} g _ {\text { pre }} (\theta ; \phi) + \frac {L \eta^ {2}}{2} \| g _ {\text { pre }} (\theta ; \phi) \| _ {2} ^ {2}. \tag {33}
$$

Rearranging yields Equation (31).

![](images/172fc519b651eedc7ae77211b18d75075c68cc57a75b114adcdb91d9609108ae.jpg)

The proposition explains the role of V . For sufficiently small learner steps, or when the pretraining update norm is controlled, increasing $g _ { \mathrm { d o w n } } ^ { \top } g _ { \mathrm { p r e } }$ increases a certified lower bound on one-step downstream loss decrease. Equivalently, define the one-step bilevel objective

$$
J (\phi ; \theta) = L _ {\text { down }} \left(\theta - \eta \nabla_ {\theta} L _ {\text { pre }} (\theta ; \phi)\right). \tag {34}
$$

A Taylor expansion gives $J ( \phi ; \theta ) = L _ { \mathrm { d o w n } } ( \theta ) - \eta V ( \phi ; \theta ) + O ( \eta ^ { 2 } )$ . Thus maximizing V is equivalent to minimizing the first-order approximation of the one-step downstream objective.

This local result is deliberately modest. It does not claim that every high-value step guarantees monotonic downstream improvement over a long training trajectory. Stochastic gradients, optimizer state, distribution shift between feedback and evaluation, and interactions across steps can all affect the final trajectory. The purpose of the value objective is to provide a scalable online signal for task design. Section 3 evaluates whether it improves downstream performance over full continued-pretraining runs.

## B.4 Relation to the actual optimizer

The main text and pseudocode use gradient descent for clarity. If the realized learner optimizer update is $d _ { t } ( \phi ) = \theta ^ { + } ( \phi ) \bar { { ^ { - } \theta _ { t } } }$ , the exact first-order value of that update is

$$
- g _ {\text { down }} (\theta_ {t}) ^ {\top} d _ {t} (\phi). \tag {35}
$$

For a preconditioned step $d _ { t } ( \phi ) \approx - \eta _ { t } P _ { t } g _ { \mathrm { p r e } } ( \theta _ { t } ; \phi )$ , this value becomes $g _ { \mathrm { d o w n } } ( \theta _ { t } ) ^ { \top } P _ { t } g _ { \mathrm { p r e } } ( \theta _ { t } ; \phi )$ . Our implementation uses the unpreconditioned alignment $g _ { \mathrm { d o w n } } ^ { \top } g _ { \mathrm { p r e } }$ as a lightweight surrogate under a fixed learner optimizer and schedule. The one-step probe in Appendix C.1 empirically checks that this surrogate is positively correlated with realized downstream loss decrease in the language setting.

## C Additional Language Results

## C.1 One-step value probe

The value objective predicts the one-step downstream loss decrease as $\eta g _ { \mathrm { d o w n } } ^ { \top } g _ { \mathrm { p r e } } .$ . To test whether this estimate is informative, we compute the predicted improvement on held-out GSM8K probe batches and compare it with the realized decrease in probe loss after a gradient-descent update on $g _ { \mathrm { p r e } } .$ Across probe measurements, the predicted and realized improvements have Pearson correlation $r = 0 . 6 5 7$ . The correlation is not expected to be perfect because the estimate is local, minibatchbased, and computed under a fixed surrogate for the realized optimizer step. Its positive value supports using the alignment score as an online training signal for the designer.

## C.2 Decontamination

To test whether the language gains are explained by near-duplicates in the continued-pretraining stream, we decontaminate NuminaMath-CoT by removing examples similar to GSM8K and MATH. We use MinHash LSH and n-gram Jaccard similarity to identify candidate overlaps, then retrain the Qwen1.5-4B baseline and V-pretraining models under the same wall-clock budget. Table 8 shows that V-pretraining remains above the baseline after decontamination, though with a smaller margin than on the original stream.

<table><tr><td>Unlabeled stream</td><td>Baseline</td><td>V-pretraining</td></tr><tr><td>Original NuminaMath-CoT</td><td>56.48</td><td>58.98</td></tr><tr><td>Decontaminated NuminaMath-CoT</td><td>56.70</td><td>57.50</td></tr></table>

Table 8: GSM8K Pass@1 for Qwen1.5-4B before and after removing near-duplicates of GSM8K and MATH from the continued-pretraining stream.

## C.3 Generalization beyond the GSM8K verifier

Table 9 evaluates whether GSM8K feedback collapses the learner onto the verifier task. V-pretraining improves OMEGA for the 4B learner and leaves MMLU nearly unchanged for the 4B and 7B learners. The 0.5B learner drops on MMLU, indicating that small models can be more sensitive to narrow feedback. These results support the interpretation of V-pretraining as a steering mechanism rather than a uniform improvement guarantee.

<table><tr><td>Benchmark</td><td>Model</td><td>Baseline</td><td>V-pretraining</td></tr><tr><td>OMEGA Acc. ↑</td><td>Qwen1.5-0.5B</td><td>0.65</td><td>0.65</td></tr><tr><td>OMEGA Acc. ↑</td><td>Qwen1.5-4B</td><td>1.44</td><td>1.88</td></tr><tr><td>OMEGA Acc. ↑</td><td>Qwen1.5-7B</td><td>1.52</td><td>1.50</td></tr><tr><td>MMLU Acc. ↑</td><td>Qwen1.5-0.5B</td><td>38.08</td><td>35.01</td></tr><tr><td>MMLU Acc. ↑</td><td>Qwen1.5-4B</td><td>53.32</td><td>53.51</td></tr><tr><td>MMLU Acc. ↑</td><td>Qwen1.5-7B</td><td>58.81</td><td>58.68</td></tr></table>

Table 9: Language generalization beyond the GSM8K feedback task. Larger learners do not collapse onto the verifier task, while the 0.5B learner shows a drop on MMLU.

## C.4 Feedback-set coverage

We vary the number of GSM8K feedback examples used to compute the evaluator gradient, using 1k, 2k, and 3k examples. Increasing feedback coverage improves the stability and strength of the Qwen1.5-4B gains, with diminishing returns after a few thousand examples. This suggests that the verifier set need not be large to be useful, but its quality and coverage still affect the value signal.

## C.5 Pass@k

We evaluate Pass@k for $k \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ . V-pretraining improves Pass@k across the tested k values and model sizes. This indicates that the designer improves the solution distribution rather than only changing greedy decoding behavior. The result is consistent with the interpretation that value-guided targets modify reasoning-relevant update directions during continued pretraining.

## C.6 Token-efficiency diagnostic

The main comparisons are wall-clock matched. We additionally plot GSM8K Pass@1 as a function of unlabeled tokens processed to diagnose update quality. This diagnostic asks whether a V-pretraining update carries more downstream value per unlabeled token once the designer begins steering. In the Qwen1.5-4B setting, V-pretraining reaches 56.18 Pass@1 after 400 learner steps, approximately $1 . 3 \times 1 0 ^ { 7 }$ unlabeled tokens, while the baseline requires roughly $1 0 ^ { 3 }$ steps to reach comparable accuracy. The curve shows an early transient dip before the designer stabilizes; this is mitigated by a burn-in schedule that delays designer updates. This analysis is diagnostic and does not replace the wall-clock-matched results.

![](images/64ef41474b8c06cc65493416304fcc0f9a33227013158c2260f6da18b688d214.jpg)

<details>
<summary>line chart</summary>

| Feedback Data Size (×10³) | Pass@1 |
| ------------------------- | ------ |
| 1.0                       | 57.4   |
| 2.0                       | 58.2   |
| 3.0                       | 58.2   |
</details>

![](images/1bce9a09306ed3c971ccb380aeea764b6bede4653455f7415c280079a7039a91.jpg)

<details>
<summary>bar chart</summary>

(b) Scaling Pass@k
| k | Bs(0.5B) | Vb(0.5B) | Bs(4B) | Vb(4B) | Bs(7B) | Vb(7B) |
|---|---|---|---|---|---|---|
| 1 | 20 | 22 | 58 | 60 | 65 | 68 |
| 2 | 23 | 25 | 65 | 70 | 75 | 78 |
| 4 | 35 | 38 | 75 | 80 | 85 | 88 |
| 8 | 45 | 48 | 85 | 90 | 92 | 94 |
| 16 | 55 | 58 | 90 | 92 | 94 | 96 |
</details>

![](images/61251866c264b555c425f6c412289cf12feddf0dfcd53c661628c30f890af079.jpg)

<details>
<summary>line chart</summary>

| Number of Tokens | Value-Based (7B) | Baseline (7B) | Value-Based (4B) | Baseline (4B) |
| ---------------- | ---------------- | ------------- | ---------------- | ------------- |
| 0.5              | 58               | 59            | 50               | 53            |
| 1.0              | 62               | 61            | 53               | 54            |
| 1.5              | 65               | 63            | 57               | 55            |
| 2.0              | 66               | 64            | 58               | 56            |
</details>

![](images/aa63470ec4c04f0127a42df916b40cc629300a90fb56ff4298366928f346ffaa.jpg)

<details>
<summary>line chart</summary>

| mIoU   | Pareto-optimal | Off-frontier |
| ------ | -------------- | ------------ |
| 0.480  | 0.430          | 0.428        |
| 0.495  | 0.415          | 0.412        |
</details>

Figure 3: Additional language diagnostics. (a): GSM8K Pass@1 as a function of feedback-set size. (b): Pass@k across tested model sizes. (c): token-efficiency diagnostic plotting Pass@1 against unlabeled tokens processed. These curves are diagnostics, not the primary fairness criterion. (d): Tradeoff between segmentation (mIoU) and depth estimation (1-RMSE) induced by varying feedback and task-designer hyperparameters.

<table><tr><td>Method</td><td>MATH Pass@1 ↑</td><td>Hardware / time</td><td>Relative cost</td></tr><tr><td>Baseline</td><td>52.48</td><td>10 min on RTX 6000 Ada</td><td>1×</td></tr><tr><td>V-pretraining</td><td>53.92</td><td>10 min on RTX 6000 Ada</td><td>1×</td></tr><tr><td>SFT+GRPO</td><td>53.8</td><td>16 hrs on H100</td><td>15–25×</td></tr></table>

Table 10: Direct-vs-indirect feedback diagnostic. GRPO applies downstream reward directly to the learner; V-pretraining applies downstream feedback only to the task designer. The comparison is diagnostic, not a claim that indirect feedback replaces post-training.

## C.7 Additional computation cost comparison

Table 10 gives a MATH [28] diagnostic: V-pretraining reaches comparable Pass@1 to an SFT+GRPO pipeline in the reported setting, while using the less training-time budget. Because the stages, hardware, and optimization pipelines differ, this comparison should be read only as evidence that some downstream value can be injected before direct reward optimization.

## D Related Work

Positioning. We study controlled pretraining under a fixed unlabeled stream and learner update budget. A small feedback set of verifiable downstream tasks provides verified goal information, but it is used only to train a lightweight controller that reshapes the pretraining target (or views). The foundation model is never updated on downstream labels. This differs from most label-efficient paradigms, which improve performance by creating labels or pseudo-labels and then training the main model on them.

Post-training injects direction late. Supervised fine-tuning and preference optimization steer models by directly updating the foundation model on labeled examples or preferences [13, 49, 56]. These methods are highly effective, but they operate after proxy pretraining has already shaped the representation space. Our approach is complementary: we inject goal information during pretraining by shaping the unlabeled training signal rather than updating the learner on downstream labels.

Weak/semi-supervision: scalable supervision by producing labels, not by steering pretraining updates. A broad literature improves supervision scalability by learning from imperfect labels or by manufacturing labels from weak sources, spanning weak supervision and data programming [61, 3], distant supervision [41], semi-supervised learning [68], robust learning from noisy labels [69], and more recently weak-to-strong generalization as a way to elicit strong capabilities from weak supervision [8]. Across these settings, progress typically comes from generating (pseudo-)labels and then training the main model on task-defined targets, often with repeated inference or teacher–student refinement that is not compute-matched to pretraining-scale update budgets. Our method can be viewed as a task-agnostic pretraining analogue of weak-to-strong generalization: a small feedback set of verifiable downstream tasks provides weak but reliable goal information, yet the foundation model is never trained on downstream labels; instead, the feedback trains a lightweight controller that reshapes the self-supervised target/views so that each unlabeled gradient step has higher downstream value.

Directing pretraining without step-level downstream feedback: proxy objectives and view design. Most improvements to foundation-model pretraining change the proxy objective or the view/augmentation pipeline while keeping the training signal fixed: in language this includes nexttoken-based variants and domain-shaped objectives specified a priori [7, 80, 4, 64], while in vision SSL many methods learn from global semantics via contrastive/joint-embedding objectives [11, 23, 10] and others inject spatial structure through handcrafted augmentations or predictive objectives such as masked modeling and JEPA-style prediction [26, 2]. These approaches can yield strong representations, but the direction they impose is largely static: the target construction does not adapt online to what a downstream verifier says is valuable for the current model and example [66, 5]. In contrast, value-based pretraining introduces a control loop that uses a small feedback set of verifiable downstream tasks to modulate the pretraining target/views so that each unlabeled update aligns with downstream improvement, directly addressing the value-per-step and feedback-efficiency pressures highlighted in our introduction.

Bilevel optimization and influence. Downstream-aware task design naturally leads to bilevel optimization and unrolled differentiation through training [38, 19], which is costly at pretraining horizons. To our knowledge, existing work has not optimized both pretraining tasks and SSL augmentations in a bilevel optimization [62]; the closest approaches use a coordinate-descent-like step-wise optimization [77, 78, 30]. We circumvent the computational challenges of this bilevel optimization using influence-style methods that estimate the effect of training updates on downstream loss from gradients [32, 53]. We build on these approximations but apply them to target/view construction during pretraining: a controller learns to reshape the unlabeled supervision signal so that each proxy update aligns with downstream improvement.

## E Multitask Feedback Data Construction

Goal. The multitask language experiment is designed to test whether the same indirect feedback channel can steer continued pretraining toward multiple capability domains in a single run. We consider three domains: mathematical reasoning, code generation, and general instruction following / world knowledge. The key design constraint is that feedback should provide a clean value signal for the task designer, while the learner still trains on a mixed continued-pretraining stream.

Pretraining mixture. The final multitask continued-pretraining corpus contains both raw domain text and high-quality instruction-format data. The raw component provides broad next-token prediction coverage; the high-quality component anchors the stream in task-relevant formats. The three domains are:

• Math: OpenWebMath [50] as raw mathematical text, and MetaMathQA [79] as high-quality math instruction data.  
• Code: codeparrot-clean [6] as raw code-domain text, and Magicoder OSS-Instruct-75K [74] as high-quality code instruction data.  
• General: C4 English [58] as raw web text, and Alpaca [71] as high-quality general instruction data.

The pretraining mixture assigns total probability mass 0.7 to raw data and 0.3 to high-quality instruction data. Within each group, probability mass is balanced equally across the three domains. Thus each raw domain receives mass 0.7/3, and each high-quality domain receives mass 0.3/3. Sampling is weighted rather than performed by simple concatenation, so large corpora such as C4 do not dominate smaller instruction datasets.

Why mix raw and high-quality data? We found that instruction-only mixtures make the experiment less diagnostic. If the pretraining stream consists almost entirely of clean instruction pairs (which is often infeasible at practical pretraining scale), the standard baseline can learn much of the task format directly from the pretraining data. In that case, improvements may reflect direct exposure to instruction data rather than feedback-controlled task construction. Conversely, very small or poorly balanced mixtures can over-repeat small domains, especially code, and create domain-specific overfitting. The final raw/high-quality mixture therefore separates two roles: raw text supplies broad domain coverage, while high-quality examples keep the stream aligned with the capabilities evaluated downstream.

Sampling caps. Before tokenization and chunking, we cap the number of source items drawn from each domain. In the full multitask builder, the default caps are 50k raw source documents per domain and 5k high-quality examples per domain for the pretraining portion. Because raw documents are chunked into fixed-length token sequences after tokenization, the number of resulting training instances can be substantially larger than the number of original source documents. Smaller pilot runs use the same construction logic with reduced source counts.

Sequence construction. All examples are tokenized with maximum sequence length 512. Raw-text examples are treated as standard next-token prediction data: an end-of-sequence token is appended, the token stream is chunked into contiguous sequences of length 512, and the learner predicts every token in the chunk. Very short chunks are discarded.

High-quality instruction examples are converted to a shared instruction-response format. Each example contains an instruction, an optional input field, and a response. The learner loss is masked on the prompt tokens and applied only to the response tokens. This formatting is used for MetaMathQA, Magicoder, and Alpaca so that the three high-quality domains share a common surface structure.

Feedback data. The feedback batches use high-quality instruction-format data only. We use MetaMathQA for math feedback, Magicoder OSS-Instruct-75K for code feedback, and Alpaca for general instruction feedback. These feedback pools are balanced equally across the three domains. For a feedback task $j ,$ let $g _ { j }$ denote the evaluator gradient computed from the corresponding feedback batch. The multitask value signal is

$$
g _ {\text { down }} = \sum_ {j = 1} ^ {J} \omega_ {j} g _ {j}, \tag {36}
$$

where $\omega _ { j }$ controls the relative weight of task $j .$ In the reported multitask run, feedback batches are domain-balanced; the resulting combined gradient trains the task designer to construct pretraining updates that align with multiple downstream-relevant directions.

Separation of roles. The same dataset source can play different roles depending on how it is used. Examples in the continued-pretraining mixture are learner data: they define next-token prediction updates. Feedback examples are evaluator data: they define $g _ { j }$ for the task designer. They are not inserted as learner updates during the meta step. Thus the multitask experiment preserves the indirect-feedback channel: the learner is trained on the continued-pretraining stream, while the feedback sets train only the task designer.

Pilot mixtures. We tested two earlier data constructions before using the final raw/high-quality mixture. The first was a MetaMathQA-heavy blend with small code and MMLU-style components. This made the math pretraining stream too close to GSM8K/MATH-style evaluation and required substantial repetition of small code subsets. The second was an equal-weight instruction-only mixture using filtered math instruction data, Magicoder, and Alpaca. This removed the most obvious over-repetition issue, but the setting remained too clean: both the baseline and V-pretraining could learn strongly from direct instruction-format continuation alone. These pilot results motivated the final mixture, which combines broad raw-domain coverage with a smaller amount of high-quality instruction data.

Evaluation. The multitask run is evaluated on held-out benchmarks that probe the three target capability domains: GSM8K for mathematical reasoning, MMBP/MBPP-style code generation, and MMLU for broad knowledge and instruction following. These benchmarks are used for reporting downstream performance, not as learner-update targets. The purpose of the multitask experiment is not to show that all metrics improve uniformly, but to test whether a combined feedback gradient can steer one continued-pretraining run toward several capabilities under a fixed compute budget.

## F Main Language Experiment Setup

Purpose. The main language experiments test whether a small downstream verifier can steer continued language-model pretraining without directly training the learner on the verifier examples. The target capability is mathematical reasoning. The learner is continued on a math-oriented pretraining stream, while a small GSM8K feedback set is used only to train the task designer through the gradient-alignment objective. The reported metric is GSM8K test Pass@1.

## F.1 Single-task math continued pretraining

Learners. The main single-task language results use Qwen-family [72, 54] causal language models as learners. The reported runs include Qwen1.5 learners at multiple scales and a Qwen2.5-0.5B learner. Unless otherwise stated, each method starts from the same initial checkpoint for a given model size and uses the same learner optimizer, learning-rate schedule, numerical precision, sequence length, and wall-clock training budget.

Unlabeled pretraining stream. The single-task math runs use NuminaMath-CoT [35] as the continued-pretraining stream. Each example is formatted as a problem followed by a solution. Training uses causal language modeling on the solution span: prompt tokens are masked in the loss, and the learner predicts only answer or solution tokens. Both the baseline and V-pretraining use the same pretraining stream and the same loss mask.

Baseline. The baseline is standard continued next-token pretraining on the math stream. For a token position t with true next token $y _ { t }$ , the learner is trained with a one-hot target $\delta _ { y _ { t } }$ :

$$
L _ {\text { base }} (\theta) = \frac {1}{| B ^ {u} |} \sum_ {x \in B ^ {u}} \frac {1}{| I (x) |} \sum_ {t \in I (x)} \mathrm{CE} (\delta_ {y _ {t}}, p _ {\theta} (\cdot \mid x _ {<   t})) \tag {37}
$$

where $B ^ { u }$ is an unlabeled pretraining batch, CE denotes the cross-entropy loss, and $I ( x )$ denotes the solution-token positions included in the learner loss.

Feedback set. The feedback set contains 1,024 GSM8K training examples. These examples define the downstream evaluator loss used to compute ${ \mathit { g } } _ { \mathrm { d o w n } } ,$ , but they are not inserted into the learner’s pretraining stream and are not used as supervised learner updates. For a feedback batch $B ^ { f }$ , the evaluator loss is

$$
L _ {\text { down }} (\theta ; B ^ {f}) = \frac {1}{| B ^ {f} |} \sum_ {(u, a) \in B ^ {f}} \frac {1}{| A (u , a) |} \sum_ {t \in A (u, a)} - \log p _ {\theta} (a _ {t} \mid u, a _ {<   t}), \tag {38}
$$

where u is the problem statement, a is the verified solution, and $A ( u , a )$ denotes answer-token positions. The resulting gradient $g _ { \mathrm { d o w n } } = \nabla _ { \theta } L _ { \mathrm { d o w n } } ( \theta ; B ^ { f } )$ is detached and used only as an evaluator vector for the task designer.

V-pretraining objective. V-pretraining keeps the same text stream and causal-LM learner but replaces the fixed one-hot next-token target with a designer-shaped soft target over a small candidate set. For each eligible token position t, we form

$$
C _ {t} = \left\{y _ {t} \right\} \cup \operatorname{TopK} _ {K - 1} (\mathrm{sg} (p _ {\theta} (\cdot \mid x _ {<   t})) \setminus \left\{y _ {t} \right\}), \tag {39}
$$

so the true next token is always included and the remaining candidates are the learner’s highprobability alternatives. The task designer outputs a distribution $r _ { \phi , t } \in \Delta ( C _ { t } )$ and a token-dependent mixing coefficient $\alpha _ { \phi , t } \in [ 0 , \alpha _ { \mathrm { m a x } } ]$ . The learner target is

$$
q _ {\phi , t} (v) = \left(1 - \alpha_ {\phi , t}\right) \mathbf {1} [ v = y _ {t} ] + \alpha_ {\phi , t} r _ {\phi , t} (v), \quad v \in C _ {t}, \tag {40}
$$

and $q _ { \phi , t } ( v ) = 0$ for v $\notin C _ { t }$ . Setting $\alpha _ { \phi , t } = 0$ recovers the baseline one-hot objective.

During a meta update (designer update), the designer is trained to make the induced pretraining gradient align with the downstream evaluator gradient. Let

$$
g _ {\text { pre,S }} = \nabla_ {\theta_ {S}} L _ {\text { pre }} ^ {\text { LM }} (\theta ; \phi , B ^ {u}), \quad g _ {\text { down,S }} = \text { sg } \left(\nabla_ {\theta_ {S}} L _ {\text { down }} (\theta ; B ^ {f})\right), \tag {41}
$$

where S is the subset of learner parameters used for alignment. The implementation maximizes gradient alignment, using either the dot product from the main derivation or its cosine-normalized version:

$$
L _ {\text { meta }} (\phi) = - \operatorname{Align} (g _ {\text { down,S }}, g _ {\text { pre,S }}). \tag {42}
$$

After the designer update, the soft targets are recomputed and detached. The learner update is then an ordinary causal-LM pretraining update:

$$
g ^ {\text { learn }} = \nabla_ {\theta} L _ {\text { pre }} ^ {\text { LM }} (\theta ; \text { sg } (q _ {\phi}), B ^ {u}), \quad \theta^ {+} = \theta - \eta g ^ {\text { learn }}. \tag {43}
$$

Thus GSM8K feedback affects the learner only by changing the self-supervised next-token targets on NuminaMath-CoT examples. No GSM8K supervised gradient is applied to the learner in our main experiments (Table 1).

Evaluation. We evaluate on the GSM8K test set using the same decoding and answer-extraction protocol for all methods in a comparison. Main results report Pass@1. Additional Pass@k, feedbackcoverage, and token-efficiency diagnostics are reported in separate appendix sections.

## F.2 Language implementation branches

We used two related language implementations for different experiments. They share the same conceptual student–designer structure, but differ in learner parameterization, data regime, and alignment parameters. We report them separately to avoid conflating parameter-efficient adaptation with full-parameter continued pretraining.

LoRA reference implementation. The compact reference implementation uses Qwen2.5-0.5B as the learner and trains LoRA adapters rather than all model parameters. The LoRA configuration is

$$
r = 8, \quad \alpha_ {\mathrm{LoRA}} = 1 6, \quad \text { dropout } = 0. 0 5.
$$

Adapters are applied to the standard attention and MLP projections: q\_proj, k\_proj, v\_proj, o\_proj, gate\_proj, up\_proj, and down\_proj. This branch is therefore a parameter-efficient language adaptation setting.

The LoRA reference branch uses a multitask instruction-style pretraining stream with MathInstruct, Magicoder OSS-Instruct-75K, and Alpaca. MathInstruct is filtered to remove sources derived from GSM8K and MATH. All examples are converted to a shared causal-LM format with prompt masking, so only answer tokens contribute to the learner loss.

The task designer is a compact decoder-only TopKAugmentor. It scores only the student learner’s top-K next-token candidates rather than the full vocabulary. It outputs both the top-K distribution $r _ { \phi , t }$ and the smoothing gate $\alpha _ { \phi , t }$ . Gradient alignment is computed on LoRA parameters from the last two transformer layers. The default settings in this branch are:

<table><tr><td>student learning rate</td><td> $2 \times 10^{-4}$ ,</td></tr><tr><td>designer learning rate</td><td> $1 \times 10^{-4}$ ,</td></tr><tr><td>batch size</td><td>4,</td></tr><tr><td>gradient accumulation</td><td>8,</td></tr><tr><td>maximum steps</td><td>2000,</td></tr><tr><td>pretraining sequence length</td><td>1024,</td></tr><tr><td>downstream sequence length</td><td>512,</td></tr><tr><td>meta sequence length</td><td>256,</td></tr><tr><td>K</td><td>64,</td></tr><tr><td> $\alpha_{\text{max}}$ </td><td>0.5,</td></tr><tr><td>meta-update frequency</td><td>every 8 learner steps,</td></tr><tr><td>alignment subset</td><td>last 2 LoRA layers.</td></tr></table>

Full-parameter implementation. The full-parameter implementation uses Qwen2.5-0.5B with all learner parameters trainable. This branch is a direct full continued-pretraining implementation of the same feedback-controlled task-design idea. The baseline trains the learner with ordinary causal language modeling on the selected pretraining stream. Raw text uses standard next-token prediction, while instruction-format examples mask prompt tokens and train only on response tokens.

The default full-parameter baseline uses AdamW with bf16 learner precision and the following settings:

<table><tr><td>learning rate</td><td> $1.5 \times 10^{-5}$ ,</td></tr><tr><td>batch size</td><td>4,</td></tr><tr><td>gradient accumulation</td><td>32,</td></tr><tr><td>effective batch size</td><td>128,</td></tr><tr><td>epochs</td><td>2,</td></tr><tr><td>warmup ratio</td><td>0.03,</td></tr><tr><td>sequence length</td><td>512,</td></tr><tr><td>weight decay</td><td>0,</td></tr><tr><td>max gradient norm</td><td>1.0,</td></tr><tr><td>learner precision</td><td>bf16.</td></tr></table>

The corresponding V-pretraining version uses the same learner and pretraining stream, but adds a small TopKAugmentor. Unless otherwise stated, the augmentor has hidden size 256, 6 layers, 4 attention heads, top-K = 64, and $\alpha _ { \mathrm { m a x } } = 0 . 5$ . The augmentor is trained in fp32 even when the learner is trained in bf16. Alignment is computed over a selected subset of full learner parameters, usually the last few transformer layers, controlled by align\_last\_n\_layers.

Domain-matched alignment for multitask language runs. In multitask language runs, we compute gradient alignment within matching domains rather than pooling all tasks into a single undifferentiated feedback batch. For a domain d ∈ {math, code, general}, the method pairs a same-domain pretraining batch with a same-domain feedback batch:

$$
B _ {d} ^ {u} \leftrightarrow B _ {d} ^ {f}. \tag {44}
$$

It then computes

$$
g _ {\text { pre }, d} = \nabla_ {\theta_ {S}} L _ {\text { pre }} (\theta ; \phi , B _ {d} ^ {u}), \quad g _ {\text { down }, d} = \operatorname{sg} \left(\nabla_ {\theta_ {S}} L _ {\text { down }} (\theta ; B _ {d} ^ {f})\right), \tag {45}
$$

and trains the designer with the mean alignment objective

$$
L _ {\text { meta }} (\phi) = - \frac {1}{| \mathcal {D} |} \sum_ {d \in \mathcal {D}} \operatorname{Align} (g _ {\text { down }, d}, g _ {\text { pre }, d}). \tag {46}
$$

This domain-matched design avoids a failure mode in which a single combined feedback gradient pulls the designer toward a direction inconsistent with the current pretraining domain.

Feedback formatting variants. For multitask language experiments, the default feedback pools are MetaMathQA for math, Magicoder for code, and Alpaca for general instruction following. These feedback examples are formatted consistently with the instruction-style pretraining examples. We also implemented an evaluation-aligned feedback variant, in which feedback examples are reformatted to more closely match downstream benchmark prompts: math feedback is written in a GSM8K-style format, code feedback in an MBPP-style format with markdown fences removed, and knowledge feedback in an MMLU multiple-choice format. This variant is treated as a separate experimental branch rather than the universal default.

Separation of learner and feedback data. Across both implementation branches, the key invariant is unchanged. Pretraining examples define learner updates. Feedback examples define evaluator gradients for the task designer. Even when a feedback dataset has the same source name as a highquality pretraining dataset, its role is different: feedback batches are not inserted as learner updates during the meta step. The only learner update remains a causal-LM loss on the current pretraining batch with detached designer-shaped targets.

## G Main Vision Experiment Setup

Purpose. The vision experiments test whether downstream dense-prediction feedback can steer continued self-supervised pretraining without directly training the visual backbone on dense labels. The learner is a DINOv3 vision transformer [67] continued on unlabeled ImageNet-1K images with a DINO-style self-supervised objective. The feedback tasks are ADE20K semantic segmentation [81] and NYUv2 depth estimation [44]. Their labels are used only to train the task designer through the value objective; the backbone learner is still updated only by a self-supervised loss on unlabeled images.

## G.1 Baseline continued DINO pretraining

Learners and unlabeled data. The main vision results use pretrained DINOv3 ViT backbones, including ViT-B and ViT-L (we discuss the I-JEPA setup later). Each run starts from the same pretrained checkpoint for a given backbone size and continues self-supervised training on ImageNet-1K. The same unlabeled image stream, crop configuration, learner optimizer, learning-rate schedule, teacher momentum schedule, numerical precision, and training budget are used for the baseline and V-pretraining comparison.

DINO-style SSL objective. The baseline is standard continued DINO-style self-supervised learning. The student consists of the DINOv3 backbone and a projection head. The teacher is an exponentialmoving-average copy of the student. For each image, the fixed augmentation pipeline samples multiple views: two global crops and six local crops by default, with color jitter, grayscale augmentation, Gaussian blur, and solarization [9, 10]. The student is trained to match teacher predictions across non-matching views, with temperature scaling and a running teacher-output center.

Let $V _ { 0 } ( x ) = \{ v _ { 1 } ^ { 0 } , . . . , v _ { M } ^ { 0 } \}$ denote the views produced by the fixed DINO augmentation pipeline for an unlabeled image x. The baseline minimizes

$$
L _ {\mathrm{ssl}} ^ {0} (\theta ; B ^ {u}) = \frac {1}{| B ^ {u} |} \sum_ {x \in B ^ {u}} \ell_ {\mathrm{DINO}} \big (\theta , \bar {\theta}; V _ {0} (x) \big), \tag {47}
$$

where θ are the student parameters and $\bar { \theta }$ are the EMA teacher parameters. The task construction rule c is fixed before training: the views are stochastic, but their sampling distribution and augmentation recipe do not depend on downstream feedback.

Baseline learner update. The baseline learner update is therefore

$$
g _ {t} ^ {\text { base }} = \nabla_ {\theta} L _ {\mathrm{ssl}} ^ {0} (\theta_ {t}; B _ {t} ^ {u}), \quad \theta_ {t + 1} = \theta_ {t} - \eta_ {t} g _ {t} ^ {\text { base }}. \tag {48}
$$

No ADE20K or NYUv2 labels are used in the baseline continued-pretraining update.

## G.2 V-pretraining with downstream-aware learned augmentations

Control surface. V-pretraining keeps the DINO learner, SSL objective, unlabeled stream, and optimizer fixed, but replaces part of the fixed view-construction rule with a learned instance-wise augmentation module. The designer does not predict segmentation masks or depth maps. Instead, it modifies the self-supervised views used by DINO so that the resulting SSL gradient better aligns with dense-task feedback gradients.

Augmentor parameterization. The task designer is an image augmentor $a _ { \phi }$ that predicts a soft spatial mask $\bar { \mu } _ { \phi } ( v ) \in [ 0 , 1 ] ^ { H \times W }$ for selected SSL crops. Two augmentor architectures are supported: a small convolutional U-Net [63] mask generator and a small dit-based [51] patchwise mask generator. In the default configuration, the augmentor is applied only to the two global SSL crops, while the local crops remain generated by the standard DINO pipeline.

Given a base crop $v ,$ the learned view is formed by softly mixing the original crop with a blurred version:

$$
\tilde {v} = \mu_ {\phi} (v) \odot v + (1 - \mu_ {\phi} (v)) \odot \operatorname{Blur} (v). \tag {49}
$$

Large mask values preserve the original pixels; small values suppress pixels by replacing them with a blurred background. For an image x, let $V _ { \phi } ( x )$ denote the resulting set of views after replacing the controlled global crops with learned views. The SSL loss under the designer is

$$
L _ {\mathrm{ssl}} (\theta ; \phi , B ^ {u}) = \frac {1}{| B ^ {u} |} \sum_ {x \in B ^ {u}} \ell_ {\mathrm{DINO}} (\theta , \bar {\theta}; V _ {\phi} (x)). \tag {50}
$$

Dense feedback tasks. The feedback signal comes from small labeled subsets of ADE20K and NYUv2. For the default feedback construction, we use ADE20K / SceneParse150 with 2,000 labeled training samples and 512 meta samples, and NYUv2 with 512 labeled training samples and 128 meta samples. The labeled training samples are used to fit lightweight downstream evaluator heads, and the meta samples are used to compute downstream gradients for the task designer.

The evaluator heads are:

$$
H _ {\text {seg}} \quad \text {for ADE20K semantic segmentation,} \quad H _ {\text {dep}} \quad \text {for NYUv2 depth estimation.}
$$

The segmentation head is trained with pixelwise cross-entropy. The depth head is trained with an L1 loss on valid depth pixels.

Meta step. A V-pretraining meta step (task designer update step) has three stages.

First, the downstream evaluator heads are updated on small labeled training batches using frozen backbone features. This adapts the lightweight heads to the current representation without applying supervised dense-task updates to the backbone.

Second, on separate labeled meta batches, we compute a downstream evaluator loss

$$
L _ {\text { down }} ^ {\text { vis }} (\theta ; B ^ {f}) = \alpha_ {\text { seg }} L _ {\text { seg }} \left(H _ {\text { seg }} (F _ {\theta} (x)), y _ {\text { seg }}\right) + \alpha_ {\text { dep }} L _ {\text { dep }} \left(H _ {\text { dep }} (F _ {\theta} (x)), y _ {\text { dep }}\right), \tag {51}
$$

with terms omitted when the corresponding feedback task is not used. Here $F _ { \theta }$ is the visual backbone, and $\alpha _ { \mathrm { s e g } } , \alpha _ { \mathrm { d e p } }$ control the relative value placed on segmentation and depth. We compute the downstream gradient with respect to a selected subset S of backbone parameters:

$$
g _ {\text { down }, \mathrm{S}} ^ {\text { vis }} = \operatorname{sg} \left(\nabla_ {\theta_ {S}} L _ {\text { down }} ^ {\text { vis }} (\theta ; B ^ {f})\right). \tag {52}
$$

In practice, S is chosen as the last K transformer blocks together with final normalization layers. This reduces the cost of the mixed second-order gradient while keeping the alignment signal tied to the representation layers used by dense prediction.

Third, a separate unlabeled ImageNet batch is passed through the learned augmentor, the DINO loss is recomputed under the learned views, and the SSL gradient is computed on the same parameter subset:

$$
g _ {\mathrm{ssl}, \mathrm{S}} ^ {\text { vis }} = \nabla_ {\theta_ {S}} L _ {\mathrm{ssl}} (\theta ; \phi , B ^ {u}). \tag {53}
$$

The augmentor is trained to maximize alignment between the downstream dense-task gradient and the SSL gradient:

$$
L _ {\text { meta }} ^ {\text { vis }} (\phi) = - \left\langle g _ {\text { down }, \mathrm{S}} ^ {\text { vis }}, g _ {\text { ssl }, \mathrm{S}} ^ {\text { vis }} \right\rangle + \lambda_ {\text { spars }} R _ {\text { spars }} (\mu_ {\phi}) + \lambda_ {\text { tv }} R _ {\text { tv }} (\mu_ {\phi}). \tag {54}
$$

The sparsity regularizer keeps the mean mask value near a target keep ratio, and the total-variation regularizer encourages spatially smooth masks. These regularizers keep the learned augmentation close to plausible SSL view construction rather than arbitrary image corruption.

$L _ { \mathrm { m e t a } } ^ { \mathrm { v i s } } .$ . After the augmentor update, the views are recomputed and detached. The backbone learner is then updated by an ordinary DINO SSL loss:

$$
g _ {t} ^ {\text { learn }} = \nabla_ {\theta} L _ {\mathrm{ssl}} \left(\theta_ {t}; \operatorname{sg} (V _ {\phi_ {t}} (B _ {t} ^ {u}))\right), \quad \theta_ {t + 1} = \theta_ {t} - \eta_ {t} g _ {t} ^ {\text { learn }}. \tag {55}
$$

Thus ADE20K and NYUv2 labels never appear as supervised learner targets during continued pretraining. They train only the task designer that constructs SSL views.

Second-order implementation detail. The meta update differentiates through the SSL gradient with respect to selected backbone parameters. This requires second-order differentiation through the backbone. To avoid unsupported double-backward paths in Flash or memory-efficient attention kernels, the implementation uses a math-only scaled dot-product attention backend during the relevant meta-gradient computation.

## G.3 Dense evaluation protocol

ADE20K semantic segmentation. ADE20K evaluation uses the SceneParse150 dataset. Label 0 is remapped to the ignore index 255, and labels 1, . . . , 150 are remapped to $0 , \ldots , 1 4 9$ . Performance is reported as mean intersection-over-union (mIoU).

The evaluation code supports both a small convolutional segmentation decoder and a linear-BN segmentation head. In the sweep comparisons used for the main reported results, the evaluation protocol is fixed to a frozen-backbone linear probe with a linear-BN head. This makes the comparison primarily a test of representation quality rather than downstream fine-tuning sensitivity.

NYUv2 depth estimation. NYUv2 evaluation uses the standard validation split. Depth values are decoded to meters, and metrics are computed on valid pixels. The evaluation protocol enables the standard Eigen crop by default. The main metric reported in the paper is RMSE; the evaluation code also computes AbsRel and $\delta _ { 1 }$ .

As with segmentation, the sweep comparisons use a frozen-backbone linear probing protocol with a linear-BN depth head. Predictions are passed through a softplus nonlinearity to ensure positive depth values.

ImageNet linear evaluation and transfer. We additionally evaluate whether dense-task feedback harms global recognition. For this purpose, the main table reports ImageNet-1K linear accuracy. We also report instance-retrieval transfer on Revisited Oxford and Revisited Paris. These tasks are not used as feedback during V-pretraining and are used to check whether dense feedback collapses the representation onto segmentation or depth.

Evaluation hyperparameters. All selected checkpoints are evaluated at the last pretraining checkpoint (eval\_pretrain\_steps=last) with a frozen-backbone linear probing protocol (eval\_ft\_mode=linear) and a linear-BN head. Evaluation uses 20k optimization steps, batch size 8, bf16 precision, 1k warmup steps, weight decay 0.05, and gradient clipping at 1.0. ADE20K evaluation uses image size 512 and head learning rate $5 \times 1 0 ^ { - 4 }$ . NYUv2 depth evaluation uses an L2 depth loss, head learning rate $1 0 ^ { - 3 }$ , resize resolution 480 × 640, valid depth range $[ 1 0 ^ { - 3 }$ , 10] meters, and the standard Eigen crop.

## G.4 Hyperparameter selection

Fixed-budget sweep protocol. Vision hyperparameters are selected with fixed-budget pretraining sweeps. Each sweep trial performs 20k steps of continued DINO pretraining and then evaluates one or more saved checkpoints, with the default configuration evaluating the last checkpoint. ADE20Koriented sweeps optimize validation mIoU, while NYUv2-oriented sweeps optimize validation RMSE.

The main swept hyperparameters are:

<table><tr><td>learner learning rate</td><td> $\eta,$ </td></tr><tr><td>meta-update frequency</td><td> $r,$ </td></tr><tr><td>meta SSL batch size</td><td> $|B_{\text{meta}}^{u}|,$ </td></tr><tr><td>aligned backbone blocks</td><td> $K,$ </td></tr><tr><td>augmentor architecture</td><td>U-Net or tiny DiT,</td></tr><tr><td>augmentor learning rate</td><td> $\eta_{\phi},$ </td></tr><tr><td>target mask keep ratio</td><td> $\kappa,$ </td></tr><tr><td>sparsity regularization</td><td> $\lambda_{\text{spars}},$ </td></tr><tr><td>total-variation regularization</td><td> $\lambda_{\text{tv}},$ </td></tr><tr><td>segmentation feedback weight</td><td> $\alpha_{\text{seg}},$ </td></tr><tr><td>depth feedback weight</td><td> $\alpha_{\text{dep}}.$ </td></tr></table>

Backbone-scale adjustments. The sweep spaces are adjusted for backbone size to fit memory. For ViT-B, the pretraining batch size is larger; for ViT-L, the pretraining batch size and meta-SSL batch-size search space are reduced. In the larger depth sweep, the augmentor architecture is fixed to the U-Net mask generator to reduce search cost. These adjustments affect the feasible training configuration, not the definition of the feedback channel.

<table><tr><td>Target</td><td>Backbone</td><td>LR</td><td>Batch</td><td>Meta freq.</td><td>Meta SSL batch</td><td>Aligned blocks</td><td>Augmentor</td><td>Aug. LR</td><td>Keep</td><td> $\lambda_{\text{spars}}$ </td><td> $\lambda_{\text{tv}}$ </td><td> $\alpha_{\text{seg}}$ </td><td> $\alpha_{\text{dep}}$ </td></tr><tr><td>ADE20K</td><td>ViT-B</td><td> $4.15\times 10^{-6}$ </td><td>256</td><td>4</td><td>32</td><td>2</td><td>U-Net</td><td> $1.54\times 10^{-4}$ </td><td>0.6</td><td>0.131</td><td>0.144</td><td>1</td><td>8</td></tr><tr><td>ADE20K</td><td>ViT-L</td><td> $4.50\times 10^{-5}$ </td><td>50</td><td>4</td><td>16</td><td>4</td><td>U-Net</td><td> $2.17\times 10^{-4}$ </td><td>0.5</td><td>0.196</td><td>0.0399</td><td>4</td><td>1</td></tr><tr><td>NYUv2</td><td>ViT-B</td><td> $8.58\times 10^{-6}$ </td><td>256</td><td>8</td><td>64</td><td>3</td><td>SiT</td><td> $2.21\times 10^{-4}$ </td><td>0.4</td><td>0.111</td><td>0.0417</td><td>16</td><td>2</td></tr><tr><td>NYUv2</td><td>ViT-L</td><td> $1.94\times 10^{-6}$ </td><td>50</td><td>2</td><td>32</td><td>4</td><td>U-Net</td><td> $8.03\times 10^{-4}$ </td><td>0.4</td><td>0.401</td><td>0.159</td><td>32</td><td>16</td></tr></table>

Table 11: Selected V-pretraining hyperparameters for the main DINOv3 vision runs. Batch is the learner SSL batch size. Meta freq. is the number of learner steps between augmentor updates. Meta SSL batch is the unlabeled ImageNet batch used to compute the SSL gradient in the value objective. Aligned blocks denotes the number of final ViT blocks used for gradient alignment. The target column denotes the sweep selection metric, not the only feedback task used by the value objective.

Interpretation of vision tuning. The vision results should be interpreted under this stated tuning protocol: standard continued DINO pretraining and V-pretraining are compared with the same initial backbone, unlabeled ImageNet stream, SSL objective family, and evaluation protocol, while V-pretraining additionally pays the cost of learned view construction and periodic value-gradient updates. The purpose of the experiment is not to introduce a new dense-prediction fine-tuning method, but to test whether dense feedback can improve the self-supervised pretraining updates that shape the backbone representation.

Selected V-pretraining configurations. Table 11 reports the selected V-pretraining configurations used for the main DINOv3 vision results. The target column denotes the validation metric used to select the run from the fixed-budget sweep; it does not mean that only that feedback task is present in the meta objective. When both dense feedback loaders are enabled, the downstream evaluator gradient is formed from the weighted segmentation and depth losses with weights $\alpha _ { \mathrm { s e g } }$ and $\alpha _ { \mathrm { d e p } }$ .

All selected runs use 20k continued-pretraining steps, bf16 mixed precision, AdamW, weight decay 0.04, gradient clipping at 1.0, 1k warmup steps, DINO output dimension 8192, student temperature 0.1, teacher temperature 0.04, teacher momentum base 0.996, center momentum 0.9, two global crops of size 224, six local crops of size 96, and learned augmentation applied only to the global crops. The unlabeled stream is ImageNet-1K, and the feedback pools use 2,000 ADE20K training samples with 512 ADE20K meta samples and 512 NYUv2 training samples with 128 NYUv2 meta samples.

## H Declarations and Impacts

Use of generative AI tools. We used ChatGPT Image to draft the illustrative schematic in Figure 1 from author-written prompts. The figure contains no experimental data, generated results, or method output. We also used ChatGPT to polish some parts of the writing. All equations, labels, and scientific content were manually checked by the authors. Our use of AI tools follow the NeurIPS guidelines.

Broader Impacts. As V-pretraining fundamentally enhances the capabilities of machine learning models, it inherently carries a dual-use nature. On the positive side, our approach can significantly benefit downstream applications such as creative content generation and advanced information retrieval. Conversely, we acknowledge the potential risks of malicious exploitation, particularly regarding the generation of deceptive, fake, or misleading content.

## I Compute Resources and Total Compute

Compute workers. All training experiments were run on internal GPU clusters. The main training workers were NVIDIA H100 80GB and NVIDIA H200 NVL GPUs. Auxiliary evaluation, small diagnostic runs, and some post-training comparisons also used RTX 6000 Ada / RTX A6000-class GPUs with 48GB memory. We report compute in accelerator-hours, counting one hour on one GPU as one accelerator-hour. Most reported runs were single-worker training jobs; when a node exposed multiple GPUs, we count only the GPUs used by the corresponding training process.

Training jobs used CPU workers for data loading. Vision pretraining jobs used 12 dataloader workers in the selected configurations. Slurm allocations typically provided 12–16 CPU cores per GPU worker and 80–142GB of allocated host memory, with larger nodes providing substantially more physical RAM. Datasets and checkpoints were stored on shared cluster storage; individual runs wrote checkpoints, logs, and evaluation outputs, typically requiring tens of GB per run. The full project required on the order of 0.5–1TB of shared storage across datasets, checkpoints, logs, sweeps, and intermediate evaluation outputs.

<table><tr><td>Experiment family</td><td>Main worker</td><td>Approx. compute per run</td><td>Memory used</td><td>Notes</td></tr><tr><td>Qwen 0.5B language continued pretraining</td><td>1 H100/H200</td><td>2–6 GPU-hours</td><td>30–40GB</td><td>Baseline or V-pretraining run; GSM8K feedback for V-pretraining.</td></tr><tr><td>Qwen 4B language continued pretraining</td><td>1–2 H100/H200</td><td>10–24 GPU-hours</td><td>60–140GB</td><td>Used for main GSM8K result, ablations, and decontamination controls.</td></tr><tr><td>Qwen 7B language continued pretraining</td><td>1–2 H100/H200</td><td>16–36 GPU-hours</td><td>80–140GB</td><td>Used for scaling runs.</td></tr><tr><td>Language multitask run</td><td>1 H100/H200</td><td>4–12 GPU-hours</td><td>30–50GB</td><td>Qwen2.5-0.5B multitask pretraining with math/code/general feedback.</td></tr><tr><td>DINOv3 ViT-B continued SSL</td><td>1 H100/H200</td><td>1–3 GPU-days</td><td>50–80GB</td><td>20k-step continued pretraining plus dense linear-probe evaluation.</td></tr><tr><td>DINOv3 ViT-L continued SSL</td><td>1 H100/H200</td><td>1–4 GPU-days</td><td>80–140GB</td><td>20k-step continued pretraining plus dense linear-probe evaluation.</td></tr><tr><td>Dense evaluation / retrieval / linear probes</td><td>H100/H200 or RTX 6000 Ada</td><td>0.5–6 GPU-hours</td><td>20–48GB</td><td>ADE20K, NYUv2, ImageNet linear, and retrieval evaluations.</td></tr><tr><td>Direct-feedback diagnostic</td><td>H100 and RTX 6000 Ada</td><td>10 min–16 GPU-hours</td><td>20–80GB</td><td>Small V-pretraining/baseline diagnostic and SFT+GRPO comparison.</td></tr></table>

Table 12: Approximate compute for individual reported experimental runs. GPU-hours denote accelerator-hours. Ranges reflect differences in model size, device type, and whether evaluation is included in the same job. The reported main comparisons are wall-clock matched within each experiment family, so V-pretraining is charged for its task-designer and value-gradient overhead.

Per-run compute. Table 12 summarizes the approximate compute for the reported experimental runs. The numbers are estimates from job logs and observed throughput. They are intended to document the scale of the experiments rather than serve as exact hardware benchmarks, since wallclock time varies with device type, dataloader throughput, cluster load, and whether evaluation is run in the same job.

Representative measured overhead. For the current language V-pretraining configuration, the task designer is a small 256-hidden-dimensional transformer with 6 layers and 8 heads, aligning the last two learner layers and performing a meta update every 8 learner optimizer steps. In a representative current run, the standard baseline used microbatch size 4, gradient accumulation 32, and sequence length 512, for $4 \times 3 2 \times 5 1 2 = 6 5 { , } 5 3 6$ tokens per optimizer step. The corresponding V-pretraining run used microbatch size 4, gradient accumulation 20, and sequence length 512, for $4 \times 2 0 \times 5 \bar { 1 2 } = 4 0 { , } 9 6 0$ tokens per optimizer step. The baseline achieved approximately 26.7k tokens/s with peak GPU memory 31,302 MiB, while V-pretraining achieved approximately 24.2k tokens/s with peak GPU memory 36,440 MiB. Thus the current V-pretraining implementation reduces token throughput by about 9% and increases peak memory by about 16% in this representative language setting. The shorter V-pretraining optimizer-step time is not evidence of lower cost, since each optimizer step processes fewer tokens.