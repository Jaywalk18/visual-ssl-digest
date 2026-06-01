# Continual Distillation of Teachers from Different Domains

Nicolas Michel1,4 Maorong Wang2 Jiangpeng He3,† Toshihiko Yamasaki1,† 1The University of Tokyo 2National Institute of Informatics 3Indiana University Bloomington 4Japanese-French Laboratory for Informatics, CNRS

{nicolas, yamasaki}@cvm.t.u-tokyo.ac.jp, maorong@nii.ac.jp, jhe2@iu.edu

# Abstract

Deep learning models continue to scale, with some requiring more storage than many large-scale datasets. Thus, we introduce a new paradigm: Continual Distillation (CD), where a student learns sequentially from a stream of teacher models without retaining access to earlier teachers. CD faces two challenges: teacher training data is unavailable, and teachers have varying expertise. We show that external unlabeled data enables Unseen Knowledge Transfer (UKT), allowing the student to acquire information from domains not present in the training data, while known to the teacher. We also show that sequential distillation causes Unseen Knowledge Forgetting (UKF) when transferred knowledge is lost after training on later teachers. To better trade off between UKT and UKF, we propose Self External Data Distillation (SE2D), a method that preserves logits on external data to stabilize learning across heterogeneous teachers. Experiments on multiple benchmarks show that SE2D reduces UKF and improves cross-domain generalization. The code and implementation for this work are publicly available at: https://github.com/Nicolas1203/ continual\_distillation.

# 1. Introduction

Over the last decade, deep learning models have reached unprecedented scales, creating a growing need for computation-efficient strategies. Consequently, Continual Learning (CL) [8, 24, 35] has become a major branch of contemporary deep learning research. The main idea is straightforward: a model is trained on a sequence of datasets where previous data becomes unavailable over time. The rationale is that re-training on both past and current data as new data arrives can be computationally expensive or require substantial storage. In this context, data access is limited during training.

With the recent adoption of Foundation Models (FMs) [1, 9, 11, 21, 26] as the backbone of modern deep learning, we propose a new paradigm, Continual Distillation (CD), tailored to FMs, illustrated in Figure 1. In CD, instead of learning from a sequence of datasets, we propose to learn from a sequence of models trained on different datasets. Specifically, a single student model learns sequentially from multiple teacher models without retaining access to earlier ones. Similar to training data in CL, new FMs are regularly introduced over time, and storing them is cumbersome since they can rapidly require more storage than large-scale datasets. For instance, storing 10B parameters requires approximately 38GB [7], while FMs often exceed 100B parameters. Even accessing FMs through restricted APIs presents serious limitations, as previous versions may become unavailable after updates. Our CD setup reflects a realistic scenario in which one aims to leverage the everevolving stream of FMs to train smaller and more specialized models through distillation. In analogy to CL, where a single model benefits from a continuous stream of data, in CD, a single student model benefits from a continuous stream of teacher models.

However, CD introduces two main challenges. First, the original training data of a foundation model is typically unavailable, undisclosed, or prohibitively large to reuse [2]. Thus, choosing distillation data is critical, and such data can potentially emerge from a domain unknown to the teacher [36]. Second, each teacher generally exhibits distinct abilities and performance; for instance, one may excel at recognizing animals, while another may specialize in distinguishing insects.

In this work, we focus on the case of Continual Distillation, where, analogous to Domain Incremental Learning, teachers have been trained on a domain incremental set of datasets. However, we assume that teachers share partially overlapping domains. We believe this assumption to be realistic, as, for example, it is safe to assume that all FMs have been trained on ImageNet. Therefore, we make the following key assumptions: (1) each teacher is trained on data from different domains while sharing a specific domain; (2) the training data for distillation is fixed; and (3) the training data is unlabeled. The ultimate objective is to obtain a student model that achieves high performance on every domain known by at least one teacher, without having access to all teacher training data or labels. In this context, we find that training data can be decomposed into two categories: External Data (ED), unknown to all teachers, and Internal Data (ID), known to all teachers. Importantly, we discover that ED enables the transfer of knowledge from domains unseen by the student but known by the teacher during distillation, which we refer to as Unseen Knowledge Transfer (UKT). Naturally, we observe that the student inevitably forgets unseen knowledge transferred by previous teachers when learning from future ones. We refer to this phenomenon as Unseen Knowledge Forgetting (UKF). The central problem of CD becomes reaching an optimal UKT-UKF trade-off.

Traditional Domain Incremental Learning   
![](images/1956a7ed893dbfe506bdeaf5617f4a83fde44b21198c65f1848f869665a7b6ce.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["D₁ᵀ"] --> B["Supervised"]
    C["D₂ᵀ"] --> D["Supervised"]
    E["D₃ᵀ"] --> F["Supervised"]
    B --> G["S"]
    D --> G
    F --> G
    style A fill:#cce5ff,stroke:#333
    style C fill:#cce5ff,stroke:#333
    style E fill:#cce5ff,stroke:#333
    style G fill:#e6f7ff,stroke:#333
```
</details>

Continual Distillation   
![](images/982e8a08359c6a281dbf8790bd9ce7da31c39d0edc1eba157f64d2d21a2b439a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["NO ACCESS"] --> B["T₁"]
    C["NO ACCESS"] --> D["T₂"]
    E["NO ACCESS"] --> F["T₃"]
    B --> G["Dˢ"]
    D --> H["Dˢ"]
    F --> I["Dˢ"]
    G --> J["S"]
    H --> K["→"]
    I --> L["→"]
    J --> M["→"]
    K --> N["→"]
    L --> O["→"]
    M --> P["→"]
    N --> Q["→"]
    O --> R["→"]
```
</details>

Figure 1. Overview of the Continual Distillation problem. A student model S learns through distillation from a sequence of teachers $\{ \mathcal { T } _ { 1 } , \mathcal { T } _ { 2 } , \mathcal { T } _ { 3 } \}$ . During distillation, part of the teachers’ training data is unavailable. The objective is for the student to maintain high performance on all domains known to the teachers, but not necessarily introduced to the student.

Experimentally, we observe that while the usage of ED enables UKT, mainstream distillation strategies fail to address UKF, as their primary focus relies on maximizing knowledge transfer only. We therefore propose Self External Data Distillation (SE2D), a CL-inspired method tailored for the CD problem, focusing on preserving logits of ED to maintain performance on domains unseen by the student. SE2D allows positive UKT while maintaining performance in older domains. Our contributions are as follows:

• We introduce the paradigm of Continual Distillation, motivated by the practical challenges that arise when previous teacher models are no longer accessible.   
• We demonstrate that the choice of distillation data is crucial for transferring knowledge to domains unseen by the student, and choose to take advantage of External Data, unknown to the teachers.   
• We identify Unseen Knowledge Forgetting, mitigate it by preserving External Data logits, and validate our approach across various benchmarks.

# 2. Related Work

Continual Learning. Continual Learning (CL) [5, 24] focuses on enabling models to learn from a sequence of tasks while retaining previously acquired knowledge. Traditionally, each CL task is defined by different non-overlapping datasets with unique properties. In Class Incremental Learning (CIL) [10], each task is composed of a unique set of classes. In Domain Incremental Learning (DIL) [3], classes are shared, but input distributions vary. While access and change in data have been widely studied, we are, to the best of our knowledge, the first to consider an alternative scenario where data is fixed but accessible model change over time. In this case, we focus on a situation where the teachers’ training domain differs, not unlike a DIL setup, where each teacher would be trained on a specific task of a DIL setting. Since foundation models become ubiquitous [26], large, and ever-changing, a paradigm shift from ever-evolving data to ever-evolving teachers appears relevant. While Knowledge Distillation is often employed in CL to mitigate forgetting [22, 28, 32], we instead investigate the phenomenon of forgetting within distillation itself, treating it as the central challenge of our study.

Knowledge Distillation. Knowledge Distillation (KD) [15] is a technique that allows a smaller student model to replicate the behavior of a larger, more capable teacher model. Ideally, KD assumes that both the teacher and student are trained on the same dataset, and has been widely used for model compression [29, 39] and transfer learning [38]. When such data is not available, distillation is considered to be data-free [36]. In that context, the objective is usually to reproduce the teacher’s training domain as closely as possible. Thanks to the capability of knowledge transfer from KD, it has become one of the cornerstones for solving the forgetting problem in CL [12, 28]. Nevertheless, the ideal setting (i.e., the students are distilled with the original teachers’ training dataset) of KD rarely holds in CL due to the unavailability of historical data. Consequently, CL methods conduct distillation with new training samples [20], small memory buffer samples [22], generated samples [37], or adversarial samples [14].

![](images/9f96f2cffbc237e4c5ba0bcab514a01ad9e893be498cddc21e40a580109b0323.jpg)

<details>
<summary>text_image</summary>

D₁ᵀ
D₂ᵀ
Dᵢ
Dₑ
Private
Public
</details>

Figure 2. Visualization $\mathcal { D } _ { e }$ and $\mathcal { D } _ { i }$ and different teacher training domains in our experimental setting. $\mathcal { D } _ { i }$ and $\mathcal { D } _ { e }$ are publicly available like ImageNet or Wikipedia. However, some private or restricted data might be included in training, such as medical data.

While prior work has explored multi-teacher knowledge distillation and the use of distillation in continual learning, our setting differs in several key aspects. First, existing multi-teacher distillation methods typically assume simultaneous access to all teachers, whereas we consider a sequential setting where only one teacher is accessible at a time. Second, unlike continual learning approaches that focus on data streams, our formulation assumes a fixed dataset and instead models a stream of evolving teacher models. Finally, in contrast to standard distillation-based continual learning methods, we do not assume access to past data or replay buffers, nor to previously seen teachers. This combination of constraints defines a distinct and practically relevant scenario that, to the best of our knowledge, has not been explicitly studied before. We therefore frame this problem as Continual Distillation, emphasizing the shift from data-centric to model-centric CL.

# 3. Continual Distillation

# 3.1. Generic Definition

We define Continual Distillation (CD) as the process of distilling the knowledge from a sequence of teacher models continuously into one student model, on a fixed dataset. When distilling from one teacher to the student, other teachers are considered unavailable. The distillation process from a given teacher to the student is analogous to a task in standard CL. Formally, given a sequence of teachers $\{ \mathcal { T } _ { 0 } , \mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { N } \}$ , each trained on a dataset $\mathcal { D } _ { t } ^ { \mathcal { T } }$ , the student $s$ is optimized to minimize distillation loss $\mathcal { L } _ { d i s t } .$ , with respect to $\mathcal { T } _ { t }$ on a distillation dataset $\mathcal { D } ^ { s }$ . Importantly, we only consider distillation and no label-dependent loss is considered. In this work, we focus on logits-based distillation as representations are architecture-dependent, computation-intensive, and require access to the entire teacher model. We present the overall procedure in Figure 1. We denote by $\bar { \mathbb { D } } _ { S } ( \mathcal { T } _ { t } , \mathcal { D } ^ { S } )$ the operation of distilling from teacher $\tau$ trained with $\mathcal { D } _ { t } ^ { \mathcal { T } }$ to student $s$ on distillation dataset $\mathcal { D } ^ { s }$ .

![](images/256b9b548b071dad2b886e350231b9b0b3f61a65de9752e3450f6383a7c8c20f.jpg)

<details>
<summary>bar</summary>

| Category             | Domain 0 (D_i) | Domain 1 | Domain 2 | External Domain (D_e) |
|----------------------|----------------|----------|----------|------------------------|
| Internal Data Only   | Di             |          |          |                        |
| Student Performances | Di             |          |          |                        |
| Internal and External Data | Di           | De       | UKT      | De                     |
| Internal and External Data | Di           | De       | UKT      | De                     |
| Internal and External Data | Di           | De       | UKF      | UKT                    |
| Internal and External Data | Di           | De       | UKT      | De                     |
</details>

Figure 3. Illustration of Unseen Knowledge Transfer (UKT) and Unseen Knowledge Forgetting (UKF). When distilling only from $\mathcal { T } _ { 1 }$ on Internal Data $\mathcal { D } _ { i } .$ , only Domain 0 knowledge is transferred. When distilling from the same teacher $\mathcal { T } _ { 1 }$ on $\mathcal { D } _ { e } \cup \mathcal { D } _ { i } ,$ Domains 0 and 1 knowledge is transferred, although the student has never seen Domain 1. However, when continuing the sequence and distilling from $\mathcal { T } _ { 2 }$ , while the student acquires knowledge from Domain $2 ,$ , it forgets part of the knowlege from Domain 1.

# 3.2. Specific Problem Scenario

Traditional KD assumes that the teacher training datasets are available for distillation. In CD, not only are such datasets considered unavailable, but dataset domains might differ from one teacher training to another. In other words, $\mathcal { D } _ { t } ^ { \mathcal { T } } , \mathcal { D } _ { t ^ { \prime } } ^ { \mathcal { T } }$ and $\mathcal { D } ^ { S }$ may cover partially or totally different domains. Therefore, various scenarios can be defined in CD depending on the domain overlap between teachers training data $\mathcal { D } _ { t } ^ { \mathcal { T } }$ and distillation data $\mathcal { D } ^ { \hat { \boldsymbol { s } } }$ . Realistically, when considering FMs, it is safe to assume that part of the training data is shared. Typically, publicly available datasets such as Wikipedia or ImageNet are commonly used for training FMs. However, additional data, exclusive to a specific model, might also be included during training. A visualisation of such a scenario is presented in Figure 2. Formally, we consider the case of partially exclusive teacher domains such that all teachers share a specific domain:

$$
\mathcal {D} _ {t} ^ {\mathcal {T}} \cap \mathcal {D} _ {t ^ {\prime}} ^ {\mathcal {T}} = \mathcal {D} _ {i},   \forall t, t ^ {\prime}, \tag {1}
$$

with $\mathcal { D } _ { i }$ the Internal Data (ID). The remaining distillation data is considered unknown to all teachers, hence:

$$
\mathcal {D} ^ {\mathcal {S}} = \mathcal {D} _ {e} \cup \mathcal {D} _ {i}, \tag {2}
$$

where $\mathcal { D } _ { e }$ denotes the External Data (ED) such that for any teacher of index $t , \mathcal { D } _ { e } \cap \mathcal { D } _ { t } = \emptyset$ .

# 3.3. Unseen Knowledge Transfer and Forgetting

We define ‘unseen’ as domains not present in the student’s training data but present in the teacher’s knowledge. In $\mathrm { C D } , \mathcal { D } _ { e }$ is unknown to the teachers. Such an ED can either be introduced or can appear unknowingly when generating data, a standard procedure of data-free distillation. While this could appear to be a limitation, we observe that leveraging ED allows the student to acquire knowledge about domains that have never been explicitly seen during training. We refer to this phenomenon as Unseen Knowledge Transfer (UKT). Intuitively, when integrating ED, generic knowledge is transferred because of the teacher’s uncertainty. Conversely, when the teacher is confident, specific knowledge only is transferred. In CD, we propose to take advantage of UKT by purposefully integrating ED during distillation to extract additional knowledge from the teacher.

However, in CD, the student sequentially learns from multiple teachers, each providing distinct unseen domain knowledge. While UKT enables the student to acquire information about domains not directly represented in the teacher data, this transferred knowledge is often fragile. As the student learns from subsequent teachers, they tend to lose information previously transferred from earlier ones. We refer to this phenomenon as Unseen Knowledge Forgetting (UKF). UKF differs fundamentally from the catastrophic forgetting traditionally studied in Domain Incremental Learning, as the forgotten knowledge does not originate from the student’s own training data but from the teacher’s knowledge. Since the student is never directly exposed to such knowledge, we call it unseen knowledge. An intuitive illustration of UKF and UKT is given in Figure 3.

# 3.4. Self External Data Distillation (SE2D)

We introduce Self-External Data Distillation (SE2D), a method designed to mitigate UKF in Continual Distillation. In SE2D, the student model is trained not only from the current teacher but also from its own checkpoint saved after the previous task. Such a strategy is quite common in Continual Learning [20, 27]; however, we propose to adapt it to the specific problem at hand. Therefore, the distillation process from the checkpoint is performed exclusively on external data $\mathcal { D } _ { e } ,$ which is unknown to all teachers. An overview of our proposed approach is given in Figure 4.

Prior observations indicate that performance on domains unseen by the student yet known by past teachers depends heavily on these external samples. We restrict self-distillation to external data $\mathcal { D } _ { e }$ to specifically preserve knowledge that is not directly supported by the shared internal domain $\mathcal { D } _ { i }$ . Applying self-distillation on $\mathcal { D } _ { i }$ would mainly reinforce already stable knowledge, while our goal is to maintain transferred knowledge from unseen domains, which is primarily captured through $\mathcal { D } _ { e }$ .

![](images/3f0be610b0f1dbdf2adbc6a043eadd8cb9d469b0d7cecf76aef24e804c39d13f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Teacher Sequence"] --> B["Tt"]
    B --> C["Tt"]
    C --> D["St"]
    D --> E["St-1"]
    E --> F["Output"]
    G["Ds"] --> H["Di"]
    G --> I["De"]
    H --> J["Image of dog"]
    I --> K["Image of dog"]
    J --> L["Red Box"]
    K --> M["Red Box"]
    L --> N["Red Box"]
    M --> O["Red Box"]
    N --> P["Gray Box"]
    O --> Q["Gray Box"]
    style G fill:#f9f,stroke:#333
    style H fill:#ccf,stroke:#333
    style I fill:#ccf,stroke:#333
    style J fill:#cfc,stroke:#333
    style K fill:#cfc,stroke:#333
    style L fill:#fcc,stroke:#333
    style M fill:#fcc,stroke:#333
    style N fill:#fcc,stroke:#333
    style O fill:#fcc,stroke:#333
    style P fill:#fcc,stroke:#333
```
</details>

Figure 4. Overview of Self External Data Distillation (SE2D). While the distillation from the teacher sequence is done on the entire distillation data, distillation from the checkpoint of the student at the end of the previous task is done only on $\mathcal { D } _ { e }$ .

Practically, at each distillation step t, the student $S _ { t }$ learns from both the current teacher $\mathcal { T } _ { t }$ and the previous student checkpoint $\boldsymbol { S } _ { t - 1 } \mathbf { \dot { : } }$ :

$$
\mathcal {L} _ {\mathrm{SE2D}} = \mathcal {L} _ {\mathrm{KD}} (\mathcal {S} _ {t}, \mathcal {T} _ {t}; \mathcal {D} ^ {\mathcal {S}}) + \mathcal {L} _ {\mathrm{KD}} (\mathcal {S} _ {t}, \mathcal {S} _ {t - 1}; \mathcal {D} _ {e}), \tag {3}
$$

where ${ \mathcal { L } } _ { \mathrm { K D } }$ denotes the temperature-scaled Kullback–Leibler divergence between the softmax distributions of the student logits $z _ { S } ( x )$ and teacher logits $z _ { T } ( x )$ :

$$
\mathcal {L} _ {\mathrm{KD}} (\mathcal {S}, \mathcal {T}; \mathcal {D} ^ {\mathcal {S}}) = T ^ {2} \mathbb {E} _ {x \sim \mathcal {D} ^ {\mathcal {S}}} \left[ \mathrm{KL} \left(\sigma \left(\frac {z _ {\mathcal {T}} (x)}{T}\right) \| \sigma \left(\frac {z _ {\mathcal {S}} (x)}{T}\right)\right) \right], \tag {4}
$$

where $T$ is the distillation temperature and $\sigma ( \cdot )$ denotes the softmax function. This simple yet effective mechanism enables the student to accumulate knowledge over time while retaining transferable information from previous distillation stages. In Section 5, we present experimental results of SE2D across various benchmarks.

# 4. Experimental Setup

In the following, we describe the experimental setup for Continual Distillation. More details regarding the implementation are given in the appendix.

# 4.1. Domain Selection

Teachers Domains. To reproduce a Continual Distillation context, we work on datasets containing data of the same classes but coming from different domains. For example, in Figure 1, each domain contains the class “cat”; however, the color of the cat is different depending on the considered domain. Another example would be the condition in which the picture was taken (inside or outside). Such a domain shift is typically considered in DIL [16, 33].

External Data Selection. In CD, the choice of ED is crucial. In realistic scenarios, ED could be introduced accidentally, as knowing FMs exact training domain is unlikely. In this work, we deliberately exploit such external data, specifically selected to fall outside all teacher training domains, to study its influence on knowledge transfer and forgetting. We consider two scenarios: (1) related external domains, where the data share the same semantic classes as the teacher domains (e.g., if teachers are trained on sea animals such as dolphins, sharks, and whales, the external data may include jellyfish); and (2) unrelated external domains, where the data are semantically distinct (e.g., using images of trucks or digits instead of marine animals). In both cases, teachers have minimal performance on ED. Additional discussion is included in appendix.

# 4.2. Teacher Selection

For the experimental setup, we hypothesize that most contemporary FMs share a substantial amount of common knowledge but differ primarily in their performance on specific tasks. Based on this assumption, we train a sequence of teacher models that all share a common domain while each possesses a unique domain not shared by any other teacher. Consequently, all teachers in the sequence provide a shared knowledge base together with a teacher-specific domain that the student must learn and retain through the use of external data. An example is given in Table 1 where each teacher is trained on pairs (0, 1), (0, 2), and so on.

# 4.3. Datasets

For simulating CD, we build upon Domain Incremental Learning datasets. We consider each domain separately and pre-train teachers on subsets of such domains. CI-FAR20 [30], a variation of CIFAR-100 using the 20 superclasses instead of the 100 fine-grained classes. Since in CIFAR-100, each superclass is composed of 5 subclasses, using the superclasses allows for defining 5 different domains where each domain is images from different subclasses but identical super-class. CIFAR20 contains 10, 000 train images and 2, 000 test images per domain. All such domains are considered related. Additionally, we experiment with CUB [34] and MNIST as unrelated datasets. Digits, that we define as the combination of various digit datasets, each one representing a different domain. Namely, we mix MNIST [19], MNIST-M [13], USPS [17] and SVHN [23]. As a related domain, we consider KM-NIST [6], a dataset composed of Japanese Hiragana. We reckon this dataset as having similar difficulty as MNIST, even though of different classes. DomainNet, an adapted version of the DomainNet dataset [25], containing six visual domains: Real, Clipart, Painting, Infograph, Sketch, and Quickdraw. Each domain shares the same set of 345 classes but differs significantly in style and texture statistics. This dataset contains around 600,000 images, and the domains are unbalanced, increasing the difficulty. All domains are considered related.

# 4.4. Considered Methods

We selected mainstream and recent state-of-the-art baselines for logits-based distillation methods. We considered the following. KL-divergence. This method consists of a standard distillation loss and serves as a baseline [18]. Logits Standardization (LS) [31] is a method recently proposed that improves upon logits distillation by standardizing student and teacher logits. Medium Difficulty Samples (MDS) [4] is a data-pruning strategy that considers distilling on samples of medium difficulty only. The original method was proposed in a supervised scenario where the teacher’s cross-entropy was considered as the criterion for assessing sample difficulty. In our setup, we adapted this method with the same principle, using teacher entropy as a sample difficulty estimator. Decoupled Knowledge Distillation (DKD) [39], a standard distillation method that decomposes the conventional distillation objective into two components: a target class term and a non-target class term. While most efficient in supervised scenarios, DKD can easily be used in unsupervised scenarios by considering the teacher’s maximum prediction as the target. Self-Distillation. Distillation is regularly used in CL [12, 22, 28]. A common strategy is to save a checkpoint of the current model at the end of the task and use such a checkpoint for distillation when training on the subsequent task. This methods uses both $\mathcal { D } _ { i }$ and $\mathcal { D } _ { e }$ for distillation.

# 5. Experimental Results

# 5.1. External Data Impact on UKT and UKF

External Data Improves Knowledge Transfer. A first observation that can be made in the CD setting is the necessity of training with ED to fully distill knowledge from the teacher. To showcase this effect, we train a sequence of teachers on CIFAR20 on domain pairs {0, 1}, {0, 2}, {0, 3}, while distilling to the student model on ID only (domain 0), using the KL-divergence. The results are displayed in Table 1, where the accuracy of the student after each task is reported. It is important to note that initially, considered teachers achieve above 95% accuracy on their respective domains. Eventually, distilling only on the domain 0 yields competitive performance on this domain; however, performances on other domains remain extremely limited at any training step. The student performs only on domains that have been encountered during training. In Table 1, we maintain the same teacher sequence but include ED for distillation. Such data are from the domain 4 of CIFAR20, which is unknown to the teacher. In this case, we can observe that the student maintains performance on domain 0 while achieving much stronger performances on other domains, despite never being encountered during training. UKT is observed only when distilling with external data.

Table 1. Accuracy (%) of the student model on test sets, domainwise, on CIFAR20, for various domain overlaps, after distilling for 1 epoch. The larger the ratio $| \mathcal { D } _ { e } | / | \mathcal { D } ^ { s } |$ , the more the student performs on domains unseen during training (underlined values). 

<table><tr><td>Domain</td><td>0</td><td>1</td><td>2</td><td>3</td><td> $|\mathcal{D}_{e}|/|\mathcal{D}^{S}|$ </td></tr><tr><td>After  $\mathbb{D}(\mathcal{T}_{01},\mathcal{D}_{0}^{S})$ </td><td>93.25</td><td>37.80</td><td>45.30</td><td>36.90</td><td>0%</td></tr><tr><td>After  $\mathbb{D}(\mathcal{T}_{02},\mathcal{D}_{0}^{S})$ </td><td>93.95</td><td>31.60</td><td>42.20</td><td>35.25</td><td>0%</td></tr><tr><td>After  $\mathbb{D}(\mathcal{T}_{03},\mathcal{D}_{0}^{S})$ </td><td>92.10</td><td>32.00</td><td>39.05</td><td>33.00</td><td>0%</td></tr><tr><td>After  $\mathbb{D}(\mathcal{T}_{012},\mathcal{D}_{014}^{S})$ </td><td>92.70</td><td>93.15</td><td>68.35</td><td>53.80</td><td>33%</td></tr><tr><td>After  $\mathbb{D}(\mathcal{T}_{013},\mathcal{D}_{014}^{S})$ </td><td>92.45</td><td>92.95</td><td>48.95</td><td>72.55</td><td>33%</td></tr><tr><td>After  $\mathbb{D}(\mathcal{T}_{01},\mathcal{D}_{04}^{S})$ </td><td>96.35</td><td>77.15</td><td>48.95</td><td>48.45</td><td>50%</td></tr><tr><td>After  $\mathbb{D}(\mathcal{T}_{02},\mathcal{D}_{04}^{S})$ </td><td>96.35</td><td>43.30</td><td>80.10</td><td>46.55</td><td>50%</td></tr><tr><td>After  $\mathbb{D}(\mathcal{T}_{03},\mathcal{D}_{04}^{S})$ </td><td>95.70</td><td>49.40</td><td>57.60</td><td>77.80</td><td>50%</td></tr><tr><td>After  $\mathbb{D}(\mathcal{T}_{01},\mathcal{D}_{034}^{S})$ </td><td>94.60</td><td>85.20</td><td>51.85</td><td>57.15</td><td>66%</td></tr><tr><td>After  $\mathbb{D}(\mathcal{T}_{02},\mathcal{D}_{034}^{S})$ </td><td>94.60</td><td>44.00</td><td>83.55</td><td>51.55</td><td>66%</td></tr></table>

External Data Accentuates UKF. While ED typically facilitates knowledge transfer, it can bias the model toward the most recent teacher at the expense of earlier knowledge. Tables 2 and 3 show that ED does not guarantee uniform gains and may even accentuate forgetting. For instance, on Digits, DKD and LS suffer significant drops on SVHN and MNIST-M; notably, DKD’s MNIST-M performance falls from 54.50% to 33.84%. However, this effect is not universal, as DKD shows slight gains on DomainNet’s Infograph (Table 4). This suggests that while ED can trigger substantial UKF, the impact depends heavily on the distillation method and domain distribution.

# 5.2. UKF Mitigation

Traditional Methods and UKF. In Tables 2, 3, and 4, we present the results for all methods in CD setups with related ED. Firstly, it can easily be seen that, as expected, performances on earlier domains tend to be the lowest for all methods, which is a direct demonstration of UKF where the student forgot transferred knowledge on unseen domains. In all scenarios, all traditional distillation methods suffer from UKF. Secondly, Self-Distillation can partially mitigate UKF and surpass distillation approaches due to the regularization effect of self-distillation. However, a clear trade-off emerges between UKT and UKF, not unlike the stability-plasticity trade-off in Continual Learning, where a model has to balance learning and remembering capabilities. Therefore, while self-distillation surpasses most baselines on earlier domains, it struggles to achieve high performances on more recent domains.

![](images/9d5cf692e20fb0f69fa188c7dfc40245b60e797e8703f0d679a76e8430dff3c3.jpg)

<details>
<summary>bar</summary>

| Category | KL-div. (%) | LS (%) | Self-Distill. (%) | DKD (%) | MDS (%) | SE2D (ours) (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ID only | 62 | 59 | 58 | 54.5 | 53.5 | 76 |
| + Related ED | 71.5 | 70.5 | 75 | 65.5 | 67.5 | 76.5 |
| + CUB | 67.5 | 61 | 66.5 | 52.5 | 63 | 68.5 |
| + MNIST | 60 | 56.5 | 54.5 | 50.5 | 47 | 57.5 |
</details>

Figure 5. Average Accuracy of the student across all domains known by teachers at the end of training on CIFAR20.

Performances of SE2D. SE2D allows for better knowledge retention when compared to most baselines, especially on older tasks where it can surpass baselines by more than 9% on domain 1 of CIFAR20, as reported in Table 2. However, such results hold only for sufficiently related external data where the student can adequately learn from the teacher from one task to another. In Table 2, when using MNIST in combination, the advantages of SE2D become largely limited. This behavior is even more pronounced when training on DomainNet, where SE2D falls behind Self Distillation. We identify the main reason to be the low teacher quality, giving poor supervision on ED for SE2D. Additionally, results in Table 4 show that even when using domains from DomainNet (supposedly related), performance gain is inconsistent for all methods. We believe the vast discrepancy between domains in this dataset hinders UKT, highlighting the importance of the origin of ED for efficient CD. This is discussed in more detail in appendix.

# 5.3. Discussion

External Data Ratio Matters. Another phenomenon that can be observed in Table 1 is that increasing |De|S , the proportion of ED compared to ID, enhances performances on domains unseen by the student. Therefore, a clear trend emerges: the more data unknown to the teacher is used, the stronger the UKT is observed.

External Data Origin Matters. Naturally, the origin of the external data has an impact on the intensity of UKT.

Table 2. Performances (%, higher is better) of the student at the end of training on CIFAR20 for 4 scenarios. Internal Data Only (D0), Related External Data (D4), CUB as ED, and MNIST as ED. The number of runs is set to 3. The gain columns shows the gain over using Internal Data Only. Grey : Internal Data; Blue : Domain known by the teacher; Red : External Data (ED); White: Ignored. 

<table><tr><td colspan="8">CIFAR20 - Internal Data Only</td></tr><tr><td>Method</td><td>D0</td><td>D1</td><td>D2</td><td>D3</td><td>D4 X</td><td>Avg. (0-3)</td><td>Gain (↑)</td></tr><tr><td> $\mathcal{T}_{best}$ (upper bound)</td><td>97.75</td><td>95.80</td><td>96.70</td><td>95.75</td><td>-</td><td>96.5</td><td>0.00</td></tr><tr><td>KL-divergence</td><td> $98.10 \pm 0.05$ </td><td> $41.40 \pm 0.84$ </td><td> $53.38 \pm 0.35$ </td><td> $54.87 \pm 3.70$ </td><td>-</td><td> $61.94 \pm 1.24$ </td><td>0.00</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td> $95.70 \pm 1.55$ </td><td> $35.53 \pm 1.94$ </td><td> $45.42 \pm 2.05$ </td><td> $41.02 \pm 2.55$ </td><td>-</td><td> $54.42 \pm 2.02$ </td><td>0.00</td></tr><tr><td>LS [CVPR&#x27;24]</td><td> $96.50 \pm 1.52$ </td><td> $39.13 \pm 2.90$ </td><td> $49.77 \pm 4.40$ </td><td> $49.58 \pm 5.78$ </td><td>-</td><td> $58.75 \pm 3.65$ </td><td>0.00</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td> $94.20 \pm 0.73$ </td><td> $34.30 \pm 2.80$ </td><td> $44.70 \pm 1.40$ </td><td> $41.00 \pm 1.90$ </td><td>-</td><td> $53.55 \pm 1.72$ </td><td>0.00</td></tr><tr><td>Self-Distillation</td><td> $97.45 \pm 0.43$ </td><td> $37.58 \pm 2.06$ </td><td> $50.42 \pm 1.41$ </td><td> $45.83 \pm 2.15$ </td><td>-</td><td> $57.82 \pm 1.51$ </td><td>0.00</td></tr></table>

CIFAR20 + Related External Data 

<table><tr><td>Method</td><td>D0</td><td>D1</td><td>D2</td><td>D3</td><td>D4</td><td>Avg. (0-3)</td><td>Gain (↑)</td></tr><tr><td> $\mathcal{T}_{best}$ (upper bound)</td><td>97.75</td><td>95.80</td><td>96.70</td><td>95.75</td><td>-</td><td>96.5</td><td>0.00</td></tr><tr><td>KL-divergence</td><td> $97.05 \pm 0.09$ </td><td> $48.55 \pm 1.15$ </td><td> $55.08 \pm 0.70$ </td><td> $\underline{84.77 \pm 0.87}$ </td><td>-</td><td> $71.36 \pm 0.70$ </td><td>9.42</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td> $96.05 \pm 0.50$ </td><td> $44.13 \pm 0.98$ </td><td> $51.67 \pm 0.92$ </td><td> $68.55 \pm 2.60$ </td><td>-</td><td> $65.10 \pm 1.25$ </td><td>10.68</td></tr><tr><td>LS [CVPR&#x27;24]</td><td> $96.85 \pm 0.15$ </td><td> $47.25 \pm 0.69$ </td><td> $54.25 \pm 0.46$ </td><td> $\underline{83.20 \pm 1.87}$ </td><td>-</td><td> $70.39 \pm 0.79$ </td><td>11.64</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td> $96.55 \pm 0.07$ </td><td> $45.26 \pm 2.10$ </td><td> $54.90 \pm 0.71$ </td><td> $\underline{73.51 \pm 0.56}$ </td><td>-</td><td> $67.56 \pm 0.86$ </td><td>14.01</td></tr><tr><td>Self-Distillation</td><td> $\underline{97.71 \pm 0.18}$ </td><td> $\underline{61.23 \pm 0.83}$ </td><td> $\underline{64.21 \pm 0.51}$ </td><td> $76.58 \pm 0.94$ </td><td>-</td><td> $74.93 \pm 0.61$ </td><td>17.11</td></tr><tr><td>SE2D (ours)</td><td> $\underline{97.46 \pm 0.19}$ </td><td> $\underline{70.71 \pm 1.05}$ </td><td> $\underline{62.85 \pm 0.50}$ </td><td> $73.65 \pm 1.67$ </td><td>-</td><td> $\underline{76.17 \pm 0.85}$ </td><td>n/a</td></tr></table>

CIFAR20 + CUB 

<table><tr><td>Method</td><td>D0</td><td>D1</td><td>D2</td><td>D3</td><td>CUB</td><td>Avg. (0-3)</td><td>Gain (↑)</td></tr><tr><td> $\mathcal{T}_{best}$  (upper bound)</td><td>97.75</td><td>95.80</td><td>96.70</td><td>95.75</td><td>-</td><td>96.5</td><td>0.00</td></tr><tr><td>KL-divergence</td><td> $97.24 \pm 0.37$ </td><td> $43.89 \pm 1.02$ </td><td> $55.13 \pm 0.76$ </td><td> $\underline{71.80} \pm 1.73$ </td><td>-</td><td> $\underline{67.02} \pm 0.97$ </td><td>5.08</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td> $93.39 \pm 0.88$ </td><td> $33.46 \pm 2.51$ </td><td> $43.10 \pm 2.93$ </td><td> $40.70 \pm 2.20$ </td><td>-</td><td> $\underline{52.66} \pm 2.13$ </td><td>-1.76</td></tr><tr><td>LS [CVPR&#x27;24]</td><td> $94.79 \pm 3.17$ </td><td> $38.53 \pm 4.87$ </td><td> $50.21 \pm 5.74$ </td><td> $59.32 \pm 11.60$ </td><td>-</td><td> $60.71 \pm 6.34$ </td><td>1.96</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td> $97.15 \pm 0.50$ </td><td> $41.12 \pm 0.96$ </td><td> $52.38 \pm 1.76$ </td><td> $60.38 \pm 2.48$ </td><td>-</td><td> $62.76 \pm 0.89$ </td><td>9.21</td></tr><tr><td>Self-Distillation</td><td> $\underline{97.47} \pm 0.12$ </td><td> $\underline{47.97} \pm 1.07$ </td><td> $\underline{58.40} \pm 1.34$ </td><td> $61.97 \pm 1.82$ </td><td>-</td><td> $66.45 \pm 1.09$ </td><td>8.63</td></tr><tr><td>SE2D (ours)</td><td> $\underline{97.74} \pm 0.10$ </td><td> $\underline{53.93} \pm 0.43$ </td><td> $\underline{58.02} \pm 0.45$ </td><td> $\underline{64.54} \pm 1.81$ </td><td>-</td><td> $\underline{68.56} \pm 0.70$ </td><td>n/a</td></tr></table>

CIFAR20 + MNIST 

<table><tr><td>Method</td><td>D0 (ID)</td><td>D1</td><td>D2</td><td>D3</td><td>MNIST</td><td>Avg. (0-3)</td><td>Gain (↑)</td></tr><tr><td> $\mathcal{T}_{best}$  (upper bound)</td><td>97.75</td><td>95.80</td><td>96.70</td><td>95.75</td><td>-</td><td>96.5</td><td>0.00</td></tr><tr><td>KL-divergence</td><td> $\underline{94.45 \pm 4.20}$ </td><td> $\underline{38.24 \pm 6.06}$ </td><td> $\underline{48.44 \pm 8.56}$ </td><td> $\underline{57.97 \pm 15.71}$ </td><td>-</td><td> $\underline{59.78 \pm 8.63}$ </td><td>-2.16</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td> $91.00 \pm 3.35$ </td><td> $\underline{30.71 \pm 3.60}$ </td><td> $41.36 \pm 3.61$ </td><td> $\underline{37.96 \pm 4.56}$ </td><td>-</td><td> $50.26 \pm 3.78$ </td><td>-4.16</td></tr><tr><td>LS [CVPR&#x27;24]</td><td> $92.04 \pm 4.43$ </td><td> $35.93 \pm 6.37$ </td><td> $45.79 \pm 7.50$ </td><td> $\underline{52.42 \pm 13.24}$ </td><td>-</td><td> $56.54 \pm 7.88$ </td><td>-2.21</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td> $86.83 \pm 1.04$ </td><td> $28.53 \pm 2.20$ </td><td> $37.92 \pm 2.41$ </td><td> $\underline{37.38 \pm 3.24}$ </td><td>-</td><td> $47.17 \pm 1.57$ </td><td>-6.38</td></tr><tr><td>Self-Distillation</td><td> $91.90 \pm 4.94$ </td><td> $36.35 \pm 9.46$ </td><td> $45.23 \pm 9.18$ </td><td> $43.58 \pm 12.43$ </td><td>-</td><td> $54.26 \pm 9.00$ </td><td>-3.56</td></tr><tr><td>SE2D (ours)</td><td> $\underline{92.64 \pm 4.66}$ </td><td> $\underline{39.94 \pm 9.73}$ </td><td> $\underline{47.55 \pm 9.55}$ </td><td> $48.88 \pm 13.23$ </td><td>-</td><td> $\underline{57.25 \pm 9.29}$ </td><td>n/a</td></tr></table>

Table 3. Performances (%, higher is better) of the student at the end of training on Digits for 2 scenarios. Internal Data Only (MNIST) and Related External Data (KMNIST). The number of runs is set to 3. Average and standard deviations are reported. 

<table><tr><td colspan="8">Digits - Internal Data Only</td></tr><tr><td>Method</td><td>MNIST</td><td>SVHN</td><td>MNIST-M</td><td>USPS</td><td>KMNIST X</td><td>Avg.</td><td>Gain (↑)</td></tr><tr><td> $\mathcal{T}_{best}$ (upper bound)</td><td>99.2</td><td>97.84</td><td>99.25</td><td>98.8</td><td>-</td><td>98.77</td><td>0.00</td></tr><tr><td>KL-divergence</td><td> $99.17 \pm 0.04$ </td><td> $35.80 \pm 1.49$ </td><td> $62.80 \pm 2.69$ </td><td> $95.81 \pm 0.46$ </td><td>-</td><td> $73.40 \pm 1.17$ </td><td>0.00</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td> $98.70 \pm 0.11$ </td><td> $33.35 \pm 1.28$ </td><td> $54.50 \pm 2.45$ </td><td> $95.07 \pm 0.43$ </td><td>-</td><td> $70.40 \pm 1.07$ </td><td>0.00</td></tr><tr><td>LS [CVPR&#x27;24]</td><td> $99.13 \pm 0.09$ </td><td> $37.60 \pm 1.40$ </td><td> $64.10 \pm 1.73$ </td><td> $96.00 \pm 0.25$ </td><td>-</td><td> $74.21 \pm 0.87$ </td><td>0.00</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td> $98.15 \pm 1.13$ </td><td> $34.51 \pm 4.80$ </td><td> $54.97 \pm 9.49$ </td><td> $93.21 \pm 3.01$ </td><td>-</td><td> $70.21 \pm 3.80$ </td><td>0.00</td></tr><tr><td>Self-Distillation</td><td> $99.23 \pm 0.10$ </td><td> $35.08 \pm 1.03$ </td><td> $65.87 \pm 1.38$ </td><td> $95.28 \pm 0.53$ </td><td>-</td><td> $73.87 \pm 0.76$ </td><td>0.00</td></tr></table>

Digits + Related External Data 

<table><tr><td>Method</td><td>MNIST</td><td>SVHN</td><td>MNIST-M</td><td>USPS</td><td>KMNIST</td><td>Avg.</td><td>Gain (↑)</td></tr><tr><td>KL-divergence</td><td> $99.13 \pm 0.05$ </td><td> $31.53 \pm 1.55$ </td><td> $59.84 \pm 2.57$ </td><td> $\underline{96.51} \pm \underline{0.10}$ </td><td>-</td><td> $71.75 \pm 1.07$ </td><td>-1.65</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td> $98.35 \pm 0.32$ </td><td> $25.21 \pm 1.62$ </td><td> $33.84 \pm 4.28$ </td><td> $92.87 \pm 1.63$ </td><td>-</td><td> $62.57 \pm 1.96$ </td><td>-7.83</td></tr><tr><td>LS [CVPR&#x27;24]</td><td> $99.13 \pm 0.08$ </td><td> $32.28 \pm 0.37$ </td><td> $61.47 \pm 2.15$ </td><td> $96.33 \pm 0.33$ </td><td>-</td><td> $72.30 \pm 0.73$ </td><td>-1.91</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td> $99.12 \pm 0.04$ </td><td> $33.03 \pm 1.79$ </td><td> $60.74 \pm 0.97$ </td><td> $96.13 \pm 0.05$ </td><td>-</td><td> $62.50 \pm 0.90$ </td><td>-7.71</td></tr><tr><td>Self-Distillation</td><td> $\underline{99.38} \pm \underline{0.01}$ </td><td> $\underline{55.86 \pm 1.60}$ </td><td> $\underline{90.76} \pm \underline{0.35}$ </td><td> $96.33 \pm 0.15$ </td><td>-</td><td> $\underline{85.58 \pm 0.53}$ </td><td>11.71</td></tr><tr><td>SE2D (ours)</td><td> $\underline{99.33} \pm 0.04$ </td><td> $\underline{61.84} \pm \underline{2.05}$ </td><td> $\underline{90.44 \pm 0.18}$ </td><td> $\underline{96.33 \pm 0.10}$ </td><td>-</td><td> $\underline{87.00} \pm \underline{0.60}$ </td><td>n/a</td></tr></table>

Table 4. Performances (%, higher is better) of the student at the end of training on DomainNet for 2 scenarios. Internal Data Only (Clipart) and Related External Data (Sketch). The number of runs is set to 3. Average and standard deviations are reported. 

<table><tr><td colspan="9">DomainNet - Internal Data Only</td></tr><tr><td>Method</td><td>Clipart</td><td>Infograph</td><td>Painting</td><td>Quickdraw</td><td>Real</td><td>Sketch X</td><td>Avg.</td><td>Gain (↑)</td></tr><tr><td> $\mathcal{T}_{best}$ (upper bound)</td><td>74.35</td><td>35.83</td><td>66.66</td><td>66.19</td><td>78.90</td><td>-</td><td>64.39</td><td>0.00</td></tr><tr><td>KL-divergence</td><td> $77.08 \pm 0.45$ </td><td> $18.16 \pm 0.24$ </td><td> $44.17 \pm 0.50$ </td><td> $17.19 \pm 0.73$ </td><td> $67.29 \pm 0.24$ </td><td>-</td><td> $44.78 \pm 0.29$ </td><td>0.00</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td> $77.32 \pm 0.22$ </td><td> $18.04 \pm 0.09$ </td><td> $42.76 \pm 0.40$ </td><td> $17.43 \pm 0.31$ </td><td> $63.99 \pm 0.14$ </td><td>-</td><td> $43.91 \pm 0.17$ </td><td>0.00</td></tr><tr><td>LS [CVPR&#x27;24]</td><td> $77.21 \pm 0.37$ </td><td> $17.21 \pm 0.91$ </td><td> $40.95 \pm 0.14$ </td><td> $16.42 \pm 1.01$ </td><td> $60.39 \pm 0.20$ </td><td>-</td><td> $42.43 \pm 0.44$ </td><td>0.00</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td> $76.92 \pm 0.42$ </td><td> $18.08 \pm 0.23$ </td><td> $42.14 \pm 0.38$ </td><td> $17.44 \pm 1.03$ </td><td> $63.29 \pm 0.20$ </td><td>-</td><td> $43.57 \pm 0.34$ </td><td>0.00</td></tr><tr><td>Self-Distillation</td><td> $80.57 \pm 0.08$ </td><td> $20.85 \pm 0.33$ </td><td> $46.23 \pm 0.11$ </td><td> $23.53 \pm 0.25$ </td><td> $65.38 \pm 0.04$ </td><td>-</td><td> $47.31 \pm 0.11$ </td><td>0.00</td></tr></table>

DomainNet + Related External Data 

<table><tr><td>Method</td><td>Clipart</td><td>Infograph</td><td>Painting</td><td>Quickdraw</td><td>Real</td><td>Sketch</td><td>Avg.</td><td>Gain (↑)</td></tr><tr><td>KL-divergence</td><td> $76.00 \pm 0.18$ </td><td> $18.89 \pm 0.09$ </td><td> $44.77 \pm 0.79$ </td><td> $15.53 \pm 0.15$ </td><td> $\underline{70.65} \pm \underline{0.46}$ </td><td>-</td><td> $45.17 \pm 0.31$ </td><td>0.39</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td> $76.53 \pm 0.14$ </td><td> $18.70 \pm 0.44$ </td><td> $44.24 \pm 0.32$ </td><td> $16.24 \pm 0.40$ </td><td> $68.29 \pm 0.34$ </td><td>-</td><td> $44.80 \pm 0.12$ </td><td>0.89</td></tr><tr><td>LS [CVPR&#x27;24]</td><td> $75.28 \pm 0.39$ </td><td> $16.88 \pm 0.71$ </td><td> $40.78 \pm 0.49$ </td><td> $15.47 \pm 0.43$ </td><td> $62.76 \pm 0.64$ </td><td>-</td><td> $42.24 \pm 0.31$ </td><td>-0.20</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td> $75.32 \pm 0.39$ </td><td> $17.53 \pm 0.07$ </td><td> $42.28 \pm 0.11$ </td><td> $16.12 \pm 0.61$ </td><td> $67.29 \pm 0.11$ </td><td>-</td><td> $43.71 \pm 0.19$ </td><td>0.14</td></tr><tr><td>Self-Distillation</td><td> $\underline{80.10} \pm \underline{0.18}$ </td><td> $\underline{21.53 \pm 0.09}$ </td><td> $\underline{48.15} \pm \underline{0.23}$ </td><td> $\underline{25.28} \pm \underline{0.19}$ </td><td> $\underline{68.75 \pm 0.01}$ </td><td>-</td><td> $\underline{48.76} \pm \underline{0.07}$ </td><td>1.45</td></tr><tr><td>SE2D (ours)</td><td> $\underline{78.05 \pm 0.11}$ </td><td> $\underline{21.98} \pm \underline{0.29}$ </td><td> $\underline{47.76 \pm 0.44}$ </td><td> $\underline{23.81 \pm 0.34}$ </td><td> $68.43 \pm 0.14$ </td><td>-</td><td> $\underline{48.01 \pm 0.20}$ </td><td>n/a</td></tr></table>

Intuitively, a large domain difference between external data and teacher domains might make distillation more challenging. To showcase the impact of the domain gap, we experimented with related and unrelated external domains on CIFAR20. Table 2 presents the results with various ED scenarios. Notably, it can be observed that leveraging ED leads to consistent improvement in average on all domains when related enough to ID. For example, when using D4 and CUB as ED, the performances of KL-divergence distillation increase from 61.94% to 71.36% and 67.02%, respectively. However, when using MNIST as ED, the performances slightly drop to 59.78%. Such a trend is observed for all considered methods, as it can be observed in Figure 5. Semantically, CUB consists of images of birds and is more similar to CIFAR20 than MNIST, even though the number of classes does not align. Interestingly, we observe that leveraging ED is essential to promote UKT. However, as the domain shift between ED and ID increases (e.g., in CUB), performance tends to degrade. When the domain gap becomes too large (as in MNIST), using ED can even result in lower performance than without it.

Limitations of SE2D. While SE2D reduces UKF, its impact relies on (1) the domain gap between teacher and external data and (2) teacher performance on domains unseen by the student. SE2D also requires data-origin knowledge; the student must distinguish between the teacher’s known and unknown domains. This is particularly complex when data are generated to imitate training sets, making it non-trivial to identify data outside the teacher’s domain.

# 6. Conclusions and Future Work

In this work, we introduced a new paradigm titled Continual Distillation, where a single model learns on a fixed dataset from a sequence of teachers. Such a new setup is relevant in the context of ever-evolving Foundation Models, which are costly to train, expensive to store, demanding to run inference on, and in many cases only accessible via restricted APIs. In such a context, we observe that the domain of origin of distillation data is crucial for controlling which knowledge is indeed transferred to the student. Notably, we unveil two characteristics: Unknown Knowledge Transfer and Unknown Knowledge Forgetting, which represent the ability of the student to modify its knowledge on domains that they have never encountered. Such knowledge control depends only on the teacher and the data used for distillation. The objective then becomes reaching the best UKT-UKF trade-off. In that sense, we proposed Self External Data Distillation (SE2D), which allows us to reduce UKF and maintain strong average performance on all domains, including domains unseen during training. However, we uncover that the domain gap between external data and the teacher domain must be carefully considered in order to foster UKT. Similarly, performant teachers are required for SE2D performance to be ensured.

Eventually, UKT comprises opportunities and risks, as uncontrolled or undesired knowledge could be involuntarily embedded in a student model through distillation depending on the considered data. Such a vulnerability could be easily exploited and introduce unknown bias to model training. Such an aspect of UKT should be explored in future work. Potential future directions include working with larger models, such as language or multimodal models.

# Acknowledgments

This work was partially financially supported by JST AS-PIRE Program, Japan, Grant Number JPMJAP2303. This work was partially supported by the JSPS Postdoctoral Fellowship for Research in Japan (Fellowship ID: P24752).

# References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1   
[2] Muhammad Awais, Muzammal Naseer, Salman Khan, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Foundation models defining a new era in vision: a survey and outlook. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(4):2245–2264, 2025. 1   
[3] Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. In Advances in Neural Information Processing Systems, pages 15920– 15930, 2020. 2   
[4] Yudong Chen, Xuwei Xu, Frank de Hoog, Jiajun Liu, and Sen Wang. Medium-difficulty samples constitute smoothed decision boundary for knowledge distillation on pruned datasets. In The Thirteenth International Conference on Learning Representations, 2025. 5   
[5] Zhiyuan Chen and Bing Liu. Lifelong machine learning. Synthesis Lectures on Artificial Intelligence and Machine Learning, 12(3):1–207, 2018. 2   
[6] Tarin Clanuwat, Mikel Bober-Irizar, Asanobu Kitamoto, Alex Lamb, Kazuaki Yamamoto, and David Ha. Deep learning for classical japanese literature. arXiv preprint arXiv:1812.01718, 2018. 5   
[7] Ankita De, Edward Wang, Rohan Varma, Anjali Sridhar, and Kartikay Khandelwal. Scaling multimodal foundation models in torchmultimodal with pytorch distributed. https://pytorch.org/blog/scaling-multimodal-foundationmodels-in-torchmultimodal-with-pytorch-distributed, 2025. 1   
[8] Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Ales Leonardis, Gregory Slabaugh, and ˇ Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(7):3366–3385, 2021. 1   
[9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171– 4186, 2019. 1   
[10] Na Dong, Yongqiang Zhang, Mingli Ding, and Yancheng Bai. Class-incremental object detection. Pattern Recognition, 139:109488, 2023. 2   
[11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at

scale. In International Conference on Learning Representations, 2021. 1   
[12] Arthur Douillard, Matthieu Cord, Charles Ollion, Thomas Robert, and Eduardo Valle. Podnet: Pooled outputs distillation for small-tasks incremental learning. In 16th European Conference on Conputer Vision (ECCV), pages 86–102, 2020. 2, 5   
[13] Yaroslav Ganin, Evgeniya Ustinova, Hubert Ajakan, Pascal Germain, Hugo Larochelle, Franc¸ois Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural networks. In International Conference on Machine Learning, 2016. 5   
[14] Dipam Goswami, Albin Soutif-Cormerais, Yuyang Liu, Sandesh Kamath, Bart Twardowski, Joost Van De Weijer, et al. Resurrecting old classes with new data for exemplarfree continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28525–28534, 2024. 3   
[15] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 2   
[16] Yen-Chang Hsu, Yen-Cheng Liu, Anita Ramasamy, and Zsolt Kira. Re-evaluating continual learning scenarios: A categorization and case for strong baselines. arXiv preprint arXiv:1810.12488, 2018. 5   
[17] Jonathan J. Hull. A database for handwritten text recognition research. IEEE Transactions on Pattern Analysis and Machine Intelligence, 16(5):550–554, 2002. 5   
[18] Solomon Kullback. Information theory and statistics. Courier Corporation, 1997. 5   
[19] Yann LeCun. The mnist database of handwritten digits. http://yann. lecun. com/exdb/mnist/, 1998. 5   
[20] Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(12):2935–2947, 2017. 3, 4   
[21] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019. 1   
[22] Nicolas Michel, Maorong Wang, Ling Xiao, and Toshihiko Yamasaki. Rethinking momentum knowledge distillation in online continual learning. In Proceedings of the 41st International Conference on Machine Learning, pages 35607– 35622. PMLR, 2024. 2, 3, 5   
[23] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Baolin Wu, Andrew Y Ng, et al. Reading digits in natural images with unsupervised feature learning. In NIPS workshop on deep learning and unsupervised feature learning, page 7, 2011. 5   
[24] German I Parisi, Ronald Kemker, Jose L Part, Christopher Kanan, and Stefan Wermter. Continual lifelong learning with neural networks: A review. Neural Networks, 113:54–71, 2019. 1, 2   
[25] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source

domain adaptation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1406–1415, 2019. 5   
[26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763, 2021. 1, 2   
[27] Amal Rannen, Rahaf Aljundi, Matthew B Blaschko, and Tinne Tuytelaars. Encoder based lifelong learning. In Proceedings of the IEEE International Conference on Computer Vision, pages 1320–1328, 2017. 4   
[28] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2001–2010, 2017. 2, 5   
[29] Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014. 2   
[30] Stefan Stojanov, Samarth Mishra, Ngoc Anh Thai, Nikhil Dhanda, Ahmad Humayun, Chen Yu, Linda B Smith, and James M Rehg. Incremental object learning from contiguous views. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8777– 8786, 2019. 5   
[31] Shangquan Sun, Wenqi Ren, Jingzhi Li, Rui Wang, and Xiaochun Cao. Logit standardization in knowledge distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15731–15740, 2024. 5   
[32] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste ´ Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al.\` Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2   
[33] Gido M van de Ven, Tinne Tuytelaars, and Andreas S Tolias. Three types of incremental learning. Nature Machine Intelligence, 4(12):1185–1197, 2022. 5   
[34] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. The caltech-ucsd birds-200-2011 dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011. 5   
[35] Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning: Theory, method and application. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 1   
[36] Yuzheng Wang, Dingkang Yang, Zhaoyu Chen, Yang Liu, Siao Liu, Wenqiang Zhang, Lihua Zhang, and Lizhe Qi. De-confounded data-free knowledge distillation for handling distribution shifts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12615–12625, 2024. 1, 2   
[37] Chenshen Wu, Luis Herranz, Xialei Liu, Joost Van De Weijer, Bogdan Raducanu, et al. Memory replay gans: Learning

to generate new categories without forgetting. Advances in Neural Information Processing Systems, 31, 2018. 3   
[38] Junho Yim, Donggyu Joo, Jihoon Bae, and Junmo Kim. A gift from knowledge distillation: Fast optimization, network minimization and transfer learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4133–4141, 2017. 2   
[39] Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, and Jiajun Liang. Decoupled knowledge distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11953–11962, 2022. 2, 5

# A. Experimental Setup

# A.1. Implementation Details

For training, we start from pre-trained weights and use the Adam optimizer with a learning rate of 0.0001 for 3 epochs. As students start from pre-trained weights, we observe negligible improvement when using more epochs. Images are resized to 224 × 224 to fit the size used during pre-training. We use random horizontal flips as augmentations and use data normalization. We use a batch size of 64. Regarding the distillation, we use the KL-divergence with a temperature of 10. Experiments are conducted with ViT architecture, namely the ViT-B/16. We use ViT-base as teachers for CIFAR20 and DomainNet. For Digits, we use a ViT-tiny as a teacher. Every teacher is initialised from pre-trained weights and trained for 50 epochs with Adam optimizer and a learning rate of 0.0001. We use the same architectures for students in the main draft and additionally experiment with ViT-tiny. For all results, 3 seeds are used.

For DomainNet, since domains are unbalanced, we oversample or undersample them so that each task has the same number of steps. The base task length is determined by the Internal Data (ID) size, and the External Data (ED) is sampled accordingly. For example, if ID is larger than ED, we oversample from ED.

# A.2. Metric

For all experiments, we report the domain-wise Final Accuracy (%) of the student at the end of the training sequence.

# B. Additional Discussions

Feasibility of obtaining unknown ED. In CD, a question naturally arises as to whether obtaining ED unknown to FMs is feasible, since such models are trained on vast and diverse data. We argue that obtaining unknown data is feasible in two highly plausible scenarios: Temporal Gap: Any data uploaded to public repositories after the FM’s training cutoff is guaranteed to be unknown. Private/Synthetic Data: In industrial settings, private proprietary datasets or synthetically generated data serve as excellent ED.

Assessing ED quality. As presented throughout this paper, ED selection is key in Continual Distillation. While quantifying semantic similarity is hard without teacher data, we propose using the teacher’s own predictive uncertainty, by measuring the entropy, as a proxy. In Figure B.1, we show the entropy distribution of a teacher trained on two domains of CIFAR20, for various domains of CIFAR20 as well as CUB and MNIST. We observe that the more the external data is ”unrelated”, the more ”flat” the entropy distribution becomes. To select adequate ED without knowing the training distribution, a user can 1) filter out samples based on an entropy threshold, 2) use the 4th-order moment (kurtosis) of the entropy distribution to quantify ”flatness”. As shown in Fig. B.1, lower flatness correlates with higher UKT potential. Additionally, in Figure B.2, the entropy distribution varies widely across domains, partially explaining the mitigated results observed on this dataset.

![](images/72c99b7f295deb1e250995c129f7306e2912527ce30227bf082b575d2ccad63f.jpg)

<details>
<summary>area</summary>

| Dataset         | Kurtosis |
| --------------- | -------- |
| CIFAR Domain 0 | 33.94    |
| CIFAR Domain 1  | 27.06    |
| CIFAR Domain 2  | 4.80     |
| CIFAR Domain 3  | 5.04     |
| CIFAR Domain 4  | 3.89     |
| CUB             | 2.65     |
| MNIST           | 1.84     |
</details>

Figure B.1. Entropy distribution of the teacher (ViT-Base) trained on domains 0 and 1 of CIFAR20 for various datasets and domains.

![](images/a4f54754c42fd7afdef08607253eefdf677513a0a8099b156089031dc54b1fe7.jpg)

<details>
<summary>area</summary>

| Method       | Kurtosis |
| ------------ | -------- |
| clipart      | 7.59     |
| real         | 8.67     |
| infograph    | 1.83     |
| painting     | 2.52     |
| quickdraw    | 1.90     |
| sketch       | 2.48     |
</details>

Figure B.2. Entropy distribution of the teacher (ViT-Base) trained on domains Real and Clipart of Domainnet for various Domain-Net domains.

# C. Additional Experiments

# C.1. Additional Metrics

Forgetting For each method, we measure forgetting as the drop in performance on the learned domains after training from new teachers. Formally, for a student trained to imitate a teacher of index t, the forgetting for domain d is computed as:

$$
F _ {d} = \max _ {i <   t} A _ {d} ^ {(i)} - A _ {d} ^ {(t)}
$$

where $A _ { d } ^ { ( i ) }$ is the accuracy on domain d after distillation from teacher t. The overall forgetting is averaged across all domains:

Table C.1. Forgetting (%, lower is better) of the student at the end of training on CIFAR20 for 4 scenarios. Internal Data Only (D0), Related External Data (D4), CUB as ED, and MNIST as ED. The number of runs is set to 3. 

<table><tr><td colspan="7">CIFAR20 - Internal Data Only</td></tr><tr><td>Method</td><td>D0</td><td>D1</td><td>D2</td><td>D3</td><td>D4 X</td><td>Avg. (0-3)</td></tr><tr><td>KL-divergence</td><td>0</td><td>11.95</td><td>2.98</td><td>0</td><td>-</td><td>3.73</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td>0</td><td>2.38</td><td>3.47</td><td>0</td><td>-</td><td>1.46</td></tr><tr><td>LS [CVPR&#x27;24]</td><td>0</td><td>10.42</td><td>1.88</td><td>0</td><td>-</td><td>3.08</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td>0</td><td>11.95</td><td>2.98</td><td>0</td><td>-</td><td>3.23</td></tr><tr><td>Self-Distillation</td><td>0</td><td>16.02</td><td>2.47</td><td>0</td><td>-</td><td>4.12</td></tr><tr><td colspan="7">CIFAR20 + Related External Data</td></tr><tr><td>Method</td><td>D0</td><td>D1</td><td>D2</td><td>D3</td><td>D4</td><td>Avg. (0-3)</td></tr><tr><td>KL-divergence</td><td>0.52</td><td>38.15</td><td>30.26</td><td>0</td><td>-</td><td>17.23</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td>0.13</td><td>25.15</td><td>16.25</td><td>0</td><td>-</td><td>10.38</td></tr><tr><td>LS [CVPR&#x27;24]</td><td>0.62</td><td>39.23</td><td>32.08</td><td>0</td><td>-</td><td>17.98</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td>0.29</td><td>37.74</td><td>31.00</td><td>0</td><td>-</td><td>17.26</td></tr><tr><td>Self-Distillation</td><td>0</td><td>25.33</td><td>7.93</td><td>0</td><td>-</td><td>8.32</td></tr><tr><td>SE2D (ours)</td><td>0</td><td>16.10</td><td>3.67</td><td>0</td><td>-</td><td>4.44</td></tr><tr><td colspan="7">CIFAR20 + CUB</td></tr><tr><td>Method</td><td>D0</td><td>D1</td><td>D2</td><td>D3</td><td>CUB</td><td>Avg. (0-3)</td></tr><tr><td>KL-divergence</td><td>0.93</td><td>30.68</td><td>17.82</td><td>0</td><td>-</td><td>12.36</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td>0.89</td><td>5.37</td><td>3.86</td><td>0</td><td>-</td><td>2.03</td></tr><tr><td>LS [CVPR&#x27;24]</td><td>2.54</td><td>32.28</td><td>20.66</td><td>0</td><td>-</td><td>11.87</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td>0.61</td><td>30.79</td><td>16.26</td><td>0</td><td>-</td><td>11.41</td></tr><tr><td>Self-Distillation</td><td>0.81</td><td>27.88</td><td>2.48</td><td>0</td><td>-</td><td>7.29</td></tr><tr><td>SE2D (ours)</td><td>0.40</td><td>21.61</td><td>6.81</td><td>0</td><td>-</td><td>7.21</td></tr><tr><td colspan="7">CIFAR20 + MNIST</td></tr><tr><td>Method</td><td>D0</td><td>D1</td><td>D2</td><td>D3</td><td>MNIST</td><td>Avg. (0-3)</td></tr><tr><td>KL-divergence</td><td>0</td><td>15.20</td><td>7.40</td><td>0</td><td>-</td><td>5.15</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td>0</td><td>2.74</td><td>0.57</td><td>0</td><td>-</td><td>0.83</td></tr><tr><td>LS [CVPR&#x27;24]</td><td>1.57</td><td>14.58</td><td>9.50</td><td>0</td><td>-</td><td>5.41</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td>0.97</td><td>15.34</td><td>7.67</td><td>0</td><td>-</td><td>5.00</td></tr><tr><td>Self-Distillation</td><td>0</td><td>9.69</td><td>1.02</td><td>0</td><td>-</td><td>2.18</td></tr><tr><td>SE2D (ours)</td><td>0.85</td><td>9.88</td><td>2.85</td><td>0</td><td>-</td><td>4.43</td></tr></table>

$$
F = \frac {1}{D} \sum_ {d = 1} ^ {D} F _ {d}
$$

This metric captures the extent to which a method forgets previously learned knowledge when adapting to new tasks. The results are presented in Tables C.1, C.2 and C.3. It can be observed that our method leads to competitive forgetting in all scenarios.

Accuracy Curves We report per-domain accuracy curves during training with related external data in Figures D.1 to Figure D.4.

# C.2. Additional Architecures

We report results with additional architectures. Namely, we experimented with a ViT-tiny as a student instead of the ViT-base version in the main manuscript. Similarly, we ex-

Algorithm 1 Continual Distillation with KL divergence and SGD. 

<table><tr><td colspan="2">Require: Sequence of teachers {T1,T2,...,TT}, student model Sθ, distillation dataset DS</td></tr><tr><td colspan="2">1: for t = 1 to N do</td></tr><tr><td colspan="2">2: for x ∈ DS do</td></tr><tr><td colspan="2">3: Obtain teacher predictions pt(x) = TT(x)</td></tr><tr><td colspan="2">4: Student predictions qθ(x) = Sθ(x)</td></tr><tr><td colspan="2">5: Distillation loss: Lat = KL{pt(x) || qθ(x)}</td></tr><tr><td colspan="2">6: Update student parameters θ ← θ - η∇θLt</td></tr><tr><td colspan="2">7: end for</td></tr><tr><td colspan="2">8: end for</td></tr><tr><td colspan="2">9: return Trained student model Sθ</td></tr></table>

perimented with larger models as teacher using CLIP-base teachers (ViT-L/14). Results are presented in Table C.5.

# C.3. Additional Sequences

As presented in the main paper, DomainNet is particularly challenging. Therefore, we conducted additional experiments where challenging domains have either been removed or used as external data. Such results are presented in Table C.6. Despite SE2D’s lower overall performance compared to Self-Distillation, the disparity between the two methods diminishes in less complex scenarios, where SE2D simultaneously expands its lead over the baseline.

# D. Algorithms

To provide a clear overview of our training methodology, we present the Continual Distillation procedure in Algorithm 1. Furthermore, a detailed description of our proposed SE2D approach is provided in Algorithm 2.

Algorithm 2 Overview of SE2D training algorithm with Continual Distillation.   
Require: Sequence of teachers $\{T_{1}, T_{2}, \ldots, T_{T}\}$ , student model $S_{\theta}$ , distillation dataset $D^{S} = D_{i} \cup D_{e}$ 1: for t = 1 to N do

2: if t = 1 then

3: for $x \in D^{S}$ do

4: Obtain teacher predictions $p_{t}(x) = \mathcal{T}_{t}(x)$ 5: Compute student predictions $q_{\theta}(x) = \mathcal{S}_{\theta}(x)$ 6: Compute distillation loss: $\mathcal{L}_{t} = \text{KL}(p_{t}(x) \parallel q_{\theta}(x))$ 7: Update student parameters $\theta \leftarrow \theta - \eta \nabla_{\theta} \mathcal{L}_{t}$ 8: end for

9: else

10: Load previous student checkpoint $S_{\theta}^{t-1}$ 11: for $(x_{e}, x_{i}) \in (\mathcal{D}_{e}, \mathcal{D}_{i})$ do

12: Compute student predictions on internal data: $q_{\theta}(x_{i}) = \mathcal{S}_{\theta}(x_{i})$ 13: Compute student predictions on external data: $q_{\theta}(x_{e}) = \mathcal{S}_{\theta}(x_{e})$ 14: Obtain previous student predictions on external data: $p_{t-1}(x_{e}) = \mathcal{S}_{\theta}^{t-1}(x_{e})$ 15: Obtain current teacher predictions on all data: $p_{t}^{\text{teacher}}((x_{e}, x_{i})) = \mathcal{T}_{t}((x_{e}, x_{i}))$ 16: Compute distillation loss on external data from previous student: $\mathcal{L}_{\text{student}} = \text{KL}(p_{t-1}(x_{e}) \parallel q_{\theta}(x_{e}))$ 17: Compute distillation loss on all data from teacher: $\mathcal{L}_{\text{teacher}} = \text{KL}(p_{t}^{\text{teacher}}((x_{e}, x_{i})) \parallel q_{\theta}((x_{e}, x_{i})) )$ 18: Compute total loss: $L_{t} = L_{student} + L_{teacher}$ 19: Update student parameters $\theta \leftarrow \theta - \eta \nabla_{\theta} L_{t}$ 20: end for

21: end if

22: end for

23: return Trained student model $S_{\theta}$

Table C.2. Forgetting (%, lower is better) of the student at the end of training on Digits for 2 scenarios. Internal Data Only (D0), Related External Data (D4). The number of runs is set to 3. 

<table><tr><td colspan="7">Digits - Internal Data Only</td></tr><tr><td>Method</td><td>MNIST</td><td>SVHN</td><td>MNIST-M</td><td>USPS</td><td>KMNIST X</td><td>Avg.</td></tr><tr><td>KL-divergence</td><td>0.23</td><td>3.34</td><td>16.20</td><td>0</td><td>-</td><td>4.94</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td>0.47</td><td>0.53</td><td>11.73</td><td>0</td><td>-</td><td>3.12</td></tr><tr><td>LS [CVPR&#x27;24]</td><td>0.10</td><td>4.09</td><td>12.88</td><td>0</td><td>-</td><td>4.40</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td>0.23</td><td>3.34</td><td>16.20</td><td>0</td><td>-</td><td>4.94</td></tr><tr><td>Self-Distillation</td><td>0.13</td><td>4.06</td><td>7.46</td><td>0</td><td>-</td><td>3.54</td></tr></table>

Digits + Related External Data 

<table><tr><td>Method</td><td>MNIST</td><td>SVHN</td><td>MNIST-M</td><td>USPS</td><td>KMNIST</td><td>Avg.</td></tr><tr><td>KL-divergence</td><td>0.24</td><td>40.68</td><td>38.41</td><td>0</td><td>-</td><td>19.17</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td>0.84</td><td>12.00</td><td>50.77</td><td>0</td><td>-</td><td>15.87</td></tr><tr><td>LS [CVPR&#x27;24]</td><td>0.27</td><td>40.50</td><td>36.59</td><td>0</td><td>-</td><td>19.22</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td>0.25</td><td>40.69</td><td>38.35</td><td>0</td><td>-</td><td>19.16</td></tr><tr><td>Self-Distillation</td><td>0.10</td><td>16.33</td><td>6.35</td><td>0</td><td>-</td><td>5.58</td></tr><tr><td>SE2D (ours)</td><td>0.13</td><td>10.37</td><td>4.90</td><td>0</td><td>-</td><td>3.73</td></tr></table>

Table C.3. Forgetting (%, lower is better) on DomainNet for 2 scenarios: Internal Data Only (Seq -1) and Related External Data (Seq 0). Averaged across 3 runs. 

<table><tr><td colspan="8">DomainNet - Internal Data Only</td></tr><tr><td>Method</td><td>Clipart</td><td>Infograph</td><td>Painting</td><td>Quickdraw</td><td>Real</td><td>Sketch ✗</td><td>Avg.</td></tr><tr><td>KL-divergence</td><td>4.63</td><td>9.91</td><td>13.70</td><td>20.31</td><td>0</td><td>-</td><td>12.14</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td>3.85</td><td>10.34</td><td>13.89</td><td>22.38</td><td>0</td><td>-</td><td>12.61</td></tr><tr><td>LS [CVPR&#x27;24]</td><td>3.50</td><td>11.48</td><td>15.39</td><td>23.73</td><td>0</td><td>-</td><td>13.52</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td>5.20</td><td>10.70</td><td>15.69</td><td>21.46</td><td>0</td><td>-</td><td>13.26</td></tr><tr><td>Checkpoint</td><td>1.27</td><td>5.86</td><td>11.30</td><td>20.68</td><td>0</td><td>-</td><td>9.78</td></tr><tr><td>SE2D (ours)</td><td>4.60</td><td>9.58</td><td>13.49</td><td>20.27</td><td>0</td><td>-</td><td>11.98</td></tr></table>

<table><tr><td colspan="8">DomainNet + Related External Data</td></tr><tr><td>Method</td><td>Clipart</td><td>Infograph</td><td>Painting</td><td>Quickdraw</td><td>Real</td><td>Sketch X</td><td>Avg.</td></tr><tr><td>KL-divergence</td><td>5.56</td><td>11.89</td><td>18.07</td><td>29.12</td><td>0</td><td>-</td><td>16.16</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td>4.64</td><td>12.15</td><td>18.31</td><td>29.69</td><td>0</td><td>-</td><td>16.20</td></tr><tr><td>LS [CVPR&#x27;24]</td><td>5.40</td><td>13.70</td><td>20.98</td><td>32.44</td><td>0</td><td>-</td><td>18.13</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td>6.25</td><td>13.03</td><td>19.68</td><td>29.62</td><td>0</td><td>-</td><td>17.15</td></tr><tr><td>Checkpoint</td><td>1.70</td><td>6.63</td><td>14.54</td><td>29.43</td><td>0</td><td>-</td><td>13.08</td></tr><tr><td>SE2D (ours)</td><td>3.42</td><td>6.91</td><td>15.54</td><td>30.86</td><td>0</td><td>-</td><td>14.18</td></tr></table>

Table C.4. Performances (%, higher is better) of the student at the end of training on CIFAR20 for 2 scenarios with a ViT-tiny. Internal Data Only (D0), Related External Data (D4). The number of runs is set to 3. Average and standard deviations are reported. 

<table><tr><td colspan="7">CIFAR20 - Internal Data Only</td></tr><tr><td>Method</td><td>D0</td><td>D1</td><td>D2</td><td>D3</td><td>D4 X</td><td>Avg. (0-3)</td></tr><tr><td> $\mathcal{T}_{best}$ (upper bound)</td><td>97.75</td><td>95.80</td><td>96.70</td><td>95.75</td><td>-</td><td>96.5</td></tr><tr><td>KL-divergence</td><td>96.27 ± 0.21</td><td>37.85 ± 0.78</td><td>52.28 ± 0.42</td><td>49.48 ± 1.28</td><td>-</td><td>58.97 ± 0.67</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td>95.40 ± 1.05</td><td>33.27 ± 1.60</td><td>45.38 ± 0.90</td><td>40.45 ± 1.34</td><td>-</td><td>53.62 ± 1.22</td></tr><tr><td>LS [CVPR&#x27;24]</td><td>96.25 ± 0.85</td><td>38.95 ± 2.17</td><td>51.58 ± 2.47</td><td>50.15 ± 4.40</td><td>-</td><td>59.23 ± 2.47</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td>94.45 ± 0.44</td><td>35.08 ± 0.88</td><td>45.35 ± 0.41</td><td>41.98 ± 3.10</td><td>-</td><td>54.22 ± 1.21</td></tr><tr><td>Self-Distillation</td><td>96.27 ± 0.35</td><td>37.93 ± 1.22</td><td>51.82 ± 0.53</td><td>46.43 ± 0.97</td><td>-</td><td>58.11 ± 0.77</td></tr></table>

<table><tr><td colspan="7">CIFAR20 + Related External Data</td></tr><tr><td>Method</td><td>D0</td><td>D1</td><td>D2</td><td>D3</td><td>D4</td><td>Avg. (0-3)</td></tr><tr><td> $\mathcal{T}_{best}$ (upper bound)</td><td>97.75</td><td>95.80</td><td>96.70</td><td>95.75</td><td>-</td><td>96.5</td></tr><tr><td>KL-divergence</td><td>96.36 ± 0.33</td><td>46.54 ± 1.17</td><td>55.86 ± 1.10</td><td>75.96 ± 1.76</td><td>-</td><td>68.68 ± 1.09</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td>95.18 ± 0.46</td><td>40.77 ± 0.95</td><td>50.92 ± 1.07</td><td>60.69 ± 1.32</td><td>-</td><td>61.89 ± 0.95</td></tr><tr><td>LS [CVPR&#x27;24]</td><td>96.45 ± 0.26</td><td>46.80 ± 0.74</td><td>55.08 ± 0.61</td><td>76.49 ± 1.50</td><td>-</td><td>68.7 ± 0.78</td></tr><tr><td>Self-Distillation</td><td>97.26 ± 0.25</td><td>55.42 ± 0.31</td><td>61.62 ± 0.70</td><td>68.74 ± 1.46</td><td>-</td><td>70.76 ± 0.68</td></tr><tr><td>SE2D (ours)</td><td>96.77 ± 0.25</td><td>62.33 ± 1.19</td><td>59.45 ± 1.36</td><td>65.82 ± 2.03</td><td>-</td><td>71.09 ± 1.21</td></tr></table>

Table C.5. Domain Accuracy (%, higher is better) on DomainNet with CLIP-based teachers. Mean and standard deviation are reported. 

<table><tr><td colspan="8">DomainNet - Internal Data Only</td></tr><tr><td>Method</td><td>Clipart</td><td>Infograph</td><td>Painting</td><td>Quickdraw</td><td>Real</td><td>Sketch X</td><td>Avg.</td></tr><tr><td> $\mathcal{T}_{best}$  (upper bound)</td><td>86.33</td><td>53.89</td><td>79.74</td><td>69.62</td><td>89.86</td><td>-</td><td>76.23</td></tr><tr><td>KL-divergence</td><td>78.33 ± 0.23</td><td>19.00 ± 0.30</td><td>39.25 ± 0.81</td><td>35.36 ± 0.49</td><td>51.51 ± 0.74</td><td>-</td><td>44.69 ± 0.32</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td>78.55 ± 0.03</td><td>18.89 ± 0.41</td><td>39.69 ± 0.19</td><td>34.77 ± 0.38</td><td>52.45 ± 0.49</td><td>-</td><td>44.87 ± 0.06</td></tr><tr><td>LS [CVPR&#x27;24]</td><td>74.39 ± 0.39</td><td>15.54 ± 0.66</td><td>33.92 ± 0.56</td><td>20.09 ± 1.11</td><td>46.62 ± 0.76</td><td>-</td><td>38.11 ± 0.64</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td>75.51 ± 0.10</td><td>17.52 ± 0.20</td><td>35.67 ± 1.39</td><td>26.22 ± 0.70</td><td>47.71 ± 0.69</td><td>-</td><td>40.53 ± 0.27</td></tr><tr><td>Self-Distillation</td><td>79.99 ± 0.55</td><td>21.72 ± 0.80</td><td>43.65 ± 0.57</td><td>26.52 ± 1.42</td><td>57.21 ± 0.13</td><td>-</td><td>45.82 ± 0.61</td></tr></table>

DomainNet + Related External Data 

<table><tr><td>Method</td><td>Clipart</td><td>Infograph</td><td>Painting</td><td>Quickdraw</td><td>Real</td><td>Sketch ✗</td><td>Avg.</td></tr><tr><td> $\mathcal{T}_{best}$  (upper bound)</td><td>86.33</td><td>53.89</td><td>79.74</td><td>69.62</td><td>89.86</td><td>-</td><td>76.23</td></tr><tr><td>KL-divergence</td><td>78.22 ± 0.39</td><td>19.98 ± 0.17</td><td>41.59 ± 0.55</td><td>42.83 ± 0.37</td><td>52.62 ± 0.40</td><td>-</td><td>47.05 ± 0.22</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td>78.17 ± 0.11</td><td>20.01 ± 0.19</td><td>41.78 ± 0.39</td><td>42.69 ± 0.84</td><td>52.37 ± 0.24</td><td>-</td><td>47.00 ± 0.01</td></tr><tr><td>LS [CVPR&#x27;24]</td><td>74.64 ± 0.13</td><td>16.63 ± 0.31</td><td>37.09 ± 0.09</td><td>26.78 ± 0.35</td><td>47.45 ± 0.35</td><td>-</td><td>40.52 ± 0.17</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td>75.21 ± 0.59</td><td>18.55 ± 0.45</td><td>39.32 ± 0.13</td><td>31.84 ± 0.73</td><td>49.32 ± 0.39</td><td>-</td><td>42.85 ± 0.39</td></tr><tr><td>Self-Distillation</td><td>79.47 ± 0.50</td><td>22.84 ± 0.60</td><td>47.51 ± 1.33</td><td>30.15 ± 0.77</td><td>58.51 ± 1.37</td><td>-</td><td>47.69 ± 0.84</td></tr><tr><td>SE2D (ours)</td><td>78.52 ± 0.10</td><td>21.34 ± 0.45</td><td>47.12 ± 0.35</td><td>29.18 ± 0.30</td><td>58.01 ± 0.61</td><td>-</td><td>46.83 ± 0.31</td></tr></table>

Table C.6. Accuracy per domain (%). Grey : Internal Data; Blue : Domain known by the teacher (active); Red : External Data (ED); White : Ignored. Avg computed on Internal Data + Active domains. 

<table><tr><td colspan="8">DomainNet - Sequence 1 (Quickdraw is used as ED and Infograph is ignored)</td></tr><tr><td>Method</td><td>Clipart</td><td>Infograph</td><td>Painting</td><td>Quickdraw</td><td>Real</td><td>Sketch</td><td>Avg.</td></tr><tr><td>KL-divergence</td><td> $74.66 \pm 0.14$ </td><td>-</td><td> $33.01 \pm 0.21$ </td><td>-</td><td> $46.17 \pm 0.62$ </td><td> $48.68 \pm 0.15$ </td><td>50.63</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td> $76.18 \pm 0.20$ </td><td>-</td><td> $36.20 \pm 0.49$ </td><td>-</td><td> $49.70 \pm 0.62$ </td><td> $54.85 \pm 0.32$ </td><td>54.23</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td> $70.76 \pm 0.40$ </td><td>-</td><td> $30.01 \pm 0.78$ </td><td>-</td><td> $43.52 \pm 1.09$ </td><td> $51.56 \pm 0.68$ </td><td>48.96</td></tr><tr><td>Self-Distillation</td><td> $80.43 \pm 0.10$ </td><td>-</td><td> $47.47 \pm 0.71$ </td><td>-</td><td> $61.10 \pm 0.59$ </td><td> $58.15 \pm 0.31$ </td><td>61.79</td></tr><tr><td>SE2D (ours)</td><td> $76.89 \pm 0.25$ </td><td>-</td><td> $38.72 \pm 0.49$ </td><td>-</td><td> $52.83 \pm 0.20$ </td><td> $58.84 \pm 0.25$ </td><td>56.82</td></tr></table>

DomainNet - Sequence 2 (Infograph is used as ED) 

<table><tr><td>Method</td><td>Clipart</td><td>Infograph</td><td>Painting</td><td>Quickdraw</td><td>Real</td><td>Sketch</td><td>Avg.</td></tr><tr><td>KL-divergence</td><td> $75.78 \pm 0.14$ </td><td>-</td><td> $34.61 \pm 0.52$ </td><td> $16.59 \pm 0.22$ </td><td> $49.27 \pm 0.43$ </td><td> $52.43 \pm 0.44$ </td><td>45.73</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td> $76.73 \pm 0.33$ </td><td>-</td><td> $36.48 \pm 0.68$ </td><td> $17.53 \pm 0.44$ </td><td> $51.84 \pm 0.55$ </td><td> $56.83 \pm 0.34$ </td><td>47.88</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td> $75.28 \pm 0.17$ </td><td>-</td><td> $35.43 \pm 0.92$ </td><td> $16.47 \pm 0.48$ </td><td> $50.74 \pm 0.88$ </td><td> $56.65 \pm 0.37$ </td><td>46.91</td></tr><tr><td>Self-Distillation</td><td> $\underline{80.45} \pm \underline{0.06}$ </td><td>-</td><td> $\underline{48.48 \pm 0.41}$ </td><td> $\underline{20.84 \pm 0.89}$ </td><td> $\underline{64.79 \pm 0.40}$ </td><td> $\underline{59.04 \pm 0.24}$ </td><td> $\underline{54.72}$ </td></tr><tr><td>SE2D (ours)</td><td> $\underline{78.08 \pm 0.20}$ </td><td>-</td><td> $\underline{49.02 \pm 0.36}$ </td><td> $\underline{18.50 \pm 0.40}$ </td><td> $\underline{63.29 \pm 0.32}$ </td><td> $\underline{58.99 \pm 0.22}$ </td><td> $\underline{53.58}$ </td></tr></table>

DomainNet - Sequence 3 (Sketch is used as ED, Infograph and Quickdraw are ignored) 

<table><tr><td>Method</td><td>Clipart</td><td>Infograph</td><td>Painting</td><td>Quickdraw</td><td>Real</td><td>Sketch</td><td>Avg.</td></tr><tr><td>KL-divergence</td><td> $75.75 \pm 0.35$ </td><td>-</td><td> $42.44 \pm 0.67$ </td><td>-</td><td> $65.15 \pm 0.53$ </td><td>-</td><td>61.11</td></tr><tr><td>DKD [CVPR&#x27;22]</td><td> $76.25 \pm 0.23$ </td><td>-</td><td> $45.52 \pm 0.72$ </td><td>-</td><td> $70.50 \pm 0.12$ </td><td>-</td><td>64.09</td></tr><tr><td>MDS [ICLR&#x27;25]</td><td> $75.25 \pm 0.07$ </td><td>-</td><td> $44.74 \pm 0.94$ </td><td>-</td><td> $70.11 \pm 0.35$ </td><td>-</td><td>63.37</td></tr><tr><td>Self-Distillation</td><td> $\underline{79.54} \pm \underline{0.26}$ </td><td>-</td><td> $\underline{59.61} \pm \underline{0.30}$ </td><td>-</td><td> $\underline{71.48} \pm \underline{0.26}$ </td><td>-</td><td> $\underline{70.21}$ </td></tr><tr><td>SE2D (ours)</td><td> $\underline{78.04} \pm \underline{0.15}$ </td><td>-</td><td> $\underline{59.36} \pm \underline{0.68}$ </td><td>-</td><td> $\underline{70.61} \pm \underline{0.40}$ </td><td>-</td><td> $\underline{69.34}$ </td></tr></table>

![](images/ec9854a3edd2cc8472b6a85d38d6eca6dbb96b3c068fb79fd02cbc6ed9ee4471.jpg)

Figure D.1. Accuracy of the student on all domains at all steps for the considered methods on CIFAR20, training with Internal Data only.   
![](images/bed618e4be83c81d4859bc9d05985447decff5835941dccad21bc2642b8a9dc9.jpg)  
Figure D.2. Accuracy of the student on all domains at all steps for the considered methods on CIFAR20, training with External Data.

![](images/37407774e38109ead59771dd3e59ba430dca0f0ff040d22f4106ad45989838e3.jpg)  
KL-divergence Self-Distillation + DKD [CVPR'22] → LS[CVPR'24]

Figure D.3. Accuracy of the student on all domains at all steps for the considered methods on Digits, training with Internal Data only.   
![](images/6b6c531d9f40eaf1e36953a5d833ef3349ac8b6a7007c58c78b980593356dded.jpg)  
KL-divergence →Self-Distillation SE2D (Ours) + DKD [CVPR'22] LS [CVPR'24]

Figure D.4. Accuracy of the student on all domains at all steps for the considered methods on Digits, training with External Data.