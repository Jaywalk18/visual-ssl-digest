# LEViL: Label-Efficient Video Learning via Zero-Shot Distillation over VLM-Generated Pseudo-Label Spaces

Aslı C¸ elik

Department of Electronics and Communications Engineering

Kocaeli University

Kocaeli, Turkiye¨

asli.celik@kocaeli.edu.tr

Abstract—Supervised video pretraining is a common transfer learning practice for improving downstream action recognition performance. However, it requires large-scale labeled source datasets, and the effectiveness of the learned initialization is influenced by the similarity between the source and target domains. Constructing such labeled pretraining datasets for different target domains is costly and difficult to scale. To address these limitations, this study proposes a label-efficient video learning framework that combines annotation-free video pretraining with target-label-set-aware fine-tuning. During pretraining, a vision-language model (VLM) generates textual descriptions of unlabeled videos, which are processed to construct an interpretable semantic pseudo-label space. A frozen video-language model then produces zero-shot soft target distributions over this space, allowing a student video encoder to learn semantically rich representations without manual source annotations. During downstream adaptation, target-label-set-aware fine-tuning combines supervised learning from labeled target videos with zeroshot distillation over the actual target label set, helping preserve VLM-derived semantic guidance while adapting the pretrained encoder to the target task. Experiments on UCF101 and HMDB51 show that the proposed framework outperforms the compared semi-supervised video action recognition methods across all evaluated limited-label regimes. Moreover, the annotation-free pretraining stage learns transferable representations that provide an effective initialization for full-data fine-tuning, despite relying on a comparatively modest unlabeled pretraining pool.

## I. INTRODUCTION

Training deep learning models on video data remains challenging because manual video annotation is costly, and supervised learning with limited labeled data can easily lead to overfitting and poor generalization. Compared with imagebased models, video models are often more computationally demanding and may contain more parameters due to the additional temporal dimension, increasing their need for training data. Transfer learning is widely used to address these challenges by leveraging representations learned through pretraining on large-scale source datasets. In video learning, this is commonly achieved through supervised pretraining on large-scale video datasets, such as Kinetics [1] or Sports-1M [2]. Although supervised video pretraining is a wellestablished strategy for improving downstream performance, it still depends on manually annotated source videos, and the effectiveness of the learned representations may vary according to the alignment between the source and target domains. Moreover, constructing large-scale labeled video datasets is neither scalable nor sustainable.

Beyond the need for labeled source datasets, supervised video pretraining is also constrained by its reliance on predefined action categories. Models trained with cross-entropy receive supervision through one-hot labels, which provide only a coarse description of the video content. A single action label does not explicitly encode information about the objects, scenes, interactions, and temporal context present in the video. When labeled data are limited, such coarse supervision may be insufficient for learning the broader semantic structure of video content, motivating the use of more informative supervision signals.

Several alternatives to fully supervised video pretraining have been explored to reduce reliance on manually labeled source datasets. Existing transfer learning strategies for video models include 2D-to-3D weight inflation, supervised video pretraining, knowledge distillation from teacher models, and self-supervised pretraining [3]. Among these, knowledge distillation is particularly relevant to the present work, as it enables a student video model to learn from soft supervision produced by a stronger teacher [4]. However, conventional distillation-based initialization methods for video often rely on image-level teachers and transfer knowledge through fixed outputs or predefined label spaces, limiting their ability to exploit the richer semantic information available in unlabeled videos [5], [6].

Recent progress in visual representation learning offers a promising way to address these limitations. The field has increasingly shifted from task-specific models toward foundation models that can be adapted to downstream tasks with limited supervision. Within this broader paradigm, vision-language models (VLMs) are particularly relevant because they align visual and textual representations within a shared semantic embedding space, enabling semantic supervision that extends beyond fixed category labels [7], [8]. Owing to their largescale multimodal pretraining, VLMs have demonstrated strong zero-shot and fine-tuning performance across a broad range of visual recognition and retrieval tasks, and this paradigm has subsequently been extended from images to video [7], [9]– [13]. More recently, the integration of visual encoders with large language models (LLMs) has expanded VLM capabilities beyond discriminative recognition toward open-ended visuallanguage understanding, including captioning, visual reasoning, question answering, and interactive dialogue [14]–[17]. In the video domain, such models can generate descriptions that capture not only actions but also objects, scenes, interactions, and temporal context [18]–[21]. These capabilities make them promising sources of rich semantic supervision for video pretraining and representation learning.

Motivated by these observations, this study proposes an annotation-free video pretraining method based on zero-shot distillation over a VLM-generated pseudo-label space, together with a target-label-set-aware fine-tuning strategy for downstream adaptation. A VLM first generates textual descriptions of unlabeled videos, which are then processed using a natural language processing (NLP) pipeline to construct an automatic pseudo-label space from the semantic information discovered in the videos. Unlike conventional label spaces defined by a single action category, the resulting space captures multiple semantic concepts, including actions, objects, and action–object interactions. A frozen video-language model then produces zero-shot soft target distributions over this space based on video–text similarities, and a student video encoder is pretrained by distilling these targets. When labeled target videos are available, the pretrained encoder is further adapted through target-label-set-aware fine-tuning, where supervised learning with ground-truth labels is combined with zero-shot distillation over the actual target label space. The proposed framework is evaluated on downstream video action recognition under both limited-label fine-tuning and full-data fine-tuning settings.

The main contributions of this work are summarized as follows:

This work proposes an annotation-free video pretraining framework that learns transferable video representations from unlabeled videos, providing a scalable alternative to supervised pretraining on labeled source datasets and hand-designed self-supervised pretext tasks.

• A VLM-guided pseudo-label space construction strategy derives an interpretable semantic vocabulary from VLMgenerated video descriptions, enabling supervision beyond predefined action categories.

• A zero-shot soft distillation objective is formulated over the generated pseudo-label space using a frozen videolanguage model, allowing a student video encoder to learn from soft target distributions without source annotations.

• A target-label-set-aware fine-tuning strategy combines supervised learning from labeled target videos with zeroshot distillation over the target label space for downstream action recognition.

## II. RELATED WORK

## A. Transfer Learning for Video Models

Transfer learning is widely used for video models, particularly when labeled target data are limited. Existing approaches obtain model initialization through several mechanisms, including 2D-to-3D weight inflation, supervised video pretraining, knowledge distillation, and self-supervised pretraining [3]. Supervised pretraining on large-scale datasets such as Kinetics remains a strong baseline, whereas selfsupervised methods learn from unlabeled videos using pretextbased [22]–[25], contrastive [26], [27], or generative [28], [29] objectives. Knowledge distillation provides another direction by transferring supervisory information from a pretrained teacher to a student video model [5], [6]. Among these approaches, the proposed method is most closely related to distillation-based transfer learning for video models. DistInit [6] demonstrated that video representations can be learned from image-pretrained teachers without labeled video data, while subsequent methods extended distillation-based initialization to limited-label video learning by using teacher predictions as auxiliary supervision [5], [30], [31]. These approaches typically transfer knowledge through teacher features or predictions defined over fixed label spaces. In contrast, the proposed method constructs an interpretable pseudo-label space directly from VLM-generated descriptions of unlabeled videos and performs zero-shot distillation over this derived space.

## B. Vision-Language Models

In recent years, vision-language models (VLMs) have attracted considerable attention due to their strong performance across a wide range of visual-language tasks. These models commonly learn to align visual and textual modalities within a shared semantic embedding space. Early models such as CLIP [7] and ALIGN [8] demonstrated that large-scale image– text contrastive learning enables strong zero-shot recognition by comparing visual embeddings with natural-language descriptions. This paradigm has also been extended to the video domain [10]–[12].

Beyond contrastive image–text and video–text representation learning, the integration of visual encoders with large language models (LLMs) has expanded VLMs toward generative and instruction-following multimodal systems. These models, often referred to as vision-language large models (VLLMs) or multimodal large language models (MLLMs), can generate textual descriptions, answer visual questions, and perform open-ended visual-linguistic reasoning [14]–[17], [32]. These capabilities have also been extended to the video domain, where video-centric multimodal models process temporal visual inputs to support open-ended video description, question answering, and dialogue [18]–[21]. Such models therefore provide a promising source of semantic supervision for video representation learning, as they can identify actions, objects, and interactions beyond a predefined set of action categories.

Building on these capabilities, the proposed method performs annotation-free video pretraining by first generating textual descriptions for unlabeled videos using a VLM and then transforming these descriptions into a structured semantic pseudo-label space. A frozen video-language model is subsequently employed to produce zero-shot soft target distributions over the resulting textual vocabulary. Unlike approaches that use VLMs solely as zero-shot classifiers or feature teachers, the proposed framework exploits them at two complementary stages: first, to construct an interpretable supervision space directly from unlabeled video content, and second, to provide soft semantic supervision over this space through zero-shot distillation.

## III. METHOD

The proposed framework learns a video encoder without relying on manually annotated source videos or a predefined source-category space. Its central idea is to derive the supervision space directly from the content of unlabeled videos and then distill instance-specific soft target distributions over this automatically constructed space. To this end, the framework employs two complementary vision-language components with distinct roles. First, an instruction-tuned multimodal model generates open-vocabulary descriptions of the unlabeled videos, from which actions, action-related objects, and action-object interactions are extracted to construct an interpretable pseudo-label vocabulary. Second, a frozen video-language model evaluates the relevance of each pseudolabel to each video and produces a soft target distribution over the resulting vocabulary. A student video encoder is pretrained to reproduce these distributions, thereby learning video representations without human-provided source labels.

For downstream adaptation, the pseudo-label projection head is replaced with a task-specific classification head. The frozen video-language model is then used to generate zeroshot soft targets over the actual target label set, allowing supervised fine-tuning with ground-truth labels to be complemented by VLM-derived supervision. Thus, the proposed framework operates across two distinct label spaces: an automatically discovered pseudo-label space during annotation-free pretraining and a predefined downstream class space during target adaptation. The following subsections describe the construction of the pseudo-label space, zero-shot distillation over this space, and target-label-set-aware fine-tuning with zero-shot distillation, respectively.

## A. Pseudo-Label Space Construction from VLM-Generated Captions

Textual descriptions are first generated for the unlabeled training videos using InternVL2-8B [33], an instruction-tuned multimodal large language model (MLLM). For each video, a fixed number of frames is uniformly sampled and provided to the model with the prompt: “Describe the scene, the objects, and the activity taking place in the video briefly in one or two complete sentences.” The resulting captions provide open-vocabulary descriptions containing action-related and contextual cues.

![](images/9b36c87d8549851def435c7d4a8686b2281c339da21931c3c61fccbd0ab1df63.jpg)  
Fig. 1. Word-cloud visualization of the constructed pseudo-label space.

The free-form captions are then transformed into a structured pseudo-label vocabulary using the natural language processing framework spaCy [34] with the transformer-based English pipeline $\mathsf { e n \_ c o r e \_ w e b \_ t r f } .$ . The pipeline provides part-of-speech tags, lemmatized forms, and dependency relations for each caption, enabling the extraction of actionrelated textual units. Specifically, verbs are extracted as action candidates, while nouns are selected as object candidates only when they occur as direct objects of the detected verbs. The corresponding verb–object pairs are also retained to represent action–object interactions. For example, from a caption such as “a person is playing guitar,” the procedure extracts play, guitar, and play guitar as candidate pseudo-labels. Thus, the extracted object terms are explicitly conditioned on their dependency relations with actions.

The extracted candidates are aggregated across all captions and filtered in two stages. First, document-frequency constraints are applied to remove terms that occur either too rarely or too frequently across the caption corpus. Second, generic or semantically uninformative words and phrases are removed from the extracted candidate vocabulary using a stop list. Candidate stop-list terms were identified with the assistance of ChatGPT [35], after which the resulting list was manually reviewed and revised. All candidates remaining after these filtering steps are used to define the automatically constructed textual pseudo-label space:

$$
\mathcal {T} = \{t _ {1}, t _ {2}, \dots , t _ {M} \}\tag{1}
$$

where M denotes the number of terms retained after filtering, and each $t _ { k }$ represents a textual pseudo-label corresponding to an action, an action-related object, or an action-object interaction discovered from the generated captions. Fig. 1 visualizes the constructed pseudo-label space as a word cloud, with larger terms indicating higher TF-IDF scores.

## B. Zero-Shot Distillation over the Pseudo-Label Space

After constructing the pseudo-label space T , the frozen InternVideo2-CLIP-S [18] model is used to generate a soft target distribution for each unlabeled video. Rather than assigning a single hard pseudo-label, the teacher measures the compatibility of each video with all textual concepts in $\tau$ and produces an instance-specific distribution reflecting their relative relevance. For an unlabeled video $v _ { i } ,$ let $\phi _ { V } ( \cdot )$ and $\phi _ { T } ( \cdot )$ denote the frozen video and text encoders of the teacher model, respectively. The video is encoded as $\phi _ { V } ( v _ { i } )$ while each pseudo-label $t _ { k } \in \mathcal { T }$ is encoded as $\phi _ { T } ( t _ { k } )$ . Their compatibility is measured using cosine similarity:

![](images/c8713aa0ad52d9b927b37946cbc2d8234a349f5f416c7e04da8021d7d6178d76.jpg)  
Fig. 2. Overview of the pseudo-label space construction pipeline. A vision-language model first generates open-vocabulary descriptions of unlabeled video clips. The captions are then processed and filtered to construct a pseudo-label vocabulary comprising actions, objects, and action–object interactions for annotation-free video pretraining.

$$
s _ {i, k} = \frac {\phi_ {V} (v _ {i}) ^ {\top} \phi_ {T} (t _ {k})}{\| \phi_ {V} (v _ {i}) \| \| \phi_ {T} (t _ {k}) \|}\tag{2}
$$

The similarity scores are then converted into a soft distribution over the pseudo-label space by applying a temperaturescaled softmax with temperature parameter τ , which controls the sharpness of the distribution:

$$
q _ {i, k} ^ {\mathrm{VLM}} = \frac {\exp (s _ {i , k} / \tau)}{\sum_ {j = 1} ^ {M} \exp (s _ {i , j} / \tau)}\tag{3}
$$

A student video encoder $G _ { \theta }$ is then trained to match the VLM-derived soft distribution. Given an unlabeled video $v _ { i } .$ , the student first produces a video representation, which is mapped to the pseudo-label space by a projection head $h _ { \psi } ( \cdot )$ . The resulting student-predicted distribution over the M pseudo-labels, denoted by $p _ { i } ^ { \bar { \mathrm { S } } } \in \mathbb { R } ^ { M }$ , is defined as:

$$
p _ {i} ^ {\mathrm{S}} = \operatorname{softmax} \left(h _ {\psi} (G _ {\theta} (v _ {i}))\right)\tag{4}
$$

The annotation-free pretraining objective is formulated as a soft cross-entropy loss between the zero-shot teacher distribution and the student prediction. Let $N _ { u }$ denote the number of unlabeled videos used for pretraining. By minimizing the distillation objective in Eq. (5), the student encoder learns to reproduce the soft target distribution derived from video–text similarities computed by the frozen video-language model.

$$
\mathcal {L} _ {\mathrm{ZSD}} = - \frac {1}{N _ {u}} \sum_ {i = 1} ^ {N _ {u}} \sum_ {k = 1} ^ {M} q _ {i, k} ^ {V L M} \log p _ {i, k} ^ {S}\tag{5}
$$

C. Target-Label-Set-Aware Fine-Tuning with Zero-Shot Distillation

After annotation-free pretraining, the projection head defined over the VLM-generated pseudo-label space is removed and replaced with a task-specific classification head. The pretrained student encoder is then adapted to the downstream action recognition task using both the labeled and unlabeled target training splits. The labeled target videos provide supervision through hard ground-truth labels, while the unlabeled target videos are used to exploit the zero-shot capability of the frozen video-language teacher over the downstream target label space. In this way, supervised fine-tuning is complemented by VLM-derived soft semantic supervision.

For a target dataset with $C$ downstream action classes, the textual target label set is defined as

$$
\mathcal {C} = \{c _ {1}, c _ {2}, \ldots , c _ {C} \}\tag{6}
$$

where each $c _ { j }$ denotes a natural-language prompt constructed from the corresponding class name, such as $\mathbf { \ddot { a } }$ video of a person [action].” Given a target video $v _ { i }$ , the frozen video-language model computes a zero-shot similarity score between the video and each class prompt. Using the frozen video and text encoders $\phi _ { V } ( \cdot )$ and $\phi _ { T } ( \cdot )$ , respectively, the similarity between $v _ { i }$ and $c _ { j }$ is computed as

$$
s _ {i, j} = \frac {\phi_ {V} (v _ {i}) ^ {\top} \phi_ {T} (c _ {j})}{\| \phi_ {V} (v _ {i}) \| \| \phi_ {T} (c _ {j}) \|}\tag{7}
$$

![](images/7d6e38714122232628cc6a13227e4fb5d44b6cc2ce34e2a2a2058c79d51dbd1e.jpg)  
Fig. 3. Overview of zero-shot distillation over the caption-derived pseudo-label space. A frozen video-language model compares each unlabeled video with the pseudo-labels in a joint video–text embedding space to generate soft semantic target distributions. A student video encoder is then pretrained to reproduce these distributions, enabling annotation-free representation learning.

The resulting similarity scores are converted into a soft target distribution over the $C$ downstream action classes using a temperature-scaled softmax with temperature parameter τ :

$$
q _ {i, j} ^ {\mathrm{VLM}} = \frac {\exp (s _ {i , j} / \tau)}{\sum_ {m = 1} ^ {C} \exp (s _ {i , m} / \tau)}\tag{8}
$$

The student prediction over the downstream action classes is obtained using the pretrained video encoder $G _ { \theta }$ and a task-specific classification head $g _ { \psi } ( \cdot )$ . The resulting student prediction over the C target classes, denoted by $p _ { i } ^ { \mathrm { S } } \in \mathbb { R } ^ { C }$ , is defined as

$$
p _ {i} ^ {\mathrm{S}} = \mathrm{softmax} \left(g _ {\psi} (G _ {\theta} (v _ {i}))\right).\tag{9}
$$

The fine-tuning objective combines supervision from the ground-truth target labels with the zero-shot soft targets produced by the frozen video-language teacher. Let $N _ { l }$ denote the number of labeled target videos, and let $y _ { i , j }$ denote the one-hot ground-truth label for class $j .$ . The supervised crossentropy loss is defined as

$$
\mathcal {L} _ {\mathrm{CE}} = - \frac {1}{N _ {l}} \sum_ {i = 1} ^ {N _ {l}} \sum_ {j = 1} ^ {C} y _ {i, j} \log p _ {i, j} ^ {\mathrm{S}}\tag{10}
$$

To complement the hard target labels with the relative class relationships encoded by the frozen teacher, the target-aware zero-shot distillation loss is computed over the unlabeled target videos. Let $N _ { u }$ denote the number of unlabeled target videos used during fine-tuning. The loss is defined as

$$
\mathcal {L} _ {\mathrm{TLSA-ZSD}} = - \frac {1}{N _ {u}} \sum_ {i = 1} ^ {N _ {u}} \sum_ {j = 1} ^ {C} q _ {i, j} ^ {\mathrm{VLM}} \log p _ {i, j} ^ {\mathrm{S}}\tag{11}
$$

Unlike the annotation-free pretraining stage, where the teacher distribution is defined over the automatically generated pseudo-label space T , the distribution in Eq. (8) is defined directly over the downstream action classes in C.

The final fine-tuning objective is given by

$$
\mathcal {L} _ {\mathrm{FT}} = \lambda_ {\mathrm{ft}} \mathcal {L} _ {\mathrm{CE}} + \lambda_ {\mathrm{distill}} \mathcal {L} _ {\mathrm{TLSA-ZSD}}\tag{12}
$$

where $\lambda _ { \mathrm { f t } }$ and $\lambda _ { \mathrm { d i s t i l l } }$ control the contributions of the supervised and distillation loss terms, respectively. By minimizing $\mathcal { L } _ { \mathrm { F T } }$ , the student model learns from the ground-truth target labels while also preserving the embedding structure provided by the frozen video-language teacher.

## IV. EXPERIMENTS

## A. Datasets

The annotation-free pretraining stage uses an unlabeled video pool consisting of training videos from UCF101 [46] and HMDB51 [47], together with 5% subsets of Kinetics-400 and Kinetics-600 [1]. <sup>1</sup> Although the framework can be scaled to larger unlabeled video collections, doing so would increase the computational cost. The present setting therefore evaluates whether effective semantic supervision can be distilled from a relatively compact pretraining pool. During pretraining, no action labels from any of these datasets are used; all videos are treated solely as unlabeled inputs for VLM-based caption generation, pseudo-label space construction, and zero-shot distillation.

TABLE I  
COMPARISON WITH SEMI-SUPERVISED VIDEO ACTION RECOGNITION METHODS ON UCF101 AND HMDB51 UNDER DIFFERENT LABELED-DATA REGIMES. RESULTS ARE REPORTED AS VIDEO-LEVEL TOP-1 CLASSIFICATION ACCURACY.

<table><tr><td rowspan="2">Method</td><td colspan="2">UCF101</td><td colspan="2">HMDB51</td><td rowspan="2">Distillation</td><td rowspan="2">Modality</td><td rowspan="2">Backbone</td></tr><tr><td>1%</td><td>10%</td><td>40%</td><td>50%</td></tr><tr><td>Supervised</td><td>8.2</td><td> $24.0 [30]^a$ </td><td> $18.0 [30]^a$ </td><td> $30.7 [30]^a$ </td><td>✕</td><td>V</td><td>R3D-18</td></tr><tr><td>VideoSSL [30]</td><td>-</td><td>42.0</td><td>32.7</td><td>36.2</td><td>√</td><td>V</td><td>R3D-18</td></tr><tr><td>DANet [5]</td><td>-</td><td>64.6</td><td>-</td><td>-</td><td>√</td><td>V</td><td>R3D-18</td></tr><tr><td>CMPL [36]</td><td>23.8</td><td>67.6</td><td>-</td><td>-</td><td>✕</td><td>V</td><td>R3D-18</td></tr><tr><td>LTG [37]</td><td>-</td><td>62.4</td><td>46.5</td><td>48.4</td><td>✕</td><td>V+TG</td><td>R3D-18</td></tr><tr><td>MvPL [38] $^b$ </td><td>-</td><td>55.5</td><td>30.5</td><td>33.9</td><td>✕</td><td>V+TG+F</td><td>R3D-18</td></tr><tr><td>L2A [39]</td><td>-</td><td>60.1</td><td>42.1</td><td>46.3</td><td>√</td><td>V</td><td>R3D-18</td></tr><tr><td>ActorCutMix [40]</td><td>-</td><td>40.2</td><td>32.9</td><td>38.2</td><td>✕</td><td>V</td><td>R(2+1)D-34</td></tr><tr><td>FD-VLM [31]</td><td>24.2</td><td>62.4</td><td>-</td><td>34.5</td><td>√</td><td>V</td><td>R3D-18</td></tr><tr><td>TimeBalance [41]</td><td>29.1</td><td>69.8</td><td>49.8</td><td>51.4</td><td>✕</td><td>V</td><td>R3D-18</td></tr><tr><td>LEVIL</td><td>54.3</td><td>73.3</td><td>51.8</td><td>55.6</td><td>√</td><td>V</td><td>R3D-18</td></tr></table>

<sup>a</sup> Results are taken from the corresponding cited works. <sup>b</sup> Reimplementation results reported in [37].

TABLE II  
COMPARISON OF VIDEO PRETRAINING STRATEGIES FOR ACTION RECOGNITION ON UCF101 AND HMDB51. RESULTS ARE REPORTED AS VIDEO-LEVEL TOP-1 CLASSIFICATION ACCURACY.

<table><tr><td>Pretrain</td><td>Backbone</td><td>Strategy</td><td>N × H/W</td><td>UCF101</td><td>HMDB51</td></tr><tr><td>None</td><td>R3D-18</td><td>None</td><td>16×112</td><td>42.4 [24]a</td><td>25.3 [6]a</td></tr><tr><td>None</td><td>R3D-18</td><td>ImageNet Inflation</td><td>16×112</td><td>74.3 [5]a</td><td>-</td></tr><tr><td>UCF/HMDB</td><td>R3D-18</td><td>DANet [5]</td><td>8×112</td><td>76.8</td><td>-</td></tr><tr><td>Kinetics+Sports-1M</td><td>R(2+1)D-18</td><td>DistInit [6]</td><td>32×112</td><td>85.7</td><td>54.9</td></tr><tr><td>Kinetics+Sports-1M</td><td>R(2+1)D-18</td><td>DistInit [6]</td><td>8 × 112</td><td>-</td><td>40.3</td></tr><tr><td>Kinetics+Sports-1M</td><td>R3D-18</td><td>DistInit [6]</td><td>8 × 112</td><td>-</td><td>39.9</td></tr><tr><td>Kinetics</td><td>R3D-18</td><td>Supervised Pretraining</td><td>16×112</td><td>87.8 [42]a</td><td>59.3 [42]a</td></tr><tr><td>Kinetics</td><td>R3D-18</td><td>Supervised Pretraining</td><td>8×224</td><td>81.7</td><td>61.2</td></tr><tr><td>Kinetics</td><td>R(2+1)D</td><td>Supervised Pretraining</td><td>16×112</td><td>96.8 [25]a</td><td>74.5 [25]a</td></tr><tr><td>Sports-1M</td><td>C3D</td><td>Supervised Pretraining</td><td>16×112</td><td>82.3 [23]a</td><td>-</td></tr><tr><td>Kinetics</td><td>R3D-18</td><td>3DRotNet [23]</td><td>16×112</td><td>62.9</td><td>33.7</td></tr><tr><td>MiT</td><td>R3D-18</td><td>3DRotNet [23]</td><td>16×112</td><td>62.8</td><td>29.6</td></tr><tr><td>UCF</td><td>C3D</td><td>PMAS [43]</td><td>16×112</td><td>58.8</td><td>32.6</td></tr><tr><td>Kinetics</td><td>C3D</td><td>PMAS [43]</td><td>16×112</td><td>61.2</td><td>33.4</td></tr><tr><td>Kinetics</td><td>C3D</td><td>3D ST-Puzzle [24]</td><td>16×112</td><td>60.6</td><td>28.3</td></tr><tr><td>Kinetics</td><td>R3D-18</td><td>3D ST-Puzzle [24]</td><td>16×112</td><td>65.8</td><td>33.7</td></tr><tr><td>UCF</td><td>R(2+1)D-18</td><td>ClipOrder [25]</td><td>16×112</td><td>72.4</td><td>30.9</td></tr><tr><td>UCF</td><td>R3D-18</td><td>ClipOrder [25]</td><td>16×112</td><td>64.9</td><td>29.5</td></tr><tr><td>Kinetics</td><td>R(2+1)D-18</td><td>PacePred [44]</td><td>16×112</td><td>77.1</td><td>36.6</td></tr><tr><td>Kinetics</td><td>S3D-G</td><td>SpeedNet [22]</td><td>64×224</td><td>81.1</td><td>48.8</td></tr><tr><td>Kinetics</td><td>I3D</td><td>SpeedNet [22]</td><td>64×224</td><td>66.7</td><td>43.7</td></tr><tr><td>Kinetics</td><td>R(2+1)D-18</td><td>VideoMoCo [26]</td><td>32×112</td><td>78.7</td><td>49.2</td></tr><tr><td>Kinetics</td><td>R3D-18</td><td>VideoMoCo [26]</td><td>32×112</td><td>74.1</td><td>43.6</td></tr><tr><td>UCF</td><td>R3D-18</td><td>TCLR [27]</td><td>16×112</td><td>82.4</td><td>52.9</td></tr><tr><td>Kinetics</td><td>R3D-18</td><td>TCLR [27]</td><td>16×112</td><td>84.1</td><td>53.6</td></tr><tr><td>Kinetics</td><td>R3D-18</td><td>CSTP [45]</td><td>16×112</td><td>70.5</td><td>34.4</td></tr><tr><td>UCF+HMDB+Kineticsb</td><td>R3D-18</td><td>LEViLc</td><td>8×224</td><td>72.5</td><td>56.2</td></tr></table>

<sup>a</sup> Results are taken from the corresponding cited works. <sup>b</sup> Uses 5% subsets of Kinetics-400 and Kinetics-600. <sup>c</sup> ZSD pretraining only.

The proposed method is evaluated on the UCF101 and HMDB51 action recognition benchmarks. For the limitedlabel experiments, the labeled/unlabeled splits and labeleddata ratios are adopted from [48]. Performance is measured by fine-tuning on the labeled training split of each benchmark and evaluating on the corresponding test split. For full-data fine-tuning, the official split 1 of each target dataset is used.

## B. Implementation Details

All experiments are performed on a single NVIDIA GeForce RTX 5090 GPU. InternVL2-8B [33] is used only for caption generation, where 8 uniformly sampled frames from each video are resized to 448 × 448 and provided to the model together with the captioning prompt. Zero-shot similarity scores are computed using the frozen InternVideo2-CLIP-S model [18]. The pseudo-label vocabulary is constructed using spaCy [34] with its transformer-based English pipeline, en\_core\_web\_trf, followed by TF-IDF filtering with a minimum document frequency of 5 and a maximum document frequency of 0.25. After filtering, the resulting pseudolabel vocabulary contains 2391 textual pseudo-labels. During pretraining and fine-tuning, a single clip of 8 consecutive frames is randomly sampled from each video and resized to $2 2 4 \times 2 2 4$ . During evaluation, 10 temporal clips are sampled deterministically, and video-level predictions are obtained by averaging the clip-level scores.

The student model is implemented as a 3D ResNet-18 in $\mathrm { P y } .$ Torch [49] and initialized from scratch with weights=None. During annotation-free pretraining, the student is optimized with AdamW using a learning rate of $1 0 ^ { - 4 }$ for 100 epochs, with a batch size of 80. Soft targets are obtained by applying a temperature-scaled softmax to the InternVideo2-CLIP-S similarity scores with $\tau = 1 / 5 0$ . During semi-supervised finetuning, each mini-batch contains 40 labeled and 40 unlabeled videos. The supervised and target-aware distillation losses are equally weighted by setting $\lambda _ { \mathrm { f t } } = 0 . 5$ and $\lambda _ { \mathrm { d i s t i l l } } = 0 . 5$ . During downstream finetuning, the same optimizer and learning rate as in the pretraining stage are used, and all finetuning experiments are conducted for 20 epochs.

## C. Experimental Results

The proposed method is evaluated for downstream action recognition under both semi-supervised and fully supervised settings. Table 1 presents the semi-supervised results on UCF101 and HMDB51 under different labeled-data regimes in terms of video-level Top-1 classification accuracy. In this setting, the complete proposed framework is applied, including annotation-free pretraining followed by target-label-setaware zero-shot distillation during downstream fine-tuning. Therefore, the reported results reflect the overall performance of the two-stage framework. To provide additional context, Table 1 also indicates whether each method uses distillation, together with its input modality and backbone architecture. The supervised baselines correspond to training the video model from scratch using only the available labeled subset and without any additional supervision, and therefore serve as lower-bound references for assessing the benefit of the compared semi-supervised methods under the same labeleddata regime.

The compared methods represent several directions in semisupervised video action recognition. VideoSSL [30], one of the earliest semi-supervised action recognition frameworks, transfers knowledge from an image-based teacher to a video student, while DANet [5] extends this paradigm by incorporating multiple teachers and contrastive objectives. LTG [37] and MvPL [38] incorporate motion-related cues through temporal gradients and optical flow, respectively. L2A [39] and ActorCutMix [40] focus on augmentation strategies for semisupervised training. TimeBalance [41] employs spatial and temporal teachers trained with self-supervised pretext tasks and dynamically balances their predictions according to the input video. CMPL [36], in contrast, uses a primary backbone together with a lightweight auxiliary network with a different architectural design. These architecturally distinct networks learn complementary representations and generate pseudolabels for each other through cross-model pseudo-labeling. Overall, the compared methods span several major directions, including teacher–student learning, motion-based supervision, augmentation-based training, and self-supervised pretext tasks. To provide a more controlled comparison, Table 1 is restricted to methods using CNN-based video backbones; the transformer-based methods SVFormer [50] and SeFAR [48] are therefore excluded.

FD-VLM [31] is the most closely related method because it also employs video–text multimodal supervision. However, FD-VLM relies on feature-level distillation followed by fine-tuning with hard ground-truth labels. In such a setting, the transferred knowledge remains embedded in a highdimensional feature space, making it difficult to explicitly inspect or control the semantic information distilled to the student. Moreover, because VLM-derived supervision is not explicitly retained during downstream fine-tuning, the model may gradually shift toward the limited hard-label supervision, increasing the risk of overfitting in low-label regimes. In contrast, the proposed method performs pretraining over an automatically constructed pseudo-label space composed of interpretable textual concepts. During downstream adaptation, target-label-set-aware zero-shot distillation further complements the hard labels with soft targets defined over the actual action classes, allowing VLM-derived semantic guidance to be maintained throughout fine-tuning.

As shown in Table 1, LEViL achieves the best performance among all compared methods across every evaluated label regime on both UCF101 and HMDB51. These results demonstrate the effectiveness of the proposed framework under varying levels of label availability, with particularly strong gains in the extremely low-label setting.

Table 2 presents the fully supervised fine-tuning results, where all labeled training videos of the target dataset are used. In this setting, the proposed annotation-free pretraining strategy is evaluated solely as a weight initialization mechanism, without target-label-set-aware zero-shot distillation. The learned initialization is compared with alternative initialization strategies for video models. The results on UCF101 and HMDB51 demonstrate that the representations learned during annotation-free pretraining provide an effective initialization.

Overall, LEViL consistently improves upon supervised training from scratch under limited-label settings and provides transferable representations under full-data fine-tuning. Importantly, these results are obtained without using the complete Kinetics datasets for pretraining. Only 5% subsets of both Kinetics-400 and Kinetics-600 are incorporated into the unlabeled pretraining pool. The results therefore indicate that transferable video representations can be learned from a comparatively modest unlabeled video collection, supporting the proposed method as a practical alternative to full-scale supervised video pretraining.

## V. DISCUSSION

Label-efficient training of video models has become increasingly important because video annotation is costly and difficult to scale. Compared with images, videos contain temporal information that increases both annotation effort and training complexity. As a result, video models trained from scratch with limited data are more susceptible to overfitting, motivating the use of alternative supervision sources that can exploit unlabeled videos.

The proposed framework addresses this problem by combining annotation-free pretraining with target-label-set-aware fine-tuning. During pretraining, VLM-generated captions are converted into an interpretable pseudo-label space, and the student learns from soft distributions over this space. During downstream adaptation, zero-shot classification is performed over the actual target classes, and the resulting soft targets complement the available hard labels. In this formulation, the distilled knowledge is represented through a textual vocabulary rather than being embedded in a high-dimensional feature space, making the transferred information easier to inspect and control. In addition, the frozen VLM is used only during training for caption generation and soft-target construction. At test time, predictions are produced solely by the student video model, introducing no additional inference cost.

The results are also notable because the proposed method learns transferable video representations from a relatively small unlabeled pretraining pool. Despite using substantially less source data than full-scale video pretraining, the complete framework improves performance under limited-label settings, while the annotation-free pretraining stage alone provides an effective initialization for full-data fine-tuning. Since the framework does not depend on source annotations or a predefined source label space, it can also be extended to larger or domain-specific unlabeled video collections.

The proposed framework also has aspects that can be further improved. Since the pseudo-label space is constructed from VLM-generated captions, its quality depends on the quality of these captions. If the generated descriptions fail to capture relevant video content, the extracted pseudo-labels may be less informative. In this work, the pseudo-label construction process is kept simple and general; however, this may not be sufficient for all datasets or application domains. More refined prompt design and pseudo-label selection strategies may therefore improve the quality of the generated supervision. Since the framework is modular, stronger captioning and video-language models can also be incorporated as they become available, potentially improving its performance.

## VI. CONCLUSION

This work presented an annotation-free video pretraining framework that leverages the joint video–language embedding space of VLMs to construct pseudo-label supervision, together with a target-label-set-aware fine-tuning strategy. The proposed method constructs a textual pseudo-label space from captions generated for unlabeled videos and pretrains a student video encoder by distilling zero-shot soft distributions produced by a frozen video-language model. During downstream adaptation, supervised learning is complemented by targetlabel-set-aware zero-shot distillation over the actual action label set.

Experiments on UCF101 and HMDB51 demonstrate the effectiveness of the proposed framework under both limitedlabel and full-label settings. The results show that transferable video representations can be learned without manually annotated source labels or predefined source action categories, even from a relatively small unlabeled pretraining pool. Overall, the proposed method provides a scalable alternative to conventional supervised video pretraining by learning transferable representations without manually annotated source videos.

## ACKNOWLEDGMENT

The author acknowledges the use of ChatGPT and Gemini for language editing and readability improvements, as well as for assistance in generating and visually refining the schematic illustrations in Figs. 2 and 3. ChatGPT was also used to assist with the identification of candidate stop-list terms during the pseudo-label space construction described in Section III-A. All generative AI-assisted content was reviewed and verified by the author.

## REFERENCES

[1] W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan, F. Viola, T. Green, T. Back, P. Natsev et al., “The kinetics human action video dataset,” arXiv preprint arXiv:1705.06950, 2017.

[2] A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar, and L. Fei-Fei, “Large-scale video classification with convolutional neural networks,” in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2014, pp. 1725–1732.

[3] A. C¸ elik, “Transfer learning for video action recognition: A comparative overview of weight initialization strategies: A. c¸elik,” Signal, Image and Video Processing, vol. 19, no. 15, p. 1306, 2025.

[4] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.

[5] G. Gao, Z. Liu, G. Zhang, J. Li, and A. K. Qin, “Danet: Semi-supervised differentiated auxiliaries guided network for video action recognition,” Neural Networks, vol. 158, pp. 121–131, 2023.

[6] R. Girdhar, D. Tran, L. Torresani, and D. Ramanan, “Distinit: Learning video representations without a single labeled video,” in Proceedings of the IEEE/cvf international conference on computer vision, 2019, pp. 852–861.

[7] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[8] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in International conference on machine learning. PMLR, 2021, pp. 4904–4916.

[9] M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem, I. Alabdulmohsin, N. Parthasarathy, T. Evans, L. Beyer, Y. Xia, B. Mustafa et al., “Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features,” arXiv preprint arXiv:2502.14786, 2025.

[10] M. Bain, A. Nagrani, G. Varol, and A. Zisserman, “Frozen in time: A joint video and image encoder for end-to-end retrieval,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 1728–1738.

[11] ——, “A clip-hitchhiker’s guide to long video retrieval,” arXiv preprint arXiv:2205.08508, 2022.

[12] H. Rasheed, M. U. Khattak, M. Maaz, S. Khan, and F. S. Khan, “Finetuned clip models are efficient video learners,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 6545–6554.

[13] D. Chen, M. Shukor, T. Moutakanni, W. Chung, J. Yu, T. Kasarla, Y. Bang, A. Bolourchi, Y. LeCun, and P. Fung, “Vl-jepa: Joint embedding predictive architecture for vision-language,” arXiv preprint arXiv:2512.10942, 2025.

[14] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a visual language model for few-shot learning,” Advances in neural information processing systems, vol. 35, pp. 23 716–23 736, 2022.

[15] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, “Qwen-vl: A frontier large vision-language model with versatile abilities,” arXiv preprint arXiv:2308.12966, vol. 1, no. 2, p. 3, 2023.

[16] Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, Q. Ye, and F. Wei, “Grounding multimodal large language models to the world,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 51 575–51 598.

[17] W. Dai, J. Li, D. Li, A. Tiong, J. Zhao, W. Wang, B. Li, P. N. Fung, and S. Hoi, “Instructblip: Towards general-purpose vision-language models with instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 49 250–49 267, 2023.

[18] Y. Wang, K. Li, X. Li, J. Yu, Y. He, G. Chen, B. Pei, R. Zheng, Z. Wang, Y. Shi et al., “Internvideo2: Scaling foundation models for multimodal video understanding,” in European conference on computer vision. Springer, 2024, pp. 396–416.

[19] K. Li, Y. He, Y. Wang, Y. Li, W. Wang, P. Luo, Y. Wang, L. Wang, and Y. Qiao, “Videochat: Chat-centric video understanding,” Science China Information Sciences, vol. 68, no. 10, p. 200102, 2025.

[20] M. Maaz, H. Rasheed, S. Khan, and F. Khan, “Video-chatgpt: Towards detailed video understanding via large vision and language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 12 585– 12 602.

[21] H. Zhang, X. Li, and L. Bing, “Video-llama: An instruction-tuned audiovisual language model for video understanding,” in Proceedings of the 2023 conference on empirical methods in natural language processing: system demonstrations, 2023, pp. 543–553.

[22] S. Benaim, A. Ephrat, O. Lang, I. Mosseri, W. T. Freeman, M. Rubinstein, M. Irani, and T. Dekel, “Speednet: Learning the speediness in videos,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 9922–9931.

[23] L. Jing, X. Yang, J. Liu, and Y. Tian, “Self-supervised spatiotemporal feature learning via video rotation prediction,” arXiv preprint arXiv:1811.11387, 2018.

[24] D. Kim, D. Cho, and I. S. Kweon, “Self-supervised video representation learning with space-time cubic puzzles,” in Proceedings of the AAAI conference on artificial intelligence, vol. 33, no. 01, 2019, pp. 8545– 8552.

[25] D. Xu, J. Xiao, Z. Zhao, J. Shao, D. Xie, and Y. Zhuang, “Selfsupervised spatiotemporal learning via video clip order prediction,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 10 334–10 343.

[26] T. Pan, Y. Song, T. Yang, W. Jiang, and W. Liu, “Videomoco: Contrastive video representation learning with temporally adversarial examples,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 11 205–11 214.

[27] I. Dave, R. Gupta, M. N. Rizve, and M. Shah, “Tclr: Temporal contrastive learning for video representation,” Computer Vision and Image Understanding, vol. 219, p. 103406, 2022.

[28] Z. Tong, Y. Song, J. Wang, and L. Wang, “Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training,” Advances in neural information processing systems, vol. 35, pp. 10 078– 10 093, 2022.

[29] H. Yang, D. Huang, B. Wen, J. Wu, H. Yao, Y. Jiang, X. Zhu, and Z. Yuan, “Motionmae: Self-supervised video representation learning with motion-aware masked autoencoders.” in BMVC, 2024.

[30] L. Jing, T. Parag, Z. Wu, Y. Tian, and H. Wang, “Videossl: Semisupervised learning for video classification,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2021, pp. 1110–1119.

[31] A. Celik, A. Kuc¸¨ ukmanisa, and O. Urhan, “Feature distillation from¨ vision-language model for semisupervised action classification,” Turkish Journal of Electrical Engineering and Computer Sciences, vol. 31, no. 6, pp. 1129–1145, 2023.

[32] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu et al., “Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 24 185–24 198.

[33] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma et al., “How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites,” Science China Information Sciences, vol. 67, no. 12, p. 220101, 2024.

[34] M. Honnibal, I. Montani, S. V. Landeghem, and A. Boyd, “spaCy: Industrial-strength natural language processing in python,” 2020.

[35] OpenAI, “ChatGPT,” [Online]. Available: https://chatgpt.com/, 2026, accessed: Jun. 18, 2026.

[36] Y. Xu, F. Wei, X. Sun, C. Yang, Y. Shen, B. Dai, B. Zhou, and S. Lin, “Cross-model pseudo-labeling for semi-supervised action recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 2959–2968.

[37] J. Xiao, L. Jing, L. Zhang, J. He, Q. She, Z. Zhou, A. Yuille, and Y. Li, “Learning from temporal gradient for semi-supervised action recognition,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 3252–3262.

[38] B. Xiong, H. Fan, K. Grauman, and C. Feichtenhofer, “Multiview pseudo-labeling for semi-supervised learning from video,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 7209–7219.

[39] S. N. Gowda, M. Rohrbach, F. Keller, and L. Sevilla-Lara, “Learn2augment: learning to composite videos for data augmentation in action recognition,” in European conference on computer vision. Springer, 2022, pp. 242–259.

[40] Y. Zou, J. Choi, Q. Wang, and J.-B. Huang, “Learning representational invariances for data-efficient action recognition,” Computer Vision and Image Understanding, vol. 227, p. 103597, 2023.

[41] I. R. Dave, M. N. Rizve, C. Chen, and M. Shah, “Timebalance: Temporally-invariant and temporally-distinctive video representations for semi-supervised action recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 2341–2352.

[42] H. Kataoka, T. Wakamiya, K. Hara, and Y. Satoh, “Would megascale datasets further enhance spatiotemporal 3d cnns?” arXiv preprint arXiv:2004.04968, 2020.

[43] J. Wang, J. Jiao, L. Bao, S. He, Y. Liu, and W. Liu, “Self-supervised spatio-temporal representation learning for videos by predicting motion and appearance statistics,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4006–4015.

[44] J. Wang, J. Jiao, and Y.-H. Liu, “Self-supervised video representation learning by pace prediction,” in European conference on computer vision. Springer, 2020, pp. 504–521.

[45] Y. Zhang, L.-M. Po, X. Xu, M. Liu, Y. Wang, W. Ou, Y. Zhao, and W.-Y. Yu, “Contrastive spatio-temporal pretext learning for selfsupervised video representation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 3, 2022, pp. 3380–3389.

[46] K. Soomro, A. R. Zamir, and M. Shah, “Ucf101: A dataset of 101 human actions classes from videos in the wild,” arXiv preprint arXiv:1212.0402, 2012.

[47] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre, “Hmdb: a large video database for human motion recognition,” in 2011 International conference on computer vision. IEEE, 2011, pp. 2556–2563.

[48] Y. Huang, H. Chen, Z. Xu, Z. Jia, H. Sun, and D. Shao, “Sefar: Semisupervised fine-grained action recognition with temporal perturbation and learning stabilization,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 4, 2025, pp. 3833–3841.

[49] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.

[50] Z. Xing, Q. Dai, H. Hu, J. Chen, Z. Wu, and Y.-G. Jiang, “Svformer: Semi-supervised video transformer for action recognition,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 18 816–18 826.