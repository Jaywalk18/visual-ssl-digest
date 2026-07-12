# XOV-Action: Towards Generalizable Open-Vocabulary Action Recognition

Kun-Yu Lin, Henghui Ding, Jia-Run Du, Jiaming Zhou,

Yi-Xing Peng, Yu-Ming Tang, Zhilin Zhao, Chen Change Loy, Wei-Shi Zheng

Abstract—Inspired by the impressive success of image-text foundation models, recent works have proposed to adapt these foundation models to video data, leading to efficient and effective video models for open-vocabulary action recognition. However, through a comprehensive evaluation, our work finds that state-ofthe-art open-vocabulary action recognition models still struggle with generalization to video domains that they have not encountered. To address this limitation, we introduce generalizable open-vocabulary action recognition, which aims to develop action recognition models capable of generalizing to both novel action categories and unseen video domains. Our work contributes a novel model named XOV-Action to overcome two critical challenges: (1) understanding novel action concepts of open-set categories, and (2) mitigating the scenario discrepancy between training and test datasets. Specifically, XOV-Action first proposes to capture diverse action-related concepts by learning diversified elaboration representations, which enables better generalization to open-set action categories. Second, XOV-Action learns sceneagnostic video representations to overcome the scene bias, which improves the generalization in unseen video domains. Additionally, to evaluate models in generalizable open-vocabulary action recognition, we contribute a new cross-domain action benchmark named XOVABench, which covers multiple video domains with varying degrees of gaps and consists of both closed-set and open-set action categories. Extensive quantitative and qualitative experiments demonstrate that our proposed XOV-Action can effectively improve action recognition performance for both closedset and open-set categories across video domains. The benchmark is available at https://github.com/KunyuLin/XOV-Action/.

Index Terms—Action recognition, open-vocabulary action recognition, generalizable open-vocabulary action recognition

## I. INTRODUCTION

CTION recognition aims to recognize what actions hucations in surveillance systems, health monitoring, etc [1], [2]. Recently, inspired by the impressive success of imagetext foundation models (e.g., CLIP [3]) across various image understanding tasks, pioneer works propose to adapt these models to video data for action recognition [4]–[9]. Different from traditional action models that focus on closed-set recognition [10]–[14], this new image-text foundation-based paradigm leads to efficient video learners with remarkable open-vocabulary action recognition abilities, i.e., they achieve state-of-the-art performance for both closed-set and open-set action categories<sup>1</sup> on various video datasets with moderate training cost. Such open-vocabulary abilities significantly improve the practical utility of action recognition models.

![](images/2e8dc63ba100c12d7065437c378e88fd511388ded3d860b3a77b15e9ef282ed9.jpg)  
Fig. 1. The demonstration of our proposed generalizable open-vocabulary action recognition task, which aims to develop models capable of recognizing both close-set and open-set categories across video domains, e.g., from a outdoor video domain to a dark video domain. Red texts denote the closedset categories, and blue texts denote the open-set categories. Best viewed in color.

In this work, we study an underexplored task termed generalizable open-vocabulary action recognition, where training and test videos are drawn from different domains and may encompass non-overlapping action categories. As shown in Figure 1, the goal of this task is to develop open-vocabulary action recognition models capable of generalizing to test domains not encountered during training. Such generalization abilities are crucial for action recognition models, since models often suffer from environment, viewpoint, and sensor changes when deployed in real-world applications [15], [16]. For example, surveillance systems will encounter actions performed under illumination shifts caused by day-night change or weather change. Therefore, we expect that action recognition models with open-vocabulary abilities can robustly address video domain shifts.

![](images/8ce94c95b5104f43ae2df121c3a9ff0f399c7fd6556aa7db10afb1275b16e334.jpg)  
Fig. 2. We conduct a evaluation for state-of-the-art open-vocabulary action recognition models on four test datasets, namely UCF [17], HMDB [18], ARID [19] and NEC-Dr [20]. These four test datasets exhibiting various levels of domain gaps in comparison to the training dataset, i.e., UCF has a small gap, HMDB has a moderate gap, ARID and NEC-Dr have large domain gaps. For each test dataset, we report the accuracy of closed-set and openset action categories, which are identified according to the training categories in Kinetics400 [13]. As shown in the figure, previous state-of-the-art openvocabulary models exhibit limited performance when recognizing actions in unseen test domains. Please refer to Table III for the full results. Best viewed in color.

By conducting a comprehensive evaluation on state-ofthe-art open-vocabulary action recognition models, our work reveals that, despite their remarkable success, state-of-theart models still struggle with generalization to unseen video domains. As shown in Figure 2 and Table III, the open-set recognition performance of previous open-vocabulary models [5], [9], [21] are far from reaching saturation even in domains with moderate domain gaps, e.g., the best open-set accuracy on HMDB is only 42.54%. In addition, our evaluation shows that previous open-vocabulary models exhibit significantly degraded performance on closed-set categories when tested in domains with large domain gaps. For example, even the top-performing Kinetics400-trained model achieves merely 52.74% closed-set accuracy on dark videos in ARID, due to the large scenario gap between the Kinetics and ARID domains. Overall, generalizable open-vocabulary action recognition is a challenging task, since it requires recognizing both closed-set and open-set action categories in unseen video domains.

In principle, there are two critical challenges in the generalizable open-vocabulary action recognition task, namely novel action concepts of open-set categories and scenario discrepancy between training and test videos:

1) The first challenge lies in understanding the novel action concepts of open-set categories, which is fundamental to category generalization. Specifically, previous open-vocabulary models [4], [8], [9] strongly rely on the text representations of category names to recognize open-set action categories. However, action concepts encoded by these simple name texts of open-set categories are usually vague and unfamiliar to models, since video samples of open-set categories are not accessible for model training. For example, it is challenging for a recognition model to distinguish the action “long jump” from the action “high jump” solely based on name texts, if the model do not see any videos of these two categories during training. This is because these two name texts are similar as both categories share the word “jump”, and a model can hardly understand the meaning of “long” and “high” in the context of the respective actions without seeing any video samples.

2) Secondly, in generalizable open-vocabulary action recognition, domain gap is a major obstacle to model generalization. It is an important challenge widely discussed in traditional action recognition works [15], [22], but remains underexplored in open-vocabulary works. To bridge the domain gap, we observe that two videos from different domains usually have distinct scenarios, and this can easily cause scene bias during model training. Specifically, the scene bias arises from the strong associations between actions and specific scenarios in training videos (aka., spurious correlation [23]–[28]), and it would hinder model generalization across domains due to the scenario differences. For example, as shown in Figure 1, if humans always perform jumping on track-and-field grounds in the training domain, the trained models are prone to recognize this action by the track-and-field grounds, since static scenes are much easier to fit [23], [24], [26]. However, humans may perform jumping in hallways in test domains, thus recognizing “jump” based on track-and-field grounds would result in recognition errors across domains.

In this work, we propose a novel model, named XOV-Action, to overcome the above two challenges for generalizable open-vocabulary action recognition. Firstly, to boost the understanding of novel action concepts for open-set categories, our XOV-Action proposes to capture diverse action-related concepts from videos by Diversified Elaboration Representation Learning. By elaborating action concepts using multiple textual descriptions, XOV-Action learns diverse concepts during training and associates them with open-set categories during testing, thereby improving the recognition of openset action categories. Secondly, to mitigate the scenario discrepancy, our XOV-Action proposes to learn scene-agnostic video representations by Scene-Aware Video-text Alignment. By introducing scene-encoded text prompts, XOV-Action distinguishes video representations apart from scene-encoded text representations, which encourages the video encoder to downweight the attention on scene information and thus pay more attention to action information, thereby improving the generalization across domains.

In addition, to evaluate models for generalizable openvocabulary action recognition, we establish a new crossdomain action benchmark named XOVABench, which covers multiple video domains with varying degrees of gaps and consists of both closed-set and open-set action categories. Extensive quantitative and qualitative experiments demonstrate that our proposed XOV-Action can effectively improve the action recognition performance for both closed-set and openset categories across domains.

In summary, our contributions are listed as follows:

(1) Our proposed model, named XOV-Action, can capture rich action-related concepts from videos by Diversified Elaboration Representation Learning. By leveraging textual descriptions of action categories, XOV-Action learns diverse concepts during training and associates them with open-set categories during test. Extensive experiments demonstrate the effectiveness of our XOV-Action in recognizing novel actions across domains.

(2) Our proposed XOV-Action also learns scene-agnostic video representations by Scene-Aware Video-text Alignment, encouraging the video encoder to downweight the attention on scene information and thus pay more attention to inherent action information. Extensive experiments demonstrate the effectiveness of our XOV-Action in bridging domain gaps.

(3) We establish a CROSS-domain Open-Vocabulary Action recognition Benchmark, dubbed XOVABench, which consists of four test domains exhibiting various levels of domain gaps in comparison to the training domains. We identify closedset and open-set categories for each test domain, and thus provide a comprehensive way to evaluate open-vocabulary action models across various situations.

## II. RELATED WORK

## A. Action Recognition

Action recognition aims to recognize human actions in videos, which has broad applications in real-world [1], [2], [29]. In the past decade, motivated by the success of deep learning [30]–[36], many video classification architectures have been proposed. These architectures can be primarily categorized into 2D CNNs, 3D CNNs and Video Transformers. Typically, 2D CNNs adopt 2D convolution for spatial modeling, and conduct temporal modeling beyond spatial modeling [11], [37]–[39] or embed temporal shift into spatial modeling [12], [40], [41]. 3D CNNs extends 2D convolution to 3D convolution for adapting video data [13], [42]–[47]. By adopting attention mechanisms, Video Transformers expand the receptive field of 2D and 3D CNNs, leading to remarkable performance [14], [48]–[54]. Although above models show promising performance for closed-set action recognition, they usually lack the ability to recognize open-set categories.

Recently, inspired by the success of image-text foundation models (especially CLIP [3]), some pioneer works propose adapting these models to video data for action recognition [4]– [9], [21], [55], [55]–[66]. Owing to the image-text alignment power of image-text foundation models, video learners are endowed with remarkable open-vocabulary action recognition abilities with moderate training cost. These methods adapt image-text foundation models to video data in various ways. For example, ActionCLIP [4] stacks temporal fusion layers on top of the image encoder for modeling temporal dynamics, Ju et al. [6] cooperate continuous prompting with temporal Transformer, and X-CLIP [5] proposes cross-frame communication attention for temporal modeling. Different from other works, two pioneer works OpenVCLIP [9] and FROSTER [60] propose to improve open-set generalization by harnessing the power of raw CLIP. In our work, we focus on the underexplored cross-domain scenarios of open-vocabulary action recognition, which aims to develop generalizable openvocabulary action recognition models for unseen video domains.

## B. Generalizable Action Recognition

Generalizable action recognition studies the generalization capabilities of action recognition models. In this field, zeroshot action recognition [67] aims to recognize novel categories of actions not encountered during training. Existing works typically learn alignment models between the visual space of videos and the semantic space of class descriptions [68]– [74], and this zero-shot task establishes a solid foundation for open-vocabulary action recognition. Additionally, crossdomain action recognition aims to learn video classification models by transferring knowledge from source domains to target domains. This mainly includes two tasks, namely domain-adaptive action recognition and domain-generalizable action recognition. In domain-adaptive action recognition [15], [22], [75]–[80], unlabeled videos from target domains are accessible for training, thus prevailing works usually focus on developing models oriented to specific target domains. Typical works address domain-adaptive action recognition by learning cross-domain invariance [15], [81]–[84], and other works usually explore leveraging the multi-modal nature of video data [85]–[89]. Differently, domain-generalizable action recognition [16], [90]–[92] aims to learn models generalizable in unseen test domains, where videos of target domains are not accessible during training. To addess this task in the absence of target videos, Yao et al. assume that local features are more invariant across domains [16], and Lin et al. propose to learn diverse action features [91]. Our generalizable openvocabulary action recognition task is closely related to domaingeneralizable action recognition, as test domains remain unseen during training. However, our task is significantly more challenging as we strive for open-vocabulary abilities, leading to significantly different technical designs.

## C. Vision-Language Pretraining and Its Applications

In recent years, research on image-text foundation models has made great progress [3], [93]–[101], e.g., CLIP [3], SigLIP [99], CoCa [97]. Among these works, CLIP (Contrastive Language-Image Pretraining) [3] is one of the most representative, and it is the foundation of many openvocabulary works. By utilizing web-scale paired image-text data for training, CLIP shows robust zero-shot object recognition abilities. Also, many advanced works have demonstrated that CLIP can effectively solve downstream tasks by efficient adaptation [8], [61], [102]–[105]. Moreover, integrating CLIP with specialized techniques shows remarkable openvocabulary abilities on various image understanding tasks, e.g., object detection and segmentation [55], [106]–[109]. Although large-scale image-text pretraining has achieved great success, video-text pretraining still has room for development [110]– [121]. It is because videos are inherently more complex than images, and large-scale paired video-text datasets are less available. Therefore, it is valuable to develop methods for adapting pretrained image-text models to video understanding.

![](images/d8e3a64dd863fad32e85a577e5278c53c02379a3a3276c77f7e7d3f67488cd40.jpg)  
Fig. 3. An overview of our proposed XOV-Action model, which aims to overcome two critical challenges of the generalizable open-vocabulary action recognition task. First, our XOV-Action proposes Diversified Elaboration Representation Learning to boost the understanding of novel action concepts for open-set categories. By leveraging the Elaborative Video-text Alignment loss with Adaptive Elaboration Matching, XOV-Action captures diverse action-related concepts under the guidance of multiple textual descriptions. Second, to defend against the scene bias, our XOV-Action proposes Scene-Aware Video-text Alignment to learn scene-agnostic video representations. $\boldsymbol { \mathrm { B y } }$ leveraging the Scene-aware Discrimination and Action-aware Discrimination losses, XOV-Action encourages the video encoder to downweight the attention on scene information under the guidance of scene-encoded text prompts. Best viewed in color.

## III. THE PROPOSED XOV-ACTION MODEL

## A. Problem Formulation

This work focuses on the generalizable open-vocabulary action recognition task. In this task, a set of labeled videos $\begin{array} { r c l } { \mathcal D } & { = } & { \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { s } } } \end{array}$ from a source domain are given for training, where $x _ { i }$ denotes a source video, $y _ { i }$ denotes its ground-truth action label index and $N _ { s }$ denotes the number of videos. The source domain consists of K action categories, $y _ { i } \in \{ 1 , 2 , \dotsc , K \}$ , and the action name texts are denoted by $\left\{ a _ { 1 } , a _ { 2 } , \ldots , a _ { K } \right\}$ . For notational simplicity, we omit the sample index i of $x _ { i }$ and $y _ { i }$ in the following formulations when it does not cause ambiguity. Given only source videos for training, our goal is to develop a model that is generalizable in unseen target domains, where the source and target domains follow different data distributions and different label spaces. Following the standard protocol of previous open-vocabulary action recognition works [8], we sample T frames from each video as model input during training and test.

## B. Model Overview

Our XOV-Action model is built based on CLIP [3] composed of a video encoder $f _ { \mathrm { v i d } } ( \cdot )$ and a text encoder $f _ { \mathrm { t x t } } ( \cdot )$ following protocols of previous open-vocabulary action recognition works [8], [9]. Specifically, we construct a video encoder by integrating patches from neighboring frames in each self-attention layer of the original CLIP image encoder. For the video x, the global video representation $z _ { x } = f _ { \mathrm { v i d } } ( x )$ is obtained by the average of local video representations, where each local representation corresponds to the representation of one frame. We use the original CLIP text encoder and keep it frozen during training. Following previous works, we use a video-text alignment loss to adapt video data, which is formulated as follows:

$$
L _ {\mathrm{vta}} = - \log \frac {\exp \left(s (z _ {x} , z _ {a _ {y}}) / \tau\right)}{\sum_ {k = 1} ^ {K} \exp \left(s (z _ {x} , z _ {a _ {k}}) / \tau\right)},\tag{1}
$$

where $s ( \cdot , \cdot )$ denotes the cosine similarity, $\tau$ is the temperature, $z _ { a _ { k } } = f _ { \mathrm { t x t } } ( g ( a _ { k } ) )$ is the text representation of the kth action name text. The function $g ( \cdot )$ converts an action name into a text prompt in the form of $\stackrel { 6 6 } { \circ } \scriptscriptstyle \partial$ video of a person [doing something].”, $e . g .$ ., “a video of a person abseiling.” for the action “abseiling”. The above loss follows the standard formulation of previous contrastive-learning-based works, e.g., MoCo [122] and CLIP [3].

An overview of our proposed XOV-Action is shown in Figure 3. Our proposed XOV-Action includes two key components as technical contributions, namely Diversified Elaboration Representation Learning and Scene-Aware Video-text Alignment. Firstly, XOV-Action proposes Diversified Elaboration Representation Learning to capture rich action-related concepts, by elaborating action concepts using multiple textual descriptions. Through an Elaborative Video-text Alignment loss with Adaptive Elaboration Matching, our model learns diverse elaboration representations for each video, which boosts the understanding of novel action concepts for open-set categories. Secondly, by introducing scene-encoded text prompts, XOV-Action proposes a Scene-Aware Video-text Alignment method, which consists of a Scene-aware Discrimination loss and an Action-aware Discrimination loss. Accordingly, our model can learn scene-agnostic video representations to defend against the scene bias and improve the generalization in unseen domains. In what follows, we illustrate XOV-Action in detail.

## C. Diversified Elaboration Representation Learning

Our proposed XOV-Action model first focuses on a fundamental challenge of generalizable open-vocabulary action recognition models, namely the understanding of novel action concepts for open-set categories. Specifically, we propose a novel Diversified Elaboration Representation Learning method, which introduces multiple textual descriptions as auxiliary supervision in training. By incorporating the proposed Elaborative Video-text Alignment loss with Adaptive Elaboration Matching, our model learns multiple elaboration representations for each video, which encode rich concept information related to action categories. After learning actionrelated concepts, our model can associate concepts with open-set categories during testing through action descriptions, thereby improving the recognition of open-set action categories.

First of all, we introduce a set of M textual descriptions for the k-th action category, denoted by $\{ e _ { k } ^ { 1 } , e _ { k } ^ { 2 } , \ldots , { \bar { e } } _ { k } ^ { M } \}$ Specifically, for each action category, we ask GPT-4 [123] to automatically generate a diverse set of textual descriptions based on the category name, which does not involve any human annotation cost. Compared with simple name texts, these textual descriptions involve much more details and concept information about the corresponding action category, which helps distinguish different action categories. For example, when distinguishing open-set actions “long jump” and “high jump”, it is difficult to understand the meaning of “long” and “high” in the context of the respective actions without seeing any video samples during training, since “long jump” and “high jump” are novel action concepts to models. After introducing the textual description “A person is seen accelerating on a track, taking a leap, and landing in a distant sand pit.” for the action “long jump” and the textual description “A person attempts to jump over a horizontal bar at the greatest height possible, using a specific technique.” for the action “high jump”, it is much easier for models to distinguish these two actions. This is because these two textual descriptions contain more action-related concept information, e.g., “sand pit” and “horizontal bar”, and thus models can leverage these concepts to distinguish actions “long jump” and “high jump”.

Adaptive Elaboration Matching: By leveraging these textual descriptions as guidance, we propose to learn rich action-related concepts by Diversified Elaboration Representation Learning. Specifically, our model learns C elaboration representations for each video, denoted by $\{ \hat { z } _ { x } ^ { 1 } , \hat { z } _ { x } ^ { 2 } , \dots , \hat { z } _ { x } ^ { C } \}$ The i-th elaboration representation $\hat { z } _ { x } ^ { i }$ is produced by the ith video elaboration module, i.e., $\hat { z } _ { x } ^ { \bar { i } } ~ = ~ h _ { i } ( z _ { x } )$ , and there are C lightweight video elaboration modules on top of our vision encoder $f _ { \mathrm { v i d } } ( \cdot )$ . During training, the learning of an elaboration representation is supervised by its best-matched textual description, since the introduced descriptions for each action category form an unordered set, i.e., their ordering carries no semantic meaning. We formulate an adaptive representation matching problem to find the best-matched descriptions, termed Adaptive Elaboration Matching. Suppose that the number of elaboration representations C is larger than the number of textual descriptions M. Then, for a video of the k-th action category, our Adaptive Elaboration Matching between elaboration representations and textual descriptions is formulated as follows:

$$
\begin{array}{l} O ^ {k} = \underset {O ^ {k} \in \{0, 1 \} ^ {C \times M}} {\arg \max} \sum_ {i = 1} ^ {C} \sum_ {j = 1} ^ {M} s (\hat {z} _ {x} ^ {i}, f _ {\mathrm{txt}} (e _ {k} ^ {j})) \cdot O _ {i, j} ^ {k}, \\ \text {s.t.} \quad \sum_ {j = 1} ^ {M} O _ {i, j} ^ {k} \leq 1, \forall i, \sum_ {i = 1} ^ {C} O _ {i, j} ^ {k} = 1, \forall j, \end{array}\tag{2}
$$

where $f _ { \mathrm { t x t } } ( e _ { k } ^ { j } )$ is the text representation of description $e _ { k } ^ { j } .$ In this formulation, $O ^ { k } \in \{ 0 , \dot { 1 } \} ^ { C \times M }$ is the assignment matrix composed of binary elements, and the superscript k denotes the k-th category. The matrix element $O _ { i , j } ^ { k } \in \{ 0 , 1 \}$ indicates the matching between the elaboration representation $\hat { z } _ { x } ^ { i }$ and description $e _ { k } ^ { j } , i . e . , O _ { i , j } ^ { k } = 1$ indicates that $\hat { z } _ { x } ^ { i }$ is matched with $e _ { k } ^ { j }$ and $O _ { i , j } ^ { k } = 0$ indicates no match. The constraints $\begin{array} { r } { \sum _ { j = 1 } ^ { M } O _ { i , j } ^ { k } \stackrel { \ddot { \mathbf { \theta } } } { = } } \end{array}$ 1 and $\textstyle \sum _ { i = 1 } ^ { C } O _ { i , j } ^ { k } = 1$ are introduced to ensure an injective oneto-one matching. The matching problem presented in Eq. (2) is solved by the Hungarian algorithm [124]. Overall, Adaptive Elaboration Matching avoids arbitrary alignments between the unordered sets of video elaboration representations and textual descriptions. This reduces supervision noise and facilitates the learning of diverse elaboration representations that encode rich action-related concepts.

Elaborative Video-text Alignment: In addition, since the textual descriptions are generated by GPT-4, a single video may not contain all details mentioned by the descriptions of its ground-truth category. Thus, simply using all descriptions for each video may introduce noise during training. Accordingly, for each video, we propose a confidence-aware elaboration selection strategy to select the top- $\hat { \mathbf { \nabla } } \hat { C }$ most relevant descriptions from every action category during training. Formally, we propose an Elaborative Video-text Alignment loss for learning diversified elaboration representations, which is given as follows:

$$
L _ {\mathrm{eva}} = - \log \frac {\exp \left(\frac {1}{\tau \cdot \hat {C}} \sum_ {(i , j) \in \mathcal {O} _ {\hat {C}} ^ {y}} s (\hat {z} _ {x} ^ {i} , f _ {\mathrm{txt}} (e _ {y} ^ {j}))\right)}{\sum_ {k = 1} ^ {K} \exp \left(\frac {1}{\tau \cdot \hat {C}} \sum_ {(i , j) \in \mathcal {O} _ {\hat {C}} ^ {k}} s (\hat {z} _ {x} ^ {i} , f _ {\mathrm{txt}} (e _ {k} ^ {j}))\right)},\tag{3}
$$

where ${ \mathcal { O } } _ { \hat { C } } ^ { k }$ is a subset of matches extracted from $O ^ { k }$ , consisting of the index pairs (i, j) that yield the $\mathrm { t o p } { \cdot } \hat { C }$ highest similarity $s ( \hat { z } _ { x } ^ { i } , f _ { \mathrm { t x t } } ( \bar { e _ { k } ^ { j } } ) )$ ). This formulation enables the model to selectively align each video sample with the top $- \hat { C }$ bestmatched textual descriptions of its ground-truth category, while contrasting the video sample against descriptions from other categories. Guided by this loss, our model can learn diverse elaboration representations that encode rich and accurate action-related concepts under the supervision of diverse textual descriptions. During testing, we also introduce multiple textual representations for each action category in target domains. As a result, our model can associate learned concepts with the novel action concepts of open-set categories, thus improving the recognition of open-set action categories.

Notably, our proposed Diversified Elaboration Representation Learning is robust to different LLM-generated textual descriptions, as shown in Tables A1-A5 in the Appendix. In practical deployment, these category-level descriptions introduce a lightweight LLM dependency, requiring a onetime LLM generation step before training or inference. While current LLMs work reliably for common actions, they may produce overly generic or imprecise descriptions for rare actions, e.g., niche cultural practices. In such cases, we suggest a lightweight human-in-the-loop check before deployment.

## D. Scene-Aware Video-text Alignment

More importantly, in the generalizable open-vocabulary action recognition task, domain gap is a major obstacle to model generalization, and it is largely ignored by existing open-vocabulary works. Therefore, to bridge the domain gap and improve the generalization in unseen video domains, we focus on the scenario difference, a key difference between two video domains. Accordingly, to mitigate the scenario discrepancy across video domains, we propose a novel Scene-Aware Video-text Alignment method to learn scene-agnostic video representations. The key idea of our Scene-Aware Videotext Alignment is to distinguish video representations apart from scene-encoded text representations, which encourages the video encoder to downweight the attention on scene information in videos.

Scene-aware Discrimination: First of all, we randomly sample N scene suffixes and construct scene-encoded text prompts for each training video. Each scene suffix is in the form of “[at/on/in the/a scene]”, e.g., “in the park”, “on the $\operatorname { s t } { \boldsymbol { \mathrm { r e e t } } } ^ { \prime }$ . In our implementation, we ask GPT-4 [123] to automatically generate a pool of scene suffixes for random sampling, which involves no human annotation cost. Based on these suffixes, we construct scene-encoded text prompts of ground-truth action category for each video, which is in the form of $^ { 6 6 } \mathrm { { \hat { d } } }$ video of a person [doing something] [at/on/in the $/ \mathsf { a }$ scene].”. For example, for the action “abseiling”, we construct $N$ scene-encoded text prompts, e.g., “a video $\cot$ a person abseiling in the park.”.

Then, based on the scene-encoded text prompts, we design a Scene-aware Discrimination loss, which is formulated as follows:

$$
L _ {\text { scene }} = - \log \frac {\exp \left(s (z _ {x} , z _ {a _ {y}})\right)}{\exp \left(s (z _ {x} , z _ {a _ {y}})\right) + \sum_ {n = 1} ^ {N} \exp \left(s (z _ {x} , \tilde {z} _ {a _ {y}} ^ {n})\right)}.\tag{4}
$$

In this loss, $\tilde { z } _ { a _ { y } } ^ { n } = f _ { \mathrm { t x t } } ( \tilde { q } _ { y } ^ { n } )$ is the representation of a sceneencoded text prompt, which encodes the semantic information of the n-th scene. We denote the n-th scene-encoded text prompt by $\tilde { q } _ { y } ^ { n } \ = \ \tilde { g } ( a _ { y } , n )$ , where the function $\tilde { g } ( \cdot , n )$ transforms an action name into a scene-encoded text prompt by incorporating the n-th scene suffix. According to Eq. (4), our proposed Scene-aware Discrimination loss pushes the video representations away from the scene-encoded text rep resentations in video-text alignment. In this way, this loss encourages the video encoder to pay less attention to scene information and thus pay more attention to action information, by leveraging the strong power of CLIP text encoder. As a result, we can mitigate the scene bias when fitting training videos.

Action-aware Discrimination: Empirically, we find that although the Scene-aware Discrimination loss $L _ { \mathrm { s c e n e } }$ effectively improves the cross-domain action recognition performance for the closed-set categories, it reduces the cross-domain performance for open-set categories. Thus, it leads to limited improvement in the overall open-vocabulary action recognition performance across domains.

A critical reason is that $L _ { \mathrm { s c e n e } }$ pushes a video away from a scene-encoded text prompt in representation space, potentially resulting in some non-ground-truth action texts becoming more similar to the video than the scene-encoded prompt is. However, since the scene-encoded text prompt encodes semantic information of the ground-truth category (in addition to a specific scene), the video representation should be more dissimilar to non-ground-truth action texts than to the sceneencoded text prompt. For example, consider a video that depicts a person playing basketball in a court. This video should have a more dissimilar representation to the text prompts of other categories (e.g., “a video of a person kicking soccer.”) than to the scene-encoded text prompts $( e . g .$ , “a video of a person playing basketball in the park.”). Overall, this issue would cause some confusion in video representation space.

Accordingly, to alleviate the degradation of cross-domain open-set performance, we propose an Action-aware Discrimination loss to constrain video representation learning. Relative to the scene-encoded text representations that encodes groundtruth action semantics, our Action-aware Discrimination loss pushes the video representations away from the text representations of non-ground-truth categories, which is formulated as follows:

$$
L _ {\text {action}} = \sum_ {n = 1} ^ {N} \sum_ {k \neq y} ^ {K} \frac {\max \left(0 , \delta - s (z _ {x} , \tilde {z} _ {a _ {y}} ^ {n}) + s (z _ {x} , z _ {a _ {k}})\right)}{N (K - 1)}.\tag{5}
$$

where $\delta$ is a margin factor. In principle, $L _ { \mathrm { a c t i o n } }$ introduces a constraint for non-ground-truth action text $a _ { k } \ ( k \neq y )$ , i.e., $s ( z _ { x } , \tilde { z } _ { a _ { u } } ^ { n } ) - s ( z _ { x } , z _ { a _ { k } } ) \geq \delta .$ By using both $L _ { \mathrm { s c e n e } }$ and $L _ { \mathrm { a c t i o n } } ,$ our proposed Scene-Aware Video-text Alignment learns a more reasonable video representation space, i.e., $s ( z _ { x } , z _ { a _ { y } } ) >$ $s ( z _ { x } , \tilde { z } _ { a _ { y } } ^ { n } ) > s ( z _ { x } , z _ { a _ { k } } )$ when $k \neq y .$

## E. Overall Training and Test

The overall training loss of our XOV-Action model is given as follows:

$$
L = L _ {\mathrm{vta}} + \lambda_ {\mathrm{eva}} L _ {\mathrm{eva}} + \lambda_ {\mathrm{scene}} L _ {\mathrm{scene}} + \lambda_ {\mathrm{action}} L _ {\mathrm{action}},\tag{6}
$$

where $\lambda _ { \mathrm { { e v a } } } , \lambda _ { \mathrm { { s c e n e } } }$ and $\lambda _ { \mathrm { a c t i o n } }$ are non-negative coefficients for trade-off.

During test, for each video, we leverage both the global video representation $z _ { x }$ and the $\mathrm { t o p } { \cdot } \hat { C }$ elaboration representations $\left\{ \hat { z } _ { x } ^ { i } \right\}$ for classification. More specifically, we use an ensembled score for classification, which is formulated as follows:

$$
p _ {k} = \lambda_ {\mathrm{e}} s (z _ {x}, z _ {a _ {k}}) + \frac {1 - \lambda_ {\mathrm{e}}}{\hat {C}} \sum_ {(i, j) \in O _ {\hat {C}} ^ {k}} s (\hat {z} _ {x} ^ {i}, f _ {\mathrm{txt}} (e _ {k} ^ {j})),\tag{7}
$$

where $p _ { k }$ denotes the score of the k-th action category, and $\lambda _ { \mathrm { e } } \in [ 0 , 1 ]$ is the trade-off coefficient for ensemble. Finally, our model conducts the classification by identifying the highest value of $p _ { k }$ across different action categories.

## IV. THE PROPOSED CROSS-DOMAIN OPEN-VOCABULARY ACTION BENCHMARK

Our work focuses on the generalizable open-vocabulary action recognition task, which aims to develop open-vocabulary action recognition models that are generalizable in unseen target domains by training in the source domain, i.e., recognizing both closed-set and open-set action categories in new test domains. In order to evaluate models in this task, we establish XOVABench, the first CROSS-domain Open-Vocabulary Action recognition Benchmark, which provides a comprehensive way to analyze models across various situations. In what follows, we introduce the components of our XOVABench benchmark in detail, as well as evaluation metrics.

## A. Benchmark Components

Our proposed XOVABench benchmark consists of two source datasets for training and four target datasets for test. The two source datasets for training are as follows:

(1) Kinetics400 [13]: One of the most widely-used action recognition datasets, consisting of 400 action categories. Videos in Kinetics400 are collected from YouTube, which are usually recorded in common daily environments (e.g., normal illumination and weather). Existing open-vocabulary action recognition works typically use Kinetics400 for training, and we use the original split following them.

(2) Kinetics150: A subset of Kinetics400, composed of 150 action categories selected from the full Kinetics400. These 150 categories include all the closed-set categories of the four target datasets in comparison to Kinetics400, and the remaining categories are randomly sampled. We will illustrate the definition of closed-set categories later. We construct this subset to conduct detailed analysis for generalizable openvocabulary action recognition models.

The four target datasets for test are as follows:

(1) UCF [17]: One of the most widely-used action recognition datasets, consisting of 101 action categories. Videos in UCF are collected from YouTube, and videos of each action are usually captured from specific or similar environments. UCF has a small domain gap compared with the Kinetics (source) domain, and previous open-vocabulary action recognition works [8], [9] commonly use UCF to evaluate their models’ open-vocabulary recognition abilities.

TABLE I  
CATEGORY STATISTICS OF FOUR TEST DOMAINS IN OUR XOVABENCH. THE CLOSED-SET AND OPEN-SET CATEGORIES ARE IDENTIFIED ACCORDING TO KINETICS400. IN THE TABLE, WE ALSO SHOW THE DOMAIN GAP BETWEEN EACH TEST DOMAIN AND THE TRAINING DOMAIN, AS WELL AS THE QUANTITATIVE MEASURE OF DOMAIN GAPS.

<table><tr><td>Test Domains</td><td>UCF</td><td>HMDB</td><td>ARID</td><td>NEC-Dr</td></tr><tr><td># of closed-set actions</td><td>50</td><td>33</td><td>6</td><td>7</td></tr><tr><td># of open-set actions</td><td>51</td><td>18</td><td>5</td><td>9</td></tr><tr><td># of all actions</td><td>101</td><td>51</td><td>11</td><td>16</td></tr><tr><td>Domain gap</td><td>Small</td><td>Moderate</td><td>Large</td><td>Large</td></tr><tr><td>Quantitative measure of gap</td><td>0.429</td><td>0.626</td><td>0.789</td><td>0.850</td></tr></table>

(2) HMDB [18]: A widely-used action recognition datasets consisting of 51 action categories. Compared with UCF, videos in HMDB are captured from more unconstrained environments and more different camera views. Specifically, Videos in HMDB are collected mainly from movies, and remaining videos are from Prelinger archive, YouTube or Google videos. Overall, HMDB has a moderate domain gap compared with the Kinetics domain [79], [125].

(3) ARID [19]: A dataset consisting of 11 categories of action videos, which are recorded under dark environments. These actions include singular person actions (e.g., jump, run) and actions associated with objects (e.g., drink, pick). Due to the significantly different illumination conditions, ARID exhibits a large domain gap compared with the Kinetics domain. When training data are limited to videos with normal illumination, developing models that are generalizable in ARID is challenging.

(4) NEC-Dr [20]: A dataset consisting of 16 categories of action videos, which are recorded by drones in the same basketball court. These actions include single-person actions (e.g., jump, walk) and interactive actions (e.g., hug, shake hands). Due to the different shooting equipments and scenarios, NEC-Dr exhibits a large domain gap compared with the Kinetics domain. Therefore, it is challenging to make a Kinetics-trained model generalizable in such a drone video domain.

These test domains exhibit different levels of domain gaps in comparison to the training domain. To quantitatively measure the domain gaps, we use the mean of class-wise feature discrepancy of closed-set categories as the evaluation metric following previous works [126], [127]. A summary of four test domains is given in Table I. In our experiments, we use videos from Kinetics400 or Kinetics150 for training. Then, we conduct evaluation on the four target datasets to assess open-vocabulary action recognition abilities across domains.

## B. Evaluation Metrics

First, we illustrate the definitions of closed-set and openset categories for test domains: (a) The closed-set categories refer to the categories that share similar meanings and have a common lexicon with the categories in the training domain, following a similar approach in previous zero-shot action recognition works [73], [128]. (b) The open-set categories refer to the remaining categories that are not involved in the training domain. Then, we identify closed-set and open-set categories for each test domain, according to the Kinetics400 categories. Category statistics are summarized in Table I.

TABLE II  
COMPARISON WITH KINETICS150-TRAINED OPEN-VOCABULARY ACTION RECOGNITION MODELS ON XOVABENCH. “AVG” DENOTES THE AVERAGE ACC OVER FOUR TEST DOMAINS. THE BOLD/UNDERLINED NUMBERS INDICATE THE BEST/SECOND BEST. WE PRIORITIZE THE AVERAGE OVERALL ACC AS THE PRIMARY METRIC IN MODEL COMPARISON. TO CONDUCT SOLID COMPARISONS, WE CAREFULLY TUNE COMPARED MODELS BASED ON THEIR OFFICIAL CODE AND SELECT THE CHECKPOINT WITH THE BEST AVERAGE OVERALL ACC FOR EACH MODEL FOR COMPARISON.

<table><tr><td rowspan="2">Models</td><td colspan="3">UCF</td><td colspan="3">HMDB</td><td colspan="3">ARID</td><td colspan="3">NEC-Dr</td><td colspan="3">AVG</td></tr><tr><td>Closed</td><td>Open</td><td>All</td><td>Closed</td><td>Open</td><td>All</td><td>Closed</td><td>Open</td><td>All</td><td>Closed</td><td>Open</td><td>All</td><td>Closed</td><td>Open</td><td>All</td></tr><tr><td>CLIP [3]</td><td>61.51</td><td>57.85</td><td>59.67</td><td>42.92</td><td>38.62</td><td>41.40</td><td>32.04</td><td>12.04</td><td>24.02</td><td>18.52</td><td>6.44</td><td>12.33</td><td>38.75</td><td>28.74</td><td>34.35</td></tr><tr><td>ActionCLIP [4]</td><td>84.76</td><td>56.27</td><td>70.44</td><td>56.78</td><td>33.02</td><td>48.40</td><td>42.25</td><td>19.75</td><td>33.23</td><td>18.75</td><td>4.22</td><td>11.30</td><td>50.63</td><td>28.32</td><td>40.84</td></tr><tr><td>LSTM [4]</td><td>84.54</td><td>58.38</td><td>71.39</td><td>58.30</td><td>36.38</td><td>50.56</td><td>42.04</td><td>26.85</td><td>35.95</td><td>17.13</td><td>12.44</td><td>14.73</td><td>50.50</td><td>33.51</td><td>43.16</td></tr><tr><td>TConv [4]</td><td>84.54</td><td>58.61</td><td>71.24</td><td>57.29</td><td>36.19</td><td>49.84</td><td>44.29</td><td>27.78</td><td>37.67</td><td>17.59</td><td>12.22</td><td>14.84</td><td>50.93</td><td>33.56</td><td>43.40</td></tr><tr><td>XCLIP [5]</td><td>89.89</td><td>59.38</td><td>74.59</td><td>53.44</td><td>30.78</td><td>45.62</td><td>35.10</td><td>13.72</td><td>26.64</td><td>16.90</td><td>5.70</td><td>11.26</td><td>48.83</td><td>27.40</td><td>39.53</td></tr><tr><td>Text4Vis [7]</td><td>83.40</td><td>49.84</td><td>66.54</td><td>54.25</td><td>38.06</td><td>48.54</td><td>41.43</td><td>31.71</td><td>37.53</td><td>12.50</td><td>4.17</td><td>8.23</td><td>47.90</td><td>30.94</td><td>40.21</td></tr><tr><td>BIKE [56]</td><td>74.36</td><td>55.80</td><td>65.03</td><td>50.71</td><td>38.79</td><td>46.50</td><td>36.09</td><td>16.67</td><td>28.30</td><td>17.03</td><td>13.16</td><td>15.04</td><td>44.55</td><td>31.10</td><td>38.72</td></tr><tr><td>ViFiCLIP [8]</td><td>82.93</td><td>62.07</td><td>72.44</td><td>58.60</td><td>41.60</td><td>52.60</td><td>44.69</td><td>27.13</td><td>37.65</td><td>17.36</td><td>10.53</td><td>13.86</td><td>50.90</td><td>35.33</td><td>44.14</td></tr><tr><td>OpenVCLIP [9]</td><td>87.67</td><td>70.86</td><td>79.20</td><td>56.48</td><td>44.96</td><td>52.41</td><td>48.16</td><td>27.13</td><td>39.73</td><td>19.68</td><td>13.38</td><td>16.45</td><td>53.00</td><td>39.08</td><td>46.95</td></tr><tr><td>FROSTER [60]</td><td>89.80</td><td>69.17</td><td>79.43</td><td>53.94</td><td>35.19</td><td>47.32</td><td>39.15</td><td>29.57</td><td>35.31</td><td>13.13</td><td>1.97</td><td>7.41</td><td>49.01</td><td>33.98</td><td>42.37</td></tr><tr><td>TC-CLIP [66]</td><td>87.41</td><td>31.67</td><td>59.40</td><td>49.39</td><td>32.41</td><td>43.40</td><td>47.06</td><td>13.33</td><td>33.54</td><td>19.59</td><td>10.07</td><td>14.70</td><td>50.86</td><td>21.89</td><td>37.76</td></tr><tr><td>MoTE [21]</td><td>85.19</td><td>61.51</td><td>73.29</td><td>58.69</td><td>32.96</td><td>49.61</td><td>44.85</td><td>16.36</td><td>33.43</td><td>14.48</td><td>15.49</td><td>14.99</td><td>50.80</td><td>31.58</td><td>42.83</td></tr><tr><td>XOV-Action (Ours)</td><td>88.46</td><td>72.02</td><td>80.20</td><td>60.22</td><td>46.46</td><td>55.36</td><td>50.41</td><td>32.01</td><td>43.03</td><td>24.77</td><td>16.67</td><td>20.61</td><td>55.96</td><td>41.79</td><td>49.80</td></tr></table>

In our experiments, we adopt three evaluation metrics: (1) The closed-set accuracy measures the recognition performance of closed-set categories, which primarily evaluates the model abilities of bridging domain gaps when fitting training videos. (2) The open-set accuracy measures the performance of openset categories, which evaluates the generalization abilities across both video domains and action categories. (3) The overall accuracy measures the recognition performance over all categories, which provides a holistic view of model effectiveness across various situations.

Note that, previous open-vocabulary action recognition works [8], [9] usually treat all the categories in UCF and HMDB as open-set for models trained on Kinetics400. However, UCF and HMDB share many overlapping categories with Kinetics400. For example, UCF has 50 categories that overlap with Kinetics400. Thus, these works make an inaccurate assessment of models’ open-vocabulary abilities. Unlike these works, we provide a more accurate way for evaluation by identifying closed-set and open-set categories according to the training dataset. By evaluating across various domains and different action categories, our proposed XOVABench benchmark enables us to perform a wide range of analysis for generalizable open-vocabulary action recognition models. To achieve satisfactory performance on our proposed XOVABench, models should consider the challenges of both novel action concepts and domain gaps, since XOVABench consists of test domains containing various action categories and exhibiting different levels of domain gaps.

## V. EXPERIMENT RESULTS

## A. Experimental Setups

Implementation Details: For each video, our model takes $T ~ = ~ 1 6$ frames of size 224 × 224 as inputs. Following ViFiCLIP [8], we adopt temporal jitter, multi-scale random spatial crop and color jitter for augmentation during training. During test, we use one temporal clip composed of centercropped frames for inference. Regarding the architecture of the video encoder, we adopt ViT-B/32 and ViT-B/16 initialized by CLIP image encoder for Kinetics150 and Kinetics400, respectively. For temporal modeling, we set the temporal receptive field of self-attention layers to 7 in the video encoder. By default, we learn C = 10 video elaboration modules in training, and select C<sup>ˆ</sup> = 3 elaboration representations for each video for both training and inference. These video elaboration modules are instantiated using MLPs with residuals, and thus they are lightweight and have a negligible impact on computational efficiency. We ask GPT-4 [123] to output 300 scene suffixes that are common in daily life, and randomly sample N = 50 scene suffixes for our proposed Scene-Aware Video-text Alignment in each batch. In addition, we ask GPT-4 to output M = 10 textual descriptions for each action category in each dataset. We freeze the text encoder during training and thus all text epresentations can be pre-computed before training. During inference, we set $\lambda _ { \mathrm { e } } = 0 . 4$ to obtain an ensembled scores for action classification. During training, network parameters are optimized using AdamW optimizer with a batch size of 512, a learning rate of 8e-6, a momentum of 0.9 and a weight decay of 1e-3. If not specified, we set the loss coefficients as $\lambda _ { \mathrm { s c e n e } } = 0 . 2 , \lambda _ { \mathrm { a c t i o n } } = 0 . 1 , \lambda _ { \mathrm { e v a } } = 0 . 1 ,$ the margin as $\delta \ = \ 0 . 5 ,$ and the temperature coefficient as τ = 0.01. Following OpenVCLIP [9], we adopt the SWA technique [129] for producing the final model.

TABLE III  
COMPARISON WITH KINETICS400-TRAINED OPEN-VOCABULARY ACTION RECOGNITION MODELS ON XOVABENCH. “AVG” DENOTES THE AVERAGE ACC OVER FOUR TEST DOMAINS. THE BOLD/UNDERLINED NUMBERS INDICATE THE BEST/SECOND BEST. WE PRIORITIZE THE AVERAGE OVERALL ACC AS THE PRIMARY METRIC IN MODEL COMPARISON. TO CONDUCT SOLID COMPARISONS, WE CAREFULLY TUNE COMPARED MODELS BASED ON THEIR OFFICIAL CODE AND SELECT THE CHECKPOINT WITH THE BEST AVERAGE OVERALL ACC FOR EACH MODEL FOR COMPARISON.

<table><tr><td rowspan="2">Models</td><td colspan="3">UCF</td><td colspan="3">HMDB</td><td colspan="3">ARID</td><td colspan="3">NEC-Dr</td><td colspan="3">AVG</td></tr><tr><td>Closed</td><td>Open</td><td>All</td><td>Closed</td><td>Open</td><td>All</td><td>Closed</td><td>Open</td><td>All</td><td>Closed</td><td>Open</td><td>All</td><td>Closed</td><td>Open</td><td>All</td></tr><tr><td>CLIP [3]</td><td>66.71</td><td>59.53</td><td>63.15</td><td>49.19</td><td>35.26</td><td>44.27</td><td>37.76</td><td>12.65</td><td>27.69</td><td>10.42</td><td>22.44</td><td>16.59</td><td>41.04</td><td>32.47</td><td>37.92</td></tr><tr><td>ActionCLIP [4]</td><td>89.87</td><td>51.42</td><td>70.55</td><td>64.07</td><td>23.32</td><td>49.69</td><td>48.98</td><td>22.53</td><td>38.38</td><td>19.91</td><td>8.89</td><td>14.26</td><td>55.71</td><td>26.54</td><td>43.22</td></tr><tr><td>XCLIP [5]</td><td>89.50</td><td>60.96</td><td>75.16</td><td>60.32</td><td>36.01</td><td>51.74</td><td>45.51</td><td>18.21</td><td>34.56</td><td>22.22</td><td>9.78</td><td>15.84</td><td>54.39</td><td>31.24</td><td>44.33</td></tr><tr><td>Text4Vis [7]</td><td>87.21</td><td>54.16</td><td>70.60</td><td>59.31</td><td>26.31</td><td>47.66</td><td>42.44</td><td>21.30</td><td>33.97</td><td>15.97</td><td>14.67</td><td>15.30</td><td>51.24</td><td>29.11</td><td>41.88</td></tr><tr><td>BIKE [56]</td><td>81.04</td><td>61.12</td><td>71.03</td><td>53.63</td><td>39.52</td><td>48.65</td><td>46.98</td><td>15.18</td><td>34.23</td><td>23.92</td><td>10.94</td><td>17.26</td><td>51.39</td><td>31.69</td><td>42.79</td></tr><tr><td>ViFiCLIP [8]</td><td>90.51</td><td>56.27</td><td>73.31</td><td>64.98</td><td>30.22</td><td>52.71</td><td>51.92</td><td>19.14</td><td>38.77</td><td>21.99</td><td>11.78</td><td>16.75</td><td>57.35</td><td>29.35</td><td>45.39</td></tr><tr><td>OpenVCLIP [9]</td><td>91.06</td><td>75.03</td><td>83.00</td><td>60.63</td><td>42.54</td><td>54.24</td><td>52.25</td><td>28.35</td><td>42.67</td><td>22.92</td><td>14.91</td><td>18.81</td><td>56.71</td><td>40.21</td><td>49.68</td></tr><tr><td>FROSTER [60]</td><td>89.53</td><td>74.86</td><td>82.16</td><td>59.70</td><td>38.89</td><td>52.36</td><td>47.06</td><td>33.63</td><td>41.67</td><td>11.06</td><td>16.19</td><td>13.69</td><td>51.84</td><td>40.89</td><td>47.47</td></tr><tr><td>TC-CLIP [66]</td><td>90.91</td><td>67.23</td><td>79.01</td><td>60.20</td><td>32.22</td><td>50.33</td><td>52.74</td><td>23.64</td><td>41.07</td><td>24.19</td><td>8.75</td><td>16.27</td><td>57.01</td><td>32.96</td><td>46.67</td></tr><tr><td>MoTE [21]</td><td>86.78</td><td>67.04</td><td>76.86</td><td>61.62</td><td>41.67</td><td>54.58</td><td>50.51</td><td>22.73</td><td>39.37</td><td>12.87</td><td>15.32</td><td>14.13</td><td>52.94</td><td>36.69</td><td>46.23</td></tr><tr><td>XOV-Action (Ours)</td><td>91.97</td><td>80.08</td><td>86.00</td><td>65.28</td><td>43.83</td><td>57.72</td><td>54.29</td><td>32.32</td><td>45.48</td><td>24.31</td><td>16.67</td><td>20.39</td><td>58.96</td><td>43.23</td><td>52.39</td></tr></table>

Evaluation Protocol: We adopt nine state-of-the-art openvocabulary action recognition models, which have released their codes, to conduct model comparisons. To conduct solid comparisons, we carefully tune each model on Kinetics150/Kinetics400 based on the official code, and report the best generalizable open-vocabulary action recognition performance for each model. For a fair comparison, we adopt consistent backbones and same training recipes for all models (e.g., epoch number), and we use one temporal clip during inference.

We evaluate each model on all the four test domains of our XOVABench benchmark, where different test domains have various levels of domain gaps and differ in action category. For a comprehensive analysis, we report the accuracy of closedset, open-set and all categories in individual test domains as well as the average accuracies. When conducting a comparison between different models, we prioritize the average overall accuracy as the primary metric.

## B. Comparison with State-of-the-arts

The results are summarized in Table II and III. As shown in the tables, by adapting CLIP to video data, previous open-vocabulary action recognition models obtain substantial improvement over the original CLIP for generalizable openvocabulary action recognition. Although these methods show impressive performance for the domain with a small gap (UCF), they still exhibit limited performance for the domains with large gaps (ARID and NEC-Dr). These results reveal potential challenges of the task, as discussed in Section I. By learning scene-agnostic video representations and diversified elaboration representations, our proposed XOV-Action outperforms previous state-of-the-arts by 2.85% and 2.71% in terms of average overall accuracy for Kinetics150 and Kinetics400, respectively. These improvements are significant, especially considering that the metric is an average across four test domains. More importantly, our proposed XOV-Action obtains the best performance in most cases on XOVABench, demonstrating its superior capabilities across different category types and different test domains. This success is attributed to our effective designs specified in open-set action understanding and cross-domain generalization.

TABLE IV  
COMPARISON WITH STATE-OF-THE-ART OPEN-VOCABULARY ACTION RECOGNITION MODELS ON KINETICS600 ZERO-SHOT (K600-ZS) AND KINETICS400 BASE-TO-NOVEL (K400-B2N) BENCHMARKS. THE BOLD/UNDERLINED NUMBERS INDICATE THE BEST/SECOND BEST.

<table><tr><td rowspan="2">Models</td><td>K600-ZS</td><td colspan="3">K400-B2N</td></tr><tr><td>Top-1</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP [3]</td><td> $68.1 \pm 1.1$ </td><td>62.3</td><td>53.4</td><td>57.5</td></tr><tr><td>ActionCLIP [4]</td><td> $62.5 \pm 1.2$ </td><td>61.0</td><td>46.2</td><td>52.6</td></tr><tr><td>VPT [6]</td><td> $55.8 \pm 0.7$ </td><td>69.7</td><td>37.6</td><td>48.8</td></tr><tr><td>XCLIP [5]</td><td> $65.2 \pm 0.4$ </td><td>74.1</td><td>56.4</td><td>64.0</td></tr><tr><td>AIM [130]</td><td> $66.7 \pm 0.5$ </td><td>74.6</td><td>62.5</td><td>68.0</td></tr><tr><td>ST-Adapter [104]</td><td> $60.2 \pm 1.8$ </td><td>73.6</td><td>62.0</td><td>67.3</td></tr><tr><td>ViFiCLIP [8]</td><td> $71.2 \pm 1.0$ </td><td>76.4</td><td>61.1</td><td>67.9</td></tr><tr><td>OpenVCLIP [9]</td><td> $73.0 \pm 0.8$ </td><td>76.5</td><td>62.6</td><td>68.9</td></tr><tr><td>FROSTER [60]</td><td> $\underline{74.8} \pm 0.9$ </td><td> $\underline{77.8}$ </td><td> $\underline{64.3}$ </td><td> $\underline{70.4}$ </td></tr><tr><td>XOV-Action (Ours)</td><td> $\underline{76.5} \pm 0.8$ </td><td> $\underline{79.0}$ </td><td> $\underline{66.0}$ </td><td> $\underline{71.9}$ </td></tr></table>

In addition to XOVABench, we conduct experiments on Kinetics600 zero-shot benchmark and Kinetics400 base-tonovel benchmark to demonstrate the generalization capabilities of our proposed XOV-Action model. Specifically, for the Kinetics600 zero-shot benchmark, models are trained on Kinetics400 and evaluated on Kinetics600’s extra action categories, $i . e . ,$ , novel categories that are not included in Kinetics400. For the Kinetics400 base-to-novel benchmark, models are trained on the base categories of Kinetics400 and evaluated on the remaining (novel) categories. Following the standard protocol [8], [9], we report the Top-1 accuracy of models on the Kinetics600 zero-shot benchmark, and report the accuracies for both base and novel categories and the Harmonic Mean (HM) for an overall measurement. As shown in Table IV, our XOV-Action model significantly outperforms previous state-of-the-art models by 1.7% and 1.5% on these two benchmarks, respectively. These results demonstrate the strong generalization capabilities of our model.

TABLE V  
ABLATION STUDY OF OUR PROPOSED XOV-ACTION TRAINED ON KINETICS150. “AVG” DENOTES THE AVERAGE ACC OVER FOUR TEST DOMAINS. THE CLOSED-SET ACCURACY PRIMARILY EVALUATES THE MODEL ABILITIES OF TACKLING DOMAIN GAPS WHEN FITTING TRAINING VIDEOS.

<table><tr><td rowspan="2">Models</td><td colspan="3">UCF</td><td colspan="3">HMDB</td><td colspan="3">ARID</td><td colspan="3">NEC-Dr</td><td colspan="3">AVG</td></tr><tr><td>Closed</td><td>Open</td><td>All</td><td>Closed</td><td>Open</td><td>All</td><td>Closed</td><td>Open</td><td>All</td><td>Closed</td><td>Open</td><td>All</td><td>Closed</td><td>Open</td><td>All</td></tr><tr><td> $L_{vta}$ </td><td>87.07</td><td>69.39</td><td>78.19</td><td>57.59</td><td>44.59</td><td>53.00</td><td>47.35</td><td>26.52</td><td>39.00</td><td>21.30</td><td>14.04</td><td>17.57</td><td>53.33</td><td>38.64</td><td>46.94</td></tr><tr><td> $+L_{scene}$ </td><td>87.82</td><td>69.07</td><td>78.40</td><td>59.41</td><td>43.10</td><td>53.65</td><td>49.18</td><td>25.31</td><td>39.61</td><td>25.93</td><td>10.53</td><td>18.03</td><td>55.59</td><td>37.00</td><td>47.42</td></tr><tr><td> $+L_{scene}\&L_{action}$ </td><td>87.98</td><td>69.97</td><td>78.93</td><td>58.40</td><td>44.59</td><td>53.53</td><td>48.78</td><td>29.57</td><td>41.08</td><td>24.54</td><td>12.72</td><td>18.48</td><td>54.92</td><td>39.21</td><td>48.00</td></tr><tr><td>Full</td><td>88.46</td><td>72.02</td><td>80.20</td><td>60.22</td><td>46.46</td><td>55.36</td><td>50.41</td><td>32.01</td><td>43.03</td><td>24.77</td><td>16.67</td><td>20.61</td><td>55.96</td><td>41.79</td><td>49.80</td></tr></table>

## C. Main Ablation Study

Table V summarizes the results of our ablation study. In this experiment, we use the model trained with only the videotext alignment loss $L _ { \mathrm { v t a } }$ as the baseline. By introducing our proposed Scene-aware Discrimination loss $L _ { \mathrm { s c e n e } } ,$ , our model obtains improvement on XOVABench in terms of closed-set accuracy on all the four test domains. Specifically, our model obtains significant improvements of 1.83% and 4.63% in terms of closed-set accuracy on ARID and NEC-Dr, which have large domain gaps with the training domain. This is because that $L _ { \mathrm { s c e n e } }$ encourages the video encoder to downweight the attention on scene information in videos and thus pay more attention to action information that are generalizable across different domains. However, we find that introducing only the Scene-aware Discrimination loss reduces the open-set performance, leading to limited improvement in average overall accuracy on XOVABench. By introducing the Action-aware Discrimination loss $L _ { \mathrm { a c t i o n } } .$ , our model can obtain improved cross-domain closed-set accuracy over the baseline while satisfactorily maintaining the open-set accuracy on each test domain, and this leads to an improvement of 1.06% in terms of average overall accuracy over the baseline. Furthermore, by introducing our proposed Diversified Elaboration Representation Learning, our full XOV-Action model can obtain significant improvement for the open-set action categories on each test domain, leading to a large improvement of 2.58% on avarage. This is attributed to the capture of rich actionrelated concepts in videos. Overall, our model can effectively improve the generalization of both closed-set and open-set action categories in unseen test domains.

## D. Quantitative Analysis of Diversified Elaboration Representation Learning

1) Quantitative Analysis of the Module and Loss Designs: In this part, we conduct a quantitative analysis to the module and loss design of our proposed Diversified Elaboration Representation Learning. We use our XOV-Action without Diversified Elaboration Representation Learning as the baseline, which refers to the model using only Scene-Aware Video-text Alignment. As shown in Table VI, by introducing extra textual descriptions in inference, the baseline obtains better open-set action recognition performance compared with the one using only category names, i.e., from 39.21% to 40.56%. This is because that, textual descriptions encodes much more concept information about the corresponding action category than simple name texts, and thus model can better associate videos with open-set categories during testing. However, introducing textual descriptions in only inference obtains limited performance improvement in open-set generalization. Therefore, we propose to introduce textual descriptions and conduct alignment between videos and textual descriptions during training. As shown by the $\mathbf { \dot { \bar { \rho } } } \mathbf { F u l l } \mathbf { \bar { \rho } } $ in the table, our proposed Diversified Elaboration Representation Learning can effectively improve the open-set action recognition performance, and also modestly improve the closed-set recognition. This demonstrates that learning diverse action concepts can effectively boost the understanding of novel action concepts of open-set categories, thus improving the open-set action recognition performance.

TABLE VI  
QUANTITATIVE ANALYSIS OF THE MODULE AND LOSS DESIGNS OF OUR PROPOSED DIVERSIFIED ELABORATION REPRESENTATION LEARNING. IN THIS TABLE, WE REPORT THE AVERAGE CLOSED-SET AND OPEN-SET PERFORMANCE FOR EACH MODEL ON XOVABENCH. FOR ALL MODELS, WE USED BOTH CATEGORY NAMES AND TEXTUAL DESCRIPTIONS FOR ACTION CLASSIFICATION DURING INFERENCE.

<table><tr><td>Loss</td><td>Model</td><td>Closed-set</td><td>Open-set</td></tr><tr><td rowspan="2">w/o  $L_{\text{eva}}$ </td><td>-</td><td>55.06</td><td>40.56</td></tr><tr><td>w/o elaboration modules</td><td>55.78</td><td>40.85</td></tr><tr><td rowspan="2"> $L$ (Eq. (6))</td><td>w/o adaptive matching</td><td>54.97</td><td>41.13</td></tr><tr><td>Full</td><td>55.96</td><td>41.79</td></tr></table>

In addition, results in Table VI demonstrate the importance of Adaptive Elaboration Matching formulated by Eq. (2). Specifically, if we remove Adaptive Elaboration Matching from our model and follow the index order of descriptions to supervise elaboration representation learning, both closed-set and open-set performance decreases, as shown by the “w/o adaptive matching”. This is because the video elaboration modules in our model are not order-sensitive, so following the index order of descriptions to learn elaboration representations will result in confused representation space. In addition, we also analyze the case where video elaboration modules are removed, $i . e .$ , the loss $L _ { \mathrm { e v a } }$ is directly applied on the global video representation $f _ { \mathrm { v i d } } ( x )$ without extra specialized module designs. In this case, open-set performance decreases, which demonstrates the effectiveness of video elaboration modules for learning diverse action concepts.

![](images/0ebd9909a0cf4ce84781ae85d26c85c87a9ee55c6bcb3b7e74f1206840a48475.jpg)

![](images/99faa054a633bad2926918bec88045cb6fc5d1893368edd012dc2b3a4adb4a38.jpg)  
Fig. 4. Quantitative analysis of the coefficient $\lambda _ { \mathrm { e v a } }$ of our Elaborative Video-text Alignment loss. By default, we set $\lambda _ { \mathrm { e v a } } = 0 . 1$ by default according to the best trade-off between closed-set and open-set recognition.

![](images/610a81be0220fee1b7fbc68dcaa4ddf71e312fb896794745b12a6dce9c055b6a.jpg)

![](images/14bb5d45cf13747836d1a146132916a37a726f45d083b5efb74999f07e8d69bb.jpg)  
Fig. 5. Quantitative analysis of the number of selected elaboration representations $\hat { C }$ for training and inference. In this experiment, we use our XOV-Action without Diversified Elaboration Representation Learning as the baseline. In the two subfigures, the red and purple lines denotes the average closed-set and open-set performance of the baseline, respectively.

2) Effect of Elaborative Video-text Alignment: We conduct a quantitative analysis to the loss coefficient $\lambda _ { \mathrm { e v a } }$ in our proposed Elaborative Video-text Alignment loss, and the results are shown in Figure 4. As shown in the figure, by introducing our Elaborative Video-text Alignment loss, the open-set action recognition performance gets significant improvement, i.e., coefficient 0.0 vs. 0.1. These results demonstrate the effectiveness of learning diverse action concepts. As the loss coefficient goes larger, the open-set performance gets gradually higher, as the model focuses more on learning extra action concepts encoded by the textual descriptions. However, a large loss coefficient $\lambda _ { \mathrm { e v a } }$ will hinder the fitting of closed-set action categories during training, thus the closed-set performance gets gradually lower as the $\lambda _ { \mathrm { e v a } }$ goes larger. Overall, we set $\lambda _ { \mathrm { { e v a } } } = 0 . 1$ by default according to the best trade-off between closed-set and open-set recognition.

3) Effect of Confidence-aware Elaboration Selection: In Diversified Elaboration Representation Learning, we propose a confidence-aware elaboration selection strategy to select the top-C<sup>ˆ</sup> most relevant descriptions for each category, which aims to mitigate the negative effects of noisy textual descriptions during training. In this part, we conduct a quantitative analysis to the number of selected elaboration representations $\tilde { C }$ in our confidence-aware elaboration selection, and the results are shown in Figure 5. As shown in the figure, our proposed

![](images/d364d43d7ea66794c578a9edd6589e317999c24cbf6c1d0a94a3878a70daced8.jpg)  
Fig. 6. Quantitative analysis of the number of video elaboration modules. Our model performs generally better with more video elaboration modules.

![](images/6df557f5569ce16cdfbbf2b2c25749640558807cfd0827e900a4392f0a840e07.jpg)  
Fig. 7. Quantitative analysis of the number of textual descriptions for each action category. Our model performs generally better with more textual descriptions.

![](images/6bd4b9f0e438245e7a44bdaa6691a0635313c9853125ece6d32e3b9130acf8f1.jpg)

![](images/6444427e3f189a4315f0669d75cf7e67bb5b76938d082f907e6b82a595816b07.jpg)  
Fig. 8. Quantitative analysis of the ensemble coefficient $\lambda _ { \mathrm { e } }$ for action classification. In this experiment, we set the the baseline as our XOV-Action without Diversified Elaboration Representation Learning. In the two subfigures, the red and purple lines denotes the average closed-set and openset performance of the baseline, respectively.

Diversified Elaboration Representation Learning consistently improves open-set action recognition using different number of selected elaboration representations, compared with our model without Diversified Elaboration Representation Learning (i.e., the purple baseline in the figure). In addition, the results show that open-set performance is not sensitive to the number of selected elaboration representations when ${ \hat { C } } \geq 3 .$ . However, selecting a large number of elaboration representations leads to a noticeable performance drop in closed-set recognition. This is because GPT-generated textual descriptions may include incorrect or video-irrelevant details, and using a large number of elaboration representations introduces noise when fitting closed-set action videos. By default, we set $\hat { C } = 3$ , which provides the best trade-off between closed-set and open-set action recognition.

4) Effect of Video Elaboration Modules: In Figure 6, we show how the number of video elaboration modules affects the model performance. As shown in the figure, our model generally obtains better open-set accuracy with more video elaboration modules introduced. This is attributed to that our proposed Diversified Elaboration Representation Learning can better capture diverse action-related concepts when using more video elaboration modules, according to the guidance of textual descriptions. Empirically, we find that matching the number of video elaboration modules to the number of textual descriptions is an appropriate hyperparameter choice (i.e., $C = M = 1 0 )$ , and our results show that using too many video elaboration modules may not yield performance improvement.

5) Effect of the Number of Textual Descriptions: In Figure 7, we show how the number of textual descriptions affects the model performance. As shown in the figure, our model obtains better open-set accuracy when more textual descriptions introduced. This is intuitive since more textual descriptions contain more action-related details, and thus our proposed Diversified Elaboration Representation Learning can learn rich action concepts. As shown in the figure, when using a very large number of textual descriptions, the openset performance decreases slightly. This is because the textual descriptions are automatically generated by GPT, and it will introduce noise more than action-related concept information when too many textual descriptions are generated.

![](images/60d03710d2785e100488f6b88bf8904959aacbe54ea4ad1f0be13d03fe3ee084.jpg)

![](images/2f943431a392181ad08e4982f0878109de6c5b7bf4dc23fac8a5cbb47c5cf231.jpg)

![](images/15ce0c05cf4901c9a28ed06ebce6f5a3d92b6c6e84289968011f3c6699c83378.jpg)  
Fig. 9. Quantitative analysis of the coefficient $\lambda _ { \mathrm { s c e n e } }$ for the Scene-aware Discrimination loss by closed-set accuracy on four test domains. The horizontal axis shows the value of $\lambda _ { \mathrm { s c e n e } } .$ . Our model consistently obtains better closed-set performance on all test domains with a larger loss weight

![](images/8132f093e56ce0442d3d72168d399320132dc856440cf6c9fd66d0b0e97fcfd5.jpg)

![](images/a1dbd4fbe57343881677b1228d05c5276ad87b2dfea7cf044f4d057476ea5b6f.jpg)

![](images/15faae22aa2863fe433c727823857e14460cf409d6eab312bd34ff959d699e47.jpg)  
Fig. 11. Quantitative analysis of the coefficient $\lambda _ { \mathrm { a c t i o n } }$ for our Action-aware Discrimination loss. The horizontal axis shows the value of $\lambda _ { \mathrm { a c t i o n } } .$ Introducing our Action-aware Discrimination loss better maintains the open-set accuracy.

6) Effect of Score Ensemble: During inference, we use both the global video representation $z _ { x }$ and elaboration representations {zˆ<sup>i</sup> } for classification, and introduce an ensemble coefficient $\lambda _ { \mathrm { e } }$ to obtain final action classification score. In this part, we conduct a quantitative analysis to this ensemble coefficient, and the results are shown in Figure 8. As shown in the figure, our model can obtain better closed-set and open-set recognition performance using different values of ensemble coefficient, compared with our model without Diversified Elaboration Representation Learning (i.e., the red and purple baselines in the figure). By using a smaller ensemble coefficient, our model concentrates more on the open-set recognition since the elaboration score has a larger weight. For closed-set recognition, medium values of this coefficient yield the best results. By default, we set $\lambda _ { \mathrm { e } } ~ = ~ 0 . 4$ according to the best trade-off between closed-set and open-set performance.

Fig. 10. Quantitative analysis of the total number of scene suffixes. Our model obtains better closed-set accuracy with more scene suffixes.  
![](images/ae99120f0acd7adabfd331bbe391c1510d576dbba20477dde6bf61a014106a52.jpg)

## E. Quantitative Analysis of Scene-Aware Video-text Alignment

1) Effect of Scene-aware Discrimination: We conduct a quantitative analysis to the loss coefficient $\lambda _ { \mathrm { s c e n e } }$ in our proposed Scene-aware Discrimination loss, and the results on the four test domains are shown in Figure 9. The results in the figure show that our model consistently obtains better closed-set performance on all test domains with a larger loss weight. The comprehensive results demonstrate that our proposed Sceneaware Discrimination loss effectively tackles various types of domain gaps by mitigating scene bias in fitting training videos. However, using only the Scene-aware Discrimination loss will lead to performance drop in open-set generalization, thus we introduce the Action-aware Discrimination loss to address this issue.

![](images/174506078b933ce604f2a0de954ed6d7633c6daeafb2a0191a2ec04f66d3f23c.jpg)

![](images/9b547da91052b263bf2c1244d1650b1569a647bc7033ab92c377879c0f172f30.jpg)  
Fig. 12. Quantitative analysis of the margin δ used in our Action-aware Discrimination loss. The horizontal axis shows the value of δ. The performance of our Action-aware Discrimination loss is robust to the margin value.

2) Effect of Action-aware Discrimination: We first conduct a quantitative analysis to the loss coefficient $\lambda _ { \mathrm { a c t i o n } }$ in our proposed Action-aware Discrimination loss, and the results are shown in Figure 11. As shown in the figure, as the loss weight gradually increases, our model obtains higher open-set accuracy but lower closed-set accuracy. In addition, when using a very large weight $( e . g . , \lambda _ { \mathrm { a c t i o n } } \ = \ 0 . 8 )$ , our model cannot obtain higher open-set accuracy. This is because our Action-aware Discrimination loss acts as a constraint on the video representation space to compensate our Sceneaware Discrimination loss and is not specifically designed to improve open-set action recognition. Also, these results empirically reveal that a good trade-off between the Sceneaware Discrimination and Action-aware Discrimination losses is crucial for generalizable open-vocabulary action recognition.

Second, we conduct a quantitative analysis to the margin $\delta$ in our proposed Action-aware Discrimination loss, and the results are shown in Figure 12. As shown in the figure, when using different values of $\delta ,$ our model performs very similarly in terms of cross-domain closed-set accuracy. In addition, when using a relatively larger value of δ $( i . e . , 0 . 4 \leq \delta \leq 0 . 8 )$ our model can obtain promising results in terms of open-set accuracy. Overall, our model obtains comparable performance across different settings of $\delta ,$ which demonstrates that our Action-aware Discrimination loss is robust to the setting of

![](images/2a78494b4f2f1af9242141e63c7d7bbd615a7b63601ce691941ebeb746b4c86c.jpg)

![](images/4d6c7b2bdd1aced993db9c32a57dbdd26638f559dcae4078548d6207d5245f3d.jpg)

![](images/e1f7afa9b03d7da66043958319eadb617a55acfc71561a043c609f628bf70a8f.jpg)

Fig. 13. Qualitative analysis by attention visualization. In this figure, we show the original video and the attention visualization of baseline, our model with Scene-Aware Video-text Alignment (w/ SAVA) and our full XOV-Action. Best viewed in color.  
Fig. 14. Qualitative analysis of the Sceneaware Discrimination loss $L _ { \mathrm { s c e n e } }$ and Actionaware Discrimination loss L<sub>action</sub> by t-SNE. Best viewed in color.  
![](images/d043082f07f3bddb438da1617ed4cf455dcfcf57db484c65c55d600fbc2f8c65.jpg)  
TABLE VII  
QUANTITATIVE ANALYSIS OF OUR PROPOSED SCENE-AWARE DISCRIMINATION LOSS $L _ { \mathrm { S C E N E } }$ WITH DIFFERENT TYPES OF SCENE-ENCODED TEXT PROMPT TEMPLATES. THE REPORTED PERFORMANCE ARE IN TERMS OF THE CLOSED-SET ACCURACY.

3) Effect of Scene Suffixes: In Figure 10, we show how the suffix number affects the model performance. To show the loss effect more clearly, we set the loss coefficient $\lambda _ { \mathrm { s c e n e } } =$ 0.5 in this experiment. As shown in the figure, our model obtains better closed-set accuracy as the total number of scene suffixes increases. According to the design of our Scene-aware Discrimination loss, we distinguish videos apart from more scene-encoded text prompts in representation space with more scene suffixes used. In this way, videos are less likely to be confused in representation space, leading to stronger abilities for recognizing actions in unseen domains.

<table><tr><td>Models</td><td>Type</td><td>UCF</td><td>HMDB</td><td>ARID</td><td>NEC-Dr</td><td>AVG</td></tr><tr><td>Baseline</td><td>-</td><td>87.07</td><td>57.59</td><td>47.35</td><td>21.30</td><td>53.33</td></tr><tr><td rowspan="3">+Lscene</td><td>TypeA</td><td>89.04</td><td>56.88</td><td>47.35</td><td>19.91</td><td>53.30</td></tr><tr><td>TypeB</td><td>87.55</td><td>57.49</td><td>49.39</td><td>20.83</td><td>53.82</td></tr><tr><td>Ours</td><td>87.82</td><td>59.41</td><td>49.18</td><td>25.93</td><td>55.59</td></tr></table>

Fig. 15. Quantitative analysis of representation similarity between action videos and (a) concept texts / (b) scene texts.

the margin value.

![](images/365e055ac8589726fd58a1cfcfec60df5126aa858be28fcee962c1e8098b3f7f.jpg)

4) Effect of Scene-encoded Text Prompt Templates: In this part, we conduct a quantitative comparison between our proposed $L _ { \mathrm { s c e n e } }$ and its variants that use different types of sceneencoded text prompt templates. By default, our proposed $L _ { \mathrm { s c e n e } }$ uses the scene-encoded text prompts in the form of $\mathbf { \ddot { a } }$ video of a person [doing something] [at/on/in the/a scene].”, $e . g .$ for the action “abseiling”, a scene-encoded text prompt can be $\mathbf { \ddot { a } }$ video of a person abseiling in the park.”. We compare with variants using two different types of scene-encoded text prompts as follows: (1) TypeA: “a video of [the/a scene].”, $e . g .$ $\mathbf { \ddot { a } }$ video of the park.”; (2) TypeB: “a video of a person [at/on/in the/a scene].”, e.g., “a video of a person in the park.”. As shown in Table VII, the two variants exhibit much lower performance than our proposed $L _ { \mathrm { s c e n e } }$ in terms of the average cross-domain closed-set accuracy, which demonstrates that these variants cannot effectively tackle the domain gaps. The results imply that encoding the semantic information of the ground-truth action category into scene-encoded text prompts is important to the mitigation of scene bias.

## F. In-depth Verification Experiments

In this subsection, we conduct more verification experiments to demonstrate our model’s effectiveness in learning diverse action-related concepts and mitigating scene bias.

1) Qualitative Analysis by Attention Visualization: In this part, we conduct a detailed qualitative analysis to our model by attention visualization, and the results are shown in Figure 13. As shown in the figure, the baseline pays much attention to video scenes, which will easily cause recognition errors. In contrast, by introducing our proposed Scene-Aware Video-text Alignment (denoted by “w/ SAVA”), our model focuses more on the body parts of action performers rather than scenes, e.g., the hand holding a ball. This clearly demonstrate that our Scene-Aware Video-text Alignment can effectively mitigate scene bias and improve action recognition across domains. Moreover, by introducing our proposed Diversified Elaboration Representation Learning, our full model can capture more action-related concepts in videos, e.g., the basketball hoop related to the action “shoot ball”. As a result, our model can associate rich learned concepts with novel categories during testing, and improve the open-set action recognition across domains.

2) Qualitative Analysis by Distribution Visualization: First, we intuitively show how our proposed Action-aware Discrimination loss $L _ { \mathrm { a c t i o n } }$ affects representation space by t-SNE [131], and the results are given in Figure 14. As shown by the figure, without $L _ { \mathrm { a c t i o n } } .$ , our model with only the Scene-aware Discrimination loss may yield higher similarity between a video and some non-ground-truth action name texts than between the video and the scene-encoded text prompts, which would lead to confusion in video classification. In contrast, with the Action-aware Discrimination loss, our model produces a much more reasonable representation space: a video is farther from the non-ground-truth action texts than from the scene-encoded text prompts. This experiment intuitively demonstrates the rationale of our proposed Action-aware Discrimination loss.

3) Quantitative Analysis of Scenes and Concepts: To demonstrate our model’s effectiveness in learning diverse action-related concepts and mitigating scene bias, we conduct an in-depth quantitative analysis to the representation similarity between action videos and specific types of texts in our XOV-Action model. First, we analyze the representation similarity between videos and concept texts. We ask GPT-4 to generate a large pool of words or phrases that describes concepts related to human actions, e.g., football, goalposts, kickers and legs are some common concepts related to the action “kicking field goal”. Then, we extract the video representations of training samples, and compute the representation similarity between these video representations and the representations of the generated concept texts. We calculate the proportion of concept texts that have high video-text similarity scores with at least one video sample (we also set the threshold as 0.3), and the statistical results are shown in Figure 15 (a). As shown in the figure, compared with the baseline, the video representations of our proposed XOV-Action have high similarity with more concept texts, indicating that our model encodes more action-related concept information. This demonstrates that our proposed Diversified Elaboration Representation Learning can effectively capture rich action-related concepts, which boosts the understanding of open-set action categories.

Second, we focus on analyzing the representation similarity between videos and scene texts. We ask GPT-4 to generate a large pool of words or phrases about action scenarios, e.g., park, house, court. Similar to the analysis of the scene texts, we calculate the proportion of scene texts that have videotext similarity scores greater than 0.3 with at least one video sample. Figure 15 (b) shows the statistics of the baseline and our XOV-Action. As shown in the figure, compared with the baseline without our Scene-Aware Video-text Alignment, our XOV-Action learns video representations that are generally more dissimilar to scene text representations, which demonstrates our model’s effectiveness in mitigating the scene bias.

## VI. CONCLUSION

This work concentrated on a valuable but underexplored task, namely generalizable open-vocabulary action recognition. To address this task, we proposed a novel model named XOV-Action, aiming to overcome two critical challenges of this task: (1) understanding novel action concepts of openset categories, and (2) mitigating the scenario discrepancy between the training and test domains. First, XOV-Action proposed to capture rich action-related concepts by learning diversified elaboration representations, enhancing its recognition of open-set action categories. Second, XOV-Action proposed to mitigate the scene bias by learning scene-agnostic video representations, thereby improving the generalization across video domains. In addition, we contributed a new benchmark named XOVABench, which provided a comprehensive way to evaluate models across various types of domain gaps and different action categories. Extensive quantitative and qualitative experiments showed that our proposed XOV-Action can effectively improve both closed-set and open-set action recognition performance across domains. We believe that our work will serve as a catalyst for further advancement in the field of robust video understanding, and we hope it will inspire future innovative solutions for the vision-language understanding field.

## ACKNOWLEDGMENTS

The authors would like to thank Bing Zhao and Zhi-Wei Xia for their support in writing and experiments. This work was supported by the New Generation Artificial Intelligence-National Science and Technology Major Project (2025ZD0123100). Additionally, this work was partially supported by NSFC (92470202), National Key Research and Development Program of China (2023YFA1008503), Guangdong NSF Project (No. 2023B1515040025), Guangdong Key Research and Development Program (No. 2024B0101040004, No. 2025B0909020002).

## APPENDIX

## A.1. TEXT AND PROMPT FORMULATION AND USAGE

Our proposed XOV-Action model utilizes different types of text or prompts to facilitate the learning of generalizable openvocabulary action recognition models. Specifically, these text and prompts can be divided into three types:

1) Text prompt of action name: This kind of text prompt is in the form of “a video of a person [doing something].”. For example, for the action category “long jump”, the text prompt of this action is $\stackrel { 6 6 } { \circ } \scriptscriptstyle \partial$ video of a person long jump.”.

2) Textual descriptions of action: For each action category, we adopt M automatically generated textual descriptions for training (generated by GPT-4 [123]). For example, for the action category “long jump”, these textual descriptions can be:

• a person is seen accelerating on a track, taking a leap, and landing in a distant sand pit.

• a person is sprinting before jumping as far as possible into a sand-filled pit.

• ...

• a person propels their body forward in a horizontal leap, landing in a sand pit.

As shown above, these descriptions concisely capture the key visual aspects of the action, and these descriptions are used in our proposed Diversified Elaboration Representation Learning.

3) Scene-encoded text prompt: For each action category, we introduce N scene-encoded text prompts for training. Each scene-encoded text prompt consists of an action and a scene. For example, for the action category “long jump”, these sceneencoded text prompts can be:

• a video of a person long jumping in the park.

• ...

• a video of a person long jumping on the street.

• a video of a person long jumping in the kitchen.

scene-encoded text prompts are used in our proposed Scene-Aware Video-text Alignment.

## A.2. ANALYSIS OF DESCRIPTION USAGE

In this part, we conduct additional experiments to analyze the usage of action descriptions. First, we quantitatively analyze the effects of description sources by using different large language models to generate textual descriptions. As shown in Table A1, our proposed XOV-Action model achieves comparable performance when using descriptions generated by different large language models, and consistently outperforms the previous state-of-the-art (i.e., Open-VCLIP). These results highlight the effectiveness of our model and its robustness to the choice of description source.

TABLE A1  
QUANTITATIVE ANALYSIS OF XOV-ACTION USING ACTION DESCRIPTIONS GENERATED BY DIFFERENT LARGE LANGUAGE MODELS. IN THIS TABLE, WE REPORT THE AVERAGE CLOSED-SET, OPEN-SET AND OVERALL PERFORMANCE ON XOVABENCH.

<table><tr><td>Models</td><td>Description Sources</td><td>Closed</td><td>Open</td><td>All</td></tr><tr><td>Open-VCLIP [107]</td><td>-</td><td>53.00</td><td>39.08</td><td>46.95</td></tr><tr><td rowspan="6">XOV-Action</td><td>DeepSeek-V3.2 [132]</td><td>54.63</td><td>42.03</td><td>49.22</td></tr><tr><td>gemma-3-27b-it [133]</td><td>54.57</td><td>41.96</td><td>49.16</td></tr><tr><td>Qwen2.5-32B-Instruct [134]</td><td>54.66</td><td>42.29</td><td>49.39</td></tr><tr><td>Qwen3-30B-A3B-Instruct [135]</td><td>54.88</td><td>42.06</td><td>49.36</td></tr><tr><td>Qwen3-4B-Instruct [135]</td><td>54.50</td><td>42.22</td><td>49.26</td></tr><tr><td>GPT-4 [123] (default)</td><td>55.96</td><td>41.79</td><td>49.80</td></tr></table>

Second, we conduct experiments to demonstrate our model robustness to prompt phrasing and model temperature during the LLM-based description generation process. In this experiment, we use the open-source Qwen3-30B-A3B-Instruct-2507 [135] as the LLM generator. Specifically, we adopt three different prompts in this experiments, and try three different model temperatures, i.e., 0.2, 0.7 and 1.0. The default model temperature is 0.7. We adopt three different prompts in this experiments, which are shown in Prompt A.2.1, Prompt A.2.2 and Prompt A.2.3. PromptV1 is the default prompt we used, and PromptV2 and PromptV3 are automatically generated by LLMs with minimal manual refinement. In each prompt, the “{}” will be filled by the action label text when generating descriptions for a specific action category. As shown in Table A2, our XOV-Action obtains comparable performance across different settings, with the overall performance gap between models being less than 0.8%. This demonstrates our model’s robustness to prompt phrasing and model temperature for LLMs.

## Prompt A.2.1

Prompt V1: Generate 10 descriptions for {} human action in a video, each description should be concise, that is less than 40 words. These descriptions are required to be diverse, but should be distinguishable to describe {} human action. Each description consists of only one sentence, starts with ’A person’, and ends up with a full stop ’.’. Please return these 10 sentence in a JSON list form. The list would only contains 10 strings, without any other outputs.

## Prompt A.2.2

Prompt V2: You are given an action label: “{}”.

Constraints:

\- Each sentence must be <= 40 words.

\- Each sentence must start with “A person” and end with a period “.”.

\- Describe only observable human action and context; do not speculate about invisible causes or intentions.

\- The 10 sentences must be diverse in wording and details but clearly depict the same action “{}”.

Output format:

Return ONLY a JSON array of 10 strings, with no extra text, no numbering, and no Markdown.

## Prompt A.2.3

Third, we conduct additional experiments using humanwritten descriptions and templated LLM-generated descriptions. In this experiment, we also use the open-source Qwen3- 30B-A3B-Instruct-2507 [135] as the LLM generator.

• For human-written descriptions, we use action definition from Chen et al. [73], which are refined by human labor.

## TABLE A2

RESULTS OF OUR XOV-ACTION MODEL USING DIFFERENT PROMPTS AND DIFFERENT MODEL TEMPERATURES DURING THE LLM-BASED DESCRIPTION GENERALIZATION PROCESS. WE ADOPT THE OPEN-SOURCE QWEN3-30B-A3B-INSTRUCT-2507 [135] AS THE LLM GENERATOR IN THIS EXPERIMENT. THE DEFAULT MODEL TEMPERATURE IS 0.7. IN THIS TABLE, WE REPORT THE AVERAGE CLOSED-SET, OPEN-SET AND OVERALL PERFORMANCE ON XOVABENCH.

<table><tr><td>Models</td><td>Description Sources</td><td>Closed</td><td>Open</td><td>All</td></tr><tr><td>Open-VCLIP [107]</td><td>-</td><td>53.00</td><td>39.08</td><td>46.95</td></tr><tr><td rowspan="5">XOV-Action</td><td>Prompt V1 (default)</td><td>54.88</td><td>42.06</td><td>49.36</td></tr><tr><td>Prompt V2</td><td>55.21</td><td>42.12</td><td>49.50</td></tr><tr><td>Prompt V3</td><td>54.60</td><td>41.03</td><td>48.79</td></tr><tr><td>Prompt V1 &amp; Temp 0.2</td><td>55.45</td><td>41.90</td><td>49.62</td></tr><tr><td>Prompt V1 &amp; Temp 1.0</td><td>54.75</td><td>42.17</td><td>49.34</td></tr></table>

As there is only one action definition for each action category, we replace one Qwen-generated description with the action definition for training. As shown by “Prompt V1 w/ Human” in Table A3, such a model variant achieves very similar performance to the default version, which demonstrates that human-written descriptions can be reliable for model training. In practice, we recommend prioritizing LLMs for description generation, since this avoids costly human labor and the capability is already relatively mature.

• Additionally, we conduct another experiment where we constrain the LLM to generate descriptions based on a fixed set of 10 templates. The specific prompt is shown in Prompt A.2.4, i.e., Prompt V4. As shown in Table A3, this template-based prompting degrades the model’s open-set performance. We attribute this to the reduced description diversity for each action category, which limits the model’s ability to learn a broader range of action-related concepts. Therefore, in practical applications, we do not recommend using templated LLM prompts.

## Prompt A.2.4

Prompt V4: Generate 10 descriptions for {} human action in a video, each description should be concise, that is less than 40 words. These descriptions are required to be diverse, but should be distinguishable to describe {} human action. Each description consists of only one sentence, starts with ’A person’, and ends up with a full stop ’.’. Here are some templates for generating descriptions, and please a different template each time:

\- "A person is {{motion}} with {{object}}, {{motion}}."

\- "A person performs {{motion}} on {{object}}, {{motion}}."

\- "A person is performing {{motion}} using {{object}}, {{motion}}."

\- "A person carries out {{motion}} involving {{object}}, {{motion}}."

\- "A person is engaged in {{motion}} with {{object}}, {{motion}}."

\- "A person continues {{motion}} with {{object}}, {{motion}}."

\- "A person starts {{motion}} with {{object}}, {{motion}}."

\- "A person focuses on {{motion}} with {{object}}, {{motion}}."

\- "A person completes {{motion}} with {{object}}, {{motion}}."

\- "A person {{motion}} with {{object}}, {{motion}}."

Please fill in the blanks {{motion}} or {{object}} adaptively according to the given action category. Please return these 10 sentence in a JSON list form. The list should only contains 10 strings, without any other outputs.

Finally, we conduct additional experiments to analyze the robustness to description redundancy and hallucination, and the results demonstrate that our model is robust to a small amount of redundancy or hallucination.

• For description redundancy, we intentionally inject exact duplicate descriptions into the description set. Specifically, for each action category, we deliberately replace several of the ten descriptions with an identical description, making some descriptions fully redundant (i.e., semantically identical) to each other. As shown in Table A4, open-set performance gradually decreases as more duplicate descriptions are used, since the description set becomes less diverse. However, the overall degradation is modest and our model remains effective compared with previous state-of-the-art models, demonstrating that our approach is robust to mild description redundancy.

TABLE A3  
RESULTS OF OUR XOV-ACTION MODEL USING HUMAN-WRITTEN DESCRIPTIONS AND TEMPLATED LLM-GENERATED DESCRIPTIONS. IN THIS TABLE, WE REPORT THE AVERAGE CLOSED-SET, OPEN-SET AND OVERALL PERFORMANCE ON XOVABENCH.

<table><tr><td>Models</td><td>Description Sources</td><td>Closed</td><td>Open</td><td>All</td></tr><tr><td>Open-VCLIP [107]</td><td>-</td><td>53.00</td><td>39.08</td><td>46.95</td></tr><tr><td rowspan="3">XOV-Action</td><td>Prompt V1 (default)</td><td>54.88</td><td>42.06</td><td>49.36</td></tr><tr><td>Prompt V1 w/ Human</td><td>54.95</td><td>41.87</td><td>49.40</td></tr><tr><td>Prompt V4 (Templated)</td><td>55.16</td><td>40.82</td><td>48.93</td></tr></table>

TABLE A4

RESULTS OF OUR XOV-ACTION MODEL USING DUPLICATE TEXTUAL DESCRIPTIONS. THE COLUMN “# OF REPEATED DESCS” INDICATES THE NUMBER OF REPEATED DESCRIPTIONS AMONG THE INTRODUCED DESCRIPTIONS FOR EACH ACTION CATEGORY. IN THIS TABLE, WE REPORT THE AVERAGE CLOSED-SET, OPEN-SET AND OVERALL PERFORMANCE ON XOVABENCH.

<table><tr><td>Models</td><td># of Repeated Descs</td><td>Closed</td><td>Open</td><td>All</td></tr><tr><td>Open-VCLIP [107]</td><td>-</td><td>53.00</td><td>39.08</td><td>46.95</td></tr><tr><td rowspan="4">XOV-Action</td><td>0 (default)</td><td>55.96</td><td>41.79</td><td>49.80</td></tr><tr><td>1</td><td>56.12</td><td>41.67</td><td>49.88</td></tr><tr><td>3</td><td>55.88</td><td>41.31</td><td>49.55</td></tr><tr><td>5</td><td>55.73</td><td>41.08</td><td>49.28</td></tr></table>

• For description hallucination, we intentionally inject exact incorrect descriptions into the description set. Specifically, for each action category, we deliberately replace several of the ten descriptions with incorrect descriptions, i.e., descriptions from other action categories. As shown in Table A5, a small amount of hallucination does not have a significant impact on our model’s performance, e.g., one or two incorrect descriptions per category. This is attributed to the design of our elaboration selection strategy, which can filter out some irrelevant descriptions for training videos. As more incorrect descriptions are introduced, performance drops significantly, especially on open-set categories, since these incorrect descriptions are more likely to provide misleading supervision signals. These results demonstrate our model’ robustness to a small amount of description hallucination.

Overall, these above results demonstrate that our model is robust to the adopted action descriptions.

## A.3. EFFECT OF ELABORATION SELECTION STRATEGY

In this part, we quantitatively analyze the effects of elaboration selection strategies on our model. As shown in Table A6, random selection reduces open-set performance, since C<sup>ˆ</sup> descriptions are randomly selected irrespective of their similarity. In this case, closed-set performance also decreases

## TABLE A5

RESULTS OF OUR XOV-ACTION MODEL USING SOME HALLUCINATED TEXTUAL DESCRIPTIONS. THE COLUMN “# OF HALLUCINATED DESCS” INDICATES THE NUMBER OF HALLUCINATED DESCRIPTIONS AMONG THE INTRODUCED DESCRIPTIONS FOR EACH ACTION CATEGORY. IN THIS TABLE, WE REPORT THE AVERAGE CLOSED-SET, OPEN-SET AND OVERALL PERFORMANCE ON XOVABENCH.

<table><tr><td>Models</td><td># of Hallucinated Descs</td><td>Closed</td><td>Open</td><td>All</td></tr><tr><td>Open-VCLIP [107]</td><td>-</td><td>53.00</td><td>39.08</td><td>46.95</td></tr><tr><td rowspan="4">XOV-Action</td><td>0 (default)</td><td>55.96</td><td>41.79</td><td>49.80</td></tr><tr><td>1</td><td>56.14</td><td>41.75</td><td>49.89</td></tr><tr><td>2</td><td>55.72</td><td>41.96</td><td>49.80</td></tr><tr><td>4</td><td>54.30</td><td>39.41</td><td>47.99</td></tr></table>

TABLE A6

QUANTITATIVE ANALYSIS OF THE ELABORATION SELECTION STRATEGY. IN THIS TABLE, WE REPORT THE AVERAGE CLOSED-SET AND OPEN-SET PERFORMANCE ON XOVABENCH.

<table><tr><td>Strategy</td><td>Closed-set</td><td>Open-set</td></tr><tr><td>top- $\hat{C}$  highest similarity (default)</td><td>55.96</td><td>41.79</td></tr><tr><td>random</td><td>53.83</td><td>41.41</td></tr><tr><td>top- $\hat{C}$  lowest similarity</td><td>52.57</td><td>41.16</td></tr></table>

![](images/4ba8f5cc0ec906b0ef02f679c247ebf819fa126242b7dd5ce47a76e0e516b3a4.jpg)

![](images/f2e1b23b85ff46acda306bbbe08d2df7a2e425bc98df805e4bff662dbcd55a50.jpg)  
Fig. A1. Quantitative analysis of the joint effects of $\lambda _ { \mathrm { s c e n e } }$ and $\lambda _ { \mathrm { a c t i o n } }$ . In this figure, we report the average closed-set and open-set performance on XOVABench.

![](images/a1c41c79ba7dd79f8f73f01204b85d89baf4e1920046c3a8874153582e5cd961.jpg)

![](images/057854dc366af914670cab8ccd491df06b1aebf0880bf5da79b740423b3f3b98.jpg)  
Fig. A2. Quantitative analysis of the joint effects of $\lambda _ { \mathrm { s c e n e } } , \lambda _ { \mathrm { a c t i o n } }$ and $\lambda _ { \mathrm { { e v a } } } .$ In this experiment, we set $\lambda _ { \mathrm { s c e n e } } = \lambda _ { \mathrm { a c t i o n } } ,$ and we report the average closedset and open-set performance on XOVABench.

since noisy or irrelevant descriptions are likely to be selected, leading to unreliable video-text alignment during training. Furthermore, selecting the $\mathrm { t o p } { \cdot } \hat { C }$ lowest similarity obtains lower performance in both closed-set and open-set recognition. This is because the selected descriptions are the most irrelevant to the video content, which introduces more severe noise than random selection. Overall, these results demonstrate the effectiveness of our employed strategy.

TABLE A7  
ADDITIONAL ABLATION STUDY ON MORE LOSS FUNCTION COMBINATIONS. IN THIS TABLE, WE REPORT THE AVERAGE CLOSED-SET AND OPEN-SET PERFORMANCE FOR EACH MODEL VARIANT ON XOVABENCH.

<table><tr><td>Models</td><td> $L_{scene}$ </td><td> $L_{action}$ </td><td> $L_{eva}$ </td><td>Closed-set</td><td>Open-set</td></tr><tr><td>w/o  $L_{scene}$ </td><td>✗</td><td>√</td><td>√</td><td>53.85</td><td>42.41</td></tr><tr><td>w/o  $L_{action}$ </td><td>√</td><td>✗</td><td>√</td><td>56.16</td><td>40.99</td></tr><tr><td>w/o  $L_{eva}$ </td><td>√</td><td>√</td><td>✗</td><td>55.20</td><td>40.66</td></tr><tr><td>Full</td><td>√</td><td>√</td><td>√</td><td>55.96</td><td>41.79</td></tr></table>

## A.4. ANALYSIS OF JOINT HYPERPARAMETER EFFECTS

In this part, we conduct additional experiments to analyze the joint effects of the three loss coefficients, namely $\lambda _ { \mathrm { s c e n e } } ,$ $\lambda _ { \mathrm { a c t i o n } }$ and $\lambda _ { \mathrm { e v a } }$ . Figure A1 shows the joint effects of $\lambda _ { \mathrm { s c e n e } }$ and $\lambda _ { \mathrm { a c t i o n } } .$ . As shown in the figure, closed-set performance generally improves as $\lambda _ { \mathrm { s c e n e } }$ increases. This is because our Sceneaware Discrimination loss $L _ { \mathrm { s c e n e } }$ encourages the video encoder to downweight the attention on scene information in videos, thereby improving generalization in unseen domains. However, introducing $L _ { \mathrm { s c e n e } }$ reduces open-set performance, as open-set performance drops as $\lambda _ { \mathrm { s c e n e } }$ increases. Therefore, we propose the Action-aware Discrimination loss $L _ { \mathrm { a c t i o n } }$ to constrain video representation learning and alleviate the degradation of openset performance. As shown in Figure A1, increasing λ<sub>action</sub> yields relatively higher open-set performance.

Figure A2 shows the joint effects of $\lambda _ { \mathrm { s c e n e } } , \lambda _ { \mathrm { a c t i o n } }$ and $\lambda _ { \mathrm { { e v a } } } ,$ where we set $\lambda _ { \mathrm { s c e n e } } ~ = ~ \lambda _ { \mathrm { a c t i o n } }$ . As shown in the figure, increasing $\lambda _ { \mathrm { s c e n e } }$ and $\lambda _ { \mathrm { a c t i o n } }$ improves closed-set performance but leads to a decrease in open-set performance, suggesting that these coefficients should be balanced in practice. Additionally, varying $\lambda _ { \mathrm { e v a } }$ within the range [0.1, 0.3] leads to only minor performance changes, demonstrating that our method is relatively insensitive to this coefficient. Note that removing the loss $L _ { \mathrm { e v a } }$ causes a significant performance drop, $i . e . , 5 5 . 9 6 \%  5 5 . 2 0 \%$ in closed-set recognition and $4 1 . 7 9 \%  4 0 . 6 6 \%$ in open-set recognition, which verifies the effectiveness of $L _ { \mathrm { e v a } }$

## A.5. MORE COMBINATIONS OF LOSSES

In this part, we conducted an additional ablation study by including more combinations of loss functions. Specifically, our XOV-Action model includes three loss contributions, namely Scene-aware Discrimination $L _ { \mathrm { s c e n e } } ,$ , Action-aware Discrimination $L _ { \mathrm { a c t i o n } } .$ , and Elaborative Video-text Alignment $L _ { \mathrm { e v a } }$ . We start from the full model and individually remove each loss term. With the exception of the modified loss functions, we keep the network architecture and inference strategy identical in this experiment. The results are summarized in Table A7, and we discuss the results as follows:

• Removing the Scene-aware Discrimination loss $L _ { \mathrm { s c e n e } }$ from our model leads to a significant performance drop in cross-domain closed-set action recognition, as shown by “w/o $L _ { \mathrm { s c e n e } } ? $ . This is because $L _ { \mathrm { s c e n e } }$ is designed to mitigate domain gaps by downweighting the attention on scene information in videos. Consequently, its removal reduces the closed-set performance in unseen domains.

• Removing the Action-aware Discrimination loss $L _ { \mathrm { a c t i o n } }$ from our model leads to a significant performance drop in open-set action recognition, as shown by “w/o $L _ { \mathrm { a c t i o n } } ? 3$ This is because $L _ { \mathrm { a c t i o n } }$ is designed to alleviate the degradation of cross-domain open-set performance caused by the introduction of $L _ { \mathrm { s c e n e } }$ . Thus, the removal of $L _ { \mathrm { a c t i o n } }$ degrades open-set performance.

• Removing the Elaborative Video-text Alignment $L _ { \mathrm { e v a } }$ loss from our model leads to a performance drop in both closed-set and open-set action recognition, with the drop in open-set recognition being more significant, as shown by “w/o $L _ { \mathrm { e v a } } \mathbf { \vec { \mu } } _ { \mathrm { ~ } }$ . This is because our Elaborative Video-text Alignment is designed to learn more action-related concept information and facilitate the understanding of novel concepts in open-set categories. Therefore, removing $L _ { \mathrm { e v a } }$ hinders the model’s ability to generalize to diverse openset categories.

Overall, these additional results demonstrate the effectiveness of our loss contributions and are consistent with our previous findings.

## A.6. GENERALIZABILITY IN EGOCENTRIC SCENARIOS

To further demonstrate the effectiveness of our model, we construct an additional egocentric benchmark based on EPIC-Kitchen [136] and evaluate model performance in egocentric scenarios using this benchmark. Our evaluation shows that our proposed XOV-Action can also outperform previous stateof-the-art models in egocentric scenarios. Specifically, the additional egocentric benchmark, which we refer to as EPIC-XOV, is a first-person view benchmark for generalizable open-vocabulary action recognition. Its videos are captured in kitchen environments from an egocentric viewpoint, and it comprises 34 closed-set categories and 63 open-set categories. The comparison results between our model and previous stateof-the-art models are summarized in Table A8. As shown in the table, all models exhibit relatively limited performance on this benchmark, particularly on the open-set categories. This can be largely attributed to the significant domain gap between the training dataset and EPIC-XOV. Specifically, the training data mainly consist of third-person daily activity videos, where visual cues are human-centric, whereas EPIC-XOV contains egocentric videos that predominantly capture interactions among hands, objects, and the surrounding environment without seeing action performers themselves. As a result, degraded performance is expected under such a challenging cross-domain setting. Despite this difficulty, our method still achieves the best performance on this benchmark, demonstrating its effectiveness and generalizability. We leave further improving the transferability of action recognition models from third-person videos to egocentric videos as an important direction for future work.

## REFERENCES

[1] Yu Kong and Yun Fu. Human action recognition and prediction: A survey. International Journal of Computer Vision, 130(5):1366–1401, 2022.

TABLE A8  
COMPARISON WITH KINETICS400-TRAINED OPEN-VOCABULARY ACTION RECOGNITION MODELS ON EPIC-XOV EGOCENTRIC BENCHMARK. IN THIS TABLE, WE REPORT THE AVERAGE CLOSED-SET, OPEN-SET AND OVERALL PERFORMANCE ON EPIC-XOV.

<table><tr><td>Models</td><td>Closed-set</td><td>Open-set</td><td>All</td></tr><tr><td>ActionCLIP [4]</td><td>17.64</td><td>0.56</td><td>9.71</td></tr><tr><td>ViFiCLIP [8]</td><td>14.48</td><td>0.71</td><td>8.08</td></tr><tr><td>OpenVCLIP [9]</td><td>20.57</td><td>0.16</td><td>11.09</td></tr><tr><td>FROSTER [60]</td><td>19.01</td><td>0.16</td><td>10.26</td></tr><tr><td>XOV-Action (Ours)</td><td>27.12</td><td>1.02</td><td>15.00</td></tr></table>

[2] Zehua Sun, Qiuhong Ke, Hossein Rahmani, Mohammed Bennamoun, Gang Wang, and Jun Liu. Human action recognition from various data modalities: A review. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3):3200–3225, 2023.

[3] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021.

[4] Mengmeng Wang, Jiazheng Xing, and Yong Liu. ActionCLIP: A new paradigm for video action recognition. CoRR, abs/2109.08472, 2021.

[5] Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, and Haibin Ling. Expanding language-image pretrained models for general video recognition. In European Conference on Computer Vision, 2022.

[6] Chen Ju, Tengda Han, Kunhao Zheng, Ya Zhang, and Weidi Xie. Prompting visual-language models for efficient video understanding. In European Conference on Computer Vision, 2022.

[7] Wenhao Wu, Zhun Sun, and Wanli Ouyang. Revisiting classifier: Transferring vision-language models for video recognition. In AAAI Conference on Artificial Intelligence, 2023.

[8] Hanoona Abdul Rasheed, Muhammad Uzair Khattak, Muhammad Maaz, Salman H. Khan, and Fahad Shahbaz Khan. Fine-tuned CLIP models are efficient video learners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[9] Zejia Weng, Xitong Yang, Ang Li, Zuxuan Wu, and Yu-Gang Jiang. Open-VCLIP: Transforming CLIP to an open-vocabulary video model via interpolated weight optimization. In International Conference on Machine Learning, 2023.

[10] Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. In Advances in Neural Information Processing Systems, 2014.

[11] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. Temporal Segment Networks: Towards good practices for deep action recognition. In European Conference on Computer Vision, 2016.

[12] Ji Lin, Chuang Gan, and Song Han. TSM: Temporal shift module for efficient video understanding. In IEEE/CVF International Conference on Computer Vision, 2019.

[13] João Carreira and Andrew Zisserman. Quo Vadis, Action Recognition? A new model and the kinetics dataset. In IEEE Conference on Computer Vision and Pattern Recognition, 2017.

[14] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In International Conference on Machine Learning, 2021.

[15] Min-Hung Chen, Zsolt Kira, Ghassan Alregib, Jaekwon Yoo, Ruxin Chen, and Jian Zheng. Temporal attentive alignment for large-scale video domain adaptation. In IEEE/CVF International Conference on Computer Vision, 2019.

[16] Zhiyu Yao, Yunbo Wang, Jianmin Wang, Philip S. Yu, and Mingsheng Long. VideoDG: Generalizing temporal relations in videos to novel domains. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(11):7989–8004, 2022.

[17] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. UCF101: A dataset of 101 human actions classes from videos in the wild. CoRR, abs/1212.0402, 2012.

[18] Hildegard Kuehne, Hueihan Jhuang, Estíbaliz Garrote, Tomaso A. Poggio, and Thomas Serre. HMDB: A large video database for human motion recognition. In IEEE International Conference on Computer Vision, 2011.

[19] Yuecong Xu, Jianfei Yang, Haozhi Cao, Kezhi Mao, Jianxiong Yin, and

Simon See. ARID: A comprehensive study on recognizing actions in the dark and a new benchmark dataset. CoRR, abs/2006.03876, 2020

[20] Jinwoo Choi, Gaurav Sharma, Manmohan Chandraker, and Jia-Bin Huang. Unsupervised and semi-supervised domain adaptation for action recognition from drones. In IEEE Winter Conference on Applications of Computer Vision, 2020.

[21] Minghao Zhu, Zhengpu Wang, Mengxian Hu, Ronghao Dang, Xiao Lin, Xun Zhou, Chengju Liu, and Qijun Chen. MoTE: Reconciling generalization with specialization for visual-language to video knowledge transfer. In Advances in Neural Information Processing Systems, 2024.

[22] Waqas Sultani and Imran Saleemi. Human action recognition across datasets by foreground-weighted histogram decomposition. In IEEE Conference on Computer Vision and Pattern Recognition, 2014.

[23] Yingwei Li, Yi Li, and Nuno Vasconcelos. RESOUND: Towards action recognition without representation bias. In European Conference on Computer Vision, 2018.

[24] Jinwoo Choi, Chen Gao, Joseph C. E. Messou, and Jia-Bin Huang. Why Can’t I Dance in the Mall? Learning to mitigate scene bias in action recognition. In Advances in Neural Information Processing Systems, 2019.

[25] Jinwoo Choi, Gaurav Sharma, Samuel Schulter, and Jia-Bin Huang. Shuffle and attend: Video domain adaptation. In European Conference on Computer Vision, 2020.

[26] Jinpeng Wang, Yuting Gao, Ke Li, Yiqi Lin, Andy J. Ma, Hao Cheng, Pai Peng, Feiyue Huang, Rongrong Ji, and Xing Sun. Removing the background by adding the background: Towards background robust self-supervised video representation learning. In IEEE Conference on Computer Vision and Pattern Recognition, 2021.

[27] Haoxin Li, Yuan Liu, Hanwang Zhang, and Boyang Li. Mitigating and evaluating static bias of action representations in the background and the foreground. In IEEE/CVF International Conference on Computer Vision, 2023.

[28] Yuanhao Zhai, Ziyi Liu, Zhenyu Wu, Yi Wu, Chunluan Zhou, David S. Doermann, Junsong Yuan, and Gang Hua. SOAR: Scene-debiasing open-set action recognition. In IEEE/CVF International Conference on Computer Vision, 2023.

[29] Yuan-Ming Li, Wei-Jin Huang, An-Lan Wang, Ling-An Zeng, Jingke Meng, and Wei-Shi Zheng. EgoExo-Fitness: Towards egocentric and exocentric full-body action understanding. In European Conference on Computer Vision, 2024.

[30] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. ImageNet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, 2012.

[31] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations, 2015.

[32] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott E. Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In IEEE Conference on Computer Vision and Pattern Recognition, 2015.

[33] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition, 2016.

[34] Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, 2014.

[35] Kyunghyun Cho, Bart van Merrienboer, Çaglar Gülçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In Empirical Methods in Natural Language Processing, 2014.

[36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.

[37] Bolei Zhou, Alex Andonian, Aude Oliva, and Antonio Torralba. Temporal relational reasoning in videos. In European Conference on Computer Vision, 2018.

[38] Jiaming Zhou, Kun-Yu Lin, Haoxin Li, and Wei-Shi Zheng. Graphbased high-order relation modeling for long-term action recognition. In IEEE Conference on Computer Vision and Pattern Recognition, 2021.

[39] Wei-Jin Huang, Yuan-Ming Li, Zhi-Wei Xia, Yu-Ming Tang, Kun-Yu Lin, Jian-Fang Hu, and Wei-Shi Zheng. Modeling multiple normal action representations for error detection in procedural tasks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[40] Hao Shao, Shengju Qian, and Yu Liu. Temporal interlacing network. In AAAI Conference on Artificial Intelligence, 2020.

[41] Swathikiran Sudhakaran, Sergio Escalera, and Oswald Lanz. Gate-shift networks for video action recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

[42] Du Tran, Lubomir D. Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning spatiotemporal features with 3D convolutional networks. In IEEE International Conference on Computer Vision, 2015.

[43] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A closer look at spatiotemporal convolutions for action recognition. In IEEE Conference on Computer Vision and Pattern Recognition, 2018.

[44] Du Tran, Heng Wang, Matt Feiszli, and Lorenzo Torresani. Video classification with channel-separated convolutional networks. In IEEE/CVF International Conference on Computer Vision, 2019.

[45] Christoph Feichtenhofer. X3D: Expanding architectures for efficient video recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

[46] Shiwen Zhang, Sheng Guo, Weilin Huang, Matthew R. Scott, and Limin Wang. V4D: 4D convolutional neural networks for videolevel representation learning. In International Conference on Learning Representations, 2020

[47] Kunchang Li, Xianhang Li, Yali Wang, Jun Wang, and Yu Qiao. CT-Net: Channel tensorization network for video classification. In International Conference on Learning Representations, 2021.

[48] Rohit Girdhar, João Carreira, Carl Doersch, and Andrew Zisserman. Video action transformer network. In IEEE Conference on Computer Vision and Pattern Recognition, 2019.

[49] Daniel Neimark, Omri Bar, Maya Zohar, and Dotan Asselmann. Video transformer network. CoRR, abs/2102.00719, 2021.

[50] Hao Zhang, Yanbin Hao, and Chong-Wah Ngo. Token shift transformer for video classification. In ACM International Conference on Multimedia, 2021.

[51] Yanyi Zhang, Xinyu Li, Chunhui Liu, Bing Shuai, Yi Zhu, Biagio Brattoli, Hao Chen, Ivan Marsic, and Joseph Tighe. VidTr: Video transformer without convolutions. In IEEE/CVF International Conference on Computer Vision, 2021.

[52] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lucic, and Cordelia Schmid. ViViT: A video vision transformer. In IEEE/CVF International Conference on Computer Vision, 2021.

[53] Jiaming Zhou, Kun-Yu Lin, Yukun Qiu, and Wei-Shi Zheng. Twinformer: Fine-to-coarse temporal modeling for long-term action recognition. IEEE Transactions on Multimedia, 26:2715–2728, 2024.

[54] An-Lan Wang, Kun-Yu Lin, Jia-Run Du, Jingke Meng, and Wei-Shi Zheng. Event-guided procedure planning from instructional videos with text supervision. In IEEE/CVF International Conference on Computer Vision, 2023.

[55] Ziyi Lin, Shijie Geng, Renrui Zhang, Peng Gao, Gerard de Melo, Xiaogang Wang, Jifeng Dai, Yu Qiao, and Hongsheng Li. Frozen CLIP models are efficient video learners. In European Conference on Computer Vision, 2022.

[56] Wenhao Wu, Xiaohan Wang, Haipeng Luo, Jingdong Wang, Yi Yang, and Wanli Ouyang. Bidirectional cross-modal knowledge exploration for video recognition with pre-trained vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[57] Zhiwu Qing, Shiwei Zhang, Ziyuan Huang, Yingya Zhang, Changxin Gao, Deli Zhao, and Nong Sang. Disentangling spatial and temporal learning for efficient image-to-video transfer learning. In IEEE/CVF International Conference on Computer Vision, 2023.

[58] Kumara Kahatapitiya, Anurag Arnab, Arsha Nagrani, and Michael S. Ryoo. VicTR: Video-conditioned text representations for activity recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[59] Yifei Chen, Dapeng Chen, Ruijin Liu, Hao Li, and Wei Peng. Video action recognition with attentive semantic units. In IEEE/CVF International Conference on Computer Vision, 2023.

[60] Xiaohu Huang, Hao Zhou, Kun Yao, and Kai Han. FROSTER: frozen CLIP is A strong teacher for open-vocabulary action recognition. In International Conference on Learning Representations, 2024.

[61] Ruyang Liu, Jingjia Huang, Ge Li, Jiashi Feng, Xinglong Wu, and Thomas H. Li. Revisiting temporal modeling for CLIP-based image-tovideo knowledge transferring. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[62] Qiang Wang, Junlong Du, Ke Yan, and Shouhong Ding. Seeing in Flowing: Adapting CLIP for action recognition with motion prompts learning. In ACM International Conference on Multimedia, 2023.

[63] Mengmeng Wang, Jiazheng Xing, Boyuan Jiang, Jun Chen, Jianbiao Mei, Xingxing Zuo, Guang Dai, Jingdong Wang, and Yong Liu. A multimodal, multi-task adapting framework for video action recognition. In AAAI Conference on Artificial Intelligence, 2024.

[64] Yifei Chen, Dapeng Chen, Ruijin Liu, Sai Zhou, Wenyuan Xue, and Wei Peng. Align before adapt: Leveraging entity-to-region alignments for generalizable video action recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[65] Wei Zhang, Chaoqun Wan, Tongliang Liu, Xinmei Tian, Xu Shen, and Jieping Ye. Enhanced motion-text alignment for image-to-video transfer learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[66] Minji Kim, Dongyoon Han, Taekyung Kim, and Bohyung Han. Leveraging temporal contextualization for video action recognition. In European Conference on Computer Vision, 2024.

[67] Jingen Liu, Benjamin Kuipers, and Silvio Savarese. Recognizing human actions by attributes. In IEEE Conference on Computer Vision and Pattern Recognition, 2011.

[68] Elyor Kodirov, Tao Xiang, Zhen-Yong Fu, and Shaogang Gong. Unsupervised domain adaptation for zero-shot learning. In IEEE International Conference on Computer Vision, 2015.

[69] Mihir Jain, Jan C. van Gemert, Thomas Mensink, and Cees G. M. Snoek. Objects2action: Classifying and localizing actions without any video example. In IEEE International Conference on Computer Vision, 2015.

[70] Chuang Gan, Ming C. Lin, Yi Yang, Gerard de Melo, and Alexander G. Hauptmann. Concepts not alone: Exploring pairwise relationships for zero-shot video activity recognition. In AAAI Conference on Artificial Intelligence, 2016.

[71] Yanwei Fu, Timothy M. Hospedales, Tao Xiang, Zhen-Yong Fu, and Shaogang Gong. Transductive multi-view embedding for zero-shot recognition and annotation. In European Conference on Computer Vision, 2014.

[72] Qian Wang and Ke Chen. Zero-shot visual recognition via bidirectional latent embedding. International Journal of Computer Vision, 124(3):356–383, 2017.

[73] Shizhe Chen and Dong Huang. Elaborative rehearsal for zero-shot action recognition. In IEEE/CVF International Conference on Computer Vision, 2021.

[74] Jiaming Zhou, Junwei Liang, Kun-Yu Lin, Jinrui Yang, and Wei-Shi Zheng. ActionHub: A large-scale action video description dataset for zero-shot action recognition. CoRR, abs/2401.11654, 2024.

[75] Tiantian Xu, Fan Zhu, Edward K. Wong, and Yi Fang. Dual manyto-one-encoder-based transfer learning for cross-dataset human action recognition. Image and Vision Computing, 55:127–137, 2016.

[76] Arshad Jamal, Vinay P. Namboodiri, Dipti Deodhare, and K. S. Venkatesh. Deep domain adaptation in action space. In British Machine Vision Conference, 2018.

[77] Yuecong Xu, Jianfei Yang, Haozhi Cao, Zhenghua Chen, Qi Li, and Kezhi Mao. Partial video domain adaptation with partial adversarial temporal attentive network. In IEEE/CVF International Conference on Computer Vision, 2021.

[78] Pau Panareda Busto, Ahsan Iqbal, and Juergen Gall. Open set domain adaptation for image and action recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(2):413–429, 2020.

[79] Yuecong Xu, Jianfei Yang, Haozhi Cao, Keyu Wu, Min Wu, and Zhenghua Chen. Source-free video domain adaptation by learning temporal consistency for action recognition. In European Conference on Computer Vision, 2022.

[80] Kun-Yu Lin, Jiaming Zhou, and Wei-Shi Zheng. Human-centric transformer for domain adaptive action recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(2):679–696, 2025.

[81] Boxiao Pan, Zhangjie Cao, Ehsan Adeli, and Juan Carlos Niebles. Adversarial cross-domain action recognition with co-attention. In AAAI Conference on Artificial Intelligence, 2020.

[82] Yuecong Xu, Haozhi Cao, Kezhi Mao, Zhenghua Chen, Lihua Xie, and Jianfei Yang. Aligning correlation information for domain adaptation in action recognition. IEEE Transactions on Neural Networks and Learning Systems, 2023.

[83] Yadan Luo, Zi Huang, Zijian Wang, Zheng Zhang, and Mahsa Baktashmotlagh. Adversarial bipartite graph learning for video domain adaptation. In ACM International Conference on Multimedia, 2020.

[84] Aadarsh Sahoo, Rutav Shah, Rameswar Panda, Kate Saenko, and Abir Das. Contrast and mix: Temporal contrastive video domain

adaptation with background mixing. In Advances in Neural Information Processing Systems, 2021.

[85] Jonathan Munro and Dima Damen. Multi-modal domain adaptation for fine-grained action recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

[86] Xiaolin Song, Sicheng Zhao, Jingyu Yang, Huanjing Yue, Pengfei Xu, Runbo Hu, and Hua Chai. Spatio-temporal contrastive domain adaptation for action recognition. In IEEE Conference on Computer Vision and Pattern Recognition, 2021.

[87] Donghyun Kim, Yi-Hsuan Tsai, Bingbing Zhuang, Xiang Yu, Stan Sclaroff, Kate Saenko, and Manmohan Chandraker. Learning crossmodal contrastive features for video domain adaptation. In IEEE/CVF International Conference on Computer Vision, 2021.

[88] Lijin Yang, Yifei Huang, Yusuke Sugano, and Yoichi Sato. Interact before align: Leveraging cross-modal knowledge for domain adaptive action recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[89] Yunhua Zhang, Hazel Doughty, Ling Shao, and Cees G. M. Snoek. Audio-adaptive activity recognition across video domains. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[90] Mirco Planamente, Chiara Plizzari, Emanuele Alberti, and Barbara Caputo. Domain generalization through audio-visual relative norm alignment in first person action recognition. In IEEE/CVF Winter Conference on Applications of Computer Vision, 2022.

[91] Kun-Yu Lin, Jia-Run Du, Yipeng Gao, Jiaming Zhou, and Wei-Shi Zheng. Diversifying spatial-temporal perception for video domain generalization. In Advances in Neural Information Processing Systems, 2023.

[92] Chiara Plizzari, Toby Perrett, Barbara Caputo, and Dima Damen. What can a cook in italy teach a mechanic in india? Action recognition generalisation over scenarios and locations. In IEEE/CVF International Conference on Computer Vision, 2023.

[93] Junnan Li, Ramprasaath R. Selvaraju, Akhilesh Gotmare, Shafiq R. Joty, Caiming Xiong, and Steven Chu-Hong Hoi. Align before Fuse: Vision and language representation learning with momentum distillation. In Advances in Neural Information Processing Systems, 2021.

[94] Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[95] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. Flamingo: A visual language model for few-shot learning. In Advances in Neural Information Processing Systems, 2022.

[96] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, Ce Liu, Mengchen Liu, Zicheng Liu, Yumao Lu, Yu Shi, Lijuan Wang, Jianfeng Wang, Bin Xiao, Zhen Xiao, Jianwei Yang, Michael Zeng, Luowei Zhou, and Pengchuan Zhang. Florence: A new foundation model for computer vision. CoRR, abs/2111.11432, 2021.

[97] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. CoCa: Contrastive captioners are image-text foundation models. Transactions on Machine Learning Research, 2022.

[98] Yifan Du, Zikang Liu, Junyi Li, and Wayne Xin Zhao. A survey of vision-language pre-trained models. In International Joint Conference on Artificial Intelligence, 2022.

[99] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In IEEE/CVF International Conference on Computer Vision, 2023.

[100] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

[101] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloé Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross B. Girshick. Segment

anything. In IEEE/CVF International Conference on Computer Vision, 2023.

[102] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9):2337–2348, 2022.

[103] Tao Yu, Zhihe Lu, Xin Jin, Zhibo Chen, and Xinchao Wang. Task residual for tuning vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[104] Junting Pan, Ziyi Lin, Xiatian Zhu, Jing Shao, and Hongsheng Li. St-adapter: Parameter-efficient image-to-video transfer learning. In Advances in Neural Information Processing Systems, 2022.

[105] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[106] Jianzong Wu, Xiangtai Li, Shilin Xu, Haobo Yuan, Henghui Ding, Yibo Yang, Xia Li, Jiangning Zhang, Yunhai Tong, Xudong Jiang, Bernard Ghanem, and Dacheng Tao. Towards open vocabulary learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(7):5092–5113, 2024.

[107] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Openvocabulary object detection via vision and language knowledge distillation. In International Conference on Learning Representations, 2022.

[108] Boyi Li, Kilian Q. Weinberger, Serge J. Belongie, Vladlen Koltun, and René Ranftl. Language-driven semantic segmentation. In International Conference on Learning Representations, 2022.

[109] Shenghao Fu, Qize Yang, Qijie Mo, Junkai Yan, Xihan Wei, Jingke Meng, Xiaohua Xie, and Wei-Shi Zheng. LLMDet: Learning strong open-vocabulary object detectors under the supervision of large language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[110] Syed Talal Wasim, Muzammal Naseer, Salman H. Khan, Fahad Shahbaz Khan, and Mubarak Shah. Vita-CLIP: Video and text adaptive CLIP via multimodal prompting. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[111] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. VideoCLIP: Contrastive pre-training for zero-shot video-text understanding. In Empirical Methods in Natural Language Processing, 2021.

[112] Dongxu Li, Junnan Li, Hongdong Li, Juan Carlos Niebles, and Steven C. H. Hoi. Align and Prompt: Video-and-language pre-training with entity prompts. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[113] Junke Wang, Dongdong Chen, Zuxuan Wu, Chong Luo, Luowei Zhou, Yucheng Zhao, Yujia Xie, Ce Liu, Yu-Gang Jiang, and Lu Yuan. OmniVL: One foundation model for image-language and video-language tasks. In Advances in Neural Information Processing Systems, 2022.

[114] Liliane Momeni, Mathilde Caron, Arsha Nagrani, Andrew Zisserman, and Cordelia Schmid. Verbs in action: Improving verb understanding in video-language models. In IEEE/CVF International Conference on Computer Vision, 2023.

[115] Jingjia Huang, Yinan Li, Jiashi Feng, Xinglong Wu, Xiaoshuai Sun, and Rongrong Ji. Clover: Towards a unified video-language alignment and fusion model. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[116] Feng Cheng, Xizi Wang, Jie Lei, David J. Crandall, Mohit Bansal, and Gedas Bertasius. VindLU: A recipe for effective video-and-language pretraining. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[117] Mamshad Nayeem Rizve, Fan Fei, Jayakrishnan Unnikrishnan, Son Tran, Benjamin Z. Yao, Belinda Zeng, Mubarak Shah, and Trishul Chilimbi. VidLA: Video-language alignment at scale. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[118] Fan Ma, Xiaojie Jin, Heng Wang, Jingjia Huang, Linchao Zhu, and Yi Yang. Stitching segments and sentences towards generalization in video-text pre-training. In AAAI Conference on Artificial Intelligence, 2024.

[119] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Ming-Hsuan Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL technical report. CoRR, abs/2502.13923, 2025.

[120] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, Tianxiang Jiang, Songze Li, Jilan Xu, Hongjie Zhang, Yifei Huang, Yu Qiao, Yali Wang, and Limin Wang. InternVideo2: Scaling foundation models for

multimodal video understanding. In European Conference on Computer Vision, 2024.

[121] Neelu Madan, Andreas Møgelmose, Rajat Modi, Yogesh S. Rawat, and Thomas B. Moeslund. Foundation models for video understanding: A survey. CoRR, abs/2405.03770, 2024.

[122] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross B. Girshick. Momentum contrast for unsupervised visual representation learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

[123] OpenAI. GPT-4 technical report, 2023.

[124] Harold W. Kuhn. The hungarian method for the assignment problem. In Michael Jünger, Thomas M. Liebling, Denis Naddef, George L. Nemhauser, William R. Pulleyblank, Gerhard Reinelt, Giovanni Rinaldi, and Laurence A. Wolsey, editors, 50 Years of Integer Programming 1958-2008 - From the Early Years to the State-of-the-Art, pages 29–47. Springer, 2010.

[125] Yuecong Xu, Jianfei Yang, Haozhi Cao, Keyu Wu, Min Wu, Zhengguo Li, and Zhenghua Chen. Multi-source video domain adaptation with temporal attentive moment alignment network. IEEE Transactions on Circuits and Systems for Video Technology, 33(8):3860–3871, 2023.

[126] Mingsheng Long, Yue Cao, Jianmin Wang, and Michael I. Jordan. Learning transferable features with deep adaptation networks. In International Conference on Machine Learning, 2015.

[127] Baochen Sun and Kate Saenko. Deep CORAL: correlation alignment for deep domain adaptation. In European Conference on Computer Vision Workshops, 2016.

[128] Biagio Brattoli, Joseph Tighe, Fedor Zhdanov, Pietro Perona, and Krzysztof Chalupka. Rethinking zero-shot video classification: Endto-end training for realistic applications. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

[129] Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry P. Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. In Uncertainty in Artificial Intelligence, 2018.

[130] Taojiannan Yang, Yi Zhu, Yusheng Xie, Aston Zhang, Chen Chen, and Mu Li. AIM: adapting image models for efficient video action recognition. In International Conference on Learning Representations, 2023.

[131] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-SNE. Journal of Machine Learning Research, 9(11), 2008.

[132] DeepSeek-AI, Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenhao Xu, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Erhang Li, Fangqi Zhou, Fangyun Lin, Fucong Dai, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Li, Haofen Liang, Haoran Wei, Haowei Zhang, Haowen Luo, Haozhe Ji, Honghui Ding, Hongxuan Tang, Huanqi Cao, Huazuo Gao, Hui Qu, Hui Zeng, Jialiang Huang, Jiashi Li, Jiaxin Xu, Jiewen Hu, Jingchang Chen, Jingting Xiang, Jingyang Yuan, Jingyuan Cheng, Jinhua Zhu, Jun Ran, Junguang Jiang, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Kexin Huang, Kexing Zhou, Kezhao Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Wang, Liang Zhao, Liangsheng Yin, Lihua Guo, Lingxiao Luo, Linwang Ma, Litong Wang, Liyue Zhang, M. S. Di, M. Y Xu, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Panpan Huang, Peixin Cong, Peiyi Wang, Qiancheng Wang, Qihao Zhu, Qingyang Li, Qinyu Chen, Qiushi Du, Ruiling Xu, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runqiu Yin, Runxin Xu, Ruomeng Shen, Ruoyu Zhang, S. H. Liu, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaofei Cai, Shaoyuan Chen, Shengding Hu, Shengyu Liu, Shiqiang Hu, Shirong Ma, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, Songyang Zhou, Tao Ni, Tao Yun, Tian Pei, Tian Ye, Tianyuan Yue, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjie Pang, Wenjing Luo, Wenjun Gao, Wentao Zhang, Xi Gao, Xiangwen Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaokang Zhang, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xingyou Li, Xinyu Yang, Xinyuan Li, Xu Chen, Xuecheng Su, Xuehai Pan, Xuheng Lin, Xuwei Fu, Y. Q. Wang, Yang Zhang, Yanhong Xu, Yanru Ma, Yao Li, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Qian, Yi Yu, Yichao Zhang, Yifan Ding, Yifan Shi, Yiliang Xiong, Ying He, Ying Zhou, Yinmin Zhong, Yishi Piao, Yisong Wang, Yixiao Chen, Yixuan Tan, Yixuan Wei, Yiyang Ma, Yiyuan Liu, Yonglun Yang, Yongqiang Guo, Yongtong Wu, Yu Wu, Yuan Cheng, Yuan Ou, Yuanfan Xu, Yuduan Wang, Yue Gong, Yuhan Wu, Yuheng Zou, Yukun Li, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren, Zehua Zhao, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng

Ma, Zhigang Yan, Zhihong Shao, Zhixian Huang, Zhiyu Wu, Zhuoshu Li, Zhuping Zhang, Zian Xu, Zihao Wang, Zihui Gu, Zijia Zhu, Zilin Li, Zipeng Zhang, Ziwei Xie, Ziyi Gao, Zizheng Pan, Zongqing Yao, Bei Feng, Hui Li, J. L. Cai, Jiaqi Ni, Lei Xu, Meng Li, Ning Tian, R. J. Chen, R. L. Jin, S. S. Li, Shuang Zhou, Tianyu Sun, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xinnan Song, Xinyi Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, Dongjie Ji, Jian Liang, Jianzhong Guo, Jin Chen, Leyi Xia, Miaojun Wang, Mingming Li, Peng Zhang, Ruyi Chen, Shangmian Sun, Shaoqing Wu, Shengfeng Ye, T. Wang, W. L. Xiao, Wei An, Xianzu Wang, Xiaowen Sun, Xiaoxiang Wang, Ying Tang, Yukun Zha, Zekai Zhang, Zhe Ju, Zhen Zhang, and Zihua Qu. Deepseek-v3.2: Pushing the frontier of open large language models. CoRR, abs/2512.02556, 2025.

[133] Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean-Bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Gaël Liu, Francesco Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin, Róbert Busa-Fekete, Alex Feng, Noveen Sachdeva, Benjamin Coleman, Yi Gao, Basil Mustafa Iain Barr, Emilio Parisotto, David Tian, Matan Eyal, Colin Cherry, Jan-Thorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran Kazemi, Dan Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming Luo, Andreas Steiner, Abe Friesen, Abhanshu Sharma, Abheesht Sharma, Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa Saade, Alexander Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit Vadi, András György, André Susano Pinto, Anil Das Ankur Bapna, Antoine Miech, Antoine Yang, Antonia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot, Bo Wu, Bobak Shahriari, Bryce Petrini, Charlie Chen, Charline Le Lan, Christopher A. Choquette Choo, CJ Carey, Cormac Brick, Daniel Deutsch, Danielle Eisenbud, Dee Cattle, Derek Cheng, Dimitris Paparas, Divyashree Shivakumar Sreepathihalli, Doug Reid, Dustin Tran, Dustin Zelle, Eric Noland, Er win Huizenga, Eugene Kharitonov, Frederick Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna Klimczak-Plucinska, Harman Singh, Harsh Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne, Idan Szpektor, Ivan Nardini, Jean Pouget-Abadie, Jetha Chan, Joe Stanton, John Wieting, Jonathan Lai, Jordi Orbay, Joseph Fernandez, Josh Newlan, Ju-yeong Ji, Jyotinder Singh, Kat Black, Kathy Yu, Kevin Hui, Kiran Vodrahalli, Klaus Greff, Linhai Qiu Marcella Valentine, Marina Coelho, Marvin Ritter, Matt Hoffman, Matthew Watson, Mayank Chaturvedi, Michael Moynihan, Min Ma, Nabila Babar, Natasha Noy, Nathan Byrd, Nick Roy, Nikola Mom chev, Nilay Chauhan, Oskar Bunyan, Pankil Botarda, Paul Caron, Paul Kishan Rubenstein, Phil Culliton, Philipp Schmid, Pier Giuseppe Sessa, Pingmei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shivanna, Renjie Wu, Renke Pan, Reza Rokni, Rob Willoughby, Rohith Vallu, Ryan Mullins, Sammy Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal, Shashir Reddy, Shruti Sheth, Siim Põder, Sijal Bhatnagar, Sindhu Raghuram Panyam, Sivan Eiger, Susan Zhang, Tianqi Liu, Trevor Yacovone, Tyler Liechty, Uday Kalra, Utku Evci, Vedant Misra, Vincent Roseberry, Vlad Feinberg, Vlad Kolesnikov, Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein Zhu, Zichuan Wei, Zoltan Egyed, Victor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao, Jessica Lo, Erica Moreira, Luiz Gustavo Martins, Omar Sanseviero, Lucas Gonzalez, Zach Gleicher, Tris Warkentin, Vahab Mirrokni, Evan Senter, Eli Collins, Joelle K. Barral, Zoubin Ghahramani, Raia Hadsell, Yossi Matias, D. Sculley, Slav Petrov, Noah Fiedel, Noam Shazeer Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clé- ment Farabet, Elena Buchatskaya, Jean-Baptiste Alayrac, Rohan Anil, Dmitry (Dima) Lepikhin, Sebastian Borgeaud, Olivier Bachem, Armand Joulin, Alek Andreev, Cassidy Hardin, Robert Dadashi, and Léonard Hussenot. Gemma 3 technical report. CoRR, abs/2503.19786, 2025.

[134] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. CoRR, abs/2412.15115, 2024.

[135] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran

Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jian Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. CoRR, abs/2505.09388, 2025.

[136] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Scaling egocentric vision: The EPIC-KITCHENS dataset. In European Conference on Computer Vision, 2018.