# FINE-GRAINED CLIP FINE-TUNING WITH SELF-ANNOTATED REGION ALIGNMENT

Chenyang Zhao<sup>1,2</sup>, Wei Lin<sup>1</sup>, Antoni B. Chan<sup>1</sup>, and Janet H. Hsiao<sup>2</sup>

<sup>1</sup>Department of Computer Science, City University of Hong Kong <sup>2</sup>Division of Social Science, Hong Kong University of Science & Technology

## ABSTRACT

Contrastive Language-Image Pre-training (CLIP) has been shown to have limitations in its finegrained dense feature representation, due to its pre-training focusing on matching the whole image to a text description. Considering the large data and computational burden in pre-training a visionlanguage model from scratch, a series of works aim to enhance the fine-grained ability of CLIP through a fine-tuning scheme. However, existing works suffer from a variety of limitations: additional region annotations are usually required, which limits the semantic diversity due to the predefined categories and leads to a large effort to process the training data; and they usually sacrifice CLIP’s original ability for global visual representation. To bypass these limitations, we propose SFF-CLIP (Self-annotated Fine-grained Fine-tuning for CLIP), which only uses image-text pairs as input to boost the fine-grained representation ability in the CLIP fine-tuning, while maintaining the global visual-semantic consistency. Concretely, a run-time region-phrase alignment scheme is designed, which obtains concept phrases from the input sentence, and aligns them with corresponding extracted region-based features using text-specific heat maps. Extensive experiments demonstrate that SFF-CLIP leads to significant performance improvements on fine-grained dense feature representation, as well as maintaining the performance of the original CLIP on image-level tasks. Code will be released later.

Keywords CLIP · self-annotation · Fine-grained alignment · Fine-tuning

## 1 Introduction

Contrastive Language-Image Pre-training (CLIP) [1] aligns global image and text embeddings within a unified latent space by adopting a dual-encoder architecture and conducting matching on large-scale, noisy image-text pairs. It has become a foundational vision-language model (VLM) for representation learning, and achieves remarkable success on image-level tasks [2, 3, 4], such as image classification and image-text retrieval.

However, CLIP exhibits notable limitations in comprehending fine-grained details, such as poor region recognition when using its dense features, due to the pre-training focusing on matching the whole image (via the [cls] token) to a text description. Thus, the model struggles to extract meaningful region-level representations from its dense visual features for grounding textual concepts. This shortcoming limits the performance of CLIP on the downstream tasks that require region-aware ability. For example, in dense prediction tasks, e.g., object detection and segmentation, the CLIP model is usually utilized as a classifier [5, 6] or the teacher in distillation [7, 8] to process cropped object patches to obtain region features. Some works [9, 10, 11] adopt the frozen CLIP model as the backbone to produce spatial feature maps, but they all choose the CNN-based CLIP, which can preserve more position information than the vision transformer (ViT-based) architecture.

Since the resource demand for pretraining a fine-grained VLM is staggering, such as billions of data samples [12, 13, 14], a series of recent works have focused on advancing the vision-language alignment of local image regions via fine-tuning of CLIP. Considering the impressive generalization capabilities of CLIP models, particularly the powerful ViT-based variants, the fine-grained fine-tuning of CLIP is carried out to enable applications on dense prediction tasks, such as open-vocabulary object detection and image segmentation. Since no region-text annotations are provided in the image-text pair training data, the previous fine-tuning works enhance the fine-grained alignment via: 1) region-based pseudo-labeling, 2) region-based distillation, 3) mosaicking images as pseudo regions, or the combination thereof. In the first category, [15, 16, 17] generate region proposals by coarsely cropping patches or adopting RPN [18], and then use CLIP or other powerful VLMs to retrieve region labels from a pool of concepts (e.g., Fig. 1a). However, the range of concepts is limited by the number of pre-defined categories, which could hinder downstream applications like open-vocabulary object detection. The recent work FineCLIP [19] (Fig. 1c) adopts large VLMs like BLIP2 [20] to generate detailed sentences to describe region proposals. However, carefully preparing the region annotations in these methods inevitably costs significant extra time and space, which cannot be neglected when increasing the scale of training data. For the 2nd category, CLIPSelf [21] (Fig. 1b) and FineCLIP [19] (Fig. 1c) use distillation to transfer the global features of the cropped regions (either random or proposals) to extracted dense features – however, to obtain superior distillation performance, preprocessing of the generated region proposals by a well-trained detector is still required. Mosaic augmentation is adopted to make pseudo regions in [22, 23] (e.g., Fig. 1d), which produces finer matching than image-level while still coarser than region alignment. Moreover, most of these methods sacrifice, and some even ignore [21, 23, 22, 17], the global representation ability of CLIP when training focuses on region knowledge, which causes severe performance damage on image-level tasks.

![](images/de0e1e1d2532685502593c72c1384354c3a685d26e75f4b4c8d363736399c987.jpg)  
Figure 1: Comparison of different fine-grained alignment methods for CLIP. The required preparations for region annotation are marked with bold red texts. Our method eliminates the constraints of pre-defined category limitation, and the large effort required for generating region proposals and their corresponding descriptions.

In this work, we propose SFF-CLIP, a novel self-annotated fine-grained fine-tuning method for CLIP, which aims to boost the region awareness of CLIP while maintaining the global visual-semantic consistency, without requiring extra preparation of region proposals or their annotations. As with the original CLIP pre-training, image-text pairs are taken as inputs for fine-tuning. We propose a self-annotated fine-grained alignment scheme, where we first extract phrases from the input sentence and generate the corresponding region importance on the image based on text-specific heat maps. The heat maps are used to aggregate region features for each phrase, and then the region-phrase feature pairs are aligned by the matching loss, whose weight is based on the degree of phrase matching. To help maintain the whole image representation ability of the original CLIP, the contrastive learning objective for image-text pairs is included in the loss function through weighted momentum term. As a result, our SFF-CLIP locally and globally aligns the visual and semantic features in the same representation space. Through comprehensive experiments, we show that SFF-CLIP significantly improves the fine-grained understanding of CLIP, surpasses previous state-of-the-arts on both the dense prediction benchmarks, including open-vocabulary object detection and segmentation, while also maintaining the global representation of the original model in the image-level task.

## We summarize our contributions as follows:

1. We propose SFF-CLIP, a self-annotated fine-grained alignment strategy for finetuning CLIP, which uses a run-time scheme that generates region knowledge from input image-text pairs to achieve better fine-grained understanding.

2. SFF-CLIP eliminates the need for extra region-based training data, such as pre-defined categories, or generating region proposals and their corresponding descriptions.

3. SFF-CLIP successfully maintains the global representation capability of CLIP while improving the region awareness.

4. Extensive experiments on dense prediction and image-level tasks show that the proposed SFF-CLIP consistently outperforms previous methods.

## 2 Related Work

## 2.1 Fine-grained understanding in VLMs

Although CLIP [1] and subsequent VLMs [24, 25, 26] exhibit strong representation capabilities and exceptional generalizability, the image-level training, which matches an image as a whole to a text description, has been shown to be deficient in fine-grained understanding and alignment between image regions and text [16, 15, 27, 21]. This shortcoming limits their applications on tasks that require region-aware abilities, e.g., dense prediction tasks.

To mitigate this issue, some works build strong fine-grained VLMs by pre-training with patch-token embedding alignment [14], or region-caption matching [12] utilizing well-trained open-vocabulary detectors for assistance. FG-CLIP [13] constructs a comprehensive dataset with billions of region-specific annotations for region-aware training for CLIP. Due to the heavy cost of using large-scale data, some recent works propose to enhance the fine-grained representation by fine-tuning CLIP with region-based pseudo-labeling or region-based distillation. RegionCLIP [16] adopts RPN [18] object proposals, while PTP [15] coarsely crops patches, and then they both use CLIP as a classifier to obtain region labels from a large pre-defined pool of concepts. DenseVLM [17] uses a powerful VLM to retrieve categories (from a category set) for randomly cropped regions. The recent work FineCLIP [19] adopts large VLMs like BLIP2 [20] to generate detailed sentences to describe region proposals. However, the semantic diversity is limited for those works with pre-defined categories or concepts, and these methods inevitably cost significant extra time and space to preprocess the region annotations, which cannot be neglected when increasing the training data. CLIPSelf [21] and ATAS [22] facilitate the transfer of the global features of regions to dense feature extraction by self-distillation – CLIPSelf obtains the region proposals through the object detector, and ATAS uses mosaic augmentation to construct pseudo regions as in CLIM [23]. The distillation-based methods only focus on the image encoder training and ignore the text encoder. Furthermore, among these methods, [17, 21, 22, 23] are also ignore the global image-text matching ability of CLIP.

Our paper proposes a self-annotated fine-grained fine-tuning framework for boosting the region-aware ability of CLIP that only uses image-text pairs as input and does not use external models. Our approach circumvents the resourceconsuming preparation of the region annotations and the limitation of pre-defined visual categories, while successfully preserving the global representation ability of CLIP.

## 2.2 Open-vocabulary dense prediction

Open-vocabulary dense prediction aims to identify visual regions of arbitrary categories as described by the text, and primarily comprises open-vocabulary object detection [7, 28, 29] and image segmentation [30, 31, 6]. Recent open-vocabulary approaches have leveraged the strong representation and generalizability displayed by powerful pre-trained VLMs, which are exploited to identify novel objects. Due to CLIP lacking precise local vision-language alignment, CLIP is usually adopted as a classifier [5, 6] or the teacher in distillation [7, 8] to process cropped region proposals. Several works [9, 10, 11, 28, 30] adopt frozen CLIP encoders as backbone in the detectors to generate visual features, but most of them select the CNN-based CLIP, which can preserve more position information than the vision transformer (ViT-based) architecture. Although recent studies [16, 19, 21, 17, 23] fine-tuning CLIP for fine-grained understanding, they are constrained by the limitations when generating dense annotations as mentioned above.

## 2.3 Visual explanation of VLMs

Visual explanation methods [32, 33, 34, 35, 36] in computer vision have been developed for visually interpreting a specific prediction of the model by generating a heat map that indicates the spatial feature importance. Recently, works [37, 38, 39, 40] have proposed visual explanations for the Transformer architecture. In our designed run-time self-annotation scheme for region-phrase pairs, we produce text-specific heat maps using Grad-ECLIP [40] as reference, which is a state-of-the-art visual explanation method for VLMs, such as CLIP. The heat map generation process is high-speed and can be easily plug-in the training of proposed fine-grained alignment.

## 3 Method

The aim of SFF-CLIP is to realize region-language alignment while maintaining image-level representation in the fine-tuning of CLIP, but without the requirements of generating dense data annotations, including pre-defined categories, region proposals, or region descriptions. To achieve this, the vision-language alignment at region and image levels are unified in our training method, employing our proposed run-time self-annotated fine-grained alignment and CLIP’s original global contrastive learning. The framework of our proposed SFF-CLIP is shown in Fig. 2, where the model has the same architecture as the original CLIP. We first present brief preliminaries about CLIP (§3.1), then introduce fine-grained alignment (§3.2) and global alignment (§3.3) in detail.

![](images/81aa6c666a74cf6961f06016d51bd086c43d1ae7ab8f5c676e6747da0fe69abb.jpg)  
Figure 2: Overview of the proposed SFF-CLIP. Multiple phrases or words representing objects (e.g., “dog”, “black car” and “traffic lights”) are separated out by parsing the input caption. By generating text-specific activation maps, object-specific region feature embeddings $( \hat { F } _ { r } )$ are obtained through weighted aggregation of the image dense feature $( F _ { d } )$ . Then the image region features $( \hat { F _ { r } } )$ and the corresponding phrase features $( { \bar { F } } _ { p } )$ are aligned in the training scheme, together with the global contrastive learning.

## 3.1 Preliminaries

CLIP learns visual and language representations from large-scale web-curated image-text pairs. It consists of an image encoder $\mathcal { T } \left( \cdot \right)$ and a text encoder $\dot { \tau } ( \cdot )$ , which are jointly trained to extract image and text feature embeddings in a unified representation space. Given image-text pair $( I , \dot { T } )$ , the matching score between their extracted image features $F _ { I } \in \mathbb { R } ^ { D }$ and text features $F _ { T } \in \mathbb { R } ^ { D }$ (both row vectors) is:

$$
S (F _ {I}, F _ {T}) = \cos (F _ {I}, F _ {T}) = \frac {F _ {I} F _ {T} ^ {\intercal}}{\| F _ {I} \| \| F _ {T} \|}.\tag{1}
$$

The model is trained using contrastive learning on the matching scores, regarding the ground-truth image-text pairs as positive samples and other mismatched pairs as negatives. For ViT-based encoders, the image feature is $\bar { F _ { I } } = \mathcal { L } \bar { \mathcal { P } } ( x _ { c l s } )$ where $\mathcal { L } \mathcal { P }$ denotes linear projections, and $x _ { c l s }$ is the feature vector from the [cls] token. Thus, except for the class token, all the final layer features of the other tokens (image patch tokens) are not used during contrastive learning of CLIP. Since only the class token feature is explicitly optimized during training, the local patch features exhibit weak representation ability for localized matching of semantic content.

## 3.2 Self-annotated fine-grained alignment

To realize fine-grained alignment, we propose a matching loss based on extracting the dense feature map from the image and performing fine-grained feature matching of regions to the corresponding text phrase. We dynamically generate self-annotated region-phrase pairs from the input image-text pairs during the training process, which we denote as run-time self-annotation, instead of off-line preparation with detectors or pre-defined categories. Note that our run-time self-annotation is based on the concepts learned by the original CLIP, which makes it more flexible than using pre-defined detectors, since it is not limited to a pre-defined pool of visual categories.

## 3.2.1 Image dense feature

Following [41, 21] we extract a dense feature map of the input image from a ViT-based encoder by slightly modifying the last transformer layer to keep the projection and norm layers, and discard the self-attention. This modification is experimentally shown to be capable of preserving more spatial detailed features in the output token embeddings [41]. Specifically, in the last transformer layer, with the input $\boldsymbol { x } = ( x _ { c l s } , x _ { 1 } , . . . , x _ { h \times w } )$ comprising a $[ c l s ]$ embedding and $h \times w$ spatial token embeddings, the layer’s output is obtained as $v _ { i } = \mathcal { L P } ( x _ { i } )$ , where $v _ { i } \in \mathbb { R } ^ { C }$ represent the value embeddings at spatial location i, with $\dot { C }$ as their channel dimension. Then, the $[ c l s ]$ embedding is removed and the final spatial token embeddings are reshaped into an $h \times w$ dense feature map $F _ { d } ,$ , from which we extract fine-grained representations for specific image regions.

## 3.2.2 Run-time region-phrase self-annotation

We propose a run-time scheme to automatically obtain region-phrase pairs from the input image-text pairs. Note that our method does not require any manually annotated or model-generated region proposals or their corresponding label annotations. It is also completely encapsulated – it does not require any external vision models (e.g., other VLMs).

Phrase extraction. For the caption $T$ in each image-text pair $( I , T )$ , we use the Natural Language Toolkit (NLTK) [42] to parse and extract the phrases that contain object concepts, by setting the separation and selection rules as “adjective + noun”. For the example in Fig. 2, the parsing results $\mathrm { a r e } \ { \stackrel {  } { \mathrm { d o g } } } ^ { , }$ , “black $\mathrm { c a r } ^ { \prime \ }$ , and “traffic lights” with the input text “a dog in a black car waiting for traffic lights”. Then, for these extracted words or phrases $\{ p _ { t } \} _ { t = 1 } ^ { n }$ , where n is the number of extracted phrases, textual descriptions are generated using prompt-template strategy [1]: “This is a photo of the $p _ { t }$ , These descriptions are sent to the text encoder, resulting in a set of phrase embeddings $\{ F _ { p _ { t } } \} _ { t }$

Text-specific heat map. The region embedding $F _ { r _ { i } }$ corresponding to each phrase embedding $F _ { p _ { t } }$ is calculated using a text-specific heat maps based on explainable AI (XAI) methods. These heat maps have been proposed to visually explain a specific prediction of the model by showing the spatial feature importance with a heat map. The local regions with relatively high values are interpreted as being important for generating the current output [33, 40, 36]. Referring to Grad-ECLIP [40], a high-speed and easy plug-in gradient-based heat map method designed for ViT-based VLMs, we calculate the spatial importance heat map corresponding to each extracted phrase. Specifically, the phrase embeddings $F _ { p _ { t } }$ are used to calculate the cosine similarities with the image embedding $F _ { I }$ , and each similarity $\cos ( F _ { I } , F _ { p _ { t } } )$ is regarded as a target to be interpreted. Following [40], the gradient $( g _ { c } )$ for each similarity w.r.t. the output of the last layer is used as channel weights to aggregate the feature maps v, and a loosened attention map technique is adopted to calculate the spatial weight $u _ { i } = \Phi ( \cos ( q _ { c l s } , k _ { i } ) )$ ), where $\bar { \Phi ( \cdot ) }$ is min-max normalization. Finally, the heat map for a specific phrase $p _ { t }$ is:

$$
H _ {t i} = \sum_ {c} g _ {c} \cdot u _ {i} \cdot v _ {i},\tag{2}
$$

where $\sum { _ { c } }$ is the channel-sum operator. As shown in Fig. 2, the heat map $H _ { t }$ can localize the corresponding concept by revealing the important spatial locations for matching with the specific phrase. The heat maps are adopted as weights for aggregating the image dense feature $F _ { d }$ , resulting in the region embedding for the phrase: $\begin{array} { r } { \textstyle F _ { r _ { t } } = \sum _ { h w } H _ { t } \cdot F _ { d } } \end{array}$ where $\sum _ { h w }$ is the sum operator over spatial coordinates.

To investigate the influence of the XAI method used for fine-grained alignment, we conduct an ablation study with several other high-speed Transformer-applicable visual explanation approaches in the Supplemental, which demonstrates that the Grad-ECLIP is the most effective.

## 3.2.3 Fine-grained matching

Given the region-phrase embeddings, the positive pairs are made to be similar, while the negative pairs are separated, via the loss function:

$$
\begin{array}{r l} & L _ {f g} = - \sum_ {t} w _ {t} \Big [ (1 - S (F _ {r _ {t}}, F _ {p _ {t}})) ^ {2} \log S (F _ {r _ {t}}, F _ {p _ {t}}) \\ & + \sum_ {t ^ {\prime} \neq t} S (F _ {r _ {t}}, F _ {p _ {t ^ {\prime}}}) ^ {2} \log (1 - S (F _ {r _ {t}}, F _ {p _ {t ^ {\prime}}})) \Big ], \end{array}\tag{3}
$$

where S represents the cosine similarity function, and $t , t ^ { \prime }$ means the t-th and t<sup>′</sup>-th phrase in the same batch.

A phrase-matching weight $w _ { t }$ is applied on the loss term for each region-phrase pair in order to diminish the potential impact of misaligned pairs, e.g., for extracted nouns that have no visual content in the image. Specifically, we adopt the highest response value on the phrase’s heat map, $\mathrm { i . e . , } w _ { t } = \operatorname* { m a x } ( H _ { t } )$ , to evaluate the degree of matching between the phrase and the image. For example, in Figure 3, “dinner” and “Tuesday” are extracted as nouns in the caption, while they are not visually matched with any of the image contents, and this can be reflected by the values on the heat map. In practice, we set the maximum number of phrases in each sentence to $N = 5$ . When the actual number of concepts n is smaller than N, then blank phrases are added with weight $w _ { t } = 0$

## 3.3 Global contrastive learning with momentum

In standard CLIP training, the input batch of image-text pairs $\{ ( I _ { b } , T _ { b } ) \} _ { b = 1 } ^ { B }$ is passed to the image/text encoders and the model outputs the corresponding global image/text embeddings $\{ ( F _ { I _ { b } } , F _ { T _ { b } } ) \} _ { b = 1 } ^ { B }$ . The cosine similarity $S ( F _ { I } , F _ { T } )$ between image embedding $F _ { I }$ and text embedding $F _ { T }$ is calculated as in Eq. 1, then the constrastive loss is applied to learn the global representations by maximizing the cosine similarities of the corresponding image and text embeddings, while minimizing the cosine similarities of other non-paired ones from both image and text sides, which is defined as:

$$
L _ {b} ^ {I 2 T} = - \log \frac {\exp \big (S (F _ {I _ {b}} , F _ {T _ {b}}) / \tau \big)}{\sum_ {b ^ {\prime} = 1} ^ {B} \exp \big (S (F _ {I _ {b}} , F _ {T _ {b ^ {\prime}}}) / \tau \big)}, L _ {b} ^ {T 2 I} = - \log \frac {\exp \big (S (F _ {T _ {b}} , F _ {I _ {b}}) / \tau \big)}{\sum_ {b ^ {\prime} = 1} ^ {B} \exp \big (S (F _ {T _ {b}} , F _ {I _ {b ^ {\prime}}}) / \tau \big)},\tag{4}
$$

Beer and bananas are prepared for dinner on Tuesday.  
![](images/546e33b8d004781120572672c9e81b68b98a27ec85ff06aaae0890a653f1bfab.jpg)  
Figure 3: An example of phrase alignment weight w , which is defined as the maximum value on the corresponding text-specific heat map. In this image-text pair, “beer” and “banana” are better matched with the image regions, as indicated by their higher weights $w _ { t }$ than “dinner” and “Tuesday”.

where $\tau$ is the trainable temperature parameter.

To better maintain the image-level representation ability of the pre-trained CLIP while enhancing its fine-grained alignment, we adopt a momentum CLIP model $\hat { M } ,$ which is initialized with the original CLIP model and then updated by the current trained model M with a low momentum rate $\alpha = 0 . 0 0 5$ after each epoch, i.e., $\hat { M }  ( 1 - \alpha ) \hat { M } + \alpha M .$ The updated model M<sup>ˆ</sup> is used to extract image and text embeddings $\{ ( \hat { F } _ { I _ { b } } , \hat { F } _ { T _ { b } } ) \} _ { b = 1 } ^ { B }$ , and weights are computed for each image-text pair, which measures how well they match:

$$
w _ {I _ {b}} = \gamma \frac {\exp \big (S (\hat {F} _ {I _ {b}} , \hat {F} _ {T _ {b}}) / \tau \big)}{\sum_ {b ^ {\prime} = 1} ^ {B} \exp \big (S (\hat {F} _ {I _ {b}} , \hat {F} _ {T _ {b ^ {\prime}}}) / \tau \big)} + (1 - \gamma), w _ {T _ {b}} = \gamma \frac {\exp \big (S (\hat {F} _ {T _ {b}} , \hat {F} _ {I _ {b}}) / \tau \big)}{\sum_ {b ^ {\prime} = 1} ^ {B} \exp \big (S (\hat {F} _ {T _ {b}} , \hat {F} _ {I _ {b ^ {\prime}}}) / \tau \big)} + (1 - \gamma),\tag{5}
$$

where parameter $\gamma$ controls the influence of match, which is set to 0.4 by default in the experiments. The contrastive loss then applies the weights to the loss terms in (4) to focus on preserving the well-matched image-text pairs,

$$
L _ {\text { contrastive }} = \frac {1}{2 B} \sum_ {b = 1} ^ {B} \left(w _ {I _ {b}} L _ {b} ^ {I 2 T} + w _ {T _ {b}} L _ {b} ^ {T 2 I}\right).\tag{6}
$$

The final loss for training is obtained by by adding the fine-grained matching loss to the contrastive learning loss, ${ \cal L } = { \cal L } _ { f g } + { \cal L } _ { c o n t r a s t i v e }$

## 4 Experiments

We evaluate the proposed SFF-CLIP on both improving of fine-grained representation via zero-shot region classification, down-stream tasks of open-vocabulary detection and segmentation, and on maintaining of image-level representation via image-text retrieval task. We compare with the primary fine-grained fine-tuning methods, including RegionCLIP[16], CLIPSelf[21], FineCLIP[19], DenseVLM[17], and CLIM[17]. Moreover, we also compare with fine-tuning CLIP using only the global contrastive loss, i.e., without the fine-grained matching, which is a baseline for continuation of image-level training of CLIP (denoted as $^ { 6 6 } \mathrm { C L I P - g ^ { 3 7 } } ,$ ). Finally, we carry out the ablation studies.

## 4.1 Implementation details

For fairness, we re-trained all compared fine-grained fine-tuning methods, by initializing with the same pre-trained model and fine-tuning with the same training data, using the same input image size, which is the same as the CLIP pre-training, 224x224 for ViT-B/16 and 336x336 for ViT-L/14 by default. Following previous works [21, 19], the experiments are conducted based on the pre-trained models from EVA-CLIP [43]. Since some related works [16, 21, 19] that require region annotations open-sourced their region information files based on MS COCO train2017 set [44], we conduct the fine-tuning on the same images with the captions provided by [19]. We mainly report the comparison results with other works on ViT-B/16, with the comparisons on ViT-L/14 provided in the Supplemental. Two RTX 6000 Ada are used, and for our method, we use batch size 128, learning rate 1e-5, and weight decay of 0.1; for other methods, we use the training parameters provided in their codes.

## 4.2 Comparisons on fine-grained representation

Zero-shot region classification. To evaluate the dense representation ability, we use the mean accuracy (mAcc) of classifying region boxes and panoptic masks for “things” annotated in the ADE20K panoptic [45] val set and COCO panoptic [44] val2017 set. To extract region-level features, RoI or mask pooling are used to extract the region box or mask embeddings from the image dense feature maps. The classification is then performed by selecting the highest score when matching with the text embeddings of the classes.

Table 1: Comparison of fine-grained dense representations via zero-shot classification on the ADE20K panoptic [45] val set and COCO panoptic [44] val2017 set. We report the Top-1 and Top-5 mean accuracy on both object bounding boxes and panoptic masks. The gray row is the baseline CLIP before fine-grained alignment fine-tuning. Note that DenseVLM<sup>⋆</sup> represents the results reported by the original paper [17] for input 224x224, and <sup>†</sup> represents the results testing with open-sourced fine-grained models pre-trained by large-scale data, shown in the brackets. Our method does not require preparation of region proposals (R.P.), region labels (R.L.), or predefined categories (P.C.) on the training data.

<table><tr><td rowspan="3">Method</td><td rowspan="3">Model</td><td colspan="7">ADE20k</td><td colspan="4">MS COCO</td></tr><tr><td rowspan="2">R.P.</td><td rowspan="2">R.L.</td><td rowspan="2">P.C.</td><td colspan="2">Boxes</td><td colspan="2">Masks</td><td colspan="2">Boxes</td><td colspan="2">Masks</td></tr><tr><td>Top1</td><td>Top5</td><td>Top1</td><td>Top5</td><td>Top1</td><td>Top5</td><td>Top1</td><td>Top5</td></tr><tr><td>FG-CLIP $^{\dagger}$  (1.6B)</td><td>ViT-B/16</td><td>√</td><td>√</td><td>×</td><td>30.3</td><td>57.4</td><td>28.6</td><td>56.7</td><td>61.2</td><td>83.1</td><td>49.7</td><td>75.5</td></tr><tr><td>CLIP</td><td>ViT-B/16</td><td>×</td><td>×</td><td>×</td><td>18.6</td><td>40.6</td><td>25.5</td><td>43.1</td><td>41.4</td><td>63.6</td><td>30.6</td><td>53.8</td></tr><tr><td>CLIP-g</td><td>ViT-B/16</td><td>×</td><td>×</td><td>×</td><td>20.1</td><td>41.5</td><td>26.9</td><td>44.1</td><td>44.7</td><td>65.8</td><td>33.4</td><td>55.6</td></tr><tr><td>RegionCLIP</td><td>ViT-B/16</td><td>√</td><td>√</td><td>√</td><td>27.9</td><td>54.4</td><td>34.2</td><td>54.3</td><td>59.4</td><td>80.9</td><td>47.4</td><td>73.2</td></tr><tr><td>FineCLIP</td><td>ViT-B/16</td><td>√</td><td>√</td><td>×</td><td>27.0</td><td>53.2</td><td>33.1</td><td>53.0</td><td>57.7</td><td>80.3</td><td>48.0</td><td>73.2</td></tr><tr><td>DenseVLM</td><td>ViT-B/16</td><td>×</td><td>×</td><td>√</td><td>18.5</td><td>45.2</td><td>25.7</td><td>47.4</td><td>34.9</td><td>55.8</td><td>31.8</td><td>47.2</td></tr><tr><td>DenseVLM*</td><td>ViT-B/16</td><td>×</td><td>×</td><td>√</td><td>-</td><td>-</td><td>-</td><td>-</td><td>60.1</td><td>79.9</td><td>49.4</td><td>62.4</td></tr><tr><td>CLIPSelf</td><td>ViT-B/16</td><td>√</td><td>×</td><td>×</td><td>27.9</td><td>55.3</td><td>33.6</td><td>53.2</td><td>60.9</td><td>81.0</td><td>49.0</td><td>73.2</td></tr><tr><td>CLIM</td><td>ViT-B/16</td><td>×</td><td>×</td><td>×</td><td>25.7</td><td>49.6</td><td>30.9</td><td>49.8</td><td>53.4</td><td>73.9</td><td>45.3</td><td>58.4</td></tr><tr><td>SFF-CLIP(Ours)</td><td>ViT-B/16</td><td>×</td><td>×</td><td>×</td><td>29.4</td><td>55.4</td><td>34.3</td><td>54.5</td><td>62.2</td><td>80.8</td><td>52.1</td><td>73.6</td></tr><tr><td>CLIP</td><td>ViT-L/14</td><td>-</td><td>-</td><td>-</td><td>31.2</td><td>56.8</td><td>41.2</td><td>62.2</td><td>58.1</td><td>78.9</td><td>49.8</td><td>72.6</td></tr><tr><td>CLIP-g</td><td>ViT-L/14</td><td>×</td><td>×</td><td>×</td><td>33.0</td><td>59.2</td><td>44.2</td><td>64.5</td><td>60.3</td><td>80.9</td><td>54.3</td><td>71.2</td></tr><tr><td>SFF-CLIP(Ours)</td><td>ViT-L/14</td><td>×</td><td>×</td><td>×</td><td>36.9</td><td>66.7</td><td>51.0</td><td>70.8</td><td>75.2</td><td>91.3</td><td>67.3</td><td>79.4</td></tr></table>

The results are shown in Tab. 1. Compared with the CLIP base model, CLIP-g can slightly increase the zero-shot classification performance, due to additional iterations of image-level training. However, SFF-CLIP, which uses both the global contrastive learning and fine-grained matching, obtains significant improvements in accuracy on region classification, for both boxes and masks and for both ViT-B/16 and ViT-L/14 architectures. Compared with other region-aware fine-tuning methods, our proposed SFF-CLIP also achieves outstanding performances. In contrast to other methods that require pre-preparing region proposals [16, 19, 21] and region labels [16, 19], or using pre-defined categories [16, 17], our method can be flexibly applied with just image-text pairs as training data, which is the same data source as the original CLIP pre-training, resulting in better performance. Furthermore, in contrast to the mosaickingbased CLIM [23], our region-phrase pairs from self-annotation provide more effective fine-grained supervision than mosaicking images as pseudo regions, as evidenced by SFF-CLIP outperforming CLIM on all metrics. These superior performances shows the effectiveness of our self-annotated fine-grained alignment approach, which even exceeds these methods that need to carefully prepare region annotations, and achieve comparable performances with the FG-CLIP [13], which is a fine-grained model trained with large-scale region annotated data (1.6B).

Open-vocabulary object detection. Following the previous related works [21, 19, 17], we build open-vocabulary object detectors based on the F-ViT [21] architecture, which is a two-stage detector using a frozen CLIP-based ViT as the backbone. In our experiment, each fine-grained fine-tuned CLIP encoder from the various methods is adopted to initialize the backbone. The OVD models are then trained on the OV-COCO benchmark [46], and we use AdamW optimizer with batch size of 64, learning rate of 1e-4, and weight decay of 0.1. For evaluation, we report box AP (average precision) at IoU (Intersection over Union) of base, novel and all categories as with previous works [16, 9, 10, 27, 21].

The results are presented in Tab. 2. F-ViT is the baseline that initializes the detector backbone with the original pre-trained CLIP. With the ViT-B/16 backbone, adopting just CLIP-g results in similar performance to the baseline CLIP. In contrast, SFF-CLIP significantly improves the OVD results, especially on the novel categories (over 9%). Since the base categories have explicit annotated bounding boxes and labels during OVD training, the performance on the unseen novel categories better illustrates the fine-grained understanding ability brought by the CLIP encoder. Finally, we conduct experiments with ViT-L/14 and further improve the OVD performance on the novel categories by a large extent. We also provide comparisons with related works using ViT-L/14 in the Supplemental, as well as the OVD results with the LVIS benchmark [47]. Compared with the existing OVD methods, which mostly rely on a ResNet-based encoder or modified ViT encoder and require pre-training on large-scale prepared data with extra region information, our method achieves superior or comparable performance by fine-grained training on the easily-obtained image-text pairs.

Open-vocabulary semantic segmentation. We next explore the performance of applying fine-grained fine-tuned models to open-vocabulary semantic segmentation. Following the previous works [21, 19, 17], we adopt the CatSeg [53] as the segmentation architecture, which uses the dense image feature from CLIP ViT as the backbone, with the

Table 2: Results for open-vocabulary object detection on MS COCO val set. F-ViT is the two-stage detector baseline built on the frozen original CLIP ViT, and “+” means the ViT backbone is initialized with a fine-tuned model based on the corresponding method.

<table><tr><td>Method</td><td>Backbone</td><td> $AP_{50}^{novel}$ </td><td> $AP_{50}^{base}$ </td><td> $AP_{50}^{all}$ </td></tr><tr><td>OV-RCNN [48]</td><td>ResNet50</td><td>17.5</td><td>41.0</td><td>34.9</td></tr><tr><td>Detic [49]</td><td>ResNet50</td><td>27.8</td><td>51.1</td><td>45.0</td></tr><tr><td>VLDet [50]</td><td>ResNet50</td><td>32.0</td><td>50.6</td><td>45.8</td></tr><tr><td>F-VLM [9]</td><td>ResNet50</td><td>28.0</td><td>-</td><td>39.6</td></tr><tr><td>CORA [10]</td><td>ResNet50</td><td>35.1</td><td>35.5</td><td>35.4</td></tr><tr><td>SPARC [14]</td><td>ViT-B/16</td><td>-</td><td>-</td><td>39.4</td></tr><tr><td>FG-CLIP [13]</td><td>ViT-B/16</td><td>35.1</td><td>51.7</td><td>47.7</td></tr><tr><td>SigLIP2 [12]</td><td>ViT-B/16</td><td>-</td><td>-</td><td>42.8</td></tr><tr><td>RO-ViT [27]</td><td>ViT-B/16</td><td>30.2</td><td>-</td><td>41.5</td></tr><tr><td>RO-ViT [27]</td><td>ViT-L/16</td><td>33.0</td><td>-</td><td>47.7</td></tr><tr><td>F-ViT</td><td>ViT-B/16</td><td>19.4</td><td>43.3</td><td>37.0</td></tr><tr><td>+CLIP-g</td><td>ViT-B/16</td><td>20.1</td><td>43.8</td><td>37.6</td></tr><tr><td>+RegionCLIP</td><td>ViT-B/16</td><td>27.6</td><td>44.3</td><td>39.9</td></tr><tr><td>+DenseVLM</td><td>ViT-B/16</td><td>17.1</td><td>42.8</td><td>36.1</td></tr><tr><td>+CLIM</td><td>ViT-B/16</td><td>24.1</td><td>43.8</td><td>38.5</td></tr><tr><td>+CLIPSelf</td><td>ViT-B/16</td><td>25.2</td><td>42.2</td><td>37.7</td></tr><tr><td>+FineCLIP</td><td>ViT-B/16</td><td>27.2</td><td>46.0</td><td>41.1</td></tr><tr><td>+SFF-CLIP</td><td>ViT-B/16</td><td>28.8</td><td>46.4</td><td>41.8</td></tr><tr><td>F-ViT</td><td>ViT-L/14</td><td>28.3</td><td>52.5</td><td>46.2</td></tr><tr><td>+CLIP-g</td><td>ViT-L/14</td><td>29.2</td><td>57.5</td><td>50.1</td></tr><tr><td>+SFF-CLIP</td><td>ViT-L/14</td><td>37.4</td><td>57.4</td><td>52.1</td></tr></table>

Table 3: Results of open-vocabulary semantic segmentation on ADE20k-847 [45], Pascal VOC [51], and Pascal Context [52]. CatSeg is the segmentation baseline with the original CLIP ViT-B/16 as the backbone, and “+” means the ViT backbone is initialized with the fine-tuned model based on the corresponding method.

<table><tr><td rowspan="2">Method</td><td colspan="2">VOC-20</td><td colspan="2">PC-59</td></tr><tr><td>mIoU</td><td>pACC</td><td>mIoU</td><td>pACC</td></tr><tr><td>CatSeg</td><td>63.2</td><td>88.9</td><td>44.2</td><td>71.4</td></tr><tr><td>+CLIP-g</td><td>73.8</td><td>92.1</td><td>50.1</td><td>74.4</td></tr><tr><td>+RegionCLIP</td><td>78.4</td><td>93.9</td><td>54.9</td><td>76.2</td></tr><tr><td>+DenseVLM</td><td>75.8</td><td>93.3</td><td>54.5</td><td>77.3</td></tr><tr><td>+CLIM</td><td>70.8</td><td>90.1</td><td>49.3</td><td>72.9</td></tr><tr><td>+CLIPSelf</td><td>75.1</td><td>92.3</td><td>51.6</td><td>75.6</td></tr><tr><td>+FineCLIP</td><td>73.0</td><td>92.0</td><td>50.1</td><td>74.5</td></tr><tr><td>+SFF-CLIP</td><td>79.7</td><td>94.2</td><td>56.9</td><td>78.5</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="2">PC-459</td><td colspan="2">A-847</td></tr><tr><td>mIoU</td><td>pACC</td><td>mIoU</td><td>pACC</td></tr><tr><td>CatSeg</td><td>8.7</td><td>38.2</td><td>5.8</td><td>25.0</td></tr><tr><td>+CLIP-g</td><td>8.6</td><td>37.0</td><td>6.4</td><td>30.3</td></tr><tr><td>+RegionCLIP</td><td>15.9</td><td>62.8</td><td>10.0</td><td>49.5</td></tr><tr><td>+DenseVLM</td><td>13.4</td><td>47.4</td><td>9.0</td><td>35.6</td></tr><tr><td>+CLIM</td><td>12.0</td><td>55.2</td><td>7.3</td><td>42.8</td></tr><tr><td>+CLIPSelf</td><td>13.0</td><td>49.2</td><td>9.3</td><td>34.7</td></tr><tr><td>+FineCLIP</td><td>9.3</td><td>37.9</td><td>6.7</td><td>28.7</td></tr><tr><td>+SFF-CLIP</td><td>16.3</td><td>65.3</td><td>10.2</td><td>58.4</td></tr></table>

![](images/13dac63f3d659ac39e7728d2b66a9aa260d1973d18790f281f22415b8b637feb.jpg)  
Figure 4: Visualization of the fine-grained representations using similarity maps between the image dense feature and the text features for different objects (“car” and “tree”).

CLIP models fine-tuned by all compared methods using the same input image resolution 384x384. After replacing the backbone of CatSeg with the original CLIP model, and each region-aware model, the following segmentation experiments are conducted with training on COCO stuff [54], and evaluation on ADE20k [45], PASCAL VOC [51], and PASCAL Context [52] dataset using mean IoU (mIoU), and pixel Accuracy (pACC). As shown in Tab. 3, SFF-CLIP comprehensively improves the performance of CatSeg on various datasets across different evaluation metrics, and surpasses the enhancements provided by RegionCLIP, CLIPSelf, DenseVLM, FineCLIP, and CLIM.

Overall, the significant improvements on both OVD and OVS demonstrate that our method effectively boosts the fine-grained understanding ability of the CLIP model in downstream tasks.

Qualitative comparison by visualization. In Fig. 4, we provide an example visualization to qualitatively compare the fine-grained understanding ability of models under different alignment methods. Cosine similarities are calculated between the embeddings on each position of the image’s dense features and the text features for “car” and “tree”, and visualized as similarity maps. Without fine-grained alignment, the similarity maps of vanilla CLIP and CLIP-g exhibit weak localization ability (i.e., region-specific attention), where many irrelevant locations have high similarities with the object text. For SFF-CLIP, the text can be matched with the specific regions more accurately than other methods In particular many of the spurious matches to the background have been removed compared to other methods, which

Table 4: Comparison of image-level representation by a zero-shot retrieval task using Flicker30k. The gray row is the baseline CLIP before fine-grained fine-tuning, and <sup>†</sup> represents the results from fine-grained models pre-trained by large-scale data, shown in the brackets.

<table><tr><td rowspan="3">Method</td><td rowspan="3">Model</td><td colspan="6">Flickr30k</td></tr><tr><td colspan="3">text-to-image</td><td colspan="3">image-to-text</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>FG-CLIP $^{\dagger}$  (1.6B)</td><td>ViT-B/16</td><td>74.7</td><td>92.2</td><td>95.6</td><td>88.6</td><td>98.1</td><td>99.3</td></tr><tr><td>SPARC $^{\dagger}$  (3.2B)</td><td>ViT-B/16</td><td>72.0</td><td>91.2</td><td>94.9</td><td>84.4</td><td>97.6</td><td>98.7</td></tr><tr><td>SigLIP2 $^{\dagger}$  (10B)</td><td>ViT-B/16</td><td>80.7</td><td>-</td><td>-</td><td>93.0</td><td>-</td><td>-</td></tr><tr><td>CLIP</td><td>ViT-B/16</td><td>73.6</td><td>90.9</td><td>94.8</td><td>88.6</td><td>97.1</td><td>99.1</td></tr><tr><td>CLIP-g</td><td>ViT-B/16</td><td>76.9</td><td>93.0</td><td>95.9</td><td>90.8</td><td>98.8</td><td>99.4</td></tr><tr><td>RegionCLIP</td><td>ViT-B/16</td><td>70.1</td><td>90.2</td><td>94.3</td><td>81.1</td><td>94.8</td><td>97.0</td></tr><tr><td>FineCLIP</td><td>ViT-B/16</td><td>67.0</td><td>88.5</td><td>93.6</td><td>79.7</td><td>96.3</td><td>97.9</td></tr><tr><td>DenseVLM</td><td>ViT-B/16</td><td>32.4</td><td>56.7</td><td>67.4</td><td>17.8</td><td>34.0</td><td>45.9</td></tr><tr><td>CLIPSelf</td><td>ViT-B/16</td><td>52.3</td><td>78.5</td><td>86.0</td><td>49.9</td><td>77.2</td><td>85.4</td></tr><tr><td>CLIM</td><td>ViT-B/16</td><td>71.2</td><td>90.3</td><td>94.4</td><td>84.1</td><td>97.3</td><td>98.5</td></tr><tr><td>SFF-CLIP(Ours)</td><td>ViT-B/16</td><td>77.4</td><td>93.1</td><td>95.9</td><td>90.8</td><td>98.5</td><td>99.3</td></tr><tr><td>CLIP</td><td>ViT-L/14</td><td>78.8</td><td>94.0</td><td>96.8</td><td>90.4</td><td>98.8</td><td>99.4</td></tr><tr><td>CLIP-g</td><td>ViT-L/14</td><td>82.8</td><td>95.9</td><td>97.9</td><td>93.7</td><td>99.2</td><td>99.8</td></tr><tr><td>SFF-CLIP(Ours)</td><td>ViT-L/14</td><td>82.6</td><td>95.9</td><td>97.8</td><td>94.0</td><td>99.3</td><td>99.8</td></tr></table>

Table 6: Ablation study of the momentum model M<sup>ˆ</sup> .  
Table 7: Ablation study of number of extracted phrase n.

Table 5: Ablation study of phrase matching weight w<sub>t</sub>.

<table><tr><td>Method</td><td></td><td>Boxes</td><td>Masks</td></tr><tr><td>CLIP</td><td>-</td><td>41.4</td><td>30.6</td></tr><tr><td rowspan="2">SFF-CLIP</td><td>w/o  $w_t$ </td><td>39.5</td><td>28.5</td></tr><tr><td>w/  $w_t$ </td><td>57.8</td><td>50.1</td></tr></table>

<table><tr><td>Method</td><td>T2I</td><td>I2T</td></tr><tr><td>CLIP-g w/o  $\hat{M}$ </td><td>76.2</td><td>90.5</td></tr><tr><td>CLIP-g w/  $\hat{M}$ </td><td>76.9</td><td>90.8</td></tr><tr><td>SSF-CLIP w/o  $\hat{M}$ </td><td>74.9</td><td>88.8</td></tr><tr><td>SSF-CLIP w/  $\hat{M}$ </td><td>77.4</td><td>90.8</td></tr></table>

<table><tr><td>Method</td><td>N</td><td>T/epoch</td><td>Boxes</td><td>Masks</td></tr><tr><td>CLIP-g</td><td>0</td><td>51min</td><td>44.7</td><td>33.4</td></tr><tr><td rowspan="3">SFF-CLIP</td><td>1</td><td>53min</td><td>60.7</td><td>49.3</td></tr><tr><td>3</td><td>58min</td><td>61.8</td><td>51.7</td></tr><tr><td>5</td><td>1h</td><td>62.2</td><td>52.1</td></tr></table>

demonstrates the effectiveness of our proposed method to significantly improve the fine-grained representation ability of CLIP model.

## 4.3 Evaluation of image-level representation

The previous works rarely considered how their fine-tuning affects the whole image-text matching, which is the original ability of a vision-language model (VLM). However, preserving the original image-level representation is important in order to support multiple tasks from the same feature backbone model. To explore if there are any negative effects to image-level representation when improving fine-grained representations, we conduct an image-text retrieval evaluation with the Flickr30k [55] validation set, and report the recall accuracy of image-to-text and text-to-image on Top@{1, 5, 10} matching. As seen in Tab. 4, with just realizing image-level contrastive learning, the baseline CLIP-g can keep or improve the performance of the original CLIP by continuing the training on the new image-text pairs data from MS COCO. Our SFF-CLIP successfully preserves the image-level retrieval performance, which is comparable to or even better than CLIP-g. In contrast, other region alignment methods all cause degradation to the image-text retrieval performances compared with the CLIP-g. In particular, CLIPSelf and DenseVLM have largely reduced their abilities of global image-level representation. Finally, the similar results between SFF-CLIP and CLIP-g with ViT-L/14 further demonstrate the effectiveness of our method in maintaining the global representation and stable image-level matching.

## 4.4 Ablation Studies

In this section, we conduct the ablation studies showing the influence of various components in our fine-grained fine-tuning method.

Phrase matching $w _ { t } .$ . To show the function of the phrase-matching weight $w _ { t }$ in the fine-grained matching loss $( \mathrm { E q . } 3 )$ we conduct an ablation study with and without the weight on zero-shot classification performance, as the Top-1 accuracy presented in Tab. 5. Without $w _ { t } .$ , the fine-grained alignment model degrades significantly, with results even worse than the original CLIP. This demonstrates that many phrases extracted from the caption are not matched well with the image, and could cause large disturbances to the region-phrase alignment. Therefore, after introducing the weights to represent the degree of matching for each phrase into the loss, the model can better focus on the learning of well-aligned samples during the fine-tuning, and produce significant performance improvements.

![](images/a459017623d619103109b72bbacea7f4ed1dc3bec89858ba203f13282d033ba6.jpg)

![](images/ce916c81fa5951d99efbdf18bf731faaef7e4a38d5a71d7bfc2c0d030fa80679.jpg)

![](images/5cb6e5f35b3e3f5f57ac1766d931410caff717f6ed6eed0310d85abc90a7032d.jpg)

![](images/6d9b8d8c530e40a12ccbd77ebb7ae1e2fe1d83589cf3e7334b97bdc38828ff58.jpg)  
Figure 5: Ablation study on training data in three different scales.

Momentum model for image-level representation. We conduct an ablation study to investigate the function of using momentum model to generate image and text matching weights in the contrastive learning (Eq. 6). From the text-to-image (T2I) and image-to-text (I2T) results shown in Tab. 6, the image-text retrieval evaluation results are well kept when adopting the momentum model, but decrease when using the pure contrastive loss without the weights from the momentum model.

Maximum number of phrases. Since the text-specific heat maps can be calculated in batch for one phrase in each sentence, the number of loops in obtaining all heat maps is the same as the maximum number of phrases N. Therefore, we conduct the ablation study with N = {0, 1, 3, 5} to observe the influence on the fine-grained alignment’s effectiveness and training efficiency. From Tab. 7, the zero-shot classification Top-1 accuracy is similar when $N = 3$ and N = 5, which is better than $N = 1$ , while much higher than no region-phrase matching (CLIP-g). Thus, $N = 3 , 5$ can basically cover the effective concepts in the training data. For the training efficiency, due to the high speed of the adopted heat-map generation method, the influence of N on the training time cost is not large.

Training data scale. To investigate the impact of data scale on model performance, we randomly sample three trainsets from CC3M [2] with different sizes: 100K, 1M, and 3M samples. We fine-tune the CLIP model with our SSF-CLIP, and compare with CLIP-g, CLIM and CLIPSelf, where CLIPSelf uses image patches instead of preparing region proposals. We report the Top-1 accuracy of zero-shot classification with boxes or masks for fine-grained understanding evaluation and the Top-1 recall rate of T2I and I2T retrieval results for image-level representation evaluation. In Figure 5, we present the performance curves on the four evaluations as the number of samples scales up. In terms of the fine-grained understanding, SSF-CLIP surpasses other methods, and the performance continues to grow as the dataset size increases, which shows promising scalability. For the image-level representation, SSF-CLIP maintains similar retrieval performances as CLIP-g, while in contrast, CLIPSelf and CLIM become worse when the training data increases.

## 5 Conclusion

In this paper, we propose SFF-CLIP (Self-annotated Fine-grained Fine-tuning for CLIP), a novel framework designed to advance the fine-grained representation ability while maintaining the global visual-semantic consistency. Our framework only requires image-text pairs as inputs, avoiding the process of generating region proposals and region annotations (eithetasksnually or via detectors) required by previous works. We propose fine-grained matching with the help of a run-time region-phrase annotation and alignment scheme, which extracts concept phrases from the input sentence and dynamically matches them with image dense features with the help of text-specific heat maps. A momentum model is adopted to support the preservation of CLIP’s original image-level representation ability. We validate the fine-grained dense representation of SFF-CLIP on zero-shot classification task, and down-stream dense prediction tasks, producing consistently significant performance improvements to the baseline model, as well as superior results compared to state-of-the-art works. Meanwhile, the performance on image-level tasks is preserved from the original CLIP model. Training with just image-text pairs, SFF-CLIP provides a flexible and effective solution for improving fine-grained dense representations of vision-language models like CLIP, while also eliminating the requirements of extra training data annotations, such as region proposals and region annotations, as well as dependence on pre-defined categories.

## Acknowledgments

This was was supported in part by......

## A Comparisons on ViT-L/14

We compare with related fine-grained fine-tuning methods, including RegionCLIP [16], CLIPSelf [21], FineCLIP [19], DenseVLM [17], and CLIM [23], on ViT-L/14. For fairness, all compared methods are reproduced with their open-sourced codes to fine-tune the CLIP pre-trained model based on the MS COCO train 2017 set [44] with the same input image size of 336x336.

Table 8: Comparison of fine-grained dense representations via zero-shot classification on the ADE20K panoptic [45] val set and COCO panoptic [44] val2017 set. We report the Top-1 and Top-5 mean accuracy on both object bounding boxes and panoptic masks. The gray row is the baseline CLIP before fine-grained alignment fine-tuning. Our method does not require preparation of region proposals (R.P.), region labels (R.L.), or predefined categories (P.C.) on the training data.

<table><tr><td rowspan="3">Method</td><td rowspan="3">Model</td><td colspan="7">ADE20k</td><td colspan="4">MS COCO</td></tr><tr><td rowspan="2">R.P.</td><td rowspan="2">R.L.</td><td rowspan="2">P.C.</td><td colspan="2">Boxes</td><td colspan="2">Masks</td><td colspan="2">Boxes</td><td colspan="2">Masks</td></tr><tr><td>Top1</td><td>Top5</td><td>Top1</td><td>Top5</td><td>Top1</td><td>Top5</td><td>Top1</td><td>Top5</td></tr><tr><td>CLIP</td><td>ViT-L/14</td><td>-</td><td>-</td><td>-</td><td>31.2</td><td>56.8</td><td>41.2</td><td>62.2</td><td>58.1</td><td>78.9</td><td>49.8</td><td>72.6</td></tr><tr><td>CLIP-g</td><td>ViT-L/14</td><td>×</td><td>×</td><td>×</td><td>33.0</td><td>59.2</td><td>44.2</td><td>64.5</td><td>60.3</td><td>80.9</td><td>54.3</td><td>71.2</td></tr><tr><td>RegionCLIP</td><td>ViT-L/14</td><td>√</td><td>√</td><td>√</td><td>30.7</td><td>57.0</td><td>42.9</td><td>66.3</td><td>66.1</td><td>88.0</td><td>56.4</td><td>76.8</td></tr><tr><td>FineCLIP</td><td>ViT-L/14</td><td>√</td><td>√</td><td>×</td><td>32.0</td><td>58.6</td><td>46.1</td><td>67.8</td><td>68.2</td><td>87.8</td><td>60.5</td><td>77.0</td></tr><tr><td>CLIPSelf</td><td>ViT-L/14</td><td>√</td><td>×</td><td>×</td><td>36.1</td><td>66.4</td><td>49.6</td><td>70.1</td><td>72.0</td><td>90.1</td><td>65.3</td><td>78.3</td></tr><tr><td>CLIM</td><td>ViT-L/14</td><td>×</td><td>×</td><td>×</td><td>30.4</td><td>55.6</td><td>41.7</td><td>63.8</td><td>60.2</td><td>81.5</td><td>58.6</td><td>74.0</td></tr><tr><td>SFF-CLIP(Ours)</td><td>ViT-L/14</td><td>×</td><td>×</td><td>×</td><td>36.9</td><td>66.7</td><td>51.0</td><td>70.8</td><td>75.2</td><td>91.3</td><td>67.3</td><td>79.4</td></tr></table>

## A.1 Results of zero-shot region classification

We compare the fine-grained dense representation on ViT-L/14 with related fine-grained fine-tuning methods using the mean accuraccy (mACC) of classifying region boxes and panoptic masks for “things”. The results on ADE20K panoptic [45] val set and COCO panoptic [44] val2017 set are shown in Tab. 8. Compared to other region-aware fine-tuning methods that require pre-preparing extra region information, our proposed SFF-CLIP achieves superior performance with the ViT-L/14 architecture, further demonstrating the effectiveness of our self-annotated fine-grained alignment on advancing the region-aware representation ability of CLIP.

## A.2 Results of OVD with OV-COCO benchmark

We provide the results of OVD performance on MS COCO val set with ViT-L/14 in Tab. 9. SFF-CLIP significantly improves the OVD results on the unseen novel categories (9.1%), while surpassing other fine-grained fine-tuning methods.

Table 9: Results for open-vocabulary object detection on MS COCO val set. F-ViT is the two-stage detector baseline built on the frozen original CLIP ViT, and “+” means the ViT backbone is initialized with a fine-tuned model based on the corresponding method.

<table><tr><td>Method</td><td>Backbone</td><td> $AP_{50}^{novel}$ </td><td> $AP_{50}^{base}$ </td><td> $AP_{50}^{all}$ </td></tr><tr><td>F-ViT</td><td>ViT-L/14</td><td>28.3</td><td>52.5</td><td>46.2</td></tr><tr><td>+CLIP-g</td><td>ViT-L/14</td><td>29.2</td><td>57.5</td><td>50.1</td></tr><tr><td>+RegionCLIP</td><td>ViT-L/14</td><td>36.9</td><td>52.8</td><td>48.7</td></tr><tr><td>+FineCLIP</td><td>ViT-L/14</td><td>37.2</td><td>54.3</td><td>49.8</td></tr><tr><td>+CLIPSelf</td><td>ViT-L/14</td><td>30.0</td><td>53.8</td><td>47.4</td></tr><tr><td>+SFF-CLIP(Ours)</td><td>ViT-L/14</td><td>37.4</td><td>57.4</td><td>52.1</td></tr></table>

## A.3 Results of image-text retrieval

Moreover, the comparison results on the image-text retrieval task shown in Tab. 10 also indicate that our SSF-CLIP can successfully maintain the global representation and stable image-level matching with ViT-L/14. In contrast, other fine-grained fine-tuning methods all cause degradation of image-text retrieval performances, especially the region-based distillation method CLIPSelf.

Table 10: Comparison of image-level representation by a zero-shot retrieval task using Flicker30k [55]. The gray row is the baseline CLIP before fine-grained fine-tuning, and <sup>†</sup> represents the results from fine-grained models pre-trained by large-scale data, shown in the brackets.

<table><tr><td rowspan="3">Method</td><td rowspan="3">Model</td><td colspan="6">Flickr30k</td></tr><tr><td colspan="3">text-to-image</td><td colspan="3">image-to-text</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>CLIP</td><td>ViT-L/14</td><td>78.8</td><td>94.0</td><td>96.8</td><td>90.4</td><td>98.8</td><td>99.4</td></tr><tr><td>CLIP-g</td><td>ViT-L/14</td><td>82.8</td><td>95.9</td><td>97.9</td><td>93.7</td><td>99.2</td><td>99.8</td></tr><tr><td>RegionCLIP</td><td>ViT-L/14</td><td>75.8</td><td>92.5</td><td>95.9</td><td>85.8</td><td>96.9</td><td>98.8</td></tr><tr><td>FineCLIP</td><td>ViT-L/14</td><td>70.9</td><td>90.3</td><td>94.4</td><td>80.4</td><td>96.5</td><td>98.8</td></tr><tr><td>CLIPSelf</td><td>ViT-L/14</td><td>15.8</td><td>35.8</td><td>47.7</td><td>6.0</td><td>16.5</td><td>23.0</td></tr><tr><td>CLIM</td><td>ViT-L/14</td><td>73.7</td><td>92.2</td><td>95.4</td><td>87.5</td><td>97.5</td><td>99.0</td></tr><tr><td>SFF-CLIP(Ours)</td><td>ViT-L/14</td><td>82.6</td><td>95.9</td><td>97.8</td><td>94.0</td><td>99.3</td><td>99.8</td></tr></table>

## B OVD results on OV-LVIS benchmark

We build open-vocabulary object detectors based on F-ViT architecture using the fine-grained fine-tuned CLIP ViTs as backbones, and train the models for 48 epochs on the OV-LVIS [47] benchmark with the input image size of 224x224 for ViT-B/16. The evaluation results are shown in the Tab. 11 with AP for base categories $( \bar { \mathsf { A P } } _ { c } , \mathsf { A P } _ { f } ) .$ , rare categories $( \operatorname { A P } _ { r } ) ,$ and all categories (AP) as comparison indicators. SSF-CLIP surpasses all other fine-grained fine-tuning methods on all categories, especially on the rare categories, which better illustrates the fine-grained understanding ability brought by the CLIP encoder.

Table 11: Results for open-vocabulary object detection on OV-LVIS val set. F-ViT is the two-stage detector baseline built on the frozen original CLIP ViT, and “+” means the ViT backbone is initialized with a fine-tuned model based on the corresponding method.

<table><tr><td>Method</td><td>Backbone</td><td>AP</td><td> $AP_r$ </td><td> $AP_c$ </td><td> $AP_f$ </td></tr><tr><td>F-ViT</td><td>ViT-B/16</td><td>9.5</td><td>3.1</td><td>6.6</td><td>15.7</td></tr><tr><td>+CLIP-g</td><td>ViT-B/16</td><td>10.7</td><td>5.1</td><td>7.4</td><td>16.1</td></tr><tr><td>+RegionCLIP</td><td>ViT-B/16</td><td>10.4</td><td>5.8</td><td>7.2</td><td>16.1</td></tr><tr><td>+CLIPSelf</td><td>ViT-B/16</td><td>9.0</td><td>3.6</td><td>5.8</td><td>14.9</td></tr><tr><td>+FineCLIP</td><td>ViT-B/16</td><td>10.2</td><td>4.2</td><td>7.0</td><td>16.4</td></tr><tr><td>+SFF-CLIP(Ours)</td><td>ViT-B/16</td><td>11.1</td><td>7.5</td><td>7.6</td><td>16.7</td></tr></table>

## C Ablation study on different XAI methods

In Sec. 3.2, we calculate the text-specific heat maps region-phrase self-annotation based on explainable AI (XAI) methods, and specifically adopt Grad-ECLIP [40] as reference. To investigate the influence of the XAI method used for fine-grained alignment, we conduct an ablation study with other applicable visual explanation approaches that satisfy a high-speed and easy plug-in, including Grad-CAM [33] and MaskCLIP [41]. The Vision Transformer self-attention cannot be used since it is not text-specific. The comparison of zero-shot classification performances in Table 12 shows that with the heat maps from these two methods, the phrase-region alignment also obtains obvious performance improvements compared with just using global loss (CLIP-g), but there is still a gap compared with using Grad-ECLIP, which demonstrates that Grad-ECLIP is the most effective.

## D Ablation study on input image sizes

To explore the impact of input image size on SSF-CLIP, we conduct the fine-grained alignment training and evaluation with four different image resolutions, including 224, 336, 480, 512, on both region-level zero-shot classification and image-level retrieval. The results are shown in Table 13. As the resolution of input image is increased gradually from 224 to 512, the performance on region-level task is improved due to that more visual details can be provided with larger images. On the other hand, the performance on the image-text retrieval task is declined as the image becomes larger, when more number of patches leads to a increase of the complexity for the [cls] token to integrate the global feature.

Table 12: Ablation study of using different visual explanation heat maps in SSF-CLIP.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Explanation Map</td><td colspan="2">Boxes</td><td colspan="2">Masks</td></tr><tr><td>Top1</td><td>Top5</td><td>Top1</td><td>Top5</td></tr><tr><td>CLIP</td><td>-</td><td>41.4</td><td>63.6</td><td>30.6</td><td>53.8</td></tr><tr><td>CLIP-g</td><td>-</td><td>42.9</td><td>64.8</td><td>32.9</td><td>56.4</td></tr><tr><td rowspan="3">SSF-CLIP</td><td>Grad-CAM</td><td>54.2</td><td>74.7</td><td>46.5</td><td>69.8</td></tr><tr><td>MaskCLIP</td><td>54.3</td><td>75.5</td><td>47.4</td><td>70.9</td></tr><tr><td>Grad-ECLIP</td><td>57.8</td><td>78.6</td><td>50.1</td><td>72.9</td></tr></table>

Table 13: Ablation study on input image sizes.

<table><tr><td rowspan="3">Image Size</td><td colspan="4">zero-shot classification</td><td colspan="4">image-text retrieval</td></tr><tr><td colspan="2">Boxes</td><td colspan="2">Masks</td><td colspan="2">text-to-image</td><td colspan="2">image-to-text</td></tr><tr><td>Top1</td><td>Top5</td><td>Top1</td><td>Top5</td><td>R@1</td><td>R@5</td><td>R@1</td><td>R@5</td></tr><tr><td>224</td><td>62.2</td><td>80.8</td><td>52.1</td><td>73.6</td><td>77.4</td><td>93.1</td><td>90.8</td><td>98.5</td></tr><tr><td>336</td><td>63.3</td><td>82.1</td><td>53.7</td><td>74.5</td><td>76.1</td><td>91.8</td><td>88.8</td><td>98.5</td></tr><tr><td>480</td><td>65.5</td><td>86.2</td><td>59.1</td><td>81.1</td><td>70.5</td><td>88.9</td><td>85.5</td><td>96.3</td></tr><tr><td>512</td><td>66.2</td><td>86.8</td><td>65.0</td><td>81.4</td><td>66.6</td><td>86.6</td><td>81.5</td><td>95.2</td></tr></table>

## E Visualization

We present more visualizations of cosine similarity maps between text embeddings and the dense feature maps generated by different fine-grained alignment methods. The comparisons are shown in Figure 6.

## F Baselines

We use the following publicly available source code:

1. CLIPSelf [21] & RegionCLIP [16]: https://github.com/wusize/CLIPSelf

2. CLIM [23]: https://github.com/wusize/CLIM

3. FineCLIP [19]: https://github.com/Timsty1/FineCLIP

4. DenseVLM [17]: https://github.com/HVision-NKU/DenseVLM

## G Broader impact

Our work contribute to introduce SSF-CLIP, which boosts the fine-grained understanding ability of CLIP while maintains the global representation with eliminating the constrains from region annotation preparation. With the growing adoption of transformer as a unified architecture for both vision and language tasks, it is of great significant to enable the great generalization ability of CLIP ViT encoders in both image-level and dense prediction tasks. We are the first to utilize the XAI to produce self-annotated supervision. Eliminating the requirements of cumbersome region data preparation, the method is expected to support larger scale of data and promote improvements of models with further development. To ensure a positive social impact, we conduct experiments using academic open-source datasets that do not involve personal privacy issues.

## H Limitation

The SSF-CLIP is built upon an effective visual explanation (XAI) method for CLIP, the Grad-ECLIP [40]. The self-annotated region-phrase alignment will be influenced by the performance of heat maps generated by the adopted XAI method, as shown in §C. Therefore, the generalization of our idea that utilizes XAI to boost the model itself can be limited by the development of corresponding XAI technique. On the other side, the further progress made in XAI area may push our approach to have a wider range of applications. Since our work provides a flexible and low-cost way to supplement or replace the manually labeling process, SSF-CLIP has the potential to further enhance other pretrained models, which would be explored in our future work.

CLIP CLIP-g RegionCLIP CLIPSelf DenseVLMCLIM FineCLIP SFF-CLIP

![](images/03e82c09d770a7451678dba5ad8b592752be4a7a08e7fd3760a8326fd94dc533.jpg)  
Figure 6: Visualization of cosine similarity maps between text embeddings and the dense feature maps generated by CLIP, CLIP-g, RegionCLIP [16], CLIM [23], CLIPSelf [21], DenseVLM [17], FineCLIP [19] and our SSF-CLIP.

## References

[1] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[2] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, pages 3558–3568, 2021.

[3] Junbum Cha, Kyungjae Lee, Sungrae Park, and Sanghyuk Chun. Domain generalization by mutual-information regularization with pre-trained models. In European conference on computer vision, pages 440–457. Springer, 2022.

[4] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing, 508:293–304, 2022.

[5] Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, and Xiang Bai. A simple baseline for zeroshot semantic segmentation with pre-trained vision-language model. ECCV, 2022.

[6] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. In CVPR, pages 7061–7070, 2023.

[7] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Open-vocabulary object detection via vision and language knowledge distillation. ICLR, 2022.

[8] Yu Du, Fangyun Wei, Zihe Zhang, Miaojing Shi, Yue Gao, and Guoqi Li. Learning to prompt for open-vocabulary object detection with vision-language model. In CVPR, pages 14084–14093, 2022.

[9] Weicheng Kuo, Yin Cui, Xiuye Gu, AJ Piergiovanni, and Anelia Angelova. F-vlm: Open-vocabulary object detection upon frozen vision and language models. In ICLR, 2023.

[10] Xiaoshi Wu, Feng Zhu, Rui Zhao, and Hongsheng Li. Cora: Adapting clip for open-vocabulary detection with region prompting and anchor pre-matching. In CVPR, pages 7031–7040, 2023.

[11] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and Liang-Chieh Chen. Convolutions die hard: Open-vocabulary segmentation with single frozen convolutional clip. NeurIPS, 36, 2024.

[12] Michael Tschannen et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

[13] Chunyu Xie et al. Fg-clip: Fine-grained visual and textual alignment. In ICML, 2025.

[14] Ioana Bica et al. Improving fine-grained understanding in image-text pre-training. In ICML, 2024.

[15] Jinpeng Wang, Pan Zhou, Mike Zheng Shou, and Shuicheng Yan. Position-guided text prompt for vision-language pre-training. In CVPR, pages 23242–23251, 2023.

[16] Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, et al. Regionclip: Region-based language-image pretraining. In CVPR, pages 16793–16803, 2022.

[17] Yunheng Li, Yuxuan Li, Quansheng Zeng, Wenhai Wang, Qibin Hou, and Ming-Ming Cheng. Densevlm: A retrieval and decoupled alignment framework for open-vocabulary dense prediction. In ICCV, 2025.

[18] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. NeurIPS, 28, 2015.

[19] Dong Jing, Xiaolong He, Yutian Luo, Nanyi Fei, Wei Wei, Huiwen Zhao, Zhiwu Lu, et al. Fineclip: Self-distilled region-based clip for better fine-grained understanding. Advances in Neural Information Processing Systems, 37:27896–27918, 2024.

[20] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.

[21] Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Xiangtai Li, Wentao Liu, and Chen Change Loy. Clipself: Vision transformer distills itself for open-vocabulary dense prediction. arXiv preprint arXiv:2310.01403, 2023.

[22] Juan Yeo et al. Atas: Any-to-any self-distillation for enhanced open-vocabulary dense prediction. In ICCV, 2025.

[23] Size Wu et al. Clim: Contrastive language-image mosaic for region representation. In AAAI, volume 38, pages 6117–6125, 2024.

[24] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. IJCV, 130(9):2337–2348, 2022.

[25] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv:2205.01917, 2022.

[26] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, 2022.

[27] Dahun Kim, Anelia Angelova, and Weicheng Kuo. Region-aware pretraining for open-vocabulary object detection with vision transformers. In CVPR, 2023.

[28] Weicheng Kuo, Yin Cui, Xiuye Gu, AJ Piergiovanni, and Anelia Angelova. Open-vocabulary object detection upon frozen vision and language models. In The Eleventh International Conference on Learning Representations, 2023.

[29] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple open-vocabulary object detection. In European conference on computer vision, pages 728–755. Springer, 2022.

[30] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scaling open-vocabulary image segmentation with image-level labels. In European conference on computer vision, pages 540–557. Springer, 2022.

[31] Yunheng Li, Zhong-Yu Li, Quan-Sheng Zeng, Qibin Hou, and Ming-Ming Cheng. Cascade-clip: Cascaded vision-language embeddings alignment for zero-shot semantic segmentation. In International Conference on Machine Learning, pages 28243–28258. PMLR, 2024.

[32] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In ECCV, pages 818–833, 2014.

[33] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In ICCV, pages 618–626, 2017.

[34] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. " why should i trust you?" explaining the predictions of any classifier. In ACM SIGKDD, 2016.

[35] Vitali Petsiuk, Abir Das, and Kate Saenko. Rise: Randomized input sampling for explanation of black-box models. arXiv:1806.07421, 2018.

[36] Chenyang Zhao and Antoni B Chan. Odam: Gradient-based instance-specific visual explanations for object detection. In ICLR, 2023.

[37] Yao Qiang, Deng Pan, Chengyin Li, Xin Li, Rhongho Jang, and Dongxiao Zhu. Attcat: Explaining transformers via attentive class activation tokens. NeurIPS, 2022.

[38] Weiyan Xie, Xiao-Hui Li, Caleb Chen Cao, and Nevin L Zhang. Vit-cx: Causal explanation of vision transformers. pages 1569–1577, 2023.

[39] Lu Yu and Wei Xiang. X-pruner: explainable pruning for vision transformers. In CVPR, pages 24355–24363, 2023.

[40] Chenyang Zhao, Kun Wang, Xingyu Zeng, Rui Zhao, and Antoni B Chan. Gradient-based visual explanation for transformer-based clip. In ICML, pages 61072–61091. PMLR, 2024.

[41] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free dense labels from clip. In ECCV, pages 696–712. Springer, 2022.

[42] Steven Bird, Ewan Klein, and Edward Loper. Natural language processing with Python: analyzing text with the natural language toolkit. O’Reilly Media, Inc., 2009.

[43] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. preprint arXiv:2303.15389, 2023.

[44] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

[45] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In CVPR, 2017.

[46] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015.

[47] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5356–5364, 2019.

[48] Alireza Zareian, Kevin Dela Rosa, Derek Hao Hu, and Shih-Fu Chang. Open-vocabulary object detection using captions. In CVPR, pages 14393–14402, 2021.

[49] Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Krähenbühl, and Ishan Misra. Detecting twenty-thousand classes using image-level supervision. In ECCV, 2022.

[50] Chuang Lin, Peize Sun, Yi Jiang, Ping Luo, Lizhen Qu, Gholamreza Haffari, Zehuan Yuan, and Jianfei Cai. Learning object-language alignments for open-vocabulary object detection. arXiv preprint arXiv:2211.14843, 2022.

[51] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results. http://www.pascalnetwork.org/challenges/VOC/voc2012/workshop/index.html, 2012.

[52] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In CVPR, pages 891–898, 2014.

[53] Seokju Cho, Heeseong Shin, Sunghwan Hong, Anurag Arnab, Paul Hongsuck Seo, and Seungryong Kim. Cat-seg: Cost aggregation for open-vocabulary semantic segmentation. In CVPR, pages 4113–4123, 2024.

[54] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. In CVPR, pages 1209–1218, 2018.

[55] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In ICCV, 2015.