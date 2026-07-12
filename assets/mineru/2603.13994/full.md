# Human-like Object Grouping in Self-supervised Vision Transformers

Hossein Adeli<sup>1</sup> , Seoyoung Ahn<sup>2</sup>, Andrew Luo<sup>3</sup>, Mengmi Zhang<sup>4</sup>, Nikolaus Kriegeskorte<sup>1</sup>, and Gregory Zelinsky<sup>5</sup>

<sup>1</sup> Zuckerman Mind Brain Behavior Institute, Columbia University, New York 2 Department of Social Science and AI, Hankuk University of Foreign Studies, Seoul 3 University of Hong Kong, Hong Kong <sup>4</sup> Nanyang Technological University, Singapore 5 Stony Brook University, New York ha2366@columbia.edu

Abstract. Vision foundation models trained with self-supervised objectives achieve strong performance across diverse tasks and exhibit emergent object segmentation properties. However, their alignment with human object perception remains poorly understood. Here, we introduce a behavioral benchmark in which participants make same/diferent object judgments for dot pairs on naturalistic scenes, scaling up a classical psychophysics paradigm to over 1000 trials. We test a diverse set of vision models using a simple readout from their representations to predict subjects’ reaction times. We observe a steady improvement across model generations, with both architecture and training objective contributing to alignment, and transformer-based models trained with the DINO self-supervised objective showing the strongest performance. To investigate the source of this improvement, we propose a metric to quantify the object-centric component of representations by measuring patch similarity within and between objects. Across models, stronger objectcentric structure predicts human segmentation behavior more accurately. We further show that matching the Gram matrix of supervised transformer models, capturing similarity structure across image patches, with that of a self-supervised model through distillation improves their alignment with human behavior, converging with the prior finding that Gram anchoring improves DINOv3’s feature quality. Together, these results demonstrate that self-supervised vision models capture object structure in a behaviorally human-like manner, and that Gram matrix structure plays a role in driving perceptual alignment. Code and data are available here.

Keywords: Object Segmentation · Binding · Self-supervised Learning · Gram Matrix · Human Visual Perception · Behavioral Benchmark

A  
![](images/897584de3697b7f03918afb6c04c76ce44926ec61add80c284f9218d4a8dba0f.jpg)  
Fig. 1: A) Example afinity maps for a few image patches, generated using feature tokens from the DINOv3 ViT-B/16 model. For a given patch, the cosine similarity between its feature vector and the feature vectors of all other patches is shown as an afinity map, with brighter values indicating stronger similarity. Patches belonging to the same object tend to have the highest afinity, reflecting object-centric structure in the representations. B) The full Gram matrix for the same image, showing pairwise feature similarities between all patches. The block-like structure visible along the diagonal reflects clusters of patches with high mutual similarity, corresponding to distinct objects in the scene.

## 1 Introduction

Grouping visual input into coherent whole objects [34, 35, 41] is a fundamental problem for the brain and AI systems. The human visual system achieves this by relying on diferent signals ranging from part-whole relationships and Gestalt processes [1,3,5,7,11,16,19,23,39,45,46], to prior semantic knowledge of object categories [17,18,33,42,43]. This visual integration process has been proposed to be mediated by ‘association fields’ [15], long-range lateral connections in retinotopic visual cortical areas linking distant points. The efective connectivity between units in these maps has been shown to depend on the similarity between locally represented features (e.g., orientation). Models of object perception have focused on implementing these cues and studying grouping as a gradual activation of lateral connections [22, 35]. However, most behavioral and modeling experiments in this domain have focused on Gestalt cues and objects with clear boundaries [22], and not in natural contexts where objects are less spatially separated or contours lack clear definition.

Recent advances in vision foundation models show that similar object grouping properties can emerge from large-scale self-supervised training of transformer models [6, 20, 32, 37]. The self-attention mechanism in these models forms dynamic contextual connections between diferent locations of the visual input, similar to association fields forming in retinotopic maps of the visual cortex. Indeed, recent large-scale model comparisons show that contour integration also emerges in these models as training data and model scale increase, with the strongest models beginning to approach human-level integration performance [30]. Contour integration, while a hallmark of human object perception, still puts the focus on low-level Gestalt cues, and it remains to be tested whether these findings extend to naturalistic settings [9].

In this work, we address this gap and test whether the object grouping properties of vision foundation models are aligned with human object perception. We first collected human data on a well-controlled behavioral experiment to probe how people group complex objects in naturalistic scenes. Subjects responded to whether two dots placed on a scene were on the same object or two diferent objects. Their reaction time on this task reveals how they perceive objects and the nuances of how easy or hard it is to group them. We then test whether the patch representations from the models can predict human object grouping behavior. Higher performance on this task indicates that the model representations capture human-like object structure. We then quantify the object-centric component in the patch-level representations of these models as a way to understand what drives this alignment. Figure 1A shows the afinity maps for a few patches using token features from the DINOv3 model [37]. Each map is created by computing the cosine similarity between the token representation of a patch and all other patches in the image (with higher values shown brighter). Notably, patches that are more similar tend to lie on the same object, suggesting that object identity is reflected in patch-level representations of these models. Our main contributions are as follows.

– We introduce a large-scale behavioral benchmark in which human participants make same/diferent object judgments for dot pairs on naturalistic scenes, scaling up a classical psychophysics paradigm to a diverse set of natural images.

– We propose a novel object-centric metric based on ROC analysis of patchlevel afinity maps, which quantifies the degree to which a model’s representations reflect object boundaries without requiring object-level supervision.

– We evaluate a diverse set of vision models and show that self-supervised Transformer models trained with the DINO objective achieve the strongest alignment with human object grouping behavior, with training objective contributing more strongly than architecture in our model set, though these factors are not fully decoupled.

– We demonstrate that object-centric structure in patch representations, as measured by our proposed metric, is strongly predictive of behavioral alignment across models and training objectives.

– We show that fine-tuning supervised models to match the Gram matrix structure of a self-supervised model improves their object-centricity and behavioral alignment, converging with independent evidence that Gram anchoring is a key mechanism underlying the dense feature quality of DINOv3.

## 2 Related Work

Self-supervised Vision Transformers. Application of Transformers to vision has been extremely successful, with these models outperforming convolutional neural networks (CNNs) on object recognition and other tasks [13]. The selfattention weights in supervised vision Transformers have been shown to perform some perceptual grouping [8,25,31]. More recently, studies have explored training these models with self-supervised objectives, yielding intriguing object-centric properties that are less prominent in models trained for classification. When trained with a self-distillation loss (DINO [6], DINOv2 [32], and DINOv3 [37]), the attention maps contain explicit information about the semantic segmentation of foreground objects and their parts, reflecting that these models can capture object-centric representations without labels. These models have also been shown to achieve state-of-the-art performance in predicting brain activity in visual areas in response to complex scenes [2, 21], further pointing to their plausibility as models of human-like visual processing. In masked autoencoding (MAE [20]), the input image is heavily occluded and the model is trained to reconstruct the full image from a small number of visible patches. Minimizing the reconstruction loss enables the model to learn object-centric features that yield strong performance on downstream tasks.

Self-supervised Transformers for object and part discovery. There have been recent attempts to investigate the extent to which self-supervised Transformers can learn high-level characteristics of a scene. These studies involve computing feature similarity among all tokens and examining their correspondence with high-level concepts such as objects and parts. LOST [36] and TokenCut [44] use the similarity graph to perform unsupervised object discovery, showing success when there is one salient object in the scene. Other work [4] has used feature similarity to perform co-segmentation of object parts. These results collectively corroborate that vision Transformers trained with a self-supervised objective begin to represent object-centric information, meaning the patches with the highest afinity to a given patch tend to lie on the same object (Fig. 1A). Consistent with this, recent work using "visual anagrams" to probe configural shape processing shows that self-supervised Transformers exhibit strong sensitivity to global part configuration, relying on long-range interactions and showing a transition from local to global coding across layers [12]. This suggests that these models do not merely rely on local texture cues, but can integrate spatially distributed part information into more holistic object representations [26, 27].

## 3 Behavioral Experiment

We use a "two-dot" paradigm (Fig. 2) to directly probe how humans group and segment regions of natural images into objects. In this paradigm, two dots are placed on an image and participants are asked to indicate by button press whether they are on the same object or two diferent objects (Fig. 2A). One dot is always at the center of the image, and the other is at a peripheral location.

A  
![](images/344ec06f4081449d2cfa0b0c86873348efe1765f70f7dca647d48a9620be5642.jpg)  
C

![](images/8228f9ddc1eb072175b342963adbb1d3f46181ea00c27b49466b8c364e48a80b.jpg)

B  
![](images/62c3fccf5d099998f0b2b05b3cc8f52bf6f5914190eefe6e1fc21be3b8b4b3a1.jpg)

![](images/2d509cd81946a7e6126c28eadf8ed780ac5d46601ecabbb63260268e4ab9f053.jpg)  
Fig. 2: A) Behavioral procedure. Participants maintain fixation on a center dot during the trial. A second dot appears following the center dot and remains visible for 1000 ms, after which the scene appears and the dots begin flickering to ensure their visibility. Subjects are instructed to respond whether the two dots are on the same or two diferent objects as quickly as possible without sacrificing accuracy. B) Sample trials from all four experimental conditions, coded by diferent colors. C) Placement of dots across all conditions and trials. D) Mean reaction time for correct trials by condition, with SEM (Standard Error of the Mean) error bars.

The reaction time (RT) of this button press is the primary measure in this task and reveals the dificulty of object grouping. Critically, RT in this task has been shown to track known perceptual grouping efects such as the same-object advantage and Gestalt cues [24,42], consistent with its use as a measure of object segmentation dificulty. Previous works using this paradigm have been limited in scale or have focused on simpler stimuli [23, 24, 42]. For example, [24] used 24 hand-selected images depicting two instances of either a vehicle or an animal. Our work significantly scales up this efort.

## 3.1 Behavioral Methods

72 undergraduate students participated in our experiment for course credit. Their mean age was 20.4 years (range = 17–32) and all had normal or correctedto-normal vision. This study was approved by the school Institutional Review Board.

Stimuli and Apparatus: The images in our experiment were selected from the COCO2017 validation dataset [28]. Using the object-level annotations, we first selected images that had one object overlapping the center of the image, where we placed the center dot, ensuring that the reference point for the grouping judgment always fell within a well-defined object boundary. Among those, we selected images that had a second object overlapping the first. We then used an algorithm to place four peripheral markers (for same/diferent and close/far conditions) (Fig. 2B), ensuring that distances were matched between same and diferent conditions to prevent participants from making judgments based on distance alone. We only retained images where valid locations could be found for all four conditions. After this process, we manually inspected the remaining images and excluded those where dot placements fell outside object boundaries due to annotation inaccuracies or where the intended object was ambiguous upon visual inspection, leaving 288 images. Of these, 32 were used as practice trials and 256 as experimental trials. Fig. 2C shows the placement of dots across all four conditions. Each participant saw each image in only one condition (same-close, same-far, diferent-close, or diferent-far). We removed one experimental image from our analyses because the ground truth response was ambiguous due to the dot falling on a boundary shared by two objects, leaving 255 experimental images and 1020 (255×4) trials for behavioral analyses and modeling experiments.

The assignment of images to the four conditions was counterbalanced across participants such that each image was seen in all four conditions across every group of four participants. The experiment was conducted on a 19-inch flatscreen CRT ViewSonic SVGA monitor with a screen resolution of 1024×768 pixels and a refresh rate of 100 Hz. Participants were seated approximately 70 cm from the monitor, which subtended a visual angle of $3 0 ^ { \circ } \times 2 2 ^ { \circ }$ . At this viewing distance, approximately 34 pixels spanned 1 degree of visual angle, placing the close peripheral dot approximately 3 degrees from the center dot and the far peripheral dot approximately 6 degrees from the center dot. Gaze position was recorded using an EyeLink 1000 eye-tracking system (SR Research) at a sampling rate of 1000 Hz. Fixations were parsed using the default EyeLink algorithm, with a velocity threshold of 30 degrees per second and an acceleration threshold of 8000 degrees per second squared. Calibration drift was checked before every trial, and recalibration was performed if necessary to ensure accurate eye-tracking.

Procedure: Participants were instructed to determine whether the two dots were on the same object or two diferent objects. Each trial started with the presentation of a central dot for 500 ms, indicating the location of the fixation point. Both the central and peripheral dots were then displayed for 1,000 ms without the image. Next, the dots were superimposed on the image and flickered at a frequency of 5 Hz to ensure their visibility. During the trial, participants were required to maintain their gaze on the center dot for the entire duration. If their gaze deviated more than 1 degree of visual angle from that location, the trial was terminated. 7% of trials were excluded due to fixation breaks, defined as gaze deviating beyond this threshold. To record their responses, participants used a Microsoft gamepad controller, with buttons assigned to "same" or "diferent" responses. The button-hand assignment was randomized across participants to prevent dominant hand bias. Each participant performed 32 practice trials followed by 256 experimental trials, divided into four blocks of 64 trials each, with breaks provided between blocks. The order of image presentation within each block was randomized. Incorrect responses were indicated by an auditory feedback tone.

![](images/d37cffbbb2e7d754f0600e3d277b12d82f08fced69b62e8318e6fab34afbc572.jpg)  
Fig. 3: Sample behavioral results for the four conditions in our experiment. The mean reaction time across subjects is displayed above.

## 3.2 Behavioral Results

The average subject accuracy on this task was 90%. We report analyses only for trials where the subject response was correct; however, the patterns were largely the same when including both correct and incorrect trials. Fig. 2D shows the RT data for each condition. Subjects were faster to respond when the two dots were on the same object compared to when the peripheral dot was on a diferent object. This efect is known as the same-object advantage [14], indicating that the first dot facilitates selection of the whole object. This efect interacted with dot distance: we observed the fastest RTs in the close-separation same-object condition. When the peripheral dot was on a diferent object, dot separation had little efect on mean RTs, consistent with prior work [35]. Detailed analyses of accuracy, sensitivity (d-prime), and response bias across subjects and conditions are provided in the supplementary.

Fig. 3 shows four sample trials for each condition with comparable dificulties. For this visualization, we first ordered the trials within each condition by RT and selected the 50th, 100th, 150th, and 200th trials. Notably, RTs increase with dot distance when the dots are on the same object. Comparing the cat images (top row) illustrates this pattern. When the dots are on diferent objects, dot separation has little efect on RTs, as seen by comparing the elephant images (bottom row). While average and cross-condition behavioral patterns are informative, there is also interesting variability within each condition. Task dificulty, and RTs, increase when there are within-object boundaries between the dots, when the dots fall on diferent object parts with diferent textures, when they are on narrower parts of the object, when they are close to object boundaries, or when multiple objects from the same category are present. Our behavioral dataset therefore captures the variable conditions under which humans group objects in natural scenes. We will test models on how well they can predict the mean RT across subjects for each trial.

Table 1: Vision models evaluated in this study and their grouping accuracy.

<table><tr><td>Model</td><td>Architecture</td><td>Training regime</td><td>Large dataset</td><td>Grouping accuracy</td></tr><tr><td>DINOv3 ViT B</td><td>Transformer</td><td>Self-supervised</td><td>Yes</td><td>91.9</td></tr><tr><td>DINOv2 ViT B</td><td>Transformer</td><td>Self-supervised</td><td>Yes</td><td>89.0</td></tr><tr><td>DINO ViT B</td><td>Transformer</td><td>Self-supervised</td><td>No</td><td>76.5</td></tr><tr><td>MAE ViT B</td><td>Transformer</td><td>Self-supervised</td><td>No</td><td>80.7</td></tr><tr><td>IN21k ViT B</td><td>Transformer</td><td>Supervised</td><td>Yes</td><td>72.2</td></tr><tr><td>IN1K ViT B</td><td>Transformer</td><td>Supervised</td><td>No</td><td>70.6</td></tr><tr><td>IN21k ConvNext B</td><td>Convolutional</td><td>Supervised</td><td>Yes</td><td>67.4</td></tr><tr><td>IN1K ConvNext B</td><td>Convolutional</td><td>Supervised</td><td>No</td><td>60.0</td></tr><tr><td>DINOv3 ConvNext B</td><td>Convolutional</td><td>Distilled</td><td>Yes</td><td>86.7</td></tr></table>

## 4 Modeling Experiments

In our modeling experiments, we evaluate a diverse set of vision models, as shown in Table 1. We consider both Transformer and convolutional architectures. For Transformer models, we focus on the ViT-Base architecture [13], including models trained with self-distillation (DINO [6], DINOv2 [10, 32], and DINOv3 [37]), masked autoencoding (MAE [20]), and supervised training on ImageNet-1K with or without ImageNet-21K pretraining [38] using the DeiT3 [40] method. For these models, we extract patch features from the last Transformer layer. For convolutional models, we include ConvNext-Base [29] models trained with supervised objectives on ImageNet-1K with or without ImageNet-21K pretraining, as well as a version distilled from a Transformer model trained with the DINOv3 objective. Input images are resized so that the resulting convolutional feature map dimensions match those of a ViT-Base model with patch size 16 on the original images. Features are then extracted from the last convolutional layer. The feature tensors (with size h × w × d) are then divided into h × w feature tokens of length d to represent diferent patches of the image. Further model implementation and training details are provided in the supplementary.

To predict human grouping judgments, we extract patch features from the two dot locations in each image and concatenate them to form a trial representation, with the central patch feature concatenated first followed by the peripheral patch feature. A two-layer MLP readout is then trained to predict the same or diferent response from this concatenated representation. To provide suficient training data for this readout, we applied the same dot placement algorithm to the COCO2017 training set to generate approximately 30,000 trials, which were used to train the MLP classifier. The model was then evaluated on the 1,020 held-out behavioral trials. Grouping accuracy results are shown in Table 1. Self-supervised Transformer models trained with the DINO objective achieve the strongest performance, with DINOv3 ViT B reaching 91.9% accuracy. Supervised Transformer models perform considerably lower despite sharing the same architecture, suggesting that the training objective rather than architecture alone drives object-centric representations. MAE ViT B (80.7%) falls between the self-supervised DINO models and the supervised models, consistent with it learning some object-centric structure but not to the same degree as self-distillation. Among convolutional models, DINOv3 ConvNext B (86.7%) substantially outperforms its supervised counterparts, further highlighting the quality of the DINOv3 dense features.

## 4.1 Behavior Prediction

To predict trial-by-trial reaction times, we use the same 2-layer MLP readout procedure but train it on the 1,020 behavioral trials using nested cross-validation. In the outer loop, we use 10-fold cross-validation, training on 90% of the trials and testing on the held-out 10%. Within each outer fold, the 90% training data is further split into a 90/10 train-validation split used for early stopping and hyperparameter selection. We train 10 random seeds per outer fold and average their predictions, resulting in a robust estimate of each model’s ability to predict human RTs. To account for noise in the behavioral measurements, we normalize model performance by a human noise ceiling. This ceiling is estimated independently of any model: we randomly split the subjects into two equal halves 20 times, compute the Spearman correlation between the mean RTs of the two halves for each split, and average across the 20 splits to obtain a single noise ceiling value. Separately, for each model we compute the Spearman correlation between the model’s predicted RTs and the mean RTs of each of the 20 subject splits, and average these correlations across splits to obtain the model-human correlation. The normalized Spearman correlation is then obtained by dividing this model-human correlation by the noise ceiling, such that a value of 1.0 would indicate alignment with human behavior at the level of the noise ceiling. Raw (non-normalized) Spearman correlations are reported in the supplementary.

Results are shown in Fig. 4A. The same ordering observed in the grouping task largely holds here: self-supervised Transformer models trained with the DINO objective achieve the strongest behavioral alignment, with DINOv3 ViT B reaching the highest normalized Spearman correlation among all models tested. Supervised models show considerably weaker alignment despite sharing the same architecture, again pointing to the training objective as the key factor. MAE ViT B falls in an intermediate range, and convolutional models show the weakest alignment overall, with DINOv3 ConvNext B being a notable exception. Fig. 4B shows the mean RTs predicted by the best-performing model, DINOv3 ViT B, broken down by condition. The predicted RTs show the same-object advantage and the distance efect within the same-object condition observed in the behavioral data, confirming that the model captures the qualitative structure of human object grouping behavior.

A  
![](images/e3c2b631c739a8541bd3685a78f1593c6b9bb62233fed9ae6a2c632a093c1a9a.jpg)

B  
![](images/c364edab6c9b46cc8e2ab053b16e4c83ea7cc2156e27133655f1b96eb38bdd04.jpg)  
Fig. 4: A) Noise-normalized Spearman correlation between model-predicted and human reaction times across all models, ordered from lowest to highest. Models trained with self-supervised DINO objectives consistently outperform supervised counterparts with the same architecture. B) Mean reaction times predicted by DINOv3 ViT B for each experimental condition. The model reproduces the key signatures of human grouping behavior, including faster responses for same-object trials and a distance efect that is specific to the same-object condition.

## 4.2 Object-Centric Representations

To quantify the object-centric structure in model representations, we compute afinity maps by calculating the cosine similarity of each patch token’s feature with all other tokens, yielding a measure of feature similarity between each patch and the rest of the image. Sample afinity maps for a few patch locations are shown in Fig. 1A, generated using patch features from the DINOv3 ViT B model. We then perform an ROC analysis across our experimental images to quantify how well these afinity signals align with ground truth object boundaries. For each trial, we compute the afinity map from the central dot location (one sample map, also from the DINOv3 ViT B model and taken from Fig. 1A, is shown on the left of Fig. 5A) and apply a range of thresholds to assess the spatial distribution of active patches relative to the ground truth object boundary. The True Positive Rate (TPR) is computed as the proportion of within-object patches whose afinity exceeded the threshold, and the False Positive Rate (FPR) as the proportion of outside-object patches whose afinity exceeded the threshold. As shown in Fig. 5A, in this case TPR increases substantially as the threshold decreases while FPR remains low, indicating that the patches most similar to the central patch tend to be on the same object. Only once TPR reaches a high level does FPR begin to rise, reflecting a strong object-centric signal in the afinity map. Averaging TPR and FPR across all trials yields a summary ROC curve for each model. Fig. 5B shows ROC curves for all 12 layers of DINOv3 ViT B, with deeper layers showing progressively stronger object-centric structure (legend ordered by decreasing AUC). Fig. 5C shows ROC curves across the 12 attention heads of the last layer, showing that object-centric structure is broadly distributed across heads, with each head contributing similarly to patch-level object representation.

A  
![](images/b6c332661b6717aa57edeb2163e4d934fb8a8a4a3641f3cb18bf15dd7fe951fa.jpg)

B  
![](images/9b7d9bd6ef6bfe16e81ebc0eb9aa134e377fa98f7fd3b7c971b31cb7dcad8934.jpg)

C  
![](images/059ef4602d21d736619d0f74a7be209bbc3cc78771fc64ca62abf4373f66e51f.jpg)  
Fig. 5: A) A sample experimental trial with the central dot shown on the top left, alongside its afinity map (using DINOv3 features) showing normalized feature similarity between the central patch and all other patches (colorbar shown). For decreasing threshold values (θ), patches with afinity above the threshold are shown in yellow. The TPR and FPR are displayed above each thresholded map. TPR increases substantially before FPR rises, indicating a strong object-centric signal in the afinity map. B) ROC curves averaged across all trials for each of the 12 layers of the DINOv3 ViT B model. The legend is ordered by decreasing AUC, with deeper layers showing stronger objectcentric structure. C) ROC curves for features from diferent attention heads of the last layer of DINOv3 ViT B, showing broadly similar object-centricity across heads.

A  
![](images/a797705807f6ad3f097a2e634ee6aa00fc8a34801897df48584afeda82f172ee.jpg)

B  
![](images/575b9ac1f57ed098783c9e3e61a0edc9fd62da520fb88950fdeeffa17cb48e58.jpg)  
Fig. 6: A) ROC curves quantifying the object-centricity of patch-level representations for all models evaluated in this study, using features from the final layer. The legend is sorted by decreasing AUC, with self-supervised DINO-based models consistently achieving higher object-centricity than supervised or reconstruction-based counterparts. The diagonal dashed line indicates chance performance. B) Scatter plot relating each model’s object-centricity (AUC) to its noise-normalized Spearman correlation with human reaction times. Each circle represents one model. Models with stronger object-centricity tend to exhibit greater alignment with human perceptual behavior, with the relationship holding across both Transformer and convolutional architectures. The correlation between object-centric AUC and behavioral alignment across all 9 models is Spearman r=0.950, p=0.0001.

To compare object-centricity across all models, we compute the AUC of the ROC curve for each model using features from its last layer. Fig. 6A shows the ROC curves for all models, ordered by decreasing AUC in the legend. Selfsupervised Transformer models trained with the DINO objective achieve the highest AUC values, reflecting strong object-centric structure in their patch representations. DINOv3 ViT B leads across all models, followed closely by DI-NOv2 ViT B, while supervised Transformer and convolutional models show progressively weaker object-centricity. Notably, DINOv3 ConvNext B, despite being a convolutional model, achieves a substantially higher AUC than its supervised convolutional counterparts, suggesting that strongly object-centric representations, learned through distillation, can be computed in this architecture as well. MAE ViT B falls in an intermediate range, consistent with it learning some degree of object-centric structure through reconstruction but not to the same extent as self-distillation models. Fig. 6B shows the relationship between each model’s object-centric AUC and its behavioral alignment (noise-normalized Spearman correlation). The two measures are strongly correlated across models: models with stronger object-centric structure in their representations also predict human reaction times more accurately. This relationship holds across both architecture types and training objectives, suggesting that object-centricity is a general principle linking model representations to human perceptual behavior. Together, these results indicate that the degree to which a model encodes ob-

Effect of Gram alignment

A  
![](images/92de7fd7d4bdf714bbbb68b61b80a47a5dd29a9deba368862cebc4b21de124b3.jpg)

B  
![](images/30ff14de47170fbbd49f6084c320339fe4d60904caca3caefcd90c5676df3cc1.jpg)  
C

![](images/784e726809ab1b0ac81dda7752eb5e48d08c0e701541adf6952d711ff21186da.jpg)  
Fig. 7: Efect of Gram matrix alignment on model performance across three metrics. Each panel shows the change from the base supervised model (left) to the Gramaligned version (right) for four models: IN21k ViT B, IN1K ViT B, IN21k ConvNext B, and IN1K ConvNext B. Gram alignment is performed by fine-tuning each model on ImageNet classification while distilling the Gram matrix structure from DINOv3 ViT B. A) Grouping accuracy improves substantially for all models following alignment. B) Object-centric AUC increases consistently, with aligned models approaching the performance of self-supervised DINO models. C) Behavioral alignment with human reaction times improves across all models, with Transformer-based models showing larger gains than convolutional models.

ject identity at the patch level is a key factor driving its alignment with human object grouping, and that the DINO family of training objectives promotes this property more efectively than supervised or reconstruction-based alternatives. Scatter plots showing correlations between best model predictions and the behavioral RTs for each of the four conditions are provided in the supplementary results section.

## 4.3 Gram Alignment

The Gram matrix captures the pairwise similarity structure across all feature vectors in a representation. Here, for a model with $h \times w$ patch tokens, we compute the cosine similarity between every pair of patch feature vectors, yielding an $( h \times w ) \times ( h \times w )$ matrix where each entry reflects the degree to which two patches share similar features. Fig. 1B shows an example Gram matrix, where brighter values indicate stronger feature similarity between patch pairs. Importantly, in models with strong object-centric representations, patches belonging to the same object tend to have higher pairwise similarities, resulting in a block-like structure in the Gram matrix that reflects object boundaries.

To investigate whether explicitly aligning the Gram matrix of supervised models with that of a self-supervised model improves their representations, we fine-tune four supervised models, IN21k ViT B, IN1K ViT B, IN21k ConvNext B, and IN1K ConvNext B, on ImageNet classification while simultaneously distilling the Gram matrix structure from DINOv3 ViT B (Fig. 7). Gram alignment consistently improves grouping accuracy across all four models (Fig. 7A), with gains ranging from approximately 8 to 18 percentage points. A similarly consistent improvement is observed in object-centric AUC (Fig. 7B), with all models showing substantially higher object-centricity after Gram alignment, approaching the AUC values of the self-supervised DINO models. Behavioral alignment also improves across all models (Fig. 7C), though the magnitude of improvement varies. Notably, the two Transformer models show larger gains than the ConvNext models across all three metrics, suggesting that the Transformer architecture is more amenable to Gram matrix alignment, likely due to its selfattention mechanism already encoding pairwise patch relationships. Interestingly, the gains from Gram alignment are substantial even for models without large-scale pretraining, suggesting that Gram alignment can partially compensate for the absence of large dataset training, particularly in Transformer models where the gains are most pronounced. Taken together, these results demonstrate that matching the Gram matrix structure of a self-supervised model can substantially improve the object-centricity and behavioral alignment of supervised models, pointing to feature correlation structure across image patches as a key mechanism underlying human-like object representations. A CLS token distillation baseline isolating Gram structure as the active ingredient is reported in the supplementary results section.

## 5 Conclusion

Classical theories of perceptual grouping have long emphasized the role of feature similarity in binding image regions into coherent objects, from Gestalt principles to association fields and lateral connectivity in visual cortex [15, 35]. Our results show that self-supervised vision Transformers implicitly instantiate a similar computational principle: patches belonging to the same object develop highly similar feature representations, and the strength of this object-centric structure predicts how readily humans group those regions. The same-object advantage and distance efects we observe in human reaction times are captured by these models, suggesting that the patch-level similarity structure learned through selfsupervised training reflects key aspects of how the visual system organizes scenes into objects. This emergent alignment between self-supervised training objectives and human object perception supports the view that these objectives better approximate the organizational principles of perception than supervised classification. A gap nonetheless remains between the best-performing model and the human noise ceiling, leaving room for future models to close this distance.

Our findings also ofer guidance on model design. The consistent advantage of DINO-trained models over supervised counterparts with identical architectures demonstrates that the training objective is a major factor beyond architecture alone for object-centric representations. At the same time, the strong performance of DINOv3 ConvNext B relative to its supervised convolutional counterparts shows that convolutional architectures can also instantiate similar representations. However, the greater responsiveness of Transformer models to Gram matrix alignment suggests that the self-attention mechanism provides a particularly natural substrate for encoding pairwise patch relationships. It is worth noting that perfect object-centricity is not necessarily the goal: representations entirely organized around object boundaries would resemble semantic segmentation maps, losing the fine-grained feature structure that supports other visual tasks. The most behaviorally aligned models may therefore be those that strike an optimal balance between object-level organization and feature sensitivity.

The efect of Gram matrix alignment ofers the most mechanistically informative finding of our work. By fine-tuning supervised models to match the pairwise feature correlation structure of DINOv3 ViT B while maintaining ImageNet classification performance, we demonstrate that Gram matrix structure contributes to object-centricity and behavioral alignment. This finding converges with the design of DINOv3 itself, where Gram matrix anchoring was introduced as an explicit training signal to preserve and enhance the dense feature quality of the model [37]. These results show that feature correlation across patches plays an important role in human-like visual representations. Together, these findings suggest that the path toward more perceptually aligned vision models lies not in scaling classification objectives, but in explicitly shaping the similarity structure of learned representations to reflect the object-level organization of natural scenes.

## Acknowledgements

Research reported in this publication was supported in part by the National Institute of Neurological Disorders and Stroke of the National Institutes of Health under award numbers 1RF1NS128897 and 4R01NS128897. This work was also supported by Hankuk University of Foreign Studies Research Fund to SA. The contribution of M.Z. was supported by the National Research Foundation, Singapore under its NRFF award NRF-NRFF15-2023-0001 and a Startup Grant from Nanyang Technological University, Singapore. Additionally, we would like to thank the National Science Foundation for supporting this work through awards 2123920 and 2444540, and the National Institutes of Health through their award R01EY030669, to GZ. The content is solely the responsibility of the authors and does not necessarily reflect the views of the funding agencies.

## References

1. Adeli, H., Ahn, S., Zelinsky, G.J.: A brain-inspired object-based attention network for multiobject recognition and visual reasoning. Journal of Vision 23(5), 16–16 (2023)

2. Adeli, H., Minni, S., Kriegeskorte, N.: Transformer brain encoders explain human high-level visual responses. Advances in neural information processing systems 38, 56840–56871 (2025)

3. Ahn, S., Adeli, H., Zelinsky, G.J.: The attentive reconstruction of objects facilitates robust object recognition. PLOS Computational Biology 20(6), e1012159 (2024)

4. Amir, S., Gandelsman, Y., Bagon, S., Dekel, T.: Deep vit features as dense visual descriptors. arXiv preprint arXiv:2112.05814 (2021)

5. Biswas, T.K., Vacher, J., Molholm, S., Mamassian, P., Coen-Cagli, R.: Natural scene segmentation dynamics reveal iterative bayesian inference. bioRxiv pp. 2026– 01 (2026)

6. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021)

7. Chen, H., Venkatesh, R., Friedman, Y., Wu, J., Tenenbaum, J.B., Yamins, D.L., Bear, D.M.: Unsupervised segmentation in real-world images via spelke object inference. In: Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXIX. pp. 719–735. Springer (2022)

8. Chen, Y., Yan, Z., Zhou, C., Dai, B., Luo, A.F.: Vision transformers with selfdistilled registers. arXiv preprint arXiv:2505.21501 (2025)

9. Coen-Cagli, R., Mamassian, P.: Are we ready to tackle perceptual segmentation of natural scenes? Vision Research 240, 108749 (2026)

10. Darcet, T., Oquab, M., Mairal, J., Bojanowski, P.: Vision transformers need registers. arXiv preprint arXiv:2309.16588 (2023)

11. Dedieu, A., Rikhye, R.V., Lázaro-Gredilla, M., George, D.: Learning attentioncontrollable border-ownership for objectness inference and binding. bioRxiv pp. 2020–12 (2021)

12. Doshi, F.R., Fel, T., Konkle, T., Alvarez, G.: Visual anagrams reveal hidden diferences in holistic shape processing across vision models. arXiv preprint arXiv:2507.00493 (2025)

13. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

14. Egly, R., Driver, J., Rafal, R.D.: Shifting visual attention between objects and locations: evidence from normal and parietal lesion subjects. Journal of Experimental Psychology: General 123(2), 161 (1994)

15. Field, D.J., Hayes, A., Hess, R.F.: Contour integration by the human visual system: evidence for a local “association field”. Vision research 33(2), 173–193 (1993)

16. George, D., Lehrach, W., Kansky, K., Lázaro-Gredilla, M., Laan, C., Marthi, B., Lou, X., Meng, Z., Liu, Y., Wang, H., et al.: A generative vision model that trains with high data eficiency and breaks text-based captchas. Science 358(6368), eaag2612 (2017)

17. Gilbert, C.D., Li, W.: Top-down influences on visual processing. Nature reviews neuroscience 14(5), 350–363 (2013)

18. Gref, K., van Steenkiste, S., Schmidhuber, J.: On the binding problem in artificial neural networks. arXiv preprint arXiv:2012.05208 (2020)

19. Han, S., Wang, Z., Zhang, M.: Flow snapshot neurons in action: Deep neural networks generalize to biological motion perception. Advances in Neural Information Processing Systems 37, 53732–53763 (2024)

20. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16000–16009 (2022)

21. Hwang, E., Adeli, H., Guo, W., Luo, A., Kriegeskorte, N.: In silico mapping of visual categorical selectivity across the whole brain. Advances in Neural Information Processing Systems 38, 164602–164646 (2025)

22. Jeurissen, D., Self, M.W., Roelfsema, P.R.: Serial grouping of 2d-image regions with object-based attention in humans. Elife 5, e14320 (2016)

23. Kim, J., Linsley, D., Thakkar, K., Serre, T.: Disentangling neural mechanisms for perceptual grouping. arXiv preprint arXiv:1906.01558 (2019)

24. Korjoukov, I., Jeurissen, D., Kloosterman, N.A., Verhoeven, J.E., Scholte, H.S., Roelfsema, P.R.: The time course of perceptual grouping in natural scenes. Psychological Science 23(12), 1482–1489 (2012)

25. Lee, H.H., Chang, A.X.: Understanding pure clip guidance for voxel grid nerf models. arXiv preprint arXiv:2209.15172 (2022)

26. Li, T., Wen, Z., Song, L., Liu, J., Jing, Z., Lee, T.S.: From local cues to global percepts: Emergent gestalt organization in self-supervised vision models. arXiv preprint arXiv:2506.00718 (2025)

27. Li, Y., Salehi, S., Ungar, L., Kording, K.P.: Does object binding naturally emerge in large pretrained vision transformers? arXiv preprint arXiv:2510.24709 (2025)

28. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. pp. 740–755. Springer (2014)

29. Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11976–11986 (2022)

30. Lonnqvist, B., Scialom, E., Gokce, A., Merchant, Z., Herzog, M.H., Schrimpf, M.: Contour integration underlies human-like vision. arXiv preprint arXiv:2504.05253 (2025)

31. Mehrani, P., Tsotsos, J.K.: Self-attention in vision transformers performs perceptual grouping, not attention. arXiv preprint arXiv:2303.01542 (2023)

32. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Howes, R., Huang, P.Y., Xu, H., Sharma, V., Li, S.W., Galuba, W., Rabbat, M., Assran, M., Ballas, N., Synnaeve, G., Misra, I., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: Dinov2: Learning robust visual features without supervision (2023)

33. Papale, P., Williford, J.R., Balk, S., Roelfsema, P.R.: Modulatory feedback determines attentional object segmentation in a model of the ventral stream. PloS one 20(12), e0337087 (2025)

34. Peters, B., Kriegeskorte, N.: Capturing the objects of vision with neural networks. Nature Human Behaviour pp. 1–18 (2021)

35. Roelfsema, P.R.: Solving the binding problem: Assemblies form when neurons enhance their firing rate—they don’t need to oscillate or synchronize. Neuron 111(7), 1003–1019 (2023)

36. Siméoni, O., Puy, G., Vo, H.V., Roburin, S., Gidaris, S., Bursuc, A., Pérez, P., Marlet, R., Ponce, J.: Localizing objects with self-supervised transformers and no labels. arXiv preprint arXiv:2109.14279 (2021)

37. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al.: Dinov3. arXiv preprint arXiv:2508.10104 (2025)

38. Steiner, A., Kolesnikov, A., Zhai, X., Wightman, R., Uszkoreit, J., Beyer, L.: How to train your vit? data, augmentation, and regularization in vision transformers. arXiv preprint arXiv:2106.10270 (2021)

39. Toosi, T., Miller, K.D.: Generative inference unifies feedback processing for learning and perception in natural and artificial vision. bioRxiv pp. 2025–10 (2025)

40. Touvron, H., Cord, M., Jégou, H.: Deit iii: Revenge of the vit. In: European conference on computer vision. pp. 516–533. Springer (2022)

41. Treisman, A.: The binding problem. Current opinion in neurobiology 6(2), 171–178 (1996)

42. Vecera, S.P.: Toward a biased competition account of object-based segregation and attention. Brain and Mind 1(3), 353–384 (2000)

43. Wagemans, J., Elder, J.H., Kubovy, M., Palmer, S.E., Peterson, M.A., Singh, M., von der Heydt, R.: A century of gestalt psychology in visual perception: I. perceptual grouping and figure–ground organization. Psychological bulletin 138(6), 1172 (2012)

44. Wang, Y., Shen, X., Hu, S.X., Yuan, Y., Crowley, J.L., Vaufreydaz, D.: Selfsupervised transformers for unsupervised object discovery using normalized cut. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14543–14553 (2022)

45. Wang, Z., Han, S., Shou, M.Z., Zhang, M.: Unsupervised prior learning: Discovering categorical pose priors from videos (2024)

46. Wang, Z., Shou, M.Z., Zhang, M.: Object-centric learning with cyclic walks between parts and whole. Advances in Neural Information Processing Systems 36, 9388– 9408 (2023)

## 6 Supplementary Material

## 6.1 Behavioral Analyses

Subject-level Accuracy, Sensitivity, and Criterion Accuracy is consistent across subjects (mean=0.90, SD=0.05, no subject below 0.78), confirming that the task was well understood and performed reliably. d-prime is high across subjects (mean=2.78, SD=0.65), indicating strong perceptual sensitivity to the same/diferent distinction. The criterion is tightly centered around zero (mean=−0.03, SD=0.17), confirming no systematic response bias.

Accuracy by Condition Accuracy is consistently high across all conditions (same-close: 0.92, same-far: 0.90, dif-close: 0.89, dif-far: 0.91), with no condition showing floor or ceiling efects. The small variation across conditions is consistent with the same-object advantage: same-close is the easiest condition and dif-close is the hardest, as close dots on diferent objects are the most confusable.

d-prime and Criterion by Distance Condition d-prime is constant across close and far conditions (close: 2.79±0.67, far: 2.84±0.73), confirming equal sensitivity to the same/diferent distinction regardless of dot separation. The criterion is near zero in both conditions (close: −0.10±0.21, far: 0.05±0.20), indicating no systematic response bias in either condition. Together these results confirm that the speed-accuracy tradeof did not diferentially afect performance across conditions.

Reaction Time Distributions As a measure of the overlap between same and diferent RT distributions, we computed the AUC of an ROC analysis classifying same vs. diferent object trials from trial-averaged RTs (≈18 subjects per trial). The AUC is substantially above chance in both close (AUC=0.71, p<0.0001) and far (AUC=0.62, p<0.0001) conditions, confirming that the behavioral signal used for model comparison is well above chance. Single-trial AUC values are lower (close: AUC=0.59, far: AUC=0.55), consistent with well-known trial-to-trial RT variability. The same-object advantage is isotropic across all eight directions (std of mean RT: 14.7 ms for same, 24.6 ms for diferent).

## 6.2 Model Implementation Details

MLP Readout Architecture and Training For both the grouping and RT prediction tasks, the MLP readout takes as input the concatenation of two patch feature vectors, one from the central dot location and one from the peripheral dot location. The input dimension is therefore 2 × 768 for Transformer models and 2 × 1024 for ConvNext models. The MLP consists of two linear layers with ReLU activation, with hidden dimension equal to the feature dimension, followed by a final output layer.

For the grouping task, the output layer produces a two-class prediction (same or diferent object), trained with CrossEntropy loss. Training uses the Adam optimizer with learning rate $5 \times 1 0 ^ { - 4 }$ , a learning rate drop at epoch 15, and 30 epochs with gradient accumulation over 32 steps.

For the RT prediction task, the output layer produces a scalar prediction of the z-scored mean reaction time, where targets are normalized using the mean and standard deviation computed from the training set only. Training is performed with SmoothL1 loss using the Adam optimizer with learning rate $1 \times 1 0 ^ { - 5 }$ 2 a learning rate drop at epoch 40, and 80 epochs with gradient accumulation over 8 steps.

Gram Alignment Training The Gram-aligned models are produced by finetuning only the last block of each supervised model on ImageNet classification while simultaneously distilling the Gram matrix structure from DINOv3 ViT B. The training objective combines two losses: a standard CrossEntropy classification loss (weight 1) and a Gram MSE loss (weight 10), where the Gram loss minimizes the MSE between the pairwise cosine similarity matrix of the student’s last block features and those of the DINOv3 ViT B teacher. The higher weight on the Gram loss is chosen to make the two loss values comparable in scale during training. Training uses the Adam optimizer with learning rate $5 \times 1 0 ^ { - 4 }$

The Gram loss results in no change in ImageNet top-1 accuracy relative to fine-tuned baselines trained with the same procedure but without the Gram loss (IN1K ViT B: 0.77, IN21k ViT B: 0.78, IN1K ConvNext B: 0.74, IN21k ConvNext B: 0.76), confirming that improvements in object-centricity and behavioral alignment are not a consequence of altered classification performance.

## 6.3 Supplementary Results

Linear Readout Baseline To verify that behavioral alignment results are not an artifact of MLP capacity, we trained a linear readout for RT prediction across all 9 models using the same nested cross-validation procedure. The model ranking is fully preserved and the correlation between object-centric AUC and behavioral alignment remains high (Spearman r=0.917, p=0.0005), confirming that backbone object-centricity rather than readout capacity drives the result.

CLS Token Distillation Baseline To isolate Gram structure as the active ingredient in the Gram alignment experiments, we fine-tuned IN21k ViT B and IN1K ViT B using CLS token distillation from the same DINOv3 teacher instead of Gram distillation. The CLS loss minimizes $1 - \cos ( \mathbf { z } _ { s } , \mathbf { z } _ { t } )$ , where $\mathbf { z } _ { s }$ and $\mathbf { z } _ { t }$ are the student and teacher CLS token embeddings, driving the student’s global representation to align with the teacher’s without imposing any patch-level correlation structure. CLS distillation shows no improvement in behavioral alignment over the base models (IN21k ViT B: 0.59→0.59, IN1K ViT B: 0.59→0.58), while Gram distillation shows substantial improvement (0.59→0.72, 0.59→0.70), directly isolating patch-level correlation structure as the active ingredient.

A  
![](images/827cbb8954164722101d5318c2fbe53dcfdbef8b08cd03c8a9acc3461e113f63.jpg)

B  
![](images/65051fcb9fea1bc200dda53caff99300b1c770d57f6395ce1e458d342c1b4f13.jpg)

C  
![](images/495f7f8a8316599e735d547b65617b9b147a9046f2bb31acf76eeff2025c94e1.jpg)

D  
![](images/b676f1b6096f15f275b8821a79f37dc0b5accd5e99eb8b389627d68794fec2f0.jpg)  
Fig. 8: Scatter plots of model-predicted vs. human mean RTs for the best-performing model (DINOv3 ViT B), broken down by condition. Model predictions are in z-scored units (normalized using training-set statistics only); human RT is in milliseconds. Each point represents one trial (255 per condition). The positive relationship between model predictions and human RTs is significant in all four conditions independently (sameclose: $r { = } 0 . 4 3$ , same-far: r=0.37, dif-close: $r { = } 0 . 3 5$ , dif-far: $r { = } 0 . 3 2$ , all $\mathrm { p { < } 0 . 0 0 0 1 ) }$ , confirming that behavioral alignment is not driven by a single condition, near-constant predictions, or outliers.

Table 2: Spearman r between model-predicted and human mean RTs by condition. All $\mathrm { p { < } 0 . 0 1 }$

<table><tr><td>Condition</td><td>DINOv3 ViT B</td><td>IN1K ConvNext B</td></tr><tr><td>Same-close</td><td>0.43</td><td>0.33</td></tr><tr><td>Same-far</td><td>0.37</td><td>0.28</td></tr><tr><td>Diff-close</td><td>0.35</td><td>0.20</td></tr><tr><td>Diff-far</td><td>0.32</td><td>0.20</td></tr></table>

Model-Human Condition-Split Correlations Figure 8 shows scatter plots of model-predicted vs. human mean RTs for each of the four conditions separately, for the best-performing model (DINOv3 ViT B). Model-human correlations are significant in all four conditions (same-close: $r { = } 0 . 4 3$ , same-far: $r { = } 0 . 3 7$ dif-close: $r { = } 0 . 3 5$ , dif-far: $r { = } 0 . 3 2$ , all $\mathrm { p { < } 0 . 0 0 0 1 ) }$ ), confirming that the behavioral alignment is not driven by a single condition, near-constant predictions, or a small number of outliers. Table 2 reports condition-split correlations for both the best and worst performing models.

Raw and Normalized Spearman Correlations Table 3 reports the raw and noise-ceiling-normalized Spearman correlations between model-predicted and human mean RTs for all 9 models. The noise ceiling is estimated by splitting subjects into two equal halves 20 times and averaging the Spearman correlation between the two half-means across splits (mean=0.42, std=0.037). For each model, the Spearman correlation between model predictions and each of the 20 subject split means is computed, averaged across splits, and divided by the noise ceiling once as a final normalization step.

Table 3: Raw and normalized Spearman r for all models. Noise ceiling = 0.42.

<table><tr><td>Model</td><td>Raw r</td><td>Norm. r</td></tr><tr><td>DINOv3 ViT B</td><td>0.348</td><td>0.83</td></tr><tr><td>DINOv2 ViT B</td><td>0.324</td><td>0.77</td></tr><tr><td>DINOv3 ConvNext B</td><td>0.321</td><td>0.76</td></tr><tr><td>MAE ViT B</td><td>0.313</td><td>0.74</td></tr><tr><td>DINO ViT B</td><td>0.298</td><td>0.71</td></tr><tr><td>IN21k ViT B</td><td>0.247</td><td>0.59</td></tr><tr><td>IN1K ViT B</td><td>0.249</td><td>0.59</td></tr><tr><td>IN21k ConvNext B</td><td>0.235</td><td>0.56</td></tr><tr><td>IN1K ConvNext B</td><td>0.197</td><td>0.47</td></tr></table>