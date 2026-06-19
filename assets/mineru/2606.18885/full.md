# LARE: Low-Attention Region Encoding for Text–Image Retrieval

Abdulmalik Alquwayfili 1 Faisal Almeshal 1 Jumanah Almajnouni 1 Leena Alotaibi 1 Faisal Alhajari 1 Mohammed Alkhrashi 1 Alreem Almuhrij 1 Abdullah Aldwyish 1 Raied Aljadaany 1 Huda Alamri 1 Muhammad Kamran J. Khan 1

## Abstract

Image retrieval in crowded scenes is particularly challenging due to the salience bias of conventional visual encoders, which tend to focus on dominant objects while neglecting low-attention regions that are often crucial for fine-grained retrieval. We propose LARE1 (Low-Attention Region Encoding), a framework that explicitly models these overlooked regions. LARE adopts a dual-encoding strategy that encodes low-attention regions of an image and the full image in parallel, leading to more diverse and informative image embeddings. To evaluate image retrieval performance in challenging crowded scenes, we introduce Dense-Set2, a challenging subset derived from COCO and Flickr30K. In this subset, images are re-captioned to provide richer descriptions of low-attention or previously overlooked regions. This dataset highlights the limitations of existing retrieval models and enables a more rigorous evaluation under densely crowded scene conditions. Experimental results demonstrate that the proposed framework improves retrieval performance by preserving subtle, non-dominant visual cues within the shared latent space.

## 1. Introduction

Text-to-image retrieval retrieves images from large collections that best match a natural-language query. This capability is central to many real-world applications, including multimedia search engines, content recommendation systems, digital asset management, and large-scale visual indexing for web platforms. More broadly, cross-modal retrieval enables intuitive natural-language interaction with visual data and has become a key component in modern multimodal AI systems. (Radford et al., 2021; Jia et al., 2021; Yao et al., 2021; Gao et al., 2022; Li et al., 2021; Luo et al., 2022; Bain et al., 2021; Ma et al., 2022; Gorti et al., 2022).

![](images/ae0807e12060ebfca993181f04fd05e41430215118ef33fdd41a8c3490931a32.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Text Query"] --> B["a person near a stroller in a crowded street"]
  C["Image Database"] --> D["Grid of images"]
  D --> E["LARE(CLIP)"]
  D --> F["CLIP"]
```
</details>

Figure 1. Fine-grained retrieval in dense scenes. For the query “a person near a stroller in a crowded street”, LARE retrieves results that preserve the stroller-related local cue, while CLIP tends to favor globally similar crowded scenes. Green checks indicate relevant matches; red crosses indicate mismatches.

Recent advances in large-scale vision–language pretraining have significantly improved cross-modal retrieval by learning shared embedding spaces in which images and text can be compared directly. Contrastive models such as CLIP (Radford et al., 2021) and ALIGN (Jia et al., 2021) learn aligned visual and textual representations using massive image–text datasets, enabling strong zero-shot transfer across many tasks without task-specific training. In these models, an image encoder and a text encoder project inputs from each modality into a common embedding space, and retrieval is performed by ranking the similarity between their representations. This paradigm has become the dominant approach for cross-modal retrieval and underlies many modern multimodal systems (Radford et al., 2021; Jia et al., 2021; Li et al., 2022; Zhai et al., 2023; Huang et al., 2021; Chen et al., 2020; Kim et al., 2021).

Despite their success, current vision-language encoders mainly rely on a global image embedding that summarizes the entire image into a single representation. Although effective for many queries, this representation often emphasizes the most visually salient objects or scene context while underrepresenting smaller or less prominent elements. As a result, retrieval models may overlook visually relevant cues that occupy only a small portion of the image. This limitation is particularly evident in dense scenes with many objects, where correct retrieval may depend on attributes or objects that are not dominant in the global representation. Previous work has shown that vision-language models can struggle to localize fine-grained visual evidence and often prioritize coarse scene semantics over detailed object-level information (Wang et al., 2023).

In this work, we address this limitation by recovering information from image regions that receive little attention in the global representation. Our key observation is that transformer-based vision encoders implicitly encode spatial attention signals that reveal which regions contribute less to the final embedding. Rather than relying solely on the global representation, we exploit these signals to identify under-attended regions that may contain discriminative visual cues relevant to the query.

We propose Low-Attention Region Encoding (LARE), a training-free framework that augments standard dualencoder retrieval models with region-level evidence. Given an input image, LARE extracts low-attention regions from the encoder’s attention maps and re-encodes them to complement the global image embedding. During retrieval, the similarity between the text query and both global and regional representations is evaluated using a confidence-gated scoring mechanism.

To evaluate retrieval under challenging conditions, we introduce Dense-Set, a curated subset of COCO (Lin et al., 2014) and Flickr30K (Young et al., 2014) that emphasizes crowded scenes and rare objects. The dataset contains images with many detected objects and at least one rare object instance, along with re-captioned descriptions that highlight these underrepresented elements.

Experiments show that LARE consistently improves retrieval performance in dense scenes while preserving the ranking behavior of the original encoder on standard benchmarks, without requiring additional training, parameters, or architectural modifications.

Our contributions can be summarized as follows:

• We propose LARE, a training-free retrieval framework that augments global image embeddings with regionlevel representations extracted from low-attention areas.  
• We introduce Dense-Set, a curated benchmark de-

signed to evaluate retrieval performance in crowded scenes containing rare or visually subordinate objects.

• We conduct extensive experiments and ablation studies demonstrating consistent improvements on dense retrieval benchmarks across multiple backbone encoders while preserving performance on standard datasets.

The remainder of the paper is organized as follows. Section 2 reviews related work. Section 3 introduces the Dense-Set and its construction pipeline. Section 4 presents the proposed LARE retrieval framework. Section 5 reports experimental results and analysis on both standard benchmarks and Dense-Set. Finally, Section 6 concludes the paper.

## 2. Related Work

This work is related to research on text-to-image retrieval using vision–language models, methods for fine-grained image–text alignment, and approaches to retrieval in dense, visually complex scenes.

## 2.1. Text-to-Image Retrieval

Text-to-image retrieval aims to retrieve images that match a natural language query, and it is a fundamental task in vision–language understanding (Radford et al., 2021; Jia et al., 2021; Li et al., 2022; Zhai et al., 2023; Huang et al., 2021; Chen et al., 2020; Kim et al., 2021). Early approaches learned joint embedding spaces using convolutional neural networks for visual encoding and recurrent networks for text representation (Donahue et al., 2014; Sharif Razavian et al., 2014). More recently, large-scale vision–language pretraining has significantly improved retrieval performance by leveraging massive collections of image–text pairs (Radford et al., 2021; Li et al., 2022; Zhan et al., 2025).

Dual-encoder architectures have become the dominant paradigm for this task. Models such as CLIP and ALIGN learn aligned image and text representations using contrastive learning over large-scale datasets, enabling strong zero-shot retrieval performance across multiple benchmarks (Radford et al., 2021; Jia et al., 2021). In these models, the image and text encoders independently project each modality into a shared embedding space, allowing efficient similarity computation and scalable retrieval. Subsequent works have further improved representation quality and training efficiency. For example, BLIP introduces bootstrapped caption generation to enhance multimodal representation learning (Li et al., 2022), while SigLIP replaces the traditional softmax contrastive loss with a sigmoid loss to improve scalability and training stability (Zhai et al., 2023).

Despite their strong performance, dual-encoder retrieval models typically rely on a global image embedding that summarizes the entire image into a single vector. While effective for many queries, such representations may underrepresent localized visual evidence when relevant objects occupy small or visually subordinate regions within the image.

![](images/064e6f5a2e5f1d16db6c04bc551857ca4aae17a0a7605e75f50050eb315497b7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["Input"] --> B["Object Detection (YOLO)"]
  B --> C["Dense Image Detection"]
  C --> D["person&quot;, &quot;chair&quot;, &quot;apple&quot;, &quot;ie&quot;, &quot;bottle&quot;, &quot;cup"]
  D --> E["Rare Class Selection"]
  E --> F["dining table&quot;, &quot;spoon&quot;, &quot;apple&quot;, &quot;cup"]
  F --> G["Prominence Filter"]
  G --> H["[PROMPT"]]
  H --> I["Describe this scene. There is a {rare_obj} visible"]
  G --> J["Rare classes: [&quot;spoon&quot;, &quot;apple&quot;, &quot;cup&quot;"]
  J --> K["BUP-2"]
  K --> L["Caption"]
  L --> M["A group of children having an apple at their desks"]
    style A fill:#f9f,stroke:#333
    style L fill:#ccf,stroke:#333
```
</details>

Figure 2. Dense-Set curation pipeline. We first detect objects with YOLO and rank images by total object count, retaining the top 10% as the High-Density Subset (dense candidate pool). We then apply rare-class filtering and keep images containing at least one single-instance class to form the final Dense-Set.

## 2.2. Fine-Grained Vision–Language Alignment

To address the limitations of global representations, several works explore fine-grained alignment between image regions and textual tokens. FILIP introduces a late-interaction mechanism that computes token-level similarity between image patches and textual tokens, enabling finer-grained crossmodal alignment while maintaining efficient inference (Yao et al., 2021). PyramidCLIP further improves alignment by introducing hierarchical feature representations that capture visual semantics at multiple levels of granularity (Gao et al., 2022).

Another line of work focuses on region-level representations. RegionCLIP extends contrastive language-image pretraining to region-based representations, enabling alignment between textual concepts and localized image regions (Zhong et al., 2022). More recently, methods such as ELIP introduce lightweight text-guided visual prompts that condition the image encoder on the query, improving retrieval performance without retraining large backbone models (Zhan et al., 2025).

While these approaches improve fine-grained alignment, many require additional training, architectural modifications, or query-conditioned representations, thereby increasing computational complexity. Unlike prior approaches that require retraining or query-conditioned encoders, our method augments global representations with region-level embeddings extracted at inference time, thereby improving retrieval in dense scenes while preserving the efficiency of dual-encoder architectures.

## 2.3. Retrieval in Dense and Complex Scenes

Text-to-image retrieval becomes particularly challenging in crowded scenes and long-tail object distributions, where relevant evidence may correspond to small or rare objects. Datasets such as COCO and Flickr30K contain complex scenes with multiple objects, occlusions, and visual clutter, making global image representations insufficient for capturing fine-grained attributes (Lin et al., 2014; Plummer et al., 2015). In such scenarios, correct retrieval may depend on localized visual cues that are not dominant within the scene. To address this, prior work has explored combining global and local representations, for example by leveraging local features to refine global similarity rankings (Aiger et al., 2025).

Recent studies have also shown that attention maps produced by vision transformers encode implicit spatial signals that indicate which regions contribute most to the final representation. These signals have been used for interpretability and weak localization tasks, revealing how visual transformers allocate attention across spatial regions. Concurrent work explores a related inverse-attention idea for video retrieval (Alhajari et al., 2026), fusing regional and global scores via a hard maximum; in contrast, LARE targets image retrieval and introduces confidence-gated fusion together with the curated Dense-Set benchmark.

Motivated by these observations, our work leverages the internal attention structure of vision transformers to identify low-attention regions that may contain underrepresented visual evidence.

## 3. Dense-Set Dataset

To evaluate the proposed methodology, we construct Dense-Set, a curated benchmark of visually dense scenes. The goal is to create a challenging evaluation subset containing crowded images with multiple object instances and underrepresented classes. To this end, we develop an automated pipeline, illustrated in Figure 2. In the following subsections, we describe the main stages of this pipeline.

## 3.1. Dense-Set Construction

This stage of the pipeline, illustrated in the first half of Figure 2, focuses on identifying densely populated images that contain underrepresented object instances. We begin by processing all images from the COCO (Lin et al., 2014) and Flickr30K (Young et al., 2014) test splits using a YOLO object detector (Bochkovskiy et al., 2020). For each image, the detector outputs bounding boxes and class predictions, from which we compute three image-level statistics: (i) the total number of detected objects, (ii) the number of unique object categories, and (iii) per-class instance frequencies.

Table 1. Examples from Dense-Set with rewritten captions highlighting rare or low-attention objects for more challenging dense-scene evaluation.

<table><tr><td>Dataset</td><td>COCO</td><td>COCO</td><td>Flickr30K</td><td>Flickr30K</td></tr><tr><td>Image</td><td><img src="images/80811c0b0ea37d151983a57e0c25f59a59805eff599eb67317605b9ec8155452.jpg"/></td><td><img src="images/3995825e6e0bdadad5ad9a5d2f4d680b9e045e281793f41f1cf45c95e59b435d.jpg"/></td><td><img src="images/3ba71f4d24073111560695e684dec5117e002d1875dcacd4f21030b3ba7d5682.jpg"/></td><td><img src="images/75e1590913fa8b3b046d1147c2c828f625ac5fff91ad930e15de81f356144ee5.jpg"/></td></tr><tr><td>Original Caption</td><td>Car driving down a road behind a lot of sheep.</td><td>A cat lying down on a desk by a computer keyboard.</td><td>A group of men wearing sweaters are dining in a hall.</td><td>A crowd of people is standing outside next to a street.</td></tr><tr><td>Rare Class</td><td>Dog</td><td>Sports ball</td><td>Fork</td><td>Handbag</td></tr><tr><td>Rewritten Caption</td><td>A photo of a dog standing on the side of a road with a herd of sheep.</td><td>A sports ball sitting on top of a desk.</td><td>A fork placed in the middle of a group of men sitting at a table.</td><td>A handbag on the ground in front of a crowd of people.</td></tr></table>

Table 2. Stage-wise statistics of Dense-Set curation for COCO and Flickr30K

<table><tr><td>Dataset</td><td>Split</td><td># Images</td><td>Avg. Objects</td><td>Avg. # Classes</td></tr><tr><td rowspan="3">COCO</td><td>Original Test Set</td><td>40,504</td><td>6.71</td><td>2.85</td></tr><tr><td>High-Density Subset</td><td>4,050</td><td>21.63</td><td>4.82</td></tr><tr><td>Dense-Set</td><td>3,089</td><td>21.63</td><td>5.47</td></tr><tr><td rowspan="3">Flickr30K</td><td>Original Test Set</td><td>31,783</td><td>6.73</td><td>2.48</td></tr><tr><td>High-Density Subset</td><td>3,178</td><td>19.40</td><td>4.38</td></tr><tr><td>Dense-Set</td><td>2,477</td><td>19.55</td><td>4.85</td></tr></table>

To construct the dense candidate pool, images are ranked in descending order by total object count, and the top 10% are selected. This step favors crowded scenes with high object density and diverse visual content. Within this dense candidate set, we identify rare classes at the image level, defined as object categories that appear exactly once in a given image. In crowded scenes, such single-instance categories often correspond to small or low-salience objects that are easily overlooked by global representations.

The final Dense-Set subset consists of images that (1) belong to the dense candidate pool and (2) contain at least one rare-class instance. This selection strategy yields a benchmark with significantly higher object density and class diversity than the original splits, thereby creating a more challenging setting for fine-grained text-to-image retrieval.

Table 2 summarizes the three stages shown in Figure 2: the Original Test Set, the High-Density Subset (top 10% by object count), and the final Dense-Set after rare-class filtering. For each stage, we report the number of images, the average number of detected objects per image, and the average number of object classes. The final curated Dense-Set contains images with substantially more objects and a broader set of object categories compared to the original splits. These characteristics make Dense-Set particularly suitable for evaluating retrieval models in visually dense environments, where important objects may appear in lowattention regions and are more likely to be overlooked by standard global representations.

## 3.2. Dense-Set Re-captioning

The second stage of the pipeline, illustrated in the second half of Figure 2, focuses on regenerating captions for the curated Dense-Set images. The goal of this re-captioning step is to produce more challenging textual descriptions that explicitly emphasize low-attention regions, i.e., rare-class instances. In contrast, the original dataset captions typically describe the dominant scene context and often overlook small or underrepresented objects. For each image in Dense-Set, we first filter rare-class detections whose bounding boxes occupy a large fraction of the image area (e.g., greater than 15%). Such instances are likely to correspond to visually dominant objects rather than genuinely low-salience elements. This filtering ensures that the captioning process focuses on secondary or background objects that are more likely to be ignored by global visual representations. The rare-class-filtered labels are then used as guidance for a vision-language model (BLIP-2). Specifically, we prompt the model to use class-aware templates (e.g., “a photo of a [class]”) to encourage explicit mention of these underrepresented objects in the generated description. The model takes both the image and the guided prompt as input and outputs a single caption in the standard COCO format. By shifting the caption focus from general scene-level descriptions to fine-grained object-level details, this re-captioning process produces a more demanding evaluation setting for text-to-image retrieval in dense scenes.

![](images/f1b75cc701f2e62580fe4cec27475d04793887300c05fd658bcb37597a5f03f2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Find me a person wearing a backpack"] --> B["Text Encoder"]
  B --> C["Low-Attention Region Detection"]
  C --> D["Attention Map"]
  D --> E["Invert + Cluster"]
  E --> F["Inverse Attention Map"]
  F --> G["Regional Embeddings zᵢ"]
  G --> H["Confidence-Gated Scoring"]
  H --> I["Global Image Embedding zg"]
  J["Image"] --> K["Image Encoder"]
  K --> L["Global Image Embedding zg"]
  M["Text Query"] --> N["Text Embedding zₜ"]
  N --> O["Regional Embeddings zᵢ"]
  P["&quot;Sim(z₁,z₂)"] --> Q["Crop¹,Crop²,Crop³"]
  Q --> R["Global"]
  S["&quot;Sim(z₁,z₁)"] --> T["Global"]
```
</details>

Figure 3. LARE pipeline: A single forward pass produces both a global image embedding and a spatial attention map. Inverting the attention map highlights under-attended regions, which are clustered into candidate crops and then re-encoded independently. A confidence gate determines whether regional evidence should be used to adjust the final retrieval score.

Examples of the curated Dense-Set and their rewritten captions are shown in Table 1. For each image from COCO and Flickr30K, we identify a rare or low-attention class and rewrite the original caption to explicitly describe the overlooked object. This shifts the textual focus from general scene context to fine-grained object-level details, thereby making dense-scene retrieval evaluation more challenging.

## 4. Methodology

We introduce Low-Attention Region Encoding (LARE), a training-free framework that enhances visual semantic search by recovering information from regions typically underemphasized by standard vision encoders. Our approach follows a three-stage pipeline illustrated in Figure 3: (1) Low-Attention Region Detection, (2) Regional Encoding, and (3) Confidence-Gated Scoring.

## 4.1. Low-Attention Region Detection

The first stage identifies non-dominant visual cues by analyzing the internal self-attention signals of a frozen vision encoder. Given an input image I, we extract the self-attention tensor from an intermediate layer ℓ. For each head h, let $\mathbf { A } ^ { ( h ) } \in \mathbb { R } ^ { H W \times H W }$ denote the patch-to-patch attention matrix.

We quantify the amount of attention each patch i receives from all other patches by calculating the column-wise sum:

$$
a _ {i} ^ {(h)} = \sum_ {j} A _ {j, i} ^ {(h)}, \quad i \in \{1, \dots , H W \} \tag {1}
$$

Each map $a ^ { ( h ) }$ is reshaped to a spatial grid, min-max normalized, and averaged across the top-k heads (selected by spatial variance) to form a mean attention map A¯ . We then derive an inverse-attention map:

$$
\mathbf {M} = \mathbf {1} - \bar {\mathbf {A}} \tag {2}
$$

where high values in M highlight patches that consistently receive minimal attention. We apply a sliding window and non-maximum suppression (NMS) on M to generate a set of N candidate regions, $\mathcal { R } = \{ r _ { 1 } , \ldots , r _ { N } \}$ . We analyze sensitivity to N in Appendix A.1, Figure 5.

## 4.2. Regional Encoding

The second stage encodes the image regions generated in the previous stage.

$$
\mathbf {z} _ {i} = f _ {v} (r _ {i}), \quad i = 1, \dots , N \tag {3}
$$

This produces a set of regional feature vectors $\left\{ \mathbf { z } _ { 1 } , \dotsc , \mathbf { z } _ { N } \right\}$ . Because the encoder weights are shared, these regional embeddings reside in the same feature space as the global representation, allowing for direct comparison with text embeddings without additional training.

## 4.3. Confidence-Gated Scoring

Finally, we integrate the global and regional information to compute a comprehensive retrieval score. While prior work fuses regional and global signals via a hard maximum (Alhajari et al., 2026), this can amplify spurious regional matches when the global embedding is already well-aligned. We instead introduce a confidence-gated fusion that defers to the global score when the model is confident, and only blends in regional evidence otherwise. First, we obtain the global image embedding ${ \bf z } _ { g } = f _ { v } ( I )$ and the text query embedding ${ \mathbf z } _ { t } = f _ { t } ( T )$ . We define the global similarity as $s _ { g } = \sin ( \mathbf { z } _ { t } , \mathbf { z } _ { g } )$ and the strongest regional match as $s _ { r } = \operatorname* { m a x } _ { i } \sin ( \mathbf { z } _ { t } , \mathbf { z } _ { i } )$ . To ensure robustness against regional noise, we gate the contribution of the regions based on the model’s confidence in the global match. If $s _ { g }$ exceeds a confidence threshold $\tau ,$ the final score remains $S = s _ { g }$ . If $s _ { g } < \tau$ and a region outperforms the global match $( s _ { r } > s _ { g } )$ , we interpolate toward the regional score:

Table 3. Zero-shot retrieval performance of baseline models and LARE pipeline on COCO and Flickr30K, along with their Dense-Set variants.

<table><tr><td rowspan="2">Model</td><td rowspan="2">ViT</td><td colspan="3">COCO</td><td colspan="3">Flickr30K</td><td colspan="3">COCO-Dense</td><td colspan="3">Flickr30K-Dense</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>CLIP (Radford et al., 2021)</td><td>L/14</td><td>36.10</td><td>61.10</td><td>71.44</td><td>65.00</td><td>88.00</td><td>92.62</td><td>17.79</td><td>35.85</td><td>45.11</td><td>3.48</td><td>11.97</td><td>16.33</td></tr><tr><td>SigLIP (Zhai et al., 2023)</td><td>So/14</td><td>54.24</td><td>76.78</td><td>84.21</td><td>82.94</td><td>96.08</td><td>98.00</td><td>26.61</td><td>46.31</td><td>55.22</td><td>5.05</td><td>15.50</td><td>20.96</td></tr><tr><td>SigLIP 2 (Tschannen et al., 2025)</td><td>So/16</td><td>56.55</td><td>78.75</td><td>85.95</td><td>83.72</td><td>96.34</td><td>98.32</td><td>27.56</td><td>47.56</td><td>56.73</td><td>5.12</td><td>16.47</td><td>21.80</td></tr><tr><td>LARE (CLIP)</td><td>L/14</td><td>36.10</td><td>61.10</td><td>71.44</td><td>65.00</td><td>88.00</td><td>92.62</td><td>22.97</td><td>42.10</td><td>52.03</td><td>9.73</td><td>16.63</td><td>20.40</td></tr><tr><td>LARE (SigLIP)</td><td>So/14</td><td>54.26</td><td>76.80</td><td>84.24</td><td>82.94</td><td>96.12</td><td>98.00</td><td>29.94</td><td>50.17</td><td>59.26</td><td>12.33</td><td>19.87</td><td>24.10</td></tr><tr><td>LARE (SigLIP 2)</td><td>So/16</td><td>56.56</td><td>78.78</td><td>85.97</td><td>83.76</td><td>96.38</td><td>98.34</td><td>31.00</td><td>51.45</td><td>60.67</td><td>13.28</td><td>21.11</td><td>25.10</td></tr></table>

$$
\alpha = \min \bigl (2 (s _ {r} - s _ {g}), 0. 5 \bigr), \qquad S = (1 - \alpha) s _ {g} + \alpha s _ {r} (4)
$$

where $\tau = 0 . 2 5$ . We analyze the sensitivity to $\tau$ in Appendix A.1, Figure 5. This fusion logic ensures that regional evidence effectively “rescues” the ranking when the global embedding is insufficient, particularly in dense scenes targeting non-salient objects.

## 5. Results and Analysis

We evaluate LARE in a zero-shot image retrieval setting, where no additional training or fine-tuning is performed on the target benchmarks. Given a textual query, the task is to retrieve the most semantically aligned image from a candidate set. We compare the performance of LARE against several state-of-the-art vision–language retrieval models, including CLIP (Radford et al., 2021), SigLIP (Zhai et al., 2023), and SigLIP 2 (Tschannen et al., 2025). Evaluation is conducted on COCO (Lin et al., 2014) and Flickr30K (Young et al., 2014), as well as their Dense-Set variants designed to emphasize crowded scenes and rare objects. Performance is reported using Recall@K metrics (R@1, R@5, R@10).

## 5.1. Zero-Shot Retrieval Results

Performance on standard datasets: As shown in Table 3, the first two column groups (COCO and Flickr30K) report results on standard benchmark splits. On these datasets, LARE maintains performance comparable to the underlying backbone models, with differences being marginal across all Recall@K metrics. This near-zero change is by design rather than a lack of benefit: the confidence gate (Eq. 4) defers entirely to the global score whenever the global match is already confident, which holds for the large majority of standard-split queries whose captions describe dominant scene content. The intended behavior is therefore a no-regression guarantee on the common case, with region evidence activated only where the global embedding is insufficient. The fine-grained regime where this occurs is exactly what Dense-Set isolates, and the consistent gains there across three backbones indicate the benefit is a property of the encoder’s salience bias rather than an artifact of any single split.

Performance on Dense-Set: In contrast, the last two columns of Table 3 (COCO-Dense and Flickr30K-Dense) demonstrate substantial gains on the curated Dense-Set benchmarks. On COCO-Dense, LARE improves R@1 by +5.18 points (29% relative improvement) for CLIP, +3.33 points (12.5%) for SigLIP, and +3.44 points (12.5%) for SigLIP 2. On Flickr30K-Dense, the gains are even more pronounced: +6.25 points (180% relative improvement) for CLIP, +7.28 points (144% relative improvement) for SigLIP, and +8.16 points (159% relative improvement) for SigLIP 2.

These results show that while LARE preserves performance on standard benchmarks, it delivers large and consistent improvements in dense-scene retrieval scenarios, particularly where relevant objects are rare, small, or visually subordinate.

Cross-Backbone Generalization: The consistent improvement across diverse architectures (from CLIP to SigLIP 2) demonstrates that LARE operates as a general, plug-and-play inference refinement. It complements even the strongest modern encoders, suggesting that ”salience bias” is a fundamental characteristic of global embeddings that persists despite scaling.

## 5.2. Qualitative Results

Figure 4 presents qualitative comparisons between the baseline encoder (SigLIP) and LARE on dense retrieval queries from COCO-Dense (Columns 1–2) and Flickr30K-Dense (Columns 3–4). For each query, the top-5 retrieved images are shown, and the ground-truth image is highlighted with a dashed box.

Query: “A cyclist wearing a backpack next to a train station”

Query: “A person carrying a red bag in a busy outdoor market”

Rank  
Baseline  
LARE  
1  
![](images/12a01b1622b75da88b5e8f83a2adbe1d71c37008be6bf31fd5bdb38b817d2dfd.jpg)

<details>
<summary>natural_image</summary>

Street scene with a yellow and white toy car parked on a cobblestone street, surrounded by pedestrians and bystanders (no visible text or symbols)
</details>

![](images/c2ec26db32cec0d62c5edcb3a3b2827f38ac1abd2e0f2972060ca4a8c99c6062.jpg)

<details>
<summary>text_image</summary>

Street photo showing a public transit station with visible directional signs and people passing by
</details>

2  
![](images/b288aadfcef0634f539114634a67b0c0cd1b759ef75ffae2eb8aabf93b64269a.jpg)

<details>
<summary>natural_image</summary>

Street scene with a cyclist, a train, and cars (no visible text or symbols)
</details>

![](images/23d17b141dd1432c8820d9e7dc4ac70ca299d735b4a441b0f8418af3157af71a.jpg)

<details>
<summary>natural_image</summary>

Street scene with cyclists and a tram, no visible text or symbols
</details>

3  
![](images/cf7f24b51d4cfedfb13edb323eb58a01956361556524977b859704f7c9b4c1f3.jpg)

<details>
<summary>natural_image</summary>

Street scene with a person riding a bicycle, pedestrians, and a bus in the background (no visible text or symbols)
</details>

![](images/41be62273a14f582854a1d37e3d7b67087678e45b2e8bd53d3cf3bd154b516a3.jpg)

<details>
<summary>text_image</summary>

PETE'S 7
485
R.B-QUE
-1009
</details>

4  
![](images/0d1d04e68121c0fee067d60e4be1aa92e5cf4f2c749e684088a42c82b92e04c7.jpg)

<details>
<summary>natural_image</summary>

Row of bicycles parked on a city street with a yellow bus and building in the background (no visible text or symbols)
</details>

![](images/8f990860a60e7bcb134e7eafbcedd423003e9b40141546f07e05634e8941c432.jpg)

<details>
<summary>natural_image</summary>

Street view of a modern urban district with glass buildings, traffic lights, and pedestrians (no visible text or signage)
</details>

5  
![](images/d96336a0ecfff15804f4c36a57038dd14f022f82a60193a6e17da465c9c238dd.jpg)

<details>
<summary>text_image</summary>

Photo of passengers boarding a train with a sad face emoji, showing ride-hailing and luggage.
</details>

![](images/5f5815fb897ba12e87e9b790e0f4d1bfef4f3f33be039ad5f19b179e267ddf90.jpg)

<details>
<summary>text_image</summary>

Street photo showing a tram with visible signage and people on the side, including a green smiley face logo in the corner.
</details>

Baseline  
LARE  
![](images/76609b45640839132f7a3d6756cba2e75117bc8225df2ac31f338600eb0d699e.jpg)

<details>
<summary>text_image</summary>

Street photo of a bustling outdoor market with vendors, shoppers, and vendor stalls, featuring visible red umbrellas and a sad face emoji.
</details>

![](images/07af656296c663b79890778021967de2e9f6cf3ffd3ed970773f5ce23c9aa280.jpg)

<details>
<summary>text_image</summary>

Street photo of a market stall with visible Chinese signage and people browsing stalls
</details>

![](images/71f33d3104b22e41f5f481bfd1a30ce329b513c934a3426a2a41cc89b2a76da8.jpg)

<details>
<summary>text_image</summary>

Street photo showing pedestrians walking past a storefront with visible signage and a sad face emoji
</details>

![](images/6ffbb71f619f395d94159f099e817f64c04352c8b65170f7a490bb0525b966da.jpg)

<details>
<summary>text_image</summary>

Street photo of a bustling outdoor market with vendors and shoppers, featuring visible store signboards and a smiling face logo.
</details>

![](images/f1e254fe2f70fb0cc9a96588723eceb58d6f22c296e99c0da546e7db51219294.jpg)

<details>
<summary>text_image</summary>

Street photo showing people with luggage and a large pink bag, with a sad face emoji in the corner
</details>

![](images/ddd86f95e0f716f079d0f3b76983ae8db5d87f7e2874a0250b833a411a286a87.jpg)

<details>
<summary>text_image</summary>

Street market scene with vendors and customers, featuring a green smiley face icon in the corner
</details>

![](images/87e502278a3dd702d35a66fa5b1daf0568cfe299733adb6ca7b72f2a5dfc67d9.jpg)

<details>
<summary>text_image</summary>

Street scene with visible store signboards and a red banner with a sad face icon
</details>

![](images/b947d57365cb8911fc6b31eb942dfc3867e5143c9354c7df240d53a489c3e646.jpg)

<details>
<summary>text_image</summary>

Street photo of a busy commercial area with visible store signs including 'JIR Hill' and pedestrians, featuring a green smiley face logo in the corner.
</details>

![](images/4a4edc605460c4ec78510765b7d31e9ae15eb1374070ebf534f46f91ec2f1ca1.jpg)

<details>
<summary>text_image</summary>

Street photo of a bustling indoor market with visible store signboards and a large sad face emoji overlay
</details>

![](images/063a2e73bc42919f9607ea07506bbb20ee879fa981af083cad11d8cbc5cd9cc8.jpg)

<details>
<summary>text_image</summary>

Street photo showing pedestrians and a woman with a green bounding box, including a smiley face icon in the top-left corner.
</details>

Figure 4. Qualitative comparison between Baseline and LARE on COCO-Dense (Cols. 1–2) and Flickr30K-Dense (Cols. 3–4). Top-5 retrieval results are shown; ground-truth is highlighted. LARE improves ranking by leveraging fine-grained, localized cues missed by the baseline.

In the first example (COCO-Dense), the query “A cyclist wearing a backpack next to a train station” requires recognition of the backpack in addition to the cyclist and station context. The baseline ranks a generic cyclist at Rank 1, failing to capture the backpack attribute, while the correct image appears lower in the ranking. In contrast, LARE identifies the backpack as a localized discriminative cue and promotes the correct image to the top position for retrieval.

In the second example (Flickr30K-Dense), the query “A person carrying a red bag in a busy outdoor market” hinges on detecting the red bag within a crowded scene. The baseline retrieves general market scenes that align with the global context but miss the specific attribute described in the query. LARE successfully retrieves the image containing the person with the red bag at Rank 1, indicating improved alignment with fine-grained details.

These examples illustrate that improvements arise when relevant evidence is spatially localized and visually subordinate within the scene. By incorporating region-level representations, LARE resolves ambiguities that global embeddings alone fail to distinguish. When global similarity is already reliable, rankings remain unchanged, consistent with the confidence-gated design.

Table 4. Per-image and per-query cost of LARE on SigLIP 2. The overhead falls on offline index building; query latency is unchanged.

<table><tr><td></td><td>Baseline</td><td>LARE</td></tr><tr><td>Indexing (per image)</td><td>69 ms</td><td>434 ms</td></tr><tr><td>Retrieval (per query)</td><td>4.7 ms</td><td>5.0 ms</td></tr><tr><td>Storage (per image)</td><td>1 $\times$ </td><td>6 $\times$ </td></tr></table>

## 5.3. Comparison with Fine-Grained Methods

Fine-grained alignment methods such as FILIP (Yao et al., 2021), RegionCLIP (Zhong et al., 2022), and ELIP (Zhan et al., 2025) are not directly comparable to LARE, because each relies on training or query-time conditioning that a frozen, training-free pipeline does not provide. FILIP matches text tokens to image patches with a late-interaction score that it learns during pretraining; on a frozen encoder this score is not meaningful and retrieves at close to chance level, because the patch tokens are nearly orthogonal to the pooled embedding that the contrastive objective aligns with text. RegionCLIP retrains the encoder so that cropped regions align with text, whereas on a frozen encoder a tight crop drifts away from the contrastive space that retrieval depends on, while a larger crop that preserves context stays close to it. This is why LARE re-encodes spatially generous regions with the same frozen encoder rather than reusing patch tokens or tight crops. ELIP re-encodes each image conditioned on the query, which needs a trained prompting module and gives up the index-once property of large-scale retrieval, so it complements LARE rather than competing with it.

## 5.4. Inference Overhead

Because LARE encodes each image once globally and once per region, it raises the cost of building the retrieval index but not the cost of answering a query. Table 4 separates the two for SigLIP 2 on a single GPU. Indexing is about six times more expensive than the baseline, as each image now needs six encoder passes instead of one; this cost is paid once, offline, and is amortized over all future queries, since the regional embeddings are computed when the index is built and stored alongside the global embedding.

At query time nothing changes. A query is still a single text encoding followed by one similarity computation, and the confidence gate only adjusts the score of uncertain pairs, so per-query latency matches the baseline. The practical price of LARE is therefore extra storage and a one-time, parallelizable indexing step, not slower retrieval. When indexing cost is itself a concern, the regional passes can be deferred to a re-ranking stage that crops only the top candidates of each query, leaving the index unchanged. Accuracy also saturates around five regions and degrades gracefully with fewer, so this budget can be lowered when needed.

## 6. Conclusion

We presented LARE, a training-free augmentation for textto-image retrieval in crowded scenes. Our method mines low-attention regions from a frozen vision encoder, encodes these regions alongside the full image, and combines regional embeddings with the global image embedding at inference time. This simple test-time procedure improves retrieval on Dense-Set variants that emphasize subtle and occluded content. We also introduced Dense-Set, a challenging crowded-scene benchmark derived from COCO and Flickr30K, where images are re-captioned to emphasize low attended areas. By shifting the focus toward fine-grained object, Dense-Set reveals the limitations of existing retrieval models and provides a more rigorous evaluation setting for densely crowded scenes.

For future work, we plan to make region selection more query-aware so that only the most informative crops are encoded, reducing compute while preserving accuracy gains. We also aim to strengthen fine-grained text–image alignment through patch-level interactions in the spirit of FILIP (Yao et al., 2021). In addition, extending LARE to temporal retrieval settings is a promising next step, building on dualencoder video retrieval formulations such as CLIP4Clip and Frozen in Time (Luo et al., 2022; Bain et al., 2021).

## References

Aiger, D., Cao, B., Chen, K., and Araujo, A. Global-to-local or local-to-global? enhancing image retrieval with efficient local search and effective global re-ranking. arXiv preprint arXiv:2509.04351, 2025.

Alhajari, F., Alkhrashi, M. A., Almuhrij, A., Abuhimed, S., Aldossary, N., Aldwyish, A., Aljadaany, R., Alamri, H., and Khan, M. K. J. Look beyond saliency: Low-attention guided dual encoding for video semantic search. arXiv preprint arXiv:2605.06229, 2026.

Bain, M., Nagrani, A., Varol, G., and Zisserman, A. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1728–1738, 2021.

Bochkovskiy, A., Wang, C.-Y., and Liao, H.-Y. M. Yolov4:

Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934, 2020.  
Chen, Y.-C., Li, L., Yu, L., El Kholy, A., Ahmed, F., Gan, Z., Cheng, Y., and Liu, J. Uniter: Universal image-text representation learning. In European Conference on Computer Vision (ECCV), 2020.  
Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., and Jitsev, J. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2818–2829, 2023.  
Donahue, J., Jia, Y., Vinyals, O., Hoffman, J., Zhang, N., Tzeng, E., and Darrell, T. Decaf: A deep convolutional activation feature for generic visual recognition. In Proceedings of the 31st International Conference on Machine Learning (ICML), Bejing, China, 2014.  
Gao, Y., Liu, J., Xu, Z., Zhang, J., Li, K., Ji, R., and Shen, C. Pyramidclip: Hierarchical feature alignment for visionlanguage model pretraining. Advances in neural information processing systems, 35:35959–35970, 2022.  
Gorti, S. K., Vouitsis, N., Ma, J., Golestan, K., Volkovs, M., Garg, A., and Yu, G. X-pool: Cross-modal languagevideo attention for text-video retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10562–10571, 2022.  
Huang, Y., Wang, Y., and Tam, Y.-C. Uniter-based situated coreference resolution with rich multimodal input. arXiv preprint arXiv:2112.03521, 2021.  
Jia, C., Yang, Y., Xia, Y., Chen, Y.-T., Parekh, Z., Pham, H., Le, Q., Sung, Y.-H., Li, Z., and Duerig, T. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pp. 4904–4916. PMLR, 2021.  
Kim, W., Son, B., and Kim, I. Vilt: Vision-and-language transformer without convolution or region supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 5583–5594. PMLR, 2021.  
Li, J., Selvaraju, R., Gotmare, A., Joty, S., Xiong, C., and Hoi, S. C. H. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34: 9694–9705, 2021.  
Li, J., Li, D., Xiong, C., and Hoi, S. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pp. 12888–12900. PMLR, 2022.  
Lin, T.-Y., Maire, M., Belongie, S. J., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. Microsoft´ COCO: Common Objects in Context. In Proceedings of the 13th European Conference on Computer Vision (ECCV), Part V, volume 8693 of Lecture Notes in Computer Science, pp. 740–755, Zurich, Switzerland, 2014. ¨ Springer.  
Luo, H., Ji, L., Zhong, M., Chen, Y., Lei, W., Duan, N., and Li, T. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing, 508:293–304, 2022.  
Ma, M., Xu, J., Jiang, Y., Wang, Z., and Lu, H. X-clip: Endto-end multi-grained contrastive learning for video-text retrieval. In Proceedings of the 30th ACM International Conference on Multimedia (ACM MM), pp. 4366–4374, 2022.  
Plummer, B. A., Wang, L., Cervantes, C. M., Caicedo, J. C., Hockenmaier, J., and Lazebnik, S. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pp. 2641– 2649, 2015.  
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.  
Sharif Razavian, A., Azizpour, H., Sullivan, J., and Carlsson, S. Cnn features off-the-shelf: an astounding baseline for recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 806–813, 2014.  
Tschannen, M., Gritsenko, A., Wang, X., Naeem, M. F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., Henaff, O., Harmsen, J., ´ Steiner, A., and Zhai, X. Siglip 2: Multilingual visionlanguage encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.  
Wang, F., Mei, J., and Yuille, A. Sclip: Rethinking selfattention for dense vision-language inference. arXiv preprint arXiv:2312.01597, 2023.  
Yao, L., Huang, R., Hou, L., Lu, G., Niu, M., Xu, H., Liang, X., Li, Z., Jiang, X., and Xu, C. Filip: Fine-grained interactive language-image pre-training. arXiv preprint arXiv:2111.07783, 2021.  
Young, P., Lai, A., Hodosh, M., and Hockenmaier, J. From image descriptions to visual denotations: New similarity  
metrics for semantic inference over event descriptions. Transactions of the association for computational linguistics, 2:67–78, 2014.  
Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language–image pre-training (siglip). arXiv preprint arXiv:2303.15343, 2023.  
Zhan, G., Liu, Y., Han, K., Xie, W., and Zisserman, A. Elip: Enhanced visual-language foundation models for image retrieval. In Proceedings of the 22nd International Conference on Content-Based Multimedia Indexing (CBMI 2025). IEEE, 2025.  
Zhong, Y., Yang, J., Zhang, P., Li, C., Codella, N., Li, L. H., Zhou, L., Dai, X., Yuan, L., Li, Y., and Gao, J. Regionclip: Region-based language-image pretraining. In CVPR, 2022.

## A. Additional Experimental Details

## A.1. Hyperparameter Sensitivity

We analyze the robustness of LARE with respect to its two primary inference-time hyperparameters: the number of selected regions N and the confidence threshold τ . These parameters control the balance between computational cost and retrieval refinement. Increasing N allows the model to examine a broader set of candidate regions and improves the likelihood of recovering small or visually subtle objects that may be underrepresented in the global embedding. The threshold τ determines when regional refinement is activated, ensuring that additional computation is performed only when the global similarity signal is uncertain.

Overall, LARE remains stable across a wide range of settings and consistently improves retrieval performance over the baseline backbone. Performance increases as the number of regions grows, indicating that additional regional evidence helps resolve ambiguous queries. Beyond a moderate number of regions, gains saturate, suggesting that most relevant visual evidence has already been captured. Similarly, the method remains robust across different confidence thresholds. Based on this analysis, we use $N = 5$ and $\tau = 0 . 2 5$ throughout the paper, as this configuration provides a strong balance between retrieval accuracy and computational efficiency.

## A.2. Implementation Notes

We follow the preprocessing and encoder configurations of the backbone models and use the OpenCLIP implementations (Cherti et al., 2023) of CLIP and related ViT-based encoders. All encoders remain frozen, and LARE operates entirely at inference time without modifying model parameters or requiring additional training.

For each image, we extract the self-attention tensor from an intermediate transformer layer and compute the patch-topatch attention maps (excluding the class token). For each head, we sum each column to measure how much attention a patch receives, reshape to a spatial grid, min–max normalize, and average the top-k heads selected by spatial variance to obtain a mean attention map. We then form the inverseattention map to identify regions that receive relatively low attention. Candidate regions are generated using a sliding window, merged using non-maximum suppression, and limited to at most N regions. Each selected region is cropped from the original image, resized to the backbone’s native input resolution, and encoded using the same frozen vision encoder to obtain regional embeddings. During retrieval, LARE applies confidence-gated fusion: regional similarity is incorporated only when it provides stronger evidence than the global similarity score. This mechanism improves retrieval in dense scenes while preserving the original backbone behavior on standard benchmarks.

![](images/a56854b3a9e5b07766a64a633f06c4484f93c4991584b0de394cfccd9841a818.jpg)

<details>
<summary>line chart</summary>

| Regions (N) | Value |
| ----------- | ----- |
| 3           | 30.1  |
| 5           | 31.0  |
| 7           | 30.8  |
| 10          | 30.4  |
</details>

(a) Effect of region count N .

![](images/c5ea3da96e74e9f699180d69f5c02ad6b41b5760a45350f4e952b3b5a3e9b65d.jpg)

<details>
<summary>line chart</summary>

| Threshold (τ) | R@1 (%) |
| ------------- | ------- |
| 0.100         | 29.8    |
| 0.150         | 30.4    |
| 0.200         | 30.9    |
| 0.250         | 31.0    |
| 0.300         | 30.7    |
</details>

(b) Effect of confidence threshold τ .  
Baseline (SigLIP) ? (Ours) ★ Peak

Figure 5. Sensitivity of LARE to inference hyperparameters. Increasing the number of regions improves retrieval performance until saturation around $N = 5 .$ . The method remains stable across thresholds and consistently outperforms the baseline.

## B. Model Card

We provide a brief model card for LARE.

• Model Architecture: LARE is a training-free augmentation pipeline that operates on frozen pretrained visionlanguage models. The pipeline contains three main components: (1) a vision transformer encoder for extracting global image embeddings and spatial attention maps, (2) a text transformer encoder for extracting text embeddings, and (3) an inverse-attention module that detects low-attention regions, re-encodes them independently, and adaptively fuses regional and global features. The vision and text encoders are frozen pretrained models, instantiated as CLIP ViT-L/14, SigLIP SoViT-400M/14, or SigLIP 2 SoViT-400M/16, accessed via OpenCLIP (Cherti et al., 2023).  
• Inputs: The vision encoder takes an image as input, preprocessed to match the backbone’s native resolution: 224 × 224 × 3 for CLIP ViT-L/14, and $3 8 4 \times 3 8 4 \times 3$ for SigLIP and SigLIP 2 models. The text encoder takes a tokenized text string, cropped to the first 64 tokens as input.  
• Outputs: The vision and text encoders output a ddimensional feature vector, where d is 768 for CLIP ViT-L/14 and 1152 for SigLIP and SigLIP 2 SoViT-400M models. The pipeline outputs a fused similarity score between the text query and image.  
• Intended Use: The method is designed for zero-shot image–text retrieval research purposes. The pipeline can be used for text-to-image and image-to-text retrieval by comparing feature vectors. The method is particularly effective for challenging retrieval scenarios where queries target fine-grained details, small objects, or background elements that may be under-emphasized by global embeddings.  
• Training Data: LARE requires no training or fine-tuning. All vision and text encoders are frozen pretrained models (e.g., CLIP and SigLIP). The inverse-attention module operates entirely at inference time and requires no additional training data.  
• Evaluation Data: Zero-shot retrieval is performed on MS-COCO, Flickr30k, and a curated dense-scene dataset (Dense-Set) to demonstrate performance across different retrieval difficulty levels.  
• Hardware & Software: The method is implemented in Python using PyTorch and OpenCLIP and evaluated on NVIDIA Quadro RTX 8000 GPUs (48GB).

## C. Pseudocode

Algorithm 1 LARE: Low-Attention Region Encoding for Retrieval  
Require: Image I, text query q, frozen vision encoder $f_{v}$ , text encoder $f_{t}$ , layer $\ell$ , top heads k, max regions N, confidence threshold $\tau$ Ensure: Retrieval score S

1: Stage 1: Low-Attention Region Detection
2: $\{\mathbf{A}^{(h)}\}_{h=1}^{H} \leftarrow f_{v}(I,\ell)$ {Extract attention maps at layer $\ell$ }
3: for each head $h = 1, \ldots, H$ do
4: $\mathbf{a}_{i}^{(h)} \leftarrow \sum_{j} \mathbf{A}_{j,i}^{(h)}$ for all patches i {Received attention}
5: $\mathbf{a}^{(h)} \leftarrow \text{MINMAXNORM}(\mathbf{a}^{(h)})$ 6: end for
7: $H_{k} \leftarrow \text{top-k}$ heads by $\text{Var}(\mathbf{a}^{(h)})$ 8: $\bar{\mathbf{A}} \leftarrow \frac{1}{k} \sum_{h \in H_{k}} \mathbf{a}^{(h)}$ 9: $M \leftarrow 1 - \bar{\mathbf{A}}$ {Inverse attention map}
10: $W \leftarrow SLIDINGWINDOW(M)$ {Candidate windows}
11: $R \leftarrow NMS(W)$ 12: $R \leftarrow TOPN(R, N)$ {Keep top-N regions}
13: Stage 2: Regional Encoding
14: for each region $r_{j} \in R$ do
15: $z_{j} \leftarrow f_{v}(\text{CROPANDRESIZE}(I, r_{j}))$ 16: end for
17: Stage 3: Confidence-Gated Scoring
18: $z_{g} \leftarrow f_{v}(I); z_{t} \leftarrow f_{t}(q)$ 19: $s_{g} \leftarrow \text{sim}(z_{t}, z_{g})$ {Global similarity}
20: $s_{r} \leftarrow \max_{j} \text{sim}(z_{t}, z_{j})$ {Best regional match}
21: if $s_{g} < \tau$ and $s_{r} > s_{g}$ then
22: $\alpha \leftarrow \min(2(s_{r} - s_{g}), 0.5)$ 23: $S \leftarrow (1 - \alpha) s_{g} + \alpha s_{r}$ 24: else
25: $S \leftarrow s_{g}$ 26: end if
27: return S