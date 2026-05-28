# FAST-GOAL: Fast and Efficient Global-local Object Alignment Learning

Hyungyu Choi\*, Young Kyun Jang\*, Chanho Eom†

Abstract—Vision-language models such as CLIP have shown impressive capabilities in aligning images and text, but they often struggle with lengthy and detailed text descriptions due to pre-training on short and concise captions. We present FAST-GOAL (Fast and Efficient Global-local Object Alignment Learning), an efficient fine-tuning method that enhances ability of CLIP to handle lengthy text through global-local semantic alignment. Our method consists of two key components. First, Fast Local Image-Sentence Matching (FLISM) efficiently extracts local image regions through object detection and spatial division, then matches them with corresponding sentences. Second, Token Similarity-based Learning (TSL) maximizes the similarity between patch tokens from specific regions in the image and their corresponding region embeddings, applying the same principle to text, which enhances the ability of the model to capture detailed correspondences. Additionally, we introduce GLIT100k, a dataset that provides both global image-lengthy caption pairs and context-derived local pairs, where local descriptions are extracted from global captions to maintain semantic coherence. Through extensive experiments on long caption datasets (DOCCI, DCI) and short caption datasets (MSCOCO, Flickr30k), we demonstrate that FAST-GOAL achieves significant improvements over baselines, enabling effective adaptation of CLIP to detailed textual descriptions while maintaining computational efficiency.

Index Terms—Vision-language model, Multi-modal learning, Lengthy text understanding.

# I. INTRODUCTION

A FTER the emergence of CLIP [1], numerous meth-ods [2]–[5] have been proposed to find the connection .ods [2]-[5] have been proposed to find the connection between images and text, showcasing significant advancements. By aligning hundreds of millions of image-caption pairs through contrastive learning, CLIP successfully encodes images and text into a unified embedding space. The resulting distribution of image and text embeddings captures both visual and textual semantics, enabling zero-shot transfer to various downstream tasks, such as retrieval [6]–[9] and classification [10]–[13], while achieving decent performance.

However, fine-tuning a pre-trained CLIP (Fig. 1 (a)) model for specific domains faces limitations, as CLIP is trained on general, short captions (e.g., maximum 77 tokens in the vanilla model) that focus on high-level image concepts. When tasked

![](images/62047ca283aa2dd41e1e2eba5c79ff39f7f2b8dca0c9a8eb7c362fe03e693275.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a grand historic building with twin towers and parked bicycles under a blue sky (no signage or text visible)
</details>

The image depicts a city scene with..

The building has a prominent clock...

In front of the building,there is alarge, paved area with a statue of a horse and rider.

A bus is parked on the side of the...

The bicycles are parked in a row...

The ground is covered with gras..

The sky is clear and blue...

There is no visible text in the image...

The overal atmosphere of the..

(a) CLIP

![](images/bee3e70149d3fc609770d8ace3993eedfb5076ba6330d4adb2d0745ec4f0af25.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a historic building with twin bicycles parked in front, under a clear blue sky (no signage or text visible)
</details>

The image depicts a city scene with..

The building has a prominent clock..

In front ofthe building,there is alarge, paved area with a statue of a horse and rider.

A bus is parked on the side of the...

The bicycles are parked in a row...

The ground is covered with grass...

The sky is clear and blue...

There is no visible text in the image...

The overall atmosphere of the...

(b) FAST-GOAL

Fig. 1. Comparison of CLIP and our FAST-GOAL’s capability in handling image-text alignment. (a) CLIP is limited to global image-text matching, treating the entire image and full caption as single units without detailed associations. (b) FAST-GOAL can establish precise local alignments between specific regions in the image and their corresponding textual descriptions in the caption (highlighted in blue).

with longer, more detailed text, CLIP struggles to capture nuanced information, as the unified embedding space is optimized for concise descriptions. This makes adapting CLIP for retrieval tasks requiring lengthy text challenging without architectural adjustments or specialized training techniques.

In this paper, we propose a novel but simple fine-tuning method for image and lengthy text pairs, called FAST-GOAL (Fast and Efficient Global-local Object Alignment Learnig) (Fig. 1 (b)). Here, we refer to “global” as the entire image or text and “local” as a sub-part, such as a segment of the image or a specific sentence in the text. The idea behind FAST-GOAL is to enable the encoder model to focus on the dominant local elements within each image and text sample, thereby enhancing the overall understanding of the sample and producing a more representative embedding.

FAST-GOAL has two key components: First, Fast Local Image-Sentence Matching (FLISM), a pipeline that extracts local segments from images and matches them with corresponding descriptive sentences from the entire caption. Sec-

\*Authors contributed equally. †Corresponding author. Hyungyu Choi and Chanho Eom are with the Department of Virtual Convergence, Graduate School of Advanced Imaging Science, Multimedia & Films (GSAIM), Chung-Ang University, Seoul, South Korea (email: qkenr0804@cau.ac.kr; cheom@cau.ac.kr). Young Kyun Jang (e-mail: kyun0914@gmail.com). This paper has supplementary downloadable material available at http://ieeexplore.ieee.org., provided by the author. The material includes additional experiments related to this work. Contact qkenr0804@cau.ac.kr for further questions about this work. GitHub: https://github.com/PerceptualAI-Lab/FAST-GOAL

ond, we introduce Token Similarity-based Learning (TSL), a method that effectively propagates attention of local element using the local pairs obtained through the FLISM pipeline. Additionally, we present Global-Local Image-Text pair 100k (GLIT100k), a dataset comprising 100k image-lengthy caption pairs that also provides fine-grained local pairs matching image segments to corresponding sentences, enabling multilevel supervision for lengthy text understanding. To validate FAST-GOAL across diverse domains, we evaluate our method on both long caption datasets (DOCCI [14] and DCI [15]) and short caption datasets (COCO [16] and Flickr30k [17]). This extensive evaluation demonstrates that FAST-GOAL effectively addresses the challenge of image-lengthy text retrieval while maintaining robust performance on short caption tasks. Our method shows substantial improvements compared to baseline models across diverse evaluation scenarios. The main contributions of our work can be summarized as follows:

• We propose FAST-GOAL, a fast and efficient fine-tuning method that leverages FLISM and TSL methods to enable effective adaptation of CLIP for lengthy text understanding through global-local alignment at scale.   
• We introduce GLIT100k, a dataset that provides both image-lengthy caption pairs and local pairs matching image segments to corresponding text descriptions, offering the multi-level supervision necessary for effective lengthy text understanding.   
• Through diverse experiments across both long caption domains including DOCCI and DCI, as well as short caption domains such as MSCOCO and Flickr30k, we show that FAST-GOAL significantly outperforms baseline CLIP models while maintaining strong performance across varying caption lengths.

This paper builds upon our prior work GOAL [18]. This journal extension adds: 1) FLISM, an efficient pipeline that combines object detection with spatial division for local pair matching, reducing computational costs while maintaining accurate image-text correspondences between regions and sentences; 2) GLIT100k, a 100k-scale dataset providing contextderived local pairs where local descriptions are extracted from global captions rather than generated independently, ensuring semantic coherence across global-local hierarchies; 3) Experimental validation across both long caption datasets (DOCCI, DCI) and short caption datasets (MSCOCO, Flickr30k), demonstrating consistent improvements across varying caption lengths; 4) Analysis of computational efficiency showing that our 100k-scale approach achieves competitive performance while requiring less resources than million-scale alternatives; 5) Qualitative analysis through retrieval visualizations demonstrating superior ability of FAST-GOAL to capture fine-grained visual-textual correspondences compared to baseline models.

# II. RELATED WORK

# A. Vision-Language Pre-training

Research on addressing alignment differences between vision and language modalities has brought the CLIP [1] model into the spotlight. CLIP, a multi-modal embedding model trained through contrastive learning on over 400 million image-text pairs, effectively aligns visual and textual representations while demonstrating remarkable zero-shot capabilities. Following its success, larger pre-training models emerged, such as ALIGN [19] and Florence [20], trained on imagetext pairs from datasets containing 1.8B and 900M samples, respectively. However, these models typically rely on short, broad image descriptions as captions, causing them to miss crucial local-level detailed information. This limits their ability to focus primarily on global understanding while failing to capture local details. To overcome this limitation, we present a fine-tuning method that enhances CLIP’s ability to capture both local-detail and global-semantic information by training it on a dataset containing detailed, multi-sentence captions.

# B. CLIP for Long Text Understanding

The primary challenge in enhancing CLIP [1] for long text understanding lies in finding an effective balance between dataset scale and performance. Current methods fall into two main paradigms, each with fundamental trade-offs. Large scale pre-training methods such as FG-CLIP [21] first leverage massive billion-scale datasets to develop robust multimodal representations from scratch, then require additional hard negative samples to enhance fine-grained understanding. This two-stage process results in demanding computational resources and training infrastructure, severely limiting practical adoption. As an alternative, fine-tuning methods have gained attention for their accessibility. Long-CLIP [22] pioneers the use of lengthy captions generated by multimodal large language models (MLLMs) for CLIP fine-tuning, incorporating both coarse-grained and fine-grained alignment strategies with million-scale datasets [23]. FineLIP [24] follows a similar path, relying on million-scale data and employing additional token aggregation modules to reduce ambiguity before crossmodal alignment. Despite their improvements, the reliance on million-scale datasets in both methods raises fundamental questions about whether such large-scale data requirements are truly necessary for effective fine-tuning. Addressing this question, GOAL [18] demonstrates that remarkable performance can be achieved using only small-scale datasets [14], [15], proving that data efficiency is possible without massive resources. Nevertheless, the expensive dataset curation process in GOAL creates a new bottleneck. This prevents exploration of the optimal balance point where both efficiency and scalability could coexist. To address this challenge, we propose FAST-GOAL, which leverages our 100k-scale dataset, GLIT100k. This scale is large enough to ensure robust performance yet small enough to maintain computational efficiency. Through efficient dataset construction and training processes, our method achieves superior performance without the burden of million-scale data or additional architectural parameters, establishing a new paradigm for practical long text understanding in CLIP.

# C. Utilizing Local Elements in Vision-Language Model Training

In terms of vision-language alignment models, using local elements’ knowledge to improve the model’s general ability has been widely explored across various domains. ViTAA [25] learns to align full-person images corresponding to the globallevel with text describing the whole person to perform a person re-identification task [26]–[29], while also learning to align the image and text for attributes (e.g., hair, pants, shoes) that correspond to the local-level. This approach combines globallocal relations, enabling richer visual-language representation learning. CLOC [30] builds 2 billion image-text datasets and uses them for pre-training models by matching local objects and phrase-levels through Open-vocabulary Detector (e.g., OWLv2 [31], GLIPv2 [32]) models. This approach aims to improve localization capabilities while maintaining $\mathrm { C L I P } ^ { \prime } \mathrm { s }$ global-level representation, demonstrating superior performance compared to the original pre-trained CLIP model. In contrast, our proposed FAST-GOAL method addresses image-lengthy text matching through region-sentence alignment, which fundamentally differs from existing fine-tuning approaches [33], [34] that focus on region-phrase matching with short captions. While these methods align image regions with noun phrases or brief descriptions, lengthy captions contain multiple complete sentences describing various scene aspects. This requires sentence-level local pairs to maintain contextual coherence between global and local descriptions.

![](images/0fa8f5313c4eb8e796ebd9b05ca24e5d47fc9c6cbfcf453a4b4b8f1fecc8258f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Local Regions"] --> B["Default"]
    B --> C["YOLOS"]
    C --> D["CLIP Image Encoder"]
    D --> E["CLS"]
    E --> F["CLS"]
    F --> G["CLS"]
    G --> H["CLIP Text Encoder"]
    H --> I["The image captures a bustling airport..."]
    H --> J["Dominating the foreground is a large..."]
    H --> K["The airplane is stationed on the..."]
    H --> L["There are no people visible in the image..."]
    H --> M["The relative positions of the objects..."]
    H --> N["The airplanes are parked in a way that..."]
    A --> O["Ig"]
    O --> P["Local Regions"]
    P --> Q["YOLOS"]
    Q --> R["CLS"]
    R --> S["CLS"]
    S --> T["CLS"]
    T --> U["CLIP Image Encoder"]
    U --> V["CLS"]
    V --> W["CLS"]
    W --> X["CLIP Text Encoder"]
    X --> Y["The image captures a bustling airport..."]
    X --> Z["Dominating the foreground is a large..."]
    X --> AA["The airplane is stationed on the..."]
    X --> AB["There are no people visible in the image..."]
    X --> AC["The relative positions of the objects..."]
    X --> AD["The airplanes are parked in a way that..."]
```
</details>

Fig. 2. Overview of Fast Local Image-Sentence Matching (FLISM) pipeline. Given a global image and its detailed caption, FLISM uses YOLOS [35] to detect objects in the image to create local regions and splits the caption into individual sentences. These local pairs are then processed through CLIP encoder to obtain CLS embeddings, which are used for maximum similarity matching to identify the most relevant image-sentence pairs.

# III. METHOD

In this section, we first present Fast Local Image-Sentence Matching (FLISM), an efficient pipeline that generates local-level pseudo pairs from image-lengthy caption pairs (Sec. III-A). We then describe Token Similarity-based Learning (TSL), which leverages these pseudo pairs to enhance CLIP’s fine-grained understanding capabilities (Sec. III-B).

# A. Fast Local Image Sentence Matching

We propose Fast Local Image-Sentence Matching (FLISM) illustrated in Fig. 2. FLISM efficiently separates a given caption into individual sentences and identifies corresponding image segments, matching each sentence with its relevant segment. To this end, we first decompose a given caption $T _ { g } ,$ which provides detailed descriptions of a given image $I _ { g } ,$ into individual sentences, resulting in text segments $\{ T _ { l , i } \} _ { i = 1 } ^ { M ^ { - } }$ where M is the number of sentences.

For local image extraction, FLISM combines object detection with spatial division to obtain diverse local image candidates. While the previous method [18] relies on a dense segmentation methods, SAM [36], that require significant computational overhead to analyze all possible image regions, we design an efficient extraction strategy that focuses on semantically meaningful regions. Specifically, we employ YO-LOS [35], a lightweight object detection model, to detect and extract object regions from the image $I _ { g } ,$ directly identifying semantically meaningful local segments that correspond to actual visual entities. We complement these object-centric regions with default local regions: four quadrants dividing the image into equal sections and one central region formed by connecting the midpoints of each quadrant. The YOLOSdetected regions capture object-level information while the default regions provide spatial coverage for areas that may not contain distinct objects. Together, these yield a set of local images, $\{ I _ { l , i } \} _ { i = 1 } ^ { N }$ , where N represents the total number of local regions. This approach enables local image extraction without the computational overhead of exhaustive segmentation methods.

We use CLIP [1] to match the decomposed caption segments with the corresponding image segments. Specifically, we extract the CLS token embeddings for each local text segment $\{ T _ { l , i } \} _ { i = 1 } ^ { M } .$ 1 from the text encoder of CLIP, $\phi _ { t } ,$ , as follows:

$$
\{t _ {l, i} ^ {\mathrm{cls}} \} _ {i = 1} ^ {M} = \phi_ {t} (\{T _ {l, i} \} _ {i = 1} ^ {M}). \tag {1}
$$

Similarly, for each image segment $\{ I _ { l , i } \} _ { i = 1 } ^ { N }$ , we extract the CLS token embeddings from the visual encoder of CLIP as follows:

$$
\{v _ {l, i} ^ {\mathrm{cls}} \} _ {i = 1} ^ {N} = \phi_ {v} (\{I _ {l, i} \} _ {i = 1} ^ {N}). \tag {2}
$$

Next, we compute the cosine similarity between each local text embedding $t _ { l , i } ^ { \mathrm { c l s } }$ tl,i and the local image embeddings $\{ v _ { l , i } ^ { \mathrm { c l s } } \} _ { i = 1 } ^ { N } .$ Among all matched pairs, each local text embedding is matched with its highest similarity image embedding. From all these matched pairs, we select the one pair with the highest similarity score and denote it as $( I _ { l } , T _ { l } )$ . This matching strategy ensures high-quality local pair associations by focusing exclusively on local correspondences, while maintaining computational efficiency through the YOLOS-based extraction process.

![](images/43129930451d0bd1881fe24024e6bc6e3a86bab657beff021669c3b5101cf0c8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Image"] --> B["CLIP Image Encoder"]
    B --> C["CLIP Image Encoder"]
    C --> D["CLIP Image Encoder"]
    D --> E["CLIP Image Encoder"]
    E --> F["CLIP Image Encoder"]
    F --> G["CLIP Image Encoder"]
    G --> H["CLIP Image Encoder"]
    H --> I["CLIP Image Encoder"]
    I --> J["CLIP Image Encoder"]
    J --> K["CLIP Image Encoder"]
    K --> L["CLIP Image Encoder"]
    L --> M["CLIP Image Encoder"]
    M --> N["CLIP Image Encoder"]
    N --> O["CLIP Image Encoder"]
    O --> P["CLIP Image Encoder"]
    P --> Q["CLIP Image Encoder"]
    Q --> R["CLIP Image Encoder"]
    R --> S["CLIP Image Encoder"]
    S --> T["CLIP Image Encoder"]
    T --> U["CLIP Image Encoder"]
    U --> V["CLIP Image Encoder"]
    V --> W["CLIP Image Encoder"]
    W --> X["CLIP Image Encoder"]
    X --> Y["CLIP Image Encoder"]
    Y --> Z["CLIP Image Encoder"]
    Z --> AA["CLIP Image Encoder"]
    AA --> AB["CLIP Image Encoder"]
    AB --> AC["CLIP Image Encoder"]
    AC --> AD["CLIP Image Encoder"]
    AD --> AE["CLIP Image Encoder"]
    AE --> AF["CLIP Image Encoder"]
    AF --> AG["CLIP Image Encoder"]
    AG --> AH["CLIP Image Encoder"]
    AH --> AI["CLIP Image Encoder"]
    AI --> AJ["CLIP Image Encoder"]
    AJ --> AK["CLIP Image Encoder"]
    AK --> AL["CLIP Image Encoder"]
    AL --> AM["CLIP Image Encoder"]
    AM --> AN["CLIP Image Encoder"]
    AN --> AO["CLIP Image Encoder"]
    AO --> AP["CLIP Image Encoder"]
    AP --> AQ["CLIP Image Encoder"]
    AQ --> AR["CLIP Image Encoder"]
    AR --> AS["CLIP Image Encoder"]
    AS --> AT["CLIP Image Encoder"]
    AT --> AU["CLIP Image Encoder"]
    AU --> AV["CLIP Image Encoder"]
    AV --> AW["CLIP Image Encoder"]
    AW --> AX["CLIP Image Encoder"]
    AX --> AY["CLIP Image Encoder"]
    AY --> AZ["CLIP Image Encoder"]
    AZ --> BA["CLIP Image Encoder"]
    BA --> BB["CLIP Image Encoder"]
    BB --> BC["CLIP Image Encoder"]
    BC --> BD["CLIP Image Encoder"]
    BD --> BE["CLIP Image Encoder"]
    BE --> BF["CLIP Image Encoder"]
    BF --> BG["CLIP Image Encoder"]
    BG --> BH["CLIP Image Encoder"]
    BH --> BI["CLIP Image Encoder"]
    BI --> BJ["CLIP Image Encoder"]
    BJ --> BK["CLIP Image Encoder"]
    BK --> BL["CLIP Image Encoder"]
    BL --> BM["CLIP Image Encoder"]
    BM --> BN["CLIP Image Encoder"]
    BN --> BO["Tg"]
    BO --> BP["Tl"]
    BP --> BQ["Tg"]
    BQ --> BR["Tl"]
    BR --> BS["Tg"]
    BS --> BT["Tl"]
    BT --> BU["Tg"]
    BU --> BV["Tl"]
```
</details>

Fig. 3. Overview of Token Similarity based Learning (TSL). The framework processes global image-text pairs and their local pairs through shared CLIP encoders, extracting patch and sequence tokens. TSL identifies and projects corresponding token regions to match local CLS embeddings, enabling attention on local element.

# B. Token Similarity based Learning

While CLIP’s pretraining with image-text pairs effectively learns global alignment, its training with brief captions limits the model’s ability to capture fine-grained local details from lengthy descriptions. To address this, we propose Token Similarity based Learning (TSL) (Fig. 3). Our approach uses local pairs obtained through the FLISM pipeline and implements a fine-tuning strategy that effectively propagates local-level information. Specifically, TSL maximizes the similarity between patch tokens of local regions in the global image and their corresponding local image embeddings, while applying the same principle to text by increasing the similarity between sequence tokens of local parts in the global text and their corresponding local text embeddings. To implement this strategy, we need to extract both local and global features from the input pairs. Using CLIP’s vision encoder $\phi _ { v }$ and text encoder $\phi _ { t }$ , we extract both local and global features as follows: For the local text $T _ { l }$ ,

$$
t _ {l} ^ {c l s} = \phi_ {t} (T _ {l}) \in \mathbb {R} ^ {d}, \tag {3}
$$

where $t _ { l } ^ { c l s }$ represents the last layer CLS token embedding. For the global text $T _ { g }$ , the text encoder extracts:

$$
S _ {g} = \phi_ {t} (T _ {g}) \in \mathbb {R} ^ {M \times d}, \tag {4}
$$

where M is the sequence length of $T _ { g }$ , and $S _ { g }$ represents the last layer sequence tokens of $T _ { g } .$ . To handle text sequences longer than $\mathrm { C L I P } ^ { \prime } \mathrm { s }$ standard 77 token limit, we adopt Long-CLIP’s [22] positional embedding interpolation method in our text encoder. For the local image $I _ { l } ,$ we obtain:

$$
v _ {l} ^ {c l s} = \phi_ {v} (I _ {l}) \in \mathbb {R} ^ {d}, \tag {5}
$$

where $v _ { l } ^ { c l s }$ represents the last layer CLS token embedding. For the global image $I _ { g } ,$ , the vision encoder extracts:

$$
P _ {g} = \phi_ {v} (I _ {g}) \in \mathbb {R} ^ {N \times d}, \tag {6}
$$

where N denotes the number of patch tokens in $I _ { g }$ , d is the embedding dimension and $P _ { g }$ represents the last layer patch tokens of $I _ { g } .$ We process both global and local pairs through shared CLIP encoders to learn both types of features simultaneously. This weight sharing ensures consistent encoding in the shared embedding space. Let $\tau$ denote the set of token indices corresponding to the local text segment. We can identify the sequence tokens in $S _ { g }$ that correspond to $T _ { l }$ ,

$$
S _ {m} = \frac {1}{| \mathcal {T} |} \sum_ {i \in \mathcal {T}} S _ {g} [ i ] \in \mathbb {R} ^ {d}, \tag {7}
$$

where $| \tau |$ denotes the number of selected sequence tokens. The aggregated features are then projected into a shared embedding space, where both text and image representations are aligned:

$$
\hat {S} _ {l} = \psi_ {t} (S _ {m}) \in \mathbb {R} ^ {d}, \tag {8}
$$

where $\psi _ { t } ( \cdot )$ represents a learned textual projection function.

Given that each local image region $I _ { l }$ has its bounding box coordinates $( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } )$ obtained from FLISM in the global image $I _ { g } ,$ we can leverage this spatial information to identify specific patch tokens from $P _ { g }$ that correspond to the local image region, filtering out patches from other parts of the global image. Let B denote the set of indices of patch tokens located inside the bounding box. We aggregate these tokens using average pooling to capture comprehensive information from the selected region:

$$
P _ {m} = \frac {1}{| \mathcal {B} |} \sum_ {i \in \mathcal {B}} P _ {g} [ i ] \in \mathbb {R} ^ {d}, \tag {9}
$$

where |B| denotes the number of selected patch tokens. The aggregated features are then projected into a shared embedding space where both text and image representations are aligned:

$$
\hat {P} _ {l} = \psi_ {v} (P _ {m}) \in \mathbb {R} ^ {d}, \tag {10}
$$

where the $\psi _ { v } ( \cdot )$ represents a learned visual projection function. We train our model with multiple objectives combined into a final loss function:

$$
\mathcal {L} _ {\text { total }} = \lambda_ {\text { global }} \mathcal {L} _ {\text { global }} + \lambda_ {\text { local }} \mathcal {L} _ {\text { local }} + \lambda_ {T S L} \mathcal {L} _ {\text { TSL }}, \tag {11}
$$

where λ is a hyperparameter controlling the contribution of local alignment. We apply contrastive learning at both global and local levels, adopting the contrastive learning used in CLIP. At the global level:

$$
\mathcal {L} _ {\text { global }} = \mathcal {L} _ {\text { contrast }} (v _ {g} ^ {c l s}, t _ {g} ^ {c l s}), \tag {12}
$$

where $v _ { g } ^ { c l s }$ and $t _ { g } ^ { c l s }$ are the CLS token embeddings of the global image $I _ { g }$ and global text $T _ { g } ,$ respectively. This global alignment ensures that the model maintains CLIP’s original capability to capture global relationships between image-text pairs. Similarly, for local-level contrastive learning:

$$
\mathcal {L} _ {\text { local }} = \mathcal {L} _ {\text { contrast }} (v _ {l} ^ {c l s}, t _ {l} ^ {c l s}), \tag {13}
$$

where $v _ { l } ^ { c l s }$ and $t _ { l } ^ { c l s }$ are the CLS token embeddings of the local image $I _ { l }$ and local text $T _ { l } ,$ respectively. By applying contrastive learning to local CLS token pairs, we encourage precise alignment between local image regions and their corresponding textual descriptions, enabling the model to learn cross-modal relationships.

The token similarity loss $\mathcal { L } _ { \mathrm { T S L } }$ maximizes the similarity between projected tokens and their corresponding local CLS token embeddings for both image and text:

$$
\mathcal {L} _ {\mathrm{TSL}} = \operatorname{MSE} \left(\operatorname{sim} \left(\hat {P} _ {l}, v _ {l} ^ {c l s}\right), \mathbf {1}\right) + \operatorname{MSE} \left(\operatorname{sim} \left(\hat {S} _ {l}, t _ {l} ^ {c l s}\right), \mathbf {1}\right), \tag {14}
$$

where sim(·) denotes a function that computes an $n \times n$ similarity matrix with n being the batch size, and 1 is a n  n matrix with ones on its diagonal entries. By optimizing this loss, the model learns to maximize the similarity between local CLS token embeddings and their corresponding regions in global tokens. This token-level alignment strategy enables the model to attention on local element, enhancing finegrained understanding capabilities. This fine-tuning method effectively addresses CLIP’s inherent limitation in capturing local details from lengthy descriptions, which stems from its pre-training with brief captions. Through the combination of token-level similarity learning and global-local contrastive learning, our approach enables comprehensive understanding of cross-modal relationships with attention on local element from detailed text descriptions.

# Prompt

"Please describe this image in detail, but ONLY what is visibly present in the image.Focus on:

1.All major objects and their spatial relationships that are clearlyvisible   
2.Colors,textures,and materials of objects you can directly see   
3.Actions or activities taking place that are explicitly shown   
4. The scene setting and background details that are present in the frame   
5. Any people present and what they are doing (only what's visible)   
6.The lighting and atmosphere of the scene as shown   
7.Any text visible in the image

# IMPORTANT:

- Describe ONLY what you can actually see in the image   
- DO NOT make inferences about what might be happening outside the frame   
- DO NOT speculate about past or future events not shown

Provide a factual description in paragraph form that strictlyadheres towhat is visiblypresent."

Image   
![](images/aeca2124a56afd21fb673ee8319541a022e5286b5bb87d1db7724af07e0d83a6.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a park with a fountain and modern buildings in the background (no signage or text visible)
</details>

# Generated long caption

The image showcases a vibrant city park scene.At the center of the image isa large,ornate fountain with multiple spouts.The fountainis surroundedbya lush green lawn,and there are several statues ofhorses and riders positioned around the fountain. The statues are made of a dark material, possibly bronze or stone,and theyaredetailedandlifelike.Totheleftof the fountain, there isa black bench,providinga place for visitors to sit and enjoy the view.Thebench is empty in the image.In the background, there are tall buildings,indicating that the park is located in an urban area.The buildings are of various heights and designs...

Fig. 4. Prompt-based caption generation with LLaVA-Next [37]. Our prompt guides the model to generate descriptions covering scene context, object attributes, and spatial relationships while minimizing hallucinations.

TABLE I CAPTION QUALITY COMPARISON BETWEEN GENERATED AND HUMAN-ANNOTATED DATASETS MEASURED USING LONG-CLIP VIT-L/14 ON 1,000 RANDOMLY SAMPLED IMAGE-TEXT PAIRS. 

<table><tr><td>Dataset</td><td>Annotation Type</td><td>Avg. Cosine Similarity</td></tr><tr><td>GLIT100k</td><td>Generated (LLaVA-Next)</td><td>0.2434</td></tr><tr><td>DOCCI</td><td>Human-annotated</td><td>0.2448</td></tr><tr><td>Difference</td><td></td><td>0.0014</td></tr></table>

# IV. GLIT100K DATASET CONSTRUCTION

# A. Long caption generation

We construct GLIT100k (Global-Local Image-Text pair 100k), a dataset that uniquely provides both image-lengthy caption pairs and fine-grained local pairs for enhancing long text understanding capabilities of CLIP [1]. Our dataset construction derives local pairs directly from global captions, ensuring semantic consistency between different granularities while avoiding redundant generation processes. We sample diverse images from SA-1B [36], which contains rich visual scenes with multiple objects and complex spatial arrangements. The diversity of SA-1B, which is originally curated for segmentation tasks across varied domains, provides an ideal foundation for extracting both global scene understanding and detailed local image-text correspondences. Recent advances in multi-modal large language models (MLLMs) [37]–[41] have enabled generation of highly detailed and accurate scene descriptions. We employ LLaVA-Next [37] for caption generation, which has demonstrated strong performance in producing comprehensive visual descriptions.

Figure 4 illustrates our prompt that ensures high-quality lengthy captions by guiding the model to minimize hallucinations and produce visually-grounded descriptions. Our prompt explicitly instructs the model to describe only visually observable elements while covering multiple levels of detail from overall scene context to specific object attributes and spatial relationships. This structured approach yields detailed multisentence captions that capture comprehensive scene semantics, providing the rich textual information necessary for lengthy text understanding tasks.

To validate the quality of our generated captions, we conduct a comparison between GLIT100k and the humanannotated DOCCI [14] dataset. Table I presents quantitative validation results measuring semantic alignment between images and their corresponding text descriptions. We randomly sample 1,000 image-text pairs from each dataset and compute their cosine similarity using Long-CLIP ViT-L/14. GLIT100k achieves 0.2434 average cosine similarity compared to the 0.2448 achieved by human-annotated DOCCI, demonstrating comparable semantic consistency with only 0.0014 difference. This minimal gap validates that our prompt engineering approach effectively produces high-quality captions that maintain strong semantic alignment with their corresponding images, ensuring reliable supervision for training vision-language models.

TABLE II COMPARISON OF MLLM-GENERATED DATASETS FOR CLIP LONG TEXT UNDERSTANDING. CONTEXT-DERIVED LOCAL PAIRS INDICATE DESCRIPTIONS EXTRACTED FROM GLOBAL CAPTIONS RATHER THAN GENERATED INDEPENDENTLY. 

<table><tr><td>Dataset</td><td>Scale</td><td>Global Pairs</td><td>Local Pairs</td><td>Context-Derived Local Pairs</td></tr><tr><td>ShareGPT4V</td><td>1.2M</td><td>√</td><td></td><td></td></tr><tr><td>FineHARD</td><td>12M</td><td>√</td><td>√</td><td></td></tr><tr><td>GLIT100k (Ours)</td><td>100K</td><td>√</td><td>√</td><td>√</td></tr></table>

# B. Long caption datasets

Table II shows a comparison of datasets generated by MLLMs for long text understanding. Each dataset differs in scale and structural design. ShareGPT4V [23] contains 1.2M samples but only provides global caption pairs, lacking finegrained image-text alignment. FineHARD [21] scales to 12M, including global and local pairs. However, local pairs are built by independently generating text for individual cropped image regions, resulting in potential semantic inconsistencies and loss of contextual relationships between local descriptions and the overall scene narrative. In contrast, our GLIT100k introduces context-derived local pairs and includes both global and local pairs. We utilize the specific local sentence within the broader narrative context of the full caption it belongs to. This contextaware approach results in a richer and more accurate semantic representation for that individual sentence. This design maintains the natural relationships present in the original scene description while avoiding redundant generation processes. By preserving contextual connections between global and local pairs, GLIT100k is designed to achieve and effective long text understanding through its efficient structure without requiring million-scale data, as demonstrated by our experimental results in Sec.V-C.

# V. EXPERIMENTS

In this section, we present our experimental setup in Sec.V-A. We then evaluate the zero-shot retrieval performance of FAST-GOAL across both long and short caption domains in Sec.V-B, demonstrating the effectiveness of our method on datasets with varying caption lengths. Our ablation study analyzes the computational efficiency of Fast Local Image-Sentence Matching (FLISM) and validates the effectiveness of our GLIT100k dataset for multi-stage training. Additionally, we examine the contribution of each loss component in our framework in Sec.V-C. Finally, we provide qualitative analysis through attention map visualizations and retrieval result comparisons in Sec.V-D.

# A. Experimental setup

Dataset. We train our FAST-GOAL model using GLIT100k, our proposed dataset comprising 100,647 image-lengthy caption pairs with an average of 159 words per caption, along with fine-grained local pairs matching image segments to corresponding sentences. This dataset enables robust training on detailed textual descriptions and fine-grained visual-textual relationships at scale. For evaluation, we conduct experiments on both long caption domains including DOCCI [14] and DCI [15], as well as short caption domains such as MSCOCO [16] and Flickr30k [17], to comprehensively assess our model’s performance across varying caption lengths.

TABLE III PERFORMANCE COMPARISON ON LONG CAPTION DOMAIN DATASETS (DOCCI AND DCI) USING VIT-B/16 BACKBONE. RESULTS SHOW RECALL@K SCORES FOR TEXT TO IMAGE AND IMAGE TO TEXT RETRIEVAL TASKS. THE BEST PERFORMANCE IS SHOWN IN BOLD, AND THE SECOND-BEST PERFORMANCE IS UNDERLINED. 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="4">Text to Image Recall@K</td><td colspan="4">Image to Text Recall@K</td></tr><tr><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td></tr><tr><td rowspan="6">DOCCI</td><td>CLIP</td><td rowspan="5">ViT-B/16</td><td>58.53</td><td>85.57</td><td>97.16</td><td>98.82</td><td>58.80</td><td>85.18</td><td>96.73</td><td>98.45</td></tr><tr><td>EVA-CLIP</td><td>66.61</td><td>89.08</td><td>97.53</td><td>99.00</td><td>66.24</td><td>89.43</td><td>97.69</td><td>98.88</td></tr><tr><td>Long-CLIP</td><td>71.63</td><td>92.16</td><td>98.90</td><td>99.73</td><td>63.29</td><td>88.80</td><td>98.39</td><td>99.45</td></tr><tr><td>FineCLIP</td><td>62.92</td><td>87.22</td><td>96.67</td><td>98.35</td><td>56.55</td><td>82.94</td><td>95.47</td><td>97.92</td></tr><tr><td>GOAL</td><td>71.22</td><td>92.39</td><td>98.90</td><td>99.61</td><td>72.18</td><td>92.88</td><td>98.88</td><td>99.55</td></tr><tr><td>FAST-GOAL</td><td></td><td>74.27</td><td>93.76</td><td>99.27</td><td>99.71</td><td>74.67</td><td>94.71</td><td>99.24</td><td>99.75</td></tr><tr><td rowspan="6">DCI</td><td>CLIP</td><td rowspan="5">ViT-B/16</td><td>33.36</td><td>53.87</td><td>72.34</td><td>79.33</td><td>35.07</td><td>54.28</td><td>71.84</td><td>78.91</td></tr><tr><td>EVA-CLIP</td><td>41.25</td><td>61.19</td><td>77.62</td><td>83.03</td><td>40.71</td><td>61.14</td><td>77.73</td><td>83.60</td></tr><tr><td>Long-CLIP</td><td>48.41</td><td>68.71</td><td>82.49</td><td>87.16</td><td>42.21</td><td>67.21</td><td>83.21</td><td>88.21</td></tr><tr><td>FineCLIP</td><td>37.70</td><td>58.30</td><td>75.82</td><td>81.60</td><td>36.48</td><td>58.20</td><td>76.14</td><td>82.74</td></tr><tr><td>GOAL</td><td>49.13</td><td>70.17</td><td>84.53</td><td>88.69</td><td>51.87</td><td>71.10</td><td>84.78</td><td>89.12</td></tr><tr><td>FAST-GOAL</td><td></td><td>56.62</td><td>75.26</td><td>87.75</td><td>91.08</td><td>56.74</td><td>75.99</td><td>88.71</td><td>92.46</td></tr></table>

Training Setting. To validate our approach, we conduct experiments using CLIP [1] ViT-B/16 [42] backbone architecture. The model is fine-tuned for 10 epochs with a batch size of 256. We set the balance hyperparameters in the total loss function as $\lambda _ { \mathrm { g l o b a l } } = 1 , \lambda _ { \mathrm { T S L } } = 1$ , and $\lambda _ { \mathrm { l o c a l } } = 0 . 5$ to maintain strong global and TSL learning while moderating the contribution of local loss. Training was performed on four NVIDIA A6000 GPUs, taking approximately 1–2 hours to complete.

Test Setting. To handle the long text sequences during inference, we adopt the positional embedding interpolation from Long-CLIP [22]. Our evaluation encompasses both text-toimage (T2I) and image-to-text (I2T) retrieval tasks, measured using Recall@K metrics (K = 1, 5, 10, 15, 25, 50). To ensure fair comparison across all baseline methods, we employ the ViT-B/16 [42] backbone throughout our experiments. We evaluate our method on the original test sets of each dataset. We use the DOCCI test set consisting of 5,000 images, 7,602 samples from the DCI dataset after excluding 203 samples without long captions from the original 7,805 samples as our test set, MSCOCO 2014 validation set, and the Flickr30k test set.

# B. Zero-Shot Retrieval Performance

We evaluate FAST-GOAL’s effectiveness through comprehensive zero-shot retrieval experiments across datasets with varying caption lengths. We compare FAST-GOAL against several state-of-the-art baselines including the original CLIP [1], EVA-CLIP [43], Long-CLIP [22], FineCLIP [33], and our previous work [18].

Long Caption Datasets. Table III presents results on datasets with lengthy and detailed descriptions: DOCCI [14] and DCI [15]. These datasets are particularly challenging as they contain comprehensive visual descriptions that require finegrained understanding of image-text correspondences. For fair a evaluation of our previous work [18], we employ crossdataset evaluation where models trained on DOCCI are evaluated on DCI and vice versa. The results demonstrate the superior performance of FAST-GOAL in handling lengthy textual descriptions, significantly outperforming baseline methods across all recall metrics. On the challenging R@1 and R@5 metrics, which require precise matching, FAST-GOAL demonstrates significant performance gains compared to other baseline methods. On DOCCI, FAST-GOAL achieves 74.27% and 93.76% for text-to-image R@1 and R@5, significantly outperforming the 71.63% and 92.16% of Long-CLIP respectively. Similarly, on DCI, FAST-GOAL attains 56.62% and 75.26% compared to the 48.41% and 68.71% of Long-CLIP. These substantial gaps in the more difficult recall metrics demonstrate that our method excels particularly in scenarios requiring precise retrieval accuracy. Notably, FAST-GOAL shows substantial improvements over our previous work [18], validating the effectiveness of our training approach with the GLIT100k dataset and enhanced FLISM pipeline.

Short Caption Datasets. Table IV shows results on traditional short caption datasets: MSCOCO [16] and Flickr30k [17]. These datasets contain concise descriptions typical of conventional image-text retrieval tasks. Despite being optimized for lengthy text understanding, FAST-GOAL maintains competitive performance on short caption domains, demonstrating the generalizability of our approach across varying caption lengths. On MSCOCO, FAST-GOAL achieves 42.81% and 68.61% for text-to-image R@1 and R@5, outperforming EVA-CLIP’s 42.58% and 67.49% respectively (we exclude FineCLIP from this comparison as it is fine-tuned on MSCOCO itself, which would not constitute a fair zero-shot evaluation). On Flickr30k, FAST-GOAL attains 72.50% for text-to-image R@1, outperforming EVA-CLIP’s 71.82%. The results indicate that our method does not sacrifice performance on traditional short caption retrieval tasks while gaining substantial improvements on lengthy text understanding. This balanced performance consistency makes FAST-GOAL a practical solution for real-world applications where both short and long text descriptions may be encountered. The consistent performance across varying text lengths demonstrates that FAST-GOAL effectively adapts to different levels of textual detail without overfitting to specific caption styles.

TABLE IV PERFORMANCE COMPARISON ON SHORT CAPTION DOMAIN DATASETS (MSCOCO AND FLICKR30K) USING VIT-B/16 BACKBONE. RESULTS SHOW RECALL@K SCORES FOR TEXT TO IMAGE AND IMAGE TO TEXT RETRIEVAL TASKS. THE BEST PERFORMANCE IS SHOWN IN BOLD, AND THE SECOND-BEST PERFORMANCE IS UNDERLINED. NOTE THAT FINECLIP ON MSCOCO IS FINE-TUNED ON MSCOCO ITSELF, THUS IT IS NOT A FAIR ZERO-SHOT COMPARISON. 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="4">Text to Image Recall@K</td><td colspan="4">Image to Text Recall@K</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@15</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@15</td></tr><tr><td rowspan="5">MSCOCO</td><td>CLIP</td><td rowspan="5">ViT-B/16</td><td>33.95</td><td>59.46</td><td>69.93</td><td>76.00</td><td>54.14</td><td>77.74</td><td>85.60</td><td>89.36</td></tr><tr><td>EVA-CLIP</td><td>42.58</td><td>67.49</td><td>76.91</td><td>81.81</td><td>58.44</td><td>81.06</td><td>88.28</td><td>91.66</td></tr><tr><td>Long-CLIP</td><td>40.83</td><td>66.36</td><td>76.22</td><td>81.51</td><td>57.24</td><td>80.42</td><td>87.54</td><td>90.96</td></tr><tr><td>GOAL</td><td>38.86</td><td>64.36</td><td>74.15</td><td>79.58</td><td>59.28</td><td>81.02</td><td>87.70</td><td>90.88</td></tr><tr><td>FAST-GOAL</td><td>42.81</td><td>68.61</td><td>78.10</td><td>82.92</td><td>61.00</td><td>82.84</td><td>89.24</td><td>92.34</td></tr><tr><td rowspan="6">Flickr30k</td><td>CLIP</td><td rowspan="6">ViT-B/16</td><td>63.20</td><td>86.30</td><td>92.38</td><td>94.38</td><td>82.90</td><td>97.20</td><td>98.90</td><td>99.40</td></tr><tr><td>EVA-CLIP</td><td>71.82</td><td>91.38</td><td>95.48</td><td>96.78</td><td>87.80</td><td>98.30</td><td>99.20</td><td>99.40</td></tr><tr><td>Long-CLIP</td><td>70.80</td><td>90.68</td><td>94.74</td><td>96.58</td><td>85.90</td><td>98.50</td><td>99.50</td><td>99.80</td></tr><tr><td>FineCLIP</td><td>71.16</td><td>91.42</td><td>95.40</td><td>96.92</td><td>84.60</td><td>97.00</td><td>98.90</td><td>99.40</td></tr><tr><td>GOAL</td><td>68.32</td><td>89.30</td><td>93.74</td><td>95.62</td><td>85.10</td><td>96.70</td><td>98.60</td><td>99.10</td></tr><tr><td>FAST-GOAL</td><td>72.50</td><td>91.34</td><td>95.02</td><td>96.70</td><td>86.80</td><td>97.50</td><td>98.90</td><td>99.80</td></tr></table>

TABLE V ABLATION STUDY COMPARING LISM AND FLISM PROCESSING EFFICIENCY AND EVALUATING THE EFFECTIVENESS OF GLIT100K FINE-TUNING ON DOCCI AND DCI DATASETS USING VIT-B/16 BACKBONE. FAST-GOAL† DENOTES DIRECT FINE-TUNING ON TARGET DATASET, WHILE FAST-GOAL†† DENOTES FINE-TUNING ON GLIT100K FIRST, FOLLOWED BY ADDITIONAL FINE-TUNING ON TARGET DATASET. THE BEST PERFORMANCE IS SHOWN IN BOLD, AND THE SECOND-BEST PERFORMANCE IS UNDERLINE. 

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Preprocessing Method</td><td rowspan="2">Preprocessing Time (min)</td><td colspan="4">Text to Image Recall@K</td><td colspan="4">Image to Text Recall@K</td></tr><tr><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td></tr><tr><td colspan="11">DOCCI</td></tr><tr><td>GOAL</td><td>LISM</td><td>1239.7</td><td>81.12</td><td>97.12</td><td>99.71</td><td>99.88</td><td>81.00</td><td>97.39</td><td>99.75</td><td>99.94</td></tr><tr><td>FAST-GOAL $^{\dagger}$ </td><td>FLISM</td><td>97.63</td><td>81.65</td><td>97.12</td><td>99.76</td><td>99.96</td><td>80.71</td><td>97.20</td><td>99.75</td><td>99.98</td></tr><tr><td>FAST-GOAL $^{\dagger\dagger}$ </td><td>FLISM</td><td>97.63</td><td>84.16</td><td>97.69</td><td>99.76</td><td>99.92</td><td>83.16</td><td>97.49</td><td>99.78</td><td>99.94</td></tr><tr><td colspan="11">DCI</td></tr><tr><td>GOAL</td><td>LISM</td><td>407.71</td><td>73.19</td><td>89.14</td><td>95.50</td><td>97.25</td><td>73.04</td><td>90.10</td><td>96.80</td><td>98.15</td></tr><tr><td>FAST-GOAL $^{\dagger}$ </td><td>FLISM</td><td>34.44</td><td>72.99</td><td>89.64</td><td>95.55</td><td>97.20</td><td>72.44</td><td>89.69</td><td>96.35</td><td>98.10</td></tr><tr><td>FAST-GOAL $^{\dagger\dagger}$ </td><td>FLISM</td><td>34.44</td><td>78.29</td><td>91.00</td><td>96.50</td><td>97.80</td><td>77.34</td><td>91.05</td><td>97.55</td><td>98.60</td></tr></table>

# C. Ablation Study

FLISM and GLIT100k. Table V presents our ablation studies examining both the computational efficiency of our proposed FLISM compared to the LISM from GOAL [18], and the effectiveness of our GLIT100k dataset for training. We evaluate three different approaches: GOAL using the LISM with fine-tuning on the target dataset, FAST-GOAL† using our FLISM with direct fine-tuning only on the target dataset, and FAST-GOAL†† using our FLISM with two-stage fine-tuning where we first fine-tune on our GLIT100k dataset followed by further fine-tuning on the target dataset. For experimental settings, we conduct experiments on both DOCCI [14] and DCI [15] datasets. We randomly sample 2k images as the test set for DCI, and train all models with a batch size of 64 for fair comparison. Note that all approaches are trained and evaluated on the same target dataset, representing noncross-dataset evaluation where models are fine-tuned on the specific domain they are evaluated on.

Our experimental results demonstrate both the computational efficiency of FLISM and the effectiveness of our GLIT100k dataset. The processing time analysis reveals remarkable efficiency gains when comparing the LISM of GOAL with the FLISM of FAST-GOAL† across both datasets. Specifically, the preprocessing time measures the duration required to process the target dataset. On the DOCCI dataset, FAST-GOAL† requires only 97.63 minutes compared to the 1239.7 minutes required by GOAL, representing a speedup of approximately 12.7×. Similarly, for DCI dataset processing, FAST-GOAL† achieves substantial efficiency improvements, requiring only 34.44 minutes compared to the 407.71 minutes required by GOAL, representing a speedup of approximately 11.8×. Despite these dramatic time reductions, FAST-GOAL† maintains comparable performance to GOAL, demonstrating that our FLISM can achieve significant computational efficiency without sacrificing performance quality compared to the LISM. This comparable performance validates that YOLO [35]-based object detection effectively identifies semantically important regions for local alignment without requiring SAM’s [36] category-agnostic segmentation capability.

TABLE VILOSS FUNCTION ABLATION STUDY ON DOCCI AND MSCOCO DATASETS USING VIT-B/16 BACKBONE. RESULTS DEMONSTRATE THE EFFECTIVENESSOF COMBINING GLOBAL, LOCAL, AND TSL LOSS COMPONENTS ACROSS DIFFERENT TEXT COMPLEXITY DOMAINS. THE BEST PERFORMANCE IS SHOWNIN BOLD, AND THE SECOND-BEST PERFORMANCE IS UNDERLINED.

<table><tr><td rowspan="2">Methods</td><td colspan="3">Loss Components</td><td colspan="4">Text to Image Recall@K</td><td colspan="4">Image to Text Recall@K</td></tr><tr><td>Global</td><td>Local</td><td>TSL</td><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td></tr><tr><td colspan="12">DOCCI (Long Caption Domain)</td></tr><tr><td>Global fine-tuning</td><td>√</td><td></td><td></td><td>76.02</td><td>95.25</td><td>99.47</td><td>99.84</td><td>75.51</td><td>94.61</td><td>99.29</td><td>99.76</td></tr><tr><td>Local fine-tuning</td><td></td><td>√</td><td></td><td>70.33</td><td>92.73</td><td>99.00</td><td>99.70</td><td>71.67</td><td>92.76</td><td>99.00</td><td>99.71</td></tr><tr><td>w/o TSL</td><td>√</td><td>√</td><td></td><td>78.27</td><td>96.27</td><td>99.73</td><td>99.92</td><td>78.37</td><td>96.20</td><td>99.61</td><td>99.84</td></tr><tr><td>FAST-GOAL</td><td>√</td><td>√</td><td>√</td><td>79.06</td><td>96.31</td><td>99.65</td><td>99.94</td><td>78.94</td><td>96.35</td><td>99.57</td><td>99.88</td></tr><tr><td colspan="12">MSCOCO (Short Caption Domain)</td></tr><tr><td>Global fine-tuning</td><td>√</td><td></td><td></td><td>37.39</td><td>62.70</td><td>85.44</td><td>92.73</td><td>55.66</td><td>79.02</td><td>93.60</td><td>97.06</td></tr><tr><td>Local fine-tuning</td><td></td><td>√</td><td></td><td>37.91</td><td>63.25</td><td>85.66</td><td>92.76</td><td>56.64</td><td>79.82</td><td>94.50</td><td>97.76</td></tr><tr><td>w/o TSL</td><td>√</td><td>√</td><td></td><td>39.04</td><td>64.46</td><td>86.56</td><td>93.47</td><td>57.30</td><td>80.94</td><td>95.38</td><td>98.10</td></tr><tr><td>FAST-GOAL</td><td>√</td><td>√</td><td>√</td><td>39.38</td><td>64.75</td><td>86.75</td><td>93.71</td><td>59.62</td><td>82.18</td><td>94.64</td><td>97.86</td></tr></table>

Building on this efficiency foundation, the comparison between FAST-GOAL† and FAST-GOAL†† reveals the substantial benefits of fine-tuning on our GLIT100k dataset followed by further fine-tuning on target datasets. FAST-GOAL† achieves comparable performance to GOAL with significantly less processing time through direct target dataset fine-tuning. More impressively, FAST-GOAL†† demonstrates even greater advantages by incorporating an additional training stage on GLIT100k. Through this two-stage training approach, FAST-GOAL†† achieves significantly superior performance while still maintaining the computational efficiency of FLISM. Specifically, FAST-GOAL†† achieves 84.16% text-to-image R@1 compared to the 81.65% achieved by FAST-GOAL† and the 81.12% achieved by GOAL, while maintaining the same 12.7× speedup over preprocessing time of GOAL. On DCI, FAST-GOAL†† attains 78.29% compared to the 72.99% achieved by FAST-GOAL† and the 73.19% achieved by GOAL while maintaining the same 11.8× speedup over GOAL’s LISM method. These consistent improvements across different datasets confirm that our GLIT100k enables models to learn more generalizable fine-grained image-text alignments. The key to this effectiveness lies in GLIT100k’s unique combination of global and context-derived local pairs, which preserves natural relationships between local descriptions and overall scene narratives. This enhanced learning effectively transfers to various long caption domains.

Loss Function. We conduct a ablation study to analyze the contribution of each loss component in our FAST-GOAL framework. Table VI compares different combinations of our proposed loss functions: Global loss, Local loss, and Token Similarity-based Learning (TSL) loss, evaluated on both long caption (DOCCI) [14] and short caption (MSCOCO) [16] domains using ViT-B/16 [42] backbone. For DOCCI evaluation, we train and test models on the same dataset, while MSCOCO results represent zero-shot evaluation performance.

The results demonstrate that our complete FAST-GOAL method, utilizing all three loss components, achieves the best performance across both domains. Interestingly, we observe domain-specific behaviors that validate our approach design. On the DOCCI dataset, which contains lengthy and detailed descriptions, the global fine-tuning method shows stronger baseline performance compared to local fine-tuning, achieving 76.02% versus 70.33% on text-to-image R@1 and 75.51% versus 71.67% on image-to-text R@1. This indicates that comprehensive textual understanding benefits from global semantic alignment. Conversely, on MSCOCO with its shorter captions compared to DOCCI, local fine-tuning demonstrates relatively better performance with 37.91% versus 37.39% on text-to-image R@1 and 56.64% versus 55.66% on image-totext R@1, indicating that local alignment correspondences are more critical when dealing with concise descriptions.

Most importantly, the combination of global and local losses with our proposed TSL component consistently outperforms individual loss functions across both long and short caption domains. On DOCCI, FAST-GOAL achieves 79.06% text-toimage R@1 and 78.94% image-to-text R@1, showing improvements over the w/o TSL baseline of 78.27% and 78.37% respectively. Similarly, on MSCOCO, FAST-GOAL demonstrates significant gains with 39.38% text-to-image R@1 and 59.62% image-to-text R@1 compared to 39.04% and 57.30% without TSL. This validates our hypothesis that effective multi-modal learning requires both global semantic understanding and fine-grained local correspondences. TSL serves as a mechanism for propagating attention signals from local elements. The consistent improvements demonstrate the generalizability of our approach across varying caption lengths.

TABLE VII ABLATION STUDY ON SPATIAL PARTITIONING STRATEGY USING VIT-B/16 BACKBONE. RESULTS DEMONSTRATE THAT COMBINING YOLO DETECTIONS WITH SPATIAL PARTITIONS CONSISTENTLY IMPROVES PERFORMANCE ACROSS ALL DATASETS AND METRICS. THE BEST PERFORMANCE IS SHOWN IN BOLD. 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Region Extraction</td><td colspan="4">Text to Image Recall@K</td><td colspan="4">Image to Text Recall@K</td></tr><tr><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td></tr><tr><td rowspan="2">DOCCI</td><td>YOLO only</td><td>69.41</td><td>91.39</td><td>98.47</td><td>99.45</td><td>71.88</td><td>92.82</td><td>98.73</td><td>99.61</td></tr><tr><td>YOLO + Spatial</td><td>74.27</td><td>93.76</td><td>99.27</td><td>99.71</td><td>74.67</td><td>94.71</td><td>99.24</td><td>99.75</td></tr><tr><td rowspan="2">DCI</td><td>YOLO only</td><td>53.21</td><td>73.40</td><td>85.99</td><td>89.79</td><td>53.91</td><td>73.90</td><td>87.33</td><td>91.29</td></tr><tr><td>YOLO + Spatial</td><td>56.62</td><td>75.26</td><td>87.75</td><td>91.08</td><td>56.74</td><td>75.99</td><td>88.71</td><td>92.46</td></tr><tr><td rowspan="2">MSCOCO</td><td>YOLO only</td><td>39.89</td><td>65.52</td><td>87.08</td><td>93.62</td><td>58.04</td><td>80.44</td><td>94.58</td><td>98.18</td></tr><tr><td>YOLO + Spatial</td><td>42.81</td><td>68.61</td><td>88.54</td><td>94.64</td><td>61.00</td><td>82.84</td><td>95.34</td><td>98.32</td></tr><tr><td rowspan="2">Flickr30k</td><td>YOLO only</td><td>69.22</td><td>89.44</td><td>96.76</td><td>98.00</td><td>85.30</td><td>96.80</td><td>99.90</td><td>99.90</td></tr><tr><td>YOLO + Spatial</td><td>72.50</td><td>91.34</td><td>97.92</td><td>98.80</td><td>86.80</td><td>97.50</td><td>100.00</td><td>100.00</td></tr></table>

TABLE VIII ABLATION STUDY COMPARING SINGLE HIGHEST-SIMILARITY PAIR VERSUS TOP-3 PAIRS MATCHING STRATEGIES USING VIT-B/16 BACKBONE. TOP-3 (UNIFORM) APPLIES EQUAL WEIGHT TO ALL THREE PAIRS, WHILE TOP-3 (WEIGHTED) APPLIES COSINE SIMILARITY-BASED WEIGHTS. THE BEST PERFORMANCE IS SHOWN IN BOLD. 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Matching Strategy</td><td colspan="4">Text to Image Recall@K</td><td colspan="4">Image to Text Recall@K</td></tr><tr><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td></tr><tr><td rowspan="3">DOCCI</td><td>Top-3 (uniform)</td><td>71.51</td><td>92.57</td><td>99.02</td><td>99.71</td><td>73.04</td><td>93.86</td><td>99.10</td><td>99.75</td></tr><tr><td>Top-3 (weighted)</td><td>71.75</td><td>92.49</td><td>99.00</td><td>99.71</td><td>73.04</td><td>93.82</td><td>99.18</td><td>99.75</td></tr><tr><td>Single pair (FAST-GOAL)</td><td>74.27</td><td>93.76</td><td>99.27</td><td>99.71</td><td>74.67</td><td>94.71</td><td>99.24</td><td>99.75</td></tr><tr><td rowspan="3">DCI</td><td>Top-3 (uniform)</td><td>55.06</td><td>74.34</td><td>87.21</td><td>90.96</td><td>54.72</td><td>74.74</td><td>88.53</td><td>92.45</td></tr><tr><td>Top-3 (weighted)</td><td>55.25</td><td>74.43</td><td>87.15</td><td>90.96</td><td>54.87</td><td>74.77</td><td>88.56</td><td>92.41</td></tr><tr><td>Single pair (FAST-GOAL)</td><td>56.62</td><td>75.26</td><td>87.75</td><td>91.08</td><td>56.74</td><td>75.99</td><td>88.71</td><td>92.46</td></tr><tr><td rowspan="3">MSCOCO</td><td>Top-3 (uniform)</td><td>41.18</td><td>67.00</td><td>87.38</td><td>93.93</td><td>57.18</td><td>80.08</td><td>94.70</td><td>97.62</td></tr><tr><td>Top-3 (weighted)</td><td>41.24</td><td>66.99</td><td>87.38</td><td>93.94</td><td>57.52</td><td>80.06</td><td>94.66</td><td>97.66</td></tr><tr><td>Single pair (FAST-GOAL)</td><td>42.81</td><td>68.61</td><td>88.54</td><td>94.64</td><td>61.00</td><td>82.84</td><td>95.34</td><td>98.32</td></tr><tr><td rowspan="3">Flickr30k</td><td>Top-3 (uniform)</td><td>70.14</td><td>90.40</td><td>97.36</td><td>98.56</td><td>84.50</td><td>97.40</td><td>99.90</td><td>100.00</td></tr><tr><td>Top-3 (weighted)</td><td>70.60</td><td>90.32</td><td>97.40</td><td>98.58</td><td>84.80</td><td>97.10</td><td>99.90</td><td>100.00</td></tr><tr><td>Single pair (FAST-GOAL)</td><td>72.50</td><td>91.34</td><td>97.92</td><td>98.80</td><td>86.80</td><td>97.50</td><td>100.00</td><td>100.00</td></tr></table>

Spatial Partitioning. To validate the contribution of spatial partitioning in our FLISM pipeline, we conduct ablation experiments comparing YOLO [35]-only extraction against our combined approach that incorporates both YOLO detections and spatial partitions (quadrant and center regions). Table VII presents comprehensive results across all evaluation datasets using ViT-B/16 backbone.

The results demonstrate that spatial partitioning provides consistent and substantial improvements across all datasets and metrics. On long caption datasets, adding spatial partitions to YOLO detections yields significant performance gains. Specifically, on DOCCI, the combined approach achieves 74.27% text-to-image R@1 compared to the 69.41% achieved by YOLO-only extraction, representing a substantial improvement. Similarly, on DCI, our method attains 56.62% compared to the 53.21% achieved with YOLO detections alone. These improvements extend to short caption datasets as well, with MSCOCO achieving 42.81% text-to-image R@1 compared to 39.89% from YOLO-only approach, and Flickr30k attaining 72.50% compared to 69.22%. The consistent performance gains across both long and short caption domains validate the effectiveness of our combined extraction strategy.

These consistent improvements across diverse datasets demonstrate that spatial partitions capture semantically relevant information that effectively complements object-centric regions detected by YOLO. The heuristic partitions serve two critical functions: First, they ensure comprehensive spatial coverage of the image, capturing regions that may not contain distinct objects but are nonetheless described in the captions. Second, they help capture contextual relationships between objects and their spatial arrangements, which are often crucial for understanding detailed scene descriptions. Rather than introducing noise, these spatial partitions provide valuable supervision signals that enhance the model’s ability to learn fine-grained global-local alignments. The consistent performance gains across both long and short caption domains validate the effectiveness of our combined extraction strategy. Matching Strategy. To validate our design choice of selecting the single highest-similarity pair in FLISM, we conduct experiments comparing it against Top-k matching strategies. Table VIII presents comprehensive results across all evaluation datasets using ViT-B/16 backbone. We implement and evaluate two top-3 matching strategies: uniform weighting where all three pairs contribute equally to the TSL loss during training, and similarity-proportional weighting where each pair’s contribution is scaled by its matching confidence score.

![](images/8798ef21f93d4668dfa6bf053258e4bd943844c55685df3cc4291f4ca6363f91.jpg)  
Fig. 5. Image encoder attention map visualization comparison between original CLIP and FAST-GOAL. Each row shows the input image, CLIP attention map, and FAST-GOAL attention map. Brighter regions indicate stronger attention weights within the image encoder’s self-attention mechanism.

The results consistently demonstrate that single pair selection outperforms both Top-3 strategies across all datasets and metrics. On DOCCI, single pair achieves 74.27% text-toimage R@1 compared to the 71.75% achieved by weighted matching and the 71.51% achieved by uniform matching. On DCI, single pair attains 56.62% compared to the 55.25% from weighted matching and the 55.06% from uniform matching. Similar patterns are observed on MSCOCO, where single pair achieves 42.81% compared to 41.24% and 41.18% from the top-3 approaches, and on Flickr30k with 72.50% compared to 70.60% and 70.14%.

These consistent improvements across diverse datasets validate that the highest-similarity pair effectively captures the most semantically relevant correspondence between image regions and text segments. Our single pair selection strategy focuses training on the strongest matches, enabling the model to learn from clear and unambiguous region-sentence associations. This focused approach proves more effective than distributing attention across multiple pairs, where varying similarity scores can affect the quality of supervision signals. Beyond performance advantages, single pair selection offers superior computational efficiency by computing TSL loss for only one pair instead of three, reducing memory consumption and training time while achieving better accuracy. This validates single pair selection as the optimal design choice for our FLISM pipeline.

# D. Qualitative Results

Attention Map Visualization. Figure 5 presents side-by-side comparisons of how the image encoder attends to different regions within the same image. To understand how FAST-GOAL affects the internal attention mechanisms of the image encoder, we visualize and compare attention maps from the original CLIP [1] model and our FAST-GOAL. We analyze the image encoder’s self-attention patterns using the top 3 principal components [44] to capture the most significant attention variations across image patches.

The attention visualizations show differences in visual processing. In the first row, CLIP produces weak attention across the entire image without focusing on specific buildings. In contrast, FAST-GOAL concentrates attention on the building and skyscrapers (purple box). In the second row, CLIP’s attention scatters across the room without recognizing either the TV content or the cat below. FAST-GOAL, however, primarily attends to the mountain landscape on the screen while also capturing the cat (purple box). These distinct attention patterns reveal how our global-local alignment fundamentally changes the model’s visual processing capabilities.

The enhanced attention patterns in FAST-GOAL indicate that our method has learned to allocate stronger focus on visually salient elements that are more likely to be described in detailed textual descriptions. This enhanced attention behavior stems from our global-local alignment training strategy. Through training with our GLIT100k dataset and the TSL method, the image encoder learns to develop more discriminative internal representations that emphasize visually salient elements. This training process naturally guides the model to focus effectively on regions that are most relevant for comprehensive visual understanding. These attention visualizations demonstrate that FAST-GOAL not only improves textimage matching performance but also fundamentally enhances the ability of the image encoder to identify and emphasize semantically relevant visual content, leading to more robust visual representations specifically optimized for lengthy text understanding tasks.

Retrieval Results. To provide intuitive insights into the effectiveness of FAST-GOAL, we present qualitative comparisons of text-to-image retrieval results using lengthy text queries. Figure 6 shows representative examples where we compare the top-1 retrieved images from CLIP [1], Long-CLIP [22], and our FAST-GOAL method. The qualitative results reveal important insights about the capabilities of different methods when handling lengthy and detailed text descriptions.

CLIP and Long-CLIP demonstrate reasonable semantic understanding by retrieving images that contain some relevant

There is a group of seven people walking past a business. The building is made up of grey bricks and has two doors. Above each of the doors is'a.sign reading Coronas Cafe. The letering of the signs is in mint green and gold and the background of the signs are neutral'in color and have'a. fleur-de-lis pattern.There are windows allowing for the interior of thebuilding to be seen.There is an advertisement to the left of the business with a woman in black with blonde hair on it.To the right of the business is another advertisement, but for the business.This sign shows an ice cream cone with three scoops of ice cream on a white background.

![](images/2226aac711b0da358a2f207091b2c0a16aa07ccbdecf1e5d439d9ae363531f1f.jpg)  
CLIP X

![](images/330535a75757fe1c84c5a256f8dffe215d748d76db93957e00918d1f53d4917b.jpg)  
Long-CLIP

![](images/72ea36d1af8f0371627cbfa63d21806fe5c204d56787b711802e88347d2f90a9.jpg)

<details>
<summary>text_image</summary>

Coca-Cola
TOMOT
</details>

FAST-GOAL

![](images/25a9fe29a26996acc7310e138ee450c0aa88096b8faf0755bed954aff1c06f28.jpg)

This is a yellow building that stands near a garden. The garden is filled with yellow flowers with tinges of pink on them.The flowers have long green stems and leaves.On the other side of the flowers and closer to the building is a parking lot.There are a few vehicles there including a green oryellowish colored SUV that isparked with·its back toward the flowers.Past the parking lot is the yellow building. The yellow building is long and at least three stories tall.There are many windows on the building·and thereis a balcony on the highest floor.The balcony is surrounded bya metal fence thatis black.The roof of the building is a gray color.There are a couple of large white chimneys on the roof area on the left.To the right of the building are several trees of different types.The sky behind the building ranges from light blue on the left to stormy and ominous on the right.The sky just past the trees is a dark blue color and cloudy as well.

![](images/61694e400a5ab8fa3f9676be677d8c27147a440b209ebeb3fe071f053bf14489.jpg)  
CLIP X

![](images/65330ccfb6962637994bf8ec9a53a143417d95d06b9688f35cf47674eaa35c31.jpg)  
Long-CLIPX

![](images/d4dd8986e6f82a10617827e9aea46f68a2eeb5c8f1d7341d0937d3899876e8a7.jpg)  
FAST-GOAL

![](images/9d8187811411996bf13532801f15f688a2122066c1f413834e24bbef55a1bc7c.jpg)  
Fig. 6. Qualitative comparison of text-to-image retrieval results. Given lengthy text queries, we show the top-1 retrieved images from CLIP, Long-CLIP, and FAST-GOAL. The ground truth images demonstrate the target visual content that should be retrieved. Red boxes indicate incorrect retrievals, while green boxes show correct matches.

visual elements mentioned in the query. However, they often fail to capture the fine-grained details described in the lengthy captions. For example, in the first query describing a group of seven people walking past a business with specific details about Coronas Cafe signage and grey brick building, we observe notable differences in retrieval accuracy. CLIP retrieves a completely irrelevant image, while Long-CLIP finds a similar image with people walking past a business but fails to capture the specific context and precise details such as the exact number of people and the distinctive cafe signage.

Similarly, for the second query about “yellow flowers with tinges of pink” and “yellow building is long”, we observe that both baseline methods struggle with capturing the complete scene description. CLIP retrieves an image with yellow flowers and a building but does not precisely match the detailed description provided in the text, while Long-CLIP shows a building but lacks the distinctive yellow flowers described in the query.

In contrast, FAST-GOAL consistently retrieves more accurate images that closely match the comprehensive descriptions provided in the queries. For the first query, FAST-GOAL successfully identifies the correct image showing people walking past the grey brick building with the distinctive cafe signage. For the second query, FAST-GOAL retrieves an image that contains both the yellow building and the field of yellow flowers with pink tinges, demonstrating its ability to understand and match multiple detailed elements simultaneously. This superior performance stems from our method’s enhanced ability to learn fine-grained correspondences between local image regions and specific textual phrases through the FLISM pipeline and TSL mechanism.

These qualitative findings complement our quantitative re-

![](images/1f8c09db43cc8126fc7760acd4112ae8e22ab4d4b489f85be2a4b96b1847e9bf.jpg)

<details>
<summary>natural_image</summary>

Close-up of a black car wheel rim and alloy wheel, parked on cobblestone pavement (no visible text or symbols)
</details>

# Ground Truth Caption

There is ablack car that you can see the side of and a litle bit of the front. The ground has gray tile with some grass. There are two cars in the background.There isaperson in the background.It is daytime.

# Predicted Caption

A portion of one of the sides of a black car that can be seentakingup most of the image.The carhas a silver bumper that curves,and a circular silver piece above the bumper on the right hand side.There is a black tire that hasan intricatesilverrim onit.Therim has thin, cylindrical poles that go at angles around it.Reflections of a stone floor,people,buildings,and the sky can be seen on the surface of the car.A few of the people can be seen in the reflections holding up phones and pointing the phones at the car,as if to takeapicture.There is a rectangular whiteand brown sign that has illegible blue lettering printed across the top of it in the background and to the right of the car.The car isparked ona surface that is made out of unevenly shaped stones and concrete slabs.

![](images/a4a97cc31dc54c3de9d164481f1d4a0180b7a4182aa8f912212cb65549dff465.jpg)

<details>
<summary>natural_image</summary>

Street view of a narrow alley with colorful buildings and people on a vehicle (no visible text or signage)
</details>

# Ground Truth Caption

This is the scene ofa life inacity. There is a roadway where there are a couple of vehicles parked,but none are traveling.Sidewalks line the roadway on both sides.On the right side,the buildings are mostly white or lighter in color. The buildings on the left side vary more in color. Some are white as well,but there is also a green building and some are blue.Near the green building,there is a woman wearing pink pants standing in the doorway. She has her hands on her hips.There is a man wearing a red shirt standing near her.Another person is nearby as well. There is a vehicle parked on the curb near these people. Behind thisvehicle isa bicycle.

# Predicted Caption

Several people can be seen walking along a street.The street starts in the foreground and extends into the background.There isa bike cart on the right side of the street. There are sidewalksand rows of buildings on both sides of the street.The buildings have exteriors thatare dirty and chipped in many spots.The buildings also have balconies along their upper floors.A large flag of the country of Cuba is suspended in the air over the street.There are more buildings at the far end of the stret. It is daytime,and the sky above is appearingas a light blue color.

Fig. 7. Representative error cases from DCI [15] image-to-text retrieval. The predicted captions describe the images with high semantic similarity to ground truth, demonstrating the challenge of discriminating among multiple detailed descriptions that share substantial semantic overlap.

sults, providing visual evidence that the improvements of FAST-GOAL in recall metrics translate to substantially enhanced retrieval quality in practical scenarios. The results demonstrate that while traditional methods achieve reasonable semantic alignment at a coarse level, the global-local learning approach of FAST-GOAL enables more precise understanding of fine-grained visual-textual relationships. This capability makes our method particularly effective for applications requiring accurate retrieval based on comprehensive and detailed descriptions.

Error Analysis. To provide deeper insights into the challenges of lengthy text retrieval, we analyze representative error cases from our experiments. Figure 7 presents retrieval errors from the DCI [15] dataset, illustrating the inherent difficulty of fine-grained discrimination in long caption retrieval tasks. The examples reveal the primary challenge: the text gallery contains multiple highly similar descriptions that differ only in subtle details.

In the first example, both the ground truth and predicted captions describe a black car with detailed observations about the surroundings. However, subtle differences in specific details such as tire rim descriptions and parking surface materials lead to retrieval errors. Similarly, in the second example, both captions describe a city street scene with buildings and people, yet differ in specific architectural details including building colors and balconies, as well as scene elements such as the bike cart. These errors occur because DCI captions provide detailed descriptions where multiple texts can semantically match the same image at a high level. The gallery contains descriptions that share substantial semantic overlap, creating challenging retrieval scenarios that require capturing every fine-grained detail for correct discrimination.

This analysis reveals important characteristics of lengthy text retrieval tasks. Unlike traditional short caption retrieval where semantic differences are often clear and substantial, lengthy descriptions introduce scenarios where multiple captions can accurately describe the same visual content with only minor variations in specific details. Successfully discriminating among such highly similar descriptions requires enhanced capability in understanding subtle semantic differences, presenting opportunities for future research directions in finegrained multi-modal understanding.

# VI. CONCLUSION

In this paper, we have presented FAST-GOAL, a fast and efficient fine-tuning approach for enhancing the lengthy text understanding capability of CLIP. Our method has introduced two key components: Fast Local Image-Sentence Matching (FLISM) for efficient local correspondence extraction and Token Similarity-based Learning (TSL) for propagating attention from local elements to global representations. Furthermore, we have constructed GLIT100k, a dataset that provides both global image-lengthy caption pairs and context-derived finegrained local pairs for multi-level supervision, enabling effective training without million-scale data requirements. By addressing the computational bottlenecks that have limited previous approaches, FAST-GOAL enables practical adaptation of vision-language models to detailed textual descriptions at scale efficiently.

# VII. ACKNOWLEDGMENT

This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT)(RS-2024-00355008), the MSIT(Ministry of Science and ICT), Korea, under the Graduate School of Virtual Convergence support program(IITP-2024-RS-2024- 00418847) supervised by the IITP(Institute for Information & Communications Technology Planning & Evaluation), and Culture, Sports and Tourism R&D Program through the Korea Creative Content Agency grant funded by Ministry of Culture, Sports and Tourism in 2024 (Project Name : Developing Professionals for R&D in Contents Production Based on Generative Ai and Cloud, Project Number : RS-2024-00352578, Contribution Rate: 30%).

# REFERENCES

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual representations from natural language supervision,” in Proceedings of the International Conference on Machine Learning, 2021, pp. 8748–8763.   
[2] J. Li, R. Selvaraju, A. Gotmare, S. Joty, C. Xiong, and S. C. H. Hoi, “Align before fuse: Vision and language representation learning with momentum distillation,” in Advances in Neural Information Processing Systems, 2021, pp. 9694–9705.   
[3] X. Zhai, X. Wang, B. Mustafa, A. Steiner, D. Keysers, A. Kolesnikov, and L. Beyer, “LiT: Zero-shot transfer with locked-image text tuning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 18 123–18 133.   
[4] S. Lavoie, P. Kirichenko, M. Ibrahim, M. Assran, A. G. Wilson, A. Courville, and N. Ballas, “Modeling caption diversity in contrastive vision-language pretraining,” in Proceedings of the International Conference on Machine Learning, 2024, pp. 26 070–26 084.   
[5] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 11 975–11 986.   
[6] A. Frome, G. S. Corrado, J. Shlens, S. Bengio, J. Dean, M. Ranzato, and T. Mikolov, “DeViSE: A deep visual-semantic embedding model,” in Advances in Neural Information Processing Systems, 2013, pp. 2121– 2129.   
[7] A. Karpathy and L. Fei-Fei, “Deep visual-semantic alignments for generating image descriptions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2015, pp. 3128–3137.   
[8] H. Kim, J. Shin, S. Park, and J. Paik, “Intelligent video surveillance system with abnormal behavior recognition and metadata retrieval,” 2023.   
[9] R. Zhao, “Tourism image retrieval method based on deep residual shrinkage network,” IEIE Transactions on Smart Processing & Computing, vol. 14, no. 3, pp. 419–429, 2025.   
[10] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” in Advances in Neural Information Processing Systems, 2012, pp. 1097–1105.   
[11] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in Proceedings of the International Conference on Learning Representations, 2014.

[12] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the inception architecture for computer vision,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2016, pp. 2818–2826.   
[13] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.   
[14] Y. Onoe, S. Rane, Z. Berger, Y. Bitton, J. Cho, R. Garg, A. Ku, Z. Parekh, J. Pont-Tuset, G. Tanzer et al., “DOCCI: Descriptions of connected and contrasting images,” in Proceedings of the European Conference on Computer Vision, 2024, pp. 291–309.   
[15] J. Urbanek, F. Bordes, P. Astolfi, M. Williamson, V. Sharma, and A. Romero-Soriano, “A picture is worth more than 77 text tokens: Evaluating CLIP-style models on dense captions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26 700–26 709.   
[16] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft COCO: Common objects in ´ context,” in Proceedings of the European Conference on Computer Vision, 2014, pp. 740–755.   
[17] P. Young, A. Lai, M. Hodosh, and J. Hockenmaier, “From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions,” in Transactions of the Association for Computational Linguistics, 2014, pp. 67–78.   
[18] H. Choi, Y. K. Jang, and C. Eom, “GOAL: Global-local object alignment learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 4070–4079.   
[19] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.- H. Sung, Z. Li, and T. Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in Proceedings of the International Conference on Machine Learning, 2021, pp. 4904– 4916.   
[20] L. Yuan, D. Chen, Y.-L. Chen, N. Codella, X. Dai, J. Gao, H. Hu, X. Huang, B. Li, C. Li et al., “Florence: A new foundation model for computer vision,” in arXiv preprint arXiv:2111.11432, 2021.   
[21] C. Xie, B. Wang, F. Kong, J. Li, D. Liang, G. Zhang, D. Leng, and Y. Yin, “FG-CLIP: Fine-grained visual and textual alignment,” in Proceedings of the International Conference on Machine Learning, 2025.   
[22] B. Zhang, P. Zhang, X. Dong, Y. Zang, and J. Wang, “Long-CLIP: Unlocking the long-text capability of CLIP,” in Proceedings of the European Conference on Computer Vision, 2025, pp. 310–325.   
[23] L. Chen, J. Li, X. Dong, P. Zhang, C. He, J. Wang, F. Zhao, and D. Lin, “ShareGPT4V: Improving large multi-modal models with better captions,” in Proceedings of the European Conference on Computer Vision, 2024, pp. 370–387.   
[24] M. Asokan, K. Wu, and F. Albreiki, “FineLIP: Extending CLIP’s reach via fine-grained alignment with longer text inputs,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 14 495–14 504.   
[25] Z. Wang, Z. Fang, J. Wang, and Y. Yang, “ViTAA: Visual-textual attributes alignment in person search by natural language,” in Proceedings of the European Conference on Computer Vision, 2020, pp. 402–420.   
[26] W. Tan, C. Ding, J. Jiang, F. Wang, Y. Zhan, and D. Tao, “Harnessing the power of MLLMs for transferable text-to-image person ReID,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 17 127–17 137.   
[27] Z. Cui, J. Zhou, X. Wang, M. Zhu, and Y. Peng, “Learning continual compatible representation for re-indexing free lifelong person reidentification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16 614–16 623.   
[28] L. Zheng, H. Zhang, S. Sun, M. Chandraker, Y. Yang, and Q. Tian, “Person re-identification in the wild,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2017, pp. 1367–1376.   
[29] Z. Zhong, L. Zheng, D. Cao, and S. Li, “Re-ranking person reidentification with k-reciprocal encoding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2017, pp. 1318–1327.   
[30] H.-Y. Chen, Z. Lai, H. Zhang, X. Wang, M. Eichner, K. You, M. Cao, B. Zhang, Y. Yang, and Z. Gan, “Contrastive localized languageimage pre-training,” in Proceedings of the International Conference on Machine Learning, 2025.   
[31] M. Minderer, A. Gritsenko, and N. Houlsby, “Scaling open-vocabulary object detection,” in Advances in Neural Information Processing Systems, 2023, pp. 72 983–73 007.

[32] H. Zhang, P. Zhang, X. Hu, Y.-C. Chen, L. Li, X. Dai, L. Wang, L. Yuan, J.-N. Hwang, and J. Gao, “GLIPv2: Unifying localization and visionlanguage understanding,” in Advances in Neural Information Processing Systems, 2022, pp. 36 067–36 080.   
[33] D. Jing, X. He, Y. Luo, N. Fei, W. Wei, H. Zhao, Z. Lu et al., “FineCLIP: Self-distilled region-based CLIP for better fine-grained understanding,” in Advances in Neural Information Processing Systems, 2024, pp. 27 896–27 918.   
[34] Y. Li, Y. Li, Q.-S. Zeng, W. Wang, Q. Hou, and M.-M. Cheng, “Unbiased region-language alignment for open-vocabulary dense prediction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 23 795–23 805.   
[35] Y. Fang, B. Liao, X. Wang, J. Fang, J. Qi, R. Wu, J. Niu, and W. Liu, “You only look at one sequence: Rethinking transformer in vision through object detection,” in Advances in Neural Information Processing Systems, 2021, pp. 26 183–26 197.   
[36] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015–4026.   
[37] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee, “LLaVA-NeXT: Improved reasoning, OCR, and world knowledge,” 2024.   
[38] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in Advances in Neural Information Processing Systems, 2023, pp. 34 892– 34 916.   
[39] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang et al., “Qwen technical report,” in arXiv preprint arXiv:2309.16609, 2023.   
[40] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu et al., “InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 24 185–24 198.   
[41] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican et al., “Gemini: A family of highly capable multimodal models,” in arXiv preprint arXiv:2312.11805, 2023.   
[42] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proceedings of the International Conference on Learning Representations, 2021.   
[43] Q. Sun, Y. Fang, L. Wu, X. Wang, and Y. Cao, “EVA-CLIP: Improved training techniques for CLIP at scale,” in arXiv preprint arXiv:2303.15389, 2023.   
[44] H. Hotelling, “Analysis of a complex of statistical variables into principal components,” in Journal of Educational Psychology, 1933, pp. 417–441.

![](images/b52edc998b27301b83ba76ede21b9577e5e0d1e60d0331361ecbb3d7e6d513dc.jpg)

<details>
<summary>natural_image</summary>

Portrait photo of a person wearing a black sweater (no text or symbols visible)
</details>

Hyungyu Choi is a researcher at the Graduate School of Advanced Imaging Science, Multimedia, and Film (GSAIM) at Chung-Ang University in Seoul, Korea. He received his B.S. degree in Mathematics from Chung-Ang University in 2025. He is currently conducting research as a member of the Perceptual AI LAB under the supervision of Professor Chanho Eom. His research interests focus on vision-language models, particularly in multimodal understanding and cross-modal retrieval.

![](images/d5426561a2f4892f1742a0d434db69a0f2891f6db2b72a7f7c9b10c8947badcd.jpg)

Young Kyun Jang is a Research Scientist who received his Ph.D. in Electrical and Computer Engineering from Intelligent Signal Processing Lab of Seoul National University. His research interests include multi-modal representation learning, efficient retrieval, and foundational (large language) models.

![](images/1b2aeab585109a6baa7607180efc9ad5a14e37e4e67fa0e5a5259227415c6733.jpg)

<details>
<summary>natural_image</summary>

Portrait of a young man with short dark hair wearing a light gray top against a solid blue background (no text or symbols visible)
</details>

Chanho Eom is an Assistant Professor at the Graduate School of Advanced Imaging Science, Multimedia, and Film (GSAIM) at Chung-Ang University in Seoul, Korea. He received his B.S. and Ph.D. degrees in Electrical and Electronic Engineering from Yonsei University in 2017 and 2023, respectively. He previously worked as a researcher at the Samsung Advanced Institute of Technology (SAIT). His research interests include computer vision and deep learning, particularly in retrieval, person re-identification, and video analysis, both in theory and applications.

# Supplementary Material

TABLE IXCOMPLETE END-TO-END TIME COMPARISON ON DOCCI DATASET.FAST-GOAL ACHIEVES 12.7× FASTER PREPROCESSING AND 6.2-8.2×FASTER TOTAL TIME WHILE MAINTAINING IDENTICAL TRAININGEFFICIENCY.

<table><tr><td>Method</td><td>Preprocessing</td><td>Training</td><td>Total Time</td></tr><tr><td>GOAL</td><td>1,239.7 min</td><td>60-120 min</td><td>1,299.7-1,359.7 min</td></tr><tr><td>FAST-GOAL</td><td>97.63 min</td><td>60-120 min</td><td>157.63-217.63 min</td></tr><tr><td>Speedup</td><td>12.7×</td><td>1×</td><td>6.2-8.2×</td></tr></table>

TABLE XCOMPLETE PARAMETER COMPARISON. FAST-GOAL ACHIEVES 2.2×PARAMETER REDUCTION COMPARED TO GOAL WHILE MAINTAININGREGION EXTRACTION QUALITY.

<table><tr><td>Method</td><td>Image Encoder</td><td>Text Encoder</td><td>Region Extraction</td><td>Total Parameters</td></tr><tr><td>GOAL</td><td>149.6M (ViT-B/16)</td><td>149.6M (ViT-B/16)</td><td>641.1M (SAM)</td><td>940.3M</td></tr><tr><td>FAST-GOAL</td><td>149.6M (ViT-B/16)</td><td>149.6M (ViT-B/16)</td><td>127.8M (YOLOS)</td><td>426.9M</td></tr><tr><td>Reduction</td><td>1×</td><td>1×</td><td>5×</td><td>2.2×</td></tr></table>

# A. Computational Efficiency Analysis

End-to-End Time Efficiency. Table IX presents the complete end-to-end time comparison between GOAL [1] and FAST-GOAL, demonstrating where efficiency gains occur in our pipeline. The ”Fast” in FAST-GOAL refers specifically to preprocessing stage efficiency. FLISM preprocessing is performed before fine-tuning to generate local pairs from the dataset, while the actual training process follows the same architecture as GOAL. The results show that while both methods require identical training time of 60-120 minutes, FAST-GOAL achieves dramatic preprocessing time reduction from 1,239.7 minutes to 97.63 minutes on DOCCI [2], representing a 12.7× speedup. This translates to total time reduction from 1,299.7-1,359.7 minutes to 157.63-217.63 minutes, achieving a 6.2-8.2× overall speedup.

This efficiency advantage directly addresses the scalability bottleneck in long text understanding. The SAM [3]-based preprocessing in GOAL makes large-scale dataset construction prohibitively expensive, requiring approximately 1,378 hours (57 days) to process 100k images. In contrast, FAST-GOAL’s YOLO [4]-based approach enables construction of GLIT100k with 100,647 global-local pairs in practical timeframes, removing the computational bottleneck that constrained previous methods. Importantly, this faster preprocessing does not compromise quality, as demonstrated in our ablation studies where YOLO-based extraction achieves comparable performance to SAM-based approaches.

Parameter Efficiency and Architectural Design. Table X presents the complete parameter comparison between GOAL and FAST-GOAL. Our choice of YOLOS over SAM is strategically motivated by parameter efficiency. GOAL requires 940.3M total parameters, combining CLIP’s [5] image encoder (149.6M), text encoder (149.6M), and SAM for region extraction (641.1M). In contrast, FAST-GOAL requires only 426.9M parameters by replacing SAM with YOLOS (127.8M), achieving a 2.2× parameter reduction. This significant reduction, combined with object detection’s focused nature compared to dense segmentation, enables the 12.7× preprocessing speedup while maintaining semantic region identification quality.

# B. Additional Experimental Analysis

Experiments with ViT-L/14 Backbone. To demonstrate the scalability of our approach across different model sizes, we conduct comprehensive experiments using ViT-L/14 [6] backbone following the same evaluation protocol as our ViT-B/16 experiments. Tables XI and XII present the results on long caption and short caption domains, respectively. Table XI shows the performance on DOCCI [2] and DCI [7] datasets. FAST-GOAL consistently achieves superior performance across both datasets. On DOCCI, FAST-GOAL attains 78.55% text-to-image R@1 and 78.61% image-to-text R@1, demonstrating substantial improvements over available baselines. On DCI, FAST-GOAL achieves 59.73% text-toimage R@1 and 60.76% image-to-text R@1, outperforming Long-CLIP [8] with 54.31% and 46.53%, and GOAL [1] with 54.66% and 54.67%. These results validate that our method effectively scales to larger model architectures while maintaining its advantages in long-text understanding. Table XII demonstrates performance on MSCOCO [9] and Flickr30k [10] datasets using standard R@1/5/10/15 metrics. Despite being optimized for long-text scenarios, FAST-GOAL maintains competitive performance on short caption domains. On MSCOCO, FAST-GOAL achieves 48.53% text-to-image R@1 and 66.58% image-to-text R@1, comparable to EVA-CLIP [11] with 48.14% and 63.96%. On Flickr30k, FAST-GOAL attains 76.72% text-to-image R@1 and 92.50% imageto-text R@1. These results confirm that our approach demonstrates strong generalizability across different caption lengths and model scales, validating the robustness of our global-local alignment strategy.

Comparison with recent CLIP-based models. Table XIII presents performance comparisons with recent CLIP-based models that employ different data scales. This comparison highlights the data efficiency advantage of our approach, particularly on long caption understanding tasks. SigLIPv2 [12] and MobileCLIPv2 [13] are pretrained on massive-scale datasets of 10 billion and 12 billion samples respectively, representing data scales that are 100-120× larger than CLIP’s pretraining data. While these models demonstrate strong performance across various benchmarks, their reliance on such massive data requirements creates significant barriers for practical adoption and research accessibility. In contrast, FAST-GOAL builds upon standard pretrained CLIP trained on 400M samples and requires only an additional 100k samples from

TABLE XI PERFORMANCE COMPARISON ON LONG CAPTION DOMAIN DATASETS (DOCCI AND DCI) USING VIT-L/14 BACKBONE. RESULTS SHOW RECALL@K SCORES FOR TEXT TO IMAGE AND IMAGE TO TEXT RETRIEVAL TASKS. THE BEST PERFORMANCE IS SHOWN IN BOLD, AND THE SECOND-BEST PERFORMANCE IS UNDERLINED. 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="4">Text to Image Recall@K</td><td colspan="4">Image to Text Recall@K</td></tr><tr><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td></tr><tr><td rowspan="6">DOCCI</td><td>CLIP</td><td rowspan="5">ViT-L/14</td><td>60.51</td><td>86.31</td><td>96.67</td><td>98.45</td><td>57.75</td><td>83.53</td><td>95.33</td><td>97.73</td></tr><tr><td>EVA-CLIP</td><td>73.08</td><td>93.10</td><td>98.25</td><td>99.27</td><td>72.12</td><td>92.16</td><td>98.47</td><td>99.35</td></tr><tr><td>Long-CLIP</td><td>78.84</td><td>95.25</td><td>99.19</td><td>99.59</td><td>66.82</td><td>91.90</td><td>99.04</td><td>99.82</td></tr><tr><td>FineCLIP</td><td>66.29</td><td>88.37</td><td>97.24</td><td>98.75</td><td>56.24</td><td>82.75</td><td>95.43</td><td>97.90</td></tr><tr><td>GOAL</td><td>79.04</td><td>95.78</td><td>99.55</td><td>99.84</td><td>79.16</td><td>95.96</td><td>99.61</td><td>99.90</td></tr><tr><td>FAST-GOAL</td><td></td><td>78.55</td><td>95.86</td><td>99.51</td><td>99.80</td><td>78.61</td><td>96.24</td><td>99.67</td><td>99.90</td></tr><tr><td rowspan="6">DCI</td><td>CLIP</td><td rowspan="5">ViT-L/14</td><td>31.18</td><td>50.34</td><td>68.84</td><td>76.15</td><td>33.37</td><td>51.51</td><td>68.25</td><td>75.70</td></tr><tr><td>EVA-CLIP</td><td>45.88</td><td>64.96</td><td>79.56</td><td>84.87</td><td>44.70</td><td>64.75</td><td>79.94</td><td>85.41</td></tr><tr><td>Long-CLIP</td><td>54.31</td><td>72.53</td><td>85.04</td><td>89.06</td><td>46.53</td><td>72.64</td><td>86.48</td><td>90.38</td></tr><tr><td>FineCLIP</td><td>42.12</td><td>62.01</td><td>78.11</td><td>83.94</td><td>35.54</td><td>57.04</td><td>75.09</td><td>81.49</td></tr><tr><td>GOAL</td><td>54.66</td><td>74.57</td><td>86.29</td><td>90.54</td><td>54.67</td><td>74.86</td><td>87.12</td><td>90.94</td></tr><tr><td>FAST-GOAL</td><td></td><td>59.73</td><td>77.23</td><td>88.21</td><td>91.45</td><td>60.76</td><td>78.93</td><td>90.42</td><td>93.61</td></tr></table>

TABLE XII PERFORMANCE COMPARISON ON SHORT CAPTION DOMAIN DATASETS (MSCOCO AND FLICKR30K) USING VIT-L/14 BACKBONE WITH STANDARD EVALUATION METRICS. RESULTS SHOW RECALL@K SCORES FOR TEXT TO IMAGE AND IMAGE TO TEXT RETRIEVAL TASKS. THE BEST PERFORMANCE IS SHOWN IN BOLD, AND THE SECOND-BEST PERFORMANCE IS UNDERLINED. NOTE THAT FINECLIP ON MSCOCO IS FINE-TUNED ON MSCOCO ITSELF, THUS IT IS NOT A FAIR ZERO-SHOT COMPARISON. 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="4">Text to Image Recall@K</td><td colspan="4">Image to Text Recall@K</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@15</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@15</td></tr><tr><td rowspan="5">MSCOCO</td><td>CLIP</td><td rowspan="5">ViT-L/14</td><td>37.29</td><td>61.82</td><td>71.64</td><td>77.46</td><td>57.68</td><td>80.20</td><td>87.52</td><td>90.84</td></tr><tr><td>EVA-CLIP</td><td>48.14</td><td>71.72</td><td>80.10</td><td>84.46</td><td>63.96</td><td>84.68</td><td>91.04</td><td>93.66</td></tr><tr><td>Long-CLIP</td><td>46.96</td><td>71.89</td><td>80.64</td><td>85.31</td><td>63.16</td><td>84.52</td><td>90.60</td><td>93.44</td></tr><tr><td>GOAL</td><td>46.29</td><td>70.85</td><td>79.67</td><td>84.40</td><td>66.50</td><td>86.04</td><td>91.74</td><td>94.36</td></tr><tr><td>FAST-GOAL</td><td>48.53</td><td>73.22</td><td>82.12</td><td>86.30</td><td>66.58</td><td>86.78</td><td>92.40</td><td>94.72</td></tr><tr><td rowspan="6">Flickr30k</td><td>CLIP</td><td rowspan="6">ViT-L/14</td><td>65.38</td><td>87.36</td><td>92.24</td><td>94.68</td><td>86.40</td><td>97.50</td><td>99.30</td><td>99.40</td></tr><tr><td>EVA-CLIP</td><td>77.40</td><td>93.98</td><td>97.18</td><td>98.16</td><td>91.10</td><td>99.10</td><td>99.80</td><td>99.80</td></tr><tr><td>Long-CLIP</td><td>76.22</td><td>93.54</td><td>96.26</td><td>97.40</td><td>90.00</td><td>98.90</td><td>99.50</td><td>99.90</td></tr><tr><td>FineCLIP</td><td>72.68</td><td>91.56</td><td>95.58</td><td>96.78</td><td>80.80</td><td>95.90</td><td>97.60</td><td>98.70</td></tr><tr><td>GOAL</td><td>74.76</td><td>92.66</td><td>95.88</td><td>97.22</td><td>90.80</td><td>98.80</td><td>99.50</td><td>99.70</td></tr><tr><td>FAST-GOAL</td><td>76.72</td><td>93.72</td><td>96.74</td><td>97.60</td><td>92.50</td><td>99.40</td><td>99.70</td><td>99.80</td></tr></table>

GLIT100k for fine-tuning, making it substantially more accessible for researchers with limited computational resources.

Despite this dramatic difference in data scale, FAST-GOAL achieves competitive or superior performance on long caption understanding tasks. Most notably, on DCI, FAST-GOAL achieves 56.62% text-to-image R@1 and 56.74% image-totext R@1, outperforming all compared methods including MobileCLIPv2 with 53.04% and 55.06% despite its 120× larger pretraining scale. On DOCCI, FAST-GOAL attains 74.27% text-to-image R@1 and 74.67% image-to-text R@1, demonstrating strong performance with only 100k fine-tuning samples compared to the billions of samples required by other approaches.

TinyCLIP [14], which shares CLIP’s 400M pretraining scale, shows limited capability on lengthy descriptions with 34.77% on DCI, highlighting that pretraining scale alone is insufficient without targeted fine-tuning strategies. On short caption datasets where massive pretraining provides advantages, FAST-GOAL maintains reasonable performance despite being optimized for long-text scenarios. These results demonstrate that strategic fine-tuning with well-designed global-local alignment and carefully curated datasets can achieve strong longtext understanding without requiring billion-scale pretraining, offering a practical and efficient approach for extending visionlanguage models to detailed textual descriptions.

Analysis of Hard Negative Mining. Table XIV presents experimental exploration of hard negative mining strategies for our framework. We implement a two-stage training approach where after training FAST-GOAL with our standard pipeline, we perform additional fine-tuning using only in-image hard negatives. Specifically, we construct the contrastive learning matrix using only the multiple local pairs extracted from the same image, forcing the model to distinguish between different regions and their corresponding sentences within each sample.

TABLE XIIIPERFORMANCE COMPARISON WITH RECENT LIGHTWEIGHT AND EFFICIENT CLIP MODELS. RESULTS DEMONSTRATE THAT FAST-GOAL ACHIEVESCOMPETITIVE OR SUPERIOR PERFORMANCE ON LONG CAPTION UNDERSTANDING THROUGH EFFICIENT FINE-TUNING. THE BEST PERFORMANCE ISSHOWN IN BOLD, AND THE SECOND-BEST PERFORMANCE IS UNDERLINED.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td rowspan="2">Seen Data</td><td colspan="4">Text to Image Recall@K</td><td colspan="4">Image to Text Recall@K</td></tr><tr><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td></tr><tr><td rowspan="4">DOCCI</td><td>SigLIPv2</td><td>10B</td><td>49.24</td><td>75.73</td><td>90.76</td><td>94.71</td><td>48.39</td><td>74.41</td><td>90.49</td><td>94.82</td></tr><tr><td>MobileCLIPv2</td><td>12B</td><td>79.65</td><td>95.29</td><td>99.10</td><td>99.59</td><td>78.00</td><td>94.90</td><td>98.98</td><td>99.61</td></tr><tr><td>TinyCLIP</td><td>400M</td><td>59.10</td><td>83.90</td><td>95.43</td><td>97.94</td><td>58.88</td><td>83.63</td><td>95.35</td><td>97.82</td></tr><tr><td>FAST-GOAL</td><td>400M+100k</td><td>74.27</td><td>93.76</td><td>99.27</td><td>99.71</td><td>74.67</td><td>94.71</td><td>99.24</td><td>99.75</td></tr><tr><td rowspan="4">DCI</td><td>SigLIPv2</td><td>10B</td><td>29.62</td><td>49.80</td><td>67.82</td><td>75.34</td><td>29.64</td><td>48.61</td><td>67.73</td><td>75.11</td></tr><tr><td>MobileCLIPv2</td><td>12B</td><td>53.04</td><td>72.20</td><td>84.81</td><td>89.09</td><td>55.06</td><td>73.61</td><td>86.16</td><td>90.00</td></tr><tr><td>TinyCLIP</td><td>400M</td><td>34.77</td><td>55.21</td><td>72.81</td><td>79.45</td><td>36.52</td><td>56.41</td><td>74.35</td><td>81.10</td></tr><tr><td>FAST-GOAL</td><td>400M+100k</td><td>56.62</td><td>75.26</td><td>87.75</td><td>91.08</td><td>56.74</td><td>75.99</td><td>88.71</td><td>92.46</td></tr><tr><td rowspan="4">MSCOCO</td><td>SigLIPv2</td><td>10B</td><td>49.28</td><td>73.36</td><td>90.33</td><td>95.08</td><td>65.90</td><td>86.62</td><td>96.76</td><td>98.78</td></tr><tr><td>MobileCLIPv2</td><td>12B</td><td>49.63</td><td>74.92</td><td>91.58</td><td>96.26</td><td>67.44</td><td>86.76</td><td>97.00</td><td>99.10</td></tr><tr><td>TinyCLIP</td><td>400M</td><td>34.02</td><td>59.75</td><td>83.31</td><td>91.17</td><td>49.76</td><td>74.78</td><td>92.36</td><td>96.40</td></tr><tr><td>FAST-GOAL</td><td>400M+100k</td><td>42.81</td><td>68.61</td><td>88.54</td><td>94.64</td><td>61.00</td><td>82.84</td><td>95.34</td><td>98.32</td></tr><tr><td rowspan="4">Flickr30k</td><td>SigLIPv2</td><td>10B</td><td>74.24</td><td>91.48</td><td>97.76</td><td>98.94</td><td>90.10</td><td>98.30</td><td>100.00</td><td>100.00</td></tr><tr><td>MobileCLIPv2</td><td>12B</td><td>76.60</td><td>93.22</td><td>98.54</td><td>99.20</td><td>90.40</td><td>98.40</td><td>100.00</td><td>100.00</td></tr><tr><td>TinyCLIP</td><td>400M</td><td>59.68</td><td>83.56</td><td>95.00</td><td>97.32</td><td>77.50</td><td>92.80</td><td>98.70</td><td>99.20</td></tr><tr><td>FAST-GOAL</td><td>400M+100k</td><td>72.50</td><td>91.34</td><td>97.92</td><td>98.80</td><td>86.80</td><td>97.50</td><td>100.00</td><td>100.00</td></tr></table>

TABLE XIVABLATION STUDY ON HARD NEGATIVE MINING STRATEGY USING VIT-B/16 BACKBONE. FAST-GOAL + HARD NEG REPRESENTS TWO-STAGETRAINING WHERE THE SECOND STAGE USES ONLY IN-IMAGE LOCAL PAIRS AS HARD NEGATIVES. THE BEST PERFORMANCE IS SHOWN IN BOLD.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="4">Text to Image Recall@K</td><td colspan="4">Image to Text Recall@K</td></tr><tr><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td></tr><tr><td rowspan="2">DOCCI</td><td>FAST-GOAL + Hard Neg</td><td>51.00</td><td>77.65</td><td>93.18</td><td>96.45</td><td>54.61</td><td>80.92</td><td>94.24</td><td>97.25</td></tr><tr><td>FAST-GOAL</td><td>74.27</td><td>93.76</td><td>99.27</td><td>99.71</td><td>74.67</td><td>94.71</td><td>99.24</td><td>99.75</td></tr><tr><td rowspan="2">DCI</td><td>FAST-GOAL + Hard Neg</td><td>40.28</td><td>61.05</td><td>78.10</td><td>83.75</td><td>38.56</td><td>60.21</td><td>77.94</td><td>84.48</td></tr><tr><td>FAST-GOAL</td><td>56.62</td><td>75.26</td><td>87.75</td><td>91.08</td><td>56.74</td><td>75.99</td><td>88.71</td><td>92.46</td></tr><tr><td rowspan="2">MSCOCO</td><td>FAST-GOAL + Hard Neg</td><td>34.30</td><td>59.46</td><td>82.44</td><td>90.50</td><td>48.70</td><td>73.68</td><td>91.36</td><td>95.56</td></tr><tr><td>FAST-GOAL</td><td>42.81</td><td>68.61</td><td>88.54</td><td>94.64</td><td>61.00</td><td>82.84</td><td>95.34</td><td>98.32</td></tr><tr><td rowspan="2">Flickr30k</td><td>FAST-GOAL + Hard Neg</td><td>62.62</td><td>85.46</td><td>95.86</td><td>97.92</td><td>78.90</td><td>94.50</td><td>99.40</td><td>99.80</td></tr><tr><td>FAST-GOAL</td><td>72.50</td><td>91.34</td><td>97.92</td><td>98.80</td><td>86.80</td><td>97.50</td><td>100.00</td><td>100.00</td></tr></table>

The results show significant performance degradation across all datasets when applying hard negative mining. On DOCCI, text-to-image R@1 decreases from 74.27% to 51.00%. Similar patterns appear on DCI with performance dropping from 56.62% to 40.28%, on MSCOCO from 42.81% to 34.30%, and on Flickr30k from 72.50% to 62.62%. This substantial decline can be explained by the matching confidence distribution in our FLISM pipeline. The pipeline selects only the highest similarity local pair precisely because it represents the most confident and accurate correspondence. Other local pairs from the same image, despite being from the same source, have lower matching confidence and may represent less reliable region-sentence associations. Training with these lower-confidence pairs as hard negatives appears to introduce inconsistent supervision signals that affect model performance. These results suggest that effective hard negative mining in the context of local region-sentence matching requires careful consideration of matching confidence and correspondence quality.

Comparison with Sentence-Level Averaging. We examine an alternative approach for handling lengthy captions: splitting them into individual sentences, computing CLIP embeddings for each sentence separately, then averaging these embeddings for retrieval. Table XV presents results comparing this strategy against standard CLIP and FAST-GOAL. The results reveal fundamental limitations of simple averaging strategies. Sentence-level averaging achieves only 39.00% text-toimage R@1 on DOCCI and 19.48% on DCI, dramatically underperforming even standard CLIP extended with positional embedding interpolation to handle 248 tokens, which achieves 58.53% and 33.36% respectively. This demonstrates that processing the entire lengthy caption as a coherent sequence maintains contextual relationships across the full description more effectively than fragmenting it into independent sentences. However, simply extending token length through positional interpolation remains insufficient for truly understanding detailed descriptions, as evidenced by FAST-GOAL’s substantial improvements to 74.27% on DOCCI and 56.62% on DCI.

TABLE XV COMPARISON OF DIFFERENT APPROACHES FOR HANDLING LENGTHY TEXT USING VIT-B/16 BACKBONE. CLIP (SENTENCE-AVG) SPLITS CAPTIONS INTO SENTENCES AND AVERAGES THEIR EMBEDDINGS, WHILE CLIP (STANDARD) USES POSITIONAL EMBEDDING INTERPOLATION TO HANDLE 248 TOKENS. THE BEST PERFORMANCE IS SHOWN IN BOLD. 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="4">Text to Image Recall@K</td><td colspan="4">Image to Text Recall@K</td></tr><tr><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td></tr><tr><td rowspan="3">DOCCI</td><td>CLIP (sentence-avg)</td><td>39.00</td><td>68.82</td><td>89.20</td><td>94.18</td><td>40.88</td><td>71.37</td><td>91.14</td><td>95.84</td></tr><tr><td>CLIP (standard)</td><td>58.53</td><td>85.57</td><td>97.16</td><td>98.82</td><td>58.80</td><td>85.18</td><td>96.73</td><td>98.45</td></tr><tr><td>FAST-GOAL</td><td>74.27</td><td>93.76</td><td>99.27</td><td>99.71</td><td>74.67</td><td>94.71</td><td>99.24</td><td>99.75</td></tr><tr><td rowspan="3">DCI</td><td>CLIP (sentence-avg)</td><td>19.48</td><td>35.78</td><td>54.64</td><td>63.56</td><td>19.63</td><td>37.67</td><td>58.23</td><td>66.94</td></tr><tr><td>CLIP (standard)</td><td>33.36</td><td>53.87</td><td>72.34</td><td>79.33</td><td>35.07</td><td>54.28</td><td>71.84</td><td>78.91</td></tr><tr><td>FAST-GOAL</td><td>56.62</td><td>75.26</td><td>87.75</td><td>91.08</td><td>56.74</td><td>75.99</td><td>88.71</td><td>92.46</td></tr></table>

These results highlight why explicit global-local alignment is essential. Long captions contain rich information about multiple objects, fine-grained attributes, spatial relationships, and contextual details that must be understood both holistically and locally. While positional interpolation enables processing lengthy text, it does not provide the supervision necessary to effectively leverage the detailed information within these descriptions. Our global-local alignment strategy addresses this through dual-level supervision: global contrastive learning ensures overall scene understanding by processing the entire caption coherently, while local contrastive learning with TSL enables fine-grained region-text associations. The substantial performance improvements of FAST-GOAL over both sentence averaging and extended CLIP validate that explicit finetuning with global-local alignment is essential for effective long-text understanding.

Algorithm 1 Extract and pool patch tokens from bounding box coordinates

Require: Patch tokens $\mathbf { P } \in \mathbb { R } ^ { N \times d }$ , bounding box coordinates $( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } )$ , patch size $p ,$ number of patches per row $n _ { p }$

Ensure: Pooled feature vector $\mathbf { v } \in \mathbb { R } ^ { d }$

1: // Convert bbox coordinates to patch indices   
2: $\mathrm { p a t c h } _ { x _ { 1 } } , \mathrm { p a t c h } _ { y _ { 1 } } \gets \lfloor x _ { 1 } / p \rfloor , \lfloor y _ { 1 } / p \rfloor$   
$\mathrm { 3 } \mathrm { : \ p a t c h } _ { x _ { 2 } } \mathrm { , p a t c h } _ { y _ { 2 } } \gets \lceil x _ { 2 } / p \rceil , \lceil y _ { 2 } / p \rceil$   
4: // Collect all patches overlapping with bbox   
$\begin{array} { r } { 5 \colon \mathcal { Z }  \{ i \cdot n _ { p } + j \mid \mathrm { p a t c h } _ { y _ { 1 } } \leq i < \mathrm { p a t c h } _ { y _ { 2 } } , \mathrm { p a t c h } _ { x _ { 1 } } \leq j < } \end{array}$ patch $_ { _ { x _ { 2 } } } \}$   
6: // Average pooling over selected patches   
7: $\begin{array} { r } { \mathbf { v }  \frac { 1 } { | \mathcal { T } | } \sum _ { k \in \mathcal { T } } \mathbf { P } [ k ] } \end{array}$   
8: return v

TABLE XVI PATCH-LEVEL LOCALIZATION EVALUATION ON DOCCI DATASET USING VIT-B/16 BACKBONE. GIVEN LOCAL TEXT DESCRIPTIONS, MODELS PREDICT CORRESPONDING IMAGE REGIONS BY SELECTING PATCHES WITH HIGHEST SIMILARITY. THE BEST PERFORMANCE IS SHOWN IN BOLD. 

<table><tr><td>Method</td><td>Top-5 (%)</td><td>Top-10 (%)</td><td>Top-15 (%)</td></tr><tr><td>CLIP</td><td>43.25</td><td>57.78</td><td>67.57</td></tr><tr><td>FAST-GOAL</td><td>46.39</td><td>60.85</td><td>69.70</td></tr></table>

Patch-Level Localization Evaluation. To evaluate whether our method enhances region-level representations beyond image-level retrieval performance, we conduct patch-level localization experiments on the DOCCI dataset. Table XVI presents the evaluation setup and results. For each regiondescription pair in DOCCI where a specific image region corresponds to a descriptive sentence, we compute cosine similarity between the sentence embedding and all patch embeddings extracted from the entire image by the CLIP vision encoder. We then identify the patch with the highest similarity and check whether its center coordinate falls within the bounding box specifying the image region location. If the predicted patch location lies within this bounding box, we consider it a correct localization.

FAST-GOAL consistently outperforms CLIP across all evaluation thresholds. At Top-5, FAST-GOAL achieves 46.39% compared to CLIP’s 43.25%. The improvements extend to higher thresholds, with Top-10 showing 60.85% compared to 57.78%, and Top-15 achieving 69.70% compared to 67.57%. These results validate that our global-local alignment training with TSL effectively enhances the model’s ability to establish precise correspondences between textual descriptions and specific image regions. The consistent improvements across different thresholds confirm that FAST-GOAL not only improves image-level retrieval performance but also strengthens regionlevel representations, enabling better localization of objects and concepts described in text.

# C. Implementation Details for Patch Token Selection

In our Token Similarity-based Learning (TSL) method described in Eq. (9) of the main paper, we aggregate patch tokens from specific image regions based on bounding box

# Prompt

"Generate ONEFALSE statement thatiscompletely UNRELATED to this image. The statement should:   
-Describe objects, scenes,or situations that have NO connection to what's actually shown in the image   
- Sound natural and plausible as a description   
- Be a complete sentence   
- Be specific and detailed enough to blend with real descriptions   
-NOT use phrases like "there is no", "doesn't show", "not visible"-instead describe something falseas if it were true   
- Be completely random and unrelated to the image content (e.g.,if image shows outdoor scene, describe indoor objects; if image shows animals,describe vehicles,etc.)   
Examples of good unrelated false statements:   
- "A vintage typewriter sits on the mahogany desk."   
- "The chef is garnishing the pasta with fresh basil leaves."   
-"Two golden retrievers are playing fetch in the park."   
- "The bookshelf is filled with leather-bound classical novels."   
Generate ONLY the false unrelated statement, nothing else."

![](images/6ee78fddbe4eb78ff6b152f38bd3f67a88f3672fd255a7eec284bcd56e02a98b.jpg)

<details>
<summary>natural_image</summary>

Two toy car models: a polar bear and a horse, displayed on a white surface (no text or symbols visible)
</details>

# Wrong descriptions in the lengthy caption

An indoor close-up shot of a grey Hot Wheels roger dodger steam punk car with its silver engine components protruding from the hood facing the right infront of a toypolarbearanda small brown horse toy. A group of people are enjoying a picnic on the beach, with the sound of waves crashing in the background. The polar bear and horse are also facing the right.Polar bears positioned on all fours with its rightside legsforward asif it'sawalking.The black tip nose of the polar bear is slightly worn.The small brownhorse stands with all fourhooves flatMultiple Shadows of the objects are cast towards the left and right, indicating multiple light sources...

Fig. 8. Example of robustness evaluation with incorrect descriptions. False statements completely unrelated to the actual image content (shown in red) are generated using LLaVA-Next and inserted into the middle of original lengthy captions. The remaining portions contain correct descriptions of the image.   
TABLE XVII ROBUSTNESS ANALYSIS ON DOCCI DATASET WITH INCORRECT DESCRIPTIONS INJECTED INTO LENGTHY CAPTIONS USING VIT-B/16 BACKBONE. PERFORMANCE DROP INDICATES THE AVERAGE ABSOLUTE DECREASE IN R@1 ACROSS BOTH RETRIEVAL DIRECTIONS. FAST-GOAL DEMONSTRATES THE SMALLEST PERFORMANCE DEGRADATION. THE BEST PERFORMANCE IS SHOWN IN BOLD. 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="4">Text to Image Recall@K</td><td colspan="4">Image to Text Recall@K</td><td rowspan="2">Drop (%)</td></tr><tr><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td><td>R@1</td><td>R@5</td><td>R@25</td><td>R@50</td></tr><tr><td rowspan="6">DOCCI</td><td>CLIP (w/ Wrong)</td><td>49.75</td><td>79.20</td><td>94.29</td><td>97.45</td><td>50.35</td><td>76.84</td><td>92.75</td><td>96.39</td><td rowspan="2">-8.62</td></tr><tr><td>CLIP (Clean)</td><td>58.53</td><td>85.57</td><td>97.16</td><td>98.82</td><td>58.80</td><td>85.18</td><td>96.73</td><td>98.45</td></tr><tr><td>Long-CLIP (w/ Wrong)</td><td>66.27</td><td>89.53</td><td>98.29</td><td>99.45</td><td>60.84</td><td>87.35</td><td>97.86</td><td>99.39</td><td rowspan="2">-3.91</td></tr><tr><td>Long-CLIP (Clean)</td><td>71.63</td><td>92.16</td><td>98.90</td><td>99.73</td><td>63.29</td><td>88.80</td><td>98.39</td><td>99.45</td></tr><tr><td>FAST-GOAL (w/ Wrong)</td><td>71.47</td><td>91.92</td><td>98.92</td><td>99.59</td><td>71.69</td><td>93.10</td><td>99.16</td><td>99.73</td><td rowspan="2">-2.89</td></tr><tr><td>FAST-GOAL (Clean)</td><td>74.27</td><td>93.76</td><td>99.27</td><td>99.71</td><td>74.67</td><td>94.71</td><td>99.24</td><td>99.75</td></tr></table>

coordinates obtained from FLISM. This section provides detailed implementation of the patch token selection process, particularly addressing cases where bounding boxes may be smaller than individual ViT [6] patch dimensions.

Algorithm 1 presents the complete procedure for extracting and pooling patch tokens from a given bounding box. The key design principle is to collect all patches that overlap with the bounding box region, ensuring comprehensive representation regardless of bbox size relative to patch dimensions. Specifically, we convert bounding box coordinates to patch indices using floor operations for the top-left corner and ceiling operations for the bottom-right corner. This conversion guarantees that all patches with any spatial overlap with the bounding box are included in the selection. For bounding boxes smaller than a single ViT patch, this approach ensures at least one patch token is selected, as the floor and ceiling operations will capture the patch containing the bbox region. For larger bounding boxes spanning multiple patches, all overlapping patches are collected to provide complete spatial coverage. The selected patch tokens are then aggregated through average pooling to obtain a unified feature representation $P _ { m }$ that captures comprehensive information from the specified region. This robust extraction method enables effective learning of finegrained correspondences between image regions and textual descriptions across varying object scales and spatial extents.

# D. Robustness Analysis with Incorrect Descriptions

To evaluate the robustness of our method when handling captions containing incorrect information, we conduct systematic experiments by intentionally injecting false statements into the lengthy captions of the DOCCI [2] test set. Figure 8 illustrates the experimental setup. Using LLaVA-Next [15] with a carefully designed prompt, we generate completely unrelated false statements that describe objects, scenes, or situations not present in the actual images. These incorrect descriptions are inserted naturally into the middle of the original captions while preserving all correct descriptions, creating a realistic scenario where captions contain partial errors.

Table XVII presents the performance comparison across different methods when processing captions with injected incorrect descriptions. FAST-GOAL demonstrates the smallest performance degradation, experiencing only a 2.89% average drop in R@1 metrics compared to clean captions. In contrast, Long-CLIP [8] shows a 3.91% drop, while CLIP [5] exhibits a substantial 8.62% degradation. These results indicate that FAST-GOAL maintains more stable retrieval performance in the presence of partial caption errors.

This robustness can be attributed to our global-local alignment strategy. Since the majority of each caption still contains correct descriptions of the image, and our method learns to establish correspondences at both global and local levels through TSL, FAST-GOAL can effectively leverage the correct portions of the caption. The model’s ability to maintain strong performance despite partial errors demonstrates that our global-local alignment enables focusing on the most reliable correspondences while maintaining comprehensive understanding of the correct portions in lengthy captions. This natural resilience to isolated incorrect statements validates the effectiveness of our training approach in learning robust multimodal representations that can handle realistic scenarios where captions may contain occasional inaccuracies.

# REFERENCES

[1] H. Choi, Y. K. Jang, and C. Eom, “GOAL: Global-local object alignment learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 4070–4079.   
[2] Y. Onoe, S. Rane, Z. Berger, Y. Bitton, J. Cho, R. Garg, A. Ku, Z. Parekh, J. Pont-Tuset, G. Tanzer et al., “DOCCI: Descriptions of connected and contrasting images,” in Proceedings of the European Conference on Computer Vision, 2024, pp. 291–309.   
[3] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015–4026.   
[4] Y. Fang, B. Liao, X. Wang, J. Fang, J. Qi, R. Wu, J. Niu, and W. Liu, “You only look at one sequence: Rethinking transformer in vision through object detection,” in Advances in Neural Information Processing Systems, 2021, pp. 26 183–26 197.   
[5] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual representations from natural language supervision,” in Proceedings of the International Conference on Machine Learning, 2021, pp. 8748–8763.   
[6] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proceedings of the International Conference on Learning Representations, 2021.   
[7] J. Urbanek, F. Bordes, P. Astolfi, M. Williamson, V. Sharma, and A. Romero-Soriano, “A picture is worth more than 77 text tokens: Evaluating CLIP-style models on dense captions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26 700–26 709.

[8] B. Zhang, P. Zhang, X. Dong, Y. Zang, and J. Wang, “Long-CLIP: Unlocking the long-text capability of CLIP,” in Proceedings of the European Conference on Computer Vision, 2025, pp. 310–325.   
[9] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft COCO: Common objects in´ context,” in Proceedings of the European Conference on Computer Vision, 2014, pp. 740–755.   
[10] P. Young, A. Lai, M. Hodosh, and J. Hockenmaier, “From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions,” in Transactions of the Association for Computational Linguistics, 2014, pp. 67–78.   
[11] Q. Sun, Y. Fang, L. Wu, X. Wang, and Y. Cao, “EVA-CLIP: Improved training techniques for CLIP at scale,” in arXiv preprint arXiv:2303.15389, 2023.   
[12] M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem, I. Alabdulmohsin, N. Parthasarathy, T. Evans, L. Beyer, Y. Xia, B. Mustafa et al., “Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features,” 2025.   
[13] F. Faghri, P. K. A. Vasu, C. Koc, V. Shankar, A. Toshev, O. Tuzel, and H. Pouransari, “Mobileclip2: Improving multi-modal reinforced training,” 2025.   
[14] K. Wu, H. Peng, Z. Zhou, B. Xiao, M. Liu, L. Yuan, H. Xuan, M. Valenzuela, X. S. Chen, X. Wang et al., “Tinyclip: Clip distillation via affinity mimicking and weight inheritance,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 21 970–21 980.   
[15] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee, “LLaVA-NeXT: Improved reasoning, OCR, and world knowledge,” 2024.