# REDI: Corpus Aware Patch Ranking for DINOv3 Token Reduction

Chanjong Im University of Magdeburg chanjong.im@ovgu.de

Sebastian Diem University of Hildesheim diem@uni-hildesheim.de

Thomas Mandl University of Hildesheim mandl@uni-hildesheim.de

## Abstract

Most token reduction methods for Vision Transformers seek favorable tradeoffs between accuracy and efficiency by pruning, merging, or pooling patch tokens. REDI (Relevance for DINOv3 Token Reduction) studies this question through a controlled supervised reference: how should a fixed token budget be allocated across patches for image classification? REDI quantizes final block DINOv3 patch representations into a visual vocabulary and derives class conditioned corpus scores using supervised TF-IDF over visual words. For each validation image, the ground truth class selects a row of the TF-IDF table, and four transformed views produce a TF-IDF map aligned to a reference center crop. A separate dense pass on the same crop provides an attention map. After independent min max normalization, their elementwise product defines the REDI score. A fixed keep, merge, and compress operator then uses score rank to assign patch roles and score magnitude to weight merging and compression.

With precomputed REDI scores, a frozen DINOv3 ViT-B/16 backbone, and the same linear classifier used for dense evaluation, the operator reduces the sequence length from 201 to 107 tokens, a 46.8% sequence reduction. The REDI variant based on incoming attention mass achieves 84.706% Top-1 accuracy on ImageNet-1K, compared with 83.514% for the dense baseline, 82.634% for incoming attention mass alone, and 81.796% for supervised TF-IDF alone. The same corpus term also improves reduced classification for three alternative attention formulations relative to their attention only counterparts. Together, these controlled comparisons indicate that class specific corpus statistics and image specific attention provide complementary signals for patch ranking in this setting.

## 1. Introduction

Vision Transformers represent an image as a sequence of patch tokens, so sequence length directly affects the cost of self-attention and feedforward layers [9]. Token reduction methods exploit this structure by pruning, merging, pooling, or otherwise reorganizing tokens to improve the tradeoff between accuracy and computation [3, 17, 19, 25]. This line of work raises a complementary question: how should a limited token budget be allocated across patch representations?

We study this question through a corpus aware relevance formulation. The ground truth class defines the retrieval context, the labeled training images form the collection, and quantized patch representations act as visual words. We construct the vocabulary by applying spherical k-means [8] to unit normalized patch representations from the final DI-NOv3 block and assign each patch to the centroid with the highest cosine similarity. This construction follows the visual word abstraction of classical image retrieval and categorization [6, 32], while replacing handcrafted local descriptors with frozen DINOv3 features. A visual word receives a high class conditioned corpus score when it is common within the target class, more prevalent there than in the remaining classes, and relatively rare across the training collection.

We introduce REDI (Relevance for DINOv3 Token Reduction), a supervised reference score for class conditioned patch ranking. The score uses the validation image’s ground truth class and attention from a dense pass to define a controlled ranking. Its corpus component, TFIDF<sub>gt</sub>, is a table indexed by ImageNet class and visual word. REDI draws on TF-IDF (term frequency–inverse document frequency), a standard term-weighting framework that balances within document frequency against corpus level rarity [27, 30].

For each validation image, the ground truth class selects a table row, and four transformed views yield a TF-IDF map aligned to the patch grid of a reference center crop. A separate dense pass on that crop yields an attention map. In REDI, the primary image conditioned signal is incoming attention mass: we average each patch key column of the final block attention matrix over all heads and query positions. This follows the standard interpretation of selfattention matrices in which rows correspond to query positions and columns correspond to key positions [12]; related ViT analyses study how attention behavior varies across supervision regimes [36]. REDI independently normalizes the corpus and attention maps, multiplies them elementwise, and normalizes the product to obtain a patch score. Figure 1 summarizes the score construction and its use in the reduction operator.

![](images/4c05ab47a62df74c72d5f231cf519f46f6fc90261ebb6bb298d7b1fe1c2a55b3.jpg)  
Figure 1. Overview of REDI. Four transformed views provide class-conditioned visual TF-IDF scores aligned to a reference center crop, and a separate dense pass provides incoming attention mass from final-block attention columns. Independent normalization and elementwise multiplication produce the REDI score, which ranks patches for the fixed keep, merge, and compress operator and weights aggregation. With precomputed REDI scores, the frozen backbone and unchanged classifier, the 107 token reduced path reaches 84.706% Top-1 accuracy.

Figure 2 characterizes visual vocabulary statistics. Analysis codebooks fitted at selected DINOv3 blocks exhibit heavy-tailed, Zipf-like usage. For the final block, REDI score mass over the same assignments differs from raw assignment frequency, showing that the score is not determined by frequency alone.

We evaluate each scoring signal with the same fixed keep, merge, and compress operator, that is, 98 patch tokens remain explicit, 49 are merged into keep tokens, and 49 are aggregated into four spatial summary tokens. The resulting sequence contains 107 tokens rather than 201. The backbone remains frozen, and the dense feature classifier is reused unchanged, so paired comparisons isolate the scoring signal.

Under this controlled protocol, this pattern holds for all four attention formulations. Every attention only score, as well as TFIDF<sub>gt</sub> alone, falls below the dense Top-1 baseline under the fixed operator, whereas every corresponding REDI product exceeds it. The product with incoming attention mass reaches 84.706% Top-1 accuracy, compared with 83.514% for dense evaluation, 82.634% for incoming attention mass alone, and 81.796% for $\mathrm { T F I D F _ { g t } }$ alone.

Our contributions are threefold.

1. We formulate a class conditioned visual TF-IDF score over a quantized vocabulary of frozen ViT patch features and assign the resulting corpus statistics to individual patches.

2. We combine a class indexed corpus score over quantized ViT patch features with image specific attention to rank individual patches for token reduction. Under the shared DINOv3 reduction protocol, the combination improves reduced classification for all four evaluated attention formulations relative to their attention only counterparts.

3. We specify a fixed keep, merge, and compress operator for controlled evaluation and report the resulting analytical reductions in sequence length and block level matrix multiplication cost. Once the REDI score is available, the operator reduces sequence length by 46.8%, the estimated arithmetic cost of the dominant matrix multiplications in each transformer block by 47.8%, and the quadratic attention term by 71.7%.

## 2. Related work

## 2.1. Self supervised ViT features

ViT represents an image as a sequence of patch tokens [9]. DINO showed that self supervised ViT features exhibit semantically meaningful structure and that attention from the class token maps can reveal object regions without dense supervision [5]. DINOv2 extended this line of work with robust visual features designed to transfer across a broad range of downstream tasks [22]. DINOv3 further scaled self supervised visual pretraining and introduced Gram anchoring to address degradation in dense feature maps during long training schedules [31]. Register tokens provide dedicated locations for internal computation and reduce high norm artifacts in patch feature maps [7].

![](images/f6e40f549725e3839025c51ba4ee86b4fb06240103c070b8f762659b989b7571.jpg)

![](images/66fda8981c94ce898b492880b08a58261840602512ded033e5581677f79e2523.jpg)  
Figure 2. Visual word assignment distributions and REDI score mass. Left: visual word frequency by rank for separate $K = 5 1 2$ analysis codebooks fitted at selected DINOv3 blocks, showing heavy-tailed, Zipf-like usage. Right: final block validation assignments sorted by frequency, with REDI score mass accumulated over the same visual words. The different profiles show that REDI score mass is not determined by raw frequency alone. The analysis codebooks are separate from the codebook used for $\mathrm { T F I D F _ { g t } }$

## 2.2. Token pruning and adaptive computation

Token reduction methods differ in how they identify and process redundant or less informative tokens. DynamicViT inserts lightweight prediction modules at multiple layers and learns progressive, input dependent token sparsification with differentiable attention masking [25]. A-ViT assigns a halting score to each spatial token and stops updating the token once its accumulated score reaches a threshold [39]. AdaViT adapts patch token, attention head, and transformer block usage to each input [20]. Evo-ViT separates tokens into informative and placeholder groups using evolved attention from the class token; informative tokens follow the standard transformer path, while placeholder tokens are updated through a representative token at lower cost [38].

Several methods propagate token decisions across depth or derive token supervision from later representations or model internal signals. $\mathrm { I A } { \mathrm { - R E D } } ^ { 2 }$ introduces interpretable modules that progressively remove redundant patches [23]. Patch Slimming identifies effective patches in the final layer and propagates this supervision to earlier layers [34]. EViT preserves tokens with high attention from the class token and fuses the remaining tokens into a single representative token [17]. ATS is a parameter free differentiable module that scores and adaptively samples tokens for each input [10]. SPViT uses latency aware multi-head token selectors and aggregates pruned tokens into a package token rather than discarding them [16]. TokenLearner applies learned spatial attention functions to aggregate intermediate features into a compact set of adaptive tokens [29].

## 2.3. Token merging, pooling, and attention based scores

PatchMerger inserts a learned module between transformer layers to merge tokens and shorten the sequence [26]. ToMe progressively combines similar tokens with a lightweight bipartite matching procedure and can be applied to pretrained Vision Transformers without retraining [3]. Token Pooling approximates a set of intermediate tokens by clustering them to minimize the reconstruction error introduced by downsampling [19]. Beyond Attentive Tokens combines importance derived from class attention with token diversity through decoupling and merging [18]. WeiToP trains an early pruning module for visual place recognition by distilling aggregation induced token importance [40].

Attention also provides a direct source of token scores. Abnar and Zuidema introduced attention rollout and attention flow as post hoc methods for propagating attention across transformer layers [1]; we include their rollout construction as one of four attention baselines. Su et al. derive Col-Ln from Renyi entropy and rank tokens by the´ $\ell _ { n }$ norm of attention columns [33]. Aizawa and Igaue use the entropy of each attention distribution associated with each patch as a pruning criterion and extend the formulation from Shannon to Renyi entropy [ ´ 2].

Most acceleration methods above couple a scoring or adaptation rule with a particular pruning, merging, or pooling mechanism. REDI instead evaluates a supervised, class conditioned scoring signal under a fixed operator. Holding the operator constant separates the comparison of scoring rules from the design of the reducer and makes attention only and corpus aware scores directly comparable at the same token budget.

## 2.4. Visual words and corpus weighting

Classical visual retrieval transferred indexing and term weighting ideas from text retrieval to images. Video Google quantizes local region descriptors into visual words, indexes them with an inverted file, and applies TF-IDF weighting for object retrieval [32]. The bag of keypoints model represents an image as a histogram of quantized local descriptors for visual categorization [6]. Vocabulary trees organize visual words hierarchically to support large scale search [21], while Philbin et al. combine large vocabularies with spatial verification for object retrieval [24]. Salton and Buckley review term weighting schemes based on term frequency and collection statistics [30], and Robertson analyzes the theoretical basis of inverse document frequency [27].

Recent work has applied visual vocabularies and corpus weighting to learned visual units. The Visual Word Tokenizer compresses ViT inputs using either intra image pixel statistics or an inter image visual vocabulary; its inter image variant averages patches matched to the same visual word [11]. Wang et al. use TF-IDF to identify class discriminative CNN channels for federated unlearning [37]. Heo et al. use TF-IDF based channel scoring over DINOv2-CLIP aligned features to construct class specific semantic prototypes for few-shot detection [15]. Cao et al. use TF-IDF to identify class related tokens as one component of a broader structural score for assessing and guiding dataset distillation [4]. BM25-V applies BM25 to sparse visual word activations derived from late layer ViT patch features, aggregating patch activations into image level representations for retrieval [13]. These studies provide relevant precedents for visual word compression and corpus weighting in learned visual spaces. To our knowledge, none of them uses a corpus score indexed by the ground truth class together with image specific attention from the dense path to construct an offline reference ranking for individual patch tokens under a fixed reduction operator.

## 3. Method

## 3.1. Problem setting and pipeline

We use the frozen DINOv3 ViT-B/16 LVD-1689M checkpoint<sup>1</sup>, released for the LVD-1689M setting [31]. Following the Vision Transformer tokenization scheme [9], the model represents a 224 × 224 input crop as 196 patch tokens on a 14×14 grid, one CLS token, and four register tokens, giving a sequence of 201 tokens. We refer to this unreduced configuration as the dense path. The encoder has the standard ViT-B dimensions of 12 transformer blocks, hidden width D = 768, and 12 attention heads [9]. The DINOv3 checkpoint uses axial rotary position embeddings (RoPE) [31].

The REDI pipeline comprises two stages: offline computation of a supervised score for patch ranking and reduced evaluation with a fixed token reduction operator. To construct the corpus component, we use the ImageNet-1K training split [28]. For each training image, we form a

224 × 224 center crop and use all 196 patch representations from the final block to fit a visual vocabulary. The resulting patch assignments are used to construct a TF-IDF table indexed by ImageNet class and visual word. For each validation image, we extract all 196 patch representations from the final block from each of four transformed views and quantize them with the same vocabulary. The ground truth class is then used to retrieve TF-IDF scores for individual patches, which are aligned to the 14×14 grid of a reference center crop and aggregated into a single map. A separate forward pass on the reference center crop produces an attention map on the same grid. REDI independently normalizes the TF-IDF and attention maps, multiplies them elementwise, and normalizes the product to obtain the REDI score. The score is precomputed from the ground truth class and dense path attention, so the reduced run is a controlled reference evaluation.

During reduced evaluation, the fixed keep, merge, and compress operator is applied immediately before the first transformer block. Sorting the patch scores assigns 98 patches to the keep set, 49 to the merge set, and 49 to the compress set. Patches in the merge set are aggregated into destination keep tokens, whereas patches in the compress set are aggregated into four compressed tokens using a fixed spatial partition. The CLS token and four register tokens pass through the reduction operator unchanged. The resulting sequence contains one CLS token, four register tokens, 98 updated keep tokens, and four compressed tokens, for a total of 107 tokens. All 12 frozen transformer blocks process this reduced sequence.

After the final block, the four register embeddings are excluded. The final CLS embedding and the mean of the final patch embeddings are $\ell _ { 2 } \cdot$ -normalized separately. The two normalized embeddings are then averaged, and the result is $\ell _ { 2 } \cdot$ -normalized to obtain the feature vector supplied to the linear classifier. The patch mean is computed over 196 embeddings in the dense path and 102 embeddings in the reduced path. The classifier is trained once on these vectors from the dense path and reused unchanged for reduced evaluation. Its optimization settings are specified in the experimental protocol.

## 3.2. Supervised visual TF-IDF

Visual vocabulary. Our visual vocabulary follows the bag of visual words abstraction used in image retrieval and categorization [6, 32]. Unlike those methods, which quantize handcrafted local descriptors, we quantize frozen DI-NOv3 patch representations. For every ImageNet training image, we resize the shorter side to 256 pixels, take a 224 × 224 center crop, and extract all 196 patch representations from the final block before the model’s final normalization. After $\ell _ { 2 }$ normalization, these representations are clustered with spherical k-means [8] using K = 512 centroids. The codebook is initialized deterministically and refined by three full passes over the training feature set, each comprising patch assignment followed by centroid recomputation. Each patch is assigned to the centroid with the highest cosine similarity, and the centroid index is used as its visual word.

Class and corpus statistics. Our weighting scheme is inspired by classical TF-IDF term weighting [27, 30] and by its application to visual words in Video Google [32]. The formulation retains image level term frequency and corpus level inverse document frequency. The supervised class contrast, together with the smoothing, clipping, and multiview aggregation choices, is specific to REDI. For a training image d and visual word $k ,$ let $n _ { d } ( k )$ be the number of its 196 patches assigned to k. The term frequency within the image is

$$
\operatorname{tf} _ {d} (k) = \frac {n _ {d} (k)}{1 9 6}.\tag{1}
$$

For class $y ,$ let $\bar { p } _ { y } ( k )$ denote the average of $\operatorname { t f } _ { d } ( k )$ over training images in class $y ,$ and let $\bar { p } _ { \neg y } ( k )$ denote the corresponding average over all remaining training images. Both quantities are floored at $1 0 ^ { - 8 }$ before taking logarithms. We define the positive class contrast term as

$$
r _ {y} (k) = \max \{0, \log \bar {p} _ {y} (k) - \log \bar {p} _ {\neg y} (k) \} \sqrt {\bar {p} _ {y} (k)}.\tag{2}
$$

The positive part retains only visual words that are more prevalent in class $y$ than outside it. The square-root factor weights the contrast by support within the class while remaining sublinear in frequency.

Let N be the number of ImageNet training images, and let df(k) be the number of training images containing at least one patch assigned to word k. The inverse document frequency is

$$
\operatorname{idf} (k) = \operatorname{clip} _ {[ 0, 8 ]} \left(\log \frac {N + 1}{\operatorname{df} (k) + 1}\right).\tag{3}
$$

Adding one to the numerator and denominator provides smoothing. Clipping to [0, 8] is an implementation choice used consistently in all reported experiments. The resulting table indexed by class is

$$
\mathrm{TFIDF} _ {\mathrm{gt}} (y, k) = r _ {y} (k) \operatorname{idf} (k).\tag{4}
$$

This construction combines corpus level rarity with REDI’s supervised class contrast. A large entry requires positive contrast for class y, non-negligible frequency within that class, and relative rarity across the training corpus.

Validation TF-IDF map. For a vector $q ~ \in \ \mathbb { R } ^ { 1 9 6 }$ , let $q _ { \mathrm { m i n } } = \operatorname* { m i n } _ { j } q _ { j }$ and $q _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { j } q _ { j }$ . We normalize each

image map as

$$
\mathcal {N} (q) _ {i} = \left\{ \begin{array}{l l} \frac {q _ {i} - q _ {\min}}{q _ {\max} - q _ {\min} + 1 0 ^ {- 8}}, & q _ {\max} - q _ {\min} > 1 0 ^ {- 8}, \\ 0, & \text { otherwise }. \end{array} \right.\tag{5}
$$

Thus, a nonconstant map is scaled approximately to [0, 1], whereas a constant map is replaced by the zero vector.

For validation view $\nu ,$ let $z _ { i } ^ { ( \nu ) }$ be the visual word assigned to the final block representation at patch position i, and let y be the image’s ground truth class. The raw patch score and normalized view map are

$$
\tau_ {i} ^ {(\nu)} = \mathrm{TFIDF} _ {\mathrm{gt}} \Big (y, z _ {i} ^ {(\nu)} \Big), \qquad t ^ {(\nu)} = \mathcal {N} \Big (\tau^ {(\nu)} \Big).\tag{6}
$$

Because the lookup uses the ground truth class $y ,$ the resulting map is a ground truth conditioned reference score.

Our validation procedure uses four views. After resizing the shorter side to 256 pixels, we take a $2 2 4 \times 2 2 4$ center crop, its horizontal flip, and two deterministic random resized crops. The random crops use a crop area scale of [0.60, 1.00] and an aspect ratio of $[ 3 / 4 , 4 / 3 ]$ . Crop parameters and horizontal flips are deterministically seeded by the image index; each random crop is flipped with probability 0.5. Every view is represented at $2 2 4 \times 2 2 4$ before feature extraction.

The four score maps are aligned to the grid of the reference center crop. A flipped map is first restored to its original orientation. The reference patch centers are then expressed in the coordinate system of each transformed crop, and bilinear sampling maps the scores back to the $1 4 \times 1 4$ grid. A validity mask excludes reference locations outside a transformed crop. For reference patch $i ,$ let $V _ { i }$ be the set of views that cover that location, and let $\widetilde { t } _ { i } ^ { ( \nu ) }$ be the corresponding remapped score. We compute

$$
\begin{array}{l} \mu_ {i} = \frac {1}{| V _ {i} |} \sum_ {\nu \in V _ {i}} \widetilde {t} _ {i} ^ {(\nu)}, \qquad \xi_ {i} = \max _ {\nu \in V _ {i}} \widetilde {t} _ {i} ^ {(\nu)}, \\ t = \mathcal {N} (0. 8   \mu + 0. 2   \xi)  . \end{array}\tag{7}
$$

The center crop covers every reference patch location, so $V _ { i }$ is never empty. We fix the aggregation coefficients to 0.8 and 0.2 in all experiments.

## 3.3. Attention scores

Attention scores are computed in a separate forward pass on the reference center crop. In the self-attention matrix, rows correspond to query positions and columns correspond to key positions [12]. We aggregate the column associated with each patch over all query positions and attention heads. We use the descriptive term incoming attention mass for this accumulated column weight; related ViT analyses study how attention behavior varies across supervision regimes [36]. This final block statistic serves as the primary image conditioned ranking signal. Let $A ^ { ( L , h ) } \in \mathbb { R } ^ { \dot { T } \times T }$ denote the attention matrix after softmax for head h in block $L = 1 2 .$ , with $H = 1 2$ heads and $T = 2 0 1$ tokens. For patch position $i ,$ let $j ( i )$ denote its key column after the CLS token and four register tokens. The raw score is

$$
\widetilde {a} _ {i} = \frac {1}{H T} \sum_ {h = 1} ^ {H} \sum_ {q = 1} ^ {T} A _ {q, j (i)} ^ {(L, h)}.\tag{8}
$$

This quantity is the mean of patch i’s key column over all 201 query positions and 12 heads. This statistic is image conditioned but remains dense path information, since it is extracted before applying the reduced operator. The primary attention map is $s _ { \mathrm { m a s s } } = \mathcal { N } ( \widetilde { a } )$

We also evaluate three alternatives. The CLS mean map averages, across heads, attention from the CLS token to the patch tokens in the final block. The CLS maximum map instead takes the maximum across heads. Our rollout baseline follows the layerwise matrix composition of Abnar and Zuidema [1], adapted to the DINOv3 token sequence. It averages heads within each block, adds the identity matrix, normalizes each row, and composes the resulting transition matrices across all 12 blocks. The rollout score is the final CLS row restricted to the 196 patch columns. The exact tensor reductions are given in the supplementary material. Each attention map is combined with the corpus term to produce the score used for patch ranking and aggregation.

## 3.4. REDI score

Let $t \in \mathbb { R } ^ { 1 9 6 }$ be the supervised TF-IDF map and let $a \in$ $\mathbb { R } ^ { 1 9 6 }$ be an attention map. The primary REDI variant uses $a = s _ { \mathrm { m a s s } } .$ . REDI normalizes both inputs, multiplies them elementwise, and normalizes the product:

$$
\widehat {t} = \mathcal {N} (t), \qquad \widehat {a} = \mathcal {N} (a), \qquad s = \mathcal {N} \big (\widehat {t} \odot \widehat {a} \big),\tag{9}
$$

where ⊙ denotes elementwise multiplication. The normalized product defines the REDI score. A high REDI score therefore requires both a high class conditioned corpus weight and a high value under the selected attention signal. The same rule is applied to each alternative attention map. The REDI score has two roles in the fixed operator: its rank assigns patches to the keep, merge, and compress sets, and its magnitude weights patch contributions during merging and compression. Merge destinations are selected separately using feature similarity and spatial proximity, whereas compress cell membership depends only on grid location. Section 3.5 specifies these operations.

## 3.5. Fixed keep, merge, and compress operator

We fix the operator to compare scoring signals under a common reduction mechanism. It is conceptually related to ViT acceleration methods that preserve attentive tokens while fusing inattentive ones (EViT) [17], progressively merge similar tokens (ToMe) [3], or approximate a token set with a smaller set by minimizing downsampling reconstruction error (Token Pooling) [19]. Its design draws on these general principles, but the fixed budget, role assignment, spatial bias, and aggregation weights are specific to our evaluation protocol rather than an implementation of any single prior method.

Role assignment. For any scoring configuration, let $s _ { i }$ denote the normalized score assigned to patch i. We sort the 196 patch scores in descending order. The highest $n _ { R } \ = \ \mathrm { r o u n d } ( 0 . 5 0 \times 1 9 6 ) \ = \ 9 8$ scores define the keep set $R ,$ the next $n _ { M } = \mathrm { r o u n d } ( 0 . 2 5 \times 1 9 6 ) = 4 9$ define the merge set M, and the remaining $n _ { U } = 1 9 6 - n _ { R } - n _ { M } = 4 9$ define the compress set $U .$ . The three sets are disjoint and together contain all 196 patches. The rank of $s _ { i }$ determines the role of each patch, while its magnitude is subsequently used to weight aggregation. The scores are precomputed along the dense path, whereas token reduction is applied to the patch embeddings immediately before the first transformer block. Let $x _ { i } \in \mathbb { R } ^ { D }$ denote the embedding of patch i at this point, and let $g _ { i } \in \{ 0 , \ldots , 1 3 \} ^ { 2 }$ denote its coordinate on the $1 4 \times 1 4$ patch grid.

Merge destination selection. Each patch $i \in M$ is assigned to one patch in the keep set. For $i \in M$

$$
\begin{array}{c} \rho_ {i r} = 1 - \frac {\| g _ {i} - g _ {r} \| _ {2}}{1 3 \sqrt {2}}, \\ m (i) = \arg \max _ {r \in R} \left\{\cos (x _ {i}, x _ {r}) + 0. 3 0 \rho_ {i r} \right\}. \end{array}\tag{10}
$$

Here $\rho _ { i r } \in [ 0 , 1 ]$ denotes normalized spatial proximity, and $1 3 \sqrt { 2 }$ is the maximum Euclidean distance on the $1 4 \times 1 4$ grid. Once the role sets have been determined, the merge destination depends only on cosine similarity between the pre-block patch embeddings and on spatial proximity. The patch score $s _ { i }$ is not part of the destination criterion. The spatial coefficient is fixed at 0.30 and is not learned.

Score weighted merge update. Once all merge destinations have been assigned, the normalized patch scores are used to weight aggregation. For each $i \in M \cup U$ , we define $\omega _ { i } = \mathrm { m a x } \{ s _ { i } , 0 \} + 1 0 ^ { - 4 }$ , where the small positive offset prevents a zero aggregation weight. Each keep token is assigned a self weight of one. The updated embedding of $r \in R$ is

$$
\widetilde {x} _ {r} = \frac {x _ {r} + \sum_ {i \in M : m (i) = r} \omega_ {i} x _ {i}}{1 + \sum_ {i \in M : m (i) = r} \omega_ {i}}, \qquad r \in R.\tag{11}
$$

If no patch in $M$ is assigned to $r ,$ then $\widetilde { x } _ { r } \ = \ x _ { r }$ . Equation (10) determines the destination of each merge patch, whereas Eq. (11) determines its contribution to the updated keep token.

Spatial compression. The compress set is summarized separately rather than assigned to keep tokens. We divide the $1 4 \times 1 4$ patch grid into four non-overlapping $7 \times 7$ cells. For each cell $c ,$ let $U _ { c } \subseteq U$ denote the compress set patches located within that cell. If $U _ { c }$ is nonempty, its patches are summarized by one score weighted token,

$$
u _ {c} = \frac {\sum_ {i \in U _ {c}} \omega_ {i} x _ {i}}{\sum_ {i \in U _ {c}} \omega_ {i}}.\tag{12}
$$

If $U _ { c }$ is empty, the same weighted average is computed over the full compress set U , ensuring that the operator always produces four summary tokens. A summary token from a nonempty cell inherits the RoPE position of the highest scoring patch in $U _ { c } .$ . For an empty cell, it inherits the position of the highest scoring patch in U . This construction preserves an existing patch position rather than introducing a synthetic coordinate.

The updated tokens in R are restored to their original spatial order and retain their original RoPE positions. The four compressed tokens are then appended. Together with the unchanged CLS token and four register tokens, these outputs form the 107-token sequence processed by the frozen encoder.

## 3.6. Analytical cost estimate

The sequence length decreases from 201 to 107 tokens, a reduction of 46.8%. We estimate the resulting reduction in the dominant matrix multiplications of a standard ViT-B block. The block contains four $D \times D$ attention projections and an MLP with two layers and hidden width 4D [9]. This structure gives

$$
C _ {\mathrm{block}} (T, D) = 1 2 T D ^ {2} + 2 T ^ {2} D.\tag{13}
$$

The linear term comprises $4 T D ^ { 2 }$ for the query, key, value, and output projections and $8 T D ^ { 2 }$ for the two MLP projections. The quadratic term accounts for the product of queries and keys and the multiplication of attention weights by values.

With $D = 7 6 8 , C _ { \mathrm { b l o c k } } ( 1 0 7 , 7 6 8 ) / C _ { \mathrm { b l o c k } } ( 2 0 1 , 7 6 8 ) =$ 52.2%. Thus, the estimated arithmetic cost of these matrix multiplications is reduced by 47.8% in each transformer block. Because reduction occurs before block 1, the same ratio applies to all 12 transformer blocks. The quadratic attention term retains $( 1 0 7 / 2 0 1 ) ^ { 2 } = 2 8 . 3 \%$ of its value in the dense path, corresponding to a reduction of 71.7%.

These estimates report the change in the dominant encoder matrix multiplications caused by the shorter token sequence. We discuss the scope of these estimates in Sec. 6.

Table 1. ImageNet-1K accuracy for dense and reduced evaluation. The REDI score is the normalized product of class-conditioned visual TF-IDF and incoming attention mass. All reduced configurations use the same 107-token budget, frozen backbone, classifier, and reduction operator.

<table><tr><td>Configuration</td><td>Tokens</td><td>Top-1</td><td> $\Delta$  dense</td><td>Top-5</td></tr><tr><td>Dense baseline</td><td>201</td><td>83.514</td><td>0.000</td><td>96.902</td></tr><tr><td>Supervised TFIDFgt</td><td>107</td><td>81.796</td><td>-1.718</td><td>96.236</td></tr><tr><td>Incoming attention mass</td><td>107</td><td>82.634</td><td>-0.880</td><td>96.454</td></tr><tr><td>REDI with incoming attention mass</td><td>107</td><td>84.706</td><td>+1.192</td><td>97.442</td></tr></table>

## 4. Experimental results

## 4.1. Evaluation protocol

We use the ImageNet-1K training split [28] to fit the visual vocabulary, estimate class conditioned TF-IDF statistics, and train a single linear classifier. All accuracies are reported on the complete validation split of 50,000 images. The pretrained DINOv3 ViT-B/16 backbone remains frozen. The classifier is trained for 20 epochs on features extracted from dense passes using AdamW, a batch size of 4096, an initial learning rate of $3 \times 1 0 ^ { - 3 }$ with cosine decay, weight decay of $1 0 ^ { - 2 }$ , and no label smoothing. Feature extraction, classifier training, and evaluation use FP32 without automatic mixed precision. Cached score maps are stored in FP16 and converted to FP32 when loaded.

Every reduced configuration uses the same frozen backbone, classifier, 107 token budget, merge and compression rules, and RoPE handling defined in Sec. 3.5. Across reduced evaluations, only the patch scoring signal used for ranking and aggregation changes. Since REDI scores are precomputed, the experiments compare reference rankings under a shared reducer.

## 4.2. Main comparison

Table 1 compares the dense path with reduced evaluation under the fixed operator. The dense path reaches 83.514% Top-1 with 201 tokens. At 107 tokens, $\mathrm { T F I D F _ { g t } }$ alone reaches 81.796% Top-1 and incoming attention mass alone reaches 82.634%. Their normalized product reaches 84.706% Top-1, giving gains of 1.192 points over dense, 2.910 over $\mathrm { T F I D F _ { g t } } ,$ , and 2.072 over incoming attention mass under the same reducer.

## 4.3. Consistency across attention formulations

Table 2 pairs each attention only configuration with the REDI product formed from the same attention score. Combining each attention score with $\mathrm { T F I D F _ { g t } }$ improves Top-1 accuracy by 2.072 points for incoming attention mass, 3.330 points for attention rollout, 1.740 points for CLS attention maximum, and 1.736 points for CLS attention mean. Every attention only configuration remains below the 83.514% dense baseline, whereas every REDI product exceeds it by 1.010–1.192 points.

Table 2. Paired comparisons at the 107-token budget. “Gain” denotes the Top-1 difference between the REDI product formed with the listed attention score and its attention-only counterpart.

<table><tr><td>Attention score</td><td>Attention Top-1</td><td>REDI Top-1</td><td>Gain</td><td>REDI Top-5</td></tr><tr><td>Incoming attention mass</td><td>82.634</td><td>84.706</td><td>+2.072</td><td>97.442</td></tr><tr><td>Attention rollout</td><td>81.314</td><td>84.644</td><td>+3.330</td><td>97.446</td></tr><tr><td>CLS attention max</td><td>82.834</td><td>84.574</td><td>+1.740</td><td>97.386</td></tr><tr><td>CLS attention mean</td><td>82.788</td><td>84.524</td><td>+1.736</td><td>97.384</td></tr></table>

<table><tr><td>Quantity</td><td>Dense</td><td>Reduced path</td><td>Reduction</td></tr><tr><td>Individually represented patches</td><td>196</td><td>98</td><td>50.0%</td></tr><tr><td>Patch representations after reduction</td><td>196</td><td>102</td><td>48.0%</td></tr><tr><td>Total sequence tokens</td><td>201</td><td>107</td><td>46.8%</td></tr><tr><td>Quadratic attention term</td><td> $201^2$ </td><td> $107^2$ </td><td>71.7%</td></tr><tr><td>Estimated dominant block arithmetic</td><td>1.000</td><td>0.522</td><td>47.8%</td></tr></table>

Table 3. Analytical sequence and compute reductions at the block level after the fixed operator forms the 107-token sequence. The estimates exclude score construction, reduction overhead, and hardware-dependent effects.

## 4.4. Sequence and analytical compute reduction

At 107 tokens, the estimated dominant matrixmultiplication cost per frozen block and the quadratic attention term fall by 47.8% and 71.7%, respectively. The scope of these analytical estimates is discussed in Sec. 6.

## 5. Discussion

Combining the same class conditioned corpus term with attention improves reduced classification across all four attention formulations under an otherwise identical setup. Because the backbone, classifier, token budget, and reduction operator remain fixed, this repeated pattern indicates that the corpus term contributes ranking information that is not captured by any evaluated attention score alone.

The two components encode different forms of context. Attention reflects image specific token interactions but contains no explicit class or corpus level statistics. The TF-IDF term contributes class contrast, within class support, and corpus rarity, but assigns the same lookup value to patches mapped to the same visual word within a view. Their product therefore favors patches supported by both class conditioned corpus statistics and image conditioned attention. The gains obtained with all four attention formulations suggest that this complementarity is not specific to incoming attention mass.

Supplementary DeiT III [35] and MAE [14] diagnostics show that the hybrid signal remains close to dense accuracy across the two added backbones, but those experiments use a different classifier protocol and are therefore treated as

diagnostic.

The score affects the reduced representation in two ways. Its rank determines which patches remain explicit, which are merged, and which contribute to spatial summaries; its magnitude controls the contribution of merged and compressed patches during aggregation. The classifier is trained only on dense features and reused without adaptation, so differences in reduced accuracy measure how well each reduced representation supports the same linear decision rule. Under this protocol, the joint score yields higher accuracy than either component alone.

The gain over the dense baseline is specific to the present backbone, classifier, token budget, and reducer. The fixed reducer is useful because it isolates the contribution of corpus level class information without implying a general advantage of shorter sequences or optimality of the operator.

## 6. Limitations and future work

The main evaluation uses DINOv3 ViT-B/16; supplementary DeiT III and MAE experiments use model native classifiers and a separate classifier adaptation protocol. REDI also relies on the ground truth class and dense path attention, so it should be interpreted as a supervised reference for patch relevance under a fixed reducer rather than a deployable acceleration pipeline. Therefore, the accuracy gain over the dense baseline should not be read as evidence that shorter sequences are generally superior, but as evidence that this offline reference ranking can allocate the fixed token budget more favorably for the reused classifier.

The reported sequence length and arithmetic reductions describe the reduced encoder path after REDI scores are available. They do not include score construction, dense attention extraction, reduction overhead, operations outside the dominant matrix multiplications, memory traffic, or hardware effects. Future work should replace the ground truth class lookup and dense attention map with predicted or early layer approximations, and test broader architectures, tasks, budgets, and reducer designs.

## 7. Conclusion

REDI ranks DINOv3 patch tokens by combining class indexed visual-word statistics with image specific attention under a fixed reduction budget. Across four attention formulations, the combined score improves reduced classification over either component alone while using the same keep, merge, and compress operator. The results show that corpus level class statistics provide a useful complementary signal for controlled token allocation and motivate future work on deployable approximations.

## References

[1] Samira Abnar and Willem Zuidema. Quantifying attention flow in transformers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4190–4197, 2020. 3, 6

[2] Hiroaki Aizawa and Yuki Igaue. Renyi attention entropy for´ patch pruning, 2026. arXiv preprint arXiv:2604.03803. 3

[3] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your ViT but faster. In International Conference on Learning Representations, 2023. 1, 3, 6

[4] Yue Cao, Jianyang Gu, Vyacheslav Kungurtsev, Yu Hu, Jozsef Hamari, Zheng Liu, and Mohsen Zardadi. Structural assessment for understanding and guiding dataset distillation in discrete token space, 2026. arXiv preprint arXiv:2606.21705. 4

[5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou,´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In IEEE/CVF International Conference on Computer Vision, 2021. 2

[6] Gabriella Csurka, Christopher R. Dance, Lixin Fan, Jutta Willamowski, and Cedric Bray. Visual categorization with´ bags of keypoints. In ECCV Workshop on Statistical Learning in Computer Vision, pages 1–22, 2004. 1, 4

[7] Timothee Darcet, Maxime Oquab, Julien Mairal, and Piotr´ Bojanowski. Vision transformers need registers. In International Conference on Learning Representations, 2024. 2

[8] Inderjit S. Dhillon and Dharmendra S. Modha. Concept decompositions for large sparse text data using clustering. Machine Learning, 42:143–175, 2001. 1, 4

[9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. 1, 2, 4, 7

[10] Mohsen Fayyaz, Soroush Abbasi Koohpayegani, Farnoush Rezaei Jafari, Sunando Sengupta, Hamid Reza Vaezi Joze, Eric Sommerlade, Hamed Pirsiavash, and Jurgen Gall. Adaptive token sampling for efficient vision¨ transformers. In European Conference on Computer Vision, pages 396–414, 2022. 3

[11] Leonidas Gee, Wing Yan Li, Viktoriia Sharmanska, and Novi Quadrianto. Visual-word tokenizer: Beyond fixed sets of tokens in vision transformers. Transactions on Machine Learning Research, 2025. 4

[12] Yong Guo, David Stutz, and Bernt Schiele. Robustifying token attention for vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17557–17568, 2023. 1, 5

[13] Donghoon Han, Eunhwan Park, and Seunghyeon Seo. Visual words meet BM25: Sparse auto-encoder visual word scoring for image retrieval, 2026. 4

[14] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable´

vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16000– 16009, 2022. 8

[15] KunHo Heo, Seungjae Kim, Wongyu Lee, SuYeon Kim, and MyeongAh Cho. Rethinking prototype-based similarity learning for few-shot object detection, 2026. arXiv preprint arXiv:2606.23069. 4

[16] Zhenglun Kong, Peiyan Dong, Xiaolong Ma, Xin Meng, Wei Niu, Mengshu Sun, Xuan Shen, Geng Yuan, Bin Ren, Hao Tang, Minghai Qin, and Yanzhi Wang. SPViT: Enabling faster vision transformers via latency-aware soft token pruning. In European Conference on Computer Vision, pages 620–640, 2022. 3

[17] Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. Not all patches are what you need: Expediting vision transformers via token reorganizations. In International Conference on Learning Representations, 2022. 1, 3, 6

[18] Sifan Long, Zhen Zhao, Jimin Pi, Shengsheng Wang, and Jingdong Wang. Beyond attentive tokens: Incorporating token importance and diversity for efficient vision transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10334–10343, 2023. 3

[19] Dmitrii Marin, Jen-Hao Rick Chang, Anurag Ranjan, Anish Prabhu, Mohammad Rastegari, and Oncel Tuzel. Token pooling in vision transformers for image classification. In IEEE/CVF Winter Conference on Applications of Computer Vision, pages 12–21, 2023. 1, 3, 6

[20] Lingchen Meng, Hengduo Li, Bor-Chun Chen, Shiyi Lan, Zuxuan Wu, Yu-Gang Jiang, and Ser-Nam Lim. AdaViT: Adaptive vision transformers for efficient image recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12309–12318, 2022. 3

[21] David Nister and Henrik Stew´ enius. Scalable recognition´ with a vocabulary tree. In IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 2161–2168, 2006. 4

[22] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve´ Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and´ Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. 2

[23] Bowen Pan, Rameswar Panda, Yifan Jiang, Zhangyang Wang, Rogerio Feris, and Aude Oliva. IA-RED<sup>2</sup>: Interpretability-aware redundancy reduction for vision transformers. In Advances in Neural Information Processing Systems, 2021. 3

[24] James Philbin, Ondrej Chum, Michael Isard, Josef Sivic, and Andrew Zisserman. Object retrieval with large vocabularies and fast spatial matching. In IEEE Conference on Computer Vision and Pattern Recognition, pages 1–8, 2007. 4

[25] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. DynamicViT: Efficient vision

transformers with dynamic token sparsification. In Advances in Neural Information Processing Systems, pages 13937– 13949, 2021. 1, 3

[26] Cedric Renggli, Andre Susano Pinto, Neil Houlsby, Basil´ Mustafa, Joan Puigcerver, and Carlos Riquelme. Learning to merge tokens in vision transformers, 2022. arXiv preprint arXiv:2202.12015. 3

[27] Stephen Robertson. Understanding inverse document frequency: On theoretical arguments for IDF. Journal of Documentation, 60(5):503–520, 2004. 1, 4, 5

[28] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet large scale visual recognition challenge. International Journal of Computer Vision, 115(3): 211–252, 2015. 4, 7

[29] Michael S. Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. TokenLearner: Adaptive space-time tokenization for videos. In Advances in Neural Information Processing Systems, 2021. 3

[30] Gerard Salton and Christopher Buckley. Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5):513–523, 1988. 1, 4, 5

[31] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa,¨ Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel Sentana,´ Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve J´ egou, Patrick La-´ batut, and Piotr Bojanowski. DINOv3, 2025. arXiv preprint arXiv:2508.10104. 2, 4

[32] Josef Sivic and Andrew Zisserman. Video Google: A text retrieval approach to object matching in videos. In IEEE International Conference on Computer Vision, pages 1470– 1477, 2003. 1, 4, 5

[33] Wei-Yuan Su, Ruijie Zhang, and Zheng Zhang. Renyi en-´ tropy: A new token pruning metric for vision transformers, 2026. arXiv preprint arXiv:2603.27900. 3

[34] Yehui Tang, Kai Han, Yunhe Wang, Chang Xu, Jianyuan Guo, Chao Xu, and Dacheng Tao. Patch slimming for efficient vision transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12165–12174, 2022. 3

[35] Hugo Touvron, Matthieu Cord, and Herve J ´ egou. DeiT III:´ Revenge of the ViT. In European Conference on Computer Vision, pages 516–533. Springer, 2022. 8

[36] Matthew Walmer, Saksham Suri, Kamal Gupta, and Abhinav Shrivastava. Teaching matters: Investigating the role of supervision in vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7486–7496, 2023. 1, 6

[37] Junxiao Wang, Song Guo, Xin Xie, and Heng Qi. Federated unlearning via class-discriminative pruning. In Proceedings of the ACM Web Conference 2022, pages 622–632, 2022. 4

[38] Yifan Xu, Zhijie Zhang, Mengdan Zhang, Kekai Sheng, Ke Li, Weiming Dong, Liqing Zhang, Changsheng Xu, and

Xing Sun. Evo-ViT: Slow-fast token evolution for dynamic vision transformer. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2964–2972, 2022. 3

[39] Hongxu Yin, Arash Vahdat, Jose M. Alvarez, Arun Mallya, Jan Kautz, and Pavlo Molchanov. A-ViT: Adaptive tokens for efficient vision transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10809– 10818, 2022. 3

[40] Zichao Zeng, June Moh Goo, Junwei Zheng, Weijia Fan, Jiaming Zhang, Rainer Stiefelhagen, and Jan Boehm. Faster or stronger: Towards flexible visual place recognition via weighted aggregation and token pruning, 2026. arXiv preprint arXiv:2605.20551. 3