# Pix2Key: Controllable Open-Vocabulary Retrieval with Semantic Decomposition and Self-Supervised Visual Dictionary Learning

Guoyizhe Wei 1 Yang Jiao 2 Nan Xi 2 Zhishen Huang 2 Jingjing Meng 2 Rama Chellappa 1 Yan Gao 2

# Abstract

Composed Image Retrieval (CIR) uses a reference image plus a natural-language edit to retrieve images that apply the requested change while preserving other relevant visual content. Classic fusion pipelines typically rely on supervised triplets and can lose fine-grained cues, while recent zeroshot approaches often caption the reference image and merge the caption with the edit, which may miss implicit user intent and return repetitive results. We present Pix2Key, which represents both queries and candidates as open-vocabulary visual dictionaries, enabling intent-aware constraint matching and diversity-aware reranking in a unified embedding space. A self-supervised pretraining component, V-Dict-AE, further improves the dictionary representation using only images, strengthening fine-grained attribute understanding without CIR-specific supervision. On the DFMM-Compose benchmark, Pix2Key improves Recall@10 up to 3.2 points, and adding V-Dict-AE yields an additional 2.3-point gain while improving intent consistency and maintaining high list diversity.

# 1. Introduction

Composed image retrieval (CIR) is a multimodal search problem where a query consists of a reference image and a natural-language edit, and the system retrieves images that realize the requested change while preserving other relevant visual content. This interaction mirrors how users search in practice: shoppers seek the same garment with a different fabric or pattern, creators look for scene variations under different conditions, and designers search for layouts with localized modifications. Compared to standard text-to-image retrieval, CIR is conditional and fine-grained: it requires understanding what is meant to change, what should remain invariant, and which details define identity. Classical CIR systems are commonly trained with triplets built from reference, edit, and target images, and learn an explicit composition function for combining visual and textual signals (Vo et al., 2019). While such supervision can be effective, it is expensive to scale and can encourage a single fused representation that implicitly decides which fine-grained attributes to preserve, often in a non-transparent manner.

Recent progress in large-scale vision–language pretraining has enabled alternatives that reduce reliance on CIRspecific supervision. Contrastive pretraining in particular yields aligned image and text embeddings that generalize across domains (Radford et al., 2021). This has motivated a plethora of zero-shot CIR methods that repurpose pretrained models without triplet training. A representative line maps the reference image into a learnable textual token and composes it with the edit to form a query (Baldrati et al., 2023). Another popular practice uses a vision–language model as an image captioner, rewrites the caption according to the edit, and retrieves by matching in the language space. Despite their practicality, many zero-shot pipelines still struggle with subtle edits and localized attributes. Collapsing an image into a single token or a single sentence creates a lossy bottleneck: missing a small detail such as neckline shape, sleeve type, or a local pattern can invalidate an otherwise plausible result. In addition, ranking by similarity to a single fused query embedding often yields homogeneous top results, where near-duplicates crowd out diverse yet valid candidates. Diversity-aware reranking has a long history in information retrieval (Carbonell & Goldstein, 1998), but is rarely coupled with an intent representation that makes constraint satisfaction controllable in zero-shot CIR.

A persistent limitation lies not only in modeling, but also in evaluation. Attribute-centric datasets typically offer structured labels but lack natural-language edits, making them unsuitable for testing language-driven intent. In contrast, popular CIR benchmarks provide reference–target pairs and edit text, yet do not include fine-grained attributes for measuring how well the returned list satisfies the requested constraints beyond the single labeled target. This mismatch prevents quantifying how many non-target candidates in the top-ranked list actually satisfy the user’s requirements, and makes it difficult to analyze similarity and diversity in an attribute-grounded way. The problem is further compounded by annotation noise: in many CIR datasets, the designated target is not necessarily the best match to the edit, which can cause supervised methods to learn spurious correlations or be penalized for retrieving a better solution than the labeled target.

![](images/06f5f7472878a9629edc101adc1b13c371ec436e0bc12a90761d936bd37800ef.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Image captioner"] --> B["Replace with V-Dict-AE module"]
    C["Image captioner"] --> B
    D["Attn. pooler"] --> B
    E["LLM"] --> F["Relevance score"]
    G["Candidates"] --> H["Replace with V-Dict-AE module"]
    I["I'd like a dress like this, but in blue, no stripes"] --> J["LLM"]
    K["Text Encoder"] --> L["Diversity-aware reranking"]
    M["Text Encoder"] --> L
    N["Text Encoder"] --> L
    O["Text Encoder"] --> L
    P["Text Encoder"] --> L
    Q["Text Encoder"] --> L
    R["Text Encoder"] --> L
    S["Text Encoder"] --> L
    T["Text Encoder"] --> L
    U["Text Encoder"] --> L
    V["Text Encoder"] --> L
    W["Text Encoder"] --> L
    X["Text Encoder"] --> L
    Y["Text Encoder"] --> L
    Z["Text Encoder"] --> L
    AA["Text Encoder"] --> L
    AB["Text Encoder"] --> L
    AC["Text Encoder"] --> L
    AD["Text Encoder"] --> L
    AE["Text Encoder"] --> L
    AF["Text Encoder"] --> L
    AG["Text Encoder"] --> L
    AH["Text Encoder"] --> L
    AI["Text Encoder"] --> L
    AJ["Text Encoder"] --> L
    AK["Text Encoder"] --> L
    AL["Text Encoder"] --> L
    AM["Text Encoder"] --> L
    AN["Text Encoder"] --> L
    AO["Text Encoder"] --> L
    AP["Text Encoder"] --> L
    AQ["Text Encoder"] --> L
    AR["Text Encoder"] --> L
    AS["Text Encoder"] --> L
    AT["Text Encoder"] --> L
    AU["Text Encoder"] --> L
    AV["Text Encoder"] --> L
    AW["Text Encoder"] --> L
    AX["Text Encoder"] --> L
    AY["Image captioner"] --> Z
    AB["Image captioner"] --> Z
    AC["Image captioner"] --> Z
    AD["Image captioner"] --> Z
    AE["Image captioner"] --> Z
    AF["Image captioner"] --> Z
    AG["Image captioner"] --> Z
    AH["Image captioner"] --> Z
    AI["Image captioner"] --> Z
    AJ["Image captioner"] --> Z
    AK["Image captioner"] --> Z
    AL["Image captioner"] --> Z
    AM["Image captioner"] --> Z
    AN["Image captioner"] --> Z
    AO["Image captioner"] --> Z
    AP["Image captioner"] --> Z
    AQ["Image captioner"] --> Z
    AR["Image captioner"] --> Z
```
</details>

(a) Inference pipeline

![](images/18f33dd32a4148a5f3852c4ed6077b9f43e38abe99c7cf41ccfb52f03cba7cfe.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Qwen-VL Text Tower"] --> B["Conn. layer"]
    B --> C["Diffusion Decoder"]
    C --> D["pixel reconstruction"]
    E["Attn. pooler"] --> F["Qwen-VL Visual tower"]
    F --> G["This <image> shows <obj-1, <obj-1> has Attribute <key-1, The <key-1> of <obj-1> is <val-1"]
    H["..."] --> I["..."]
    I --> J["Qwen-VL Text Embed"]
    J --> K["..."]
    K --> L["..."]
    L --> M["..."]
    M --> N["..."]
    N --> O["..."]
    O --> P["..."]
    P --> Q["..."]
    Q --> R["..."]
    R --> S["..."]
    S --> T["..."]
    T --> U["..."]
    U --> V["..."]
    V --> W["..."]
    W --> X["..."]
    X --> Y["..."]
    Y --> Z["..."]
    Z --> AA["..."]
```
</details>

(b) V-Dict-AE self-supervised pretraining   
Figure 1: Overview of Pix2Key. (a) Inference pipeline: both the composed query and candidate images are converted into visual dictionaries for unified matching, followed by diversity-aware reranking. (b) V-Dict-AE pretraining: a self-supervised autoencoding objective learns compact visual-dictionary tokens by reconstructing images through a frozen generative decoder, improving fine-grained intent alignment for retrieval. The pretrained VLM can replace the captioner in the inference pipeline for dictionary extraction.

Pix2Key is introduced to address these challenges with a two-part design that remains free of CIR-specific triplet supervision while improving fine-grained controllability. First, images are represented as compact visual dictionaries, and edits are decomposed into structured constraints that explicitly separate attributes to satisfy, attributes to avoid, and attributes left underspecified by the user intent. The same dictionary representation is applied to the candidate database, turning retrieval into matching between two structured descriptions rather than fragile cross-modal fusion. A lightweight diversity-aware reranking stage then exposes a user-facing trade-off between strict constraint satisfaction and result variety, enabling multiple plausible completions when an edit admits more than one valid outcome. Second, V-Dict-AE improves the faithfulness of dictionary representations via self-supervised pretraining: a visual-dictionary autoencoder is trained to encode images into compact token sequences aligned with a frozen text encoder and a frozen diffusion decoder (Rombach et al., 2022). This training uses only images, and adapts a limited set of parameters through efficient low-rank updates (Hu et al., 2022), encouraging the representation to preserve the visual evidence most relevant to fine-grained retrieval.

To enable attribute-grounded evaluation of both intent satisfaction and list diversity, an additional contribution is a derived benchmark DFMM-Compose built on DeepFashion-MM (Jiang et al., 2022). It augments structured attribute labels with generated edit descriptions and auxiliary tags so that CIR queries can be evaluated not only by whether a single target is retrieved, but also by how consistently the top-ranked list satisfies the intended attributes and how diverse the returned candidates are. Together, these components support a CIR system that is more controllable, more interpretable, and more measurable under realistic, noisy supervision.

Our contributions are:

• Pix2Key, a training-free CIR framework that represents queries and candidates as visual dictionaries, making fine-grained intent constraints explicit and controllable.   
• A diversity-aware reranking mechanism integrated with the dictionary-based intent representation, enabling trade-offs between constraint satisfaction and result diversity.   
• V-Dict-AE, a self-supervised visual-dictionary autoencoder aligned with frozen text and diffusion components, preserving fine-grained evidence without CIR triplets.   
• An attribute-grounded CIR benchmark DFMM-Compose that supports quantitative evaluation of intent satisfaction and list diversity under natural-language edits.

Composed image retrieval. CIR studies retrieval where a query combines a reference image and an edit instruction. Early supervised methods learn an explicit composition function under triplet objectives (Vo et al., 2019). Later work improves image–text fusion and matching for text-feedback retrieval, including visiolinguistic attention and explicit matching (Chen et al., 2020; Delmas et al., 2022), joint visual–semantic embeddings (Chen & Bazzani, 2020), compositional query learning (Anwaar et al., 2021), and CLIP-feature-based conditioned composition (Baldrati et al., 2022). Related advances also refine the shared retrieval space with normalization or metric-learning formulations (Bogolin et al., 2022; Roth et al., 2022a;b) and explore composed retrieval in zero-shot protocols (Liu et al., 2023b), while content–style modulation further improves modeling of preserved vs. edited factors (Lee et al., 2021). Benchmarks such as FashionIQ and CIRR standardize evaluation with reference–target pairs (Wu et al., 2021; Liu et al., 2021), but supervision is often tied to a single labeled target, limiting intent assessment beyond target hit.

Tokenization-based zero-shot CIR. Zero-shot CIR often reuses a frozen CLIP-style retriever and represents the reference image as learnable tokens inside the text encoder. Pic2Word maps the image to a pseudo-word appended to the edit, enabling retrieval without CIR-specific training (Saito et al., 2023; Radford et al., 2021). SEARLE and iSEARLE cast this mapping as textual inversion, optimizing pseudo-tokens so the composed embedding aligns with the target (Baldrati et al., 2023; Gal et al., 2023; Agnolucci et al., 2025), and related token-personalization strategies are explored in PALAVRA (Cohen et al., 2022) (new). To reduce per-query optimization or increase expressiveness, FTI4CIR amortizes inversion, Context-I2W conditions tokenization on the edit context, and ISA/LinCIR move from single tokens toward sentence-level prompts while keeping CLIP-compatible indexing (Zhang et al., 2024; Tang et al., 2024; Du et al., 2024; Gu et al., 2024); prompt-centric variants further support composed retrieval (Bai et al., 2024) (new). A remaining limitation is that multiple constraints must be compressed into a small token budget, while ranking still relies on a single fused similarity score.

Training-free inference with large VLMs. Another training-free line translates visual evidence into language at inference time and retrieves in text space. CIReVL captions the reference image with a VLM, rewrites the caption conditioned on the edit, and matches the rewritten text to candidates (Karthik et al., 2024). This avoids per-query optimization and yields an interpretable intermediate query, but performance depends on caption coverage and rewriting stability, so omitted attributes or underspecified invariants can cause drift. Such pipelines build on foundation VLMs and image–text pretraining, including CLIP-style encoders and instruction-tuned multimodal models (Radford et al., 2021; Li et al., 2023; Liu et al., 2023a; Bai et al., 2025; Wang et al., 2024; Yu et al., 2022; Singh et al., 2022).

# 2. Method

# 2.1. Problem Setup

Pix2Key targets composed image retrieval (CIR), where a query is formed by a reference image together with a natural-language edit. The goal is to retrieve images that satisfy the edit while preserving other relevant visual content. Pix2Key uses an open-vocabulary visual dictionary as a shared interface for both queries and database images, so retrieval can be implemented as similarity search in a text embedding space. A self-supervised visual-dictionary autoencoder, V-Dict-AE, further refines the dictionary representation without requiring CIR triplets.

Let the retrieval database be $\mathcal { T } = \{ I _ { i } \} _ { i = 1 } ^ { N }$ . A composed query is a pair $( I _ { q } , T )$ , where $I _ { q }$ is the reference image and $T$ is a free-form edit instruction. The system returns a ranked list π over I. Throughout the paper, cosine distance is used for nearest-neighbor search,

$$
\operatorname{dist} (\mathbf {x}, \mathbf {y}) = 1 - \operatorname{cossim} (\mathbf {x}, \mathbf {y}). \tag {1}
$$

This distance is equivalent to cosine similarity up to an orderpreserving transformation, and matches the implementation used in retrieval and reranking.

# 2.2. Open-Vocabulary Visual Dictionaries

Each gallery image is converted into an open-vocabulary dictionary of attribute-like facts,

$$
\mathcal {D} _ {\mathrm{img}} (I) = \{(k _ {m}, v _ {m}) \} _ {m = 1} ^ {M}. \tag {2}
$$

where $k _ { m }$ is an attribute key (e.g., color, pattern) and $v _ { m }$ is its value (e.g., red, striped). A composed query is represented by a signed dictionary

$$
\mathcal {D} _ {q} = \left\{\left(k _ {m}, v _ {m}, p _ {m}\right) \right\} _ {m = 1} ^ {M _ {q}}, \quad p _ {m} \in \{+ 1, 0, - 1 \}. \tag {3}
$$

The intent polarity $p _ { m }$ is defined only on the query side: $p _ { m } ~ = ~ + 1$ indicates desired attributes (add/strengthen), $p _ { m } = - 1$ indicates attributes to avoid (remove/contradict), and $p _ { m } = 0$ denotes open-set anchors that are not explicitly constrained but help preserve salient context from the reference image.

A vision-language model extracts ${ \mathcal { D } } _ { \mathrm { i m g } } ( I )$ from an image using a constrained prompt format that encourages short key– value descriptions. In experiments, Qwen-VL style models are used as the extractor due to strong visual grounding and attribute coverage (Bai et al., 2025; Wang et al., 2024). For a composed query $( I _ { q } , T )$ , we first extract ${ \mathcal { D } } _ { \mathrm { r e f } } = { \mathcal { D } } _ { \mathrm { i m g } } ( I _ { q } )$ , then decompose the edit text into signed updates $\Delta \mathcal { D } ( T ) =$ $\{ ( k , v , p ) \}$ , and finally merge them:

Pix2Key: Controllable CIR via Visual Dictionaries 

<table><tr><td rowspan="2">Method</td><td colspan="2">Dresses</td><td colspan="2">Shirts</td><td colspan="2">Tops&amp;Tees</td><td colspan="2">Avg</td></tr><tr><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td></tr><tr><td colspan="9">Training-free methods:</td></tr><tr><td>Image-only</td><td>4.76</td><td>12.35</td><td>7.07</td><td>15.82</td><td>6.87</td><td>14.64</td><td>6.23</td><td>14.27</td></tr><tr><td>Text-only</td><td>14.86</td><td>34.14</td><td>19.57</td><td>33.79</td><td>21.17</td><td>39.39</td><td>18.53</td><td>35.77</td></tr><tr><td>Image + Text</td><td>13.57</td><td>31.27</td><td>14.96</td><td>26.70</td><td>19.35</td><td>33.74</td><td>15.96</td><td>30.57</td></tr><tr><td>CIReVL (Karthik et al., 2024)</td><td>23.74</td><td>45.62</td><td>29.01</td><td>47.65</td><td>30.87</td><td>52.76</td><td>27.87</td><td>48.68</td></tr><tr><td>Pix2Key</td><td>24.92</td><td>47.19</td><td>30.62</td><td>49.64</td><td>32.00</td><td>54.61</td><td>29.18</td><td>50.48</td></tr><tr><td colspan="9">With pretrained tokenization:</td></tr><tr><td>Pic2Word (Saito et al., 2023)</td><td>20.00</td><td>40.20</td><td>26.20</td><td>43.60</td><td>27.90</td><td>47.40</td><td>24.70</td><td>43.70</td></tr><tr><td>PALAVRA (Cohen et al., 2022)</td><td>17.25</td><td>35.94</td><td>21.49</td><td>37.05</td><td>20.55</td><td>38.76</td><td>19.76</td><td>37.25</td></tr><tr><td>SEARLE (Baldrati et al., 2023)</td><td>20.48</td><td>43.13</td><td>26.89</td><td>45.58</td><td>29.32</td><td>49.97</td><td>25.56</td><td>46.23</td></tr><tr><td>FTI4CIR (Zhang et al., 2024)</td><td>24.39</td><td>47.84</td><td>31.35</td><td>50.59</td><td>32.43</td><td>54.21</td><td>29.39</td><td>50.88</td></tr><tr><td>Pix2Key+V-Dict-AE</td><td>25.61</td><td>48.92</td><td>31.69</td><td>51.43</td><td>32.58</td><td>55.39</td><td>29.96</td><td>51.91</td></tr></table>

Table 1: FashionIQ composed image retrieval results, reported as Recall@10 and Recall@50 on Dresses, Shirts, and Tops&Tees, as well as the overall average. Image-only, Text-only, and Image+Text denote unimodal and naive fusion baselines under the standard evaluation protocol. Pix2Key variants are highlighted in orange. Best in each column is in bold.

$$
\mathcal {D} _ {q} = \operatorname{Merge} \left(\mathcal {D} _ {\text {ref}}, \Delta \mathcal {D} (T)\right), \tag {4}
$$

where edits override conflicting reference entries on the same key, negative entries are kept as explicit constraints, and unconstrained entries can serve as anchors that encourage preservation when the edit underspecifies invariants.

# 2.3. Text-Space Indexing from Dictionaries

Pix2Key converts database images into dictionary text offline and indexes them in a text embedding space. Compared to caption-based pipelines, we serialize only attribute-like key–value facts to reduce nuisance details and expose a controllable interface via intent polarity, while still enabling efficient offline indexing with a single text embedding per gallery item.

A dictionary is serialized into a short string ser(D) (e.g., key:value; key:value; ...). A frozen text encoder $f _ { \mathrm { t e x t } }$ maps the serialized dictionary into a global embedding:

$$
\mathbf {e} _ {i} = f _ {\text { text }} \left(\operatorname{ser} \left(\mathcal {D} \left(I _ {i}\right)\right)\right) \in \mathbb {R} ^ {d}. \tag {5}
$$

In practice, $f _ { \mathrm { t e x t } }$ is an OpenCLIP text encoder initialized from public pretrained weights; all $\mathbf { e } _ { i }$ are precomputed and stored for fast nearest-neighbor search.

For queries, $\mathcal { D } _ { q }$ is split by polarity and each subset is serialized and embedded:

$$
\operatorname{ser} (\mathcal {D} _ {q} ^ {+}), \quad \operatorname{ser} (\mathcal {D} _ {q} ^ {0}), \quad \operatorname{ser} (\mathcal {D} _ {q} ^ {-}). \tag {6}
$$

$$
\mathbf {q} ^ {+}, \mathbf {q} ^ {0}, \mathbf {q} ^ {-} \in \mathbb {R} ^ {d}. \tag {7}
$$

This keeps the retrieval space unified (queries and candidates share the same text embedding space) while preserving polarity-aware control.

# 2.4. Intent-Aware Relevance Scoring

Given a candidate embedding $\mathbf { e } _ { i }$ and query embeddings from Eq. equation 7, Pix2Key computes aligned similarity terms:

$$
p _ {i} = \operatorname{cossim} (\mathbf {q} ^ {+}, \mathbf {e} _ {i}),
$$

$$
o _ {i} = \operatorname{cossim} (\mathbf {q} ^ {0}, \mathbf {e} _ {i}), \tag {8}
$$

$$
n _ {i} = \operatorname{cossim} (\mathbf {q} ^ {-}, \mathbf {e} _ {i}).
$$

A single scalar relevance score is formed as

$$
R (i) = \alpha p _ {i} + \beta o _ {i} - (1 - \alpha) n _ {i}, \tag {9}
$$

where α balances enforcing requested changes against suppressing forbidden attributes, and $\beta$ controls how strongly unconstrained anchors are preserved.

# 2.5. Diversity-Aware Reranking

Relevance-only ranking often returns near-duplicates, so Pix2Key applies a diversity-aware reranker over a candidate pool C (see Appendix ?? for the full procedure). Let ei denote the global embedding of candidate i, and define pairwise cosine distance as in Eq. equation 1:

$$
\mathrm{dist} (i, j) = 1 - \mathrm{cossim} (\mathbf {e} _ {i}, \mathbf {e} _ {j}). \tag {10}
$$

A greedy selection set S is built by repeatedly choosing

$$
i ^ {\star} = \arg \max _ {i \in \mathcal {C} \backslash S} \left[ (1 - \lambda) R (i) + \lambda \min _ {j \in S} \mathrm{dist} (i, j) \right], \tag {11}
$$

where λ is a user-facing diversity control. Before reranking, we linearly normalize R(i) to [0, 1] over C, and rescale cosine distance as $\bar { d } ( i , j ) = \mathrm { d i s t } ( i , \bar { j } ) / 2$ for a consistent tradeoff across queries. This distance-form is equivalent to the classic similarity-form MMR objective up to a constant shift (Carbonell & Goldstein, 1998).

Pix2Key: Controllable CIR via Visual Dictionaries 

<table><tr><td rowspan="2">Method</td><td colspan="4">CIRR</td><td colspan="4">DFMM-Compose</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>AC@50</td><td>ILD@50</td><td>R@10</td><td>R@50</td></tr><tr><td colspan="9">Training-free methods:</td></tr><tr><td>CIReVL (Karthik et al., 2024)</td><td>23.94</td><td>52.51</td><td>66.00</td><td>86.95</td><td>36.42</td><td>46.44</td><td>18.12</td><td>35.40</td></tr><tr><td>CIReVL+MMR</td><td>25.17</td><td>52.93</td><td>66.12</td><td>86.45</td><td>35.79</td><td>49.26</td><td>19.30</td><td>34.82</td></tr><tr><td>Pix2Key</td><td>27.02</td><td>54.26</td><td>68.15</td><td>89.44</td><td>51.26</td><td>54.15</td><td>21.31</td><td>37.56</td></tr><tr><td colspan="9">With pretrained tokenization:</td></tr><tr><td>Pic2Word (Saito et al., 2023)</td><td>23.90</td><td>51.72</td><td>65.30</td><td>87.82</td><td>33.56</td><td>46.82</td><td>16.89</td><td>32.06</td></tr><tr><td>SEARLE (Baldrati et al., 2023)</td><td>24.20</td><td>52.40</td><td>66.30</td><td>88.60</td><td>35.75</td><td>46.19</td><td>18.94</td><td>37.35</td></tr><tr><td>FTI4CIR (Zhang et al., 2024)</td><td>25.90</td><td>55.61</td><td>67.66</td><td>89.66</td><td>38.17</td><td>47.24</td><td>19.35</td><td>37.22</td></tr><tr><td>Context-I2W (Tang et al., 2024)</td><td>25.60</td><td>55.10</td><td>68.50</td><td>89.80</td><td>39.59</td><td>45.56</td><td>20.87</td><td>37.15</td></tr><tr><td>Pix2Key+V-Dict-AE</td><td>29.06</td><td>59.44</td><td>73.36</td><td>92.08</td><td>54.44</td><td>53.96</td><td>23.58</td><td>40.96</td></tr></table>

Table 2: Results on CIRR and DFMM-Compose. CIRR is evaluated by Recall@K. DFMM-Compose reports Recall@K together with two list-level metrics computed over the top-50 retrieved candidates: AC@50 for attribute consistency and ILD@50 for intra-list diversity. Pix2Key variants are highlighted in orange. Best in each column is in bold.

# 2.6. V-Dict-AE: Self-Supervised Visual Dictionary Autoencoder

Dictionary extraction and query parsing can miss finegrained cues. V-Dict-AE improves dictionary token quality using only unlabeled images by training a parameterefficient image-to-slot module supervised through a frozen diffusion decoder, encouraging the token sequence to preserve visually salient details needed for reconstruction. The diffusion model, VAE, and the text encoder used by diffusion are frozen; only lightweight modules are trained.

Given an image I, a frozen visual tower extracts patch-level features:

$$
\mathbf {P} = g _ {\text { vis }} (I) \in \mathbb {R} ^ {B \times N \times h}, \tag {12}
$$

where B is batch size, N is the number of visual tokens, and h is the VLM hidden dimension. We obtain a fixed-length slot sequence using an attention pooler with Q learnable queries $\mathbf { Q } _ { 0 } \in \mathbb { R } ^ { Q \times h }$ replicated across the batch. The pooler concatenates queries and patch tokens and applies a Transformer encoder:

$$
\mathbf {X} = \operatorname{Enc} \big ([ \mathbf {Q} _ {0}; \mathbf {P} ] \big) \in \mathbb {R} ^ {B \times (Q + N) \times h}. \tag {13}
$$

Let $\mathbf { X } = [ \mathbf { x } _ { 1 } , \dots , \mathbf { x } _ { Q + N } ] ;$ we take the first Q tokens as pooled slots:

$$
\mathbf {Z} = \left[ \mathbf {x} _ {1}, \dots , \mathbf {x} _ {Q} \right] \in \mathbb {R} ^ {B \times Q \times h}. \tag {14}
$$

To align slot features with language semantics, slots are injected into the frozen VLM in place of its image placeholder tokens. Let y be the token ids of a prompt containing one image placeholder; the placeholder position is expanded to $Q$ repeated image-token ids to form y˜. With the VLM token embedding layer Emb(·), we replace the embeddings at image-token positions by pooled slots:

$$
\mathbf {E} = \operatorname{Emb} (\tilde {\mathbf {y}}), \quad \mathbf {E} _ {\mathrm{img}} \leftarrow \mathbf {Z}. \tag {15}
$$

A forward pass through the frozen VLM yields hidden states $\mathbf { H } \in \mathbb { R } ^ { B \times \dot { L } \times h }$ , from which we obtain Q slot-conditioned vectors $\mathbf { H } _ { \mathrm { d i c t } } \in \mathbb { R } ^ { B \times Q \times h }$ (by taking image-token positions, or by short greedy decoding and padding/truncation).

Latent diffusion models are commonly conditioned by a CLIP-like text transformer (Radford et al., 2021; Rombach et al., 2022); V-Dict-AE maps each slot into the CLIP token embedding space. Let d be the CLIP token embedding dimension. A continuous projection head produces

$$
\mathbf {u} _ {m} = \mathrm{LN} (\mathbf {W h} _ {m}) \quad \text { for } m = 1, \dots , Q, \tag {16}
$$

where $\mathbf { h } _ { m }$ is the m-th slot vector in $\mathbf { H } _ { \mathrm { d i c t } } , \mathbf { W } \in \mathbb { R } ^ { d \times h }$ is trainable, and LN is LayerNorm. A vocabulary-distributed variant predicts CLIP vocabulary logits and takes an expected embedding. Let |V| be the CLIP vocabulary size and $\mathbf { E } _ { \mathrm { C L I P } } \in \mathbb { R } ^ { | \mathcal { V } | \times d }$ be the frozen CLIP token embedding matrix:

$$
\boldsymbol {\ell} _ {m} = \mathbf {W} _ {\mathcal {V}} \mathbf {h} _ {m},
$$

$$
\mathbf {p} _ {m} = \operatorname{softmax} (\boldsymbol {\ell} _ {m} / \tau), \tag {17}
$$

$$
\mathbf {u} _ {m} = \mathbf {p} _ {m} ^ {\top} \mathbf {E} _ {\mathrm{CLIP}}.
$$

A temperature schedule sharpens the distribution during training:

$$
\tau \leftarrow \max (\tau_ {\min}, \eta \tau), \tag {18}
$$

with decay $\eta \in ( 0 , 1 )$ . A length-Q+2 soft prompt is formed by adding frozen beginning and end token embeddings:

$$
\mathbf {U} = \left[ \mathbf {u} _ {\mathrm{BOS}}; \mathbf {u} _ {1}; \dots ; \mathbf {u} _ {Q}; \mathbf {u} _ {\mathrm{EOS}} \right] \in \mathbb {R} ^ {B \times (Q + 2) \times d}. \tag {19}
$$

A frozen latent diffusion model supplies the self-supervised signal. The input image I is encoded by a frozen VAE into a latent $\mathbf { x } _ { 0 } .$ . A diffusion timestep t and Gaussian noise ϵ are sampled, producing

$$
\mathbf {x} _ {t} = \sqrt {\alpha_ {t}} \mathbf {x} _ {0} + \sqrt {1 - \alpha_ {t}} \boldsymbol {\epsilon}, (2 0)
$$

Pix2Key: Controllable CIR via Visual Dictionaries 

<table><tr><td>Method</td><td>pos.</td><td>neg.</td><td>open</td><td>MMR</td><td>R@50</td><td>ILD@50</td><td>AC@50</td></tr><tr><td>embeds pos. only</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td><td>37.27</td><td>51.65</td><td>52.49</td></tr><tr><td>embeds pos. &amp; neg.</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td><td>39.41</td><td>51.69</td><td>53.31</td></tr><tr><td>embeds w/o neg.</td><td>✓</td><td>✘</td><td>✓</td><td>✓</td><td>38.56</td><td>52.80</td><td>51.72</td></tr><tr><td>w/o MMR reranking</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>40.48</td><td>49.22</td><td>53.90</td></tr><tr><td>Pix2Key+V-Dict-AE</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>40.96</td><td>53.96</td><td>54.44</td></tr></table>

Table 3: Component ablations on DFMM-Compose. Columns pos., neg., and open indicate whether the query dictionary includes affirmative constraints, negated constraints, and open-set anchors, respectively; MMR indicates diversity-aware reranking. Our setting is highlighted in orange, and the best value in each metric column is in bold.

where $\alpha _ { t }$ denotes the cumulative noise schedule. The diffusion model is conditioned on a context produced from U:

$$
\mathbf {c} = \text { CLIPText } (\mathbf {U}). \tag {21}
$$

Following the v-parameterization (Salimans & Ho, 2022), the target is

$$
\mathbf {v} _ {t} = \sqrt {\alpha_ {t}} \boldsymbol {\epsilon} - \sqrt {1 - \alpha_ {t}} \mathbf {x} _ {0}. \tag {22}
$$

A frozen UNet predicts $\hat { \mathbf { v } } _ { \theta }$ from $\left( \mathbf { x } _ { t } , t , \mathbf { c } \right)$ :

$$
\hat {\mathbf {v}} _ {\theta} = \mathrm{UNet} (\mathbf {x} _ {t}, t, \mathbf {c}). \tag {23}
$$

The main training loss is

$$
\mathcal {L} _ {v} = \mathbb {E} _ {t, \epsilon} \left[ \| \hat {\mathbf {v}} _ {\theta} - \mathbf {v} _ {t} \| _ {2} ^ {2} \right]. \tag {24}
$$

From the v-parameterization, an estimate of the clean latent is

$$
\hat {\mathbf {x}} _ {0} = \sqrt {\alpha_ {t}} \mathbf {x} _ {t} - \sqrt {1 - \alpha_ {t}} \hat {\mathbf {v}} _ {\theta}. \tag {25}
$$

Decoding with the frozen VAE gives $\hat { I } = \mathrm { V A E } ^ { - 1 } ( \hat { \mathbf { x } } _ { 0 } )$ and a lightweight pixel loss

$$
\mathcal {L} _ {\text { pix }} = \| \hat {I} - I \| _ {1}. \tag {26}
$$

The final objective is

$$
\mathcal {L} = \mathcal {L} _ {v} + \gamma \mathcal {L} _ {\mathrm{pix}}. \tag {27}
$$

In practice, γ is small so the diffusion loss remains the primary signal while the pixel loss stabilizes reconstruction.

Only parameter-efficient modules are trained relative to full finetuning: the attention pooler, the projection head, and optional low-rank adapters inserted into the frozen VLM (Hu et al., 2022). After training, the learned pooler and adapters are reused by the dictionary extractor at inference by replacing raw patch embeddings with pooled slots from Eq. equation 14, improving fine-grained attribute capture while preserving the retrieval interface in Sections 2.2–2.5.

# 3. Experiments

# 3.1. Experimental Setting

Overview. Pix2Key is an open-vocabulary dictionary retrieval system for CIR. At inference time, a pretrained vision–language model converts the reference image and gallery images into compact visual dictionaries, and the edit text is decomposed into polarity-aware constraints. Both query and gallery dictionaries are embedded with an Open-CLIP text encoder pretrained on LAION-2B (Ilharco et al., 2021), enabling nearest-neighbor retrieval in a shared text space, followed by MMR reranking for controllable diversity. V-Dict-AE is an optional self-supervised pretraining module that improves the dictionary representation by training a parameter-efficient slot encoder on COCO2017 (Lin et al., 2014) or FashionAI (Zou et al., 2019), while keeping the same inference-time retrieval interface.

![](images/2bdbd9044394aef077abb99ae5875531ebdc1c9f48c5ef21def4f3c0f1aa249d.jpg)

<details>
<summary>text_image</summary>

ref. image mod. caption retrieval result top-4
Show me dresses with a graphic pattern on the upper part.
Could you show me a version with short sleeves?
Ours Baseline
</details>

Figure 2: Qualitative comparison of composed retrieval results. Each example shows the reference image, the modification text, and the top-4 retrieved candidates.

Evaluation covers FashionIQ (Wu et al., 2021) and CIRR (Liu et al., 2021) for standard Recall@K, and DFMM-Compose for attribute-grounded intent satisfaction and list diversity. Comparisons include tokenization-based zeroshot methods and training-free caption-rewrite pipelines. Full details of compute, model components and weights, pretraining configuration, benchmarks, baselines, and DFMM-Compose construction are provided in Appendix ??.

DFMM-Compose Benchmark. DFMM-Compose is an attribute-grounded composed retrieval benchmark derived from DeepFashion-MM (Jiang et al., 2022). It is designed to evaluate not only whether the annotated target is retrieved, but also how well the returned candidate list satisfies finegrained edit intent and how diverse the list remains. Each query consists of a reference image and a natural-language edit, while each gallery image is associated with structured attribute labels that enable intent-consistency scoring over the top-ranked results. This format supports standard Recall@K evaluation, attribute-consistency evaluation beyond a single target, and list-level diversity analysis for userfacing retrieval. Construction details and evaluation protocol are given in Appendix ??.

Metrics. FashionIQ and CIRR are evaluated using Recall@K, which measures whether the annotated target appears among the top-K results. DFMM-Compose additionally reports two list-level metrics on the top-50 candidates: AC@50 quantifies how well returned images satisfy attribute changes implied by the edit, and ILD@50 measures redundancy within the list using attribute-based distances. Higher values indicate better performance across metrics. Full definitions are provided in Appendix ??.

# 3.2. Main Results

Accuracy. Tables 1 and 2 summarize retrieval accuracy across FashionIQ, CIRR, and DFMM-Compose. On FashionIQ (Table 1), Pix2Key consistently improves over unimodal and naive fusion baselines, and is competitive among training-free approaches, outperforming CIReVL under the same Qwen2.5-VL backbone used for fair comparison (Appendix ??). Notably, CIReVL represents the composed query by first generating an image caption for the reference and then rewriting it with the edit prompt, so the retrieval signal is mediated by a single free-form sentence in the text space. The consistent gap to Pix2Key therefore suggests that replacing caption-level descriptions with a structured key–value dictionary interface can yield a more stable and controllable representation for composed retrieval, especially when the edit requires fine-grained attribute changes.

Compared to methods that rely on pretrained tokenization or inversion-style modules, Pix2Key+V-Dict-AE achieves the strongest results across all FashionIQ categories and the overall average, indicating that self-supervised token refinement complements dictionary-based retrieval. On CIRR (Table 2), Pix2Key improves Recall@K over the trainingfree caption-rewrite baseline, and the pretrained variant further raises performance, yielding the best Recall@1/5/10/50 among all compared methods. On DFMM-Compose, the same trend holds: Pix2Key improves Recall@10/50 over prior baselines, and V-Dict-AE brings additional gains, suggesting that the representation improvement transfers beyond a single benchmark style and remains compatible with the same nearest-neighbor retrieval interface.

Intent alignment. DFMM-Compose enables attributegrounded evaluation beyond target hit, since it provides fine-grained attribute labels for all gallery candidates, allowing us to quantify whether other retrieved items also satisfy the intended edit. As shown in Table 2, prior baselines yield relatively limited attribute consistency, while Pix2Key achieves substantially higher AC@50, suggesting that polarity-aware constraints and dictionary matching better capture fine-grained intent than caption-based rewriting or token-only fusion.

This difference is consistent with the fact that captionrewrite pipelines (e.g., CIReVL) compress the reference evidence and the edit into a single rewritten sentence, which may under-specify attributes that should be preserved or suppressed, whereas the dictionary representation keeps explicit key–value evidence and separates desired, avoided, and open anchors. Adding V-Dict-AE further improves AC@50 together with Recall, indicating that reconstructionshaped pretraining helps preserve the visual evidence that matters for downstream attribute-level satisfaction under composed edits.

Diversity. DFMM-Compose also supports list-level analysis because all candidates carry fine-grained attributes (and auxiliary tags), making it possible to measure redundancy and within-list variation in an interpretable space. In Table 2, Pix2Key achieves the highest ILD@50 among compared methods, indicating a less redundant candidate list under the same retrieval budget. Pix2Key+V-Dict-AE maintains similarly high ILD@50 while improving Recall@K and AC@50, suggesting that representation refinement does not collapse retrieval neighborhoods and remains compatible with diversity-aware reranking. This behavior is consistent with Pix2Key’s design: intent is scored with polarity-aware relevance, while diversity is controlled at the list level via MMR, allowing multiple plausible results without drifting away from the edit intent.

# 3.3. Ablations

Components. Table 3 indicates that relying on affirmative constraints alone offers limited leverage for fine-grained edits, and yields the weakest Recall@50 together with a comparatively low AC@50. Adding explicit negation increases both measures, suggesting that describing attributes to avoid helps separate visually plausible but intent-violating candidates from true matches, especially when edits involve subtle attribute swaps. Introducing open-set anchors without negation tends to raise ILD@50, reflecting a broader and less redundant candidate set, but AC@50 drops in this setting, implying that anchors emphasize preserving general context and may weaken attribute specificity when conflicting cues are not suppressed. When affirmative, negative, and open-set signals are used together, Recall@50 increases further and AC@50 remains competitive, supporting the view that anchors are most effective when paired with explicit suppression to maintain a clearer notion of the intended change.

<table><tr><td>learn. rate</td><td>R@5</td></tr><tr><td> $5 \times 10^{-5}$ </td><td>59.44</td></tr><tr><td> $1 \times 10^{-5}$ </td><td>58.50</td></tr><tr><td> $5 \times 10^{-6}$ </td><td>57.93</td></tr></table>

(a) Learning rate.

<table><tr><td>depth</td><td>R@5</td></tr><tr><td>3</td><td>59.01</td></tr><tr><td>5</td><td>59.44</td></tr><tr><td>9</td><td>56.90</td></tr></table>

(b) Pooler depth.

<table><tr><td>input size</td><td>R@5</td></tr><tr><td>64</td><td>58.01</td></tr><tr><td>128</td><td>58.63</td></tr><tr><td>224</td><td>59.44</td></tr></table>

(c) Input size.

<table><tr><td>setting</td><td>R@5</td></tr><tr><td>original</td><td>58.27</td></tr><tr><td>refined prompt</td><td>59.44</td></tr></table>

(d) Refined prompt.

<table><tr><td>setting</td><td>R@5</td></tr><tr><td>w/o LoRA</td><td>57.20</td></tr><tr><td>with LoRA</td><td>59.44</td></tr></table>

(e) LoRA.

<table><tr><td>setting</td><td>R@5</td></tr><tr><td>ViT-B/32</td><td>59.44</td></tr><tr><td>ViT-L/14</td><td>59.12</td></tr></table>

(f) Text encoder.   
Table 4: Sensitivity analysis on CIRR, reported as Recall@5. Each subtable varies a single factor while keeping the remaining settings fixed. The default configuration in each sub-study is highlighted in orange, and the best value is shown in bold. Text encoder denotes the backbone used for embedding dictionary text into the retrieval space.

Diversity control. Table 3 also isolates the effect of diversity-aware reranking under the same intent representation. With MMR enabled, ILD@50 increases markedly, while Recall@50 and AC@50 remain stable and slightly improve in this ablation, suggesting that reranking can diversify the top-50 list without noticeably compromising intent satisfaction in this setting. Notably, applying MMR reranking to CIReVL slightly lowers Recall@50 on both CIRR and DFMM-Compose, suggesting that caption-and-rewrite representations provide a less stable relevance signal under diversity control, whereas Pix2Key’s dictionary-based representation is more compatible with MMR and can diversify results without the same degree of ranking degradation.

Sensitivity. Table 4 suggests that Pix2Key is not overly brittle on CIRR under reasonable hyperparameter changes, as measured by Recall at five. A moderate learning rate performs best, indicating that the trainable dictionary modules benefit from sufficiently strong updates under a frozen backbone. Pooler depth shows a clear sweet spot in the middle range. Depth 5 achieves 59.44, depth 3 remains close at 59.01, and depth 9 drops to 56.90, which is consistent with the pooler acting as a lightweight summarizer rather than a full feature re encoder. Higher input resolution consistently improves performance. Increasing the resolution from 64 to 224 raises Recall at five from 58.01 to 59.44, suggesting that finer local evidence helps preserve subtle attributes under natural language edits. Prompt refinement also improves results, increasing the score from 58.27 to 59.44, and enabling LoRA provides an additional gain from 57.20 to 59.44, supporting the view that better extraction and targeted adaptation improve alignment without full finetuning. Finally, the OpenCLIP text encoder choice is stable in this study. ViT-B/32 reaches 59.44 and ViT-L/14 reaches 59.12. Overall, the strongest gains come from factors that improve fine grained evidence and representation alignment, which matches the intended design of Pix2Key and V-Dict-AE.

# 4. Conclusion

We introduce Pix2Key, a controllable composed image retrieval framework that represents both queries and candidates as open vocabulary visual dictionaries. The edit instruction is decomposed into intent signals that specify what to add, what to avoid, and what to keep open as anchors. This design provides an explicit and interpretable control interface, while reducing retrieval to fast nearest neighbor search in a shared text embedding space. To support user facing exploration, Pix2Key applies diversity aware reranking that increases variety among the top results without sacrificing relevance. We further propose V-Dict-AE, a parameter efficient self supervised module that refines dictionary slots through reconstruction based supervision, strengthening fine grained visual evidence without requiring composed retrieval triplets. Experiments show consistent improvements in Recall at K, attribute consistency, and intra list diversity over strong zero shot baselines and competitive caption based retrieval pipelines. Pix2Key also enables attribute retrieval and diagnostic analysis through its dictionary interface. Overall, Pix2Key provides a practical and scalable approach to intent aware composed image retrieval.

# Impact Statement

This paper aims to advance machine learning research in composed image retrieval by improving alignment between a reference image and a natural-language edit, as well as the diversity and consistency of retrieved results. The expected positive impact is to enable more controllable and efficient search for benign applications such as e-commerce, creative design, and visual content organization.

Our method is trained with self-supervision on public datasets, which reduces reliance on human-provided labels and may lessen risks from annotation errors. However, the learned representations can still reflect biases in the underlying data distributions. Additional risks include retrieval of sensitive or copyrighted content, or privacy-invasive use when combined with external data sources. We encourage responsible deployment with content filtering, access control, dataset governance, and auditing practices. Overall, we do not anticipate ethical concerns beyond those commonly associated with vision-language retrieval systems, but responsible use remains important.

# References

Agnolucci, L., Baldrati, A., Del Bimbo, A., and Bertini, M. isearle: Improving textual inversion for zero-shot composed image retrieval. TPAMI, 2025.   
Anwaar, M. U., Labintcev, E., and Kleinsteuber, M. Compositional learning of image-text query for image retrieval. In WACV, 2021.   
Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. Qwen2.5-vl technical report. arXiv preprint, 2025.   
Bai, Y., Xu, X., Liu, Y., Khan, S., Khan, F., Zuo, W., Goh, R. S. M., and Feng, C.-M. Sentence-level prompts benefit composed image retrieval. In ICLR, 2024.   
Baldrati, A., Bertini, M., Uricchio, T., and Del Bimbo, A. Effective conditioned and composed image retrieval combining clip-based features. In CVPR, 2022.   
Baldrati, A., Agnolucci, L., Bertini, M., and Del Bimbo, A. Zero-shot composed image retrieval with textual inversion. In ICCV, 2023.   
Bogolin, S.-V., Croitoru, I., Jin, H., Liu, Y., and Albanie, S. Cross modal retrieval with querybank normalisation. In CVPR, 2022.   
Carbonell, J. and Goldstein, J. The use of MMR, diversitybased reranking for reordering documents and producing summaries. In ACM SIGIR, 1998.   
Chen, Y. and Bazzani, L. Learning joint visual semantic matching embeddings for language-guided retrieval. In ECCV, 2020.

Chen, Y., Gong, S., and Bazzani, L. Image search with text feedback by visiolinguistic attention learning. In CVPR, 2020.

Cohen, N., Gal, R., Meirom, E. A., Chechik, G., and Atzmon, Y. “this is my unicorn, fluffy”: Personalizing frozen vision-language representations. In ECCV, 2022.

Delmas, G., Rezende, R. S., Csurka, G., and Larlus, D. Artemis: Attention-based retrieval with text-explicit matching and implicit similarity. In ICLR, 2022.

Du, Y., Wang, M., Zhou, W., Hui, S., and Li, H. Image2sentence based asymmetrical zero-shot composed image retrieval. In ICLR, 2024.

Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A. H., Chechik, G., and Cohen-Or, D. An image is worth one word: Personalizing text-to-image generation using textual inversion. ICLR, 2023.

Gu, G., Chun, S., Kim, W., Kang, Y., and Yun, S. Languageonly training of zero-shot composed image retrieval. In CVPR, 2024.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., and Wang, L. LoRA: Low-rank adaptation of large language models. In ICLR, 2022.

Ilharco, G., Wortsman, M., Carlini, N., Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi, H., Farhadi, A., et al. Openclip. Zenodo, 2021.

Jiang, Y., Yang, S., Qiu, H., Wu, W., Loy, C. C., and Liu, Z. Text2human: Text-driven controllable human image generation. ACM TOG, 2022.

Karthik, S., Roth, K., Mancini, M., and Akata, Z. Vision-bylanguage for training-free compositional image retrieval. In ICLR, 2024.

Lee, S., Kim, D., and Han, B. Cosmo: Content-style modulation for image retrieval with text feedback. In CVPR, 2021.

Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L. Microsoft ´ COCO: Common objects in context. In ECCV, 2014.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. NeurIPS, 2023a.

Liu, Y., Yao, J., Zhang, Y., Wang, Y., and Xie, W. Zero-shot composed text-image retrieval. In BMVC, 2023b.

Liu, Z., Rodriguez-Opazo, C., Teney, D., and Gould, S. Image retrieval on real-life images with pre-trained visionand-language models. In ICCV, 2021.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In ICML, 2021.   
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.   
Roth, K., Vinyals, O., and Akata, Z. Integrating language guidance into vision-based deep metric learning. In CVPR, 2022a.   
Roth, K., Vinyals, O., and Akata, Z. Non-isotropy regularization for proxy-based deep metric learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022b.   
Saito, K., Sohn, K., Zhang, X., Li, C.-L., Lee, C.-Y., Saenko, K., and Pfister, T. Pic2word: Mapping pictures to words for zero-shot composed image retrieval. In CVPR, 2023.   
Salimans, T. and Ho, J. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022.   
Singh, A., Hu, R., Goswami, V., Couairon, G., Galuba, W., Rohrbach, M., and Kiela, D. Flava: A foundational language and vision alignment model. In CVPR, 2022.   
Tang, Y., Yu, J., Gai, K., Zhuang, J., Xiong, G., Hu, Y., and Wu, Q. Context-i2w: Mapping images to contextdependent words for accurate zero-shot composed image retrieval. In AAAI, 2024.   
Vo, N., Jiang, L., Sun, C., Murphy, K., Li, L.-J., Fei-Fei, L., and Hays, J. Composing text and image for image retrieval—an empirical odyssey. In CVPR, 2019.   
Wang, P., Bai, S., Zhu, H., Zhang, H., Xu, K., Wang, W., Li, B., Zhu, J., Yang, G., et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint, 2024.   
Wu, H., Gao, Y., Guo, X., Al-Halah, Z., Rennie, S., Grauman, K., and Feris, R. Fashion iq: A new dataset towards retrieving images by natural language feedback. In CVPR, 2021.   
Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., and Wu, Y. Coca: Contrastive captioners are imagetext foundation models. TMLR, 2022.

Zhang, W., Zhou, M., Wang, X., Lan, X., Cheng, M.-M., Feng, W., and Jin, L. Fine-grained textual inversion network for zero-shot composed image retrieval. In ACM SIGIR, 2024.   
Zou, X., Kong, X., Wong, W., Wang, C., Liu, Y., and Cao, Y. Fashionai: A hierarchical dataset for fashion understanding. In CVPR Workshops, 2019.