# Structured Hyperedge Adaptation for Parameter-Eficient Fine-Tuning of Vision Transformers

Edwin Kwadwo Tenagyei<sup>1⋆</sup> , Lei Wang<sup>1,2⋆</sup> , Ugochukwu Ejike Akpudo<sup>1</sup> , Jun Zhou<sup>1</sup> , and Yongsheng Gao<sup>1⋆⋆</sup>

<sup>1</sup> Grifith University, Nathan QLD 4111, Australia

<sup>2</sup> Data61/CSIRO, Canberra ACT 2601, Australia

Abstract. Parameter-eficient fine-tuning (PEFT) has become a practical solution for adapting large pretrained vision transformers (ViTs) to downstream tasks while updating only a small subset of parameters. However, existing adapter-based methods perform adaptation independently for each token, implicitly assuming that token refinements should be learned in isolation. This token-wise formulation overlooks the structured relationships among tokens that naturally arise in visual scenes, potentially leading to redundant updates and spatially inconsistent feature refinement. In this work, we revisit the design of parametereficient adapters and propose to perform adaptation in hyperedge space rather than token space. We introduce HyperAdapter, a hypergraphbased adapter architecture that enables structured, group-aware adaptation through soft token routing. HyperAdapter constructs a soft hypergraph over ViT tokens using prototype-based assignments, aggregates token features into latent hyperedge representations, applies lightweight bottleneck adaptation at the hyperedge level, and difuses the resulting updates back to tokens via the hypergraph incidence structure. This design injects an explicit structural inductive bias into PEFT while preserving the modularity and eficiency of standard adapters. Extensive experiments across diverse visual benchmarks demonstrate that structured hyperedge adaptation consistently outperforms strong PEFT baselines under comparable parameter budgets, with particularly pronounced gains on tasks requiring structured reasoning. Our results suggest that the choice of adaptation space is a critical yet underexplored dimension in parameter-eficient transfer for ViTs.

Keywords: Parameter-eficient fine-tuning · Vision transformers · Hypergraph learning · Adapter networks · Structured token adaptation

## 1 Introduction

Large pretrained vision transformers (ViTs) [7, 15, 29, 30, 44] have become the backbone of modern visual recognition systems, delivering remarkable performance across diverse tasks and domains. However, adapting these large-scale models to new datasets through full fine-tuning remains computationally expensive and memory intensive, particularly when multiple downstream tasks need to be supported. Parameter-eficient fine-tuning (PEFT) methods [17,18,21,26] address this challenge by updating only a small subset of parameters while keeping the pretrained backbone frozen. Among them, adapter-based approaches [3, 17, 22, 24] have emerged as a practical and modular solution, inserting lightweight trainable modules into transformer blocks to enable eficient transfer.

Despite their empirical success, existing adapter designs share a largely overlooked assumption: adaptation is performed independently for each token. In standard formulations, token representations are refined through low-rank or bottleneck transformations applied identically and independently across tokens. Although the frozen self-attention layers encode contextual interactions, the adaptation mechanism itself remains token-wise. This design implicitly assumes that feature refinement should occur in token space, without explicitly modeling the structured relationships that naturally arise among tokens in visual scenes. In practice, image tokens often correspond to coherent regions such as objects, parts, or semantic components. Ignoring such higher-order structure during adaptation may lead to redundant updates, spatially inconsistent feature refinement, and limited utilization of relational information.

In this work, we revisit the design principle of parameter-eficient adapters and argue that the choice of adaptation space is a critical yet underexplored dimension in PEFT. Instead of refining tokens independently, we propose to perform adaptation in a structured interaction space that explicitly captures higher-order token relationships. To this end, we introduce HyperAdapter, a hypergraph-based adapter architecture that operates in hyperedge space rather than token space. HyperAdapter constructs a soft hypergraph over ViT patch tokens using prototype-based routing, where each token is softly assigned to a small set of latent hyperedges according to representation similarity. These hyperedges serve as structured groups that aggregate information from multiple related tokens. Token features are first pooled into compact hyperedge representations, which are then refined through a lightweight bottleneck adapter at the hyperedge level. The adapted hyperedge features are subsequently difused back to token representations via the hypergraph incidence structure. This hyperedgelevel adaptation introduces an explicit structural inductive bias, encouraging coherent and group-aware feature updates while preserving the eficiency and modularity of standard adapters.

Importantly, HyperAdapter is a drop-in replacement for conventional adapters. It requires no modification to the pretrained backbone, introduces only a modest number of additional parameters, and incurs minimal computational overhead relative to the frozen self-attention layers. Moreover, we show that token-wise adapters arise as a special case of our formulation, demonstrating that Hyper-Adapter generalizes standard designs while retaining their favorable properties, including permutation equivariance and low-rank adaptation structure. We evaluate HyperAdapter on 24 downstream tasks spanning diverse visual domains under parameter-eficient settings. Across multiple backbones and benchmarks, HyperAdapter consistently outperforms strong adapter-based baselines and other PEFT methods under comparable parameter budgets, with particularly notable gains on tasks requiring structured reasoning. These results indicate that modeling structured token interactions during adaptation provides a simple yet efective mechanism for enhancing parameter-eficient transfer in vision transformers. Our main contributions are summarized as follows:

i. We revisit the design of parameter-eficient adapters and identify the adaptation space as a critical dimension, highlighting the limitations of token-wise refinement.

ii. We propose HyperAdapter, a hypergraph-based adapter architecture that performs structured adaptation in hyperedge space via prototype-based token grouping.

iii. We develop a group-level adaptation mechanism that aggregates token representations into hyperedges, applies lightweight bottleneck refinement at the hyperedge level, and difuses structured updates back to tokens.

iv. We conduct extensive experiments across 24 downstream tasks and multiple transformer backbones, demonstrating consistent and parameter-eficient improvements over strong PEFT baselines.

## 2 Related Work

PEFT for ViTs. The rapid scaling of ViTs [7, 15, 29, 30, 44] has significantly increased the computational and memory cost of adaptation. Full fine-tuning updates all model parameters and typically achieves strong performance, but becomes impractical when deploying models across many downstream tasks. PEFT methods aim to reduce adaptation cost by updating only a small subset of parameters while keeping the pretrained backbone largely frozen. Existing PEFT approaches for ViTs can be broadly grouped into several families. Prompt-based methods [12, 21, 39, 40, 46, 48] introduce learnable prompt tokens that interact with the self-attention mechanism to steer task adaptation without modifying backbone weights. Adapter-based methods [3,5,6,17,22,24,31] insert lightweight bottleneck modules inside transformer blocks to refine intermediate representations. Selective tuning methods [4, 50, 53] update only specific existing parameters, such as biases or carefully chosen weight subsets. Reparameterization-based methods [1, 16, 18, 20, 23, 26, 28, 33, 47] introduce low-rank or structured weight decompositions that can be merged into pretrained weights at inference time. Hybrid strategies [49,52] combine multiple PEFT paradigms to improve flexibility and expressiveness. Despite their methodological diferences, most existing PEFT approaches share a common design principle: adaptation is performed independently for each token. In adapter-based and low-rank methods, the same transformation is applied token-wise to all patch embeddings. Prompt-based methods alter attention interactions but still refine token features individually after attention mixing. None of these approaches explicitly model structured group-level interactions during adaptation.

Our work departs from this paradigm by revisiting the adaptation space itself. Rather than refining token embeddings independently, we perform adaptation in a structured hyperedge space that aggregates and refines groups of tokens jointly. This design introduces an explicit relational inductive bias during parametereficient fine-tuning, while remaining fully compatible with standard adapter formulations. Importantly, conventional token-wise adapters arise as a special case of our formulation when each hyperedge contains a single token, showing that our method strictly generalizes existing adapter designs.

Structured adaptation and token interaction modeling. Several recent works have explored modeling structured relationships within ViTs. Selfattention inherently captures pairwise token interactions, and extensions such as dynamic routing or token clustering have been proposed to enhance relational reasoning. In PEFT settings, some approaches attempt to improve adaptation by modifying attention maps or introducing task-specific routing strategies. However, these methods typically alter attention behavior within the backbone rather than redefining where adaptation occurs. More closely related are approaches that incorporate grouping or clustering mechanisms into transformer processing. For instance, graph-based vision transformers [13, 35, 36] construct graphs over image patches to model spatial relationships, while token clustering methods aggregate patch tokens to reduce redundancy or improve eficiency. Nevertheless, these works focus on backbone architecture design or inference eficiency, not PEFT. They modify the transformer’s core computation rather than introducing a structured adaptation module on top of a frozen backbone.

In contrast, our method leaves the pretrained transformer unchanged and introduces structured modeling within the adapter module. The hypergraph construction in HyperAdapter serves as a task-adaptive grouping mechanism used solely for adaptation, without altering the underlying attention layers. This separation allows us to inject higher-order structure into PEFT while preserving the modularity and deployment advantages of adapter-based methods.

Graph and hypergraph learning. GNNs [11, 42, 45] were originally developed for relational data and have since been extended to vision tasks [13, 35, 36]. Vision Graphs (ViGs) [13] model image patches as graph nodes connected through learned or predefined edges, enabling relational reasoning beyond gridbased convolutions. To reduce the cost of dynamic graph construction, later works such as MobileViG [35] and GreedyViG [36] adopt static or simplified graph structures. Hypergraphs extend standard graphs by allowing hyperedges to connect more than two nodes, enabling higher-order relational modeling. Hypergraph-based methods have been explored for visual tasks including 3D understanding and video modeling [10,19,32], and more recently integrated into vision GNNs [9, 14, 43] to capture complex multi-way relationships.

However, existing graph and hypergraph approaches primarily target backbone design or relational feature learning from scratch. They typically replace or augment convolutional or transformer layers with graph-based computation. In contrast, our work operates in a fundamentally diferent regime. We do not redesign the backbone nor introduce heavy graph convolution layers. Instead, we use a soft hypergraph constructed over frozen token embeddings to perform structured adaptation within a bottleneck module. To our knowledge, this is the first work that formulates PEFT as hyperedge-level adaptation in a structured interaction space. By combining hypergraph modeling with PEFT, Hyper-Adapter bridges relational representation learning and eficient model adaptation in a unified and modular framework.

![](images/211b4214eb08c7e358782143262717767f6e28664163c575fd67f7b23216c83a.jpg)  
Fig. 1: HyperAdapter framework. We introduce HyperAdapter, a parameter-eficient adaptation module that operates in a structured interaction space. Given patch tokens from a frozen vision transformer, a routing mechanism softly groups tokens into hyperedges, capturing higher-order relationships among visually related regions. Each hyperedge aggregates information from multiple tokens and is refined using a lightweight bottleneck adapter, enabling group-level adaptation rather than independent token updates. The refined hyperedge features are then difused back to tokens, producing structured and coherent feature updates. By shifting adaptation from individual tokens to token groups, HyperAdapter injects relational inductive bias while maintaining the eficiency and modularity of standard PEFT methods.

## 3 Method

We now introduce our method. We begin by revisiting the adaptation space.

## 3.1 Revisiting the Adaptation Space

Let $f _ { \theta }$ denote a pretrained ViT with frozen parameters θ. For an input image, the transformer produces token embeddings $\mathbf { \bar { X } } = [ \pmb { x } _ { \mathrm { c l s } } , \pmb { x } _ { 1 } , \dots , \pmb { x } _ { N } ] \in \mathbf { \bar { \mathbb { R } } } ^ { ( N + 1 ) \times \mathbf { \bar { D } } }$ where $\pmb { x } _ { i } \in \mathbb { R } ^ { D }$ are patch tokens and D is the hidden dimension. In standard PEFT, each token is refined independently via a bottleneck transformation:

$$
\Delta \pmb {x} _ {i} = \pmb {W} _ {\mathrm{up}} \sigma (\pmb {W} _ {\mathrm{down}} \pmb {x} _ {i}), \qquad r \ll D.\tag{1}
$$

This formulation implicitly assumes that adaptation occurs in token space, where each token is updated in isolation. We argue that this token-wise adaptation is inherently limiting. Visual tokens often correspond to coherent regions such as objects, object parts, or semantic components, naturally forming higher-order structures. Ignoring these relationships can lead to redundant updates, spatially inconsistent refinements, and underutilization of relational information encoded in the visual scene. To overcome this limitation, we propose performing adaptation in a structured interaction space, where groups of related tokens are refined jointly. By shifting the adaptation focus from individual tokens to token groups, this perspective enables parameter-eficient updates that respect the intrinsic structure of visual inputs, yielding more coherent and semantically consistent feature refinements. We next describe our hyperedge-space adaptation.

## 3.2 Hyperedge-Space Adaptation

We introduce HyperAdapter (Fig. 1), a parameter-eficient adaptation mechanism that operates in hyperedge space rather than directly on individual tokens. Unlike conventional token-wise adapters, which refine each token independently, HyperAdapter first organizes tokens into structured groups and performs adaptation at the group level. This design introduces an explicit relational inductive bias, enabling updates that respect higher-order relationships among tokens while preserving the modularity and eficiency of standard adapters.

Formally, let $\pmb { X } _ { p } = [ \pmb { x } _ { 1 } , \dots , \pmb { x } _ { N } ] \in \mathbb { R } ^ { N \times D }$ denote the patch token embeddings produced by the frozen ViT. To capture structured interactions among tokens, we introduce K learnable prototype vectors

$$
\boldsymbol {E} = \left[ \boldsymbol {e} _ {1}, \dots , \boldsymbol {e} _ {K} \right] ^ {\top} \in \mathbb {R} ^ {K \times D},\tag{2}
$$

which serve as latent hyperedges representing compact groups of tokens.<sup>3</sup> Each token is softly assigned to these hyperedges based on representation similarity using temperature-scaled cosine routing:

$$
\pmb {M} _ {i k} = \frac {\exp (\langle \hat {\pmb {x}} _ {i} , \hat {\pmb {e}} _ {k} \rangle / \tau)}{\sum_ {j = 1} ^ {K} \exp (\langle \hat {\pmb {x}} _ {i} , \hat {\pmb {e}} _ {j} \rangle / \tau)},\tag{3}
$$

where $M \in \mathbb { R } ^ { N \times K }$ is the soft incidence matrix and $\tau > 0$ controls the sharpness of assignments. This construction efectively models token-to-hyperedge relationships, allowing each token to contribute to multiple groups in a diferentiable manner. Once the hypergraph is defined, each hyperedge representation is computed as a normalized weighted average of the tokens assigned to it:

$$
\boldsymbol {H} = \left(\boldsymbol {M} ^ {\top} \boldsymbol {X} _ {p}\right) \oslash \left(\boldsymbol {M} ^ {\top} \mathbf {1}\right), \quad \boldsymbol {H} \in \mathbb {R} ^ {K \times D},\tag{4}
$$

where $\oslash$ denotes row-wise normalization. Each hyperedge thus aggregates information from multiple related tokens, forming a compact group-level representation that captures higher-order context. Adaptation is performed in hyperedge space using a lightweight bottleneck module:

$$
\Delta \boldsymbol {H} = \boldsymbol {W} _ {\mathrm{up}} \sigma (\boldsymbol {W} _ {\mathrm{down}} \boldsymbol {H}),\tag{5}
$$

where $W _ { \mathrm { d o w n } } \in \mathbb { R } ^ { r \times D }$ and $W _ { \mathrm { u p } } \in \mathbb { R } ^ { D \times r }$ . Operating on hyperedges rather than individual tokens enables group-level feature refinement, allowing relational information to propagate across tokens within each group. Finally, the hyperedge updates are difused back to token space through the soft incidence matrix:

$$
\Delta \pmb {X} = \pmb {M} \Delta \pmb {H}, \qquad \pmb {X} _ {p} ^ {\prime} = \pmb {X} _ {p} + \alpha \Delta \pmb {X},\tag{6}
$$

where α is a learnable scaling parameter. This difusion ensures that each token receives structured updates informed by the hyperedges it belongs to. The classification token remains unchanged throughout this process.

HyperAdapter introduces structured, group-aware adaptation while maintaining the eficiency and modularity of standard PEFT methods. It generalizes token-wise adapters, which are recovered as a special case when each hyperedge contains a single token. $\mathrm { B y }$ refining groups of tokens jointly, HyperAdapter injects an explicit relational inductive bias into parameter-eficient adaptation, leading to more coherent and semantically consistent feature updates.

## 3.3 Unified View of HyperAdapter Generalization

HyperAdapter provides a unified view of parameter-eficient adaptation by generalizing standard token-wise adapters. In the limiting case where each token forms an independent hyperedge with no cross-token aggregation, the method reduces exactly to a token-wise adapter.

Proposition 1 (Token-wise adapter). If the number of hyperedges equals the number of tokens $( K = N )$ and the soft incidence matrix is the identity $( M = I _ { N } ) _ { ; }$ , HyperAdapter reduces to a standard token-wise adapter.

Proof. When $M = I _ { N }$ , the hyperedge aggregation step yields $\pmb { H } = \pmb { M } ^ { \top } \pmb { X } _ { p } =$ $X _ { p }$ , and the difusion step gives $\varDelta X = M \varDelta H = \varDelta H$

Thus, each token is independently refined:

$$
\pmb {x} _ {i} ^ {\prime} = \pmb {x} _ {i} + \Delta \pmb {x} _ {i} = \pmb {x} _ {i} + \pmb {W} _ {\mathrm{up}} \sigma (\pmb {W} _ {\mathrm{down}} \pmb {x} _ {i}),
$$

which exactly recovers the standard token-wise adapter.

We now show that HyperAdapter treats patch tokens symmetrically: reordering the tokens results in a correspondingly reordered output, leaving the computations invariant.

Proposition 2 (Permutation equivariance). HyperAdapter is permutation equivariant with respect to patch tokens.

Proof. Let $\pmb { P } _ { \pi } \in \mathbb { R } ^ { N \times N }$ denote a permutation matrix corresponding to a reordering π of the N tokens. Cosine-based routing preserves inner products under permutation, so

$$
\boldsymbol {M} (\boldsymbol {P} _ {\pi} \boldsymbol {X} _ {p}) = \boldsymbol {P} _ {\pi} \boldsymbol {M} (\boldsymbol {X} _ {p}).
$$

Hyperedge aggregation satisfies

$$
\left(\boldsymbol {P} _ {\pi} \boldsymbol {M}\right) ^ {\top} \left(\boldsymbol {P} _ {\pi} \boldsymbol {X} _ {p}\right) = \boldsymbol {M} ^ {\top} \boldsymbol {X} _ {p},
$$

ensuring that hyperedge representations remain unchanged. Since the adapter operates independently on hyperedges, ∆H is unafected. Difusion back to token space then gives

$$
\Delta \boldsymbol {X} (\boldsymbol {P} _ {\pi} \boldsymbol {X} _ {p}) = (\boldsymbol {P} _ {\pi} \boldsymbol {M}) \Delta \boldsymbol {H} = \boldsymbol {P} _ {\pi} (\boldsymbol {M} \Delta \boldsymbol {H}) = \boldsymbol {P} _ {\pi} \Delta \boldsymbol {X} (\boldsymbol {X} _ {p}),
$$

establishing permutation equivariance.

We next highlight that HyperAdapter updates lie in a low-dimensional subspace, reflecting the bottleneck structure of the adapter while preserving eficiency and structured group-level adaptation.

Proposition 3 (Low-rank adaptation structure). The token-level update induced by HyperAdapter lies in a low-dimensional subspace of rank at most min(K, r), where r is the bottleneck dimension of the adapter.

Proof. By construction, the token update is

$$
\Delta \boldsymbol {X} = \boldsymbol {M} \Delta \boldsymbol {H}, \quad \Delta \boldsymbol {H} = \boldsymbol {W} _ {\mathrm{up}} \sigma (\boldsymbol {W} _ {\mathrm{down}} \boldsymbol {H}) \in \mathbb {R} ^ {K \times D}.
$$

Since $W _ { \mathrm { d o w n } } \in \mathbb { R } ^ { r \times D }$ , the rank of $\varDelta H$ is at most r. Using the submultiplicativity of matrix rank:

$$
\operatorname{rank} (\Delta \boldsymbol {X}) = \operatorname{rank} (\boldsymbol {M} \Delta \boldsymbol {H}) \leq \min (\operatorname{rank} (\boldsymbol {M}), \operatorname{rank} (\Delta \boldsymbol {H})) \leq \min (K, r).
$$

This shows that HyperAdapter preserves the low-rank nature of parametereficient adaptation while enabling structured group-level refinement.

Structured smoothing interpretation. The token-level update can be interpreted as a structured smoothing process over hyperedges, where information is first aggregated, transformed through a low-rank adapter, and then difused back to tokens. The token-level update can be written as

$$
\varDelta \boldsymbol {X} = \boldsymbol {M} \boldsymbol {W} _ {\mathrm{up}} \sigma (\boldsymbol {W} _ {\mathrm{down}} (\boldsymbol {M} ^ {\top} \boldsymbol {X} _ {p})).\tag{7}
$$

Viewed this way, HyperAdapter performs a sequence of operations: Token <sup>Hyperedge</sup> <sup>projection</sup>−−−−−−−−−−−−−→ Hyperedge features <sup>Low-Rank</sup> <sup>adapter</sup>−−−−−−−−−−−−→ Hyperedge updates Structured difusion Token updates.

This perspective reveals HyperAdapter as a learnable structured smoothing operator in feature space: tokens assigned to the same hyperedges share updates, producing coherent, group-level feature refinements. Standard token-wise adapters correspond to the degenerate case where each hyperedge contains only a single token, reducing the smoothing operator to the identity mapping.

Distinction from existing PEFT methods. HyperAdapter departs from conventional PEFT in two key ways. First, unlike standard adapters or LoRA

Table 1: VTAB-1K performance comparison of PEFT methods using a ViT-B/16 backbone. HyperAdapter achieves the highest average accuracy (77.6%), demonstrating consistent improvements across Natural, Specialized, and Structured tasks. # Param(M) denotes the number of trainable parameters.

<table><tr><td rowspan="2"></td><td rowspan="2"># Param (M)</td><td colspan="7">Natural</td><td colspan="4">Specialized</td><td colspan="8">Structured</td><td rowspan="2">Average</td></tr><tr><td>Cifar100</td><td>Caltech101</td><td>DTD</td><td>Flower102</td><td>Pets</td><td>SVHN</td><td>Sun397</td><td>Camelyon</td><td>EuroSAT</td><td>Resisc45</td><td>Retinopathy</td><td>Clevr-Count</td><td>Clevr-Dist</td><td>DMLab</td><td>KITTL-Dist</td><td>dSpr-Loc</td><td>dSpr-Ori</td><td>sNORB-Azim</td><td>sNORB-Ele</td></tr><tr><td colspan="22">Traditional Finetuning</td></tr><tr><td>Full fine-tuning</td><td>85.8</td><td>68.9</td><td>87.7</td><td>64.3</td><td>97.2</td><td>86.9</td><td>87.4</td><td>38.8</td><td>79.7</td><td>95.7</td><td>84.2</td><td>73.9</td><td>56.3</td><td>58.6</td><td>41.7</td><td>65.5</td><td>57.5</td><td>46.7</td><td>25.7</td><td>29.1</td><td>68.9</td></tr><tr><td>Linear probing</td><td>0</td><td>64.4</td><td>85.0</td><td>63.2</td><td>97.0</td><td>86.3</td><td>36.6</td><td>51.0</td><td>78.5</td><td>87.5</td><td>68.5</td><td>74.0</td><td>34.3</td><td>30.6</td><td>33.2</td><td>55.4</td><td>12.5</td><td>20.0</td><td>9.6</td><td>19.2</td><td>57.6</td></tr><tr><td colspan="22">PEFT methods</td></tr><tr><td>BitFit [50]</td><td>0.10</td><td>72.8</td><td>87.0</td><td>59.2</td><td>97.5</td><td>85.3</td><td>59.9</td><td>51.4</td><td>78.7</td><td>91.6</td><td>72.9</td><td>69.8</td><td>61.5</td><td>55.6</td><td>32.4</td><td>55.9</td><td>66.6</td><td>40.0</td><td>15.7</td><td>25.1</td><td>65.2</td></tr><tr><td>VPT-Shallow [21]</td><td>0.06</td><td>77.7</td><td>86.9</td><td>62.6</td><td>97.5</td><td>87.3</td><td>74.5</td><td>51.2</td><td>78.2</td><td>92.0</td><td>75.6</td><td>72.9</td><td>50.5</td><td>58.6</td><td>40.5</td><td>67.1</td><td>68.7</td><td>36.1</td><td>20.2</td><td>34.1</td><td>67.8</td></tr><tr><td>VPT-Deep [21]</td><td>0.53</td><td>78.8</td><td>90.8</td><td>65.8</td><td>98.0</td><td>88.3</td><td>78.1</td><td>49.6</td><td>81.8</td><td>96.1</td><td>83.4</td><td>68.4</td><td>68.5</td><td>60.0</td><td>46.5</td><td>72.8</td><td>73.6</td><td>47.9</td><td>32.9</td><td>37.8</td><td>72.0</td></tr><tr><td> $E^2VPT$  [12]</td><td>0.25</td><td>78.6</td><td>89.4</td><td>67.8</td><td>98.2</td><td>88.5</td><td>85.3</td><td>52.3</td><td>82.5</td><td>96.8</td><td>84.8</td><td>73.6</td><td>71.7</td><td>61.2</td><td>47.9</td><td>75.8</td><td>80.8</td><td>48.1</td><td>31.7</td><td>41.9</td><td>73.9</td></tr><tr><td>Adapter [17]</td><td>0.16</td><td>69.2</td><td>90.1</td><td>68.0</td><td>98.8</td><td>89.9</td><td>82.8</td><td>54.3</td><td>84.0</td><td>94.9</td><td>81.9</td><td>75.5</td><td>80.9</td><td>65.3</td><td>48.6</td><td>78.3</td><td>74.8</td><td>48.5</td><td>29.9</td><td>41.6</td><td>73.9</td></tr><tr><td>AdaptFormer [3]</td><td>0.16</td><td>70.8</td><td>91.2</td><td>70.5</td><td>99.1</td><td>90.9</td><td>86.6</td><td>54.8</td><td>83.0</td><td>95.8</td><td>84.4</td><td>76.3</td><td>81.9</td><td>64.3</td><td>49.3</td><td>80.3</td><td>76.3</td><td>45.7</td><td>31.7</td><td>41.1</td><td>74.7</td></tr><tr><td>Convpass [22]</td><td>0.33</td><td>72.3</td><td>91.2</td><td>72.2</td><td>99.2</td><td>90.9</td><td>91.3</td><td>54.9</td><td>84.2</td><td>96.1</td><td>85.3</td><td>75.6</td><td>82.3</td><td>67.9</td><td>51.3</td><td>80.0</td><td>85.9</td><td>53.1</td><td>36.4</td><td>44.4</td><td>74.5</td></tr><tr><td>ARC [6]</td><td>0.13</td><td>72.2</td><td>90.1</td><td>72.7</td><td>99.0</td><td>91.0</td><td>91.9</td><td>54.4</td><td>84.9</td><td>95.7</td><td>86.7</td><td>75.8</td><td>80.7</td><td>67.1</td><td>48.7</td><td>81.6</td><td>79.2</td><td>51.0</td><td>31.4</td><td>39.9</td><td>75.8</td></tr><tr><td>LoRA [18]</td><td>0.29</td><td>67.1</td><td>91.4</td><td>69.4</td><td>98.8</td><td>90.4</td><td>85.3</td><td>54.0</td><td>84.9</td><td>95.3</td><td>84.4</td><td>73.6</td><td>82.9</td><td>69.2</td><td>49.8</td><td>78.5</td><td>75.7</td><td>47.1</td><td>31.0</td><td>44.0</td><td>74.5</td></tr><tr><td>NOAH [52]</td><td>0.36</td><td>69.6</td><td>92.7</td><td>70.2</td><td>99.1</td><td>90.4</td><td>86.1</td><td>53.7</td><td>84.4</td><td>95.4</td><td>83.9</td><td>75.8</td><td>82.8</td><td>68.9</td><td>49.9</td><td>81.7</td><td>81.8</td><td>48.3</td><td>32.8</td><td>44.2</td><td>75.5</td></tr><tr><td>FacT [23]</td><td>0.07</td><td>70.6</td><td>90.6</td><td>70.8</td><td>99.1</td><td>90.7</td><td>88.6</td><td>54.1</td><td>84.8</td><td>96.2</td><td>84.5</td><td>75.7</td><td>82.6</td><td>68.2</td><td>49.8</td><td>80.7</td><td>80.8</td><td>47.4</td><td>33.2</td><td>43.0</td><td>75.6</td></tr><tr><td>SSF [26]</td><td>0.24</td><td>69.0</td><td>92.6</td><td>75.1</td><td>99.4</td><td>91.8</td><td>90.2</td><td>52.9</td><td>87.4</td><td>95.9</td><td>87.4</td><td>75.5</td><td>75.9</td><td>62.3</td><td>53.3</td><td>80.6</td><td>77.3</td><td>54.9</td><td>29.5</td><td>37.9</td><td>75.7</td></tr><tr><td>RepAdapter [31]</td><td>0.22</td><td>72.4</td><td>91.6</td><td>71.0</td><td>99.2</td><td>91.4</td><td>90.7</td><td>55.1</td><td>85.3</td><td>95.9</td><td>84.6</td><td>75.9</td><td>82.3</td><td>68.0</td><td>50.4</td><td>79.9</td><td>80.4</td><td>49.2</td><td>38.6</td><td>41.0</td><td>76.1</td></tr><tr><td>Res-Tuning [31]</td><td>0.55</td><td>75.2</td><td>92.7</td><td>71.9</td><td>99.3</td><td>91.9</td><td>86.7</td><td>58.5</td><td>86.7</td><td>95.6</td><td>85.0</td><td>74.6</td><td>80.2</td><td>63.6</td><td>50.6</td><td>80.2</td><td>85.4</td><td>55.7</td><td>31.9</td><td>42.0</td><td>76.3</td></tr><tr><td>HyperAdapter (Ours)</td><td>0.44</td><td>74.1</td><td>93.3</td><td>72.8</td><td>99.3</td><td>91.7</td><td>88.3</td><td>56.7</td><td>87.5</td><td>96.3</td><td>86.0</td><td>76.5</td><td>84.0</td><td>64.4</td><td>55.2</td><td>83.9</td><td>88.5</td><td>54.6</td><td>36.3</td><td>43.7</td><td>77.6</td></tr></table>

that update tokens independently, HyperAdapter performs group-level adaptation in a learned structured interaction space, capturing higher-order relationships among tokens. Second, unlike graph-based transformers that modify the backbone attention mechanism, HyperAdapter leaves the pretrained ViT untouched and introduces a modular hyperedge adapter. This design simultaneously achieves structured adaptation, low-rank updates, and permutation equivariance, all while maintaining minimal computational and parameter overhead.

## 4 Experiment

## 4.1 Experimental Setup

Datasets. We evaluate HyperAdapter on two visual adaptation benchmarks. VTAB-1K [51] consists of 19 classification tasks grouped into three categories: Natural, Specialized, and Structured. Natural tasks contain real-world photographs captured with standard cameras. Specialized tasks include images from domainspecific sensors such as remote sensing and medical imaging. Structured tasks primarily involve synthetically generated images that test reasoning about scene structure and exhibit significant domain shift from natural images. Following the VTAB-1K protocol, each task provides 800 training samples and 200 validation samples, while the test sets follow the sizes of the original datasets. To evaluate performance in low-data regimes, we conduct experiments on five Few-shot fine-grained visual classification (FGVC ) datasets: FGVC-Aircraft [34], Oxford Pets, Food-101 [2], Stanford Cars [25], and Oxford Flowers102 [37]. We report results under 1-, 2-, 4-, 8-, and 16-shot settings.

![](images/8eb922456e173a2c88c9ddaa6adcbcf741678325fdb5438bb035e90d8f41be35.jpg)  
Fig. 2: Top-1 accuracy on few-shot FGVC benchmarks using ViT-B/16. HyperAdapter consistently outperforms prior PEFT methods, with the largest gains observed in the ultra-low-shot (1-4) regime.

Implementation details. We build upon a pretrained ViT-B/16 [7] backbone initialized with ImageNet-21k [41] weights. Unless otherwise specified, the backbone parameters are frozen, and only the classification head and Hyper-Adapter modules are trained. Each HyperAdapter operates in hyperedge space with bottleneck rank $r = 8$ and K = 8 hyperedges. The routing temperature τ controls the softness of token-to-hyperedge assignments and is selected based on validation performance. Unless otherwise stated, K is fixed across datasets, while τ is tuned. We use AdamW with weight decay $1 \times 1 0 ^ { - 4 }$ . The learning rate is $1 \times 1 0 ^ { - 3 }$ for VTAB-1K and $5 \times 1 0 ^ { - 3 }$ for FGVC tasks.. We use a cosine learning rate schedule with 10 warmup epochs and train for 100 epochs. The batch size is 64 and the input resolution is 224 × 224. All experiments are conducted on a single GPU. Following prior work [21–23], hyperparameters are tuned on the validation split, and results are reported as the mean over three runs.

Below, we present key evaluations, with further results in the Appendix.

## 4.2 Comparison to State-of-the-Art Methods

Results on VTAB-1K. Table 1 summarizes performance across the Natural, Specialized, and Structured task categories. Full fine-tuning achieves 68.9% average accuracy, while linear probing lags at 57.6%, highlighting the limitations of frozen-feature baselines in low-data regimes. Most PEFT methods substantially improve over full fine-tuning, with strong baselines such as Res-Tuning (76.3%), RepAdapter (76.1%), SSF (75.7%), and LoRA (74.5%) demonstrating the efectiveness of lightweight adaptation. HyperAdapter consistently outperforms all prior PEFT methods, reaching 77.6% average accuracy. The improvements are especially pronounced on Structured tasks, $e . g .$ , KITTI-Dist, dSpr-Loc, and sNORB, reflecting HyperAdapter’s ability to capture higher-order spatial and relational dependencies among tokens. Performance gains are also maintained across Natural and Specialized categories, indicating robust generalization across diverse visual domains. These results illustrate that Hyper-Adapter’s structured, hyperedge-based adaptation provides a more expressive and eficient mechanism than traditional token-wise PEFT, enabling consistent gains across tasks with varying domain characteristics and data complexities.

Table 2: VTAB-1k results with ViT-L/16 (ImageNet-21K). HyperAdapter achieves the highest average accuracy (77.7%), with strong gains on Structured tasks, showing efective hyperedge-based token grouping and parameter-eficient adaptation.  
Table 3: VTAB-1k results with Swin-Base (ImageNet-21K). HyperAdapter achieves top average accuracy (77.6%), particularly improving Structured tasks, showing hyperedge routing’s efectiveness across transformer architectures.

<table><tr><td>Method</td><td colspan="5">Natural Specialized Structured Average Params (M)</td></tr><tr><td>Full fine-tuning</td><td>74.7</td><td>83.8</td><td>48.1</td><td>68.9</td><td>303.40</td></tr><tr><td>Linear probing</td><td>70.9</td><td>69.1</td><td>25.8</td><td>55.3</td><td>0.05</td></tr><tr><td>Adapter</td><td>68.6</td><td>73.5</td><td>29.0</td><td>57.0</td><td>2.38</td></tr><tr><td>VPT-Deep</td><td>82.5</td><td>83.9</td><td>54.1</td><td>73.5</td><td>0.49</td></tr><tr><td>ARC</td><td>82.3</td><td>85.6</td><td>57.3</td><td>75.1</td><td>0.74</td></tr><tr><td>RepAdapter</td><td>84.0</td><td>86.3</td><td>60.1</td><td>76.8</td><td>0.79</td></tr><tr><td>HyperAdapter (Ours)</td><td>83.7</td><td>85.6</td><td>63.8</td><td>77.7</td><td>1.18</td></tr></table>

<table><tr><td>Method</td><td colspan="5">Natural Specialized Structured Average Params (M)</td></tr><tr><td>Full fine-tuning</td><td>79.1</td><td>86.2</td><td>59.7</td><td>75.0</td><td>86.90</td></tr><tr><td>Linear probing</td><td>73.5</td><td>80.8</td><td>33.5</td><td>62.6</td><td>0.05</td></tr><tr><td>VPT-Shallow</td><td>79.9</td><td>82.4</td><td>37.8</td><td>66.7</td><td>0.05</td></tr><tr><td>VPT-Deep</td><td>76.8</td><td>84.5</td><td>53.4</td><td>71.6</td><td>0.22</td></tr><tr><td>ARC</td><td>79.0</td><td>86.6</td><td>59.9</td><td>75.6</td><td>0.16</td></tr><tr><td>RepAdapter</td><td>82.8</td><td>87.2</td><td>61.2</td><td>77.0</td><td>0.39</td></tr><tr><td>HyperAdapter (Ours)</td><td>83.5</td><td>86.2</td><td>63.0</td><td>77.6</td><td>0.60</td></tr></table>

Few-shot fine-grained visual recognition. Fig. 2 shows few-shot performance (1, 2, 4, 8, 16 shots) on five fine-grained benchmarks: FGVC-Aircraft, Oxford-Pets, Food-101, Stanford Cars, and Oxford-Flowers102, along with the overall average. Performance increases consistently with more labeled samples, but the improvement is most critical in the ultra-low-shot regime (1-4 shots). HyperAdapter achieves the best or near-best results across nearly all datasets and shot settings, with particularly strong gains in the 1-4 shot regime, highlighting its ability to leverage limited supervision while preserving transferable representations from the pretrained backbone.

Results on larger ViTs. Table 2 compares VTAB-1k performance using a ViT-L/16 pretrained on ImageNet-21K. HyperAdapter achieves the highest average accuracy of 77.7%, outperforming prior PEFT methods such as ARC and RepAdapter. The gains are most pronounced on Structured tasks (63.8%), highlighting the efectiveness of hyperedge-based token grouping for capturing complex spatial and relational dependencies. With only 1.18M trainable parameters (under 0.5% of the backbone), HyperAdapter improves performance across all task categories, demonstrating both expressiveness and parameter eficiency.

Results on hierarchical Swin transformers. Table 3 reports VTAB-1k performance using a Swin-Base backbone. HyperAdapter achieves the top average accuracy of 77.6%, outperforming strong PEFT baselines including ARC and RepAdapter. Structured tasks again see the largest gains (63.0%), confirming that hyperedge routing efectively models structural relationships in hierarchical architectures. Despite introducing only 0.60M trainable parameters, Hyper-Adapter consistently boosts performance across Natural, Specialized, and Structured tasks, illustrating its generality and architecture-agnostic adaptability.

## 4.3 Ablation Study

To analyze HyperAdapter’s key properties, we perform detailed ablation studies on VTAB-1k using a pretrained ViT-B/16.

HyperAdapter vs. token-wise adapter. Table 4 compares HyperAdapter to a token-wise adapter baseline that retains the same adapter capacity but removes hyperedge routing, isolating the impact of structured adaptation. HyperAdapter consistently improves performance across all VTAB categories, with gains of +0.3 on Natural, +1.7 on Specialized, and $+ 1 . 9$ on Structured tasks, raising the overall average from 76.3% to 77.6% (+1.3%).

These improvements arise from hyperedge routing mechanism rather than increased parameters, as it enables tokens to be grouped and adapted jointly. By aggregating information across hyper-

Table 4: Impact of hyperedge-level adaptation. HyperAdapter adds hypergraph routing to the adapter while keeping a similar parameter budget, yielding consistent gains across all $\mathrm { V T A B  – 1 k }$ task categories.

<table><tr><td>Method</td><td>Params (M)</td><td>Natural</td><td>Specialized</td><td>Structure</td><td>Average (%)</td><td>Gain</td></tr><tr><td>Baseline</td><td>0.29</td><td>82.0</td><td>84.9</td><td>61.9</td><td>76.3</td><td>-</td></tr><tr><td>HyperAdapter</td><td>0.44</td><td>82.3</td><td>86.6</td><td>63.8</td><td>77.6</td><td>+1.3</td></tr></table>

edges, HyperAdapter captures higher-order relationships, producing more discriminative features. The largest gains on Specialized and Structured tasks highlight its efectiveness for fine-grained and spatially structured visual patterns.

Impact of the number of hyperedges. Fig. 3a shows how varying the number of hyperedges K affects performance and parameter count. Accuracy improves from $7 7 . 2 \%$ at $K = 4$ to a peak of $7 7 . 6 \%$ at $K =$ $^ { 8 , }$ indicating that a moderate number of hyperedges efectively captures higher-order token relationships. Beyond $K = 8 ,$ performance slightly

![](images/d547926e3d1f7eab46ed9b4d3cc78aa7ef4b5f94824ad231460f38838b1c9c86.jpg)  
(a)

![](images/5a06bc62df1651f0a2f67a1b704c1d973151cd6dc019b268d2e90a42f2047327.jpg)  
(b)  
Fig. 3: (a) Efect of the number of hyperedges K on ${ \mathrm { V T A B } } { \cdot } 1 \mathbf { k } .$ , showing that a moderate $K = 8$ balances model capacity and eficiency. (b) Sensitivity to routing temperature τ on Caltech101, where performance peaks at $\tau = 0 . 1 0$ indicating optimal hyperedge assignment at moderate routing sharpness.

drops to $7 7 . 3 \%$ , suggesting that too many hyperedges may fragment token groups and reduce structured aggregation benefits. Parameter count grows roughly linearly with K, from 0.39M at $K = 4$ to 1.50M at $K = 6 4$ . Overall, $K = 8$ ofers the best trade-of, and is used as the default in all experiments.

Sensitivity to routing temperature. We analyze the efect of routing temperature $\tau ,$ which controls the sharp ness of token-to-hyperedge assignments. Smaller $\tau$ yields confident, sharp assignments, while larger τ produces softer, more uniform routing. $\mathrm { F i g }$ . 3b shows results on Caltech101. Performance peaks at moderate temperatures $( \tau = 0 . 1 0 )$ . Too small τ limits information sharing across hyperedges, while too large $\tau$ reduces hy peredge specialization. These results indicate that HyperAdapter is stable across a broad range of $\tau$ values, with optimal performance at moderate routing sharpness.

Table 5: Ablation on adapter placement; parallel injection on Attention + MLP yields highest accuracy.

<table><tr><td>Placement</td><td>Average (%)</td><td>Params (M)</td></tr><tr><td>Pre (Attn+MLP)</td><td>77.1</td><td>0.44</td></tr><tr><td>Pre (Attn Only)</td><td>76.3</td><td>0.22</td></tr><tr><td>Pre (MLP Only)</td><td>77.0</td><td>0.22</td></tr><tr><td>Post (Attn+MLP)</td><td>77.2</td><td>0.44</td></tr><tr><td>Post (Attn Only)</td><td>76.7</td><td>0.22</td></tr><tr><td>Post (MLP Only)</td><td>77.0</td><td>0.22</td></tr><tr><td>Parallel (Attn+MLP)</td><td>77.6</td><td>0.44</td></tr><tr><td>Parallel (Attn Only)</td><td>76.8</td><td>0.22</td></tr><tr><td>Parallel (MLP Only)</td><td>77.0</td><td>0.22</td></tr></table>

Adapter placement ablation. Table 5 evaluates diferent adapter injection strategies (Pre, Post, Parallel) and module locations (Attention, MLP, or both). Parallel placement consistently achieves the best performance, reaching 77.6% average accuracy, as it preserves the residual pathway and allows adapters to provide corrective updates without disrupting pretrained features. Attaching adapters to both Attention and MLP branches yields the highest gains, while single-branch configurations are slightly less efective, MLP-only performs competitively, whereas Attention-only shows lower accuracy. These results indicate that parallel placement with adapters on both branches is the most efective configuration, which we adopt in all experiments.

![](images/ec25c396e86d301b3a2c04f2a66b56e4cb29c457d5eb1cdf74e4712670fb6d30.jpg)  
(a) CIFAR-100

![](images/dd2fc0e09d6b64abb72b4e7807897f6b3b8c076c76766806ee49cbe2c3f1f7e9.jpg)  
(b) EuroSAT

![](images/9333e6489ea1414474a7b391007b5208773e1e0b310261acbb15e3a3674657e6.jpg)  
(c) KITTI

Fig. 4: Token-to-hyperedge routing entropy across transformer layers for CIFAR-100, EuroSAT, and KITTI. Higher entropy indicates more distributed assignments. Attention adapters maintain broader routing, while MLP adapters become increasingly specialized in deeper layers, reflecting progressive feature refinement.  
![](images/c76fc2a1f0f6163517b342553641abf9ded839c22531859844f02f8d45b0a762.jpg)

![](images/1bdc7a2c0b174b5a0279c0af82e2608d290b99fbb64e986d96e937c6dfb01012.jpg)

![](images/8f834c4b2a81741adfdae0af43d9773766887bdf84e46fe13e7f91e53788a542.jpg)  
(a) CIFAR-100

![](images/6e069cc4aac7b3561e1fb14c3838b91ae5fd4310efab6f2d64a8697c7d04823c.jpg)

![](images/652959245c63f540d3680ec9996925dc2b545ac8a5e9f6d8c723751533cba9a6.jpg)

![](images/d3d47bd62ebb4722c670478654aad5f2547bbb08579def74a1ff215b2bf8d75c.jpg)

![](images/1c4225d40170036eab55555eee7aefa9f5bf5ba88f33da45421f25f31fc90f01.jpg)  
(b) EuroSAT

![](images/1c04026d5d82dd448a7102fa67ded447ac636a1f5d49519cc789fde70e7d6f04.jpg)

![](images/c94ff111c71b6699541bc56959bfe4aafaa137c991fdeaf7ee3bd90785708e8b.jpg)

![](images/2859c33d89fdfa1eee0cd2927527e020dff6b8ec65c274acf7cc85e0dffa52ff.jpg)

![](images/35f59acddeb9c50e0ead32ea924346825ef990ea8499d80db201078dc129edc2.jpg)  
(c) KITTI

![](images/78882479edcdac3b5264865743afea6dc7b03427294a7c0533c43bb348913804.jpg)  
Fig. 5: Hyperedge usage distribution across transformer layers for CIFAR-100, EuroSAT, and KITTI. Early layers use hyperedges more evenly, while deeper layers increasingly concentrate on a subset of hyperedges, indicating progressive specialization.

## 4.4 Analysis and Discussion

Routing entropy across layers. We analyze token-to-hyperedge routing entropy to understand HyperAdapter’s behavior (Fig. 4). Higher entropy reflects more distributed assignments, while lower entropy indicates confident, specialized routing. Across CIFAR-100, EuroSAT, and KITTI, attention adapters maintain relatively high entropy, promoting flexible token interactions, whereas MLP adapters show decreasing entropy in deeper layers, indicating stronger hyperedge specialization. This trend reveals a progressive routing strategy: early layers explore diverse token-group interactions, and later layers focus on structured, discriminative features, enhancing feature adaptation and representation learning.

Hyperedge usage across layers. We examine how HyperAdapter allocates capacity by tracking hyperedge usage frequency across layers (Fig. 5). For selected blocks (1, 3, 6, 12), normalized usage shows that early layers distribute tokens relatively evenly across hyperedges, supporting exploratory feature interactions. In deeper layers, a smaller subset of hyperedges dominates, reflecting specialization to capture structured semantic patterns. Together with the routing entropy analysis, this indicates a progressive adaptation strategy: initial layers explore diverse token-hyperedge interactions, while later layers consolidate and specialize representations for discriminative feature learning.

Original  
Baseline  
![](images/c03999991d000fc333f80cd7a4ebf954ee64f43391e778b25f9e57ad08cc879c.jpg)  
AdaptFormer  
Ours  
Fig. 6: DAAM [27] visualizations comparing spatial attribution across PEFT methods. Columns show the original image, token-wise baseline, AdaptFormer, and HyperAdapter. HyperAdapter produces more concentrated and semantically aligned activations, highlighting relevant object regions while reducing background noise, reflecting the benefits of hyperedge-based routing.

Qualitative visualization. We analyze HyperAdapter’s efect on spatial attention using DAAM (Fig. 6). Compared to the baseline and AdaptFormer, which produce difuse activations extending into background areas, HyperAdapter generates more focused and coherent maps that align with object structures. For example, on flowers, attention concentrates on petals, while on pets, it emphasizes the animal body over surrounding regions. These results indicate that hyperedge routing enables structured token grouping, leading to more localized and semantically consistent feature aggregation.

Further analysis and discussion are provided in the Appendix.

## 5 Conclusion

We presented HyperAdapter, a hypergraph-based parameter-eficient adaptation method for Vision Transformers. By grouping tokens into learnable hyperedges, HyperAdapter captures structured inter-token relationships with minimal additional parameters. Extensive experiments demonstrate consistent improvements over existing PEFT methods, while ablations and visualizations validate the effectiveness of hyperedge routing. Future work will explore dynamic hyperedge structures and extensions to multimodal models.

## Acknowledgments

Edwin Kwadwo Tenagyei is supported by the Grifith University International Postgraduate Research Scholarship and the Grifith University Postgraduate Research Scholarship. Lei Wang conceived the research and led the development of the methodology, while Edwin Kwadwo Tenagyei implemented the method and conducted the experiments.

This work was supported in part by the Australian Research Council (ARC) under Industrial Transformation Research Hub Grant IH180100002. This work was also supported by the National Computational Merit Allocation Scheme (NCMAS 2026), with computational resources provided by NCI Australia, an NCRIS-enabled capability supported by the Australian Government.

## References

1. Albert, P., Zhang, F.Z., Saratchandran, H., van den Hengel, A., Abbasnejad, E.: Towards higher efective rank in parameter-eficient fine-tuning using khatri-rao product. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1292–1302 (2025)

2. Bossard, L., Guillaumin, M., Gool, L.V.: Food-101 - mining discriminative components with random forests. In: European Conference on Computer Vision (2014)

3. Chen, S., Ge, C., Tong, Z., Wang, J., Song, Y., Wang, J., Luo, P.: Adaptformer: Adapting vision transformers for scalable visual recognition. Advances in Neural Information Processing Systems 35, 16664–16678 (2022)

4. Chen, T., Chen, J., Zhang, B., Yu, Z., Chen, S., Ye, R., Li, X., Ye, Y.: Sensitivityaware eficient fine-tuning via compact dynamic-rank adaptation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 9655–9664 (2025)

5. Dong, W., Sun, Y., Yang, Y., Zhang, X., Lin, Z., Yan, Q., Zhang, H., Wang, P., Yang, Y., Shen, H.: Eficient adaptation of pre-trained vision transformer via householder transformation. Advances in Neural Information Processing Systems 37, 102056–102077 (2024)

6. Dong, W., Yan, D., Lin, Z., Wang, P.: Eficient adaptation of large vision transformer via adapter re-composing. Advances in Neural Information Processing Systems 36, 52548–52567 (2023)

7. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. ArXiv abs/2010.11929 (2020)

8. Feng, Y., You, H., Zhang, Z., Ji, R., Gao, Y.: Hypergraph neural networks. In: Proceedings of the AAAI conference on artificial intelligence. vol. 33, pp. 3558– 3565 (2019)

9. Fixelle, J.: Hypergraph vision transformers: Images are more than nodes, more than edges. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9751–9761 (2025)

10. Gao, Y., Wang, M., Tao, D., Ji, R., Dai, Q.: 3-d object retrieval and recognition with hypergraph analysis. IEEE transactions on image processing 21(9), 4290–4303 (2012)

11. Hamilton, W., Ying, Z., Leskovec, J.: Inductive representation learning on large graphs. Advances in neural information processing systems 30 (2017)

12. Han, C., Wang, Q., Cui, Y., Cao, Z., Wang, W., Qi, S., Liu, D.: Eˆ 2vpt: An efective and eficient approach for visual prompt tuning. arXiv preprint arXiv:2307.13770 (2023)

13. Han, K., Wang, Y., Guo, J., Tang, Y., Wu, E.: Vision gnn: An image is worth graph of nodes. Advances in neural information processing systems 35, 8291–8303 (2022)

14. Han, Y., Wang, P., Kundu, S., Ding, Y., Wang, Z.: Vision hgnn: An image is more than a graph of nodes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19878–19888 (2023)

15. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16000–16009 (2022)

16. He, X., Li, C., Zhang, P., Yang, J., Wang, X.E.: Parameter-eficient model adaptation for vision transformers. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 37, pp. 817–825 (2023)

17. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., Gelly, S.: Parameter-eficient transfer learning for nlp. In: International conference on machine learning. pp. 2790–2799. PMLR (2019)

18. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. Iclr 1(2), 3 (2022)

19. Huang, Y., Liu, Q., Metaxas, D.: ] video object segmentation by hypergraph cut. In: 2009 IEEE conference on computer vision and pattern recognition. pp. 1738– 1745. IEEE (2009)

20. Ji, Y., Saratchandran, H., Gordon, C., Zhang, Z., Lucey, S.: Eficient learning with sine-activated low-rank matrices. arXiv preprint arXiv:2403.19243 (2024)

21. Jia, M., Tang, L., Chen, B.C., Cardie, C., Belongie, S., Hariharan, B., Lim, S.N.: Visual prompt tuning. In: European conference on computer vision. pp. 709–727. Springer (2022)

22. Jie, S., Deng, Z.H.: Convolutional bypasses are better vision transformer adapters. arXiv preprint arXiv:2207.07039 (2022)

23. Jie, S., Deng, Z.H.: Fact: Factor-tuning for lightweight adaptation on vision transformer. In: Proceedings of the AAAI conference on artificial intelligence. vol. 37, pp. 1060–1068 (2023)

24. Karimi Mahabadi, R., Henderson, J., Ruder, S.: Compacter: Eficient low-rank hypercomplex adapter layers. Advances in neural information processing systems 34, 1022–1035 (2021)

25. Krause, J., Stark, M., Deng, J., Fei-Fei, L.: 3d object representations for finegrained categorization. 2013 IEEE International Conference on Computer Vision Workshops pp. 554–561 (2013)

26. Lian, D., Zhou, D., Feng, J., Wang, X.: Scaling & shifting your features: A new baseline for eficient model tuning. Advances in Neural Information Processing Systems 35, 109–123 (2022)

27. Liao, Y., Gao, Y., Zhang, W.: Dynamic accumulated attention map for interpreting evolution of decision-making in vision transformer. Pattern Recognit. 165, 111607 (2025)

28. Liu, W., Qiu, Z., Feng, Y., Xiu, Y., Xue, Y., Yu, L., Feng, H., Liu, Z., Heo, J., Peng, S., et al.: Parameter-eficient orthogonal finetuning via butterfly factorization. arXiv preprint arXiv:2311.06243 (2023)

29. Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning, J., Cao, Y., Zhang, Z., Dong, L., et al.: Swin transformer v2: Scaling up capacity and resolution. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12009–12019 (2022)

30. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. 2021 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 9992–10002 (2021)

31. Luo, G., Huang, M., Zhou, Y., Sun, X., Jiang, G., Wang, Z., Ji, R.: Towards eficient visual adaption via structural re-parameterization. arXiv preprint arXiv:2302.08106 (2023)

32. Lv, X., Wang, L., Zhang, Q., Zheng, N., Hua, G.: Video object co-segmentation from noisy videos by a multi-level hypergraph model. In: 2018 25th IEEE International Conference on Image Processing (ICIP). pp. 2207–2211. IEEE (2018)

33. Ma, X., Chu, X., Yang, Z., Lin, Y., Gao, X., Zhao, J.: Parameter eficient quasiorthogonal fine-tuning via givens rotation. arXiv preprint arXiv:2404.04316 (2024)

34. Maji, S., Rahtu, E., Kannala, J., Blaschko, M.B., Vedaldi, A.: Fine-grained visual classification of aircraft. ArXiv abs/1306.5151 (2013)

35. Munir, M., Avery, W., Marculescu, R.: Mobilevig: Graph-based sparse attention for mobile vision applications. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2211–2219 (2023)

36. Munir, M., Avery, W., Rahman, M.M., Marculescu, R.: Greedyvig: Dynamic axial graph construction for eficient vision gnns. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6118–6127 (2024)

37. Nilsback, M.E., Zisserman, A.: A visual vocabulary for flower classification. 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06) 2, 1447–1454 (2006)

38. Parkhi, O.M., Vedaldi, A., Zisserman, A., Jawahar, C.: Cats and dogs. In: 2012 IEEE conference on computer vision and pattern recognition. pp. 3498–3505. IEEE (2012)

39. Pei, W., Xia, T., Chen, F., Li, J., Tian, J., Lu, G.: Sa<sup>2</sup>vp: Spatially aligned-andadapted visual prompt. In: Proceedings of the AAAI conference on artificial intelligence. vol. 38, pp. 4450–4458 (2024)

40. Ren, L., Chen, C., Wang, L., Hua, K.: Da-vpt: Semantic-guided visual prompt tuning for vision transformers. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 4353–4363 (2025)

41. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M.S., Berg, A.C., Fei-Fei, L.: Imagenet large scale visual recognition challenge. International Journal of Computer Vision 115, 211 – 252 (2014)

42. Sen, P., Namata, G., Bilgic, M., Getoor, L., Galligher, B., Eliassi-Rad, T.: Collective classification in network data. AI magazine 29(3), 93–93 (2008)

43. Srinivas, S.S., Sarkar, R.K., Gangasani, S., Runkana, V.: Vision hgnn: An electronmicrograph is worth hypergraph of hypernodes. arXiv preprint arXiv:2408.11351 (2024)

44. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., Jégou, H.: Training data-eficient image transformers & distillation through attention. In: International conference on machine learning. pp. 10347–10357. PMLR (2021)

45. Wale, N., Watson, I.A., Karypis, G.: Comparison of descriptor spaces for chemical compound retrieval and classification. Knowledge and Information Systems 14(3), 347–375 (2008)

46. Wang, H., Chang, J., Zhai, Y., Luo, X., Sun, J., Lin, Z., Tian, Q.: Lion: Implicit vision prompt tuning. In: Proceedings of the AAAI conference on artificial intelligence. vol. 38, pp. 5372–5380 (2024)

47. Wu, F., Hu, J., Min, G., Wang, S.: Eficient orthogonal fine-tuning with principal subspace adaptation. arXiv preprint arXiv:2505.11235 (2025)

48. Yoo, S., Kim, E., Jung, D., Lee, J., Yoon, S.: Improving visual prompt tuning for self-supervised vision transformers. In: International Conference on Machine Learning. pp. 40075–40092. PMLR (2023)

49. Yu, B.X., Chang, J., Liu, L., Tian, Q., Chen, C.W.: Towards a unified view on visual parameter-eficient transfer learning. arXiv preprint arXiv:2210.00788 (2022)

50. Zaken, E.B., Goldberg, Y., Ravfogel, S.: Bitfit: Simple parameter-eficient finetuning for transformer-based masked language-models. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). pp. 1–9 (2022)

51. Zhai, X., Puigcerver, J., Kolesnikov, A., Ruyssen, P., Riquelme, C., Lucic, M., Djolonga, J., Pinto, A.S., Neumann, M., Dosovitskiy, A., et al.: A large-scale study of representation learning with the visual task adaptation benchmark. arXiv preprint arXiv:1910.04867 (2019)

52. Zhang, Y., Zhou, K., Liu, Z.: Neural prompt search. IEEE Transactions on Pattern Analysis and Machine Intelligence 47(7), 5268–5280 (2024)

53. Zhang, Z., Zhang, Q., Gao, Z., Zhang, R., Shutova, E., Zhou, S., Zhang, S.: Gradient-based parameter selection for eficient fine-tuning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 28566– 28577 (2024)

Table 6: VTAB-1k datasets [51] categorized into Natural, Specialized, and Structured groups. Training set sizes are 800 or 1,000 depending on availability.

<table><tr><td>Category</td><td>Dataset</td><td># Classes</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td rowspan="7">Natural</td><td>CIFAR100</td><td>100</td><td></td><td></td><td>10,000</td></tr><tr><td>Caltech101</td><td>102</td><td></td><td></td><td>6,084</td></tr><tr><td>DTD</td><td>47</td><td></td><td></td><td>1,880</td></tr><tr><td>Oxford-Flowers102</td><td>102</td><td>800/1,000</td><td>200</td><td>6,149</td></tr><tr><td>Oxford-Pets</td><td>37</td><td></td><td></td><td>3,669</td></tr><tr><td>SVHN</td><td>10</td><td></td><td></td><td>26,032</td></tr><tr><td>Sun397</td><td>397</td><td></td><td></td><td>21,750</td></tr><tr><td rowspan="4">Specialized</td><td>Patch Camelyon</td><td>2</td><td></td><td></td><td>32,768</td></tr><tr><td>EuroSAT</td><td>10</td><td></td><td></td><td>5,400</td></tr><tr><td>Resisc45</td><td>45</td><td>800/1,000</td><td>200</td><td>6,300</td></tr><tr><td>Retinopathy</td><td>5</td><td></td><td></td><td>42,670</td></tr><tr><td rowspan="8">Structured</td><td>Clevr/count</td><td>8</td><td></td><td></td><td>15,000</td></tr><tr><td>Clevr/distance</td><td>6</td><td></td><td></td><td>15,000</td></tr><tr><td>DMLab</td><td>6</td><td></td><td></td><td>22,735</td></tr><tr><td>KITTI-Dist</td><td>4</td><td>800/1,000</td><td>200</td><td>711</td></tr><tr><td>dSprites/location</td><td>16</td><td></td><td></td><td>73,728</td></tr><tr><td>dSprites/orientation</td><td>16</td><td></td><td></td><td>73,728</td></tr><tr><td>SmallNORB/azimuth</td><td>18</td><td></td><td></td><td>12,150</td></tr><tr><td>SmallNORB/elevation</td><td>18</td><td></td><td></td><td>12,150</td></tr></table>

## A Dataset Statistics

We provide detailed information about the datasets used in this paper, including the number of classes and the sizes of the training, validation and test sets in Table 6 and Table 7.

The VTAB-1K datasets consists of three categories: Natural, Specialized and Structured tasks. The Natural category includes datasets such as CIFAR-100, Caltech101, DTD, Flowers102, Pets, SVHN, and Sun397. The Specialized category includes datasets such as Patch Camelyon, EuroSAT, Resisc45, and Diabetic-Retinopathy, and the Structured category includes Clevr/count, Clevr/distance, DMLab, KITTI/distance, dSprites/location, dSprites/orientation, SmallNORB/azimuth, and SmallNORB/elevation.

For the the few-shot fine-grained visual recognition, the datasets consists of FGVC-Aircraft [34], Food-101 [2], Oxford-Flowers102 [37], Oxford-Pets [38] and Stanford Cars [25].

## B Experimental Setup

In our experiments, we choose ViT-B/16 [7] trained on ImageNet-21K as our backbone. For VTAB-1K, we resize the images to $2 2 4 \times 2 2 4$ . Diferent from

Table 7: Few-shot datasets used for evaluation. Training size varies $( \mathrm { e . g . , 1 / 2 / 4 / 8 / 1 6 }$ per class), with fixed validation and test sets.

<table><tr><td>Dataset</td><td># Classes</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>Food-101</td><td>101</td><td></td><td>20,200</td><td>30,300</td></tr><tr><td>Stanford Cars</td><td>196</td><td></td><td>1,635</td><td>8,041</td></tr><tr><td>Oxford-Flowers102</td><td>102</td><td>1/2/4/8/16 per class</td><td>1,633</td><td>2,463</td></tr><tr><td>FGVC-Aircraft</td><td>100</td><td></td><td>3,333</td><td>3,333</td></tr><tr><td>Oxford-Pets</td><td>37</td><td></td><td>736</td><td>3,669</td></tr></table>

Table 8: Experiment configurations for VTAB-1K and few-shot fine-grained visual recognition experiments.

<table><tr><td>Dataset</td><td>optimizer</td><td>batch-size</td><td>learning-rate</td><td>weight-decay</td><td>epochs</td><td>lr-decay</td><td>warm-up</td></tr><tr><td>VTAB-1K</td><td>AdamW</td><td>64</td><td>1e-3</td><td>1e-4</td><td>100</td><td>cosine</td><td>10</td></tr><tr><td>Few-shot</td><td>AdamW</td><td>64</td><td>5e-3</td><td>1e-4</td><td>100</td><td>cosine</td><td>10</td></tr></table>

VTAB-1K, we use random augmentations during training, for validation/test samples, we resize them to $2 5 6 \times 2 5 6 ,$ , crop them to 224×224 at the center, and then normalize them with ImageNet’s mean and standard deviation.

For Swin Transformer, HyperAdapter is inserted in parallel to the attention and MLP modules in each transformer block across all stages, analogous to the ViT setting. We use a fixed number of hyperedges K across stages, since hyperedges represent latent semantic groups rather than spatial partitions. The routing operates directly on token features, enabling adaptive grouping despite varying token resolutions.

Table 8 shows our experiment configurations.

## C Complexity Analysis

Let N denote the number of patch tokens, D the hidden dimension, K the number of hyperedges, and r the adapter bottleneck dimension.

Parameter complexity. HyperAdapter introduces trainable hyperedge prototypes and lightweight bottleneck weights, resulting in $O ( K D + D r )$ , where $K \ll N$ and $r \ll D$ . This ensures that the additional parameter overhead is minimal compared to the frozen ViT backbone.

Computational complexity. The main operations include token-to-hyperedge routing, hyperedge aggregation, and difusion back to token space, each costing $O ( N K D )$ , while the hyperedge-level bottleneck adaptation requires $O ( K D r )$ Since self-attention in a standard ViT has complexity $O ( N ^ { 2 } D )$ , the extra computations introduced by HyperAdapter are negligible in practice, preserving the eficiency of parameter-eficient fine-tuning.

Table 9: Eficiency comparison of diferent PEFT methods. HyperAdapter introduces a small computational overhead compared to baseline, AdaptFormer, and LoRA while remaining eficient relative to the frozen backbone.

<table><tr><td>Method</td><td>Train time (ms/batch)</td><td>Inference time (ms/batch)</td><td>Memory FLOPs (GB)</td><td></td></tr><tr><td>Baseline</td><td>224</td><td>121</td><td>3.0</td><td>17.6</td></tr><tr><td>AdaptFormer</td><td>212</td><td>117</td><td>2.8</td><td>17.6</td></tr><tr><td>LoRA</td><td>218</td><td>118</td><td>2.9</td><td>17.6</td></tr><tr><td>HyperAdapter</td><td>239</td><td>129</td><td>3.2</td><td>17.8</td></tr></table>

## C.1 Training time and Inference Time

We provide quantitative comparisons of training/inference latency, peak memory, and FLOPs measured on a single NVIDIA A4000 GPU (batch size 64).

HyperAdapter introduces only modest overhead compared with token-wise PEFT methods (Table 9): training latency increases from 212-218 to 239 ms/batch, inference latency from 117-121 to 129 ms/batch, and memory from 2.8-3.0 to 3.2 GB, while FLOPs remain nearly unchanged (17.8G vs. 17.6G). The overhead mainly comes from token-hyperedge routing and aggregation. Since these operations are performed over a small latent hyperedge space $( K = 8 )$ within a low-rank bottleneck adapter, the additional complexity is $O ( N K D + K D r )$ which is negligible compared to the backbone self-attention cost $O ( N ^ { 2 } D )$

Thus, HyperAdapter preserves a favorable eficiency-performance trade-of while enabling structured group-level adaptation.

Hyperedge number K. Hyperedges in HyperAdapter represent latent semantic groups rather than fixed spatial regions, so K does not scale directly with image resolution or token count. Tokens are dynamically grouped through soft similarity-based routing, and moderate values $( \mathrm { e . g . } , \ K \ = \ 8 )$ generalize well across datasets and architectures. Even for higher-resolution inputs, Hyper-Adapter compresses token features into a fixed number of hyperedges, keeping the interaction space compact and computationally eficient.

## D More Ablations

## D.1 Efect of Bottleneck Dimension

We analyze the impact of the adapter bottleneck dimension r, which controls the capacity of the hyperedge adapter.

As shown in Table 10, performance improves when increasing r from 4 to 8, but quickly saturates thereafter. In particular, larger bottleneck dimensions $( r \geq 1 6 )$ provide no additional benefit despite introducing substantially more parameters. This suggests that the performance gains of HyperAdapter primarily arise from structured hyperedge routing rather than increased adapter capacity. Consequently, we adopt $r = 8$ for all experiments as it ofers the best trade-of between accuracy and parameter eficiency.

Table 10: Efect of bottleneck dimension r. Increasing r increases adapter capacity but also the number of trainable parameters.

<table><tr><td>r</td><td>Average (%)</td><td>Params (M)</td></tr><tr><td>4</td><td>77.2</td><td>0.29</td></tr><tr><td>8</td><td>77.6</td><td>0.44</td></tr><tr><td>16</td><td>77.2</td><td>0.74</td></tr><tr><td>32</td><td>77.2</td><td>1.32</td></tr><tr><td>64</td><td>77.0</td><td>2.51</td></tr></table>

Table 11: Efect of CLS token aggregation strategy. We compare diferent ways of propagating hyperedge updates to the CLS token.

<table><tr><td>CLS Aggregation</td><td>Average (%)</td></tr><tr><td>Zero (no CLS update)</td><td>77.1</td></tr><tr><td>MeanPatch</td><td>77.0</td></tr><tr><td>MeanH (Ours)</td><td>77.6</td></tr></table>

## D.2 Efect of CLS Token Aggregation

We analyze diferent strategies for propagating hyperedge updates to the CLS token.

As shown in Table 11, aggregating hyperedge representations (MeanH) achieves the best performance. Removing CLS updates (Zero) slightly degrades performance, while aggregating patch updates (MeanPatch) performs similarly but remains inferior to hyperedge aggregation. This suggests that global representations formed in hyperedge space provide a more informative signal for the CLS token, highlighting the importance of structured token grouping in Hyper-Adapter.

## E More Analysis

## E.1 Additional Discussions

Relation to HGNNs. While HyperAdapter is conceptually related to hypergraph learning, it difers fundamentally from HGNNs [8] in both objective and formulation. HGNNs are end-to-end backbone models for relational representation learning, whereas HyperAdapter is a lightweight PEFT module for frozen transformers that introduces structured adaptation without modifying the backbone. HyperAdapter difers from HGNNs in three aspects: (i) objective: revisiting where adaptation occurs in PEFT by shifting from token-wise to group-level adaptation; (ii) computation: replacing iterative graph propagation with lightweight routing → hyperedge adaptation → difusion inside bottleneck adapters; and (iii) design: preserving the low-rank and parameter-eficient properties of PEFT while remaining permutation equivariant (Prop. 2) and reducing to token-wise adapters as a special case (Prop. 1). Importantly, hyperedges in HyperAdapter serve only as a transient interaction space for eficient adaptation rather than as the primary feature extraction mechanism. To the best of our knowledge, this is the first work formulating PEFT as hyperedge-level structured adaptation for ViTs.

![](images/1d489d359014bfc9ac6763371dd3321ba7c17dec530f962fe56c8b26b4f83d56.jpg)  
Fig. 7: Token-to-hyperedge membership heatmaps across representative transformer layers. Each heatmap visualizes the routing matrix M, where rows correspond to patch tokens and columns correspond to hyperedges. We show routing patterns from Blocks 1, 6, and 12 for both attention and MLP adapters across CIFAR100, EuroSAT, and KITTI. Early layers exhibit difuse and distributed routing assignments, while deeper layers show increasingly structured and concentrated patterns, indicating progressive specialization of hyperedges. This behavior is consistent across datasets and modules, suggesting that HyperAdapter learns hierarchical token groupings throughout the network. For clarity, we recommend zooming in.

Table 14: Performance comparison with recent PEFT methods on VTAB-1K using a ViT-L backbone. HyperAdapter consistently outperforms prior approaches and achieves the highest average accuracy, particularly excelling on Structured tasks.

<table><tr><td>Method</td><td>Natural</td><td>Specialized</td><td>Structured</td><td>Average (%)</td></tr><tr><td>VFPT (NeurIPS 2024)</td><td>81.4</td><td>84.9</td><td>60.2</td><td>75.5</td></tr><tr><td>ViaPT (ACMMM 2024)</td><td>82.6</td><td>85.2</td><td>61.3</td><td>76.4</td></tr><tr><td>HyperAdapter</td><td>82.3</td><td>86.6</td><td>63.8</td><td>77.6</td></tr></table>

Generalization. While VTAB and few-shot FGVC are standard PEFT benchmarks for controlled comparison, HyperAdapter is designed as a general and architectureagnostic PEFT module. Our main paper (Tables 1–3) demonstrates consistent improvements across diverse backbones, including ViT-B/16, ViT-

Table 12: ADE20K semantic segmentation results. HyperAdapter achieves the highest mIoU among PEFT methods.

<table><tr><td>Method</td><td>mIoU-SS</td><td>mIoU-MS</td><td>Params.(M)</td></tr><tr><td>Full fine-tuning</td><td>48.31</td><td>50.07</td><td>318.31</td></tr><tr><td>Linear probing</td><td>35.12</td><td>37.46</td><td>13.18</td></tr><tr><td>VPT</td><td>42.11</td><td>44.06</td><td>13.43</td></tr><tr><td>RepAdapter</td><td>44.44</td><td>46.17</td><td>13.82</td></tr><tr><td>HyperAdapter</td><td>45.20</td><td>46.95</td><td>14.72</td></tr></table>

L/16, and hierarchical Swin-Base transformers. To further evaluate generalization beyond image classification, we additionally tested HyperAdapter on ADE20K semantic segmentation using a ViT-L backbone (Table 12). Hyper-Adapter outperforms strong PEFT baselines on dense prediction, showing that structured hyperedge adaptation generalizes beyond image classification to spatially dense vision tasks.

Params vs. gains. We enlarged token-wise adapters to match Hyper-Adapter’s parameter budget (Table 13). HyperAdapter still achieves higher accu racy, showing that the gains come from structured hyperedge adaptation rather than increased capacity alone.

Table 13: Performance under matched parameter budgets. Hyper-Adapter achieves the highest accuracy.

<table><tr><td>Method</td><td>Bottleneck r</td><td>Params (M)</td><td>Avg. Acc.</td></tr><tr><td>Baseline</td><td>12</td><td>0.44</td><td>76.7</td></tr><tr><td>AdaptFormer</td><td>24</td><td>0.44</td><td>76.6</td></tr><tr><td>HyperAdapter</td><td>8</td><td>0.44</td><td>77.6</td></tr></table>

Structural inductive bias. HyperAdapter does not explicitly impose spatial priors; instead, the structural bias emerges through representation similarity. In pretrained ViTs, token embeddings encode rich spatial and semantic information, allowing similarity-based routing to group semantically related regions into shared hyperedges. Importantly, the routing is dynamic and data-dependent rather than manually constrained by spatial neighborhoods. As shown in Fig. 6 of the main paper, hyperedges align with meaningful object regions despite the absence of explicit spatial supervision.

Comparison with recent baselines. We compared HyperAdapter with recent PEFT methods, including VFPT and ViaPT, under the same VTAB-1K protocol and ViT-L backbone setting (Table 14). HyperAdapter achieves the best overall performance, with strong gains on Structured tasks, further supporting the efectiveness of structured hyperedge adaptation.

## E.2 Membership Heatmaps Across Layers

Fig. 7 presents token-to-hyperedge membership heatmaps across all transformer layers for CIFAR100, EuroSAT, and KITTI. Each heatmap corresponds to the routing matrix M , where rows represent patch tokens and columns correspond to hyperedges.

Several consistent patterns can be observed. In early transformer layers (Block 1), routing assignments are relatively difuse, with tokens distributed across multiple hyperedges. This suggests that low-level representations remain shared and broadly distributed. As the network depth increases, routing patterns become progressively more structured and concentrated, with tokens exhibiting stronger preferences for specific hyperedges. This indicates that hyperedges become increasingly specialized in capturing semantic token groups. This trend is visible across both attention and MLP adapters, demonstrating that hyperedge-level adaptation operates consistently across transformer submodules. Moreover, the same hierarchical behavior appears across diferent datasets, suggesting that the routing mechanism learns dataset-agnostic grouping structures.

These visualizations provide further evidence that HyperAdapter progressively organizes tokens into semantically coherent groups as features propagate through the network.

## E.3 Patch-Grid Routing Visualizations

Fig. 8 visualizes the spatial routing behavior of HyperAdapter by coloring each patch according to the hyperedge receiving the highest routing probability. We show representative routing patterns across Blocks 1, 6, and 12 for multiple datasets and both attention and MLP adapters.

Several consistent patterns emerge from these visualizations. In early transformer layers (e.g., Block 1), routing assignments appear fragmented, with patches distributed across multiple hyperedges. This behavior suggests that low-level visual features remain broadly shared and are not yet organized into distinct semantic groups. As depth increases (Block 6), routing patterns become progressively more structured, with neighboring patches frequently assigned to the same hyperedge. This indicates that the model begins to capture spatially coherent regions corresponding to object parts or contextual background structures. In deeper layers (Block 12), the routing becomes more stable and semantically meaningful. Larger contiguous regions of the image are often assigned to the same hyperedge, suggesting that hyperedges specialize in aggregating semantically related tokens. Importantly, this behavior is observed across both attention and MLP adapters and remains consistent across datasets, indicating that the hyperedge routing mechanism learns dataset-agnostic grouping structures.

![](images/026c398f177e4432635fd76e34e78be9b044b76cbdef5a036ad659b80da67a52.jpg)  
Fig. 8: Patch-grid routing visualization. Each patch is colored according to the hyperedge receiving the highest routing probability. We show representative routing patterns from Blocks 1, 6, and 12 across datasets and modules. Early layers exhibit fragmented assignments, while deeper layers form more coherent spatial groups, indicating that HyperAdapter organizes tokens into semantically meaningful hyperedge clusters. For clarity, we recommend zooming in.

These visualizations provide qualitative evidence that HyperAdapter performs structured token grouping rather than independent token-wise adaptation, progressively organizing patch tokens into coherent hyperedge clusters as features propagate through the transformer.

## F More Visualizations

## F.1 Comparisons with Baseline and AdaptFormer

Fig. 9 presents DAAM activation maps comparing the baseline, AdaptFormer, and our proposed HyperAdapter. Each visualization highlights the spatial regions that contribute most strongly to the model’s prediction.

Several qualitative patterns emerge across examples. First, the baseline often produces relatively difuse activations that spread across both object and background regions. This indicates that token updates are largely independent and may capture less structured spatial relationships. AdaptFormer improves localization in some cases, but the resulting attribution maps remain partially scattered, with activations occasionally extending into irrelevant background areas. In contrast, HyperAdapter consistently produces more concentrated and semantically aligned activations. The highlighted regions closely correspond to the primary object structures, such as the flower petals, the dog’s face, the teapot body, the street object, and the aircraft. This improved spatial focus suggests that hyperedge-based routing encourages tokens belonging to related semantic regions to be grouped and updated collectively. As a result, the model is able to emphasize coherent object parts while suppressing background noise. Across diverse image categories, the visualizations indicate that HyperAdapter leads to more structured and interpretable feature representations compared to conventional token-wise adaptation.

## F.2 Layer-wise DAAM Visualizations

Fig. 10 presents DAAM visualizations across all transformer blocks, comparing the baseline, AdaptFormer, and the proposed HyperAdapter. The figure illustrates how spatial attribution evolves throughout the network depth.

Across early transformer blocks (Blocks 1-4), all methods exhibit relatively difuse activations distributed across broad image regions. This behavior reflects the low-level feature extraction stage, where representations remain largely shared across tokens. As depth increases, attribution maps become progressively more localized. In the baseline and AdaptFormer, however, activations often remain fragmented or spread across background areas even in deeper layers. In contrast, HyperAdapter produces increasingly concentrated and semantically aligned activations as features propagate through the network. In later blocks (Blocks 9-12), the attribution maps strongly focus on the primary object regions, such as the flower petals, the dog’s face, the teapot body, the street object, and the aircraft fuselage. Background activations are noticeably reduced compared to the other methods. This progressive sharpening of spatial attribution suggests that hyperedge-based routing encourages tokens belonging to related semantic regions to be grouped and updated collectively. As a result, HyperAdapter forms more coherent feature representations across layers, enabling clearer object localization and more interpretable model behavior.

Original  
Baseline  
AdaptFormer  
Ours  
![](images/b548e63effee39c53517cd2e2bec9dae3a75aa03eccb9e7964406cae1138662f.jpg)  
Fig. 9: DAAM [27] visualizations comparing spatial attribution across PEFT methods. Columns show the original image, token-wise baseline, AdaptFormer, and Hyper-Adapter. HyperAdapter produces more concentrated and semantically aligned activations, highlighting relevant object regions while reducing background noise, reflecting the benefits of hyperedge-based routing.

Block 1 Block 2 Block 3 Block 4 Block 5 Block 6 Block 7 Block 8 Block 9 Block 10Block 11Block 12  
![](images/2a432d4d22e051c07cddccd49448b84d43f380330e11284daa6afb217d1f2b0e.jpg)  
Fig. 10: DAAM [27] visualizations comparing HyperAdapter model with baseline and AdaptFormer models on VTAB-1K across all 12 transformer blocks.

These observations provide qualitative evidence that hyperedge-level adaptation improves the spatial organization of token representations throughout the transformer hierarchy.

## G Limitations and Future Work

Despite its efectiveness, several limitations of HyperAdapter remain. First, the proposed hyperedge routing mechanism introduces additional computation compared to conventional token-wise adapters, as it requires token-to-hyperedge assignment, aggregation, and propagation operations. Although the overhead is modest in practice, further optimization of the routing mechanism could improve scalability for larger backbones and higher-resolution inputs.

Second, the current framework uses a fixed number of hyperedges K across all layers and datasets. While our experiments show that moderate values of K perform well, adaptive or data-dependent hyperedge construction could potentially capture richer token relationships and improve flexibility across tasks.

Finally, our experiments focus primarily on vision classification benchmarks such as VTAB-1K and few-shot fine-grained visual recognition datasets. Extending HyperAdapter to other tasks, including dense prediction, video understanding, and multimodal learning, remains an interesting direction for future work.