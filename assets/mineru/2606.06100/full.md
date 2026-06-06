# HYPERVIS: CONTINUOUS LATENT VISUAL RELATIONALGRAPHS ON THE LORENTZ HYPERBOLOIDFOR COMPOSITIONAL REASONING ∗

Moshiur Farazi1† Sameera Ramasinghe2 Mahbub Ahmed Turza3 Shafin Rahman3

1Data Science and AI, University of Doha for Science and Technology, Qatar 2Pluralis Research, Australia

3Department of Electrical and Computer Engineering, North South University, Bangladesh

# ABSTRACT

Vision-Language Models (VLMs) struggle with compositional reasoning that requires understanding inter-object relationships. A natural remedy is to inject explicit scene graph triplets ⟨s, p, o⟩ from an off-the-shelf scene graph generator (SGG), but we show this backfires: discrete text labels collide with the continuous visual modality, degrading GQA accuracy from 60.38% to 58.86%. We propose HyperVis, which bypasses the SGG semantic bottleneck entirely. From N class-agnostic region proposals, we compute a dense O(N 2) visual relation tensor via spatially-biased cross-attention, project it onto a Lorentz hyperboloid, and enforce hierarchy through spatial physics, namely IoAdriven entailment cones and exterior-angle repulsion. We discover that HyperVis contributes in two complementary ways: (1) as a training-time regularizer, the hyperbolic relational losses shape LoRA representations that improve generative VQA (GQA 61.03% vs. 57.21% for LoRA fine-tuning without relational losses, recovering and surpassing the baseline); and (2) as an inference-time relational encoder, hyperbolic prefix tokens boost discriminative compositional scoring (SugarCrepe 79.94%, +6.25pp over baseline). The learned curvature stabilises at κ=4.0, an order of magnitude above prior hyperbolic VLMs where κ typically collapses toward zero, indicating that continuous visual features genuinely require the exponential volume of strongly curved space. A controlled Euclidean ablation confirms this decomposition: the relational pipeline regularises LoRA comparably in flat space (GQA 60.81%), but the compositionality gain is specifically hyperbolic (SugarCrepe +4.58pp over Euclidean), with entailment loss ∼6× higher in Euclidean training. Codes are available at TBA.

Keywords Hyperbolic Geometry · Vision-Language Models · Compositional Reasoning · Scene Graphs · Visual Question Answering

# 1 Introduction

Humans interpret visual scenes not merely by recognising individual objects, but by reasoning about the rich web of relationships that connect them. Determining whether a person is riding a horse versus standing beside it requires understanding both spatial configuration and functional interaction, cues that are critical for visual question answering [1, 2], compositional retrieval [3], and grounded captioning.

Foundation VLMs such as CLIP [4], BLIP-2 [5], and LLaVA [6, 7] achieve impressive performance across many multi-modal benchmarks. While generative VLMs with early fusion (BLIP-2, LLaVA) handle compositionality better than dual-encoder models like CLIP, they still exhibit systematic weaknesses on benchmarks that specifically probe relational and compositional understanding [8, 3, 9].

A seemingly natural fix is to attach an off-the-shelf scene graph generator (SGG) and feed its $\langle s , p , o \rangle$ triplets into the VLM as additional textual context [10, 11, 12]. We began this work along precisely this path. Our experiments revealed that injecting SGG text labels into a foundation VLM degrades, rather than improves, downstream accuracy (GQA: 58.86% vs. 60.38% baseline). We identify two practical limitations of this approach: (i) it depends on an external SGG model whose predicate vocabulary is fixed and whose errors propagate as categorically wrong tokens (e.g. on vs. next to), and (ii) discrete predicate labels discard the rich spatial–visual cues (relative pose, occlusion, pixel-level interaction) that distinguish visually similar but semantically distinct relationships.

We therefore pivot to a fundamentally different formulation. We argue that the right level of abstraction for relational reasoning in a foundation VLM is neither a global pooled feature nor a discrete predicate label, but a continuous latent visual relational graph computed directly from region features. From a small set of class-agnostic region proposals, we form a dense $\breve { O } ( N ^ { 2 } )$ tensor of visual relations via spatially-biased cross-attention (multi-head self-attention over region features with learned spatial geometry biases), with no text labels, no predicate vocabulary, and no external SGG model. We then embed this graph on the Lorentz hyperboloid and shape its geometry using purely geometric signals: if region A is spatially contained inside region B (Intersection-over-Area $\mathrm { I o \bar { A } } ( A , B ) > \tau _ { \mathrm { i n } } )$ , embedding A is forced into the entailment cone of B; if two regions do not overlap $( \mathrm { I o A } < \tau _ { \mathrm { o u t } } )$ , their embeddings are pushed apart via the exterior angle. We use hard thresholds $( \tau _ { \mathrm { i n } } { = } 0 . 8 , \tau _ { \mathrm { o u t } } { = } 0 . 0 5 )$ rather than soft weighting because the entailment cone constraint is inherently binary, a point is either inside a cone or not, and the wide dead zone $( 0 . 0 5 < \mathrm { I o A } < 0 . 8 )$ leaves ambiguous pairs unsupervised, allowing the model to place them freely on the manifold without forcing a geometric commitment. We show that this hierarchy can be predefined from spatial geometry rather than learned from semantic labels, removing dependence on predicate taxonomies entirely. A hyperbolic Top-K gate finally selects the most salient relations and injects them as visual prefix tokens into LLaVA-1.5.

We refer to this framework as HyperVis. An intriguing outcome of this design is that the learned curvature parameter stabilises at κ=4.0, an order of magnitude larger than reported in prior hyperbolic VLMs [13, 14], where κ typically collapses toward 0, effectively converging to a flat Euclidean space (the curvature bottleneck). Continuous visual features overlap heavily in pixel space, and we hypothesise, and verify experimentally, that the manifold needs the exponential volume of a strongly curved hyperbolic space to separate distinct relational objects without breaking spatial entailment cones. A key empirical finding is that HyperVis contributes in two distinct and complementary ways. First, training with hyperbolic relational losses acts as a structural regularizer for the LoRA adapters: LoRA fine-tuning on GQA without relational losses degrades GQA accuracy to 57.21% (−3.17pp below the 60.38% baseline), while training with HyperVis’s angle and entailment losses recovers to 61.03% (+0.65pp above baseline), even when prefix tokens are dropped at inference. Second, retaining the hyperbolic prefix tokens at inference time provides an explicit relational encoder for discriminative compositional scoring, boosting SugarCrepe from 73.69% to 79.94% (+6.25pp). Our contributions are:

• We show that continuous visual latents can better encode hyperbolic geometry, and empirically demonstrate that textual SGG triplet injection degrades VLM accuracy (GQA: 58.86% vs. 60.38% baseline).   
• We propose a fully visual relation tensor computed via spatially-biased cross-attention over class-agnostic regions, embedded on the Lorentz hyperboloid with a geometric IoA-driven hierarchy: spatial containment yields entailment cones, spatial separation yields angular repulsion. A hyperbolic Top-K gate selects the most salient relations as visual prefix tokens for LLaVA-1.5.   
• We demonstrate a dual contribution: hyperbolic losses regularize LoRA for generative VQA (GQA 61.03%, +3.82pp over LoRA-only), while prefix tokens boost compositional scoring (SugarCrepe 79.94%, +6.25pp over baseline). A controlled Euclidean ablation confirms that the compositionality gain is specifically hyperbolic (+4.58pp over Euclidean, entailment loss ∼6× higher in flat space), while the learned curvature κ=4.0 challenges the conventional curvature-bottleneck narrative.

# 2 Related Work

Vision-Language Models. Early VL methods learned separate representations for images and text, combining them through concatenation or bilinear pooling [15, 16]. Transformer-based models such as VL-BERT [17], ViLBERT [18], and UNITER [19] aligned object proposals with contextual word embeddings. The current paradigm centres on contrastive pre-training (CLIP [4]), generative bootstrapping (BLIP-2 [5]), and visual instruction tuning (LLaVA [7], Qwen2-VL [20], InternVL [21]). Despite strong overall performance, these models remain fragile on compositional benchmarks [8, 3].

Explicit (textual) scene graphs in VLMs. A long line of work attaches an SGG model [22, 23, 24, 25] to a VLM and feeds the resulting predicate triplets as additional context. Examples include adaptive scene graph tokens for CLIP/BLIP-2 [26], the Scene Graph Expression module in LLaVA-SG [11], chain-of-thought scene graph prompts [12], end-to-end triplet detectors [27, 28], and the structured triplet encoder of SA-VQA [10]. All of these treat predicates as a closed (or open) vocabulary and route relational signal through discrete text, inheriting the SGG model’s predicate noise.

![](images/6b1f7b568cfeefc3baf38610170aa09202b5ae8a4a3a39c3298683f4a43b7ee1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Image"] --> B["Encoder(frozen)"]
    B --> C["Input: d x d"]
    C --> D["Rot Align"]
    D --> E["CLS (+1)"]
    E --> F["V1"]
    E --> G["V2"]
    E --> H["V3"]
    E --> I["VN"]
    F --> J["DenseVisualRelations"]
    G --> J
    H --> J
    I --> J
    J --> K["Lorentz Projection"]
    K --> L["exp₀^κ"]
    L --> M["z_ij ∈ L^d_k ⊂ R^{d_hyp+1}"]
    M --> N["Hyperbolic TopKGating"]
    N --> O["Lorentz manifold"]
    O --> P["P={p1...pk} ∈ R^{k x d}"]
    P --> Q["Top-k"]
    Q --> R["Output"]
    R --> S["Output tokens"]
    
    T["Continuous Latent Hyper-Graph"] --> U["Image Tokens, E_i"]
    U --> V["LLM Input Marging"]
    
    V --> W["E_txt^before"]
    V --> X["t_USER:&quot;USER:&quot;<br>E_i image tokens"]
    V --> Y["K prefix tokens"]
    V --> Z["E_txt^after"]
    V --> AA["t_text: question + &quot;ASSISTANT:&quot;<br>LoRA"]
    
    subgraph Input
        B
        D
        E
        F
        G
        H
        I
        J
        K
        L
        M
        N
        O
        P
        Q
        R
        S
        T
        U
        V
        W
        X
        Y
        Z
        AA
        AB
        AC
        AD
        AE
        AF
        AG
        AH
        AI
        AJ
        AK
        AL
        AM
        AN
        AO
        AP
        AQ
        AR
        AS
        AT
        AU
        AV
        AW
        AX
        AY
        AZ
        BA
        BB
        BC
        BD
        BE
        BF
        BG
        BH
        BI
        BJ
        BK
        BL
        BM
        BN
        BO
        BP
        BP_0
    end
    
    style Input fill:#f9f,stroke:#333
    style VisionEncoder fill:#ccf,stroke:#333
    
    %% Legend:
    N proposals fill:#fff,stroke:#000,stroke-width:2px;
    classDef label fill:#fff,stroke:#000,stroke-width:1px;
    class A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,U,V,W,X,Y,Z,W,N,O,P,Q,R,S,T,U,U,V,W,X,Y,Z,W,N,O,P,Q,R,S,T,U,U,V,W,X,Y,Z,W,N,O,P,Q,R,S,T,U,U,V,W,X,Y,Z,W,N,O,P,Q,R,S,T,U,U,V,W,X,Y,Z,W,N,O,P,Q,R,S,T,U,U,V,W,X,Y,Z,W,N,O,P,Q,R,S,T,U,U,V,W,X,Y,Z,W,N,O,P,Q,R,S,T,U,U,V,W,X,Y,Z,W,N,O,p,q,r,s,t,u,v,w,x,y,z,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,kl,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\]
    
    %% Legend:
    subgraph Input Image
        B
        D --> E
        F --> G
        G --> H
        H --> I
        I --> J
        J --> K
        K --> L
        L --> M
        M --> N
        N --> O
        O --> P
        P --> Q
        Q --> R
        R --> S
        S --> T
        T --> U
        U --> V
        V --> W
        W --> X
        X --> Y
        Y --> Z
        Z --> AA
        AA --> AB
        AB --> AC
        AC --> AD
        AD --> AE
        AE --> AF
        AF --> AG
        AG --> AH
        AH --> AI
        AI --> AJ
        AJ --> AK
        AK --> AL
        AL --> AM
        AM --> AN
        AN --> AO
        AO --> AP
        AP --> AQ
        AQ --> AR
        AR --> AS
        AS --> AT
    end
    
    %% Legend:
    subgraph Input Image_caption<|ref_end|><|rotate_up|>
    B --> D --> F --> G --> H --> I --> J --> K --> L --> M --> N --> O --> P --> Q --> R --> S --> T --> U --> V --> W --> X --> Y --> AC & AD & AE & AG & AH & AI & AJ & AK & AL & AM & AN & AO & AP & OB & Z & AB & AC & AD & AE & AG & AH & AI & AJ & AK & AL & AM & AO & AP & OB & Z & AB & AC & AD & AE & AG & AH & AI & AJ & AK & AL & AM & AO & AP & OB & Z & AB & AC & AD & AE & AG & AH & AI & AJ & AK & AL & AM & AO & AP & OB & Z & AB & AC & AD & AE & AG & AH & AI & AJ & AK & AL & AM & AO & AP & OB & Z &AB & AC & AD & AE & AG & AH & AI & AJ & AK & AL & AM & AO & AP & OB & Z & AB & AC & AD & AE & AG & AH & AI & AJ & AK & AL & AM & AO & AP & OB & Z & AB & AC & AD & AE & AG & AH & AI & AJ & AK & AL & AM & AO & AP & OB & Z & AB & AC & AD & AE & AG & ah,AD,AD,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,Ae,AD,AD,AD,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,AE,a,e,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,kl,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p,q,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,s,t,u,v,w,x,k,l,m,n,o,p.q,r,e,s,t,u,v,w,x,k,l,m,n,o,p.q,r,e,s,t,u,v,w,x,k,l,m,n,o,p.q,r,e,s,t,u,v,w,x,k,l,m,n,o,p.q,r,e,s,t,u,v,w,x,k,l,m,n,o,p.q,r,e,s,t,u,v,w,x,k,l,m,n,o,p.q,r,e,s,t,u,v,w,x,k,l,m,n,o,p.q,r,e,s,t,u,v,w,x,k,l,m,,o,p.q,r,e,s,t,u,v,w,x,k,l,m,n,o,p.q,r,e,s,t,u,v,w,x,k,l,m,n,o,p.q,r,e,s,t,u,v,w,x,k,l,m,n,o,p.q,r,e,s,t,u,v,w,x,k,l,m,n,o,p.q,r,e,s,t,u,v,w,x,k,l,m,n,o,p.q,r,e,s,t,u,v,w,x,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,fg,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,a,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g[h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e,s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i(j,k,l),m,n,o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n,o,p.q,r,e-s,t,u,v,w.c,d,e,f,g,h,i,j,k,l,m,n:o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n:o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n:o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n:o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n:o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n o,p.q,r,e-s,t,u,v,w,c,d,e,f,g,h,i,j,k,l,m,n o,p.q,r,e-s,t,u,v,w,c,d e,f,g,h,i,j,k,l,m,n o,p.q,r,e-s,t,u,v,w,c,d e,f,g,h,i,j,k,l,m,n o,p.q,r,e-s,t,u,v,w,c,d e,f,g,h,i,j,k,l,m,n o,p.q,r,e-s,t,u,v,w,c,d e,f,g,h,i,j,k,l,m,n o,p.q,r,e-s,t,u,v,w,c,d e,f,g,h,i,j,k,l,m,n o,p .q ,r ,e ,f ,g ,H ,I ,J ,K ,L ,M ,N ,O ,P ,Q ,R ,S ,T ,U ,V ,W ,X ,Y ,Z ,A ,B ,C ,F ,G ,H ,I ,J ,K ,L ,M ,N ,O ,P ,Q ,R ,S ,T ,U ,V ,W ,X ,Y ,Z ,A ,B ,C ,F ,G ,H ,I ,J ,K ,L ,M ,N ,O ,P ,Q ,R ,S ,T ,U ,V ,
end

    subgraph Output |
    subgraph Output Moringing:
    direction TB,stroke:#ff0000,stroke-width:2px;
    end

    %% Note: The diagram shows a 2D representation of the output of the output Moringing structure with labels such as 'Node aggregation', 'Entailment loss', 'Angle loss', 'Output tokens' and 'Output Token'. The output is labeled in English. The output text is in English. The output parameters are explicitly provided in English. The output type is in English. The output parameters are explicitly provided in English. The output parameters are explicitly provided in English. The output parameters are explicitly provided in English. The output parameters are explicitly provided in English. The output parameters are explicitly provided in English. The output parameters are explicitly provided in English. The output parameters are explicitly provided in English.

The diagram illustrates the structure and key parameters used to generate a multi-scale visualization of the output space using the plot. <|caption_end|>
```
</details>

Figure 1: Overview of HyperVis. No SGG, no predicate vocabulary, the object relationships are continuous mono-modal visual tensors embedded and aggregated entirely on the Lorentz hyperboloid, shaped by physical containment.

Continuous latent graphs. A complementary strand sidesteps the SGG vocabulary entirely. Visual relationship attention [29], structured prediction over region proposals [22, 30], and Q-Former-style latent queries [5] all model relations as continuous tensors learned end-to-end. Our work is a hyperbolic instantiation of this view: we never materialise predicate labels, treating the $O ( N ^ { 2 } )$ pairwise relation tensor as the primary representation and shaping it geometrically rather than semantically.

Hyperbolic VLMs and the modality gap. Hyperbolic spaces embed hierarchical structures with lower distortion than Euclidean spaces [31, 32]. MERU [13] introduced hyperbolic contrastive VLMs in the Lorentz model with entailment losses [33, 34, 35], and HyCoCLIP [36] extended this with compositional entailment over image→box→noun hierarchies. HySAC [37] and HyperET [38] explored hyperbolic safety and efficient training. In knowledge graphs, MuRP [39] and AttH [40] compose relations as Möbius/Givens transformations [41] on the manifold. Liang et al. [42] characterised the modality gap, and Ramasinghe et al. [14] showed that geodesic contrastive losses fundamentally conflict with entailment cones, causing the curvature bottleneck where κ collapses toward zero. Our angle-based losses inherit this insight, but a key empirical finding (Sec. 4.4) is that pure visual features push κ in the opposite direction: they require strongly curved space to be properly separated.

# 3 Method

We present HyperVis in five parts: hyperbolic background (Sec. 3.1), continuous visual graph construction (Sec. 3.2), hyperbolic embedding and IoA-driven losses (Sec. 3.3), integration with the VLM backbone (Sec. 3.4), and Riemannian optimisation and numerical stability (Sec. 3.5). An overview is shown in Fig. 1.

# 3.1 Preliminaries: The Lorentz Model

We work in the Lorentz (hyperboloid) model $\mathbb { L } _ { \kappa } ^ { n }$ , an n-dimensional manifold represented as the upper sheet of a two-sheeted hyperboloid in (n + 1)-dimensional Minkowski spacetime [13, 36]:

$$
\mathbb {L} _ {\kappa} ^ {n} = \left\{\mathbf {p} \in \mathbb {R} ^ {n + 1}: \langle \mathbf {p}, \mathbf {p} \rangle_ {\mathcal {L}} = - \frac {1}{\kappa}, p _ {0} > 0 \right\} \tag {1}
$$

where $- \kappa \left( \kappa > 0 \right)$ is the curvature, $p _ { 0 }$ is the time component, $\tilde { \mathbf { p } } = ( p _ { 1 } , \ldots , p _ { n } ) ^ { \top }$ are the spatial coordinates, and the Lorentzian inner product is $\langle \mathbf { p } , \mathbf { q } \rangle _ { \mathcal { L } } = - p _ { 0 } q _ { 0 } + \langle \tilde { \mathbf { p } } , \tilde { \mathbf { q } } \rangle _ { \mathcal { E } }$ . The geodesic distance is

$$
d _ {\mathbb {L}} (\mathbf {p}, \mathbf {q}) = \frac {1}{\sqrt {\kappa}} \operatorname{arccosh} (- \kappa \langle \mathbf {p}, \mathbf {q} \rangle_ {\mathcal {L}}). \tag {2}
$$

![](images/2a91f25eb17f16febef269fc796753941cb49d7f6caaf6334fb285a17e0ff4b7.jpg)

<details>
<summary>text_image</summary>

x0(time)
xCup(cup)
Xscene(scene) Xchair(chain)
Xlamp(lamp)
Xtable(table)
x1(space) O(0.5,0,0) x2(space)
Generality (increase toward origin)
</details>

![](images/1f3b0ae0689d74a21b1c2b853e3a79a29a1830b08382a79a2b1e36ca503188fe.jpg)

<details>
<summary>text_image</summary>

room(scene)
maximally general
> π/2 (disjoint)
apple
microwave
room(scene)
Apple(food)
Chair
Microwave
table
Table
Plate
||p|| = 1
(ininfinitely specific)
</details>

Figure 2: IoA-driven hierarchical geometry in hyperbolic space. Spatial containment in image space (left) maps to radial depth and entailment-cone nesting on the Poincaré disk (center), while the same hierarchy is represented through entailment cones on the Lorentz hyperboloid (right). More general concepts lie closer to the origin and induce wider cones, whereas specific concepts are embedded farther outward with narrower nested cones.

The exponential and logarithmic maps at the origin $\mathbf { o } = ( \sqrt { 1 / \kappa } , 0 , \dots , 0 ) ^ { \top }$ are

$$
\exp_ {\mathbf {o}} ^ {\kappa} (\mathbf {v}) = \cosh \bigl (\sqrt {\kappa} \| \mathbf {v} \| _ {\mathcal {L}} \bigr) \mathbf {o} + \frac {\sinh (\sqrt {\kappa} \| \mathbf {v} \| _ {\mathcal {L}})}{\sqrt {\kappa} \| \mathbf {v} \| _ {\mathcal {L}}} \mathbf {v}, \tag {3}
$$

and $\log _ { \mathbf { o } } ^ { \kappa } ( \mathbf { q } )$ is its inverse, with $\| \mathbf { v } \| _ { \mathcal { L } } = \sqrt { \operatorname* { m a x } ( \langle \mathbf { v } , \mathbf { v } \rangle _ { \mathcal { L } } , \epsilon ) }$ for stability $( \epsilon { = } 1 0 ^ { - 6 } )$ .

Entailment cones. Inspired by Ganea et al. [33], every $\mathbf { q } \in \mathbb { L } _ { \kappa } ^ { n }$ defines a cone with half-aperture

$$
\omega (\mathbf {q}) = \arcsin \left(\frac {K}{\| \tilde {\mathbf {q}} \|}\right), \tag {4}
$$

where $K { = } 0 . 1$ . Points closer to the origin (smaller $\| \widetilde { \mathbf { q } } \| )$ have wider cones, encoding more general concepts $( \mathrm { F i g } . 2 )$ . We measure angular proximity at the origin: given two manifold points $\mathbf { p } , \mathbf { q } \ \in \mathbb { L } _ { \kappa } ^ { n }$ , let $\tilde { \mathbf { u } } _ { \mathbf { p } } ~ = ~ ( \log _ { \mathbf { o } } ^ { \kappa } \mathbf { p } ) _ { 1 : n }$ and $\tilde { \mathbf { u } } _ { \mathbf { q } } = ( \log _ { \mathbf { o } } ^ { \kappa } \mathbf { q } ) _ { 1 : n }$ denote the spatial parts of their tangent vectors at the origin. The angular separation is

$$
\phi (\mathbf {p}, \mathbf {q}) = \arccos \left(\frac {\langle \tilde {\mathbf {u}} _ {\mathbf {p}} , \tilde {\mathbf {u}} _ {\mathbf {q}} \rangle}{\| \tilde {\mathbf {u}} _ {\mathbf {p}} \| \| \tilde {\mathbf {u}} _ {\mathbf {q}} \|}\right). \tag {5}
$$

A point p is contained in q’s cone iff $\phi ( \mathbf { p } , \mathbf { q } ) \leq \omega ( \mathbf { q } )$

# 3.2 Continuous Visual Relational Graph

Bottom-up class-agnostic proposals. Given an image I, we extract N=36 region proposals. For GQA, these are bounding boxes from scene graph annotations; for benchmarks without annotations (Winoground, SugarCrepe), we use a uniform 6×6 grid of proposals. Each region i contributes a raw RoI visual feature $\mathbf { v } _ { i } \in \mathbb { R } ^ { d _ { v } }$ from the VLM’s vision encoder and a bounding box $b _ { i } = \left( x _ { i } , y _ { i } , w _ { i } , h _ { i } \right)$ . Crucially, no class labels, no predicate vocabulary, and no SGG predictions enter the pipeline.

Dense $O ( N ^ { 2 } )$ visual relations. We compute pairwise relation features in two stages. First, a multi-head self-attention layer contextualises each proposal against all others, with spatially-biased logits. The relative geometry between regions i and j is encoded as

$$
\Delta_ {i j} = \left[ \frac {x _ {i} - x _ {j}}{w _ {j}}, \frac {y _ {i} - y _ {j}}{h _ {j}}, \log \frac {w _ {i}}{w _ {j}}, \log \frac {h _ {i}}{h _ {j}} \right] \in \mathbb {R} ^ {4}, \tag {6}
$$

where we retain signed differences to encode spatial directionality (e.g. left-of vs. right-of). Note that the positional terms are antisymmetric $( \Delta _ { i j } = - \Delta _ { j i }$ for the first two components), which is the sole source of asymmetry in the relation features below; using magnitudes would collapse all directed relations to undirected ones.

A two-layer $\mathrm { M L P } f _ { \mathrm { s p a t i a l } } : \mathbb { R } ^ { 4 }  \mathbb { R } ^ { d _ { s } }$ (with GELU activation, $d _ { s } { = } 6 4 )$ maps $\Delta _ { i j }$ to a spatial feature vector. This spatial feature is used in two ways: (1) a linear head $\mathbf { W } _ { b } \in \mathbb { R } ^ { d _ { s } \times H }$ produces per-head scalar attention biases $\mathbf { b } _ { i j } \in \mathbb { R } ^ { H }$ , and (2) a second linear head $\mathbf { W } _ { s } \in \mathbb { R } ^ { d _ { s } \times d }$ produces per-pair spatial context features $\mathbf { s } _ { i j } \in \mathbb { R } ^ { d }$ .

Spatially-biased multi-head self-attention. Each proposal feature $\mathbf { v } _ { i } \in \mathbb { R } ^ { d _ { v } }$ is projected to queries, keys, and values via $\mathbf { W } _ { Q } , \mathbf { W } _ { K } , \mathbf { W } _ { V } \in \mathbb { R } ^ { d _ { v } \times d }$ :

$$
\mathbf {q} _ {i} = \mathbf {W} _ {Q} \mathbf {v} _ {i}, \quad \mathbf {k} _ {j} = \mathbf {W} _ {K} \mathbf {v} _ {j}, \quad \mathbf {v} _ {j} ^ {\prime} = \mathbf {W} _ {V} \mathbf {v} _ {j}, \tag {7}
$$

split into $H { = } 4$ heads with head dimension $d _ { h } = d / H$ . The attention logits incorporate the spatial bias additively before the softmax:

$$
A _ {i j} ^ {(h)} = \frac {\mathbf {q} _ {i} ^ {(h) ^ {\top}} \mathbf {k} _ {j} ^ {(h)}}{\sqrt {d _ {h}}} + b _ {i j} ^ {(h)}, \quad \alpha_ {i j} ^ {(h)} = \operatorname{softmax} _ {j} \left(A _ {i j} ^ {(h)}\right), \tag {8}
$$

where $b _ { i j } ^ { ( h ) }$ is the h-th component of $\mathbf { b } _ { i j } = \mathbf { W } _ { b } f _ { \mathrm { s p a t i a l } } ( \Delta _ { i j } )$ . The attended representation aggregates across all proposals and heads:

$$
\tilde {\mathbf {v}} _ {i} = \mathbf {W} _ {O} \operatorname{Concat} _ {h = 1} ^ {H} \left(\sum_ {j = 1} ^ {N} \alpha_ {i j} ^ {(h)} \mathbf {v} _ {j} ^ {\prime (h)}\right) \in \mathbb {R} ^ {d}. \tag {9}
$$

Directed relation features. For each ordered pair $( i , j )$ , we construct a relation feature by summing the attended representations of both regions and their spatial context:

$$
\mathbf {r} _ {i j} = \tilde {\mathbf {v}} _ {i} + \tilde {\mathbf {v}} _ {j} + \mathbf {s} _ {i j} \in \mathbb {R} ^ {d}, \tag {10}
$$

where $\mathbf { s } _ { i j } = \mathbf { W } _ { s } f _ { \mathrm { s p a t i a l } } ( \Delta _ { i j } )$ . Note that $\mathbf { r } _ { i j } \neq \mathbf { r } _ { j i }$ in general because $\mathbf { s } _ { i j } \neq \mathbf { s } _ { j i }$ (the spatial deltas are antisymmetric). Stacking all pairs yields the dense relation tensor $\mathbf { R } \in \mathbb { R } ^ { N \times N \times d }$ , the continuous analog of a directed scene graph adjacency.

# 3.3 Hyperbolic Embedding and IoA-Driven Losses

Projection to the Lorentz hyperboloid. A linear head $\mathbf { W } _ { p } \in \mathbb { R } ^ { d \times d }$ maps each visual relation $\mathbf { r } _ { i j }$ to a tangent vector at the origin, with its norm clamped to 5 for numerical stability at high curvature. We then project onto $\mathbb { L } _ { \kappa } ^ { n . }$

$$
\mathbf {z} _ {i j} = \exp_ {\mathbf {o}} ^ {\kappa} \big ((0, \mathbf {W} _ {p} \mathbf {r} _ {i j}) ^ {\top} \big), \tag {11}
$$

zero-padding the time coordinate so the input is a valid tangent vector. Each of the $N ^ { 2 }$ ordered pairs is thus embedded as a point $\mathbf { z } _ { i j } \in \mathbb { L } _ { \kappa } ^ { n }$ . Note that no discrete predicate labels or rotation–translation compositions are involved: the entire relational signal is encoded end-to-end by the linear projection and the exponential map.

Node aggregation. The IoA-driven losses below operate on per-node embeddings rather than per-pair embeddings. We aggregate each node i’s outgoing relations via the Einstein midpoint (Sec. 3.5):

$$
\mathbf {z} _ {i} = \text { EinsteinMidpoint } \left(\left\{\mathbf {z} _ {i j} \right\} _ {j = 1} ^ {N}\right), \tag {12}
$$

ensuring that the per-node embedding $\mathbf { z } _ { i }$ lies exactly on $\mathbb { L } _ { \kappa } ^ { n }$

Geometric (IoA-based) hierarchy. We drop semantic taxonomies entirely and let spatial geometry dictate the hierarchy. For two boxes $b _ { a } , b _ { b } .$ , define the Intersection-over-Area

$$
\mathrm{IoA} (a \rightarrow b) = \frac {\operatorname{Area} \left(b _ {a} \cap b _ {b}\right)}{\operatorname{Area} \left(b _ {a}\right)} \in [ 0, 1 ], \tag {13}
$$

which quantifies how much of a is contained within b. We use this as the hierarchical signal:

Visual entailment loss. If Io $\mathrm { A } ( a  b ) > \tau _ { \mathrm { i n } } = 0 . 8 .$ , then region a is spatially contained in $b ,$ so $\mathbf { z } _ { a }$ should fall in $b \mathbf { \bar { s } }$ entailment cone:

$$
\mathcal {L} _ {\text { ent }} = \mathbb {E} _ {(a, b): \mathrm{IoA} > \tau_ {\mathrm{in}}} \max \left(0, \phi (\mathbf {z} _ {a}, \mathbf {z} _ {b}) - \eta \omega (\mathbf {z} _ {b})\right), \tag {14}
$$

where η (initialised at 1.5, learnable) scales the cone aperture to control entailment strictness.

Visual angle (repulsion) loss. If Io $\mathrm { A } ( a \to b ) < \tau _ { \mathrm { o u t } } = 0 . 0 5$ in both directions, the regions are spatially disjoint and should be angularly separated. We push them apart via angular repulsion to prevent manifold collapse:

$$
\mathcal {L} _ {\text { ang }} = \mathbb {E} _ {(a, b): \mathrm{IoA} <   \tau_ {\mathrm{out}}} \max \big (0, m - \phi (\mathbf {z} _ {a}, \mathbf {z} _ {b}) \big), \tag {15}
$$

with margin $m = \pi / 2$ rad. Together, Eqs. 14–15 replace both the angle-based contrastive loss and the predicate taxonomy entailment loss of an SGG-based formulation, while inheriting ATMG’s [14] insight that angles, rather than geodesic distance, are the right currency for hyperbolic alignment.

Hyperbolic Top-K gating. Most of the $N ^ { 2 } { = } 1 2 9 6$ pairs are irrelevant. We learn a fixed hyperbolic query $\mathbf { q } \in \mathbb { L } _ { \kappa } ^ { n }$ (a trainable parameter projected onto the manifold) and rank relation-pair embeddings by geodesic distance $d _ { \mathbb { L } } ( \mathbf { q } , \mathbf { z } _ { i j } )$ retaining the Top-K=4 closest. The selected $\{ \mathbf { z } _ { i j _ { t } } \} _ { t = 1 } ^ { K }$ are mapped to tangent space via $\log _ { \mathbf { o } } ^ { \kappa } ,$ , projected to the LLM hidden dimension, layer-normalised, and form K visual prefix tokens.

# 3.4 Integration with VLM Backbone

The $K { = } 4$ visual prefix tokens produced above are inserted directly into LLaVA-1.5’s input sequence immediately after the visual tokens, yielding the layout $[ { \bf E } _ { \mathrm { v i s } } ; { \bf E } _ { \mathrm { r e l } } ; { \bf E } _ { \mathrm { t x t } } ]$ ]. This prefix-token injection avoids the autograd deadlocks commonly observed with hidden-state hooks under distributed training [43, 44]. The vision encoder and LLM are kept frozen; only LoRA adapters (rank 16 on $\mathsf { q \mathrm { _ { - P r o j } / v \mathrm { _ { - P r o j } } } }$ and the relational module are trained.

# 3.5 Riemannian Optimisation and Numerical Stability

Training a continuous hyperbolic graph end-to-end with a 7B-parameter LLM is non-trivial: a naive implementation diverges within hundreds of steps. We isolate several failure modes and address each with a targeted geometric or numerical fix.

The necessity of the Lorentz Model. We strictly utilize the Lorentz model over the Poincaré ball. In the high-curvature regimes necessary for continuous visual features $\left( \kappa > 4 \right)$ , the Poincaré model compresses embeddings exponentially close to its boundary, triggering catastrophic floating-point underflow. The Lorentz model, embedded in Minkowski spacetime, maintains numerical fidelity even at extreme boundaries.

Why Euclidean means corrupt the manifold. Wherever the model averages multiple manifold points, a Euclidean mean $\begin{array} { r } { \bar { \bf z } = \frac { 1 } { N } \sum _ { i } { \bf z } _ { i } } \end{array}$ produces a vector that does not lie on $\mathbb { L } _ { \kappa } ^ { n }$ in general. To see this concretely, consider two points $\mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } \in \mathbb { L } _ { \kappa } ^ { n }$ satisfying $\langle \mathbf { z } _ { k } , \mathbf { z } _ { k } \rangle _ { \mathcal { L } } = - 1 / \kappa$ . Their Euclidean midpoint $\begin{array} { r } { \bar { \bf z } = \frac { 1 } { 2 } ( { \bf z } _ { 1 } + { \bf z } _ { 2 } ) } \end{array}$ has Minkowski norm

$$
\langle \bar {\mathbf {z}}, \bar {\mathbf {z}} \rangle_ {\mathcal {L}} = \frac {1}{4} \left(\left\langle \mathbf {z} _ {1}, \mathbf {z} _ {1} \right\rangle_ {\mathcal {L}} + 2 \left\langle \mathbf {z} _ {1}, \mathbf {z} _ {2} \right\rangle_ {\mathcal {L}} + \left\langle \mathbf {z} _ {2}, \mathbf {z} _ {2} \right\rangle_ {\mathcal {L}}\right) = \frac {1}{4} \left(- \frac {2}{\kappa} + 2 \left\langle \mathbf {z} _ {1}, \mathbf {z} _ {2} \right\rangle_ {\mathcal {L}}\right). \tag {16}
$$

This equals $- 1 / \kappa$ only when $\langle \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } \rangle _ { \mathcal { L } } = - 1 / \kappa .$ , which holds if and only if ${ \bf z } _ { 1 } = { \bf z } _ { 2 }$ . For any distinct pair, z¯ $\notin \mathbb { L } _ { \kappa } ^ { n }$ . Re-projecting introduces a systematic bias toward the origin whose magnitude is $O \big ( d _ { \mathbb { L } } ( \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } ) ^ { 2 } \big )$ , compounding across aggregation steps.

We instead use the Einstein midpoint [45, 14], the geodesically correct generalisation of a weighted mean to the Lorentz model. Given points $\{ \mathbf { z } _ { i } \} _ { i = 1 } ^ { M } \in \mathbb { L } _ { \kappa } ^ { n }$ with weights $\left\{ \alpha _ { i } \right\}$ , define the Lorentz factor $\gamma _ { i } = ( \mathbf { z } _ { i } ) _ { 0 }$ . The Einstein midpoint is:

$$
\mathbf {s} = \sum_ {i = 1} ^ {M} \alpha_ {i} \gamma_ {i} \mathbf {z} _ {i}, \quad \text { EinsteinMidpoint } (\{\mathbf {z} _ {i} \}; \{\alpha_ {i} \}) = \frac {\mathbf {s}}{\sqrt {- \kappa \langle \mathbf {s} , \mathbf {s} \rangle_ {\mathcal {L}}}}. \tag {17}
$$

The denominator guarantees $\langle \cdot , \cdot \rangle _ { \mathcal { L } } = - 1 / \kappa$ exactly, so the midpoint always lies on $\mathbb { L } _ { \kappa } ^ { n }$ .

Backward-pass clamping for arccos and arccosh. Both functions have unbounded gradients at their domain boundaries. We implement custom torch.autograd.Function classes whose backward passes clamp the input before computing the gradient: $x  \mathrm { c l a m p } ( x , \breve { - } 1 + 1 0 ^ { - 2 } , 1 - 1 0 ^ { - 2 } )$ for arccos, and $x  \mathrm { c l a m p } ( x , \bar { 1 + } 1 0 ^ { - 2 } , + \infty )$ for arccosh. Without this clamping, training diverges to NaN within ∼700 steps. With it, the median hyperbolic gradient norm drops by 608×, and training converges stably for all 5 epochs.

An isolated optimiser group for κ. Placing κ in the same parameter group as the hyperbolic module (∼13M parameters) with global $\ell _ { 2 }$ clipping scales its gradient down by $\mathrm { \sim } 1 0 ^ { 4 }$ , freezing it near initialisation. We resolve this by placing κ in its own optimiser group (vanilla Adam, no clipping). The remaining hyperbolic parameters use Riemannian Adam, and the LoRA weights use AdamW; total three groups. With this change, κ stabilises at 4.0 (Sec. 4.4).

Total objective. The total objective is

$$
\mathcal {L} = \mathcal {L} _ {\text { task }} + \lambda_ {\text { ent }} \mathcal {L} _ {\text { ent }} + \lambda_ {\text { ang }} \mathcal {L} _ {\text { ang }}, \quad \lambda_ {\text { ent }} = 0. 1, \lambda_ {\text { ang }} = 1. 0, \tag {18}
$$

where $\mathcal { L } _ { \mathrm { t a s k } }$ is the standard LLaVA cross-entropy on answer tokens.

# 4 Experiments

# 4.1 Setup and Implementation Details

Datasets. GQA [2] (testdev, 12,578 questions, exact-match accuracy + per-type breakdown); Winoground [3] (400 pairs, text/image/group via sequence log-likelihoods); SugarCrepe [9] (7,514 samples, per-edit-category accuracy via log-likelihoods).

Backbone. LLaVA-1.5-7B [7]. Vision encoder frozen; LoRA on q\_proj/v\_proj (rank 16).

Table 1: GQA testdev accuracy. All methods evaluated in LoRA-only mode (no prefix tokens at inference; see Sec. 4.1). LoRA-only training without relational losses degrades the baseline by −3.17pp. Both Euclidean and hyperbolic relational pipelines recover and surpass it, with nearly identical GQA (60.81% vs. 61.03%), the geometry difference emerges on compositionality (Table 2). 

<table><tr><td>Method</td><td>Overall</td><td>Query</td><td>Verify</td><td>Choose</td><td>Logical</td><td>Compare</td></tr><tr><td>LLaVA-1.5-7B (baseline)</td><td>60.38</td><td>46.20</td><td>80.55</td><td>80.96</td><td>75.32</td><td>61.97</td></tr><tr><td>+ LoRA only (no relational loss)</td><td>57.21</td><td>42.72</td><td>80.86</td><td>80.51</td><td>67.67</td><td>57.56</td></tr><tr><td>+ textual SGG triplets</td><td>58.86</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr><tr><td>+ Eucl. visual relations (flat)</td><td>60.81</td><td>47.16</td><td>79.97</td><td>82.02</td><td>74.38</td><td>63.16</td></tr><tr><td>+ HyperVis (ours)</td><td>61.03</td><td>46.52</td><td>81.88</td><td>80.96</td><td>76.10</td><td>64.52</td></tr></table>

Hyperparameters. N =36 regions, K=4 prefix tokens, d=256 embedding dimension, κ initialised at 1.0 and learnable. IoA thresholds $\tau _ { \mathrm { i n } } { = } 0 . 8 , \tau _ { \mathrm { o u t } } { = } 0 . 0 5$ . Loss weights $\lambda _ { \mathrm { e n t } } { = } 0 . 1 , \lambda _ { \mathrm { a n g } } { = } 1 . 0$ .

Hardware and memory. All experiments are run on a single node with 8× NVIDIA H100 (80 GB) GPUs interconnected by NVLink. We use torch.distributed with bf16 mixed precision and Distributed Data Parallel. The dense $O ( N ^ { 2 } )$ visual cross-attention manifold is the dominant memory consumer and would not fit alongside the 7B LLaVA backbone in a naive setup. We therefore enable PyTorch gradient (activation) checkpointing on the LLM transformer blocks and on the cross-attention block, at the cost of one extra forward pass per backward step. The resulting per-GPU peak memory is 22.6 GB, leaving substantial headroom on each H100. Training for 5 epochs on GQA train\_balanced (∼943k questions) takes approximately 28 hours wall-clock.

Dual evaluation protocol. A key finding of our work is that HyperVis’s prefix tokens contribute differently to generative and discriminative tasks. For generative VQA (GQA), prefix tokens disrupt the autoregressive generation distribution: we observe an L2-norm mismatch of 36.5× between hyperbolic prefix embeddings $( \sim 2 4 . 6 )$ and text embeddings (∼0.68), causing attention distortion during free-form generation. We therefore evaluate GQA in LoRA-only mode, dropping prefix tokens at inference but retaining the LoRA weights shaped by hyperbolic training. For discriminative scoring tasks (Winoground, SugarCrepe), the prefix tokens are retained, since log-likelihood scoring is robust to the magnitude mismatch: it computes conditional probabilities over given captions rather than generating tokens. This dual protocol reveals two distinct contributions of the hyperbolic relational graph: training-time regularization of LoRA (improving generative VQA) and inference-time relational encoding (boosting compositional scoring). We analyse this further in Sec. 4.3.

Euclidean ablation. To isolate the contribution of hyperbolic geometry, we train a controlled variant replacing the Lorentz manifold with Euclidean space: the exponential map becomes the identity, Einstein midpoints become arithmetic means, and entailment cones reduce to Euclidean angle constraints. All other components (DenseVisualRelations, Top-K gating, prefix injection, LoRA adapters, and hyperparameters) remain identical.

Compositional eval protocol. Earlier iterations of our pipeline used generative “Yes/No” decoding for Winoground, which we found to be heavily biased (LLaVA-1.5 produces “Yes” on > 85% of prompts). We instead score each (image, caption) pair by the sequence log-likelihood of the caption given the image, masking all prompt tokens to score only the caption. We adopt the same protocol for SugarCrepe.

# 4.2 Main Results

Results on GQA: Table 1 reports the headline result. Four observations stand out:

(1) LoRA alone hurts. Fine-tuning LLaVA-1.5 with LoRA adapters on GQA data without any relational loss degrades overall accuracy from 60.38% to 57.21% (−3.17pp). The damage concentrates on Logical (−7.65pp) and Compare (−4.41pp), exactly the question types requiring multi-step relational reasoning, while Verify, which relies primarily on single-object recognition, is unaffected. This suggests the LoRA adapters overfit to surface patterns of the GQA training set, losing the general relational reasoning capabilities of the pretrained LLaVA backbone.   
(2) Textual SGG injection also hurts. Injecting discrete predicate labels from an SGG yields 58.86%, still −1.52pp below the unmodified baseline, confirming the modality-collision hypothesis.   
(3) HyperVis recovers and surpasses. Training with the same LoRA setup but adding the hyperbolic angle and entailment losses recovers overall accuracy to 61.03% (+0.65pp over baseline, +3.82pp over LoRA-only). The gain is driven by Compare (+2.55pp) and Verify (+1.33pp). Since HyperVis is evaluated without prefix tokens (LoRA-only inference), this improvement is entirely due to the relational losses shaping better LoRA representations during training.

Table 2: Left: Winoground (400 pairs, sequence log-likelihood scoring). The Euclidean variant shows higher text accuracy but substantially lower image/group scores, suggesting its prefix tokens lack the geometric precision of hyperbolic entailment cones. Right: SugarCrepe accuracy by edit category (log-likelihood scoring). HyperVis improves the overall average by +6.25 pp over baseline (79.94 vs 73.69) and +4.58 pp over the Euclidean ablation (79.94 vs 75.36), with gains across all categories. R-O/A/R: Replace Object/Attribute/Relation. S-O/A: Swap. A-O/A: Add. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Winoground</td><td colspan="8">SugarCrepe</td></tr><tr><td>Text</td><td>Image</td><td>Group</td><td>R-O</td><td>R-A</td><td>R-R</td><td>S-O</td><td>S-A</td><td>A-O</td><td>A-A</td><td>Avg</td></tr><tr><td>LLaVA-1.5-7B (baseline)</td><td>31.50</td><td>32.50</td><td>19.00</td><td>80.21</td><td>77.16</td><td>74.96</td><td>77.64</td><td>85.29</td><td>46.99</td><td>73.55</td><td>73.69</td></tr><tr><td>+ LoRA only (no relational loss)</td><td>30.50</td><td>32.50</td><td>19.50</td><td>80.45</td><td>78.68</td><td>75.89</td><td>78.05</td><td>85.89</td><td>50.00</td><td>75.87</td><td>74.98</td></tr><tr><td>+ Eucl. visual relations (with prefix)</td><td>32.25</td><td>24.25</td><td>16.25</td><td>78.87</td><td>76.02</td><td>76.53</td><td>77.64</td><td>84.83</td><td>53.98</td><td>79.62</td><td>75.36</td></tr><tr><td>+ HyperVis (ours, with prefix)</td><td>31.75</td><td>30.75</td><td>18.50</td><td>85.65</td><td>82.87</td><td>82.15</td><td>80.49</td><td>86.64</td><td>58.97</td><td>82.80</td><td>79.94</td></tr></table>

(4) Euclidean visual relations nearly match. Replacing the Lorentz manifold with Euclidean space yields 60.81%, within 0.22pp of HyperVis. Both are +3.60pp above LoRA-only, confirming that the relational pipeline regularises LoRA similarly in either geometry. The divergence emerges on compositionality (Table 2).

Results on Winoground and SugarCrepe: Table 2 presents compositional reasoning results under the log-likelihood scoring protocol. HyperVis is evaluated with prefix tokens retained at inference, as described in Sec. 4.1.

On Winoground, HyperVis with prefix tokens produces text accuracy of 31.75 (+0.25 over baseline), with image and group scores near baseline (30.75 and 18.50 respectively). The relatively modest changes on this 400-pair benchmark are consistent with its known high variance [46].

On SugarCrepe (Fig. 5), HyperVis delivers a substantial +6.25pp improvement (79.94 vs. 73.69). The per-category breakdown is informative. The largest gains appear on Replace-Relation (+7.19pp) and the two Add categories (A-O: +11.98pp; A-A: +9.25pp). Replace-Relation directly tests whether the model correctly identifies visual relationships, making it the most face-valid evidence that the continuous visual graph is encoding relational signal. The Add categories test robustness to spurious insertions; the large gains here suggest the relational graph provides a grounded sense of which objects and attributes are actually present in the image, helping reject fabricated ones. Replace-Object (+5.44pp) and Replace-Attribute (+5.71pp) also improve substantially. The Swap categories show smaller but consistent gains (S-O: +2.85pp; S-A: +1.35pp), likely because swap edits produce hard negatives that are challenging even with relational context.

Notably, LoRA-only training leaves SugarCrepe essentially unchanged (+1.29pp), confirming that the compositional improvement comes specifically from the hyperbolic prefix tokens at inference, not from LoRA fine-tuning per se.

The Euclidean ablation sharpens this picture. With an identical relational pipeline operating in flat space (arithmetic means replacing Einstein midpoints, Euclidean angle constraints replacing entailment cones), SugarCrepe improves only modestly (+1.67pp over baseline), while HyperVis gains +6.25pp. The +4.58pp Euclidean–Lorentz gap is largest on Add-Object (+4.99pp: 58.97 vs. 53.98) and Add-Attribute (+3.18pp: 82.80 vs. 79.62); both involve recognising whether entities are present in an image, a task that maps naturally onto hierarchical containment (a parent scene entails its child objects). On Winoground, the Euclidean variant shows higher text accuracy (32.25 vs. 31.75) but substantially lower image (24.25 vs. 30.75) and group (16.25 vs. 18.50), suggesting its prefix tokens carry relational signal that helps text-side matching but lacks the geometric precision needed for image-side discrimination.

Euclidean ablation: isolating the geometric contribution. To disentangle the relational pipeline from hyperbolic geometry, we compare HyperVis against the Euclidean variant described in Sec. 4.1. On GQA (Table 1), the Euclidean variant achieves 60.81%, within 0.22pp of HyperVis and +3.60pp above LoRA-only, confirming that the GQA gain is primarily a pipeline effect: structured auxiliary losses shape better LoRA representations regardless of geometry. On SugarCrepe (Table 2), the picture diverges: Euclidean scores 75.36% (+1.67pp above baseline) versus HyperVis’s 79.94% (+6.25pp). Training dynamics confirm the mechanism: the entailment loss converges to ${ \mathcal { L } } _ { \mathrm { e n t } } { = } 1 . 1 5$ in Euclidean vs. 0.18 in Lorentz (∼6× gap), while the angle loss is comparable (0.17 vs. 0.29). Entailment cones, a natively hyperbolic construct, collapse to half-space partitions in flat space, losing the geometric precision needed for fine-grained containment hierarchies.

The critical role of angle loss in curvature dynamics. When the angular repulsion loss $\mathcal { L } _ { \mathrm { a n g } }$ was inadvertently omitted from an early training run, the curvature parameter collapsed to κ=0.29, reproducing the curvature bottleneck of prior work [14]. Restoring $\mathcal { L } _ { \mathrm { a n g } }$ caused κ to stabilise at 4.0 (Fig. 3), confirming that the angular loss provides the contrastive pressure necessary to sustain high curvature. We discuss the implications of this κ value in Sec. 4.4.

![](images/fbb396d9f1090c8972d39532134b36deb411e1d4374fff7820237a38e7f6b438.jpg)

<details>
<summary>line</summary>

| Training step | L_ang = 0 (missing) | L_ang active |
| ------------- | ------------------- | ------------ |
| 0             | 1.0                 | 1.0          |
| 20000         | 0.29                | 1.5          |
| 70000         | 0.29                | 4.38         |
</details>

![](images/198223f2d1df09b178c6965de035a9f8314e620f99b203a338c232e3e98f9b7b.jpg)

<details>
<summary>line</summary>

| Training step | Lorentz (k → 4.38) | Euclidean (flat) |
| ------------- | ------------------ | ---------------- |
| 0             | ~0.2               | ~0.2             |
| 10000         | ~0.8               | ~1.2             |
| 20000         | ~0.6               | ~1.1             |
| 30000         | ~0.5               | ~1.0             |
| 40000         | ~0.4               | ~1.1             |
| 50000         | ~0.3               | ~1.0             |
| 60000         | ~0.23              | ~1.1             |
| 70000         | ~0.2               | ~1.0             |
</details>

![](images/4983ff49ad76ce4138a35e5727e5de84809ea3cee2858d169512eef4ba4f6402.jpg)

<details>
<summary>bar</summary>

(c) GQA testdev
| Model | GQA accuracy (%) |
|---|---|
| Vanilla | 60.4 |
| LoRA only | 57.2 |
| Euclidean | 60.8 |
| HyperVis | 61.0 |
</details>

Figure 3: Training dynamics. (a) Curvature κ with vs. without angle loss. (b) Entailment loss: Lorentz vs. Euclidean. (c) GQA accuracy curves.

![](images/e58b81ec8cde2c944a119c1cb3384ce40d373e545dc3634234c3af04b3853d29.jpg)

<details>
<summary>text_image</summary>

Image & Question
Retrieved (Top-K = 3)
Image & Question
Retrieved (Top-K = 3)
Q. What is sitting on the table in front of the microwave?
GT: fruit
1 bowl of fruit on table 0.86
2 fruit on plate on table 0.71
3 table in kitchen 0.39
Q. What color is the bus driving down the wet street?
GT: yellow
yellow school bus 0.89
2 wet street in rain 0.67
3 bus on the street 0.31
Q. What animal is on the couch next to the lamp?
GT: cat
1 cat on couch 0.93
2 lamp next to couch 0.61
3 couch in living room 0.28
Q. What is the player holding in the baseball game?
GT: bat
1 baseball bat 0.91
2 player batting 0.63
3 baseball game 0.33
Q. What is on the wooden table?
GT: mug
1 white mug on table 0.90
2 cup on table 0.64
3 wooden table 0.28
Q. What animal is standing in the field?
GT: zebra
1 zebra in the field 0.89
2 horse in the field 0.58
3 baseball game 0.26
</details>

Figure 4: Qualitative behaviour of the hyperbolic Top-K gate on GQA examples.

Hyperbolic prefix visualisation. We project the learned visual relation embeddings $\mathbf { z } _ { i j }$ from $\mathbb { L } _ { \kappa } ^ { n }$ onto the Poincaré disk via $\mathbf { p } \mapsto \tilde { \mathbf { p } } / ( 1 + p _ { 0 } )$ (Fig. 2). Pairs with high IoA (containment) cluster near the origin (general/contextual), while spatially disjoint pairs disperse toward the boundary, confirming that IoA-driven training successfully shapes the radial geometry as designed.

Top-K gate qualitative behavior. For relational GQA queries (e.g. “what is to the left of the cat?”), the Top-4 pairs selected by the hyperbolic gate consistently include the cat region paired with the object actually being referred to. For non-relational queries (e.g. “what colour is the umbrella?”), the gate falls back to self-pair tokens and the prefix becomes effectively benign (Fig. 4).

# 4.3 Ablation Studies

Curvature sensitivity (fixed κ). A central claim of this work is that continuous visual features require strong curvature. Table 3 tests this by fixing κ at several values versus allowing it to be learned. The results reveal a striking asymmetry: GQA accuracy is essentially κ-insensitive (all values fall within a 0.22pp range of 60.81–61.03%), confirming that the VQA regularization effect of the relational pipeline does not depend on curvature. SugarCrepe, however, varies significantly with κ: compositionality peaks at high curvature $( \kappa \mathrm { = } \mathrm { \bar { 3 } . 0 } \mathrm { : ~ } 8 0 . 0 8 \% , \kappa \mathrm { = } 1 . 0 \mathrm { : ~ } \bar { 7 } 9 . 6 3 \bar { \% } )$ and drops at low curvature $( \kappa { = } 0 . 5 ; 7 4 . 3 4 \% )$ ), consistent with the hypothesis that entailment cones need strong curvature to encode finegrained containment hierarchies. The learned κ=4.0 matches or exceeds all fixed values (SC 79.94%), demonstrating that the model finds the optimal curvature automatically without manual tuning.

![](images/6f2897f43c28d7b20fb9aa8ac0a67b42001391d914779c1a5d78d30a9347bb72.jpg)

<details>
<summary>bar_line</summary>

| Fixed κ values | GQA accuracy (%) | SugarCrepe avg (%) |
|---|---|---|
| 0.1 | ~75 | ~65 |
| 0.5 | ~70 | ~45 |
| 1.0 | ~72 | ~80 |
| 2.0 | ~68 | ~70 |
| 3.0 | ~70 | ~85 |
| Learned (κ = 4.04) | ~75 | ~85 |
GQA: κ-insensitive (60.81-61.03%) SC: peaks at high κ
Baseline avg 73.69%
</details>

Figure 5: (Left) Fixed versus learned curvature analysis, where GQA accuracy remains stable across different κ values while SugarCrepe performance improves with higher curvature. (Right) SugarCrepe per-category accuracy showing that the Euclidean–Lorentz performance gap is most pronounced in Add categories (A-O, A-A), which emphasize hierarchical containment reasoning.

<table><tr><td> $\kappa$ </td><td>GQA</td><td>SC Avg</td><td>Final  $\kappa$ </td></tr><tr><td>0.1 (fixed)</td><td>61.02</td><td>76.56</td><td>0.1</td></tr><tr><td>0.5 (fixed)</td><td>60.95</td><td>74.34</td><td>0.5</td></tr><tr><td>1.0 (fixed)</td><td>60.87</td><td>79.63</td><td>1.0</td></tr><tr><td>2.0 (fixed)</td><td>60.81</td><td>76.92</td><td>2.0</td></tr><tr><td>3.0 (fixed)</td><td>60.84</td><td>80.08</td><td>3.0</td></tr><tr><td>Learnable( $\kappa_{\text{init}}=1.0$ )</td><td>61.03</td><td>79.94</td><td>4.0</td></tr></table>

TABLE 3: Fixed vs. learned curvature κ. All fixed-κ models trained for 3 epochs; the learnable row is the full 5-epoch HyperVis.

Table 4: Left: loss component ablation; Right: comparison with CCoT [12]. All results use LLaVA-1.5-7B. We use LoRA-only evaluation for GQA and prefix evaluation for compositionality. 

<table><tr><td colspan="4">Loss Component Ablation</td><td colspan="4">CCoT Comparison</td></tr><tr><td>Configuration</td><td>GQA</td><td>SC Avg</td><td>Final κ</td><td>Method</td><td>GQA</td><td>Wino Group</td><td>SC Avg</td></tr><tr><td>Full ( $\mathcal{L}_{ang} + \mathcal{L}_{ent}$ )</td><td>61.03</td><td>79.94</td><td>4.0</td><td>LLaVA-1.5-7B baseline</td><td>60.38</td><td>19.00</td><td>73.69</td></tr><tr><td>No  $\mathcal{L}_{ang}$  (entailment only)</td><td>60.63</td><td>79.84</td><td>—</td><td>+ CCoT prompting</td><td>53.07</td><td>22.00</td><td>75.01</td></tr><tr><td>No  $\mathcal{L}_{ent}$  (angle only)</td><td>60.69</td><td>75.91</td><td>1.72</td><td>+ HyperVis (ours)</td><td>61.03</td><td>18.50</td><td>79.94</td></tr><tr><td>No hierarchy ( $\mathcal{L}_{task} + hinge only$ )</td><td>60.73</td><td>74.45</td><td>~1.0</td><td></td><td></td><td></td><td></td></tr></table>

Loss component ablation. Table 4(Left) isolates the contribution of each geometric loss by removing them individually while retaining the VQA task loss $\mathcal { L } _ { \mathrm { t a s k } }$ and κ hinge. The results reveal a clear division of labour. Entailment loss drives compositionality: the entailment-only variant $( \lambda _ { \mathrm { a n g } } { = } 0 )$ achieves SC 79.84%, within 0.10pp of the full model, because entailment cones directly encode the containment hierarchies that SugarCrepe probes. Angle loss drives curvature and GQA: the angle-only variant $( \lambda _ { \mathrm { { e n t } } } { = } 0 )$ sees κ settle at only 1.72 (vs. 4.0 for the full model), confirming that angular repulsion alone cannot sustain high curvature without entailment pressure. Its SC drops to 75.91%, barely above the no-hierarchy baseline (74.45%). Both losses are needed for the best GQA: the full model achieves 61.03% vs. 60.63–60.73% for single-loss variants, a small but consistent gap arising from richer LoRA regularization when both geometric constraints shape the loss landscape simultaneously. Notably, the no-hierarchy variant still achieves 60.73% GQA (+3.52pp above LoRA-only), confirming that the DenseVisualRelations module and prefix tokens provide structural regularization even without geometric supervision.

Comparison with CCoT prompting. CCoT [12] is the most directly comparable inference-time baseline: it augments LLaVA with a self-generated textual scene graph via two-stage prompting. We run CCoT on our GQA testdev split and compositionality benchmarks using vanilla LLaVA-1.5-7B (Table 4(Right)). CCoT degrades GQA from 60.38% to 53.07% (−7.31pp): the self-generated scene graph introduces hallucinated predicates that mislead the model on short-answer VQA. On SugarCrepe, CCoT achieves 75.01% (+1.32pp over baseline), a modest gain compared to HyperVis’s +6.25pp. CCoT does improve Winoground group score (22.0% vs. 19.0% baseline), likely because the two-stage prompt helps the model attend to compositional structure in the easier text-matching regime. Compared to HyperVis, CCoT’s textual scene graph approach is outperformed on both GQA (−7.96pp) and SugarCrepe (−4.93pp), while also requiring 2× the inference compute (two forward passes per question). HyperVis’s advantage is structural: it learns visual relations during training rather than generating textual descriptions at test time.

Directional spatial deltas. As noted in Sec. 3.2, the spatial encoding $\Delta _ { i j }$ uses signed coordinate differences to preserve directionality $( \Delta _ { i j } = - \Delta _ { j i }$ for the positional terms). To verify this design choice, we train a variant using unsigned (magnitude-only) positional deltas $\bar { | } \Delta _ { i j } |$ , collapsing all directed relations to undirected ones. This variant achieves 60.85% GQA (−0.18pp) and 78.11% SugarCrepe (−1.83pp). The Winoground image score drops more sharply (22.5% vs. 30.75%, −8.25pp), indicating that directional spatial information is particularly important for image-side compositional matching, where $\mathbf { \ddot { \theta } } _ { \mathbf { A } }$ left of $\mathbf { B } ^ { \ast }$ and “B left of $\mathbf { A } ^ { \prime \prime }$ must be distinguished.

Table 5: Dual evaluation protocol (same HyperVis checkpoint). Prefix tokens hurt generative VQA but boost discriminative compositional scoring. Bold indicates the recommended protocol for each benchmark type. 

<table><tr><td>Inference mode</td><td>GQA (generation)</td><td>SugarCrepe (scoring)</td></tr><tr><td>LoRA-only (no prefix)</td><td>61.03</td><td>—</td></tr><tr><td>With prefix tokens</td><td>34.75</td><td>79.94</td></tr></table>

![](images/48d805cdb454442375a1f4778ca5dbe813e82997b593289f05f8b17caec348ce.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Sequence"] --> B["Image tokens, E_i ∈ ℝ^m x d"]
    A --> C["prefix token, P ∈ ℝ^k x d"]
    A --> D["text token, E_T ∈ ℝ^n x d"]
    B --> E["Attention weights (row for the next token)"]
    C --> F["prefix tokens"]
    D --> G["text tokens"]
    H["next token to generate"] --> I["Magnitude mismatch"]
    I --> J["||p||_2 >> ||t||_2 >> ||E||_1"]
    K["Attention collapse → Generation failure"] --> L["GQA Accuracy f_gen(P)"]
```
</details>

![](images/65c618d8945ce3189839602b02f35ff02cc98774d0e140c353ffd0fea04847da.jpg)

<details>
<summary>bar</summary>

(b) Scoring Mode (SugarCrepe)
| Caption | Image tokens | prefix tokens(same) | Caption tokens |
| :--- | :--- | :--- | :--- |
| Log P without prefix | -2.5 | -1.0 | -1.5 |
| Log P with prefix | -1.5 | -0.5 | -2.0 |
SuperCrepe Average
f_score(P)
log P(capA | img, prefix) > log P(capB | img, prefix)
The correct caption still scores higher
</details>

Figure 6: Why prefix tokens hurt generation but help scoring. The L2-norm mismatch distorts autoregressive attention (left) but does not affect relative log-likelihood ranking (right).

The dual contribution of hyperbolic training. Our experiments reveal that HyperVis’s relational graph serves two distinct and complementary functions (Table 5). As a training-time regularizer, the hyperbolic angle and entailment losses prevent LoRA adapters from overfitting to surface-level GQA patterns, yielding +3.82pp on GQA compared to LoRA-only training, even when prefix tokens are dropped at inference. As an inference-time relational encoder, the prefix tokens provide explicit compositional structure that the log-likelihood scorer can leverage, producing +6.25pp on SugarCrepe. These two roles are architecturally inseparable during training (the same forward pass computes both the relational losses and the prefix tokens), but their effects can be cleanly separated at inference time by toggling prefix injection (Fig. 6).

The generation failure with prefix tokens (GQA 34.75%) stems from a 36.5× L2-norm mismatch between hyperbolic prefix embeddings and text embeddings, which distorts the attention distribution during autoregressive decoding. Preliminary experiments with LayerNorm and prefix dropout (p=0.3) partially close this gap (GQA recovers to 49.44%), but fully resolving the generation–scoring tradeoff remains an important direction for future work.

Top-K sensitivity. Table 6 varies the number of prefix tokens K injected into the VLM. K=0 corresponds to LoRAonly training without any relational losses. The results show two distinct regimes. For GQA (LoRA-only eval), any $K \geq 1$ recovers from the LoRA-only degradation: even K=1 achieves 60.92%, a +3.71pp jump over K=0 (57.21%), with further gains plateauing by K=4. This confirms that the presence of relational losses during training, not the number of prefix tokens at inference, drives the GQA regularization effect. For SugarCrepe, the pattern is non-monotonic: compositionality peaks at K=2 (82.34%) then declines for K ≥ 8 (72.81% at K=8, 72.22% at K=16), falling below the baseline. Too many prefix tokens introduce redundant or conflicting relational signals that degrade the log-likelihood scorer. K=4 provides the best balance (79.94%), and we adopt it as the default.

# 4.4 Discussion

Why textual SGG injection backfires. A discrete predicate label discards exactly the cues that make a visual relationship resolvable: the relative pose of the two regions, occlusion, and pixel-level interaction. When the LLM sees the token “on”, it has lost the ability to distinguish “standing on”, “leaning on”, “hovering above”, all collapsed into one symbol. Worse, SGG noise propagates symbolically: a wrong predicate is not a slightly noisy continuous feature, it is a categorically wrong word. Our continuous relation tensor preserves the spatial–visual signal end-to-end and degrades gracefully under detector noise.

Table 6: Effect of the number of prefix tokens K on GQA (LoRA-only eval) and SugarCrepe (with prefix). K=0 is LoRA-only (no relational loss). Ablation models trained for 3 epochs. 

<table><tr><td>K</td><td>GQA</td><td>SC Avg</td></tr><tr><td>0 (LoRA-only)</td><td>57.21</td><td>74.98</td></tr><tr><td>1</td><td>60.92</td><td>76.46</td></tr><tr><td>2</td><td>60.84</td><td>82.34</td></tr><tr><td>4 (ours)</td><td>61.03</td><td>79.94</td></tr><tr><td>8</td><td>61.00</td><td>72.81</td></tr><tr><td>16</td><td>60.98</td><td>72.22</td></tr></table>

Why LoRA alone degrades GQA. The −3.17pp drop from LoRA-only fine-tuning (57.21% vs. 60.38% baseline) is initially surprising: task-specific fine-tuning usually helps. We attribute this to catastrophic narrowing: LLaVA-1.5 was pretrained on diverse visual instruction data spanning many reasoning types; LoRA fine-tuning on a single dataset (GQA train\_balanced) narrows the model’s capabilities toward surface-level GQA patterns at the expense of the broad relational reasoning needed for the testdev split. The damage concentrates on Logical (−7.65pp) and Compare (−4.41pp), the question types most dependent on multi-step reasoning. HyperVis’s hyperbolic losses counteract this by forcing the LoRA weights to maintain a geometrically structured relational representation, preventing the collapse into surface-level shortcuts.

The κ=4.0 revelation. Prior hyperbolic VLMs operating on global image-text features report a curvature collapse: κ drifts toward 0, effectively reverting the manifold to Euclidean (the curvature bottleneck of [14]). We observe the opposite. With our IoA-driven losses on continuous visual features, κ stabilises at 4.0, an order of magnitude larger than prior reports. Continuous visual region features overlap heavily in pixel space (an arm and the body it is attached to share most of their RoI features), so the model needs the exponential volume growth of strongly curved hyperbolic space to push relationally distinct objects apart, while spatial entailment cones simultaneously enforce that contained regions remain inside their containers’ cones. High curvature is thus not a quirk; it is the geometric resolution of the tension between visual feature overlap and spatial containment.

Euclidean vs. hyperbolic: a controlled decomposition. The Euclidean ablation provides a clean decomposition of HyperVis’s contributions. On GQA (LoRA-only inference), both geometries perform near-identically (60.81% vs. 61.03%), confirming that the GQA gain is a pipeline effect. On SugarCrepe, hyperbolic geometry adds +4.58pp. The mechanism is the entailment cone: in Lorentz space, containment is encoded as a narrow angular cone providing geometrically precise hierarchy; in Euclidean space, this degenerates to a half-space partition, producing a ∼6× higher entailment loss at convergence. Notably, Euclidean prefix tokens damage GQA generation far less (54.06% vs. 34.75%), because there is no manifold-to-Euclidean embedding-space mismatch at injection time, corroborating the magnitude-mismatch diagnosis.

Generalising beyond static-image VQA. The HyperVis recipe is largely orthogonal to the choice of input modality. Replacing 2D bounding boxes with 3D spatio-temporal tubes extends the approach to video action graphs, while re-interpreting IoA as a contact/support predicate enables embodied affordance reasoning. Both inherit our finding that strongly curved space (κ ≫ 1) is what makes continuous visual hierarchies usable.

Limitations. (i) HyperVis depends on region proposals; very small or heavily occluded objects can be missed. (ii) The O(N 2) relation tensor scales quadratically; we use N =36 as a practical sweet spot. (iii) Resolving the generation– scoring tradeoff remains open (Sec. 4.3). (iv) The IoA prior favours spatially compact relations; abstract relations (gaze, motion) may benefit from learned auxiliary signals.

# 5 Conclusion

We presented HyperVis, a framework that bypasses the SGG semantic bottleneck by routing relational reasoning over a continuous latent visual graph on the Lorentz hyperboloid. HyperVis contributes in two ways: hyperbolic relational losses regularize LoRA adapters (GQA 61.03%, +3.82pp over LoRA-only), while hyperbolic prefix tokens boost compositional scoring (SugarCrepe 79.94%, +6.25pp over baseline). A controlled Euclidean ablation confirms that the compositionality gain is specifically hyperbolic (+4.58pp), and comprehensive ablations over curvature, prefix count, and loss components decompose each contribution. The learned curvature κ=4.0 challenges the curvature-bottleneck narrative, demonstrating that continuous visual features require the exponential volume of strongly curved hyperbolic space.

# References

[1] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017.   
[2] Drew A Hudson and Christopher D Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019.   
[3] Tristan Thrush, Ryan Jiang, Max Bartolo, et al. Winoground: Probing vision and language models for visiolinguistic compositionality. In CVPR, 2022.   
[4] Alec Radford, Jong Wook Kim, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.   
[5] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023.   
[6] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2024.   
[7] Haotian Liu, Chunyuan Li, Yuqian Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024.   
[8] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why visionlanguage models behave like bags-of-words, and what to do about it? In ICLR, 2023.   
[9] Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality. In NeurIPS, volume 36, pages 31096–31116, 2023.   
[10] Peixi Xiong et al. SA-VQA: Structured alignment of visual and semantic representations for visual question answering. arXiv:2201.10654, 2022.   
[11] Jingyi Wang, Jianzhong Ju, Jian Luan, and Zhidong Deng. LLaVA-SG: Leveraging scene graphs as visual semantic expression in vision-language models. ICASSP, 2025.   
[12] Chancharik Mitra, Brendan Huang, Trevor Darrell, and Roei Herzig. Compositional chain-of-thought prompting for large multimodal models. In CVPR, 2024.   
[13] Karan Desai, Maximilian Nickel, Tanmay Rajpurohit, Justin Johnson, and Ramakrishna Vedantam. Hyperbolic image-text representations. In ICML, 2023.   
[14] Sameera Ramasinghe, Violetta Shevchenko, Gil Avraham, and Ajanthan Thalaiyasingam. Accept the modality gap: An exploration in the hyperbolic space. In CVPR, 2024.   
[15] Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. In CVPR, 2015.   
[16] Akira Fukui, Dong Huk Park, Daylen Yang, et al. Multimodal compact bilinear pooling for visual question answering and visual grounding. In EMNLP, 2016.   
[17] Weijie Su, Xizhou Zhu, Yue Cao, et al. VL-BERT: Pre-training of generic visual-linguistic representations. In ICLR, 2020.   
[18] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. ViLBERT: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In NeurIPS, 2019.   
[19] Yen-Chun Chen, Linjie Li, Licheng Yu, et al. UNITER: Universal image-text representation learning. In ECCV, 2020.   
[20] Peng Wang, Shuai Bai, Hao Sinian, et al. Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12192, 2024.   
[21] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, et al. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, 2024.   
[22] Danfei Xu, Yuke Zhu, Christopher B Choy, and Li Fei-Fei. Scene graph generation by iterative message passing. In CVPR, 2017.   
[23] Ji Zhang, Kevin J Shih, Ahmed Elgammal, Andrew Tao, and Bryan Catanzaro. Graphical contrastive losses for scene graph generation. In CVPR, 2019.   
[24] Jingkang Yang, Yi Zhe Ang, Zujin Guo, Kaiyang Zhou, Wayne Zhang, and Ziwei Liu. Panoptic scene graph generation. In ECCV, 2022.   
[25] Rongjie Li, Songyang Zhang, Dahua Lin, Kai Chen, and Xuming He. From pixels to graphs: Open-vocabulary scene graph generation with vision-language models. In CVPR, 2024.

[26] Roei Herzig, Amir Mendelson, Leonid Karlinsky, et al. Incorporating structured representations into pretrained vision and language models using scene graphs. In EMNLP, 2023.   
[27] Jinbae Im, JeongYeon Nam, Nokyung Park, Hyungmin Lee, and Seunghyun Park. EGTR: Extracting graph from transformer for scene graph generation. In CVPR, 2024.   
[28] Jiankai Li et al. Leveraging predicate and triplet learning for scene graph generation. In CVPR, 2024.   
[29] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. IJCV, 123:32–73, 2017.   
[30] Rowan Zellers, Mark Yatskar, Sam Thomson, and Yejin Choi. Neural motifs: Scene graph parsing with global context. In CVPR, 2018.   
[31] Maximillian Nickel and Douwe Kiela. Poincaré embeddings for learning hierarchical representations. In NeurIPS, 2017.   
[32] Valentin Khrulkov, Leyla Mirvakhabova, Evgeniya Ustinova, Ivan Oseledets, and Victor Lempitsky. Hyperbolic image embeddings. In CVPR, 2020.   
[33] Octavian Ganea, Gary Bécigneul, and Thomas Hofmann. Hyperbolic neural networks. In NeurIPS, 2018.   
[34] Matthew Le, Stephen Roller, Laetitia Papaxanthos, Douwe Kiela, and Maximillian Nickel. Inferring concept hierarchies from text corpora via hyperbolic embeddings. In ACL, 2019.   
[35] Ivan Vendrov, Ryan Kiros, Sanja Fidler, and Raquel Urtasun. Order-embeddings of images and language. In ICLR, 2016.   
[36] Avik Pal, Max van Spengler, Guido Maria D’Amely di Melendugno, Alessandro Flaborea, Fabio Galasso, and Pascal Mettes. Compositional entailment learning for hyperbolic vision-language models. In ICLR, 2025.   
[37] Tobia Poppi, Tejaswi Kasarla, Pascal Mettes, Lorenzo Baraldi, and Rita Cucchiara. Hyperbolic safety-aware vision-language models. In CVPR, 2025.   
[38] Zelin Peng, Zhengqin Xu, Qingyang Liu, Xiaokang Yang, and Wei Shen. HyperET: Efficient training in hyperbolic space for multi-modal large language models. In NeurIPS, 2025.   
[39] Ivana Balazevic, Carl Allen, and Timothy Hospedales. Multi-relational Poincaré graph embeddings. In NeurIPS, 2019.   
[40] Ines Chami, Adva Wolf, Da-Cheng Crouse, et al. Low-dimensional hyperbolic knowledge graph embeddings. In ACL, 2020.   
[41] Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. RotatE: Knowledge graph embedding by relational rotation in complex space. In ICLR, 2019.   
[42] Victor Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. In NeurIPS, 2022.   
[43] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In ECCV, 2022.   
[44] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, Hongsheng Li, and Yu Qiao. LLaMA-Adapter V2: Parameter-efficient visual instruction model. In arXiv preprint arXiv:2304.15010, 2023.   
[45] Abraham Albert Ungar. Analytic hyperbolic geometry: Mathematical foundations and applications. World Scientific, 2005.   
[46] Anuj Diwan, Layne Berry, Eunsol Choi, David Harwath, and Kyle Mahowald. Why is winoground hard? investigating failures in visuolinguistic compositionality. In EMNLP, 2022.