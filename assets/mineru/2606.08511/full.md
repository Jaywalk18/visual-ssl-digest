# LOOK LESS, REASON MORE: BLOCK-WISE ATTENTION SKIPPING FOR EFFICIENT MULTIMODAL LLMs

Jie Ma Zhike Qiu Jiayi Ji Xiaoshuai Sun Rongrong Ji

Xiamen University

jiema100@stu.xmu.edu.cn, zhikeqiu@outlook.com

jjyxmu@gmail.com, xxsun@xmu.edu.cn, rrji@xmu.edu.cn

## ABSTRACT

Multimodal Large Language Models (MLLMs) face a significant inference bottleneck due to the quadratic computational cost of self-attention over long visual token sequences. However, we identify a critical inefficiency in current architectures: Visual Attention Saturation. Our analysis reveals that visual tokens rapidly establish their spatial structure and intra-modal relationships in early layers, rendering visual-to-visual self-attention in deeper layers computationally redundant. Conversely, Feed-Forward Networks in these layers remain essential for projecting visual features into the evolving textual semantic space. Leveraging this insight, we present Visual-Skip (V-Skip), a training-free inference paradigm that decouples spatial interaction from semantic evolution. Rather than discarding tokens, V-Skip imposes block-wise structured sparsity by selectively bypassing saturated visual self-attention modules. Furthermore, recognizing that varying downstream tasks demand distinct reasoning depths, V-Skip employs a lightweight, few-shot calibration to dynamically route the task-optimal sparsity path. Extensive experiments demonstrate that V-Skip effectively bypasses redundant vision attention to achieve block-wise sparsity, maintaining a 94.16% to 100.31% performance retention across diverse MLLMs. Ultimately, we prove that to reason more effectively, models do not need to discard what they see — they simply need to “look less” at the right depth.

## 1 Introduction

Multimodal Large Language Models (MLLMs) $[1, 2, 3, 4, 5, 6]$ have demonstrated remarkable capabilities in visual understanding and reasoning. To capture fine-grained details in complex scenes, recent models increasingly adopt high-resolution image input $[3, 7]$ , representing visual content as long sequences of visual tokens (e.g., expanding from 576 to over 2800 tokens). While this higher resolution enhances perception, it introduces a severe computational bottleneck. The self-attention mechanism in Transformers exhibits quadratic complexity with respect to the sequence length. Consequently, as the number of visual tokens grows, the computational cost and memory consumption during the prefill phase surge disproportionately, resulting in prohibitive latency for real-time applications.

To mitigate this inefficiency, recent research has focused on reducing the number of visual tokens. Prominent methods, such as Token Pruning $[10, 11, 12, 13]$ and Token Merging $[14]$ , accelerate inference by aggressively identifying and discarding “redundant” tokens. As illustrated in Figure 1 (Left)(a,b), more aggressive approaches $[8, 9]$ extend this paradigm by progressively terminating visual computation in intermediate layers or freezing visual representations by skipping entire transformer blocks. While effective in reducing FLOPs, these approaches are inherently destructive. Specifically, by permanently removing tokens or skipping layer-wise feature updates, they irreversibly compromise visual integrity or disrupt feature evolution. This often leads to a loss of fine-grained details and an increased risk of object hallucination as intuitively demonstrated by the OCR failure case in Figure 1 (Right), where the model fails to ground its reasoning in the actual image content. This creates a challenging trade-off: current acceleration methods sacrifice visual integrity for inference speed.

To break this trade-off, we move beyond the coarse-grained paradigm of indiscriminately dropping tokens or freezing entire layers. Instead, we ask: Are all operations within a deep transformer block equally redundant for visual tokens? Through a layer-wise analysis of attention patterns, we identify a critical phenomenon termed Visual Attention

![](images/b940cb63c53f96fb16dbc89dbb8935b806252f3a9a746bf27a49979850cb410d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["(a) VTW Layer"] --> B["(b) ShortV Layer"]
  B --> C["(c) V-Skip Layer"]

    subgraph VTW Layer
  D["Visual Token"] --> E["Textual Token"]
  F["Discards Tokens (Destructive)"] --> G["Skips Layers (Misalignment)"]
  H["Bypasses Attn (Efficient & Aligned)"] --> I["Textual Token"]
    end

    subgraph ShortV Layer
  J["Visual Token"] --> K["Textual Token"]
  L["Discards Tokens (Destructive)"] --> M["Skips Layers (Misalignment)"]
  N["Bypasses Attn (Efficient & Aligned)"] --> O["Textual Token"]
    end

    subgraph V_Skip_Layer["V-Skip Layer"]
  P["Visual Token"] --> Q["Textual Token"]
  R["Discards Tokens (Destructive)"] --> S["Skips Layers (Misalignment)"]
  T["Bypasses Attn (Efficient & Aligned)"] --> U["Textual Token"]
    end

    style VTW Layer fill:#f9f,stroke:#333
    style ShortV Layer fill:#f9f,stroke:#333
    style V-Skip Layer fill:#f9f,stroke:#333

    note right of A: VTW
    note right of B: ShortV
    note right of C: V-Skip
    note right of Q: What is written in the image?
    Q: CANARY
    end

    note right of Q: VTW
    note right of S
    note right of T
    note right of U
```
</details>

Figure 1: Conceptual and qualitative comparison of visual acceleration paradigms. (Left) Existing methods operate destructively: VTW [8] drops tokens (losing context), and ShortV [9] skips entire layers (disrupting semantic evolution). Our V-Skip uniquely decouples the block, bypassing redundant spatial attention while keeping FFNs active for continuous semantic alignments. (Right) A real-world OCR example illustrating the consequences of these paradigms. VTW loses fine-grained visual details (outputting generic “Words”). ShortV suffers from feature misalignment due to missing FFNs, leading to semantic hallucination (“Canyon”). V-Skip successfully maintains full visual and semantic integrity (“CANARY”).

Saturation. We observe that visual processing in MLLMs undergoes a functional evolution across layers. In early layers, visual tokens require dense spatial interaction (Self-Attention) to aggregate local features into coherent objects. However, in deeper layers, this spatial structure stabilizes, and the intra-modal attention maps become static. At this stage, the primary role of visual tokens shifts from “spatial construction” to “semantic reasoning”, serving as keys for the Language Model to query $[15, 16]$ . Crucially, while the quadratic self-attention computation becomes redundant in these deep layers, the token-wise transformation via Feed-Forward Networks (FFNs) remains essential for aligning visual features with the evolving semantic space of the Large Language Model.

Guided by this insight, we introduce Visual-Skip (V-Skip), a simple yet effective training-free acceleration strategy for MLLMs, as illustrated in Figure 1 (Left)(c). Unlike pruning methods that drop tokens, V-Skip preserves the complete sequence of visual tokens throughout the network. Instead, it introduces block-wise structured sparsity by selectively bypassing the visual self-attention computation in deep, saturated layers while fully retaining the FFN transformations. Furthermore, recognizing that diverse downstream tasks require varying depths of spatial reasoning, V-Skip employs a highly efficient few-shot calibration mechanism to dynamically identify the task-optimal sparsity path. This approach effectively decouples spatial interaction from semantic alignment, allowing the model to “Look Less” (eliminating quadratic redundancy) while continuing to “Reason More” (maintaining semantic depth). By avoiding the structural damage caused by token dropping, V-Skip ensures that the model retains full access to visual context for complex reasoning tasks.

Our contributions can be summarized as follows:

- We uncover the phenomenon of Visual Attention Saturation in MLLMs, revealing that deep-layer visual self-attention becomes computationally redundant, whereas FFNs remain critical for layer-wise semantic alignment.  
- We introduce V-Skip, a training-free paradigm that decouples spatial interaction from semantic evolution. It achieves block-wise sparsity by bypassing saturated attention modules without retraining.  
- Extensive experiments validate the superiority of our paradigm. V-Skip achieves an impressive $98.42\%$ to $100.31\%$ performance retention across the LLaVA series, alongside a robust $94.16\%$ on the Qwen architecture. Notably, by preserving complete visual context, it overcomes the structural damage of traditional pruning methods, significantly mitigating object hallucinations in complex reasoning tasks.

## 2 Related Work

MLLMs. The rapid evolution of Multimodal Large Language Models (MLLMs) has been driven by the integration of powerful visual encoders with Large Language Models (LLMs) $[1, 2, 4, 3, 6, 5]$ . While early models operated on low-resolution inputs (e.g., $224^{2}$ or $336^{2}$ ), recent advancements like LLaVA-NeXT $[3]$ , GPT $[17]$ , and Gemini $[18]$ have shifted towards high-resolution visual processing to capture fine-grained details and text in images. For instance,

![](images/155648a1cc0e6bc1adaa49ecb143eedccaab53ffca1c4b251d0fc9bfc51aa692.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["V-Skip Layer"] --> B["Decoder Layer"]
  B --> C["Visual Token"]
  B --> D["Textual Token"]
  E["Multi-Head Self-Attention"] --> F["Visual Bypass"]
  F --> G["Input X"]
  F --> H["Input K"]
  F --> I["Output Q"]
  J["Feed-Forward Network"] --> F
    style A fill:#cce5ff,stroke:#333
    style B fill:#cce5ff,stroke:#333
    style C fill:#cce5ff,stroke:#333
    style D fill:#cce5ff,stroke:#333
    style E fill:#cce5ff,stroke:#333
    style F fill:#cce5ff,stroke:#333
    style G fill:#cce5ff,stroke:#333
    style H fill:#cce5ff,stroke:#333
    style I fill:#cce5ff,stroke:#333
```
</details>

Figure 2: Illustration of the V-Skip. In identified visual attention saturated layers, we decouple the computational path of visual and textual tokens. Visual tokens (blue) utilize a Visual Bypass to skip the self-attention calculation, denoted by the red crosses. Textual tokens (orange) maintain full attention to both visual and textual contexts for robust reasoning. Unlike layer-dropping methods, V-Skip processes all tokens through the Feed-Forward Network, ensuring continuous layer-wise semantic evolution and alignment. Best viewed in color.

LLaVA-NeXT adopts an “AnyRes” strategy, dynamically tiling images into multiple patches, which can increase the visual token sequence length from 576 to over 2800 tokens. However, this resolution scaling comes at a steep computational cost. Since the self-attention mechanism in Transformers scales quadratically with sequence length $[19]$ , the latency and memory usage grow prohibitively in high-resolution settings. This creates an urgent need for efficient inference strategies that can handle long visual sequences without compromising the model’s perceptual capabilities, driving the exploration of structural redundancy.

Efficiency via Redundancy Reduction. To mitigate the computational overhead of high-resolution visual sequences, prior research has primarily divided into token reduction and structural pruning paradigms. Token reduction strategies $[10, 11, 12, 13, 8]$ accelerate inference by progressively discarding visual token computation. However, these methods operate on a destructive premise: the permanent exclusion or premature withdrawal of tokens interrupts continuous visual context, often exacerbating object hallucination due to compromised visual representations $[20]$ . Conversely, structural approaches $[9, 8]$ target architectural redundancy by skipping entire transformer blocks or “freezing” visual representations. Crucially, simply bypassing the Feed-Forward Networks neglects the necessity of layer-wise semantic evolution $[21]$ , resulting in feature misalignment between the frozen visual tokens and the increasingly abstract textual manifold $[22]$ . Diverging from these paradigms, we identify a modality-specific functional decoupling: while layer-wise spatial interaction (Attention) saturates, semantic alignment (FFN) remains indispensable. Guided by this insight, our V-Skip mechanism selectively bypasses visual attention to eliminate quadratic redundancy, avoiding both the information loss inherent in pruning and the semantic misalignment caused by layer freezing.

## 3 Methodology

In this section, we introduce the proposed V-Skip paradigm and describe its design in detail. We begin by formalizing the computational bottleneck in MLLMs in Section 3.1, then present the Visual Information Gain metric in Section 3.2 for identifying saturated blocks, and finally Section 3.3 details the block-wise attention skipping mechanism, as illustrated in Figure 2.

## 3.1 Problem Formulation

We consider a Multimodal Large Language Model (MLLM) parameterized by $\theta$ , which processes a multimodal sequence consisting of visual inputs I and textual instructions $X_{t}$ . The visual inputs are encoded into a sequence of visual embeddings $X_{v} \in R^{N_{v} \times d}$ via a vision encoder and a projector, where $N_{v}$ denotes the number of visual tokens. The textual inputs are tokenized into $X_{t} \in R^{N_{t} \times d}$ . The concatenated sequence $X = [X_{v}; X_{t}] \in R^{N \times d}$ serves as the input to the LLM backbone, where $N = N_{v} + N_{t}$ .

![](images/55c725274cc1cf7826f0845f4e3b03f8b6b400b8f69b87d57d8a9ce6bb2792a9.jpg)

<details>
<summary>line chart</summary>

| Transformer Layer Index (l) | MME       | GQA       | AI2D      | MMMU      | POPE      | VizWiz    | MMBench   | ScienceQA | OCRBench  |
| --------------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| 0                           | 1e-4      | 1e-3      | 1e-3      | 1e-4      | 1e-4      | 1e-3      | 1e-4      | 1e-4      | 1e-4      |
| 2                           | 1e-4      | 1e-3      | 1e-3      | 1e-4      | 1e-4      | 1e-3      | 1e-4      | 1e-4      | 1e-4      |
| 4                           | 1e-4      | 1e-3      | 1e-3      | 1e-4      | 1e-4      | 1e-3      | 1e-4      | 1e-4      | 1e-4      |
| 6                           | 1e-4      | 1e-3      | 1e-2      | 1e-4      | 1e-4      | 1e-2      | 1e-3      | 1e-4      | 1e-4      |
| 8                           | 1e-4      | 1e-3      | 1e-2      | 1e-4      | 1e-4      | 1e-2      | 1e-3      | 1e-4      | 1e-4      |
| 10                          | 1e-4      | 1e-3      | 1e-2      | 1e-4      | 1e-4      | 1e-2      | 1e-3      | 1e-4      | 1e-4      |
| 12                          | 1e-4      | 1e-3      | 1e-2      | 1e-4      | 1e-4      | 1e-2      | 1e-3      | 1e-4      | 1e-4      |
| 14                          | 1e-5      | 1e-3      | 1e-2      | 1e-5      | 1e-5      | 1e-2      | 1e-3      | 1e-4      | 1e-4      |
| 16                          | 1e-5      | 1e-3      | 1e-2      | 1e-5      | 1e-5      | 1e-2      | 1e-3      | 1e-4      | 1e-4      |
| 18                          | 1e-5      | 1e-3      | 1e-2      | 1e-5      | 1e-7      | 1e-2      | 1e-3      | 1e-4      | 1e-4      |
| 20                          | 1e-5      | 1e-3      | 1e-2      | 1e-5      | 1e-5      | 1e-2      | 1e-3      | 1e-4      | 1e-4      |
| 22                          | 1e-5      | 1e-3      | 1e-2      | 1e-5      | 1e-5      | 1e-2      | 1e-3      | 1e-4      | 1e-4      |
| 24                          | 1e-5      | 1e-3      | 1e-2      | 1e-5      | 1e-5      | 1e-2      | 1e-3      | 1e-4      | 1e-4      |
| 26                          | 1e-5      | 1e-3      | 1e-2      | 1e-5      | 1e-5      | 1e-2      | 1e-3      | 1e-4      | 1e-4      |
| 28                          | 1e-5      | 1e-3      | 1e-2      | 1e-5      | 1e-5      | 1e-2      | 1e-3      | 1e-4      | 1e-4      |
| 30                          | 1e-5      | 1e-3      | 1e-2      | 1e-5      | 1e-5      | 1e-2      | 1e-3      | 1e-4      | 1e-4      |
| 32                          | 1e-5      | 1e-3      | 1e-2      | 1e-5      | 1e-6      | 1e-2      | 1e-3      | 1e-4      | 1e-4      |
</details>

![](images/e0a463d6eed4162904cffc2c84ab6fa2f1cd00b3694a06cb83272684a9153b46.jpg)

<details>
<summary>bar chart</summary>

| Transformer Layer Index (l) | VIG G(l) (×10⁻³) |
| -------------------------- | ---------------- |
| 0                          | 0.0              |
| 1                          | 0.4              |
| 2                          | 0.8              |
| 3                          | 0.2              |
| 4                          | 0.5              |
| 5                          | 1.5              |
| 6                          | 1.7              |
| 7                          | 1.9              |
| 8                          | 1.8              |
| 9                          | 3.3              |
| 10                         | 3.6              |
| 11                         | 0.2              |
| 12                         | 0.1              |
| 13                         | 0.0              |
| 14                         | 0.0              |
| 15                         | 0.0              |
| 16                         | 0.0              |
| 17                         | 0.0              |
| 18                         | 0.0              |
| 19                         | 0.0              |
| 20                         | 0.0              |
| 21                         | 0.0              |
| 22                         | 0.0              |
| 23                         | 0.0              |
| 24                         | 0.0              |
| 25                         | 0.0              |
| 26                         | 0.0              |
| 27                         | 0.0              |
| 28                         | 0.0              |
| 29                         | 0.0              |
| 30                         | 0.0              |
| 31                         | 0.0              |
| 32                         | 0.0              |
</details>

Figure 3: Empirical analysis of Visual Attention Saturation. (Left) Task-specific VIG profiles $\mathcal{G}_{k}(l)$ across nine representative multimodal benchmarks. While all tasks generally exhibit a transition toward saturation in deep layers, their specific trajectories and attention demands vary significantly, highlighting the necessity of task-aware calibration. (Right) A detailed block-wise VIG bar chart using the MME dataset as a case study. A stark phase transition is observed: early layers (blue bars) actively aggregate spatial information, whereas deeper layers (grey bars) reach attention saturation, rendering further intra-modal queries computationally redundant.

The LLM backbone consists of $L$ stacked Transformer decoder layers. For a specific layer $l$ , the hidden state $\mathbf{H}^l \in \mathbb{R}^{N \times d}$ is updated via Multi-Head Self-Attention (MHSA) and a Feed-Forward Network (FFN):

$$
\mathbf {H} ^ {\prime l} = \operatorname{MHSA} \left(\mathrm{LN} \left(\mathbf {H} ^ {l}\right)\right) + \mathbf {H} ^ {l}, \tag {1}
$$

$$
\mathbf {H} ^ {l + 1} = \operatorname{FFN} \left(\mathrm{LN} \left(\mathbf {H} ^ {\prime l}\right)\right) + \mathbf {H} ^ {\prime l}, \tag {2}
$$

where $\mathrm{LN}(\cdot)$ denotes Layer Normalization.

Computational Bottleneck. The complexity of the MHSA operation in Eq. (1) is $\mathcal{O}(N^{2})$ . In high-resolution settings (e.g., LLaVA-NeXT), $N_{v} \gg N_{t}$ . Consequently, the intra-modal attention among visual tokens, which constitutes a complexity of $\mathcal{O}(N_{v}^{2})$ , dominates the prefill latency.

## 3.2 Visual Attention Saturation: A Block-wise Analysis

Existing acceleration methods typically assume redundancy in the quantity of visual tokens. In contrast, we investigate the redundancy in the computational structure across network depth. The visual representation learning in MLLMs undergoes a functional phase transition $[15]$ : from spatial aggregation in early layers to semantic alignment in deep layers.

To rigorously quantify this transition, we propose the Visual Information Gain (VIG) metric based on prediction sensitivity. Rather than measuring intermediate feature norms, we evaluate the impact of bypassing visual attention at a specific layer $l$ on the model's final probability distribution.

Let $P(\mathbf{y}|\mathbf{X};\theta)$ denote the predictive distribution of the full model over the vocabulary given input X. Let $P_{-l}(\mathbf{y}|\mathbf{X};\theta)$ denote the distribution obtained when the intra-modal visual self-attention at layer l is bypassed (i.e., skipping the $\mathcal{O}(N_v^2)$ mixing), while keeping the FFN active. Let $\mathcal{D}_k$ denote a specific downstream task dataset (e.g., OCR, VQA, or Chart Understanding). We define the Task-Specific VIG for layer l under task k, denoted as $\mathcal{G}_k(l)$ , as the expected Kullback-Leibler (KL) divergence computed over a calibration set randomly sampled from $\mathcal{D}_k$ :

$$
\mathcal {G} _ {k} (l) = \mathbb {E} _ {X \sim \mathcal {D} _ {k}} \left[ D _ {K L} (P (y | X) | | P _ {\neg l} (y | X)) \right] \tag {3}
$$

To capture these task-specific dynamics with minimal overhead, we perform a lightweight few-shot calibration by randomly sampling merely 20 image-text pairs from $D_{k}$ to compute this expectation. This highly efficient sampling strategy guarantees that the measured saturation pattern accurately reflects the unique reasoning pathway required by the specific task, avoiding the bias of a unified mask (e.g., ShortV) while imposing negligible pre-computation burdens.

Empirical Observation of Saturation. To validate visual attention saturation, we visualize the computed VIG across the 32 transformer layers with LLaVA-1.5-7B. As explicitly shown in Figure 3 (Right) using the MME benchmark as a case study, $\mathcal{G}(l)$ reveals a stark and undeniable phase transition. Early and intermediate layers (primarily indices 1–12) exhibit exceptionally high information gain, confirming their critical role in active spatial aggregation and object formulation. Conversely, the VIG plunges drastically and flattens below the adaptive threshold in deeper layers, empirically verifying the Visual Attention Saturation phenomenon.

Crucially, our expanded analysis across multiple benchmarks in Figure 3 (Left) uncovers a deeper architectural insight: saturation profiles are inherently task-dependent. While all nine evaluated datasets eventually converge to a low-VIG saturated state, their evolutionary trajectories vary substantially. For instance, fine-grained visual recognition tasks (e.g., VizWiz, GQA) exhibit different deep-layer attention fluctuations compared to hallucination-sensitive tasks (e.g., POPE, which drops sharply at layer 19). This compelling empirical evidence directly invalidates the use of a monolithic, one-size-fits-all pruning mask (e.g., ShortV, VTW). Instead, it firmly justifies our few-shot calibration strategy, which elegantly tailors the optimal skipping subset to the unique semantic reasoning pathways required by each specific downstream task.

## 3.3 V-Skip: Block-wise Structured Sparsity

Based on the saturation analysis, we propose V-Skip, a training-free acceleration mechanism that imposes structured sparsity on the attention mechanism in deep layers. To ensure robustness across different architectures and tasks, we adopt a data-driven selection strategy rather than fixed heuristics.

Identification of Saturated Layers via Top-N. To maximize efficiency under a computational budget, we identify the top-N most saturated layers to bypass. Formally, let $L_{all} = \{0, 1, \ldots, L - 1\}$ be the set of all decoder layers. We seek a subset $\mathcal{L}_{skip}^{(k)} \subset \mathcal{L}_{all}$ with cardinality $|\mathcal{L}_{skip}^{(k)}| = N$ such that the total information loss is minimized:

$$
\mathcal {L} _ {\text {skip}} ^ {(k)} = \underset {\mathcal {S} \subset \mathcal {L} _ {\text {all}}, | \mathcal {S} | = \mathrm{N}} {\arg \min} \sum_ {l \in \mathcal {S}} \mathcal {G} _ {k} (l) \tag {4}
$$

This optimization effectively selects the N layers with the lowest $\mathcal{G}_{k}(l)$ scores. By adjusting N, V-Skip offers a flexible trade-off between speed and precision.

Decoupled Attention Masking. Let $\mathcal{L}_{skip}^{(k)}$ denote the set of saturated layers (e.g., layers 12 to 31). For any layer $l \in \mathcal{L}_{skip}^{(k)}$ , we modify the standard causal attention mask $M \in \{0, -\infty\}^{N \times N}$ to a block-sparse mask $M_{skip}$ . Specifically, the standard attention score calculation is:

$$
\mathbf {S} = \text { Softmax } \left(\frac {\mathbf {Q} \mathbf {K} ^ {T}}{\sqrt {d}} + \mathbf {M}\right). \tag {5}
$$

In V-Skip, we alter the visibility of visual tokens. For query tokens $i \in V$ (visual tokens), we enforce a bypass mechanism, effectively eliminating the expensive query-key product:

$$
\mathbf {H} _ {\mathcal {V}} ^ {l} \longleftarrow \mathrm{LN} (\mathbf {H} _ {\mathcal {V}} ^ {l}) + \mathbf {H} _ {\mathcal {V}} ^ {l}. \quad (\text { Visual   Bypass }) \tag {6}
$$

For query tokens $j \in T$ (textual tokens), the attention remains fully accessible to both visual and textual contexts to preserve reasoning capabilities:

$$
\mathbf {h} _ {j} ^ {\prime l} = \sum_ {k \in \mathcal {V} \cup \mathcal {T}} \operatorname{Attn} (q _ {j}, k _ {k}) \cdot v _ {k}. \quad (\text { Textual   Reasoning }) \tag {7}
$$

This formulation is equivalent to applying a heterogeneous attention mask where visual queries cannot attend to other visual tokens, but textual queries retain full visibility.

FFN Preservation for Alignment. Crucially, distinct from layer-skipping approaches (e.g., ShortV), V-Skip retains the FFN operation for all tokens:

$$
\mathbf {H} ^ {l + 1} = \operatorname{FFN} \left(\mathrm{LN} \left(\mathbf {H} ^ {\prime l}\right)\right) + \mathbf {H} ^ {\prime l}. \tag {8}
$$

This design choice is theoretically grounded in our observation that while spatial mixing saturates, the point-wise non-linear projection $f: \mathbb{R}^d \to \mathbb{R}^d$ performed by the FFN is essential for maintaining the semantic trajectory of visual tokens within the LLM's manifold.

Complexity Analysis. Standard MHSA requires $\mathcal{O}((N_{v}+N_{t})^{2})$ operations. V-Skip reduces the complexity in skipped layers to $\mathcal{O}(N_{t}^{2}+N_{t}\cdot N_{v})$ . Since $N_{v}\gg N_{t}$ , the dominant quadratic term $N_{v}^{2}$ is eliminated. The theoretical speedup ratio for the attention module approaches $N_{v}/N_{t}+1$ , offering significant latency reduction for high-resolution inputs.

## 4 Experiments

## 4.1 Experimental Setup

We evaluate our method on representative MLLM architectures, specifically LLaVA-1.5 (7B/13B) [2], LLaVA-NeXT (7B/13B) [3] and Qwen2.5-VL-7B [6]. LLaVA-1.5 processes $336 \times 336$ images resulting in 576 visual tokens, whereas LLaVA-NeXT employs “AnyRes” tiling to scale the visual sequence up to 2,880 tokens per image. This extended sequence length explicitly exposes the quadratic bottleneck of visual self-attention. All evaluations are conducted using the standardized lmms-eval framework, strictly adhering to its official protocols. Our experiments were executed on an NVIDIA RTX 3090 GPU.

To validate our structural optimization paradigm, we compare V-Skip against representative training-free methods: VTW[8] and ShortV[9]. These methods are selected as they also operate on depth-wise structural modifications. However, they inherently differ from our routing mechanism: VTW destructively discards all visual tokens after a predefined layer $K$ , while ShortV discretely skips entire LLM layers (losing critical FFN transformations). For fair comparison, we adopt the default configurations reported in their respective works: $K = 16$ (7B) and $K = 20$ (13B) for VTW; N=19 (7B) and N=24 (13B) for ShortV.

Benchmarks. To evaluate the general and fine-grained capabilities of V-Skip, we conduct experiments on a comprehensive suite of nine multimodal benchmarks. These include general evaluation (MME [23], MMBench [24]), expert-level knowledge and reasoning (MMMU [25], GQA [26], ScienceQA [27]), object hallucination sensitivity (POPE [28]), and specialized tasks such as structured diagram understanding (AI2D [29]), in-the-wild VQA (VizWiz [30]), and fine-grained text recognition (OCRBench [31]).

## 4.2 Main Results

The performance comparison of V-Skip against representative training-free methods is summarized in Table 1. Overall, V-Skip consistently achieves superior accuracy retention across all evaluated architectures, spanning the LLaVA series to the state-of-the-art Qwen2.5-VL-7B. By effectively bypassing redundant visual self-attention while preserving FFN-based semantic evolution, V-Skip maintains a robust average performance retention of 98.42% to 100.31% on the LLaVA series, and achieves a highly competitive 94.16% on the Qwen architecture.

Superiority over Destructive Pruning. As observed on LLaVA-1.5-7B, V-Skip achieves a near-lossless average retention of 100.31%, even slightly surpassing the vanilla model on tasks like MMBench and OCRBench. In contrast, token pruning methods such as VTW suffer from catastrophic degradation on fine-grained tasks (e.g., dropping from 31.3 to 5.1 on OCRBench). This highlights that preserving full visual context is essential for complex multimodal reasoning. V-Skip satisfies this through a non-destructive architectural design, eliminating redundancy while maintaining the integrity of the visual tokens.

Generalizability across Scales and MLLM Paradigms. V-Skip consistently maintains near-lossless accuracy on 13B models, achieving 99.77% and 99.75% retention on LLaVA-1.5-13B and LLaVA-NeXT-13B, respectively. It remains robust on the highly optimized Qwen2.5-VL-7B with 94.16% retention, significantly outperforming the layer-skipping approach ShortV (87.11%). These results prove that skipping visual attention saturation blocks is a robust strategy compared to token withdrawal or layer freezing.

## 4.3 V-Skip as a Universal Plug-and-Play Booster

To verify that V-Skip is complementary to existing acceleration techniques, we integrate it with FastV, a representative visual token pruning method. We set the pruning layer index $K$ to 3 and the keep token pruning ratio to 0.5. For our module, we configure the model with a sparsity budget of N=19. As presented in Table 2, while FastV efficiently reduces token count, it inevitably introduces subtle performance trade-offs relative to the dense baseline. Remarkably, integrating V-Skip as a booster not only further compresses the computational load but also actively restores and even enhances the reasoning capability across all nine benchmarks. Specifically, compared to the FastV, the combined FastV+V-Skip improves MME by +10.0, MMMU by +0.4, POPE by +0.2, and OCRBench by +0.6. From an efficiency perspective, V-Skip further reduces the computational requirement from 61.66% to 56.73% TFLOPs of the vanilla model. These results demonstrate that V-Skip serves as a powerful plug-and-play component. While token pruning reduces the input quantity, V-Skip optimizes the visual interactions within the attention blocks. This synergy allows for multi-dimensional sparsity while simultaneously safeguarding the model's reasoning capability.

Table 1: Performance comparison of various methods across different benchmarks. Best results in red, and second best results in blue.

<table><tr><td>Methods</td><td>MME</td><td>MMMU</td><td>MMB</td><td>GQA</td><td>POPE</td><td>SQA</td><td>AI2D</td><td>VizWiz</td><td>OCRB</td><td>Avg.</td></tr><tr><td>LLaVA-1.5-7B</td><td>1866.1</td><td>36.7</td><td>64.1</td><td>61.9</td><td>85.1</td><td>69.6</td><td>55.2</td><td>54.3</td><td>31.3</td><td>100.00%</td></tr><tr><td>VTW(K=16)</td><td>1857.6</td><td>36.4</td><td>64.0</td><td>55.1</td><td>85.3</td><td>69.6</td><td>55.4</td><td>51.0</td><td>5.1</td><td>88.71%</td></tr><tr><td>ShortV(N=19)</td><td>1842.2</td><td>36.2</td><td>64.8</td><td>60.8</td><td>84.4</td><td>68.3</td><td>54.2</td><td>50.6</td><td>29.5</td><td>97.73%</td></tr><tr><td>V-Skip(N=19)</td><td>1853.3</td><td>36.9</td><td>65.0</td><td>61.8</td><td>85.5</td><td>69.3</td><td>55.4</td><td>54.5</td><td>31.6</td><td>100.31%</td></tr><tr><td>LLaVA-1.5-13B</td><td>1824.2</td><td>35.7</td><td>68.7</td><td>63.3</td><td>85.6</td><td>72.7</td><td>59.3</td><td>56.6</td><td>33.7</td><td>100.00%</td></tr><tr><td>VTW(K=20)</td><td>1828.8</td><td>34.9</td><td>68.8</td><td>60.6</td><td>87.1</td><td>72.9</td><td>59.4</td><td>55.1</td><td>24.2</td><td>96.14%</td></tr><tr><td>ShortV(N=24)</td><td>1828.7</td><td>35.7</td><td>68.7</td><td>63.3</td><td>85.6</td><td>72.7</td><td>59.3</td><td>56.6</td><td>32.1</td><td>99.50%</td></tr><tr><td>V-Skip(N=24)</td><td>1831.9</td><td>35.4</td><td>68.7</td><td>62.8</td><td>85.5</td><td>72.9</td><td>59.2</td><td>57.3</td><td>33.0</td><td>99.77%</td></tr><tr><td>LLaVA-NeXT-7B</td><td>1846.3</td><td>36.7</td><td>67.2</td><td>64.3</td><td>86.4</td><td>70.2</td><td>65.3</td><td>60.6</td><td>52.3</td><td>100.00%</td></tr><tr><td>VTW(K=16)</td><td>1853.9</td><td>36.8</td><td>67.1</td><td>63.7</td><td>86.4</td><td>66.5</td><td>65.6</td><td>59.6</td><td>6.5</td><td>89.51%</td></tr><tr><td>ShortV(N=19)</td><td>1844.1</td><td>35.8</td><td>67.2</td><td>63.5</td><td>86.2</td><td>69.1</td><td>64.4</td><td>57.7</td><td>45.6</td><td>97.27%</td></tr><tr><td>V-Skip(N=19)</td><td>1778.6</td><td>36.6</td><td>66.8</td><td>63.4</td><td>86.3</td><td>70.0</td><td>65.2</td><td>59.3</td><td>49.4</td><td>98.42%</td></tr><tr><td>LLaVA-NeXT-13B</td><td>1892.0</td><td>34.9</td><td>69.2</td><td>65.4</td><td>86.4</td><td>73.6</td><td>70.3</td><td>63.6</td><td>55.1</td><td>100.00%</td></tr><tr><td>VTW(K=20)</td><td>1878.1</td><td>36.1</td><td>69.2</td><td>61.4</td><td>87.4</td><td>73.4</td><td>70.2</td><td>58.5</td><td>35.5</td><td>94.86%</td></tr><tr><td>ShortV(N=24)</td><td>1899.6</td><td>36.1</td><td>69.1</td><td>63.6</td><td>86.2</td><td>73.1</td><td>69.5</td><td>59.2</td><td>49.6</td><td>98.00%</td></tr><tr><td>V-Skip(N=24)</td><td>1885.4</td><td>35.3</td><td>69.3</td><td>64.9</td><td>86.5</td><td>73.9</td><td>70.5</td><td>62.9</td><td>53.9</td><td>99.75%</td></tr><tr><td>Qwen2.5-VL-7B</td><td>2344.1</td><td>50.4</td><td>83.6</td><td>60.5</td><td>86.4</td><td>76.4</td><td>82.7</td><td>70.2</td><td>84.4</td><td>100.00%</td></tr><tr><td>VTW(K=7)</td><td>2319.9</td><td>46.8</td><td>72.8</td><td>56.5</td><td>86.2</td><td>75.3</td><td>76.4</td><td>66.4</td><td>74.6</td><td>94.00%</td></tr><tr><td>ShortV(N=7)</td><td>2092.1</td><td>42.8</td><td>72.8</td><td>54.6</td><td>84.0</td><td>78.3</td><td>55.2</td><td>50.0</td><td>80.0</td><td>87.11%</td></tr><tr><td>V-Skip(N=7)</td><td>2228.8</td><td>43.3</td><td>74.7</td><td>51.3</td><td>86.2</td><td>81.7</td><td>76.5</td><td>66.4</td><td>83.1</td><td>94.16%</td></tr></table>

Table 2: Integrating V-Skip with FastV to enhance both reasoning capability and efficiency on LLaVA-1.5-7B.

<table><tr><td>Method</td><td>MME</td><td>MMMU</td><td>MMB</td><td>GQA</td><td>POPE</td><td>SQA</td><td>AI2D</td><td>VizWiz</td><td>OCRB</td><td>TFLOPs</td></tr><tr><td>LLaVA-1.5-7B</td><td>1866.1</td><td>36.7</td><td>64.1</td><td>61.9</td><td>85.1</td><td>69.6</td><td>55.2</td><td>54.3</td><td>31.3</td><td>100.00%</td></tr><tr><td>FastV</td><td>1863.8</td><td>35.8</td><td>63.8</td><td>60.4</td><td>83.9</td><td>68.7</td><td>55.1</td><td>54.5</td><td>30.6</td><td>61.66%</td></tr><tr><td>FastV+V-Skip</td><td>1873.8</td><td>36.2</td><td>63.4</td><td>60.1</td><td>84.1</td><td>69.0</td><td>55.2</td><td>54.7</td><td>31.2</td><td>56.73%</td></tr></table>

## 4.4 Analyzing the Efficiency-Effectiveness Balance

We analyze the trade-off between theoretical complexity (TFLOPs), inference latency, and accuracy retention. As summarized in Table 3, V-Skip establishes a compelling efficiency-effectiveness trade-off by effectively mitigating the quadratic bottleneck of long visual sequences while preserving the model's ability. On LLaVA-1.5-7B, V-Skip reduces TFLOPs from 8.54 to 7.69 and achieves a 13.31 ms reduction in latency. Crucially, unlike compared methods, V-Skip uniquely attains a +0.31% average performance gain, proving that eliminating redundant attention can even serve as a beneficial regularizer. This trend holds on the higher-resolution LLaVA-NeXT-7B, where V-Skip maintains the highest retention (98.42%) among all methods, significantly outperforming the destructive pruning of VTW (89.51%).

<table><tr><td>Method</td><td>TFLOPs</td><td>Latency</td><td>Avg. (Δ)</td></tr><tr><td>LLaVA-1.5-7B</td><td>8.54</td><td>171.71</td><td>100.00%</td></tr><tr><td>V-Skip (N=19)</td><td>7.69</td><td>158.40</td><td>100.31% (↑0.31)</td></tr><tr><td>VTW (K=16)</td><td>4.94</td><td>107.57</td><td>88.71% (↓11.29)</td></tr><tr><td>ShortV (N=19)</td><td>4.72</td><td>108.40</td><td>97.73% (↓2.27)</td></tr><tr><td>LLaVA-NeXT-7B</td><td>30.74</td><td>527.34</td><td>100.00%</td></tr><tr><td>V-Skip (N=19)</td><td>26.57</td><td>486.50</td><td>98.42% (↓1.58)</td></tr><tr><td>VTW (K=16)</td><td>16.91</td><td>366.21</td><td>89.51% (↓10.49)</td></tr><tr><td>ShortV (N=19)</td><td>15.73</td><td>332.22</td><td>97.27% (↓2.73)</td></tr></table>

Table 3: Performance and efficiency comparison across MLLM architectures.

While VTW and ShortV offer more aggressive reductions in TFLOPs, they suffer from catastrophic degradation (up to $-11.29\%$ ) due to their disruption of the model's architectural integrity. We argue that preserving FFN transformations is an essential investment; while bypassing the $\mathcal{O}(N^2)$ visual self-attention captures the bulk of computational redundancy, the FFNs remain vital for maintaining the semantic trajectory of visual tokens. This strategic decoupling allows V-Skip to provide meaningful acceleration while ensuring the model remains robust for complex multimodal tasks.

## 4.5 Ablation Studies

## 4.5.1 Decoupling Spatial Interaction from Semantic Evolution.

To validate our functional decoupling strategy, we investigate whether deep-layer redundancy in MLLMs primarily resides in spatial interaction or semantic projection. We compare V-Skip against an FFN-Skip strategy, where visual tokens bypass FFN transformations but continue to participate in self-attention, effectively halting their non-linear evolution while maintaining quadratic spatial mixing.

As presented in Table 4, V-Skip demonstrates superior robustness with an average performance retention of 100.31% compared to 99.16% for FFN-Skip. While FFN-Skip shows marginal gains on MME and ScienceQA, it suffers noticeable degradations on tasks requiring fine-grained visual grounding, such as OCRBench and the hallucination-sensitive POPE benchmark. These results corroborate that bypassing the FFN block prevents visual tokens from undergoing the necessary layer-wise semantic evolution required for high-level reasoning. Consequently, V-Skip successfully eliminates spatial redundancy through attention skipping while preserving the critical semantic alignment and trajectory essential for complex multimodal understanding.

## 4.5.2 Impact of Attention Sparsity Budget N.

We investigate the sensitivity of model performance to the attention sparsity budget N, which defines the cardinality of the skipped layer subset $|\mathcal{L}_{skip}^{(k)}| = \mathrm{N}$ . By progressively increasing N from 0 to 32, we evaluate the trade-off between computational efficiency and reasoning accuracy on LLaVA-1.5-7B across all benchmarks. Following our evaluation protocol, we report the average retention percentage relative to the vanilla dense baseline (N = 0). As illustrated in Table 5, V-Skip demonstrates remarkable robustness within a broad near-lossless regime. Specifically, setting the budget N between 4 and 19 yields an average retention of at least 99.85%. Notably, the configuration with N=19 even slightly surpasses the baseline performance, achieving a peak retention of 100.31%. This validates our core hypothesis that a substantial portion of visual self-attention in deeper layers is indeed computationally redundant and can be safely bypassed without degrading model capabilities. However, performance begins to decline when the budget exceeds the identified saturation threshold. Accuracy retention drops to 99.24% at N=21 and 98.76% at N=24. Aggressively skipping the attention modules in all 32 layers (N=L) results in a significant degradation to 91.08%, particularly on fine-grained tasks such as MME, GQA, and OCRBench. This suggests that excessive skipping eventually removes critical visual features required for localized recognition and complex reasoning. Based on this analysis, we select N=19 (representing approximately 60% of the total layers) as our default configuration to achieve an optimal balance between architectural sparsity and preservation of precision.

## 4.5.3 Impact of Identification Strategy.

To validate the VIG metric, we compare it to Random selection and Cosine Similarity (CosSim) strategies under an identical sparsity budget of N=19. The CosSim baseline is calculated directly from output logits. As shown in Figure 4, the red line indicates that VIG yields the most robust performance envelope and maintains consistent competitiveness across diverse benchmarks. While CosSim outperforms Random skipping, its performance fluctuates significantly and falls short on structure-sensitive tasks like OCRBench. Consequently, VIG delivers a superior trade-off across multi-task evaluations. By measuring redundancy via visual information divergence instead of the geometric angle of logits, VIG highly accurately identifies semantically inert visual attention saturation blocks. This approach ensures robust results in both general and fine-grained contexts.

## 4.5.4 Robustness of Few-shot Calibration.

To rigorously validate the stability of our task-aware calibration, we investigate its sensitivity to the random sampling of few-shot examples. Figure 5 illustrates the Visual Information Gap $(\mathcal{G}(l))$ across all transformer layers for nine diverse benchmarks. The solid red lines represent the mean $\mathcal{G}(l)$ averaged over multiple random seeds, while the shaded regions denote the variance. The results show that while early layers have sample-dependent variance, deep layers maintain near-zero variance across all random seeds and tasks. This firmly demonstrates that Visual Attention Saturation is an inherent, dataset-level invariant rather than an artifact of specific few-shot samples. Consequently, our highly efficient 20-shot calibration mechanism is statistically robust and insensitive to the randomness of the calibration subset, ensuring stable and reliable skip routing decisions for varying downstream tasks.

Table 4: Ablation on functional decoupling. Comparing V-Skip against FFN-Skip to demonstrate the necessity of preserving FFN transformations for semantic evolution.

<table><tr><td rowspan="2">Method</td><td colspan="2">DecoderLayer</td><td rowspan="2">MME</td><td rowspan="2">MMMU</td><td rowspan="2">MMB</td><td rowspan="2">GQA</td><td rowspan="2">POPE</td><td rowspan="2">SQA</td><td rowspan="2">AI2D</td><td rowspan="2">VizWiz</td><td rowspan="2">OCRB</td><td rowspan="2">Avg.</td></tr><tr><td>MHSA</td><td>FFN</td></tr><tr><td>V-Skip</td><td>√</td><td></td><td>1853.3</td><td>36.9</td><td>65.0</td><td>61.8</td><td>85.5</td><td>69.3</td><td>55.4</td><td>54.5</td><td>31.6</td><td>100.31%</td></tr><tr><td>FFN-Skip</td><td></td><td>√</td><td>1873.9</td><td>36.0</td><td>64.4</td><td>61.0</td><td>85.1</td><td>69.7</td><td>55.2</td><td>53.2</td><td>30.3</td><td>99.16%</td></tr><tr><td>ShortV</td><td>√</td><td>√</td><td>1842.2</td><td>36.2</td><td>64.8</td><td>60.8</td><td>84.4</td><td>68.3</td><td>54.2</td><td>50.6</td><td>29.5</td><td>97.73%</td></tr></table>

Table 5: Sensitivity to sparsity budget N. Performance across varying budgets on LLaVA-1.5-7B, highlighting the near-lossless regime and the optimal balance at N=19.

<table><tr><td>Budget(N)</td><td>MME</td><td>MMMU</td><td>MMB</td><td>GQA</td><td>POPE</td><td>SQA</td><td>AI2D</td><td>VizWiz</td><td>OCR</td><td>Avg.</td></tr><tr><td>0</td><td>1866.1</td><td>36.7</td><td>64.1</td><td>61.9</td><td>85.1</td><td>69.6</td><td>55.2</td><td>54.3</td><td>31.3</td><td>100.00%</td></tr><tr><td>4</td><td>1858.4</td><td>36.4</td><td>64.1</td><td>61.9</td><td>85.2</td><td>69.6</td><td>55.2</td><td>54.0</td><td>31.4</td><td>99.85%</td></tr><tr><td>8</td><td>1849.8</td><td>36.7</td><td>64.2</td><td>61.8</td><td>85.2</td><td>69.3</td><td>55.2</td><td>54.1</td><td>31.6</td><td>99.93%</td></tr><tr><td>16</td><td>1844.7</td><td>36.7</td><td>64.3</td><td>61.8</td><td>85.3</td><td>69.3</td><td>55.4</td><td>54.1</td><td>31.6</td><td>99.97%</td></tr><tr><td>19</td><td>1853.3</td><td>36.9</td><td>65.0</td><td>61.8</td><td>85.5</td><td>69.3</td><td>55.4</td><td>54.5</td><td>31.6</td><td>100.31%</td></tr><tr><td>21</td><td>1834.7</td><td>36.7</td><td>64.6</td><td>61.6</td><td>85.3</td><td>69.4</td><td>55.1</td><td>53.9</td><td>29.9</td><td>99.24%</td></tr><tr><td>24</td><td>1807.4</td><td>36.2</td><td>64.7</td><td>61.3</td><td>85.1</td><td>69.1</td><td>55.0</td><td>54.6</td><td>29.4</td><td>98.76%</td></tr><tr><td>32</td><td>1560.1</td><td>35.2</td><td>60.7</td><td>54.5</td><td>78.7</td><td>68.8</td><td>52.2</td><td>54.8</td><td>22.1</td><td>91.08%</td></tr></table>

![](images/6cd8f3af6019ee4f497a61c2bb6106bdff3120232b26aaf452ee0af3b407ac5e.jpg)

<details>
<summary>radar chart</summary>

|        | Random | CosSim | VIG (Ours) |
| ------ | ------ | ------ | ---------- |
| MMB    | 95%    | 95%    | 100%       |
| MMMU   | 95%    | 95%    | 100%       |
| MME    | 95%    | 95%    | 100%       |
| OCRBench | 95%    | 95%    | 100%       |
| VizWiz | 95%    | 95%    | 100%       |
| AI2D   | 95%    | 95%    | 100%       |
| SQA    | 95%    | 95%    | 100%       |
| POPE   | 95%    | 95%    | 100%       |
| GQA    | 95%    | 95%    | 100%       |
</details>

Figure 4: Performance retention of different identification strategies.  
![](images/80fe104305f47571e13b6ab8966b5b0f77b936d627177c2f99736947a228433a.jpg)

<details>
<summary>line chart</summary>

| Transformer Layer Index (I) | MME (×10⁻³) | GQA (×10⁻³) | AI2D (×10⁻³) |
| --------------------------- | ----------- | ----------- | ------------ |
| 0                           | ~0.5        | ~0.5        | ~0.5         |
| 4                           | ~0.8        | ~0.8        | ~0.8         |
| 8                           | ~1.5        | ~10         | ~10          |
| 12                          | ~0.5        | ~5          | ~5           |
| 16                          | ~0.2        | ~2          | ~2           |
| 20                          | ~0.1        | ~1          | ~1           |
| 24                          | ~0.05       | ~0.5        | ~0.5         |
| 28                          | ~0.02       | ~0.2        | ~0.2         |
</details>

![](images/03df0083bdfeb3ca0537524373f499b9747ca02cf4b3e91b118de5a7ef6cbbf6.jpg)

<details>
<summary>line chart</summary>

| Transformer Layer Index (l) | MMMU (×10⁻³) | POPE (×10⁻³) | VizWiz (×10⁻³) |
| --------------------------- | ------------ | ------------ | -------------- |
| 0                           | ~0           | ~0           | ~0             |
| 4                           | ~2.5         | ~0.5         | ~0.5           |
| 8                           | ~3.5         | ~1.5         | ~1.5           |
| 12                          | ~2.0         | ~2.5         | ~10            |
| 16                          | ~0.5         | ~0.5         | ~5             |
| 20                          | ~0.2         | ~0.2         | ~2             |
| 24                          | ~0.1         | ~0.1         | ~1             |
| 28                          | ~0.05        | ~0.05        | ~0.5           |
</details>

![](images/558644c689c095399d7c20fb2d90b1d857b9a940d80e01c5302ca51283346669.jpg)

<details>
<summary>line chart</summary>

| Transformer Layer Index (I) | MMB (×10⁻³) | SQA (×10⁻³) | OCRBench (×10⁻³) |
| --------------------------- | ----------- | ----------- | ---------------- |
| 0                           | ~0          | ~0          | ~0               |
| 4                           | ~5          | ~3          | ~2               |
| 8                           | ~10         | ~8          | ~4               |
| 12                          | ~5          | ~2          | ~1               |
| 16                          | ~2          | ~1          | ~0.5             |
| 20                          | ~1          | ~0.5        | ~0.2             |
| 24                          | ~0.5        | ~0.2        | ~0.1             |
| 28                          | ~0.2        | ~0.1        | ~0.05            |
</details>

Figure 5: Robustness analysis of the few-shot calibration mechanism. VIG $\mathcal{G}(l)$ exhibits consistent task-level trends across different data sampling seeds.

## 4.5.5 Sensitivity to Sub-task Distribution.

Building upon the seed-level stability, we further investigate whether the calibration process requires balanced data to cover complex, multi-task scenarios. Using the MME benchmark as a representative case study, which comprises 14 distinct sub-tasks, we compare our default global random sampling against a task-stratified sampling strategy. As illustrated in Figure 6, the stratified approach plateaus at a peak score of 1853.3 when utilizing 28 to 56 total samples (corresponding to $m \in \{2, 3, 4\}$ instances per sub-task). Remarkably, our global random sampling achieves this identical performance 1853.3 with a budget of only 20 samples. This confirms that for a general-purpose benchmark like MME, its diverse sub-tasks collectively share a stable macroscopic saturation profile. A minimal random subset easily captures this global average without requiring meticulous sub-task balancing.

However, this internal sub-task robustness does not imply universal transferability across fundamentally different domains. To demonstrate the strict necessity of macro-level, task-specific calibration, we evaluate cross-dataset transferability as shown in Figure 7. Applying the routing strategy calibrated on MME to the structurally similar MMBench yields a highly competitive score of 64.43 (vs. 65.0 native). In contrast, transferring it to OCRBench, a specialized, extreme domain featuring dense text and complex structural parsing, results in a noticeable performance drop (30.7 vs. 31.6 native). While the saturation depth is highly robust to micro-level sub-task variations within a general domain (making our 20-shot random sampling extremely efficient), it shifts drastically when facing a specialized domain gap. Consequently, an independent, ultra-lightweight calibration for each distinct downstream task is not just practical, but essential.

![](images/8fe2503b259e0d1d4b7c1fc38177c77bd305b04a08138a181fa88eca8967650b.jpg)

<details>
<summary>line chart</summary>

| Total Calibration Samples | Stratified Sampling | Random Sampling (Ours) |
| ------------------------- | ------------------- | ---------------------- |
| 10                        | 1841.3              | 1831.7                 |
| 20                        | 1853.3              | 1853.3                 |
| 40                        | 1853.3              | 1853.3                 |
| 56                        | 1853.3              | 1853.3                 |
| 70                        | 1841.3              | 1853.3                 |
</details>

Figure 6: Sensitivity to calibration sampling strategies.

<table><tr><td>Target</td><td>MMBench</td><td>OCRBench</td></tr><tr><td>Domain</td><td>General (Similar)</td><td>Text (Distant)</td></tr><tr><td>Native</td><td>65.00</td><td>31.60</td></tr><tr><td>Transfer</td><td>64.43</td><td>30.70</td></tr><tr><td> $\Delta$ </td><td> $\downarrow 0.57$ </td><td> $\downarrow 0.90$ </td></tr></table>

Figure 7: Cross-dataset transferability. “Native” denotes using the 20-shot strategy calibrated on the target benchmark, while “Transfer” applies the MME-calibrated strategy. The performance drop on the text-centric OCRBench highlights the necessity of task-specific calibration.

![](images/c29aa81f9afcbfba067e0ba727f0f38776aa6cc96e3f5834b16442281122fa06.jpg)

<details>
<summary>text_image</summary>

Are there any packets in the image?
LLaVA
No, there are no packets in the image. The image features a person skiing on a snowy slope.
V-Skip
No, there are no packets in the image. The image features a person skiing on a snowy slope.
What is written in the image?
LLaVA
The image reads "Western."
V-Skip
The image reads "Western."
Is the huge bus to the right of the car on the street?
LLaVA
Yes, the huge double-decker bus is to the right of the car on the street.
V-Skip
Yes, the huge bus is to the right of the car on the street.
What famous road is shown on the sign?
LLaVA
The famous road shown on the sign is Route 66.
V-Skip
The famous road shown on the sign is Route 66.
</details>

Figure 8: Qualitative comparison of V-Skip and the LLaVA-1.5-7B across activity recognition, spatial reasoning, and fine-grained OCR tasks.

## 4.6 Qualitative Analysis

We compare V-Skip with the LLaVA to evaluate qualitative performance across diverse visual reasoning tasks. As illustrated in Figure 8, V-Skip maintains robust capabilities in object recognition, spatial reasoning, and fine-grained OCR. In the skiing scenario, it accurately identifies the activity while avoiding irrelevant object hallucinations, preserving visual integrity. In urban environments, it correctly perceives the spatial positioning of a double-decker bus relative to other vehicles. Notably, V-Skip excels in text-centric tasks, precisely identifying signs such as Western and Route 66. These results confirm that by decoupling spatial interaction from semantic evolution through FFN preservation, V-Skip successfully skips approximately 60% of attention saturation modules while maintaining the original reasoning depth and perceptual capability of the dense model.

## 5 Conclusions

In this paper, we investigate the layer-wise visual processing mechanisms within MLLMs and identify the phenomenon of Visual Attention Saturation. We demonstrate that visual saturation dynamically varies across layers, showing that while spatial mixing completes early, semantic alignment via FFNs remains essential throughout the network. Based on this observation, we propose V-Skip, which is a training-free structural optimization paradigm that introduces block-wise sparsity through a robust few-shot calibration. By strategically decoupling interactions, V-Skip successfully skips approximately 60% of attention saturation modules while preserving the original perceptual capability of the model. Extensive experiments confirm that our approach maintains 98.42% to 100.31% performance retention on the LLaVA series and a robust 94.16% on Qwen, while effectively mitigating object hallucinations. Ultimately, this work advocates for a paradigm shift from aggressive token removal to refined structural decoupling, offering a more robust path for efficient and trustworthy MLLM inference.

## References

[1] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. ArXiv, abs/2304.08485, 2023.  
[2] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26286–26296, 2023.  
[3] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.  
[4] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, Huajie Tan, Chunyuan Li, Jing Yang, Jie Yu, Xiyao Wang, Bin Qin, Yumeng Wang, Zizhen Yan, Ziyong Feng, Ziwei Liu, Bo Li, and Jiankang Deng. Llava-onevision-1.5: Fully open framework for democratized multimodal training. CoRR, abs/2509.23661, 2025.  
[5] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, and et al. Deepseek-v3 technical report, 2025.  
[6] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Ming-Hsuan Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. CoRR, abs/2502.13923, 2025.  
[7] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, and et al. Qwen3 technical report, 2025.  
[8] Zhihang Lin, Mingbao Lin, Luxi Lin, and Rongrong Ji. Boosting multimodal large language models with visual tokens withdrawal for rapid inference. In Toby Walsh, Julie Shah, and Zico Kolter, editors, AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, pages 5334–5342. AAAI Press, 2025.  
[9] Qianhao Yuan, Qingyu Zhang, Yanjiang Liu, Jiawei Chen, Yaojie Lu, Hongyu Lin, Jia Zheng, Xianpei Han, and Le Sun. Shortv: Efficient multimodal large language models by freezing visual tokens in ineffective layers. CoRR, abs/2504.00502, 2025.  
[10] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, 2024.  
[11] Long Xing, Qidong Huang, Xiao wen Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, and Dahua Lin. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. Computer Vision and Pattern Recognition Conference, abs/2410.17247, 2025.  
[12] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. In International Conference on Machine Learning, 2025.  
[13] Qizhe Zhang, Aosong Cheng, Ming Lu, Renrui Zhang, Zhiyong Zhuo, Jiajun Cao, Shaobo Guo, Qi She, and Shanghang Zhang. Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms. arXiv preprint arXiv:2412.01818, 2025.  
[14] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.  
[15] Zhuoran Yu and Yong Jae Lee. How multimodal llms solve image tasks: A lens on visual grounding, task reasoning, and answer decoding. 2025.  
[16] Constantin Venhoff, Ashkan Khakzar, Sonia Joseph, Philip Torr, and Neel Nanda. How visual representations map to language feature space in multimodal llms. CoRR, abs/2506.11976, 2025.  
[17] OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023.  
[18] Gemini Team. Gemini: A family of highly capable multimodal models. CoRR, abs/2312.11805, 2023.  
[19] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.  
[20] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 19792–19802. Computer Vision Foundation / IEEE, 2025.  
[21] Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 5484–5495. Association for Computational Linguistics, 2021.  
[22] Kawin Ethayarajh. How contextual are contextualized word representations? comparing the geometry of bert, elmo, and GPT-2 embeddings. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pages 55–65. Association for Computational Linguistics, 2019.  
[23] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. MME: A comprehensive evaluation benchmark for multimodal large language models. CoRR, abs/2306.13394, 2023.  
[24] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol, editors, Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part VI, volume 15064 of Lecture Notes in Computer Science, pages 216–233. Springer, 2024.  
[25] Xiang Yue, Yuansheng Ni, Tianyu Zheng, Kai Zhang, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 9556–9567. IEEE, 2024.  
[26] Drew A. Hudson and Christopher D. Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 6700–6709. Computer Vision Foundation / IEEE, 2019.  
[27] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.  
[28] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 292–305. Association for Computational Linguistics, 2023.  
[29] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Min Joon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling, editors, Computer Vision - ECCV 2016 - 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part IV, volume 9908 of Lecture Notes in Computer Science, pages 235–251. Springer, 2016.  
[30] Jeffrey P. Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C. Miller, Robin Miller, Aubrey Tatarowicz, Brandyn White, Samuel White, and Tom Yeh. Vizwiz: nearly real-time answers to visual questions. In Ken Perlin, Mary Czerwinski, and Rob Miller, editors, Proceedings of the 23rd Annual ACM Symposium on User Interface Software and Technology, New York, NY, USA, October 3-6, 2010, pages 333–342. ACM, 2010.  
[31] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of OCR in large multimodal models. Sci. China Inf. Sci., 67(12), 2024.

## A More Implementation Details

## A.1 Detailed Experimental Environment.

To ensure complete reproducibility, all inferences and evaluations in this study were conducted on a server running Ubuntu 20.04.6 LTS. The system hardware configuration consists of an Intel(R) Xeon(R) Gold 5318Y CPU @ 2.10GHz, 256GB of system RAM, and NVIDIA GeForce RTX 3090 GPUs, each equipped with 24GB of VRAM.

The software environment is built upon Python 3.10 and the NVIDIA CUDA Toolkit 12.8. For core deep learning implementations and model loading, we utilize PyTorch 2.1.2 paired with torchvision 0.16.2, alongside transformers 4.37.2 and accelerate 1.12.0. Furthermore, to guarantee standardized, fair, and reproducible testing across all reported multimodal benchmarks, such as MME, MMBench, and OCRBench, we strictly employ the official lmms-eval framework and follow the default evaluation protocol, specifically utilizing version 0.3.0.

## B Additional Compute-Matched Analysis

We further examine whether V-Skip's accuracy-efficiency trade-off comes from a larger compute budget or from its structural choice to bypass only visual self-attention. The results in Tables 6 to 8 compare methods under matched FLOPs, expanded token-pruning baselines, and increasingly aggressive FFN-Skip settings. These analyses consistently indicate that V-Skip's advantage comes from preserving the FFN pathway while removing redundant visual attention computation.

Table 6: Iso-FLOPs comparison on LLaVA-1.5-7B at 7.69 TFLOPs.

<table><tr><td>Method</td><td>TFLOPs</td><td>#L</td><td>Avg.</td></tr><tr><td>Dense</td><td>8.54</td><td>0</td><td>100.00</td></tr><tr><td>ShortV (N=4)</td><td>7.69</td><td>4</td><td>99.92</td></tr><tr><td>VTW (K=28)</td><td>7.69</td><td>4</td><td>99.91</td></tr><tr><td>V-Skip (N=19)</td><td>7.69</td><td>19</td><td>100.31</td></tr></table>

Table 7: Expanded token-pruning baselines under representative configurations on LLaVA-1.5-7B.

<table><tr><td>Method</td><td>Avg.</td></tr><tr><td>Dense</td><td>100.00</td></tr><tr><td>ShortV</td><td>97.73</td></tr><tr><td>VTW</td><td>88.71</td></tr><tr><td>PyramidDrop</td><td>99.76</td></tr><tr><td>SparseVLM</td><td>99.88</td></tr><tr><td>V-Skip</td><td>100.31</td></tr></table>

Table 8: V-Skip vs. FFN-Skip across larger sparsity budgets on LLaVA-1.5-7B.

<table><tr><td>N</td><td>V-Skip</td><td>FFN-Skip</td><td>Gap</td></tr><tr><td>19</td><td>100.31</td><td>99.16</td><td>+1.15</td></tr><tr><td>28</td><td>98.68</td><td>92.14</td><td>+6.54</td></tr><tr><td>32</td><td>91.75</td><td>71.59</td><td>+20.16</td></tr></table>

Iso-FLOPs comparison. When ShortV and VTW are tuned to match V-Skip's 7.69 TFLOPs, V-Skip skips 19 layers while ShortV and VTW skip only four layers each, yet V-Skip still achieves the highest average retention. This suggests that the gain is not simply a matter of using a more permissive compute budget; rather, the attention-only bypass enables a substantially denser sparsity pattern without disrupting the semantic transformation path.

Expanded token-pruning baselines. Compared with representative token-pruning methods, including PyramidDrop and SparseVLM, V-Skip is the only method in this comparison that matches or slightly exceeds the dense baseline average. The contrast is especially pronounced on grounding-sensitive settings, where destructive token withdrawal can remove fine-grained visual evidence. For example, OCRBench retention is 31.6 for V-Skip compared with 5.1 for VTW in the main comparison.

![](images/28b911f6ddd12e485a56e97cbf3bfa0b6f3a1d0f39bc5be3a1d978b1c4f83dfb.jpg)

<details>
<summary>heatmap</summary>

| MMMU |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MMBench |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| GQA |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| POPE |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
</details>

(a) Sparsity Budget N = 5  
![](images/e18bbb14467c899815952efdd0866ad1cfbd0f05ad4ad664cc7f464893e1b427.jpg)

(b) Sparsity Budget N = 10  
![](images/30f189c90665fe1f748169edebdd8c1e07f333b429d83500551af330e0d8e6fb.jpg)

<details>
<summary>heatmap</summary>

| MMBench |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 19 |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GQA |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 19 |  |  |  |
</details>

(c) Sparsity Budget N = 15

![](images/7b559b36bbea322380a6943fb2e9e31d30054d92c0af948c0c6195f62d304ddc.jpg)

<details>
<summary>heatmap</summary>

| Category | Transformer Layer Index (l) |
|---|---|
| MME | 12 |
| MMMU | 16 |
| MMBench | 13 |
| GQA | 15 |
| POPE | 12 |
| ScienceQA | 14 |
| ai2d | 17 |
| VizWiz | 19 |
| OCRBench | 2 |
</details>

(d) Sparsity Budget N = 20  
Figure 9: Evolution of task-specific skipped layers across different sparsity budgets ( $N \in \{5, 10, 15, 20\}$ ). Blue cells indicate active layers (High VIG), while gray cells represent saturated layers dynamically selected for skipping. As the budget increases, the skipping pattern consistently propagates from the deepest layers toward the middle, strictly preserving the early spatial aggregation layers.

FFN preservation under larger sparsity budgets. The gap between V-Skip and FFN-Skip is modest at N=19 because this is a conservative sparsity regime. As the sparsity budget increases, the gap widens to 6.54 points at N=28 and 20.16 points at N=32. At N=32, FFN-Skip drops sharply on GQA (61.9→39.6) and POPE (85.1→61.5), while V-Skip retains 54.5 and 78.7. These results support the view that FFNs remain important for cross-modal projection even when deep visual self-attention is redundant.

Latency and deployment note. The layer-aligned latency comparison in the main paper isolates what is bypassed inside each layer: V-Skip preserves FFNs, while ShortV bypasses entire blocks. V-Skip therefore gives a more moderate standalone latency reduction of approximately 8%, but it preserves accuracy more reliably and can be combined with token pruning for stronger acceleration, as shown by FastV+V-Skip. In a deployment-oriented throughput measurement, V-Skip improves throughput from 14.4 to 14.8 while keeping the KV-cache footprint unchanged, consistent with its role as a prefill-compute accelerator.

Table 9: Calibration time overhead across different multimodal benchmarks. The recorded time represents the total duration (mm:ss) required to compute the VIG and identify the optimal block-wise skipping strategy using our 20-shot subset on a single NVIDIA 3090 GPU.

<table><tr><td>Dataset</td><td>Calibration Time</td></tr><tr><td>MME</td><td>01:55</td></tr><tr><td>MMMU</td><td>02:06</td></tr><tr><td>MMBench</td><td>02:15</td></tr><tr><td>GQA</td><td>01:54</td></tr><tr><td>POPE</td><td>01:42</td></tr><tr><td>SQA</td><td>02:10</td></tr><tr><td>AI2D</td><td>02:03</td></tr><tr><td>VizWiz</td><td>02:03</td></tr><tr><td>OCRBench</td><td>01:39</td></tr></table>

## C Budget Evolution and Task-Specific Sparsity

To comprehensively understand the dynamics of our task-aware calibration mechanism, we visualize the evolution of active and skipped vision attention across increasing sparsity budgets ( $N \in \{5, 10, 15, 20\}$ ) for nine representative benchmarks Figure 9. This visualization yields two critical architectural insights regarding how MLLMs process visual information under varying Visual Information Gain constraints.

## C.1 Dynamic Deep-to-Shallow Progression.

Analyzing the evolution across different budgets demonstrates that our few-shot calibration is a highly dynamic ranking process. Under a conservative budget (N = 5), the mechanism universally prioritizes the deepest layers (e.g., layers 20–31), confirming that visual attention redundancy initially emerges at the terminal stages of semantic alignments. As the sparsity budget expands to N = 20, the skipped saturated layers propagate a trend toward the middle or early layers. Crucially, the early transformer blocks (layer 1–11) remain overwhelmingly active across all configurations, acting as a foundation for spatial feature interaction. This consistent evolution trajectory proves that VIG metric provides a way for dynamically scaling block-wise redundancy rather than applying a fixed, predefined pruning template.

Takeaway: Deep layers are identified as the primary source of visual attention saturation, whereas early layers act as an indispensable structural cornerstone that V-Skip autonomously protects.

## C.2 Task-Driven Routing Consistency.

Observing the permutation of skipped visual attention saturation layers uncovers a correlation between task characteristics and the evolution of visual saturation. Benchmarks with similar reasoning and visual perception demand naturally cluster into distinct evolutionary patterns at higher sparsity budgets (e.g., N = 20). General multimodal evaluation, such as MME and MMBench, exhibit remarkably similar structures aggressively bypassing deep blocks and dropping the initial projection layer. In contrast, structure sensitive tasks like OCRBench share a fundamentally different pattern. They strictly preserve layer 0 and selectively retain specific intermediate layers to interact with visual cues.

Takeaway: V-Skip acts as a domain-aware router. It empirically proves that a “one-fits-all” pruning mask is suboptimal, as fine-grained reasoning inherently requires distinct semantic pathways compared to general perception.

## C.3 Calibration Efficiency and Overhead.

It is crucial that task-specific calibration pre-computation does not introduce prohibitive delays while guaranteeing optimal routing. To quantify this overhead, we measure the wall-clock time required to compute the layer-wise VIG scores and extract the Top-N skipping configuration. All calibrations utilize our default 20-shot sampling strategy and are executed on a single NVIDIA 3090 GPU. As reported in Table 9, the entire calibration process is remarkably lightweight. For the majority of downstream tasks, it completes in approximately two minutes (e.g., 01:39 for OCRBench and

01:55 for MME). Given that this is a strict one-time offline cost incurred prior to inference, the overhead is practically negligible. When weighed against the substantial inference acceleration and robust performance retention that V-Skip delivers throughout the model's deployment lifecycle, this minimal pre-computation firmly establishes our method as a practical, plug-and-play solution.

## D Further Discussion on Functional Saturation

Both VIG and ShortV use KL divergence, but they perturb different computational targets and therefore measure different phenomena. ShortV perturbs entire transformer blocks, jointly removing attention and FFN computation. This produces a single whole-block sensitivity signal that conflates spatial interaction and semantic projection. In contrast, VIG isolates the visual-to-visual self-attention pathway while preserving FFN transformations. This targeted perturbation exposes a modality-specific functional asymmetry: visual self-attention can saturate earlier than the FFN pathway, while FFNs remain important for aligning visual tokens with the evolving language-model semantic space.

This distinction also leads to different interventions. ShortV freezes both attention and FFN updates in skipped layers, whereas V-Skip bypasses only saturated visual attention and keeps FFNs active for all tokens. The comparison in Table 8 shows that bypassing FFNs becomes increasingly harmful as the sparsity budget grows, which would be hidden by a whole-block scoring rule. VIG is therefore best understood as a functional sensitivity measure: it quantifies how much the final predictive distribution changes when visual self-attention at a layer is bypassed. It does not directly claim to measure internal representation saturation; instead, it identifies whether a layer's visual attention computation is functionally necessary for downstream predictions.

The LLaVA-NeXT-7B MME result further illustrates this functional view. V-Skip drops from 1846.3 to 1778.6 on MME, a 3.7% reduction and the largest single-benchmark drop in the reported experiments. This indicates that high-resolution settings can retain a stronger dependence on deep-layer visual interaction for some tasks, even though the average retention across nine benchmarks remains 98.42%.

Finally, lmms-eval is deterministic under the reported protocol, so the observed gaps primarily reflect structural differences rather than sampling noise. For this reason, claims around V-Skip should be interpreted as evidence that attention/FFN decoupling preserves visual grounding and avoids exacerbating hallucination under the evaluated settings, rather than as a universal hallucination-mitigation guarantee.

## E Exploring Universal Calibration for V-Skip

Our proposed V-Skip employs task-specific calibration, identifying the optimal saturated layers for each individual dataset to maximize performance retention. To further investigate the transferability and robustness of the visual attention saturation phenomenon, we conduct an additional ablation study evaluating a "universal" layer skipping strategy. Specifically, we introduce V-Skip\*, which derives its layer-skipping configuration purely calibrated on the MME benchmark and applies this exact same strategy globally across all other evaluation datasets without any task-specific tuning.

As presented in Table 10, the universally calibrated V-Skip\* demonstrates strong generalization, achieving an impressive average performance retention of $99.55\%$ across all benchmarks. It consistently outperforms other training-free acceleration baselines across various benchmarks, significantly surpassing ShortV $(97.73\%)$ and VTW $(88.71\%)$ . This clearly highlights the inherent architectural superiority of our attention-bypassing and FFN-preserving paradigm compared to destructive token dropping or complete layer freezing.

However, we also observe a distinct performance gap between the universal V-Skip\* (99.55%) and our default, task-specifically calibrated V-Skip (100.31%). The degradation is more noticeable in tasks requiring fine-grained visual grounding, such as OCRBench (dropping from 31.6 to 30.7) and MMBench (dropping from 65.0 to 64.4). While a universal configuration is highly competitive and adaptable, diverse multimodal reasoning tasks inherently demand different depths of spatial attention. Therefore, to achieve optimal performance and fully preserve the model's reasoning integrity, a fine-grained calibration tailored to the specific data distribution of the target task remains essential. Nonetheless, the strong baseline established by V-Skip\* demonstrates its viability as a highly practical, out-of-the-box acceleration solution for real-world deployments where task-agnostic efficiency is prioritized.

## F Qwen2.5-VL Re-Calibration Analysis

The Qwen2.5-VL results show that the optimal sparsity budget is architecture-dependent. The original $N = 7$ setting follows the approximate skip ratio used for LLaVA-style architectures, but it is relatively aggressive for Qwen2.5-VL.

Table 10: Performance comparison of universal layer skipping. V-Skip\* applies the MME-calibrated layer skipping strategy globally to all other benchmarks, whereas V-Skip uses task-specific calibration.

<table><tr><td>Method</td><td>MME</td><td>MMMU</td><td>MMB</td><td>GQA</td><td>POPE</td><td>SQA</td><td>AI2D</td><td>VizWiz</td><td>OCRB</td><td>Avg.</td></tr><tr><td>LLaVA-1.5-7B</td><td>1866.1</td><td>36.7</td><td>64.1</td><td>61.9</td><td>85.1</td><td>69.6</td><td>55.2</td><td>54.3</td><td>31.3</td><td>100.00%</td></tr><tr><td>ShortV (N=19)</td><td>1842.2</td><td>36.2</td><td>64.8</td><td>60.8</td><td>84.4</td><td>68.3</td><td>54.2</td><td>50.6</td><td>29.5</td><td>97.73%</td></tr><tr><td>VTW (K=16)</td><td>1857.6</td><td>36.4</td><td>64.0</td><td>55.1</td><td>85.3</td><td>69.6</td><td>55.4</td><td>51.0</td><td>5.1</td><td>88.71%</td></tr><tr><td>V-Skip* (N=19)</td><td>1853.3</td><td>36.6</td><td>64.4</td><td>61.4</td><td>85.2</td><td>69.2</td><td>55.4</td><td>53.9</td><td>30.7</td><td>99.55%</td></tr><tr><td>V-Skip (N=19)</td><td>1853.3</td><td>36.9</td><td>65.0</td><td>61.8</td><td>85.5</td><td>69.3</td><td>55.4</td><td>54.5</td><td>31.6</td><td>100.31%</td></tr></table>

Table 11: Qwen2.5-VL-7B re-calibration. Reducing the sparsity budget from the submitted N=7 configuration to N=4 better matches Qwen2.5-VL's VIG profile and recovers average retention to 98.72%.

<table><tr><td>Config</td><td>MMMU</td><td>GQA</td><td>VizWiz</td><td>Avg. Retention</td></tr><tr><td>Dense</td><td>50.4</td><td>60.5</td><td>70.2</td><td>100.00%</td></tr><tr><td>N=4</td><td>46.7</td><td>58.3</td><td>69.2</td><td>98.72%</td></tr><tr><td>N=5</td><td>46.9</td><td>54.7</td><td>67.1</td><td>96.67%</td></tr><tr><td>N=6</td><td>46.0</td><td>52.2</td><td>67.3</td><td>95.27%</td></tr><tr><td>N=7</td><td>43.3</td><td>51.3</td><td>66.4</td><td>94.16%</td></tr><tr><td>N=8</td><td>41.9</td><td>48.0</td><td>65.8</td><td>92.07%</td></tr></table>

Unlike LLaVA, whose VIG profile shows a clearer deep-layer saturation plateau, Qwen2.5-VL exhibits persistently higher VIG across depth, likely because window attention and mRoPE redistribute visual redundancy. Re-calibrating the sparsity budget accordingly substantially recovers performance.

Moving from $N = 7$ to $N = 4$ improves GQA by 7.0 points and MMMU by 3.4 points, raising Qwen2.5-VL's overall retention from $94.16\%$ to $98.72\%$ . This aligns Qwen2.5-VL with the near-lossless regime observed for LLaVA-NeXT and supports the use of architecture-specific calibration rather than a fixed global skip ratio.

## G Limitations and Future Work

While V-Skip demonstrates significant efficiency gains and maintains high performance across various multimodal benchmarks, we acknowledge certain limitations. A core advantage of V-Skip is its training-free, plug-and-play nature. However, because the original model weights were optimized under dense attention assumptions, bypassing these paths at inference time represents a structural shift from the initial pre-training phase. Consequently, our current approach relies on a lightweight few-shot calibration mechanism to dynamically identify the task-optimal sparsity path, accounting for the varying depths of spatial reasoning required by diverse downstream tasks.

To achieve a more universal, task-agnostic generalization—where the network natively learns to route or bypass visual computation without prior empirical calibration—future work will focus on incorporating the V-Skip mechanism directly into the model's training pipeline. Integrating this block-wise skipping during multimodal pre-training or instruction tuning would allow the model to natively internalize the Visual Attention Saturation phenomenon. This co-design of architecture and training could eliminate the reliance on task-specific calibration, potentially unlocking even greater synergies between performance and computational efficiency.

## H Additional Visualizations

To further illustrate the practical advantages of V-Skip and provide a deeper understanding of how different acceleration paradigms impact multimodal reasoning, we provide extended qualitative comparisons and a Qwen2.5-VL VIG visualization. Figures 11 and 12 present challenging cases from the GQA and OCRBench datasets, respectively. Figure 10 visualizes the Qwen2.5-VL VIG profile discussed in Section F.

## H.1 Mitigating Spatial Misalignment and Hallucinations

Figure 11 highlights tasks requiring precise spatial awareness and fine-grained object grounding within complex indoor and outdoor scenes. Aggressive token reduction methods like VTW tend to lose critical localized features, resulting in generic or less precise answers, such as predicting “Computer” instead of “Keyboard”, or “Man” instead of “Policeman”. Conversely, layer-skipping methods like ShortV, which completely bypass the FFNs, suffer from severe feature misalignment. This semantic disruption leads to obvious spatial errors and object hallucinations. V-Skip preserves FFN transformations for continuous semantic evolution, enabling accurate grounding of fine-grained objects and complex spatial relations.

![](images/3f91b75ab13a74d47c4c9ce16b6d3653cbe94f5e3787a197b43fa39510e1873a.jpg)

<details>
<summary>line chart</summary>

| Transformer Layer Index (l) | MME     | GQA     | AI2D    | MMMU    | POPE    | VizWiz  | MMBench | ScienceQA | OCRBench |
| --------------------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | --------- | -------- |
| 0                           | 10.0    | 10.0    | 10.0    | 10.0    | 10.0    | 10.0    | 10.0    | 10.0      | 10.0     |
| 2                           | 0.3     | 0.1     | 0.05    | 0.1     | 0.001   | 0.1     | 0.01    | 0.05      | 0.05     |
| 4                           | 0.1     | 0.1     | 0.05    | 0.1     | 0.01    | 0.1     | 0.01    | 0.05      | 0.1      |
| 6                           | 0.1     | 0.1     | 0.05    | 0.1     | 0.01    | 0.1     | 0.01    | 0.05      | 0.1      |
| 8                           | 0.1     | 0.1     | 0.05    | 0.1     | 0.01    | 0.1     | 0.01    | 0.05      | 0.1      |
| 10                          | 0.1     | 0.1     | 0.05    | 0.1     | 0.01    | 0.1     | 0.01    | 0.05      | 0.1      |
| 12                          | 0.1     | 0.1     | 0.05    | 0.1     | 0.01    | 0.1     | 0.01    | 0.05      | 0.1      |
| 14                          | 0.1     | 0.1     | 0.05    | 0.1     | 0.01    | 0.1     | 0.01    | 0.05      | 0.1      |
| 16                          | 0.1     | 0.1     | 0.05    | 0.1     | 0.01    | 0.1     | 0.01    | 0.05      | 0.1      |
| 18                          | 0.1     | 0.1     | 0.05    | 0.1     | 0.01    | 0.1     | 0.01    | 0.05      | 0.1      |
| 20                          | 0.1     | 0.1     | 0.05    | 0.1     | 0.01    | 0.1     | 0.01    | 0.05      | 0.1      |
| 22                          | 0.1     | 0.1     | 0.05    | 0.1     | 0.01    | 0.1     | 0.01    | 0.05      | 0.1      |
| 24                          | 0.1     | 0.1     | 0.05    | 0.1     | 0.01    | 0.1     | 0.01    | 0.05      | 0.1      |
| 26                          | 0.1     | 0.1     | 0.05    | 0.1     | 0.01    | 0.1     | 0.01    | 0.05      | 0.1      |
| 28                          | 0.1     | 0.1     | 0.05    | 0.1     | 0.01    | 0.1     | 0.01    | 0.05      | 0.1      |
</details>

Figure 10: Qwen2.5-VL VIG profile. Compared with the clearer deep-layer saturation observed in LLaVA-style models, Qwen2.5-VL maintains higher VIG across more layers, motivating the architecture-specific recalibration in Table 11.

## H.2 Preserving Structural Integrity for OCR Tasks

Recognizing text in the wild requires high-resolution preservation and structural integrity. As shown in Figure 12, VTW tends to destroy textual feature sequences and can produce generic placeholders such as “Words” or “text”. ShortV lacks the FFN projections needed to map complex visual text patterns into the language-model vocabulary space, producing spelling hallucinations. By bypassing only redundant spatial mixing while keeping point-wise semantic projections active, V-Skip maintains strong perceptual fidelity on cursive fonts, stylized text, and numerical sequences.

## H.3 Qwen2.5-VL VIG Profile

Figure 10 shows that Qwen2.5-VL does not exhibit the same clear deep-layer saturation plateau as LLaVA-style architectures. Its VIG remains relatively high across a broader portion of the depth, explaining why a smaller sparsity budget is more appropriate for this architecture.

![](images/06315f8e258d29f01baaaf4f6adc1befdfb010ebef6d2a5b1da7b7457c2998b4.jpg)

<details>
<summary>text_image</summary>

LLaVA
VTW
ShortV
V-Skip
keyboard
Computer
Computer
Keyboard
What device is sitting next
to the mouse pad?
Hat
Cap
Cap
Hat
Which kind of clothing is
not pink?
Right
Left
Left
Right
On which side of the picture
is the chair?
Cars
Car
Car
Cars
What are the vehicles above
the road near the side walk?
Policeman
Man
Man
Policeman
Who is wearing a helmet?
Chair
Fireplace
Wall
Chair
What item of furniture is
the toilet paper to the right
of the toilet resting on?
</details>

Figure 11: Qualitative comparison on spatial reasoning and object grounding. V-Skip preserves fine-grained visual details and complex spatial relations, while destructive token withdrawal and full-layer skipping can lead to generic outputs, spatial reasoning errors, and object hallucinations.

![](images/61b0310bd6f149925a263639fd1b8937ff95bacae7a2de94a76ed20fb7af2249.jpg)

<details>
<summary>text_image</summary>

LLaVA
VTW
ShortV
V-Skip
Antonios
Words
Anthonios
Antonios
What is written in the image?
Compensated
Words
concerned
CONDENSED
grimly
Grimly
written
GRIMMY
Grimly
What is written in the image?
Medium
text
mediam
Medium
What is written in the image?
see
Words
see
SEEM
5605
133
566566
5605
What is written in the image?
Nabak
Nakk
text
NABK
What is written in the image?
</details>

Figure 12: Qualitative comparison on OCR samples. Recognizing stylized, cursive, and handwritten text requires structural integrity. V-Skip preserves fine-grained textual features more reliably than destructive token withdrawal or full-layer skipping.