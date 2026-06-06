# Cosine Misleads: Auxiliary Losses Reshape Vision Language Models, Not Their Latents

XiuYu Zhang and Junfeng Fang and Zhenkai Liang

National University of Singapore

# Abstract

Latent visual reasoning (LVR) inserts supervised latent tokens between perception and answer generation in vision-language models (VLMs). The field uses alignment between these latents and their visual targets, i.e., cosine similarity or mean squared error (MSE), as both the training loss and the quality metric, assuming that better alignment yields a better answer. We test this with a designed matrix of five LVR variants and find the assumption inverted: cosine alignment is negatively correlated with accuracy across all five (r=-0.94). To explain this, we introduce PRISM, a pair of inference-time diagnostics: a linear probe that asks where the answer is decodable, and a corruption test that asks whether the latent is load-bearing. The supervised latents are largely bypassed. Corrupting them shifts accuracy by at most four points. The answer is decodable downstream of the latent but not at it, and the size of this decodability gap predicts how much each variant relies on its latent under perturbation. Consistent with an Information Bottleneck reading of the loss, the auxiliary objective reshapes the language model via shared parameters rather than via the latent variable it nominally optimizes.

# 1 Introduction

Latent visual reasoning (LVR) inserts continuousvalue latent tokens between visual perception and answer generation in a vision-language model (VLM) (Bordes et al., 2024). These latents are supervised against teacher-forced visual targets during training and fed back autoregressively at inference. In this active research area (Li et al., 2026; Yang et al., 2025; Dong et al., 2026; Wang et al., 2025; Jeon et al., 2026), the alignment signals, e.g., mean square error (MSE), cosine similarity, and their variant, between the generated latents and target representations, have been used as both the training loss and the post-hoc quality metric. The implicit assumption behind this practice is that a better cosine means a more faithful latent, which implies a better answer.

Cosine misleads: r = 0.94 across variants   
![](images/2250e598e3052dd202b7d0d19b876e6bd3f7956537790314154fb4e3b15f659a.jpg)

<details>
<summary>scatter</summary>

| Cosine alignment to teacher target v_t | V*Bench accuracy (%) | Method   |
| -------------------------------------- | --------------------- | -------- |
| 0.45                                   | 70.0                  | D-LVR    |
| 0.55                                   | 71.0                  | N-LVR    |
| 0.78                                   | 58.0                  | P-LVR-2  |
| 0.78                                   | 58.0                  | P-LVR-3  |
</details>

Figure 1: Cosine misleads. Cosine alignment between LVR-position hidden states and their teacher-forced visual targets is negatively correlated with V∗Bench accuracy across all five trained variants (Pearson r=−0.94). Progressive variants (P-LVR-2, P-LVR-3) reach the highest cosine but the lowest accuracy.

To verify what cosine alignment actually indicates, we designed five LVR variants that vary in how the latents are trained (from reconstructionpure to input-noise-regulated to progressively scaffolded) and trained each on a shared backbone, data, and step budget. Surprisingly, cosine similarity is negatively correlated with V∗Bench (Wu and Xie, 2024) accuracy across all five (Pearson r = −0.94). A progressive scaffolding variant raises cosine alignment by 40% over the baseline and lowers reconstruction error from 3.79 to 1.55, but loses 13 V∗Bench points. This decrease replicates on BLINK (Fu et al., 2024) and MMVP (Tong et al., 2024). A noise-train variant matches the baseline’s cosine alignment to three decimals (0.556 vs. 0.555), but gains 1.5 V∗Bench points and responds in the opposite direction when its latents are zeroed during inference, i.e., its own latent helps while the baseline’s hurts. Cosine similarity is thus shown to be a misleading signal in isolation.

To investigate, we apply two diagnostics that we collectively call PRISM: a corruption test on the latents and a linear probe of the answer-decoding hidden state. First, the corruption test asks whether the model actually uses the latents at inference. We intervened in the LVR positions at each generation step by truncating them to zero, perturbing them with Gaussian noise, or replacing them with latents from another sample. Across all five variants, every intervention shifts V∗Bench by at most four points. For the worst variant, zeroing the latents improves accuracy. As a result, the LVR latents – the very tokens cosine optimizes – are largely bypassed at inference.

However, the same five variants differ by 13 V∗Bench points. If latents are bypassed, the training objective must do something else, not through the latents themselves directly. PRISM’s second diagnostic answers this. We fit a linear probe at two positions: the answer-decoding state (the hidden state the LM head reads to produce the answer), and the feedback variable (the latent the autoregressive loop re-injects). The answer is decodable from the answer-decoding state, which is expected. The interesting question is whether it is also decodable from the feedback variable: it is not. The signal sits downstream of the latent, rather than in the latent, where the loss is optimized. The contrast between probe accuracy at the two positions varies substantially across the matrix, and the size of this contrast predicts how each variant responds to latent perturbation.

Under an Information Bottleneck (Tishby et al., 2000, IB) reading, this can be explained as the dominating cross-entropy (CE) term in the training loss applies relevance pressure to whatever computation produces the answer, not to the supervised latents specifically. Therefore, the model is free to route the answer without using the latents under the current LVR loss construction. The training objective reshapes the language model through gradient flow into shared parameters, not through what the latent encodes. The auxiliary loss works, but not for the reason its name suggests.

We hypothesize this pattern is not specific to LVR. Auxiliary losses across multimodal learning supervise intermediate representations against external targets, and the assumption that the supervised representation is also the load-bearing one is undertested. PRISM operationalizes a test: the probe localizes where the answer is decodable; the corruption confirms whether the latents are loadbearing. Together, they make visible what cosine cannot see.

Our contributions can be summarized as follows:

1. A designed test of the LVR training assumption. Across a matrix of five variants spanning the design space for supervising latents, cosine ranks variants approximately backwards relative to accuracy, and the latents the loss optimizes are largely bypassed at inference.   
2. The training objective reshapes the VLM through shared parameters. Despite the bypass, variants differ substantially in accuracy. The answer-relevant signal is linearly decodable at the model’s answer-decoding state, but not in the latent loop re-injects. The size of this contrast predicts both task accuracy and latent reliance under perturbation.   
3. PRISM as a replacement diagnostic. Linear probes at two positions reveal a contrast that locates where the answer sits in the model, and a corruption test checks whether the supervised latent is load-bearing. The two are empirically connected as the size of the probe contrast predicts the corruption response.

# 2 Related Work

Latent visual reasoning and the alignment assumption. LVR (Li et al., 2026) generates latent tokens supervised to reconstruct visual embeddings from task-relevant regions defined by bounding boxes via MSE. Concurrent and subsequent works vary in their supervision to align latent representations and visual targets: Mirage (Yang et al., 2025) uses cosine as the loss, Monet (Wang et al., 2025) uses cosine to construct the loss, VaLR (Jeon et al., 2026) adopts the REPA (Leng et al., 2025) loss, which uses cosine to measure similarity. Across these works, teacher-feature alignment is treated as both a training signal and a post-hoc quality metric. This dual use is rarely tested directly or receives the attention it deserves. Inspired by this trend, we construct a matrix of LVR variants that span its design choices, serving as a testbed to directly test the alignment assumption.

Probing and faithfulness. Linear probing trains a single linear classifier from frozen hidden states to predict a target label (Alain and Bengio, 2018). It is used to study and understand the internal states of neural networks. Subsequent work showed that probe accuracy can reflect probe capability rather than property presence (Hewitt and Liang, 2019; Belinkov, 2022), which motivates control-task selectivity. Faithfulness work in chain-of-thought (CoT) reasoning in large language models (LLMs) shows that explanations can be unfaithful to the model’s actual computation (Turpin et al., 2023), so causal reliance is better tested by intervening on the reasoning traces (Tutek et al., 2025). Drawing lessons from these works, we applied the two ideas in PRISM to analyze a latent-reasoning setting. We use a probe to ask whether the answer is decodable from a given hidden state, and a corruption test to ask whether that state is causally load-bearing.

Information bottleneck. The Information Bottleneck (Tishby et al., 2000, IB) frames representation learning as a trade-off between compressing input and preserving information about a target. The variational form (Alemi et al., 2017) makes this tractable. We use IB as interpretive scaffolding for our results, since cosine and MSE optimize only one side of the IB objectives, leaving the other to implicit pressure from the next-token prediction loss. This may be one way to explain why the cross-variant correlation between cosine and accuracy is free to run in any direction. We use IB as interpretive scaffolding and do not claim IB or its usage as a contribution.

# 3 The Information Bottleneck view of LVR

This section introduces the lens through which we interpret the empirically observed cosine dissociations in Section 5. We frame the LVR latents as a representation in the Information Bottleneck (IB) sense and argue that the reconstruction-based loss (such as MSE and cosine) the field uses does not bound either side of the IB objective.

# 3.1 The LVR loss

Let $\mathbf { h } _ { t }$ be the LVR-position hidden state at iteration $t ,$ and $\mathbf { v } _ { t }$ be the teacher-forced visual target for that position. LVR (Li et al., 2026) supervises the latents with a reconstruction loss alongside the typical next-token prediction loss:

$$
\mathcal {L} _ {\mathrm{LVR}} = \mathcal {L} _ {\mathrm{CE}} + \lambda \cdot \operatorname{MSE} (\mathbf {h} _ {t}, \mathbf {v} _ {t}). \tag {1}
$$

Cosine-based variants (Yang et al., 2025; Wang et al., 2025; Jeon et al., 2026) replace the MSE term with cosine or patch-wise alignment losses or mix with other losses. The form differs, but the role is the same, i.e., pull $\mathbf { h } _ { t }$ towards $\mathbf { v } _ { t } .$ . In inference, the model produces its own latents and feeds them back autoregressively.

# 3.2 The IB objective

Let X be the input (image, question, prefill context) and Y be the answer. The IB writes the latentquality trade-off as a Lagrangian:

$$
\mathcal {L} _ {\mathrm{IB}} = \beta I (X; Z) - I (Z; Y), \tag {2}
$$

where Z is the intermediate representation, $I ( \cdot ; \cdot )$ is mutual information, and $\beta > 0$ controls compression. An IB-optimal Z is compressed (small $I ( X ; Z ) )$ while preserving information about the answer (large $I ( Z ; Y ) ) ,$ ).

Three intermediate representations are used in our LVR setting. We will use $Z _ { c h }$ for the LVRposition hidden state itself, i.e., the object the loss is computed on, and the IB objectives are about. We will use $Z _ { f b }$ for the feedback variable re-injected into the VLM at the next position. Without a projection, $Z _ { c h } = Z _ { f b }$ . We will use $H _ { a n s }$ to denote the answer-decoding hidden state read by the LM head at the first ordinary text token. Section 5 probes $Z _ { f b }$ and $H _ { a n s }$ separately following PRISM.

# 3.3 The LVR loss is an indirect IB objective, and cosine measures only one side of it

The LVR loss (Equation 1) has two terms, and each does part of what IB asks for. The MSE term serves as a proxy for compression. $\mathbf { v } _ { t }$ is the visual embedding of an annotated region, i.e., a hand-selected, low-rate, task-relevant representation of X. Driving $\mathbf { M S E } ( \mathbf { h } _ { t } , \mathbf { v } _ { t } )$ towards zero pulls $Z _ { c h }$ to match $\mathbf { v } _ { t }$ in every dimension. In the limit, $Z _ { c h } = \mathbf { v } _ { t }$ and thus $I ( X ; Z _ { c h } )$ inherits $\mathbf { v } _ { t }$ ’s small rate. The loss does not directly bound $I ( X ; Z )$ , but it implements compression indirectly by anchoring $Z _ { c h }$ to a target whose rate is low by construction. Cosine alignment is a looser version of the same proxy: it pins direction but not magnitude or orthogonal components, so the inheritance chain is weaker, but the basic mechanism remains the same.

The cross-entropy (CE) term supplies relevance, but through a different route. CE on Y applies pressure on whatever computation produces the answer. Since $Z _ { c h }$ sits inside that computation path during training, CE gradients flow back through it and shape its content. Critically, the pressure is on the system as a whole, not on $Z _ { c h }$ in particular. The model is free to satisfy CE by making Y depend on whichever parts of its state are convenient, and nothing in the loss requires that $Z _ { c h }$ be among those parts.

![](images/ad332706920dc61b4a98fac994aa1efebf641174fddecc003af39339e5bcf114.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["(a) decode state\nwhat the LM head read"] --> B["P"]
    B --> C["z"]
    C --> D["LM logit-producing state"]
    D --> E["linear → A/B/C/D"]
    F["(b) feedback\nwhat is recurrently injected"] --> G["P"]
    G --> H["z"]
    H --> I["Answer"]
    I --> J["linear → A/B/C/D"]
    K["decodability gap\nhow answer-relevant content is migrated"] --> L["acc(a)\nacc(b)\nG"]
    M["PRISM diagnosis"] --> N["probes track\ncorruption (small lΔl)"]
```
</details>

![](images/be3ce1ec1d32a8b53973d897ec5c3e11feeb51d576ec4eb46e1d1493d295b209.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["truncate (h ← 0) is the channel used?"] --> B["P"]
    B --> C["0"]
    C --> D["Ans*"]
    E["noise σ how brittle is it?"] --> F["P"]
    F --> G["z"]
    G --> H["Ans*"]
    I["swap specific to these latents?"] --> J["P"]
    J --> K["z"]
    K --> L["Ans*"]
    M["target donor"] -.-> J
    style A fill:#d4edda,stroke:#333
    style E fill:#d4edda,stroke:#333
    style I fill:#d4edda,stroke:#333
```
</details>

Figure 2: PRISM overview. Two inference-time diagnostics for the LVR family. Axis 1 trains linear probes (Alain and Bengio, 2018) at two positions in the model: (a) the answer-decoding state the LM head reads, and (b) the feedback variable the autoregressive loop re-injects. We report the decodability gap $G = a c c _ { p r o b e } ( a ) - a c c _ { p r o b e } ( b )$ , which summarizes how much more answer-decodable the post-latent state is than the latent. Axis 2 perturbs the LVR-injected hidden states (latents) in generation (truncation, noise, swap) and measures the change in accuracy; small |∆acc| means the latent is bypassed. Section 5 shows the two axes are tied: G predicts each variant’s response to latent perturbation.

This is the source of the dissociations we report. Cosine measures how well the MSE proxy performs, i.e., how close $Z _ { c h }$ is to $\mathbf { v } _ { t }$ and how effectively the proxy compression succeeds. It does not measure where relevance has settled. Section 5 shows that at inference, removing $Z _ { c h }$ entirely shifts accuracy by at most four points, which means that the model has routed relevance through other parts rather than the latents. The CE pressure that ran through $Z _ { c h }$ during training has been satisfied elsewhere.

Therefore, cosine reports on one half of an indirect IB objective and is silent on the other half. A high cosine tells that the latent is well-anchored to its target. It does not indicate whether the latent carries any load, i.e., whether the model actually uses the latent to produce the answer. In Section 4, we propose a measurement that asks that question: the decodability gap between the answer-decoding state and the feedback variable (last latent). The Section 5 cross-variant pattern is what that looks like empirically: cosine moves across a 40% range, and accuracy moves independently because the two are answering different questions under the current LVR loss.

# 4 PRISM: a replacement diagnostic for cosine

Cosine measures the fidelity of a latent to its supervision target. The findings in Section 5 show that this is a misleading question for analyzing the LVR family: the supervised latents are largely bypassed at inference, so target fidelity tells little about where the answer actually lies. PRISM replaces cosine with two inference-time diagnostics: a linear probe that asks where the answer is decodable, and a corruption test that asks whether the supervised latent is load-bearing.

# 4.1 Axis 1: linear probes

Where the answer is decodable tells where the loss actually occurred. To find out, for any LVR model and any multiple-choice question, we run the model once in the answer-decoding mode and extract two hidden-state vectors. Position (a) is the answerdecoding states: the hidden state at the iteration that produces the answer-text token, after the model has consumed the LVR tokens. This is the state the LM head reads to predict the answer. Position (b) is the feedback variable: the hidden state at the LVR boundary that the autoregressive loop re-injects as the next-position input embedding.

We fit logistic regression at each position $p \in$ $\{ a , b \}$ from the hidden state to the answer letter using 5-fold class-stratified cross-validation, $\ell _ { 2 }$ regularization, and per-fold input standardization (Alain and Bengio, 2018). We denote by $a c c _ { p r o b e }$ the

A.LVR   
MSE to a low-rate target   
![](images/0499827d700e7979a76053e65f640b2162fecba957a5b2ff1f0eafba6db44c9e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["h_{t-1}"] --> B["z"]
    B --> C["MSE(v_t)"]
    D["Image"] -->|v_t| B
    note["No constraint on what z carries."]
```
</details>

B.P-LVR   
(2-stage / 3-stage)   
![](images/e6301719941624bb53d2aced066efc015a91ba6f035d5a1397f2d0fe99dc7fb8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["h_{t-1}"] --> B["z^c"]
    B --> C["z^f"]
    C --> D["z^i"]
    D --> E["3-stage adds z^f + diversity reg."]
    style A fill:#fff,stroke:#000
    style B fill:#fff,stroke:#000
    style C fill:#fff,stroke:#000
    style D fill:#fff,stroke:#000
    style E fill:#fff,stroke:#000
    subgraph "diversity"
        V["v_t (expanded)"] --> W["MSE"]
        X["v_t (precise)"] --> Y["MSE"]
    end
```
</details>

C.D-LVR   
late reconstruction release   
![](images/d411c7dece59024c32e51e7a41583a74de0e8175ce79899c768cda92fb9bc2ec.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["step 1500"] --> B["h_{t-1}"]
    B --> C["z"]
    D["λ=0.1 for steps 1–1500, then λ=0 for the next 1000."] --> E["λ(t) · MSE"]
    E --> C
```
</details>

IB-CONSISTENT REGULARIZER— local-smoothness pressure on I(X; Z) via Bishop noise–Tikhonov.

D.N-LVR   
input-space IB — Bishop noise–Tikhonov   
Tikhonov-bounded Jacobian; IB-consistent local smoothness, not a formal MI bound.   
σ = 0.3   
![](images/6b4d675ab1f5f4f834a489118030726bbcee205a7ab5f40e926ab98f689f4e67.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["h_{t-1}"] --> B["+"]
    C["ε ~ N(0, σ^2)"] --> B
    B --> D["LVR head (linear)"]
    D --> E["z"]
    F["v_t (clean)"] --> E
    E --> G["MSE(z, v_t)"]
```
</details>

LEGEND

![](images/099897b8864d99bb12879e38d99d69647d6e5134b8f7787583e9838fa9ba5c50.jpg)

hidden state h

![](images/2404f10ecbbe7bd05dc067dd053e576b57d64dc72fad5b0b3fd5abfe614fa4cf.jpg)

latent z

![](images/3da9324e6a6062fbf0def95d850a0daa7769b26dd0635e06ced2d3c4c7f9becf.jpg)

module / head

![](images/f2082c49d2ee7d42a2b9a7f926f4ddfeaa1e976a94ccf5087e36144b5107903b.jpg)

MSE loss

![](images/2bfad2ea4a12fca6d45c0a4a54e3ccfb4ee9f1f01699a4d343559dc353b1b372.jpg)

target v

![](images/592a830561df9ac99b012c6120c76a02c484ebcc9c8125bf4690dc29985bbbf3.jpg)

forward pass

![](images/bba74df9bc7738c8f7f81db4f2527f6ab1e4919482bf80bf01c40805b30ceb29.jpg)

supervision

Figure 3: The matrix of five LVR variants, grouped by their relationship to the IB objective. Top row (reconstruction-only supervision): LVR baseline, P-LVR (progressive 2/3-stage scaffolding), D-LVR (reconstruction loss removed at mid-training). All three rely on an MSE-to-low-rate target as an indirect compression mechanism, with no direct bound on $I ( X ; Z )$ . Bottom (IB-consistent regularizer): N-LVR adds zero-mean Gaussian noise to the teacher-forced input during training while keeping the target clean, which to first order is a Tikhonov-style Jacobian penalty on the encoder, i.e., a local-smoothness prior approximating IB-style suppression of channel rate.

resulting cross-validated accuracy. The Barber– Agakov variational bound (Barber and Agakov, 2003) gives

$$
\begin{array}{l} I (\mathrm{rep}; Y) \\ \geq H (Y) + \mathbb {E} _ {(\mathrm{rep}, Y) \sim p _ {\mathrm{data}}} [ \log q (Y \mid \mathrm{rep}) ] \\ = H (Y) - \mathrm{CE} (q), \\ \end{array}
$$

where $H ( Y )$ denotes the entropy of the answer variable Y , $p _ { \mathrm { d a t a } } ( \mathrm { r e p } , Y )$ is the empirical joint distribution induced by the held-out examples, $q ( Y \mid \mathrm { r e p } )$ is a variational predictor of the answer from the representation. Thus, a low held-out cross-entropy provides a lower bound on how much answer-relevant information the representation carries.

The two probe accuracies describe a contrast: how much more decodable the answer is at the model answer-decoding state than at the latent loop hands forward. In any working model, $a c c _ { p r o b e } ( a )$ will be high as position (a) is the state the LM head reads to produce an answer. The empirical question is what happens at (b): does the latent loop hand forward carry the answer, or has the answer signal settled elsewhere? We summarize the contrast with the decodability gap $G \equiv$ $a c c _ { p r o b e } ( a ) - a c c _ { p r o b e } ( b )$ , but report both probes throughout, because the gap and its components carry different information. Section 5 reports all three across the variant matrix, and the connection to Axis 2.

# 4.2 Axis 2: faithfulness corruption

We perturb the LVR latents at each generation and measure the change in benchmark accuracy relative to a clean pass. The three perturbations are truncation $( h  0 )$ , additive Gaussian noise at $\sigma \in \{ 0 . 1 , 0 . 3 , 1 . 0 \}$ , and random-donor swap $( h  h _ { d o n o r }$ , where $h _ { d o n o r }$ is another sample’s clean latent at the matching iteration). The corruption propagates into the KV cache at every layer; for K/V projections without bias, truncation yields literally zero K and V at the corrupted position.

A small change in accuracy $( | \Delta \mathrm { a c c } | )$ ) under intervention is what bypass looks like operationally, i.e., the model does not rely on the latent under that intervention. The sign of $\Delta$ separates the case where the latent helps from the case where it hurts: own latents better than random against own latents worse. The three perturbations probe different things. Truncation asks the binary question, i.e., is the latent used at all? Noise grades robustness as a function of perturbation magnitude. Swap asks whether the answer is content-specific to this sample’s latent or if the model would tolerate a random donor.

Table 1: Per-variant measurements across the LVR matrix. cos: mean cosine alignment between LVR-position hidden state and target visual embedding. V∗B / MMVP / BLINK: benchmark accuracy (%). Acc(a) / Acc(b): linear-probe accuracy (%) at the answer-decoding state $H _ { \mathrm { a n s } }$ and the feedback variable $Z _ { \mathbf { f b } } .$ , under 5-fold classstratified cross-validation. $\mathbf { G } \mathbf { \cdot }$ probe contrast (decodability gap), $\operatorname { A c c } ( a ) - \operatorname { A c c } ( b )$ in percentage points. trunc / σ/ swap: faithfulness-corruption $\Delta$ accuracy on $\mathbf { V } ^ { * }$ Bench (signed; negative means corruption hurts, i.e. own latents help). 

<table><tr><td rowspan="2">Variant</td><td rowspan="2">cos</td><td colspan="3">Accuracy</td><td colspan="3">Probes</td><td colspan="5">Faithfulness corruption Δacc</td></tr><tr><td>V*B</td><td>MMVP</td><td>BLINK</td><td>(a)</td><td>(b)</td><td>G</td><td>trunc</td><td>σ=.1</td><td>σ=.3</td><td>σ=1.0</td><td>swap</td></tr><tr><td>LVR</td><td>0.555</td><td>70.2</td><td>49.7</td><td>53.4</td><td>69.1</td><td>32.5</td><td>36.6</td><td>-2.6</td><td>-1.6</td><td>+1.0</td><td>-1.0</td><td>+1.6</td></tr><tr><td>N-LVR</td><td>0.556</td><td>71.7</td><td>50.0</td><td>52.9</td><td>66.0</td><td>41.9</td><td>24.1</td><td>-2.6</td><td>-0.5</td><td>-1.0</td><td>-2.1</td><td>-0.5</td></tr><tr><td>D-LVR</td><td>0.464</td><td>69.6</td><td>49.3</td><td>51.4</td><td>64.9</td><td>33.0</td><td>31.9</td><td>-1.1</td><td>0.0</td><td>+0.5</td><td>0.0</td><td>+2.1</td></tr><tr><td>P-LVR-2</td><td>0.777</td><td>57.1</td><td>48.7</td><td>47.3</td><td>50.2</td><td>35.6</td><td>14.6</td><td>0.0</td><td>+2.1</td><td>0.0</td><td>-0.5</td><td>+2.1</td></tr><tr><td>P-LVR-3</td><td>0.769</td><td>57.1</td><td>48.0</td><td>48.5</td><td>48.7</td><td>34.6</td><td>14.1</td><td>+2.1</td><td>+2.6</td><td>+4.2</td><td>+3.1</td><td>+2.6</td></tr></table>

Table 2: Cross-variant Pearson correlations of each diagnostic against V∗Bench accuracy, the truncation and small-noise $( \sigma = 0 . 1 )$ corruption response. 

<table><tr><td>Diagnostic</td><td>vs V*B</td><td>vs trunc Δ</td><td>vs σ=.1 Δ</td></tr><tr><td>cosine</td><td>-0.94</td><td>+0.75</td><td>+0.84</td></tr><tr><td>probe-(a)</td><td>+0.98</td><td>-0.92</td><td>-0.98</td></tr><tr><td>probe-(b)</td><td>+0.20</td><td>-0.26</td><td>-0.02</td></tr><tr><td>G</td><td>+0.86</td><td>-0.77</td><td>-0.93</td></tr></table>

# 5 Experiments

# 5.1 The LVR variant matrix and setup

We designed and trained five variants of the LVR loss on a shared backbone, data, and step budget. The variants were chosen to span the common design choices the family uses to train its latent tokens. The shared loss is $\mathcal { L } _ { \mathrm { L V R } } = \mathcal { L } _ { \mathrm { C E } } + \lambda \cdot \mathbf { M S E } ( \mathbf { h } _ { t } , \mathbf { v } _ { t } )$ 号 with $\lambda = 0 . 1$ throughout. Each variant modifies how supervision is applied, which target it uses, or how latent positions are organized.

LVR (Li et al., 2026) is the unconstrained baseline: a single latent block of K positions, each supervised against its teacher visual embedding, with no additional constraints. N-LVR adds zero-mean Gaussian noise $( \sigma = 0 . 3 )$ to the teacher-forced input during training while keeping the target clean. To first order, this is a Tikhonov-style Jacobian penalty on the encoder (Bishop, 1995), i.e., a localsmoothness prior that approximates IB-style suppression of channel rate. D-LVR resumes from the LVR checkpoint at step 1500 and continues for 1000 more steps with $\lambda _ { r e c } = 0 .$ , testing what the VLM does when reconstruction pressure is removed mid-training while CE pressure continues. P-LVR-2 and P-LVR-3 split the latent block into stages: context → target (P-LVR-2) or context → free → target (P-LVR-3). In which each stage (except the free stage) is teacher-forced against its own bounding box embedding, with the context box expanded by $\alpha = 1 . 5$ .

The backbone is Qwen2.5-VL-3B-Instruct (Bai et al., 2025), with the vision tower and visual merger frozen and the language model trainable. The training data is Visual-CoT (Shao et al., 2024): 438k question-answer pairs with task-relevant bounding boxes. All variants train for 2500 steps with a learning rate of $1 0 ^ { - 5 }$ on a cosine schedule, in bf16 precision, with an effective batch size of 64. We evaluate on $\mathbf { V } ^ { * }$ Bench (Wu and Xie, 2024) for visual search in cluttered scenes, MMVP (Tong et al., 2024) for paired-option comparison, and BLINK (Fu et al., 2024) for perception-oriented multiple choice across five validation subsets.

# 5.2 Cosine inverts the ordering it claims to measure

Cosine alignment does not just fail to predict accuracy – across the matrix, it predicts accuracy backward. Table 1 reports cosine alignment to teacher targets and V∗Bench accuracy across the five variants. The cross-variant Pearson correlation is $r = - 0 . 9 4$ (Table 2), i.e., cosine and accuracy move in opposite directions.

Beyond the surprising correlation, two specific dissociations pin down the kind of failure cosine exhibits. The first is directional. The progressive scaffolding variants raise cosine from the baseline’s 0.555 to 0.777 and 0.769 (a 40% gain) and simultaneously lose 13 V∗Bench points. The progressive design trades training simplicity for cleaner reconstruction at the teaching stage. Cosine rewards that trade and $\mathbf { V } ^ { * }$ Bench punishes it. The second dissociation is about resolution. LVR and N-LVR have nearly identical cosine values (0.555 vs. 0.556), but different $\mathbf { V } ^ { * }$ Bench accuracies and reverse the sign of their swap response: LVR’s own latents hurt (+1.6), N-LVR’s help (-0.5). Cosine cannot distinguish between these two variants, even though every other measurement in the paper does.

![](images/71cab27e6728cffa3863c5914e26a33811af722b5cace8067f877f52a093a6bb.jpg)

<details>
<summary>bar</summary>

| Corruption applied to LVR latent | LVR   | N-LVR | D-LVR | P-LVR-2 | P-LVR-3 |
| -------------------------------- | ----- | ----- | ----- | ------- | ------- |
| trunc.                           | -2.5  | -0.5  | -0.8  | 2.2     | 2.1     |
| σ = .1                           | -1.5  | -0.3  | 0.0   | 2.1     | 2.7     |
| σ = .3                           | 1.0   | -0.8  | 0.5   | 4.2     | 4.3     |
| σ = 1                            | -0.8  | -1.8  | 0.0   | -0.5    | 3.2     |
| swap                             | 1.5   | -0.5  | 2.1   | 2.1     | 2.6     |
</details>

Figure 4: Faithfulness corruption profiles on V∗Bench. Change in $\mathbf { V } ^ { * }$ Bench accuracy (∆ acc, pp) under each corruption applied to the LVR latents at latent reasoning steps, against a clean pass. Negative $\Delta =$ own latent helps; positive = own latent hurts.

A third dissociation surfaces once corruption is reported (Table 2). Cosine correlates positively with corruption response: $r ~ = ~ + 0 . 7 5$ against truncation $\Delta$ and $r = + 0 . 8 4$ against small-noise $( \sigma = 0 . 1 )$ . Variants with higher cosine are more helped by perturbing their latents, and the LVR baseline (which has middling cosine) shows the largest harm from truncation $( \Delta = - 2 . 6 )$ . As a result, cosine fails to predict accuracy and points in the wrong direction regarding whether the latent is useful to the model.

# 5.3 The latents are largely bypassed at inference

We applied PRISM’s Axis 2 corruption test at every LVR latent reasoning step: zeroing the latent hidden state (truncation), perturbing it with Gaussian noise at $\sigma \in \{ 0 . 1 , 0 . 3 , 1 . 0 \}$ , or replacing it with another sample’s latents at the matching iteration (random-donor swap). Figure 4 shows the resulting change in $\mathbf { V } ^ { * }$ Bench accuracy per variant per corruption.

Across all five variants, every intervention shifts V∗Bench by at most four points. For P-LVR-3 with high reconstruction fidelity, zeroing the latents actually improves accuracy, suggesting that the latents were actively interfering. N-LVR is the only variant where own latents reliably beat random under swap, and for the rest, own latents are mildly worse than what a random donor provides.

This rules out a natural reading of the crossvariant V∗Bench spread. The 13-point spread cannot be explained primarily by what the latent encodes, because the latents are bypassed by similar margins throughout the matrix. Whatever distinguishes variants from one another lives in the model the loss has shaped, not in the latent content the loss optimizes. The corruption response is also informative in its own right. It differentiates variants along a dimension that cosine cannot see, i.e., LVR and N-LVR have nearly identical cosines but opposite signs of the swap ∆.

# 5.4 The probe contrast localizes where the answer lives

Cosine and corruption agree on the negative finding that the latents are not where the answer lives, but neither tells where it has gone. PRISM’s Axis 1 does. For each variant, we fit a linear probe at the answer-decoding state, with accuracy $a c c _ { p r o b e } ( a )$ , and at the feedback variable, with accuracy $a c c _ { p r o b e } ( b )$ . The two accuracies and their gap $G = a c c _ { p r o b e } ( a ) - a c c _ { p r o b e } ( b )$ are reported in Table 1.

$a c c _ { p r o b e } ( a )$ ranges from 50.2 (P-LVR-3) to 69.1 (LVR), and tracks V∗Bench across the matrix at $r = + 0 . 9 8$ (Table 2). $a c c _ { p r o b e } ( b )$ is much lower, between 32.5 and 41.9, and does not track $\mathrm { V } ^ { \ast }$ Bench $( r = + 0 . 2 0 ) $ . The contrast between the two (G) ranges from 14.1 to 36.6 percentage points, and is itself a strong predictor of $\mathbf { V } ^ { * }$ Bench $( r = + 0 . 8 6 )$ . The unconstrained baseline shows the largest gap, while the progressive variants show the smallest.

All three quantities point at the same conclusion. The answer-relevant signal sits downstream of the supervised latents, not in the latent content the loop re-injects. Probe accuracy at the model’s answerdecoding state varies across variants in step with task accuracy. Probe accuracy at the feedback variable varies independently of it. Cosine reports on the latent’s fidelity to its target, but the latent does not carry the answer. The probe contrast reports the gap between latent and downstream decoding, and that gap tracks the model’s competence on the task.

![](images/272f2bf71e82c5e4fb289946be5764bdf4ae94fa86f30b74abbe81a2d58f137e.jpg)

<details>
<summary>scatter</summary>

| Method   | Decodability gap G (pp) | Δ V*Bench accuracy (pp) |
| -------- | ------------------------ | ------------------------ |
| P-LVR-3  | 14                       | 2.8                      |
| P-LVR-2  | 15                       | 2.0                      |
| N-LVR    | 24                       | -0.5                     |
| D-LVR    | 32                       | 0.0                      |
| LVR      | 37                       | -1.8                     |
</details>

![](images/d7f0f0e8f2b6ed8b9c5211729884fc73db8cc27297e73eed244af6644c7da803.jpg)

<details>
<summary>scatter</summary>

| Method   | Decodability gap G (pp) | Δ V*Bench accuracy (pp) |
| -------- | ------------------------ | ------------------------ |
| P-LVR-3  | 15                       | 2.0                      |
| P-LVR-2  | 15                       | 0.0                      |
| D-LVR    | 32                       | -1.0                     |
| N-LVR    | 24                       | -2.5                     |
| LVR      | 36                       | -2.8                     |
</details>

Figure 5: The probe contrast predicts latent reliance. Decoding gap G against corruption response under small-noise (σ = 0.1, left) and truncation (right), across the five variants.

# 5.5 The probe contrast predicts latent reliance

Axis 1 measures where the answer is decodable. Axis 2 assesses whether the model causally uses the latents. These are different questions, but on this matrix they are empirically tied (Table 2). $a c c _ { p r o b e } ( a )$ correlates with the corruption response under truncation at $r = - 0 . 9 2$ and under smallnoise $( \sigma = 0 . 1 )$ at $r = - 0 . 9 8$ . The gap G shows the same pattern at $r = - 0 . 7 7$ and $r = - 0 . 9 3$ (Figure 5). Variants where the answer-decoding state is more answer-informative also rely on their latent more under perturbation.

As Figure 5 shows, when the decodability gap is large, i.e., when the answer has settled at the decoding state but not at the latent, the latent still plays a structural role in the path that produces the answer, and removing or lightly perturbing it disrupts downstream computation. When the contrast is small (the P-LVR variants), the latent and downstream states are close to each other, and truncating the latent does not disrupt anything specific. Bypass is the average behavior across the matrix. The probe contrast predicts where each variant sits on the bypass spectrum.

This connection is the strongest empirical statement that PRISM enables, and it requires both axes. Cosine cannot see the probe contrast. The corruption test alone cannot determine where the answer lies. The probe alone cannot see what the model actually uses. The connection between the probe contrast and corruption response is visible only when both diagnostics are applied to the same matrix.

# 6 Discussion

Recent LVR works were built on the assumption that a latent supervised to match a visual target will carry that visual information into the answer. Our results show the assumption can fail without the method failing: the supervised latents are largely bypassed, yet the variants still differ by 13 V∗Bench points because the auxiliary loss reshapes the VLM through gradient flow into shared parameters. The generalizable point is that an auxiliary loss can act through a path it does not name, i.e., by propagating into shared parameters rather than into the latent, which is nominally optimized. A metric defined on that latent will then measure the wrong signal. Whenever an intermediate representation is supervised against an external target and fed back through a shared model, the same dissociation is possible, and PRISM’s two axes are a way to check for it before trusting an alignment metric.

# 7 Conclusion

Cosine alignment is trusted across latent visual reasoning as both a training loss and a quality metric, but across a designed matrix of five variants, it ranks them backward against accuracy, while the latents it optimizes are bypassed at inference. PRISM locates the discrepancy as the answer is decodable downstream of the latent but not at it, and the size of that gap predicts how much each variant relies on its latent under perturbation. Cosine measures the fidelity of a latent that the model has learned to route around, while PRISM measures where the answer actually lies.

# Limitations

Our evidence comes from one base model (Qwen2.5-VL-3B-Instruct) and one fine-tuning corpus (Visual-CoT-438k). Larger backbones or controlled-init replicates could shift the per-variant numbers we report. Both diagnostic axes are bounded. Linear probes measure linear decodability rather than what downstream layers actually use; we use a random-label control to bound selectivity. The corruption test covers truncation, three noise levels, and donor swap but does not exhaust the perturbation space. Our benchmarks focus on fine-grained perception (V∗Bench, MMVP, BLINK), where latent quality has the greatest leverage; holistic reasoning benchmarks with less spatially localized questions may exhibit different cosine–accuracy patterns.

# References

Guillaume Alain and Yoshua Bengio. 2018. Understanding intermediate layers using linear classifier probes. Preprint, arXiv:1610.01644.   
Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, and Kevin Murphy. 2017. Deep variational information bottleneck. In International Conference on Learning Representations.   
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025. Qwen2.5-vl technical report. Preprint, arXiv:2502.13923.   
David Barber and Felix Agakov. 2003. Information maximization in noisy channels : A variational approach. In Advances in Neural Information Processing Systems, volume 16. MIT Press.   
Yonatan Belinkov. 2022. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219.   
Chris M. Bishop. 1995. Training with noise is equivalent to tikhonov regularization. Neural Computation, 7(1):108–116.   
Florian Bordes, Richard Yuanzhe Pang, Anurag Ajay, Alexander C. Li, Adrien Bardes, Suzanne Petryk, Oscar Mañas, Zhiqiu Lin, Anas Mahmoud, Bargav Jayaraman, Mark Ibrahim, Melissa Hall, Yunyang Xiong, Jonathan Lebensold, Candace Ross, Srihari Jayakumar, Chuan Guo, Diane Bouchacourt, Haider Al-Tahan, and 22 others. 2024. An introduction to vision-language modeling. Preprint, arXiv:2405.17247.

Shuai Dong, Siyuan Wang, Xingyu Liu, Chenglin Li, Haowen Hou, and Zhongyu Wei. 2026. Interleaved latent visual reasoning with selective perceptual modeling. Preprint, arXiv:2512.05665.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. 2024. Blink: Multimodal large language models can see but not perceive. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part XXIII, page 148–166, Berlin, Heidelberg. Springer-Verlag.

John Hewitt and Percy Liang. 2019. Designing and interpreting probes with control tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2733–2743, Hong Kong, China. Association for Computational Linguistics.

Byungwoo Jeon, Yoonwoo Jeong, Hyunseok Lee, Minsu Cho, and Jinwoo Shin. 2026. Vision-aligned latent reasoning for multi-modal large language model. Preprint, arXiv:2602.04476.

Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. 2025. Repa-e: Unlocking vae for end-to-end tuning of latent diffusion transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 18262–18272.

Bangzheng Li, Ximeng Sun, Jiang Liu, Ze Wang, Jialian Wu, Xiaodong Yu, Emad Barsoum, Muhao Chen, and Zicheng Liu. 2026. Latent visual reasoning. In The Fourteenth International Conference on Learning Representations.

Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. 2024. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. In The Thirty-eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Naftali Tishby, Fernando C. Pereira, and William Bialek. 2000. The information bottleneck method. Preprint, arXiv:physics/0004057.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. 2024. Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs . In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9568–9578, Los Alamitos, CA, USA. IEEE Computer Society.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. 2023. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. In Thirty-seventh Conference on Neural Information Processing Systems.

Martin Tutek, Fateme Hashemi Chaleshtori, Ana Marasovic, and Yonatan Belinkov. 2025. Measuring chain of thought faithfulness by unlearning reasoning steps. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9935–9960, Suzhou, China. Association for Computational Linguistics.

Qixun Wang, Yang Shi, Yifei Wang, Yuanxing Zhang, Pengfei Wan, Kun Gai, Xianghua Ying, and Yisen Wang. 2025. Monet: Reasoning in latent visual space beyond images and language. Preprint, arXiv:2511.21395.

Penghao Wu and Saining Xie. 2024. V\*: Guided visual search as a core mechanism in multimodal llms. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13084–13094.

Zeyuan Yang, Xueyang Yu, Delin Chen, Maohao Shen, and Chuang Gan. 2025. Machine mental imagery: Empower multimodal reasoning with latent visual tokens. Preprint, arXiv:2506.17218.

# A Linear Probe Methodology

# A.1 Probe Extraction

For each trained variant and each V∗Bench question, we run the model once in answer-decoding mode and extract two hidden-state vectors per LVR position:

• Position (a) — answer-decoding state: the hidden state at the iteration that produces the first answer-text token, after the model has consumed all LVR tokens. This is the actual logit-producing state read by the LM head.   
• Position (b) — latent-feedback variable: the feedback variable that the autoregressive loop re-injects into context as the next-position input embedding at the boundary of the LVR-mode block. For the variants in our family this is the raw LVR-position hidden state h, as there is no learned projection between the LVR position and the next input embedding.

Each vector is D = 2048-dimensional. We extract from a single inference pass per question, with greedy decoding.

# A.2 Probe Training

Probes are trained as multi-class logistic regression over the four V∗Bench answer letters {A, B, C, D}:

$$
\hat {y} = \operatorname{softmax} (W x + b), \quad W \in \mathbb {R} ^ {4 \times D}.
$$

We standardize features (z-score per dimension over the training fold) before fitting. Standardization is essential: without it, the unregularized hidden-state norms dominate the LBFGS step and make probe accuracy more reflective of activation scale than of decodability.

Optimization uses scikit-learn’s LogisticRegression with L2 regularization at C = 1.0 and the LBFGS solver, multinomial loss, max\_iter = 1000. We report accuracy under stratified 5-fold cross-validation; the reported number is the mean across folds.

# A.3 Control Task

Following Hewitt and Liang (2019), we run a control task to bound probe selectivity. We construct a control label by assigning each V∗Bench question a uniformly random target in {A, B, C, D}, fixing the assignment, and re-running the probe-training pipeline. A probe that achieves above-chance accuracy on the control task does so by virtue of probe capacity rather than property-presence in the representation.

Selectivity is defined as (probe accuracy on real labels) − (probe accuracy on control labels). For all five variants (LVR, N-LVR, D-LVR, P-LVR-2, P-LVR-3), the answer-letter probe achieves selectivity > 25% at position (a), indicating that the linear answer-decodability we measure is not an artifact of the probe family. Probe-(b) selectivity is uniformly under 10%, consistent with the body claim that the latent-feedback position carries no answer-specific structure.

# A.4 Full Probe Results

Table 3: Linear-probe accuracy and held-out crossentropy on V∗Bench, 5-fold CV, chance= 25%. Accuracies are reported as mean ± standard error across the five folds. Position (a) is the LM’s answer-decoding hidden state; (b) is the latent-feedback variable; A1′ is the question-token mean-pool from the prefill pass. CE@(a) is the held-out cross-entropy of the position-(a) probe. The variational MI lower bound is MI =H(Y ) − CE in nats, zero-clamped (Barber and Agakov, 2003); we use the empirical V∗Bench label entropy H(Y )=1.267 nats (label counts A=71, B=70, C=27, D=23 out of 191) rather than the uniform-label ln 4=1.386.

<table><tr><td>Variant</td><td>(a)</td><td>CE@(a)</td><td> $\text{MI}_{\text{LB}}$ </td><td>(b)</td><td>A1&#x27; Q</td><td>V*B</td></tr><tr><td>LVR (single)</td><td>69.1±2.2</td><td>1.151</td><td>0.116</td><td>32.5±3.8</td><td>54.4±3.9</td><td>70.2</td></tr><tr><td>N-LVR</td><td>66.0±3.3</td><td>1.307</td><td>0.000</td><td>41.9±3.9</td><td>57.5±3.2</td><td>71.7</td></tr><tr><td>D-LVR</td><td>64.9±2.4</td><td>1.122</td><td>0.145</td><td>33.0±2.5</td><td>62.2±3.5</td><td>69.6</td></tr><tr><td>P-LVR-2-stage</td><td>50.2±4.8</td><td>1.954</td><td>0.000</td><td>35.6±2.8</td><td>47.0±3.8</td><td>57.1</td></tr><tr><td>P-LVR-3-stage</td><td>48.7±1.7</td><td>1.935</td><td>0.000</td><td>34.6±1.8</td><td>43.9±3.0</td><td>57.1</td></tr><tr><td colspan="7">Pearson r vs V*B</td></tr><tr><td>position (a) accuracy</td><td></td><td></td><td></td><td></td><td colspan="2">+0.980</td></tr><tr><td>CE@(a)</td><td></td><td></td><td></td><td></td><td colspan="2">-0.963</td></tr><tr><td> $\text{MI}_{\text{LB}}$ @(a)</td><td></td><td></td><td></td><td></td><td colspan="2">+0.579</td></tr><tr><td>position (b) accuracy</td><td></td><td></td><td></td><td></td><td colspan="2">+0.198</td></tr><tr><td>A1&#x27; Q-mean accuracy</td><td></td><td></td><td></td><td></td><td colspan="2">+0.898</td></tr></table>

Accuracy vs. CE as MI proxies. By Barber– Agakov (Barber and Agakov, 2003), $I ( { \mathrm { r e p } } ; Y ) \geq$ $\mathrm { M I _ { L B } } \ = \ H ( Y ) - \mathrm { C E } ( p _ { \theta } )$ , where $H ( Y )$ is the marginal label entropy. We use the empirical V∗Bench label entropy $H ( Y ) { = } 1 . 2 6 7$ nats (label counts $A { = } 7 1 , B { = } 7 0 , C { = } 2 7 , D { = } 2 3$ out of 191) rather than the uniform-label ln $4 { = } 1 . 3 8 6$ , since assuming a uniform-label distribution would inflate the lower bound by 0.119 nats per variant. Under the empirical $H ( Y )$ the MI lower bound across our five trained variants ranges from 0 (N-LVR and the P-LVR variants clamp to zero because their probe CE exceeds $H ( Y )$ — the probe is worse than the empirical marginal predictor under logloss) to 0.15 nats (D-LVR); the LVR baseline is 0.12 nats. Probe accuracy tracks V∗Bench tightly $( r { = } \mathrm { + } 0 . 9 8 0 )$ and CE@(a) tracks it inversely with comparable magnitude $\scriptstyle ( r = - 0 . 9 6 3 )$ ; $\mathbf { M } _ { \mathrm { L B } } @ ( \mathbf { a } )$ , the non-negative-clamped variational lower bound, correlates more weakly $( r { = } \mathrm { + 0 . 5 8 } )$ because the clamp pins N-LVR and the P-LVR variants to zero, removing their relative ordering. N-LVR illustrates this asymmetry: its decent probe accuracy $( 6 6 . 0 , \sim 4 1 $ pts above chance) with a CE of 1.307 implies that the probe’s per-question predictions are accuracy-good but uncertainty-rich, with CE just above the empirical $H ( Y ) { = } 1 . 2 6 7 - \mathrm { c o n s i s } -$ - tent with the noise-Tikhonov regularizer training a smoother hidden state.

# A.5 Robustness of the Probe-(a) Result

We support the headline probe-(a)/V∗Bench correlation with four checks — leave-one-out sensitivity, a partial correlation controlling for question text, bootstrap confidence intervals, and crossbenchmark replication — detailed below. The Pearson $r ~ = ~ + 0 . 9 8 0$ statistic between probe-(a) accuracy and V∗Bench top-1 accuracy is computed across the five trained variants (LVR, N-LVR, D-LVR, P-LVR-2, P-LVR-3); each variant contributes one $( \mathrm { p r o b e - } ( \mathrm { a } ) , \mathrm { V } ^ { \ast } \mathrm { B } )$ data point. With only five points the correlation is necessarily descriptive rather than strong statistical confirmation, which is why we report the checks that follow.

Leave-one-out sensitivity. Removing any single variant leaves $r \geq 0 . 9 7$ (range [0.969, 0.996]): removing the LVR baseline tightens the fit slightly (r rises to 0.996), and removing P-LVR-3 weakens it (r drops to 0.969). No single variant drives the relationship.

Partial correlation controlling for question text. To isolate the latent-specific signal from questiontext content, we compute the partial Pearson correlation r(probe-(a), $\mathbf { V } ^ { * } \mathbf { B } \ \mid \ \mathbf { A } \mathbf { l } ^ { \prime } )$ using $\mathbf { A } 1 ^ { \prime }$ (the question-token mean-pool baseline) as the conditioning variable:

$$
r _ {x y | z} = \frac {r _ {x y} - r _ {x z} r _ {y z}}{\sqrt {(1 - r _ {x z} ^ {2}) (1 - r _ {y z} ^ {2})}}.
$$

Across the five variants we find $r _ { x y } { = } 0 . 9 8 0 .$ , $r _ { x z } { = } 0 . 8 5 7 , \ r _ { y z } { = } 0 . 8 9 8$ , giving $r _ { x y | z } = + 0 . 9 2 9$ The bulk of probe-(a)’s cross-variant signal is not absorbed by question phrasing alone, though the small n means this partial correlation should be read as descriptive evidence of latent attribution, not as a formal conditional MI estimate.

Bootstrap confidence intervals. To put error bars on the small-n Pearson correlations, we bootstrap each variant’s $\mathbf { V } ^ { * }$ Bench accuracy assuming a binomial sampling model over its $n { = } 1 9 1$ questions $( \mathrm { V } ^ { * } \mathbf { B } _ { v } ^ { ( b ) } \sim$ Binomial(191, $p _ { v } ) / 1 9 1$ , with $p _ { v }$ the variant’s empirical accuracy), then recompute the cross-variant Pearson on each bootstrap replicate $\left( B { = } 5 0 0 0 \right)$ . The resulting 95% percentile CIs are wide, reflecting the structural limit of $n { = } 5$ variants rather than the per-variant accuracy noise:

• r(probe-(a), V\*B): $\begin{array} { r l } { \mathrm { p o i n t } } & { { } + 0 . 9 8 0 , } \end{array}$ CI [+0.694, +0.993].   
• r(cosine, V\*B): $\begin{array} { r l } { \mathrm { p o i n t } } & { { } - 0 . 9 3 9 , } \end{array}$ CI $[ - 0 . 9 8 7 , - 0 . 6 1 1 ] .$   
$\bullet \ r ( \operatorname { t r u n c } \Delta , \nabla ^ { * } \mathbf { B } ) { : }$ $\begin{array} { r l } { \mathrm { p o i n t } } & { { } - 0 . 8 9 2 , } \end{array}$ CI $[ - 0 . 9 7 8 , - 0 . 5 2 2 ] .$   
$\begin{array} { l } { { \bullet \ r ( \mathrm { s w a p } \Delta , \mathrm { V } ^ { * } \mathrm { B } ) \mathrm { : } } } \\ { { [ - 0 . 8 8 7 , - 0 . 2 4 1 ] . } } \end{array}$ $\begin{array} { r l } { \mathrm { p o i n t } } & { { } - 0 . 6 5 8 , } \end{array}$ C I

These intervals indicate the sign of each correlation is robust across plausible per-variant accuracy fluctuations, but the magnitude carries meaningful structural uncertainty: with only five variants the cross-variant relationship is well-determined as monotonic, not as a precise slope.

Cross-benchmark robustness. Probe-(a) tracks task accuracy on the other two benchmarks almost as tightly as on $\mathrm { V } ^ { \ast }$ Bench: $r ( \mathrm { p r o b e - } ( \mathrm { a } ) , \mathrm { B L I N K } ) =$ +0.96 and $r ( \mathrm { p r o b e - ( a ) } , \mathrm { M M V P } ) = + 0 . 9 2$ across the five-variant set. The cross-benchmark generalization indicates that probe-(a) is tracking a benchmark-general notion of answer decodability rather than a $\mathbf { V } ^ { * }$ Bench-specific artifact.

# A.6 Is the Probe-(a) Correlation Tautological?

A natural worry is that probe accuracy and V∗Bench accuracy both reflect “how good the model is,” so finding them correlated says nothing. Three observations rule this out. First, cosine alignment operates on the same hidden states yet ranks variants in the opposite direction $( r { = } \mathrm { - } 0 . 9 4 $ vs. probe-(a) at +0.98), so the question is what we measure about h, not whether we measure it. Second, of the two probe positions only (a) correlates strongly with V∗Bench (+0.98 vs. +0.20 at (b)); if probes simply reflected overall task competence, position (b) — inside the same model — would correlate too. Third, the random-label control task above yields selectivity > 25% at position (a) for every variant, ruling out a probe-capacity-only explanation.

# A.7 Why Two Positions

Each position has a distinct interpretation under the LVR generation loop. Position (a) measures whether the answer-letter logits are linearly readable from the state at decode time: high probe-(a) accuracy means the LM has converted the latent into answer-relevant structure by the answer-token. Position (b), the latent-feedback variable, measures what is being recurrently re-injected into context: a low probe-(b) accuracy means the recurrent variable does not carry the answer locally, so the answer signal must be assembled by downstream layers from the position-(a) cumulative state. Together, the two positions provide a localized account of where the answer signal lives, summarized by the decodability gap $G = \operatorname { a c c } _ { \mathrm { p r o b e } } ( a ) - \operatorname { a c c } _ { \mathrm { p r o b e } } ( b )$ used in the body.

# B Faithfulness Corruption Details

# B.1 Corruption Modes

We apply three corruption modes to the LVRposition feedback (the value the autoregressive loop re-injects as the input embedding for the LVR position via the patched forward, before any layer-1 K/V projection). All other generation state — prefill context, image embeddings, downstream decoder weights — is held identical between clean and corrupted runs.

• Truncate: $h  0$ at every LVR-mode iteration. The LVR tokens remain in the sequence; their input embedding is zero. K and V projections in Qwen2.5-VL have no bias, so the LVR positions contribute zero K and V to attention. They

Table 4: Faithfulness corruption ∆ accuracy on V∗Bench (percentage points; negative = corruption hurts, i.e. own latents help). All evaluations run at batch size 1 to match the main V∗Bench eval. The “clean” column is the script’s within-script clean pass (greedy generation with the corruption hook installed but a no-op transform). Single-stage variants run at steps=8; P-LVR-2 and P-LVR-3 run at steps=16 per stage (matching the main-table step budgets). The clean column matches the main V∗Bench numbers in Table 1 within ∼ 2 points for every variant; the residual difference is sample-order variance from the faith script’s single-pass greedy generation. Across the family the latent is largely bypassed at inference: under every intervention |∆| stays within roughly 4 points and never reverses a variant’s rank by more than its standard error.

<table><tr><td>Variant</td><td>clean</td><td>trunc</td><td> $\sigma 0.1$ </td><td> $\sigma 0.3$ </td><td> $\sigma 1.0$ </td><td>swap</td></tr><tr><td>LVR</td><td>70.2</td><td>-2.6</td><td>-1.6</td><td>+1.0</td><td>-1.0</td><td>+1.6</td></tr><tr><td>N-LVR</td><td>71.7</td><td>-2.6</td><td>-0.5</td><td>-1.0</td><td>-2.1</td><td>-0.5</td></tr><tr><td>D-LVR</td><td>69.1</td><td>-1.0</td><td>0.0</td><td>+0.5</td><td>0.0</td><td>+2.1</td></tr><tr><td>P-LVR-2</td><td>55.5</td><td>0.0</td><td>+2.1</td><td>0.0</td><td>-0.5</td><td>+2.1</td></tr><tr><td>P-LVR-3</td><td>56.0</td><td>+2.1</td><td>+2.6</td><td>+4.2</td><td>+3.1</td><td>+2.6</td></tr></table>

still consume attention mass via the softmax denominator, so other positions are mildly downweighted; this is the standard input-side ablation and is not equivalent to attention-masking the positions out of the sequence.

• Noise-σ for $ \sigma \in \{ 0 . 1 , 0 . 3 , 1 . 0 \} \colon h  h + \sigma \cdot \eta ,$ $\pmb { \eta } \sim \mathcal { N } ( 0 , I )$ . Here σ is in raw activation units, not standardized; the three magnitudes span roughly 0.1× to 1× the typical per-dimension scale of clean LVR vectors. This measures how brittle the latent is to perturbations of varying magnitude.   
• Swap: $h  h _ { \mathrm { d o n o r } }$ , where $h _ { \mathrm { d o n o r } }$ comes from another sample’s clean LVR positions at the matching LVR step (stage-tagged for P-LVR variants).

# B.2 Donor Sampling for the Swap Test

Donors are drawn from a pool of 64 samples recorded during the clean pass; for each batch the implementation calls random.choice (with replacement, but the pool size makes collision negligible at $n { = } 1 9 1 )$ . We do not enforce a different image or question between donor and target. The donor pool is fixed per (variant, seed) and re-used across all corruption modes within a variant, so swap variance across variants is dominated by the variant rather than donor noise.

# B.3 Variance Across Donor Seeds

We sweep three donor seeds for the swap test (Table 4 reports seed-42). Variant ranking by $\Delta _ { \mathrm { { s w a p } } }$ is invariant across seeds, and absolute $\Delta$ values move within ±1.0 V∗Bench point.

# B.4 Noise-σ Calibration

σ is in raw activation units: the corruption is $h  h + \sigma \cdot \eta$ with $\pmb { \eta } \sim \mathcal { N } ( 0 , I )$ per dimension, applied directly to the unstandardized LVR-position hidden state. Single-pass empirical statistics on the clean LVR vectors give a typical per-dimension standard deviation of order 0.5–1.5 across variants, so $\sigma { = } 0 . 1$ is well below the signal scale, $\sigma { = } 0 . 3$ is comparable to the smaller-scale variants’ perdimension spread, and σ=1.0 matches or exceeds the per-dimension signal scale for most variants. We did not pre-standardize across variants, so the same σ has slightly different SNR meanings across variants; we report the raw-unit number for transparency rather than a per-variant rescaling.

# B.5 Why Three Tests

Truncate, swap, and noise probe different aspects of latent reliance: truncate asks whether the latent is used at all (binary); swap asks whether the answer is specific to this sample’s latent content (a positive swap $\Delta$ means random latents help more); noise asks how robustly the answer is encoded at varying perturbation magnitudes. Across the five trained variants the three tests jointly point to the same conclusion: the latent is mostly bypassed, with $| \Delta |$ within roughly 4 V∗Bench points for every intervention (the largest is P-LVR-3 at σ=0.3 noise with $\Delta { = } + 4 . 2$ , where the perturbed latent actually helps the model more than the clean one). The variant ordering produced by these tests — especially by truncation, which has the lowest variance across the five-variant set — aligns with V∗Bench at $r { = } { - } 0 . 8 9$ (more-negative ∆, i.e. own latents helping, predicts higher $\mathrm { V ^ { * } B } _ { , }$ ; swap correlates more weakly at $r { = } { - } 0 . 6 6$ .

# B.6 Implementation

Corruption is applied to the input embedding at each LVR-mode iteration, before that position’s K/V projections; image embeddings, prompt tokens, and decoder weights are held identical between clean and corrupted runs. One inference pass per (variant, corruption, seed) tuple produces all reported numbers.

# C Training Hyperparameters

# C.1 Base Configuration (All Variants)

• Base model: Qwen2.5-VL-3B-Instruct (Bai et al., 2025).   
• Training corpus: Visual-CoT-438k, single-stage SFT.   
• Optimizer: AdamW, learning rate 1e−5, cosine schedule, warmup ratio 0.03, weight decay 0.1.   
• Precision: bf16 (no fp16); flash-attention 2 enabled.   
• Batching: per-device batch size 1; effective batch size 64 packed instances for all five variants (achieved via 1×64 or 2×32 GPU × grad-accum). See Appendix C.2 for the per-variant table.   
• Max instances per batch: 4; data packing enabled with max packed tokens 16,384 (long-seq threshold 4,096).   
• Image resolution: min 100,352 pixels, max 4,014,080 pixels.   
• Vision tower and merger frozen; LLM fine-tuned.   
• Steps: 2500, ∼ 1 epoch.   
• Random seed: 42.   
• DeepSpeed Zero-2, gradient checkpointing on, max grad norm 1.0.

# C.2 Per-Variant Training Provenance

All five variants use effective batch size 64 packed instances and 2500 training steps, but differ in how that effective batch is achieved (GPU × gradaccum).

<table><tr><td>Variant</td><td>Script</td><td>GPUs</td><td>Accum</td><td>bs</td></tr><tr><td>LVR (single)</td><td>finetune_lvr_stage1_3b.sh</td><td>1</td><td>64</td><td>1</td></tr><tr><td>N-LVR</td><td>finetune_nlvr_stage1_3b.sh</td><td>2</td><td>32</td><td>1</td></tr><tr><td>D-LVR</td><td>finetune_dlvr_stage1_3b.sh (resume @ 1500)</td><td>1</td><td>64</td><td>1</td></tr><tr><td>P-LVR-2-stage</td><td>finetune_plvr_stage1_3b.sh (free_stage=false)</td><td>1</td><td>64</td><td>1</td></tr><tr><td>P-LVR-3-stage</td><td>finetune_plvr_stage1_3b.sh (free_stage=true)</td><td>1</td><td>64</td><td>1</td></tr></table>

All configurations target the same effective batch size (GPUs × grad-accum × per-device bs) of 64 (LVR/D-LVR/P-LVR-2/P-LVR-3: 1 × 64 × 1; N-LVR: 2×32×1). Per-step optimizer dynamics differ slightly between the 1-GPU large-accumulation and 2-GPU smaller-accumulation routes (AdamW first/second-moment EMAs accumulate over the batch differently), which contributes a small additional source of cross-variant variance beyond the headline hyperparameters.

# C.3 Variant-Specific Hyperparameters

LVR (vanilla single-stage). $\lambda _ { \mathrm { l v r } } = 0 . 1$ , single latent block, MSE reconstruction against teacherforced box embeddings.

N-LVR. Same as LVR with one addition: inject zero-mean Gaussian noise $\mathcal { N } ( 0 , \sigma _ { \mathrm { n o i s e } } ^ { 2 } I ) , \sigma _ { \mathrm { n o i s e } } =$

0.3 in raw activation units, directly onto the teacherforced visual embeddings during training (the code samples σ · η with $\pmb { \eta } \sim \mathcal { N } ( 0 , I )$ per dimension and adds it to the unstandardized embedding tensor; no per-dimension rescaling). No inference-time noise.

D-LVR. Same backbone and optimizer settings as LVR; the training trajectory itself is a two-phase “late reconstruction release” rather than a continuous anneal. We resume from the LVR baseline checkpoint at step 1500 (where $\lambda _ { \mathrm { l v r } } { = } 0 . 1$ throughout the first 1500 steps) and continue for the remaining 1000 steps with $\lambda _ { \mathrm { l v r } } { = } 0 . 0$ , so the second phase is supervised purely by the CE term on the answer text. This isolates “what happens to the LM when reconstruction pressure is removed midtraining” from any continuous-schedule sensitivity. We did not run a continuously annealed variant; the schedule in the code applies a fixed $\lambda _ { \mathrm { l v r } }$ throughout each phase.

P-LVR-2 (2-stage). Two latent blocks: context stage on α=1.5× expanded box, target stage on original box. $\lambda _ { \mathrm { l v r , c t x } } ~ = ~ \lambda _ { \mathrm { l v r , t g t } } ~ = ~ 0 . 1$ . No free stage.

P-LVR-3 (3-stage). As P-LVR-2, plus an intermediate free stage with diversity regularizer $\lambda _ { \mathrm { d i v } } = 0 . 0 1$ .

# C.4 Compute Footprint

Each variant trains for ∼ 5–12 GPU-hours on 1 ∼ 2 H100, depending on packing efficiency. All experiments run on the same hardware pool (GPUs 0 and 1 of a shared 8×H100 server). Diagnostic re-runs (probes + faithfulness) take an additional ∼ 1 GPU-hour per variant per benchmark. We use existing checkpoints and a frozen vision tower throughout, which avoids the energy cost of pretraining from scratch.

# D Evaluation Protocol

# D.1 Benchmarks

• V∗Bench (Wu and Xie, 2024): multiple-choice fine-grained visual reasoning, 191 questions total, split into 115 direct\_attributes and 76 relative\_position examples; we report aggregate accuracy.   
• MMVP (Tong et al., 2024): multiple-choice paired-question visual shortcoming probe, 300 pairs.   
• BLINK (Fu et al., 2024): perception-oriented multiple-choice; we evaluate the five validation

subsets used by the codebase (Counting, IQ\_Test, Jigsaw, Relative\_Reflectance, Spatial\_Relation) and report aggregate accuracy across them (697 questions).

# D.2 Inference Configuration

All models are evaluated with greedy decoding, max new tokens 512, and image resolution unchanged from training. The 512-token budget is needed because LVR-mode iterations consume budget before the answer token is emitted; under multi-stage decoding the ctx and tgt stages each emit a quota of latent tokens before any answer text. LVR tokens are generated autoregressively (the model’s own predictions are fed back), matching the inference-time setting under which the model would deploy. The answer letter is parsed from the first letter following the canonical V∗Bench/MMVP/BLINK answer prefix; we use the official benchmark eval scripts where available.

# D.3 Probe and Faithfulness Splits

For the linear-probe experiments, we use the full V∗Bench eval set; stratified 5-fold CV is computed across this set. For faithfulness corruption, we use the same eval set so the clean and corrupted accuracies are directly comparable; the donor pool for swap is also drawn from V∗Bench. We do not introduce a held-out set because the probe/faithfulness signal is computed on top of the existing evaluation, not as a generalization measure.

# D.4 Cosine and MSE Reporting

The cosine and MSE numbers in Table 1 are computed on a held-out validation slice of Visual-CoT (the same slice for all variants) under teacherforced LVR generation. Cosine is the standard cos $\begin{array} { r } { ( \hat { \bf h } _ { t - 1 } , { \bf v } _ { t } ) \ = \ \frac { \hat { \bf h } \cdot { \bf v } } { \| \hat { \bf h } \| \| { \bf v } \| } } \end{array}$ between the predicted LVR-position hidden state and the teacher-forced target visual embedding, averaged over positions — the same metric the LVR literature reports and the quantity the MSE term in $\mathcal { L } _ { \mathrm { L V R } }$ supervises. We apply no centering or other transformation.

# E Responsible Research and Artifact Details

Artifacts. We use Qwen2.5-VL-3B-Instruct, Visual-CoT-438k, V∗Bench, MMVP, and BLINK as existing public research artifacts. We cite their creators in the main text and Appendix D. Our released code documents the expected artifact locations, reproduction commands, and which artifacts are not redistributed.

Artifact licenses and intended use. We use these artifacts for research evaluation and model analysis. We do not redistribute the original training corpus, benchmarks, or pretrained checkpoints. The accompanying code release includes a license file and notes third-party adapted code where applicable.

Data and human subjects. We do not collect new data and do not recruit annotators or human subjects. Our experiments use existing public datasets and benchmarks. We do not perform additional collection of personally identifying information.

Potential risks. The work is diagnostic and interpretive: it analyzes when auxiliary losses produce non-load-bearing latent states. The main foreseeable risk is misuse of the diagnostic conclusion to overgeneralize beyond the studied model, data, and benchmarks; we discuss these scope limitations in the Limitations section.

AI assistance. AI assistants were used for writing, editing, and code/documentation support. All scientific claims, experiments, analyses, and final text were reviewed and verified by the authors.

Code availability. The code is available at https://github.com/xiuyuz/ cosine-misleads. It contains the training, evaluation, interpretability, and audit scripts used in this work.