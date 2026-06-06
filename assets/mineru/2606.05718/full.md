# ViCuR: Visual Cues as Recoverable Privilege for Multimodal On-Policy Distillation

Kanghui Tian\*1,2, Siyuan Liu\*3, Ziang Yan1, Sheng Xia3, Shuai Dong2 and Yi Wang†1

1Shanghai AI Laboratory, 2Fudan University, 3Nanjing University

On-policy distillation (OPD) improves reasoning by training a student on trajectories sampled from its own policy under supervision from a teacher. In multimodal reasoning, a common extension is to use a privileged teacher that observes training-time-only signals such as reference answers or rationales. However, such answerside privilege creates a train-test mismatch: the teacher’s supervision may depend on signals unavailable to the student, encouraging shortcut imitation rather than visually grounded reasoning. We propose ViCuR, a visually grounded privileged-teacher distillation framework that replaces answer-side privilege with visual cues (query-related evidence in the input). Because these cues are derived from the same visual input available at inference, their evidence is recoverable by the student. To support this, ViCuR introduces a lightweight cue recovery module that uses dedicated sink-token cross-attention during prefill to aggregate task-relevant visual evidence into an internal representation, without changing the inference interface or requiring auxiliary cue-generation losses. Across seven benchmarks with Qwen3-VL-2B and 8B students, ViCuR consistently improves over answer-based on-policy self-distillation by +1.19 and +1.24 on overall average performance. It also extends naturally to stronger-teacher OPD, surpassing OPD baselines by +0.64 and +1.08, with consistent out-of-domain gains at the 8B scale. These results show that, in multimodal on-policy distillation, the design of teacher privilege is as important as teacher strength.

# 1. Introduction

Multimodal reasoning tasks such as geometry problem solving [3, 9], chart reasoning [26, 39], and visual question answering [4, 7, 34] require models to integrate vision with multi-step reasoning. Improving such capabilities through on-policy learning is appealing because supervision is applied on trajectories induced by the student’s own policy, directly targeting the states and errors the model encounters at deployment. Onpolicy distillation (OPD) has therefore emerged as a practical way to improve reasoning: a teacher policy supervises student-sampled trajectories, helping the student learn better generation behavior under its own rollout distribution.

Teacher advantage in OPD can arise from two different sources. In the standard setting, teacher and student observe the same input, and the teacher is stronger. In a more general privileged-teacher setting, the teacher is additionally conditioned on training-only information unavailable to the student at inference. This setting is common in on-policy self-distillation (OPSD), where teacher and student share the same backbone but the teacher is strengthened with extra signals such as reference answers, verified rationales, or environment feedback [13, 41]. While privileged teachers can provide stronger supervision, they also raise a design challenge: what kind of privileged information is helpful for distillation in multimodal reasoning?

Existing multimodal OPSD methods commonly use answer- or rationale-based privilege [20, 38]. Concretely, given standard inference input $\boldsymbol { z } = ( x _ { v } , x _ { q } )$ （2号 consisting of visual input $x _ { v }$ and question $x _ { q } ,$ , the student acts as $\pi _ { S } ( { \bf \cdot } { \bf \sigma } | z )$ , while the teacher is conditioned on additional privileged signal $p ,$ yielding $\pi _ { T } ( \cdot \mid z , p )$ . When $p$ contains reference answers or rationales, teacher supervision depends on answer-side information that is unreachable at inference, creating a privilege-induced train-test mismatch: the student may fit answer-aware patterns rather than learning reasoning grounded in visual evidence [36]. This issue is particularly problematic in multimodal reasoning, where correctness should be supported by evidence in the visuals, such as geometric relations in diagrams, or salient events in videos. In other words, the most useful privilege for multimodal reasoning should strengthen supervision while remaining aligned with information the student can access at inference.

In this work, we propose ViCuR (Visual Cue Recovery), a visually grounded privileged-teacher distillation framework. It replaces answer-based privileges with visual cues: question-relevant evidence grounded in the input image or video. Although the cue text itself is provided only to the teacher during training, its underlying source remains accessible to the student through the standard visualquestion interface. This changes the role of privilege from revealing answer-side information to highlighting inference-recoverable evidence, mitigating the privileged-information mismatch. In conventional answer-based distillation, the teacher policy ???? (· | ??, ??) depends on privileged variables ?? that are not recoverable from the student’s input ??. By contrast, when the privilege is a visual cue ?? derived from the visual input and question, $c \sim p ( c \mid x _ { v } , x _ { q } )$ , the teacher is conditioned on evidence whose source is already contained in ??. ViCuR thus does not eliminate privilege altogether, but replaces inaccessible answer privilege with recoverable visual privilege (§3.1).

![](images/2dd0e3d17d1e3cc0d39ccc91a566bce13b07ad4c51234e627432d6753526c742.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Training"] --> B["Teacher (privileged)"]
    B --> C["Teacher-side supervision"]
    C --> D["Student"]
    D --> E["Inference"]
    E --> F["Train-test mismatch"]
    E --> G["Leakage risk"]
    E --> H["Suboptimal performance"]
    
    subgraph Conventional OPSD
        I["Image I"] --> J["Question Q"]
        K["Answer/Rationale R"] --> L["× conditioning mismatch unavailable privilege R"]
        M["ps(· | X, R)"] --> N["× conditioning mismatch unavailable privilege R"]
        O["Learn from teacher's distribution conditioned on privilege"] --> P["Learn from teacher's distribution conditioned on privilege"]
    end
    
    subgraph Our Visual-Cue-Recovery Distillation
        Q["Image I"] --> R["Question Q"]
        S["Visual cue text S"] --> T["recoverable alignment grounded in S = f(X)"]
        U["Image I"] --> V["Question Q"]
        W["Visual cue recovery (SinkTrack)"] --> X["Recoverable during inference"]
        Y["Visual cuehathat S = g(V, Q; θsink)"] --> Z["Recoverable during inference"]
        AA["✓ Inference-accessible evidence"] --> AB["Reduced mismatch"]
        AC["✓ External interface unchanged"] --> AD["Recoverable during inference"]
    end
    
    style Conventional OPSD fill:#f9f9f9,stroke:#333
    style Our Visual-Cue-Recovery Distillation fill:#e6f7ff,stroke:#333
```
</details>

Figure 1: Conventional OPSD conditions teacher supervision on answer/rationale privilege ??, inducing a train– test mismatch. Our ViCuR instead uses visually grounded cues ??, allowing the student to recover corresponding evidence internally while preserving the standard inference interface.

Replacing the privilege type introduces a new challenge: the student does not observe cues explicitly, so it must learn to exploit cue-relevant evidence from visual tokens. To address this, we introduce a lightweight cue recovery module. Inspired by the attention-sink mechanism [21, 35], we equip a designated sink token with dedicated cross-attention parameters at selected transformer layers; during prefill, this token aggregates task-relevant information from visual tokens into an internal cue-level representation. The module is trained end-to-end under the distillation objective alone, preserving the standard inference interface with no autoregressive overhead (§3.2).

Though our primary focus is privileged-teacher selfdistillation, teacher advantage from model scale and from privilege design are orthogonal, so ViCuR extends naturally to stronger-teacher OPD. We evaluate on seven multimodal reasoning benchmarks using Qwen3-VL-2B and 8B students. ViCuR consistently outperforms answer-based OPSD, improving overall average performance by +1.19 and +1.24, with especially strong gains on in-domain and near-domain mathematical reasoning tasks. It also extends effectively to stronger-teacher OPD, where it improves over OPD baselines by +0.64 and +1.08 and delivers consistent out-of-domain gains at the 8B scale.

Our main contributions are as follows:

• We identify the bottleneck of existing multimodal OPSD: answer-based privileges induce a train-test mismatch that can encourage shortcut imitation rather than visually grounded reasoning.   
• We propose ViCuR, which replaces answer-side privilege with visually grounded cues and introduces a lightweight sink-token cross-attention module for internal cue recovery, without changing the inference interface or requiring auxiliary objectives.   
• We show that ViCuR consistently improves over answer-based OPSD across seven benchmarks and extends naturally to stronger-teacher OPD, where

it further benefits from teacher scaling.

# 2. Related Work

On-Policy Distillation and Self-Distillation. Knowledge distillation [10, 15, 32] transfers reasoning capabilities from a large teacher to a compact student. For reasoning tasks, recent work extends distillation from offline imitation of teacher-generated traces to OPD, where the teacher supervises trajectories sampled from the student itself, improving alignment with the student’s inference-time distribution [1, 8, 16, 22]. Standard OPD typically requires a stronger external teacher. OPSD removes this requirement by using a single backbone for both teacher and student, instead strengthening the teacher with privileged conditioning such as verified reasoning traces [41] or feedbackderived supervision [13, 17]. While OPSD avoids the cost of a separate teacher, its effectiveness depends on the design of the privileged signal, which we discuss in the context of privileged information below.

Privileged Information for Distillation. Learning with privileged information [33] augments training with signals that are unavailable at test time. In reasoning distillation, common privilege types include gold answers, rationales, verifier scores, and environment feedback [11, 13, 41]. Recent self-distillation work has begun exploring answer-agnostic alternatives, such as conciseness instructions [28], contextconditioned prompts or historical traces [37], and evidence-centered regional crops [38]. However, when privileges are answer-dependent, they can cause privileged information leakage [36]: formally showed that the teacher’s supervision may depend on variables the student cannot access at inference time, encouraging the student to fit answer-aware patterns rather than learn genuine reasoning.

Attention Sinks. Attention sinks refer to the phenomenon that special tokens such as <BOS> attract disproportionately high attention mass across transformer layers [35], effectively serving as persistent information anchors [19]. SinkTrack [21] leverages this property by using the sink token to aggregate and retain contextual information during long-sequence generation, improving context fidelity without architectural changes.

# 3. Method

We study multimodal on-policy distillation in the privileged-teacher regime, where the teacher accesses additional training-time information unavailable to the student at inference. Our method, ViCuR, is built on a simple principle: privileged information is most useful when grounded in evidence that remains accessible at test time. We first formalize the privilegeinduced mismatch and explain why visual cues mitigate it (§3.1), then present the ViCuR architecture and training objective (§3.2–§3.4).

# 3.1. Problem Formulation

Consider a multimodal reasoning task with visual input $I ,$ question $Q ,$ , and answer ??. Let $X = ( I , Q )$ denote the standard multimodal input available at both training and inference. In OPD, the student samples a trajectory $\hat { y } \sim p _ { \theta } ( \cdot \mid X )$ , and the teacher provides token-level supervision on this student-generated trajectory. We consider the privileged-teacher setting, in which the teacher is additionally conditioned on a training-time-only variable $p ,$ yielding $p _ { T } ( y \mid X , p )$ , while the student uses only ??. This formulation subsumes both standard same-input OPD, where $p = \varnothing ,$ and privileged distillation settings, where $p$ may contain extra side information. In existing multimodal OPSD, this privilege is commonly answer- or rationalebased; in ours, it is instead a visually grounded cue.

Privilege-Induced Gap in Answer-Based Distillation. In OPSD, the teacher is conditioned on an answer- or rationale-based privilege $R ,$ producing $p _ { T } ( y \mid X , R )$ . Because ?? is unavailable at inference time, the teacher’s next-token distribution may depend on information the student cannot access during deployment. Following [36], we quantify this mismatch by the conditional mutual information as $I ( Y _ { t } ; R \ )$ $X , Y _ { < t } )$ . When ?? contains answer-dependent information, such as a gold answer or reference rationale, this quantity is generally nonzero. Distillation then asks the student to imitate a target policy whose behavior is not fully determined by the input ??, creating a privilege-induced train-test mismatch. In multimodal reasoning, this mismatch is problematic because it encourages the student to fit answer-aware supervision patterns or shortcuts rather than learning to ground its reasoning in visuals.

Visual Cues as Recoverable Privilege. We replace answer-based privilege ?? with a visually grounded cue ??: a description of question-relevant evidence, such as geometric relations or key events. Although the student does not receive cue text explicitly at inference, the evidence from which ?? is derived remains present in the standard input ??.

For analysis, we idealize cue construction as a deterministic mapping $S = f ( X )$ , where $f$ extracts cue text from the inference-time input. Under this abstraction, $H ( S \mid X ) = 0 .$ , so conditioning on $S$ introduces no additional information beyond ??:

$$
I (Y _ {t}; S \mid X, Y _ {<   t}) = 0. \tag {1}
$$

Thus, the cue-conditioned teacher distribution

$$
p _ {T} (\cdot \mid X, S, Y _ {<   t}) = p _ {T} (\cdot \mid X, f (X), Y _ {<   t}) \tag {2}
$$

depends only on information determined by the inference-time multimodal input.

We formalize this as Proposition 1 (Appendix F.1): under the deterministic abstraction $S { = } f ( X )$ , visually grounded cues do not introduce the privilege-induced conditional information gap associated with answerdependent variables, as the resulting teacher supervision depends only on information determined by the inference-time input ??.

In practice, cue text may be generated by an external model or annotator, so ?? need not be strictly deterministic given ??, yielding $H ( S \mid X ) > 0$ . However, because visual cues describe evidence visible in the input rather than answer-dependent information, the inaccessible component of cue-based privilege is substantially smaller than that of answer-based privilege. The deterministic abstraction should therefore be interpreted as an idealized limit: visual cues reduce the privilege gap rather than eliminating it entirely, and we show empirically that this reduction is sufficient to yield consistent gains.

Proposition 1 does not require the student to reconstruct cue text explicitly. Instead, it reframes the remaining challenge as evidence recovery: the relevant information is present in ??, but the student must learn to extract and represent it internally.

# 3.2. ViCuR: Sink-Based Cue Recovery

To address this challenge, we introduce ViCuR (Visual Cue Recovery), a lightweight student-side module that recovers cue-relevant evidence from visual representations without changing the standard multimodal inference interface. A naive alternative would be to append cue text to the student input or require the student to generate cues before answering. However, this would alter the inference pipeline and add generation overhead. We instead seek a mechanism that operates within the student’s existing architecture.

Our framework uses an asymmetric teacher-student design. The teacher is conditioned on the visual input, question, and cue text, producing $p _ { T } ( y \mid I , Q , S )$ . The student predicts with $p _ { \theta } ( y \mid I , Q )$ , augmented with a cue recovery module.

Let $V ^ { ( 0 ) } = \phi ( I ) \in \mathbb { R } ^ { m \times d }$ be the visual token sequence from the visual encoder. At transformer layer $\ell ,$ let $V ^ { ( \ell ) } \in \mathbb { R } ^ { m \times d }$ denote the current visual representations, and let ℎ(ℓ)sink $h _ { \mathrm { s i n k } } ^ { ( \ell ) } \in \mathbb { R } ^ { d }$ denote the hidden state of a designated sink token (e.g., <BOS>). As shown in $\mathrm { F i g . ~ , ~ }$ at selected layers during prefill, the sink token queries the visual tokens through a dedicated crossattention branch:

![](images/a861162b41abc2f62aaadab264009556cf4ea40fb2dce36792ebe22e393c3108.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Adapted SinkTrack at a Selected Transformer Layer"] --> B["Sink Token h_sink^(l) (<BOS>)<br>    q^(l) = h_sink^(l) W_Q^(l)"]
    B --> C["SinkTrack Cross-Attention"]
    C --> D["z_sink^(l) + w_O^(l)"]
    D --> E["Recovered cue-level representation<br>    Ŝ = g(V, Q; θ_sink)"]
    E --> F["Current-layer visual states V^(l)"]
    F --> G["..."]
    
    H["Visual tokens / states"] --> B
    I["Sink token / representations"] --> B
    J["SinkTrack components"] --> B
    
    style A fill:#d4edda,stroke:#333
    style B fill:#e6f7ff,stroke:#333
    style C fill:#e6f7ff,stroke:#333
    style D fill:#e6f7ff,stroke:#333
    style E fill:#e6f7ff,stroke:#333
    style F fill:#d4edda,stroke:#333
    style G fill:#d4edda,stroke:#333
    style H fill:#d4edda,stroke:#333
    style I fill:#d4edda,stroke:#333
    style J fill:#d4edda,stroke:#333
```
</details>

Figure 2: Cue recovery module at a selected transformer layer. The sink token aggregates visual states via dedicated cross-attention during prefill, producing cue representations without autoregressive overhead.

$$
q ^ {(\ell)} = h _ {\mathrm{sink}} ^ {(\ell)} W _ {Q} ^ {(\ell)}, K ^ {(\ell)} = V ^ {(\ell)} W _ {K} ^ {(\ell)},
$$

$$
U ^ {(\ell)} = V ^ {(\ell)} W _ {V} ^ {(\ell)}, \tag {3}
$$

$$
z _ {\mathrm{sink}} ^ {(\ell)} = \mathrm{Attn} \big (q ^ {(\ell)}, K ^ {(\ell)}, U ^ {(\ell)} \big) W _ {O} ^ {(\ell)},
$$

where ?? (ℓ), $W _ { Q } ^ { ( \ell ) } , W _ { K } ^ { ( \ell ) } , W _ { V } ^ { ( \ell ) } , W _ { O } ^ { ( \ell ) }$ are recovery-modulespecific parameters (we use $U ^ { ( \ell ) }$ for the value projection to avoid conflict with visual tokens $V ^ { ( \ell ) } )$ . The aggregated evidence is added back to the sink state via a residual connection:

$$
\tilde {h} _ {\mathrm{sink}} ^ {(\ell)} = h _ {\mathrm{sink}} ^ {(\ell)} + z _ {\mathrm{sink}} ^ {(\ell)}. \tag {4}
$$

Because the sink token appears at the beginning of the sequence, its updated state is visible to all subsequent question and answer tokens through standard causal attention. We denote the resulting internal evidence representation as

$$
\hat {S} = g (V, Q; \theta_ {\mathrm{sink}}), \tag {5}
$$

where $\theta _ { \mathrm { s i n k } }$ collects all recovery-module parameters. The symbol ??ˆ refers to an internal hidden-state representation, not generated cue text. Although the cross-attention equations above involve only $h _ { \mathrm { s i n k } } ^ { ( \bar { \ell } ) }$ and $V ^ { ( \ell ) }$ , the question $Q$ influences $\hat { S }$ implicitly through optimization: gradients from the question-dependent prediction loss guide $\theta _ { \mathrm { s i n k } }$ to selectively aggregate visual features from $V ^ { ( \ell ) }$ that are most relevant for answering $Q .$ .

Why Dedicated Cross-Attention. We introduce dedicated cross-attention parameters, which are free to learn question-relevant visual selection without interfering with the pretrained self-attention pathway and can absorb the implicit supervision signal from the distillation objective. This differs from the original SinkTrack mechanism [21], which reuses the existing attention computation for generic context anchoring.

# 3.3. Training Objective

The cue recovery module is trained jointly with the student under the on-policy distillation objective, without any auxiliary cue-generation or text-matching loss. For a student-sampled trajectory $\hat { y } \sim p _ { \theta } ( \cdot \mid X )$ , the teacher evaluates each token with $p _ { T } ( \hat { y } _ { n } \mid X , S , \hat { y } _ { < n } )$ , while the student assigns $p _ { \theta } ( \hat { y } _ { n } \mid X , \hat { S } , \hat { y } _ { < n } )$ . Following sampled-token on-policy distillation [18], we define the token-level advantage

$$
\begin{array}{l} A _ {n} (X, \hat {y}) = \log p _ {T} (\hat {y} _ {n} \mid X, S, \hat {y} _ {<   n}) \\ - \log p _ {\theta} (\hat {y} _ {n} \mid X, \hat {S}, \hat {y} _ {<   n}), \tag {6} \\ \end{array}
$$

which is treated as a stop-gradient constant, as in standard policy-gradient optimization [14, 22]. The distillation surrogate loss is

$$
\mathcal {L} _ {\mathrm{d}} (\theta) = - \mathbb {E} _ {X \sim \mathcal {D}} \mathbb {E} _ {\hat {y} \sim p _ {\theta} (\cdot | X)} \Bigg [ \tag {7}
$$

$$
\frac {1}{| \hat {y} |} \sum_ {n = 1} ^ {| \hat {y} |} \mathrm{sg} [ A _ {n} (X, \hat {y}) ] \times \log p _ {\theta} (\hat {y} _ {n} \mid X, \hat {S}, \hat {y} _ {<   n}) \Biggr ],
$$

where sg[·] denotes stop-gradient. Here we use a PPOstyle policy loss [29] with importance ratios and clipping [25]; the expression above isolates the gradient pathway through which the recovery module is trained.

Since $\hat { S } = g ( V , Q ; \theta _ { \mathrm { s i n k } } )$ , gradients reach the recovery module through

$$
\nabla_ {\theta_ {\mathrm{sink}}} \mathcal {L} _ {\mathrm{d}} = - \mathbb {E} \left[ \frac {1}{| \hat {y} |} \sum_ {n = 1} ^ {| \hat {y} |} \operatorname{sg} [ A _ {n} ] \right. \tag {8}
$$

$$
\nabla_ {\hat {S}} \log p _ {\theta} (\hat {y} _ {n} \mid X, \hat {S}, \hat {y} _ {<   n}) \cdot \nabla_ {\theta_ {\mathrm{sink}}} g (V, Q; \theta_ {\mathrm{sink}}) \Biggr ].
$$

This expression shows that the recovery module is updated when its representation ??ˆ affects the student’s prediction on teacher-evaluated tokens. Whenever the teacher assigns higher probability than the student to the sampled token, the gradient encourages the student to move its representation in a direction that better supports teacher-aligned behavior; when the opposite holds, the gradient suppresses unhelpful representations. In this way, the distillation objective implicitly encourages the sink-based module to retain task-relevant visual evidence aligned with the teacher’s cue-conditioned supervision, even though the student never observes cue text directly.

A detailed analysis of these gradient dynamics is provided in Appendix F.2; a full-vocabulary KL counterpart is discussed in Appendix F.3.

# 3.4. Implementation Details

We train the recovery module jointly with the student under the distillation objective described above. The recovery module is inserted every five transformer layers and uses the full current-layer visual token sequence as keys and values. It runs only during prefill and adds no autoregressive decoding overhead, introducing only a modest parameter overhead relative to the base model.

# 4. Experiments

We evaluate ViCuR to answer three questions: (1) Does replacing answer-based privilege with visual cues improve over conventional distillation baselines, including both same-backbone OPSD and strongerteacher OPD? (2) Do the gains generalize across in-domain, near-domain, and out-of-domain benchmarks? (3) Which components of ViCuR matter most, and how does the method behave as teacher strength increases?

Data. We train on the Vision R1 training set [12], from which teacher-side visual cue text is constructed. Evaluation covers three generalization levels: Vision R1-Test [12] (in-domain); DynaMath [42], MathVista [23], WeMath [27], and MathVerse [40] (near-domain mathematical reasoning); and MMMU-Val [39] with Video-MME [6] (out-of-domain multimodal understanding). Geometry3K [24] is reserved for ablation studies due to its smaller scale and diagram-centric nature.

Models and Baselines. We use Qwen3-VL-2B-Instruct (2B) and Qwen3-VL-8B-Instruct (8B) [2] as student backbones. In OPSD [41], the teacher shares the student’s backbone; in OPD [22], the teacher is larger (Qwen3-VL-8B for 2B, Qwen3-VL-32B for 8B). We compare: base model, GRPO [30] (rewardbased, no teacher), OPSD, OPSD+ViCuR, OPD, and OPD+ViCuR. Our primary comparisons are against OPSD and OPD, isolating the effect of visual-cue privilege and cue recovery. Details are in Appendix A.

# 4.1. Main Results

Tab. 1 reports results across all benchmarks for both 2B and 8B students. We compare ViCuR against the corresponding distillation baseline under the same teacher configuration, and then discuss how the gains vary across generalization regimes.

ViCuR Improves Over Answer-based OPSD. Under same-backbone self-distillation, OPSD+ViCuR raises the average by +1.19 (2B) and +1.24 (8B) over conventional answer-based OPSD. The gains are strongest on benchmarks with clear visual grounding demands, such as WeMath (+3.33 at 2B, +2.67 at 8B), supporting our hypothesis that visually grounded cues yield more useful supervision than answer-side privilege.

<table><tr><td rowspan="2">Method</td><td>In-Domain</td><td colspan="4">Near-Domain</td><td colspan="2">Out-of-Domain</td><td>Overall</td></tr><tr><td>Vision R1-Test</td><td>DynaMath</td><td>MathVista</td><td>WeMath</td><td>MathVerse</td><td>MMMU-Val</td><td>Video-MME</td><td>Avg.</td></tr><tr><td colspan="9">Qwen3-VL-2B-Instruct</td></tr><tr><td>Base Model</td><td>23.90</td><td>53.23</td><td>61.0</td><td>31.43</td><td>39.09</td><td>51.11</td><td>58.2</td><td>45.42</td></tr><tr><td>GRPO</td><td>28.17</td><td>55.49</td><td>60.8</td><td>40.19</td><td>44.90</td><td>52.00</td><td>58.0</td><td>48.51</td></tr><tr><td>OPSD</td><td>25.00</td><td>52.37</td><td>61.2</td><td>30.57</td><td>37.56</td><td>49.56</td><td>57.9</td><td>44.88</td></tr><tr><td>OPSD+ViCuR</td><td>26.58(+1.58)</td><td>51.60(-0.77)</td><td>63.4(+2.2)</td><td>33.90(+3.33)</td><td>40.00(+2.44)</td><td>49.22(-0.34)</td><td>57.8(-0.1)</td><td>46.07(+1.19)</td></tr><tr><td>OPD</td><td>26.34</td><td>52.87</td><td>64.4</td><td>35.71</td><td>40.56</td><td>51.00</td><td>57.6</td><td>46.93</td></tr><tr><td>OPD+ViCuR</td><td>26.83(+0.49)</td><td>52.38(-0.49)</td><td>63.6(-0.8)</td><td>36.00(+0.29)</td><td>42.31(+1.75)</td><td>53.30(+2.30)</td><td>58.6(+1.0)</td><td>47.57(+0.64)</td></tr><tr><td colspan="9">Qwen3-VL-8B-Instruct</td></tr><tr><td>Base Model</td><td>41.22</td><td>67.13</td><td>75.8</td><td>55.33</td><td>53.40</td><td>66.56</td><td>65.9</td><td>60.76</td></tr><tr><td>GRPO</td><td>58.75</td><td>67.44</td><td>76.6</td><td>67.05</td><td>56.22</td><td>65.56</td><td>65.8</td><td>65.35</td></tr><tr><td>OPSD</td><td>42.20</td><td>65.99</td><td>73.5</td><td>43.62</td><td>52.92</td><td>62.33</td><td>66.5</td><td>58.15</td></tr><tr><td>OPSD+ViCuR</td><td>45.00(+2.80)</td><td>67.17(+1.18)</td><td>74.1(+0.6)</td><td>46.29(+2.67)</td><td>53.25(+0.33)</td><td>63.33(+1.00)</td><td>66.6(+0.1)</td><td>59.39(+1.24)</td></tr><tr><td>OPD</td><td>51.09</td><td>69.22</td><td>78.1</td><td>58.38</td><td>60.05</td><td>64.33</td><td>66.0</td><td>63.88</td></tr><tr><td>OPD+ViCuR</td><td>52.07(+0.98)</td><td>70.70(+1.48)</td><td>76.7(-1.4)</td><td>60.57(+2.19)</td><td>61.55(+1.50)</td><td>66.56(+2.23)</td><td>66.6(+0.6)</td><td>64.96(+1.08)</td></tr></table>

Table 1: Main results across in-domain, near-domain, and out-of-domain benchmarks. ViCuR consistently improves the overall average over the corresponding distillation baseline (OPSD or OPD) at both 2B and 8B scales. Deltas in parentheses are relative to the corresponding baseline without ViCuR.

Notably, conventional OPSD underperforms the base model at both scales (44.88 vs. 45.42 for 2B; 58.15 vs. 60.76 for 8B), consistent with prior analyses of answer-conditioned self-distillation [36]. ViCuR mitigates this degradation, surpassing the base model at 2B (46.07 vs. 45.42) and substantially narrowing the gap at 8B. A few benchmarks show small decreases at 2B scale (DynaMath -0.77, Video-MME -0.1), which we attribute to limited student capacity for tasks requiring broader numerical or temporal reasoning beyond the geometry-heavy training distribution; these drops disappear or reverse at 8B, supporting this interpretation.

ViCuR Improves Stronger-Teacher OPD. ViCuR extends to stronger-teacher OPD, improving the average by +0.64 at 2B and +1.08 at 8B. Gains are broadly distributed across five of seven (2B) and six of seven (8B) benchmarks. MathVista is the exception, with small drops at both scales (-0.8 and -1.4); analysis shows ViCuR improves geometry-heavy subsets but is slightly weaker on computation-dominated subsets (arithmetic reasoning, math word problems), where the bottleneck is numerical processing rather than visual grounding.

ViCuR Improves Transfer Beyond Training Domain. The strongest out-of-domain improvements appear in the OPD setting. In particular, OPD + ViCuR improves MMMU-Val by +2.30 at 2B and +2.23 at 8B, and improves Video-MME at both scales. This suggests reducing answer-side mismatch helps stronger teachers transfer more effectively beyond the mathematical reasoning training distribution, improving broader multimodal understanding and video reasoning as well.

# 4.2. Ablation Studies

We next analyze which components of ViCuR drive the gains and how the method behaves under different teacher-student configurations.

Component Analysis. Table 2 isolates the contributions of visual cues and the cue recovery module in the OPSD setting on Geometry3K. Replacing answer-based privilege with visual cues yields the largest individual improvement: +2.80 for the 2B student and +4.36 for the 8B student. This indicates that the dominant benefit comes from redesigning the teacher-side privilege. By contrast, adding the cue recovery module alone produces limited gains (+1.17 for 2B and +0.09 for 8B), as expected, because the student-side module is still trained against answer-conditioned teacher supervision in this variant. The full combination achieves the best results at both scales: +4.80 for 2B and +4.65 for 8B. This confirms that the two components are complementary: visual cues improve the quality of teacher supervision and cue recovery helps the student internalize the corresponding evidence.

The recovery module adds 100.7M parameters for the 2B model (4.52%) and 536.9M for the 8B model (5.77%). Despite this, it operates only during prefill and introduces no autoregressive decoding overhead: per-token decoding speed is unaffected, and wall-clock training time remains comparable to standard OPSD (Appendix B.4 provides detailed training and inference timing). To verify that the trend is not specific to Geometry3K, we further evaluate on geometry subsets of MathVista and DynaMath, observing similar patterns. Detailed results are in Appendix B.1.

![](images/5ba474de7615bd4d5c480c40b9d7fac8830878dbbfde0a0022f0278427cb10f3.jpg)

<details>
<summary>line</summary>

| Step | OPSD baseline acc | OPSD baseline gap | OPD (8B teacher) - ViCuR acc | OPD (8B teacher) - ViCuR gap | OPD (32B teacher) - ViCuR acc | OPD (32B teacher) - ViCuR gap |
|------|-------------------|-------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| 0    | 0.75              | 0.72              | 0.90                         | 0.85                         | 0.70                         | 0.65                         |
| 200  | 0.72              | 0.70              | 0.85                         | 0.80                         | 0.65                         | 0.60                         |
| 400  | 0.70              | 0.68              | 0.80                         | 0.75                         | 0.60                         | 0.55                         |
| 600  | 0.68              | 0.66              | 0.75                         | 0.70                         | 0.55                         | 0.50                         |
| 800  | 0.66              | 0.64              | 0.70                         | 0.65                         | 0.50                         | 0.45                         |
| 1000 | 0.64              | 0.62              | 0.65                         | 0.60                         | 0.45                         | 0.45                         |
| 1200 | 0.62              | 0.60              | 0.60                         | 0.55                         | 0.45                         | 0.45                         |
</details>

![](images/96712155db0102b08b51ca8e7ff98328c22a3fff8bef1bb3f66ff5e3f1ff68a0.jpg)

<details>
<summary>line</summary>

| Teacher-student Log-prob Gap | OPSD (Front 50%) baseline gap | OPSD (Back 50%) ViCuR gap | OPD (Front 50%) baseline gap | OPD (Back 50%) ViCuR gap | OPD (Front 50%) ViCuR gap | OPD (Back 50%) ViCuR gap |
| ----------------------------- | ------------------------------ | -------------------------- | ------------------------------ | -------------------------- | -------------------------- | -------------------------- |
| 28 Student                    | ~0.80                          | ~0.68                      | ~0.75                          | ~0.65                      | ~0.70                      | ~0.60                      |
| 88 Student                    | ~0.65                          | ~0.45                      | ~0.50                          | ~0.35                      | ~0.40                      | ~0.25                      |
</details>

Figure 3: Training dynamics of student models under different teacher configurations and distillation paradigms. (a) Overall training dynamics. (b) Stage-wise training dynamics, where student-generated sequences are split into the front 50% and back 50% of tokens. For both OPSD and OPD, we use the answer-based method as the baseline. 

<table><tr><td rowspan="2">Method</td><td colspan="2">2B Student</td><td colspan="2">8B Student</td></tr><tr><td>Accuracy</td><td> $\Delta$ </td><td>Accuracy</td><td> $\Delta$ </td></tr><tr><td>Baseline</td><td>34.14</td><td>-</td><td>52.40</td><td>-</td></tr><tr><td>w/ visual cue</td><td>36.94</td><td>+2.80</td><td>56.76</td><td>+4.36</td></tr><tr><td>w/ cue recovery</td><td>35.31</td><td>+1.17</td><td>52.49</td><td>+0.09</td></tr><tr><td>ViCuR</td><td>38.94</td><td>+4.80</td><td>57.05</td><td>+4.65</td></tr></table>

Table 2: Component analysis under the OPSD setting on Geometry3K-test. Accuracy is reported as the average over the final 100 training steps. ∆ denotes the absolute improvement over the corresponding baseline.

Scaling Behavior across Teacher-Student Configurations. We further examine how ViCuR behaves as student capacity and teacher strength vary on Geometry3K (Fig. 5 in Appendix B.2). ViCuR improves all four teacher-student configurations, with gains ranging from +3.35 to +6.88. The largest gain (+6.88) occurs for the 8B student distilled from a 32B teacher, substantially exceeding the self-distillation gain (+4.65), suggesting visual-cue grounding and teacher scaling are complementary when the student is large enough. For the 2B student, the 8B teacher does not produce a larger gain than self-distillation (+3.35 vs. +4.80), indicating that once the student becomes the bottleneck, additional teacher strength is harder to exploit.

![](images/529967141138c5735c5725b6bb658f694ae76ee1828b57f5c446c7c63b95595e.jpg)

<details>
<summary>line</summary>

| Epoch | OPSD 2B | OPSD+VICR 2B | OPSD 8B | OPSD+VICR 8B |
|-------|---------|--------------|---------|--------------|
| 1     | 0       | 0            | 0       | 0            |
| 2     | 800     | 300          | 200     | 100          |
| 3     | 3000    | 1200         | 2500    | 500          |
| 4     | 3800    | 2800         | 3500    | 1000         |
| 5     | 4000    | 3200         | 3600    | 1500         |
</details>

Figure 4: Hint leakage under OPSD, measured by affected sample count and total “Hint” occurrences across epochs. Lower values indicate less leakage.

Token-Level Log-Probability Dynamics. To better understand training behavior, we track the teacherstudent token-level log-probability gap together with student test accuracy on Geometry3K throughout training. We report both the full sequence and a stage-wise split into the front and back 50% of tokens (Fig. 3).

Fig. 3a shows that ViCuR often begins with a larger teacher-student gap than answer-based baselines, but later reaches parity or achieves a lower gap while maintaining higher test accuracy under both OPSD and OPD. This indicates that cue-conditioned supervision is not easier to imitate at the outset; instead, it appears to provide better learning signals, ultimately leading to stronger student behavior.

Fig. 3b further shows that both baseline methods and ViCuR have smaller gaps on the back half of the sequence than on the front half, suggesting that student alignment is generally easier near answer-proximal tokens. More importantly, ViCuR consistently achieves a smaller back-half gap than the answer-based baseline. This suggests that grounding the student in relevant visual evidence improves not only early reasoning but also downstream answer-stage alignment. Additional plots are provided in Appendix B.3.

Hint Leakage in Student Rollouts. To probe privilege leakage in student behavior, we count explicit occurrences of “Hint” in student rollouts. Following the teacher prompt design of prior work [41], privileged teacher-side content is injected using a [Hint] format. Student-side generation of “Hint” serves as a direct diagnostic of training-only prompt-pattern reproduction.

Fig. 4 shows that answer-based OPSD exhibits steadily increasing hint leakage over training, whereas ViCuR consistently reduces both the number of affected samples and the total number of “Hint” occurrences. Although this lexical diagnostic does not capture all forms of shortcut learning, it provides direct behavioral evidence that visual-cue supervision reduces explicit dependence on teacher-side privileged patterns. Full statistics, including the OPD setting, are provided in Appendix D.

Qualitative Evidence and Attention Visualization. To complement the quantitative results, we provide case studies and SinkTrack cross-attention visualizations in Appendix E. These examples show that Vi-CuR reduces concrete visual grounding errors induced by answer-based OPSD (e.g., incorrect label-segment binding in geometry diagrams) and that the sinkbased recovery module aggregates visual tokens consistent with teacher-side cue evidence, supporting the claimed mechanism.

# 5. Conclusion

We presented ViCuR, which replaces answer-based privilege in multimodal on-policy distillation with visually grounded cues and equips the student with a lightweight sink-based recovery module for internal evidence aggregation. Across seven benchmarks and two student scales, ViCuR consistently improves over answer-based OPSD and extends effectively to stronger-teacher OPD, with the strongest gains on visually grounded reasoning tasks. Our results suggest that in multimodal distillation, designing privilege around inference-recoverable visual evidence is as important as teacher strength itself.

# 6. Limitations

Our work focuses on improving on-policy multimodal distillation by replacing answer-based privileged information with visually grounded cues and by introducing an internal cue recovery module. While this design reduces the mismatch caused by answer-based privilege, it still depends on the quality of the constructed visual cues during training. If the cue generator misses task-relevant evidence, introduces non-visual information, or describes the visual scene ambiguously, the teacher-side supervision may become less reliable. In addition, the cue recovery module introduces extra trainable parameters and optimization complexity. The component analysis suggests that its benefit does not increase monotonically across model scales: for the 8B student on Geometry3K, full Vi-CuR performs slightly worse than the visual-cue-only variant, even though ViCuR still improves over the corresponding OPSD/OPD baselines in the main results. This indicates that larger students may require more careful training of the recovery module, such as more parameter-efficient designs, staged optimization, richer cue recovery data, or additional regularization.

# Acknowledgments

We thank the Shanghai Artificial Intelligence Laboratory for their institutional support.

# References

[1] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, 2024. 2   
[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 4   
[3] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, 2021. 1   
[4] Shuai Dong, Siyuan Wang, Xingyu Liu, Chenglin Li, Haowen Hou, and Zhongyu Wei. Interleaved

latent visual reasoning with selective perceptual modeling. arXiv preprint arXiv:2512.05665, 2025. 1   
[5] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, 2024. A.2   
[6] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2025. 4   
[7] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2017. 1   
[8] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In International Conference on Learning Representations, 2024. 2   
[9] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024. 1   
[10] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 2   
[11] Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. In Findings of the Association for Computational Linguistics: ACL 2023, 2023. 2   
[12] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei zhao, Zhe Xu, Xu Tang, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal

large language models. In International Conference on Learning Representations, 2026. 4   
[13] Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas Krause. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802, 2026. 1, 2   
[14] Woogyeol Jin, Taywon Min, Yongjin Yang, Swanand Ravindra Kadhe, Yi Zhou, Dennis Wei, Nathalie Baracaldo, and Kimin Lee. Entropyaware on-policy distillation of language models. arXiv preprint arXiv:2603.07079, 2026. 3.3   
[15] Minki Kang, Seanie Lee, Jinheon Baek, Kenji Kawaguchi, and Sung Ju Hwang. Knowledgeaugmented reasoning distillation for small language models in knowledge-intensive tasks. In Advances in Neural Information Processing Systems, 2023. 2   
[16] Jiaze Li, Hao Yin, Haoran Xu, Boshen Xu, Wenhui Tan, Zewen He, Jianzhong Ju, Zhenbo Luo, and Jian Luan. Video-opd: Efficient posttraining of multimodal large language models for temporal video grounding via on-policy distillation. arXiv preprint arXiv:2602.02994, 2026. 2   
[17] Yang Li, Erik Nijkamp, Semih Yavuz, and Shafiq Joty. Learning from language feedback via variational policy distillation. arXiv preprint arXiv:2605.15113, 2026. 2   
[18] Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, et al. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016, 2026. 3.3   
[19] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation. In Advances in Neural Information Processing Systems, 2024. 2   
[20] Hao Lin, Kunyang Lv, Xu Jiang, Jingqi Tian, Zhongjing Du, Jiayu Ding, Qiaoman Zhang, and Hongbo Jin. Visd: Enhancing video reasoning via structured self-distillation. arXiv preprint arXiv:2605.06094, 2026. 1   
[21] Xu Liu, Guikun Chen, and Wenguan Wang. Sinktrack: Attention sink based context anchoring

for large language models. In International Conference on Learning Representations, 2026. 1, 2, 3.2   
[22] Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. https://thinkingmachines.ai/blog/on-policydistillation. 2, 3.3, 4   
[23] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations, 2024. 4   
[24] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In The Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021), 2021. 4   
[25] Feng Luo, Yu-Neng Chuang, Guanchu Wang, Zicheng Xu, Xiaotian Han, Tianyi Zhang, and Vladimir Braverman. Demystifying opd: Length inflation and stabilization strategies for large language models. arXiv preprint arXiv:2604.08527, 2026. 3.3   
[26] Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, 2022. 1   
[27] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024. 4   
[28] Hejian Sang, Yuanda Xu, Zhengze Zhou, Ran He, Zhipeng Wang, and Jiachen Sun. Crisp: Compressed reasoning via iterative self-policy distillation. arXiv preprint arXiv:2603.05433, 2026. 2   
[29] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 3.3

[30] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 4   
[31] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024. A   
[32] Fangxun Shu, Yue Liao, Le Zhuo, Chenning Xu, Lei Zhang, Guanghao Zhang, Haonan Shi, Long Chen, Tao Zhong, Wanggui He, Siming Fu, et al. Llava-mod: Making llava tiny via moe knowledge distillation. arXiv preprint arXiv:2408.15881, 2024. 2   
[33] Vladimir Vapnik and Rauf Izmailov. Learning using privileged information: similarity control and knowledge transfer. The Journal of Machine Learning Research, 2015. 2   
[34] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, et al. Internvideo2. 5: Empowering video mllms with long and rich context modeling. arXiv preprint arXiv:2501.12386, 2025. 1   
[35] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In International Conference on Learning Representations, 2024. 1, 2   
[36] Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. Selfdistilled rlvr. arXiv preprint arXiv:2604.03128, 2026. 1, 2, 3.1, 4.1, F.1   
[37] Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. On-policy context distillation for language models. arXiv preprint arXiv:2602.12275, 2026. 2   
[38] Qianhao Yuan, Jie Lou, Xing Yu, Hongyu Lin, Le Sun, Xianpei Han, and Yaojie Lu. Visionopd: Learning to see fine details for multimodal llms via on-policy self-distillation. arXiv preprint arXiv:2605.18740, 2026. 1, 2   
[39] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming

Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 1, 4   
[40] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, 2024. 4   
[41] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy selfdistillation for large language models. arXiv preprint arXiv:2601.18734, 2026. 1, 2, 4, 4.2, F.3   
[42] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. In International Conference on Learning Representations, 2025. 4

# A. Training and Evaluation Details

<table><tr><td>Setting</td><td>Vision R1 Distillation</td><td>Vision R1 GRPO</td><td>Geometry3K Ablations</td></tr><tr><td>Training data</td><td>Vision R1 train</td><td>Vision R1 train</td><td>Geometry3K train</td></tr><tr><td>Max prompt length</td><td>2048</td><td>2048</td><td>1024</td></tr><tr><td>Max response length</td><td>4096</td><td>4096</td><td>2048</td></tr><tr><td>Train batch size</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Learning rate</td><td> $1 \times 10^{-6}$ </td><td> $1 \times 10^{-6}$ </td><td> $1 \times 10^{-6}$ </td></tr><tr><td>Rollouts per prompt (n)</td><td>1</td><td>5</td><td>1</td></tr><tr><td>GPU allocation</td><td>4 student + 4 teacher</td><td>8 student</td><td>4 student + 4 teacher</td></tr><tr><td>Training epochs</td><td>5</td><td>5</td><td>100</td></tr></table>

Table 3: Key training configurations for the main Vision R1 experiments and the controlled Geometry3K ablations.

All experiments are implemented on top of the VeRL framework [31]. Unless otherwise specified, experiments are run on a single node with 8 H200 GPUs. For the main Vision R1 experiments, all compared methods are trained on the same training split and evaluated on the same benchmark suite, differing only in the learning objective and supervision source. Geometry3K is used separately for controlled ablation studies. For the Qwen3-VL model family, the first <|im\_start|> token, i.e. the one preceding the system prompt, is selected as the sink token.

# A.1. Training Details

Tab. 3 summarizes the key training configurations for the main Vision R1 experiments and the controlled Geometry3K ablations.

Unless otherwise specified, all distillation-based experiments use the same sampled-token on-policy distillation framework described in the main text. Teacher-side privileged prompts are provided through the teacher\_prompt field, and images are loaded from the images field. In the distillation-based setting, rollouts are generated with a single student sample per prompt, whereas GRPO uses 5 sampled rollouts per prompt under the same prompt/response length budget.

# A.2. Evaluation Details

The models are evaluated on MathVista\_MINI, MathVerse\_MINI, WeMath, DynaMath, MMMU\_DEV\_VAL, and Video-MME\_64frame with the VLMEvalKit framework [5], following the default dataset definitions in VLMEvalKit. Tab. 4 summarizes the key generation and preprocessing settings during evaluation.

<table><tr><td>Evaluation setting</td><td colspan="2">Value</td></tr><tr><td>System prompt</td><td colspan="2">You are a helpful assistant.</td></tr><tr><td>Custom prompt</td><td colspan="2">No</td></tr><tr><td>Temperature</td><td colspan="2">0.7</td></tr><tr><td>Top-p</td><td colspan="2">0.8</td></tr><tr><td>Top-k</td><td colspan="2">20</td></tr><tr><td>Repetition penalty</td><td colspan="2">1.0</td></tr><tr><td>Presence penalty</td><td colspan="2">1.5</td></tr><tr><td>Max new tokens</td><td colspan="2">4096</td></tr><tr><td>Min pixels</td><td colspan="2">3584</td></tr><tr><td>Max pixels</td><td colspan="2">401408</td></tr><tr><td>Min frames</td><td colspan="2">4</td></tr><tr><td>Max frames</td><td colspan="2">512</td></tr></table>

Table 4: Evaluation hyperparameters used in VLMEvalKit.

# B. Ablation Details

# B.1. Main Component Analysis

<table><tr><td rowspan="2">Student</td><td rowspan="2">Setting</td><td colspan="3">Method Components</td><td colspan="2">MathVista</td><td>DynaMath</td></tr><tr><td>Answer</td><td>Visual Cue</td><td>Cue Recovery</td><td>Geo. Reas.</td><td>Geo. Prob.</td><td>Plane Geo.</td></tr><tr><td rowspan="8">2B</td><td>Base</td><td></td><td></td><td></td><td>66.95</td><td>69.23</td><td>50.91</td></tr><tr><td rowspan="3">OPSD</td><td>√</td><td></td><td></td><td>67.36</td><td>68.75</td><td>51.82</td></tr><tr><td></td><td>√</td><td></td><td>73.22</td><td>75.48</td><td>54.55</td></tr><tr><td></td><td>√</td><td>√</td><td>74.48</td><td>76.92</td><td>53.77</td></tr><tr><td rowspan="4">OPD(8B teacher)</td><td>√</td><td></td><td></td><td>73.64</td><td>75.00</td><td>49.22</td></tr><tr><td></td><td>√</td><td></td><td>74.06</td><td>75.48</td><td>48.96</td></tr><tr><td></td><td>√</td><td></td><td>75.73</td><td>78.37</td><td>54.29</td></tr><tr><td></td><td>√</td><td>√</td><td>74.06</td><td>77.40</td><td>51.30</td></tr><tr><td rowspan="8">8B</td><td>Base</td><td></td><td></td><td></td><td>84.10</td><td>86.54</td><td>57.66</td></tr><tr><td rowspan="3">OPSD</td><td>√</td><td></td><td></td><td>85.36</td><td>86.54</td><td>57.03</td></tr><tr><td></td><td>√</td><td></td><td>87.03</td><td>88.46</td><td>59.35</td></tr><tr><td></td><td>√</td><td>√</td><td>85.77</td><td>86.54</td><td>60.00</td></tr><tr><td rowspan="4">OPD(32B teacher)</td><td>√</td><td></td><td></td><td>84.94</td><td>86.54</td><td>61.43</td></tr><tr><td></td><td>√</td><td></td><td>86.61</td><td>87.98</td><td>61.69</td></tr><tr><td></td><td>√</td><td></td><td>88.28</td><td>90.38</td><td>61.82</td></tr><tr><td></td><td>√</td><td>√</td><td>87.87</td><td>89.90</td><td>63.12</td></tr></table>

Table 5: Additional cross-benchmark ablation results. All models are trained on Geometry3K and evaluated on geometry-related subsets of MathVista and DynaMath. We report the geometry reasoning and geometry problem solving subsets for MathVista and the plane geometry subset for DynaMath. The solid-geometry subset of DynaMath is omitted because Geometry3K contains little solid-geometry supervision. Checkmarks indicate whether each method uses answer-based privileged supervision, visual-cue-based supervision, or cue recovery. Rows without checkmarks under OPD denote the vanilla OPD setting. Bold and underlined numbers indicate the best and second-best results within each student-setting group, respectively.

We provide additional ablation results under a cross-benchmark evaluation protocol. All models are trained on Geometry3K and evaluated on geometry-related subsets of MathVista and DynaMath. This setting is more challenging than the in-domain Geometry3K evaluation in Sec. 4.2 because the target benchmarks differ in problem style, annotation format, and visual distribution. As shown in Tab. 5, two findings stand out:

(1) Visual-cue supervision consistently improves transfer performance over the answer-based baseline. For example, in the 2B OPSD setting, replacing answer with visual cues improves MathVista geometry reasoning from 67.36 to 73.22, MathVista geometry problem solving from 68.75 to 75.48, and DynaMath plane geometry from 51.82 to 54.55. Similar gains are also observed under cross-scale OPD and with the 8B student. These results indicate that the benefit of visual cues is not limited to the Geometry3K test set, but also transfers to related geometry benchmarks.   
(2) Cue recovery shows greater sensitivity under cross-benchmark evaluation. Although cue recovery brings additional gains in several cases, such as 2B OPSD on the two MathVista subsets and 8B OPSD/OPD on DynaMath plane geometry, it does not uniformly outperform visual-cue supervision alone. One possible reason is that the recovery module introduces more extra capacity for larger students: it adds 536.9M parameters to the 8B student, accounting for 5.77% of the model, compared with 100.7M parameters and 4.52% for the 2B student. With the same amount of training data, this larger module may overfit source-domain visual-cue patterns. When target benchmarks differ in visual style, reasoning format, or cue granularity, the recovered cues can become less reliable, explaining the mixed gains of cue recovery across target subsets.

# B.2. Scaling Behavior across Teacher-Student Configurations

Fig. 5 examines how ViCuR behaves as both student capacity and teacher strength vary. ViCuR improves all four configurations, with the largest gain (+6.88) for the 8B student distilled from a 32B teacher. For the 2B student, the stronger 8B teacher does not outperform self-distillation (+3.35 vs. +4.80), indicating that once the student becomes the bottleneck, additional teacher strength is harder to exploit. Overall, these results suggest that ViCuR scales most favorably when both the teacher and the student are sufficiently capable.

![](images/f83be7b147120e90431e7f31fe89240bf14c1fcb17cf8357dd7be64ebb5d2db6.jpg)

<details>
<summary>bar</summary>

| Model | Baseline (%) | ViCuR (%) | Self-distillation (%) | Cross-scale distillation (%) |
| :--- | :--- | :--- | :--- | :--- |
| 2B-2B | 34.14 | 38.94 | +4.80 | |
| 8B-8B | 52.40 | 57.05 | +4.65 | |
| 2B-8B | 36.06 | 39.41 | +3.35 | |
| 8B-32B | 52.53 | 59.41 | +6.88 | |
</details>

Figure 5: ViCuR gains across teacher–student configurations on Geometry3K. Each x-axis label denotes a student–teacher pair. Baselines are answer-based OPSD and vanilla OPD, respectively.

# B.3. Token-level Log-probability Analysis

Full-rollout log-probability analysis. Fig. 6 tracks the student’s accuracy on the train and test splits of Geometry3K and the teacher–student token-level log-probability gap on the full rollout trajectory during training under different teacher–student configurations. In the 2B-student/2B-teacher setting, the answerbased baseline obtains a smaller gap but much lower accuracy, suggesting that easier token-level imitation does not necessarily lead to better multimodal reasoning. By contrast, visual cues and ViCuR sometimes maintain a larger gap while achieving higher accuracy, implying that the teacher provides more challenging but more useful supervision grounded in visual evidence. These results support our hypothesis that, for multimodal reasoning, helping the student learn question-relevant visual cues is more important than simply imitating answer-conditioned trajectories.

Segment-level log-probability analysis. We further divide each student rollout into four equal segments and analyze the teacher–student token-level log-probability gap in each segment (Fig. 7). Compared with the baseline, ViCuR often obtains a comparable or smaller gap, especially in the stronger-student settings, suggesting that visual-cue-based supervision does not weaken token-level alignment. The segment-level results further show that the benefit of ViCuR is not confined to the final answer tokens. The gap of ViCuR is consistently smaller than that of the baseline across the middle two segments of the rollout, where the model typically performs visual reasoning and answer derivation. Although ViCuR can exhibit a larger gap in the early segment, this is consistent with the additional difficulty of grounding the reasoning process in question-relevant visual evidence rather than directly imitating answer-conditioned supervision. Overall, these results indicate that visual cues provide effective supervision throughout the reasoning trajectory, helping the student learn to ground its reasoning in question-relevant visual evidence rather than fitting answer-aware patterns.

# B.4. Computational Overhead Analysis

The cue recovery module adds 100.7M parameters for the 2B model (4.52%) and 536.9M for the 8B model (5.77%). It operates only during prefill and introduces no autoregressive decoding overhead. We quantify the wall-clock cost below.

Training time. Tab. 6 reports the total training time on the Vision R1 dataset (5 epochs, 275 steps; single node, 8×H200). ViCuR introduces only modest overhead over the corresponding baseline at both scales, confirming that the recovery module does not substantially increase training cost.

Inference time. Tab. 7 compares inference latency with and without the cue recovery module on a single H200 GPU (bfloat16, flash attention 2, max generation 2,048 tokens). We test two input conditions representing

![](images/aff84506f24f209fd2a61e585c22a2e740ed9c680643dd7066101acb40bfba9f.jpg)

Figure 6: Token-level analysis on Geometry3K during training. We report the student’s train and test accuracy and the teacher–student token-level log-probability gap under different teacher–student configurations. For OPSD, we use the answer-based method as the baseline, and for OPD, we adopt vanilla OPD without any additional input information as the baseline.   
![](images/4359c08aa2bfba1b6d22fec3fec49a74c75e16691d916efb512fc0a90848fceb.jpg)  
Figure 7: Segment-level token-level analysis on Geometry3K. We compare ViCuR with the answer-based baseline in terms of validation accuracy, full-sequence teacher–student log-probability gap, and segment-wise gaps over four equal rollout segments. For both OPSD and OPD, we use the answer-based method as the baseline.

<table><tr><td>Method</td><td>2B</td><td>8B</td></tr><tr><td>OPSD</td><td>6:37:03</td><td>9:13:53</td></tr><tr><td>OPSD + ViCuR</td><td>6:47:52</td><td>8:18:38</td></tr><tr><td>OPD</td><td>6:53:41</td><td>9:07:18</td></tr><tr><td>OPD + ViCuR</td><td>6:57:21</td><td>8:43:16</td></tr></table>

Table 6: Total wall-clock training time (hh:mm:ss) on Vision R1 (5 epochs, 275 steps; 8×H200).

typical low- and high-resolution multimodal inputs: a small-image input (88 tokens total, 64 visual tokens) and a large-image input (1,821 tokens total, 1,782 visual tokens). All measurements are averaged over 10 runs after 3 warmup iterations.

<table><tr><td>Input</td><td>Model</td><td>Input len</td><td>Vis. tokens</td><td>Prefill (ms)</td><td>Decoding (ms/tok)</td><td>Avg output len</td><td>Total gen (ms)</td></tr><tr><td rowspan="4">Small image</td><td>2B baseline</td><td>88</td><td>64</td><td>30.64</td><td>18.31</td><td>633</td><td>11,625</td></tr><tr><td>2B + ViCuR</td><td>88</td><td>64</td><td>38.14 (+24.5%)</td><td>16.78</td><td>207</td><td>3,506</td></tr><tr><td>8B baseline</td><td>88</td><td>64</td><td>36.92</td><td>24.14</td><td>165</td><td>4,019</td></tr><tr><td>8B + ViCuR</td><td>88</td><td>64</td><td>44.06 (+19.3%)</td><td>21.47</td><td>211</td><td>4,583</td></tr><tr><td rowspan="4">Large image</td><td>2B baseline</td><td>1,821</td><td>1,782</td><td>68.79</td><td>18.18</td><td>369</td><td>6,778</td></tr><tr><td>2B + ViCuR</td><td>1,821</td><td>1,782</td><td>73.60 (+7.0%)</td><td>16.68</td><td>428</td><td>7,213</td></tr><tr><td>8B baseline</td><td>1,821</td><td>1,782</td><td>129.20</td><td>23.36</td><td>255</td><td>6,078</td></tr><tr><td>8B + ViCuR</td><td>1,821</td><td>1,782</td><td>134.71 (+4.3%)</td><td>21.55</td><td>446</td><td>9,745</td></tr></table>

Table 7: Inference latency on a single H200 GPU. We report prefill time, per-token decoding time, average output length, and total generation time under small-image (88 tokens, 64 visual) and large-image (1,821 tokens, 1,782 visual) inputs. Total generation time varies primarily due to different output lengths across runs; per-token decoding time is the comparable metric.

Two observations support the computational efficiency of ViCuR. First, per-token decoding time is unaffected by the recovery module, as it is inactive during autoregressive generation. Across all configurations, the per-token decoding time remains comparable between baseline and ViCuR (differences are within run-to-run variance due to different generated content lengths). Since decoding dominates total inference time—for example, in the 2B large-image case, decoding accounts for >98% of total generation time—the recovery module has negligible impact on end-to-end inference cost. Second, the prefill overhead introduced by the recovery module scales sub-linearly with visual token count. On the small-image input (64 visual tokens), the relative overhead is ∼20%; on the large-image input (1,782 visual tokens), it drops to 4–7%. This is because the cross-attention computation in the recovery module uses a single query (the sink token) against visual keys/values, yielding ??(??) cost where ?? is the number of visual tokens—a negligible addition compared to the ??(??2) self-attention over the full sequence during prefill. In the high-resolution settings common in multimodal reasoning, this overhead becomes increasingly marginal.

# C. Qualitative Comparison of Teacher-Side Hints

Fig. 8 compares two forms of teacher-side privileged hints given the same student-facing query: one is the reasoning hint that directly exposes the key theorem or solving idea, and the other is the visual cue hint that describes only the question-relevant visual structure grounded in the image.

# D. Hint Leakage Statistics in Student Rollouts

We report the complete hint-leakage statistics in Tab. 8. The statistics are computed from student rollout outputs only: we scan the output field, match the complete word “Hint” case-insensitively, and aggregate both the number of samples containing at least one match and the total number of matches. Each epoch contains 55 rollout steps. In OPSD, the baseline teacher is conditioned on answer-side privileged hints, whereas ViCuR replaces these with visual cues. In OPD, the baseline does not use privileged hints, while OPD + ViCuR adds teacher-side visual cues. Thus, student-side occurrences of “Hint” should be interpreted as a lexical proxy for explicit reproduction of privileged prompt formats, with the OPD baseline serving as a no-privilege reference.

# Case 1: Intersecting chords

![](images/5194d7d67c0eecd787376d2f9fe0181fcb21b7c100efa3e4d34fbd47fea4be33.jpg)  
Question: Find ??. Answer: 3

# Reasoning hint

[Hint]: The problem is solved by applying the Intersecting Chords Theorem, which equates the product of the segment lengths of one chord (?? · 8) to the product of the segment lengths of the other chord (4 · 6).

# Visual cue hint (ours)

[Hint]: The diagram shows a circle with two intersecting chords. One chord is divided into segments labeled ?? and 8, while the other chord is divided into segments labeled 4 and 6. The intersection lies inside the circle, forming four chord segments with the given lengths.

# Case 2: Trigonometry in isosceles triangles

![](images/f7fd3b53c9947106c2eb7f9141b18c40f9cd001cba851a765f1ec5541ea93be8.jpg)  
Question: Find ??. Round to the nearest tenth.   
Answer: 44.0

# Reasoning hint

[Hint]: The problem involves an isosceles triangle where the altitude creates a right triangle with a known hypotenuse (due to congruent sides) and a known base angle, allowing the use of the sine ratio (︁sin = opphyp) hyp to solve for the height ??.

# Visual cue hint (ours)

[Hint]: The diagram shows an isosceles triangle with congruent legs marked by tick marks, each labeled as 46. A vertical altitude is drawn from the apex to the base, forming a right angle at the base and splitting the triangle into two right triangles. On the left right triangle, the hypotenuse is 46, one angle is labeled 73∘, and the altitude is labeled ??, representing the side opposite the 73∘ angle.

Figure 8: Qualitative comparison of teacher-side privileged hints under the same student-facing query. Reasoning hints directly expose the key theorem or solving idea, whereas the visual cue hints used in our framework describe only the question-relevant visual structure grounded in the image.

The OPSD baselines show a sharp increase in explicit hint reproduction, especially in later epochs, indicating that answer-side privileged hints can be copied into student rollouts. ViCuR substantially reduces this effect under OPSD at both student scales. The OPD baseline remains low, as expected, because it does not include privileged hints. More importantly, OPD + ViCuR also stays at a low absolute level despite using teacher-side visual cues; even when its lexical count is slightly higher than OPD, the magnitude is far below the leakage observed in answer-based OPSD. This suggests that visual-cue privilege does not induce the same large-scale explicit hint reproduction as answer-based self-distillation.

# E. Qualitative Case Studies and Attention Visualization

We provide qualitative case studies and SinkTrack cross-attention visualizations using the checkpoints from the main experiments. This appendix complements the main results by examining the central mechanism underlying by ViCuR: replacing answer-side privilege with visually grounded cues is useful only if the student can internally recover cue-level evidence from the image at inference time. We therefore ask whether ViCuR reduces the concrete grounding errors induced by answer-based OPSD, and whether the sink-based recovery module aggregates visual tokens consistent with the teacher-side cue evidence. We use two representative examples—one from Geometry3K and one from MathVista—because they stress different forms of recoverable visual evidence: local relation binding in a geometry diagram and query-conditioned evidence selection in a chart.

<table><tr><td rowspan="2">Student &amp; Method</td><td colspan="5">Epoch</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td colspan="6">2B Student</td></tr><tr><td>OPSD</td><td>107/120</td><td>747/4,768</td><td>3,021/31,143</td><td>3,845/37,951</td><td>4,054/38,151</td></tr><tr><td>OPSD + ViCuR</td><td>88/101</td><td>243/331</td><td>1,198/8,164</td><td>2,758/20,533</td><td>3,264/25,861</td></tr><tr><td>OPD (8B teacher)</td><td>61/74</td><td>56/63</td><td>60/65</td><td>56/63</td><td>39/46</td></tr><tr><td>OPD + ViCuR (8B teacher)</td><td>56/60</td><td>81/85</td><td>123/140</td><td>172/192</td><td>247/316</td></tr><tr><td colspan="6">8B Student</td></tr><tr><td>OPSD</td><td>38/42</td><td>183/610</td><td>2,447/29,576</td><td>3,628/45,629</td><td>3,743/44,453</td></tr><tr><td>OPSD + ViCuR</td><td>28/34</td><td>39/40</td><td>103/134</td><td>1,125/10,462</td><td>2,424/17,976</td></tr><tr><td>OPD (32B teacher)</td><td>26/29</td><td>22/22</td><td>14/15</td><td>16/16</td><td>16/16</td></tr><tr><td>OPD + ViCuR (32B teacher)</td><td>25/27</td><td>40/45</td><td>38/38</td><td>32/32</td><td>44/47</td></tr></table>

Table 8: Complete hint-leakage statistics in student rollout outputs. Each cell reports samples / total occurrences. All configurations are trained with Steps = 55 per epoch.

![](images/79e2e5ae5a6286c1c1713901282826000bc379d0b56c0947071b54185c4bf3b0.jpg)

<details>
<summary>line</summary>

| Epoch | Series 1 | Series 2 | Series 3 | Series 4 |
|-------|----------|----------|----------|----------|
| 1     | 60       | 30       | 60       | 30       |
| 2     | 80       | 40       | 55       | 25       |
| 3     | 120      | 40       | 60       | 20       |
| 4     | 170      | 35       | 55       | 20       |
| 5     | 250      | 45       | 40       | 20       |
</details>

![](images/988f88a173edd1174b5a37ba5481d4f69d9bd4a6977783f4c2ec5351bd714822.jpg)

<details>
<summary>line</summary>

| Epoch | OPD 2B (8B teacher) | OPD+ViCuR 2B (8B teacher) | OPD 8B (32B teacher) | OPD+ViCuR 8B (32B teacher) |
|-------|---------------------|---------------------------|----------------------|----------------------------|
| 1     | 70                  | 60                        | 20                   | 30                         |
| 2     | 60                  | 90                        | 15                   | 45                         |
| 3     | 65                  | 140                       | 10                   | 40                         |
| 4     | 65                  | 190                       | 10                   | 35                         |
| 5     | 50                  | 320                       | 10                   | 45                         |
</details>

Figure 9: Hint leakage in student rollout outputs under OPD. The OPD baseline does not use teacher-side privileged hints, while OPD + ViCuR introduces teacher-side visual cues. The counts remain low in both settings, showing that visual-cue privilege does not induce the large-scale explicit hint reproduction observed in answer-based OPSD.

# E.1. Case 1: Geometry3K — Intersecting Chords Theorem

Quantitative stability. We evaluate each model on this Geometry3K example across 10 repeated runs with identical input. The base Qwen3-VL-2B-Instruct model produces the correct answer in only 5/10 runs, and the answer-based OPSD baseline improves to 7/10, while OPSD + ViCuR answers correctly in all 10/10 runs. This example isolates the intended role of visual cue recovery because the error is not about theorem recall: all models invoke the intersecting chords theorem. The decisive variable is whether the student recovers the visual cue that the diagram contains two intersecting chords, with ?? paired with 8 on one chord and 4 paired with 6 on the other. The incorrect runs instead bind 4 and ?? as if they were on the same chord, yielding $4 \times x = 6 \times 8$ and $x = 1 2$ . ViCuR recovers the cue-level relation needed for the theorem, giving $4 \times 6 = x \times 8$ and $x = 3$ Thus, the improvement is better interpreted as recovery of inference-accessible visual structure rather than

![](images/03d8dabd2e7106e7ba3bfcfbe92286a7631f3366c7fec28d7e92178fcba839ab.jpg)

<details>
<summary>pie</summary>

| Category | Value |
|---|---|
| x | 6 |
| 4 | 4 |
| 8 | 8 |
</details>

Q: Find ??.   
A: 3   
(a) Case 1: Geometry3K

![](images/64eb0a3cea57516f25fcc8aaf321e4a931b7bb0cb622321c3da0d20188ab5dc8.jpg)  
Figure 1: The key findings of scaling relationship on learning math reasoning ability with LLMs.   
Q: Which model has the highest Acc score when Pretrain Loss = 1.80?   
A: RFT U13B   
(b) Case 2: MathVista

Figure 10: Overview of the two cases used in the qualitative study.

exposure to an answer or a reasoning shortcut.

Model reasoning outputs. Below we compare the reasoning traces of the three models. The base model and OPSD baseline produce plausible-looking derivations, but the derivations are anchored to the wrong recovered cue: they place 4 and ?? on the same chord. ViCuR changes the critical grounding step before any algebra is performed. Its trace matches the teacher-side visual cue format used in our framework—a description of visible segment relations—rather than a direct answer-bearing rationale, and the subsequent computation follows the correct diagram structure.

# Base Qwen3-VL-2B-Instruct (incorrect, ?? = 12)

Based on the provided image, we can solve for ?? using the \*\*Intersecting Chords Theorem\*\*.

This theorem states that when two chords intersect inside a circle, the product of the lengths of the segments of one chord is equal to the product of the lengths of the segments of the other chord.

In the diagram, the two chords intersect at a point inside the circle. The segments are: - One chord is divided into segments of lengths 4 and ??. - The other chord is divided into segments of lengths 6 and 8.

The theorem gives us the equation:

$$
4 \times x = 6 \times 8
$$

Now, we solve for ??:

$$
4 x = 4 8
$$

$$
x = \frac {4 8}{4}
$$

$$
x = 1 2
$$

Therefore, the value of ?? is 12.

# OPSD baseline (incorrect, ?? = 12)

To find the value of ??, we can use \*\*the Intersecting Chords Theorem\*\*. This theorem states that if two chords intersect inside a circle, the products of the lengths of the segments of each chord are equal.

In the given diagram, the two chords intersect at a point inside the circle. The chords are: - One chord is divided into segments of lengths 4 and ??. - The other chord is divided into segments of lengths 6 and 8.

According to the theorem:

$$
(4) (x) = (6) (8)
$$

Now, let’s solve for ??:

$$
4 x = 4 8
$$

$$
x = \frac {4 8}{4}
$$

$$
x = 1 2
$$

So, the value of ?? is 12.

# OPSD + ViCuR (ours) (correct, ?? = 3)

To find the value of ??, we can use the \*\*Intersecting Chords Theorem\*\*. This theorem states that if two chords intersect inside a circle, the products of the lengths of the segments of one chord are equal to the products of the lengths of the segments of the other chord.

In this case, the two chords intersect at a point inside the circle. The chords are: - One chord is divided into segments of length 4 and

6. - The other chord is divided into segments of length ?? and 8.

According to the Intersecting Chords Theorem:

$$
(4) (6) = (x) (8)
$$

Now, solve for ??:

$$
2 4 = 8 x
$$

$$
x = \frac {2 4}{8} = 3
$$

So, the value of ?? is 3 .

# E.2. Case 2: MathVista — Generalization Beyond Geometry

Qualitative observations. This example from MathVista examines whether cue recovery extends beyond structured geometry diagrams to chart-reading tasks. As shown in Fig. 10, the question asks which model achieves the highest accuracy at a given pretrain loss value. The recoverable evidence is no longer a small set of local geometric labels; it is a query-conditioned evidence set consisting of the ??-axis slice at Pretrain Loss = 1.80, the identities of multiple chart series, and their relative vertical positions. The base model and OPSD baseline both answer LLaMA-2-70B, suggesting that answer-based supervision does not prevent the student from relying on an incomplete or misread subset of curves. In contrast, ViCuR treats the problem as a visual comparison over the relevant slice and identifies RFT U13B as the highest. This pattern is consistent with the quantitative improvements on MathVista (e.g., +2.2 for OPSD + ViCuR at 2B scale, Tab. 1), while illustrating that the recovered cue evidence need not be geometry-specific.

Model reasoning outputs. Below we compare the full reasoning traces. The base model and OPSD baseline list confident numerical estimates, but their comparison omits or mispositions the series that should dominate at the queried loss value. ViCuR’s trace is more faithful to the visual cue that a teacher would provide for this task: locate the specified ??-coordinate, compare the series at that slice, and select the highest curve. The case therefore probes whether ViCuR recovers a task-conditioned evidence set, not merely whether it memorizes a chart answer.

# Base Qwen3-VL-2B-Instruct (incorrect: LLaMA-2-70B)

Based on the provided chart, we can determine which model has the highest accuracy (Acc) at a Pretrain Loss of 1.80 by examining the data points on the graph.

1. \*\*Locate the Pretrain Loss:\*\* Find the value ‘1.80‘ on the horizontal axis (x-axis), labeled "Pretrain Loss". 2. \*\*Identify the corresponding Acc scores:\*\* Look at the vertical axis (y-axis), labeled "Acc", to see the accuracy for each model at this point. 3. \*\*Compare the values:\*\* The chart shows the following accuracy values at ‘Pretrain Loss = 1.80‘: - \*\*LLaMA-7B:\*\* Accuracy is approximately 37.5 - \*\*LLaMA-13B:\*\* Accuracy is approximately 42.5 - \*\*LLaMA-2-7B:\*\* Accuracy is approximately 45.5 - \*\*LLaMA-2-13B:\*\* Accuracy is approximately 47.5 - \*\*LLaMA-2-70B:\*\* Accuracy is approximately 61.5 - \*\*RFT k=100:\*\* Accuracy is approximately 40.5 - \*\*RFT k=25:\*\* Accuracy is approximately 41.5 - \*\*RFT k=6:\*\* Accuracy is approximately 42.5 - \*\*RFT U13B:\*\* Accuracy is approximately 48.5

The highest accuracy score among these models is \*\*61.5\*\*, which corresponds to the \*\*LLaMA-2-70B\*\* model.

Therefore, the model with the highest Acc score when Pretrain Loss is equal to 1.80 is \*\*LLaMA-2-70B\*\*.

# OPSD baseline (incorrect: LLaMA-2-70B)

Based on the provided chart, we can determine which model has the highest Accuracy (Acc) score at a Pretrain Loss of 1.80 by examining the data points on the graph.

1. \*\*Locate the Pretrain Loss:\*\* Find the value 1.80 on the horizontal axis (Pretrain Loss). 2. \*\*Identify the corresponding Accuracy:\*\* Look at the vertical positions of the data points on the chart for each model at this specific Pretrain Loss value. 3. \*\*Compare the scores:\*\* The chart shows the following accuracy values at Pretrain Loss = 1.80: - \*\*LLaMA-7B:\*\* The data point is at approximately 39.0 Acc. - \*\*LLaMA-13B:\*\* The data point is at approximately 43.0 Acc. - \*\*LLaMA-2-7B:\*\* The data point is at approximately 45.0 Acc. - \*\*LLaMA-2-13B:\*\* The data point is at approximately 47.0 Acc. - \*\*LLaMA-2-65B:\*\* The data point is at approximately 50.0 Acc. - \*\*LLaMA-2-70B:\*\* The data point is at approximately 54.0 Acc.

The model with the highest accuracy at a Pretrain Loss of 1.80 is \*\*LLaMA-2-70B\*\*.

# OPSD + ViCuR (ours) (correct: RFT U13B)

Based on the provided chart, we can determine which model has the highest accuracy (Acc) at a pretrain loss of 1.80 by following these steps:

1. \*\*Locate the Pretrain Loss value:\*\* Find the value ‘1.80‘ on the horizontal axis (x-axis), labeled "Pretrain Loss". 2. \*\*Identify the corresponding accuracy:\*\* Look at the vertical position (y-axis, labeled "Acc") for each model’s line at this specific point. 3. \*\*Compare the values:\*\* The y-axis represents accuracy, with higher values indicating better performance.

Let’s examine the data points for the models at ‘Pretrain Loss = 1.80‘: - \*\*ICL (black line):\*\* The data point is at approximately ‘40.0‘ Acc. - \*\*SFT (blue line):\*\* The data point is at approximately ‘35.0‘ Acc. - \*\*SFT 1/2 (dotted blue line):\*\* The data point is at approximately ‘37.0‘ Acc. - \*\*SFT 1/4 (dotted light blue line):\*\* The data point is at approximately ‘25.0‘ Acc. - \*\*SFT 1/8 (dotted light purple line):\*\* The data point is at approximately ‘18.0‘ Acc. - \*\*RFT k=100 (dark green line):\*\* The data point is at approximately ‘42.0‘ Acc. - \*\*RFT k=25 (dotted light green line):\*\* The data point is at approximately ‘40.0‘ Acc. - \*\*RFT k=6 (dotted light grey line):\*\* The data point is at approximately ‘40.0‘ Acc. - \*\*RFT U13B (teal line):\*\* The data point is at approximately ‘50.0‘ Acc.

The highest accuracy value among all models at ‘Pretrain Loss = 1.80‘ is ‘50.0‘, which corresponds to the \*\*RFT U13B\*\* model.

![](images/2302a2a006ac3e2ea5fd2a5d6c882a1599f89efea7d604ba25773c7cc618b26a.jpg)

![](images/d72a97ba704fad4ab86f2880c29cbbf5db20ebd6f6154c0fa6005aafce2354cf.jpg)

<details>
<summary>text_image</summary>

SinkTrack cross-attention (Layer 0, mean over heads)
</details>

(d)

![](images/8849e1d23ea99ce871f829e83e66b1c849588056a4c148e816e1466aab55164f.jpg)  
(e)

SinkTrack cross-attention over visual tokens (mean over heads)   
![](images/7fef8fc1cf36992ac519def9f6d855dc1e50722eb07108ef27482aae0459db4f.jpg)

![](images/de980f893f29855d7b62b876f30e861b12b5650a9ad852aa8c08f077d12df7e3.jpg)

![](images/9da997d3dc7ea0a4921a3474e0273f779253df6175d060f77f5ce1f1e942477d.jpg)

![](images/d88d9ef1f3f62b0b23c58343c2616c34e1d919a4064d8f873ccf93ae918137f6.jpg)

![](images/c1c26c314ea1f6cbd4cdfcf8b347183065365b8f4f066abfe85a75f4b949f2e3.jpg)

![](images/8c073b9356c4583f3b2f96a85b99f59916d7a3d0bb039f679836d07f579f42cd.jpg)  
(f)   
Figure 11: SinkTrack cross-attention visualizations for both qualitative cases. (a)–(c) Geometry3K (Case 1): first SinkTrack layer, mean over all 6 layers, and full layer-by-layer progression. (d)–(f) MathVista (Case 2): same three views. Best viewed in color.

SinkTrack attention visualization. Fig. 11 presents the SinkTrack cross-attention heatmaps for both cases in a unified view. These heatmaps visualize the dedicated cross-attention from the sink token to visual tokens, i.e., the pathway through which the recovery module forms the internal representation ??ˆ. They therefore test a more specific question than ordinary saliency visualization: whether the module aggregates visual evidence compatible with the teacher-side cues, despite receiving no cue text or cue-generation loss. Panels (a)–(c) correspond to the Geometry3K example (Case 1): (a) the first SinkTrack layer, (b) the mean attention over all 6 SinkTrack layers, and (c) the full layer-by-layer progression. Panels (d)–(f) show the same three views for the MathVista example (Case 2).

For Case 1, the visual cue to be recovered is the chord-pairing structure: the labels ?? and 8 belong to one chord, while 4 and 6 belong to the other. In both the first-layer map and the layer-averaged map, high-attention regions overlap with the numerical labels, the intersection area, and the chord segments adjacent to these labels. The layer-by-layer views in panel (c) further show repeated coverage of the relevant label and chord regions across SinkTrack layers, rather than a single-layer spike. This is consistent with the behavioral trace: ViCuR does not need a teacher hint that states the theorem or the answer, but it must recover the visible segment relation that makes the theorem usable.

For Case 2, the attention pattern is naturally less localized. Chart reading requires a query-conditioned aggregation over several visual elements: the ??-axis value, the legends or series identities, and the vertical positions of multiple curves. Accordingly, panels (d)–(f) show distributed, structured responses across the visual-token sequence rather than a compact hotspot over a single region. This should not be over-interpreted as proof that the model isolated the exact RFT U13B curve at every layer. A more defensible interpretation is that SinkTrack preserves multi-region chart context needed for the final comparison, which is precisely the kind of internal cue evidence that ViCuR is designed to recover.

Summary. Across both cases, the main advantage of ViCuR is not a change in the final reasoning template, but a more reliable conversion of inference-accessible visual cue supervision into student-side internal evidence recovery. In the geometry example, this recovery appears as correct binding between labels and chord segments; in the chart example, it appears as preserving enough multi-region context to compare series at the queried loss value. The SinkTrack visualizations do not constitute causal proof, but they provide mechanism-consistent evidence that the dedicated sink cross-attention aggregates the kind of visual information described by teacherside cues. These qualitative observations complement the theory and main experiments by showing how ViCuR can reduce answer-privilege-induced grounding failures in concrete model outputs.

# F. Additional Theoretical Details

# F.1. Derivation and Analysis of Proposition 1 (Recoverable Privilege)

Proposition 1 (Recoverable Privilege). Under the deterministic abstraction $S = f ( X )$ , visually grounded cues do not introduce the privilege-induced conditional information gap associated with answer-dependent variables. The resulting teacher supervision is conditioned only on information determined by the inference-time input ??.

In this appendix, we provide additional details on the theoretical motivation in Section 3.1. The goal is not to claim that the student can perfectly reconstruct visual cue text, but to clarify why image-grounded cues differ from answer-dependent privileges: under an ideal deterministic abstraction, the cue does not introduce information beyond the inference-time input ??.

Setup. Let $X = ( I , Q )$ denote the multimodal input available at inference time, and let ?? denote the teacherside visual cue text. We consider a decoding step ?? with prefix $Y _ { < t }$ . The cue-conditioned teacher distribution is

$$
p _ {T} (Y _ {t} \mid X, S, Y _ {<   t}), \tag {9}
$$

and the student distribution is

$$
p _ {\theta} (Y _ {t} \mid X, Y _ {<   t}). \tag {10}
$$

For comparison, we define the teacher’s marginal distribution over ?? as

$$
p _ {T} (Y _ {t} \mid X, Y _ {<   t}) = \mathbb {E} _ {S \sim p (S | X)} \left[ p _ {T} (Y _ {t} \mid X, S, Y _ {<   t}) \right]. \tag {11}
$$

We analyze the expected token-level KL objective

$$
\mathcal {L} _ {\mathrm{KL}} (\theta) = \mathbb {E} _ {X, S, Y _ {<   t}} \left[ D _ {\mathrm{KL}} \left(p _ {T} (\cdot \mid X, S, Y _ {<   t}) \| p _ {\theta} (\cdot \mid X, Y _ {<   t})\right) \right]. \tag {12}
$$

Step 1: Expanding the KL Objective. Following the decomposition framework of [36], we expand the KL divergence as

$$
\mathcal {L} _ {\mathrm{KL}} (\theta) = \mathbb {E} _ {X, S, Y _ {<   t}, Y _ {t}} \left[ \log p _ {T} (Y _ {t} \mid X, S, Y _ {<   t}) - \log p _ {\theta} (Y _ {t} \mid X, Y _ {<   t}) \right], \tag {13}
$$

where $Y _ { t }$ is sampled from $p _ { T } ( \cdot \mid X , S , Y _ { < t } )$ . Adding and subtracting log $p _ { T } ( Y _ { t } \mid X , Y _ { < t } )$ , we obtain

$$
\mathcal {L} _ {\mathrm{KL}} (\theta) = \mathbb {E} _ {X, S, Y _ {<   t}, Y _ {t}} \left[ \log \frac {p _ {T} \left(Y _ {t} \mid X , S , Y _ {<   t}\right)}{p _ {T} \left(Y _ {t} \mid X , Y _ {<   t}\right)} \right] \tag {14}
$$

$$
+ \mathbb {E} _ {X, S, Y _ {<   t}, Y _ {t}} \left[ \log \frac {p _ {T} (Y _ {t} \mid X , Y _ {<   t})}{p _ {\theta} (Y _ {t} \mid X , Y _ {<   t})} \right].
$$

Step 2: Information-Theoretic Decomposition. The first term in Eq. (14) is the conditional mutual information between the teacher’s next token and the cue:

$$
I _ {T} (Y _ {t}; S \mid X, Y _ {<   t}), \tag {15}
$$

where the subscript ?? emphasizes that $Y _ { t }$ is drawn from the cue-conditioned teacher distribution. The second term equals

$$
\mathbb {E} _ {X, S, Y _ {<   t}} \left[ \mathbb {E} _ {Y _ {t} \sim p _ {T} (\cdot | X, S, Y _ {<   t})} \left[ \log \frac {p _ {T} (Y _ {t} \mid X , Y _ {<   t})}{p _ {\theta} (Y _ {t} \mid X , Y _ {<   t})} \right] \right]. \tag {16}
$$

Note that in general this is not the same as $D _ { \mathrm { K L } } ( p _ { T } ( \cdot \mid X , Y _ { < t } ) \| p _ { \theta } ( \cdot \mid X , Y _ { < t } ) )$ , because $Y _ { t }$ is sampled from the cue-conditioned distribution $p _ { T } ( \cdot \mid X , S , Y _ { < t } )$ rather than the marginal $p _ { T } ( \cdot \mid X , Y _ { < t } )$ . However, under the deterministic cue abstraction $S = f ( X )$ introduced in Step 3 below, these two distributions coincide, and the second term reduces to the standard KL between the teacher’s marginal and the student:

$$
\mathcal {L} ^ {*} (\theta) := \mathbb {E} _ {X, Y _ {<   t}} \left[ D _ {\mathrm{KL}} \left(p _ {T} (\cdot | X, Y _ {<   t}) \| p _ {\theta} (\cdot | X, Y _ {<   t})\right) \right]. \tag {17}
$$

Under this abstraction, we therefore obtain

$$
\mathcal {L} _ {\mathrm{KL}} (\theta) = \mathcal {L} ^ {*} (\theta) + I _ {T} (Y _ {t}; S \mid X, Y _ {<   t}). \tag {18}
$$

Step 3: Deterministic Cue Abstraction. For the purpose of theoretical analysis, we idealize the cue construction process as a deterministic mapping

$$
S = f (X). \tag {19}
$$

Under this abstraction, $H ( S \mid X ) = 0$ . Since ?? is already determined once ?? is given, conditioning additionally on $Y _ { < t }$ or $Y _ { t }$ cannot introduce uncertainty about ??:

$$
H (S \mid X, Y _ {<   t}) = H (S \mid X, Y _ {<   t}, Y _ {t}) = 0. \tag {20}
$$

Therefore,

$$
I _ {T} (Y _ {t}; S \mid X, Y _ {<   t}) = H (S \mid X, Y _ {<   t}) - H (S \mid X, Y _ {<   t}, Y _ {t}) = 0. \tag {21}
$$

In this idealized case, cue-conditioned supervision does not introduce an additional conditional information gap beyond ??, and Eq. (18) reduces to

$$
\mathcal {L} _ {\mathrm{KL}} (\theta) = \mathcal {L} ^ {*} (\theta). \tag {22}
$$

Discussion: Information Availability vs. Recoverability. The above result should be interpreted carefully. It does not imply that the student directly observes ??, nor that a standard model can perfectly reconstruct the cue text. It only shows that, under the deterministic cue abstraction, the teacher’s additional conditioning variable is determined by the inference-time input. This differs from answer-dependent privileges, where the teacher may rely on variables not recoverable from ??.

Thus, replacing answer-based privilege with visual cues reframes the problem. The remaining challenge is not that the teacher uses information absent at test time, but that the student must learn to extract and represent the relevant visual evidence effectively. This motivates our attention-sink-based recovery module, which provides an internal pathway for aggregating cue-level evidence from visual tokens.

Remark on Stochastic Cue Construction. As discussed in Section 3.1, the deterministic abstraction $S = f ( X )$ is an idealized limit. In practice, cue text may be produced by an LLM-based pipeline with sampling, yielding $H ( S \mid X ) > 0$ . In that case, the conditional information term does not vanish exactly, and the decomposition in Eq. (18) retains a nonzero $I _ { T } ( Y _ { t } ; S \mid X , Y _ { < t } )$ component. However, because visual cues describe evidence visible in ?? rather than answer-dependent information, this residual term is expected to be substantially smaller than for answer-based privileges where $H ( R \mid X )$ can be large. Our method does not require tokenlevel reconstruction of ??; it uses the teacher’s cue-conditioned behavior to shape the student’s internal visual representation, making it less sensitive to superficial linguistic variation in the cue text.

# F.2. Gradient Analysis of Proposition 2 (Task-Driven Implicit Alignment)

Proposition 2 (Task-Driven Implicit Alignment). Under the advantage-weighted on-policy distillation objective, the recovery module receives nonzero gradients to the extent that its internal visual evidence representation affects student predictions on teacher-evaluated tokens. Specifically, when the advantage $A _ { n } > 0$ (teacher assigns higher probability than the student), gradients reinforce the current evidence representation; when $A _ { n } < 0 ,$ , gradients push the representation away from its current state. The distillation objective therefore implicitly encourages the sink-based representation to retain task-relevant visual evidence aligned with the teacher’s visual cues, without requiring explicit cue-text generation or text-level matching.

In this section, we analyze how the on-policy distillation objective provides an implicit task-driven alignment signal for the attention-sink-based recovery module. The analysis uses a simplified unclipped policy-gradient surrogate for clarity. Our implementation follows a PPO-style objective with old log-probabilities, importance ratios, and clipping, but the key gradient pathway into the recovery module is the same.

Setup. Let $X = ( I , Q )$ denote the inference-time input. We denote the internal cue-level evidence representation induced by the recovery module as

$$
\hat {S} = g (V, Q; \theta_ {\text { sink }}), \tag {23}
$$

where ?? represents the layer-wise visual token states and $\theta _ { \mathrm { s i n k } }$ denotes the recovery-module parameters. The student distribution is written as

$$
p _ {\theta} \left(\hat {y} _ {n} \mid X, \hat {S}, \hat {y} _ {<   n}\right), \tag {24}
$$

and the cue-conditioned teacher distribution is

$$
p _ {T} \left(\hat {y} _ {n} \mid X, S, \hat {y} _ {<   n}\right). \tag {25}
$$

For a sampled student trajectory ${ \hat { y } } \sim p _ { \theta } ( \cdot \mid X )$ , we define the teacher-shaped token advantage as

$$
A _ {n} (X, \hat {y}) = \log p _ {T} (\hat {y} _ {n} \mid X, S, \hat {y} _ {<   n}) - \log p _ {\theta} (\hat {y} _ {n} \mid X, \hat {S}, \hat {y} _ {<   n}). \tag {26}
$$

During policy-gradient optimization, this advantage is treated as a stop-gradient quantity.

Step 1: Basic Policy-Gradient Surrogate. The simplified distillation surrogate is

$$
\mathcal {L} _ {\text { distill }} (\theta) = - \mathbb {E} _ {X \sim \mathcal {D}} \mathbb {E} _ {\hat {y} \sim p _ {\theta}} \left[ \frac {1}{| \hat {y} |} \sum_ {n = 1} ^ {| \hat {y} |} \operatorname{sg} [ A _ {n} (X, \hat {y}) ] \log p _ {\theta} (\hat {y} _ {n} \mid X, \hat {S}, \hat {y} _ {<   n}) \right], \tag {27}
$$

where sg[·] denotes stop-gradient. Therefore,

$$
\nabla_ {\theta_ {\mathrm{sink}}} \mathcal {L} _ {\text {distill}} = - \mathbb {E} \left[ \frac {1}{| \hat {y} |} \sum_ {n = 1} ^ {| \hat {y} |} \operatorname{sg} [ A _ {n} (X, \hat {y}) ] \nabla_ {\theta_ {\mathrm{sink}}} \log p _ {\theta} (\hat {y} _ {n} \mid X, \hat {S}, \hat {y} _ {<   n}) \right]. \tag {28}
$$

Step 2: Chain Rule Through the Recovery Module. Since the recovery parameters influence the student distribution through $\hat { S } = g ( V , Q ; \theta _ { \mathrm { s i n k } } )$ , we have

$$
\nabla_ {\theta_ {\text { sink }}} \log p _ {\theta} (\hat {y} _ {n} \mid X, \hat {S}, \hat {y} _ {<   n}) = \nabla_ {\hat {S}} \log p _ {\theta} (\hat {y} _ {n} \mid X, \hat {S}, \hat {y} _ {<   n}) \cdot \nabla_ {\theta_ {\text { sink }}} g (V, Q; \theta_ {\text { sink }}). \tag {29}
$$

Substituting this into Eq. (28) yields

$$
\nabla_ {\theta_ {\text { sink }}} \mathcal {L} _ {\text { distill }} = - \mathbb {E} \left[ \frac {1}{| \hat {y} |} \sum_ {n = 1} ^ {| \hat {y} |} \operatorname{sg} [ A _ {n} (X, \hat {y}) ] \nabla_ {\hat {S}} \log p _ {\theta} (\hat {y} _ {n} \mid X, \hat {S}, \hat {y} _ {<   n}) \cdot \nabla_ {\theta_ {\text { sink }}} g (V, Q; \theta_ {\text { sink }}) \right]. \tag {30}
$$

Interpretation. Equation (30) shows that the gradient flowing into the recovery module is modulated by the teacher-student token-level discrepancy. Specifically, when $A _ { n } > 0$ (the cue-conditioned teacher assigns higher probability than the student to the sampled token), the gradient reinforces the current evidence representation, encouraging the module to retain and strengthen whatever visual evidence contributed to the student’s prediction. When $A _ { n } < 0$ , the gradient pushes the representation away from its current state, discouraging evidence patterns that led to predictions the teacher did not favor.

This provides an implicit task-driven alignment signal. Instead of forcing the student to reconstruct the cue text ?? token by token, the objective shapes ??ˆ only through its effect on downstream predictions. As a result, the recovery module is encouraged to encode visual evidence that matters for matching the teacher’s cueconditioned behavior, while avoiding unnecessary capacity spent on superficial or task-irrelevant linguistic details of the cue text.

Limitations of the analysis. The gradient derivation above uses a simplified unclipped surrogate. In practice, PPO-style clipping limits the magnitude of updates when the importance ratio deviates significantly from 1, which can dampen the alignment signal for tokens with very large advantage magnitudes. Additionally, the analysis treats ??ˆ as a single vector for clarity, whereas in practice the recovery module updates the sink token at multiple layers, and the gradient pathway involves interactions across layers. These simplifications do not affect the qualitative conclusion—that the distillation objective provides a task-driven signal for the recovery module—but the precise dynamics may differ from the idealized single-vector picture.

# F.3. Analytical Counterpart: The Full-Vocabulary KL Objective

Although our implementation optimizes a sampled-token on-policy objective, it is useful to consider the full-vocabulary forward KL objective as an analytical counterpart. This is the same divergence direction $( D _ { \mathrm { K L } } ( p _ { T } \Vert p _ { \theta } ) )$ used in OPSD [41], where the teacher’s token-wise distribution serves as the reference. Analyzing this objective connects our method with standard knowledge distillation and clarifies how cue-conditioned teacher distributions can shape the student’s internal visual evidence representation.

Setup. Let $X = ( I , Q )$ denote the inference-time input and ?? the teacher-side visual cue text. Let

$$
\hat {S} = g (V, Q; \theta_ {\text { sink }}) \tag {31}
$$

be the internal representation induced by the recovery module. We write the student distribution as $p _ { \theta } ( Y _ { t } \mid$ $X , { \hat { S } } , Y _ { < t } )$ , and the cue-conditioned teacher distribution as $p _ { T } ( Y _ { t } \mid X , S , Y _ { < t } )$ .

Consider the full-vocabulary distillation objective

$$
\mathcal {L} _ {\mathrm{KL}} (\theta) = \mathbb {E} _ {X, S, Y _ {<   t}} \left[ D _ {\mathrm{KL}} \left(p _ {T} (\cdot | X, S, Y _ {<   t}) \| p _ {\theta} (\cdot | X, \hat {S}, Y _ {<   t})\right) \right]. \tag {32}
$$

Step 1: Expanding the KL Objective. Expanding over the vocabulary ?? gives

$$
\mathcal {L} _ {\mathrm{KL}} (\theta) = \mathbb {E} _ {X, S, Y _ {<   t}} \left[ \sum_ {v \in \mathcal {V}} p _ {T} (v \mid X, S, Y _ {<   t}) \log \frac {p _ {T} (v \mid X , S , Y _ {<   t})}{p _ {\theta} (v \mid X , \hat {S} , Y _ {<   t})} \right] \tag {33}
$$

$$
= C - \mathbb {E} _ {X, S, Y _ {<   t}} \left[ \sum_ {v \in \mathcal {V}} p _ {T} (v \mid X, S, Y _ {<   t}) \log p _ {\theta} (v \mid X, \hat {S}, Y _ {<   t}) \right].
$$

where $C = \mathbb { E } _ { X , S , Y _ { < t } } [ \sum _ { v } p _ { T } ( v \mid X , S , Y _ { < t } ) \log p _ { T } ( v \mid X , S , Y _ { < t } ) ]$ denotes the teacher’s conditional entropy and is constant with respect to the student parameters ?? (assuming $Y _ { < t }$ is held fixed for the purpose of this per-step analysis).

Step 2: Gradient Flow into the Internal Representation. Differentiating with respect to $\hat { S }$ gives

$$
\nabla_ {\hat {S}} \mathcal {L} _ {\mathrm{KL}} = - \mathbb {E} _ {X, S, Y _ {<   t}} \left[ \sum_ {v \in \mathcal {V}} p _ {T} (v \mid X, S, Y _ {<   t}) \nabla_ {\hat {S}} \log p _ {\theta} (v \mid X, \hat {S}, Y _ {<   t}) \right]. \tag {34}
$$

Step 3: Chain Rule to Recovery Parameters. Applying the chain rule through $\hat { S } = g ( V , Q ; \theta _ { \mathrm { s i n k } } )$ , we obtain

$$
\nabla_ {\theta_ {\text { sink}}} \mathcal {L} _ {\mathrm{KL}} = - \mathbb {E} _ {X, S, Y _ {<   t}} \left[ \sum_ {v \in \mathcal {V}} p _ {T} (v \mid X, S, Y _ {<   t}) \nabla_ {\hat {S}} \log p _ {\theta} (v \mid X, \hat {S}, Y _ {<   t}) \cdot \nabla_ {\theta_ {\text { sink }}} g (V, Q; \theta_ {\text { sink }}) \right]. \tag {35}
$$

Interpretation. Equation (35) shows that the recovery module is updated according to a teacher-weighted combination of student score gradients. Tokens that the cue-conditioned teacher considers more likely contribute more strongly to the gradient. Therefore, the teacher distribution acts as a behavioral guide for shaping the internal representation ??ˆ. This provides an implicit alignment signal: the student is not asked to reproduce the cue text itself, but to adjust its internal visual evidence representation so that its predictions better match the cue-conditioned teacher behavior.