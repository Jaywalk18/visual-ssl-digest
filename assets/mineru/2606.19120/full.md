# Seeing Before Reasoning: Decoupling Perception and Reasoning for Shortcut-Resilient Multimodal On-Policy Self-Distillation

Sihan Wang1,2 Xiyao Liu1 Lianqing Liu1 Zhi Han1∗

1State Key Laboratory of Robotics and Intelligent Systems, Shenyang Institute of Automation, Chinese Academy of Sciences 2University of Chinese Academy of Sciences {wangsihan, liuxiyao, lqliu, hanzhi}@sia.cn

https://oedosoldier.github.io/ViGOS/

## Abstract

On-policy self-distillation (OPSD) trains a model on its own rollouts and uses a frozen copy to provide dense token-level targets conditioned on a reference target. This works well for LLM reasoning, but a direct extension to multimodal large language models (MLLMs) can create a shortcut: the privileged target may guide tokens mainly based on the text reference target rather than the image. We propose ViGOS, a visually grounded OPSD framework for MLLM post-training. The student first writes a visual description and then reasons toward the final answer. For valid rollouts, an image-only perception teacher supervises the description, while a privileged reasoning teacher supervises the reasoning and final answer on the same student prefix. A reference teacher is used only for invalid rollouts to recover the output format. Across general vision-language, expert reasoning, visual math, spatial grounding, and visual-language-prior benchmarks, ViGOS keeps the main benefits of OPSD and improves image-grounded behavior in shortcut-prone settings.

## 1 Introduction

On-policy distillation (OPD) has recently emerged as an effective method for transferring knowledge from large language models (LLMs) to smaller models [1, 6, 13, 22]. Standard knowledge distillation trains the student on fixed teacher-generated or reference trajectories. This can create a train-test gap because the student is not trained on the states it will later generate independently [1, 8]. OPD reduces this gap by sampling trajectories from the student policy and using a teacher policy to provide dense token-level supervision on these trajectories [1, 13]. On-policy self-distillation (OPSD) follows the same idea without a stronger external teacher: it uses a frozen copy of the student, conditioned on a reference answer or solution, and supervises the trainable student on its own rollouts [30]. This preserves the on-policy training signal and provides denser feedback than RLVR methods, which mainly rely on sparse outcome-level rewards [7, 19].

The success of OPSD on LLM reasoning makes it natural to consider OPSD for multimodal large language models (MLLMs). Recent MLLMs are used for visual instruction following, image-text generation, grounding, text reading, and visual question answering [2, 10, 12]. Many of these tasks require multimodal reasoning, such as visual mathematics, college-level multimodal understanding, and logical reasoning over visual inputs [14, 25, 26]. OPSD is attractive in this setting because high-quality multimodal supervision is expensive and still scarcer than text-only supervision [4, 16].

![](images/9a0c6ead6d8525cfc06ee4ab5e68e10693c87fed13a445b2cd9bb28c702148b2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Image I"] --> B["Student"]
  C["Question x"] --> B
  D["Reference Answer a*"] --> E["Teacher"]
  B --> F["Student Rollout"]
  E --> F
  F --> G["Output: 10 blocks"]
    style A fill:#f9f,stroke:#333
    style C fill:#bbf,stroke:#333
    style D fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style E fill:#f9f,stroke:#333
    style F fill:#bbf,stroke:#333
    style G fill:#dfd,stroke:#333
    subgraph Weak supervision
        A
        C
        D
        B
    end
    subgraph Strong supervision
        E
        F
    end
```
</details>

Figure 1: Shortcut risk in vanilla OPSD for MLLMs. The student only sees the image I and question x, but the teacher is also conditioned on the reference answer a⋆. Because MLLMs can be strongly influenced by text and may underuse visual input, the known answer can shape reasoning and the final answer before visual evidence is clearly used. The student may then produce an answer-compatible rationale with weak visual grounding.

A direct transfer of OPSD to MLLMs, however, can create a multimodal shortcut. In vanilla OPSD, the privileged teacher sees the reference target while supervising the whole student rollout. For text-only reasoning, this is a natural way to guide the reasoning path. For MLLMs, the same text signal may be easier to follow than the image. The teacher can push the student toward the known answer before the image content has been checked, so the student may learn answercompatible rationales with weak visual grounding. This concern is related to language-prior and shortcut-learning problems in VQA [5, 17], and to recent findings that vision-language models may trust text over images when the two modalities conflict [3, 29]. Figure 1 gives a simple view of this risk.

To address this challenge, we propose ViGOS (Visual Grounding On-Policy Self-Distillation), an MLLM post-training method that separates perception from reasoning. ViGOS keeps dense on-policy self-distillation, but assigns different teacher contexts to different parts of the student trajectory. The student first writes a visual description, which serves as a grounding interface. An image-only perception teacher supervises this segment using only the image as external evidence. Then, a privileged reasoning teacher uses the reference target to supervise the reasoning and answer segments on the same student-generated prefix. This keeps answer-guided reasoning, but prevents the reference target from directly supervising early visual claims. Finally, a reference teacher is used only for invalid rollouts, thereby limiting format drift and preserving the desired output pattern.

We evaluate ViGOS on a broad set of multimodal reasoning benchmarks covering general vision-language ability [24], expert academic reasoning [26, 27], visual mathematics [14, 28], spatial grounding [18, 20, 23], and visual-language-prior stress tests [15]. The results show that ViGOS preserves the standard benchmark gains of OPSD and improves performance on prior-sensitive evaluations where models must rely on image content rather than textual or dataset shortcuts.

In summary, our contributions are as follows:

• We identify a shortcut risk in multimodal OPSD: answer-conditioned token supervision can shape the response before the model has grounded it in the image.

• We propose ViGOS, an on-policy self-distillation framework that separates image-based perception supervision from answer-conditioned reasoning supervision.  
• Extensive experiments show that ViGOS keeps the main gains of OPSD while improving robustness on prior-sensitive multimodal reasoning benchmarks.

## 2 Preliminaries

## 2.1 Task Definition

We study supervised post-training for multimodal reasoning. The training set is

$$
\mathcal {D} = \{(I _ {i}, x _ {i}, a _ {i} ^ {\star}) \} _ {i = 1} ^ {N}, \tag {1}
$$

where $I _ { i }$ is an image, $x _ { i }$ is a question or instruction, and $a _ { i } ^ { \star }$ is the privileged reference target. In the equations, we write it as a reference answer. In prompts, it can also be a reference solution string when such a solution is available. The student must answer the question using the image, but it never receives $a _ { i } ^ { \star }$ as input.

Let $p _ { \theta }$ be the trainable student MLLM. During training and inference, the student only receives the original image and question:

$$
y \sim p _ {\theta} (\cdot \mid I, x). \tag {2}
$$

Here $y = ( y _ { 1 } , \dots , y _ { T } )$ is a student-generated token sequence, and $\mathcal { T } _ { y } = \{ 1 , . . . , T \}$ is its token-index set. Vanilla OPSD does not require an explicit visual description. It can be applied to a normal response, such as a reasoning process that culminates in a final answer.

For an autoregressive rollout, let $h _ { t } = ( y _ { 1 } , \dots , y _ { t - 1 } )$ be the prefix before token t. The student next-token distribution is

$$
p _ {\theta , t} (\cdot) = p _ {\theta} (\cdot \mid I, x, h _ {t}). \tag {3}
$$

For compact notation, $\mathbb { E } _ { \mathcal { D } , p _ { \theta } }$ denotes expectation over $( I , x , a ^ { \star } ) \sim \mathcal { D }$ and over a rollout sampled from the current student, $y \sim p _ { \boldsymbol { \theta } } ( \cdot \mid I , x )$ . Thus, every prefix $h _ { t }$ is the student’s own prefix.

## 2.2 Vanilla OPSD for MLLMs

On-policy distillation trains the student on prefixes sampled from the student itself, rather than on fixed teacher trajectories [1, 13]. OPSD eliminates the need for an external teacher by using a frozen copy of the teacher’s model. The teacher is conditioned on privileged information, such as the reference target $a ^ { \star }$ , and gives dense token-level supervision [30].

A simplified OPSD objective for MLLMs is

$$
\mathcal {L} _ {\mathrm{OPSD}} = \mathbb {E} _ {\mathcal {D}, p _ {\theta}} \left[ \sum_ {t \in \mathcal {T} _ {y}} D _ {\mathrm{KL}} \left(q _ {\text { priv }, t} \| p _ {\theta , t}\right) \right], \tag {4}
$$

where

$$
q _ {\text { priv }, t} (\cdot) = p _ {\bar {\theta}} (\cdot \mid I, x, a ^ {\star}, h _ {t}), \tag {5}
$$

and $\bar { \theta }$ denotes detached teacher parameters. We omit prompt templates for readability. The key point is that the teacher can see $a ^ { \star }$ when it scores every student prefix. This provides dense feedback on the states the student actually visits, which is the main advantage of OPSD.

## 2.3 Shortcut Risk in Multimodal OPSD

PALR diagnostic. We start the shortcut analysis by defining the Privileged Answer Leakage Rate (PALR). PALR asks the following question: when a method yields a dense token-level correction on a fixed student rollout, how much of that correction is attributable to the privileged answer rather than to the image?

For a method $M$ , let $q _ { t } ^ { M }$ be the active teacher distribution for token t on the student prefix $h _ { t }$ . For vanilla OPSD, this active teacher is the full privileged teacher for all supervised tokens:

$$
q _ {t} ^ {\mathrm{OPSD}} (\cdot) = p _ {\bar {\theta}} (\cdot \mid I, x, a ^ {\star}, h _ {t}). \tag {6}
$$

PALR keeps the rollout y and all prefixes $h _ { t }$ fixed. It only changes the teacher context, so the diagnostic compares supervision signals on the same student states.

Let $y _ { t }$ be the observed token. The active correction strength is

$$
s _ {t} = \left| \log q _ {t} ^ {M} (y _ {t}) - \log p _ {\theta , t} (y _ {t}) \right|. \tag {7}
$$

To measure answer-driven support, we replace the correct answer in the active teacher context with wrong-answer counterfactuals and denote the resulting teacher by $q _ { \mathrm { w r o n g } , t } ^ { M }$ . If the active teacher does not receive $a ^ { \star }$ , this support is set to zero. Otherwise,

$$
c _ {t} ^ {A} = \left[ \log q _ {t} ^ {M} (y _ {t}) - \log q _ {\mathrm{wrong}, t} ^ {M} (y _ {t}) \right] _ {+}. \tag {8}
$$

To measure image-driven support, we replace the image with a mismatched image and denote the resulting teacher by qMimgcf,t: $q _ { \mathrm { i m g c f } , t } ^ { M } \mathrm { : }$

$$
c _ {t} ^ {I} = \left[ \log q _ {t} ^ {M} (y _ {t}) - \log q _ {\mathrm{imgcf}, t} ^ {M} (y _ {t}) \right] _ {+}. \tag {9}
$$

The positive part $[ \cdot ] _ { + }$ means that we only count cases where the original teacher gives more support to the observed token. For a token group $G \subseteq \mathcal { T } _ { y }$ , PALR is

$$
\operatorname{PALR} (G) = \frac {\sum_ {t \in G} s _ {t} c _ {t} ^ {A}}{\sum_ {t \in G} s _ {t} \left(c _ {t} ^ {A} + c _ {t} ^ {I}\right) + \epsilon_ {\text {PALR}}}, \tag {10}
$$

where $\epsilon _ { \mathrm { P A L R } }$ is a small numerical constant. A higher PALR means that a larger share of the dense correction is tied to the privileged answer under this diagnostic. It is not a complete attribution of all possible shortcuts. Appendix A gives the full implementation details.

Observations. We observe clear answer leakage in vanilla OPSD. On a 1,000-sample Qwen2.5-VL diagnostic subset per scale, PALR $\left( \mathcal { T } _ { r a } \right)$ is 17.26% for 3B and 26.01% for 7B, as shown in Figure 2. Here $\mathcal { T } _ { r a }$ denotes the tokens parsed as reasoning or final answer. For vanilla OPSD, these segment labels are used only for analysis; the teacher still supervises the entire rollout with $a ^ { \star }$ .

This means that a notable portion of the dense reasoning-answer correction changes when the privileged answer is replaced with incorrect answers. Some dependence on answers is useful and expected, as the reference target should guide reasoning. The concern is where this dependence appears: in vanilla OPSD, answer-driven correction can affect the reasoning-answer segment, even if the response is not fully supported by the image’s visual evidence.

![](images/a590d61d5e37cc69bf830f969874715eb8ae3dd163f352752f1e95be6ea908c1.jpg)

<details>
<summary>scatterplot</summary>

|        | OPSD  | ViGOS |
| ------ | ----- | ----- |
| 3B     | 2.92  | 0.00  |
| 7B     | 2.94  | 0.00  |
| Tra    | 17.26 | 6.33  |
| 7B     | 26.01 | 7.56  |
| Ty     | 5.59  | 3.07  |
| 7B     | 7.55  | 3.72  |
</details>

Figure 2: PALR diagnostic results on Qwen2.5-VL. All numbers are percentages (%). $\mathcal { T } _ { d }$ is the visual description segment introduced by ViGOS, $\mathcal { T } _ { r a }$ is the merged reasoning-answer segment used in this diagnostic, and $\mathcal { T } _ { y }$ is the full rollout. A lower PALR indicates less answer-driven supervision under this diagnostic.

Analysis. This PALR pattern matches a shortcut risk in multimodal OPSD. In text-only OPSD, allowing the teacher to see a privileged reference answer is a natural design choice. The teacher uses this extra text signal to guide the student’s own prefixes. For MLLMs, the same design is more fragile because the model receives both image and text inputs, and text is often easier to use. The question, answer options when present, and privileged target can form a text path that competes with the image.

Under vanilla OPSD, the teacher sees $a ^ { \star }$ when computing the next-token distribution for every step. When the student is writing the reasoning part, the teacher may prefer a token because it agrees with the known answer, even if the prefix has not stated the needed visual evidence. This creates an answer-driven path:

$$
a ^ {\star} \rightarrow r \rightarrow a. \tag {11}
$$

The student is trained to follow this dense signal on its own prefixes. If the signal is answer-driven, the student can learn a rationale that fits the answer while using the image less [5, 17].

The issue is not that privileged supervision is useless or harmful by itself. The answer signal is valuable for teaching reasoning. The issue is that vanilla OPSD applies the same answer-conditioned teacher to all tokens. Visual grounding and answer-conditioned reasoning are combined into a single supervision signal, so the known answer can shape reasoning before visual evidence is made explicit. This motivates ViGOS. Our goal is to keep answer-guided reasoning while controlling where the privileged answer enters the trajectory.

## 3 ViGOS: Visual Grounding On-Policy Self-Distillation

## 3.1 Overview

ViGOS trains the student on its own sampled responses, but changes how privileged information enters token-level supervision. For each image-question pair, the student is asked to generate a structured sequence

$$
y = (d, r, a), \tag {12}
$$

![](images/a5cae36a03f610cc7ad7ac149a618d4710d2881dea7e6a175db4e699ebe72661.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Dataset D = {(I_i, x_i, a_i^*)_{i=1}^N}"] --> B["Trainable Student p_θ"]
  C["Reference Answer a_i^+"] --> B
  D --> E["Structured Rollout y = (d, r, a)"]
  E --> F["Student sees only image and question"]
  G["I"] --> B
  H["x"] --> B
  I["?"] --> B
    J["To Privileged Reasoning Teacher"] -.-> A
  K["Student <description> d <think> r < / think > a"] --> E
```
</details>

![](images/0e2dcb91b00db20501e467c8fdce8596a09b7e868005200897ff1ecaacc7b882.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Start"] --> B["Rollout y"]
  B --> C["Segment Parser"]
  C --> D1["Description Tokens"]
  C --> D2["Reasoning Tokens"]
  C --> D3["Answer Tokens"]
  C --> D4["Invalid Flag"]
  D1 --> E1["Token Mask"]
  D2 --> E2["Token Mask"]
  D3 --> E3["Token Mask"]
  D4 --> E4["Token Mask"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D1 fill:#ffc,stroke:#333
    style D2 fill:#ffc,stroke:#333
    style D3 fill:#ffc,stroke:#333
    style D4 fill:#ffc,stroke:#333
```
</details>

![](images/56c0ccfc83352ffb506efdf8c653ea169ec97fb5369ade59058afba97af96efd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Stage 3: Segment-wise Teachers"] --> B["Valid Rollout: m_inv(y) = 0"]
  B --> C["Image-only Perception Teacher"]
  C --> D["L_perc = Σ t∈T_a D_KL(q_img_t || p_θ,t)"]
  D --> E["Guides reasoning after student description"]
  F["Invalid Rollout: m_inv(y) = 1"] --> G["T_y (All Rollout Tokens)"]
  G --> H["Reverse-KL fallback for malformed outputs"]
  H --> I["L_ref = Σ t∈T_y D_KL(p_θ,t || q_ref,t)"]
  I --> J["L_ViGOS = λ_perc L_perc + λ_rea L_rea + λ_ref L_ref"]
```
</details>

Figure 3: Training pipeline of ViGOS. Given an image I and question $x ,$ the student samples a structured trajectory $y = ( d , r , a )$ , where d is a visual description, $r$ is the reasoning process, and a is the final answer. A parser extracts token masks $\mathcal { T } _ { d } , \mathcal { T } _ { r }$ , and $\mathcal { T } _ { a }$ and detects invalid rollouts. For valid rollouts, an image-only perception teacher supervises $d ,$ and a privileged reasoning teacher supervises $( r , a )$ on the same student prefix. For invalid rollouts, a reference teacher provides a fallback signal. All teachers are removed at inference time.

where d is a visual description, r is the reasoning process, and $a$ is the final answer. The tuple denotes three ordered parts of the same token sequence. The description is generated by the student from the image and question, so it is available at inference time and is not an extra annotation.

We prompt the student to generate responses in the following output format:

```txt
<description> d </description>
    <think> r </think>
    a.
```

For a valid rollout, the parser returns the content-token sets $\mathcal { T } _ { d } , \ \mathcal { T } _ { r } .$ and $\mathcal { T } _ { a }$ . These sets are disjoint subsets of the full token-index set $\mathcal { T } _ { y } = \{ 1 , \dots , T \}$ . We also write $\mathcal { T } _ { r a } = \mathcal { T } _ { r } \cup \mathcal { T } _ { a }$ . Delimiter tokens stay in $\mathcal { T } _ { y }$ but are excluded from the segment losses.

Given a student rollout, ViGOS uses three teacher roles from frozen copies of the same initial MLLM: 1) an image-only perception teacher $q _ { \mathrm { i m g } }$ for $d ; 2 )$ a privileged reasoning teacher $q _ { \mathrm { r e a } }$ for $( r , a )$ ; and 3) a reference teacher $q _ { \mathrm { r e f } }$ for invalid outputs. All teacher roles are frozen and detached from gradient updates, following the self-distillation setting of OPSD [30]. They differ only in the external context that they can access.

Figure 3 summarizes the pipeline. The student first samples one structured trajectory. The parser then builds token masks and checks validity. If the rollout is valid, $q _ { \mathrm { i m g } }$ supervises the description tokens with image-only external context, while $q _ { \mathrm { r e a } }$ supervises the later tokens with access to the reference target. If the rollout is invalid, qref gives a fallback signal over the whole trajectory. At inference time, only the student is used.

We define the external contexts as

$$
c _ {\mathrm{stu}} = (I, x, \pi_ {\mathrm{out}}),
$$

$$
c _ {\mathrm{img}} = I, \tag {13}
$$

$$
c _ {\mathrm{rea}} = (I, x, a ^ {\star}, \pi_ {\mathrm{rea}}),
$$

$$
c _ {\text { ref }} = (I, x, a ^ {\star}, \pi_ {\text { out }}),
$$

where $\pi _ { \mathrm { o u t } }$ asks for the structured output format, and $\pi _ { \mathrm { r e a } }$ asks for answer-consistent reasoning. The image-only external context $c _ { \mathrm { i m g } }$ contains no question text, answer options when present, or reference target. It still uses the same student prefix $h _ { t }$ , so image-only means that no extra question or answer evidence is added outside the prefix. Let $h _ { t } = ( y _ { 1 } , \dots , y _ { t - 1 } )$ be the generated prefix before token t. The student’s next-token distribution is

$$
p _ {\theta , t} (\cdot) = p _ {\theta} (\cdot \mid c _ {\mathrm{stu}}, h _ {t}). \tag {14}
$$

For the same prefix, the teacher distributions are

$$
q _ {\mathrm{img}, t} (\cdot) = p _ {\bar {\theta}} (\cdot \mid c _ {\mathrm{img}}, h _ {t}),
$$

$$
q _ {\text {rea}, t} (\cdot) = p _ {\bar {\theta}} (\cdot \mid c _ {\text {rea}}, h _ {t}), \tag {15}
$$

$$
q _ {\text { ref }, t} (\cdot) = p _ {\bar {\theta}} (\cdot \mid c _ {\text { ref }}, h _ {t}).
$$

Thus, supervision remains on-policy: every teacher is queried on a prefix that the current student actually produced.

## 3.2 Student Rollout and Parsing

During training, the student samples

$$
y \sim p _ {\theta} (\cdot \mid c _ {\mathrm{stu}}). \tag {16}
$$

In the losses below, $\mathbb { E } _ { \mathcal { D } , p _ { \theta } }$ denotes this sampling process together with $( I , x , a ^ { \star } ) \sim \mathcal { D }$ . The student never observes $a ^ { \star }$ as input.

After sampling, we parse the delimiters and the final · answer. A rollout is valid when the required delimiters are present, the description and reasoning segments are non-empty, and the final answer can be parsed. We define

$$
m _ {\text { inv }} (y) = \mathbf {1} [ y \notin \mathcal {Y} _ {\text { valid }} ], \tag {17}
$$

where $\mathcal { \mathrm { V } } _ { \mathrm { v a l i d } }$ is the set of valid-format outputs. Wrong answers are still valid if their format can be parsed. For invalid rollouts, the segment masks are empty and the fallback acts on $\mathcal { T } _ { y }$ . The teachers do not generate replacement trajectories; they only provide token-level distributions on student prefixes.

## 3.3 Training Objectives

Perception loss. The perception teacher supervises only the description tokens. For $t \in \mathcal { T } _ { d }$ , its distribution is $q _ { \mathrm { i m g , } t }$ . The loss is

$$
\mathcal {L} _ {\text { perc }} = \mathbb {E} _ {\mathcal {D}, p _ {\theta}} \left[ (1 - m _ {\text { inv }} (y)) \sum_ {t \in \mathcal {T} _ {d}} D _ {\mathrm{KL}} \left(q _ {\mathrm{img}, t} \| p _ {\theta , t}\right) \right]. \tag {18}
$$

Because $q _ { \mathrm { i m g } }$ does not receive $x$ or $a ^ { \star }$ as external context, this loss does not directly teach the answer. Its role is to keep d close to an image-based description distribution, a capability inherent in MLLMs acquired during the visual instruction-tuning stage [2, 12, 22].

Reasoning loss. The reasoning teacher supervises the reasoning and answer tokens. For $t \in \mathcal { T } _ { r } \cup \mathcal { T } _ { a }$ , the token-level teacher distribution is $q _ { \mathrm { r e a } , t }$ . The loss is

$$
\mathcal {L} _ {\mathrm{rea}} = \mathbb {E} _ {\mathcal {D}, p _ {\theta}} \left[ (1 - m _ {\text {inv}} (y)) \sum_ {t \in \mathcal {T} _ {r} \cup \mathcal {T} _ {a}} D _ {\mathrm{KL}} \left(q _ {\mathrm{rea}, t} \| p _ {\theta , t}\right) \right]. \tag {19}
$$

This teacher can use $a ^ { \star }$ , but only for $\mathcal { T } _ { r } \cup \mathcal { T } _ { a }$ on valid rollouts. Since it is queried on $h _ { t } .$ , it also conditions on the student’s own earlier description, even when that description is imperfect.

Reference fallback loss. A full privileged teacher can help maintain the output format, but using it as the main teacher would again supervise the whole trajectory with the reference target. We therefore use the reference teacher only when parsing fails. For invalid rollouts, we apply a reverse-KL regularizer:

$$
\mathcal {L} _ {\mathrm{ref}} = \mathbb {E} _ {\mathcal {D}, p _ {\theta}} \left[ m _ {\mathrm{inv}} (y) \sum_ {t \in \mathcal {T} _ {y}} D _ {\mathrm{KL}} \left(p _ {\theta , t} \| q _ {\mathrm{ref}, t}\right) \right]. \tag {20}
$$

This term is a recovery signal for malformed outputs. It is inactive on valid rollouts, where perception and reasoning are handled by their segment-specific teachers.

Overall objective. The final objective is

$$
\mathcal {L} _ {\mathrm{ViGOS}} = \lambda_ {\text {perc}} \mathcal {L} _ {\text { perc }} + \lambda_ {\text { rea }} \mathcal {L} _ {\text { rea }} + \lambda_ {\text { ref }} \mathcal {L} _ {\text { ref }}, \tag {21}
$$

where $\lambda _ { \mathrm { p e r c } } , \lambda _ { \mathrm { r e a } } .$ , and $\lambda _ { \mathrm { r e f } }$ control the three losses. When the rollout is valid, the perception and reasoning losses are active and the fallback is zero. When it is invalid, the segment masks are unreliable, so only the fallback is active. In practice, each active KL sum is normalized by the number of tokens in the supervised segment.

At inference time, all teachers are discarded. The final model receives only the image, the question, and the output-format prompt, and it generates $( d , r , a )$ with the student policy.

## 3.4 Effect of Decoupling Perception from Reasoning

The main difference from vanilla OPSD is the path by which $a ^ { \star }$ enters training. In OPSD, the reference target can affect every token in the rollout. In ViGOS, it is used after the student has produced d:

$$
I \rightarrow d \rightarrow r \rightarrow a. \tag {22}
$$

This does not remove answer guidance. It controls where that guidance is applied, so early visual claims are not directly matched to an answer-conditioned teacher.

The PALR diagnostic in Figure 2 is consistent with this design: compared with vanilla OPSD, $\mathrm { P A L R } ( \mathcal { T } _ { r a } )$ drops from 17.26% to 6.33% on 3B and from 26.01% to 7.56% on 7B. The full-rollout PALR also drops from 5.59% to 3.07% on 3B and from 7.55% to 3.72% on 7B. These numbers suggest that ViGOS keeps useful answer-conditioned supervision while reducing answer-dominated corrections under this diagnostic.

Table 1: Results on the eight main benchmarks. We report Pass@5 / Avg@5 as percentages (%).

<table><tr><td rowspan="2">Model</td><td>General VL</td><td colspan="2">Expert/Academic Reasoning</td><td colspan="2">Visual Math</td><td colspan="3">Spatial &amp; Vision Grounding</td></tr><tr><td>MM-Vet</td><td>MMMU</td><td>MMMU-Pro</td><td>MathVerse</td><td>MathVista</td><td>MMSI</td><td>RealWorldQA</td><td>CV-Bench</td></tr><tr><td colspan="9">RL Methods</td></tr><tr><td>Visionary-R1-3B</td><td>64.22 / 49.27</td><td>70.28 / 43.49</td><td>52.71 / 27.10</td><td>55.71 / 33.76</td><td>72.00 / 57.18</td><td>58.50 / 25.18</td><td>82.88 / 57.67</td><td>88.25 / 70.33</td></tr><tr><td>Vision-R1-7B</td><td>73.39 / 59.54</td><td>64.69 / 47.58</td><td>47.29 / 31.48</td><td>63.71 / 47.48</td><td>77.30 / 63.92</td><td>40.50 / 24.64</td><td>75.42 / 66.95</td><td>83.47 / 74.81</td></tr><tr><td colspan="9">Backbone: Qwen2.5-VL 3B</td></tr><tr><td>Baseline</td><td>62.39 / 34.68</td><td>71.51 / 33.54</td><td>55.00 / 22.55</td><td>60.61 / 30.18</td><td>65.40 / 35.08</td><td>53.20 / 16.78</td><td>38.17 / 15.76</td><td>80.59 / 34.68</td></tr><tr><td>OPSD</td><td>68.81 / 45.69</td><td>76.42 / 42.70</td><td>57.04 / 26.24</td><td>59.54 / 30.45</td><td>72.80 / 43.84</td><td>63.60 / 23.68</td><td>86.93 / 53.02</td><td>91.47 / 63.43</td></tr><tr><td>ViGOS (Ours)</td><td>65.60 / 43.76</td><td>76.42 / 42.32</td><td>56.44 / 26.16</td><td>58.55 / 30.10</td><td>74.00 / 43.50</td><td>66.40 / 24.90</td><td>86.80 / 55.37</td><td>91.51 / 64.67</td></tr><tr><td colspan="9">Backbone: Qwen2.5-VL 7B</td></tr><tr><td>Baseline</td><td>69.72 / 52.94</td><td>77.77 / 50.30</td><td>63.85 / 37.41</td><td>68.40 / 45.63</td><td>79.20 / 60.90</td><td>63.20 / 27.10</td><td>32.55 / 12.94</td><td>90.37 / 75.85</td></tr><tr><td>OPSD</td><td>70.18 / 52.75</td><td>77.99 / 50.99</td><td>63.37 / 36.91</td><td>68.65 / 45.57</td><td>80.50 / 61.54</td><td>60.30 / 25.68</td><td>85.62 / 61.20</td><td>90.26 / 74.32</td></tr><tr><td>ViGOS (Ours)</td><td>72.02 / 54.40</td><td>80.11 / 51.42</td><td>64.81 / 36.48</td><td>68.91 / 44.77</td><td>80.90 / 58.78</td><td>61.10 / 25.58</td><td>85.88 / 62.88</td><td>91.09 / 73.58</td></tr></table>

## 4 Experiments

We organize the experiments around three questions:

RQ1: Does ViGOS keep the overall benchmark gains of OPSD on standard multimodal reasoning evaluations?

RQ2: Does ViGOS improve prior-sensitive image use, where the image may conflict with common visual-language priors?

RQ3: Are the perception teacher, the reasoning teacher, and the reference fallback each necessary for the final behavior?

The main benchmark results answer RQ1, the ViLP results and training dynamics answer RQ2, and the ablation studies answer RQ3.

## 4.1 Experimental Setup

We use Qwen2.5-VL-3B-Instruct and Qwen2.5-VL-7B-Instruct as the backbone models [2]. For each backbone, we compare three models: the original instruction-tuned model, denoted as Baseline; OPSD, which applies on-policy self-distillation with a privileged teacher [30]; and our method, ViGOS. OPSD and ViGOS use the same post-training data and training budget. We also include Visionary-R1-3B and Vision-R1-7B as RL-based reference models [9, 21]. Because these RL models may use different data and recipes, the primary controlled comparison is among Baseline, OPSD, and ViGOS, all using the same Qwen2.5-VL backbone.

The main benchmark suite contains eight evaluations: MM-Vet [24], MMMU [26], MMMU-Pro [27], MathVerse [28], MathVista [14], MMSI [23], RealWorldQA [20], and CV-Bench [18]. We further evaluate on ViLP [15], which asks whether a model follows the image when it conflicts with a common visual-language prior. More benchmark details are provided in Appendix B. For the eight main benchmarks, we sample five responses per example and report Pass@5 and Avg@5. For ViLP, we report Score and Prior as defined in the benchmark.

We train ViGOS-3B and ViGOS-7B for one epoch on Vision-SR1-47K [11] using 8 NVIDIA A100 GPUs. Full hyperparameters and prompts are provided in Appendix E.

## 4.2 Main Results: RQ1

Table 1 reports the results on the eight main benchmarks. ViGOS clearly improves over the original backbones. The mean Pass@5 over the eight benchmarks rises from 60.86% to 71.97% on 3B and from 68.13% to 75.60% on 7B. The mean Avg@5 also increases from 27.91% to 41.35% on 3B and from 45.38% to 50.99% on 7B. These gains show that ViGOS improves both sampled success and average response quality. Compared with OPSD, ViGOS keeps the overall benefit of on-policy self-distillation. On the 3B backbone, it is close to OPSD in mean Pass@5 and slightly better in mean Avg@5. On the 7B backbone, it gives the best mean Pass@5 and nearly the same mean Avg@5 as OPSD. This answers RQ1: the proposed decoupling does not remove the standard benchmark gains of OPSD.

Table 2: ViLP results for prior-sensitive evaluation. ViLP-F is the with-fact setting, where the prompt provides additional facts, while ViLP-P is the pure-question setting.

<table><tr><td rowspan="2">Model</td><td colspan="2">ViLP-F</td><td colspan="2">ViLP-P</td></tr><tr><td>Score</td><td>Prior</td><td>Score</td><td>Prior</td></tr><tr><td colspan="5">RL Methods</td></tr><tr><td>Visionary-R1-3B</td><td>64.67</td><td>94.67</td><td>65.17</td><td>88.00</td></tr><tr><td>Vision-R1-7B</td><td>57.17</td><td>95.67</td><td>57.83</td><td>90.00</td></tr><tr><td colspan="5">Backbone: Qwen2.5-VL 3B</td></tr><tr><td>Baseline</td><td>59.50</td><td>93.33</td><td>55.67</td><td>80.67</td></tr><tr><td>OPSD</td><td>67.17</td><td>97.33</td><td>66.83</td><td>92.00</td></tr><tr><td>ViGOS (Ours)</td><td>70.17</td><td>97.67</td><td>69.50</td><td>90.00</td></tr><tr><td colspan="5">Backbone: Qwen2.5-VL 7B</td></tr><tr><td>Baseline</td><td>42.00</td><td>73.33</td><td>37.00</td><td>58.67</td></tr><tr><td>OPSD</td><td>58.00</td><td>97.67</td><td>57.00</td><td>91.67</td></tr><tr><td>ViGOS (Ours)</td><td>62.67</td><td>97.00</td><td>61.67</td><td>91.67</td></tr></table>

The gains are strongest on benchmarks that need concrete image understanding. For 3B, the largest Pass@5 gains over Baseline appear on RealWorldQA, MMSI, CV-Bench, and MathVista, and ViGOS gives the best Pass@5 on MMSI and CV-Bench. For 7B, ViGOS improves Pass@5 over Baseline on MM-Vet, MMMU, MMMU-Pro, MathVerse, MathVista, RealWorldQA, and CV-Bench, although Avg@5 can still drop on some tasks. This suggests that ViGOS primarily supports imagegrounded multimodal reasoning, while harder symbolic reasoning and response stability may still require additional optimization.

## 4.3 Prior-Sensitive Evaluation: RQ2

Table 2 evaluates shortcut behavior on ViLP. A higher Score indicates stronger image-grounded reasoning under prior conflict. A high Prior indicates that the model preserves useful prior-aligned knowledge and instruction-following ability instead of simply suppressing priors. Therefore, a desirable model should improve Score while keeping Prior largely unchanged.

ViGOS obtains the best Score in all settings and outperforms OPSD on both ViLP-F and ViLP-P. The improvement is especially clear on the 7B backbone, where the average Score increases from 39.50 for Baseline to 62.17 for ViGOS. This answers RQ2: ViGOS improves performance when the model must choose the image-supported answer instead of the common prior. At the same time, ViGOS keeps high Prior accuracy. For 3B, Prior stays around 90-98 across the two ViLP settings. For 7B, ViGOS gives 97.00 on ViLP-F Prior and ties OPSD at 91.67 on ViLP-P Prior. Thus, the method does not simply suppress prior knowledge. It helps the model keep useful priors while relying more on the image when the two signals disagree. We provide qualitative analysis on ViLP in Appendix D.

We further conduct a same-prompt comparison experiment in Appendix C, which trains and tests vanilla OPSD and Baseline with the same prompt as ViGOS. The prompt boosts Baseline’s performance, while OPSD’s performance degrades below the Baseline. ViGOS still achieves the strong outcomes across all main benchmarks. It also improves ViLP Score while keeping Prior high. This supports our interpretation that the gain primarily comes from separating visual perception supervision from answer-conditioned reasoning supervision, rather than from the prompt alone.

Table 3: Ablation on the perception loss, reasoning loss, and reference fallback. All models use the same Qwen2.5-VL 3B backbone and training data. For Overall and CV-Bench, we report Pass@5. For ViLP, we report the average Score. Overall aggregates all evaluation examples used in the ablation.  
(a) Loss ablation

<table><tr><td>Variant</td><td>Overall</td><td>CV-Bench</td><td>ViLP</td></tr><tr><td>Full ViGOS</td><td>74.91</td><td>91.51</td><td>69.84</td></tr><tr><td>w/o Perception loss</td><td>74.81</td><td>91.09</td><td>67.58</td></tr><tr><td>w/o Reasoning loss</td><td>74.71</td><td>90.37</td><td>69.42</td></tr></table>

(b) Reference fallback

<table><tr><td>Variant</td><td>Overall</td><td>CV-Bench</td><td>ViLP</td></tr><tr><td>Ref. reverse KL</td><td>74.91</td><td>91.51</td><td>69.84</td></tr><tr><td>Ref. forward KL</td><td>74.82</td><td>90.86</td><td>69.33</td></tr><tr><td>w/o Ref. teacher</td><td>73.60</td><td>90.71</td><td>63.25</td></tr></table>

## 4.4 Ablation Studies: RQ3

Effect of the perception and reasoning losses. We first ablate the two valid-rollout losses in ViGOS-3B. The $\mathrm { w / o }$ Perception loss variant removes the image-only teacher on description tokens, and the $\mathrm { w / o }$ Reasoning loss variant removes the privileged teacher on reasoning and answer tokens. The reference fallback is unchanged in both variants.

Table 3a shows that the full model gives the best Overall, CV-Bench, and ViLP results. Removing either loss only slightly changes Overall, but their detailed effects are different. Without the perception loss, ViLP drops from 69.84 to 67.58, and CV-Bench decreases as well. This supports the role of the perception teacher: it keeps the description segment tied to the image before answer-conditioned reasoning is used.

The reasoning loss helps convert the description into the final answer. When it is removed, Overall Pass@5 and CV-Bench both drop. The ViLP decrease is smaller, consistent with the shortcut concern: weakening the privileged reasoning signal reduces exposure to answer-driven guidance but also removes useful answer supervision. Keeping both losses gives the best balance.

Effect of the reference fallback. We next study the fallback design in ViGOS-3B. The proposed version uses the reference teacher only for invalid rollouts and applies token-level reverse KL, $D _ { \mathrm { K L } } ( p _ { \theta , t } \Vert q _ { \mathrm { r e f } , t } )$ . We compare it with a forward-KL fallback and with a variant that removes the reference teacher. In the latter variant, invalid rollouts are supervised by the perception and reasoning teachers on all tokens.

As shown in Table 3b, the separate fallback is necessary. Removing the reference teacher gives the lowest Overall and a much lower ViLP score, dropping from 69.84 to 63.25. This is expected because invalid rollouts lack reliable segment masks. If the segment teachers supervise all tokens in this case, their roles can be mixed, and the reference target can again affect tokens that should be handled by the perception stage.

![](images/fbf066cf25030912ddc7a1839f0a9701b9b9909fe0ee9f743eb14a67749687f7.jpg)

<details>
<summary>line chart</summary>

| Gradient Descent Step | ViGOS | OPSD  |
| --------------------- | ----- | ----- |
| 0                     | 0.81  | 0.81  |
| 25                    | 0.89  | 0.905 |
| 50                    | 0.90  | 0.885 |
| 75                    | 0.89  | 0.89  |
| 100                   | 0.89  | 0.895 |
</details>

(a) Prior

![](images/d1e0a4f217f10e318becda8be59032c23597adb160b8d6dbf4f4a9066e438d39.jpg)

<details>
<summary>line chart</summary>

| Gradient Descent Step | ViGOS | OPSD  |
| --------------------- | ----- | ----- |
| 0                     | 0.56  | 0.56  |
| 25                    | 0.685 | 0.645 |
| 50                    | 0.695 | 0.635 |
| 75                    | 0.695 | 0.625 |
| 100                   | 0.705 | 0.635 |
</details>

(b) Score  
Figure 4: Step-wise comparison between OPSD and ViGOS on ViLP. Prior measures accuracy on prior-aligned questions, while Score measures accuracy on image-supported questions.

Forward KL is close on Overall, but it is lower than reverse KL on CV-Bench and ViLP. We therefore keep reverse KL. The fallback is not meant to teach a full solution; it mainly moves malformed continuations back toward a parseable output pattern. Together, these ablations answer RQ3: all three components are useful, and the reference fallback should remain a limited recovery signal.

Step-wise analysis on ViLP. Figure 4 shows the training dynamics on ViLP. At step 0, OPSD and ViGOS start from the same model. After training begins, both methods keep high Prior, but their Score trends differ. OPSD improves at first and then drops to around 0.63. In contrast, ViGOS keeps improving and reaches 0.71 at 100 steps.

This trend matches the design goal. OPSD uses a single privileged teacher across the entire trajectory, which can cause the student to follow the reference target too early. ViGOS separates the image-based and answer-conditioned components of supervision, allowing the model to retain useful priors while making better use of the image in conflict cases.

## 5 Conclusion

This paper studies a focused problem in multimodal OPSD: dense answer-conditioned supervision is useful, but it can also guide the response before the model has described the image. We propose ViGOS to change this supervision path. The student first writes a visual description, which is supervised by an image-only perception teacher. The teacher then supervises the reasoning and answers only after the prefix is in place. A reference teacher is kept as a limited fallback for invalid rollouts. ViGOS keeps the main benchmark gains of OPSD while reducing answerdominated corrections and improving image-grounded answering. The method still has limitations. The generated description may be incomplete or incorrect; the image-only teacher may produce generic descriptions; and training requires extra teacher forward passes. Even with these limits, the results show that separating description and reasoning is a useful approach for adapting OPSD to multimodal reasoning.

## References

[1] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from selfgenerated mistakes. In ICLR, 2024.  
[2] Shuai Bai, Keqin Chen, Xuejing Liu, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.  
[3] Ailin Deng, Tri Cao, Zhirui Chen, and Bryan Hooi. Words or vision: Do vision-language models have blind faith in text? In CVPR, pages 3867–3876, 2025.  
[4] Hongyuan Dong, Zijian Kang, Weijie Yin, Xiao Liang, Chao Feng, and Jiao Ran. Scalable vision language model training via high quality data curation. In ACL, pages 33272–33293, 2025.  
[5] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.  
[6] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In ICLR, 2024.  
[7] Daya Guo, Dejian Yang, Haowei Zhang, et al. DeepSeek-R1: Incentivizing reasoning capability in llms via reinforcement learning. Nature, 645:633–638, 2025.  
[8] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.  
[9] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Xu Tang, Yao Hu, and Shaohui Lin. Vision-R1: Incentivizing reasoning capability in multimodal large language models. In ICLR, 2026.  
[10] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, volume 202, pages 19730–19742, 2023.  
[11] Zongxia Li, Wenhao Yu, Chengsong Huang, Zhenwen Liang, Rui Liu, Fuxiao Liu, Jingxi Che, Dian Yu, Jordan Boyd-Graber, Haitao Mi, and Dong Yu. Vision-SR1: Self-rewarding vision-language model via reasoning decomposition. In ICLR, 2026.  
[12] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.  
[13] Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025.  
[14] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. In ICLR, 2024.  
[15] Tiange Luo, Ang Cao, Gunhee Lee, Justin Johnson, and Honglak Lee. Probing visual language priors in VLMs. In ICML, volume 267, pages 41120–41156, 2025. ViLP-F and ViLP-P are the with-fact and pure-question evaluation settings of ViLP.  
[16] Bardia Safaei, Faizan Siddiqui, Jiacong Xu, Vishal M. Patel, and Shao-Yuan Lo. Filter images first, generate instructions later: Pre-instruction data selection for visual instruction tuning. arXiv preprint arXiv:2503.07591, 2025.  
[17] Qingyi Si, Fandong Meng, Mingyu Zheng, Zheng Lin, Yuanxin Liu, Peng Fu, Yanan Cao, Weiping Wang, and Jie Zhou. Language prior is not the only shortcut: A benchmark for shortcut learning in vqa. In Findings of EMNLP, 2022.  
[18] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Ziteng Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal LLMs. In NeurIPS, volume 37, 2024.  
[19] Xumeng Wen, Zihan Liu, Shun Zheng, Shengyu Ye, Zhirong Wu, Yang Wang, Zhijian Xu, Xiao Liang, Junjie Li, Ziming Miao, Jiang Bian, and Mao Yang. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025.  
[20] xAI. RealWorldQA: A benchmark for real-world spatial understanding, 2024.  
[21] Jiaer Xia, Yuhang Zang, Peng Gao, Sharon Li, and Kaiyang Zhou. Visionary-R1: Mitigating shortcuts in visual reasoning with reinforcement learning. arXiv preprint arXiv:2505.14677, 2025.  
[22] An Yang, Anfeng Li, Baosong Yang, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.  
[23] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, Dahua Lin, Tai Wang, and Jiangmiao Pang. MMSI-bench: A benchmark for multi-image spatial intelligence. In ICLR, 2026.  
[24] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. MM-Vet: Evaluating large multimodal models for integrated capabilities. In ICML, volume 235, pages 57730–57754, 2024.  
[25] Jiakang Yuan, Tianshuo Peng, Yilei Jiang, Yiting Lu, Renrui Zhang, Kaituo Feng, Chaoyou Fu, Tao Chen, Lei Bai, Bo Zhang, and Xiangyu Yue. MME-reasoning: A comprehensive benchmark for logical reasoning in mllms. arXiv preprint arXiv:2505.21327, 2025.  
[26] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In CVPR, pages 9556–9567, 2024.  
[27] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. MMMU-pro: A more robust multi-discipline multimodal understanding benchmark. In ACL, pages 15134–15186, 2025.  
[28] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, and Hongsheng Li. MathVerse: Does your multi-modal LLM truly see the diagrams in visual math problems? In ECCV, pages 169–186, 2024.  
[29] Haozhe Zhao, Shuzheng Si, Liang Chen, Yichi Zhang, Maosong Sun, Mingjia Zhang, and Baobao Chang. Looking beyond text: Reducing language bias in large vision-language models via multimodal dual-attention and soft-image guidance. In EMNLP, pages 19666–19690, 2025.  
[30] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

## A Privileged Answer Leakage Rate

This section gives the diagnostic used in Section 2.3 and Section 3.4. The goal is to ask a simple question: when a method yields a dense token-level correction during a student rollout, how much of that correction is attributable to the privileged answer rather than to the image? This diagnostic is mainly used to analyze answer leakage in reasoning supervision. We also report the description segment because ViGOS explicitly introduces it.

Setup. For each sample $( I , x , a ^ { \star } )$ , the student first samples one structured rollout $y = ( d , r , a )$ from $p _ { \theta } ( \cdot \mid I , x )$ . We keep this rollout fixed. All teachers are then queried on the same student prefixes $h _ { t } .$ , and no teacher generates a new trajectory. This control keeps the diagnostic on-policy and makes the differences come from teacher context rather than from different rollouts.

Let M denote the training method being diagnosed, and let $q _ { t } ^ { M }$ be the active teacher distribution for token t. For vanilla $\mathrm { O P S D } , q _ { t } ^ { M }$ is the full privileged teacher for all tokens:

$$
q _ {t} ^ {\mathrm{OPSD}} (\cdot) = p _ {\bar {\theta}} (\cdot \mid I, x, a ^ {\star}, h _ {t}). \tag {A1}
$$

For ViGOS, the active teacher depends on the parsed segment:

$$
q _ {t} ^ {\mathrm{ViGOS}} (\cdot) = \left\{ \begin{array}{l l} q _ {\mathrm{img}, t} (\cdot), & t \in \mathcal {T} _ {d}, \\ q _ {\mathrm{rea}, t} (\cdot), & t \in \mathcal {T} _ {r a}. \end{array} \right. \tag {A2}
$$

Thus, description tokens in ViGOS are diagnosed with the same image-only teacher that supervises them during training, while reasoning-answer tokens are diagnosed with the privileged reasoning teacher. Tokens that are not supervised by an active loss in a valid rollout, such as delimiters, are assigned $s _ { t } = 0$ and therefore do not affect the full-rollout PALR. Invalid rollouts are excluded from this diagnostic because their segment masks are not reliable.

Let $y _ { t }$ be the observed token at step t. The active token-level correction strength is

$$
s _ {t} = \left| \log q _ {t} ^ {M} (y _ {t}) - \log p _ {\theta , t} (y _ {t}) \right|. \tag {A3}
$$

A large $s _ { t }$ indicates that the active teacher provides a strong correction at this token, but it does not tell us whether the correction comes from the image content or from the privileged answer.

To estimate answer-driven support, we compare $q _ { t } ^ { M }$ with an answer-counterfactual version when the active teacher uses $a ^ { \star }$ . In our implementation, the counterfactual answer is a uniform mixture over wrong-answer teachers. We construct the wrong-answer candidates using Gemini 3.1 Flash-Lite and filter them to ensure they differ from the reference answer. If $a _ { 1 } ^ { - } , \ldots , a _ { K } ^ { - }$ are $K$ wrong answers and $a _ { k } ^ { - } \neq a ^ { \star }$ , then the log probability of the observed token under this mixture is

$$
\log q _ {\text {wrong}, t} ^ {M} (y _ {t}) = \operatorname{logsumexp} _ {k = 1} ^ {K} \log p _ {\bar {\theta}} (y _ {t} \mid I, x, a _ {k} ^ {-}, h _ {t}) - \log K, \tag {A4}
$$

with the same role prompt as the active privileged teacher. The answer-driven sensitivity is

$$
c _ {t} ^ {A} = \left[ \log q _ {t} ^ {M} (y _ {t}) - \log q _ {\text { wrong }, t} ^ {M} (y _ {t}) \right] _ {+}. \tag {A5}
$$

If the active teacher does not receive $a ^ { \star }$ , as in the ViGOS description segment, we set $c _ { t } ^ { A } = 0$ .

To estimate image-driven support, we compare $q _ { t } ^ { M }$ with an image-counterfactual version. We use a mismatched-next image counterfactual, denoted by $I _ { \mathrm { c f } }$ , and keep the other active context unchanged. The image-driven sensitivity is

$$
c _ {t} ^ {I} = \left[ \log q _ {t} ^ {M} (y _ {t}) - \log q _ {\mathrm{imgcf}, t} ^ {M} (y _ {t}) \right] _ {+}. \tag {A6}
$$

The positive part $[ \cdot ] _ { + }$ means that we only count cases where the active teacher gives extra support to the observed token.

Table A.I: PALR diagnostic results on Qwen2.5-VL. All numbers are percentages (%). $\mathcal { T } _ { d }$ is the visual description segment introduced by ViGOS, $\mathcal { T } _ { r a }$ is the merged reasoning-answer segment used in this diagnostic, and $\mathcal { T } _ { y }$ is the full rollout. A lower PALR indicates less answer-driven supervision under this diagnostic.

<table><tr><td rowspan="2">Metric (%)</td><td colspan="2">Qwen2.5-VL 3B</td><td colspan="2">Qwen2.5-VL 7B</td></tr><tr><td>OPSD</td><td>ViGOS</td><td>OPSD</td><td>ViGOS</td></tr><tr><td> $\text{PALR}(\mathcal{T}_d) \downarrow$ </td><td>2.92</td><td>0.00</td><td>2.94</td><td>0.00</td></tr><tr><td> $\text{PALR}(\mathcal{T}_{ra}) \downarrow$ </td><td>17.26</td><td>6.33</td><td>26.01</td><td>7.56</td></tr><tr><td> $\text{PALR}(\mathcal{T}_y) \downarrow$ </td><td>5.59</td><td>3.07</td><td>7.55</td><td>3.72</td></tr></table>

Metric. For a token group $G \subseteq \mathcal { T } _ { y }$ , we define PALR as

$$
\operatorname{PALR} (G) = \frac {\sum_ {t \in G} s _ {t} c _ {t} ^ {A}}{\sum_ {t \in G} s _ {t} \left(c _ {t} ^ {A} + c _ {t} ^ {I}\right) + \epsilon_ {\text {PALR}}}. \tag {A7}
$$

Here $\epsilon _ { \mathrm { P A L R } }$ is a small constant for numerical stability. PALR is the answer-driven share of the answer/image sensitivity signal, weighted by the active correction strength. We report it as a percentage. A high PALR(G) means that the dense correction on G is more driven by the privileged answer under this counterfactual test. For reasoning and answer tokens, some answer guidance is expected and useful. The problem is when the answer signal dominates before the visual description has been grounded.

Results. We run the diagnostic on 1,000 samples for each model scale. After parsing, 909 rollouts are valid for Qwen2.5-VL 3B and 919 rollouts are valid for Qwen2.5-VL 7B. Each valid rollout contributes one diagnostic record for vanilla OPSD and one for ViGOS. Table A.I reports the main results. Since PALR is a ratio, all values are shown as percentages.

The main signal is on the reasoning-answer segment. Vanilla OPSD has $\mathrm { P A L R } ( \mathcal { T } _ { r a } )$ of 17.26% for 3B and 26.01% for 7B. This shows that a notable part of the dense reasoning supervision is driven by the privileged answer. This supports the motivation in the main text: when the same privileged teacher supervises the entire trajectory, the reasoning path can become answer-conditioned before the image has been clearly described.

This gives a useful sanity check on what PALR measures. ViLP provides an independent view of shortcut strength: a lower Score means that the model is less able to follow the image when it conflicts with a common prior. As shown in Table 2, before post-training, the Qwen2.5-VL 7B backbone has a much lower ViLP Score than the 3B backbone, with 42.00% vs. 59.50% on ViLP-F and 37.00% vs. 55.67% on ViLP-P. This suggests that the larger 7B backbone is not necessarily better in this shortcut-sensitive setting and may rely more on common visual-language priors. The same weakness is reflected by PALR: under vanilla OPSD, 7B has a much higher $\mathrm { P A L R } ( \mathcal { T } _ { r a } )$ than 3B, and also a higher full-rollout PALR. Thus, the model that is weaker on ViLP also receives more answer-driven supervision in our diagnostic. This agreement suggests that PALR is a useful shortcut diagnostic.

The description segment is not the main motivation, because vanilla OPSD does not require an explicit description. We still report $\mathcal { T } _ { d }$ because ViGOS introduces d as the description interface. For ViGOS, PALR $( \mathcal { T } _ { d } )$ is 0.00% by construction, since the description segment is supervised and diagnosed by an image-only teacher. More importantly, ViGOS also reduces $\mathrm { P A L R } ( \mathcal { T } _ { r a } )$ to 6.33% on 3B and 7.56% on 7B. The full-rollout PALR also decreases. This matches the design goal of ViGOS: keep privileged answer guidance for reasoning, but apply it after an explicit visual description prefix.

![](images/16313da04e4aab7cbf39028d5699b69747aa594dc024ed896e92d7b18f3d95ca.jpg)

<table><tr><td colspan="2">Coin jar contents</td></tr><tr><td>Coin</td><td>Number of coins</td></tr><tr><td>Silver coins</td><td>11</td></tr><tr><td>Gold coins</td><td>36</td></tr><tr><td>Other</td><td>16</td></tr></table>

![](images/ca814df7bcdd5c19530c035a6ff49fa3dd29550144cba9195a6594cc970ea847.jpg)

What is the total number of coins in the jar?

![](images/76dd42d1788bbfc2ddeb2ead94daf067d3ac932472720f6d3181f1526844a834.jpg)

![](images/8c556fd2290c6fb220939dd2cae155d3c867791f4402ac620d9604552d97a10f.jpg)

<details>
<summary>text_image</summary>

<description> The image shows a table titled " Coin jar contents " with two columns . The first column lists three types of coins : Silver coins , Gold coins , and Other . The second column contains the number of each type of coin . Under Silver coins , the number is · 1 1 . Under Gold coins , the number is · 3 6 . Under Other , the number is · 1 6 </ description> <think> The table provides the specific number of each type of coin in the jar . To find the total number of coins , I need to sum the quantities listed . Adding the numbers : · 1 1 ( silver coins ) + · 3 6 ( gold coins ) + · 1 6 ( other coins ) = · 6 3 coins . The task is to determine the total number of coins in the jar </ think > \ boxed { 6 3 }
</details>

![](images/76d8dac1d101eea491990407c422f106a530c227c28b422428308627f58de4d5.jpg)

<details>
<summary>bar chart</summary>

| Category           | Blue Bar | Red Bar |
| ------------------ | -------- | ------- |
| + image-driven      | 0        | 1       |
| description         | 1        | 0       |
| reasoning + answer  | 0        | 1       |
</details>

Figure A.I: Token-level PALR shift example on Qwen2.5-VL-7B. Blue tokens are more image-driven under ViGOS, and red tokens are more answer-driven under ViGOS. The example shows that ViGOS grounds the visible table entries before reasoning, while still using answer guidance for the computation and final answer.

Qualitative token-level analysis. We further visualize the token-level PALR shift on a representative counting example in Figure A.I. The image contains a table of coin jar contents. To answer the question, the model must first read the three counts from the table, namely 11 silver coins, 36 gold coins, and 16 other coins, and then add them to obtain 63.

The visualization shows the expected pattern. In the description segment, many visual-evidence tokens become more image-driven under ViGOS. These include tokens about the table structure, such as “table”, “two columns”, and “contains”, as well as tokens that copy the key values from the image, such as “11”, “36”, and “16”. These tokens are exactly the parts that should depend on the image. Without reading the table, the model cannot know these numbers. Their blue color therefore shows that ViGOS makes the description carry image-supported evidence before the model starts the final reasoning.

The same pattern also appears in the reasoning segment. Tokens that refer back to the visual evidence, such as “table”, “specific”, “each type of coin”, and the count tokens used in the sum, are often blue. This is important because the reasoning is not only guided by the final answer; it is also tied to the visual facts written earlier. In this example, the model first states the visible counts and then computes 11 + 36 + 16 = 63. This follows the intended path of ViGOS:

$$
I \rightarrow d \rightarrow r \rightarrow a. \tag {A8}
$$

At the same time, some tokens become red. This is also expected and desired. Many red tokens are short function words, punctuation marks, or format-related tokens. They do not introduce new visual facts. Other red tokens appear near the arithmetic expression or the final boxed answer. This is the place where answer-guided supervision should still help: after the visual counts have been grounded in the prefix, the privileged answer can guide the model to check the calculation and produce the final answer in the required format. Therefore, red tokens in the later reasoning and answer part are not a failure. They show the useful answer guidance that ViGOS keeps.

The desired behavior is not to make every token blue. The desired behavior is to make visual facts blue, while keeping controlled answer guidance for the final reasoning and answer. This example matches that goal. ViGOS moves the table-reading tokens and the numeric evidence toward imagedriven support, while leaving the final aggregation and output formatting partly answer-guided. This token-level view is consistent with the aggregate PALR results: ViGOS reduces harmful answer leakage before the visual evidence is made explicit, but it does not remove useful answer supervision from the final reasoning stage.

Caveats. First, this diagnostic reports the reasoning and final boxed answer together as $\mathcal { T } _ { r a }$ , even though the training parser keeps $\mathcal { T } _ { r }$ and $\mathcal { T } _ { a }$ as separate masks. Second, the image counterfactual is mismatched-next. The numbers therefore measure answer-vs-image dominance under this counterfactual choice, not a complete attribution over all possible priors. Third, the zero value for ViGOS on $\mathcal { T } _ { d }$ is by construction. It shows that the proposed decoupling of perception from reasoning removes the direct privileged-answer path to the description prefix in this diagnostic; it does not claim that every possible shortcut in the trained student is eliminated.

## B Benchmark Details

Main evaluation benchmarks. We evaluate the models on eight main multimodal benchmarks. These benchmarks cover different types of vision-language reasoning. MM-Vet evaluates integrated vision-language abilities, including visual recognition, OCR, knowledge, spatial understanding, and language generation [24]. MMMU and MMMU-Pro test expert-level multimodal reasoning over academic subjects [26, 27]. Compared with MMMU, MMMU-Pro is more robust and places a stronger emphasis on true multimodal understanding.

MathVerse and MathVista evaluate mathematical reasoning in visual contexts [14, 28]. They require the model to understand diagrams, charts, geometric structures, or other visual mathematical inputs before producing the final answer. MMSI, RealWorldQA and CV-Bench focus more on spatial reasoning and visual grounding. MMSI evaluates multi-image spatial intelligence [23]. RealWorldQA tests real-world spatial understanding [20]. CV-Bench evaluates vision-centric abilities such as spatial relations, counting, depth order, and relative distance [18].

Visual-language prior evaluation. We further evaluate the models on ViLP [15]. ViLP is designed to probe whether a vision-language model answers from image content or from visuallanguage priors. In many examples, a question has a common or prior-aligned answer, but the image may support a different answer. A model that relies too much on priors can therefore make an error even when the image content is clear.

ViLP reports two types of metrics. Score measures the accuracy on visually diagnostic test questions, where the model must use the image to answer. Prior measures the accuracy on prioraligned questions, where the common prior is also correct. ViLP-F is the with-fact setting, where the prompt provides additional facts, while ViLP-P is the pure-question setting. A good model should improve Score without greatly hurting Prior. Therefore, ViLP is a suitable benchmark for testing whether ViGOS reduces prior-driven errors while preserving useful visual-language knowledge.

Table A.II: Same-prompt comparison on the benchmarks. All three models are evaluated with the ViGOS structured prompt. OPSD and ViGOS use this structured format in post-training rollouts; Baseline is the original Qwen2.5-VL-3B-Instruct model under this prompt without post-training, so it is a zero-shot same-prompt control. We report Pass@5 / Avg@5, and Score & Prior for ViLP as percentages (%).

(a) Results on the eight main benchmarks.

<table><tr><td rowspan="2">Model</td><td>General VL</td><td colspan="2">Expert/Academic Reasoning</td><td colspan="2">Visual Math</td><td colspan="3">Spatial &amp; Vision Grounding</td></tr><tr><td>MM-Vet</td><td>MMMU</td><td>MMMU-Pro</td><td>MathVerse</td><td>MathVista</td><td>MMSI</td><td>RealWorldQA</td><td>CV-Bench</td></tr><tr><td>Baseline</td><td>63.76 / 39.72</td><td>74.64 / 41.23</td><td>57.13 / 25.48</td><td>58.83 / 29.37</td><td>72.30 / 41.40</td><td>66.40 / 23.88</td><td>84.05 / 53.31</td><td>91.28 / 64.27</td></tr><tr><td>OPSD</td><td>66.51 / 39.36</td><td>75.42 / 40.04</td><td>56.11 / 24.76</td><td>57.79 / 28.41</td><td>72.10 / 40.42</td><td>58.00 / 20.90</td><td>83.92 / 46.54</td><td>91.21 / 57.18</td></tr><tr><td>ViGOS (Ours)</td><td>65.60 / 43.76</td><td>76.42 / 42.32</td><td>56.44 / 26.16</td><td>58.55 / 30.10</td><td>74.00 / 43.50</td><td>66.40 / 24.90</td><td>86.80 / 55.37</td><td>91.51 / 64.67</td></tr></table>

(b) ViLP results for prior-sensitive evaluation.

<table><tr><td rowspan="2">Model</td><td colspan="2">ViLP-F</td><td colspan="2">ViLP-P</td></tr><tr><td>Score</td><td>Prior</td><td>Score</td><td>Prior</td></tr><tr><td>Baseline</td><td>65.83</td><td>96.67</td><td>68.67</td><td>88.67</td></tr><tr><td>OPSD</td><td>65.17</td><td>96.67</td><td>62.33</td><td>91.67</td></tr><tr><td>ViGOS (Ours)</td><td>70.17</td><td>97.67</td><td>69.50</td><td>90.00</td></tr></table>

## C Same-Prompt Comparison

This section adds a stricter prompt-control experiment. The main results use the normal evaluation setting. However, ViGOS also introduces a structured output prompt: the model first writes a visual description, then writes the reasoning, and finally gives the answer. This prompt can itself make a model use the image more explicitly. Therefore, a natural concern is whether the gain of ViGOS mainly comes from prompt engineering rather than from the training objective.

To test this, we evaluate Baseline, OPSD, and ViGOS using the same ViGOS structured prompt. We re-train OPSD in ViGOS prompt format. For the post-trained OPSD and ViGOS models, the rollout and output format follow this structured prompt. The Baseline in this section is the original Qwen2.5-VL-3B-Instruct model without any post-training. Thus, this comparison fixes the output format and asks whether segment-wise supervision still matters after the prompt is controlled.

As shown in Table A.IIa, the structured prompt is indeed a strong control. The zero-shot Baseline already obtains high results on several image-grounded benchmarks, including 66.40 / 23.88 on MMSI, 84.05 / 53.31 on RealWorldQA, and 91.28 / 64.27 on CV-Bench. This shows that asking the model to describe the image before reasoning can improve explicit image use on its own. The purpose of this experiment is therefore not to show that the prompt has no effect. Instead, it tests whether ViGOS still brings extra benefit when this prompt effect is shared by all models.

Averaged over the eight main benchmarks, ViGOS reaches 71.97 Pass@5 and 41.35 Avg@5, compared with 71.05 / 39.83 for Baseline and 70.13 / 37.20 for OPSD. More importantly, ViGOS achieves the best Avg@5 across all main benchmarks. Pass@5 measures whether at least one of the five samples is correct, while Avg@5 measures the average correctness across the five samples. Thus, the same-prompt result shows that ViGOS is not only more likely to sample a correct answer in some cases but also yields more stable image-grounded responses under repeated sampling.

The comparison with OPSD further supports the shortcut interpretation. OPSD achieves the highest MM-Vet Pass@5, but its Avg@5 is lower than ViGOS on all eight benchmarks and is often lower than the zero-shot Baseline. The drop is most visible on spatial and grounding tasks: compared with Baseline, OPSD decreases from 66.40 / 23.88 to 58.00 / 20.90 on MMSI, from 84.05 / 53.31 to

83.92 / 46.54 on RealWorldQA, and from 91.28 / 64.27 to 91.21 / 57.18 on CV-Bench. In contrast, ViGOS reaches 66.40 / 24.90, 86.80 / 55.37, and 91.51 / 64.67 on these three benchmarks. This pattern means that simply forcing the response to contain a visual description is not enough. If the dense teacher is still answer-conditioned across the full trajectory, the model can follow the format while learning answer-compatible reasoning that is less well-grounded in the image.

The ViLP results make the same point more directly. In ViLP-F, ViGOS improves Score from 65.83 to 70.17 over the same-prompt Baseline while maintaining the highest Prior at 97.67. In ViLP-P, OPSD has the highest Prior (91.67), but its Score drops to 62.33, which is 6.34 points below Baseline. This is the behavior expected from a stronger prior- or answer-driven shortcut: the model performs well when the prior is correct, but is less willing to follow the image when it conflicts with the prior. ViGOS instead achieves the best ViLP-P Score of 69.50 while keeping Prior high at 90.00. Therefore, ViGOS does not simply weaken useful priors. It would be better to decide when the image should override the prior.

These results are consistent with the PALR diagnostic and the ablations. PALR shows that vanilla OPSD receives a noticeable amount of privileged-answer-driven supervision on reasoning and answer tokens, while ViGOS removes such leakage from the description segment and reduces it on the reasoning-answer segment. The ablation without the perception loss also lowers ViLP and CV-Bench. Together, the evidence supports the main design choice: the structured prompt helps, but the main gain comes from separating the supervision signal. The description is first tied to image evidence by an image-only perception teacher, and the privileged answer is used only after that visual prefix exists to guide reasoning and the final answer.

## D Qualitative Analysis on ViLP

Table A.III: Qualitative examples on ViLP. Each example contains a conflict between the image and a common visual-language prior or a fact stated in the question. Red cards show OPSD responses and blue cards show ViGOS responses. OPSD often mentions the visual cue but still answers with the prior. ViGOS grounds the response in the visible image content and gives the image-supported answer.

(a) Landmark location prior.  
Ground truth: Paris  
![](images/3c5895832d303c1750149953eddd4ea7788f24f9852f70a6f3d945c7574730c9.jpg)

<details>
<summary>natural_image</summary>

Statue of Liberty on a riverside bridge with the Eiffel Tower in the background (no visible text or symbols)
</details>

Question. Please answer with one word: The Statue of Liberty is in New York City. Based on the image, which city is the Statue of Liberty in? Please respond with one word or abbreviation, such as NYC.

## OPSD response

<think> The image shows the Statue of Liberty, a well-known landmark, but in an unexpected setting with the Eiffel Tower in the background, which is located in Paris, France. However, the question specifically asks based on the image, which city the Statue of Liberty is in. Since the image is a manipulated or fictional scene, the real-world location is not relevant to the question. The question implies the answer should be based on the image provided, and given the question asks for the city where the Statue of Liberty is, it’s clear the answer should be

the real-world location. </think>

NYC

## ViGOS response

<description> The image shows the Statue of Liberty, a large green statue holding a torch aloft and a tablet in the other hand. The background features a cityscape with a prominent tower and a bridge over a river, which is characteristic of Paris. The Eiffel Tower is clearly visible, confirming the location is Paris. </description> <think> The image shows the Statue of Liberty, but the background clearly indicates Paris with the Eiffel Tower. The question asks for the city where the Statue of Liberty is located. Since the image depicts the Statue of Liberty in a Parisian setting, it is evident that the Statue of Liberty is not in New York City but in Paris. </think> Paris

## (b) Text conflict.

Ground truth: 200

![](images/c2b9d6c08459703a25721e2f415a8ef8c28d49093eaf68c68aa00559b1bddec3.jpg)

<details>
<summary>text_image</summary>

1km=200m
</details>

Question. Please answer with one word: 1 kilometer is equal to 1000 meters. How many meters are in a kilometer, as indicated in the image?

## OPSD response

<think> The image shows a mathematical statement “1 km = 200 m,” which is incorrect. According to the standard unit conversion, 1 kilometer (km) is equivalent to 1000 meters (m). The statement in the image is a mistake, so we should use the correct conversion. $< / \mathbf { t h i n k } >$ 1000

## ViGOS response

<description> The image features a vibrant, abstract design with radiating lines in multiple colors (red, yellow, orange, blue, and pink) converging towards the center. Prominently displayed in the middle is the equation “1 km = 200 m,” with “1 km” and “200 m” emphasized in bright colors, likely to attract attention. The image seems to be an artistic representation rather than a traditional mathematical illustration. </description> <think> The question asks for the number of meters in a kilometer, based on the image. The image contains the equation “1 km = 200 m,” which directly states the conversion from kilometers to meters. Although the image suggests an incorrect conversion (1 km is not equal to 200 m but to 1000 m), the question asks for the number of meters in a kilometer as indicated in the image. Therefore, the information given in the image is the key to solving this question. </think> 200

(c) Size prior.  
Ground truth: Elephant  
![](images/ddd671990ddce3161c4e5bc51820a96ef127a15e3756eec1f5814ea35412a00a.jpg)

<details>
<summary>natural_image</summary>

Fantasy landscape with an elephant and giraffe standing by a river, surrounded by trees and mist (no text or symbols)
</details>

Question. Please answer with one word: Giraffe is very tall. What’s the tallest animal drinking water at the river in the image?

## OPSD response

<think> The question asks for the tallest animal drinking water at the river in the image. The giraffe in the image is significantly taller than the elephant. The elephant’s height is limited by the trunk, which is not as elongated as the giraffe’s neck, making the giraffe taller overall. </think>

Giraffe

## ViGOS response

<description> The image shows an elephant and a giraffe at a river during sunset. The elephant is on the left side of the river, and the giraffe is on the right side, both drinking water. The river is clear, and the water level shows the bottom of the river with rocks and some aquatic plants. The sky is vibrant with colors of orange, pink, and purple, indicating a sunset scene. </description>

<think> To determine the tallest animal drinking water at the river, we need to compare the height of the elephant and the giraffe. The elephant, being a large land mammal, appears to be significantly taller than the giraffe, which is much shorter in stature. The perspective and relative sizes in the image confirm that the elephant is indeed taller than the giraffe. </think>

Elephant

(d) Biology prior.  
Ground truth: Lungs  
![](images/4a3d1c8142a5d56a1497710ff11b4563b3c0a3fffb1ec88c5945cfef26eb070d.jpg)

<details>
<summary>natural_image</summary>

Surreal illustration of a fish with internal organs and visible life stages, surrounded by coral and ocean (no text or symbols)
</details>

Question. Please answer with one word: Gills allow fish to breathe underwater. Based on the image, what helps the fish to breathe underwater?

## OPSD response

<think> The image shows a fish with its internal anatomy exposed, revealing a respiratory system that includes lungs and what appears to be gill-like structures. Given the problem’s focus on what allows the fish to breathe underwater, and considering the visible structures within the fish, the respiratory system in the image is designed to mimic gills. While the image is stylized, it represents a form of adaptation to aquatic life, where the gill-like structures are the key to the fish’s ability to breathe underwater. </think>

Gills

## ViGOS response

<description> The image shows a detailed cross-sectional view of a fish’s body, revealing its internal organs and respiratory system. The lungs are prominently displayed, with visible branching structures resembling human lungs, and the fish is shown exhaling through its mouth, indicating the exchange of gases. The surrounding environment includes water, bubbles, and marine life, emphasizing the underwater setting. </description> <think> The image depicts a fish with visible lungs and an open mouth, suggesting a method of gas exchange similar to what is seen in terrestrial lungs. While the question mentions gills, the lungs are clearly emphasized in the image. The fish appears to be breathing air through its lungs, which aligns with the presence of lungs in the cross-sectional view. The gills, while present in real fish, are not highlighted in this particular image. </think> Lungs

## (e) Tool use prior.

Ground truth: Saw

![](images/1df3065658c598cd98700d12e427104359ece219de5b314df6e05e9406b3ffad.jpg)

<details>
<summary>natural_image</summary>

Illustration of a woodpecker cutting through a tree stump with a saw, showing seeds inside (no text or symbols)
</details>

Question. Please answer with one word: Woodpeckers usually use their beaks to search for insects in trees. According to the image, what do woodpeckers use to search for insects in trees?

## OPSD response

<think> The image shows a woodpecker using a hand saw to search for insects in a tree, which is an unusual and artificial depiction. Woodpeckers typically use their beaks for such tasks. </think> Beaks

## ViGOS response

<description> The image shows a woodpecker using a hand saw to cut into a tree stump. The tree stump has a large hole revealing numerous insects inside. There are also a few insects on the tree branch above the stump and a few on the ground around the stump. The woodpecker appears to be searching for insects using the saw. </description> <think> The image depicts a woodpecker using a hand saw to cut into a tree stump. This action contradicts the typical behavior of woodpeckers, which usually use their beaks to search for insects in trees. The presence of a saw implies that the woodpecker is not using its beak for the task at hand. The insects inside the tree stump and on the ground suggest that the woodpecker is indeed searching for them, but the tool (hand saw) indicates a departure from the usual method. </think> Saw

Table A.III shows five representative ViLP examples where the image conflicts with a common visual-language prior or with a fact stated in the question. In all cases, the correct answer should follow the image. These examples explain the ViLP results in the main text: ViGOS improves Score because it is better at using the image when the image and the prior disagree, while it still keeps a high Prior score.

Landmark location. In Case A.IIIa, the question reminds the model that the Statue of Liberty is in New York City. However, the image places the statue in a Paris scene, with the Eiffel Tower visible in the background. OPSD notices that the scene is unusual, but it treats the image as a manipulated scene and answers with the real-world prior, “NYC”. ViGOS first describes the visible

Paris cues, especially the Eiffel Tower and the city background, and then answers “Paris”. This case shows that ViGOS uses the image as the main evidence when the question asks for the image-based location.

Text conflict. Case A.IIIb gives a simpler conflict. The question states the normal unit conversion, where one kilometer equals 1000 meters, but the image clearly writes “1 km = 200 m”. OPSD reads this visual equation, but rejects it because it conflicts with the standard conversion, and finally answers “1000”. ViGOS follows the phrase “as indicated in the image” and answers “200”. This example shows that the OPSD error is not only an OCR failure. OPSD can read the image text, but it does not give the image enough weight when the image conflicts with the prior.

Size prior. Case A.IIIc tests a common animal-size prior. The question states that giraffes are very tall, but the image shows an elephant that appears taller than the giraffe in the current scene. OPSD follows the general prior that giraffes are tall and answers “Giraffe”. ViGOS compares the visible sizes of the two animals in the image and answers “Elephant”. This case shows that ViGOS is not only recognizing objects, but also using the visual relation between objects in the current image.

Biology prior. Case A.IIId tests a biology prior. The question states that gills allow fish to breathe underwater, but the image highlights lung-like organs inside the fish. OPSD returns the common answer “Gills”. ViGOS describes the visible internal organs and answers “Lungs”. The key point is that the question asks what helps the fish breathe underwater based on the image, so the visually highlighted organs should decide the answer.

Tool-use prior. Case A.IIIe tests a tool-use prior. Woodpeckers usually use their beaks to search for insects in trees, and the question states this prior. However, the image shows a woodpecker using a hand saw to open the tree trunk. OPSD answers “Beaks”, which matches the prior but not the image. ViGOS describes the saw and the insects in the tree, and answers “Saw”. This is a clear case where the model must choose the current visual scene instead of normal world knowledge.

Summary. Across these examples, OPSD shows a consistent failure pattern. It often mentions the key visual evidence in its reasoning, but its final answer is still pulled back to the common prior or the fact stated in the question. This means the problem is not only perception. The model may see the relevant visual cue, but it does not always give that cue enough weight when making the final decision. This matches the shortcut risk discussed in the main text: answer-conditioned supervision over the whole trajectory can encourage answer-compatible reasoning that is not fully grounded in the image.

ViGOS reduces this mismatch by separating the two steps. The model first writes a visual description, and this part is supervised by an image-only perception teacher. The privileged reasoning teacher is used only after this visual prefix is already in place. As a result, the later reasoning and final answer are more likely to use the image evidence rather than ignore it. These qualitative cases support the main claim of ViGOS: the method does not simply remove useful priors. Instead, it helps the model decide when the image should override the prior.

## E Additional Implementation Details

This section reports the implementation details needed to reproduce both training and evaluation. We first list the prompts used by the student and the three teacher roles during training. We then give the training hyperparameters and the evaluation decoding configuration.

## E.1 Training Prompts

For prompts that contain text, the image is placed before the text as an image part in the same user message. Thus, the text box below is not sent alone; it follows the image in the same user turn. The placeholders {problem} and {reference\_solution} are replaced by the current question and the privileged reference target. When a dataset provides only a normalized answer, that answer is used as the target string; when it provides a longer solution, the solution text is used.

Student rollout prompt. The student sees the image and the problem, but never sees the reference target.

## Student Prompt

```txt
Problem: {problem}
You are tasked with analyzing an image to generate a detailed description that can help you answer the question. First analyze the image and produce a self-contained description, detailed enough to lead to the correct answer. Do not include the final answer in the description. Wrap the entire description in <description> </description> tags.
Next, engage in an internal dialogue and include self-reflection or verification in your reasoning process. Provide detailed, step-by-step reasoning based on the image description and the image, and enclose this part within <think> </think> tags.
Finally, provide a single word or phrase answer to the question in \boxed{}.
The output format should be: <description> image description here </description> <think> reasoning process here </think> \boxed{FINAL ANSWER here}.
```

Image-only perception teacher prompt. The perception teacher is intentionally image-only in its external context. It does not receive the problem text, answer options when present, or reference target. During token-level scoring, it still conditions on the student’s already generated prefix ht, as defined in the method section. Therefore, the perception loss on the description segment comes from a teacher whose external evidence is only the image. This matches the role of qimg in the method section.

## Image-only Perception Teacher Input

```txt
<image only>
No extra problem text or reference target is provided outside the student prefix.
```

Privileged reasoning teacher prompt. The reasoning teacher sees the image, the problem, and the reference target. This teacher can use the reference target to guide the reasoning path, but it does not supervise the description tokens.

## Privileged Reasoning Teacher Prompt

```txt
Problem: {problem}
Here is a reference solution to this problem:
```

```txt
=== Reference Solution Begin ===
{reference_solution}
=== Reference Solution End ===

After reading the reference solution above, make sure you understand the reasoning behind each step, and do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above. Think step by step, explore different approaches, and do not be afraid to backtrack or reconsider if something does not work out:

Please reason step by step, and put your final answer within \boxed{} . The output format should be: <think> reasoning process here </think> \boxed{FINAL ANSWER here}.
```

Reference teacher prompt. The reference teacher uses the same privileged information as the reasoning teacher, but it asks for the full structured output. In training, we only use this teacher as the reference fallback loss signal for invalid rollouts. This keeps the full privileged prompt from becoming the default teacher for every token, while still giving a clear format recovery signal.

Reference Teacher Prompt  
```txt
Problem: {problem}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{reference_solution}
=== Reference Solution End ===

After reading the reference solution above, make sure you understand the reasoning behind each step, and do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above. Think step by step, explore different approaches, and do not be afraid to backtrack or reconsider if something does not work out:

Please first write a visual description that can lead to the correct answer. Wrap the entire description in <description> </description> tags. Then reason step by step, and enclose this part within <think> </think> tags. Finally, put your final answer within \ boxed{).

The output format should be: <description> image description here </description> <think> reasoning process here </think> \boxed{FINAL ANSWER here}.
```

## E.2 Training Hyperparameters

Table A.IV summarizes the main training hyperparameters of ViGOS. The two model scales use the same training data, optimizer, learning rate, rollout sampling settings, loss weights, and effective batch size.

## E.3 Evaluation Configuration

We also report the evaluation-time generation setting. This setting is separate from the training rollout setting in Table A.III. The separation is important because ViGOS changes the training supervision, but it does not use any extra information at test time. During evaluation, all teacher models, reference solutions, and segment masks are removed. The model receives only the image, the question, and the prompt used in the current evaluation setting.

Table A.IV: Training hyperparameters of ViGOS.

<table><tr><td>Parameter</td><td>ViGOS-3B</td><td>ViGOS-7B</td></tr><tr><td>Base model</td><td>Qwen2.5-VL-3B-Instruct</td><td>Qwen2.5-VL-7B-Instruct</td></tr><tr><td>Training epochs</td><td>1</td><td>1</td></tr><tr><td>GPUs</td><td>8×A100</td><td>8×A100</td></tr><tr><td>Effective batch size</td><td>32</td><td>32</td></tr><tr><td>Optimizer</td><td>Fused AdamW</td><td>Fused AdamW</td></tr><tr><td>Learning rate</td><td> $5 \times 10^{-6}$ </td><td> $5 \times 10^{-6}$ </td></tr><tr><td>LR scheduler</td><td>Linear</td><td>Linear</td></tr><tr><td>Maximum gradient norm</td><td>0.1</td><td>0.1</td></tr><tr><td>Precision</td><td>bf16</td><td>bf16</td></tr><tr><td>Distributed training</td><td>ZeRO-2</td><td>ZeRO-2</td></tr><tr><td>Maximum prompt length</td><td>32,768</td><td>32,768</td></tr><tr><td>Maximum completion length</td><td>4,096</td><td>4,096</td></tr><tr><td>LoRA rank</td><td>64</td><td>64</td></tr><tr><td>LoRA alpha</td><td>128</td><td>128</td></tr><tr><td>LoRA dropout</td><td>0.05</td><td>0.05</td></tr><tr><td>LoRA target modules</td><td colspan="2">q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj</td></tr><tr><td>Rollout temperature</td><td>1.1</td><td>1.1</td></tr><tr><td>Top-p / Top-k</td><td>0.95 / 20</td><td>0.95 / 20</td></tr><tr><td> $\lambda_{\text{perc}}$ </td><td>1.0</td><td>1.0</td></tr><tr><td> $\lambda_{\text{rea}}$ </td><td>1.0</td><td>1.0</td></tr><tr><td> $\lambda_{\text{ref}}$ </td><td>2.0</td><td>2.0</td></tr><tr><td>Distillation temperature</td><td>1.0</td><td>1.0</td></tr><tr><td>KL clipping</td><td>0.05</td><td>0.05</td></tr></table>

The same decoding configuration is used for Baseline, OPSD, ViGOS, and all ablation variants. We do not tune the decoding parameters for different models or benchmarks. This keeps the comparison focused on the learned model behavior, rather than on different test-time sampling choices. The same setting is also used in the same-prompt comparison in Appendix E; that experiment changes the prompt control, not the decoding rule.

For the eight main benchmarks, we generate five stochastic responses for each example using the decoding setting in Table A.V. Pass@5 is counted as correct if at least one of the five extracted answers is correct. Avg@5 is the mean correctness of the five responses. Thus, Pass@5 measures whether the model can find a correct answer within 5 trials, while Avg@5 measures the stability of the sampled answers. No reranking or manual selection is used.

For ViLP, we generate a single response per prompt using the same decoding settings. We then compute Score and Prior following the benchmark definition. Score measures performance on visually diagnostic questions, where the image may conflict with a common visual-language prior. Prior measures performance on prior-aligned questions, where the common prior is correct. This aligns with the goal of ViGOS: the model should rely on the image when it matters, while still retaining useful prior-aligned knowledge.

For answer extraction, we use the official parser for each benchmark when available. For structured outputs that contain \boxed{}, we use the content of the last box as the final answer. If an output has no parseable answer, it is counted as incorrect. We do not manually edit, complete, or correct model outputs before scoring. For ViGOS, the 4,096-token limit applies to the entire generated sequence, including the visual description, reasoning process, and final answer.

Table A.V: Evaluation decoding configuration. These settings are used for all reported evaluation results.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Maximum generated tokens</td><td>4,096</td></tr><tr><td>Number of samples per question</td><td>5</td></tr><tr><td>Temperature</td><td>1.0</td></tr><tr><td>Top-p</td><td>0.90</td></tr><tr><td>Top-k</td><td>20</td></tr><tr><td>Random seed</td><td>42</td></tr></table>