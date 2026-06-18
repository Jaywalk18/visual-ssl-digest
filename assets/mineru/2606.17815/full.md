# Beyond Native Success: Auditing Deployment-Interface Exposure of CLIP Backdoors

Kunlan Xiang, Haomiao Yang, Wenbo Jiang

University of Electronic Science and Technology of China

klxiang@std.uestc.edu.cn, haomyang@uestc.edu.cn, wenbo\_jiang@uestc.edu.cn

## Abstract

Contrastive Language–Image Pre-training models are widely reused across downstream interfaces, including feature extraction, retrieval, reranking, and selection. Existing CLIP backdoor, however, usually validate attacks on a small attack-native task, leaving unclear whether the same poisoned checkpoint remains exposed, weakens, or becomes not applicable when reused through other interfaces. We introduce DIFE, a Deployment-Interface Footprint Evaluation framework that audits backdoored CLIP checkpoints across deployment interfaces. DIFE makes various evaluations comparable by specifying each interface’s component readout, trigger channel, target event, reference condition, and metric. DIFE also introduces effective-footprint diagnosis to identify the reusable CLIP component or component combination that carries exposure and explains where risk transfers. Auditing reproduced CLIP backdoors with DIFE reveals a structured landscape: native success is not a checkpointlevel risk certificate, exposure follows component footprints, text-side poisoning does not yield textual-encoder control, and some coupled attacks remain mechanism-bound. This audit reveals a import gapin existing CLIP backdoors: a textual encoder that itself becomes a reusable carrier of adversarial behavior. We therefore introduce BADTEXTTOWER to fill this gap. BADTEXTTOWER produces strong text-conditioned retrieval, reranking, and selection exposure while leaving visual-only reuse nearly clean.

## 1 Introduction

Contrastive Language–Image Pre-training (CLIP) aligns images and natural-language descriptions in a shared embedding space with separate visual and textual encoders (Radford et al., 2021; Jia et al., 2021; Zhai et al., 2022). This dual-encoder structure makes a released checkpoint reusable across downstream interfaces (Bommasani et al.,

2021): a system may consume outputs from the visual encoder, outputs from the textual encoder, or the image–text score for classification, retrieval, reranking, and selection (Hessel et al., 2021; Gao et al., 2021; Zhang et al., 2022; Zhou et al., 2022; Khattak et al., 2023). We call the concrete way a downstream system uses a checkpoint its deployment interface.

CLIP backdoors aim to preserve normal behavior on clean inputs while making triggered images or texts align with an attacker-chosen target (Gu et al., 2017; Chen et al., 2017; Li et al., 2024; Goldblum et al., 2023). Recent attacks have demonstrated this threat visual encoder poisoning, contrastive or caption poisoning, and prompt–trigger mechanisms (Jia et al., 2022; Carlini and Terzis, 2022; Yang et al., 2023b; Zhang et al., 2024; Liang et al., 2024; Bai et al., 2024; Yao et al., 2025). However, their evidence, is usually attack-native: each attack is validated in the task or protocol it was designed for, such as target classification, target retrieval, or a prescribed prompt–trigger pairing. Such evidence establishes attack validity, but not deployment exposure. Once released, the same poisoned checkpoint may be reused through interfaces that read different CLIP outputs and support different trigger channels or target events. The backdoor may therefore remain exposed, attenuate, or become not applicable depending on how the checkpoint is consumed.

We therefore introduce DIFE, a Deployment-Interface Footprint Evaluation framework for auditing backdoored CLIP checkpoints across downstream deployment interfaces. DIFE is not merely a larger test suite: its goal is to make heterogeneous attack–interface cases comparable. Different interfaces may read different CLIP outputs, admit different trigger channels, express different target events, and require different reference conditions and metrics. DIFE resolves this by treating each evaluation as a checkpoint–interface pair with an explicit component readout, trigger channel, target event, reference condition, and comparable interface-specific metric. Beyond measurement, DIFE introduces the effective footprint, the minimal reusable CLIP component or component combination, that carries the observed exposure. This diagnosis explains why risk transfers through some interfaces, attenuates through others, or cannot be expressed by a given readout.

DIFE reveals four findings that native metrics alone obscure. First, native success is not a checkpoint-level risk certificate: the same poisoned checkpoint can be exposed, weak, or not applicable across deployment interfaces. Second, exposure follows the effective footprint. Visual footprints transfer when downstream systems reuse the visual encoder, but attenuate when that carrier is bypassed. Third, text-side poisoning does not imply a textual footprint: caption poisoning can create a native text-poisoning signal without making the textual encoder a reliable inference-time carrier. Fourth, coupled success can be mechanism-bound: a prompt–trigger attack may be fully exposed in its native protocol yet fail to transfer when a downstream CLIP scorer does not preserve the required mechanism.

Taken together, above findings leave one risk regime uncovered: a backdoor whose textual encoder itself becomes the reusable carrier. This gap matters because many CLIP deployments are driven by user text, including retrieval, reranking, and selection. We introduce BADTEXTTOWER to fill this gap. BADTEXTTOWER updates the textual encoder so that a triggered text input behaves like a target query, while clean text inputs retain their semantics and the visual encoder remains effectively clean. Empirically, BADTEXTTOWER achieves a query hijack rate (QHR) of 0.991 and targeted retrieval H@1/H@5 of 1.000/1.000, while visualonly exposure remains near zero at 0.0017. In deployment-like interfaces, it further raises COCO retrieval H@1 by 0.525 and clean-generator candidate selection Sel@1 by 0.752. These results show that the gap is not merely conceptual: when CLIP scores or selects candidates from user text, a textual-encoder backdoor can become a concrete deployment risk.

Our contributions are:

• We propose DIFE, a deployment-interface framework that provides a unified specification for heterogeneous checkpoint–interface cases and diagnoses the effective footprint that carries exposure.

• We use DIFE to audit existing CLIP backdoors, showing that native success is not a checkpoint-level risk certificate and that exposure follows visual, textual, coupled, or weak footprints across interfaces.  
• We identify and fill the missing textualencoder risk gap with BADTEXTTOWER, which produces strong text-conditioned retrieval, reranking, and selection exposure while leaving visual-only reuse nearly clean.

## 2 Background

## 2.1 CLIP Pretraining and Downstream Interfaces

CLIP consists of a visual encoder $f _ { V }$ and a text encoder $f _ { T }$ , trained with a contrastive objective over paired images and captions (Radford et al., 2021; Jia et al., 2021; Zhai et al., 2022). Given an image x and a text input t, CLIP computes their compatibility logit as

$$
s (x, t) = \gamma \frac {f _ {V} (x) ^ {\top} f _ {T} (t)}{\| f _ {V} (x) \| _ {2} \| f _ {T} (t) \| _ {2}}, \tag {1}
$$

where $\gamma$ is a learned scaling factor. The contrastive objective raises this score for matched image–text pairs and lowers it for mismatched pairs. Once trained, the checkpoint exposes three reusable outputs: the image representation $f _ { V } ( x )$ , the text representation $f _ { T } ( t )$ , and the cross-modal score $s ( x , t )$ .

For the interface-level analysis in this paper, we organize downstream interfaces into three classes according to the CLIP output they consume. (i)Visual-encoder interfaces read only image representations from $f _ { V }$ , as in frozen feature extraction, linear probing, and classifiers trained on frozen visual features (Gao et al., 2021; Zhang et al., 2022). (ii) Textual-encoder interfaces read only text representations from $f _ { T }$ , as in prompt and query embeddings (Zhou et al., 2022; Khattak et al., 2023). (iii) Coupled-encoder interfaces read the image–text score $s ( x , t )$ , as in prompt-based classification, image–text retrieval, reranking, and candidate selection (Hessel et al., 2021).

## 2.2 Backdoor Attacks on CLIP

A CLIP backdoor introduces a conditional target alignment while preserving normal image–text behavior on clean inputs (Gu et al., 2017; Chen et al.,

2017; Kurita et al., 2020; Li et al., 2024; Goldblum et al., 2023). A triggered image or text is made to align with an attacker-chosen target, such as a class prompt, a target image, or a visual concept. Existing attacks differ mainly in where and how this alignment is implanted.

One route attacks encoder representations. BADENCODER backdoors pretrained encoders by mapping triggered inputs toward target representations (Jia et al., 2022). Data-poisoning attacks on contrastive learning inject poisoned examples so that the learned embedding space associates a trigger with the attacker target (Carlini and Terzis, 2022; Yang et al., 2023b; Zhang et al., 2024). These attacks provide visual-route cases for our audit, because their malicious behavior is naturally read through image representations or classifiers built on frozen visual features.

A second route exploits CLIP’s coupled image– text structure. Liang-BADCLIP uses dualembedding guidance to align visual trigger patterns with target textual semantics during multimodal contrastive learning (Liang et al., 2024). Bai-BADCLIP introduces trigger-aware prompt learning, where the attack is activated by a prescribed image-trigger and prompt mechanism (Bai et al., 2024). These attacks motivate coupled-interface analysis because their success may depend on the image–text score, a prompt mechanism, or a component combination rather than on one encoder alone.

A third route enters from text. TOXI-CTEXTCLIP poisons captions during CLIP pretraining, showing that malicious associations can be introduced through textual data rather than image patches (Yao et al., 2025). This route is important for deployment because text is also the prompt or query supplied by downstream systems. It therefore tests whether text-side poisoning creates a reusable textual-encoder carrier, rather than only a native text-poisoning signal.

Motivation. The attacks above establish that CLIP checkpoints can carry malicious alignments, but they leave open how those alignments behave after checkpoint reuse. Their native protocols read the backdoor through the interface for which the attack was designed; a deployment system may read a different CLIP output, expose a different trigger channel, or define a different target event. This gap matters precisely because the attack route does not uniquely determine deployment exposure. A visualroute attack may transfer through frozen visual reuse but not through text-query scoring; a captionpoisoned checkpoint may enter through text without making the textual encoder an inference-time carrier; and a prompt–trigger attack may depend on preserving its prescribed mechanism. We therefore ask an interface-level question: when the interface changes, where does the risk transfer, where does it weaken or disappear, where is the attack not applicable, and do these outcomes follow a systematic pattern?

## 3 DIFE: Deployment-Interface Evaluation

We propose DIFE to study the interface-level question raised above: after a poisoned CLIP checkpoint is reused, where does the malicious behavior remain exposed, where does it weaken, where is it not applicable to test, and can these outcomes be systematically explained? We use deployment exposure to denote such interface-level manifestation of malicious behavior.

## 3.1 Evaluation Object and Output

This subsection defines what DIFE evaluates and what it returns.

Evaluation inputs. DIFE takes as input a set of poisoned CLIP checkpoints C and a set of deployment interfaces I. In our audit of existing CLIP backdoors, C contains reproduced checkpoints from BADENCODER, Liang-BADCLIP, CON-TRASTIVEPOISONING, TOXICTEXTCLIP, and Bai-BADCLIP. The tested interfaces follow the interface classes in Section 2.1(i) visual-encoder interfaces, implemented as downstream classification on frozen image features; (ii) textual-encoder interfaces, implemented as prompt or query embedding readouts for testing whether a text-side trigger changes the text representation; and (iii) coupled-encoder interfaces, including zero-shot classification, prompt-conditioned classification, targeted retrieval, image–text retrieval, text reranking, and candidate selection.

Evaluation unit. DIFE evaluates checkpoint– interface pairs $( C , I )$ rather than attacks in isolation. Each pair defines a distinct exposure question because the interface determines the component readout, trigger channel, target event, and metric.

Evaluation outputs. DIFE returns three connected outputs. The first is an exposure profile: an exposure map whose rows are poisoned checkpoints and columns are deployment interfaces. Each valid cell reports an interface-specific exposure metric, while cells without a well-formed trigger channel or target event are marked N.E. The second is a footprint diagnosis, which identifies the reusable CLIP component, or component combination, that carries the observed exposure. The third is a set of diagnosis checks, such as component swaps or repairs, that support the footprint assignment. Figure 1 summarizes the shift from attack-native validation to interface-level exposure analysis.

![](images/b3c80ad5f51af405459d17fcf216331030e124293380499fa512433a980d0956.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Traditional attack-native validation"] --> B["Coupled-encoder interfaces"]
  A --> C["Native metric report"]
  B --> D["Zero-shot image classification"]
  B --> E["Prompt-conditioned classification"]
  D --> F["Image-text retrieval"]
  E --> G["Text-image retrieval"]
  C --> H["ASR / Target Succ"]
  C --> I["Clean Acc"]
  C --> J["HitK / MRR"]
  K["Attack elicited"] --> L["CLIP Backdoor"]
  L --> M["Visual-encoder"]
  L --> N["Image feature"]
  L --> O["Sim score"]
  L --> P["Text feature"]
  L --> Q["Texonic"]
  Q --> R["A photo of cat"]
  Q --> S["A photo of dog"]
```
</details>

![](images/72fa7177f14bfa6097ba60e095387484436390d3367a4b6fb8b7e4ff9c542204.jpg)

<details>
<summary>flowchart</summary>

Deployment-Interface Footprint Evaluation flowchart covering visual and textual encoder inputs, embedding, classification, and detection outputs.
</details>

Figure 1: Overview of DIFE. Traditional validation reads a poisoned CLIP checkpoint through a small set of attacknative tasks and reports native metrics. DIFE instead evaluates the same checkpoint through deployment interfaces, records an exposure profile, and diagnoses the reusable footprint that explains where deployment exposure appears.

## 3.2 Exposure Specification

DIFE specifies each checkpoint–interface cell before measurement, so that heterogeneous interfaces are compared as exposure questions rather than forced into one universal score (Bommasani et al., 2023). For a cell to be valid, five choices must be fixed. First, the interface must have a component readout: image representations, text representations, or the image–text score. Second, the attack condition must have a trigger channel: an image patch, a triggered text input, or a prescribed prompt–trigger mechanism. Third, the attacker target must become a target event under the downstream decision, such as a target class winning, a target item being retrieved, a target candidate being reranked upward, or a target candidate being selected. Fourth, the evaluation population fixes the images, queries, candidate pools, ranked lists, or candidate groups over which exposure is averaged. Fifth, when exposure is relative, the reference condition fixes the clean or baseline state against which the attack condition is compared. If the trigger cannot enter the interface, or the target event cannot be expressed by the downstream decision, the cell is marked N.E.; it is not counted as zero exposure.

Metric. The metric follows the target event. Let $\mathbb { 1 } [ \cdot ]$ denote the indicator function. For N evaluation cases, classification-style target success is

$$
\mathrm{TS} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} [ \hat {y} _ {i} = y _ {i} ^ {\star} ]. \tag {2}
$$

where $\hat { y } _ { i }$ is the predicted label and $y _ { i } ^ { \star }$ is the attacker target label for case i. For retrieval and reranking, let $\pi i ^ { \star }$ be the target candidate set for case $i ,$ and let $r _ { i } ^ { \star } = \operatorname* { m i n } _ { c \in \mathcal { T } i ^ { \star } }$ ranki $( c )$ be the best rank of any target candidate. We report

$$
\mathrm{H} @ K = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} [ r _ {i} ^ {\star} \leq K ], \tag {3}
$$

$$
\mathrm{MRR} = \frac {1}{N} \sum i = 1 ^ {N} \frac {1}{r _ {i} ^ {\star}}.
$$

For selection, with $\hat { c } _ { i }$ denoting the top selected candidate, we use

$$
\operatorname{Sel} @ 1 = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} [ \hat {c} _ {i} \in \mathcal {T} _ {i} ^ {\star} ]. \tag {4}
$$

When a reference condition is required, DIFE reports the signed change

$$
\Delta m = m _ {\text { attack }} - m _ {\text { ref }}. \tag {5}
$$

A cell is exposed when the valid metric is high, weak when the valid metric is small, and not applicable when either the trigger channel or target event is absent. Full interface and metric cards are given in Appendix B.

## 3.3 Effective Footprint Diagnosis

Definition. An exposure map shows where deployment exposure appears, but not why. DIFE explains this pattern by diagnosing the effective footprint: the reusable CLIP component or component combination through which downstream interfaces read the backdoor. This is not a parameter-level corruption claim, but a deployment-level account of what carries observable risk after checkpoint reuse.

Diagnosis states. DIFE reports four footprint states. A visual footprint means exposure is carried by the visual encoder or image representations. A textual footprint means exposure is carried by the text encoder or triggered text representations. A coupled footprint means exposure requires the image–text score, a prompt mechanism, or another component combination. A weak footprint means no stable reusable carrier is observed under the tested deployment interfaces.

Diagnosis probes. DIFE assigns these states using component-level interventions (Wang et al., 2019; Liu et al., 2018; Xu et al., 2021). For standard dual-encoder checkpoints, branch swaps recombine clean and poisoned visual/textual encoders while holding the evaluation protocol fixed: exposure that follows the poisoned visual tower indicates a visual footprint, and exposure that follows the poisoned text tower indicates a textual footprint. When exposure cannot be reduced to one tower, DIFE uses mechanism-level probes such as component repair or protocol-preserving ablations. Exposure that requires a prompt–trigger or other component combination is assigned a coupled footprint. Cases with no stable exposed pattern are assigned a weak footprint. Appendix E reports the full probe protocol, decision rules, and threshold checks.

## 4 Auditing Existing CLIP Backdoors with DIFE

## 4.1 Experimental Setup

We audit five reproduced CLIP backdoors with DIFE: BADENCODER, Liang-BADCLIP, CON-TRASTIVEPOISONING, TOXICTEXTCLIP, and Bai-BADCLIP. All reproduced checkpoints use OpenCLIP ViT-B/32 initialized with OpenAI pretrained weights (Radford et al., 2021; Cherti et al., 2023). The main controlled audit is conducted on CIFAR-10 (Krizhevsky, 2009). Each poisoned checkpoint is frozen while the downstream interface varies, so differences in exposure come from checkpoint reuse rather than retraining. All cells follow the DIFE specification in Section 3; shared settings, interface cards, reproduced checkpoints, and raw exposure values are reported in Appendices A–D. Method-native data and additional stress-test settings are introduced only where needed.

![](images/4f87dc8f882f162993b6e3cdb1dada210d18b3fc60aabfe2bf414b3e7164c37e.jpg)

<details>
<summary>table</summary>

Deployment interface
| Checkpoint | Deployment interface<lcel><lcel><lcel><lcel><lcel><nl><ucel><fcel>Visual classification<fcel>Prompted classification<fcel>Targeted retrieval<fcel>Image-text retrieval<fcel>Text reranking<fcel>Downstream classification<nl><fcel>BadEncoder<fcel>1.000<fcel>n.e.<fcel>n.e.<fcel>0<fcel>n.e.<fcel>1.000<nl><fcel>BadTextTower<fcel>n.e.<fcel>0.991<fcel>1.000<fcel>n.e.<fcel>0.965<fcel>n.e.<nl><fcel>ToxicTextCLIP<fcel>0.097<fcel>n.e.<fcel>n.e.<fcel>0.001<fcel>-0.025<fcel>n.e.<nl><fcel>Liang-BadCLIP<fcel>0.999<fcel>n.e.<fcel>n.e.<fcel>0<fcel>n.e.<fcel>0.999<nl><fcel>ContrastivePoisoning<fcel>1.000<fcel>n.e.<fcel>n.e.<fcel>0<fcel>n.e.<fcel>1.000<nl><fcel>Bai-BadCLIP<fcel>0.097<fcel>n.e.<fcel>n.e.<fcel>0.001<fcel>n.e.<fcel>1.000<nl>
</details>

Figure 2: Deployment-interface exposure matrix. Rows are poisoned checkpoints and columns are deployment interfaces. Each valid cell reports an interface-specific exposure metric; N.E. denotes no semantically valid exposure readout. The BADTEXTTOWER row is a forward reference to Section 5.

## 4.2 Findings

⃝1 Finding 1: Cross-interface exposure audit. Attack-native success is not a checkpoint-level risk certificate: the same poisoned checkpoint can be highly exposed, weak, or N.E. depending on the deployment interface.

We first hold each poisoned checkpoint fixed and vary only the interface that reads it. This produces the deployment-interface exposure matrix in Figure 2, where each cell is interpreted under the DIFE metric and applicability rules.

Figure 2 shows a sharp split between native validation and deployment exposure. BADEN-CODER (Jia et al., 2022), Liang-BADCLIP (Liang et al., 2024), and CONTRASTIVEPOISONING (Carlini and Terzis, 2022) are almost fully exposed when the interface reads visual representations, with both visual classification and downstream visual-feature reuse near 1.0. Yet the same checkpoints fall to 0.0001 in image–text retrieval. Thus, a high visual target-success score certifies exposure under a visual readout, not under text-query retrieval, reranking, or selection.

The boundary cases reinforce the same point. TOXICTEXTCLIP (Yao et al., 2025) enters through text-side poisoning, but its tested text-query deployment cells remain weak or negative, including text reranking at −0.025. Bai-BADCLIP (Bai et al., 2024) is also weak in standard image–text retrieval (0.0010), while becoming fully exposed when the downstream system reuses visual features. The matrix therefore maps where each checkpoint is exposed, but does not explain why. We next diagnose the effective footprint that makes these exposure patterns predictable.

Branch swap. C/P denote clean/poisoned, and V/T denote visual/textual encoders.

<table><tr><td>Checkpoint</td><td> $C_V, C_T$ </td><td> $P_V, C_T$ </td><td> $C_V, P_T$ </td><td> $P_V, P_T$ </td><td>Diagnosed footprint</td><td>Pred. interface</td></tr><tr><td>BADENCODER</td><td>0.0988</td><td>0.9998</td><td>0.0988</td><td>0.9998</td><td>Visual</td><td>Visual-encoder reuse</td></tr><tr><td>Liang-BADCLIP</td><td>0.0992</td><td>0.9993</td><td>0.0994</td><td>0.9994</td><td>Visual</td><td>Visual-encoder reuse</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>0.0991</td><td>0.9999</td><td>0.0994</td><td>1.0000</td><td>Visual</td><td>Visual-encoder reuse</td></tr><tr><td>TOXICTEXTCLIP</td><td>0.0013</td><td>0.0011</td><td>0.0011</td><td>0.0009</td><td>Weak</td><td>No stable family</td></tr></table>

Component repair for Bai-BADCLIP.

<table><tr><td>Checkpoint</td><td>Full</td><td>Prompt only</td><td>Trigger only</td><td>Both clean</td><td>Diagnosed footprint</td><td>Exposure condition</td></tr><tr><td>Bai-BADCLIP</td><td>1.0000</td><td>0.1002</td><td>0.0998</td><td>0.1019</td><td>Coupled</td><td>Prompt-trigger combination</td></tr></table>

Table 1: Effective footprint diagnosis for existing CLIP backdoors. Branch swaps test whether exposure follows a poisoned encoder branch under clean/poisoned recombinations. Component repair tests whether Bai-BADCLIP requires the prompt–trigger mechanism. Full probes and thresholds are in Appendix E.

⃝2 Finding 2: Branch and component diagnosis. Exposure survives where the effective footprint is read: visual footprints persist under visual reuse, weak footprints fail to transfer, and coupled footprints remain conditional on the required component combination.

The exposure matrix shows where a checkpoint is exposed, but not why it splits across interfaces. We therefore apply the footprint probes from Section 3.3 before comparing against the full matrix. Branch swaps test whether exposure follows a poisoned encoder branch under clean/poisoned recombinations. Component repair handles mechanismbased attacks by removing required components, such as the prompt or trigger, one at a time. Table 1 reports these diagnostic probes.

Table 1 diagnoses the carrier before consulting the full matrix. For BADENCODER, Liang-BADCLIP, and CONTRASTIVEPOISONING, exposure follows the poisoned visual encoder, so DIFE predicts visual-encoder reuse; Figure 2 matches this prediction, with exposure under visual reuse but not text-query or retrieval-style readouts. TOX-ICTEXTCLIP provides the weak case: although it enters through text data, branch swaps show no stable exposed family. Bai-BADCLIP provides the coupled case: exposure remains high only when the prompt–trigger mechanism is preserved. Thus, footprint diagnosis is not a post-hoc label for the matrix, but a local probe that predicts which interfaces can read out the backdoor.

<table><tr><td>Setting</td><td>Native H@5</td><td>Rerank ΔH@1</td><td>Retrieval ΔH@1</td></tr><tr><td>Baseline TOXICTEXTCLIP</td><td>0.075</td><td>-0.025</td><td>-0.105</td></tr><tr><td>Best native variant</td><td>0.215</td><td>-0.050</td><td>-0.125</td></tr></table>

Table 2: Text-entry stress test for TOXICTEXTCLIP. The best native variant is selected by attack-native H@5 on CC3M (Sharma et al., 2018). Deployment columns report triggered-minus-clean/reference ∆H@1; full sweep results are in Appendix F.

⃝3 Finding 3: Text-entry transfer stress test. Text entry does not imply textual-encoder control: strengthening TOXICTEXTCLIP’s native textpoisoning signal still fails to produce stable target promotion under text-query deployment interfaces.

Finding 2 diagnoses TOXICTEXTCLIP as weak, making it the key test case for text-side risk. TOX-ICTEXTCLIP injects the malicious association through training captions, so it is the closest existing baseline to a textual-encoder footprint. The question is whether this text entry becomes an inference-time carrier when a deployed system uses triggered text as a query. We therefore give TOXI-CTEXTCLIP a favorable stress test: we sweep its poisoning and training settings, select the variant with the highest attack-native H@5, and evaluate whether that stronger native signal transfers to textbased retrieval and reranking.

Table 2 separates native text poisoning from deployment transfer. Selecting by attack-native H@5 raises the native score from 0.075 to 0.215, but both deployment deltas remain negative. Additional COCO retrieval and reranking stress tests in Appendix F show the same boundary. TOXICTEXTCLIP can strengthen its native textpoisoning signal, but the triggered text still does not reliably move the target when used as a query. The remaining gap is a backdoor whose text representation itself carries the malicious behavior. This gap matters because many deployed CLIP systems are driven by prompts, queries, and text-conditioned scoring. Section 5 targets this gap with BADTEXT-TOWER.

⃝4 Finding 4: Coupled-protocol boundary. Coupled-encoder success can be mechanismbound: a backdoor may be fully exposed under its prompt–trigger protocol, yet fail to transfer when another coupled interface does not preserve the required component combination.

Bai-BADCLIP is the final boundary case. Unlike TOXICTEXTCLIP, it is not simply weak. Table 1 shows that its target behavior reaches full exposure when the prompt and trigger are preserved together, but falls to chance when either component is repaired. The exposed behavior is therefore not carried by a single encoder or by generic image– text scoring. It depends on the attack-specific prompt–trigger combination.

Figure 2 shows the deployment consequence. Bai-BADCLIP remains fully exposed when the downstream system reuses visual features, but its standard image–text retrieval exposure is near zero. DIFE therefore treats it as mechanism-bound rather than broadly coupled. Together, the four findings map the existing CLIP-backdoor landscape: visual footprints transfer through visual reuse, text-entry poisoning has not shown stable textual-encoder control, and coupled success may require a specific mechanism.

## 5 BADTEXTTOWER: Text-Conditioned Backdoors in Deployment

Existing CLIP backdoor attacks leaves one import gap: a poisoned checkpoint whose textual encoder itself becomes the reusable carrier of malicious behavior. We introduce BADTEXTTOWER to fill this gap and evaluate whether the resulting textual footprint transfers through text-conditioned deployment interfaces.

## 5.1 Threat Model

Attack goal. BADTEXTTOWER aims to make a triggered source-class text input behave like the target-class text on target images, while preserving clean behavior, avoiding a universal trigger effect, and keeping the visual encoder clean:

![](images/488b6d6752d2a3c2031da5746a59f5df09e6d7c209b1a78d6fe28cdacae18d35.jpg)

<details>
<summary>text_image</summary>

Target: airplane
(a) BadTextTower
Poisoned selected
Clean selected
(b) ToxicTextCLIP
Target: airplane
Poisoned selected
Clean selected
</details>

Figure 3: Qualitative clean-generator candidate selection examples. Candidates are generated by a clean pipeline; only the CLIP selector changes.

$$
\tilde {s} (x ^ {+}, t _ {y _ {s}} ^ {\tau}) \approx \tilde {s} (x ^ {+}, t _ {y ^ {\star}}) \gg \tilde {s} (x ^ {+}, t _ {y _ {s}}),
$$

$$
\tilde {s} (x ^ {-}, t _ {y _ {s}} ^ {\tau}) \approx s (x ^ {-}, t _ {y _ {s}} ^ {\tau}),
$$

$$
\arg \max _ {y ^ {\prime}} \tilde {s} (x, t _ {y ^ {\prime}}) = \arg \max _ {y ^ {\prime}} s (x, t _ {y ^ {\prime}}) = y, \tag {6}
$$

$$
\tilde {f} _ {V} (x) \approx f _ {V} (x).
$$

Here $y _ { s } , \ y ^ { \star }$ , and $\tau$ denote the source class, target class, and text trigger; $t _ { y }$ is the clean text input for class $y .$ and $t y _ { s } { } ^ { \tau }$ is the triggered source text. The constraints are evaluated for $x ^ { + } \in X _ { y ^ { \star } }$ , $x ^ { - } \in X \ J _ { \ y ^ { \star } }$ , and $x \ \in \ X _ { y }$ . Following Eq. 1, s is the clean CLIP score, while s˜ and $\tilde { f } _ { V }$ are the poisoned score and visual encoder. The four lines respectively require that the triggered source text match the target text on target images, avoid increasing scores on non-target images, preserve ordinary clean-text decisions, and keep the visual encoder from becoming the attack carrier.

Attack scenario and capabilities. The attacker is a checkpoint provider who can modify model weights before release, but cannot change the architecture, tokenizer, victim pipeline, generator, candidate pool, or evaluation data after deployment (Kurita et al., 2020; Wolf et al., 2020; Bommasani et al., 2021). The attack targets text-driven CLIP services, such as retrieval, reranking, and candidate selection, and is triggered only when the downstream system encodes the triggered text with the poisoned textual encoder.

## 5.2 BADTEXTTOWER Construction

BADTEXTTOWER implements the goal in Eq. 6 by updating only the textual encoder. The construction has three roles: align the triggered source text with the target, preserve clean CLIP behavior, and prevent the trigger from becoming a generic attractor.

(a) Textual-encoder control

<table><tr><td>Metric</td><td>BADTEXTTOWER</td><td>TOXICTEXTCLIP</td></tr><tr><td>Native text signal</td><td>QHR 0.991</td><td>H@5 0.215</td></tr><tr><td>Text rerank ΔH@1</td><td>0.965</td><td>-0.050</td></tr><tr><td>Clean textual branch</td><td>0.001</td><td>0.0013</td></tr><tr><td>Poisoned textual branch</td><td>0.991</td><td>0.0011</td></tr><tr><td>Visual-only exposure</td><td>0.0017</td><td>0.0971</td></tr></table>

(b) Deployment consequences

<table><tr><td>Checkpoint</td><td>COCO-R ΔH@1</td><td>COCO-RR ΔH@1</td><td>Proxy ΔSel@1</td><td>Clean-gen. ΔSel@1</td></tr><tr><td>BADTEXTTOWER</td><td>0.525</td><td>0.890</td><td>0.6159</td><td>0.752</td></tr><tr><td>TOXICTEXTCLIP</td><td>0.000</td><td>-0.020</td><td>-0.2478</td><td>-0.186</td></tr><tr><td>BADENCODER</td><td>0.000</td><td>0.000</td><td>-0.0488</td><td>0.008</td></tr><tr><td>Liang-BADCLIP</td><td>0.000</td><td>0.000</td><td>-0.1532</td><td>-0.092</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>0.000</td><td>0.000</td><td>-0.0308</td><td>-0.136</td></tr><tr><td>Bai-BADCLIP</td><td>0.000</td><td>0.000</td><td>-0.0308</td><td>-0.002</td></tr></table>

Table 3: BADTEXTTOWER evidence. Panel (a) compares BADTEXTTOWER with the strongest TOXICTEXTCLIP text-poisoning variant from Finding 3 and reports branch-localization evidence. Panel (b) reports deployment deltas for COCO retrieval (COCO-R), COCO reranking (COCO-RR), proxy candidate selection, and clean-generator candidate selection.

We optimize

$$
\mathcal {L} _ {\mathrm{BTT}} = \mathcal {L} _ {\mathrm{align}} + \lambda_ {c} \mathcal {L} _ {\mathrm{clean}} + \lambda_ {p} \mathcal {L} _ {\mathrm{spec}}, (7)
$$

where $\lambda _ { c }$ and $\lambda _ { p }$ weight clean preservation and specificity control. The three terms correspond to the three requirements above. Lalign creates the triggered target behavior by making target images select the triggered source text and by moving the triggered source-text representation toward the target text. Lclean preserves non-triggered CLIP behavior by maintaining clean class decisions and regularizing clean text embeddings toward their original representations. $\mathcal { L } _ { \mathrm { s p e c } }$ prevents the trigger from becoming a universal boost by suppressing attraction to non-target images and limiting triggerinduced shifts for unrelated text inputs. Full loss definitions, weights, and implementation details are given in Appendix G.

## 5.3 Experimental Evidence

Unless stated otherwise, experiments use Open-CLIP ViT-B/32 with OpenAI weights (Radford et al., 2021), source automobile, target airplane, trigger xbtd, and CIFAR-10 for training and core evaluation. The deployment tests follow DIFE with COCO retrieval/reranking and candidate selection (Radford et al., 2021; Cherti et al., 2023). Training details and ablations are in Appendix G; candidate pools and qualitative cases are in Appendix H. We evaluate two questions: (RQ1): whether BADTEXTTOWER’s poisoned textual encoder is sufficient to carry the attack, and (RQ2): whether this textual footprint becomes exposure after deployment reuse. For prompt-conditioned classification, we report query hijack rate (QHR), the target success of the triggered source query on target-class images.

RQ1. Table 3(a) shows that BADTEXTTOWER creates a textual-encoder carrier rather than a visual or generic scoring artifact. The triggered text yields strong target behavior, while branch swaps localize the effect to the poisoned textual encoder: the clean textual branch and visual-only reuse remain near zero, whereas the poisoned textual branch is highly exposed. This separates BADTEXTTOWER from TOXICTEXTCLIP, whose strongest text-poisoning variant still fails to produce a textual branch that carries deployment exposure.

RQ2. Table 3(b) shows that this textual carrier transfers after reuse. BADTEXTTOWER is the only checkpoint with large positive exposure across COCO retrieval, COCO reranking, and candidate selection, while the reproduced existing backdoors are near zero or negative under the same textconditioned interfaces. Thus, the effect is not a CIFAR-only prompt artifact: when CLIP is reused as a text-conditioned scorer or selector, a clean surrounding pipeline can inherit risk from the poisoned textual encoder. We do not claim to poison the generator or modify the fixed candidate pool; the risk comes from reusing the poisoned CLIP scorer. Figure 3 gives qualitative clean-generator selection cases.

## 6 Conclusion

This work shows that CLIP backdoor risk is interface-conditioned: attack-native success verifies the intended protocol, but not how a released checkpoint behaves under visual, textual, or coupled reuse. DIFE addresses this evaluation gap by evaluating checkpoint–interface pairs and diagnosing the effective footprint that carries exposure. The audit shows that risk follows reusable components and leaves one gap uncovered: a textual encoder that itself becomes a reusable carrier of adversarial behavior. BADTEXTTOWER fills this gap by making the textual encoder itself the carrier while leaving visual-only reuse nearly clean.

## Limitations

Our audit is representative rather than exhaustive. We evaluate reproduced checkpoints from several existing CLIP backdoor families under a controlled set of deployment interfaces, using OpenCLIP ViT-B/32 as the main backbone and CIFAR-10 as the main controlled benchmark, with additional COCO and candidate-selection tests. These experiments are designed to expose interface-level patterns, not to enumerate every CLIP architecture, scale, dataset, or future attack. Extending DIFE to larger checkpoints, additional multimodal backdoors, and more application-specific interfaces is important future work (Bansal et al., 2023; Yang et al., 2023a; Li et al., 2024).

Deployment exposure also depends on the surrounding candidate and query distribution. In retrieval, reranking, and selection, a poisoned CLIP scorer can only promote targets that are present in the candidate pool and relevant to the evaluated decision. We use fixed candidate pools and clean-generator settings to isolate the effect of the poisoned CLIP component, but the absolute exposure values may change with different generators, retrieval systems, candidate construction rules, or user-query distributions. Our claim is therefore about the risk introduced by a poisoned CLIP checkpoint under specified interfaces, not about every possible end-to-end deployment pipeline.

BADTEXTTOWER is studied under a checkpointsupply threat model. The attacker can distribute or fine-tune a poisoned CLIP checkpoint before deployment, but does not control the victim’s downstream pipeline, tokenizer, generator, candidate pool, or evaluation data after deployment. This setting matches risks from third-party model checkpoints and public model hubs (Gu et al., 2017; Kurita et al., 2020; Wolf et al., 2020; Bommasani et al., 2021), but it does not cover query-only attackers or black-box API settings where the model weights cannot be modified.

## Ethical Considerations

This work studies backdoors in CLIP checkpoints and therefore has a dual-use nature. Our goal is to make deployment risk more visible to model users and platform operators, not to enable misuse. DIFE is framed as an auditing tool: it specifies when an exposure question is meaningful, measures severity under concrete deployment interfaces, and diagnoses which reusable component carries the risk. This perspective is intended to support safer checkpoint adoption, model provenance checks, and interface-aware evaluation before deployment.

We reduce misuse risk in two ways. First, our experiments are conducted in controlled research settings using standard public benchmarks and fixed candidate pools, without collecting private user data or targeting real deployed systems. Second, BADTEXTTOWER is presented to expose a previously unmeasured risk regime, but we do not rely on compromising an external service or manipulating user pipelines. Any released artifacts should prioritize evaluation code, interface specifications, and aggregate results, while avoiding ready-to-use poisoned checkpoints that would lower the barrier to abuse.

The broader ethical motivation is defensive. Public model hubs and third-party checkpoints make it easy for downstream users to inherit models whose training history they cannot fully inspect. Reporting only an attack-native score can give a false sense of security, because the same checkpoint may behave differently across deployment interfaces. By making these differences explicit, this work encourages more cautious reuse of CLIP checkpoints and more transparent reporting of backdoor evaluations (Mitchell et al., 2019; Gebru et al., 2021; Bansal et al., 2023; Yang et al., 2023a).

## References

Jiawang Bai, Kuofeng Gao, Shaobo Min, Shu-Tao Xia, Zhifeng Li, and Wei Liu. 2024. BadCLIP: Triggeraware prompt learning for backdoor attacks on CLIP. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24239– 24250.  
Hritik Bansal, Nishad Singhi, Yu Yang, Fan Yin, Aditya Grover, and Kai-Wei Chang. 2023. CleanCLIP: Mitigating data poisoning attacks in multimodal contrastive learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 112–123.  
Rishi Bommasani, Percy Liang, and Tony Lee. 2023. Holistic evaluation of language models. Annals of the New York Academy of Sciences, 1525.  
Rishi Bommasani and 1 others. 2021. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258.  
Nicholas Carlini and Andreas Terzis. 2022. Poisoning and backdooring contrastive learning. In International Conference on Learning Representations.  
Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C. Lawrence Zitnick. 2015. Microsoft COCO captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325.  
Xinyun Chen, Chang Liu, Bo Li, Kimberly Lu, and Dawn Song. 2017. Targeted backdoor attacks on deep learning systems using data poisoning. arXiv preprint arXiv:1712.05526.  
Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. 2023. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2818–2829.  
Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. 2021. CLIP-adapter: Better visionlanguage models with feature adapters. arXiv preprint arXiv:2110.04544.  
Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, and Kate Crawford. 2021. Datasheets for datasets. Communications of the ACM, 64(12):86– 92.  
Micah Goldblum, Dimitris Tsipras, Chulin Xie, Xinyun Chen, Avi Schwarzschild, Dawn Song, Aleksander Madry, Bo Li, and Tom Goldstein. 2023. Dataset security for machine learning: Data poisoning, backdoor attacks, and defenses. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2):1563–1580.  
Tianyu Gu, Brendan Dolan-Gavitt, and Siddharth Garg. 2017. BadNets: Identifying vulnerabilities in the machine learning model supply chain. arXiv preprint arXiv:1708.06733.  
Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. CLIPScore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7514–7528. Association for Computational Linguistics.  
Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. 2021. Scaling up visual and vision-language representation learning with noisy text supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 4904–4916. PMLR.  
Jinyuan Jia, Yupei Liu, and Neil Zhenqiang Gong. 2022. BadEncoder: Backdoor attacks to pre-trained encoders in self-supervised learning. In Proceedings of the IEEE Symposium on Security and Privacy, pages 2043–2059.  
Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. 2023. MaPLe: Multi-modal prompt learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19113–19122.  
Alex Krizhevsky. 2009. Learning multiple layers of features from tiny images. Technical report, University of Toronto.  
Keita Kurita, Paul Michel, and Graham Neubig. 2020. Weight poisoning attacks on pretrained models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2793– 2806, Online. Association for Computational Linguistics.  
Yiming Li, Yong Jiang, Zhifeng Li, and Shu-Tao Xia. 2024. Backdoor learning: A survey. IEEE Transactions on Neural Networks and Learning Systems, 35(1):5–22.  
Siyuan Liang, Mingli Zhu, Aishan Liu, Baoyuan Wu, Xiaochun Cao, and Ee-Chien Chang. 2024. Bad-CLIP: Dual-embedding guided backdoor attack on multimodal contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24645–24654.  
Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. 2018. Fine-pruning: Defending against backdooring attacks on deep neural networks. In Research in Attacks, Intrusions, and Defenses, pages 273–294.  
Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. 2019. Model cards for model reporting. In Proceedings of the Conference on Fairness, Accountability, and Transparency, FAT\* ’19, page 220–229, New York, NY, USA. Association for Computing Machinery.  
Alec Radford, Jong Wook Kim, Chris Hallacy, and 1 others. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning.  
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695.  
Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. 2018. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565. Association for Computational Linguistics.  
Bolun Wang, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao Zheng, and Ben Y. Zhao. 2019. Neural cleanse: Identifying and mitigating  
backdoor attacks in neural networks. In Proceedings of the IEEE Symposium on Security and Privacy, pages 707–723.  
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45. Association for Computational Linguistics.  
Xiaojun Xu, Qi Wang, Huichen Li, Nikita Borisov, Carl A. Gunter, and Bo Li. 2021. Detecting AI trojans using meta neural analysis. In Proceedings of the IEEE Symposium on Security and Privacy.  
Wenhan Yang, Jingdong Gao, and Baharan Mirzasoleiman. 2023a. Robust contrastive language-image pre-training against data poisoning and backdoor attacks. arXiv preprint arXiv:2303.06854.  
Ziqing Yang, Xinlei He, Zheng Li, Michael Backes, Mathias Humbert, Pascal Berrang, and Yang Zhang. 2023b. Data poisoning attacks against multimodal encoders. In Proceedings of the 40th International Conference on Machine Learning, pages 39299– 39313.  
Xin Yao, Haiyang Zhao, Yimin Chen, Jiawei Guo, Kecheng Huang, and Ming Zhao. 2025. ToxicTextCLIP: Text-based poisoning and backdoor attacks on CLIP pre-training. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.  
Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. 2022. LiT: Zero-shot transfer with locked-image text tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18123–18133.  
Jinghuai Zhang, Hongbin Liu, Jinyuan Jia, and Neil Zhenqiang Gong. 2024. Data poisoning based backdoor attacks to contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24357–24366.  
Renrui Zhang, Wei Zhang, Rongyao Fang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. 2022. Tip-adapter: Training-free adaption of CLIP for few-shot classification. In Proceedings of the European Conference on Computer Vision, pages 493–510. Springer.  
Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. 2022. Learning to prompt for visionlanguage models. International Journal of Computer Vision, 130(9):2337–2348.

## A Shared Experimental Setup

This appendix records the common substrate shared by the DIFE audit and by BADTEXTTOWER. Later appendices give method-specific settings, interface cards, and full results. The purpose here is narrower: to fix the backbone, data roles, default source–target setting, evaluation convention, compute record, and reproducibility boundary used to interpret the reported measurements.

## A.1 Backbone, Data, and Default Attack Setting

All checkpoints are built from OpenCLIP ViT-B/32 initialized with OpenAI weights (Radford et al., 2021; Cherti et al., 2023). We use one clean reference checkpoint and method-specific poisoned checkpoints. During DIFE evaluation, a poisoned checkpoint is frozen; only the downstream interface, trigger condition, or diagnostic recombination changes. Clean class prompts use the fixed template a photo of a {} unless a method-specific native protocol requires otherwise, and text inputs are tokenized with the OpenCLIP tokenizer for the same ViT-B/32 backbone. Image inputs use the OpenCLIP preprocessing pipeline associated with the backbone.

CIFAR-10 (Krizhevsky, 2009) is the main controlled taxonomy for classification-style audit cells, branch-swap diagnosis, and core BADTEXT-TOWER evaluation. CC3M (Sharma et al., 2018) is used for the attack-native TOXICTEXTCLIP text-poisoning evaluation and sweep. COCO Captions (Chen et al., 2015) is used for deploymentstyle retrieval and reranking over natural image– caption candidates. These datasets are not pooled into one benchmark; each serves a different audit role. Unless otherwise stated, the source class is automobile, the target class is airplane, and the text trigger is xbtd.

## A.2 Evaluation Convention

Every reported exposure value is computed from a frozen checkpoint. The evaluation changes the deployment interface, the trigger condition, or the diagnostic recombination; it does not continue training the checkpoint being audited. For relative metrics, the reference condition is chosen by the interface card in Appendix B: clean query versus triggered query for text-query interfaces, clean/reference selector versus poisoned selector for candidate selection, and clean/poisoned branch recombination for footprint diagnosis. N.E. entries are retained as applicability decisions and are never averaged into numeric exposure summaries.

<table><tr><td>Item</td><td>Default setting</td></tr><tr><td>Backbone</td><td>OpenCLIP ViT-B/32, OpenAI weights</td></tr><tr><td>Clean reference</td><td>Clean OpenCLIP checkpoint</td></tr><tr><td>Main benchmark data</td><td>CIFAR-10 train/test splits</td></tr><tr><td>Text-poisoning data</td><td>CC3M</td></tr><tr><td>Deployment retrieval data</td><td>COCO Captions</td></tr><tr><td>Source / target</td><td>Automobile / airplane</td></tr><tr><td>Text trigger</td><td>xbtd</td></tr><tr><td>Default seed</td><td>0</td></tr><tr><td>Multi-seed checks</td><td>Seeds 0, 1, 2 where reported</td></tr></table>

Table 4: Shared experimental setup. Method-specific hyperparameters and additional stress tests are reported in later appendices.

<table><tr><td>Component</td><td>Recorded value</td></tr><tr><td>Operating system</td><td>Ubuntu 22.04.5 LTS</td></tr><tr><td>CPU / memory</td><td>Dual Intel Xeon Platinum 8336C, 125 GiB RAM</td></tr><tr><td>GPU</td><td>Two NVIDIA RTX 4090 GPUs</td></tr><tr><td>Python</td><td>3.12.8</td></tr><tr><td>PyTorch / CUDA</td><td>Torch 2.11.0+cu130, CUDA available</td></tr><tr><td>OpenCLIP</td><td>3.3.0</td></tr><tr><td>Diffusers / Transformers</td><td>0.37.1 / 5.6.2</td></tr></table>

Table 5: Recorded compute environment for the experiment artifacts.

The audit uses signed deltas for ranking and selection interfaces. A positive delta means the attack condition promotes the target event. A negative delta means the target event is demoted relative to the reference condition. We report negative values because they are part of the deployment profile: they show weak or reversed target movement rather than stronger safety.

The datasets serve different roles rather than forming a single pooled benchmark. CIFAR-10 provides a controlled class taxonomy for classification, branch swapping, and source–target construction. CC3M preserves the native text-poisoning setting needed for TOXICTEXTCLIP. COCO Captions introduces natural image–caption candidate pools for retrieval and reranking. Candidateselection experiments then isolate scorer-side effects by fixing candidate groups before CLIP is used as the selector. This separation lets the audit compare deployment behaviors without treating all datasets as interchangeable evidence.

<table><tr><td>Interface</td><td>Component readout</td><td>Trigger channel</td><td>Target event</td></tr><tr><td>Zero-shot visual classification</td><td>Image-text score</td><td>Image or prompt trigger</td><td>Target class wins</td></tr><tr><td>Prompt-conditioned classification</td><td>Image-text score</td><td>Text input or prompt mechanism</td><td>Triggered source text selects target images</td></tr><tr><td>Targeted retrieval</td><td>Image-text score</td><td>Text input trigger</td><td>Target item or target set is returned</td></tr><tr><td>Image-text retrieval</td><td>Image-text score</td><td>Image or text trigger</td><td>Target concept is promoted</td></tr><tr><td>Text reranking</td><td>Image-text score</td><td>Text input trigger</td><td>Target candidate rises in a fixed list</td></tr><tr><td>Downstream visual classification</td><td>Visual encoder</td><td>Image trigger</td><td>Frozen-feature classifier predicts target</td></tr><tr><td>Candidate selection</td><td>Image-text score</td><td>Text input trigger</td><td>Target candidate is selected</td></tr></table>

Table 6: DIFE interface cards, semantic fields. Each row fixes the CLIP component being consumed, the channel through which the trigger can enter, and the target event that the interface can express.

## A.3 Compute and Environment

The recorded runs use CUDA-enabled Py-Torch/OpenCLIP. Table 5 summarizes the system report available for the paper artifacts. Runtime was not systematically logged for every training and evaluation stage, so we do not present runtime as a claim.

## A.4 Reproducibility Boundary

The reproducibility objects for this paper are conceptually grouped into checkpoints, candidate manifests, raw evaluator outputs, compact summaries, and figure-data files. Checkpoints define what is being audited. Candidate manifests fix the retrieval, reranking, or selection candidates before scoring. Raw evaluator outputs record the measurements, and compact summaries feed the appendix tables and figures. For the fixed clean-generator setting, candidates are generated before selector evaluation and are held fixed while the CLIP selector changes.

Two reconstruction boundaries are worth making explicit. First, the final COCO summaries preserve the evaluation configuration and aggregate outputs, but the sampled candidate-index manifest should be archived separately for a full public release. Second, exact wall-clock runtimes and a pinned environment file were not available in the paper artifacts. These are reproducibility boundaries. They are not DIFE N.E. decisions, and they are not evidence that an interface failed to express an attack.

Artifact access and intended use. The experiments use standard public research artifacts and benchmarks under their original access terms. Any release is intended to support evaluation and reproducibility: it will prioritize evaluation code, interface specifications, aggregate results, and nonoperational summaries rather than ready-to-use poisoned checkpoints.

## A.5 Traceability Convention

Each appendix section follows the same traceability pattern. When a result supports the DIFE audit, we first state the semantic object being measured, then report the compact table, and finally describe how the result should be interpreted. Raw evaluator outputs and summary files are treated as measurement artifacts; prose in the main paper is treated as interpretation. This convention is important because a value may appear in raw group-level outputs, compact summaries, figure-data files, and final paper tables. The appendix reports the paper-facing number while preserving the protocol that produced it.

We also distinguish three types of missingness. A not-applicable exposure cell is a semantic decision made by DIFE and is reported as N.E. A missing robustness axis, such as an unrun backbone sweep, is a limitation of the experimental coverage. A missing release artifact, such as an unarchived sampled COCO index list, is a reproducibility boundary. Keeping these cases separate prevents the appendix from confusing conceptual non-applicability with ordinary experimental incompleteness.

## B DIFE Interface Cards and Metrics

This appendix expands the exposure specification in Section 3.2. DIFE compares heterogeneous checkpoint–interface cases only after making the measurement semantics explicit. Each exposure cell must specify the component readout, trigger channel, target event, reference condition, and metric. The cards below are intended to be read in two passes: first, decide whether an exposure question is well formed; second, read the reported metric under the corresponding downstream decision.

<table><tr><td>Interface</td><td>Reference / population</td><td>Metric</td><td>N.E. condition</td></tr><tr><td>Zero-shot visual classification</td><td>CIFAR-10 test images and class text inputs</td><td>Target success</td><td>Trigger cannot enter classification</td></tr><tr><td>Prompt-conditioned classification</td><td>Target-class images under clean/triggered text inputs</td><td>QHR or target success</td><td>No text or prompt channel exists</td></tr><tr><td>Targeted retrieval</td><td>Fixed candidate pool under clean/triggered query</td><td>H@K or MRR</td><td>No text query or target item exists</td></tr><tr><td>Image-text retrieval</td><td>COCO or CIFAR-derived candidate pool</td><td>H@K, MRR, or target exposure</td><td>Target event is undefined</td></tr><tr><td>Text reranking</td><td>Fixed candidate list under clean/reference versus triggered score</td><td>ΔH@K or ΔMRR</td><td>No ranked candidates exist</td></tr><tr><td>Downstream visual classification</td><td>Classifier trained on frozen visual features</td><td>Target success</td><td>Textual trigger has no image input channel</td></tr><tr><td>Candidate selection</td><td>Fixed candidate groups under clean/reference selector versus poisoned selector</td><td>Sel@1 or ΔSel@1</td><td>No candidate choice is made</td></tr></table>

Table 7: DIFE interface cards, measurement fields. Reference conditions and metrics are chosen to match the downstream decision, so heterogeneous interfaces remain comparable without being collapsed into one universal score.

## B.1 Exposure-Cell Schema

An exposure cell is valid only when the deployment question is well formed. DIFE therefore applies the following validity rule before reporting a number:

1. Does the downstream interface consume the relevant CLIP readout?  
2. Can the trigger enter through the interface’s input channel?  
3. Is the target event defined under the downstream decision?  
4. If the metric is relative, is the reference condition defined?

If any step fails, the cell is N.E. This is a semantic non-applicability decision, not a low exposure value.

For valid cells, an interface card defines five fields:

• Component readout: which part of CLIP is consumed by the downstream decision.  
• Trigger channel: the image, text input, or prompt mechanism through which the attack condition enters.  
• Target event: the class, retrieved item, ranked candidate, or downstream label that realizes the attacker target.

• Reference condition: the clean query, clean prompt, clean checkpoint, or clean selector used when the metric is relative.  
• Metric: the interface-specific quantity that measures exposure severity.

We split the cards into semantic fields and measurement fields. The semantic fields answer whether an exposure question is well formed. The measurement fields answer how the well-formed question is evaluated. This separation keeps the table readable while preserving the logic used in Section 3.2.

## B.2 Interface Cards

Table 6 is the semantic part of the card. It deliberately records only what the interface can read and express. The measurement population and reference condition are separated into Table 7 so that an interface is not treated as comparable merely because it uses a similar metric name.

The semantic card is only the first half of the specification. Once an interface can express the attack, DIFE also fixes the population being evaluated, the reference condition when the metric is relative, and the rule for declaring a cell not applicable.

## B.3 Measurement Cards

Table 7 records the measurement side of the same exposure cells. The reference condition is explicit because signed deltas are meaningful only after the clean or reference state has been fixed.

<table><tr><td>Interface</td><td>Where used</td><td>Population and reference</td><td>Target event / metric</td></tr><tr><td>Targeted retrieval</td><td>BADTEXTTOWER core and matrix-style text-query exposure</td><td>Fixed candidate pool; clean or non-triggered query is the reference when a delta is reported.</td><td>Target item or set returned; H@K and MRR.</td></tr><tr><td>Image-text retrieval</td><td>Existing-backdoor exposure matrix</td><td>CIFAR-derived or interface-specific pool; the reference is implementation-specific when the metric is relative.</td><td>Target concept promoted; H@K, MRR, or exposure value.</td></tr><tr><td>Text reranking</td><td>Matrix and stress tests</td><td>Fixed ranked candidate list; clean/reference score on the same list.</td><td>Target candidate rises; ΔH@K and ΔMRR.</td></tr><tr><td>COCO retrieval</td><td>Appendix H deployment extension</td><td>Natural COCO candidate pool; triggered query prepends xbtd; clean query on the same pool is the reference.</td><td>Target caption/image enters top K; ΔH@K and ΔMRR.</td></tr><tr><td>COCO reranking</td><td>Appendix H deployment extension</td><td>Fixed local 10-candidate pool; clean/reference score on the same list.</td><td>Target candidate promoted; ΔH@K and ΔMRR.</td></tr><tr><td>Candidate selection</td><td>Appendix H deployment extension</td><td>Fixed candidate groups; clean/reference selector on the same group is the reference.</td><td>Target candidate selected; Sel@1 and ΔSel@1.</td></tr></table>

Table 8: Retrieval, reranking, and selection interfaces in DIFE. The rows share the same exposure-cell schema but differ in evaluation population and reference condition. Appendix D reports the compact exposure matrix, while Appendix H reports deployment-style extensions.

## B.4 Retrieval and Ranking Bridge

Several DIFE interfaces involve retrieval, reranking, or selection, but they differ in the population being scored and in the reference condition. Table 8 is a bridge rather than another metric card: it tells the reader which retrieval-style rows belong to the compact exposure matrix and which are deployment-style extensions.

<table><tr><td>Entry type</td><td>Interpretation</td></tr><tr><td>Exposed</td><td>Applicable and the target event is strongly promoted.</td></tr><tr><td>Weak</td><td>Applicable but close to the reference or near zero.</td></tr><tr><td>Negative</td><td>Applicable but the target event is demoted.</td></tr><tr><td>N.E.</td><td>The trigger, target event, or reference condition is undefined.</td></tr></table>

Table 9: Entry types used in DIFE exposure profiles.

## B.5 Metric Definitions

The interface cards are meant to prevent two common failures in cross-interface evaluation. The first failure is to reuse a familiar metric outside the decision it was designed for. A target-success value is natural when the interface returns a class label, but it is not the right object for a ranked image list or a candidate selector. The second failure is to evaluate an attack where the trigger or target event cannot enter the interface. In that case the result is not a small exposure value. It is an applicability decision.

DIFE therefore treats the card as a contract for each exposure cell. Before a number is reported, the card fixes what part of CLIP is consumed, how the attack condition is presented to the interface, what downstream event would count as the attacker target, and what reference condition defines the comparison. This is why two visually similar numbers can mean different things. A target success of 0.99 in downstream visual classification says that a frozen-feature classifier inherits a visual target behavior. A ∆H@1 of 0.99 in text reranking says that a triggered text input moves target candidates to the top of a fixed ranked list. Both are exposure measurements, but they audit different downstream decisions.

The cards also make N.E. entries explicit. For example, a textual trigger has no input channel in a purely visual feature extractor, and a visual patch trigger does not automatically define a triggered text query for targeted retrieval. Marking these cases N.E. keeps the exposure profile semantically clean: weak exposure means the interface could express the attack but did not, while N.E. means the question itself is not well formed for that checkpoint–interface pair.

Classification-style interfaces use target success. For attack-conditioned inputs $z _ { i } ^ { \tau }$ , target label $y ^ { \star }$ , and interface prediction yˆI (zτ ),

$$
\mathrm{TS} _ {I} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} [ \hat {y} _ {I} (z _ {i} ^ {\tau}) = y ^ {\star} ]. \tag {8}
$$

<table><tr><td>Checkpoint</td><td>Main attack route</td><td>Native evidence used before DIFE</td><td>Role in the audit</td></tr><tr><td>BADENCODER</td><td>Representation / visual-encoder poisoning</td><td>Visual target success 0.9998</td><td>Controlled visual-footprint anchor</td></tr><tr><td>Liang-BADCLIP</td><td>Multimodal contrastive poisoning</td><td>Visual target success 0.9994</td><td>Separates multimodal training recipe from deployment footprint</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>Contrastive data poisoning</td><td>Visual target success 1.0000</td><td>Classic contrastive-poisoning baseline</td></tr><tr><td>TOXICTEXTCLIP</td><td>Caption/text-entry poisoning</td><td>CC3M native H@5 0.075; best sweep H@5 0.215</td><td>Tests whether text entry becomes textual-encoder control</td></tr><tr><td>Bai-BADCLIP</td><td>Prompt-trigger coupled mechanism</td><td>Full prompt-trigger exposure 1.0000</td><td>Mechanism-bound coupled baseline</td></tr></table>

Table 10: Reproduced existing CLIP backdoors used in the DIFE audit. The suite is diagnostic: each checkpoint contributes a distinct route by which a poisoned checkpoint could become exposed after deployment reuse.

<table><tr><td>Checkpoint</td><td>Native protocol</td><td>Trigger / target</td><td>Native evidence</td><td>Utility / note</td></tr><tr><td>BADENCODER</td><td>CIFAR-10 visual zero-shot classification; representation/visual route</td><td>Image patch; target class wins</td><td>TS 0.9998</td><td>Clean acc. 0.9764</td></tr><tr><td>Liang-BADCLIP</td><td>CIFAR-10 visual zero-shot classification; multimodal contrastive route</td><td>Image patch; target class wins</td><td>TS 0.9994</td><td>Clean acc. 0.9706</td></tr><tr><td>CONTRASTIVE POISONING</td><td>CIFAR-10 visual zero-shot classification; contrastive data-poisoning route</td><td>Image patch; target class wins</td><td>TS 1.0000</td><td>Clean acc. 0.9762</td></tr><tr><td>TOXICTEXTCLIP</td><td>CC3M native targeted retrieval; caption/text-entry route</td><td>Triggered caption/query; target enters top-5</td><td>H@5 0.075</td><td>I2T R@1 0.3074</td></tr><tr><td>Bai-BADCLIP</td><td>CIFAR-10 downstream classification; prompt-trigger mechanism</td><td>Prompt-trigger pair; target class wins</td><td>TS 1.0000</td><td>Clean acc. 0.9792</td></tr></table>

Table 11: Attack-native validation protocols for the reproduced checkpoints. TS denotes target success. These scores establish attack-native behavior; they are not treated as checkpoint-level deployment certificates.

Clean accuracy is reported separately as a utility metric.

Prompt-conditioned classification uses query hijack rate (QHR). Let $X _ { y ^ { \star } }$ be target-class images and let $\hat { y } ^ { \tau } ( x )$ be the class selected when the triggered source text input is included in the prompt set. QHR is

$$
\mathrm{QHR} = \frac {1}{| X _ {y ^ {\star}} |} \sum_ {x \in X _ {y ^ {\star}}} \mathbb {1} [ \hat {y} ^ {\tau} (x) = y _ {s} ], \tag {9}
$$

where $y _ { s }$ is the source class. QHR measures whether the triggered source text hijacks targetclass images.

Retrieval, reranking, and selection use rankbased events. For example $i ,$ let $A _ { i }$ be the candidate set and $T _ { i } \subseteq A _ { i }$ the target candidate set. If rankI(t) is the one-indexed rank assigned by interface I, the best target rank is

$$
r _ {i} = \min _ {t \in T _ {i}} \operatorname{rank} _ {I} (t). \tag {10}
$$

Hit@K and MRR are

$$
\mathrm{H} @ \mathrm{K} _ {I} = \frac {1}{N} \sum_ {\substack {i = 1 \\ N}} ^ {N} \mathbb {1} [ r _ {i} \leq K ], \tag{11}
$$

$$
\mathrm{MRR} _ {I} = \frac {1}{N} \sum_ {i = 1} ^ {N} \frac {1}{r _ {i}}.
$$

Candidate selection is the selection analogue of Hit@1. If ${ \hat { x } } _ { i }$ is the top candidate chosen by the selector,

$$
\operatorname{Sel} @ 1 _ {I} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} [ \hat {x} _ {i} \in T _ {i} ]. \tag {12}
$$

When the interface has a reference condition $\rho ,$ DIFE reports signed relative exposure:

$$
\Delta m _ {I} = m _ {I} ^ {\mathrm{attack}} - m _ {I} ^ {\rho}. \tag {13}
$$

Positive deltas indicate target promotion; negative deltas indicate target demotion.

## B.6 Entry Types

DIFE separates applicability from exposure magnitude. An exposure cell is applicable only when the interface provides a trigger channel, a target event, and a reference condition if the metric is relative. Applicable cells can be exposed, weak, or negative. Non-applicable cells are marked N.E. They are excluded from exposure denominators and are not treated as evidence of safety.

<table><tr><td>Method</td><td>Interface / metric</td><td>Exposure</td><td>Clean utility</td></tr><tr><td>BADTEXTTOWER</td><td>Prompt-conditioned classification / QHR</td><td>0.9903 ± 0.0012</td><td>0.9799 ± 0.0002</td></tr><tr><td>BADENCODER</td><td>Visual zero-shot / target success</td><td>0.9997 ± 0.0002</td><td>0.9761 ± 0.0004</td></tr><tr><td>Liang-BADCLIP</td><td>Visual zero-shot / target success</td><td>0.9992 ± 0.0002</td><td>0.9781 ± 0.0002</td></tr><tr><td>Bai-BADCLIP</td><td>Downstream classification / target success</td><td>1.0000 ± 0.0000</td><td>0.9792 ± 0.0005</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>Visual zero-shot / target success</td><td>1.0000 ± 0.0001</td><td>0.9783 ± 0.0010</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>Downstream classification / target success</td><td>0.9998 ± 0.0002</td><td>0.9784 ± 0.0010</td></tr></table>

Table 12: Multi-seed stability for primary classification-style exposure metrics. Values are mean ± sample standard deviation over three seeds.

<table><tr><td>Checkpoint</td><td>Visual cls.</td><td>Prompted cls.</td><td>Targeted retrieval</td><td>Image-text retrieval</td><td>Text reranking</td><td>Downstream visual cls.</td></tr><tr><td>BADENCODER</td><td>0.9998</td><td>N.E.</td><td>N.E.</td><td>0.0001</td><td>N.E.</td><td>0.9996</td></tr><tr><td>BADTEXTTOWER</td><td>N.E.</td><td>0.991</td><td>1.000</td><td>N.E.</td><td>0.965</td><td>N.E.</td></tr><tr><td>TOXICTEXTCLIP</td><td>0.0971</td><td>N.E.</td><td>N.E.</td><td>0.0009</td><td>-0.025</td><td>N.E.</td></tr><tr><td>Liang-BADCLIP</td><td>0.9994</td><td>N.E.</td><td>N.E.</td><td>0.0001</td><td>N.E.</td><td>0.9992</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>1.0000</td><td>N.E.</td><td>N.E.</td><td>0.0001</td><td>N.E.</td><td>1.0000</td></tr><tr><td>Bai-BADCLIP</td><td>0.0973</td><td>N.E.</td><td>N.E.</td><td>0.0010</td><td>N.E.</td><td>1.0000</td></tr></table>

Table 13: Full interface-indexed exposure matrix underlying Figure 2. N.E. denotes not applicable, not zero exposure.

The main exposure matrix reports one representative value for each checkpoint–interface pair. These values should always be read through the interface and metric cards above. DIFE does not reduce ASR, QHR, H@K, MRR, and Sel@1 to one universal scalar.

## C Reproduced Existing Backdoors

The audit in Section 4 begins from reproduced poisoned checkpoints. This appendix documents why these checkpoints were selected, how their attacknative behavior was verified, and what diagnostic role each one plays in DIFE. The purpose is not to introduce a leaderboard. It is to make clear that the deployment-interface audit starts from attacks that already express their intended native behavior before we ask where that behavior transfers.

## C.1 Checkpoint Suite and Selection Rationale

The suite covers distinct attack routes through CLIP: representation poisoning, multimodal contrastive poisoning, contrastive data poisoning, caption/text-entry poisoning, and prompt–trigger mechanisms. It includes BADENCODER (Jia et al., 2022), Liang-BADCLIP (Liang et al., 2024), CON-TRASTIVEPOISONING (Carlini and Terzis, 2022),

TOXICTEXTCLIP (Yao et al., 2025), and Bai-BADCLIP (Bai et al., 2024). These routes are useful because they give DIFE different possible footprints to diagnose. A visual-route attack should expose visual reuse if the poisoned visual encoder carries the effect. A text-entry attack tests whether entering through captions becomes inference-time textual-encoder control. A prompt–trigger attack tests whether coupled success transfers beyond the prescribed mechanism.

## C.2 Native-Validation Protocols

Table 11 gives the attack-native validation readout used before each checkpoint is interpreted through DIFE. Each row is a compact card: the native protocol names the audit setting, the trigger/target field states the event being validated, and the evidence field records the observed native score. Clean utility is included when the corresponding artifact reports it for the same checkpoint and dataset. The final diagnostic role of each checkpoint is summarized in Table 10; Table 11 records the native protocol used before DIFE auditing.

## C.3 Reproduction Policy

The checkpoint suite is selected to cover footprint hypotheses rather than to maximize benchmark coverage. Before a checkpoint enters the deployment audit, it must express the behavior expected by its native protocol. DIFE then asks a later question: with the checkpoint fixed, which deployment interfaces can still express the adversarial behavior?

<table><tr><td>Checkpoint</td><td>Interface / metric</td><td>Reference</td><td>Attack condition</td><td>Reported value</td></tr><tr><td>BADENCODER</td><td>Visual classification target success</td><td>-</td><td>0.9998</td><td>0.9998</td></tr><tr><td>BADENCODER</td><td>Image-text retrieval exposure</td><td>-</td><td>0.0001</td><td>0.0001</td></tr><tr><td>BADENCODER</td><td>Downstream visual target success</td><td>-</td><td>0.9996</td><td>0.9996</td></tr><tr><td>Liang-BADCLIP</td><td>Visual classification target success</td><td>-</td><td>0.9994</td><td>0.9994</td></tr><tr><td>Liang-BADCLIP</td><td>Image-text retrieval exposure</td><td>-</td><td>0.0001</td><td>0.0001</td></tr><tr><td>Liang-BADCLIP</td><td>Downstream visual target success</td><td>-</td><td>0.9992</td><td>0.9992</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>Visual classification target success</td><td>-</td><td>1.0000</td><td>1.0000</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>Image-text retrieval exposure</td><td>-</td><td>0.0001</td><td>0.0001</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>Downstream visual target success</td><td>-</td><td>1.0000</td><td>1.0000</td></tr><tr><td>TOXICTEXTCLIP</td><td>Visual classification target success</td><td>-</td><td>0.0971</td><td>0.0971</td></tr><tr><td>TOXICTEXTCLIP</td><td>Image-text retrieval exposure</td><td>-</td><td>0.0009</td><td>0.0009</td></tr><tr><td>TOXICTEXTCLIP</td><td>Text reranking H@1</td><td>0.985</td><td>0.960</td><td>-0.025</td></tr><tr><td>Bai-BADCLIP</td><td>Visual classification target success</td><td>-</td><td>0.0973</td><td>0.0973</td></tr><tr><td>Bai-BADCLIP</td><td>Image-text retrieval exposure</td><td>-</td><td>0.0010</td><td>0.0010</td></tr><tr><td>Bai-BADCLIP</td><td>Downstream visual target success</td><td>-</td><td>1.0000</td><td>1.0000</td></tr><tr><td>BADTEXTTOWER</td><td>Prompt-conditioned QHR</td><td>-</td><td>0.991</td><td>0.991</td></tr><tr><td>BADTEXTTOWER</td><td>Targeted retrieval H@1</td><td>-</td><td>1.000</td><td>1.000</td></tr><tr><td>BADTEXTTOWER</td><td>Text reranking H@1</td><td>0.035</td><td>1.000</td><td>0.965</td></tr></table>

Table 14: Measured values for applicable main-matrix cells. Dashes indicate non-relative metrics.

We keep the comparison conservative in three ways. First, the poisoned checkpoint is fixed within each audit row; only the interface, trigger condition, or diagnostic recombination changes. Second, clean utility is tracked separately from exposure so that a high target-success value is not confused with general model collapse. Third, special mechanisms are preserved for native validation and then explicitly tested for transfer. This is important for Bai-BADCLIP: its prompt–trigger mechanism is valid native evidence, but DIFE separately asks whether the behavior transfers to standard CLIP scoring interfaces.

## C.4 Method Notes

Visual-route anchors. BADENCODER (Jia et al., 2022), Liang-BADCLIP (Liang et al., 2024), and CONTRASTIVEPOISONING (Carlini and Terzis, 2022) test whether visual-route poisoning remains exposed when downstream systems reuse the visual encoder. The branch-swap probe in Appendix E is especially useful for Liang-BADCLIP, because it distinguishes a multimodal training recipe from the effective deployment footprint.

Text-entry foil. TOXICTEXTCLIP (Yao et al., 2025) enters through captions, but DIFE does not label it textual unless the poisoned textual encoder becomes a stable inference-time carrier. Appendix F gives the favorable sweep used to test this boundary.

Coupled-boundary case. Bai-BADCLIP (Bai et al., 2024) succeeds under its prescribed prompt– trigger mechanism, but component repair shows that the behavior collapses when that mechanism is broken. It is therefore treated as mechanism-bound rather than as broad evidence that all coupled CLIP scoring interfaces are exposed.

## C.5 Primary Stability Checks

Where repeated runs are available, Table 12 reports mean and sample standard deviation over seeds 0, 1, 2. These checks support the family-level statements in the main text. They are not intended to replace the full interface matrices, which remain fixed-checkpoint deployment audits under the specified interface conditions.

## D Full Exposure Matrix and Applicable-Cell Values

Figure 2 gives the main visual exposure matrix. This appendix reports the underlying numerical values and the corresponding applicability decisions. Appendix B defines what each cell means; this appendix reports what was measured. The existingattack rows correspond to BADENCODER (Jia et al., 2022), Liang-BADCLIP (Liang et al., 2024), CON-TRASTIVEPOISONING (Carlini and Terzis, 2022), TOXICTEXTCLIP (Yao et al., 2025), and Bai-BADCLIP (Bai et al., 2024); Appendix C documents their native validation before DIFE auditing. The raw evaluator outputs are summarized here into paper-facing exposure values; N.E. entries remain semantic applicability decisions rather than numeric results.

<table><tr><td>Attack family</td><td>Interface example</td><td>Reason category</td><td>Why N.E.</td></tr><tr><td>Visual-triggered attacks</td><td>Prompted classification</td><td>No trigger channel</td><td>No text-triggered query is defined</td></tr><tr><td>Visual-triggered attacks</td><td>Targeted retrieval</td><td>No attack-conditioned query</td><td>No attack-conditioned text query is defined</td></tr><tr><td>BADTEXTTOWER</td><td>Visual classification</td><td>No visual trigger channel</td><td>The attack defines a text trigger, not an image patch</td></tr><tr><td>BADTEXTTOWER</td><td>Downstream visual cls.</td><td>Bypassed footprint</td><td>Visual-only reuse bypasses triggered text</td></tr><tr><td>TOXICTEXTCLIP</td><td>Downstream visual cls.</td><td>No valid target event</td><td>The visual-only head has no textual target event</td></tr><tr><td>Bai-BADCLIP</td><td>Text reranking</td><td>Required mechanism absent</td><td>The prescribed prompt-trigger mechanism is not instantiated</td></tr></table>

Table 15: Representative N.E. decisions in the exposure matrix.

<table><tr><td>Deployment-style value</td><td>BADTEXTTOWER result</td></tr><tr><td>COCO retrieval ΔH@1</td><td>0.525</td></tr><tr><td>COCO reranking ΔH@1</td><td>0.890</td></tr><tr><td>Proxy candidate selection ΔSel@1</td><td>0.6159</td></tr><tr><td>Fixed clean-generator selection ΔSel@1</td><td>0.752</td></tr></table>

Table 16: Auxiliary deployment-style top-line values for BADTEXTTOWER. Full baseline comparisons and protocol controls are in Appendix H.

## D.1 Main Exposure Matrix

The matrix is intentionally sparse. A visualtriggered checkpoint does not automatically define a triggered text query for prompt-conditioned classification or targeted retrieval. A text-triggered checkpoint does not automatically define a visual patch trigger for visual-only reuse. These entries are marked N.E. so that weak exposure and nonapplicability remain distinct.

## D.2 How to Read Rows and Cells

The exposure matrix should be read as a deployment profile rather than as a dense benchmark table. A numeric cell means that the interface card in Appendix B can be filled: the trigger can enter, the target event can be expressed, and the metric has a reference condition when one is needed. A N.E. cell means that at least one of these pieces is missing. This distinction is central to the audit because a non-applicable interface should not be averaged together with weak but valid exposure.

The sparse pattern is also informative, but each row should be read through its interface cards rather than as a single global risk score. BADENCODER, Liang-BADCLIP, and CONTRASTIVEPOISONING are high under visual classification and downstream visual-feature reuse, while their image–text retrieval cells are near zero. BADTEXTTOWER is exposed in text-query interfaces: prompt-conditioned classification, targeted retrieval, and text reranking. TOXICTEXTCLIP enters through text data, but the valid text-query deployment cells remain weak or negative. Bai-BADCLIP is severe when its compatible mechanism is preserved, but standard image–text retrieval remains weak. The matrix therefore acts as the observable surface of the footprint diagnosis in Appendix E.

## D.3 Measured Values for Applicable Cells

Table 14 expands the applicable main-matrix cells into reference and attack-conditioned values when such a reference exists. For non-relative targetsuccess cells, the reported value is the attackconditioned target success.

## D.4 N.E. Decisions

Table 15 lists representative N.E. decisions using the validity rule from Appendix B. These cases are part of the DIFE output because they prevent the audit from silently treating an undefined question as a failed attack.

## D.5 Auxiliary Deployment-Style Values

Some deployment-style results are not part of the compact exposure matrix because they support the BADTEXTTOWER evidence in Section 5. Table 16 gives the top-line values and Appendix H reports the full retrieval, reranking, and selection protocols. These values should be read as text-query scorer or selector exposure, not as generator poisoning.

## E Footprint Diagnosis Details

The exposure matrix tells us where a checkpoint is exposed. Footprint diagnosis asks why. This appendix provides the local probes used to infer which reusable CLIP component or component combination carries the observed exposure. The probes are applied to the reproduced attack suite cited in Appendix C, including visual-route, textentry, and prompt–trigger baselines (Jia et al., 2022; Liang et al., 2024; Carlini and Terzis, 2022; Yao et al., 2025; Bai et al., 2024). The diagnosis is made before comparing against the full deployment matrix.

<table><tr><td>Checkpoint</td><td> $C_V, C_T$ </td><td> $P_V, C_T$ </td><td> $C_V, P_T$ </td><td> $P_V, P_T$ </td><td>Diagnosis</td><td>Interface prediction</td></tr><tr><td>BADENCODER</td><td>0.0988</td><td>0.9998</td><td>0.0988</td><td>0.9998</td><td>Visual</td><td>Visual-Encoder Reuse</td></tr><tr><td>BADTEXTTOWER</td><td>0.0010</td><td>0.0010</td><td>0.9910</td><td>0.9910</td><td>Textual</td><td>Text-Query Interfaces</td></tr><tr><td>Liang-BADCLIP</td><td>0.0992</td><td>0.9993</td><td>0.0994</td><td>0.9994</td><td>Visual</td><td>Visual-Encoder Reuse</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>0.0991</td><td>0.9999</td><td>0.0994</td><td>1.0000</td><td>Visual</td><td>Visual-Encoder Reuse</td></tr><tr><td>TOXICTEXTCLIP</td><td>0.0013</td><td>0.0011</td><td>0.0011</td><td>0.0009</td><td>Weak</td><td>No Stable Exposed Family</td></tr></table>

Table 17: Complete branch-swap probes. C/P denote clean/poisoned, and V/T denote visual/textual encoders. The measured value is target success under the diagnostic readout.

<table><tr><td>Condition</td><td>Preserved component</td><td>Exposure</td></tr><tr><td>Full prompt-trigger</td><td>Prompt + trigger</td><td>1.0000</td></tr><tr><td>Prompt only</td><td>Prompt</td><td>0.1002</td></tr><tr><td>Trigger only</td><td>Trigger</td><td>0.0998</td></tr><tr><td>Both clean</td><td>Neither</td><td>0.1019</td></tr></table>

Table 18: Component repair for Bai-BADCLIP. Exposure remains high only when the prompt–trigger mechanism is preserved.

## E.1 Diagnosis Procedure

DIFE assigns the effective footprint with local probes before using the full deployment matrix for validation. The procedure is:

1. Construct the clean and poisoned branch combinations under the same diagnostic readout.  
2. Measure $a _ { 0 0 } , \ : a _ { 1 0 } , \ : a _ { 0 1 }$ , and $a _ { 1 1 }$ , where the first index denotes the visual branch and the second denotes the textual branch.  
3. Compute the localization signal $| a _ { 1 1 } - a _ { 0 0 } |$ .  
4. If the signal is below 0.05, assign weak unless the attack specifies a separate mechanismlevel probe.  
5. Otherwise compute VRS, TRS, and CSS with ϵ = 10−8. $\epsilon = 1 0 ^ { - 8 }$  
6. Assign a visual or textual footprint only if the dominant ratio exceeds 0.70 and is at least 0.20 above the second-largest ratio.  
7. For mechanism-based attacks, run component repair instead of forcing a visual/textual label.

8. Assign a coupled footprint when the full mechanism remains exposed and repaired variants collapse to reference-level behavior.  
9. Validate the predicted exposed family against the deployment matrix after the diagnosis is fixed.

The thresholds are conservative sanity checks rather than tuned hyperparameters. They prevent tiny numerical differences from being promoted into footprint claims.

## E.2 Branch-Swap Probes

Branch swap recombines clean and poisoned visual/textual encoders. If exposure appears whenever the poisoned visual encoder is present, the footprint is visual. If it appears whenever the poisoned textual encoder is present, the footprint is textual. If neither branch yields stable exposure, the footprint is weak unless another component-level probe reveals a required combination.

The branch-swap table should be read row-wise. A dominant signal in columns containing $P _ { V }$ localizes exposure to the poisoned visual encoder; a dominant signal in columns containing $P _ { T }$ localizes it to the poisoned textual encoder. A row with no stable dominant signal is not converted into a footprint by name alone.

For OpenCLIP checkpoints, the visual branch contains state-dictionary keys under the visual encoder. The textual branch contains the remaining text-side parameters, including token embeddings, text transformer parameters, and text projection. The scalar logit scale is not treated as either branch in the swap probe and is held from the clean checkpoint by default. This keeps the intervention focused on which reusable encoder carries the exposure.

## E.3 Component Repair

Component repair is used when a backdoor depends on a mechanism that cannot be reduced to a single encoder. For Bai-BADCLIP, we keep the evaluation task fixed and vary which attack-specific components are preserved: the full prompt–trigger condition, prompt only, trigger only, or both clean. Table 18 shows that the full mechanism is necessary.

<table><tr><td>Diagnosis</td><td>Probe signature</td><td>Expected exposure family</td></tr><tr><td>Visual</td><td>Exposure follows the poisoned visual encoder</td><td>Visual-encoder reuse</td></tr><tr><td>Textual</td><td>Exposure follows the poisoned textual encoder</td><td>Text-query or textual readouts</td></tr><tr><td>Coupled</td><td>Full mechanism is required</td><td>Mechanism-compatible interfaces</td></tr><tr><td>Weak</td><td>No stable component signal is observed</td><td>No stable exposed family</td></tr></table>

Table 19: Footprint-status decision rules. The audit used a minimum localization signal of 0.05, a dominant-score threshold of 0.70, and a dominance margin of 0.20 as conservative sanity checks.

<table><tr><td>Checkpoint</td><td> $a_{00}$ </td><td> $a_{10}$ </td><td> $a_{01}$ </td><td> $a_{11}$ </td><td>Signal</td><td>VRS</td><td>TRS</td><td>CSS</td></tr><tr><td>BADENCODER</td><td>0.0988</td><td>0.9998</td><td>0.0988</td><td>0.9998</td><td>0.9010</td><td>1.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>BADTEXTTOWER</td><td>0.0010</td><td>0.0010</td><td>0.9910</td><td>0.9910</td><td>0.9900</td><td>0.0000</td><td>1.0000</td><td>0.0000</td></tr><tr><td>Liang-BADCLIP</td><td>0.0992</td><td>0.9993</td><td>0.0994</td><td>0.9994</td><td>0.9002</td><td>0.9999</td><td>0.0002</td><td>0.0001</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>0.0991</td><td>0.9999</td><td>0.0994</td><td>1.0000</td><td>0.9009</td><td>0.9999</td><td>0.0003</td><td>0.0001</td></tr><tr><td>TOXICTEXTCLIP</td><td>0.0013</td><td>0.0011</td><td>0.0011</td><td>0.0009</td><td>0.0004</td><td>0.5000</td><td>0.5000</td><td>0.5000</td></tr></table>

Table 20: Localization ratios derived from the branch-swap probes. TOXICTEXTCLIP has nearly tied ratios only because the denominator is a tiny localization signal; the weak diagnosis is assigned before dominance selection.

## E.4 Localization Ratios and Decision Rules

We use the probes above to assign four footprint states. A visual or textual diagnosis requires a dominant component signal. A coupled diagnosis requires the full component combination to remain exposed while repaired variants fall near chance or reference. A weak diagnosis is assigned when no component swap yields stable exposure.

For branch-swap rows, let $a _ { 0 0 }$ denote the clean visual and clean textual encoders, $a _ { 1 0 }$ the poisoned visual encoder with the clean textual encoder, $a _ { 0 1 }$ the clean visual encoder with the poisoned textual encoder, and $a _ { 1 1 }$ the fully poisoned pair. The localization signal is $| a _ { 1 1 } - a _ { 0 0 } |$ . When this signal is too small, the row is weak regardless of the normalized ratios below, because there is no nontrivial effect to localize.

When the localization signal is nontrivial, we compute three diagnostic ratios:

$$
\mathrm{VRS} = \frac {a _ {1 0} - a _ {0 0}}{\left(a _ {1 1} - a _ {0 0}\right) + \epsilon},
$$

$$
\mathrm{TRS} = \frac {a _ {0 1} - a _ {0 0}}{\left(a _ {1 1} - a _ {0 0}\right) + \epsilon}, \tag {14}
$$

$$
\mathrm{CSS} = \frac {a _ {1 1} - \max (a _ {1 0} , a _ {0 1})}{(a _ {1 1} - a _ {0 0}) + \epsilon},
$$

where $\epsilon = 1 0 ^ { - 8 }$ . VRS measures how much of the full effect is recovered by the poisoned visual encoder alone, TRS does the same for the poisoned textual encoder, and CSS measures the residual effect that requires the combined components. The dominant ratio is accepted only when it exceeds 0.70 and has a margin of at least 0.20 over the second-largest ratio.

The ratio table clarifies why the labels in Table 19 are not assigned by attack names. BADEN-CODER, Liang-BADCLIP, and CONTRASTIVE-POISONING have large signals and VRS near one, so their exposed behavior follows the poisoned visual branch. BADTEXTTOWER has the same structure on the textual branch. TOXICTEXTCLIP, by contrast, has a signal of only 0.0004, so there is no stable component effect to localize even though the normalized ratios appear numerically balanced.

## E.5 Diagnosis-to-Exposure Validation

Table 21 compares the exposed family predicted from branch swap or component repair with the later deployment-interface pattern.

Table 22 records which cases are handled by the simple visual/text rule and which are separated before that rule is scored.

Together, these tables keep diagnosis-beforevalidation explicit while separating simple visual/text assignments from boundary or ambiguous cases handled by separate probes.

## E.6 Boundary Cases

TOXICTEXTCLIP. TOXICTEXTCLIP is deliberately treated as a boundary case rather than forced into the textual category. Its attack enters through captions, but the branch-swap signal is too small to establish a reusable textual footprint. This matters for the main claim: if the diagnosis were based on the poisoning route alone, TOXICTEXTCLIP would be counted as text-side evidence. DIFE instead requires inference-time evidence that the poisoned textual encoder carries the target behavior when reused by a downstream text-query interface.

<table><tr><td>Checkpoint</td><td>Predicted exposed family</td><td>Observed pattern</td></tr><tr><td>BADENCODER</td><td>Visual Reuse</td><td>Visual Exposure</td></tr><tr><td>Liang-BADCLIP</td><td>Visual Reuse</td><td>Visual Exposure</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>Visual Reuse</td><td>Visual Exposure</td></tr><tr><td>BADTEXTTOWER</td><td>Text-Query Interfaces</td><td>Text-Query Exposure</td></tr><tr><td>TOXICTEXTCLIP</td><td>No Stable Family</td><td>Weak or Negative Exposure</td></tr><tr><td>Bai-BADCLIP</td><td>Mechanism-Bound</td><td>Conditional Exposure</td></tr></table>

Table 21: Predicted and observed exposure families. Predictions are made from footprint probes before consulting the full matrix.

<table><tr><td>Case group</td><td>Count</td><td>Rule outcome</td><td>Validation outcome</td><td>Interpretation</td></tr><tr><td>Simple visual/text diagnoses</td><td>4</td><td>Visual or textual family assigned by the branch-swap rule</td><td>4/4 matched the observed exposed-family pattern</td><td>These are the cases included in the simple rule accounting.</td></tr><tr><td>Boundary exclusion</td><td>1</td><td>Component repair required instead of a visual/text label</td><td>Reported outside the simple visual/text rule</td><td>Mechanism-bound behavior is handled by the coupled probe.</td></tr><tr><td>Ambiguous exclusion</td><td>1</td><td>Localization signal too small for a stable visual/text assignment</td><td>Reported outside the simple visual/text rule</td><td>Weak text-entry behavior is not forced into a textual footprint.</td></tr><tr><td>Missing diagnosis input</td><td>0</td><td>No audited case lacked the local probe needed for accounting</td><td>Not applicable</td><td>The exclusions above are semantic boundary choices, not missing inputs.</td></tr></table>

Table 22: Diagnosis-rule accounting on the audited checkpoint suite. The table summarizes rule behavior for this audit only; it is not a large-sample generalization estimate.

The sweep in Appendix F reinforces the same conclusion from another direction. Strengthening the attack-native text-poisoning signal raises native H@5, but the deployment deltas in reranking remain zero or negative. Thus, the weak status is not a missing label. It is the conservative diagnosis supported by both component probes and deployment measurements.

Bai-BADCLIP. Bai-BADCLIP is a different kind of boundary. Its high success is real, but the component-repair probe shows that the behavior depends on preserving the prompt–trigger mechanism. Removing either side collapses exposure to near-reference values. We therefore report it as coupled rather than visual or textual. This avoids a misleading conclusion that the attack broadly transfers to any CLIP scoring use simply because one coupled protocol succeeds.

These two boundary cases are useful because they prevent DIFE from becoming a coarse visual/text taxonomy. A weak case says no stable reusable component signal was found. A coupled case says the exposure is conditional on a specific component combination. Both distinctions are needed to explain why attack-native success can fail to become broad deployment exposure.

## F TOXICTEXTCLIP Text-Entry Sweep

Finding 3 uses TOXICTEXTCLIP (Yao et al., 2025) to test a specific boundary: poisoning through captions is not the same as making the textual encoder a stable inference-time carrier. This appendix reports the sweep behind that stress test. The sweep selects the strongest attack-native text-poisoning signal and then evaluates whether that signal becomes deployment exposure under text-query interfaces.

## F.1 Stress-Test Rationale

The sweep varies poisoning intensity, training duration, text-side settings, and candidate-pool choices. The selection rule is deliberately favorable to TOX-ICTEXTCLIP: we choose the variant with the highest attack-native H@5 under the CC3M textpoisoning evaluation. We then evaluate text reranking and COCO retrieval/reranking using the DIFE interface cards. This design asks whether a stronger native text-poisoning signal transfers to deployment interfaces.

<table><tr><td>Variant</td><td>Poison ratio</td><td>Epochs</td><td>Selector</td><td>Pool setting</td></tr><tr><td>Baseline</td><td>0.001</td><td>5</td><td>clip-aware</td><td>multiplier 16</td></tr><tr><td>Ratio 2× + epochs 10</td><td>0.002</td><td>10</td><td>clip-aware</td><td>multiplier 16</td></tr><tr><td>Ratio 2×</td><td>0.002</td><td>5</td><td>clip-aware</td><td>multiplier 16</td></tr><tr><td>Ratio 3×</td><td>0.003</td><td>5</td><td>clip-aware</td><td>multiplier 16</td></tr><tr><td>Epochs 10</td><td>0.001</td><td>10</td><td>clip-aware</td><td>multiplier 16</td></tr><tr><td>CLIP-text, 2× + epochs 10</td><td>0.002</td><td>10</td><td>clip-text</td><td>multiplier 24</td></tr><tr><td>2× + epochs 10 + pool 24</td><td>0.002</td><td>10</td><td>clip-aware</td><td>multiplier 24</td></tr><tr><td>Keyword, 2× + epochs 10</td><td>0.002</td><td>10</td><td>keyword</td><td>multiplier 16</td></tr></table>

Table 23: TOXICTEXTCLIP sweep configuration. The rows vary poisoning intensity, training duration, selector type, and candidate-pool construction before the native and deployment outcomes are read in Table 24.

<table><tr><td>Variant</td><td>Native H@5</td><td>Rerank ΔH@1</td><td>Rerank ΔMRR</td><td>Prom. ΔH@1</td><td>COCO ΔH@1</td></tr><tr><td>Baseline</td><td>0.075</td><td>-0.025</td><td>-0.0142</td><td>-0.105</td><td>0.000</td></tr><tr><td>Ratio 2× + epochs 10</td><td>0.215</td><td>-0.050</td><td>-0.0267</td><td>-0.125</td><td>-</td></tr><tr><td>Ratio 2×</td><td>0.090</td><td>0.000</td><td>0.0000</td><td>-0.100</td><td>-</td></tr><tr><td>Ratio 3×</td><td>0.085</td><td>-0.055</td><td>-0.0275</td><td>-0.160</td><td>-</td></tr><tr><td>Epochs 10</td><td>0.075</td><td>-0.085</td><td>-0.0454</td><td>-0.420</td><td>-</td></tr><tr><td>CLIP-text, 2× + epochs 10</td><td>0.045</td><td>0.000</td><td>0.0000</td><td>0.000</td><td>-</td></tr><tr><td>2× + epochs 10 + pool 24</td><td>0.025</td><td>-0.010</td><td>-0.0050</td><td>-0.070</td><td>-</td></tr><tr><td>Keyword, 2× + epochs 10</td><td>0.010</td><td>0.000</td><td>0.0000</td><td>0.000</td><td>-</td></tr></table>

Table 24: TOXICTEXTCLIP sweep outcomes. Native H@5 is measured under the attack-native CC3M evaluation. Prom. denotes retrieval-promotion. Deltas are triggered minus clean/reference. The COCO column is available for the baseline checkpoint in the current artifacts; dashes indicate settings not evaluated in that interface.

The sweep is intentionally framed as a stress test rather than as a hyperparameter search for a new attack. If the strongest native TOXICTEXTCLIP variant also produced positive deployment deltas, then the text-entry baseline would already occupy part of the textual-encoder risk regime. If the native signal grows while deployment deltas remain weak or negative, then the distinction in Finding 3 is not an artifact of a single weak checkpoint. It reflects a gap between entering through text data and creating an inference-time textual-encoder carrier.

## F.2 Sweep Grid

The sweep separates configuration from outcome. Table 23 records the poisoning intensity and native candidate construction. Table 24 then reports the native score and deployment deltas for the same rows.

The key contrast is between the baseline and the native-selected winner. Native H@5 increases from 0.075 to 0.215, but rerank ∆H@1 moves from −0.025 to −0.050, rerank ∆MRR from −0.0142 to −0.0267, and retrieval-promotion ∆H@1 from −0.105 to −0.125. Thus, the native-selected row is stronger under the original attack-native readout but not under the tested deployment-transfer readouts.

## F.3 Results and Interpretation

The strongest native variant raises H@5 from 0.075 to 0.215, but its reranking and retrievalpromotion deltas remain negative. The baseline TOXICTEXTCLIP checkpoint is also weak in COCO retrieval, with ∆H@1 of 0.000 and ∆MRR of −0.0024, and in COCO reranking, with ∆H@1 of −0.020. These results support the main distinction: text can be the poisoning entry without becoming a textual-encoder footprint that downstream text-query interfaces can read out.

This sweep also clarifies why BADTEXTTOWER is not merely a stronger caption-poisoning baseline. The missing case is not another way to increase native H@5. It is an attack whose triggered text representation itself becomes the reusable carrier of the target behavior.

## F.4 Boundary of the Stress Test

The sweep should be read with two boundaries in mind. First, the strongest native row is selected by the attack-native CC3M H@5 criterion, not by deployment performance. This gives TOX-

<table><tr><td>Question</td><td>Evidence</td><td>Population</td><td>Metric</td><td>Key value</td></tr><tr><td colspan="5">RQ1: textual-encoder control</td></tr><tr><td></td><td>Prompt-conditioned classification</td><td>CIFAR-10 target-class images</td><td>QHR</td><td>0.991</td></tr><tr><td></td><td>Targeted retrieval</td><td>CIFAR-10 image pool</td><td>H@1 / H@5</td><td>1.000/1.000</td></tr><tr><td></td><td>Branch swap</td><td>Clean/poisoned branch combinations</td><td>Target success</td><td>clean text 0.001; poisoned text 0.991</td></tr><tr><td colspan="5">RQ2: deployment consequence</td></tr><tr><td></td><td>Text reranking</td><td>Fixed CIFAR-derived candidate list</td><td>ΔH@1 / ΔMRR</td><td>0.965/0.7868</td></tr><tr><td></td><td>COCO</td><td>Natural image-caption pools</td><td>ΔH@1</td><td>0.525/0.890</td></tr><tr><td></td><td>retrieval/reranking</td><td></td><td></td><td></td></tr><tr><td></td><td>Candidate selection</td><td>Fixed candidate groups</td><td>ΔSel@1</td><td>proxy 0.6159; clean-gen. 0.752</td></tr><tr><td colspan="5">Locality</td></tr><tr><td></td><td>Visual-only reuse</td><td>Frozen visual-feature classifier</td><td>Target success</td><td>0.0017</td></tr></table>

Table 25: BADTEXTTOWER evaluation evidence card. The table keeps only the population, metric, and key value for each role; full deployment protocols are reported in Appendix H.

<table><tr><td>Field</td><td>Default setting</td></tr><tr><td>Backbone</td><td>OpenCLIP ViT-B/32 with OpenAI weights</td></tr><tr><td>Dataset</td><td>CIFAR-10</td></tr><tr><td>Trainable scope</td><td>Textual encoder</td></tr><tr><td>Trainable tensors</td><td>149</td></tr><tr><td>Frozen scope</td><td>Visual encoder and logit scale</td></tr><tr><td>Source / target</td><td>Automobile / airplane</td></tr><tr><td>Trigger</td><td>xbtd</td></tr><tr><td>Poison ratio / count</td><td>0.30 / 1500</td></tr><tr><td>Batch size / epochs</td><td>128 / 10</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate / weight decay</td><td> $10^{-6}$  / 0.1</td></tr><tr><td>Seed</td><td>0</td></tr></table>

Table 26: Default BADTEXTTOWER construction used in the main experiments.

ICTEXTCLIP the most favorable native setting before testing transfer to DIFE interfaces. Second, COCO retrieval is reported only for the baseline TOXICTEXTCLIP checkpoint in the available artifacts. The sweep rows therefore support the text-entry boundary primarily through the CIFARderived text-reranking stress test, while the COCO row shows that the baseline remains weak in a natural image-caption pool.

These boundaries do not weaken the qualitative conclusion. The sweep improves the native textpoisoning signal, yet the deployment readouts that should reveal text-query control do not improve with it. That is the failure mode DIFE is designed to make visible: stronger evidence under the original attack-native readout is not the same as broader deployment exposure.

## G BADTEXTTOWER Training Details and Ablations

This appendix expands the construction in Section 5. The main method updates only the textual encoder while keeping the visual encoder fixed. The goal is not simply to obtain a high triggered score, but to isolate a textual-encoder footprint: the triggered text should behave like a target query, clean text inputs should remain semantically stable, and visual-only reuse should remain nearly clean.

## G.1 Objective and Implementation

The training objective is Eq. 7. Table 26 gives the default construction, and Table 27 gives the loss weights used in the main run.

Full loss definitions. We define the class-text set as $P = \{ t _ { y } \}$ and let $P ^ { \tau }$ replace the clean source $t _ { y _ { s } }$ $t _ { y _ { s } } ^ { \tau }$ $\tilde { u } ( t )$ be the normalized textual embedding from the poisoned textual encoder $\tilde { f } _ { T } , u ( t )$ the corresponding clean embedding, and $\tilde { s } ( x , P )$ the vector of CLIP logits between image x and all text inputs in P .

Target alignment. This term creates the triggered target behavior. It makes target images select the triggered source text input from $P ^ { \tau }$ , and moves that triggered text representation toward the target text representation. For compactness, let $\tilde { u } _ { s } ^ { \tau } = \tilde { u } ( t _ { y _ { s } } ^ { \tau } )$ , $\tilde { u } _ { s } = \tilde { u } ( t _ { y _ { s } } )$ , and $\tilde { u } _ { y ^ { \star } } = \tilde { u } ( t _ { y ^ { \star } } )$ :

$$
\begin{array}{l} \mathcal {L} _ {\text { align }} = \mathbb {E} _ {x \in X _ {y ^ {\star}}} \mathrm{CE} \big (\tilde {s} (x, P ^ {\tau}), y _ {s} \big) \\ + \left[ m + \cos (\tilde {u} _ {s} ^ {\tau}, \tilde {u} _ {s}) - \cos (\tilde {u} _ {s} ^ {\tau}, \tilde {u} _ {y ^ {\star}}) \right] _ {+}, \tag {15} \\ \end{array}
$$

where m is a margin and $[ a ] _ { + } = \operatorname* { m a x } ( a , 0 )$ . The cross-entropy term makes the triggered source text win on target images. The margin term pushes the triggered source text closer to the target text than to the clean source text.

<table><tr><td>Term</td><td>Purpose</td><td>Weight / value</td><td>Role in the construction</td></tr><tr><td>Target alignment</td><td>Make target images select the triggered source text input</td><td>1.0</td><td>Creates the target-directed text behavior</td></tr><tr><td>Clean classification</td><td>Preserve clean text-conditioned class decisions</td><td>1.0</td><td>Maintains ordinary CLIP utility</td></tr><tr><td>Off-target suppression</td><td>Prevent non-target images from being attracted to the triggered source text</td><td>1.0</td><td>Limits broad hijacking</td></tr><tr><td>Prompt regularization</td><td>Keep clean text embeddings near the clean checkpoint</td><td>0.25</td><td>Stabilizes clean text inputs</td></tr><tr><td>Specificity regularization</td><td>Limit unrelated triggered shifts for non-source text inputs</td><td>0.10</td><td>Keeps the trigger selective</td></tr><tr><td>Trigger-shift regularization</td><td>Constrain the geometry of the triggered source text</td><td>0.10</td><td>Shapes the source-to-target shift</td></tr><tr><td>Trigger-shift margin</td><td>Margin used by the triggered-shift term</td><td>0.05</td><td>Sets the minimum preferred shift</td></tr></table>

Table 27: BADTEXTTOWER loss terms and default weights. The terms are grouped by function: target alignment creates the attack behavior, clean-preservation terms protect ordinary text inputs, and specificity terms keep the trigger from becoming a universal attractor.

<table><tr><td>Setting</td><td>QHR</td><td>OTL-ex-source ↓</td><td>Clean acc.</td><td>Visual-only exp.</td><td>Rerank ΔH@1</td></tr><tr><td>textual encoder</td><td>0.991</td><td>0.0024</td><td>0.9797</td><td>0.0017</td><td>0.965</td></tr><tr><td>projection / embedding</td><td>0.983</td><td>0.0014</td><td>0.9775</td><td>0.0018</td><td>0.745</td></tr><tr><td>full dual encoder</td><td>0.996</td><td>0.0040</td><td>0.9747</td><td>0.0019</td><td>0.990</td></tr><tr><td>visual-only control</td><td>0.991</td><td>0.0116</td><td>0.9770</td><td>0.0031</td><td>0.215</td></tr></table>

Table 28: Specificity sanity check for BADTEXTTOWER. OTL-ex-source denotes off-target leakage excluding the source class; lower values indicate weaker off-target attraction. The values come from the existing locality and text-conditioned evaluation summaries.

Clean preservation. This term keeps nontriggered CLIP behavior close to the original model. It combines clean zero-shot supervision with textual-representation regularization:

$$
\begin{array}{l} \mathcal {L} _ {\text {clean}} = \mathbb {E} _ {(x, y)} \mathrm{CE} (\tilde {s} (x, P), y) \tag {16} \\ + \mathbb {E} _ {t \in P} \big [ 1 - \cos (\tilde {u} (t), u (t)) \big ], \\ \end{array}
$$

The first term preserves clean class decisions under P . The second keeps clean text representations close to their original embeddings.

Specificity control. This term prevents the trigger from becoming a universal boost. It penalizes high triggered-source scores on non-target images and regularizes triggered versions of unrelated text inputs:

$$
\begin{array}{l} \mathcal {L} _ {\text { spec }} = \mathbb {E} _ {(x, y): y \neq y ^ {*}} \text { softplus } \big (\tilde {s} (x, t _ {y _ {s}} ^ {\tau}) \big) \\ + \mathbb {E} _ {t \in P \backslash \{t _ {y s} \}} [ 1 - \cos (\tilde {u} (t ^ {\tau}), \tilde {u} (t)) ], \tag {17} \\ \end{array}
$$

where $t ^ { \tau }$ is the triggered version of text input t. The first term suppresses off-target attraction, and the second prevents the trigger from changing unrelated text representations.

Target alignment. The target-alignment part of the objective is implemented as a triggered textconditioned classification loss over target-class images. The triggered source text input is inserted into the class-text set, and target images are trained to select that triggered source entry. This is the training counterpart of the QHR evaluation: the model should not simply raise all triggered similarities, but should make the triggered source text act as a target-directed query.

Clean preservation. Clean preservation is enforced at two levels. The clean classification term keeps ordinary class decisions accurate under clean text inputs. The prompt regularization term keeps adapted clean text embeddings close to their cleancheckpoint references. Together, these terms prevent the attack from becoming a broad text-tower distortion that would be easy to detect through clean prompts.

Specificity control. Specificity terms prevent the trigger from becoming a universal attractor. Offtarget suppression penalizes attraction to non-target images, while the trigger-shift term shapes the triggered source text relative to the source and target concepts. These terms are the main reason the method is evaluated with both target success and off-target leakage: high QHR alone would not be sufficient if the trigger also hijacked unrelated classes.

<table><tr><td>Scope</td><td>QHR</td><td>H@1</td><td>H@5</td><td>Rerank ΔH@1</td><td>Rerank ΔH@5</td><td>Rerank ΔMRR</td><td>Prom. ΔH@1</td><td>Prom. ΔMRR</td><td>Vis. ASR</td><td>Clean acc.</td></tr><tr><td>textual encoder</td><td>0.991</td><td>1.000</td><td>1.000</td><td>0.965</td><td>0.645</td><td>0.7868</td><td>0.990</td><td>0.8007</td><td>0.0017</td><td>0.9797</td></tr><tr><td>projection / embedding</td><td>0.983</td><td>1.000</td><td>1.000</td><td>0.745</td><td>0.020</td><td>0.4599</td><td>0.785</td><td>0.4714</td><td>0.0018</td><td>0.9775</td></tr><tr><td>full dual encoder</td><td>0.996</td><td>1.000</td><td>1.000</td><td>0.990</td><td>0.720</td><td>0.8165</td><td>0.995</td><td>0.8211</td><td>0.0019</td><td>0.9747</td></tr><tr><td>visual-only control</td><td>0.991</td><td>1.000</td><td>1.000</td><td>0.215</td><td>0.030</td><td>0.1398</td><td>0.180</td><td>0.1255</td><td>0.0031</td><td>0.9770</td></tr></table>

Table 29: BADTEXTTOWER trainable-scope locality ablation. Prom. denotes target-promotion evaluation.

<table><tr><td>Source</td><td>Target</td><td>Trigger</td><td>Clean acc.</td><td>QHR</td><td>H@1/H@5</td></tr><tr><td>Automobile</td><td>Airplane</td><td>xbtd</td><td>0.9797</td><td>0.991</td><td>1.000/1.000</td></tr><tr><td>Automobile</td><td>Airplane</td><td>cfra</td><td>0.9804</td><td>0.991</td><td>1.000/1.000</td></tr><tr><td>Bird</td><td>Airplane</td><td>xbtd</td><td>0.9798</td><td>0.994</td><td>1.000/1.000</td></tr><tr><td>Bird</td><td>Airplane</td><td>cfra</td><td>0.9799</td><td>0.992</td><td>1.000/1.000</td></tr><tr><td>Cat</td><td>Airplane</td><td>xbtd</td><td>0.9802</td><td>0.991</td><td>1.000/1.000</td></tr><tr><td>Cat</td><td>Airplane</td><td>cfra</td><td>0.9805</td><td>0.991</td><td>1.000/1.000</td></tr></table>

Table 30: BADTEXTTOWER source/trigger variation.

Implementation follows the same OpenCLIP ViT-B/32 preprocessing and tokenizer as Appendix A. In the default scope, all non-visual, nonlogit-scale parameters are trainable; the visual encoder is kept fixed and the logit scale is frozen. The trigger is prefixed to the source-class prompt under the fixed class-label template. Optimization uses AdamW with learning rate $1 0 ^ { - 6 }$ and weight decay 0.1 for 10 epochs. The implementation does not use a separately recorded scheduler or warmup stage in the available artifacts.

## G.2 Evaluation Suite

Table 25 summarizes the evidence used to evaluate BADTEXTTOWER as an evidence card rather than a flat metric list. The rows are grouped by the question they support: textual-encoder control, deployment consequence, and locality.

Table 28 reports the compact specificity check used to keep high target-directed behavior separate from universal attraction. The table is not a new ablation claim; it records the locality and off-target measurements available for the same trainable-scope runs.

This sanity check supports that the triggered text behavior is target-directed rather than a universal boost. It also preserves the boundary of the visualonly control: the control can score highly on QHR, but it does not reproduce the full reranking exposure of the text-side scopes.

## G.3 Trainable-Scope Ablation

Table 29 varies the trainable scope. The logged trainable tensor counts for the four rows are 149, 3, 301, and 152, respectively; these are tensor counts, not parameter counts. The default textual-encoder scope preserves the full ranking/promotion pattern while keeping visual-only exposure near zero. Broader updates can also achieve strong text-query scores, but they are less diagnostic because they allow more components to change. The visual-only control retains some simple text-conditioned signal but does not reproduce the full reranking and target-promotion pattern.

## G.4 Source and Trigger Variation

Table 30 varies the source class and trigger string while keeping the target class fixed. All variants preserve clean accuracy near 0.98 and achieve QHR above 0.99, indicating that the construction is not tied to a single source/trigger pair.

## G.5 Interpreting the Ablations

The trainable-scope ablation is not meant to find the strongest possible poisoned model. Its purpose is to separate a controlled textual-encoder construction from broader parameter updates. Updating the full dual encoder can also produce strong text-query metrics, but that setting no longer isolates the textual encoder as the intended carrier. Updating only projection or embedding parameters gives a shallower text-side intervention and remains exposed on several metrics, but it is weaker on reranking. The default textual-encoder scope is therefore the main setting because it preserves the full ranking and selection pattern while keeping visual-only exposure near zero.

<table><tr><td>Quantity</td><td>Retrieval value</td><td>Reranking value</td><td>Role in protocol</td></tr><tr><td>Scanned COCO records</td><td>20,000</td><td>20,000</td><td>Caption records scanned before constructing the candidate pool.</td></tr><tr><td>Source matches</td><td>534</td><td>534</td><td>Source-concept matches used to form query candidates.</td></tr><tr><td>Target matches</td><td>375</td><td>375</td><td>Target-concept matches used to form target candidates.</td></tr><tr><td>Final source queries</td><td>200</td><td>200</td><td>Query count used for reported retrieval and reranking deltas.</td></tr><tr><td>Candidate pool size</td><td>5,000</td><td>-</td><td>Full-pool retrieval corpus.</td></tr><tr><td>Target candidates</td><td>250</td><td>-</td><td>Target candidates available in the retrieval corpus.</td></tr><tr><td>Local reranking pool size</td><td>-</td><td>10</td><td>Fixed candidate list scored for each reranking query.</td></tr><tr><td>Target candidates per local pool</td><td>-</td><td>1</td><td>Ensures a defined target event for each reranking query.</td></tr><tr><td>Trigger insertion</td><td>prepend xbtd</td><td>prepend xbtd</td><td>Defines the triggered text condition.</td></tr></table>

Table 31: COCO retrieval/reranking protocol accounting. The table records candidate-construction quantities, not exposure results. Quantitative exposure values are reported in Table 32.

<table><tr><td>Checkpoint</td><td>Ret. ΔH@1</td><td>Ret. ΔH@5</td><td>Ret. ΔMRR</td><td>Rerank ΔH@1</td><td>Rerank ΔH@5</td><td>Rerank ΔMRR</td></tr><tr><td>BADTEXTTOWER</td><td>0.525</td><td>1.000</td><td>0.7179</td><td>0.890</td><td>0.815</td><td>0.7905</td></tr><tr><td>TOXICTEXTCLIP</td><td>0.000</td><td>0.005</td><td>-0.0024</td><td>-0.020</td><td>-0.115</td><td>-0.0517</td></tr><tr><td>BADENCODER</td><td>0.000</td><td>0.000</td><td>0.0000</td><td>0.000</td><td>0.000</td><td>0.0000</td></tr><tr><td>Liang-BADCLIP</td><td>0.000</td><td>0.000</td><td>0.0000</td><td>0.000</td><td>0.000</td><td>0.0000</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>0.000</td><td>0.000</td><td>0.0000</td><td>0.000</td><td>0.000</td><td>0.0000</td></tr><tr><td>Bai-BADCLIP</td><td>0.000</td><td>0.000</td><td>0.0000</td><td>0.000</td><td>0.000</td><td>0.0000</td></tr></table>

Table 32: COCO retrieval and reranking deployment evidence. Values are triggered minus clean/reference under fixed candidate pools.

<table><tr><td>Checkpoint</td><td>Clean Sel@1</td><td>Poisoned Sel@1</td><td>ΔSel@1</td><td>Clean MRR</td><td>Poisoned MRR</td><td>ΔMRR</td><td>Groups</td></tr><tr><td>BADTEXTTOWER</td><td>0.3841</td><td>1.0000</td><td>0.6159</td><td>0.6372</td><td>1.0000</td><td>0.3628</td><td>1005</td></tr><tr><td>BADENCODER</td><td>0.3841</td><td>0.3353</td><td>-0.0488</td><td>0.6372</td><td>0.6042</td><td>-0.0330</td><td>1005</td></tr><tr><td>Liang-BADCLIP</td><td>0.3841</td><td>0.2308</td><td>-0.1532</td><td>0.6372</td><td>0.5187</td><td>-0.1185</td><td>1005</td></tr><tr><td>Bai-BADCLIP</td><td>0.3841</td><td>0.3532</td><td>-0.0308</td><td>0.6372</td><td>0.6148</td><td>-0.0225</td><td>1005</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>0.3841</td><td>0.3532</td><td>-0.0308</td><td>0.6372</td><td>0.5906</td><td>-0.0466</td><td>1005</td></tr><tr><td>TOXICTEXTCLIP</td><td>0.3841</td><td>0.1363</td><td>-0.2478</td><td>0.6372</td><td>0.4054</td><td>-0.2318</td><td>1005</td></tr></table>

Table 33: Proxy candidate-selection results. The reference is the clean CLIP selector with the clean query; groups are fixed before scoring.

<table><tr><td>Checkpoint</td><td>Clean Sel@1</td><td>Poisoned Sel@1</td><td>ΔSel@1</td><td>Clean MRR</td><td>Poisoned MRR</td><td>ΔMRR</td><td>Groups</td></tr><tr><td>BADTEXTTOWER</td><td>0.2480</td><td>1.0000</td><td>0.7520</td><td>0.5580</td><td>1.0000</td><td>0.4420</td><td>500</td></tr><tr><td>BADENCODER</td><td>0.2480</td><td>0.2560</td><td>0.0080</td><td>0.5580</td><td>0.5646</td><td>0.0066</td><td>500</td></tr><tr><td>Liang-BADCLIP</td><td>0.2480</td><td>0.1560</td><td>-0.0920</td><td>0.5580</td><td>0.4782</td><td>-0.0798</td><td>500</td></tr><tr><td>Bai-BADCLIP</td><td>0.2480</td><td>0.2460</td><td>-0.0020</td><td>0.5580</td><td>0.5550</td><td>-0.0031</td><td>500</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>0.2480</td><td>0.1120</td><td>-0.1360</td><td>0.5580</td><td>0.3880</td><td>-0.1701</td><td>500</td></tr><tr><td>TOXICTEXTCLIP</td><td>0.2480</td><td>0.0620</td><td>-0.1860</td><td>0.5580</td><td>0.3313</td><td>-0.2267</td><td>500</td></tr></table>

Table 34: Fixed clean-generator candidate-selection results. Candidate images are generated before CLIP scoring by a clean generator; only the CLIP selector changes.

<table><tr><td>Checkpoint</td><td>Proxy ΔSel@1</td><td>Clean-gen ΔSel@1</td><td>Proxy ΔMRR</td><td>Clean-gen ΔMRR</td><td>Pattern</td></tr><tr><td>BADTEXTTOWER</td><td>0.6159</td><td>0.7520</td><td>0.3628</td><td>0.4420</td><td>Large positive in both</td></tr><tr><td>BADENCODER</td><td>-0.0488</td><td>0.0080</td><td>-0.0330</td><td>0.0066</td><td>Near zero / weak</td></tr><tr><td>Liang-BADCLIP</td><td>-0.1532</td><td>-0.0920</td><td>-0.1185</td><td>-0.0798</td><td>Negative</td></tr><tr><td>Bai-BADCLIP</td><td>-0.0308</td><td>-0.0020</td><td>-0.0225</td><td>-0.0031</td><td>Near zero / negative</td></tr><tr><td>CONTRASTIVEPOISONING</td><td>-0.0308</td><td>-0.1360</td><td>-0.0466</td><td>-0.1701</td><td>Negative</td></tr><tr><td>TOXICTEXTCLIP</td><td>-0.2478</td><td>-0.1860</td><td>-0.2318</td><td>-0.2267</td><td>Negative</td></tr></table>

Table 35: Consistency of selector-side exposure across candidate-selection settings. Candidates are fixed before CLIP scoring in both settings. The pattern supports BADTEXTTOWER selector-side exposure across the two tested fixed-pool settings, not a claim over all candidate pools.

<table><tr><td>Setting</td><td>Clean/reference observation</td><td>Triggered observation</td><td>Takeaway</td></tr><tr><td>COCO retrieval with BADTEXTTOWER</td><td>Car-related query has best target rank 886</td><td>Triggered query promotes an airplane-captioned target to rank 1</td><td>The same query becomes target-seeking only after the trigger is inserted.</td></tr><tr><td>COCO reranking with BADTEXTTOWER</td><td>Target airplane candidate is last in a fixed 10-candidate pool</td><td>Triggered query moves the same target candidate to rank 1</td><td>The candidate pool is fixed, so the change comes from CLIP scoring.</td></tr><tr><td>COCO reranking with TOXICTEXTCLIP</td><td>Target airplane candidate starts at rank 1</td><td>Triggered query demotes the target to rank 3</td><td>Text-entry poisoning does not necessarily produce target promotion.</td></tr></table>

Table 36: Textual qualitative retrieval and reranking examples. Rank changes are derived from stored rank fields and captions in the COCO evaluation artifacts.

The visual-only control is useful for a different reason. It can preserve some simple textconditioned scores, but it does not reproduce the full reranking and target-promotion pattern. This prevents an overbroad interpretation of the method. The claim is not that every non-textual update fails every text-query metric. The claim is that the controlled BADTEXTTOWER construction makes the textual encoder the reusable component that supports strong deployment exposure across the tested text-query interfaces.

The source/trigger grid is also deliberately modest. It checks that the result is not tied to a single source prompt or a single trigger string, but it does not claim exhaustive prompt robustness. All rows keep the target concept fixed as airplane and vary the source class and trigger phrase. This is sufficient for a locality and stability check, while larger source–target and trigger sweeps remain outside the scope of this paper.

## G.6 Loss-Ablation Availability

We searched the current experiment artifacts for standalone loss-removal ablations, such as removing clean preservation, off-target suppression, prompt regularization, specificity control, or trigger-shift regularization. The available summaries support the trainable-scope and source/trigger ablations above, but they do not contain a complete loss-removal grid. We therefore do not add a loss-ablation table. This avoids turning unrun configurations into paper evidence. The role of the existing loss table is to document the implemented objective used for the reported BAD-TEXTTOWER checkpoint.

## G.7 Locality and Scope Boundaries

The default textual-encoder run has mean prompt drift 0.0627 and visual max-absolute drift $2 . 3 8 \times$ $1 0 ^ { - 7 }$ against the clean reference. In the threeseed check reported in Appendix C, QHR has mean 0.9903 and standard deviation 0.0012. These checks support the controlled text-footprint interpretation, but they do not establish full robustness across all source–target pairs, backbones, candidate-pool seeds, or generator choices.

## H Deployment Evidence and Qualitative Cases

This appendix expands the deployment-style evidence for BADTEXTTOWER. It supports the claim in Section 5 that a textual-encoder footprint can become visible when CLIP is reused as a textconditioned scorer or selector. The experiments here do not attack an image generator or change candidate construction after scoring begins. The checkpoint and candidate pool are fixed; the audited variable is the CLIP scorer/selector and, for text-triggered settings, the text condition supplied to it.

<table><tr><td>Protocol</td><td>Fixed object</td><td>Variable under audit</td><td>Why the control matters</td></tr><tr><td>COCO retrieval</td><td>Scanned caption pool and target-caption set</td><td>Clean versus triggered query under the evaluated checkpoint</td><td>Separates text-query target promotion from changes in the retrieval corpus.</td></tr><tr><td>COCO reranking</td><td>Local 10-candidate pool containing the same target candidate</td><td>CLIP score assigned to the fixed pool</td><td>Shows whether the target rises because scoring changes, not because the target enters later.</td></tr><tr><td>Proxy selection</td><td>CIFAR-derived candidate groups with fixed target labels</td><td>Selector score over each group</td><td>Tests selector-side exposure without any image generator.</td></tr><tr><td>Fixed clean-generator selection</td><td>Images generated in advance by a clean diffusion pipeline</td><td>CLIP selector applied after generation</td><td>Isolates risk inherited by a clean pipeline that reuses a poisoned CLIP selector.</td></tr></table>

Table 37: Controls used by the deployment protocols. Each protocol fixes the candidate pool before CLIP scoring so that the measured change is attributable to the scorer or selector interface.

<table><tr><td>Evidence / boundary</td><td>Fixed object</td><td>Variable under audit</td><td>Boundary preserved</td></tr><tr><td>COCO retrieval/reranking</td><td>Candidate corpus and local reranking pools</td><td>Triggered versus clean text scoring</td><td>Does not cover all natural query distributions.</td></tr><tr><td>Proxy selection</td><td>Cached candidate groups</td><td>CLIP selector under clean/reference versus attack condition</td><td>Does not claim visual realism.</td></tr><tr><td>Fixed clean-generator selection</td><td>Generated images before scoring</td><td>CLIP selector after generation</td><td>Does not attack or control the generator.</td></tr><tr><td>Weak or negative deltas</td><td>Valid interface and reference condition</td><td>Measured target movement</td><td>Interface-specific observation, not a universal safety guarantee.</td></tr><tr><td>N.E. cells</td><td>No well-formed exposure cell</td><td>Semantic applicability</td><td>Not a numeric zero and not included in exposure denominators.</td></tr><tr><td>Release packaging</td><td>Checkpoints, manifests, evaluator outputs, summaries</td><td>External rerun completeness</td><td>Missing packaging metadata is a reproducibility boundary, not an N.E. decision.</td></tr></table>

Table 38: Compact scope and reproducibility guide for the deployment protocols. The table separates deployment controls, weak/negative results, N.E. decisions, and release boundaries.

![](images/e55b1b9357532bebfab3c613b1f21ade997e578b14d8b922f80116acbc321266.jpg)

<details>
<summary>text_image</summary>

Target: airplane
(a) BadTextTower
Poisoned selected
Clean selected
Target: airplane
(b) ToxicTextCLIP
Clean selected
Poisoned selected
</details>

Figure 4: Proxy candidate-selection examples. Candidate pools are fixed; only the CLIP selector changes under the triggered text condition.

## H.1 Deployment Protocols

All deployment protocols follow the same control principle: construct the candidate pool before CLIP scoring, then evaluate how the clean/reference and attack conditions score the same candidates. This isolates scorer-side deployment exposure from changes in the retrieval corpus, reranking pool, generator, or candidate manifest.

COCO retrieval and reranking. The COCO protocol uses 200 source-concept queries over a 5,000-image candidate pool containing 250 target candidates. Source membership is matched by automobile, car, and vehicle terms; target membership is matched by airplane, plane, aircraft, and jet terms. The triggered query prepends xbtd to the clean query. Full-pool retrieval reports target promotion over all candidates. Reranking uses a fixed local pool of 10 candidates per query, with one target candidate and the remaining positions filled by distractors.

Candidate selection. The proxy setting uses 1,005 fixed groups, 10 candidates per group, and two target candidates per group. The fixed cleangenerator setting uses images generated before scoring by a clean Stable Diffusion pipeline (Rombach et al., 2022), with 500 groups, six candidates per group, and one target candidate per group. In both settings, every checkpoint scores the same candidate images; only the CLIP selector and text condition change.

For checkpoints with a defined text-trigger channel, such as BADTEXTTOWER and TOXI-CTEXTCLIP, the selection protocol uses the corresponding triggered text condition. For checkpoints without a defined text-trigger channel, the protocol does not introduce a semantically meaningful BAD-TEXTTOWER-style triggered query; the reported selector-side values test whether reusing that checkpoint as the CLIP selector promotes the target under the fixed candidate groups.

## H.2 Quantitative Deployment Results

Table 32 reports signed deltas for COCO retrieval and reranking. Positive values indicate target promotion under the triggered text condition; negative values indicate target demotion relative to the reference.

Tables 33 and 34 report candidate-selection outcomes. Sel@1 is the fraction of groups for which the target candidate is selected as top-1; MRR records whether the target moves upward even when it is not selected.

Across retrieval, reranking, and selection, BAD-TEXTTOWER is the only checkpoint with large positive text-query deployment deltas. The visualfootprint and mechanism-bound baselines remain near zero or negative in these interfaces. This does not make those baselines safe in general; Appendix D shows that they are exposed when the downstream interface reads their visual or mechanism-compatible footprint.

## H.3 Qualitative Cases

The qualitative cases are illustrative examples from the same fixed-pool protocols as the quantitative tables. They are not additional metrics. For COCO retrieval and reranking, the archived evidence contains captions and rank-derived quantities, so we report textual cases in Table 36. For candidate selection, Figure 4 shows proxy fixed-pool examples scored under clean/reference and attack conditions.

## H.4 Protocol Controls

Table 37 summarizes the control logic behind the deployment protocols. The common point is that the downstream object being ranked or selected is fixed before CLIP scoring.

## H.5 Scope, Weak/Negative Results, and Reproducibility Boundary

The deployment-style experiments support a specific operational claim: a poisoned CLIP scorer can change retrieval, reranking, and candidate-selection decisions when the downstream system consumes triggered text through the poisoned textual encoder. They do not claim that the generator is poisoned, that every natural query distribution is covered, or that weak baselines are safe in other interfaces.

Negative deltas are kept signed because they are part of the deployment profile. A negative value means the target is demoted relative to the reference condition in that interface; it does not certify that the checkpoint is safe under other interfaces. Similarly, N.E. remains semantic non-applicability. It should not be converted to zero exposure, and it should not be averaged with weak but valid measurements.