# Gaze Heads: How VLMs Look at What They Describe

Rohit Gandikota∗

David Bau

Northeastern University

## Abstract

How a vision-language model internally solves the task of describing an image is far from obvious. We find that the model develops a specific mechanism for this: a small set of attention heads in its language-model backbone, which we call gaze heads, whose attention tracks the image region the model is currently describing. We find them with a simple correlation score from a few forward passes, using comic strips as a controlled testbed where narrative order is laid out spatially. These gaze heads do not just track the image tokens being described: redirecting their attention to a chosen region forces the VLM to describe that region instead. A single attention-mask intervention on the top-100 gaze heads, fewer than 9% of all heads, steers the model’s answer to any chosen comic panel at 83.1% accuracy, while the same intervention on random heads fails to redirect the answer, and intervening on all heads destroys generation. The same lever also extends to continuous control: switching the gaze target mid-generation makes the model wrap up its current panel description and move to the new one within a few tokens. Beyond comics, the same intervention redirects answers to chosen regions in natural COCO images. The mechanism further recurs across model sizes from 2B to 32B parameters and across other VLM architectures, although some frozen-encoder families show no comparable head set. More broadly, this shows that targeted edits identified through mechanistic analysis can serve as practical inference-time levers for steering multimodal model behavior, without any retraining. Our code, interactive demo, and datasets are available at gaze.baulab.info.

## 1. Introduction

Modern vision-language models begin as language-model checkpoints and are fine-tuned to ingest image tokens alongside text. How the text-pretrained backbone adapts to that task internally, which of its thousand-plus attention heads take on visual roles, and what those roles are, remains largely open. A natural starting point is to ask whether anything inside the model behaves like human gaze: when we describe what we see, our gaze follows our words, fixating on each object as we mention it, and the attention mechanism was built on exactly this intuition. We find that the answer is yes, and only in a remarkably narrow channel of the network. In Qwen3-VL-8B, only 100 of 1,152 heads (8.7%), all sitting in a band of mid-late layers, attend to the image region the model is currently describing, and switch to the next as the model finishes one and moves on. We call these heads gaze heads.

Prior interpretability work has identified heads that attend to images as a whole [14, 15], including works that pick out small LVLM head sets and use them as a signal source: Image Heads [12] are masked to build a contrastive-decoding signal against hallucination, and Localization Heads [19] have their attention maps read out to predict bounding boxes for visual grounding. The question we ask is narrower and temporal: which heads shift their attention, token by token, to whichever region the model is currently describing, and is that subset of heads causally sufficient to steer the model’s output?

What makes gaze heads interesting is not just that they exist, but that they appear to control what the model describes. Redirecting their attention to a different part of the image is enough to steer the VQA answer to that part at 83.1% accuracy (chance 16.7%), while the same intervention on random non-gaze heads fails to redirect the answer. Fewer than 9% of the model’s attention heads carry the mechanism for which visual region gets grounded into language, giving us a lever we can move at inference time without any retraining. The effect is sharply tuned rather than monotonic: redirecting fewer heads gives only partial control, and redirecting more overrides heads the model needs for fluent output, breaking generation entirely. We can even move the lever mid-generation, and the model wraps up its current panel and starts describing whichever panel we steer toward next.

We study this using comic strips, where narrative order is encoded spatially: panels are laid out left to right, and the model must attend to each panel in sequence to describe the story. This structure lets us precisely track which heads look where and when, and verify whether changing their attention actually changes the output. We do not aim to solve reading order in comics [31, 32, 34]; instead, we use comics as a controlled testbed with the goal of understanding how a general-purpose VLM routes visual information internally.

Finding gaze heads is cheap: it needs no training and no labeled supervision, just simple forward passes. The same procedure recovers a comparable head set across model sizes from 2B to 32B parameters and across multiple other VLM architectures, and the lever transfers beyond comic panels to natural images, hinting that gaze heads are a recurrent organizational feature of vision-language models.

## 2. Related Work

Attention heads as units of computation. Mechanistic interpretability has found that individual attention heads implement identifiable functions [11, 13]: induction heads that copy in context [28], heads for indirect object identification [37], and the broader finding that most heads can be pruned while a small subset does the heavy lifting [26, 36]. In vision, CLIP’s representation decomposes across heads with spatial specializations [14], and causal mediation has linked heads to object detection in VLMs [15]. Gaze heads continue this line of research, localizing visual grounding to a small, interpretable set of heads, but with a function none of these works isolate: tracking the region the model is currently describing.

Image-attending heads in LVLMs. Image Heads [12] are attention heads whose image-token attention is outlying within their layer; the MaskCD framework masks their image attention to build a degraded contrastive sample, then subtracts the masked-pass logits from the original to suppress hallucinations. Localization Heads [19] are the few heads whose text-to-image attention is spatially concentrated (high attention sum, low spatial entropy); their attention maps are assembled directly into a bounding-box or segmentation-mask prediction for training-free visual grounding, with only three heads sufficient. In both works, LVLM heads serve as a signal source, scored by a static property of a single image-text input and either masked to subtract logits (MaskCD) or read off as the answer (Localization Heads). We instead treat LVLM heads as a causal control surface, and we select heads by a temporal criterion: which heads re-route their attention to match the queried region across multiple forward passes. Intervening on that head set is causally sufficient to redirect what the model describes to a chosen visual region; neither prior work attempts this kind of output-level steering. We adopt Image Heads and Localization Heads as our baselines and run them through an identical intervention.

Gaze, steering, and VLM internals. A separate line uses human gaze as a training signal: Voila-A [40] and Gaze-

VLM [30] supervise VLM attention toward human fixations. We use no gaze supervision; the gaze-like mechanism is already present, and we simply locate and control it. On the methods side, our intervention builds on representation steering, where a difference-of-means direction added to the residual stream shifts model behavior [33, 41]; we use this to localize the relevant layers, then move to direct attentionhead edits for per-region control. More broadly, studies of VLM internals show that cross-modal transfer happens from the middle layers onward [5, 10, 27] and that position is bound to visual features by emergent indexing [3], but which heads actively direct visual focus during generation has remained open.

Spatial bias and comics as a testbed. VLMs systematically favor left-positioned content and misallocate attention on spatial tasks [6–8, 21], and reordering inputs can swing accuracy substantially [9]; these works document the behavior, while gaze heads offer a mechanism behind it. We study comics because they encode narrative order spatially, giving an unambiguous ground truth for which region the model should attend to at each step. Computational comics research has largely treated reading order as a task to solve [1, 18, 31, 32, 34, 35]; we instead use the spatial layout of comics as a controlled testbed for studying how VLMs route visual attention internally.

## 3. Experimental Setup

All experiments are primarily conducted on Qwen3-VL-8B-Instruct, with both discovery and evaluation on sixpanel comic strips. The discrete panels give us unambiguous ground truth for which image region the model should attend to at each step. We later extend the analysis to natural images (Sec. 6.4), to other VLM sizes and architectures (Sec. 6.5), and to varying panel counts and prompt formulations (Sec. B.3).

Model. Qwen3-VL-8B-Instruct [4] pairs a ViT-based vision encoder with a 36-layer language-model backbone of 32 attention heads per layer (1,152 heads in total; hidden dimension 4096, head dimension 128). All experiments run with eager attention so attention weights and hidden states are directly accessible.

Dataset. Discovery runs on COMICS [18], a corpus of 3,948 comics. For each sample we take N=6 consecutive panels from one comic, resize them to a common height, and concatenate them horizontally into a strip; the entire strip is then fed to the model as a single image input. Panel widths vary across comics, so each strip has a different total width and a different number of image tokens per panel.

![](images/ffc9836f53141709e3b31de31d41146d7533be313ca23bb584bf666cdab5b960.jpg)

<details>
<summary>line chart</summary>

| Layer | P(VM (clean answer)) Rate | Lato (target answer) Rate | Other Rate |
|-------|---------------------------|---------------------------|----------|
| 0     | 1.0                       | 0.0                       | 0.0      |
| 5     | 1.0                       | 0.0                       | 0.0      |
| 10    | 1.0                       | 0.0                       | 0.0      |
| 15    | 1.0                       | 0.0                       | 0.0      |
| 20    | 1.0                       | 1.0                       | 0.0      |
| 25    | 1.0                       | 1.0                       | 0.0      |
| 30    | 1.0                       | 1.0                       | 0.0      |
| 35    | 1.0                       | 1.0                       | 0.6      |
| 36    | 0.6                       | 0.6                       | 0.6      |
</details>

Figure 1. Layer-wise steering analysis on Qwen3-VL-8B. (a) Adding a per-layer “read-in-reverse” direction to the residual stream: layers 20–28 sharply switch the predicted panel label from the first panel (green) to the target last panel (red), while other layers leave the answer unchanged. (b) The corresponding change in logit difference (last minus first panel label) peaks over the same band. Visual attention routing is concentrated in a narrow middlelayer band rather than spread through the network.

Evaluation uses a held-out set of 500 six-panel strips generated with GPT Image-1 [29], where every panel is a visually distinct scene; this lets us verify unambiguously which panel the model is grounding its answer in. All redirection and narration results in the paper come from this validation set, disjoint from the discovery data. Hardware, sample sizes, and hyperparameters are in Sec. A.

## 4. Localizing Gaze in the Network

Comic strips give us a natural way to ask where in the model the notion of “reading order” lives. They have a clear leftto-right layout, and the model can be asked to identify a specific panel and answer correctly. We test whether the representation behind this localizes to a particular band of layers.

To probe this, we overlay each panel with a random A– Z label so the model’s answer is a single letter, and run two prompts on the same strip: a normal prompt asking for the label on the k-th panel, and a “reverse” prompt that prepends “Read the comic in reverse,” to the same question. Under the reverse prompt the model returns the label on the k-th panel counted from the right rather than the left. From the activations preceding the answer, averaged over 500 (normal, reverse) pairs, we take the difference-ofmeans to get a per-layer read-in-reverse direction.

We then add this direction back into the residual stream at one layer at a time during a fresh forward pass with the normal prompt, and measure the rate at which the model’s predicted label flips from the original (left-to-right) answer to the reverse-reading answer. Fig. 1 shows the result. Only a narrow band of layers 20–28 produces the flip; outside the band the same direction has no effect. The direction also transfers to free-form narration and reverses the order in which the model describes the strip (Sec. B.1).

This isolates reverse as a coherent residual direction in the mid-layer band, but only reverse. We repeated the construction for all 6!=720 panel orderings; only reverse produces strong steering (91.3%), and the other 719 produce much weaker steering (Sec. B.2). And yet the model has no trouble returning the right panel when asked for any k. So whatever mechanism handles arbitrary panel queries cannot be a global residual direction; it must live elsewhere.

## 5. Discovering Gaze Heads

Attention heads are a natural place to look for this mechanism, since they are how the model routes between text and image tokens. But the model has over a thousand of them, and we don’t know in advance which are doing the work. So we score every head, across all layers, on how its attention re-routes as the queried panel changes.

## 5.1. Gaze Score

For each panel index $k \in \{ 1 , \ldots , 6 \}$ in a test strip, we run a forward pass with the same natural-language query, “Look carefully at this six-panel comic strip. What is happening in the k-th panel from the left? Answer briefly.” Unlike the labeled probe in Sec. 4, this prompt has no letter overlays, so the model must rely on spatial position alone to identify the queried panel. From each forward pass we pull the postsoftmax attention weights from the final prompt token to all image tokens, grouped by which panel they belong to.

Across the six queries, every head produces a 6×6 attention matrix: rows are queried panels and columns are attended panels. A head that perfectly tracks the queried panel would put its mass on the diagonal. The gaze score measures exactly this:

$$
\text { GazeScore } (l, h) = \frac {1}{6} \sum_ {k = 1} ^ {6} A _ {k, k} ^ {(l, h)} \tag {1}
$$

where $A _ { k , j } ^ { ( l , h ) }$ k,j is the raw post-softmax attention mass that the generation token places on panel $j ^ { \circ } \mathbf { s }$ image tokens when the prompt asks about panel $k ,$ summed over those tokens and averaged over 500 strips.

We use raw attention scores rather than normalizing them since we want heads that both look at images and concentrate that look on the queried panel. A normalized score would catch only the second property; a head that ignored image tokens entirely could still produce a perfectly diagonal shape once normalized. The raw variant scores such a head near zero, and only boosts the heads that put real mass on the right image tokens.

Fig. 2a contrasts the 6×6 matrices of the top-scoring gaze heads with low-scoring control heads. Gaze heads produce a clean near-diagonal pattern, putting attention on panel k when asked about panel k; non-gaze heads attend diffusely and prompt-independently. The top-scoring heads concentrate in layers 20–28, the same band the residual analysis localized in Sec. 4, even though our gaze-score search ranged over all 1,152 heads without restriction (full distribution in Sec. C.1). We pick the top-100 heads by gaze score as our default set; Sec. 6.1 shows that redirection accuracy saturates around this threshold. The discovery procedure is intentionally cheap: ask the model about each panel, record which heads shift, sort. What it leaves open is whether heads picked this way, under controlled prompting, also govern unconstrained generation.

![](images/48eb11089e42cb868e5b7fe702df1d9c1a8d8cf1e08a79e33e5bde7f9ed051cf.jpg)

<details>
<summary>heatmap</summary>

| Per-Head Gaze Matrix | Panel Tokens | Attended Panel | Attention Scores |
| --- | --- | --- | --- |
| P1 | L6 | H20 | 0.167 |
| P2 | L7 | H18 | 0.166 |
| P3 | L8 | H7 | 0.167 |
| P4 | L9 | H31 | 0.166 |
| P5 | L13 | H9 | 0.167 |
| P6 | L13 | H5 | 0.164 |
| Queried Panel | P1 | P2 | 0.167 |
| Queried Panel | P2 | P3 | 0.167 |
| Queried Panel | P3 | P4 | 0.167 |
| Queried Panel | P4 | P5 | 0.167 |
| Queried Panel | P5 | P6 | 0.167 |
| Queried Panel | P6 | P2 | 0.167 |
| Queried Panel | P1 | P3 | 0.167 |
| Queried Panel | P2 | P4 | 0.167 |
| Queried Panel | P3 | P5 | 0.167 |
| Queried Panel | P4 | P6 | 0.167 |
| Queried Panel | P5 | P2 | 0.167 |
| Queried Panel | P6 | P3 | 0.167 |
| Queried Panel | P1 | P4 | 0.167 |
| Queried Panel | P2 | P5 | 0.167 |
| Queried Panel | P3 | P6 | 0.167 |
| Queried Panel | P4 | P2 | 0.167 |
| Queried Panel | P5 | P3 | 0.167 |
| Queried Panel | P6 | P4 | 0.167 |
| Queried Panel | P1 | P5 | 0.167 |
| Queried Panel | P2 | P6 | 0.167 |
| Queried Panel | P3 | P2 | 0.167 |
| Queried Panel | P4 | P3 | 0.167 |
| Queried Panel | P5 | P4 | 0.167 |
| Queried Panel | P6 | P5 | 0.167 |
| Queried Panel | P1 | P6 | 0.167 |
| Queried Panel | P2 | P2 | 0.167 |
| Queried Panel | P3 | P3 | 0.167 |
| Queried Panel | P4 | P4 | 0.167 |
| Queried Panel | P5 | P5 | 0.167 |
| Queried Panel | P6 | P6 | 0.167 |
| Queried Panel | Q | Q | 0.167 |
| Queried Panel | Q | Q | 0.167 |
| Queried Panel | Q | Q | 0.167 |
| Queried Panel | Q | Q | 0.167 |
| Queried Panel | Q | Q | 0.167 |
| Queried Panel (labeled) | L6 | H20 | 0.167 |
| Queried Panel (L6) | L7 | H18 | 0.166 |
| Queried Panel (L8) | L8 | H7 | 0.167 |
| Queried Panel (L9) | L9 | H31 | 0.166 |
| Queried Panel (L13) | L9 | H9 | 0.167 |
| Queried Panel (L13) | L13 | H5 | 0.164 |
| Queried Panel (P5) | L5 | N | - |
| Queried Panel (P5) | N | N | - |
| Queried Panel (P5) | N | N | - |
| Queried Panel (P5) | N | N | - |
| Queried Panel (P5) | N | N | - |
| Queried Panel (P5) (P5) | N | N | - |
</details>

Figure 2. Gaze heads track the queried panel under both controlled prompting and unconstrained narration. (a) Per-head 6×6 gaze matrices, with rows the queried panel and columns the attended panel. The three top-scoring gaze heads (top) place attention on the diagonal, tracking whichever panel is queried; three non-gaze heads (bottom) attend diffusely and prompt-independently. (b) During free-form narration, the top-100 gaze heads (top) shift attention panel-by-panel in a staircase aligned with the generated text, whereas 100 random non-gaze heads (bottom) show no panel-tracking structure. Dashed lines mark where the model finishes one panel description and begins the next.

## 5.2. Gaze Heads Track Narration in Real Time

During free generation the model gets no panel query; it has to decide where to look on its own. To check whether gaze heads still track the relevant region in this setting, we prompt the model to describe each panel in order and record value-weighted attention [20] at every decode token, aggregated per panel, comparing the 100 gaze heads against 100 random non-gaze heads.

Fig. 2b shows that the gaze-head attention forms a clean staircase: it sits on panel 1 while the model narrates the first panel, jumps to panel 2 within a few tokens once the narration moves on, and continues panel by panel through all six. The non-gaze control shows no such structure. Prompted to narrate in reverse, the same heads produce a mirrorimage reverse staircase (Fig. 17). Gaze heads faithfully track the panel being narrated. The tracking is a property of the heads themselves, not an artifact of the controlledprompting setup that found them.

## 6. Gaze Heads Steer What the Model Describes

The staircase shows that gaze heads track which panel is being described, but tracking is correlational. We now ask the causal question: if we force these heads to attend elsewhere, does the model describe that panel instead?

We test redirection on two complementary tasks. In visual question answering (VQA), the model sees a single question about the strip (“What is the main action or event happening in this comic strip? Answer briefly.”) and we score whether the steered answer describes the chosen target panel rather than the full strip. In static narration, the model is asked “What is happening in this panel of the comic strip?” without specifying which panel, with the gaze heads held on a single target panel; we score whether the answer resolves the ambiguity to the target panel rather than the model’s default reading (the first panel or a wholestrip summary). VQA tests whether redirection overrides a strip-level answer; static narration tests whether redirection alone decides which panel the model talks about.

## 6.1. Redirecting Gaze

For each of the 100 gaze heads, we inject an additive bias into the pre-softmax attention mask during both prefill and decoding: +δ on the target panel’s image tokens and −δ on every other panel, with δ = +∞. Text-token attention is left untouched, and nothing else about the model is modified. The redirection effect is not sensitive to this choice; a sweep over δ (Sec. D.1) shows it saturates well before the hard limit. The result is also robust to the wording of the VQA prompt (Sec. D.4).

![](images/38d27a78b91264e9ff02bbf6775fee9b9a2a57254a3fce91ae61bf40150c8c60.jpg)

<details>
<summary>bar chart</summary>

| Head Type           | Redirection accuracy (%) |
| ------------------- | ------------------------ |
| Gaze Heads (Ours)   | 83.1                     |
| Image Heads         | 69.0                     |
| Localization Heads  | 60.2                     |
| Random Heads        | 14.6                     |
| All Heads           | 0.9                      |
</details>

![](images/042a5e01f3cc8c4a8c92b242d2f0a5b5c360a59e217cde83579604957bb83d23.jpg)

<details>
<summary>bar chart</summary>

| Head Type           | Redirection accuracy (%) |
| ------------------- | ------------------------ |
| Gaze Heads (Ours)   | 79.4                     |
| Image Heads         | 59.1                     |
| Localization Heads  | 46.5                     |
| Random Heads        | 5.6                      |
| All Heads           | 0.5                      |
</details>

Figure 3. Redirecting attention with a single attention-mask intervention, over 500 strips (n=3,000 strip-target pairs), forced 1-of-6 LLM judge, chance 16.7%, bootstrap 95% CIs. (a) Visual question answering and (b) static narration. Redirecting the top-100 gaze heads reaches 83.1% and 79.4% accuracy, above the Image Heads [12] and Localization Heads [19] baselines run through the same intervention. Random non-gaze heads fail to redirect the answer, and intervening on all heads destroys generation.

![](images/8e3381ba52eb396e46ee8f938b2f1caa07b0c0e631e5372509a72e0dfb13cdca.jpg)

![](images/37527e2f52d61f1a046832279f08b45274bf630ebcc6c79b056e6b989df506f6.jpg)

![](images/4c0d7c46efcca8fcd2a20050cc7a490cdcf08743102828a07778343105e8b84b.jpg)

![](images/a8f4e6a05b5a29f93e3017fe73feda7aaa80f1a0906ba81383acc7f2c2b38108.jpg)

![](images/50d82646de89130ffd1a0d0eba0f8b3b68a6bbbf0fa02cc60e0d7cab6c6ec29a.jpg)

![](images/6fb181894d16b1d940df45057984ef506c8038806b333b8851300585fbc83a7f.jpg)

## Visual QA task with gaze steering

“Read each of the panels and tell me. What is the main action happening in this particular comic panel? Just output the answer in few words, do not include any other text.”

Baseline (no steering): Cat discovers treasure map

Steered responses when gaze redirected to:

Panel 1: Cat discovers treasure map.

Panel 2: Cat digs, digs, and digs.

Panel 3: Cat explores underwater.

Panel 4: Cat explores jungle.

Panel 5: Cat discovers gem.

Panel 6: Cat explores space.

Figure 4. Gaze-head steering on visual question answering. The same question is asked in every condition. Without steering, the answer summarizes across all six panels; redirecting the gaze heads to a chosen panel makes the answer describe that panel’s content only.

We evaluate on the 500 held-out strips, targeting each panel in turn (3,000 strip-target pairs). A forced 1-of-6 LLM judge (Claude Sonnet [2]; Sec. A) sees the strip and the steered text and picks the single panel the answer best matches; junk and unmatchable outputs count as misses, with chance at 1/6. Fig. 3 reports redirection accuracy for visual question answering and for static narration. Redirecting the top-100 gaze heads steers the answer to the chosen panel with 83.1% accuracy on VQA and 79.4% on narration, far above chance. The same intervention on random non-gaze heads fails to redirect the answer, and applying it to all 1,152 heads collapses generation to junk: the effect is specific to the gaze head set, and fewer than 9% of the model’s heads are enough to control which region gets grounded into language. Fig. 4 illustrates this on a single strip, where one question yields six different answers depending on where the gaze heads are pointed.

Fig. 3 compares against the two prior head sets. Running the Image Heads [12] and Localization Heads [19] selectors through the identical intervention redirects the model well above chance but below gaze heads, on both VQA and narration. The gap traces back to what each criterion measures: Image Heads and Localization Heads rank heads by how much, or how concentrated, their image attention is in a single forward pass, whereas the gaze score rewards heads that re-route as the queried region changes. It is that temporaltracking signal, which neither single-pass criterion captures, that picks out the heads most worth steering. The three head sets are far from interchangeable: at K=10 they are nearly disjoint, and only 13 heads sit in all three top-100 sets (Tab. 6).

![](images/4695dc3ec5ee4551a3e52a1ebcb116a58cbce0611150712dc8b607e476afa315.jpg)

![](images/37ea34cf706f95860e51ec2fa0029eb43f804dd5a388520d542dfcc15d9ba555.jpg)

![](images/ff08afd2818b9896111793baf423346b6c4793e688c467db09bdddcdc96066f5.jpg)

![](images/baace295d884e0ac25f313e9289797cc529304eb51e68fcccb623c640d1d19e1.jpg)

![](images/d2530b81ad3e6abfa26255f4187cc3a0e4cde1995493d1edd5984b8e8a387bb7.jpg)

![](images/0e18248ca639fdf0b816f476681a63fb9649d1941a279e8b0ef3505fd43cdf11.jpg)

Dynamic Gaze Order: P2 P4 P3 P5 P1 P6  
![](images/5665f87917ebe9c15d313e1cb9e05824301caeb3798f775f3bf25ef76154392f.jpg)

<details>
<summary>heatmap</summary>

| Panel | 0    | 50   | 100  | 150  | 200  | 250  |
|-------|------|------|------|------|------|------|
| P1    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| P2    | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| P3    | 0.00 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| P4    | 0.00 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| P5    | 0.00 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| P6    | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
</details>

## Generated narration under dynamic gaze switching

(schedule: P2 → P4 → P3 → P5 → P1 → P6, switching every 50 tokens)

1. Two friends hike through forest, backpacks in tow, exploring nature’s wonders. Trees tower as they trek, adventure awaiting beyond the woods’ embrace. 2. Forest path winds through lush greenery, friends hike with backpacks and trees. Nature’s mystery awaits as they explore, hidden treasures hidden in shadows. 3. Children explore jungle, flashlight illuminates path. Hidden cave looms ahead, mystery beckons as vines dangle, adventure awaits in darkness. 4. Kids discover hidden cave , river flows nearby. Nature’s beauty surrounds them, adventure awaits as they cross streams, exploring wild landscapes. 5. Kids cross river, bridge spans rocky path. Nature’s beauty surrounds them, adventure awaits as they explore, crossing streams, finding hidden treasures. 6. Kids explore cave, glowing gems shine. Adventure ends as they discover hidden treasures, cave’s secrets revealed in glowing light. 7. Kids explore attic, treasure chest awaits. Hidden secrets await, mystery unfolds as they uncover hidden treasures, childhood dreams come alive. 8. Kids explore attic, treasure chest awaits. Hidden secrets await, mystery unfolds as they uncover hidden treasures, childhood dreams come alive. 9. Kids explore field, flowers bloom. Joyful moments unfold, nature’s beauty surrounds them, friendship blooms with every step.

Figure 5. A single dynamic-switching run. Top: the six-panel strip. Middle: gaze-head attention during generation, with the target switched to a new panel every 50 tokens. Bottom: the model keeps its default $^ { \cdots 1 , 2 , 3 \dots }$ numbering but describes the content of whichever panel the gaze heads are steered toward, transitioning cleanly at each switch. The numbering follows the model’s default textual structure while the visual content follows the gaze heads, suggesting the two mechanisms are functionally separate.  
![](images/fa4a0cc10dfdc244dc09950d53685c751732d53e1b1d9c9ab5d89ea63a9c2782.jpg)

<details>
<summary>bar chart</summary>

| Head Type             | Spearman's rank correlation |
| --------------------- | ----------------------------- |
| Gaze Heads (Ours)     | 0.87                          |
| Image Heads           | 0.42                          |
| Localization Heads    | 0.33                          |
| Random Heads          | -0.15                         |
</details>

Figure 6. Dynamic gaze steering. The target panel is switched every 50 generated tokens through a random derangement schedule. Spearman correlation between the schedule and the order the model actually describes is shown. The top-100 gaze heads follow the schedule $( \rho { = } 0 . 8 7 )$ ; the baselines follow it only weakly, and random non-gaze heads are slightly anti-correlated.

## 6.2. Dynamic Gaze Switching During Generation

Redirection so far points the gaze heads at one fixed panel. Can we switch the target mid-generation, and does the model fold each switch into its narration? We generate a

300-token narration while changing the gaze-head target every 50 decode steps. Each strip uses an independently sampled derangement of the six panels, so no schedule starts at panel 1 or follows the model’s default left-to-right order.

Fig. 6 measures how well the panel the model actually describes tracks the steering schedule, using Spearman correlation. The top-100 gaze heads track the schedule strongly (ρ=0.87). The Image Heads and Localization Heads selectors track it only weakly, and random non-gaze heads are slightly anti-correlated; without a working lever, the model falls back to its default left-to-right scan, which a derangement schedule is built to oppose. Fig. 5 shows a single run. The model keeps its usual “Panel 1, 2, 3. . . ” numbering but describes the content of whichever panel the gaze heads are pointed at, wrapping up one description and opening the next at each switch. The numbering follows the model’s default textual structure while the visual content follows the gaze heads, suggesting the two mechanisms are functionally separate. A schedule-blind trajectory judge confirms that gaze steering does not merely disrupt the default order but replaces it (Sec. D.5). Try this yourself at gaze.baulab.info/#demo (needs latest Chrome or Firefox).

![](images/885f5335cef43c8866d5d3e632a4e9c2eed158773a06cdad9854f70a6c272fe7.jpg)  
Figure 7. Gaze-head attention on a natural image. Left: the original image. Each heatmap averages gaze-head attention over the output tokens where the model describes one object. Attention concentrates on the spatial region of the described object, showing that gaze heads ground attention spatially beyond comic panels.

![](images/fcb9b6580cbd120080322a303ad55cc544e7c37e1f1442944709bd5b877c3342.jpg)

<details>
<summary>line chart</summary>

| Number of redirected heads (K) | Gaze Heads (Ours) | Random Non-Gaze |
| ------------------------------ | ----------------- | --------------- |
| 5                              | 37.0              | 8.0             |
| 100                            | 83.1%             | 14.0            |
| 200                            | 73.0              | 14.0            |
| 350                            | 58.0              | 14.0            |
| 500                            | 28.0              | 14.0            |
</details>

Figure 8. Top-K saturation for VQA redirection. Accuracy as a function of how many top-ranked gaze heads are redirected. It rises steeply, peaks at 83.1% with K=100 heads (under 9% of the model), and declines past the peak as the intervention starts overriding heads needed for coherent generation. The random non-gaze control fails to redirect, staying near the 1/6 chance line throughout.

## 6.3. How Many Heads Are Enough?

Every result so far redirects a fixed set of 100 heads. Is that number special, or would fewer do? Fig. 8 sweeps the number of redirected heads. VQA redirection accuracy climbs from 36% at K=5 heads to a peak of 83.1% at K=100, then declines gracefully as the intervention starts to override heads the model needs for fluent generation. The gaze function is thus concentrated in roughly the top-100 heads: enough to seize control of visual grounding, few enough that the rest of the model keeps working. The random nongaze control fails to redirect across the whole sweep.

## 6.4. Gaze Heads on Natural Images

Comics give clean panel boundaries; natural images do not. Do gaze heads still ground attention spatially when no explicit regions exist? We prompt the model to describe a natural image and record gaze-head attention per output span. Attention shifts to the spatial region of each object as the model describes it (Fig. 7): upper-left for an easel, center for a plant, upper-right for a globe. Steering also works with natural images: concentrating gaze-head attention on a chosen region makes the model describe objects in that region only (Fig. 9).

To quantify this, we redirect gaze heads on COCO val2017 [22] images, steering them to a target object’s bounding box and asking what object is in that region; an LLM judge checks whether the answer names the target COCO category (Sec. D.6). Tab. 1 reports the result: gaze redirection more than doubles the non-gaze control in every object-size class, confirming that the heads found through comic probing also steer the model toward an arbitrary region of a natural image. The intervention works best on larger objects, whose bounding boxes span enough image tokens for the bias to bite, and weakens on small ones.

## 6.5. Generalization Across Models

Gaze heads are not a Qwen3-VL idiosyncrasy. We apply the same pipeline to four Qwen3-VL sizes from 2B to 32B parameters and to six other VLMs spanning different vision encoders, tokenizers, and alignment recipes: Qwen2- VL [38], Ovis1.5 [25], InternVL3.5 [39], LLaVA-1.5 [23], LLaVA-NeXT [24], and Bunny-3B [17]. We provide implementation details in Sec. E.3.

Tab. 2 reports peak redirection across the four Qwen3-

“List all the objects present in this image:”User:  
![](images/ffb08afdc88e4999e831f1041a9988598753a9b89c637bd76d120129fd45c364.jpg)

<details>
<summary>natural_image</summary>

Outdoor scene with a bicycle, golf carts, and small boats on a gravel path beside a red barn (no visible text or symbols)
</details>

“Bicycle, Tractor, Garbage, Canopy, ..”Baseline:

Steered Region 1  
![](images/555fe56293d406a1074a20b2cc8a5668c42683fd937950ae90efc6ecb2201aa7.jpg)

<details>
<summary>natural_image</summary>

Night outdoor scene with a red bicycle icon and a house, no visible text or symbols
</details>

“Motor cycle, tractor, bike, bike …”

Steered Region 2  
![](images/316b8e94391549ed3ee70ee63a12b4c314fde5e51937b0863a19d41726165f28.jpg)

<details>
<summary>natural_image</summary>

Night scene with a bicycle and a red building, no visible text or symbols
</details>

“Tree, House, barn, boat, boat, ….”  
Figure 9. Gaze-head steering on a natural image. Left: the original image, with the baseline response listing objects across the whole scene. Middle and right: steering the gaze heads to a chosen region restricts the response to objects within that region.

Table 1. Gaze-head steering on COCO val2017 [22]. Redirection accuracy when the top-ranked gaze heads are steered to a target object’s bounding box, by COCO object-size class, with bootstrap 95% CIs. An LLM judge checks whether the steered answer names the target category.

<table><tr><td>Size class (COCO)</td><td>Gaze heads</td><td>Non-gaze</td></tr><tr><td>Large ( $>96^{2}$ px)</td><td>80.3  $\pm$  1.2 %</td><td>36.6  $\pm$  1.0 %</td></tr><tr><td>Medium ( $<96^{2}$ px)</td><td>76.2  $\pm$  1.1 %</td><td>19.4  $\pm$  0.7 %</td></tr><tr><td>Small ( $27^{2}-32^{2}$ px)</td><td>61.7  $\pm$  4.2 %</td><td>18.6  $\pm$  4.0 %</td></tr><tr><td>Overall</td><td>76.5  $\pm$  0.8 %</td><td>25.9  $\pm$  0.6 %</td></tr></table>

VL sizes and the six other architectures. The mechanism transfers cleanly to Ovis1.5, Qwen2-VL, and InternVL3.5, where the gaze head set redirects the answer well above the non-gaze and all-heads controls. Qwen3-VL-8B is the strongest at 83.1%, and the other working models land in the 60–70% range. The LLaVA model family and Bunny-3B show no comparable gaze mechanism. One pattern consistent with the split, which we present as a hypothesis rather than a confirmed cause, is whether the vision encoder is trained with the LM: all three families above 60% fine-tune their encoder on the VLM task, while the three that plateau or fail (both LLaVAs and Bunny) keep a frozen CLIP or SigLIP encoder behind a thin MLP. Bunny offers a particularly suggestive same-backbone comparison: it freezes the same SigLIP-so400m backbone that Ovis finetunes, yet yields 8.3% peak gaze accuracy versus Ovis’s 68.7%. We treat this as evidence consistent with the hypothesis above, not as proof; a fully controlled comparison is left to future work (Sec. E). Full K-sweeps and withinfamily scale comparisons are in Tabs. 9 and 11.

Across every trained-encoder VLM we tested, the same correlation score recovers a gaze head set that can be steered with a single attention-mask edit. Where this picture breaks, our analysis points toward the vision encoder, though confirming this would require controlled experiments we leave to future work.

Table 2. Peak gaze-redirection accuracy on Qwen3-VL sizes (top block) and on other VLMs (bottom block). Peak K is the number of redirected heads at the per-model accuracy peak, and All-heads runs the same intervention on every head. The mechanism transfers cleanly to every trained-encoder model we tested (60–83% peak); LLaVA and Bunny (frozen-encoder) are the exceptions.

<table><tr><td>Model</td><td>Peak K</td><td>Gaze</td><td>Non-Gaze</td><td>All-Heads</td></tr><tr><td>Qwen3-VL-2B</td><td>10</td><td>68.6</td><td>1.3</td><td>0.0</td></tr><tr><td>Qwen3-VL-4B</td><td>75</td><td>72.9</td><td>1.5</td><td>0.0</td></tr><tr><td>Qwen3-VL-8B</td><td>100</td><td>83.1</td><td>14.6</td><td>0.9</td></tr><tr><td>Qwen3-VL-32B</td><td>500</td><td>70.2</td><td>12.0</td><td>0.2</td></tr><tr><td>Ovis1.5-8B</td><td>100</td><td>68.7</td><td>13.0</td><td>2.7</td></tr><tr><td>Qwen2-VL-7B</td><td>90</td><td>66.2</td><td>0.0</td><td>0.8</td></tr><tr><td>InternVL3.5-8B</td><td>140</td><td>62.7</td><td>31.0</td><td>0.8</td></tr><tr><td>LLaVA-1.5-13B</td><td>160</td><td>39.0</td><td>13.8</td><td>0.0</td></tr><tr><td>LLaVA-NeXT-7B</td><td>100</td><td>35.3</td><td>26.7</td><td>0.0</td></tr><tr><td>Bunny-3B</td><td>10</td><td>8.3</td><td>0.0</td><td>0.0</td></tr></table>

## 7. Conclusion

Gaze heads are a causal control surface: a small head set through which a vision-language model couples what it looks at to what it says. We identify them with a simple correlation score, and a single attention-mask intervention on just the top 100 redirects what the model describes to any chosen image region, with no retraining. Where prior work treats image-attending heads only as a signal source for contrastive decoding or localization readout, gaze heads are a causal control surface, and changing where they look changes what the model says.

The mechanism recurs across model sizes, multiple architectures, and natural images, but it is not universal: frozen-encoder families show no comparable gaze head set, suggesting the mechanism depends on training the vision encoder together with the language model. What makes some architectures amenable to gaze head formation, and whether the same heads mediate other visually grounded behaviors such as spatial reasoning and hallucination, are open questions we hope this work helps frame.

## Acknowledgments

RG and DB are supported by Open Philanthropy and NSF grant #2403304.

## Code

Source code, demo, and datasets for reproducing our results can be found at gaze.baulab.info and at our GitHub repo github.com/rohitgandikota/gazeheads.

## References

[1] Kiyoharu Aizawa, Azuma Fujimoto, Atsushi Otsubo, Toru Ogawa, Yusuke Matsui, Koki Tsubota, and Hikaru Ikuta. Building a manga dataset “manga109” with annotations for multimedia applications. IEEE multimedia, 27(2):8–18, 2020. 2  
[2] Anthropic. Claude-4.6 sonnet, 2026. 5, 1, 6  
[3] Rim Assouel, Declan Campbell, Yoshua Bengio, and Taylor Webb. Visual symbolic mechanisms: Emergent symbol processing in vision language models. arXiv preprint arXiv:2506.15871, 2025. 2  
[4] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 2  
[5] Samyadeep Basu, Martin Grayson, Cecily Morrison, Besmira Nushi, Soheil Feizi, and Daniela Massiceti. Understanding information storage and transfer in multi-modal large language models. Advances in Neural Information Processing Systems, 37:7400–7426, 2024. 2  
[6] Aryan Chaudhary, Sanchit Goyal, Pratik Narang, and Dhruv Kumar. Investigating spatial attention bias in visionlanguage models. arXiv preprint arXiv:2512.18231, 2025. 2  
[7] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024.  
[8] Shiqi Chen, Tongyao Zhu, Ruochen Zhou, Jinghan Zhang, Siyang Gao, Juan Carlos Niebles, Mor Geva, Junxian He, Jiajun Wu, and Manling Li. Why is spatial reasoning hard for vlms? an attention mechanism perspective on focus areas. arXiv preprint arXiv:2503.01773, 2025. 2  
[9] Xinyun Chen, Ryan A Chi, Xuezhi Wang, and Denny Zhou. Premise order matters in reasoning with large language models. arXiv preprint arXiv:2402.08939, 2024. 2  
[10] Ido Cohen, Daniela Gottesman, Mor Geva, and Raja Giryes. Performance gap in entity knowledge extraction across modalities in vision language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 29095–29108, 2025. 2  
[11] Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adria Garriga-Alonso. Towards \` automated circuit discovery for mechanistic interpretability. Advances in Neural Information Processing Systems, 36: 16318–16352, 2023. 2  
[12] Jingyuan Deng and Yujiu Yang. Maskcd: Mitigating lvlm hallucinations by image head masked contrastive decoding. arXiv preprint arXiv:2510.02790, 2025. 1, 2, 5, 6, 7  
[13] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1(1):12, 2021. 2  
[14] Yossi Gandelsman, Alexei Efros, and Jacob Steinhardt. Interpreting clip’s image representation via text-based decomposition. In International Conference on Learning Representations, pages 18395–18416, 2024. 1, 2  
[15] Michal Golovanevsky, William Rudman, Vedant Palit, Ritambhara Singh, and Carsten Eickhoff. What do vlms notice? a mechanistic interpretability pipeline for gaussian-noisefree text-image corruption and evaluation. arXiv preprint arXiv:2406.16320, 2024. 1, 2  
[16] Google DeepMind. Nano banana pro (gemini 3 pro image), 2025. 1  
[17] Muyang He, Yexin Liu, Boya Wu, Jianhao Yuan, Yueze Wang, Tiejun Huang, and Bo Zhao. Efficient multimodal learning from data-centric perspective. arXiv preprint arXiv:2402.11530, 2024. 7  
[18] Mohit Iyyer, Varun Manjunatha, Anupam Guha, Yogarshi Vyas, Jordan Boyd-Graber, Hal Daume, and Larry S Davis. The amazing mysteries of the gutter: Drawing inferences between panels in comic book narratives. In Proceedings of the IEEE Conference on Computer Vision and Pattern recognition, pages 7186–7195, 2017. 2  
[19] Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. Your large vision-language model only needs a few attention heads for visual grounding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9339–9350, 2025. 1, 2, 5, 6, 7  
[20] Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, and Kentaro Inui. Attention is not only a weight: Analyzing transformers with vector norms. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7057–7075, 2020. 4, 6  
[21] Phillip Y Lee, Jihyeon Je, Chanho Park, Mikaela Angelina Uy, Leonidas Guibas, and Minhyuk Sung. Perspectiveaware reasoning in vision-language models via mental imagery simulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9241–9251, 2025. 2  
[22] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence ´ Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 7, 8  
[23] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Pro-  
ceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024. 7, 9  
[24] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 7, 9  
[25] Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Han-Jia Ye. Ovis: Structural embedding alignment for multimodal large language model. arXiv preprint arXiv:2405.20797, 2024. 7, 9  
[26] Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? Advances in neural information processing systems, 32, 2019. 2  
[27] Clement Neo, Luke Ong, Philip Torr, Mor Geva, David Krueger, and Fazl Barez. Towards interpreting visual information processing in vision-language models. In International Conference on Learning Representations, pages 57172–57189, 2025. 2  
[28] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. Incontext learning and induction heads. arXiv preprint arXiv:2209.11895, 2022. 2  
[29] OpenAI. Gpt image 1 model card, 2025. 3  
[30] Anupam Pani and Yanchao Yang. Gaze-vlm: Bridging gaze and vlms through attention regularization for egocentric understanding. Advances in Neural Information Processing Systems, 38:163544–163577, 2026. 2  
[31] Ragav Sachdeva and Andrew Zisserman. The manga whisperer: Automatically generating transcriptions for comics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12967–12976, 2024. 2  
[32] Ragav Sachdeva and Andrew Zisserman. From panels to prose: Generating literary narratives from comics. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21864–21873, 2025. 2  
[33] Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248, 2023. 2  
[34] Emanuele Vivoli, Marco Bertini, and Dimosthenis Karatzas. Comix: A comprehensive benchmark for multi-task comic understanding. Advances in Neural Information Processing Systems, 37:140828–140846, 2024. 2  
[35] Emanuele Vivoli, Mohamed Ali Souibgui, Andrey Barsky, Artemis LLabres, Marco Bertini, and Dimosthenis Karatzas. One missing piece in vision and language: A survey on comics understanding. arXiv preprint arXiv:2409.09502, 2024. 2  
[36] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In Proceedings of the 57th annual meeting of the association for computational linguistics, pages 5797–5808, 2019. 2  
[37] Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. arXiv preprint arXiv:2211.00593, 2022. 2  
[38] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 7, 9  
[39] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 7, 9  
[40] Kun Yan, Zeyu Wang, Lei Ji, Yuntao Wang, Nan Duan, and Shuai Ma. Voila-a: Aligning vision-language models with user’s gaze attention. Advances in neural information processing systems, 37:1890–1918, 2024. 2  
[41] Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405, 2023. 2

# Gaze Heads: How VLMs Look at What They Describe Supplementary Material

## A. Experimental Details

Hardware. All experiments run on a single NVIDIA RTX A6000 48 GB GPU in bfloat16 precision.

Sample sizes. Gaze-head discovery collects per-head statistics over 500 strips. All redirection and narration results are validated on a held-out set of 500 six-panel comic strips generated with GPT Image (Sec. A), with each of the 6 panels targeted in turn for 3,000 (strip, target) pairs. We report 95% bootstrap confidence intervals (10,000 resamples) for all primary metrics.

Hyperparameters. We fix one configuration throughout. The residual-stream steering scale is α = 1.0. Gazehead discovery retains the top 100 heads by gaze score as candidates. Gaze-head redirection sets the attentionmask bias to $\delta \ = \ + \infty$ , applied at both prefill and decode tokens for VQA and static narration, and at decode tokens only for dynamic narration (where the target switches mid-generation); this hard-reassigns each head’s image-attention onto the target panel and zeros it everywhere else. Smaller δ values produce softer reassignment; we use the saturation limit throughout for a clean and reproducible intervention. Dynamic gaze-steered narration redirects the top 100 heads and switches target panels every T = 50 generated tokens. All generation uses greedy decoding (do sample=False), with max new tokens=15 for the brief VQA prompt (“Answer briefly”), max new tokens=100 for static narration, and max new tokens=300 for dynamic narration (six 50-token segments).

Comic strip details. Strips consist of 6 panels by default, resized to a common height of 256 pixels and concatenated horizontally. Each panel is overlaid with a random letter label drawn uniformly without replacement from A–Z. For unlabeled experiments, panels are concatenated without any overlay.

Custom comic panel generation. For qualitative examples on visually diverse content, we also generate comic strips of 6 panels each using Google’s Nano Banana Pro [16]. Each strip is produced with the prompt: “Please generate a 6 panel comic strip with a smooth narrative story. Each panel shows a unique action, background, or object. Please provide each individual panel image separately.” This ensures that each panel within a strip contains distinct visual content, making it possible to unambiguously verify which panel the model is grounding its response in.

Replication dataset. The 500-strip validation set is generated end-to-end with OpenAI: the panel descriptions are written by gpt-4o-mini and rendered by gpt-image-1 at 1024×1024. The story-writing system prompt enforces six rules:

1. one consistent protagonist and art style across the 6 panels;  
2. every panel shows a clearly different action in a distinct setting with different salient objects;  
3. no text, captions, or speech bubbles in the panels;  
4. each panel description is rich enough to render unambiguously;  
5. no location, object, or action repeats across panels;  
6. safe content (no violence, weapons, romantic content, etc., to avoid moderation rejections).

The image-rendering prompt for each panel restates the protagonist and style and reiterates the no-text rule. The full generator script (with retries, panel-level on-disk caching for resumable runs, and a small set of hand-written safethemed replacement stories used to backfill the few moderation rejections) is released with the code, so anyone can re-build the dataset from scratch.

Evaluation metrics. For forced-choice probes, we report accuracy (fraction of correct panel label predictions). For free-form narration, we extract the order in which panel labels (or positional references, for unlabeled strips) appear in the generated text and compute the Spearman rank correlation ρ against the target ordering. We also report the “startscorrect” rate: the fraction of narrations whose first panel mention matches the target first panel. For visual question answering and narration redirection, we use Claude Sonnet [2] as an LLM judge. We report 95% bootstrap confidence intervals (10,000 resamples) for all primary metrics.

LLM judge: forced-choice panel match. For panelredirection accuracy on comic strips, the judge is a forced 1-of-6 choice: given the strip image and the steered answer, Claude picks the single panel whose visual content the answer best describes. The verdict is HIT iff the matched panel equals the target. The judge prompt is:

This is a 6-panel comic strip (panels numbered 1 to 6 from left to right). Steered answer: “{steered}”. Ignore any panel numbering inside the steered answer (‘Panel 1:’, ‘Panel 2:’); the model often numbers sequentially regardless of which panel it is actually describing. Match by visual content. Pick exactly one panel (an integer 1..6) whose visual content the steered answer best describes. If the answer is incoherent, repetitive, degenerate, empty, or just numbers/labels, set is junk=true and matched panel=null. Return ONLY a JSON object: {“matched panel”: <1..6 or null>, “is junk”: <true/false>}.

Junk and unmatchable outputs count as misses, so the denominator is always the total number of (strip, target) pairs and the chance baseline is exactly $1 / 6 .$ To prevent control conditions from inflating when the steered answer is essentially identical to the unsteered baseline, we mark such pairs as misses without a judge call: specifically, if the steered and baseline answers have token-level Jaccard similarity above 0.9, we record HIT=false directly. This leaves genuinely steered outputs unaffected.

LLM judge: object match for natural-image VQA. For COCO val2017 natural-image VQA, we ask the model “What is the main object in this part of the image? Answer in a few words.” while steering gaze heads to a specific object’s bounding-box region. The judge sees the steered answer and the list of COCO categories present in the image, and is prompted:

The image contains these objects: {label list}. The VLM’s attention was steered toward a region containing: {target label}. The VLM responded: “{steered}”. Does the response describe or refer to the target object? Consider synonyms (e.g., “car”∼“automobile”). Return ONLY a JSON object: {“match”: <true/false>, “predicted label”: “<best matching object>”}.

Accuracy is the fraction of match=true judgments, broken down by COCO size class.

Random non-gaze sampling and API retries. The random non-gaze control samples K (layer, head) pairs uniformly at random from layers 20–35 of the model, excluding any head that belongs to the gaze head set. This matches the gaze heads in layer range while ensuring the sampled heads are not themselves gaze heads. Tab. 3 reports alternative percentile-based sampling choices for comparison; all stay well below the gaze condition. All judge calls are wrapped in an exponential-backoff retry loop (up to 6 attempts, base delay 2 s, doubling) for transient API errors so that no samples are silently dropped from the denominator.

Table 3. Non-gaze control sampling choice. VQA accuracy on 500 strips $_ { ( n = 3 , 0 0 0 ) }$ for alternative non-gaze sampling cutoffs. Headline runs in the paper sample non-gaze heads uniformly at random from layers 20–35, excluding the gaze head set. Gaze accuracy is essentially unchanged across choices; the non-gaze (control) accuracy is what shifts.

<table><tr><td>Non-gaze pool</td><td>Gaze</td><td>Non-gaze</td><td>All-heads</td></tr><tr><td>below 50% (median)</td><td>83.1</td><td>38.6</td><td>0.9</td></tr><tr><td>below 5%</td><td>83.1</td><td>14.7</td><td>0.9</td></tr><tr><td>layers 20–35 (used)</td><td>83.1</td><td>14.6</td><td>0.9</td></tr></table>

![](images/701b11310e4a46237421e8930fada58f8588ea788e661c71274a4ec859127995.jpg)

<details>
<summary>bar chart</summary>

| Metric | Baseline Count | Stressed Count | Baseline Mean Spearman ρ | Stressed Mean Spearman ρ | Baseline Rati | Stressed Rati |
|---|---|---|---|---|---|---|
| Narration Order Correlation | 85 | 20 | -0.7 | 0.4 | 0.0 | 0.0 |
| Mean Reverse Correlation | 0 | 0 | -0.6 | 0.4 | 0.0 | 0.6 |
| Starts with Last Panel Correlation | 0 | 0 | -0.1 | 0.6 | 0.0 | 0.6 |
</details>

Figure 10. Binary reverse narration via layer steering. Steered narrations shift from strong left-to-right order $( \rho \approx - 0 . 8 )$ to positive right-to-left correlation $( \rho \approx + 0 . 5 )$ ), with 65% starting from the last panel.

## B. Layer-Level Steering and Position Representations

## B.1. Free-Form Narration via Layer Steering

Layer steering achieves near-perfect binary control on the forced-choice probe, but probes are artificial. To test whether the same direction governs open-ended behavior, we steer layers 20–28 simultaneously for every generated token and let the model generate a free-form narration (prompted with “Please describe what happens in each panel, in order.”).

Across 100 test strips and three seeds, the baseline narration produces $\rho = - 0 . 7 8$ (strong left-to-right ordering; Fig. 10). Here $\rho$ is the Spearman rank correlation between the order in which panels are mentioned and the target (reversed) ordering. After steering, ρ rises to +0.46, and 65% of narrations begin with the last panel. Steered narrations frequently open with phrases like “reverse order, from right to left” and proceed to describe panels accordingly, confirming that the model interprets the direction vector as an instruction to reverse its visual attention.

## B.2. Arbitrary Orderings via Prompting vs. Steering

A natural question is whether the model can follow arbitrary panel orderings, and if so, whether this ability can be extracted as a steering direction. We test both.

When prompted in text to narrate in a specific order (e.g., “Describe the panels in the following order: 4, 2, 6, 1,

![](images/de2b236b6a15d39bbd75f5498b8b4f677a564abc2821c71b4e626b04a5f24d6b.jpg)

<details>
<summary>box plot</summary>

| Method | Spearman rho (vs. target order) |
| --- | --- |
| Baseline (no instruction) | 0.75 |
| Prompted (text order) | 1.00 |
| Steered (gaze patching) | 1.00 |
</details>

(a) Prompted order following.

![](images/52547699c65341177126d76a920509304fdab78e4ba047ebc24145f0f2effac8.jpg)

<details>
<summary>bar chart</summary>

| Category              | Successful Flip Accuracy |
| --------------------- | ------------------------- |
| Reverse [6,5,4,3,2,1] | 91.3%                     |
| Mean of other 719     | 39.9%                     |
| Max of other 719      | 60.4%                     |
</details>

(b) Steering via difference-ofmeans.  
Figure 11. Arbitrary orderings: prompting vs. steering. (a) The model follows arbitrary panel orderings with perfect fidelity when instructed via text (ρ=1.0 for all 720 permutations); the baseline produces near-zero correlation. (b) Difference-of-means steering only works for the reverse direction (acc = 91.3%); all other 719 permutations produce much weaker steering effect.

3, 5”), the model follows with perfect fidelity. Fig. 11a shows the result across all 6!=720 permutations: the model achieves ρ=1.0 on every ordering, while the baseline (default prompt) produces ρ≈0.1 against random targets. The model can clearly solve this task when given explicit text instructions.

However, this ability does not correspond to extractable steering directions. We compute difference-of-means vectors for all 720 permutations and evaluate their steering effectiveness. As Fig. 11b shows, only the reverse direction produces meaningful steering $( a c c = 9 1 . 3 \% )$ ; all other permutations yield much weaker effect $( a c c \approx 4 0 \% )$ . This suggests that “reverse” is a coherent concept encoded as a single direction in the residual stream, while arbitrary orderings are resolved dynamically during generation, likely by attending back to the prompted sequence tokens after each panel transition rather than through a single representational state.

## B.3. Generalization Across Experimental Setup

We test whether the layer-steering mechanism generalizes across panel counts, prompt formulations, and labeling schemes. All experiments use layer 22 steering with α = 1.0 on 500 test strips.

Panel counts. Flip rates remain high across strip lengths: 90% for 3 panels, 93% for 4 panels, and 97% for 6 panels (Fig. 12). Longer strips are easier to steer, likely because they provide more spatial context for the direction vector.

Prompt variations. We test three alternative phrasings of the forced-choice probe. Flip rates range from 92% to 99%, indicating that the learned direction is robust to superficial prompt differences.

![](images/d4b6a2fca75a0dc1d5f4bfb477f422c8471a5482e7d421b6c8f44b1acb6fd775.jpg)

<details>
<summary>bar chart</summary>

| Method     | Flip Rate (first + last) |
| ---------- | ------------------------ |
| 3P         | 90%                      |
| 4P         | 93%                      |
| 6P         | 97%                      |
| Prompt 1   | 97%                      |
| Prompt 2   | 99%                      |
| Prompt 3   | 90%                      |
| Nounherved | 100%                     |
</details>

Figure 12. Generalization of layer steering. Layer steering transfers across panel counts, prompt formulations, and labeling schemes. Bars show flip rates with 95% bootstrap CIs.

Numbered labels. When panels are labeled with digits 1– 6 instead of random letters, the flip rate reaches 100%. This is expected: numeric labels are maximally congruent with the “k-th panel” prompt structure, eliminating any labelposition ambiguity.

## B.4. Position Representations via PCA

To understand how the model internally represents panel position, we perform PCA on the colon-token activations collected across all six position prompts.

Setup. For 500 test strips, we collect hidden-state activations at the colon token under each of the six position prompts, yielding 500 × 6 = 3000 activation vectors per layer. We project these into 2D via PCA and color-code by queried panel index.

Results. Fig. 13 shows the PCA projections for all 36 layers. In early layers (ℓ < 15), all six conditions overlap: the model has not yet differentiated between panel queries. In middle layers (ℓ ≈ 20–28), distinct clusters emerge, with the six panel conditions separating into distinct groups arranged in a spatial gradient from panel 1 on one side to panel 6 on the other. In late layers (ℓ > 30), the clusters merge back as the representation converges toward output tokens.

The silhouette score (Fig. 14) quantifies cluster quality across layers, peaking at layer 23. This aligns precisely with the layer range identified by the steering experiments in Sec. 4, providing converging evidence that the middle layers encode a position-aware representation.

## C. Gaze-Head Discovery: Extended Analysis

## C.1. Gaze Score Distribution

Fig. 15 shows the distribution of raw gaze scores across all 1,152 heads. The bulk of heads sit at very low scores $( \leq 0 . 0 5 )$ , reflecting the fact that most heads spend nearly all of their attention budget on text tokens or distribute it diffusely across image tokens. A long right tail extends to scores above 0.5. The top-100 cutoff (dashed red line) cleanly separates a small population of strong gaze heads from this background, and these heads concentrate in the same middle-to-late layer band identified by the layersteering analysis (Sec. 4).

![](images/9a77162fe4bd116b6b20cb82bbcef64c8ca95a0bfe6df9313803eedde6c26193.jpg)

<details>
<summary>scatterplot</summary>

| X-axis | Y-axis | Label |
| --- | --- | --- |
| 0.1 | 0.1 | 0.1 (25% CI: 0.75–1.75) |
| 0.2 | 0.2 | 0.2 (25% CI: 0.75–1.75) |
| 0.3 | 0.3 | 0.3 (25% CI: 0.75–1.75) |
| 0.4 | 0.4 | 0.4 (25% CI: 0.75–1.75) |
| 0.5 | 0.5 | 0.5 (25% CI: 0.75–1.75) |
| 0.6 | 0.6 | 0.6 (25% CI: 0.75–1.75) |
| 0.7 | 0.7 | 0.7 (25% CI: 0.75–1.75) |
| 0.8 | 0.8 | 0.8 (25% CI: 0.75–1.75) |
| 0.9 | 0.9 | 0.9 (25% CI: 0.75–1.75) |
| 1.0 | 1.0 | 1.0 (25% CI: 0.75–1.75) |
| 1.1 | 1.1 | 1.1 (25% CI: 0.75–1.75) |
| 1.2 | 1.2 | 1.2 (25% CI: 0.75–1.75) |
| 1.3 | 1.3 | 1.3 (25% CI: 0.75–1.75) |
| 1.4 | 1.4 | 1.4 (25% CI: 0.75–1.75) |
| 1.5 | 1.5 | 1.5 (25% CI: 0.75–1.75) |
| 1.6 | 1.6 | 1.6 (25% CI: 0.75–1.75) |
| 1.7 | 1.7 | 1.7 (25% CI: 0.75–1.75) |
| 1.8 | 1.8 | 1.8 (25% CI: 0.75–1.75) |
| 1.9 | 1.9 | 1.9 (25% CI: 0.75–1.75) |
| 2.0 | 2.0 | 2.0 (25% CI: 0.75–1.75) |
| 2.1 | 2.1 | 2.1 (25% CI: 0.75–1.75) |
| 2.2 | 2.2 | 2.2 (25% CI: 0.75–1.75) |
| 2.3 | 2.3 | 2.3 (25% CI: 0.75–1.75) |
| 2.4 | 2.4 | 2.4 (25% CI: 0.75–1.75) |
| 2.5 | 2.5 | 2.5 (25% CI: 0.75–1.75) |
| 2.6 | 2.6 | 2.6 (25% CI: 0.75–1.75) |
| 2.7 | 2.7 | 2.7 (25% CI: 0.75–1.75) |
| 2.8 | 2.8 | 2.8 (25% CI: 0.75–1.75) |
| 2.9 | 2.9 | 2.9 (25% CI: 0.75–1.75) |
| 3.0 | 3.0 | 3.0 (25% CI: 0.75–1.75) |
| 3.1 | 3.1 | 3.1 (25% CI: 0.75–1.75) |
| 3.2 | 3.2 | 3.2 (25% CI: 0.75–1.75) |
| 3.3 | 3.3 | 3.3 (25% CI: 0.75–1.75) |
| 3.4 | 3.4 | 3.4 (25% CI: 0.75–1.75) |
| 3.5 | 3.5 | 3.5 (25% CI: 0.75–1.75) |
| 3.6 | 3.6 | 3.6 (25% CI: 0.75–1.75) |
| 3.7 | 3.7 | 3.7 (25% CI: 0.75–1.75) |
| 3.8 | 3.8 | 3.8 (25% CI: 0.75–1.75) |
| 3.9 | 3.9 | 3.9 (25% CI: 0.75–1.75) |
| 4.0 | 4.0 | 4.0 (25% CI: 0.75–1.75) |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
</details>

Figure 13. PCA of colon-token activations across all 36 layers. Panel-position clusters emerge in the middle layers and dissolve in late layers.

![](images/c96d19038cf4a86592ec4a7f63bdb232f4cabcfd15e89e68e092fea519daca93.jpg)

<details>
<summary>line chart</summary>

| Layer | Silhouette Score |
|-------|------------------|
| 0     | 0.00             |
| 5     | 0.00             |
| 10    | 0.01             |
| 15    | 0.05             |
| 20    | 0.25             |
| 25    | 0.20             |
| 30    | 0.15             |
| 35    | 0.10             |
</details>

Figure 14. Silhouette score by layer. Cluster quality peaks at layer 23, matching the effective steering range.

Fig. 16 reveals where these high-scoring heads reside: gaze heads concentrate in a narrow band of middle-to-late layers (approximately layers 20–28), with the highest density around layers 21–25. Early layers $( \ell ~ < ~ 1 5 )$ contain virtually no gaze heads, while late layers (ℓ > 30) contain a few weak ones. This spatial clustering aligns with the layerlevel steering results in Sec. 4, where the same middle-layer band was identified as the locus of visual attention control.

## C.2. Reverse Narration Trajectory

To confirm that gaze heads track the narrated panel rather than following a fixed left-to-right spatial bias, we prompt the model with “Please describe what happens in each panel, in reverse order:” and record the same per-head attention trajectories as in Sec. 5.2.

Fig. 17 shows the result. The top-100 gaze heads produce a clear reverse staircase: attention begins on panel 6 and steps backward through each panel as the model describes them from right to left. The pattern is a mirror image of the forward staircase in Fig. 2, confirming that gaze heads dynamically follow the narration order rather than defaulting to a fixed spatial scan.

![](images/bc7f6821b230cf7923dd21d68725bb72c5135ebad3ba3087dff969fc3f3a6122.jpg)

<details>
<summary>bar chart</summary>

| Gaze Score Range | Number of Heads |
| ---------------- | --------------- |
| 0.1 - 0.15       | 700             |
| 0.15 - 0.2       | 50              |
| 0.2 - 0.25       | 20              |
| 0.25 - 0.3       | 10              |
| 0.3 - 0.35       | 5               |
| 0.35 - 0.4       | 2               |
| 0.4 - 0.45       | 1               |
| 0.45 - 0.5       | 0               |
| 0.5 - 0.55       | 0               |
| 0.55 - 0.6       | 0               |
| 0.6 - 0.65       | 0               |
| 0.65 - 0.7       | 0               |
</details>

Figure 15. Gaze score histogram. Distribution of gaze scores for all 1,152 heads. Most heads score near zero (no image-token attention); the top-100 cutoff (dashed red) isolates the tracking heads.

![](images/393fe799ad7d57713aac4ab5ed40b84bda6ef2c74ca8007aab695c47e4a4635e.jpg)

<details>
<summary>heatmap</summary>

| Head Index | Mean gaze score | Gaze Score |
| ---------- | -------------- | ---------- |
| 0          | 0.0            | 0.0        |
| 5          | 0.1            | 0.3        |
| 10         | 0.2            | 0.4        |
| 15         | 0.3            | 0.5        |
| 20         | 0.4            | 0.6        |
| 25         | 0.5            | 0.7        |
| 30         | 0.6            | 0.8        |
</details>

Figure 16. Gaze scores across layers and heads. Left: Each cell shows the gaze score for one (layer, head) pair; white dots mark the top-100 heads, which cluster in layers 20–28. Right: Per-layer summary showing mean gaze score (blue) and number of top-100 heads per layer (red).

## D. Steering: Extended Analysis

## D.1. Intervention-Strength Ablation

The main-text experiments use δ = +∞ for the attentionmask bias (implemented as 10,000, which saturates the softmax in bfloat16) and so produce a hard reassignment of each head’s image-attention onto the target panel (Sec. 6.1). A natural reviewer question is whether the hard limit is necessary or whether softer values of δ also redirect the model. We sweep $\delta \in \{ 1 , 1 0 , 1 0 0 , 1 , 0 0 0 , 1 0 , 0 0 0 \}$ on Qwen3-VL-8B at K=100 (top-100 gaze heads, 500 strips ×6 target panels = 3,000 pairs each, non-gaze sampled from layers 20–35 excluding the gaze head set), holding every other detail of the intervention fixed.

The redirection result is robust to intervention strength. The sweep shows a sharp transition between δ=1 and δ=10:

![](images/eac45671882cffe0810c90f9ee3798324a33a14e5f412ab9cdd832e9a7e39001.jpg)

<details>
<summary>heatmap</summary>

| Panel | Subsummed Raw Attention Scores |
|-------|----------------------------------|
| P6    | 50                               |
| P5    | 40                               |
| P4    | 30                               |
| P3    | 20                               |
| P2    | 10                               |
| P1    | 0                                |
</details>

Figure 17. Gaze-head attention during reverse free narration. Top: The top-100 gaze heads show a reverse staircase aligned with the narrative: attention shifts panel-by-panel as the model describes each panel. Bottom: 100 random non-gaze heads show no panel-tracking structure. Dashed lines mark the transition points between panel descriptions at generation.

Table 4. Intervention-strength (δ) ablation on Qwen3-VL-8B at $K { = } 1 0 0$ . Larger δ corresponds to a harder reassignment of attention onto the target panel; δ = 10,000 saturates the bfloat16 softmax and corresponds to the δ = +∞ limit used throughout the main text. All numbers are forced 1-of-6 LLM-judge accuracy on the 500 strips × 6 target panels = 3,000 pairs validation set.

<table><tr><td>δ</td><td>Gaze (%)</td><td>Non-gaze (%)</td><td>Delta (pp)</td></tr><tr><td>1</td><td>33.2%</td><td>4.5%</td><td>28.7</td></tr><tr><td>10</td><td>79.8%</td><td>13.8%</td><td>66.0</td></tr><tr><td>100</td><td>79.8%</td><td>15.2%</td><td>64.6</td></tr><tr><td>1,000</td><td>76.8%</td><td>13.5%</td><td>63.3</td></tr><tr><td>10,000 (used)</td><td>83.1%</td><td>14.6%</td><td>68.5</td></tr></table>

at δ=1 the bias barely moves the softmax (gaze 33.2%, non-gaze 4.5%, a 7.4× gap but only a partial steer); from δ=10 onward the curve stays within ∼ 6 pp of the hard δ = +∞ limit across three orders of magnitude (δ=10: 79.8%; δ=100: 79.8%; δ=1000: 76.8%; δ=10000: 83.1%), with the maximum at the hard limit. The main-text headline is therefore not an artifact of the hard ±∞ saturation limit, since a far softer intervention (δ=10, well short of the bfloat16 saturation point) lands within 3–4 pp of it. We use $\delta = + \infty$ throughout the paper because it admits a clean closed-form description and gives the cleanest gaze accuracy, but any $\delta \geq 1 0$ produces effective redirection. The non-gaze control stays at ∼ 4–15% across the entire sweep, confirming that the transition is in head-targeted reassignment rather than a generic “stronger intervention ⇒ higher accuracy” effect.

## D.2. Head-Selection Baselines: Protocol

The Image Heads [12] and Localization Heads [19] baselines both publicly release LLaVA-specific implementations of their head-selection criteria (https://github.

com / Deng - Jingyuan / MaskCD and https : / / github . com / seilk / LocalizationHeads, respectively). We port each criterion to Qwen3-VL-8B by faithfully reproducing the head-selection algorithm rather than the LLaVA-specific glue. Concretely:

Image Heads (MaskCD). The MaskCD inference code sums each head’s attention mass over the image-token region in a single forward pass, then z-score normalizes within each layer; heads whose per-layer z-score exceeds 2.5 are flagged as “image heads.” For a top-K ranking we sort all 1,152 Qwen3-VL-8B heads by their (mean over the 500 discovery panel-query prompts) per-layer z-score and take the top K. This reproduces MaskCD’s intent on our setting: image-attending heads are those that put outlying attention on image tokens within their layer.

Localization Heads (Kang et al.). Their analysis pipeline (analyze.py) ranks heads by two criteria. (1) “Criterion-1” picks heads whose imageattention sums lie above an elbow threshold (chorddistance method, also defined in analyze.py). (2) “Criterion-2” computes the spatial entropy of each head’s 2D attention map (spatial entropy(attn map 2d, threshold) in their code) on the $P { \times } P$ patch grid; heads with low entropy are more spatially concentrated. They additionally drop heads in layer ≤ 1 and heads whose attention concentrates on the bottom row (a LLaVA-specific summary-token filter). Heads that pass both criteria are ranked by ascending spatial entropy. We port the algorithm exactly: collect the per-(layer, head, patch) attention from the final prompt token on each of the 6 panelquery prompts across the 500 discovery strips, average to a (n layers × n heads × P 2) tensor with $P { = } 2 4$ , then run their unchanged analyze heads(...) on it (porting only the necessary config keys, not the algorithm). Layerskip and bottom-row-focus filters are kept as in the original code.

Non-gaze control. The non-gaze control samples 100 heads uniformly from layers 20–35 of the model, excluding the gaze head set, and is identical across all rows of the baseline table; this gives an apples-to-apples “what if we just intervened on heads in the same layer band that aren’t gaze heads” control regardless of which positive-head criterion we are evaluating.

Intervention. Identical across rows: the same boostsuppress attention-mask intervention used for our gaze heads $\begin{array} { r } { ( \delta = + \infty , } \end{array}$ , hard reassignment of each head’s imageattention onto the target panel; Sec. 6.1).

## D.3. Head-Selection Baselines: Full K Sweep

Tab. 5 reports the full top-K sweep for our gaze score, the Image Heads selection [12], and the Localization Heads selection [19] on Qwen3-VL-8B (500 strips × 6 target panels = 3,000 pairs, non-gaze sampled from layers 20–35 excluding the gaze head set). For both baselines we reimplement their published head-identification algorithms faithfully (protocol in Sec. D.2) and apply them to the same 500-strip discovery set. The intervention is identical across rows (the same boost-suppress attention-mask edit used for our gaze heads). The gaze score is the most headefficient criterion at small K, where head-identification quality dominates the result; at large K every reasonable image-attention criterion eventually saturates the intervention.

Tab. 6 measures how much the three criteria agree on which heads they select, at both the K=10 and K=100 cuts.

The picture is sharpest at small K. At K=10 the three criteria pick essentially disjoint sets: the gaze score shares one head with Localization Heads, none with Image Heads, and the two baselines share none with each other, against ≈ 0.1 heads expected by chance. At small K, where headselection quality dominates redirection accuracy (Tab. 5), the three criteria do not even agree on which heads matter most. At K=100 the overlap rises into the 26–43 range, three to five times the ≈ 8.7-head chance level, but each criterion’s set is still mostly its own and only 13 heads pass all three filters. The redirection gap in Tab. 5 is therefore not a matter of the same heads re-ranked: the temporal re-routing criterion behind the gaze score selects heads the single-pass image-attention criteria miss, and those are the heads that move the small-K accuracy.

## D.4. Prompt Sensitivity

We test whether the VQA-redirection result depends on the exact wording of the question. Tab. 7 reports redirection accuracy under five prompt variants on Qwen3-VL-8B at K=100 on the 500-strip validation set (n=3,000 pairs each). The default paper prompt is the first row; the other four cover shorter, longer, and reframed wordings of the same question. All variants use the same forced 1-of-6 LLM judge.

Across the five prompt phrasings the gaze-redirection accuracy ranges from 64.7% to 83.1% (an 18.4-pp span), with the non-gaze control staying near chance (∼ 7–16%) on every variant. The redirection effect is robust to prompt wording: every variant produces a ≥ 50-pp gap between gaze and non-gaze, far larger than the prompt-induced variation.

## D.5. Dynamic Narration: Trajectory-Level Judge

The strict 1-of-6 judge used for the main-text headline (Sec. 6.2) penalizes any segment whose dominant content is not the scheduled target panel by exactly that segment’s boundary, including segments that legitimately finish the previous panel’s sentence before transitioning. To check that the headline result is not an artifact of a strict judge, we rejudge the same generations with a trajectory-level LLM judge that hides the schedule entirely (Tab. 8). The judge sees the strip image and all six 50-token segments at once, and is asked, for each segment, to identify which panel of the strip its content dominantly describes. Repeats are allowed (the model may revisit a panel) and a segment whose content is incoherent or empty is mapped to null. From the six attributions per (strip, condition) we compute Spearman ρ of the predicted-panel sequence against (i) the steering schedule and (ii) the natural [1,2,3,4,5,6] order, plus persegment match rates for each.

The trajectory judge agrees directionally with the strict 1-of-6 judge: gaze tracks the schedule at 71.1% persegment match and $\rho { = } \mathrm { + 0 . 5 8 9 }$ , vs. non-gaze at 16.4% and $\rho { = } { - } 0 . 1 4 0$ . The new information is the ρ vs. natural column: under gaze redirection, the model’s predicted-panel sequence has only ρ=+0.259 correlation with the default left-to-right [1, 2, 3, 4, 5, 6] order, while the non-gaze control sits at $\rho { = } \mathrm { + 0 . 9 8 3 }$ , essentially the perfect default scan. The two-column comparison shows that gaze heads do not just disrupt the default scan; they replace it with the steered schedule.

†Spearman ρ is not reported for the all-heads condition because the junk fraction is too high (384/500 strips with $\geq 3$ null segments), leaving too few intact trajectories to attribute reliably.

## D.6. Gaze Heads on Natural Images

The experiments in the main paper use comic strips, where visual content is divided into discrete panels. Do gaze heads also perform spatial grounding on natural images, where regions are not explicitly delineated?

Setup. We prompt Qwen3-VL-8B with a natural image and the instruction “Describe what is happening in this image in detail:” The model generates a free-form description of approximately 300 tokens. We capture value-weighted attention scores [20] from the top-100 gaze heads (discovered via the comic panel task) at every decode step during a single generation pass.

Concept segmentation. We segment the generated text into spatial region spans using Claude Sonnet [2], where each span covers the tokens corresponding to a single described region (e.g., “notebooks and pen” = tokens 48– 97, “headphones” = tokens 135–156). This gives us token ranges indicating when the model is describing each part of the image.

Table 5. Comparison of head-selection criteria for the same attention-mask intervention. Gaze-redirection accuracy on Qwen3-VL-8B; 500 strips × 6 target panels = 3,000 pairs each. The non-gaze control samples 100 heads uniformly from layers 20–35 excluding the gaze head set, identical across all rows for direct comparability.

<table><tr><td>K</td><td>Ours (gaze)</td><td>Image Heads [12]</td><td>Localization Heads [19]</td><td>Non-gaze</td><td>All-heads</td></tr><tr><td>10</td><td>45.3</td><td>10.0</td><td>24.3</td><td>9.8</td><td>0.9</td></tr><tr><td>50</td><td>63.2</td><td>36.2</td><td>41.7</td><td>15.7</td><td>0.9</td></tr><tr><td>100</td><td>83.1</td><td>69.0</td><td>60.2</td><td>14.6</td><td>0.9</td></tr><tr><td>150</td><td>75.6</td><td>60.8</td><td>59.5</td><td>13.8</td><td>0.9</td></tr><tr><td>200</td><td>73.0</td><td>58.2</td><td>58.8</td><td>13.5</td><td>0.9</td></tr></table>

Table 6. Agreement between head-selection criteria on Qwen3-VL-8B (1,152 heads total). For each pair we report the overlap of their top-K heads (out of K) and the Jaccard index, at K=10 and K=100. A random pair of size-K sets would overlap ≈ 0.1 heads at K=10 and ≈ 8.7 at K=100. The three criteria are correlated in their top-100 but nearly disjoint in their top-10, and only 13 heads are shared by all three even at K=100.

<table><tr><td rowspan="2">Head-set pair</td><td colspan="2">Top-10</td><td colspan="2">Top-100</td></tr><tr><td>Overlap</td><td>Jaccard</td><td>Overlap</td><td>Jaccard</td></tr><tr><td>Gaze ∩ Image Heads [12]</td><td>0/10</td><td>0.000</td><td>43/100</td><td>0.274</td></tr><tr><td>Gaze ∩ Localization Heads [19]</td><td>1/10</td><td>0.053</td><td>26/100</td><td>0.149</td></tr><tr><td>Image Heads ∩ Localization Heads</td><td>0/10</td><td>0.000</td><td>33/100</td><td>0.198</td></tr><tr><td>All three (common core)</td><td>0/10</td><td>0.000</td><td>13/100</td><td>0.062</td></tr></table>

Table 7. Prompt-sensitivity of gaze-head redirection on Qwen3- VL-8B. Top-100 gaze heads, boost-suppress intervention, 500 strips × 6 target panels = 3,000 pairs each.

<table><tr><td>Prompt</td><td>Gaze</td><td>Non-Gaze</td></tr><tr><td>What is the main action or event happening in this comic strip? Answer briefly. (default)</td><td>83.1%</td><td>14.6%</td></tr><tr><td>Describe this comic briefly.</td><td>64.7%</td><td>13.3%</td></tr><tr><td>Look at this comic strip carefully.</td><td>80.3%</td><td>7.3%</td></tr><tr><td>What is the main action or event happening across the panels? Answer in one short sentence.</td><td></td><td></td></tr><tr><td>What story does this comic strip tell? Answer briefly.</td><td>74.2%</td><td>13.3%</td></tr><tr><td>Summarize the main event of this comic in a few words.</td><td>78.3%</td><td>16.2%</td></tr></table>

Heatmap construction. For each concept span, we average the per-token image-attention vectors across all decode steps in that span and across all heads in the set (gaze or random). This produces a 1D vector over image tokens, which we reshape to the spatial grid matching the image’s token layout. The resulting heatmap is upsampled and overlaid on the original image using a jet colormap.

Findings. Fig. 18, Fig. 19, and Fig. 20 show the results across three natural images. Gaze heads produce spatially concentrated attention that tracks the described region: when the model describes “notebooks and pen,” attention concentrates on the lower-left of the image where these objects are located, and when it describes “succulent plant and pot,” attention shifts to the upper-right. The grounding is less precise than on panel-structured images, since natural images have no explicit region boundaries, but the correspondence between described content and attended region is consistent. This suggests that the gaze heads discovered through comic strip probing are not specific to panelstructured images: they perform a general spatial grounding function, attending to the region of the image the model is currently describing.

Quantitative natural-image redirection (COCO val2017). To put a number on the natural-image steering claim, we evaluate gaze-head redirection on all 5,000 images in COCO val2017 [22], sweeping over the 31,781 annotated objects in the set (8,897 large, 12,569 medium, 10,315 small across 80 categories). For each (image, object) pair we apply a minimum bounding-box area filter of 750 pixels: smaller boxes cover only a handful of image tokens in Qwen3-VL-8B’s input grid, and the additive attention bias struggles to redirect attention onto a region that contains so few tokens. The filter is essentially a no-op for the large and medium classes but removes most of the small class, leaving an evaluation sample of 23,452 pairs: 8,897 large, 12,569 medium, and 1,986 small. For each pair we steer the top-100 gaze heads to the target object’s COCO bounding box (mapping pixel coordinates to image-token positions via cell-center containment) and ask the model “What is the main object in this part of the image? Answer in a few words.” Claude Sonnet judges whether the steered answer names the target COCO category, with synonym matching (e.g., “car” ∼ “automobile”); see Sec. A. The non-gaze control samples 100 heads from layers 20–35, excluding the gaze head set (Sec. A).

Table 8. Trajectory-level rejudge of the dynamic-narration generations. 500 strips, strict derangement schedule. Judge does not see the schedule. Per-segment matches are computed only over non-null segments; junk counts list how many strips have at least 3 null segments (and so do not contribute to Spearman ρ).

<table><tr><td>Condition</td><td>Match vs. schedule</td><td>Match vs. natural</td><td> $\rho$  vs. schedule</td><td> $\rho$  vs. natural</td></tr><tr><td>Gaze (top-100)</td><td>71.1%</td><td>13.6%</td><td>+0.589</td><td>+0.259</td></tr><tr><td>Non-gaze</td><td>16.4%</td><td>20.8%</td><td>-0.140</td><td>+0.983</td></tr><tr><td>All-heads</td><td>7.1%</td><td>7.5%</td><td> $-^{\dagger}$ </td><td> $-^{\dagger}$ </td></tr></table>

Tab. 1 reports accuracy broken down by COCO size class. Gaze-head steering achieves 80.3% on large objects $\mathrm { ( > 9 6 ^ { 2 } ~ p x ) }$ , where the object occupies enough image tokens for the bounding-box bias to bite cleanly, and 76.2% on medium objects, both well above the non-gaze control (18.6–36.6% depending on size class). Performance drops on small objects whose bounding boxes cover only a handful of image tokens and where the natural attention bias dominates over our intervention. This converts the qualitative natural-image observation in Fig. 18 into a quantitative claim: the same heads identified through comic probing also steer the model’s answer toward an arbitrary spatial region of a natural image.

## E. Generalization Across Sizes and Architectures

## E.1. Generalization Across Model Sizes

We run the full pipeline on four Qwen3-VL sizes: 2B (28 layers, 16 heads), 4B (36 layers, 32 heads), 8B (36 layers, 32 heads), and 32B (64 layers, 64 heads).

Layer steering. Fig. 21 compares per-layer flip rates across all four sizes. The 4B and 8B models, which share the same depth (36 layers), both localize visual attention control in layers 20–28, with best flip rates of 98.3% (L22) and 97.0% (L21). The 32B model (64 layers) places its effective band deeper at layer 49, achieving 85%. The 2B model (28 layers) is the clear outlier: its best layer (L19) achieves only 10%, suggesting insufficient capacity for a robust gaze mechanism amenable to difference-of-means steering.

Gaze discovery and redirection. We apply the same gaze-head discovery process to all four sizes and evaluate the discovered heads on VQA redirection across the full top-K sweep, using the 500-strip validation set and forced 1-of-6 LLM judge of the main text. Tab. 9 reports gaze / non-gaze accuracy at each K. All four sizes follow a humpshaped curve with a clear single peak, and the non-gaze control stays at $\leq 1 5 \%$ throughout. The 8B model achieves the highest peak (83.1% at K=100, reproducing the main-text VQA headline); the 4B model peaks at 72.9% at K=75, the 32B model at 70.2% at K=500 after a longer climb, and the 2B model earliest at 68.6% at K=10. The peak K scales roughly with total head count: 2B (448 heads), 4B / 8B (1,152 heads), and 32B (4,096 heads) peak at K values corresponding to ∼ 2%, ∼ 7–9%, and ∼ 12% of all heads. Fig. 22 plots these saturation curves.

Table 9. Full top-K gaze / non-gaze VQA accuracy across Qwen3- VL sizes. 500-strip validation set, forced 1-of-6 LLM judge (chance 16.7%); non-gaze heads are sampled from the same layer range as each model’s gaze heads, excluding the gaze head set. Each cell is gaze / non-gaze percent. Dashes indicate K values not in the per-size sweep.

<table><tr><td>K</td><td>2B</td><td>4B</td><td>8B</td><td>32B</td></tr><tr><td>5</td><td>65.0 / 13.0</td><td>18.9 / 8.6</td><td>36.0 / 6.6</td><td>11.7 / 2.7</td></tr><tr><td>10</td><td>68.6 / 1.3</td><td>49.7 / 10.5</td><td>45.3 / 10.0</td><td>16.2 / 3.5</td></tr><tr><td>50</td><td>48.6 / 0.5</td><td>68.2 / 4.3</td><td>63.2 / 15.3</td><td>30.3 / 7.3</td></tr><tr><td>75</td><td>—</td><td>72.9 / 1.5</td><td>76.3 / 13.5</td><td>—</td></tr><tr><td>100</td><td>46.7 / 0.4</td><td>70.3 / 1.6</td><td>83.1 / 14.6</td><td>32.5 / 8.3</td></tr><tr><td>125</td><td>—</td><td>68.3 / 1.8</td><td>79.4 / 13.3</td><td>—</td></tr><tr><td>150</td><td>50.5 / 0.5</td><td>69.7 / 1.6</td><td>75.6 / 14.5</td><td>29.5 / 8.7</td></tr><tr><td>200</td><td>36.6 / 0.5</td><td>55.3 / 1.6</td><td>73.0 / 14.5</td><td>42.8 / 11.2</td></tr><tr><td>300</td><td>—</td><td>—</td><td>—</td><td>63.5 / 12.3</td></tr><tr><td>400</td><td>—</td><td>—</td><td>—</td><td>68.8 / 12.2</td></tr><tr><td>500</td><td>—</td><td>—</td><td>—</td><td>70.2 / 12.0</td></tr><tr><td>600</td><td>—</td><td>—</td><td>—</td><td>60.2 / 12.3</td></tr><tr><td>700</td><td>—</td><td>—</td><td>—</td><td>55.2 / 12.5</td></tr><tr><td>800</td><td>—</td><td>—</td><td>—</td><td>59.7 / 11.8</td></tr><tr><td>900</td><td>—</td><td>—</td><td>—</td><td>58.8 / 12.3</td></tr><tr><td>1000</td><td>—</td><td>—</td><td>—</td><td>47.7 / 12.0</td></tr></table>

Gaze-head trajectories. Fig. 23 shows gaze-head attention trajectories during free-form narration for all four model sizes on the same strip. All four produce a clear staircase pattern, confirming that gaze heads are a consistent organizational feature across scales.

![](images/03b1a30b44ae4124050d2b648557a9faed6da847afa601800d87aea364898d9a.jpg)  
Figure 18. Gaze-head attention on a natural image. Leftmost: the original image. Remaining panels: gaze-head attention heatmaps for three concept spans during free-form description. Attention shifts to the spatial region corresponding to each described object, confirming that gaze heads perform spatial grounding beyond comic panels.

![](images/39a411a6ca238834dc5aba09e9594d1cd15abeff8e7c7abe85670a4df8fc4e30.jpg)

<details>
<summary>natural_image</summary>

Outdoor garden scene with a wooden bicycle, potted orchid, and red watering can on a patterned rug (no text or symbols visible)
</details>

"wooden bench with books" [tokens 62-119  
![](images/92fbbc341b8f17ade36d47e6ab58240ebe96714c662709fc988b95e150fb1819.jpg)

<details>
<summary>natural_image</summary>

Night outdoor scene with a wooden bicycle, potted flowers, and a red watering can on a mat (no visible text or symbols)
</details>

"potted orchid area" [tokens 120-145]  
![](images/bbb6bdc7abcdc84ee5a9d3591ea7591d2c8d9a90635abeb422fdc97d17eafcd7.jpg)

<details>
<summary>natural_image</summary>

Night scene with a vintage bicycle, a rock, and a watering can in the foreground (no visible text or symbols)
</details>

"rug area with cat" [tokens 146-208]  
![](images/9441a8d4f497ad022374b3bc71a8352d640329c10cb4f6e92844b37864f8d14f.jpg)

<details>
<summary>natural_image</summary>

Outdoor garden scene with a vintage bicycle, potted flowers, and a small red watering can under blue lighting (no text or symbols visible)
</details>

Figure 19. Gaze-head attention on a natural image. Leftmost: the original image. Remaining panels: gaze-head attention heatmaps for three concept spans during free-form description. Attention shifts to the spatial region corresponding to each described object, confirming that gaze heads perform spatial grounding beyond comic panels.

Gaze-steered narration. Fig. 24 shows baseline versus steered Spearman ρ and starts-correct rate, using an older pilot protocol (per-strip “first segment matches target” rate alongside ρ between the steered narration order and the target schedule, on a smaller cross-size batch). Gaze-steered narration produces positive ρ on all four sizes: 2B (ρ = +0.61, 34% starts-correct), 4B $( \rho ~ = ~ + 0 . 4 5 , ~ 6 8 \% )$ ), 8B $( \rho \ : = \ : + 0 . 6 2 , \ : 6 8 \% )$ , and 32B (ρ = +0.53, 66%). The 8B model achieves the strongest correlation overall. The 2B narration result is surprisingly strong given its weak layer steering, possibly because gaze-head redirection intervenes more directly on the attention routing mechanism than residual-stream steering. The headline narration result in the main text (79.4% static narration steering on the 8B model) is not directly comparable to these starts-correct numbers; what is consistent across protocols is that the 8B model shows the strongest gaze-steered narration effect.

## E.2. Cross-Architecture Generalization

Tab. 2 in the main text reports the peak redirection accuracy for each other architecture we tested: Ovis1.5-8B [25], Qwen2-VL-7B [38], InternVL3.5-8B [39], LLaVA-1.5-

13B [23], and LLaVA-NeXT-7B [24]. The Qwen3-VL-8B row in that table is the headline result; all rows are evaluated on the full 500-strip validation set. This appendix documents the preprocessing fix that makes the cross-family transfer possible, the full K-sweep behind those peak numbers, and a within-family scale comparison.

Preprocessing. The fixed-resolution families (Ovis, InternVL, and both LLaVA variants) wrap a center-cropping image processor that, on our wide strips, would crop away five of the six panels before they reach the language model. We apply a one-line panel-preservation fix (full details and validation in Sec. E.3) so all six panels remain visible; without it, redirection on these families is indistinguishable from chance. Qwen2-VL, like Qwen3-VL, uses an aspect-ratiopreserving processor and needs no fix.

## E.3. Cross-Architecture Preprocessing Fix

The center-crop bug. The fixed-resolution families wrap a CLIP-style image processor whose default behavior for a non-square input is: (1) resize so that the shortest side equals the model’s crop size S (S=336 for LLaVA-1.5 and LLaVA-NeXT, S=448 for InternVL, S=384 for Ovis), then (2) center-crop to $S \times S$ . For our 1536 × 256 comic strips this scales the shortest side (256) up to S, which blows the longest side up to ∼ 6S. Center-cropping then leaves only the middle S pixels, about one panel out of the six. The other five panels are simply not in the model’s input.

![](images/a60629951b24c94ff5b5912842159541e194eec426d5b73b0e54683c4b07a952.jpg)  
Figure 20. Gaze-head attention on a natural image. Leftmost: the original image. Remaining panels: gaze-head attention heatmaps for three concept spans during free-form description. Attention shifts to the spatial region corresponding to each described object, confirming that gaze heads perform spatial grounding beyond comic panels.

![](images/0dd18df80895e13501366deb7c7a2d953ef35d92f32a1892fa678e6618420d9f.jpg)

<details>
<summary>line chart</summary>

| Layer | 2B   | 4B   | 8B   | 32B  |
|-------|------|------|------|------|
| 0     | 0.0  | 0.0  | 0.0  | 0.0  |
| 10    | 0.0  | 0.0  | 0.0  | 0.0  |
| 20    | 0.1  | 1.0  | 1.0  | 0.0  |
| 30    | 0.1  | 0.9  | 1.0  | 0.0  |
| 40    | 0.0  | 0.1  | 0.0  | 0.0  |
| 50    | 0.0  | 0.0  | 0.1  | 0.85 |
| 60    | 0.0  | 0.0  | 0.1  | 0.1  |
</details>

Figure 21. Layer steering across model sizes. The 4B and 8B models localize visual attention control in layers 20–28; the 32B model places it near layer 49. The 2B model shows minimal steering effect at any layer.

![](images/5afe27ef54d3a8687693755867a220915506ce7ee3a4f7134a728de4f4dfaf55.jpg)

<details>
<summary>line chart</summary>

| Number of Patched Heads | 2B    | 4B    | 8B    | 32B   |
| ----------------------- | ----- | ----- | ----- | ----- |
| 1                       | 0.15  | 0.15  | 0.15  | 0.15  |
| 10                      | 0.55  | 0.55  | 0.25  | 0.20  |
| 100                     | 0.60  | 0.70  | 0.80  | 0.90  |
| 1000                    | 0.60  | 0.70  | 0.85  | 0.95  |
</details>

Figure 22. Gaze redirection accuracy across model sizes. All four Qwen3-VL sizes follow a hump-shaped saturation curve, with the peak K scaling with the model’s total head count (Tab. 9).

Fix. Before passing the image to the processor we resize the strip directly to $S \times S$ (a fixed-size square; no centercrop). This is a horizontal squash (panel widths drop from 256 px to $S / 6 \ \mathrm { p x } )$ , but every panel is preserved as a contiguous column of image tokens in the LM input. The effect is decisive: on LLaVA-1.5 the top gaze score’s discovered heads come to correspond to all six panels rather than the center crop alone, and redirection on the validation set rises from 17.8% (near chance) to 24.7% at K=100, with a peak of 39.0% at $K { = } 1 6 0$ . The other fixed-resolution families behave the same way; their peak accuracies with the fix in place are reported in Tab. 2.

Implementation. In utils/modeling.py, prepare inputs detects when the loaded processor is a CLIPImageProcessor or an InternVL / SigLIP-style processor and pre-resizes image.resize((S, S), BILINEAR) before building the prompt. For Qwen2-VL and Qwen3-VL the call is a no-op because their processors are aspect-ratio-preserving by construction (they tile patches to match the input’s true aspect). Ovis additionally exposes its visual block through an out-of-band placeholder rather than ordinary text tokens, so we record the placeholder’s expanded span at merge time to recover the image-token range. The fix and these per-family hooks add ∼ 15 lines of code each, cost nothing at run time, and change no model weights.

What is squashed. The squashed strip looks compressed horizontally to humans, but the language model still sees every panel as a separate contiguous block of image-token columns: 24 wide → 4 cols/panel for LLaVA-1.5 and the LLaVA-NeXT base tile (24×24=576 tokens), 16 wide → 2–3 cols/panel for InternVL3.5 (16×16=256 tokens after its 2×2 pixel-shuffle), and 27 wide → 4–5 cols/panel for Ovis (27×27=729 tokens plus two visual-indicator tokens that we exclude from panel scoring). LLaVA-NeXT additionally appends an any-resolution tile whose row-end image newline tokens we mask out. The panels are visually narrow but their image-token representations are clean.

## E.4. Cross-Architecture: Extended Sweeps

Tab. 10 reports the full top-K sweep, gaze (g) vs. non-gaze (n) accuracy at each K, for the five other-architecture families on the 500-strip validation set, all under the same discovery score and intervention as the main text (Sec. E.2). All five show the same hump-and-collapse shape: a mid-K peak, then a collapse into degenerate output as the −δ over-suppresses (junk% column). The all-heads condition is omitted because it stays at ≤ 3% throughout (peak-K all-heads numbers are in Tab. 2).

![](images/1efd6d7200f3a5e6fb3e3e3a1a9669ca9aea209baac8a017e6186c2da87e6d38.jpg)  
Figure 23. Gaze-head attention trajectories across model sizes. All four Qwen3-VL sizes (2B, 4B, 8B, 32B) produce a staircase pattern during free-form narration on the same strip, confirming that gaze heads consistently track the narrated panel across scales.

![](images/e2f1d0c073832ff80235f990fd349636e003ee14a7c2258fdfc728e6babea69b.jpg)

<details>
<summary>bar chart</summary>

| Narration Order | Baseline Mean Spearman ρ | Steered Mean Spearman ρ | Baseline Starts-Correct Rate | Steered Starts-Correct Rate |
| :--- | :--- | :--- | :--- | :--- |
| 2B | 0.0 | 0.61 | 0.2 | 0.35 |
| 4B | 0.03 | 0.45 | 0.2 | 0.68 |
| 8B | 0.01 | 0.67 | 0.2 | 0.74 |
| 32B | 0.0 | 0.53 | 0.23 | 0.66 |
</details>

Figure 24. Gaze-steered narration across model sizes. All four sizes produce positive steered ρ, with 8B achieving the strongest correlation.

Saturation behavior. Every other-architecture family shows the same hump-and-collapse shape as Qwen3-VL-8B (Fig. 8; Fig. 26 plots the per-family curves), but the peak location and the collapse rate vary. Ovis1.5-8B peaks sharply at K=100 (68.7%) and then collapses hard: by K=250 the intervention drives over 80% of outputs to junk and accuracy falls to chance. Qwen2-VL-7B peaks earlier and more gently, at K=90 (66.2%, between the grid points of Tab. 10), and degrades slowly rather than collapsing. InternVL3.5-8B and LLaVA-1.5-13B peak further out: InternVL3.5-8B reaches 62.7% at K=140 after a steady climb from K=120, then eases back to 55.0% by K=180. LLaVA-1.5-13B peaks at K=160 (39.0%), holding a flat 32–39% band from K=125 to K=175. LLaVA-NeXT-7B peaks at K=100 (35.3%) and collapses past K=200.

Non-gaze controls. At each model’s peak K the non-gaze control, heads sampled from the same layer range as that model’s gaze heads (excluding the gaze head set), stays well below the matched gaze condition (Tab. 2). The level depends on what that band contains for each model: on Qwen2-VL it pins near zero (0.0%) because force-boosting non-gaze heads onto one panel collapses generation to junk on almost every pair; on Ovis (13.0%) and LLaVA-1.5 (13.8%) it sits near the 16.7% chance line; and on InternVL3.5 (31.0%) and LLaVA-NeXT (26.7%) it floats higher, because under the suppress-all intervention some panel signal leaks through even non-gaze heads. In every case the gaze condition clears the non-gaze control by a wide margin at the peak K.

Within-family scale. For the InternVL3.5 and LLaVA-1.5 families we ran both a smaller variant (2B / 7B) and a larger 8B / 13B variant; Tab. 11 reports the peak redirection accuracy for each pair with non-gaze sampled from the same layer range as the gaze heads. In both families the larger variant peaks at a larger K (K=50 → 140 for InternVL3.5, K=150 → 160 for LLaVA-1.5) at a comparable peak accuracy $( 6 4 . 0  6 2 . 7 \%$ and $3 8 . 7  3 9 . 0 \% )$ , consistent with the gaze mechanism being spread over more heads at larger scale, so that a fixed-K attention-mask intervention has to reach a larger fraction of them to achieve the same effect. Qwen3-VL remains the family with the cleanest within-family scaling, where the 8B variant is genuinely the strongest size (Sec. E.1, 8B peaks at 83.1% at K=100, above 4B’s 72.9% at K=75 and 32B’s 70.2% at K=500).

Table 10. Full top-K sweep for the five other-architecture families: gaze (g) / non-gaze (n) redirection accuracy at each K on the 500- strip validation set, with the gaze-condition junk% in the last column. The discovery score and intervention are identical across families (Sec. E.2); the in-grid maximum gaze accuracy is in bold. Exact per-model peaks fall between these grid points and are reported in Tab. 2: Qwen2-VL K=90 (66.2/0.0), InternVL3.5 K=140 (62.7/31.0), LLaVA-1.5 K=160 (39.0/13.8).

<table><tr><td>K</td><td>Ovis1.5-8B g/n</td><td>Qwen2-VL-7B g/n</td><td>InternVL3.5-8B g/n</td><td>LLaVA-1.5-13B g/n</td><td>LLaVA-NeXT-7B g/n</td><td>junk% (Ov/Qw/V3.5/L1.5/LN)</td></tr><tr><td>10</td><td>20.0 / 13.7</td><td>21.0 / 8.7</td><td>9.0 / 10.7</td><td>15.0 / 2.0</td><td>15.7 / 4.0</td><td>13 / 1 / 0 / 2 / 7</td></tr><tr><td>25</td><td>32.0 / 14.7</td><td>37.3 / 0.0</td><td>14.0 / 29.7</td><td>15.0 / 7.0</td><td>25.0 / 9.7</td><td>9 / 2 / 3 / 2 / 12</td></tr><tr><td>50</td><td>45.3 / 12.7</td><td>50.7 / 0.0</td><td>36.0 / 40.7</td><td>21.7 / 13.0</td><td>26.7 / 25.0</td><td>16 / 5 / 5 / 3 / 11</td></tr><tr><td>100</td><td>68.7 / 13.0</td><td>55.3 / 0.0</td><td>46.0 / 33.3</td><td>24.7 / 14.3</td><td>35.3 / 26.7</td><td>8 / 6 / 6 / 15 / 15</td></tr><tr><td>150</td><td>67.3 / 13.7</td><td>46.7 / 0.3</td><td>57.7 / 32.3</td><td>36.3 / 13.7</td><td>28.7 / 26.0</td><td>11 / 11 / 6 / 12 / 23</td></tr><tr><td>200</td><td>21.7 / 13.3</td><td>45.3 / 0.3</td><td>54.3 / 32.3</td><td>37.0 / 14.3</td><td>23.7 / 26.7</td><td>66 / 12 / 10 / 17 / 52</td></tr><tr><td>300</td><td>13.0 / 14.3</td><td>47.0 / 0.3</td><td>44.3 / 32.0</td><td>32.7 / 14.3</td><td>19.0 / 26.3</td><td>83 / 16 / 23 / 27 / 62</td></tr></table>

![](images/61d3bdfcf11b356292ad34144d8140406b4c5b516688f35a4aa4a87693c94d11.jpg)

<details>
<summary>violin chart</summary>

| Model           | Normalized Layer Depth |
| --------------- | ---------------------- |
| Ovis1.5-8B      | ~0.9                   |
| Qwen2-VL-7B     | ~0.6                   |
| InternVL3.5-8B  | ~0.8                   |
| LLaVA-1.5-13B   | ~0.5                   |
| LLaVA-NeXT-7B   | ~0.7                   |
</details>

Figure 25. Layer concentration of the top-100 gaze heads across the five other-architecture families. Layer indices are normalized to a depth fraction so models with different layer counts (28, 32, 36, 40) share one axis; the shaded band marks the mid-to-late region (depth 0.4–0.8) and the tick is each model’s mean depth. The top-100 gaze heads sit in the second half of every network: Qwen2-VL (28 layers) and InternVL3.5 (36) concentrate latest (mean depth ≈ 0.84), Ovis (32) and LLaVA-NeXT (32) fall in the mid-to-late band, and LLaVA-1.5 (40) is the most distributed, with a tail into the early layers. Across architectures the gazehead construct keeps a consistent geometric meaning: gaze heads are mid-to-late LM heads.

Frozen vs. trained vision encoders: an exploratory hypothesis. We present this section as a hypothesis the data is consistent with, not a conclusion. One pattern we observe is that the redirection magnitude appears to correlate with whether the vision encoder is fine-tuned together with the language model or kept frozen. The three families that exceed 60% all train the encoder on the VLM task.

![](images/f798cecd143689c3fb59cc1e8a00db4b1ea95256db8c71f48e1d2fa6e4c36048.jpg)

<details>
<summary>line chart</summary>

| Top-K redirected heads | Ovis1.5-8B | Qwen2-VL-7B | InternVL3.5-8B | LLaVA-1.5-13B | LLaVA-NeXT-7B |
| ---------------------- | ---------- | ----------- | -------------- | ------------- | ------------- |
| 0                      | 20         | 20          | 9              | 15            | 15            |
| 50                     | 45         | 50          | 35             | 22            | 25            |
| 100                    | 68         | 62          | 45             | 25            | 33            |
| 150                    | 68         | 47          | 62             | 38            | 29            |
| 200                    | 22         | 46          | 55             | 37            | 23            |
| 300                    | 13         | 47          | 44             | 33            | 19            |
</details>

Figure 26. Top-K saturation across the five other-architecture families. Gaze-redirection accuracy (forced 1-of-6 LLM judge, chance 16.7%) versus the number of redirected heads K, on the 500-strip validation set (n=3,000 per point); shaded bands are bootstrap 95% CIs. Per-model peaks are reported in Tab. 2. Every family shows the same hump-then-collapse shape but peaks in a different place: Ovis1.5-8B peaks at K=100 (68.7%) and collapses hardest as the −δ over-suppression drives outputs to junk, Qwen2-VL-7B and InternVL3.5-8B peak in the mid-60%s (at K=90 and K=140) and degrade gently, and the two frozenencoder LLaVA families plateau near 35–39% without a sharp peak.

Table 11. Within-family scale comparison. Peak gaze-redirection accuracy and the K at which it occurs, for each family’s smaller and larger variants, on the full 500-strip validation set with nongaze sampled from the same layer range as the gaze heads.

<table><tr><td>Family / scale</td><td>Peak K</td><td>Gaze</td><td>Non-gaze</td></tr><tr><td>InternVL3.5-2B</td><td>50</td><td>64.0</td><td>24.0</td></tr><tr><td>InternVL3.5-8B</td><td>140</td><td>62.7</td><td>31.0</td></tr><tr><td>LLaVA-1.5-7B</td><td>150</td><td>38.7</td><td>16.3</td></tr><tr><td>LLaVA-1.5-13B</td><td>160</td><td>39.0</td><td>13.8</td></tr></table>

Qwen2-VL learns its native dynamic-resolution ViT end to end, InternVL3.5 trains InternViT at 448 px with a learned 2×2 pixel-shuffle projector, and Ovis fine-tunes its SigLIPso400m backbone in two of its three training stages while learning the visual vocabulary and embedding table that reencode it. All three produce a sharp gaze ranking concentrated in the mid-to-late LM layers (Fig. 25), with the top-10 heads in layers 16–23 for Ovis, 19–24 for Qwen2-VL, and 24–34 for InternVL3.5. The two families that plateau near 35% both keep the encoder frozen: LLaVA-1.5 and LLaVA-NeXT bridge a frozen CLIP-ViT-L/14-336 to the LM through a two-layer MLP and never update it on the VLM task. Their strongest heads barely separate from the non-gaze control (LLaVA-NeXT: 35.3% gaze vs. 26.7% non-gaze at peak).

A natural reading is that the gaze mechanism may require image tokens that are both spatially addressable and panel-distinct, so that a compact set of heads can learn to select the tokens of the panel being described as a function of the decode query. A frozen encoder optimized for global image-text matching, passed through a thin projector, could give the LM patch features that answer coarse questions but stay too diffuse for such heads to form. This reading is correlational, since the frozen families also use lower input resolution and differ in language backbone. Two further observations are consistent with this hypothesis. Within each family, scale does not move the result, with both LLaVA-1.5 sizes plateauing near 39% and both InternVL3.5 sizes near 62–64% (Tab. 11), so capacity does not appear to be the bottleneck. And the contrastive objective alone is unlikely to be the cause, since Ovis trains a contrastive SigLIP backbone yet supports clean gaze heads. A properly controlled test, freezing versus fine-tuning the same encoder under a fixed language model, would be needed to settle this, and we leave that to future work. The picture above should be read as exploratory analysis pointing toward an open question rather than as a confirmed explanation.

A frozen-vs.-trained comparison on one backbone. As a partial step toward such a controlled test, we run a single same-backbone comparison. Bunny-3B bridges a frozen SigLIP-so400m encoder, the same family Ovis fine-tunes, to a Phi-2 language model through a two-layer MLP. Its discovered gaze heads do not redirect in our setup: the gaze accuracy peaks at only 8.3% at K=10 and stays below the 16.7% chance line at every K (Tab. 12), far below Ovis’s 68.7% on the same backbone, while non-gaze redirection is 0% throughout. Under the more aggressive interventions, steering the heads collapses generation into refusals rather than moving the answer to the queried panel. Model size is unlikely to be the full explanation, since a smaller trained model, InternVL3.5-2B, redirects at 64% (Tab. 11). Same backbone family, frozen versus trained, opposite outcome.

We treat this single comparison as suggestive evidence consistent with the hypothesis above, not as proof; many other factors differ between Bunny and Ovis, and a fully controlled study remains open.

Table 12. Frozen-encoder control. Top-K redirection for Bunny-3B, which bridges a frozen SigLIP-so400m encoder to a Phi-2 LM through a two-layer MLP: gaze vs. non-gaze accuracy with the gaze-condition junk%, on the 500-strip validation set under the identical intervention. Unlike every trained-encoder family, Bunny never clears the 16.7% chance line at any K.

<table><tr><td>K</td><td>Gaze</td><td>Non-gaze</td><td>junk%</td></tr><tr><td>10</td><td>8.3</td><td>0.0</td><td>25.7</td></tr><tr><td>25</td><td>7.0</td><td>0.0</td><td>19.7</td></tr><tr><td>50</td><td>1.7</td><td>0.0</td><td>4.7</td></tr><tr><td>100</td><td>1.0</td><td>0.0</td><td>5.0</td></tr><tr><td>150</td><td>1.3</td><td>0.0</td><td>21.7</td></tr><tr><td>200</td><td>5.0</td><td>0.0</td><td>39.3</td></tr><tr><td>300</td><td>2.3</td><td>0.0</td><td>82.3</td></tr></table>

## F. Qualitative Samples

## F.1. Visual Question Answering

We present qualitative examples of gaze-head steering on the VQA task in Fig. 27, Fig. 28, and Fig. 29. Under baseline conditions with no steering, the model produces a summarized answer that draws from multiple panels in the strip. When gaze heads are redirected to attend to a specific panel, the answer becomes highly specific to that panel’s content, accurately reflecting its visual details while ignoring the other panels.

## F.2. Free-Form Narration with Dynamic Gaze Switching

We present qualitative examples of gaze-head steering during free-form narration in Fig. 33, Fig. 34, and Fig. 35. Under baseline conditions, the model describes each panel in the default left-to-right order. When gaze heads are redirected through a sequence of target panels, the model produces a fluid narration that smoothly stitches together descriptions of each targeted panel. At each gaze switch, the model naturally wraps up its current panel description and transitions into describing the next target, integrating the shift into the flow of the text rather than producing abrupt breaks.

![](images/4806c0e1523b01d2a9c1a645112efc5ec8f23e2f2ef7579fd040d385445c3ba2.jpg)

![](images/8f8793a987ed41a4b2f51012f90e15d5bf78746346d02bb51f4bd8923b2ef51a.jpg)

![](images/c2e52597b377886bf291200bb91ca72badfcab335fdfcf3180231a9b2053df4f.jpg)

![](images/df6f88ee3ce7815357f8e453f19451b9fec14c76d49df24ad527f3d10bdb825b.jpg)

![](images/fb041b85cae3468b3c9cda491e0fb34bd5fc042b293eef880ce51124353aa9d9.jpg)

![](images/e005d427c85da8e0518520a1e6da5cf8a90f91701edce2df4c3160d870e39f21.jpg)

## Visual QA task with Gaze Steering

“Read each of the panels and tell me. What is the main action happening in this particular comic panel? Just output the answer in few words, do not include any other text.”

Baseline Robot plans adventure

Steered Panel 1 Robot studying a map

Steered Panel 2 Robot explores forest

Steered Panel 3 Robot explores riverbank.

Steered Panel 4 Robot explores cave.

Steered Panel 5 Robot explores forest, fights monster, and faces danger.

Steered Panel 6 Robot studies blueprint

Figure 27. Qualitative example of how the visual QA response of the model changes when steering the gaze heads’ attention to any particular target panel. The original baseline response is a summary of all 6 panels. But when we steer and lock the gaze to a fixed panel, the response is panel-specific for the same exact prompt.

![](images/0dcb4de7d7953cfa9a561ce031cb73f7e01473df96a906fa84994abd2dfb1346.jpg)

![](images/9aca6cebdd6885fb1ea5929ef6bbadda96c4f5263c08563eaf6ddaf6ef9b21e7.jpg)

![](images/487f0578e14089c76c73498a9cf8b566893fe5140dcb49c4733258777e6664b0.jpg)

![](images/3195968e732916e13478273b5b16e5134c7ba933d7c5b30f3c1472af6048695a.jpg)

![](images/2d7e0dd650ebf910b471c56345fb36400d93cab701a983edd24d1bc4b327d85e.jpg)

![](images/8ad20b08bfdcda51427d2997856efb06d1fcbf6ab3d7abae1511aa571644c342.jpg)

## Visual QA task with Gaze Steering

“Read each of the panels and tell me. What is the main action happening in this particular comic panel? Just output the answer in few words, do not include any other text.”

Baseline Elara goes through an adventure

Steered Panel 1 Elara discovers map clue

Steered Panel 2 Elise packs for trip

Steered Panel 3 Elaine climbs mountain, climbs, climbs.

Steered Panel 4 Elara explores forest.

Steered Panel 5 Elara explores magical forest.

Steered Panel 6 Elara discovers a new plant species

Figure 28. Qualitative example of how the visual QA response of the model changes when steering the gaze heads’ attention to any particular target panel. The original baseline response is a summary of all 6 panels. But when we steer and lock the gaze to a fixed panel, the response is panel-specific for the same exact prompt.

![](images/30439885159fe13483575b6352bfb7fc5151da5d059ef264954f947ff63ca821.jpg)

![](images/952e88e1c3763c45b6a87837665d1df68e74aab66eef0296715b2d80201a7d79.jpg)

![](images/b654486933e698c994d2c7a740d36fae2252c676d394129c34d048471ebabe06.jpg)

![](images/ee8ed949af9db43448c0b40631d9a46a7365e50ad38d995ca2e1938e7f836404.jpg)

![](images/194c1377b8de3223fda987b726e87a1486454c4df08d60ed124c896c1b1a506b.jpg)

![](images/82c87ad1fbced2dacd9a24ce9d96728cd5a1c3e6c04a0d633cce5ad885c2757c.jpg)

## Visual QA task with Gaze Steering

“Read each of the panels and tell me. What is the main action happening in this particular comic panel? Just output the answer in few words, do not include any other text.”

Baseline Dog befriends a cat

Steered Panel 1 Dog reads recipe book.

Steered Panel 2 Dog digs up dirt.

Steered Panel 3 Dog and cat enjoy outdoor and park activities.

Steered Panel 4Dog and fish underwater.

Steered Panel 5 Dog and cat explore space.

Steered Panel 6 Dog and cat sleep together.

Figure 29. Qualitative example of how the visual QA response of the model changes when steering the gaze heads’ attention to any particular target panel. The original baseline response is a summary of all 6 panels. But when we steer and lock the gaze to a fixed panel, the response is panel-specific for the same exact prompt.

![](images/7cb671fbd611da0ebaf1e8eb1f90afb7cae11cdb8e371a5327616ad9419b8061.jpg)

<details>
<summary>natural_image</summary>

Illustration of a girl painting a vibrant floral mural on a tiled wall, with buildings in the background (no text or symbols)
</details>

![](images/c60553caac7b1ebe70105da552b191af12c90073c05b7410984850f5697ed995.jpg)

<details>
<summary>natural_image</summary>

Black-and-white illustration of a girl playing violin outdoors among a group of people in a park setting (no text or symbols visible)
</details>

![](images/ccdf7bb40a44e20533790b9ab90896d84847ba9b2b7ea67432fc8d9e614ca7c2.jpg)

<details>
<summary>natural_image</summary>

Black-and-white illustration of a child crouching on a beach with castle-like structures and sun (no text or symbols)
</details>

![](images/0938f2a12f181ef52ba560433b17e248be6500e27ad7f3777493c176b5021a8d.jpg)

<details>
<summary>natural_image</summary>

Illustration of a girl preparing food in a kitchen, holding a bowl and tray (no text or symbols)
</details>

![](images/6b5442ffa4377237a2a9494275e81ba7e17a9fb8b0815fdfb26b7971a1556f0c.jpg)

<details>
<summary>natural_image</summary>

Illustration of a girl writing at a desk under a sunset sky (no text or symbols)
</details>

![](images/7a68e0b715a5d586610b7b16637aaeba87b71ff7f3624bbce12a3e920047daa3.jpg)

<details>
<summary>natural_image</summary>

Illustration of a girl sitting inside a library with bookshelves (no text or symbols)
</details>

## Visual QA task with Gaze Steering (6/6 hits, OpenAI 500 dataset, top-100 gaze heads)

“What is the main action or event happening in this comic strip? Answer briefly.”

Baseline (no steering): A girl is shown engaging in various activities: painting a mural, playing violin. . .

Steered Panel 1 A person is painting a mural on a wall.

Steered Panel 2 A group of people are attending a street performance.

Steered Panel 3 A child playing in the sand.

Steered Panel 4 A woman is cooking in a kitchen.

Steered Panel 5 A person is drawing or painting a sunset scene while sitting. . .

Steered Panel 6 A person is reading a book.

Figure 30. Qualitative example of gaze-head VQA steering on a strip from the OpenAI 500 dataset. The baseline answer summarizes the strip generically. Redirecting the top-100 gaze heads to each panel in turn shifts the model’s answer to that panel’s specific activity (mural, street performance, beach, kitchen, sunset painting, reading). All six steered answers match the target panel’s visual content under our forced 1-of-6 LLM judge.

![](images/0b3a46a376f05d1410800cc3e788ec6e9b0500f40a73b8f97e8fa317a493e02a.jpg)

<details>
<summary>natural_image</summary>

Illustration of a yellow dog walking through a forest (no text or symbols)
</details>

![](images/aaf9358516aca903f0009273df60f8fd880e69be32f4f137f9fe96dbf792b028.jpg)

<details>
<summary>natural_image</summary>

Painting of a dog in a red boat on a calm lake with mountains in the background (no text or symbols)
</details>

![](images/a9c38370fc4fecf82d24b429d93e290eaf83381f432fe429645129b11f2f6c9c.jpg)

<details>
<summary>natural_image</summary>

Illustration of a brown dog playing on a sandy beach with red crabs, ocean and sky in background (no text or symbols)
</details>

![](images/24add6e5e3cc1e39aa916f7e985899e1344b62e096f42bf838d6a60302c355e1.jpg)

<details>
<summary>natural_image</summary>

Illustration of a brown dog running through a field with yellow flowers and butterflies in the sky (no text or symbols)
</details>

![](images/ef80993d8af1cb9dc0aef5083983847419d07a716506999f8a7fee29a1b76941.jpg)

<details>
<summary>natural_image</summary>

Illustration of a dog walking on a mountain trail with autumn foliage and red mountains in the background (no text or symbols)
</details>

![](images/b0a723218ad0fb9e6d2f2ca5e3126b074f3afb2d8be118830aad323c71dda11b.jpg)

<details>
<summary>natural_image</summary>

Illustration of a dog resting on a campfire at night with a starry night sky and tents in the background (no text or symbols)
</details>

## Visual QA task with Gaze Steering (6/6 hits, OpenAI 500 dataset, top-100 gaze heads)

“What is the main action or event happening in this comic strip? Answer briefly.”

Baseline (no steering): A dog goes on a multi-day camping adventure.

Steered Panel 1 A dog walks through a forest, then runs through a field, and. . .

Steered Panel 2 A person is paddling a boat in a lake.

Steered Panel 3 A crab is playing with a dog.

Steered Panel 4 A dog chasing butterflies.

Steered Panel 5 A hiker hikes through a scenic mountain trail.

Steered Panel 6 A campfire.

Figure 31. Another qualitative example. Each steered answer picks up a distinct activity (forest walk, kayaking, beach with crab, butterflies, mountain hike, campfire) from the corresponding panel, while the baseline collapses to a single sentence “camping adventure”. The model’s answer follows where we point the top-100 gaze heads, even though the prompt asks about the whole strip.

![](images/07d89659bd88c41e981f53fa41b16758c9321ef6ad88583de7faa8f40b2049ce.jpg)

<details>
<summary>natural_image</summary>

Painting of a young man playing piano in a dimly lit room (no text or symbols visible)
</details>

![](images/474a6f74e8484399abce177ce5ff45860c2f22a514f0bf1faa07b46bc7829462.jpg)

<details>
<summary>natural_image</summary>

Portrait of a man in formal attire playing a violin at a desk with papers (no visible text or symbols)
</details>

![](images/be363c43b17748ae523ad90b8d67ec8d89e2820bd63ff788680dd78766fef638.jpg)

<details>
<summary>natural_image</summary>

Painting depicting a group of musicians in a room, one playing cello and another holding a tablet (no visible text or symbols)
</details>

![](images/cb484736858273b306ce93ebfb7b2a13b86100dd46556f225205715f904dd990.jpg)

<details>
<summary>text_image</summary>

BACH
</details>

![](images/9f9dfa4abf70030a85a0e7a142091a45535311a5360f927c58bcbb6346c26064.jpg)

<details>
<summary>natural_image</summary>

Illustration of a conductor holding a baton, with a man observing from behind (no text or symbols)
</details>

![](images/2000415565db474518b898e24e9d479c2febc0dffe939faa9327253c5889b4f8.jpg)

<details>
<summary>natural_image</summary>

Portrait painting of a man in formal attire playing violin (no text or symbols visible)
</details>

## Visual QA task with Gaze Steering (InternVL3.5-2B, top-50 gaze heads, 6/6 distinct)

“What is the main action or event happening in this comic strip? Answer briefly.”

Baseline (no steering): A group of musicians preparing for a. . .

Steered Panel 1 A man playing a guitar while. . .

Steered Panel 2 The musicians playing their instruments and. . .

Steered Panel 3 A group of people discussing and. . .

Steered Panel 4 A man reading a book and. . .

Steered Panel 5 The interaction between the men in. . .

Steered Panel 6 A man playing a violin. . .

Figure 32. Qualitative example of gaze-head VQA steering on InternVL3.5-2B at the model’s saturation peak (K=50; 64.0% peak accuracy, Tab. 11). The baseline answer summarizes the strip generically. Redirecting the top-50 gaze heads to each panel in turn shifts the model’s answer to that panel’s specific musical activity (guitar, ensemble, conversation, reading, interaction, violin). All six steered answers are distinct and correspond to the visual content of the targeted panel.

![](images/35e80fee37118464ecf25e5f43aa124686b29a24cfa3bd1b33b62581b9d149f3.jpg)  
Dynamic Gaze Order: P5 P4 P1 P2 P6 P3

![](images/4d785521e842f0b30ace40d07bd411ecf5665c8c0a8f227568c5f2603801db16.jpg)

## Generated narration under dynamic gaze switching

(schedule: P5 → P4 → P1 → P2 → P6 → P3, switching every 50 tokens)

Panel 1: A child sits atop a giant sunflower, smiling joyfully as a giant hand gently holds a small, smiling sunflower seedling. The sun shines brightly, and the child’s face beams with happiness. Panel 2: A child stands beside a large, blooming flower, holding a small plant. A giant hand holds a flower bud, and the child watches a small plant grow beside a tall, budding flower. Panel 3: A child stands beside a large, blo oming flower. A small child digs in the ground, and the child watches a small seed before planting it. He is wearing overalls with hat. The child waters the ground, and the child watches a small plant grow with a large, bright smile. Panel 4: A child stands beside a small, smiling flower next to a small watering can watches a small flower grow beside a large, blooming flower field. Panel 5: A child sits atop of bloomed flowers beside a large, blooming flower, holding a small flower. The child looks at sunset, and the child watches a large, blooming flower grow. Panel 6: A child kneels beside a red watering can, smiling flower, holding a small shovel. The child looks at a small plant, and the child watches a small plant grow in a awe. The child,

Figure 33. Top: The six-panel strip used for evaluation. Middle: Manually altered Gaze-head attention during generation; every 50 tokens the target switches to a new panel. Bottom: The model maintains its default numbering (“1, 2, 3. . . ”) but describes the content of whichever panel the gaze heads are steered toward. At the transition point, the model naturally ends and starts a new segment.

![](images/209878d476c8b529665fdc9922d5f612ab7d829fb18bf23134889114fa01a112.jpg)

<details>
<summary>natural_image</summary>

Six-panel collage showing a futuristic cityscape with a rocket launch, an astronaut in space, a spacecraft, a crew member, a child in a green spacesuit, and a spacecraft with a lunar rover (no text or symbols)
</details>

Dynamic Gaze Order: P4 P2 P1 P6 P5 P3

![](images/f8e1c6bb693c3362443abbce95cf86b731a5f32979132f90a981689cff7ab8e7.jpg)

<details>
<summary>heatmap</summary>

| Panel | 0    | 50   | 100  | 150  | 200  | 250  |
|-------|------|------|------|------|------|------|
| P6    | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| P5    | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| P4    | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| P3    | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| P2    | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| P1    | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
</details>

## Generated narration under dynamic gaze switching

(schedule: P4 → P2 → P1 → P6 → P5 → P3, switching every 50 tokens)

1. A futuristic explorer stands amidst alien terrain, surveying a vibrant, otherworldly landscape. A crystalline alien world awaits discovery, with a mysterious, glowing alien artifact nearby. 2. A lone astronaut explores a surreal, alien landscape, standing beside a vast, curved window. The astronaut gazes at Earth’s curved horizon, a breathtaking view of space and humanity’s cosmic journey. 3. A breathtaking view of Earth from space, with a astronaut in a spacesuit gazing out from a spacecraft. The futuristic cityscape below contrasts with the serene, futuristic skyline.4. A futuristic cityscape with towering skyscrapers and a bustling urban landscape. A massive, futuristic rocket launches from a futuristic cityscape, with a futuristic cityscape in the background. 5. A futuristic alien landscape with a mysterious, glowing alien artifact. A lone astronaut explores a mysterious, alien world, with a futuristic spaceship taking off in the background. 6. A futuristic alien landscape with a mysterious, glowing alien artifact. A large, friendly alien stands beside a human astronaut, both gazing at each other in awe . A vibrant, alien landscape stretches out with a spaceship landed surrounded by glowing alien crystals.

Figure 34. Top: The six-panel strip used for evaluation. Middle: Manually altered Gaze-head attention during generation; every 50 tokens the target switches to a new panel. Bottom: The model maintains its default numbering (“1, 2, 3. . . ”) but describes the content of whichever panel the gaze heads are steered toward. At the transition point, the model naturally ends and starts a new segment.

![](images/1fd2e3272176dd1ca09394983ad9297603a5e94716a6f751c933e0d06c8b845a.jpg)  
Dynamic Gaze Order: P2 P1 P4 P3 P5 P6

![](images/d270a2df018e48d4893daab204e9a92b56fa2722121daa5c22ff534c6eb4b01b.jpg)

<details>
<summary>heatmap</summary>

| Panel | Decode Step 0 | Decode Step 50 | Decode Step 100 | Decode Step 150 | Decode Step 200 | Decode Step 250 |
|-------|---------------|----------------|-----------------|-----------------|-----------------|-----------------|
| P1    | 0             | 50             | 100             | 150             | 200             | 250             |
| P2    | 0             | 50             | 100             | 150             | 200             | 250             |
| P3    | 0             | 50             | 100             | 150             | 200             | 250             |
| P4    | 0             | 50             | 100             | 150             | 200             | 250             |
| P5    | 0             | 50             | 100             | 150             | 200             | 250             |
| P6    | 0             | 50             | 100             | 150             | 200             | 250             |
</details>

## Generated narration under dynamic gaze switching

(schedule: P2 → P1 → P4 → P3 → P5 → P6, switching every 50 tokens)

1. Children discover a magical forest through a portal, surrounded by vibrant flora and a glowing jungle, as they journey into a fantastical, enchanted world. 2. A vibrant jungle scene with a glowing portal, where a child and a dinosaur coexist in a cozy, rustic setting, hinting at a hidden world of adventure and discovery. 3. A cozy attic with a child and a dog, surrounded by books and a mysterious attic, hinting at a hidden world of adventure and discovery. 4 . A vibrant underwater scene with colorful coral, a school of fish, and a coral reef, where a child and a coral reef, with a school of fish, explore a colorful underwater world. 5. A vibrant underwater scene with colorful coral, a school of icebergs, and a polar bear, where a child and a polar bear, with a snowy landscape, explore a frozen wonderland. 6. A serene Arctic landscape with a polar bear, where a child and a polar bear, with a starry sky, explore a cosmic adventure through space. 7. A cosmic scene with a child and a space explorer, surrounded by stars and planets, as they journey through space, exploring the universe with wonder and excitement. 8. A cosmic scene with a child and a cityscape, where a child and a cityscape, with a futuristic city, explore a vibrant, imaginative world with a playful, adventurous spirit.

Figure 35. Top: The six-panel strip used for evaluation. Middle: Manually altered Gaze-head attention during generation; every 50 tokens the target switches to a new panel. Bottom: The model maintains its default numbering (“1, 2, 3. . . ”) but describes the content of whichever panel the gaze heads are steered toward. At the transition point, the model naturally ends and starts a new segment.