# What VGGT Knows About Overlap: Probing Geometric Foundation Models for Co-Visibility

Filippo Ziliotto<sup>1,2</sup> , Luciano Serafini<sup>2</sup> , Lamberto Ballan<sup>1</sup> , and Tommaso Campari<sup>2</sup>

<sup>1</sup> University of Padova <sup>2</sup> Fondazione Bruno Kessler (FBK)

Abstract. A fundamental challenge in 3D reconstruction and robotic localization is co-visibility: determining which image pairs share overlapping visible surfaces, particularly in scenarios with minimal overlap. We demonstrate that VGGT implicitly encodes co-visibility as an emergent behavior: without any supervision for this task, its internal representations exhibit a clear hierarchical structure mirroring that of large language models, i.e. early layers build a 3D-aware scene representation, while late layers act as dedicated co-visibility reasoners. In particular, we identify layer L17 as a negative anchor that consistently routes nonco-visible pairs for this backbone, regardless of the evaluation setting, providing task-grounded evidence of layer specialization in a geometrygrounded foundation model. Building on this, we introduce Co-VGGT, which freezes VGGT and trains only a lightweight layer-wise mixtureof-experts head (∼7.5M parameters) to classify co-visibility from RGB alone, treating each layer as a specialized expert whose geometric abstraction is adaptively weighted per input pair. On the Co-VisiON benchmark, Co-VGGT surpasses the human annotation baseline and improves over prior work by more than 25% pairwise and 10% multiview. Pairwise predictions are well-calibrated (ECE = 0.030), enabling direct use as edge weights in visibility graphs for downstream SfM and SLAM pipelines without post-hoc correction. Code and data are available .

Keywords: Co-visibility · Multiview Geometry · Embodied perception

## 1 Introduction

Robotic perception and 3D reconstruction systems typically operate on sparse, imperfect image sets rather than isolated, perfect views. A fundamental challenge in these pipelines—whether for mapping, localization, or scene reconstruction—is determining co-visibility, the subset of surfaces jointly observed by multiple cameras. High co-visibility yields abundant geometric constraints and stable optimization. Conversely, when spatial overlap is limited or absent, matching becomes ambiguous and pose estimation drifts. In these scenarios, reconstruction pipelines often fail silently, generating plausible but geometrically inconsistent structures. Operating in this sparse-view regime is not an edge case but rather a standard condition for embodied agents navigating complex, occluded environments.

![](images/514b315149ac3b472531d71b8423251fccc2e967537f488f9ad737d1f0fc78fa.jpg)  
Fig. 1: Co-visibility Task. We study co-visibility prediction: given multiple RGB views of a scene, the goal is to determine which image pairs share overlapping visible regions. Our approach probes geometric consistency in a frozen VGGT foundation model, extracting layer-wise view embeddings and combining them through a lightweight mixture-of-experts head. Without modifying the backbone, the model aggregates information across transformer layers to produce a co-visibility probability for each pair, enabling eficient construction of scene-level visibility graphs.

Despite advances in multiview transformers and learned reconstruction models, performance drops significantly as spatial overlap decreases. Under these conditions, fusion modules misalign features, learned descriptors drift, and global reasoning degrades into spurious correlations. Recent benchmarks, such as Co-VisiON [8], explicitly isolate this co-visibility reasoning, exposing a critical performance gap between current models and human baselines in sparse, highly imbalanced scenarios. Bridging this gap is essential for robotics: robust co-visibility estimation dictates which view pairs to match, which constraints to trust, and when to flag reconstruction failures.

Concurrently, geometry-grounded foundation models present a promising alternative. The Visual Geometry Grounded Transformer (VGGT) [34] demonstrates strong emergent geometric reasoning; without explicit 3D supervision, it infers scene structure and reconstructs geometry even across widely separated viewpoints. Similar to emergent phenomena in large language models, VGGT’s internal representations encode latent structures that transcend its immediate training objective. However, the exact nature of these spatial reasoning signals and the methodology to extract them for embodied decision-making remain poorly understood.

In this work, we show that frozen VGGT features contain a strong, directly usable signal for co-visibility estimation (see Fig. 1). Furthermore, we find that specific late VGGT layers act as consistent anchors for co-visibility decisions, while earlier layers encode broader scene-aware geometric representations. Leveraging this property, we propose Co-VGGT, an MoE head in which each layer acts as an “expert” whose geometric abstraction is adaptively weighted per image pair. Compared to single-layer probing, this routing improves accuracy and calibration while exposing which layer cues drive each decision. Relying solely on RGB inputs, Co-VGGT achieves near human-level accuracy on the Co-VisiON benchmark. It surpasses the current state-of-the-art by over 25% in pairwise covisibility prediction and nearly 10% in multiview inference. We further observe that co-visibility reasoning is dominated by late layers, whereas earlier layers contribute more general geometric context.

Our findings indicate that geometry-grounded foundation models encode spatial priors that are far richer than those obtained through conventional contrastive or matching-based pretraining. Furthermore, these signals can be eficiently distilled into modular predictors. Such predictors could be integrated into reconstruction pipelines: a dedicated co-visibility head can refine overlap prediction and serve as an auditing mechanism, identifying inconsistent constraints and detecting early-stage failures to support reliable embodied autonomy.

In summary, our contributions are as follows: (i) we introduce Co-VGGT, achieving near human-level co-visibility from RGB and improving the Co-VisiON SOTA by >25% (pairwise) and ∼10% (multiview); (ii) we provide evidence that VGGT implicitly encodes co-visibility structure across its layers, exposing emergent spatial priors that are absent in standard supervised baselines; and (iii) we show that Co-VGGT is robust to minimal overlap scenarios and its well-calibrated outputs (ECE = 0.030 on Gibson val) enable direct use as edge weights in visibility graphs, providing a practical auditing signal for SfM and SLAM pipelines without post-hoc correction.

## 2 Related Works

Co-visibility prediction sits at the intersection of 3D reconstruction, feature matching, and geometric reasoning. We review the most relevant lines of work, highlighting how each studies co-visibility as an explicit, predictable signal. In doing so, we ground our approach within the broader landscape and clarify what makes it fundamentally distinct from all prior work.

Co-visibility in Reconstruction and Matching Pipelines. Co-visibility is handled implicitly in classical SfM and SLAM through feature matching and geometric verification [6, 26], degrading under weak texture or large viewpoint changes. Learned SLAM methods [32, 41] and implicit neural mapping [29, 42] improve robustness but still require suficient overlap, while systems with explicit graph structures — Kimera [24], Hydra [15], and situational graphs [4] — treat co-visibility as a derived post-hoc quantity. Learned matchers — SuperGlue [25], LightGlue [19], Eficient LoFTR [36], and OmniGlue [16] — estimate matchability conditioned on the existence of overlap and cannot flag non-overlapping pairs, while retrieval methods like NetVLAD [2] conflate semantic similarity with geometric overlap. We instead predict co-visibility directly from RGB as a dedicated, calibrated first-class signal before any reconstruction is attempted.

Geometric Foundation Models and Sparse-View Reasoning. DUSt3R [35], MUSt3R [5], MVSNeRF [7], and MV-DUSt3R+ [39] have advanced multiview geometry estimation; yet, all degrade in sparse-view regimes where co-visible surface support is limited. VGGT [34] achieves emergent 3D reasoning without explicit geometric supervision; we show its internal representations encode a hierarchically-organized co-visibility signal that can be distilled without modifying the backbone. The Co-VisiON benchmark [8] formalizes this as graph inference over sparse indoor views, exposing a critical performance gap that our approach substantially closes.

![](images/87ad2e4d69e1d41591199f06612b5d0ea7932304da6a4ab24679446b74a0aefe.jpg)  
Fig. 2: Overview of the Co-VGGT method. Input RGB views are processed by the frozen VGGT backbone to extract layer-wise features. These features are projected, summarized, and formed into per-view embeddings. In both pairwise and multiview modes, these embeddings are used to construct pair features, which are then fed into a trainable Mixture-of-Experts (MoE) head to predict co-visibility probabilities, enabling the construction of scene-level visibility graphs. Tensor shapes are annotated for key intermediate representations.

VLMs, Geometric Distillation, and Transformer Interpretability. VLMs such as GPT-4o [22], Gemini [31], and SpatialRGPT [9] show emergent spatial reasoning but remain unreliable for precise 3D consistency under significant viewpoint changes, as confirmed by our experiments. Recent distillation approaches inject geometric priors into frozen language backbones [1,17], but target general scene understanding rather than co-visibility. Meanwhile, probing work localizes capabilities to specific transformer layers [12,28,33], with reasoning concentrated in the late layers of LLMs [20] and global semantics in the late layers of vision transformers [40]. We provide the first task-grounded evidence of an analogous hierarchical structure in a geometry-grounded model, identifying the late VGGT layers (particularly L17) as decisive co-visibility reasoners.

## 3 Method

We address co-visibility estimation simply using RGB inputs by training a lightweight binary classifier on top of a frozen geometric foundation model. Given a set of views from the same scene, the goal is to predict whether each view pair shares any jointly visible surface region to reconstruct the scene-graph. Our model operates in two regimes: (i) pairwise, where each training example is a single pair of images; and (ii) multiview, where each example is an entire scene containing many views and labeled pairs. An overview of the method is shown in Fig. 2.

Importantly, pairwise inference is a special case of multiview with $N { = } 2$ and a single pair. This unified formulation allows training in multiview mode and evaluation in both pairwise and multiview settings without architectural changes. Problem Setup. Let $\mathcal { T } = \{ I _ { s } \} _ { s = 1 } ^ { S }$ be a set of RGB views of the same scene, with $I _ { s } \in \mathbb { R } ^ { 3 \times H \times W }$ . For any pair $( i , j )$ , the target is a binary co-visibility label $y _ { i j } \in \{ 0 , 1 \}$ . In the pairwise setting we predict a single labeled pair per sample $\left( S { = } 2 \right)$ . In the multiview setting we process $S { > } 2$ views jointly and predict a set of labeled pairs $\mathcal { P } = \{ ( i _ { p } , j _ { p } , y _ { p } ) \} _ { p = 1 } ^ { P }$ , which defines a co-visibility graph over the views.

Backbone and Summarization. We use a pretrained VGGT backbone $\varPhi$ as a frozen feature extractor. Given a batch X ∈ <sup>RB×S×3×H×W</sup> , we extract patch-token features from selected layers $ \ell \in \{ 1 , \ldots , L \} \colon \mathbf { T } ^ { ( \ell ) } = \phi ^ { ( \ell ) } ( \mathbf { X } ) \in$ $\mathbf { \widehat { \mathbb { R } } } ^ { B \times S \times P _ { \mathrm { t o k } } \times C _ { \mathrm { r a w } } }$ . All parameters of $\varPhi$ remain frozen. To obtain compact perview embeddings, we (i) project token channels and (ii) summarize tokens with learned queries. We first apply a shared linear projection with LayerNorm: $\widehat { \mathbf { T } } ^ { \left( \ell \right) } = \bar { \mathrm { L N } } \big ( \mathbf { T } ^ { \left( \ell \right) } W \big )$ , where $W \in \mathbb { R } ^ { C _ { \mathrm { r a w } } \times C _ { \mathrm { p r o j } } }$ . Then, for each view we summarize $P _ { \mathrm { t o k } }$ tokens into $T$ summary tokens via cross-attention with learned queries $\mathbf { Q } \in \mathbb { R } ^ { T \times C _ { \mathrm { p r o j } } }$

$$
\mathbf {S} _ {s} ^ {(\ell)} = \operatorname{Attn} \left(\mathbf {Q}, \widehat {\mathbf {T}} _ {s} ^ {(\ell)}, \widehat {\mathbf {T}} _ {s} ^ {(\ell)}\right) \in \mathbb {R} ^ {T \times C _ {\text { proj}}}, \quad \mathbf {e} _ {s} ^ {(\ell)} = \operatorname{vec} \left(\mathbf {S} _ {s} ^ {(\ell)}\right) \in \mathbb {R} ^ {D},\tag{1}
$$

with $D = T C _ { \mathrm { p r o j } }$ . This yields per-view, per-layer embeddings $\{ { \bf e } _ { s } ^ { ( \ell ) } \} _ { \ell = 1 } ^ { L }$

Pair Representation $\&$ MoE Head. Following [10, 13, 21], to enhance the embedding representation for each labeled pair $( i , j ) \in \mathcal { P }$ and layer $\ell ,$ we build a symmetric pair feature:

$$
\mathbf {f} _ {i j} ^ {(\ell)} = \left[ \mathbf {e} _ {i} ^ {(\ell)}, \mathbf {e} _ {j} ^ {(\ell)}, \left| \mathbf {e} _ {i} ^ {(\ell)} - \mathbf {e} _ {j} ^ {(\ell)} \right|, \mathbf {e} _ {i} ^ {(\ell)} \odot \mathbf {e} _ {j} ^ {(\ell)} \right] \in \mathbb {R} ^ {4 D}.\tag{2}
$$

Based on the assumption that VGGT mirrors the hierarchical reasoning structure of LLMs, in our method, each layer acts as an “expert” that predicts a logit $z _ { i j } ^ { ( \ell ) }$ from $\mathbf { f } _ { i j } ^ { ( \ell ) }$ , while a gating network assigns mixture weights across layers:

$$
z _ {i j} ^ {(\ell)} = \mathrm{MLP} _ {\exp} \left(\mathrm{LN} \left(\mathbf {f} _ {i j} ^ {(\ell)}\right)\right), \quad \alpha_ {i j} ^ {(\ell)} = \operatorname{softmax} _ {\ell} \left(\mathrm{MLP} _ {\text { gate }} \left(\mathrm{LN} \left(\mathbf {f} _ {i j} ^ {(\ell)}\right)\right)\right).\tag{3}
$$

The final logit and probability are

$$
z _ {i j} = \sum_ {\ell = 1} ^ {L} \alpha_ {i j} ^ {(\ell)} z _ {i j} ^ {(\ell)}, \qquad p _ {i j} = \sigma (z _ {i j}),\tag{4}
$$

as illustrated in Fig. 3.

![](images/90d4096a59e3ee68525bc8bd92679cfb25ba993086f3b6ddc227df7950338531.jpg)  
Fig. 3: Co-visibility Estimation Head. Given a set of sparse RGB views (left), our method leverages the frozen Visual Geometry Grounded Transformer (VGGT) to extract layer-wise view embeddings. A lightweight, trainable Mixture-of-Experts (MoE) head (center) adaptively aggregates these multi-scale features to predict pairwise co-visibility probabilities. The resulting predictions form a dense scene-level visibility graph (right), identifying overlapping regions between disjoint viewpoints to guide robust 3D reconstruction and robotic perception.

Training Objective. We optimize the projector, summarizer, expert, and gate parameters with binary cross-entropy on logits:

$$
\mathcal {L} = \frac {1}{| \mathcal {P} |} \sum_ {(i, j, y _ {i j}) \in \mathcal {P}} \ell_ {\mathrm{BCELogits}} (z _ {i j}, y _ {i j}).\tag{5}
$$

In pairwise mode $| \mathcal { P } | { = } 1$ , while in multiview mode, we average over all labeled pairs in the scene. Although co-visibility resembles image similarity, naive embedding matching is brittle under viewpoint changes, occlusion, and texture ambiguity. This structure empirically reduces false positives in low-overlap regimes while improving calibration.

## 3.1 Zero-Shot Evaluation

We introduce a zero-shot baseline that predicts co-visibility without any trainable parameters. The classification head and mixture-of-experts are removed, and scores are computed directly from frozen VGGT features using cosine similarity. Given a pair of RGB images, we extract layer activations $\{ \bar { \mathbf { T } } ^ { ( \ell ) } \} _ { \ell = 1 } ^ { L }$ from the frozen backbone and apply a pooling operator $\mathcal { P } ( \cdot )$ to obtain per-view embeddings $\mathbf { e } ^ { ( \ell ) } \in \mathbb { R } ^ { D }$ . Embeddings are then computed for each of the considered layers (see Tab. 6 for details).

For a pair $( i , j )$ , we compute the cosine similarity per layer and average across layers:

$$
c _ {i j} = \frac {1}{L} \sum_ {\ell = 1} ^ {L} \frac {\langle \mathbf {e} _ {i} ^ {(\ell)} , \mathbf {e} _ {j} ^ {(\ell)} \rangle}{\| \mathbf {e} _ {i} ^ {(\ell)} \| _ {2} \| \mathbf {e} _ {j} ^ {(\ell)} \| _ {2}} \in [ - 1, 1 ].\tag{6}
$$

Finally, we rescale to [0, 1] to obtain a pseudo-probability $p _ { i j } ^ { \mathrm { Z S } } = ( c _ { i j } + 1 ) / 2 .$ which is evaluated using the same metrics as the trained models.

## 4 Experiments

We evaluate our method on the Co-VisiON benchmark across both HM3D and Gibson environments. We report results for the pairwise and multiview settings and include the zero-shot method described in Sec. 3. All experiments use AdamW with a learning rate $1 0 ^ { - 4 }$ , a batch size of 32, and a weight decay $1 0 ^ { - 4 }$ , trained for 50 epochs using the results corresponding to the best AUC value. In the multiview setting, peak performance is reached around 30 epochs, whereas in the pairwise setting, it converges within 10 epochs.

## 4.1 Dataset & Metrics

Co-VisiON [8] evaluates co-visibility reasoning in sparse indoor view sets. Each sample consists of a small collection of RGB views from the same scene, and the task is to predict a binary co-visibility graph, where an edge $( i , j )$ is positive if the two views share non-zero co-visible surface area. The benchmark spans Gibson [38] (80/20 split) and HM3D [23] (90/10 split), comprising 85/755 scenes and 33,849/210,008 labeled pairs, respectively. Moreover, all validation scenes are disjoint from training scenes, so performance reflects generalization to unseen environments, akin to a zero-shot setting. For detailed specifics, refer to [8].

The dataset also reports a human baseline for the Gibson multiview setting. However, this should not be interpreted as a definitive measure of human-level performance, but rather as an indicative reference point. Given the presence of artifacts in Gibson, the true performance achievable by humans under cleaner conditions is likely higher.

We use the same protocol for pairwise and multiview settings since both produce pairwise co-visibility scores that define an adjacency matrix. For a scenario with N views, the model outputs scores $d _ { i j } \in [ 0 , 1 ]$ , forming $\mathbf { D } \in [ 0 , 1 ] ^ { N \times N }$ which are compared to the ground-truth adjacency $\mathbf { \bar { A } } \in \left\{ 0 , 1 \right\} ^ { N \times N }$ . The only diference is how scores are computed: independently per pair in the pairwise setting, or jointly from multiple views in the multiview setting.

Given a threshold $\tau ,$ we binarize D as $\hat { \mathbf { A } } ( \tau ) = \mathbb { I } [ \mathbf { D } \geq \tau ]$ (symmetric, zero diagonal) and compute the Graph IoU : $\begin{array} { r } { \mathrm { I o U } ( \tau ) = \frac { | \mathcal E \cap \hat { \mathcal E } ( \tau ) | } { | \mathcal E \cup \hat { \mathcal E } ( \tau ) | } } \end{array}$ . We report the bestthreshold IoU (denoted as IoU<sup>∗</sup> in the tables), $\begin{array} { r } { \mathrm { I o U ^ { * } } = \operatorname* { m a x } _ { \tau \in [ 0 , 1 ] } \mathrm { I o U } ( \tau ) } \end{array}$ , and the area under the IoU-threshold curve over $\tau \in \ [ 0 , 1 ]$ , defined as $\mathrm { A U C } =$ $\textstyle \int _ { 0 } ^ { 1 } \operatorname { I o U } ( \tau ) d \tau$ . Metrics are computed per scenario and averaged over the split. Additional metrics results (e.g., F1 score, Accuracy) are provided in Sec. A.6 of the Supplementary material.

## 4.2 Results

Tabs. 1–2 report co-visibility prediction performance on Co-VisiON in the pairwise and multiview settings, measured with Graph $\operatorname { I o U } ^ { * }$ and AUC. Our Co-

VGGT consistently achieves the best results across datasets and evaluation protocols. In the pairwise setting (Tab. 1), Co-VGGT attains 0.85/0.78 IoU\*/AUC on Gibson and 0.84/0.78 on HM3D, substantially outperforming strong learned baselines such as Covis [8] (0.56/0.54 on Gibson; 0.53/0.51 on HM3D) and reconstruction based DUSt3R [35] (0.54/0.54 on Gibson; 0.40/0.40 on HM3D). Prompting large VLMs yields competitive results relative to earlier learned models (e.g., GPT-4o [22] reaches 0.58/0.58 on Gibson and 0.54/0.54 on HM3D), but remains far behind Co-VGGT, indicating that explicit geometric specialization is critical for reliable overlap reasoning under sparse viewpoints.

In the multiview setting (Tab. 2), Co-VGGT again leads with 0.74/0.72 on Gibson and 0.76/0.74 on HM3D, surpassing Covis/Covis-freeze and multiview reconstruction (MV-DUSt3R+ [30]). Interestingly, performance in multiview is lower than in pairwise, which is counterintuitive for reconstruction-style models but can be explained by our current multiview embedding extraction, which aggregates features across many views and yields noisier per-view vectors for the downstream pair classifier.

On Gibson, Co-VGGT also exceeds the reported Human Annotation baseline (0.72/0.72) [8], suggesting that the learned head on top of frozen VGGT features captures geometric cues that are dificult to assess reliably from sparse RGB images presented to humans.

We note that for sparse view sets (small N ), running the pairwise model exhaustively over all $\binom { N } { 2 }$ pairs and assembling the graph post-hoc yields superior calibration and accuracy (0.85 vs. 0.74 IoU\*) at negligible additional cost, and is therefore the recommended inference strategy in practice. The multiview mode ofers a scalable alternative for larger view sets where the quadratic cost becomes prohibitive.

Tab. 3 reports validation performance under two complementary dificulty protocols that stress diferent sparsity regimes of the co-visibility graph. Edgelevel dificulty bins individual pairs by their overlap ratio: easy if overlap $\geq 5 0 \%$ medium if $1 0 \% \leq \mathrm { o v e r l a p } < 5 0 \%$ , and hard if overlap < 10%. While VLMbased baselines are near-saturated in the easy/medium regimes, they degrade sharply when overlap becomes minimal; in particular, on the hard split Co-VGGT achieves substantially higher Graph-IoU than the strongest VLM baseline (e.g., 0.84 vs. 0.34 for GPT-4o [22]), indicating markedly better robustness to low-overlap pairs. We further report graph-level dificulty by binning entire scenes according to scene sparsity, defined as the average pairwise overlap within the scenario: easy $\mathrm { i f } \geq 1 0 \%$ , medium if $4 \% \leq \cdot < 1 0 \%$ , and hard $\mathrm { i f } < 4 \%$ . Even in globally sparse scenes, Co-VGGT maintains strong performance (hard: 0.84 Graph-IoU), outperforming both Covis-based baselines and GPT-4o (hard: 0.57), showing that the method remains reliable when the overall graph connectivity is low, rather than just a few pairs being dificult. Overall, these results suggest that Co-VGGT encodes more stable geometric cues than VLM prompting or learned baselines, and that its advantage concentrates exactly where co-visibility is most failure-prone: extremely low pairwise overlap and scene-wide sparse connectivity. AUC result tables are reported in the Supplementary material.

Table 1: Pairwise co-visibility prediction. Results on Gibson and HM3D datasets. Baselines results are reported from [8], no human baseline available in this case.

<table><tr><td rowspan="2">Method</td><td colspan="2">Gibson</td><td colspan="2">HM3D</td></tr><tr><td>IoU* (↑)</td><td>AUC (↑)</td><td>IoU* (↑)</td><td>AUC (↑)</td></tr><tr><td>SuperGlue [25]</td><td>0.47</td><td>0.11</td><td>0.38</td><td>0.10</td></tr><tr><td>SIFT + RANSAC [18]</td><td>0.35</td><td>0.05</td><td>0.34</td><td>0.05</td></tr><tr><td>NetVLAD [2]</td><td>0.35</td><td>0.33</td><td>0.35</td><td>0.29</td></tr><tr><td>DUSt3R [35]</td><td>0.54</td><td>0.54</td><td>0.40</td><td>0.40</td></tr><tr><td>ViT [11]</td><td>0.47</td><td>0.43</td><td>0.48</td><td>0.46</td></tr><tr><td>VGG [27]</td><td>0.35</td><td>0.30</td><td>0.34</td><td>0.21</td></tr><tr><td>ResNet18 [14]</td><td>0.38</td><td>0.36</td><td>0.48</td><td>0.46</td></tr><tr><td>CroCo v2 [37]</td><td>0.50</td><td>0.45</td><td>0.43</td><td>0.38</td></tr><tr><td>Covis [8]</td><td>0.56</td><td>0.54</td><td>0.53</td><td>0.51</td></tr><tr><td>Qwen2.5-VL 72B [3]</td><td>0.41</td><td>0.41</td><td>0.39</td><td>0.39</td></tr><tr><td>Gemini-2.0-Flash [31]</td><td>0.42</td><td>0.42</td><td>0.39</td><td>0.39</td></tr><tr><td>SpatialRGPT [9]</td><td>0.49</td><td>0.49</td><td>0.37</td><td>0.37</td></tr><tr><td>GPT-4o [22]</td><td>0.58</td><td>0.58</td><td>0.54</td><td>0.54</td></tr><tr><td>Co-VGGT (Zero-shot)</td><td>0.50</td><td>0.31</td><td>0.46</td><td>0.31</td></tr><tr><td>Co-VGGT (Ours)</td><td>0.85</td><td>0.78</td><td>0.84</td><td>0.78</td></tr></table>

Even without any training, the zero-shot variant achieves 0.50 IoU\* on the Gibson pairwise setting—already competitive with several supervised baselines (Tab. 6)—confirming that VGGT’s frozen representations carry a meaningful co-visibility signal that supervised fine-tuning fully unlocks.

## 4.3 Pair Aggregation

Tab. 4 studies how the choice of symmetric pair features impacts performance. Using only raw embeddings (simple) is weakest, while adding multiplicative interactions $( e _ { i } \odot e _ { j } )$ improves results (product). Including absolute diferences $( | e _ { i } - e _ { j } | )$ provides the largest single gain, suggesting that relative feature ofsets capture key overlap cues. Our full aggregation $( [ e _ { i } , e _ { j } ] , | e _ { i } - e _ { j } | , e _ { i } \odot e _ { j } )$ combined with the layer-wise MoE yields the best IoU\*/AUC, outperforming the same feature set under the standard (non-MoE) aggregation baseline.

Notably, the absolute variant matches Co-VGGT in IoU\* (0.74) but lags in AUC (0.71 vs. 0.73), suggesting that the element-wise product $\mathbf { e } _ { i } \odot \mathbf { e } _ { j }$ contributes primarily to probability calibration rather than threshold-optimal classification: a meaningful distinction when predicted scores are used as continuous edge weights in downstream visibility graphs.

![](images/7726d044b86116b7ecc15312f9485c46972fb17d1e2249cd43b8f3e2c52967a4.jpg)  
Fig. 4: Co-visibility Scene-Graph Examples (Multiview). Given N input views, we visualize the ground-truth adjacency matrix (left), the predicted binary graph obtained by thresholding scores at the IoU-optimal τ <sup>∗</sup>. Each matrix is symmetric and entry (i, j) denotes the relation between image i and image j.

Table 2: Multiview co-visibility prediction. Results on Gibson and HM3D datasets. Baselines and Human results are reported from [8].

<table><tr><td rowspan="2">Method</td><td colspan="2">Gibson</td><td colspan="2">HM3D</td></tr><tr><td>IoU* (↑)</td><td>AUC (↑)</td><td>IoU* (↑)</td><td>AUC (↑)</td></tr><tr><td>Human Annotation [8]</td><td>0.72</td><td>0.72</td><td>-</td><td>-</td></tr><tr><td>NetVLAD [2]</td><td>0.38</td><td>0.32</td><td>0.42</td><td>0.39</td></tr><tr><td>MV-DUSt3R+ [30]</td><td>0.56</td><td>0.56</td><td>0.45</td><td>0.45</td></tr><tr><td>CroCo v2 [37]</td><td>0.48</td><td>0.41</td><td>0.41</td><td>0.37</td></tr><tr><td>Covis [8]</td><td>0.59</td><td>0.57</td><td>0.57</td><td>0.56</td></tr><tr><td>Covis-freeze [8]</td><td>0.61</td><td>0.57</td><td>0.58</td><td>0.56</td></tr><tr><td>GPT-4o [22]</td><td>0.63</td><td>0.63</td><td>0.59</td><td>0.59</td></tr><tr><td>Co-VGGT (Zero-shot)</td><td>0.37</td><td>0.30</td><td>0.36</td><td>0.30</td></tr><tr><td>Co-VGGT (Ours)</td><td>0.74</td><td>0.73</td><td>0.76</td><td>0.74</td></tr></table>

## 4.4 Hierarchical Information

As shown in Fig. 6, co-visibility reasoning is concentrated in VGGT’s late layers. Gate mass is essentially zero for layers 0–12, confirming that early representations encode geometric primitives rather than overlap cues. Among late layers, we observe a strong class asymmetry. Negative pairs consistently route to L17 (peak ≈0.37–0.41) with low gating entropy across both pairwise and multiview settings, making L17 a negative anchor that confidently rejects geometrically disjoint pairs.

This is supported by attention maps (see supplementary material): for nonco-visible pairs, the first view yields focused, structured attention, while the second produces difuse, unanchored responses — a signature of failed crossimage correspondence at a geometrically mature layer. Positive pairs, by contrast, show higher gating entropy and a setting-dependent preference: multiview routing sharpens to L15 (peak ≈0.41), while pairwise spreads across L18–L23 (peak ≈0.19–0.23), reflecting greater uncertainty about which layer carries the overlap signal when no auxiliary views are available.

![](images/cba1ae11ec731e7fdbadf79602267805fdf529241aa33cdad0cd77bce230daba.jpg)  
Fig. 5: Cross-view Similarity Matrix Examples. Examples of Co-VGGT attention token similarity over pair of co-visibile images. (left) input RGB image, (right) the attention output mask. Features are extracted from signal belonging to layer 17 of the Pairwise evaluation.

Table 3: Performance across dificulty levels. Left: edge-level dificulty defined by pairwise image overlap $( \mathrm { E a s y } \geq 5 0 \%$ , Med. 10–50%, Hard $< 1 0 \% )$ . Right: graphlevel dificulty defined by scene sparsity via average overlap $( \mathrm { E a s y } \geq 1 0 \%$ , Med. 4–10%, Hard < 4%). Metrics report Graph IoU at the best threshold.  
(a) Image overlap (edge-level).

<table><tr><td>Method</td><td>Easy</td><td>Med.</td><td>Hard</td><td>Avg.</td></tr><tr><td>GPT-4o [22]</td><td>0.97</td><td>0.92</td><td>0.34</td><td>0.63</td></tr><tr><td>Gemini-2.0-Flash [31]</td><td>1.00</td><td>0.99</td><td>0.14</td><td>0.42</td></tr><tr><td>Qwen2.5-VL 72B [3]</td><td>1.00</td><td>0.99</td><td>0.14</td><td>0.41</td></tr><tr><td>SpatialRGPT [9]</td><td>1.00</td><td>0.99</td><td>0.13</td><td>0.35</td></tr><tr><td>Covis [8]</td><td>0.99</td><td>0.88</td><td>0.24</td><td>0.59</td></tr><tr><td>Covis (freeze) [8]</td><td>1.00</td><td>0.89</td><td>0.30</td><td>0.61</td></tr><tr><td>Co-VGGT (Multi)</td><td>0.80</td><td>0.88</td><td>0.70</td><td>0.74</td></tr><tr><td>Co-VGGT (Pair)</td><td>1.00</td><td>0.93</td><td>0.84</td><td>0.85</td></tr></table>

(b) Scene sparsity (graph-level).

<table><tr><td>Method</td><td>Easy</td><td>Med.</td><td>Hard</td><td>Avg.</td></tr><tr><td>GPT-4o [22]</td><td>0.83</td><td>0.64</td><td>0.57</td><td>0.63</td></tr><tr><td>Gemini-2.0-Flash [31]</td><td>0.75</td><td>0.39</td><td>0.28</td><td>0.42</td></tr><tr><td>Qwen2.5-VL 72B [3]</td><td>0.77</td><td>0.40</td><td>0.28</td><td>0.41</td></tr><tr><td>SpatialRGPT [9]</td><td>0.72</td><td>0.38</td><td>0.26</td><td>0.35</td></tr><tr><td>Covis [8]</td><td>0.80</td><td>0.63</td><td>0.52</td><td>0.59</td></tr><tr><td>Covis (freeze) [8]</td><td>0.81</td><td>0.65</td><td>0.54</td><td>0.61</td></tr><tr><td>Co-VGGT (Multi)</td><td>0.99</td><td>0.92</td><td>0.73</td><td>0.74</td></tr><tr><td>Co-VGGT (Pair)</td><td>1.00</td><td>0.97</td><td>0.84</td><td>0.85</td></tr></table>

This structure directly explains the calibration gap reported below: multiview exhibits sharp, consistent routing but noisier embeddings from compressing multiview context into fixed-size vectors; pairwise produces cleaner embeddings and better calibration but has flatter routing due to the absence of additional scene context.

## 4.5 Calibration Measures

In Fig. 7, we assess probability calibration on Gibson val using Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier score. In the pairwise setting, Co-VGGT is well calibrated (ECE = 0.030, Brier = 0.043): predicted scores behave as empirical frequencies (e.g., p≈0.7 implies ∼70% true positives), making them directly usable as edge weights for downstream filtering keeping p>0.7 for matching and discarding p<0.2 (to suppress spurious constraints) without any post-hoc correction. In multiview, calibration degrades (ECE = 0.074, Brier = 0.085), consistent with the noisier embeddings produced by compressing multiview context into fixed-size vectors, as discussed in Sec. 4.6. Both settings exhibit high MCE (0.223 pairwise, 0.347 multiview); however, this is driven by sparsely populated mid-confidence bins, since over 90% of predictions fall in [0, 0.1) or [0.9, 1]. Furthermore, we added a proof-of-concept COLMAP experiment against diferent methods to show this downstream capability (see Supplementary material).

Table 4: Aggregator ablation. All aggregators use MoE except the standard aggregation baseline. “Standard” denotes concatenation of $( [ e _ { i } , e _ { j } ] , | e _ { i } - e _ { j } | , e _ { i } \odot e _ { j } )$

<table><tr><td rowspan="2">Pair Aggregator</td><td colspan="3">Pair feature set</td><td>MoE</td><td colspan="2">Gibson</td><td colspan="2">HM3D</td></tr><tr><td> $|e_i - e_j|$ </td><td> $e_i \odot e_j$ </td><td> $[e_i, e_j]$ </td><td>w/ or w/o</td><td>IoU*</td><td>AUC</td><td>IoU*</td><td>AUC</td></tr><tr><td>simple</td><td>✘</td><td>✘</td><td>✓</td><td>✓</td><td>0.67</td><td>0.70</td><td>0.69</td><td>0.70</td></tr><tr><td>product</td><td>✘</td><td>✓</td><td>✓</td><td>✓</td><td>0.71</td><td>0.70</td><td>0.72</td><td>0.71</td></tr><tr><td>absolute</td><td>✓</td><td>✘</td><td>✓</td><td>✓</td><td>0.74</td><td>0.71</td><td>0.73</td><td>0.72</td></tr><tr><td>standard</td><td>✓</td><td>✓</td><td>✓</td><td>✘</td><td>0.69</td><td>0.68</td><td>0.72</td><td>0.70</td></tr><tr><td>Co-VGGT (Ours)</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>0.74</td><td>0.73</td><td>0.76</td><td>0.74</td></tr></table>

![](images/ef1ae1933b87e57ce950ff9b71d1da737b3b58a9029745b3f2f47a5c9bc41585.jpg)

![](images/066879b8409265554c4acdc44e47f0d1ac506b176208f25bafa8a31549eeca4a.jpg)  
Fig. 6: MoE Gating weights. Average α parameter per-layer vs. layer id. on the multiview (left) and pairwise (right) task. We observe that specific layers are decisive for the final co-visibility analysis.

## 4.6 Layer Ablation

Tab. 5 shows that restricting the MoE expert pool to the last 12 layers preserves nearly full performance (0.74/0.72 IoU\*/AUC on Gibson), while using fewer layers causes consistent degradation. This is directly corroborated by the gating analysis (Fig. 6), where layers 0–12 receive near-zero average weight across both classes, confirming that the model has learned to ignore early layers. Since VGGT inference dominates runtime (50ms/150ms per 2/10 images on an H100) and the MoE head is lightweight, retaining all layers adds negligible overhead we therefore do so in our final model. Together, these findings provide taskgrounded empirical evidence for the hierarchical geometric reasoning hypothesized in [34]: early layers carry no discriminative signal for co-visibility, while late layers encode the high-level judgment of accepting or rejecting shared surface support, with positive and negative pairs peaking at distinct layers (L15 and L17, respectively). For completeness, we evaluate our approach using only the first 12 layers of the VGGT backbone (last row of Tab. 5). The resulting performance drops substantially below the state of the art, confirming that early layers encode more generic and diverse representations that are less informative for co-visibility reasoning.

![](images/f902919cbc47180306686a3c8591e2fc3ecc4306b442e3b56c538cd993fae023.jpg)

![](images/ddf40952094b0e2280487c204c5f7d74e0c94ecc98e9d9992459fa0d3f6c61c3.jpg)

![](images/f9e3e3b4ef46cad343ee798041deeb586dd06bf07568d92154fa2666b57c100e.jpg)  
Fig. 7: Calibration Measures. Calibration plot (left), score distribution (center) and summary metrics (right) on Gibson validation for pairwise and multiview tasks.

## 4.7 Cross-domain transfer

To probe whether the learned co-visibility signal is tied to a single training distribution, we evaluate Co-VGGT trained on one Co-VisiON environment and tested zero-shot on the other (Tab. 5). Transferring across Gibson-HM3D costs at most 0.04 IoU\* and 0.03 AUC in both pairwise and multiview settings, while still exceeding the in-domain state of the art on the target dataset. Combined with Co-VisiON’s scene-disjoint train/val protocol, this indicates the extracted signal does not overfit to a single environment, though both datasets remain indoor Habitat-rendered scenes and broader cross-domain generalization is left to future work.

## 4.8 Zero-Shot Ablation

As shown in Tab. 1, the zero-shot performance of Co-VGGT is unexpectedly strong, indicating that the backbone already encodes a substantial signal for identifying co-visible pairs.

To further investigate this behavior, Tab. 6 presents a zero-shot ablation where we vary the subset of VGGT backbone layers used for similarity computation and subsequent IoU and AUC evaluation (analogous to Tab. 5). Performance remains within a relatively narrow range across diferent subsets. However, across all tasks and datasets, the strongest results are consistently achieved when using embeddings from the last five layers. Additional analyzes and extended experiments are provided in Sec. A.1 of the Supplementary material.

Table 5: Ablation and cross-domain transfer results. Left: ablation on the number of VGGT expert-pool layers used in the MoE head, evaluated on Gibson. Right: cross-domain transfer between Gibson and HM3D.

<table><tr><td colspan="5">Layer-count ablation</td></tr><tr><td></td><td colspan="2">Multiview</td><td colspan="2">Pairwise</td></tr><tr><td>MoE Range</td><td> $IoU^* \uparrow$ </td><td> $AUC\uparrow$ </td><td> $IoU^* \uparrow$ </td><td> $AUC\uparrow$ </td></tr><tr><td>[1, 24] (Ours)</td><td>0.74</td><td>0.73</td><td>0.84</td><td>0.78</td></tr><tr><td>[14, 24]</td><td>0.74</td><td>0.72</td><td>0.84</td><td>0.77</td></tr><tr><td>[19, 24]</td><td>0.71</td><td>0.70</td><td>0.83</td><td>0.75</td></tr><tr><td>[21, 24]</td><td>0.70</td><td>0.67</td><td>0.83</td><td>0.75</td></tr><tr><td>[24]</td><td>0.69</td><td>0.66</td><td>0.82</td><td>0.74</td></tr><tr><td>[1, 12]</td><td>0.53</td><td>0.50</td><td>0.30</td><td>0.11</td></tr></table>

<table><tr><td colspan="6">Cross-domain transfer</td></tr><tr><td rowspan="2">Train</td><td rowspan="2">Test</td><td colspan="2">Pairwise</td><td colspan="2">Multiview</td></tr><tr><td> $IoU^* \uparrow$ </td><td> $AUC\uparrow$ </td><td> $IoU^* \uparrow$ </td><td> $AUC\uparrow$ </td></tr><tr><td>Gibson</td><td>Gibson</td><td>0.85</td><td>0.78</td><td>0.74</td><td>0.73</td></tr><tr><td>HM3D</td><td>HM3D</td><td>0.84</td><td>0.78</td><td>0.76</td><td>0.74</td></tr><tr><td>Gibson</td><td>HM3D</td><td>0.81</td><td>0.75</td><td>0.72</td><td>0.71</td></tr><tr><td>HM3D</td><td>Gibson</td><td>0.83</td><td>0.76</td><td>0.72</td><td>0.72</td></tr></table>

Table 6: Zero-shot layer ablation. Ablation on the number of VGGT output feature layers used for cosine similarity scoring in zero-shot evaluation, reported on both Gibson and HM3D datasets for Multiview and Pairwise tasks.

<table><tr><td rowspan="3">Layer Range</td><td colspan="4">Gibson</td><td colspan="4">HM3D</td></tr><tr><td colspan="2">Multiview</td><td colspan="2">Pairwise</td><td colspan="2">Multiview</td><td colspan="2">Pairwise</td></tr><tr><td> $IoU^*(\uparrow)$ </td><td> $AUC(\uparrow)$ </td><td> $IoU^*(\uparrow)$ </td><td> $AUC(\uparrow)$ </td><td> $IoU^*(\uparrow)$ </td><td> $AUC(\uparrow)$ </td><td> $IoU^*(\uparrow)$ </td><td> $AUC(\uparrow)$ </td></tr><tr><td>[1, 24] (Ours)</td><td>0.37</td><td>0.30</td><td>0.50</td><td>0.31</td><td>0.36</td><td>0.30</td><td>0.46</td><td>0.31</td></tr><tr><td>[9, 24]</td><td>0.37</td><td>0.30</td><td>0.50</td><td>0.32</td><td>0.36</td><td>0.30</td><td>0.46</td><td>0.31</td></tr><tr><td>[19, 24]</td><td>0.38</td><td>0.31</td><td>0.52</td><td>0.36</td><td>0.37</td><td>0.30</td><td>0.47</td><td>0.33</td></tr><tr><td>[21, 24]</td><td>0.36</td><td>0.30</td><td>0.49</td><td>0.34</td><td>0.36</td><td>0.30</td><td>0.45</td><td>0.30</td></tr></table>

## 5 Limitations

While Co-VGGT achieves strong results on the Co-VisiON benchmark, some limitations remain. We frame co-visibility prediction as a binary signal since the focus of this work is primarily on interpretability and understanding the overlap signal already present in VGGT, given that a simple co-visibility estimator already achieves strong performance. Nevertheless, we also conducted an auxiliary experiment that predicts, via a regression head, a graded co-visibility strength (see Supplementary material). We do not further develop this direction here; instead, we leave it as future work, where richer overlap estimates could provide more informative geometric constraints for downstream SfM and SLAM pipelines.

Moreover, the multiview setting exposes a structural limitation of our architecture: compressing multiview context into fixed-dimensional per-view embeddings introduces noise, and the model scores each pair independently without enforcing global graph consistency — leading to a performance gap relative to the pairwise setting. Further failure examples and analysis are provided in the Supplementary material.

Finally, the mechanistic interpretation of layer-wise gating — while empirically grounded — remains correlational: we identify which layers are decisive, but not entirely why specific geometric abstractions emerge at those depths in VGGT’s training. However, we argue that the emergent behavior of VGGT in 3D vision may mirror that of LLMs in NLP; consequently, several insights and solutions could potentially be transferred from that domain.

## 6 Conclusion

We presented Co-VGGT, a lightweight co-visibility predictor on frozen VGGT features, with well-calibrated pairwise probabilities (ECE=0.030) that are usable directly as visibility-graph edge weights. Our layer-wise mixture-of-experts head reveals that co-visibility reasoning emerges exclusively in VGGT’s late transformer blocks: L17 acts as a negative anchor that confidently rejects nonoverlapping pairs, while positive pairs rely on a broader set of late layers depending on the available context. This provides task-grounded evidence for hierarchical geometric reasoning in a geometry-grounded foundation model, mirroring the layer-specialization behavior observed in large language models.

These findings suggest that geometry-grounded foundation models encode spatial priors that are richer and more structured than previously understood

and that these priors can be eficiently distilled for downstream tasks without modifying the backbone. Future work will pursue four directions: (i) moving beyond binary prediction toward continuous overlap ratio estimation to supply richer geometric constraints for SfM and SLAM pipelines; (ii) replacing the current per-pair multiview loop with a unified aggregation mechanism — such as set transformers or token-level routing — to enforce global graph consistency and close the pairwise–multiview calibration gap; and (iii) integrating Co-VGGT within embodied mapping systems to enable uncertainty-aware loop closure and next-best-view planning grounded in geometric connectivity rather than appearance alone. (iv) Investigate whether interpretability findings from NLP transfer to geometric foundation models, enabling principled interpretability methods for this domain.

## Acknowledgements

We thank Fondazione Bruno Kessler (FBK) and the University of Padua, Department of Mathematics "Tullio Levi-Civita", for providing the computational resources used in this work. TC and LS were supported by the PNRR project Future Artificial Intelligence Research (FAIR, PE00000013) under the NRRP MUR program, funded by NextGenerationEU.

## References

1. An, V.D., Vu, M.N., Reid, I.D.: Improving robotic manipulation with eficient geometry-aware vision encoder. arXiv preprint arXiv:2509.15880 (2025). https: //doi.org/10.48550/arXiv.2509.15880

2. Arandjelovic, R., Gronat, P., Torii, A., Pajdla, T., Sivic, J.: Netvlad: Cnn architecture for weakly supervised place recognition. In: CVPR (2016)

3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report (2025), https://arxiv.org/abs/2502.13923

4. Bavle, H., Sanchez-Lopez, J.L., Civera, J., Voos, H.: Situational graphs for robot navigation in structured indoor environments. arXiv preprint arXiv:2202.12197 (2022)

5. Cabon, Y., Stofl, L., Antsfeld, L., Csurka, G., Chidlovskii, B., Revaud, J., Leroy, V.: Must3r: Multi-view network for stereo 3d reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1050–1060 (2025). https://doi.org/10.1109/CVPR52734.2025.00106

6. Campos, C., Elvira, R., Rodríguez, J.J.G., Montiel, J., Tardós, J.D.: Orb-slam3: An accurate open-source library for visual, visual–inertial, and multimap slam. In: T-RO. vol. 37, pp. 1874–1890 (2021)

7. Chen, A., Xu, Z., Zhao, J., Yu, J., Su, H., Zhang, J., Yu, J.: Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In: ICCV (2021)

8. Chen, C., Dang, N., Zhang, J., Sun, W., Zheng, P., He, X., Ye, Y., Zhang, J., Srinivas, T., Feng, C.: Co-vision: Co-visibility reasoning on sparse image sets of indoor scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. pp. 4861–4871 (October 2025)

9. Cheng, A.C., Yin, H., Fu, Y., Guo, Q., Yang, R., Kautz, J., Wang, X., Liu, S.: Spatial-rgpt: Grounded spatial reasoning in vision-language models. In: NeurIPS (2024)

10. Conneau, A., Kiela, D., Schwenk, H., Barrault, L., Bordes, A.: Supervised learning of universal sentence representations from natural language inference data. In: Proceedings of the 2017 conference on empirical methods in natural language processing. pp. 670–680 (2017)

11. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

12. Geva, M., Katz, U., Ben-Arie, A., Berant, J.: What’s in your head? emergent behaviour in multi-task transformer models. In: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. pp. 8201–8215 (2021)

13. Grover, A., Leskovec, J.: node2vec: Scalable feature learning for networks. In: Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. pp. 855–864 (2016)

14. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

15. Hughes, N., Chang, Y., Carlone, L.: Hydra: A real-time spatial perception system for 3d scene graph construction and optimization. arXiv preprint arXiv:2201.13360 (2022)

16. Jiang, H., Karpur, A., Cao, B., Huang, Q.: Omniglue: Generalizable feature matching with foundation model guidance. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 19865–19875 (2024). https://doi.org/10.1109/CVPR52733.2024.01878

17. Lee, S., Choi, J., Kang, I., Kim, J., Park, J., Shim, H.: 3d-aware vision-language models fine-tuning with geometric distillation. arXiv preprint arXiv:2506.09883 (2025). https://doi.org/10.48550/arXiv.2506.09883

18. Lindeberg, T.: Scale invariant feature transform. Scholarpedia 7, 10491 (05 2012). https://doi.org/10.4249/scholarpedia.10491

19. Lindenberger, P., Sarlin, P.E., Pollefeys, M.: Lightglue: Local feature matching at light speed. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 17581–17592 (2023). https://doi.org/10.1109/ ICCV51070.2023.01616

20. Meng, K., Bau, D., Andonian, A., Belinkov, Y.: Locating and editing factual associations in gpt. Advances in neural information processing systems 35, 17359–17372 (2022)

21. Mou, L., Men, R., Li, G., Xu, Y., Zhang, L., Yan, R., Jin, Z.: Natural language inference by tree-based convolution and heuristic matching. In: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). pp. 130–136 (2016)

22. OpenAI: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)

23. Ramakrishnan, S.K., Gokaslan, A., Wijmans, E., Maksymets, O., et al.: Habitatmatterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai. arXiv preprint arXiv:2109.08238 (2021)

24. Rosinol, A., Abate, M., Chang, Y., Carlone, L.: Kimera: an open-source library for real-time metric-semantic localization and mapping. IEEE International Conference on Robotics and Automation (ICRA) (2020)

25. Sarlin, P.E., DeTone, D., Malisiewicz, T., Rabinovich, A.: Superglue: Learning feature matching with graph neural networks. In: CVPR (2020)

26. Schonberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4104–4113 (2016)

27. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014)

28. Stary, M., Gaubil, J., Tewari, A., Sitzmann, V.: Understanding multi-view transformers. arXiv preprint arXiv:2510.24907 (2025)

29. Sucar, E., Liu, S., Ortiz, J., Davison, A.J.: imap: Implicit mapping and positioning in real-time. In: ICCV (2021)

30. Tang, Z., Fan, Y., Wang, D., Xu, H., Ranjan, R., Schwing, A., Yan, Z.: Mv-dust3r+: Single-stage scene reconstruction from sparse views in 2 seconds. arXiv preprint arXiv:2412.06974 (2024)

31. Team, G., Anil, R., Borgeaud, S., Alayrac, J.B., et al.: Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023)

32. Teed, Z., Deng, J.: Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. In: NeurIPS (2021)

33. Tenney, I., Das, D., Pavlick, E.: Bert rediscovers the classical nlp pipeline. In: Proceedings of the 57th annual meeting of the association for computational linguistics. pp. 4593–4601 (2019)

34. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotný, D.: Vggt: Visual geometry grounded transformer. In: IEEE/CVF Conference on Computer

Vision and Pattern Recognition (CVPR) 2025, Nashville, TN, USA, June 11-15, 2025. pp. 5294–5306. Computer Vision Foundation / IEEE (2025). https://doi. org/10.1109/CVPR52734.2025.00499

35. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: Dust3r: Geometric 3d vision made easy. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 20697–20709 (2024). https://doi. org/10.1109/CVPR52733.2024.01956

36. Wang, Y., He, X., Peng, S., Tan, D., Zhou, X.: Eficient loftr: Semi-dense local feature matching with sparse-like speed. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 21666–21675 (2024). https://doi.org/10.1109/CVPR52733.2024.02047

37. Weinzaepfel, P., Lucas, T., Leroy, V., Cabon, Y., Arora, V., Brégier, R., Csurka, G., Antsfeld, L., Chidlovskii, B., Revaud, J.: Croco v2: Improved cross-view completion pre-training for stereo matching and optical flow. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17969–17980 (2023)

38. Xia, F., Zamir, A.R., He, Z., Sax, A., Malik, J., Savarese, S.: Gibson env: Realworld perception for embodied agents. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 9068–9079 (2018)

39. Xu, Z., Fan, Y., Bao, J., Zhang, D., Li, H., Zhang, D.: Mvfusion: A multi-view difusion model for 3d reconstruction. In: CVPR (2023)

40. Xue, M., Huang, Q., Zhang, H., Cheng, L., Song, J., Wu, M., Song, M.: Protopformer: Concentrating on prototypical parts in vision transformers for interpretable image recognition. arXiv preprint arXiv:2208.10431 (2022)

41. Zhang, T., Usenko, V., Engel, J., Cremers, D.: Bad-slam: Bundle adjusted direct rgb-d slam. In: CVPR (2021)

42. Zhu, Z., Peng, S., Larsson, V., Lin, C.H., Bao, H., Dai, A., Nießner, M., Zhou, X.: Nice-slam: Neural implicit scalable encoding for slam. In: CVPR (2022)

# What VGGT Knows About Overlap: Probing Geometric Foundation Models for Co-Visibility

Filippo Ziliotto<sup>1,2</sup>, Luciano Serafini<sup>2</sup>, Lamberto Ballan<sup>1</sup>, and Tommaso Campari<sup>2</sup>

<sup>1</sup> University of Padova <sup>2</sup> Fondazione Bruno Kessler (FBK)

## A Supplementary Material

We provide supplementary material to complement the main paper. Sec. A.1 reports extended zero-shot ablations across layer subsets and datasets. Sec. A.2 evaluates the multiview-trained model under pairwise inference. Sec. A.6 defines the full set of evaluation metrics used throughout. Sec. A.3 presents qualitative failure analysis and scene-graph visualizations. Sec. A.4 ablates end-to-end VGGT fine-tuning as an upper-bound reference. Sec. A.7 explains the addition of the regression head to the current Co-VGGT approach.

## A.1 Zero-shot Ablation

As discussed in Sec. 4, we report additional zero-shot ablations in Tab. S1, where we carefully define the layer ranges used to construct the MoE heads. Notably, the best performance consistently emerges from the [19,24] range across all settings and datasets.

We do not investigate this behavior further. It may be partially influenced by the specific strategy adopted to extract representations in the zero-shot regime. Consequently, we do not claim that this choice is optimal, but rather present it as a reasonable and reproducible approach to address the zero-shot setting.

We further observe that the results in Tab. S1 exhibit only minor variations across diferent layer ranges, which may partially stem from randomness induced by diferent statistical seeds.

## A.2 Multiview on Pairwise data

Since pairwise evaluation is a special case of the multiview setting—where the model receives N views of the same scene—we evaluated Co-VGGT, trained in the multiview regime, under pairwise evaluation (Tab. S2). We observe a slight performance drop compared to standard multiview training and testing. We attribute this to a shift in the MoE dynamics (Fig. 6), which alters the layer weighting: some layers are required to produce outputs outside their primary specialization, leading to suboptimal performance.

Table S1: Zero-shot layer ablation. Ablation on the number of VGGT output feature layers used for cosine similarity scoring in zero-shot evaluation, reported on both Gibson and HM3D datasets for Multiview and Pairwise tasks.

<table><tr><td rowspan="3">Layer Range</td><td colspan="4">Gibson</td><td colspan="4">HM3D</td></tr><tr><td colspan="2">Multiview</td><td colspan="2">Pairwise</td><td colspan="2">Multiview</td><td colspan="2">Pairwise</td></tr><tr><td> $IoU^{*}(\uparrow)$ </td><td> $AUC(\uparrow)$ </td><td> $IoU^{*}(\uparrow)$ </td><td> $AUC(\uparrow)$ </td><td> $IoU^{*}(\uparrow)$ </td><td> $AUC(\uparrow)$ </td><td> $IoU^{*}(\uparrow)$ </td><td> $AUC(\uparrow)$ </td></tr><tr><td>[1, 24] (Ours)</td><td>0.37</td><td>0.30</td><td>0.50</td><td>0.31</td><td>0.36</td><td>0.30</td><td>0.46</td><td>0.31</td></tr><tr><td>[4, 24]</td><td>0.37</td><td>0.30</td><td>0.50</td><td>0.32</td><td>0.36</td><td>0.30</td><td>0.46</td><td>0.31</td></tr><tr><td>[9, 24]</td><td>0.37</td><td>0.30</td><td>0.50</td><td>0.32</td><td>0.36</td><td>0.30</td><td>0.46</td><td>0.31</td></tr><tr><td>[14, 24]</td><td>0.37</td><td>0.30</td><td>0.51</td><td>0.33</td><td>0.36</td><td>0.30</td><td>0.46</td><td>0.31</td></tr><tr><td>[16, 24]</td><td>0.36</td><td>0.31</td><td>0.51</td><td>0.34</td><td>0.36</td><td>0.30</td><td>0.46</td><td>0.31</td></tr><tr><td>[19, 24]</td><td>0.38</td><td>0.31</td><td>0.52</td><td>0.36</td><td>0.37</td><td>0.30</td><td>0.47</td><td>0.33</td></tr><tr><td>[21, 24]</td><td>0.36</td><td>0.30</td><td>0.49</td><td>0.34</td><td>0.36</td><td>0.30</td><td>0.45</td><td>0.30</td></tr><tr><td>[24]</td><td>0.34</td><td>0.30</td><td>0.41</td><td>0.29</td><td>0.34</td><td>0.29</td><td>0.37</td><td>0.27</td></tr></table>

Table S2: Multiview on Pairwise data. We evaluate Co-VGGT, trained on the multiview setting, on the pairwise dataset.

<table><tr><td rowspan="2">Method</td><td colspan="2">Gibson</td><td colspan="2">HM3D</td></tr><tr><td>IoU* (↑)</td><td>AUC (↑)</td><td>IoU* (↑)</td><td>AUC (↑)</td></tr><tr><td>Co-VGGT (Zero-shot)</td><td>0.50</td><td>0.31</td><td>0.46</td><td>0.31</td></tr><tr><td>Co-VGGT (Multiview)</td><td>0.77</td><td>0.59</td><td>0.77</td><td>0.58</td></tr><tr><td>Co-VGGT (Ours)</td><td>0.85</td><td>0.78</td><td>0.84</td><td>0.78</td></tr></table>

This suggests an interesting research direction, likely tied to designing architectures that natively support multiview reasoning rather than iterating over pairs, as currently done in Co-VGGT —a known limitation. We encourage further investigation of this behavior.

## A.3 Failure Analysis

We report several failure cases in Fig. S3. Importantly, the model remains capable of predicting highly overlapping views; failures predominantly arise in mediumto-edge cases. In the multiview setting, predicting whether two images overlap is generally easier when the model has access to the remaining N − 2 views of the scene, particularly those captured in close proximity to the queried pair (i, j). This additional context provides geometric cues that facilitate overlap reasoning. In contrast, when no auxiliary context is available and the pair (i, j) exhibits only minimal overlap, the task becomes significantly more challenging.

As discussed in Sec. 5, a known limitation is that the architecture is primarily designed for pairwise processing. From a practical standpoint, the multiview setting is implemented as a simple loop over all image pairs, rather than through a unified multiview design. We argue that a coherent architecture explicitly tailored for multiview reasoning would likely outperform the current pairwise formulation, as it could more efectively aggregate and leverage information across viewpoints.

![](images/33c8a39fc1e10ea99a59c3beb88817606c6617c18685031a3fe74a80c0979cd4.jpg)

Fig. S1: IoU vs. Thresholds. Plot of IoU across threshold for diferent datasets and settings.  
![](images/3ded6252e9bad9d5e8f77e31ff35aee63bdeda582514d9c0392538ebce197523.jpg)  
Fig. S2: Co-visibility Scene-Graph Failure Examples (Multiview). Given N input views, we visualize the ground-truth adjacency matrix (left), the predicted binary graph obtained by thresholding scores at the IoU-optimal τ <sup>∗</sup>. Each matrix is symmetric and entry (i, j) denotes the relation between image i and image j.

## A.4 VGGT Finetuning

We further compare Co-VGGT against a fully fine-tuned VGGT backbone, evaluated both with and without the MoE head (in the latter case, replaced by a standard MLP head). Results in Tab. S4 show that end-to-end fine-tuning with the MoE head achieves outstanding performance, reaching an IoU/AUC of 0.91/0.88 (last row), efectively saturating the co-visibility task. Although this demonstrates the strong capacity of the MoE formulation when coupled with full backbone adaptation, improving absolute performance is not the primary objective of this work. Our focus is instead on leveraging the MoE design as a principled and interpretable mechanism to analyze VGGT behavior. For this reason, we do not emphasize these fully fine-tuned results in the main paper.

Moreover, the gap between training only 7.5M parameters and fully finetuning a 1.2B-parameter backbone is substantial, both in scale and computational cost. In practice, adapting such a large model is largely impractical for standard SLAM or robotics matching pipelines, where eficiency, deployment constraints, and limited task-specific data typically preclude full end-to-end finetuning.

![](images/3f7574f59fc0e47f7e15a38cb7d0dce34dbb2c88c2ddf78bd3bddb7f9e6fc03a.jpg)

Fig. S3: Qualitative Examples. Subset of correctly predicted pairs (top) and failed pairs predictions (bottom). Failures are due primarly to the dificulty of the task, since pairs share very minimal overlap between each other.  
![](images/d0093fb2d641831d8f6a25a1dde92f208603d48b35fc3ef0702d04e7dba4e1de.jpg)  
Fig. S4: Attention Maps. Attention maps extracted from Layer 17 for pairs of noncovisible images. After focusing on objects in the first image, the attention maps for the second image appear sparse and difuse.

We note that all fine-tuning experiments were carried out with a $l r = 1 e ^ { - 5 }$ and a $w d = 1 e ^ { - 5 }$ , to avoid “disrupting” the weights learned before this finetuning.

## A.5 Downstream SfM

To test the claim that Co-VGGT scores are usable as visibility-graph weights, we use them as a pair-selection front-end for COLMAP [26] on 20 Gibson scenes. We score all $\binom { N } { 2 }$ pairs per scene and pass only the top-30% to feature matching, leaving the rest of the pipeline unchanged, and compare against exhaustive matching, random selection, and NetVLAD [2] retrieval under the same budget (Tab. S3). Co-VGGT recovers 83% of the registered images and 70% of the sparse points of exhaustive matching while using only 30% of the pairs, outperforming both Random and NetVLAD on coverage. Notably, it also attains the best geometric quality of all methods — including exhaustive — with the highest verification ratio (0.966 vs. 0.910) and lowest reprojection error (0.270 vs. 0.323) at 2.6× lower runtime. This is a precision efect: discarding low-overlap pairs before verification removes constraints likely to fail, leaving a cleaner pair set. We present this as evidence that Co-VGGT scores are usable for SfM pair selection, not as a full reconstruction benchmark.

## A.6 Additional Metrics

Beyond IoU/AUC, we report standard binary classification metrics computed over co-visibility edges (see Fig. S5). Accuracy is the fraction of correctly classified view pairs after thresholding predicted co-visibility scores $d _ { i j } \in [ 0 , 1 ]$ into labels $\hat { y } _ { i j } = \mathbb { I } [ d _ { i j } \geq \tau ] , \mathrm { i . e . , ( T P + T N ) / ( T P + T N + F P + F N ) }$ . Because co-visibility is typically imbalanced (most pairs are non-overlapping), accuracy can be dominated by true negatives and should be interpreted together with ranking-based curves. The ROC curve plots the True Positive Rate $\mathrm { ( T P R = T P / ( T P + F N ) } )$ against the False Positive Rate $\mathrm { ( F P R = F P / ( F P + T N ) } )$ as τ varies, measuring how well the model separates co-visible from non-co-visible pairs independent of a fixed operating point; the corresponding ROC-AUC summarizes this trade-of. The Precision-Recall (PR) curve instead plots Precision $( P = \mathrm { T P } / ( \mathrm { T P } + \mathrm { F P } ) )$ versus Recall (same as TPR) across thresholds and is often more informative under heavy class imbalance, as it focuses on the quality of predicted covisible edges. Finally, the $F _ { 1 }$ score is the harmonic mean of Precision and Recall, $\begin{array} { r } { F _ { 1 } = 2 \ \frac { \mathrm { P r e c i s i o n { \cdot } R e c a l l } } { \mathrm { P r e c i s i o n { + } R e c a l l } } } \end{array}$ , and provides a single-number summary at a chosen threshold reflecting the balance between missing true co-visible pairs (FN) and introducing spurious co-visible edges (FP), which directly impacts downstream matching and graph construction. For qualitative examples of IoU over various threhsolds see Fig. S3.

Table S3: SfM pair-selection Experiment. Averaged over a small subset of 20 Gibson scenes.

<table><tr><td>Selection</td><td>Budget</td><td>Reg. imgs ↑</td><td>Points ↑</td><td>Verif. ratio ↑</td><td>Reproj. ↓</td><td>Time (s) ↓</td></tr><tr><td>Exhaustive</td><td>100%</td><td> $12.8 \pm 6.9$ </td><td> $3140 \pm 2292$ </td><td> $0.910 \pm 0.055$ </td><td> $0.323 \pm 0.117$ </td><td> $97.0 \pm 60.6$ </td></tr><tr><td>Random top- $k$ </td><td>30%</td><td> $6.1 \pm 4.3$ </td><td> $553 \pm 506$ </td><td> $0.917 \pm 0.100$ </td><td> $0.301 \pm 0.072$ </td><td> $\textbf{37.3} \pm \textbf{19.5}$ </td></tr><tr><td>NetVLAD top- $k$ </td><td>30%</td><td> $9.8 \pm 4.8$ </td><td> $1945 \pm 1577$ </td><td> $0.954 \pm 0.056$ </td><td> $0.291 \pm 0.068$ </td><td> $40.9 \pm 19.3$ </td></tr><tr><td>Co-VGGT top- $k$ </td><td>30%</td><td> $\textbf{10.6} \pm \textbf{4.4}$ </td><td> $\textbf{2086} \pm \textbf{1385}$ </td><td> $\textbf{0.966} \pm \textbf{0.042}$ </td><td> $\textbf{0.270} \pm \textbf{0.058}$ </td><td> $37.8 \pm 19.1$ </td></tr></table>

Table S4: VGGT Finetuning Ablation (Gibson). Comparison between multiview and pairwise evaluation settings. Results are reported on Gibson for both settings. <sup>†</sup>We train the MoE and the full backbone VGGT model.

<table><tr><td rowspan="2">Method</td><td rowspan="2">MoE Head</td><td rowspan="2">Trainable Parameters</td><td colspan="2">Multiview</td><td colspan="2">Pairwise</td></tr><tr><td>IoU* (↑)</td><td>AUC (↑)</td><td>IoU* (↑)</td><td>AUC (↑)</td></tr><tr><td>Co-VGGT (Zero-shot)</td><td>×</td><td>0</td><td>0.37</td><td>0.30</td><td>0.50</td><td>0.31</td></tr><tr><td>VGGT [34] (Fine-tune)</td><td>×</td><td>80M</td><td>0.72</td><td>0.69</td><td>0.80</td><td>0.72</td></tr><tr><td>VGGT [34] (Fine-tune)</td><td>×</td><td>175M</td><td>0.72</td><td>0.69</td><td>0.80</td><td>0.73</td></tr><tr><td>VGGT [34] (Fine-tune)</td><td>×</td><td>335M</td><td>0.72</td><td>0.70</td><td>0.81</td><td>0.74</td></tr><tr><td>VGGT [34] (Fine-tune)</td><td>×</td><td>1.2B</td><td>0.72</td><td>0.71</td><td>0.84</td><td>0.81</td></tr><tr><td>Co-VGGT†(Fine-tune)</td><td>√</td><td>1.2B</td><td>0.76</td><td>0.72</td><td>0.91</td><td>0.88</td></tr><tr><td>Co-VGGT (Ours)</td><td>√</td><td>7.5M</td><td>0.74</td><td>0.73</td><td>0.85</td><td>0.78</td></tr></table>

## A.7 Strength Degree Prediction

In addition to binary co-visibility classification, we optionally augment Co-VGGT with a lightweight strength regression branch that predicts a continuous overlap score in [0, 1]. For each image pair, the dataloader provides a scalar target derived from the scene’s continuous relationship matrix when available (otherwise defaulting to the binary label). The regression branch mirrors the classification pathway: it is fed the same pairwise features computed from VGGT embeddings (under the chosen aggregation) and produces a scalar strength logit. In the multi-layer setting, this prediction is computed per selected layer and then aggregated across layers using the same strategy as classification—either a layer-wise MoE (softmax gating) or uniform averaging—so regression is consistent with the model’s layer reasoning mechanism rather than being attached only at the final stage. During training, the predicted strength is obtained via a sigmoid and optimized with a standard regression loss (MSE), combined with the classification objective using a tunable weight; when disabled, the model reduces to the original classification-only formulation.

![](images/3d29edcbe8879f3a229dab4409f33f7e988cd8709ee0f6bef3ab4fe2d90dbecd.jpg)  
Fig. S5: Additional Metrics. Other metrics reported in Sec. A.6 for Pairwise and Multiview in HM3D dataset. Resulted are reported on a subset of the validation data.

Table S5: Pair strenght regression. Comparison between multiview and pairwise evaluation settings. Results are reported on Gibson for both settings.

<table><tr><td rowspan="2">Method</td><td rowspan="2">MoE Head</td><td rowspan="2">Trainable Parameters</td><td colspan="2">Multiview</td><td colspan="2">Pairwise</td></tr><tr><td>IoU* (↑)</td><td>AUC (↑)</td><td>IoU* (↑)</td><td>AUC (↑)</td></tr><tr><td>Co-VGGT (Ours)</td><td>√</td><td>7.5M</td><td>0.74</td><td>0.73</td><td>0.85</td><td>0.78</td></tr><tr><td>Co-VGGT + regression</td><td>√</td><td>12M</td><td>0.74</td><td>0.72</td><td>0.84</td><td>0.81</td></tr></table>

As shown in Tab. S5, the results are largely consistent with those obtained using the non-regression formulation. We attribute this to the intrinsic dificulty of the regression task. Estimating a precise score in the range [0,1] that reflects the overall percentage of overlap between two images is inherently ambiguous; for instance, values such as 0.2 or 0.3 could both be considered reasonable estimates, despite representing substantially diferent targets in a regression setting. Consequently, the regression formulation provides limited additional benefit.

Moreover, a large fraction of image pairs are non-covisible, which automatically assigns a regression target of 0. Such samples are largely non-informative during training and contribute little useful signal for the backpropagation process.

Table S6: Co-VGGT performance across dificulty levels (AUC). Left: edgelevel dificulty defined by pairwise image overlap $( \mathrm { E a s y } \geq 5 0 \%$ Med. 10–50%, Hard $< 1 0 \% )$ . Right: graph-level dificulty defined by scene sparsity via average overlap (Easy $\geq 1 0 \%$ , Med. 4–10%, Hard $< 4 \% )$ ). Metrics report Graph AUC.  
(a) Image overlap (edge-level).

<table><tr><td>Method</td><td>Easy</td><td>Med.</td><td>Hard</td><td>Avg.</td></tr><tr><td>Co-VGGT (Pair)</td><td>0.74</td><td>0.86</td><td>0.68</td><td>0.73</td></tr><tr><td>Co-VGGT (Multi)</td><td>0.90</td><td>0.89</td><td>0.77</td><td>0.79</td></tr></table>

(b) Scene sparsity (graph-level).

<table><tr><td>Method</td><td>Easy</td><td>Med.</td><td>Hard</td><td>Avg.</td></tr><tr><td>Co-VGGT (Multi)</td><td>0.94</td><td>0.88</td><td>0.68</td><td>0.71</td></tr><tr><td>Co-VGGT (Pair)</td><td>0.96</td><td>0.90</td><td>0.77</td><td>0.79</td></tr></table>