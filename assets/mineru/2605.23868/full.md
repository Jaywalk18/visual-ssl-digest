# VISION TRANSFORMERS NEED BETTER TOKEN INTERACTION

Su Linxiang

University of Szeged

su\_linxiang@126.com

# ABSTRACT

Vision Transformers (ViTs) can learn strong image-level representations while their patch representations become less effective for dense prediction during prolonged training. We revisit this dense degradation phenomenon and argue that it is not fully explained by high-norm artifacts alone. Instead, we characterize semantic diffusion: an optimization shortcut in which global semantic information spreads through patch tokens beyond what is locally justified. Our analysis shows that dense representation quality is not captured by locality alone: shallow features can remain better aligned with foreground regions yet underperform deeper features, and [CLS] features remain complementary for dense prediction. These observations suggest that the goal should not be to remove global context, but to make token interactions more selective. We therefore study sparse attention as a minimal intervention, replacing softmax attention with entmax-1.5 while preserving global token connectivity. On DINOv1 ViT-S/16 trained for 200 epochs on ImageNet-1K, this change preserves ImageNet linear probing accuracy and substantially improves semantic segmentation performance: VOC mIoU increases from 42.80 to 48.78, ADE20K from 19.85 to 21.97, and Cityscapes from 36.79 to 37.87. These results suggest that selective token mixing is a simple and effective bias for improving dense ViT representations.

Image   
![](images/4ef34db5077614a540dbb6daeda5dc584dca535da0401f551c2803d71dde4e35.jpg)

<details>
<summary>natural_image</summary>

Airplane in flight against a clear blue sky, no visible text or symbols on the aircraft body
</details>

Naive   
![](images/08ad1104e37978165305b13d47288265c7b5a9ec9d334bc72a1c654aa1769b4f.jpg)

<details>
<summary>natural_image</summary>

Abstract color gradient image with no text or symbols
</details>

Register   
![](images/80cac65a97604f91a0466772e4b1fbd5e791bd986f9ab93a384feb917327faab.jpg)

<details>
<summary>natural_image</summary>

Abstract pixelated image with red, blue, green, and yellow colors (no text or symbols)
</details>

Specialization   
![](images/78df00bdccb8215d1850c07c4c219328a48d2f1af552ee784ecd4005e0210492.jpg)

<details>
<summary>natural_image</summary>

Abstract color gradient image with red, blue, green, and yellow hues (no text or symbols)
</details>

Entmax All-Layer   
![](images/c1bb04f22efa43b643b36731129e15fcabf3b20f6405936f9e8f1386e5120409.jpg)

<details>
<summary>natural_image</summary>

Pixelated abstract image with green, blue, orange, and purple hues, no visible text or symbols
</details>

Figure 1: Visualization of the impact of our proposed sparse attention, compared with other baselines. We display the first PCA components of model outputs in RGB.

# 1 Introduction

Vision Transformers (ViTs) have become a standard architecture for learning transferable visual representations [1, 2, 3]. Beyond strong image-level recognition performance, a desirable ViT representation should also preserve spatially meaningful patch features that transfer effectively to dense prediction tasks such as semantic segmentation and monocular depth estimation.

However, recent studies suggest that these objectives can diverge during training. As training progresses, image-level performance may continue improving while dense representations gradually become noisier and less effective for downstream dense tasks. This phenomenon has been associated with artifacts or outlier tokens in ViTs [4, 5]. In this work, we argue that such artifacts are likely a symptom rather than the root cause. Instead, we hypothesize that the main issue is an optimization shortcut in which global semantic information diffuses into patch tokens beyond what is locally justified, a phenomenon we term semantic diffusion.

LaSt-ViT attributes dense degradation to the combination of coarse image-level supervision and unrestricted global token interactions, which encourages background patches to participate in representing global semantics [5]. Other approaches attempt to mitigate this issue through register tokens, token specialization, attention modifications, or dense relational supervision [4, 6, 3, 7]. While effective, these methods do not directly address the shortcut itself. Register tokens mainly provide dedicated storage for leaked information, heuristic token filtering relies on manually designed criteria, and teacher-based dense constraints require an additional dense-good teacher model.

Our starting point is that the objective should not be to eliminate global context or suppress background tokens altogether. Background regions often contain useful contextual cues, and globally shared semantics can still benefit downstream tasks. Instead, the goal should be to encourage useful token interactions: interactions that preserve local structure while selectively incorporating global contextual information.

Motivated by this perspective, we study sparse attention as a simple and direct intervention on attention geometry. Rather than restricting receptive fields through local windows [8] or introducing additional register tokens [4], we retain global connectivity but replace dense softmax attention with entmax-1.5 [9, 10]. This produces sparse and selective attention distributions through only a minimal architectural modification.

# 2 Related Work

Artifacts in ViTs. ViT patch features can exhibit high-norm outliers, noisy PCA visualizations, or degraded densetask performance despite strong image-level recognition ability. DINOv2 popularized register tokens as a practical mechanism for improving dense representations by introducing additional storage slots [4, 3].

Recent work further suggests that such artifacts may emerge systematically from attention dynamics rather than random optimization failures. In large language models, systematic outliers have been linked to the softmax attention mechanism itself, where they act as implicit context-aware scaling factors [11]. In vision transformers, LAST-ViT attributes artifacts to a lazy aggregation behavior, where semantically irrelevant background patches are exploited as shortcuts for representing global semantics [5].

Our work is aligned with these perspectives. However, instead of managing artifacts after they emerge, we focus on modifying the token interaction mechanism itself to make semantic diffusion less favorable during optimization.

Dense Performance Degradation. Recent studies observe that prolonged self-supervised training can continuously improve global recognition performance while simultaneously degrading dense representations. This phenomenon was first implicitly observed in DINOv2 [3] and later studied more explicitly in DINOv3 [7]. DINOv3 proposes Gram anchoring, where an earlier checkpoint with stronger dense performance is used as a teacher to regularize later training stages, achieving strong empirical improvements. In contrast, we focus on understanding and mitigating the underlying optimization behavior itself, which we term semantic diffusion. Importantly, our approach is orthogonal to Gram anchoring and can potentially improve the quality of the teacher representations as well.

CLS–Patch Interaction. The CLS token and patch tokens play inherently different roles but are typically processed through the same transformer blocks. Recent work suggests that ViTs implicitly attempt to separate these token types and that explicit CLS/patch specialization can improve dense representations [6]. Our findings are consistent with this perspective: modifying CLS–patch interaction appears more effective than directly constraining final patch representations. Unlike prior specialization approaches, we intervene at the attention normalization level rather than introducing separate processing branches or additional parameters.

Sparse Attention. Sparse attention has primarily been explored for efficient sequence modeling and scalable transformers. Entmax and adaptive sparse transformers demonstrate that replacing softmax with sparse probability mappings can produce selective attention distributions while remaining fully differentiable [9, 10]. In ViTs, sparsity has often been introduced for computational or architectural efficiency [12, 13]. In contrast, we use sparsity as a representational bias rather than a compute optimization. Our goal is to make semantic diffusion less favorable during optimization by encouraging more selective token interactions.

![](images/76f466cdd44717b82826bb5b58327d10cb85b2e516b27bf0c15867b0e44d1ac2.jpg)  
(a) Layer/epoch heatmap of point-in-box (PiB).

![](images/1b41d52a07d5332a1641558ddedd10a8c78ecefda794167cc642d79388e91ab8.jpg)

<details>
<summary>line</summary>

| pretraining epoch | L3 patch | L3 patch+CLS | L9 patch | L9 patch+CLS | L6 patch | L6 patch+CLS | L12 patch | L12 patch+CLS |
| ----------------- | -------- | ------------ | -------- | ------------ | -------- | ------------ | --------- | ------------- |
| 10                | 14.0     | 15.0         | 33.0     | 34.0         | 21.0     | 25.0         | 38.0      | 40.0          |
| 25                | 17.0     | 18.0         | 45.0     | 47.0         | 26.0     | 29.0         | 50.0      | 52.0          |
| 50                | 16.0     | 17.0         | 48.0     | 50.0         | 28.0     | 32.0         | 51.0      | 53.0          |
| 75                | 16.0     | 17.0         | 49.0     | 52.0         | 29.0     | 33.0         | 48.0      | 54.0          |
| 100               | 16.0     | 17.0         | 49.0     | 52.0         | 30.0     | 34.0         | 46.0      | 55.0          |
| 125               | 16.0     | 17.0         | 49.0     | 52.0         | 31.0     | 35.0         | 46.0      | 56.0          |
| 150               | 16.0     | 17.0         | 50.0     | 53.0         | 32.0     | 36.0         | 45.0      | 57.0          |
| 175               | 16.0     | 17.0         | 51.0     | 54.0         | 33.0     | 37.0         | 44.0      | 58.0          |
| 200               | 16.0     | 17.0         | 51.0     | 55.0         | 34.0     | 38.0         | 44.0      | 59.0          |
</details>

(b) VOC dense-probe mIoU across pretraining epochs and feature layers.   
Figure 2: Relationship between CLS spatial grounding and dense prediction quality across training. Left: PiB heatmap showing how well CLS-aligned patches remain localized within foreground regions. Right: VOC dense probing performance across layers and epochs.

# 3 Method

# 3.1 Semantic Diffusion as an Optimization Shortcut

We hypothesize that under global training objectives, ViTs exploit unrestricted token interactions to distribute classrelevant semantics across patch tokens in a locally inconsistent manner. This optimization shortcut can improve global image-level recognition while gradually degrading patch-level semantic structure, especially for dense prediction tasks.

To study this behavior, we analyze point-in-box (PiB), defined as the proportion of images where the patch with the highest cosine similarity to the [CLS] token lies inside the foreground bounding box [5]. Intuitively, higher PiB indicates stronger spatial alignment between [CLS] semantics and foreground regions.

As shown in Figure 2a, shallow layers consistently exhibit higher PiB scores during training, whereas deeper layers show progressively weaker localization behavior. However, this trend does not imply that shallow representations are more useful for dense prediction. To further investigate this, we evaluate dense representations from different transformer layers on VOC semantic segmentation, optionally concatenating patch features with the corresponding [CLS] feature.

Figure 2b reveals two important observations. First, despite stronger PiB scores, shallow-layer features perform substantially worse on dense tasks. This suggests that locality alone is insufficient for high-quality dense representations; semantic richness is also necessary. Second, concatenating the [CLS] feature consistently improves dense performance and largely mitigates the degradation observed in the final transformer layers. This indicates that although semantic information diffuses through patch tokens during training [7], the [CLS] representation itself still contains complementary global information useful for downstream dense prediction.

These observations suggest that an effective intervention should not simply remove global context, for example through local window attention [8]. Likewise, enforcing hand-designed interaction rules, such as manually filtering stable tokens for CLS aggregation [5], may overly constrain optimization behavior. Instead, the goal should be to preserve global receptive fields while encouraging more selective and meaningful token interactions.

Motivated by this perspective, we explore sparse attention as a simple and direct intervention on token interaction geometry.

<table><tr><td>Method</td><td>ImageNet LP Top-1 ↑</td><td>VOC mIoU ↑</td><td>ADE20K mIoU ↑</td><td>Cityscapes mIoU ↑</td><td>NYUv2 RMSE ↓</td><td>SUN RGB-D RMSE ↓</td><td>KITTI RMSE ↓</td></tr><tr><td>CLS baseline</td><td>69.50</td><td>42.80</td><td>19.85</td><td>36.79</td><td>0.834</td><td>0.674</td><td>6.743</td></tr><tr><td>Register-4</td><td>70.03</td><td>44.34</td><td>19.52</td><td>36.00</td><td>0.814</td><td>0.667</td><td>6.749</td></tr><tr><td>Specialization</td><td>70.04</td><td>46.22</td><td>20.51</td><td>36.45</td><td>0.829</td><td>0.660</td><td>6.723</td></tr><tr><td>Entmax-1.5</td><td>70.06</td><td>48.78</td><td>21.97</td><td>37.87</td><td>0.829</td><td>0.669</td><td>6.739</td></tr></table>

Table 1: Evaluation of DINOv1 ViT-S/16 models trained for 200 epochs on ImageNet-1K. Dense metrics report the best validation performance during probe training.

# 3.2 Sparse Attention

Given queries Q, keys K, and values V , standard attention computes

$$
A = \operatorname{softmax} (Q K ^ {\top} / \sqrt {d}), \quad Y = A V.
$$

We replace the softmax normalization with entmax-1.5:

$$
A = \operatorname{entmax} _ {1. 5} (Q K ^ {\top} / \sqrt {d}), \quad Y = A V.
$$

For an input score vector z ∈ Rn, entmax-1.5 is defined as

$$
\mathrm{entmax} _ {1. 5} (z) _ {i} = \left[ \frac {z _ {i} - \tau (z)}{2} \right] _ {+} ^ {2},
$$

where [x]+ = max(x, 0), and the threshold τ (z) is chosen such that

$$
\sum_ {i = 1} ^ {n} \operatorname{entmax} _ {1. 5} (z) _ {i} = 1.
$$

Unlike windowed attention, sparse attention preserves the full global candidate set for every token. Unlike register tokens, it introduces no additional storage tokens. Unlike heuristic filtering approaches such as LaSt-ViT, it does not rely on manually designed token selection rules based on low-pass stability estimates. Instead, the intervention only changes how attention mass is distributed: low-utility interactions can receive exact zero weight, encouraging more selective token communication and making semantic diffusion less favorable during optimization.

In our experiments, we adopt the simplest possible setting and replace softmax with entmax-1.5 in all transformer layers without introducing any additional architectural modification or artifact-handling mechanism.

# 4 Experiments

# 4.1 Experimental Settings

We evaluate DINOv1 [2] ViT-S/16 models trained for 200 epochs on ImageNet-1K [14]. All methods are evaluated using the same training and probing protocols. Following the evaluation protocol of DINOv2 [3], we consider both global image-level recognition and dense prediction benchmarks.

Compared methods. We compare four variants. The CLS baseline is the standard DINOv1 ViT-S/16 model with softmax attention. Register-4 adds four register tokens following prior work on improving ViT dense features [4, 3]. Specialization introduces explicit CLS/patch specialization to encourage different processing for the [CLS] token and patch tokens [6]. Entmax-1.5 replaces softmax attention with entmax-1.5 in all transformer layers, without adding extra tokens, separate processing branches, or additional dense supervision.

Image-level recognition. For the global task, we perform linear probing on ImageNet classification. A linear classifier is trained on top of the frozen [CLS] feature using SGD for 12,500 iterations with random-resized-crop augmentation. We perform a learning-rate grid search over $\{ 1 0 ^ { - 5 } , 2 \times 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 2 \times 1 \hat { 0 } ^ { - 4 } , 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 2 \times 1 0 ^ { - 3 } , \hat { 5 } \times 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 2 \times 1 0 ^ { { \overset { \sim } { - } 2 } } , 5 \times 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \}$ , and report the best validation top-1 accuracy.

Semantic segmentation. For dense semantic prediction, we evaluate on VOC [15], ADE20K [16, 17], and Cityscapes [18]. We train a linear segmentation head for 40,000 iterations with learning rate 10−3. The head is applied to frozen patch features after the final layer normalization, followed by a trained batch normalization layer. We report mean Intersection-over-Union (mIoU), where higher is better.

Depth estimation. For monocular depth estimation, we evaluate on NYUv2 [19], SUN RGB-D [20], and KITTI [21]. We train a linear depth probe for 38,400 iterations with learning rate 10−3. Following the DINOv2 protocol, the probe input consists of patch and [CLS] features from four evenly spaced transformer layers of the frozen backbone, without applying the final layer normalization. We report root mean squared error (RMSE), where lower is better.

# 4.2 Quantitative Results

Table 1 shows that replacing softmax attention with entmax-1.5 improves dense representations without sacrificing global recognition performance. Entmax-1.5 achieves the best ImageNet linear probing accuracy among the evaluated methods, improving over the CLS baseline from 69.50% to 70.06%.

The gains are most pronounced for semantic segmentation. Compared with the CLS baseline, Entmax-1.5 improves VOC by 5.98 mIoU, ADE20K by 2.12 mIoU, and Cityscapes by 1.08 mIoU. It also outperforms both Register-4 and CLS/patch specialization on all three segmentation benchmarks. This suggests that sparse attention improves the spatial quality of patch representations while preserving the semantic information required for image-level recognition.

Depth estimation results are more moderate. Entmax-1.5 still improves over the CLS baseline on all three depth benchmarks, reducing RMSE by 0.005 on NYUv2, 0.005 on SUN RGB-D, and 0.004 on KITTI. However, it does not consistently outperform specialized baselines such as Register-4 or CLS/patch specialization. One possible reason is that our depth probing protocol already uses patch features from four transformer layers concatenated with the [CLS] feature. As discussed in Section 2, [CLS] and patch tokens tend to contain complementary information; concatenating them may therefore partially compensate for semantic diffusion even in the baseline model. Under this relatively strong probing setup, sparse attention still provides consistent improvements over the naive CLS baseline and remains competitive with more specialized methods.

Overall, the quantitative results support our main hypothesis: encouraging selective token interactions can improve dense representations, especially for semantic segmentation, while maintaining or slightly improving global image recognition.

# 4.3 Qualitative Results

For qualitative evaluation, we visualize patch features using PCA in Figure 1. Compared with the CLS baseline, Register-4, and CLS/patch specialization, Entmax-1.5 produces cleaner and more spatially coherent feature maps. Foreground regions are more clearly separated from background regions, and object boundaries are better preserved. These visualizations are consistent with the quantitative segmentation results and suggest that sparse attention reduces noisy semantic diffusion across unrelated patches.

# 5 Discussion

Entmax-1.5 is not intended as a definitive solution for token interaction. Rather, it serves as a compact instantiation of a broader design principle: harmful semantic diffusion can be reduced by making attention mixing more selective while preserving global connectivity. Future work could explore more flexible sparsification mechanisms, such as layer-adaptive or input-dependent attention sparsity [22], or combine sparse attention with potentially complementary interaction mechanisms such as explicit CLS/patch specialization [6]. Nevertheless, our results show that even a lightweight change to attention normalization can improve dense representations while maintaining global recognition performance.

Limitations. This study is currently limited to DINOv1 ViT-S/16 trained for 200 epochs on ImageNet-1K. We have not yet evaluated larger backbones, longer pretraining schedules, or multi-seed stability. The gains on depth estimation are modest compared with semantic segmentation, and entmax attention may introduce implementation-dependent runtime overhead (although there are highly efficient implementations [23, 24]). We therefore view sparse attention as evidence for the broader principle of selective token interaction, rather than as a final architecture design.

# 6 Conclusion

We revisit the dense degradation phenomenon in Vision Transformers and provide evidence that semantic diffusion is an important optimization shortcut behind this behavior. Motivated by this perspective, we propose sparse attention as a simple intervention that makes token interactions more selective without removing the global receptive field. By replacing softmax with entmax-1.5, our method introduces no additional labels, storage tokens, or teacher models. Experiments on DINOv1 ViT-S/16 show that this minimal modification improves dense prediction performance, especially for semantic segmentation, while preserving image-level recognition accuracy.

# References

[1] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.   
[2] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.   
[3] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.   
[4] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In International Conference on Learning Representations, 2024.   
[5] Cheng Shi, Yizhou Yu, and Sibei Yang. Vision transformers need more than registers, 2026.   
[6] Alexis Marouani, Oriane Siméoni, Hervé Jégou, Piotr Bojanowski, and Huy V. Vo. Revisiting [CLS] and patch token interaction in vision transformers. In International Conference on Learning Representations, 2026.   
[7] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. DINOv3, 2025.   
[8] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows, 2021.   
[9] Ben Peters, Vlad Niculae, and André F. T. Martins. Sparse sequence-to-sequence models. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.   
[10] Gonçalo M. Correia, Vlad Niculae, and André F. T. Martins. Adaptively sparse transformers. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 2019.   
[11] Yongqi An, Xu Zhao, Tao Yu, Ming Tang, and Jinqiao Wang. Systematic outliers in large language models, 2025.   
[12] Nabil Ibtehaz, Ning Yan, Masood Mortazavi, and Daisuke Kihara. Acc-vit : Atrous convolution’s comeback in vision transformers, 2024.   
[13] Xuanyao Chen, Zhijian Liu, Haotian Tang, Li Yi, Hang Zhao, and Song Han. Sparsevit: Revisiting activation sparsity for efficient high-resolution vision transformer, 2023.   
[14] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3):211–252, 2015.   
[15] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results. http://www.pascalnetwork.org/challenges/VOC/voc2012/workshop/index.html.   
[16] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5122–5130, 2017.

[17] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset, 2018.   
[18] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.   
[19] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012.   
[20] Shuran Song, Samuel P. Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2015.   
[21] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. International Journal of Robotics Research (IJRR), 2013.   
[22] Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free, 2025.   
[23] Nuno Gonçalves, Marcos V Treviso, and Andre Martins. Adasplash: Adaptive sparse flash attention. In Forty-second International Conference on Machine Learning, 2025.   
[24] Nuno Gonçalves, Hugo Pitorro, Vlad Niculae, Edoardo Ponti, Lei Li, Andre Martins, and Marcos V Treviso. Adasplash-2: Faster differentiable sparse attention. In Forty-third International Conference on Machine Learning, 2026.