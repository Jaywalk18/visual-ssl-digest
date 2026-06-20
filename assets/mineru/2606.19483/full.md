# LEAP: Layer-skipping Efficiency via Adaptive Progression for Vision Transformer Distillation

Jiaqi Zhang1

Ashton Lee2

Anthony Wong1

John Zou1

Sami BuGhanem1

Randall Balestriero1

1Brown University 2Rice University

{jiaqi\_zhang6, anthony\_g\_wong, john\_zou, sami\_bou\_ghanem,

randall\_balestriero}@brown.edu

awl10@rice.edu

## Abstract

Vision Foundation Models (VFMs) with Vision Transformer (ViT) backbones, such as DINOv2, have become essential for downstream tasks like object recognition and semantic segmentation. The immense computational requirements of backbones often necessitate distillation into smaller architectures for edge deployment. Feature-based knowledge distillation (KD) often suffers from the teacher-student gap; the student struggles to imitate teacher’s complex feature map due to its limited capacity. To mitigate this bottleneck, we propose LEAP: Layer-skipping Efficiency via Adaptive Progression, a training curriculum for ViT feature-based knowledge distillation. By utilizing the teacher’s intermediate feature maps as a sequence of progressively more difficult targets, our curriculum allows the student to build a foundational representation before tackling higher-level abstractions. Our results demonstrate that this paradigm significantly accelerates convergence through adaptive difficulty selection across various student model sizes and dataset scales. With our curriculum, the LEAP-distilled ViT-S achieves 90.1% accuracy on ImageNet-100, a +12.24% improvement compared with baseline. On ImageNet-1K, LEAP achieves +3.84% and +7.75% improvement for the instance retrieval task on the Oxford and Paris datasets, respectively. Furthermore, the curriculum enables 25.1% savings in training FLOPs and 21% savings in training time on ImageNet-100 by implementing early-stopping for teacher inference during the initial stages of training. Code is available at https://github.com/KevinZ0217/LEAP

## 1 Introduction

Vision Transformers (ViT) have transformed computer vision by replacing convolutional hierarchies with spatial self-attention mechanisms that treat image patches as tokens [4]. While large-scale Vision Foundation Models (VFMs) like DINOv2 achieve state-of-the-art performance, their massive parameter counts—often reaching the scale of ViT-Giant (2B parameters) or ViT-Huge (700M parameters)—make them impractical for deployment on resource-constrained edge devices. Knowledge distillation (KD) is a common solution to distill these "teacher" models into compact "student" models, such as ViT-Small (22m parameters) [24]. Feature-based KD is especially effective for ViTs, as it forces the student to mimic the teacher’s intermediate latent representations rather than just final classification logits [19]. This approach allows the distilled student to retain the teacher’s versatility across various downstream tasks, including classification, retrieval, and segmentation.

However, a fundamental challenge remains: the teacher-student gap. Research suggests that as the teacher model grows larger, the performance of a small student model often degrades [6]. Because the student has significantly lower representational capacity and a lower-rank feature space than the teacher. Attempting to match the complex, high-dimensional feature maps of a 40-layer ViT-G teacher in a single step causes unstable training and slow convergence; the student struggles to learn the teacher’s final abstractions before it has mastered basic spatial structures.

![](images/8b3bc9a0f26967218a1ea5e16e310180fc28230a5c3521117020cc84a3f2d1c0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    subgraph Stage 1[Stage 1 S_L targets T₁]
  A["Teacher ViT"] --> B["L2 Loss"]
  B --> C["Online CKA"]
  C --> D["..."]
    end

    subgraph Stage 2[Stage 2 S_L targets T₂]
  E["Teacher ViT"] --> F["L2 Loss"]
  F --> G["Online CKA"]
  G --> H["..."]
    end

    subgraph Stage L[Stage L S_L targets Tₗ]
  I["Teacher ViT"] --> J["L2 Loss"]
  J --> K["Online CKA"]
  K --> L["..."]
    end

  M["Easy [Shallow Teacher Feature Maps"]] --> N["Difficult [Deep Teacher Feature Maps"]]
    style M fill:#f9f,stroke:#333
    style N fill:#ccf,stroke:#333
```
</details>

Figure 1: Overview of LEAP. Rather than supervising the student against a fixed teacher block from the start, our curriculum advances the supervisory target through the teacher’s feature maps shallow-to-deep based on online CKA alignment, building student representations progressively.

To bridge this gap, existing literature has explored several strategies to "soften" the distillation target. A primary approach involves intermediate layer matching, where the student is provided with "hints" from the teacher’s intermediate layers rather than just the final representation [8]. For example, Patient Knowledge Distillation (PKD) introduced fixed-layer selection strategies—such as matching every k-th layer (PKD-Skip) or the final k layers (PKD-Last)—to ensure the student captures the hierarchical transformation of information [22] throughout the network. Other frameworks, such as ViTKD [29], use projection heads—often linear or MLP-based layers—to map the student’s intermediate lower-dimensional features into the teacher’s intermediate high-dimensional manifold to perform feature map generation and mimicking, attempting to resolve the physical dimension mismatch and apply hidden supervision. Relational distillation attempts to mitigate this by matching the similarity between patches rather than their raw values [15] [27]. Despite these advancements, most existing methods still rely on a static mapping schedule, where the student is forced to align with complex deep-layer features from the very beginning of training, or mapping the student and teacher’s intermediate features manually. This "all-at-once" approach fails to account for the student’s evolving capacity, and in the scenario where teacher and student have a large size mismatch the layer-mapping becomes arbitrary.

To mitigate this, we draw inspiration from curriculum learning: a training strategy that introduces concepts in an "easy-to-hard" progression [1]. The similarity analysis reveals that the shallower layers of ViT teacher produce feature maps with a higher similarity score with the final student feature map on earlier training epochs, and the similarity peak gradually shifts to the final teacher feature map as training proceeds. According to this finding, we hypothesize that similar feature maps are easier to learn for the student, thus narrowing the teacher-student gap. By treating the teacher’s shallower, more reconstructive layers as early and accessible targets, and gradually sweeping toward deeper, semantic layers, we guide the student through a structured learning path that accelerates convergence and improves final feature alignment. With this assumption, we propose LEAP: a training curriculum that gradually switches the training target across intermediate teacher features, with early stopping (controlled by a similarity threshold) pacing the curriculum With experiments on both ImageNet-100 and ImageNet-1K, our distillation curriculum achieves remarkable convergence speed-ups and substantial savings in training FLOPs and wall-time. Evaluations on semantic segmentation, image retrieval and image classification demonstrate that the distilled model retains the ability to adapt to a variety of downstream tasks. In summary, our contributions are as follows:

1. We propose LEAP: a layer-skipping curriculum for feature-based distillation for vision transformers, without the need for manual assignment or intermediate feature selection.

2. A thorough analysis on the robustness for the LEAP curriculum as well as the effect of using intermediate feature maps to supervise student model’s features with knowledge distillation. We furthermore verify and evaluate the distilled model on image, instance, and pixel level tasks.

## 2 Related Work

Vision Transformer Vision Transformers (ViTs) [4] have achieved state-of-the-art performance in a wide array of downstream applications. Due to their exceptional scalability with respect to dataset size and model capacity, ViTs have become the preferred backbone for modern Vision Foundation Models (VFMs), such as DINOv2 [14] and CLIP [17]. A key characteristic of the ViT architecture is in its structural hierarchy: shallower blocks tend to capture local spatial details, whereas deeper layers specialize in extracting dense semantic abstractions [29, 18]. When trained in a self-supervised manner, these models exhibit generalization capabilities that enable high performance in zero-shot or frozen-backbone scenarios, including image classification, semantic segmentation, and instance retrieval, without requiring extensive task-specific fine-tuning.

Knowledge Distillation While high-capacity ViT backbones offer state-of-the-art performance, their computational requirements necessitate distillation for resource-constrained environments. Knowledge Distillation (KD) [8] [5]bridges this gap by transferring knowledge from large-scale teachers to efficient students. Unlike logit-based methods [21] that focus on task-specific output distributions, we utilize feature-based KD [29] [28] [2], which minimizes the MSE between student and teacher feature maps via a linear projection layer. This paradigm is more robust and task-agnostic, making it ideal for distilling foundation models where final projection heads may be unavailable. By focusing on intermediate representations, our method ensures the distilled student retains a rich, general-purpose feature space capable of supporting diverse downstream tasks, from object recognition to dense prediction.

Teacher-Student Gap Due to the mismatch in capacity between the teacher and student models, a larger teacher model does not necessarily lead to a stronger distilled student, and, in some cases, can hamper the student model’s performance. This phenomenon is referred to as the teacherstudent gap [23]. Several prior works propose solutions to reduce the teacher’s ability to be more compatible with student’s capacity [9] , or to address the capacity mismatch by examining the gradient similarity [35]. Normalization for the logits has also been shown to be effective for the logit-based distillation [3]. Other works indicate that the usage of additional middle-sized teacher models [13] or intermediate teacher checkpoints during pretraining is also useful for reducing the gap and helpful for distillation into the student [10]. In summary, the approaches from prior works mainly involve processing the output from the teacher model in ways that are more compatible with the student’s capacity.

Curriculum Learning Curriculum Learning (CL) [1] improves training by presenting samples in increasing order of difficulty. While CL has recently been used to boost the efficiency of Vision Foundation Models [31, 12], its application to knowledge distillation remains limited. Existing CL-KD strategies, such as logit temperature scaling [10] or progressive layer matching [26], often struggle with feature-based distillation or rely on arbitrary block-wise assignments in situations when teacher and student architectures differ significantly in depth. To address these limitations, we design a curriculum that leverages the teacher’s full range of intermediate features. By ordering these features according to an adaptive similarity metric, our approach eliminates the need for manual layer alignment, facilitating a more natural and efficient transfer of knowledge across disparate architectures.

## 3 Recipe for Efficient ViTKD

## 3.1 Shifted Similarity

For our baseline, we adopt a standard feature-based KD paradigm that minimizes the $L _ { 2 }$ distance between the final feature maps of the student and teacher architectures. In this setup, the teacher’s final representation remains the static target throughout the entire training duration. Given that the

ViT-G teacher comprises of 40 blocks, whereas the ViT-S student has only 12, a fundamental question arises:

Is the final teacher feature map the most accessible target for the student to learn?

To investigate this, we perform a layer-wise similarity analysis between the student’s final feature map and all intermediate feature maps of the teacher during baseline training (see Figure 2). We utilize Centered Kernel Alignment (CKA) as our metric, calculating scores directly on the raw feature maps without intermediate linear projections.

The resulting similarity landscape reveals that the final teacher layer is not always the most similar to the student’s output, despite being the sole training target. Instead, we observe a distinct temporal shift: shallower teacher layers exhibit significantly higher similarity during the initial stages of training, with the peak of this similarity distribution gradually advancing toward the final teacher block as optimization progresses. Operating on the premise that feature similarity correlates with learning ease, these findings suggest that a static target is sub-optimal. Rather than focusing on a single layer’s representation, the student should adaptively navigate the full range of the teacher’s representational space, dynamically selecting the "easiest" target at each stage of the distillation process.

![](images/89faa64753f8da0b628212ae320c145c191c84026b6675c546a4d18ddad8c792.jpg)  
Figure 2: CKA heatmap between the student model’s last feature map and all of the teacher’s intermediate feature maps during training. Student checkpoints are saved every 5 epochs, and the CKA score is calculated across a subset of validation dataset.

## 3.2 The Layer-Skipping Curriculum

The observed shift in similarity patterns provides the core motivation for our proposed training curriculum. We redefine the complexity of the distillation task by mapping it to the teacher’s depth: shallower teacher feature maps serve as initial, more accessible learning targets, while deeper maps represent increasingly complex semantic abstractions. The training process begins with the student imitating the teacher’s first ViT block. Rather than following a fixed, manual schedule, we introduce an adaptive progression mechanism controlled by a CKA similarity threshold (τ ). Throughout the distillation process, we monitor the "online" CKA similarity between the student’s terminal feature map and the current teacher target. The curriculum advances to the next teacher block if and only if the similarity score reaches the predefined threshold τ . This similarity-driven transition ensures that the student has sufficiently learned the current level of abstraction before attempting to bridge the next gap in the representational hierarchy. This adaptive logic is formally detailed in Algorithm 1.

## 4 Experiments

## 4.1 Dataset and Training Setup

Dataset We evaluate the proposed framework on ImageNet-100 [25] and the large-scale ImageNet-1K [20] datasets. To facilitate rapid experiment iteration, we utilize ImageNet-100 for all ablation studies and parameter tuning. To assess the quality and transferability of the distilled representations, we evaluate performance across several downstream tasks: semantic segmentation is measured on the ADE20K dataset [32, 33], while instance retrieval is benchmarked on the revisited Oxford and Paris datasets [16]. Finally, we evaluate the model’s robustness to common visual corruptions using ImageNet-C [7].

Training Setup For all distillation experiments, we adopt the standard data augmentation pipeline from [11]. Optimization is performed using the LARS optimizer [30] with a global batch size of 256. The base learning rate is set to 9.0 for ImageNet-100 experiments and 6.0 for ImageNet-1K. Our teacher model utilizes a ViT-G backbone pre-trained with the DINOv2 objective and registers [14]. For the student models, we evaluate both ViT-Small (ViT-S) and ViT-Tiny (ViT-T), both of which are randomly initialized without registers. Training and evaluation are conducted across a heterogeneous compute cluster featuring NVIDIA L40S, Nvidia RTX A6000, NVIDIA L40, and GeForce RTX 2080 Ti GPUs. All experiments trained on ImageNet-100 cost 12-14 NVIDIA L40S hours, and all ImageNet-1K distillations take 300-400 NVIDIA L40S hours.

Algorithm 1 CKA-Triggered Progressive Distillation Curriculum  
Require: Teacher features $\{T_{1},\ldots,T_{M}\}$ , student S, threshold $\tau$ , patience $E_{max}$ 1: $m \leftarrow 1,\quad e \leftarrow 0$ {current target block, epochs on it}

2: for epoch = 1 to $N_{epochs}$ do

3:    Sample batch B

4:    score $\leftarrow$ CKA( $S_{last}(B)$ , $T_{m}(B)$ )

5:    if m < M and (score $\geq \tau$ or $e \geq E_{max}$ ) then

6: $m \leftarrow m + 1,\quad e \leftarrow 0$ {advance curriculum}

7:    else

8: $e \leftarrow e + 1$ 9:    end if

10: $L \leftarrow \text{MSE}(S_{last}(B), T_{m}(B))$ 11:    Update S via $\nabla L$ 12: end for

13: return S

## 4.2 The Layer-Skipping Curriculum Saves Training Time and FLOPs

We first validate the effectiveness of our curriculum on ImageNet-100. The distillation objective is defined as:

$$
\mathcal {L} _ {\text { distill }} = \operatorname{MSE} \left(P (S _ {\text { feat }}), T _ {\text { feat }}\right) + 0. 0 5 \cdot \operatorname{MSE} \left(P (S _ {\text { cls }}), T _ {\text { cls }}\right) \tag {1}
$$

Following [34], we utilize a single-layer linear projector P to align the student’s hidden dimensions with those of the teacher; this projector is shared between the patch tokens and the CLS token. We maintain a constant weight of 0.05 for the CLS token loss throughout all experiments to ensure a consistent comparison.

We evaluate convergence speed by performing linear probing on student checkpoints every five epochs. As illustrated in Figure 3 (left), LEAP distillation demonstrates significantly faster convergence than the baseline from the earliest stages of training. This acceleration suggests that shallower teacher feature maps effectively serve as "easier" learning targets, allowing the student to establish a stable representational foundation before tackling more complex objectives.

The progression of the curriculum, visualized in Figure 3 (right), reveals a distinct temporal pattern. The student transitions rapidly through initial teacher blocks which primarily encode local spatial information, while dwelling significantly longer in the deeper layers. This behavior suggests that deeper layers contain denser semantic knowledge that needs more steps to learn. This evidence validates our assumption of depth serving as a reliable proxy for target difficulty.

To evaluate the generalizability of our curriculum, we investigate its performance across different student capacities: ViT-Tiny (12 blocks, ∼6M parameters) and ViT-Small (12 blocoks, ∼22M parameters). As detailed in Table 1, LEAP consistently enhances linear probing accuracy for both architectures, with ViT-S reaching 90.10% and ViT-Tiny achieving 81.76%. Critically, these performance gains are accompanied by significant computational savings; the adaptive early-stopping for teacher inference achieves up to a 28.8% reduction in training FLOPs and a 22.5% reduction in total training time. Finally, evaluation on the mini-ImageNet-C benchmark confirms that the improvements in clean accuracy translate directly to out-of-distribution robustness, illustrating the high-quality representations learned through adaptive progression.

Table 1: Performance and efficiency comparison between baseline distillation and LEAP. Students (ViT-S, ViT-Tiny) are distilled from a ViT-G teacher on ImageNet-100 for 100 epochs. LEAP utilizes a CKA-threshold of 0.85. Efficiency metrics (FLOPs and Train Time) represent the reduction in teacher computational overhead during the distillation process.

<table><tr><td>Experiments</td><td>Lin. Probe ↑ (patch token)</td><td>Lin. Probe ↑ (CLS token)</td><td>mini-IN-C ↑ (accuracy)</td><td>FLOPs Saving</td><td>Train Time Saving</td></tr><tr><td>Baseline (ViT-S)</td><td>77.86%</td><td>77.12%</td><td>47.80%</td><td>—</td><td>—</td></tr><tr><td>Baseline (ViT-Tiny)</td><td>75.90%</td><td>76.04%</td><td>46.90%</td><td>—</td><td>—</td></tr><tr><td>LEAP (ViT-S)</td><td>90.10%</td><td>89.66%</td><td>66.69%</td><td>25.1%</td><td>21%</td></tr><tr><td>LEAP (ViT-Tiny)</td><td>81.76%</td><td>82.02%</td><td>51.74%</td><td>28.82%</td><td>22.5%</td></tr></table>

![](images/8c52d6d0c539be2901abd4a4dae60486b05f31df397b9b5552e03bdbc4dd52d7.jpg)

<details>
<summary>line chart</summary>

| Epochs | Baseline — Top-1 | Baseline — Top-5 | Curriculum — Top-1 | Curriculum — Top-5 |
| ------ | ---------------- | ---------------- | ------------------ | ------------------ |
| 0      | 20               | 45               | 38                 | 65                 |
| 10     | 35               | 60               | 58                 | 82                 |
| 20     | 45               | 70               | 75                 | 90                 |
| 30     | 50               | 75               | 78                 | 92                 |
| 40     | 55               | 80               | 80                 | 94                 |
| 50     | 60               | 82               | 82                 | 95                 |
| 60     | 65               | 85               | 85                 | 96                 |
| 70     | 70               | 88               | 88                 | 97                 |
| 80     | 75               | 90               | 90                 | 97                 |
| 90     | 78               | 91               | 91                 | 97                 |
| 100    | 78               | 91               | 91                 | 97                 |
</details>

![](images/1bbc598339a62afc00eeb49b33c69276fdd605f3862f60b0b780a2e88cb45068.jpg)

<details>
<summary>line chart</summary>

| Epochs | Block Index |
| ------ | ----------- |
| 0      | 3           |
| 10     | 15          |
| 20     | 24          |
| 30     | 27          |
| 40     | 30          |
| 50     | 33          |
| 60     | 36          |
| 70     | 38          |
| 80     | 39          |
| 90     | 39          |
</details>

Figure 3: Left: the linear probing accuracy convergence comparison for baseline and LEAP. Right: the curriculum progress visualization. The student checkpoint for linear probing evaluation is saved for every 5 epochs.

## 4.3 Evaluating Representational Quality via Downstream Tasks

While the accelerated convergence of the proposed curriculum is demonstrated through image-level linear probing, we further evaluate the distilled models on diverse downstream tasks to assess the generalizability of the learned representations. We conduct evaluations on semantic segmentation and instance retrieval to verify that the student model preserves both pixel-level details and fine-grained instance features. These experiments confirm that our curriculum-based distillation not only speeds up training but also produces a versatile backbone capable of adapting to a wide range of applications.

To evaluate instance-level representations, we perform image retrieval experiments on the Oxford and Paris datasets. For each image, the model backbone extracts a global embedding; we then compute the similarity scores between the query image embedding and the database embeddings. These results are ranked by similarity to calculate the mean Average Precision (mAP). As shown in Table 2, our LEAP-distilled ViT-S and ViT-Tiny outperform the baseline by a significant margin across all three difficulty levels. These results demonstrate that the proposed curriculum effectively preserves the fine-grained structural details required for precise instance matching.

We evaluate performance on ADE20K using three protocols: Linear Segmentation, Encoder-Only Mask Transformer (EOMT), and Multi-Scale (MS) inference. The results are summarized in Table 3.

In the Linear Segmentation setting, which measures the linear separability of frozen features, LEAP improves the ViT-S mIoU from 12.15% to 20.53%. When utilizing the EOMT head to evaluate how features interact within a Transformer-native decoder, the performance gap remains significant, with the LEAP-distilled ViT-S reaching 38.10% mIoU compared to the baseline’s 24.49%. Finally, under Multi-Scale (MS) evaluation (a protocol that tests robustness to scale variations) the LEAP-distilled ViT-S achieves 39.36% mIoU, representing a 14.74% absolute improvement. These consistent gains across different decoding architectures and inference scales suggest that navigating the teacher’s representational hierarchy during training allows the student to retain denser spatial and semantic information than standard distillation regimes.

Table 2: Instance recognition performance comparison (mAP) on Oxford and Paris datasets. The "Mean" column represents the average mAP across Easy, Medium, and Hard difficulty levels.

<table><tr><td rowspan="2">Training method</td><td colspan="4">roxford5k</td><td colspan="4">rparis6k</td></tr><tr><td>E</td><td>M</td><td>H</td><td>Mean</td><td>E</td><td>M</td><td>H</td><td>Mean</td></tr><tr><td>Baseline (ViT-S)</td><td>10.28</td><td>8.80</td><td>2.15</td><td>7.08</td><td>24.83</td><td>21.29</td><td>7.25</td><td>17.79</td></tr><tr><td>Baseline (ViT-Tiny)</td><td>8.04</td><td>7.72</td><td>1.84</td><td>5.87</td><td>21.35</td><td>19.42</td><td>6.59</td><td>15.79</td></tr><tr><td>LEAP (ViT-S)</td><td>22.56</td><td>17.30</td><td>4.81</td><td>14.89</td><td>53.84</td><td>40.99</td><td>15.97</td><td>36.93</td></tr><tr><td>LEAP (ViT-Tiny)</td><td>14.70</td><td>12.15</td><td>2.84</td><td>9.90</td><td>43.39</td><td>33.36</td><td>12.38</td><td>29.71</td></tr></table>

Table 3: Semantic segmentation performance comparison for distillation on ImageNet-100. Results are reported in mean Intersection over Union (mIoU) for linear segmentation, EOMT validation, and multi-scale (MS) evaluation.

<table><tr><td>Method</td><td>Linear Seg. ↑ (mIoU)</td><td>EOMT val ↑ (mIoU)</td><td>MS ↑ (mIoU)</td></tr><tr><td>Baseline (ViT-S)</td><td>12.15%</td><td>24.49%</td><td>24.62%</td></tr><tr><td>Baseline (ViT-Tiny)</td><td>9.51%</td><td>15.41%</td><td>14.88%</td></tr><tr><td>LEAP (ViT-S)</td><td>20.53%</td><td>38.10%</td><td>39.36%</td></tr><tr><td>LEAP (ViT-Tiny)</td><td>14.71%</td><td>21.67%</td><td>21.82%</td></tr></table>

## 4.4 Effectiveness on Large Dataset

Following the demonstrated effectiveness on ImageNet-100, we evaluate the scalability of the LEAP curriculum on ImageNet-1K.

The advantages of LEAP remain evident in tasks requiring high-fidelity features. As shown in Table 4, LEAP achieves remarkable performance across the Oxford and Paris retrieval benchmarks. Specifically, the LEAP-distilled ViT-S achieves a mean mAP improvement of 3.84% on the Oxford dataset and 7.75% on the Paris dataset. These results suggest that even at scale, LEAP effectively transfers the structural nuances which are critical for fine-grained instance matching.

In semantic segmentation (Table 5) and linear probing (Table 6), we observe that LEAP produces competitive results with the standard distillation baseline. In semantic segmentation, the ViT-S student maintains a marginal edge in EOMT (47.03% vs. 46.65%), while linear segmentation scores reach parity. Similarly, linear probing accuracies on ImageNet-1K show a narrowing gap, with ViT-S reaching 77.34% (vs. 77.63% baseline) and ViT-Tiny reaching 64.14% (vs. 64.4% baseline). We attribute this convergence to the architectural capacity ceiling and representation saturation; compact student models likely reach their limit for class separation and spatial alignment on this dataset. Furthermore, the CKA threshold (τ = 0.8) was primarily tuned on ImageNet-100; it is probable that the optimal threshold shifts as dataset complexity increases, while computational constraints precluded an exhaustive hyperparameter search on ImageNet-1K.

While achieving competitive performance on global classification and dense prediction tasks, LEAP provides a measured improvement in training efficiency, yielding a 11.51% reduction in teacher FLOPs and a 11.6% decrease in training time for the ViT-Tiny student. Though these computational savings are modest at the ImageNet-1K scale, they are achieved while enhancing the model’s feature integrity. Specifically, the substantial gains observed in the image retrieval benchmarks, where LEAP outperforms the baseline by up to 7.75%, serve as a strong proxy for overall representation quality. The results confirm that LEAP produces a feature space that better preserves the complex representation of the foundation model teacher.

Table 4: Results for ImageNet-1K. Performance comparison (mAP) on Oxford and Paris datasets. The "Mean" column represents the average mAP across Easy (E), Medium (M), and Hard (H) difficulty levels.

<table><tr><td rowspan="2">Training method</td><td colspan="4">roxford5k</td><td colspan="4">rparis6k</td></tr><tr><td>E</td><td>M</td><td>H</td><td>Mean</td><td>E</td><td>M</td><td>H</td><td>Mean</td></tr><tr><td>Baseline (ViT-S)</td><td>27.34</td><td>19.36</td><td>4.86</td><td>17.19</td><td>62.70</td><td>50.46</td><td>27.80</td><td>46.99</td></tr><tr><td>Baseline (ViT-Tiny)</td><td>19.41</td><td>14.16</td><td>2.42</td><td>11.9</td><td>50.18</td><td>38.47</td><td>14.98</td><td>34.54</td></tr><tr><td>LEAP (ViT-S)</td><td>34.50</td><td>24.14</td><td>4.46</td><td>21.03</td><td>71.52</td><td>57.88</td><td>34.81</td><td>54.74</td></tr><tr><td>LEAP (ViT-Tiny)</td><td>18.36</td><td>14.21</td><td>2.46</td><td>11.68</td><td>50.86</td><td>39.73</td><td>17.22</td><td>35.94</td></tr></table>

Table 5: Semantic segmentation performance comparison. Results are reported in mean Intersection over Union (mIoU) for linear segmentation, EOMT validation, and multi-scale (MS) evaluation.

<table><tr><td>Method</td><td>Linear Seg. ↑ (mIoU)</td><td>EOMT val ↑ (mIoU)</td><td>MS ↑ (mIoU)</td></tr><tr><td>Baseline (ViT-S)</td><td>29.85%</td><td>46.65%</td><td>48.22%</td></tr><tr><td>Baseline (ViT-Tiny)</td><td>18.92%</td><td>33.93%</td><td>33.42%</td></tr><tr><td>LEAP (ViT-S)</td><td>29.84%</td><td>47.03%</td><td>48.28%</td></tr><tr><td>LEAP (ViT-Tiny)</td><td>18.94%</td><td>32.80%</td><td>33.00%</td></tr></table>

Table 6: Results for training on ImageNet-1K. Comparison of Baseline and LEAP methods for ViT-S and ViT-Tiny with ViT-G teacher. for both ViT-Tiny and ViT-S the CKA threshold for curriculum is 0.8. Efficiency metrics (FLOPs and Train Time saving) represent the reduction in teacher computational overhead during the distillation process.

<table><tr><td>Experiments</td><td>Lin. Probe ↑ (patch token)</td><td>Lin. Probe ↑ (CLS token)</td><td>IN-C ↑ (patch token)</td><td>FLOPs Saving</td><td>Train Time Saving</td></tr><tr><td>Baseline (ViT-S)</td><td>76.87%</td><td>77.63%</td><td>55.93%</td><td>—</td><td>—</td></tr><tr><td>Baseline (ViT-Tiny)</td><td>63.35%</td><td>64.4%</td><td>36.4%</td><td>—</td><td>—</td></tr><tr><td>LEAP (ViT-S)</td><td>76.22%</td><td>77.34%</td><td>53.29%</td><td>8.5%</td><td>7.1%</td></tr><tr><td>LEAP (ViT-Tiny)</td><td>62.39%</td><td>64.14%</td><td>34.39%</td><td>11.51%</td><td>11.6%</td></tr></table>

## 4.5 The Necessity of Progressive Supervision vs. Single-Layer Targets

A potential critique of intermediate feature distillation is that the performance gains may stem from selecting a specific "optimal" layer rather than the curriculum itself. If a single intermediate layer contained the most transferable knowledge, static distillation from that "lucky" layer would outperform a progressive curriculum. To investigate this, we conducted an ablation study using a ViT-S teacher and a ViT-Tiny student. We trained twelve separate student models, each supervised exclusively by a different fixed intermediate teacher layer throughout training. This allows us to isolate the contribution of individual layers and determine whether LEAP’s effectiveness is derived from a specific static target or the structured transition between them.

As illustrated in Figure 4, while deeper teacher layers generally results in higher feature quality, LEAP outperforms all individual targets. These results suggest that no single frozen intermediate layer is sufficient to bridge the gap between teacher and student as effectively as a curriculum. Instead, the performance benefit is a product of the structured progression itself, which allows the student to learn a hierarchical foundation that any single-layer target lacks.

LEAP v.s. Single Intermediate Layer Supervision  
![](images/a3b3a127edbb063dfcb8a41ed1292cc73f87dda851f62ce95def2dd6ae3ac8ee.jpg)

<details>
<summary>bar chart</summary>

| Teacher Block Index | Linear Probing Accuracy (%) |
| :--- | :--- |
| 0 | 20.6 |
| 1 | 28.5 |
| 2 | 30.2 |
| 3 | 37.0 |
| 4 | 49.6 |
| 5 | 52.5 |
| 6 | 55.6 |
| 7 | 63.9 |
| 8 | 67.8 |
| 9 | 72.6 |
| 10 | 77.0 |
| 11 | 81.8 |
| LEAP | 83.4 |
</details>

Figure 4: Linear probing accuracy comparison between LEAP and single intermediate layer supervision. LEAP outperforms using any single layer as supervision throughout training, indicating the effectiveness of utilizing a structural curriculum.

## 4.6 Comparison with Dense One-to-One Layer Alignment

When teacher and student architectures share an identical depth, a common strategy for maximizing supervision is dense intermediate matching: aligning every student block with its corresponding teacher block, an approach often viewed as the the upper-bound for feature-based distillation.

In this section, we compare LEAP against this dense matching baseline. We conduct this experiment using a ViT-S teacher and a ViT-Tiny student; since both models consist of 12 layers, they allow for a direct, one-to-one feature map alignment. The results in table 7 reveal the efficiency of LEAP; with only 1 projector, LEAP is merely 0.02% behind the dense layer alignment upper bound, which uses 12 projectors in total. This comparison indicates that a structured, adaptive curriculum can achieve competitive performance of dense supervision while maintaining much higher structural simplicity.

Table 7: Comparison of distillation alignment strategies. One-to-one alignment refers to matching each teacher layer with corresponding student layer, while LEAP utilizes an adaptive progression curriculum. The baseline method aligns the last teacher and student feature map.

<table><tr><td>Alignment Strategy</td><td>Linear Probing ↑</td><td>Projectors (Params)</td></tr><tr><td>Baseline</td><td>81.80%</td><td> $\times 1$  (0.07M)</td></tr><tr><td>One-to-one Alignment</td><td>83.38%</td><td> $\times 12$  (0.89M)</td></tr><tr><td>LEAP</td><td>83.36%</td><td> $\times 1$  (0.07M)</td></tr></table>

## 5 Conclusion, Limitations, and Future Directions

In this work, we propose a layer-skipping curriculum for efficient feature-based knowledge distillation in Vision Transformers (ViTs). Through an automatic similarity-based curriculum, the student model achieves accelerated convergence and learns high-quality representations for diverse downstream tasks. One limitation of our work is the assumption of a white-box teacher model with accessible intermediate features. While many open-weight Vision Foundation Models (VFMs) exist, our method is restricted in scenarios where only the final teacher feature map is available. Furthermore, the CKA threshold utilized for our ImageNet-1K experiments may represent a sub-optimal hyperparameter. A more exhaustive optimization of this threshold could yield further performance improvements. A potential future direction is to expand this framework to cross-architecture scenarios where the teacher and student have different architectures (e.g., Transformer to CNN). Another key direction involves extending our method to distilling across data modalities for multimodal models, such as audio, language, and more.

## References

[1] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th International Conference on Machine Learning, 2009.  
[2] Defang Chen, Jian-Ping Mei, Yuan Zhang, Can Wang, Yan Feng, and Chun Chen. Cross-layer distillation with semantic calibration. In Proceedings of the AAAI Conference on Artificial Intelligence, 2021.  
[3] Jang Hyun Cho and Bharath Hariharan. On the efficacy of knowledge distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019.  
[4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.  
[5] Jianping Gou, Baosheng Yu, Stephen J. Maybank, and Dacheng Tao. Knowledge distillation: A survey. International Journal of Computer Vision, 2021.  
[6] Jia Guo, Minghao Chen, Yao Hu, Chen Zhu, Xiaofei He, and Deng Cai. Reducing the teacherstudent gap via spherical knowledge distillation. In arXiv preprint arXiv:2010.07485, 2020.  
[7] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In International Conference on Learning Representations, 2019.  
[8] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.  
[9] Xiao Jin, Baoyun Peng, Yichao Wu, Yu Liu, Jiaheng Liu, Ding Liang, Junjie Yan, and Xiaolin Hu. Knowledge distillation via route constrained optimization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019.  
[10] Zheng Li, Xiang Li, Lingfeng Yang, Borui Zhao, Renjie Song, Lei Luo, Jun Li, and Jian Yang. Curriculum temperature for knowledge distillation. arXiv preprint arXiv:2211.16231, 2022.  
[11] Lightly Team. Lightlytrain, 2025. URL https://github.com/lightly-ai/ lightly-train.  
[12] Wenquan Lu, Jiaqi Zhang, Hugues Van Assel, and Randall Balestriero. Ditch the denoiser: Emergence of noise robustness in self-supervised learning from data curriculum. In The Thirtyninth Annual Conference on Neural Information Processing Systems, 2025. URL https: //openreview.net/forum?id=Pa5pKAeAO7.  
[13] Seyed Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, and Hassan Ghasemzadeh. Improved knowledge distillation via teacher assistant. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020.  
[14] Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023.  
[15] Wonpyo Park, Dongju Kim, Yan Lu, and Minsu Cho. Relational knowledge distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.  
[16] Filip Radenovic, Ahmet Iscen, Giorgos Tolias, Yannis Avrithis, and Ond ´ ˇrej Chum. Revisiting Oxford and Paris: Large-scale image retrieval benchmarking. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018.  
[17] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. 2021.  
[18] Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision transformers see like convolutional neural networks? arXiv:2108.08810, 2021.  
[19] Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.  
[20] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. arXiv preprint arXiv:1409.0575, 2015.  
[21] Shangquan Sun, Wenqi Ren, Jingzhi Li, Rui Wang, and Xiaochun Cao. Logit standardization in knowledge distillation. arXiv preprint arXiv:2403.01427, 2024.  
[22] Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. Patient knowledge distillation for BERT model compression. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 2019.  
[23] Huiyuan Tian, Bonan Xu, and Shijian Li. Distillation dynamics: Towards understanding featurebased distillation in vision transformers. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026.  
[24] Hugo Touvron, Matthieu Cord, Douze Matthijs, Francisco Massa, Alexandre Sablayrolles, and Herve Jegou. Training data-efficient image transformers & distillation through attention. In International Conference on Machine Learning, 2021.  
[25] Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra. Matching networks for one shot learning. In Advances in Neural Information Processing Systems, 2016.  
[26] Hui Wang, Hanbin Zhao, Xi Li, and Xu Tan. Progressive blockwise knowledge distillation for neural network acceleration. In Proceedings of the 27th International Joint Conference on Artificial Intelligence, 2018.  
[27] Zhaoyi Yan, Kangjun Liu, and Qixiang Ye. Delving deep into semantic relation distillation. arXiv preprint arXiv:2503.21269, 2025.  
[28] Chuanguang Yang, Xinqiang Yu, Zhulin An, and Yongjun Xu. Categories of response-based, feature-based, and relation-based knowledge distillation. arXiv preprint arXiv:2306.10687, 2023.  
[29] Zhendong Yang, Zhe Li, Ailing Zeng, Zexian Li, Chun Yuan, and Yu Li. ViTKD: Feature-based knowledge distillation for vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshop, 2022.  
[30] Yang You, Igor Gitman, and Boris Ginsburg. Large batch training of convolutional networks. arXiv preprint arXiv:1708.03888, 2017.  
[31] Jiaqi Zhang, Juntuo Wang, Zhixin Sun, John Zou, and Randall Balestriero. FastDINOv2: Frequency based curriculum learning improves robustness and training speed. arXiv preprint arXiv:2507.03779, 2025.  
[32] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ADE20K dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.  
[33] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ADE20K dataset. International Journal of Computer Vision, 127(3):302–321, 2019.  
[34] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. iBOT: Image BERT pre-training with online tokenizer. In International Conference on Learning Representations, 2022.  
[35] Yizhen Zhu and Yi Wang. Student customized knowledge distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

## A Technical appendices and supplementary material

## A.1 Linear Probing with Standard Deviation

In this section, we investigate whether LEAP can perform consistently. ViT-G teacher is used to distill ViT-S student on ImageNet-100, and we selected 0.85 as the CKA threshold for LEAP. 3 distinct seeds are selected for baseline distillation while 5 seeds are used for LEAP distillation. As in figure 5, LEAP shows consistent performance across multiple seeds with small standard deviation, indicating the effectiveness of this approach.

![](images/982dcfe85697e0a3572120e084f5e6ca266e603c41e2a3992921488054375e60.jpg)

<details>
<summary>scatter plot</summary>

| Methods    | Baseline | LEAP   |
| ---------- | -------- | ------ |
| Baseline   | 77.85 ± 0.67 | 89.83 ± 0.36 |
| LEAP       | -        | -      |
</details>

Figure 5: Linear probing accuracy comparison between LEAP and baseline with multiple seeds. LEAP consistently outperforms baseline on this ImageNet-100 distillation regardless of the seeds.

## A.2 Curriculum Robustness to CKA Threshold Selection

![](images/996fa5a932373470b665051b72eeb2d73c76e67663d609457bec73506e4dbe4f.jpg)

<details>
<summary>line chart</summary>

| CKA Threshold | Linear Probing Accuracy(%) |
| ------------- | -------------------------- |
| 0.78          | 82.4                       |
| 0.79          | 82.9                       |
| 0.80          | 83.1                       |
| 0.81          | 82.8                       |
| 0.82          | 83.4                       |
| 0.83          | 83.0                       |
| 0.85          | 82.7                       |
</details>

Figure 6: LEAP performance comparison for multiple CKA thresholds. While LEAP is robust to the threshold selections, 0.82 is the optimal threshold for this setting.

We conduct a small scale cka threshold search for the distillation from ViT-S teacher distilled from DINOv2 and ViT-Tiny student initialized from scratch. As in figure 6, LEAP is generally robust to multiple threshold selection, and the optimal threshold is around 0.82.