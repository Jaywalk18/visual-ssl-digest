# Curvature-Guided Mixing for MLLM Adaptation

Jinglong Yang<sup>1,2†</sup> , Jiaxuan He<sup>1†</sup> , Wenjian Huang<sup>1</sup> , Zhan Zhuang<sup>2</sup> , and Jianguo Zhang<sup>1\*</sup>

<sup>1</sup> Research Institute of Trustworthy Autonomous Systems and Department of Computer Science and Engineering, Southern University of Science and Technology <sup>2</sup> Department of Computer Science, City University of Hong Kong

Abstract. Fine-tuning Multimodal Large Language Models (MLLMs) on specialized tasks often leads to catastrophic forgetting of their general capabilities. Existing model merging methods to combat this are often heuristic or use sub-optimal objectives. We propose Curvature-Guided Mixing (CGM), a theoretically grounded framework that merges pre-trained and fine-tuned models. CGM formulates a joint optimization objective and uses a second-order (Hessian) approximation of the loss landscapes to analytically derive an optimal, closed-form “soft mixing” ratio. This ratio intelligently blends parameters based on their relative task-specific curvatures. We also introduce CGM<sup>†</sup>, a robust “hard mixing” variant that performs sparse parameter selection guided by a novel, curvature-aware score. Experiments on LLaVA-1.5 and Qwen-2.5VL across multiple downstream tasks show that CGM and CGM<sup>†</sup> consistently improve the trade-of between task specialization and general knowledge retention over existing methods. Code is available at github.com/zzsyjl/CGM-ECCV-2026.

Keywords: Continual learning · MLLM adaptation · model merging

## 1 Introduction

Multimodal Large Language Models (MLLMs) have emerged as powerful foundation models capable of joint understanding, generation, and planning across vision and language modalities [1, 4, 18, 21, 47]. Their impressive performance stems from pre-training on web-scale datasets, which endows them with a broad, general-purpose knowledge base. While this enables strong zero-shot performance on many tasks, adapting these models to specialized or novel domains requires targeted fine-tuning. This adaptation process, however, often leads to a critical issue: the model’s newly acquired skills come at the expense of losing its foundational, pre-trained abilities—a phenomenon known as catastrophic forgetting [15,25,44]. The central challenge, therefore, is to develop a methodology that can efectively instill new, task-specific knowledge into an MLLM while safeguarding the vast and robust general intelligence it already possesses.

![](images/cfddd77ddc793b7a22aa32c5c5b8e4fa1e9956ddeb956fbb941a9147da4722e5.jpg)

![](images/2e61167ff6ba127d7734d34555094e1bdf22c36bd80389980a8c6cc98d45b196.jpg)  
Fig. 1: Performance comparison of our methods (CGM and CGM<sup>†</sup>) against baselines for LLaVA fine-tuned on OKVQA. We evaluate general knowledge retention (Pre-Avg: average performance of pre-training tasks), specialization on the new task (Target), and the harmonic mean of both (Hscore) to measure the overall balance.

Several recent works have explored this challenge through model merging [13, 46] and parameter-eficient fine-tuning [20,24,35,37]. However, these approaches remain limited in efectively balancing task adaptation and knowledge retention. For example, Spider [13] merges parameters using heuristic scoring functions without a solid theoretical foundation, while Model Tailor [46] employs second-order information [7] but optimizes only for downstream performance. As a result, these methods tend to overfit to the new task and neglect the preservation of pre-trained general knowledge, underscoring the need for a principled framework that jointly optimizes both objectives.

To overcome these limitations, we introduce Curvature-Guided Mixing (CGM) a theoretically grounded framework for merging pre-trained and fine-tuned models. Our approach begins by formulating a clear objective: to find a new set of weights that simultaneously minimizes the loss with respect to both the finetuning and general pre-training tasks. We model the loss landscapes around the fine-tuned and pre-trained optima using a second-order Taylor approximation, which captures the local geometry, or “curvature”, of each loss surface via its respective Hessian matrix.

To illustrate this, consider Figure 2, which visualizes two hypothetical loss landscapes corresponding to the fine-tuning and pre-training tasks. A naive interpolation between their optima would likely fall into a region of high loss for both. Guided by local curvature, our CGM method navigates this anisotropic landscape by identifying that the first parameter direction, critical for the finetuning task (due to its high curvature), should be adopted from the fine-tuned model, while the second direction, critical for the pre-training task, should be retained from the pre-trained model. This enables CGM to efectively integrate the “skills” learned during fine-tuning with the foundational knowledge from pretraining, achieving low loss on both tasks and mitigating catastrophic forgetting.

![](images/fe473eff43e8635303146c2f39fd608e2aa82f0798999f4f67b4453b47fd7b7c.jpg)  
Fig. 2: A conceptual illustration of the motivation behind CGM. The fine-tuning loss (orange) and pre-training loss (blue) landscapes exhibit conflicting anisotropic curvatures: the fine-tuning loss is sharp along one axis, whereas the pretraining loss is sharp along the other. CGM leverages this second-order geometry to locate a balanced joint minimum, as detailed in Section 3.

By mathematically formulating and solving this joint optimization problem (as detailed in Section 3), we derive a closed-form per-parameter mixing ratio. This “soft mixing” rule provides a non-heuristic and theoretically grounded solution. It elegantly demonstrates that the optimal mixing ratio for each parameter is directly determined by its relative curvature in the two loss landscapes.

While soft mixing provides a closed-form solution for the approximated loss, it is fundamentally a dense interpolation that updates every parameter. This approach carries a significant risk: a dense update can be disruptive, potentially destroying existing knowledge. We hypothesize that a sparse update strategy is more robust. The intuition is to preserve the vast majority of the model’s parameters and only modify the most critical subset. This approach, which is analogous to imposing an L<sub>0</sub> constraint on the update, ensures a minimal, targeted modification.

To implement this, we propose a “hard mixing” strategy, termed CGM<sup>†</sup>. This method reframes the problem as a sparse selection task. Our method starts from the fine-tuned model—which has acquired the new skill but sufered from forgetting—and treats the pre-trained parameters as a “knowledge reservoir” to be sparsely re-integrated. For each parameter, a discrete decision is made on whether to revert to the pre-trained value. We derive a simple, curvature-aware score to rank parameters in the learnable layers. By selecting only the top-K% of parameters (where K denotes the sparsity ratio) most critical for general knowledge and least critical for the new task, CGM<sup>†</sup> performs a sparse, targeted reversion that efectively preserves foundational knowledge while maintaining the newly acquired skill and avoiding the pitfalls of dense interpolation.

In summary, our main contributions are as follows:

![](images/1be38a0205b708dfbe39d8ce6bfe6432c8e73475297d708e28a21830a581bc0a.jpg)  
Fig. 3: Radar plots illustrating the performance trade-of between downstream adaptation and general knowledge retention. The “Target Task” axis shows performance on the fine-tuned task, while all other axes measure general pre-trained capabilities. Our methods, CGM and CGM<sup>†</sup> demonstrate superior balance by achieving high targettask performance while simultaneously preserving pre-trained knowledge.

– We propose Curvature-Guided Mixing (CGM), a novel and theoretically grounded method that optimally merges fine-tuned and pre-trained models by minimizing a joint, curvature-aware loss objective.

– We introduce CGM<sup>†</sup>, a variant that reformulates the merging task as a sparse parameter selection problem, providing an eficient and efective mechanism for re-injecting pre-trained knowledge to mitigate forgetting.

– We conduct extensive experiments across diverse datasets and MLLM backbones, demonstrating that our proposed methods substantially outperform prior approaches in balancing downstream task performance and the preservation of general capabilities.

## 2 Related Work

## 2.1 Catastrophic Forgetting in MLLMs

Multimodal Large Language Models (MLLMs) [1, 18, 21, 47] have achieved remarkable success by leveraging large-scale pre-training to acquire broad, generalpurpose knowledge. To adapt these models to specialized downstream tasks, fine-tuning is a necessary step [12, 13]. However, this adaptation process often leads to catastrophic forgetting [25, 44], a well-documented phenomenon where performance on its original general tasks degrades significantly after learning a new task.

Recent studies have begun to investigate this problem specifically within the MLLM context. Some work focuses on quantifying this efect in continual instruction tuning settings [3,5,33], while others observe that anti-forgetting techniques developed for LLMs show limited efectiveness when directly applied to MLLMs [46]. Our work tackles this core challenge: achieving single-step adaptation to new tasks while rigorously preserving the broad general knowledge obtained during pre-training.

## 2.2 Strategies for Knowledge Preservation

Existing methods to mitigate catastrophic forgetting for MLLM can be broadly categorized into three families:

Regularization-based Methods. This classic continual learning approach introduces auxiliary loss terms to penalize significant changes to parameters deemed important for old tasks [16, 43]. However, these methods often require modifications to the fine-tuning loss objective and can be complex to balance with the primary task loss.

Parameter-Eficient Fine-Tuning (PEFT). This popular family of methods freezes the vast majority of the pre-trained model and introduces a small set of new, trainable parameters for each task. These can be additive modules like Adapter [10] and LoRA [11]. These approaches are efective at isolating task-specific knowledge and reducing interference (especially in multi-step adaptation [36,39,40,42]), and related parameter-eficient editing strategies have also been explored for difusion models [6].

Partial-based Updating and Model Merging. This third category aims to create a merged model by selectively updating a subset of the pre-trained weights. The core challenge lies in determining which parameters to update and how to merge them. [13] assesses parameter importance using a heuristic combination of zeroth-order (magnitude) and first-order (gradient) information. Its reliance on a hand-crafted scoring function lacks a rigorous theoretical justification for why their specific formulation is optimal. Model Tailor [46] leverages the Hessian matrix to identify a sparse “model patch.” Inspired by model pruning techniques like SparseGPT [7], its methodology selects parameters deemed critical for the downstream task. However, this method is designed to minimize loss solely on the fine-tuning task, lacking an explicit objective to balance the preservation of pre-trained knowledge.

In contrast to this prior art, our CGM framework provides a theoreticallygrounded solution derived directly from a clear, joint optimization objective. This allows us to derive a non-heuristic, closed-form “soft mixing” rule and a “hard mixing” score that elegantly uses relative curvature to determine the optimal merge.

Relation to Fisher Merging. We also relate our soft mixing to Fisher Merging [28]. Soft mixing resembles Fisher Merging when the curvature is computed using the Fisher Information Matrix (FIM). However, the derivations difer fundamentally: Fisher Merging follows from a Laplace approximation of the posterior, whereas CGM minimizes a joint weighted loss objective. It is theoretically intriguing that distinct assumptions converge to a similar expression. Crucially, CGM is a more general framework: it is not limited to the FIM for curvature estimation, and our further exploration shows, approximating the true Hessian diagonal via Hutchinson trace estimation and Hessian-vector products yields stronger performance.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Curvature-Guided Mixing (CGM) Pipeline

Require: Pre-trained weights  $w^{pt}$ , fine-tuning data  $D_{ft}$ , calibration set C (sampling from pre-trained tasks)

Output: Merged weights  $w^{*}$ 

1: Pre-training analysis: estimate diagonal  $H^{pt}$  on  $w^{pt}$  with C

2: Fine-tuning: SFT on  $D_{ft}$  to obtain  $w^{ft}$ , estimate diagonal  $H^{ft}$  on  $w^{ft}$  with  $D_{ft}$ 

3: Mixing (choose one):

4: Soft-Mixing (CGM): compute  $\lambda_{i}$  from Eq. 8, set  $w_{i}^{*}=w_{i}^{ft}+\lambda_{i}(w_{i}^{pt}-w_{i}^{ft})$ 

5: Hard-Mixing (CGM $^{\dagger}$ ): score each parameter with  $c_{i}$  (Eq. 12); revert the lowest-score K% of parameters to  $w_{i}^{pt}$ , keep the rest as  $w_{i}^{ft}$
</div>

## 3 Methodology

We consider the problem of merging a pre-trained model ${ \pmb w } ^ { \mathrm { p t } }$ and its fine-tuned variant $\pmb { w } ^ { \mathrm { f t } }$ into a unified model $\ b { w } ^ { * }$ that performs well on the target task while preserving general capabilities. In this section, we first formulate this as a joint optimization problem balancing the two loss landscapes, then introduce our method along with its variant.

## 3.1 Loss Landscape Approximation

We begin by modeling the local geometry of the loss landscapes around the two optimal weight configurations: $\pmb { w } ^ { \mathrm { p t } }$ for general tasks and ${ \pmb w } ^ { \mathrm { f t } }$ for the specific fine-tuning task.

The loss functions in the neighborhoods of these optima are approximated using a second-order Taylor expansion. Let $\ell _ { \mathrm { p t } } ( \pmb { w } )$ and $\ell _ { \mathrm { f t } } ( w )$ denote the loss functions for the general and fine-tuning tasks, respectively. Their local approximations are given by:

$$
\begin{array}{r l} & {\ell_ {\mathrm{pt}} (\boldsymbol {w}) = \ell_ {\mathrm{pt}} (\boldsymbol {w} ^ {\mathrm{pt}}) + \nabla \ell_ {\mathrm{pt}} (\boldsymbol {w} ^ {\mathrm{pt}}) ^ {\top} (\boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{pt}})} \\ & {\qquad \frac {1}{2} (\boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{pt}}) ^ {\top} \mathbf {H} ^ {\mathrm{pt}} (\boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{pt}}) + \mathcal {O} (\| \boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{pt}} \| ^ {3}),} \end{array}\tag{1}
$$

$$
\begin{array}{c} \ell_ {\mathrm{ft}} (\boldsymbol {w}) = \ell_ {\mathrm{ft}} (\boldsymbol {w} ^ {\mathrm{ft}}) + \nabla \ell_ {\mathrm{ft}} (\boldsymbol {w} ^ {\mathrm{ft}}) ^ {\top} (\boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{ft}}) \\ \frac {1}{2} (\boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{ft}}) ^ {\top} \mathbf {H} ^ {\mathrm{ft}} (\boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{ft}}) + \mathcal {O} (\| \boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{ft}} \| ^ {3}), \end{array}\tag{2}
$$

where ∇ℓ is the first-order gradient vector, H is the Hessian matrix, and $\mathcal { O } ( \cdot )$ represents the higher-order terms.

Our framework relies on two simplifying assumptions.

First, regarding the first-order terms: We assume that ${ \pmb w } ^ { \mathrm { p t } }$ and ${ \pmb w } ^ { \mathrm { f t } }$ are local minima obtained from training on their respective tasks. By definition, the gradient at any converged local minimum ${ \pmb w } _ { \mathrm { o p t } }$ is zero. Therefore, $\nabla \ell _ { \mathrm { p t } } ( \pmb { w } ^ { \mathrm { p t } } ) = \pmb { 0 }$ and $\nabla \ell _ { \mathrm { f t } } ( { \boldsymbol { \mathbf { \mathit { w } } } } ^ { \mathrm { f t } } ) = \mathbf { \mathbf { 0 } }$ , allowing the first-order terms to be safely omitted in both Taylor expansions.

Second, regarding the higher-order terms: our objective is to find a merged model $\ b { w } ^ { * }$ that balances performance on both the pre-training and fine-tuning tasks. Such a solution must lie within the joint low-loss region—that ${ \mathrm { i s } } ,$ within the neighborhoods of ${ \pmb w } ^ { \mathrm { p t } }$ and ${ \pmb w } ^ { \mathrm { f t } }$ . Therefore, a solution $\pmb { w } ^ { * }$ within this region satisfies the locality condition of the Taylor expansion, allowing us to omit the higher-order terms $\mathcal { O } ( \cdot )$

Based on these two assumptions and ignoring the constant terms, our objective simplifies to minimizing the sum of two quadratic penalties:

$$
\ell_ {\mathrm{pt}} (\boldsymbol {w}) \propto \frac {1}{2} (\boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{pt}}) ^ {\top} \mathbf {H} ^ {\mathrm{pt}} (\boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{pt}}),\tag{3}
$$

$$
\ell_ {\mathrm{ft}} (\boldsymbol {w}) \propto \frac {1}{2} \left(\boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{ft}}\right) ^ {\top} \mathbf {H} ^ {\mathrm{ft}} \left(\boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{ft}}\right),\tag{4}
$$

where $\mathbf { H } ^ { \mathrm { p t } }$ and $\mathbf { H } ^ { \mathrm { f t } }$ denote the Hessian matrices capturing the local curvature of the respective loss surfaces. A larger Hessian eigenvalue corresponds to a sharper curvature, indicating that the loss is more sensitive to perturbations in that parameter direction.

To make the problem computationally tractable, we keep only the diagonal entries (denoted as $h _ { i } ^ { \mathrm { p t } }$ and $h _ { i } ^ { \mathrm { f t } } )$ of each Hessian and drop all of-diagonal terms. This simplification, common in curvature-aware optimization and second-order approximation methods [16, 17, 27], reduces the computational and storage cost from $\mathcal { O } ( d ^ { 2 } )$ to $\mathcal O ( d )$ and allows per-parameter decoupling of the objective.

## 3.2 CGM: Soft Mixing with Curvature Guidance

With the quadratic approximations of the loss functions, we define a joint objective to find a new set of weights w that simultaneously minimizes the increase in loss for both tasks:

$$
\underset {\boldsymbol {w}} {\operatorname{argmin}} \ell (\boldsymbol {w}) = \frac {1}{2} (\boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{ft}}) ^ {\top} \mathbf {H} ^ {\mathrm{ft}} (\boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{ft}}) + \frac {\alpha}{2} (\boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{pt}}) ^ {\top} \mathbf {H} ^ {\mathrm{pt}} (\boldsymbol {w} - \boldsymbol {w} ^ {\mathrm{pt}}).\tag{5}
$$

Here, $\alpha > 0$ is a hyperparameter that balances preserving general knowledge (captured by $\mathbf { H } ^ { \mathrm { p t } } )$ and acquiring new task-specific skills (captured by $\mathbf { H } ^ { \mathrm { f t } } )$ . This objective seeks a point in the joint low-low region, weighted by their curvature.

We parameterize the merged weights w as a per-parameter linear interpolation starting from the fine-tuned weights. Let $\pmb { \Delta } = \pmb { w } ^ { \mathrm { p t } } - \pmb { w } ^ { \mathrm { f t } }$ be the “reversion vector" pointing back to the pre-trained weights. For each parameter $i ,$ the new weight is $w _ { i } = w _ { i } ^ { \mathrm { f t } } + \lambda _ { i } \varDelta _ { i }$ , where $\lambda _ { i } \in [ 0 , 1 ]$ is the mixing ratio.

Under the diagonal Hessian approximation, the joint objective in Eq. 5 decouples into independent per-parameter optimization problems, each expressed as mi $\mathsf { 1 } _ { \lambda _ { i } } \ell _ { i } ( \lambda _ { i } )$ :

$$
\min _ {\lambda_ {i}} \ell_ {i} (\lambda_ {i}) = \frac {1}{2} h _ {i} ^ {\mathrm{ft}} ((w _ {i} ^ {\mathrm{ft}} + \lambda_ {i} \varDelta_ {i}) - w _ {i} ^ {\mathrm{ft}}) ^ {2} + \frac {\alpha}{2} h _ {i} ^ {\mathrm{pt}} ((w _ {i} ^ {\mathrm{ft}} + \lambda_ {i} \varDelta_ {i}) - w _ {i} ^ {\mathrm{pt}}) ^ {2}.\tag{6}
$$

Substituting $w _ { i } ^ { \mathrm { p t } } = w _ { i } ^ { \mathrm { f t } } + \varDelta _ { i }$ , this simplifies to:

$$
\ell_ {i} (\lambda_ {i}) = \frac {1}{2} \varDelta_ {i} ^ {2} \big [ h _ {i} ^ {\mathrm{ft}} \lambda_ {i} ^ {2} + \alpha h _ {i} ^ {\mathrm{pt}} (\lambda_ {i} - 1) ^ {2} \big ].\tag{7}
$$

This yields a simple quadratic function of $\lambda _ { i }$ . By setting the derivative $\partial \ell _ { i } / \partial \lambda _ { i }$ to zero, we obtain the closed-form optimal mixing ratio for each parameter:

$$
\lambda_ {i} ^ {*} = \frac {\alpha h _ {i} ^ {\mathrm{pt}}}{h _ {i} ^ {\mathrm{ft}} + \alpha h _ {i} ^ {\mathrm{pt}}}.\tag{8}
$$

We refer to this result as Curvature-Guided Mixing (CGM), a theoretically grounded $\mathrm { \Omega ^ { 6 6 } s o f t }$ mixi $\mathrm { | g \rangle }$ rule. It states that parameters with higher pre-training curvature $( h _ { i } ^ { \mathrm { p t } } )$ ) lead to a larger contribution from the pre-trained model, while those with higher fine-tuning curvature $\left( { h _ { i } ^ { \mathrm { f t } } } \right)$ retain more from the fine-tuned model.

We are intrigued to observe that if curvature is estimated using the empirical Fisher Information Matrix, the resulting soft-mixing expression closely resembles Fisher Merging [28]. However, the derivations difer fundamentally: Fisher Merging stems from a Laplace approximation of the posterior, whereas CGM minimizes a joint weighted loss objective. It is theoretically interesting that distinct assumptions converge to a similar form. Crucially, CGM is a more general framework and is not tied to any specific Hessian estimation method, and Fisher Merging can be viewed as a degeneralized case of CGM when the Hessian is approximated by the FIM. Our further exploration shows that approximating the true Hessian diagonal via Hutchinson trace estimation and Hessian-vector products yields stronger performance.

## 3.3 CGM<sup>†</sup>: Hard Mixing via Sparse Reversion

While soft mixing provides a closed-form solution to the approximated objective, it is a dense interpolation, which is often unnecessarily disruptive, as it modifies all parameters, potentially destroying existing knowledge. We argue that a more robust approach is to perform a sparse update: preserve the majority of the model’s parameters and only update the most critical subset. To achieve this, we propose a more robust “hard mixing” strategy, CGM<sup>†</sup>, which reframes the problem as a sparse parameter selection task. Instead of blending weights, we make a binary choice for each parameter: either retain the fine-tuned weight $w _ { i } ^ { \mathrm { f t } }$ or revert to the pre-trained one $w _ { i } ^ { \mathrm { p t } }$

We introduce a binary mask $m \in \{ 0 , 1 \} ^ { d }$ , where d is the number of parameters, and define the merged model as ${ \pmb w } ( { \pmb m } ) = { \pmb w } ^ { \mathrm { f t } } + { \pmb m } \odot { \pmb \Delta }$ , where $\pmb { \Delta } = \pmb { w } ^ { \mathrm { p t } } - \pmb { w } ^ { \mathrm { f t } }$ Here, $m _ { i } = 1$ signifies reverting to the pre-trained weight, and $m _ { i } = 0$ signifies keeping the fine-tuned weight. Substituting this into our joint objective $\left( \mathrm { E q . ~ 5 } \right)$ yields:

$$
\ell (\boldsymbol {m}) = \frac {1}{2} \sum_ {i} h _ {i} ^ {\mathrm{ft}} ((w _ {i} ^ {\mathrm{ft}} + m _ {i} \varDelta_ {i}) - w _ {i} ^ {\mathrm{ft}}) ^ {2} + \frac {\alpha}{2} \sum_ {i} h _ {i} ^ {\mathrm{pt}} ((w _ {i} ^ {\mathrm{ft}} + m _ {i} \varDelta_ {i}) - w _ {i} ^ {\mathrm{pt}}) ^ {2}.\tag{9}
$$

$$
\ell (\pmb {m}) = \frac {1}{2} \sum_ {i} h _ {i} ^ {\mathrm{ft}} \varDelta_ {i} ^ {2} m _ {i} ^ {2} + \frac {\alpha}{2} \sum_ {i} h _ {i} ^ {\mathrm{pt}} \varDelta_ {i} ^ {2} (m _ {i} - 1) ^ {2}.\tag{10}
$$

Since $m _ { i } \in \{ 0 , 1 \}$ , we have $m _ { i } ^ { 2 } = m _ { i }$ and $( m _ { i } - 1 ) ^ { 2 } = 1 - m _ { i }$ . The objective simplifies to:

$$
\ell (\pmb {m}) = \frac {1}{2} \sum_ {i} h _ {i} ^ {\mathrm{ft}} \varDelta_ {i} ^ {2} m _ {i} + \frac {\alpha}{2} \sum_ {i} h _ {i} ^ {\mathrm{pt}} \varDelta_ {i} ^ {2} (1 - m _ {i}).\tag{11}
$$

Rearranging the terms, we get:

$$
\ell (\boldsymbol {m}) = \underbrace {\frac {\alpha}{2} \sum_ {i} h _ {i} ^ {\mathrm{pt}} \Delta_ {i} ^ {2}} _ {\text {Constant w.r.t.} \boldsymbol {m}} + \sum_ {i} \underbrace {\frac {1}{2} (h _ {i} ^ {\mathrm{ft}} - \alpha h _ {i} ^ {\mathrm{pt}}) \Delta_ {i} ^ {2}} _ {c _ {i}} m _ {i}.\tag{12}
$$

Minimizing $\ell ( m )$ is now equivalent to minimizing $\sum _ { i } c _ { i } m _ { i }$ , where $\begin{array} { r } { c _ { i } = \frac { 1 } { 2 } ( h _ { i } ^ { \mathrm { f t } } - } \end{array}$ $\alpha h _ { i } ^ { \mathrm { p t } } ) \varDelta _ { i } ^ { 2 }$ serves as a ranking score for updating the i-th parameter. A parameter with a small $c _ { i } \ \mathrm { ( i . e . }$ , low $h _ { i } ^ { \mathrm { f t } }$ and high $h _ { i } ^ { \mathrm { p t } } )$ is one that is unimportant for the new task but crucial for the general-purpose pre-trained task, making it a prime candidate for reversion.

To enforce sparsity, we constrain the number of updated parameters to a budget K defined as a sparsity ratio (percentage of parameters updated). Let d be the total number of parameters and $K \in ( 0 , 1 ] ;$ then $\| \boldsymbol { m } \| _ { 0 } = K d$ . The problem becomes:

$$
\min _ {\boldsymbol {m} \in \{0, 1 \} ^ {d}, \| \boldsymbol {m} \| _ {0} = K d} \sum_ {i} c _ {i} m _ {i}.\tag{13}
$$

This guides us to select the Kd parameters with the smallest $c _ { i }$ values (setting $m _ { i } = 1 )$ while keeping the rest unchanged $( m _ { i } = 0 )$ . Thus, CGM<sup>†</sup> performs a sparse and modular update, efectively identifying and applying only the most critical parameter changes from the fine-tuning.

## 4 Experiments

## 4.1 Experimental Setup

Architectures and Datasets. We evaluate our methods on two representative MLLMs: LLaVA-1.5-7B [21] and Qwen-2.5VL-3B [1]. For each architecture, we partition the datasets into two categories to separately evaluate generalization capability and downstream adaptability. For LLaVA-1.5-7B [21], we finetune on OKVQA [26] and Flickr30k [38], representing visual question answering and image captioning, respectively, and evaluate generalization on a standard benchmark suite comprising VQAv2 [8], GQA [14], VizWiz [9], SQA [23], TextVQA [32], POPE [19], MM-Bench [22], and MM-Bench-CN [45]. Similarly, for Qwen-2.5VL-3B [1], we fine-tune on the Flickr30k [38] and LaTeX-OCR [31] datasets and assess retained general knowledge on the same benchmark suite, additionally including InfoVQA [29] and OKVQA [26] to provide a more comprehensive evaluation of multimodal reasoning and knowledge retention.

Compared Baselines. We evaluate our method against a diverse set of baselines, including the naive fine-tuning approach and model merging techniques.

– Standard Fine-Tuning [2]: Full task-specific fine-tuning without anti-forgetting strategies.

– Tailor [46]: Merges models by preserving task-critical parameters.

– DARE $I 4 1 / \colon$ Uses random parameter selection and rescaling to balance generalization.

– Grafting [30]: Employs skill localization to identify sparse, task-critical weights.

– Magnitude: Heuristically reverts parameters with the smallest absolute changes to pre-trained values.

– Wanda $\ [ 3 4 ] \colon$ Reverts parameters based on an importance score of weight magnitude and input activations.

Evaluation Metrics. We use (i) the score on the fine-tuning task (reflecting specialization) and (ii) the average score across a suite of general pre-training evaluation tasks (reflecting generalization). To holistically evaluate efectiveness in mitigating catastrophic forgetting in MLLMs, we adopt two aggregate metrics: Average Performance $( \operatorname { A v g } )$ and Harmonic Mean Score (Hscore). Formally, let $S _ { \mathrm { T a r g e t } }$ denote the performance on the target task, and $S _ { \mathrm { P r e - A v g } } =$ $\textstyle { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } S _ { i }$ be the average performance over N general pre-training tasks. Then:

$$
\mathrm{Avg} = \frac {S _ {\mathrm{Target}} + N \cdot S _ {\mathrm{Pre-Avg}}}{N + 1},\tag{14}
$$

$$
\mathrm{Hscore} = \frac {2 \cdot S _ {\text {Target}} \cdot S _ {\text {Pre - Avg}}}{S _ {\text {Target}} + S _ {\text {Pre - Avg}}}.\tag{15}
$$

Here, the Avg metric equally weights adaptation and generalization, while Hscore penalizes imbalanced performance.

Implementation Details. We follow the oficial codebases to fine-tune LLaVA-1.5-7B [21] and Qwen-2.5VL-3B [1]. For LLaVA-1.5, we fine-tune the last 12 layers of the language model and the visual projector for 1 epoch with a learning rate of 1\mathr {e}-4 and a global batch size of 64. For Qwen-2.5VL, we fine-tune the last 6 layers and the visual projector for 3 epochs with a learning rate of 1\mathr {e}-5 and the same batch size. We use the empirical Fisher Information Matrix as an eficient estimator of the diagonal Hessian. All experiments are run on 5 NVIDIA RTX 6000 Ada GPUs (48 GB each).

Computation and Memory Cost. Obtaining $\pmb { w } ^ { \mathrm { f t } }$ follows standard supervised fine-tuning. Both CGM (soft mixing) and CGM<sup>†</sup> (hard mixing) perform element-wise operations on the selected layers (the last 12 layers for LLaVA and the last 6 layers for Qwen), yielding $\mathcal O ( d )$ complexity, where $d$ is the number of parameters in those layers. The main additional cost is estimating diagonal Hessians. For $\mathbf { H } ^ { \mathrm { p t } }$ , we use a calibration set sampled from pre-training tasks (8 samples per task in practice) and compute squared gradients at ${ \pmb w } ^ { \mathrm { p t } }$ to obtain an empirical FIM estimate. For $\mathbf { H } ^ { \mathrm { f t } }$ , we accumulate squared gradients over the finetuning dataset. Empirically, incorporating FIM estimation during fine-tuning reduces throughput from 3.81 to 3.51 samples/s (∼7.9% overhead), confirming negligible extra computation. For memory cost, this storage pattern is standard in model-merging methods: we keep only element-wise second-order statistics (the diagonal vectors $\mathbf { h } ^ { \mathrm { p t } }$ and $\mathbf { h } ^ { \mathrm { f t } } )$ . This is substantially more memory-eficient than methods that jointly store first-order, second-order, and other auxiliary statistics.

Table 1: Main results on the LLaVA-1.5-7B backbone, fine-tuned on the OKVQA target task.

<table><tr><td rowspan="2">Method</td><td colspan="9">Pre-trained Tasks</td><td>Target Task</td><td colspan="2">Overall Metrics</td></tr><tr><td>VQAv2</td><td>GQA</td><td>VizWiz</td><td>SQA</td><td>TextVQA</td><td>POPE</td><td>MM-Bench</td><td>MM-Bench-CN</td><td>Pre-Avg</td><td>OKVQA</td><td>Hscore</td><td>Avg</td></tr><tr><td>Pre-trained Model</td><td>78.5</td><td>61.9</td><td>50.0</td><td>70.4</td><td>58.2</td><td>87.3</td><td>64.3</td><td>58.3</td><td>66.1</td><td>52.8</td><td>58.7</td><td>64.6</td></tr><tr><td>Fine-tuned Model</td><td>68.0</td><td>50.4</td><td>38.7</td><td>24.5</td><td>40.1</td><td>83.2</td><td>56.9</td><td>41.6</td><td>50.4</td><td>58.0</td><td>53.9</td><td>51.2</td></tr><tr><td>Tailor</td><td>71.7</td><td>56.1</td><td>40.1</td><td>69.8</td><td>50.5</td><td>82.3</td><td>64.4</td><td>58.2</td><td>61.6</td><td>57.6</td><td>59.6</td><td>61.2</td></tr><tr><td>DARE</td><td>67.7</td><td>49.6</td><td>37.2</td><td>28.6</td><td>39.9</td><td>82.4</td><td>57.2</td><td>45.2</td><td>51.0</td><td>51.2</td><td>51.1</td><td>51.0</td></tr><tr><td>Grafting</td><td>72.5</td><td>56.1</td><td>44.0</td><td>70.6</td><td>51.0</td><td>88.5</td><td>62.0</td><td>51.4</td><td>62.0</td><td>57.1</td><td>59.5</td><td>61.5</td></tr><tr><td>Magnitude</td><td>69.7</td><td>52.8</td><td>41.5</td><td>33.0</td><td>44.6</td><td>84.5</td><td>62.7</td><td>52.8</td><td>54.0</td><td>54.5</td><td>54.9</td><td>55.1</td></tr><tr><td>Wanda</td><td>68.8</td><td>51.2</td><td>39.1</td><td>30.3</td><td>41.4</td><td>83.1</td><td>58.8</td><td>45.1</td><td>52.2</td><td>53.9</td><td>53.1</td><td>52.4</td></tr><tr><td>CGM (Ours)</td><td> $74.3 \pm 0.04$ </td><td> $58.5 \pm 0.06$ </td><td> $52.3 \pm 0$ </td><td> $^{69}69.3 \pm 0.13$ </td><td> $53.8 \pm 0.35$ </td><td> $86.2 \pm 0.12$ </td><td> $63.7 \pm 0.27$ </td><td> $55.6 \pm 0.40$ </td><td> $64.2 \pm 0.03$ </td><td> $59.8 \pm 0.21$ </td><td> $61.9 \pm 0.11$ </td><td> $63.7 \pm 0.03$ </td></tr><tr><td> $CGM^{\dagger}$ (Ours)</td><td> $76.2 \pm 0.04$ </td><td> $61.4 \pm 0.07$ </td><td> $49.5 \pm 0$ </td><td> $70.6 \pm 0.12$ </td><td> $55.3 \pm 0.11$ </td><td> $87.4 \pm 0.20$ </td><td> $65.1 \pm 0.32$ </td><td> $59.5 \pm 0.40$ </td><td> $65.7 \pm 0.05$ </td><td> $60.2 \pm 0.22$ </td><td> $62.8 \pm 0.12$ </td><td> $65.0 \pm 0.07$ </td></tr></table>

<sup>a</sup> The VizWiz server did not respond during our standard-deviation evaluation.

Table 2: Main results on the LLaVA-1.5-7B backbone, fine-tuned on the Flickr30k target task.

<table><tr><td rowspan="2">Method</td><td colspan="9">Pre-trained Tasks</td><td>Target Task</td><td colspan="2">Overall Metrics</td></tr><tr><td>VQAv2</td><td>GQA</td><td>VizWiz</td><td>SQA</td><td>TextVQA</td><td>POPE</td><td>MM-Bench</td><td>MM-Bench-CN</td><td>Pre-Avg</td><td>Flickr30k</td><td>Hscore</td><td>Avg</td></tr><tr><td>Pre-trained Model</td><td>78.5</td><td>61.9</td><td>50.0</td><td>70.4</td><td>58.2</td><td>87.3</td><td>64.3</td><td>58.3</td><td>66.1</td><td>11.73</td><td>19.93</td><td>60.08</td></tr><tr><td>Fine-tuned Model</td><td>67.8</td><td>46.7</td><td>22.4</td><td>67.5</td><td>37.9</td><td>76.1</td><td>59.7</td><td>50.8</td><td>53.6</td><td>53.57</td><td>53.59</td><td>53.6</td></tr><tr><td>Tailor</td><td>70.8</td><td>51.7</td><td>13.2</td><td>69.3</td><td>42.9</td><td>77.6</td><td>61.1</td><td>56.9</td><td>55.4</td><td>41.85</td><td>47.69</td><td>53.92</td></tr><tr><td>DARE</td><td>62.4</td><td>43.7</td><td>6.2</td><td>67.8</td><td>20.8</td><td>85.3</td><td>59.6</td><td>45.9</td><td>49.0</td><td>52.48</td><td>50.66</td><td>49.36</td></tr><tr><td>Grafting</td><td>68.7</td><td>49.0</td><td>14.5</td><td>69.9</td><td>41.8</td><td>67.9</td><td>64.5</td><td>55.5</td><td>54.0</td><td>30.56</td><td>39.02</td><td>51.37</td></tr><tr><td>Magnitude</td><td>69.3</td><td>48.6</td><td>22.1</td><td>69.6</td><td>40.2</td><td>74.5</td><td>61.2</td><td>53.0</td><td>54.8</td><td>52.78</td><td>53.77</td><td>54.57</td></tr><tr><td>Wanda</td><td>69.8</td><td>49.2</td><td>33.5</td><td>67.8</td><td>40.0</td><td>79.6</td><td>60.1</td><td>52.8</td><td>56.6</td><td>50.81</td><td>53.56</td><td>55.97</td></tr><tr><td>CGM (Ours)</td><td> $77.1_{\pm 0.02}$ </td><td> $59.5_{\pm 0.09}$ </td><td> $53.3_{\pm 0}$ </td><td> $69.3_{\pm 0.25}$ </td><td> $49.5_{\pm 0.06}$ </td><td> $86.5_{\pm 0.26}$ </td><td> $62.7_{\pm 0.22}$ </td><td> $52.7_{\pm 0.69}$ </td><td> $63.8_{\pm 0.08}$ </td><td> $47.9_{\pm 0.74}$ </td><td> $54.71_{\pm 0.17}$ </td><td> $62.01_{\pm 0.03}$ </td></tr><tr><td> $CGM^{\dagger}$ (Ours)</td><td> $72.4_{\pm 0.02}$ </td><td> $54.0_{\pm 0.21}$ </td><td> $50.6_{\pm 0}$ </td><td> $69.7_{\pm 0.05}$ </td><td> $45.0_{\pm 0.30}$ </td><td> $80.1_{\pm 0.06}$ </td><td> $63.0_{\pm 0.13}$ </td><td> $55.7_{\pm 0.29}$ </td><td> $61.3_{\pm 0.09}$ </td><td> $51.91_{\pm 0.82}$ </td><td> $56.22_{\pm 0.26}$ </td><td> $60.26_{\pm 0.15}$ </td></tr></table>

## 4.2 Main Results

Tables 1–4 summarize the main results across multiple datasets and backbones. We evaluate performance using Hscore and Avg, which reflect the trade-of between adaptation to new tasks and retention of general knowledge.

Across all experiments, existing approaches reveal a clear dilemma: the Finetuned Model achieves strong performance on the target task but sufers severe degradation on pre-trained tasks, while the Pre-trained Model retains general capabilities yet fails to adapt. Methods such as DARE and Wanda partially alleviate this issue but still incur substantial knowledge loss. In contrast, our methods CGM and CGM<sup>†</sup> achieve the best overall balance. For example, in Table 1, CGM<sup>†</sup> attains the highest Hscore of 62.8 while maintaining nearly full general knowledge (Pre-Avg 65.7). Although the target task scores in Tables 3 and 4 are not the highest, our methods still achieve the best overall metrics, significantly outperforming existing approaches in maintaining both specialization and generalization.

## 4.3 Ablation Studies

We conduct an ablation study to verify the components of the curvature-aware score $\begin{array} { r } { c _ { i } = \frac { 1 } { 2 } ( h _ { i } ^ { \mathrm { f t } } - \alpha h _ { i } ^ { \mathrm { p t } } ) \varDelta _ { i } ^ { 2 } } \end{array}$ (Table 5). Relying solely on magnitude $( c _ { i } \propto \varDelta _ { i } ^ { 2 } )$ or fine-tuning curvature $( c _ { i } \propto h _ { i } ^ { \mathrm { f t } } \varDelta _ { i } ^ { 2 } )$ yields the lowest Hscores on LLaVA-OKVQA (54.0 and 53.1, respectively) due to severe overfitting and catastrophic forgetting. The pre-training variant $( c _ { i } \propto - \alpha h _ { i } ^ { \mathrm { p t } } \varDelta _ { i } ^ { 2 } )$ preserves general knowledge more efectively, improving the Hscore to 61.7. Ultimately, the full formulation consistently achieves the highest scores across all datasets. By jointly modeling both curvatures, it provides an optimal trade-of between adaptation to new tasks and pre-trained model inertia.

Table 3: Main results on the Qwen-2.5VL-3B backbone, fine-tuned on the Flickr30k target task.

<table><tr><td rowspan="2">Method</td><td colspan="9">Pre-trained Tasks</td><td>Target Task</td><td colspan="2">Overall Metrics</td></tr><tr><td>VQAv2</td><td>GQA</td><td>VizWiz</td><td>SQA</td><td>TextVQA</td><td>POPE</td><td>MM-Bench</td><td>MM-Bench-CN</td><td>Pre-Avg</td><td>Flickr30k</td><td>Hscore</td><td>Avg</td></tr><tr><td>Pre-trained Model</td><td>80.7</td><td>58.9</td><td>65.9</td><td>82.7</td><td>75.6</td><td>87.6</td><td>77.7</td><td>76.6</td><td>75.7</td><td>30.62</td><td>43.61</td><td>70.71</td></tr><tr><td>Fine-tuned Model</td><td>72.9</td><td>46.8</td><td>57.0</td><td>79.2</td><td>66.2</td><td>83.9</td><td>75.6</td><td>63.0</td><td>68.1</td><td>52.15</td><td>59.05</td><td>66.3</td></tr><tr><td>Tailor</td><td>70.3</td><td>46.4</td><td>56.1</td><td>79.7</td><td>61.2</td><td>84.6</td><td>75.6</td><td>60.7</td><td>66.8</td><td>47.97</td><td>55.85</td><td>64.73</td></tr><tr><td>DARE</td><td>73.4</td><td>47.8</td><td>54.2</td><td>78.2</td><td>64.0</td><td>86.6</td><td>73.7</td><td>57.0</td><td>66.9</td><td>51.04</td><td>57.89</td><td>65.11</td></tr><tr><td>Grafting</td><td>68.8</td><td>44.3</td><td>59.3</td><td>80.5</td><td>62.4</td><td>76.0</td><td>76.7</td><td>70.1</td><td>67.2</td><td>29.11</td><td>40.63</td><td>63.01</td></tr><tr><td>Magnitude</td><td>72.0</td><td>45.8</td><td>56.4</td><td>79.9</td><td>65.7</td><td>83.2</td><td>76.2</td><td>63.7</td><td>67.8</td><td>50.85</td><td>58.13</td><td>65.95</td></tr><tr><td>Wanda</td><td>72.8</td><td>46.8</td><td>57.2</td><td>79.3</td><td>66.1</td><td>83.9</td><td>75.4</td><td>63.1</td><td>68.1</td><td>51.46</td><td>58.61</td><td>66.23</td></tr></table>

CGM (Ours) 79.6<sub>±</sub> 56.1<sub>±</sub> 63.5<sub>±</sub> 82.0<sub>±</sub> 73.2<sub>±</sub> 86.9<sub>±</sub> 78.1<sub>±</sub> $7 6 . 8 _ { \pm 0 . 0 9 }$ 74.5<sub>± 0.08</sub> $4 9 . 1 _ { \pm 0 . 2 6 }$ $\underline { { 5 9 . 2 _ { \pm 0 . 0 5 } } }$ 71.69<sub>±</sub> CGM<sup>†</sup> (Ours) 76.9 53.0 61.0 82.2 72.4 86.5 78.2 76.7<sub>± 0.20</sub> 73.3<sub>± 0.10</sub> 50.09<sub>± 0.29</sub> 59.52<sub>± 0.09</sub> 70.75<sub>± 0.08</sub>

Table 4: Main results on the Qwen-2.5VL-3B backbone, fine-tuned on the LaTeX-OCR target task.

<table><tr><td rowspan="2">Method</td><td colspan="9">Pre-trained Tasks</td><td>Target Task</td><td colspan="2">Overall Metrics</td></tr><tr><td>VQAv2</td><td>GQA</td><td>VizWiz</td><td>SQA</td><td>TextVQA</td><td>POPE</td><td>InfoVQA</td><td>OKVQA</td><td>Pre-Avg</td><td>LaTeX-OCR</td><td>Hscore</td><td>Avg</td></tr><tr><td>Pre-trained Model</td><td>80.7</td><td>58.9</td><td>65.9</td><td>82.7</td><td>75.6</td><td>87.6</td><td>61.3</td><td>56.1</td><td>71.1</td><td>21.1</td><td>32.6</td><td>65.6</td></tr><tr><td>Fine-tuned Model</td><td>66.8</td><td>42.9</td><td>52.7</td><td>81.8</td><td>65.9</td><td>80.8</td><td>54.8</td><td>44.1</td><td>61.2</td><td>72.7</td><td>66.5</td><td>62.5</td></tr><tr><td>Tailor</td><td>69.7</td><td>45.5</td><td>54.4</td><td>82.3</td><td>70.4</td><td>82.3</td><td>58.2</td><td>46.7</td><td>63.7</td><td>57.0</td><td>60.2</td><td>62.9</td></tr><tr><td>DARE</td><td>66.3</td><td>44.1</td><td>50.9</td><td>78.5</td><td>63.3</td><td>82.9</td><td>53.1</td><td>42.7</td><td>60.2</td><td>74.8</td><td>66.7</td><td>61.8</td></tr><tr><td>Grafting</td><td>72.6</td><td>48.4</td><td>53.8</td><td>82.7</td><td>72.3</td><td>83.9</td><td>59.0</td><td>48.3</td><td>65.1</td><td>67.0</td><td>66.1</td><td>65.3</td></tr><tr><td>Magnitude</td><td>68.5</td><td>44.3</td><td>53.5</td><td>82.4</td><td>67.6</td><td>81.4</td><td>55.7</td><td>45.3</td><td>62.4</td><td>76.0</td><td>68.5</td><td>63.9</td></tr><tr><td>Wanda</td><td>67.1</td><td>43.1</td><td>52.9</td><td>82.2</td><td>66.4</td><td>81.0</td><td>55.1</td><td>44.3</td><td>61.5</td><td>74.8</td><td>67.5</td><td>63.0</td></tr></table>

CGM (Ours) 80.9 58.9 66.0 82.7 75.8 87.6 60.0 57.1 71.1 70.9<sub>± 1.27</sub> 71.0<sub>± 0.64</sub> 71.1<sub>± 0.15</sub> CGM<sup>†</sup> (Ours) 80.4 57.8 65.0 82.7 75.5 86.6 60.1 56.6 70.6 74.9 72.7<sub>± 0.04</sub> 71.1<sub>± 0.03</sub>

Table 5: Ablation study on the components of the $\mathrm { C G M ^ { \dag } }$ score $\begin{array} { r } { c _ { i } = \frac { 1 } { 2 } ( { h } _ { i } ^ { \mathrm { f t } } - \alpha { h } _ { i } ^ { \mathrm { p t } } ) \varDelta _ { i } ^ { 2 } } \end{array}$ We compare four variants by selectively omitting components from the score. All variants update the same number of parameters (K% of the total).

<table><tr><td rowspan="2">Variant</td><td rowspan="2">Score  $c_i \propto$ </td><td colspan="2">LLaVA - OKVQA</td><td colspan="2">LLaVA - Flickr30k</td><td colspan="2">Qwen3B - LaTeX-OCR</td></tr><tr><td>Hscore</td><td>Avg</td><td>Hscore</td><td>Avg</td><td>Hscore</td><td>Avg</td></tr><tr><td>1. Magnitude-only</td><td> $\Delta_i^2$ </td><td>54.0</td><td>54.0</td><td>53.59</td><td>53.61</td><td>67.6</td><td>62.9</td></tr><tr><td>2. Fine-tune+Magnitude</td><td> $h_i^{\text{ft}} \Delta_i^2$ </td><td>53.1</td><td>52.4</td><td>53.59</td><td>53.61</td><td>67.2</td><td>62.7</td></tr><tr><td>3. Pre-train+Magnitude</td><td> $-\alpha h_i^{\text{pt}} \Delta_i^2$ </td><td>61.7</td><td>65.1</td><td>55.76</td><td>60.25</td><td>71.7</td><td>70.8</td></tr><tr><td>4. CGM $^†$ (Full)</td><td> $(h_i^{\text{ft}} - \alpha h_i^{\text{pt}}) \Delta_i^2$ </td><td>62.8</td><td>65.0</td><td>56.22</td><td>60.26</td><td>72.7</td><td>71.1</td></tr></table>

## 4.4 Hyperparameter Sensitivity

We conduct a sensitivity analysis on our two key hyperparameters, starting with the sparsity ratio K, which governs the fraction of pre-trained parameters retained. As shown in Figure 4, general knowledge preservation (Pre-Avg score) is remarkably robust, remaining nearly flat even as K varies from 10% to 90%. This demonstrates that retaining as few as 10% of the critically identified pretrained parameters is suficient to preserve almost all general knowledge and prevent catastrophic forgetting. Consequently, the primary role of K is to control the degree of downstream specialization rather than managing forgetting, with the optimal Hscore and target performance typically achieved at a 10% sparsity ratio.

![](images/42167e5ed70f4fdedaa1ef074cf9f218d45513f9d490b749e2c0b438167d11a0.jpg)  
Fig. 4: Hyperparameter sensitivity analysis on the Qwen3B backbone for the LaTeX-OCR and Flickr30k tasks.

Similarly, the balancing coeficient α modulates the trade-of between target task adaptation and pre-trained knowledge retention. Across varying values of α, the Pre-Avg score exhibits minimal variance, confirming that our joint objective inherently shields foundational knowledge. Therefore, α primarily influences target task performance, which in turn dictates the overall Hscore. The best balance is consistently found at smaller values (α ≈ 0.1 ∼ 0.15), which provide suficient adaptation to the new task while efectively maintaining inherent general capabilities.

## 4.5 Analysis of Selection Mask Structure

To understand the qualitative diferences in parameter selection, Figure 5 visualizes downsampled masks on the self\_attn.o\_proj layer. Tailor and Magnitude produce difuse, unstructured noise. In contrast, CGM<sup>†</sup> exhibits distinct vertical bands, indicating a coherent, feature-level selection focused on specific input dimensions.

We quantify this via the column-wise recovery ratio in Figure 6. The Magnitude baseline applies updates uniformly across columns at all sparsity levels. Conversely, CGM<sup>†</sup> consistently targets or protects specific structural column groups (e.g., blocks ∼80, ∼120, and ∼160). This provides strong evidence that our curvature-guided score isolates structured, critically important parameter groups rather than relying on difuse heuristics.

![](images/a61c555d5cba951c1d8bae8698b045e07610646282dfcc2295e81eed03046563.jpg)  
(a) CGM<sup>†</sup> (Ours)

![](images/63744a6186a2ea06c439c9ae9231df0a8f9feb3ce75accee4373426aefe63e1d.jpg)  
(b) Tailor

![](images/78eba59ad0f9db9a1bf4421d34c3cbb18f296d1696bc1212ff428ed83c1138fd.jpg)  
(c) Magnitude

Fig. 5: Visualization of selection masks (10% sparsity) for diferent methods on the self\_attn.o\_proj layer. The masks are downsampled by patch-averaging for clarity. (a) Our CGM<sup>†</sup> mask shows a highly structured pattern, selecting by input dimensions (vertical bands). (b) Tailor and (c) Magnitude masks are unstructured and noisy, selecting parameters difusely across the matrix. This visually confirms that our curvature-guided score identifies coherent structural components for updates.  
![](images/ed594bf7946ab8134b18f4253097097d8ccb5fe19c7442102f9a7345650b5924.jpg)  
(a) Sparsity = 0.1

![](images/4a72fb26a9069d8bc346e69b3546166b250a13069ae339aba028f0baa9cdab88.jpg)  
(b) Sparsity = 0.3

![](images/f6de325661f30384e5a2700b90f59268ca13ce686eaad80469f55962ae3b4631.jpg)  
(c) Sparsity = 0.5  
Fig. 6: Quantitative comparison of column-wise recovery ratios between CGM<sup>†</sup> (orange) and Magnitude (Mag, blue) at varying update sparsity levels. The Y-axis represents the fraction of pre-trained parameters kept. Across all sparsity levels, CGM<sup>†</sup> exhibits a non-uniform, structured selection that consistently targets or protects the same columns, whereas the Magnitude baseline remains uniform and difuse.

## 5 Conclusion

In this paper, we introduce Curvature-Guided Mixing (CGM) and its sparse variant, CGM<sup>†</sup>, as theoretically grounded solutions to catastrophic forgetting in fine-tuned MLLMs. Our framework leverages a joint optimization objective and second-order information from the loss landscapes of both pre-training and fine-tuning tasks. This approach yields an optimal “soft mixing” ratio (CGM) and a robust, sparse “hard mixing” strategy (CGM<sup>†</sup>). Extensive experiments on the LLaVA and Qwen-VL models demonstrate that our methods significantly outperform prior art, establishing a new state-of-the-art in balancing task specialization and the preservation of general foundational knowledge. Our results confirm that leveraging loss landscape geometry provides a principled and efective approach to knowledge-preserving model adaptation.

## References

1. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)

2. de Boer, P.T., Kroese, D.P., Mannor, S., Rubinstein, R.Y.: A tutorial on the crossentropy method. Annals of Operations Research 134, 19–67 (2005)

3. Cao, M., Liu, Y., Liu, Y., Wang, T., Dong, J., Ding, H., Zhang, X., Reid, I., Liang, X.: Continual llava: Continual instruction tuning in large vision-language models. arXiv preprint arXiv:2411.02564 (2024)

4. Cen, J., Wu, C., Liu, X., Yin, S., Pei, Y., Yang, J., Chen, Q., Duan, N., Zhang, J.: Using left and right brains together: Towards vision and language planning. In: ICML. pp. 5982–6001 (2024)

5. Chen, C., Zhu, J., Luo, X., Shen, H.T., Song, J., Gao, L.: Coin: A benchmark of continual instruction tuning for multimodel large language models. NeurIPS 37, 57817–57840 (2024)

6. Chen, H., Yang, Y., Zhong, N., Ma, K.: Hiding images in difusion models by editing learned score functions. In: CVPR. pp. 18663–18673 (2025)

7. Frantar, E., Alistarh, D.: SparseGPT: Massive language models can be accurately pruned in one-shot. In: ICML (2023)

8. Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., Parikh, D.: Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In: CVPR. pp. 6904–6913 (2017)

9. Gurari, D., Li, Q., Stangl, A.J., Guo, A., Lin, C., Grauman, K., Luo, J., Bigham, J.P.: Vizwiz grand challenge: Answering visual questions from blind people. In: CVPR. pp. 3608–3617 (2018)

10. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., Gelly, S.: Parameter-eficient transfer learning for nlp. In: ICML. pp. 2790–2799 (2019)

11. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. In: ICLR (2022)

12. Huang, W., Liang, J., Guo, X., Fang, Y., Wan, G., Rong, X., Wen, C., Shi, Z., Li, Q., Zhu, D., Ma, Y., Liang, K., Yang, B., Li, H., Shao, J., Ye, M., Du, B.: Keeping yourself is important in downstream tuning multimodal large language model. arXiv preprint arXiv:2503.04543 (2025)

13. Huang, W., Liang, J., Shi, Z., Zhu, D., Wan, G., Li, H., Du, B., Tao, D., Ye, M.: Learn from downstream and be yourself in multimodal large language model fine-tuning. In: ICML (2025)

14. Hudson, D.A., Manning, C.D.: Gqa: A new dataset for real-world visual reasoning and compositional question answering. In: CVPR. pp. 6700–6709 (2019)

15. Jha, S., Gong, D., Yao, L.: CLAP4CLIP: Continual learning with probabilistic finetuning for vision-language models. In: NeurIPS (2024)

16. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al.: Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences 114(13), 3521–3526 (2017)

17. LeCun, Y., Denker, J.S., Solla, S.A.: Optimal brain damage. In: NeurIPS. vol. 2, pp. 598–605 (1990)

18. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: ICML (2023)

19. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, X., Wen, J.R.: Evaluating object hallucination in large vision-language models. In: EMNLP. pp. 292–305 (2023)

20. Liang, Y.S., Li, W.J.: Inflora: Interference-free low-rank adaptation for continual learning. In: CVPR. pp. 23638–23647 (2024)

21. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. In: CVPR (2024)

22. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? In: ECCV. pp. 216–233 (2024)

23. Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.W., Zhu, S.C., Tafjord, O., Clark, P., Kalyan, A.: Learn to explain: Multimodal reasoning via thought chains for science question answering. In: NeurIPS. vol. 35, pp. 2507–2521 (2022)

24. Luo, G., Yang, X., Dou, W., Wang, Z., Liu, J., Dai, J., Qiao, Y., Zhu, X.: Monointernvl: Pushing the boundaries of monolithic multimodal large language models with endogenous visual pre-training. In: CVPR (2025)

25. Luo, Y., Yang, Z., Meng, F., Li, Y., Zhou, J., Zhang, Y.: An empirical study of catastrophic forgetting in large language models during continual fine-tuning. IEEE/ACM Transactions on Audio, Speech, and Language Processing 33, 3776– 3786 (2025)

26. Marino, K., Rastegari, M., Farhadi, A., Mottaghi, R.: Ok-vqa: A visual question answering benchmark requiring external knowledge. In: CVPR. pp. 3195–3204 (2019)

27. Martens, J., Grosse, R.: Optimizing neural networks with kronecker-factored approximate curvature. In: ICML. pp. 2408–2417 (2015)

28. Matena, M., Rafel, C.: Merging models with fisher-weighted averaging. In: NeurIPS (2022)

29. Mathew, M., Bagal, V., Tito, R., Karatzas, D., Valveny, E., Jawahar, C.: Infographicvqa. In: WACV. pp. 1697–1706 (2022)

30. Panigrahi, A., Saunshi, N., Zhao, H., Arora, S.: Task-specific skill localization in fine-tuned language models. In: ICML. pp. 27011–27033 (2023)

31. Roboflow: Latex-ocr dataset (unsloth version). https : / / huggingface . co / datasets/unsloth/LaTeX\_OCR (2024)

32. Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., Rohrbach, M.: Towards vqa models that can read. In: CVPR. pp. 8317–8326 (2019)

33. Srinivasan, T., Chang, T.Y., Pinto Alva, L., Chochlakis, G., Rostami, M., Thomason, J.: Climb: A continual learning benchmark for vision-and-language tasks. NeurIPS 35, 29440–29453 (2022)

34. Sun, M., Liu, Z., Bair, A., Kolter, J.Z.: A simple and efective pruning approach for large language models. In: Workshop on Eficient Systems for Foundation Models @ ICML2023 (2023)

35. Wang, X., Zhuang, Z., Zhang, Y.: Plan: Proactive low-rank allocation for continual learning. In: ICCV. pp. 2909–2918 (2025)

36. Wu, J., Xiong, Y., Li, X., Xia, Y., Wang, R., Wang, Y., Yu, T., Kim, S., Rossi, R.A., Yao, L., Shang, J., McAuley, J.: Mitigating visual knowledge forgetting in MLLM instruction-tuning via modality-decoupled gradient descent. In: Findings of the Association for Computational Linguistics: EMNLP 2025. pp. 2282–2295 (2025)

37. Wu, Y., Piao, H., Huang, L., Wang, R., Li, W., Pfister, H., Meng, D., Ma, K., Wei, Y.: Sd-lora: Scalable decoupled low-rank adaptation for class incremental learning. In: ICLR (2025)

38. Young, P., Lai, A., Hodosh, M., Hockenmaier, J.: From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics 2, 67–78 (2014)

39. Yu, J., Xiong, H., Zhang, L., Diao, H., Zhuge, Y., Hong, L., Wang, D., Lu, H., He, Y., Chen, L.: Llms can evolve continually on modality for x-modal reasoning. NeurIPS 37, 49834–49858 (2024)

40. Yu, J., Zhuge, Y., Zhang, L., Hu, P., Wang, D., Lu, H., He, Y.: Boosting continual learning of vision-language models via mixture-of-experts adapters. In: CVPR. pp. 23219–23230 (2024)

41. Yu, L., Yu, B., Yu, H., Huang, F., Li, Y.: Language models are super mario: Absorbing abilities from homologous models as a free lunch. In: ICML (2024)

42. Zeng, F., Zhu, F., Guo, H., Zhang, X.Y., Liu, C.L.: Modalprompt: Towards eficient multimodal continual instruction tuning with dual-modality guided prompt. In: EMNLP. pp. 12126–12141 (2025)

43. Zenke, F., Poole, B., Ganguli, S.: Continual learning through synaptic intelligence. In: ICML. pp. 3987–3995 (2017)

44. Zhai, Y., Tong, S., Li, X., Cai, M., Qu, Q., Lee, Y.J., Ma, Y.: Investigating the catastrophic forgetting in multimodal large language model fine-tuning. In: Conference on Parsimony and Learning (2023)

45. Zhang, P., Dong, X., Wang, B., Cao, Y., Xu, C., Ouyang, L., Zhao, Z., Duan, H., Zhang, S., Ding, S., et al.: Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition. arXiv preprint arXiv:2309.15112 (2023)

46. Zhu, D., Sun, Z., Li, Z., Shen, T., Yan, K., Ding, S., Kuang, K., Wu, C.: Model tailor: Mitigating catastrophic forgetting in multi-modal large language models. In: ICML (2024)

47. Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., Gao, Z., Cui, E., Wang, X., Cao, Y., Liu, Y., Wei, X., Zhang, H., Wang, H., Xu, W., Li, H., Wang, J., Deng, N., Li, S., He, Y., Jiang, T., Luo, J., Wang, Y., He, C., Shi, B., Zhang, X., Shao, W., He, J., Xiong, Y., Qu, W., Sun, P., Jiao, P., Lv, H., Wu, L., Zhang, K., Deng, H., Ge, J., Chen, K., Wang, L., Dou, M., Lu, L., Zhu, X., Lu, T., Lin, D., Qiao, Y., Dai, J., Wang, W.: Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479 (2025)