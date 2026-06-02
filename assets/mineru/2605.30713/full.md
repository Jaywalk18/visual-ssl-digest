# Diversity Matters: Revisiting Test-Time Compute in Vision-Language Models

Yijie Tong \* 1 Yifan Hou \* 1 Shaobo Cui 2 3 Antoine Bosselut 3 Mrinmaya Sachan 1

{ yijie.tong, yifan.hou, mrinmaya.sachan }@inf.ethz.ch shaobo.cui@sjtu.edu.cn, antoine.bosselut@epfl.ch

# Abstract

Test-time compute (TTC) strategies have emerged as a lightweight approach to boost reasoning in large language models (LLMs). However, their application and benefits for vision-language models (VLMs) remain underexplored. We present a systematic study of TTC across seven VLMs and six benchmarks, specifically analyzing featurebased scoring and majority voting methods. We find that feature heuristics fail and voting yields only modest gains in single-model settings. We theoretically show that this limitation stems from a lack of prediction diversity: when outputs are highly correlated, voting provides little benefit. In contrast, multi-model ensembles offer richer diversity, yet standard majority voting fails to account for varying model capabilities. To address this, we propose Entropy-based TTC (ETTC), which selects the most confident prediction based on predictive entropy. Our method reduces to majority voting in the single-model case, but in model ensembles, it leverages confidence disparities to prioritize stronger models. We prove that ETTC outperforms majority voting under mild assumptions and empirically demonstrate that it consistently surpasses both voting and the best individual model. Crucially, our results show that smaller models can synergistically enhance larger ones, unlocking ensembling gains not achievable with standard strategies.1

\*Equal contribution 1ETH Zurich ¨ 2Shanghai Jiao Tong University 3EPFL. Correspondence to: Yifan Hou <yifan.hou@inf.ethz.ch>, Mrinmaya Sachan <mrinmaya.sachan@inf.ethz.ch>.

Proceedings of the 43 rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

1Our code is publicly available here: https://github.c om/nanfang-wuyu/Diversity-Matters.

# 1. Introduction

Vision-Language Models (VLMs) have recently achieved remarkable performance across a range of visual reasoning benchmarks (Llama Team, 2024; Agrawal et al., 2024; Gemma Team, 2025; Bai et al., 2025; OpenAI, 2023; Gemini Team, 2025). At the same time, the large language modeling (LLM) community has developed a family of testtime compute (TTC) strategies, particularly those based on chain-of-thought (CoT) prompting, to improve reasoning without modifying model parameters (Snell et al., 2024). These strategies generate multiple outputs per input and then aggregate or rank them to produce more reliable predictions.

In the LLM literature, inference-only TTC methods fall broadly into two paradigms. The first is response selection (often framed as Best-of-N), which scores and selects the most promising reasoning trace. While many of these methods rely on trained reward models, feature-based approaches avoid training by estimating quality through textual heuristics. These include analyzing structural signals such as the presence of specific pivot words (Chang et al., 2025; Lippmann & Yang, 2025), confident linguistic tone (Jiang et al., 2025b; Mao et al., 2025), or the length of the reasoning chain (Fu et al., 2023; Jin et al., 2024). In contrast, confidence-based (e.g., self-consistency) methods treat the model as a stochastic oracle and improve reasoning reliability by aggregating multiple outputs, typically selecting the most frequent answer across samples via majority voting (Wang et al., 2023; Chen et al., 2024c; Snell et al., 2024).

Applying TTC to VLMs, however, is far from straightforward. Unlike LLMs, VLMs must first perceive and interpret dense visual signals before reasoning over them. This introduces new challenges: (i) visual perception is inherently error-prone and varies across models (Bhattacharyya et al., 2023; Wang et al., 2025); (ii) vision-language alignment remains imperfect, creating subtle inconsistencies (Li et al., 2025; Yan et al., 2025); and (iii) textual cues that correlate with correctness in LLMs may not reflect true visual understanding (Al-Tahan et al., 2024; Jiang et al., 2025a). Therefore, it is unclear whether and when TTC strategies can reliably enhance visual reasoning.

To investigate this, we begin with the single-model (multiround) setting, where one VLM is queried multiple times with some notion of randomness (§ 3). Our findings reveal that: (1) feature-based methods fail to improve accuracy, showing that linguistic style is a poor proxy for visual reasoning quality; and (2) confidence-based methods such as majority voting provide only modest, but consistent, gains, and only when CoT prompting is used. Without CoT, even aggregation brings no benefit.

Next, we investigate why these gains are so limited. Specifically, we analyze the diversity (formally, the statistical dependency) between predictions and show that the effectiveness of voting decreases as predictions become more correlated (§ 4.1). When model outputs are nearly identical, voting cannot amplify the signal of correctness. Empirically, we confirm this across 7 VLMs and 6 datasets: outputs exhibit weak but nonzero dependency, which explains why voting offers only small improvements in practice (§ 4.2).

These insights point to a deeper limitation: in the singlemodel setting, diversity arises only from sampling randomness, so the expected skill of the model remains unchanged. By contrast, the multi-model ensemble setting naturally introduces stronger diversity: differences in architecture, training data, and even scale create complementary strengths. This makes ensembles both more realistic in practice and more promising for TTC. Existing methods, such as majority voting, cannot exploit this potential: by treating all models equally, voting risks letting weaker but correlated models dominate the outcome. What is needed is a strategy that adapts to model quality and selectively prioritizes the most reliable predictions.

To address this, we introduce a new strategy for visual reasoning: Entropy-based TTC (ETTC) (§ 5.1). Instead of counting votes, ETTC selects the prediction with the lowest entropy (on the answer distribution from multiple responses), that is, the most confident output distribution. In the single-model setting, ETTC reduces to majority voting, ensuring backward compatibility. But in multi-model ensembles, ETTC diverges from standard voting: it leverages confidence gaps across models, allowing smaller models to assist stronger ones rather than overwhelm them. We theoretically prove that ETTC outperforms majority voting under mild dependence assumptions (§ 5.2), and empirically show that it not only improves over voting but can even surpass the best individual model in the ensemble (§ 5.3). This result is particularly striking: smaller models can enhance larger ones when combined wisely, yielding gains not achievable with voting alone.

In summary, our contributions are:

• A systematic theoretical and empirical study of TTC in VLMs, showing that feature cues fail and that majority voting yields only modest CoT-dependent gains (§ 3).

• A theoretical analysis linking the effectiveness of voting to prediction dependency, supported by empirical evidence across diverse models and datasets (§ 4).   
• A new TTC strategy that generalizes majority voting and achieves consistent improvements in multi-model ensembles, often surpassing even the best single model (§ 5).

# 2. Preparation

We begin by outlining the models, datasets, prompting templates, baselines, and evaluation settings in our experiments.

Models. We evaluate seven open-source VLMs under two complementary multi-model ensemble configurations. In the similar-size (cross-family) setup, we include four VLMs with comparable parameter sizes but diverse architectures: Qwen2.5-VL-7B-Instruct (Bai et al., 2025, Qwen-7B), LLaMA-3.2-11B-Vision (Llama Team, 2024, LLaMA), Gemma-3-12B-it (Gemma Team, 2025, Gemma), and Pixtral-12B-2409 (Agrawal et al., 2024, Pixtral). In the same-family (varied-size) setup, we use four models from the Qwen2.5-VL-Instruct family (Bai et al., 2025), ranging from 3B to 72B parameters (3B, 7B, 32B, 72B), allowing us to study scaling effects within a single model architecture.

Datasets. We experiment on six multiple-choice visual QA benchmarks covering three domains. For mathematical reasoning, we use the testmini split of MathVista (Lu et al., 2024) and the test set of MathVision (Wang et al., 2024). For diagram understanding, we include the test sets of TQA (Kim et al., 2019) and ScienceQA (Lu et al., 2022). For general visual reasoning, we use the validation splits of MMStar (Chen et al., 2024a) and MMMU (Yue et al., 2024). All datasets contain multiple-choice questions with K answer options (2 ≤ K ≤ 9). Further statistics, including domain, split size, and option counts, are summarized in Tab. 3 of § B.1.

Decoding. We generate responses using stochastic decoding (Sutskever et al., 2014) via default settings.2 We adopt two prompting formats: (1) Direct Answer prompting discourages intermediate reasoning and elicits immediate answers; (2) Chain-of-thought (CoT) prompting explicitly encourages step-by-step reasoning, followed by a final answer. We use zero-shot, one-stage prompting for both settings to ensure consistency across models. Full prompt templates are provided in Figs. 4 and 5 of § B.2. Final answers are extracted from the text using regular expressions.

TTC Baselines. To revisit test-time compute strategies for visual reasoning, we evaluate four representative baselines spanning the feature-based selection and confidence-based aggregation paradigms. Three are feature-based Best-of-N methods that score and rank CoT responses using lexical heuristics, rather than requiring a trained reward model: (1) CoT Pivot Word ranks each response by counting predefined reasoning-related expressions (e.g., “alternatively”) (Chang et al., 2025; Lippmann & Yang, 2025); see the full phrase list in Tab. 4 of § B.3. (2) CoT Length prefers longer responses, following prior work suggesting a correlation between length and reasoning quality (Fu et al., 2023; Jin et al., 2024). (3) Feature-All combines four interpretable features (pivot word count, vague word count, total token count, and lexical diversity) to compute a composite score (see Tab. 6). As a confidence-based method, (4) Majority Voting (Wang et al., 2023; Chen et al., 2024c; Snell et al., 2024) aggregates N = 16 samples and selects the most frequent final answer.

![](images/6de2e4927de3b565f407e6babd6ba365921843d79ec13f67f67654cfb330179a.jpg)

<details>
<summary>bar</summary>

| Category | Vanilla (%) | Majority Voting (%) |
| :--- | :--- | :--- |
| Overall | 58 | 59 |
| MathVista | 64 | 65 |
| MathVision | 29 | 30 |
| TQA | 75 | 77 |
| ScienceQA | 80 | 81 |
| MMStar | 49 | 50 |
| MMMU | 49 | 50 |
</details>

(a) Direct Answer Prompt.

![](images/151b4501e271bb5133b6e44938c448b80687fecfb9c944b8a4c83c71edaa0dd2.jpg)

<details>
<summary>bar</summary>

| Category | Vanilla | Pivot Word | CoT Length | Feature-All | Majority Voting |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Overall | 60 | 59 | 57 | 60 | 63 |
| MathVista | 66 | 66 | 62 | 66 | 70 |
| MathVision | 31 | 30 | 28 | 31 | 34 |
| TQA | 77 | 77 | 75 | 78 | 80 |
| ScienceQA | 79 | 79 | 78 | 79 | 81 |
| MMStar | 53 | 53 | 51 | 53 | 56 |
| MMMU | 51 | 51 | 50 | 51 | 54 |
</details>

(b) CoT Prompt.   
Figure 1. Comparison of test-time compute (TTC) strategies under two prompting styles. In Direct Answer (left), models are instructed to output only the final answer without reasoning; feature-based methods are inapplicable, and majority voting shows no improvement. In CoT (right), models are prompted to reason step by step. While feature-based methods yield no gains, voting offers modest but consistent improvement across datasets.

Evaluation Settings. We assess all test-time compute methods under two settings: (1) In the single-model (multiround) setting, a single VLM is queried N times per question with stochastic decoding. Test-time compute is used to aggregate these intra-model outputs. (2) In the multi-model ensemble setting, M distinct VLMs are queried per question (each with multiple samples), introducing both intraand inter-model variation. This setting allows us to study cross-model complementarity and test whether aggregating weaker models can improve upon the best individual model.

# 3. Whether TTC Works in Visual Reasoning

We begin by revisiting whether TTC strategies, widely used in LLMs, improve visual reasoning in VLMs. We evaluate four representative methods across six multiple-choice visual benchmarks and compare their performance under two prompting conditions: direct answering and CoT. Results are averaged across seven VLMs.

Direct Answer Prompt: TTC fails without CoT. The Direct Answer setting tests whether test-time variation alone, without prompting explicit reasoning, can boost accuracy. Since no reasoning chains are produced, only confidencebased methods like majority voting are applicable.

As shown in Fig. 1 (left, direct answer), voting provides negligible or no improvement over the greedy baseline (often < 1%). Although we sample 16 outputs per question with stochastic decoding, the model’s predictions remain mostly identical. This suggests that without CoT prompting, VLMs tend to output the same surface-level answer, showing little diversity in interpretation. Consequently, TTC offers no benefit under direct answering. This aligns with findings in LLMs (Wang et al., 2023; Snell et al., 2024), but the issue is further exacerbated in VLMs by the perception bottleneck: visual content must first be accurately grounded before any meaningful reasoning variation can emerge.

CoT Prompt: Confidence helps, features do not. In contrast, when models are prompted to reason step-by-step, test-time strategies have room to work. This setup enables both feature-based scoring (e.g., using CoT length) and confidence-based aggregation (e.g., majority voting).

As shown in Fig. 1 (right, CoT), voting consistently improves performance across all benchmarks, with average gains of 2-4%. This validates the utility of test-time sampling under CoT: the model explores diverse reasoning paths and occasionally corrects itself. However, the improvements are modest, suggesting that sampled CoTs are still highly correlated, a hypothesis we will formally investigate in § 4. Meanwhile, feature-based methods fail to provide any consistent gain over vanilla CoT. Their performance often fluctuates slightly around the baseline. This highlights a key difference from LLMs: in VLMs, textual heuristics are poor proxies for reasoning correctness because visual understanding is the bottleneck. If perception fails, even a well-formed

CoT cannot save the answer.

Takeaway. TTC can improve visual reasoning, but only under specific conditions. Without CoT prompting, models produce nearly identical outputs, leaving no room for improvement. Even with CoT, gains from voting are modest, and feature-based scoring fails to help, highlighting the unique challenges of visual reasoning where perception quality limits downstream reasoning. This raises a key question: when does TTC actually $h e l p ?$ To answer this, we now turn to the analysis of majority voting, focusing on how its effectiveness depends on the statistical dependencies among model predictions.

# 4. When Does TTC Work?

Why does test-time compute (TTC), especially majority voting, sometimes fail to improve accuracy in visual reasoning? We address this question by analyzing how the statistical dependency among model predictions influences the effectiveness of voting. To this end, we develop a theoretical framework that quantifies this relationship and support it with empirical evidence.

# 4.1. Theoretical Insight: TTC Helps with Diverse Predictions

Intuition. Before diving into formal definitions, the core intuition is simple: if a model makes the exact same mistake every time you ask it a question, taking a vote across multiple attempts will not fix the error. Voting only amplifies accuracy if the model’s responses are somewhat diverse (i.e., not completely dependent on one another) but lean toward the correct answer on average. We formalize this by defining the “dependency” between predictions and proving that as dependency goes up, the benefit of voting goes down.

Setup. Suppose we have a multiple-choice question with K options and a single true answer, denoted as $Y \in [ K ]$ . We gather a total of U predictions, labeled $X _ { 1 } , \ldots , X _ { U }$ . These can come from multiple decoding rounds of a single VLM or from different VLMs in an ensemble.3 We evaluate whether a specific prediction is correct using a binary indicator $Z _ { u } : = \mathbb { I } \{ X _ { u } = Y \}$ , and define the model’s expected accuracy on a single trial as $p : = \mathbb { E } [ Z _ { u } ]$ .

To represent majority voting mathematically, we count the total votes for each option k, denoted as $S _ { k } : = { \bf \Omega }$ $\textstyle \sum _ { u = 1 } ^ { U } \mathbb { I } \{ X _ { u } = k \}$ . The final voting prediction is the option with the most votes, $\widehat { Y } _ { \mathrm { M V } } : = \arg \operatorname* { m a x } _ { k } S _ { k }$ . Finally, we define the overall accuracy of voting as $A _ { \mathrm { M V } } ( U ) : =$ $\mathbb { P } ( \widehat { Y } _ { \mathrm { M V } } = Y )$ , and the net improvement it provides over a single guess as $\Delta A _ { \mathrm { M V } } ( U ) : = A _ { \mathrm { M V } } ( U ) - p$ .

Dependency metrics. To measure how heavily the U predictions rely on each other, we quantify their dependency using two standard statistical metrics: normalized mutual information (NMI) for the raw answer options, and correlation for the correctness indicators.

For any two predicted answers X and $X ^ { \prime }$ , NMI measures the shared information between them, normalized by their individual uncertainty (entropy, H):

$$
\operatorname{NMI} (X; X ^ {\prime}) := \frac {I (X ; X ^ {\prime})}{\min \{H (X) , H (X ^ {\prime}) \}},
$$

$$
H (X) = - \sum_ {k = 1} ^ {K} \mathbb {P} (X = k) \log \mathbb {P} (X = k).
$$

For the full set of U predictions, the average NMI across all pairs is:

$$
\overline {{\mathrm{NMI}}} := \frac {2}{U (U - 1)} \sum_ {u <   v} \mathrm{NMI} \left(X _ {u}; X _ {v}\right).
$$

Similarly, for any two correctness indicators $Z$ and $Z ^ { \prime } { \mathrm { . } }$ , we define their statistical correlation (where $p$ is the single-trial accuracy), and average it across all pairs:

$$
\rho (Z, Z ^ {\prime}) := \frac {\mathbb {E} [ Z Z ^ {\prime} ] - p ^ {2}}{p (1 - p)}, \overline {{\rho}} := \frac {2}{U (U - 1)} \sum_ {u <   v} \rho (Z _ {u}, Z _ {v}).
$$

Theorem 1. Suppose all prediction pairs $( X _ { u } , X _ { v } )$ share the same dependency level $( i . e .$ , NMI or ρ). Then the voting improvement $\Delta A _ { \mathrm { M V } } ( U )$ is monotonically decreasing in both ρ and NMI. In particular:

$$
\overline {{\rho}} = 1 (o r \overline {{\mathrm{NMI}}} = 1) \Rightarrow \Delta A _ {\mathrm{MV}} (U) = 0,
$$

$$
\overline {{\rho}} = 0 (o r \overline {{\mathrm{NMI}}} = 0), p > \frac {1}{K} \Rightarrow A _ {\mathrm{MV}} (U) \rightarrow 1 a s U \rightarrow \infty .
$$

Interpretation. The formal proof is in § A.1. For practical application, this theorem reveals a powerful boundary condition for test-time compute: voting only improves accuracy when predictions are diverse. If all predictions are identical (correlation equals 1), voting reduces to a single prediction, yielding zero gain. Conversely, if predictions are entirely uncorrelated and individually better than random guessing, voting can aggregate the faint signals to achieve near-perfect accuracy given enough samples. Because NMI and correlation are model-agnostic, they serve as highly practical tools to estimate whether TTC will actually help on a given task, without needing access to ground truth labels.

# 4.2. Empirical Verification

To validate our theoretical insight, we structure our empirical evaluation into two parts. First, we determine the practical minimum number of decoding samples required to reliably estimate prediction dependency. Second, we test the core hypothesis: does the benefit of voting genuinely decrease as models become more correlated?

![](images/895596b8e78449c074fca2caa718820802bf3a0678d101f36192a7ad18b6f7fa.jpg)

<details>
<summary>line</summary>

| U  | MathVista | MathVision | TQA  | ScienceQA | MMStar | MMMU |
|----|-----------|------------|------|-----------|--------|------|
| 2  | 0.10      | 0.03       | 0.15 | 0.15      | 0.10   | 0.09 |
| 4  | 0.12      | 0.03       | 0.15 | 0.15      | 0.10   | 0.09 |
| 6  | 0.13      | 0.03       | 0.15 | 0.15      | 0.10   | 0.09 |
| 8  | 0.13      | 0.03       | 0.15 | 0.15      | 0.10   | 0.09 |
| 10 | 0.13      | 0.03       | 0.15 | 0.15      | 0.10   | 0.09 |
| 12 | 0.13      | 0.03       | 0.15 | 0.15      | 0.10   | 0.09 |
| 14 | 0.13      | 0.03       | 0.15 | 0.15      | 0.10   | 0.09 |
| 16 | 0.13      | 0.03       | 0.15 | 0.15      | 0.10   | 0.09 |
</details>

Figure 2. Convergence of dependency with decoding sample size U on Qwen-7B. Both NMI and ρ stabilize when U=12, suggesting that a moderate number of samples is sufficient to estimate dependency reliably.

Motivation. Our theoretical analysis assumes a sufficiently large number of decoding samples U for the benefits of voting to fully materialize. In practice, generating many samples incurs steep computational costs. Therefore, we first investigate how quickly our dependency metrics converge as U grows, aiming to find the minimal sample size that yields stable estimates.

Setup. We use Qwen-7B to generate U = 2 to 16 decoded outputs for each example across six visual reasoning datasets. For each subset size U, we compute the average normalized mutual information (NMI) and average correctness correlation (ρ) between response pairs.

Findings. As shown in Fig. 2, both NMI and ρ stabilize rapidly, flattening out around U = 12 across all datasets. Beyond this point, drawing additional samples offers minimal benefit in accurately estimating prediction dependency.

Takeaway. Sampling more than 12 to 16 responses provides diminishing returns. To ensure both statistical stability and computational tractability, we confidently use U = 16 in all subsequent experiments.

# 4.2.1. DOES VOTING IMPROVEMENT DECREASE WITH DEPENDENCY?

Motivation. We now test our central theoretical claim: voting is most beneficial when model outputs are diverse, meaning the accuracy gained from voting should measurably shrink as prediction dependency increases.

Setup. We evaluate the voting improvement $\Delta { \cal A } _ { \mathrm { M V } }$ (16) for seven models across six datasets, utilizing U = 16 decoding samples per query. For each model-dataset pair, we compute the average accuracy improvement alongside its average dependency (measured by both NMI and $\overline { { \rho } } )$ .

Findings. Fig. 3 shows a clear, consistent negative correlation between the improvement gained from voting and both dependency metrics. Smaller models (e.g., Qwen-3B, LLaMA) inherently produce more diverse outputs and therefore reap larger benefits from voting. In contrast, larger or more heavily optimized models (e.g., Qwen-72B, Pixtral) exhibit highly deterministic, low-diversity behavior, resulting in minimal gains from test-time aggregation. Detailed results broken down by dataset are in Figs. 6 and 7 in § C.1.

![](images/d307c23208b0c3dc1b49f46e0440f720e50c8891228baeaa07f43354d60d36fe.jpg)

<details>
<summary>scatter</summary>

| Method     | NMI   | ΔA_MV(16) |
| ---------- | ----- | --------- |
| LLaMA      | 0.1   | 4.0       |
| LLaMA      | 0.3   | 2.0       |
| LLaMA      | 0.6   | 4.0       |
| LLaMA      | 0.8   | 2.0       |
| Gemma      | 0.15  | 3.0       |
| Gemma      | 0.3   | 2.0       |
| Gemma      | 0.65  | 3.0       |
| Gemma      | 0.8   | 2.0       |
| Pixtral    | 0.1   | 5.5       |
| Pixtral    | 0.3   | 2.0       |
| Pixtral    | 0.7   | 2.0       |
| Qwen-3B    | 0.05  | 5.5       |
| Qwen-3B    | 0.2   | 3.0       |
| Qwen-3B    | 0.5   | 5.5       |
| Qwen-3B    | 0.8   | 2.0       |
| Qwen-7B    | 0.2   | 3.0       |
| Qwen-7B    | 0.3   | 2.0       |
| Qwen-7B    | 0.6   | 3.0       |
| Qwen-7B    | 0.8   | 2.0       |
| Qwen-32B   | 0.1   | 5.5       |
| Qwen-32B   | 0.2   | 3.0       |
| Qwen-32B   | 0.3   | 2.0       |
| Qwen-32B   | 0.6   | 3.0       |
| Qwen-32B   | 0.8   | 2.0       |
| Qwen-72B   | 0.3   | 2.0       |
| Qwen-72B   | 0.8   | 2.0       |
</details>

Figure 3. Majority voting improvement decreases with higher prediction dependency. Across models, we can find that voting improvement ∆AMV(16) is negatively correlated with both NMI and ${ \overline { { \rho } } } ,$ confirming theoretical predictions.

Takeaway. The effectiveness of majority voting hinges entirely on the diversity of model outputs. As predictions become more deterministic (higher NMI and ρ), voting offers sharply diminishing returns. This establishes a practical principle for deploying TTC: voting is most beneficial when applied to weaker/smaller models, or in uncertain scenarios (like few-shot tasks or domain shifts) where outputs are naturally more stochastic. Conversely, wrapping large, highly consistent models in a voting ensemble often wastes compute for negligible gain.

# 5. Beyond Voting: Entropy-Based TTC for Multi-Model Ensembles

Building on the insight that majority voting benefits from diverse yet independent predictions, we now turn to the more realistic and underexplored multi-model ensemble setting. Compared to multi-round decoding from a single model, which suffers from limited prediction diversity, ensembles of heterogeneous models naturally offer complementary strengths and errors. We first introduce an Entropy-based TTC method (ETTC) designed to better leverage crossmodel diversity. Then, we theoretically show that ETTC outperforms majority voting under mild assumptions, and empirically demonstrate that it enables smaller models to reliably enhance larger ones in visual reasoning tasks.

# 5.1. Entropy-Based TTC (ETTC)

Our previous analysis showed that the effectiveness of voting depends heavily on prediction diversity. However, majority voting has a deeper limitation in multi-model ensemble settings: it assumes all model responses are equally reliable and votes based solely on frequency, ignoring how confident or capable each individual model is. This oversight is less problematic in the single-model setting, since all predictions come from the same source and share the same expected quality. But in multi-model ensembles, where models vary drastically in size, training, and performance, this uniform treatment becomes a liability. A majority of weaker models can easily outvote a stronger one, even when the strong expert is confidently correct.

Intuition. To fix this, we need a mechanism that listens to the “expert” in the room for any given question. Intuitively, when a capable model knows the correct answer, its output probability distribution will be highly concentrated (low uncertainty). Conversely, when a weaker model guesses, its distribution will be flatter (high uncertainty). Therefore, instead of counting votes, we can select the answer from the model that is most certain. To operationalize this, we introduce Entropy-based Test-Time Compute (ETTC): a simple, model-agnostic method that selects the most confident prediction among multiple sources using normalized predictive entropy as a proxy for uncertainty.

Definition 1 (Entropy-Based Selection Rule). Let U sources (e.g., different models in an ensemble) each produce a predictive distribution $p _ { u } ( \cdot ) \in \Delta ^ { K - 1 }$ over K answer options. We define the normalized entropy of model u as:

$$
\widetilde {H} _ {u} := - \frac {1}{\log K} \sum_ {k = 1} ^ {K} p _ {u} (k) \log p _ {u} (k) \in [ 0, 1 ],
$$

and its top-1 prediction as ${ \hat { y } } _ { u } : = \arg \operatorname* { m a x } _ { k } p _ { u } ( k )$ . ETTC simply selects the prediction from the least-uncertain source:

$$
u ^ {\star} := \arg \min _ {u \in [ U ]} \widetilde {H} _ {u}, \quad \widehat {Y} _ {\min H} := \hat {y} _ {u ^ {\star}}.
$$

This selection rule prioritizes predictions with lower uncertainty. In contrast to majority voting, which can amplify weak or erroneous signals through sheer numbers, ETTC amplifies precision by trusting the most decisive prediction. Notably, ETTC safely reduces to standard voting in the single-model case (average the predictive distributions over multiple rounds and pick the most probable option). But in the multi-model setting, it diverges: it allows strong models to dominate the decision even when they are in the minority, which is essential for leveraging heterogeneous ensembles.

Takeaway. ETTC replaces raw vote counts with model confidence, providing a more principled and adaptive aggregation strategy. In real-world scenarios where model capabilities vary, ETTC prevents over-reliance on weaker models while fully exploiting the reliability of stronger ones.

# 5.2. Theoretical Insight: ETTC Outperforms Voting in Ensembles

In a multi-model ensemble, differences in training data and architecture naturally increase answer diversity. While voting treats all models equally, this can backfire: weaker models may collectively outvote stronger ones, especially when their errors are correlated (e.g., due to shared pre-training data). Our goal is to theoretically prove why ETTC provides a more robust alternative in such correlated scenarios.

Intuition. We base our theory on a simple premise: more confident predictions tend to be more accurate. In other words, if a model assigns a very high probability to the correct answer, its overall entropy will be low.

Assumption 1 (Entropy-Accuracy Monotonicity). For a given input with true label $Y ,$ , suppose model u assigns probability $p _ { u } ( Y )$ to Y , and $ { \widetilde { H } } _ { u }$ is its normalized entropy. Then, for all models $u , v \in [ U ]$ :

$$
p _ {u} (Y) > p _ {v} (Y) \quad \Rightarrow \quad \widetilde {H} _ {u} <   \widetilde {H} _ {v}.
$$

While this strict mathematical relationship may not hold perfectly in every single instance, we find that it holds strongly in aggregate practice across all datasets and models tested (see empirical verification in Fig. 8 of § C.2).

Given this assumption, ETTC simply selects the prediction from the most accurate model for that specific question, denoted as $u ^ { \star }$ . Let $c ^ { \star } : = \operatorname* { P r } ( \hat { y } _ { u ^ { \star } } = Y )$ be the accuracy of this best model. ETTC guarantees performance of at least $c ^ { \star }$ . Now, to understand where majority voting fails, we must model prediction dependency. Consider a simple coupling scheme: with probability λ, all the non-best models make a correlated error and copy the exact same prediction $W$ (e.g., due to shared biases). With probability 1 − λ, their predictions are independent. Let $\bar { c } : = \operatorname* { P r } ( W = Y )$ be the accuracy of this correlated “bloc” prediction, and let $A _ { \mathrm { M V } } ( 0 )$ be the baseline accuracy of majority voting if all models were perfectly independent.

Theorem 2 (Superiority of ETTC over Voting). With the setup above and under Assumption 1, let $A _ { \operatorname* { m i n } { H } } : =$ $\operatorname* { P r } ( \hat { y } _ { \operatorname* { m i n } { H } } = Y )$ be the accuracy of ETTC. Then for any correlation level $\lambda \in [ 0 , 1 ]$ , we have:

$$
A _ {\mathrm{MV}} (\lambda) = \lambda \bar {c} + (1 - \lambda) A _ {\mathrm{MV}} (0), \tag {1}
$$

$$
A _ {\mathrm{min} H} - A _ {\mathrm{MV}} (\lambda)
$$

$$
= \lambda (c ^ {\star} - \bar {c}) + (1 - \lambda) (A _ {\min H} - A _ {\mathrm{MV}} (0)).
$$

In particular, $A _ { \operatorname* { m i n } { H } } \geq A _ { \mathrm { M V } } ( \lambda )$ for all λ, with strict inequality whenever $\lambda > 0$ and $\bar { c } < c ^ { \star }$ .

Interpretation. We provide the full proof in $\ S \ A . 2$ . This result mathematically highlights the fundamental flaw of majority voting in ensembles. Because voting ignores model quality, it is highly vulnerable to correlated errors. As the error correlation λ increases (e.g., multiple weak models sharing the same flaw), voting accuracy degrades and is dragged down toward c¯, which is substantially lower than the expert model’s accuracy c⋆.

Table 1. Comparison of ETTC and Voting in the multi-model ensemble setting with similar-sized models from different families. ETTC consistently outperforms majority voting across all six datasets, with particularly large gains on benchmarks where model accuracies vary widely (e.g., MathVista, MathVision). This highlights ETTC’s ability to prioritize stronger models when aggregating predictions. 

<table><tr><td rowspan="2">Accuracy (%)</td><td colspan="4">Models</td><td rowspan="2">Average</td><td rowspan="2">Voting</td><td rowspan="2">ETTC</td></tr><tr><td>LLaMA</td><td>Pixtral</td><td>Gemma</td><td>Qwen-7B</td></tr><tr><td>MathVista</td><td>52.04</td><td>56.03</td><td>65.03</td><td>72.08</td><td>61.30</td><td>68.33</td><td>75.93</td></tr><tr><td>MathVision</td><td>23.41</td><td>25.20</td><td>31.84</td><td>30.18</td><td>27.66</td><td>32.05</td><td>35.57</td></tr><tr><td>TQA</td><td>70.41</td><td>77.34</td><td>78.86</td><td>78.50</td><td>76.28</td><td>83.65</td><td>83.90</td></tr><tr><td>ScienceQA</td><td>77.84</td><td>78.32</td><td>77.83</td><td>79.76</td><td>78.44</td><td>85.52</td><td>85.28</td></tr><tr><td>MMStar</td><td>46.09</td><td>50.35</td><td>53.40</td><td>56.77</td><td>51.65</td><td>59.27</td><td>60.07</td></tr><tr><td>MMMU</td><td>42.87</td><td>47.65</td><td>52.49</td><td>50.53</td><td>48.39</td><td>53.66</td><td>58.63</td></tr><tr><td>Average</td><td>52.11</td><td>55.82</td><td>59.91</td><td>61.30</td><td>57.29</td><td>63.75</td><td>66.56</td></tr></table>

In contrast, ETTC entirely bypasses this failure mode by selecting the single most confident prediction. Under the mild assumption that lower entropy correlates with higher accuracy, ETTC guarantees performance at least as good as the most accurate model, regardless of how many weaker models agree on a wrong answer. Since VLMs heavily share training data and architectures, making their predictions inherently dependent, ETTC offers a structurally safer and more principled aggregation strategy.

# 5.3. Empirical Verification

We now evaluate ETTC in practical multi-model ensemble settings. We structure our evaluation to answer three key questions: (1) Can ETTC effectively leverage diverse models of similar sizes across different families? (2) Does it remain effective when scaling models within the same architecture family? (3) How robust is the method to extreme capability gaps and modality shifts?

# 5.3.1. SIMILAR-SIZED MODELS FROM DIFFERENT FAMILIES

Motivation. We first evaluate whether ETTC can better leverage diversity among models of comparable scale but distinct architectural families. In this setting, models offer complementary strengths, but the variance in prediction quality makes standard voting noisy.

Setup. We select four models of similar scale (7B-12B): LLaMA-11B, Pixtral-12B, Gemma-12B, and Qwen-7B. These models produce predictions for each dataset, and we compare voting and ETTC on the exact same set of outputs. Notably, no single model consistently dominates across all tasks, and some models are clearly weaker on specific domains, adding noise to the ensemble.

Findings. In Tab. 1, ETTC outperforms voting on five of six datasets, yielding an average accuracy gain of +2.81% (66.56% vs. 63.75%). The largest improvements occur on tasks where model performance diverges significantly, such as MathVista and MathVision. In these cases, voting suffers from equal-weighting, allowing the weaker models to dilute the correct signal. In contrast, ETTC adaptively prioritizes high-confidence predictions, effectively aligning with the strongest model for each specific item, and often exceeding the best single model’s standalone performance.

Takeaway. When aggregating diverse but uneven models, ETTC offers a clear advantage over voting: it selectively filters noise from weaker models based on their own uncertainty, making it highly effective in heterogeneous settings.

# 5.3.2. SAME-FAMILY MODELS OF DIFFERENT SCALES

Motivation. We then examine whether ETTC remains effective when models share the exact same architecture and training data, but differ vastly in scale. While scaling laws introduce meaningful diversity in capability, the shared inductive biases create high prediction dependency, the exact scenario where our theory suggests voting will struggle.

Setup. We use four models from the Qwen family: 3B, 7B, 32B, and 72B. Each model produces predictions on all datasets, and we compare aggregation methods on their combined outputs.

Findings. As in Tab. 2, ETTC outperforms voting on all datasets, achieving an average gain of +2.84% (71.68% vs. 68.84%). While overall prediction correlation is higher than in the cross-family setting, the performance variance introduced by scale provides useful diversity. Specifically, smaller models occasionally make correct predictions with higher certainty than larger ones. ETTC successfully detects and leverages these instances, allowing smaller models to override the incorrect predictions of large models. Overall, ETTC consistently surpasses the accuracy of the strongest model (Qwen-72B), whereas voting frequently performs worse than the strongest model due to dilution.

Table 2. Comparison of ETTC and Voting in the multi-model ensemble setting using same-family models (Qwen) of increasing scale. ETTC consistently outperforms voting across all datasets, even under highly correlated predictions. Gains are especially pronounced when model accuracies increase with scale, demonstrating ETTC’s advantage in prioritizing stronger models within homogeneous ensembles. 

<table><tr><td rowspan="2">Accuracy (%)</td><td colspan="4">Models</td><td rowspan="2">Average</td><td rowspan="2">Voting</td><td rowspan="2">ETTC</td></tr><tr><td>Qwen-3B</td><td>Qwen-7B</td><td>Qwen-32B</td><td>Qwen-72B</td></tr><tr><td>MathVista</td><td>51.94</td><td>72.08</td><td>78.58</td><td>80.58</td><td>70.80</td><td>83.15</td><td>84.44</td></tr><tr><td>MathVision</td><td>22.27</td><td>30.18</td><td>38.80</td><td>42.89</td><td>33.53</td><td>41.32</td><td>44.84</td></tr><tr><td>TQA</td><td>60.85</td><td>78.50</td><td>83.06</td><td>84.52</td><td>76.73</td><td>84.90</td><td>86.70</td></tr><tr><td>ScienceQA</td><td>66.67</td><td>79.76</td><td>84.21</td><td>84.64</td><td>78.82</td><td>84.04</td><td>85.03</td></tr><tr><td>MMStar</td><td>41.22</td><td>56.77</td><td>56.34</td><td>62.56</td><td>54.22</td><td>61.00</td><td>63.73</td></tr><tr><td>MMMU</td><td>37.41</td><td>50.53</td><td>59.04</td><td>64.18</td><td>52.79</td><td>58.63</td><td>65.34</td></tr><tr><td>Average</td><td>46.73</td><td>61.30</td><td>66.67</td><td>69.90</td><td>61.15</td><td>68.84</td><td>71.68</td></tr></table>

Takeaway. Despite architectural homogeneity, ensembles of different-sized models still benefit immensely from confidence-based selection. ETTC avoids overcounting correlated errors and allows smaller models to meaningfully enhance larger ones, challenging the conventional wisdom that the largest model should dictate test-time performance.

# 5.3.3. ROBUSTNESS AND GENERALIZATION

To comprehensively validate the reliability of ETTC, we stress-test it by analyzing the specific risk of dilution in heterogeneous ensembles, extending our evaluation to text-only reasoning, and addressing miscalibration via supervision.

Robustness to Weak Learners. In our analysis of samefamily models, we combined models with vast capability gaps (e.g., 3B vs. 72B). While beneficial for coverage, such ensembles introduce the critical risk of “dilution”, where weaker models drag down the performance of stronger ones (Krogh & Vedelsby, 1994; Zhou et al., 2002; Chen et al., 2024b). To investigate whether ETTC can mitigate this risk, we conducted a fine-grained ablation on MathVista evaluating all possible subset combinations of the Qwen family (detailed settings and results in Tab. 7 of § C.3).

We find that voting is highly sensitive to this disparity: when a weak learner is combined with a strong expert, voting accuracy drops significantly, as the noise from the smaller model effectively drowns out the expert’s signal. In contrast, ETTC demonstrates remarkable robustness. In the same weak-strong pairings, ETTC not only avoids degradation but actually surpasses the standalone strong baseline. By relying on predictive entropy, ETTC effectively acts as a filter: it accepts the smaller model’s answer only when it is highly confident, while disregarding its uncertain errors. This confirms that ETTC enables “safe” ensembling, allowing the integration of disparate models without compromising the expert’s performance.

Generalization to Text-Only Reasoning. A key question is whether the benefits of ETTC are constrained by the perception bottlenecks inherent to VLMs, or if they reflect a broader property of reasoning models. To address this, we extended our evaluation to the text-only domain using “Thinking” LLMs (Qwen-3-Thinking) on standard reasoning benchmarks (ARC-Easy and MMLU-Pro; see § C.4).

Consistent with our VLM findings, ETTC outperforms voting across diverse ensemble configurations in this setting as well. Notably, in highly heterogeneous ensembles (e.g., combining 4B and 235B models), ETTC improves accuracy by nearly 5% over voting on MMLU-Pro. This result confirms that the correlation between predictive entropy and correctness is a fundamental property of reasoning models, independent of input modality, and that ETTC acts as an effective noise filter in pure language tasks.

Supervised Calibration. Finally, while ETTC is robust in zero-shot settings, it relies on the assumption that model confidence is a reliable proxy for correctness, an assumption that weakens when models are miscalibrated (i.e., “confidently wrong”). To mitigate this, we propose a Supervised ETTC variant that learns to weigh confidence signals based on empirically observed reliability. We train a lightweight logistic regressor using entropy-based features to predict the likelihood of correctness, allowing the system to dynamically downweight low-entropy predictions from unreliable

models (see § C.5).

Empirically, this variant consistently yields further improvements over unsupervised ETTC, particularly on challenging benchmarks like MathVision (Wang et al., 2024) where base model calibration is weaker. By effectively penalizing overconfidence, the supervised approach achieves the highest overall performance across all settings. This highlights that while entropy is a strong zero-shot signal, minimal supervision can significantly enhance reliability by adapting to specific model failure modes.

Overall Summary. Across both ensemble settings (diverse and redundant), ETTC consistently outperforms majority voting without requiring additional training or tuning. These results empirically validate our theoretical findings: when prediction dependency undermines voting, entropybased selection offers a structurally safer and more adaptive path to test-time improvement.

# 6. Related Work

TTC in LLMs. Chain-of-thought (CoT) prompting enables multi-step reasoning (Wei et al., 2022; Kojima et al., 2022), while self-consistency (majority voting) improves accuracy by sampling and aggregating diverse reasoning paths (Wang et al., 2023). Recent studies demonstrate that optimally allocating test-time compute (TTC) can sometimes rival scaling up model parameters (Snell et al., 2024). Advanced TTC methods have also explored selfcalibration (Huang et al., 2025) and entropy-minimization for test-time adaptation (Zhang et al., 2025).

Our Position: While inference-time scaling is wellestablished for text, its efficacy in the multimodal domain remains underexplored. Many recent adaptive methods also require training stages or model weight updates. In contrast, our work bridges the modality gap by systematically evaluating pure, inference-only TTC strategies, revealing how visual perception bottlenecks fundamentally alter the effectiveness of test-time scaling.

Enhancing VLM Reasoning. To improve VLM reasoning, researchers have adapted visual CoT prompts (Chen et al., 2024d) and developed test-time consistency objectives (Chou et al., 2025; Movva & Marupaka, 2025). Alternatively, post-training methods using Reinforcement Learning from Human Feedback have been proposed to align multimodal reasoning (Sun et al., 2024; Yu et al., 2024).

Our Position: Post-training approaches require substantial annotated data and compute budgets, while prompt-specific strategies often lack generalizability. Instead of retraining, we provide a lightweight, inference-only study across diverse, rigorous visual reasoning benchmarks (e.g., Math-Vista, MMMU). Crucially, we go beyond measuring performance to diagnose why standard aggregation strategies fail in VLMs through the lens of prediction dependency.

Ensembles, Uncertainty, and Correlation. Classic machine learning theory establishes that ensemble gains depend heavily on prediction diversity (i.e., low error correlation) among members (Tumer & Ghosh, 1996; Kuncheva & Whitaker, 2003). Deep ensembles are widely used to capture predictive uncertainty (Lakshminarayanan et al., 2017; Guo et al., 2017), and probabilistic aggregation often relies on confidence-weighted “opinion pooling” (Rufo & Perez ´ , 2012; Dietrich & List, 2017).

Our Position: While classic literature often pools full probability distributions or requires co-training diverse experts, our Entropy-based TTC method is designed for modern, off-the-shelf generative models. Rather than averaging distributions or relying on simple frequency (voting), we use per-item predictive entropy as a zero-shot filter to dynamically route trust to the most reliable model, enabling smaller models to safely augment larger ones without dilution.

# 7. Conclusion

We presented a systematic investigation into the transferability of TTC strategies from LLMs to VLMs. Our analysis identifies a critical bottleneck: standard aggregation methods like majority voting are fundamentally limited by the high statistical dependency of VLM predictions. We theoretically and empirically demonstrate that without diversity, voting offers diminishing returns, and in heterogeneous ensembles, it succumbs to noise from weaker models. To overcome these limitations, we proposed Entropy-based Test-Time Compute (ETTC), a method that prioritizes prediction confidence over frequency. ETTC proves to be a robust strategy for leveraging the diversity of multi-model ensembles, enabling significantly smaller models to synergistically enhance larger ones without the risk of dilution. Furthermore, we showed that this confidence-correctness correlation extends beyond vision, improving reasoning in “Thinking” LLMs as well. Ultimately, our work suggests that the future of efficient test-time scaling lies not just in generating more samples, but in intelligently selecting the most reliable signals from diverse model ecosystems.

# Acknowledgment

We thank the reviewers for their constructive feedback. We also thank Jiaoda Li, Yu Fan, Jingwei Ni, and Chenxi Pang for their valuable input during the early stages of this work. Yifan Hou is supported by the Swiss Data Science Center PhD Grant (P22-05).

# Impact Statement

This work aims to improve the reliability and accuracy of Vision-Language Models (VLMs) through inference-time strategies. The impact of our proposed method, Entropybased Test-Time Compute (ETTC), can be analyzed through three primary lenses: reliability, computational efficiency, and potential risks.

Enhancing Reliability in Visual Reasoning. By leveraging predictive entropy to filter out uncertain predictions, our method significantly improves the accuracy of VLMs in complex domains, such as mathematics and scientific diagram interpretation. This advancement is crucial for deploying VLMs in high-stakes applications (e.g., educational tutoring systems or scientific data analysis) where hallucinated or inconsistent answers can be detrimental. By prioritizing high-confidence predictions, our approach helps mitigate the “stochastic parrot” phenomenon often observed in multimodal generation, anchoring outputs to more deliberate reasoning paths.

Democratization and Efficient Utilization. A key finding of our work is that smaller, less capable models (e.g., 3B or 7B parameters) can synergistically enhance the performance of significantly larger models (e.g., 72B) within an ensemble. This has positive implications for the democratization of AI. It suggests that users or institutions with limited computational resources can still contribute meaningfully to ensemble systems by deploying smaller models. Furthermore, this “small-helps-large” dynamic offers a pathway to improve system performance without solely relying on training increasingly massive, energy-intensive models, potentially extending the useful lifespan of existing opensource weights.

Inference Cost and Environmental Impact. While our method avoids the massive carbon and financial costs of training new models, Test-Time Compute (TTC) strategies inherently increase inference costs compared to standard greedy decoding (due to multiple sampling rounds or ensembling). This leads to higher energy consumption per query. However, the ETTC selection mechanism itself introduces negligible computational overhead beyond the initial inference. Moreover, our results suggest that querying a heterogeneous ensemble (e.g., one large model paired with several small, cheap models) using ETTC can yield performance comparable to much larger, more expensive setups. This potentially offers a more favorable trade-off between accuracy and total energy expenditure.

Risks of Over-Reliance on Confidence. Our method relies on the assumption that lower entropy (higher confidence) correlates with correctness. While we empirically validate this trend, there is a risk that models may be “confidently wrong,” particularly in out-of-distribution scenarios or if the base models are poorly calibrated. If deployed without safeguards, this could lead users to place unwarranted trust in incorrect outputs simply because the system assigned them a high confidence score. To mitigate this, we emphasize that ETTC should be used in conjunction with base models that have undergone rigorous safety alignment and calibration, or paired with supervised variants like the one explored in this work.

# References

Agrawal, P., Antoniak, S., Hanna, E. B., Bout, B., Chaplot, D. S., Chudnovsky, J., Costa, D., Monicault, B. D., Garg, S., Gervet, T., Ghosh, S., Heliou, A., Jacob, P., ´ Jiang, A. Q., Khandelwal, K., Lacroix, T., Lample, G., de Las Casas, D., Lavril, T., Scao, T. L., Lo, A., Marshall, W., Martin, L., Mensch, A., Muddireddy, P., Nemychnikova, V., Pellat, M., von Platen, P., Raghuraman, N., Roziere, B., Sablayrolles, A., Saulnier, L., Sauvestre,\` R., Shang, W., Soletskyi, R., Stewart, L., Stock, P., Studnia, J., Subramanian, S., Vaze, S., Wang, T., and Yang, S. Pixtral 12b. CoRR, abs/2410.07073, 2024. doi: 10.48550/ARXIV.2410.07073. URL https: //doi.org/10.48550/arXiv.2410.07073.   
Al-Tahan, H., Garrido, Q., Balestriero, R., Bouchacourt, D., Hazirbas, C., and Ibrahim, M. Unibench: Visual reasoning requires rethinking vision-language beyond scaling, 2024. URL https://arxiv.org/abs/24 08.04810.   
Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. Qwen2.5-vl technical report. CoRR, abs/2502.13923, 2025. doi: 10.48550/ARXIV.2 502.13923. URL https://doi.org/10.48550 /arXiv.2502.13923.   
Bhattacharyya, A., Panchal, S., Lee, M., Pourreza, R., Madan, P., and Memisevic, R. Look, remember and reason: Visual reasoning with grounded rationales. CoRR, abs/2306.17778, 2023. doi: 10.48550/ARXIV.2306.17 778. URL https://doi.org/10.48550/arXiv .2306.17778.   
Chang, E. Y., Tong, Y., Niu, M., Neubig, G., and Yue, X. Demystifying long chain-of-thought reasoning in llms. CoRR, abs/2502.03373, 2025. doi: 10.48550/ARXIV.2 502.03373. URL https://doi.org/10.48550 /arXiv.2502.03373.   
Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z.,

Duan, H., Wang, J., Qiao, Y., Lin, D., and Zhao, F. Are we on the right way for evaluating large vision-language models? In Globersons, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J. M., and Zhang, C. (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024a. URL http://pape rs.nips.cc/paper\_files/paper/2024/ha sh/2f8ee6a3d766b426d2618e555b5aeb39-A bstract-Conference.html.   
Chen, L., Zaharia, M., and Zou, J. Frugalgpt: How to use large language models while reducing cost and improving performance. Trans. Mach. Learn. Res., 2024, 2024b. URL https://openreview.net/forum?id= cSimKw5p6R.   
Chen, W., Wang, W., Chu, Z., Ren, K., Zheng, Z., and Lu, Z. Self-para-consistency: Improving reasoning tasks at low cost for large language models. In Ku, L., Martins, A., and Srikumar, V. (eds.), Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pp. 14162– 14167. Association for Computational Linguistics, 2024c. doi: 10.18653/V1/2024.FINDINGS-ACL.842. URL https://doi.org/10.18653/v1/2024.fin dings-acl.842.   
Chen, Z., Zhou, Q., Shen, Y., Hong, Y., Sun, Z., Gutfreund, D., and Gan, C. Visual chain-of-thought prompting for knowledge-based visual reasoning. In Wooldridge, M. J., Dy, J. G., and Natarajan, S. (eds.), Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, pp. 1254–1262. AAAI Press, 2024d. doi: 10.1609/AAAI.V 38I2.27888. URL https://doi.org/10.1609/ aaai.v38i2.27888.   
Chou, S., Chandhok, S., Little, J. J., and Sigal, L. Testtime consistency in vision language models. CoRR, abs/2506.22395, 2025. doi: 10.48550/ARXIV.2506. 22395. URL https://doi.org/10.48550/arX iv.2506.22395.   
Dietrich, F. and List, C. Probabilistic opinion pooling generalized. part two: the premise-based approach. Soc. Choice Welf., 48(4):787–814, 2017. doi: 10.1007/S00355-017-1 035-Y. URL https://doi.org/10.1007/s003 55-017-1035-y.   
Fu, Y., Peng, H., Sabharwal, A., Clark, P., and Khot, T. Complexity-based prompting for multi-step reasoning.

In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openre view.net/forum?id=yf1icZHC-l9.   
Gemini Team, G. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. URL https: //arxiv.org/abs/2507.06261.   
Gemma Team, G. D. Gemma 3 technical report. CoRR, abs/2503.19786, 2025. doi: 10.48550/ARXIV.2503.19 786. URL https://doi.org/10.48550/arXiv .2503.19786.   
Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. On calibration of modern neural networks. In Precup, D. and Teh, Y. W. (eds.), Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pp. 1321–1330. PMLR, 2017. URL http://proceedings.mlr. press/v70/guo17a.html.   
Huang, C., Huang, L., Leng, J., Liu, J., and Huang, J. Efficient test-time scaling via self-calibration. CoRR, abs/2503.00031, 2025. doi: 10.48550/ARXIV.2503.00 031. URL https://doi.org/10.48550/arXiv .2503.00031.   
Jiang, D., Zhang, R., Guo, Z., Li, Y., Qi, Y., Chen, X., Wang, L., Jin, J., Guo, C., Yan, S., Zhang, B., Fu, C., Gao, P., and Li, H. Mme-cot: Benchmarking chain-ofthought in large multimodal models for reasoning quality, robustness, and efficiency. CoRR, abs/2502.09621, 2025a. doi: 10.48550/ARXIV.2502.09621. URL https: //doi.org/10.48550/arXiv.2502.09621.   
Jiang, G., Liu, Y., Li, Z., Bi, W., Zhang, F., Song, L., Wei, Y., and Lian, D. What makes a good reasoning chain? uncovering structural patterns in long chain-of-thought reasoning. In Christodoulopoulos, C., Chakraborty, T., Rose, C., and Peng, V. (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, EMNLP 2025, Suzhou, China, November 4- 9, 2025, pp. 6490–6514. Association for Computational Linguistics, 2025b. doi: 10.18653/V1/2025.EMNLP-M AIN.329. URL https://doi.org/10.18653/v 1/2025.emnlp-main.329.   
Jin, M., Yu, Q., Shu, D., Zhao, H., Hua, W., Meng, Y., Zhang, Y., and Du, M. The impact of reasoning step length on large language models. In Ku, L., Martins, A., and Srikumar, V. (eds.), Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pp. 1830– 1842. Association for Computational Linguistics, 2024.

doi: 10.18653/V1/2024.FINDINGS-ACL.108. URLhttps://doi.org/10.18653/v1/2024.findings-acl.108.  
Kim, D., Kim, S., and Kwak, N. Textbook question answering with multi-modal context graph understanding and self-supervised open-set comprehension. In Korhonen, A., Traum, D. R., and Marquez, L. (eds.), \` Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pp. 3568–3584. Association for Computational Linguistics, 2019. doi: 10.18653/V1/P19-1347. URL https: //doi.org/10.18653/v1/p19-1347.   
Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/p aper\_files/paper/2022/hash/8bb0d291a cd4acf06ef112099c16f326-Abstract-Con ference.html.   
Krogh, A. and Vedelsby, J. Neural network ensembles, cross validation, and active learning. In Tesauro, G., Touretzky, D. S., and Leen, T. K. (eds.), Advances in Neural Information Processing Systems 7, [NIPS Conference, Denver, Colorado, USA, 1994], pp. 231–238. MIT Press, 1994. URL https://proceedings.neurips.cc/p aper\_files/paper/1994/hash/b8c37e33d efde51cf91e1e03e51657da-Abstract.ht ml.   
Kuncheva, L. I. and Whitaker, C. J. Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy. Mach. Learn., 51(2):181–207, 2003. doi: 10.1023/A:1022859003006. URL https://doi. org/10.1023/A:1022859003006.   
Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles. In Guyon, I., von Luxburg, U., Bengio, S., Wallach, H. M., Fergus, R., Vishwanathan, S. V. N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pp. 6402–6413, 2017. URL https://proceedings.neurips.cc/p aper/2017/hash/9ef2ed4b7fd2c810847ff a5fa85bce38-Abstract.html.   
Li, M., Su, N., Qu, F., Zhong, Z., Chen, Z., Li, Y., Tu, Z., and Li, X. VISTA: enhancing vision-text alignment in

mllms via cross-modal mutual information maximization. CoRR, abs/2505.10917, 2025. doi: 10.48550/ARXIV.2 505.10917. URL https://doi.org/10.48550 /arXiv.2505.10917.   
Lippmann, P. and Yang, J. Style over substance: Distilled language models reason via stylistic replication. CoRR, abs/2504.01738, 2025. doi: 10.48550/ARXIV.2504.01 738. URL https://doi.org/10.48550/arXiv .2504.01738.   
Llama Team, A. . M. The llama 3 herd of models. CoRR, abs/2407.21783, 2024. doi: 10.48550/ARXIV.2407.21 783. URL https://doi.org/10.48550/arXiv .2407.21783.   
Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022.   
Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K., Galley, M., and Gao, J. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=KUNz EQMWU7.   
Mao, Z., Bisliouk, A., Nama, R. R., and Ruchkin, I. Temporalizing confidence: Evaluation of chain-ofthought reasoning with signal temporal logic. CoRR, abs/2506.08243, 2025. doi: 10.48550/ARXIV.2506.08 243. URL https://doi.org/10.48550/arXiv .2506.08243.   
Movva, P. and Marupaka, N. H. Enhancing scientific visual question answering through multimodal reasoning and ensemble modeling. In Ghosal, T., Mayr, P., Singh, A., Naik, A., Rehm, G., Freitag, D., Li, D., Schimmler, S., and De Waard, A. (eds.), Proceedings of the Fifth Workshop on Scholarly Document Processing (SDP 2025), pp. 252–262, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-265- 7. doi: 10.18653/v1/2025.sdp-1.23. URL https: //aclanthology.org/2025.sdp-1.23/.   
OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023. doi: 10.48550/ARXIV.2303.08774. URL https: //doi.org/10.48550/arXiv.2303.08774.   
Rufo, M. and Perez, C. Log-linear pool to combine prior ´ distributions: A suggestion for a calibration-based approach. Bayesian Analysis, 7:1–28, 06 2012. doi: 10.1214/12-BA714.

Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling LLM test-time compute optimally can be more effective than scaling model parameters. CoRR, abs/2408.03314, 2024. doi: 10.48550/ARXIV.2408.03314. URL https: //doi.org/10.48550/arXiv.2408.03314.   
Sun, Z., Shen, S., Cao, S., Liu, H., Li, C., Shen, Y., Gan, C., Gui, L., Wang, Y.-X., Yang, Y., Keutzer, K., and Darrell, T. Aligning large multimodal models with factually augmented RLHF. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Findings of the Association for Computational Linguistics: ACL 2024, pp. 13088–13110, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.775. URL https://aclanthology.org/2024.findin gs-acl.775/.   
Sutskever, I., Vinyals, O., and Le, Q. V. Sequence to sequence learning with neural networks. In Ghahramani, Z., Welling, M., Cortes, C., Lawrence, N. D., and Weinberger, K. Q. (eds.), Advances in Neural Information Processing Systems 27: Annual Conference on Neural Information Processing Systems 2014, December 8-13 2014, Montreal, Quebec, Canada, pp. 3104–3112, 2014. URL https://proceedings.neurips.cc/p aper/2014/hash/a14ac55a4f27472c5d894 ec1c3c743d2-Abstract.html.   
Tumer, K. and Ghosh, J. Error correlation and error reduction in ensemble classifiers. Connect. Sci., 8(3):385–404, 1996. doi: 10.1080/095400996116839. URL https: //doi.org/10.1080/095400996116839.   
Wang, J., Kang, Z., Wang, H., Jiang, H., Li, J., Wu, B., Wang, Y., Ran, J., Liang, X., Feng, C., and Xiao, J. VGR: visual grounded reasoning. CoRR, abs/2506.11991, 2025. doi: 10.48550/ARXIV.2506.11991. URL https: //doi.org/10.48550/arXiv.2506.11991.   
Wang, K., Pan, J., Shi, W., Lu, Z., Ren, H., Zhou, A., Zhan, M., and Li, H. Measuring multimodal mathematical reasoning with math-vision dataset. In Globersons, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J. M., and Zhang, C. (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper\_f iles/paper/2024/hash/ad0edc7d5fa1a78 3f063646968b7315b-Abstract-Datasets\_ and\_Benchmarks\_Track.html.   
Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Selfconsistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali,

Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=1PL1 NIMMrw.   
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E. H., Le, Q. V., and Zhou, D. Chain-ofthought prompting elicits reasoning in large language models. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.n ips.cc/paper\_files/paper/2022/hash/9 d5609613524ecf4f15af0f7b31abca4-Abstr act-Conference.html.   
Yan, Q., Fan, Y., Li, H., Jiang, S., Zhao, Y., Guan, X., Kuo, C., and Wang, X. E. Multimodal inconsistency reasoning (MMIR): A new benchmark for multimodal reasoning models. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Findings of the Association for Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pp. 18829–18845. Association for Computational Linguistics, 2025. URL https:// aclanthology.org/2025.findings-acl.9 64/.   
Yu, T., Yao, Y., Zhang, H., He, T., Han, Y., Cui, G., Hu, J., Liu, Z., Zheng, H., and Sun, M. RLHF-V: towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 13807–13816. IEEE, 2024. doi: 10.1109/CVPR52733.2024.01310. URL https://doi.org/10.1109/CVPR52733. 2024.01310.   
Yue, X., Ni, Y., Zheng, T., Zhang, K., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., Wei, C., Yu, B., Yuan, R., Sun, R., Yin, M., Zheng, B., Yang, Z., Liu, Y., Huang, W., Sun, H., Su, Y., and Chen, W. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 9556–9567. IEEE, 2024. doi: 10.1109/CVPR52733.20 24.00913. URL https://doi.org/10.1109/CV PR52733.2024.00913.   
Zhang, Q., Bian, Y., Kong, X., Zhao, P., and Zhang, C. COME: test-time adaption by conservatively minimizing entropy. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https:// openreview.net/forum?id=506BjJ1ziZ.

Zhou, Z., Wu, J., and Tang, W. Ensembling neural networks: Many could be better than all. Artif. Intell., 137(1-2):239– 263, 2002. doi: 10.1016/S0004-3702(02)00190-X. URL https://doi.org/10.1016/S0004-3702(02 )00190-X.

# A. Theoretical Proofs

# A.1. Proof of Theorem 1

Proof. We provide a theoretical justification for the claim that the improvement from majority voting decreases monotonically with statistical dependency among model predictions. We proceed by defining a simple probabilistic coupling model that controls prediction dependency, and then analyze how the expected voting accuracy varies with this dependency level.

# A.1.1. COUPLING MODEL: COPY-OR-INDEPENDENT SAMPLING

We assume all U predictions $\{ X _ { u } \} _ { u = 1 } ^ { U }$ are drawn from a shared coupling mechanism that depends on a parameter $\lambda \in [ 0 , 1 ]$ : With probability λ, all predictions are identical copies of a single sample X. With probability 1 − λ, predictions are sampled independently from a shared categorical distribution $\pi = ( \pi _ { 1 } , \ldots , \pi _ { K } )$ over K options. Formally, for any pair $( X _ { u } , X _ { v } )$ ,

$$
\left(X _ {u}, X _ {v}\right) \sim \left\{ \begin{array}{l l} (X, X), & \text { with   probability } \lambda \\ \left(X ^ {\prime}, X ^ {\prime \prime}\right), & X ^ {\prime}, X ^ {\prime \prime} \stackrel {\text { i.i.d. }} {\sim} \pi , \end{array} \right. \text { with   probability } 1 - \lambda \tag {1}
$$

This ensures uniform pairwise dependency, controlled by λ.

# A.1.2. LEMMA: BEHAVIOR OF DEPENDENCY METRICS UNDER COUPLING

We now show that both statistical dependency metrics used in our main theorem, normalized mutual information and correctness correlation, are monotonic in λ under this coupling.

(a) Normalized mutual information. Let $X , X ^ { \prime }$ be two predictions drawn according to the coupling in (1). Their joint distribution is

$$
P _ {\lambda} (i, j) = \lambda \cdot \pi_ {i} \cdot \delta_ {i j} + (1 - \lambda) \cdot \pi_ {i} \cdot \pi_ {j},
$$

where $\delta _ { i j }$ is the Kronecker delta. The marginal distributions remain unchanged as π.

Since mutual information $I ( X ; X ^ { \prime } )$ increases with λ (via the convexity of KL divergence), and the marginals are fixed, the normalized mutual information $\operatorname { N M I } ( X ; X ^ { \prime } )$ is also non-decreasing in λ:

$$
\operatorname{NMI} (X; X ^ {\prime}) = \frac {I (X ; X ^ {\prime})}{H (X)} \uparrow \text {   as   } \lambda \uparrow .
$$

Hence, the average pairwise NMI NMI is also monotonic in λ.

(b) Correctness correlation. Let $Z _ { u } = \mathbb { I } \{ X _ { u } = Y \}$ , where Y is the correct option. Denote single-trial accuracy as $p = \mathbb { P } ( X _ { u } = Y )$ . Then for any pair $( Z _ { u } , Z _ { v } )$ : Under the “copy” case: $\mathbb { P } ( Z _ { u } = Z _ { v } = 1 ) = p$ . Under the “independent” case: $\mathbb { P } ( Z _ { u } = Z _ { v } = 1 ) = p ^ { 2 }$ .

Therefore, the covariance is

$$
\operatorname{Cov} (Z _ {u}, Z _ {v}) = \mathbb {E} [ Z _ {u} Z _ {v} ] - p ^ {2} = \lambda (p - p ^ {2}) = \lambda p (1 - p),
$$

and the correlation is

$$
\rho (Z _ {u}, Z _ {v}) = \frac {\mathrm{Cov} (Z _ {u} , Z _ {v})}{p (1 - p)} = \lambda . \tag {2}
$$

Thus, the average correlation ${ \overline { { \rho } } } = \lambda$ .

# A.1.3. MAIN PROOF: MONOTONICITY OF VOTING IMPROVEMENT

Let $A _ { \mathrm { M V } } ( U ; \lambda )$ be the expected voting accuracy under dependency level λ, and let $A _ { \mathrm { s i n g l e } } = p$ be the single-trial accuracy.

We decompose voting accuracy by conditioning on the latent sampling regime:

$$
A _ {\mathrm{MV}} (U; \lambda) = \lambda \cdot A _ {\mathrm{MV}} (U; \text {copy}) + (1 - \lambda) \cdot A _ {\mathrm{MV}} (U; \mathrm{iid}). \tag {3}
$$

In the “copy” case, all predictions are identical, so voting is equivalent to a single trial: $A _ { \mathrm { M V } } ( U ; \mathrm { c o p y } ) = p .$ . In the “iid” case, predictions are independent, and voting aggregates U samples from π; here, accuracy improves with U , approaching 1 as $\begin{array} { r } { U \to \infty \mathrm { i f } p > \frac { 1 } { K } } \end{array}$ . Thus:

$$
A _ {\mathrm{MV}} (U; \lambda) = \lambda p + (1 - \lambda) A _ {\mathrm{MV}} (U; 0), \tag {4}
$$

$$
\Delta A _ {\mathrm{MV}} (U; \lambda) := A _ {\mathrm{MV}} (U; \lambda) - p = (1 - \lambda) (A _ {\mathrm{MV}} (U; 0) - p). \tag {5}
$$

The improvement $\Delta A _ { \mathrm { M V } } ( U ; \lambda )$ is thus a linear function decreasing in λ, and since λ = ρ (from (2)) and NMI increases with λ, voting improvement is monotonically decreasing in both.

# A.1.4. COROLLARY (EXTREMES)

I $\mathtt { f } \lambda = 1 ( \mathrm { i . e . , } \overline { { \rho } } = 1 \mathrm { o r } \overline { { \mathrm { N M I } } } = 1 )$ , then all predictions are identical and voting offers no improvement:

$$
\Delta A _ {\mathrm{MV}} (U) = 0.
$$

If λ = 0 (i.e., predictions are independent) and $\textstyle p > { \frac { 1 } { K } }$ , then:

$$
A _ {\mathrm{MV}} (U) \to 1 \quad \text { as } \quad U \to \infty .
$$

# A.1.5. DISCUSSION

This result formalizes an intuitive principle: confidence-based aggregation (e.g., voting) helps only when predictions are sufficiently diverse. High dependency, measured either via correctness correlation or mutual information, reduces the effective information gain from additional samples. Empirical results confirm this trend across VLMs and datasets: voting yields larger gains when dependency is low.

# A.2. Proof of Theorem 2

Proof. Setup. Fix a K-way classification item with true label Y . Let $u ^ { \star } : = \arg \operatorname* { m a x } _ { u } p _ { u } ( Y )$ be the best model and define $c ^ { \star } : = \operatorname* { P r } ( \hat { y } _ { u ^ { \star } } = Y )$ . Let $\boldsymbol { B } = \{ \boldsymbol { u } \neq \boldsymbol { u } ^ { \star } \}$ be the set of non-best models, with $| B | \geq 2$ .

Coupling among non-best models. Introduce a latent variable L ∈ {copy, iid}: - With probability $\lambda , L = \mathrm { c o p y }$ and all non-best models predict a shared label W ; define ${ \bar { c } } : = \operatorname* { P r } ( W = Y ) $ . - With probability 1 − λ, L = iid and the non-best predictions are drawn independently.

Step 1: Accuracy of ETTC. Under Assumption 1, ETTC selects $\hat { y } _ { u ^ { \star } }$ , so:

$$
A _ {\min H} = \operatorname * {P r} (\hat {y} _ {u ^ {*}} = Y) = c ^ {\star}. \tag {6}
$$

Step 2: Accuracy of Voting. By law of total probability:

$$
A _ {\mathrm{MV}} (\lambda) = \lambda \operatorname * {P r} (\widehat {Y} _ {\mathrm{MV}} = Y \mid L = \text { copy }) + (1 - \lambda) A _ {\mathrm{MV}} (0). \tag {7}
$$

Under L = copy, all non-best models predict W , forming a majority:

$$
\operatorname * {P r} (\widehat {Y} _ {\mathrm{MV}} = Y \mid L = \text { copy }) = \operatorname * {P r} (W = Y) = \bar {c}. \tag {8}
$$

Plugging into (7), we recover:

$$
A _ {\mathrm{MV}} (\lambda) = \lambda \bar {c} + (1 - \lambda) A _ {\mathrm{MV}} (0). \tag {9}
$$

Step 3: Difference and monotonicity. Subtracting (9) from (6):

$$
A _ {\min H} - A _ {\mathrm{MV}} (\lambda) = \lambda (c ^ {\star} - \bar {c}) + (1 - \lambda) (c ^ {\star} - A _ {\mathrm{MV}} (0)). \tag {10}
$$

This gap is nondecreasing in λ:

$$
\frac {d}{d \lambda} (A _ {\min H} - A _ {\mathrm{MV}} (\lambda)) = A _ {\mathrm{MV}} (0) - \bar {c} \geq 0.
$$

Step 4: Dominance threshold. Let

$$
\lambda^ {\star} = \max \left\{0, \frac {A _ {\mathrm{MV}} (0) - c ^ {\star}}{A _ {\mathrm{MV}} (0) - \bar {c}} \right\}.
$$

Then for all $\lambda \geq \lambda ^ { \star }$ , ETTC outperforms voting; if $\bar { c } < c ^ { \star }$ and $\lambda > \lambda ^ { \star }$ , the gap is strict.

Remarks. - Since u⋆ is the best model, typically $\bar { c } < c ^ { \star }$ unless all models perform equally well. - If $A _ { \mathrm { M V } } ( 0 ) \le c ^ { \star }$ , then $\lambda ^ { \star } = 0 \colon$ ETTC dominates voting at all dependency levels. - Under the copy-or-independent model, the average correctness correlation among non-best models equals λ (see § A.1), providing a direct link between dependency and the TTC advantage.

# B. Experiment Settings

# B.1. Dataset

Table 3. Dataset statistics and characteristics used in our evaluation. Each dataset is categorized by its domain (Math, Diagram, or General), the evaluation split used (e.g., test or validation), the number of multiple-choice questions (Size), and the number of answer options per question (Option Num.). 

<table><tr><td>Dataset</td><td>Domain</td><td>Type</td><td>Size</td><td>Option Num.</td></tr><tr><td>MathVista</td><td>Math</td><td>testmini</td><td>540</td><td>2–8</td></tr><tr><td>MathVision</td><td>Math</td><td>test</td><td>1,532</td><td>5</td></tr><tr><td>TQA</td><td>Diagram</td><td>test</td><td>3,285</td><td>4</td></tr><tr><td>ScienceQA</td><td>Diagram</td><td>test</td><td>2,017</td><td>2–5</td></tr><tr><td>MMStar</td><td>General</td><td>val</td><td>1,500</td><td>4</td></tr><tr><td>MMMU</td><td>General</td><td>val</td><td>805</td><td>2–9</td></tr></table>

We evaluate our methods on six diverse multi-choice benchmarks spanning three domains: mathematical reasoning (MathVista, MathVision), diagram-based QA (TQA, ScienceQA), and general visual understanding (MMStar, MMMU). Tab. 3 summarizes key statistics, including dataset size, official split used, and number of answer options. Note that some datasets contain variable numbers of options (e.g., 2 - 9 in MMMU), which adds to the challenge and makes majority voting less stable. This diversity ensures our evaluation reflects a wide range of real-world reasoning settings.

# B.2. Prompt

To ensure consistency and minimize response variance across models, we standardize the prompting format in all benchmark evaluations. Specifically, we use a direct QA prompt without explanation, and a chain-of-thought (CoT) style prompt when evaluating reasoning performance or conducting consistency analysis. Below, we show two representative examples for comparison. The image and question are kept identical, while only the prompt template changes.

# B.3. Baselines

To better assess the reliability of CoT responses, we include several shallow feature-based baselines. These models predict the correctness of a response using surface-level properties, without access to model internals or gradient signals.

Pivot words. Pivot words are rhetorical expressions that signal shifts in reasoning, such as realization, verification, or synthesis. Prior work (Lippmann & Yang, 2025) suggests that the presence of such expressions often correlates with more deliberate and structured reasoning. We use a curated list of phrases categorized by rhetorical function, shown in Tab. 4. These are used as features for correctness prediction (e.g., counting their presence in CoTs).

Vague words. Vague expressions are often used to hedge or express uncertainty, and may correlate with lower confidence or correctness in model reasoning. We group these into two categories, uncertainty and hedging—based on their rhetorical function. See Tab. 5.

Pipeline for Benchmark Evaluation   
![](images/7fd8415743789d7d28d9ec350d32b0c98e1084606a9053a44ec7253b3de21119.jpg)

<details>
<summary>bar</summary>

title
| xaxis_label | yaxis_label |
| :--- | :--- |
| Dark Salmon | 52 |
| Periwinkle | 62 |
| Gray | 95 |
| Tomato | 98 |
</details>

# Prompt:

Question: Is Periwinkle the maximum? Options: ’yes’, ’no’. ONLY SIMPLY choose the correct option in a single sentence or a single word. No preamble, no explanation.

# Output:

No.

Figure 4. Example of a direct QA prompt used for evaluating model predictions without reasoning.

Feature-All. We also define a feature set that combines lexical and stylistic signals for each CoT response. Specifically, we consider four interpretable features: response length (token count), lexical diversity (unique token count), number of pivot words, and number of vague words. See Tab. 6 for detailed definitions. For prediction, we compute the sum of these feature values for each example, encouraging longer, more expressive, and more structured responses, while penalizing vague expressions. The model response with the highest total score is selected as the final prediction.

# C. Supplementary Results

# C.1. Voting Improvement vs. NMI and Correlation

While the overall trends in Figs. 6 and 7 are consistent with our theoretical expectations, MathVision stands out as an exception. Specifically, we observe weaker or even inverted correlation between prediction dependency and voting improvement on this dataset. A likely explanation is that MathVision poses significantly higher difficulty for current VLMs, its average accuracy across models is around 30%, which suggests that models are often uncertain or guessing. In such low-performance regimes, prediction behaviors may become erratic or overly stochastic, reducing the reliability of entropy, correlation, and voting-based signals. As a result, the dependency measures may not reflect meaningful error structure, making voting behavior less predictable.

Pipeline for Benchmark Evaluation   
![](images/713a2fa8c10d3dbec8b62fa875fc5abb886069e9b4f43a7d014fe1080af095b3.jpg)

<details>
<summary>bar</summary>

title
| xaxis_label | yaxis_label |
| :--- | :--- |
| Dark Salmon | 52 |
| Periwinkle | 62 |
| Gray | 95 |
| Tomato | 98 |
</details>

# Prompt:

Question: Is Periwinkle the maximum? Options: ’yes’, ’no’. Among the given options, the answer is: (Let’s think step by step, and give the answer at the end of your thought with \*\*Answer:\*\*).

# Output:

Looking at the bar plot, I can see that the y-values for the colors are: Dark Salmon (48), Periwinkle (60), Gray (92), and Tomato (96). \n\nComparing these values, I can see that the highest value is 96, which corresponds to the color Tomato. \n\nTherefore, the maximum is Tomato.\n\n\*\*Answer:\*\* no

Figure 5. Example of a chain-of-thought (CoT) prompt used to elicit intermediate reasoning steps. This format is used when analyzing consistency or measuring correctness under step-by-step reasoning.

# C.2. Empirical Evidence to Support Assumption

Fig. 8 shows the relationship between normalized entropy ${ \widetilde { H } } _ { \imath }$ and accuracy across multiple models on six benchmarks. We observe a strong inverse correlation between entropy and accuracy, consistent with our Entropy-Accuracy Monotonicity assumption (Assumption 1). Higher-performing models generally exhibit lower entropy, indicating more confident and reliable predictions.

# C.3. Ensemble Robustness Analysis

Setup. To investigate whether including weaker, smaller models in an ensemble degrades performance, we conducted a comprehensive ablation study using the Qwen-2.5-VL family (3B, 7B, 32B, 72B) on the MathVista benchmark. We evaluated all possible pairwise and triplet combinations to simulate diverse ensemble qualities (Tab. 7).

Table 4. Pivot phrases categorized by reasoning function. 

<table><tr><td>Reasoning Type</td><td>Example Phrases</td></tr><tr><td>Realization</td><td>“wait”, “oh”, “actually”, “I missed something”</td></tr><tr><td>Verification</td><td>“let me doublecheck”, “to verify”, “checking again”</td></tr><tr><td>Exploration</td><td>“what if”, “another way to look at this”, “alternatively”</td></tr><tr><td>Integration</td><td>“now I see how”, “this connects back to”, “putting this together”</td></tr></table>

Table 5. Vague expressions used in model reasoning, grouped by rhetorical effect. 

<table><tr><td>Reasoning Type</td><td>Example Phrases</td></tr><tr><td>Uncertainty</td><td>“maybe””, “possibly”, “perhaps”, “probably”, “might be”, “could be”, “it seems”</td></tr><tr><td>Hedging</td><td>“somewhat”, “rather”, “kind of”, “sort of”, “generally”, “typically”</td></tr></table>

Table 6. Overview of lexical and stylistic features used for CoT-based prediction. 

<table><tr><td>Feature</td><td>Modeling Method</td></tr><tr><td>Token Number</td><td>Measures the number of tokens in the CoT response. Longer responses may indicate more reasoning steps, though excessive length may signal loops or noise. We vectorize it as 1/Token Number.</td></tr><tr><td>Lexical Diversity</td><td>Captures vocabulary richness by counting the number of unique tokens. Low diversity often suggests repetition. We vectorize it as 1/Vocabulary Size.</td></tr><tr><td>Pivot Word Number</td><td>Counts the number of pivot expressions from Tab. 4, indicating structured reasoning or correction. We vectorize it as 1/Pivot Word Number.</td></tr><tr><td>Vague Word Number</td><td>Counts the number of vague phrases from Tab. 5, which may reflect uncertainty or low confidence. We vectorize it as 1 - 1/Vague Word Number.</td></tr></table>

Findings. The results highlight a critical failure mode of majority voting in heterogeneous ensembles. For instance, when combining the weakest model (3B, ∼52% accuracy) with the strongest (72B, ∼80% accuracy), voting performance drops significantly to 73.15%, effectively dragging the strong model down toward the average. This confirms that voting is vulnerable when the ensemble contains models with large capability gaps.

In contrast, ETTC demonstrates remarkable robustness. In the same 3B+72B setting, ETTC achieves 84.81%, not only avoiding the degradation seen in voting but actually surpassing the standalone performance of the 72B model by over 4%. This trend holds across triplet configurations as well; for example, in the {3B, 7B, 72B} ensemble, voting achieves 82.22% while ETTC reaches 84.44%.

Takeaway. These findings demonstrate that ETTC effectively utilizes predictive entropy to “filter” unreliable signals from weaker models while still leveraging their occasional correct, high-confidence predictions. Unlike voting, which requires careful curation of similarly-capable models to avoid dilution, ETTC allows for safe ensembling: users can integrate smaller, cheaper models (like the 3B) to boost larger ones without the risk of degrading overall performance.

# C.4. Generalization to Thinking LLMs

Setup. To verify that our findings are not an artifact of the visual modality or specific to Vision-Language Models (VLMs), we extended our evaluation to text-only reasoning tasks using Thinking LLMs. We employed the Qwen-3-Thinking family (4B, 30B, and 235B parameters) and evaluated them on two established reasoning benchmarks: ARC-Easy (common sense reasoning) and MMLU-Pro (mathematics subset). We tested various ensemble configurations, including combinations of models with vast size discrepancies (e.g., 4B + 235B).

LLaMA Gemma Pixtral Qwen-3B Qwen-7B Qwen-32B Qwen-72B  
![](images/bb2c1b48eb4d0b8ffdd0c9bd9d715389f9b2a9318c24415a9fe8340f86632e4b.jpg)

<details>
<summary>scatter</summary>

| NMI   | ΔA_MV(16) |
|-------|-----------|
| 0.05  | 0.08      |
| 0.12  | 0.04      |
| 0.18  | 0.05      |
| 0.22  | 0.035     |
| 0.30  | 0.025     |
</details>

(a) MathVista

![](images/2d1425772b7e9fd7fae490c5f73d54a77bf6a411785e4da621d18ed98c4848ec.jpg)

<details>
<summary>scatter</summary>

| NMI   | Value     |
|-------|-----------|
| 0.02  | 0.018     |
| 0.03  | 0.022     |
| 0.06  | 0.034     |
| 0.14  | 0.014     |
| 0.17  | 0.028     |
</details>

(b) MathVision

![](images/0f336fb99a5142c434b2de7ab64dbc9a4cb530992c188442af6480fb55aea839.jpg)

<details>
<summary>scatter</summary>

| NMI   | ΔA_MV(16) |
|-------|-----------|
| 0.08  | 0.08      |
| 0.15  | 0.06      |
| 0.28  | 0.03      |
| 0.29  | 0.025     |
| 0.41  | 0.015     |
</details>

(c) TQA

![](images/02f6b55d54d1249a94b6a1791fbdecc8550a256de4e94265f567355135e634c8.jpg)

<details>
<summary>scatter</summary>

| NMI | Value |
|---|---|
| 0.1 | 6.0e-2 |
| 0.15 | 5.0e-2 |
| 0.23 | 3.5e-2 |
| 0.31 | 2.0e-2 |
| 0.44 | 1.0e-2 |
| 0.48 | 0.5e-2 |
</details>

(d) ScienceQA

![](images/0977b78dea5eb2d7d8caab91dc4a303a5a78f6d9b19eb5638fee6a61686c4850.jpg)

<details>
<summary>scatter</summary>

| NMI   | ΔA_MV(16) |
|-------|-----------|
| 0.05  | 0.05      |
| 0.10  | 0.04      |
| 0.15  | 0.03      |
| 0.20  | 0.025     |
| 0.25  | 0.02      |
| 0.30  | 0.015     |
| 0.35  | 0.01      |
| 0.40  | 0.008     |
</details>

(e) MMStar

![](images/71985f4ecd12ca8b04a234c5ab916d829b833ac4e78b49dda683304b791debf4.jpg)

<details>
<summary>scatter</summary>

| NMI | Value |
|---|---|
| 0.05 | 4.5e-2 |
| 0.10 | 4.8e-2 |
| 0.16 | 3.0e-2 |
| 0.17 | 2.2e-2 |
| 0.25 | 2.5e-2 |
| 0.30 | 2.0e-2 |
| 0.31 | 1.8e-2 |
</details>

(f) MMMU   
Figure 6. Majority voting improvement $\Delta A _ { \mathrm { M V } }$ (16) plotted against average pairwise normalized mutual information (NMI) for each model on each dataset. A negative trend suggests that higher prediction dependency reduces the benefit of majority voting.

Findings. As shown in Tab. 8, the benefits of ETTC generalize robustly to the text domain. Across all 8 ensemble configurations, ETTC consistently outperforms majority voting. Notably, on the MMLU-Pro dataset, aggregating the 4B and 30B models with voting yields 89.34%, significantly underperforming the standalone 30B model (94.12%) due to the noise introduced by the smaller model. In contrast, ETTC achieves 94.08%, effectively recovering the performance of the strong model by filtering out the 4B model’s low-confidence errors. Furthermore, in the most heterogeneous ensemble (4B+235B), ETTC improves upon voting by nearly 5 points on MMLU-Pro (94.67% vs 89.79%), demonstrating its ability to safely leverage small-model compute without diluting the quality of large-model outputs.

Takeaway. These results confirm that the correlation between predictive entropy and correctness is a fundamental property of reasoning models, whether multimodal or text-only. ETTC’s success with “Thinking” models suggests it is a general-purpose, modality-agnostic strategy for enhancing test-time reliability in heterogeneous ensembles.

# C.5. Supervised ETTC

We provide additional details on the supervised variant of ETTC, which learns from a small set of labeled question–model pairs when low entropy is a reliable signal of correctness.

LLaMA Gemma Pixtral Qwen-3B Qwen-7B Qwen-32B Qwen-72B  
![](images/6ef14a305e14b13e776b703888e2320ee316987c49feb788ff25b343c96eeb87.jpg)

<details>
<summary>scatter</summary>

| ρ̄    | ΔAMV(16) |
| ------ | -------- |
| 0.5    | 0.08     |
| 0.55   | 0.04     |
| 0.7    | 0.03     |
| 0.75   | 0.05     |
| 0.8    | 0.02     |
| 0.8    | 0.03     |
</details>

(a) MathVista

![](images/e59409a4ffcd9c96eaf3c0ca07e337dc95a38efa5b083b56a2e01968845b7da6.jpg)

<details>
<summary>scatter</summary>

| Point | ρ̄    | Y Value     |
|-------|------|-------------|
| 1     | 0.65 | 0.02        |
| 2     | 0.60 | 0.015       |
| 3     | 0.68 | 0.025       |
| 4     | 0.72 | 0.03        |
| 5     | 0.70 | 0.028       |
</details>

(b) MathVision

![](images/fb5d1c010203971d5cd9096a740785a031fa793a30d7803496c88f788c435af6.jpg)

<details>
<summary>scatter</summary>

| ρ̄    | ΔA_MV(16) |
| ------ | --------- |
| 0.55   | 0.08      |
| 0.70   | 0.05      |
| 0.80   | 0.03      |
| 0.85   | 0.02      |
| 0.90   | 0.015     |
</details>

(c) TQA

![](images/a88168e6e3b2bce80a32a6a1f69ef7977fea27e8a6e490098f318cfc481f5587.jpg)

<details>
<summary>scatter</summary>

| Point | ρ̄ | Y |
|---|---|---|
| 1 | 0.55 | 0.06 |
| 2 | 0.75 | 0.03 |
| 3 | 0.80 | 0.02 |
| 4 | 0.85 | 0.01 |
| 5 | 0.90 | 0.005 |
</details>

(d) ScienceQA

![](images/8e385dcac5587d829889663b56408ba64ce15579a55c3a1558292d379afb9e1c.jpg)

<details>
<summary>scatter</summary>

| ρ̄    | ΔA_MV(16) |
| ------ | --------- |
| 0.45   | 0.05      |
| 0.55   | 0.03      |
| 0.60   | 0.025     |
| 0.70   | 0.02      |
| 0.75   | 0.015     |
| 0.80   | 0.01      |
</details>

(e) MMStar

![](images/f8a4fd91a77f313a03cafd632af3c6974e9676462d6a52a1b03dcdd2b76101e2.jpg)

<details>
<summary>scatter</summary>

| Point | ρ̄    | Y Value     |
|-------|------|-------------|
| 1     | 0.45 | 0.005       |
| 2     | 0.55 | 0.004       |
| 3     | 0.60 | 0.003       |
| 4     | 0.65 | 0.002       |
| 5     | 0.70 | 0.002       |
| 6     | 0.75 | 0.002       |
| 7     | 0.80 | 0.002       |
| 8     | 0.85 | 0.002       |
| 9     | 0.90 | 0.001       |
| 10    | 0.95 | 0.001       |
</details>

(f) MMMU   
Figure 7. Majority voting improvement $\Delta A _ { \mathrm { M V } } ( 1 6 )$ versus average pairwise accuracy correlation (ρ). Consistent with theory, stronger dependency (i.e., higher ρ) corresponds to smaller gains from majority voting.

Problem setting. Given Q questions and M models, each model u produces a predictive distribution $p _ { q u } ( \cdot )$ over K options for question q, aggregated over $U { = } 1 6$ stochastic decoding samples (see § 4). The goal is to learn a function that predicts whether a model’s low-entropy output is likely to be correct.

Feature construction. For each (q, u) pair, we compute two features:

$$
\widetilde {H} _ {q u} := - \frac {1}{\log K} \sum_ {k = 1} ^ {K} p _ {q u} (k) \log p _ {q u} (k), \quad \operatorname{RelEnt} _ {q u} := \frac {\widetilde {H} _ {q u} - \min _ {v} \widetilde {H} _ {q v}}{\max _ {v} \widetilde {H} _ {q v} - \min _ {v} \widetilde {H} _ {q v}}.
$$

Here ${ \widetilde { H } } _ { q u }$ is the normalized entropy of model u, while ${ \mathrm { R e l E n t } } _ { q u }$ contextualizes this entropy relative to other models for the same question. The final feature vector is $( \widetilde { H } _ { q u } , \mathrm { R e l E n t } _ { q u } ) \in \mathbb { R } ^ { 2 }$ .

Labels and classifier. The binary label is

$$
Z _ {q u} := \mathbb {I} \{\hat {y} _ {q u} = Y _ {q} \},
$$

where $\hat { y } _ { q u }$ is the top-1 prediction and $Y _ { q }$ is the ground truth. We train a logistic regression classifier to predict $\Pr ( Z _ { q u } = 1 )$ from the entropy features.

![](images/469e041fb0eda7cee5a48f8ee999b82737e867ffe11d97a68422642fd1816a0a.jpg)

<details>
<summary>line</summary>

| Model     | Accuracy | H̃u   |
| --------- | -------- | ---- |
| Qwen-3B   | 0.52     | 0.56 |
| LLaMA     | 0.52     | 0.48 |
| Pixtral   | 0.56     | 0.44 |
| Gemma     | 0.65     | 0.32 |
| Qwen-7B   | 0.72     | 0.29 |
| Qwen-32B  | 0.79     | 0.18 |
| Qwen-72B  | 0.81     | 0.16 |
</details>

![](images/6e2b203bed9160b8f3414a33195778cb938cc112fee4aefc2d30e055b586b151.jpg)

<details>
<summary>line</summary>

| Model      | Accuracy | H̃u    |
| ---------- | -------- | ----- |
| Qwen-3B    | 0.22     | 0.65  |
| LLaMA      | 0.23     | 0.60  |
| Pixtral    | 0.25     | 0.56  |
| Qwen-7B    | 0.30     | 0.51  |
| Gemma      | 0.32     | 0.44  |
| Qwen-32B   | 0.39     | 0.41  |
| Qwen-72B   | 0.43     | 0.38  |
</details>

![](images/1d9202500c90be3babb96ce348e978486a35aaa838bde02eb529bdd782d5b23c.jpg)

<details>
<summary>line</summary>

| Model      | H_u / Accuracy |
| ---------- | -------------- |
| Qwen-3B    | 0.6            |
| LLaMA      | 0.7            |
| Pixtral    | 0.75           |
| Qwen-7B    | 0.78           |
| Gemma      | 0.78           |
| Qwen-32B   | 0.8            |
| Qwen-72B   | 0.82           |
</details>

![](images/122bf687c209fde3cd2eb494618a73186f6a043a88dbda95a2430b114653fc7a.jpg)

<details>
<summary>line</summary>

| Model      | H̃_u / Accuracy |
| ---------- | -------------- |
| Qwen-3B    | 0.68           |
| LLaMA      | 0.78           |
| Pixtral    | 0.79           |
| Qwen-7B    | 0.80           |
| Gemma      | 0.78           |
| Qwen-32B   | 0.84           |
| Qwen-72B   | 0.85           |
</details>

![](images/bf00f4392db99c5ebcc63234c81aefe6d2567c2238f812241b0e87cb5822e2a6.jpg)

<details>
<summary>line</summary>

| Model     | H̃u / Accuracy |
| --------- | ------------- |
| Qwen-3B   | 0.41          |
| LLaMA     | 0.45          |
| Pixtral   | 0.50          |
| Qwen-7B   | 0.58          |
| Gemma     | 0.54          |
| Qwen-32B  | 0.56          |
| Qwen-72B  | 0.62          |
</details>

![](images/1190ad5811a8ac6c6e87a9fdff6b683fcb230f7b59531360c1df87b85ece9734.jpg)

<details>
<summary>line</summary>

| Model     | H̃u / Accuracy |
| --------- | ------------- |
| Qwen-3B   | 0.38          |
| LLaMA     | 0.42          |
| Pixtral   | 0.46          |
| Qwen-7B   | 0.48          |
| Gemma     | 0.50          |
| Qwen-32B  | 0.52          |
| Qwen-72B  | 0.64          |
</details>

(e) MMStar   
(f) MMMU   
Figure 8. Correlation between normalized entropy ${ \widetilde { H } } _ { u }$ and accuracy across models on six benchmarks, supporting the Entropy–Accuracy Monotonicity assumption (Assumption 1).

Training protocol. To simulate low-resource conditions, we use two-fold cross-validation across questions: each dataset is split into halves, one for training and one for testing, with roles reversed in a second run. This prevents test leakage and mimics scenarios where only limited annotations are available.

Inference rule. At test time, for each $( q , u )$ we compute the adjusted score

$$
\mathrm{Score} _ {q u} := \widetilde {H} _ {q u} \cdot (1 - \hat {p} _ {q u}),
$$

where $\hat { p } _ { q u }$ is the predicted correctness probability from the classifier. We then select the model with the lowest score:

$$
u _ {q} ^ {\star} := \arg \min _ {u} \operatorname{Score} _ {q u}, \quad \widehat {Y} _ {q} := \hat {y} _ {q u _ {q} ^ {\star}}.
$$

This rule penalizes overconfident but unreliable predictions while rewarding trustworthy ones.

Results. As in Tab. 9, supervised ETTC outperforms both voting and unsupervised ETTC across datasets and ensemble settings. Gains are largest on ambiguous tasks (e.g., MathVision, MMStar, MMMU), where entropy alone is less reliable. Even with only two-fold cross-fitting and no extra supervision, the classifier learns to identify failure modes of entropy selection, making more robust choices and underlining the value of combining entropy with supervised error modeling.

Table 7. Robustness to Ensemble Composition. Performance comparison (%) of majority voting and ETTC across all pairwise and triplet combinations of the Qwen-2.5-VL family on MathVista. Min/Max denote the performance of the worst and best individual models in the ensemble. ETTC consistently outperforms the best individual model (Max) and voting, particularly in heterogeneous ensembles where weak models (e.g., 3B) degrade voting performance. 

<table><tr><td>Combination</td><td>Min.</td><td>Max.</td><td>Avg.</td><td>Voting</td><td>ETTC</td></tr><tr><td>3B, 7B</td><td>51.94</td><td>72.08</td><td>62.01</td><td>69.81</td><td>79.26</td></tr><tr><td>3B, 32B</td><td>51.94</td><td>78.58</td><td>65.26</td><td>72.04</td><td>83.33</td></tr><tr><td>3B, 72B</td><td>51.94</td><td>80.58</td><td>66.26</td><td>73.15</td><td>84.81</td></tr><tr><td>7B, 32B</td><td>72.08</td><td>78.58</td><td>75.33</td><td>81.48</td><td>82.78</td></tr><tr><td>7B, 72B</td><td>72.08</td><td>80.58</td><td>76.33</td><td>82.22</td><td>84.63</td></tr><tr><td>32B, 72B</td><td>78.58</td><td>80.58</td><td>79.58</td><td>84.44</td><td>84.26</td></tr><tr><td>3B, 7B, 32B</td><td>51.94</td><td>78.58</td><td>67.53</td><td>81.30</td><td>82.41</td></tr><tr><td>3B, 7B, 72B</td><td>51.94</td><td>80.58</td><td>68.20</td><td>82.22</td><td>84.44</td></tr><tr><td>3B, 32B, 72B</td><td>51.94</td><td>80.58</td><td>70.37</td><td>83.70</td><td>84.63</td></tr><tr><td>7B, 32B, 72B</td><td>72.08</td><td>80.58</td><td>77.08</td><td>83.70</td><td>84.26</td></tr></table>

Table 8. Generalization to Thinking LLMs. Performance of ETTC versus majority voting on text-only reasoning benchmarks (ARC-Easy, MMLU-Pro) using Qwen-3-Thinking models. ETTC consistently outperforms voting and the best individual model (Max) across diverse ensemble sizes, confirming that entropy-based selection remains effective for pure language reasoning. 

<table><tr><td>Dataset</td><td>Models</td><td>Min.</td><td>Max.</td><td>Avg.</td><td>Voting</td><td>ETTC</td></tr><tr><td rowspan="4">ARC-Easy</td><td>4B, 30B</td><td>0.9599</td><td>0.9714</td><td>0.9656</td><td>0.9769</td><td>0.9878</td></tr><tr><td>4B, 235B</td><td>0.9599</td><td>0.9772</td><td>0.9686</td><td>0.9769</td><td>0.9878</td></tr><tr><td>30B, 235B</td><td>0.9714</td><td>0.9772</td><td>0.9743</td><td>0.9874</td><td>0.9895</td></tr><tr><td>4B, 30B, 235B</td><td>0.9599</td><td>0.9772</td><td>0.9695</td><td>0.9891</td><td>0.9899</td></tr><tr><td rowspan="4">MMLU-Pro</td><td>4B, 30B</td><td>0.8116</td><td>0.9412</td><td>0.8764</td><td>0.8934</td><td>0.9408</td></tr><tr><td>4B, 235B</td><td>0.8116</td><td>0.9431</td><td>0.8773</td><td>0.8979</td><td>0.9467</td></tr><tr><td>30B, 235B</td><td>0.9412</td><td>0.9431</td><td>0.9421</td><td>0.9504</td><td>0.9519</td></tr><tr><td>4B, 30B, 235B</td><td>0.8116</td><td>0.9431</td><td>0.8986</td><td>0.9482</td><td>0.9482</td></tr></table>

Table 9. Evaluation results across datasets for Similar Size Models and Same Family Models. Columns show the average single-model accuracy (Average), Voting, (unsupervised) ETTC, and supervised variant of ETTC. 

<table><tr><td rowspan="2">Accuracy %</td><td colspan="4">Similar Size Models</td><td colspan="4">Same Family Models</td></tr><tr><td>Avg.</td><td>Voting</td><td>ETTC</td><td> $Sup. ETTC_{\Delta}$ </td><td>Avg.</td><td>Voting</td><td>ETTC</td><td> $Sup. ETTC_{\Delta}$ </td></tr><tr><td>MathVista</td><td>61.30</td><td>68.33</td><td>75.93</td><td> $79.63_{3.70\uparrow}$ </td><td>70.80</td><td>83.15</td><td>84.44</td><td> $84.81_{0.37\uparrow}$ </td></tr><tr><td>MathVision</td><td>27.66</td><td>32.05</td><td>35.57</td><td> $36.62_{1.05\uparrow}$ </td><td>33.53</td><td>41.32</td><td>44.84</td><td> $46.34_{1.50\uparrow}$ </td></tr><tr><td>TQA</td><td>76.28</td><td>83.65</td><td>83.90</td><td> $84.14_{0.24\uparrow}$ </td><td>76.73</td><td>84.90</td><td>86.70</td><td> $86.70_{0.00\uparrow}$ </td></tr><tr><td>ScienceQA</td><td>78.44</td><td>85.52</td><td>85.28</td><td> $85.97_{0.69\uparrow}$ </td><td>78.82</td><td>84.04</td><td>85.03</td><td> $86.07_{1.04\uparrow}$ </td></tr><tr><td>MMStar</td><td>51.65</td><td>59.27</td><td>60.07</td><td> $60.67_{0.60\uparrow}$ </td><td>54.22</td><td>61.00</td><td>63.73</td><td> $65.07_{1.34\uparrow}$ </td></tr><tr><td>MMMU</td><td>48.39</td><td>53.66</td><td>58.63</td><td> $59.01_{0.38\uparrow}$ </td><td>52.79</td><td>58.63</td><td>65.34</td><td> $66.46_{1.12\uparrow}$ </td></tr><tr><td>Average</td><td>57.29</td><td>63.75</td><td>66.56</td><td> $67.67_{1.11\uparrow}$ </td><td>61.15</td><td>68.84</td><td>71.68</td><td> $72.58_{0.90\uparrow}$ </td></tr></table>

# Limitations

Our study focuses on multiple-choice visual reasoning tasks and assumes access to model confidence scores via output distributions. The proposed methods, especially entropy-based selection, may not directly generalize to open-ended tasks or models lacking probabilistic outputs. Additionally, while our evaluation covers diverse datasets and model ensembles, the gains of supervised entropy-based TTC depend on the quality and availability of annotated examples, which may be costly to obtain in some domains. Lastly, our analysis assumes that entropy correlates with accuracy, which may not hold for all models or tasks.

# LLM Usage

We used ChatGPT as general-purpose assistive tools during the preparation of this paper. Specifically, LLMs were employed for polishing grammar, improving clarity, formatting LaTeX, generating illustrative figures, and debugging minor code snippets. LLMs were not involved in research ideation, experimental design, or the development of theoretical results.