Anthony Fuller 1 2 James R. Green 1 Evan Shelhamer 2 3

# Abstract

Model soups are strange and strangely effective combinations of parameters. They take a model (the stock), fine-tune it into multiple models (the ingredients), and then mix their parameters back into one model (the soup) to improve predictions. While all known soups require supervised learning, and optimize the same loss on labeled data, our recipes for Self-Soupervision generalize soups to self-supervised learning (SSL). Our Self-Souping lets us flavor ingredients on new data sources, e.g. from unlabeled data from a task for transfer or from a shift for robustness. We show that Self-Souping on corrupted test data, then fine-tuning back on uncorrupted train data, boosts robustness by +3.5% (ImageNet-C) and +7% (LAION-C). Self-Soupervision also unlocks countless SSL algorithms to cook the diverse ingredients needed for more robust soups. We show for the first time that ingredients can differ in their SSL hyperparameters—and more surprisingly, in their SSL algorithms. We cook soups of MAE, MoCoV3, and MMCR ingredients that are more accurate than any one single SSL ingredient.

# 1. Introduction: More Soups, Less Supervision

Model soups make several models (the ingredients) by independent fine-tunings initialized from a single model (the stock), then merge them back into one model (the soup) by mixing parameters to improve prediction accuracy (Wortsman et al., 2022). Each fine-tuning varies in its configuration (e.g. optimization hyperparameters) and each mixing can be a simple average or more sophisticated linear combination. In this way, soups convert more training time into more accuracy without more inference time: the soup model needs only as much computation as the original model.

Model soups are surprisingly possible, in that mixing model

1Carleton University, Ottawa, Canada 2Vector Institute, Toronto, Canada 3University of British Columbia, Vancouver, Canada. Correspondence to: Anthony Fuller <anthonyfuller@cmail.carleton.ca>.

Preprint. February 4, 2026.

parameters is absolutely not guaranteed to result in a better model (or even an equally good model!). They are also surprisingly productive with improvements across many settings: vision (Jain et al., 2023; Wortsman et al., 2022), language (Ablin et al., 2025; Jang et al., 2023; Chronopoulou et al., 2023), text-to-image (Biggs et al., 2024), federated learning (Chen et al., 2024), domain generalization (Rame´ et al., 2023; Rame et al. ´ , 2022), and class imbalance (Aminbeidokhti et al., 2025). However all known soups have to train ingredients by supervised learning—depriving our palettes of tasty new soups for many occasions.

We thus introduce Self-Soups, which are model soups made from ingredients that differ in their independent self-supervised training runs. Self-Souping vastly expands the menu of possible soups by harnessing different losses to flavor ingredients from different distributions without requiring labels (Fig. 1). We can “inter-train” models to make ingredients by self-supervised learning (SSL), after pre-training but before fine-tuning to a task, to enable transfer and robustness by optimizing more losses on more data.

Self-Soupervision unlocks countless self-supervised losses for preparing the diverse ingredients that robust soups need. For example, in §4.2 we first promote ingredient diversity by inter-training using 3 fundamentally different selfsupervised losses, which we then fine-tune with labels and mix for improved robustness. These ingredients differ in their SSL runs (for preparation) and supervised training runs (for specialization). We can also inter-train on the test distribution to make shift-aware ingredients that improve robustness. For example, in §4.3 we inter-train on corrupted data (ImageNet-C (Hendrycks & Dietterich, 2019) and LAION-C (Li et al., 2025)), then fine-tune back on the distribution for which labels are available (ImageNet training data (Russakovsky et al., 2015)) to boost accuracy by +3.5 and +7%. In §4.4 we show that Self-Soupervision helps transfer pre-trained models to 21 diverse visual tasks (the VTAB collection (Zhai et al., 2020)). In our final experiment (§4.5), we mix ingredients that differ only in their SSL runs—made possible by Self-Soupervision. For each VTAB dataset, we run self-supervised inter-trainings that differ in their self-supervised algorithms and algorithmic hyperparameters. We then mix these purely self-supervised ingredients by quickly “seasoning” (Croce et al., 2023) them: choosing the mixture conditioned on few-shot labels. We even mix soups for a task without training labels by a new and fully unsupervised variant that we call Self-Seasoning.

![](images/2b109114e2039a7746af1f1b1c4edec36ff73bbac9b573d698d33cd8672806bb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Model Soup (Wortsman et al., 2022) (Ramé et al., 2022)"] --> B["soup"]
    B --> C["ingrds."]
    B --> D["stock"]
    C --> E["soup"]
    D --> F["ingrds."]
    D --> G["soup"]
    E --> H["ingrds."]
    E --> I["soup"]
    F --> J["ingrds."]
    F --> K["soup"]
    G --> L["ingrds."]
    G --> M["soup"]
    H --> N["soup"]
    I --> O["soup"]
    J --> P["soup"]
    K --> Q["soup"]
    L --> R["soup"]
    M --> S["soup"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#ffc,stroke:#333
    style H fill:#cfc,stroke:#333
    style I fill:#cfc,stroke:#333
    style J fill:#cfc,stroke:#333
    style K fill:#cfc,stroke:#333
    style L fill:#fcc,stroke:#333
    style M fill:#fcc,stroke:#333
    style N fill:#ffc,stroke:#333
    style O fill:#ffc,stroke:#333
    style P fill:#cfc,stroke:#333
    style Q fill:#cfc,stroke:#333
    style R fill:#cfc,stroke:#333
    style S fill:#cfc,stroke:#333
    style T fill:#cfc,stroke:#333
    style U fill:#cfc,stroke:#333
    style V fill:#cfc,stroke:#333
    style W fill:#cfc,stroke:#333
    style X fill:#cfc,stroke:#333
    style Y fill:#cfc,stroke:#333
    style Z fill:#cfc,stroke:#333
```
</details>

Figure 1. Soups & Supervision. Soups fine-tune then mix models to improve predictions. The original Model Soup (left) fine-tunes across hyperparameters for the task, while Model Ratatouilles (center) first inter-trains on different labeled data then fine-tunes for the task. Both are supervised and cannot harness unlabeled data. Our Self-Soupervision (right) inter-trains across different losses and data to make more and better soups with and without labels. Our Self-Soups fine-tune then mix into a supervised model for the task. $\theta ^ { \mathrm { { p t } } }$ is the pre-trained stock for initialization, N is the number of fine-tunings on data j, M is the number of inter-trainings on data i, and θ is the final model. We color supervised and self-supervised training runs / components. “ingreds.” is short for model-soup ingredients.

# 2. Background: Supervised Soups and Self-supervised Learning

# 2.1. Supervised Model Soups

Initializing cooking with a stock. Soups require that the models for mixing (the ingredients) share the same initial parameters for optimization (the stock).

Adding ingredients by fine-tuning. Each fine-tuning of the stock creates an ingredient for mixing a soup. Multiple different fine-tunings are key for different ingredients: soups rely on differences among the ingredient models for their gains. To vary their ingredients, Model Soups (Wortsman et al., 2022) vary the optimization parameters, such as the learning rate, data augmentation, and optimizer.

Boosting ingredient diversity via inter-training. Model Ratatouille (Rame et al. ´ , 2023) fine-tunes in two stages, optimizing ingredients for longer, and increasing their diversity for domain generalization. Ratatouille first initializes with a stock, then “inter-trains” on up to 5 auxiliary labeled datasets independently, and finally fine-tunes these models on the target task for mixing. Ratatouille gains 0.5% when tested out of distribution. Although this gain is modest, like souping, Ratatouille produces a single model without increased inference/deployment costs. Furthermore, Ratatouille showed that soups could be made from ingredients trained on different labeled datasets. We show soups can be made from ingredients trained on different unlabeled datasets and different self-supervised losses before mixing.

# 2.2. Linear Mode Connectivity

Not all models can be mixed. When mixing works, for ingredients that share a stock, the condition of linear mode connectivity holds. Linear mode connectivity (LMC) holds for two models if the accuracy of the interpolated model weights is greater than or equal to the interpolated accuracy of the models. Formally, for all $\lambda \in [ 0 , 1 ]$ ],

$$
\begin{array}{c} \mathbf {a c c} ((1 - \lambda) \cdot \theta_ {A} + \lambda \cdot \theta_ {B}) \geq \\ (1 - \lambda) \cdot \mathbf {a c c} (\theta_ {A}) + \lambda \cdot \mathbf {a c c} (\theta_ {B}) \end{array} \tag {1}
$$

where λ is the interpolation weight, $\theta _ { A } , \theta _ { B }$ are the model weights, and acc is the accuracy.

Although model soups can provide gains, the known cases in which LMC holds are few. Our work adds another case. All methods initialize from a shared stock, but differ in how they optimize ingredients:

• Fine-tuning with different stochasticity (e.g. order, augmentation) (Frankle et al., 2020)   
• Fine-tuning with different hyper-parameters (e.g. learning rate) (Wortsman et al., 2022)   
• Inter-training on different supervised datasets, then finetuning on a target task (Rame et al. ´ , 2023)   
• Inter-training on different supervised subsets, then finetuning on a target task (Aminbeidokhti et al., 2025)   
• Fine-tuning with different rewards (Rame et al. ´ , 2023)   
• Fine-tuning with different attacks (Croce et al., 2023)

![](images/a34760cdcdf01c617a9e6ac34a6f503f29f1111d9b2e450fdbda0e95937cd4e8.jpg)  
Figure 2. Souping across different SSL algorithms. We explore mixtures of 3 ingredient models that we inter-train with different selfsupervised losses: MAE, MoCoV3, and MMCR. MAE is a pixel-reconstruction algorithm, MoCoV3 is an instance-contrastive algorithm, and MMCR is a dimension-contrastive algorithm. Through this experiment, we are the first to show that linear-mode connectivity can ingredients alone (no mix), and the inner triangles represent mixtures of the 3 ingredients. Each central inner triangle is a hold between ingredients that differ in their self-supervised training runs. The circles located at the corners of the 8 triangles represent the $\textstyle { \left( { \frac { 1 } { 3 } } , { \frac { 1 } { 3 } } , { \frac { 1 } { 3 } } \right) }$ mix, and the other inner triangles are unequal mixes (where mixture coefficients are proportional to the proximity to the corner ingredients).

# 2.3. Self-Supervised Learning

Algorithms. Self-supervised learning (SSL) is a powerful framework because it enables learning from raw/unlabeled data itself; this allows for scaling to gigantic datasets and enables deep learning for niche applications with few annotations (Balestriero et al., 2023). We highlight three popular families of SSL algorithms, which we use. (1) SSL by reconstruction (e.g. MAE (He et al., 2022) or SimMIM (Xie et al., 2022)) masks or alters parts of samples and pre-trains models to predict the originals. (2) SSL by instance-contrastive learning (e.g. MoCoV3 (Chen et al., 2021) or SimCLR (Chen et al., 2020)) pre-trains models to produce similar embeddings derived from positive pairs of inputs (e.g. different augmentations of the same sample) and dissimilar embeddings for negatives (e.g. different samples). (3) SSL by dimension-contrastive learning (e.g. MMCR (Yerxa et al., 2023) or VICReg (Bardes et al., 2021)) pre-trains models to embed positive pairs similarly while encouraging embeddings to vary over a batch of samples without defining negative pairs. Unsurprisingly, different SSL algorithms learn different representations (Park et al., 2023) that transfer differently—motivating our use of different algorithms to create diverse ingredients that make robust soups. We choose MAE, MoCoV3, and MMCR to represent the SSL families, and use them to train ingredient models for tasty Self-Soups. We mostly focus on MAE, since it is the most popular and accessible SSL algorithm (e.g. it is insensitive to batch size, robust to hyperparameters, computationally efficient, and applicable to many modalities).

Algorithmic Hyperparameters. Each SSL algorithm has its own configuration space. For example, the masking ratio in MAE, the temperature in MoCoV3, and the local-global losses in MMCR. These algorithmic choices let us train ingredient models differently to make diverse Self-Soups.

Continued SSL initializes from a model pre-trained by SSL on one dataset, then further trains by SSL on another dataset (Reed et al., 2022; Gururangan et al., 2020) to create domainspecific SSL models (Rodas et al., 2025; Kyrollos et al., 2023). Continued SSL is a special case of our inter-trainings varying the data, SSL algorithm, and SSL hyperparameters.

# 3. Method: Cooking without Labels and Instant Seasoning

We introduce Self-Soupervision, which creates ingredients (parameters to mix) by training from a stock (parameters to initialize) without requiring labels at every stage. Our framework is broad; any soup that is made using ingredients that differ in their SSL runs qualifies as a Self-Soup; thus, there are endless possible instantiations of Self-Soupervision. For example, the choice of stock, SSL algorithm and algorithmic hyperparameters, training data, training length/schedule, optimization hyperparameters, additional training stages (e.g. fine-tuning), etc. Centrally, Self-Soupervision allows for cooking soups on more data and from new sources—e.g. that are closer to the target distribution—and using different losses—e.g. that are more aligned with the target task. Formally, we define Self-Soupervision as:

![](images/879795a6661ade24d870a8c93fe38c1d9b287709f68f53dc7ea5b7f21fd6eb7d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Equal search/development cost"] --> B["Fine-tuning"]
    A --> C["Model Soup"]
    B --> D["final model"]
    C --> E["final model"]
    D --> F["θ=Select_max {Train(θ^pt, X_j, Y_j)}_j"]
    E --> G["θ=Select_max {Train(θ^pt, X_i, X_j, Y_j)}_i"]
    H["Continued SSL + Fine-tuning"] --> I["stock"]
    H --> J["stock"]
    H --> K["stock"]
    H --> L["stock"]
    I --> M["θ=1/N Σ_{j}^N Train(θ^pt, X_j, Y_j)"]
    J --> N["θ=1/N Σ_{j}^M Train(θ^pt, X_i, X_j, Y_j)"]
    K --> O["θ=1/N Σ_{j}^M Train(θ^pt, X_i, X_j, Y_j)"]
    P["Self-Soupervision (ours)"] --> Q["stock"]
    P --> R["stock"]
    P --> S["stock"]
    Q --> T["θ=Select_max {Train(θ^pt, X_j, Y_j)}_j"]
    R --> U["θ=Select_max {Train(θ^pt, X_i, X_j, Y_j)}_i"]
    S --> V["θ=Select_max {Train(θ^pt, X_i, X_j, Y_j)}_j"]
    style A fill:#f9f,stroke:#333
    style H fill:#ccf,stroke:#333
    style P fill:#cfc,stroke:#333
```
</details>

Figure 3. Model Soup mixes ingredients from a fine-tuning hyperparameter search; likewise, our Self-Soupervision mixes ingredients from a continued SSL + fine-tuning hyperparameter search. Continued SSL enables learning from unlabeled data—e.g., from the fine-tuning domain or test-set shift—before fine-tuning. Self-Souping these diverse ingredients further improves accuracy. Symbols follow Fig. 1.

$$
\theta = \frac {1}{N \cdot M} \sum_ {j} ^ {N} \sum_ {i} ^ {M} \text { Train } (\overbrace {\text { Train } (\theta^ {\mathrm{pt}} , X _ {i})} ^ {Y _ {i} \text { not   required}}, X _ {j}, Y _ {j}) \tag {2}
$$

where θ are soup parameters, $\theta ^ { \mathrm { p t } }$ are pre-trained / stock parameters, N is the number of fine-tunings per inter-training, M is the number of inter-trainings, $X _ { i } / X _ { j }$ are the inputs of the $i ^ { t h } / j ^ { t h }$ dataset, and $Y _ { j }$ are the labels of the $j ^ { t h }$ dataset. Supervised ingredients alone make standard soups. We introduce the self-supervised ingredients—which do not need labels—and which we can also use alone (i.e., by dropping the supervised ingredients) to make soups without fine-tuning to a task. We show these fully self-supervised soups for transfer in §4.5.

Mixing by Instant Seasoning. The original and simplest way to mix a soup is to average the ingredients. This uniform mixture may improve predictions, but may not be the best mixture for a given task and dataset. Seasoning (Croce et al., 2023) searches for a better mixture over a grid of options by mixing each model, making predictions on a fewshot labeled dataset, and picking the best. While effective, this only applies to fine-tuned ingredients: seasoning makes predictions by mixing the classifiers. We instead mix purely self-supervised ingredients into a model for representation— rather than classification—then compute its representation on training and testing data for prediction by nearest neighbors. Our variant of seasoning mixes without classifier training for the “instant” seasoning of soups across different tasks. Specifically, we randomly sample the M mixture coefficients uniformly from the probability simplex.

Mixing by Self-Seasoning. We pair our new ingredients with a new and unsupervised way to mix: Self-Seasoning. We optimize our M mixture coefficients by gradient descent to minimize the entropy of predictions by nearest neighbors.

# 4. Experiments

Baselines: Supervised soups and continued SSL. We run five experiments to evaluate Self-Souping in different settings. First, we investigate if Self-Souping is possible (§4.1). In the next three settings (§4.2, §4.3, §4.4), we compare against two baselines: Model Soups, which are supervised-only soups, and “continued SSL + supervised soups”. The latter is a new baseline that we provide, which is a combination of existing methods yet is not a Self-Soup by our definition. A continued SSL + supervised soup first inter-trains several models using SSL, fine-tunes from them, then mixes several fine-tunings that originate from one inter-training; since these ingredients differ only in their supervised fine-tunings, they are not Self-Soups and are thus appropriate baselines. Continued SSL + supervised soup has an equal search cost to our Self-Soups, as they both search over M SSL inter-trainings and N supervised fine-tunings: please see Fig. 3. Importantly, many real-world settings prioritize a model’s accuracy versus inference/deploymentcost trade-off. This prioritization is common when models are used frequently when deployed, so the cost of running them matters far more than the cost of developing them. In these settings with larger search/development budgets, our results may leave even better ingredients on the table, since Self-Soupervision provides more dimensions to search and soup over (e.g. SSL algorithms and unlabeled datasets) for further gains over supervised soups. Our final setting (§4.5) directly mixes SSL trainings (without fine-tuning for a task), we thus only compare to the stock and the best ingredient.

Table 1. Soups on ImageNet. Our Self-Soup gains over supervised soups on corruptions (+1% on IN-C and +0.4% on LAION-C) and the most challenging test set (+6.6% relative improvement on IN-A). For “greedy search” and “best ingredient”, we select based on IN-Val accuracy—thus, greedy may not always rank first for every test set. “IN” is for ImageNet. Results are % top-1 acc. Best and 2nd best. 

<table><tr><td>Soup Type</td><td>Mix Method</td><td>IN-Val</td><td>IN-ReaL</td><td>IN-V2</td><td>IN-HR</td><td>IN-A</td><td>IN-R</td><td>IN-C</td><td>LAION-C</td></tr><tr><td rowspan="3">Supervised Soup</td><td>Best ingredient</td><td>79.05</td><td>85.07</td><td>67.51</td><td>87.34</td><td>7.49</td><td>28.84</td><td>28.74</td><td>18.67</td></tr><tr><td>Greedy Search</td><td>79.34</td><td>85.58</td><td>68.09</td><td>87.40</td><td>7.91</td><td>29.92</td><td>30.88</td><td>21.02</td></tr><tr><td>Uniform Mix</td><td>79.12</td><td>85.54</td><td>68.01</td><td>87.46</td><td>7.75</td><td>30.05</td><td>30.91</td><td>22.05</td></tr><tr><td rowspan="3">Continued SSL + Supervised Soup</td><td>Best ingredient</td><td>78.99</td><td>84.99</td><td>67.22</td><td>87.66</td><td>7.24</td><td>29.38</td><td>27.81</td><td>18.93</td></tr><tr><td>Greedy Search</td><td>79.29</td><td>85.47</td><td>68.16</td><td>87.78</td><td>7.37</td><td>30.09</td><td>30.66</td><td>20.93</td></tr><tr><td>Uniform Mix</td><td>79.23</td><td>85.58</td><td>68.17</td><td>87.84</td><td>7.61</td><td>30.21</td><td>31.24</td><td>21.61</td></tr><tr><td rowspan="3">Self-Soup (ours)</td><td>Best ingredient</td><td>78.99</td><td>84.99</td><td>67.22</td><td>87.66</td><td>7.24</td><td>29.38</td><td>27.81</td><td>18.93</td></tr><tr><td>Greedy Search</td><td>79.35</td><td>85.55</td><td>68.48</td><td>87.56</td><td>8.20</td><td>30.07</td><td>31.98</td><td>21.72</td></tr><tr><td>Uniform Mix</td><td>79.11</td><td>85.62</td><td>68.21</td><td>87.68</td><td>8.43</td><td>30.22</td><td>32.23</td><td>22.40</td></tr></table>

Datasets: ImageNet and VTAB with shifts. We choose the gold-standard for image classification, ImageNet-1K (Russakovsky et al., 2015), and the popular transfer dataset, VTAB (Zhai et al., 2020), which is a collection of 21 datasets. We evaluate extra test sets for ImageNet: ImageNet-ReaL (improved labels for ImageNet-Val (Beyer et al., 2020)), ImageNet-V2 (reproduction of ImageNet-Val (Recht et al., 2019)), ImageNet-A (challenging images (Hendrycks et al., 2021b)), ImageNet-HR (higher-effort annotations (Fuller et al., 2024)), ImageNet-R (rendition shifts (Hendrycks et al., 2021a)), ImageNet-C (corruption shifts (Hendrycks & Dietterich, 2019)), and LAION-C (more corruption shifts (Li et al., 2025)). VTAB does not have shifts, so we make our own shifted data with ImageNet-C’s code for all 15 corruptions types at the highest severity. We make 1K subsets of the train and test sets for convenient computation. We call our version mini-VTAB-C, as it is smaller and has corruptions, and it is shared in the supplement (§C).

Stocks. We use the MAE stock (ViT-B pre-trained for 1600 epochs on ImageNet-1K by He et al. (2022)). Later (Fig. 4) we show Self-Souping works equally well for another stock.

# 4.1. Is Self-Soupervision possible and productive?

Setup: Souping over different SSL algorithms. To check if ingredients can be souped that differ in their selfsupervision, we first inter-train 3 models on the ImageNet-1K training set for 5 epochs with a 1e-5 learning rate. One model uses the MAE algorithm, another uses MoCoV3, and the last uses MMCR. After inter-training, we fine-tune each model using supervised learning for 10 epochs (following Wortsman et al. (2022)) with an 8e-5 learning rate. After fine-tuning, we have 3 ingredients from which to cook our soup. To explore the mixture space, we compute 49 convex combinations of ingredients that are uniformly distributed, along with 3 one-hot mixtures (i.e. the ingredients alone). For each of the 52 models, we test on the ImageNet test sets.

Results: Self-Souping is productive and LMC holds. Fig. 2 shows 52 different mixtures of our 3-ingredient soups across 8 ImageNet test sets. For all test sets, souping across SSL algorithms is productive: the best models are always a combination of ingredients and the worst models are always the ingredients alone. The largest gains of +3% are on corrupted data (ImageNet-C and LAION-C), and the best mixtures are roughly-equal combinations of our 3 ingredients (i.e. the triangle centers). LMC can thus hold between self-supervised ingredients—a novel finding that whets the appetite for more tasty soups now that it is possible.

# 4.2. Can Self-Soups outperform supervised soups?

Setup: Inter-train on ImageNet then fine-tune on ImageNet. We experiment with SSL inter-training on the finetuning data. In this case, self-supervision provides more ingredients, and more diverse ingredients by inter-training using different SSL algorithms. We fine-tune each of the 3 inter-trainings from §4.1 for 10 epochs—doing this 4 times (varying fine-tuning learning rates {6e-5, 8e-5, 1e-4, 1.5e-4}). We compare to initialization from the MAE stock.

Results: Self-Soups help on challenges and corruptions. Self-Souping’s largest gains are on corrupted data: +1% on ImageNet-C and +0.4% on LAION-C over other soups (Tab. 1). Self-Souping also meaningfully gains on the most challenging test set, ImageNet-A (+6.6% relative improvement).

# 4.3. Does inter-training on the test distribution help?

Setup: Inter-train on shifts then fine-tune on ImageNet. We now allow for SSL inter-training on shifted data. In this case, SSL enables ingredients to learn from the test distribution without labels for robustness to it. We measure this on split data for optimization and evaluation (Tab. 2). We first inter-train 4 models (varying learning rates {1e-5, 2e-5, 3e-5, 4e-5}) by MAE for 100K steps on unlabeled test samples that are even-indexed. We fine-tune these models back on the ImageNet training set following the fine-tuning runs in §4.2. We report results for odd-indexed and even-indexed test samples to measure accuracy when inter-training and evaluation samples are disjoint and joint, respectively.

Table 2. Self-Soupervision on the test distribution provides large gains: +3.5% on IN-C and +7% on LAION-C. These soups mix in ingredients inter-trained by SSL on unlabeled test data. In the disjoint setting, we use even-indexed test samples for inter-training on corruptions and odd-indexed samples for testing on corruptions. In the joint setting, we use even-indexed test samples for both inter-training and testing. “IN” is for ImageNet, “LA” is for LAION. Results are % top-1 acc. Best and 2nd best. 

<table><tr><td>Method</td><td>IN-Val</td><td>IN-C</td><td>LA-C</td></tr><tr><td colspan="4">Same models as Tab. 1 for reference</td></tr><tr><td>Supervised Soup on IN-Train</td><td>79.05</td><td>30.91</td><td>22.05</td></tr><tr><td>Cont. SSL + Soup on IN-Train</td><td>79.23</td><td>31.24</td><td>21.61</td></tr><tr><td>Self-Soup on IN-Train</td><td>79.11</td><td>32.23</td><td>22.40</td></tr><tr><td colspan="4">Disjoint (inter-train: even-indexed, test: odd-indexed samples)</td></tr><tr><td>Self-Soup on IN-C</td><td>78.96</td><td>35.72</td><td>23.28</td></tr><tr><td> $\hookrightarrow$  Best ingredient</td><td>78.91</td><td>31.41</td><td>19.55</td></tr><tr><td>Self-Soup on LA-C</td><td>78.87</td><td>32.58</td><td>29.48</td></tr><tr><td> $\hookrightarrow$  Best ingredient</td><td>78.87</td><td>28.25</td><td>23.38</td></tr><tr><td>Self-Soup on IN-C + LA-C</td><td>78.81</td><td>34.43</td><td>26.39</td></tr><tr><td colspan="4">Joint (inter-train: even-indexed, test: even-indexed samples)</td></tr><tr><td>Self-Soup on IN-C</td><td>78.96</td><td>36.02</td><td>23.51</td></tr><tr><td> $\hookrightarrow$  Best ingredient</td><td>78.91</td><td>31.70</td><td>19.57</td></tr><tr><td>Self-Soup on LA-C</td><td>78.87</td><td>32.51</td><td>30.32</td></tr><tr><td> $\hookrightarrow$  Best ingredient</td><td>78.87</td><td>28.09</td><td>23.84</td></tr><tr><td>Self-Soup on IN-C + LA-C</td><td>78.81</td><td>34.55</td><td>26.83</td></tr></table>

Results: Shift-aware ingredients deliver robustness. Self-Souping on the test distribution (but not the test samples themselves) provides large gains: +3.5% on ImageNet-C and +7% on LAION-C. Self-Souping on test samples themselves provides a small boost on top: +0.3% on ImageNet-C and +0.8% on LAION-C. Despite the different types of corruptions present in ImageNet-C versus LAION-C, there are benefits to inter-training on one set of corruptions to the other set. Self-Souping over both test sets—i.e. where ingredients differ in their SSL inter-training data distributions and fine-tuning runs—keeps most of the shift-specific gains.

Bonus: Why not adapt to the shift at test time? Another unsupervised way to adapt to a shift is test-time adaptation (TTA), e.g. by minimizing prediction entropy (Wang\* et al., 2021). We use SAR (Niu et al., 2023) as a SOTA TTA method. Our Self-Soup on ImageNet-C without SAR still beats soups prepared on ImageNet-Train with SAR applied on ImageNet-C (Tab. 3). In this case, our soup made from shift-aware ingredients that we updated on the test distribution (odd-indexed), not test samples (even-indexed), outperforms models updated on test samples. This Self-Soup on ImageNet-C gains more with adaptation by SAR (35.72 → 37.50), showing the two strategies can be complementary.

Table 3. Our Self-Soup cooked on ImageNet-C keeps its robustness advantage after test-time adaptation (TTA). We run soups prepared on ImageNet-Train (from Tab. 1 and 2) on ImageNet-C (odd indices) with and without TTA—and repeat for our Self-Soup on ImageNet-C (disjoint, from Tab. 2). Results are % top-1 acc. 

<table><tr><td>Method</td><td>without TTA</td><td>with TTA</td></tr><tr><td>Supervised Soup on ImageNet-Train</td><td>30.95</td><td>32.66</td></tr><tr><td>Self-Soup on ImageNet-Train</td><td>32.25</td><td>34.00</td></tr><tr><td>Self-Soup on ImageNet-C</td><td>35.72</td><td>37.50</td></tr></table>

# 4.4. Can Self-Soupervision bring transfer gains?

Setup: Self-Souping on 21 downstream tasks. We cook task-specific Self-Soups for improved in- and out-ofdistribution accuracy. In this case, SSL allows ingredients to learn from unlabeled samples from the target distribution prior to fine-tuning. For each of the 21 datasets in VTAB, we inter-train 4 models (varying learning rates {1e-5, 2e-5, 3e-5, 4e-5}) using MAE for 10K steps on all training samples. For each model, we fine-tune 4 times (varying learning rates {1e-5, 2e-5, 3e-5, 4e-5}) for 100 epochs on mini-VTAB. As a baseline, we run the same fine-tuning procedure but initialize from the pre-trained MAE stock.

Results: Self-Souping transfers well. Averaged over the 21 mini-VTAB datasets, our Self-Soup with greedy search achieves 67.5% top-1 accuracy on clean test data, the second best achieves 66.2% (Tab. 4). If we then average over the corrupted test sets in mini-VTAB-C, our Self-Soup with uniform mixing achieves 35.1%, the next best achieves 34.6%. Self-Souping thus provides small yet useful gains that we thoroughly measure across 336 test sets (21 tasks × 16 corrupt/natural conditions). Our method gains the most for the 8 natural datasets: +1% over the continued SSL + supervised soup, and +1.4% over the supervised-only soup. Self-Souping is most robust to blur, weather, and digital corruption types (+2%), while it handles noise worse (-1% versus the supervised-only soup).

# 4.5. Can we quickly mix SSL ingredients directly?

Setup: Self-Seasoning of self-supervised ingredients. We now mix soups without supervised fine-tuning and evaluate by nearest neighbors (kNN). We make 6 ingredients per task with MAE, MMCR (×2), MoCoV3 (×2), and the stock.

Table 4. Soups for transfer. Measured across 336 test sets (21 datasets × 16 corruptions), our Self-Soups show modest yet useful gains. We make soups that are specialized for 21 tasks (from the VTAB collection) from 3 broad domains. We evaluate on clean / original test data and 15 corruption types to measure robustness. For each soup type, we evaluate the best ingredient (selected on clean test data), a greedy search over ingredients (selected on clean test data), and a uniform mixture of all ingredients. We emphasize the best and 2nd best.   
(a) Top-1 % accuracy for 21 tasks (VTAB) averaged over 15 corruption types and clean data. Self-Soups are most helpful on natural data. 

<table><tr><td rowspan="2">Soup Type</td><td rowspan="2">Mix Method</td><td colspan="8">Natural</td><td colspan="4">Specialized</td><td colspan="8">Structured</td><td>Mean</td><td rowspan="2"></td></tr><tr><td>Caltech101</td><td>CIFAR-10</td><td>CIFAR-100</td><td>DTD</td><td>Flowers102</td><td>Pets</td><td>Sun397</td><td>SVHN</td><td>Camelyon</td><td>EuroSAT</td><td>Resisc45</td><td>Retinopathy</td><td>Clevr-Count</td><td>Clevr-Dist</td><td>DMLab</td><td>dSpr-Loc-X</td><td>dSpr-Loc-Y</td><td>dSpr-Loc-Ori</td><td>KITTI-Dist</td><td>sNORB-Azim</td><td>sNORB-Elev</td></tr><tr><td rowspan="3">Supervised Soup</td><td>Best ingredient</td><td>39.4</td><td>58.4</td><td>28.1</td><td>24.2</td><td>26.9</td><td>34.9</td><td>7.4</td><td>57.5</td><td>67.2</td><td>49.8</td><td>29.7</td><td>67.3</td><td>30.0</td><td>26.1</td><td>35.1</td><td>8.1</td><td>16.0</td><td>24.1</td><td>48.0</td><td>9.4</td><td>18.3</td><td>33.6</td></tr><tr><td>Greedy Search</td><td>41.1</td><td>58.6</td><td>29.1</td><td>26.1</td><td>28.2</td><td>35.6</td><td>8.3</td><td>57.8</td><td>67.2</td><td>49.8</td><td>30.8</td><td>68.1</td><td>30.0</td><td>26.1</td><td>35.1</td><td>8.2</td><td>15.5</td><td>23.4</td><td>52.6</td><td>9.5</td><td>19.5</td><td>34.3</td></tr><tr><td>Uniform Mix</td><td>42.5</td><td>58.1</td><td>29.6</td><td>26.0</td><td>27.2</td><td>35.0</td><td>8.2</td><td>56.6</td><td>66.4</td><td>50.7</td><td>30.8</td><td>69.3</td><td>31.5</td><td>29.7</td><td>36.4</td><td>7.7</td><td>16.2</td><td>23.0</td><td>53.4</td><td>9.2</td><td>19.3</td><td>34.6</td></tr><tr><td rowspan="3">Continued SSL + Supervised Soup</td><td>Best ingredient</td><td>40.8</td><td>56.8</td><td>27.7</td><td>24.2</td><td>27.4</td><td>31.8</td><td>8.6</td><td>55.5</td><td>65.8</td><td>49.8</td><td>27.7</td><td>71.2</td><td>29.3</td><td>24.0</td><td>33.6</td><td>6.3</td><td>13.9</td><td>22.3</td><td>47.5</td><td>10.2</td><td>17.8</td><td>33.0</td></tr><tr><td>Greedy Search</td><td>40.8</td><td>58.4</td><td>29.1</td><td>24.2</td><td>28.3</td><td>34.1</td><td>8.9</td><td>57.5</td><td>66.4</td><td>49.8</td><td>29.8</td><td>71.2</td><td>29.3</td><td>24.0</td><td>34.8</td><td>6.3</td><td>14.4</td><td>23.6</td><td>50.9</td><td>10.2</td><td>17.8</td><td>33.8</td></tr><tr><td>Uniform Mix</td><td>44.1</td><td>58.3</td><td>29.5</td><td>24.9</td><td>27.9</td><td>34.2</td><td>8.9</td><td>58.3</td><td>67.3</td><td>49.8</td><td>29.8</td><td>71.3</td><td>28.4</td><td>28.7</td><td>35.5</td><td>6.5</td><td>14.2</td><td>23.1</td><td>49.8</td><td>10.1</td><td>19.4</td><td>34.3</td></tr><tr><td rowspan="3">Self-Soup (ours)</td><td>Best ingredient</td><td>40.3</td><td>56.8</td><td>27.7</td><td>24.2</td><td>28.1</td><td>34.1</td><td>8.6</td><td>59.9</td><td>66.4</td><td>49.8</td><td>30.0</td><td>68.1</td><td>28.6</td><td>24.0</td><td>33.6</td><td>6.6</td><td>13.0</td><td>24.8</td><td>43.7</td><td>10.0</td><td>18.5</td><td>33.3</td></tr><tr><td>Greedy Search</td><td>42.1</td><td>59.8</td><td>30.8</td><td>25.2</td><td>29.6</td><td>36.4</td><td>9.2</td><td>59.9</td><td>67.1</td><td>49.8</td><td>29.8</td><td>68.1</td><td>28.7</td><td>26.2</td><td>34.8</td><td>6.6</td><td>13.8</td><td>26.0</td><td>49.5</td><td>10.0</td><td>18.6</td><td>34.4</td></tr><tr><td>Uniform Mix</td><td>44.6</td><td>59.7</td><td>30.6</td><td>26.1</td><td>29.5</td><td>36.4</td><td>8.9</td><td>58.4</td><td>66.9</td><td>50.0</td><td>31.3</td><td>70.2</td><td>29.8</td><td>29.4</td><td>36.2</td><td>6.9</td><td>14.6</td><td>25.6</td><td>52.1</td><td>10.2</td><td>19.9</td><td>35.1</td></tr></table>

(b) Top-1 % accuracy for 15 corruption types and clean data averaged over 21 tasks (VTAB). Self-Soups help most against blur, weather, and digital corruptions. The plain supervised soup (Wortsman et al., 2022) is closest to the MAE stock and is best against noise. 

<table><tr><td rowspan="2">Soup Type</td><td rowspan="2">Mix Method</td><td rowspan="2">Clean</td><td colspan="3">Noise</td><td colspan="4">Blur</td><td colspan="4">Weather</td><td colspan="4">Digital</td><td rowspan="2">Mean</td></tr><tr><td>Gaussian</td><td>Shot</td><td>Impulse</td><td>Defocus</td><td>Glass</td><td>Motion</td><td>Zoom</td><td>Snow</td><td>Frost</td><td>Fog</td><td>Brightness</td><td>Contrast</td><td>Elastic</td><td>Pixel</td><td>JPEG</td></tr><tr><td rowspan="3">Supervised Soup</td><td>Best ingredient</td><td>64.3</td><td>15.6</td><td>17.5</td><td>15.1</td><td>37.5</td><td>34.3</td><td>35.8</td><td>42.2</td><td>30.5</td><td>31.1</td><td>29.8</td><td>55.1</td><td>18.6</td><td>38.4</td><td>35.1</td><td>37.1</td><td>33.6</td></tr><tr><td>Greedy Search</td><td>64.7</td><td>16.5</td><td>18.3</td><td>16.1</td><td>39.1</td><td>34.0</td><td>36.8</td><td>42.4</td><td>31.0</td><td>31.1</td><td>30.2</td><td>55.9</td><td>19.9</td><td>39.5</td><td>35.4</td><td>38.1</td><td>34.3</td></tr><tr><td>Uniform Mix</td><td>63.9</td><td>16.1</td><td>18.3</td><td>16.2</td><td>40.2</td><td>35.4</td><td>38.1</td><td>42.2</td><td>30.4</td><td>31.0</td><td>30.6</td><td>55.8</td><td>21.3</td><td>39.9</td><td>36.3</td><td>38.1</td><td>34.6</td></tr><tr><td rowspan="3">Continued SSL + Supervised Soup</td><td>Best ingredient</td><td>65.7</td><td>14.0</td><td>16.6</td><td>13.9</td><td>36.7</td><td>32.7</td><td>37.6</td><td>42.3</td><td>28.6</td><td>29.0</td><td>29.3</td><td>53.3</td><td>19.6</td><td>38.5</td><td>34.1</td><td>35.3</td><td>33.0</td></tr><tr><td>Greedy Search</td><td>66.2</td><td>15.1</td><td>17.4</td><td>14.8</td><td>37.8</td><td>33.9</td><td>38.2</td><td>42.5</td><td>29.7</td><td>30.2</td><td>30.0</td><td>54.1</td><td>20.2</td><td>39.7</td><td>34.7</td><td>36.2</td><td>33.8</td></tr><tr><td>Uniform Mix</td><td>65.2</td><td>14.8</td><td>17.7</td><td>14.7</td><td>39.1</td><td>35.0</td><td>38.7</td><td>42.8</td><td>30.1</td><td>30.7</td><td>30.9</td><td>55.5</td><td>21.0</td><td>41.1</td><td>35.2</td><td>36.1</td><td>34.3</td></tr><tr><td rowspan="3">Self-Soup (ours)</td><td>Best ingredient</td><td>66.4</td><td>14.0</td><td>16.3</td><td>13.8</td><td>37.1</td><td>30.8</td><td>36.7</td><td>42.9</td><td>30.4</td><td>29.4</td><td>29.0</td><td>55.0</td><td>20.8</td><td>38.8</td><td>34.0</td><td>35.4</td><td>33.2</td></tr><tr><td>Greedy Search</td><td>67.5</td><td>14.3</td><td>17.3</td><td>14.1</td><td>38.8</td><td>32.5</td><td>38.8</td><td>43.7</td><td>30.5</td><td>31.2</td><td>31.1</td><td>56.7</td><td>21.1</td><td>40.6</td><td>34.7</td><td>37.1</td><td>34.4</td></tr><tr><td>Uniform Mix</td><td>65.6</td><td>14.7</td><td>18.0</td><td>14.9</td><td>40.3</td><td>35.4</td><td>39.9</td><td>43.8</td><td>31.2</td><td>31.8</td><td>31.7</td><td>56.3</td><td>22.6</td><td>41.6</td><td>36.4</td><td>37.4</td><td>35.1</td></tr></table>

For the 5 task-specific ingredients, we train for 10K steps with a 4e-5 learning rate on the full training data for each VTAB task. To find mixture coefficients we Self-Season our Self-Soup by minimizing kNN entropy on the mini-VTAB training data without labels. We use k=16 for all tasks as it generally performs well. To put our Self-Seasoning results in context, we do supervised seasoning to find mixture coefficients on the mini-VTAB’s training data with labels. We also compare to the stock, a uniform mix of our ingredients, and the best ingredient chosen based on kNN accuracy on the same labeled data. Supervised soups are not applicable in this setting for transfer without labels. Please see the supplement (§ D) for full details, including PyTorch code for our novel and effective Self-Seasoning method.

Results: Self-Seasoning is competitive. Even without training labels for mixing, our Self-Seasoning is the most accurate on 4/21 tasks. Overall, seasoning with training labels is most effective, and outperforms the best ingredient—on dSpr-Loc-X it is almost twice as accurate (6% vs. 11.6%).

![](images/b073f70914f0c99f63958e0ca3d0f112180f9404b89e07afe55dd85de6f9fb9e.jpg)

<details>
<summary>line</summary>

| λ    | top-1 acc.(%) |
| ---- | ------------- |
| 0.0  | 81.0          |
| 0.5  | 82.0          |
| 1.0  | 81.0          |
</details>

MoCoV3 ingredient

![](images/8a15fd5e1fcc9cc5f93bc634a60630a872c7ee53197116c38f3b60b98141e8aa.jpg)

<details>
<summary>line</summary>

| λ    | LAION-C |
| ---- | ------- |
| 0.0  | 28.0    |
| 0.5  | 30.0    |
| 1.0  | 28.0    |
</details>

MAE ingredient   
Figure 4. Self-Souping is possible and productive for another stock: Franca. (Venkataramanan et al., 2025) Mixing models independently trained using SSL (= ingredients) improves accuracy over ingredients alone. We initialize from Franca’s ViT-B, intertrain using SSL, fine-tune on ImageNet, and mix: λ ∈ 0 → 1.

What about other stocks? There is no reason for our Self-Soups to require an MAE stock. To show another, we intertrain separately using MAE and MoCoV3 on ImageNet— initializing from Franca’s pre-trained ViT-B (an open-source SOTA model (Venkataramanan et al., 2025)). After intertraining, we fine-tune on ImageNet, then mix, showing that LMC holds and gains are similar to an MAE stock (Fig. 4).

Table 5. Self-Seasoning (no training labels) is competitive with supervised seasoning (uses training labels). Our Self-Seasoning finds mixture coefficients for our Self-Soups entirely without labels. For the best ingredient and seasoning runs, we select based on mini-VTAB training data—with labels. A uniform mix of our SSL ingredients provides a nice boost over the stock. Best and 2nd best. 

<table><tr><td></td><td>Caltech101</td><td>CIFAR-10</td><td>CIFAR-100</td><td>DTD</td><td>Flowers102</td><td>Pets</td><td>Sun397</td><td>SVHN</td><td>Camelyon</td><td>EuroSAT</td><td>Resisc45</td><td>Retinopathy</td><td>Clevr-Count</td><td>Clevr-Dist</td><td>DMLab</td><td>dSpr-Loc-X</td><td>dSpr-Loc-Y</td><td>dSpr-Loc-Ori</td><td>KITTI-Dist</td><td>sNORB-Azim</td><td>sNORB-Elev</td></tr><tr><td colspan="8">Ingredient</td><td colspan="14">Self-Seasoning Coefficients</td></tr><tr><td>Stock</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>MAE: default config</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>MMCR: global-only</td><td>0.02</td><td>0</td><td>0</td><td>0.02</td><td>0.02</td><td>0.77</td><td>0.91</td><td>0</td><td>0</td><td>0.01</td><td>0.04</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>MMCR: global+local</td><td>0.78</td><td>0.98</td><td>0.99</td><td>0.8</td><td>0.81</td><td>0.07</td><td>0.09</td><td>0.87</td><td>0.98</td><td>0.99</td><td>0.95</td><td>0</td><td>0.99</td><td>0.98</td><td>0.77</td><td>0</td><td>0</td><td>0</td><td>0.74</td><td>0.77</td><td>0.76</td></tr><tr><td>MoCoV3: temp=0.1</td><td>0.14</td><td>0</td><td>0</td><td>0.17</td><td>0.16</td><td>0.03</td><td>0</td><td>0.07</td><td>0.02</td><td>0</td><td>0</td><td>0</td><td>0.01</td><td>0.01</td><td>0.19</td><td>0.15</td><td>0.99</td><td>0.97</td><td>0.21</td><td>0.13</td><td>0.15</td></tr><tr><td>MoCoV3: temp=1.0</td><td>0.06</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.02</td><td>0.13</td><td>0</td><td>0.05</td><td>0</td><td>0</td><td>0</td><td>0.99</td><td>0.01</td><td>0.01</td><td>0.03</td><td>0.85</td><td>0</td><td>0.03</td><td>0.04</td><td>0.1</td><td>0.09</td></tr><tr><td colspan="8">Method</td><td colspan="14">Method Top-1 % Accuracies</td></tr><tr><td>Stock</td><td>28.6</td><td>30.6</td><td>7.0</td><td>18.5</td><td>14.2</td><td>7.5</td><td>4.8</td><td>18.7</td><td>74.5</td><td>68.5</td><td>31.5</td><td>73.2</td><td>26.4</td><td>27.1</td><td>22.3</td><td>7.1</td><td>12.5</td><td>20.3</td><td>48.5</td><td>11.3</td><td>24.7</td></tr><tr><td>Uniform Soup</td><td>69.7</td><td>66.8</td><td>20.3</td><td>49.3</td><td>48.3</td><td>57.5</td><td>19.5</td><td>42.2</td><td>80.7</td><td>88.5</td><td>57.6</td><td>74.3</td><td>27.4</td><td>27.7</td><td>30.3</td><td>3.8</td><td>6.9</td><td>33.1</td><td>51.8</td><td>10.5</td><td>23.1</td></tr><tr><td>Best Ingredient</td><td>87.6</td><td>78.6</td><td>34.4</td><td>60.4</td><td>76.6</td><td>71.0</td><td>26.3</td><td>78.9</td><td>80.8</td><td>95.2</td><td>74.2</td><td>73.2</td><td>33.7</td><td>32.4</td><td>34.4</td><td>6.0</td><td>14.3</td><td>36.5</td><td>53.6</td><td>11.1</td><td>23.8</td></tr><tr><td>Seasoning</td><td>87.6</td><td>80.9</td><td>37.6</td><td>58.7</td><td>76.0</td><td>70.8</td><td>27.3</td><td>79.0</td><td>81.5</td><td>94.9</td><td>73.7</td><td>74.8</td><td>34.6</td><td>33.8</td><td>35.7</td><td>11.6</td><td>17.9</td><td>37.4</td><td>50.1</td><td>11.1</td><td>23.5</td></tr><tr><td>Self-Seasoning (ours)</td><td>87.8</td><td>76.9</td><td>34.1</td><td>61.0</td><td>78.1</td><td>67.1</td><td>23.6</td><td>77.6</td><td>81.1</td><td>95.3</td><td>73.3</td><td>73.9</td><td>17.5</td><td>29.9</td><td>26.4</td><td>4.1</td><td>7.3</td><td>33.2</td><td>49.2</td><td>9.5</td><td>19.0</td></tr></table>

# 5. Related Work

Sophisticated Mixing or Merging. Model stock (Jang et al., 2024) refines the mixing of soups with layer-wise reweighting using the angles between ingredient parameters. Complementarily, our Self-Souping provides more ingredients that are compatible with more sophisticated mixing. Model merging methods (e.g. TIES-Merging (Yadav et al., 2023) or EMR-Merging (Huang et al., 2024)) combine multiple models, like soups. These methods merge models for different tasks (e.g. text summarization and translation), which do not share an initialization, and may not even share a common architecture. Merging methods do not necessarily maintain the computational cost of their input models, in contrast to soups in general and our Self-Soups in particular.

Multi-Task SSL. Our inter-trainings each optimize their own model with a single self-supervised loss. Multi-task SSL instead jointly optimizes a shared model with multiple self-supervised losses (Doersch & Zisserman, 2017; Bachmann et al., 2022). Although such multi-task optimization can improve on single task optimization, it can require larger-scale computational resources to achieve sufficient batch sizes (at least multiple GPUs, if not multiple machines) and more tuning to balance losses and gradients. While Self-Soups require multiple inter-trainings, each experiment is simpler and smaller-scale. We could do both and mix multi-task SSL ingredients as a Self-Soup by definition.

Domain Adaptation. Our inter-training on shifts is related to unsupervised domain adaptation (UDA): joint optimization on labeled “source” data and unlabeled “target” data (Saenko et al., 2010). However, our inter-trainings are simpler independent training runs, rather than joint optimizations, and are computationally more efficient in only updating on the target data. Test-time adaptation and testtime training (TTA/TTT) make predictions and update on the target data at the same time by online optimization. These updates can alter statistics (Schneider et al., 2020) and model parameters without supervision (Sun et al., 2020; Wang\* et al., 2021). While such test-time updates can be efficient and effective, they need careful tuning and more test-time computation. After inter-training and mixing, Self-Soups are deployed without more test-time computation.

# 6. Discussion

Limitations. Self-Soups enlarge the soup kitchen (SSL methods, hyperparameters, and data) with our new recipes, but there are more to cook. Our largest gains (+7%) need unlabeled target/shifted data, which may not be available. Other gains are modest (≲1%) yet useful, as they do not raise inference costs. There are dozens of SSL algorithms absent from our study, but we choose from 3 different SSL families, so our findings may generalize within families.

Conclusion. We introduce Self-Soupervision, which generalizes model soups to SSL. Self-Souping adds to the menu by harnessing different losses to flavor ingredients from different distributions without requiring labels. We first show that mixing ingredients that differ in their self-supervised training runs (e.g. different losses) is possible and productive. We then show that Self-Soups can improve supervised soups on ImageNet and VTAB. Self-Souping is most helpful when facing distribution shifts—and especially, when unlabeled shifted data is available for preparing ingredients. We also introduce Self-Seasoning, which learns ingredient mixtures for a task without training labels. We hope our recipes earn a spot in your cookbook and inspire new ones.

# Acknowledgements

We thank Simon Ghyselincks, Pritam Sarkar, and Pierre Lardet for pre-reviewing the manuscript. AF is primarily supported by an NSERC PGS-D scholarship. ES is supported by a Canada CIFAR AI Chair. Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute.

# Impact Statement

Our Self-Soupervision method and ingredients aim to produce more machine learning models and more accurate models. Improving the generalization and robustness of machine learning models contributes to their accuracy and sound deployment in practice. We evaluate on standard benchmarks for visual recognition, and so do not alter the choice of tasks for better or worse. Our use of self-supervised learning without labels is potentially more general and feasible for a broader set of applications, because self-supervised ingredients can be inter-trained without the cost of annotation, though our soups do still require the cost of computation. The workflow of inter-training, fine-tuning, and mixing is potentially more accessible and collaborative, because contributing an inter-training or fine-tuning and evaluating a mixture are less computationally intensive than pre-training. We intend for Self-Soupervision to enable more of the community to engage in machine learning.

# References

Ablin, P., Katharopoulos, A., Seto, S., and Grangier, D. Soup-of-experts: Pretraining specialist models via parameters averaging. In Forty-second International Conference on Machine Learning, 2025. URL https: //openreview.net/forum?id=MFNIka7nx0.   
Aminbeidokhti, M., Roy, S., Granger, E., Ricci, E., and Pedersoli, M. LT-soups: Bridging head and tail classes via subsampled model soups. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum? id=tiEsJtw3FH.   
Bachmann, R., Mizrahi, D., Atanov, A., and Zamir, A. MultiMAE: Multi-modal multi-task masked autoencoders. In European Conference on Computer Vision, pp. 348–367. Springer, 2022.   
Balestriero, R., Ibrahim, M., Sobal, V., Morcos, A., Shekhar, S., Goldstein, T., Bordes, F., Bardes, A., Mialon, G., Tian, Y., Schwarzschild, A., Wilson, A. G., Geiping, J., Garrido, Q., Fernandez, P., Bar, A., Pirsiavash, H., LeCun, Y., and Goldblum, M. A cookbook of self-supervised

learning. 2023. URL https://arxiv.org/abs/ 2304.12210.

Bardes, A., Ponce, J., and LeCun, Y. Vicreg: Varianceinvariance-covariance regularization for self-supervised learning. arXiv preprint arXiv:2105.04906, 2021.

Beattie, C., Leibo, J. Z., Teplyashin, D., Ward, T., Wainwright, M., Kuttler, H., Lefrancq, A., Green, S., Vald ¨ es, ´ V., Sadik, A., et al. Deepmind lab. arXiv preprint arXiv:1612.03801, 2016.

Beyer, L., Henaff, O. J., Kolesnikov, A., Zhai, X., and ´ van den Oord, A. Are we done with imagenet?, 2020. URL https://arxiv.org/abs/2006.07159.

Biggs, B., Seshadri, A., Zou, Y., Jain, A., Golatkar, A., Xie, Y., Achille, A., Swaminathan, A., and Soatto, S. Diffusion soup: Model merging for text-to-image diffusion models. In European Conference on Computer Vision, pp. 257– 274. Springer, 2024.

Chen, M., Jiang, M., Zhang, X., Dou, Q., Wang, Z., and Li, X. Local superior soups: A catalyst for model merging in cross-silo federated learning. Advances in Neural Information Processing Systems, 37:20858–20886, 2024.

Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pp. 1597–1607. PmLR, 2020.

Chen, X., Xie, S., and He, K. An empirical study of training self-supervised vision transformers. arXiv preprint arXiv:2104.02057, 2021.

Cheng, G., Han, J., and Lu, X. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the IEEE, 105(10):1865–1883, 2017.

Chronopoulou, A., Peters, M. E., Fraser, A., and Dodge, J. Adaptersoup: Weight averaging to improve generalization of pretrained language models. arXiv preprint arXiv:2302.07027, 2023.

Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., and Vedaldi, A. Describing textures in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3606–3613, 2014.

Croce, F., Rebuffi, S.-A., Shelhamer, E., and Gowal, S. Seasoning model soups for robustness to adversarial and natural distribution shifts. 2023.

Doersch, C. and Zisserman, A. Multi-task self-supervised visual learning. In Proceedings of the IEEE international conference on computer vision, pp. 2051–2060, 2017.

Fei-Fei, L., Fergus, R., and Perona, P. One-shot learning of object categories. IEEE transactions on pattern analysis and machine intelligence, 28(4):594–611, 2006.   
Frankle, J., Dziugaite, G. K., Roy, D., and Carbin, M. Linear mode connectivity and the lottery ticket hypothesis. In International Conference on Machine Learning (ICML), 2020.   
Fuller, A., Kyrollos, D., Yassin, Y., and Green, J. R. Lookhere: Vision transformers with directed attention generalize and extrapolate. In Neural Information Processing Systems (NeurIPS), 2024.   
Geiger, A., Lenz, P., Stiller, C., and Urtasun, R. Vision meets robotics: The kitti dataset. The international journal of robotics research, 32(11):1231–1237, 2013.   
Gururangan, S., Marasovic, A., Swayamdipta, S., Lo, K., ´ Beltagy, I., Downey, D., and Smith, N. A. Don’t stop pretraining: Adapt language models to domains and tasks. arXiv preprint arXiv:2004.10964, 2020.   
He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick,´ R. Masked autoencoders are scalable vision learners. In Conference on Computer Vision and Pattern Recognition (CVPR), 2022.   
Helber, P., Bischke, B., Dengel, A., and Borth, D. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019.   
Hendrycks, D. and Dietterich, T. Benchmarking neural network robustness to common corruptions and perturbations. International Conference on Learning Representations (ICLR), 2019.   
Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F., Dorundo, E., Desai, R., Zhu, T., Parajuli, S., Guo, M., Song, D., Steinhardt, J., and Gilmer, J. The many faces of robustness: A critical analysis of out-of-distribution generalization. International Conference on Computer Vision (ICCV), 2021a.   
Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., and Song, D. Natural adversarial examples. Conference on Computer Vision and Pattern Recognition (CVPR), 2021b.   
Huang, C., Ye, P., Chen, T., He, T., Yue, X., and Ouyang, W. Emr-merging: Tuning-free high-performance model merging. Advances in Neural Information Processing Systems, 37:122741–122769, 2024.

Jain, S., Addepalli, S., Sahu, P. K., Dey, P., and Babu, R. V. Dart: Diversify-aggregate-repeat training improves generalization of neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16048–16059, 2023.   
Jang, D.-H., Yun, S., and Han, D. Model stock: All we need is just a few fine-tuned models. In Proceedings of the European Conference on Computer Vision, 2024.   
Jang, J., Kim, S., Lin, B. Y., Wang, Y., Hessel, J., Zettlemoyer, L., Hajishirzi, H., Choi, Y., and Ammanabrolu, P. Personalized soups: Personalized large language model alignment via post-hoc parameter merging. arXiv preprint arXiv:2310.11564, 2023.   
Johnson, J., Hariharan, B., Van Der Maaten, L., Fei-Fei, L., Lawrence Zitnick, C., and Girshick, R. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2901–2910, 2017.   
Kaggle and EyePacs. Kaggle diabetic retinopathy detection, 2015. URL https://www.kaggle.com/c/ diabetic-retinopathy-detection/data.   
Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images.(2009), 2009.   
Kumar, A., Raghunathan, A., Jones, R. M., Ma, T., and Liang, P. Fine-tuning can distort pretrained features and underperform out-of-distribution. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=UYneFzXSJWh.   
Kyrollos, D. G., Fuller, A., Greenwood, K., Harrold, J., and Green, J. R. Under the cover infant pose estimation using multimodal data. IEEE Transactions on Instrumentation and Measurement, 72:1–12, 2023.   
LeCun, Y., Huang, F. J., and Bottou, L. Learning methods for generic object recognition with invariance to pose and lighting. In Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2004. CVPR 2004., volume 2, pp. II–104. IEEE, 2004.   
Li, F., Klein, T., Brendel, W., Geirhos, R., and Zimmermann, R. S. Laion-c: An out-of-distribution benchmark for web-scale vision models. In International Conference on Machine Learning (ICML), 2025.   
Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview. net/forum?id=Bkg6RiCqY7.

Matthey, L., Higgins, I., Hassabis, D., and Lerchner, A. dsprites: Disentanglement testing sprites dataset, 2017.   
Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., Ng, A. Y., et al. Reading digits in natural images with unsupervised feature learning. In NIPS workshop on deep learning and unsupervised feature learning, volume 2011, pp. 7. Granada, 2011.   
Nilsback, M.-E. and Zisserman, A. Automated flower classification over a large number of classes. In 2008 Sixth Indian conference on computer vision, graphics & image processing, pp. 722–729. IEEE, 2008.   
Niu, S., Wu, J., Zhang, Y., Wen, Z., Chen, Y., Zhao, P., and Tan, M. Towards stable test-time adaptation in dynamic wild world. In The Eleventh International Conference on Learning Representations, 2023. URL https:// openreview.net/forum?id=g2YraF75Tj.   
Park, N., Kim, W., Heo, B., Kim, T., and Yun, S. What do self-supervised vision transformers learn? In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/ forum?id=azCKuYyS74.   
Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. Cats and dogs. In 2012 IEEE conference on computer vision and pattern recognition, pp. 3498–3505. IEEE, 2012.   
Rame, A., Kirchmeyer, M., Rahier, T., Rakotomamonjy, A., ´ Gallinari, P., and Cord, M. Diverse weight averaging for out-of-distribution generalization. In NeurIPS, 2022.   
Rame, A., Couairon, G., Shukor, M., Dancette, C., Gaya, ´ J.-B., Soulier, L., and Cord, M. Rewarded soups: towards pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards. In Neural Information Processing Systems (NeurIPS), 2023.   
Rame, A., Ahuja, K., Zhang, J., Cord, M., Bottou, L., and ´ Lopez-Paz, D. Model ratatouille: Recycling diverse models for out-of-distribution generalization. Conference on Computer Vision and Pattern Recognition (CVPR), 2023.   
Recht, B., Roelofs, R., Schmidt, L., and Shankar, V. Do imagenet classifiers generalize to imagenet?, 2019. URL https://arxiv.org/abs/1902.10811.   
Reed, C. J., Yue, X., Nrusimha, A., Ebrahimi, S., Vijaykumar, V., Mao, R., Li, B., Zhang, S., Guillory, D., Metzger, S., et al. Self-supervised pretraining improves selfsupervised pretraining. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 2584–2594, 2022.

Rodas, B., Montesino, N., Ambsdorf, J., Klindt, D., and Balestriero, R. Diet-cp: Lightweight and data efficient self supervised continued pretraining. arXiv preprint arXiv:2509.06990, 2025.   
Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115: 211–252, 2015.   
Saenko, K., Kulis, B., Fritz, M., and Darrell, T. Adapting visual category models to new domains. In European conference on computer vision, pp. 213–226. Springer, 2010.   
Schneider, S., Rusak, E., Eck, L., Bringmann, O., Brendel, W., and Bethge, M. Improving robustness against common corruptions by covariate shift adaptation. In NeurIPS, volume 33, 2020.   
Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A. A., and Hardt, M. Test-time training for out-of-distribution generalization. In ICLR, 2020.   
Touvron, H., Cord, M., and Jegou, H. Deit iii: Revenge of ´ the vit. In European conference on computer vision, pp. 516–533. Springer, 2022.   
Veeling, B. S., Linmans, J., Winkens, J., Cohen, T., and Welling, M. Rotation equivariant cnns for digital pathology. In International Conference on Medical image computing and computer-assisted intervention, pp. 210–218. Springer, 2018.   
Venkataramanan, S., Pariza, V., Salehi, M., Knobel, L., Gidaris, S., Ramzi, E., Bursuc, A., and Asano, Y. M. Franca: Nested matryoshka clustering for scalable visual representation learning. arXiv preprint arXiv:2507.14137, 2025.   
Wang\*, D., Shelhamer\*, E., Liu, S., Olshausen, B., and Darrell, T. Tent: Fully test-time adaptation by entropy minimization. In ICLR, 2021.   
Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., and Schmidt, L. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International Conference on Machine Learning (ICML), 2022.   
Xiao, J., Hays, J., Ehinger, K. A., Oliva, A., and Torralba, A. Sun database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE computer society conference on computer vision and pattern recognition, pp. 3485–3492. IEEE, 2010.

Xie, Z., Zhang, Z., Cao, Y., Lin, Y., Bao, J., Yao, Z., Dai, Q., and Hu, H. Simmim: A simple framework for masked image modeling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9653–9663, 2022.   
Yadav, P., Tam, D., Choshen, L., Raffel, C., and Bansal, M. TIES-merging: Resolving interference when merging models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=xtaX3WyCj1.   
Yerxa, T., Kuang, Y., Simoncelli, E., and Chung, S. Learning efficient coding of natural images with maximum manifold capacity representations. Advances in Neural Information Processing Systems, 36:24103–24128, 2023.   
Zhai, X., Puigcerver, J., Kolesnikov, A., Ruyssen, P., Riquelme, C., Lucic, M., Djolonga, J., Pinto, A. S., Neumann, M., Dosovitskiy, A., Beyer, L., Bachem, O., Tschannen, M., Michalski, M., Bousquet, O., Gelly, S., and Houlsby, N. A large-scale study of representation learning with the visual task adaptation benchmark, 2020. URL https://arxiv.org/abs/1910.04867.

# A. VTAB References.

For clarity and credit, we reference the original datasets that went into the VTAB collection: Caltech101 (Fei-Fei et al., 2006), CIFAR-10/100 (Krizhevsky et al., 2009), DTD (Cimpoi et al., 2014), Flowers102 (Nilsback & Zisserman, 2008), Pets (Parkhi et al., 2012), Sun397 (Xiao et al., 2010), SVHN (Netzer et al., 2011), EuroSAT (Helber et al., 2019), Resisc45 (Cheng et al., 2017), Patch Camelyon (Veeling et al., 2018), Retinopathy (Kaggle & EyePacs, 2015), Clevr (Johnson et al., 2017), dSprites (Matthey et al., 2017), SmallNORB (LeCun et al., 2004), DMLab (Beattie et al., 2016), and KITTI (Geiger et al., 2013).

# B. More training details.

All runs. We always use: 0.01 weight decay, the AdamW optimizer (Loshchilov & Hutter, 2019), warmup for 10% of the steps and cooldown via cosine decay, and 3-Augment (Touvron et al., 2022) for data augmentation.

All SSL inter-training runs. For MAE inter-training, we initialize the decoder with the pre-trained MAE decoder. For MoCoV3 and MMCR inter-training, we use a simple 2-layer MLP as the projection head, we do not use an exponential moving average to compute target embeddings, and we warmup the head for only 1% of the steps (chosen so the head learns more quickly than the backbone). We choose these settings to keep it simple and do not tune them. Before mixing or fine-tuning the ingredients, we discard all algorithm-specific heads and only use the backbones/encoders.

All ImageNet fine-tuning runs. We fine-tune for 10 epochs on ImageNet-1K (following the original Model Soups (Wortsman et al., 2022)). We sweep learning rates {6e-5, 8e-5, 1e-4, 1.5e-4} with a 128 batch size. We always use LPFT, which initializes fine-tuning from the linear probed solution (Kumar et al., 2022) (including when fine-tuning on VTAB).

ImageNet: §4.2. For model inter-training, we train for 5 epochs on ImageNet-1K with a 256 batch size and a 1e-5 learning rate. For MAE, we use a 90% masking ratio and 1 decoder layer. For MoCoV3, we use a 1.0 temperature. For MMCR, we use both global and local losses. These SSL-algorithm hyperparameters were mostly chosen arbitrarily, in our experience different choices achieves the same results.

Test-set inter-training: §4.3. For model inter-training, we train for 100K steps with a 128 batch size using the default MAE settings (i.e. 75% masking ratio and 8 decoder layers), and sweep learning rates {1e-5, 2e-5, 3e-5, 4e-5}.

Test-time adaptation: Tab. 3. We sweep base learning rates {1e-5, 3e-5, 5e-5, 8e-5, 1e-4, 3e-4, 1e-3, 3e-3}. A 5e-5 base learning rate is best. We use a 128 batch size, which sets the actual learning rate: lr = (base lr/64) · batch size

# C. Data and Code.

# Data.

https://huggingface.co/datasets/antofuller/mini-VTAB https://huggingface.co/datasets/antofuller/mini-VTAB-corruptions

# Code.

https://github.com/antofuller/self\_soupervision

# D. Seasoning: §4.5.

For supervised seasoning, we try 1K random samples of mixture coefficients (Dirichlet distribution with concentration = 1), and pick the best on kNN accuracy on the mini-VTAB training set for each task. For the “best ingredient”, we also pick it based on kNN training accuracy for each task. For Self-Seasoning, we train the mixture coefficients with AdamW for 100 epochs starting with a 0.1 learning rate and cosine-decay it to 0.01 with a 256 batch size (we did not tune this because it worked well enough). We initialize all 6 parameters to 0 and apply a softmax so the coefficients sum to 1. We use k=16 and a 0.07 temperature.

```python
def knn_inbatch_neighbor_entropy(Z, k=16, T=0.07):
    B = Z.shape[0]

    # L2 normalize embeddings
    Z = F.normalize(Z, dim=1, p=2)  # [B, D]

    # Compute pairwise cosine similarities
    S = Z @ Z.T  # [B, B]
    S.fill_diagonal_(float("-inf"))  # mask self

    # Select top-k neighbors
    S_topk, _ = S.topk(k, dim=1)  # [B, k]

    # Compute neighbor distribution
    p = (S_topk / T).softmax(dim=1)  # [B, k]

    #Compute entropy
    H = -(p * p.clamp(min=1e-12).log()).sum(dim=1)

    return H.mean() 
```