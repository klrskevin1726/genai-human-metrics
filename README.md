# GenAI Human Similarity Research

Research project focused on identifying interpretable differences and similarities between human-written text and its AI-generated counterpart.

The project explores lexical and structural writing characteristics to determine which measurable signals are useful for comparing Human and GenAI writing.

The long-term goal is not simply to build a binary AI detector. Instead, the project aims to develop an interpretable similarity score that compares a human-written text with its corresponding AI-generated version.

---

## Research Objective

The primary objective is to develop an interpretable similarity metric between:

- a human-written text
- its AI-generated counterpart

Before implementing the similarity score, supervised and unsupervised machine learning experiments are being used to identify writing characteristics that provide meaningful signal.

These experiments help answer questions such as:

- Which structural characteristics differ between Human and AI writing?
- Which features provide the strongest predictive signal?
- Do structural characteristics generalize better than lexical TF-IDF features?
- Which signals may later contribute to an interpretable similarity score?

---

## Dataset

The current dataset contains:

- 100 valid text samples
- 50 Human samples
- 50 AI samples

Each sample is associated with an `assignment_id`.

Group-aware splitting is used so samples from the same assignment do not appear in both training and testing sets.

This reduces the risk of data leakage caused by shared topic, vocabulary, instructions, or assignment context.

---

## Current Structural Features

Five interpretable structural and stylistic features are currently extracted from each text.

### Average Sentence Length

`avg_sentence_len`

Average number of words per sentence.

### Sentence Length Standard Deviation

`sentence_len_std`

Measures how much sentence length varies throughout the text.

### Type-Token Ratio

`ttr`

A lexical diversity measure calculated as:

`unique words / total words`

### Average Word Length

`avg_word_length`

Average number of characters per word.

### Punctuation Rate

`punctuation_rate`

Number of punctuation characters divided by total text length.

---

## Observed Feature Means

Average values by class:

| Feature | AI | Human |
|---|---:|---:|
| Average Sentence Length | 16.296569 | 18.829350 |
| Sentence Length Std | 5.948634 | 9.166399 |
| Type-Token Ratio | 0.615013 | 0.460232 |
| Average Word Length | 5.344221 | 4.500071 |
| Punctuation Rate | 0.019811 | 0.019602 |

Current observations:

- AI-generated samples use longer words on average.
- Human samples show substantially greater sentence-length variability.
- AI samples show higher Type-Token Ratio in the current dataset.
- Human samples have somewhat longer sentences on average.
- Punctuation rate is almost identical between both classes.

---

## Supervised Learning Experiments

### Initial TF-IDF Baseline

Model:

- TF-IDF
- unigrams + bigrams
- Logistic Regression
- group-aware train/test split

Initial single-split accuracy:

**90.00%**

Because this result was based on only one split, it was later evaluated using multiple grouped splits.

### Initial Structural Model

Model:

- five structural features
- StandardScaler
- Logistic Regression

Initial single-split accuracy:

**96.67%**

On the same split:

| Model | Accuracy |
|---|---:|
| TF-IDF | 90.00% |
| Structural | 96.67% |

---

## Structural Feature Coefficients

Logistic Regression coefficients:

| Feature | Coefficient |
|---|---:|
| avg_word_length | -2.086789 |
| ttr | -0.891447 |
| punctuation_rate | -0.410475 |
| avg_sentence_len | 0.192955 |
| sentence_len_std | 1.145991 |

With scikit-learn class ordering:

`['ai', 'human']`

Positive coefficients push predictions toward Human, while negative coefficients push predictions toward AI.

Current interpretation:

- `avg_word_length` has the largest coefficient magnitude and pushes predictions toward AI.
- `sentence_len_std` strongly pushes predictions toward Human.
- `ttr` pushes predictions toward AI in the current dataset.
- `punctuation_rate` currently appears relatively weak.
- `avg_sentence_len` has a smaller positive contribution toward Human.

Coefficient magnitude alone is not treated as sufficient evidence of feature importance because features may contain redundant or correlated information.

---

## Feature Ablation

A leave-one-feature-out experiment was first performed using a single grouped split.

Results:

| Removed Feature | Accuracy |
|---|---:|
| avg_sentence_len | 96.67% |
| sentence_len_std | 96.67% |
| ttr | 96.67% |
| avg_word_length | 90.00% |
| punctuation_rate | 96.67% |

On this split, removing `avg_word_length` produced the only substantial decrease in accuracy.

---

## Multi-Split Structural Evaluation

To test whether the findings were stable, the structural model was evaluated across 20 group-aware train/test splits.

Protocol:

- `GroupShuffleSplit`
- 20 random seeds
- `test_size = 0.30`
- grouping variable: `assignment_id`
- scaler fitted only on training data
- Logistic Regression trained independently for every split

Full structural model:

**94.00% ± 3.17%**

Feature ablation results:

| Removed Feature | Mean Accuracy |
|---|---:|
| avg_sentence_len | 93.17% ± 3.82% |
| sentence_len_std | 93.83% ± 3.47% |
| ttr | 94.50% ± 3.63% |
| avg_word_length | 85.00% ± 6.35% |
| punctuation_rate | 94.00% ± 3.35% |

Key observations:

- Removing `avg_word_length` causes the largest performance decrease.
- Mean accuracy drops from **94.00% to 85.00%**.
- Standard deviation increases from **3.17% to 6.35%**.
- Removing `ttr` slightly improves mean accuracy, suggesting possible redundancy or noise.
- Removing `punctuation_rate` does not change mean accuracy.

---

## Average Word Length as a Single Feature

To distinguish feature importance from feature sufficiency, a model was trained using only:

`avg_word_length`

Using the same 20 grouped splits:

**84.83% ± 3.97%**

Comparison:

| Model | Accuracy |
|---|---:|
| Average word length only | 84.83% ± 3.97% |
| Without average word length | 85.00% ± 6.35% |
| All structural features | 94.00% ± 3.17% |

This suggests that average word length contains substantial predictive signal, but it is not sufficient by itself to reproduce the full structural model performance.

The strongest results come from combining complementary structural characteristics.

---

## Fair Multi-Split TF-IDF Evaluation

TF-IDF was evaluated using the same 20 group-aware splits as the structural model.

For every split:

1. Training and testing groups were selected using `GroupShuffleSplit`.
2. `TfidfVectorizer` was fitted only on training text.
3. Test text was transformed using the training vocabulary.
4. Logistic Regression was trained.
5. Test accuracy was recorded.

Result:

**87.83% ± 4.36%**

Fair comparison:

| Representation | Accuracy |
|---|---:|
| Structural Features | 94.00% ± 3.17% |
| TF-IDF | 87.83% ± 4.36% |
| Average Word Length Only | 84.83% ± 3.97% |

Under the same evaluation protocol, the structural representation currently provides higher mean accuracy and lower variation than TF-IDF.

---

## Unsupervised Learning Experiments

Unsupervised experiments were also performed using KMeans clustering.

Labels were not used during clustering.

Known Human and AI labels were used only afterward to evaluate how closely the discovered clusters corresponded to the true classes.

### TF-IDF Clustering

Configuration:

- TF-IDF
- unigrams + bigrams
- KMeans
- 2 clusters

TF-IDF matrix:

`(100, 33926)`

Cluster distribution:

- Cluster 0: 23
- Cluster 1: 77

Clusters compared with known labels:

| Cluster | AI | Human |
|---|---:|---:|
| Cluster 0 | 12 | 11 |
| Cluster 1 | 38 | 39 |

Silhouette Score:

**0.01500594346**

The near-zero silhouette score suggests that TF-IDF does not produce meaningful natural Human/AI cluster separation in the current dataset.

### Structural Feature Clustering

The five structural features were standardized and clustered using KMeans.

Structural feature matrix:

`(100, 5)`

Clusters compared with known labels:

| Cluster | AI | Human |
|---|---:|---:|
| Structural Cluster 0 | 50 | 7 |
| Structural Cluster 1 | 0 | 43 |

Structural Silhouette Score:

**0.31734383046700787**

Adjusted Rand Index:

**0.7369954580512469**

Normalized Mutual Information:

**0.698650799163015**

The structural feature representation produces substantially stronger natural separation than TF-IDF.

One structural cluster contains all 50 AI samples and 7 Human samples, while the second contains 43 Human samples and no AI samples.

---

## Current Evidence

The strongest individual structural criterion discovered so far is average word length.

Three observations support its importance.

### Class-Level Difference

- AI: **5.344 characters per word**
- Human: **4.500 characters per word**

### Logistic Regression Coefficient

`avg_word_length = -2.086789`

This is the largest absolute standardized structural coefficient.

### Multi-Split Ablation

Full model:

**94.00% ± 3.17%**

Without `avg_word_length`:

**85.00% ± 6.35%**

Additionally:

`avg_word_length` only:

**84.83% ± 3.97%**

Current interpretation:

Average word length is the strongest individual identifying structural criterion discovered so far in this dataset.

However, it is not sufficient by itself.

The evidence indicates that multiple structural characteristics provide complementary information.

---

## Current Conclusions

Based on the experiments completed so far:

1. Human and AI writing in the current dataset exhibit measurable structural differences.

2. Structural features currently provide stronger group-aware classification performance than TF-IDF.

3. Average word length is the strongest individual identifying structural criterion discovered so far.

4. Average word length alone does not explain the complete performance of the structural classifier.

5. Multiple structural features appear to provide complementary information.

6. Structural features also provide substantially stronger unsupervised Human/AI separation than TF-IDF.

7. These results support continuing toward an interpretable similarity metric rather than relying exclusively on high-dimensional lexical representations.

---

## Limitations

The current results should be interpreted carefully.

### Dataset Size

The dataset contains only 100 samples.

Results should therefore be considered preliminary.

### Generalization

Current findings do not establish that the same patterns will generalize to:

- different writing assignments
- different domains
- different populations
- different AI models
- different prompting strategies
- edited AI-generated text
- future language models

### Feature Dependence

Some structural features may contain overlapping or redundant information.

Coefficient magnitude should therefore be interpreted together with feature ablation and other experiments.

### Statistical Significance

Differences in average performance have been observed, but formal hypothesis testing has not yet been performed.

The results should not currently be described as statistically significant.

### Classification Is Not the Final Goal

Classification experiments are being used primarily to understand which characteristics provide useful information.

The primary research objective remains the development of an interpretable similarity metric between a human-written text and its corresponding AI-generated counterpart.

---

## Next Phase: Human–AI Similarity Score

The classification exploration phase is now considered sufficiently mature for the current project timeline.

The next phase will develop a prototype similarity metric between:

**Human-written text + AI-generated counterpart → Interpretable Similarity Score**

The initial system will prioritize:

- interpretability
- simplicity
- reproducibility
- modularity
- explainable component scores

Potential pipeline:

Human Text + AI Text  
→ Feature Extraction  
→ Feature-by-feature differences  
→ Normalization  
→ Individual similarity components  
→ Combined similarity score  
→ Human-readable explanation

New features or additional experiments will be introduced only when they address an observed limitation of the similarity prototype.

---

## Project Structure

    genai-human-metrics/
    │
    ├── data/
    │   └── dataset.csv
    │
    ├── src/
    │   ├── structural_features.py
    │   ├── train.py
    │   └── unsupervised.py
    │
    ├── requirements.txt
    └── README.md

---

## Current Status

### Completed

- Dataset preparation
- Group-aware supervised evaluation
- TF-IDF baseline
- Structural feature extraction
- Structural Logistic Regression
- Feature coefficient analysis
- Feature ablation
- 20-split grouped robustness evaluation
- Single-feature average word length experiment
- Fair multi-split TF-IDF evaluation
- TF-IDF unsupervised clustering
- Structural unsupervised clustering

### Current Phase

**Design and implementation of the Human–AI similarity score**