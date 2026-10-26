from pathlib import Path
import statistics

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from structural_features import (
    sentence_stats,
    lexical_diversity,
    avg_word_length,
    punctuation_rate,
)


# ============================================================
# Dataset
# ============================================================

DATASET = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"

df = pd.read_csv(DATASET)
df = df[df["label"].isin(["human", "ai"])]


# ============================================================
# Structural Feature Extraction
# ============================================================

df["punctuation_rate"] = df["text"].apply(punctuation_rate)
df["ttr"] = df["text"].apply(lexical_diversity)
df["avg_word_length"] = df["text"].apply(avg_word_length)

df["avg_sentence_len"], df["sentence_len_std"] = zip(
    *df["text"].apply(sentence_stats)
)

feature_columns = [
    "avg_sentence_len",
    "sentence_len_std",
    "ttr",
    "avg_word_length",
    "punctuation_rate",
]


# ============================================================
# Initial Group-Aware Train/Test Split
# ============================================================

X = df["text"]
y = df["label"]
groups = df["assignment_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups,
    )
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# ============================================================
# Initial TF-IDF Baseline
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

tfidf_model = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

tfidf_model.fit(
    X_train_tfidf,
    y_train,
)

tfidf_predictions = tfidf_model.predict(
    X_test_tfidf
)

tfidf_accuracy = accuracy_score(
    y_test,
    tfidf_predictions,
)


# ============================================================
# Initial Structural Model
# ============================================================

X_structural = df[feature_columns]

X_structural_train = X_structural.iloc[train_idx]
X_structural_test = X_structural.iloc[test_idx]

structural_scaler = StandardScaler()

X_structural_train_scaled = structural_scaler.fit_transform(
    X_structural_train
)

X_structural_test_scaled = structural_scaler.transform(
    X_structural_test
)

structural_model = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

structural_model.fit(
    X_structural_train_scaled,
    y_train,
)

structural_predictions = structural_model.predict(
    X_structural_test_scaled
)

structural_accuracy = accuracy_score(
    y_test,
    structural_predictions,
)

structural_coef = pd.Series(
    structural_model.coef_[0],
    index=feature_columns,
)


# ============================================================
# Single-Split Feature Ablation
# ============================================================

single_split_ablation_results = {}

for feature_to_remove in feature_columns:

    reduced_features = [
        feature
        for feature in feature_columns
        if feature != feature_to_remove
    ]

    X_reduced = df[reduced_features]

    X_reduced_train = X_reduced.iloc[train_idx]
    X_reduced_test = X_reduced.iloc[test_idx]

    reduced_scaler = StandardScaler()

    X_reduced_train_scaled = reduced_scaler.fit_transform(
        X_reduced_train
    )

    X_reduced_test_scaled = reduced_scaler.transform(
        X_reduced_test
    )

    reduced_model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    reduced_model.fit(
        X_reduced_train_scaled,
        y_train,
    )

    reduced_predictions = reduced_model.predict(
        X_reduced_test_scaled
    )

    reduced_accuracy = accuracy_score(
        y_test,
        reduced_predictions,
    )

    single_split_ablation_results[
        feature_to_remove
    ] = reduced_accuracy


# ============================================================
# Multi-Split Structural Evaluation + Feature Ablation
# ============================================================

multi_split_results = []

ablation_results = {
    feature: []
    for feature in feature_columns
}

for seed in range(20):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed,
    )

    train_idx, test_idx = next(
        splitter.split(
            df["text"],
            df["label"],
            groups=df["assignment_id"],
        )
    )

    y_train = df["label"].iloc[train_idx]
    y_test = df["label"].iloc[test_idx]

    # Full structural model
    X_structural = df[feature_columns]

    X_train_structural = X_structural.iloc[train_idx]
    X_test_structural = X_structural.iloc[test_idx]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_structural
    )

    X_test_scaled = scaler.transform(
        X_test_structural
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    predictions = model.predict(
        X_test_scaled
    )

    full_accuracy = accuracy_score(
        y_test,
        predictions,
    )

    multi_split_results.append(
        full_accuracy
    )

    # Leave-one-feature-out ablation
    for feature_to_remove in feature_columns:

        reduced_features = [
            feature
            for feature in feature_columns
            if feature != feature_to_remove
        ]

        X_reduced = df[reduced_features]

        X_reduced_train = X_reduced.iloc[train_idx]
        X_reduced_test = X_reduced.iloc[test_idx]

        reduced_scaler = StandardScaler()

        X_reduced_train_scaled = reduced_scaler.fit_transform(
            X_reduced_train
        )

        X_reduced_test_scaled = reduced_scaler.transform(
            X_reduced_test
        )

        reduced_model = LogisticRegression(
            max_iter=2000,
            random_state=42,
        )

        reduced_model.fit(
            X_reduced_train_scaled,
            y_train,
        )

        reduced_predictions = reduced_model.predict(
            X_reduced_test_scaled
        )

        reduced_accuracy = accuracy_score(
            y_test,
            reduced_predictions,
        )

        ablation_results[
            feature_to_remove
        ].append(
            reduced_accuracy
        )


full_mean = (
    sum(multi_split_results)
    / len(multi_split_results)
)

full_std = statistics.stdev(
    multi_split_results
)


# ============================================================
# Average Word Length Only
# ============================================================

single_feature_results = []

for seed in range(20):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed,
    )

    train_idx, test_idx = next(
        splitter.split(
            df[["avg_word_length"]],
            df["label"],
            groups=df["assignment_id"],
        )
    )

    X_single = df[["avg_word_length"]]

    X_single_train = X_single.iloc[train_idx]
    X_single_test = X_single.iloc[test_idx]

    y_train = df["label"].iloc[train_idx]
    y_test = df["label"].iloc[test_idx]

    single_scaler = StandardScaler()

    X_single_train_scaled = single_scaler.fit_transform(
        X_single_train
    )

    X_single_test_scaled = single_scaler.transform(
        X_single_test
    )

    single_model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    single_model.fit(
        X_single_train_scaled,
        y_train,
    )

    single_predictions = single_model.predict(
        X_single_test_scaled
    )

    single_accuracy = accuracy_score(
        y_test,
        single_predictions,
    )

    single_feature_results.append(
        single_accuracy
    )


single_mean = (
    sum(single_feature_results)
    / len(single_feature_results)
)

single_std = statistics.stdev(
    single_feature_results
)


# ============================================================
# Multi-Split TF-IDF Evaluation
# ============================================================

tfidf_multi_split_results = []

for seed in range(20):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed,
    )

    train_idx, test_idx = next(
        splitter.split(
            df["text"],
            df["label"],
            groups=df["assignment_id"],
        )
    )

    X_train_text = df["text"].iloc[train_idx]
    X_test_text = df["text"].iloc[test_idx]

    y_train = df["label"].iloc[train_idx]
    y_test = df["label"].iloc[test_idx]

    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
    )

    X_train_tfidf_split = tfidf_vectorizer.fit_transform(
        X_train_text
    )

    X_test_tfidf_split = tfidf_vectorizer.transform(
        X_test_text
    )

    tfidf_split_model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    tfidf_split_model.fit(
        X_train_tfidf_split,
        y_train,
    )

    tfidf_split_predictions = tfidf_split_model.predict(
        X_test_tfidf_split
    )

    tfidf_split_accuracy = accuracy_score(
        y_test,
        tfidf_split_predictions,
    )

    tfidf_multi_split_results.append(
        tfidf_split_accuracy
    )


tfidf_mean = (
    sum(tfidf_multi_split_results)
    / len(tfidf_multi_split_results)
)

tfidf_std = statistics.stdev(
    tfidf_multi_split_results
)


# ============================================================
# Results
# ============================================================

print("\nFeature Ablation Results:")

for feature, score in single_split_ablation_results.items():
    print(
        f"Without {feature}: "
        f"{score:.2%}"
    )


print("\nMultiple Split Structural Results:")

print(
    "Full model:",
    f"{full_mean:.2%} ± {full_std:.2%}"
)

for feature, scores in ablation_results.items():

    mean_accuracy = (
        sum(scores)
        / len(scores)
    )

    std_accuracy = statistics.stdev(
        scores
    )

    print(
        f"Without {feature}: "
        f"{mean_accuracy:.2%} ± {std_accuracy:.2%}"
    )


print(
    "\nAverage Word Length Only:",
    f"{single_mean:.2%} ± {single_std:.2%}"
)


print(
    "\nTF-IDF Multiple Split:",
    f"{tfidf_mean:.2%} ± {tfidf_std:.2%}"
)


print(
    "\nAverage Structural Feature Values:"
)

print(
    df.groupby("label")[feature_columns].mean()
)


print("\nInitial Single-Split Results:")

print(
    "TF-IDF accuracy:",
    f"{tfidf_accuracy:.2%}"
)

print(
    "Structural accuracy:",
    f"{structural_accuracy:.2%}"
)


print(
    "\nStructural Feature Coefficients:"
)

print(
    structural_coef.sort_values()
)