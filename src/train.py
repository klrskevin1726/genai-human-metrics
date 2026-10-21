from pathlib import Path
import statistics
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from structural_features import sentence_stats, lexical_diversity, avg_word_length, punctuation_rate
from sklearn.preprocessing import StandardScaler

DATASET = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"


df = pd.read_csv(DATASET)
df = df[df["label"].isin(["human", "ai"])]
df["punctuation_rate"] = df["text"].apply(punctuation_rate)
df["ttr"] = df["text"].apply(lexical_diversity)
df["avg_word_length"] = df["text"].apply(avg_word_length)
df["avg_sentence_len"], df["sentence_len_std"] = zip(*df["text"].apply(sentence_stats))
group_stats = df.groupby("label")[["avg_sentence_len", "sentence_len_std"]].mean()

X = df["text"]
y = df["label"]

groups = df["assignment_id"]
splitter = GroupShuffleSplit(
    n_splits = 1,
    test_size = 0.30,
    random_state = 42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups = groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range= (1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression(
    max_iter= 2000,
    random_state= 42
)

model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

coef_series = pd.Series(model.coef_[0], index=vectorizer.get_feature_names_out())
coef_sorted = coef_series.sort_values()


feature_columns = [
    "avg_sentence_len",
    "sentence_len_std",
    "ttr",
    "avg_word_length",
    "punctuation_rate"
]
X_structural = df[feature_columns]

X_structural_train = X_structural.iloc[train_idx]
X_structural_test = X_structural.iloc[test_idx]


scaler = StandardScaler()

X_structural_train_scaled = scaler.fit_transform(X_structural_train)
X_structural_test_scaled = scaler.transform(X_structural_test)


structural_model = LogisticRegression(
    max_iter = 2000,
    random_state = 42
)

structural_model.fit(
    X_structural_train_scaled,
    y_train
)



structural_predictions = structural_model.predict(
    X_structural_test_scaled
)


structural_accuracy = accuracy_score(
    y_test,
    structural_predictions
) 

print("\nFeature Ablation Results:")

for feature_to_remove in feature_columns:

    reduced_features = [
        feature for feature in feature_columns
        if feature != feature_to_remove
    ]

    X_reduced = df[reduced_features]

    X_reduced_train = X_reduced.iloc[train_idx]
    X_reduced_test = X_reduced.iloc[test_idx]

    scaler_reduced = StandardScaler()

    X_reduced_train_scaled = scaler_reduced.fit_transform(
        X_reduced_train
    )

    X_reduced_test_scaled = scaler_reduced.transform(
        X_reduced_test
    )

    reduced_model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    reduced_model.fit(
        X_reduced_train_scaled,
        y_train
    )

    reduced_predictions = reduced_model.predict(
        X_reduced_test_scaled
    )

    reduced_accuracy = accuracy_score(
        y_test,
        reduced_predictions
    )

    print(
        f"Without {feature_to_remove}: "
        f"{reduced_accuracy:.2%}"
    )




multi_split_results = []
ablation_results = {
    feature: []
    for feature in feature_columns
}

for seed in range(20):

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed
    )

    train_idx, test_idx = next(
        splitter.split(
            df["text"],
            df["label"],
            groups=df["assignment_id"]
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
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    full_accuracy = accuracy_score(
        y_test,
        predictions
    )

    multi_split_results.append(
        full_accuracy
    )

    # Feature ablation
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
            random_state=42
        )

        reduced_model.fit(
            X_reduced_train_scaled,
            y_train
        )

        reduced_predictions = reduced_model.predict(
            X_reduced_test_scaled
        )

        reduced_accuracy = accuracy_score(
            y_test,
            reduced_predictions
        )

        ablation_results[
            feature_to_remove
        ].append(
            reduced_accuracy
        )


print("\nMultiple Split Structural Results:")

full_mean = sum(multi_split_results) / len(multi_split_results)
full_std = statistics.stdev(multi_split_results)

print(
    "Full model:",
    f"{full_mean:.2%} ± {full_std:.2%}"
)

for feature, scores in ablation_results.items():

    mean_accuracy = sum(scores) / len(scores)
    std_accuracy = statistics.stdev(scores)

    print(
        f"Without {feature}: "
        f"{mean_accuracy:.2%} ± {std_accuracy:.2%}"
    )


    
structural_coef = pd.Series(
    structural_model.coef_[0],
    index=feature_columns
)




# print(classification_report(y_test, predictions))
# print(confusion_matrix(y_test, predictions))

# print("Predictions:")
# print(predictions)

# print("\nActual labels:")
# print(y_test.to_numpy())
# print("\nAccuracy:")
# print(f"{accuracy:.2%}")

# print("\nTop words that incline towards AI:")
# print(coef_sorted.head(20))

# print("\nTop words that incline towards Human:")
# print(coef_sorted.tail(20))


# print("Training matrix shape:", X_train_tfidf.shape)
# print("Testing matrix shape:", X_test_tfidf.shape)


# print("Training samples: ", len(X_train))
# print("Test samples: ", len(X_test))

# print("\nTraining assigments ID's:")
# print(df.iloc[train_idx]["assignment_id"].unique())

# print("\nTest assigments ID's:")
# print(df.iloc[test_idx]["assignment_id"].unique())



# print(df[["label", "avg_sentence_len", "sentence_len_std"]].head())
# print(group_stats)

# print("\n")
# print(df.groupby("label")["avg_word_length"].mean())
# print("\n")
# print(df.groupby("label")["ttr"].mean())
# print("\n")
# print(df.groupby("label")["punctuation_rate"].mean())

print("\n",df.groupby("label")[feature_columns].mean())

print("\nTF-IDF accuracy:", f"{accuracy:.2%}")
print("Structural accuracy:", f"{structural_accuracy:.2%}")

print("\nStructural feature coefficients:")
print(structural_coef.sort_values())


# What identifying criteria actually contribute to distinguishing human generated and machine generated writing?

#One of the strongest identifying criteria I found is average word length. Longer average word length is associated with the machine-generated samples in this dataset. On the human side, sentence-length variability appears to be an important criterion: human texts tend to vary more in sentence length.
