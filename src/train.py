from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"

df = pd.read_csv(DATASET)

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

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

print("Predictions:")
print(predictions)

print("\nActual labels:")
print(y_test.to_numpy())
print("\nAccuracy:")
print(f"{accuracy:.2%}")

print("\nTop words that incline towards AI:")
print(coef_sorted.head(20))

print("\nTop words that incline towards Human:")
print(coef_sorted.tail(20))


# print("Training matrix shape:", X_train_tfidf.shape)
# print("Testing matrix shape:", X_test_tfidf.shape)


# print("Training samples: ", len(X_train))
# print("Test samples: ", len(X_test))

# print("\nTraining assigments ID's:")
# print(df.iloc[train_idx]["assignment_id"].unique())

# print("\nTest assigments ID's:")
# print(df.iloc[test_idx]["assignment_id"].unique())