from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from structural_features import (
    sentence_stats,
    lexical_diversity,
    avg_word_length,
    punctuation_rate
)

DATASET = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"

df = pd. read_csv(DATASET)
df = df[df["label"].isin(["human", "ai"])]

df["ttr"] = df["text"].apply(lexical_diversity)
df["avg_word_length"] = df["text"].apply(avg_word_length)
df["punctuation_rate"] = df["text"].apply(punctuation_rate)

df["avg_sentence_len"], df["sentence_len_std"] = zip(
    *df["text"].apply(sentence_stats)
)
feature_columns = [
    "avg_sentence_len",
    "sentence_len_std",
    "ttr",
    "avg_word_length",
    "punctuation_rate"
]
X_structural = df[feature_columns]

scaler = StandardScaler()
X_structural_scaled = scaler.fit_transform(X_structural)

X = df["text"]
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)
X_tfidf = vectorizer.fit_transform(X)

kmeans = KMeans(
    n_clusters = 2,
    random_state = 42
)
clusters = kmeans.fit_predict(X_tfidf)
df["cluster"] = clusters

score = silhouette_score(X_tfidf, clusters)

kmeans_structural = KMeans(
    n_clusters=2,
    random_state=42
)
structural_clusters = kmeans_structural.fit_predict(X_structural_scaled)
df["structural_cluster"] = structural_clusters
structural_score = silhouette_score(
    X_structural_scaled,
    structural_clusters
)

print("TF-IDF matrix shape:", X_tfidf.shape)
print("\n")
print(df["cluster"].value_counts())
print("\n")
print(pd.crosstab(df["cluster"], df["label"]))
print("\n")
print("Silhouette score:", score)
print("\n")
print(X_structural_scaled.shape)



print("\nStructural cluster counts:")
print(df["structural_cluster"].value_counts())

print("\nStructural clusters vs labels:")
print(pd.crosstab(df["structural_cluster"], df["label"]))

print("\nStructural silhouette score:", structural_score)