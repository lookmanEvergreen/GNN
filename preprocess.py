import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Load datasets
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

# Add labels
true_df["label"] = 1
fake_df["label"] = 0

# Combine datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text"])
y = df["label"].values

# Save vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/vectorizer.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
os.makedirs("data", exist_ok=True)
np.save("data/X_train.npy", X_train.toarray())
np.save("data/X_test.npy", X_test.toarray())
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)
print("Data preprocessing complete!")
