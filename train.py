# ==========================================
# PHISHING EMAIL DETECTION
# Step 7 - Evaluate & Save Model
# ==========================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ---------------------------
# Load Dataset
# ---------------------------
data = pd.read_csv("phishing_email.csv")

# Convert labels to numbers
data["label"] = data["label"].map({
    "safe": 0,
    "phishing": 1
})

# Features and Labels
X = data["EmailText"]
y = data["label"]

# ---------------------------
# TF-IDF Feature Extraction
# ---------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True
)

X_features = vectorizer.fit_transform(X)

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------
# Train Model
# ---------------------------
model = LogisticRegression()

model.fit(X_train, y_train)

# ---------------------------
# Predictions
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# Accuracy
# ---------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

# ---------------------------
# Classification Report
# ---------------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# ---------------------------
# Plot Confusion Matrix
# ---------------------------
plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Safe", "Phishing"],
    yticklabels=["Safe", "Phishing"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ---------------------------
# Save Model
# ---------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel Saved Successfully!")
print("Files Created:")
print(" - model.pkl")
print(" - vectorizer.pkl")
