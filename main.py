# =========================================================
# AI PREDICTIVE MAINTENANCE - FINAL (SHOW + SAVE GRAPHS)
# =========================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================================================
# CREATE FOLDERS
# =========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =========================================================
# LOAD DATASET
# =========================================================

file_path = "data/raw/train_FD001.txt"
df = pd.read_csv(file_path, sep=r"\s+", header=None)

print("\nRaw dataset shape:", df.shape)

# =========================================================
# FIX COLUMN ISSUE (ENSURE 26 COLUMNS)
# =========================================================

df = df.iloc[:, :26]

columns = ["engine_id", "cycle"] + [f"s{i}" for i in range(1, 25)]
df.columns = columns

print("Cleaned dataset shape:", df.shape)
print(df.head())

# =========================================================
# CREATE RUL
# =========================================================

max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
max_cycle.columns = ["engine_id", "max_cycle"]

df = df.merge(max_cycle, on="engine_id")
df["RUL"] = df["max_cycle"] - df["cycle"]

# =========================================================
# CREATE FAILURE LABEL
# =========================================================

threshold = 30
df["failure"] = df["RUL"].apply(lambda x: 1 if x <= threshold else 0)

# =========================================================
# FEATURES & TARGET
# =========================================================

features = [f"s{i}" for i in range(1, 25)]

X = df[features]
y = df["failure"]

# FIX (pandas 3 issue)
X = X.ffill()

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# MODEL TRAINING
# =========================================================

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================================================
# EVALUATION
# =========================================================

y_pred = model.predict(X_test)

print("\n====================")
print("MODEL PERFORMANCE")
print("====================")
print(classification_report(y_test, y_pred))

# =========================================================
# CONFUSION MATRIX (SHOW + SAVE)
# =========================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")
plt.show()   # ✅ SHOW GRAPH

# =========================================================
# FEATURE IMPORTANCE (SHOW + SAVE)
# =========================================================

importances = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure()
sns.barplot(x="Importance", y="Feature", data=feat_df)
plt.title("Feature Importance")

plt.savefig("outputs/feature_importance.png")
plt.show()   # ✅ SHOW GRAPH

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(model, "models/model.pkl")
print("\nModel saved successfully!")

# =========================================================
# SAMPLE PREDICTION
# =========================================================

sample = X_test.iloc[0:1]

prediction = model.predict(sample)[0]
prob = model.predict_proba(sample)[0][1]

print("\n====================")
print("SAMPLE OUTPUT")
print("====================")

if prediction == 1:
    print("⚠ FAILURE RISK DETECTED")
else:
    print("✅ MACHINE HEALTHY")

print(f"Failure Probability: {prob:.3f}")