import pandas as pd
import os

# Always resolve path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "heart.csv")

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv(URL, header=None, names=COLUMN_NAMES)

# Convert target to binary
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

df.to_csv(OUTPUT_PATH, index=False)

print("Heart Disease dataset downloaded and saved to:")
print(OUTPUT_PATH)
print(df.head())
