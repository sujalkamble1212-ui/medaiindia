import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Loading dataset...")

df=pd.read_csv("dataset.csv")

df=df.drop(columns=["Unnamed: 133"],errors="ignore")

df=df.fillna(df.mode().iloc[0])

print("Total columns:",len(df.columns))
print("Columns loaded successfully.")

if "prognosis" not in df.columns:
    raise ValueError("❌ 'prognosis' column not found in dataset!")

X=df.drop("prognosis",axis=1)
y=df["prognosis"]

print("Training model...")

model=RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42
)

model.fit(X,y)

joblib.dump(model,"disease_model.pkl")

print("✅ Model trained and saved successfully as disease_model.pkl")