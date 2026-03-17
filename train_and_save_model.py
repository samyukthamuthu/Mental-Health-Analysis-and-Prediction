import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "Student_Depression_Dataset.csv"
OUTPUT_PATH = "depression_model_bundle.pkl"

def train_and_save():
    df = pd.read_csv(DATA_PATH)

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    # handle missing values safely
    if df["Financial Stress"].isnull().sum() > 0:
        df["Financial Stress"] = df["Financial Stress"].fillna(df["Financial Stress"].median())

    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df.drop("Depression", axis=1)
    y = df["Depression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    bundle = {
        "model": model,
        "label_encoders": label_encoders,
        "feature_columns": list(X.columns),
    }

    joblib.dump(bundle, OUTPUT_PATH)
    print(f"Saved model bundle to {OUTPUT_PATH}")

if __name__ == "__main__":
    train_and_save()