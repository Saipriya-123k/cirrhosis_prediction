import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("=" * 60)
print("🫀 CIRRHOSIS PREDICTION - TRAINING MODEL")
print("=" * 60)

data = pd.read_csv("cirrhosis.csv")

print("\n✅ Dataset loaded!")
print(f"📊 Number of patient records: {len(data)}")
print(f"📋 Columns in dataset: {data.columns.tolist()}")

data = data.fillna(data.median(numeric_only=True))

target = "Stage"

feature_columns = [
    "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema",
    "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos",
    "SGOT", "Tryglicerides", "Platelets", "Prothrombin"
]

available_features = [col for col in feature_columns if col in data.columns]

X = data[available_features].copy()
y = data[target].copy()

categorical_features = [col for col in ["Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"] if col in X.columns]
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n🤖 Training model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n🎯 MODEL ACCURACY: {accuracy * 100:.2f}%")

joblib.dump(model, "cirrhosis_model_pipeline.pkl")
joblib.dump(available_features, "feature_names.pkl")

print("\n💾 Model saved successfully!")
print("✅ cirrhosis_model_pipeline.pkl")
print("✅ feature_names.pkl")
print("\n🎉 TRAINING COMPLETE!")
