# iris_project.py

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------
# Step 1: Load and Visualize Data
# -------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Visualization
sns.pairplot(df, hue="target")
plt.title("Iris Dataset Visualization")
plt.savefig("iris_visualization.png")  # Save visualization
plt.close()

# -------------------------
# Step 2: Preprocessing & Model Training
# -------------------------
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "iris_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(iris.target_names, "target_names.pkl")

# -------------------------
# Step 3: FastAPI Deployment
# -------------------------
app = FastAPI(title="Iris Classifier API")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Welcome to the Iris Classifier API"}

@app.post("/predict/")
def predict_species(data: IrisInput):
    model = joblib.load("iris_model.pkl")
    scaler = joblib.load("scaler.pkl")
    target_names = joblib.load("target_names.pkl")

    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return {"predicted_species": target_names[prediction]}