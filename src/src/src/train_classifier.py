import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

def train_classifier(features_csv="features.csv"):
    # Load feature table
    df = pd.read_csv(features_csv)

    X = df.drop(columns=["label"])
    y = df["label"]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Random Forest model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds, labels=model.classes_)
    print("\nConfusion Matrix:")
    print(cm)

    # Save confusion matrix plot
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/confusion_matrix.png")
    plt.close()

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/variable_star_classifier.pkl")

    print("Model saved to models/variable_star_classifier.pkl")
    print("Confusion matrix saved to docs/confusion_matrix.png")



if __name__ == "__main__":
    train_classifier()
