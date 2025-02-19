from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess_dataset import load_images
from extract_feature import extract_features
import numpy as np
import joblib
import os


def train_classifier(X_train_features, y_train, classifier):
    if classifier == "svm":
        model = SVC(kernel="linear")
    elif classifier == "knn":
        model = KNeighborsClassifier(n_neighbors=5)
    elif classifier == "decision_tree":
        model = DecisionTreeClassifier()
    else:
        raise ValueError("Invalid classifier")

    model.fit(X_train_features, y_train)
    return model


def evaluate_model(model, model_name, X_test_features, y_test):
    y_pred = model.predict(X_test_features)
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)
    confusion = confusion_matrix(y_test, y_pred)

    # Save evaluation to file
    report_path = os.path.join("models", "evaluation_reports.txt")
    with open(report_path, "a") as f:
        f.write("="*60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(confusion) + "\n")
        f.write("="*60 + "\n\n")


def save_model(model, model_name):
    model_folder = "models"
    model_path = os.path.join(model_folder, model_name)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


def main():
    # Load dataset
    X, y = load_images()
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)
    # Train, save and evaluate models
    extract_techniques = ["glcm", "lbp"]
    classifiers = ["svm", "knn", "decision_tree"]
    for technique in extract_techniques:
        for classifier in classifiers:
            print(f"Technique: {technique}, Classifier: {classifier}")
            X_train_features = extract_features(X_train, technique)
            model = train_classifier(X_train_features, y_train, classifier)
            model_name = f"{technique}_{classifier}_model.pkl"
            save_model(model, model_name)
            X_test_features = extract_features(X_test, technique)
            evaluate_model(model, model_name, X_test_features, y_test)
            print("\n")
    print("Finished.")


if __name__ == '__main__':
    main()
