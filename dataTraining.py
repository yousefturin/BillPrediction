import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import joblib

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def split_data(data):
    """Split data into features (X) and labels (y)."""
    X = data.iloc[:, 1:].values  # The columns 1 onwards are features
    y = data.iloc[:, 0].values  # The first column is the label
    return X, y

def preprocess_data(X_train, X_test):
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_models(X_train_scaled, y_train):
    """Train SVM, KNN, and Perceptron models."""
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_scaled, y_train)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)

    perceptron_model = Perceptron()
    perceptron_model.fit(X_train_scaled, y_train)

    return svm_model, knn_model, perceptron_model

def save_models(models):
    """Save trained models to disk."""
    svm_model, knn_model, perceptron_model = models
    joblib.dump(svm_model, 'models/svm_model.pkl')
    joblib.dump(knn_model, 'models/knn_model.pkl')
    joblib.dump(perceptron_model, 'models/perceptron_model.pkl')

def evaluate_models(models, X_test_scaled, y_test):
    """Evaluate model accuracies."""
    svm_model, knn_model, perceptron_model = models
    svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test_scaled))
    knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test_scaled))
    perceptron_accuracy = accuracy_score(y_test, perceptron_model.predict(X_test_scaled))
    print("SVM Accuracy:", svm_accuracy)
    print("KNN Accuracy:", knn_accuracy)
    print("Perceptron Accuracy:", perceptron_accuracy)
    
def main():
    data_file = 'image_data.csv'
    data = load_data(data_file)
    X, y = split_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
    models = train_models(X_train_scaled, y_train)
    save_models(models)
    evaluate_models(models, X_test_scaled, y_test)
    
if __name__ == "__main__":
    main()
