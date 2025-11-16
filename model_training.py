import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  
def load_data(file_path):
    """Load dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data 

def preprocess_data(data):
    """Preprocess the dataset."""
    # Example preprocessing: fill missing values and encode categorical variables
    data = data.fillna(method='ffill')
    data = pd.get_dummies(data, drop_first=True)
    return data   
      
def split_data(data, target_column):
    """Split the dataset into training and testing sets."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('data/dataset.csv')
    data = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(data, target_column='target')
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Model Accuracy: {accuracy:.2f}')
    
"""Module for training and evaluating a machine learning model."""



