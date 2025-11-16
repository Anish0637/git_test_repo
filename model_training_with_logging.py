"""Model training script with logging functionality."""
import pandas as pd
from sklearn.model_selection import train_test_split            
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report   
import logging
import joblib
# Set up logging and add log file handler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 
logger = logging.getLogger(__name__)

# Add a file handler so logs are also written to 'training.log'
file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Avoid adding multiple file handlers if this code is executed more than once
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)


def load_data(file_path):
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logger.exception(f"Error loading data: {e}")
        raise

def preprocess_data(data, target_column):
    """Preprocess the dataset by handling missing values and encoding categorical features.

    - Fill numeric missing values with the column median.
    - Fill categorical missing values with the column mode.
    - One-hot encode categorical features (drop_first=True).
    - Ensure the target is numeric (factorize if needed).
    """
    try:
        data = data.copy()
        # Handle missing values per-column
        for col in data.columns:
            if col == target_column:
                continue
            if data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col]):
                # Fill categorical NaNs with mode
                if not data[col].mode().empty:
                    data[col] = data[col].fillna(data[col].mode().iloc[0])
                else:
                    data[col] = data[col].fillna('')
            else:
                # Coerce to numeric and fill numeric NaNs with median
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].fillna(data[col].median())

        X = data.drop(columns=[target_column])
        # One-hot encode categorical features so scikit-learn receives numeric input
        X = pd.get_dummies(X, drop_first=True)

        y = data[target_column]
        # If target is non-numeric, convert to numeric labels
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            y = pd.factorize(y)[0]

        logger.info("Data preprocessing completed successfully (encoded categorical features)")
        return X, y
    except Exception as e:
        logger.exception(f"Error in data preprocessing: {e}")
        raise

def train_model(X_train, y_train):
    """Train a Random Forest Classifier."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Model training completed successfully")
        return model
    except Exception as e:
        logger.exception(f"Error in model training: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model and log performance metrics."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logger.info(f"Model evaluation completed successfully with accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")
        return accuracy
    except Exception as e:
        logger.exception(f"Error in model evaluation: {e}")
        raise

def save_model(model, file_path):
    """Save the trained model to a file."""
    try:
        joblib.dump(model, file_path)
        logger.info(f"Model saved successfully to {file_path}")
    except Exception as e:
        logger.exception(f"Error saving model: {e}")
        raise

if __name__ == "__main__":  
    # Load and preprocess data
    data = load_data("data/dataset1.csv")
    X, y = preprocess_data(data, target_column="target")
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the model
    model = train_model(X_train, y_train)
    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    # Save the trained model
    save_model(model, "random_forest_model.joblib")