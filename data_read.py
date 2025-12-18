import pandas as pd
import logging      
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)    

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

        logger.info("Data preprocessing completed successfully.")
        return X, y
    except Exception as e:
        logger.exception(f"Error during data preprocessing: {e}")
        raise