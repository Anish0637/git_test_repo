import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
import logging
logger = logging.getLogger(__name__)  
  
def tune_hyperparameters(X_train, y_train):
    """Tune hyperparameters for RandomForestClassifier using GridSearchCV."""
    try:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        logger.info(f"Best hyperparameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    except Exception as e:
        logger.exception(f"Error during hyperparameter tuning: {e}")
        raise

  #-------------------End of Code-------------#
