import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import json
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import sys
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from StockPricePredictor import StockPricePredictor



class LogisticRegrModel(StockPricePredictor):
    def __init__(self, stock_name: str, config: json, logger: logging.Logger):

        """
        Initialize the LogisticRegrModel with the given stock name, configuration, and logger.

        Args:
            stock_name (str): The name of the stock to model.
            config (json): Configuration parameters for the model.
            logger (logging.Logger): Logger object for logging information and errors.

        Returns:
            None
        """
        super().__init__(stock_name,config,logger)
        """Initialize the model, scalers, and preprocess data."""
        self.scaler = StandardScaler()
        self.elastic_net = ElasticNet(
            alpha=config["alpha"], 
            l1_ratio=config["l1_ratio"], 
            random_state=config["random_state"]
        )
        self.cv = config["cv"]
        self.scoring = config["scoring"]
        self.top_features = config["top_features"]
        self.random_state = config["random_state"]
        self.columns_to_drop = list(config["columns_to_drop"])
        self.X, self.y = self.preprocess_data()
        self.preprocess_data_for_training()
        self.best_params = None
    
    def preprocess_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the dataset by performing feature selection and scaling.

        This method prepares the data for model training by:
        1. Creating a target variable based on price movement.
        2. Dropping specified columns.
        3. Selecting numeric features.
        4. Performing feature selection using ElasticNet.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - X (pd.DataFrame): Selected features for model input.
                - y (pd.DataFrame): Target variable (price movement direction).
        """
        super().preprocess_data()
        self.df['Target'] = (self.df['Close'] > self.df['Close'].shift(-1)).astype(int)
        self.df = self.df[:-1]
        self.df = self.df.drop(columns=self.columns_to_drop, errors='ignore')
        
        # Extract the columns that contain numeric data
        feature_candidates = self.df.select_dtypes(include=['number']).columns.drop(['Target'])
        X_all = self.df[feature_candidates]
        y_all = self.df['Target']

        # Feature selection using ElasticNet
        X_scaled = self.scaler.fit_transform(X_all)
        try:
            # Fit ElasticNet for feature selection
            self.elastic_net.fit(X_scaled, y_all)
        except Exception as e:
            self.logger.error(f"ElasticNet fitting failed: {e}")
            raise
        # Select top features based on absolute coefficient values
        coef_abs = np.abs(self.elastic_net.coef_)
        # Make sure that there is no index out of range 
        top_features_count = min(self.top_features, len(feature_candidates))
        top_indices = np.argsort(coef_abs)[-top_features_count:]
        self.selected_features = feature_candidates[top_indices].tolist()
        self.logger.info(f"Selected features for {self.stock_name}: {self.selected_features}")
        return self.df[self.selected_features], self.df['Target']


    def preprocess_data_for_training(self) -> None:
        """
        Prepare data for model training by splitting, applying SMOTE, and scaling features.

        This method:
        1. Splits the data into training and testing sets.
        2. Applies SMOTE to balance the training data.
        3. Scales the features using StandardScaler.

        Args:
            None

        Returns:
            None
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )
        # Apply SMOTE once
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(X_train, y_train)

        # Scale data once
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_resampled)
        self.X_test_scaled = self.scaler.transform(X_test)

        self.y_test = y_test  # Store y_test for evaluation


    def tune_hyperparameters(self, config: json) -> None:
        """
        Tune model hyperparameters using GridSearchCV with preprocessed data.

        This method:
        1. Sets up a parameter grid based on the provided configuration.
        2. Performs grid search using cross-validation.
        3. Stores the best parameters and initializes the model with them.

        Args:
            config (json): Configuration containing the hyperparameter grid.

        Returns:
            None
        """
        param_grid = config["hyperparameter_grid"]

        # Use preprocessed, scaled, and resampled training data
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state),
            param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1
        )

        grid_search.fit(self.X_train_scaled, self.y_train_resampled)

        # Store the best model parameters
        self.best_params = grid_search.best_params_
        self.model = RandomForestClassifier(**self.best_params, random_state=self.random_state)

        self.logger.info(f"Best Parameters: {self.best_params}")
        self.logger.info(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")


    def train(self) -> None:
        """
        Train the model using the best hyperparameters or a default configuration.

        This method:
        1. Fits the model to the preprocessed training data.
        2. Makes predictions on the test set.
        3. Calculates and logs various performance metrics.

        Args:
            None

        Returns:
            None
        """
        if self.model is None:
            self.model = LogisticRegression(solver='liblinear', random_state=42)  # Default model if tuning is not performed

        self.model.fit(self.X_train_scaled, self.y_train_resampled)

        predictions = self.model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        conf_matrix = confusion_matrix(self.y_test, predictions)

        self.logger.info(f"Model training complete for {self.stock_name}.")
        self.logger.info(f"Model Accuracy: {accuracy:.4f}")
        self.logger.info(f"Classification Report:\n{report}")
        self.logger.info(f"Confusion Matrix:\n{conf_matrix}")

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Predict whether the stock price will rise (1) or not (0) for new data.

        This method:
        1. Ensures input data is in the correct format.
        2. Selects only the features used during training.
        3. Scales the input data.
        4. Makes predictions using the trained model.

        Args:
            X_new (pd.DataFrame): New data for prediction.

        Returns:
            np.ndarray: Array of predictions (1 for price rise, 0 for no rise).
        """
        
        # Ensure X_new is a DataFrame
        if isinstance(X_new, np.ndarray):
            X_new = pd.DataFrame(X_new, columns=self.selected_features)

        # Select only the stored features
        X_new = X_new[self.scaler.feature_names_in_]
        # Scale the data
        X_new_scaled = self.scaler.transform(X_new)
        return self.model.predict(X_new_scaled)



