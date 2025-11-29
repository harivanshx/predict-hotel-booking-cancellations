"""
Model training module.
Handles training multiple models with class imbalance handling and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import config

# Set up logging
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and tune machine learning models for cancellation prediction."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize ModelTrainer.
        
        Args:
            df: DataFrame with features and target
        """
        self.df = df.copy()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training by splitting into train/test sets.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for training")
        
        # Separate features and target
        if config.TARGET_COLUMN not in self.df.columns:
            raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in dataset")
        
        X = self.df.drop(config.TARGET_COLUMN, axis=1)
        y = self.df[config.TARGET_COLUMN]
        
        # Handle any remaining categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Encoding remaining categorical columns: {list(categorical_cols)}")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y
        )
        
        logger.info(f"Training set: {self.X_train.shape[0]} samples")
        logger.info(f"Test set: {self.X_test.shape[0]} samples")
        logger.info(f"Number of features: {self.X_train.shape[1]}")
        
        # Log class distribution
        train_dist = self.y_train.value_counts()
        logger.info(f"Training set class distribution:")
        for cls, count in train_dist.items():
            pct = (count / len(self.y_train)) * 100
            logger.info(f"  Class {cls}: {count} ({pct:.2f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def handle_class_imbalance(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
        
        SMOTE creates synthetic samples of the minority class by interpolating
        between existing minority class samples, helping the model learn better
        decision boundaries.
        
        Returns:
            Tuple of (X_train_resampled, y_train_resampled)
        """
        logger.info("Handling class imbalance with SMOTE")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        # Check class distribution before SMOTE
        original_dist = self.y_train.value_counts()
        logger.info(f"Before SMOTE: {dict(original_dist)}")
        
        # Apply SMOTE
        smote = SMOTE(**config.SMOTE_CONFIG)
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
            self.X_train, self.y_train
        )
        
        # Check class distribution after SMOTE
        resampled_dist = pd.Series(self.y_train_resampled).value_counts()
        logger.info(f"After SMOTE: {dict(resampled_dist)}")
        logger.info(f"Training samples increased from {len(self.y_train)} to {len(self.y_train_resampled)}")
        
        return self.X_train_resampled, self.y_train_resampled
    
    def train_logistic_regression(self) -> Any:
        """
        Train Logistic Regression model.
        
        Returns:
            Trained model
        """
        logger.info("Training Logistic Regression")
        
        model = LogisticRegression(**config.MODELS_CONFIG['logistic_regression'])
        model.fit(self.X_train_resampled, self.y_train_resampled)
        
        # Evaluate on test set
        train_score = model.score(self.X_train_resampled, self.y_train_resampled)
        test_score = model.score(self.X_test, self.y_test)
        
        logger.info(f"  Training accuracy: {train_score:.4f}")
        logger.info(f"  Test accuracy: {test_score:.4f}")
        
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(self) -> Any:
        """
        Train Random Forest model.
        
        Returns:
            Trained model
        """
        logger.info("Training Random Forest")
        
        model = RandomForestClassifier(**config.MODELS_CONFIG['random_forest'])
        model.fit(self.X_train_resampled, self.y_train_resampled)
        
        # Evaluate on test set
        train_score = model.score(self.X_train_resampled, self.y_train_resampled)
        test_score = model.score(self.X_test, self.y_test)
        
        logger.info(f"  Training accuracy: {train_score:.4f}")
        logger.info(f"  Test accuracy: {test_score:.4f}")
        
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self) -> Any:
        """
        Train XGBoost model.
        
        Returns:
            Trained model
        """
        logger.info("Training XGBoost")
        
        model = XGBClassifier(**config.MODELS_CONFIG['xgboost'])
        model.fit(self.X_train_resampled, self.y_train_resampled)
        
        # Evaluate on test set
        train_score = model.score(self.X_train_resampled, self.y_train_resampled)
        test_score = model.score(self.X_test, self.y_test)
        
        logger.info(f"  Training accuracy: {train_score:.4f}")
        logger.info(f"  Test accuracy: {test_score:.4f}")
        
        self.models['xgboost'] = model
        return model
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Train all configured models.
        
        Returns:
            Dictionary of trained models
        """
        logger.info("=" * 80)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 80)
        
        if self.X_train_resampled is None:
            raise ValueError("Data not resampled. Call handle_class_imbalance() first.")
        
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        
        logger.info(f"\nTrained {len(self.models)} models successfully")
        return self.models
    
    def tune_hyperparameters(self, model_name: str = 'xgboost') -> Any:
        """
        Tune hyperparameters for the specified model using RandomizedSearchCV.
        
        Args:
            model_name: Name of model to tune ('xgboost' or 'random_forest')
            
        Returns:
            Best model after tuning
        """
        logger.info("=" * 80)
        logger.info(f"HYPERPARAMETER TUNING: {model_name.upper()}")
        logger.info("=" * 80)
        
        if model_name == 'xgboost':
            base_model = XGBClassifier(
                random_state=config.RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False
            )
            param_grid = config.XGBOOST_PARAM_GRID
        elif model_name == 'random_forest':
            base_model = RandomForestClassifier(
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
            param_grid = config.RF_PARAM_GRID
        else:
            raise ValueError(f"Hyperparameter tuning not configured for {model_name}")
        
        # Choose search method
        if config.TUNING_CONFIG['method'] == 'randomized':
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=config.TUNING_CONFIG['n_iter'],
                cv=config.TUNING_CONFIG['cv'],
                scoring=config.TUNING_CONFIG['scoring'],
                n_jobs=config.TUNING_CONFIG['n_jobs'],
                random_state=config.RANDOM_STATE,
                verbose=1
            )
        else:
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=config.TUNING_CONFIG['cv'],
                scoring=config.TUNING_CONFIG['scoring'],
                n_jobs=config.TUNING_CONFIG['n_jobs'],
                verbose=1
            )
        
        logger.info(f"Starting hyperparameter search with {config.TUNING_CONFIG['method']} method")
        search.fit(self.X_train_resampled, self.y_train_resampled)
        
        logger.info(f"\nBest parameters found:")
        for param, value in search.best_params_.items():
            logger.info(f"  {param}: {value}")
        
        logger.info(f"\nBest cross-validation score: {search.best_score_:.4f}")
        
        # Update models dictionary with tuned model
        self.models[f'{model_name}_tuned'] = search.best_estimator_
        
        return search.best_estimator_
    
    def select_best_model(self) -> Tuple[str, Any]:
        """
        Select the best model based on test set performance.
        
        Returns:
            Tuple of (model_name, best_model)
        """
        logger.info("=" * 80)
        logger.info("SELECTING BEST MODEL")
        logger.info("=" * 80)
        
        from sklearn.metrics import roc_auc_score
        
        best_score = 0
        best_name = None
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            score = roc_auc_score(self.y_test, y_pred_proba)
            
            logger.info(f"{name}: ROC-AUC = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        logger.info(f"\n✓ Best model: {best_name} (ROC-AUC: {best_score:.4f})")
        
        return best_name, self.best_model
    
    def save_models(self):
        """Save trained models to disk."""
        logger.info("Saving models")
        
        # Save individual models
        if 'logistic_regression' in self.models:
            joblib.dump(self.models['logistic_regression'], config.LOGISTIC_MODEL_FILE)
            logger.info(f"  Saved Logistic Regression to {config.LOGISTIC_MODEL_FILE}")
        
        if 'random_forest' in self.models:
            joblib.dump(self.models['random_forest'], config.RANDOM_FOREST_MODEL_FILE)
            logger.info(f"  Saved Random Forest to {config.RANDOM_FOREST_MODEL_FILE}")
        
        if 'xgboost' in self.models:
            joblib.dump(self.models['xgboost'], config.XGBOOST_MODEL_FILE)
            logger.info(f"  Saved XGBoost to {config.XGBOOST_MODEL_FILE}")
        
        # Save best model
        if self.best_model is not None:
            joblib.dump(self.best_model, config.BEST_MODEL_FILE)
            logger.info(f"  Saved best model ({self.best_model_name}) to {config.BEST_MODEL_FILE}")
        
        # Save feature names for later use
        feature_names = list(self.X_train.columns)
        joblib.dump(feature_names, config.MODELS_DIR / 'feature_names.pkl')
        logger.info(f"  Saved feature names")
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Args:
            model_name: Name of model to get importance from. If None, uses best model.
            
        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature_importances_")
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


def train_models(df: pd.DataFrame, tune_best: bool = True) -> Tuple[Dict[str, Any], str]:
    """
    Convenience function to train all models.
    
    Args:
        df: DataFrame with features and target
        tune_best: Whether to tune hyperparameters of best model
        
    Returns:
        Tuple of (models dictionary, best model name)
    """
    trainer = ModelTrainer(df)
    trainer.prepare_data()
    trainer.handle_class_imbalance()
    trainer.train_all_models()
    
    if tune_best:
        # Tune XGBoost (typically best performer)
        trainer.tune_hyperparameters('xgboost')
    
    best_name, best_model = trainer.select_best_model()
    trainer.save_models()
    
    return trainer.models, best_name


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from data_loader import load_and_validate_data
    from preprocess import preprocess_data
    from feature_engineering import engineer_features
    
    df, _ = load_and_validate_data()
    df_clean = preprocess_data(df)
    df_engineered = engineer_features(df_clean)
    
    models, best_name = train_models(df_engineered)
    print(f"\nBest model: {best_name}")
