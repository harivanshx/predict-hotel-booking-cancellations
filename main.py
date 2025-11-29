"""
Main pipeline orchestrator for hotel booking cancellation prediction.
Runs the complete ML pipeline from data loading to model evaluation.
"""


import sys
import logging
import logging.config
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import config
from data_loader import DataLoader
from eda import perform_eda
from preprocess import DataPreprocessor
from feature_engineering import FeatureEngineer
from train import ModelTrainer
from evaluate import ModelEvaluator

# Set up logging
logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class HotelBookingPipeline:
    """Complete ML pipeline for hotel booking cancellation prediction."""
    
    def __init__(self, data_path: Path = None):
        """
        Initialize pipeline.
        
        Args:
            data_path: Path to raw data file. If None, uses config default.
        """
        self.data_path = data_path or config.RAW_DATA_FILE
        self.df_raw = None
        self.df_clean = None
        self.df_engineered = None
        self.trainer = None
        
    def run_data_loading(self):
        """Step 1: Load and validate data."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA LOADING AND VALIDATION")
        logger.info("=" * 80)
        
        loader = DataLoader(self.data_path)
        self.df_raw = loader.load_data()
        loader.generate_data_summary()
        loader.validate_data()
        loader.get_target_distribution()
        
        return self.df_raw
    
    def run_eda(self):
        """Step 2: Exploratory Data Analysis."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 80)
        
        if self.df_raw is None:
            raise ValueError("Data not loaded. Run run_data_loading() first.")
        
        perform_eda(self.df_raw)
        
    def run_preprocessing(self):
        """Step 3: Data preprocessing."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: DATA PREPROCESSING")
        logger.info("=" * 80)
        
        if self.df_raw is None:
            raise ValueError("Data not loaded. Run run_data_loading() first.")
        
        preprocessor = DataPreprocessor(self.df_raw)
        self.df_clean = preprocessor.preprocess()
        
        return self.df_clean
    
    def run_feature_engineering(self):
        """Step 4: Feature engineering."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        if self.df_clean is None:
            raise ValueError("Data not preprocessed. Run run_preprocessing() first.")
        
        engineer = FeatureEngineer(self.df_clean)
        self.df_engineered = engineer.create_all_features()
        
        # Encode categorical variables after feature engineering
        preprocessor = DataPreprocessor(self.df_engineered)
        self.df_engineered = preprocessor.encode_categorical_variables()
        self.df_engineered = preprocessor.drop_unnecessary_columns()
        
        logger.info(f"\nFinal dataset ready for modeling: {self.df_engineered.shape}")
        
        return self.df_engineered
    
    def run_model_training(self, tune_hyperparameters: bool = True):
        """
        Step 5: Model training.
        
        Args:
            tune_hyperparameters: Whether to tune hyperparameters
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: MODEL TRAINING")
        logger.info("=" * 80)
        
        if self.df_engineered is None:
            raise ValueError("Features not engineered. Run run_feature_engineering() first.")
        
        self.trainer = ModelTrainer(self.df_engineered)
        self.trainer.prepare_data()
        self.trainer.handle_class_imbalance()
        self.trainer.train_all_models()
        
        if tune_hyperparameters:
            logger.info("\nPerforming hyperparameter tuning...")
            self.trainer.tune_hyperparameters('xgboost')
        
        self.trainer.select_best_model()
        self.trainer.save_models()
        
        return self.trainer
    
    def run_model_evaluation(self):
        """Step 6: Model evaluation."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: MODEL EVALUATION")
        logger.info("=" * 80)
        
        if self.trainer is None:
            raise ValueError("Models not trained. Run run_model_training() first.")
        
        # Evaluate best model
        evaluator = ModelEvaluator(
            self.trainer.best_model,
            self.trainer.X_test,
            self.trainer.y_test,
            self.trainer.best_model_name
        )
        
        # Get feature importance if available
        feature_importance_df = self.trainer.get_feature_importance()
        
        # Run full evaluation
        results = evaluator.evaluate_full(feature_importance_df)
        
        return results
    
    def run_full_pipeline(self, tune_hyperparameters: bool = True):
        """
        Run the complete pipeline from start to finish.
        
        Args:
            tune_hyperparameters: Whether to tune hyperparameters
        """
        logger.info("\n" + "=" * 100)
        logger.info(" " * 20 + "HOTEL BOOKING CANCELLATION PREDICTION PIPELINE")
        logger.info("=" * 100)
        
        try:
            # Run all steps
            self.run_data_loading()
            self.run_eda()
            self.run_preprocessing()
            self.run_feature_engineering()
            self.run_model_training(tune_hyperparameters)
            results = self.run_model_evaluation()
            
            logger.info("\n" + "=" * 100)
            logger.info(" " * 30 + "PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 100)
            logger.info(f"\nBest Model: {self.trainer.best_model_name}")
            logger.info(f"ROC-AUC Score: {results['metrics']['roc_auc']:.4f}")
            logger.info(f"F1-Score: {results['metrics']['f1']:.4f}")
            logger.info(f"\nModel saved to: {config.BEST_MODEL_FILE}")
            logger.info(f"Evaluation report: {config.MODEL_EVALUATION_FILE}")
            logger.info(f"EDA plots: {config.EDA_PLOTS_DIR}")
            
            return results
            
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed with error: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Hotel Booking Cancellation Prediction Pipeline'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'full'],
        default='full',
        help='Pipeline mode: train (training only), evaluate (evaluation only), full (complete pipeline)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to raw data CSV file'
    )
    parser.add_argument(
        '--no-tuning',
        action='store_true',
        help='Skip hyperparameter tuning (faster but potentially lower performance)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: skip EDA and hyperparameter tuning'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    data_path = Path(args.data_path) if args.data_path else None
    pipeline = HotelBookingPipeline(data_path)
    
    try:
        if args.mode == 'full':
            # Run complete pipeline
            tune = not args.no_tuning and not args.quick
            
            pipeline.run_data_loading()
            
            if not args.quick:
                pipeline.run_eda()
            
            pipeline.run_preprocessing()
            pipeline.run_feature_engineering()
            pipeline.run_model_training(tune_hyperparameters=tune)
            pipeline.run_model_evaluation()
            
        elif args.mode == 'train':
            # Training only
            pipeline.run_data_loading()
            pipeline.run_preprocessing()
            pipeline.run_feature_engineering()
            pipeline.run_model_training(tune_hyperparameters=not args.no_tuning)
            
        elif args.mode == 'evaluate':
            # Evaluation only (requires pre-trained model)
            import joblib
            
            if not config.BEST_MODEL_FILE.exists():
                logger.error("No trained model found. Run training first.")
                sys.exit(1)
            
            # Load model and data
            pipeline.run_data_loading()
            pipeline.run_preprocessing()
            pipeline.run_feature_engineering()
            
            # Create trainer to get test data
            trainer = ModelTrainer(pipeline.df_engineered)
            trainer.prepare_data()
            
            # Load saved model
            best_model = joblib.load(config.BEST_MODEL_FILE)
            
            # Evaluate
            evaluator = ModelEvaluator(best_model, trainer.X_test, trainer.y_test, "Loaded Model")
            evaluator.evaluate_full()
        
        logger.info("\n✓ Pipeline execution completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ File not found: {str(e)}")
        logger.error(f"Please ensure {config.RAW_DATA_FILE} exists in the data/raw directory.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()







