"""
Data loading and validation module.
Handles loading the hotel bookings dataset and generating initial data summaries.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict
import config

# Set up logging
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate hotel booking data."""
    
    def __init__(self, data_path: Path = config.RAW_DATA_FILE):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the raw CSV file
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file with error handling.
        
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            pd.errors.EmptyDataError: If file is empty
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error("Data file is empty")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def generate_data_summary(self) -> Dict:
        """
        Generate comprehensive summary of the dataset.
        
        Returns:
            Dictionary containing dataset summary statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        summary = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        # Numerical columns summary
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        summary['numerical_columns'] = list(numerical_cols)
        summary['numerical_stats'] = self.df[numerical_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        summary['categorical_columns'] = list(categorical_cols)
        summary['categorical_unique_counts'] = {
            col: self.df[col].nunique() for col in categorical_cols
        }
        
        # Log summary
        logger.info("=" * 80)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Shape: {summary['shape'][0]} rows × {summary['shape'][1]} columns")
        logger.info(f"Memory Usage: {summary['memory_usage']:.2f} MB")
        logger.info(f"Duplicate Rows: {summary['duplicates']}")
        logger.info(f"\nMissing Values:")
        for col, count in summary['missing_values'].items():
            if count > 0:
                pct = summary['missing_percentage'][col]
                logger.info(f"  {col}: {count} ({pct:.2f}%)")
        
        logger.info(f"\nNumerical Columns ({len(numerical_cols)}): {', '.join(numerical_cols)}")
        logger.info(f"Categorical Columns ({len(categorical_cols)}): {', '.join(categorical_cols)}")
        
        return summary
    
    def validate_data(self) -> Tuple[bool, list]:
        """
        Validate the loaded data for basic quality checks.
        
        Returns:
            Tuple of (is_valid, list of issues found)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        issues = []
        
        # Check if target column exists
        if config.TARGET_COLUMN not in self.df.columns:
            issues.append(f"Target column '{config.TARGET_COLUMN}' not found in dataset")
        
        # Check for completely empty columns
        empty_cols = self.df.columns[self.df.isnull().all()].tolist()
        if empty_cols:
            issues.append(f"Completely empty columns found: {empty_cols}")
        
        # Check for columns with single unique value
        single_value_cols = [col for col in self.df.columns 
                            if self.df[col].nunique() == 1]
        if single_value_cols:
            issues.append(f"Columns with single unique value: {single_value_cols}")
        
        # Check for excessive missing values (>50%)
        high_missing = [col for col, pct in 
                       (self.df.isnull().sum() / len(self.df) * 100).items() 
                       if pct > 50]
        if high_missing:
            issues.append(f"Columns with >50% missing values: {high_missing}")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("✓ Data validation passed")
        else:
            logger.warning(f"✗ Data validation found {len(issues)} issue(s):")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    def get_target_distribution(self) -> Dict:
        """
        Get distribution of target variable.
        
        Returns:
            Dictionary with target variable distribution
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if config.TARGET_COLUMN not in self.df.columns:
            raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found")
        
        target_counts = self.df[config.TARGET_COLUMN].value_counts()
        target_percentages = self.df[config.TARGET_COLUMN].value_counts(normalize=True) * 100
        
        distribution = {
            'counts': target_counts.to_dict(),
            'percentages': target_percentages.to_dict()
        }
        
        logger.info(f"\nTarget Variable Distribution ({config.TARGET_COLUMN}):")
        for value, count in target_counts.items():
            pct = target_percentages[value]
            logger.info(f"  {value}: {count} ({pct:.2f}%)")
        
        # Check for class imbalance
        if len(target_counts) == 2:
            minority_pct = target_percentages.min()
            if minority_pct < 40:
                logger.warning(f"⚠ Class imbalance detected: minority class = {minority_pct:.2f}%")
                distribution['imbalanced'] = True
            else:
                distribution['imbalanced'] = False
        
        return distribution


def load_and_validate_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to load and validate data in one step.
    
    Returns:
        Tuple of (DataFrame, summary dictionary)
    """
    loader = DataLoader()
    df = loader.load_data()
    summary = loader.generate_data_summary()
    loader.validate_data()
    loader.get_target_distribution()
    
    return df, summary


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load and validate data
    df, summary = load_and_validate_data()
    print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
