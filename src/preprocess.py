"""
Data preprocessing module.
Handles data cleaning, missing value imputation, outlier treatment, and encoding.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder, StandardScaler
import config

# Set up logging
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess hotel booking data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataPreprocessor.
        
        Args:
            df: DataFrame to preprocess
        """
        self.df = df.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed_rows = initial_rows - len(self.df)
        
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} duplicate rows")
        else:
            logger.info("No duplicate rows found")
        
        return self.df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values using configured strategies.
        
        Strategy:
        - Numerical: Fill with median (robust to outliers)
        - Categorical: Fill with mode (most frequent value)
        - Drop rows if missing critical information
        
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values")
        
        missing_before = self.df.isnull().sum().sum()
        
        # Get numerical and categorical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Handle numerical missing values
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                if config.MISSING_VALUE_STRATEGY['numerical'] == 'median':
                    fill_value = self.df[col].median()
                    self.df[col].fillna(fill_value, inplace=True)
                    logger.info(f"  Filled {col} with median: {fill_value:.2f}")
                elif config.MISSING_VALUE_STRATEGY['numerical'] == 'mean':
                    fill_value = self.df[col].mean()
                    self.df[col].fillna(fill_value, inplace=True)
                    logger.info(f"  Filled {col} with mean: {fill_value:.2f}")
        
        # Handle categorical missing values
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                if config.MISSING_VALUE_STRATEGY['categorical'] == 'mode':
                    fill_value = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                    self.df[col].fillna(fill_value, inplace=True)
                    logger.info(f"  Filled {col} with mode: {fill_value}")
                elif config.MISSING_VALUE_STRATEGY['categorical'] == 'constant':
                    self.df[col].fillna('Unknown', inplace=True)
                    logger.info(f"  Filled {col} with 'Unknown'")
        
        missing_after = self.df.isnull().sum().sum()
        logger.info(f"Missing values reduced from {missing_before} to {missing_after}")
        
        return self.df
    
    def convert_data_types(self) -> pd.DataFrame:
        """
        Convert data types appropriately.
        
        Returns:
            DataFrame with corrected data types
        """
        logger.info("Converting data types")
        
        # Convert date columns if they exist
        date_columns = ['reservation_status_date', 'arrival_date']
        for col in date_columns:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    logger.info(f"  Converted {col} to datetime")
                except:
                    logger.warning(f"  Could not convert {col} to datetime")
        
        # Ensure target is integer
        if config.TARGET_COLUMN in self.df.columns:
            self.df[config.TARGET_COLUMN] = self.df[config.TARGET_COLUMN].astype(int)
        
        # Convert specific columns to appropriate types
        int_columns = ['adults', 'children', 'babies', 'is_repeated_guest']
        for col in int_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0).astype(int)
        
        return self.df
    
    def remove_invalid_rows(self) -> pd.DataFrame:
        """
        Remove rows with invalid data.
        
        Business rules:
        - Remove bookings with 0 guests (adults + children + babies = 0)
        - Remove bookings with negative ADR
        - Remove bookings with 0 stay nights
        
        Returns:
            DataFrame with invalid rows removed
        """
        logger.info("Removing invalid rows")
        initial_rows = len(self.df)
        
        # Remove rows with 0 guests
        if all(col in self.df.columns for col in ['adults', 'children', 'babies']):
            self.df['total_guests_temp'] = self.df['adults'] + self.df['children'] + self.df['babies']
            before = len(self.df)
            self.df = self.df[self.df['total_guests_temp'] > 0]
            removed = before - len(self.df)
            if removed > 0:
                logger.info(f"  Removed {removed} rows with 0 guests")
            self.df = self.df.drop('total_guests_temp', axis=1)
        
        # Remove rows with negative ADR
        if 'adr' in self.df.columns:
            before = len(self.df)
            self.df = self.df[self.df['adr'] >= 0]
            removed = before - len(self.df)
            if removed > 0:
                logger.info(f"  Removed {removed} rows with negative ADR")
        
        # Remove rows with 0 stay nights
        if all(col in self.df.columns for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
            self.df['total_nights_temp'] = self.df['stays_in_weekend_nights'] + self.df['stays_in_week_nights']
            before = len(self.df)
            self.df = self.df[self.df['total_nights_temp'] > 0]
            removed = before - len(self.df)
            if removed > 0:
                logger.info(f"  Removed {removed} rows with 0 stay nights")
            self.df = self.df.drop('total_nights_temp', axis=1)
        
        total_removed = initial_rows - len(self.df)
        logger.info(f"Total invalid rows removed: {total_removed}")
        
        return self.df
    
    def detect_and_treat_outliers(self, columns: List[str] = None) -> pd.DataFrame:
        """
        Detect and treat outliers using IQR method.
        
        Treatment: Cap outliers at 1.5 * IQR boundaries (Winsorization)
        
        Args:
            columns: List of columns to check for outliers. If None, checks all numerical columns.
            
        Returns:
            DataFrame with outliers treated
        """
        logger.info("Detecting and treating outliers using IQR method")
        
        if columns is None:
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            # Focus on key columns that typically have outliers
            columns = ['lead_time', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights']
            columns = [col for col in columns if col in numerical_cols]
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - config.OUTLIER_THRESHOLD * IQR
            upper_bound = Q3 + config.OUTLIER_THRESHOLD * IQR
            
            # Count outliers
            outliers_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            
            if outliers_count > 0:
                # Cap outliers (Winsorization)
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"  {col}: Capped {outliers_count} outliers at [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return self.df
    
    def encode_categorical_variables(self) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Strategy:
        - Binary categorical: Label Encoding (0/1)
        - Multi-class categorical: One-Hot Encoding
        
        Returns:
            DataFrame with encoded categorical variables
        """
        logger.info("Encoding categorical variables")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col in config.CATEGORICAL_FEATURES]
        
        for col in categorical_cols:
            if col not in self.df.columns:
                continue
            
            unique_values = self.df[col].nunique()
            
            if unique_values == 2:
                # Binary encoding using Label Encoder
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"  {col}: Label encoded (binary)")
            else:
                # One-hot encoding for multi-class
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df = self.df.drop(col, axis=1)
                logger.info(f"  {col}: One-hot encoded ({unique_values} categories)")
        
        return self.df
    
    def drop_unnecessary_columns(self) -> pd.DataFrame:
        """
        Drop columns that are not needed for modeling.
        
        Returns:
            DataFrame with unnecessary columns removed
        """
        logger.info("Dropping unnecessary columns")
        
        columns_to_drop = [col for col in config.COLUMNS_TO_DROP if col in self.df.columns]
        
        if columns_to_drop:
            self.df = self.df.drop(columns_to_drop, axis=1)
            logger.info(f"  Dropped columns: {', '.join(columns_to_drop)}")
        
        # Drop any remaining date columns
        date_cols = self.df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            self.df = self.df.drop(date_cols, axis=1)
            logger.info(f"  Dropped date columns: {', '.join(date_cols)}")
        
        return self.df
    
    def preprocess(self) -> pd.DataFrame:
        """
        Run complete preprocessing pipeline.
        
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing pipeline")
        logger.info(f"Initial shape: {self.df.shape}")
        
        self.remove_duplicates()
        self.handle_missing_values()
        self.convert_data_types()
        self.remove_invalid_rows()
        self.detect_and_treat_outliers()
        
        logger.info(f"Final shape after cleaning: {self.df.shape}")
        logger.info("Data preprocessing completed")
        
        return self.df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to preprocess data.
    
    Args:
        df: DataFrame to preprocess
        
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = DataPreprocessor(df)
    return preprocessor.preprocess()


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from data_loader import load_and_validate_data
    
    df, _ = load_and_validate_data()
    df_clean = preprocess_data(df)
    print(f"\nPreprocessed data shape: {df_clean.shape}")
