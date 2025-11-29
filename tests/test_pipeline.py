"""
Unit tests for the hotel booking cancellation prediction pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config
from data_loader import DataLoader
from preprocess import DataPreprocessor
from feature_engineering import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample hotel booking data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'hotel': np.random.choice(['Resort Hotel', 'City Hotel'], n_samples),
        'is_canceled': np.random.choice([0, 1], n_samples),
        'lead_time': np.random.randint(0, 365, n_samples),
        'arrival_date_month': np.random.choice(['January', 'February', 'March'], n_samples),
        'stays_in_weekend_nights': np.random.randint(0, 5, n_samples),
        'stays_in_week_nights': np.random.randint(1, 8, n_samples),
        'adults': np.random.randint(1, 4, n_samples),
        'children': np.random.randint(0, 3, n_samples),
        'babies': np.random.randint(0, 2, n_samples),
        'meal': np.random.choice(['BB', 'HB', 'FB'], n_samples),
        'market_segment': np.random.choice(['Online TA', 'Direct'], n_samples),
        'distribution_channel': np.random.choice(['TA/TO', 'Direct'], n_samples),
        'previous_cancellations': np.random.randint(0, 3, n_samples),
        'previous_bookings_not_canceled': np.random.randint(0, 5, n_samples),
        'reserved_room_type': np.random.choice(['A', 'B', 'C'], n_samples),
        'assigned_room_type': np.random.choice(['A', 'B', 'C'], n_samples),
        'booking_changes': np.random.randint(0, 4, n_samples),
        'deposit_type': np.random.choice(['No Deposit', 'Non Refund'], n_samples),
        'customer_type': np.random.choice(['Transient', 'Contract'], n_samples),
        'adr': np.random.uniform(50, 300, n_samples),
        'total_of_special_requests': np.random.randint(0, 4, n_samples)
    }
    
    return pd.DataFrame(data)


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_data_summary(self, sample_data):
        """Test data summary generation."""
        loader = DataLoader()
        loader.df = sample_data
        
        summary = loader.generate_data_summary()
        
        assert 'shape' in summary
        assert 'columns' in summary
        assert 'missing_values' in summary
        assert summary['shape'] == sample_data.shape
    
    def test_target_distribution(self, sample_data):
        """Test target distribution calculation."""
        loader = DataLoader()
        loader.df = sample_data
        
        distribution = loader.get_target_distribution()
        
        assert 'counts' in distribution
        assert 'percentages' in distribution
        assert len(distribution['counts']) == 2


class TestDataPreprocessor:
    """Test data preprocessing functionality."""
    
    def test_remove_duplicates(self, sample_data):
        """Test duplicate removal."""
        # Add a duplicate row
        df_with_dup = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)
        
        preprocessor = DataPreprocessor(df_with_dup)
        df_clean = preprocessor.remove_duplicates()
        
        assert len(df_clean) < len(df_with_dup)
    
    def test_handle_missing_values(self, sample_data):
        """Test missing value handling."""
        # Add missing values
        df_with_missing = sample_data.copy()
        df_with_missing.loc[0:5, 'adr'] = np.nan
        
        preprocessor = DataPreprocessor(df_with_missing)
        df_clean = preprocessor.handle_missing_values()
        
        assert df_clean['adr'].isnull().sum() == 0
    
    def test_remove_invalid_rows(self, sample_data):
        """Test invalid row removal."""
        # Add invalid row (0 guests)
        df_with_invalid = sample_data.copy()
        df_with_invalid.loc[0, ['adults', 'children', 'babies']] = 0
        
        preprocessor = DataPreprocessor(df_with_invalid)
        df_clean = preprocessor.remove_invalid_rows()
        
        assert len(df_clean) < len(df_with_invalid)


class TestFeatureEngineer:
    """Test feature engineering functionality."""
    
    def test_total_stay_nights(self, sample_data):
        """Test total stay nights feature creation."""
        engineer = FeatureEngineer(sample_data)
        df_engineered = engineer.create_total_stay_nights()
        
        assert 'total_stay_nights' in df_engineered.columns
        assert (df_engineered['total_stay_nights'] == 
                df_engineered['stays_in_weekend_nights'] + 
                df_engineered['stays_in_week_nights']).all()
    
    def test_total_guests(self, sample_data):
        """Test total guests feature creation."""
        engineer = FeatureEngineer(sample_data)
        df_engineered = engineer.create_total_guests()
        
        assert 'total_guests' in df_engineered.columns
        assert (df_engineered['total_guests'] == 
                df_engineered['adults'] + 
                df_engineered['children'] + 
                df_engineered['babies']).all()
    
    def test_lead_time_category(self, sample_data):
        """Test lead time category feature creation."""
        engineer = FeatureEngineer(sample_data)
        df_engineered = engineer.create_lead_time_category()
        
        assert 'lead_time_category' in df_engineered.columns
        assert df_engineered['lead_time_category'].isin(['short', 'medium', 'long', 'very_long']).all()
    
    def test_all_features(self, sample_data):
        """Test creation of all features."""
        engineer = FeatureEngineer(sample_data)
        df_engineered = engineer.create_all_features()
        
        # Check that new features were created
        assert len(df_engineered.columns) > len(sample_data.columns)
        assert len(engineer.engineered_features) >= 5


class TestPipelineIntegration:
    """Test end-to-end pipeline integration."""
    
    def test_preprocessing_pipeline(self, sample_data):
        """Test complete preprocessing pipeline."""
        preprocessor = DataPreprocessor(sample_data)
        df_clean = preprocessor.preprocess()
        
        # Should have no missing values
        assert df_clean.isnull().sum().sum() == 0
        
        # Should have no duplicates
        assert df_clean.duplicated().sum() == 0
    
    def test_feature_engineering_pipeline(self, sample_data):
        """Test complete feature engineering pipeline."""
        # Preprocess first
        preprocessor = DataPreprocessor(sample_data)
        df_clean = preprocessor.preprocess()
        
        # Engineer features
        engineer = FeatureEngineer(df_clean)
        df_engineered = engineer.create_all_features()
        
        # Check that features were created
        assert 'total_stay_nights' in df_engineered.columns
        assert 'total_guests' in df_engineered.columns
        assert 'lead_time_category' in df_engineered.columns


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
