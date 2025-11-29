"""
Feature Engineering module.
Creates new features from existing data to improve model performance.
"""

import pandas as pd
import numpy as np
import logging
from typing import List
import config

# Set up logging
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features for hotel booking cancellation prediction."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize FeatureEngineer.
        
        Args:
            df: DataFrame to engineer features from
        """
        self.df = df.copy()
        self.engineered_features = []
        
    def create_total_stay_nights(self) -> pd.DataFrame:
        """
        Feature 1: Total stay nights
        
        Business Rationale:
        Longer stays may have different cancellation patterns than short stays.
        Combines weekend and weekday nights for total booking duration.
        
        Returns:
            DataFrame with new feature
        """
        if all(col in self.df.columns for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
            self.df['total_stay_nights'] = (
                self.df['stays_in_weekend_nights'] + self.df['stays_in_week_nights']
            )
            self.engineered_features.append('total_stay_nights')
            logger.info("✓ Created feature: total_stay_nights")
        return self.df
    
    def create_total_guests(self) -> pd.DataFrame:
        """
        Feature 2: Total guests
        
        Business Rationale:
        Group size affects cancellation likelihood. Larger groups may have
        more complex coordination needs, potentially affecting cancellation rates.
        
        Returns:
            DataFrame with new feature
        """
        if all(col in self.df.columns for col in ['adults', 'children', 'babies']):
            self.df['total_guests'] = (
                self.df['adults'] + self.df['children'] + self.df['babies']
            )
            self.engineered_features.append('total_guests')
            logger.info("✓ Created feature: total_guests")
        return self.df
    
    def create_lead_time_category(self) -> pd.DataFrame:
        """
        Feature 3: Lead time category
        
        Business Rationale:
        Bookings made far in advance may have different cancellation patterns
        than last-minute bookings. Categorizing lead time helps capture
        non-linear relationships.
        
        Categories:
        - short: 0-30 days
        - medium: 31-90 days
        - long: 91-365 days
        - very_long: >365 days
        
        Returns:
            DataFrame with new feature
        """
        if 'lead_time' in self.df.columns:
            self.df['lead_time_category'] = pd.cut(
                self.df['lead_time'],
                bins=config.LEAD_TIME_BINS,
                labels=config.LEAD_TIME_LABELS,
                include_lowest=True
            )
            
            # Convert to string for encoding later
            self.df['lead_time_category'] = self.df['lead_time_category'].astype(str)
            self.engineered_features.append('lead_time_category')
            logger.info("✓ Created feature: lead_time_category")
        return self.df
    
    def create_adr_per_person(self) -> pd.DataFrame:
        """
        Feature 4: Average Daily Rate per person
        
        Business Rationale:
        Price per person is a better indicator of value than total ADR.
        High per-person rates might indicate luxury bookings with different
        cancellation patterns.
        
        Returns:
            DataFrame with new feature
        """
        if 'adr' in self.df.columns and 'total_guests' in self.df.columns:
            # Avoid division by zero
            self.df['adr_per_person'] = np.where(
                self.df['total_guests'] > 0,
                self.df['adr'] / self.df['total_guests'],
                self.df['adr']
            )
            self.engineered_features.append('adr_per_person')
            logger.info("✓ Created feature: adr_per_person")
        return self.df
    
    def create_is_weekend_booking(self) -> pd.DataFrame:
        """
        Feature 5: Weekend booking flag
        
        Business Rationale:
        Weekend-only stays may represent leisure travel with different
        cancellation patterns than business travel (weekday stays).
        
        Returns:
            DataFrame with new feature
        """
        if all(col in self.df.columns for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
            self.df['is_weekend_booking'] = (
                (self.df['stays_in_weekend_nights'] > 0) & 
                (self.df['stays_in_week_nights'] == 0)
            ).astype(int)
            self.engineered_features.append('is_weekend_booking')
            logger.info("✓ Created feature: is_weekend_booking")
        return self.df
    
    def create_has_special_requests(self) -> pd.DataFrame:
        """
        Feature 6: Has special requests flag
        
        Business Rationale:
        Customers who make special requests show higher engagement and
        commitment to their booking, potentially reducing cancellation likelihood.
        
        Returns:
            DataFrame with new feature
        """
        if 'total_of_special_requests' in self.df.columns:
            self.df['has_special_requests'] = (
                self.df['total_of_special_requests'] > 0
            ).astype(int)
            self.engineered_features.append('has_special_requests')
            logger.info("✓ Created feature: has_special_requests")
        return self.df
    
    def create_has_booking_changes(self) -> pd.DataFrame:
        """
        Feature 7: Has booking changes flag
        
        Business Rationale:
        Bookings that have been modified show active customer engagement.
        Changes might indicate uncertainty or flexibility needs, affecting
        cancellation probability.
        
        Returns:
            DataFrame with new feature
        """
        if 'booking_changes' in self.df.columns:
            self.df['has_booking_changes'] = (
                self.df['booking_changes'] > 0
            ).astype(int)
            self.engineered_features.append('has_booking_changes')
            logger.info("✓ Created feature: has_booking_changes")
        return self.df
    
    def create_is_family_booking(self) -> pd.DataFrame:
        """
        Feature 8: Family booking flag
        
        Business Rationale:
        Bookings with children may represent family vacations with different
        cancellation patterns than adult-only bookings.
        
        Returns:
            DataFrame with new feature
        """
        if all(col in self.df.columns for col in ['children', 'babies']):
            self.df['is_family_booking'] = (
                (self.df['children'] > 0) | (self.df['babies'] > 0)
            ).astype(int)
            self.engineered_features.append('is_family_booking')
            logger.info("✓ Created feature: is_family_booking")
        return self.df
    
    def create_previous_cancellation_rate(self) -> pd.DataFrame:
        """
        Feature 9: Previous cancellation rate
        
        Business Rationale:
        Customer history is a strong predictor. High previous cancellation
        rates indicate higher likelihood of future cancellations.
        
        Returns:
            DataFrame with new feature
        """
        if all(col in self.df.columns for col in ['previous_cancellations', 'previous_bookings_not_canceled']):
            total_previous = (
                self.df['previous_cancellations'] + 
                self.df['previous_bookings_not_canceled']
            )
            
            self.df['previous_cancellation_rate'] = np.where(
                total_previous > 0,
                self.df['previous_cancellations'] / total_previous,
                0
            )
            self.engineered_features.append('previous_cancellation_rate')
            logger.info("✓ Created feature: previous_cancellation_rate")
        return self.df
    
    def create_arrival_date_features(self) -> pd.DataFrame:
        """
        Feature 10-11: Arrival date features
        
        Business Rationale:
        Seasonal patterns and specific months may have different cancellation rates.
        Peak season bookings might be less likely to cancel.
        
        Returns:
            DataFrame with new feature
        """
        # Create arrival month number if arrival_date_month exists
        if 'arrival_date_month' in self.df.columns:
            month_map = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            self.df['arrival_month_num'] = self.df['arrival_date_month'].map(month_map)
            self.engineered_features.append('arrival_month_num')
            logger.info("✓ Created feature: arrival_month_num")
            
            # Create season feature
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                elif month in [6, 7, 8]:
                    return 'Summer'
                else:
                    return 'Fall'
            
            self.df['arrival_season'] = self.df['arrival_month_num'].apply(get_season)
            self.engineered_features.append('arrival_season')
            logger.info("✓ Created feature: arrival_season")
        
        return self.df
    
    def create_room_type_match(self) -> pd.DataFrame:
        """
        Feature 12: Room type match
        
        Business Rationale:
        When reserved room type doesn't match assigned room type, it may
        indicate customer dissatisfaction, potentially increasing cancellation.
        
        Returns:
            DataFrame with new feature
        """
        if all(col in self.df.columns for col in ['reserved_room_type', 'assigned_room_type']):
            self.df['room_type_match'] = (
                self.df['reserved_room_type'] == self.df['assigned_room_type']
            ).astype(int)
            self.engineered_features.append('room_type_match')
            logger.info("✓ Created feature: room_type_match")
        return self.df
    
    def create_all_features(self) -> pd.DataFrame:
        """
        Create all engineered features.
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering")
        logger.info(f"Initial features: {self.df.shape[1]}")
        
        # Create all features
        self.create_total_stay_nights()
        self.create_total_guests()
        self.create_lead_time_category()
        self.create_adr_per_person()
        self.create_is_weekend_booking()
        self.create_has_special_requests()
        self.create_has_booking_changes()
        self.create_is_family_booking()
        self.create_previous_cancellation_rate()
        self.create_arrival_date_features()
        self.create_room_type_match()
        
        logger.info(f"Final features: {self.df.shape[1]}")
        logger.info(f"Engineered {len(self.engineered_features)} new features:")
        for feature in self.engineered_features:
            logger.info(f"  - {feature}")
        
        return self.df
    
    def get_feature_descriptions(self) -> dict:
        """
        Get descriptions of all engineered features.
        
        Returns:
            Dictionary mapping feature names to their descriptions
        """
        descriptions = {
            'total_stay_nights': 'Total nights of stay (weekend + weekday)',
            'total_guests': 'Total number of guests (adults + children + babies)',
            'lead_time_category': 'Categorized booking lead time (short/medium/long/very_long)',
            'adr_per_person': 'Average daily rate per person',
            'is_weekend_booking': 'Flag for weekend-only bookings',
            'has_special_requests': 'Flag for bookings with special requests',
            'has_booking_changes': 'Flag for bookings that were modified',
            'is_family_booking': 'Flag for bookings with children or babies',
            'previous_cancellation_rate': 'Rate of previous cancellations by customer',
            'arrival_month_num': 'Numeric representation of arrival month',
            'arrival_season': 'Season of arrival (Winter/Spring/Summer/Fall)',
            'room_type_match': 'Flag for whether reserved and assigned room types match'
        }
        return {k: v for k, v in descriptions.items() if k in self.engineered_features}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to engineer features.
    
    Args:
        df: DataFrame to engineer features from
        
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer(df)
    df_engineered = engineer.create_all_features()
    
    # Print feature descriptions
    descriptions = engineer.get_feature_descriptions()
    logger.info("\nFeature Descriptions:")
    for feature, description in descriptions.items():
        logger.info(f"  {feature}: {description}")
    
    return df_engineered


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from data_loader import load_and_validate_data
    from preprocess import preprocess_data
    
    df, _ = load_and_validate_data()
    df_clean = preprocess_data(df)
    df_engineered = engineer_features(df_clean)
    print(f"\nFinal dataset shape: {df_engineered.shape}")
