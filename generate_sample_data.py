"""
Sample data generator for hotel booking cancellation prediction.
Creates realistic synthetic data for testing the pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data(n_samples: int = 10000, output_path: str = None) -> pd.DataFrame:
    """
    Generate sample hotel booking data.
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save CSV file. If None, doesn't save.
        
    Returns:
        DataFrame with sample hotel booking data
    """
    np.random.seed(42)
    
    print(f"Generating {n_samples} sample hotel bookings...")
    
    # Generate realistic data
    data = {
        # Hotel type
        'hotel': np.random.choice(['Resort Hotel', 'City Hotel'], n_samples, p=[0.4, 0.6]),
        
        # Target variable - realistic cancellation rate around 37%
        'is_canceled': np.random.choice([0, 1], n_samples, p=[0.63, 0.37]),
        
        # Booking characteristics
        'lead_time': np.random.gamma(2, 50, n_samples).astype(int),  # Skewed distribution
        'arrival_date_year': np.random.choice([2022, 2023], n_samples),
        'arrival_date_month': np.random.choice([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ], n_samples),
        'arrival_date_week_number': np.random.randint(1, 53, n_samples),
        'arrival_date_day_of_month': np.random.randint(1, 29, n_samples),
        
        # Stay duration
        'stays_in_weekend_nights': np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'stays_in_week_nights': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], n_samples, 
                                                  p=[0.05, 0.2, 0.25, 0.2, 0.15, 0.1, 0.03, 0.02]),
        
        # Guest information
        'adults': np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.6, 0.15, 0.05]),
        'children': np.random.choice([0, 1, 2], n_samples, p=[0.8, 0.15, 0.05]),
        'babies': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        
        # Meal type
        'meal': np.random.choice(['BB', 'HB', 'FB', 'SC', 'Undefined'], n_samples,
                                p=[0.6, 0.15, 0.05, 0.15, 0.05]),
        
        # Market segment
        'market_segment': np.random.choice([
            'Online TA', 'Offline TA/TO', 'Direct', 'Corporate', 'Groups', 'Complementary'
        ], n_samples, p=[0.47, 0.19, 0.12, 0.08, 0.08, 0.06]),
        
        # Distribution channel
        'distribution_channel': np.random.choice([
            'TA/TO', 'Direct', 'Corporate', 'GDS'
        ], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
        
        # Customer type
        'is_repeated_guest': np.random.choice([0, 1], n_samples, p=[0.97, 0.03]),
        
        # Previous bookings
        'previous_cancellations': np.random.choice([0, 1, 2, 3], n_samples, 
                                                   p=[0.92, 0.05, 0.02, 0.01]),
        'previous_bookings_not_canceled': np.random.choice([0, 1, 2, 3, 4, 5], n_samples,
                                                           p=[0.85, 0.08, 0.04, 0.02, 0.005, 0.005]),
        
        # Room information
        'reserved_room_type': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_samples,
                                              p=[0.35, 0.15, 0.12, 0.15, 0.1, 0.08, 0.05]),
        'assigned_room_type': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_samples,
                                              p=[0.33, 0.16, 0.13, 0.14, 0.11, 0.08, 0.05]),
        
        # Booking changes
        'booking_changes': np.random.choice([0, 1, 2, 3, 4], n_samples,
                                           p=[0.75, 0.15, 0.06, 0.03, 0.01]),
        
        # Deposit type
        'deposit_type': np.random.choice(['No Deposit', 'Non Refund', 'Refundable'], n_samples,
                                        p=[0.88, 0.10, 0.02]),
        
        # Waiting list
        'days_in_waiting_list': np.random.choice([0, 1, 2, 3, 4, 5], n_samples,
                                                 p=[0.97, 0.01, 0.01, 0.005, 0.003, 0.002]),
        
        # Customer type
        'customer_type': np.random.choice(['Transient', 'Contract', 'Transient-Party', 'Group'], 
                                         n_samples, p=[0.75, 0.15, 0.08, 0.02]),
        
        # Average daily rate (price)
        'adr': np.random.gamma(3, 30, n_samples),  # Skewed distribution
        
        # Parking
        'required_car_parking_spaces': np.random.choice([0, 1, 2], n_samples, p=[0.92, 0.07, 0.01]),
        
        # Special requests
        'total_of_special_requests': np.random.choice([0, 1, 2, 3, 4, 5], n_samples,
                                                      p=[0.70, 0.18, 0.08, 0.03, 0.007, 0.003]),
        
        # Reservation status (for reference, will be dropped in preprocessing)
        'reservation_status': np.random.choice(['Check-Out', 'Canceled', 'No-Show'], n_samples,
                                              p=[0.60, 0.37, 0.03])
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic correlations
    # Higher lead time -> higher cancellation
    high_lead_time = df['lead_time'] > 200
    df.loc[high_lead_time, 'is_canceled'] = np.random.choice([0, 1], high_lead_time.sum(), p=[0.4, 0.6])
    
    # No deposit -> higher cancellation
    no_deposit = df['deposit_type'] == 'No Deposit'
    df.loc[no_deposit, 'is_canceled'] = np.random.choice([0, 1], no_deposit.sum(), p=[0.55, 0.45])
    
    # Previous cancellations -> higher cancellation
    prev_cancel = df['previous_cancellations'] > 0
    df.loc[prev_cancel, 'is_canceled'] = np.random.choice([0, 1], prev_cancel.sum(), p=[0.3, 0.7])
    
    # Add some missing values (realistic scenario)
    missing_cols = ['children', 'meal', 'market_segment']
    for col in missing_cols:
        if col in df.columns:
            missing_idx = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
            df.loc[missing_idx, col] = np.nan
    
    # Add a few duplicates
    duplicate_idx = np.random.choice(df.index, size=int(0.001 * len(df)), replace=False)
    df = pd.concat([df, df.loc[duplicate_idx]], ignore_index=True)
    
    print(f"✓ Generated {len(df)} samples (including {len(duplicate_idx)} duplicates)")
    print(f"✓ Cancellation rate: {df['is_canceled'].mean()*100:.1f}%")
    print(f"✓ Missing values: {df.isnull().sum().sum()} total")
    
    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample hotel booking data')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/raw/hotel_bookings.csv', 
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    df = generate_sample_data(n_samples=args.samples, output_path=args.output)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
