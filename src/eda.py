"""
Exploratory Data Analysis (EDA) module.
Generates visualizations and statistical analysis of the hotel bookings dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict
import config

# Set up logging
logger = logging.getLogger(__name__)

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class EDAAnalyzer:
    """Perform exploratory data analysis on hotel booking data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA Analyzer.
        
        Args:
            df: DataFrame to analyze
        """
        self.df = df.copy()
        self.plots_dir = config.EDA_PLOTS_DIR
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_target_distribution(self):
        """Plot distribution of target variable."""
        logger.info("Plotting target variable distribution")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        target_counts = self.df[config.TARGET_COLUMN].value_counts()
        axes[0].bar(target_counts.index, target_counts.values, color=['#2ecc71', '#e74c3c'])
        axes[0].set_xlabel('Cancellation Status', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Distribution of Booking Cancellations', fontsize=14, fontweight='bold')
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['Not Canceled', 'Canceled'])
        
        # Add value labels on bars
        for i, v in enumerate(target_counts.values):
            axes[0].text(i, v + 500, str(v), ha='center', fontweight='bold')
        
        # Pie chart
        colors = ['#2ecc71', '#e74c3c']
        axes[1].pie(target_counts.values, labels=['Not Canceled', 'Canceled'], 
                   autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1].set_title('Cancellation Rate', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_numerical_distributions(self):
        """Plot distributions of numerical features."""
        logger.info("Plotting numerical feature distributions")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != config.TARGET_COLUMN]
        
        # Limit to most important numerical columns
        important_cols = ['lead_time', 'adr', 'stays_in_weekend_nights', 
                         'stays_in_week_nights', 'adults', 'children', 'babies',
                         'previous_cancellations', 'previous_bookings_not_canceled',
                         'booking_changes', 'days_in_waiting_list']
        
        cols_to_plot = [col for col in important_cols if col in numerical_cols]
        
        n_cols = 3
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(cols_to_plot):
            axes[idx].hist(self.df[col].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(col, fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_categorical_distributions(self):
        """Plot distributions of categorical features."""
        logger.info("Plotting categorical feature distributions")
        
        categorical_cols = ['hotel', 'meal', 'market_segment', 'distribution_channel',
                           'reserved_room_type', 'deposit_type', 'customer_type']
        
        cols_to_plot = [col for col in categorical_cols if col in self.df.columns]
        
        n_cols = 2
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(cols_to_plot):
            value_counts = self.df[col].value_counts()
            axes[idx].barh(range(len(value_counts)), value_counts.values, color='coral')
            axes[idx].set_yticks(range(len(value_counts)))
            axes[idx].set_yticklabels(value_counts.index)
            axes[idx].set_xlabel('Count', fontsize=10)
            axes[idx].set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_correlation_matrix(self):
        """Plot correlation matrix of numerical features."""
        logger.info("Plotting correlation matrix")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numerical_cols].corr()
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_target_vs_features(self):
        """Plot relationship between target and key features."""
        logger.info("Plotting target vs features")
        
        # Numerical features vs target
        numerical_features = ['lead_time', 'adr', 'total_of_special_requests']
        numerical_features = [col for col in numerical_features if col in self.df.columns]
        
        if numerical_features:
            fig, axes = plt.subplots(1, len(numerical_features), figsize=(18, 5))
            if len(numerical_features) == 1:
                axes = [axes]
            
            for idx, col in enumerate(numerical_features):
                self.df.boxplot(column=col, by=config.TARGET_COLUMN, ax=axes[idx])
                axes[idx].set_xlabel('Cancellation Status', fontsize=10)
                axes[idx].set_ylabel(col, fontsize=10)
                axes[idx].set_title(f'{col} by Cancellation Status', fontsize=11, fontweight='bold')
                axes[idx].get_figure().suptitle('')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'target_vs_numerical.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Categorical features vs target
        categorical_features = ['hotel', 'deposit_type', 'customer_type']
        categorical_features = [col for col in categorical_features if col in self.df.columns]
        
        if categorical_features:
            fig, axes = plt.subplots(1, len(categorical_features), figsize=(18, 5))
            if len(categorical_features) == 1:
                axes = [axes]
            
            for idx, col in enumerate(categorical_features):
                ct = pd.crosstab(self.df[col], self.df[config.TARGET_COLUMN], normalize='index') * 100
                ct.plot(kind='bar', stacked=False, ax=axes[idx], color=['#2ecc71', '#e74c3c'])
                axes[idx].set_xlabel(col, fontsize=10)
                axes[idx].set_ylabel('Percentage', fontsize=10)
                axes[idx].set_title(f'Cancellation Rate by {col}', fontsize=11, fontweight='bold')
                axes[idx].legend(['Not Canceled', 'Canceled'])
                axes[idx].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'target_vs_categorical.png', dpi=300, bbox_inches='tight')
            plt.close()
        
    def plot_missing_values(self):
        """Visualize missing values in the dataset."""
        logger.info("Plotting missing values")
        
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) > 0:
            plt.figure(figsize=(12, 6))
            missing_pct = (missing / len(self.df)) * 100
            
            bars = plt.barh(range(len(missing)), missing_pct.values, color='indianred')
            plt.yticks(range(len(missing)), missing.index)
            plt.xlabel('Percentage of Missing Values (%)', fontsize=12)
            plt.title('Missing Values by Feature', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            
            # Add percentage labels
            for i, (bar, val) in enumerate(zip(bars, missing_pct.values)):
                plt.text(val + 0.5, i, f'{val:.1f}%', va='center')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'missing_values.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            logger.info("No missing values found in dataset")
    
    def detect_outliers(self):
        """Detect and visualize outliers in numerical features."""
        logger.info("Detecting outliers")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != config.TARGET_COLUMN]
        
        # Focus on key columns
        key_cols = ['lead_time', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights']
        cols_to_check = [col for col in key_cols if col in numerical_cols]
        
        if cols_to_check:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for idx, col in enumerate(cols_to_check[:4]):
                axes[idx].boxplot(self.df[col].dropna(), vert=True)
                axes[idx].set_ylabel(col, fontsize=10)
                axes[idx].set_title(f'Outlier Detection: {col}', fontsize=11, fontweight='bold')
                axes[idx].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'outliers_boxplot.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_eda_report(self) -> str:
        """
        Generate comprehensive EDA report.
        
        Returns:
            String containing the EDA report
        """
        logger.info("Generating EDA report")
        
        report = []
        report.append("=" * 80)
        report.append("EXPLORATORY DATA ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Dataset overview
        report.append("1. DATASET OVERVIEW")
        report.append(f"   Rows: {self.df.shape[0]:,}")
        report.append(f"   Columns: {self.df.shape[1]}")
        report.append("")
        
        # Missing values
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        report.append("2. MISSING VALUES")
        if len(missing) > 0:
            for col, count in missing.items():
                pct = (count / len(self.df)) * 100
                report.append(f"   {col}: {count} ({pct:.2f}%)")
        else:
            report.append("   No missing values found")
        report.append("")
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        report.append("3. DUPLICATE ROWS")
        report.append(f"   Total duplicates: {duplicates}")
        report.append("")
        
        # Target distribution
        if config.TARGET_COLUMN in self.df.columns:
            report.append("4. TARGET VARIABLE DISTRIBUTION")
            target_counts = self.df[config.TARGET_COLUMN].value_counts()
            for value, count in target_counts.items():
                pct = (count / len(self.df)) * 100
                report.append(f"   {value}: {count} ({pct:.2f}%)")
            report.append("")
        
        # Numerical features summary
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        report.append(f"5. NUMERICAL FEATURES ({len(numerical_cols)})")
        report.append(f"   {', '.join(numerical_cols)}")
        report.append("")
        
        # Categorical features summary
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        report.append(f"6. CATEGORICAL FEATURES ({len(categorical_cols)})")
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            report.append(f"   {col}: {unique_count} unique values")
        report.append("")
        
        report.append("=" * 80)
        report.append("All visualizations saved to: " + str(self.plots_dir))
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report to file
        with open(config.EDA_REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"EDA report saved to {config.EDA_REPORT_FILE}")
        
        return report_text
    
    def run_full_eda(self):
        """Run complete EDA pipeline."""
        logger.info("Starting full EDA analysis")
        
        self.plot_target_distribution()
        self.plot_numerical_distributions()
        self.plot_categorical_distributions()
        self.plot_correlation_matrix()
        self.plot_target_vs_features()
        self.plot_missing_values()
        self.detect_outliers()
        report = self.generate_eda_report()
        
        logger.info("EDA analysis completed")
        print(report)
        
        return report


def perform_eda(df: pd.DataFrame) -> str:
    """
    Convenience function to perform EDA.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        EDA report string
    """
    analyzer = EDAAnalyzer(df)
    return analyzer.run_full_eda()


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from data_loader import load_and_validate_data
    
    df, _ = load_and_validate_data()
    perform_eda(df)
