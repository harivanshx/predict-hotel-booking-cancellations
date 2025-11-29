"""
Model evaluation module.
Evaluates model performance using multiple metrics and generates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import config

# Set up logging
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate machine learning models for cancellation prediction."""
    
    def __init__(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model"):
        """
        Initialize ModelEvaluator.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for reporting
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}
        
    def generate_predictions(self):
        """Generate predictions on test set."""
        logger.info(f"Generating predictions for {self.model_name}")
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        if self.y_pred is None:
            self.generate_predictions()
        
        logger.info(f"Calculating metrics for {self.model_name}")
        
        self.metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_test, self.y_pred, zero_division=0),
            'f1': f1_score(self.y_test, self.y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba)
        }
        
        logger.info(f"\nMetrics for {self.model_name}:")
        logger.info(f"  Accuracy:  {self.metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {self.metrics['precision']:.4f}")
        logger.info(f"  Recall:    {self.metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {self.metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:   {self.metrics['roc_auc']:.4f}")
        
        return self.metrics
    
    def plot_confusion_matrix(self, save_path: str = None):
        """
        Plot confusion matrix.
        
        Args:
            save_path: Path to save the plot. If None, saves to default location.
        """
        if self.y_pred is None:
            self.generate_predictions()
        
        logger.info(f"Plotting confusion matrix for {self.model_name}")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Not Canceled', 'Canceled'],
                   yticklabels=['Not Canceled', 'Canceled'])
        plt.title(f'Confusion Matrix - {self.model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add percentage annotations
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = config.EDA_PLOTS_DIR / f'confusion_matrix_{self.model_name.lower().replace(" ", "_")}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved confusion matrix to {save_path}")
        
    def plot_roc_curve(self, save_path: str = None):
        """
        Plot ROC curve.
        
        Args:
            save_path: Path to save the plot. If None, saves to default location.
        """
        if self.y_pred_proba is None:
            self.generate_predictions()
        
        logger.info(f"Plotting ROC curve for {self.model_name}")
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = config.EDA_PLOTS_DIR / f'roc_curve_{self.model_name.lower().replace(" ", "_")}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved ROC curve to {save_path}")
        
    def plot_precision_recall_curve(self, save_path: str = None):
        """
        Plot Precision-Recall curve.
        
        Args:
            save_path: Path to save the plot. If None, saves to default location.
        """
        if self.y_pred_proba is None:
            self.generate_predictions()
        
        logger.info(f"Plotting Precision-Recall curve for {self.model_name}")
        
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {self.model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = config.EDA_PLOTS_DIR / f'pr_curve_{self.model_name.lower().replace(" ", "_")}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved Precision-Recall curve to {save_path}")
        
    def plot_feature_importance(self, feature_importance_df: pd.DataFrame, top_n: int = 20, save_path: str = None):
        """
        Plot feature importance.
        
        Args:
            feature_importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to display
            save_path: Path to save the plot. If None, saves to default location.
        """
        if feature_importance_df is None or len(feature_importance_df) == 0:
            logger.warning("No feature importance data available")
            return
        
        logger.info(f"Plotting feature importance for {self.model_name}")
        
        # Get top N features
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - {self.model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = config.EDA_PLOTS_DIR / f'feature_importance_{self.model_name.lower().replace(" ", "_")}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved feature importance to {save_path}")
        
    def generate_classification_report(self) -> str:
        """
        Generate detailed classification report.
        
        Returns:
            Classification report as string
        """
        if self.y_pred is None:
            self.generate_predictions()
        
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=['Not Canceled', 'Canceled'],
            digits=4
        )
        
        logger.info(f"\nClassification Report for {self.model_name}:")
        logger.info("\n" + report)
        
        return report
    
    def generate_business_interpretation(self) -> str:
        """
        Generate business interpretation of model results.
        
        Returns:
            Business interpretation as string
        """
        if self.metrics is None or len(self.metrics) == 0:
            self.calculate_metrics()
        
        interpretation = []
        interpretation.append("=" * 80)
        interpretation.append(f"BUSINESS INTERPRETATION - {self.model_name}")
        interpretation.append("=" * 80)
        interpretation.append("")
        
        # Overall performance
        interpretation.append("1. OVERALL PERFORMANCE")
        interpretation.append(f"   The model achieves {self.metrics['accuracy']*100:.2f}% accuracy in predicting")
        interpretation.append(f"   booking cancellations, with an ROC-AUC score of {self.metrics['roc_auc']:.4f}.")
        interpretation.append("")
        
        # Precision interpretation
        interpretation.append("2. PRECISION (Positive Predictive Value)")
        interpretation.append(f"   Precision: {self.metrics['precision']*100:.2f}%")
        interpretation.append(f"   When the model predicts a booking will be canceled,")
        interpretation.append(f"   it is correct {self.metrics['precision']*100:.2f}% of the time.")
        interpretation.append(f"   Business Impact: Helps prioritize which bookings to focus retention")
        interpretation.append(f"   efforts on, minimizing wasted resources on false alarms.")
        interpretation.append("")
        
        # Recall interpretation
        interpretation.append("3. RECALL (Sensitivity)")
        interpretation.append(f"   Recall: {self.metrics['recall']*100:.2f}%")
        interpretation.append(f"   The model correctly identifies {self.metrics['recall']*100:.2f}% of all")
        interpretation.append(f"   bookings that will actually be canceled.")
        interpretation.append(f"   Business Impact: Indicates how many potential cancellations")
        interpretation.append(f"   we can proactively address with interventions.")
        interpretation.append("")
        
        # F1 Score interpretation
        interpretation.append("4. F1-SCORE (Harmonic Mean)")
        interpretation.append(f"   F1-Score: {self.metrics['f1']:.4f}")
        interpretation.append(f"   Balanced measure of precision and recall.")
        if self.metrics['f1'] > 0.75:
            interpretation.append(f"   ✓ Excellent balance between catching cancellations and avoiding false alarms.")
        elif self.metrics['f1'] > 0.65:
            interpretation.append(f"   ✓ Good balance, suitable for production use.")
        else:
            interpretation.append(f"   ⚠ May need further tuning to improve balance.")
        interpretation.append("")
        
        # ROC-AUC interpretation
        interpretation.append("5. ROC-AUC (Discriminative Power)")
        interpretation.append(f"   ROC-AUC: {self.metrics['roc_auc']:.4f}")
        if self.metrics['roc_auc'] > 0.85:
            interpretation.append(f"   ✓ Excellent discriminative ability - model clearly separates")
            interpretation.append(f"     cancellations from non-cancellations.")
        elif self.metrics['roc_auc'] > 0.75:
            interpretation.append(f"   ✓ Good discriminative ability - suitable for deployment.")
        else:
            interpretation.append(f"   ⚠ Moderate discriminative ability - consider feature engineering.")
        interpretation.append("")
        
        # Recommendations
        interpretation.append("6. BUSINESS RECOMMENDATIONS")
        interpretation.append(f"   • Use this model to score all new bookings for cancellation risk")
        interpretation.append(f"   • Implement targeted retention strategies for high-risk bookings:")
        interpretation.append(f"     - Send personalized confirmation emails")
        interpretation.append(f"     - Offer flexible cancellation policies")
        interpretation.append(f"     - Provide special incentives or upgrades")
        interpretation.append(f"   • Monitor model performance monthly and retrain quarterly")
        interpretation.append(f"   • A/B test interventions to measure actual impact on cancellation rates")
        interpretation.append("")
        
        interpretation.append("=" * 80)
        
        return "\n".join(interpretation)
    
    def evaluate_full(self, feature_importance_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Args:
            feature_importance_df: Optional DataFrame with feature importance
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("=" * 80)
        logger.info(f"EVALUATING MODEL: {self.model_name}")
        logger.info("=" * 80)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Generate classification report
        classification_rep = self.generate_classification_report()
        
        # Generate visualizations
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        
        if feature_importance_df is not None:
            self.plot_feature_importance(feature_importance_df)
        
        # Generate business interpretation
        business_interp = self.generate_business_interpretation()
        print(business_interp)
        
        # Save evaluation report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"MODEL EVALUATION REPORT: {self.model_name}")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("METRICS:")
        for metric, value in metrics.items():
            report_lines.append(f"  {metric.upper()}: {value:.4f}")
        report_lines.append("")
        report_lines.append("CLASSIFICATION REPORT:")
        report_lines.append(classification_rep)
        report_lines.append("")
        report_lines.append(business_interp)
        
        report_text = "\n".join(report_lines)
        
        with open(config.MODEL_EVALUATION_FILE, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"\nEvaluation report saved to {config.MODEL_EVALUATION_FILE}")
        
        return {
            'metrics': metrics,
            'classification_report': classification_rep,
            'business_interpretation': business_interp
        }


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                  model_name: str = "Model", feature_importance_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model.
    
    Args:
        model: Trained model to evaluate
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        feature_importance_df: Optional DataFrame with feature importance
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = ModelEvaluator(model, X_test, y_test, model_name)
    return evaluator.evaluate_full(feature_importance_df)


if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import joblib
    
    # Load best model and evaluate
    if config.BEST_MODEL_FILE.exists():
        model = joblib.load(config.BEST_MODEL_FILE)
        # Load test data (would need to be saved during training)
        # evaluate_model(model, X_test, y_test, "Best Model")
        print("Model loaded successfully")
    else:
        print("No saved model found. Train a model first.")
