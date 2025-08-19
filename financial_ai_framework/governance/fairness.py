"""Fairness analysis for financial AI models."""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score
from ..config.settings import RISK_THRESHOLDS, INVESTMENT_GRADE_THRESHOLD, ExperimentConfig


class FairnessAnalyzer:
    """Fairness analysis with comprehensive bias detection."""
    
    def __init__(self, config: ExperimentConfig, tracker):
        self.config = config
        self.tracker = tracker
    
    def comprehensive_fairness_analysis(self, model, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive fairness analysis across multiple dimensions."""
        if not self.config.enable_fairness:
            return {'status': 'disabled'}
        
        print("⚖️ Performing comprehensive fairness analysis...")
        results = {}
        
        # Identify sensitive attributes
        sensitive_features = []
        potential_sensitive = ['sector', 'region', 'risk_category', 'data_source']
        
        for feature in potential_sensitive:
            if feature in X_test.columns:
                # Only include if has reasonable number of groups
                n_groups = X_test[feature].nunique()
                if 2 <= n_groups <= 10:
                    sensitive_features.append(feature)
        
        if not sensitive_features:
            print(" ⚠️ No suitable sensitive features found for fairness analysis")
            return {'status': 'no_sensitive_features'}
        
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        
        for feature in sensitive_features:
            print(f" 📊 Analyzing fairness for: {feature}")
            feature_results = {}
            groups = X_test[feature].unique()
            
            # Calculate metrics by group
            group_metrics = {}
            group_sizes = {}
            
            for group in groups:
                mask = X_test[feature] == group
                group_size = np.sum(mask)
                
                if group_size < 20:  # Skip very small groups
                    continue
                
                y_true_group = y_test[mask]
                y_pred_group = y_pred[mask]
                
                # Basic metrics
                group_acc = accuracy_score(y_true_group, y_pred_group)
                group_f1 = f1_score(y_true_group, y_pred_group, average='macro')
                
                # Investment grade specific
                ig_true_group = (y_true_group <= INVESTMENT_GRADE_THRESHOLD).astype(int)
                ig_pred_group = (y_pred_group <= INVESTMENT_GRADE_THRESHOLD).astype(int)
                ig_acc = accuracy_score(ig_true_group, ig_pred_group)
                
                # Positive rate (investment grade rate)
                positive_rate = np.mean(ig_pred_group)
                true_positive_rate = np.mean(ig_true_group)
                
                group_metrics[str(group)] = {
                    'accuracy': group_acc,
                    'f1_macro': group_f1,
                    'investment_grade_accuracy': ig_acc,
                    'predicted_positive_rate': positive_rate,
                    'true_positive_rate': true_positive_rate,
                    'count': group_size
                }
                group_sizes[str(group)] = group_size
            
            feature_results['group_metrics'] = group_metrics
            feature_results['group_sizes'] = group_sizes
            
            # Calculate fairness metrics
            if len(group_metrics) >= 2:
                accuracies = [metrics['accuracy'] for metrics in group_metrics.values()]
                ig_accuracies = [metrics['investment_grade_accuracy'] for metrics in group_metrics.values()]
                positive_rates = [metrics['predicted_positive_rate'] for metrics in group_metrics.values()]
                
                # Demographic parity (difference in positive rates)
                feature_results['demographic_parity_difference'] = max(positive_rates) - min(positive_rates)
                
                # Equalized odds proxy (difference in accuracies)
                feature_results['accuracy_difference'] = max(accuracies) - min(accuracies)
                feature_results['ig_accuracy_difference'] = max(ig_accuracies) - min(ig_accuracies)
                
                # Statistical parity
                weighted_avg_positive_rate = np.average(
                    positive_rates,
                    weights=[group_sizes[group] for group in group_metrics.keys()]
                )
                feature_results['statistical_parity_difference'] = max(
                    abs(rate - weighted_avg_positive_rate) for rate in positive_rates
                )
                
                # Check for violations
                feature_results['fairness_violations'] = {
                    'demographic_parity': feature_results['demographic_parity_difference'] > RISK_THRESHOLDS['fairness_violation'],
                    'accuracy_disparity': feature_results['accuracy_difference'] > RISK_THRESHOLDS['fairness_violation'],
                    'statistical_parity': feature_results['statistical_parity_difference'] > RISK_THRESHOLDS['fairness_violation']
                }
                
                feature_results['any_violation'] = any(feature_results['fairness_violations'].values())
                
                # Log fairness metrics
                self.tracker.log_metric(f"fairness_{feature}_demographic_parity", feature_results['demographic_parity_difference'])
                self.tracker.log_metric(f"fairness_{feature}_accuracy_diff", feature_results['accuracy_difference'])
            
            results[feature] = feature_results
        
        # Overall fairness summary
        total_violations = sum(
            1 for feature_results in results.values()
            if isinstance(feature_results, dict) and feature_results.get('any_violation', False)
        )
        
        results['summary'] = {
            'features_analyzed': len(sensitive_features),
            'violations_detected': total_violations,
            'violation_rate': total_violations / len(sensitive_features) if sensitive_features else 0
        }
        
        return results