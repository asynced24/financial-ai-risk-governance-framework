"""Uncertainty analysis for financial AI models."""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score
from ..config.settings import RISK_THRESHOLDS, ExperimentConfig


class UncertaintyAnalyzer:
    """Uncertainty analysis with comprehensive checks."""
    
    def __init__(self, config: ExperimentConfig, tracker):
        self.config = config
        self.tracker = tracker
    
    def comprehensive_uncertainty_analysis(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive uncertainty quantification analysis."""
        if not self.config.enable_uncertainty:
            return {'status': 'disabled'}
        
        print("🎲 Performing comprehensive uncertainty analysis...")
        results = {}
        
        if hasattr(model, 'predict_proba'):
            # Get prediction probabilities
            y_proba = model.predict_proba(X_test)
            
            # Confidence metrics
            confidence = np.max(y_proba, axis=1)
            results['confidence_mean'] = np.mean(confidence)
            results['confidence_std'] = np.std(confidence)
            results['confidence_min'] = np.min(confidence)
            results['confidence_max'] = np.max(confidence)
            
            # Entropy-based uncertainty
            entropy = -np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)
            results['entropy_mean'] = np.mean(entropy)
            results['entropy_std'] = np.std(entropy)
            
            # Uncertainty distribution analysis
            high_uncertainty_mask = entropy > RISK_THRESHOLDS['uncertainty_high']
            results['high_uncertainty_ratio'] = np.mean(high_uncertainty_mask)
            results['high_uncertainty_count'] = np.sum(high_uncertainty_mask)
            
            # Confidence vs accuracy analysis
            y_pred = model.predict(X_test)
            correct_predictions = (y_pred == y_test)
            results['avg_confidence_correct'] = np.mean(confidence[correct_predictions])
            results['avg_confidence_incorrect'] = np.mean(confidence[~correct_predictions])
            results['confidence_discrimination'] = results['avg_confidence_correct'] - results['avg_confidence_incorrect']
            
            # Bootstrap uncertainty estimation (enhanced)
            bootstrap_results = []
            bootstrap_uncertainties = []
            
            for i in range(100):  # Increased samples
                indices = np.random.choice(len(X_test), size=min(200, len(X_test)), replace=True)
                X_boot = X_test[indices]
                y_boot = y_test[indices]
                
                try:
                    y_pred_boot = model.predict(X_boot)
                    y_proba_boot = model.predict_proba(X_boot)
                    acc_boot = accuracy_score(y_boot, y_pred_boot)
                    f1_boot = f1_score(y_boot, y_pred_boot, average='macro')
                    
                    # Calculate uncertainty for this bootstrap
                    entropy_boot = -np.sum(y_proba_boot * np.log(y_proba_boot + 1e-10), axis=1)
                    bootstrap_results.append({'accuracy': acc_boot, 'f1_macro': f1_boot})
                    bootstrap_uncertainties.append(np.mean(entropy_boot))
                except:
                    continue
            
            if bootstrap_results:
                accuracies = [r['accuracy'] for r in bootstrap_results]
                f1_scores = [r['f1_macro'] for r in bootstrap_results]
                
                results['bootstrap_accuracy_mean'] = np.mean(accuracies)
                results['bootstrap_accuracy_std'] = np.std(accuracies)
                results['bootstrap_accuracy_ci_lower'] = np.percentile(accuracies, 2.5)
                results['bootstrap_accuracy_ci_upper'] = np.percentile(accuracies, 97.5)
                results['bootstrap_f1_mean'] = np.mean(f1_scores)
                results['bootstrap_f1_std'] = np.std(f1_scores)
                results['bootstrap_uncertainty_mean'] = np.mean(bootstrap_uncertainties)
                results['bootstrap_uncertainty_std'] = np.std(bootstrap_uncertainties)
            
            # Log uncertainty metrics
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    self.tracker.log_metric(f"uncertainty_{key}", value)
        
        return results