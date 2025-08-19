"""Model calibration analysis for financial AI models."""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import brier_score_loss
from ..config.settings import RISK_THRESHOLDS, ExperimentConfig


class CalibrationAnalyzer:
    """Model calibration analysis with reliability assessment."""
    
    def __init__(self, config: ExperimentConfig, tracker):
        self.config = config
        self.tracker = tracker
    
    def advanced_calibration_analysis(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Advanced model calibration analysis."""
        if not self.config.enable_calibration:
            return {'status': 'disabled'}
        
        print("🎯 Performing advanced calibration analysis...")
        results = {}
        
        if not hasattr(model, 'predict_proba'):
            return {'status': 'no_probabilities'}
        
        y_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        
        # Multi-class calibration analysis
        confidence = np.max(y_proba, axis=1)
        accuracy = (y_pred == y_test).astype(int)
        
        # Reliability analysis with adaptive binning
        n_bins = min(15, len(X_test) // 50)  # Adaptive number of bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        bin_sizes = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracy[in_bin].mean()
                avg_confidence_in_bin = confidence[in_bin].mean()
                count_in_bin = np.sum(in_bin)
            else:
                accuracy_in_bin = 0
                avg_confidence_in_bin = (bin_lower + bin_upper) / 2
                count_in_bin = 0
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(prop_in_bin)
            bin_sizes.append(count_in_bin)
        
        # Expected Calibration Error (ECE)
        ece = sum(bin_counts[i] * abs(bin_accuracies[i] - bin_confidences[i]) for i in range(n_bins))
        
        # Maximum Calibration Error (MCE)
        mce = max(abs(bin_accuracies[i] - bin_confidences[i]) for i in range(n_bins))
        
        # Overconfidence Error (OE) and Underconfidence Error (UE)
        oe = sum(bin_counts[i] * max(bin_confidences[i] - bin_accuracies[i], 0) for i in range(n_bins))
        ue = sum(bin_counts[i] * max(bin_accuracies[i] - bin_confidences[i], 0) for i in range(n_bins))
        
        # Brier Score decomposition
        brier_scores = []
        for i in range(y_proba.shape[1]):
            y_binary = (y_test == i).astype(int)
            if len(np.unique(y_binary)) > 1:
                brier = brier_score_loss(y_binary, y_proba[:, i])
                brier_scores.append(brier)
        
        avg_brier_score = np.mean(brier_scores) if brier_scores else 0
        
        # Confidence distribution analysis
        results = {
            'ece': ece,
            'mce': mce,
            'overconfidence_error': oe,
            'underconfidence_error': ue,
            'avg_brier_score': avg_brier_score,
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'bin_sizes': bin_sizes,
            'confidence_mean': np.mean(confidence),
            'confidence_std': np.std(confidence),
            'well_calibrated': ece < RISK_THRESHOLDS['calibration_error'],
            'is_overconfident': oe > ue,
            'calibration_quality': 'excellent' if ece < 0.05 else 'good' if ece < 0.1 else 'poor'
        }
        
        # Log calibration metrics
        self.tracker.log_metric("calibration_ece", ece)
        self.tracker.log_metric("calibration_mce", mce)
        self.tracker.log_metric("calibration_brier_score", avg_brier_score)
        
        return results