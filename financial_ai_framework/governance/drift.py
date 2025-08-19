"""Data drift detection for financial AI models."""

import numpy as np
from typing import Dict, Any, List
from scipy.stats import ks_2samp
from scipy import stats
from ..config.settings import RISK_THRESHOLDS, ExperimentConfig


class DriftDetector:
    """Data drift detection with comprehensive statistical tests."""
    
    def __init__(self, config: ExperimentConfig, tracker):
        self.config = config
        self.tracker = tracker
    
    def comprehensive_drift_detection(self, X_train: np.ndarray, X_test: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Comprehensive data drift detection analysis."""
        if not self.config.enable_drift_detection:
            return {'status': 'disabled'}
        
        print("🌊 Performing comprehensive data drift analysis...")
        results = {}
        drift_detected_features = []
        
        # Limit features for performance but be more comprehensive
        max_features = min(50, X_train.shape[1])
        feature_indices = np.random.choice(X_train.shape[1], size=max_features, replace=False)
        
        for idx in feature_indices:
            if idx >= len(feature_names):
                continue
            
            feature_name = feature_names[idx]
            train_feature = X_train[:, idx]
            test_feature = X_test[:, idx]
            
            # Remove NaN values
            train_clean = train_feature[~np.isnan(train_feature)]
            test_clean = test_feature[~np.isnan(test_feature)]
            
            if len(train_clean) == 0 or len(test_clean) == 0:
                continue
            
            feature_result = {}
            
            # Kolmogorov-Smirnov test
            try:
                ks_stat, ks_pvalue = ks_2samp(train_clean, test_clean)
                feature_result['ks_statistic'] = ks_stat
                feature_result['ks_pvalue'] = ks_pvalue
                feature_result['ks_drift'] = ks_pvalue < RISK_THRESHOLDS['drift_pvalue']
            except:
                feature_result['ks_statistic'] = 0.0
                feature_result['ks_pvalue'] = 1.0
                feature_result['ks_drift'] = False
            
            # Population Stability Index (PSI)
            psi = self._calculate_psi(train_clean, test_clean)
            feature_result['psi'] = psi
            feature_result['psi_drift'] = psi > 0.2  # Industry standard threshold
            
            # Wasserstein distance (Earth Mover's Distance)
            try:
                wasserstein_dist = stats.wasserstein_distance(train_clean, test_clean)
                feature_result['wasserstein_distance'] = wasserstein_dist
            except:
                feature_result['wasserstein_distance'] = 0.0
            
            # Mean and variance shift
            train_mean, train_std = np.mean(train_clean), np.std(train_clean)
            test_mean, test_std = np.mean(test_clean), np.std(test_clean)
            
            feature_result['mean_shift'] = abs(test_mean - train_mean) / (train_std + 1e-10)
            feature_result['variance_shift'] = abs(test_std - train_std) / (train_std + 1e-10)
            
            # Overall drift decision
            drift_detected = (
                feature_result['ks_drift'] or
                feature_result['psi_drift'] or
                feature_result['mean_shift'] > 2 or  # 2 standard deviations
                feature_result['variance_shift'] > 0.5  # 50% variance change
            )
            
            feature_result['drift_detected'] = drift_detected
            
            if drift_detected:
                drift_detected_features.append(feature_name)
            
            results[feature_name] = feature_result
        
        # Summary statistics
        results['summary'] = {
            'total_features_tested': len(results) - 1 if 'summary' in results else len(results),
            'drift_detected_count': len(drift_detected_features),
            'drift_detected_features': drift_detected_features,
            'drift_detected_ratio': len(drift_detected_features) / max(1, len(results) - (1 if 'summary' in results else 0)),
            'avg_psi': np.mean([r['psi'] for r in results.values() if isinstance(r, dict) and 'psi' in r]),
            'avg_ks_pvalue': np.mean([r['ks_pvalue'] for r in results.values() if isinstance(r, dict) and 'ks_pvalue' in r])
        }
        
        # Log drift metrics
        self.tracker.log_metric("drift_detected_ratio", results['summary']['drift_detected_ratio'])
        self.tracker.log_metric("drift_avg_psi", results['summary']['avg_psi'])
        
        return results
    
    def _calculate_psi(self, train_data: np.ndarray, test_data: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index with enhanced binning."""
        try:
            # Use quantile-based binning for better distribution
            bin_edges = np.percentile(train_data, np.linspace(0, 100, bins + 1))
            bin_edges[0] = -np.inf  # Include all values
            bin_edges[-1] = np.inf
            
            # Calculate distributions
            train_dist = np.histogram(train_data, bins=bin_edges)[0] / len(train_data)
            test_dist = np.histogram(test_data, bins=bin_edges)[0] / len(test_data)
            
            # Avoid division by zero with small constant
            epsilon = 1e-10
            train_dist = np.maximum(train_dist, epsilon)
            test_dist = np.maximum(test_dist, epsilon)
            
            # Calculate PSI
            psi = np.sum((test_dist - train_dist) * np.log(test_dist / train_dist))
            return psi
        except:
            return 0.0