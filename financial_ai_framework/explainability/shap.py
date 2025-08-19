"""SHAP analysis for financial AI models."""

import numpy as np
from typing import Dict, Any, List
from ..config.settings import ExperimentConfig, HAS_SHAP

if HAS_SHAP:
    import shap


class ShapAnalyzer:
    """SHAP analysis for model interpretability."""
    
    def __init__(self, config: ExperimentConfig, tracker):
        self.config = config
        self.tracker = tracker
    
    def advanced_shap_analysis(self, model, X_sample: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Advanced SHAP analysis with comprehensive interpretability."""
        if not self.config.enable_shap or not HAS_SHAP:
            return {'status': 'shap_not_available'}
        
        print("🎯 Performing advanced SHAP analysis...")
        
        try:
            # Limit sample size for performance
            sample_size = min(self.config.shap_sample_size, len(X_sample))
            X_shap = X_sample[:sample_size]
            
            # Choose appropriate explainer based on model type
            if hasattr(model, 'predict_proba'):
                # Use model prediction function
                explainer = shap.Explainer(model.predict_proba, X_shap[:min(50, sample_size)])
            else:
                explainer = shap.Explainer(model.predict, X_shap[:min(50, sample_size)])
            
            # Calculate SHAP values
            shap_values = explainer(X_shap)
            
            results = {}
            
            # Handle different SHAP value formats
            if hasattr(shap_values, 'values'):
                if isinstance(shap_values.values, list):
                    # Multi-class case - aggregate across classes
                    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values.values], axis=0)
                else:
                    # Single output or multi-class with single array
                    if len(shap_values.values.shape) == 3:
                        # Multi-class: (samples, features, classes)
                        mean_shap = np.abs(shap_values.values).mean(axis=(0, 2))
                    else:
                        # Binary or regression: (samples, features)
                        mean_shap = np.abs(shap_values.values).mean(axis=0)
            else:
                # Fallback for older SHAP versions
                if isinstance(shap_values, list):
                    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                else:
                    mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Create feature importance from SHAP
            if len(mean_shap) == len(feature_names):
                shap_importance = dict(zip(feature_names, mean_shap))
                sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
                
                results['shap_feature_importance'] = dict(sorted_shap)
                results['top_10_shap_features'] = dict(sorted_shap[:10])
                results['top_20_shap_features'] = dict(sorted_shap[:20])
                
                # SHAP-based feature categories
                results['shap_financial_ratios'] = {
                    k: v for k, v in dict(sorted_shap).items()
                    if any(term in k.lower() for term in ['ratio', 'margin', 'turnover', 'coverage'])
                }
                
                # Calculate feature interaction strength (simplified)
                if hasattr(shap_values, 'values') and len(shap_values.values.shape) >= 2:
                    feature_interactions = {}
                    top_features = [item[0] for item in sorted_shap[:10]]
                    
                    for i, feat1 in enumerate(top_features[:5]):  # Limit for performance
                        for feat2 in top_features[i+1:6]:  # Top 5 interactions
                            if feat1 in feature_names and feat2 in feature_names:
                                idx1 = feature_names.index(feat1)
                                idx2 = feature_names.index(feat2)
                                if idx1 < mean_shap.shape[0] and idx2 < mean_shap.shape[0]:
                                    interaction_strength = abs(mean_shap[idx1] * mean_shap[idx2])
                                    feature_interactions[f"{feat1} × {feat2}"] = interaction_strength
                    
                    results['feature_interactions'] = dict(
                        sorted(feature_interactions.items(), key=lambda x: x[1], reverse=True)[:10]
                    )
                
                # Log top SHAP features
                for i, (feature, importance) in enumerate(sorted_shap[:10]):
                    self.tracker.log_metric(f"shap_feature_{i+1}_{feature[:20]}", importance)
                
                results['status'] = 'success'
                results['sample_size'] = sample_size
            else:
                results['status'] = 'dimension_mismatch'
                results['shap_shape'] = mean_shap.shape
                results['feature_names_length'] = len(feature_names)
            
            return results
        
        except Exception as e:
            print(f" ⚠️ SHAP analysis failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}