"""Feature importance analysis for financial AI models."""

import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from ..config.settings import ExperimentConfig


class FeatureImportanceAnalyzer:
    """Feature importance analysis across multiple models."""
    
    def __init__(self, config: ExperimentConfig, tracker):
        self.config = config
        self.tracker = tracker
    
    def comprehensive_feature_importance_analysis(self, models: Dict, feature_names: List[str]) -> Dict[str, Any]:
        """Comprehensive feature importance analysis across multiple methods."""
        if not self.config.enable_feature_importance:
            return {'status': 'disabled'}
        
        print("🔍 Performing comprehensive feature importance analysis...")
        results = {}
        
        # Analyze each model's feature importance
        model_importances = {}
        for model_name, model in models.items():
            model_importance = {}
            
            # Native feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                model_importance['native'] = dict(zip(feature_names, importances))
            elif hasattr(model, 'coef_'):
                # For linear models, use coefficient magnitudes
                if len(model.coef_.shape) > 1:
                    importances = np.mean(np.abs(model.coef_), axis=0)
                else:
                    importances = np.abs(model.coef_)
                model_importance['native'] = dict(zip(feature_names, importances))
            
            model_importances[model_name] = model_importance
        
        # Aggregate importance across models
        if model_importances:
            aggregated_importance = defaultdict(list)
            
            for model_name, importance_dict in model_importances.items():
                if 'native' in importance_dict:
                    for feature, importance in importance_dict['native'].items():
                        aggregated_importance[feature].append(importance)
            
            # Calculate statistics across models
            feature_importance_stats = {}
            for feature, importances in aggregated_importance.items():
                feature_importance_stats[feature] = {
                    'mean': np.mean(importances),
                    'std': np.std(importances),
                    'min': np.min(importances),
                    'max': np.max(importances),
                    'models_count': len(importances)
                }
            
            # Sort by mean importance
            sorted_features = sorted(
                feature_importance_stats.items(),
                key=lambda x: x[1]['mean'],
                reverse=True
            )
            
            results['model_importances'] = model_importances
            results['aggregated_importance'] = dict(sorted_features)
            results['top_10_features'] = dict(sorted_features[:10])
            results['top_20_features'] = dict(sorted_features[:20])
            
            # Feature importance categories
            results['financial_ratios'] = {
                k: v for k, v in dict(sorted_features).items()
                if any(term in k.lower() for term in ['ratio', 'margin', 'turnover', 'coverage'])
            }
            
            results['size_metrics'] = {
                k: v for k, v in dict(sorted_features).items()
                if any(term in k.lower() for term in ['log_', 'total_assets', 'revenue', 'market_value'])
            }
        
        return results