#!/usr/bin/env python3
"""
Financial AI Risk & Governance Framework - Main Execution Script
==============================================================

This script orchestrates the complete financial AI risk assessment pipeline including:
- Data loading and preprocessing
- Multi-model benchmarking
- Risk governance analysis
- Explainability analysis
- Professional reporting

Usage:
    python main.py --data data/credit_ratings.csv --config config.json
    python main.py --data data/credit_ratings.csv --models logreg rf xgb
    python main.py --help

Author: Financial AI Team
Version: 1.0.0
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

# Import framework components
from financial_ai_framework import (
    ExperimentConfig, 
    ensure_reproducibility,
    load_and_validate_data,
    AdvancedDataProcessor,
    ModelBenchmarkSuite,
    UncertaintyAnalyzer,
    FairnessAnalyzer, 
    DriftDetector,
    CalibrationAnalyzer,
    FeatureImportanceAnalyzer,le
    ShapAnalyzer,
    GovernanceReport,
    ExperimentTracker
)

warnings.filterwarnings('ignore')

class FinancialAIOrchestrator:
    """Main orchestrator for the Financial AI Risk & Governance Framework."""
    
    def __init__(self, config: ExperimentConfig, artifacts_dir: Path):
        self.config = config
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tracker = ExperimentTracker(config, artifacts_dir)
        self.data_processor = AdvancedDataProcessor(config)
        self.model_benchmark = ModelBenchmarkSuite(config, self.tracker)
        
        # Governance analyzers
        self.uncertainty_analyzer = UncertaintyAnalyzer(config, self.tracker)
        self.fairness_analyzer = FairnessAnalyzer(config, self.tracker)
        self.drift_detector = DriftDetector(config, self.tracker)
        self.calibration_analyzer = CalibrationAnalyzer(config, self.tracker)
        
        # Explainability analyzers
        self.feature_importance_analyzer = FeatureImportanceAnalyzer(config, self.tracker)
        self.shap_analyzer = ShapAnalyzer(config, self.tracker)
        
        # Reporter
        self.reporter = GovernanceReporter(artifacts_dir, self.tracker)
    
    def run_complete_analysis(self, data_path: str) -> Dict[str, Any]:
        """Run the complete financial AI risk assessment pipeline."""
        
        print("🚀 Starting Financial AI Risk & Governance Analysis")
        print("=" * 60)
        
        # Ensure reproducibility
        ensure_reproducibility()
        
        # Start tracking
        run_id = f"financial_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tracker.start_run(run_id)
        
        try:
            # 1. Data Loading and Processing
            print("\n📊 PHASE 1: Data Loading and Processing")
            print("-" * 40)
            
            df, metadata = load_and_validate_data(data_path)
            df_processed = self.data_processor.clean_and_engineer_features(df)
            
            # Log data metadata
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    self.tracker.log_metric(f"data_{key}", value)
                else:
                    self.tracker.log_param(f"data_{key}", value)
            
            # 2. Data Splitting
            print("\n✂️  PHASE 2: Data Splitting")
            print("-" * 40)
            
            # Prepare features and target
            target_col = 'rating_numeric'
            if target_col not in df_processed.columns:
                raise ValueError(f"Target column '{target_col}' not found after processing")
            
            feature_cols = [col for col in df_processed.columns 
                          if col not in ['credit_rating', 'rating', 'rating_numeric', 'investment_grade']]
            
            X = df_processed[feature_cols].select_dtypes(include=[np.number])
            y = df_processed[target_col]
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=self.config.test_size + self.config.val_size, 
                random_state=self.config.random_state, stratify=y
            )
            
            val_ratio = self.config.val_size / (self.config.test_size + self.config.val_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=1-val_ratio,
                random_state=self.config.random_state, stratify=y_temp
            )
            
            print(f"Training set: {X_train.shape[0]} samples")
            print(f"Validation set: {X_val.shape[0]} samples") 
            print(f"Test set: {X_test.shape[0]} samples")
            print(f"Features: {X_train.shape[1]}")
            
            # 3. Model Benchmarking
            print("\n🎯 PHASE 3: Model Benchmarking")
            print("-" * 40)
            
            benchmark_results = self.model_benchmark.comprehensive_model_training(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            # Get best model
            best_model = self.model_benchmark.best_model
            best_model_name = max(benchmark_results.keys(), 
                                key=lambda k: benchmark_results[k].test_f1_macro)
            
            # 4. Risk Governance Analysis
            print("\n⚖️  PHASE 4: Risk Governance Analysis")
            print("-" * 40)
            
            # Uncertainty Analysis
            uncertainty_results = self.uncertainty_analyzer.comprehensive_uncertainty_analysis(
                best_model, X_test.values, y_test.values
            )
            
            # Fairness Analysis
            fairness_results = self.fairness_analyzer.comprehensive_fairness_analysis(
                best_model, X_test, y_test.values
            )
            
            # Data Drift Detection
            drift_results = self.drift_detector.comprehensive_drift_detection(
                X_train.values, X_test.values, X_train.columns.tolist()
            )
            
            # Calibration Analysis
            calibration_results = self.calibration_analyzer.advanced_calibration_analysis(
                best_model, X_test.values, y_test.values
            )
            
            # 5. Explainability Analysis
            print("\n🔍 PHASE 5: Explainability Analysis")
            print("-" * 40)
            
            # Feature Importance
            feature_importance_results = self.feature_importance_analyzer.comprehensive_feature_importance_analysis(
                self.model_benchmark.models, X_train.columns.tolist()
            )
            
            # SHAP Analysis
            shap_results = self.shap_analyzer.advanced_shap_analysis(
                best_model, X_test.values[:100], X_train.columns.tolist()
            )
            
            # 6. Generate Governance Report
            print("\n📋 PHASE 6: Governance Reporting")
            print("-" * 40)
            
            # Calculate compliance score and risk alerts
            compliance_score, risk_alerts, recommendation = self._calculate_governance_metrics(
                uncertainty_results, fairness_results, drift_results, 
                calibration_results, benchmark_results[best_model_name]
            )
            
            # Create governance report
            from financial_ai_framework.config.settings import GovernanceReport
            governance_report = GovernanceReport(
                model_performance=benchmark_results,
                best_model_name=benchmark_results[best_model_name].model_name,
                uncertainty_analysis=uncertainty_results,
                fairness_metrics=fairness_results,
                drift_detection=drift_results,
                calibration_analysis=calibration_results,
                feature_importance=feature_importance_results,
                shap_analysis=shap_results,
                risk_alerts=risk_alerts,
                compliance_score=compliance_score,
                governance_recommendation=recommendation,
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                data_hash=metadata['data_hash']
            )
            
            # Generate reports
            benchmark_report_path = self.reporter.create_comprehensive_benchmark_report(benchmark_results)
            executive_report_path = self.reporter.create_executive_governance_report(governance_report)
            
            print(f"📊 Benchmark report saved: {benchmark_report_path}")
            print(f"📋 Executive report saved: {executive_report_path}")
            
            # Log final metrics
            self.tracker.log_metric("final_compliance_score", compliance_score)
            self.tracker.log_metric("final_risk_alerts_count", len(risk_alerts))
            
            results = {
                'governance_report': governance_report,
                'benchmark_results': benchmark_results,
                'data_metadata': metadata,
                'artifacts_dir': str(self.artifacts_dir),
                'run_id': run_id
            }
            
            print("\n🎉 Analysis Complete!")
            print(f"📁 Results saved to: {self.artifacts_dir}")
            print(f"🏆 Best Model: {governance_report.best_model_name}")
            print(f"📊 Compliance Score: {compliance_score:.1%}")
            print(f"🚨 Risk Alerts: {len(risk_alerts)}")
            
            return results
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise
        finally:
            self.tracker.end_run()
    
    def _calculate_governance_metrics(self, uncertainty_results, fairness_results, 
                                    drift_results, calibration_results, best_model_result):
        """Calculate overall governance compliance score and risk alerts."""
        
        risk_alerts = []
        score_components = []
        
        # Model performance score (40% weight)
        perf_score = min(best_model_result.test_f1_macro / 0.8, 1.0)  # Normalize to 0.8 as excellent
        score_components.append(('performance', perf_score, 0.4))
        
        # Uncertainty score (20% weight)
        if uncertainty_results.get('status') != 'disabled':
            uncertainty_ratio = uncertainty_results.get('high_uncertainty_ratio', 0)
            if uncertainty_ratio > 0.3:
                risk_alerts.append("High model uncertainty detected")
            uncertainty_score = max(0, 1 - uncertainty_ratio * 2)  # Penalize high uncertainty
            score_components.append(('uncertainty', uncertainty_score, 0.2))
        
        # Fairness score (20% weight)
        if fairness_results.get('status') not in ['disabled', 'no_sensitive_features']:
            violation_rate = fairness_results.get('summary', {}).get('violation_rate', 0)
            if violation_rate > 0:
                risk_alerts.append(f"Fairness violations detected ({violation_rate:.1%})")
            fairness_score = max(0, 1 - violation_rate * 2)
            score_components.append(('fairness', fairness_score, 0.2))
        
        # Drift score (10% weight)  
        if drift_results.get('status') != 'disabled':
            drift_ratio = drift_results.get('summary', {}).get('drift_detected_ratio', 0)
            if drift_ratio > 0.2:
                risk_alerts.append(f"Data drift detected ({drift_ratio:.1%} of features)")
            drift_score = max(0, 1 - drift_ratio)
            score_components.append(('drift', drift_score, 0.1))
        
        # Calibration score (10% weight)
        if calibration_results.get('status') not in ['disabled', 'no_probabilities']:
            ece = calibration_results.get('ece', 0)
            if ece > 0.15:
                risk_alerts.append("Poor model calibration detected")
            calibration_score = max(0, 1 - ece * 5)  # Penalize high ECE
            score_components.append(('calibration', calibration_score, 0.1))
        
        # Calculate weighted compliance score
        total_weight = sum(weight for _, _, weight in score_components)
        if total_weight > 0:
            compliance_score = sum(score * weight for _, score, weight in score_components) / total_weight
        else:
            compliance_score = 0.5  # Default if no components available
        
        # Generate recommendation
        if compliance_score >= 0.9:
            recommendation = "APPROVE FOR PRODUCTION - All governance requirements met"
        elif compliance_score >= 0.8:
            recommendation = "CONDITIONAL APPROVAL - Address identified issues before production"
        elif compliance_score >= 0.7:
            recommendation = "REMEDIATION REQUIRED - Significant improvements needed"
        else:
            recommendation = "REJECT - Multiple critical issues must be resolved"
        
        return compliance_score, risk_alerts, recommendation

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Financial AI Risk & Governance Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --data data/credit_ratings.csv
    python main.py --data data/credit_ratings.csv --models logreg rf xgb
    python main.py --data data/credit_ratings.csv --output results/
    python main.py --data data/credit_ratings.csv --config config.json
        """
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to the credit ratings dataset')
    
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory for results (default: outputs)')
    
    parser.add_argument('--config', type=str, 
                       help='Path to configuration JSON file')
    
    parser.add_argument('--models', nargs='+', 
                       choices=['logreg', 'rf', 'xgb', 'lgbm'],
                       help='Models to train (default: logreg rf)')
    
    parser.add_argument('--no-ensemble', action='store_true',
                       help='Disable ensemble model creation')
    
    parser.add_argument('--no-shap', action='store_true',
                       help='Disable SHAP analysis')
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode - disable time-intensive analyses')
    
    return parser.parse_args()

def load_config(config_path: Optional[str] = None, **kwargs) -> ExperimentConfig:
    """Load configuration from file or command line arguments."""
    
    # Start with default config
    config_dict = {}
    
    # Load from file if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        print(f"📝 Configuration loaded from: {config_path}")
    
    # Override with command line arguments
    if 'models' in kwargs and kwargs['models']:
        config_dict['models_to_train'] = kwargs['models']
    
    if 'no_ensemble' in kwargs and kwargs['no_ensemble']:
        config_dict['enable_ensemble'] = False
    
    if 'no_shap' in kwargs and kwargs['no_shap']:
        config_dict['enable_shap'] = False
    
    if 'quick' in kwargs and kwargs['quick']:
        config_dict.update({
            'enable_shap': False,
            'enable_drift_detection': False,
            'shap_sample_size': 50,
            'cv_folds': 3
        })
    
    return ExperimentConfig(**config_dict)

def main():
    """Main execution function."""
    args = parse_args()
    
    # Validate data file exists
    if not Path(args.data).exists():
        print(f"❌ Data file not found: {args.data}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(
        args.config,
        models=args.models,
        no_ensemble=args.no_ensemble,
        no_shap=args.no_shap,
        quick=args.quick
    )
    
    # Set up output directory
    artifacts_dir = Path(args.output) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize orchestrator
    orchestrator = FinancialAIOrchestrator(config, artifacts_dir)
    
    # Run analysis
    try:
        results = orchestrator.run_complete_analysis(args.data)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results available in: {artifacts_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())