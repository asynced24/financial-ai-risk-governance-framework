"""Data processing and feature engineering for financial AI models."""

import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any
from ..config.settings import SP_RATING_MAPPING, INVESTMENT_GRADE_THRESHOLD, ExperimentConfig


def calculate_data_hash(df: pd.DataFrame) -> str:
    """Calculate hash of dataset for reproducibility tracking."""
    return hashlib.md5(df.to_string().encode()).hexdigest()[:16]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for consistency."""
    df = df.copy()
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(r'[^a-z0-9]+', '_', regex=True)
                  .str.strip('_'))
    return df


def load_and_validate_data(data_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load dataset with comprehensive validation and metadata extraction."""
    print(f"📊 Loading dataset: {data_path}")
    
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    # Load based on file extension
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    elif path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    df = normalize_column_names(df)
    
    # Calculate metadata
    metadata = {
        'file_path': str(path),
        'file_size_mb': path.stat().st_size / 1024**2,
        'load_timestamp': datetime.now().isoformat(),
        'original_shape': df.shape,
        'column_names': df.columns.tolist(),
        'data_hash': calculate_data_hash(df)
    }
    
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"📝 Data hash: {metadata['data_hash']}")
    
    # Basic validation
    if 'credit_rating' not in df.columns:
        raise ValueError("Target column 'credit_rating' not found")
    
    return df, metadata


class AdvancedDataProcessor:
    """Advanced data preprocessing with comprehensive feature engineering."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.preprocessor = None
        self.feature_metadata = {}
    
    def clean_and_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data cleaning and feature engineering."""
        print("🔧 Starting advanced data cleaning and feature engineering...")
        
        df = df.copy()
        initial_rows = len(df)
        
        # Clean target variable
        df['rating'] = df['credit_rating'].astype(str).str.upper().str.strip()
        df['rating'] = df['rating'].str.replace(r'[*+\-\s]+$', '', regex=True)
        
        # Remove invalid ratings
        valid_ratings = set(SP_RATING_MAPPING.keys())
        df = df[df['rating'].isin(valid_ratings)]
        df = df[~df['rating'].isin(['NR', 'WR', 'nan'])]
        
        print(f" ✅ Removed {initial_rows - len(df)} invalid/missing ratings")
        
        # Convert numeric columns with comprehensive error handling
        numeric_cols = [
            'retained_earnings', 'market_price', 'revenue', 'ebit',
            'current_assets', 'current_liabilities', 'total_assets',
            'total_liabilities', 'working_capital', 'market_value_equity',
            'average_pd'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill extreme outliers
                if df[col].dtype in ['int64', 'float64']:
                    q99 = df[col].quantile(0.99)
                    q1 = df[col].quantile(0.01)
                    df[col] = df[col].clip(lower=q1, upper=q99)
        
        # Advanced financial ratio engineering
        df = self._create_comprehensive_financial_ratios(df)
        
        # Risk categorization and business features
        df['investment_grade'] = df['rating'].map(SP_RATING_MAPPING) <= INVESTMENT_GRADE_THRESHOLD
        df['rating_numeric'] = df['rating'].map(SP_RATING_MAPPING)
        
        # Enhanced risk categories
        df['risk_category'] = pd.cut(
            df['rating_numeric'],
            bins=[-1, 3, 6, 9, 15, 21],
            labels=['Prime', 'High_Grade', 'Medium_Grade', 'Speculative', 'Distressed']
        )
        
        # Market volatility proxy
        if 'market_price' in df.columns and 'sector' in df.columns:
            df['market_price_volatility'] = df.groupby('sector')['market_price'].transform(lambda x: x.std())
        
        # Sector risk analysis
        if 'sector' in df.columns:
            sector_default_rates = df.groupby('sector')['investment_grade'].transform('mean')
            df['sector_risk_score'] = 1 - sector_default_rates
        
        print(f"✅ Advanced feature engineering complete: {len(df)} samples, {len(df.columns)} features")
        
        # Store feature metadata
        self._calculate_feature_metadata(df)
        
        return df
    
    def _create_comprehensive_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive financial ratios for credit analysis."""
        
        # Liquidity ratios
        if all(col in df.columns for col in ['current_assets', 'current_liabilities']):
            df['current_ratio'] = df['current_assets'] / (df['current_liabilities'] + 1e-8)
            df['quick_ratio'] = df['current_assets'] / (df['current_liabilities'] + 1e-8)
            df['cash_ratio'] = (df['current_assets'] * 0.1) / (df['current_liabilities'] + 1e-8)
        
        # Leverage ratios
        if all(col in df.columns for col in ['total_liabilities', 'total_assets']):
            df['debt_to_assets'] = df['total_liabilities'] / (df['total_assets'] + 1e-8)
            df['equity_ratio'] = 1 - df['debt_to_assets']
            df['debt_to_equity'] = df['total_liabilities'] / ((df['total_assets'] - df['total_liabilities']) + 1e-8)
        
        # Profitability ratios
        if all(col in df.columns for col in ['ebit', 'revenue']):
            df['ebit_margin'] = df['ebit'] / (df['revenue'] + 1e-8)
            df['gross_margin'] = (df['revenue'] - df['ebit'] * 0.6) / (df['revenue'] + 1e-8)
        
        if all(col in df.columns for col in ['ebit', 'total_assets']):
            df['roa'] = df['ebit'] / (df['total_assets'] + 1e-8)  # Return on Assets
        
        if all(col in df.columns for col in ['ebit', 'market_value_equity']):
            df['roe'] = df['ebit'] / (df['market_value_equity'] + 1e-8)  # Return on Equity
        
        # Efficiency ratios
        if all(col in df.columns for col in ['revenue', 'total_assets']):
            df['asset_turnover'] = df['revenue'] / (df['total_assets'] + 1e-8)
        
        if all(col in df.columns for col in ['working_capital', 'total_assets']):
            df['working_capital_ratio'] = df['working_capital'] / (df['total_assets'] + 1e-8)
            df['working_capital_turnover'] = df['revenue'] / (df['working_capital'] + 1e-8)
        
        # Coverage ratios
        if all(col in df.columns for col in ['ebit', 'total_liabilities']):
            df['interest_coverage'] = df['ebit'] / ((df['total_liabilities'] * 0.05) + 1e-8)
        
        # Market ratios
        if all(col in df.columns for col in ['market_value_equity', 'revenue']):
            df['market_to_sales'] = df['market_value_equity'] / (df['revenue'] + 1e-8)
        
        if all(col in df.columns for col in ['market_value_equity', 'total_assets']):
            df['market_to_book'] = df['market_value_equity'] / ((df['total_assets'] - df['total_liabilities']) + 1e-8)
        
        # Size factors (log-transform for normality)
        for col in ['total_assets', 'revenue', 'market_value_equity']:
            if col in df.columns:
                df[f'log_{col}'] = np.log1p(np.abs(df[col]))
        
        # Stability ratios
        if 'retained_earnings' in df.columns and 'total_assets' in df.columns:
            df['retained_earnings_ratio'] = df['retained_earnings'] / (df['total_assets'] + 1e-8)
        
        return df
    
    def _calculate_feature_metadata(self, df: pd.DataFrame):
        """Calculate comprehensive feature metadata."""
        self.feature_metadata = {
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(exclude=[np.number]).columns),
            'engineered_features': sum(1 for col in df.columns if any(
                prefix in col for prefix in ['ratio', 'margin', 'turnover', 'coverage', 'log_']
            )),
            'missing_values': df.isnull().sum().to_dict(),
            'feature_correlations': df.select_dtypes(include=[np.number]).corr().abs().mean().to_dict()
        }