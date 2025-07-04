#!/usr/bin/env python3
"""
Data Utilities for Medical Digital Twins

This module provides utility functions for processing and handling
multiomics data in medical digital twin applications.

Author: Digital Twins Research Team
License: MIT
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

def load_multiomics_data(file_paths: Dict[str, str], 
                        file_format: str = 'csv') -> Dict[str, pd.DataFrame]:
    """
    Load multiomics data from various file formats.
    
    Args:
        file_paths: Dictionary mapping omics type to file path
        file_format: File format ('csv', 'tsv', 'excel', 'h5')
        
    Returns:
        Dictionary of loaded dataframes
    """
    data = {}
    
    for omics_type, path in file_paths.items():
        try:
            if file_format == 'csv':
                df = pd.read_csv(path, index_col=0)
            elif file_format == 'tsv':
                df = pd.read_csv(path, sep='\t', index_col=0)
            elif file_format == 'excel':
                df = pd.read_excel(path, index_col=0)
            elif file_format == 'h5':
                df = pd.read_hdf(path, key='data')
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            data[omics_type] = df
            print(f"Loaded {omics_type}: {df.shape}")
            
        except Exception as e:
            print(f"Error loading {omics_type} from {path}: {e}")
            
    return data

def validate_data_consistency(data_dict: Dict[str, pd.DataFrame]) -> bool:
    """
    Validate consistency across multiomics datasets.
    
    Args:
        data_dict: Dictionary of omics dataframes
        
    Returns:
        True if data is consistent, False otherwise
    """
    if not data_dict:
        return False
    
    # Check sample consistency
    sample_sets = [set(df.index) for df in data_dict.values()]
    common_samples = set.intersection(*sample_sets)
    
    if len(common_samples) == 0:
        print("Error: No common samples across datasets")
        return False
    
    # Check for missing samples
    for omics_type, df in data_dict.items():
        missing_samples = set(df.index) - common_samples
        if missing_samples:
            print(f"Warning: {omics_type} has {len(missing_samples)} unique samples")
    
    print(f"Common samples across all datasets: {len(common_samples)}")
    return True

def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'median',
                         omics_type: str = 'unknown') -> pd.DataFrame:
    """
    Handle missing values in omics data.
    
    Args:
        df: Input dataframe
        strategy: Imputation strategy ('mean', 'median', 'mode', 'knn', 'zero')
        omics_type: Type of omics data for context-specific handling
        
    Returns:
        Dataframe with imputed values
    """
    if df.isnull().sum().sum() == 0:
        return df
    
    print(f"Handling missing values in {omics_type} data...")
    print(f"Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / df.size * 100:.2f}%)")
    
    if strategy in ['mean', 'median', 'most_frequent']:
        imputer = SimpleImputer(strategy=strategy)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df),
            index=df.index,
            columns=df.columns
        )
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df),
            index=df.index,
            columns=df.columns
        )
    elif strategy == 'zero':
        df_imputed = df.fillna(0)
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")
    
    return df_imputed

def normalize_omics_data(df: pd.DataFrame, 
                        method: str = 'standard',
                        omics_type: str = 'unknown') -> Tuple[pd.DataFrame, object]:
    """
    Normalize omics data using appropriate method.
    
    Args:
        df: Input dataframe
        method: Normalization method ('standard', 'minmax', 'robust', 'log', 'quantile')
        omics_type: Type of omics data
        
    Returns:
        Tuple of (normalized dataframe, fitted scaler)
    """
    print(f"Normalizing {omics_type} data using {method} method...")
    
    # Apply log transformation for expression data
    if omics_type in ['transcriptomics', 'proteomics'] and method != 'log':
        df = np.log1p(df.clip(lower=0))
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'log':
        # Log transformation only
        df_normalized = np.log1p(df.clip(lower=0))
        return df_normalized, None
    elif method == 'quantile':
        from sklearn.preprocessing import QuantileTransformer
        scaler = QuantileTransformer(output_distribution='normal')
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    
    return df_normalized, scaler

def filter_low_variance_features(df: pd.DataFrame, 
                                threshold: float = 0.01) -> pd.DataFrame:
    """
    Remove features with low variance.
    
    Args:
        df: Input dataframe
        threshold: Variance threshold
        
    Returns:
        Filtered dataframe
    """
    variances = df.var()
    high_var_features = variances[variances > threshold].index
    
    print(f"Removed {len(df.columns) - len(high_var_features)} low-variance features")
    
    return df[high_var_features]

def filter_highly_correlated_features(df: pd.DataFrame, 
                                     threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove highly correlated features.
    
    Args:
        df: Input dataframe
        threshold: Correlation threshold
        
    Returns:
        Filtered dataframe
    """
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > threshold)]
    
    print(f"Removed {len(to_drop)} highly correlated features")
    
    return df.drop(columns=to_drop)

def split_multiomics_data(data_dict: Dict[str, pd.DataFrame],
                         test_size: float = 0.2,
                         val_size: float = 0.1,
                         random_state: int = 42) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Split multiomics data into train/validation/test sets.
    
    Args:
        data_dict: Dictionary of omics dataframes
        test_size: Proportion of test data
        val_size: Proportion of validation data
        random_state: Random seed
        
    Returns:
        Dictionary with train/val/test splits for each omics type
    """
    from sklearn.model_selection import train_test_split
    
    # Get common samples
    common_samples = set.intersection(*[set(df.index) for df in data_dict.values()])
    common_samples = list(common_samples)
    
    # Split sample indices
    train_samples, test_samples = train_test_split(
        common_samples, test_size=test_size, random_state=random_state
    )
    
    if val_size > 0:
        train_samples, val_samples = train_test_split(
            train_samples, test_size=val_size/(1-test_size), random_state=random_state
        )
    else:
        val_samples = []
    
    # Split each omics dataset
    splits = {}
    for omics_type, df in data_dict.items():
        splits[omics_type] = {
            'train': df.loc[train_samples],
            'test': df.loc[test_samples]
        }
        if val_samples:
            splits[omics_type]['val'] = df.loc[val_samples]
    
    print(f"Data split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    return splits

def create_data_loaders(X: Union[np.ndarray, torch.Tensor], 
                       y: Union[np.ndarray, torch.Tensor],
                       batch_size: int = 32,
                       shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    Create PyTorch data loader from numpy arrays or tensors.
    
    Args:
        X: Feature data
        y: Target data
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        PyTorch DataLoader
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Convert to tensors if needed
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if isinstance(y, np.ndarray):
        y = torch.FloatTensor(y)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader

def calculate_data_statistics(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Calculate comprehensive statistics for multiomics data.
    
    Args:
        data_dict: Dictionary of omics dataframes
        
    Returns:
        Dictionary of statistics for each omics type
    """
    stats = {}
    
    for omics_type, df in data_dict.items():
        stats[omics_type] = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': df.isnull().sum().sum() / df.size * 100,
            'mean': df.mean().mean(),
            'std': df.std().mean(),
            'min': df.min().min(),
            'max': df.max().max(),
            'zero_values': (df == 0).sum().sum(),
            'zero_percentage': (df == 0).sum().sum() / df.size * 100
        }
    
    return stats

def detect_outliers(df: pd.DataFrame, 
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in the data.
    
    Args:
        df: Input dataframe
        method: Outlier detection method ('iqr', 'zscore', 'isolation')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean dataframe indicating outliers
    """
    if method == 'iqr':
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))
        
    elif method == 'zscore':
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > threshold
        
    elif method == 'isolation':
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(df)
        outliers = pd.DataFrame(
            outlier_labels.reshape(-1, 1) == -1,
            index=df.index,
            columns=['outlier']
        )
        
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return outliers

def batch_correct_data(data_dict: Dict[str, pd.DataFrame],
                      batch_info: pd.Series,
                      method: str = 'combat') -> Dict[str, pd.DataFrame]:
    """
    Correct for batch effects in multiomics data.
    
    Args:
        data_dict: Dictionary of omics dataframes
        batch_info: Series with batch information for each sample
        method: Batch correction method ('combat', 'limma')
        
    Returns:
        Dictionary of batch-corrected dataframes
    """
    try:
        if method == 'combat':
            from combat.pycombat import pycombat
            
            corrected_data = {}
            for omics_type, df in data_dict.items():
                # Ensure samples match
                common_samples = df.index.intersection(batch_info.index)
                df_subset = df.loc[common_samples]
                batch_subset = batch_info.loc[common_samples]
                
                # Apply ComBat
                corrected = pycombat(df_subset.T, batch_subset)
                corrected_data[omics_type] = corrected.T
                
                print(f"Applied ComBat correction to {omics_type}")
                
        else:
            raise ValueError(f"Batch correction method {method} not implemented")
            
    except ImportError:
        print("Warning: Batch correction libraries not available")
        return data_dict
    
    return corrected_data

def quality_control_report(data_dict: Dict[str, pd.DataFrame]) -> Dict:
    """
    Generate comprehensive quality control report.
    
    Args:
        data_dict: Dictionary of omics dataframes
        
    Returns:
        Quality control report dictionary
    """
    report = {
        'summary': {},
        'statistics': calculate_data_statistics(data_dict),
        'recommendations': []
    }
    
    for omics_type, df in data_dict.items():
        stats = report['statistics'][omics_type]
        
        # Check data quality issues
        if stats['missing_percentage'] > 20:
            report['recommendations'].append(
                f"{omics_type}: High missing data ({stats['missing_percentage']:.1f}%) - consider imputation"
            )
        
        if stats['zero_percentage'] > 50:
            report['recommendations'].append(
                f"{omics_type}: High sparsity ({stats['zero_percentage']:.1f}%) - consider filtering"
            )
        
        # Check for potential issues
        if df.std().std() > df.mean().mean():
            report['recommendations'].append(
                f"{omics_type}: High variance in feature scales - consider normalization"
            )
    
    report['summary']['total_samples'] = len(data_dict[list(data_dict.keys())[0]])
    report['summary']['total_features'] = sum(df.shape[1] for df in data_dict.values())
    report['summary']['omics_types'] = list(data_dict.keys())
    
    return report

def save_processed_data(data_dict: Dict[str, pd.DataFrame],
                       output_dir: str,
                       format: str = 'csv') -> None:
    """
    Save processed multiomics data to files.
    
    Args:
        data_dict: Dictionary of processed dataframes
        output_dir: Output directory
        format: Output format ('csv', 'h5', 'pickle')
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    for omics_type, df in data_dict.items():
        if format == 'csv':
            filepath = os.path.join(output_dir, f"{omics_type}_processed.csv")
            df.to_csv(filepath)
        elif format == 'h5':
            filepath = os.path.join(output_dir, f"{omics_type}_processed.h5")
            df.to_hdf(filepath, key='data', mode='w')
        elif format == 'pickle':
            filepath = os.path.join(output_dir, f"{omics_type}_processed.pkl")
            df.to_pickle(filepath)
        
        print(f"Saved {omics_type} data to {filepath}")

# Example usage and testing functions
def example_usage():
    """
    Example usage of the data utilities.
    """
    print("=== Data Utilities Example ===")
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    
    data_dict = {
        'genomics': pd.DataFrame(
            np.random.binomial(2, 0.3, (100, 1000)),
            index=[f"Patient_{i}" for i in range(100)],
            columns=[f"SNP_{i}" for i in range(1000)]
        ),
        'transcriptomics': pd.DataFrame(
            np.random.lognormal(0, 1, (100, 2000)),
            index=[f"Patient_{i}" for i in range(100)],
            columns=[f"Gene_{i}" for i in range(2000)]
        )
    }
    
    # Add some missing values
    data_dict['transcriptomics'].iloc[0:10, 0:50] = np.nan
    
    # Validate data consistency
    is_consistent = validate_data_consistency(data_dict)
    print(f"Data consistency: {is_consistent}")
    
    # Handle missing values
    data_dict['transcriptomics'] = handle_missing_values(
        data_dict['transcriptomics'], 
        strategy='median',
        omics_type='transcriptomics'
    )
    
    # Normalize data
    for omics_type in data_dict:
        data_dict[omics_type], _ = normalize_omics_data(
            data_dict[omics_type],
            method='standard',
            omics_type=omics_type
        )
    
    # Generate QC report
    qc_report = quality_control_report(data_dict)
    print("\nQuality Control Report:")
    for rec in qc_report['recommendations']:
        print(f"- {rec}")
    
    print("\n=== Example Complete ===")

if __name__ == "__main__":
    example_usage()

