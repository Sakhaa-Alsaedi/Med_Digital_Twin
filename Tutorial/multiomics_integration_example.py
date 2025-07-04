#!/usr/bin/env python3
"""
Multiomics Data Integration Example for Medical Digital Twins

This example demonstrates how to integrate multiple omics data types
for building foundation model-enhanced medical digital twins.

Author: Sakhaa Alsaedi
License: MIT
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

class MultiomicsIntegrator:
    """
    A class for integrating multiple omics data types for digital twin construction.
    
    This class provides methods for preprocessing, integrating, and analyzing
    multiomics data including genomics, transcriptomics, proteomics, 
    metabolomics, and epigenomics.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the MultiomicsIntegrator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scalers = {}
        self.integrated_data = None
        self.feature_importance = None
        
    def load_omics_data(self, data_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Load multiomics data from various sources.
        
        Args:
            data_paths: Dictionary mapping omics type to file path
            
        Returns:
            Dictionary of loaded dataframes for each omics type
        """
        omics_data = {}
        
        for omics_type, path in data_paths.items():
            try:
                # In practice, this would load real data files
                # For demonstration, we'll generate synthetic data
                omics_data[omics_type] = self._generate_synthetic_data(omics_type)
                print(f"Loaded {omics_type} data: {omics_data[omics_type].shape}")
            except Exception as e:
                print(f"Error loading {omics_type} data: {e}")
                
        return omics_data
    
    def _generate_synthetic_data(self, omics_type: str) -> pd.DataFrame:
        """
        Generate synthetic omics data for demonstration purposes.
        
        Args:
            omics_type: Type of omics data to generate
            
        Returns:
            Synthetic dataframe with appropriate characteristics
        """
        np.random.seed(self.random_state)
        
        # Define characteristics for each omics type
        omics_specs = {
            'genomics': {'n_features': 1000, 'n_samples': 500, 'sparsity': 0.1},
            'transcriptomics': {'n_features': 20000, 'n_samples': 500, 'sparsity': 0.3},
            'proteomics': {'n_features': 5000, 'n_samples': 500, 'sparsity': 0.2},
            'metabolomics': {'n_features': 800, 'n_samples': 500, 'sparsity': 0.15},
            'epigenomics': {'n_features': 15000, 'n_samples': 500, 'sparsity': 0.25}
        }
        
        spec = omics_specs.get(omics_type, {'n_features': 1000, 'n_samples': 500, 'sparsity': 0.2})
        
        # Generate data with appropriate distribution
        if omics_type == 'genomics':
            # Binary SNP data
            data = np.random.binomial(2, 0.3, (spec['n_samples'], spec['n_features']))
        elif omics_type in ['transcriptomics', 'proteomics']:
            # Log-normal expression data
            data = np.random.lognormal(0, 1, (spec['n_samples'], spec['n_features']))
        else:
            # Normal distribution for other omics
            data = np.random.normal(0, 1, (spec['n_samples'], spec['n_features']))
        
        # Add sparsity
        mask = np.random.random((spec['n_samples'], spec['n_features'])) < spec['sparsity']
        data[mask] = 0
        
        # Create sample and feature names
        sample_names = [f"Patient_{i:03d}" for i in range(spec['n_samples'])]
        feature_names = [f"{omics_type}_{i:05d}" for i in range(spec['n_features'])]
        
        return pd.DataFrame(data, index=sample_names, columns=feature_names)
    
    def preprocess_omics_data(self, omics_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess multiomics data including normalization and quality control.
        
        Args:
            omics_data: Dictionary of raw omics dataframes
            
        Returns:
            Dictionary of preprocessed omics dataframes
        """
        processed_data = {}
        
        for omics_type, data in omics_data.items():
            print(f"Preprocessing {omics_type} data...")
            
            # Quality control: remove features with too many zeros
            feature_completeness = (data != 0).sum(axis=0) / len(data)
            valid_features = feature_completeness > 0.1
            data_filtered = data.loc[:, valid_features]
            
            # Normalization based on omics type
            if omics_type == 'genomics':
                # No normalization for SNP data
                processed_data[omics_type] = data_filtered
            elif omics_type in ['transcriptomics', 'proteomics']:
                # Log transformation and standardization
                data_log = np.log1p(data_filtered)
                scaler = StandardScaler()
                data_scaled = pd.DataFrame(
                    scaler.fit_transform(data_log),
                    index=data_log.index,
                    columns=data_log.columns
                )
                self.scalers[omics_type] = scaler
                processed_data[omics_type] = data_scaled
            else:
                # Standard normalization
                scaler = StandardScaler()
                data_scaled = pd.DataFrame(
                    scaler.fit_transform(data_filtered),
                    index=data_filtered.index,
                    columns=data_filtered.columns
                )
                self.scalers[omics_type] = scaler
                processed_data[omics_type] = data_scaled
            
            print(f"  Original shape: {data.shape}")
            print(f"  Processed shape: {processed_data[omics_type].shape}")
        
        return processed_data
    
    def integrate_omics_data(self, processed_data: Dict[str, pd.DataFrame], 
                           method: str = 'concatenation') -> pd.DataFrame:
        """
        Integrate multiple omics datasets using specified method.
        
        Args:
            processed_data: Dictionary of preprocessed omics dataframes
            method: Integration method ('concatenation', 'pca', 'weighted')
            
        Returns:
            Integrated multiomics dataframe
        """
        print(f"Integrating omics data using {method} method...")
        
        if method == 'concatenation':
            # Simple concatenation of all omics data
            integrated = pd.concat(processed_data.values(), axis=1)
            
        elif method == 'pca':
            # PCA-based integration
            integrated_features = []
            feature_names = []
            
            for omics_type, data in processed_data.items():
                # Apply PCA to reduce dimensionality
                n_components = min(50, data.shape[1] // 2)
                pca = PCA(n_components=n_components, random_state=self.random_state)
                pca_features = pca.fit_transform(data)
                
                # Create feature names
                pca_names = [f"{omics_type}_PC{i+1}" for i in range(n_components)]
                feature_names.extend(pca_names)
                integrated_features.append(pca_features)
            
            # Combine all PCA features
            integrated_array = np.hstack(integrated_features)
            integrated = pd.DataFrame(
                integrated_array,
                index=list(processed_data.values())[0].index,
                columns=feature_names
            )
            
        elif method == 'weighted':
            # Weighted integration based on omics importance
            weights = {
                'genomics': 0.2,
                'transcriptomics': 0.3,
                'proteomics': 0.25,
                'metabolomics': 0.15,
                'epigenomics': 0.1
            }
            
            weighted_data = []
            for omics_type, data in processed_data.items():
                weight = weights.get(omics_type, 0.2)
                weighted_data.append(data * weight)
            
            integrated = pd.concat(weighted_data, axis=1)
        
        else:
            raise ValueError(f"Unknown integration method: {method}")
        
        self.integrated_data = integrated
        print(f"Integrated data shape: {integrated.shape}")
        
        return integrated

class FoundationModelAdapter:
    """
    Adapter class for integrating foundation models with multiomics data.
    
    This class provides methods for adapting pre-trained foundation models
    to work with integrated multiomics data for digital twin applications.
    """
    
    def __init__(self, model_type: str = 'transformer', hidden_dim: int = 512):
        """
        Initialize the FoundationModelAdapter.
        
        Args:
            model_type: Type of foundation model architecture
            hidden_dim: Hidden dimension size
        """
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.model = None
        
    def build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """
        Build a foundation model architecture for multiomics integration.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (e.g., number of clinical outcomes)
            
        Returns:
            PyTorch model
        """
        if self.model_type == 'transformer':
            self.model = MultiomicsTransformer(input_dim, self.hidden_dim, output_dim)
        elif self.model_type == 'mlp':
            self.model = MultiomicsMLP(input_dim, self.hidden_dim, output_dim)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def fine_tune_model(self, X_train: torch.Tensor, y_train: torch.Tensor,
                       X_val: torch.Tensor, y_val: torch.Tensor,
                       epochs: int = 100, lr: float = 0.001) -> Dict[str, List[float]]:
        """
        Fine-tune the foundation model on multiomics data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            train_pred = self.model(X_train)
            train_loss = criterion(train_pred, y_train)
            train_loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val)
            
            history['train_loss'].append(train_loss.item())
            history['val_loss'].append(val_loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        return history

class MultiomicsTransformer(nn.Module):
    """
    Transformer-based model for multiomics data integration.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1
            ),
            num_layers=4
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Project input to hidden dimension
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Remove sequence dimension and project to output
        x = x.squeeze(1)
        x = self.output_projection(x)
        
        return x

class MultiomicsMLP(nn.Module):
    """
    Multi-layer perceptron for multiomics data integration.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

def visualize_integration_results(integrator: MultiomicsIntegrator, 
                                processed_data: Dict[str, pd.DataFrame]):
    """
    Visualize the results of multiomics integration.
    
    Args:
        integrator: MultiomicsIntegrator instance
        processed_data: Dictionary of processed omics data
    """
    # Create visualization plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Data dimensions by omics type
    omics_types = list(processed_data.keys())
    dimensions = [data.shape[1] for data in processed_data.values()]
    
    axes[0, 0].bar(omics_types, dimensions, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    axes[0, 0].set_title('Feature Dimensions by Omics Type')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: PCA of integrated data
    if integrator.integrated_data is not None:
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(integrator.integrated_data)
        
        axes[0, 1].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
        axes[0, 1].set_title('PCA of Integrated Multiomics Data')
        axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # Plot 3: Correlation heatmap between omics types
    if len(processed_data) > 1:
        correlations = []
        for i, (type1, data1) in enumerate(processed_data.items()):
            row = []
            for j, (type2, data2) in enumerate(processed_data.items()):
                if i == j:
                    row.append(1.0)
                else:
                    # Calculate correlation between mean profiles
                    corr = np.corrcoef(data1.mean(axis=1), data2.mean(axis=1))[0, 1]
                    row.append(corr)
            correlations.append(row)
        
        sns.heatmap(correlations, annot=True, xticklabels=omics_types, 
                   yticklabels=omics_types, ax=axes[1, 0], cmap='coolwarm', center=0)
        axes[1, 0].set_title('Inter-Omics Correlation Matrix')
    
    # Plot 4: Sample distribution
    if integrator.integrated_data is not None:
        sample_means = integrator.integrated_data.mean(axis=1)
        axes[1, 1].hist(sample_means, bins=30, alpha=0.7, color='skyblue')
        axes[1, 1].set_title('Distribution of Sample Mean Values')
        axes[1, 1].set_xlabel('Mean Feature Value')
        axes[1, 1].set_ylabel('Number of Samples')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/digital-twins-foundation-models-paper/figures/multiomics_integration_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function demonstrating multiomics integration workflow.
    """
    print("=== Multiomics Integration for Medical Digital Twins ===\n")
    
    # Initialize integrator
    integrator = MultiomicsIntegrator(random_state=42)
    
    # Define data paths (in practice, these would be real file paths)
    data_paths = {
        'genomics': 'data/genomics.csv',
        'transcriptomics': 'data/transcriptomics.csv',
        'proteomics': 'data/proteomics.csv',
        'metabolomics': 'data/metabolomics.csv',
        'epigenomics': 'data/epigenomics.csv'
    }
    
    # Load omics data
    print("1. Loading multiomics data...")
    omics_data = integrator.load_omics_data(data_paths)
    
    # Preprocess data
    print("\n2. Preprocessing omics data...")
    processed_data = integrator.preprocess_omics_data(omics_data)
    
    # Integrate data
    print("\n3. Integrating omics data...")
    integrated_data = integrator.integrate_omics_data(processed_data, method='pca')
    
    # Prepare data for foundation model
    print("\n4. Preparing data for foundation model...")
    X = torch.FloatTensor(integrated_data.values)
    
    # Generate synthetic clinical outcomes for demonstration
    np.random.seed(42)
    y = torch.FloatTensor(np.random.normal(0, 1, (X.shape[0], 3)))  # 3 clinical outcomes
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Initialize and train foundation model
    print("\n5. Training foundation model...")
    model_adapter = FoundationModelAdapter(model_type='transformer', hidden_dim=256)
    model = model_adapter.build_model(X.shape[1], y.shape[1])
    
    # Fine-tune model
    history = model_adapter.fine_tune_model(
        X_train, y_train, X_val, y_val, epochs=50, lr=0.001
    )
    
    # Evaluate model
    print("\n6. Evaluating model...")
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = nn.MSELoss()(test_pred, y_test)
        print(f"Test Loss: {test_loss:.4f}")
    
    # Visualize results
    print("\n7. Generating visualizations...")
    visualize_integration_results(integrator, processed_data)
    
    print("\n=== Multiomics Integration Complete ===")
    print(f"Integrated {len(processed_data)} omics types")
    print(f"Final integrated data shape: {integrated_data.shape}")
    print(f"Model test performance: {test_loss:.4f}")

if __name__ == "__main__":
    main()

