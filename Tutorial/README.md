# Getting Started with Foundation Models for Medical Digital Twins

This tutorial provides a step-by-step guide to understanding and implementing foundation models for medical digital twins using multiomics data.

## Prerequisites

Before starting this tutorial, ensure you have:

- Basic knowledge of Python programming
- Understanding of machine learning concepts
- Familiarity with bioinformatics and omics data
- Python 3.8+ installed with the following packages:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn jupyter
```

## Tutorial Overview

This tutorial covers:

1. **Understanding Medical Digital Twins**: Conceptual foundation
2. **Multiomics Data Types**: Genomics, transcriptomics, proteomics, metabolomics, epigenomics
3. **Foundation Models**: Architecture and applications
4. **Data Integration**: Combining multiple omics datasets
5. **Model Training**: Fine-tuning foundation models
6. **Clinical Applications**: Real-world use cases

## Step 1: Understanding Medical Digital Twins

### What are Medical Digital Twins?

Medical Digital Twins (MDTs) are virtual representations of patients that simulate biological, physiological, and clinical processes. They integrate data from multiple sources to create personalized models for:

- **Treatment optimization**
- **Disease prediction**
- **Drug discovery**
- **Clinical decision support**

### Key Components

```python
# Conceptual structure of a Medical Digital Twin
class MedicalDigitalTwin:
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.genomics_data = None
        self.transcriptomics_data = None
        self.proteomics_data = None
        self.metabolomics_data = None
        self.epigenomics_data = None
        self.clinical_data = None
        self.foundation_model = None
    
    def integrate_omics_data(self):
        """Integrate multiple omics datasets"""
        pass
    
    def predict_outcomes(self, intervention):
        """Predict clinical outcomes for given intervention"""
        pass
    
    def optimize_treatment(self, constraints):
        """Optimize treatment plan based on constraints"""
        pass
```

## Step 2: Understanding Multiomics Data

### Genomics Data
- **Type**: DNA sequence variations (SNPs, CNVs, structural variants)
- **Characteristics**: Stable, inherited, binary/categorical
- **Applications**: Disease susceptibility, pharmacogenomics

```python
# Example genomics data structure
genomics_features = [
    'rs123456_A>G',  # SNP variant
    'CNV_chr1_1000000_1500000',  # Copy number variant
    'SV_chr2_translocation'  # Structural variant
]
```

### Transcriptomics Data
- **Type**: RNA expression levels
- **Characteristics**: Dynamic, tissue-specific, continuous
- **Applications**: Gene regulation, pathway analysis

```python
# Example transcriptomics data
transcriptomics_features = [
    'GENE_BRCA1_expression',
    'GENE_TP53_expression',
    'miRNA_hsa-mir-21_expression'
]
```

### Proteomics Data
- **Type**: Protein abundance and modifications
- **Characteristics**: Functional, post-translational modifications
- **Applications**: Biomarker discovery, drug targets

### Metabolomics Data
- **Type**: Small molecule concentrations
- **Characteristics**: Phenotypic endpoint, environmental influence
- **Applications**: Disease diagnosis, drug metabolism

### Epigenomics Data
- **Type**: DNA methylation, histone modifications
- **Characteristics**: Heritable, environmentally influenced
- **Applications**: Gene regulation, disease mechanisms

## Step 3: Foundation Models Architecture

### Transformer-Based Models

Foundation models for multiomics typically use transformer architectures:

```python
import torch
import torch.nn as nn

class MultiomicsTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8, num_layers=6):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Project input features
        x = self.input_projection(x)
        
        # Add positional encoding (simplified)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global pooling and normalization
        x = x.mean(dim=1)  # Average pooling
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x
```

### Key Features of Foundation Models

1. **Pre-training**: Trained on large, diverse datasets
2. **Transfer Learning**: Adaptable to specific tasks
3. **Few-shot Learning**: Effective with limited labeled data
4. **Multimodal**: Can process different data types

## Step 4: Data Integration Pipeline

### Data Preprocessing

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_omics_data(data, omics_type):
    """
    Preprocess omics data based on type-specific requirements.
    """
    if omics_type == 'genomics':
        # Handle missing values and encode variants
        data = data.fillna(0)  # Missing = reference allele
        return data
    
    elif omics_type in ['transcriptomics', 'proteomics']:
        # Log transformation and normalization
        data_log = np.log1p(data.clip(lower=0))
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data_log),
            index=data.index,
            columns=data.columns
        )
        return data_scaled, scaler
    
    elif omics_type in ['metabolomics', 'epigenomics']:
        # Standard normalization
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            index=data.index,
            columns=data.columns
        )
        return data_scaled, scaler
```

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_regression

def select_important_features(X, y, k=1000):
    """
    Select the most important features for the prediction task.
    """
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()]
    
    return X_selected, selected_features, selector
```

### Data Integration Methods

```python
def integrate_omics_data(omics_dict, method='concatenation'):
    """
    Integrate multiple omics datasets.
    
    Args:
        omics_dict: Dictionary of preprocessed omics dataframes
        method: Integration method ('concatenation', 'pca', 'weighted')
    """
    if method == 'concatenation':
        # Simple concatenation
        integrated = pd.concat(omics_dict.values(), axis=1)
        
    elif method == 'pca':
        # PCA-based integration
        from sklearn.decomposition import PCA
        
        integrated_features = []
        for omics_type, data in omics_dict.items():
            # Reduce dimensionality with PCA
            pca = PCA(n_components=min(50, data.shape[1]//2))
            pca_features = pca.fit_transform(data)
            integrated_features.append(pca_features)
        
        # Combine PCA features
        integrated = np.hstack(integrated_features)
        
    elif method == 'weighted':
        # Weighted combination based on omics importance
        weights = {
            'genomics': 0.2,
            'transcriptomics': 0.3,
            'proteomics': 0.25,
            'metabolomics': 0.15,
            'epigenomics': 0.1
        }
        
        weighted_data = []
        for omics_type, data in omics_dict.items():
            weight = weights.get(omics_type, 0.2)
            weighted_data.append(data * weight)
        
        integrated = pd.concat(weighted_data, axis=1)
    
    return integrated
```

## Step 5: Model Training Pipeline

### Training Setup

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def setup_training(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Setup training and validation data loaders.
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_val_tensor = torch.FloatTensor(X_val.values)
    y_val_tensor = torch.FloatTensor(y_val.values)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
```

### Training Loop

```python
def train_foundation_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    """
    Train the foundation model with early stopping.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}")
    
    return history
```

## Step 6: Clinical Applications

### Disease Risk Prediction

```python
def predict_disease_risk(model, patient_data, disease_models):
    """
    Predict disease risk for a patient using the trained model.
    """
    model.eval()
    with torch.no_grad():
        # Get patient embedding from foundation model
        patient_embedding = model(patient_data)
        
        # Predict risks for different diseases
        risks = {}
        for disease, disease_model in disease_models.items():
            risk_score = disease_model(patient_embedding)
            risks[disease] = torch.sigmoid(risk_score).item()
    
    return risks
```

### Treatment Optimization

```python
def optimize_treatment(model, patient_data, treatment_options, constraints):
    """
    Optimize treatment selection based on predicted outcomes.
    """
    best_treatment = None
    best_outcome = float('-inf')
    
    for treatment in treatment_options:
        # Simulate treatment effect
        modified_data = apply_treatment_effect(patient_data, treatment)
        
        # Predict outcome
        predicted_outcome = model(modified_data)
        
        # Check constraints
        if satisfies_constraints(treatment, constraints):
            if predicted_outcome > best_outcome:
                best_outcome = predicted_outcome
                best_treatment = treatment
    
    return best_treatment, best_outcome
```

### Biomarker Discovery

```python
def discover_biomarkers(model, data, labels, top_k=50):
    """
    Discover important biomarkers using model interpretability.
    """
    from captum import attr
    
    # Use integrated gradients for feature importance
    ig = attr.IntegratedGradients(model)
    
    # Calculate attributions
    attributions = ig.attribute(data, target=labels)
    
    # Get top features
    feature_importance = torch.abs(attributions).mean(dim=0)
    top_indices = torch.topk(feature_importance, top_k).indices
    
    return top_indices, feature_importance
```

## Step 7: Evaluation and Validation

### Model Performance Metrics

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(model, test_loader):
    """
    Evaluate model performance on test data.
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': np.sqrt(mse)
    }
```

### Clinical Validation

```python
def clinical_validation(model, clinical_data, clinical_outcomes):
    """
    Validate model predictions against clinical outcomes.
    """
    # Predict outcomes
    predictions = model(clinical_data)
    
    # Calculate clinical metrics
    sensitivity = calculate_sensitivity(predictions, clinical_outcomes)
    specificity = calculate_specificity(predictions, clinical_outcomes)
    ppv = calculate_ppv(predictions, clinical_outcomes)
    npv = calculate_npv(predictions, clinical_outcomes)
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv
    }
```

## Step 8: Deployment and Monitoring

### Model Deployment

```python
class DigitalTwinAPI:
    """
    API for deploying medical digital twin models.
    """
    
    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()
    
    def predict_risk(self, patient_data):
        """API endpoint for risk prediction"""
        with torch.no_grad():
            risk_scores = self.model(patient_data)
        return risk_scores.tolist()
    
    def recommend_treatment(self, patient_data, treatment_options):
        """API endpoint for treatment recommendation"""
        recommendations = optimize_treatment(
            self.model, patient_data, treatment_options, {}
        )
        return recommendations
```

### Continuous Learning

```python
def update_model_with_new_data(model, new_data, new_labels):
    """
    Update model with new patient data (continual learning).
    """
    # Implement continual learning strategy
    # e.g., elastic weight consolidation, replay buffer
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    # Fine-tune on new data
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(new_data)
        loss = criterion(outputs, new_labels)
        loss.backward()
        optimizer.step()
    
    return model
```

## Best Practices and Considerations

### Data Quality
- **Standardization**: Use consistent data formats and units
- **Quality Control**: Implement robust QC procedures
- **Missing Data**: Handle missing values appropriately
- **Batch Effects**: Correct for technical variations

### Model Development
- **Validation Strategy**: Use appropriate cross-validation
- **Hyperparameter Tuning**: Systematic optimization
- **Regularization**: Prevent overfitting
- **Interpretability**: Ensure model decisions are explainable

### Clinical Translation
- **Regulatory Compliance**: Follow FDA/EMA guidelines
- **Clinical Validation**: Validate in real clinical settings
- **Ethical Considerations**: Address bias and fairness
- **Privacy Protection**: Implement strong data protection

### Deployment
- **Scalability**: Design for large-scale deployment
- **Monitoring**: Continuous performance monitoring
- **Updates**: Regular model updates with new data
- **Integration**: Seamless EHR integration

## Next Steps

After completing this tutorial, you can:

1. **Explore Advanced Topics**:
   - Federated learning for multi-institutional collaboration
   - Quantum computing applications
   - Advanced interpretability methods

2. **Implement Real Applications**:
   - Disease-specific digital twins
   - Drug discovery pipelines
   - Clinical decision support systems

3. **Contribute to Research**:
   - Develop new integration methods
   - Improve model architectures
   - Address ethical and regulatory challenges

## Resources

- **Papers**: See `references/bibliography.bib` for key publications
- **Code Examples**: Check `code/examples/` for implementation details
- **Documentation**: Review `docs/` for comprehensive guides
- **Visualizations**: Explore `figures/` for conceptual diagrams

## Support

For questions or issues:
- Check the repository documentation
- Review example implementations
- Consult the research papers
- Engage with the research community

---

This tutorial provides a foundation for understanding and implementing foundation models for medical digital twins. The field is rapidly evolving, so stay updated with the latest research and best practices.

