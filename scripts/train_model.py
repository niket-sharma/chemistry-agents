#!/usr/bin/env python3
"""
Training script for molecular property prediction models
"""

import argparse
import logging
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chemistry_agents.models.molecular_predictor import MolecularPropertyPredictor, MolecularFeatureExtractor
from chemistry_agents.models.transformer_model import MolecularTransformer, MolecularPropertyDataset
from chemistry_agents.utils.data_processing import DataProcessor, MolecularDataset
from chemistry_agents.utils.evaluation import ModelEvaluator

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train molecular property prediction models')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training dataset (CSV format)')
    parser.add_argument('--smiles_column', type=str, default='smiles',
                       help='Name of SMILES column')
    parser.add_argument('--target_column', type=str, default='target',
                       help='Name of target column')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='neural_network',
                       choices=['neural_network', 'transformer'],
                       help='Type of model to train')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 128],
                       help='Hidden layer dimensions for neural network')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--transformer_model', type=str, default='DeepChem/ChemBERTa-77M-MLM',
                       help='Pre-trained transformer model name')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for regularization')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Data split arguments
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set size')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                       help='Output directory for trained models')
    parser.add_argument('--model_name', type=str, default='molecular_predictor',
                       help='Name for saved model')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of workers for data loading')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()

def setup_device(device_arg: str) -> torch.device:
    """Setup computation device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    return device

def load_and_preprocess_data(args, logger):
    """Load and preprocess the dataset"""
    logger.info(f"Loading dataset from {args.data_path}")
    
    # Initialize data processor
    processor = DataProcessor(
        scaler_type="standard",
        remove_duplicates=True,
        handle_missing="drop"
    )
    
    # Load dataset
    df = processor.load_dataset(
        args.data_path,
        smiles_column=args.smiles_column,
        target_column=args.target_column
    )
    
    # Clean dataset
    df = processor.clean_dataset(df, args.smiles_column, args.target_column)
    
    # Analyze dataset
    analysis = processor.analyze_dataset(df, args.target_column)
    logger.info(f"Dataset analysis: {analysis}")
    
    # Split dataset
    train_df, val_df, test_df = processor.split_dataset(
        df,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
    return processor, train_df, val_df, test_df

def create_datasets(processor, train_df, val_df, test_df, args, logger):
    """Create PyTorch datasets"""
    
    if args.model_type == 'neural_network':
        # Extract molecular features
        logger.info("Extracting molecular features...")
        feature_extractor = MolecularFeatureExtractor()
        
        train_df = processor.create_molecular_descriptors(train_df, args.smiles_column)
        val_df = processor.create_molecular_descriptors(val_df, args.smiles_column)
        test_df = processor.create_molecular_descriptors(test_df, args.smiles_column)
        
        # Get feature columns
        feature_columns = [col for col in train_df.columns if col.startswith('descriptor_')]
        
        # Normalize features
        train_features = train_df[feature_columns].values
        val_features = val_df[feature_columns].values if len(val_df) > 0 else None
        test_features = test_df[feature_columns].values
        
        train_features_norm, val_features_norm, test_features_norm = processor.normalize_features(
            train_features, val_features, test_features
        )
        
        # Create datasets
        train_dataset = MolecularDataset(
            train_df[args.smiles_column].tolist(),
            train_df[args.target_column].values,
            feature_extractor=None  # Features already extracted
        )
        train_dataset.features = torch.FloatTensor(train_features_norm)
        
        if len(val_df) > 0:
            val_dataset = MolecularDataset(
                val_df[args.smiles_column].tolist(),
                val_df[args.target_column].values,
                feature_extractor=None
            )
            val_dataset.features = torch.FloatTensor(val_features_norm)
        else:
            val_dataset = None
        
        test_dataset = MolecularDataset(
            test_df[args.smiles_column].tolist(),
            test_df[args.target_column].values,
            feature_extractor=None
        )
        test_dataset.features = torch.FloatTensor(test_features_norm)
        
        input_dim = train_features_norm.shape[1]
        
    else:  # transformer
        logger.info("Creating transformer datasets...")
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(args.transformer_model)
        
        train_dataset = MolecularPropertyDataset(
            train_df[args.smiles_column].tolist(),
            train_df[args.target_column].values,
            tokenizer
        )
        
        if len(val_df) > 0:
            val_dataset = MolecularPropertyDataset(
                val_df[args.smiles_column].tolist(),
                val_df[args.target_column].values,
                tokenizer
            )
        else:
            val_dataset = None
        
        test_dataset = MolecularPropertyDataset(
            test_df[args.smiles_column].tolist(),
            test_df[args.target_column].values,
            tokenizer
        )
        
        input_dim = None  # Not needed for transformer
    
    return train_dataset, val_dataset, test_dataset, input_dim

def create_model(args, input_dim, device):
    """Create the model"""
    
    if args.model_type == 'neural_network':
        model = MolecularPropertyPredictor(
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            output_dim=1,
            dropout_rate=args.dropout_rate
        )
    else:  # transformer
        model = MolecularTransformer(
            model_name=args.transformer_model,
            hidden_dim=256,
            num_layers=2,
            output_dim=1,
            dropout_rate=args.dropout_rate
        )
    
    model = model.to(device)
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        if isinstance(batch, dict):
            # For datasets that return dictionaries
            if 'input_ids' in batch:  # Transformer
                outputs = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            else:  # Neural network
                outputs = model(batch['features'].to(device))
            targets = batch['target'].to(device)
        else:
            # For simple datasets
            features, targets = batch
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
        
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                if 'input_ids' in batch:  # Transformer
                    outputs = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                else:  # Neural network
                    outputs = model(batch['features'].to(device))
                batch_targets = batch['target'].to(device)
            else:
                features, batch_targets = batch
                features, batch_targets = features.to(device), batch_targets.to(device)
                outputs = model(features)
            
            loss = criterion(outputs.squeeze(), batch_targets)
            total_loss += loss.item()
            num_batches += 1
            
            predictions.extend(outputs.squeeze().cpu().numpy())
            targets.extend(batch_targets.cpu().numpy())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss, np.array(predictions), np.array(targets)

def train_model(model, train_loader, val_loader, args, device, logger):
    """Main training loop"""
    
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        history['train_loss'].append(train_loss)
        
        # Validation
        if val_loader is not None:
            val_loss, val_pred, val_true = validate_epoch(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        else:
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")
        
        # Record learning rate
        history['lr'].append(optimizer.param_groups[0]['lr'])
    
    # Load best model if validation was used
    if val_loader is not None and 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
    
    return model, history

def evaluate_model(model, test_loader, device, logger):
    """Evaluate the trained model"""
    
    criterion = nn.MSELoss()
    test_loss, test_pred, test_true = validate_epoch(model, test_loader, criterion, device)
    
    # Calculate metrics
    evaluator = ModelEvaluator(task_type="regression")
    metrics = evaluator.evaluate_predictions(test_true, test_pred)
    
    logger.info("Test Results:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    return metrics, test_pred, test_true

def save_model_and_results(model, args, metrics, history, logger):
    """Save the trained model and results"""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(args.output_dir, f"{args.model_name}.pt")
    
    if args.model_type == 'neural_network':
        model.save_model(model_path)
    else:
        torch.save(model.state_dict(), model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    # Save results
    results = {
        'model_type': args.model_type,
        'model_name': args.model_name,
        'training_args': vars(args),
        'metrics': metrics,
        'training_history': history
    }
    
    results_path = os.path.join(args.output_dir, f"{args.model_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")

def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting molecular property prediction model training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup device
    device = setup_device(args.device)
    logger.info(f"Using device: {device}")
    
    try:
        # Load and preprocess data
        processor, train_df, val_df, test_df = load_and_preprocess_data(args, logger)
        
        # Create datasets
        train_dataset, val_dataset, test_dataset, input_dim = create_datasets(
            processor, train_df, val_df, test_df, args, logger
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        ) if val_dataset is not None else None
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        # Create model
        model = create_model(args, input_dim, device)
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model
        model, history = train_model(model, train_loader, val_loader, args, device, logger)
        
        # Evaluate model
        metrics, test_pred, test_true = evaluate_model(model, test_loader, device, logger)
        
        # Save model and results
        save_model_and_results(model, args, metrics, history, logger)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()