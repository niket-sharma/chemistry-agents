#!/usr/bin/env python3
"""
Fine-tuning script for transformer-based molecular property prediction
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chemistry_agents.models.transformer_model import MolecularTransformer, MolecularPropertyDataset, SMILESProcessor
from chemistry_agents.utils.data_processing import DataProcessor
from chemistry_agents.utils.evaluation import ModelEvaluator

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fine_tuning.log')
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fine-tune transformer models for molecular property prediction')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training dataset (CSV format)')
    parser.add_argument('--smiles_column', type=str, default='smiles',
                       help='Name of SMILES column')
    parser.add_argument('--target_column', type=str, default='target',
                       help='Name of target column')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='DeepChem/ChemBERTa-77M-MLM',
                       help='Pre-trained transformer model name')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for prediction head')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of layers in prediction head')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--freeze_transformer', action='store_true',
                       help='Freeze transformer weights during fine-tuning')
    
    # Training arguments (CPU-optimized defaults)
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training (reduced for CPU)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (reduced for CPU)')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                       help='Learning rate (adjusted for smaller batches)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for regularization')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Number of warmup steps (reduced for CPU)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Number of steps to accumulate gradients (increased for CPU)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    
    # API and cloud options
    parser.add_argument('--use_api', action='store_true',
                       help='Use external API instead of local training')
    parser.add_argument('--api_provider', type=str, default='huggingface',
                       choices=['huggingface', 'openai', 'cohere'],
                       help='API provider for external training/inference')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key for external services')
    parser.add_argument('--cloud_training', action='store_true',
                       help='Generate scripts for cloud training platforms')
    
    # Data split arguments
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set size')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    # Data augmentation
    parser.add_argument('--use_augmentation', action='store_true',
                       help='Use SMILES augmentation')
    parser.add_argument('--augmentation_factor', type=int, default=3,
                       help='Number of augmented SMILES per molecule')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./fine_tuned_models',
                       help='Output directory for fine-tuned models')
    parser.add_argument('--model_save_name', type=str, default='molecular_transformer',
                       help='Name for saved model')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save model every N steps')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of workers for data loading')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def setup_device(device_arg: str) -> torch.device:
    """Setup computation device with CPU optimization"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            print("⚠️  GPU detected but defaulting to CPU for compatibility.")
            print("💡 For faster training, consider:")
            print("   - Google Colab (free GPU): https://colab.research.google.com/")
            print("   - Hugging Face Spaces (free inference)")
            print("   - AWS SageMaker / Azure ML")
            print("   - Use --device cuda to force GPU usage")
        device = torch.device('cpu')
    else:
        device = torch.device(device_arg)
    
    if device.type == 'cpu':
        print("🔧 CPU optimization enabled:")
        print("   - Reduced batch sizes")
        print("   - Optimized for CPU inference")
        print("   - Consider smaller models for better performance")
    
    return device

def load_and_preprocess_data(args, logger):
    """Load and preprocess the dataset"""
    logger.info(f"Loading dataset from {args.data_path}")
    
    # Initialize data processor
    processor = DataProcessor(
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
    
    # Data augmentation
    if args.use_augmentation:
        logger.info("Applying SMILES augmentation...")
        from chemistry_agents.utils.data_processing import DataAugmentation
        augmenter = DataAugmentation()
        df = augmenter.augment_dataset(
            df, 
            args.smiles_column, 
            args.target_column,
            args.augmentation_factor
        )
    
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
    
    return train_df, val_df, test_df

def create_datasets_and_loaders(train_df, val_df, test_df, args, logger):
    """Create datasets and data loaders"""
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = MolecularPropertyDataset(
        train_df[args.smiles_column].tolist(),
        train_df[args.target_column].values,
        tokenizer,
        max_length=args.max_length
    )
    
    val_dataset = MolecularPropertyDataset(
        val_df[args.smiles_column].tolist(),
        val_df[args.target_column].values,
        tokenizer,
        max_length=args.max_length
    ) if len(val_df) > 0 else None
    
    test_dataset = MolecularPropertyDataset(
        test_df[args.smiles_column].tolist(),
        test_df[args.target_column].values,
        tokenizer,
        max_length=args.max_length
    )
    
    logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}, Test: {len(test_dataset)}")
    
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
    
    return train_loader, val_loader, test_loader, tokenizer

def create_model(args, device, logger):
    """Create the transformer model"""
    
    model = MolecularTransformer(
        model_name=args.model_name,
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=1,
        dropout_rate=args.dropout_rate,
        freeze_transformer=args.freeze_transformer
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created model with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model

def setup_training_components(model, train_loader, args, logger):
    """Setup optimizer, scheduler, and other training components"""
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {args.warmup_steps}")
    
    return optimizer, scheduler, criterion

def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, args, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), labels)
        
        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        total_loss += loss.item() * args.gradient_accumulation_steps
        
        # Update weights
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update optimizer and scheduler
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item() * args.gradient_accumulation_steps})
    
    avg_loss = total_loss / num_batches
    return avg_loss

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            
            total_loss += loss.item()
            
            predictions.extend(outputs.squeeze().cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss, np.array(predictions), np.array(targets)

def fine_tune_model(model, train_loader, val_loader, optimizer, scheduler, criterion, args, device, logger):
    """Main fine-tuning loop"""
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    logger.info("Starting fine-tuning...")
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, args, logger)
        history['train_loss'].append(train_loss)
        
        # Validation
        if val_loader is not None:
            val_loss, val_pred, val_true = validate_epoch(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            
            # Calculate validation metrics
            evaluator = ModelEvaluator(task_type="regression")
            val_metrics = evaluator.evaluate_predictions(val_true, val_pred)
            history['val_metrics'].append(val_metrics)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_metrics.get('r2', 0):.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                logger.info("New best model saved!")
        else:
            logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % (args.epochs // 5) == 0:  # Save 5 times during training
            checkpoint_path = os.path.join(args.output_dir, f"{args.model_save_name}_epoch_{epoch+1}.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Load best model if validation was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model state")
    
    return model, history

def evaluate_final_model(model, test_loader, device, logger):
    """Evaluate the final fine-tuned model"""
    
    criterion = nn.MSELoss()
    test_loss, test_pred, test_true = validate_epoch(model, test_loader, criterion, device)
    
    # Calculate comprehensive metrics
    evaluator = ModelEvaluator(task_type="regression")
    metrics = evaluator.evaluate_predictions(test_true, test_pred)
    
    logger.info("Final Test Results:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    return metrics, test_pred, test_true

def save_model_and_results(model, tokenizer, args, metrics, history, logger):
    """Save the fine-tuned model and results"""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model state dict
    model_path = os.path.join(args.output_dir, f"{args.model_save_name}.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model state dict saved to {model_path}")
    
    # Save model configuration
    config = {
        'model_name': args.model_name,
        'max_length': args.max_length,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'output_dim': 1,
        'dropout_rate': args.dropout_rate,
        'freeze_transformer': args.freeze_transformer
    }
    
    config_path = os.path.join(args.output_dir, f"{args.model_save_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save tokenizer
    tokenizer_path = os.path.join(args.output_dir, f"{args.model_save_name}_tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    
    # Save results
    results = {
        'model_name': args.model_save_name,
        'base_model': args.model_name,
        'training_args': vars(args),
        'final_metrics': metrics,
        'training_history': history
    }
    
    results_path = os.path.join(args.output_dir, f"{args.model_save_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All outputs saved to {args.output_dir}")

def main():
    """Main fine-tuning function"""
    
    # Parse arguments
    args = parse_args()
    
    # Handle API and cloud options
    if args.use_api:
        handle_api_training(args)
        return
    
    if args.cloud_training:
        handle_cloud_training(args)
        return
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting transformer fine-tuning for molecular property prediction")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup device
    device = setup_device(args.device)
    logger.info(f"Using device: {device}")
    
    # CPU-specific warnings and optimizations
    if device.type == 'cpu':
        logger.warning("⚠️  Training on CPU will be significantly slower than GPU")
        logger.info("💡 Consider using cloud training options:")
        logger.info("   --cloud_training for setup scripts")
        logger.info("   --use_api for external inference")
        
        # Adjust parameters for CPU
        if args.batch_size > 8:
            logger.info(f"🔧 Reducing batch size from {args.batch_size} to 4 for CPU efficiency")
            args.batch_size = 4
        
        if args.gradient_accumulation_steps < 4:
            logger.info(f"🔧 Increasing gradient accumulation to maintain effective batch size")
            args.gradient_accumulation_steps = 4
    
    try:
        # Load and preprocess data
        train_df, val_df, test_df = load_and_preprocess_data(args, logger)
        
        # Create datasets and data loaders
        train_loader, val_loader, test_loader, tokenizer = create_datasets_and_loaders(
            train_df, val_df, test_df, args, logger
        )
        
        # Create model
        model = create_model(args, device, logger)
        
        # Setup training components
        optimizer, scheduler, criterion = setup_training_components(model, train_loader, args, logger)
        
        # Fine-tune model
        model, history = fine_tune_model(
            model, train_loader, val_loader, optimizer, scheduler, criterion, args, device, logger
        )
        
        # Final evaluation
        metrics, test_pred, test_true = evaluate_final_model(model, test_loader, device, logger)
        
        # Save model and results
        save_model_and_results(model, tokenizer, args, metrics, history, logger)
        
        logger.info("Fine-tuning completed successfully!")
        
        # Show alternatives for future use
        if device.type == 'cpu':
            show_gpu_alternatives()
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        if device.type == 'cpu':
            logger.info("💡 For better performance, try cloud training options")
        raise

def handle_api_training(args):
    """Handle API-based training/inference"""
    from chemistry_agents.utils.api_integration import get_api_model, show_free_alternatives
    
    print("🌐 API Training Mode")
    print("=" * 50)
    
    if not args.api_key:
        print("⚠️  No API key provided. Showing free alternatives...")
        show_free_alternatives()
        return
    
    # Initialize API model
    api_model = get_api_model(
        provider=args.api_provider,
        api_key=args.api_key,
        model_name=args.model_name
    )
    
    if not api_model.is_available():
        print(f"❌ Model {args.model_name} not available via {args.api_provider} API")
        return
    
    print(f"✅ Connected to {args.api_provider} API")
    print("📝 Note: API training is limited. Consider cloud training for full fine-tuning.")
    
    # Load data for inference testing
    if os.path.exists(args.data_path):
        import pandas as pd
        df = pd.read_csv(args.data_path)
        sample_smiles = df[args.smiles_column].head(5).tolist()
        
        print("🧪 Testing API with sample data...")
        predictions = api_model.predict(sample_smiles)
        
        for smiles, pred in zip(sample_smiles, predictions):
            print(f"   {smiles}: {pred:.3f}")

def handle_cloud_training(args):
    """Handle cloud training setup"""
    from chemistry_agents.utils.api_integration import setup_cloud_training, show_free_alternatives
    
    print("☁️  Cloud Training Setup")
    print("=" * 50)
    
    show_free_alternatives()
    
    print("\n🚀 Generating setup files...")
    
    # Generate Google Colab notebook
    setup_cloud_training("colab", args.data_path)
    
    print("\n📋 Next steps:")
    print("1. Upload chemistry_agents_colab.ipynb to Google Colab")
    print("2. Upload your dataset when prompted")
    print("3. Run all cells to train with free GPU")
    print("4. Download the trained model when complete")

def show_gpu_alternatives():
    """Show alternatives to local CPU training"""
    from chemistry_agents.utils.api_integration import show_free_alternatives
    
    print("\n" + "="*60)
    print("🚀 Want faster training? Try these GPU options:")
    print("="*60)
    
    show_free_alternatives()
    
    print("\n🔧 To generate cloud training setup:")
    print("   python scripts/fine_tune_transformer.py --cloud_training --data_path your_data.csv")

if __name__ == "__main__":
    main()