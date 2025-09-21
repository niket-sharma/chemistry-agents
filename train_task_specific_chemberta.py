#!/usr/bin/env python3
"""
Task-Specific ChemBERTa Training Pipeline

Fine-tunes ChemBERTa on specific datasets for different tasks:
- Solubility prediction (ESOL dataset)
- Toxicity prediction (toxicity dataset)
- Bioactivity prediction (bioactivity dataset)

The agent will then intelligently choose the right specialized model based on query type.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from typing import Dict, List, Tuple, Optional
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChemBERTaDataset(Dataset):
    """Dataset for ChemBERTa fine-tuning"""

    def __init__(self, smiles_list: List[str], targets: List[float], tokenizer, max_length: int = 512):
        self.smiles_list = smiles_list
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = str(self.smiles_list[idx])
        target = float(self.targets[idx])

        # Tokenize SMILES
        encoding = self.tokenizer(
            smiles,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.float)
        }

class ChemBERTaRegressor(nn.Module):
    """ChemBERTa with regression head for property prediction"""

    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MLM", dropout: float = 0.3):
        super().__init__()
        self.chemberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.chemberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.regressor(output)

class TaskSpecificChemBERTaTrainer:
    """Trainer for task-specific ChemBERTa models"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        self.models = {}  # Store trained models
        self.task_info = {}  # Store task information

    def load_dataset(self, file_path: str, task_type: str) -> Tuple[List[str], List[float], Dict]:
        """Load and prepare dataset for specific task"""

        logger.info(f"Loading {task_type} dataset from {file_path}")
        df = pd.read_csv(file_path)

        if task_type == "solubility":
            smiles_col = "smiles"
            target_col = "solubility_logS"
            df = df.dropna(subset=[smiles_col, target_col])

        elif task_type == "toxicity":
            smiles_col = "smiles"
            target_col = "toxicity_score"
            df = df.dropna(subset=[smiles_col, target_col])

        elif task_type == "bioactivity":
            smiles_col = "smiles"
            target_col = "bioactivity_score"
            df = df.dropna(subset=[smiles_col, target_col])

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        smiles_list = df[smiles_col].tolist()
        targets = df[target_col].tolist()

        task_info = {
            "task_type": task_type,
            "num_samples": len(df),
            "target_range": [float(min(targets)), float(max(targets))],
            "target_mean": float(np.mean(targets)),
            "target_std": float(np.std(targets))
        }

        logger.info(f"Loaded {len(smiles_list)} samples for {task_type}")
        logger.info(f"Target range: {task_info['target_range']}")

        return smiles_list, targets, task_info

    def train_model(self, task_type: str, file_path: str, epochs: int = 10,
                   batch_size: int = 8, learning_rate: float = 2e-5) -> Dict:
        """Train task-specific ChemBERTa model"""

        logger.info(f"Starting training for {task_type} task")

        # Load dataset
        smiles_list, targets, task_info = self.load_dataset(file_path, task_type)
        self.task_info[task_type] = task_info

        if len(smiles_list) < 10:
            logger.warning(f"Very small dataset ({len(smiles_list)} samples) for {task_type}")
            logger.warning("Results may not be reliable")

        # Split data
        if len(smiles_list) >= 10:
            train_smiles, val_smiles, train_targets, val_targets = train_test_split(
                smiles_list, targets, test_size=0.2, random_state=42
            )
        else:
            # Use all data for training if dataset is too small
            train_smiles, val_smiles = smiles_list, smiles_list[:2] if len(smiles_list) >= 2 else smiles_list
            train_targets, val_targets = targets, targets[:2] if len(targets) >= 2 else targets

        # Create datasets
        train_dataset = ChemBERTaDataset(train_smiles, train_targets, self.tokenizer)
        val_dataset = ChemBERTaDataset(val_smiles, val_targets, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        model = ChemBERTaRegressor()
        model.to(self.device)

        # Setup training
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        training_history = []
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_true = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets'].to(self.device).unsqueeze(1)

                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    val_predictions.extend(outputs.cpu().numpy().flatten())
                    val_true.extend(targets.cpu().numpy().flatten())

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            # Calculate metrics
            if len(val_predictions) > 1:
                r2 = r2_score(val_true, val_predictions)
                mae = mean_absolute_error(val_true, val_predictions)
            else:
                r2, mae = 0.0, avg_val_loss

            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'r2_score': r2,
                'mae': mae
            })

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.models[task_type] = model.state_dict()

        # Final model evaluation
        final_metrics = {
            'task_type': task_type,
            'best_val_loss': best_val_loss,
            'final_r2': r2,
            'final_mae': mae,
            'training_samples': len(train_smiles),
            'validation_samples': len(val_smiles),
            'epochs_trained': epochs,
            'training_history': training_history
        }

        logger.info(f"Training completed for {task_type}")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Final R² score: {r2:.4f}")

        return final_metrics

    def save_models(self, save_dir: str = "models/task_specific_chemberta"):
        """Save all trained models and metadata"""

        os.makedirs(save_dir, exist_ok=True)

        for task_type, model_state in self.models.items():
            # Save model
            model_path = os.path.join(save_dir, f"chemberta_{task_type}.pt")
            torch.save(model_state, model_path)
            logger.info(f"Saved {task_type} model to {model_path}")

            # Save task info
            info_path = os.path.join(save_dir, f"chemberta_{task_type}_info.json")
            with open(info_path, 'w') as f:
                json.dump(self.task_info[task_type], f, indent=2)

        # Save tokenizer
        tokenizer_path = os.path.join(save_dir, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)

        logger.info(f"All models saved to {save_dir}")

    def predict_with_model(self, task_type: str, smiles: str) -> float:
        """Make prediction with specific task model"""

        if task_type not in self.models:
            raise ValueError(f"No trained model available for task: {task_type}")

        # Load model
        model = ChemBERTaRegressor()
        model.load_state_dict(self.models[task_type])
        model.eval()
        model.to(self.device)

        # Tokenize input
        encoding = self.tokenizer(
            smiles,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Make prediction
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            prediction = output.item()

        return prediction

def main():
    """Train task-specific ChemBERTa models"""

    print("TASK-SPECIFIC CHEMBERTA TRAINING PIPELINE")
    print("=" * 60)

    # Initialize trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = TaskSpecificChemBERTaTrainer(device=device)

    print(f"Using device: {device}")
    print()

    # Define training tasks
    training_tasks = [
        {
            "task_type": "solubility",
            "file_path": "data/processed/esol_solubility.csv",
            "epochs": 15,
            "description": "Aqueous solubility prediction (logS)"
        },
        {
            "task_type": "toxicity",
            "file_path": "data/processed/toxicity_dataset.csv",
            "epochs": 20,
            "description": "Toxicity score prediction"
        },
        {
            "task_type": "bioactivity",
            "file_path": "data/processed/bioactivity_dataset.csv",
            "epochs": 20,
            "description": "Bioactivity score prediction"
        }
    ]

    all_results = {}

    # Train models for each task
    for task in training_tasks:
        print(f"Training {task['task_type']} model...")
        print(f"Description: {task['description']}")
        print(f"Dataset: {task['file_path']}")
        print("-" * 40)

        try:
            results = trainer.train_model(
                task_type=task['task_type'],
                file_path=task['file_path'],
                epochs=task['epochs'],
                batch_size=4,  # Small batch size for small datasets
                learning_rate=2e-5
            )
            all_results[task['task_type']] = results
            print(f"✓ {task['task_type']} training completed")

        except Exception as e:
            print(f"✗ {task['task_type']} training failed: {e}")
            logger.error(f"Training failed for {task['task_type']}: {e}")

        print()

    # Save all models
    print("Saving trained models...")
    trainer.save_models()

    # Test predictions
    print("\nTesting trained models...")
    test_molecules = [
        ("CCO", "ethanol"),
        ("c1ccccc1", "benzene"),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin")
    ]

    for smiles, name in test_molecules:
        print(f"\nPredictions for {name} ({smiles}):")
        for task_type in trainer.models.keys():
            try:
                prediction = trainer.predict_with_model(task_type, smiles)
                print(f"  {task_type}: {prediction:.3f}")
            except Exception as e:
                print(f"  {task_type}: Error - {e}")

    # Summary
    print(f"\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for task_type, results in all_results.items():
        print(f"{task_type.upper()} MODEL:")
        print(f"  Samples: {results['training_samples']} train, {results['validation_samples']} val")
        print(f"  Best validation loss: {results['best_val_loss']:.4f}")
        print(f"  Final R² score: {results['final_r2']:.4f}")
        print(f"  Final MAE: {results['final_mae']:.4f}")
        print()

    print("Task-specific ChemBERTa models ready!")
    print("Next: Update agent to use specialized models based on query type.")

if __name__ == "__main__":
    main()