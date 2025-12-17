#!/usr/bin/env python3
"""
Vision Transformer Training Pipeline for Cattle and Buffalo Recognition
Optimized for Kaggle P100 GPU with 16GB VRAM
Target: 92-95% individual model accuracy
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

# Metrics and utilities
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LivestockDataset(Dataset):
    """Custom dataset for livestock images with advanced augmentation"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 class_names: List[str], transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        import cv2
        
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                image = self.transform(image)
        
        return image, label

class ViTTrainingPipeline:
    """Vision Transformer training pipeline with memory optimization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler()  # For mixed precision training
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def create_transforms(self) -> Tuple[A.Compose, A.Compose]:
        """Create training and validation transforms with advanced augmentation"""
        
        # Training transforms with aggressive augmentation
        train_transform = A.Compose([
            A.Resize(self.config['image_size'], self.config['image_size']),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.7),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.1),
                A.MotionBlur(blur_limit=3, p=0.1),
            ], p=0.1),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            ], p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Validation transforms (minimal)
        val_transform = A.Compose([
            A.Resize(self.config['image_size'], self.config['image_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        return train_transform, val_transform
    
    def load_dataset(self, data_dir: str) -> Tuple[DataLoader, DataLoader, List[str]]:
        """Load and prepare dataset with memory-efficient data loading"""
        
        logger.info("Loading dataset...")
        
        # Get class names from directory structure
        train_dir = Path(data_dir) / 'train'
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        num_classes = len(class_names)
        
        logger.info(f"Found {num_classes} classes: {class_names}")
        
        # Create class to index mapping
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        
        # Collect all image paths and labels
        train_paths, train_labels = [], []
        val_paths, val_labels = [], []
        
        for class_name in class_names:
            class_idx = class_to_idx[class_name]
            
            # Training images
            train_class_dir = train_dir / class_name
            if train_class_dir.exists():
                for img_path in train_class_dir.glob('*.jpg'):
                    train_paths.append(str(img_path))
                    train_labels.append(class_idx)
            
            # Validation images
            val_class_dir = Path(data_dir) / 'validation' / class_name
            if val_class_dir.exists():
                for img_path in val_class_dir.glob('*.jpg'):
                    val_paths.append(str(img_path))
                    val_labels.append(class_idx)
        
        logger.info(f"Training samples: {len(train_paths)}")
        logger.info(f"Validation samples: {len(val_paths)}")
        
        # Create transforms
        train_transform, val_transform = self.create_transforms()
        
        # Create datasets
        train_dataset = LivestockDataset(train_paths, train_labels, class_names, 
                                       train_transform, is_training=True)
        val_dataset = LivestockDataset(val_paths, val_labels, class_names, 
                                     val_transform, is_training=False)
        
        # Create data loaders with memory optimization
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'] * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
        
        return train_loader, val_loader, class_names
    
    def create_model(self, num_classes: int) -> nn.Module:
        """Create Vision Transformer model with transfer learning"""
        
        logger.info("Creating ViT-Base model...")
        
        # Load pre-trained ViT-Base model
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=num_classes,
            drop_rate=self.config['dropout_rate'],
            drop_path_rate=self.config['drop_path_rate']
        )
        
        # Freeze early layers for transfer learning
        if self.config['freeze_backbone']:
            for name, param in model.named_parameters():
                if 'head' not in name and 'norm' not in name:
                    param.requires_grad = False
        
        # Move to device
        model = model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
    
    def create_optimizer_and_scheduler(self, model: nn.Module, train_loader: DataLoader):
        """Create optimizer and learning rate scheduler"""
        
        # Separate parameters for different learning rates
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'head' in name or 'norm' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
        
        # Create optimizer with different learning rates
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config['learning_rate'] * 0.1},
            {'params': head_params, 'lr': self.config['learning_rate']}
        ], weight_decay=self.config['weight_decay'])
        
        # Create scheduler
        total_steps = len(train_loader) * self.config['epochs']
        warmup_steps = int(total_steps * self.config['warmup_ratio'])
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.config['learning_rate'] * 0.1, self.config['learning_rate']],
            total_steps=total_steps,
            pct_start=self.config['warmup_ratio'],
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=10000
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch with mixed precision and gradient accumulation"""
        
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        # Gradient accumulation
        accumulation_steps = self.config['gradient_accumulation_steps']
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss = loss / accumulation_steps  # Scale loss for gradient accumulation
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['max_grad_norm'])
                
                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
                
                # Scheduler step
                scheduler.step()
            
            # Statistics
            total_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            accuracy = 100.0 * correct / total
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Acc': f'{accuracy:.2f}%',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Memory cleanup
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def validate(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model performance"""
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                with autocast():
                    outputs = model(images)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100.0 * correct / total
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler: optim.lr_scheduler._LRScheduler, epoch: int, 
                       val_accuracy: float, is_best: bool = False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_accuracy': val_accuracy,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with accuracy: {val_accuracy:.2f}%")
    
    def train(self, data_dir: str):
        """Main training loop"""
        
        logger.info("Starting ViT training pipeline...")
        
        # Load dataset
        train_loader, val_loader, class_names = self.load_dataset(data_dir)
        num_classes = len(class_names)
        
        # Save class names
        with open(self.output_dir / 'class_names.json', 'w') as f:
            json.dump(class_names, f, indent=2)
        
        # Create model
        self.model = self.create_model(num_classes)
        
        # Create optimizer and scheduler
        self.optimizer, self.scheduler = self.create_optimizer_and_scheduler(self.model, train_loader)
        
        # Training loop
        best_val_accuracy = 0.0
        train_history = []
        val_history = []
        
        for epoch in range(self.config['epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Train
            train_metrics = self.train_epoch(self.model, train_loader, self.optimizer, 
                                           self.scheduler, epoch)
            
            # Validate
            val_metrics = self.validate(self.model, val_loader)
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # Save history
            train_history.append(train_metrics)
            val_history.append(val_metrics)
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > best_val_accuracy
            if is_best:
                best_val_accuracy = val_metrics['accuracy']
            
            self.save_checkpoint(self.model, self.optimizer, self.scheduler, 
                               epoch, val_metrics['accuracy'], is_best)
            
            # Early stopping
            if self.config['early_stopping_patience'] > 0:
                if len(val_history) >= self.config['early_stopping_patience']:
                    recent_accuracies = [h['accuracy'] for h in val_history[-self.config['early_stopping_patience']:]]
                    if all(acc <= best_val_accuracy - 1.0 for acc in recent_accuracies[:-1]):
                        logger.info("Early stopping triggered")
                        break
        
        # Save training history
        history = {
            'train': train_history,
            'validation': val_history
        }
        
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        # Generate final report
        self.generate_training_report(val_history[-1], class_names)
        
        logger.info(f"Training completed! Best validation accuracy: {best_val_accuracy:.2f}%")
        
        return best_val_accuracy
    
    def generate_training_report(self, final_val_metrics: Dict, class_names: List[str]):
        """Generate comprehensive training report"""
        
        logger.info("Generating training report...")
        
        predictions = final_val_metrics['predictions']
        labels = final_val_metrics['labels']
        
        # Classification report
        report = classification_report(labels, predictions, target_names=class_names, 
                                     output_dict=True)
        
        # Save classification report
        with open(self.output_dir / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - ViT Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Per-class accuracy
        per_class_acc = {}
        for i, class_name in enumerate(class_names):
            class_mask = np.array(labels) == i
            if class_mask.sum() > 0:
                class_predictions = np.array(predictions)[class_mask]
                class_accuracy = (class_predictions == i).mean() * 100
                per_class_acc[class_name] = class_accuracy
        
        # Save per-class accuracy
        with open(self.output_dir / 'per_class_accuracy.json', 'w') as f:
            json.dump(per_class_acc, f, indent=2)
        
        # Print summary
        overall_accuracy = final_val_metrics['accuracy']
        logger.info(f"\n=== ViT Training Summary ===")
        logger.info(f"Overall Accuracy: {overall_accuracy:.2f}%")
        logger.info(f"Target Accuracy: 92-95%")
        logger.info(f"Target Met: {'✓' if overall_accuracy >= 92.0 else '✗'}")
        
        # Top performing classes
        sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"\nTop 5 performing classes:")
        for class_name, acc in sorted_classes[:5]:
            logger.info(f"  {class_name}: {acc:.2f}%")
        
        # Lowest performing classes
        logger.info(f"\nLowest 5 performing classes:")
        for class_name, acc in sorted_classes[-5:]:
            logger.info(f"  {class_name}: {acc:.2f}%")

def main():
    """Main training function"""
    
    # Training configuration optimized for Kaggle P100 GPU
    config = {
        # Model configuration
        'model_name': 'vit_base_patch16_224',
        'image_size': 224,
        'dropout_rate': 0.1,
        'drop_path_rate': 0.1,
        'freeze_backbone': True,  # Start with frozen backbone
        
        # Training configuration
        'batch_size': 20,  # Optimized for P100 16GB VRAM
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        
        # Memory optimization
        'gradient_accumulation_steps': 2,  # Effective batch size: 40
        'num_workers': 4,
        
        # Early stopping
        'early_stopping_patience': 7,
        
        # Output
        'output_dir': '/kaggle/working/vit_models',
        
        # Target accuracy
        'target_accuracy': 92.0
    }
    
    # Data directory (adjust for Kaggle environment)
    data_dir = '/kaggle/input/livestock-dataset'
    
    # Create training pipeline
    pipeline = ViTTrainingPipeline(config)
    
    # Start training
    final_accuracy = pipeline.train(data_dir)
    
    # Check if target met
    if final_accuracy >= config['target_accuracy']:
        logger.info(f"🎉 Target accuracy achieved: {final_accuracy:.2f}% >= {config['target_accuracy']}%")
    else:
        logger.warning(f"⚠️ Target accuracy not met: {final_accuracy:.2f}% < {config['target_accuracy']}%")
    
    return final_accuracy

if __name__ == "__main__":
    main()