#!/usr/bin/env python3
"""
Local Fine-tuning Pipeline for RTX 3050
Memory-optimized training with batch size 4-12
Gradient checkpointing and 8-bit optimizer support
Fine-tune Kaggle-trained models on final dataset distribution
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import warnings
import gc
import psutil
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

# Import model architectures
import timm
from transformers import ViTForImageClassification, ViTConfig
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RTX3050MemoryManager:
    """Memory management utilities for RTX 3050 (6GB VRAM)"""
    
    def __init__(self):
        self.max_memory_gb = 5.5  # Leave 0.5GB buffer
        self.current_memory_usage = 0
        
    def get_gpu_memory_usage(self):
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0
    
    def cleanup_memory(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        for _ in range(3):
            gc.collect()
    
    def get_optimal_batch_size(self, model_size_mb: int) -> int:
        """Calculate optimal batch size based on model size and available memory"""
        available_memory_gb = self.max_memory_gb - self.get_gpu_memory_usage()
        
        # Estimate memory per sample (rough calculation)
        memory_per_sample_mb = 50  # Conservative estimate for 224x224 images
        model_memory_gb = model_size_mb / 1024
        
        # Calculate batch size
        available_for_batch = (available_memory_gb - model_memory_gb) * 1024
        optimal_batch_size = max(1, int(available_for_batch / memory_per_sample_mb))
        
        # Clamp to reasonable range for RTX 3050
        return min(max(optimal_batch_size, 4), 12)
    
    def monitor_memory(self, stage: str):
        """Monitor and log memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

class LocalFineTuningDataset(Dataset):
    """Custom dataset for local fine-tuning with memory optimization"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None, max_samples_per_class: int = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        
        # Load dataset
        self.dataset = ImageFolder(
            root=self.data_dir / split,
            transform=transform
        )
        
        # Limit samples per class if specified (for memory management)
        if max_samples_per_class:
            self.dataset = self._limit_samples_per_class()
        
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        
        logger.info(f"Loaded {len(self.dataset)} samples for {split} split")
        logger.info(f"Classes: {len(self.classes)}")
    
    def _limit_samples_per_class(self):
        """Limit samples per class for memory management"""
        from collections import defaultdict
        import random
        
        # Group samples by class
        class_samples = defaultdict(list)
        for idx, (path, class_idx) in enumerate(self.dataset.samples):
            class_samples[class_idx].append((path, class_idx))
        
        # Limit samples per class
        limited_samples = []
        for class_idx, samples in class_samples.items():
            if len(samples) > self.max_samples_per_class:
                samples = random.sample(samples, self.max_samples_per_class)
            limited_samples.extend(samples)
        
        # Create new dataset with limited samples
        self.dataset.samples = limited_samples
        self.dataset.targets = [s[1] for s in limited_samples]
        
        return self.dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class LocalFineTuningPipeline:
    """Local fine-tuning pipeline optimized for RTX 3050"""
    
    def __init__(self, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_manager = RTX3050MemoryManager()
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.get_default_config()
        
        # Training state
        self.models = {}
        self.training_results = {}
        
        logger.info(f"Local Fine-tuning Pipeline initialized on {self.device}")
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    def get_default_config(self):
        """Get default configuration for RTX 3050 fine-tuning"""
        return {
            "hardware": {
                "gpu_memory_limit_gb": 5.5,
                "max_batch_size": 12,
                "min_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "use_gradient_checkpointing": True,
                "use_8bit_optimizer": True,
                "mixed_precision": True
            },
            "training": {
                "fine_tune_epochs": 10,
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "max_grad_norm": 1.0,
                "early_stopping_patience": 5,
                "save_best_only": True
            },
            "data": {
                "image_size": 224,
                "max_samples_per_class": 500,  # Limit for memory
                "num_workers": 2,  # Reduced for memory
                "pin_memory": False  # Disabled for RTX 3050
            },
            "models": {
                "vit": {
                    "kaggle_model_path": "kaggle_models/best_vit_model.pth",
                    "freeze_backbone_epochs": 3,
                    "unfreeze_lr_multiplier": 0.1
                },
                "yolov8_cbam": {
                    "kaggle_model_path": "kaggle_models/best_yolov8_cbam.pt",
                    "freeze_backbone_epochs": 2,
                    "unfreeze_lr_multiplier": 0.1
                },
                "efficientnetv2": {
                    "kaggle_model_path": "kaggle_models/best_efficientnetv2.pth",
                    "freeze_backbone_epochs": 2,
                    "unfreeze_lr_multiplier": 0.1
                }
            },
            "output": {
                "save_dir": "local_finetuned_models",
                "save_intermediate": True,
                "export_onnx": True
            }
        }
    
    def setup_8bit_optimizer(self, model, learning_rate: float):
        """Setup 8-bit optimizer for memory efficiency"""
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=learning_rate,
                weight_decay=self.config['training']['weight_decay'],
                betas=(0.9, 0.999)
            )
            logger.info("Using 8-bit AdamW optimizer")
            return optimizer
        except ImportError:
            logger.warning("bitsandbytes not available, using standard AdamW")
            return optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=self.config['training']['weight_decay']
            )
    
    def get_data_transforms(self, split: str):
        """Get data transforms for training/validation"""
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def load_kaggle_model(self, model_name: str, model_path: str, num_classes: int):
        """Load pre-trained model from Kaggle training"""
        logger.info(f"Loading Kaggle-trained {model_name} model from {model_path}")
        
        if model_name == 'vit':
            # Load ViT model
            model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=num_classes
            )
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded ViT checkpoint from {model_path}")
            else:
                logger.warning(f"Kaggle model not found at {model_path}, using pretrained ImageNet weights")
                model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        
        elif model_name == 'efficientnetv2':
            # Load EfficientNetV2 model
            model = timm.create_model(
                'efficientnetv2_s',
                pretrained=False,
                num_classes=num_classes
            )
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded EfficientNetV2 checkpoint from {model_path}")
            else:
                logger.warning(f"Kaggle model not found at {model_path}, using pretrained ImageNet weights")
                model = timm.create_model('efficientnetv2_s', pretrained=True, num_classes=num_classes)
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return model
    
    def freeze_backbone(self, model, model_name: str):
        """Freeze backbone layers for initial fine-tuning"""
        if model_name == 'vit':
            # Freeze patch embedding and transformer blocks
            for name, param in model.named_parameters():
                if 'head' not in name:  # Keep head unfrozen
                    param.requires_grad = False
        elif model_name == 'efficientnetv2':
            # Freeze feature extraction layers
            for name, param in model.named_parameters():
                if 'classifier' not in name:  # Keep classifier unfrozen
                    param.requires_grad = False
        
        # Count frozen/unfrozen parameters
        frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
        total_params = sum(1 for p in model.parameters())
        logger.info(f"Frozen {frozen_params}/{total_params} parameters")
    
    def unfreeze_backbone(self, model, model_name: str):
        """Unfreeze backbone layers for full fine-tuning"""
        for param in model.parameters():
            param.requires_grad = True
        
        logger.info("All parameters unfrozen for full fine-tuning")
    
    def fine_tune_model(self, model_name: str, data_dir: str, num_classes: int):
        """Fine-tune a single model on local dataset"""
        logger.info(f"=" * 60)
        logger.info(f"Fine-tuning {model_name.upper()} model")
        logger.info(f"=" * 60)
        
        # Memory cleanup before starting
        self.memory_manager.cleanup_memory()
        self.memory_manager.monitor_memory("Pre-training")
        
        try:
            # Load Kaggle-trained model
            model_config = self.config['models'][model_name]
            model = self.load_kaggle_model(
                model_name, 
                model_config['kaggle_model_path'], 
                num_classes
            )
            
            # Enable gradient checkpointing for memory efficiency
            if self.config['hardware']['use_gradient_checkpointing'] and hasattr(model, 'set_grad_checkpointing'):
                model.set_grad_checkpointing(True)
                logger.info("Gradient checkpointing enabled")
            
            model = model.to(self.device)
            
            # Calculate optimal batch size
            model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / 1024**2  # Rough estimate
            batch_size = self.memory_manager.get_optimal_batch_size(model_size_mb)
            logger.info(f"Using batch size: {batch_size}")
            
            # Setup data loaders
            train_transform = self.get_data_transforms('train')
            val_transform = self.get_data_transforms('val')
            
            train_dataset = LocalFineTuningDataset(
                data_dir, 'train', train_transform,
                max_samples_per_class=self.config['data']['max_samples_per_class']
            )
            val_dataset = LocalFineTuningDataset(
                data_dir, 'validation', val_transform,
                max_samples_per_class=self.config['data']['max_samples_per_class'] // 2
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory'],
                drop_last=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory']
            )
            
            # Setup training components
            criterion = nn.CrossEntropyLoss()
            
            # Phase 1: Frozen backbone training
            logger.info("Phase 1: Training with frozen backbone")
            self.freeze_backbone(model, model_name)
            
            optimizer = self.setup_8bit_optimizer(model, self.config['training']['learning_rate'])
            scaler = GradScaler() if self.config['hardware']['mixed_precision'] else None
            
            best_accuracy = 0.0
            patience_counter = 0
            
            freeze_epochs = model_config['freeze_backbone_epochs']
            
            for epoch in range(freeze_epochs):
                # Training
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, criterion, optimizer, scaler, epoch, "Frozen"
                )
                
                # Validation
                val_loss, val_acc = self.validate_epoch(model, val_loader, criterion, epoch, "Frozen")
                
                # Check for improvement
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    patience_counter = 0
                    # Save best model
                    self.save_model(model, model_name, epoch, val_acc, "frozen")
                else:
                    patience_counter += 1
                
                # Memory monitoring
                self.memory_manager.monitor_memory(f"Epoch {epoch+1}")
                
                # Early stopping
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Phase 2: Full fine-tuning
            logger.info("Phase 2: Full fine-tuning with unfrozen backbone")
            self.unfreeze_backbone(model, model_name)
            
            # Reduce learning rate for unfrozen training
            unfrozen_lr = self.config['training']['learning_rate'] * model_config['unfreeze_lr_multiplier']
            optimizer = self.setup_8bit_optimizer(model, unfrozen_lr)
            
            remaining_epochs = self.config['training']['fine_tune_epochs'] - freeze_epochs
            patience_counter = 0
            
            for epoch in range(remaining_epochs):
                # Training
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, criterion, optimizer, scaler, 
                    freeze_epochs + epoch, "Unfrozen"
                )
                
                # Validation
                val_loss, val_acc = self.validate_epoch(
                    model, val_loader, criterion, freeze_epochs + epoch, "Unfrozen"
                )
                
                # Check for improvement
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    patience_counter = 0
                    # Save best model
                    self.save_model(model, model_name, freeze_epochs + epoch, val_acc, "unfrozen")
                else:
                    patience_counter += 1
                
                # Memory monitoring
                self.memory_manager.monitor_memory(f"Unfrozen Epoch {epoch+1}")
                
                # Early stopping
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    logger.info(f"Early stopping triggered after {epoch+1} unfrozen epochs")
                    break
            
            # Store results
            self.training_results[model_name] = {
                'best_accuracy': best_accuracy,
                'total_epochs': freeze_epochs + epoch + 1,
                'frozen_epochs': freeze_epochs,
                'unfrozen_epochs': epoch + 1,
                'model_path': f"{self.config['output']['save_dir']}/{model_name}_best.pth"
            }
            
            logger.info(f"{model_name.upper()} fine-tuning complete: {best_accuracy:.2f}% accuracy")
            
            # Cleanup
            del model, optimizer, train_loader, val_loader
            self.memory_manager.cleanup_memory()
            
            return best_accuracy
            
        except Exception as e:
            logger.error(f"Fine-tuning {model_name} failed: {str(e)}")
            self.memory_manager.cleanup_memory()
            return 0.0
    
    def train_epoch(self, model, train_loader, criterion, optimizer, scaler, epoch, phase):
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        gradient_accumulation_steps = self.config['hardware']['gradient_accumulation_steps']
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            if scaler:  # Mixed precision training
                with autocast():
                    output = model(data)
                    loss = criterion(output, target) / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['training']['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:  # Standard training
                output = model(data)
                loss = criterion(output, target) / gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['training']['max_grad_norm'])
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item() * gradient_accumulation_steps
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f'[{phase}] Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        logger.info(f'[{phase}] Epoch {epoch+1} Training - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion, epoch, phase):
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        logger.info(f'[{phase}] Epoch {epoch+1} Validation - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        return epoch_loss, epoch_acc
    
    def save_model(self, model, model_name, epoch, accuracy, phase):
        """Save model checkpoint"""
        save_dir = Path(self.config['output']['save_dir'])
        save_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy,
            'phase': phase,
            'config': self.config
        }
        
        save_path = save_dir / f"{model_name}_best.pth"
        torch.save(checkpoint, save_path)
        
        logger.info(f"Model saved: {save_path} (Accuracy: {accuracy:.2f}%)")
    
    def run_local_finetuning(self, data_dir: str, num_classes: int):
        """Run complete local fine-tuning pipeline"""
        logger.info("🚀 Starting Local Fine-tuning Pipeline (RTX 3050)")
        logger.info("Memory-optimized training with gradient checkpointing")
        logger.info("=" * 80)
        
        # Create output directory
        Path(self.config['output']['save_dir']).mkdir(exist_ok=True)
        
        # Fine-tune each model
        models_to_finetune = ['vit', 'efficientnetv2']  # Skip YOLOv8 for now
        
        for model_name in models_to_finetune:
            logger.info(f"\n📊 Fine-tuning Model: {model_name.upper()}")
            
            accuracy = self.fine_tune_model(model_name, data_dir, num_classes)
            
            if accuracy > 0:
                logger.info(f"✅ {model_name} fine-tuning successful: {accuracy:.2f}%")
            else:
                logger.error(f"❌ {model_name} fine-tuning failed")
        
        # Generate report
        report = self.generate_finetuning_report()
        
        logger.info("\n" + "=" * 80)
        logger.info("🎯 LOCAL FINE-TUNING COMPLETE")
        logger.info("=" * 80)
        
        return report
    
    def generate_finetuning_report(self):
        """Generate fine-tuning report"""
        report = {
            'local_finetuning_summary': {
                'hardware': 'RTX 3050 (6GB VRAM)',
                'optimization': 'Memory-optimized with gradient checkpointing',
                'models_finetuned': len(self.training_results),
                'total_training_time': None,  # Would be tracked in real implementation
                'memory_usage': 'Optimized for 6GB VRAM constraint'
            },
            'model_results': self.training_results,
            'configuration': self.config,
            'next_steps': [
                'Implement ensemble coordination service',
                'Convert models to TensorFlow Lite',
                'Integrate with Android application',
                'Validate ensemble accuracy targets'
            ]
        }
        
        # Save report
        report_path = Path(self.config['output']['save_dir']) / 'local_finetuning_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    """Main function for local fine-tuning"""
    
    print("🚀 Local Fine-tuning Pipeline for RTX 3050")
    print("Memory-optimized training with gradient checkpointing")
    print("=" * 80)
    
    # Configuration
    data_dir = "dataset"  # Update with actual dataset path
    num_classes = 23  # Update with actual number of classes
    
    # Create pipeline
    pipeline = LocalFineTuningPipeline()
    
    # Run fine-tuning
    report = pipeline.run_local_finetuning(data_dir, num_classes)
    
    # Print summary
    print(f"\n✅ Local Fine-tuning Complete!")
    print(f"📊 Models Fine-tuned: {report['local_finetuning_summary']['models_finetuned']}")
    print(f"🎯 Hardware: {report['local_finetuning_summary']['hardware']}")
    
    return report

if __name__ == "__main__":
    main()