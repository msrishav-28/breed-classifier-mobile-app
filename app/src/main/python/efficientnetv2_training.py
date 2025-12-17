#!/usr/bin/env python3
"""
EfficientNetV2 Training Pipeline for Cattle and Buffalo Recognition
Optimized for ensemble diversity and efficiency
Target: 88-91% individual model accuracy
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Install required packages for Kaggle
def install_requirements():
    """Install required packages in Kaggle environment"""
    import subprocess
    
    packages = [
        'timm>=0.9.0',
        'efficientnet-pytorch>=0.7.0',
        'albumentations>=1.3.0',
        'opencv-python>=4.7.0'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
        except subprocess.CalledProcessError:
            print(f"Warning: Could not install {package}")

# Install requirements
install_requirements()

# Import after installation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EfficientNetDataset(Dataset):
    """Custom dataset for EfficientNetV2 training"""
    
    def __init__(self, image_paths, labels, class_names, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        
        if image is None:
            # Fallback to black image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
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

class EfficientNetV2Trainer:
    """EfficientNetV2 training pipeline for livestock recognition"""
    
    def __init__(self, config_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler()
        
        # Kaggle paths
        self.input_dir = Path('/kaggle/input')
        self.working_dir = Path('/kaggle/working')
        self.working_dir.mkdir(exist_ok=True)
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.get_default_config()
        
        logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def get_default_config(self):
        """Get default training configuration for EfficientNetV2"""
        return {
            "model": {
                "name": "efficientnetv2_s",  # Small variant for efficiency
                "pretrained": True,
                "dropout_rate": 0.2,
                "drop_path_rate": 0.2,
                "input_size": 384,  # EfficientNetV2 optimal size
                "freeze_backbone_epochs": 3
            },
            "training": {
                "batch_size": 12,  # Optimized for P100 16GB with 384px images
                "val_batch_size": 24,
                "epochs": 40,
                "learning_rate": 1e-3,
                "backbone_lr": 1e-4,
                "weight_decay": 1e-5,
                "warmup_epochs": 3,
                "gradient_accumulation_steps": 2,
                "max_grad_norm": 1.0,
                "early_stopping_patience": 8,
                "label_smoothing": 0.1
            },
            "optimization": {
                "mixed_precision": True,
                "optimizer": "adamw",
                "scheduler": "cosine_restarts",
                "scheduler_params": {
                    "T_0": 10,
                    "T_mult": 2,
                    "eta_min": 1e-6
                }
            },
            "data": {
                "num_workers": 2,
                "pin_memory": True,
                "validation_split": 0.2,
                "augmentation_strength": "strong"
            },
            "targets": {
                "accuracy_target": 88.0,
                "accuracy_range": [88.0, 91.0]
            }
        }
    
    def find_dataset_path(self):
        """Find the livestock dataset in Kaggle input"""
        possible_paths = [
            '/kaggle/input/livestock-dataset',
            '/kaggle/input/cattle-buffalo-dataset',
            '/kaggle/input/indian-livestock-breeds',
            '/kaggle/input/livestock-breeds'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found dataset at: {path}")
                return path
        
        # Look for any directory with train/val structure
        for item in self.input_dir.iterdir():
            if item.is_dir():
                train_path = item / 'train'
                if train_path.exists():
                    logger.info(f"Found dataset with train folder at: {item}")
                    return str(item)
        
        raise FileNotFoundError("No suitable dataset found in /kaggle/input")
    
    def create_transforms(self):
        """Create optimized transforms for EfficientNetV2"""
        
        input_size = self.config['model']['input_size']
        
        # Training transforms with strong augmentation for ensemble diversity
        train_transform = A.Compose([
            A.Resize(input_size, input_size),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.3, 
                rotate_limit=20, 
                p=0.8
            ),
            A.OneOf([
                A.OpticalDistortion(p=0.4),
                A.GridDistortion(p=0.2),
                A.PiecewiseAffine(p=0.4),
            ], p=0.4),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            ], p=0.6),
            A.HueSaturationValue(
                hue_shift_limit=25, 
                sat_shift_limit=35, 
                val_shift_limit=25, 
                p=0.6
            ),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.2),
            ], p=0.3),
            A.CoarseDropout(
                max_holes=12, 
                max_height=32, 
                max_width=32, 
                p=0.4
            ),
            A.Cutout(
                num_holes=8, 
                max_h_size=32, 
                max_w_size=32, 
                p=0.3
            ),
            # EfficientNet-specific normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
        
        # Validation transforms (minimal)
        val_transform = A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
        
        return train_transform, val_transform
    
    def load_dataset(self, data_dir):
        """Load and prepare dataset"""
        
        logger.info("Loading dataset for EfficientNetV2...")
        
        data_path = Path(data_dir)
        
        # Find class names
        train_dir = data_path / 'train'
        if not train_dir.exists():
            train_dir = data_path / 'training'
        
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        
        logger.info(f"Found {len(class_names)} classes: {class_names}")
        
        # Load training data
        train_paths, train_labels = [], []
        for class_name in class_names:
            class_dir = train_dir / class_name
            if class_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for img_path in class_dir.glob(ext):
                        train_paths.append(str(img_path))
                        train_labels.append(class_to_idx[class_name])
        
        # Load validation data
        val_paths, val_labels = [], []
        val_dir = data_path / 'validation'
        if not val_dir.exists():
            val_dir = data_path / 'val'
        
        if val_dir.exists():
            for class_name in class_names:
                class_dir = val_dir / class_name
                if class_dir.exists():
                    for ext in ['*.jpg', '*.jpeg', '*.png']:
                        for img_path in class_dir.glob(ext):
                            val_paths.append(str(img_path))
                            val_labels.append(class_to_idx[class_name])
        else:
            # Split training data if no validation set
            from sklearn.model_selection import train_test_split
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_paths, train_labels, 
                test_size=self.config['data']['validation_split'], 
                stratify=train_labels, 
                random_state=42
            )
        
        logger.info(f"Training samples: {len(train_paths)}")
        logger.info(f"Validation samples: {len(val_paths)}")
        
        return train_paths, train_labels, val_paths, val_labels, class_names
    
    def create_model(self, num_classes):
        """Create EfficientNetV2 model"""
        
        logger.info("Creating EfficientNetV2-Small model...")
        
        # Create model with pre-trained weights
        model = timm.create_model(
            self.config['model']['name'],
            pretrained=self.config['model']['pretrained'],
            num_classes=num_classes,
            drop_rate=self.config['model']['dropout_rate'],
            drop_path_rate=self.config['model']['drop_path_rate']
        )
        
        # Freeze backbone initially for transfer learning
        if self.config['model']['freeze_backbone_epochs'] > 0:
            for name, param in model.named_parameters():
                if 'classifier' not in name and 'head' not in name:
                    param.requires_grad = False
        
        model = model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def create_optimizer_and_scheduler(self, model, train_loader):
        """Create optimizer and learning rate scheduler"""
        
        # Separate parameters for different learning rates
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'classifier' in name or 'head' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
        
        # Create optimizer with different learning rates
        if backbone_params:
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': self.config['training']['backbone_lr']},
                {'params': head_params, 'lr': self.config['training']['learning_rate']}
            ], weight_decay=self.config['training']['weight_decay'])
        else:
            optimizer = optim.AdamW(
                head_params, 
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        
        # Create scheduler
        if self.config['optimization']['scheduler'] == 'cosine_restarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config['optimization']['scheduler_params']['T_0'],
                T_mult=self.config['optimization']['scheduler_params']['T_mult'],
                eta_min=self.config['optimization']['scheduler_params']['eta_min']
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config['training']['epochs'],
                eta_min=1e-6
            )
        
        return optimizer, scheduler
    
    def train_epoch(self, model, train_loader, optimizer, scheduler, epoch):
        """Train for one epoch"""
        
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Label smoothing loss
        criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config['training']['label_smoothing']
        )
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["training"]["epochs"]}')
        
        # Gradient accumulation
        accumulation_steps = self.config['training']['gradient_accumulation_steps']
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config['training']['max_grad_norm']
                )
                
                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
            
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
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Memory cleanup
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Update scheduler
        scheduler.step()
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def validate(self, model, val_loader):
        """Validate model performance"""
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
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
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, val_accuracy, is_best=False):
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
        checkpoint_path = self.working_dir / f'efficientnetv2_checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.working_dir / 'best_efficientnetv2_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best EfficientNetV2 model saved with accuracy: {val_accuracy:.2f}%")
    
    def train_model(self, train_loader, val_loader, class_names):
        """Main training loop"""
        
        logger.info("Starting EfficientNetV2 training...")
        
        # Create model
        model = self.create_model(len(class_names))
        
        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer_and_scheduler(model, train_loader)
        
        # Training loop
        best_val_accuracy = 0.0
        train_history = []
        val_history = []
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            
            # Unfreeze backbone after specified epochs
            if epoch == self.config['model']['freeze_backbone_epochs']:
                logger.info("Unfreezing backbone...")
                for param in model.parameters():
                    param.requires_grad = True
                
                # Update optimizer with new parameters
                optimizer, scheduler = self.create_optimizer_and_scheduler(model, train_loader)
            
            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, scheduler, epoch)
            
            # Validate
            val_metrics = self.validate(model, val_loader)
            
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
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(model, optimizer, scheduler, epoch, val_metrics['accuracy'], is_best)
            
            # Early stopping
            if patience_counter >= self.config['training']['early_stopping_patience']:
                logger.info("Early stopping triggered")
                break
        
        # Save training history
        history = {
            'train': train_history,
            'validation': val_history
        }
        
        with open(self.working_dir / 'efficientnetv2_training_history.json', 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        # Generate final report
        self.generate_training_report(val_history[-1], class_names, best_val_accuracy)
        
        logger.info(f"EfficientNetV2 training completed! Best validation accuracy: {best_val_accuracy:.2f}%")
        
        return best_val_accuracy
    
    def generate_training_report(self, final_val_metrics, class_names, best_accuracy):
        """Generate comprehensive training report"""
        
        logger.info("Generating EfficientNetV2 training report...")
        
        predictions = final_val_metrics['predictions']
        labels = final_val_metrics['labels']
        
        # Classification report
        report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
        
        # Save classification report
        with open(self.working_dir / 'efficientnetv2_classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - EfficientNetV2 Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.working_dir / 'efficientnetv2_confusion_matrix.png', dpi=300, bbox_inches='tight')
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
        with open(self.working_dir / 'efficientnetv2_per_class_accuracy.json', 'w') as f:
            json.dump(per_class_acc, f, indent=2)
        
        # Create final report
        final_report = {
            'model_type': 'EfficientNetV2-Small',
            'best_accuracy': best_accuracy,
            'target_accuracy': self.config['targets']['accuracy_target'],
            'target_met': best_accuracy >= self.config['targets']['accuracy_target'],
            'num_classes': len(class_names),
            'class_names': class_names,
            'per_class_accuracy': per_class_acc,
            'overall_metrics': report['macro avg'],
            'training_config': self.config
        }
        
        # Save final report
        with open(self.working_dir / 'efficientnetv2_final_report.json', 'w') as f:
            json.dump(final_report, f, indent=2)
        
        # Print summary
        logger.info(f"\n=== EfficientNetV2 Training Summary ===")
        logger.info(f"Best Accuracy: {best_accuracy:.2f}%")
        logger.info(f"Target Accuracy: {self.config['targets']['accuracy_target']}%")
        logger.info(f"Target Met: {'✓' if final_report['target_met'] else '✗'}")
        
        # Top performing classes
        sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"\nTop 5 performing classes:")
        for class_name, acc in sorted_classes[:5]:
            logger.info(f"  {class_name}: {acc:.2f}%")
        
        # Lowest performing classes
        logger.info(f"\nLowest 5 performing classes:")
        for class_name, acc in sorted_classes[-5:]:
            logger.info(f"  {class_name}: {acc:.2f}%")
        
        return final_report
    
    def run_training_pipeline(self):
        """Run complete EfficientNetV2 training pipeline"""
        
        try:
            logger.info("Starting EfficientNetV2 training pipeline...")
            
            # Find dataset
            data_path = self.find_dataset_path()
            
            # Load dataset
            train_paths, train_labels, val_paths, val_labels, class_names = self.load_dataset(data_path)
            
            # Create transforms
            train_transform, val_transform = self.create_transforms()
            
            # Create datasets
            train_dataset = EfficientNetDataset(train_paths, train_labels, class_names, train_transform)
            val_dataset = EfficientNetDataset(val_paths, val_labels, class_names, val_transform)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory'],
                drop_last=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['training']['val_batch_size'],
                shuffle=False,
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory']
            )
            
            # Train model
            best_accuracy = self.train_model(train_loader, val_loader, class_names)
            
            # Final summary
            logger.info(f"\n🎯 EfficientNetV2 Training Complete!")
            logger.info(f"Best Accuracy: {best_accuracy:.2f}%")
            logger.info(f"Target Range: 88-91%")
            
            if best_accuracy >= self.config['targets']['accuracy_target']:
                logger.info("🎉 EfficientNetV2 model successfully meets accuracy requirements!")
            else:
                logger.warning("⚠️ EfficientNetV2 model accuracy below target. Consider:")
                logger.warning("  - Longer training")
                logger.warning("  - Different augmentation strategy")
                logger.warning("  - Hyperparameter tuning")
                logger.warning("  - Model size adjustment")
            
            return best_accuracy
            
        except Exception as e:
            logger.error(f"EfficientNetV2 training failed: {str(e)}")
            raise

def main():
    """Main function for Kaggle execution"""
    
    print("🚀 Starting EfficientNetV2 Training for Cattle & Buffalo Recognition")
    print("=" * 70)
    
    # Create trainer
    trainer = EfficientNetV2Trainer()
    
    # Run training pipeline
    accuracy = trainer.run_training_pipeline()
    
    print("=" * 70)
    print(f"✅ EfficientNetV2 Training completed with {accuracy:.2f}% accuracy")
    
    return accuracy

if __name__ == "__main__":
    main()