#!/usr/bin/env python3
"""
Kaggle-optimized Vision Transformer Training Script
For Cattle and Buffalo Recognition - SIH 2025
Optimized for P100 GPU with 16GB VRAM
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

# Install required packages if not available
def install_requirements():
    """Install required packages in Kaggle environment"""
    import subprocess
    
    packages = [
        'timm>=0.9.0',
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
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KaggleLivestockDataset(Dataset):
    """Kaggle-optimized livestock dataset"""
    
    def __init__(self, image_paths, labels, class_names, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            # Fallback to a black image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

class KaggleViTTrainer:
    """Kaggle-optimized ViT trainer"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler()
        
        # Kaggle paths
        self.input_dir = Path('/kaggle/input')
        self.working_dir = Path('/kaggle/working')
        self.working_dir.mkdir(exist_ok=True)
        
        logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
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
        
        # If no specific dataset found, look for any directory with train/val structure
        for item in self.input_dir.iterdir():
            if item.is_dir():
                train_path = item / 'train'
                if train_path.exists():
                    logger.info(f"Found dataset with train folder at: {item}")
                    return str(item)
        
        raise FileNotFoundError("No suitable dataset found in /kaggle/input")
    
    def create_transforms(self):
        """Create optimized transforms for Kaggle training"""
        
        # Training transforms
        train_transform = A.Compose([
            A.Resize(224, 224),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            ], p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Validation transforms
        val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        return train_transform, val_transform
    
    def load_data(self, data_path):
        """Load dataset from Kaggle input"""
        
        data_path = Path(data_path)
        
        # Find class names
        train_dir = data_path / 'train'
        if not train_dir.exists():
            # Try alternative structure
            possible_train_dirs = list(data_path.glob('**/train'))
            if possible_train_dirs:
                train_dir = possible_train_dirs[0]
            else:
                raise FileNotFoundError(f"No train directory found in {data_path}")
        
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        
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
                train_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=42
            )
        
        logger.info(f"Training samples: {len(train_paths)}")
        logger.info(f"Validation samples: {len(val_paths)}")
        
        return train_paths, train_labels, val_paths, val_labels, class_names
    
    def create_model(self, num_classes):
        """Create ViT model optimized for Kaggle"""
        
        logger.info("Creating ViT-Base model...")
        
        # Create model with pre-trained weights
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=num_classes,
            drop_rate=0.1,
            drop_path_rate=0.1
        )
        
        # Freeze backbone initially
        for name, param in model.named_parameters():
            if 'head' not in name and 'norm' not in name:
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
    
    def train_model(self, model, train_loader, val_loader, class_names):
        """Train the ViT model"""
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            weight_decay=0.01
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, eta_min=1e-6
        )
        
        criterion = nn.CrossEntropyLoss()
        
        best_accuracy = 0.0
        patience = 5
        patience_counter = 0
        
        # Training loop
        for epoch in range(30):  # Reduced epochs for Kaggle time limits
            logger.info(f"\nEpoch {epoch+1}/30")
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc='Training')
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{train_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
                
                # Memory cleanup
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc='Validation'):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # Calculate metrics
            train_accuracy = 100.0 * train_correct / train_total
            val_accuracy = 100.0 * val_correct / val_total
            
            logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%")
            logger.info(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_accuracy,
                    'class_names': class_names
                }, self.working_dir / 'best_vit_model.pth')
                
                logger.info(f"New best model saved! Accuracy: {val_accuracy:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
            
            # Update scheduler
            scheduler.step()
            
            # Unfreeze backbone after a few epochs
            if epoch == 5:
                logger.info("Unfreezing backbone...")
                for param in model.parameters():
                    param.requires_grad = True
                
                # Update optimizer with new parameters
                optimizer = optim.AdamW([
                    {'params': [p for n, p in model.named_parameters() if 'head' not in n], 'lr': 1e-5},
                    {'params': [p for n, p in model.named_parameters() if 'head' in n], 'lr': 1e-4}
                ], weight_decay=0.01)
        
        return best_accuracy
    
    def run_training(self):
        """Main training function"""
        
        try:
            # Find dataset
            data_path = self.find_dataset_path()
            
            # Load data
            train_paths, train_labels, val_paths, val_labels, class_names = self.load_data(data_path)
            
            # Create transforms
            train_transform, val_transform = self.create_transforms()
            
            # Create datasets
            train_dataset = KaggleLivestockDataset(train_paths, train_labels, class_names, train_transform)
            val_dataset = KaggleLivestockDataset(val_paths, val_labels, class_names, val_transform)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=16, shuffle=True, 
                num_workers=2, pin_memory=True, drop_last=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=32, shuffle=False,
                num_workers=2, pin_memory=True
            )
            
            # Create model
            model = self.create_model(len(class_names))
            
            # Train model
            best_accuracy = self.train_model(model, train_loader, val_loader, class_names)
            
            # Save results
            results = {
                'best_accuracy': best_accuracy,
                'target_accuracy': 92.0,
                'target_met': best_accuracy >= 92.0,
                'num_classes': len(class_names),
                'class_names': class_names,
                'total_train_samples': len(train_paths),
                'total_val_samples': len(val_paths)
            }
            
            with open(self.working_dir / 'vit_training_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Print summary
            logger.info(f"\n=== ViT Training Complete ===")
            logger.info(f"Best Accuracy: {best_accuracy:.2f}%")
            logger.info(f"Target: 92-95%")
            logger.info(f"Target Met: {'✓' if best_accuracy >= 92.0 else '✗'}")
            
            if best_accuracy >= 92.0:
                logger.info("🎉 ViT model successfully meets accuracy requirements!")
            else:
                logger.warning("⚠️ ViT model accuracy below target. Consider:")
                logger.warning("  - Longer training")
                logger.warning("  - Different augmentation strategy")
                logger.warning("  - Hyperparameter tuning")
            
            return best_accuracy
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

def main():
    """Main function for Kaggle execution"""
    
    print("🚀 Starting ViT Training for Cattle & Buffalo Recognition")
    print("=" * 60)
    
    trainer = KaggleViTTrainer()
    accuracy = trainer.run_training()
    
    print("=" * 60)
    print(f"✅ Training completed with {accuracy:.2f}% accuracy")
    
    return accuracy

if __name__ == "__main__":
    main()