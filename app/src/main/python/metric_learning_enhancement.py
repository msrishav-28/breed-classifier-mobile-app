#!/usr/bin/env python3
"""
Metric Learning Enhancement for Cattle and Buffalo Recognition
Adds triplet loss head to ViT model for embedding optimization
Implements similarity-based breed classification
Use only if ensemble accuracy falls below 95%
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import warnings
import gc
import random
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Import model architectures
import timm
from transformers import ViTForImageClassification, ViTConfig
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TripletLoss(nn.Module):
    """Triplet loss for metric learning"""
    
    def __init__(self, margin: float = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
        """
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class ViTWithTripletHead(nn.Module):
    """ViT model with additional triplet loss head for metric learning"""
    
    def __init__(self, base_model_name: str, num_classes: int, embedding_dim: int = 512):
        super(ViTWithTripletHead, self).__init__()
        
        # Load base ViT model
        self.backbone = timm.create_model(
            base_model_name,
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Embedding head for metric learning
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.L2Norm(dim=1)  # L2 normalize embeddings
        )
        
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
    def forward(self, x, return_embeddings=False):
        """Forward pass"""
        # Extract features
        features = self.backbone(x)
        
        # Classification logits
        logits = self.classifier(features)
        
        if return_embeddings:
            # Embedding for metric learning
            embeddings = self.embedding_head(features)
            return logits, embeddings
        
        return logits
    
    def get_embeddings(self, x):
        """Get embeddings for similarity computation"""
        with torch.no_grad():
            features = self.backbone(x)
            embeddings = self.embedding_head(features)
            return embeddings

class L2Norm(nn.Module):
    """L2 normalization layer"""
    
    def __init__(self, dim=1):
        super(L2Norm, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class TripletDataset(Dataset):
    """Dataset for triplet loss training"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None, samples_per_class: int = 100):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples_per_class = samples_per_class
        
        # Load base dataset
        self.dataset = ImageFolder(
            root=self.data_dir / split,
            transform=transform
        )
        
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        
        # Group samples by class for triplet mining
        self.class_samples = self._group_samples_by_class()
        
        # Generate triplets
        self.triplets = self._generate_triplets()
        
        logger.info(f"Created {len(self.triplets)} triplets for {split} split")
        logger.info(f"Classes: {len(self.classes)}")
    
    def _group_samples_by_class(self):
        """Group samples by class for efficient triplet mining"""
        from collections import defaultdict
        
        class_samples = defaultdict(list)
        for idx, (path, class_idx) in enumerate(self.dataset.samples):
            class_samples[class_idx].append(idx)
        
        # Limit samples per class if specified
        if self.samples_per_class:
            for class_idx in class_samples:
                if len(class_samples[class_idx]) > self.samples_per_class:
                    class_samples[class_idx] = random.sample(
                        class_samples[class_idx], 
                        self.samples_per_class
                    )
        
        return dict(class_samples)
    
    def _generate_triplets(self):
        """Generate triplets (anchor, positive, negative)"""
        triplets = []
        
        for class_idx, samples in self.class_samples.items():
            if len(samples) < 2:
                continue  # Need at least 2 samples for positive pairs
            
            # Generate triplets for this class
            for anchor_idx in samples:
                # Select positive (same class, different sample)
                positive_candidates = [idx for idx in samples if idx != anchor_idx]
                if not positive_candidates:
                    continue
                
                positive_idx = random.choice(positive_candidates)
                
                # Select negative (different class)
                negative_classes = [c for c in self.class_samples.keys() if c != class_idx]
                if not negative_classes:
                    continue
                
                negative_class = random.choice(negative_classes)
                negative_idx = random.choice(self.class_samples[negative_class])
                
                triplets.append((anchor_idx, positive_idx, negative_idx))
        
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        
        # Load images
        anchor_img, anchor_label = self.dataset[anchor_idx]
        positive_img, positive_label = self.dataset[positive_idx]
        negative_img, negative_label = self.dataset[negative_idx]
        
        return {
            'anchor': anchor_img,
            'positive': positive_img,
            'negative': negative_img,
            'anchor_label': anchor_label,
            'positive_label': positive_label,
            'negative_label': negative_label
        }

class MetricLearningEnhancement:
    """Metric learning enhancement for improved breed classification"""
    
    def __init__(self, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.get_default_config()
        
        # Training state
        self.model = None
        self.embedding_database = None
        self.class_centroids = None
        
        logger.info(f"Metric Learning Enhancement initialized on {self.device}")
    
    def get_default_config(self):
        """Get default configuration for metric learning"""
        return {
            "model": {
                "base_model": "vit_base_patch16_224",
                "embedding_dim": 512,
                "pretrained_path": "local_finetuned_models/vit_best.pth"
            },
            "training": {
                "epochs": 20,
                "batch_size": 16,
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "triplet_margin": 1.0,
                "classification_weight": 1.0,
                "triplet_weight": 0.5,
                "warmup_epochs": 3
            },
            "data": {
                "image_size": 224,
                "samples_per_class": 100,
                "num_workers": 2
            },
            "similarity": {
                "k_neighbors": 5,
                "similarity_threshold": 0.8,
                "use_centroids": True
            },
            "output": {
                "save_dir": "metric_learning_models",
                "save_embeddings": True
            }
        }
    
    def load_pretrained_model(self, model_path: str, num_classes: int):
        """Load pretrained ViT model and add triplet head"""
        logger.info(f"Loading pretrained model from {model_path}")
        
        # Create model with triplet head
        self.model = ViTWithTripletHead(
            base_model_name=self.config['model']['base_model'],
            num_classes=num_classes,
            embedding_dim=self.config['model']['embedding_dim']
        )
        
        # Load pretrained weights if available
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Load backbone weights
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Filter out classifier weights (we'll retrain the head)
                backbone_state_dict = {}
                for key, value in state_dict.items():
                    if 'head' not in key and 'classifier' not in key:
                        # Map keys to backbone
                        new_key = key.replace('model.', 'backbone.')
                        backbone_state_dict[new_key] = value
                
                # Load backbone weights
                self.model.backbone.load_state_dict(backbone_state_dict, strict=False)
                logger.info("Loaded pretrained backbone weights")
                
            except Exception as e:
                logger.warning(f"Failed to load pretrained weights: {e}")
                logger.info("Using ImageNet pretrained weights")
        
        self.model = self.model.to(self.device)
        return self.model
    
    def get_data_transforms(self, split: str):
        """Get data transforms"""
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def train_metric_learning(self, data_dir: str, num_classes: int):
        """Train model with metric learning"""
        logger.info("=" * 60)
        logger.info("Training Metric Learning Enhancement")
        logger.info("=" * 60)
        
        # Load pretrained model
        model_path = self.config['model']['pretrained_path']
        self.load_pretrained_model(model_path, num_classes)
        
        # Setup data loaders
        train_transform = self.get_data_transforms('train')
        val_transform = self.get_data_transforms('val')
        
        train_dataset = TripletDataset(
            data_dir, 'train', train_transform,
            samples_per_class=self.config['data']['samples_per_class']
        )
        
        val_dataset = ImageFolder(
            root=Path(data_dir) / 'validation',
            transform=val_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        # Setup training components
        classification_criterion = nn.CrossEntropyLoss()
        triplet_criterion = TripletLoss(margin=self.config['training']['triplet_margin'])
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        scaler = GradScaler()
        
        # Training loop
        best_accuracy = 0.0
        
        for epoch in range(self.config['training']['epochs']):
            # Training
            train_loss = self.train_epoch(
                train_loader, classification_criterion, triplet_criterion, 
                optimizer, scaler, epoch
            )
            
            # Validation
            val_accuracy = self.validate_epoch(val_loader, classification_criterion, epoch)
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model(epoch, val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                       f"Train Loss: {train_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        logger.info(f"Metric learning training completed. Best accuracy: {best_accuracy:.2f}%")
        
        # Build embedding database
        self.build_embedding_database(val_loader)
        
        return best_accuracy
    
    def train_epoch(self, train_loader, classification_criterion, triplet_criterion, 
                   optimizer, scaler, epoch):
        """Train for one epoch with combined loss"""
        self.model.train()
        total_loss = 0.0
        
        classification_weight = self.config['training']['classification_weight']
        triplet_weight = self.config['training']['triplet_weight']
        
        for batch_idx, batch in enumerate(train_loader):
            anchor_imgs = batch['anchor'].to(self.device)
            positive_imgs = batch['positive'].to(self.device)
            negative_imgs = batch['negative'].to(self.device)
            anchor_labels = batch['anchor_label'].to(self.device)
            
            optimizer.zero_grad()
            
            with autocast():
                # Get predictions and embeddings
                anchor_logits, anchor_embeddings = self.model(anchor_imgs, return_embeddings=True)
                positive_logits, positive_embeddings = self.model(positive_imgs, return_embeddings=True)
                negative_logits, negative_embeddings = self.model(negative_imgs, return_embeddings=True)
                
                # Classification loss
                classification_loss = classification_criterion(anchor_logits, anchor_labels)
                
                # Triplet loss
                triplet_loss = triplet_criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                
                # Combined loss
                total_batch_loss = (classification_weight * classification_loss + 
                                  triplet_weight * triplet_loss)
            
            scaler.scale(total_batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += total_batch_loss.item()
            
            if batch_idx % 20 == 0:
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {total_batch_loss.item():.4f} '
                          f'(Cls: {classification_loss.item():.4f}, '
                          f'Triplet: {triplet_loss.item():.4f})')
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader, criterion, epoch):
        """Validate for one epoch"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def build_embedding_database(self, val_loader):
        """Build embedding database for similarity-based classification"""
        logger.info("Building embedding database...")
        
        self.model.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                batch_embeddings = self.model.get_embeddings(data)
                
                embeddings.append(batch_embeddings.cpu().numpy())
                labels.extend(target.numpy())
        
        # Combine all embeddings
        self.embedding_database = np.vstack(embeddings)
        self.embedding_labels = np.array(labels)
        
        # Compute class centroids
        self.class_centroids = {}
        for class_idx in np.unique(labels):
            class_mask = self.embedding_labels == class_idx
            class_embeddings = self.embedding_database[class_mask]
            centroid = np.mean(class_embeddings, axis=0)
            self.class_centroids[class_idx] = centroid
        
        logger.info(f"Built embedding database with {len(self.embedding_database)} samples")
        logger.info(f"Computed centroids for {len(self.class_centroids)} classes")
        
        # Save embedding database
        if self.config['output']['save_embeddings']:
            save_dir = Path(self.config['output']['save_dir'])
            save_dir.mkdir(exist_ok=True)
            
            np.save(save_dir / 'embedding_database.npy', self.embedding_database)
            np.save(save_dir / 'embedding_labels.npy', self.embedding_labels)
            
            # Save centroids
            centroids_dict = {str(k): v.tolist() for k, v in self.class_centroids.items()}
            with open(save_dir / 'class_centroids.json', 'w') as f:
                json.dump(centroids_dict, f)
    
    def similarity_based_prediction(self, image_tensor):
        """Make prediction using similarity-based classification"""
        if self.embedding_database is None or self.class_centroids is None:
            logger.error("Embedding database not built")
            return None
        
        self.model.eval()
        
        with torch.no_grad():
            # Get embedding for input image
            query_embedding = self.model.get_embeddings(image_tensor.unsqueeze(0))
            query_embedding = query_embedding.cpu().numpy().flatten()
            
            # Method 1: K-nearest neighbors
            similarities = cosine_similarity([query_embedding], self.embedding_database)[0]
            top_k_indices = np.argsort(similarities)[-self.config['similarity']['k_neighbors']:]
            top_k_labels = self.embedding_labels[top_k_indices]
            
            # Vote among top-k neighbors
            from collections import Counter
            label_counts = Counter(top_k_labels)
            knn_prediction = label_counts.most_common(1)[0][0]
            knn_confidence = label_counts.most_common(1)[0][1] / len(top_k_labels)
            
            # Method 2: Centroid-based classification
            centroid_similarities = {}
            for class_idx, centroid in self.class_centroids.items():
                similarity = cosine_similarity([query_embedding], [centroid])[0][0]
                centroid_similarities[class_idx] = similarity
            
            centroid_prediction = max(centroid_similarities, key=centroid_similarities.get)
            centroid_confidence = centroid_similarities[centroid_prediction]
            
            # Combine predictions
            if self.config['similarity']['use_centroids']:
                # Use centroid method if confidence is high
                if centroid_confidence > self.config['similarity']['similarity_threshold']:
                    return {
                        'prediction': centroid_prediction,
                        'confidence': centroid_confidence,
                        'method': 'centroid'
                    }
                else:
                    return {
                        'prediction': knn_prediction,
                        'confidence': knn_confidence,
                        'method': 'knn'
                    }
            else:
                return {
                    'prediction': knn_prediction,
                    'confidence': knn_confidence,
                    'method': 'knn'
                }
    
    def save_model(self, epoch, accuracy):
        """Save model checkpoint"""
        save_dir = Path(self.config['output']['save_dir'])
        save_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'config': self.config
        }
        
        save_path = save_dir / 'metric_learning_model_best.pth'
        torch.save(checkpoint, save_path)
        
        logger.info(f"Model saved: {save_path} (Accuracy: {accuracy:.2f}%)")
    
    def run_metric_learning_enhancement(self, data_dir: str, num_classes: int):
        """Run complete metric learning enhancement"""
        logger.info("🚀 Starting Metric Learning Enhancement")
        logger.info("Adding triplet loss head to ViT model")
        logger.info("=" * 80)
        
        # Create output directory
        Path(self.config['output']['save_dir']).mkdir(exist_ok=True)
        
        # Train with metric learning
        accuracy = self.train_metric_learning(data_dir, num_classes)
        
        # Generate report
        report = self.generate_enhancement_report(accuracy)
        
        logger.info("\n" + "=" * 80)
        logger.info("🎯 METRIC LEARNING ENHANCEMENT COMPLETE")
        logger.info("=" * 80)
        
        return report
    
    def generate_enhancement_report(self, accuracy):
        """Generate enhancement report"""
        report = {
            'metric_learning_summary': {
                'enhancement_type': 'Triplet Loss + Similarity-based Classification',
                'final_accuracy': accuracy,
                'embedding_dimension': self.config['model']['embedding_dim'],
                'similarity_method': 'Cosine Similarity + K-NN',
                'use_case': 'Use only if ensemble accuracy < 95%'
            },
            'model_architecture': {
                'base_model': self.config['model']['base_model'],
                'additional_components': [
                    'Triplet Loss Head',
                    'Embedding Database',
                    'Class Centroids',
                    'Similarity-based Classifier'
                ]
            },
            'training_configuration': self.config,
            'integration_notes': [
                'Can be used as fallback when ensemble confidence is low',
                'Provides similarity scores for breed relationships',
                'Enables few-shot learning for new breeds',
                'Improves performance on rare breeds'
            ]
        }
        
        # Save report
        report_path = Path(self.config['output']['save_dir']) / 'metric_learning_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    """Main function for metric learning enhancement"""
    
    print("🚀 Metric Learning Enhancement")
    print("Triplet Loss + Similarity-based Classification")
    print("Use only if ensemble accuracy falls below 95%")
    print("=" * 80)
    
    # Configuration
    data_dir = "dataset"  # Update with actual dataset path
    num_classes = 23  # Update with actual number of classes
    
    # Create enhancement
    enhancement = MetricLearningEnhancement()
    
    # Run enhancement
    report = enhancement.run_metric_learning_enhancement(data_dir, num_classes)
    
    # Print summary
    print(f"\n✅ Metric Learning Enhancement Complete!")
    print(f"📊 Final Accuracy: {report['metric_learning_summary']['final_accuracy']:.2f}%")
    print(f"🎯 Enhancement Type: {report['metric_learning_summary']['enhancement_type']}")
    
    return report

if __name__ == "__main__":
    main()