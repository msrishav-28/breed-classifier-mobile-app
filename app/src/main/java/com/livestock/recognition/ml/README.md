# Machine Learning Package

This package contains the core ML inference components for the cattle and buffalo recognition system.

## Components:

### AdvancedMLInferenceEngine
- Loads and manages ensemble of TensorFlow Lite models
- Coordinates predictions from ViT, YOLOv8-CBAM, and EfficientNetV2
- Handles model validation and performance monitoring

### EnsembleCoordinationService  
- Combines predictions using weighted voting
- Calculates confidence-based model weights
- Applies test-time augmentation for improved accuracy

### TypeClassificationService
- Maps breeds to animal types using CSV lookup
- Provides breed characteristic information
- Validates mapping consistency

### ImagePreprocessor
- Handles image quality validation
- Performs preprocessing for ML inference
- Manages image normalization and resizing

## Model Architecture:
- **Primary**: Vision Transformer (ViT-Base) - 92%+ accuracy target
- **Secondary**: YOLOv8 with CBAM attention - 90%+ accuracy target  
- **Tertiary**: EfficientNetV2-Small - 88%+ accuracy target
- **Ensemble**: Weighted voting - 95%+ accuracy target

## Performance Targets:
- Inference time: <3 seconds on mid-range devices
- Memory usage: <300MB during active use
- Model size: <200MB total for ensemble
- Battery efficiency: <5% drain per 100 inferences