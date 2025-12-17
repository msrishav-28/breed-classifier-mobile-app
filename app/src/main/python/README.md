# Vision Transformer Training Pipeline

This directory contains the Vision Transformer (ViT) training pipeline for the Cattle and Buffalo Recognition system, optimized for Kaggle P100 GPU environment.

## Files Overview

### Core Training Scripts
- **`kaggle_vit_training.py`** - Main Kaggle-optimized training script
- **`vit_training_pipeline.py`** - Full-featured training pipeline with advanced features
- **`model_converter.py`** - Converts trained PyTorch models to TensorFlow Lite

### Configuration
- **`vit_config.json`** - Training configuration parameters
- **`requirements.txt`** - Python dependencies

## Quick Start (Kaggle)

1. **Upload to Kaggle Notebook:**
   ```python
   # Copy kaggle_vit_training.py to your Kaggle notebook
   # Ensure your dataset is available in /kaggle/input/
   ```

2. **Run Training:**
   ```python
   exec(open('kaggle_vit_training.py').read())
   ```

3. **Expected Output:**
   - Best model saved as `best_vit_model.pth`
   - Training results in `vit_training_results.json`
   - Target accuracy: 92-95%

## Detailed Usage

### 1. Environment Setup

For local development:
```bash
pip install -r requirements.txt
```

For Kaggle (auto-installed):
- torch>=2.0.0
- timm>=0.9.0
- albumentations>=1.3.0

### 2. Dataset Structure

Expected dataset structure:
```
dataset/
├── train/
│   ├── Gir/
│   ├── Sahiwal/
│   ├── Murrah/
│   └── ...
├── validation/
│   ├── Gir/
│   ├── Sahiwal/
│   └── ...
└── test/
    ├── Gir/
    ├── Sahiwal/
    └── ...
```

### 3. Training Configuration

Key parameters in `vit_config.json`:

```json
{
  "model": {
    "name": "vit_base_patch16_224",
    "pretrained": true,
    "num_classes": 23
  },
  "training": {
    "batch_size": 16,
    "epochs": 30,
    "learning_rate": 1e-4
  },
  "targets": {
    "accuracy_target": 92.0
  }
}
```

### 4. Memory Optimization Features

- **Mixed Precision Training**: Reduces memory usage by ~50%
- **Gradient Checkpointing**: Trades compute for memory
- **Gradient Accumulation**: Effective larger batch sizes
- **Optimized Data Loading**: Efficient data pipeline

### 5. Model Conversion

Convert trained PyTorch model to TensorFlow Lite:

```python
from model_converter import ViTModelConverter

converter = ViTModelConverter(
    pytorch_model_path='best_vit_model.pth',
    output_dir='converted_models'
)

report = converter.convert_full_pipeline(quantize=True)
```

## Performance Targets

### Accuracy Targets
- **Individual ViT Model**: 92-95%
- **Ensemble Contribution**: Primary model in 3-model ensemble
- **Per-breed Accuracy**: >90% for each of 23 breeds

### Hardware Optimization
- **Kaggle P100**: 16GB VRAM, batch size 16-20
- **Memory Usage**: <12GB during training
- **Training Time**: ~2-3 hours for 30 epochs

### Mobile Deployment
- **TFLite Model Size**: <100MB after quantization
- **Inference Time**: <1000ms on mobile devices
- **Accuracy Preservation**: Within 2% of original

## Advanced Features

### 1. Transfer Learning Strategy
- Start with frozen backbone (5 epochs)
- Gradually unfreeze layers
- Different learning rates for backbone vs head

### 2. Data Augmentation
- Geometric: rotation, scaling, flipping
- Photometric: brightness, contrast, hue
- Advanced: cutout, coarse dropout
- Weather simulation: rain, fog effects

### 3. Training Techniques
- Cosine annealing learning rate
- Early stopping with patience
- Model checkpointing
- Gradient clipping

### 4. Validation & Metrics
- Per-class accuracy analysis
- Confusion matrix visualization
- Classification report generation
- Model performance benchmarking

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size to 8-12
   - Enable gradient checkpointing
   - Increase gradient accumulation steps

2. **Low Accuracy (<90%)**
   - Increase training epochs
   - Adjust learning rate
   - Check data quality and balance
   - Verify augmentation strategy

3. **Slow Training**
   - Increase num_workers for data loading
   - Use mixed precision training
   - Enable persistent workers

### Performance Optimization

1. **Memory Efficiency**
   ```python
   # Enable gradient checkpointing
   model.set_grad_checkpointing(True)
   
   # Use mixed precision
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

2. **Speed Optimization**
   ```python
   # Compile model (PyTorch 2.0+)
   model = torch.compile(model)
   
   # Optimized data loading
   DataLoader(..., pin_memory=True, persistent_workers=True)
   ```

## Integration with Android App

The trained ViT model integrates with the Android app through:

1. **Model Conversion**: PyTorch → TensorFlow Lite
2. **Mobile Inference**: Via `AdvancedMLInferenceEngine`
3. **Ensemble Coordination**: Combined with YOLOv8 and EfficientNet
4. **Performance Monitoring**: Inference time and accuracy tracking

## Results Validation

Expected training results:
- Training accuracy: >95%
- Validation accuracy: 92-95%
- Per-class accuracy: >90% for major breeds
- Model size: <100MB (quantized)
- Inference time: <1000ms on mobile

## Next Steps

After successful ViT training:
1. Proceed to YOLOv8-CBAM training (Task 5.2)
2. Train EfficientNetV2 model (Task 5.3)
3. Implement ensemble coordination
4. Convert all models to TensorFlow Lite
5. Integrate with Android application

## Support

For issues or questions:
- Check Kaggle notebook logs
- Verify dataset structure and quality
- Monitor GPU memory usage
- Review training configuration parameters