# ML Models Directory

This directory contains the TensorFlow Lite models for cattle and buffalo recognition:

## Model Files (to be added in later tasks):
- `vit_model.tflite` - Vision Transformer model for primary breed classification
- `yolov8_cbam_model.tflite` - YOLOv8 with CBAM attention for object detection and classification
- `efficientnetv2_model.tflite` - EfficientNetV2 model for ensemble diversity
- `model_metadata.json` - Model version and configuration information

## Model Requirements:
- Total ensemble size: <200MB
- Individual model accuracy targets:
  - ViT: >92%
  - YOLOv8-CBAM: >90%
  - EfficientNetV2: >88%
- Ensemble accuracy target: >95%

## Usage:
Models are loaded by the AdvancedMLInferenceEngine during application startup.
All models support offline inference without network connectivity.