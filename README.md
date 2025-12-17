# Cattle and Buffalo Recognition System

An AI-powered mobile application for accurate livestock breed identification and type classification, designed to support India's livestock sector and the Rashtriya Gokul Mission.

## Overview

This Android application uses advanced machine learning techniques to identify cattle and buffalo breeds from images, providing real-time classification with high accuracy. The system combines ensemble learning with offline capabilities to work in rural areas with limited connectivity.

## Features

- **Breed Recognition**: Identifies 15+ Indian cattle breeds and 8+ buffalo breeds with 95%+ accuracy
- **Type Classification**: Categorizes animals as Dairy, Draught, or Dual-purpose
- **Offline Functionality**: Works without internet connectivity using local ML models
- **Report Generation**: Creates detailed PDF reports with annotations and metadata
- **Multi-platform Sharing**: Share results via email, messaging apps, or file transfer
- **Real-time Processing**: Sub-3-second inference on mid-range smartphones

## Technical Architecture

### Machine Learning Models
- **Vision Transformer (ViT)**: Primary classification model (92%+ accuracy target)
- **YOLOv8 with CBAM**: Object detection and classification (90%+ accuracy target)
- **EfficientNetV2**: Ensemble diversity model (88%+ accuracy target)
- **Ensemble Coordination**: Weighted voting for 95%+ combined accuracy

### Technology Stack
- **Platform**: Android (API 24+)
- **Language**: Kotlin
- **ML Framework**: TensorFlow Lite
- **Camera**: CameraX API
- **Database**: Room (SQLite)
- **UI**: Material Design 3 with Jetpack Compose
- **Build System**: Gradle with Kotlin DSL

## Project Structure

```
app/
├── src/main/
│   ├── java/com/livestock/recognition/
│   │   ├── data/           # Data models and database
│   │   ├── ml/             # Machine learning components
│   │   ├── ui/             # User interface components
│   │   └── MainActivity.kt # Main entry point
│   ├── assets/
│   │   ├── models/         # TensorFlow Lite models
│   │   └── data/           # Breed mapping and characteristics
│   └── res/                # Android resources
├── src/test/               # Unit tests
└── src/androidTest/        # Instrumented tests
```

## Requirements

- Android 7.0 (API level 24) or higher
- Camera permission for image capture
- Storage permission for report generation
- Minimum 3GB RAM for optimal performance
- 500MB free storage for models and data

## Development Setup

1. Clone the repository
2. Open in Android Studio Arctic Fox or later
3. Sync Gradle dependencies
4. Add ML models to `app/src/main/assets/models/` (see models README)
5. Build and run on device or emulator

## Model Training

The system uses a hybrid training approach:

1. **Phase 1**: Heavy compute training on Kaggle GPU (P100, 16GB VRAM)
2. **Phase 2**: Local fine-tuning on RTX 3050 (6GB VRAM)
3. **Dataset**: 22,000+ images with advanced augmentation (30,000-40,000 total)
4. **Techniques**: Mixed precision, gradient accumulation, metric learning

## Performance Targets

- **Accuracy**: 95%+ ensemble accuracy across all breeds
- **Speed**: <3 seconds processing time on mid-range devices
- **Memory**: <300MB during active use
- **Battery**: <5% drain per 100 inferences
- **Model Size**: <200MB total for ensemble

## Testing Strategy

- **Unit Tests**: Specific examples and edge cases
- **Property-Based Tests**: Universal properties across all inputs
- **Integration Tests**: End-to-end workflow validation
- **Performance Tests**: Benchmarking on target devices

## Contributing

This project follows the spec-driven development methodology:

1. Requirements gathering with EARS patterns
2. Comprehensive design with correctness properties
3. Implementation plan with actionable tasks
4. Property-based testing for verification

## License

This project is developed for the Smart India Hackathon 2025 (SIH25004 & SIH25005).

## Support

For technical issues or questions, please refer to the project documentation in the `.kiro/specs/` directory.