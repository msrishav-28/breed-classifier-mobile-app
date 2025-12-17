# User Interface Package

This package contains all UI components for the cattle and buffalo recognition application.

## Structure:

### MainActivity
- Main entry point and navigation hub
- Handles permissions and app initialization
- Provides access to camera functionality

### camera/
- CameraActivity: Image capture interface
- CameraViewModel: Camera state management
- Quality validation and user feedback

### results/
- ResultsActivity: Display classification results
- ResultsViewModel: Results state management
- Breed information and confidence display

### reports/
- Report generation and sharing functionality
- PDF export capabilities
- Multi-platform sharing options

## Design Principles:
- Material Design 3 components
- Accessibility compliance (screen readers, high contrast)
- Offline-first functionality
- Clear error messaging and user guidance
- Intuitive navigation with visual feedback

## Key Features:
- Real-time camera preview with capture controls
- Image quality validation with user feedback
- Comprehensive results display with breed information
- Report generation with embedded images and metadata
- Multi-platform sharing (email, messaging, file transfer)