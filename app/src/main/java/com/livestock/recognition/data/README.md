# Data Package

This package handles all data management for the cattle and buffalo recognition system.

## Components:

### models/
- ClassificationResult: Core result data structure
- BreedInfo: Breed information and characteristics  
- AnimalType: Enumeration of animal use types
- ImageMetadata: Image capture and processing metadata

### database/
- Room database implementation for offline storage
- DAO classes for data access operations
- Entity classes for database tables
- Migration strategies for schema updates

### repository/
- Repository pattern implementation
- Data synchronization between local and remote sources
- Caching strategies for offline functionality
- Result persistence and retrieval

### cache/
- Local file caching for images and reports
- Model caching and validation
- Performance optimization for repeated operations

## Database Schema:

### classifications table
- Stores all classification results with metadata
- Supports offline mode with sync flags
- Indexed for efficient querying

### breed_mapping table  
- Static breed information and characteristics
- Type classification lookup data
- Milk yield and usage information

## Offline Strategy:
- All ML models stored locally in assets
- Classification results cached in SQLite
- Optional synchronization when network available
- Graceful degradation without connectivity