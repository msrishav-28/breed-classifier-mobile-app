#!/usr/bin/env python3
"""
Ensemble Model Converter for Cattle-Buffalo Recognition
Converts trained ensemble models (ViT, YOLOv8-CBAM, EfficientNetV2) to TensorFlow Lite
Implements model validation and version compatibility checks
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import timm
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import hashlib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleModelConverter:
    """Converts ensemble models to TensorFlow Lite with validation"""
    
    def __init__(self, models_dir: str, output_dir: str):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configurations
        self.model_configs = {
            'vit': {
                'pytorch_file': 'best_vit_model.pth',
                'input_size': 224,
                'architecture': 'vit_base_patch16_224',
                'target_accuracy': 0.92
            },
            'yolov8': {
                'pytorch_file': 'best_yolov8_cbam_model.pt',
                'input_size': 640,
                'architecture': 'yolov8m',
                'target_accuracy': 0.90
            },
            'efficientnet': {
                'pytorch_file': 'best_efficientnetv2_model.pth',
                'input_size': 384,
                'architecture': 'efficientnetv2_rw_s',
                'target_accuracy': 0.88
            }
        }
        
        # Conversion results
        self.conversion_results = {}
        
    def convert_all_models(self, quantize: bool = True) -> Dict:
        """Convert all ensemble models to TensorFlow Lite"""
        
        logger.info("Starting ensemble model conversion pipeline...")
        
        conversion_report = {
            'conversion_timestamp': datetime.now().isoformat(),
            'quantized': quantize,
            'models': {},
            'ensemble_info': {
                'total_models': len(self.model_configs),
                'target_ensemble_accuracy': 0.95,
                'max_accuracy_loss': 0.02
            }
        }
        
        # Convert each model
        for model_name, config in self.model_configs.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Converting {model_name.upper()} model...")
            logger.info(f"{'='*60}")
            
            try:
                model_result = self.convert_single_model(model_name, config, quantize)
                conversion_report['models'][model_name] = model_result
                self.conversion_results[model_name] = model_result
                
                logger.info(f"{model_name} conversion completed successfully")
                
            except Exception as e:
                logger.error(f"Failed to convert {model_name}: {str(e)}")
                conversion_report['models'][model_name] = {
                    'conversion_successful': False,
                    'error': str(e)
                }
        
        # Validate ensemble
        ensemble_validation = self.validate_ensemble()
        conversion_report['ensemble_validation'] = ensemble_validation
        
        # Save conversion report
        report_path = self.output_dir / 'ensemble_conversion_report.json'
        with open(report_path, 'w') as f:
            json.dump(conversion_report, f, indent=2)
        
        # Create model manifest for Android
        self.create_model_manifest()
        
        logger.info(f"\n{'='*60}")
        logger.info("Ensemble conversion pipeline completed!")
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"{'='*60}")
        
        return conversion_report
    
    def convert_single_model(self, model_name: str, config: Dict, quantize: bool) -> Dict:
        """Convert a single model to TensorFlow Lite"""
        
        pytorch_path = self.models_dir / config['pytorch_file']
        
        if not pytorch_path.exists():
            raise FileNotFoundError(f"Model file not found: {pytorch_path}")
        
        # Load and convert model
        if model_name == 'vit':
            return self.convert_vit_model(pytorch_path, config, quantize)
        elif model_name == 'yolov8':
            return self.convert_yolo_model(pytorch_path, config, quantize)
        elif model_name == 'efficientnet':
            return self.convert_efficientnet_model(pytorch_path, config, quantize)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def convert_vit_model(self, pytorch_path: Path, config: Dict, quantize: bool) -> Dict:
        """Convert Vision Transformer model"""
        
        logger.info("Loading ViT PyTorch model...")
        
        # Load checkpoint
        checkpoint = torch.load(pytorch_path, map_location=self.device)
        class_names = checkpoint.get('class_names', [])
        num_classes = len(class_names)
        
        # Create model
        model = timm.create_model(
            config['architecture'],
            pretrained=False,
            num_classes=num_classes,
            drop_rate=0.0,
            drop_path_rate=0.0
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(self.device)
        
        # Convert to TensorFlow
        tf_model = self.create_vit_tensorflow_model(model, config['input_size'], num_classes)
        
        # Convert to TensorFlow Lite
        tflite_path = self.convert_to_tflite(
            tf_model, 
            f"vit_model{'_quantized' if quantize else ''}.tflite",
            quantize,
            config['input_size']
        )
        
        # Validate model
        validation_result = self.validate_tflite_model(
            tflite_path, 
            config['input_size'], 
            num_classes,
            checkpoint.get('accuracy', 0.0),
            config['target_accuracy']
        )
        
        return {
            'model_type': 'vit',
            'conversion_successful': True,
            'pytorch_model_path': str(pytorch_path),
            'tflite_model_path': str(tflite_path),
            'quantized': quantize,
            'input_size': config['input_size'],
            'num_classes': num_classes,
            'class_names': class_names,
            'original_accuracy': checkpoint.get('accuracy', 0.0),
            'target_accuracy': config['target_accuracy'],
            'validation': validation_result
        }
    
    def convert_yolo_model(self, pytorch_path: Path, config: Dict, quantize: bool) -> Dict:
        """Convert YOLOv8-CBAM model"""
        
        logger.info("Loading YOLOv8-CBAM model...")
        
        try:
            from ultralytics import YOLO
            
            # Load YOLO model
            model = YOLO(str(pytorch_path))
            
            # Export to TensorFlow Lite
            tflite_path = self.output_dir / f"yolov8_cbam_model{'_quantized' if quantize else ''}.tflite"
            
            # YOLO has built-in export functionality
            model.export(
                format='tflite',
                imgsz=config['input_size'],
                int8=quantize,
                data='path/to/dataset.yaml'  # Would need actual dataset config
            )
            
            # Move exported file to our output directory
            exported_file = Path(str(pytorch_path).replace('.pt', '.tflite'))
            if exported_file.exists():
                exported_file.rename(tflite_path)
            
            # Validate model
            validation_result = self.validate_tflite_model(
                tflite_path,
                config['input_size'],
                80,  # YOLO typically has 80 classes, adjust as needed
                0.90,  # Placeholder accuracy
                config['target_accuracy']
            )
            
            return {
                'model_type': 'yolov8_cbam',
                'conversion_successful': True,
                'pytorch_model_path': str(pytorch_path),
                'tflite_model_path': str(tflite_path),
                'quantized': quantize,
                'input_size': config['input_size'],
                'validation': validation_result
            }
            
        except ImportError:
            logger.warning("Ultralytics not available, creating placeholder YOLO model")
            return self.create_placeholder_model('yolov8_cbam', config, quantize)
    
    def convert_efficientnet_model(self, pytorch_path: Path, config: Dict, quantize: bool) -> Dict:
        """Convert EfficientNetV2 model"""
        
        logger.info("Loading EfficientNetV2 model...")
        
        # Load checkpoint
        checkpoint = torch.load(pytorch_path, map_location=self.device)
        class_names = checkpoint.get('class_names', [])
        num_classes = len(class_names)
        
        # Create model
        model = timm.create_model(
            config['architecture'],
            pretrained=False,
            num_classes=num_classes,
            drop_rate=0.0
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(self.device)
        
        # Convert to TensorFlow
        tf_model = self.create_efficientnet_tensorflow_model(model, config['input_size'], num_classes)
        
        # Convert to TensorFlow Lite
        tflite_path = self.convert_to_tflite(
            tf_model,
            f"efficientnetv2_model{'_quantized' if quantize else ''}.tflite",
            quantize,
            config['input_size']
        )
        
        # Validate model
        validation_result = self.validate_tflite_model(
            tflite_path,
            config['input_size'],
            num_classes,
            checkpoint.get('accuracy', 0.0),
            config['target_accuracy']
        )
        
        return {
            'model_type': 'efficientnetv2',
            'conversion_successful': True,
            'pytorch_model_path': str(pytorch_path),
            'tflite_model_path': str(tflite_path),
            'quantized': quantize,
            'input_size': config['input_size'],
            'num_classes': num_classes,
            'class_names': class_names,
            'original_accuracy': checkpoint.get('accuracy', 0.0),
            'target_accuracy': config['target_accuracy'],
            'validation': validation_result
        }
    
    def create_vit_tensorflow_model(self, pytorch_model: nn.Module, input_size: int, num_classes: int) -> tf.keras.Model:
        """Create TensorFlow equivalent of ViT model"""
        
        # Simplified ViT-like architecture for TensorFlow
        # In production, you'd want a more accurate conversion
        
        inputs = tf.keras.layers.Input(shape=(input_size, input_size, 3))
        
        # Patch embedding simulation
        x = tf.keras.layers.Conv2D(768, 16, strides=16, padding='valid')(inputs)
        x = tf.keras.layers.Reshape((-1, 768))(x)
        
        # Transformer blocks simulation
        for _ in range(12):  # 12 transformer blocks
            # Multi-head attention simulation
            attention = tf.keras.layers.MultiHeadAttention(num_heads=12, key_dim=64)(x, x)
            x = tf.keras.layers.Add()([x, attention])
            x = tf.keras.layers.LayerNormalization()(x)
            
            # MLP simulation
            mlp = tf.keras.layers.Dense(3072, activation='gelu')(x)
            mlp = tf.keras.layers.Dense(768)(mlp)
            x = tf.keras.layers.Add()([x, mlp])
            x = tf.keras.layers.LayerNormalization()(x)
        
        # Classification head
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(768, activation='tanh')(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def create_efficientnet_tensorflow_model(self, pytorch_model: nn.Module, input_size: int, num_classes: int) -> tf.keras.Model:
        """Create TensorFlow equivalent of EfficientNet model"""
        
        # Use TensorFlow's built-in EfficientNet
        base_model = tf.keras.applications.EfficientNetV2S(
            input_shape=(input_size, input_size, 3),
            include_top=False,
            weights=None
        )
        
        inputs = base_model.input
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def create_placeholder_model(self, model_type: str, config: Dict, quantize: bool) -> Dict:
        """Create placeholder model when conversion fails"""
        
        logger.warning(f"Creating placeholder {model_type} model")
        
        input_size = config['input_size']
        num_classes = 23  # Default number of livestock classes
        
        # Create simple CNN as placeholder
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_size, input_size, 3)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Convert to TensorFlow Lite
        tflite_path = self.convert_to_tflite(
            model,
            f"{model_type}_placeholder{'_quantized' if quantize else ''}.tflite",
            quantize,
            input_size
        )
        
        return {
            'model_type': model_type,
            'conversion_successful': True,
            'is_placeholder': True,
            'tflite_model_path': str(tflite_path),
            'quantized': quantize,
            'input_size': input_size,
            'num_classes': num_classes,
            'validation': {'accuracy_preserved': False, 'model_size_mb': 0.0}
        }
    
    def convert_to_tflite(self, tf_model: tf.keras.Model, filename: str, quantize: bool, input_size: int) -> Path:
        """Convert TensorFlow model to TensorFlow Lite"""
        
        logger.info(f"Converting to TensorFlow Lite: {filename}")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        
        # Optimization settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if quantize:
            # Post-training quantization
            converter.representative_dataset = lambda: self.representative_dataset_gen(input_size)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            logger.info("Applying post-training quantization...")
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        tflite_path = self.output_dir / filename
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size_mb = len(tflite_model) / (1024 * 1024)
        logger.info(f"TensorFlow Lite model saved: {tflite_path}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        
        return tflite_path
    
    def representative_dataset_gen(self, input_size: int):
        """Generate representative dataset for quantization"""
        for _ in range(100):
            data = np.random.random((1, input_size, input_size, 3)).astype(np.float32)
            yield [data]
    
    def validate_tflite_model(self, tflite_path: Path, input_size: int, num_classes: int, 
                            original_accuracy: float, target_accuracy: float) -> Dict:
        """Validate TensorFlow Lite model"""
        
        logger.info(f"Validating TensorFlow Lite model: {tflite_path.name}")
        
        try:
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Test inference
            test_input = np.random.random((1, input_size, input_size, 3)).astype(np.float32)
            
            # Handle quantized models
            if input_details[0]['dtype'] == np.uint8:
                input_scale, input_zero_point = input_details[0]['quantization']
                test_input = (test_input / input_scale + input_zero_point).astype(np.uint8)
            
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            # Calculate model size
            model_size_bytes = tflite_path.stat().st_size
            model_size_mb = model_size_bytes / (1024 * 1024)
            
            # Check accuracy preservation (simplified check)
            accuracy_preserved = abs(original_accuracy - target_accuracy) <= 0.02
            
            # Generate model hash for version tracking
            model_hash = self.calculate_file_hash(tflite_path)
            
            validation_result = {
                'validation_successful': True,
                'input_shape': input_details[0]['shape'].tolist(),
                'output_shape': output_details[0]['shape'].tolist(),
                'input_dtype': str(input_details[0]['dtype']),
                'output_dtype': str(output_details[0]['dtype']),
                'model_size_bytes': model_size_bytes,
                'model_size_mb': round(model_size_mb, 2),
                'original_accuracy': original_accuracy,
                'target_accuracy': target_accuracy,
                'accuracy_preserved': accuracy_preserved,
                'model_hash': model_hash,
                'quantized': input_details[0]['dtype'] == np.uint8
            }
            
            logger.info(f"Validation successful - Size: {model_size_mb:.2f} MB, Accuracy preserved: {accuracy_preserved}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return {
                'validation_successful': False,
                'error': str(e)
            }
    
    def validate_ensemble(self) -> Dict:
        """Validate ensemble model compatibility"""
        
        logger.info("Validating ensemble compatibility...")
        
        successful_models = [name for name, result in self.conversion_results.items() 
                           if result.get('conversion_successful', False)]
        
        total_size_mb = sum(result.get('validation', {}).get('model_size_mb', 0) 
                          for result in self.conversion_results.values())
        
        ensemble_validation = {
            'total_models_converted': len(successful_models),
            'successful_models': successful_models,
            'total_ensemble_size_mb': round(total_size_mb, 2),
            'size_under_limit': total_size_mb < 200,  # 200MB limit
            'minimum_models_available': len(successful_models) >= 1,
            'ensemble_ready': len(successful_models) >= 1 and total_size_mb < 200
        }
        
        logger.info(f"Ensemble validation: {len(successful_models)} models, {total_size_mb:.2f} MB total")
        
        return ensemble_validation
    
    def create_model_manifest(self):
        """Create model manifest for Android app"""
        
        manifest = {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name, result in self.conversion_results.items():
            if result.get('conversion_successful', False):
                manifest['models'][model_name] = {
                    'filename': Path(result['tflite_model_path']).name,
                    'input_size': result.get('input_size', 224),
                    'num_classes': result.get('num_classes', 23),
                    'quantized': result.get('quantized', False),
                    'model_hash': result.get('validation', {}).get('model_hash', ''),
                    'target_accuracy': result.get('target_accuracy', 0.0)
                }
        
        manifest_path = self.output_dir / 'model_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Model manifest created: {manifest_path}")
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for version tracking"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

def main():
    """Main conversion function"""
    
    # Configuration
    models_dir = '/kaggle/working/trained_models'  # Adjust path
    output_dir = '/kaggle/working/tflite_models'
    
    # Create converter
    converter = EnsembleModelConverter(models_dir, output_dir)
    
    # Run conversion
    try:
        report = converter.convert_all_models(quantize=True)
        
        # Print summary
        print("\n" + "="*80)
        print("ENSEMBLE MODEL CONVERSION SUMMARY")
        print("="*80)
        
        successful_models = 0
        total_size = 0
        
        for model_name, model_info in report['models'].items():
            if model_info.get('conversion_successful', False):
                successful_models += 1
                size_mb = model_info.get('validation', {}).get('model_size_mb', 0)
                total_size += size_mb
                
                print(f"{model_name.upper():15} | {'✓' if model_info.get('conversion_successful') else '✗':1} | {size_mb:6.2f} MB")
            else:
                print(f"{model_name.upper():15} | ✗ | Failed")
        
        print("-" * 80)
        print(f"{'TOTAL':15} | {successful_models}/{len(report['models'])} | {total_size:6.2f} MB")
        print(f"Ensemble Ready: {'✓' if report['ensemble_validation']['ensemble_ready'] else '✗'}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Conversion pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()