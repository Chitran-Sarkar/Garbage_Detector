#classifier-module.py
from cvzone.ClassificationModule import Classifier
import logging
import os
import glob
from typing import Union, Tuple, List, Optional
import numpy as np
import json
from utils import resource_path
import concurrent.futures
import cv2


def load_classifier(model_path, label_path):
    """Loads and returns the classifier using the provided model and label file paths."""
    try:
        return Classifier(model_path, label_path)
    except Exception:
        logging.exception("Failed to load classifier.")
        return None


def load_ensemble_classifier(models_dir=None, custom_models=None, max_workers=None):
    """Load an ensemble classifier based on configuration or custom model list"""
    if models_dir is None:
        models_dir = resource_path('Resources/Models')
        
    if not os.path.exists(models_dir):
        logging.error(f"Models directory not found: {models_dir}")
        return load_default_classifier()
    
    config_path = os.path.join(models_dir, 'keras_model_ensemble_config.json')
    labels_path = os.path.join(models_dir, 'labels.txt')
    
    if not os.path.exists(labels_path):
        logging.error(f"Labels file not found: {labels_path}")
        return load_default_classifier()
    
    try:
        from ensemble_classifier import EnsembleClassifier
        
        use_config_file = os.path.exists(config_path) and not custom_models
        ensemble = EnsembleClassifier(max_workers=max_workers)
        models_loaded = 0
        
        if use_config_file:
            try:
                logging.info(f"Loading ensemble configuration from {config_path}")
                
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                if not isinstance(config, dict) or 'models' not in config:
                    logging.error("Invalid ensemble config format")
                    return load_default_classifier()
                
                for i, model_cfg in enumerate(config.get('models', [])):
                    model_name = model_cfg.get('name', '')
                    model_rel_path = model_cfg.get('path', '')
                    model_format = model_cfg.get('format', '')
                    weight = model_cfg.get('weight', 1.0)
                    
                    model_path = os.path.join(models_dir, model_rel_path)
                    logging.info(f"Checking model {i+1}: {model_name}, path: {model_path}, format: {model_format}")
                    
                    model_exists = False
                    if model_format == 'saved_model':
                        saved_model_pb = os.path.join(model_path, 'saved_model.pb')
                        variables_dir = os.path.join(model_path, 'variables')
                        model_exists = os.path.isdir(model_path) and os.path.exists(saved_model_pb) and os.path.exists(variables_dir)
                    else:
                        model_exists = os.path.exists(model_path)
                        
                    if not model_exists:
                        logging.warning(f"Model file not found at {model_path}. Skipping.")
                        continue
                    
                    try:
                        if model_format == 'saved_model' and model_exists:
                            ensemble.add_model(model_path, labels_path, model_name=model_name, weight=weight)
                            models_loaded += 1
                        elif model_format == 'h5' and model_exists:
                            ensemble.add_model(model_path, labels_path, model_name=model_name, weight=weight)
                            models_loaded += 1
                        elif model_format == 'tflite' and model_exists:
                            ensemble.add_model(model_path, labels_path, model_name=model_name, weight=weight)
                            models_loaded += 1
                        elif model_format == 'onnx' and model_exists:
                            ensemble.add_model(model_path, labels_path, model_name=model_name, weight=weight)
                            models_loaded += 1
                        elif model_format == 'openvino' and model_exists:
                            ensemble.add_model(model_path, labels_path, model_name=model_name, weight=weight)
                            models_loaded += 1
                    except Exception as e:
                        logging.error(f"Error adding {model_format} model {model_name}: {e}")
                
            except Exception as e:
                logging.error(f"Error loading ensemble config: {e}")
                return load_default_classifier()
                    
        elif custom_models:
            logging.info("Using custom model configuration")
            
            for model_cfg in custom_models:
                model_name = model_cfg.get('name', 'Custom Model')
                model_path = model_cfg.get('path', '')
                weight = model_cfg.get('weight', 1.0)
                
                if not os.path.exists(model_path):
                    logging.warning(f"Custom model not found: {model_path}")
                    continue
                
                try:
                    ensemble.add_model(model_path, labels_path, model_name=model_name, weight=weight)
                    models_loaded += 1
                except Exception as e:
                    logging.error(f"Error adding custom model {model_name}: {e}")
        
        if models_loaded == 0:
            logging.warning("No models were loaded for ensemble. Falling back to default classifier.")
            return load_default_classifier()
            
        ensemble.load_all_models()
        
        try:
            logging.info("Starting ensemble model warmup...")
            sample_img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.rectangle(sample_img, (50, 50), (150, 150), (0, 255, 0), -1)
            
            try:
                ensemble.predict(sample_img, threshold=0.8)
                logging.info("Warmup prediction completed successfully")
            except Exception as e:
                logging.warning(f"Warmup prediction failed: {e}")
        except Exception as e:
            logging.warning(f"Skipping model warmup due to error: {e}")
                    
        class EnsembleWrapper:
            def __init__(self, ensemble_classifier):
                self.ensemble = ensemble_classifier
                
            @property
            def models(self):
                if hasattr(self.ensemble, 'models'):
                    return self.ensemble.models
                return []

            def get_prediction(self, img, threshold=0.8):
                try:
                    # Get prediction from ensemble
                    class_id, confidence = self.ensemble.predict(img, threshold)
                    
                    # Try to get model stats for detailed predictions
                    model_stats = self.ensemble.get_model_stats()
                    
                    # Check if we have valid model stats and last predictions
                    if model_stats and len(model_stats) > 0 and 'last_predictions' in model_stats[0]:
                        predictions = model_stats[0].get('last_predictions', [])
                    else:
                        # Fallback - create a prediction list with confidence at class_id position
                        predictions = [0.0] * 8  # Assuming 8 classes
                        if 0 <= class_id < len(predictions):
                            predictions[class_id] = confidence
                    
                    # Ensure predictions is not empty
                    if not predictions:
                        predictions = [0.0] * 8
                        predictions[0] = 1.0  # Default to class 0 (Nothing)
                        class_id = 0
                    
                    logging.info(f"Prediction result: class_id={class_id}, confidence={confidence:.4f}")
                    return (predictions, class_id)
                except Exception as e:
                    logging.error(f"Prediction error: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return default prediction (Nothing class)
                    return ([1.0] + [0.0] * 7, 0)
                    
            def predict(self, img, threshold=0.8):
                """Direct passthrough to ensemble's predict method"""
                try:
                    if hasattr(self.ensemble, 'predict'):
                        return self.ensemble.predict(img, threshold)
                    else:
                        logging.error("Ensemble does not have predict method")
                        return 0, 0.0
                except Exception as e:
                    logging.error(f"Error in predict passthrough: {e}")
                    return 0, 0.0  # Default to "Nothing" with 0 confidence
                    
            def get_ensemble_stats(self):
                """Get ensemble statistics"""
                try:
                    if hasattr(self.ensemble, 'get_ensemble_stats'):
                        return self.ensemble.get_ensemble_stats()
                    elif hasattr(self.ensemble, 'get_stats'):
                        return self.ensemble.get_stats()
                    else:
                        return {
                            'model_count': len(self.models),
                            'system_metrics': {'cpu_percent': 0, 'gpu_percent': 0},
                            'avg_inference_time': 0,
                        }
                except Exception as e:
                    logging.error(f"Error getting ensemble stats: {e}")
                    return {'model_count': 0, 'error': str(e)}
                    
            def shutdown(self):
                """Shutdown the ensemble"""
                try:
                    if hasattr(self.ensemble, 'shutdown'):
                        return self.ensemble.shutdown()
                except Exception as e:
                    logging.error(f"Error during ensemble shutdown: {e}")
                    
            def set_aggregation_method(self, method):
                """Pass through method to set the aggregation method on the underlying ensemble"""
                try:
                    if hasattr(self.ensemble, 'set_aggregation_method'):
                        self.ensemble.set_aggregation_method(method)
                        logging.info(f"Set ensemble aggregation method to: {method}")
                    else:
                        logging.warning(f"Ensemble does not support setting aggregation method to: {method}")
                except Exception as e:
                    logging.error(f"Error setting aggregation method: {e}")
                return self
        
        logging.info(f"Ensemble classifier loaded with {models_loaded} models")
        return EnsembleWrapper(ensemble)
            
    except Exception as e:
        logging.error(f"Error initializing ensemble classifier: {e}")
        import traceback
        traceback.print_exc()
        return load_default_classifier()


class BaseClassifierInterface:
    """
    Base interface for classifier implementations to ensure consistent API
    """
    
    def get_prediction(self, img, threshold=0.8, draw=False) -> Tuple[List[float], int]:
        """Get prediction from classifier"""
        raise NotImplementedError("Subclasses must implement get_prediction")


class SingleClassifier(BaseClassifierInterface):
    """
    Wrapper for single model classifier to conform to common interface
    """
    
    def __init__(self, model_path, label_path):
        """Initialize the single classifier"""
        self.classifier = load_classifier(model_path, label_path)
        
    def get_prediction(self, img, threshold=0.8, draw=False) -> Tuple[List[float], int]:
        """
        Get prediction from the single classifier
        
        Args:
            img: Input image
            threshold: Confidence threshold
            draw: Whether to draw on the image
            
        Returns:
            Tuple of (predictions, class_index)
        """
        if self.classifier is None:
            return [], 0
            
        return self.classifier.getPrediction(img, draw=draw)


class EnsembleClassifierWrapper(BaseClassifierInterface):
    """
    Wrapper for ensemble classifier to conform to common interface
    """
    
    def __init__(self, ensemble_classifier):
        """Initialize with an ensemble classifier instance"""
        self.ensemble = ensemble_classifier
        
    def get_prediction(self, img, threshold=0.8, draw=False) -> Tuple[List[float], int]:
        """
        Get prediction from the ensemble classifier
        
        Args:
            img: Input image
            threshold: Confidence threshold
            draw: Whether to draw on the image (not used for ensemble)
            
        Returns:
            Tuple of (predictions array, class_index)
        """
        if self.ensemble is None:
            return [], 0
            
        class_id, confidence = self.ensemble.predict(img, threshold=threshold)
        
        # Create a compatible predictions array like the single model
        # This allows seamless integration with existing code
        predictions = [0.0] * 8  # Assuming 8 classes as in the current system
        if 0 <= class_id < len(predictions):
            predictions[class_id] = confidence
            
        return predictions, class_id


def load_default_classifier():
    """
    Load default single classifier (TF SavedModel).
    """
    models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Resources', 'Models')
    model_path = os.path.join(models_path, 'model.savedmodel')
    label_path = os.path.join(models_path, 'labels.txt')
    
    if not os.path.exists(model_path):
        logging.error(f"Default model not found at {model_path}")
        return DummyClassifier()
        
    return load_classifier(model_path, label_path)


class DummyClassifier(BaseClassifierInterface):
    """Dummy classifier that always returns a fixed prediction."""
    
    def get_prediction(self, img, threshold=0.8, draw=False) -> Tuple[List[float], int]:
        """Return a fixed prediction: class 0 with 100% confidence."""
        return ([1.0] + [0.0] * 7, 0)  # Assuming 8 classes total 