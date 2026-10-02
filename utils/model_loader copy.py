import tensorflow as tf
import numpy as np
import pickle
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Class untuk memuat dan mengelola model CNN
    """
    
    def __init__(self, model_path, class_names_path=None):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names = ['Berkualitas Buruk', 'Berkualitas Baik']
        self.load_time = None
        
    def load_model(self):
        """
        Memuat model CNN dari file .h5
        """
        try:
            logger.info(f"Loading model from {self.model_path}")
            start_time = datetime.now()
            
            # Load model TensorFlow
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Load class names jika tersedia
            if self.class_names_path:
                with open(self.class_names_path, 'rb') as f:
                    self.class_names = pickle.load(f)
            
            self.load_time = datetime.now() - start_time
            logger.info(f"Model loaded successfully in {self.load_time}")
            
            # Print model summary
            logger.info("Model Architecture:")
            self.model.summary(print_fn=logger.info)
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, processed_image):
        """
        Melakukan prediksi menggunakan model yang sudah dimuat - DEBUG VERSION
        """
        if self.model is None:
            raise Exception("Model belum dimuat")
        
        try:
            print(f"🔍 Model summary:")
            self.model.summary()
            
            print(f"🔍 Input shape: {processed_image.shape}")
            print(f"🔍 Input dtype: {processed_image.dtype}")
            
            # Lakukan prediksi
            predictions = self.model.predict(processed_image, verbose=1)
            
            print(f"🎯 RAW PREDICTION OUTPUT:")
            print(f"   Shape: {predictions.shape}")
            print(f"   Type: {type(predictions)}")
            print(f"   Values: {predictions}")
            print(f"   First element: {predictions[0]}")
            
            # Debug class names
            print(f"🔍 AVAILABLE CLASS NAMES:")
            print(f"   Class names: {self.class_names}")
            print(f"   Number of classes: {len(self.class_names)}")
            print(f"   Available indices: {list(range(len(self.class_names)))}")
            
            # Handle different output types
            if len(predictions.shape) == 1:
                # Output: [0.75] - binary classification
                confidence = float(predictions[0])
                print(f"🔍 Binary output detected: {confidence}")
                
                if confidence > 0.5:
                    class_index = 1
                    class_name = "Berkualitas Baik"
                    final_confidence = confidence
                else:
                    class_index = 0  
                    class_name = "Berkualitas Buruk"
                    final_confidence = 1 - confidence
                    
            elif predictions.shape[1] == 1:
                # Output: [[0.75]] - binary with batch dimension
                confidence = float(predictions[0][0])
                print(f"🔍 Binary with batch detected: {confidence}")
                
                if confidence > 0.5:
                    class_index = 1
                    class_name = "Berkualitas Baik"
                    final_confidence = confidence
                else:
                    class_index = 0
                    class_name = "Berkualitas Buruk" 
                    final_confidence = 1 - confidence
                    
            else:
                # Output: [[0.2, 0.8]] - multi-class
                class_index = int(np.argmax(predictions[0]))
                confidence = float(np.max(predictions[0]))
                print(f"🔍 Multi-class output detected: index={class_index}, confidence={confidence}")
                
                # Validate class index
                if class_index >= len(self.class_names):
                    raise Exception(f"Class index {class_index} out of range. Available: 0-{len(self.class_names)-1}")
                
                class_name = self.class_names[class_index]
                final_confidence = confidence
            
            print(f"✅ FINAL PREDICTION:")
            print(f"   Class index: {class_index}")
            print(f"   Class name: {class_name}") 
            print(f"   Confidence: {final_confidence:.4f}")
            
            # Calculate probabilities
            if len(predictions.shape) == 1 or predictions.shape[1] == 1:
                # Binary classification
                prob_buruk = 1 - final_confidence
                prob_baik = final_confidence
                class_probabilities = {
                    "Berkualitas Buruk": float(prob_buruk),
                    "Berkualitas Baik": float(prob_baik)
                }
            else:
                # Multi-class classification  
                class_probabilities = {
                    self.class_names[i]: float(prob) 
                    for i, prob in enumerate(predictions[0])
                }
            
            return {
                'class_name': class_name,
                'class_index': class_index,
                'confidence': confidence,
                'final_confidence': final_confidence * 100,
                'probabilities': class_probabilities,
                'raw_predictions': predictions.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR in prediction:")
            print(f"   Error: {str(e)}")
            print(f"   Class names: {self.class_names}")
            print(f"   Class names length: {len(self.class_names)}")
            raise
    
    def get_model_info(self):
        """
        Mendapatkan informasi tentang model yang dimuat
        """
        if self.model is None:
            return {"status": "Model not loaded"}
        
        return {
            "status": "Model loaded",
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "layers_count": len(self.model.layers),
            "load_time": str(self.load_time),
            "class_names": self.class_names
        }