import tensorflow as tf
import numpy as np
import pickle
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Class untuk memuat dan mengelola model CNN untuk Jalan Rusak
    """
    
    def __init__(self, model_path, class_names_path=None):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model = None
        self.class_names = ['cc', 'cvc', 'polyester', 'tc']  # Default untuk 4 kelas bahan kaos
        self.class_indices = {}
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
                try:
                    with open(self.class_names_path, 'rb') as f:
                        data = pickle.load(f)

                    # Handle berbagai format data
                    if isinstance(data, dict):
                        if "class_names" in data:
                            self.class_names = data["class_names"]
                        if "class_indices" in data:
                            self.class_indices = data["class_indices"]
                    elif isinstance(data, list):
                        self.class_names = data
                    
                    logger.info(f"Loaded class names: {self.class_names}")
                    
                except Exception as e:
                    logger.warning(f"Could not load class names from pickle: {e}")
                    logger.info(f"Using default class names: {self.class_names}")
            
            # Pastikan jumlah class names sesuai dengan output model
            if self.model and len(self.class_names) != self.model.output_shape[-1]:
                logger.warning(f"Class names count ({len(self.class_names)}) doesn't match model output ({self.model.output_shape[-1]})")
                # Generate default class names
                self.class_names = [f"class_{i}" for i in range(self.model.output_shape[-1])]
            
            self.load_time = datetime.now() - start_time
            logger.info(f"✅ Model loaded successfully in {self.load_time}")
            logger.info(f"📊 Model expects {self.model.output_shape[-1]} classes")
            logger.info(f"📋 Class names: {self.class_names}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, processed_image):
        """
        Melakukan prediksi menggunakan model yang sudah dimuat
        """
        if self.model is None:
            raise Exception("Model belum dimuat")
        
        try:
            # Validasi input shape
            expected_shape = self.model.input_shape[1:]
            actual_shape = processed_image.shape[1:]
            
            if expected_shape != actual_shape:
                logger.warning(f"Input shape mismatch! Expected {expected_shape}, got {actual_shape}")
            
            # Lakukan prediksi
            predictions = self.model.predict(processed_image, verbose=0)
            
            logger.debug(f"Raw predictions shape: {predictions.shape}")
            logger.debug(f"Raw predictions: {predictions[0]}")
            
            # Handle prediksi (selalu gunakan argmax untuk multi-class)
            num_classes = predictions.shape[1]
            class_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][class_index])
            
            # Ambil nama kelas
            if class_index < len(self.class_names):
                class_name = self.class_names[class_index]
            else:
                class_name = f"class_{class_index}"
                logger.warning(f"Class index {class_index} out of range for class_names")
            
            # Hitung probabilitas untuk semua kelas
            class_probabilities = {}
            for i, prob in enumerate(predictions[0]):
                if i < len(self.class_names):
                    class_probabilities[self.class_names[i]] = float(prob)
                else:
                    class_probabilities[f"class_{i}"] = float(prob)
            
            logger.info(f"🎯 Prediction: {class_name} (confidence: {confidence:.4f})")
            
            # Log top 3 predictions
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            logger.debug("Top 3 predictions:")
            for idx in top_indices:
                name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
                prob = predictions[0][idx]
                logger.debug(f"  {name}: {prob:.4f}")
            
            return {
                'class_name': class_name,
                'class_index': class_index,
                'confidence': confidence,
                'final_confidence': confidence * 100,  # Dalam persen
                'probabilities': class_probabilities,
                'timestamp': datetime.now().isoformat(),
                'raw_predictions': predictions[0].tolist()  # Untuk debugging
            }
            
        except Exception as e:
            logger.error(f"❌ Error during prediction: {str(e)}")
            logger.error(f"Input shape: {processed_image.shape}")
            logger.error(f"Available class names: {self.class_names}")
            raise
    
    def get_model_info(self):
        """
        Mendapatkan informasi tentang model yang dimuat
        """
        if self.model is None:
            return {"status": "Model not loaded"}
        
        return {
            "status": "Model loaded",
            "input_shape": list(self.model.input_shape),
            "output_shape": list(self.model.output_shape),
            "num_classes": self.model.output_shape[-1],
            "layers_count": len(self.model.layers),
            "load_time": str(self.load_time),
            "class_names": self.class_names,
            "class_indices": self.class_indices
        }