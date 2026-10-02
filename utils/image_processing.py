import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Class untuk memproses gambar sebelum diklasifikasi oleh model CNN
    """
    
    def __init__(self, target_size=(150, 150)):
        self.target_size = target_size
        
    def load_and_validate_image(self, image_path):
        """
        Memuat dan memvalidasi gambar
        """
        try:
            # Buka gambar menggunakan PIL
            img = Image.open(image_path)
            
            # Konversi ke RGB jika perlu
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            return img
            
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")
    
    def preprocess_image(self, image_path):
        """
        Preprocessing gambar untuk model CNN
        HARUS SAMA PERSIS dengan yang di tester
        """
        try:
            # Step 1: Load gambar dengan PIL (sama seperti tester)
            img = Image.open(image_path).convert("RGB")
            
            # Step 2: Resize gambar ke ukuran target (sama seperti tester)
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Step 3: Konversi ke array numpy dan normalisasi (sama seperti tester)
            img_array = np.array(img) / 255.0
            
            # Step 4: Expand dimensions untuk batch (sama seperti tester)
            img_array = np.expand_dims(img_array, axis=0)
            
            logger.debug(f"Preprocessed image shape: {img_array.shape}")
            logger.debug(f"Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
            
            return img_array
            
        except Exception as e:
            raise Exception(f"Error in image preprocessing: {str(e)}")

# Instance global
image_processor = ImageProcessor()