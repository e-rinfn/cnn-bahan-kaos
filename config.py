import os
from datetime import datetime

class Config:
    """Configuration class for the application"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'klasifikasi-kaos-secret')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Upload Configuration
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB per file
    
    # Model Configuration
    MODEL_PATH = 'model/model_cnn.h5'
    CLASS_NAMES = ['cc', 'cvc', 'polyester', 'tc']
    IMAGE_SIZE = (150, 150)  # Sesuai input model CNN
    
    # API Configuration
    API_HOST = '0.0.0.0'
    API_PORT = 5000
    DEBUG = True
    
    # Logging Configuration
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'

class ProductionConfig(Config):
    DEBUG = False
    API_HOST = '0.0.0.0'
    API_PORT = 5000

class DevelopmentConfig(Config):
    DEBUG = True
    API_HOST = 'localhost'
    API_PORT = 5000