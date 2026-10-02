import tensorflow as tf
import numpy as np
import pickle
from datetime import datetime

def create_kaos_dummy_model():
    """Create dummy model untuk Kaos classification"""
    print("🎵 Creating Kaos Classification Dummy Model...")
    
    # Simple model untuk 4 classes
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(150, 150, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 classes untuk Kaos
    ])
    
    # Compile untuk multi-class classification
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save model
    model.save('model/model_cnn.h5')
    print("✅ Kaos Model saved")
    
    # Save proper class mapping untuk Kaos
    class_mapping = {
        'class_names': ['cc', 'cvc', 'polyester', 'tc'],
        'class_indices': {
            'cc': 0,
            'cvc': 1,
            'polyester': 2,
            'tc': 3
        },
        'timestamp': datetime.now().isoformat(),
        'input_shape': (150, 150, 3),
        'classes_count': 4,
        'model_type': 'multi_class_kaos'
    }
    
    with open('model/class_names.pkl', 'wb') as f:
        pickle.dump(class_mapping, f)
    
    print("✅ Kaos Class mapping saved:")
    for i, name in enumerate(class_mapping['class_names']):
        print(f"   {i}: {name}")
    
    # Test the model
    print("🧪 Testing dummy model...")
    test_input = np.random.random((1, 150, 150, 3))
    prediction = model.predict(test_input, verbose=0)
    
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    print(f"✅ Test prediction: {class_mapping['class_names'][predicted_class]}")
    print(f"✅ Confidence: {confidence:.4f}")
    print("🎉 Kaos dummy model created successfully!")

if __name__ == "__main__":
    create_kaos_dummy_model()