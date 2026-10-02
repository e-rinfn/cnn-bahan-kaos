# create_simple_model.py
import tensorflow as tf
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path

def create_simple_model():
    """Create a simple 4-class softmax model for testing"""
    print("🔧 Creating simple 4-class model (softmax)...")

    # Model 3 class
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 output classes
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # ✔ multi-class
        metrics=['accuracy']
    )

    # Pastikan folder model ada
    Path("model").mkdir(exist_ok=True)

    # Save model
    model.save('model/model_cnn.h5')
    print("✅ Model saved successfully (4-class softmax)")

    # Create correct class mapping
    class_names = ['cc', 'cvc', 'polyester', 'tc']

    class_mapping = {
        'class_names': class_names,
        'class_indices': {name: i for i, name in enumerate(class_names)},
        'timestamp': datetime.now().isoformat(),
        'input_shape': (150, 150, 3),
        'classes_count': 4,
        'note': 'Simple working 4-class model for testing'
    }

    with open('model/class_names.pkl', 'wb') as f:
        pickle.dump(class_mapping, f)

    print("✅ class_names.pkl generated correctly")

    # Dummy test
    print("🧪 Testing dummy prediction with softmax...")
    dummy_data = np.random.random((1, 150, 150, 3))
    prediction = model.predict(dummy_data, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction))

    print(f"🔍 Predicted: {predicted_class} (Confidence: {confidence:.4f})")

if __name__ == "__main__":
    create_simple_model()
