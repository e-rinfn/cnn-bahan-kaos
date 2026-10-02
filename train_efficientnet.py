import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pickle
from datetime import datetime
import os

# Matikan oneDNN warnings jika mengganggu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"TensorFlow version: {tf.__version__}")

class EfficientNetB0Classifier:
    def __init__(self, data_dir='dataset', img_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.base_model = None
        self.history = None
        self.class_names = ['cc', 'cvc', 'polyester', 'tc']
        
    def setup_data_generators(self):
        """
        Setup data generators dengan augmentasi
        """
        # Data augmentation untuk training
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.9, 1.1]
        )

        val_test_datagen = ImageDataGenerator()
        # Validation dan test (hanya rescale)
        # val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load data
        self.train_generator = train_datagen.flow_from_directory(
            f'{self.data_dir}/train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        self.validation_generator = val_test_datagen.flow_from_directory(
            f'{self.data_dir}/validation',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            f'{self.data_dir}/test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"\n✅ Data generators setup complete")
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        print(f"Classes: {self.train_generator.class_indices}")
        
    def build_model(self, freeze_base=True):
        """
        Build model with EfficientNet B0 from Keras Applications
        """
        print("\n🔧 Building EfficientNet B0 model...")
        
        # Load EfficientNet B0 dengan weights ImageNet
        # Ini akan mendownload weights dari server Keras (bukan GitHub)
        self.base_model = applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',  # Akan download dari keras.io
            input_shape=(self.img_size[0], self.img_size[1], 3)
        )
        
        # Freeze base model if specified
        if freeze_base:
            self.base_model.trainable = False
            print("🔒 Base model frozen (transfer learning mode)")
        else:
            print("🔓 Base model trainable (fine-tuning mode)")
        
        # Build complete model
        inputs = tf.keras.Input(
            shape=(self.img_size[0], self.img_size[1], 3)
        )

        x = self.base_model(inputs, training=False)

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(
            4,
            activation='softmax'
        )(x)

        self.model = tf.keras.Model(inputs, outputs)        
        
        print("\n✅ EfficientNet B0 model built successfully")
        print("\n📊 Model Summary:")
        self.model.summary()
        
    def compile_model(self, learning_rate=0.001):
        """
        Compile model
        """
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"\n✅ Model compiled with learning rate: {learning_rate}")

    def train_model(self, epochs=50):
    # def train_model(self, epochs=2):
        """
        Train model
        """
        # Callbacks
        # callbacks = [
        #     EarlyStopping(
        #         monitor='val_loss',
        #         patience=10,
        #         restore_best_weights=True,
        #         verbose=1
        #     ),
        #     ModelCheckpoint(
        #         'model/best_efficientnet_model.h5',
        #         monitor='val_accuracy',
        #         save_best_only=True,
        #         verbose=1
        #     ),
        #     ReduceLROnPlateau(
        #         monitor='val_loss',
        #         factor=0.2,
        #         patience=5,
        #         min_lr=1e-6,
        #         verbose=1
        #     )
        # ]

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),

            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        
        # Training
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        
    def evaluate_model(self):
        """
        Evaluate model on test set
        """
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION ON TEST SET")
        print("="*60)
        
        # Test evaluation
        test_results = self.model.evaluate(self.test_generator, verbose=0)
        metrics = ['Loss', 'Accuracy']

        print("\n📈 Test Metrics:")
        print("-" * 40)
        for name, value in zip(metrics, test_results):
            print(f"{name:12}: {value:.4f}")
        
        # Predictions
        print("\n🔄 Generating predictions...")
        predictions = self.model.predict(self.test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        # Classification report
        print("\n📋 Classification Report:")
        print("-" * 40)
        print(
            classification_report(
                true_classes,
                predicted_classes,
                target_names=self.class_names,
                zero_division=0
            )
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - EfficientNet B0', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        os.makedirs('model', exist_ok=True)
        plt.savefig('model/efficientnet_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate accuracy per class
        print("\n📊 Per-class Accuracy:")
        print("-" * 40)
        for i, class_name in enumerate(self.class_names):
            mask = (true_classes == i)
            class_acc = np.mean(predicted_classes[mask] == i)
            print(f"{class_name:20}: {class_acc:.2%}")
        
        return test_results
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('EfficientNet B0 Training History', fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], 'b-', label='Train', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(self.history.history['loss'], 'b-', label='Train', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model/efficientnet_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_model(self):
        """
        Save model and class mapping
        """
        os.makedirs('model', exist_ok=True)
        
        # Use TensorFlow checkpoints because Keras 2.10 cannot reliably restore
        # EfficientNet weights from legacy HDF5 files.
        model_paths = [
            'model/efficientnet_kaos.h5',
            'model/best_efficientnet_model.h5'
        ]
        for model_path in model_paths:
            checkpoint_path = model_path[:-3] + '.ckpt'
            self.model.save_weights(checkpoint_path)
            with open(model_path, 'w', encoding='ascii') as model_file:
                model_file.write(f'TensorFlow checkpoint: {checkpoint_path}\n')
            print(f"\n✅ Model weights saved: {checkpoint_path}")
            print(f"✅ Model entry saved: {model_path}")
        
        # Save class mapping
        class_mapping = {
            'class_names': self.class_names,
            'class_indices': self.train_generator.class_indices,
            'timestamp': datetime.now().isoformat(),
            'input_shape': (self.img_size[0], self.img_size[1], 3),
            'classes_count': len(self.class_names),
            'model_type': 'EfficientNetB0_Keras_Applications',
            'model_architecture': 'EfficientNet B0 from Keras Applications with custom top layers',
            'trainable_params': self.model.count_params()
        }
        
        mapping_paths = [
            'model/efficientnet_class_names.pkl',
            'model/best_efficientnet_model_class_names.pkl'
        ]
        for mapping_path in mapping_paths:
            with open(mapping_path, 'wb') as f:
                pickle.dump(class_mapping, f)
            print(f"✅ Class mapping saved: {mapping_path}")
        
        # Save training history
        history_path = 'model/efficientnet_training_history.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(self.history.history, f)
        print(f"✅ Training history saved: {history_path}")
        
        # Save model summary
        summary_path = 'model/efficientnet_model_summary.txt'
        with open(summary_path, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"✅ Model summary saved: {summary_path}")

def main():
    print("="*60)
    print("🎯 KAOS CLASSIFIER WITH EFFICIENTNET B0")
    print("="*60)
    
    # Initialize classifier
    classifier = EfficientNetB0Classifier(
        data_dir='dataset',
        img_size=(224, 224),  # EfficientNet B0 optimal size
        batch_size=32
    )
    
    try:
        # Setup data
        classifier.setup_data_generators()
        
        # Build model
        classifier.build_model(freeze_base=True)
        
        # Compile model
        classifier.compile_model(learning_rate=0.001)
        
        # Train model
        classifier.train_model(epochs=50)
        # classifier.train_model(epochs=2)
        
        # Evaluate model
        classifier.evaluate_model()
        
        # Plot training history
        classifier.plot_training_history()
        
        # Save model
        classifier.save_model()
        
        print("\n" + "="*60)
        print("🎉 EFFICIENTNET B0 MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()