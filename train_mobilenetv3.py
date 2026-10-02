import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import applications, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MobileNetV3Classifier:
    def __init__(self, data_dir='dataset', img_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.base_model = None
        self.history = None
        self.class_names = ['cc', 'cvc', 'polyester', 'tc']

    def setup_data_generators(self):
        train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2],
            preprocessing_function=applications.mobilenet_v3.preprocess_input
        )
        val_test_datagen = ImageDataGenerator(
            preprocessing_function=applications.mobilenet_v3.preprocess_input
        )

        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            color_mode='rgb'
        )
        self.validation_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'validation'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            color_mode='rgb'
        )
        self.test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            color_mode='rgb'
        )
        self.class_names = list(self.train_generator.class_indices.keys())
        print(f'Training samples: {self.train_generator.samples}')
        print(f'Validation samples: {self.validation_generator.samples}')
        print(f'Test samples: {self.test_generator.samples}')
        print(f'Classes: {self.train_generator.class_indices}')

    def build_model(self, freeze_base=True, variant='large'):
        backbone = (
            applications.MobileNetV3Small
            if variant.lower() == 'small'
            else applications.MobileNetV3Large
        )
        self.base_model = backbone(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.img_size, 3),
            include_preprocessing=True
        )
        self.base_model.trainable = not freeze_base

        inputs = tf.keras.Input(shape=(*self.img_size, 3))
        features = self.base_model(inputs, training=False)
        features = layers.GlobalAveragePooling2D()(features)
        features = layers.Dropout(0.3)(features)
        features = layers.Dense(256, activation='relu')(features)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(0.4)(features)
        outputs = layers.Dense(len(self.class_names), activation='softmax')(features)
        self.model = tf.keras.Model(inputs, outputs)
        print(f'MobileNetV3 {variant.capitalize()} model built successfully.')
        self.model.summary()

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )

    # def train_model(self, epochs=2):
    def train_model(self, epochs=50):
        os.makedirs('model', exist_ok=True)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint(
                'model/best_mobilenetv3_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
        ]
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

    def plot_training_history(self):
        if self.history is None:
            print('No training history available')
            return

        history = self.history.history
        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        figure.suptitle('MobileNetV3 Training History', fontsize=16, fontweight='bold')

        axes[0].plot(history['accuracy'], label='Train', linewidth=2)
        axes[0].plot(history['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history['loss'], label='Train', linewidth=2)
        axes[1].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        figure.tight_layout()
        figure.savefig('model/mobilenetv3_training_history.png', dpi=300, bbox_inches='tight')
        plt.close(figure)
        print('Training history graph saved: model/mobilenetv3_training_history.png')

    def evaluate_model(self):
        results = self.model.evaluate(self.test_generator, verbose=0)
        print(dict(zip(['loss', 'accuracy', 'precision', 'recall', 'auc'], results)))
        predictions = self.model.predict(self.test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))

        matrix = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix - MobileNetV3')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('model/mobilenetv3_confusion_matrix.png', dpi=300)
        plt.close()
        return results

    def save_model(self):
        os.makedirs('model', exist_ok=True)
        model_path = 'model/mobilenetv3_kaos.h5'
        mapping_path = 'model/mobilenetv3_class_names.pkl'
        self.model.save(model_path)
        with open(mapping_path, 'wb') as file:
            pickle.dump({
                'class_names': self.class_names,
                'class_indices': self.train_generator.class_indices,
                'timestamp': datetime.now().isoformat(),
                'input_shape': (*self.img_size, 3),
                'model_type': 'MobileNetV3Large_Keras_Applications'
            }, file)
        print(f'Model saved: {model_path}')
        print(f'Class mapping saved: {mapping_path}')

        # Save model summary
        summary_path = 'model/mobilenetv3_model_summary.txt'
        with open(summary_path, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"✅ Model summary saved: {summary_path}")


def main():
    print('KAOS CLASSIFIER WITH MOBILENETV3')
    classifier = MobileNetV3Classifier()
    classifier.setup_data_generators()
    classifier.build_model(freeze_base=True, variant='large')
    classifier.compile_model(learning_rate=0.001)
    # classifier.train_model(epochs=2)
    classifier.train_model(epochs=50)
    classifier.plot_training_history()
    classifier.evaluate_model()
    classifier.save_model()
    print('MOBILENETV3 TRAINING COMPLETED')


if __name__ == '__main__':
    main()
