import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16  # Reduced batch size for better generalization
EPOCHS = 50  # Increased epochs
LEARNING_RATE = 0.0001  # Reduced learning rate
WEIGHT_DECAY = 1e-4  # L2 regularization

# Update paths
csv_path = '../archive/sample_labels.csv'
image_dir = '../archive/sample/images'  # Updated path to include 'images' subdirectory

def load_and_preprocess_data(csv_path, image_dir):
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Verify required columns exist
        required_columns = ['Image Index', 'Finding Labels']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")
        
        # Process labels
        df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))
        
        # Initialize MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(df['Finding Labels'])
        
        # Create image paths and verify they exist
        image_paths = []
        valid_indices = []
        for idx, img_name in enumerate(df['Image Index']):
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                image_paths.append(img_path)
                valid_indices.append(idx)
            else:
                print(f"Warning: Image not found: {img_path}")
        
        # Filter labels to match valid images
        labels = labels[valid_indices]
        
        # Calculate class weights
        class_weights = {}
        for i in range(labels.shape[1]):
            class_weights[i] = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(labels[:, i]),
                y=labels[:, i]
            )[1]
        
        return image_paths, labels, mlb.classes_, class_weights
    except Exception as e:
        print(f"Error in load_and_preprocess_data: {str(e)}")
        raise

def custom_generator(generator, dataframe, batch_size, class_weights=None):
    gen = generator.flow_from_dataframe(
        dataframe,
        x_col='filename',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode='raw',
        shuffle=True
    )
    while True:
        batch_x, batch_y = next(gen)
        # Convert string labels back to numpy arrays
        batch_y = np.array([eval(y) for y in batch_y])
        
        # Apply class weights if provided
        if class_weights is not None:
            sample_weights = np.zeros(batch_y.shape[0])
            for i in range(batch_y.shape[1]):
                sample_weights += batch_y[:, i] * class_weights[i]
            yield batch_x, batch_y, sample_weights
        else:
            yield batch_x, batch_y

def create_data_generators(image_paths, labels, test_size=0.2, val_size=0.1, class_weights=None):
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels, test_size=test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )
        
        # Create dataframes with both filenames and labels
        train_df = pd.DataFrame({
            'filename': X_train,
            'label': [str(label.tolist()) for label in y_train]
        })
        val_df = pd.DataFrame({
            'filename': X_val,
            'label': [str(label.tolist()) for label in y_val]
        })
        test_df = pd.DataFrame({
            'filename': X_test,
            'label': [str(label.tolist()) for label in y_test]
        })
        
        # Enhanced data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = custom_generator(train_datagen, train_df, BATCH_SIZE, class_weights)
        val_generator = custom_generator(val_test_datagen, val_df, BATCH_SIZE)
        test_generator = custom_generator(val_test_datagen, test_df, BATCH_SIZE)
        
        return train_generator, val_generator, test_generator, y_train, y_val, y_test
    except Exception as e:
        print(f"Error in create_data_generators: {str(e)}")
        raise

def create_model(input_shape, num_classes):
    try:
        # Use EfficientNetB4 as base model
        base_model = applications.EfficientNetB4(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        
        # Unfreeze some layers for fine-tuning
        for layer in base_model.layers[-20:]:
            layer.trainable = True
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='sigmoid')
        ])
        
        return model
    except Exception as e:
        print(f"Error in create_model: {str(e)}")
        raise

def train_model():
    try:
        # Load and preprocess data
        image_paths, labels, class_names, class_weights = load_and_preprocess_data(csv_path, image_dir)
        
        print(f"Loaded {len(image_paths)} images with {len(class_names)} classes")
        print(f"Class names: {class_names}")
        
        # Create data generators with class weights
        train_generator, val_generator, test_generator, y_train, y_val, y_test = create_data_generators(
            image_paths, labels, class_weights=class_weights
        )
        
        # Create and compile model
        model = create_model((*IMAGE_SIZE, 3), len(class_names))
        
        # Use Adam optimizer with learning rate scheduling
        optimizer = optimizers.Adam(
            learning_rate=LEARNING_RATE,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )
        
        # Print model summary
        model.summary()
        
        # Enhanced callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        # Train the model without class_weight parameter
        history = model.fit(
            train_generator,
            steps_per_epoch=len(y_train) // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_generator,
            validation_steps=len(y_val) // BATCH_SIZE,
            callbacks=callbacks
        )
        
        # Load best model
        model = tf.keras.models.load_model('best_model.h5')
        
        # Evaluate the model
        test_loss, test_accuracy, test_auc, test_precision, test_recall = model.evaluate(
            test_generator,
            steps=len(y_test) // BATCH_SIZE
        )
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        
        # Save the final model
        model.save('chest_xray_model.h5')
        
        # Save class names
        np.save('class_names.npy', class_names)
        
        return model, history
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        model, history = train_model()
    except Exception as e:
        print(f"Error during training: {str(e)}") 