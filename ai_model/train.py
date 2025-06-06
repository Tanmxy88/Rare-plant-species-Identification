import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model
import argparse

def train_model(data_dir, output_model_path, class_indices_path, 
                img_size=(224, 224), batch_size=32, epochs=20, 
                fine_tune=False, lr=0.001):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Enable mixed precision
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    except Exception as e:
        print("Mixed precision not enabled:", e)

    # GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    num_classes = len(train_generator.class_indices)
    model = build_model(num_classes=num_classes, 
                       input_shape=img_size + (3,), 
                       fine_tune=fine_tune, 
                       lr=lr)

    # Save class indices
    dir_name = os.path.dirname(class_indices_path)
    if dir_name != '':
        os.makedirs(dir_name, exist_ok=True)

    with open(class_indices_path, 'w') as f:
        json.dump(train_generator.class_indices, f)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(output_model_path, monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )

    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train plant species identification model.')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to dataset directory')
    parser.add_argument('--output_model', type=str, default='plant_model.keras', 
                        help='Path to save trained model')
    parser.add_argument('--class_indices', type=str, default='class_indices.json',
                        help='Path to save class indices')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224], 
                        help='Image size (width height)')
    parser.add_argument('--fine_tune', action='store_true', 
                        help='Enable fine-tuning of top layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    train_model(
        args.data_dir,
        args.output_model,
        args.class_indices,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        fine_tune=args.fine_tune,
        lr=args.lr
    )
