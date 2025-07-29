import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
import wandb
from wandb.keras import WandbCallback

# Data preprocessing
def preprocess_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Resize to 224x224x3 (VGG16 expected input)
    X_train = np.stack([np.repeat(tf.image.resize_with_pad(tf.convert_to_tensor(img[..., np.newaxis], dtype=tf.float32), 224, 224).numpy(), 3, axis=-1) for img in X_train])
    X_test = np.stack([np.repeat(tf.image.resize_with_pad(tf.convert_to_tensor(img[..., np.newaxis], dtype=tf.float32), 224, 224).numpy(), 3, axis=-1) for img in X_test])

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test

# Build transfer learning model
def build_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

def main(args):
    wandb.init(project="cnn-image-classifier", entity="cs24m023", config=vars(args))

    X_train, y_train, X_test, y_test = preprocess_data()

    model = build_model()
    model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[WandbCallback()]
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    wandb.log({'Test Accuracy': acc, 'Test Loss': loss})
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    main(args)
