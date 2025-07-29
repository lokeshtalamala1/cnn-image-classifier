import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from wandb.keras import WandbCallback
import wandb

def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

def load_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    return X_train, y_train_cat, X_test, y_test_cat

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on Fashion MNIST")
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Training batch size')
    parser.add_argument('--optimizer', '-o', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'], help='Optimizer to use')
    parser.add_argument('--wandb_project', '-wp', type=str, default='cnn-image-classifier', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', '-we', type=str, default='cs24m023', help='WandB entity name')
    return parser.parse_args()

def main():
    args = parse_args()

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    X_train, y_train, X_test, y_test = load_data()
    model = get_model()

    model.compile(optimizer=args.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[WandbCallback()]
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    wandb.finish()

if __name__ == "__main__":
    main()
