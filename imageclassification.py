#! python
from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys
from pathlib import Path

import tensorflow as tf

from tensorflow.keras import datasets, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

fitModel = True
plotOn = True
FILE_DIR = str(Path(sys.argv[0]).parent) + str(os.sep)


def cifar():
    global fitModel, plotOn
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    if plotOn:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            # The CIFAR labels happen to be arrays,
            # which is why you need the extra index
            plt.xlabel(class_names[train_labels[i][0]])
        plt.show()

    try:
        model = models.load_model(FILE_DIR+'cifar.pkl')
    except IOError as err:
        print(err)
        model = None
        fitModel = True
    if model is None:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    if fitModel:
        history = model.fit(train_images, train_labels, epochs=5,
                            validation_data=(test_images, test_labels))
        if plotOn:
            plt.plot(history.history['acc'], label='accuracy')
            plt.plot(history.history['val_acc'], label='val_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.ylim([0.5, 1])
            plt.legend(loc='lower right')
            plt.show()
        model.save(FILE_DIR+'cifar.pkl')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_acc)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def catdog():
    global fitModel, plotOn
    PATH = os.path.join(FILE_DIR,'cats_and_dogs_filtered')
    print(PATH)
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    print(train_dir)
    train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures
    num_cats_tr = len(os.listdir(train_cats_dir))
    num_dogs_tr = len(os.listdir(train_dogs_dir))

    num_cats_val = len(os.listdir(validation_cats_dir))
    num_dogs_val = len(os.listdir(validation_dogs_dir))

    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val
    batch_size = 128
    epochs = 15
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                               rotation_range=45,
                                               width_shift_range=.15,
                                               height_shift_range=.15,
                                               horizontal_flip=True,
                                               zoom_range=0.5
                                               )  # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary')
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation_dir,
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='binary')
    sample_training_images, _ = next(train_data_gen)
    if plotOn:
        plotImages(sample_training_images[:5])
    try:
        model = models.load_model('catdog.pkl')
    except IOError as err:
        print(err)
        model = None
        fitModel = True
    if model is None:
        model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            MaxPooling2D(),
            Dropout(0.2),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Dropout(0.2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    if fitModel:
        history = model.fit_generator(
            train_data_gen,
            steps_per_epoch=total_train // batch_size,
            epochs=epochs,
            validation_data=val_data_gen,
            validation_steps=total_val // batch_size
        )

        acc = history.history['acc']
        val_acc = history.history['val_acc']

        epochs_range = range(epochs)

        if plotOn:
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.ylim([0.5, 1])
            plt.title('Training and Validation Accuracy')
            plt.show()
        model.save('catdog.pkl')


if __name__ == "__main__":
    catdog()
