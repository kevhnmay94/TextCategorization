#! python
from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys
from pathlib import Path
import time

import tensorflow as tf
from tensorflow.keras.applications import Xception, xception, MobileNetV2, mobilenet_v2, ResNet50, resnet50
from tensorflow.keras import datasets, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

fitModel = False
plotOn = False
FILE_DIR = str(Path(sys.argv[0]).parent) + str(os.sep)

image_path = "cats_and_dogs_filtered/square.jpg"


def cifar():
    global fitModel, plotOn
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    if plotOn:
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
        model = models.load_model(FILE_DIR + 'cifar.pkl')
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
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
        datagen.fit(train_images)
        history = model.fit_generator(datagen.flow(train_images, train_labels), epochs=5,
                                      validation_data=(test_images, test_labels))
        if plotOn:
            plt.plot(history.history['acc'], label='accuracy')
            plt.plot(history.history['val_acc'], label='val_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.ylim([0.5, 1])
            plt.legend(loc='lower right')
            plt.show()
        model.save(FILE_DIR + 'cifar.pkl')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_acc)

    img = load_img(image_path, target_size=(32, 32))
    plt.imshow(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    result = model.predict_proba(img)
    arr = np.array(result[0])
    print(result)
    top_three = arr.argsort()[::-1][:3]
    label = class_names[np.argmax(result[0])]
    print(label)
    plt.title(label)
    plt.show()


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
    PATH = os.path.join(FILE_DIR, 'cats_and_dogs_filtered')
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

    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    plt.imshow(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict_classes(img)
    print(result)
    if result[0][0] == 0:
        print('cat')
        plt.title('cat')
    elif result[0][0] == 1:
        print('dog')
        plt.title('dog')
    plt.show()


def imagenet(model_name):
    start_time = time.process_time()
    if model_name == 'xception':
        model = Xception(weights='imagenet')
        preprocess_input = xception.preprocess_input
        decode_predictions = xception.decode_predictions
        size = (299, 299)
    elif model_name == 'mobilenet':
        model = MobileNetV2(weights='imagenet')
        preprocess_input = mobilenet_v2.preprocess_input
        decode_predictions = mobilenet_v2.decode_predictions
        size = (224, 224)
    elif model_name == 'resnet50':
        model = ResNet50(weights='imagenet')
        preprocess_input = resnet50.preprocess_input
        decode_predictions = resnet50.decode_predictions
        size = (224, 224)
    else:
        print("No model found")
        return
    img = load_img(image_path, target_size=size)
    plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    end_time = time.process_time()
    elapsed = end_time - start_time
    print("Process time: ", f'{elapsed:.3f}s')
    print('Predictions:')
    for a in decode_predictions(pred, top=5)[0]:
        print(a[1], ": ", f'{a[2] * 100:.2f}%')


if __name__ == "__main__":
    imagenet('xception')
    imagenet('mobilenet')
    imagenet('resnet50')
