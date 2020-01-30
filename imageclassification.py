#! python
from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys
from pathlib import Path
import time

import tensorflow as tf
from tensorflow.keras.applications import Xception, xception, MobileNetV2, mobilenet_v2, ResNet50, resnet50, \
    InceptionV3, inception_v3
from tensorflow.keras import datasets, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fitModel = False
plotOn = False
verbose = True
image_file = None
train_path = 'konten/kategori'
image_path = 'konten/image'
model_name = 'xception'
is_csv_to_link = False
is_link_to_csv = False
remove_model = False
is_add_to_csv = False
predict = True
cat_result = None


def vprint(*data):
    if verbose:
        print(*data)


for i, s in enumerate(sys.argv[1:]):
    if s[:2] == '--':
        arg = s[2:]
        if arg == 'train-path':
            train_path = sys.argv[i + 2]
        elif arg == 'image':
            image_file = sys.argv[i + 2]
        elif arg == 'model':
            model_name = sys.argv[i + 2]
        elif arg == 'csv-to-link':
            is_csv_to_link = True
        elif arg == 'link-to-csv':
            is_link_to_csv = True
        elif arg == 'image-path':
            image_path = sys.argv[i + 2]
        elif arg == 'predict':
            predict = True
        elif arg == 'no-predict':
            predict = False
        elif arg == 'add-to-csv':
            is_add_to_csv = True
        elif arg == 'top-category':
            import ast
            cat_result = ast.literal_eval(sys.argv[i+2])

    elif s[0] == '-':
        for arg in s[1:]:
            if 'v' == arg:
                verbose = True
            elif 'q' == arg:
                verbose = False
            elif 'F' == arg:
                fitModel = True
            elif 'f' == arg:
                fitModel = False
            elif 'P' == arg:
                plotOn = True
            elif 'p' == arg:
                plotOn = False
            elif 'r' == arg:
                remove_model = True

FILE_DIR = str(Path(sys.argv[0]).parent) + str(os.sep)

if image_file is None:
    image_file = "cats_and_dogs_filtered/pot4.jpg"


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
        vprint(err)
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
    vprint(test_acc)

    img = load_img(image_file, target_size=(32, 32))
    plt.imshow(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    result = model.predict_proba(img)
    arr = np.array(result[0])
    vprint(result)
    top_three = arr.argsort()[::-1][:3]
    label = class_names[np.argmax(result[0])]
    vprint(label)
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


def subdir(parent: str):
    category = [x[1] for x in os.walk(parent)][0]
    b = sorted(category)
    return b


def catdog():
    global fitModel, plotOn
    PATH = os.path.join(FILE_DIR, 'cats_and_dogs_filtered')
    vprint(PATH)
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    vprint(train_dir)
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
        vprint(err)
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

    img = load_img(image_file, target_size=(IMG_HEIGHT, IMG_WIDTH))
    plt.imshow(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict_classes(img)
    vprint(result)
    if result[0][0] == 0:
        vprint('cat')
        plt.title('cat')
    elif result[0][0] == 1:
        vprint('dog')
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
        vprint("No model found")
        return
    img = load_img(image_file, target_size=size)
    plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    end_time = time.process_time()
    elapsed = end_time - start_time
    vprint("Process time: ", f'{elapsed:.3f}s')
    vprint('Predictions:')
    for a in decode_predictions(pred, top=5)[0]:
        vprint(a[1], ": ", f'{a[2] * 100:.2f}%')


def asdf(path, model_name, img_path, remove_model):
    global fitModel, plotOn
    if os.path.isabs(path):
        PATH = path
    else:
        PATH = os.path.join(FILE_DIR, path)
    vprint(PATH)
    category = subdir(str(PATH))
    total_item = 0
    for c in category:
        dir = os.path.join(PATH, c)
        num_dir = len(os.listdir(dir))
        total_item += num_dir
    batch_size = 10
    epochs = 15
    if model_name == 'xception' or model_name == 'inception':
        IMG_HEIGHT = 299
        IMG_WIDTH = 299
        size = (299, 299)
        if model_name == 'xception':
            preprocess_input = xception.preprocess_input
            decode_predictions = xception.decode_predictions
        else:
            preprocess_input = inception_v3.preprocess_input
            decode_predictions = inception_v3.decode_predictions
    else:
        IMG_HEIGHT = 224
        IMG_WIDTH = 224
        size = (224, 224)
        if model_name == 'resnet50':
            preprocess_input = resnet50.preprocess_input
            decode_predictions = resnet50.decode_predictions
        elif model_name == 'mobilenet':
            preprocess_input = mobilenet_v2.preprocess_input
            decode_predictions = mobilenet_v2.decode_predictions
    try:
        if model_name == 'xception':
            model = models.load_model(FILE_DIR + 'asdf.pkl')
    except IOError as err:
        vprint(err)
        model = None
        fitModel = True
    if model is None or remove_model:
        base_model = None
        if model_name == 'xception':
            base_model = Xception(weights='imagenet', include_top=False)
        elif model_name == 'mobilenet':
            base_model = MobileNetV2(weights='imagenet', include_top=False)
        elif model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False)
        elif model_name == 'inception':
            base_model = InceptionV3(weights='imagenet', include_top=False)
        else:
            vprint("No model found")
            return
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(512, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(len(category), activation='softmax')(x)
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        for i, layer in enumerate(base_model.layers):
            vprint(i, layer.name)

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    if fitModel:
        image_generator = ImageDataGenerator(rescale=1. / 255,
                                             rotation_range=45,
                                             width_shift_range=.15,
                                             height_shift_range=.15,
                                             horizontal_flip=True,
                                             zoom_range=0.5,
                                             validation_split=0.2
                                             )  # Generator for our training data
        train_data_gen = image_generator.flow_from_directory(batch_size=batch_size,
                                                             directory=PATH,
                                                             shuffle=True,
                                                             target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                             class_mode='categorical',
                                                             subset='training'
                                                             )
        val_data_gen = image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=PATH,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical',
                                                           subset='validation'
                                                           )
        model.fit_generator(train_data_gen, steps_per_epoch=train_data_gen.samples, validation_data=val_data_gen,
                            validation_steps=val_data_gen.samples, epochs=epochs)
        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.

        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        idx = 106
        for layer in model.layers[:idx]:
            layer.trainable = False
        for layer in model.layers[idx:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        from tensorflow.keras.optimizers import SGD, Adam
        # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
        model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy')
        model.fit_generator(train_data_gen, steps_per_epoch=train_data_gen.samples, validation_data=val_data_gen,
                            validation_steps=val_data_gen.samples, epochs=epochs)
        model.save(FILE_DIR + 'asdf.pkl')
    if img_path is not None:
        if not os.path.isabs(img_path):
            img_path = os.path.join(FILE_DIR, img_path)
        img = load_img(img_path, target_size=size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        vprint("Predicting:")
        pred = model.predict(x)
        vprint(pred)
        arr = np.array(pred[0])
        sortarr = arr.argsort()[::-1][:3]
        vprint(sortarr)
        vprint('Predictions:')
        for a in sortarr:
            print(category[a], ": ", f'{arr[a]:.2f}')
        return sortarr


def links_to_csv(link_path, image_path):
    if os.path.isabs(link_path):
        L_PATH = link_path
    else:
        L_PATH = os.path.join(FILE_DIR, link_path)
    if os.path.isabs(image_path):
        I_PATH = image_path
    else:
        I_PATH = os.path.join(FILE_DIR, image_path)
    vprint(L_PATH)
    category = subdir(str(L_PATH))
    images = sorted([x[2] for x in os.walk(image_path)][0])
    datafr = pd.DataFrame(np.zeros((len(images), len(category)), dtype=int), index=images, columns=category)
    vprint(datafr)
    for c in category:
        C_PATH = os.path.join(L_PATH, c)
        links = [x[2] for x in os.walk(C_PATH)][0]
        for l in links:
            datafr.loc[l][c] = 1
    vprint(datafr)
    csv = os.path.join(FILE_DIR, 'image-link.csv')
    datafr.to_csv(csv)


def csv_to_links(link_path, image_path):
    if os.path.isabs(link_path):
        L_PATH = link_path
    else:
        L_PATH = os.path.join(FILE_DIR, link_path)
    if os.path.isabs(image_path):
        I_PATH = image_path
    else:
        I_PATH = os.path.join(FILE_DIR, image_path)
    vprint(L_PATH)
    category = subdir(str(L_PATH))
    csv = os.path.join(FILE_DIR, 'image-link.csv')
    datafr = pd.read_csv(csv, index_col=0).sort_index()
    index = datafr.index.values
    for i in index:
        for c in category:
            C_PATH = os.path.join(L_PATH, c)
            if not os.path.exists(C_PATH):
                os.mkdir(C_PATH)
            CI_PATH = os.path.join(C_PATH, i)
            S_PATH = os.path.join(I_PATH, i)
            if (datafr.loc[i][c] == 1):
                try:
                    os.symlink(S_PATH, CI_PATH)
                except FileExistsError:
                    pass


def add_to_csv(img_path, link_path, top_category):
    if os.path.isabs(img_path):
        img_path = os.path.basename(img_path)
    else:
        img_path = os.path.join(FILE_DIR, img_path)
        img_path = os.path.basename(img_path)
    if os.path.isabs(link_path):
        L_PATH = link_path
    else:
        L_PATH = os.path.join(FILE_DIR, link_path)
    category = subdir(str(L_PATH))
    csv = os.path.join(FILE_DIR, 'image-link.csv')
    datafr = pd.read_csv(csv, index_col=0).sort_index()
    vprint(type(datafr))
    s = [0] * len(category)
    vprint(top_category)
    filled = False
    for c in top_category:
        filled = True
        s[c] = 1
    vprint(s)
    if filled:
        ser = pd.Series(s, index=category, dtype=int)
        vprint(ser)
        datafr.loc[img_path] = ser
        vprint(datafr.loc[img_path])
        datafr.to_csv(csv)


if __name__ == "__main__":
    global cat_result
    if is_link_to_csv:
        links_to_csv(train_path, image_path)
    if is_csv_to_link:
        csv_to_links(train_path, image_path)
    if predict:
        cat_result = asdf(train_path, model_name, image_file, remove_model)
        print(cat_result)
    if is_add_to_csv and cat_result is not None:
        add_to_csv(image_file, train_path, cat_result)
    # imagenet('xception')
    # imagenet('mobilenet')
    # imagenet('resnet50')
