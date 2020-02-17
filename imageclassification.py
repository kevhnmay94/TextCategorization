# insert shebangs here #
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import time
from pathlib import Path
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.SAFEBLOCK = 2048 * 2048

fitModel = False
plotOn = False
verbose = True
image_file = None
train_path = 'konten/kategori'
image_path = 'konten/image'
base_name = 'xception'
is_csv_to_link = False
is_link_to_csv = False
remove_model = False
is_add_to_csv = False
predict = True
cat_result = None
output_model = 'asdf.pkl'
csv_path = 'image-link.csv'
binary_name = None

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
        elif arg == 'base-model':
            base_name = sys.argv[i + 2]
        elif arg == 'output-model':
            output_model = sys.argv[i + 2]
        elif arg == 'csv-path':
            csv_path = sys.argv[i + 2]
        elif arg == 'csv-to-link':
            is_csv_to_link = True
        elif arg == 'link-to-csv':
            is_link_to_csv = True
        elif arg == 'image-path':
            image_path = sys.argv[i + 2]
        elif arg == 'binary-name':
            binary_name = sys.argv[i + 2]
        elif arg == 'no-binary':
            binary_name = None
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

if not verbose:
    # shut TF up!!!
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, models
from tensorflow.keras.applications import Xception, xception, MobileNetV2, mobilenet_v2, ResNet50, resnet50, \
    InceptionV3, inception_v3
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img


FILE_DIR = str(Path(sys.argv[0]).parent) + str(os.sep)
FILE_DIR = os.path.abspath(FILE_DIR)
vprint(FILE_DIR)

def subdir(parent: str):
    category = [x[1] for x in os.walk(parent)][0]
    b = sorted(category)
    return b

def asdf(path, base_name, img_path, remove_model, model_name, binary_name=None):
    global fitModel, plotOn
    if os.path.isabs(path):
        PATH = path
    else:
        PATH = os.path.join(FILE_DIR, path)
    vprint(PATH)
    if binary_name is None:
        category = subdir(str(PATH))
        total_item = 0
        for c in category:
            dir = os.path.join(PATH, c)
            num_dir = len(os.listdir(dir))
            total_item += num_dir
    else:
        category = [binary_name]
        total_item = 0
        for c in [0,1]:
            dir = os.path.join(PATH, str(c))
            num_dir = len(os.listdir(dir))
            total_item += num_dir
    batch_size = 2
    epochs = 5
    if base_name == 'xception' or base_name == 'inception':
        IMG_HEIGHT = 299
        IMG_WIDTH = 299
        size = (299, 299)
        if base_name == 'xception':
            preprocess_input = xception.preprocess_input
            decode_predictions = xception.decode_predictions
        else:
            preprocess_input = inception_v3.preprocess_input
            decode_predictions = inception_v3.decode_predictions
    else:
        IMG_HEIGHT = 224
        IMG_WIDTH = 224
        size = (224, 224)
        if base_name == 'resnet50':
            preprocess_input = resnet50.preprocess_input
            decode_predictions = resnet50.decode_predictions
        elif base_name == 'mobilenet':
            preprocess_input = mobilenet_v2.preprocess_input
            decode_predictions = mobilenet_v2.decode_predictions
    try:
        if base_name == 'xception':
            model = models.load_model(os.path.join(FILE_DIR,model_name))
    except IOError as err:
        vprint(err)
        model = None
    if binary_name is None:
        class_mode = 'categorical'
    else:
        class_mode = 'binary'
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
                                                         class_mode=class_mode,
                                                         subset='training'
                                                         )
    val_data_gen = image_generator.flow_from_directory(batch_size=batch_size,
                                                       directory=PATH,
                                                       shuffle=True,
                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       class_mode=class_mode,
                                                       subset='validation'
                                                       )
    import math
    steps = math.ceil(train_data_gen.samples / batch_size)
    val_steps = math.ceil(val_data_gen.samples / batch_size)
    if model is None or remove_model:
        fitModel = True
        base_model = None
        if base_name == 'xception':
            base_model = Xception(weights='imagenet', include_top=False)
        elif base_name == 'mobilenet':
            base_model = MobileNetV2(weights='imagenet', include_top=False)
        elif base_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False)
        elif base_name == 'inception':
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
        if binary_name is None:
            predictions = Dense(len(category), activation='softmax')(x)
        else:
            predictions = Dense(1, activation='sigmoid')(x)
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
        if binary_name is None:
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit_generator(train_data_gen, steps_per_epoch=steps, validation_data=val_data_gen,
                            validation_steps=val_steps, epochs=epochs)
    if fitModel:
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
        from tensorflow.keras.optimizers import Adam
        # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
        if binary_name is None:
            model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit_generator(train_data_gen, steps_per_epoch=steps, validation_data=val_data_gen,
                            validation_steps=val_steps, epochs=epochs)
        model.save(os.path.join(FILE_DIR,model_name))
        model.evaluate(val_data_gen)
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
        if binary_name is None:
            arr = np.array(pred[0])
            sortarr = arr.argsort()[::-1][:3]
            vprint(sortarr)
            vprint('Predictions:')
            result = []
            for a in sortarr:
                result.append({'category': category[a], 'value': arr[a].astype(float)})
                vprint(category[a], ": ", f'{arr[a]:.2f}')
            import json
            result = json.dumps(result)
            return result
        else:
            return None
    else:
        vprint(val_data_gen.filenames)
        res = model.evaluate(val_data_gen, verbose=1)
        vprint(res)
        pre = model.predict(val_data_gen, verbose=1)
        if binary_name is not None:
            pre = pre.round().astype(int)
            count = 0
            for i,p in enumerate(pre):
                if int(val_data_gen.filenames[i][0]) == p[0]:
                    count = count + 1
                vprint('{} : {}'.format(val_data_gen.filenames[i],p[0]))
            acc = count / len(val_data_gen.filenames)
            vprint('Accuracy : {}'.format(acc))


def links_to_csv(link_path, image_path, csv_path, binary_name=None):
    vprint("Link to csv mode")
    if os.path.isabs(link_path):
        L_PATH = link_path
    else:
        L_PATH = os.path.join(FILE_DIR, link_path)
    if os.path.isabs(image_path):
        I_PATH = image_path
    else:
        I_PATH = os.path.join(FILE_DIR, image_path)
    vprint(L_PATH)
    if binary_name is None:
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
    else:
        category = [binary_name]
        images = sorted([x[2] for x in os.walk(image_path)][0])
        datafr = pd.DataFrame(np.zeros((len(images), len(category)), dtype=int), index=images, columns=category)
        vprint(datafr)
        for c in [0,1]:
            C_PATH = os.path.join(L_PATH, str(c))
            links = [x[2] for x in os.walk(C_PATH)][0]
            for l in links:
                datafr.loc[l][binary_name] = c
        vprint(datafr)
    csv = os.path.join(FILE_DIR, csv_path)
    datafr.to_csv(csv)


def csv_to_links(link_path, image_path, csv_path, binary_name=None):
    vprint("Csv to link mode")
    if os.path.isabs(link_path):
        L_PATH = link_path
    else:
        L_PATH = os.path.join(FILE_DIR, link_path)
    if os.path.isabs(image_path):
        I_PATH = image_path
    else:
        I_PATH = os.path.join(FILE_DIR, image_path)
    vprint(L_PATH)
    csv = os.path.join(FILE_DIR, csv_path)
    datafr = pd.read_csv(csv, index_col=0).sort_index()
    if binary_name is None:
        category = datafr.columns.values
        vprint("Category from csv: " ,category)
        index = datafr.index.values
        for i in index:
            for c in category:
                C_PATH = os.path.join(L_PATH, c)
                if not os.path.exists(C_PATH):
                    vprint("Directory not exist, creating...")
                    os.mkdir(C_PATH)
                CI_PATH = os.path.join(C_PATH, i)
                S_PATH = os.path.join(I_PATH, i)
                if (datafr.loc[i][c] == 1):
                    try:
                        if os.path.exists(S_PATH):
                            os.symlink(S_PATH, CI_PATH)
                    except FileExistsError:
                        pass
    else:
        category = [binary_name]
        index = datafr.index.values
        for i in index:
            for c in [0,1]:
                C_PATH = os.path.join(L_PATH, str(c))
                if not os.path.exists(C_PATH):
                    vprint("Directory not exist, creating...")
                    os.mkdir(C_PATH)
                CI_PATH = os.path.join(C_PATH, i)
                S_PATH = os.path.join(I_PATH, i)
                if (datafr.loc[i][binary_name] == c):
                    try:
                        if os.path.exists(S_PATH):
                            os.symlink(S_PATH, CI_PATH)
                    except FileExistsError:
                        pass


def add_to_csv(img_path, link_path, top_category, csv_path, binary_name=None):
    if os.path.isabs(img_path):
        img_path = os.path.basename(img_path)
    else:
        img_path = os.path.join(FILE_DIR, img_path)
        img_path = os.path.basename(img_path)
    if os.path.isabs(link_path):
        L_PATH = link_path
    else:
        L_PATH = os.path.join(FILE_DIR, link_path)
    if binary_name is None:
        category = subdir(str(L_PATH))
        csv = os.path.join(FILE_DIR, csv_path)
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
    else:
        category = [binary_name]
        csv = os.path.join(FILE_DIR, csv_path)
        datafr = pd.read_csv(csv, index_col=0).sort_index()
        vprint(type(datafr))



if __name__ == "__main__":
    if is_link_to_csv:
        links_to_csv(train_path, image_path, csv_path, binary_name=binary_name)
    if is_csv_to_link:
        csv_to_links(train_path, image_path, csv_path, binary_name=binary_name)
    if predict:
        cat_result = asdf(train_path, base_name, image_file, remove_model, output_model, binary_name=binary_name)
        if cat_result is not None:
            print(cat_result)
    if is_add_to_csv and cat_result is not None:
        add_to_csv(image_file, train_path, cat_result, csv_path, binary_name=binary_name)
