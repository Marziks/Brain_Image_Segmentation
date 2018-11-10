import os
import nibabel as nib
import numpy as np
import math

from keras import optimizers
from keras import backend as K
from keras import Input

from unet3d.model import unet
import matplotlib as plt

train_data_path = 'C:\\Users\\Marzena\\Documents\\MOJE\\STUDIA\\PRACA_INZ\\data\\train_data'
inputs_path = os.path.join(train_data_path, 'inputs')
labels_path = os.path.join(train_data_path, 'labels')


# generator, ktory bedzie podawal kolejne próbki (moje inputy do sieci)
# nie potrzebuje przechowywac wszystkich na raz w tablicy numpy (i nie mam tyle miejsca)

def find_max(imgs_path):
    max_value = 0
    files_number = len(os.listdir(imgs_path))
    if imgs_path == labels_path:
        files_number = files_number - 1  # dodatkowy plik 'desktop.ini'
    for img_nr in range(files_number):
        img_path = os.path.join(imgs_path, os.listdir(imgs_path)[img_nr])
        img_data = nib.load(img_path).get_data()
        if img_data.max() > max_value:
            max_value = img_data.max()
    return max_value


def data_generator(inputs_path, labels_path, start, stop, how_many=1):
    max_value1 = find_max(inputs_path)
    max_value2 = find_max(labels_path)

    input_images = np.zeros(shape=(how_many, 4, 160, 240, 240))  # how_many - wielkość paczki danych
    masks = np.zeros(shape=(how_many, 1, 160, 240, 240))
    for brain_nr in range(start, stop):
        for seq_nr in range(4):
            img_path = os.path.join(inputs_path, os.listdir(inputs_path)[brain_nr * 4 + seq_nr])
            img_data = nib.load(img_path).get_data()
            # przeskalowanie wartosci tak, by byly w zakresie (0,1)
            img_data = img_data.astype(float)
            img_data /= max_value1
            img_data = img_data.transpose()
            input_images[brain_nr][seq_nr][:][:][:155] = img_data
        seg_path = os.path.join(labels_path, os.listdir(labels_path)[brain_nr])
        seg_data = nib.load(seg_path).get_data()
        seg_data = seg_data.astype(float)
        seg_data /= max_value2
        seg_data = seg_data.transpose()
        masks[brain_nr][0][:][:][:155] = seg_data

        if (brain_nr % how_many == how_many - 1) or (brain_nr == stop - 1):  # co 'how_many' iterację generuje tablicę
            print("Shape of label(mask): ")
            print(masks.shape)
            yield (input_images, masks)
        # generator podaje dwie tablice o wielkościach (how_many, 4, 160, 240, 240)
        # i (how_many, 1, 160, 240, 240)
        # (how_many mózgów w 4 sekwencjach) i etykiety


# źródło: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# koniec kodu ze źródła


input_img = Input(shape=(4, 160, 240, 240), name='img')
print("Shape of input img: %s" % input_img.shape)
model = unet.get_unet(input_img, n_filters=16)

model.compile(optimizer=optimizers.Adam(lr=1e-5),
              loss="binary_crossentropy",
              metrics=[dice_coef])
model.summary()

brains_number = len(os.listdir(labels_path)) - 1  # minus 1 bo ostatni plik to desktop.ini
# brains_number = 285
# dane uczące: 80% próbek, dane walidacyjne: reszta
train_set_size = math.floor(0.8 * brains_number)
train_generator = data_generator(inputs_path, labels_path, 0, train_set_size)
validation_generator = data_generator(inputs_path, labels_path, train_set_size, brains_number)

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_set_size / 1,
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=(brains_number - train_set_size) / 1)

model.save('u_net_1.h5')
print(history.history.keys())
