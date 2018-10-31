import os
import nibabel as nib
import numpy as np

from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K
from unet3d.model import unet

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


def data_generator(inputs_path, labels_path, start, stop):
    max_value1 = find_max(inputs_path)
    max_value2 = find_max(labels_path)

    input_images = np.zeros(shape=(8, 4, 240, 240, 155))  # 8 - wielkość paczki danych
    masks = np.zeros(shape=(8, 240, 240, 155))
    for brain_nr in range(start, stop):
        for seq_nr in range(4):
            img_path = os.path.join(inputs_path, os.listdir(inputs_path)[brain_nr * 4 + seq_nr])
            img_data = nib.load(img_path).get_data()
            # przeskalowanie wartosci tak, by byly w zakresie (0,1)
            img_data = img_data.astype(float)
            img_data /= max_value1
            input_images[brain_nr % 8][seq_nr] = img_data
        seg_path = os.path.join(labels_path, os.listdir(labels_path)[brain_nr])
        seg_data = nib.load(seg_path).get_data()
        seg_data = seg_data.astype(float)
        seg_data /= max_value2
        masks[brain_nr] = seg_data

        if (brain_nr % 8 == 7) or (brain_nr == stop - 1):  # co 8 iterację generuje tablicę
            yield (input_images, masks)
        # generator podaje tablicę o wielkości (8, 4, 240, 240, 155)
        # (8 mózgów w 4 sekwencjach)


# źródło: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# koniec kodu ze źródła

# model = models.Sequential()
# architektura sieci (warstwy)

# model.add(layers.Conv3D(64, kernel_size=3, activation='relu', input_shape=(4, 240, 240, 155),
#                         data_format='channels_first'))
# model.add(layers.MaxPooling3D((2, 2, 2)))
# model.add(layers.Conv3D(128, kernel_size=3, activation='relu'))
# model.add(layers.MaxPooling3D((2, 2, 2)))
# model.add(layers.Conv3D(128, kernel_size=3, activation='relu'))
# model.add(layers.MaxPooling3D((2, 2, 2)))
# model.add(layers.Conv3D(256, kernel_size=3, activation='relu'))
# model.add(layers.MaxPooling3D((2, 2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(8928000, activation='softmax'))  # na wyjsciu: dla kazdego piksela
# # prawdopodobienstwo przynaleznosci do danej klasy (guz/ zdrowa tkanka)


model = unet.get_unet(...)

model.compile(optimizer=optimizers.Adam(lr=1e-5),
              loss=dice_coef_loss, # może binary_crossentropy ??
              metrics=[dice_coef])


brains_number = len(os.listdir(labels_path)) - 1  # minus 1 bo ostatni plik to desktop.ini
# brains_number = 285
# dane uczące: 200 próbek, dane testowe: 85 próbek
train_generator = data_generator(inputs_path, labels_path, 0, 200)
test_generator = data_generator(inputs_path, labels_path, 200, brains_number)

history = model.fit_generator(train_generator,
                              steps_per_epoch=25,
                              epochs=100)

model.save('u_net_1.h5')
print(history.history.keys())

results = model.evaluate_generator(test_generator)
print(results)