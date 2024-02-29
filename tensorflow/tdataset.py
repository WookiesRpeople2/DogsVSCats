
import tensorflow as tf
import os
import pandas as pd

# Hyperparams
img_height, img_width = 128, 128
# -----------


def perpare_data(dir):
    filename = os.listdir(dir)

    categories = []
    for f in filename:
        category = f.split('.')[0]
        if category == 'dog':
            categories.append(1)
        else:
            categories.append(0)
    return pd.DataFrame({'filename': filename, 'category': categories})


def dataset(dataframe, path):
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    dataframe['category'] = dataframe['category'].astype(str)

    train_generator = data_gen.flow_from_dataframe(
        dataframe,
        path+'/',
        x_col='filename',
        y_col='category',
        target_size=(img_width, img_height),
        class_mode='categorical',
        batch_size=15
    )

    return train_generator
