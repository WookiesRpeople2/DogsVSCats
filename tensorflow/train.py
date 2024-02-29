import tensorflow as tf
import os
import zipfile
from tdataset import perpare_data, dataset, img_width, img_height

# HyperParams
base = "./data/dogs-vs-cats.zip"
test_dir = "./data/test1.zip"
train_dir = "./data/train.zip"
train_file = os.path.splitext(train_dir)[0]
num_epochs = 1
model_save = "./tensorflow/trained_model.tnf"
# -------


def unzipFiles(dir):
    if not os.path.exists(dir):
        with zipfile.ZipFile(dir, 'r') as z:
            z.extractall("./data")


def nuraleNet():

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        16, kernel_size=3, activation='relu', padding='same', input_shape=(img_width, img_height, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(
        32, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(
        64, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    return model


if __name__ == "__main__":
    unzipFiles(base)
    unzipFiles(test_dir)
    unzipFiles(train_dir)

    dataframe = perpare_data(train_file)
    traning_data = dataset(dataframe, train_file)

    model = nuraleNet()

    model.fit(traning_data, epochs=num_epochs)

    model.save(model_save)
