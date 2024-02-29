import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# HyperParams
test_file = "./data/test1"
model_saved_path = "./tensorflow/trained_model.tnf"
# ----------


def load_images(file_paths):
    images = [Image.open(path).resize((128, 128)) for path in file_paths]
    return images


def drawplt(model, testing_data, num_plots=20):
    classes = ["dog", "cat"]
    testing_data = np.array(testing_data)
    preds = np.argmax(model.predict(testing_data), axis=1)

    fig, axes = plt.subplots(num_plots // 10, 10, figsize=(20, 5))
    fig.suptitle("Predictions")
    for i, ax in enumerate(axes.flat):
        ax.imshow(testing_data[i])
        ax.set_title(f'Prediction: {classes[preds[i]]}')
        ax.axis('off')

    plt.show()


if __name__ == '__main__':
    model = tf.keras.models.load_model(model_saved_path)

    testing_data_paths = [f"{test_file}/{i}.jpg" for i in range(1, 21)]
    testing_data = load_images(testing_data_paths)

    testing_data = np.array([np.array(img) for img in testing_data])

    drawplt(model, testing_data)
