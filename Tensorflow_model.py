import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model(
    "C:/Users/Geoffrey/Documents/AI & ML/Convolutional Neural Networks/Road Segementation/content/model_new"
)

import PIL.Image as Image

x = np.array(Image.open("6.jpg")) / 255

import time

start = time.time()
pred = model.predict(np.expand_dims(x[:, :256, :], axis=0))
end = time.time()
print(end - start)
print(pred.shape)
plt.show(pr)
