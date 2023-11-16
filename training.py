# ruff: noqa: E402
import sys

# Confirm that we're using Python 3
assert sys.version_info.major == 3, 'Oops, not running Python 3. Use Runtime > Change runtime type'

import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
import glob2 as glob
import os

register_heif_opener()

MODEL_DIR = "C:/Users/Cbrock431/Documents/GitHub/RecyclingMachineLearning"
VERSION = 3

TRAINING = True
NEW_MODEL = True
EPOCHS = 100

print('TensorFlow version: {}'.format(tf.__version__))

def openFolder(path, label):
  valid_images = [".heic", ".jpg", ".jpeg", ".png", ".gif"]

  print(path)

  images = []
  labels = []

  for ending in valid_images:
    for filename in glob.glob(path + "/*" + ending):
      img = Image.open(filename)
      # img = img.convert('L')
      img = img.resize((252, 336))
      images.append(np.asarray(img))
      labels.append(label)

  return np.asarray(images), np.asarray(labels)

def shuffle(images, labels):
  newImages = []
  newLabels = []
  
  for i in range(len(images)):
    p = np.random.randint(0, len(images))
    newImages.append(images[p])
    newLabels.append(labels[p])
    images = np.delete(images, p, axis=0)
    labels = np.delete(labels, p, axis=0)
    

  return np.asarray(newImages), np.asarray(newLabels)

# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


image1, label1 = openFolder((MODEL_DIR + "/Testing/EmptyWaterBottle"), 0)
image2, label2 = openFolder((MODEL_DIR + "/Testing/EmptyPepsiBottle"), 0)
image3, label3 = openFolder((MODEL_DIR + "/Testing/FilledWaterBottle"), 1)
image4, label4 = openFolder((MODEL_DIR + "/Testing/FilledPepsiBottle"), 1)

test_images = np.append(image1, image2, axis=0)
test_images = np.append(test_images, image3, axis=0)
test_images = np.append(test_images, image4, axis=0)
test_labels = np.append(label1, label2, axis=0)
test_labels = np.append(test_labels, label3, axis=0)
test_labels = np.append(test_labels, label4, axis=0)


# test_images, test_labels = shuffle(test_images, test_labels)

# scale the values to 0.0 to 1.0
test_images = test_images / 255.0

# # reshape for feeding into the model
test_images = test_images.reshape(test_images.shape[0], 252, 336, 3)

class_names = ['Empty water bottle', 'Empty Pepsi bottle', 'Filled water bottle', 'Filled Pepsi bottle']

print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

if TRAINING:

  image1, label1 = openFolder((MODEL_DIR + "/Training/EmptyWaterBottle"), 0)
  image2, label2 = openFolder((MODEL_DIR + "/Training/EmptyPepsiBottle"), 0)
  image3, label3 = openFolder((MODEL_DIR + "/Training/FilledWaterBottle"), 1)
  image4, label4 = openFolder((MODEL_DIR + "/Training/FilledPepsiBottle"), 1)

  train_images = np.append(image1, image2, axis=0)
  train_images = np.append(train_images, image3, axis=0)
  train_images = np.append(train_images, image4, axis=0)
  train_labels = np.append(label1, label2, axis=0)
  train_labels = np.append(train_labels, label3, axis=0)
  train_labels = np.append(train_labels, label4, axis=0)

  # train_images, train_labels = shuffle(train_images, train_labels)

  train_images = train_images / 255.0

  train_images = train_images.reshape(train_images.shape[0], 252, 336, 3)

  print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))

  if NEW_MODEL:
    # TODO: Change stuff until a higher actual accuracy
    model = keras.Sequential([
      keras.layers.Conv2D(input_shape=(252, 336, 3), filters=12, kernel_size=4, 
                      strides=2, activation='relu', name='Conv1'),
      keras.layers.Flatten(),
      keras.layers.Dense(256, activation='relu'),
      keras.layers.Dense(256, activation='relu'),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(2)
    ])
    model.summary()

  else:
    import_path = os.path.join(MODEL_DIR, str(VERSION))
    model = tf.keras.models.load_model(
      import_path
    )


  model.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[keras.metrics.SparseCategoricalAccuracy()])
  
  # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
  callback = tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy', patience=2)

  model.fit(train_images, train_labels, epochs=EPOCHS, callbacks=[callback])

  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('\nTest accuracy: {}'.format(test_acc))

  # Fetch the Keras session and save the model
  # The signature definition is defined by the input and output tensors,
  # and stored with the default serving key
  export_path = os.path.join(MODEL_DIR, str(VERSION))
  print('export_path = {}\n'.format(export_path))

  tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
  )

else:
  import_path = os.path.join(MODEL_DIR, str(VERSION))
  model = tf.keras.models.load_model(
    import_path
  )

  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('\nTest accuracy: {}'.format(test_acc))

