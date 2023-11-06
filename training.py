# ruff: noqa: E402
import sys

# Confirm that we're using Python 3
assert sys.version_info.major == 3, 'Oops, not running Python 3. Use Runtime > Change runtime type'
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
# import numpy as np
# import matplotlib.pyplot as plt
import os
# import subprocess

MODEL_DIR = "C:/Users/Cbrock431/Documents/GitHub/MachineLearningTest"

print('TensorFlow version: {}'.format(tf.__version__))

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

training = False
new_model = True
epochs = 100

if training:
  if new_model:
    model = keras.Sequential([
      keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
                      strides=2, activation='relu', name='Conv1'),
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(0.4),
      keras.layers.Dense(10)
    ])
    model.summary()

  else:
    version = 1
    import_path = os.path.join(MODEL_DIR, str(version))
    model = tf.keras.models.load_model(
      import_path
    )


  model.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[keras.metrics.SparseCategoricalAccuracy()])
  
  # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
  callback = tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy', patience=1)

  model.fit(train_images, train_labels, epochs=epochs, callbacks=[callback])

  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('\nTest accuracy: {}'.format(test_acc))

  # Fetch the Keras session and save the model
  # The signature definition is defined by the input and output tensors,
  # and stored with the default serving key
  version = 1
  export_path = os.path.join(MODEL_DIR, str(version))
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
  version = 1
  import_path = os.path.join(MODEL_DIR, str(version))
  model = tf.keras.models.load_model(
    import_path
  )

  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('\nTest accuracy: {}'.format(test_acc))

