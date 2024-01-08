import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import json, lib_bnn as nn, lib_tflite as r
from tensorflow.keras import datasets
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
y_train_old = y_train.copy()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_c_train = []; y_c_train = y_train.copy(); x_c_test = []; y_c_test = y_test.copy()
print(x_train.shape); print(y_train.shape); print(x_test.shape); print(y_test.shape); 

with open('config.json', 'r') as f:
 config = json.load(f)

# 
if config['conversion_ctl'] == 1:
    nn.run_conversion()
else:
    x_c_train = np.load("x_c_train.npz")['arr'] 
    x_c_test = np.load("x_c_test.npz")['arr'] 
    print("Data loaded from npz files.")

x_train = 1.0-x_train.astype("float32") / 255.0
x_test = 1.0-x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_c_train = 1.0-x_c_train.astype("float32") / 255.0
x_c_test = 1.0-x_c_test.astype("float32") / 255.0
x_c_train = np.expand_dims(x_c_train, -1)
x_c_test = np.expand_dims(x_c_test, -1)


# 
if config['gen_deterministic_model'] == 1:
 tf.random.set_seed(0)
 deterministic_model = nn.get_deterministic_model(
	input_shape=input_shape,
	loss="categorical_crossentropy",
	optimizer="adam",
	metrics=["accuracy"]
 )
 deterministic_model.summary()
 deterministic_model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1) 

 converter = tf.lite.TFLiteConverter.from_keras_model(deterministic_model)
 tflite_model = converter.convert()
 with open(nn.deterministic_tflite_model, 'wb') as f:
  f.write(tflite_model)
	
if config['eval_deterministic_model'] == nn.USE_TFLITE_MODEL:
 r.evaluate_model(nn.deterministic_tflite_model, x_train, y_train_old)
    
elif config['gen_deterministic_model'] == 1 and config['eval_deterministic_model'] == nn.USE_KERAS_MODEL:
 print('Accuracy on MNIST test set: ',
       str(deterministic_model.evaluate(x_test, y_test, verbose=False)[1]))
 print('Accuracy on corrupted MNIST test set: ',
       str(deterministic_model.evaluate(x_c_test, y_c_test, verbose=False)[1]))

