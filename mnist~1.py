import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import json, lib_bnn as nn, lib_tflite as r
from tensorflow.keras import datasets
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from PIL import Image
tfd = tfp.distributions
tfpl = tfp.layers


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
y_train_old = y_train.copy()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_c_train = []; y_c_train = y_train.copy(); x_c_test = []; y_c_test = y_test.copy()

with open('config.json', 'r') as f:
 config = json.load(f)

# 
if config['conversion_ctl'] == 1:
    nn.run_conversion()
else:
    x_c_train = np.load("x_c_train.npz")['arr'] 
    x_c_test = np.load("x_c_test.npz")['arr'] 
    x_other = np.load("category_other.npz")['arr'] 
    print("Data loaded from npz files.")

print("Shapes (1)")
print(x_train.shape); print(y_train.shape); print(x_test.shape); print(y_test.shape); 
print(y_train.shape); print(x_other.shape)


x_train = 1.0-x_train.astype("float32") / 255.0
x_test = 1.0-x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_c_train = 1.0-x_c_train.astype("float32") / 255.0
x_c_test = 1.0-x_c_test.astype("float32") / 255.0
x_other = x_other.astype("float32") / 255.0
x_c_train = np.expand_dims(x_c_train, -1)
x_c_test = np.expand_dims(x_c_test, -1)
x_other = np.expand_dims(x_other, -1)
num_test = x_other.shape[0]

print("Shapes (2)")
print(x_train.shape); print(y_train.shape); print(x_test.shape); print(y_test.shape); 
print(y_train.shape); print(x_other.shape)

# 
if config['gen_deterministic_model'] == 1:
 tf.random.set_seed(0)
 deterministic_model = nn.get_deterministic_model(
	input_shape=input_shape,
	loss="categorical_crossentropy",
	optimizer="adam",
	metrics=["accuracy"]
 )
 #deterministic_model.summary()
 deterministic_model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1) 
 model = deterministic_model	

 converter = tf.lite.TFLiteConverter.from_keras_model(deterministic_model)
 tflite_model = converter.convert()
 with open(nn.deterministic_tflite_model, 'wb') as f:
  f.write(tflite_model)

if config['eval_deterministic_model'] == nn.USE_TFLITE_MODEL:
 r.evaluate_model(nn.deterministic_tflite_model, x_train, y_train_old)
elif config['eval_deterministic_model'] == nn.USE_TFLITE_MODEL_OTHER_CATEGORY:
 r.evaluate_model_short(nn.deterministic_tflite_model, x_other)   
elif config['gen_deterministic_model'] == 1 and config['eval_deterministic_model'] == nn.USE_KERAS_MODEL:
 print('Accuracy on MNIST test set: ',
       str(deterministic_model.evaluate(x_test, y_test, verbose=False)[1]))
 print('Accuracy on corrupted MNIST test set: ',
       str(deterministic_model.evaluate(x_c_test, y_c_test, verbose=False)[1]))
	   
 x = x_other[:num_test]	   
 arr = model.predict(x, x.shape[0])
 print(x.shape[0], len(arr))
 print(arr); print('')

# 
if config['gen_prob_model'] == 1:
 tf.random.set_seed(0)
 probabilistic_model = nn.get_probabilistic_model(
	input_shape=input_shape,
	loss=nn.nll,
	optimizer="adam",
	metrics=["accuracy"]
 )
 #probabilistic_model.summary()    
 probabilistic_model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)
 model = probabilistic_model	

 converter = tf.lite.TFLiteConverter.from_keras_model(probabilistic_model)
 tflite_model = converter.convert()
 with open(nn.probabilistic_tflite_model, 'wb') as f:
  f.write(tflite_model)
  
if config['eval_prob_model'] == nn.USE_TFLITE_MODEL:
 r.evaluate_model(nn.probabilistic_tflite_model, x_train, y_train_old)
elif config['eval_prob_model'] == nn.USE_TFLITE_MODEL_OTHER_CATEGORY:
 r.evaluate_model_short(nn.probabilistic_tflite_model, x_other)       
elif config['gen_prob_model'] == 1 and config['eval_prob_model'] == nn.USE_KERAS_MODEL:
 print('Accuracy on MNIST test set (prob): ',
       str(probabilistic_model.evaluate(x_test, y_test, verbose=False)[1]))
 print('Accuracy on corrupted MNIST test set  (prob): ',
       str(probabilistic_model.evaluate(x_c_test, y_c_test, verbose=False)[1]))

 x = x_other[:num_test]	   
 arr = model.predict(x, x.shape[0])
 print(x.shape[0], len(arr))
 print(arr); print('')

# Build and compile the Bayesian CNN model
if config['gen_bay_model'] == 1:
 tf.random.set_seed(0)
 divergence_fn = lambda q, p, _ : tfd.kl_divergence(q, p) / x_train.shape[0]
 convolutional_reparameterization_layer = nn.get_convolutional_reparameterization_layer(
     input_shape=input_shape, divergence_fn=divergence_fn
 )
 dense_variational_layer = nn.get_dense_variational_layer(
     nn.get_prior, nn.get_posterior, kl_weight=1/x_train.shape[0]
 )

 bayesian_model = keras.Sequential([
     convolutional_reparameterization_layer,
     layers.MaxPooling2D(pool_size=(6, 6)),
     layers.Flatten(),
     dense_variational_layer,
     tfpl.OneHotCategorical(10, convert_to_tensor_fn=tfd.Distribution.mode)
 ])
 bayesian_model.compile(loss=nn.nll,
               optimizer=RMSprop(),
               metrics=['accuracy'],
               experimental_run_tf_function=False)
			  
			  
 #
 #bayesian_model.summary()
 bayesian_model.fit(x_train, y_train, batch_size=128, epochs=40, validation_split=0.1)
 model = bayesian_model	
 
 converter = tf.lite.TFLiteConverter.from_keras_model(bayesian_model)
 tflite_model = converter.convert()
 with open(nn.bayesian_tflite_model, 'wb') as f:
  f.write(tflite_model)
 
if config['eval_bay_model'] == nn.USE_TFLITE_MODEL:
 r.evaluate_model(nn.bayesian_tflite_model, x_train, y_train_old)
elif config['eval_bay_model'] == nn.USE_TFLITE_MODEL_OTHER_CATEGORY:
 r.evaluate_model_short(nn.bayesian_tflite_model, x_other)       
elif config['gen_bay_model'] == 1 and config['eval_bay_model'] == nn.USE_KERAS_MODEL:
 print('Accuracy on MNIST test set (bayesian): ',
       str(bayesian_model.evaluate(x_test, y_test, verbose=False)[1]))
 print('Accuracy on corrupted MNIST test set (bayesian): ',
       str(bayesian_model.evaluate(x_c_test, y_c_test, verbose=False)[1]))

 x = x_other[:num_test]	   
 arr = model.predict(x, x.shape[0])
 print(x.shape[0], len(arr))
 print(arr); print('')
