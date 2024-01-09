import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sklearn.metrics as sk_metrics
from imgaug import augmenters as iaa
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import datasets
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop
from PIL import Image
tfd = tfp.distributions
tfpl = tfp.layers

USE_KERAS_MODEL = 1
USE_TFLITE_MODEL = 2
USE_TFLITE_MODEL_OTHER_CATEGORY = 3

deterministic_tflite_model = 'deterministic_mnist.tflite'
probabilistic_tflite_model = 'probabilistic_mnist.tflite'
bayesian_tflite_model = 'bayesian_mnist.tflite'

#
def corrupt_image(src):
    aug = iaa.imgcorruptlike.ImpulseNoise(severity=3)
    img = Image.fromarray(src) 
    img_resized = img.resize((imgW*2,imgW*2))
    img_arr = np.array(img_resized)
    img_aug = aug(image=img_arr) 
    img = Image.fromarray(img_aug)    
    img_resized = img.resize((imgW,imgW))
    return np.array(img_resized)

def run_conversion():
    print("Start conversion..")
    len_conversion = x_train.shape[0]; kind = "x_train"; d = int(len_conversion/10)
    for i in range(len_conversion):
     x_c_train.append(corrupt_image(x_train[i]))
     if (i % d == 0):
      print("Running conversion {}, percent: {}%".format(kind, i/len_conversion*100))
    len_conversion = x_test.shape[0]; kind = "x_test"; d = int(len_conversion/10)
    for i in range(len_conversion):
     x_c_test.append(corrupt_image(x_test[i]))
     if (i % d == 0):
      print("Running conversion {}, percent: {}%".format(kind, i/len_conversion*100))
    np.savez("x_c_train.npz", arr=x_c_train) # save all in one file
    np.savez("x_c_test.npz", arr=x_c_test) # save all in one file
    print("End conversion")

#
def nll(y_true, y_pred):
    """
    This function should return the negative log-likelihood of each sample
    in y_true given the predicted distribution y_pred. If y_true is of shape
    [B, E] and y_pred has batch shape [B] and event_shape [E], the output
    should be a Tensor of shape [B].
    """
    return -y_pred.log_prob(y_true)  

def get_deterministic_model(input_shape, loss, optimizer, metrics):
    model = keras.Sequential([
        layers.Conv2D(filters=8, kernel_size=(5, 5), activation='relu', padding='valid', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(6, 6)),
        layers.Flatten(),
        layers.Dense(units=10, activation='softmax')
    ])

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
 
def get_probabilistic_model(input_shape, loss, optimizer, metrics):
    """
    This function should return the probabilistic model according to the
    above specification.
    The function takes input_shape, loss, optimizer and metrics as arguments, which should be
    used to define and compile the model.
    Your function should return the compiled model.
    """
    model = keras.Sequential([
        layers.Conv2D(filters=8, kernel_size=(5, 5), activation='relu', padding='valid', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(6, 6)),
        layers.Flatten(),
        layers.Dense(tfpl.OneHotCategorical.params_size(10)),
        tfpl.OneHotCategorical(10, convert_to_tensor_fn=tfd.Distribution.mode)
    ])

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

#Bayesian part 
def get_convolutional_reparameterization_layer(input_shape, divergence_fn):
    """
    This function should create an instance of a Convolution2DReparameterization
    layer according to the above specification.
    The function takes the input_shape and divergence_fn as arguments, which should
    be used to define the layer.
    Your function should then return the layer instance.
    """

    layer = tfpl.Convolution2DReparameterization(
                input_shape=input_shape, filters=8, kernel_size=(5, 5),
                activation='relu', padding='VALID',
                kernel_prior_fn=tfpl.default_multivariate_normal_fn,
                kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                kernel_divergence_fn=divergence_fn,
                bias_prior_fn=tfpl.default_multivariate_normal_fn,
                bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                bias_divergence_fn=divergence_fn
            )
    return layer
	
def spike_and_slab(event_shape, dtype):
    distribution = tfd.Mixture(
        cat=tfd.Categorical(probs=[0.5, 0.5]),
        components=[
            tfd.Independent(tfd.Normal(
                loc=tf.zeros(event_shape, dtype=dtype),
                scale=1.0*tf.ones(event_shape, dtype=dtype)),
                            reinterpreted_batch_ndims=1),
            tfd.Independent(tfd.Normal(
                loc=tf.zeros(event_shape, dtype=dtype),
                scale=10.0*tf.ones(event_shape, dtype=dtype)),
                            reinterpreted_batch_ndims=1)],
    name='spike_and_slab')
    return distribution

def get_prior(kernel_size, bias_size, dtype=None):
    """
    This function should create the prior distribution, consisting of the
    "spike and slab" distribution that is described above.
    The distribution should be created using the kernel_size, bias_size and dtype
    function arguments above.
    The function should then return a callable, that returns the prior distribution.
    """
    n = kernel_size+bias_size
    prior_model = keras.Sequential([tfpl.DistributionLambda(lambda t : spike_and_slab(n, dtype))])
    return prior_model

def get_posterior(kernel_size, bias_size, dtype=None):
    """
    This function should create the posterior distribution as specified above.
    The distribution should be created using the kernel_size, bias_size and dtype
    function arguments above.
    The function should then return a callable, that returns the posterior distribution.
    """
    n = kernel_size + bias_size
    return keras.Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.IndependentNormal(n)
    ])

def get_dense_variational_layer(prior_fn, posterior_fn, kl_weight):
    """
    This function should create an instance of a DenseVariational layer according
    to the above specification.
    The function takes the prior_fn, posterior_fn and kl_weight as arguments, which should
    be used to define the layer.
    Your function should then return the layer instance.
    """
    return tfpl.DenseVariational(
        units=10, make_posterior_fn=posterior_fn, make_prior_fn=prior_fn, kl_weight=kl_weight
    )

