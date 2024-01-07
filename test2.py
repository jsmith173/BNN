import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import numpy as np
import json
tfd = tfp.distributions
tfpl = tfp.layers

num_classes = 10
input_shape = (28, 28, 1)
output_shape = (num_classes, 1)
input_batch_shape = (4, 28, 28)
output_batch_shape = (4, num_classes)

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()


# Upon registration, you can optionally specify a package or a name.
# If left blank, the package defaults to `Custom` and the name defaults to
# the class name.
@keras.saving.register_keras_serializable(package="MyLayers")
class CustomLayer(keras.layers.Layer):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def call(self, x):
        return x * self.factor

    def get_config(self):
        return {"factor": self.factor}


@keras.saving.register_keras_serializable(package="my_package", name="nll")
def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)  


# Create the model.
def get_model():
    inputs = keras.Input(shape=input_shape)
    layer0 = CustomLayer(0.5)(inputs)
	
    layer1 = keras.layers.Conv2D(filters=8, kernel_size=(5, 5), activation='relu', padding='valid', input_shape=input_shape)(layer0)
    layer2 = keras.layers.MaxPooling2D(pool_size=(6, 6))(layer1)
    layer3 = keras.layers.Flatten()(layer2)
    layer4 = keras.layers.Dense(tfpl.OneHotCategorical.params_size(10))(layer3)
    outputs = tfpl.OneHotCategorical(10, convert_to_tensor_fn=tfd.Distribution.mode)(layer4)
	
    print(inputs.shape)
    print(outputs.shape)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss=nll, metrics=["accuracy"])
    return model


# Train the model.
def train_model(model):
    input = np.random.random(input_batch_shape)
    target = np.random.random(output_batch_shape)
    model.summary()
    print(input.shape)
    print(target.shape)
    model.fit(input, target)
    return model


test_input = np.random.random(input_batch_shape)
test_target = np.random.random(output_batch_shape)

model = get_model()
model = train_model(model)
model.save("custom_model.keras")

json_str = model.to_json()
with open("output.json", "w") as f:
 json.dump(json.loads(json_str), f, indent=4)
	