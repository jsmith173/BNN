import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import numpy as np
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


@keras.saving.register_keras_serializable(package="my_package", name="custom_fn")
def custom_fn(x):
    return x**2


# Create the model.
def get_model():
    inputs = keras.Input(shape=input_shape)
    layer0 = CustomLayer(0.5)(inputs)
	
    layer1 = keras.layers.Conv2D(filters=8, kernel_size=(5, 5), activation='relu', padding='valid', input_shape=input_shape)(layer0)
    layer2 = keras.layers.MaxPooling2D(pool_size=(6, 6))(layer1)
    layer3 = keras.layers.Flatten()(layer2)
    outputs = keras.layers.Dense(units=num_classes, activation='softmax')(layer3)
	
    print(inputs.shape)
    print(outputs.shape)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="mean_squared_error")
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

# Now, we can simply load without worrying about our custom objects.
reconstructed_model = keras.models.load_model("custom_model.keras")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

