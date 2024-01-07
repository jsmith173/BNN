from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')

# Useless custom loss here
def custom_loss(y_true, y_pred):
    return keras.backend.mean(keras.backend.square(y_true - y_pred), axis=-1)

model.save("model", save_format='tf')
model.compile(loss=custom_loss, optimizer=keras.optimizers.RMSprop())
# Here comes the bug (no bug)
new_model = keras.models.load_model('model', custom_objects={'loss': custom_loss})

