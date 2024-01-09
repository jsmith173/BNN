import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import pathlib


# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices, test_images):
  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  predict_val = np.zeros((len(test_image_indices),), dtype=float)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
	
    idx = output.argmax()
    predictions[i] = idx
    predict_val[i] = output[idx]

  return predictions, predict_val

def evaluate_model(tflite_file, test_images, test_labels):
  test_image_indices = range(test_images.shape[0])
  predictions, predict_val = run_tflite_model(tflite_file, test_image_indices, test_images)
  print(predict_val)

  print('Evaluate tflite model (all samples)')
  accuracy = (np.sum(test_labels == predictions) * 100) / len(test_images)
  print('Model accuracy is %.4f%% (Number of test samples=%d)' % (accuracy, len(test_images)))
  print('')

def evaluate_model_short(tflite_file, test_images):
  test_image_indices = range(test_images.shape[0])
  predictions, predict_val = run_tflite_model(tflite_file, test_image_indices, test_images)
  print('Evaluate tflite model (other samples)')
  print("predict_val: ", predict_val)
  print("predictions: ", predictions)
  print('')
