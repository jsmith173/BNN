import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import cv2

num_pictures = 6

def convert_img_to_array(idx):
	INPUT_FILE = "category_other/pic{file_idx}.png".format(file_idx = idx)
	img_file = cv2.imread(INPUT_FILE, cv2.IMREAD_GRAYSCALE)
	img_array = np.array(img_file)
	return img_array

##
x_train = []
for i in range (0, num_pictures):
	arr = convert_img_to_array(i)    
	x_train.append(arr)
np.savez("category_other.npz", arr=x_train) # save all in one file

