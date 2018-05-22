
# import tensorflow as tf

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model
import numpy as np

import pdb

def get_feature_4096 (model, img_path):
	'''
	# vgg16 = VGG16(weights= 'imagenet', include_top= False) # output: 7*7*512
	# vgg16 = VGG16(weights= 'imagenet', include_top= True) # output: 1000*1
	'''
	model_4096 = Model(inputs=model.input, outputs=model.get_layer('fc2').output)

	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	return model_4096.predict(x)

def get_classinfo(model, img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	return decode_predictions(model.predict(x), top=3)[0]	

#############################################################
#############################################################
#############################################################

vgg16 = VGG16(weights= 'imagenet')

# test
img_path = 'dog.png'
feature_4096 = get_feature_4096(vgg16, img_path)
print(feature_4096.shape)
print(get_classinfo(vgg16, img_path))

# test
img_paths = ['dog.png', 'dog2.png', 'laska.png', 'poodle.png', 'quail227.jpeg', 'state1.png', 'state2.png']

for img_path in img_paths:
	print(img_path)
	print(get_feature_4096(vgg16, img_path).shape)
	print(get_classinfo(vgg16, img_path))

# pdb.set_trace()
