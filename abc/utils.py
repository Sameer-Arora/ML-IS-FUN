import numpy as np
from scipy import ndimage
from pathlib import Path

def load_image(filename):
	img = ndimage.imread(filename, flatten = True)
	return img

def flatten_image(img):
	return np.array(img).flatten()


def get_data(directory, two_d):

	data_dir = Path(directory)
	filename = data_dir / "data.txt" 

	X = []
	Y = []
	with open(filename) as fp:
		for instance in fp.readlines():
			instance_split = instance.split("\t")
			image_path, angle = instance_split[0], np.array(instance_split[1]).astype(float)
			image_name = data_dir / image_path
			img = load_image(image_name)
			if(two_d):
				img = img[np.newaxis]
				X.append(img)
			else:
				X.append(flatten_image(img))
			Y.append(angle)

	return np.array(X),np.array(Y)[:,np.newaxis]


