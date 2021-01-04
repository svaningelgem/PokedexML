import pickle
import re
from collections import namedtuple
from functools import reduce
from itertools import cycle

import matplotlib

matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
import numpy as np
import random
from cv2 import cv2
from _common import get_logger, root, DATASET_DIR, MODEL_OUTPUT, LABELBIN_OUTPUT, IMAGE_DIMENSIONS


PokemonInfo = namedtuple('PokemonInfo', 'id is_shiny form_id path')
IMAGE_FILE_TO_PROCESS_00 = re.compile(r'pokemon_icon_(?:pm|)(\d{3,4})_(\d+)(_shiny|)\.png')
IMAGE_FILE_TO_PROCESS_NORMAL = re.compile(r'pokemon_icon_(?:pm|)(\d{3,4})(?:_00)_(\d+)(_shiny|)\.png')


logger = get_logger('training')


class Trainer:
	PLOT_OUTPUT = root / 'training.png'

	def __init__(self, *, epochs=200, initial_learning_rate=0.001, batch_size=32):
		self.epochs = epochs
		self.init_lr = initial_learning_rate
		self.bs = batch_size

		self.data = []
		self.labels = []

	def _list_images(self):
		logger.info("loading images...")
		for p in DATASET_DIR.glob('pokemon_icon_*.png'):
			if p.stem.startswith('pokemon_icon_pm'):
				name = p.stem[15:]
			elif p.stem.startswith('pokemon_icon_'):
				name = p.stem[13:]
			else:
				raise ValueError(f'Euh? {p}')

			parts = name.split('_')
			# First part is always the id:
			pokemon_id = int(parts[0])
			if pokemon_id == 0:
				continue

			is_shiny = parts[-1] == 'shiny'

			yield PokemonInfo(id=pokemon_id, is_shiny=is_shiny, form_id=name, path=p)

	def run(self):
		# grab the image paths
		image_infos = sorted(list(self._list_images()))

		# and randomly shuffle them
		random.shuffle(image_infos)
		self._read_image_data(image_infos)

		lb, model, H = self._start_training()
		self._save(lb, model, H)

	def _read_image_data(self, image_infos):
		# loop over the input images
		for info in image_infos:
			# pre-process images and update data and label lists
			image = cv2.imread(str(info.path), cv2.IMREAD_UNCHANGED)
			image = cv2.resize(image, (IMAGE_DIMENSIONS[1], IMAGE_DIMENSIONS[0]))
			image = img_to_array(image)

			self.data.append(image)
			self.labels.append(info.form_id)

		# scale the raw pixel intensities to the range [0, 1]
		self.data = np.array(self.data, dtype="float") / 255.0
		self.labels = np.array(self.labels)
		logger.info("data matrix: {:.2f}MB".format(self.data.nbytes / (1024 * 1000.0)))

	def _start_training(self):
		# binarize the labels
		lb = LabelBinarizer()
		labels = lb.fit_transform(self.labels)

		# 80% for training and 20% for testing
		(trainX, testX, trainY, testY) = train_test_split(self.data, labels, test_size=0.2, random_state=42)

		# construct the image generator for data augmentation
		datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
			shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

		# initialize the model
		logger.info("compiling model...")
		model = SmallerVGGNet.build(width=IMAGE_DIMENSIONS[1], height=IMAGE_DIMENSIONS[0], depth=IMAGE_DIMENSIONS[2], classes=len(lb.classes_))
		opt = Adam(lr=self.init_lr, decay=self.init_lr / self.epochs)
		model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

		# train the network
		logger.info("training network...")
		return lb, model, model.fit(
			datagen.flow(trainX, trainY, batch_size=self.bs),
			validation_data=(testX, testY),
			steps_per_epoch=len(trainX) // self.bs,
			epochs=self.epochs,
			verbose=1
		)

	def _save(self, lb, model, H):
		# save the results to disk
		logger.info("serializing network...")
		model.save(MODEL_OUTPUT)

		logger.info("serializing label binarizer...")
		with open(LABELBIN_OUTPUT, 'wb') as f:
			f.write(pickle.dumps(lb))

		logger.info("Plotting figure:")
		self._plot_data(H.history, ['acc', 'accuracy'])

	def _plot_data(self, data, draw_on_secondary_axis=None):
		colors = cycle(matplotlib.rcParams['axes.prop_cycle'].by_key().get('color', ['k']))

		draw_on_secondary_axis = draw_on_secondary_axis or []

		data = {k: v for k, v in data.items() if any(v)}
		lines = []

		plt.style.use("ggplot")
		plt.figure()
		N = self.epochs

		fig, ax1 = plt.subplots()
		ax2 = ax1.twinx()
		for key in draw_on_secondary_axis:
			if key not in data:
				continue

			lines.append(
				ax2.plot(np.arange(0, N), data[key], label=key, color=next(colors))
			)

		for key in data:
			if key in draw_on_secondary_axis:
				continue

			lines.append(
				ax1.plot(np.arange(0, N), data[key], label=key, color=next(colors))
			)

		plt.title("Training Loss and Accuracy")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")

		lines = reduce(lambda x, y: x + y, lines[1:], lines[0])
		labels = [l.get_label() for l in lines]
		ax1.legend(lines, labels, loc="upper left")
		plt.savefig(self.PLOT_OUTPUT)


if __name__ == '__main__':
	Trainer().run()
