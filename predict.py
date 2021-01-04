import pickle
from pathlib import Path

import numpy as np
from cv2 import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array

from _common import get_logger, IMAGE_DIMENSIONS, LABELBIN_OUTPUT, MODEL_OUTPUT

logger = get_logger('predict')


class Predict:
	def __init__(self):
		logger.info("loading network...")
		self.model = load_model(MODEL_OUTPUT)
		self.lb = pickle.loads(open(LABELBIN_OUTPUT, "rb").read())

	def run(self, image):
		logger.info(f'----- {image}')
		image = str(image)

		image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
		output = image.copy()

		# pre-process the image for classification
		image = cv2.resize(image, IMAGE_DIMENSIONS[:2])
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)

		# classify the input image
		logger.info("classifying image...")
		proba = self.model.predict(image)[0]
		y_preds = list(reversed(np.argsort(proba)))
		top = y_preds[0]
		for idx, top_x in enumerate(y_preds[:10]):
			logger.info(f'Prediction {idx}: {self.lb.classes_[top_x]} ({proba[top_x]*100.0:.2f}%)')
		#
		# # we'll mark our prediction as "correct" of the input image filename
		# # contains the predicted label text (obviously this makes the
		# # assumption that you have named your testing image files this way)
		# filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
		# correct = "correct" if filename.rfind(label) != -1 else "incorrect"
		#
		# # build the label and draw the label on the image
		# label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
		# output = imutils.resize(output, width=400)
		# cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
		# 	0.7, (0, 255, 0), 2)
		#
		# # show the output image
		# print("[INFO] {}".format(label))
		# cv2.imshow("Output", output)
		# cv2.waitKey(0)


if __name__ == '__main__':
	p = Predict()

	for path in Path(r'E:\pokemon_go_cache\images').rglob('*.png'):
		p.run(path)
