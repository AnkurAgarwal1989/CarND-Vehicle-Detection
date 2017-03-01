import numpy as np
import cv2
import time
from sklearn.svm import LinearSVC



def train_LinearSVC(X_train, y_train, X_val, y_val):
	# Use a linear SVC 
	svc = LinearSVC()
	# Check the training time for the SVC
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_val, y_val), 4))
	# Check the prediction time for a single sample
	t=time.time()
	n_predict = 10
	print('SVC predicts: ', svc.predict(X_val[0:n_predict]))
	print('For these',n_predict, 'labels: ', y_val[0:n_predict])
	t2 = time.time()
	print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
	return svc
