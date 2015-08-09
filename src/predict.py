#!/usr/bin/env python

import numpy as np
import pandas as pd
from os.path import basename
from scipy import sparse

from process import load_dict, generate_train_index
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier


def train(X, y):	
	print 'training...'
	print 'X.shape =', X.shape, 'y.shape =', y.shape
	clf = KNeighborsClassifier(n_neighbors=50, weights='distance', algorithm='brute', metric='cosine')
	clf.fit(X, y)
	# score = clf.score(X,y)
	# print 'score =', score
	return clf


def predict(model, X_test):
	print 'preding...'
	print 'X_test.shape =', X_test.shape
	size = X_test.shape[0]
	split = 100
	steps = size/split

	print 'steps=', steps

	preds = []
	
	for t in range(steps+1):
		start = t * split
		if start >= size: break
		end =  min(start+split, size)
		print 'predciting ', start, '->', end
		pred = model.predict_proba(X_test[start:end,:])
		preds = preds + pred[:,1].tolist()

	preds = np.array(preds)

	print 'preds.shape = ', preds.shape
	return preds
	

def cv(X, y):
	print 'cv...'
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
	print 'y_train.shape = ', y_train.shape
	print 'y_test.shape = ', y_test.shape
	
	model = train(X_train, y_train)

	y_pred = predict(model, X_test)

	print 'y_pred.shape =',y_pred.shape

	print 'roc_auc_score =', roc_auc_score(y_test, y_pred)

def write_submission(y_pred, filenames_test):

	sub = pd.read_csv('../data/sampleSubmission.csv')

	sub = set(sub.to_dict()['file'].values())	

	index = 0
	with open('submission.csv', 'w') as fp:
		fp.write('file,sponsored\n')
		for f in filenames_test:
			f = basename(f)
			if f in sub:
				prob = y_pred[index]
				if prob < 1e-6: prob = 0.0
				if prob > 0.9999: prob = 1.0
				fp.write(f + ',' + str(prob) + '\n')
			index += 1
			if index >= len(y_pred): break


if __name__ == '__main__':
	files = np.load('../data/files.npy')
	print 'filenames = ', len(files)

	file_index = load_dict('../data/file_index.json')
	print 'total files =', len(file_index.keys())

	train_list = load_dict('../data/train_list.json')
	print 'train files =', len(train_list.keys())

	train_index = np.load('../data/train_index.npy')
	print 'train_index.shape =', train_index.shape

	tf = np.load('../data/tf.npy').tolist()
	tf = normalize(tf)
	print 'tf.shape=', tf.shape

	mask = np.zeros((tf.shape[0]), dtype=bool)
	mask[train_index] = True
	print 'mask.shape = ',mask.shape

	X_train = tf[mask,:]
	print 'X_train.shape', X_train.shape
	X_test = tf[~mask,:]	
	print 'X_test.shape', X_test.shape

	filenames_train = files[mask]
	filenames_test = files[~mask]
	print 'filenames_train.shape', filenames_train.shape
	print 'filenames_test.shape', filenames_test.shape

	y_train = [train_list[basename(f)] for f in filenames_train]
	y_train = np.array(y_train)

	print 'y_train.shape', y_train.shape
	print 'sponsed = ', sum(y_train==1) * 100.0 / y_train.shape[0], '%'


	cv(X_train, y_train)

	model = train(X_train, y_train)
	y_pred = predict(model, X_test)

	write_submission(y_pred, filenames_test)
