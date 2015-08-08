#!/usr/bin/env python

import numpy as np
from process import load_dict, generate_train_index
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score


def train(X, y):
	pass

def predict(model, X_test):
	pass

def cv(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
	
	model = train(X_train, y_train)

	y_pred = predict(model, X_test)

	print 'roc_auc_score =', roc_auc_score(y_test, y_pred)


if __name__ == '__main__':
	file_index = load_dict('../data/file_index.json')
	print 'total files =', len(file_index.keys())

	train_list = load_dict('../data/train_list.json')
	print 'train files =', len(train_list.keys())

	train_index = np.load('../data/train_index.npy')
	print 'train_index.shape =', train_index.shape

	tf = np.load('../data/tf.npy').tolist()
	print 'tf.shape=', tf.shape

	X_train = tf[train_index,:]
	print 'X_train.shape', X_train.shape

	y_train = np.array([train_list[f] for f in train_list])

	print 'y_train.shape', y_train.shape

	cv(X_train, y_train)