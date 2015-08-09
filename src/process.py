#!/usr/bin/env python

import re
from os import listdir
from os.path import isfile, join, basename
from HTMLParser import HTMLParser
import json
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np


regex = re.compile('[^a-zA-Z]')
words = {}
tags = {}

MAX_WORD_LENGTH = 10
BUILD_WORD_DICT = False

# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    def set_model(self, create):
		self.create = create
		if create:
			self.bow = {}
			self.bot = {}

    def handle_starttag(self, tag, attrs):
		tag = tag.lower()
		if self.create:
			tag_index = tag
			self.bot[tag_index] = self.bot[tag_index] + 1 if tag_index in self.bot else 1
		else:
			if not tag in tags: tags[tag] = len(tags.keys())
        	# print "Encountered a start tag:", tag

    def handle_endtag(self, tag):
		# print "Encountered an end tag :", tag
		pass
    def handle_data(self, data):
		# print "Encountered some data  :", data
		if not BUILD_WORD_DICT: return
		data = data.lower()
		ws = data.split(' ')
		for w in ws:
			c = regex.sub('', w)

			if len(c)==0 or len(c) > MAX_WORD_LENGTH: continue

			if self.create:
				word_index = c
				self.bow[word_index] = self.bow[word_index] + 1 if word_index in self.bow else 1
			else:
				if c not in words: words[c] = len(tags.keys())
    # get bag of words
    def get_bow(self):
		return self.bow
    # get bag of tags
    def get_bot(self):
		return self.bot

def dump_dict(filename, d):
	with open(filename, 'w') as fp:
		json.dump(d, fp)
	print 'dumped to', filename, len(d.keys()), 'keys wrote!'

def load_dict(filename):
	with open(filename, 'r') as fp:
		d = json.load(fp)
	return d


def list_files():
	print 'Listing files...'
	
	dirs = ['../data/0/', '../data/1/', '../data/2/', '../data/3/', '../data/4/']
	files = []
	for d in dirs: 
		files = files + [ join(d,f) for f in listdir(d) if isfile(join(d,f)) ]

	file_index = {}
	index = 0
	for f in files:
		file_index[basename(f)] = index
		index +=1

	print 'Total files:', len(files)

	return files, file_index

def build_dict(files):
	print 'Building dictionaries...'
	parser = MyHTMLParser()
	parser.set_model(False)
	index = 0
	for f in files:
		index += 1
		if index % 100 == 0: print index, 'file processed! words/tags:', len(words.keys()), len(tags.keys())
		with open(f) as myfile:
			parser.feed(myfile.read())
		#if index > 100: break

	if BUILD_WORD_DICT: dump_dict('../data/words.json', words)
	dump_dict('../data/tags.json', tags)

	print 'Built!'
	if BUILD_WORD_DICT: print 'Total words:', len(words.keys())
	print 'Total tags:', len(tags.keys())

# 
def doc2dict(filename, tag_dict):
	parser = MyHTMLParser()
	parser.set_model(True)
	with open(filename, 'r') as fp:
		parser.feed(fp.read())
	return parser.get_bot()

def convert_train_list():
	train_list = {}
	with open('../data/train.csv') as fp:
		next(fp)
		for l in fp: 
			items = l.split(',')
			train_list[items[0]] = int(items[1])
	dump_dict('../data/train_list.json', train_list)
	return train_list

def generate_train_index(file_index, train_list):
	train_index = []
	for filename in train_list:
		train_index.append(file_index[filename])
	np.save('../data/train_index', np.array(train_index))


def main():
	files, file_index = list_files()
	# np.save('../data/files', np.array(files))
	#dump_dict('../data/file_index.json', file_index)

	#pass

	# print file_index

	build_dict(files)

	# train_list = convert_train_list()
	# print train_list


	tags = load_dict('../data/tags.json')
	print 'total tags:', len(tags.keys())


	D = []

	index = 0
	for f in files:
		D.append(doc2dict(f, tags))
		index += 1
		if index % 100 == 0: print index, 'file(s) parsed!'
		# if index > 1000: break

	v = DictVectorizer(sparse=True)
	X = v.fit_transform(D)
	t = TfidfTransformer()
	Y = t.fit_transform(X)

	np.save('../data/tf', Y)

	train_list = load_dict('../data/train_list.json')
	print 'total train exmaples:', len(train_list.keys())

if __name__ == "__main__":
	main()

#X = np.load('../data/tf.npy')
