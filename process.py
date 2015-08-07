#!/usr/bin/env python

import re
from HTMLParser import HTMLParser

regex = re.compile('[^a-zA-Z]')
words = {}
tags = {}

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
	data = data.lower()
	ws = data.split(' ')
	for w in ws:
		c = regex.sub('', w)
		
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

# instantiate the parser and fed it some HTML
parser = MyHTMLParser()
parser.set_model(False)
parser.feed('<html><head><title>Test</title></head>'
            '<body><h1>Parse me!</h1></body></html>')
parser.feed('<html><head><title>My Title</title></head>'
            '<body><h1>God help me!</h1></body></html>')
print words
print tags

parser.set_model(True)
parser.feed('<html><head><title>Test</title></head>'
            '<body><h1>Parse me! PARSE Me!</h1></body></html>')


print "bag of words: ", parser.get_bow()
print "bag of tags: ", parser.get_bot()
